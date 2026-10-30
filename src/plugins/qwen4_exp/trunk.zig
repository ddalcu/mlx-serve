//! The qwen4_exp trunk module: the loaded weights (layers with their hyper-connection and PLE blocks, the final
//! mixer, the MTP head), the n-gram state, and the per-model caches its forward keeps (QSA block constants and pooled
//! rope, the compiled closures, quant params, decode requant copies, angle rows, the persistent batched GDN group).
//! Its methods are the family's forward (`forward.zig`, `mtp.zig`, `ple.zig`, `hc.zig`, `qsa/`) and the trunk ops
//! (`trunk/`).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const model = @import("../../sdk/config/model.zig");
const qwen4_mod = @import("qwen4_exp.zig");
const mtp = @import("mtp.zig");
const distributed_mod = @import("../../engines/mlx/cluster/distributed.zig");
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const MoeLayerWeights = @import("../../engines/mlx/layer_weights.zig").MoeLayerWeights;
const PersistentSsmGroup = @import("../../engines/mlx/ssm/group.zig").PersistentSsmGroup;
const VerifyLaps = @import("../../engines/mlx/batched/row_axis_verify.zig").VerifyLaps;
const HcWeights = @import("weights.zig").HcWeights;
const QsaPooledRope = @import("qsa/pooled.zig").QsaPooledRope;
const QsaBlockConsts = @import("qsa/pooled.zig").QsaBlockConsts;
const BitsCache = @import("../../sdk/quant/params.zig").BitsCache;
const AttnDqCopy = @import("../../sdk/blocks/attention/decode_attn_quant.zig").AttnDqCopy;
const QkAngleSlot = @import("../../sdk/blocks/rope/qk_norm_rope.zig").QkAngleSlot;
const DtypeCastCache = @import("../../mlx/dtype.zig").DtypeCastCache;
const lane_qmm = @import("../../sdk/blocks/qmm/lane_qmm.zig");

pub const Trunk = struct {
    /// Borrowed: its owner (the plugin's model, or the engine's legacy shell) frees it after the trunk.
    config: model.ModelConfig,
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,

    emb_w: mlx.mlx_array,
    emb_s: mlx.mlx_array,
    emb_b: mlx.mlx_array,
    lm_head_w: mlx.mlx_array,
    lm_head_s: mlx.mlx_array,
    lm_head_b: mlx.mlx_array,
    /// A ternary 2-bit pack: its projections read the 2-bit kernels (`trunk/linear.zig` `qmatmul`).
    ternary_2bit: bool = false,
    moe_layers: []MoeLayerWeights,
    /// Arrays the loader built (transposed bf16, folded hyper-connection weights): freed with the trunk.
    moe_owned_bf16: []mlx.mlx_array = &.{},
    /// The final hyper-connection read, in place of `model.norm`.
    qwen4_mixer: HcWeights,
    /// The n-gram hash and the mmapped table (plus its GPU arm).
    qwen4: ?*qwen4_mod.Qwen4State = null,
    /// The in-checkpoint MTP layer; null without one (or on a pipeline rank).
    qwen4_mtp: ?mtp.Qwen4Mtp = null,
    /// Whose per-request state the head holds (`qwen4MtpActivate`).
    qwen4_mtp_owner: ?*mtp.Qwen4MtpState = null,

    /// `[hidden]` ones for the weight-less RMS norms; 1.0 (bf16) for the dispatch probe; `rms_norm_eps` as f32.
    ones_hidden: mlx.mlx_array,
    one: mlx.mlx_array,
    rms_eps_arr: mlx.mlx_array,
    /// YaRN: the rope denominators, the mscale vector, the host inverse frequencies; null on an unscaled rope.
    rope_freqs_yarn: ?mlx.mlx_array = null,
    yarn_mscale: ?mlx.mlx_array = null,
    yarn_inv_freq: ?[]const f64 = null,
    yarn_mscale_cast: DtypeCastCache = .{},

    /// Run the residual streams in f32 (the fixture oracle's tighter bar).
    qwen4_stream_f32: bool = false,
    qsa_consts: QsaBlockConsts = .{},
    qsa_pooled_rope: QsaPooledRope = .{},
    /// Bumped by every forward entry that can reach QSA; keys the per-forward QSA scratch.
    fwd_gen: u64 = 0,
    /// Test meter: how many times `qsaScoreBank` built or extended its transposed operand.
    qsa_score_bank_builds: usize = 0,
    /// Pipeline rank 0: the worker holding the later layers; tensor rank 0: the worker holding the other expert slices.
    dist_peer: ?*distributed_mod.Peer = null,
    dist_tp: ?*distributed_mod.Peer = null,

    compiled_moe_routing: ?mlx.mlx_closure = null,
    compiled_gdn_gate: ?mlx.mlx_closure = null,
    /// The hyper-connection elementwise tails, one kernel each: silu(down), mean(sigma(up)*normed), 2sigma(inject),
    /// stream + out*inj.
    compiled_hc_silu: ?mlx.mlx_closure = null,
    compiled_hc_mix: ?mlx.mlx_closure = null,
    compiled_hc_inj: ?mlx.mlx_closure = null,
    compiled_hc_write: ?mlx.mlx_closure = null,
    gdn_ones_w: ?mlx.mlx_array = null,
    gdn_q_scale: ?mlx.mlx_array = null,
    gdn_k_scale: ?mlx.mlx_array = null,
    gdn_eps: ?mlx.mlx_array = null,
    /// A verify forward in flight: the recurrent layers keep each row's state for the partial-accept rollback.
    spec_capture_ssm: bool = false,
    bits_cache: BitsCache = .{},
    /// Decode-only int8/nvfp4 copies of dense attention projections, by the weight's pointer (null = ineligible).
    attn_dq: std.AutoHashMapUnmanaged(usize, ?AttnDqCopy) = .empty,
    qk_angle_cache: [2]QkAngleSlot = .{ .{}, .{} },
    /// The round-cost tracer's per-forward arm bits.
    cost_trace_active: bool = false,
    cost_attention: u8 = 0,
    cost_moe: u8 = 0,
    verify_laps: VerifyLaps = .{},
    /// The joined `[1, sum(rows), V]` logits of the last grouped verify whose head ran once (released by the next).
    verify_joined_logits: mlx.mlx_array = .{ .ctx = null },
    /// Layer stride for the decode async-eval ladder (0 = off) and the batched tick's.
    decode_async_ladder: u32 = 0,
    decode_async_ladder_qwen4: u32 = @import("forward.zig").QWEN4_DECODE_ASYNC_LADDER_DEFAULT,
    ssm_group: PersistentSsmGroup = .{},
    /// Test seams (no env): `false` runs the per-tick merge/split and the per-slot KV append.
    persistent_group_state_override: ?bool = null,
    batched_kv_append_override: ?bool = null,
    /// The MTP cost profile's trunk fingerprints (homogeneous affine-4/gs-64; the oQ4e mixed layout).
    mtp_uniform_affine_trunk: bool = false,
    mtp_oqe_affine_trunk: bool = false,

    pub var qwen4_trace: ?*mtp.Qwen4Trace = null;
    pub var qsa_engaged_logged: bool = false;

    // attention.zig
    pub const qwen4AttnProjected = @import("attention.zig").qwen4AttnProjected;
    pub const qwen4AttnWith = @import("attention.zig").qwen4AttnWith;
    // forward.zig
    pub const forwardQwen4Stream = @import("forward.zig").forwardQwen4Stream;
    pub const forwardQwen4VerifyRows = @import("forward.zig").forwardQwen4VerifyRows;
    pub const forwardQwen4With = @import("forward.zig").forwardQwen4With;
    pub const QWEN4_DECODE_ASYNC_LADDER_DEFAULT = @import("forward.zig").QWEN4_DECODE_ASYNC_LADDER_DEFAULT;
    pub const qwen4VerifyRowsEligible = @import("forward.zig").qwen4VerifyRowsEligible;
    // hc.zig
    pub const addVerifyPle = @import("hc.zig").addVerifyPle;
    pub const compileQwen4Hc = @import("hc.zig").compileQwen4Hc;
    pub const hcFlush = @import("hc.zig").hcFlush;
    pub const hcGroupNorm = @import("hc.zig").hcGroupNorm;
    pub const hcInjCallback = @import("hc.zig").hcInjCallback;
    pub const hcMixCallback = @import("hc.zig").hcMixCallback;
    pub const hcRead = @import("hc.zig").hcRead;
    pub const HcRead = @import("hc.zig").HcRead;
    pub const hcReadFusedFor = @import("hc.zig").hcReadFusedFor;
    pub const hcReadNormed = @import("hc.zig").hcReadNormed;
    pub const hcReadPending = @import("hc.zig").hcReadPending;
    pub const hcReadStandin = @import("hc.zig").hcReadStandin;
    pub const hcReadVerifyRows = @import("hc.zig").hcReadVerifyRows;
    pub const hcSiluCallback = @import("hc.zig").hcSiluCallback;
    pub const hcWrite = @import("hc.zig").hcWrite;
    pub const hcWriteCallback = @import("hc.zig").hcWriteCallback;
    pub const hcWriteOrDefer = @import("hc.zig").hcWriteOrDefer;
    // mtp.zig
    pub const HeadPlace = @import("mtp.zig").HeadPlace;
    pub const loadQwen4Mtp = @import("mtp.zig").loadQwen4Mtp;
    pub const mtpHeadForceBatched = @import("mtp.zig").mtpHeadForceBatched;
    pub const mtpMoeRows = @import("mtp.zig").mtpMoeRows;
    pub const mtpPadSeq = @import("mtp.zig").mtpPadSeq;
    pub const qwen4BuildDraftRerank = @import("mtp.zig").qwen4BuildDraftRerank;
    pub const qwen4DraftRerankReady = @import("mtp.zig").qwen4DraftRerankReady;
    pub const qwen4DraftSelect = @import("mtp.zig").qwen4DraftSelect;
    pub const qwen4DraftSelectBatched = @import("mtp.zig").qwen4DraftSelectBatched;
    pub const qwen4DraftShortlist = @import("mtp.zig").qwen4DraftShortlist;
    pub const qwen4DraftShortlistsBatched = @import("mtp.zig").qwen4DraftShortlistsBatched;
    pub const qwen4MtpActivate = @import("mtp.zig").qwen4MtpActivate;
    pub const qwen4MtpAdopt = @import("mtp.zig").qwen4MtpAdopt;
    pub const qwen4MtpAdvance = @import("mtp.zig").qwen4MtpAdvance;
    pub const qwen4MtpApplyKvQuant = @import("mtp.zig").qwen4MtpApplyKvQuant;
    pub const Qwen4MtpBatchRow = @import("mtp.zig").Qwen4MtpBatchRow;
    pub const qwen4MtpForward = @import("mtp.zig").qwen4MtpForward;
    pub const qwen4MtpForwardBatched = @import("mtp.zig").qwen4MtpForwardBatched;
    pub const qwen4MtpForwardLive = @import("mtp.zig").qwen4MtpForwardLive;
    pub const qwen4MtpForwardOn = @import("mtp.zig").qwen4MtpForwardOn;
    pub const qwen4MtpForwardPlaced = @import("mtp.zig").qwen4MtpForwardPlaced;
    pub const qwen4MtpForwardPlacedOn = @import("mtp.zig").qwen4MtpForwardPlacedOn;
    pub const Qwen4MtpLive = @import("mtp.zig").Qwen4MtpLive;
    pub const qwen4MtpLiveModule = @import("mtp.zig").qwen4MtpLiveModule;
    pub const qwen4MtpLiveState = @import("mtp.zig").qwen4MtpLiveState;
    pub const qwen4MtpMarkQsaLeftover = @import("mtp.zig").qwen4MtpMarkQsaLeftover;
    pub const qwen4MtpMarks = @import("mtp.zig").qwen4MtpMarks;
    pub const Qwen4MtpOut = @import("mtp.zig").Qwen4MtpOut;
    pub const Qwen4MtpOutBuf = @import("mtp.zig").Qwen4MtpOutBuf;
    pub const Qwen4MtpProject = @import("mtp.zig").Qwen4MtpProject;
    pub const qwen4MtpRelease = @import("mtp.zig").qwen4MtpRelease;
    pub const qwen4MtpReset = @import("mtp.zig").qwen4MtpReset;
    pub const qwen4MtpResetOwned = @import("mtp.zig").qwen4MtpResetOwned;
    pub const Qwen4MtpState = @import("mtp.zig").Qwen4MtpState;
    pub const qwen4MtpStateNew = @import("mtp.zig").qwen4MtpStateNew;
    pub const qwen4MtpSwapState = @import("mtp.zig").qwen4MtpSwapState;
    pub const qwen4MtpTruncate = @import("mtp.zig").qwen4MtpTruncate;
    pub const qwen4MtpTruncateLive = @import("mtp.zig").qwen4MtpTruncateLive;
    pub const qwen4MtpTruncateOn = @import("mtp.zig").qwen4MtpTruncateOn;
    pub const qwen4MtpTruncateToMarkLive = @import("mtp.zig").qwen4MtpTruncateToMarkLive;
    pub const Qwen4Trace = @import("mtp.zig").Qwen4Trace;
    // ple.zig
    pub const advancePlePrev = @import("ple.zig").advancePlePrev;
    pub const pleAdvanceSerial = @import("ple.zig").pleAdvanceSerial;
    pub const pleClaimSpecCapture = @import("ple.zig").pleClaimSpecCapture;
    pub const pleDeferrable = @import("ple.zig").pleDeferrable;
    pub const pleEmbedding = @import("ple.zig").pleEmbedding;
    pub const pleEmbeddingGpu = @import("ple.zig").pleEmbeddingGpu;
    pub const pleEvalIds = @import("ple.zig").pleEvalIds;
    pub const pleFillFromIds = @import("ple.zig").pleFillFromIds;
    pub const pleForward = @import("ple.zig").pleForward;
    pub const pleGatherBf16 = @import("ple.zig").pleGatherBf16;
    pub const pleHostIds = @import("ple.zig").pleHostIds;
    pub const plePrepareIds = @import("ple.zig").plePrepareIds;
    pub const plePrevOf = @import("ple.zig").plePrevOf;
    pub const pleSettleFromArray = @import("ple.zig").pleSettleFromArray;
    pub const releasePlePending = @import("ple.zig").releasePlePending;
    // qsa/decode_gather.zig
    pub const qsaBatchedAttn = @import("qsa/decode_gather.zig").qsaBatchedAttn;
    // qsa/indexer.zig
    pub const qsaAppendKeys = @import("qsa/indexer.zig").qsaAppendKeys;
    pub const qsaAppendPooled = @import("qsa/indexer.zig").qsaAppendPooled;
    pub const qsaBlockVisibility = @import("qsa/indexer.zig").qsaBlockVisibility;
    pub const qsaChunkSelect = @import("qsa/indexer.zig").qsaChunkSelect;
    pub const qsaInstallRing = @import("qsa/indexer.zig").qsaInstallRing;
    pub const qsaMask = @import("qsa/indexer.zig").qsaMask;
    pub const qsaMaskBatched = @import("qsa/indexer.zig").qsaMaskBatched;
    pub const qsaMaskFromQk = @import("qsa/indexer.zig").qsaMaskFromQk;
    pub const qsaPooledCosSin = @import("qsa/indexer.zig").qsaPooledCosSin;
    pub const qsaPooledKeys = @import("qsa/indexer.zig").qsaPooledKeys;
    pub const qsaPooledKeysComposed = @import("qsa/indexer.zig").qsaPooledKeysComposed;
    pub const qsaPooledRopeComposed = @import("qsa/indexer.zig").qsaPooledRopeComposed;
    pub const qsaPoolNorm = @import("qsa/indexer.zig").qsaPoolNorm;
    pub const qsaScoreBank = @import("qsa/indexer.zig").qsaScoreBank;
    pub const qsaScoreCols = @import("qsa/indexer.zig").qsaScoreCols;
    pub const qsaScoreOperand = @import("qsa/indexer.zig").qsaScoreOperand;
    pub const qsaSelectBlocks = @import("qsa/indexer.zig").qsaSelectBlocks;
    pub const qsaSelectMaskOps = @import("qsa/indexer.zig").qsaSelectMaskOps;
    pub const qsaSlotEntry = @import("qsa/indexer.zig").qsaSlotEntry;
    // qsa/pooled.zig
    pub const markQsaPooledRopeStale = @import("qsa/pooled.zig").markQsaPooledRopeStale;
    pub const resetQsaPooledRope = @import("qsa/pooled.zig").resetQsaPooledRope;
    // trunk/linear.zig
    pub const qmatmul = @import("trunk/linear.zig").qmatmul;
    pub const lmHeadProject = @import("trunk/linear.zig").lmHeadProject;
    pub const quantParamsFor = @import("trunk/linear.zig").quantParamsFor;
    pub const quantParamsHinted = @import("trunk/linear.zig").quantParamsHinted;
    pub const mtpProjectionRows = @import("trunk/linear.zig").mtpProjectionRows;
    pub const verifyFeatureEnabled = @import("trunk/linear.zig").verifyFeatureEnabled;
    pub const rowGroupServes = @import("trunk/linear.zig").rowGroupServes;
    pub const attnProj = @import("trunk/linear.zig").attnProj;
    pub const attnDqFor = @import("trunk/linear.zig").attnDqFor;
    pub const rmsNorm = @import("trunk/linear.zig").rmsNorm;
    pub const silu = @import("trunk/linear.zig").silu;
    pub const computeGeglu = @import("trunk/linear.zig").computeGeglu;
    pub const denseMLP = @import("trunk/linear.zig").denseMLP;
    pub const embedding = @import("trunk/linear.zig").embedding;
    pub const applyVisionEmbeddingsWith = @import("trunk/linear.zig").applyVisionEmbeddingsWith;
    pub const lmHeadForeign = @import("trunk/linear.zig").lmHeadForeign;
    // trunk/attn.zig
    pub const gatedFullAttnWith = @import("trunk/attn.zig").gatedFullAttnWith;
    pub const gatedFullAttnProjected = @import("trunk/attn.zig").gatedFullAttnProjected;
    pub const gatedAttnTail = @import("trunk/attn.zig").gatedAttnTail;
    pub const qkAngleRowsFor = @import("trunk/attn.zig").qkAngleRowsFor;
    pub const qkAngleFor = @import("trunk/attn.zig").qkAngleFor;
    pub const ropeCosSinFromFreqs = @import("trunk/attn.zig").ropeCosSinFromFreqs;
    pub const ropeInvFreq = @import("trunk/attn.zig").ropeInvFreq;
    pub const ropeApplyCosSin = @import("trunk/attn.zig").ropeApplyCosSin;
    pub const ropeAtPositions = @import("trunk/attn.zig").ropeAtPositions;
    pub const applyMrope = @import("trunk/attn.zig").applyMrope;
    pub const beginMropeChunk = @import("trunk/attn.zig").beginMropeChunk;
    pub const mropeCosSinAt = @import("trunk/attn.zig").mropeCosSinAt;
    pub const yarnActive = @import("trunk/attn.zig").yarnActive;
    pub const yarnScaleRotated = @import("trunk/attn.zig").yarnScaleRotated;
    pub const yarnScaleQK = @import("trunk/attn.zig").yarnScaleQK;
    pub const stackedBatchedAttn = @import("trunk/attn.zig").stackedBatchedAttn;
    pub const buildBatchedDecodeMask = @import("trunk/attn.zig").buildBatchedDecodeMask;
    pub const padAndStackBatchedKV = @import("trunk/attn.zig").padAndStackBatchedKV;
    pub const perSlotBatchedAttn = @import("trunk/attn.zig").perSlotBatchedAttn;
    // trunk/gdn.zig
    pub const gatedDeltaNet = @import("trunk/gdn.zig").gatedDeltaNet;
    pub const gatedDeltaNetProjected = @import("trunk/gdn.zig").gatedDeltaNetProjected;
    pub const gdnCheckFused = @import("trunk/gdn.zig").gdnCheckFused;
    pub const gdnProjGpu = @import("trunk/gdn.zig").gdnProjGpu;
    pub const gdnProjArrays = @import("trunk/gdn.zig").gdnProjArrays;
    pub const conv1dWithCache = @import("trunk/gdn.zig").conv1dWithCache;
    pub const compileGdnGate = @import("trunk/gdn.zig").compileGdnGate;
    pub const computeGdnGate = @import("trunk/gdn.zig").computeGdnGate;
    // trunk/moe.zig
    pub const moeRouterLogits = @import("trunk/moe.zig").moeRouterLogits;
    pub const moeSharedDown = @import("trunk/moe.zig").moeSharedDown;
    pub const moeSharedProject = @import("trunk/moe.zig").moeSharedProject;
    pub const moeSharedGateTail = @import("trunk/moe.zig").moeSharedGateTail;
    pub const moeAddGatedShared = @import("trunk/moe.zig").moeAddGatedShared;
    pub const moeAddSharedDown = @import("trunk/moe.zig").moeAddSharedDown;
    pub const moeAddSharedGateRows = @import("trunk/moe.zig").moeAddSharedGateRows;
    pub const moeSharedDownRows = @import("trunk/moe.zig").moeSharedDownRows;
    pub const moeVerifyRowsSolo = @import("trunk/moe.zig").moeVerifyRowsSolo;
    pub const moeVerifyRows = @import("trunk/moe.zig").moeVerifyRows;
    pub const moeMLP = @import("trunk/moe.zig").moeMLP;
    pub const moeTensorParallel = @import("trunk/moe.zig").moeTensorParallel;
    pub const moeExpertsRouted = @import("trunk/moe.zig").moeExpertsRouted;
    pub const moeMLP2WithRouter = @import("trunk/moe.zig").moeMLP2WithRouter;
    pub const moeDecodeGatherQmv = @import("trunk/moe.zig").moeDecodeGatherQmv;
    pub const moeCudaDecode = @import("trunk/moe.zig").moeCudaDecode;
    pub const moeDecodeGatherQmvRows = @import("trunk/moe.zig").moeDecodeGatherQmvRows;
    pub const moeAffine4Decode = @import("trunk/moe.zig").moeAffine4Decode;
    pub const moeFp4Decode = @import("trunk/moe.zig").moeFp4Decode;
    pub const moeMLP2 = @import("trunk/moe.zig").moeMLP2;
    pub const useGatherQmvDecode = @import("trunk/moe.zig").useGatherQmvDecode;
    pub const scaleExperts = @import("trunk/moe.zig").scaleExperts;
    pub const computeMoeRouting = @import("trunk/moe.zig").computeMoeRouting;
    pub const moeRoutingUncompiled = @import("trunk/moe.zig").moeRoutingUncompiled;
    pub const compileMoeRouting = @import("trunk/moe.zig").compileMoeRouting;
    // trunk/verify.zig
    pub const verifyRowsJoined = @import("trunk/verify.zig").verifyRowsJoined;
    pub const attnVerifyRows = @import("trunk/verify.zig").attnVerifyRows;
    pub const gdnVerifyRows = @import("trunk/verify.zig").gdnVerifyRows;
    pub const forwardRowAxisVerify = @import("trunk/verify.zig").forwardRowAxisVerify;
    pub const takeJoinedVerifyLogits = @import("trunk/verify.zig").takeJoinedVerifyLogits;
    // trunk/batched.zig
    pub const forwardMoeBatchedDecode = @import("trunk/batched.zig").forwardMoeBatchedDecode;
    pub const forwardMoeBatchedVerify = @import("trunk/batched.zig").forwardMoeBatchedVerify;
    pub const forwardMoeBatchedRows = @import("trunk/batched.zig").forwardMoeBatchedRows;
    pub const batchedGdnReady = @import("trunk/batched.zig").batchedGdnReady;
    pub const ssmGroupRelease = @import("trunk/batched.zig").ssmGroupRelease;
    pub const ssmGroupDrop = @import("trunk/batched.zig").ssmGroupDrop;
    pub const flushDeferredPle = @import("ple.zig").flushDeferredPle;
    pub const flushDeferredPleGroup = @import("ple.zig").flushDeferredPleGroup;
    pub const settleDeferredPle = @import("ple.zig").settleDeferredPle;
    pub const discardDeferredPle = @import("ple.zig").discardDeferredPle;
    pub const mergeSsmAcrossSlots = @import("trunk/batched.zig").mergeSsmAcrossSlots;
    pub const splitSsmToSlots = @import("trunk/batched.zig").splitSsmToSlots;
    // trunk/warmup.zig
    pub const warmupSpecVerify = @import("trunk/warmup.zig").warmupSpecVerify;

    /// The serial forward: a slot that sat in a batched group gets its own entries back first.
    pub fn forwardWith(self: *Trunk, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
        if (ctx.batch_slots == null) try self.ssmGroupRelease(ctx);
        return self.forwardQwen4With(ctx, token_ids);
    }

    /// `forwardWith` publishing the pre-mixer stream's last row (`out_hidden`).
    pub fn forwardWithCapture(self: *Trunk, ctx: *ForwardCtx, token_ids: mlx.mlx_array, out_hidden: *mlx.mlx_array) !mlx.mlx_array {
        const saved = ctx.capture_hidden;
        ctx.capture_hidden = out_hidden;
        defer ctx.capture_hidden = saved;
        return self.forwardWith(ctx, token_ids);
    }

    /// `forwardWith` publishing the stream's last row and every row.
    pub fn forwardWithCaptureAll(self: *Trunk, ctx: *ForwardCtx, token_ids: mlx.mlx_array, out_hidden: *mlx.mlx_array, out_hidden_all: *mlx.mlx_array) !mlx.mlx_array {
        const saved = ctx.capture_hidden;
        const saved_all = ctx.capture_hidden_all;
        ctx.capture_hidden = out_hidden;
        ctx.capture_hidden_all = out_hidden_all;
        defer {
            ctx.capture_hidden = saved;
            ctx.capture_hidden_all = saved_all;
        }
        return self.forwardWith(ctx, token_ids);
    }

    /// The trunk over `weights` with `config` (both borrowed), the closures not compiled (`prepare`).
    pub fn init(io: std.Io, a: std.mem.Allocator, config: model.ModelConfig, weights: *model.Weights) !Trunk {
        return @import("trunk_load.zig").load(io, a, &config, weights, .{});
    }

    pub fn actDtype(self: *const Trunk) mlx.mlx_dtype {
        return self.config.actDtype();
    }

    pub fn ssmStateDtype(self: *const Trunk) mlx.mlx_dtype {
        return self.config.ssmStateDtype();
    }

    pub fn useCurrentThreadStream(self: *Trunk) void {
        self.s = mlx.gpuStream();
    }

    /// The compiled routing, GDN gate and hyper-connection closures the forward reads.
    pub fn prepare(self: *Trunk) void {
        if (mlx.noGpuBackend()) return;
        self.compileMoeRouting();
        if (self.config.linear_num_key_heads > 0) self.compileGdnGate();
        self.compileQwen4Hc();
    }

    pub fn deinit(self: *Trunk) void {
        if (self.verify_joined_logits.ctx != null) _ = mlx.mlx_array_free(self.verify_joined_logits);
        lane_qmm.release(@intFromPtr(self));
        inline for (.{ self.compiled_moe_routing, self.compiled_gdn_gate, self.compiled_hc_silu, self.compiled_hc_mix, self.compiled_hc_inj, self.compiled_hc_write }) |c| {
            if (c) |cl| _ = mlx.mlx_closure_free(cl);
        }
        inline for (.{ self.gdn_ones_w, self.gdn_q_scale, self.gdn_k_scale, self.gdn_eps, self.rope_freqs_yarn, self.yarn_mscale }) |a| {
            if (a) |arr| _ = mlx.mlx_array_free(arr);
        }
        if (self.yarn_inv_freq) |f| self.allocator.free(f);
        _ = mlx.mlx_array_free(self.ones_hidden);
        _ = mlx.mlx_array_free(self.one);
        _ = mlx.mlx_array_free(self.rms_eps_arr);
        self.qsa_consts.deinit();
        self.qsa_pooled_rope.deinit();
        self.yarn_mscale_cast.deinit();
        {
            var it = self.attn_dq.valueIterator();
            while (it.next()) |maybe| if (maybe.*) |copy| {
                _ = mlx.mlx_array_free(copy.w);
                _ = mlx.mlx_array_free(copy.s);
                if (copy.b.ctx != null) _ = mlx.mlx_array_free(copy.b);
            };
            self.attn_dq.deinit(self.allocator);
        }
        for (&self.qk_angle_cache) |*slot| {
            if (slot.arr.ctx != null) _ = mlx.mlx_array_free(slot.arr);
        }
        self.ssm_group.deinit();
        self.allocator.free(self.moe_layers);
        if (self.qwen4_mtp) |*m| {
            m.deinit(self.allocator);
            self.qwen4_mtp = null;
        }
        if (self.qwen4) |st| {
            st.deinit();
            self.allocator.destroy(st);
            self.qwen4 = null;
        }
        for (self.moe_owned_bf16) |a| _ = mlx.mlx_array_free(a);
        self.allocator.free(self.moe_owned_bf16);
    }
};
