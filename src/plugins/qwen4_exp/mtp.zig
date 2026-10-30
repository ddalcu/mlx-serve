//! qwen4_exp's MTP head (`Qwen4Mtp`, `Qwen4MtpState` swapped onto the module): its load, forward over the pre-mixer
//! stream, QSA layer, history capture and rollback, the draft rerank, and the head-graph meters.

const std = @import("std");
const mtp_mod = @import("../../sdk/spec/mtp/mtp.zig");
const mlx = @import("../../mlx/ffi.zig");
const mrope = @import("../../sdk/vision/mrope.zig");
const exl3 = @import("../../sdk/quant/exl3/format.zig");
const kv_quant = @import("../../engines/mlx/cache/kv_quant.zig");
const KVQuantConfig = @import("../../engines/mlx/cache/kv_quant.zig").KVQuantConfig;
const model_mod = @import("../../sdk/config/model.zig");
const log = @import("../../sdk/util/log.zig");
const ModelConfig = @import("../../sdk/config/model.zig").ModelConfig;
const Weights = @import("../../sdk/config/model.zig").Weights;
const fused_activations = @import("../../sdk/blocks/mlp/fused_activations.zig");
const verify_projection = @import("../../sdk/blocks/linear/verify_projection.zig");
const MAX_BATCH_ROWS = @import("../../engines/mlx/batched/layered.zig").MAX_BATCH_ROWS;
const expectRowsByteEqual = @import("../../engines/mlx/batched/row_axis_verify.zig").expectRowsByteEqual;
const logicalSsmCapture = @import("../../engines/mlx/batched/row_axis_verify.zig").logicalSsmCapture;
const MoeLayerWeights = @import("../../engines/mlx/layer_weights.zig").MoeLayerWeights;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const KVCacheSnapshot = @import("../../engines/mlx/kv/cache.zig").KVCacheSnapshot;
const testing = @import("../../engines/mlx/kv/cache.zig").testing;
const init = @import("../../engines/mlx/load/init.zig").init;
const initMoeLayers = @import("../../engines/mlx/load/layers_layered.zig").initMoeLayers;
const weightTriple = @import("../../engines/mlx/load/weights.zig").weightTriple;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const SSMCacheEntrySnapshot = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntrySnapshot;
const ssmFreeSpecCapture = @import("../../engines/mlx/ssm/entry.zig").ssmFreeSpecCapture;
const ssmRestore = @import("../../engines/mlx/ssm/entry.zig").ssmRestore;
const ssmSnapshot = @import("../../engines/mlx/ssm/entry.zig").ssmSnapshot;
const ssmSnapshotDeinit = @import("../../engines/mlx/ssm/entry.zig").ssmSnapshotDeinit;
const Trunk = @import("trunk.zig").Trunk;
const concatAxis0 = @import("../../mlx/array.zig").concatAxis0;
const materializedOwnedCopy = @import("../../mlx/array.zig").materializedOwnedCopy;
const attn256MaxDiff = @import("../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const SingleVerifyFeature = @import("../../sdk/blocks/linear/verify_projection.zig").SingleVerifyFeature;
const mtpHeadRowsFit = @import("../../sdk/blocks/linear/verify_projection.zig").mtpHeadRowsFit;
const verifySharedHardware = @import("../../sdk/blocks/linear/verify_projection.zig").verifySharedHardware;
const VqmmLane = @import("../../sdk/blocks/linear/verify_qmm.zig").VqmmLane;
const vqmmLaneFor = @import("../../sdk/blocks/linear/verify_qmm.zig").vqmmLaneFor;
const fusedSwiGLU = @import("../../sdk/blocks/mlp/fused_activations.zig").fusedSwiGLU;
const gatherExpertMm = @import("../../sdk/blocks/moe/experts.zig").gatherExpertMm;
const MoeDecodeDispatchArm = @import("../../sdk/blocks/moe/gather_qmv_rows.zig").MoeDecodeDispatchArm;
const moeDecodeDispatchArm = @import("../../sdk/blocks/moe/gather_qmv_rows.zig").moeDecodeDispatchArm;
const moeRowsAffineBank = @import("../../sdk/blocks/moe/gather_qmv_rows.zig").moeRowsAffineBank;
const MoeMlpWeights = @import("../../sdk/blocks/moe/weights.zig").MoeMlpWeights;
const endMropeChunk = @import("../../engines/mlx/methods/mrope.zig").endMropeChunk;
const qwen4Standin = @import("../../sdk/diag/standin.zig").qwen4Standin;
const mlxSettledActiveBytes = @import("../../sdk/testing/parity.zig").mlxSettledActiveBytes;
const qwen4CompareRows = @import("../../sdk/testing/parity.zig").qwen4CompareRows;
const qwen4ReadF32 = @import("../../sdk/testing/parity.zig").qwen4ReadF32;
const testReadF32 = @import("../../sdk/testing/parity.zig").testReadF32;
const Qwen4TestSlot = @import("forward.zig").Qwen4TestSlot;
const loadHcWeights = @import("load.zig").loadHcWeights;
const qsaHistoryRows = @import("qsa/checkpoint.zig").qsaHistoryRows;
const QSA_RING_ROWS = @import("qsa/history.zig").QSA_RING_ROWS;
const QsaHeadMark = @import("qsa/history.zig").QsaHeadMark;
const QsaHeadMarkSet = @import("qsa/history.zig").QsaHeadMarkSet;
const qsaLeftoverAt = @import("qsa/history.zig").qsaLeftoverAt;
const qsaResliceToKeep = @import("qsa/history.zig").qsaResliceToKeep;
const qsaTestAppendPool = @import("qsa/history.zig").qsaTestAppendPool;
const qsaTestMeanBlocks = @import("qsa/history.zig").qsaTestMeanBlocks;
const reserveQsaHistoryWithHead = @import("qsa/history.zig").reserveQsaHistoryWithHead;
const ssmFreeQsaState = @import("qsa/history.zig").ssmFreeQsaState;
const truncatePooled = @import("qsa/history.zig").truncatePooled;
const headPlaceRopePos = @import("qsa/head_place.zig").headPlaceRopePos;
const qsaScoreFusedActive = @import("../../sdk/blocks/attention/sparse_block/score.zig").qsaScoreFusedActive;
const qsaScoreFusedActiveFor = @import("../../sdk/blocks/attention/sparse_block/score.zig").qsaScoreFusedActiveFor;
const qsaArraysAllEqual = @import("../../sdk/testing/parity.zig").qsaArraysAllEqual;
const HcWeights = @import("weights.zig").HcWeights;
const gather_qmv_rows_vars = @import("../../sdk/blocks/moe/gather_qmv_rows.zig");

pub var qwen4_mtp_head_graphs: usize = 0;
pub var mtp_verify_moe_group_graphs: u64 = 0;
pub var mtp_verify_moe_group_last_ops: u64 = 0;
pub var mtp_head_force_batched_override: ?bool = null;

/// qwen4_exp MTP head (vLLM #53896 / SGLang #36497 `residual_linear_shared`):
///   x = fc_hidden(rms_{4H}(stream)) per stream + fc_embedding(rms_H(embed))
///   → one full-attention decoder layer (own hc/QSA/MoE) → mixer → lm_head.
/// Keeps its own KV cache (index `num_hidden_layers`) + indexer key history.
pub const Qwen4Mtp = struct {
    layer: MoeLayerWeights,
    pre_norm_emb: mlx.mlx_array,
    pre_norm_hidden: mlx.mlx_array,
    fc_emb_w: mlx.mlx_array,
    fc_emb_s: mlx.mlx_array,
    fc_emb_b: mlx.mlx_array,
    fc_hid_w: mlx.mlx_array,
    fc_hid_s: mlx.mlx_array,
    fc_hid_b: mlx.mlx_array,
    mixer: HcWeights,
    owned: []mlx.mlx_array,
    cache: KVCache,
    entry: SSMCacheEntry,
    seq_offset: usize = 0,
    /// The head's QSA leftovers at the trunk's checkpoint positions
    /// (`qwen4MtpMarkQsaLeftover`): what a warm clamp reads once the ring has
    /// moved past them. Per REQUEST — `qwen4MtpReset` clears it and an adopt
    /// replaces it with the committed set.
    qsa_marks: QsaHeadMarkSet = .{},
    /// Absolute position of the head's key row 0 (the first draft position
    /// after a reset).
    pos_base: c_int = 0,
    /// Cross-request EV controller seed — the in-checkpoint head's twin of
    /// `mtp.MtpModel.ev_seed_*`: the last HEALTHY request's per-index
    /// acceptance EMAs and base depth, written by `Generator.deinit` and read
    /// by the first `nextMtp` round of the next request. Without it every
    /// request re-warms the controller (MTP_EV_WARMUP_ROUNDS legacy rounds
    /// plus a +1/round base climb), which is a sizeable share of a short
    /// generation. Per-LOADED-MODEL state, exactly like the sidecar's: it
    /// lives on the head, so it dies with the head and can never reach
    /// another model. `qwen4MtpReset` deliberately does not clear it — a
    /// reset starts a new REQUEST, which is precisely who should inherit it.
    ev_seed_accept: ?[mtp_mod.MAX_DEPTH]f32 = null,
    ev_seed_m_lo: u32 = 1,
    /// Draft rerank (`mtp.rerankSelect`): a low-bit copy of the TRUNK lm_head
    /// that shortlists 32 candidates, which the real head then re-scores.
    /// Built at LOAD by `qwen4BuildDraftRerank` — not in `loadQwen4Mtp`, which
    /// runs while the Trunk is still under construction and `lm_head_w`
    /// is not assigned yet, but from the scheduler's load path once it is.
    /// `tried` makes it (and the refusal) one-shot, so the draft path's own
    /// ask is a pure read; a null `rerank` after it means the full readout is
    /// this head's draft path.
    rerank: ?mtp_mod.RerankCoarse = null,
    rerank_logged: bool = false,
    rerank_tried: bool = false,

    /// Frees what the loader built and the head's per-request state.
    pub fn deinit(m: *Qwen4Mtp, a: std.mem.Allocator) void {
        for (m.owned) |arr| _ = mlx.mlx_array_free(arr);
        a.free(m.owned);
        m.cache.deinit();
        _ = mlx.mlx_array_free(m.entry.conv_state);
        _ = mlx.mlx_array_free(m.entry.ssm_state);
        ssmFreeQsaState(&m.entry);
        m.qsa_marks.deinit();
        if (m.rerank) |*rc| rc.deinit();
    }
};

/// Load the qwen4_exp MTP head when the pack ships `mtp.*` (null otherwise).
pub fn loadQwen4Mtp(allocator: std.mem.Allocator, config: ModelConfig, weights: *Weights, name_buf: *[256]u8, s: mlx.mlx_stream) !?Qwen4Mtp {
    const mtp_prefix = if (weights.get("language_model.mtp.fc_hidden.weight") != null)
        "language_model.mtp"
    else if (weights.get("mtp.fc_hidden.weight") != null)
        "mtp"
    else
        return null;
    var mcfg = config;
    // The head's experts stay whole on a tensor-parallel rank 0 (`distributed.headOnly`).
    if (mcfg.dist_shard) |sh| {
        mcfg.moe_intermediate_size *= sh.size;
        mcfg.shared_expert_intermediate_size *= sh.size;
        mcfg.dist_shard = null;
    }
    mcfg.weight_prefix = mtp_prefix;
    mcfg.num_hidden_layers = 1;
    mcfg.full_attention_interval = 1; // layer 0 is the full-attention layer
    mcfg.linear_attn_tail_from = 0;
    mcfg.ple_layer_idx = -1;
    const ml = try initMoeLayers(allocator, mcfg, weights, name_buf, s);
    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    errdefer owned.deinit(allocator);
    try owned.appendSlice(allocator, ml.owned_bf16);
    allocator.free(ml.owned_bf16);
    const layer = ml.moe_layers[0];
    allocator.free(ml.moe_layers);
    const entry = ml.ssm_entries[0];
    allocator.free(ml.ssm_entries);
    var fe_buf: [80]u8 = undefined;
    var fh_buf: [80]u8 = undefined;
    var mixer_buf: [80]u8 = undefined;
    var pne_buf: [80]u8 = undefined;
    var pnh_buf: [80]u8 = undefined;
    const fe = try weightTriple(weights, try std.fmt.bufPrint(&fe_buf, "{s}.fc_embedding", .{mtp_prefix}), null, &owned, allocator, s);
    const fh = try weightTriple(weights, try std.fmt.bufPrint(&fh_buf, "{s}.fc_hidden", .{mtp_prefix}), null, &owned, allocator, s);
    const mixer = try loadHcWeights(weights, try std.fmt.bufPrint(&mixer_buf, "{s}.hyper_connection_mixer", .{mtp_prefix}), false, config.hc_count, config.hidden_size, &owned, allocator, s);
    const pne = weights.get(try std.fmt.bufPrint(&pne_buf, "{s}.pre_fc_norm_embedding.weight", .{mtp_prefix})) orelse return error.MissingWeight;
    const pnh = weights.get(try std.fmt.bufPrint(&pnh_buf, "{s}.pre_fc_norm_hidden.weight", .{mtp_prefix})) orelse return error.MissingWeight;
    var cache = try KVCache.init(allocator, config.num_hidden_layers + 1);
    errdefer cache.deinit();
    log.info("[qwen4] MTP head loaded (1 hyper-connected QSA+MoE layer; drafts armed by --mtp)\n", .{});
    return .{
        .layer = layer,
        .pre_norm_emb = pne,
        .pre_norm_hidden = pnh,
        .fc_emb_w = fe.w,
        .fc_emb_s = fe.sc,
        .fc_emb_b = fe.bi,
        .fc_hid_w = fh.w,
        .fc_hid_s = fh.sc,
        .fc_hid_b = fh.bi,
        .mixer = mixer,
        .owned = try owned.toOwnedSlice(allocator),
        .cache = cache,
        .entry = entry,
    };
}

/// The head's PER-REQUEST half. Every request owns one and installs it on the
/// module (`qwen4MtpActivate`) before any head operation; the module's own copy
/// is whatever the last owner swapped out.
pub const Qwen4MtpState = struct {
    cache: KVCache,
    entry: SSMCacheEntry,
    seq_offset: usize = 0,
    qsa_marks: QsaHeadMarkSet = .{},
    pos_base: c_int = -1,

    pub const Meta = struct {
        stash_len: usize,
        origin: c_int,
        cache_step: usize,
    };

    pub fn meta(self: *const Qwen4MtpState) Meta {
        return .{
            .stash_len = self.seq_offset,
            .origin = self.pos_base,
            .cache_step = self.cache.step,
        };
    }

    pub fn restoreMeta(self: *Qwen4MtpState, m: Meta) void {
        self.seq_offset = m.stash_len;
        self.pos_base = m.origin;
        self.cache.step = m.cache_step;
    }

    pub fn deinit(self: *Qwen4MtpState) void {
        self.cache.deinit();
        ssmFreeQsaState(&self.entry);
        self.qsa_marks.deinit();
    }
};

/// Pointers to one head's live runtime. The tick holds N of these without a module swap.
pub const Qwen4MtpLive = struct {
    cache: *KVCache,
    entry: *SSMCacheEntry,
    seq_offset: *usize,
    pos_base: *c_int,
    qsa_marks: *QsaHeadMarkSet,
};

pub fn qwen4MtpLiveState(st: *Qwen4MtpState) Qwen4MtpLive {
    return .{
        .cache = &st.cache,
        .entry = &st.entry,
        .seq_offset = &st.seq_offset,
        .pos_base = &st.pos_base,
        .qsa_marks = &st.qsa_marks,
    };
}

pub fn qwen4MtpLiveModule(m: *Qwen4Mtp) Qwen4MtpLive {
    return .{
        .cache = &m.cache,
        .entry = &m.entry,
        .seq_offset = &m.seq_offset,
        .pos_base = &m.pos_base,
        .qsa_marks = &m.qsa_marks,
    };
}

/// A fresh, empty per-request head state (what `qwen4MtpReset` leaves behind).
pub fn qwen4MtpStateNew(self: *Trunk) !Qwen4MtpState {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    var entry = m.entry;
    entry.conv_state = .{ .ctx = null };
    entry.ssm_state = .{ .ctx = null };
    entry.initialized = false;
    entry.aux_state = .{ .ctx = null };
    entry.qsa_key_buf = .{ .ctx = null };
    entry.qsa_key_rows = 0;
    entry.qsa_hist_rows = 0;
    entry.qsa_pooled = .{ .ctx = null };
    entry.qsa_pooled_buf = .{ .ctx = null };
    entry.qsa_pooled_blocks = 0;
    entry.qsa_score_bank = .{ .ctx = null };
    entry.qsa_score_buf = .{ .ctx = null };
    entry.qsa_score_blocks = 0;
    entry.spec_state_seq = .{ .ctx = null };
    entry.spec_conv_input = .{ .ctx = null };
    entry.spec_ple_input = .{ .ctx = null };
    const cache = try KVCache.initWithConfig(self.allocator, self.config.num_hidden_layers + 1, m.cache.config);
    return .{ .cache = cache, .entry = entry };
}

pub fn qwen4MtpSwapState(m: *Qwen4Mtp, st: *Qwen4MtpState) void {
    std.mem.swap(KVCache, &m.cache, &st.cache);
    std.mem.swap(SSMCacheEntry, &m.entry, &st.entry);
    std.mem.swap(usize, &m.seq_offset, &st.seq_offset);
    std.mem.swap(QsaHeadMarkSet, &m.qsa_marks, &st.qsa_marks);
    std.mem.swap(c_int, &m.pos_base, &st.pos_base);
}

/// Install `st` as the head's live state (swapping the previous owner's out). Idempotent.
pub fn qwen4MtpActivate(self: *Trunk, st: *Qwen4MtpState) void {
    const m = &(self.qwen4_mtp orelse return);
    if (self.qwen4_mtp_owner == st) return;
    if (self.qwen4_mtp_owner) |prev| qwen4MtpSwapState(m, prev);
    qwen4MtpSwapState(m, st);
    self.qwen4_mtp_owner = st;
}

/// Take `st` back off the module before it is freed.
pub fn qwen4MtpRelease(self: *Trunk, st: *Qwen4MtpState) void {
    const m = &(self.qwen4_mtp orelse return);
    if (self.qwen4_mtp_owner != st) return;
    qwen4MtpSwapState(m, st);
    self.qwen4_mtp_owner = null;
}

/// Reset the MTP head's per-request state (KV, indexer keys, position).
pub fn qwen4MtpReset(self: *Trunk) !void {
    const m = &(self.qwen4_mtp orelse return);
    try m.cache.reinit(self.config.num_hidden_layers + 1, m.cache.config);
    ssmFreeQsaState(&m.entry);
    m.qsa_marks.deinit();
    m.seq_offset = 0;
    m.pos_base = -1;
}

pub fn qwen4MtpResetOwned(self: *Trunk, enable_mtp: bool) void {
    if (enable_mtp) self.qwen4MtpReset() catch {};
}

/// Re-creates the head cache under `cfg` (the engine's choice for the head) with the HEAD's own geometry (layers +
/// 1, its key head dim).
pub fn qwen4MtpApplyKvQuant(self: *Trunk, cfg: KVQuantConfig) !void {
    const m = &(self.qwen4_mtp orelse return);
    const n_layers: u32 = self.config.num_hidden_layers + 1;
    if (std.meta.eql(m.cache.config, cfg) and m.cache.entries.len == n_layers) return;
    try m.cache.reinit(n_layers, cfg);
}

/// Record the head's QSA leftover at `row`, the trunk's checkpoint position in the head's
/// own row space (absolute minus the head's base): a warm restore clamps the head to
/// exactly that row, by then a whole generated tail below its ring. A head that is not at
/// `row` — one whose history window skipped these chunks — marks nothing. Best effort; a
/// missing mark is a blind head, never a wrong one.
pub fn qwen4MtpMarkQsaLeftover(self: *Trunk, row: usize) !void {
    const m = &(self.qwen4_mtp orelse return);
    if (row != m.seq_offset) return;
    const pos: c_int = @intCast(m.seq_offset);
    const ratio = @max(m.entry.qsa_ratio, 1);
    // An aligned position needs no rows at all: the pooled bank is the whole history there.
    if (pos == 0 or @mod(pos, ratio) == 0 or m.entry.aux_state.ctx == null) return;
    const rows = try qsaLeftoverAt(m.entry.aux_state, qsaHistoryRows(&m.entry), pos, ratio, self.s);
    if (rows.ctx == null) return;
    // The copy is lazy and its parent is the ring the next append rewrites.
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, rows);
        _ = mlx.mlx_eval(vec);
    }
    m.qsa_marks.put(pos, rows);
}

/// The marks a commit stores beside the head's QSA half.
pub fn qwen4MtpMarks(self: *const Trunk) []const QsaHeadMark {
    if (self.qwen4_mtp) |*m| return m.qsa_marks.slice();
    return &.{};
}

/// MTP draft logits: `stream_prev` `[B,S,hc*H]` is the trunk's pre-mixer
/// stream at positions p.., `token_ids` `[B,S]` the tokens at p+1.., and
/// `pos_offset` = p+1 of the first row (the draft query's own position).
/// Returns `[B,S,V]` logits for positions p+2... Caller frees.
/// `logits` is the last-row projection (null unless `.all_rows`/`.last_row`),
/// `stream` the pre-mixer hyper-connection stream the next chain step reads,
/// and `mixed` the MIXER output — the vector the lm_head consumes — which
/// only `.mixed_last_row` fills. The three are distinct spaces: `stream` is
/// `[B,S,hc*H]`, `mixed` is `[B,1,H]`, and handing the wrong one to the
/// lm_head is a silent shape error, not a wrong answer.
pub const Qwen4MtpOut = struct {
    logits: mlx.mlx_array,
    stream: mlx.mlx_array,
    mixed: mlx.mlx_array = .{ .ctx = null },
};

/// Owns the rows built so far, so a failure part-way frees exactly them.
pub const Qwen4MtpOutBuf = struct {
    items: []Qwen4MtpOut,
    built: usize = 0,
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator, n: usize) !Qwen4MtpOutBuf {
        return .{ .items = try allocator.alloc(Qwen4MtpOut, n), .allocator = allocator };
    }

    fn deinit(self: *Qwen4MtpOutBuf) void {
        for (self.items[0..self.built]) |o| {
            if (o.logits.ctx != null) _ = mlx.mlx_array_free(o.logits);
            if (o.stream.ctx != null) _ = mlx.mlx_array_free(o.stream);
            if (o.mixed.ctx != null) _ = mlx.mlx_array_free(o.mixed);
        }
        self.allocator.free(self.items);
    }

    fn abandon(self: *Qwen4MtpOutBuf) []Qwen4MtpOut {
        return self.items;
    }
};

/// What `qwen4MtpForward` owes its caller past the head's own layer. The
/// mixer read + the 248320-wide lm_head run over whatever rows `stream`
/// carries, so a caller that consumes ONE row says so instead of paying
/// for `S` of them and slicing (the merged history step after a partial
/// accept is `1 + accepted` rows wide, of which the drafter wants the
/// last).
pub const Qwen4MtpProject = enum {
    /// Every row's logits + the full `[B,S,hc*H]` stream (fixture oracles).
    all_rows,
    /// The LAST row only: `h` is sliced BEFORE the mixer, so `logits` is
    /// `[B,1,V]` and `stream` is that one row.
    last_row,
    /// No logits at all (history append): mixer + lm_head are skipped and
    /// `logits` comes back null; `stream` is the full `[B,S,hc*H]`.
    none,
    /// The LAST row's mixer output in `mixed`, and NO lm_head: the draft
    /// rerank path shortlists through its own coarse head, so the 248320-
    /// wide projection — 675 MB, the single biggest read in a draft step —
    /// never runs. `logits` comes back null; `stream` is the last row.
    mixed_last_row,
};

/// Truncate the head's committed history to `len` rows.
pub fn qwen4MtpTruncate(self: *Trunk, len: usize) !void {
    const m = &(self.qwen4_mtp orelse return);
    var live = qwen4MtpLiveModule(m);
    try self.qwen4MtpTruncateLive(&live, len);
}

pub fn qwen4MtpTruncateOn(self: *Trunk, st: *Qwen4MtpState, len: usize) !void {
    var live = qwen4MtpLiveState(st);
    try self.qwen4MtpTruncateLive(&live, len);
}

pub fn qwen4MtpTruncateLive(self: *Trunk, live: *Qwen4MtpLive, len: usize) !void {
    if (len >= live.seq_offset.*) return;
    try live.cache.truncate(len, self.s);
    if (live.entry.aux_state.ctx != null or live.entry.qsa_pooled.ctx != null) {
        if (len == 0) {
            ssmFreeQsaState(live.entry);
        } else {
            const keep: c_int = @intCast(len);
            if (!try qsaResliceToKeep(live.entry, keep, self.s)) {
                if (live.entry.aux_state.ctx == null) {
                    live.entry.qsa_hist_rows = keep;
                    try truncatePooled(&live.entry.qsa_pooled, @intCast(len), live.entry.qsa_ratio, self.s, true);
                } else {
                    const hist = qsaHistoryRows(live.entry);
                    const drop_tail = hist - keep;
                    const ks = mlx.getShape(live.entry.aux_state);
                    const new_held = ks[1] - drop_tail;
                    if (new_held < @mod(keep, @max(live.entry.qsa_ratio, 1))) {
                        try self.qwen4MtpTruncateToMarkLive(live, keep);
                    } else {
                        const start = [_]c_int{ 0, 0, 0 };
                        const stop = [_]c_int{ ks[0], new_held, ks[2] };
                        const strides = [_]c_int{ 1, 1, 1 };
                        var view = mlx.mlx_array_new();
                        defer _ = mlx.mlx_array_free(view);
                        try mlx.check(mlx.mlx_slice(&view, live.entry.aux_state, &start, 3, &stop, 3, &strides, 3, self.s));
                        const owned = try materializedOwnedCopy(self.s, view);
                        const pooled = live.entry.qsa_pooled;
                        live.entry.qsa_pooled = .{ .ctx = null };
                        ssmFreeQsaState(live.entry);
                        live.entry.qsa_pooled = pooled;
                        live.entry.aux_state = owned;
                        live.entry.qsa_key_rows = new_held;
                        live.entry.qsa_hist_rows = keep;
                        try truncatePooled(&live.entry.qsa_pooled, @intCast(len), live.entry.qsa_ratio, self.s, true);
                    }
                }
            }
        }
    }
    live.qsa_marks.dropAbove(@intCast(len));
    live.seq_offset.* = len;
}

pub fn qwen4MtpTruncateToMarkLive(self: *Trunk, live: *Qwen4MtpLive, keep: c_int) !void {
    const want_lv = @mod(keep, @max(live.entry.qsa_ratio, 1));
    var rows: mlx.mlx_array = .{ .ctx = null };
    if (want_lv > 0) {
        const marked = live.qsa_marks.find(keep) orelse return error.MtpHeadQsaRingGap;
        if (mlx.getShape(marked)[1] != want_lv) return error.MtpHeadQsaRingGap;
        rows = mlx.mlx_array_new();
        mlx.check(mlx.mlx_array_set(&rows, marked)) catch |err| {
            _ = mlx.mlx_array_free(rows);
            return err;
        };
    }
    const pooled = live.entry.qsa_pooled;
    live.entry.qsa_pooled = .{ .ctx = null };
    ssmFreeQsaState(live.entry);
    live.entry.qsa_pooled = pooled;
    live.entry.aux_state = rows;
    live.entry.qsa_key_rows = want_lv;
    live.entry.qsa_hist_rows = keep;
    try truncatePooled(&live.entry.qsa_pooled, keep, live.entry.qsa_ratio, self.s, true);
}

/// Adopt a committed head history from the prefix cache: the head's KV plus the QSA aux
/// entry it is only valid with, trimmed to `want` rows. Any failure resets the head to blank.
pub fn qwen4MtpAdopt(
    self: *Trunk,
    kv_snap: *const KVCacheSnapshot,
    aux: *const SSMCacheEntrySnapshot,
    marks: []const QsaHeadMark,
    pos_base: c_int,
    want: usize,
) !void {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    if (kv_snap.step > 0 and pos_base < 0) return error.MtpHeadInvalidOrigin;
    // The head's row count IS its cache's step (`qwen4MtpAdvance`).
    if (m.cache.step != m.seq_offset) return error.MtpHeadStepGap;
    // A history that is not exactly as long as the KV is not adoptable at any length.
    if (aux.aux_state.ctx == null and aux.qsa_pooled.ctx == null) return error.MtpHeadNoQsaHistory;
    const hist: c_int = if (aux.qsa_rows > 0) aux.qsa_rows else blk: {
        if (aux.aux_state.ctx == null) break :blk 0;
        const sh = mlx.getShape(aux.aux_state);
        break :blk if (sh.len >= 2) sh[1] else 0;
    };
    if (hist != @as(c_int, @intCast(kv_snap.step))) return error.MtpHeadQsaHistoryGap;
    errdefer self.qwen4MtpReset() catch {};
    try m.cache.restore(kv_snap);
    // `ssmRestore` replaces the aux state, so a stale history cannot survive.
    try ssmRestore(&m.entry, aux);
    // Same discipline for the marks: the clamp below reads the committed set, never a
    // previous request's.
    m.qsa_marks.deinit();
    m.qsa_marks = QsaHeadMarkSet.share(marks);
    m.seq_offset = kv_snap.step;
    m.pos_base = pos_base;
    // Unconditional clamp: a snapshot's buffer can be longer than the matched prefix.
    try self.qwen4MtpTruncate(want);
    if (m.seq_offset != want) return error.MtpHeadTrimGap;
}

/// Commit a forward's `seq_len` rows to the head's own bookkeeping: `KVCache.update`
/// advances `step` only at layer 0, and the head's layer is `num_hidden_layers`. Without
/// this the committed snapshot's step disagreed with its QSA history and every restore declined.
pub fn qwen4MtpAdvance(cache: *KVCache, seq_offset: *usize, seq_len: c_int) void {
    seq_offset.* += @intCast(seq_len);
    cache.step = seq_offset.*;
}

/// `mrope_ctx` (image turns): the head's rows sit at ABSOLUTE positions
/// `pos_base + seq_offset ..`, so its attention + QSA indexer read the
/// slot's 3-D table there (prompt rows spanning the image) and the
/// scalar `offset + delta` past it — the same two arms as the trunk.
pub fn qwen4MtpForward(self: *Trunk, stream_prev: mlx.mlx_array, token_ids_in: mlx.mlx_array, pos_offset: c_int, mrope_ctx: ?mrope.PositionContext, project: Qwen4MtpProject) !Qwen4MtpOut {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    var live = qwen4MtpLiveModule(m);
    return self.qwen4MtpForwardLive(m, &live, stream_prev, token_ids_in, pos_offset, mrope_ctx, project, null);
}

pub fn qwen4MtpForwardOn(self: *Trunk, st: *Qwen4MtpState, stream_prev: mlx.mlx_array, token_ids_in: mlx.mlx_array, pos_offset: c_int, mrope_ctx: ?mrope.PositionContext, project: Qwen4MtpProject) !Qwen4MtpOut {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    var live = qwen4MtpLiveState(st);
    return self.qwen4MtpForwardLive(m, &live, stream_prev, token_ids_in, pos_offset, mrope_ctx, project, null);
}

/// A draft row written past a padded history append. Key slots `[live_end, dead_end)`
/// are dead history rows: hidden from its attention and its QSA block selection. The
/// row ropes at `live_end + (slot - dead_end)`, the position it would hold with no pad.
pub const HeadPlace = struct {
    /// 0-d int32 slot one past the last live history row.
    live_end: mlx.mlx_array,
    dead_end: c_int,
};

pub fn qwen4MtpForwardPlaced(self: *Trunk, stream_prev: mlx.mlx_array, token_ids_in: mlx.mlx_array, place: *const HeadPlace, project: Qwen4MtpProject) !Qwen4MtpOut {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    var live = qwen4MtpLiveModule(m);
    return self.qwen4MtpForwardLive(m, &live, stream_prev, token_ids_in, 0, null, project, place);
}

pub const Qwen4MtpBatchRow = struct {
    st: *Qwen4MtpState,
    stream: mlx.mlx_array,
    token_ids: mlx.mlx_array,
    pos_offset: c_int,
    mrope_ctx: ?mrope.PositionContext = null,
};

pub fn mtpPadSeq(self: *Trunk, arr: mlx.mlx_array, want: c_int) !mlx.mlx_array {
    const sh = mlx.getShape(arr);
    if (sh[1] == want) {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&out, arr));
        return out;
    }
    const high = [_]c_int{want - sh[1]};
    const z = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(z);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_pad(&out, arr, &[_]c_int{1}, 1, &[_]c_int{0}, 1, &high, 1, z, "constant", self.s));
    return out;
}

pub fn mtpHeadForceBatched() bool {
    return mtp_head_force_batched_override orelse false;
}

pub fn mtpMoeRows(self: *Trunk, x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    // EXL3 head rows ride the decode chain only.
    if (self.config.exl3 != null and exl3.kernels.usesPrefillArm(exl3.kernels.rowsOfShape(mlx.getShape(x))))
        return error.Exl3MtpRowsExceedDecode;
    if (mw.shared_ungated or mw.shared_expert_gate_w == null or qwen4Standin().moe_shared) return self.moeMLP(x, mw);
    var routed = mw.*;
    routed.shared_expert_gate_w = null;
    const expert_sum = try self.moeMLP(x, &routed);
    defer _ = mlx.mlx_array_free(expert_sum);
    const down = (try self.moeSharedDown(.mtp_rows, x, mw, &.{})).?;
    defer _ = mlx.mlx_array_free(down);
    const gate_logit = try self.mtpProjectionRows(x, mw.shared_expert_gate_w.?, mw.shared_expert_gate_s.?, mw.shared_expert_gate_b.?, false);
    defer _ = mlx.mlx_array_free(gate_logit);
    return self.moeSharedGateTail(expert_sum, gate_logit, down);
}

pub fn qwen4MtpForwardBatched(
    self: *Trunk,
    rows: []const Qwen4MtpBatchRow,
    project: Qwen4MtpProject,
    depth: u32,
) ![]Qwen4MtpOut {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    if (rows.len == 0) return error.EmptyMtpBatch;
    if (rows.len > MAX_BATCH_ROWS) return error.MtpBatchTooWide;
    var any_mrope = false;
    var widths: [MAX_BATCH_ROWS]c_int = undefined;
    for (rows, 0..) |r, i| {
        const shape = mlx.getShape(r.stream);
        if (shape.len != 3) return error.MtpHeadStreamShape;
        widths[i] = shape[1];
        if (r.mrope_ctx != null) any_mrope = true;
    }
    const fits = mtpHeadRowsFit(widths[0..rows.len], self.config.hc_count);
    const use_batch = fits and !any_mrope and (rows.len > 1 or mtpHeadForceBatched());
    if (!use_batch) {
        var buf = try Qwen4MtpOutBuf.init(self.allocator, rows.len);
        errdefer buf.deinit();
        for (rows, 0..) |r, i| {
            buf.items[i] = try self.qwen4MtpForwardOn(r.st, r.stream, r.token_ids, r.pos_offset, r.mrope_ctx, project);
            buf.built = i + 1;
        }
        return buf.abandon();
    }

    const Once = struct {
        var logged: bool = false;
    };
    if (!Once.logged) {
        Once.logged = true;
        log.info("[batched] mtp head engaged (slots={d} depth={d})\n", .{ rows.len, depth });
    }

    qwen4_mtp_head_graphs += 1;
    self.fwd_gen +%= 1;
    const cfg = &self.config;
    const hc: c_int = @intCast(cfg.hc_count);
    const hidden: c_int = @intCast(cfg.hidden_size);
    const N: c_int = @intCast(rows.len);
    var seqs: [MAX_BATCH_ROWS]c_int = undefined;
    var s_max: c_int = 0;
    for (rows, 0..) |r, i| {
        const sh = mlx.getShape(r.stream);
        seqs[i] = sh[1];
        if (seqs[i] > s_max) s_max = seqs[i];
        if (r.st.seq_offset == 0) r.st.pos_base = r.pos_offset;
        if (r.pos_offset != r.st.pos_base + @as(c_int, @intCast(r.st.seq_offset))) return error.MtpPositionGap;
    }

    var tok_parts: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var str_parts: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    for (tok_parts[0..rows.len], 0..) |*p, i| {
        p.* = .{ .ctx = null };
        str_parts[i] = .{ .ctx = null };
    }
    defer for (tok_parts[0..rows.len], 0..) |p, i| {
        if (p.ctx != null) _ = mlx.mlx_array_free(p);
        if (str_parts[i].ctx != null) _ = mlx.mlx_array_free(str_parts[i]);
    };
    for (rows, 0..) |r, i| {
        var tok2 = mlx.mlx_array_new();
        errdefer if (tok2.ctx != null) {
            _ = mlx.mlx_array_free(tok2);
        };
        try mlx.check(mlx.mlx_reshape(&tok2, r.token_ids, &[_]c_int{ 1, seqs[i] }, 2, self.s));
        tok_parts[i] = try self.mtpPadSeq(tok2, s_max);
        _ = mlx.mlx_array_free(tok2);
        tok2 = .{ .ctx = null };
        str_parts[i] = try self.mtpPadSeq(r.stream, s_max);
    }
    const token_ids = try concatAxis0(self.s, tok_parts[0..rows.len]);
    defer _ = mlx.mlx_array_free(token_ids);
    const stream_prev = try concatAxis0(self.s, str_parts[0..rows.len]);
    defer _ = mlx.mlx_array_free(stream_prev);

    const e = try self.embedding(token_ids);
    defer _ = mlx.mlx_array_free(e);
    const en = try self.rmsNorm(e, m.pre_norm_emb);
    defer _ = mlx.mlx_array_free(en);
    const ep = try self.mtpProjectionRows(en, m.fc_emb_w, m.fc_emb_s, m.fc_emb_b, false);
    defer _ = mlx.mlx_array_free(ep);
    const hn = try self.rmsNorm(stream_prev, m.pre_norm_hidden);
    defer _ = mlx.mlx_array_free(hn);
    var hn4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hn4);
    try mlx.check(mlx.mlx_reshape(&hn4, hn, &[_]c_int{ N, s_max, hc, hidden }, 4, self.s));
    const hp = try self.mtpProjectionRows(hn4, m.fc_hid_w, m.fc_hid_s, m.fc_hid_b, false);
    defer _ = mlx.mlx_array_free(hp);
    var ep4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ep4);
    try mlx.check(mlx.mlx_reshape(&ep4, ep, &[_]c_int{ N, s_max, 1, hidden }, 4, self.s));
    var x4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x4);
    try mlx.check(mlx.mlx_add(&x4, hp, ep4, self.s));
    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_reshape(&h, x4, &[_]c_int{ N, s_max, hc * hidden }, 3, self.s));

    var slot_ctxs: [MAX_BATCH_ROWS]ForwardCtx = undefined;
    var slot_ptrs: [MAX_BATCH_ROWS]*ForwardCtx = undefined;
    for (rows, 0..) |r, i| {
        slot_ctxs[i] = .{
            .cache = &r.st.cache,
            .moe_seq_offset = &r.st.seq_offset,
            .ssm_entries = null,
            .capture_hidden = null,
            .vision_embeddings = null,
            .qsa_entry = &r.st.entry,
            .qsa_pos_base = r.st.pos_base,
        };
        slot_ptrs[i] = &slot_ctxs[i];
    }

    const li: u32 = cfg.num_hidden_layers;
    const lw = &m.layer;
    const fa = switch (lw.attn) {
        .full => |w| w,
        .linear => return error.MtpLayerNotAttention,
    };

    var h_parts: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var mlp_mix: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var mlp_inj: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    for (h_parts[0..rows.len], 0..) |*p, i| {
        p.* = .{ .ctx = null };
        mlp_mix[i] = .{ .ctx = null };
        mlp_inj[i] = .{ .ctx = null };
    }
    defer for (h_parts[0..rows.len], 0..) |p, i| {
        if (p.ctx != null) _ = mlx.mlx_array_free(p);
        if (mlp_mix[i].ctx != null) _ = mlx.mlx_array_free(mlp_mix[i]);
        if (mlp_inj[i].ctx != null) _ = mlx.mlx_array_free(mlp_inj[i]);
    };
    {
        const hsh = mlx.getShape(h);
        for (rows, 0..) |r, i| {
            const i_c: c_int = @intCast(i);
            var h_i = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(h_i);
            try mlx.check(mlx.mlx_slice(&h_i, h, &[_]c_int{ i_c, 0, 0 }, 3, &[_]c_int{ i_c + 1, seqs[i], hsh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
            var pre = try self.hcRead(h_i, &lw.hc_attn.?, 1, seqs[i]);
            defer pre.deinit();
            const ao = try self.qwen4AttnWith(slot_ptrs[i], pre.mixed, &fa, &r.st.entry, li, @intCast(r.st.seq_offset), r.st.pos_base, 1, seqs[i], seqs[i] > 1);
            defer _ = mlx.mlx_array_free(ao);
            h_i = try self.hcWrite(h_i, ao, pre.inj, 1, seqs[i]);
            const pre2 = try self.hcRead(h_i, &lw.hc_mlp.?, 1, seqs[i]);
            mlp_mix[i] = pre2.mixed;
            mlp_inj[i] = pre2.inj;
            h_parts[i] = h_i;
        }
    }
    _ = mlx.mlx_array_free(h);
    h = .{ .ctx = null };
    const mix_all = try concatAxis0(self.s, mlp_mix[0..rows.len]);
    defer _ = mlx.mlx_array_free(mix_all);
    const mo_all = switch (lw.mlp) {
        .moe => |*mw| try self.mtpMoeRows(mix_all, mw),
        .dense => |*dw| try self.denseMLP(mix_all, dw),
    };
    defer _ = mlx.mlx_array_free(mo_all);
    const msh = mlx.getShape(mo_all);
    for (0..rows.len) |i| {
        const i_c: c_int = @intCast(i);
        var mo_i = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mo_i);
        try mlx.check(mlx.mlx_slice(&mo_i, mo_all, &[_]c_int{ i_c, 0, 0 }, 3, &[_]c_int{ i_c + 1, seqs[i], msh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
        h_parts[i] = try self.hcWrite(h_parts[i], mo_i, mlp_inj[i], 1, seqs[i]);
    }
    for (rows, 0..) |r, i| {
        qwen4MtpAdvance(&r.st.cache, &r.st.seq_offset, seqs[i]);
    }

    var buf = try Qwen4MtpOutBuf.init(self.allocator, rows.len);
    errdefer buf.deinit();

    if (project == .none) {
        for (0..rows.len) |i| {
            buf.items[i] = .{ .logits = .{ .ctx = null }, .stream = h_parts[i] };
            h_parts[i] = .{ .ctx = null };
            buf.built = i + 1;
        }
        return buf.abandon();
    }

    var last_parts: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    for (last_parts[0..rows.len]) |*p| p.* = .{ .ctx = null };
    defer for (last_parts[0..rows.len]) |p| {
        if (p.ctx != null) _ = mlx.mlx_array_free(p);
    };
    for (0..rows.len) |i| {
        var stream_i = h_parts[i];
        h_parts[i] = .{ .ctx = null };
        errdefer _ = mlx.mlx_array_free(stream_i);
        const ssh = mlx.getShape(stream_i);
        if ((project == .last_row or project == .mixed_last_row) and ssh[1] > 1) {
            var last = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(last);
            try mlx.check(mlx.mlx_slice(&last, stream_i, &[_]c_int{ 0, ssh[1] - 1, 0 }, 3, &[_]c_int{ 1, ssh[1], ssh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
            _ = mlx.mlx_array_free(stream_i);
            stream_i = last;
        }
        last_parts[i] = stream_i;
    }
    const last_h = try concatAxis0(self.s, last_parts[0..rows.len]);
    defer _ = mlx.mlx_array_free(last_h);
    for (last_parts[0..rows.len]) |*part| {
        if (part.ctx != null) _ = mlx.mlx_array_free(part.*);
        part.* = .{ .ctx = null };
    }
    const n_rows: c_int = @intCast(rows.len);
    const mix = try self.hcRead(last_h, &m.mixer, n_rows, 1);
    if (mix.inj.ctx != null) _ = mlx.mlx_array_free(mix.inj);
    defer if (mix.mixed.ctx != null) {
        _ = mlx.mlx_array_free(mix.mixed);
    };
    var logits_all: mlx.mlx_array = .{ .ctx = null };
    defer if (logits_all.ctx != null) {
        _ = mlx.mlx_array_free(logits_all);
    };
    if (project != .mixed_last_row) {
        logits_all = try self.mtpProjectionRows(mix.mixed, self.lm_head_w, self.lm_head_s, self.lm_head_b, true);
    }
    const lsh = if (logits_all.ctx != null) mlx.getShape(logits_all) else mlx.getShape(mix.mixed);
    for (0..rows.len) |i| {
        const i_c: c_int = @intCast(i);
        var stream_i = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(stream_i);
        try mlx.check(mlx.mlx_slice(&stream_i, last_h, &[_]c_int{ i_c, 0, 0 }, 3, &[_]c_int{ i_c + 1, 1, mlx.getShape(last_h)[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
        if (project == .mixed_last_row) {
            var mixed_i = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(mixed_i);
            try mlx.check(mlx.mlx_slice(&mixed_i, mix.mixed, &[_]c_int{ i_c, 0, 0 }, 3, &[_]c_int{ i_c + 1, 1, mlx.getShape(mix.mixed)[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
            buf.items[i] = .{ .logits = .{ .ctx = null }, .stream = stream_i, .mixed = mixed_i };
        } else {
            var logits = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(logits);
            try mlx.check(mlx.mlx_slice(&logits, logits_all, &[_]c_int{ i_c, 0, 0 }, 3, &[_]c_int{ i_c + 1, 1, lsh[lsh.len - 1] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
            buf.items[i] = .{ .logits = logits, .stream = stream_i };
        }
        buf.built = i + 1;
    }
    return buf.abandon();
}

pub fn qwen4MtpForwardLive(self: *Trunk, m: *Qwen4Mtp, live: *Qwen4MtpLive, stream_prev: mlx.mlx_array, token_ids_in: mlx.mlx_array, pos_offset: c_int, mrope_ctx: ?mrope.PositionContext, project: Qwen4MtpProject, place: ?*const HeadPlace) !Qwen4MtpOut {
    qwen4_mtp_head_graphs += 1;
    self.fwd_gen +%= 1; // per-forward QSA scratch key
    const cfg = &self.config;
    const hc: c_int = @intCast(cfg.hc_count);
    const hidden: c_int = @intCast(cfg.hidden_size);
    const shape = mlx.getShape(stream_prev);
    const batch: c_int = shape[0];
    const seq_len: c_int = shape[1];
    const is_prefill = seq_len > 1;
    if (place) |p| {
        if (seq_len != 1 or mrope_ctx != null or p.dead_end <= 0 or p.dead_end > @as(c_int, @intCast(live.seq_offset.*))) return error.MtpPlacement;
    } else {
        if (live.seq_offset.* == 0) live.pos_base.* = pos_offset;
        if (pos_offset != live.pos_base.* + @as(c_int, @intCast(live.seq_offset.*))) return error.MtpPositionGap;
    }

    var token_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(token_ids);
    try mlx.check(mlx.mlx_reshape(&token_ids, token_ids_in, &[_]c_int{ batch, seq_len }, 2, self.s));
    const e = try self.embedding(token_ids);
    defer _ = mlx.mlx_array_free(e);
    const en = try self.rmsNorm(e, m.pre_norm_emb);
    defer _ = mlx.mlx_array_free(en);
    const ep = try self.qmatmul(en, m.fc_emb_w, m.fc_emb_s, m.fc_emb_b);
    defer _ = mlx.mlx_array_free(ep);
    const hn = try self.rmsNorm(stream_prev, m.pre_norm_hidden); // ONE rms over all hc*H
    defer _ = mlx.mlx_array_free(hn);
    const shape4 = [_]c_int{ batch, seq_len, hc, hidden };
    var hn4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hn4);
    try mlx.check(mlx.mlx_reshape(&hn4, hn, &shape4, 4, self.s));
    const hp = try self.qmatmul(hn4, m.fc_hid_w, m.fc_hid_s, m.fc_hid_b);
    defer _ = mlx.mlx_array_free(hp);
    var ep4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ep4);
    try mlx.check(mlx.mlx_reshape(&ep4, ep, &[_]c_int{ batch, seq_len, 1, hidden }, 4, self.s));
    var x4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x4);
    try mlx.check(mlx.mlx_add(&x4, hp, ep4, self.s));
    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_reshape(&h, x4, &[_]c_int{ batch, seq_len, hc * hidden }, 3, self.s));

    var ctx: ForwardCtx = .{ .cache = live.cache, .moe_seq_offset = live.seq_offset, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    defer if (ctx.head_rope_pos.ctx != null) {
        _ = mlx.mlx_array_free(ctx.head_rope_pos);
    };
    if (place) |p| {
        ctx.head_place = p;
        ctx.head_rope_pos = try headPlaceRopePos(self.s, p, live.pos_base.* + @as(c_int, @intCast(live.seq_offset.*)) - p.dead_end);
    }
    if (mrope_ctx) |mc| {
        ctx.mrope_pos = mc.pos;
        ctx.mrope_total = mc.total;
        ctx.mrope_delta = mc.delta;
    }
    try self.beginMropeChunk(&ctx, @intCast(live.pos_base.* + @as(c_int, @intCast(live.seq_offset.*))), @intCast(seq_len), mlx.mlx_array_dtype(h));
    defer endMropeChunk(&ctx);
    const li: u32 = cfg.num_hidden_layers;
    const lw = &m.layer;
    var pre = try self.hcRead(h, &lw.hc_attn.?, batch, seq_len);
    defer pre.deinit();
    const attn_out = switch (lw.attn) {
        .full => |fa| try self.qwen4AttnWith(&ctx, pre.mixed, &fa, live.entry, li, @intCast(live.seq_offset.*), live.pos_base.*, batch, seq_len, is_prefill),
        .linear => return error.MtpLayerNotAttention,
    };
    defer _ = mlx.mlx_array_free(attn_out);
    h = try self.hcWrite(h, attn_out, pre.inj, batch, seq_len);
    var pre2 = try self.hcRead(h, &lw.hc_mlp.?, batch, seq_len);
    defer pre2.deinit();
    const mlp_out = switch (lw.mlp) {
        .moe => |*mw| try self.moeMLP(pre2.mixed, mw),
        .dense => |*dw| try self.denseMLP(pre2.mixed, dw),
    };
    defer _ = mlx.mlx_array_free(mlp_out);
    h = try self.hcWrite(h, mlp_out, pre2.inj, batch, seq_len);
    qwen4MtpAdvance(live.cache, live.seq_offset, seq_len);

    // History append wants the stream and nothing else: the mixer read and
    // the vocab-wide projection are pure, so skipping them is free.
    if (project == .none) return .{ .logits = .{ .ctx = null }, .stream = h };
    // Last-row-only: cut `h` HERE, so the mixer + lm_head see one row
    // instead of S. Same values as slicing the outputs afterwards.
    var out_seq: c_int = seq_len;
    if ((project == .last_row or project == .mixed_last_row) and seq_len > 1) {
        const hs = mlx.getShape(h);
        var h_last = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(h_last);
        try mlx.check(mlx.mlx_slice(
            &h_last,
            h,
            &[_]c_int{ 0, seq_len - 1, 0 },
            3,
            &[_]c_int{ hs[0], seq_len, hs[2] },
            3,
            &[_]c_int{ 1, 1, 1 },
            3,
            self.s,
        ));
        _ = mlx.mlx_array_free(h);
        h = h_last;
        out_seq = 1;
    }

    const mix = try self.hcRead(h, &m.mixer, batch, out_seq);
    if (mix.inj.ctx != null) _ = mlx.mlx_array_free(mix.inj);
    // Rerank draft: the mixer output IS the answer. Skipping
    // `lmHeadProject` here is the whole point — the 248320-wide 8-bit
    // projection is 675 MB, more than every other read in a draft step put
    // together, and a greedy draft only ever needed its argmax.
    if (project == .mixed_last_row) {
        return .{ .logits = .{ .ctx = null }, .stream = h, .mixed = mix.mixed };
    }
    defer _ = mlx.mlx_array_free(mix.mixed);
    const logits = try self.lmHeadProject(mix.mixed);
    return .{ .logits = logits, .stream = h };
}

/// Build the qwen4_exp head's coarse rerank head. ONE-SHOT (`rerank_tried`
/// caches the refusal too) and infallible — false keeps the full-vocab
/// draft projection.
///
/// Called at LOAD (`scheduler.doLoadOnInferenceThread`, the site both the
/// boot load and the `/v1/load` cold load route through), the sidecar
/// arm's `maybeBuildDraftRerank`-at-bind twin: a full `requantizeRows` of
/// the trunk lm_head plus a synchronous eval of its ~240 MB is a LOAD
/// cost, and inside the first request's draft chain it was first-token
/// latency on the inference thread with the stream drained mid-round.
pub fn qwen4BuildDraftRerank(self: *Trunk) bool {
    const m = &(self.qwen4_mtp orelse return false);
    if (m.rerank_tried) return m.rerank != null;
    m.rerank_tried = true;
    const io = std.Io.Threaded.global_single_threaded.io();
    const t0 = std.Io.Timestamp.now(io, .awake);
    // The ONE env read: the built head carries its width from here on.
    const rc = mtp_mod.buildRerankCoarse(self.s, self, mtp_mod.DRAFT_HEAD_BITS) orelse return false;
    m.rerank = rc;
    const bytes = mtp_mod.rerankCoarseBytes(rc.rows, @intCast(self.config.hidden_size), rc.bits);
    const ms: u64 = @intCast(@divTrunc(t0.untilNow(io, .awake).nanoseconds, std.time.ns_per_ms));
    log.info(
        "[qwen4] MTP draft rerank: coarse lm_head {d}-bit ({d} MB, {d} ms) + exact top-32 rescoring\n",
        .{ rc.bits, bytes / (1024 * 1024), ms },
    );
    return true;
}

/// Is the qwen4_exp head's draft rerank available? A pure read once the
/// load-time build has run; the build is only reached here when it did
/// NOT (an offline path, a directly-constructed Trunk), and it is
/// one-shot either way.
pub fn qwen4DraftRerankReady(self: *Trunk) bool {
    const m = &(self.qwen4_mtp orelse return false);
    if (m.rerank_tried) return m.rerank != null;
    return self.qwen4BuildDraftRerank();
}

/// One greedy draft id from the mixer output `x` `[B,1,H]`. Shares the
/// sidecar head's scheme verbatim; a coarse-head absence or a shortlist
/// failure falls through to the full trunk-head readout, which is exactly
/// what this path replaced.
pub fn qwen4DraftSelect(self: *Trunk, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !mlx.mlx_array {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    if (try mtp_mod.rerankSelect(self.s, self, &m.rerank, &m.rerank_logged, x, suppress_mask)) |tok| return tok;
    return mtp_mod.fullReadoutArgmax(self.s, self, x, suppress_mask);
}

/// The exact re-scored top-32 shortlist for a SAMPLED draft. Null = the
/// coarse head is unavailable and the caller falls back to a greedy draft.
pub fn qwen4DraftShortlist(self: *Trunk, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !?mtp_mod.Shortlist {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    return mtp_mod.rerankShortlist(self.s, self, &m.rerank, &m.rerank_logged, x, suppress_mask);
}

/// Per-row exact re-scored shortlists for a batched SAMPLED draft step.
/// False = no coarse head; the caller drafts greedily instead.
pub fn qwen4DraftShortlistsBatched(
    self: *Trunk,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
    out: []mtp_mod.Shortlist,
) !bool {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    return mtp_mod.rerankShortlistsBatched(self.s, self, &m.rerank, &m.rerank_logged, x, suppress_mask, out);
}

pub fn qwen4DraftSelectBatched(self: *Trunk, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !mlx.mlx_array {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    if (try mtp_mod.rerankSelectBatched(self.s, self, &m.rerank, &m.rerank_logged, x, suppress_mask)) |tok| return tok;
    return mtp_mod.fullReadoutArgmax(self.s, self, x, suppress_mask);
}

/// Test hook: layer-0 intermediates of the qwen4 forward (fixture bisect).
pub const Qwen4Trace = struct {
    mixed_attn: mlx.mlx_array = .{ .ctx = null },
    inj_attn: mlx.mlx_array = .{ .ctx = null },
    attn_out: mlx.mlx_array = .{ .ctx = null },
    mixed_mlp: mlx.mlx_array = .{ .ctx = null },
    inj_mlp: mlx.mlx_array = .{ .ctx = null },
    mlp_out: mlx.mlx_array = .{ .ctx = null },
    ple_emb: mlx.mlx_array = .{ .ctx = null },
    ple_out: mlx.mlx_array = .{ .ctx = null },
    qsa_mask: mlx.mlx_array = .{ .ctx = null },
    attn3_out: mlx.mlx_array = .{ .ctx = null },
    pub fn set(dst: *mlx.mlx_array, src: mlx.mlx_array) void {
        if (dst.ctx != null) _ = mlx.mlx_array_free(dst.*);
        dst.* = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(dst, src);
    }
    pub fn deinit(self: *Qwen4Trace) void {
        inline for (.{ &self.mixed_attn, &self.inj_attn, &self.attn_out, &self.mixed_mlp, &self.inj_mlp, &self.mlp_out, &self.ple_emb, &self.ple_out, &self.qsa_mask, &self.attn3_out }) |f| {
            if (f.ctx != null) _ = mlx.mlx_array_free(f.*);
        }
    }
};

// ── tests ──

test "qwen4 MTP head: two explicit states with different origins and stash lengths round-trip" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    var xfm: Trunk = undefined;
    xfm.s = mlx.gpuStream();
    xfm.allocator = t.allocator;
    xfm.qwen4_mtp_owner = null;
    const head_cache = try KVCache.init(t.allocator, 1);
    var head: Qwen4Mtp = undefined;
    head.cache = head_cache;
    head.entry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    head.qsa_marks = .{};
    head.seq_offset = 0;
    head.pos_base = 0;
    xfm.qwen4_mtp = head;
    const m = &xfm.qwen4_mtp.?;
    defer m.cache.deinit();

    const ca = try KVCache.init(t.allocator, 1);
    var a: Trunk.Qwen4MtpState = .{ .cache = ca, .entry = m.entry, .seq_offset = 7, .pos_base = 3 };
    a.cache.step = 7;
    defer a.deinit();
    const cb = try KVCache.init(t.allocator, 1);
    var b: Trunk.Qwen4MtpState = .{ .cache = cb, .entry = m.entry, .seq_offset = 2, .pos_base = 11 };
    b.cache.step = 2;
    defer b.deinit();

    const live_a = Trunk.qwen4MtpLiveState(&a);
    const live_b = Trunk.qwen4MtpLiveState(&b);
    try t.expect(live_a.cache == &a.cache);
    try t.expect(live_b.cache == &b.cache);
    try t.expect(xfm.qwen4_mtp_owner == null);
    try t.expectEqual(@as(usize, 0), m.seq_offset);

    const sa = a.meta();
    const sb = b.meta();
    try t.expectEqual(@as(usize, 7), sa.stash_len);
    try t.expectEqual(@as(c_int, 3), sa.origin);
    try t.expectEqual(@as(usize, 2), sb.stash_len);
    try t.expectEqual(@as(c_int, 11), sb.origin);

    live_a.seq_offset.* = 99;
    live_b.pos_base.* = -1;
    try t.expectEqual(@as(usize, 0), m.seq_offset);
    try t.expectEqual(@as(c_int, 0), m.pos_base);

    a.restoreMeta(sa);
    b.restoreMeta(sb);
    try t.expectEqual(sa, a.meta());
    try t.expectEqual(sb, b.meta());
    try t.expect(xfm.qwen4_mtp_owner == null);
}

test "qwen4 MTP head: two requests keep their own state through activate/release" {
    // Bar: after A appends and B activates, A's rows are not on the module; A re-activating
    // sees them again; releasing hands the module its own copy back.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    var xfm: Trunk = undefined;
    xfm.s = mlx.gpuStream();
    xfm.allocator = t.allocator;
    xfm.qwen4_mtp_owner = null;
    const head_cache = try KVCache.init(t.allocator, 1);
    var head: Qwen4Mtp = undefined;
    head.cache = head_cache;
    head.entry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    head.qsa_marks = .{};
    head.seq_offset = 0;
    head.pos_base = 0;
    xfm.qwen4_mtp = head;
    const m = &xfm.qwen4_mtp.?;
    defer m.cache.deinit();

    const ca = try KVCache.init(t.allocator, 1);
    var a: Trunk.Qwen4MtpState = .{ .cache = ca, .entry = m.entry };
    defer a.deinit();
    const cb = try KVCache.init(t.allocator, 1);
    var b: Trunk.Qwen4MtpState = .{ .cache = cb, .entry = m.entry };
    defer b.deinit();

    xfm.qwen4MtpActivate(&a);
    m.seq_offset = 7;
    m.pos_base = 3;
    xfm.qwen4MtpActivate(&a); // idempotent
    try t.expectEqual(@as(usize, 7), m.seq_offset);
    xfm.qwen4MtpActivate(&b);
    try t.expectEqual(@as(usize, 0), m.seq_offset);
    try t.expectEqual(@as(c_int, -1), m.pos_base);
    m.seq_offset = 2;
    xfm.qwen4MtpActivate(&a);
    try t.expectEqual(@as(usize, 7), m.seq_offset);
    try t.expectEqual(@as(c_int, 3), m.pos_base);
    try t.expectEqual(@as(usize, 2), b.seq_offset);
    xfm.qwen4MtpRelease(&a);
    try t.expectEqual(@as(usize, 0), m.seq_offset);
    try t.expectEqual(@as(usize, 7), a.seq_offset);
    try t.expect(xfm.qwen4_mtp_owner == null);
}

test "qwen4 MTP head: N=1 FORCE_BATCHED draft ids and logits mlx_equal solo at depths 1/2/4" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const allocator = t.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    mtp_head_force_batched_override = true;
    defer mtp_head_force_batched_override = null;

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Trunk.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try t.expect(xfm.qwen4_mtp != null);
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();

    var host_ids: [16]i32 = undefined;
    for (&host_ids, 0..) |*v, i| v.* = @intCast(1 + i);
    const slot = try Qwen4TestSlot.init(allocator, config.num_hidden_layers);
    defer slot.deinit(allocator);
    var stream = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stream);
    slot.ctx.capture_stream_all = &stream;
    _ = mlx.mlx_array_free(try slot.forward(&xfm, host_ids[0..8]));

    const H = struct {
        fn row(st: mlx.mlx_stream, arr: mlx.mlx_array, at: usize) !mlx.mlx_array {
            const sh = mlx.getShape(arr);
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&out, arr, &[_]c_int{ 0, @intCast(at), 0 }, 3, &[_]c_int{ sh[0], @intCast(at + 1), sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, st));
            return out;
        }
        fn tok(v: i32) mlx.mlx_array {
            const sh = [_]c_int{ 1, 1 };
            var x: i32 = v;
            return mlx.mlx_array_new_data(@ptrCast(&x), &sh, 2, .int32);
        }
        fn argmax(alloc: std.mem.Allocator, logits: mlx.mlx_array, st: mlx.mlx_stream) !i32 {
            const f = try qwen4ReadF32(alloc, logits, st);
            defer alloc.free(f);
            var best: f32 = -std.math.inf(f32);
            var id: i32 = 0;
            for (f, 0..) |x, i| if (x > best) {
                best = x;
                id = @intCast(i);
            };
            return id;
        }
    };
    const h0 = try H.row(s, stream, 0);
    defer _ = mlx.mlx_array_free(h0);
    var tok_s: i32 = host_ids[1];
    var tok_b: i32 = host_ids[1];

    const depths = [_]u32{ 1, 2, 4 };
    for (depths) |depth| {
        var solo = try xfm.qwen4MtpStateNew();
        defer solo.deinit();
        var bat = try xfm.qwen4MtpStateNew();
        defer bat.deinit();
        var h_s = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&h_s, h0));
        defer _ = mlx.mlx_array_free(h_s);
        var h_b = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&h_b, h0));
        defer _ = mlx.mlx_array_free(h_b);
        tok_s = host_ids[1];
        tok_b = host_ids[1];
        var d: u32 = 0;
        while (d < depth) : (d += 1) {
            const po_s: c_int = if (solo.seq_offset == 0) 1 else solo.pos_base + @as(c_int, @intCast(solo.seq_offset));
            const po_b: c_int = if (bat.seq_offset == 0) 1 else bat.pos_base + @as(c_int, @intCast(bat.seq_offset));
            const id_s = H.tok(tok_s);
            defer _ = mlx.mlx_array_free(id_s);
            const id_b = H.tok(tok_b);
            defer _ = mlx.mlx_array_free(id_b);
            const os = try xfm.qwen4MtpForwardOn(&solo, h_s, id_s, po_s, null, .last_row);
            defer _ = mlx.mlx_array_free(os.logits);
            defer _ = mlx.mlx_array_free(os.stream);
            qwen4_mtp_head_graphs = 0;
            const brows = [_]Trunk.Qwen4MtpBatchRow{
                .{ .st = &bat, .stream = h_b, .token_ids = id_b, .pos_offset = po_b },
            };
            const bout = try xfm.qwen4MtpForwardBatched(&brows, .last_row, depth);
            defer {
                for (bout) |o| {
                    if (o.logits.ctx != null) _ = mlx.mlx_array_free(o.logits);
                    if (o.stream.ctx != null) _ = mlx.mlx_array_free(o.stream);
                    if (o.mixed.ctx != null) _ = mlx.mlx_array_free(o.mixed);
                }
                allocator.free(bout);
            }
            try t.expectEqual(@as(usize, 1), qwen4_mtp_head_graphs);
            try t.expect(try qsaArraysAllEqual(os.logits, bout[0].logits, s));
            const id_solo = try H.argmax(allocator, os.logits, s);
            const id_bat = try H.argmax(allocator, bout[0].logits, s);
            try t.expectEqual(id_solo, id_bat);
            std.debug.print("[mtp mlx_equal N=1] depth={d} step={d} id={d} graphs=1\n", .{ depth, d, id_solo });
            tok_s = id_solo;
            tok_b = id_bat;
            _ = mlx.mlx_array_free(h_s);
            _ = mlx.mlx_array_free(h_b);
            h_s = mlx.mlx_array_new();
            h_b = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&h_s, os.stream));
            try mlx.check(mlx.mlx_array_set(&h_b, bout[0].stream));
        }
    }
}

/// A loaded qwen4_exp pack for the head tests; heap-held because the Trunk keeps a pointer to the weights.
const Qwen4HeadRig = struct {
    config: model_mod.ModelConfig,
    weights: model_mod.Weights,
    xfm: Trunk,

    fn load(alloc: std.mem.Allocator) !*Qwen4HeadRig {
        const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
        if (mlx.noGpuBackend()) return error.SkipZigTest;
        const io = std.Io.Threaded.global_single_threaded.io();
        const rig = try alloc.create(Qwen4HeadRig);
        errdefer alloc.destroy(rig);
        rig.config = try model_mod.parseConfig(io, alloc, std.mem.span(model_dir));
        rig.weights = try model_mod.loadWeights(io, alloc, std.mem.span(model_dir));
        model_mod.resolveWeightPrefix(&rig.config, &rig.weights);
        rig.xfm = try Trunk.init(io, alloc, rig.config, &rig.weights);
        rig.xfm.compileQwen4Hc();
        rig.xfm.compileGdnGate();
        rig.xfm.compileMoeRouting();
        return rig;
    }

    fn deinit(rig: *Qwen4HeadRig, alloc: std.mem.Allocator) void {
        rig.xfm.deinit();
        rig.weights.deinit();
        if (rig.config.ngram_table_path) |p| alloc.free(p);
        alloc.destroy(rig);
    }

    /// `n` seeded pre-mixer stream rows `[1,n,hc*H]`.
    fn stream(rig: *Qwen4HeadRig, seed: u64, n: c_int) !mlx.mlx_array {
        const width: c_int = @intCast(rig.config.hc_count * rig.config.hidden_size);
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, seed));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&out, &[_]c_int{ 1, n, width }, 3, .bfloat16, 0, 1, key, rig.xfm.s));
        return out;
    }

    fn ids(rig: *Qwen4HeadRig, first: c_int, n: c_int) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_arange(&out, @floatFromInt(first), @floatFromInt(first + n), 1, .int32, rig.xfm.s));
        return out;
    }

    /// Committed history of `n` rows, appended in prefill-sized chunks.
    fn history(rig: *Qwen4HeadRig, st: *Trunk.Qwen4MtpState, n: c_int) !void {
        var done: c_int = 0;
        while (done < n) {
            const chunk = @min(n - done, 2048);
            const h = try rig.stream(@intCast(done + 1), chunk);
            defer _ = mlx.mlx_array_free(h);
            const id = try rig.ids(100 + done, chunk);
            defer _ = mlx.mlx_array_free(id);
            const out = try rig.xfm.qwen4MtpForwardOn(st, h, id, st.pos_base + @as(c_int, @intCast(st.seq_offset)) + @intFromBool(st.seq_offset == 0), null, .none);
            try mlx.check(mlx.mlx_array_eval(out.stream));
            _ = mlx.mlx_array_free(out.stream);
            done += chunk;
        }
    }

    /// One round's padded history: `1+m` rows at the committed end, rows past `a` built from `dead_seed`.
    fn paddedHistory(rig: *Qwen4HeadRig, st: *Trunk.Qwen4MtpState, m: c_int, a: c_int, dead_seed: u64) !void {
        const live = try rig.stream(77, 1 + a);
        defer _ = mlx.mlx_array_free(live);
        const dead = try rig.stream(dead_seed, m - a);
        defer _ = mlx.mlx_array_free(dead);
        var h = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h);
        const hv = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ live, dead }, 2);
        defer _ = mlx.mlx_vector_array_free(hv);
        try mlx.check(mlx.mlx_concatenate_axis(&h, hv, 1, rig.xfm.s));
        const live_ids = try rig.ids(5000, 1 + a);
        defer _ = mlx.mlx_array_free(live_ids);
        const dead_ids = try rig.ids(9000 + @as(c_int, @intCast(dead_seed)), m - a);
        defer _ = mlx.mlx_array_free(dead_ids);
        var id = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(id);
        const iv = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ live_ids, dead_ids }, 2);
        defer _ = mlx.mlx_vector_array_free(iv);
        try mlx.check(mlx.mlx_concatenate_axis(&id, iv, 0, rig.xfm.s));
        const out = try rig.xfm.qwen4MtpForwardOn(st, h, id, st.pos_base + @as(c_int, @intCast(st.seq_offset)), null, .none);
        _ = mlx.mlx_array_free(out.stream);
    }

    /// `steps` placed draft rows chained like `mtpChainBuild`: each feeds the next its stream and argmax.
    /// Returns every step's logits, concatenated `[steps, V]`.
    fn placedDrafts(rig: *Qwen4HeadRig, alloc: std.mem.Allocator, st: *Trunk.Qwen4MtpState, place: *const Trunk.HeadPlace, steps: usize) !mlx.mlx_array {
        const s = rig.xfm.s;
        var h = try rig.stream(55, 1);
        defer _ = mlx.mlx_array_free(h);
        var tok = mlx.mlx_array_new_int(4242);
        defer _ = mlx.mlx_array_free(tok);
        const parts = try alloc.alloc(mlx.mlx_array, steps);
        defer alloc.free(parts);
        var built: usize = 0;
        defer for (parts[0..built]) |p| {
            _ = mlx.mlx_array_free(p);
        };
        for (parts) |*p| {
            var id = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(id);
            try mlx.check(mlx.mlx_reshape(&id, tok, &[_]c_int{1}, 1, s));
            const out = try rig.xfm.qwen4MtpForwardPlacedOn(st, h, id, place, .last_row);
            _ = mlx.mlx_array_free(h);
            h = out.stream;
            p.* = out.logits;
            built += 1;
            _ = mlx.mlx_array_free(tok);
            tok = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_argmax_axis(&tok, out.logits, -1, false, s));
        }
        const vec = mlx.mlx_vector_array_new_data(parts.ptr, parts.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var all = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&all, vec, 1, s));
        return all;
    }
};

test "qwen4 MTP head: placed draft rows ignore dead history rows, below and past the QSA budget and the gather floor (QWEN4_TEST_MODEL)" {
    // Bar: changing the dead rows' ids and streams leaves every placed draft's logits bit-equal; a rerun is bit-equal.
    const t = std.testing;
    const allocator = t.allocator;
    const rig = try Qwen4HeadRig.load(allocator);
    defer rig.deinit(allocator);
    const s = rig.xfm.s;
    const m: c_int = 5;
    for ([_]c_int{ 40, 2100, 9000 }) |base| {
        for ([_]c_int{ 0, 2, 4, 5 }) |a| {
            var outs: [3]mlx.mlx_array = undefined;
            var n_out: usize = 0;
            defer for (outs[0..n_out]) |o| {
                _ = mlx.mlx_array_free(o);
            };
            for ([_]u64{ 11, 12, 11 }) |dead_seed| {
                var st = try rig.xfm.qwen4MtpStateNew();
                defer st.deinit();
                try rig.history(&st, base);
                const hist0: c_int = @intCast(st.seq_offset);
                try rig.paddedHistory(&st, m, a, dead_seed);
                var place: Trunk.HeadPlace = .{ .live_end = mlx.mlx_array_new_int(hist0 + 1 + a), .dead_end = hist0 + 1 + m };
                defer _ = mlx.mlx_array_free(place.live_end);
                outs[n_out] = try rig.placedDrafts(allocator, &st, &place, 3);
                try mlx.check(mlx.mlx_array_eval(outs[n_out]));
                n_out += 1;
            }
            const dead_blind = try qsaArraysAllEqual(outs[0], outs[1], s);
            const rerun = try qsaArraysAllEqual(outs[0], outs[2], s);
            std.debug.print("[mtp placed] base={d} a={d} m={d} dead_blind={} rerun_equal={}\n", .{ base, a, m, dead_blind, rerun });
            try t.expect(dead_blind);
            try t.expect(rerun);
        }
    }
}

test "qwen4 MTP head: truncating to the live end after placed drafts leaves the committed head unchanged (QWEN4_TEST_MODEL)" {
    // Bar: a head that ran placed drafts and one that did not are bit-equal after the truncate, below and past the QSA budget.
    const t = std.testing;
    const allocator = t.allocator;
    const rig = try Qwen4HeadRig.load(allocator);
    defer rig.deinit(allocator);
    const s = rig.xfm.s;
    const m: c_int = 5;
    const a: c_int = 2;
    for ([_]c_int{ 40, 2100 }) |base| {
        var next: [2]mlx.mlx_array = undefined;
        var n_next: usize = 0;
        defer for (next[0..n_next]) |o| {
            _ = mlx.mlx_array_free(o);
        };
        var lens: [2]usize = undefined;
        for ([_]usize{ 3, 0 }, 0..) |steps, arm| {
            var st = try rig.xfm.qwen4MtpStateNew();
            defer st.deinit();
            try rig.history(&st, base);
            const hist0: c_int = @intCast(st.seq_offset);
            try rig.paddedHistory(&st, m, a, 11);
            var place: Trunk.HeadPlace = .{ .live_end = mlx.mlx_array_new_int(hist0 + 1 + a), .dead_end = hist0 + 1 + m };
            defer _ = mlx.mlx_array_free(place.live_end);
            if (steps > 0) {
                const d = try rig.placedDrafts(allocator, &st, &place, steps);
                try mlx.check(mlx.mlx_array_eval(d));
                _ = mlx.mlx_array_free(d);
            }
            try rig.xfm.qwen4MtpTruncateOn(&st, @intCast(hist0 + 1 + a));
            lens[arm] = st.seq_offset;
            // The next round's history and draft read the committed rows, and nothing past them.
            try rig.paddedHistory(&st, m, m, 13);
            const end: c_int = @intCast(st.seq_offset);
            var place2: Trunk.HeadPlace = .{ .live_end = mlx.mlx_array_new_int(end), .dead_end = end };
            defer _ = mlx.mlx_array_free(place2.live_end);
            next[n_next] = try rig.placedDrafts(allocator, &st, &place2, 2);
            try mlx.check(mlx.mlx_array_eval(next[n_next]));
            n_next += 1;
        }
        const same = try qsaArraysAllEqual(next[0], next[1], s);
        std.debug.print("[mtp truncate] base={d} len_after={d}/{d} next_round_equal={}\n", .{ base, lens[0], lens[1], same });
        try t.expectEqual(lens[1], lens[0]);
        try t.expect(same);
    }
}

test "MTP head row widths pick the verify lane, the MoE arm and the QSA score kernel" {
    const t = std.testing;
    try t.expectEqual(VqmmLane.none, vqmmLaneFor(1, 5120, 248320, false, 8));
    try t.expectEqual(VqmmLane.msg, vqmmLaneFor(2, 5120, 248320, false, 8));
    try t.expectEqual(VqmmLane.msg, vqmmLaneFor(4, 5120, 248320, false, 8));
    try t.expectEqual(VqmmLane.none, vqmmLaneFor(1, 5120, 17408, false, 8));
    try t.expectEqual(VqmmLane.splitk, vqmmLaneFor(2, 5120, 17408, false, 8));
    try t.expectEqual(VqmmLane.splitk, vqmmLaneFor(4, 5120, 17408, false, 8));
    try t.expectEqual(VqmmLane.none, vqmmLaneFor(8, 5120, 17408, false, 8));

    gather_qmv_rows_vars.moe_rows_fused_override = true;
    defer gather_qmv_rows_vars.moe_rows_fused_override = null;
    try t.expectEqual(MoeDecodeDispatchArm.gather_qmv, moeDecodeDispatchArm(1, 1, 10, false, false));
    try t.expectEqual(MoeDecodeDispatchArm.rows, moeDecodeDispatchArm(2, 1, 10, false, false));
    try t.expectEqual(MoeDecodeDispatchArm.rows, moeDecodeDispatchArm(4, 1, 10, false, false));
    gather_qmv_rows_vars.moe_rows_fused_override = false;
    try t.expectEqual(MoeDecodeDispatchArm.sorted, moeDecodeDispatchArm(2, 1, 10, false, false));
    try t.expectEqual(MoeDecodeDispatchArm.sorted, moeDecodeDispatchArm(2, 2, 10, false, false));
    try t.expectEqual(MoeDecodeDispatchArm.sorted, moeDecodeDispatchArm(4, 2, 10, false, false));

    try t.expect(!qsaScoreFusedActiveFor(2, 4, 128));
    try t.expect(!qsaScoreFusedActiveFor(4, 4, 128));
    try t.expectEqual(qsaScoreFusedActive(), qsaScoreFusedActiveFor(1, 4, 128));
}

test "MTP head: N=2/4 mlx_equal drafts/logits vs solo" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const allocator = t.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Trunk.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try t.expect(xfm.qwen4_mtp != null);
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();

    var host_ids: [32]i32 = undefined;
    for (&host_ids, 0..) |*v, i| v.* = @intCast(1 + i);
    const slot_a = try Qwen4TestSlot.init(allocator, config.num_hidden_layers);
    defer slot_a.deinit(allocator);
    const slot_b = try Qwen4TestSlot.init(allocator, config.num_hidden_layers);
    defer slot_b.deinit(allocator);
    var stream_a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stream_a);
    var stream_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stream_b);
    slot_a.ctx.capture_stream_all = &stream_a;
    slot_b.ctx.capture_stream_all = &stream_b;
    _ = mlx.mlx_array_free(try slot_a.forward(&xfm, host_ids[0..8]));
    _ = mlx.mlx_array_free(try slot_b.forward(&xfm, host_ids[3..8]));

    const H = struct {
        fn row(st: mlx.mlx_stream, arr: mlx.mlx_array, at: usize) !mlx.mlx_array {
            return span(st, arr, at, 1);
        }
        fn span(st: mlx.mlx_stream, arr: mlx.mlx_array, at: usize, count: usize) !mlx.mlx_array {
            const sh = mlx.getShape(arr);
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&out, arr, &[_]c_int{ 0, @intCast(at), 0 }, 3, &[_]c_int{ sh[0], @intCast(at + count), sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, st));
            return out;
        }
        fn tok(v: i32) mlx.mlx_array {
            const sh = [_]c_int{ 1, 1 };
            var x: i32 = v;
            return mlx.mlx_array_new_data(@ptrCast(&x), &sh, 2, .int32);
        }
        fn top2(alloc: std.mem.Allocator, logits: mlx.mlx_array, st: mlx.mlx_stream) !struct { id: i32 } {
            const f = try qwen4ReadF32(alloc, logits, st);
            defer alloc.free(f);
            var b0: f32 = -std.math.inf(f32);
            var id: i32 = 0;
            for (f, 0..) |x, i| {
                if (x > b0) {
                    b0 = x;
                    id = @intCast(i);
                }
            }
            return .{ .id = id };
        }
        fn pos(st: *Trunk.Qwen4MtpState) c_int {
            return if (st.seq_offset == 0) st.pos_base else st.pos_base + @as(c_int, @intCast(st.seq_offset));
        }
    };
    const ha = try H.row(s, stream_a, 0);
    defer _ = mlx.mlx_array_free(ha);
    const hb = try H.row(s, stream_b, 0);
    defer _ = mlx.mlx_array_free(hb);
    {
        var warm = try xfm.qwen4MtpStateNew();
        defer warm.deinit();
        const idw = H.tok(host_ids[1]);
        defer _ = mlx.mlx_array_free(idw);
        const o = try xfm.qwen4MtpForwardOn(&warm, ha, idw, 1, null, .last_row);
        try mlx.check(mlx.mlx_array_eval(o.logits));
        if (o.logits.ctx != null) _ = mlx.mlx_array_free(o.logits);
        _ = mlx.mlx_array_free(o.stream);
        try mlx.check(mlx.mlx_synchronize(s));
    }

    const ns = [_]usize{ 2, 4 };
    const depths = [_]u32{ 1, 2, 4 };
    for ([_]bool{ false, true }) |with_stash| {
        for (ns) |N| {
            const n_use: usize = N;
            for (depths) |depth| {
                var solo: [4]Trunk.Qwen4MtpState = undefined;
                var bat: [4]Trunk.Qwen4MtpState = undefined;
                for (0..n_use) |k| {
                    solo[k] = try xfm.qwen4MtpStateNew();
                    bat[k] = try xfm.qwen4MtpStateNew();
                    solo[k].pos_base = @intCast(1 + (k % 2) * 7);
                    bat[k].pos_base = solo[k].pos_base;
                }
                defer for (0..n_use) |k| {
                    solo[k].deinit();
                    bat[k].deinit();
                };
                var hs: [4]mlx.mlx_array = undefined;
                var hb2: [4]mlx.mlx_array = undefined;
                const srcs = [_]mlx.mlx_array{ ha, hb, ha, hb };
                const tok0 = [_]i32{ host_ids[1], host_ids[4], host_ids[2], host_ids[5] };
                for (0..n_use) |k| {
                    hs[k] = mlx.mlx_array_new();
                    const initial = if (with_stash) try H.span(s, if (k % 2 == 0) stream_a else stream_b, 0, 2 + k % 2) else try H.row(s, srcs[k], 0);
                    defer _ = mlx.mlx_array_free(initial);
                    try mlx.check(mlx.mlx_array_set(&hs[k], initial));
                    hb2[k] = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_array_set(&hb2[k], hs[k]));
                }
                defer for (hs[0..n_use], hb2[0..n_use]) |a, c| {
                    _ = mlx.mlx_array_free(a);
                    _ = mlx.mlx_array_free(c);
                };
                var tok_s: [4]i32 = tok0;
                var tok_b: [4]i32 = tok0;
                var steps_s: [4]u32 = @splat(0);
                var steps_b: [4]u32 = @splat(0);
                var d: u32 = 0;
                while (d < depth) : (d += 1) {
                    var ids_s: [4]mlx.mlx_array = undefined;
                    var ids_b: [4]mlx.mlx_array = undefined;
                    for (0..n_use) |k| {
                        if (with_stash and d == 0) {
                            const count = 2 + k % 2;
                            const shape = [_]c_int{ 1, @intCast(count) };
                            ids_s[k] = mlx.mlx_array_new_data(host_ids[k + 1 ..][0..count].ptr, &shape, 2, .int32);
                            ids_b[k] = mlx.mlx_array_new_data(host_ids[k + 1 ..][0..count].ptr, &shape, 2, .int32);
                        } else {
                            ids_s[k] = H.tok(tok_s[k]);
                            ids_b[k] = H.tok(tok_b[k]);
                        }
                    }
                    defer for (0..n_use) |k| {
                        _ = mlx.mlx_array_free(ids_s[k]);
                        _ = mlx.mlx_array_free(ids_b[k]);
                    };
                    var solo_out: [4]Trunk.Qwen4MtpOut = undefined;
                    for (0..n_use) |k| {
                        solo_out[k] = try xfm.qwen4MtpForwardOn(&solo[k], hs[k], ids_s[k], H.pos(&solo[k]), null, .last_row);
                    }
                    defer for (0..n_use) |k| {
                        if (solo_out[k].logits.ctx != null) _ = mlx.mlx_array_free(solo_out[k].logits);
                        _ = mlx.mlx_array_free(solo_out[k].stream);
                    };
                    var brows_buf: [4]Trunk.Qwen4MtpBatchRow = undefined;
                    for (0..n_use) |k| {
                        brows_buf[k] = .{ .st = &bat[k], .stream = hb2[k], .token_ids = ids_b[k], .pos_offset = H.pos(&bat[k]) };
                    }
                    const bout = try xfm.qwen4MtpForwardBatched(brows_buf[0..n_use], .last_row, depth);
                    defer {
                        for (bout) |o| {
                            if (o.logits.ctx != null) _ = mlx.mlx_array_free(o.logits);
                            if (o.stream.ctx != null) _ = mlx.mlx_array_free(o.stream);
                            if (o.mixed.ctx != null) _ = mlx.mlx_array_free(o.mixed);
                        }
                        allocator.free(bout);
                    }
                    for (0..n_use) |k| {
                        const eq_logits = try qsaArraysAllEqual(solo_out[k].logits, bout[k].logits, s);
                        if (!eq_logits) {
                            const eq_stream = try qsaArraysAllEqual(solo_out[k].stream, bout[k].stream, s);
                            var row_mix = try xfm.hcRead(bout[k].stream, &xfm.qwen4_mtp.?.mixer, 1, 1);
                            defer row_mix.deinit();
                            const row_logits = try xfm.lmHeadProject(row_mix.mixed);
                            defer _ = mlx.mlx_array_free(row_logits);
                            const eq_readout = try qsaArraysAllEqual(row_logits, bout[k].logits, s);
                            std.debug.print("[mtp mlx_equal] N={d} depth={d} step={d} row={d} logits=NE stream={} readout={}\n", .{ N, depth, d, k, eq_stream, eq_readout });
                        }
                        try t.expect(eq_logits);
                        const ts = try H.top2(allocator, solo_out[k].logits, s);
                        const tb = try H.top2(allocator, bout[k].logits, s);
                        try t.expectEqual(ts.id, tb.id);
                        tok_s[k] = ts.id;
                        tok_b[k] = tb.id;
                        steps_s[k] += 1;
                        steps_b[k] += 1;
                        _ = mlx.mlx_array_free(hs[k]);
                        _ = mlx.mlx_array_free(hb2[k]);
                        hs[k] = mlx.mlx_array_new();
                        hb2[k] = mlx.mlx_array_new();
                        try mlx.check(mlx.mlx_array_set(&hs[k], solo_out[k].stream));
                        try mlx.check(mlx.mlx_array_set(&hb2[k], bout[k].stream));
                    }
                }
                std.debug.print("[mtp matched draft steps] stash={} N={d} depth={d} solo={any} batched={any}\n", .{ with_stash, N, depth, steps_s[0..n_use], steps_b[0..n_use] });
            }
        }
    }
}

test "qwen4 MTP head: a clamp below the raw ring is named, not an emptied history" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const s = mlx.gpuStream();
    var xfm: Trunk = undefined;
    xfm.s = s;
    xfm.allocator = t.allocator;

    const hd: c_int = 16;
    const ratio: c_int = 4;
    // The live warm turn: the head commits at L, the trunk restores at its checkpoint
    // (SSM_SNAPSHOT_BACKOFF below the prompt end, plus the tokens this turn generated), and
    // `qwen4MtpAdopt` clamps the head to that. The gap is wider than the ring and mid-block.
    const L: c_int = 200;
    const want: c_int = 43;
    try t.expect(L - want > QSA_RING_ROWS and @mod(want, ratio) != 0);

    var prng = std.Random.DefaultPrng.init(0x4D7F_1105);
    const keys = try attn256RandBf16(prng.random(), &[_]c_int{ 1, L, hd }, s);
    defer _ = mlx.mlx_array_free(keys);

    var live: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = ratio };
    defer ssmFreeQsaState(&live);
    try qsaTestAppendPool(&xfm, &live, keys, 0, s);

    // What `commitWithState` stores for the head, and what `qwen4MtpAdopt` seeds it back from.
    var head_aux = ssmSnapshot(&live);
    defer ssmSnapshotDeinit(&head_aux);

    // Built before the assignment: the scan guard reads `.cache = try` as the reinit defect.
    const head_cache = try KVCache.init(t.allocator, 1);
    var head: Qwen4Mtp = undefined;
    head.cache = head_cache;
    head.entry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    head.qsa_marks = .{};
    head.seq_offset = 0;
    head.pos_base = 0;
    xfm.qwen4_mtp = head;
    const m = &xfm.qwen4_mtp.?;
    defer {
        ssmFreeQsaState(&m.entry);
        m.qsa_marks.deinit();
        m.cache.deinit();
    }

    try ssmRestore(&m.entry, &head_aux);
    m.seq_offset = @intCast(L);
    m.cache.step = @intCast(L);
    try t.expectError(error.MtpHeadQsaRingGap, xfm.qwen4MtpTruncate(@intCast(want)));

    // The guard is not the ring: a clamp the ring CAN serve still trims, and the append after
    // it re-pools from that leftover exactly as a full bank would.
    const near: c_int = L - 5;
    try t.expect(@mod(near, ratio) != 0);
    try ssmRestore(&m.entry, &head_aux);
    m.seq_offset = @intCast(L);
    m.cache.step = @intCast(L);
    try xfm.qwen4MtpTruncate(@intCast(near));
    try t.expectEqual(near, m.entry.qsa_hist_rows);
    {
        var again = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(again);
        try mlx.check(mlx.mlx_slice(&again, keys, &[_]c_int{ 0, near, 0 }, 3, &[_]c_int{ 1, L, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try qsaTestAppendPool(&xfm, &m.entry, again, near, s);
    }
    const n_blocks = @divTrunc(L, ratio);
    const ref = try qsaTestMeanBlocks(keys, 0, n_blocks, ratio, s);
    defer _ = mlx.mlx_array_free(ref);
    try t.expectEqual(n_blocks, m.entry.qsa_pooled_blocks);
    try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(m.entry.qsa_pooled, ref, s));
}

test "qwen4 MTP head: a warm clamp restores exactly from the leftover marked at the trunk's checkpoint" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const s = mlx.gpuStream();
    var xfm: Trunk = undefined;
    xfm.s = s;
    xfm.allocator = t.allocator;

    const hd: c_int = 16;
    const ratio: c_int = 4;
    // The live warm turn: the trunk checkpoints at a stride boundary and at its always-on
    // backoff position, and the head runs on to L (prompt end plus the generated tail). Both
    // clamps are mid-block and more than QSA_RING_ROWS below the head's committed length —
    // the second is the live gap (SSM_SNAPSHOT_BACKOFF 30 + 124 generated).
    const L: c_int = 400;
    const cp_lo: c_int = 46;
    const cp_hi: c_int = L - 154;
    inline for (.{ cp_lo, cp_hi }) |cp| {
        try t.expect(L - cp > QSA_RING_ROWS and @mod(cp, ratio) != 0);
    }

    var prng = std.Random.DefaultPrng.init(0x51A_C0DE);
    const keys = try attn256RandBf16(prng.random(), &[_]c_int{ 1, L, hd }, s);
    defer _ = mlx.mlx_array_free(keys);

    // Built before the assignment: the scan guard reads `.cache = try` as the reinit defect.
    const head_cache = try KVCache.init(t.allocator, 1);
    var head: Qwen4Mtp = undefined;
    head.cache = head_cache;
    head.entry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = ratio };
    head.qsa_marks = .{};
    head.seq_offset = 0;
    head.pos_base = 0;
    xfm.qwen4_mtp = head;
    const m = &xfm.qwen4_mtp.?;
    defer {
        ssmFreeQsaState(&m.entry);
        m.qsa_marks.deinit();
        m.cache.deinit();
    }

    // The head appends in chunks and the trunk checkpoints at the end of each, exactly as the
    // prefill loop does: the mark is taken while the ring still ends at that row.
    var from: c_int = 0;
    inline for (.{ cp_lo, cp_hi, L }) |to| {
        var chunk = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(chunk);
        try mlx.check(mlx.mlx_slice(&chunk, keys, &[_]c_int{ 0, from, 0 }, 3, &[_]c_int{ 1, to, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try qsaTestAppendPool(&xfm, &m.entry, chunk, from, s);
        m.seq_offset = @intCast(to);
        if (to != L) try xfm.qwen4MtpMarkQsaLeftover(@intCast(to));
        from = to;
    }

    // What `commitWithState` stores for the head, and what `qwen4MtpAdopt` seeds it back from.
    var head_aux = ssmSnapshot(&m.entry);
    defer ssmSnapshotDeinit(&head_aux);
    var committed_marks = QsaHeadMarkSet.share(xfm.qwen4MtpMarks());
    defer committed_marks.deinit();
    try t.expectEqual(@as(usize, 2), committed_marks.len);

    inline for (.{ cp_hi, cp_lo }) |want| {
        // A head that appended straight to `want` — the state the clamp must reproduce.
        var refe: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = ratio };
        defer ssmFreeQsaState(&refe);
        {
            var upto = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(upto);
            try mlx.check(mlx.mlx_slice(&upto, keys, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, want, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            try qsaTestAppendPool(&xfm, &refe, upto, 0, s);
        }

        try ssmRestore(&m.entry, &head_aux);
        m.qsa_marks.deinit();
        m.qsa_marks = QsaHeadMarkSet.share(committed_marks.slice());
        m.seq_offset = @intCast(L);
        m.cache.step = @intCast(L);
        try xfm.qwen4MtpTruncate(@intCast(want));

        try t.expectEqual(@as(usize, @intCast(want)), m.seq_offset);
        try t.expectEqual(want, m.entry.qsa_hist_rows);
        try t.expectEqual(@mod(want, ratio), mlx.getShape(m.entry.aux_state)[1]);
        const ref_lv = try qsaLeftoverAt(refe.aux_state, qsaHistoryRows(&refe), want, ratio, s);
        defer _ = mlx.mlx_array_free(ref_lv);
        try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(m.entry.aux_state, ref_lv, s));
        try t.expectEqual(@divTrunc(want, ratio), mlx.getShape(m.entry.qsa_pooled)[1]);
        try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(m.entry.qsa_pooled, refe.qsa_pooled, s));

        // ...and the leftover is live: the next append re-pools through it to the full bank.
        {
            var rest = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(rest);
            try mlx.check(mlx.mlx_slice(&rest, keys, &[_]c_int{ 0, want, 0 }, 3, &[_]c_int{ 1, L, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            try qsaTestAppendPool(&xfm, &m.entry, rest, want, s);
        }
        const full = try qsaTestMeanBlocks(keys, 0, @divTrunc(L, ratio), ratio, s);
        defer _ = mlx.mlx_array_free(full);
        try t.expectEqual(@divTrunc(L, ratio), m.entry.qsa_pooled_blocks);
        try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(m.entry.qsa_pooled, full, s));
    }
}

test "qwen4 MTP head: a clamp whose retained ring is shorter than the kept leftover is not resliced" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const s = mlx.gpuStream();
    var xfm: Trunk = undefined;
    xfm.s = s;
    xfm.allocator = t.allocator;

    const hd: c_int = 16;
    const ratio: c_int = 4;
    // The band a live head's own ring can hold and still not serve: a clamp of nearly the
    // whole ring leaves fewer rows than a mid-block `keep` needs re-pooled. Its low end
    // (`QSA_RING_ROWS - ratio + 1`) the ring still serves — the same clamp must be exact
    // either way.
    const keep: c_int = 199;
    try t.expect(@mod(keep, ratio) == ratio - 1);

    var prng = std.Random.DefaultPrng.init(0xBA4D_0007);
    var drop_tail: c_int = QSA_RING_ROWS - ratio + 1;
    while (drop_tail <= QSA_RING_ROWS) : (drop_tail += 1) {
        const L: c_int = keep + drop_tail;
        const keys = try attn256RandBf16(prng.random(), &[_]c_int{ 1, L, hd }, s);
        defer _ = mlx.mlx_array_free(keys);
        const ref_pooled = try qsaTestMeanBlocks(keys, 0, @divTrunc(keep, ratio), ratio, s);
        defer _ = mlx.mlx_array_free(ref_pooled);
        const full = try qsaTestMeanBlocks(keys, 0, @divTrunc(L, ratio), ratio, s);
        defer _ = mlx.mlx_array_free(full);
        const ref_lv = try qsaLeftoverAt(keys, L, keep, ratio, s);
        defer _ = mlx.mlx_array_free(ref_lv);

        for ([_]bool{ true, false }) |with_mark| {
            // Built before the assignment: the scan guard reads `.cache = try` as the reinit defect.
            const head_cache = try KVCache.init(t.allocator, 1);
            var head: Qwen4Mtp = undefined;
            head.cache = head_cache;
            head.entry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = ratio };
            head.qsa_marks = .{};
            head.seq_offset = 0;
            head.pos_base = 0;
            xfm.qwen4_mtp = head;
            const m = &xfm.qwen4_mtp.?;
            defer {
                ssmFreeQsaState(&m.entry);
                m.qsa_marks.deinit();
                m.cache.deinit();
            }

            // The head runs live to L — its ring and capacity buffer are its own, the state a
            // commit-time trim clamps — and the trunk checkpoints at `keep` on the way.
            var from: c_int = 0;
            for ([_]c_int{ keep, L }) |to| {
                var chunk = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(chunk);
                try mlx.check(mlx.mlx_slice(&chunk, keys, &[_]c_int{ 0, from, 0 }, 3, &[_]c_int{ 1, to, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
                try qsaTestAppendPool(&xfm, &m.entry, chunk, from, s);
                m.seq_offset = @intCast(to);
                if (to == keep and with_mark) try xfm.qwen4MtpMarkQsaLeftover(@intCast(keep));
                from = to;
            }
            m.cache.step = @intCast(L);
            try t.expectEqual(QSA_RING_ROWS, mlx.getShape(m.entry.aux_state)[1]);

            const ring_serves = QSA_RING_ROWS - drop_tail >= @mod(keep, ratio);
            if (!with_mark and !ring_serves) {
                try t.expectError(error.MtpHeadQsaRingGap, xfm.qwen4MtpTruncate(@intCast(keep)));
                continue;
            }
            try xfm.qwen4MtpTruncate(@intCast(keep));
            try t.expectEqual(keep, m.entry.qsa_hist_rows);
            try t.expect(mlx.getShape(m.entry.aux_state)[1] >= @mod(keep, ratio));
            const got_lv = try qsaLeftoverAt(m.entry.aux_state, qsaHistoryRows(&m.entry), keep, ratio, s);
            defer _ = mlx.mlx_array_free(got_lv);
            try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got_lv, ref_lv, s));
            try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(m.entry.qsa_pooled, ref_pooled, s));

            // ...and the leftover is live: the append after the clamp re-pools through it.
            {
                var rest = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(rest);
                try mlx.check(mlx.mlx_slice(&rest, keys, &[_]c_int{ 0, keep, 0 }, 3, &[_]c_int{ 1, L, hd }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
                try qsaTestAppendPool(&xfm, &m.entry, rest, keep, s);
            }
            try t.expectEqual(@divTrunc(L, ratio), m.entry.qsa_pooled_blocks);
            try t.expectEqual(@as(f32, 0.0), try attn256MaxDiff(m.entry.qsa_pooled, full, s));
        }
    }
}

test "qwen4MtpResetOwned: a non-MTP request leaves the head untouched" {
    const t = std.testing;
    var xfm: Trunk = undefined;
    xfm.s = mlx.gpuStream();
    xfm.allocator = t.allocator;
    const head_cache = try KVCache.init(t.allocator, 1);
    var head: Qwen4Mtp = undefined;
    head.cache = head_cache;
    head.entry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    head.qsa_marks = .{};
    head.seq_offset = 7;
    head.pos_base = 3;
    xfm.qwen4_mtp = head;
    const m = &xfm.qwen4_mtp.?;
    defer {
        ssmFreeQsaState(&m.entry);
        m.qsa_marks.deinit();
        m.cache.deinit();
    }
    xfm.qwen4MtpResetOwned(false);
    try t.expectEqual(@as(usize, 7), m.seq_offset);
    try t.expectEqual(@as(i32, 3), m.pos_base);
}

pub fn qwen4MtpForwardPlacedOn(self: *Trunk, st: *Qwen4MtpState, stream_prev: mlx.mlx_array, token_ids_in: mlx.mlx_array, place: *const HeadPlace, project: Qwen4MtpProject) !Qwen4MtpOut {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    var live = qwen4MtpLiveState(st);
    return self.qwen4MtpForwardLive(m, &live, stream_prev, token_ids_in, 0, null, project, place);
}

test "qwen4 MTP head: last-row and no-logits projections match the full block (QWEN4_TEST_MODEL)" {
    // `MtpHeadRef.qwen4Step` consumes ONE row of logits, and the merged
    // history step after a partial accept hands the head `1 + accepted` rows —
    // so the mixer read + the vocab-wide lm_head used to run over the whole
    // block for a single row. `.last_row` cuts the stream first, `.none`
    // (appendHistory) skips both stages.
    //
    // Bars: the STREAM is a pure slice / a pure skip, so it is bit-identical
    // on both. The last row's LOGITS are not asserted bit-identical past S=1 —
    // a quantized projection at M=1 and at M=S are different kernels, and only
    // the S=1 arm shares one — so the multi-row arm takes the fixture bar.
    // That is the right bar here: these logits only steer which DRAFTS the
    // head proposes, and the trunk's verify forward decides every byte that
    // ships, so a last-ulp difference can move acceptance and never output.
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Trunk.init(io, allocator, config, &weights);
    defer xfm.deinit();
    if (xfm.qwen4_mtp == null) return error.SkipZigTest;
    xfm.compileQwen4Hc();
    xfm.compileMoeRouting();

    const hcH: usize = @as(usize, config.hc_count) * @as(usize, config.hidden_size);
    const dt = mlx.mlx_array_dtype(xfm.qwen4_mtp.?.pre_norm_hidden);

    for ([_]usize{ 1, 5 }) |S| {
        // Distinct rows: identical rows cannot tell a last-row slice from a
        // first-row one.
        const host = try allocator.alloc(f32, S * hcH);
        defer allocator.free(host);
        var rs: u64 = 0x9E3779B97F4A7C15;
        for (host) |*x| {
            rs = rs *% 6364136223846793005 +% 1442695040888963407;
            x.* = (@as(f32, @floatFromInt((rs >> 40) & 0xFFFF)) / 32768.0) - 1.0;
        }
        const hshape = [_]c_int{ 1, @intCast(S), @intCast(hcH) };
        const sp_f32 = mlx.mlx_array_new_data(host.ptr, &hshape, 3, .float32);
        defer _ = mlx.mlx_array_free(sp_f32);
        var stream_prev = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stream_prev);
        try mlx.check(mlx.mlx_astype(&stream_prev, sp_f32, dt, s));

        var id_buf: [5]i32 = undefined;
        for (0..S) |i| id_buf[i] = @intCast(7 + i * 13);
        const ishape = [_]c_int{ 1, @intCast(S) };
        const ids = mlx.mlx_array_new_data(&id_buf, &ishape, 2, .int32);
        defer _ = mlx.mlx_array_free(ids);

        try xfm.qwen4MtpReset();
        const full = try xfm.qwen4MtpForward(stream_prev, ids, 1, null, .all_rows);
        defer _ = mlx.mlx_array_free(full.logits);
        defer _ = mlx.mlx_array_free(full.stream);
        const full_l = try qwen4ReadF32(allocator, full.logits, s);
        defer allocator.free(full_l);
        const full_h = try qwen4ReadF32(allocator, full.stream, s);
        defer allocator.free(full_h);
        try testing.expectEqual(S * hcH, full_h.len);
        const v: usize = full_l.len / S;

        try xfm.qwen4MtpReset();
        const last = try xfm.qwen4MtpForward(stream_prev, ids, 1, null, .last_row);
        defer _ = mlx.mlx_array_free(last.logits);
        defer _ = mlx.mlx_array_free(last.stream);
        const last_l = try qwen4ReadF32(allocator, last.logits, s);
        defer allocator.free(last_l);
        const last_h = try qwen4ReadF32(allocator, last.stream, s);
        defer allocator.free(last_h);
        try testing.expectEqual(v, last_l.len);
        try testing.expectEqualSlices(f32, full_h[(S - 1) * hcH ..], last_h);
        if (S == 1) {
            try testing.expectEqualSlices(f32, full_l, last_l);
        } else {
            const c = qwen4CompareRows(last_l, full_l[(S - 1) * v ..], 1, v);
            std.debug.print("[qwen4] MTP last-row projection S={d}: cos {d:.6} argmax {d}/1\n", .{ S, c.min_cos, c.argmax_agree });
            try testing.expect(c.min_cos > 0.9999);
            try testing.expect(c.argmax_agree == 1);
        }

        try xfm.qwen4MtpReset();
        const bare = try xfm.qwen4MtpForward(stream_prev, ids, 1, null, .none);
        defer _ = mlx.mlx_array_free(bare.stream);
        try testing.expect(bare.logits.ctx == null);
        const bare_h = try qwen4ReadF32(allocator, bare.stream, s);
        defer allocator.free(bare_h);
        try testing.expectEqualSlices(f32, full_h, bare_h);
    }
}

test "qwen4 MTP head KV: the head cache is rebuilt under the scheme it is handed" {
    const t = std.testing;
    var xfm: Trunk = undefined;
    xfm.config = .{};
    xfm.config.num_hidden_layers = 4;
    xfm.config.head_dim = 64;
    xfm.allocator = t.allocator;
    var head: Qwen4Mtp = undefined;
    const head_cache = try KVCache.init(t.allocator, xfm.config.num_hidden_layers + 1);
    head.cache = head_cache;
    xfm.qwen4_mtp = head;
    defer xfm.qwen4_mtp.?.cache.deinit();
    try t.expectEqual(kv_quant.Scheme.off, xfm.qwen4_mtp.?.cache.config.scheme);
    try xfm.qwen4MtpApplyKvQuant(.dense);
    try t.expectEqual(kv_quant.Scheme.off, xfm.qwen4_mtp.?.cache.config.scheme);
    const trunk = kv_quant.KVQuantConfig.affine(8);
    try xfm.qwen4MtpApplyKvQuant(trunk);
    try t.expectEqual(trunk.scheme, xfm.qwen4_mtp.?.cache.config.scheme);
    try t.expectEqual(trunk.bits, xfm.qwen4_mtp.?.cache.config.bits);
}

test "qwen4 MTP head: quantized KV rollback + re-append matches dense draft ids (QWEN4_TEST_MODEL)" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Trunk.init(io, allocator, config, &weights);
    defer xfm.deinit();
    if (xfm.qwen4_mtp == null) return error.SkipZigTest;
    xfm.compileQwen4Hc();
    xfm.compileMoeRouting();

    const hcH: usize = @as(usize, config.hc_count) * @as(usize, config.hidden_size);
    const dt = mlx.mlx_array_dtype(xfm.qwen4_mtp.?.pre_norm_hidden);
    const S: usize = 4;
    const keep: usize = 2;
    const host = try allocator.alloc(f32, S * hcH);
    defer allocator.free(host);
    var rs: u64 = 0x9E3779B97F4A7C15;
    for (host) |*x| {
        rs = rs *% 6364136223846793005 +% 1442695040888963407;
        x.* = (@as(f32, @floatFromInt((rs >> 40) & 0xFFFF)) / 32768.0) - 1.0;
    }
    const hshape = [_]c_int{ 1, @intCast(S), @intCast(hcH) };
    const sp_f32 = mlx.mlx_array_new_data(host.ptr, &hshape, 3, .float32);
    defer _ = mlx.mlx_array_free(sp_f32);
    var stream_prev = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stream_prev);
    try mlx.check(mlx.mlx_astype(&stream_prev, sp_f32, dt, s));
    var id_buf: [4]i32 = .{ 7, 20, 33, 46 };
    const ishape = [_]c_int{ 1, @intCast(S) };
    const ids = mlx.mlx_array_new_data(&id_buf, &ishape, 2, .int32);
    defer _ = mlx.mlx_array_free(ids);

    const Run = struct {
        fn lastLogits(xf: *Trunk, st: mlx.mlx_stream, alloc: std.mem.Allocator, stream: mlx.mlx_array, tok: mlx.mlx_array, kkeep: usize) ![]f32 {
            try xf.qwen4MtpReset();
            const full = try xf.qwen4MtpForward(stream, tok, 1, null, .all_rows);
            defer _ = mlx.mlx_array_free(full.logits);
            defer _ = mlx.mlx_array_free(full.stream);
            try xf.qwen4MtpTruncate(kkeep);
            const sh = mlx.getShape(stream);
            var tail = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(tail);
            try mlx.check(mlx.mlx_slice(&tail, stream, &[_]c_int{ 0, @intCast(kkeep), 0 }, 3, &[_]c_int{ sh[0], sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, st));
            const tsh = mlx.getShape(tok);
            var ttail = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ttail);
            try mlx.check(mlx.mlx_slice(&ttail, tok, &[_]c_int{ 0, @intCast(kkeep) }, 2, &[_]c_int{ tsh[0], tsh[1] }, 2, &[_]c_int{ 1, 1 }, 2, st));
            const more = try xf.qwen4MtpForward(tail, ttail, @intCast(kkeep + 1), null, .last_row);
            defer _ = mlx.mlx_array_free(more.logits);
            defer _ = mlx.mlx_array_free(more.stream);
            return qwen4ReadF32(alloc, more.logits, st);
        }
        fn argmaxGap(row: []const f32) struct { id: usize, gap: f32 } {
            var b0: usize = 0;
            var b1: usize = 0;
            for (row, 0..) |x, i| {
                if (x > row[b0]) {
                    b1 = b0;
                    b0 = i;
                } else if (i != b0 and (b1 == b0 or x > row[b1])) {
                    b1 = i;
                }
            }
            return .{ .id = b0, .gap = row[b0] - row[b1] };
        }
    };

    const dense_l = try Run.lastLogits(&xfm, s, allocator, stream_prev, ids, keep);
    defer allocator.free(dense_l);
    try xfm.qwen4MtpApplyKvQuant(kv_quant.KVQuantConfig.affine(8));
    try testing.expectEqual(kv_quant.Scheme.affine, xfm.qwen4_mtp.?.cache.config.scheme);
    try testing.expectEqual(@as(u8, 8), xfm.qwen4_mtp.?.cache.config.bits);
    const quant_l = try Run.lastLogits(&xfm, s, allocator, stream_prev, ids, keep);
    defer allocator.free(quant_l);
    try testing.expectEqual(dense_l.len, quant_l.len);
    const d = Run.argmaxGap(dense_l);
    const q = Run.argmaxGap(quant_l);
    if (d.id != q.id) {
        try testing.expect(d.gap <= 0.15);
    }
}

test "qwen4MtpForward: every error path of the MTP head forward releases its stream (fault-injected, QWEN4_TEST_MODEL)" {
    // The claim under proof: `h` is born with ONE function-scope errdefer, and
    // every later rebind either consumes the old handle (`hcWrite`, which
    // frees its input only past its last `try`) or frees it itself (the
    // `.last_row` slice) — so no error path leaks it and none frees it twice.
    // Argument cannot settle that; a sweep can.
    //
    // Method: fault the k-th checked mlx-c call for every k the forward
    // issues, on every projection, at S > 1 so the slice path is live, and
    // require live Metal bytes back at the pre-call baseline each time. The
    // head's own state (KV, aux, seq_offset) is reset on BOTH sides of the
    // measurement, so only ownership can move the number.
    //
    // What the oracle can and cannot see: MLX is lazy, so a handle leaked
    // before anything downstream of it has been evaluated holds no buffer and
    // is invisible here — the equality is a NECESSARY condition, sharp for
    // every k at or past the first evaluation inside the head (the QSA host
    // reads, the eval cadence, the projection) and weak before it. The static
    // scan above covers the whole function unconditionally, and the DOUBLE
    // free half is unconditional at every k: it aborts this binary rather
    // than failing an assertion.
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Trunk.init(io, allocator, config, &weights);
    defer xfm.deinit();
    if (xfm.qwen4_mtp == null) return error.SkipZigTest;
    xfm.compileQwen4Hc();
    xfm.compileMoeRouting();

    const S: usize = 5; // > 1: keeps the `.last_row` / `.mixed_last_row` slice live
    const hcH: usize = @as(usize, config.hc_count) * @as(usize, config.hidden_size);
    const dt = mlx.mlx_array_dtype(xfm.qwen4_mtp.?.pre_norm_hidden);
    const host = try allocator.alloc(f32, S * hcH);
    defer allocator.free(host);
    var rs: u64 = 0x243F6A8885A308D3;
    for (host) |*x| {
        rs = rs *% 6364136223846793005 +% 1442695040888963407;
        x.* = (@as(f32, @floatFromInt((rs >> 40) & 0xFFFF)) / 32768.0) - 1.0;
    }
    const hshape = [_]c_int{ 1, @intCast(S), @intCast(hcH) };
    const sp_f32 = mlx.mlx_array_new_data(host.ptr, &hshape, 3, .float32);
    defer _ = mlx.mlx_array_free(sp_f32);
    var stream_prev = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stream_prev);
    try mlx.check(mlx.mlx_astype(&stream_prev, sp_f32, dt, s));
    var id_buf: [S]i32 = undefined;
    for (0..S) |i| id_buf[i] = @intCast(7 + i * 13);
    const ishape = [_]c_int{ 1, @intCast(S) };
    const ids = mlx.mlx_array_new_data(&id_buf, &ishape, 2, .int32);
    defer _ = mlx.mlx_array_free(ids);

    inline for ([_]Trunk.Qwen4MtpProject{ .all_rows, .last_row, .mixed_last_row, .none }) |proj| {
        // Un-faulted warm-up: builds every lazy artefact the forward touches
        // and counts the checked ops it issues. Its outputs are freed HERE, or
        // every later baseline would carry them and the equality would be a
        // statement about nothing.
        try xfm.qwen4MtpReset();
        const c0 = mlx.op_count.load(.monotonic);
        const warm = try xfm.qwen4MtpForward(stream_prev, ids, 1, null, proj);
        const n_ops = mlx.op_count.load(.monotonic) - c0;
        try testing.expect(warm.stream.ctx != null);
        if (warm.logits.ctx != null) _ = mlx.mlx_array_free(warm.logits);
        if (warm.mixed.ctx != null) _ = mlx.mlx_array_free(warm.mixed);
        _ = mlx.mlx_array_free(warm.stream);

        var errs: usize = 0;
        var k: u64 = 1;
        while (k <= n_ops) : (k += 1) {
            try xfm.qwen4MtpReset();
            const base = mlxSettledActiveBytes(s);
            mlx.fault.arm(k);
            const r = xfm.qwen4MtpForward(stream_prev, ids, 1, null, proj);
            mlx.fault.disarm();
            if (r) |o| {
                // Reachable only where a fallback arm below the head swallowed
                // the injected error; the outputs are still ours to free.
                if (o.logits.ctx != null) _ = mlx.mlx_array_free(o.logits);
                if (o.mixed.ctx != null) _ = mlx.mlx_array_free(o.mixed);
                if (o.stream.ctx != null) _ = mlx.mlx_array_free(o.stream);
            } else |_| {
                errs += 1;
            }
            try xfm.qwen4MtpReset();
            try testing.expectEqual(base, mlxSettledActiveBytes(s));
        }
        std.debug.print("[qwen4] MTP fault sweep {s}: {d} checked ops, {d} error paths, live bytes back at baseline on all\n", .{ @tagName(proj), n_ops, errs });
        try testing.expect(errs >= 1);
    }
    try xfm.qwen4MtpReset();
}

test "reserveQsaHistory: the MTP head entry is reserved at the same length" {
    var slot = [_]SSMCacheEntry{.{ .ssm_state = .{ .ctx = null }, .conv_state = .{ .ctx = null }, .initialized = false }};
    var head: SSMCacheEntry = .{ .ssm_state = .{ .ctx = null }, .conv_state = .{ .ctx = null }, .initialized = false };
    const tokens: usize = 8192;
    reserveQsaHistoryWithHead(slot[0..], &head, tokens);
    try testing.expectEqual(tokens, slot[0].qsa_reserve_rows);
    try testing.expectEqual(tokens, head.qsa_reserve_rows);
}

test "qwen4 MTP adopt rejects a negative origin before mutating nonempty state" {
    const allocator = testing.allocator;
    var source = try KVCache.init(allocator, 1);
    defer source.deinit();
    var snapshot = try source.snapshot();
    defer snapshot.deinit();
    snapshot.step = 1;

    const pooled_data = [_]f32{0};
    const pooled = mlx.mlx_array_new_data(&pooled_data, &.{ 1, 1, 1 }, 3, .float32);
    var aux = SSMCacheEntrySnapshot{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = false,
        .qsa_pooled = pooled,
        .qsa_rows = 1,
    };
    defer ssmSnapshotDeinit(&aux);

    var xfm: Trunk = undefined;

    xfm.allocator = allocator;
    xfm.s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(xfm.s);
    const head_cache = try KVCache.init(allocator, 1);
    var head: Qwen4Mtp = undefined;
    head.cache = head_cache;
    head.entry = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
    head.seq_offset = 0;
    head.pos_base = 7;
    head.qsa_marks = .{};
    xfm.qwen4_mtp = head;
    defer {
        const live = &xfm.qwen4_mtp.?;
        live.cache.deinit();
        _ = mlx.mlx_array_free(live.entry.conv_state);
        _ = mlx.mlx_array_free(live.entry.ssm_state);
        ssmFreeQsaState(&live.entry);
        live.qsa_marks.deinit();
    }

    try testing.expectError(error.MtpHeadInvalidOrigin, xfm.qwen4MtpAdopt(&snapshot, &aux, &.{}, -1, 1));
    try testing.expectEqual(@as(usize, 0), xfm.qwen4_mtp.?.cache.step);
    try testing.expectEqual(@as(usize, 0), xfm.qwen4_mtp.?.seq_offset);
    try testing.expectEqual(@as(c_int, 7), xfm.qwen4_mtp.?.pos_base);
    try xfm.qwen4MtpAdopt(&snapshot, &aux, &.{}, 8, 1);
    try testing.expectEqual(@as(usize, 1), xfm.qwen4_mtp.?.cache.step);
    try testing.expectEqual(@as(usize, 1), xfm.qwen4_mtp.?.seq_offset);
    try testing.expectEqual(@as(c_int, 8), xfm.qwen4_mtp.?.pos_base);
}

test "single-stream verify feature selection excludes prefill and ordinary decoding" {
    var xfm: Trunk = undefined;
    const old_features = verify_projection.single_verify_test_features;
    defer verify_projection.single_verify_test_features = old_features;
    verify_projection.single_verify_test_features = @splat(true);
    inline for (.{ SingleVerifyFeature.routing, SingleVerifyFeature.down_reduce }) |feature| {
        for ([_]bool{ false, true }) |capture| {
            xfm.spec_capture_ssm = capture;
            for ([_]c_int{ 1, 2, 4 }) |batch| {
                for ([_]c_int{ 1, 2, 3, 4, 5, 6, 7, 16, 4096 }) |width| {
                    try testing.expectEqual(capture and batch == 1 and width >= 2 and width <= 6, xfm.verifyFeatureEnabled(feature, false, batch, width));
                    try testing.expect(xfm.verifyFeatureEnabled(feature, true, batch, width));
                }
            }
        }
        xfm.spec_capture_ssm = true;
        verify_projection.single_verify_test_features[@backingInt(feature)] = false;
        try testing.expect(!xfm.verifyFeatureEnabled(feature, false, 1, 5));
        try testing.expect(xfm.verifyFeatureEnabled(feature, true, 1, 5));
        verify_projection.single_verify_test_features[@backingInt(feature)] = true;
    }
}

const SingleVerifyTestResult = struct {
    logits: mlx.mlx_array,
    last: mlx.mlx_array,
    all: mlx.mlx_array,

    fn deinit(self: *@This()) void {
        _ = mlx.mlx_array_free(self.logits);
        _ = mlx.mlx_array_free(self.last);
        _ = mlx.mlx_array_free(self.all);
    }
};

fn singleVerifyTestRound(shell: *@import("../../engines/mlx/transformer.zig").Transformer, slot: *Qwen4TestSlot, row: mlx.mlx_array) !SingleVerifyTestResult {
    const Gen = @import("../../engines/mlx/generate.zig").Generator;
    var gen: Gen = undefined;
    gen.xfm = shell;
    gen.ctx = slot.ctx;
    defer slot.ctx = gen.ctx;
    var state: Gen.MtpRoundState = undefined;
    state.tracing = false;
    state.verify_input = row;
    try gen.mtpRoundVerify(&state);
    var result: SingleVerifyTestResult = .{ .logits = state.verify_logits, .last = state.new_hidden, .all = state.verify_hidden_all };
    errdefer result.deinit();
    const values = mlx.mlx_vector_array_new_data(&.{ result.logits, result.last, result.all }, 3);
    defer _ = mlx.mlx_vector_array_free(values);
    try mlx.check(mlx.mlx_eval(values));
    return result;
}

test "single-stream verify optimizations preserve native outputs and captures (QWEN4_TEST_MODEL)" {
    const path = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const old_features = verify_projection.single_verify_test_features;
    defer verify_projection.single_verify_test_features = old_features;
    var config = try model_mod.parseConfig(io, a, std.mem.span(path));
    defer config.deinit(a);
    var shell = try @import("../../engines/mlx/plugin/driver.zig").load(io, a, &config, std.mem.span(path), mlx.gpuStream());
    defer shell.deinit();
    const xfm = shell.qwen4Trunk().?;
    const prefix_len = if (std.c.getenv("SINGLE_MTP_TEST_CONTEXT")) |value| try std.fmt.parseInt(usize, std.mem.span(value), 10) else 4101;
    const ids = try a.alloc(i32, prefix_len + 64);
    defer a.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(1 + (i * 7919) % 40000);
    var slots: [2]*Qwen4TestSlot = undefined;
    var initialized: usize = 0;
    defer for (slots[0..initialized]) |slot| slot.deinit(a);
    const prefill_routes = verify_projection.mtp_verify_kernel_calls[2];
    for (&slots, 0..) |*target, arm| {
        verify_projection.single_verify_test_features = @splat(arm == 1);
        target.* = try Qwen4TestSlot.init(a, config.num_hidden_layers);
        initialized += 1;
        target.*.cache.config = KVQuantConfig.affine(8);
        target.*.ctx.skip_lm_head = true;
        var start: usize = 0;
        while (start < prefix_len) {
            const end = @min(start + 4096, prefix_len);
            const output = try target.*.forward(xfm, ids[start..end]);
            try mlx.check(mlx.mlx_array_eval(output));
            _ = mlx.mlx_array_free(output);
            start = end;
        }
        target.*.ctx.skip_lm_head = false;
    }
    try testing.expectEqual(prefill_routes, verify_projection.mtp_verify_kernel_calls[2]);
    var position = prefix_len;
    for ([_]usize{ 1, 2, 3, 4, 5, 6, 7 }) |width| {
        const input = mlx.mlx_array_new_data(@ptrCast(ids[position..][0..width].ptr), &.{ 1, @as(c_int, @intCast(width)) }, 2, .int32);
        defer _ = mlx.mlx_array_free(input);
        verify_projection.single_verify_test_features = @splat(false);
        var reference = try singleVerifyTestRound(&shell, slots[0], input);
        defer reference.deinit();
        const before = [_]u64{ verify_projection.mtp_verify_kernel_calls[2], verify_projection.mtp_verify_indexed_input_calls, verify_projection.mtp_verify_expert_pairs_calls, verify_projection.mtp_verify_expert_reduce_calls };
        verify_projection.single_verify_test_features = @splat(true);
        var candidate = try singleVerifyTestRound(&shell, slots[1], input);
        defer candidate.deinit();
        const after = [_]u64{ verify_projection.mtp_verify_kernel_calls[2], verify_projection.mtp_verify_indexed_input_calls, verify_projection.mtp_verify_expert_pairs_calls, verify_projection.mtp_verify_expert_reduce_calls };
        inline for (.{ "logits", "last", "all" }) |field| try expectRowsByteEqual(a, xfm.s, @field(reference, field), @field(candidate, field), "single verify " ++ field);
        for (slots[0].entries, slots[1].entries) |*left, *right| {
            inline for (.{ "conv_state", "ssm_state", "aux_state", "qsa_pooled", "spec_conv_input", "spec_ple_input" }) |field| {
                const x = @field(left.*, field);
                const y = @field(right.*, field);
                if (x.ctx == null or y.ctx == null) {
                    try testing.expect(x.ctx == null and y.ctx == null);
                } else try expectRowsByteEqual(a, xfm.s, x, y, "single verify " ++ field);
            }
            if (left.spec_state_seq.ctx != null) {
                const x = try logicalSsmCapture(xfm.s, left);
                defer _ = mlx.mlx_array_free(x);
                const y = try logicalSsmCapture(xfm.s, right);
                defer _ = mlx.mlx_array_free(y);
                try expectRowsByteEqual(a, xfm.s, x, y, "single verify logical capture");
            } else try testing.expect(right.spec_state_seq.ctx == null);
            try testing.expectEqual(left.ple_prev_valid, right.ple_prev_valid);
            try testing.expectEqualSlices(u32, &left.ple_prev, &right.ple_prev);
            ssmFreeSpecCapture(left);
            ssmFreeSpecCapture(right);
        }
        for (before, after) |old, new| {
            if (verifySharedHardware() and width >= 2 and width <= 6) {
                try testing.expect(new > old);
            } else try testing.expectEqual(old, new);
        }
        std.debug.print("[single-verify-exact] context={d} width={d} calls={any}\n", .{ prefix_len, width, after });
        position += width;
    }
}
