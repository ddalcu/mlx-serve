//! engines/mlx/forward_ctx.
//! Owns SpecTree, CaptureLayers, ForwardCtx.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const HeadPlace = @import("../../plugins/qwen4_exp/mtp.zig").HeadPlace;
const abi = @import("../../abi/plugin.zig");
const row_attn = @import("../../sdk/blocks/attention/row_attn.zig");
const PlePending = @import("../../plugins/qwen4_exp/ple.zig").PlePending;
const QsaArmCounts = @import("../../plugins/qwen4_exp/qsa/arms.zig").QsaArmCounts;
const KVCache = @import("kv/cache.zig").KVCache;
const SSMCacheEntry = @import("ssm/entry.zig").SSMCacheEntry;
const Transformer = @import("transformer.zig").Transformer;

/// A draft tree's verify rows: row w sits at position offset + depth[w], and
/// its keys and recurrence follow its ancestors.
pub const SpecTree = struct {
    /// `gdn_decode.treeTable`: `[W]` parent rows (-1 for the root), then each
    /// row's conv-input sources.
    parents: mlx.mlx_array,
    attn: row_attn.Tree,

    pub fn deinit(self: *const SpecTree) void {
        for ([_]mlx.mlx_array{ self.parents, self.attn.depth, self.attn.path }) |a| _ = mlx.mlx_array_free(a);
    }
};

// ── Forward context ──
//
// Routes per-request mutable state (KV cache, MoE seq offset, SSM entries,
// hidden-state capture target, vision embeddings) into the forward pass via
// a single struct. The legacy single-slot path uses `Transformer.defaultCtx()`
// which points at the Transformer's own fields — semantically identical to the
// pre-refactor code. Concurrent batching (Phase 1+) constructs one ctx per
// in-flight request so multiple slots can share a Transformer's weights while
// owning their own caches.
/// DFlash capture request: intermediate layer OUTPUTS, `hidden_states[i+1]`
/// semantics (the residual stream AFTER layer `ids[k]` completes, before the
/// next layer's input norm). Standard forward path only (v1).
pub const CaptureLayers = struct {
    /// Trunk layer indices to capture, ascending.
    ids: []const u32,
    /// Filled per forward: `out[k]` = h after layer `ids[k]`, ALL positions
    /// `[B, L, H]`, refcount-shared with the live graph. Caller owns the
    /// handles (init with `mlx_array_new`, free after use).
    out: []mlx.mlx_array,
};

pub const ForwardCtx = struct {
    cache: *KVCache,
    /// A later pipeline rank's input: the previous rank's hyper-connection stream, in place of tokens.
    stream_in: ?mlx.mlx_array = null,
    moe_seq_offset: *usize,
    ssm_entries: ?[]SSMCacheEntry,
    ssm_member_gen: u64 = 0,
    capture_hidden: ?*mlx.mlx_array,
    /// A spoken frame's input rows `[1, rows, hidden]` in place of the ids' embedding (`Forward.embeds`); served
    /// models only.
    step_embeds: ?mlx.mlx_array = null,
    /// Like `capture_hidden` but receives the FULL post-final-norm hidden
    /// `[B, L, H]` (all positions, refcount-shared) instead of the last
    /// position only. Used by the Qwen MTP head, whose committed-history
    /// cache needs the trunk hidden at every verify/prefill position. The
    /// hybrid path (both fields) hands over the residual BEFORE the final
    /// norm: that is what the Nemotron-H head reads.
    capture_hidden_all: ?*mlx.mlx_array = null,
    /// qwen4_exp: receives the PRE-mixer hyper-connection stream `[B, L,
    /// hc*hidden]` (all positions) — what the MTP head consumes.
    capture_stream_all: ?*mlx.mlx_array = null,
    /// DFlash: capture the residual stream at these layer OUTPUTS (all
    /// positions). Honored by `forwardStandardWith` only — DflashModel.bind
    /// refuses non-standard-path targets so this can never be silently
    /// ignored on an engaged path.
    capture_layers: ?*CaptureLayers = null,
    /// An additive attention term composed into the standard PREFILL path's
    /// mask, `[1, 1, q_len, kv_len]` bf16. It must ALREADY be causal: a
    /// "causal"-mode layer swaps to an array mask carrying this verbatim, so
    /// a term that only masked padding would drop causality on the
    /// full-attention layers. Sliding layers ADD it to their band mask.
    ///
    /// Entries must be FINITE (the reference's -1e9, not -inf): a left-padded
    /// prompt has query rows whose every visible key is padding, and an
    /// all -inf row softmaxes to NaN.
    ///
    /// Set only by the LTX text encoder (`ltx_video.gemmaCapture4`, a plugin declaring `Caps.prefill_mask`), whose
    /// prompts are left-padded to a fixed width the reference masks out. Null
    /// everywhere else — and null is what keeps the fused prefill kernels,
    /// which mask from geometry alone and cannot carry an extra term, on
    /// their normal path. One-shot forwards only: a chunked prefill would
    /// need a per-chunk slice of it.
    prefill_mask_add: ?mlx.mlx_array = null,
    /// PLD spec-decode: when true, the GatedDeltaNet trunk records per-position
    /// SSM/conv state during the verify forward (see `SSMCacheEntry.spec_*`),
    /// so partial-accept rollback needs no re-forward. Set only by `nextPld`
    /// for the verify pass on a GDN model; the flag reaches the layers via
    /// `Transformer.spec_capture_ssm`.
    capture_ssm_seq: bool = false,
    /// Verify rows forming a draft tree (null = a chain).
    tree: ?*const SpecTree = null,
    /// A speculative VERIFY forward (drafts behind one committed token), never a prompt chunk.
    spec_verify: bool = false,
    /// Hand the graph built so far to the GPU every this many layers (0 =
    /// build it whole): a spec round's verify waits on the host otherwise.
    pipeline_build: u32 = 0,
    vision_embeddings: ?mlx.mlx_array,
    /// Chunked vision prefill (issue #197): number of image/audio placeholder
    /// tokens already spliced by earlier chunks of this prefill. The splice's
    /// row lookup is a cumsum over the CURRENT forward's placeholder mask, so
    /// without this every chunk would restart at vision row 0. Set per chunk
    /// by the generate.zig prefill loop; 0 for unchunked/decode forwards.
    vision_splice_offset: usize = 0,
    /// Qwen3-VL interleaved M-RoPE. `mrope_pos` is the server-computed flat
    /// [3 × mrope_total] i32 position-id table (axis-major: t,h,w rows), borrowed
    /// from the slot. `mrope_delta` shifts the scalar decode RoPE (decode tokens
    /// are text → t=h=w). `mrope_cos/sin_cur` are the per-prefill-chunk cos/sin
    /// tables, built once in forwardMoeWith and consumed by every full-attn layer
    /// that chunk, then freed. All null/0 for non-Qwen / text-only requests.
    mrope_pos: ?[]const i32 = null,
    mrope_total: usize = 0,
    mrope_delta: i32 = 0,
    mrope_cos_cur: ?mlx.mlx_array = null,
    mrope_sin_cur: ?mlx.mlx_array = null,
    /// qwen4_exp QSA: the bool `[B, 1, S, kv]` mask (indexer-selected blocks
    /// ∧ causal) `gatedFullAttnWith` must apply this layer. Null-ctx = dense.
    qsa_mask: mlx.mlx_array = .{ .ctx = null },
    /// qwen4_exp QSA: the sorted `[B, S, kb]` int32 block selection (INT_MAX
    /// past each row's count) for the three gather arms — `gatherQsa256`
    /// (prefill), `qsaVerifyGatherAttn` (S = 2..15) and `qsaDecodeGatherAttn`
    /// (S == 1); set INSTEAD of `qsa_mask`. Null-ctx = not selected.
    qsa_blocks: mlx.mlx_array = .{ .ctx = null },
    /// Which QSA arm served each full-attention call of THIS request. The
    /// `[qsa-arms]` line is a per-request meter, so the tally is owned by the
    /// seam that logs it (`Generator.logQsaArms` over the Generator's own
    /// ctx) — a Transformer-level counter merged every concurrent slot's
    /// calls (batched decode + interleaved prefill) into whichever request
    /// finished first. A batched forward attributes each call to every
    /// participating slot's own ctx (`noteQsaArmForSlots`).
    ///
    /// Scoped to the TRUNK: the qwen4 MTP head's forward builds its own ctx
    /// and its single QSA layer is not part of this tally.
    qsa_arms: QsaArmCounts = .{},
    /// Phase 2 (Plan ricky): when true, attention call sites consume the
    /// cache's quantized K/V triples directly via `kv_quant.quantAttention`
    /// instead of dequantizing through `DenseKVView`. Only effective when
    /// the cache scheme is .affine (.off has no quant triple to consume).
    /// Default false → unchanged dense SDPA path.
    kv_attn_fused: bool = false,
    /// Batched-embeddings: additive key-padding mask [B, 1, 1, T] consumed
    /// by the BERT encoder forward so padded positions never attend. Null
    /// (the default) keeps the unmasked single-sequence path.
    key_pad_mask: ?mlx.mlx_array = null,
    /// Return the post-final-norm hidden instead of logits (caller frees
    /// either way), skipping the 262K-vocab lm_head projection. Set by the
    /// LTX text encoder (which wants the residual stream, not a next token).
    skip_lm_head: bool = false,
    /// Batched decode over a GatedDeltaNet trunk (`forwardMoeBatchedDecode`).
    /// Non-null means: this forward runs at `batch == batch_slots.len`,
    /// `seq_len == 1`, and every per-slot KV cache lives on the slot ctxs
    /// here — NOT on `ctx.cache`, which is only a shape donor in this mode.
    /// The GDN layers need no branch (their state is merged into this ctx's
    /// `ssm_entries` at B=N and split back out by the driver); only the
    /// full-attention layers do, because K/V lengths differ per slot and
    /// each slot ropes at its own offset.
    ///
    /// `batch_rope_offsets` is the matching int32 `[N]` offset array for
    /// `mlx_fast_rope_dynamic` — already carrying each slot's M-RoPE delta,
    /// so the attention layer never re-derives a per-slot offset.
    batch_slots: ?[]const *ForwardCtx = null,
    batch_rope_offsets: ?mlx.mlx_array = null,
    /// MTP head: one QSA entry, not `ssm_entries[layer]`. Null on the trunk.
    qsa_entry: ?*SSMCacheEntry = null,
    /// Absolute position of this slot's QSA key row 0 (0 on the trunk).
    qsa_pos_base: c_int = 0,
    /// MTP head draft row placed past dead history rows (`qwen4MtpForwardPlacedOn`), and its
    /// `[1]` int32 RoPE position. Null on every other forward.
    head_place: ?*const HeadPlace = null,
    head_rope_pos: mlx.mlx_array = .{ .ctx = null },
    /// qwen4_exp pipelined decode: the n-gram PLE rows are a HOST gather keyed
    /// on the token ids, and the pipelined step builds its graph on a token
    /// that is still a lazy sample. With `ple_defer` set, `pleEmbedding` hands
    /// the graph an unfilled leaf and records it here; the caller fills it via
    /// `Transformer.flushDeferredPle` (syncing on the token THEN) right before
    /// the graph is evaluated. Set by `generate.lazyForward` (decode) and by
    /// the MTP verify build in `Generator.nextMtp` (whose `[1, 1+m]` input is
    /// the still-lazy draft chain); a forward evaluated with `ple_pending`
    /// still set reads zero rows.
    ple_defer: bool = false,
    ple_pending: ?PlePending = null,
    /// A served model's id for this slot (`Forward.slots`), 0 until its first forward, and what its last `defer_host`
    /// forward left owed (`plugin/settle.zig`).
    plugin_slot: u32 = 0,
    plugin_owed: abi.Owed = .none,

    /// The shell's own sequence (`defaultCtx`, rebuilt per call): one id across its forwards.
    pub const shell_slot: u32 = std.math.maxInt(u32);
};
