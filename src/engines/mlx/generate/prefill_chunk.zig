//! Prefill chunking: chunk width, tail merge, SSM checkpoint cadence, vision chunking and allocator-cache clears.
const std = @import("std");
const env = @import("../../../env.zig");
const transformer_mod = @import("../transformer.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const mtp_mod = @import("../../../sdk/spec/mtp/mtp.zig");
const testing = std.testing;
const generate_mod = @import("../generate.zig");
const WIDE_PREFILL_CHUNK = @import("../generate.zig").WIDE_PREFILL_CHUNK;
const attn256 = @import("../../../sdk/blocks/attention/attn256.zig");


/// `--prefill-chunk` for an embedded engine's u32 chunk (ds4): 0 = the engine's default.
pub fn explicitPrefillChunkU32() u32 {
    if (!generate_mod.prefillCli().explicit) return 0;
    return std.math.cast(u32, generate_mod.prefillCli().override) orelse std.math.maxInt(u32);
}

/// THIS model's greedy tail: its `mtp_greedy_tail` outranks `--mtp-greedy-tail`.
pub fn mtpGreedyTailFor(override: ?bool) bool {
    return override orelse generate_mod.mtp_greedy_tail_default;
}

/// Per-prefill state for the adaptive width; lives on `runPrefill`'s stack. The policy is
/// `server.adaptivePrefillWidth`.
/// The one place a chosen width becomes the running width.
pub fn commitAdaptiveWidth(cur: *usize, st: *AdaptiveWidthState, w: u32) void {
    cur.* = w;
    st.transitions +|= 1;
    st.width_min = @min(st.width_min, w);
    st.width_max = @max(st.width_max, w);
}

pub const AdaptiveWidthState = struct {
    supporting: u8 = 0,
    /// One-way ratchet: a prefill that has stepped down never widens again.
    ratcheted: bool = false,
    transitions: u32 = 0,
    width_min: u32 = 0,
    width_max: u32 = 0,
};

/// Why a request left speculative decoding for the rest of its generation.
pub const SpecDisableReason = enum {
    none,
    acceptance,
    max_ctx,
    /// A thinking budget closed the thought through a plain forward; a spec mode
    /// other than MTP is not resynced, the answer decodes regular.
    think_bound,
    /// A forced tool-call opener went through a plain forward (`ToolForce`).
    tool_choice,
    /// The measured round cost more per token than a measured serial token (`MtpAdaptive`).
    adaptive,
    /// A speculating slot gained company: it decodes plain so it can join the batched group.
    company,
};

/// Effective MTP history window for a prefill forwarding `prefix_len`
/// positions: 0 (capture everything) unless windowing is on AND the tail is
/// past the threshold — short/medium prompts keep byte-identical behavior.
pub fn effectiveMtpHistoryWindow(prefix_len: usize, window: usize) usize {
    if (window == 0 or prefix_len <= mtp_mod.HISTORY_WINDOW_THRESHOLD) return 0;
    return window;
}

/// Does prefill chunk [pos, end) contribute MTP history? Zero window = all
/// chunks; otherwise only chunks overlapping the last `window` positions of
/// the prefix (a boundary chunk contributes whole — the window is a floor,
/// never an exact cut, so acceptance never loses mid-chunk context).
pub fn chunkNeedsMtpHistory(pos: usize, end: usize, prefix_len: usize, window: usize) bool {
    _ = pos;
    if (window == 0) return true;
    return end > prefix_len - @min(window, prefix_len);
}

/// One layer's materialized-score budget for unfused-SDPA prefill (see
/// boundedPrefillChunk). 4 GiB keeps the full 8K chunk for every context up
/// to ~16K on 16-head models (further on fewer heads) and degrades gradually
/// to the 512 floor as heads × context grows toward 262K.
pub const PREFILL_SCORES_BUDGET_BYTES: u64 = 4 << 30;
/// Lower bound for the auto-capped prefill chunk; also its rounding grain
/// (repeating chunk sizes let the MLX allocator cache reuse score buffers).
pub const PREFILL_CHUNK_FLOOR: usize = 512;

/// MLX's fused SDPA kernels cover head_dim <= 128. Every Gemma-4 and
/// Qwen3.5/3.6 checkpoint ships head_dim=256, which falls back to the
/// composed path that MATERIALIZES a [heads, chunk, total_kv] bf16 score
/// tensor per layer — at an 8K chunk × 255K ctx × 16 heads that is ~67 GB
/// and an uncatchable Metal command-buffer OOM. Cap the chunk so ONE layer's
/// score tensor stays within PREFILL_SCORES_BUDGET_BYTES at this prompt's
/// FINAL KV length (the last chunk attends to everything). Fused head dims
/// and short contexts return `base_chunk` untouched, so typical traffic
/// keeps full prefill throughput; the cap only bites when heads × total_ctx
/// actually outgrows the budget. Never raises a caller-lowered base.
///
/// DELIBERATELY ignores the msv_attn_p256 fused kernel (unlike
/// prefillEvalCadence / prefillMemoryNeeded, which drop their score term via
/// transformer.prefillHeadDimFused): the fused kernel removes the SCORE
/// transient, but a big chunk still scales the OTHER per-chunk transients
/// (MoE gather buffers, per-chunk KV concat) — measured LIVE on
/// gemma-4-26B-A4B at a 99K prompt: fused @ chunk 8192 = 736 tok/s / 61.2 GB
/// peak vs fused @ chunk 1024 = 712 tok/s / 39.5 GB. +3% speed is not worth
/// +22 GB peak (a 64 GB Mac dies), so the cap stays keyed on raw head_dim.
///
/// `sliding_band_arch` (config.has_sliding_window) picks the policy family:
/// archs WITHOUT sliding-band layers (qwen3_5/3_6: GDN + full attention)
/// additionally cap the auto chunk at 2048 — composed-causal prefill
/// measured strictly faster and ~9 GB lighter there (see the inline
/// comment). Gemma keeps the formula-only policy for its fused band layers.
/// `score_head_dim` is the width the SCORE tensor is contracted at
/// (`ModelConfig.prefillScoreHeadDim`) — not necessarily `head_dim`. An MLA arch
/// scores at 192 while storing 128-wide values, and reading `head_dim` there
/// silently exempted the one arch in the tree whose composed path materializes
/// the biggest score tensors. The two hd-256 policy branches below stay keyed
/// on 256 exactly: both were measured on hd-256 checkpoints and neither
/// generalizes (the fused kernel is hd-256-only; the 2048 composed cap was
/// tuned against a 27B's own prefill ladder).
pub fn boundedPrefillChunk(base_chunk: usize, score_head_dim: u32, n_heads: u32, total_ctx: usize, sliding_band_arch: bool, is_moe: bool, long_ctx_gated: bool) usize {
    const head_dim = score_head_dim;
    if (head_dim <= 128 or n_heads == 0 or total_ctx == 0) return base_chunk;
    // Non-sliding hd-256 archs under FUSED causal (the default since the
    // budgeted-dispatch flip): no score tensor exists, so the scores-budget
    // formula below is moot — and its old shrink (1024 at 64K on 24 heads)
    // starved the dequant+GEMM qmm route, which needs M >= 2048 to engage
    // (the 64K rung was the ladder's weakest for exactly this reason).
    // MoE keeps the 4096 cap: expert-gather transients scale with the chunk
    // (the gemma-26B@99K lesson: +3% speed for +22 GB peak is a bad trade).
    // DENSE hybrids have no gather transients and a full-size chunk halves
    // the per-chunk dequant sweeps: chunk 8192 measured +1.4% over 4096 at
    // the 8K rung on Qwen3.6-27B dense (M4 Max), flat at 32K.
    // Never raises a caller-lowered base.
    if (head_dim == 256 and !sliding_band_arch and transformer_mod.fused256CausalMode() == .all) {
        // A long-context-gated MoE prices the 8192 rung per request in `chooseRequestPrefillChunk`
        // (+8% prefill for +3 GB peak here); every other MoE keeps the measured 4096 cap.
        return @min(base_chunk, if (is_moe and !long_ctx_gated) @as(usize, 4096) else @as(usize, 8192));
    }
    // Composed-causal fallback (no fused causal kernel): SMALL chunks
    // measured strictly faster AND lighter on the 27B ( ladder,
    // M4 Max): 8K 225 -> 235.8 tok/s and peak 28.9 -> 19.8 GB at chunk
    // 2048; 32K 205.4 -> 209.3. Chunk boundaries ARE block-level causal
    // skipping for composed attention. Sliding-band archs (gemma) keep big
    // chunks — their local layers run the fused band kernel, which
    // block-skips in-kernel and wants the fewest KV re-walks (26B@99K:
    // 712 tok/s at the formula chunk).
    const causal_cap: usize = if (sliding_band_arch or head_dim != 256) base_chunk else @min(base_chunk, 2048);
    const per_row: u64 = @as(u64, n_heads) * @as(u64, total_ctx) * 2;
    const max_chunk: u64 = PREFILL_SCORES_BUDGET_BYTES / per_row;
    if (max_chunk >= causal_cap) return causal_cap;
    const floored = @max(
        @as(u64, PREFILL_CHUNK_FLOOR),
        max_chunk - (max_chunk % PREFILL_CHUNK_FLOOR),
    );
    return @intCast(@min(floored, @as(u64, causal_cap)));
}

/// The prefill chunk `initWithOptions` will actually use for a request:
/// --prefill-chunk / default, capped by
/// boundedPrefillChunk. Exported so server.zig's admission guard
/// (checkAttentionMemory) models the SAME chunk the prefill will run with —
/// the guard and the real prefill must not drift.
pub fn effectivePrefillChunk(head_dim: u32, n_heads: u32, total_ctx: usize, sliding_band_arch: bool, is_moe: bool, long_ctx_gated: bool, pinned_chunk: usize) usize {
    // `pinned_chunk` is the machine-sized cap frozen at load
    // (`ModelConfig.pinned_prefill_chunk`, from `server.resolvePrefillChunk`):
    // the widest chunk whose one-off transient reserve still leaves this box a
    // real KV budget. It NARROWS, never raises, and an explicit
    // `--prefill-chunk` outranks it — the operator asked for a width.
    const launch: usize = if (head_dim == 0 and !generate_mod.prefillCli().explicit) WIDE_PREFILL_CHUNK else generate_mod.prefillCli().override;
    const base: usize = if (generate_mod.prefillCli().explicit)
        launch
    else if (pinned_chunk == 0)
        @min(launch, generate_mod.prefillCli().machine_cap)
    else
        @min(launch, pinned_chunk, generate_mod.prefillCli().machine_cap);
    return boundedPrefillChunk(base, head_dim, n_heads, total_ctx, sliding_band_arch, is_moe, long_ctx_gated);
}

/// A prefill width under the decode share's cap (0 = no cap).
pub fn decodeShareCapped(width: usize, cap: usize) usize {
    return if (cap == 0) width else @min(width, cap);
}

/// Pick the end position of the next prefill chunk starting at `pos`.
///
/// Base behavior: advance by `default_chunk` (the memory-bound `PREFILL_CHUNK`),
/// clamped to `prefix_len`. When SSM checkpointing is active, shrink the chunk so
/// it ends exactly on the next `ssm_cp_stride`-aligned ABSOLUTE position — that
/// lays down a stride-aligned SSM snapshot without changing what the model sees
/// (attention is causal; SSM/conv update chunk-locally, so the forward result is
/// identical to an unchunked run).
///
/// Pulled out of the prefill loop so the chunk-count behavior is unit-testable:
/// the stride directly controls how many chunks a cold prefill costs, and on
/// large MoE/hybrid models each extra chunk re-streams the (huge) expert weights
/// from HBM — the dominant cold-prefill cost. A too-small stride therefore
/// silently tanks MoE prefill throughput (~25% on 35B-class models for an
/// 850-token prompt at stride 256). Keeping typical prompts single-chunk is what
/// `ssm_checkpoint_stride`'s default guards.
pub const TAIL_MERGE_MAX = @import("../../../sdk/bases/run.zig").TAIL_MERGE_MAX;

/// The tail a chunk of `default_chunk` tokens may absorb where the width is chosen per chunk:
/// an eighth of the width keeps the flat 512's ~6% bound at every rung (at the ladder floor
/// 512 is +100% of the transient a step-down just bought). Reach it through `tailMergeMaxFor`.
pub fn tailMergeMax(default_chunk: usize) usize {
    return @min(TAIL_MERGE_MAX, @max(default_chunk / 8, 1));
}

/// The tail bound the chunk loop uses: scaled only under the per-chunk adaptive width, the
/// flat constant everywhere else (chunk boundaries are not byte-stable).
pub fn tailMergeMaxFor(default_chunk: usize, adaptive_width: bool) usize {
    return if (adaptive_width) tailMergeMax(default_chunk) else TAIL_MERGE_MAX;
}

pub fn nextChunkEnd(
    pos: usize,
    prefix_len: usize,
    default_chunk: usize,
    want_ssm_cp: bool,
    ssm_cp_stride: usize,
    ssm_cp_offset: usize,
    // Is the per-chunk adaptive width live for this prefill? Only then does the merge bound scale.
    adaptive_width: bool,
    // Does this arch get the dense low anchors? Same gate as the checkpoint-retention
    // `ssm_low_anchors` (`longCtxGated`, #794). A non-gated arch keeps its previous
    // stride-boundary chunking byte-for-byte: a coarsened stride must not buy
    // extra MoE chunks (each one re-streams the expert weights).
    dense_low_grid: bool,
) usize {
    var end = @min(pos + default_chunk, prefix_len);
    if (want_ssm_cp and ssm_cp_stride > 0) {
        const abs_pos = pos + ssm_cp_offset;
        const abs_end = end + ssm_cp_offset;
        const next_boundary_abs = ((abs_pos / ssm_cp_stride) + 1) * ssm_cp_stride;
        // A stride boundary lands inside this chunk — end exactly on it
        // (never tail-merge past it; the boundary IS the snapshot point).
        var cut_abs: usize = if (next_boundary_abs > abs_pos and next_boundary_abs < abs_end) next_boundary_abs else 0;
        // A dense low anchor inside the chunk cuts there too: a snapshot may only
        // stamp an evaluated position, and an anchor earlier than the stride
        // boundary must win the cut or a wide chunk jumps over it. 0 for a
        // non-gated arch, and whenever stride <= 8192 — there the grid already
        // has this spacing and fine-stride chunking stays byte-identical.
        const low_anchor = if (dense_low_grid) ssmLowAnchorEnd(pos, end, ssm_cp_offset, ssm_cp_stride) else @as(usize, 0);
        if (low_anchor > pos and low_anchor < end and (cut_abs == 0 or low_anchor + ssm_cp_offset < cut_abs))
            cut_abs = low_anchor + ssm_cp_offset;
        if (cut_abs != 0) return cut_abs - ssm_cp_offset;
    }
    if (end < prefix_len and prefix_len - end < tailMergeMaxFor(default_chunk, adaptive_width)) {
        // Absorb a tiny tail instead of paying a full graph build + eval
        // barrier for a few tokens. With checkpointing active this can only
        // extend within a boundary-free span (the boundary case returned
        // above), so at most a snapshot < TAIL_MERGE_MAX tokens before the
        // end is skipped — and the always-on end-of-prompt snapshot lands
        // right there anyway.
        end = prefix_len;
    }
    return end;
}

/// Tokens held back from the chunked-prefill loop and forwarded together with
/// the final (logits) forward when SSM checkpointing is active, so the
/// always-on snapshot lands SSM_SNAPSHOT_BACKOFF tokens BEFORE the prompt end.
///
/// A snapshot exactly at the prompt end is unreachable for the next turn's
/// prefix match: the template's generation-prompt suffix
/// ("<|im_start|>assistant\n" + think opener) renders differently once the
/// turn enters history, so the match always falls a few tokens short of the
/// full prompt — "[hot-cache] hybrid miss (no checkpoint ≤ 870 of 897)" was
/// llmprobe's prompt-cache-prefix cell failing (the class; fine
/// strides used to mask it by laying boundaries underneath). 30 covers every
/// template suffix in the fleet with margin (ChatML+think ≈ 7 tokens, laguna
/// pre-opened think ≈ 10) while keeping the tail forward UNDER the
/// prefill-eval-cadence threshold (seq >= 32): a 65-token tail was treated as
/// a prefill and paid ~450ms of mid-loop eval bubbles on the 27B, where the
/// 31-token tail rides the verify-shaped fast path. Cold cost is then ~zero
/// (the final forward pays its full weight sweep for ONE token anyway).
/// Warm restores re-forward ≤ backoff+1 tokens.
pub const SSM_SNAPSHOT_BACKOFF: usize = 30;

/// How many trailing prompt tokens the final (logits) forward covers: the
/// held-back snapshot window plus the last token itself. Pure so the
/// backoff/loop-bound interaction is unit-testable.
///
/// A restored tail that fits the window (`restored`: the prefill resumes
/// from a checkpoint) IS the cold run's final span, so it forwards as one
/// span of the same rows: a 30 + 1 split reads different kernel tilings and
/// greedy flips at a bf16 near-tie. No new snapshot: the restored one is it.
pub fn ssmSnapshotBackoff(want_ssm_cp: bool, prefix_len: usize, restored: bool) usize {
    if (!want_ssm_cp) return 0;
    if (prefix_len <= SSM_SNAPSHOT_BACKOFF) return if (restored) prefix_len else 0;
    return SSM_SNAPSHOT_BACKOFF;
}

/// Under `longCtxGated` the chunk loop lays dense checkpoint anchors every 8192
/// through the first 65536, whatever stride the request coarsened to (#794): a
/// deep entry's grid then always answers `highestSsmPosAtOrBelow` for a fork
/// inside the low band. At stride <= 8192 the grid already has this spacing, so
/// every path below is a no-op there. Every other arch passes `dense_low_grid`
/// false and keeps its previous grid.
pub const SSM_GRID_LOW_DENSE_STRIDE: usize = 8192;
pub const SSM_GRID_LOW_DENSE_TOP: usize = 65_536;

/// PURE: must the chunk loop snapshot a chunk that ends exactly at absolute
/// position `abs_end`? The stride multiples, plus the dense low anchors when
/// `dense_low_grid`. A snapshot may only stamp a position the chunk actually
/// evaluated, so the anchors are realised by cutting the chunk at them in
/// `nextChunkEnd` — never by stamping a future state at an earlier position.
pub fn ssmCapturePos(stride: usize, abs_end: usize, dense_low_grid: bool) bool {
    if (stride == 0) return false;
    if (abs_end % stride == 0) return true;
    return dense_low_grid and abs_end % SSM_GRID_LOW_DENSE_STRIDE == 0 and abs_end <= SSM_GRID_LOW_DENSE_TOP;
}

/// PURE: end the chunk at the first dense low anchor strictly inside
/// `(pos, end)` (relatives to `prompt_ids[0]`, i.e. what `nextChunkEnd`
/// returns). 0 when there is none or the stride is already dense — chunk
/// boundaries stay byte-identical at stride <= 8192.
pub fn ssmLowAnchorEnd(pos: usize, end: usize, offset: usize, stride: usize) usize {
    if (stride <= SSM_GRID_LOW_DENSE_STRIDE) return 0;
    const abs_pos = pos + offset;
    const abs_end = end + offset;
    const a: usize = (abs_pos / SSM_GRID_LOW_DENSE_STRIDE + 1) * SSM_GRID_LOW_DENSE_STRIDE;
    if (a > SSM_GRID_LOW_DENSE_TOP or a <= abs_pos or a >= abs_end) return 0;
    return a - offset;
}

/// Effective SSM-checkpoint stride for a model, given the base (configured)
/// stride: checkpointing never sub-divides the prefill chunk, on ANY arch.
///
/// The old policy kept the fine base stride on dense (non-MoE) hybrids on the
/// theory that their prefill is compute-bound so extra chunks are ~free. That
/// was true before `prefillDqGemm`: since the dq route only engages at
/// M >= 2048, a fine stride pushes EVERY projection of EVERY chunk onto the
/// slow small-M qmm path, and the per-chunk fixed costs (graph build, eval
/// barrier, MTP history capture) multiply on top. Measured on Qwen3.6-27B
/// dense GDN (M4 Max): stride 256 chunked an 8K prompt into 33
/// pieces at 211 tok/s vs 254 at coarse chunks — a 17-20% cold-prefill tax at
/// every context length, which is how llm_context_benchmarks read us 19%
/// behind oMLX. MoE pays even more (expert re-streaming, ~25%).
///
/// Warm mid-prompt reuse granularity drops to the chunk size, but the
/// always-on end-of-prompt snapshot still covers the dominant append-growth
/// multi-turn case (llmprobe's cache-hit tests restore from it at any
/// stride). `base == 0` (checkpointing disabled) is preserved, and a larger
/// explicit stride is never shrunk.
pub fn effectiveSsmCheckpointStride(base: usize, prefill_chunk: usize) usize {
    if (base == 0) return 0;
    return @max(base, prefill_chunk);
}

/// Tokens of KV capacity this prefill reserves up front (#353). A multi-chunk prompt reserves
/// itself plus one chunk of slack on every arch, since each chunk-by-chunk grow copies the whole
/// cache; generation headroom is pre-bought only under `longCtxGated`.
pub fn reservedPrefillTokens(
    config: *const model_mod.ModelConfig,
    seq: u64,
    max_tokens: u64,
    chunk: u64,
) u64 {
    const prompt = if (seq > chunk) seq +| chunk else 0;
    if (!config.longCtxGated()) return prompt;
    return @max(prompt, transformer_mod.KVCache.reservedTokens(seq, max_tokens, chunk, config.max_position_embeddings));
}

/// A prompt checkpoints SSM state at its chunk boundaries, a vision prompt included (it chunks like text, #197):
/// without a checkpoint a hybrid's image turn is a guaranteed hot-cache miss.
pub fn shouldCheckpointSsmPrefill(stride: u32, has_ssm: bool) bool {
    return stride > 0 and has_ssm;
}

/// Placeholder tokens (vision + audio soft tokens) in `ids` — the number of
/// source rows a chunk's splice consumes. Host-side count, no GPU sync.
pub fn countSpliceRows(ids: []const i32, image_token_id: u32, audio_token_id: u32, video_token_id: u32) usize {
    var n: usize = 0;
    for (ids) |id| {
        if ((image_token_id > 0 and id == @as(i32, @intCast(image_token_id))) or
            (audio_token_id > 0 and id == @as(i32, @intCast(audio_token_id))) or
            (video_token_id > 0 and id == @as(i32, @intCast(video_token_id)))) n += 1;
    }
    return n;
}

/// Number of chunks a cold prefill of `prefix_len` tokens splits into for the
/// given chunk size / SSM-checkpoint stride. Mirrors the loop in `init` exactly
/// (drives the same `nextChunkEnd`), so a test on this is a faithful proxy for
/// the real prefill chunk count. Each chunk on a memory-bound MoE prefill
/// re-streams the expert weights, so this is effectively the cold-prefill
/// weight-traffic multiplier.
pub fn prefillChunkCount(
    prefix_len: usize,
    default_chunk: usize,
    want_ssm_cp: bool,
    ssm_cp_stride: usize,
    ssm_cp_offset: usize,
    adaptive_width: bool,
    dense_low_grid: bool,
) usize {
    var pos: usize = 0;
    var n: usize = 0;
    while (pos < prefix_len) {
        const end = nextChunkEnd(pos, prefix_len, default_chunk, want_ssm_cp, ssm_cp_stride, ssm_cp_offset, adaptive_width, dense_low_grid);
        pos = end;
        n += 1;
    }
    return n;
}

/// Generated tokens between two returns of the decode loop's transients to
/// MLX's allocator. 256 is what the non-speculative paths have always used.
pub const CACHE_CLEAR_INTERVAL: u32 = 256;

/// PURE: has `interval` tokens passed since the last clear?
///
/// Interval arithmetic, not `step % interval == 0`: a spec-decode round emits
/// `1 + accepted` tokens, so a modulo test can step clean over every multiple
/// and never fire (issue #110 — at stride 5 it fires zero times in the first
/// 1024 steps). For the stride-1 paths this is byte-identical to the modulo
/// form it replaces.
pub fn shouldClearAllocatorCache(step: u32, last_clear: u32, interval: u32) bool {
    if (interval == 0) return false;
    return step -| last_clear >= interval;
}

/// PURE: does this request return MLX's allocator pool once, between its last
/// prefill chunk and its first decode tick?
///
/// The per-chunk clear runs before the last chunks' transient is freed, so that
/// transient parks in the pool up to the cap and the first decode tick allocates
/// on top of it (8.1 GB parked at the first tick of a 393k prefill). Gated on
/// `longCtxGated`: every other arch keeps its previous call pattern exactly.
pub fn clearsPoolAtPrefillEnd(config: *const model_mod.ModelConfig) bool {
    return config.longCtxGated();
}

// ── tests ──


test "decodeShareCapped: the share cap narrows an explicit and an env width" {
    const saved_override = generate_mod.prefillCli().override;
    const saved_explicit = generate_mod.prefillCli().explicit;
    defer {
        generate_mod.prefillCli().override = saved_override;
        generate_mod.prefillCli().explicit = saved_explicit;
    }
    generate_mod.prefillCli().override = 8192;
    generate_mod.prefillCli().explicit = true;
    const explicit = effectivePrefillChunk(128, 8, 1024, false, false, false, 2048);
    try std.testing.expectEqual(@as(usize, 8192), explicit);
    try std.testing.expectEqual(@as(usize, 1024), decodeShareCapped(explicit, 1024));
    try std.testing.expectEqual(@as(usize, 8192), decodeShareCapped(explicit, 0));
}

test "effectivePrefillChunk: the machine cap bounds every auto width, an explicit width passes" {
    const saved_cap = generate_mod.prefillCli().machine_cap;
    const saved_explicit = generate_mod.prefillCli().explicit;
    const saved_override = generate_mod.prefillCli().override;
    defer {
        generate_mod.prefillCli().machine_cap = saved_cap;
        generate_mod.prefillCli().explicit = saved_explicit;
        generate_mod.prefillCli().override = saved_override;
    }
    generate_mod.prefillCli().machine_cap = 2048;
    generate_mod.prefillCli().explicit = false;
    generate_mod.prefillCli().override = WIDE_PREFILL_CHUNK;
    // A per-request rung (8192) and an unpinned adaptive cap (0) both stop at the machine's cap.
    try std.testing.expectEqual(@as(usize, 2048), effectivePrefillChunk(256, 24, 16384, false, true, true, 8192));
    try std.testing.expectEqual(@as(usize, 2048), effectivePrefillChunk(256, 24, 16384, false, true, true, 0));
    generate_mod.prefillCli().explicit = true;
    generate_mod.prefillCli().override = 8192;
    try std.testing.expectEqual(@as(usize, 8192), effectivePrefillChunk(256, 24, 16384, false, true, true, 8192));
}

/// Walk a chunked prefill as `runPrefill` does, taking the width from `widths` at each boundary (the last entry repeats).
fn walkChunkEnds(
    out: []usize,
    prefix_len: usize,
    widths: []const usize,
    want_ssm_cp: bool,
    stride: usize,
    offset: usize,
) []usize {
    var pos: usize = 0;
    var n: usize = 0;
    while (pos < prefix_len and n < out.len) {
        const w = widths[@min(n, widths.len - 1)];
        const end = nextChunkEnd(pos, prefix_len, w, want_ssm_cp, stride, offset, true, false);
        out[n] = end;
        n += 1;
        pos = end;
    }
    return out[0..n];
}

test "a width that changes at every boundary never moves a checkpoint" {
    // `ssm_cp_stride` derives from the launch width and every ladder rung divides it, so a
    // mixed-width prefill hits the same checkpoint positions as a fixed-width one.
    const t = testing;
    const mixed = [_]usize{ 4096, 4096, 2048, 1024, 512, 512, 1024, 2048, 4096, 512, 2048, 1024, 4096, 512 };
    const fixed = [_]usize{4096};

    for ([_]usize{ 8192, 4096 }) |stride| {
        for ([_]usize{ 0, 1000 }) |offset| {
            for ([_]usize{ 40960, 40000 }) |prefix_len| {
                var fb: [512]usize = undefined;
                var mb: [512]usize = undefined;
                const f = walkChunkEnds(&fb, prefix_len, &fixed, true, stride, offset);
                const m = walkChunkEnds(&mb, prefix_len, &mixed, true, stride, offset);

                for ([_][]usize{ f, m }) |ends| {
                    var pos: usize = 0;
                    for (ends) |end| {
                        try t.expect(end > pos);
                        try t.expect(end <= prefix_len);
                        // No chunk straddles a stride boundary.
                        const abs_pos = pos + offset;
                        const abs_end = end + offset;
                        const next_b = ((abs_pos / stride) + 1) * stride;
                        if (end != prefix_len) try t.expect(!(next_b > abs_pos and next_b < abs_end));
                        pos = end;
                    }
                    try t.expectEqual(prefix_len, pos);
                }

                var fi: usize = 0;
                var mi: usize = 0;
                while (true) {
                    while (fi < f.len and (f[fi] + offset) % stride != 0) fi += 1;
                    while (mi < m.len and (m[mi] + offset) % stride != 0) mi += 1;
                    if (fi >= f.len or mi >= m.len) break;
                    try t.expectEqual(f[fi], m[mi]);
                    fi += 1;
                    mi += 1;
                }
                while (fi < f.len and (f[fi] + offset) % stride != 0) fi += 1;
                while (mi < m.len and (m[mi] + offset) % stride != 0) mi += 1;
                try t.expectEqual(fi >= f.len, mi >= m.len);
            }
        }
    }
}

test "every ladder rung the adaptive width can take divides the checkpoint stride" {
    const t = testing;
    const stride = effectiveSsmCheckpointStride(256, generate_mod.prefillCli().override);
    try t.expect(stride > 0);
    for ([_]usize{ 8192, 4096, 2048, 1024, PREFILL_CHUNK_FLOOR }) |rung| {
        try t.expect(rung <= stride);
        try t.expectEqual(@as(usize, 0), stride % rung);
    }
}

test "commitAdaptiveWidth: the summary follows the running width from both commit sites" {
    const t = testing;
    var cur: usize = 2048;
    var st: AdaptiveWidthState = .{ .width_min = 2048, .width_max = 2048 };
    commitAdaptiveWidth(&cur, &st, 1024);
    try t.expectEqual(@as(usize, 1024), cur);
    try t.expectEqual(@as(u32, 1), st.transitions);
    try t.expectEqual(@as(u32, 1024), st.width_min);
    try t.expectEqual(@as(u32, 2048), st.width_max);
    commitAdaptiveWidth(&cur, &st, 4096);
    try t.expectEqual(@as(usize, 4096), cur);
    try t.expectEqual(@as(u32, 2), st.transitions);
    try t.expectEqual(@as(u32, 1024), st.width_min);
    try t.expectEqual(@as(u32, 4096), st.width_max);
}

test "tailMergeMax: the tail a chunk may absorb scales with the width" {
    const t = testing;
    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMax(8192)); // no-op where it was justified
    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMax(4096));
    try t.expectEqual(@as(usize, 256), tailMergeMax(2048));
    try t.expectEqual(@as(usize, 128), tailMergeMax(1024));
    try t.expectEqual(@as(usize, 64), tailMergeMax(PREFILL_CHUNK_FLOOR));
    try t.expectEqual(@as(usize, 1), tailMergeMax(1));
    for ([_]usize{ 8192, 4096, 2048, 1024, PREFILL_CHUNK_FLOOR }) |w| {
        try t.expect(tailMergeMax(w) * 8 <= w);
    }

    try t.expectEqual(@as(usize, 8492), nextChunkEnd(0, 8492, 8192, false, 0, 0, true, false));
    try t.expectEqual(@as(usize, 512), nextChunkEnd(0, 812, PREFILL_CHUNK_FLOOR, false, 0, 0, true, false));
}

test "the scaled tail-merge bound is gated on the per-chunk adaptive width" {
    // The scaled bound is gated: every arch whose resolved chunk is under 4096 would otherwise
    // merge a smaller tail and run an extra chunk.
    const t = testing;

    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMaxFor(PREFILL_CHUNK_FLOOR, false));
    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMaxFor(1024, false));
    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMaxFor(2048, false));
    try t.expectEqual(@as(usize, 64), tailMergeMaxFor(PREFILL_CHUNK_FLOOR, true));
    try t.expectEqual(@as(usize, 256), tailMergeMaxFor(2048, true));
    try t.expectEqual(tailMergeMaxFor(4096, false), tailMergeMaxFor(4096, true));
    try t.expectEqual(tailMergeMaxFor(8192, false), tailMergeMaxFor(8192, true));

    try t.expectEqual(@as(usize, 812), nextChunkEnd(0, 812, PREFILL_CHUNK_FLOOR, false, 0, 0, false, false));
    try t.expectEqual(@as(usize, 512), nextChunkEnd(0, 812, PREFILL_CHUNK_FLOOR, false, 0, 0, true, false));
    try t.expectEqual(@as(usize, 1324), nextChunkEnd(0, 1324, 1024, false, 0, 0, false, false));
    try t.expectEqual(@as(usize, 1024), nextChunkEnd(0, 1324, 1024, false, 0, 0, true, false));
    try t.expectEqual(@as(usize, 8492), nextChunkEnd(0, 8492, 8192, false, 0, 0, false, false));
    try t.expectEqual(@as(usize, 8492), nextChunkEnd(0, 8492, 8192, false, 0, 0, true, false));

    try t.expectEqual(@as(usize, 1), prefillChunkCount(812, PREFILL_CHUNK_FLOOR, false, 0, 0, false, false));
    try t.expectEqual(@as(usize, 2), prefillChunkCount(812, PREFILL_CHUNK_FLOOR, false, 0, 0, true, false));
}

test "nextChunkEnd: a tiny trailing remainder merges into the last chunk" {
    // A chat-templated prompt often lands a token or two past the chunk size
    // (8192-target prompts tokenize to 8193). A 1-token trailing chunk pays a
    // FULL graph + eval-barrier + cache-clear for one token — pure overhead.
    // Without checkpoint alignment, remainders under the merge floor extend
    // the current chunk instead.
    try testing.expectEqual(@as(usize, 8193), nextChunkEnd(0, 8193, 8192, false, 0, 0, false, false));
    // A substantial remainder stays its own chunk.
    try testing.expectEqual(@as(usize, 8192), nextChunkEnd(0, 8192 + 600, 8192, false, 0, 0, false, false));
    // Mid-prompt chunks are untouched.
    try testing.expectEqual(@as(usize, 8192), nextChunkEnd(0, 16385, 8192, false, 0, 0, false, false));
    try testing.expectEqual(@as(usize, 16385), nextChunkEnd(8192, 16385, 8192, false, 0, 0, false, false));
    // With SSM-checkpoint alignment active, a tiny tail STILL merges: the old
    // 1-token trailing chunk existed only to lay a snapshot one token before
    // the always-on end-of-prompt snapshot — pure overhead. A boundary strictly
    // INSIDE the chunk still wins over merging (next case).
    try testing.expectEqual(@as(usize, 8193), nextChunkEnd(0, 8193, 8192, true, 8192, 0, false, false));
    try testing.expectEqual(@as(usize, 4096), nextChunkEnd(0, 8193, 8192, true, 4096, 0, false, false));
}

test "prefillChunkCount: SSM-checkpoint stride controls cold-prefill chunking" {
    const PREFILL_CHUNK: usize = 8192;
    // Non-hybrid (or checkpointing off): a sub-PREFILL_CHUNK prompt is ONE chunk.
    try testing.expectEqual(@as(usize, 1), prefillChunkCount(851, PREFILL_CHUNK, false, 0, 0, false, false));
    try testing.expectEqual(@as(usize, 1), prefillChunkCount(8000, PREFILL_CHUNK, false, 0, 0, false, false));
    // Tail merge: one token past a chunk boundary is still ONE chunk.
    try testing.expectEqual(@as(usize, 1), prefillChunkCount(8193, PREFILL_CHUNK, false, 0, 0, false, false));
    try testing.expectEqual(@as(usize, 2), prefillChunkCount(16385, PREFILL_CHUNK, false, 0, 0, false, false));
    // Mechanically, a raw fine stride still splits an 851-token prefill into 4
    // chunks (851 spans boundaries 256/512/768) — which is why
    // effectiveSsmCheckpointStride coarsens every stride to the prefill chunk:
    // per-chunk costs (expert re-streaming on MoE, sub-dq-gemm-floor GEMMs +
    // fixed overhead everywhere) taxed cold prefill 17-25%.
    try testing.expectEqual(@as(usize, 4), prefillChunkCount(851, PREFILL_CHUNK, true, 256, 0, false, false));
    // Boundary alignment is ABSOLUTE (warm path passes an offset): a tail-only
    // prefill starting mid-sequence still snaps to global strides. offset=2000,
    // prefix tail of 200 (abs 2000..2200), stride 256 -> boundary 2048/2304? only
    // 2048 falls inside (2000..2200) -> 2 chunks.
    try testing.expectEqual(@as(usize, 2), prefillChunkCount(200, PREFILL_CHUNK, true, 256, 2000, false, false));
}

test "the gated dense low grid anchors the first 65536 under a coarse stride" {
    // Gated on `longCtxGated` (#794): the capture predicate is stride multiples
    // plus the 8192 anchors under the dense low top, so a deep entry's grid
    // always answers a restore point for a fork inside the low band.
    try testing.expect(ssmCapturePos(32_768, 8_192, true));
    try testing.expect(ssmCapturePos(32_768, 24_576, true));
    try testing.expect(ssmCapturePos(32_768, 32_768, true)); // stride multiple — same point as before
    try testing.expect(ssmCapturePos(32_768, 65_536, true)); // the top itself still counts
    try testing.expect(!ssmCapturePos(32_768, 40_959, true));
    try testing.expect(!ssmCapturePos(32_768, 73_728, true)); // the high band keeps the coarse spacing
    try testing.expect(!ssmCapturePos(0, 8_192, false)); // checkpointing off stays off
    // A stride at or under 8192 (v8's dense grid) captures EXACTLY what it always
    // did: the dense rule adds no point to a grid that already has this spacing.
    for ([_]usize{ 256, 512, 1024, 2048, 4096, 8192 }) |fine| {
        var x: usize = 0;
        while (x <= 70_000) : (x += 1_023) {
            try testing.expectEqual(x % fine == 0, ssmCapturePos(fine, x, true));
        }
    }
    // Chunking drives the grid: a cold 40k prefill at width 4096 under stride
    // 32768 (the production boot's coarsening) now snapshots exactly the four low
    // points — driven through `nextChunkEnd`, the same fn the real prefill loop
    // runs. (A prompt ending within the tail-merge floor of a multiple merges
    // that boundary into the always-on end-of-prompt snapshot instead — the
    // end snapshot at `prefix_len - 30` covers every fork the swallowed point
    // could have served.)
    var grid: [64]u32 = undefined;
    var n: usize = 0;
    var pos: usize = 0;
    while (true) {
        const end = nextChunkEnd(pos, 40_000, 4_096, true, 32_768, 0, false, true);
        if (ssmCapturePos(32_768, end, true)) {
            grid[n] = @intCast(end);
            n += 1;
        }
        pos = end;
        if (pos >= 40_000) break;
    }
    try testing.expectEqual(@as(usize, 4), n);
    try testing.expectEqual(@as(u32, 8_192), grid[0]);
    try testing.expectEqual(@as(u32, 16_384), grid[1]);
    try testing.expectEqual(@as(u32, 24_576), grid[2]);
    try testing.expectEqual(@as(u32, 32_768), grid[3]);
    // The coarse-stride prefill chunks EXACTLY like the v8 fine grid for this
    // prompt: same boundaries, so the dense anchors cost zero extra chunks.
    try testing.expectEqual(
        prefillChunkCount(40_000, 4_096, true, 8_192, 0, false, true),
        prefillChunkCount(40_000, 4_096, true, 32_768, 0, false, true),
    );
    // A chunk WIDER than the anchor spacing gets cut at the anchor (the snapshot
    // may only stamp evaluated positions): a 32k-wide chunk from 6k ends at 8192.
    try testing.expectEqual(@as(usize, 8_192), nextChunkEnd(6_000, 40_000, 32_768, true, 32_768, 0, false, true));
    try testing.expectEqual(@as(usize, 16_384), nextChunkEnd(8_192, 40_000, 32_768, true, 32_768, 0, false, true));
    // Inside the dense band the FIRST anchor wins the cut, ahead of any later
    // stride boundary: a 32k-wide chunk from 40k ends at 40960, not 65536.
    try testing.expectEqual(@as(usize, 40_960), nextChunkEnd(40_000, 100_000, 32_768, true, 32_768, 0, false, true));
    // Past the top the cut stops and the high band keeps the coarse stride alone.
    try testing.expectEqual(@as(usize, 98_304), nextChunkEnd(65_536, 100_000, 32_768, true, 32_768, 0, false, true));
    // Fine strides never get an extra cut: the helper answers 0 at every stride
    // the dense rule is inactive for, whatever the chunk geometry.
    for ([_]usize{ 0, 256, 512, 1024, 2048, 4096, 8192 }) |fine| {
        var p: usize = 1_234;
        while (p < 70_000) : (p += 3_037) {
            try testing.expectEqual(@as(usize, 0), ssmLowAnchorEnd(p, p + 4_096, 61, fine));
        }
    }
}

test "the dense low grid is gated: a non-gated arch at width 16384 keeps main's chunk count" {
    // Same gate as the retention `ssm_low_anchors`: a non-gated hybrid whose
    // stride coarsened to 16384 keeps its stride-boundary chunking — a warm
    // restore landing off the 8192 grid must not get its chunks cut at every
    // multiple below 65536 (each extra MoE chunk re-streams the experts).
    // `main`'s chunk loop, re-derived here: cut at a stride boundary inside
    // the chunk, else absorb a tail under `TAIL_MERGE_MAX`.
    const t = testing;
    const stride = effectiveSsmCheckpointStride(2048, 16_384);
    try t.expectEqual(@as(usize, 16_384), stride);
    const offset = 4_096; // off the anchor grid: a warm restore mid-sequence
    const prefix_len: usize = 100_000 - offset;
    var main_chunks: usize = 0;
    var pos: usize = 0;
    while (pos < prefix_len) {
        var end: usize = @min(pos + stride, prefix_len);
        const abs_pos: usize = pos + offset;
        const abs_end: usize = end + offset;
        const nb: usize = ((abs_pos / stride) + 1) * stride;
        if (nb > abs_pos and nb < abs_end) end = nb - offset;
        if (end < prefix_len and prefix_len - end < TAIL_MERGE_MAX) end = prefix_len;
        main_chunks += 1;
        pos = end;
    }
    // Ungated: the count is main's, at fixed and adaptive width alike.
    try t.expect(main_chunks > 0);
    try t.expectEqual(main_chunks, prefillChunkCount(prefix_len, stride, true, stride, offset, false, false));
    try t.expectEqual(main_chunks, prefillChunkCount(prefix_len, stride, true, stride, offset, true, false));
    // Gated (the long-context line): the anchor cuts are paid — the first
    // wide chunk is cut at abs 8192, and the anchor point captures.
    try t.expectEqual(@as(usize, 4_096), nextChunkEnd(0, prefix_len, stride, true, stride, offset, false, true));
    try t.expect(!ssmCapturePos(stride, 8_192, false));
    try t.expect(ssmCapturePos(stride, 8_192, true));
}

test "boundedPrefillChunk: fused head dims and short contexts keep the base chunk" {
    // head_dim <= 128 rides MLX's fused SDPA — no materialized scores, no cap,
    // at ANY context length.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 128, 16, 1_000_000, true, false, false));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 64, 32, 1_000_000, true, false, false));
    // hd 256 but short context: 16 heads x 8192 ctx x 8192 chunk x 2B
    // = 2 GiB scores, inside the 4 GiB budget -> full chunk kept. This is the
    // fleet-protection property: every Gemma-4 / Qwen3.5/3.6 checkpoint ships
    // head_dim 256, so typical prompts must keep full prefill throughput.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 16, 8192, true, false, false));
    // Degenerate inputs never cap.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 0, 100_000, true, false, false));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 16, 0, true, false, false));
}

test "boundedPrefillChunk: caps hd-256 long context even with the fused kernel active" {
    // The msv_attn_p256 kernel removes the SCORE transient, but a big chunk
    // still scales the MoE-gather / KV-concat transients — measured +22 GB
    // peak for +3% speed at a 99K prompt. The cap deliberately ignores
    // prefillHeadDimFused (see the fn doc); pin that with the override ON.
    attn256.fused256_override = true;
    defer attn256.fused256_override = null;
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(8192, 256, 16, 100_000, true, false, false));
}

test "boundedPrefillChunk: long context shrinks to the scores budget, floored and rounded" {
    // gemma-4-26B geometry (16 heads): budget/(16*ctx*2) …
    // ctx 32768 -> exactly 4096.
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 16, 32768, true, false, false));
    // ctx 100000 -> raw 1342, rounded down to the 512 grain -> 1024.
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(8192, 256, 16, 100_000, true, false, false));
    // ctx 262144 (the PR-#69 255K case) -> 512.
    try testing.expectEqual(@as(usize, 512), boundedPrefillChunk(8192, 256, 16, 262_144, true, false, false));
    // Qwen3.6-27B geometry (24 heads) at 262144: raw 341 -> floor 512.
    try testing.expectEqual(@as(usize, 512), boundedPrefillChunk(8192, 256, 24, 262_144, true, false, false));
    // e4b geometry (8 heads) at 131072: exactly 2048.
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 256, 8, 131_072, true, false, false));
}

test "boundedPrefillChunk: a 192-wide MLA score is budgeted, and the hd-256 policies stay hd-256" {
    // A hybrid MLA arch can declare head_dim 128 (its value width) so it fell
    // through the "fused SDPA covers it" early-out — while its MLA scores at
    // qk width 192 on the composed path. 32 heads.
    //
    // Short prompts keep the full chunk: 32 * 8192 * 8192 * 2B = 4 GiB exactly.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 192, 32, 8192, false, true, false));
    // Then it halves with the prompt, holding one score tensor at the budget.
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 192, 32, 16384, false, true, false));
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 192, 32, 32768, false, true, false));
    // The 38201-token prompt that measured a +28.8 GB peak at chunk 8192:
    // 4 GiB / (32 * 38201 * 2) = 1757 -> floored to the 512 grain.
    try testing.expectEqual(@as(usize, 1536), boundedPrefillChunk(8192, 192, 32, 38_201, false, true, false));
    try testing.expectEqual(@as(usize, 512), boundedPrefillChunk(8192, 192, 32, 262_144, false, true, false));

    // The two hd-256-measured policies must NOT adopt this arch:
    // - the fused-kernel branch (no score tensor) is hd-256-only,
    // - the 2048 composed cap was tuned on a 27B's own prefill ladder.
    attn256.fused256_override = true;
    defer attn256.fused256_override = null;
    // With the fused override on, a real hd-256 MoE takes the 4096 branch...
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 32, 8192, false, true, false));
    // ...and the 192-wide arch still gets its honest full chunk at the same shape.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 192, 32, 8192, false, true, false));
}

test "MTP history window: threshold gate and chunk membership" {
    // Below/at the 16384 threshold the window never engages — behavior (and
    // temp-0 output) stays byte-identical to full-history capture.
    try testing.expectEqual(@as(usize, 0), effectiveMtpHistoryWindow(1000, 8192));
    try testing.expectEqual(@as(usize, 0), effectiveMtpHistoryWindow(16384, 8192));
    try testing.expectEqual(@as(usize, 8192), effectiveMtpHistoryWindow(16385, 8192));
    try testing.expectEqual(@as(usize, 8192), effectiveMtpHistoryWindow(65536, 8192));
    // 0 = full history at any length (the --mtp-history-window 0 escape).
    try testing.expectEqual(@as(usize, 0), effectiveMtpHistoryWindow(65536, 0));

    // Chunk membership at prefix 32768, window 8192: the window starts at
    // 24576. Chunks entirely before it skip capture; the boundary chunk
    // (ending past 24576) captures WHOLE.
    try testing.expect(!chunkNeedsMtpHistory(0, 8192, 32768, 8192));
    try testing.expect(!chunkNeedsMtpHistory(16384, 24576, 32768, 8192));
    try testing.expect(chunkNeedsMtpHistory(24576, 32768, 32768, 8192));
    try testing.expect(chunkNeedsMtpHistory(20000, 24577, 32768, 8192));
    // Zero window: every chunk captures.
    try testing.expect(chunkNeedsMtpHistory(0, 8192, 32768, 0));
    // Window >= prefix degenerates to full capture (no underflow).
    try testing.expect(chunkNeedsMtpHistory(0, 512, 4096, 8192));
}

test "boundedPrefillChunk: never raises a caller-lowered base chunk" {
    // --prefill-chunk 1024 with headroom for 4096: the explicit lower value wins.
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(1024, 256, 16, 32768, true, false, false));
    // Even the floor never raises a tiny explicit base.
    try testing.expectEqual(@as(usize, 256), boundedPrefillChunk(256, 256, 16, 262_144, true, false, false));
}

test "boundedPrefillChunk: fused-causal (default) non-sliding hd-256 — MoE caps at 4096, dense keeps the full chunk" {
    // With the causal arm FUSED (default since the budgeted-dispatch flip) no
    // score tensor exists, so the scores-budget formula is moot for the
    // qwen3_5/3_6 class — and its old shrink to 1024 at 64K starved the
    // dequant+GEMM qmm route (engages at M >= 2048): the 64K rung was the
    // ladder's weakest for exactly this reason.
    std.debug.assert(attn256.fused256_override == null);
    // DENSE hybrids (Qwen3.6-27B class): no expert-gather transients, and a
    // full-size chunk halves per-chunk dequant sweeps — chunk 8192 measured
    // +1.4% over 4096 at the 8K rung (M4 Max), flat at 32K.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 8192, false, false, false));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 65536, false, false, false));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 140_000, false, false, false));
    // MoE keeps the 4096 cap: expert-gather transients scale with the chunk
    // (gemma-26B@99K: +3% speed for +22 GB peak is a bad trade).
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 24, 8192, false, true, false));
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 24, 140_000, false, true, false));
    // Never raises a caller-lowered base.
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(1024, 256, 24, 8192, false, false, false));
    // Sliding-band archs (gemma: fused band kernel wants big chunks) keep
    // the formula-only policy.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 16, 8192, true, false, false));
    // Fused head dims never cap regardless of arch.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 128, 24, 1_000_000, false, false, false));
}

test "boundedPrefillChunk: MiMo builds no prefill score tensor, so its chunk never shrinks with the prompt" {
    const cfg: model_mod.ModelConfig = .{ .model_type = "mimo_v2", .head_dim = 192, .num_attention_heads = 64, .has_sliding_window = true };
    try testing.expectEqual(@as(u32, 0), cfg.prefillScoreHeadDim());
    for ([_]usize{ 8192, 32768, 262_144 }) |ctx| try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, cfg.prefillScoreHeadDim(), 64, ctx, true, true, false));
    try testing.expect(transformer_mod.prefillHeadDimFused(cfg.prefillScoreHeadDim()));
}

test "boundedPrefillChunk: DeepSeek-V4.1 attends a window plus its index picks, so its chunk never shrinks with the prompt" {
    const cfg: model_mod.ModelConfig = .{ .model_type = "deepseek_v41", .head_dim = 512, .num_attention_heads = 64 };
    try testing.expectEqual(@as(u32, 0), cfg.prefillScoreHeadDim());
    for ([_]usize{ 16384, 65536, 131_072, 1 << 20 }) |ctx| try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(2048, cfg.prefillScoreHeadDim(), 64, ctx, false, true, false));
}

test "effectivePrefillChunk: an arch with no prefill score tensor defaults to the wide chunk, still under the pin and an explicit width" {
    const saved = .{ generate_mod.prefillCli().override, generate_mod.prefillCli().explicit };
    defer {
        generate_mod.prefillCli().override = saved[0];
        generate_mod.prefillCli().explicit = saved[1];
    }
    generate_mod.prefillCli().override = 8192;
    generate_mod.prefillCli().explicit = false;
    try testing.expectEqual(WIDE_PREFILL_CHUNK, effectivePrefillChunk(0, 64, 32768, true, true, false, 0));
    try testing.expectEqual(WIDE_PREFILL_CHUNK, effectivePrefillChunk(0, 64, 32768, true, true, false, WIDE_PREFILL_CHUNK));
    try testing.expectEqual(@as(usize, 8192), effectivePrefillChunk(0, 64, 32768, true, true, false, 8192));
    // Every other arch keeps the launch width.
    try testing.expectEqual(@as(usize, 8192), effectivePrefillChunk(128, 32, 32768, false, false, false, WIDE_PREFILL_CHUNK));
    generate_mod.prefillCli().override = 4096;
    generate_mod.prefillCli().explicit = true;
    try testing.expectEqual(@as(usize, 4096), effectivePrefillChunk(0, 64, 32768, true, true, false, WIDE_PREFILL_CHUNK));
}

test "boundedPrefillChunk: a long-context-gated MoE hd-256 arch offers the 8192 rung, an ungated one caps at 4096" {
    std.debug.assert(attn256.fused256_override == null);
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 8192, false, true, true));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 140_000, false, true, true));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 400_000, false, true, true));
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 24, 8192, false, true, false));
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 24, 140_000, false, true, false));
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(2048, 256, 24, 8192, false, true, true));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 140_000, false, false, true));
}

test "boundedPrefillChunk: composed-causal keeps the 2048 cap + score formula" {
    // Composed causal, where SMALLER
    // chunks measured faster on the 27B ladder (M4 Max): 8K
    // prompt 225 -> 235.8 tok/s at chunk 2048 (peak 28.9 -> 19.8 GB).
    // Chunking IS block-level causal skipping for composed attention, and
    // the score transient shrinks with it.
    attn256.fused256_override = false;
    defer attn256.fused256_override = null;
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 256, 24, 8192, false, false, false));
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 256, 24, 32768, false, false, false));
    // The scores-budget formula still wins BELOW the cap: 64K on 24 heads
    // yields 1024 (measured better than 2048 there: 186 vs 182.3 tok/s).
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(8192, 256, 24, 65536, false, false, false));
}

test "ssmSnapshotBackoff: engages only under checkpointing and past the backoff length" {
    // No checkpointing (pure-attention archs, stride 0, vision): zero — the
    // final forward stays the classic 1-token logits pass.
    try testing.expectEqual(@as(usize, 0), ssmSnapshotBackoff(false, 8192, false));
    // Short cold prompts: nothing to back off (loop must keep >= 1 token).
    try testing.expectEqual(@as(usize, 0), ssmSnapshotBackoff(true, SSM_SNAPSHOT_BACKOFF, false));
    try testing.expectEqual(@as(usize, 0), ssmSnapshotBackoff(true, 1, false));
    // A restored tail inside the window is the cold path's final span: one forward of
    // the same rows, no new snapshot (greedy bytes must match the cold run).
    try testing.expectEqual(SSM_SNAPSHOT_BACKOFF, ssmSnapshotBackoff(true, SSM_SNAPSHOT_BACKOFF, true));
    try testing.expectEqual(@as(usize, 1), ssmSnapshotBackoff(true, 1, true));
    try testing.expectEqual(@as(usize, 0), ssmSnapshotBackoff(true, 0, true));
    // Checkpointing + long prompt: the always-on snapshot lands backoff
    // tokens before the prompt end, where the next turn's prefix match can
    // reach it (template generation-suffix divergence class).
    try testing.expectEqual(SSM_SNAPSHOT_BACKOFF, ssmSnapshotBackoff(true, 8192, false));
    try testing.expectEqual(SSM_SNAPSHOT_BACKOFF, ssmSnapshotBackoff(true, SSM_SNAPSHOT_BACKOFF + 1, true));
    // The tail forward must stay UNDER the prefill-eval-cadence threshold
    // (seq >= 32 turns it into a "prefill" costing ~450ms of eval bubbles):
    // tail = backoff + 1 <= 31.
    try testing.expect(SSM_SNAPSHOT_BACKOFF + 1 < 32);
}

test "effectiveSsmCheckpointStride: checkpointing never sub-divides the prefill chunk (dense AND MoE)" {
    const PREFILL_CHUNK: usize = 8192;
    // Disabled stays disabled.
    try testing.expectEqual(@as(usize, 0), effectiveSsmCheckpointStride(0, PREFILL_CHUNK));
    // Sub-chunk strides coarsen on ALL archs: they push every projection under
    // prefillDqGemm's M>=2048 floor (slow small-M qmm) and multiply per-chunk
    // fixed costs. Measured on Qwen3.6-27B dense (M4 Max): stride
    // 256 = 33 chunks at 8K = 211 tok/s vs 254 coarse.
    try testing.expectEqual(@as(usize, 8192), effectiveSsmCheckpointStride(256, PREFILL_CHUNK));
    // A larger explicit stride is respected (never shrunk).
    try testing.expectEqual(@as(usize, 16384), effectiveSsmCheckpointStride(16384, PREFILL_CHUNK));
    // End-to-end: an 851-tok prefill is 1 chunk (was 4 at the raw 256 stride
    // on dense hybrids — the llm_context_benchmarks small-prompt regression).
    try testing.expectEqual(@as(usize, 1), prefillChunkCount(851, PREFILL_CHUNK, true, effectiveSsmCheckpointStride(256, PREFILL_CHUNK), 0, false, false));
    // An 8K prefill splits only at the (memory-bound) chunk size, never
    // finer: 2 chunks at chunk 4096, not 33.
    try testing.expectEqual(@as(usize, 2), prefillChunkCount(8238, 4096, true, effectiveSsmCheckpointStride(256, PREFILL_CHUNK), 0, false, false));
}

test "a vision prefill checkpoints SSM state like text" {
    const prefix_len: usize = 1587;
    const checkpoint_stride: u32 = 256;
    try testing.expect(shouldCheckpointSsmPrefill(checkpoint_stride, true));
    try testing.expect(!shouldCheckpointSsmPrefill(0, true));
    try testing.expectEqual(
        @as(usize, checkpoint_stride),
        nextChunkEnd(0, prefix_len, prefix_len, true, @intCast(checkpoint_stride), 0, false, false),
    );
}

// ── Allocator-cache clear cadence (issue #110) ───────────────────────────────

test "clear cadence survives variable spec strides" {
    // A spec-decode round emits `1 + accepted` tokens, so `step` advances by a
    // VARIABLE amount and `step % 256 == 0` can walk clean over every multiple
    // — silently, because a decode path that never clears is output-identical
    // to one that does. Interval arithmetic against the last clear cannot be
    // stepped over.
    for ([_]u32{ 1, 3, 5, 9 }) |stride| {
        var step: u32 = 0;
        var last_clear: u32 = 0;
        var clears: usize = 0;
        var max_gap: u32 = 0;
        while (step < 4096) {
            step += stride;
            if (shouldClearAllocatorCache(step, last_clear, CACHE_CLEAR_INTERVAL)) {
                const gap = step - last_clear;
                if (gap > max_gap) max_gap = gap;
                last_clear = step;
                clears += 1;
            }
        }
        try testing.expect(clears >= 15);
        // A stride can overshoot the interval by at most stride-1 tokens.
        try testing.expect(max_gap <= CACHE_CLEAR_INTERVAL + stride);
    }
}

test "the allocator pool is returned at the prefill/decode handover, long-context gate only" {
    // The per-chunk clear runs BEFORE the last chunks' transient is freed, so it
    // parks in MLX's pool up to the cap and the first decode tick allocates on top
    // of it (measured 8.1 GB parked at the first tick of a 393k prefill; a cold
    // 524k died 0.5 GB over the plan ceiling on exactly that tick). One clear at the
    // handover returns it. Every other arch keeps its previous call pattern.
    const t = testing;
    var qwen4 = model_mod.ModelConfig{ .model_type = "qwen4_exp", .legacy = .{ .caps = .{ .long_ctx = true } } };
    try t.expect(clearsPoolAtPrefillEnd(&qwen4));
    for ([_][]const u8{
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_next",
        "lfm2",
        "nemotron_h",
        "bailing_hybrid",
        "llama",
        "gemma4",
        "deepseek_v4",
        "muse_glimmer",
    }) |mt| {
        var cfg = model_mod.ModelConfig{ .model_type = mt };
        try t.expect(!clearsPoolAtPrefillEnd(&cfg));
    }
}

test "countSpliceRows counts image, audio and video placeholders, extras only when declared" {
    const ids = [_]i32{ 7, 99, 99, 8, 88, 9, 77 };
    try testing.expectEqual(@as(usize, 2), countSpliceRows(&ids, 99, 0, 0));
    try testing.expectEqual(@as(usize, 3), countSpliceRows(&ids, 99, 88, 0));
    try testing.expectEqual(@as(usize, 4), countSpliceRows(&ids, 99, 88, 77));
    try testing.expectEqual(@as(usize, 0), countSpliceRows(ids[0..1], 99, 88, 77));
    // An audio-only model declares no image token: token 0 is text.
    try testing.expectEqual(@as(usize, 1), countSpliceRows(&[_]i32{ 0, 88, 0 }, 0, 88, 0));
}

test "reservedPrefillTokens: every arch reserves its prompt; only qwen4_exp pre-buys generation headroom" {
    // Headroom is a pure cost on an arch nobody measured, so it is gated; the prompt is written anyway.
    const t = std.testing;
    const chunk: u64 = 4096;
    const seq: u64 = 200_000;
    const max_tokens: u64 = 2048;

    var qwen4 = model_mod.ModelConfig{ .model_type = "qwen4_exp", .legacy = .{ .caps = .{ .long_ctx = true } }, .max_position_embeddings = 1_048_576 };
    const KVC = transformer_mod.KVCache;
    try t.expectEqual(
        KVC.reservedTokens(seq, max_tokens, chunk, 1_048_576),
        reservedPrefillTokens(&qwen4, seq, max_tokens, chunk),
    );
    try t.expect(reservedPrefillTokens(&qwen4, seq, max_tokens, chunk) > seq);

    for ([_][]const u8{ "qwen3_5", "qwen3_5_moe", "lfm2", "nemotron_h", "bailing_hybrid", "llama", "mimo_v2" }) |mt| {
        var cfg = model_mod.ModelConfig{ .model_type = mt, .max_position_embeddings = 262_144 };
        try t.expectEqual(seq + chunk, reservedPrefillTokens(&cfg, seq, max_tokens, chunk));
        // One chunk never grows mid-prefill: nothing to reserve.
        try t.expectEqual(@as(u64, 0), reservedPrefillTokens(&cfg, chunk, max_tokens, chunk));
    }

    var cache = try KVC.init(t.allocator, 4);
    defer cache.deinit();
    cache.reserve(0);
    try t.expectEqual(@as(usize, 0), cache.reserve_tokens);
}

test "explicitPrefillChunkU32: unset is 0, a flag past u32 clamps instead of overflowing" {
    const saved = .{ generate_mod.prefillCli().explicit, generate_mod.prefillCli().override };
    defer {
        generate_mod.prefillCli().explicit = saved[0];
        generate_mod.prefillCli().override = saved[1];
    }
    generate_mod.prefillCli().explicit = false;
    try std.testing.expectEqual(@as(u32, 0), explicitPrefillChunkU32());
    generate_mod.prefillCli().explicit = true;
    generate_mod.prefillCli().override = 256;
    try std.testing.expectEqual(@as(u32, 256), explicitPrefillChunkU32());
    generate_mod.prefillCli().override = 5_000_000_000;
    try std.testing.expectEqual(std.math.maxInt(u32), explicitPrefillChunkU32());
}
