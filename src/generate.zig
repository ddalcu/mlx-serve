const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const tokenizer_mod = @import("tokenizer.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");
const json_grammar = @import("json_grammar.zig");
const json_schema = @import("json_schema.zig");
const token_mask = @import("token_mask.zig");
const io_util = @import("io_util.zig");
const pld_index = @import("pld_index.zig");
const drafter_mod = @import("drafter.zig");
const mtp_mod = @import("mtp.zig");

const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const ForwardCtx = transformer_mod.ForwardCtx;
const SSMCacheEntrySnapshot = transformer_mod.SSMCacheEntrySnapshot;
const ssmSnapshot = transformer_mod.ssmSnapshot;
const ssmSnapshotDeinit = transformer_mod.ssmSnapshotDeinit;
const ssmRestore = transformer_mod.ssmRestore;
const SSMCheckpoint = transformer_mod.SSMCheckpoint;
const captureSsmCheckpoint = transformer_mod.captureSsmCheckpoint;
const DrafterModel = drafter_mod.DrafterModel;
const KVCache = transformer_mod.KVCache;

/// Module-level overrides for prefill behavior. Defaults match the original
/// hardcoded values; main.zig may overwrite these from CLI flags before
/// `serve()` runs. Per-request reads happen on the same thread that did the
/// CLI parse, so no atomicity needed.
pub var prefill_chunk_override: usize = 8192;
pub var prefill_trace_force: bool = false;

/// MTP prefill-history window (`--mtp-history-window`; 0 = full history).
/// Same set-once-at-CLI-parse contract as `prefill_chunk_override`.
/// DEFAULT 0 (full): the A/B gate failed for windowing — at 64K ctx on the
/// stock Qwen3.6-27B head, window 8192 cost 14 acceptance points (68.2% ->
/// 54.0%) and 4.2 decode tok/s for ZERO prefill benefit (184.7 vs 185.1
/// tok/s); at 32K it was a wash. Qwen's stock head drafts from deep history.
pub var mtp_history_window_override: usize = 0;

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
pub fn boundedPrefillChunk(base_chunk: usize, head_dim: u32, n_heads: u32, total_ctx: usize, sliding_band_arch: bool) usize {
    if (head_dim <= 128 or n_heads == 0 or total_ctx == 0) return base_chunk;
    // Archs with NO sliding-band layers (has_sliding_window=false: the
    // qwen3_5/3_6 class) run every hd-256 prefill through the composed
    // causal path, where SMALL chunks measured strictly faster AND lighter
    // on the 27B (2026-07-12 ladder, M4 Max): 8K 225 -> 235.8 tok/s and
    // peak 28.9 -> 19.8 GB at chunk 2048; 32K 205.4 -> 209.3. Chunk
    // boundaries ARE block-level causal skipping for composed attention.
    // Sliding-band archs (gemma) keep big chunks — their local layers run
    // the fused band kernel, which block-skips in-kernel and wants the
    // fewest KV re-walks (26B@99K: 712 tok/s at the formula chunk).
    const causal_cap: usize = if (sliding_band_arch) base_chunk else @min(base_chunk, 2048);
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
/// MLX_SERVE_PREFILL_CHUNK env (explicit tuning knob — honored verbatim,
/// never safety-capped) > --prefill-chunk / default, capped by
/// boundedPrefillChunk. Exported so server.zig's admission guard
/// (checkAttentionMemory) models the SAME chunk the prefill will run with —
/// the guard and the real prefill must not drift.
pub fn effectivePrefillChunk(head_dim: u32, n_heads: u32, total_ctx: usize, sliding_band_arch: bool) usize {
    const env_chunk = readEnvUsize("MLX_SERVE_PREFILL_CHUNK", 0);
    if (env_chunk > 0) return env_chunk;
    return boundedPrefillChunk(prefill_chunk_override, head_dim, n_heads, total_ctx, sliding_band_arch);
}

/// Read an unsigned integer from an environment variable, falling back to
/// `default` when unset, empty, or unparseable. Uses libc getenv to stay
/// allocator-free at call sites.
fn readEnvUsize(name: [:0]const u8, default: usize) usize {
    const raw = std.c.getenv(name.ptr);
    if (raw == null) return default;
    const slice = std.mem.sliceTo(raw.?, 0);
    if (slice.len == 0) return default;
    return std.fmt.parseInt(usize, slice, 10) catch default;
}

/// Truthy if the env var is exactly "1". Anything else (unset, "0", "true",
/// "yes") is false — keep matching surface tight to avoid surprises.
fn readEnvBool(name: [:0]const u8) bool {
    const raw = std.c.getenv(name.ptr);
    if (raw == null) return false;
    const slice = std.mem.sliceTo(raw.?, 0);
    return std.mem.eql(u8, slice, "1");
}

/// Grammar-constrained sampling state. The caller owns `grammar`, `token_bytes`,
/// and `mask_buf`; the generator only reads them. `mask_buf.len` must equal
/// `token_bytes.bytes.len` (the tokenizer's vocab size).
pub const Constraint = struct {
    grammar: *json_grammar.Grammar,
    token_bytes: *const token_mask.TokenBytes,
    mask_buf: []bool,
};

/// RAII bundle for grammar-constrained sampling. Owns the parsed schema,
/// grammar state machine, and per-step mask buffer. The embedded `Constraint`
/// holds pointers/slices into the surrounding struct, so this struct must NOT
/// be moved after `initFromValue`. Construct via `var sc: SchemaConstraint =
/// undefined; try sc.initFromValue(...);` and pass `&sc.constraint` to
/// `SamplingParams`.
pub const SchemaConstraint = struct {
    schema: json_schema.Schema,
    grammar: json_grammar.Grammar,
    mask_buf: []bool,
    constraint: Constraint,
    allocator: std.mem.Allocator,

    /// Initialize in-place from a JSON schema value. On failure, any partial
    /// allocations made during this call are freed and the struct is left
    /// undefined (do not call `deinit`).
    pub fn initFromValue(
        self: *SchemaConstraint,
        allocator: std.mem.Allocator,
        schema_value: std.json.Value,
        token_bytes: *const token_mask.TokenBytes,
    ) !void {
        self.allocator = allocator;
        self.schema = try json_schema.parse(allocator, schema_value);
        errdefer self.schema.deinit();

        self.grammar = try json_grammar.Grammar.init(allocator, &self.schema);
        errdefer self.grammar.deinit();

        self.mask_buf = try allocator.alloc(bool, token_bytes.bytes.len);
        errdefer allocator.free(self.mask_buf);

        self.constraint = .{
            .grammar = &self.grammar,
            .token_bytes = token_bytes,
            .mask_buf = self.mask_buf,
        };
    }

    pub fn deinit(self: *SchemaConstraint) void {
        self.allocator.free(self.mask_buf);
        self.grammar.deinit();
        self.schema.deinit();
    }
};

/// Per-token logprob info (OpenAI format).
pub const TokenLogprob = struct {
    token_id: u32,
    logprob: f32,
};

/// Logprob result for a single generated token.
pub const LogprobResult = struct {
    token_logprob: f32, // logprob of the chosen token
    top_logprobs: []TokenLogprob, // top N alternatives (caller must free)
};

/// Sampling parameters for token generation.
pub const SamplingParams = struct {
    temperature: f32 = 1.0,
    top_p: f32 = 1.0,
    top_k: u32 = 0, // 0 = disabled
    repeat_penalty: f32 = 1.0,
    presence_penalty: f32 = 0.0, // 0.0 = disabled
    seed: ?u64 = null,
    /// When non-null, generation is constrained to outputs that satisfy the
    /// grammar at byte level. Forces a synchronous sampling path (no lazy
    /// pipeline) since grammar advancement requires the realized token id.
    constraint: ?*Constraint = null,
};

/// Generation result (for non-streaming use).
pub const GenerationResult = struct {
    text: []u8,
    token_ids: []u32,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: []const u8,
    prefill_tps: f64,
    decode_tps: f64,
    /// Wall-clock nanoseconds spent on prefill (prompt processing).
    prefill_ns: u64 = 0,
    /// Wall-clock nanoseconds spent on decode (token generation).
    decode_ns: u64 = 0,
    /// Prompt tokens served from a KV-cache prefix (hot prefix cache for MLX,
    /// persistent-session prefix reuse for llama). `prompt_tokens - cached_tokens`
    /// is what was actually run through the model this turn, so `prefill_tps`
    /// reflects real compute rather than an inflated full-prompt rate.
    cached_tokens: u32 = 0,
    logprobs: ?[]LogprobResult = null, // per-token logprobs (caller must free)
};

/// Throughput in tokens/sec. Returns 0 when no time elapsed so unmeasured paths
/// report 0 rather than inf / NaN.
pub fn tokensPerSec(tokens: u64, elapsed_ns: u64) f64 {
    if (elapsed_ns == 0) return 0.0;
    const tok_f: f64 = @floatFromInt(tokens);
    const ns_f: f64 = @floatFromInt(elapsed_ns);
    return tok_f * @as(f64, @floatFromInt(std.time.ns_per_s)) / ns_f;
}

/// True prefill compute throughput: divides by the tokens actually pushed through
/// the model (prompt minus the prefix served from KV cache). A near-full cache
/// hit therefore reports the small suffix's real rate, not an inflated
/// full-prompt number. With `cached_tokens == 0` this is just the full-prompt
/// rate, matching the pre-instrumentation behavior.
pub fn prefillTokensPerSec(prompt_tokens: u32, cached_tokens: u32, prefill_ns: u64) f64 {
    const uncached: u32 = if (prompt_tokens > cached_tokens) prompt_tokens - cached_tokens else 0;
    return tokensPerSec(uncached, prefill_ns);
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
/// A trailing remainder smaller than this merges into the preceding chunk
/// instead of becoming its own chunk. Chat-templated prompts routinely land a
/// token or two past a chunk multiple (an "8192-token" prompt tokenizes to
/// 8193); a 1-token final chunk pays a full graph build + eval barrier +
/// cache clear for one token. The merged chunk's attention-score transient
/// grows by at most TAIL_MERGE_MAX/default_chunk (~6% at 8192) — within the
/// score-budget slack `boundedPrefillChunk` already carries.
pub const TAIL_MERGE_MAX: usize = 512;

pub fn nextChunkEnd(
    pos: usize,
    prefix_len: usize,
    default_chunk: usize,
    want_ssm_cp: bool,
    ssm_cp_stride: usize,
    ssm_cp_offset: usize,
) usize {
    var end = @min(pos + default_chunk, prefix_len);
    if (want_ssm_cp and ssm_cp_stride > 0) {
        const abs_pos = pos + ssm_cp_offset;
        const abs_end = end + ssm_cp_offset;
        const next_boundary_abs = ((abs_pos / ssm_cp_stride) + 1) * ssm_cp_stride;
        if (next_boundary_abs > abs_pos and next_boundary_abs < abs_end) {
            end = next_boundary_abs - ssm_cp_offset;
        }
    } else if (end < prefix_len and prefix_len - end < TAIL_MERGE_MAX) {
        // No checkpoint alignment to respect — absorb a tiny tail.
        end = prefix_len;
    }
    return end;
}

/// Effective SSM-checkpoint stride for a model, given the base (configured)
/// stride and whether the model is MoE.
///
/// On dense / non-MoE-hybrid models (dense Gemma sliding-window, LFM2, Nemotron-H)
/// prefill is compute-bound, so a fine stride is ~free and buys finer warm
/// mid-prompt reuse — keep the base stride. On MoE models prefill is
/// memory-bound on the per-expert weights: every checkpoint-induced chunk
/// re-streams ~all expert weights from HBM (the dominant cold-prefill cost), so
/// a fine stride silently taxes prefill ~25% on 26B/35B-class MoE. For MoE we
/// coarsen the stride to at least `prefill_chunk`, so checkpointing never
/// sub-divides the memory-bound chunk — MoE prefill is then never over-chunked
/// at any prompt length, while the always-on end-of-prompt snapshot still
/// provides the dominant append-growth multi-turn reuse. `base == 0`
/// (checkpointing disabled) is preserved.
pub fn effectiveSsmCheckpointStride(base: usize, is_moe: bool, prefill_chunk: usize) usize {
    if (base == 0) return 0;
    if (is_moe) return @max(base, prefill_chunk);
    return base;
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
) usize {
    var pos: usize = 0;
    var n: usize = 0;
    while (pos < prefix_len) {
        const end = nextChunkEnd(pos, prefix_len, default_chunk, want_ssm_cp, ssm_cp_stride, ssm_cp_offset);
        pos = end;
        n += 1;
    }
    return n;
}

/// Step-based generator. Call `init` to prefill, then `next` per token.
/// Uses a fully-lazy async pipeline matching mlx-lm: sample + next forward are
/// built as a single lazy computation graph, async_eval'd together. The GPU
/// never idles between token generation steps.
pub const Generator = struct {
    xfm: *Transformer,
    /// Forward-pass context. Stores per-request KVCache pointer, moe_seq_offset
    /// pointer, ssm_entries slice, vision_embeddings handle, and capture_hidden
    /// override. The legacy single-slot path uses `xfm.defaultCtx()` (pointing at
    /// the Transformer's own fields). Phase 2 concurrent batching constructs a
    /// per-slot ForwardCtx pointing at the slot's own KVCache, etc., so multiple
    /// generators can share one Transformer's weights without colliding on
    /// per-request state. Stored by value; `&self.ctx` is what we pass to
    /// `xfm.forwardWith` / `lazyForward` / drafter step.
    ctx: ForwardCtx,
    tok: *const Tokenizer,
    next_token_id: u32,
    step: u32,
    max_tokens: u32,
    sampling: SamplingParams,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: []const u8,
    done: bool,
    eos_token_ids: []const u32,
    generated_ids: std.ArrayList(u32),
    consecutive_pad: u32 = 0, // count of consecutive token-0 (pad) generations
    timeout_ns: u64, // 0 = no timeout; measures SILENCE, not total time (see StallClock)
    stall: StallClock = .{},
    timer: io_util.Stopwatch,
    logprobs_n: u32 = 0, // 0 = disabled, >0 = number of top_logprobs to return
    last_logprob: ?LogprobResult = null, // logprob result for the most recently returned token
    // Async pipeline state: pre-computed forward pass logits for next decode step
    pending_logits: mlx.mlx_array = .{},
    has_pending_logits: bool = false,
    // Deferred token: lazy array from async pipeline, eval'd at start of next iteration
    pending_token: mlx.mlx_array = .{},
    has_pending_token: bool = false,

    // ── Spec-decode shared state (PLD + drafter) ──
    // Post-final-norm hidden state at the last produced token's position.
    // Owned by the Generator (freed in `deinit`). Captured by
    // `forwardCaptureHidden` during prefill final-token forward and every
    // verify forward — used by drafter as h_prev seed and by PLD verify
    // partial-accept rollback.
    last_hidden: mlx.mlx_array = .{},
    has_last_hidden: bool = false,
    /// PRNG for PLD / drafter stochastic-verify accept test (probability-
    /// ratio requires a uniform draw per draft step). Seeded from
    /// `sampling.seed` when set, otherwise from system time at init.
    prng: std.Random.DefaultPrng = std.Random.DefaultPrng.init(0),

    // ── PLD (Prompt Lookup Decoding) state ──
    // Owned copy of the input prompt ids — needed because PLD's n-gram lookup
    // table is `prompt + generated_ids`, and the caller-supplied `prompt_ids`
    // slice is freed after `init` returns. `generated_ids` (above) tracks
    // post-prefill tokens; `prompt_ids_owned` is the immutable prefix.
    prompt_ids_owned: []u32 = &.{},
    /// Allocator that owns `prompt_ids_owned`. Stored so `deinit` can free it
    /// without requiring callers to thread the allocator a second time. (Other
    /// owned slices are freed via the `allocator` argument to `deinit` for
    /// historical reasons; this one is set during `initWithOptions`.)
    prompt_ids_alloc: ?std.mem.Allocator = null,
    /// Stats for PLD benchmark logging. `pld_attempted` counts every step
    /// where lookup found a candidate (so a verify forward ran);
    /// `pld_accepted_tokens` is the cumulative number of *drafted* tokens
    /// (not including the always-accepted t1) that were successfully verified.
    pld_attempted: u64 = 0,
    pld_accepted_tokens: u64 = 0,

    // ── Gemma 4 assistant drafter state ──
    // External drafter model (cross-attends into target's KV). When
    // `drafter != null`, callers use `nextDrafter` instead of `next`. The
    // drafter is owned by the server (loaded once at startup); the Generator
    // only holds a non-owning pointer.
    drafter: ?*DrafterModel = null,
    /// Number of tokens proposed per round (= drafter forwards + 1 verify token).
    /// Defaults to 4 (3 drafter steps + 1 t1 prepend → length-4 verify).
    drafter_block_size: u32 = 4,
    /// Stats: count of nextDrafter calls that ran a verify forward.
    drafter_attempted: u64 = 0,
    /// Stats: cumulative draft tokens accepted (excluding always-accepted t1).
    drafter_accepted_tokens: u64 = 0,

    // ── Qwen native MTP head state ──
    // The model's own one-layer multi-token-prediction head (src/mtp.zig).
    // When `mtp != null`, callers use `nextMtp` instead of `next`. The head
    // is owned by the server (loaded with the model); the Generator only
    // holds a non-owning pointer. `mtp_cache` is the head's committed-history
    // KV cache — OWNED by the Generator (built during prefill, freed in
    // `deinit`).
    mtp: ?*mtp_mod.MtpModel = null,
    mtp_cache: ?KVCache = null,
    /// CONFIGURED max tokens drafted per round (verify length = depth + 1).
    mtp_depth: u32 = mtp_mod.DEFAULT_DEPTH,
    /// CURRENT adaptive depth (see updateMtpDepth). Starts at `mtp_depth`,
    /// demoted/promoted per windowed acceptance, never exceeds `mtp_depth`.
    mtp_depth_current: u32 = mtp_mod.DEFAULT_DEPTH,
    /// Stats: count of nextMtp calls that ran a verify forward.
    mtp_attempted: u64 = 0,
    /// Stats: cumulative draft tokens accepted (excluding always-accepted t1).
    mtp_accepted_tokens: u64 = 0,
    /// Adaptive-depth moving window: per-round drafted/accepted counts.
    mtp_window_drafted: [MTP_DEPTH_WINDOW]u8 = [_]u8{0} ** MTP_DEPTH_WINDOW,
    mtp_window_accepted: [MTP_DEPTH_WINDOW]u8 = [_]u8{0} ** MTP_DEPTH_WINDOW,
    mtp_window_idx: u32 = 0,
    mtp_rounds_since_switch: u32 = 0,
    /// Rounds remaining during which promotion is blocked (set after a
    /// demotion so a failed depth excursion isn't immediately retried).
    mtp_promote_cooldown: u32 = 0,
    /// Cumulative drafted tokens across rounds. The EV controller varies m
    /// per round, so `attempts x depth` no longer measures proposals — this
    /// is the honest per_draft_pct denominator.
    mtp_drafted_tokens: u64 = 0,
    /// Rounds where the confidence gate extended into chunk B.
    mtp_ext_rounds: u64 = 0,
    /// EV controller: conditional acceptance EMA per draft index,
    /// a[i] = P(draft i accepted | drafts 0..i-1 accepted). Optimistic prior;
    /// warmup rounds pull the low indices to reality before it can matter.
    mtp_ev_accept: [mtp_mod.MAX_DEPTH]f32 = [_]f32{MTP_EV_PRIOR} ** mtp_mod.MAX_DEPTH,
    /// Rounds seen by the EV controller (drives the legacy-behavior warmup).
    mtp_ev_rounds: u32 = 0,
    /// Last round's planned m_lo (base-depth climb damping: +1/round max).
    mtp_ev_m_lo_prev: u32 = 1,
    /// Round-cost surface selected once for this target+MTP head. The M5/G17
    /// NAX surface requires the measured trunk, native sidecar, and 3-bit
    /// draft-only head; every other combination keeps the M1-M4 surface.
    mtp_ev_costs: MtpEvCosts = MTP_EV_DEFAULT_COSTS,
    /// Per-phase wall-time trace (MLX_SERVE_MTP_TRACE=1; else untouched).
    mtp_trace: MtpTrace = .{},
    /// Deferred committed-history append: the round's (tokens, true verify
    /// hiddens) pair, folded into the NEXT round's first draft step as one
    /// multi-row head forward instead of a separate appendHistory forward.
    /// Rounds with no successor (EOS/length/runtime disable) never pay for
    /// the append; the stash is freed unconsumed in `deinit`.
    mtp_hist_stash: ?MtpHistStash = null,

    // ── Phase 1: SSM checkpoints captured during prefill ──
    /// Owned SSM-state snapshots taken at stride-aligned positions during
    /// chunked prefill. Drained by the scheduler in `commitSlotIfApplicable`
    /// via `takeSsmCheckpoints()`. Empty on non-hybrid models or when
    /// `ssm_checkpoint_stride == 0`. Allocator: the Generator's `allocator`
    /// (passed to `initWithOptions`); the same allocator must be passed to
    /// `deinit` for any checkpoint that wasn't taken.
    ssm_checkpoints: std.ArrayList(SSMCheckpoint) = std.ArrayList(SSMCheckpoint).empty,
    /// Allocator used for `ssm_checkpoints` storage and each checkpoint's
    /// per-layer slice. Set during `initWithOptions`. We track it separately
    /// from the `allocator` argument to `deinit` because `takeSsmCheckpoints`
    /// transfers ownership: the consumer (HotPrefixCache) must use the SAME
    /// allocator to free, since the layer-slice backing memory was allocated
    /// here.
    ssm_checkpoint_alloc: ?std.mem.Allocator = null,

    // ── Runtime acceptance gate ──
    // Set to true mid-request when the per-request acceptance rate
    // (`*_accepted_tokens / *_attempted`) falls below
    // `RUNTIME_GATE_MIN_RATE` after `RUNTIME_GATE_WARMUP` attempts. When set,
    // both `nextPld` and `nextDrafter` short-circuit to `next()` for the
    // remainder of the request — the prompt-time gate could not foresee that
    // the workload's *runtime* draft acceptance rate wasn't paying for the
    // per-step verify overhead. The flag is sticky for the rest of the
    // generation; we never re-enable speculation within a single request.
    spec_disabled_runtime: bool = false,
    /// Yield-gate counters: enabled-mode `nextPld` steps and drafted tokens
    /// accepted since the last (re-)enable. Reset on mid-request re-enable so
    /// a fresh workload region (e.g. file echo after a novel preamble) gets a
    /// fresh economic evaluation instead of inheriting the bad early yield.
    yield_steps: u64 = 0,
    yield_accepted: u64 = 0,
    /// Steps spent in disabled mode since the gate tripped (drives the
    /// periodic `specShouldReenable` re-check).
    disabled_steps: u64 = 0,
    /// Number of attempts before the runtime gate considers disabling.
    /// Below this we trust the prompt-time gate.
    ///
    /// Override at runtime via `SPEC_GATE_WARMUP` env var (parsed in `runtimeGateWarmup()`
    /// once per request). Lower values make the gate trip sooner,
    /// reducing regression-tail damage at the cost of fewer chances for slow-warmup
    /// workloads to amortize spec overhead.
    pub const RUNTIME_GATE_WARMUP: u64 = 5;

    /// Read the warmup threshold for this call. Env-overridable so we can A/B
    /// without rebuilding. Anything outside `[1, 64]` falls back to the default.
    pub fn runtimeGateWarmup() u64 {
        const n = readEnvUsize("SPEC_GATE_WARMUP", @intCast(RUNTIME_GATE_WARMUP));
        if (n < 1 or n > 64) return RUNTIME_GATE_WARMUP;
        return @intCast(n);
    }
    /// Minimum per-draft acceptance probability. Below this after warmup,
    /// speculation is disabled for the rest of the request.
    ///
    /// History: pre-v5 this gate compared `accepted/attempted` (per-round
    /// average) against 0.30 — but with `block_size=4` the max value of that
    /// ratio is 3.0, so the 0.30 threshold corresponded to ~10% per-draft
    /// probability, well below where verify+draft overhead actually breaks
    /// even. Empirically creative-content workloads regress at 22-47% per-draft
    /// acceptance
    /// while the gate stayed off (per-round avg 0.66-1.58, all above 0.30).
    /// Switching to a per-draft probability with threshold 0.50 cleanly cuts
    /// off the regressing tail while leaving heavy-echo workloads (84-97%
    /// per-draft) running unmolested.
    pub const RUNTIME_GATE_MIN_PER_DRAFT_RATE: f32 = 0.50;

    /// Pure helper: should the runtime gate disable speculation given the
    /// observed per-request stats? `drafts_per_round` is the number of
    /// drafted tokens proposed in each verify (= `block_size - 1` for the
    /// drafter, or `pld_draft_len` for PLD); we divide accepts by attempts ×
    /// drafts_per_round to get the per-draft acceptance probability.
    /// Returns true iff `attempted >= warmup` AND per-draft probability is
    /// below `RUNTIME_GATE_MIN_PER_DRAFT_RATE`.
    ///
    /// `drafts_per_round == 0` is treated as "no speculative work happens
    /// per round" → never trip (defensive — current callers always pass
    /// >= 1).
    pub fn runtimeGateShouldDisable(attempted: u64, accepted: u64, drafts_per_round: u32) bool {
        if (attempted < runtimeGateWarmup()) return false;
        if (drafts_per_round == 0) return false;
        const drafts_proposed = attempted * @as(u64, drafts_per_round);
        const rate = @as(f32, @floatFromInt(accepted)) /
            @as(f32, @floatFromInt(drafts_proposed));
        return rate < RUNTIME_GATE_MIN_PER_DRAFT_RATE;
    }

    // ── PLD yield gate (cold-path economics) ──
    // The per-draft gate above only counts verify ROUNDS, so a workload where
    // the n-gram lookup rarely matches never accumulates enough "attempts" to
    // trip it — yet every no-match step pays PLD's unpipelined cold forward
    // (measured −14% vs the async-pipelined `next()` on creative content).
    // The yield gate instead counts EVERY enabled-mode nextPld step: if the
    // speculation is yielding fewer than YIELD_GATE_MIN_YIELD extra (drafted,
    // accepted) tokens per step after YIELD_GATE_WARMUP steps, the cold-path
    // tax outweighs the wins → disable. Paired with `specShouldReenable`,
    // which flips PLD back on when the generated tail turns repetitive.
    // Warmup 32 (not higher): the re-enable check bounds the cost of a
    // premature trip to ≤SPEC_REENABLE_INTERVAL pipelined-fallback steps,
    // so we can gate early and recover the pipeline sooner on novel content.
    pub const YIELD_GATE_WARMUP: u64 = 32;
    pub const YIELD_GATE_MIN_YIELD: f32 = 0.25;

    pub fn yieldGateShouldDisable(steps_total: u64, accepted: u64) bool {
        if (steps_total < YIELD_GATE_WARMUP) return false;
        const yield_rate = @as(f32, @floatFromInt(accepted)) /
            @as(f32, @floatFromInt(steps_total));
        return yield_rate < YIELD_GATE_MIN_YIELD;
    }

    // ── Mid-request spec re-enable ──
    // While the yield gate has PLD disabled, the COMMITTED sequence (prompt +
    // generated) is re-scored every SPEC_REENABLE_INTERVAL steps: what
    // fraction of the recent generated positions would have had a PLD lookup
    // hit (their key-gram appears earlier in committed)? This catches the
    // echo workload where the model repeats PROMPT content (file edits, tool
    // results) — self-repetition scoring misses it because the echoed tail
    // never repeats itself. Above the threshold, PLD is worth re-engaging at
    // the cost of one pipeline drain.
    pub const SPEC_REENABLE_INTERVAL: u64 = 32;
    pub const SPEC_REENABLE_WINDOW: usize = 32;
    pub const SPEC_REENABLE_MIN_FRACTION: f32 = 0.25;
    pub const SPEC_REENABLE_MIN_TOKENS: usize = 16;

    pub fn specShouldReenable(committed: []const u32, generated_len: usize) bool {
        if (generated_len < SPEC_REENABLE_MIN_TOKENS) return false;
        const window = @min(SPEC_REENABLE_WINDOW, generated_len);
        const frac = pld_index.tailMatchFraction(committed, window, 3);
        return frac >= SPEC_REENABLE_MIN_FRACTION;
    }

    /// Emit a stable, easy-to-grep one-line summary of spec-decode acceptance
    /// for this request. External tooling parses the `[spec-stats]` prefix;
    /// keep the format stable.
    ///
    /// No-op when this Generator never ran a speculative path. Drafter and
    /// PLD are mutually exclusive within a single request (drafter > PLD per
    /// dispatch), so the branching here is unambiguous.
    ///
    /// Field semantics:
    /// - `attempts` = number of speculative rounds (one verify forward each).
    /// - `accepts` = total drafted tokens accepted across all rounds (excludes
    ///   the always-committed t1 token at the start of each round).
    /// - `avg_per_round` = accepts/attempts. Bounded by `(block_size - 1)` for
    ///   drafter and `pld_draft_len` for PLD. Equals the metric the runtime
    ///   gate compares against `RUNTIME_GATE_MIN_RATE`.
    /// - `per_draft_pct` (drafter only) = accepts / (attempts × (block_size-1)),
    ///   the per-draft acceptance probability comparable to vLLM's reported
    ///   "62% acceptance rate" metric.
    pub fn logSpecStats(self: *const Generator) void {
        if (self.mtp != null and self.mtp_attempted > 0) {
            const avg_per_round: f64 = @as(f64, @floatFromInt(self.mtp_accepted_tokens)) /
                @as(f64, @floatFromInt(self.mtp_attempted));
            // Depth varies per round under the EV controller — the honest
            // denominator is the DRAFTED count, not attempts x cap.
            const drafts_proposed: u64 = if (self.mtp_drafted_tokens > 0)
                self.mtp_drafted_tokens
            else
                self.mtp_attempted * @as(u64, self.mtp_depth);
            const per_draft_pct: f64 = if (drafts_proposed > 0)
                100.0 * @as(f64, @floatFromInt(self.mtp_accepted_tokens)) /
                    @as(f64, @floatFromInt(drafts_proposed))
            else
                0.0;
            log.info(
                "  [spec-stats] mode=mtp attempts={d} accepts={d} avg_per_round={d:.2} per_draft_pct={d:.1}% depth={d} drafted={d} ext_rounds={d} runtime_disabled={s}\n",
                .{
                    self.mtp_attempted,
                    self.mtp_accepted_tokens,
                    avg_per_round,
                    per_draft_pct,
                    self.mtp_depth,
                    self.mtp_drafted_tokens,
                    self.mtp_ext_rounds,
                    if (self.spec_disabled_runtime) "true" else "false",
                },
            );
            return;
        }
        if (self.drafter != null and self.drafter_attempted > 0) {
            const avg_per_round: f64 = @as(f64, @floatFromInt(self.drafter_accepted_tokens)) /
                @as(f64, @floatFromInt(self.drafter_attempted));
            const drafts_per_round: u32 = if (self.drafter_block_size >= 1) self.drafter_block_size - 1 else 0;
            const drafts_proposed: u64 = self.drafter_attempted * @as(u64, drafts_per_round);
            const per_draft_pct: f64 = if (drafts_proposed > 0)
                100.0 * @as(f64, @floatFromInt(self.drafter_accepted_tokens)) /
                    @as(f64, @floatFromInt(drafts_proposed))
            else
                0.0;
            log.info(
                "  [spec-stats] mode=drafter attempts={d} accepts={d} avg_per_round={d:.2} per_draft_pct={d:.1}% block_size={d} runtime_disabled={s}\n",
                .{
                    self.drafter_attempted,
                    self.drafter_accepted_tokens,
                    avg_per_round,
                    per_draft_pct,
                    self.drafter_block_size,
                    if (self.spec_disabled_runtime) "true" else "false",
                },
            );
        } else if (self.pld_attempted > 0) {
            const avg_per_round: f64 = @as(f64, @floatFromInt(self.pld_accepted_tokens)) /
                @as(f64, @floatFromInt(self.pld_attempted));
            log.info(
                "  [spec-stats] mode=pld attempts={d} accepts={d} avg_per_round={d:.2} runtime_disabled={s}\n",
                .{
                    self.pld_attempted,
                    self.pld_accepted_tokens,
                    avg_per_round,
                    if (self.spec_disabled_runtime) "true" else "false",
                },
            );
        }
    }

    /// Prefill the prompt and prepare for token-by-token generation.
    /// Backwards-compatible — prefer `initWithOptions` for new callers.
    pub fn init(
        io: std.Io,
        allocator: std.mem.Allocator,
        xfm: *Transformer,
        tok: *const Tokenizer,
        prompt_ids: []const u32,
        max_tokens: u32,
        sampling: SamplingParams,
        eos_token_ids: []const u32,
    ) !Generator {
        return initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{});
    }

    pub const InitOptions = struct {
        /// Skip the lazy pre-forward of the first sampled token. When set,
        /// init samples t1 synchronously and leaves `pending_logits/pending_token`
        /// empty — the cache lands at exactly `prompt_len` with t1 NOT in cache.
        /// `nextPld` v2 (mirroring `nextDrafter`) drives every step from that
        /// invariant: verify input is `[t1, draft[0..m-1]]` length `1+m`; full
        /// accept commits `1+m` tokens with cache landing at `prompt_len + TE_new`
        /// and NO post-step forward. Saves one decode-step forward per accepted
        /// PLD step at the cost of losing the lazy-pipeline overlap on cold
        /// (no-match) steps. The prompt-time gate disables PLD on novel content
        /// where cold-path dominates.
        pld_enabled: bool = false,
        /// Enable Gemma 4 assistant drafter. When set, `drafter` must be
        /// non-null and already `bind()`-ed to `xfm`. Init's prefill final-token
        /// forward captures the post-final-norm hidden state into
        /// `Generator.last_hidden` (reused for the drafter's first-step
        /// h_prev — see comment in `nextDrafter`). Same lazy-pre-forward
        /// skip semantics as PLD.
        drafter_enabled: bool = false,
        /// Non-owning pointer to the loaded drafter (must be non-null when
        /// `drafter_enabled` is true).
        drafter: ?*DrafterModel = null,
        /// Number of tokens per draft round. Default 4 (3 drafter steps +
        /// 1 t1 prepend → length-4 verify forward).
        drafter_block_size: u32 = 4,
        /// Enable the Qwen native MTP head. When set, `mtp` must be non-null
        /// and `bind()`-ed to `xfm`. Prefill builds the head's committed-
        /// history KV cache chunk-by-chunk (full-hidden capture) and the
        /// final-token forward captures `last_hidden`, exactly like the
        /// drafter path. Same lazy-pre-forward skip semantics as PLD/drafter.
        mtp_enabled: bool = false,
        /// Non-owning pointer to the loaded MTP head.
        mtp: ?*mtp_mod.MtpModel = null,
        /// Max tokens drafted per nextMtp round. 0 = auto (`--mtp-depth` not
        /// passed): resolved by `resolveMtpDepthCap` — MTP_ADAPTIVE_NAX_CAP
        /// for the measured M5 target+sidecar profile, otherwise
        /// MTP_ADAPTIVE_DEFAULT_CAP under the EV controller; DEFAULT_DEPTH in
        /// fixed mode. Explicit depths remain unchanged.
        mtp_depth: u32 = 0,
        /// When set, this slice (rather than `prompt_ids`) becomes the
        /// `prompt_ids_owned` source for PLD's n-gram lookup. Used by the
        /// server's KV-cache-reuse path to forward only the trailing tokens
        /// while still giving PLD the full prompt for matching.
        lookup_prompt: ?[]const u32 = null,
        /// Per-slot forward context (Phase 2 concurrent batching). When null,
        /// `initWithOptions` builds one from `xfm.defaultCtx()` so the legacy
        /// single-slot path is unchanged. Phase 2 callers pass a ForwardCtx
        /// whose `cache` / `moe_seq_offset` / `ssm_entries` / `vision_embeddings`
        /// point at the slot's own state. Stored by value on the Generator.
        ctx: ?ForwardCtx = null,
        /// Skip the lazy first-token pre-forward (regular path only). When set,
        /// init samples t1 synchronously and leaves `pending_logits` /
        /// `pending_token` empty — cache.step lands at exactly prompt_len with
        /// t1 NOT in cache. The first `next()` call's transition shim will
        /// sync-forward `[t1]` to seed pending_logits before the lazy chain.
        /// Used by the Phase 2 scheduler so a slot's cache state matches
        /// `forwardBatchedDecode`'s expectation (cache.step == prompt_len at
        /// the start of every decode tick). PLD / drafter paths already skip
        /// the lazy pre-forward unconditionally; this flag generalizes that
        /// behavior to the regular sampling path. Has no effect when
        /// `pld_enabled` or `drafter_enabled` is true.
        skip_lazy_preforward: bool = false,
        /// Phase 1 (performance-plan): during prefill, capture an SSM
        /// checkpoint every `ssm_checkpoint_stride` tokens. 0 = disabled.
        /// Snapshots land in `Generator.ssm_checkpoints` for the caller to
        /// drain into the hot prefix cache via `takeSsmCheckpoints()`. Only
        /// effective when the model has hybrid layers (otherwise the
        /// `ssm_entries` slice is empty and snapshots become no-op stubs).
        /// Chunked prefill aligns chunk ends to stride positions so each
        /// snapshot reflects a coherent state.
        ssm_checkpoint_stride: u32 = 0,
        /// Cap on the number of checkpoints retained. The first stride-aligned
        /// position is always captured; if more would land than `ssm_checkpoint_max`,
        /// the oldest checkpoints are dropped to keep the latest run of
        /// positions. 0 = unlimited (rely on the hot-cache byte budget to bound).
        ssm_checkpoint_max: u32 = 16,
        /// Phase 1: absolute position of the FIRST token in `prompt_ids`.
        /// On a cold prefill this is 0. On the warm path (where the
        /// scheduler restored some prefix and now forwards only the tail),
        /// callers pass `hot_matched` so the captured checkpoints stamp
        /// absolute positions usable by future warm-path lookups against
        /// the full prompt.
        ssm_checkpoint_pos_offset: usize = 0,
        /// Cooperative abort for abandoned requests: checked between prefill
        /// chunks. The scheduler passes `&slot.cancelled`, set by the conn
        /// thread when the client disconnects mid-prefill. When it flips,
        /// `initWithOptions` returns `error.Cancelled` instead of grinding
        /// out the rest of a multi-minute ghost prefill.
        cancel_flag: ?*const std.atomic.Value(bool) = null,
        /// LIVE prefill progress, in tokens actually forwarded so far by THIS
        /// prefill. Bumped once per chunk (not per token), read off-thread by
        /// the metrics gauge sampler.
        ///
        /// Without it the panel is blind during prefill: `prompt_tokens_total`
        /// and `prefill_time_seconds` only advance at request COMPLETION, and
        /// generated tokens only accrue during decode — so a multi-minute
        /// prefill saturates the GPU while both tiles read 0 / "—". The
        /// scheduler resets it to 0 when the prefill ends.
        prefill_progress: ?*std.atomic.Value(u64) = null,
    };

    /// Selects the source slice that `initWithOptions` will dupe into
    /// `prompt_ids_owned`. When `lookup_prompt` is non-null it wins (server
    /// cache-reuse path: full original prompt for PLD lookup); otherwise the
    /// caller's `prompt_ids` is used (back-compat path).
    pub fn pickLookupPromptSource(prompt_ids: []const u32, lookup_prompt: ?[]const u32) []const u32 {
        return lookup_prompt orelse prompt_ids;
    }

    pub fn initWithOptions(
        io: std.Io,
        allocator: std.mem.Allocator,
        xfm: *Transformer,
        tok: *const Tokenizer,
        prompt_ids: []const u32,
        max_tokens: u32,
        sampling: SamplingParams,
        eos_token_ids: []const u32,
        options: InitOptions,
    ) !Generator {
        const s = xfm.s;
        // Per-slot ForwardCtx (Phase 2). Stored by value on the Generator;
        // callers either supply one (scheduler) or fall through to
        // `xfm.defaultCtx()` for the legacy single-slot path. We pass
        // `&ctx` to every forward call below; the cache/moe/ssm fields
        // mutate in-place through their pointers.
        var ctx: ForwardCtx = options.ctx orelse xfm.defaultCtx();

        const ids_i32 = try allocator.alloc(i32, prompt_ids.len);
        defer allocator.free(ids_i32);
        for (prompt_ids, 0..) |id, i| {
            ids_i32[i] = @intCast(id);
        }

        // Clone the lookup prompt for the lifetime of the Generator. PLD's
        // n-gram lookup needs `prompt + generated`, and the caller-owned
        // slice can be freed before `nextPld` runs. When `options.lookup_prompt`
        // is set (server cache-reuse path), it carries the full original prompt
        // so PLD's match coverage isn't gutted when only a trailing tail was
        // forwarded into the KV cache. Defaults to `prompt_ids` otherwise.
        // Allocated up front so init's errdefer paths don't have to track
        // partial state.
        const owned_src = pickLookupPromptSource(prompt_ids, options.lookup_prompt);
        const prompt_owned = try allocator.dupe(u32, owned_src);
        errdefer allocator.free(prompt_owned);

        // Split prefill: process first N-1 tokens (cache-only, skip lm_head eval),
        // then the last token (produces logits for sampling). This mirrors mlx-lm's
        // generate_step which avoids the expensive lm_head projection over the full
        // sequence length. For vocab_size=262144, skipping lm_head on N-1 tokens
        // avoids a [N-1, hidden] @ [hidden, 262144] matmul.
        //
        // Chunked prefill: large prompts are processed in PREFILL_CHUNK-sized pieces
        // to bound peak activation memory. Each chunk fills KV cache entries for its
        // positions, gets eval'd, and intermediates are freed before the next chunk.
        // Without chunking, Gemma-4 MoE's 2 MLPs × 4 stacked layers can spike to
        // ~20 GB of activations alone on a 50k-token prompt, causing Metal OOM.
        // Vision requests skip chunking since image token positions must be visible
        // in a single forward pass for spliceVisionEmbeddings to work correctly.
        // PREFILL_CHUNK overridable via env MLX_SERVE_PREFILL_CHUNK for tuning,
        // or via the module-level `prefill_chunk_override` (set by --prefill-chunk
        // CLI flag in main.zig). Env var wins if both are set (and skips the
        // safety cap below — it's the explicit escape hatch).
        //
        // Safety cap: on unfused head dims (>128 — every Gemma-4/Qwen3.5/3.6)
        // the composed SDPA materializes [heads, chunk, total_kv] scores per
        // layer, so the chunk shrinks with the prompt's FINAL KV length to keep
        // that one tensor bounded (boundedPrefillChunk). Warm-path restores
        // start at ssm_checkpoint_pos_offset, so the final KV length is that
        // offset plus everything we're about to forward.
        const total_ctx_for_chunk = options.ssm_checkpoint_pos_offset + prompt_ids.len;
        const PREFILL_CHUNK: usize = effectivePrefillChunk(
            xfm.config.head_dim,
            xfm.config.num_attention_heads,
            total_ctx_for_chunk,
            xfm.config.has_sliding_window,
        );
        // Phase-level prefill instrumentation. Enabled at debug level OR via
        // MLX_SERVE_PREFILL_TRACE=1 (which forces the trace line at info).
        // Phase 0 of plan 04 — gives us a decomposed view of where cold prefill
        // time goes (chunked-forward vs eval vs last-token-forward).
        const trace_force: bool = prefill_trace_force or readEnvBool("MLX_SERVE_PREFILL_TRACE");
        const trace_enabled = log.isDebug() or trace_force;
        var prefill_sw = io_util.Stopwatch.init(io);
        var chunked_ns: u64 = 0;
        var eval_ns: u64 = 0;
        var n_chunks: usize = 0;

        // Phase 1: SSM checkpointing during prefill. When enabled, the chunked
        // prefill loop forces a chunk boundary at every multiple of
        // `ssm_checkpoint_stride`, then snapshots `ctx.ssm_entries` after that
        // chunk evaluates. Snapshots accumulate in `Generator.ssm_checkpoints`
        // for the scheduler to drain in `commitSlotIfApplicable`. Plain-attn
        // models have an empty `ssm_entries` slice, so this becomes a no-op
        // even at stride > 0 — but we still bail early so we never allocate
        // empty checkpoints.
        var ssm_checkpoints: std.ArrayList(SSMCheckpoint) = std.ArrayList(SSMCheckpoint).empty;
        errdefer {
            for (ssm_checkpoints.items) |*cp| cp.deinit(allocator);
            ssm_checkpoints.deinit(allocator);
        }
        const want_ssm_cp =
            options.ssm_checkpoint_stride > 0 and
            ctx.ssm_entries != null and
            ctx.ssm_entries.?.len > 0;
        // Coarsen the checkpoint stride for MoE so memory-bound expert-weight
        // re-streaming doesn't tax cold prefill (see effectiveSsmCheckpointStride).
        // The predicate is config.isMoe() (real experts), NOT moe_layers != null:
        // dense qwen3_5 (GDN hybrid) rides the MoE forward path structurally, and
        // coarsening it to PREFILL_CHUNK silently disabled every prefix-cache hit
        // under 8K-token prompts (caught live by llmprobe cache-hit-reported on
        // Qwen3.6-27B dense, 2026-06-10).
        // Coarsen against the UNCAPPED base chunk: the head_dim safety cap
        // above must not densify MoE checkpoint spacing (16× more captures at
        // 255K ctx otherwise). nextChunkEnd already shortens a chunk to land
        // on stride boundaries, so a capped chunk stays compatible.
        const ssm_cp_stride: usize = if (want_ssm_cp)
            effectiveSsmCheckpointStride(@intCast(options.ssm_checkpoint_stride), xfm.config.isMoe(), @max(PREFILL_CHUNK, prefill_chunk_override))
        else
            0;
        // Absolute KV position of `prompt_ids[0]`. Warm-path callers (the
        // scheduler after restoring a checkpoint) pass the matched prefix
        // length so the snapshots stamp positions valid in the full original
        // sequence, not relative offsets inside the tail-only prefill.
        const ssm_cp_offset: usize = options.ssm_checkpoint_pos_offset;

        // Qwen native MTP: build the head's committed-history KV cache during
        // prefill. Entry j pairs (trunk hidden at prompt position j, token at
        // j+1); the (hidden[last], t1) pair is appended by the first nextMtp
        // round. On KV-prefix reuse the history covers only the freshly
        // forwarded tail — RoPE offsets are cache-relative, so a late-starting
        // history is self-consistent (sliding-window history semantics).
        const mtp_active = options.mtp_enabled and options.mtp != null;
        var mtp_cache: ?KVCache = if (mtp_active) try options.mtp.?.makeCache(allocator) else null;
        errdefer if (mtp_cache) |*mc| mc.deinit();

        if (prompt_ids.len > 1) {
            const prefix_len = prompt_ids.len - 1;
            const has_vision = ctx.vision_embeddings != null;
            const default_chunk = if (has_vision) prefix_len else PREFILL_CHUNK;
            // Last-window MTP history: chunks entirely before the window skip
            // the full-hidden capture AND the head forward (see
            // mtp.SUGGESTED_HISTORY_WINDOW). 0 = capture every chunk.
            const mtp_hist_window = effectiveMtpHistoryWindow(prefix_len, mtp_history_window_override);

            var pos: usize = 0;
            while (pos < prefix_len) {
                // Abandoned-request abort: the client disconnected and the
                // conn thread flagged the slot. Bail before the next chunk —
                // the KV built so far is freed with the slot.
                if (options.cancel_flag) |cf| {
                    if (cf.load(.acquire)) return error.Cancelled;
                }
                // Pick this chunk's end. Normal path: hit the configured chunk
                // size. Phase 1 path: if a checkpoint stride boundary lands
                // inside the would-be chunk, shrink the chunk so it ends
                // exactly on that boundary. That gives us an snapshot-point
                // every `stride` tokens without changing the model's seen
                // input — the forward result is identical to the unchunked
                // version because attention is causal and SSM/conv update
                // chunk-locally. Boundary alignment is in ABSOLUTE position
                // (pos + offset), so the saved snapshot list is correct for
                // the full prompt, not the truncated tail.
                const end = nextChunkEnd(pos, prefix_len, default_chunk, want_ssm_cp, ssm_cp_stride, ssm_cp_offset);
                const chunk_len: c_int = @intCast(end - pos);
                const chunk_shape = [_]c_int{ 1, chunk_len };
                const chunk_input = mlx.mlx_array_new_data(@ptrCast(&ids_i32[pos]), &chunk_shape, 2, .int32);
                defer _ = mlx.mlx_array_free(chunk_input);

                const chunk_start_ns = if (trace_enabled) prefill_sw.read() else 0;
                // Phase 2 experiment: when MLX_SERVE_COMPILE_FORWARD=1 wired a
                // compiled closure at load time, route this chunk through it.
                // The compiled closure uses xfm.defaultCtx (xfm.cache + xfm.ssm_entries),
                // which matches the prefill `ctx` when the scheduler has swapped
                // the slot's cache onto the Transformer (the single-slot legacy
                // and Phase-2-swapped path both satisfy this). Hidden-capture
                // and vision splice paths don't pass through this chunk loop
                // (they take the last_input branch), so they're already safe.
                // Optional-slice equality: same-ness here means both null or
                // both point at the same backing memory. We accept ssm_entries
                // null↔null too because plain-attn models legitimately have
                // both ctx and xfm carry null.
                const ssm_match = blk: {
                    if (ctx.ssm_entries == null and xfm.ssm_entries == null) break :blk true;
                    if (ctx.ssm_entries == null or xfm.ssm_entries == null) break :blk false;
                    break :blk ctx.ssm_entries.?.ptr == xfm.ssm_entries.?.ptr and
                        ctx.ssm_entries.?.len == xfm.ssm_entries.?.len;
                };
                // History windowing: a chunk before the window needs no
                // capture, which ALSO re-qualifies it for the compiled
                // trunk forward (capture is what disqualifies MTP chunks).
                const mtp_capture = mtp_active and chunkNeedsMtpHistory(pos, end, prefix_len, mtp_hist_window);
                var chunk_hidden_all = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(chunk_hidden_all);
                const chunk_logits = if (xfm.compiled_forward != null and
                    !mtp_capture and
                    ctx.cache == &xfm.cache and
                    ssm_match and
                    ctx.capture_hidden == null and
                    ctx.vision_embeddings == null)
                    try xfm.forwardCompiled(chunk_input)
                else if (mtp_capture) blk: {
                    var last_unused = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(last_unused);
                    break :blk try xfm.forwardWithCaptureAll(&ctx, chunk_input, &last_unused, &chunk_hidden_all);
                } else try xfm.forwardWith(&ctx, chunk_input);
                _ = mlx.mlx_array_free(chunk_logits);
                if (trace_enabled) chunked_ns += prefill_sw.read() - chunk_start_ns;

                // MTP history for this chunk: hiddens [pos, end) pair with
                // tokens [pos+1, end+1) — prompt_ids[end] always exists since
                // the chunk loop spans [0, prefix_len) and prompt_ids has
                // prefix_len + 1 entries.
                if (mtp_capture) {
                    try mtp_mod.appendHistory(
                        options.mtp.?,
                        xfm,
                        &mtp_cache.?,
                        prompt_ids[pos + 1 .. end + 1],
                        chunk_hidden_all,
                        @intCast(mtp_cache.?.step),
                    );
                }

                // Eval KV cache — materializes this chunk's K/V, frees activation graph
                const eval_start_ns = if (trace_enabled) prefill_sw.read() else 0;
                {
                    const eval_vec = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(eval_vec);
                    for (ctx.cache.entries) |*entry| {
                        if (!entry.initialized) continue;
                        _ = mlx.mlx_vector_array_append_value(eval_vec, entry.keys);
                        _ = mlx.mlx_vector_array_append_value(eval_vec, entry.values);
                    }
                    // Materialize this chunk's MTP history entries alongside
                    // the trunk KV so the chunk's activation graph (incl. the
                    // full-hidden capture) can be freed before the next chunk.
                    if (mtp_cache) |*mc| {
                        for (mc.entries) |*entry| {
                            if (!entry.initialized) continue;
                            _ = mlx.mlx_vector_array_append_value(eval_vec, entry.keys);
                            _ = mlx.mlx_vector_array_append_value(eval_vec, entry.values);
                        }
                    }
                    // Phase 1: also force SSM state to materialize so the
                    // snapshot we take below holds a concrete tensor, not a
                    // lazy node that would re-execute the prefill graph if
                    // anyone reads from it later.
                    const abs_end_for_cp = end + ssm_cp_offset;
                    const should_capture = want_ssm_cp and ssm_cp_stride > 0 and abs_end_for_cp % ssm_cp_stride == 0;
                    if (should_capture) {
                        for (ctx.ssm_entries.?) |*ssm| {
                            if (!ssm.initialized) continue;
                            if (ssm.conv_state.ctx != null) {
                                _ = mlx.mlx_vector_array_append_value(eval_vec, ssm.conv_state);
                            }
                            if (ssm.ssm_state.ctx != null) {
                                _ = mlx.mlx_vector_array_append_value(eval_vec, ssm.ssm_state);
                            }
                        }
                    }
                    _ = mlx.mlx_eval(eval_vec);
                }
                _ = mlx.mlx_clear_cache();
                if (trace_enabled) eval_ns += prefill_sw.read() - eval_start_ns;

                // Phase 1: snapshot SSM state at stride-aligned boundaries.
                // We snapshot AFTER the eval above so the underlying buffers
                // are realized; the snapshot is just a refcount-share of the
                // already-resident state.
                const abs_end_for_cp2 = end + ssm_cp_offset;
                if (want_ssm_cp and ssm_cp_stride > 0 and abs_end_for_cp2 % ssm_cp_stride == 0) {
                    const cp = try captureSsmCheckpoint(allocator, ctx.ssm_entries.?, abs_end_for_cp2, xfm.s);
                    try ssm_checkpoints.append(allocator, cp);
                    // Keep the buffer bounded — drop the oldest if we've
                    // accumulated more than the configured max. Front-removal
                    // is O(n) but `n` is tiny (≤ ssm_checkpoint_max). We keep
                    // the latest positions because they're closer to the
                    // end-of-prompt, which is where most multi-turn warm
                    // requests match.
                    if (options.ssm_checkpoint_max > 0 and
                        ssm_checkpoints.items.len > options.ssm_checkpoint_max)
                    {
                        var oldest = ssm_checkpoints.orderedRemove(0);
                        oldest.deinit(allocator);
                    }
                }

                pos = end;
                n_chunks += 1;
                // Publish progress once per chunk — same cadence discipline as
                // `inflight_generated_tokens` (once per decode tick), never per token.
                if (options.prefill_progress) |p| p.store(@intCast(pos), .monotonic);
            }

            // Phase 1: always-on snapshot at the post-prefill position
            // (= prefix_len, i.e., prompt_ids.len - 1). The stride loop
            // captures snapshots at [stride, 2*stride, ...]; this final
            // capture covers the most common warm-path case where the next
            // turn's prompt fully matches turn-1's prompt and matched lands
            // at prompt_ids.len. Without this, a stride=256 setup with a
            // 750-token prompt could only restore at position 512 (losing
            // ~234 tokens of potential reuse to the next stride boundary).
            // With it, the cache restores to position 749 (~99% of the
            // prompt) and only the last token + new tail re-forwards.
            // Skipped on `prompt_ids.len == 1` (no prefill chunks ran).
            if (want_ssm_cp and prefix_len > 0) {
                const final_abs = prefix_len + ssm_cp_offset;
                // Skip if we already captured at this exact position (the
                // chunked loop would have done so when prefix_len happens
                // to be a stride multiple).
                const already_have = ssm_checkpoints.items.len > 0 and
                    ssm_checkpoints.items[ssm_checkpoints.items.len - 1].pos == final_abs;
                if (!already_have) {
                    // SSM state is already materialized — the chunked loop
                    // evaluated it at every chunk boundary. The final chunk
                    // may have been a stride-aligned one (already evaluated)
                    // or a partial tail (also evaluated). The snapshot is a
                    // cheap refcount-share.
                    const cp = try captureSsmCheckpoint(allocator, ctx.ssm_entries.?, final_abs, xfm.s);
                    try ssm_checkpoints.append(allocator, cp);
                    if (options.ssm_checkpoint_max > 0 and
                        ssm_checkpoints.items.len > options.ssm_checkpoint_max)
                    {
                        var oldest = ssm_checkpoints.orderedRemove(0);
                        oldest.deinit(allocator);
                    }
                }
            }
        }

        // Process last token (or single token for len=1) — this applies lm_head
        // on just 1 token, producing the logits we need for sampling.
        const last_shape = [_]c_int{ 1, 1 };
        const last_idx = prompt_ids.len - 1;
        const last_input = mlx.mlx_array_new_data(&ids_i32[last_idx], &last_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(last_input);

        // Drafter (Gemma 4 assistant) needs the post-final-norm hidden as
        // its first-step h_prev — captured here so we don't need a second
        // forward at the start of `nextDrafter`.
        const drafter_active = options.drafter_enabled and options.drafter != null;
        const pld_active = options.pld_enabled;
        const need_capture = drafter_active or mtp_active;
        var captured_hidden: mlx.mlx_array = mlx.mlx_array_new();
        var has_captured_hidden = false;
        const last_start_ns = if (trace_enabled) prefill_sw.read() else 0;
        const logits = if (need_capture) blk: {
            has_captured_hidden = true;
            break :blk try xfm.forwardWithCapture(&ctx, last_input, &captured_hidden);
        } else try xfm.forwardWith(&ctx, last_input);
        if (trace_enabled) {
            const last_ns = prefill_sw.read() - last_start_ns;
            const total_ns = prefill_sw.read();
            const ms = std.time.ns_per_ms;
            std.debug.print(
                "  [prefill-trace] tokens={d} chunks={d} chunk_size={d} chunked={d}ms eval={d}ms last_token={d}ms total={d}ms{s}{s}\n",
                .{
                    prompt_ids.len,
                    n_chunks,
                    PREFILL_CHUNK,
                    chunked_ns / ms,
                    eval_ns / ms,
                    last_ns / ms,
                    total_ns / ms,
                    if (need_capture) " [capture-hidden]" else "",
                    if (pld_active) " [pld]" else "",
                },
            );
        }
        errdefer if (has_captured_hidden) {
            _ = mlx.mlx_array_free(captured_hidden);
        };

        // Attach the SSM-checkpoint buffer to whichever Generator variant
        // we're about to return. Clears the local list so the errdefer above
        // doesn't double-free. All four init paths below call this once
        // before returning their Generator.
        const attachCp = struct {
            fn f(g: *Generator, list: *std.ArrayList(SSMCheckpoint), a: std.mem.Allocator) void {
                g.ssm_checkpoints = list.*;
                g.ssm_checkpoint_alloc = a;
                list.* = std.ArrayList(SSMCheckpoint).empty;
            }
        }.f;

        // Constrained generation skips the lazy first-sample fast path: we cannot
        // sample the first token until we have applied the grammar mask, and we
        // cannot pipeline because grammar advancement depends on the realized id.
        if (sampling.constraint != null) {
            // Grammar-constrained requests never speculate; release the MTP
            // history cache if dispatch enabled it anyway.
            if (mtp_cache) |*mc| {
                mc.deinit();
                mtp_cache = null;
            }
            var gen = Generator{
                .xfm = xfm,
                .ctx = ctx,
                .tok = tok,
                .next_token_id = 0,
                .step = 0,
                .max_tokens = max_tokens,
                .sampling = sampling,
                .prompt_tokens = @intCast(prompt_ids.len),
                .completion_tokens = 0,
                .finish_reason = "length",
                .done = false,
                .eos_token_ids = eos_token_ids,
                .generated_ids = std.ArrayList(u32).empty,
                .timeout_ns = 0,
                .timer = io_util.Stopwatch.init(io),
                .last_hidden = if (has_captured_hidden) captured_hidden else mlx.mlx_array_new(),
                .has_last_hidden = has_captured_hidden,
                .prompt_ids_owned = prompt_owned,
                .prompt_ids_alloc = allocator,
            };
            gen.pending_logits = logits;
            gen.has_pending_logits = true;
            attachCp(&gen, &ssm_checkpoints, allocator);
            return gen;
        }

        // Drafter / PLD-v2 / MTP path: sample synchronously and DO NOT
        // pre-forward the sampled token. The first nextDrafter / nextPld /
        // nextMtp call needs the cache at exactly prompt_len (last prompt
        // token forwarded; first sampled token deferred). The lazy
        // pre-forward path below would over-advance the cache and corrupt
        // every verify forward.
        if (drafter_active or pld_active or mtp_active) {
            const sample_lazy = sampleTokenLazy(logits, sampling, s);
            _ = mlx.mlx_array_free(logits);
            try mlx.check(mlx.mlx_array_eval(sample_lazy));
            var first_val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&first_val, sample_lazy));
            _ = mlx.mlx_array_free(sample_lazy);

            const mtp_cost_profile: mtp_mod.MtpCostProfile = if (mtp_active)
                options.mtp.?.m5NaxCostProfile(xfm)
            else
                .generic;
            var gen = Generator{
                .xfm = xfm,
                .ctx = ctx,
                .tok = tok,
                .next_token_id = @intCast(first_val),
                .step = 0,
                .max_tokens = max_tokens,
                .sampling = sampling,
                .prompt_tokens = @intCast(prompt_ids.len),
                .completion_tokens = 0,
                .finish_reason = "length",
                .done = false,
                .eos_token_ids = eos_token_ids,
                .generated_ids = std.ArrayList(u32).empty,
                .timeout_ns = 0,
                .timer = io_util.Stopwatch.init(io),
                .last_hidden = if (need_capture) captured_hidden else mlx.mlx_array_new(),
                .has_last_hidden = need_capture,
                .prng = std.Random.DefaultPrng.init(sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds())),
                .prompt_ids_owned = prompt_owned,
                .prompt_ids_alloc = allocator,
                .drafter = if (drafter_active) options.drafter else null,
                .drafter_block_size = options.drafter_block_size,
                .mtp = if (mtp_active) options.mtp else null,
                .mtp_cache = mtp_cache,
                .mtp_depth = resolveMtpDepthCapForProfile(options.mtp_depth, mtp_cost_profile),
                .mtp_ev_costs = mtpEvCosts(mtp_cost_profile),
                // Start at depth 1 and climb with evidence: the cheap depth
                // is the safe default (1.11x on cold/creative content), and
                // hot workloads promote within ~8 rounds.
                .mtp_depth_current = 1,
            };
            mtp_cache = null; // ownership transferred to the Generator
            // pending_logits/pending_token left empty — the lazy pipeline is
            // skipped under PLD / drafter / MTP. The speculative `next*` paths
            // drive every subsequent step with predictable cache offset.
            attachCp(&gen, &ssm_checkpoints, allocator);
            return gen;
        }

        // Phase 2: scheduler-managed slots ask init to sample t1 synchronously
        // and skip the lazy pre-forward. Cache lands at prompt_len with t1 NOT
        // in cache — matches `forwardBatchedDecode`'s expectation and the
        // PLD / drafter init path's invariant. Generator.next's transition
        // shim handles the bootstrap on the first decode tick.
        if (options.skip_lazy_preforward) {
            const sample_lazy = sampleTokenLazy(logits, sampling, s);
            _ = mlx.mlx_array_free(logits);
            try mlx.check(mlx.mlx_array_eval(sample_lazy));
            var first_val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&first_val, sample_lazy));
            _ = mlx.mlx_array_free(sample_lazy);

            var gen = Generator{
                .xfm = xfm,
                .ctx = ctx,
                .tok = tok,
                .next_token_id = @intCast(first_val),
                .step = 0,
                .max_tokens = max_tokens,
                .sampling = sampling,
                .prompt_tokens = @intCast(prompt_ids.len),
                .completion_tokens = 0,
                .finish_reason = "length",
                .done = false,
                .eos_token_ids = eos_token_ids,
                .generated_ids = std.ArrayList(u32).empty,
                .timeout_ns = 0,
                .timer = io_util.Stopwatch.init(io),
                .last_hidden = if (has_captured_hidden) captured_hidden else mlx.mlx_array_new(),
                .has_last_hidden = has_captured_hidden,
                .prng = std.Random.DefaultPrng.init(sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds())),
                .prompt_ids_owned = prompt_owned,
                .prompt_ids_alloc = allocator,
            };
            attachCp(&gen, &ssm_checkpoints, allocator);
            return gen;
        }

        // Regular path: sample first token lazily, then build the next forward pass
        const lazy_token = sampleTokenLazy(logits, sampling, s);
        _ = mlx.mlx_array_free(logits);

        const next_logits = try lazyForward(xfm, &ctx, lazy_token);

        // Async-eval the decode pipeline (single-token graphs, much smaller)
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            _ = mlx.mlx_vector_array_append_value(eval_vec, lazy_token);
            _ = mlx.mlx_vector_array_append_value(eval_vec, next_logits);
            _ = mlx.mlx_async_eval(eval_vec);
        }

        // Sync to get the first token value
        try mlx.check(mlx.mlx_array_eval(lazy_token));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
        _ = mlx.mlx_array_free(lazy_token);

        var gen = Generator{
            .xfm = xfm,
            .ctx = ctx,
            .tok = tok,
            .next_token_id = @intCast(val),
            .step = 0,
            .max_tokens = max_tokens,
            .sampling = sampling,
            .prompt_tokens = @intCast(prompt_ids.len),
            .completion_tokens = 0,
            .finish_reason = "length",
            .done = false,
            .eos_token_ids = eos_token_ids,
            .generated_ids = std.ArrayList(u32).empty,
            .timeout_ns = 0,
            .timer = io_util.Stopwatch.init(io),
            .last_hidden = if (has_captured_hidden) captured_hidden else mlx.mlx_array_new(),
            .has_last_hidden = has_captured_hidden,
            .prng = std.Random.DefaultPrng.init(sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds())),
            .prompt_ids_owned = prompt_owned,
            .prompt_ids_alloc = allocator,
        };

        gen.pending_logits = next_logits;
        gen.has_pending_logits = true;

        attachCp(&gen, &ssm_checkpoints, allocator);
        return gen;
    }

    pub fn deinit(self: *Generator, allocator: std.mem.Allocator) void {
        if (self.last_logprob) |*lp| {
            allocator.free(lp.top_logprobs);
        }
        if (self.has_pending_logits) {
            _ = mlx.mlx_array_free(self.pending_logits);
            self.has_pending_logits = false;
        }
        if (self.has_pending_token) {
            _ = mlx.mlx_array_free(self.pending_token);
            self.has_pending_token = false;
        }
        if (self.has_last_hidden) {
            _ = mlx.mlx_array_free(self.last_hidden);
            self.has_last_hidden = false;
        }
        if (self.prompt_ids_alloc) |a| {
            a.free(self.prompt_ids_owned);
            self.prompt_ids_owned = &.{};
            self.prompt_ids_alloc = null;
        }
        if (self.mtp_cache) |*mc| {
            mc.deinit();
            self.mtp_cache = null;
        }
        if (self.mtp_hist_stash) |*st| {
            st.deinit();
            self.mtp_hist_stash = null;
        }
        // Publish the EV surface for the next request when the experimental
        // cross-request seed is explicitly enabled.
        // Only healthy runs qualify — a runtime-disabled or barely-sampled
        // run would poison the next request's plans (inference thread only,
        // same discipline as every other head-state write).
        if (self.mtp) |head| {
            if (mtpAdaptiveEnabled() and mtpEvSeedEnabled() and
                !self.spec_disabled_runtime and self.mtp_attempted >= 8)
            {
                head.ev_seed_accept = self.mtp_ev_accept;
                head.ev_seed_m_lo = self.mtp_ev_m_lo_prev;
            }
        }
        // Free any SSM checkpoints the caller didn't claim. Each layer-slice
        // was allocated by `ssm_checkpoint_alloc` (= the allocator passed to
        // `initWithOptions`), so we use that one. The ArrayList itself was
        // also created with that allocator.
        if (self.ssm_checkpoint_alloc) |a| {
            for (self.ssm_checkpoints.items) |*cp| cp.deinit(a);
            self.ssm_checkpoints.deinit(a);
            self.ssm_checkpoints = std.ArrayList(SSMCheckpoint).empty;
            self.ssm_checkpoint_alloc = null;
        } else {
            // Defensive: if init never set it, the list is empty — but the
            // backing ArrayList state may still need a deinit call. Use the
            // passed allocator as a fallback.
            self.ssm_checkpoints.deinit(allocator);
            self.ssm_checkpoints = std.ArrayList(SSMCheckpoint).empty;
        }
        self.generated_ids.deinit(allocator);
    }

    /// Transfer ownership of accumulated SSM checkpoints to the caller.
    /// Returns an owned slice; caller must free each `SSMCheckpoint` via
    /// `cp.deinit(allocator)` and the slice itself via `allocator.free`,
    /// where `allocator` is the same one passed to `initWithOptions`.
    /// After return, `ssm_checkpoints` is empty and the Generator owns
    /// nothing related to checkpoints.
    pub fn takeSsmCheckpoints(self: *Generator) []SSMCheckpoint {
        const a = self.ssm_checkpoint_alloc orelse return &[_]SSMCheckpoint{};
        const out = self.ssm_checkpoints.toOwnedSlice(a) catch return &[_]SSMCheckpoint{};
        return out;
    }

    /// Legacy→batched transition (scheduler.runBatchedDecodeTick): consume
    /// the lazy pipeline state so the slot can join a batched tick. The
    /// legacy pipelined decode keeps a lookahead token ALREADY FORWARDED
    /// into the KV cache (`pending_token` / `next_token_id`) plus
    /// `pending_logits` for the position after it. Dropping that state and
    /// re-forwarding `next_token_id` would append a duplicate position to
    /// the cache and re-emit an already-emitted token — corrupting every
    /// stream whose slot enters a batch mid-generation
    /// (tests/test_batched_transition.sh).
    ///
    /// Returns the token to emit this step (the pipelined lookahead), or
    /// null when generation stopped (`checkStop`: EOS / pad-run /
    /// max_tokens / timeout — `finish_reason` is set). On return:
    /// `next_token_id` is sampled but NOT in the cache and pending state is
    /// empty — exactly the batched-tick entry invariant.
    pub fn drainPipelineForBatch(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        try self.resolvePendingToken();
        if (try self.checkStop()) {
            if (self.has_pending_logits) {
                _ = mlx.mlx_array_free(self.pending_logits);
                self.has_pending_logits = false;
            }
            return null;
        }
        // Both pipeline shapes (fresh-from-prefill and post-`next()` fast
        // path) carry pending_logits alongside the in-cache lookahead; a
        // lookahead without logits would force a re-forward of an in-cache
        // token, which is the corruption this method exists to prevent.
        if (!self.has_pending_logits) return error.MissingPendingLogits;

        const token = self.next_token_id;
        self.completion_tokens += 1;
        self.step += 1;
        try self.generated_ids.append(allocator, token);
        if (self.step % 256 == 0) _ = mlx.mlx_clear_cache();

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        const lazy = sampleTokenLazy(step_logits, self.sampling, self.xfm.s);
        _ = mlx.mlx_array_free(step_logits);
        try mlx.check(mlx.mlx_array_eval(lazy));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
        _ = mlx.mlx_array_free(lazy);
        self.next_token_id = @intCast(val);
        return token;
    }

    /// Resolve the deferred pending token: eval the lazy array and extract the u32 value.
    /// This is called at the START of each iteration, giving the GPU maximum time
    /// to compute since the async_eval at the END of the previous iteration.
    fn resolvePendingToken(self: *Generator) !void {
        if (!self.has_pending_token) return;
        try mlx.check(mlx.mlx_array_eval(self.pending_token));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, self.pending_token));
        _ = mlx.mlx_array_free(self.pending_token);
        self.has_pending_token = false;
        self.next_token_id = @intCast(val);
    }

    const DrainResult = union(enum) {
        /// No pending pipeline state — the spec entry invariant already holds.
        already_clean,
        /// One token was emitted while draining; caller returns it this step.
        drained: u32,
        /// The drained token hit a stop condition — generation is over.
        stopped,
        /// Unexpected half-state; do not re-enable speculation.
        stay_disabled,
    };

    /// Transition from the pipelined `next()` state back to the spec-decode
    /// entry invariant (next_token_id known but NOT in cache, no pending
    /// state). The pipeline holds `pending_token` (lazy, its forward already
    /// in the cache) and `pending_logits` (logits for the position after it):
    /// resolving the token, emitting it, and sampling its successor from
    /// `pending_logits` WITHOUT forwarding lands exactly on the invariant.
    /// One sync. Also handles the shim-seeded state (`pending_logits` only).
    fn drainPipelineForSpec(self: *Generator, allocator: std.mem.Allocator) !DrainResult {
        if (!self.has_pending_logits) {
            if (!self.has_pending_token) return .already_clean;
            // pending_token without pending_logits never occurs in the
            // pipelined state machine; bail rather than risk the invariant.
            return .stay_disabled;
        }
        try self.resolvePendingToken();
        if (try self.checkStop()) return .stopped;
        const token = self.next_token_id;
        self.completion_tokens += 1;
        self.step += 1;
        try self.generated_ids.append(allocator, token);
        if (self.step % 256 == 0) _ = mlx.mlx_clear_cache();

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        const lazy = sampleTokenLazy(step_logits, self.sampling, self.xfm.s);
        _ = mlx.mlx_array_free(step_logits);
        try mlx.check(mlx.mlx_array_eval(lazy));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
        _ = mlx.mlx_array_free(lazy);
        self.next_token_id = @intCast(val);
        return .{ .drained = token };
    }

    /// Result of one `nextPld` step. Yields 1..=(1+max_draft_len) tokens.
    /// Caller owns `tokens` (must `allocator.free` it).
    pub const PldStepResult = struct {
        /// Tokens to emit this step (always at least the already-decided t1).
        /// On a full-accept, contains [t1, ...all_drafts]. On partial accept j,
        /// contains [t1, draft[0..j]] (the corrected fallback is stored as the
        /// generator's pending `next_token_id`, NOT included here — same
        /// "pending becomes next-step's first" convention as `nextDrafter`).
        tokens: []const u32,
        /// Number of *drafted* tokens accepted (not counting t1). 0..=draft_len.
        accepted_tokens: u32,
        /// Whether n-gram lookup found a candidate this step. False means PLD
        /// degraded to a single regular forward (no speculative work done).
        used_lookup: bool,
    };

    /// PLD draft+verify decode step. The draft comes from an n-gram lookup
    /// over `prompt_ids_owned ++ generated_ids`, NOT a model call — that's
    /// what makes PLD model-agnostic and cheap.
    ///
    /// `key_len` is the n-gram size used for matching (default 3). `draft_len`
    /// is the maximum number of speculative tokens to verify per step (default
    /// 5). Both are clamped to safe upper bounds internally.
    ///
    /// Returns `null` only when generation is already done. When no n-gram
    /// match exists (cold start, novel output), falls back to the regular
    /// `next()` path and returns a single-token result with `used_lookup=false`.
    pub fn nextPld(
        self: *Generator,
        allocator: std.mem.Allocator,
        draft_len: u32,
        key_len: u32,
    ) !?PldStepResult {
        if (self.done) return null;
        std.debug.assert(self.sampling.constraint == null); // PLD + grammar not supported
        std.debug.assert(self.logprobs_n == 0); // PLD + logprobs not supported

        // Runtime acceptance gate: if a prior step set the flag, fall back
        // to the regular `next()` path. Under v2, PLD's exit invariant has
        // `t1 NOT in cache` (matches `nextDrafter`) — `next()`'s transition
        // shim seeds `pending_logits` synchronously via `forward([t1])` when
        // it sees `!has_pending_logits and !has_pending_token`. So the
        // hand-off works even though pending state is empty.
        if (self.spec_disabled_runtime) {
            self.disabled_steps += 1;
            // Periodic re-enable check: when the generated tail turns
            // repetitive (file/tool echo after a novel preamble), PLD pays
            // again. Drain the `next()` pipeline back to the spec entry
            // invariant; the drained token (if any) is this step's emit and
            // speculation resumes on the following call.
            if (self.disabled_steps % SPEC_REENABLE_INTERVAL == 0) reenable: {
                const gen = self.generated_ids.items;
                const prompt_toks = self.prompt_ids_owned;
                const committed_check = try allocator.alloc(u32, prompt_toks.len + gen.len);
                defer allocator.free(committed_check);
                @memcpy(committed_check[0..prompt_toks.len], prompt_toks);
                @memcpy(committed_check[prompt_toks.len..], gen);
                //if (log.isDebug()) {
                //    const dbg_frac = pld_index.tailMatchFraction(committed_check, @min(SPEC_REENABLE_WINDOW, gen.len), 3);
                //    log.debug("  pld re-enable check: disabled_steps={d} gen={d} tail_match={d:.2}\n", .{ self.disabled_steps, gen.len, dbg_frac });
                //}
                if (!specShouldReenable(committed_check, gen.len)) break :reenable;
                switch (try self.drainPipelineForSpec(allocator)) {
                    .stay_disabled => break :reenable,
                    .stopped => return null,
                    .already_clean => {
                        //log.info("  pld=re-enabled (generated tail turned repetitive after {d} disabled steps)\n", .{self.disabled_steps});
                        self.spec_disabled_runtime = false;
                        self.disabled_steps = 0;
                        self.yield_steps = 0;
                        self.yield_accepted = 0;
                        // Invariant already holds — fall through to the
                        // enabled flow below in this same call.
                    },
                    .drained => |drained_tok| {
                        //log.info("  pld=re-enabled (generated tail turned repetitive after {d} disabled steps)\n", .{self.disabled_steps});
                        self.spec_disabled_runtime = false;
                        self.disabled_steps = 0;
                        self.yield_steps = 0;
                        self.yield_accepted = 0;
                        const tokens = try allocator.alloc(u32, 1);
                        tokens[0] = drained_tok;
                        return PldStepResult{
                            .tokens = tokens,
                            .accepted_tokens = 0,
                            .used_lookup = false,
                        };
                    },
                }
            }
            if (self.spec_disabled_runtime) {
                const tok_opt = try self.next(allocator);
                if (tok_opt == null) return null;
                const tokens = try allocator.alloc(u32, 1);
                tokens[0] = tok_opt.?;
                return PldStepResult{
                    .tokens = tokens,
                    .accepted_tokens = 0,
                    .used_lookup = false,
                };
            }
        }

        const xfm = self.xfm;
        const s = xfm.s;

        // ── INVARIANT going INTO this call (mirrors `nextDrafter`) ──
        //   cache.step = prompt_len + tokens_emitted   (NOT + 1)
        //   t1 = next_token_id (= "this step's first emit"); NOT in cache yet.
        //   pending_logits / pending_token are empty (init's PLD branch and
        //   every nextPld exit leave them empty under v2).
        //
        // Cold path (no n-gram match): forward([t1]) length 1 advances cache
        // by 1, produces logits at position +1 → sample lookahead → emit t1,
        // set next_token_id = lookahead. Loses A's lazy pipeline overlap on
        // cold steps; the prompt-time n-gram gate disables PLD on novel
        // content where cold-path dominates.
        //
        // Verify path: input = `[t1, draft[0..m-1]]` length 1+m. Walk
        // verify_logits[i] vs draft[i] for i=0..m-1; full accept commits 1+m
        // tokens and exits with cache at prompt_len + TE_new (no post-step
        // forward — that is the per-step saving over v1).
        const t1: u32 = self.next_token_id;

        // Cap draft_len so the verify forward stays a small fixed cost.
        const max_draft: u32 = @min(draft_len, 15);
        const klen: u32 = @max(@as(u32, 1), key_len);

        // ── Phase 1: Lookup ──
        // committed = prompt + generated_ids + [t1]. Key = trailing klen tokens
        // (ends at t1). The lookup returns candidates for "what comes after t1".
        const prompt = self.prompt_ids_owned;
        const generated = self.generated_ids.items;
        const total_len = prompt.len + generated.len + 1;

        var committed = try allocator.alloc(u32, total_len);
        defer allocator.free(committed);
        @memcpy(committed[0..prompt.len], prompt);
        @memcpy(committed[prompt.len .. prompt.len + generated.len], generated);
        committed[total_len - 1] = t1;

        var draft_slice: ?[]const u32 = null;
        if (klen <= total_len - 1) {
            const key_start = total_len - klen;
            const key = committed[key_start..total_len];
            const lookup = pld_index.PldLookup{ .committed = committed, .key_len = klen };
            draft_slice = lookup.findMatch(key, max_draft);
        }
        if (draft_slice) |d| {
            if (d.len == 0) draft_slice = null;
        }

        const stochastic = self.sampling.temperature > 0.01;

        // ── Phase 2: Cold path (no n-gram match) ──
        // Forward([t1]) length 1: cache.step += 1, produces logits at that
        // position. Sample the lookahead, emit t1, set next_token_id =
        // lookahead. Cache exits at prompt_len + TE_new where TE_new = TE + 1.
        if (draft_slice == null) {
            const t1_i32: i32 = @intCast(t1);
            const t1_shape = [_]c_int{ 1, 1 };
            const t1_input = mlx.mlx_array_new_data(&t1_i32, &t1_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(t1_input);

            const cold_logits = try xfm.forwardWith(&self.ctx, t1_input); // cache.step += 1
            defer _ = mlx.mlx_array_free(cold_logits);

            const lazy = sampleTokenLazy(cold_logits, self.sampling, s);
            try mlx.check(mlx.mlx_array_eval(lazy));
            var lv: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&lv, lazy));
            _ = mlx.mlx_array_free(lazy);
            const new_t1: u32 = @intCast(lv);

            try self.generated_ids.append(allocator, t1);
            self.completion_tokens += 1;
            self.step += 1;
            if (self.step % 256 == 0) _ = mlx.mlx_clear_cache();
            self.next_token_id = new_t1;

            // Yield gate: cold steps pay the unpipelined forward; if the
            // workload isn't yielding accepted drafts to pay for it, fall
            // back to the pipelined `next()` (re-enable check above can
            // bring PLD back when the tail turns repetitive).
            self.yield_steps += 1;
            if (yieldGateShouldDisable(self.yield_steps, self.yield_accepted)) {
                log.info(
                    "  pld=disabled (yield gate: {d} drafted tokens over {d} steps < {d:.2}/step)\n",
                    .{ self.yield_accepted, self.yield_steps, YIELD_GATE_MIN_YIELD },
                );
                self.spec_disabled_runtime = true;
                self.disabled_steps = 0;
            }

            const tokens = try allocator.alloc(u32, 1);
            tokens[0] = t1;
            return PldStepResult{
                .tokens = tokens,
                .accepted_tokens = 0,
                .used_lookup = false,
            };
        }

        const draft = draft_slice.?;
        const m: u32 = @intCast(draft.len);

        // ── Phase 3: Snapshot KV + per-layer SSM + moe_seq_offset + DSV4 ──
        // Cache enters at cache.step = prompt_len + TE.
        //
        // The snapshots below are the FALLBACK rollback path (pure-attention,
        // Mamba2, LFM2). On a GatedDeltaNet trunk the verify forward instead
        // CAPTURES per-position SSM/conv state (capture_ssm_seq), and partial
        // accept rolls back by slicing that capture + truncating the KV cache —
        // no re-forward of the accepted prefix, which on this arch re-runs the
        // expensive 48-layer sequential recurrence.
        //
        // The KV-cache truncate length is anchored on `moe_seq_offset`, NOT
        // `cache.step`: on a GDN trunk layer 0 is a linear-attention layer that
        // never calls `cache.update`, and `cache.step` only advances under
        // `if (layer == 0)` — so it stays stale (~0) for this family. The
        // full-attention KV entries instead track `moe_seq_offset` (both advance
        // by seq_len per forward), so that is the real KV length to roll back to.
        var kv_snap = try self.ctx.cache.snapshot();
        defer kv_snap.deinit();
        var ssm_snaps: ?[]SSMCacheEntrySnapshot = null;
        defer if (ssm_snaps) |snaps| {
            for (snaps) |*sn| ssmSnapshotDeinit(sn);
            xfm.allocator.free(snaps);
        };
        if (self.ctx.ssm_entries) |entries| {
            const out = try xfm.allocator.alloc(SSMCacheEntrySnapshot, entries.len);
            for (entries, 0..) |*entry, i| out[i] = ssmSnapshot(entry);
            ssm_snaps = out;
        }
        const moe_seq_offset_snap = self.ctx.moe_seq_offset.*;

        // ── Phase 4: Verify forward `[t1, draft[0..m-1]]` length 1+m ──
        // cache.step at start = prompt_len + TE; after = prompt_len + TE + 1 + m.
        //   verify_logits[0]   predicts the slot AFTER t1     → candidate for draft[0]
        //   verify_logits[i]   predicts the slot AFTER draft[i-1] (i = 1..m-1)
        //                                                     → candidate for draft[i]
        //   verify_logits[m]   predicts the slot AFTER draft[m-1]
        //                                                     → "bonus" position (full-accept new_t1)
        const seq_len: c_int = @intCast(1 + m);
        const verify_input_buf = try allocator.alloc(i32, 1 + m);
        defer allocator.free(verify_input_buf);
        verify_input_buf[0] = @intCast(t1);
        for (draft, 0..) |d, i| verify_input_buf[1 + i] = @intCast(d);
        const verify_shape = [_]c_int{ 1, seq_len };
        const verify_input = mlx.mlx_array_new_data(verify_input_buf.ptr, &verify_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(verify_input);

        // Enable per-position SSM capture for the verify pass on a GDN trunk so
        // partial accept can roll back without a re-forward. Self-detecting:
        // only GatedDeltaNet layers actually populate `spec_state_seq`, so
        // pure-attention / Mamba2 / LFM2 fall through to the snapshot fallback.
        self.ctx.capture_ssm_seq = self.ctx.ssm_entries != null;
        const verify_logits = try xfm.forwardWith(&self.ctx, verify_input);
        self.ctx.capture_ssm_seq = false;
        // Always free the transient capture buffers before returning, however
        // we exit this round (full accept, partial accept, or error).
        defer if (self.ctx.ssm_entries) |entries| {
            for (entries) |*entry| transformer_mod.ssmFreeSpecCapture(entry);
        };
        // verify_logits shape [1, 1+m, V]. Sliced and freed below.
        self.pld_attempted += 1;

        const vl_shape = mlx.getShape(verify_logits);
        const slice_strides = [_]c_int{ 1, 1, 1 };

        // Slice all 1+m per-position logits up front so we can sample the
        // correction from the original verify forward (cache state aligned)
        // without re-running forward, and re-use them for both stochastic
        // accept tests and the correction sample.
        const per_pos_logits = try allocator.alloc(mlx.mlx_array, 1 + m);
        defer {
            for (per_pos_logits) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(per_pos_logits);
        }
        for (per_pos_logits, 0..) |*slot, idx| {
            slot.* = mlx.mlx_array_new();
            const start = [_]c_int{ 0, @intCast(idx), 0 };
            const stop = [_]c_int{ vl_shape[0], @as(c_int, @intCast(idx)) + 1, vl_shape[2] };
            try mlx.check(mlx.mlx_slice(slot, verify_logits, &start, 3, &stop, 3, &slice_strides, 3, s));
        }
        _ = mlx.mlx_array_free(verify_logits);

        // ── Phase 5: Walk drafts. accepted ∈ [0, m]. Full accept = m. ──
        // verify_logits[i] is the prediction for draft[i] (i = 0..m-1).
        // No separate "first-position" test under v2 — the verify forward
        // covers it.
        var accepted: u32 = 0;
        if (stochastic) {
            var i: u32 = 0;
            while (i < m) : (i += 1) {
                const target_p = try probsAtLastPos(per_pos_logits[i], self.sampling, s);
                defer _ = mlx.mlx_array_free(target_p);
                const p_draft = try probAt(target_p, draft[i], s);
                const accept_prob: f32 = @min(1.0, p_draft);
                const u: f32 = self.prng.random().float(f32);
                if (u >= accept_prob) break;
                accepted += 1;
            }
        } else {
            var i: u32 = 0;
            while (i < m) : (i += 1) {
                var argmax_arr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(argmax_arr);
                try mlx.check(mlx.mlx_argmax_axis(&argmax_arr, per_pos_logits[i], 2, false, s));
                try mlx.check(mlx.mlx_array_eval(argmax_arr));
                var argmax_val: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&argmax_val, argmax_arr));
                if (@as(u32, @intCast(argmax_val)) != draft[i]) break;
                accepted += 1;
            }
        }
        const full_accept = accepted == m;

        // ── Phase 6: Sample new_t1 from per_pos_logits[accepted] ──
        //   - full accept (accepted == m): per_pos_logits[m] predicts the slot
        //     after the last accepted draft (= "bonus" token).
        //   - partial (accepted < m):  per_pos_logits[accepted] is the model's
        //     prediction at the rejected slot. Stochastic samples from the
        //     residual `max(target_p − one_hot(draft[accepted]), 0)` to preserve
        //     the marginal distribution conditional on "not draft[accepted]"
        //     (Leviathan et al). Greedy: argmax of the rejected slot's logits.
        //
        // This indexing differs from v1: v1 sampled from `verify_logits[accepted-1]`
        // because t1 occupied no input slot; v2 has t1 at index 0 of the verify
        // input, so the "prediction one past the last accepted" lives at
        // index `accepted`. Off-by-one here would silently corrupt output.
        const correction_logits = per_pos_logits[accepted];
        const new_t1: u32 = blk: {
            if (stochastic) {
                const probs = try probsAtLastPos(correction_logits, self.sampling, s);
                defer _ = mlx.mlx_array_free(probs);
                if (!full_accept) {
                    const onehot = try pldOneHotRow(draft[accepted], vl_shape[2], s);
                    defer _ = mlx.mlx_array_free(onehot);
                    break :blk try sampleResidual(probs, onehot, s);
                } else {
                    break :blk try sampleFromProbs(probs, s);
                }
            } else {
                const lazy = sampleTokenLazy(correction_logits, self.sampling, s);
                try mlx.check(mlx.mlx_array_eval(lazy));
                var v: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&v, lazy));
                _ = mlx.mlx_array_free(lazy);
                break :blk @intCast(v);
            }
        };

        // ── Phase 7: Cache rollback on partial accept ──
        // After verify (length 1+m), cache.step = prompt_len + TE + 1 + m.
        // Full accept: TE_new = TE + 1 + m → no rollback.
        // Partial: must land at prompt_len + TE + 1 + accepted = prompt_len + TE_new
        // (TE_new = TE + 1 + accepted). Rollback then re-forward
        // `[t1, draft[0..accepted-1]]` length 1+accepted (with hidden capture
        // not needed here — just the cache advance).
        //
        // The accepted=0 case (= first draft rejected) MUST still re-forward
        // [t1] length 1: in v1 the t1 forward had been done eagerly before
        // verify; v2 includes t1 IN the verify forward, so rollback rolls
        // both t1 AND the drafts. Skipping the re-forward here would leave
        // the cache at prompt_len + TE — one short of the post-emit invariant.
        if (!full_accept) {
            // Fast GatedDeltaNet path: the verify forward captured per-position
            // SSM/conv state, so roll back by truncating the KV cache to the
            // accepted length (keeping verify's already-correct K/V for those
            // positions) and slicing the captured state — NO re-forward of the
            // accepted prefix. Detect via a populated capture on the first SSM
            // entry; absent it (pure-attention / Mamba2 / LFM2) we take the
            // proven restore + re-forward fallback below. Byte-identical either
            // way (pinned by tests/test_pld_equivalence.sh).
            const gdn_captured = if (self.ctx.ssm_entries) |entries|
                entries.len > 0 and entries[0].spec_state_seq.ctx != null
            else
                false;

            if (gdn_captured) {
                const accepted_len: usize = 1 + @as(usize, accepted);
                // `truncate` overwrites cache.step with its length arg; on this
                // family cache.step is a stale counter the model never reads
                // (positioning is moe_seq_offset), so preserve the snapshot's
                // value to keep the prefix cache's kv_step bookkeeping identical
                // to the restore-based fallback.
                const step_keep = kv_snap.step;
                try self.ctx.cache.truncate(moe_seq_offset_snap + accepted_len, s);
                self.ctx.cache.step = step_keep;
                for (self.ctx.ssm_entries.?) |*entry| {
                    try transformer_mod.ssmRollbackFromCapture(entry, accepted, s);
                }
                self.ctx.moe_seq_offset.* = moe_seq_offset_snap + accepted_len;
            } else {
                try self.ctx.cache.restore(&kv_snap);
                if (ssm_snaps) |snaps| {
                    for (self.ctx.ssm_entries.?, snaps) |*entry, *sn| try ssmRestore(entry, sn);
                }
                self.ctx.moe_seq_offset.* = moe_seq_offset_snap;

                const re_seq_len: c_int = @intCast(1 + accepted);
                const re_input_buf = try allocator.alloc(i32, 1 + accepted);
                defer allocator.free(re_input_buf);
                re_input_buf[0] = @intCast(t1);
                for (draft[0..accepted], 0..) |d, i| re_input_buf[1 + i] = @intCast(d);
                const re_shape = [_]c_int{ 1, re_seq_len };
                const re_input = mlx.mlx_array_new_data(re_input_buf.ptr, &re_shape, 2, .int32);
                defer _ = mlx.mlx_array_free(re_input);
                const re_logits = try xfm.forwardWith(&self.ctx, re_input);
                _ = mlx.mlx_array_free(re_logits);
            }
        }

        // ── Phase 8: Commit emitted tokens ──
        // Tokens emitted: [t1, draft[0..accepted]] = 1 + accepted.
        const num_emit: u32 = 1 + accepted;
        const tokens = try allocator.alloc(u32, num_emit);
        tokens[0] = t1;
        for (draft[0..accepted], 0..) |d, i| tokens[1 + i] = d;

        try self.generated_ids.append(allocator, t1);
        for (draft[0..accepted]) |d| try self.generated_ids.append(allocator, d);

        self.pld_accepted_tokens += accepted;
        self.completion_tokens += num_emit;
        self.step += num_emit;
        if (self.step % 256 == 0) _ = mlx.mlx_clear_cache();

        // No post-step forward — `next_token_id = new_t1` and exit. The next
        // nextPld call sees t1 NOT in cache (new invariant).
        self.next_token_id = new_t1;

        // Yield-gate accounting for verify steps (cold steps update in their
        // own branch above).
        self.yield_steps += 1;
        self.yield_accepted += accepted;

        // Runtime acceptance gate: after warmup, if the per-draft acceptance
        // probability is below the threshold, disable speculation for the rest
        // of this request (the re-enable check can bring it back when the
        // generated tail turns repetitive). PLD's `drafts_per_round` is the
        // upper-bound draft length (`max_draft`); matches with shorter accepts
        // still divide by this max so a workload with consistently-short
        // n-gram matches DOES get throttled.
        if (runtimeGateShouldDisable(self.pld_attempted, self.pld_accepted_tokens, max_draft)) {
            const drafts_proposed: u64 = self.pld_attempted * @as(u64, max_draft);
            const rate: f32 = if (drafts_proposed > 0)
                @as(f32, @floatFromInt(self.pld_accepted_tokens)) /
                    @as(f32, @floatFromInt(drafts_proposed))
            else
                0.0;
            log.info(
                "  pld=disabled (runtime per-draft rate {d:.2} < {d:.2} after {d} attempts)\n",
                .{ rate, RUNTIME_GATE_MIN_PER_DRAFT_RATE, self.pld_attempted },
            );
            self.spec_disabled_runtime = true;
            self.disabled_steps = 0;
        }

        return PldStepResult{
            .tokens = tokens,
            .accepted_tokens = accepted,
            .used_lookup = true,
        };
    }

    /// Result of one `nextDrafter` step. Same shape as PLD's result so the
    /// outer wrapper can share token-emit / EOS-check logic.
    pub const DrafterStepResult = struct {
        /// Tokens to emit this step. On a full accept this is
        /// `[t1, ...all_drafts]` (length `block_size`); on partial accept j
        /// it is `[t1, draft[0..j]]` (length `1+j`). The corrected fallback
        /// becomes `next_token_id` for the next call.
        tokens: []const u32,
        /// Number of *drafted* tokens accepted (excludes always-accepted t1).
        accepted_tokens: u32,
    };

    /// Drafter-assisted decode step. Mirrors `nextPld` but the draft comes
    /// from `block_size - 1` autoregressive forwards through the Gemma 4
    /// assistant drafter (cross-attending into target's KV) instead of an
    /// n-gram lookup. Verify is identical: target forward over
    /// `[t1, draft0..draft_{m-1}]` with greedy / stochastic accept.
    ///
    /// Algorithm:
    ///   1. Run `block_size - 1` drafter steps. Each step's input is
    ///      `concat(target.embed(prev_tok)*scale, h_prev)`. `prev_tok` starts
    ///      at `next_token_id` (= t1); after step i it's the just-sampled
    ///      `draft[i]`. `h_prev` starts at `last_hidden` (captured at
    ///      prefill or the previous accept's verify-forward); after step i
    ///      it's the drafter's own `post_proj` output.
    ///      All drafter forwards in one round share `rope_offset =
    ///      target.cache.step` (per upstream `set_shared_kv`).
    ///   2. Snapshot KV + SSM, run target verify forward over
    ///      `[t1, draft0..draft_{m-1}]` length `block_size` with
    ///      `forwardCaptureHidden` so we get the new `h_prev` at position m.
    ///   3. Walk argmax(verify_logits[i]) vs draft[i] for i in 0..m-1.
    ///      Greedy: equal → accept. Stochastic: standard speculative-decoding
    ///      ratio test using `probAt(target_p, draft[i])` (the drafter's
    ///      masked-LM-head produces probabilistic logits, so we treat its
    ///      sampled draft as a one-hot proposal — same simplification PLD
    ///      uses).
    ///   4. Full accept (j == m): emit drafts, sample new pending from
    ///      verify_logits[m-1] (the target's prediction one position past the
    ///      last accepted draft — already computed during verify), update
    ///      `last_hidden` to the captured post-final-norm hidden.
    ///   5. Partial accept (j < m): roll back KV+SSM, re-forward
    ///      `[t1, draft[0..j-1]]` length `j+1` (with hidden capture) so
    ///      cache lands at exactly `+j+1`. Sample correction from the
    ///      *original* verify_logits[j] (the model's prediction at the
    ///      rejected position).
    pub fn nextDrafter(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
        if (self.done) return null;
        std.debug.assert(self.drafter != null);
        std.debug.assert(self.has_last_hidden); // captured at init or last accept
        std.debug.assert(self.sampling.constraint == null); // grammar + drafter unsupported
        std.debug.assert(self.logprobs_n == 0); // logprobs + drafter unsupported

        // Runtime acceptance gate: if a prior step set the flag, fall back
        // to the regular `next()` path. Drafter's exit invariant is "t1 NOT
        // in cache" (different from `next()`'s expected entry), so `next()`
        // contains a transition shim that synchronously seeds pending_logits
        // when has_pending_logits is false. The shim makes this hand-off safe.
        if (self.spec_disabled_runtime) {
            const tok_opt = try self.next(allocator);
            if (tok_opt == null) return null;
            const tokens = try allocator.alloc(u32, 1);
            tokens[0] = tok_opt.?;
            return DrafterStepResult{
                .tokens = tokens,
                .accepted_tokens = 0,
            };
        }

        const xfm = self.xfm;
        const s = xfm.s;
        const drafter = self.drafter.?;
        const m: u32 = @max(@as(u32, 1), self.drafter_block_size - 1);
        const t1: u32 = self.next_token_id; // already-decided token at position cache.step

        // RoPE offset: position the drafter's queries rotate by. Per upstream
        // `set_shared_kv`, this is `target.cache.step` and stays constant
        // across all `m` drafter steps in this round.
        const rope_offset: c_int = @intCast(self.ctx.cache.step);

        // ── Phase 1: draft `m` tokens lazily, no per-step CPU sync ──
        //
        // The drafter loop builds a chained lazy graph: each step's sampled
        // token is a [1]-shaped mlx_array fed directly to the next step's
        // `embedTargetTokenArr` as the indexer, and forward as the next step's
        // `prev_token`. No `mlx_array_eval` / `mlx_array_item_int32` calls
        // here — the entire m-step chain plus the verify forward (built
        // below) materialize as a single async graph and evaluate together.
        // For block_size=8 (31B), this collapses 7 GPU→CPU syncs into 0,
        // saving ~70-100ms of Metal command-buffer sync latency per round.
        var drafts = try allocator.alloc(u32, m);
        errdefer allocator.free(drafts);

        // `draft_arrs[i]` is the lazy [1] argmax output of drafter step i.
        // Owned here; freed at end of nextDrafter (after verify uses them).
        const draft_arrs = try allocator.alloc(mlx.mlx_array, m);
        defer {
            for (draft_arrs) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(draft_arrs);
        }

        // Wrap t1 as a [1] mlx_array so the FIRST drafter step can use the
        // same lazy-chain helper as subsequent steps. This array is also
        // reshaped + reused as the leading element of the verify input below.
        const t1_i32: i32 = @intCast(t1);
        const t1_shape = [_]c_int{1};
        const t1_arr = mlx.mlx_array_new_data(&t1_i32, &t1_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(t1_arr);

        // `h_prev_owner` rolls forward through the drafter. Starts at the
        // captured target hidden; subsequent steps use the drafter's
        // post_proj output. The output is itself a lazy mlx_array, so the
        // chain stays lazy across all m steps.
        var h_prev_owner: ?mlx.mlx_array = null;
        defer if (h_prev_owner) |h| {
            _ = mlx.mlx_array_free(h);
        };

        {
            var prev_tok_arr: mlx.mlx_array = t1_arr;
            var i: u32 = 0;
            while (i < m) : (i += 1) {
                const h_prev_arg: mlx.mlx_array = if (h_prev_owner) |h| h else self.last_hidden;
                const step_out = try drafter_mod.stepArr(drafter, xfm, self.ctx.cache, prev_tok_arr, h_prev_arg, rope_offset);
                // Sample lazily — `sampleTokenLazy` for greedy returns the
                // argmax as a [1]-shaped lazy array. NO eval here.
                draft_arrs[i] = sampleTokenLazy(step_out.logits, self.sampling, s);
                _ = mlx.mlx_array_free(step_out.logits);

                // Roll h_prev forward.
                if (h_prev_owner) |h_old| {
                    _ = mlx.mlx_array_free(h_old);
                }
                h_prev_owner = step_out.h_prev_next;
                // The next step's prev_token is THIS step's lazy sample.
                prev_tok_arr = draft_arrs[i];
            }
        }

        // ── Phase 2: snapshot KV + SSM + DSV4 ──
        var kv_snap = try self.ctx.cache.snapshot();
        defer kv_snap.deinit();
        var ssm_snaps: ?[]SSMCacheEntrySnapshot = null;
        defer if (ssm_snaps) |snaps| {
            for (snaps) |*sn| ssmSnapshotDeinit(sn);
            xfm.allocator.free(snaps);
        };
        if (self.ctx.ssm_entries) |entries| {
            const out = try xfm.allocator.alloc(SSMCacheEntrySnapshot, entries.len);
            for (entries, 0..) |*entry, idx| out[idx] = ssmSnapshot(entry);
            ssm_snaps = out;
        }
        const moe_seq_offset_snap = self.ctx.moe_seq_offset.*;

        // ── Phase 3: build verify input by concatenating [t1, drafts...] ──
        //
        // Build verify_input as a [1, 1+m] tensor without any CPU sync. The
        // m draft tokens are still lazy mlx_arrays at this point; we reshape
        // each [1] → [1,1] and stack along axis=1 with t1 reshaped the same
        // way. The forward pass that consumes verify_input is then chained
        // onto the drafter's lazy graph.
        const reshape_2d = [_]c_int{ 1, 1 };
        var t1_2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t1_2d);
        try mlx.check(mlx.mlx_reshape(&t1_2d, t1_arr, &reshape_2d, 2, s));

        // Stack: each draft_arr[i] is shape [1]; reshape each to [1,1] and
        // collect into a vector_array along with t1_2d, then concat axis=1.
        var verify_input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(verify_input);
        {
            const drafts_2d = try allocator.alloc(mlx.mlx_array, m);
            defer {
                for (drafts_2d) |arr| _ = mlx.mlx_array_free(arr);
                allocator.free(drafts_2d);
            }
            for (draft_arrs, drafts_2d) |dlazy, *out| {
                out.* = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_reshape(out, dlazy, &reshape_2d, 2, s));
            }
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, t1_2d);
            for (drafts_2d) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
            try mlx.check(mlx.mlx_concatenate_axis(&verify_input, vec, 1, s));
        }

        var new_hidden = mlx.mlx_array_new();
        // Captures the post-final-norm hidden at the LAST input position
        // (= position m, predicting the bonus token if all drafts accept).
        const verify_logits = try xfm.forwardWithCapture(&self.ctx, verify_input, &new_hidden);
        // verify_logits shape: [1, 1+m, V]
        self.drafter_attempted += 1;

        // ── Phase 4: decide longest accepted prefix ──
        //
        // Greedy mode: argmax over the entire [1, 1+m, V] verify_logits in
        // one op (yields [1, 1+m] indices). Stochastic mode: sample-residual
        // / accept-prob path needs per-position logits, so it slices below.
        // Either way, we collapse all per-step syncs into ONE eval at the
        // end of this round.
        const stochastic = self.sampling.temperature > 0.01;
        const vl_shape = mlx.getShape(verify_logits);

        // Stochastic path needs per-position logits to compute target probs
        // and (on partial accept) build the residual. Greedy path skips
        // slicing entirely. `per_pos_logits` is null in greedy mode.
        var per_pos_logits: ?[]mlx.mlx_array = null;
        defer if (per_pos_logits) |slots| {
            for (slots) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(slots);
        };
        if (stochastic) {
            const slots = try allocator.alloc(mlx.mlx_array, 1 + m);
            const slice_strides = [_]c_int{ 1, 1, 1 };
            for (slots, 0..) |*slot, idx| {
                slot.* = mlx.mlx_array_new();
                const start = [_]c_int{ 0, @intCast(idx), 0 };
                const stop = [_]c_int{ vl_shape[0], @as(c_int, @intCast(idx)) + 1, vl_shape[2] };
                try mlx.check(mlx.mlx_slice(slot, verify_logits, &start, 3, &stop, 3, &slice_strides, 3, s));
            }
            per_pos_logits = slots;
        }

        // Build the greedy argmax tensor lazily; it'll be eval'd alongside
        // the rest of the round below.
        var verify_argmax = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(verify_argmax);
        if (!stochastic) {
            try mlx.check(mlx.mlx_argmax_axis(&verify_argmax, verify_logits, 2, false, s));
        }
        _ = mlx.mlx_array_free(verify_logits);

        // ── Phase 4b: batched eval — drafts + verify_argmax + new_hidden ──
        //
        // Submit the entire round (drafter chain + verify forward + argmax)
        // to the GPU in a single async dispatch. Then sync ONCE per array we
        // need on the CPU. For block_size=8, this collapses ~14 individual
        // sync points (7 drafter samples + 7 per-position argmaxes in the
        // old code) into approximately 2: one effective sync to wait for
        // GPU completion (the first `mlx_array_eval`), and zero-cost evals
        // afterward since the work is already done.
        //
        // CORRECTNESS: `mlx_array_data_int32` only returns valid data once
        // the array is eval'd. We explicitly eval each array we will read.
        // `verify_input` is NOT eval'd separately because MLX may fuse it
        // into the forward pass without materializing a CPU-readable buffer
        // — instead we read drafts via per-array `mlx_array_item_int32` on
        // each `draft_arrs[i]` (cheap after the first sync).
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            for (draft_arrs) |arr| _ = mlx.mlx_vector_array_append_value(eval_vec, arr);
            if (!stochastic) {
                _ = mlx.mlx_vector_array_append_value(eval_vec, verify_argmax);
            }
            _ = mlx.mlx_vector_array_append_value(eval_vec, new_hidden);
            try mlx.check(mlx.mlx_async_eval(eval_vec));
        }
        // Extract drafts. First eval sync waits for the GPU; subsequent
        // evals are no-ops since they were queued together.
        for (draft_arrs, 0..) |arr, idx| {
            try mlx.check(mlx.mlx_array_eval(arr));
            var v: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&v, arr));
            drafts[idx] = @intCast(v);
        }
        if (!stochastic) {
            // Force verify_argmax to materialize before bulk-reading. It's a
            // separate branch from the drafter chain (drafts → concat →
            // verify → argmax), so eval'ing the drafts above doesn't pull
            // verify_argmax along with them. This was the v26.5.6 bug that
            // produced 0% acceptance on 26B/31B (verify ran longer than the
            // drafter chain, so the data buffer was read while the GPU was
            // still writing it).
            try mlx.check(mlx.mlx_array_eval(verify_argmax));
        }

        var accepted: u32 = 0;
        if (stochastic) {
            // Stochastic verify (Leviathan et al. probability-ratio test).
            // The drafted token came from argmax of the drafter's masked LM
            // head, so we treat it as a one-hot proposal: accept with
            // probability `min(1, target_p[draft[i]])`, otherwise stop and
            // sample from the residual at the rejected position.
            var k: u32 = 0;
            while (k < m) : (k += 1) {
                const target_p = try probsAtLastPos(per_pos_logits.?[k], self.sampling, s);
                defer _ = mlx.mlx_array_free(target_p);
                const p_draft = try probAt(target_p, drafts[k], s);
                const accept_prob: f32 = @min(1.0, p_draft);
                const u: f32 = self.prng.random().float(f32);
                if (u >= accept_prob) break;
                accepted += 1;
            }
        } else {
            // Bulk-read the [1, 1+m] argmax indices and scan for first
            // mismatch in CPU. No more GPU syncs in this branch.
            const argmax_data = mlx.mlx_array_data_int32(verify_argmax) orelse {
                return error.MlxArrayDataNull;
            };
            var k: u32 = 0;
            while (k < m) : (k += 1) {
                const target_argmax: u32 = @intCast(argmax_data[k]);
                if (target_argmax != drafts[k]) break;
                accepted += 1;
            }
        }

        // Sample the next pending token from the verify output at position
        // `accepted`:
        //   - full accept (accepted == m): position m predicts the bonus
        //     token one past the last draft.
        //   - partial accept: position `accepted` predicts the model's
        //     replacement for the rejected draft.
        // For greedy, position `accepted`'s argmax is already in
        // `argmax_data[accepted]` — no extra GPU work. For stochastic, we
        // need the actual probability distribution at that position, so we
        // sample from `per_pos_logits[accepted]` (with residual correction
        // on partial accept per Leviathan et al).
        const next_pending: u32 = blk: {
            if (stochastic) {
                const correction_logits = per_pos_logits.?[accepted];
                const probs = try probsAtLastPos(correction_logits, self.sampling, s);
                defer _ = mlx.mlx_array_free(probs);
                if (accepted < m) {
                    const onehot = try pldOneHotRow(drafts[accepted], vl_shape[2], s);
                    defer _ = mlx.mlx_array_free(onehot);
                    break :blk try sampleResidual(probs, onehot, s);
                } else {
                    break :blk try sampleFromProbs(probs, s);
                }
            } else {
                // Greedy: reuse the bulk-read argmax row. Already eval'd in
                // the single async eval above; no GPU sync here.
                const argmax_data = mlx.mlx_array_data_int32(verify_argmax) orelse {
                    return error.MlxArrayDataNull;
                };
                break :blk @intCast(argmax_data[accepted]);
            }
        };

        // ── Phase 5: commit / rollback ──
        if (accepted == m) {
            // Full accept: cache at +1+m. Emit [t1, ...drafts]. Pending = next_pending.
            // The captured `new_hidden` is the post-final-norm hidden at
            // position m — the last accepted draft's position. That's the
            // h_prev for the NEXT round (drafting from t = next_pending; the
            // hidden corresponds to draft[m-1], which is what next_pending
            // follows). This matches the convention `nextDrafter` uses.
            const tokens = try allocator.alloc(u32, 1 + m);
            tokens[0] = t1;
            for (drafts, 0..) |d, idx| tokens[1 + idx] = d;

            try self.generated_ids.append(allocator, t1);
            for (drafts) |d| try self.generated_ids.append(allocator, d);

            if (self.has_last_hidden) _ = mlx.mlx_array_free(self.last_hidden);
            self.last_hidden = new_hidden;
            self.has_last_hidden = true;

            self.drafter_accepted_tokens += m;
            self.next_token_id = next_pending;
            self.step += 1 + m;
            self.completion_tokens += 1 + m;

            // drafts buffer transferred into tokens copy; free original.
            allocator.free(drafts);
            self.checkDrafterRuntimeGate();
            return DrafterStepResult{
                .tokens = tokens,
                .accepted_tokens = m,
            };
        }

        // Partial accept (accepted < m). Cache over-advanced by (m - accepted).
        // The captured new_hidden is for position m (which we're rolling back
        // past) — discard it. Roll back KV+SSM, then re-forward
        // [t1, drafts[0..accepted]] length 1+accepted with hidden capture so
        // last_hidden lands at the position immediately past the last
        // accepted draft (where next_pending will live).
        _ = mlx.mlx_array_free(new_hidden);

        try self.ctx.cache.restore(&kv_snap);
        if (ssm_snaps) |snaps| {
            for (self.ctx.ssm_entries.?, snaps) |*entry, *sn| try ssmRestore(entry, sn);
        }
        self.ctx.moe_seq_offset.* = moe_seq_offset_snap;

        const re_seq_len: c_int = @intCast(1 + accepted);
        const re_input_buf = try allocator.alloc(i32, 1 + accepted);
        defer allocator.free(re_input_buf);
        re_input_buf[0] = @intCast(t1);
        for (drafts[0..accepted], 0..) |d, idx| re_input_buf[1 + idx] = @intCast(d);
        const re_shape = [_]c_int{ 1, re_seq_len };
        const re_input = mlx.mlx_array_new_data(re_input_buf.ptr, &re_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(re_input);

        var re_new_hidden = mlx.mlx_array_new();
        const re_logits = try xfm.forwardWithCapture(&self.ctx, re_input, &re_new_hidden);
        _ = mlx.mlx_array_free(re_logits);

        const tokens = try allocator.alloc(u32, 1 + accepted);
        tokens[0] = t1;
        for (drafts[0..accepted], 0..) |d, idx| tokens[1 + idx] = d;

        try self.generated_ids.append(allocator, t1);
        for (drafts[0..accepted]) |d| try self.generated_ids.append(allocator, d);

        if (self.has_last_hidden) _ = mlx.mlx_array_free(self.last_hidden);
        self.last_hidden = re_new_hidden;
        self.has_last_hidden = true;

        self.drafter_accepted_tokens += accepted;
        self.next_token_id = next_pending;
        self.step += 1 + accepted;
        self.completion_tokens += 1 + accepted;

        allocator.free(drafts);
        self.checkDrafterRuntimeGate();
        return DrafterStepResult{
            .tokens = tokens,
            .accepted_tokens = accepted,
        };
    }

    /// Runtime acceptance gate for the drafter: after warmup, if the per-draft
    /// acceptance probability is below `RUNTIME_GATE_MIN_PER_DRAFT_RATE`,
    /// disable speculation for the rest of this request. Sticky for the rest
    /// of the generation.
    fn checkDrafterRuntimeGate(self: *Generator) void {
        if (self.spec_disabled_runtime) return;
        const drafts_per_round: u32 = if (self.drafter_block_size >= 1) self.drafter_block_size - 1 else 0;
        if (!runtimeGateShouldDisable(self.drafter_attempted, self.drafter_accepted_tokens, drafts_per_round)) return;
        const drafts_proposed: u64 = self.drafter_attempted * @as(u64, drafts_per_round);
        const rate: f32 = if (drafts_proposed > 0)
            @as(f32, @floatFromInt(self.drafter_accepted_tokens)) /
                @as(f32, @floatFromInt(drafts_proposed))
        else
            0.0;
        log.info(
            "  drafter=disabled (runtime per-draft rate {d:.2} < {d:.2} after {d} attempts)\n",
            .{ rate, RUNTIME_GATE_MIN_PER_DRAFT_RATE, self.drafter_attempted },
        );
        self.spec_disabled_runtime = true;
    }

    /// Qwen native-MTP speculative round. Structure mirrors `nextDrafter`
    /// (same verify invariant: cache.step = prompt_len + emitted, t1 NOT in
    /// cache on entry, verify input `[t1, draft[0..m-1]]`, bonus from row m,
    /// partial-accept snapshot/restore + re-forward) with one addition: the
    /// MTP head's committed-history KV cache. Draft steps append m temporary
    /// entries built from MTP-PREDICTED hiddens; after the verify decision we
    /// restore the round-boundary snapshot and re-append the committed pairs
    /// from TRUE trunk hiddens, so the history never accumulates drift.
    pub const MtpHistStash = struct {
        /// `[n]` int32 committed token ids: `[t1, drafts[0..accepted]]`.
        ids: mlx.mlx_array,
        /// `[1, n, H]` trunk hiddens paired 1:1 with `ids` (a lazy concat of
        /// last_hidden + a verify-capture slice — the handle pins the ~90 KB
        /// parent until consumed, deliberately NOT a deep copy).
        hidden: mlx.mlx_array,
        n: usize,
        /// Head-cache position of ids[0]'s entry (the producing round's
        /// mtp_off0). The consume-time truncate drops the producing round's
        /// stale draft tail past it.
        off0: usize,

        pub fn deinit(self: *MtpHistStash) void {
            _ = mlx.mlx_array_free(self.ids);
            _ = mlx.mlx_array_free(self.hidden);
        }
    };

    /// Round origin of the MTP head cache: with a pending stash the cache
    /// still holds the PREVIOUS round's draft tail (its step is stale), so
    /// the committed length is the stash origin plus the entries the stash
    /// itself will append; without one, the cache is fully committed.
    pub fn mtpRoundOff0(stash: ?MtpHistStash, cache_step: usize) usize {
        if (stash) |st| return st.off0 + st.n;
        return cache_step;
    }

    pub fn nextMtp(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
        if (self.done) return null;
        std.debug.assert(self.mtp != null);
        std.debug.assert(self.mtp_cache != null);
        std.debug.assert(self.has_last_hidden);
        std.debug.assert(self.sampling.constraint == null); // grammar + MTP unsupported
        std.debug.assert(self.logprobs_n == 0); // logprobs + MTP unsupported

        // Runtime acceptance gate: same hand-off contract as the drafter
        // (`next()`'s transition shim seeds pending_logits).
        if (self.spec_disabled_runtime) {
            const tok_opt = try self.next(allocator);
            if (tok_opt == null) return null;
            const tokens = try allocator.alloc(u32, 1);
            tokens[0] = tok_opt.?;
            return DrafterStepResult{
                .tokens = tokens,
                .accepted_tokens = 0,
            };
        }

        const xfm = self.xfm;
        const s = xfm.s;
        const head = self.mtp.?;
        const mc = &self.mtp_cache.?;
        // Cross-request EV seed: inherit the head's last healthy acceptance
        // surface so the controller plans from round 1 instead of re-warming
        // (~10 legacy rounds + a +1/round base climb — a third of a short
        // generation). Demotion stays instant (EMA decay + sticky disable are
        // per-request), so a workload change costs a few rounds, not the win.
        if (self.mtp_ev_rounds == 0 and self.mtp_attempted == 0 and
            mtpAdaptiveEnabled() and mtpEvSeedEnabled())
        {
            if (head.ev_seed_accept) |seed| {
                self.mtp_ev_accept = seed;
                self.mtp_ev_m_lo_prev = @min(@max(head.ev_seed_m_lo, 1), mtp_mod.MAX_DEPTH);
                self.mtp_ev_rounds = MTP_EV_WARMUP_ROUNDS;
            }
        }
        // Round plan: fixed mode (and EV warmup) is today's single chunk at
        // the windowed adaptive depth; post-warmup EV mode plans a base chunk
        // m_lo plus a confidence-gated extension to m_hi (see the EV
        // controller section below). `m` is the tokens actually drafted this
        // round — it grows from m_lo to m_hi iff the gate clears.
        const plan = self.mtpRoundPlan();
        const m_lo: u32 = plan.m_lo;
        const m_max: u32 = plan.m_hi;
        var m: u32 = m_lo;
        const t1: u32 = self.next_token_id;

        const tracing = mtpTraceEnabled();
        var ph: io_util.Stopwatch = undefined;
        if (tracing) ph = io_util.Stopwatch.init(self.timer.io);

        // ── Phase 0: record the MTP history length at the round boundary ──
        // No snapshot: a snapshot refcount-shares the head's KV buffer, which
        // forces every draft append's slice_update to copy-on-write the WHOLE
        // history buffer (~268 MB/append at 64k). Rollback is truncate —
        // offset-only — since draft entries only ever append past mtp_off0.
        // With a pending history stash the cache still holds the PREVIOUS
        // round's draft tail (mc.step is stale until the consume-time
        // truncate), so the committed origin comes from the stash.
        const mtp_off0: usize = mtpRoundOff0(self.mtp_hist_stash, mc.step);

        // ── Phase 1: draft m tokens lazily, no per-step CPU sync ──
        // Each step's sampled token ([1] lazy array) feeds the next step's
        // embedding lookup; the MTP post-norm hidden chains as next h_prev.
        // Two-chunk shape (EV mode only, when m_max > m_lo): chunk A also
        // builds a lazy log-confidence per draft; ONE bounded sync at the
        // chunk boundary reads the ids + confidences, and the round extends
        // into chunk B iff the chain confidence clears the plan's tau. When
        // the plan is single-chunk this block is byte-identical in shape to
        // the fixed-depth path — no confidence graph, no extra sync.
        var drafts = try allocator.alloc(u32, m_max);
        errdefer allocator.free(drafts);
        const draft_arrs = try allocator.alloc(mlx.mlx_array, m_max);
        var n_drafted: u32 = 0;
        defer {
            for (draft_arrs[0..n_drafted]) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(draft_arrs);
        }
        const consider_ext = m_max > m_lo;
        var conf_arrs: ?[]mlx.mlx_array = if (consider_ext) try allocator.alloc(mlx.mlx_array, m_lo) else null;
        var n_conf: u32 = 0;
        defer if (conf_arrs) |slots| {
            for (slots[0..n_conf]) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(slots);
        };

        const t1_i32: i32 = @intCast(t1);
        const t1_shape = [_]c_int{1};
        const t1_arr = mlx.mlx_array_new_data(&t1_i32, &t1_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(t1_arr);

        var h_prev_owner: ?mlx.mlx_array = null;
        defer if (h_prev_owner) |h| {
            _ = mlx.mlx_array_free(h);
        };

        {
            // Draft proposal sampler. Greedy (argmax) drafts make the one-hot
            // proposal model exact AND remove draft-side sampling noise from
            // the acceptance rate; opt in via MLX_SERVE_MTP_DRAFT_GREEDY=1.
            var draft_sampling = self.sampling;
            if (mtpDraftGreedy()) draft_sampling.temperature = 0.0;

            var prev_tok_arr: mlx.mlx_array = t1_arr;
            var i: u32 = 0;
            while (i < m) : (i += 1) {
                const h_prev_arg: mlx.mlx_array = if (h_prev_owner) |h| h else self.last_hidden;
                const step_out = if (i == 0 and self.mtp_hist_stash != null) blk: {
                    // Deferred history append (stashed at the END of the
                    // previous round, Phase 5a) merged into this round's
                    // first draft: ONE (n+1)-row head forward appends the
                    // committed-history entries AND the first draft entry,
                    // replacing the old per-round appendHistory forward.
                    // RoPE offsets and cache-append order are byte-identical
                    // to the appendHistory-then-stepArr sequence (pinned by
                    // the merged-forward equivalence test in mtp.zig).
                    var st = self.mtp_hist_stash.?;
                    self.mtp_hist_stash = null;
                    defer st.deinit();
                    // Drop the previous round's stale draft tail — the old
                    // Phase 5a truncate, moved to consume time (offset-only).
                    try mc.truncate(st.off0, s);
                    var merged_ids = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(merged_ids);
                    var merged_hidden = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(merged_hidden);
                    {
                        const idv = mlx.mlx_vector_array_new();
                        defer _ = mlx.mlx_vector_array_free(idv);
                        _ = mlx.mlx_vector_array_append_value(idv, st.ids);
                        _ = mlx.mlx_vector_array_append_value(idv, prev_tok_arr);
                        try mlx.check(mlx.mlx_concatenate_axis(&merged_ids, idv, 0, s));
                        const hv = mlx.mlx_vector_array_new();
                        defer _ = mlx.mlx_vector_array_free(hv);
                        _ = mlx.mlx_vector_array_append_value(hv, st.hidden);
                        _ = mlx.mlx_vector_array_append_value(hv, h_prev_arg);
                        try mlx.check(mlx.mlx_concatenate_axis(&merged_hidden, hv, 1, s));
                    }
                    break :blk try mtp_mod.forward(head, xfm, mc, merged_ids, merged_hidden, @intCast(st.off0), true);
                } else try mtp_mod.stepArr(head, xfm, mc, prev_tok_arr, h_prev_arg, @intCast(mtp_off0 + i));
                draft_arrs[i] = sampleTokenLazy(step_out.logits, draft_sampling, s);
                n_drafted = i + 1;
                if (conf_arrs != null and i < m_lo) {
                    // Chunk-A confidence: log p_head(draft) — built from the
                    // step's own logits BEFORE they're freed (lazy graphs
                    // hold their inputs internally).
                    conf_arrs.?[i] = try draftConfidenceGraph(step_out.logits, draft_arrs[i], s);
                    n_conf = i + 1;
                }
                _ = mlx.mlx_array_free(step_out.logits);
                if (h_prev_owner) |h_old| {
                    _ = mlx.mlx_array_free(h_old);
                }
                h_prev_owner = step_out.hidden_next;
                prev_tok_arr = draft_arrs[i];

                // ── chunk-A boundary: the one bounded sync of the round ──
                if (consider_ext and i + 1 == m_lo) {
                    if (tracing) {
                        self.mtp_trace.add(.draft, ph.read());
                        ph.reset();
                    }
                    const chain_ln = try readChainConfidence(draft_arrs[0..m_lo], conf_arrs.?[0..m_lo], s);
                    if (chain_ln >= plan.tau_ln) {
                        m = m_max;
                        self.mtp_ext_rounds += 1;
                    }
                    if (tracing) {
                        self.mtp_trace.add(.sync, ph.read());
                        ph.reset();
                    }
                }
            }
        }
        if (tracing) {
            self.mtp_trace.add(if (m > m_lo) .ext else .draft, ph.read());
            ph.reset();
        }

        // ── Phase 2: record rollback anchors (NO snapshot on the GDN path) ──
        // A KVCache.snapshot() refcount-shares the KV buffers, which forces
        // verify's slice_update writes to COPY-on-write every full-attention
        // layer's WHOLE buffer — ~4.3 GB per round at 64k context, the
        // dominant round cost at long context. On a GDN trunk (every real MTP
        // target is qwen3_5-family hybrid) rollback needs only the pre-verify
        // LENGTH (KV truncate is offset-only; the stale tail past it is
        // unreachable) plus the verify pass's per-position SSM capture, so no
        // snapshot is taken at all. A hypothetical pure-attention target
        // (ssm_entries == null) keeps the proven snapshot + re-forward path.
        const kv_step_snap = self.ctx.cache.step;
        const gdn_trunk = self.ctx.ssm_entries != null;
        var kv_snap: ?transformer_mod.KVCacheSnapshot = if (gdn_trunk) null else try self.ctx.cache.snapshot();
        defer if (kv_snap) |*snap| snap.deinit();
        const moe_seq_offset_snap = self.ctx.moe_seq_offset.*;

        // ── Phase 3: verify input [t1, drafts...] as one [1, 1+m] tensor ──
        const reshape_2d = [_]c_int{ 1, 1 };
        var t1_2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t1_2d);
        try mlx.check(mlx.mlx_reshape(&t1_2d, t1_arr, &reshape_2d, 2, s));

        var verify_input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(verify_input);
        {
            const drafts_2d = try allocator.alloc(mlx.mlx_array, m);
            defer {
                for (drafts_2d) |arr| _ = mlx.mlx_array_free(arr);
                allocator.free(drafts_2d);
            }
            for (draft_arrs[0..m], drafts_2d) |dlazy, *out| {
                out.* = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_reshape(out, dlazy, &reshape_2d, 2, s));
            }
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, t1_2d);
            for (drafts_2d) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
            try mlx.check(mlx.mlx_concatenate_axis(&verify_input, vec, 1, s));
        }

        var new_hidden = mlx.mlx_array_new();
        var verify_hidden_all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(verify_hidden_all);
        // Enable per-position SSM capture for the verify pass on a GDN trunk
        // so partial accept can roll back without re-forwarding the accepted
        // prefix (mirrors nextPld — the re-forward re-runs the 48-layer
        // sequential recurrence AND a full trunk weight read, and at depth > 1
        // MOST rounds are partial, so it dominated the round cost).
        self.ctx.capture_ssm_seq = self.ctx.ssm_entries != null;
        // Captures the post-final-norm hidden at the LAST position (next
        // round's h_prev) AND all 1+m positions (history re-append).
        const verify_logits = try xfm.forwardWithCaptureAll(&self.ctx, verify_input, &new_hidden, &verify_hidden_all);
        self.ctx.capture_ssm_seq = false;
        // Always free the transient capture buffers before returning, however
        // we exit this round (full accept, partial accept, or error).
        defer if (self.ctx.ssm_entries) |entries| {
            for (entries) |*entry| transformer_mod.ssmFreeSpecCapture(entry);
        };
        self.mtp_attempted += 1;
        self.mtp_drafted_tokens += m;
        if (tracing) {
            self.mtp_trace.add(.verify, ph.read());
            ph.reset();
        }

        // ── Phase 4: decide longest accepted prefix ──
        // Stochastic path is fully BATCHED: accept probabilities for every
        // draft AND a candidate correction token for every possible reject
        // position are built lazily (draft ids stay lazy arrays — never read
        // on the CPU inside a graph-building loop), then ONE async eval
        // realizes the whole round. The old per-draft probAt()/sampleResidual()
        // calls cost one GPU round-trip sync EACH — 3-5 syncs per round that
        // stalled the pipeline for milliseconds while the GPU sat idle.
        const stochastic = self.sampling.temperature > 0.01;
        const vl_shape = mlx.getShape(verify_logits);

        var per_pos_probs: ?[]mlx.mlx_array = null;
        defer if (per_pos_probs) |slots| {
            for (slots) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(slots);
        };
        var accept_p_vec = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(accept_p_vec);
        var corr_samples: ?[]mlx.mlx_array = null;
        defer if (corr_samples) |slots| {
            for (slots) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(slots);
        };

        if (stochastic) {
            const slice_strides = [_]c_int{ 1, 1, 1 };
            // Filtered + softmaxed target probs for ALL 1+m positions in one
            // batched kernel set, then per-position slice VIEWS (no copies).
            const probs_all = try probsAllPositions(verify_logits, self.sampling, s);
            defer _ = mlx.mlx_array_free(probs_all);
            const slots = try allocator.alloc(mlx.mlx_array, 1 + m);
            per_pos_probs = slots;
            for (slots, 0..) |*slot, idx| {
                slot.* = mlx.mlx_array_new();
                const start = [_]c_int{ 0, @intCast(idx), 0 };
                const stop = [_]c_int{ vl_shape[0], @as(c_int, @intCast(idx)) + 1, vl_shape[2] };
                try mlx.check(mlx.mlx_slice(slot, probs_all, &start, 3, &stop, 3, &slice_strides, 3, s));
            }

            // accept_p_vec[k] = target_p[k][draft_k], gathered with the LAZY
            // draft id array → [m] f32 after one eval.
            {
                const taken = try allocator.alloc(mlx.mlx_array, m);
                defer {
                    for (taken) |arr| _ = mlx.mlx_array_free(arr);
                    allocator.free(taken);
                }
                for (0..m) |k| {
                    taken[k] = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_take_axis(&taken[k], slots[k], draft_arrs[k], -1, s));
                }
                const vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(vec);
                for (taken) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
                var cat = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(cat);
                try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, s));
                try mlx.check(mlx.mlx_astype(&accept_p_vec, cat, .float32, s));
            }

            // Candidate correction for every possible reject position a<m
            // (residual sample) plus the full-accept bonus at a=m. Only the
            // one at the realized `accepted` is read; the rest are a few
            // vocab-length ops of throwaway GPU work — far cheaper than a
            // second synchronous softmax+categorical round-trip.
            var indices = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(indices);
            try mlx.check(mlx.mlx_arange(&indices, 0, @as(f64, @floatFromInt(vl_shape[2])), 1, .int32, s));

            const corrs = try allocator.alloc(mlx.mlx_array, 1 + m);
            corr_samples = corrs;
            for (corrs, 0..) |*slot, a| {
                slot.* = mlx.mlx_array_new();
                if (a < m) {
                    // residual = max(target_p − one_hot(draft_a), 0); the
                    // one-hot is built from the lazy draft id (arange == id).
                    var onehot_b = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(onehot_b);
                    try mlx.check(mlx.mlx_equal(&onehot_b, indices, draft_arrs[a], s));
                    var onehot = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(onehot);
                    try mlx.check(mlx.mlx_astype(&onehot, onehot_b, .float32, s));
                    var diff = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(diff);
                    try mlx.check(mlx.mlx_subtract(&diff, per_pos_probs.?[a], onehot, s));
                    const zero = mlx.mlx_array_new_float(0.0);
                    defer _ = mlx.mlx_array_free(zero);
                    var residual = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(residual);
                    try mlx.check(mlx.mlx_maximum(&residual, diff, zero, s));
                    var log_res = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(log_res);
                    try mlx.check(mlx.mlx_log(&log_res, residual, s));
                    const null_key = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(null_key);
                    try mlx.check(mlx.mlx_random_categorical(slot, log_res, -1, null_key, s));
                } else {
                    var log_p = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(log_p);
                    try mlx.check(mlx.mlx_log(&log_p, per_pos_probs.?[m], s));
                    const null_key = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(null_key);
                    try mlx.check(mlx.mlx_random_categorical(slot, log_p, -1, null_key, s));
                }
            }
        }

        var verify_argmax = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(verify_argmax);
        if (!stochastic) {
            try mlx.check(mlx.mlx_argmax_axis(&verify_argmax, verify_logits, 2, false, s));
        }
        _ = mlx.mlx_array_free(verify_logits);
        if (tracing) {
            self.mtp_trace.add(.corr, ph.read());
            ph.reset();
        }

        // ── Phase 4b: one batched async eval for the whole round ──
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            for (draft_arrs[0..m]) |arr| _ = mlx.mlx_vector_array_append_value(eval_vec, arr);
            if (stochastic) {
                _ = mlx.mlx_vector_array_append_value(eval_vec, accept_p_vec);
                for (corr_samples.?) |arr| _ = mlx.mlx_vector_array_append_value(eval_vec, arr);
            } else {
                _ = mlx.mlx_vector_array_append_value(eval_vec, verify_argmax);
            }
            _ = mlx.mlx_vector_array_append_value(eval_vec, new_hidden);
            _ = mlx.mlx_vector_array_append_value(eval_vec, verify_hidden_all);
            try mlx.check(mlx.mlx_async_eval(eval_vec));
        }
        for (draft_arrs[0..m], 0..) |arr, idx| {
            try mlx.check(mlx.mlx_array_eval(arr));
            var v: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&v, arr));
            drafts[idx] = @intCast(v);
        }
        if (!stochastic) {
            // Separate graph branch from the draft chain — force it before
            // bulk-reading (see the v26.5.6 0%-acceptance note in nextDrafter).
            try mlx.check(mlx.mlx_array_eval(verify_argmax));
        }

        var accepted: u32 = 0;
        if (stochastic) {
            // One-hot proposal (argmax draft): accept with min(1, target_p).
            try mlx.check(mlx.mlx_array_eval(accept_p_vec));
            const p_data = mlx.mlx_array_data_float32(accept_p_vec) orelse {
                return error.MlxArrayDataNull;
            };
            var k: u32 = 0;
            while (k < m) : (k += 1) {
                const accept_prob: f32 = @min(1.0, p_data[k]);
                const u: f32 = self.prng.random().float(f32);
                if (u >= accept_prob) break;
                accepted += 1;
            }
        } else {
            const argmax_data = mlx.mlx_array_data_int32(verify_argmax) orelse {
                return error.MlxArrayDataNull;
            };
            var k: u32 = 0;
            while (k < m) : (k += 1) {
                const target_argmax: u32 = @intCast(argmax_data[k]);
                if (target_argmax != drafts[k]) break;
                accepted += 1;
            }
        }

        const next_pending: u32 = blk: {
            if (stochastic) {
                // Pre-sampled in the round batch; realized already.
                const corr = corr_samples.?[accepted];
                try mlx.check(mlx.mlx_array_eval(corr));
                var v: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&v, corr));
                break :blk @intCast(v);
            } else {
                const argmax_data = mlx.mlx_array_data_int32(verify_argmax) orelse {
                    return error.MlxArrayDataNull;
                };
                break :blk @intCast(argmax_data[accepted]);
            }
        };

        if (tracing) {
            self.mtp_trace.add(.eval, ph.read());
            ph.reset();
        }
        log.debug("  [mtp-round] off0={d} t1={d} m={d}/{d} drafts={any} accepted={d}\n", .{ mtp_off0, t1, m, m_max, drafts[0..m], accepted });

        // ── Phase 5a: stash the committed history for a DEFERRED append ──
        // The old shape paid a second head forward (appendHistory) here every
        // round to rebuild history from true verify hiddens — then the next
        // round's first draft re-entered the head anyway. Instead, stash the
        // (tokens, hiddens) pair — tokens as [t1, drafts[0..accepted]], the
        // hiddens the SAME concat of h_prev + ORIGINAL verify hiddens the old
        // appendHistory received — and fold the append into that first draft
        // step (the i==0 merged branch above). The head cache keeps this
        // round's draft tail past mtp_off0 until the consume-time truncate;
        // nothing reads it in between. Rounds with no successor (EOS/length/
        // runtime disable) never pay for the append; deinit frees the stash.
        {
            std.debug.assert(self.mtp_hist_stash == null);
            const n_commit: usize = accepted;
            const ids_i32 = try allocator.alloc(i32, 1 + n_commit);
            defer allocator.free(ids_i32);
            ids_i32[0] = @intCast(t1);
            for (drafts[0..n_commit], 0..) |d, idx| ids_i32[1 + idx] = @intCast(d);
            const id_shape = [_]c_int{@intCast(1 + n_commit)};
            const stash_ids = mlx.mlx_array_new_data(ids_i32.ptr, &id_shape, 1, .int32);
            errdefer _ = mlx.mlx_array_free(stash_ids);

            var hist_hidden = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(hist_hidden);
            if (n_commit == 0) {
                try mlx.check(mlx.mlx_array_set(&hist_hidden, self.last_hidden));
            } else {
                const vh_shape = mlx.getShape(verify_hidden_all);
                var vh_slice = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(vh_slice);
                const start = [_]c_int{ 0, 0, 0 };
                const stop = [_]c_int{ 1, @intCast(n_commit), vh_shape[2] };
                const strides = [_]c_int{ 1, 1, 1 };
                try mlx.check(mlx.mlx_slice(&vh_slice, verify_hidden_all, &start, 3, &stop, 3, &strides, 3, s));
                const vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(vec);
                _ = mlx.mlx_vector_array_append_value(vec, self.last_hidden);
                _ = mlx.mlx_vector_array_append_value(vec, vh_slice);
                try mlx.check(mlx.mlx_concatenate_axis(&hist_hidden, vec, 1, s));
            }
            self.mtp_hist_stash = .{
                .ids = stash_ids,
                .hidden = hist_hidden,
                .n = 1 + n_commit,
                .off0 = mtp_off0,
            };
        }
        if (tracing) {
            self.mtp_trace.add(.hist, ph.read());
            ph.reset();
        }

        // ── Phase 5b: commit / rollback the trunk ──
        if (accepted == m) {
            const tokens = try allocator.alloc(u32, 1 + m);
            tokens[0] = t1;
            for (drafts[0..m], 0..) |d, idx| tokens[1 + idx] = d;

            try self.generated_ids.append(allocator, t1);
            for (drafts[0..m]) |d| try self.generated_ids.append(allocator, d);

            if (self.has_last_hidden) _ = mlx.mlx_array_free(self.last_hidden);
            self.last_hidden = new_hidden;
            self.has_last_hidden = true;

            self.mtp_accepted_tokens += m;
            self.next_token_id = next_pending;
            self.step += 1 + m;
            self.completion_tokens += 1 + m;

            allocator.free(drafts);
            if (mtpAdaptiveEnabled()) self.updateMtpEvRound(m, m) else self.updateMtpDepth(m, m);
            if (tracing) self.mtp_trace.add(.commit, ph.read());
            self.mtpTraceRoundEnd(m, m, m_lo);
            return DrafterStepResult{
                .tokens = tokens,
                .accepted_tokens = m,
            };
        }

        // Partial accept: roll back the trunk. On a GDN trunk the verify pass
        // captured per-position SSM/conv state, so roll back by truncating the
        // KV cache to the accepted length and slicing the capture — NO
        // re-forward of the accepted prefix (mirrors nextPld's fast path; the
        // re-forward is a full trunk weight read, and at depth > 1 most rounds
        // are partial, so it dominated the round cost). The next round's
        // h_prev is the TRUE verify hidden at the last committed position
        // (input index `accepted`), which forwardWithCaptureAll captured.
        // Non-GDN archs keep the proven restore + re-forward fallback.
        _ = mlx.mlx_array_free(new_hidden);

        const gdn_captured = if (self.ctx.ssm_entries) |entries|
            entries.len > 0 and entries[0].spec_state_seq.ctx != null
        else
            false;

        var re_new_hidden = mlx.mlx_array_new();
        if (gdn_captured) {
            const accepted_len: usize = 1 + @as(usize, accepted);
            // `truncate` overwrites cache.step with its length arg; on this
            // family cache.step is a stale counter the model never reads
            // (positioning is moe_seq_offset), so preserve the pre-verify
            // value — keeps prefix-cache kv_step bookkeeping identical to
            // the restore-based fallback (same rule as nextPld).
            try self.ctx.cache.truncate(moe_seq_offset_snap + accepted_len, s);
            self.ctx.cache.step = kv_step_snap;
            for (self.ctx.ssm_entries.?) |*entry| {
                try transformer_mod.ssmRollbackFromCapture(entry, accepted, s);
            }
            self.ctx.moe_seq_offset.* = moe_seq_offset_snap + accepted_len;

            const vh_shape = mlx.getShape(verify_hidden_all);
            const start = [_]c_int{ 0, @intCast(accepted), 0 };
            const stop = [_]c_int{ 1, @as(c_int, @intCast(accepted)) + 1, vh_shape[2] };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&re_new_hidden, verify_hidden_all, &start, 3, &stop, 3, &strides, 3, s));
        } else if (kv_snap) |*snap| {
            try self.ctx.cache.restore(snap);
            self.ctx.moe_seq_offset.* = moe_seq_offset_snap;

            const re_seq_len: c_int = @intCast(1 + accepted);
            const re_input_buf = try allocator.alloc(i32, 1 + accepted);
            defer allocator.free(re_input_buf);
            re_input_buf[0] = @intCast(t1);
            for (drafts[0..accepted], 0..) |d, idx| re_input_buf[1 + idx] = @intCast(d);
            const re_shape = [_]c_int{ 1, re_seq_len };
            const re_input = mlx.mlx_array_new_data(re_input_buf.ptr, &re_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(re_input);

            const re_logits = try xfm.forwardWithCapture(&self.ctx, re_input, &re_new_hidden);
            _ = mlx.mlx_array_free(re_logits);
        } else {
            // GDN trunk whose verify pass produced no capture — cannot roll
            // back safely. Unreachable on real targets (every qwen3_5-family
            // GDN layer populates the capture when capture_ssm_seq is set);
            // pinned by tests/test_mtp_equivalence.sh.
            _ = mlx.mlx_array_free(re_new_hidden);
            return error.MtpRollbackUnavailable;
        }

        const tokens = try allocator.alloc(u32, 1 + accepted);
        tokens[0] = t1;
        for (drafts[0..accepted], 0..) |d, idx| tokens[1 + idx] = d;

        try self.generated_ids.append(allocator, t1);
        for (drafts[0..accepted]) |d| try self.generated_ids.append(allocator, d);

        if (self.has_last_hidden) _ = mlx.mlx_array_free(self.last_hidden);
        self.last_hidden = re_new_hidden;
        self.has_last_hidden = true;

        self.mtp_accepted_tokens += accepted;
        self.next_token_id = next_pending;
        self.step += 1 + accepted;
        self.completion_tokens += 1 + accepted;

        allocator.free(drafts);
        if (mtpAdaptiveEnabled()) self.updateMtpEvRound(m, accepted) else self.updateMtpDepth(m, accepted);
        if (tracing) self.mtp_trace.add(.commit, ph.read());
        self.mtpTraceRoundEnd(m, accepted, m_lo);
        return DrafterStepResult{
            .tokens = tokens,
            .accepted_tokens = accepted,
        };
    }

    // ── MTP adaptive depth ──
    // Unlike the drafter's binary gate, the MTP head has a useful fallback
    // BETWEEN "full depth" and "off": depth 1. Measured on Qwen3.6-27B
    // (M4 Max, greedy): creative content runs 48% per-draft at depth 2
    // (a regression vs AR) but 73% at depth 1 (1.11× AR); code runs 89% at
    // depth 2 (1.45× AR). A windowed controller demotes/promotes between
    // depths and only disables outright when even depth 1 can't pay for its
    // verify overhead.
    pub const MTP_DEPTH_WINDOW: u32 = 16; // rounds in the moving window
    pub const MTP_DEPTH_SWITCH_WARMUP: u32 = 5; // rounds before re-evaluating after a switch
    // Thresholds assume the capture-based rollback (no re-forward on partial
    // accept): a rejected draft costs ONLY its own MTP-layer + draft-head
    // pass (~2 ms), not a second trunk forward (~30-50 ms). Extra depth pays
    // whenever the marginal accept probability clears draft-cost/trunk-cost
    // ≈ 0.05-0.10, so the demote floor sits far lower than the old
    // re-forward-era 0.60 — hysteresis band keeps switch churn down.
    pub const MTP_DEMOTE_BELOW: f32 = 0.40; // per-draft rate at depth > 1 → step down
    pub const MTP_PROMOTE_ABOVE: f32 = 0.60; // per-draft rate below configured depth → step up
    pub const MTP_DISABLE_BELOW: f32 = 0.50; // per-draft rate at depth 1 → disable (sticky)
    pub const MTP_PROMOTE_COOLDOWN: u32 = 32; // rounds promotion stays blocked after a demotion

    /// Greedy (argmax) MTP draft proposals — DEFAULT ON: the acceptance rule
    /// (`min(1, p_target[draft])` + one-hot residual) models the proposal as
    /// deterministic, so argmax drafts make the math exact where sampled
    /// drafts only approximated it — and measured per-draft acceptance is
    /// equal-or-higher with far less run-to-run variance (drafts no longer
    /// carry sampling noise). MLX_SERVE_MTP_DRAFT_GREEDY=0 reverts.
    var mtp_draft_greedy_cache: ?bool = null;
    fn mtpDraftGreedy() bool {
        if (mtp_draft_greedy_cache) |v| return v;
        var on = true;
        if (std.c.getenv("MLX_SERVE_MTP_DRAFT_GREEDY")) |p| {
            const val = std.mem.span(p);
            if (val.len > 0 and val[0] == '0') on = false;
        }
        mtp_draft_greedy_cache = on;
        return on;
    }

    /// Pure depth-policy step. `rate` is the windowed per-draft acceptance
    /// probability. Returns the new depth; 0 means "disable speculation".
    pub fn mtpNextDepth(current: u32, configured: u32, rate: f32) u32 {
        if (current > 1 and rate < MTP_DEMOTE_BELOW) return current - 1;
        if (current <= 1 and rate < MTP_DISABLE_BELOW) return 0;
        if (current < configured and rate > MTP_PROMOTE_ABOVE) return current + 1;
        return current;
    }

    /// Confidence-gated depth decision. Demoting is cheap (still
    /// speculating) so it reacts on a small sample; DISABLING is sticky and
    /// PROMOTING raises verify cost, so both require more evidence. A
    /// 16-round window at a true 73% per-draft rate essentially never dips
    /// below the 0.50 disable floor; a 5-round window does (observed live:
    /// an early-story cold streak disabled a request that would have run
    /// 1.11x at depth 1).
    pub fn mtpDepthDecision(current: u32, configured: u32, rate: f32, window_rounds: u32, promote_blocked: bool) u32 {
        const next_depth = mtpNextDepth(current, configured, rate);
        if (next_depth == 0 and window_rounds < MTP_DEPTH_WINDOW) return current;
        if (next_depth > current and (window_rounds < 8 or promote_blocked)) return current;
        return next_depth;
    }

    /// Windowed adaptive-depth update, called once per nextMtp round with
    /// that round's (drafted, accepted) counts.
    fn updateMtpDepth(self: *Generator, drafted: u32, accepted: u32) void {
        const idx = self.mtp_window_idx % MTP_DEPTH_WINDOW;
        self.mtp_window_drafted[idx] = @intCast(drafted);
        self.mtp_window_accepted[idx] = @intCast(accepted);
        self.mtp_window_idx += 1;
        if (self.mtp_promote_cooldown > 0) self.mtp_promote_cooldown -= 1;
        if (self.mtp_rounds_since_switch < MTP_DEPTH_SWITCH_WARMUP) {
            self.mtp_rounds_since_switch += 1;
            return;
        }
        const n = @min(self.mtp_window_idx, MTP_DEPTH_WINDOW);
        var drafted_sum: u32 = 0;
        var accepted_sum: u32 = 0;
        var i: u32 = 0;
        while (i < n) : (i += 1) {
            drafted_sum += self.mtp_window_drafted[i];
            accepted_sum += self.mtp_window_accepted[i];
        }
        if (drafted_sum == 0) return;
        const rate = @as(f32, @floatFromInt(accepted_sum)) / @as(f32, @floatFromInt(drafted_sum));
        const next_depth = mtpDepthDecision(self.mtp_depth_current, self.mtp_depth, rate, n, self.mtp_promote_cooldown > 0);
        if (next_depth == self.mtp_depth_current) return;
        if (next_depth == 0) {
            log.info(
                "  mtp=disabled (windowed per-draft rate {d:.2} < {d:.2} at depth 1)\n",
                .{ rate, MTP_DISABLE_BELOW },
            );
            self.spec_disabled_runtime = true;
            return;
        }
        log.debug("  [mtp-depth] {d} -> {d} (windowed per-draft rate {d:.2})\n", .{ self.mtp_depth_current, next_depth, rate });
        if (next_depth < self.mtp_depth_current) self.mtp_promote_cooldown = MTP_PROMOTE_COOLDOWN;
        self.mtp_depth_current = next_depth;
        self.mtp_rounds_since_switch = 0;
        // Reset the window so the new depth is judged on its own rounds.
        self.mtp_window_idx = 0;
    }

    // ── MTP EV (expected-value) adaptive controller ──
    // Fixed-depth drafting is the warm-decode ceiling: at ~77% per-draft the
    // marginal chain decays with index, so one global depth wastes verify
    // width on hard stretches and leaves easy stretches (code boilerplate
    // where 8/8 accept) under-drafted. The EV controller tracks CONDITIONAL
    // per-index acceptance EMAs a[i] = P(draft i accepted | i-1 accepted) and
    // plans each round as two chunks: a base chunk `m_lo` (the static EV
    // optimum), then — when the head's own confidence on chunk A clears a
    // cost-derived threshold tau — an extension to `m_hi`. Only rounds that
    // CONSIDER extension pay the one bounded chunk-A sync; when the plan
    // collapses to m_lo == m_hi the round is byte-identical in shape to the
    // fixed-depth path (no confidence graph, no sync).
    // Disable via MLX_SERVE_MTP_ADAPTIVE=0 (reverts to the windowed
    // fixed-depth controller above).

    /// Default depth cap when `--mtp-depth` is not passed (0 = auto) and the
    /// EV controller is active. 6 keeps the verify forward at seq 1+6 = 7,
    /// the split-K verify-qmm kernel's ceiling on M1-M4. Eligible M5/G17
    /// targets use MTP_ADAPTIVE_NAX_CAP instead: their measured NAX round-cost
    /// surface makes depths 7/8 profitable. Explicit depths always win.
    pub const MTP_ADAPTIVE_DEFAULT_CAP: u32 = 6;
    pub const MTP_ADAPTIVE_NAX_CAP: u32 = 8;
    /// Rounds of legacy (fixed-depth windowed) behavior while the EMAs fill.
    /// Warmup, but converges in ROUNDS, not 43 s of offline calibration.
    pub const MTP_EV_WARMUP_ROUNDS: u32 = 10;
    /// EMA step for the per-index acceptance estimates. 0.15 demotes fast on
    /// cold streaks (5 consecutive rejects: 0.72 -> 0.32) without letting a
    /// single unlucky round move the plan.
    pub const MTP_EV_EMA_BETA: f32 = 0.15;
    /// Optimistic prior for unobserved indices. Deliberately ABOVE the
    /// measured average per-draft rate (~77%): a deep index is only ever
    /// observed when extension fires, and on this cost surface the
    /// break-even conditional acceptance for a ramp position is ~0.78 — a
    /// realistic prior would sit razor-under it and extension would never
    /// get its first trial (measured live: ext_rounds=0 on a pure-echo
    /// workload). The tau gate (only near-perfect-confidence rounds extend)
    /// plus demote-fast EMAs bound the cost of an optimistic trial to a few
    /// rounds.
    pub const MTP_EV_PRIOR: f32 = 0.85;
    /// Clamp band for the extension confidence threshold.
    pub const MTP_EV_TAU_MIN: f32 = 0.05;
    pub const MTP_EV_TAU_MAX: f32 = 0.95;

    /// Round-cost model in units of the fixed round cost (verify-forward
    /// floor + round eval/read + commit ≈ 1.0 ≈ 32 ms on the 27B since the
    /// deferred history append). Ratios are machine-stable where absolute ms
    /// are not. Refit via MLX_SERVE_MTP_TRACE on Qwen3.6-27B GDN (M4 Max,
    /// 2026-07-13, saturated fixed depths, same-session sweep AFTER the
    /// deferred-append round shape landed): T(1)=42.0, T(3)=62.6,
    /// T(6)=111.6, T(7)=114.9 ms — the surface is PIECEWISE: ~10.3 ms
    /// marginal per position in the flat verify region (seq <= 4), ~16.3
    /// ms/pos for positions 4-6, and position 7 nearly free (+3.3 ms —
    /// verify seq 8 rides the same row tile as 5-7), averaged into
    /// per_pos_hi. ATTRIBUTION (the GDN-vs-qmm µbench in transformer.zig,
    /// MLX_SERVE_GDN_UBENCH=1): the ladder is ~90% qmm ROW-COUNT cost — the
    /// GDN recurrence kernel is nearly flat over verify widths (0.13→0.26 ms
    /// per dispatch, T 2→64) and contributes <1 ms/round; the earlier
    /// "GDN sequential width ramp" reading was a mis-attribution.
    /// The old linear ~1.5 ms/pos model came from a depth-6 run whose
    /// windowed controller was silently demoting underneath — never fit
    /// costs from a run whose realized m_avg you didn't check.
    /// Override for live tuning (an explicit override selects the generic
    /// two-region surface even on M5, so all four values remain the
    /// complete backwards-compatible contract):
    /// MLX_SERVE_MTP_EV_COSTS="draft,per_pos_lo,per_pos_hi,sync".
    pub const MtpEvCosts = struct {
        draft: f32, // one sequential MTP-head step (fwd + draft lm_head)
        per_pos_lo: f32, // marginal verify+capture per position, flat region
        per_pos_hi: f32, // ... beyond flat_max (qmm row-tile ramp)
        flat_max: u32, // last draft index in the flat verify region
        sync: f32, // the chunk-A confidence read-back
        /// First draft depth whose verify forward lands on the M5 NAX tile.
        /// Zero disables the third region. Depth k verifies k+1 rows, so the
        /// default NAX M=8 takeover starts at draft position 7.
        nax_from: u32 = 0,
        per_pos_nax: f32 = 0.0,
    };
    /// 2026-07-13 refit #2, AFTER the split-K verify-qmm kernel landed
    /// (transformer.verifyQmm): same-session saturated sweep T(1)=44.6,
    /// T(3)=54.4, T(6)=89.9 ms → floor ≈ 40 ms, flat marginal 4.9 ms/pos
    /// (the kernel holds verify qmms near the weight-stream floor through
    /// M=7), ramp 11.8 ms/pos (stock lm_head at growing M + attention +
    /// eval/read).
    pub const MTP_EV_DEFAULT_COSTS: MtpEvCosts = .{ .draft = 0.06, .per_pos_lo = 0.06, .per_pos_hi = 0.24, .flat_max = 3, .sync = 0.02 };
    /// M5 Max/G17 refit (2026-07-17), same-session saturated fixed-depth
    /// sweep after the NAX m16 verify lane landed: T(1..4) ~= 41.35 ms,
    /// T(6)=62.15 ms, T(8)=68.39 ms. In floor units this identifies
    /// draft+hi ~= .21 and draft+nax ~= .10; T(8)/T(6) is reproduced by
    /// 2.19/1.99 = 1.1005 (measured 1.1004). The profile is selected only
    /// for the calibrated dense Qwen3.6-27B homogeneous affine-4/gs-64
    /// checkpoint with its native affine-8/gs-32 sidecar and successfully
    /// built affine-3/gs-64 draft head, when the trunk lm_head routes both
    /// M=8 and M=9 through NAX; every other combination retains DEFAULT.
    pub const MTP_EV_G17_NAX_COSTS: MtpEvCosts = .{
        .draft = 0.06,
        .per_pos_lo = 0.06,
        .per_pos_hi = 0.15,
        .flat_max = 3,
        .sync = 0.02,
        .nax_from = 7,
        .per_pos_nax = 0.04,
    };
    /// M5 Max/G17 affine-4/gs-32 sidecar refit (2026-07-18). A saturated
    /// fixed-depth sweep gave T(1)=36.04, T(3)=43.01, T(6)=62.56, and
    /// T(8)=66.06 ms. The fitted composite marginals (`draft + verify`) are
    /// .107/.200/.054; depths 4 and 5 independently validate the rounded
    /// .11/.20/.05 surface after matching-baseline correction. The split
    /// between draft and verify is not separately identifiable.
    pub const MTP_EV_G17_NAX_Q4_GS32_COSTS: MtpEvCosts = .{
        .draft = 0.03,
        .per_pos_lo = 0.08,
        .per_pos_hi = 0.17,
        .flat_max = 3,
        .sync = 0.02,
        .nax_from = 7,
        .per_pos_nax = 0.02,
    };

    /// Marginal round cost of draft position k (1-based).
    pub fn mtpEvMarginalCost(costs: MtpEvCosts, k: u32) f32 {
        const verify_cost = if (costs.nax_from != 0 and k >= costs.nax_from)
            costs.per_pos_nax
        else if (k <= costs.flat_max)
            costs.per_pos_lo
        else
            costs.per_pos_hi;
        return costs.draft + verify_cost;
    }

    /// One round's draft plan. `m_hi > m_lo` means "pay the chunk-A sync and
    /// extend to m_hi when the chain log-confidence clears tau_ln".
    pub const MtpRoundPlan = struct {
        m_lo: u32,
        m_hi: u32,
        tau_ln: f32,
    };

    /// Resolve the configured depth cap. 0 = auto (`--mtp-depth` not passed):
    /// MTP_ADAPTIVE_NAX_CAP only when the EV controller and a calibrated G17
    /// cost profile are both active, MTP_ADAPTIVE_DEFAULT_CAP otherwise, and
    /// DEFAULT_DEPTH in fixed mode. Explicit values always win.
    pub fn mtpDepthCapForProfile(configured: u32, adaptive: bool, profile: mtp_mod.MtpCostProfile) u32 {
        if (configured != 0) return @min(mtp_mod.MAX_DEPTH, @max(1, configured));
        if (!adaptive) return mtp_mod.DEFAULT_DEPTH;
        return switch (profile) {
            .generic => MTP_ADAPTIVE_DEFAULT_CAP,
            .g17_nax_q8_gs32, .g17_nax_q4_gs32 => MTP_ADAPTIVE_NAX_CAP,
        };
    }

    pub fn resolveMtpDepthCapForProfile(configured: u32, profile: mtp_mod.MtpCostProfile) u32 {
        return mtpDepthCapForProfile(configured, mtpAdaptiveEnabled(), profile);
    }

    /// Legacy q8 boolean selector retained for source compatibility.
    pub fn mtpDepthCapFor(configured: u32, adaptive: bool, nax_profile: bool) u32 {
        const profile: mtp_mod.MtpCostProfile = if (nax_profile) .g17_nax_q8_gs32 else .generic;
        return mtpDepthCapForProfile(configured, adaptive, profile);
    }

    /// Legacy q8 boolean selector retained for source compatibility.
    pub fn resolveMtpDepthCap(configured: u32, nax_profile: bool) u32 {
        const profile: mtp_mod.MtpCostProfile = if (nax_profile) .g17_nax_q8_gs32 else .generic;
        return resolveMtpDepthCapForProfile(configured, profile);
    }

    /// Expected committed tokens for an m-deep round: the always-committed t1
    /// plus the acceptance chain sum (draft k lands iff drafts 0..k all land).
    pub fn mtpEvExpectedTokens(a: []const f32, m: u32) f32 {
        var chain: f32 = 1.0;
        var tok: f32 = 1.0;
        var k: u32 = 0;
        while (k < m and k < a.len) : (k += 1) {
            chain *= a[k];
            tok += chain;
        }
        return tok;
    }

    /// Round cost in verify-base units (piecewise per-position marginals).
    pub fn mtpEvRoundCost(costs: MtpEvCosts, m: u32, with_sync: bool) f32 {
        var c: f32 = 1.0 + (if (with_sync) costs.sync else 0.0);
        var k: u32 = 1;
        while (k <= m) : (k += 1) c += mtpEvMarginalCost(costs, k);
        return c;
    }

    /// Pure EV plan: pick (m_lo, m_hi, tau) maximizing expected tok/round-cost.
    /// `m_lo_max` damps the base-depth climb (hysteresis — the caller passes
    /// last round's m_lo + 1); demotions are never damped.
    ///  1. m_lo = argmax over single-chunk depths of E(m)/T(m).
    ///  2. m_hi = deepest position whose marginal chain still pays under FULL
    ///     confidence in chunk A (the best case the gate can certify).
    ///  3. tau: extend when the confidence-implied chain beats the stop rate
    ///     on the margin — c*S/dt > r  =>  tau = r*dt/S.
    /// There is deliberately NO separate "is the sync worth it" gate: tau
    /// already keeps low-confidence rounds single-chunk, the horizon check
    /// collapses m_hi on cold EMAs (killing the sync entirely), and a
    /// prior-weighted expected-gain gate measurably starves exploration —
    /// deep indices are only observed when extension fires, so a gate fed by
    /// their priors blocks the first trial forever (live: ext_rounds=0 on
    /// pure echo).
    pub fn mtpEvPlanFor(a: []const f32, cap_in: u32, costs: MtpEvCosts, m_lo_max: u32) MtpRoundPlan {
        const cap: u32 = @intCast(@min(@as(usize, @max(1, cap_in)), a.len));
        const lo_cap: u32 = @min(cap, @max(1, m_lo_max));
        var m_lo: u32 = 1;
        var best_r: f32 = 0.0;
        var m: u32 = 1;
        while (m <= lo_cap) : (m += 1) {
            const r = mtpEvExpectedTokens(a, m) / mtpEvRoundCost(costs, m, false);
            if (r > best_r) {
                best_r = r;
                m_lo = m;
            }
        }
        if (m_lo >= cap) return .{ .m_lo = m_lo, .m_hi = m_lo, .tau_ln = 0.0 };
        var m_hi: u32 = m_lo;
        var cond: f32 = 1.0;
        var s_sum: f32 = 0.0; // expected extension tokens, conditional on chunk A
        var t_sum: f32 = 0.0; // extension marginal cost (piecewise)
        while (m_hi < cap) {
            cond *= a[m_hi];
            const mc = mtpEvMarginalCost(costs, m_hi + 1);
            if (cond <= best_r * mc) break;
            s_sum += cond;
            t_sum += mc;
            m_hi += 1;
        }
        if (m_hi == m_lo) return .{ .m_lo = m_lo, .m_hi = m_lo, .tau_ln = 0.0 };
        // The TAU_MAX clamp doubles as the exploration valve: on razor-thin
        // horizons the honest tau approaches 1 ("never extend"), and 0.95
        // lets near-perfect-confidence rounds through so the deep EMAs can
        // observe reality at all.
        const tau = std.math.clamp(best_r * t_sum / s_sum, MTP_EV_TAU_MIN, MTP_EV_TAU_MAX);
        return .{ .m_lo = m_lo, .m_hi = m_hi, .tau_ln = @log(tau) };
    }

    /// Update the conditional acceptance EMAs from one realized round.
    /// Acceptance is prefix-structured: indices < accepted saw a success, the
    /// index AT `accepted` saw the reject (when one happened), and deeper
    /// indices were never conditionally reached — no observation.
    pub fn mtpEvObserve(a: []f32, drafted: u32, accepted: u32, beta: f32) void {
        var i: usize = 0;
        while (i < accepted and i < a.len) : (i += 1) a[i] += beta * (1.0 - a[i]);
        if (accepted < drafted and accepted < a.len) a[accepted] += beta * (0.0 - a[accepted]);
    }

    /// Chain log-confidence of a drafted chunk: sum of per-draft log p_head,
    /// clamped to <= 0 per term (a log-prob is never positive; bf16 noise
    /// can be). NaN poisons to -inf so a broken confidence can never extend.
    pub fn mtpChainLogConf(confs: []const f32) f32 {
        var sum: f32 = 0.0;
        for (confs) |c| {
            if (std.math.isNan(c)) return -std.math.inf(f32);
            sum += @min(0.0, c);
        }
        return sum;
    }

    /// Per-phase wall-time accumulator behind MLX_SERVE_MTP_TRACE=1. Pure
    /// bookkeeping; `nextMtp` stamps phases with a Stopwatch and emits one
    /// summary line every LOG_EVERY rounds. Zero cost when the env is absent
    /// (every stamp is guarded on the cached env check).
    pub const MtpTrace = struct {
        pub const LOG_EVERY: u32 = 32;
        pub const Phase = enum(u4) { draft, sync, ext, verify, corr, eval, hist, commit };
        pub const N_PHASES = @typeInfo(Phase).@"enum".fields.len;

        rounds: u32 = 0,
        ns: [N_PHASES]u64 = [_]u64{0} ** N_PHASES,
        drafted: u64 = 0,
        accepted: u64 = 0,
        extended: u32 = 0,

        pub fn add(self: *MtpTrace, phase: Phase, dur_ns: u64) void {
            self.ns[@intFromEnum(phase)] += dur_ns;
        }

        /// Close one round; true when a summary line is due (caller logs,
        /// then calls reset()).
        pub fn endRound(self: *MtpTrace, drafted_n: u32, accepted_n: u32, was_extended: bool) bool {
            self.rounds += 1;
            self.drafted += drafted_n;
            self.accepted += accepted_n;
            if (was_extended) self.extended += 1;
            return self.rounds >= LOG_EVERY;
        }

        pub fn avgMs(self: *const MtpTrace, phase: Phase) f64 {
            if (self.rounds == 0) return 0.0;
            return @as(f64, @floatFromInt(self.ns[@intFromEnum(phase)])) /
                (@as(f64, @floatFromInt(self.rounds)) * 1e6);
        }

        pub fn totalAvgMs(self: *const MtpTrace) f64 {
            if (self.rounds == 0) return 0.0;
            var total: u64 = 0;
            for (self.ns) |v| total += v;
            return @as(f64, @floatFromInt(total)) / (@as(f64, @floatFromInt(self.rounds)) * 1e6);
        }

        pub fn reset(self: *MtpTrace) void {
            self.* = .{};
        }
    };

    /// Adaptive (EV) controller gate — DEFAULT ON. MLX_SERVE_MTP_ADAPTIVE=0
    /// reverts to the fixed-depth windowed controller for same-boot A/Bs.
    var mtp_adaptive_cache: ?bool = null;
    pub fn mtpAdaptiveEnabled() bool {
        if (mtp_adaptive_cache) |v| return v;
        var on = true;
        if (std.c.getenv("MLX_SERVE_MTP_ADAPTIVE")) |p| {
            const val = std.mem.span(p);
            if (val.len > 0 and val[0] == '0') on = false;
        }
        mtp_adaptive_cache = on;
        return on;
    }

    /// Cross-request EV seeding gate — default ON; set
    /// MLX_SERVE_MTP_EV_SEED=0 to keep request planning independent.
    var mtp_ev_seed_cache: ?bool = null;
    fn mtpEvSeedEnabledFromEnv(raw: ?[]const u8) bool {
        const value = raw orelse return true;
        return value.len == 0 or value[0] != '0';
    }

    fn mtpEvSeedEnabled() bool {
        if (mtp_ev_seed_cache) |v| return v;
        const raw: ?[]const u8 = if (std.c.getenv("MLX_SERVE_MTP_EV_SEED")) |p| std.mem.span(p) else null;
        const on = mtpEvSeedEnabledFromEnv(raw);
        mtp_ev_seed_cache = on;
        return on;
    }

    var mtp_trace_cache: ?bool = null;
    fn mtpTraceEnabled() bool {
        if (mtp_trace_cache) |v| return v;
        const on = readEnvBool("MLX_SERVE_MTP_TRACE");
        mtp_trace_cache = on;
        return on;
    }

    fn parseMtpEvCostsOverride(raw: []const u8) ?MtpEvCosts {
        var values: [4]f32 = undefined;
        var it = std.mem.splitScalar(u8, raw, ',');
        var i: usize = 0;
        while (i < values.len) : (i += 1) {
            const part = it.next() orelse return null;
            const value = std.fmt.parseFloat(f32, std.mem.trim(u8, part, " ")) catch return null;
            if (!std.math.isFinite(value)) return null;
            values[i] = value;
        }
        if (it.next() != null) return null;
        if (values[0] <= 0.0 or values[1] <= 0.0 or values[2] <= 0.0 or values[3] < 0.0) return null;

        var c = MTP_EV_DEFAULT_COSTS;
        c.draft = values[0];
        c.per_pos_lo = values[1];
        c.per_pos_hi = values[2];
        c.sync = values[3];
        return c;
    }

    /// Pure profile/override selector. A valid explicit four-value override
    /// starts from DEFAULT (rather than silently inheriting the hardware
    /// profile), so a value copied from an M1-M4 tuning run means the same
    /// thing on M5. Empty/partial/malformed values are ignored atomically;
    /// they must not silently leave an auto-cap-8 target on generic costs.
    pub fn mtpEvCostsForProfile(profile: mtp_mod.MtpCostProfile, override: ?[]const u8) MtpEvCosts {
        const selected = switch (profile) {
            .generic => MTP_EV_DEFAULT_COSTS,
            .g17_nax_q8_gs32 => MTP_EV_G17_NAX_COSTS,
            .g17_nax_q4_gs32 => MTP_EV_G17_NAX_Q4_GS32_COSTS,
        };
        if (override) |raw| {
            return parseMtpEvCostsOverride(raw) orelse selected;
        }
        return selected;
    }

    /// Legacy q8 boolean selector retained for source compatibility.
    pub fn mtpEvCostsFor(nax_profile: bool, override: ?[]const u8) MtpEvCosts {
        const profile: mtp_mod.MtpCostProfile = if (nax_profile) .g17_nax_q8_gs32 else .generic;
        return mtpEvCostsForProfile(profile, override);
    }

    fn mtpEvCosts(profile: mtp_mod.MtpCostProfile) MtpEvCosts {
        return mtpEvCostsForProfile(
            profile,
            if (std.c.getenv("MLX_SERVE_MTP_EV_COSTS")) |p| std.mem.span(p) else null,
        );
    }

    /// Per-round draft plan. Fixed mode (and EV warmup): today's adaptive
    /// depth, no extension — byte-identical round shape to the legacy path.
    /// Post-warmup EV mode: the pure plan over the acceptance EMAs, with the
    /// base-depth climb damped to one step per round.
    fn mtpRoundPlan(self: *Generator) MtpRoundPlan {
        const cap: u32 = @min(@max(@as(u32, 1), self.mtp_depth), mtp_mod.MAX_DEPTH);
        if (!mtpAdaptiveEnabled() or self.mtp_ev_rounds < MTP_EV_WARMUP_ROUNDS) {
            const d = @min(@max(@as(u32, 1), self.mtp_depth_current), cap);
            self.mtp_ev_m_lo_prev = d;
            return .{ .m_lo = d, .m_hi = d, .tau_ln = 0.0 };
        }
        const plan = mtpEvPlanFor(self.mtp_ev_accept[0..cap], cap, self.mtp_ev_costs, self.mtp_ev_m_lo_prev + 1);
        self.mtp_ev_m_lo_prev = plan.m_lo;
        return plan;
    }

    /// Track the only evidence that can justify sticky-disable: whether the
    /// first draft landed while the EV base depth was exactly one. Wider base
    /// rounds reset the probation window, and later extension misses do not
    /// count against the depth-one floor. Returns a rate only after a full
    /// fresh window has been observed.
    fn mtpFloorDisableObserve(
        drafted_window: *[MTP_DEPTH_WINDOW]u8,
        accepted_window: *[MTP_DEPTH_WINDOW]u8,
        window_idx: *u32,
        m_lo: u32,
        drafted: u32,
        accepted: u32,
    ) ?f32 {
        std.debug.assert(drafted >= 1);
        if (m_lo != 1) {
            window_idx.* = 0;
            return null;
        }

        const idx = window_idx.* % MTP_DEPTH_WINDOW;
        drafted_window[idx] = 1;
        accepted_window[idx] = @intFromBool(accepted > 0);
        window_idx.* += 1;
        const n = @min(window_idx.*, MTP_DEPTH_WINDOW);
        if (n < MTP_DEPTH_WINDOW) return null;

        var accepted_sum: u32 = 0;
        var i: u32 = 0;
        while (i < n) : (i += 1) accepted_sum += accepted_window[i];
        return @as(f32, @floatFromInt(accepted_sum)) / @as(f32, @floatFromInt(n));
    }

    /// EV-mode per-round update: EMAs always; during warmup the legacy
    /// windowed controller keeps running (today's behavior while EMAs fill);
    /// post-warmup only the sticky disable floor is checked — EV owns depth.
    fn updateMtpEvRound(self: *Generator, drafted: u32, accepted: u32) void {
        mtpEvObserve(&self.mtp_ev_accept, drafted, accepted, MTP_EV_EMA_BETA);
        self.mtp_ev_rounds += 1;
        if (self.mtp_ev_rounds <= MTP_EV_WARMUP_ROUNDS) {
            self.updateMtpDepth(drafted, accepted);
            // Warmup may evaluate several depths. None of that mixed evidence
            // belongs in the post-warmup depth-one sticky-disable window.
            if (self.mtp_ev_rounds == MTP_EV_WARMUP_ROUNDS) self.mtp_window_idx = 0;
            return;
        }
        // EV owns promotion/demotion. Sticky-disable is judged only from a
        // full, homogeneous window of first-draft outcomes at base depth one.
        const rate = mtpFloorDisableObserve(
            &self.mtp_window_drafted,
            &self.mtp_window_accepted,
            &self.mtp_window_idx,
            self.mtp_ev_m_lo_prev,
            drafted,
            accepted,
        ) orelse return;
        if (rate < MTP_DISABLE_BELOW) {
            log.info(
                "  mtp=disabled (EV: depth-1 first-draft rate {d:.2} < {d:.2})\n",
                .{ rate, MTP_DISABLE_BELOW },
            );
            self.spec_disabled_runtime = true;
        }
    }

    /// Close one traced round; emits + resets the summary at the cadence.
    fn mtpTraceRoundEnd(self: *Generator, m: u32, accepted: u32, m_lo: u32) void {
        if (!mtpTraceEnabled()) return;
        if (!self.mtp_trace.endRound(m, accepted, m > m_lo)) return;
        const t = &self.mtp_trace;
        log.info(
            "  [mtp-trace] rounds={d} avg_ms draft={d:.2} sync={d:.2} ext={d:.2} verify={d:.2} corr={d:.2} eval={d:.2} hist={d:.2} commit={d:.2} total={d:.2} | m_avg={d:.2} acc_avg={d:.2} ext_rate={d:.2}\n",
            .{
                t.rounds,
                t.avgMs(.draft),
                t.avgMs(.sync),
                t.avgMs(.ext),
                t.avgMs(.verify),
                t.avgMs(.corr),
                t.avgMs(.eval),
                t.avgMs(.hist),
                t.avgMs(.commit),
                t.totalAvgMs(),
                @as(f64, @floatFromInt(t.drafted)) / @as(f64, @floatFromInt(t.rounds)),
                @as(f64, @floatFromInt(t.accepted)) / @as(f64, @floatFromInt(t.rounds)),
                @as(f64, @floatFromInt(t.extended)) / @as(f64, @floatFromInt(t.rounds)),
            },
        );
        t.reset();
    }

    /// Returns the next token ID, or null when generation is finished.
    ///
    /// Pipeline architecture (matches mlx-lm's generator pattern):
    ///
    ///   The KEY to effective pipelining is the ORDER of operations:
    ///   1. Build next step's lazy graph (depends on pending lazy token)
    ///   2. async_eval the next graph — GPU computes pending token as a DEPENDENCY,
    ///      then continues with the forward pass
    ///   3. eval(pending_token) — returns INSTANTLY since GPU already computed it
    ///   4. Return the token while GPU continues computing the next forward pass
    ///
    ///   This mirrors mlx-lm's: _step(y) → async_eval(next_y) → yield y.item()
    ///   where y.item() is instant because async_eval forced y's computation.
    pub fn next(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        if (self.done) return null;
        if (self.sampling.constraint != null) return self.nextConstrained(allocator);

        // Transition shim: speculative-decode paths may exit with
        // `next_token_id` set but `pending_logits` unset (drafter's exit
        // invariant is "t1 NOT in cache" — its hand-off to `next()` would
        // otherwise crash on the slow path which assumes pending_logits is
        // always lazily seeded). When we observe that state, synchronously
        // forward `[next_token_id]` to seed `pending_logits` so the fast
        // path picks up cleanly. PLD's exit state already matches `next()`'s
        // invariant, so this only fires for drafter→next runtime-gate
        // fallbacks (and any future spec methods that share drafter's shape).
        if (!self.has_pending_logits and !self.has_pending_token and
            self.step < self.max_tokens and self.logprobs_n == 0)
        {
            const tok_i32: i32 = @intCast(self.next_token_id);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            self.pending_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
            self.has_pending_logits = true;
        }

        // ── Phase 1: Build and submit the NEXT step FIRST ──
        // This forces the GPU to compute the pending token as a dependency,
        // so when we eval it in Phase 2, it's already ready.
        if (self.has_pending_logits and self.logprobs_n == 0 and self.step + 1 < self.max_tokens) {
            const step_logits = self.pending_logits;
            self.has_pending_logits = false;

            const lazy_token = sampleTokenLazy(step_logits, self.sampling, self.xfm.s);
            _ = mlx.mlx_array_free(step_logits);

            if (lazyForward(self.xfm, &self.ctx, lazy_token)) |next_logits| {
                const arr = [_]mlx.mlx_array{ lazy_token, next_logits };
                const vec = mlx.mlx_vector_array_new_data(&arr, 2);
                _ = mlx.mlx_async_eval(vec);
                _ = mlx.mlx_vector_array_free(vec);

                // NOW resolve the pending token — GPU already computed it as a
                // dependency of the graph we just submitted. Should be instant.
                try self.resolvePendingToken();

                // Check stop conditions on the resolved token
                if (try self.checkStop()) return null;

                const token = self.next_token_id;
                self.completion_tokens += 1;
                self.step += 1;
                try self.generated_ids.append(allocator, token);

                if (self.step % 256 == 0) _ = mlx.mlx_clear_cache();

                // Store new pending state
                self.pending_token = lazy_token;
                self.has_pending_token = true;
                self.pending_logits = next_logits;
                self.has_pending_logits = true;

                return token;
            } else |_| {
                // lazyForward failed — fall through to slow path
                try mlx.check(mlx.mlx_array_eval(lazy_token));
                var val: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
                _ = mlx.mlx_array_free(lazy_token);
                self.next_token_id = @intCast(val);
                self.has_pending_token = false;
            }
        }

        // ── Phase 2: Slow path (first token, last token, logprobs, or pipeline miss) ──
        try self.resolvePendingToken();

        if (try self.checkStop()) return null;

        const token = self.next_token_id;
        self.completion_tokens += 1;
        self.step += 1;
        try self.generated_ids.append(allocator, token);

        if (self.step % 256 == 0) _ = mlx.mlx_clear_cache();

        const step_logits = if (self.has_pending_logits) blk: {
            const logits = self.pending_logits;
            self.has_pending_logits = false;
            break :blk logits;
        } else blk: {
            const tok_i32: i32 = @intCast(token);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            break :blk try self.xfm.forwardWith(&self.ctx, tok_input);
        };

        // Logprobs: fully synchronous
        if (self.logprobs_n > 0) {
            defer _ = mlx.mlx_array_free(step_logits);
            const result = try sampleToken(allocator, step_logits, self.sampling, self.generated_ids.items, self.logprobs_n, self.xfm.s);
            self.next_token_id = result.token_id;
            if (self.last_logprob) |*lp| allocator.free(lp.top_logprobs);
            self.last_logprob = result.logprob_result;
            if (self.step < self.max_tokens) self.startAsyncForward(result.token_id);
            return token;
        }

        // Last token or pipeline bootstrap
        const lazy_token = sampleTokenLazy(step_logits, self.sampling, self.xfm.s);
        _ = mlx.mlx_array_free(step_logits);

        if (self.step < self.max_tokens) {
            const next_logits = lazyForward(self.xfm, &self.ctx, lazy_token) catch {
                try mlx.check(mlx.mlx_array_eval(lazy_token));
                var val: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
                _ = mlx.mlx_array_free(lazy_token);
                self.next_token_id = @intCast(val);
                return token;
            };

            const arr = [_]mlx.mlx_array{ lazy_token, next_logits };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);

            self.pending_token = lazy_token;
            self.has_pending_token = true;
            self.pending_logits = next_logits;
            self.has_pending_logits = true;
        } else {
            try mlx.check(mlx.mlx_array_eval(lazy_token));
            var val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
            _ = mlx.mlx_array_free(lazy_token);
            self.next_token_id = @intCast(val);
        }

        return token;
    }

    /// Synchronous, grammar-constrained sampling step. Used whenever
    /// `sampling.constraint` is non-null. Builds a token mask from the grammar's
    /// current state, applies it to the pending logits, samples, advances the
    /// grammar by the sampled token's bytes, and pre-launches the next forward
    /// pass to overlap with the next mask build.
    fn nextConstrained(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        if (!self.has_pending_logits) {
            self.done = true;
            return null;
        }
        if (self.stall.expired(self.timer.read(), self.generated_ids.items.len, self.timeout_ns)) {
            self.done = true;
            self.finish_reason = "length";
            return null;
        }
        if (self.step >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
            return null;
        }

        const constraint = self.sampling.constraint.?;
        const s = self.xfm.s;

        _ = try token_mask.buildMask(constraint.grammar, constraint.token_bytes, constraint.mask_buf);

        // Also allow every stop-id the generator recognises once the grammar is
        // complete. `token_mask.buildMask` only knows about `tokenizer.eos_id`,
        // but models often have additional stop tokens (e.g. `<|im_end|>` for
        // Qwen, `<end_of_turn>` for Gemma 4) registered via the config — without
        // this, the model can never stop.
        if (constraint.grammar.isComplete()) {
            for (self.eos_token_ids) |eos_id| {
                if (eos_id < constraint.mask_buf.len) constraint.mask_buf[eos_id] = true;
            }
        }

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        defer _ = mlx.mlx_array_free(step_logits);

        var masked_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked_logits);
        try applyGrammarMask(allocator, &masked_logits, step_logits, constraint.mask_buf, s);

        // Synchronous sample: we need the realized token id to advance the grammar.
        const lazy = sampleTokenLazy(masked_logits, self.sampling, s);
        try mlx.check(mlx.mlx_array_eval(lazy));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
        _ = mlx.mlx_array_free(lazy);
        const token: u32 = @intCast(val);
        self.next_token_id = token;

        // Stop on EOS — do not advance grammar or include in output.
        for (self.eos_token_ids) |eos_id| {
            if (token == eos_id) {
                self.done = true;
                self.finish_reason = "stop";
                return null;
            }
        }
        if (token == 0) {
            self.consecutive_pad += 1;
            if (self.consecutive_pad >= 3) {
                self.done = true;
                self.finish_reason = "stop";
                return null;
            }
        } else {
            self.consecutive_pad = 0;
        }

        // Advance the grammar by the sampled token's byte sequence. The mask
        // guarantees every byte is accepted (or the token has no byte form, e.g. a
        // special tag) — so a rejection here means a bug we want to surface.
        if (token < constraint.token_bytes.bytes.len) {
            if (constraint.token_bytes.bytes[token]) |bytes| {
                for (bytes) |b| {
                    const ok = try constraint.grammar.acceptByte(b);
                    if (!ok) {
                        log.warn("[grammar] sampled token {d} produced byte 0x{x} that was rejected — disabling further mask enforcement\n", .{ token, b });
                        constraint.grammar.dead = true;
                        break;
                    }
                }
            }
        }

        self.completion_tokens += 1;
        self.step += 1;
        try self.generated_ids.append(allocator, token);
        if (self.step % 256 == 0) _ = mlx.mlx_clear_cache();

        if (self.step < self.max_tokens) {
            const tok_i32: i32 = @intCast(token);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            const next_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
            const arr = [_]mlx.mlx_array{next_logits};
            const vec = mlx.mlx_vector_array_new_data(&arr, 1);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);
            self.pending_logits = next_logits;
            self.has_pending_logits = true;
        } else {
            self.done = true;
            self.finish_reason = "length";
        }

        return token;
    }

    /// Check all stop conditions. Returns true if generation should stop.
    fn checkStop(self: *Generator) !bool {
        if (self.step >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
            return true;
        }
        if (self.stall.expired(self.timer.read(), self.generated_ids.items.len, self.timeout_ns)) {
            self.done = true;
            self.finish_reason = "length";
            return true;
        }
        for (self.eos_token_ids) |eos_id| {
            if (self.next_token_id == eos_id) {
                self.done = true;
                self.finish_reason = "stop";
                return true;
            }
        }
        if (self.next_token_id == 0) {
            self.consecutive_pad += 1;
            if (self.consecutive_pad >= 3) {
                self.done = true;
                self.finish_reason = "stop";
                return true;
            }
        } else {
            self.consecutive_pad = 0;
        }
        return false;
    }

    /// Legacy sync forward for logprobs path.
    fn startAsyncForward(self: *Generator, token_id: u32) void {
        const tok_i32: i32 = @intCast(token_id);
        const tok_shape = [_]c_int{ 1, 1 };
        const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tok_input);

        const logits = self.xfm.forwardWith(&self.ctx, tok_input) catch return;
        const arr = [_]mlx.mlx_array{logits};
        const vec = mlx.mlx_vector_array_new_data(&arr, 1);
        _ = mlx.mlx_async_eval(vec);
        _ = mlx.mlx_vector_array_free(vec);

        self.pending_logits = logits;
        self.has_pending_logits = true;
    }
};

/// Build forward pass from a lazy sampled token array.
/// Reshapes [1] -> [1, 1] and calls transformer forward. All lazy (no eval).
fn lazyForward(xfm: *Transformer, ctx: *ForwardCtx, lazy_token: mlx.mlx_array) !mlx.mlx_array {
    const tok_shape = [_]c_int{ 1, 1 };
    var reshaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(reshaped);
    try mlx.check(mlx.mlx_reshape(&reshaped, lazy_token, &tok_shape, 2, xfm.s));
    return try xfm.forwardWith(ctx, reshaped);
}

/// Sample a token lazily from logits — returns a lazy MLX array (no eval).
/// Handles temperature scaling, top-k, and top-p, but defers materialization.
/// The returned array has shape [1] with the sampled token ID.
/// Caller must free the returned array.
/// Compute the probability distribution over the vocabulary at the LAST
/// position of `logits_3d` (shape `[B, S, V]`), with the SAME temperature +
/// top-k + top-p masking the sampler would apply. Both `target_p` and `draft_q`
/// in the stochastic-verify accept test must be computed via this function so
/// the ratio `p[draft] / q[draft]` is well-defined over the kept support.
/// Caller owns the returned array; shape `[B, V]`.
/// Batched sibling of `probsAtLastPos`: temperature → top-k → top-p →
/// softmax over EVERY position of `[1, L, V]` logits in one set of
/// row-parallel kernels. A per-position loop pays L separate ~vocab-sized
/// sort/topk kernel launches per spec-decode round; batched it's one each.
/// All filter helpers operate on the last axis, so leading dims pass through.
fn probsAllPositions(logits_3d: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    var current = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&current, logits_3d));

    if (sampling.temperature != 1.0) {
        const t = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(t);
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_divide(&scaled, current, t, s));
        _ = mlx.mlx_array_free(current);
        current = scaled;
    }
    if (sampling.top_k > 0) {
        var masked = mlx.mlx_array_new();
        applyTopK(&masked, current, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = masked;
    }
    if (sampling.top_p < 1.0) {
        var masked = mlx.mlx_array_new();
        applyTopP(&masked, current, sampling.top_p, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    var probs = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_softmax_axis(&probs, current, -1, true, s));
    _ = mlx.mlx_array_free(current);
    return probs;
}

/// Lazy log-confidence of one MTP draft: `logits[draft] − logsumexp(logits)`
/// = log p_head(draft). Two vocab reductions on the head's own (draft-head)
/// logits — must be built BEFORE the caller frees the logits handle (lazy
/// graphs hold their inputs internally). Returns a `[1]`-shaped lazy array.
fn draftConfidenceGraph(logits: mlx.mlx_array, draft_id: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var lse = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lse);
    try mlx.check(mlx.mlx_logsumexp_axis(&lse, logits, -1, false, s));
    var taken = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken);
    try mlx.check(mlx.mlx_take_axis(&taken, logits, draft_id, -1, s));
    const flat = [_]c_int{1};
    var t_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t_flat);
    try mlx.check(mlx.mlx_reshape(&t_flat, taken, &flat, 1, s));
    var l_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(l_flat);
    try mlx.check(mlx.mlx_reshape(&l_flat, lse, &flat, 1, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_subtract(&out, t_flat, l_flat, s));
    return out;
}

/// The chunk-A boundary sync: ONE bounded GPU round-trip that realizes the
/// chunk's draft ids (needed on the CPU later anyway) plus their
/// confidences, and returns the chain log-confidence
/// `Σ min(0, ln p_head(draft_i))` for the extension gate.
fn readChainConfidence(draft_arrs: []const mlx.mlx_array, conf_arrs: []const mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var conf_vec = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conf_vec);
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (conf_arrs) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
        var cat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cat);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, s));
        try mlx.check(mlx.mlx_astype(&conf_vec, cat, .float32, s));
    }
    {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        for (draft_arrs) |arr| _ = mlx.mlx_vector_array_append_value(eval_vec, arr);
        _ = mlx.mlx_vector_array_append_value(eval_vec, conf_vec);
        try mlx.check(mlx.mlx_async_eval(eval_vec));
    }
    try mlx.check(mlx.mlx_array_eval(conf_vec));
    const data = mlx.mlx_array_data_float32(conf_vec) orelse return error.MlxArrayDataNull;
    return Generator.mtpChainLogConf(data[0..conf_arrs.len]);
}

fn probsAtLastPos(logits_3d: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(logits_3d);
    const seq_len = shape[1];
    var current = mlx.mlx_array_new();
    if (seq_len == 1) {
        const sq_shape = [_]c_int{ shape[0], shape[2] };
        try mlx.check(mlx.mlx_reshape(&current, logits_3d, &sq_shape, 2, s));
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ shape[0], seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits_3d, &start, 3, &stop, 3, &strides, 3, s));
        const sq_shape = [_]c_int{ shape[0], shape[2] };
        try mlx.check(mlx.mlx_reshape(&current, sliced, &sq_shape, 2, s));
    }

    // Apply temperature → top-k → top-p (same order as `sampleTokenLazy`).
    if (sampling.temperature != 1.0) {
        const t = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(t);
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_divide(&scaled, current, t, s));
        _ = mlx.mlx_array_free(current);
        current = scaled;
    }
    if (sampling.top_k > 0) {
        var masked = mlx.mlx_array_new();
        applyTopK(&masked, current, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = masked;
    }
    if (sampling.top_p < 1.0) {
        var masked = mlx.mlx_array_new();
        applyTopP(&masked, current, sampling.top_p, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    // Softmax: tokens at -inf become 0, kept tokens renormalize to sum=1.
    var probs = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_softmax_axis(&probs, current, -1, true, s));
    _ = mlx.mlx_array_free(current);
    return probs;
}

/// Read `probs[0, token_id]` as f32. Forces realization with a single eval.
fn probAt(probs: mlx.mlx_array, token_id: u32, s: mlx.mlx_stream) !f32 {
    const idx_val: i32 = @intCast(token_id);
    const idx_shape = [_]c_int{1};
    const idx_arr = mlx.mlx_array_new_data(&idx_val, &idx_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(idx_arr);

    var taken = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken);
    try mlx.check(mlx.mlx_take_axis(&taken, probs, idx_arr, -1, s));

    // Cast to f32 so item_float32 is exact regardless of source dtype (bf16 etc.).
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, taken, .float32, s));
    try mlx.check(mlx.mlx_array_eval(as_f32));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, as_f32));
    return v;
}

/// Sample one token from probability distribution `probs` (shape `[B, V]`).
/// Returns a u32 token id (caller can append directly).
fn sampleFromProbs(probs: mlx.mlx_array, s: mlx.mlx_stream) !u32 {
    // mlx_random_categorical takes logits and applies softmax. Feed log(probs)
    // so the categorical's softmax recovers the original distribution.
    var log_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs);
    try mlx.check(mlx.mlx_log(&log_probs, probs, s));

    const null_key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(null_key);
    var sampled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sampled);
    try mlx.check(mlx.mlx_random_categorical(&sampled, log_probs, -1, null_key, s));
    try mlx.check(mlx.mlx_array_eval(sampled));
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, sampled));
    return @intCast(v);
}

/// Build a one-hot float32 row vector of shape `[1, vocab]` with 1.0 at
/// `index` and 0.0 elsewhere. Used by PLD's stochastic-verify reject path,
/// which models the draft (an n-gram lookup, not a probabilistic model) as a
/// degenerate one-hot distribution. Caller owns the returned array.
fn pldOneHotRow(index: u32, vocab: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var indices = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(indices);
    try mlx.check(mlx.mlx_arange(&indices, 0, @as(f64, @floatFromInt(vocab)), 1, .int32, s));

    const target_val: i32 = @intCast(index);
    const tgt_shape = [_]c_int{1};
    const target_idx = mlx.mlx_array_new_data(&target_val, &tgt_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(target_idx);

    var mask_bool = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_bool);
    try mlx.check(mlx.mlx_equal(&mask_bool, indices, target_idx, s));

    var mask_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_f32);
    try mlx.check(mlx.mlx_astype(&mask_f32, mask_bool, .float32, s));

    const out_shape = [_]c_int{ 1, vocab };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, mask_f32, &out_shape, 2, s));
    return out;
}

/// Sample from the residual distribution `residual = max(target - draft, 0)`,
/// renormalized. Used on stochastic-verify reject so the corrected token
/// preserves the target distribution (per Leviathan et al. speculative
/// decoding paper).
fn sampleResidual(target_probs: mlx.mlx_array, draft_probs: mlx.mlx_array, s: mlx.mlx_stream) !u32 {
    var diff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(diff);
    try mlx.check(mlx.mlx_subtract(&diff, target_probs, draft_probs, s));

    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var residual = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(residual);
    try mlx.check(mlx.mlx_maximum(&residual, diff, zero, s));

    return sampleFromProbs(residual, s);
}

pub fn sampleTokenLazy(logits: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) mlx.mlx_array {
    const shape = mlx.getShape(logits);
    const seq_len = shape[1];

    // Greedy + seq_len==1 (the decode hot path): one mlx op total. argmax_axis
    // over the vocab dim of a `[1, 1, V]` tensor yields a `[1, 1]` int array,
    // which downstream (resolvePendingToken / lazyForward / async_eval vector)
    // treats identically to `[1]`. Skipping the otherwise-needed reshape +
    // argmax-on-2D combo cuts ~one FFI call per decode step.
    if (seq_len == 1 and sampling.temperature < 0.01) {
        var result = mlx.mlx_array_new();
        _ = mlx.mlx_argmax_axis(&result, logits, -1, false, s);
        return result;
    }

    // Extract last position: [1, seq_len, vocab] -> [1, vocab]
    // `current` is the single owned intermediate — freed before each reassignment.
    var current = mlx.mlx_array_new();

    if (seq_len == 1) {
        const sq_shape = [_]c_int{ 1, shape[2] };
        _ = mlx.mlx_reshape(&current, logits, &sq_shape, 2, s);
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ 1, seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        _ = mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s);

        const sq_shape = [_]c_int{ 1, shape[2] };
        _ = mlx.mlx_reshape(&current, sliced, &sq_shape, 2, s);
    }

    // Greedy: argmax (no temperature)
    if (sampling.temperature < 0.01) {
        var result = mlx.mlx_array_new();
        _ = mlx.mlx_argmax_axis(&result, current, -1, false, s);
        _ = mlx.mlx_array_free(current);
        return result;
    }

    // Scale by 1/temperature
    if (sampling.temperature != 1.0) {
        const temp_arr = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(temp_arr);
        var next = mlx.mlx_array_new();
        _ = mlx.mlx_divide(&next, current, temp_arr, s);
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    // Apply top-k filtering (lazy)
    if (sampling.top_k > 0) {
        var next = mlx.mlx_array_new();
        applyTopK(&next, current, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    // Apply top-p filtering (lazy)
    if (sampling.top_p < 1.0) {
        var next = mlx.mlx_array_new();
        applyTopP(&next, current, sampling.top_p, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    // Sample from categorical distribution (lazy — no eval!)
    var sampled = mlx.mlx_array_new();
    const null_key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(null_key);
    _ = mlx.mlx_random_categorical(&sampled, current, -1, null_key, s);
    _ = mlx.mlx_array_free(current);

    return sampled; // Shape [1], lazy
}

/// Convenience: generate all tokens at once (non-streaming).
pub fn generate(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    logprobs_n: u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    var gen = try Generator.init(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids);
    gen.timeout_ns = timeout_ns;
    gen.logprobs_n = logprobs_n;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill: {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    var logprob_results = std.ArrayList(LogprobResult).empty;
    defer {
        if (logprobs_n == 0) {
            for (logprob_results.items) |*lp| allocator.free(lp.top_logprobs);
            logprob_results.deinit(allocator);
        }
    }

    timer.reset();
    while (try gen.next(allocator)) |token_id| {
        try output_ids.append(allocator, token_id);
        if (logprobs_n > 0) {
            if (gen.last_logprob) |lp| {
                try logprob_results.append(allocator, lp);
                gen.last_logprob = null; // Transfer ownership
            }
        }
    }

    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    log.debug("Decode: {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        decode_ns / std.time.ns_per_ms,
        num_decoded,
        decode_tps,
    });

    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);

    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = if (logprobs_n > 0) try logprob_results.toOwnedSlice(allocator) else null,
    };
}

/// PLD-enabled non-streaming variant of `generate`. Model-agnostic — works on
/// every supported architecture, no extra weights required. Logprobs and
/// constrained sampling are unsupported (asserted out by `nextPld`).
///
/// `draft_len` and `key_len` come from server config (`--pld-draft-len` /
/// `--pld-key-len`); typical values are 5 and 3 respectively.
pub fn generatePld(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    draft_len: u32,
    key_len: u32,
    lookup_prompt: ?[]const u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{ .pld_enabled = true, .lookup_prompt = lookup_prompt });
    gen.timeout_ns = timeout_ns;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill (PLD): {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    timer.reset();

    // Decode loop. Each `nextPld` returns 1..=(1+draft_len) tokens. Stop on
    // EOS / max_tokens / timeout. We check stop conditions on every emitted
    // token (drafts can include EOS just like regular sampling) so the early
    // exit is correct.
    decode: while (!gen.done and gen.completion_tokens < max_tokens) {
        const result = (try gen.nextPld(allocator, draft_len, key_len)) orelse break;
        defer allocator.free(result.tokens);
        // Match `generate`'s convention: stop tokens are NOT included in
        // output_ids. Check before appending — the speculative path has to do
        // this explicitly because `nextPld` emits multiple tokens at once and
        // can't return-null mid-batch like the single-token `next` does.
        for (result.tokens) |tok_id| {
            if (isEosId(tok_id, eos_token_ids)) {
                gen.done = true;
                gen.finish_reason = "stop";
                break :decode;
            }
            try output_ids.append(allocator, tok_id);
            if (output_ids.items.len >= max_tokens) {
                gen.done = true;
                gen.finish_reason = "length";
                break :decode;
            }
        }
        if (timeout_ns > 0 and timer.read() >= timeout_ns) {
            gen.done = true;
            gen.finish_reason = "length";
            break;
        }
    }

    return finishPldResult(&gen, &output_ids, allocator, prefill_tps, timer, tok);
}

/// Drafter-enabled non-streaming variant of `generate`. Mirrors
/// `generatePld` (multi-token-per-step emit pattern) but the draft comes from
/// a Gemma 4 assistant drafter cross-attending into the target's KV cache
/// instead of an n-gram lookup.
///
/// `drafter` must already be `bind()`-ed to `xfm`. `block_size` is the
/// per-round token budget (drafter forwards = block_size - 1; verify forward
/// length = block_size).
pub fn generateDrafter(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    drafter: *DrafterModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    block_size: u32,
    lookup_prompt: ?[]const u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{
        .drafter_enabled = true,
        .drafter = drafter,
        .drafter_block_size = block_size,
        .lookup_prompt = lookup_prompt,
    });
    gen.timeout_ns = timeout_ns;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill (drafter): {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    timer.reset();

    decode: while (!gen.done and gen.completion_tokens < max_tokens) {
        const result = (try gen.nextDrafter(allocator)) orelse break;
        defer allocator.free(result.tokens);
        for (result.tokens) |tok_id| {
            if (isEosId(tok_id, eos_token_ids)) {
                gen.done = true;
                gen.finish_reason = "stop";
                break :decode;
            }
            try output_ids.append(allocator, tok_id);
            if (output_ids.items.len >= max_tokens) {
                gen.done = true;
                gen.finish_reason = "length";
                break :decode;
            }
        }
        if (timeout_ns > 0 and timer.read() >= timeout_ns) {
            gen.done = true;
            gen.finish_reason = "length";
            break;
        }
    }

    return finishDrafterResult(&gen, &output_ids, allocator, prefill_tps, timer, tok);
}

fn finishDrafterResult(
    gen: *Generator,
    output_ids: *std.ArrayList(u32),
    allocator: std.mem.Allocator,
    prefill_tps: f64,
    timer: io_util.Stopwatch,
    tok: *const Tokenizer,
) !GenerationResult {
    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    if (gen.drafter_attempted > 0) {
        const avg_acc: f64 = @as(f64, @floatFromInt(gen.drafter_accepted_tokens)) / @as(f64, @floatFromInt(gen.drafter_attempted));
        log.info("Decode (drafter): {d}ms ({d} tokens, {d:.3} tok/s; drafter accept={d}/{d} attempts, avg {d:.2} tokens/attempt)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
            gen.drafter_accepted_tokens,
            gen.drafter_attempted,
            avg_acc,
        });
    } else {
        log.debug("Decode (drafter): {d}ms ({d} tokens, {d:.3} tok/s; no draft attempts)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
        });
    }
    gen.logSpecStats();
    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);
    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = null,
    };
}

/// MTP-enabled non-streaming variant of `generate`. Mirrors `generateDrafter`
/// but drives `nextMtp` (the model's own multi-token-prediction head).
/// `head` must already be `bind()`-ed to `xfm`.
pub fn generateMtp(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    head: *mtp_mod.MtpModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    depth: u32,
    lookup_prompt: ?[]const u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{
        .mtp_enabled = true,
        .mtp = head,
        .mtp_depth = depth,
        .lookup_prompt = lookup_prompt,
    });
    gen.timeout_ns = timeout_ns;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill (mtp): {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    timer.reset();

    decode: while (!gen.done and gen.completion_tokens < max_tokens) {
        const result = (try gen.nextMtp(allocator)) orelse break;
        defer allocator.free(result.tokens);
        for (result.tokens) |tok_id| {
            if (isEosId(tok_id, eos_token_ids)) {
                gen.done = true;
                gen.finish_reason = "stop";
                break :decode;
            }
            try output_ids.append(allocator, tok_id);
            if (output_ids.items.len >= max_tokens) {
                gen.done = true;
                gen.finish_reason = "length";
                break :decode;
            }
        }
        if (timeout_ns > 0 and timer.read() >= timeout_ns) {
            gen.done = true;
            gen.finish_reason = "length";
            break;
        }
    }

    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    if (gen.mtp_attempted > 0) {
        const avg_acc: f64 = @as(f64, @floatFromInt(gen.mtp_accepted_tokens)) / @as(f64, @floatFromInt(gen.mtp_attempted));
        log.info("Decode (mtp): {d}ms ({d} tokens, {d:.3} tok/s; mtp accept={d}/{d} attempts, avg {d:.2} tokens/attempt)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
            gen.mtp_accepted_tokens,
            gen.mtp_attempted,
            avg_acc,
        });
    } else {
        log.debug("Decode (mtp): {d}ms ({d} tokens, {d:.3} tok/s; no draft attempts)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
        });
    }
    gen.logSpecStats();
    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);
    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = null,
    };
}

fn finishPldResult(
    gen: *Generator,
    output_ids: *std.ArrayList(u32),
    allocator: std.mem.Allocator,
    prefill_tps: f64,
    timer: io_util.Stopwatch,
    tok: *const Tokenizer,
) !GenerationResult {
    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    if (gen.pld_attempted > 0) {
        // "Tokens saved" = accepted_tokens (drafts that landed) + 0 from
        // verify forwards that ran. Acceptance ratio is per-position, so we
        // compute average tokens accepted per attempt for visibility.
        const avg_acc: f64 = @as(f64, @floatFromInt(gen.pld_accepted_tokens)) / @as(f64, @floatFromInt(gen.pld_attempted));
        log.info("Decode (PLD): {d}ms ({d} tokens, {d:.3} tok/s; pld accept={d}/{d} attempts, avg {d:.2} tokens/attempt)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
            gen.pld_accepted_tokens,
            gen.pld_attempted,
            avg_acc,
        });
    } else {
        log.debug("Decode (PLD): {d}ms ({d} tokens, {d:.3} tok/s; no n-gram matches found)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
        });
    }
    gen.logSpecStats();
    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);
    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = null,
    };
}

pub fn isEosId(id: u32, eos: []const u32) bool {
    for (eos) |e| if (id == e) return true;
    return false;
}

/// Max cycle length (in tokens) scanned by `isDegenerateTailLoop`, and how many
/// identical repetitions of that cycle count as "stuck". A real answer — prose,
/// code, a markdown table — essentially never repeats an identical ≤8-token
/// cycle 16 times in a row, so these won't fire on legitimate output, while a
/// model that has collapsed into spamming one short phrase is caught within a
/// few dozen tokens instead of running all the way to `max_tokens`.
pub const degenerate_loop_max_period: usize = 8;
pub const degenerate_loop_reps: usize = 16;

/// Stall clock for the request timeout: the deadline measures time since the
/// last PRODUCED token, not since the request started. A wall-clock request
/// timeout kills legitimate long generations — live capture 2026-07-03:
/// Qwen3.6-27B writing a 33KB file in one tool call decodes for >300s at
/// ~30 tok/s and was guillotined mid-call by the 300s default, which then
/// surfaced as a "butchered" path-only tool call. Progress is detected from
/// the generated-token COUNT at each check, so every decode path (regular,
/// PLD, drafter, MTP — which don't all share an emit site) resets the clock
/// without instrumentation; a request that stops producing (hung forward,
/// deadlock) still times out after `timeout_ns` of silence.
pub const StallClock = struct {
    last_progress_ns: u64 = 0,
    last_progress_count: usize = 0,

    pub fn expired(self: *StallClock, now_ns: u64, generated_count: usize, timeout_ns: u64) bool {
        if (generated_count != self.last_progress_count) {
            self.last_progress_count = generated_count;
            self.last_progress_ns = now_ns;
        }
        if (timeout_ns == 0) return false;
        return now_ns -| self.last_progress_ns >= timeout_ns;
    }
};

/// Detect a degenerate tail loop: the model is stuck emitting the same short
/// token cycle over and over. Returns true when the last `reps` repetitions of
/// some period-`p` cycle (1 ≤ p ≤ `max_period`) are byte-identical.
///
/// Motivation: Gemma 4 12B sometimes collapses after a large/confusing tool
/// result and spams the thinking opener `<|channel>thought` forever; with no
/// repeat penalty (the default) and a now-generous `max_tokens`, nothing else
/// stops it. The decode loop calls this each tick and cuts the slot short.
///
/// Pure and cheap: only the trailing `max_period * reps` ids are inspected, so
/// cost is independent of total generated length.
pub fn isDegenerateTailLoop(tokens: []const u32, max_period: usize, reps: usize) bool {
    if (max_period == 0 or reps < 2) return false;
    var p: usize = 1;
    while (p <= max_period) : (p += 1) {
        const span = p * reps;
        if (tokens.len < span) continue;
        const tail = tokens[tokens.len - span ..];
        var periodic = true;
        var i: usize = p;
        while (i < tail.len) : (i += 1) {
            if (tail[i] != tail[i - p]) {
                periodic = false;
                break;
            }
        }
        if (periodic) return true;
    }
    return false;
}

/// Compute mean-pooled, L2-normalized embedding from token IDs.
/// Returns a float32 array of shape [hidden_size]. Caller must free the returned slice.
pub fn computeEmbedding(
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    token_ids: []const u32,
) ![]f32 {
    const seqs = [_][]const u32{token_ids};
    const rows = try computeEmbeddingsBatch(allocator, xfm, &seqs);
    defer allocator.free(rows);
    return rows[0];
}

/// GPU batch-size cap for encoder embedding forwards: bounds padded-batch
/// memory while keeping the GPU saturated.
pub const EMBED_MAX_BATCH: usize = 64;

/// One padded batch of token sequences ready for an encoder forward.
pub const PaddedBatch = struct {
    ids: []i32, // [B * max_len] row-major, pad id 0
    lengths: []usize, // [B]
    max_len: usize,

    pub fn deinit(self: *PaddedBatch, allocator: std.mem.Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.lengths);
    }
};

/// Pad `seqs` into one [B, max_len] i32 buffer (pad id 0). Padded positions
/// are excluded from attention (`buildKeyPadMask`) and pooling
/// (`maskedMeanPoolNormalize`), so the pad id value never leaks into results.
pub fn buildPaddedBatch(allocator: std.mem.Allocator, seqs: []const []const u32) !PaddedBatch {
    var max_len: usize = 0;
    for (seqs) |seq| max_len = @max(max_len, seq.len);
    if (max_len == 0) return error.EmptyInput;

    const ids = try allocator.alloc(i32, seqs.len * max_len);
    errdefer allocator.free(ids);
    const lengths = try allocator.alloc(usize, seqs.len);
    errdefer allocator.free(lengths);
    @memset(ids, 0);
    for (seqs, 0..) |seq, b| {
        lengths[b] = seq.len;
        for (seq, 0..) |id, t| ids[b * max_len + t] = @intCast(id);
    }
    return .{ .ids = ids, .lengths = lengths, .max_len = max_len };
}

/// Additive key-padding mask [B, 1, 1, max_len] (bf16): 0 over real keys,
/// -inf over padding. Broadcasts across heads and query positions; padded
/// QUERIES still produce garbage rows, but pooling drops them.
pub fn buildKeyPadMask(allocator: std.mem.Allocator, lengths: []const usize, max_len: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const buf = try allocator.alloc(f32, lengths.len * max_len);
    defer allocator.free(buf);
    for (lengths, 0..) |len, b| {
        for (0..max_len) |t| {
            buf[b * max_len + t] = if (t < len) 0 else -std.math.inf(f32);
        }
    }
    const shape = [_]c_int{ @intCast(lengths.len), 1, 1, @intCast(max_len) };
    const f32_mask = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_mask);
    var mask = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&mask, f32_mask, .bfloat16, s));
    return mask;
}

/// Mean-pool `hidden` [B, T, H] over each row's first `lengths[b]` positions
/// and L2-normalize. Returns B owned rows of H f32 each (plus the outer
/// slice); caller frees all.
pub fn maskedMeanPoolNormalize(allocator: std.mem.Allocator, hidden: mlx.mlx_array, lengths: []const usize, s: mlx.mlx_stream) ![][]f32 {
    const shape = mlx.getShape(hidden);
    const batch: usize = @intCast(shape[0]);
    const seq_len: usize = @intCast(shape[1]);
    const dim: usize = @intCast(shape[2]);

    // Pool weights [B, T, 1]: 1/len over real positions, 0 over padding — a
    // weighted sum along T is then exactly the masked mean. f32 weights also
    // promote a bf16 hidden so the final data extraction is float32-safe.
    const wbuf = try allocator.alloc(f32, batch * seq_len);
    defer allocator.free(wbuf);
    for (lengths, 0..) |len, b| {
        const denom: f32 = @floatFromInt(@max(len, 1));
        for (0..seq_len) |t| {
            wbuf[b * seq_len + t] = if (t < len) 1.0 / denom else 0.0;
        }
    }
    const wshape = [_]c_int{ shape[0], shape[1], 1 };
    const weights = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 3, .float32);
    defer _ = mlx.mlx_array_free(weights);

    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, hidden, weights, s));

    var pooled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_sum_axis(&pooled, weighted, 1, false, s)); // [B, H]

    // L2 normalize rows: pooled / max(sqrt(sum(pooled^2)), eps).
    var squared = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(squared);
    try mlx.check(mlx.mlx_multiply(&squared, pooled, pooled, s));

    var sum_sq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum_sq);
    try mlx.check(mlx.mlx_sum_axis(&sum_sq, squared, -1, true, s));

    var norm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm);
    try mlx.check(mlx.mlx_sqrt(&norm, sum_sq, s));

    const eps = mlx.mlx_array_new_float(1e-12);
    defer _ = mlx.mlx_array_free(eps);
    var norm_safe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm_safe);
    try mlx.check(mlx.mlx_maximum(&norm_safe, norm, eps, s));

    var normalized = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(normalized);
    try mlx.check(mlx.mlx_divide(&normalized, pooled, norm_safe, s));

    try mlx.check(mlx.mlx_array_eval(normalized));
    const data_ptr = mlx.mlx_array_data_float32(normalized) orelse return error.MlxError;

    const rows = try allocator.alloc([]f32, batch);
    var done: usize = 0;
    errdefer {
        for (rows[0..done]) |r| allocator.free(r);
        allocator.free(rows);
    }
    for (rows, 0..) |*row, b| {
        row.* = try allocator.alloc(f32, dim);
        @memcpy(row.*, data_ptr[b * dim .. (b + 1) * dim]);
        done += 1;
    }
    return rows;
}

/// Compute embeddings for many token sequences in GPU batches: each chunk of
/// up to `EMBED_MAX_BATCH` sequences is padded to its own max length,
/// forwarded ONCE through the encoder with a key-padding mask,
/// masked-mean-pooled, and L2-normalized. Input order preserved. Caller
/// frees every returned row and the outer slice.
pub fn computeEmbeddingsBatch(
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    seqs: []const []const u32,
) ![][]f32 {
    const results = try allocator.alloc([]f32, seqs.len);
    var filled: usize = 0;
    errdefer {
        for (results[0..filled]) |r| allocator.free(r);
        allocator.free(results);
    }
    var start: usize = 0;
    while (start < seqs.len) {
        const sub = seqs[start..@min(start + EMBED_MAX_BATCH, seqs.len)];
        var pb = try buildPaddedBatch(allocator, sub);
        defer pb.deinit(allocator);

        const shape = [_]c_int{ @intCast(sub.len), @intCast(pb.max_len) };
        const input = mlx.mlx_array_new_data(pb.ids.ptr, &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(input);

        // A single sequence has no padding, so it needs no mask.
        var mask: ?mlx.mlx_array = null;
        defer if (mask) |m| {
            _ = mlx.mlx_array_free(m);
        };
        if (sub.len > 1) mask = try buildKeyPadMask(allocator, pb.lengths, pb.max_len, xfm.s);

        const hidden = try xfm.forwardEmbeddingMasked(input, mask);
        defer _ = mlx.mlx_array_free(hidden);

        const rows = try maskedMeanPoolNormalize(allocator, hidden, pb.lengths, xfm.s);
        defer allocator.free(rows);
        for (rows, 0..) |r, i| {
            results[start + i] = r;
            filled += 1;
        }
        start += sub.len;
    }
    return results;
}

const SampleResult = struct {
    token_id: u32,
    logprob_result: ?LogprobResult = null,
};

/// Sample a token from the last position's logits.
/// temperature <= 0.01: greedy argmax. Otherwise: scale logits, apply top_p, and sample.
/// If logprobs_n > 0, also computes logprobs for the sampled token and top N alternatives.
fn sampleToken(allocator: std.mem.Allocator, logits: mlx.mlx_array, sampling: SamplingParams, generated_ids: ?[]const u32, logprobs_n: u32, s: mlx.mlx_stream) !SampleResult {
    const shape = mlx.getShape(logits);
    const seq_len = shape[1];

    // Extract last position: [1, seq_len, vocab] -> [1, vocab]
    var last_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(last_logits);

    if (seq_len == 1) {
        const sq_shape = [_]c_int{ 1, shape[2] };
        try mlx.check(mlx.mlx_reshape(&last_logits, logits, &sq_shape, 2, s));
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ 1, seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s));

        const sq_shape = [_]c_int{ 1, shape[2] };
        try mlx.check(mlx.mlx_reshape(&last_logits, sliced, &sq_shape, 2, s));
    }

    // Track current working logits (avoid copies when no transform needed)
    var current = last_logits;

    // Apply repeat penalty to already-generated tokens
    var penalized = mlx.mlx_array_new();
    var penalized_owned = false;
    defer if (penalized_owned) {
        _ = mlx.mlx_array_free(penalized);
    };

    const needs_penalty = (sampling.repeat_penalty != 1.0 or sampling.presence_penalty != 0.0);
    if (needs_penalty) {
        if (generated_ids) |ids| {
            if (ids.len > 0) {
                try applyRepeatPenalty(&penalized, current, ids, sampling.repeat_penalty, sampling.presence_penalty, s);
                current = penalized;
                penalized_owned = true;
            }
        }
    }

    // Greedy if temperature is ~0
    if (sampling.temperature < 0.01) {
        const token_id = try argmax(current, s);
        var logprob_result: ?LogprobResult = null;
        if (logprobs_n > 0) {
            logprob_result = try computeLogprobs(allocator, current, token_id, logprobs_n, s);
        }
        return .{ .token_id = token_id, .logprob_result = logprob_result };
    }

    // Scale logits by 1/temperature
    var scaled = mlx.mlx_array_new();
    var scaled_owned = false;
    defer if (scaled_owned) {
        _ = mlx.mlx_array_free(scaled);
    };

    if (sampling.temperature != 1.0) {
        const temp_arr = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(temp_arr);
        try mlx.check(mlx.mlx_divide(&scaled, current, temp_arr, s));
        current = scaled;
        scaled_owned = true;
    }

    // For logprobs, remember the logits after temp scaling but before filtering
    const logprobs_logits = current;

    // Apply top-k filtering
    var after_topk = mlx.mlx_array_new();
    var topk_owned = false;
    defer if (topk_owned) {
        _ = mlx.mlx_array_free(after_topk);
    };

    if (sampling.top_k > 0) {
        try applyTopK(&after_topk, current, sampling.top_k, s);
        current = after_topk;
        topk_owned = true;
    }

    // Apply top-p (nucleus) sampling
    var after_topp = mlx.mlx_array_new();
    var topp_owned = false;
    defer if (topp_owned) {
        _ = mlx.mlx_array_free(after_topp);
    };

    if (sampling.top_p < 1.0) {
        try applyTopP(&after_topp, current, sampling.top_p, s);
        current = after_topp;
        topp_owned = true;
    }

    // Sample from categorical distribution
    var sampled = mlx.mlx_array_new();

    if (sampling.seed) |seed| {
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, seed));
        try mlx.check(mlx.mlx_random_categorical(&sampled, current, -1, key, s));
    } else {
        const null_key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(null_key);
        try mlx.check(mlx.mlx_random_categorical(&sampled, current, -1, null_key, s));
    }

    // Eval and extract
    try mlx.check(mlx.mlx_array_eval(sampled));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, sampled));

    const token_id: u32 = @intCast(val);

    // Compute logprobs after sampling (we now know the token_id)
    var logprob_result: ?LogprobResult = null;
    if (logprobs_n > 0) {
        logprob_result = try computeLogprobs(allocator, logprobs_logits, token_id, logprobs_n, s);
    }

    _ = mlx.mlx_array_free(sampled);
    return .{ .token_id = token_id, .logprob_result = logprob_result };
}

/// Compute log-probabilities from logits. Returns the logprob of the chosen token
/// and the top N alternatives with their token IDs and logprobs.
fn computeLogprobs(allocator: std.mem.Allocator, logits: mlx.mlx_array, chosen_token: u32, top_n: u32, s: mlx.mlx_stream) !LogprobResult {
    // Compute log_softmax = log(softmax(logits)) on GPU
    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, logits, -1, true, s));

    var log_probs_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs_raw);
    try mlx.check(mlx.mlx_log(&log_probs_raw, probs, s));

    // Cast to float32 for CPU readback (model may produce float16 logits)
    var log_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs);
    try mlx.check(mlx.mlx_astype(&log_probs, log_probs_raw, .float32, s));

    // Get top-k logprobs using mlx_topk (returns top values in descending order)
    const k: c_int = @intCast(@min(top_n + 1, 20)); // +1 to ensure chosen token is included
    var topk_vals_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(topk_vals_raw);
    try mlx.check(mlx.mlx_topk(&topk_vals_raw, log_probs, k, s));

    var topk_vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(topk_vals);
    try mlx.check(mlx.mlx_astype(&topk_vals, topk_vals_raw, .float32, s));

    // Eval both arrays to CPU
    try mlx.check(mlx.mlx_array_eval(log_probs));
    try mlx.check(mlx.mlx_array_eval(topk_vals));

    // Read the logprob of the chosen token from the full array
    const lp_shape = mlx.getShape(log_probs);
    const vocab_size: usize = @intCast(lp_shape[lp_shape.len - 1]);
    const lp_data = mlx.mlx_array_data_float32(log_probs);
    const chosen_logprob: f32 = if (lp_data) |ptr|
        (if (chosen_token < vocab_size) ptr[chosen_token] else -100.0)
    else
        -100.0;

    // Read top-k values and find their token IDs by scanning the full logprobs
    const topk_data = mlx.mlx_array_data_float32(topk_vals);
    const actual_k: usize = @intCast(k);

    var top_logprobs = try allocator.alloc(TokenLogprob, @min(top_n, @as(u32, @intCast(actual_k))));
    var filled: usize = 0;

    if (topk_data) |tk_ptr| {
        if (lp_data) |full_ptr| {
            // Track used token IDs to avoid duplicates
            var used = std.AutoHashMap(u32, void).init(std.heap.page_allocator);
            defer used.deinit();

            // For each top-k value, find the matching token ID in the full array
            for (0..actual_k) |i| {
                if (filled >= top_n) break;
                const target_val = tk_ptr[i];
                // Find token ID with this logprob value (skip already-used IDs)
                for (0..vocab_size) |tid| {
                    if (full_ptr[tid] == target_val and !used.contains(@intCast(tid))) {
                        const tid_u32: u32 = @intCast(tid);
                        top_logprobs[filled] = .{
                            .token_id = tid_u32,
                            .logprob = target_val,
                        };
                        used.put(tid_u32, {}) catch {};
                        filled += 1;
                        break;
                    }
                }
            }
        }
    }

    // Shrink if we didn't fill all slots
    if (filled < top_logprobs.len) {
        top_logprobs = allocator.realloc(top_logprobs, filled) catch top_logprobs;
    }

    return .{
        .token_logprob = chosen_logprob,
        .top_logprobs = top_logprobs,
    };
}

/// Apply a grammar token mask to logits. `mask[i]==true` keeps `logits[i]`,
/// `false` replaces it with `-inf`. The mask is broadcast over leading dims so
/// `logits` can be either `[1, vocab]` or `[1, 1, vocab]`.
fn applyGrammarMask(allocator: std.mem.Allocator, res: *mlx.mlx_array, logits: mlx.mlx_array, mask: []const bool, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    const vocab_size: usize = @intCast(shape[shape.len - 1]);
    const logit_mask = try maskForLogitVocab(allocator, mask, vocab_size);
    defer logit_mask.deinit(allocator);

    // Zig's `bool` is one byte and matches MLX's `.bool_` storage exactly.
    const arr_shape = [_]c_int{@intCast(vocab_size)};
    const mask_arr = mlx.mlx_array_new_data(@ptrCast(logit_mask.slice.ptr), &arr_shape, 1, .bool_);
    defer _ = mlx.mlx_array_free(mask_arr);

    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);

    try mlx.check(mlx.mlx_where(res, mask_arr, logits, neg_inf, s));
}

const LogitMaskView = struct {
    slice: []const bool,
    owned: ?[]bool = null,

    fn deinit(self: LogitMaskView, allocator: std.mem.Allocator) void {
        if (self.owned) |buf| allocator.free(buf);
    }
};

fn maskForLogitVocab(allocator: std.mem.Allocator, mask: []const bool, vocab_size: usize) !LogitMaskView {
    if (mask.len == vocab_size) return .{ .slice = mask };

    var adjusted = try allocator.alloc(bool, vocab_size);
    @memset(adjusted, false);
    const copy_len = @min(mask.len, vocab_size);
    @memcpy(adjusted[0..copy_len], mask[0..copy_len]);
    return .{ .slice = adjusted, .owned = adjusted };
}

/// Apply top-k filtering: keep only the top k logits, set the rest to -inf.
fn applyTopK(res: *mlx.mlx_array, logits: mlx.mlx_array, k: u32, s: mlx.mlx_stream) !void {
    // Get the top-k values (returned in descending order)
    var topk_vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(topk_vals);
    try mlx.check(mlx.mlx_topk(&topk_vals, logits, @intCast(k), s));

    // Get the minimum of the top-k values (the k-th largest) as cutoff
    var cutoff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cutoff);
    try mlx.check(mlx.mlx_min_axis(&cutoff, topk_vals, -1, true, s));

    // Mask: logits >= cutoff
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_greater_equal(&mask, logits, cutoff, s));

    // Replace masked-out logits with -inf
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(res, mask, logits, neg_inf, s));
}

/// Apply top-p (nucleus) sampling: mask logits outside the top-p probability mass.
/// Works on the original (unsorted) logits by computing which tokens to keep.
fn applyTopP(res: *mlx.mlx_array, logits: mlx.mlx_array, top_p: f32, s: mlx.mlx_stream) !void {
    // Sort logits ascending to get sorted probabilities
    var sorted_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_logits);
    try mlx.check(mlx.mlx_sort_axis(&sorted_logits, logits, -1, s));

    // Softmax of sorted logits (ascending order: smallest probs first)
    var sorted_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_probs);
    try mlx.check(mlx.mlx_softmax_axis(&sorted_probs, sorted_logits, -1, true, s));

    // Cumulative sum from smallest to largest
    var cumsum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cumsum);
    try mlx.check(mlx.mlx_cumsum(&cumsum, sorted_probs, -1, false, true, s));

    // Find the cutoff: tokens where cumsum <= (1 - top_p) are outside the nucleus
    const threshold = mlx.mlx_array_new_float(1.0 - top_p);
    defer _ = mlx.mlx_array_free(threshold);

    var outside_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(outside_mask);
    try mlx.check(mlx.mlx_less_equal(&outside_mask, cumsum, threshold, s));

    // Set outside-nucleus logits to -inf in sorted space
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);

    // where(outside_mask, -inf, sorted_logits) — mask out the low-prob tokens
    try mlx.check(mlx.mlx_where(res, outside_mask, neg_inf, sorted_logits, s));

    // Note: categorical sampling doesn't care about token ordering,
    // but the sampled index will be in sorted space. We need to unsort.
    // Since categorical returns an index into the logits array, and we want
    // the original vocab index, we need to work in original space instead.

    // Better approach: find the minimum logit value that's in the nucleus,
    // then mask original logits below that threshold.
    _ = mlx.mlx_array_free(res.*);
    res.* = mlx.mlx_array_new();

    // The cutoff logit is the smallest logit still in the nucleus.
    // In sorted (ascending) order, tokens with cumsum > (1-top_p) are in nucleus.
    // The first such token's logit value is our threshold.
    // We can achieve this by: where(cumsum > 1-top_p, sorted_logits, +inf) then take min
    var in_nucleus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(in_nucleus);
    try mlx.check(mlx.mlx_greater(&in_nucleus, cumsum, threshold, s));

    const pos_inf = mlx.mlx_array_new_float(std.math.inf(f32));
    defer _ = mlx.mlx_array_free(pos_inf);

    var nucleus_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nucleus_logits);
    try mlx.check(mlx.mlx_where(&nucleus_logits, in_nucleus, sorted_logits, pos_inf, s));

    // Min value = the cutoff
    var min_val = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(min_val);
    try mlx.check(mlx.mlx_min_axis(&min_val, nucleus_logits, -1, true, s));

    // Mask original logits: keep if >= cutoff, else -inf
    var keep_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keep_mask);
    try mlx.check(mlx.mlx_greater_equal(&keep_mask, logits, min_val, s));

    try mlx.check(mlx.mlx_where(res, keep_mask, logits, neg_inf, s));
}

/// Apply repeat penalty to already-generated tokens.
/// Uses pure MLX GPU ops — no CPU readback, preserves lazy evaluation graph.
fn applyRepeatPenalty(res: *mlx.mlx_array, logits: mlx.mlx_array, generated_ids: []const u32, repeat_penalty: f32, presence_penalty: f32, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    const vocab_size: usize = @intCast(shape[shape.len - 1]);

    // Collect unique token ids
    var seen_set = std.AutoHashMap(u32, void).init(std.heap.page_allocator);
    defer seen_set.deinit();
    for (generated_ids) |id| {
        if (id < vocab_size) {
            seen_set.put(id, {}) catch continue;
        }
    }

    if (seen_set.count() == 0) return;

    // Build boolean mask: true at positions of seen tokens
    const mask_data = try std.heap.page_allocator.alloc(u8, vocab_size);
    defer std.heap.page_allocator.free(mask_data);
    @memset(mask_data, 0);

    var it = seen_set.keyIterator();
    while (it.next()) |id_ptr| {
        mask_data[id_ptr.*] = 1;
    }

    const arr_shape = [_]c_int{ 1, @intCast(vocab_size) };
    const mask_arr = mlx.mlx_array_new_data(mask_data.ptr, &arr_shape, 2, .bool_);
    defer _ = mlx.mlx_array_free(mask_arr);

    var current = logits;

    // Repeat penalty: multiply seen tokens by 1/penalty (positive) or penalty (negative)
    // This is equivalent to: where(mask & logits > 0, logits / penalty, where(mask, logits * penalty, logits))
    // Simplified: where(mask, where(logits > 0, logits / penalty, logits * penalty), logits)
    var penalized = mlx.mlx_array_new();
    var penalized_owned = false;
    defer if (penalized_owned) {
        _ = mlx.mlx_array_free(penalized);
    };

    if (repeat_penalty != 1.0) {
        const rp = mlx.mlx_array_new_float(repeat_penalty);
        defer _ = mlx.mlx_array_free(rp);
        const inv_rp = mlx.mlx_array_new_float(1.0 / repeat_penalty);
        defer _ = mlx.mlx_array_free(inv_rp);
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);

        // positive_mask = logits > 0
        var positive_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(positive_mask);
        try mlx.check(mlx.mlx_greater(&positive_mask, current, zero, s));

        // penalized_positive = logits * (1/penalty)
        var pen_pos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pen_pos);
        try mlx.check(mlx.mlx_multiply(&pen_pos, current, inv_rp, s));

        // penalized_negative = logits * penalty
        var pen_neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pen_neg);
        try mlx.check(mlx.mlx_multiply(&pen_neg, current, rp, s));

        // sign_selected = where(positive, logits/penalty, logits*penalty)
        var sign_selected = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sign_selected);
        try mlx.check(mlx.mlx_where(&sign_selected, positive_mask, pen_pos, pen_neg, s));

        // result = where(mask, sign_selected, logits)
        try mlx.check(mlx.mlx_where(&penalized, mask_arr, sign_selected, current, s));
        current = penalized;
        penalized_owned = true;
    }

    // Presence penalty: subtract from seen tokens
    if (presence_penalty != 0.0) {
        const pp = mlx.mlx_array_new_float(presence_penalty);
        defer _ = mlx.mlx_array_free(pp);

        // Cast mask to float for arithmetic
        var mask_float = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask_float);
        try mlx.check(mlx.mlx_astype(&mask_float, mask_arr, .float16, s));

        // subtract = mask * presence_penalty
        var subtract = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(subtract);
        try mlx.check(mlx.mlx_multiply(&subtract, mask_float, pp, s));

        // result = current - subtract
        try mlx.check(mlx.mlx_subtract(res, current, subtract, s));
    } else {
        try mlx.check(mlx.mlx_copy(res, current, s));
    }
}

/// Greedy argmax over the last axis.
fn argmax(last_logits: mlx.mlx_array, s: mlx.mlx_stream) !u32 {
    var argmax_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(argmax_arr);
    try mlx.check(mlx.mlx_argmax_axis(&argmax_arr, last_logits, -1, false, s));

    try mlx.check(mlx.mlx_array_eval(argmax_arr));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, argmax_arr));

    return @intCast(val);
}

// ── Tests ──

const testing = std.testing;

test "SamplingParams defaults" {
    const params = SamplingParams{};
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.temperature, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.top_p, 0.001);
    try testing.expectEqual(@as(u32, 0), params.top_k);
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.repeat_penalty, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), params.presence_penalty, 0.001);
    try testing.expect(params.seed == null);
}

test "SamplingParams custom values" {
    const params = SamplingParams{
        .temperature = 0.7,
        .top_p = 0.9,
        .top_k = 40,
        .repeat_penalty = 1.1,
        .presence_penalty = 0.5,
        .seed = 42,
    };
    try testing.expectApproxEqAbs(@as(f32, 0.7), params.temperature, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.9), params.top_p, 0.001);
    try testing.expectEqual(@as(u32, 40), params.top_k);
    try testing.expectEqual(@as(u64, 42), params.seed.?);
}

test "GenerationResult fields" {
    // Just verifying the struct shape compiles correctly with all fields
    const result = GenerationResult{
        .text = @constCast("hello"),
        .token_ids = @constCast(&[_]u32{ 1, 2, 3 }),
        .prompt_tokens = 10,
        .completion_tokens = 3,
        .finish_reason = "stop",
        .prefill_tps = 100.0,
        .decode_tps = 35.0,
        .logprobs = null,
    };
    try testing.expectEqual(@as(u32, 10), result.prompt_tokens);
    try testing.expectEqual(@as(u32, 3), result.completion_tokens);
    try testing.expectEqualStrings("stop", result.finish_reason);
    try testing.expect(result.logprobs == null);
}

test "tokensPerSec basic and zero-time" {
    // 100 tokens in 1 second = 100 tok/s.
    try testing.expectApproxEqAbs(@as(f64, 100.0), tokensPerSec(100, std.time.ns_per_s), 1e-6);
    // 50 tokens in 0.5s = 100 tok/s.
    try testing.expectApproxEqAbs(@as(f64, 100.0), tokensPerSec(50, std.time.ns_per_s / 2), 1e-6);
    // Zero elapsed → 0, never inf/NaN.
    try testing.expectEqual(@as(f64, 0.0), tokensPerSec(100, 0));
}

test "prefillTokensPerSec divides by uncached tokens" {
    // Cold prefill: 754 tokens, none cached, 2s → 377 tok/s.
    try testing.expectApproxEqAbs(
        @as(f64, 377.0),
        prefillTokensPerSec(754, 0, 2 * std.time.ns_per_s),
        1e-6,
    );
    // Warm prefill: 754-token prompt, 700 cached, only 54 ran. A fast 54-token
    // suffix in 0.1s is 540 tok/s — NOT 7540 (the inflated full-prompt rate).
    try testing.expectApproxEqAbs(
        @as(f64, 540.0),
        prefillTokensPerSec(754, 700, std.time.ns_per_s / 10),
        1e-6,
    );
    // Full cache hit: 0 uncached → 0 tok/s (no compute happened).
    try testing.expectEqual(@as(f64, 0.0), prefillTokensPerSec(754, 754, std.time.ns_per_s));
    // Defensive: cached > prompt (shouldn't happen) clamps to 0, no underflow.
    try testing.expectEqual(@as(f64, 0.0), prefillTokensPerSec(10, 20, std.time.ns_per_s));
}

test "maskForLogitVocab pads and truncates to logits size" {
    const short_mask = [_]bool{ true, false, true };
    const padded = try maskForLogitVocab(testing.allocator, &short_mask, 5);
    defer padded.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 5), padded.slice.len);
    try testing.expect(padded.slice[0]);
    try testing.expect(!padded.slice[1]);
    try testing.expect(padded.slice[2]);
    try testing.expect(!padded.slice[3]);
    try testing.expect(!padded.slice[4]);

    const long_mask = [_]bool{ false, true, true, true };
    const truncated = try maskForLogitVocab(testing.allocator, &long_mask, 2);
    defer truncated.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), truncated.slice.len);
    try testing.expect(!truncated.slice[0]);
    try testing.expect(truncated.slice[1]);
}

test "argmax selects highest value" {
    // Create a simple logits array [1, 5] with values [0.1, 0.5, 0.9, 0.2, 0.3]
    const data = [_]f32{ 0.1, 0.5, 0.9, 0.2, 0.3 };
    const shape = [_]c_int{ 1, 5 };
    const s = mlx.gpuStream();
    const arr = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(arr);

    const result = try argmax(arr, s);
    try testing.expectEqual(@as(u32, 2), result); // index 2 has value 0.9
}

test "argmax with negative values" {
    const data = [_]f32{ -5.0, -1.0, -3.0, -0.5, -2.0 };
    const shape = [_]c_int{ 1, 5 };
    const s = mlx.gpuStream();
    const arr = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(arr);

    const result = try argmax(arr, s);
    try testing.expectEqual(@as(u32, 3), result); // index 3 has value -0.5 (highest)
}

test "applyRepeatPenalty reduces seen token logits" {
    const s = mlx.gpuStream();
    // logits: [1.0, 2.0, 3.0, 4.0, 5.0]
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const shape = [_]c_int{ 1, 5 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // Penalize tokens at indices 1 and 3
    const generated = [_]u32{ 1, 3 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyRepeatPenalty(&res, logits, &generated, 2.0, 0.0, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: untouched → 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), res_data[0], 0.01);
    // Index 1: positive, divided by 2.0 → 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), res_data[1], 0.01);
    // Index 2: untouched → 3.0
    try testing.expectApproxEqAbs(@as(f32, 3.0), res_data[2], 0.01);
    // Index 3: positive, divided by 2.0 → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[3], 0.01);
    // Index 4: untouched → 5.0
    try testing.expectApproxEqAbs(@as(f32, 5.0), res_data[4], 0.01);
}

test "applyRepeatPenalty with negative logits" {
    const s = mlx.gpuStream();
    // Mix of positive and negative logits
    const data = [_]f32{ -2.0, 3.0, -1.0, 4.0 };
    const shape = [_]c_int{ 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // Penalize all tokens
    const generated = [_]u32{ 0, 1, 2, 3 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyRepeatPenalty(&res, logits, &generated, 2.0, 0.0, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: negative, multiplied by 2.0 → -4.0
    try testing.expectApproxEqAbs(@as(f32, -4.0), res_data[0], 0.01);
    // Index 1: positive, divided by 2.0 → 1.5
    try testing.expectApproxEqAbs(@as(f32, 1.5), res_data[1], 0.01);
    // Index 2: negative, multiplied by 2.0 → -2.0
    try testing.expectApproxEqAbs(@as(f32, -2.0), res_data[2], 0.01);
    // Index 3: positive, divided by 2.0 → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[3], 0.01);
}

test "applyRepeatPenalty presence penalty" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]c_int{ 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const generated = [_]u32{ 0, 2 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyRepeatPenalty(&res, logits, &generated, 1.0, 0.5, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: seen, presence penalty subtracted → 1.0 - 0.5 = 0.5
    try testing.expectApproxEqAbs(@as(f32, 0.5), res_data[0], 0.01);
    // Index 1: unseen → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[1], 0.01);
    // Index 2: seen → 3.0 - 0.5 = 2.5
    try testing.expectApproxEqAbs(@as(f32, 2.5), res_data[2], 0.01);
    // Index 3: unseen → 4.0
    try testing.expectApproxEqAbs(@as(f32, 4.0), res_data[3], 0.01);
}

test "applyRepeatPenalty combined penalties" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 2.0, -1.0, 3.0 };
    const shape = [_]c_int{ 1, 3 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const generated = [_]u32{ 0, 1 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyRepeatPenalty(&res, logits, &generated, 2.0, 1.0, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: positive, divide by 2.0 = 1.0, then - 1.0 = 0.0
    try testing.expectApproxEqAbs(@as(f32, 0.0), res_data[0], 0.01);
    // Index 1: negative, multiply by 2.0 = -2.0, then - 1.0 = -3.0
    try testing.expectApproxEqAbs(@as(f32, -3.0), res_data[1], 0.01);
    // Index 2: unseen → 3.0
    try testing.expectApproxEqAbs(@as(f32, 3.0), res_data[2], 0.01);
}

test "sampleToken greedy selects argmax" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    // Create logits [1, 1, 5] — 5 vocab entries, token at index 3 has highest
    const data = [_]f32{ 1.0, 0.5, 0.1, 5.0, 0.2 };
    const logits_shape = [_]c_int{ 1, 1, 5 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.0 };
    const result = try sampleToken(allocator, logits, params, null, 0, s);
    try testing.expectEqual(@as(u32, 3), result.token_id);
}

test "sampleToken with temperature produces valid token" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    const data = [_]f32{ 1.0, 2.0, 3.0 };
    const logits_shape = [_]c_int{ 1, 1, 3 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.5 };
    const result = try sampleToken(allocator, logits, params, null, 0, s);
    // Token should be in valid range
    try testing.expect(result.token_id < 3);
}

test "sampleToken from prefill logits (seq_len > 1)" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    // [1, 3, 4] — 3 positions, 4 vocab, should take last position
    const data = [_]f32{
        0.1, 0.2, 0.3, 0.4, // pos 0
        0.5, 0.6, 0.7, 0.8, // pos 1
        9.0, 0.1, 0.1, 0.1, // pos 2 — token 0 is clearly highest
    };
    const logits_shape = [_]c_int{ 1, 3, 4 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.0 };
    const result = try sampleToken(allocator, logits, params, null, 0, s);
    try testing.expectEqual(@as(u32, 0), result.token_id); // pos 2, index 0 = 9.0
}

test "Generator.runtimeGateShouldDisable below warmup never trips" {
    // Even with zero accepts, before the warmup count we trust the prompt-time
    // gate and never disable speculation mid-decode. drafts_per_round is the
    // typical drafter setting (block_size=4 → 3 drafts per round).
    try testing.expect(!Generator.runtimeGateShouldDisable(0, 0, 3));
    try testing.expect(!Generator.runtimeGateShouldDisable(1, 0, 3));
    try testing.expect(!Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP - 1, 0, 3));
}

test "Generator.runtimeGateShouldDisable trips at warmup with low per-draft rate" {
    // Synthetic low-accept scenario: 5 verify attempts, drafts_per_round=3
    // (drafter at block_size=4). 5 attempts × 3 = 15 drafts proposed.
    // 0 accepted → 0.00 < 0.50 → trip. Same with 1 accepted (0.067).
    try testing.expect(Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP, 0, 3));
    try testing.expect(Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP, 1, 3));
    // 7 accepted out of 15 = 0.467 — still below 0.50 → trip.
    try testing.expect(Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP, 7, 3));
    // 8 accepted out of 15 = 0.533 → keeps running.
    try testing.expect(!Generator.runtimeGateShouldDisable(Generator.RUNTIME_GATE_WARMUP, 8, 3));
}

test "Generator.runtimeGateShouldDisable does not trip with high per-draft rate" {
    // Echo workloads on Gemma drafter: ~93% per-draft acceptance (E4B from
    // bench: 67/(24*3) = 93.1%). Well above threshold → keeps running.
    try testing.expect(!Generator.runtimeGateShouldDisable(24, 67, 3));
    // PLD heavy-echo: ~4 of 5 drafts accepted per attempt = 0.80 per-draft.
    try testing.expect(!Generator.runtimeGateShouldDisable(20, 80, 5));
    // Edge case at exactly the threshold (rate == 0.50) — strict less-than,
    // so does NOT trip.
    try testing.expect(!Generator.runtimeGateShouldDisable(10, 15, 3)); // 15/30 = 0.50
}

test "Generator.runtimeGateShouldDisable creative-content regression scenario" {
    // The Phase 1 bench's exact regression cases on creative prompts:
    //   E4B drafter (bs=4 → drafts_per_round=3): 39/59 attempts → 22.0% per-draft → trip
    //   E2B drafter (bs=2 → drafts_per_round=1): 31/66 attempts → 47.0% per-draft → trip
    //   31B drafter (bs=8 → drafts_per_round=7): 60/(38*7) → 22.6% per-draft → trip
    try testing.expect(Generator.runtimeGateShouldDisable(59, 39, 3)); // E4B creative
    try testing.expect(Generator.runtimeGateShouldDisable(66, 31, 1)); // E2B creative
    try testing.expect(Generator.runtimeGateShouldDisable(38, 60, 7)); // 31B creative
    // The 26B-A4B@bs=2 creative case: 37/(60*1) = 61.7% → above threshold,
    // so the runtime gate alone does NOT save it. MoE regressions need the
    // separate `default_enable_drafter` opt-out at startup.
    try testing.expect(!Generator.runtimeGateShouldDisable(60, 37, 1));
}

test "Generator.runtimeGateShouldDisable handles drafts_per_round=0" {
    // Defensive: if a caller somehow passes a degenerate config (block_size=1
    // → drafts_per_round=0), don't divide by zero. We return false (no trip).
    try testing.expect(!Generator.runtimeGateShouldDisable(100, 0, 0));
}

test "Generator.yieldGateShouldDisable trips on cold-path-dominated workloads" {
    // The 2026-06-10 baseline regression: PLD forced on for a creative essay
    // prompt where the n-gram lookup almost never matches. The per-draft gate
    // never trips (it only counts verify ROUNDS, and there are few), but every
    // step pays the unpipelined cold forward → −14% measured on E2B. The
    // yield gate counts ALL enabled-mode steps: accepted-drafted-tokens per
    // step below the threshold after warmup → disable.
    // Creative: 128 steps, ~6 drafted tokens accepted → yield 0.047 → trip.
    try testing.expect(Generator.yieldGateShouldDisable(128, 6));
    // Heavy echo: 40 steps, 80 accepted (2.0/step) → stay on.
    try testing.expect(!Generator.yieldGateShouldDisable(40, 80));
    // Inside warmup: never trip, even at zero yield.
    try testing.expect(!Generator.yieldGateShouldDisable(Generator.YIELD_GATE_WARMUP - 1, 0));
    // Exactly at warmup with healthy yield: stay on.
    try testing.expect(!Generator.yieldGateShouldDisable(Generator.YIELD_GATE_WARMUP, Generator.YIELD_GATE_WARMUP));
}

test "Generator.specShouldReenable gates mid-request PLD re-activation" {
    // Disabled-mode periodic check on the COMMITTED sequence (prompt +
    // generated). The decisive case: the model echoes PROMPT content (file
    // edit / tool result) after a novel preamble tripped the yield gate. The
    // echoed tail never repeats ITSELF, so self-repetition scoring misses it;
    // tailMatchFraction sees the prompt occurrence.
    var committed: [96]u32 = undefined;
    // prompt = 48-token "file", generated = 16 novel preamble + 32 echo of the file
    for (committed[0..48], 0..) |*t, i| t.* = @intCast(i + 100);
    for (committed[48..64], 0..) |*t, i| t.* = @intCast(i + 9000);
    for (committed[64..96], 0..) |*t, i| t.* = @intCast(i + 100);
    try testing.expect(Generator.specShouldReenable(&committed, 48));

    // Fully novel committed sequence → stay disabled.
    var novel: [96]u32 = undefined;
    for (&novel, 0..) |*t, i| t.* = @intCast(i * 7 + 1);
    try testing.expect(!Generator.specShouldReenable(&novel, 48));

    // Too little generated yet → not enough signal, stay disabled.
    try testing.expect(!Generator.specShouldReenable(&committed, 8));
}

test "InitOptions.lookup_prompt overrides prompt_ids_owned source" {
    // When the server's cache-reuse path forwards only a trailing-tail
    // prompt slice but supplies the full original prompt via
    // `InitOptions.lookup_prompt`, PLD's n-gram buffer must be cloned from
    // the full slice — not the truncated tail.
    const tail = [_]u32{99};
    const full = [_]u32{ 10, 20, 30, 99 };
    const src = Generator.pickLookupPromptSource(&tail, &full);
    try testing.expectEqual(@as(usize, 4), src.len);
    try testing.expectEqualSlices(u32, &full, src);
}

test "InitOptions.lookup_prompt = null preserves existing behavior" {
    // Back-compat path: when callers don't set `lookup_prompt`, the source
    // is the unmodified `prompt_ids` slice — same buffer the Generator
    // received pre-fix.
    const prompt = [_]u32{ 1, 2, 3, 4, 5 };
    const src = Generator.pickLookupPromptSource(&prompt, null);
    try testing.expectEqual(prompt.len, src.len);
    try testing.expectEqualSlices(u32, &prompt, src);
    try testing.expectEqual(@as([*]const u32, prompt[0..].ptr), src.ptr);
}

test "StallClock: progress resets the deadline, silence expires it, 0 disables" {
    var clock = StallClock{};
    const s = std.time.ns_per_s;
    // Producing tokens keeps resetting the deadline — a healthy generation
    // can run arbitrarily long (the live bug: a 33KB tool call at 30 tok/s
    // takes >300s and was guillotined mid-call by the wall-clock timeout).
    try std.testing.expect(!clock.expired(0 * s, 0, 300 * s));
    try std.testing.expect(!clock.expired(299 * s, 1000, 300 * s)); // progress at 299s
    try std.testing.expect(!clock.expired(598 * s, 2000, 300 * s)); // progress again
    // No new tokens for the full window -> stalled.
    try std.testing.expect(!clock.expired(700 * s, 2000, 300 * s));
    try std.testing.expect(clock.expired(898 * s, 2000, 300 * s));
    // 0 = disabled, even after silence.
    var off = StallClock{};
    try std.testing.expect(!off.expired(0, 0, 0));
    try std.testing.expect(!off.expired(10_000 * s, 0, 0));
}

test "isDegenerateTailLoop catches a repeated channel-opener cycle" {
    const P = degenerate_loop_max_period;
    const R = degenerate_loop_reps;

    // Gemma 4 12B failure mode: the model spams the thinking opener
    // `<|channel>thought\n` — model that as a 3-token cycle. After enough
    // identical repetitions the tail is a pure period-3 loop → fire.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        try ids.appendSlice(testing.allocator, &[_]u32{ 7, 8, 9 }); // some real prefix
        var k: usize = 0;
        while (k < R + 4) : (k += 1) {
            try ids.appendSlice(testing.allocator, &[_]u32{ 101, 102, 103 }); // <|channel>,thought,\n
        }
        try testing.expect(isDegenerateTailLoop(ids.items, P, R));
    }

    // A single token stuck on repeat (period 1) also counts once it passes R.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        var k: usize = 0;
        while (k < R + 2) : (k += 1) try ids.append(testing.allocator, 42);
        try testing.expect(isDegenerateTailLoop(ids.items, P, R));
    }
}

test "isDegenerateTailLoop does not fire on healthy or briefly-repeating output" {
    const P = degenerate_loop_max_period;
    const R = degenerate_loop_reps;

    // Strictly increasing ids — no cycle at all.
    {
        var ids: [200]u32 = undefined;
        for (&ids, 0..) |*v, i| v.* = @intCast(i);
        try testing.expect(!isDegenerateTailLoop(&ids, P, R));
    }
    // A short burst of repetition (well under R reps) must be left alone — a
    // model legitimately writing "ha ha ha" or a few identical list bullets.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        try ids.appendSlice(testing.allocator, &[_]u32{ 1, 2, 3, 4, 5 });
        var k: usize = 0;
        while (k < R - 1) : (k += 1) try ids.appendSlice(testing.allocator, &[_]u32{ 50, 51 });
        try testing.expect(!isDegenerateTailLoop(ids.items, P, R));
    }
    // Periodic tail but with a longer period than we scan for → ignored.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        var k: usize = 0;
        var base: u32 = 0;
        while (k < R) : (k += 1) {
            // period = P + 3 (> max_period); never a pure short cycle.
            var j: u32 = 0;
            while (j < P + 3) : (j += 1) try ids.append(testing.allocator, base + j);
            base = 0; // same long block repeats, but its period exceeds the scan window
        }
        try testing.expect(!isDegenerateTailLoop(ids.items, P, R));
    }
    // Too few tokens to judge.
    try testing.expect(!isDegenerateTailLoop(&[_]u32{ 1, 1 }, P, R));
}

test "nextChunkEnd: a tiny trailing remainder merges into the last chunk" {
    // A chat-templated prompt often lands a token or two past the chunk size
    // (8192-target prompts tokenize to 8193). A 1-token trailing chunk pays a
    // FULL graph + eval-barrier + cache-clear for one token — pure overhead.
    // Without checkpoint alignment, remainders under the merge floor extend
    // the current chunk instead.
    try testing.expectEqual(@as(usize, 8193), nextChunkEnd(0, 8193, 8192, false, 0, 0));
    // A substantial remainder stays its own chunk.
    try testing.expectEqual(@as(usize, 8192), nextChunkEnd(0, 8192 + 600, 8192, false, 0, 0));
    // Mid-prompt chunks are untouched.
    try testing.expectEqual(@as(usize, 8192), nextChunkEnd(0, 16385, 8192, false, 0, 0));
    try testing.expectEqual(@as(usize, 16385), nextChunkEnd(8192, 16385, 8192, false, 0, 0));
    // With SSM-checkpoint alignment active, behavior is unchanged (boundaries
    // must stay stride-aligned for the prefix cache).
    try testing.expectEqual(@as(usize, 8192), nextChunkEnd(0, 8193, 8192, true, 8192, 0));
}

test "prefillChunkCount: SSM-checkpoint stride controls cold-prefill chunking" {
    const PREFILL_CHUNK: usize = 8192;
    // Non-hybrid (or checkpointing off): a sub-PREFILL_CHUNK prompt is ONE chunk.
    try testing.expectEqual(@as(usize, 1), prefillChunkCount(851, PREFILL_CHUNK, false, 0, 0));
    try testing.expectEqual(@as(usize, 1), prefillChunkCount(8000, PREFILL_CHUNK, false, 0, 0));
    // Tail merge: one token past a chunk boundary is still ONE chunk.
    try testing.expectEqual(@as(usize, 1), prefillChunkCount(8193, PREFILL_CHUNK, false, 0, 0));
    try testing.expectEqual(@as(usize, 2), prefillChunkCount(16385, PREFILL_CHUNK, false, 0, 0));
    // The regression: a fine stride splits an 851-token prefill into 4 chunks
    // (851 spans boundaries 256/512/768). Harmless on compute-bound dense models
    // but on a memory-bound MoE prefill each chunk re-streams the expert weights
    // (~25% slower on 35B-class). The non-MoE path keeps this fine stride.
    try testing.expectEqual(@as(usize, 4), prefillChunkCount(851, PREFILL_CHUNK, true, 256, 0));
    // Boundary alignment is ABSOLUTE (warm path passes an offset): a tail-only
    // prefill starting mid-sequence still snaps to global strides. offset=2000,
    // prefix tail of 200 (abs 2000..2200), stride 256 -> boundary 2048/2304? only
    // 2048 falls inside (2000..2200) -> 2 chunks.
    try testing.expectEqual(@as(usize, 2), prefillChunkCount(200, PREFILL_CHUNK, true, 256, 2000));
}

test "boundedPrefillChunk: fused head dims and short contexts keep the base chunk" {
    // head_dim <= 128 rides MLX's fused SDPA — no materialized scores, no cap,
    // at ANY context length.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 128, 16, 1_000_000, true));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 64, 32, 1_000_000, true));
    // hd 256 but short context: 16 heads x 8192 ctx x 8192 chunk x 2B
    // = 2 GiB scores, inside the 4 GiB budget -> full chunk kept. This is the
    // fleet-protection property: every Gemma-4 / Qwen3.5/3.6 checkpoint ships
    // head_dim 256, so typical prompts must keep full prefill throughput.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 16, 8192, true));
    // Degenerate inputs never cap.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 0, 100_000, true));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 16, 0, true));
}

test "boundedPrefillChunk: caps hd-256 long context even with the fused kernel active" {
    // The msv_attn_p256 kernel removes the SCORE transient, but a big chunk
    // still scales the MoE-gather / KV-concat transients — measured +22 GB
    // peak for +3% speed at a 99K prompt. The cap deliberately ignores
    // prefillHeadDimFused (see the fn doc); pin that with the override ON.
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(8192, 256, 16, 100_000, true));
}

test "boundedPrefillChunk: long context shrinks to the scores budget, floored and rounded" {
    // gemma-4-26B geometry (16 heads): budget/(16*ctx*2) …
    // ctx 32768 -> exactly 4096.
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 16, 32768, true));
    // ctx 100000 -> raw 1342, rounded down to the 512 grain -> 1024.
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(8192, 256, 16, 100_000, true));
    // ctx 262144 (the PR-#69 255K case) -> 512.
    try testing.expectEqual(@as(usize, 512), boundedPrefillChunk(8192, 256, 16, 262_144, true));
    // Qwen3.6-27B geometry (24 heads) at 262144: raw 341 -> floor 512.
    try testing.expectEqual(@as(usize, 512), boundedPrefillChunk(8192, 256, 24, 262_144, true));
    // e4b geometry (8 heads) at 131072: exactly 2048.
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 256, 8, 131_072, true));
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
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(1024, 256, 16, 32768, true));
    // Even the floor never raises a tiny explicit base.
    try testing.expectEqual(@as(usize, 256), boundedPrefillChunk(256, 256, 16, 262_144, true));
}

test "boundedPrefillChunk: composed-causal hd-256 archs cap at 2048" {
    // Archs with NO sliding-band layers (qwen3_5/3_6 class: GDN + full
    // attention, has_sliding_window=false) run every hd-256 prefill through
    // the composed causal path, where SMALLER chunks measured faster on the
    // 27B ladder (2026-07-12, M4 Max): 8K prompt 225 -> 235.8 tok/s at
    // chunk 2048 (peak 28.9 -> 19.8 GB), 32K 205.4 -> 209.3. Chunking IS
    // block-level causal skipping for composed attention, and the score
    // transient shrinks with it. Cap the auto chunk at 2048.
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 256, 24, 8192, false));
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 256, 24, 16384, false));
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 256, 24, 32768, false));
    // The scores-budget formula still wins BELOW the cap: 64K on 24 heads
    // yields 1024 (measured better than 2048 there: 186 vs 182.3 tok/s).
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(8192, 256, 24, 65536, false));
    // Never raises a caller-lowered base.
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(1024, 256, 24, 8192, false));
    // Sliding-band archs (gemma: fused band kernel wants big chunks) keep
    // the formula-only policy.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 16, 8192, true));
    // Fused head dims never cap regardless of arch.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 128, 24, 1_000_000, false));
}

test "effectiveSsmCheckpointStride: MoE coarsens to PREFILL_CHUNK, dense keeps fine" {
    const PREFILL_CHUNK: usize = 8192;
    // Disabled stays disabled regardless of model type.
    try testing.expectEqual(@as(usize, 0), effectiveSsmCheckpointStride(0, false, PREFILL_CHUNK));
    try testing.expectEqual(@as(usize, 0), effectiveSsmCheckpointStride(0, true, PREFILL_CHUNK));
    // Non-MoE hybrid keeps the fine base stride (cheap chunking, finer warm reuse).
    try testing.expectEqual(@as(usize, 256), effectiveSsmCheckpointStride(256, false, PREFILL_CHUNK));
    // MoE coarsens to at least PREFILL_CHUNK so checkpointing never sub-divides
    // the memory-bound chunk -> MoE prefill is single-chunk for any prompt the
    // mem-bound path wouldn't already split.
    try testing.expectEqual(@as(usize, 8192), effectiveSsmCheckpointStride(256, true, PREFILL_CHUNK));
    // A larger explicit stride is respected (never shrunk).
    try testing.expectEqual(@as(usize, 16384), effectiveSsmCheckpointStride(16384, true, PREFILL_CHUNK));
    // End-to-end: with the effective stride, an 851-tok MoE prefill is 1 chunk
    // (was 4 at the raw 256), while a dense/non-MoE hybrid stays at 4.
    try testing.expectEqual(@as(usize, 1), prefillChunkCount(851, PREFILL_CHUNK, true, effectiveSsmCheckpointStride(256, true, PREFILL_CHUNK), 0));
    try testing.expectEqual(@as(usize, 4), prefillChunkCount(851, PREFILL_CHUNK, true, effectiveSsmCheckpointStride(256, false, PREFILL_CHUNK), 0));
}

fn mtpEvTestGenerator() Generator {
    var g: Generator = undefined;
    g.mtp_depth = Generator.MTP_ADAPTIVE_DEFAULT_CAP;
    g.mtp_depth_current = 1;
    g.mtp_window_drafted = [_]u8{0} ** Generator.MTP_DEPTH_WINDOW;
    g.mtp_window_accepted = [_]u8{0} ** Generator.MTP_DEPTH_WINDOW;
    g.mtp_window_idx = 0;
    g.mtp_rounds_since_switch = 0;
    g.mtp_promote_cooldown = 0;
    g.mtp_ev_accept = [_]f32{Generator.MTP_EV_PRIOR} ** mtp_mod.MAX_DEPTH;
    g.mtp_ev_rounds = Generator.MTP_EV_WARMUP_ROUNDS;
    g.mtp_ev_m_lo_prev = 1;
    g.mtp_ev_costs = Generator.MTP_EV_DEFAULT_COSTS;
    g.spec_disabled_runtime = false;
    return g;
}

test "mtpNextDepth: adaptive depth policy transitions" {
    const configured: u32 = 3;
    // Hot at configured depth: stay.
    try testing.expectEqual(@as(u32, 3), Generator.mtpNextDepth(3, configured, 0.9));
    // Sagging at depth > 1: step down (one level at a time). The demote
    // floor is 0.40 under capture-based rollback (a rejected draft costs
    // only its own head pass, not a trunk re-forward).
    try testing.expectEqual(@as(u32, 2), Generator.mtpNextDepth(3, configured, 0.35));
    try testing.expectEqual(@as(u32, 1), Generator.mtpNextDepth(2, configured, 0.30));
    // Mid-band (0.40..0.60): hold.
    try testing.expectEqual(@as(u32, 3), Generator.mtpNextDepth(3, configured, 0.48));
    try testing.expectEqual(@as(u32, 2), Generator.mtpNextDepth(2, configured, 0.55));
    try testing.expectEqual(@as(u32, 1), Generator.mtpNextDepth(1, configured, 0.55));
    // Hot below configured depth: promote (band top is 0.60).
    try testing.expectEqual(@as(u32, 2), Generator.mtpNextDepth(1, configured, 0.73));
    try testing.expectEqual(@as(u32, 3), Generator.mtpNextDepth(2, configured, 0.70));
    // Never exceeds configured.
    try testing.expectEqual(@as(u32, 3), Generator.mtpNextDepth(3, configured, 0.99));
    // Depth 1 below the floor: disable (0).
    try testing.expectEqual(@as(u32, 0), Generator.mtpNextDepth(1, configured, 0.40));
    // Demote-before-disable: a terrible rate at depth 2 still goes through 1.
    try testing.expectEqual(@as(u32, 1), Generator.mtpNextDepth(2, configured, 0.10));
}

test "mtpDepthDecision: confidence gates on disable, promote, cooldown" {
    const W = Generator.MTP_DEPTH_WINDOW;
    // Disable needs a FULL window of evidence; small samples hold at depth 1.
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(1, 3, 0.20, 5, false));
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(1, 3, 0.20, W - 1, false));
    try testing.expectEqual(@as(u32, 0), Generator.mtpDepthDecision(1, 3, 0.20, W, false));
    // Promote needs >= 8 rounds AND no active cooldown.
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(1, 3, 0.95, 7, false));
    try testing.expectEqual(@as(u32, 2), Generator.mtpDepthDecision(1, 3, 0.95, 8, false));
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(1, 3, 0.95, 8, true));
    // Demote reacts on a small sample, even during cooldown.
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(2, 3, 0.30, 5, true));
}

test "MTP EV seed defaults on and explicit zero disables" {
    try testing.expect(Generator.mtpEvSeedEnabledFromEnv(null));
    try testing.expect(Generator.mtpEvSeedEnabledFromEnv(""));
    try testing.expect(Generator.mtpEvSeedEnabledFromEnv("1"));
    try testing.expect(!Generator.mtpEvSeedEnabledFromEnv("0"));
    try testing.expect(!Generator.mtpEvSeedEnabledFromEnv("0-disabled"));
}

test "mtpDepthCapFor: auto cap follows the selected cost profile; explicit always wins" {
    // 0 = auto (--mtp-depth not passed).
    try testing.expectEqual(Generator.MTP_ADAPTIVE_DEFAULT_CAP, Generator.mtpDepthCapForProfile(0, true, .generic));
    try testing.expectEqual(@as(u32, 6), Generator.mtpDepthCapForProfile(0, true, .generic));
    for ([_]mtp_mod.MtpCostProfile{ .g17_nax_q8_gs32, .g17_nax_q4_gs32 }) |profile| {
        try testing.expectEqual(Generator.MTP_ADAPTIVE_NAX_CAP, Generator.mtpDepthCapForProfile(0, true, profile));
        try testing.expectEqual(@as(u32, 8), Generator.mtpDepthCapForProfile(0, true, profile));
    }
    for ([_]mtp_mod.MtpCostProfile{ .generic, .g17_nax_q8_gs32, .g17_nax_q4_gs32 }) |profile| {
        try testing.expectEqual(mtp_mod.DEFAULT_DEPTH, Generator.mtpDepthCapForProfile(0, false, profile));
    }
    // Explicit values ignore both controller mode and profile, and remain
    // clamped to [1, MAX_DEPTH].
    try testing.expectEqual(@as(u32, 5), Generator.mtpDepthCapForProfile(5, true, .generic));
    try testing.expectEqual(@as(u32, 5), Generator.mtpDepthCapForProfile(5, false, .g17_nax_q8_gs32));
    try testing.expectEqual(@as(u32, 7), Generator.mtpDepthCapForProfile(7, true, .generic));
    try testing.expectEqual(@as(u32, 8), Generator.mtpDepthCapForProfile(8, true, .generic));
    try testing.expectEqual(@as(u32, 2), Generator.mtpDepthCapForProfile(2, true, .g17_nax_q4_gs32));
    try testing.expectEqual(mtp_mod.MAX_DEPTH, Generator.mtpDepthCapForProfile(12, true, .generic));

    // The original boolean helpers remain source-compatible and map true to
    // the pre-existing q8 profile.
    try testing.expectEqual(@as(u32, 6), Generator.mtpDepthCapFor(0, true, false));
    try testing.expectEqual(@as(u32, 8), Generator.mtpDepthCapFor(0, true, true));
}

test "mtpFloorDisableObserve: extension misses do not poison depth one" {
    const W = Generator.MTP_DEPTH_WINDOW;
    var drafted = [_]u8{0} ** W;
    var accepted = [_]u8{0} ** W;
    var idx: u32 = 0;

    var rate: ?f32 = null;
    for (0..W) |_| {
        // An eight-wide extension accepted its first draft but no later one.
        rate = Generator.mtpFloorDisableObserve(&drafted, &accepted, &idx, 1, 8, 1);
    }
    try testing.expectApproxEqAbs(@as(f32, 1.0), rate.?, 1e-5);
    for (drafted) |sample| try testing.expectEqual(@as(u8, 1), sample);

    drafted = [_]u8{0} ** W;
    accepted = [_]u8{0} ** W;
    idx = 0;
    for (0..W) |i| {
        rate = Generator.mtpFloorDisableObserve(
            &drafted,
            &accepted,
            &idx,
            1,
            8,
            @intFromBool(i % 2 == 0),
        );
    }
    try testing.expectApproxEqAbs(Generator.MTP_DISABLE_BELOW, rate.?, 1e-5);
    try testing.expect(!(rate.? < Generator.MTP_DISABLE_BELOW));
}

test "mtpFloorDisableObserve: disable needs 16 fresh failures at base depth one" {
    const W = Generator.MTP_DEPTH_WINDOW;
    var drafted = [_]u8{0} ** W;
    var accepted = [_]u8{0} ** W;
    var idx: u32 = 0;

    // Fifteen depth-one failures are insufficient.
    for (0..W - 1) |_| {
        try testing.expectEqual(
            @as(?f32, null),
            Generator.mtpFloorDisableObserve(&drafted, &accepted, &idx, 1, 1, 0),
        );
    }
    // A wider base round invalidates that probation window.
    try testing.expectEqual(
        @as(?f32, null),
        Generator.mtpFloorDisableObserve(&drafted, &accepted, &idx, 2, 4, 0),
    );
    try testing.expectEqual(@as(u32, 0), idx);

    // It takes another complete run of depth-one failures to produce a rate.
    for (0..W - 1) |_| {
        try testing.expectEqual(
            @as(?f32, null),
            Generator.mtpFloorDisableObserve(&drafted, &accepted, &idx, 1, 1, 0),
        );
    }
    const rate = Generator.mtpFloorDisableObserve(&drafted, &accepted, &idx, 1, 1, 0);
    try testing.expectApproxEqAbs(@as(f32, 0.0), rate.?, 1e-5);
}

test "updateMtpEvRound: sticky disable uses the first draft at base depth one" {
    var good = mtpEvTestGenerator();
    for (0..Generator.MTP_DEPTH_WINDOW * 2) |_| good.updateMtpEvRound(8, 1);
    try testing.expect(!good.spec_disabled_runtime);
    for (good.mtp_window_drafted) |sample| try testing.expectEqual(@as(u8, 1), sample);
    for (good.mtp_window_accepted) |sample| try testing.expectEqual(@as(u8, 1), sample);

    var bad = mtpEvTestGenerator();
    for (0..Generator.MTP_DEPTH_WINDOW - 1) |_| {
        bad.updateMtpEvRound(8, 0);
        try testing.expect(!bad.spec_disabled_runtime);
    }
    bad.updateMtpEvRound(8, 0);
    try testing.expect(bad.spec_disabled_runtime);
}

test "updateMtpEvRound: warmup and wider base rounds reset floor evidence" {
    var g = mtpEvTestGenerator();
    g.mtp_ev_rounds = Generator.MTP_EV_WARMUP_ROUNDS - 1;
    g.mtp_window_idx = 7;
    g.updateMtpEvRound(4, 0);
    try testing.expectEqual(Generator.MTP_EV_WARMUP_ROUNDS, g.mtp_ev_rounds);
    try testing.expectEqual(@as(u32, 0), g.mtp_window_idx);

    for (0..Generator.MTP_DEPTH_WINDOW - 1) |_| g.updateMtpEvRound(1, 0);
    try testing.expect(!g.spec_disabled_runtime);
    try testing.expectEqual(Generator.MTP_DEPTH_WINDOW - 1, g.mtp_window_idx);

    g.mtp_ev_m_lo_prev = 2;
    g.updateMtpEvRound(4, 0);
    try testing.expectEqual(@as(u32, 0), g.mtp_window_idx);
    try testing.expect(!g.spec_disabled_runtime);
}

test "mtpEvExpectedTokens: 1 + sum of acceptance chain products" {
    const a = [_]f32{ 0.5, 0.5, 0.5 };
    try testing.expectApproxEqAbs(@as(f32, 1.5), Generator.mtpEvExpectedTokens(&a, 1), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.75), Generator.mtpEvExpectedTokens(&a, 2), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.875), Generator.mtpEvExpectedTokens(&a, 3), 1e-5);
    // Zero acceptance: every round still commits exactly the t1 bonus token.
    const z = [_]f32{ 0.0, 0.0 };
    try testing.expectApproxEqAbs(@as(f32, 1.0), Generator.mtpEvExpectedTokens(&z, 2), 1e-5);
}

test "mtpEvRoundCost: piecewise marginals (flat verify region, then the GDN width ramp)" {
    const costs = Generator.MtpEvCosts{ .draft = 0.10, .per_pos_lo = 0.09, .per_pos_hi = 0.22, .flat_max = 3, .sync = 0.02 };
    try testing.expectApproxEqAbs(@as(f32, 1.19), Generator.mtpEvRoundCost(costs, 1, false), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.57), Generator.mtpEvRoundCost(costs, 3, false), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.59), Generator.mtpEvRoundCost(costs, 3, true), 1e-5);
    // Positions 4+ pay the ramp: +0.32 each instead of +0.19.
    try testing.expectApproxEqAbs(@as(f32, 2.21), Generator.mtpEvRoundCost(costs, 5, false), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.19), Generator.mtpEvMarginalCost(costs, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.32), Generator.mtpEvMarginalCost(costs, 4), 1e-6);
}

test "mtpEvCostsFor: G17 profiles are explicit and env tuning stays generic" {
    const generic = Generator.mtpEvCostsForProfile(.generic, null);
    try testing.expectEqual(Generator.MTP_EV_DEFAULT_COSTS, generic);

    const q8 = Generator.mtpEvCostsForProfile(.g17_nax_q8_gs32, null);
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_COSTS, q8);
    try testing.expectEqual(@as(u32, 7), q8.nax_from);

    const q4 = Generator.mtpEvCostsForProfile(.g17_nax_q4_gs32, null);
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_Q4_GS32_COSTS, q4);
    try testing.expectEqual(@as(u32, 7), q4.nax_from);

    // An explicit four-value override retains its historical meaning instead
    // of inheriting an implicit hardware-only third region, for every profile.
    for ([_]mtp_mod.MtpCostProfile{ .generic, .g17_nax_q8_gs32, .g17_nax_q4_gs32 }) |profile| {
        const tuned = Generator.mtpEvCostsForProfile(profile, "0.10, 0.11, 0.22, 0.03");
        try testing.expectApproxEqAbs(@as(f32, 0.10), tuned.draft, 1e-6);
        try testing.expectApproxEqAbs(@as(f32, 0.11), tuned.per_pos_lo, 1e-6);
        try testing.expectApproxEqAbs(@as(f32, 0.22), tuned.per_pos_hi, 1e-6);
        try testing.expectApproxEqAbs(@as(f32, 0.03), tuned.sync, 1e-6);
        try testing.expectEqual(@as(u32, 0), tuned.nax_from);
        try testing.expectApproxEqAbs(@as(f32, 0.0), tuned.per_pos_nax, 1e-6);
    }

    // Invalid overrides are atomic no-ops. In particular, an empty variable
    // must not combine cap 8 with the generic costs that previously starved it.
    const invalid = [_][]const u8{
        "",
        "0.10,0.11",
        "garbage,0.11,0.22,0.03",
        "nan,0.11,0.22,0.03",
        "0.10,0.11,0.22,0.03,0.04",
        "0.10,-0.11,0.22,0.03",
    };
    for (invalid) |raw| {
        try testing.expectEqual(Generator.MTP_EV_DEFAULT_COSTS, Generator.mtpEvCostsForProfile(.generic, raw));
        try testing.expectEqual(Generator.MTP_EV_G17_NAX_COSTS, Generator.mtpEvCostsForProfile(.g17_nax_q8_gs32, raw));
        try testing.expectEqual(Generator.MTP_EV_G17_NAX_Q4_GS32_COSTS, Generator.mtpEvCostsForProfile(.g17_nax_q4_gs32, raw));
    }

    try testing.expectEqual(Generator.MTP_EV_DEFAULT_COSTS, Generator.mtpEvCostsFor(false, null));
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_COSTS, Generator.mtpEvCostsFor(true, null));
}

test "MTP_EV_G17_NAX_COSTS reproduces the measured M5 depth-6/depth-8 ratio" {
    const costs = Generator.MTP_EV_G17_NAX_COSTS;
    const t6 = Generator.mtpEvRoundCost(costs, 6, false);
    const t8 = Generator.mtpEvRoundCost(costs, 8, false);
    try testing.expectApproxEqAbs(@as(f32, 1.99), t6, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.19), t8, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 68.39 / 62.15), t8 / t6, 2e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.21), Generator.mtpEvMarginalCost(costs, 6), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.10), Generator.mtpEvMarginalCost(costs, 7), 1e-6);
}

test "MTP_EV_G17_NAX_Q4_GS32_COSTS encodes calibrated composite marginals" {
    const costs = Generator.MTP_EV_G17_NAX_Q4_GS32_COSTS;
    try testing.expect(costs.draft > 0.0);
    try testing.expect(costs.per_pos_lo > 0.0);
    try testing.expect(costs.per_pos_hi > 0.0);
    try testing.expect(costs.per_pos_nax > 0.0);
    try testing.expectApproxEqAbs(@as(f32, 0.11), Generator.mtpEvMarginalCost(costs, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.20), Generator.mtpEvMarginalCost(costs, 4), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.05), Generator.mtpEvMarginalCost(costs, 7), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.93), Generator.mtpEvRoundCost(costs, 6, false), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.03), Generator.mtpEvRoundCost(costs, 8, false), 1e-5);
}

test "mtpEvPlanFor: M5 NAX surfaces open depth 8 from realistic warmup EMAs" {
    const p = Generator.MTP_EV_PRIOR;
    const a = [_]f32{ 0.97, 0.89, p, p, p, p, p, p };

    // The M1-M4 surface stops at the first expensive ramp position.
    const generic = Generator.mtpEvPlanFor(&a, 8, Generator.MTP_EV_DEFAULT_COSTS, 3);
    try testing.expectEqual(@as(u32, 3), generic.m_lo);
    try testing.expect(generic.m_hi < 8);

    // The M5 fit captures both its cheaper intermediate widths and the NAX
    // takeover at draft position 7, so the same evidence reaches depth 8.
    for ([_]Generator.MtpEvCosts{ Generator.MTP_EV_G17_NAX_COSTS, Generator.MTP_EV_G17_NAX_Q4_GS32_COSTS }) |costs| {
        const nax = Generator.mtpEvPlanFor(&a, 8, costs, 3);
        try testing.expectEqual(@as(u32, 3), nax.m_lo);
        try testing.expectEqual(@as(u32, 8), nax.m_hi);
        try testing.expect(nax.tau_ln < 0.0);
    }

    const cold = [_]f32{ 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };
    for ([_]Generator.MtpEvCosts{ Generator.MTP_EV_G17_NAX_COSTS, Generator.MTP_EV_G17_NAX_Q4_GS32_COSTS }) |costs| {
        const cold_plan = Generator.mtpEvPlanFor(&cold, 8, costs, 8);
        try testing.expectEqual(@as(u32, 1), cold_plan.m_lo);
        try testing.expectEqual(@as(u32, 1), cold_plan.m_hi);
    }
}

test "mtpEvPlanFor: mid-decay acceptance picks a shallow base and a confidence-gated extension" {
    const costs = Generator.MtpEvCosts{ .draft = 0.10, .per_pos_lo = 0.09, .per_pos_hi = 0.22, .flat_max = 3, .sync = 0.02 };
    // Conditional acceptance decays: unconditional EV peaks at m=2, but the
    // marginal chain CONDITIONAL on chunk A landing stays profitable through
    // the flat verify region — the "draft deeper on easy stretches" shape.
    const a = [_]f32{ 0.7, 0.6, 0.55, 0.5, 0.45, 0.42, 0.4, 0.38 };
    const plan = Generator.mtpEvPlanFor(&a, 8, costs, 8);
    try testing.expectEqual(@as(u32, 2), plan.m_lo);
    try testing.expectEqual(@as(u32, 3), plan.m_hi);
    // tau = r(m_lo)*t_ext/S = 1.5362*0.19/0.55 = 0.5307 -> ln = -0.6335.
    try testing.expectApproxEqAbs(@as(f32, -0.6335), plan.tau_ln, 5e-3);
}

test "mtpEvPlanFor: hot flat acceptance rides the flat region and extends into the ramp on confidence" {
    const costs = Generator.MtpEvCosts{ .draft = 0.10, .per_pos_lo = 0.09, .per_pos_hi = 0.22, .flat_max = 3, .sync = 0.02 };
    const a = [_]f32{ 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9 };
    const plan = Generator.mtpEvPlanFor(&a, 8, costs, 8);
    // Static optimum m=3 (r=2.1904); ramp positions 4..6 pay only under full
    // confidence (0.9^k chain vs r*0.32 = 0.70 threshold).
    try testing.expectEqual(@as(u32, 3), plan.m_lo);
    try testing.expectEqual(@as(u32, 6), plan.m_hi);
    // tau = 2.1904*0.96/2.439 = 0.8621 -> ln = -0.1484.
    try testing.expectApproxEqAbs(@as(f32, -0.1484), plan.tau_ln, 5e-3);
}

test "mtpEvPlanFor: cold acceptance collapses to depth 1, single chunk" {
    const costs = Generator.MtpEvCosts{ .draft = 0.10, .per_pos_lo = 0.09, .per_pos_hi = 0.22, .flat_max = 3, .sync = 0.02 };
    const a = [_]f32{ 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };
    const plan = Generator.mtpEvPlanFor(&a, 8, costs, 8);
    try testing.expectEqual(@as(u32, 1), plan.m_lo);
    try testing.expectEqual(@as(u32, 1), plan.m_hi);
}

test "mtpEvPlanFor: m_lo_max damps the climb without killing the extension horizon" {
    const costs = Generator.MtpEvCosts{ .draft = 0.10, .per_pos_lo = 0.09, .per_pos_hi = 0.22, .flat_max = 3, .sync = 0.02 };
    const a = [_]f32{ 0.7, 0.6, 0.55, 0.5, 0.45, 0.42, 0.4, 0.38 };
    // Same EMAs as the mid-decay case, but the controller may only raise the
    // base one step (hysteresis): m_lo caps at 1 while m_hi stays deeper.
    const plan = Generator.mtpEvPlanFor(&a, 8, costs, 1);
    try testing.expectEqual(@as(u32, 1), plan.m_lo);
    try testing.expectEqual(@as(u32, 3), plan.m_hi);
    try testing.expect(plan.tau_ln < 0.0);
}

test "mtpEvPlanFor: unobserved deep indices at the prior still open the extension horizon (exploration)" {
    const costs = Generator.MtpEvCosts{ .draft = 0.10, .per_pos_lo = 0.09, .per_pos_hi = 0.22, .flat_max = 3, .sync = 0.02 };
    // The echo shape after warmup: shallow indices observed hot, deep indices
    // never reached (still at MTP_EV_PRIOR). The horizon must open past m_lo
    // so extension can get its first trial — this is the live ext_rounds=0
    // regression (a prior at the measured average sat razor-under the ramp
    // break-even of ~0.78 and extension never fired on pure echo).
    const p = Generator.MTP_EV_PRIOR;
    const a = [_]f32{ 0.97, 0.97, p, p, p, p, p };
    const plan = Generator.mtpEvPlanFor(&a, 7, costs, 8);
    try testing.expectEqual(@as(u32, 3), plan.m_lo);
    try testing.expect(plan.m_hi > plan.m_lo);
    // tau = r(3)*0.32/0.85 = 0.8898 -> ln = -0.1168 (under the 0.95 clamp).
    try testing.expectApproxEqAbs(@as(f32, -0.1168), plan.tau_ln, 5e-3);
}

test "mtpEvPlanFor: DEFAULT costs carry the post-verify-qmm surface (2026-07-13 refit #2)" {
    // Pins MTP_EV_DEFAULT_COSTS to the surface measured AFTER the split-K
    // verify-qmm kernel (same-session saturated sweep, Youssofal 27B,
    // M4 Max: T(1)=44.6, T(3)=54.4, T(6)=89.9 ms → floor ≈ 40 ms,
    // draft+lo = 0.12, draft+hi = 0.30). Flat-region positions are now
    // nearly free, so the base holds 3 deep into marginal acceptance where
    // the pre-kernel surface demoted to 2.
    const costs = Generator.MTP_EV_DEFAULT_COSTS;
    // Hot uniform 90%: base rides the flat region, extension into the ramp.
    const hot = [_]f32{ 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9 };
    const hot_plan = Generator.mtpEvPlanFor(&hot, 8, costs, 8);
    try testing.expectEqual(@as(u32, 3), hot_plan.m_lo);
    try testing.expectEqual(@as(u32, 5), hot_plan.m_hi);
    try testing.expectApproxEqAbs(@as(f32, -0.1196), hot_plan.tau_ln, 5e-3);
    // Marginal 75%: base stays 3 on the cheap flat region, short extension.
    const mid = [_]f32{ 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75 };
    const mid_plan = Generator.mtpEvPlanFor(&mid, 8, costs, 8);
    try testing.expectEqual(@as(u32, 3), mid_plan.m_lo);
    try testing.expectEqual(@as(u32, 4), mid_plan.m_hi);
    try testing.expectApproxEqAbs(@as(f32, -0.2179), mid_plan.tau_ln, 5e-3);
}

test "mtpEvPlanFor: cap 1 is a plain depth-1 round" {
    const costs = Generator.MTP_EV_DEFAULT_COSTS;
    const a = [_]f32{ 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9 };
    const plan = Generator.mtpEvPlanFor(&a, 1, costs, 8);
    try testing.expectEqual(@as(u32, 1), plan.m_lo);
    try testing.expectEqual(@as(u32, 1), plan.m_hi);
}

test "mtpRoundOff0: a pending history stash overrides the stale cache length" {
    // No stash: the head cache is fully committed — its step IS the origin.
    try testing.expectEqual(@as(usize, 42), Generator.mtpRoundOff0(null, 42));
    // Pending stash: the cache still carries the previous round's draft tail
    // (step is stale/uncommitted), so the origin is where the stash's entries
    // will END once the consume-time truncate + merged forward run. A round
    // that read mc.step here would draft at the WRONG RoPE offsets.
    const st = Generator.MtpHistStash{
        .ids = .{ .ctx = null },
        .hidden = .{ .ctx = null },
        .n = 3, // t1 + 2 accepted drafts
        .off0 = 40,
    };
    // cache_step (46 = 40 committed + 6 stale draft entries) must be ignored.
    try testing.expectEqual(@as(usize, 43), Generator.mtpRoundOff0(st, 46));
}

test "mtpEvObserve: conditional EMA updates hit accepted indices, the reject index, and nothing past it" {
    var a = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    // 3 drafted, 1 accepted: index 0 saw a success, index 1 saw the reject,
    // index 2 was never conditionally reached (no observation).
    Generator.mtpEvObserve(&a, 3, 1, 0.15);
    try testing.expectApproxEqAbs(@as(f32, 0.575), a[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.425), a[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), a[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), a[3], 1e-5);
    // Full accept: every drafted index saw a success, none saw a reject.
    var b = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    Generator.mtpEvObserve(&b, 2, 2, 0.15);
    try testing.expectApproxEqAbs(@as(f32, 0.575), b[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.575), b[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), b[2], 1e-5);
}

test "mtpChainLogConf: sums clamped log-confidences; NaN can never pass a gate" {
    try testing.expectApproxEqAbs(@as(f32, -0.3), Generator.mtpChainLogConf(&[_]f32{ -0.1, -0.2 }), 1e-5);
    // Positive numeric noise clamps to 0 (a log-prob is never > 0).
    try testing.expectApproxEqAbs(@as(f32, -0.1), Generator.mtpChainLogConf(&[_]f32{ 0.05, -0.1 }), 1e-5);
    // NaN -> -inf: `chain >= tau_ln` is false for every finite tau.
    const nan_chain = Generator.mtpChainLogConf(&[_]f32{ -0.1, std.math.nan(f32) });
    try testing.expect(nan_chain == -std.math.inf(f32));
    try testing.expect(!(nan_chain >= @log(@as(f32, 0.05))));
}

test "MtpTrace: per-phase accumulation, round averaging, log cadence, reset" {
    var t = Generator.MtpTrace{};
    // 2 rounds: draft 2ms+4ms, eval 10ms+30ms.
    t.add(.draft, 2_000_000);
    t.add(.eval, 10_000_000);
    try testing.expect(!t.endRound(3, 2, false));
    t.add(.draft, 4_000_000);
    t.add(.eval, 30_000_000);
    try testing.expect(!t.endRound(7, 6, true));
    try testing.expectApproxEqAbs(@as(f64, 3.0), t.avgMs(.draft), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 20.0), t.avgMs(.eval), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.avgMs(.sync), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 23.0), t.totalAvgMs(), 1e-9);
    try testing.expectEqual(@as(u64, 10), t.drafted);
    try testing.expectEqual(@as(u64, 8), t.accepted);
    try testing.expectEqual(@as(u32, 1), t.extended);
    // Log line falls due exactly at LOG_EVERY rounds.
    var i: u32 = 2;
    while (i < Generator.MtpTrace.LOG_EVERY - 1) : (i += 1) {
        try testing.expect(!t.endRound(1, 1, false));
    }
    try testing.expect(t.endRound(1, 1, false));
    t.reset();
    try testing.expectEqual(@as(u32, 0), t.rounds);
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.avgMs(.draft), 1e-9);
}

test "buildPaddedBatch pads to max length with zeros and records lengths" {
    const seqs = [_][]const u32{
        &[_]u32{ 101, 7592, 102 },
        &[_]u32{ 101, 102 },
    };
    var pb = try buildPaddedBatch(testing.allocator, &seqs);
    defer pb.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 3), pb.max_len);
    try testing.expectEqual(@as(usize, 2), pb.lengths.len);
    try testing.expectEqual(@as(usize, 3), pb.lengths[0]);
    try testing.expectEqual(@as(usize, 2), pb.lengths[1]);
    const expected = [_]i32{ 101, 7592, 102, 101, 102, 0 };
    try testing.expectEqualSlices(i32, &expected, pb.ids);
}

test "buildKeyPadMask is additive zero on real keys, -inf on padding" {
    const s = mlx.gpuStream();
    const lengths = [_]usize{ 3, 1 };
    const mask = try buildKeyPadMask(testing.allocator, &lengths, 3, s);
    defer _ = mlx.mlx_array_free(mask);
    try testing.expectEqualSlices(c_int, &[_]c_int{ 2, 1, 1, 3 }, mlx.getShape(mask));
    var f32_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32_mask);
    try mlx.check(mlx.mlx_astype(&f32_mask, mask, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f32_mask));
    const data = mlx.mlx_array_data_float32(f32_mask).?;
    // Batch row 0: all three keys real.
    try testing.expectEqual(@as(f32, 0), data[0]);
    try testing.expectEqual(@as(f32, 0), data[1]);
    try testing.expectEqual(@as(f32, 0), data[2]);
    // Batch row 1: one real key, two padded.
    try testing.expectEqual(@as(f32, 0), data[3]);
    try testing.expect(std.math.isInf(data[4]) and data[4] < 0);
    try testing.expect(std.math.isInf(data[5]) and data[5] < 0);
}

test "maskedMeanPoolNormalize excludes padded positions and unit-normalizes" {
    const s = mlx.gpuStream();
    // hidden [2, 3, 2]; row 0 has 2 real positions (pad slot holds garbage
    // that must not leak into the pool), row 1 has 3.
    const data = [_]f32{
        1, 0, 3, 4, 100, 100,
        0, 2, 0, 4, 0,   6,
    };
    const shape = [_]c_int{ 2, 3, 2 };
    const hidden = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(hidden);
    const lengths = [_]usize{ 2, 3 };
    const rows = try maskedMeanPoolNormalize(testing.allocator, hidden, &lengths, s);
    defer {
        for (rows) |r| testing.allocator.free(r);
        testing.allocator.free(rows);
    }
    // Row 0: mean of (1,0),(3,4) = (2,2) → L2-normalized (1/√2, 1/√2).
    try testing.expectApproxEqAbs(@as(f32, 0.70710678), rows[0][0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.70710678), rows[0][1], 1e-4);
    // Row 1: mean of (0,2),(0,4),(0,6) = (0,4) → normalized (0, 1).
    try testing.expectApproxEqAbs(@as(f32, 0.0), rows[1][0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 1.0), rows[1][1], 1e-4);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;
