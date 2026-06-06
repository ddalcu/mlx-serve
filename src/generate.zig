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

/// Module-level overrides for prefill behavior. Defaults match the original
/// hardcoded values; main.zig may overwrite these from CLI flags before
/// `serve()` runs. Per-request reads happen on the same thread that did the
/// CLI parse, so no atomicity needed.
pub var prefill_chunk_override: usize = 8192;
pub var prefill_trace_force: bool = false;

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
    timeout_ns: u64, // 0 = no timeout
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
        // CLI flag in main.zig). Env var wins if both are set.
        const PREFILL_CHUNK: usize = readEnvUsize("MLX_SERVE_PREFILL_CHUNK", prefill_chunk_override);
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
        const ssm_cp_stride: usize = if (want_ssm_cp) @intCast(options.ssm_checkpoint_stride) else 0;
        // Absolute KV position of `prompt_ids[0]`. Warm-path callers (the
        // scheduler after restoring a checkpoint) pass the matched prefix
        // length so the snapshots stamp positions valid in the full original
        // sequence, not relative offsets inside the tail-only prefill.
        const ssm_cp_offset: usize = options.ssm_checkpoint_pos_offset;

        if (prompt_ids.len > 1) {
            const prefix_len = prompt_ids.len - 1;
            const has_vision = ctx.vision_embeddings != null;
            const default_chunk = if (has_vision) prefix_len else PREFILL_CHUNK;

            var pos: usize = 0;
            while (pos < prefix_len) {
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
                var end = @min(pos + default_chunk, prefix_len);
                if (want_ssm_cp) {
                    const abs_pos = pos + ssm_cp_offset;
                    const abs_end = end + ssm_cp_offset;
                    const next_boundary_abs = ((abs_pos / ssm_cp_stride) + 1) * ssm_cp_stride;
                    if (next_boundary_abs > abs_pos and next_boundary_abs < abs_end) {
                        end = next_boundary_abs - ssm_cp_offset;
                    }
                }
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
                const chunk_logits = if (xfm.compiled_forward != null and
                    ctx.cache == &xfm.cache and
                    ssm_match and
                    ctx.capture_hidden == null and
                    ctx.vision_embeddings == null)
                    try xfm.forwardCompiled(chunk_input)
                else
                    try xfm.forwardWith(&ctx, chunk_input);
                _ = mlx.mlx_array_free(chunk_logits);
                if (trace_enabled) chunked_ns += prefill_sw.read() - chunk_start_ns;

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
                    const cp = try captureSsmCheckpoint(allocator, ctx.ssm_entries.?, abs_end_for_cp2);
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
                    const cp = try captureSsmCheckpoint(allocator, ctx.ssm_entries.?, final_abs);
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
        const need_capture = drafter_active;
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

        // Drafter / PLD-v2 path: sample synchronously and DO NOT pre-forward
        // the sampled token. The first nextDrafter / nextPld call needs the
        // cache at exactly prompt_len (last prompt token forwarded; first
        // sampled token deferred). The lazy pre-forward path below would
        // over-advance the cache and corrupt every verify forward.
        if (drafter_active or pld_active) {
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
                .last_hidden = if (need_capture) captured_hidden else mlx.mlx_array_new(),
                .has_last_hidden = need_capture,
                .prng = std.Random.DefaultPrng.init(sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds())),
                .prompt_ids_owned = prompt_owned,
                .prompt_ids_alloc = allocator,
                .drafter = if (drafter_active) options.drafter else null,
                .drafter_block_size = options.drafter_block_size,
            };
            // pending_logits/pending_token left empty — the lazy pipeline is
            // skipped under PLD / drafter. The speculative `next*` paths drive
            // every subsequent step with predictable cache offset.
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

        const verify_logits = try xfm.forwardWith(&self.ctx, verify_input);
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

        // Runtime acceptance gate: after warmup, if the per-draft acceptance
        // probability is below the threshold, disable speculation for the rest
        // of this request. Sticky for the rest of the generation. PLD's
        // `drafts_per_round` is the upper-bound draft length (`max_draft`);
        // matches with shorter accepts still divide by this max so a workload
        // with consistently-short n-gram matches DOES get throttled.
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
        if (self.timeout_ns > 0 and self.timer.read() >= self.timeout_ns) {
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
        if (self.timeout_ns > 0 and self.timer.read() >= self.timeout_ns) {
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
    const s = xfm.s;
    const prompt_len: c_int = @intCast(token_ids.len);
    const shape = [_]c_int{ 1, prompt_len };

    const ids_i32 = try allocator.alloc(i32, token_ids.len);
    defer allocator.free(ids_i32);
    for (token_ids, 0..) |id, i| {
        ids_i32[i] = @intCast(id);
    }

    const input = mlx.mlx_array_new_data(ids_i32.ptr, &shape, 2, .int32);
    defer _ = mlx.mlx_array_free(input);

    // Get hidden states: [1, seq_len, hidden_size]
    const hidden = try xfm.forwardEmbedding(input);
    defer _ = mlx.mlx_array_free(hidden);

    // Mean pool over sequence dimension (axis=1): [1, hidden_size]
    var pooled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_mean_axis(&pooled, hidden, 1, false, s));

    // L2 normalize: pooled / sqrt(sum(pooled^2))
    var squared = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(squared);
    try mlx.check(mlx.mlx_multiply(&squared, pooled, pooled, s));

    var sum_sq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum_sq);
    try mlx.check(mlx.mlx_sum_axis(&sum_sq, squared, -1, true, s));

    var norm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm);
    try mlx.check(mlx.mlx_sqrt(&norm, sum_sq, s));

    // Add epsilon to avoid division by zero
    const eps = mlx.mlx_array_new_float(1e-12);
    defer _ = mlx.mlx_array_free(eps);
    var norm_safe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm_safe);
    try mlx.check(mlx.mlx_maximum(&norm_safe, norm, eps, s));

    var normalized = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(normalized);
    try mlx.check(mlx.mlx_divide(&normalized, pooled, norm_safe, s));

    // Eval and extract float data
    try mlx.check(mlx.mlx_array_eval(normalized));
    const n_shape = mlx.getShape(normalized);
    const dim: usize = @intCast(n_shape[n_shape.len - 1]);
    const data_ptr = mlx.mlx_array_data_float32(normalized) orelse return error.MlxError;

    const result = try allocator.alloc(f32, dim);
    @memcpy(result, data_ptr[0..dim]);
    return result;
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

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

