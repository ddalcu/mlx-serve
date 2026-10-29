//! The Generator: per-request decode state (fields), init/deinit and the step mover; its methods live
//! in the generate/, spec/, grammar/ and sampling/ files and are aliased here.
const depth_bounds = @import("../spec/mtp_depth_bounds.zig");
const plugin_settle = @import("../plugin/settle.zig");
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const keyed_sample = @import("../sampling/keyed_sample.zig");
const transformer_mod = @import("../transformer.zig");
const dsv4_mod = @import("../../../plugins/deepseek_v4/deepseek_v4.zig");
const dsv41_mod = @import("../../../plugins/deepseek_v41/deepseek_v41.zig");
const mlx_stream = @import("../modules/stream.zig");
const tokenizer_mod = @import("../../../sdk/tokenizer/tokenizer.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const log = @import("../../../sdk/util/log.zig");
const json_grammar = @import("../../../sdk/grammar/json_grammar.zig");
const json_schema = @import("../../../sdk/grammar/json_schema.zig");
const token_mask = @import("../../../sdk/grammar/token_mask.zig");
const rp_mod = @import("../../../sdk/grammar/protocol.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const pld_index = @import("../../../sdk/spec/pld_index.zig");
const mtp_lookup = @import("../spec/mtp_lookup.zig");
const drafter_mod = @import("../../../plugins/gemma4/assistant_drafter.zig");
const mtp_mod = @import("../../../sdk/spec/mtp/mtp.zig");
const mtp_acceptance = @import("../../../sdk/spec/mtp_acceptance.zig");
const round_cost = @import("../spec/round_cost.zig");
const dflash_policy = @import("../spec/dflash/policy.zig");
const group_cost = @import("../spec/mtp_group_cost.zig");
const group_planner = @import("../spec/mtp_group_planner.zig");
const ane_mod = @import("../../../sdk/ane/offload.zig");
const scheduler_mod = @import("../scheduler.zig");

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
const dflash_mod = @import("../spec/dflash/dflash.zig");
const lilicorr_mod = @import("../spec/dflash/lilicorr.zig");
const DflashModel = dflash_mod.DflashModel;
const gdn_decode = @import("../../../sdk/blocks/gdn/decode.zig");
const KVCache = transformer_mod.KVCache;
const testing = std.testing;
const CACHE_CLEAR_INTERVAL = @import("prefill_chunk.zig").CACHE_CLEAR_INTERVAL;
const DflashPol = @import("../spec/dflash.zig").DflashPol;
const LogprobResult = @import("../sampling/logprobs.zig").LogprobResult;
const MtpCacheRef = @import("../spec/mtp_head_ref.zig").MtpCacheRef;
const MtpHeadRef = @import("../spec/mtp_head_ref.zig").MtpHeadRef;
const SamplingParams = @import("../sampling/params.zig").SamplingParams;
const SpecDisableReason = @import("prefill_chunk.zig").SpecDisableReason;
const SsmLag = @import("pipeline.zig").SsmLag;
const StallClock = @import("../../../sdk/request/stops.zig").StallClock;
const clearsPoolAtPrefillEnd = @import("prefill_chunk.zig").clearsPoolAtPrefillEnd;
const shouldClearAllocatorCache = @import("prefill_chunk.zig").shouldClearAllocatorCache;

pub const Generator = struct {
    pub const specTreeFor = @import("../spec/tree.zig").specTreeFor;
    pub const MtpGraphFn = @import("../spec/tree.zig").MtpGraphFn;
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
    /// `step` at the last `mlx_clear_cache()`. Advanced only by `advanceStep`,
    /// which is the ONE place `step` may move.
    last_cache_clear_step: u32 = 0,
    max_tokens: u32,
    sampling: SamplingParams,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: []const u8,
    done: bool,
    eos_token_ids: []const u32,
    generated_ids: std.ArrayList(u32),
    /// Degenerate-tail detection must not reconsider a reasoning loop after
    /// its boundary has activated the final-answer grammar. The scheduler
    /// scans only this suffix; a later loop in the answer is still handled by
    /// the ordinary repetition-stop policy.
    loop_guard_start: usize = 0,
    consecutive_pad: u32 = 0, // count of consecutive token-0 (pad) generations
    timeout_ns: u64, // 0 = no timeout; measures SILENCE, not total time (see StallClock)
    stall: StallClock = .{},
    timer: io_util.Stopwatch,
    logprobs_n: u32 = 0, // 0 = disabled, >0 = number of top_logprobs to return
    last_logprob: ?LogprobResult = null, // logprob result for the most recently returned token
    /// Logprobs for the token the NEXT `next()` call will return.
    ///
    /// The decode loop returns `next_token_id` and, in the same call, forwards
    /// it to sample its successor — so the result `sampleToken` hands back
    /// describes the token AFTER the one being returned. Publishing it as
    /// `last_logprob` there shifted the whole array by one: the caller zips
    /// token_ids with logprobs by index, so at temp 0 every entry reported the
    /// distribution of the token that FOLLOWED it (a one-token "OK" reply came
    /// back with `<|role_end|>` at rank 1). Seeded at init with t1's own
    /// distribution — t1 is sampled from the prefill's final forward, which
    /// this loop never sees.
    pending_logprob: ?LogprobResult = null,
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
    /// A spoken reply's state (`speech.zig`), owned with the generator's allocator.
    speech: ?*@import("speech.zig").State = null,
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
    /// Did init actually arm PLD for this generator (`InitOptions.pld_enabled`
    /// AFTER the deepseek_v4 chokepoint guard)? `nextPld` declines to the
    /// plain serial step when false, and the scheduler's tick dispatch
    /// (`specTickMode`) requires it alongside `slot.enable_pld` — the caller's
    /// flag alone must never put a verify forward through the trunk. This is
    /// PLD's counterpart of the `gen.mtp != null` / `gen.drafter != null`
    /// conjuncts; PLD has no model handle, so the bit has to be explicit.
    pld_enabled: bool = false,
    /// Stats for PLD benchmark logging. `pld_attempted` counts every step
    /// where lookup found a candidate (so a verify forward ran);
    /// `pld_accepted_tokens` is the cumulative number of *drafted* tokens
    /// (not including the always-accepted t1) that were successfully verified.
    pld_attempted: u64 = 0,
    pld_accepted_tokens: u64 = 0,
    /// DeepSeek-V4 DSpark: init armed the native block-parallel draft mode
    /// (dsv4 checkpoint shipping mtp.* stages + a clean request).
    /// Mutually exclusive with pld/drafter/mtp by the chokepoint's
    /// construction; `nextDspark` declines to the serial step when false.
    dspark_enabled: bool = false,
    /// Sampled-request acceptance (the MTP one-hot Leviathan rule over
    /// filtered target probs) instead of raw argmax equality. Set by the
    /// chokepoint when the request samples (temp ≥ 0.01, top_k ≠ 1) and the
    /// stochastic arm isn't env-killed; meaningless unless `dspark_enabled`.
    dspark_stochastic: bool = false,
    dspark_attempted: u64 = 0,
    dspark_accepted_tokens: u64 = 0,
    /// Rounds whose drafts were copied from the context (`dsparkCopyStep`), included in `dspark_attempted`.
    dspark_copies: u64 = 0,
    dspark_copy_drafted: u64 = 0,
    dspark_copy_accepted: u64 = 0,
    /// DeepSeek-V4.1's stage rounds under `dflash_policy` (verify rows by the confidence head, plain ticks
    /// when drafting does not pay); null = every round verifies the whole block.
    dspark_pol: ?DflashPol = null,
    dspark_stats: dflash_policy.PositionStats = .{},
    dspark_plain: u64 = 0,

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

    // ── DFlash block-drafter state ──
    // External block-parallel assistant (src/engines/mlx/spec/dflash/dflash.zig). When
    // `dflash != null`, callers use `nextDflash` instead of `next`. Weights
    // are owned by the server (per-model); the per-request context cache
    // (`dflash_ctx`) is OWNED by the Generator — built during prefill from
    // the trunk's capture_layers hiddens, freed in `deinit`.
    dflash: ?*DflashModel = null,
    dflash_ctx: ?dflash_mod.DflashCtx = null,
    /// Effective block size (assistant config, clamped by --draft-block-size).
    /// Drafts per round = dflash_block_size - 1.
    dflash_block_size: u32 = 0,
    /// Request-class break-even yield, normalized by the scheduler to the
    /// effective draft width. Thinking requests use the lower calibration
    /// because their less-predictable reasoning preamble pays back later.
    dflash_min_accepted_per_round: f32 = DFLASH_GATE_MIN_ACCEPTED_PER_ROUND,
    /// Drafts this round ran (0 = serial), read by the table feed.
    dflash_round_width: u32 = 0,
    /// Stats: count of nextDflash calls that ran a verify forward.
    dflash_attempted: u64 = 0,
    /// Round policy (`dflash_policy.zig`); null = the fixed block and the sticky yield gate.
    dflash_pol: ?DflashPol = null,
    /// Stats: rounds drafted from the context (no assistant forward), the tokens they proposed and the ones kept.
    dflash_copies: u64 = 0,
    dflash_copy_drafted: u64 = 0,
    dflash_copy_accepted: u64 = 0,
    /// Stats: rounds that verified one row (no drafts), whether or not the assistant drafted for them.
    dflash_plain: u64 = 0,
    /// The round's wall time bills a draft nobody verified, so it says nothing about the width's cost.
    dflash_unobserved: bool = false,
    /// Per-position acceptance and verify widths (`[spec-stats]`).
    dflash_stats: dflash_policy.PositionStats = .{},
    /// Plain-round captures not yet appended to the assistant context: one array per target layer
    /// per round, oldest first. One append later costs what one token's does, so a plain stretch pays none.
    dflash_pending: std.ArrayList(mlx.mlx_array) = .empty,
    dflash_pending_rounds: u32 = 0,
    /// Positions those parked captures cover (a forced commit parks several rows at once).
    dflash_pending_rows: u32 = 0,
    /// Tree rounds that verified a context copy: consecutive ones that kept
    /// none of it, and rounds left without copies after four of those.
    dflash_copy_misses: u8 = 0,
    dflash_copy_silent: u8 = 0,
    /// Emitted tokens past the KV cache: the last one, when the budget ended on
    /// a token the previous verify already decided. Commits key on the rest.
    unforwarded_tail: u32 = 0,
    /// A plain pipelined step forwards the stop token and one lookahead before it sees the stop,
    /// and recurrent state cannot be rewound: the state at the last two such steps, shared by
    /// refcount, so a finishing slot rolls back to what it commits (`rollbackSsmTo`).
    ssm_lag: [2]SsmLag = .{ .{}, .{} },
    /// Stats: cumulative draft tokens accepted (excluding always-accepted t1).
    dflash_accepted_tokens: u64 = 0,
    /// Per-phase wall-time trace (diag dflash-trace; else untouched).
    /// Unlike the MTP trace this one INSERTS eval barriers to attribute a
    /// fully-lazy round — a traced round is slower than a real one by
    /// whatever overlap the barriers destroy. Diagnostic only.
    dflash_trace: DflashTrace = .{},
    /// Trace-only stopwatch across the scheduler gap (round return → next
    /// round entry); bills the `.gap` phase. Null when not tracing.
    dflash_gap_watch: ?io_util.Stopwatch = null,

    // ── Qwen native MTP head state ──
    // The model's own one-layer multi-token-prediction head (src/sdk/spec/mtp/mtp.zig).
    // When `mtp != null`, callers use `nextMtp` instead of `next`. The head
    // is owned by the server (loaded with the model); the Generator only
    // holds a non-owning pointer. `mtp_cache` is the head's committed-history
    // KV cache — OWNED by the Generator (built during prefill, freed in
    // `deinit`).
    mtp: ?MtpHeadRef = null,
    /// Installed verifier route; selected once at init, not checked per token.
    mtp_accept_graph: MtpGraphFn = mtpBatchedExactGraph,
    mtp_accept_prefix: mtp_acceptance.PrefixFn = mtp_acceptance.exactPrefix,
    mtp_accept_param: f32 = 0,
    /// A sampled request drafts only depth 0 from the draft sampler, every later depth by argmax.
    mtp_greedy_tail: bool = false,
    /// A row-exact trunk verifies sampled rounds with the serial sampler itself:
    /// row i draws with the key serial decoding uses at that position and a
    /// draft stays only when it is that draw, so the stream is serial's.
    mtp_serial_accept: bool = false,
    /// This request's kept / tried first drafts (`mtpExactDepth`).
    mtp_kept: u32 = 0,
    mtp_tried: u32 = 0,
    mtp_cycle_watch: ?io_util.Stopwatch = null,
    /// Does the model have a usable MTP head? `--no-mtp` clears it; a per-request `enable_mtp:false` does not.
    model_has_mtp: bool = false,
    mtp_cache: ?MtpCacheRef = null,
    /// Absolute target position represented by MTP-cache position 0. Usually
    /// zero; nonzero when the head keeps only a suffix of a restored/long
    /// prompt. Used to map the sidecar's relative offsets into Qwen M-RoPE.
    mtp_position_base: usize = 0,
    /// CONFIGURED max tokens drafted per round (verify length = depth + 1).
    mtp_depth: u32 = mtp_mod.DEFAULT_DEPTH,
    /// The cap WITHOUT the per-silicon row: an explicit --mtp-max-depth, else
    /// the adaptive default. The row is the cold-start cap; once the table
    /// has trusted widths it may plan up to this instead (the M4 base row
    /// of 4 measured -6% against what the table found at 6).
    mtp_depth_free: u32 = mtp_mod.DEFAULT_DEPTH,
    /// CURRENT adaptive depth (see updateMtpDepth). Starts at `mtp_depth`,
    /// demoted/promoted per windowed acceptance, never exceeds `mtp_depth`.
    mtp_depth_current: u32 = mtp_mod.DEFAULT_DEPTH,
    /// Stats: count of nextMtp calls that ran a verify forward.
    mtp_attempted: u64 = 0,
    /// Stats: cumulative draft tokens accepted (excluding always-accepted t1).
    mtp_accepted_tokens: u64 = 0,
    /// Adaptive-depth moving window: per-round drafted/accepted counts.
    mtp_window_drafted: [MTP_DEPTH_WINDOW]u8 = @splat(0),
    mtp_window_accepted: [MTP_DEPTH_WINDOW]u8 = @splat(0),
    mtp_window_idx: u32 = 0,
    mtp_rounds_since_switch: u32 = 0,
    /// Rounds remaining during which promotion is blocked (set after a
    /// demotion so a failed depth excursion isn't immediately retried).
    mtp_promote_cooldown: u32 = 0,
    /// Cumulative drafted tokens across rounds. The EV controller varies m
    /// per round, so `attempts x depth` no longer measures proposals — this
    /// is the honest per_draft_pct denominator.
    mtp_drafted_tokens: u64 = 0,
    /// Prompt-lookup rounds, kept apart from the MTP
    /// counters so the head's acceptance stats stay the head's.
    mtp_lookup_rounds: u64 = 0,
    mtp_lookup_drafted: u64 = 0,
    mtp_lookup_accepted: u64 = 0,
    /// Prompt + committed output, indexed up to `mtp_lookup_gen` generated tokens.
    mtp_lookup_idx: ?mtp_lookup.Index = null,
    mtp_lookup_gen: usize = 0,
    /// Landed drafts per round, smoothed, for lookup and MTP rounds (`mtp_lookup.gate`).
    mtp_lookup_ema: f32 = @floatFromInt(mtp_lookup.MAX_DRAFT),
    mtp_round_ema: f32 = 0,
    /// The previous round was a lookup round.
    mtp_after_lookup: bool = false,
    /// The last lookup round landed every draft (`mtp_lookup.gate` streak).
    mtp_lookup_streak: bool = false,
    /// Rounds where the confidence gate extended into chunk B.
    mtp_ext_rounds: u64 = 0,
    /// Speculative rounds that rolled recurrent state back on a partial accept.
    partial_rounds: u64 = 0,
    /// Extension dry-spell gate: consecutive extension-CONSIDERED rounds
    /// whose confidence gate did not clear, and the single-chunk cooldown
    /// that a full dry streak triggers (see mtpExtDryAllows).
    mtp_ext_dry_streak: u32 = 0,
    mtp_ext_cooldown: u32 = 0,
    /// EV controller: conditional acceptance EMA per draft index,
    /// a[i] = P(draft i accepted | drafts 0..i-1 accepted). Optimistic prior;
    /// warmup rounds pull the low indices to reality before it can matter.
    mtp_ev_accept: [mtp_mod.MAX_DEPTH]f32 = @splat(MTP_EV_PRIOR),
    /// Rounds seen by the EV controller (drives the legacy-behavior warmup).
    mtp_ev_rounds: u32 = 0,
    /// Last round's planned m_lo (base-depth climb damping: +1/round max).
    mtp_ev_m_lo_prev: u32 = 1,
    /// Round-cost surface selected once for this target+MTP head. The M5/G17
    /// NAX surface requires the measured trunk, native sidecar, and 3-bit
    /// draft-only head; every other combination keeps the M1-M4 surface.
    mtp_ev_costs: MtpEvCosts = MTP_EV_DEFAULT_COSTS,
    /// Live-cost EMAs (ms), 0 = unseeded. `mtp_ev_sync_ms` is the measured
    /// chunk-A confidence-read sync — updated ONLY on extension-considered
    /// rounds, so it holds the true cost of a sync when one happens (not the
    /// suppressed-round average). `mtp_ev_round_ms` is the realized round
    /// wall-clock. Their ratio drives the cost-aware exploration throttle
    /// (mtpExtDryThresholdFor) so a dry sync tax self-limits to a small
    /// fraction of the round. Always-on; the trace is not required.
    mtp_ev_sync_ms: f32 = 0,
    mtp_ev_round_ms: f32 = 0,
    mtp_regime: MtpRegime = .{},
    mtp_regime_verdict: ?bool = null,
    /// Width-trial schedule for the round-cost table (one 2-round block per
    /// period at m_lo+1, so the table learns the next cliff once).
    mtp_width_trial: MtpWidthTrial = .{},
    /// Consecutive plans with the same m_lo (the base has stopped climbing).
    mtp_m_lo_streak: u32 = 0,
    /// Drafts of the previous TWO rounds (0 = serial), every round: a round
    /// is a transition unless both predecessors ran its width (the first
    /// round after a width change is the transition, the second is still
    /// elevated — live, every w4 sample taken beside a trial read 17% high
    /// on the M4 base and fed the churn that produced it).
    spec_round_prev_width: ?u32 = null,
    spec_round_prev_width2: ?u32 = null,
    /// Shape of the previous two MTP rounds: a shape change is a transition too.
    spec_round_prev_two_chunk: bool = false,
    spec_round_prev_two_chunk2: bool = false,
    /// Wall clock between round ends: the regime compares the quantity
    /// tok/s reports, so per-round work OUTSIDE the round stopwatch (token
    /// publish, stop checks, scheduler) must be in the denominator — fewer
    /// rounds is part of a wider shape's win.
    mtp_regime_clock: ?io_util.Stopwatch = null,
    /// True while this generator is the ONLY decoding stream. Contention is
    /// strictly one-sided — it only ADDS time — so a busy server simply
    /// stops feeding the kv-term learner rather than teaching it a lie.
    /// Set per tick by the scheduler.
    spec_cost_solo: bool = true,
    /// Draft-depth cap while this slot verifies inside a batched group (0 = none):
    /// the group's rows must stay on the split-K verify lane (`scheduler.mtpSubGroupPlan`).
    mtp_group_cap: u32 = 0,
    /// Read with a non-zero `mtp_group_cap`: draft to the cap (`mtpGroupPlan`).
    mtp_group_fill: bool = false,
    mtp_planner_owned: bool = false,
    mtp_planner_pending: bool = false,
    mtp_planner_width: ?u8 = null,
    mtp_planner_probe: bool = false,
    mtp_planner_probes: u8 = 0,
    mtp_planner_recovery: group_planner.Recovery = .{},
    mtp_planner_recovering: bool = false,
    mtp_planner_history: group_planner.History = .{},
    mtp_hidden_stale: bool = false,
    mtp_planner_plain_ticks: u32 = 0,
    mtp_planner_prime_ticks: u32 = 0,
    mtp_planner_spec_rounds: u32 = 0,
    mtp_planner_max_gap_ms: f32 = 0,
    /// This slot's next `mtpRoundBegin` inits the chain and returns `.open` so the
    /// scheduler can run the head step for the group in one graph.
    mtp_batch_head: bool = false,
    /// Per-phase wall-time trace (diag mtp-trace; else untouched).
    mtp_trace: MtpTrace = .{},
    /// Trace-only: stopwatch running across the scheduler gap (round return
    /// → next round entry); bills the `.gap` phase. Null when not tracing.
    mtp_gap_watch: ?io_util.Stopwatch = null,
    /// Deferred committed-history append: the round's (tokens, true verify
    /// hiddens) pair, folded into the NEXT round's first draft step as one
    /// multi-row head forward instead of a separate appendHistory forward.
    /// Rounds with no successor (EOS/length/runtime disable) never pay for
    /// the append; the stash is freed unconsumed in `deinit`.
    mtp_hist_stash: ?MtpHistStash = null,
    /// Cross-round pre-draft (round pipelining): the NEXT round's chunk-A
    /// draft chain, built and async-dispatched at the CURRENT round's tail
    /// so the head chain runs on the GPU while the CPU emits tokens. The
    /// build consumes `mtp_hist_stash`, so the two are mutually exclusive
    /// (asserted at consume). Freed unconsumed in `deinit`.
    mtp_pre_draft: ?MtpPreDraft = null,
    /// Lazy pre-drafts built from a round's GPU results, and how many the host read kept.
    mtp_lazy_built: u32 = 0,
    mtp_lazy_kept: u32 = 0,

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
    // `nextPld`, `nextDrafter`, and `nextDflash` short-circuit to `next()` for the
    // remainder of the request — the prompt-time gate could not foresee that
    // the workload's *runtime* draft acceptance rate wasn't paying for the
    // per-step verify overhead. The flag is sticky for the rest of the
    // generation; we never re-enable speculation within a single request.
    spec_disabled_runtime: bool = false,
    /// Why `spec_disabled_runtime` was set, for `[spec-stats]`.
    spec_disable_reason: SpecDisableReason = .none,
    /// Adaptive serial controller: which arm this request is on and the KV bucket it was decided in.
    mtp_adaptive: MtpAdaptive = .{},
    /// This request has released the module-owned MTP head (sticky serial arm reached a serial
    /// block boundary). Per request: one slow request must not turn MTP off process-wide.
    mtp_head_released: bool = false,
    /// This request's realized MTP price (ms per emitted token) over a trailing window.
    mtp_price: MtpPriceWindow = .{},
    /// Ticks left in a bounded serial block (the serial probe); the slot stays an MTP slot.
    mtp_serial_left: u32 = 0,
    mtp_low_accept: u32 = 0,
    mtp_tok_ema: f32 = 0,
    /// Where a serial block sits in its ramp back to `nextMtp`'s entry invariant.
    mtp_serial_exit: MtpSerialExit = .none,
    /// Ticks of the current serial block spent warming (the previous round's tail).
    mtp_serial_warm: u32 = 0,
    mtp_serial_giveup_logged: bool = false,
    /// Inter-tick wall clock feeding the table's serial cell. Null = the next read seeds.
    mtp_serial_clock: ?io_util.Stopwatch = null,
    /// Yield-gate counters: enabled-mode `nextPld` steps and drafted tokens
    /// accepted since the last (re-)enable. Reset on mid-request re-enable so
    /// a fresh workload region (e.g. file echo after a novel preamble) gets a
    /// fresh economic evaluation instead of inheriting the bad early yield.
    yield_steps: u64 = 0,
    yield_accepted: u64 = 0,
    /// Steps spent in disabled mode since the gate tripped (drives the
    /// periodic `specShouldReenable` re-check).
    disabled_steps: u64 = 0,
    pub const RUNTIME_GATE_WARMUP = @import("../spec/gates.zig").RUNTIME_GATE_WARMUP;
    pub const RUNTIME_GATE_MIN_PER_DRAFT_RATE = @import("../spec/gates.zig").RUNTIME_GATE_MIN_PER_DRAFT_RATE;
    pub const runtimeGateShouldDisable = @import("../spec/gates.zig").runtimeGateShouldDisable;
    pub const DFLASH_GATE_WARMUP = @import("../spec/gates.zig").DFLASH_GATE_WARMUP;
    pub const DFLASH_GATE_MIN_ACCEPTED_PER_ROUND = @import("../spec/gates.zig").DFLASH_GATE_MIN_ACCEPTED_PER_ROUND;
    pub const DFLASH_THINKING_GATE_MIN_ACCEPTED_PER_ROUND = @import("../spec/gates.zig").DFLASH_THINKING_GATE_MIN_ACCEPTED_PER_ROUND;
    pub const DFLASH_MOE_GATE_MIN_ACCEPTED_PER_ROUND = @import("../spec/gates.zig").DFLASH_MOE_GATE_MIN_ACCEPTED_PER_ROUND;
    pub const dflashPolicyFor = @import("../spec/gates.zig").dflashPolicyFor;
    pub const dflashGateShouldDisable = @import("../spec/gates.zig").dflashGateShouldDisable;
    pub const YIELD_GATE_WARMUP = @import("../spec/gates.zig").YIELD_GATE_WARMUP;
    pub const YIELD_GATE_MIN_YIELD = @import("../spec/gates.zig").YIELD_GATE_MIN_YIELD;
    pub const yieldGateShouldDisable = @import("../spec/gates.zig").yieldGateShouldDisable;
    pub const SPEC_REENABLE_INTERVAL = @import("../spec/gates.zig").SPEC_REENABLE_INTERVAL;
    pub const SPEC_REENABLE_WINDOW = @import("../spec/gates.zig").SPEC_REENABLE_WINDOW;
    pub const SPEC_REENABLE_MIN_FRACTION = @import("../spec/gates.zig").SPEC_REENABLE_MIN_FRACTION;
    pub const SPEC_REENABLE_MIN_TOKENS = @import("../spec/gates.zig").SPEC_REENABLE_MIN_TOKENS;
    pub const specShouldReenable = @import("../spec/gates.zig").specShouldReenable;
    pub const logQsaArms = @import("../spec/stats_log.zig").logQsaArms;
    pub const logSpecStats = @import("../spec/stats_log.zig").logSpecStats;

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
    pub const CancelledCheckpointSink = @import("options.zig").CancelledCheckpointSink;
    pub const InitOptions = @import("options.zig").InitOptions;
    pub const InterleaveHook = @import("options.zig").InterleaveHook;
    pub const WriteThroughHook = @import("options.zig").WriteThroughHook;
    pub const ChunkWidthHook = @import("options.zig").ChunkWidthHook;
    pub const pickLookupPromptSource = @import("../spec/pld.zig").pickLookupPromptSource;
    pub const DsparkArm = @import("../spec/dspark.zig").DsparkArm;
    pub const dsparkArmFor = @import("../spec/dspark.zig").dsparkArmFor;
    pub const dsparkEnabled = @import("../spec/dspark.zig").dsparkEnabled;
    pub const initWithOptions = @import("init.zig").initWithOptions;

    /// Return the prefill's parked transients once, at the handover to decode.
    /// Shares `advanceStep`'s clock: a clear here is a clear, so the decode
    /// cadence measures its next interval from this point.
    pub fn clearPoolBeforeDecode(self: *Generator) void {
        if (!clearsPoolAtPrefillEnd(&self.xfm.config)) return;
        _ = mlx.mlx_clear_cache();
        self.last_cache_clear_step = self.step;
    }

    /// The ONE place `step` and `completion_tokens` advance — every decode path
    /// (plain, constrained, PLD, drafter, MTP, batched) routes through here.
    ///
    /// Advancing them by hand is how three paths — `nextDrafter`, `nextMtp` and
    /// `scheduler.runBatchedDecodeTick` — ended up never calling
    /// `mlx_clear_cache()` at all (issue #110). MLX parks freed buffers in a
    /// size-keyed pool instead of returning them to the OS, so a decode path
    /// with no clear ratchets the process footprint while `active_bytes` stays
    /// flat: the reporter's process reached 81 GB with the panel reading 19.6.
    /// A `-mtp` checkpoint on a dense trunk defaults straight onto one of them.
    /// A source-scan test pins that no new path can reintroduce the hole.
    pub fn advanceStep(self: *Generator, n: u32) void {
        self.completion_tokens += n;
        self.step += n;
        if (shouldClearAllocatorCache(self.step, self.last_cache_clear_step, CACHE_CLEAR_INTERVAL)) {
            _ = mlx.mlx_clear_cache();
            self.last_cache_clear_step = self.step;
        }
    }
    pub const hasPrePayloadConstraint = @import("../grammar/constraint.zig").hasPrePayloadConstraint;
    pub const takeConstraintSpan = @import("../grammar/constraint.zig").takeConstraintSpan;
    pub const ConstraintRecovery = @import("../grammar/constraint.zig").ConstraintRecovery;
    pub const forceConstraintTransition = @import("../grammar/constraint.zig").forceConstraintTransition;
    pub const commitForcedConstraintToken = @import("../grammar/constraint.zig").commitForcedConstraintToken;

    pub fn loopGuardStart(self: *const Generator) usize {
        return @min(self.loop_guard_start, self.generated_ids.items.len);
    }

    pub fn deinit(self: *Generator, allocator: std.mem.Allocator) void {
        // A GPU-arm PLE settle still owed dies with the slot's history (the finish settled it first when the cache kept it).
        self.xfm.discardDeferredPle(&self.ctx);
        plugin_settle.release(self.xfm, &self.ctx);
        for (&self.ssm_lag) |*lag| lag.deinit(self.xfm.allocator);
        if (group_planner.enabled() and self.mtp_planner_owned) log.info("[mtp-planner-stats] tokens={d} plain={d} prime={d} spec={d} probes={d} max_gap_ms={d:.2} recovery={d}\n", .{ self.completion_tokens, self.mtp_planner_plain_ticks, self.mtp_planner_prime_ticks, self.mtp_planner_spec_rounds, self.mtp_planner_probes, self.mtp_planner_max_gap_ms, self.mtp_planner_recovery.rounds });
        if (self.last_logprob) |*lp| {
            allocator.free(lp.top_logprobs);
        }
        if (self.pending_logprob) |*lp| {
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
        if (self.speech) |st| {
            st.deinit();
            st.a.destroy(st);
            self.speech = null;
        }
        if (self.prompt_ids_alloc) |a| {
            a.free(self.prompt_ids_owned);
            self.prompt_ids_owned = &.{};
            self.prompt_ids_alloc = null;
        }
        if (self.xfm.qwen4Trunk()) |t| t.markQsaPooledRopeStale();
        if (self.mtp_cache) |*mc| {
            mc.deinit();
            self.mtp_cache = null;
        }
        if (self.dflash_ctx) |*dc| {
            dc.deinit();
            self.dflash_ctx = null;
        }
        for (self.dflash_pending.items) |a| _ = mlx.mlx_array_free(a);
        self.dflash_pending.deinit(allocator);
        self.dflash_pending = .empty;
        self.dflash_pending_rounds = 0;
        self.dflash_pending_rows = 0;
        if (self.mtp_hist_stash) |*st| {
            st.deinit();
            self.mtp_hist_stash = null;
        }
        if (self.mtp_pre_draft) |*pd| {
            pd.deinit(allocator);
            self.mtp_pre_draft = null;
        }
        if (self.mtp_lookup_idx) |*idx| {
            idx.deinit();
            self.mtp_lookup_idx = null;
        }
        // Publish the EV surface for the next request when the experimental
        // cross-request seed is explicitly enabled.
        // Only healthy runs qualify — a runtime-disabled or barely-sampled
        // run would poison the next request's plans (inference thread only,
        // same discipline as every other head-state write).
        if (self.mtp) |head| {
            // A forced-depth run never consulted the EV plan, so its surface
            // is not one the controller chose — publishing it would hand a
            // later ordinary request a diagnostic's numbers.
            if (mtpPinnedDepth() == null and
                !self.spec_disabled_runtime and self.mtp_attempted >= 8 and
                (!group_planner.enabled() or !self.mtp_planner_owned))
            {
                head.setEvSeed(self.mtp_ev_accept, self.mtp_ev_m_lo_prev);
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
    pub const drainPipelineForBatch = @import("pipeline.zig").drainPipelineForBatch;
    pub const sampleLazy = @import("pipeline.zig").sampleLazy;
    pub const upcomingToken = @import("pipeline.zig").upcomingToken;
    pub const resolvePendingToken = @import("pipeline.zig").resolvePendingToken;
    const DrainResult = @import("pipeline.zig").DrainResult;
    pub const drainPipelineForSpec = @import("pipeline.zig").drainPipelineForSpec;
    pub const ForcedCommit = @import("pipeline.zig").ForcedCommit;
    pub const commitForcedTokens = @import("pipeline.zig").commitForcedTokens;
    pub const PldStepResult = @import("../spec/pld.zig").PldStepResult;
    pub const nextDspark = @import("../spec/dspark.zig").nextDspark;
    pub const dsparkCopyStep = @import("../spec/dspark.zig").dsparkCopyStep;
    pub const dsparkStep = @import("../spec/dspark.zig").dsparkStep;
    pub const dsparkPlainPipelined = @import("../spec/dspark.zig").dsparkPlainPipelined;
    pub const dsparkPolicyStep = @import("../spec/dspark.zig").dsparkPolicyStep;
    pub const sigmoid = @import("../spec/dspark.zig").sigmoid;
    pub const dsparkVerifyDrafts = @import("../spec/dspark.zig").dsparkVerifyDrafts;
    pub const commitDsparkRound = @import("../spec/dspark.zig").commitDsparkRound;
    pub const nextDraw = @import("../spec/dspark.zig").nextDraw;
    pub const dsparkStochasticRound = @import("../spec/dspark.zig").dsparkStochasticRound;
    pub const rollbackSsmFromCapture = @import("pipeline.zig").rollbackSsmFromCapture;
    pub const nextPld = @import("../spec/pld.zig").nextPld;
    pub const DrafterStepResult = @import("../spec/drafter.zig").DrafterStepResult;
    pub const correctionDraw = @import("../spec/drafter.zig").correctionDraw;
    pub const nextDrafter = @import("../spec/drafter.zig").nextDrafter;
    pub const nextDflash = @import("../spec/dflash.zig").nextDflash;
    const VerifyIn = @import("../spec/dflash.zig").VerifyIn;
    pub const dflashVerify = @import("../spec/dflash.zig").dflashVerify;
    const DFLASH_COPY_MAX = @import("../spec/dflash.zig").DFLASH_COPY_MAX;
    pub const dflashCopyRound = @import("../spec/dflash.zig").dflashCopyRound;
    const DFLASH_PENDING_MAX = @import("../spec/dflash.zig").DFLASH_PENDING_MAX;
    pub const dflashPlainRound = @import("../spec/dflash.zig").dflashPlainRound;
    pub const dflashPark = @import("../spec/dflash.zig").dflashPark;
    pub const dflashCanPipeline = @import("../spec/dflash.zig").dflashCanPipeline;
    pub const dflashPlainPipelined = @import("../spec/dflash.zig").dflashPlainPipelined;
    pub const dflashEvalPark = @import("../spec/dflash.zig").dflashEvalPark;
    pub const dflashLeavePipeline = @import("../spec/dflash.zig").dflashLeavePipeline;
    pub const dflashFlushPending = @import("../spec/dflash.zig").dflashFlushPending;
    pub const dflashContextLen = @import("../spec/dflash.zig").dflashContextLen;
    pub const dflashSettle = @import("../spec/dflash.zig").dflashSettle;
    pub const dflashRecordDrafted = @import("../spec/dflash.zig").dflashRecordDrafted;
    pub const dflashPlainMs = @import("../spec/dflash.zig").dflashPlainMs;
    pub const dflashRowsFor = @import("../spec/dflash.zig").dflashRowsFor;
    pub const dflashPolicyRows = @import("../spec/dflash.zig").dflashPolicyRows;
    pub const logDflashRound = @import("../spec/dflash.zig").logDflashRound;
    const VERIFY_PIPELINE_LAYERS = @import("../spec/dflash_tree.zig").VERIFY_PIPELINE_LAYERS;
    const DFLASH_TREE_W = @import("../spec/dflash_tree.zig").DFLASH_TREE_W;
    const DFLASH_COPY_MATCH = @import("../spec/dflash_tree.zig").DFLASH_COPY_MATCH;
    const DFLASH_COPY_WINDOW = @import("../spec/dflash_tree.zig").DFLASH_COPY_WINDOW;
    pub const treeNoise = @import("../spec/dflash_tree.zig").treeNoise;
    const dflashTreeNodes = @import("../spec/dflash_tree.zig").dflashTreeNodes;
    pub const dflashTreeRound = @import("../spec/dflash_tree.zig").dflashTreeRound;
    pub const yieldSpecToCompany = @import("../spec/gates.zig").yieldSpecToCompany;
    pub const checkDflashRuntimeGate = @import("../spec/gates.zig").checkDflashRuntimeGate;
    pub const checkDrafterRuntimeGate = @import("../spec/gates.zig").checkDrafterRuntimeGate;
    pub const MtpHistStash = @import("../spec/mtp/chain.zig").MtpHistStash;
    pub const mtpRoundOff0 = @import("../spec/mtp/chain.zig").mtpRoundOff0;
    pub const mtpCommittedLen = @import("../spec/mtp/chain.zig").mtpCommittedLen;
    pub const mtpCommittedHistoryLen = @import("../spec/mtp/chain.zig").mtpCommittedHistoryLen;
    pub const mtpMropeContext = @import("../spec/mtp/chain.zig").mtpMropeContext;
    pub const verifySerialSamples = @import("../spec/mtp/chain.zig").verifySerialSamples;
    pub const verifySerialSamplesAt = @import("../spec/mtp/chain.zig").verifySerialSamplesAt;
    pub const mtpCoupledDrafts = @import("../spec/mtp/chain.zig").mtpCoupledDrafts;
    pub const mtpSamplingDraw = @import("../spec/mtp/chain.zig").mtpSamplingDraw;
    pub const MtpPreDraft = @import("../spec/mtp/chain.zig").MtpPreDraft;
    pub const mtpChainInit = @import("../spec/mtp/chain.zig").mtpChainInit;
    pub const acceptGraphFor = @import("../spec/mtp/chain.zig").acceptGraphFor;
    pub const acceptPrefixFor = @import("../spec/mtp/chain.zig").acceptPrefixFor;
    const mtpLookupEnabled = @import("../spec/mtp/chain.zig").mtpLookupEnabled;
    pub const mtpRoundUntimed = @import("../spec/mtp/chain.zig").mtpRoundUntimed;
    pub const mtpLookupObserve = @import("../spec/mtp/chain.zig").mtpLookupObserve;
    pub const mtpLookupCosts = @import("../spec/mtp/chain.zig").mtpLookupCosts;
    pub const mtpRoundAcceptObserve = @import("../spec/mtp/chain.zig").mtpRoundAcceptObserve;
    pub const mtpLookupIndex = @import("../spec/mtp/chain.zig").mtpLookupIndex;
    pub const mtpLookupDrafts = @import("../spec/mtp/chain.zig").mtpLookupDrafts;
    pub const mtpLookupChain = @import("../spec/mtp/chain.zig").mtpLookupChain;
    pub const MtpLazyVerdict = @import("../spec/mtp/chain.zig").MtpLazyVerdict;
    pub const mtpLazyVerdict = @import("../spec/mtp/chain.zig").mtpLazyVerdict;
    const concatIds = @import("../spec/mtp/chain.zig").concatIds;
    pub const LazyGate = @import("../spec/mtp/chain.zig").LazyGate;
    pub const mtpLazyPredraftAllowedFor = @import("../spec/mtp/chain.zig").mtpLazyPredraftAllowedFor;
    const mtpLazyPredraftEnabled = @import("../spec/mtp/chain.zig").mtpLazyPredraftEnabled;
    pub const mtpLazyGate = @import("../spec/mtp/chain.zig").mtpLazyGate;
    pub const mtpLazyPreDraft = @import("../spec/mtp/chain.zig").mtpLazyPreDraft;
    pub const mtpLazyKeeps = @import("../spec/mtp/chain.zig").mtpLazyKeeps;
    pub const mtpPreDraftResolve = @import("../spec/mtp/chain.zig").mtpPreDraftResolve;
    const mtpPaddedHeadEnabled = @import("../spec/mtp/chain.zig").mtpPaddedHeadEnabled;
    pub const mtpPaddedHistory = @import("../spec/mtp/chain.zig").mtpPaddedHistory;
    pub const mtpPaddedHistoryAppend = @import("../spec/mtp/chain.zig").mtpPaddedHistoryAppend;
    pub const mtpPaddedAppend = @import("../spec/mtp/chain.zig").mtpPaddedAppend;
    pub const mtpChainBuild = @import("../spec/mtp/chain_build.zig").mtpChainBuild;
    const mtpChainDispatch = @import("../spec/mtp/chain_build.zig").mtpChainDispatch;
    pub const mtpChainDispatchBatched = @import("../spec/mtp/chain_build.zig").mtpChainDispatchBatched;
    pub const mtpDraftActiveRows = @import("../spec/mtp/chain_build.zig").mtpDraftActiveRows;
    pub const DraftStepPath = @import("../spec/mtp/chain_build.zig").DraftStepPath;
    pub const mtpDraftStepPath = @import("../spec/mtp/chain_build.zig").mtpDraftStepPath;
    pub const mtpRerankDrafts = @import("../spec/mtp/chain_build.zig").mtpRerankDrafts;
    const mtpRowRerankMode = @import("../spec/mtp/chain_build.zig").mtpRowRerankMode;
    pub const mtpChainBuildBatched = @import("../spec/mtp/chain_build.zig").mtpChainBuildBatched;
    const mtpGroupSampledDrafts = @import("../spec/mtp/chain_build.zig").mtpGroupSampledDrafts;
    const mtpChainBuildLanes = @import("../spec/mtp/chain_build.zig").mtpChainBuildLanes;
    pub const mtpBatchHeadPosition = @import("../spec/mtp/chain_build.zig").mtpBatchHeadPosition;
    const mtpChainBuildBatchedActive = @import("../spec/mtp/chain_build.zig").mtpChainBuildBatchedActive;
    pub const MtpBatchedGraph = @import("../spec/mtp/batched_graph.zig").MtpBatchedGraph;
    pub const mtpBatchedExactGraph = @import("../spec/mtp/batched_graph.zig").mtpBatchedExactGraph;
    pub const mtpBatchedAcceptGraph = @import("../spec/mtp/batched_graph.zig").mtpBatchedAcceptGraph;
    pub const mtpBatchedTypicalGraph = @import("../spec/mtp/batched_graph.zig").mtpBatchedTypicalGraph;
    pub const mtpBatchedTokenV3Graph = @import("../spec/mtp/batched_graph.zig").mtpBatchedTokenV3Graph;
    const mtpBatchedLossyGraph = @import("../spec/mtp/batched_graph.zig").mtpBatchedLossyGraph;
    pub const mtpMaybePreDraft = @import("../spec/mtp/serial.zig").mtpMaybePreDraft;
    pub const mtpPreDraftOpenAllows = @import("../spec/mtp/serial.zig").mtpPreDraftOpenAllows;
    pub const mtpPreDraftOpen = @import("../spec/mtp/serial.zig").mtpPreDraftOpen;
    pub const mtpGroupPreDraft = @import("../spec/mtp/serial.zig").mtpGroupPreDraft;
    pub const mtpSerialMayResume = @import("../spec/mtp/serial.zig").mtpSerialMayResume;
    pub const mtpResumeAfterForced = @import("../spec/mtp/serial.zig").mtpResumeAfterForced;
    pub const mtpDetachHead = @import("../spec/mtp/serial.zig").mtpDetachHead;
    pub const mtpApplyStash = @import("../spec/mtp/serial.zig").mtpApplyStash;
    const mtpSerialOneToken = @import("../spec/mtp/serial.zig").mtpSerialOneToken;
    pub const mtpSerialWallMs = @import("../spec/mtp/serial.zig").mtpSerialWallMs;
    pub const mtpAdaptiveModelOk = @import("../spec/mtp/serial.zig").mtpAdaptiveModelOk;
    pub const serialCellWanted = @import("../spec/mtp/serial.zig").serialCellWanted;
    pub const invalidateSerialClock = @import("../spec/mtp/serial.zig").invalidateSerialClock;
    pub const invalidateRoundClock = @import("../spec/mtp/serial.zig").invalidateRoundClock;
    pub const observeSerialTick = @import("../spec/mtp/serial.zig").observeSerialTick;
    pub const mtpSerialCaptureReady = @import("../spec/mtp/serial.zig").mtpSerialCaptureReady;
    pub const mtpSerialGiveUp = @import("../spec/mtp/serial.zig").mtpSerialGiveUp;
    pub const mtpSerialTick = @import("../spec/mtp/serial.zig").mtpSerialTick;
    pub const mtpSerialCaptureTick = @import("../spec/mtp/serial.zig").mtpSerialCaptureTick;
    pub const MtpRoundState = @import("../spec/mtp/round.zig").MtpRoundState;
    pub const MtpRowVerdict = @import("../spec/mtp/round.zig").MtpRowVerdict;
    pub const mtpAcceptRowGreedy = @import("../spec/mtp/round.zig").mtpAcceptRowGreedy;
    pub const mtpGreedyVerdict = @import("../spec/mtp/round.zig").mtpGreedyVerdict;
    pub const mtpGreedyPrefix = @import("../spec/mtp/round.zig").mtpGreedyPrefix;
    pub const mtpRetained = @import("../spec/mtp/round.zig").mtpRetained;
    pub const mtpFullAccept = @import("../spec/mtp/round.zig").mtpFullAccept;
    pub const MTP_GROUP_ROWS_MAX = @import("../spec/mtp/round.zig").MTP_GROUP_ROWS_MAX;
    pub const MtpRoundOpen = @import("../spec/mtp/round.zig").MtpRoundOpen;
    pub const MtpRoundBegin = @import("../spec/mtp/round.zig").MtpRoundBegin;
    pub const nextMtp = @import("../spec/mtp/round.zig").nextMtp;
    pub const mtpRoundBegin = @import("../spec/mtp/round.zig").mtpRoundBegin;
    pub const mtpRoundContinue = @import("../spec/mtp/round.zig").mtpRoundContinue;
    pub const mtpRoundAfterChain = @import("../spec/mtp/round.zig").mtpRoundAfterChain;
    pub const mtpRoundVerify = @import("../spec/mtp/round.zig").mtpRoundVerify;
    pub const mtpGroupVerify = @import("../spec/mtp/round.zig").mtpGroupVerify;
    pub const mtpGroupAccept = @import("../spec/mtp/round.zig").mtpGroupAccept;
    pub const mtpGroupAcceptBatchable = @import("../spec/mtp/round.zig").mtpGroupAcceptBatchable;
    const mtpGroupSampledAccept = @import("../spec/mtp/round.zig").mtpGroupSampledAccept;
    pub const mtpPaddedStash = @import("../spec/mtp/round.zig").mtpPaddedStash;
    pub const mtpPaddedHidden = @import("../spec/mtp/round.zig").mtpPaddedHidden;
    pub const mtpStopPrefix = @import("../spec/mtp/round.zig").mtpStopPrefix;
    pub const mtpRoundFinish = @import("../spec/mtp/round_finish.zig").mtpRoundFinish;
    pub const MTP_DEPTH_WINDOW = @import("../spec/mtp/depth.zig").MTP_DEPTH_WINDOW;
    pub const MTP_DEPTH_SWITCH_WARMUP = @import("../spec/mtp/depth.zig").MTP_DEPTH_SWITCH_WARMUP;
    pub const MTP_DEMOTE_BELOW = @import("../spec/mtp/depth.zig").MTP_DEMOTE_BELOW;
    pub const MTP_PROMOTE_ABOVE = @import("../spec/mtp/depth.zig").MTP_PROMOTE_ABOVE;
    pub const MTP_DISABLE_BELOW = @import("../spec/mtp/depth.zig").MTP_DISABLE_BELOW;
    pub const MTP_PROMOTE_COOLDOWN = @import("../spec/mtp/depth.zig").MTP_PROMOTE_COOLDOWN;
    pub const MTP_SAMPLED_PROPOSAL_MIN_TEMP = @import("../spec/mtp/depth.zig").MTP_SAMPLED_PROPOSAL_MIN_TEMP;
    pub const mtpDraftGreedyFor = @import("../spec/mtp/depth.zig").mtpDraftGreedyFor;
    pub const mtpDraftGreedyAt = @import("../spec/mtp/depth.zig").mtpDraftGreedyAt;
    pub const MTP_DRAFT_TEMP = @import("../spec/mtp/depth.zig").MTP_DRAFT_TEMP;
    pub const MTP_DRAFT_TOP_P = @import("../spec/mtp/depth.zig").MTP_DRAFT_TOP_P;
    pub const MTP_DRAFT_TOP_K = @import("../spec/mtp/depth.zig").MTP_DRAFT_TOP_K;
    pub const mtpDraftTempFor = @import("../spec/mtp/depth.zig").mtpDraftTempFor;
    pub const mtpDraftSamplingFor = @import("../spec/mtp/depth.zig").mtpDraftSamplingFor;
    pub const mtpDraftSampling = @import("../spec/mtp/depth.zig").mtpDraftSampling;
    pub const specAcceptProb = @import("../spec/mtp/depth.zig").specAcceptProb;
    pub const mtpNextDepth = @import("../spec/mtp/depth.zig").mtpNextDepth;
    pub const mtpDepthDecision = @import("../spec/mtp/depth.zig").mtpDepthDecision;
    pub const updateMtpDepth = @import("../spec/mtp/depth.zig").updateMtpDepth;
    pub const MTP_ADAPTIVE_DEFAULT_CAP = @import("../spec/mtp/depth.zig").MTP_ADAPTIVE_DEFAULT_CAP;
    pub const MTP_ADAPTIVE_NAX_CAP = @import("../spec/mtp/depth.zig").MTP_ADAPTIVE_NAX_CAP;
    pub const MTP_CLIFF_TOLERANCE = @import("../spec/mtp/depth.zig").MTP_CLIFF_TOLERANCE;
    pub const MTP_EV_WARMUP_ROUNDS = @import("../spec/mtp/depth.zig").MTP_EV_WARMUP_ROUNDS;
    pub const MTP_EV_EMA_BETA = @import("../spec/mtp/depth.zig").MTP_EV_EMA_BETA;
    pub const MTP_EV_PRIOR = @import("../spec/mtp/depth.zig").MTP_EV_PRIOR;
    pub const MTP_EV_TAU_MIN = @import("../spec/mtp/depth.zig").MTP_EV_TAU_MIN;
    pub const MTP_EV_TAU_MAX = @import("../spec/mtp/depth.zig").MTP_EV_TAU_MAX;
    pub const MTP_EV_EXPLORE_MIN_R = @import("../spec/mtp/depth.zig").MTP_EV_EXPLORE_MIN_R;
    pub const MtpEvCosts = @import("../spec/mtp/ev_costs.zig").MtpEvCosts;
    pub const MTP_EV_DEFAULT_COSTS = @import("../spec/mtp/ev_costs.zig").MTP_EV_DEFAULT_COSTS;
    pub const MTP_EV_G17_NAX_COSTS = @import("../spec/mtp/ev_costs.zig").MTP_EV_G17_NAX_COSTS;
    pub const MTP_EV_G17_NAX_Q4_GS32_COSTS = @import("../spec/mtp/ev_costs.zig").MTP_EV_G17_NAX_Q4_GS32_COSTS;
    pub const MTP_EV_G17_NAX_Q4_GS64_COSTS = @import("../spec/mtp/ev_costs.zig").MTP_EV_G17_NAX_Q4_GS64_COSTS;
    pub const MTP_EV_G17_NAX_Q6_GS64_COSTS = @import("../spec/mtp/ev_costs.zig").MTP_EV_G17_NAX_Q6_GS64_COSTS;
    pub const MTP_EV_G17_NAX_Q8_GS64_COSTS = @import("../spec/mtp/ev_costs.zig").MTP_EV_G17_NAX_Q8_GS64_COSTS;
    pub const MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS = @import("../spec/mtp/ev_costs.zig").MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS;
    pub const MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS = @import("../spec/mtp/ev_costs.zig").MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS;
    pub const MTP_EV_G17_NAX_QWEN4_MIXED_4_8_GS64_COSTS = @import("../spec/mtp/ev_costs.zig").MTP_EV_G17_NAX_QWEN4_MIXED_4_8_GS64_COSTS;
    pub const mtpExactDepth = @import("../spec/mtp/observe.zig").mtpExactDepth;
    const MTP_EXACT_PROBE = @import("../spec/mtp/observe.zig").MTP_EXACT_PROBE;
    pub const mtpExactAcceptance = @import("../spec/mtp/observe.zig").mtpExactAcceptance;
    const MTP_EXACT_PRIOR_ROUNDS = @import("../spec/mtp/observe.zig").MTP_EXACT_PRIOR_ROUNDS;
    pub const mtpExactObserve = @import("../spec/mtp/observe.zig").mtpExactObserve;
    pub const mtpRoundEndObserve = @import("../spec/mtp/observe.zig").mtpRoundEndObserve;
    pub const specObserveRound = @import("../spec/mtp/observe.zig").specObserveRound;
    pub const persistRoundCost = @import("../spec/mtp/ev_plan.zig").persistRoundCost;
    pub const mtpKvLen = @import("../spec/mtp/ev_plan.zig").mtpKvLen;
    pub const mtpEvMarginalCost = @import("../spec/mtp/ev_plan.zig").mtpEvMarginalCost;
    pub const MtpRoundPlan = @import("../spec/mtp/ev_plan.zig").MtpRoundPlan;
    pub const mtpDepthCapForProfile = @import("../spec/mtp/ev_plan.zig").mtpDepthCapForProfile;
    pub const mtpDepthCapFree = @import("../spec/mtp/ev_plan.zig").mtpDepthCapFree;
    pub const mtpDepthCapForProfileChip = @import("../spec/mtp/ev_plan.zig").mtpDepthCapForProfileChip;
    pub const mtpDepthCapResolved = @import("../spec/mtp/ev_plan.zig").mtpDepthCapResolved;
    pub const resolveMtpDepthCapForProfile = @import("../spec/mtp/ev_plan.zig").resolveMtpDepthCapForProfile;
    pub const mtpDepthCapFor = @import("../spec/mtp/ev_plan.zig").mtpDepthCapFor;
    pub const resolveMtpDepthCap = @import("../spec/mtp/ev_plan.zig").resolveMtpDepthCap;
    pub const mtpEvExpectedTokens = @import("../spec/mtp/ev_plan.zig").mtpEvExpectedTokens;
    pub const mtpEvMarginalCostAt = @import("../spec/mtp/ev_plan.zig").mtpEvMarginalCostAt;
    pub const mtpEvRoundCost = @import("../spec/mtp/ev_plan.zig").mtpEvRoundCost;
    pub const mtpEvRoundCostAt = @import("../spec/mtp/ev_plan.zig").mtpEvRoundCostAt;
    pub const mtpEvPlanFor = @import("../spec/mtp/ev_plan.zig").mtpEvPlanFor;
    pub const mtpEvPlanForAt = @import("../spec/mtp/ev_plan.zig").mtpEvPlanForAt;
    pub const MTP_EV_TABLE_MIN_MARGINAL = @import("../spec/mtp/ev_plan.zig").MTP_EV_TABLE_MIN_MARGINAL;
    pub const MtpCostSource = @import("../spec/mtp/ev_plan.zig").MtpCostSource;
    pub const MtpDepthPolicy = @import("../spec/mtp/ev_plan.zig").MtpDepthPolicy;
    pub const MTP_ACCEPT_MAX_KV = @import("../spec/mtp/ev_plan.zig").MTP_ACCEPT_MAX_KV;
    pub const mtpDepthPolicyFor = @import("../spec/mtp/ev_plan.zig").mtpDepthPolicyFor;
    pub const mtpDepthPolicy = @import("../spec/mtp/ev_plan.zig").mtpDepthPolicy;
    pub const mtpBasePlan = @import("../spec/mtp/ev_plan.zig").mtpBasePlan;
    pub const mtpEvPlanSrc = @import("../spec/mtp/ev_plan.zig").mtpEvPlanSrc;
    pub const mtpEvObserve = @import("../spec/mtp/ev_plan.zig").mtpEvObserve;
    pub const mtpChainLogConf = @import("../spec/mtp/ev_plan.zig").mtpChainLogConf;
    pub const MtpTrace = @import("../spec/mtp/trace.zig").MtpTrace;
    pub const SubLap = @import("../spec/mtp/trace.zig").SubLap;
    pub const DflashTrace = @import("../spec/mtp/trace.zig").DflashTrace;
    const dflashTraceEnabled = @import("../spec/mtp/trace.zig").dflashTraceEnabled;
    pub const dflashTraceRoundEnd = @import("../spec/mtp/trace.zig").dflashTraceRoundEnd;
    const mtpPredraftEnabled = @import("../spec/mtp/trace.zig").mtpPredraftEnabled;
    pub const mtpTraceOn = @import("../spec/mtp/trace.zig").mtpTraceOn;
    pub const mtpTraceSub = @import("../spec/mtp/trace.zig").mtpTraceSub;
    const mtpTraceEnabled = @import("../spec/mtp/trace.zig").mtpTraceEnabled;
    pub const mtpEvCosts = @import("../spec/mtp/ev_costs.zig").mtpEvCosts;
    pub const MTP_EXT_DRY_ROUNDS = @import("../spec/mtp/regime.zig").MTP_EXT_DRY_ROUNDS;
    pub const MTP_EXT_DRY_COOLDOWN = @import("../spec/mtp/regime.zig").MTP_EXT_DRY_COOLDOWN;
    pub const MTP_EXT_DRY_MIN = @import("../spec/mtp/regime.zig").MTP_EXT_DRY_MIN;
    pub const MTP_EXT_SYNC_BUDGET = @import("../spec/mtp/regime.zig").MTP_EXT_SYNC_BUDGET;
    pub const mtpExtDryThresholdFor = @import("../spec/mtp/regime.zig").mtpExtDryThresholdFor;
    pub const mtpExtDryAllows = @import("../spec/mtp/regime.zig").mtpExtDryAllows;
    pub const MTP_EV_COST_BETA = @import("../spec/mtp/regime.zig").MTP_EV_COST_BETA;
    pub const mtpEmaMs = @import("../spec/mtp/regime.zig").mtpEmaMs;
    pub const MtpRegime = @import("../spec/mtp/regime.zig").MtpRegime;
    pub const MTP_REGIME_EXPLORE_PERIOD = @import("../spec/mtp/regime.zig").MTP_REGIME_EXPLORE_PERIOD;
    pub const MTP_REGIME_EXPLORE_DRAG = @import("../spec/mtp/regime.zig").MTP_REGIME_EXPLORE_DRAG;
    pub const MTP_REGIME_EXPLORE_PERIOD_MAX = @import("../spec/mtp/regime.zig").MTP_REGIME_EXPLORE_PERIOD_MAX;
    pub const MTP_REGIME_EXPLORE_BLOCK = @import("../spec/mtp/regime.zig").MTP_REGIME_EXPLORE_BLOCK;
    const regimeEma = @import("../spec/mtp/regime.zig").regimeEma;
    pub const mtpRegimeWallMs = @import("../spec/mtp/regime.zig").mtpRegimeWallMs;
    pub const mtpRegimeObserve = @import("../spec/mtp/regime.zig").mtpRegimeObserve;
    pub const MTP_REGIME_MARGIN = @import("../spec/mtp/regime.zig").MTP_REGIME_MARGIN;
    pub const mtpRegimeTwoChunkWorse = @import("../spec/mtp/regime.zig").mtpRegimeTwoChunkWorse;
    pub const mtpRegimeVerdict = @import("../spec/mtp/regime.zig").mtpRegimeVerdict;
    pub const mtpRegimeExplorePeriod = @import("../spec/mtp/regime.zig").mtpRegimeExplorePeriod;
    pub const mtpRegimeForce = @import("../spec/mtp/regime.zig").mtpRegimeForce;
    const mtpRegimeForceAt = @import("../spec/mtp/regime.zig").mtpRegimeForceAt;
    pub const mtpPinnedDepth = @import("../spec/mtp/regime.zig").mtpPinnedDepth;
    pub const mtpPlanInBounds = @import("../spec/mtp/regime.zig").mtpPlanInBounds;
    pub const MTP_ADAPTIVE_MARGIN = @import("../spec/mtp/regime.zig").MTP_ADAPTIVE_MARGIN;
    pub const MTP_ADAPTIVE_CONFIRM = @import("../spec/mtp/regime.zig").MTP_ADAPTIVE_CONFIRM;
    pub const MTP_ADAPTIVE_PROBE_TOKENS = @import("../spec/mtp/regime.zig").MTP_ADAPTIVE_PROBE_TOKENS;
    pub const MTP_ADAPTIVE_PROBE_WARM = @import("../spec/mtp/adaptive.zig").MTP_ADAPTIVE_PROBE_WARM;
    pub const MTP_ADAPTIVE_MIN_KV = @import("../spec/mtp/adaptive.zig").MTP_ADAPTIVE_MIN_KV;
    pub const mtpAdaptiveKvEligible = @import("../spec/mtp/adaptive.zig").mtpAdaptiveKvEligible;
    pub const mtpAdaptiveBucketOf = @import("../spec/mtp/adaptive.zig").mtpAdaptiveBucketOf;
    pub const mtpAdaptiveBucket = @import("../spec/mtp/adaptive.zig").mtpAdaptiveBucket;
    pub const mtpAdaptiveRegimeMoved = @import("../spec/mtp/adaptive.zig").mtpAdaptiveRegimeMoved;
    pub const mtpAdaptiveSyncWindow = @import("../spec/mtp/adaptive.zig").mtpAdaptiveSyncWindow;
    pub const MTP_PRICE_WINDOW = @import("../spec/mtp/adaptive.zig").MTP_PRICE_WINDOW;
    pub const MtpPriceWindow = @import("../spec/mtp/adaptive.zig").MtpPriceWindow;
    pub const MtpAdaptiveVote = @import("../spec/mtp/adaptive.zig").MtpAdaptiveVote;
    pub const MtpAdaptiveArm = @import("../spec/mtp/adaptive.zig").MtpAdaptiveArm;
    pub const MtpAdaptiveAction = @import("../spec/mtp/adaptive.zig").MtpAdaptiveAction;
    pub const MtpSerialExit = @import("../spec/mtp/adaptive.zig").MtpSerialExit;
    pub const mtpAdaptiveVoteFor = @import("../spec/mtp/adaptive.zig").mtpAdaptiveVoteFor;
    pub const mtpSerialProbeUseful = @import("../spec/mtp/adaptive.zig").mtpSerialProbeUseful;
    pub const mtpSerialProbeArm = @import("../spec/mtp/adaptive.zig").mtpSerialProbeArm;
    pub const MtpAdaptive = @import("../spec/mtp/adaptive.zig").MtpAdaptive;
    pub const mtpHeadPositionDrift = @import("../spec/mtp/adaptive.zig").mtpHeadPositionDrift;
    pub const mtpHeadDriftNow = @import("../spec/mtp/adaptive.zig").mtpHeadDriftNow;
    pub const mtpAdaptiveHeadMayResume = @import("../spec/mtp/adaptive.zig").mtpAdaptiveHeadMayResume;
    pub const stickyOnSerialSwitch = @import("../spec/mtp/adaptive.zig").stickyOnSerialSwitch;
    pub const stickyReleaseNow = @import("../spec/mtp/adaptive.zig").stickyReleaseNow;
    pub const mtpModuleHeadReleased = @import("../spec/mtp/adaptive.zig").mtpModuleHeadReleased;
    pub const mtpReleasePending = @import("../spec/mtp/adaptive.zig").mtpReleasePending;
    pub const mtpMaybeReleaseModuleHead = @import("../spec/mtp/adaptive.zig").mtpMaybeReleaseModuleHead;
    pub const mtpAdaptiveArchEligible = @import("../spec/mtp/adaptive.zig").mtpAdaptiveArchEligible;
    pub const MTP_PROBE_LOW_TOK = @import("../spec/mtp/adaptive.zig").MTP_PROBE_LOW_TOK;
    pub const MTP_PROBE_LOW_ROUNDS = @import("../spec/mtp/adaptive.zig").MTP_PROBE_LOW_ROUNDS;
    pub const mtpLowAcceptStreak = @import("../spec/mtp/adaptive.zig").mtpLowAcceptStreak;
    pub const mtpSerialProbeEarned = @import("../spec/mtp/adaptive.zig").mtpSerialProbeEarned;
    pub const mtpAdaptiveSerialStep = @import("../spec/mtp/adaptive.zig").mtpAdaptiveSerialStep;
    pub const mtpPlannerMode = @import("../spec/mtp/planner.zig").mtpPlannerMode;
    pub const mtpPlannerObserve = @import("../spec/mtp/planner.zig").mtpPlannerObserve;
    pub const mtpRoundPlan = @import("../spec/mtp/planner.zig").mtpRoundPlan;
    pub const mtpGroupPlan = @import("../spec/mtp/planner.zig").mtpGroupPlan;
    const planMs = @import("../spec/mtp/planner.zig").planMs;
    pub const mtpRoundPlanTraced = @import("../spec/mtp/planner.zig").mtpRoundPlanTraced;
    pub const mtpRoundPlanInner = @import("../spec/mtp/planner.zig").mtpRoundPlanInner;
    pub const mtpWidthTrialPlan = @import("../spec/mtp/planner.zig").mtpWidthTrialPlan;
    pub const mtpWidthTrialTarget = @import("../spec/mtp/planner.zig").mtpWidthTrialTarget;
    pub const MtpWidthTrial = @import("../spec/mtp/planner.zig").MtpWidthTrial;
    pub const mtpWidthTrialForce = @import("../spec/mtp/planner.zig").mtpWidthTrialForce;
    pub const mtpWidthTrialPeriod = @import("../spec/mtp/planner.zig").mtpWidthTrialPeriod;
    pub const mtpProbePeriod = @import("../spec/mtp/planner.zig").mtpProbePeriod;
    pub const mtpFloorDisableObserve = @import("../spec/mtp/planner.zig").mtpFloorDisableObserve;
    pub const updateMtpEvRound = @import("../spec/mtp/planner.zig").updateMtpEvRound;
    pub const mtpTraceRoundEnd = @import("../spec/mtp/planner.zig").mtpTraceRoundEnd;
    pub const pushSsmLag = @import("pipeline.zig").pushSsmLag;
    pub const rollbackSsmTo = @import("pipeline.zig").rollbackSsmTo;
    pub const next = @import("step.zig").next;
    pub const nextReasoningStep = @import("step.zig").nextReasoningStep;
    pub const nextChoiceStep = @import("step.zig").nextChoiceStep;
    pub const nextConstrained = @import("step.zig").nextConstrained;
    pub const nextConstrainedToken = @import("step.zig").nextConstrainedToken;
    pub const checkStop = @import("step.zig").checkStop;
    pub const startAsyncForward = @import("step.zig").startAsyncForward;
};

// ── tests ──

