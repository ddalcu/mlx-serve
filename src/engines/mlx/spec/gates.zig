//! Runtime spec gates: warmups, per-draft rate, DFlash and yield gates, re-enable, yielding to company.
const std = @import("std");
const model_mod = @import("../../../sdk/config/model.zig");
const log = @import("../../../sdk/util/log.zig");
const pld_index = @import("../../../sdk/spec/pld_index.zig");
const testing = std.testing;
const Generator = @import("../generate/generator.zig").Generator;

/// Number of attempts before the runtime gate considers disabling.
/// Below this we trust the prompt-time gate. Lower values make the gate trip sooner,
/// reducing regression-tail damage at the cost of fewer chances for slow-warmup
/// workloads to amortize spec overhead.
pub const RUNTIME_GATE_WARMUP: u64 = 5;

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
    if (attempted < RUNTIME_GATE_WARMUP) return false;
    if (drafts_per_round == 0) return false;
    const drafts_proposed = attempted * @as(u64, drafts_per_round);
    const rate = @as(f32, @floatFromInt(accepted)) /
        @as(f32, @floatFromInt(drafts_proposed));
    return rate < RUNTIME_GATE_MIN_PER_DRAFT_RATE;
}

// DFlash has different economics from sequential-token drafters: one
// assistant call proposes the whole block and the M5 NAX verify lane makes
// wide blocks cheap. Per-draft percentage therefore penalizes block 16
// even when it is decisively profitable. The stable cross-width signal is
// accepted drafts per verify round. M5 Muse sweeps put code/tool traffic at
// 4.4-15.0 accepted/round and the regressing prose/vision class at
// 1.0-1.5; two is the measured break-even boundary. M4 block-5 model-card
// workloads accepting 62-86% remain above it as well.
// Warmup 8, not 3: three rounds tripped on ONE dry opening
// (llmprobe's 16k rung: 0/3, drafter off for the whole request at 25
// tok/s; the same prompt restored from cache decoded at 73).
pub const DFLASH_GATE_WARMUP: u64 = 8;
pub const DFLASH_GATE_MIN_ACCEPTED_PER_ROUND: f32 = 2.0;
pub const DFLASH_THINKING_GATE_MIN_ACCEPTED_PER_ROUND: f32 = 1.0;
/// Absolute floor for a SPARSE target, applied after the width scaling.
/// See `scheduler.dflashGateMinimum` for the measurement it comes from —
/// a MoE verify reads every expert its block's positions route to, so its
/// break-even acceptance is several times a dense trunk's.
pub const DFLASH_MOE_GATE_MIN_ACCEPTED_PER_ROUND: f32 = 1.8;

/// The round policy (`dflash_policy.zig`) drives DFlash where the model declares it (`Caps.dflash_policy`); every
/// other model keeps the fixed block until it is A/B'd on its own.
pub fn dflashPolicyFor(config: *const model_mod.ModelConfig) bool {
    return config.caps().dflash_policy;
}

pub fn dflashGateShouldDisable(attempted: u64, accepted: u64, min_avg: f32) bool {
    if (attempted < DFLASH_GATE_WARMUP) return false;
    const avg = @as(f32, @floatFromInt(accepted)) /
        @as(f32, @floatFromInt(attempted));
    return avg < min_avg;
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
/// Steps of enabled-mode PLD before the yield gate may trip. Every warmup step pays PLD's UNPIPELINED cold
/// forward (plus a synchronous host read of the sampled token) against `next()`'s async-pipelined step, so the
/// tax is a fraction of the AR step cost; 32 was calibrated before the Laguna mscale fix made that step ~3x
/// cheaper. See the `yield-gate warmup is 8` test for the sweep.
pub const YIELD_GATE_WARMUP: u64 = 8;
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

/// Decode plain in the batched group. Sticky for DFlash (plain rounds do not extend the
/// assistant context); PLD's periodic re-enable check resumes once the slot is solo.
pub fn yieldSpecToCompany(self: *Generator, mode: []const u8) void {
    if (self.spec_disabled_runtime) return;
    log.info("  {s}=disabled (company: decoding plain in the batched group)\n", .{mode});
    self.spec_disabled_runtime = true;
    self.spec_disable_reason = .company;
}

/// DFlash runtime economics gate. Sticky within the request: once a
/// `DFLASH_GATE_WARMUP`-round sample proves the block-parallel path yields
/// less than its width-normalized request-class threshold, subsequent
/// ticks use the regular pipelined decoder through `nextDflash`'s entry
/// fallback.
pub fn checkDflashRuntimeGate(self: *Generator) void {
    if (self.spec_disabled_runtime or self.dflash_pol != null) return;
    if (!dflashGateShouldDisable(
        self.dflash_attempted,
        self.dflash_accepted_tokens,
        self.dflash_min_accepted_per_round,
    )) return;
    // The table's own measure outranks the calibrated bar: a round emitting
    // tokens cheaper than the plain step it would yield to stays on.
    if (self.xfm.round_cost.roundBeatsSerial(self.dflash_block_size -| 1, self.mtpKvLen()) == true) return;
    const avg = @as(f32, @floatFromInt(self.dflash_accepted_tokens)) /
        @as(f32, @floatFromInt(self.dflash_attempted));
    log.info(
        "  dflash=disabled (runtime yield {d:.2} accepted/round < {d:.2} after {d} attempts)\n",
        .{ avg, self.dflash_min_accepted_per_round, self.dflash_attempted },
    );
    self.spec_disabled_runtime = true;
}

/// Runtime acceptance gate for the drafter: after warmup, if the per-draft
/// acceptance probability is below `RUNTIME_GATE_MIN_PER_DRAFT_RATE`,
/// disable speculation for the rest of this request. Sticky for the rest
/// of the generation.
pub fn checkDrafterRuntimeGate(self: *Generator) void {
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

// ── tests ──


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

test "Generator.dflashGateShouldDisable uses accepted yield across block widths" {
    // Never decide from fewer than the DFlash warmup number of paid verifies.
    try testing.expect(!Generator.dflashGateShouldDisable(
        Generator.DFLASH_GATE_WARMUP - 1,
        0,
        Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND,
    ));

    // Muse prose/vision measured 1.0-1.5 accepted drafts per round at both
    // block 8 and block 16: that class loses to serial and must fall back.
    const w = Generator.DFLASH_GATE_WARMUP;
    try testing.expect(Generator.dflashGateShouldDisable(w, w * 1, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));
    try testing.expect(Generator.dflashGateShouldDisable(w, w * 2 - 1, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));

    // Exactly two is the strict break-even boundary; code/tool traffic at
    // 4.4+ and echo traffic near a full block remain on DFlash.
    try testing.expect(!Generator.dflashGateShouldDisable(w, w * 2, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));
    try testing.expect(!Generator.dflashGateShouldDisable(w, w * 5, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));
    try testing.expect(!Generator.dflashGateShouldDisable(w, w * 15, Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND));

    // Thinking preambles recovered from ~1.4 early to 4.4 whole-request; the
    // resolved-mode threshold keeps that path alive while still cutting off
    // a truly non-yielding reasoning request.
    try testing.expect(!Generator.dflashGateShouldDisable(w, w + w / 3, Generator.DFLASH_THINKING_GATE_MIN_ACCEPTED_PER_ROUND));
    try testing.expect(Generator.dflashGateShouldDisable(w, w - w / 3, Generator.DFLASH_THINKING_GATE_MIN_ACCEPTED_PER_ROUND));
}

test "Generator.yieldGateShouldDisable trips on cold-path-dominated workloads" {
    // The baseline regression: PLD forced on for a creative essay
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

test "yield-gate warmup is 8 — the cold-path tax tripled when the AR step got 3x cheaper" {
    // This constant is pure economics and the economics MOVED. Every warmup
    // step pays PLD's unpipelined cold forward plus a synchronous host read of
    // the sampled token, against `next()`'s async-pipelined step — an
    // ~absolute tax measured as a share of the AR step. Fixing the Laguna
    // mscale promotion made that AR step ~3x cheaper, so the same tax became a
    // ~3x larger share and 32 steps of it stopped being affordable.
    //
    // Swept on Laguna XS (one boot, serial vs unconstrained alternating per
    // request, server-reported timings, 5 runs median), against a matrix that
    // deliberately contains PLD's WIN case as well as its loss cases —
    // tuning on loss cases alone drives the warmup to zero and throws the win
    // away:
    //
    //   warmup   echo-edit   code-edit  free-form   explain      qa
    //       32     +76.9%       -4.3%      -1.2%     -1.4%     -7.2%
    //       16     +70.1%       -3.4%      -0.4%     -0.3%     -0.8%
    //        8     +77.4%       +0.3%      -0.2%     -0.3%     -1.5%
    //        4     +76.6%       +3.9%      -0.2%     -0.1%     -1.7%
    //
    // 8 recovers essentially the whole loss while leaving the +77% untouched.
    // 4 measured no worse, but a gate that decides on four observations is
    // fitting noise, and a short preamble before a file echo is the NORMAL
    // agent shape — exactly what a too-eager trip would punish. A premature
    // trip is bounded anyway: `specShouldReenable` re-checks every
    // SPEC_REENABLE_INTERVAL steps.
    try testing.expectEqual(@as(u64, 8), Generator.YIELD_GATE_WARMUP);

    // Sweepable without a rebuild, same contract as `runtimeGateWarmup`:
    // out-of-range values fall back rather than silently disabling the gate.
    try testing.expectEqual(Generator.YIELD_GATE_WARMUP, YIELD_GATE_WARMUP);
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
