//! MTP regime gate: dry-extension, live costs, two-chunk regime choice.
const depth_bounds = @import("../mtp_depth_bounds.zig");
const std = @import("std");
const env = @import("../../../../env.zig");
const io_util = @import("../../../../sdk/util/io_util.zig");
const testing = std.testing;
const Generator = @import("../../generate/generator.zig").Generator;
const MtpRoundPlan = @import("ev_plan.zig").MtpRoundPlan;


/// Extension dry-spell gate constants: after MTP_EXT_DRY_ROUNDS
/// consecutive extension-CONSIDERED rounds whose confidence gate never
/// cleared, consideration collapses to single-chunk for
/// MTP_EXT_DRY_COOLDOWN rounds, then re-opens for a fresh trial.
/// Rationale (post-round-pipelining sweep): a two-chunk
/// round pays a REAL mid-pipeline cost the EV surface can't see — the
/// chunk-A boundary sync fires ~0.3 ms after the pre-draft dispatch, so
/// it blocks on the still-running head chain + confidence graphs
/// (measured sync 0.5-3.7 ms/round at ext_rate 0.00; EV default lost
/// 3-6 tok/s at 0.5K to fixed depths from this tax alone). The gate is
/// fed by the REALIZED extension rate, never by priors — the war-story
/// failure mode (docs/gotchas/engine-mlx.md, ext_rounds=0 on echo) was
/// a prior-fed expected-gain gate blocking the FIRST trial; this one
/// guarantees a fresh 16-round trial every 48 rounds, and a single
/// extension firing resets the streak entirely. Worst-case waste on a
/// permanently-dry workload: a third of rounds pay the sync. The
/// cooldown is deliberately SHORT relative to a typical request (a
/// 160-token echo ≈ 70 rounds): a 64-round blackout swallowed the whole
/// echoing stretch after a dry preamble and re-created the very
/// ext_rounds=0 regression this design avoids (caught by
/// tests/test_mtp_equivalence.sh).
pub const MTP_EXT_DRY_ROUNDS: u32 = 16;
pub const MTP_EXT_DRY_COOLDOWN: u32 = 32;
/// Floor for the cost-aware dry threshold. Even the most expensive sync
/// tolerates this many dry considered rounds before a cooldown, so the
/// worst-case fresh-trial cadence (MTP_EXT_DRY_MIN + MTP_EXT_DRY_COOLDOWN)
/// still fits inside a typical echo stretch (~70 rounds) — the horizon is
/// never closed regardless of measured cost.
pub const MTP_EXT_DRY_MIN: u32 = 3;
/// Tolerated cumulative chunk-A sync per dry exploration burst, expressed
/// as a fraction of one round. The cost-aware threshold is
/// SYNC_BUDGET / (sync_ms/round_ms): a costlier sync backs off sooner, a
/// near-free one keeps the full MTP_EXT_DRY_ROUNDS budget. Calibrated so
/// the measured 8K operating point (sync ~2.4 ms of a ~45 ms round,
/// frac ~0.053) lands near a ~15% exploration duty (threshold ~6, cooldown
/// 32) — matching oMLX's duty-bounded probing.
pub const MTP_EXT_SYNC_BUDGET: f32 = 0.30;

/// Cost-aware dry-explore threshold (consecutive dry considered rounds
/// tolerated before a cooldown), derived from the LIVE-measured sync
/// fraction `sync_ms / round_ms`. Bounded to
/// [MTP_EXT_DRY_MIN, MTP_EXT_DRY_ROUNDS] so a fresh trial always fits
/// inside an echo stretch. Unmeasured (either EMA still 0) keeps the fixed
/// budget — no behavior change until the live cost is observed.
pub fn mtpExtDryThresholdFor(sync_ms: f32, round_ms: f32) u32 {
    if (sync_ms <= 0.0 or round_ms <= 0.0) return MTP_EXT_DRY_ROUNDS;
    const frac = sync_ms / round_ms; // measured exploration cost fraction
    if (frac <= 0.0) return MTP_EXT_DRY_ROUNDS;
    const max_f: f32 = @floatFromInt(MTP_EXT_DRY_ROUNDS);
    // Tolerate ~SYNC_BUDGET round-times of accumulated sync per dry burst:
    // a costlier sync (larger frac) yields fewer tolerated dry rounds.
    const raw = @min(MTP_EXT_SYNC_BUDGET / frac, max_f);
    const rounded: u32 = @intFromFloat(@round(@max(0.0, raw)));
    return @min(MTP_EXT_DRY_ROUNDS, @max(MTP_EXT_DRY_MIN, rounded));
}

/// Pure dry-spell policy step, called once per post-warmup EV round
/// whose plan considers extension. Returns whether the two-chunk shape
/// is allowed this round; mutates the streak/cooldown state. The caller
/// bumps `streak` on a considered-but-not-extended round and zeroes it
/// when extension fires. `threshold` is the cost-aware dry limit
/// (mtpExtDryThresholdFor) — the live-cost lever that shortens dry
/// exploration bursts when the measured sync is expensive.
pub fn mtpExtDryAllows(dry_streak: *u32, cooldown: *u32, threshold: u32) bool {
    if (cooldown.* > 0) {
        cooldown.* -= 1;
        return false;
    }
    if (dry_streak.* >= threshold) {
        cooldown.* = MTP_EXT_DRY_COOLDOWN - 1;
        dry_streak.* = 0;
        return false;
    }
    return true;
}

/// EMA weight for the live round/sync cost EMAs. Slower than the
/// acceptance EMA (cost drifts with context/thermal, not content) and
/// paired with the seed-on-first-sample rule so warmup fills them cleanly.
pub const MTP_EV_COST_BETA: f32 = 0.10;

/// Fold a nanosecond wall-time sample into a millisecond EMA. Seeds on the
/// first sample (prev <= 0). Pure — the live-cost measurement plumbing is
/// just Stopwatch reads feeding this.
pub fn mtpEmaMs(prev_ms: f32, sample_ns: u64) f32 {
    const sample_ms = @as(f32, @floatFromInt(sample_ns)) / 1.0e6;
    if (prev_ms <= 0.0) return sample_ms;
    return prev_ms + MTP_EV_COST_BETA * (sample_ms - prev_ms);
}

/// Measured ms-per-emitted-token of the two round SHAPES the EV plan can
/// take: a two-chunk plan (extension considered, sync paid whether or not
/// it fires) against a single-chunk plan at m_lo. Throughput is tokens
/// over round cost, and both halves are observable per round — no cost
/// surface, no units. The hand-fitted marginals can only say whether the
/// extension's VERIFY pays; they cannot see that on an M4 base the whole
/// two-chunk round runs 17.4 ms/token against 14.4 single (cap 5+ lost
/// 17% to cap 4 on echo) while on an M4 Max the same shape is a wash.
pub const MtpRegime = struct {
    two_ms: f32 = 0,
    two_tok: f32 = 0,
    two_m: u32 = 0,
    one_ms: f32 = 0,
    one_tok: f32 = 0,
    one_m: u32 = 0,
    /// Shape of the previous round: a round whose shape differs from its
    /// predecessor is a TRANSITION and is not observed. The minority shape
    /// was only ever measured on transition rounds and read 5-7% slow
    /// (M4 Max 27B @16k interleaved 13.65 vs 12.9 ms/tok; homogeneous arms
    /// 13.0 vs 13.2) — the verify width change is a one-off cost.
    last_two: ?bool = null,
    /// Trial schedule: the minority shape runs for rounds [trial_start,
    /// trial_end) and the next block starts at next_trial. Explicit, not
    /// `idx % period` — the period moves with the EMAs, and a block's own
    /// observation moved it by one per round so `idx % period` stayed
    /// inside the block: M4 base v4 ran 14 of 72 rounds as trials
    /// against 7 on v3 and lost 2-4%.
    sched_verdict: ?bool = null,
    trial_end: u32 = 0,
    next_trial: u32 = 0,
    /// Diagnostics for `[spec-stats]`: the round the first verdict formed
    /// at, and trial blocks started — splits ext_rounds into pre-verdict
    /// (seeding) and scheduled exploration.
    verdict_round: u32 = 0,
    trials: u32 = 0,
    /// Idempotency: `mtpRegimeForce` mutates the schedule, and
    /// `mtpRoundPlan` has two call sites (pre-draft at the previous
    /// round's tail, or the round's own entry). Asking twice for the
    /// same round answers the same and advances nothing.
    last_idx: ?u32 = null,
    last_force: ?bool = null,
};

/// The worse regime still runs one round in this many (at least) so its
/// EMAs keep tracking the live workload (context grows, thermals move).
pub const MTP_REGIME_EXPLORE_PERIOD: u32 = 8;
/// Exploration drag the period is sized to: a shape G worse than the
/// other, run once in G/DRAG rounds, costs ~DRAG of throughput. M4 base
/// measured two-chunk 26% worse; at 1-in-8 that was a 2% structural
/// drag that kept cap 5 from cap-4 parity (52.3 vs 54.6).
pub const MTP_REGIME_EXPLORE_DRAG: f32 = 0.01;
pub const MTP_REGIME_EXPLORE_PERIOD_MAX: u32 = 128;
/// A trial is a BLOCK of consecutive rounds: the first is the transition
/// (unobserved), the second is the steady-state measurement.
pub const MTP_REGIME_EXPLORE_BLOCK: u32 = 2;

pub fn regimeEma(prev: f32, sample: f32) f32 {
    if (prev <= 0.0) return sample;
    return prev + MTP_EV_COST_BETA * (sample - prev);
}

/// Wall time since the previous round ended (the first round reads its
/// own stopwatch), so inter-round work is charged to the shape.
pub fn mtpRegimeWallMs(self: *Generator, round_ms: f32) f32 {
    if (self.mtp_regime_clock) |*c| {
        const ns = c.read();
        c.reset();
        return @as(f32, @floatFromInt(ns)) / @as(f32, std.time.ns_per_ms);
    }
    self.mtp_regime_clock = io_util.Stopwatch.init(self.timer.io);
    return round_ms;
}

/// Both shapes are compared at the SAME base depth: a shape observed at
/// a new m_lo reseeds (the warmup climb's depth-1..3 rounds read 19
/// ms/tok against 11 for a two-chunk round at m_lo 4 — not a comparison).
pub fn mtpRegimeObserve(r: *MtpRegime, two_chunk: bool, m_lo: u32, round_ms: f32, tokens: f32) void {
    if (round_ms <= 0.0 or tokens <= 0.0) return;
    // The first observed round has no predecessor to transition from
    // (dropping it cost one extra pre-verdict extension round, the whole
    // of the M1 Pro 27B's -2.6% on v5.2).
    const transition = r.last_two != null and r.last_two.? != two_chunk;
    r.last_two = two_chunk;
    if (transition) return;
    if (two_chunk) {
        if (r.two_m != m_lo) {
            r.two_ms = 0;
            r.two_tok = 0;
            r.two_m = m_lo;
        }
        r.two_ms = regimeEma(r.two_ms, round_ms);
        r.two_tok = regimeEma(r.two_tok, tokens);
    } else {
        if (r.one_m != m_lo) {
            r.one_ms = 0;
            r.one_tok = 0;
            r.one_m = m_lo;
        }
        r.one_ms = regimeEma(r.one_ms, round_ms);
        r.one_tok = regimeEma(r.one_tok, tokens);
    }
}

/// The minority shape is measured from interleaved rounds and reads a
/// few percent slow (M4 Max 27B @16k: two-chunk 13.4 ms/tok interleaved
/// vs 13.0 homogeneous), so a throttle needs a margin the noise cannot
/// cross; the M4 base loss it exists for is 21%.
pub const MTP_REGIME_MARGIN: f32 = 0.05;

/// Null until BOTH shapes have been measured at the same base depth.
pub fn mtpRegimeTwoChunkWorse(r: MtpRegime) ?bool {
    return mtpRegimeVerdict(r, null);
}

/// Verdict with HYSTERESIS: a standing "worse" only flips to "better"
/// once two-chunk is at or below single, not merely inside the margin.
/// The majority shape's first observed round after a trial block is
/// still elevated (M1 Pro 9B: single 23.8 steady, 24.6-24.9 after a
/// block), which pulled the ratio inside the margin and flipped the
/// verdict 5-7 times per boot on v5.2, each flip a run of two-chunk
/// rounds on a box where that shape loses 10%.
pub fn mtpRegimeVerdict(r: MtpRegime, prev: ?bool) ?bool {
    if (r.two_tok <= 0.0 or r.one_tok <= 0.0 or r.two_m != r.one_m) return null;
    const ratio = (r.two_ms / r.two_tok) / (r.one_ms / r.one_tok);
    if (prev == true) return ratio > 1.0;
    return ratio > 1.0 + MTP_REGIME_MARGIN;
}

/// Rounds between trials of the worse shape, from the measured gap.
pub fn mtpRegimeExplorePeriod(r: MtpRegime) u32 {
    if (r.two_tok <= 0.0 or r.one_tok <= 0.0) return MTP_REGIME_EXPLORE_PERIOD;
    const two = r.two_ms / r.two_tok;
    const one = r.one_ms / r.one_tok;
    const gap = @abs(two - one) / @min(two, one);
    const block: f32 = @floatFromInt(MTP_REGIME_EXPLORE_BLOCK);
    const p: u32 = @intFromFloat(@ceil(block * gap / MTP_REGIME_EXPLORE_DRAG));
    return @min(MTP_REGIME_EXPLORE_PERIOD_MAX, @max(MTP_REGIME_EXPLORE_PERIOD, p));
}

/// Which shape this round runs: null = as the plan wrote it (two-chunk),
/// false = single-chunk. The plan's own default is two-chunk, so that
/// side measures itself; an unmeasured SINGLE is tried at once (v2
/// compared one_m to a two_m only a two-chunk round can set, and pinned
/// single forever). Once both are measured the worse shape runs as a
/// trial BLOCK every period so its EMA keeps tracking the workload; a
/// single-chunk trial is what gets it measured at all when the horizon
/// opens on every round (pure echo never plans one by itself).
pub fn mtpRegimeForce(r: *MtpRegime, round_idx: u32) ?bool {
    if (r.last_idx == round_idx) return r.last_force;
    const force = mtpRegimeForceAt(r, round_idx);
    r.last_idx = round_idx;
    r.last_force = force;
    return force;
}

pub fn mtpRegimeForceAt(r: *MtpRegime, round_idx: u32) ?bool {
    if (r.two_tok <= 0.0) return null;
    if (r.one_tok <= 0.0 or r.one_m != r.two_m) return false;
    const worse = mtpRegimeVerdict(r.*, r.sched_verdict) orelse return null;
    if (r.sched_verdict != worse) {
        if (r.sched_verdict == null) r.verdict_round = round_idx;
        r.sched_verdict = worse;
        r.trial_end = 0;
        r.next_trial = round_idx + mtpRegimeExplorePeriod(r.*);
    }
    const minority: ?bool = if (worse) null else false;
    const majority: ?bool = if (worse) false else null;
    if (round_idx < r.trial_end) return minority;
    if (round_idx >= r.next_trial) {
        r.trials += 1;
        r.trial_end = round_idx + MTP_REGIME_EXPLORE_BLOCK;
        r.next_trial = r.trial_end + mtpRegimeExplorePeriod(r.*);
        return minority;
    }
    return majority;
}

/// Per-round draft plan. Fixed mode (and EV warmup): today's adaptive
/// depth, no extension — byte-identical round shape to the legacy path.
/// Post-warmup EV mode: the pure plan over the acceptance EMAs, with the
/// base-depth climb damped to one step per round and two-chunk plans
/// gated by the extension dry-spell policy.
/// DIAGNOSTIC (--mtp-min-depth n --mtp-max-depth n): every round drafts exactly
/// n, the EV/windowed controllers never demote or disable — the
/// per-index acceptance meter (`acc_idx=` on the trace line).
pub fn mtpPinnedDepth() ?u32 {
    return depth_bounds.active.pinned();
}

/// A plan held inside the depth range: the EV argmax may sit outside it.
pub fn mtpPlanInBounds(plan: MtpRoundPlan, lo: u32, hi: u32) MtpRoundPlan {
    var p = plan;
    p.m_lo = depth_bounds.clampWidth(p.m_lo, lo, hi);
    p.m_hi = depth_bounds.clampWidth(p.m_hi, p.m_lo, hi);
    if (p.m_hi == p.m_lo and plan.m_hi != plan.m_lo) p.tau_ln = 0.0;
    return p;
}

// Adaptive serial: the EV controller picks the best depth but never compares a round with
// the serial step it replaces, and the acceptance floor is a model fitted where a verify
// row costs ~0.1 of a forward. On qwen4_exp a verify row is bytes (62.7k prose: 47-58 tok/s
// speculative vs 55 serial). So the two measured prices are compared per KV bucket:
//     planned ms/token = roundMs(m_lo) / E[tokens per round at m_lo]
//     serial  ms/token = the bucket's measured plain-decode token
// Any missing input is `.undecided`. The only cost a request may pay is a bounded serial
// probe for a bucket that has never seen a serial token.

/// A round worse than serial by less than this is a tie.
pub const MTP_ADAPTIVE_MARGIN: f32 = 0.05;
/// Consecutive rounds that must vote serial.
pub const MTP_ADAPTIVE_CONFIRM: u32 = 3;
/// Serial ticks one probe runs; `MTP_ADAPTIVE_PROBE_WARM` of them are discarded.
pub const MTP_ADAPTIVE_PROBE_TOKENS: u32 = 8;

// ── tests ──


test "mtpRegime: the worse shape runs as a scheduled trial block, unmeasured runs as planned" {
    var r = Generator.MtpRegime{};
    // Unseeded: the plan stands (two-chunk measures itself); once two-chunk
    // is measured, an unmeasured single is forced AT ONCE. The first round
    // of a shape is its transition and is not observed.
    try testing.expect(Generator.mtpRegimeTwoChunkWorse(r) == null);
    try testing.expect(Generator.mtpRegimeForce(&r, 1) == null);
    Generator.mtpRegimeObserve(&r, true, 4, 129.0, 5.65); // first round counts
    try testing.expect(Generator.mtpRegimeForce(&r, 2) == false);
    // M4 base echo (measured): two-chunk 22.83 vs single 18.11 ms/tok = 26%
    // worse, so a 2-round trial block recurs every 53 rounds (1% drag). The
    // single's first round is a transition (dropped), the second seeds.
    Generator.mtpRegimeObserve(&r, false, 4, 90.5, 5.0);
    try testing.expect(Generator.mtpRegimeForce(&r, 3) == false);
    Generator.mtpRegimeObserve(&r, false, 4, 90.5, 5.0);
    try testing.expect(Generator.mtpRegimeTwoChunkWorse(r).?);
    try testing.expectEqual(@as(u32, 53), Generator.mtpRegimeExplorePeriod(r));
    try testing.expect(Generator.mtpRegimeForce(&r, 14) == false); // verdict forms: next trial at 67
    try testing.expect(Generator.mtpRegimeForce(&r, 66) == false);
    try testing.expect(Generator.mtpRegimeForce(&r, 67) == null); // block
    try testing.expect(Generator.mtpRegimeForce(&r, 68) == null);
    try testing.expect(Generator.mtpRegimeForce(&r, 69) == false); // and not a third
    try testing.expectEqual(@as(u32, 69 + 53), r.next_trial);
    try testing.expectEqual(@as(u32, 14), r.verdict_round);
    try testing.expectEqual(@as(u32, 1), r.trials);
    // Asked twice for the same round (pre-draft + entry): same answer, no
    // schedule drift.
    try testing.expect(Generator.mtpRegimeForce(&r, 122) == null);
    try testing.expect(Generator.mtpRegimeForce(&r, 122) == null);
    try testing.expectEqual(@as(u32, 2), r.trials);
    try testing.expectEqual(@as(u32, 124 + 53), r.next_trial);
    // Two-chunk 64 ms for 6 (10.7) against single 56 for 4.96 (11.3):
    // two-chunk runs every round and a single-chunk block is scheduled once
    // per period to keep the other regime's EMA alive.
    var m = Generator.MtpRegime{};
    Generator.mtpRegimeObserve(&m, true, 4, 64.0, 6.0);
    Generator.mtpRegimeObserve(&m, false, 4, 56.0, 4.96);
    Generator.mtpRegimeObserve(&m, false, 4, 56.0, 4.96);
    try testing.expect(!Generator.mtpRegimeTwoChunkWorse(m).?);
    try testing.expectEqual(@as(u32, 12), Generator.mtpRegimeExplorePeriod(m)); // 5.8% gap
    try testing.expect(Generator.mtpRegimeForce(&m, 5) == null); // verdict forms: next trial at 17
    try testing.expect(Generator.mtpRegimeForce(&m, 16) == null);
    try testing.expect(Generator.mtpRegimeForce(&m, 17) == false);
    try testing.expect(Generator.mtpRegimeForce(&m, 18) == false);
    try testing.expect(Generator.mtpRegimeForce(&m, 19) == null);
    // M4 Max 27B @16k: 13.4 vs 12.95 ms/tok is inside the margin — the plan
    // stands (two-chunk measured +3.7% at arm level).
    var n = Generator.MtpRegime{};
    Generator.mtpRegimeObserve(&n, true, 4, 80.4, 6.0);
    Generator.mtpRegimeObserve(&n, false, 4, 64.75, 5.0);
    Generator.mtpRegimeObserve(&n, false, 4, 64.75, 5.0);
    try testing.expect(!Generator.mtpRegimeTwoChunkWorse(n).?);
    // Hysteresis (M1 Pro 9B v5.2): a standing "worse" at 26.4 vs 23.8 does
    // not flip when the post-block single reads 25.3 (ratio 1.04, inside the
    // margin); it flips only once two-chunk is at or below single.
    var h = Generator.MtpRegime{ .two_ms = 26.4, .two_tok = 1.0, .two_m = 4, .one_ms = 25.3, .one_tok = 1.0, .one_m = 4 };
    try testing.expect(Generator.mtpRegimeVerdict(h, true).?);
    try testing.expect(!Generator.mtpRegimeVerdict(h, null).?);
    h.one_ms = 26.5;
    try testing.expect(!Generator.mtpRegimeVerdict(h, true).?);
}

test "mtpRegime: a simulated round loop reaches BOTH shapes, and the worse one keeps being re-tried" {
    // Live (M4 base): v2's "try the unmeasured single shape at
    // once" rule compared one_m against a two_m that only a two-chunk round
    // can set, so it forced single-chunk forever — caps 5/6 reported cap-4
    // numbers with ext_rounds=0 and no verdict line, which reads as a PASS
    // on an echo workload. v4's `idx % period` gate then ran trial chains
    // because the block's own observation moved the period (14 of 72 rounds
    // as trials against 7). The gate is driven here exactly as mtpRoundPlan
    // drives it: ask, run that shape, feed the observation.
    const Sim = struct {
        fn run(two_ms: f32, one_ms: f32, rounds: u32) struct { two: u32, one: u32 } {
            var r = Generator.MtpRegime{};
            var two: u32 = 0;
            var one: u32 = 0;
            var i: u32 = 0;
            while (i < rounds) : (i += 1) {
                // The plan always offers two-chunk (pure echo); the gate decides.
                const two_chunk = Generator.mtpRegimeForce(&r, i) orelse true;
                if (two_chunk) two += 1 else one += 1;
                Generator.mtpRegimeObserve(&r, two_chunk, 4, if (two_chunk) two_ms else one_ms, if (two_chunk) 6.0 else 5.0);
            }
            return .{ .two = two, .one = one };
        }
    };
    // M4 base: two-chunk 26% worse. Pre-verdict 4 rounds, then 2-round
    // blocks every 53: 4 + 2*3 = 10 two-chunk rounds in 200, not 20.
    const worse = Sim.run(129.0, 90.5, 200);
    try testing.expect(worse.one > 185);
    try testing.expect(worse.two >= 6 and worse.two <= 12);
    // M4 Max: two-chunk better (5.8% gap): 2-round single blocks every 12.
    const better = Sim.run(64.0, 56.0, 200);
    try testing.expect(better.two > 160);
    try testing.expect(better.one >= 25 and better.one <= 40);
}

test "mtpRegimeObserve: seeds on the first sample, moves by the cost beta, reseeds on a new base depth" {
    var r = Generator.MtpRegime{};
    Generator.mtpRegimeObserve(&r, true, 3, 45.0, 4.0); // first round counts
    try testing.expect(r.two_tok > 0.0);
    Generator.mtpRegimeObserve(&r, false, 3, 45.0, 4.0); // transition: dropped
    try testing.expect(r.one_tok == 0.0);
    Generator.mtpRegimeObserve(&r, false, 3, 50.0, 4.0);
    try testing.expectApproxEqAbs(@as(f32, 50.0), r.one_ms, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 4.0), r.one_tok, 1e-6);
    Generator.mtpRegimeObserve(&r, false, 3, 60.0, 5.0);
    try testing.expectApproxEqAbs(50.0 + Generator.MTP_EV_COST_BETA * 10.0, r.one_ms, 1e-4);
    try testing.expectApproxEqAbs(4.0 + Generator.MTP_EV_COST_BETA * 1.0, r.one_tok, 1e-4);
    // The climb moved m_lo: the depth-3 rounds are not the depth-4 regime.
    Generator.mtpRegimeObserve(&r, false, 4, 70.0, 5.0);
    try testing.expectApproxEqAbs(@as(f32, 70.0), r.one_ms, 1e-6);
    // A two-chunk reading at a different base depth yields no verdict.
    Generator.mtpRegimeObserve(&r, true, 3, 80.0, 6.0);
    Generator.mtpRegimeObserve(&r, true, 3, 80.0, 6.0);
    try testing.expect(Generator.mtpRegimeTwoChunkWorse(r) == null);
}

test "mtpExtDryAllows: a dry streak collapses to single-chunk, cooldown expires into a fresh trial" {
    var streak: u32 = 0;
    var cooldown: u32 = 0;
    const thr = Generator.MTP_EXT_DRY_ROUNDS;
    // Considering rounds below the streak threshold stay allowed.
    var i: u32 = 0;
    while (i < thr - 1) : (i += 1) {
        try testing.expect(Generator.mtpExtDryAllows(&streak, &cooldown, thr));
        streak += 1; // caller records "considered, did not extend"
    }
    // The full dry streak trips the cooldown: consideration collapses.
    streak = thr;
    try testing.expect(!Generator.mtpExtDryAllows(&streak, &cooldown, thr));
    try testing.expectEqual(@as(u32, 0), streak);
    // Cooldown rounds stay collapsed and count down.
    var blocked: u32 = 0;
    while (!Generator.mtpExtDryAllows(&streak, &cooldown, thr)) blocked += 1;
    try testing.expectEqual(Generator.MTP_EXT_DRY_COOLDOWN - 1, blocked);
    // After the cooldown, consideration re-opens (fresh trial, streak 0).
    try testing.expectEqual(@as(u32, 0), streak);
    try testing.expect(Generator.mtpExtDryAllows(&streak, &cooldown, thr));
    // An extension firing resets the streak (caller does streak = 0) — the
    // gate never trips on workloads where extension pays.
}

test "mtpExtDryAllows: a lower cost-aware threshold trips the cooldown sooner" {
    var streak: u32 = 0;
    var cooldown: u32 = 0;
    const thr: u32 = 4; // e.g. an expensive sync throttled hard
    var i: u32 = 0;
    while (i < thr - 1) : (i += 1) {
        try testing.expect(Generator.mtpExtDryAllows(&streak, &cooldown, thr));
        streak += 1;
    }
    streak = thr;
    try testing.expect(!Generator.mtpExtDryAllows(&streak, &cooldown, thr));
    try testing.expectEqual(@as(u32, 0), streak);
    // The fresh trial still fits inside an echo stretch even at the floor.
    try testing.expect(Generator.MTP_EXT_DRY_MIN + Generator.MTP_EXT_DRY_COOLDOWN < 70);
}

test "mtpExtDryThresholdFor: cost-aware dry threshold scales inversely with the measured sync fraction" {
    const D = Generator;
    // Unmeasured (either EMA still 0) keeps the fixed dry budget — no behavior
    // change until the live cost is known.
    try testing.expectEqual(D.MTP_EXT_DRY_ROUNDS, D.mtpExtDryThresholdFor(0.0, 45.0));
    try testing.expectEqual(D.MTP_EXT_DRY_ROUNDS, D.mtpExtDryThresholdFor(2.4, 0.0));
    // A near-free sync (tiny fraction) tolerates the full dry budget.
    try testing.expectEqual(D.MTP_EXT_DRY_ROUNDS, D.mtpExtDryThresholdFor(0.2, 45.0));
    // An expensive sync (large fraction) collapses toward the floor.
    try testing.expectEqual(D.MTP_EXT_DRY_MIN, D.mtpExtDryThresholdFor(9.0, 45.0));
    // Monotonic: a costlier sync never tolerates MORE dry rounds.
    try testing.expect(D.mtpExtDryThresholdFor(4.0, 45.0) <= D.mtpExtDryThresholdFor(1.0, 45.0));
    // Always bounded — never below the floor (a zero threshold would trip
    // every round and close the horizon; the fresh-trial guarantee survives
    // any cost).
    try testing.expect(D.mtpExtDryThresholdFor(1000.0, 45.0) >= D.MTP_EXT_DRY_MIN);
    // The measured 8K operating point (sync ~2.4 ms of a ~45 ms round) lands
    // in the throttled band, well below the fixed 16.
    const at_8k = D.mtpExtDryThresholdFor(2.4, 45.0);
    try testing.expect(at_8k >= D.MTP_EXT_DRY_MIN and at_8k < D.MTP_EXT_DRY_ROUNDS);
}

test "mtpEmaMs: seeds on the first sample then folds nanoseconds into a ms EMA" {
    const D = Generator;
    // First sample seeds (prev <= 0): 45 ms == 45_000_000 ns, exactly.
    try testing.expectApproxEqAbs(@as(f32, 45.0), D.mtpEmaMs(0.0, 45_000_000), 1e-3);
    // A subsequent sample folds at MTP_EV_COST_BETA toward the new value.
    const folded = D.mtpEmaMs(45.0, 40_000_000); // 40 ms sample
    try testing.expectApproxEqAbs(45.0 + D.MTP_EV_COST_BETA * (40.0 - 45.0), folded, 1e-3);
}
