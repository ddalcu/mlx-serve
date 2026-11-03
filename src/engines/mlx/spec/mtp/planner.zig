//! MTP round planner entry points, width trials and the EV round update.
const depth_bounds = @import("../mtp_depth_bounds.zig");
const std = @import("std");
const log = @import("../../../../sdk/util/log.zig");
const mtp_mod = @import("../../../../sdk/spec/mtp/mtp.zig");
const round_cost = @import("../round_cost.zig");
const group_cost = @import("../mtp_group_cost.zig");
const group_planner = @import("../mtp_group_planner.zig");
const testing = std.testing;
const Generator = @import("../../generate/generator.zig").Generator;
const MTP_DEPTH_WINDOW = @import("depth.zig").MTP_DEPTH_WINDOW;
const MTP_DISABLE_BELOW = @import("depth.zig").MTP_DISABLE_BELOW;
const MTP_EV_EMA_BETA = @import("depth.zig").MTP_EV_EMA_BETA;
const MTP_EV_WARMUP_ROUNDS = @import("depth.zig").MTP_EV_WARMUP_ROUNDS;
const MTP_EXT_DRY_ROUNDS = @import("regime.zig").MTP_EXT_DRY_ROUNDS;
const MtpCostSource = @import("ev_plan.zig").MtpCostSource;
const MtpRoundPlan = @import("ev_plan.zig").MtpRoundPlan;
const MtpTrace = @import("trace.zig").MtpTrace;
const mtpBasePlan = @import("ev_plan.zig").mtpBasePlan;
const mtpDepthPolicy = @import("ev_plan.zig").mtpDepthPolicy;
const mtpDepthPolicyFor = @import("ev_plan.zig").mtpDepthPolicyFor;
const mtpEvExpectedTokens = @import("ev_plan.zig").mtpEvExpectedTokens;
const mtpEvObserve = @import("ev_plan.zig").mtpEvObserve;
const mtpEvTestGenerator = @import("ev_plan.zig").mtpEvTestGenerator;
const mtpExtDryAllows = @import("regime.zig").mtpExtDryAllows;
const mtpExtDryThresholdFor = @import("regime.zig").mtpExtDryThresholdFor;
const mtpPinnedDepth = @import("regime.zig").mtpPinnedDepth;
const mtpPlanInBounds = @import("regime.zig").mtpPlanInBounds;
const mtpRegimeExplorePeriod = @import("regime.zig").mtpRegimeExplorePeriod;
const mtpRegimeForce = @import("regime.zig").mtpRegimeForce;
const mtpTraceEnabled = @import("trace.zig").mtpTraceEnabled;


/// One debug line per planned round with every input the planner read.
pub fn mtpPlannerMode(self: *const Generator) u8 {
    if (self.mtp == null) return 0;
    const sharp = self.mtpDraftSampling(0).temperature > 0.01;
    return group_cost.GroupShape.samplingMode(self.mtpRerankDrafts(), false, sharp, self.sampling.temperature > 0.01);
}

pub fn mtpPlannerObserve(self: *Generator, observed: u32, accepted: u32) void {
    if (!group_planner.enabled() or !self.mtp_planner_owned) return;
    self.mtp_planner_history.observe(observed, @min(observed, accepted));
    self.mtp_planner_spec_rounds += 1;
    self.mtp_hidden_stale = false;
    self.mtp_planner_recovery.observe(self.completion_tokens, self.mtp_planner_recovering);
    if (self.mtp_planner_probe and !self.mtp_planner_recovering) self.mtp_planner_probes +|= 1;
}

pub fn mtpRoundPlan(self: *Generator) MtpRoundPlan {
    if (group_planner.enabled()) if (self.mtp_planner_width) |width| {
        return .{ .m_lo = width, .m_hi = width, .tau_ln = 0 };
    };
    const plan = self.mtpRoundPlanTraced();
    if (self.mtp_group_cap > 0) {
        const cap = @min(depth_bounds.lookupCap(self.mtp_group_cap, depth_bounds.active), if (self.mtp) |h| h.maxDepth() else mtp_mod.MAX_DEPTH);
        return mtpPlanInBounds(mtpGroupPlan(plan, cap, self.mtp_group_fill and mtpPinnedDepth() == null), depth_bounds.floorFor(depth_bounds.active, cap), cap);
    }
    return plan;
}

/// A lane's plan inside a batched verify. The group pays for its widest lane's rows,
/// so where those rows are one fixed tile (`fill`) every lane drafts to the cap.
pub fn mtpGroupPlan(solo: MtpRoundPlan, cap: u32, fill: bool) MtpRoundPlan {
    if (fill) return .{ .m_lo = cap, .m_hi = cap, .tau_ln = 0 };
    var plan = solo;
    plan.m_lo = @min(plan.m_lo, cap);
    plan.m_hi = @min(plan.m_hi, cap);
    return plan;
}

/// Round time the plan prices depth `m` at, in ms.
pub fn planMs(src: MtpCostSource, m: u32) f32 {
    return src.roundCost(m, false) / src.scale;
}

pub fn mtpRoundPlanTraced(self: *Generator) MtpRoundPlan {
    if (!log.isDebug()) return self.mtpRoundPlanInner();
    const disabled_before = self.spec_disabled_runtime;
    const serial_left_before = self.mtp_serial_left;
    const plan = self.mtpRoundPlanInner();
    const kv_len = self.mtpKvLen();
    const t = &self.xfm.round_cost;
    const b = self.mtpAdaptiveBucket(kv_len);
    const src = MtpCostSource.init(self.mtp_ev_costs, kv_len, t);
    const serial_step: []const u8 = if (!disabled_before and self.spec_disabled_runtime)
        "switch"
    else if (self.mtp_serial_left > serial_left_before)
        "probe"
    else
        "none";
    var cells: [256]u8 = undefined;
    const a = self.mtp_ev_accept;
    const cap_row: u32 = @min(@max(@as(u32, 1), self.mtp_depth), mtp_mod.MAX_DEPTH);
    const cap_free: u32 = @min(@max(cap_row, self.mtp_depth_free), if (self.mtp) |h| h.maxDepth() else mtp_mod.MAX_DEPTH);
    const target = mtpWidthTrialTarget(t, kv_len, plan, cap_free, self.mtp_m_lo_streak >= 2);
    log.debug(
        "  [mtp-plan] r={d} kv={d} bucket={s} m_lo={d} m_hi={d} tau={d:.3} a=[{d:.2},{d:.2},{d:.2},{d:.2},{d:.2},{d:.2}] src={s} trial={s}{d} serial_step={s} streak={d} serial_left={d} drops=t{d}/c{d}/b{d} cap={d}/{d} solo={} rte={d} wt={d}/{d} tgt={?d} cells={s}\n",
        .{
            self.mtp_ev_rounds,                        kv_len,
            round_cost.bucketName(t.layout, b),        plan.m_lo,
            plan.m_hi,                                 plan.tau_ln,
            a[0],                                      a[1],
            a[2],                                      a[3],
            a[4],                                      a[5],
            if (src.fromTable()) "table" else "prior", if (plan.width_trial) "w" else "-",
            if (plan.width_trial) plan.m_lo else 0,    serial_step,
            self.mtp_m_lo_streak,                      self.mtp_serial_left,
            t.dropped_transition,                      t.dropped_contended,
            t.dropped_bad,                             cap_row,
            cap_free,                                  self.spec_cost_solo,
            self.mtp_regime.trial_end,                 self.mtp_width_trial.next_trial,
            self.mtp_width_trial.trial_end,            target,
            t.formatBucket(b, &cells),
        },
    );
    if (src.fromTable()) log.debug("  [mtp-plan-ms] w1..6=[{d:.1},{d:.1},{d:.1},{d:.1},{d:.1},{d:.1}]\n", .{ planMs(src, 1), planMs(src, 2), planMs(src, 3), planMs(src, 4), planMs(src, 5), planMs(src, 6) });
    return plan;
}

pub fn mtpRoundPlanInner(self: *Generator) MtpRoundPlan {
    if (mtpPinnedDepth()) |pinned| {
        const d = @min(pinned, self.mtp_depth);
        self.mtp_ev_m_lo_prev = d;
        return .{ .m_lo = d, .m_hi = d, .tau_ln = 0.0 };
    }
    if (self.mtp_serial_accept) {
        const d = depth_bounds.clampWidth(self.mtpExactDepth(@min(@max(@as(u32, 1), self.mtp_depth), 2)), depth_bounds.floorFor(depth_bounds.active, self.mtp_depth), self.mtp_depth);
        self.mtp_ev_m_lo_prev = d;
        return .{ .m_lo = d, .m_hi = d, .tau_ln = 0.0 };
    }
    const cap_row: u32 = @min(@max(@as(u32, 1), self.mtp_depth), mtp_mod.MAX_DEPTH);
    const cap_free: u32 = @min(@max(cap_row, self.mtp_depth_free), if (self.mtp) |h| h.maxDepth() else mtp_mod.MAX_DEPTH);
    var cap: u32 = cap_row;
    const kv_len = self.mtpKvLen();
    if (self.mtp_ev_rounds < MTP_EV_WARMUP_ROUNDS) {
        const d = depth_bounds.clampWidth(self.mtp_depth_current, depth_bounds.floorFor(depth_bounds.active, cap), cap);
        self.mtp_ev_m_lo_prev = d;
        // Independent of the depth controller's kill switch and warmup.
        _ = self.mtpAdaptiveSerialStep(d, kv_len);
        return .{ .m_lo = d, .m_hi = d, .tau_ln = 0.0 };
    }
    const src = MtpCostSource.init(self.mtp_ev_costs, kv_len, &self.xfm.round_cost);
    if (src.fromTable() and !self.xfm.round_cost.first_use_logged) {
        self.xfm.round_cost.first_use_logged = true;
        var buf: [256]u8 = undefined;
        log.info("[mtp] cost table: bucket {s} measured {s} (ms/tok) replaces the fitted surface (scale {d:.4} at w{d})\n", .{
            round_cost.bucketName(self.xfm.round_cost.layout, src.bucket),
            self.xfm.round_cost.formatBucket(src.bucket, &buf),
            src.scale,
            self.xfm.round_cost.narrowestMeasured(src.bucket) orelse 0,
        });
    }
    // The per-silicon row is the COLD-START cap: the plan may exceed it
    // only up to the widest TRUSTED width (never onto the prior's guess
    // above the row — that reopens the regime gate on ties the row had
    // closed), and the width trial may reach one past that to measure.
    if (src.fromTable()) cap = @min(cap_free, @max(cap_row, self.xfm.round_cost.widestMeasured(src.bucket) orelse cap_row));
    const policy = mtpDepthPolicyFor(mtpDepthPolicy(), kv_len);
    var plan = mtpPlanInBounds(mtpBasePlan(policy, self.mtp_ev_accept[0..cap], cap, src, self.mtp_ev_m_lo_prev + 1), depth_bounds.floorFor(depth_bounds.active, cap), cap);
    if (plan.m_lo == self.mtp_ev_m_lo_prev) self.mtp_m_lo_streak +|= 1 else self.mtp_m_lo_streak = 0;
    self.mtp_ev_m_lo_prev = plan.m_lo;
    // Live-cost lever: shorten dry exploration bursts when the MEASURED
    // chunk-A sync is an expensive fraction of the round (mtpExtDryThresholdFor).
    const dry_threshold = mtpExtDryThresholdFor(self.mtp_ev_sync_ms, self.mtp_ev_round_ms);
    if (plan.m_hi > plan.m_lo and
        !mtpExtDryAllows(&self.mtp_ext_dry_streak, &self.mtp_ext_cooldown, dry_threshold))
    {
        plan.m_hi = plan.m_lo;
        plan.tau_ln = 0.0;
    }
    if (plan.m_hi > plan.m_lo) {
        const force = mtpRegimeForce(&self.mtp_regime, self.mtp_ev_rounds);
        const worse = self.mtp_regime.sched_verdict;
        if (worse != null and worse != self.mtp_regime_verdict) {
            self.mtp_regime_verdict = worse;
            const r = self.mtp_regime;
            log.info("[mtp] regime gate: two-chunk {d:.2} ms/tok vs single {d:.2} ms/tok -> two-chunk {s} (period {d})\n", .{ r.two_ms / r.two_tok, r.one_ms / r.one_tok, if (worse.?) "throttled" else "every round", mtpRegimeExplorePeriod(r) });
        }
        if (force == false) {
            plan.m_hi = plan.m_lo;
            plan.tau_ln = 0.0;
        }
    }
    if (self.mtpAdaptiveSerialStep(plan.m_lo, kv_len)) return plan;
    // Width trial: a single-chunk round at the width the table needs
    // next (`mtpWidthTrialTarget`), one 2-round block per period. Never
    // inside a regime trial block (that block is the regime's own
    // measurement), and only while solo.
    if (self.spec_cost_solo and self.mtp_ev_rounds >= self.mtp_regime.trial_end) {
        const base_settled = self.mtp_m_lo_streak >= 2;
        if (mtpWidthTrialTarget(&self.xfm.round_cost, kv_len, plan, cap_free, base_settled)) |target| {
            // A next width nobody has timed is priced from the prior: learn it now, not a period from now.
            if (policy == .accept and self.xfm.round_cost.rawMs(target, src.bucket) == null) self.mtp_width_trial.startAt(self.mtp_ev_rounds);
            const period = if (policy == .accept)
                mtpProbePeriod(&self.mtp_ev_accept, src, plan.m_lo)
            else
                mtpWidthTrialPeriod(&self.xfm.round_cost, kv_len, plan.m_lo);
            const reread = round_cost.schedulePeriodReread(self.xfm.round_cost.layout);
            if (mtpWidthTrialForce(&self.mtp_width_trial, self.mtp_ev_rounds, period, reread)) {
                plan = mtpPlanInBounds(mtpWidthTrialPlan(target), depth_bounds.floorFor(depth_bounds.active, cap_free), cap_free);
            }
        }
    }
    return plan;
}

/// The single-chunk plan a width trial runs. Built from a scalar on
/// purpose: `plan = .{ .m_lo = plan.m_lo + 1, .m_hi = plan.m_lo + 1 }`
/// reads the already-written m_lo for m_hi (result-location aliasing)
/// and planned a two-chunk round — the simulated loop caught it.
pub fn mtpWidthTrialPlan(width: u32) MtpRoundPlan {
    return .{ .m_lo = width, .m_hi = width, .tau_ln = 0.0, .width_trial = true };
}

/// Which width a trial measures: the plan's own base when the bucket
/// the plan reads has not measured it, else an unmeasured m_lo+1 under
/// any shape, else a periodic m_lo+1 on single-chunk plans. Null =
/// nothing to try.
pub fn mtpWidthTrialTarget(t: *const round_cost.Table, kv_len: u32, plan: MtpRoundPlan, cap: u32, base_settled: bool) ?u32 {
    const b = t.bucketToRead(kv_len) orelse t.bucketOf(kv_len);
    // A single-chunk plan feeds its own base every round; only a
    // two-chunk plan (extensions and syncs never feed) owes a trial of
    // it — and only once the base has stopped climbing, or the first EV
    // rounds trial widths the plan is merely passing through (M1 Pro
    // 27B warm boot: one block at w3 in rep 1).
    if (t.measuredMs(plan.m_lo, b) == null) return if (plan.m_hi > plan.m_lo and base_settled) plan.m_lo else null;
    // m_lo-1 is deliberately NOT trialled: every trial block is a 3-4%
    // hit on the request that carries it (peer cells), the
    // prior below the anchor already prices narrower widths, and the
    // one cliff this was for (M1 Pro 27B, 4 -> 5) is found from 4 by
    // the m_lo+1 trial below.
    if (plan.m_lo >= cap) return null;
    // An unmeasured m_lo+1 is measured under ANY shape: a two-chunk
    // plan extends there every round on echo, blind to its cost (the
    // M1 Pro 27B's 4 -> 5 at +150 ms); the regime gate would catch it,
    // the table should know it — unless its first sample already
    // settled it as clearly worse. A settled one is re-tried only where
    // a single-chunk plan cannot otherwise reach it, at the long period.
    const up = plan.m_lo + 1;
    if (t.measuredMs(up, b) == null and !t.clearlyWorse(up, plan.m_lo, b)) return up;
    if (plan.m_hi == plan.m_lo) return up;
    return null;
}

/// Width-trial schedule: same shape as the regime's (explicit
/// trial_end/next_trial, a 2-round block — transition then measurement),
/// idempotent per round because `mtpRoundPlan` has two call sites.
pub const MtpWidthTrial = round_cost.TrialSchedule;

/// `reread` is the layout's answer (`round_cost.schedulePeriodReread`): every layout re-reads
/// the period so a cold bucket cannot inherit a neighbour's date.
pub fn mtpWidthTrialForce(t: *MtpWidthTrial, round_idx: u32, period: u32, reread: bool) bool {
    return t.force(round_idx, period, reread);
}

/// Rounds between width trials: the regime's drag rule on the measured
/// ms/tok gap between m_lo+1 and m_lo (a width G worse, tried once in
/// G/DRAG rounds, costs ~DRAG of throughput); the default period while
/// either is unmeasured, so an unknown width is learned soon.
pub fn mtpWidthTrialPeriod(t: *const round_cost.Table, kv_len: u32, m_lo: u32) u32 {
    const b = t.bucketToRead(kv_len) orelse return round_cost.EXPLORE_PERIOD_COLD;
    // An untrusted target keeps the cold period until it is either
    // trusted or settled as clearly worse (then its raw sample sizes
    // the gap): a raw 10%-better w5 read as a 60-round period and no
    // 22-round request ever trialled it again (M4 Max 27B, w5 stuck
    // at one sample).
    if (t.msPerTok(m_lo + 1, b) == null and !t.clearlyWorse(m_lo + 1, m_lo, b)) return round_cost.EXPLORE_PERIOD_COLD;
    return round_cost.trialPeriod(t.msPerTok(m_lo, b), t.rawMsPerTok(m_lo + 1, b));
}

/// Probe period when the plan's tokens come from the acceptance EMAs: the same drag rule,
/// over the EV rates of m_lo and m_lo+1. The probe is what refreshes `a[m_lo]`, so a stale
/// estimate shortens its own period as soon as the shallower indices move.
pub fn mtpProbePeriod(a: []const f32, src: MtpCostSource, m_lo: u32) u32 {
    const t = src.table orelse return round_cost.EXPLORE_PERIOD_COLD;
    if (!src.fromTable()) return round_cost.EXPLORE_PERIOD_COLD;
    if (t.measuredMs(m_lo + 1, src.bucket) == null) {
        // One sample at a verify-width cliff is enough to stop asking at the cold period.
        if (!t.clearlyWorse(m_lo + 1, m_lo, src.bucket)) return round_cost.EXPLORE_PERIOD_COLD;
        return round_cost.trialPeriod(t.msPerTok(m_lo, src.bucket), t.rawMsPerTok(m_lo + 1, src.bucket));
    }
    const here = src.roundCost(m_lo, false) / mtpEvExpectedTokens(a, m_lo);
    const up = src.roundCost(m_lo + 1, false) / mtpEvExpectedTokens(a, m_lo + 1);
    return round_cost.trialPeriod(here, up);
}

/// Track the only evidence that can justify sticky-disable: whether the
/// first draft landed while the EV base depth was exactly one. Wider base
/// rounds reset the probation window, and later extension misses do not
/// count against the depth-one floor. Returns a rate only after a full
/// fresh window has been observed.
pub fn mtpFloorDisableObserve(
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
pub fn updateMtpEvRound(self: *Generator, drafted: u32, accepted: u32) void {
    mtpEvObserve(&self.mtp_ev_accept, drafted, accepted, MTP_EV_EMA_BETA);
    self.mtp_ev_rounds += 1;
    if (mtpPinnedDepth() != null) return;
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
        self.spec_disable_reason = .acceptance;
    }
}

/// Close one traced round; emits + resets the summary at the cadence.
pub fn mtpTraceRoundEnd(self: *Generator, m: u32, accepted: u32, m_lo: u32) void {
    if (!mtpTraceEnabled()) return;
    if (!self.mtp_trace.endRound(m, accepted, m > m_lo)) return;
    const t = &self.mtp_trace;
    var acc_buf: [64]u8 = undefined;
    log.info(
        MtpTrace.LINE_FMT,
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
            t.avgMs(.predraft),
            t.avgMs(.gap),
            t.totalAvgMs(),
            @as(f64, @floatFromInt(t.drafted)) / @as(f64, @floatFromInt(t.rounds)),
            @as(f64, @floatFromInt(t.accepted)) / @as(f64, @floatFromInt(t.rounds)),
            @as(f64, @floatFromInt(t.extended)) / @as(f64, @floatFromInt(t.rounds)),
            t.accIdxStr(&acc_buf),
            t.subAvgMs(.build),
            t.subAvgMs(.ple),
            t.subAvgMs(.ple_sync),
            t.subAvgMs(.dispatch),
            t.subAvgMs(.wait),
            t.subAvgMs(.chain),
        },
    );
    t.reset();
}

// ── tests ──


test "mtpFloorDisableObserve: extension misses do not poison depth one" {
    const W = Generator.MTP_DEPTH_WINDOW;
    var drafted: [W]u8 = @splat(0);
    var accepted: [W]u8 = @splat(0);
    var idx: u32 = 0;

    var rate: ?f32 = null;
    for (0..W) |_| {
        // An eight-wide extension accepted its first draft but no later one.
        rate = Generator.mtpFloorDisableObserve(&drafted, &accepted, &idx, 1, 8, 1);
    }
    try testing.expectApproxEqAbs(@as(f32, 1.0), rate.?, 1e-5);
    for (drafted) |sample| try testing.expectEqual(@as(u8, 1), sample);

    drafted = @splat(0);
    accepted = @splat(0);
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
    try testing.expectApproxEqAbs(@as(f32, 0.5), rate.?, 1e-5);
    // A 50% depth-one window is comfortably above the breakeven floor —
    // this rate must never disable (the old 0.50 floor sat exactly here).
    try testing.expect(!(rate.? < Generator.MTP_DISABLE_BELOW));
}

test "mtpFloorDisableObserve: disable needs 16 fresh failures at base depth one" {
    const W = Generator.MTP_DEPTH_WINDOW;
    var drafted: [W]u8 = @splat(0);
    var accepted: [W]u8 = @splat(0);
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

test "mtpGroupPlan: a tile group drafts every lane to the cap, a split-K group only clamps" {
    const solo = Generator.MtpRoundPlan{ .m_lo = 1, .m_hi = 5, .tau_ln = -0.4 };
    const clamped = Generator.mtpGroupPlan(solo, 3, false);
    try testing.expectEqual(@as(u32, 1), clamped.m_lo);
    try testing.expectEqual(@as(u32, 3), clamped.m_hi);
    // The tile's rows are paid by the widest lane, so a shallow lane wastes them.
    const filled = Generator.mtpGroupPlan(solo, 3, true);
    try testing.expectEqual(@as(u32, 3), filled.m_lo);
    try testing.expectEqual(@as(u32, 3), filled.m_hi);
}

test "mtpWidthTrial: blocks per period, idempotent per round, period grows with the measured gap" {
    var wt = Generator.MtpWidthTrial{};
    var forced: u32 = 0;
    var i: u32 = 10;
    while (i < 210) : (i += 1) {
        const f = Generator.mtpWidthTrialForce(&wt, i, 8, true);
        try testing.expectEqual(f, Generator.mtpWidthTrialForce(&wt, i, 8, true)); // asked twice, same answer
        if (f) forced += 1;
    }
    // Identity checkable from the log: forced rounds == block * trials.
    try testing.expectEqual(wt.trials * round_cost.EXPLORE_BLOCK, forced);
    try testing.expect(wt.trials >= 17 and wt.trials <= 19);
    // Unmeasured next width: default period. 10% worse: 60. 21% worse: 128.
    var t = round_cost.Table{};
    try testing.expectEqual(round_cost.EXPLORE_PERIOD_COLD, Generator.mtpWidthTrialPeriod(&t, 1000, 4));
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = t.observe(4, 1000, 40.0, 4.0, true, false);
        _ = t.observe(5, 1000, 55.0, 5.0, true, false);
    }
    try testing.expectEqual(@as(u32, 60), Generator.mtpWidthTrialPeriod(&t, 1000, 4));
    // One better-looking sample is not a period: cold until trusted.
    var u = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES) |_| _ = u.observe(4, 1000, 40.0, 4.0, true, false);
    _ = u.observe(5, 1000, 45.0, 5.0, true, false);
    try testing.expectEqual(round_cost.EXPLORE_PERIOD_COLD, Generator.mtpWidthTrialPeriod(&u, 1000, 4));
    for (0..round_cost.MIN_SAMPLES) |_| _ = t.observe(6, 1000, 80.0, 6.0, true, false);
    try testing.expectEqual(@as(u32, 128), Generator.mtpWidthTrialPeriod(&t, 1000, 5));
}

test "round_cost: a simulated round loop measures every width the chooser picks and settles under the cliff" {
    // Driven as mtpRoundPlan drives it: ask the plan (prior until the
    // bucket has two widths), run that width, feed the table with a
    // synthetic machine whose round cost is flat-ish to depth 4 and cliffs
    // at 5 (M1 Pro 27B shape). Acceptance is high, so the prior alone
    // would sit at depth 6+.
    const Machine = struct {
        fn roundMs(m: u32) f32 {
            return switch (m) {
                0 => 50.0,
                1 => 56.0,
                2 => 62.0,
                3 => 68.0,
                4 => 74.0,
                else => 74.0 + 35.0 * @as(f32, @floatFromInt(m - 4)),
            };
        }
    };
    const a = [_]f32{ 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95 };
    var t = round_cost.Table{};
    var wt = Generator.MtpWidthTrial{};
    var prev: ?u32 = null;
    var prev2: ?u32 = null;
    var prev_two = false;
    var prev_two2 = false;
    var late_wide: u32 = 0;
    var streak: u32 = 0;
    var picked: [round_cost.MAX_WIDTH + 1]u32 = @splat(0);
    var m_lo_prev: u32 = 1;
    var last_m: u32 = 0;
    var i: u32 = 10;
    while (i < 600) : (i += 1) {
        const src = Generator.MtpCostSource.init(Generator.MTP_EV_DEFAULT_COSTS, 1000, &t);
        var plan = Generator.mtpEvPlanSrc(&a, 8, src, m_lo_prev + 1);
        if (plan.m_lo == m_lo_prev) streak += 1 else streak = 0;
        if (Generator.mtpWidthTrialTarget(&t, 1000, plan, 8, streak >= 2)) |target| {
            // `t` is a default (`.legacy`) table, so this loop runs the arm-once schedule it was fitted on.
            if (Generator.mtpWidthTrialForce(&wt, i, Generator.mtpWidthTrialPeriod(&t, 1000, plan.m_lo), round_cost.schedulePeriodReread(t.layout))) plan = Generator.mtpWidthTrialPlan(target);
        }
        // Run it: a two-chunk plan extends (confidence is high), so the
        // realized width is m_hi — and, as in mtpRoundEndObserve, only a
        // single-chunk round feeds the table.
        const m = plan.m_hi;
        const two_chunk = plan.m_hi > plan.m_lo;
        picked[m] += 1;
        if (i >= 200 and m >= 5) late_wide += 1;
        last_m = m;
        const tokens: f32 = Generator.mtpEvExpectedTokens(&a, m);
        const transition = prev_two or prev_two2 or (if (prev) |p| p != m else true) or (if (prev2) |p| p != m else true);
        if (!two_chunk) _ = t.observe(m, 1000, Machine.roundMs(m), tokens, true, transition);
        prev2 = prev;
        prev = m;
        prev_two2 = prev_two;
        prev_two = two_chunk;
        m_lo_prev = plan.m_lo;
    }
    // The cliff was found from ONE sample (clearly worse: never trusted,
    // never re-trialled at the cold period) and the loop settled under it.
    try testing.expect(t.rawMs(5, 0) != null);
    try testing.expect(t.clearlyWorse(5, 4, 0));
    try testing.expect(t.measuredMs(4, 0) != null);
    try testing.expect(last_m == 4);
    try testing.expect(picked[4] > 450);
    // Before the table activates the prior extends 4 -> 5 blind every round
    // (live, the regime gate throttles that shape; the sim has no gate), so
    // the bar is the settled tail: from round 200 on, width 5 appears only
    // as scheduled re-trial blocks (three in 400 rounds at this cliff).
    try testing.expect(late_wide <= 3 * round_cost.EXPLORE_BLOCK);
    try testing.expect(picked[6] + picked[7] + picked[8] <= 4); // the prior's first rounds only
    // Identity: forced rounds == 2 * trials (each block is 2 rounds).
    try testing.expect(wt.trials >= 2);
}

test "mtpProbePeriod: the probe date follows the acceptance EMAs, and one clearly worse sample still backs it off" {
    var t = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = t.observe(2, 1000, 42.0, 3.0, true, false);
        _ = t.observe(3, 1000, 48.0, 3.5, true, false);
    }
    const src = Generator.MtpCostSource.init(Generator.MTP_EV_DEFAULT_COSTS, 1000, &t);
    // Depth 3 reads clearly worse than 2: a long wait.
    const cold = [_]f32{ 0.7, 0.6, 0.2, 0.5, 0.5, 0.5 };
    const far = Generator.mtpProbePeriod(&cold, src, 2);
    try testing.expect(far >= 48);
    // The shallow indices saturate (prose turned into echo): the stale a[2] no longer holds the probe off.
    const echo = [_]f32{ 1.0, 1.0, 0.2, 0.5, 0.5, 0.5 };
    try testing.expect(Generator.mtpProbePeriod(&echo, src, 2) < far);
    // An unmeasured next width is learned at the cold period, unless its one sample was a cliff.
    try testing.expectEqual(round_cost.EXPLORE_PERIOD_COLD, Generator.mtpProbePeriod(&echo, src, 3));
    _ = t.observe(4, 1000, 70.0, 2.0, true, false);
    try testing.expect(Generator.mtpProbePeriod(&echo, src, 3) >= 100);
}

test "characterization: a sidecar (legacy-layout) boot plans exactly as 26.9.1 did" {
    // A sidecar (legacy-layout) boot must plan exactly as before: a store-version bump made
    // every boot cold and a cold table runs every round as a two-chunk round (-25% decode on
    // the 27B). Every expectation below was recorded by running the 26.9.1 planner.
    const G = Generator;
    const a = [_]f32{ 0.85, 0.78, 0.70, 0.60, 0.50, 0.42, 0.35, 0.30 };
    const kv: u32 = 8192; // one 27B rung; ~3.0-3.5 tok/round at m_lo 2-3
    const costs = G.MTP_EV_DEFAULT_COSTS;
    const cap: u32 = 6;

    const legacy_empty = round_cost.Table{ .layout = .legacy };
    try testing.expectEqual(@as(usize, 3), legacy_empty.bucketOf(kv));
    try testing.expectEqual(@as(usize, 3), round_cost.bucketFor(kv));

    // The warm table, restored from the file 26.9.1 wrote.
    const rc1_body = "rc1\n2 3 44.0000 2.7000 3\n3 3 64.0000 3.0500 3\n4 3 92.0000 3.2000 3\n";
    const warm = round_cost.parse(rc1_body, .legacy) orelse return error.TestUnexpectedResult;
    const warm_src = G.MtpCostSource.init(costs, kv, &warm);
    try testing.expect(warm_src.fromTable());

    // Warm: a measured marginal the position cannot repay closes the valve; every round is single-chunk.
    var prev: u32 = 1;
    for (0..8) |_| {
        const p = G.mtpEvPlanSrc(&a, cap, warm_src, prev + 1);
        try testing.expectEqual(@as(u32, 2), p.m_lo);
        try testing.expectEqual(@as(u32, 2), p.m_hi); // no extension
        try testing.expectEqual(@as(f32, 0.0), p.tau_ln);
        prev = p.m_lo;
    }

    // The same request on a cold table: every round extends.
    const cold_src = G.MtpCostSource.init(costs, kv, null);
    try testing.expect(!cold_src.fromTable());
    const cold0 = G.mtpEvPlanSrc(&a, cap, cold_src, 2);
    try testing.expectEqual(@as(u32, 2), cold0.m_lo);
    try testing.expectEqual(@as(u32, 4), cold0.m_hi);
    try testing.expectApproxEqAbs(@as(f32, -0.444614), cold0.tau_ln, 1e-5);
    prev = cold0.m_lo;
    for (0..7) |_| {
        const p = G.mtpEvPlanSrc(&a, cap, cold_src, prev + 1);
        try testing.expectEqual(@as(u32, 3), p.m_lo);
        try testing.expectEqual(@as(u32, 4), p.m_hi); // two-chunk, every round
        try testing.expectApproxEqAbs(@as(f32, -0.477666), p.tau_ln, 1e-5);
        prev = p.m_lo;
    }

    // A warm table whose cells make one extension position pay: extend by exactly one.
    var shallow = round_cost.Table{ .layout = .legacy };
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = shallow.observe(2, kv, 44.0, 2.70, true, false);
        _ = shallow.observe(3, kv, 51.0, 3.20, true, false);
        _ = shallow.observe(4, kv, 62.0, 3.50, true, false);
    }
    const shallow_src = G.MtpCostSource.init(costs, kv, &shallow);
    prev = 1;
    for (0..8) |_| {
        const p = G.mtpEvPlanSrc(&a, cap, shallow_src, prev + 1);
        try testing.expectEqual(@as(u32, 2), p.m_lo);
        try testing.expectEqual(@as(u32, 3), p.m_hi);
        try testing.expectApproxEqAbs(@as(f32, -0.488353), p.tau_ln, 1e-5);
        prev = p.m_lo;
    }

    // The width is a table-state property too: warm holds the base at 2, cold climbs to 3.
    var wide = round_cost.Table{ .layout = .legacy };
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = wide.observe(2, kv, 44.0, 2.70, true, false);
        _ = wide.observe(3, kv, 51.0, 3.20, true, false);
        _ = wide.observe(4, kv, 55.0, 4.00, true, false);
        _ = wide.observe(5, kv, 75.0, 4.20, true, false);
    }
    const wide_src = G.MtpCostSource.init(costs, kv, &wide);
    prev = 1;
    for (0..8) |_| {
        const p = G.mtpEvPlanSrc(&a, cap, wide_src, prev + 1);
        try testing.expectEqual(@as(u32, 2), p.m_lo);
        try testing.expectEqual(@as(u32, 4), p.m_hi);
        try testing.expectApproxEqAbs(@as(f32, -0.506371), p.tau_ln, 1e-5);
        prev = p.m_lo;
    }
    prev = 1;
    var cold_final: u32 = 0;
    for (0..8) |_| {
        const p = G.mtpEvPlanSrc(&a, cap, cold_src, prev + 1);
        cold_final = p.m_lo;
        prev = p.m_lo;
    }
    try testing.expectEqual(@as(u32, 3), cold_final);
    try testing.expect(cold_final != G.mtpEvPlanSrc(&a, cap, wide_src, 8).m_lo);

    try testing.expectEqual(
        @as(?u32, 3),
        G.mtpWidthTrialTarget(&warm, kv, .{ .m_lo = 2, .m_hi = 2, .tau_ln = 0 }, cap, true),
    );
}

test "characterization: a sidecar boot's width-trial SCHEDULE re-reads its period like qwen4" {
    // Which ROUND carries a trial: the previous arm-once schedule kept a
    // neighbour bucket's date; the re-read is every layout's.
    const G = Generator;
    const legacy = round_cost.Table{ .layout = .legacy };
    const long = round_cost.Table{ .layout = .long };
    try testing.expect(round_cost.schedulePeriodReread(legacy.layout));
    try testing.expect(round_cost.schedulePeriodReread(long.layout));

    const run = struct {
        fn f(reread: bool) ?u32 {
            var wt = G.MtpWidthTrial{};
            var i: u32 = 12;
            while (i < 200) : (i += 1) {
                const period: u32 = if (i < 18) 124 else round_cost.EXPLORE_PERIOD_COLD;
                if (G.mtpWidthTrialForce(&wt, i, period, reread)) return i;
            }
            return null;
        }
    }.f;
    try testing.expectEqual(@as(?u32, 136), run(false)); // the arm-once schedule
    try testing.expectEqual(@as(?u32, 20), run(true)); // shipped: every layout
}
