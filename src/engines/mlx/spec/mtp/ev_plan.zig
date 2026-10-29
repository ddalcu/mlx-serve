//! MTP EV round planning: depth caps, cost sources, depth policy.
const depth_bounds = @import("../mtp_depth_bounds.zig");
const std = @import("std");
const env = @import("../../../../env.zig");
const transformer_mod = @import("../../transformer.zig");
const log = @import("../../../../sdk/util/log.zig");
const io_util = @import("../../../../sdk/util/io_util.zig");
const mtp_mod = @import("../../../../sdk/spec/mtp/mtp.zig");
const round_cost = @import("../round_cost.zig");
const ane_mod = @import("../../../../sdk/ane/offload.zig");
const Transformer = transformer_mod.Transformer;
const testing = std.testing;
const depth_mod = @import("depth.zig");
const Generator = @import("../../generate/generator.zig").Generator;
const MTP_ADAPTIVE_DEFAULT_CAP = @import("depth.zig").MTP_ADAPTIVE_DEFAULT_CAP;
const MTP_ADAPTIVE_NAX_CAP = @import("depth.zig").MTP_ADAPTIVE_NAX_CAP;
const MTP_EV_EXPLORE_MIN_R = @import("depth.zig").MTP_EV_EXPLORE_MIN_R;
const MTP_EV_TAU_MAX = @import("depth.zig").MTP_EV_TAU_MAX;
const MTP_EV_TAU_MIN = @import("depth.zig").MTP_EV_TAU_MIN;
const MtpEvCosts = @import("ev_costs.zig").MtpEvCosts;
const init = @import("../../generate/generator.zig").Generator.init;


/// Write the model's round-cost table to disk when this request folded
/// new samples (request end; inference thread, so no lock).
pub fn persistRoundCost(self: *Generator) void {
    const t = &self.xfm.round_cost;
    const folded = round_cost.totalFolded(t);
    if (folded == t.stored_at) return;
    round_cost.storeCached(self.timer.io, self.xfm.round_cost_key_buf[0..self.xfm.round_cost_key_len], t);
    t.stored_at = folded;
}

/// KV length the next verify forward reads.
pub fn mtpKvLen(self: *const Generator) u32 {
    return @intCast(self.prompt_ids_owned.len + self.generated_ids.items.len);
}

/// Marginal round cost of draft position k (1-based), kv-blind. This is
/// the fitted SHAPE; `mtpEvMarginalCostAt` applies the kv term.
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
    /// A width-trial round (single-chunk at m_lo+1, measuring the round
    /// cost table): skipped by the regime gate, which compares shapes
    /// at ONE base depth.
    width_trial: bool = false,
};

/// Resolve the configured depth cap. 0 = auto (`--mtp-max-depth` not passed):
/// MTP_ADAPTIVE_NAX_CAP only when the EV controller and a calibrated G17
/// cost profile are both active, MTP_ADAPTIVE_DEFAULT_CAP otherwise, and
/// DEFAULT_DEPTH in fixed mode. Explicit values always win.
pub fn mtpDepthCapForProfile(configured: u32, adaptive: bool, profile: mtp_mod.MtpCostProfile) u32 {
    const chip = ane_mod.chipBrand();
    const cap = mtpDepthCapResolved(configured, adaptive, profile, chip);
    // Name the row ONCE when a per-silicon measurement is what fenced the
    // depth. Without it `[spec-stats] depth=4` on an M1 Pro reads the same
    // as the EV controller having chosen 4, or as `--mtp-max-depth 4` — and
    // the fence is exactly what someone debugging MTP there needs to see.
    if (configured == 0 and adaptive and !depth_mod.mtp_depth_cap_logged) {
        const row = mtp_mod.adaptiveDepthCapForMachine(chip, MTP_ADAPTIVE_DEFAULT_CAP);
        if (cap == row.cap and row.cap < MTP_ADAPTIVE_DEFAULT_CAP) {
            depth_mod.mtp_depth_cap_logged = true;
            log.info("[mtp] adaptive depth cap {d} ({s} row, default {d})\n", .{
                row.cap,
                row.label,
                MTP_ADAPTIVE_DEFAULT_CAP,
            });
        }
    }
    return depth_bounds.liftCap(cap, depth_bounds.active, mtp_mod.MAX_DEPTH);
}

/// The cap with no per-silicon row applied: explicit wins, else the
/// adaptive default (fixed mode keeps DEFAULT_DEPTH).
pub fn mtpDepthCapFree(configured: u32) u32 {
    const free = if (configured != 0) @min(mtp_mod.MAX_DEPTH, @max(1, configured)) else MTP_ADAPTIVE_DEFAULT_CAP;
    return depth_bounds.liftCap(free, depth_bounds.active, mtp_mod.MAX_DEPTH);
}

/// Same, with the chip string injected (tests, and the one live caller).
pub fn mtpDepthCapForProfileChip(configured: u32, adaptive: bool, profile: mtp_mod.MtpCostProfile, chip: []const u8) u32 {
    return mtpDepthCapResolved(configured, adaptive, profile, chip);
}

/// Same, with the machine's MEASURED verify ladder when one exists.
///
/// The per-silicon rows exist because the EV controller cannot see time:
/// it scores accepted-tokens-per-round, which on an M1 Pro IMPROVED
/// (3.65 -> 4.00) while realized tok/s fell 21%. A measured curve sees
/// exactly the cliff those rows encode, on whatever chip and whatever
/// quant geometry is actually resident — which the chip key cannot
/// express (the split-K lane is 4-bit/g64 only, so a 6-bit pack on the
/// same box is a different answer). The calibrated NAX profiles keep
/// their own cap until a probe on that silicon is validated against it.
pub fn mtpDepthCapResolved(
    configured: u32,
    adaptive: bool,
    profile: mtp_mod.MtpCostProfile,
    chip: []const u8,
) u32 {
    if (configured != 0) return @min(mtp_mod.MAX_DEPTH, @max(1, configured));
    if (!adaptive) return mtp_mod.DEFAULT_DEPTH;
    return switch (profile) {
        // The per-silicon row is the COLD-START cap: the measured
        // round-cost table may plan above it on trusted widths.
        .generic => mtp_mod.adaptiveDepthCapForMachine(chip, MTP_ADAPTIVE_DEFAULT_CAP).cap,
        .g17_nax_q8_gs32, .g17_nax_q4_gs32, .g17_nax_q6_gs64, .g17_nax_q8_gs64, .g17_nax_oq4e_q4_gs64 => MTP_ADAPTIVE_NAX_CAP,
        // The dense 27B q4/gs64 surface stops paying past depth 6 on M5.
        .g17_nax_q4_gs64 => MTP_ADAPTIVE_DEFAULT_CAP,
        // qwen4's measured surface has no NAX region to reach: depths 7-8
        // price at .345/pos against sub-60% tail acceptance even on
        // saturated echo, so the calibrated cap keeps the default — the
        // NAX cap exists for surfaces that flatten past position 6.
        .g17_nax_qwen4_q4_gs64, .g17_nax_qwen4_mixed_4_8_gs64 => MTP_ADAPTIVE_DEFAULT_CAP,
    };
}

pub fn resolveMtpDepthCapForProfile(configured: u32, profile: mtp_mod.MtpCostProfile) u32 {
    return mtpDepthCapForProfile(configured, true, profile);
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

pub fn mtpEvMarginalCostAt(costs: MtpEvCosts, k: u32, kv_len: u32) f32 {
    _ = kv_len;
    return mtpEvMarginalCost(costs, k);
}

/// Round cost in verify-base units (piecewise per-position marginals).
pub fn mtpEvRoundCost(costs: MtpEvCosts, m: u32, with_sync: bool) f32 {
    return mtpEvRoundCostAt(costs, m, with_sync, 0);
}

pub fn mtpEvRoundCostAt(costs: MtpEvCosts, m: u32, with_sync: bool, kv_len: u32) f32 {
    var c: f32 = 1.0 + (if (with_sync) costs.sync else 0.0);
    var k: u32 = 1;
    while (k <= m) : (k += 1) c += mtpEvMarginalCostAt(costs, k, kv_len);
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
    return mtpEvPlanForAt(a, cap_in, costs, m_lo_max, 0);
}

pub fn mtpEvPlanForAt(a: []const f32, cap_in: u32, costs: MtpEvCosts, m_lo_max: u32, kv_len: u32) MtpRoundPlan {
    return mtpEvPlanSrc(a, cap_in, MtpCostSource.init(costs, kv_len, null), m_lo_max);
}

/// Smallest marginal the table may report, in floor units: a measured
/// pair that reads a wider round CHEAPER is noise, and a non-positive
/// marginal would make every extension free.
pub const MTP_EV_TABLE_MIN_MARGINAL: f32 = 0.02;

/// Where the EV plan's round costs come from: the measured round-cost
/// table when the kv bucket (or its nearest active neighbour) has
/// MIN_WIDTHS measured widths, else the fitted surface as the cold-start
/// prior. The table is in ms; the plan compares ratios but has one
/// absolute threshold (`MTP_EV_EXPLORE_MIN_R`), so the table is scaled
/// into FLOOR UNITS at its narrowest measured width — the two sources
/// agree there by construction and the table's slopes take over.
pub const MtpCostSource = struct {
    costs: MtpEvCosts,
    kv_len: u32 = 0,
    table: ?*const round_cost.Table = null,
    bucket: usize = 0,
    scale: f32 = 0,
    /// False = the table prices round TIME only and tokens come from the acceptance EMAs.
    table_tokens: bool = true,

    pub fn init(costs: MtpEvCosts, kv_len: u32, table: ?*const round_cost.Table) MtpCostSource {
        var src = MtpCostSource{ .costs = costs, .kv_len = kv_len, .table = table };
        const t = table orelse return src;
        const b = t.bucketToRead(kv_len) orelse return src;
        const ref = t.narrowestMeasured(b) orelse return src;
        const ref_ms = t.measuredMs(ref, b) orelse return src;
        if (!(ref_ms > 0) or ref > mtp_mod.MAX_DEPTH) return src;
        src.bucket = b;
        src.scale = mtpEvRoundCostAt(costs, ref, false, kv_len) / ref_ms;
        return src;
    }

    pub fn fromTable(self: MtpCostSource) bool {
        return self.scale > 0;
    }

    /// Realized tokens per single-chunk round at a MEASURED width, else
    /// null (the acceptance-EMA model fills in). The model's E(6) on the
    /// M4 Max @16k said 6.85 while the measured single-6 round emitted
    /// 6.0 — the 6th draft's rejections cost a rollback the model cannot
    /// see, and the table can.
    pub fn measuredTokens(self: MtpCostSource, m: u32) ?f32 {
        if (self.scale <= 0 or !self.table_tokens) return null;
        return self.table.?.measuredTok(m, self.bucket);
    }

    pub fn roundCost(self: MtpCostSource, m: u32, with_sync: bool) f32 {
        if (self.scale > 0) {
            const sync: f32 = if (with_sync) self.costs.sync else 0.0;
            const t = self.table.?;
            if (t.roundMs(m, self.bucket)) |ms| return ms * self.scale + sync;
            // Past the widest measured width every extra position costs
            // max(last measured slope, prior marginal): continuous with
            // the table, never more optimistic than the prior (a shallow
            // measured slope run upward let the plan race to 8 unmeasured).
            if (t.widestMeasured(self.bucket)) |w| {
                if (m > w) {
                    const slope = (t.lastSlope(self.bucket) orelse 0.0) * self.scale;
                    var c = t.measuredMs(w, self.bucket).? * self.scale;
                    var k: u32 = w + 1;
                    while (k <= m) : (k += 1) {
                        c += @max(slope, mtpEvMarginalCostAt(self.costs, k, self.kv_len));
                        // One sample is evidence for WORSE, never for
                        // cheaper: an untrusted cell floors the cost. Round TIME
                        // alone is stable to a few percent, so where no token
                        // column is read the sample prices its width.
                        if (t.rawMs(k, self.bucket)) |raw| c = if (self.table_tokens) @max(c, raw * self.scale) else raw * self.scale;
                    }
                    return c + sync;
                }
            }
        }
        return mtpEvRoundCostAt(self.costs, m, with_sync, self.kv_len);
    }

    pub fn marginal(self: MtpCostSource, k: u32) f32 {
        if (self.scale > 0 and k >= 1) {
            return @max(self.roundCost(k, false) - self.roundCost(k - 1, false), MTP_EV_TABLE_MIN_MARGINAL);
        }
        return mtpEvMarginalCostAt(self.costs, k, self.kv_len);
    }
};

/// How the base depth is chosen. `.accept`: expected tokens come from the acceptance EMAs
/// (every round refreshes every index it drafted) and the table prices round TIME only, in
/// ONE chunk, so no round reads confidences mid-round. `.legacy`: the table's realized
/// tokens where measured, plus the confidence-gated chunk B.
pub const MtpDepthPolicy = enum { accept, legacy };
/// `.accept` serves KV lengths below this. It was fitted where a round is CPU-build bound and
/// the chunk-A confidence read costs its whole wait; from here up the verify forward dominates,
/// the read overlaps GPU work, and a shallow base + chunk B undercuts every single-chunk
/// depth, so the legacy planner stays.
pub const MTP_ACCEPT_MAX_KV: u32 = 8192;

pub fn mtpDepthPolicyFor(configured: MtpDepthPolicy, kv_len: u32) MtpDepthPolicy {
    return if (kv_len >= MTP_ACCEPT_MAX_KV) .legacy else configured;
}

/// Test seam: forces the configured policy (past `MTP_ACCEPT_MAX_KV` it is legacy either way).
pub var mtp_depth_policy_override: ?MtpDepthPolicy = null;
pub fn mtpDepthPolicy() MtpDepthPolicy {
    return mtp_depth_policy_override orelse .accept;
}

pub fn mtpBasePlan(policy: MtpDepthPolicy, a: []const f32, cap: u32, src_in: MtpCostSource, m_lo_max: u32) MtpRoundPlan {
    var src = src_in;
    src.table_tokens = policy == .legacy;
    var plan = mtpEvPlanSrc(a, cap, src, m_lo_max);
    if (policy == .accept) {
        plan.m_hi = plan.m_lo;
        plan.tau_ln = 0.0;
    }
    return plan;
}

pub fn mtpEvPlanSrc(a: []const f32, cap_in: u32, src: MtpCostSource, m_lo_max: u32) MtpRoundPlan {
    const cap: u32 = @intCast(@min(@as(usize, @max(1, cap_in)), a.len));
    const lo_cap: u32 = @min(cap, @max(1, m_lo_max));
    var m_lo: u32 = 1;
    var best_r: f32 = 0.0;
    // With measured costs the standing base (m_lo_max - 1, what the
    // caller planned last round) keeps its place unless a challenger
    // beats it by the switch margin: cells are EMAs of a few samples
    // and a 5% reversal between neighbours is noise, while every
    // change of base is a transition round the table will not count
    // and the model pays for.
    const standing: u32 = if (src.fromTable()) @min(lo_cap, m_lo_max -| 1) else 0;
    var standing_r: f32 = 0.0;
    var m: u32 = 1;
    while (m <= lo_cap) : (m += 1) {
        const tok = src.measuredTokens(m) orelse mtpEvExpectedTokens(a, m);
        const r = tok / src.roundCost(m, false);
        if (m == standing) standing_r = r;
        if (r > best_r) {
            best_r = r;
            m_lo = m;
        }
    }
    if (standing >= 1 and m_lo != standing and best_r <= standing_r * (1.0 + round_cost.SWITCH_MARGIN)) {
        m_lo = standing;
        best_r = standing_r;
    }
    if (m_lo >= cap) return .{ .m_lo = m_lo, .m_hi = m_lo, .tau_ln = 0.0 };
    var m_hi: u32 = m_lo;
    var cond: f32 = 1.0;
    var s_sum: f32 = 0.0; // expected extension tokens, conditional on chunk A
    var t_sum: f32 = 0.0; // extension marginal cost (piecewise)
    while (m_hi < cap) {
        cond *= a[m_hi];
        const mc = src.marginal(m_hi + 1);
        if (cond <= best_r * mc) break;
        s_sum += cond;
        t_sum += mc;
        m_hi += 1;
    }
    if (m_hi == m_lo) {
        // Exploration valve for the horizon itself: the
        // index one past m_lo is observable ONLY when extension fires —
        // at m_lo-deep rounds a[m_lo] never updates — so an EMA dragged
        // cold under an earlier workload (or an easier tau regime)
        // would otherwise close the horizon FOREVER: m stays at m_lo,
        // a[m_lo] never re-proves itself, and pure echo runs at
        // ext_rounds=0 (the exact class the TAU_MAX valve exists for,
        // resurfaced by refit #3's honest marginals). Whenever the base
        // itself pays, keep ONE extension position reachable at the
        // clamped tau; the dry-spell gate bounds the consideration tax
        // on workloads where confidence never clears it.
        if (best_r <= MTP_EV_EXPLORE_MIN_R) return .{ .m_lo = m_lo, .m_hi = m_lo, .tau_ln = 0.0 };
        const mc = src.marginal(m_lo + 1);
        // A MEASURED marginal the position cannot repay even at full
        // confidence (1 <= best_r * mc) closes the valve: the valve
        // exists to observe a[m_lo], and with the table that is the
        // width trial's job. The fitted prior keeps the valve exactly
        // as Phase 1 measured it.
        if (src.fromTable() and best_r * mc >= 1.0) return .{ .m_lo = m_lo, .m_hi = m_lo, .tau_ln = 0.0 };
        const s = @max(a[m_lo], 1e-6);
        const tau_x = std.math.clamp(best_r * mc / s, MTP_EV_TAU_MIN, MTP_EV_TAU_MAX);
        return .{ .m_lo = m_lo, .m_hi = m_lo + 1, .tau_ln = @log(tau_x) };
    }
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

// ── tests ──


pub fn mtpEvTestGenerator() Generator {
    var g: Generator = undefined;
    g.mtp_depth = Generator.MTP_ADAPTIVE_DEFAULT_CAP;
    g.mtp_depth_current = 1;
    g.mtp_window_drafted = @splat(0);
    g.mtp_window_accepted = @splat(0);
    g.mtp_window_idx = 0;
    g.mtp_rounds_since_switch = 0;
    g.mtp_promote_cooldown = 0;
    g.mtp_ev_accept = @splat(Generator.MTP_EV_PRIOR);
    g.mtp_ev_rounds = Generator.MTP_EV_WARMUP_ROUNDS;
    g.mtp_ev_m_lo_prev = 1;
    g.mtp_ev_costs = Generator.MTP_EV_DEFAULT_COSTS;
    g.spec_disabled_runtime = false;
    return g;
}

test "mtpDepthCapFor: auto cap follows the selected cost profile; explicit always wins" {
    // 0 = auto (--mtp-max-depth not passed).
    // .generic's auto cap is per-silicon, so the chip must be injected here
    // or the assertion is a property of whichever Mac runs the suite.
    try testing.expectEqual(Generator.MTP_ADAPTIVE_DEFAULT_CAP, Generator.mtpDepthCapForProfileChip(0, true, .generic, "Apple M4 Max"));
    try testing.expectEqual(@as(u32, 6), Generator.mtpDepthCapForProfileChip(0, true, .generic, ""));
    try testing.expectEqual(@as(u32, 4), Generator.mtpDepthCapForProfileChip(0, true, .generic, "Apple M1 Pro"));
    // An explicit depth still outranks the table on a measured chip.
    try testing.expectEqual(@as(u32, 8), Generator.mtpDepthCapForProfileChip(8, true, .generic, "Apple M1 Pro"));
    try testing.expectEqual(mtp_mod.DEFAULT_DEPTH, Generator.mtpDepthCapForProfileChip(0, false, .generic, "Apple M1 Pro"));
    // The 27B q4/gs64 pack peaks at depth 6 on M5 and loses at 8.
    try testing.expectEqual(Generator.MTP_ADAPTIVE_DEFAULT_CAP, Generator.mtpDepthCapForProfile(0, true, .g17_nax_q4_gs64));
    for ([_]mtp_mod.MtpCostProfile{ .g17_nax_q8_gs32, .g17_nax_q4_gs32, .g17_nax_q6_gs64, .g17_nax_q8_gs64, .g17_nax_oq4e_q4_gs64 }) |profile| {
        try testing.expectEqual(Generator.MTP_ADAPTIVE_NAX_CAP, Generator.mtpDepthCapForProfile(0, true, profile));
        try testing.expectEqual(@as(u32, 8), Generator.mtpDepthCapForProfile(0, true, profile));
    }
    // qwen4's calibrated surface keeps the DEFAULT cap: no NAX region exists
    // to justify opening depths 7-8 on that arch.
    try testing.expectEqual(Generator.MTP_ADAPTIVE_DEFAULT_CAP, Generator.mtpDepthCapForProfile(0, true, .g17_nax_qwen4_q4_gs64));
    try testing.expectEqual(@as(u32, 6), Generator.mtpDepthCapForProfile(0, true, .g17_nax_qwen4_q4_gs64));
    for ([_]mtp_mod.MtpCostProfile{ .generic, .g17_nax_q8_gs32, .g17_nax_q4_gs32, .g17_nax_q4_gs64, .g17_nax_q6_gs64, .g17_nax_q8_gs64, .g17_nax_oq4e_q4_gs64, .g17_nax_qwen4_q4_gs64, .g17_nax_qwen4_mixed_4_8_gs64 }) |profile| {
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

    const live_generic = mtp_mod.adaptiveDepthCapForMachine(ane_mod.chipBrand(), Generator.MTP_ADAPTIVE_DEFAULT_CAP).cap;
    try testing.expectEqual(live_generic, Generator.mtpDepthCapFor(0, true, false));
    try testing.expectEqual(@as(u32, 8), Generator.mtpDepthCapFor(0, true, true));
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

test "Qwen4 mixed MTP cost prior prices the measured width-seven fallback" {
    const costs = Generator.mtpEvCosts(.g17_nax_qwen4_mixed_4_8_gs64);
    // The 8k forced-depth echo sweep measures about 5 ms per position
    // through depth 5, then an approximately 8 ms step at depth 6.
    // Width 7 is outside the single-stream verifier's optimized envelope.
    const optimized = Generator.mtpEvMarginalCost(costs, 5);
    const fallback = Generator.mtpEvMarginalCost(costs, 6);
    try testing.expect(fallback > 1.4 * optimized);
    try testing.expect(Generator.mtpEvMarginalCost(costs, 7) < fallback);

    // At this confidence the old smooth ramp picks depth 6. The measured
    // fallback cost makes depth 5 the better cold-start expected rate.
    const acceptance: [6]f32 = @splat(0.97);
    const plan = Generator.mtpEvPlanFor(&acceptance, 6, costs, 6);
    try testing.expectEqual(@as(u32, 5), plan.m_lo);
}

test "mtpEvCosts: each G17 profile selects its own measured surface" {
    const generic = Generator.mtpEvCosts(.generic);
    try testing.expectEqual(Generator.MTP_EV_DEFAULT_COSTS, generic);

    const q8 = Generator.mtpEvCosts(.g17_nax_q8_gs32);
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_COSTS, q8);
    try testing.expectEqual(@as(u32, 7), q8.nax_from);

    const q4 = Generator.mtpEvCosts(.g17_nax_q4_gs32);
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_Q4_GS32_COSTS, q4);
    try testing.expectEqual(@as(u32, 7), q4.nax_from);

    const q4_gs64 = Generator.mtpEvCosts(.g17_nax_q4_gs64);
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_Q4_GS64_COSTS, q4_gs64);
    try testing.expectEqual(@as(u32, 7), q4_gs64.nax_from);

    const oq4e = Generator.mtpEvCosts(.g17_nax_oq4e_q4_gs64);
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS, oq4e);
    try testing.expectEqual(@as(u32, 7), oq4e.nax_from);

    const qwen4 = Generator.mtpEvCosts(.g17_nax_qwen4_q4_gs64);
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS, qwen4);

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

test "MTP_EV_G17_NAX_Q4_GS64_COSTS reproduces the calibrated M5 surface" {
    const costs = Generator.MTP_EV_G17_NAX_Q4_GS64_COSTS;
    try testing.expectApproxEqAbs(@as(f32, 0.075), Generator.mtpEvMarginalCost(costs, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.195), Generator.mtpEvMarginalCost(costs, 4), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.08), Generator.mtpEvMarginalCost(costs, 7), 1e-6);
    const t1 = Generator.mtpEvRoundCost(costs, 1, false);
    const t3 = Generator.mtpEvRoundCost(costs, 3, false);
    const t6 = Generator.mtpEvRoundCost(costs, 6, false);
    const t8 = Generator.mtpEvRoundCost(costs, 8, false);
    try testing.expectApproxEqAbs(@as(f32, 1.075), t1, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.225), t3, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.81), t6, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.97), t8, 1e-5);
}

test "MTP_EV_G17_NAX_Q6_GS64_COSTS reproduces the calibrated M5 surface" {
    const costs = Generator.MTP_EV_G17_NAX_Q6_GS64_COSTS;
    try testing.expectApproxEqAbs(@as(f32, 0.06), Generator.mtpEvMarginalCost(costs, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.115), Generator.mtpEvMarginalCost(costs, 4), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.145), Generator.mtpEvMarginalCost(costs, 7), 1e-6);
    const t6 = Generator.mtpEvRoundCost(costs, 6, false);
    const t8 = Generator.mtpEvRoundCost(costs, 8, false);
    try testing.expectApproxEqAbs(@as(f32, 1.525), t6, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.815), t8, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 76.79 / 64.495), t8 / t6, 1e-3);
}

test "MTP_EV_G17_NAX_Q8_GS64_COSTS bridges the discontinuous NAX takeover" {
    const costs = Generator.MTP_EV_G17_NAX_Q8_GS64_COSTS;
    try testing.expectApproxEqAbs(@as(f32, 0.06), Generator.mtpEvMarginalCost(costs, 4), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.19), Generator.mtpEvMarginalCost(costs, 5), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.045), Generator.mtpEvMarginalCost(costs, 7), 1e-6);
    const t6 = Generator.mtpEvRoundCost(costs, 6, false);
    const t8 = Generator.mtpEvRoundCost(costs, 8, false);
    try testing.expectApproxEqAbs(@as(f32, 1.62), t6, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.71), t8, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 85.94 / 53.0575), t6, 2e-3);
    try testing.expectApproxEqAbs(@as(f32, 90.68 / 53.0575), t8, 2e-3);
}

test "MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS reproduces the measured M5 surface" {
    const costs = Generator.MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS;
    try testing.expectApproxEqAbs(@as(f32, 0.095), Generator.mtpEvMarginalCost(costs, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.22), Generator.mtpEvMarginalCost(costs, 4), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.025), Generator.mtpEvMarginalCost(costs, 7), 1e-6);
    const t6 = Generator.mtpEvRoundCost(costs, 6, false);
    const t8 = Generator.mtpEvRoundCost(costs, 8, false);
    try testing.expectApproxEqAbs(@as(f32, 1.945), t6, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.995), t8, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 62.79 / 61.20), t8 / t6, 5e-4);
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
    for ([_]Generator.MtpEvCosts{ Generator.MTP_EV_G17_NAX_COSTS, Generator.MTP_EV_G17_NAX_Q4_GS32_COSTS, Generator.MTP_EV_G17_NAX_Q4_GS64_COSTS, Generator.MTP_EV_G17_NAX_Q6_GS64_COSTS, Generator.MTP_EV_G17_NAX_Q8_GS64_COSTS, Generator.MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS }) |costs| {
        const nax = Generator.mtpEvPlanFor(&a, 8, costs, 3);
        try testing.expectEqual(@as(u32, 3), nax.m_lo);
        try testing.expectEqual(@as(u32, 8), nax.m_hi);
        try testing.expect(nax.tau_ln < 0.0);
    }

    // qwen4_exp's measured surface has no NAX takeover — a verify row is
    // bytes at every width — so the same base depth stands but the same
    // evidence must NOT extend to depth 8.
    const qwen4_plan = Generator.mtpEvPlanFor(&a, 8, Generator.MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS, 3);
    try testing.expectEqual(@as(u32, 3), qwen4_plan.m_lo);
    try testing.expect(qwen4_plan.m_hi < 8);

    const cold = [_]f32{ 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };
    for ([_]Generator.MtpEvCosts{ Generator.MTP_EV_G17_NAX_COSTS, Generator.MTP_EV_G17_NAX_Q4_GS32_COSTS, Generator.MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS, Generator.MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS }) |costs| {
        const cold_plan = Generator.mtpEvPlanFor(&cold, 8, costs, 8);
        try testing.expectEqual(@as(u32, 1), cold_plan.m_lo);
        try testing.expectEqual(@as(u32, 1), cold_plan.m_hi);
    }
    // Qwen3.8's measured depth-1 marginal is cheap enough that even this
    // synthetic 20% first-draft case clears the exploration floor narrowly.
    // The base stays at one and only one confidence-gated position is exposed.
    const q38_cold = Generator.mtpEvPlanFor(&cold, 8, Generator.MTP_EV_G17_NAX_Q4_GS64_COSTS, 8);
    try testing.expectEqual(@as(u32, 1), q38_cold.m_lo);
    try testing.expectEqual(@as(u32, 2), q38_cold.m_hi);
    try testing.expect(q38_cold.tau_ln < 0.0);
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

test "mtpEvPlanFor: a stale-cold EMA one past m_lo cannot close the horizon permanently" {
    // a[1] is observable ONLY when extension fires (at m=1 rounds it never
    // updates), so a value dragged down under an old workload must not be
    // able to lock the horizon shut — the live regression was ext_rounds=0
    // on the equivalence echo after refit #3's honest marginals. When the
    // base pays, one extension position stays reachable at the clamped
    // exploration tau; realized extensions then re-observe a[1] honestly.
    const costs = Generator.MTP_EV_DEFAULT_COSTS;
    const a = [_]f32{ 0.75, 0.28, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85 };
    const plan = Generator.mtpEvPlanFor(&a, 8, costs, 8);
    try testing.expectEqual(@as(u32, 1), plan.m_lo);
    try testing.expectEqual(@as(u32, 2), plan.m_hi);
    // tau clamps to TAU_MAX: only near-perfect-confidence rounds extend.
    try testing.expectApproxEqAbs(@log(Generator.MTP_EV_TAU_MAX), plan.tau_ln, 1e-6);
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

test "MtpCostSource: a measured cliff stops the plan where the fitted surface would extend" {
    // M1 Pro 27B: depth 4 -> 5 is a +43% round. The fitted surface prices
    // position 5 at 0.26 floor units and extends; the table has measured
    // it. Same acceptance, same cap, two answers.
    const a = [_]f32{ 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95 };
    const prior = Generator.mtpEvPlanForAt(&a, 8, Generator.MTP_EV_DEFAULT_COSTS, 8, 10000);
    try testing.expect(prior.m_hi >= 5);
    var t = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = t.observe(3, 10000, 60.0, 4.0, true, false);
        _ = t.observe(4, 10000, 70.0, 5.0, true, false);
        _ = t.observe(5, 10000, 102.0, 6.0, true, false);
    }
    const src = Generator.MtpCostSource.init(Generator.MTP_EV_DEFAULT_COSTS, 10000, &t);
    try testing.expect(src.fromTable());
    // Anchored at the narrowest measured width: the two sources agree
    // there, and below it the prior's shape fills in.
    try testing.expectApproxEqAbs(Generator.mtpEvRoundCostAt(Generator.MTP_EV_DEFAULT_COSTS, 3, false, 10000), src.roundCost(3, false), 1e-4);
    try testing.expectApproxEqAbs(Generator.mtpEvRoundCostAt(Generator.MTP_EV_DEFAULT_COSTS, 2, false, 10000), src.roundCost(2, false), 1e-4);
    try testing.expect(src.marginal(5) > 2.5 * src.marginal(4));
    // Past the widest measured width the cliff's slope continues (it is
    // steeper than the prior's marginal here).
    try testing.expectApproxEqAbs(src.roundCost(5, false) + (src.roundCost(5, false) - src.roundCost(4, false)), src.roundCost(6, false), 1e-3);
    const measured = Generator.mtpEvPlanSrc(&a, 8, src, 8);
    try testing.expectEqual(@as(u32, 4), measured.m_hi);
    try testing.expect(measured.m_lo <= 4);
    // ONE sample at the cliff (untrusted) already floors the horizon.
    var t1 = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = t1.observe(3, 10000, 60.0, 4.0, true, false);
        _ = t1.observe(4, 10000, 70.0, 5.0, true, false);
    }
    _ = t1.observe(5, 10000, 102.0, 6.0, true, false);
    const src1 = Generator.MtpCostSource.init(Generator.MTP_EV_DEFAULT_COSTS, 10000, &t1);
    try testing.expectEqual(@as(u32, 4), Generator.mtpEvPlanSrc(&a, 8, src1, 8).m_hi);
    // ...and the trial stops asking for it (two-chunk plan, nothing owed).
    try testing.expect(Generator.mtpWidthTrialTarget(&t1, 10000, .{ .m_lo = 4, .m_hi = 5, .tau_ln = 0 }, 8, true) == null);
    // An unmeasured base is trialled only under a two-chunk plan with a settled base.
    const empty_t = round_cost.Table{};
    try testing.expect(Generator.mtpWidthTrialTarget(&empty_t, 1000, .{ .m_lo = 3, .m_hi = 3, .tau_ln = 0 }, 8, true) == null);
    try testing.expect(Generator.mtpWidthTrialTarget(&empty_t, 1000, .{ .m_lo = 3, .m_hi = 5, .tau_ln = 0 }, 8, false) == null);
    try testing.expectEqual(@as(u32, 3), Generator.mtpWidthTrialTarget(&empty_t, 1000, .{ .m_lo = 3, .m_hi = 5, .tau_ln = 0 }, 8, true).?);
    try testing.expect(Generator.mtpWidthTrialPeriod(&t1, 10000, 4) >= 100);
    // An inactive bucket with no active neighbour is the prior, verbatim.
    const empty = round_cost.Table{};
    const cold = Generator.MtpCostSource.init(Generator.MTP_EV_DEFAULT_COSTS, 10000, &empty);
    try testing.expect(!cold.fromTable());
    const cold_plan = Generator.mtpEvPlanSrc(&a, 8, cold, 8);
    try testing.expectEqual(prior.m_hi, cold_plan.m_hi);
    try testing.expectEqual(prior.m_lo, cold_plan.m_lo);
}

test "mtpBasePlan: the accept policy plans one chunk from the acceptance EMAs, whatever the table's token column says" {
    // Round TIME favours depth 3; the w3 cell's token count froze pessimistic (one bad trial).
    var t = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = t.observe(2, 1000, 42.0, 3.0, true, false);
        _ = t.observe(3, 1000, 48.0, 2.2, true, false);
        _ = t.observe(4, 1000, 56.0, 3.9, true, false);
    }
    const a = [_]f32{ 0.9, 0.85, 0.8, 0.6, 0.5, 0.5 };
    const src = Generator.MtpCostSource.init(Generator.MTP_EV_DEFAULT_COSTS, 1000, &t);
    const legacy = Generator.mtpBasePlan(.legacy, &a, 6, src, 6);
    try testing.expect(legacy.m_lo != 3);
    const plan = Generator.mtpBasePlan(.accept, &a, 6, src, 6);
    try testing.expectEqual(@as(u32, 3), plan.m_lo);
    try testing.expectEqual(plan.m_lo, plan.m_hi); // no chunk B: no confidence sync, drafts stay on the rerank path
    // Long context stays on the legacy planner, whatever is configured.
    try testing.expectEqual(Generator.MtpDepthPolicy.accept, Generator.mtpDepthPolicyFor(.accept, 8191));
    try testing.expectEqual(Generator.MtpDepthPolicy.legacy, Generator.mtpDepthPolicyFor(.accept, 8192));
    try testing.expectEqual(Generator.MtpDepthPolicy.legacy, Generator.mtpDepthPolicyFor(.legacy, 100));
}

test "MtpCostSource: under the accept policy one clean time sample prices an unmeasured width" {
    var t = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = t.observe(3, 1000, 45.0, 4.0, true, false);
        _ = t.observe(4, 1000, 51.0, 5.0, true, false);
    }
    _ = t.observe(5, 1000, 57.0, 6.0, true, false);
    var src = Generator.MtpCostSource.init(Generator.MTP_EV_DEFAULT_COSTS, 1000, &t);
    const guarded = src.roundCost(5, false) / src.scale;
    try testing.expect(guarded > 58.0); // the prior's marginal, floored by the sample
    src.table_tokens = false;
    try testing.expectApproxEqAbs(@as(f32, 57.0), src.roundCost(5, false) / src.scale, 1e-3);
}

test "mtpEvPlanFor: DEFAULT costs carry the post-sdpa-split surface (2026-08-15 refit #4)" {
    // Pins MTP_EV_DEFAULT_COSTS to the surface measured AFTER the hd-256
    // causal sdpa split (same-session saturated forced-depth echo sweep,
    // Jundot oQ4e 27B @8K cold reps, M4 Max: T(1)=44.6 .. T(8)=142.3 →
    // floor ≈ 38.2 ms, marginals k<=4 ≈ 0.20, k5-6 ≈ 0.36, k7-8 ≈ 0.62).
    // flat_max 4: the old hi over-priced k4 at 0.34 (measured 0.24), so
    // moderate content under-drafted it. The k>=7 third region carries the
    // plain-SIMD verify-qmm register cliff — only reachable when
    // --mtp-max-depth forces past the generic cap of 6.
    const costs = Generator.MTP_EV_DEFAULT_COSTS;
    // Hot uniform 90%: base now rides the WIDER flat region (m_lo 4, was 3
    // under refit #3), extension one step into the ramp.
    const hot = [_]f32{ 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9 };
    const hot_plan = Generator.mtpEvPlanFor(&hot, 8, costs, 8);
    try testing.expectEqual(@as(u32, 4), hot_plan.m_lo);
    try testing.expectEqual(@as(u32, 5), hot_plan.m_hi);
    try testing.expectApproxEqAbs(@as(f32, -0.0943), hot_plan.tau_ln, 5e-3);
    // Marginal 75%: base stays 3, one cheap flat extension position.
    const mid = [_]f32{ 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75 };
    const mid_plan = Generator.mtpEvPlanFor(&mid, 8, costs, 8);
    try testing.expectEqual(@as(u32, 3), mid_plan.m_lo);
    try testing.expectEqual(@as(u32, 4), mid_plan.m_hi);
    try testing.expectApproxEqAbs(@as(f32, -0.786), mid_plan.tau_ln, 5e-3);
    // The oQ4e head's measured decayed chain still plans a 2-deep base.
    const oq4e = [_]f32{ 0.78, 0.50, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45 };
    const oq4e_plan = Generator.mtpEvPlanFor(&oq4e, 8, costs, 8);
    try testing.expectEqual(@as(u32, 2), oq4e_plan.m_lo);
}

test "mtpEvPlanFor: cap 1 is a plain depth-1 round" {
    const costs = Generator.MTP_EV_DEFAULT_COSTS;
    const a = [_]f32{ 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9 };
    const plan = Generator.mtpEvPlanFor(&a, 1, costs, 8);
    try testing.expectEqual(@as(u32, 1), plan.m_lo);
    try testing.expectEqual(@as(u32, 1), plan.m_hi);
}

fn mtpBoundsPlanFixture(xfm: *Transformer, prompt: []u32, depth: u32) Generator {
    xfm.config.model_type = "qwen4_exp";
    xfm.round_cost = .{ .layout = .long, .first_use_logged = true };
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = xfm.round_cost.observe(2, 1000, 29.0, 2.6, true, false);
        _ = xfm.round_cost.observe(3, 1000, 31.0, 2.9, true, false);
        _ = xfm.round_cost.observe(4, 1000, 36.0, 3.1, true, false);
        _ = xfm.round_cost.observe(5, 1000, 42.0, 3.5, true, false);
    }
    return Generator{
        .xfm = xfm,
        .ctx = undefined,
        .tok = undefined,
        .next_token_id = 0,
        .step = 0,
        .max_tokens = 1,
        .sampling = .{ .temperature = 0 },
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .finish_reason = "length",
        .done = false,
        .eos_token_ids = &.{},
        .generated_ids = .empty,
        .timeout_ns = 0,
        .timer = io_util.Stopwatch.init(testing.io),
        .last_hidden = .{ .ctx = null },
        .has_last_hidden = false,
        .prompt_ids_owned = prompt,
        .mtp_depth = depth,
        .mtp_depth_free = depth,
        .mtp_ev_rounds = 64,
        .mtp_ev_accept = .{ 0.92, 0.9, 0.88, 0.86, 0.84, 0.8, 0.8, 0.8 },
        .mtp_ev_m_lo_prev = 4,
        .spec_cost_solo = false,
    };
}

test "MTP depth bounds: an EV plan outside min..max is clamped, one inside is untouched" {
    const wide = Generator.MtpRoundPlan{ .m_lo = 3, .m_hi = 7, .tau_ln = -1.5 };
    const low = Generator.mtpPlanInBounds(wide, 5, 8);
    try testing.expectEqual(@as(u32, 5), low.m_lo);
    try testing.expectEqual(@as(u32, 7), low.m_hi);
    try testing.expectEqual(@as(f32, -1.5), low.tau_ln);

    const high = Generator.mtpPlanInBounds(wide, 1, 2);
    try testing.expectEqual(@as(u32, 2), high.m_lo);
    try testing.expectEqual(@as(u32, 2), high.m_hi);
    // The extension collapsed: no confidence read, as in any single-chunk plan.
    try testing.expectEqual(@as(f32, 0.0), high.tau_ln);

    const pinned = Generator.mtpPlanInBounds(wide, 4, 4);
    try testing.expectEqual(@as(u32, 4), pinned.m_lo);
    try testing.expectEqual(@as(u32, 4), pinned.m_hi);

    const inside = Generator.mtpPlanInBounds(wide, 1, 8);
    try testing.expectEqual(wide, inside);
}

test "MTP depth bounds: both policies keep table-priced plans inside min..max" {
    const before = depth_bounds.active;
    const policy_before = mtp_depth_policy_override;
    defer {
        depth_bounds.active = before;
        mtp_depth_policy_override = policy_before;
    }
    var xfm: Transformer = undefined;
    var prompt: [1000]u32 = @splat(0);
    for ([_]Generator.MtpDepthPolicy{ .accept, .legacy }) |policy| {
        mtp_depth_policy_override = policy;
        depth_bounds.active = .{};
        var gen = mtpBoundsPlanFixture(&xfm, &prompt, 6);
        try testing.expectEqual(@as(u32, if (policy == .accept) 4 else 3), gen.mtpRoundPlan().m_lo);

        depth_bounds.active = .{ .min = 5 };
        gen = mtpBoundsPlanFixture(&xfm, &prompt, 6);
        var plan = gen.mtpRoundPlan();
        try testing.expect(plan.m_lo >= 5 and plan.m_hi >= plan.m_lo and plan.m_hi <= 6);

        depth_bounds.active = .{ .max = 2 };
        gen = mtpBoundsPlanFixture(&xfm, &prompt, 2);
        plan = gen.mtpRoundPlan();
        try testing.expect(plan.m_lo <= 2 and plan.m_hi <= 2);
    }
}

test "MTP depth bounds: min == max drafts that depth every round" {
    const before = depth_bounds.active;
    defer depth_bounds.active = before;
    var xfm: Transformer = undefined;
    var prompt: [1000]u32 = @splat(0);
    for ([_]u32{ 2, 4, 5 }) |depth| {
        depth_bounds.active = .{ .min = depth, .max = depth };
        var gen = mtpBoundsPlanFixture(&xfm, &prompt, depth);
        for (0..4) |_| {
            const plan = gen.mtpRoundPlan();
            try testing.expectEqual(depth, plan.m_lo);
            try testing.expectEqual(depth, plan.m_hi);
        }
    }
}

test "MTP depth bounds: the windowed controller never leaves min..max and never disables above a floor of 1" {
    const before = depth_bounds.active;
    defer depth_bounds.active = before;
    depth_bounds.active = .{ .min = 3, .max = 5 };
    var hot = mtpEvTestGenerator();
    hot.mtp_depth = 5;
    hot.mtp_depth_current = 3;
    for (0..64) |_| {
        hot.updateMtpDepth(5, 5);
        try testing.expect(hot.mtp_depth_current >= 3 and hot.mtp_depth_current <= 5);
    }
    try testing.expectEqual(@as(u32, 5), hot.mtp_depth_current);

    var cold = mtpEvTestGenerator();
    cold.mtp_depth = 5;
    cold.mtp_depth_current = 5;
    for (0..64) |_| {
        cold.updateMtpDepth(5, 0);
        try testing.expect(cold.mtp_depth_current >= 3 and cold.mtp_depth_current <= 5);
    }
    try testing.expectEqual(@as(u32, 3), cold.mtp_depth_current);
    try testing.expect(!cold.spec_disabled_runtime);
}

test "MTP depth bounds: a min above the default cap lifts both caps, an explicit max leaves them" {
    const before = depth_bounds.active;
    defer depth_bounds.active = before;
    depth_bounds.active = .{ .min = 7 };
    try testing.expectEqual(@as(u32, 7), Generator.mtpDepthCapForProfile(0, true, .generic));
    try testing.expectEqual(@as(u32, 7), Generator.mtpDepthCapFree(0));
    depth_bounds.active = .{ .min = 2, .max = 4 };
    try testing.expectEqual(@as(u32, 4), Generator.mtpDepthCapForProfile(4, true, .generic));
    try testing.expectEqual(@as(u32, 4), Generator.mtpDepthCapFree(4));
}

test "MTP depth bounds: warmup, group fill and native head caps constrain the final plan" {
    const before = depth_bounds.active;
    defer depth_bounds.active = before;
    var xfm: Transformer = undefined;
    var prompt: [1000]u32 = @splat(0);
    depth_bounds.active = .{ .min = 3, .max = 5 };
    var gen = mtpBoundsPlanFixture(&xfm, &prompt, 5);
    gen.mtp_ev_rounds = 0;
    gen.mtp_depth_current = 1;
    gen.mtp_batch_head = true; // No serial probe on a grouped lane.
    try testing.expectEqual(@as(u32, 3), gen.mtpRoundPlan().m_lo);
    gen.mtp_group_cap = 8;
    gen.mtp_group_fill = true;
    try testing.expectEqual(@as(u32, 5), gen.mtpRoundPlan().m_hi);
    gen.mtp_group_cap = 2; // The verify row budget wins over the floor.
    try testing.expectEqual(@as(u32, 2), gen.mtpRoundPlan().m_lo);
    gen.mtp_group_cap = 0;
    gen.mtp_depth = 3; // Native multi-head cap resolved at initialization.
    depth_bounds.active = .{ .min = 5, .max = 5 };
    try testing.expectEqual(@as(u32, 3), gen.mtpRoundPlan().m_hi);
}
