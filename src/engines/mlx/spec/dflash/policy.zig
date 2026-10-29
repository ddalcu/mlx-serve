//! DFlash round policy for a sparse target, where a verify row is NOT free (GLM-5.3-Flash: every
//! extra row routes to more distinct experts). Pure data, no MLX: the generator feeds it each
//! round and reads the next decision back.
//!
//! A round drafts once, then verifies the first R rows (t1 and R - 1 drafts). R maximizes
//! emitted tokens per ms under (a) a per-position acceptance probability calibrated online
//! from the drafter's confidence and (b) the cost of an R-row verify. R = 1 is a plain round.
//! When drafts keep failing to pay for their own forward, drafting is skipped with a probe backoff.
const std = @import("std");

/// t1 plus up to 15 drafts.
pub const MAX_ROWS: u32 = 16;
pub const MAX_DRAFTS: u32 = MAX_ROWS - 1;

/// Round ms by verify rows; index = rows, 0 unused.
pub const Costs = [MAX_ROWS + 1]f32;

/// GLM-5.3-Flash verify ladder on an M5 Ultra at 1k context (ms by rows, linear between).
const LADDER = [_][2]f32{ .{ 1, 17.7 }, .{ 2, 25.5 }, .{ 3, 29.6 }, .{ 4, 33.6 }, .{ 5, 36.8 }, .{ 6, 42.4 }, .{ 7, 44.8 }, .{ 8, 47.9 }, .{ 9, 51.1 }, .{ 12, 60.5 }, .{ 16, 69.1 } };

/// The ladder scaled so one row costs `plain_ms`: a faster or slower machine moves it whole.
pub fn priorCosts(plain_ms: f32) Costs {
    var out: Costs = undefined;
    out[0] = 0;
    const scale = plain_ms / LADDER[0][1];
    for (1..MAX_ROWS + 1) |r| {
        const rows: f32 = @floatFromInt(r);
        var ms: f32 = LADDER[LADDER.len - 1][1];
        for (LADDER[0 .. LADDER.len - 1], LADDER[1..]) |lo, hi| {
            if (rows <= hi[0]) {
                ms = lo[1] + (hi[1] - lo[1]) * (rows - lo[0]) / (hi[0] - lo[0]);
                break;
            }
        }
        out[r] = ms * scale;
    }
    return out;
}

/// A plain round's ms when nothing has been measured yet.
pub const PRIOR_PLAIN_MS: f32 = LADDER[0][1];

pub const Choice = struct {
    /// Rows to verify, t1 included (1 = a plain round).
    rows: u32,
    /// Expected tokens the round emits.
    tokens: f32,
    /// Verify cost of `rows`.
    ms: f32,
};

/// A row count within this fraction of the best rate loses to the cheaper one: the model is not
/// that exact, and a verify's cost is certain where its tokens are not.
const ROW_TIE: f32 = 0.02;

/// The row count with the best expected tokens per ms. `p[k]` is the probability that draft k is
/// accepted given drafts 0..k-1 were; `costs` are the verify ms by rows.
pub fn chooseRows(p: []const f32, costs: *const Costs) Choice {
    const n: usize = @min(p.len, MAX_DRAFTS); // typed: a bare @min narrows to u4 and n + 2 overflows at 15
    var tokens: [MAX_ROWS + 1]f32 = undefined;
    tokens[1] = 1;
    var alive: f32 = 1;
    for (p[0..n], 0..) |pk, k| {
        alive *= if (pk >= 0) @min(pk, 1) else 0; // a NaN is no acceptance
        tokens[k + 2] = tokens[k + 1] + alive;
    }
    var best: f32 = 0;
    for (1..n + 2) |r| best = @max(best, tokens[r] / costs[r]);
    for (1..n + 2) |r| {
        if (tokens[r] / costs[r] >= best * (1 - ROW_TIE)) return .{ .rows = @intCast(r), .tokens = tokens[r], .ms = costs[r] };
    }
    return .{ .rows = 1, .tokens = 1, .ms = costs[1] };
}

/// Probability that a draft is accepted given its predecessors were, by the drafter's confidence in
/// it: online bins over `-log10(1 - conf)` (the selector's shares crowd 0.9..1, where a linear axis has one
/// bin), each pulled toward a logistic prior fitted on recorded rounds by a few pseudo-counts and forgetting
/// its old rounds, so a change of content class re-calibrates in a few rounds.
pub const Calibration = struct {
    const BINS = 12;
    const PER_UNIT: f32 = 4;
    /// Pseudo-observations the prior is worth.
    const PRIOR_WEIGHT: f32 = 3;
    /// A bin forgets this share of its history per observation it takes.
    const KEEP: f32 = 0.97;

    hit: [BINS]f32 = @splat(0),
    seen: [BINS]f32 = @splat(0),

    /// 0 at a coin flip's confidence, 1 at 0.9, 2 at 0.99, capped at 3.
    fn axis(conf: f32) f32 {
        return -std.math.log10(@max(1 - std.math.clamp(conf, 0, 1), 1e-3));
    }

    fn bin(conf: f32) usize {
        return @min(BINS - 1, @as(usize, @intFromFloat(axis(conf) * PER_UNIT)));
    }

    fn prior(conf: f32) f32 {
        return 1 / (1 + @exp(-(-1.15 + 1.58 * axis(conf))));
    }

    pub fn probability(self: *const Calibration, conf: f32) f32 {
        const b = bin(conf);
        return (self.hit[b] + PRIOR_WEIGHT * prior(conf)) / (self.seen[b] + PRIOR_WEIGHT);
    }

    pub fn observe(self: *Calibration, conf: f32, accepted: bool) void {
        const b = bin(conf);
        self.hit[b] = self.hit[b] * KEEP + @as(f32, if (accepted) 1 else 0);
        self.seen[b] = self.seen[b] * KEEP + 1;
    }
};

/// What a plain round costs, learned from the plain rounds themselves and nothing else: slowly, and by at
/// most a quarter per sample, so a stall or a foreign GPU job cannot pose as the machine's speed.
pub const PlainCost = struct {
    ms: f32 = PRIOR_PLAIN_MS,

    pub fn sample(self: *PlainCost, ms: f32) void {
        if (!std.math.isFinite(ms) or ms <= 0) return;
        const bounded = std.math.clamp(ms, self.ms * 0.8, self.ms * 1.25);
        self.ms += 0.1 * (bounded - self.ms);
    }
};

/// How often a copied continuation survives verify, by how far back it agreed with the context (under or
/// over `STRONG`): the share of positions reached that were kept, forgetting old rounds.
pub const CopyStats = struct {
    pub const STRONG: u32 = 16;
    const PRIOR = [2]f32{ 0.7, 0.9 };
    const PRIOR_WEIGHT: f32 = 3;
    const KEEP: f32 = 0.95;

    hit: [2]f32 = .{ PRIOR[0] * PRIOR_WEIGHT, PRIOR[1] * PRIOR_WEIGHT },
    tries: [2]f32 = .{ PRIOR_WEIGHT, PRIOR_WEIGHT },

    fn class(suffix: u32) usize {
        return @intFromBool(suffix >= STRONG);
    }

    pub fn probability(self: *const CopyStats, suffix: u32) f32 {
        const c = class(suffix);
        return self.hit[c] / self.tries[c];
    }

    /// A round that verified `drafts` copied tokens and kept `accepted` of them.
    pub fn observe(self: *CopyStats, suffix: u32, drafts: u32, accepted: u32) void {
        const c = class(suffix);
        self.hit[c] = self.hit[c] * KEEP + @as(f32, @floatFromInt(accepted));
        self.tries[c] = self.tries[c] * KEEP + @as(f32, @floatFromInt(@min(accepted + 1, drafts)));
    }
};

/// Whether this round drafts at all. Drafting costs a forward whether or not its rows are kept, so
/// a stretch where it does not pay is skipped with a probe backoff: each probe that loses doubles
/// the plain rounds before the next, a probe that wins clears the debt.
pub const Drafting = struct {
    /// Longest stretch of plain rounds between two probes.
    const MAX_PERIOD: u32 = 32;
    const ALPHA: f32 = 0.3;

    /// Averages over drafted rounds of the plain-round ms their tokens replaced and of the ms they took:
    /// the ratio weighs a round by its time, where a mean of per-round ratios swings on every short miss.
    gain: f32 = 0,
    wall: f32 = 0,
    /// Plain rounds still owed before the next draft.
    owed: u32 = 0,
    /// Plain rounds the next losing round will owe.
    period: u32 = 1,
    /// Rounds skipped, for stats.
    skipped: u64 = 0,
    /// Drafted rounds that followed a skip stretch.
    probes: u64 = 0,
    after_skip: bool = false,

    pub fn shouldDraft(self: *Drafting) bool {
        if (self.owed > 0) {
            self.owed -= 1;
            self.skipped += 1;
            self.after_skip = true;
            return false;
        }
        if (self.after_skip) self.probes += 1;
        self.after_skip = false;
        return true;
    }

    /// A drafted round: `gain` = the ms of plain rounds its tokens would have cost, `wall` = what it took.
    pub fn record(self: *Drafting, gain: f32, wall: f32) void {
        if (self.wall <= 0) {
            self.gain = gain;
            self.wall = wall;
        } else {
            self.gain += ALPHA * (gain - self.gain);
            self.wall += ALPHA * (wall - self.wall);
        }
        if (self.gain >= self.wall) {
            self.owed = 0;
            self.period = @max(1, self.period / 2);
        } else {
            self.owed = self.period;
            self.period = @min(self.period * 2, MAX_PERIOD);
        }
    }
};

/// Per-draft-position acceptance and the verify widths a request ran, for `[spec-stats]`.
pub const PositionStats = struct {
    reached: [MAX_DRAFTS]u64 = @splat(0),
    accepted: [MAX_DRAFTS]u64 = @splat(0),
    /// Rounds by verify rows; index = rows.
    rows: [MAX_ROWS + 1]u64 = @splat(0),

    /// A round that verified `drafts` drafts and kept `accepted` of them: it reached every position
    /// up to the first rejection and no further.
    pub fn record(self: *PositionStats, drafts: u32, accepted: u32) void {
        const m: usize = @min(drafts, MAX_DRAFTS);
        self.rows[m + 1] += 1;
        for (0..m) |k| {
            if (k > accepted) break;
            self.reached[k] += 1;
            if (k < accepted) self.accepted[k] += 1;
        }
    }

    /// `1.00/0.95/...`: conditional acceptance by draft position, down to the deepest one any round reached.
    pub fn formatAcc(self: *const PositionStats, buf: []u8) []const u8 {
        var w = std.Io.Writer.fixed(buf);
        for (self.reached, self.accepted, 0..) |reached, kept, k| {
            if (reached == 0) break;
            if (k > 0) w.writeAll("/") catch break;
            w.print("{d:.2}", .{@as(f32, @floatFromInt(kept)) / @as(f32, @floatFromInt(reached))}) catch break;
        }
        return w.buffered();
    }

    /// `1:12,2:3,8:100`: rounds by verify rows.
    pub fn formatRows(self: *const PositionStats, buf: []u8) []const u8 {
        var w = std.Io.Writer.fixed(buf);
        var first = true;
        for (self.rows, 0..) |n, rows| {
            if (n == 0) continue;
            if (!first) w.writeAll(",") catch break;
            first = false;
            w.print("{d}:{d}", .{ rows, n }) catch break;
        }
        return w.buffered();
    }
};

const testing = std.testing;

fn approx(a: f32, b: f32) !void {
    try testing.expectApproxEqAbs(b, a, 0.02);
}

test "priorCosts: scaled to the measured plain step, never cheaper for more rows" {
    const c = priorCosts(17.7);
    try testing.expectApproxEqAbs(@as(f32, 17.7), c[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 25.5), c[2], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 47.9), c[8], 1e-3);
    var r: u32 = 2;
    while (r <= MAX_ROWS) : (r += 1) try testing.expect(c[r] >= c[r - 1]);
    // A faster machine's plain step moves the whole ladder with it.
    const half = priorCosts(8.85);
    try testing.expectApproxEqAbs(c[8] / 2, half[8], 1e-3);
}

test "chooseRows: a plain round when nothing pays, the whole chain when everything does" {
    const costs = priorCosts(17.7);
    const weak = [_]f32{ 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1 };
    try testing.expectEqual(@as(u32, 1), chooseRows(&weak, &costs).rows);
    const sure: [7]f32 = @splat(0.96);
    const c = chooseRows(&sure, &costs);
    try testing.expectEqual(@as(u32, 8), c.rows);
    try approx(c.tokens, 6.96);
    try approx(c.ms, 47.9);
}

test "chooseRows: a fifteen-draft chain is priced through its last row, and a poisoned input still answers" {
    const costs = priorCosts(17.7);
    const sure: [15]f32 = @splat(0.98);
    const c = chooseRows(&sure, &costs);
    try testing.expect(c.rows >= 8 and c.rows <= 16);
    // A NaN probability must not leave the function without a choice.
    const bad: [3]f32 = .{ std.math.nan(f32), 0.9, 0.9 };
    try testing.expect(chooseRows(&bad, &costs).rows >= 1);
}

test "chooseRows: the second row pays for its own +8 ms jump only past 48% acceptance" {
    const costs = priorCosts(17.7);
    try testing.expectEqual(@as(u32, 1), chooseRows(&[_]f32{0.40}, &costs).rows);
    try testing.expectEqual(@as(u32, 2), chooseRows(&[_]f32{0.60}, &costs).rows);
}

test "chooseRows: rows past a poor draft are not bought for the confident draft behind it" {
    const costs = priorCosts(17.7);
    // Draft 1 almost never lands, so draft 2 is only reached 5% of the time.
    const c = chooseRows(&[_]f32{ 0.95, 0.05, 0.99, 0.99 }, &costs);
    try testing.expectEqual(@as(u32, 2), c.rows);
}

test "chooseRows: within 2% of the best rate the cheaper verify wins" {
    var costs: Costs = @splat(20.0);
    costs[1] = 10.0;
    costs[2] = 12.0;
    costs[3] = 12.2;
    // The third row buys 1.5% over two rows: not worth verifying.
    try testing.expectEqual(@as(u32, 2), chooseRows(&[_]f32{ 0.5, 0.0958 }, &costs).rows);
    // At 15% it is.
    try testing.expectEqual(@as(u32, 3), chooseRows(&[_]f32{ 0.5, 0.5 }, &costs).rows);
}

test "Calibration: no data answers the prior, then the data takes over, bin by bin" {
    var c = Calibration{};
    try approx(c.probability(0.9), 0.606);
    try approx(c.probability(0.2), 0.269);
    for (0..40) |_| c.observe(0.9, false);
    try testing.expect(c.probability(0.9) < 0.3);
    try approx(c.probability(0.2), 0.269);
    // The bin is a confidence range: 0.88 shares 0.9's evidence.
    try testing.expect(c.probability(0.88) < 0.3);
}

test "Calibration: an old regime fades" {
    var c = Calibration{};
    for (0..80) |_| c.observe(0.7, true);
    try testing.expect(c.probability(0.7) > 0.9);
    for (0..80) |_| c.observe(0.7, false);
    try testing.expect(c.probability(0.7) < 0.2);
}

// A drafted round is judged by `record(gain, wall)`: the plain-round ms its tokens would have cost, against
// the ms it took. Ten against ten is break-even.

test "PlainCost: a stall moves it little, a steady new speed is reached" {
    var c = PlainCost{};
    c.sample(400); // a compile stall, a foreign GPU job
    try testing.expect(c.ms > PRIOR_PLAIN_MS and c.ms < 18.5);
    var faster = PlainCost{};
    for (0..80) |_| faster.sample(14.0);
    try testing.expect(faster.ms < 14.5 and faster.ms >= 14.0);
    var slower = PlainCost{};
    for (0..80) |_| slower.sample(22.0);
    try testing.expect(slower.ms > 21.0 and slower.ms <= 22.0);
}

test "CopyStats: a strong match starts trusted, and misses teach it otherwise" {
    var c = CopyStats{};
    try testing.expect(c.probability(20) > c.probability(8));
    for (0..12) |_| c.observe(20, 14, 0);
    try testing.expect(c.probability(20) < 0.3);
    try approx(c.probability(8), 0.7); // the other class is untouched
    var d = CopyStats{};
    for (0..12) |_| d.observe(20, 14, 14);
    try testing.expect(d.probability(20) > 0.9);
}

test "Drafting: a losing draft owes plain rounds, doubling to a cap; a winning probe resets" {
    var d = Drafting{};
    try testing.expect(d.shouldDraft());
    d.record(7, 10); // loses: one plain round owed
    try testing.expect(!d.shouldDraft());
    try testing.expect(d.shouldDraft());
    d.record(7, 10); // loses again: two owed
    try testing.expect(!d.shouldDraft());
    try testing.expect(!d.shouldDraft());
    try testing.expect(d.shouldDraft());
    var guard: u32 = 0;
    // However long it loses, the wait stops growing at 32 plain rounds.
    while (guard < 12) : (guard += 1) {
        d.record(5, 10);
        var owed: u32 = 0;
        while (!d.shouldDraft()) owed += 1;
        try testing.expect(owed <= 32);
    }
    d.record(5, 10);
    var owed: u32 = 0;
    while (!d.shouldDraft()) owed += 1;
    try testing.expectEqual(@as(u32, 32), owed);
    // A probe that pays clears the debt.
    d.record(60, 10);
    try testing.expect(d.shouldDraft());
    try testing.expect(d.probes > 0 and d.skipped > 0);
}

test "Drafting: a thin win with scattered misses keeps drafting, because the rounds are weighed by their time" {
    var d = Drafting{};
    // 1.6x on average, a quarter of the rounds below break-even: the mean of the ratios would dip under 1.
    const rounds = [_][2]f32{ .{ 70, 40 }, .{ 18, 40 }, .{ 70, 40 }, .{ 70, 40 }, .{ 18, 40 }, .{ 120, 45 }, .{ 18, 40 }, .{ 120, 45 }, .{ 70, 40 }, .{ 18, 40 }, .{ 70, 40 }, .{ 120, 45 } };
    for (rounds) |r| {
        try testing.expect(d.shouldDraft());
        d.record(r[0], r[1]);
    }
}

test "Drafting: one failed round among winners does not stop drafting" {
    var d = Drafting{};
    for (0..6) |_| {
        try testing.expect(d.shouldDraft());
        d.record(25, 10);
    }
    try testing.expect(d.shouldDraft());
    d.record(6, 10);
    try testing.expect(d.shouldDraft());
}

test "PositionStats: conditional acceptance counts only the positions a round reached" {
    var st = PositionStats{};
    st.record(7, 7); // every position reached and kept
    st.record(7, 2); // 0 and 1 kept, 2 rejected, 3.. never reached
    st.record(0, 0); // a plain round
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("1.00/1.00/0.50/1.00/1.00/1.00/1.00", st.formatAcc(&buf));
    try testing.expectEqualStrings("1:1,8:2", st.formatRows(&buf));
}

test "PositionStats: the widest copied chain records without overflow" {
    var st = PositionStats{};
    st.record(15, 15);
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("16:1", st.formatRows(&buf));
}

test "PositionStats: a position no round reached is not listed" {
    var st = PositionStats{};
    st.record(3, 0);
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("0.00", st.formatAcc(&buf));
    st.record(5, 5);
    try testing.expectEqualStrings("0.50/1.00/1.00/1.00/1.00", st.formatAcc(&buf));
}
