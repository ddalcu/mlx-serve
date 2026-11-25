//! How wide a speculative round runs. Pure data, no MLX and no clock: a model's round costs are a `Ladder` timed once
//! at load, a request's acceptance is counted from its own tokens, and a round verifies the rows that buy the most
//! expected tokens per ms. Nothing here reads time inside a request, so a seeded request makes the same choices again.
const std = @import("std");

/// t1 plus up to 15 drafts.
pub const MAX_ROWS: u32 = 16;
pub const MAX_DRAFTS: u32 = MAX_ROWS - 1;

/// Round ms by verify rows; index = rows, 0 unused.
pub const Costs = [MAX_ROWS + 1]f32;

pub const Choice = struct {
    /// Rows to verify, t1 included (1 = a plain round).
    rows: u32,
    /// Expected tokens the round emits.
    tokens: f32,
    /// Cost of `rows`.
    ms: f32,

    pub fn rate(c: Choice) f32 {
        return c.tokens / c.ms;
    }
};

/// A row count within this fraction of the best rate loses to the cheaper one: the model is not
/// that exact, and a verify's cost is certain where its tokens are not.
pub const ROW_TIE: f32 = 0.02;

/// The row count with the best expected tokens per ms. `p[k]` is the probability that draft k is
/// accepted given drafts 0..k-1 were; `costs` are the round's ms by rows.
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
    /// The drafter's confidence is its own acceptance estimate (DSpark's confidence head, a draft's probability),
    /// the prior the bins start from; else the logistic fit.
    head: bool = false,

    /// 0 at a coin flip's confidence, 1 at 0.9, 2 at 0.99, capped at 3.
    fn axis(conf: f32) f32 {
        return -std.math.log10(@max(1 - std.math.clamp(conf, 0, 1), 1e-3));
    }

    fn bin(conf: f32) usize {
        return @min(BINS - 1, @as(usize, @intFromFloat(axis(conf) * PER_UNIT)));
    }

    fn prior(self: *const Calibration, conf: f32) f32 {
        if (self.head) return std.math.clamp(conf, 0, 1);
        return 1 / (1 + @exp(-(-1.15 + 1.58 * axis(conf))));
    }

    pub fn probability(self: *const Calibration, conf: f32) f32 {
        const b = bin(conf);
        return (self.hit[b] + PRIOR_WEIGHT * self.prior(conf)) / (self.seen[b] + PRIOR_WEIGHT);
    }

    pub fn observe(self: *Calibration, conf: f32, accepted: bool) void {
        const b = bin(conf);
        self.hit[b] = self.hit[b] * KEEP + @as(f32, if (accepted) 1 else 0);
        self.seen[b] = self.seen[b] * KEEP + 1;
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

/// The copy round that pays most, of a copy `len` tokens long that agreed `suffix` tokens back, under `costs`.
pub fn copyChoice(copy: *const CopyStats, suffix: u32, len: usize, costs: *const Costs) Choice {
    var p: [MAX_DRAFTS]f32 = undefined;
    const n = @min(len, MAX_DRAFTS);
    @memset(p[0..n], copy.probability(suffix));
    return chooseRows(p[0..n], costs);
}

/// How often a drafter's draft is kept, by its place in the block and given the drafts before it were: counts of this
/// request's own rounds, forgetting old ones. A place few rounds reached leans on the rate over all places, and that on
/// `prior`. Plain steps fade the counts, so a request that stopped drafting returns to the prior and drafts again.
pub const Acceptance = struct {
    /// What a drawn draft lands on prose, and what an argmax draft does under a greedy request.
    pub const SAMPLED: f32 = 0.7;
    pub const GREEDY: f32 = 0.9;
    /// Pseudo-observations the prior is worth to the overall rate, and the overall rate to a place.
    const PRIOR_WEIGHT: f32 = 6;
    const PLACE_WEIGHT: f32 = 4;
    /// The share of its history a place keeps per observation, the overall rate per round, every count per plain step.
    const KEEP: f32 = 0.95;
    const ROUND_KEEP: f32 = 0.9;
    const IDLE_KEEP: f32 = 0.9;

    prior: f32,
    hit: [MAX_DRAFTS]f32 = @splat(0),
    tries: [MAX_DRAFTS]f32 = @splat(0),
    all_hit: f32 = 0,
    all_tries: f32 = 0,

    /// The rate over every place.
    pub fn overall(self: *const Acceptance) f32 {
        return (self.all_hit + PRIOR_WEIGHT * self.prior) / (self.all_tries + PRIOR_WEIGHT);
    }

    /// `p[k]`: the probability draft k is kept given the drafts before it were.
    pub fn fill(self: *const Acceptance, p: []f32) void {
        const base = self.overall();
        for (p, self.hit[0..p.len], self.tries[0..p.len]) |*o, h, t| o.* = (h + PLACE_WEIGHT * base) / (t + PLACE_WEIGHT);
    }

    /// A round that verified `drafts` drafts and kept the first `accepted`: it tested them and the first one refused.
    pub fn observe(self: *Acceptance, drafts: u32, accepted: u32) void {
        const tested = @min(drafts, accepted + 1, MAX_DRAFTS);
        const kept = @min(accepted, tested);
        for (self.hit[0..tested], self.tries[0..tested], 0..) |*h, *t, k| {
            h.* = h.* * KEEP + @as(f32, if (k < kept) 1 else 0);
            t.* = t.* * KEEP + 1;
        }
        self.all_hit = self.all_hit * ROUND_KEEP + @as(f32, @floatFromInt(kept));
        self.all_tries = self.all_tries * ROUND_KEEP + @as(f32, @floatFromInt(tested));
    }

    /// A plain step.
    pub fn idle(self: *Acceptance) void {
        for (&self.hit, &self.tries) |*h, *t| {
            h.* *= IDLE_KEEP;
            t.* *= IDLE_KEEP;
        }
        self.all_hit *= IDLE_KEEP;
        self.all_tries *= IDLE_KEEP;
    }
};

/// What a model's rounds cost on this machine, in ms: timed once at load (`Timing`), never inside a request.
pub const Ladder = struct {
    /// A plain step of the serial pipeline.
    plain: f32,
    /// A round that verifies drafts it was handed, by their count (0 = none: a plain step run as a round).
    copies: [MAX_ROWS]f32,
    /// A round whose drafter drafts, by the drafts it verifies (0 = it drafted and verified none).
    drafts: [MAX_ROWS]f32,
    /// The most drafts the drafter makes.
    block: u32,

    /// The costs a copy round is chosen on: a plain step, then the round by its rows.
    pub fn copyCosts(l: *const Ladder) Costs {
        return l.costs(&l.copies, MAX_DRAFTS);
    }

    /// The costs a drafted round is chosen on.
    pub fn draftCosts(l: *const Ladder) Costs {
        return l.costs(&l.drafts, l.block);
    }

    fn costs(l: *const Ladder, by_drafts: *const [MAX_ROWS]f32, most: u32) Costs {
        var out: Costs = @splat(std.math.inf(f32));
        out[1] = l.plain;
        for (1..most + 1) |k| out[k + 1] = by_drafts[k];
        return out;
    }
};

/// The rounds a load runs to time a `Ladder`, and what they took. Each cell runs `REPS` rounds after one that settles
/// what the cell before it left queued, and keeps the least: a first launch compiles, a neighbour's work stalls one.
/// A plain step rides a pipeline whose ticks trade time with each other, so it is timed as the mean of a cell's
/// ticks, the least of `PLAIN_CELLS` cells.
pub const Timing = struct {
    pub const REPS: u32 = 4;
    pub const PLAIN_CELLS: u32 = 2;
    pub const Kind = enum { plain, copy, draft };
    pub const Round = struct { kind: Kind, drafts: u32 };

    block: u32,
    cell: u32 = 0,
    rep: u32 = 0,
    /// The plain cell's ticks so far.
    ticks: f32 = 0,
    plain: f32 = std.math.inf(f32),
    copies: [MAX_ROWS]f32 = @splat(std.math.inf(f32)),
    drafts: [MAX_ROWS]f32 = @splat(std.math.inf(f32)),

    pub fn cells(t: *const Timing) u32 {
        return PLAIN_CELLS + MAX_ROWS + t.block;
    }

    fn at(t: *const Timing) Round {
        if (t.cell < PLAIN_CELLS) return .{ .kind = .plain, .drafts = 0 };
        if (t.cell < PLAIN_CELLS + MAX_ROWS) return .{ .kind = .copy, .drafts = t.cell - PLAIN_CELLS };
        return .{ .kind = .draft, .drafts = t.cell + 1 - PLAIN_CELLS - MAX_ROWS };
    }

    /// The round to run now; null once every cell is timed.
    pub fn next(t: *const Timing) ?Round {
        return if (t.cell < t.cells()) t.at() else null;
    }

    /// The round `next` named ended `ms` after the round before it did.
    pub fn took(t: *Timing, ms: f32) void {
        const r = t.next() orelse return;
        if (t.rep > 0) switch (r.kind) {
            .plain => t.ticks += ms,
            .copy => if (std.math.isFinite(ms) and ms > 0) {
                t.copies[r.drafts] = @min(t.copies[r.drafts], ms);
            },
            .draft => if (std.math.isFinite(ms) and ms > 0) {
                t.drafts[r.drafts] = @min(t.drafts[r.drafts], ms);
            },
        };
        t.rep += 1;
        if (t.rep > REPS) {
            if (r.kind == .plain and std.math.isFinite(t.ticks) and t.ticks > 0) t.plain = @min(t.plain, t.ticks / REPS);
            t.ticks = 0;
            t.rep = 0;
            t.cell += 1;
        }
    }

    /// The ladder, once every cell is timed. A draft that verifies nothing costs what the drafter adds to a round
    /// (the least a drafted round took over the same round handed its drafts) over the round with none.
    pub fn ladder(t: *const Timing) ?Ladder {
        if (t.next() != null) return null;
        var l: Ladder = .{ .plain = t.plain, .copies = t.copies, .drafts = t.drafts, .block = t.block };
        var drafter = std.math.inf(f32);
        for (1..t.block + 1) |k| drafter = @min(drafter, t.drafts[k] - t.copies[k]);
        l.drafts[0] = t.copies[0] + @max(drafter, 0);
        for ([_]f32{ l.plain, l.drafts[0] } ++ l.copies) |ms| if (!std.math.isFinite(ms)) return null;
        for (l.drafts[1 .. t.block + 1]) |ms| if (!std.math.isFinite(ms)) return null;
        return l;
    }
};

/// A request's round chooser: the model's ladder, and what this request's rounds have kept.
pub const Controller = struct {
    ladder: Ladder,
    acc: Acceptance,
    copy: CopyStats = .{},

    pub const Round = struct { kind: Timing.Kind, drafts: u32 = 0 };

    /// The drafted round that pays most, of at most `cap` drafts; one row = a plain step pays more.
    pub fn drafted(c: *const Controller, cap: u32) Choice {
        var p: [MAX_DRAFTS]f32 = undefined;
        const n = @min(cap, c.ladder.block, MAX_DRAFTS);
        c.acc.fill(p[0..n]);
        return chooseRows(p[0..n], &c.ladder.draftCosts());
    }

    /// The copy round that pays most, of a copy `len` tokens long that agreed `suffix` tokens back.
    pub fn copied(c: *const Controller, suffix: u32, len: u32) Choice {
        return copyChoice(&c.copy, suffix, len, &c.ladder.copyCosts());
    }

    /// The round to run: a copy (`copy_len` > 0 tokens that agreed `suffix` back), the drafter's block of at most
    /// `cap`, or a plain step, by expected tokens per ms: the copy on a tie with the block (it drafts nothing), a
    /// plain step when neither buys a row.
    pub fn choose(c: *const Controller, cap: u32, suffix: u32, copy_len: u32) Round {
        const d = c.drafted(cap);
        const k = c.copied(suffix, copy_len);
        if (k.rows > 1 and (d.rows == 1 or k.rate() >= d.rate())) return .{ .kind = .copy, .drafts = k.rows - 1 };
        if (d.rows > 1) return .{ .kind = .draft, .drafts = d.rows - 1 };
        return .{ .kind = .plain };
    }
};

const testing = std.testing;

/// A ladder shaped like a native drafter's: a 10 ms plain step, a round's fixed cost over it, then a cost per row.
fn testLadder(fixed: f32, per_row: f32, drafter: f32) Ladder {
    var l: Ladder = .{ .plain = 10, .copies = undefined, .drafts = @splat(std.math.inf(f32)), .block = 7 };
    for (&l.copies, 0..) |*c, k| c.* = 10 + (if (k > 0) fixed else 1) + per_row * @as(f32, @floatFromInt(k));
    for (0..8) |k| l.drafts[k] = l.copies[k] + drafter;
    return l;
}

test "chooseRows: the rows with the most expected tokens per ms, the cheaper within 2%" {
    var costs: Costs = @splat(20.0);
    costs[1] = 10.0;
    costs[2] = 12.0;
    costs[3] = 12.2;
    try testing.expectEqual(@as(u32, 2), chooseRows(&[_]f32{ 0.5, 0.0958 }, &costs).rows);
    try testing.expectEqual(@as(u32, 3), chooseRows(&[_]f32{ 0.5, 0.5 }, &costs).rows);
    try testing.expectEqual(@as(u32, 1), chooseRows(&[_]f32{ 0.1, 0.9 }, &costs).rows);
}

test "Acceptance: the prior, then the request's own rounds, a place of its own once it has them" {
    var a: Acceptance = .{ .prior = 0.7 };
    var p: [7]f32 = undefined;
    a.fill(&p);
    for (p) |x| try testing.expectApproxEqAbs(@as(f32, 0.7), x, 1e-6);
    // The first draft always lands, the second never: the third is never tested.
    for (0..40) |_| a.observe(7, 1);
    a.fill(&p);
    try testing.expect(p[0] > 0.9 and p[1] < 0.15);
    try testing.expectApproxEqAbs(a.overall(), p[2], 1e-6);
    try testing.expect(a.overall() > 0.4 and a.overall() < 0.6);
}

test "Acceptance: plain steps fade a losing stretch back to the prior" {
    var a: Acceptance = .{ .prior = 0.7 };
    for (0..30) |_| a.observe(4, 0);
    var p: [4]f32 = undefined;
    a.fill(&p);
    try testing.expect(p[0] < 0.2);
    var steps: u32 = 0;
    while (p[0] < 0.65) : (steps += 1) {
        a.idle();
        a.fill(&p);
    }
    // Not at once, and not never.
    try testing.expect(steps > 8 and steps < 64);
}

test "Timing: a cell keeps the least of its timed rounds, the round after a change of cell untimed" {
    var t: Timing = .{ .block = 7 };
    var rounds: u32 = 0;
    while (t.next()) |r| : (rounds += 1) {
        const base: f32 = switch (r.kind) {
            .plain => 10,
            .copy => 14 + @as(f32, @floatFromInt(r.drafts)),
            .draft => 17 + @as(f32, @floatFromInt(r.drafts)),
        };
        // The first round of a cell carries the cell before; one timed round of each stalls on a compile. A plain
        // cell's ticks trade time (a short one, then a long one): its mean is the step, and one of the two cells stalls.
        t.took(if (t.rep == 0)
            900
        else if (r.kind == .plain)
            (if (t.cell == 0 and t.rep == 2) 4000 else if (t.rep % 2 == 1) base - 3 else base + 3)
        else if (t.rep == 2) base + 4000 else base + @as(f32, @floatFromInt(t.rep)));
    }
    try testing.expectEqual(t.cells() * (Timing.REPS + 1), rounds);
    const l = t.ladder().?;
    try testing.expectEqual(@as(f32, 10), l.plain);
    try testing.expectEqual(@as(f32, 15), l.copies[0]);
    try testing.expectEqual(@as(f32, 30), l.copies[15]);
    try testing.expectEqual(@as(f32, 25), l.drafts[7]);
    // A draft that verifies nothing: the drafter's 3 ms over a round with no drafts.
    try testing.expectEqual(@as(f32, 18), l.drafts[0]);
}

test "Timing: no ladder until every cell is timed, none from a cell that never was" {
    var t: Timing = .{ .block = 7 };
    try testing.expect(t.ladder() == null);
    while (t.next()) |_| t.took(std.math.nan(f32));
    try testing.expect(t.ladder() == null);
}

test "Controller: plain when no round pays, the drafter's block while it lands, a copy when it pays more" {
    // A round costs 4.5 ms over a plain step and 1.3 ms a row; the drafter adds 3 ms.
    var c: Controller = .{ .ladder = testLadder(4.5, 1.3, 3), .acc = .{ .prior = 0.7 } };
    const first = c.choose(7, 0, 0);
    try testing.expectEqual(Timing.Kind.draft, first.kind);
    try testing.expect(first.drafts >= 3);
    // Drafts that keep missing: a plain step.
    for (0..20) |_| c.acc.observe(first.drafts, 0);
    try testing.expectEqual(Timing.Kind.plain, c.choose(7, 0, 0).kind);
    // A strong copy is verified whatever the drafter does, and whole once copies have been landing.
    const copy = c.choose(7, 40, 15);
    try testing.expectEqual(Timing.Kind.copy, copy.kind);
    try testing.expect(copy.drafts >= 8);
    for (0..20) |_| c.copy.observe(40, 15, 15);
    try testing.expectEqual(@as(u32, 15), c.choose(7, 40, 15).drafts);
    // Plain steps bring drafting back.
    var steps: u32 = 0;
    while (c.choose(7, 0, 0).kind == .plain and steps < 200) : (steps += 1) c.acc.idle();
    try testing.expect(steps > 4 and steps < 200);
}

test "Controller: a row that costs as much as a step is not bought, a wide copy is cut where its rows get dear" {
    // Rows past 8 cost four times the ones before them.
    var l = testLadder(4.5, 1.3, 3);
    for (9..16) |k| l.copies[k] = l.copies[8] + 5.2 * @as(f32, @floatFromInt(k - 8));
    var c: Controller = .{ .ladder = l, .acc = .{ .prior = 0.7 } };
    for (0..20) |_| c.copy.observe(40, 15, 15);
    try testing.expectEqual(@as(u32, 8), c.choose(7, 40, 15).drafts);
    // A slow machine: a verified row costs most of a plain step, so only a sure draft is worth one.
    const slow: Controller = .{ .ladder = testLadder(9, 8, 3), .acc = .{ .prior = 0.7 } };
    try testing.expectEqual(Timing.Kind.plain, slow.choose(7, 0, 0).kind);
}

fn approx(a: f32, b: f32) !void {
    try testing.expectApproxEqAbs(b, a, 0.02);
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

test "Calibration: a head prior reads the drafter's confidence as its acceptance until verifies say otherwise" {
    var head: Calibration = .{ .head = true };
    const fitted: Calibration = .{};
    try testing.expectApproxEqAbs(@as(f32, 0.93), head.probability(0.93), 1e-4);
    try testing.expect(fitted.probability(0.93) < 0.7);
    for (0..60) |_| head.observe(0.93, false);
    try testing.expect(head.probability(0.93) < 0.2);
}
