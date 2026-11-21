//! DeepSeek-V4.1 DSpark round policy, `MLX_SERVE_DSPARK_POLICY2`. Pure data, no MLX.
//!
//! The rows a round verifies maximize `E[tokens] - rate * ms`: `rate` is the request's own decode tokens per ms, `ms` the
//! round's whole wall time by rows (draft included), learned from the round timers. A confidence bin keeps learning when
//! the policy stops verifying it: under greedy the round's next token decides the first draft it did not verify. The
//! drafting backoff judges a probe by the probe and waits at most `Drafting.MAX_PERIOD` plain rounds.
const std = @import("std");
const policy = @import("policy.zig");

pub const MAX_ROWS = policy.MAX_ROWS;
pub const MAX_DRAFTS = policy.MAX_DRAFTS;
pub const Costs = policy.Costs;
pub const Choice = policy.Choice;

/// A DeepSeek-V4.1 round's wall ms by verify rows on an M5 Ultra, in plain steps: `BASE` for a drafted one-row round
/// (the draft forward and the trunk row), `ROW` for each further row; the line through adaptive requests' round ms
/// against their mean verify rows.
const BASE: f32 = 1.379;
const ROW: f32 = 0.225;

fn priorMs(plain_ms: f32, rows: usize) f32 {
    return plain_ms * (BASE + ROW * @as(f32, @floatFromInt(rows -| 1)));
}

/// A drafted round's wall ms by rows: each width's own decayed mean, pulled toward the prior ladder (scaled by what every
/// width measured) by `PRIOR_WEIGHT` pseudo-rounds, so an unvisited width moves with the visited ones.
pub const RoundCost = struct {
    const PRIOR_WEIGHT: f32 = 4;
    const KEEP: f32 = 0.97;

    n: [MAX_ROWS + 1]f32 = @splat(0),
    sum: [MAX_ROWS + 1]f32 = @splat(0),
    scale: f32 = 1,

    pub fn ms(self: *const RoundCost, plain_ms: f32, rows: usize) f32 {
        return (self.sum[rows] + PRIOR_WEIGHT * self.scale * priorMs(plain_ms, rows)) / (self.n[rows] + PRIOR_WEIGHT);
    }

    /// Ms by rows (index = rows, 0 unused); a further row never costs less.
    pub fn costs(self: *const RoundCost, plain_ms: f32) Costs {
        var out: Costs = undefined;
        out[0] = 0;
        for (1..MAX_ROWS + 1) |r| out[r] = @max(if (r > 1) out[r - 1] else 0, self.ms(plain_ms, r));
        return out;
    }

    /// A drafted round of `rows` rows took `took` ms: bounded to a quarter either side of the guess, as `PlainCost`.
    pub fn sample(self: *RoundCost, plain_ms: f32, rows: usize, took: f32) void {
        if (rows < 1 or rows > MAX_ROWS or !std.math.isFinite(took) or took <= 0) return;
        const guess = self.ms(plain_ms, rows);
        const bounded = std.math.clamp(took, guess * 0.8, guess * 1.25);
        self.n[rows] = self.n[rows] * KEEP + 1;
        self.sum[rows] = self.sum[rows] * KEEP + bounded;
        const want = bounded / priorMs(plain_ms, rows);
        self.scale += 0.05 * (std.math.clamp(want, self.scale * 0.8, self.scale * 1.25) - self.scale);
    }
};

/// The request's decode tokens per ms over its drafted rounds and plain ticks, forgetting old rounds. Until it has
/// rounds it answers a prior of `PRIOR_TOKENS` tokens a round at `PRIOR_ROUND` plain steps.
pub const Rate = struct {
    const PRIOR_TOKENS: f32 = 2.6;
    const PRIOR_ROUND: f32 = 1.92;
    const PRIOR_WEIGHT: f32 = 4;
    const KEEP: f32 = 0.97;

    tokens: f32 = 0,
    ms: f32 = 0,

    pub fn perMs(self: *const Rate, plain_ms: f32) f32 {
        return (self.tokens + PRIOR_WEIGHT * PRIOR_TOKENS) / (self.ms + PRIOR_WEIGHT * PRIOR_ROUND * plain_ms);
    }

    pub fn add(self: *Rate, tokens: f32, ms: f32) void {
        if (!std.math.isFinite(ms) or ms <= 0) return;
        self.tokens = self.tokens * KEEP + tokens;
        self.ms = self.ms * KEEP + ms;
    }
};

/// Within this share of a round's tokens of the best value, the fewer rows win.
const ROW_TIE: f32 = 0.02;

/// The rows with the best `tokens - rate * ms`, the long-run rate's optimum. `p[k]` is the probability that draft k is
/// accepted given drafts 0..k-1 were; `costs` the round's ms by rows; `rate` tokens per ms.
pub fn chooseRows(p: []const f32, costs: *const Costs, rate: f32) Choice {
    const n: usize = @min(p.len, MAX_DRAFTS);
    var tokens: [MAX_ROWS + 1]f32 = undefined;
    tokens[1] = 1;
    var alive: f32 = 1;
    for (p[0..n], 0..) |pk, k| {
        alive *= if (pk >= 0) @min(pk, 1) else 0; // a NaN is no acceptance
        tokens[k + 2] = tokens[k + 1] + alive;
    }
    var best: usize = 1;
    for (1..n + 2) |r| if (tokens[r] - rate * costs[r] > tokens[best] - rate * costs[best]) {
        best = r;
    };
    const bar = tokens[best] - rate * costs[best] - ROW_TIE * tokens[best];
    for (1..best + 1) |r| if (tokens[r] - rate * costs[r] >= bar) return .{ .rows = @intCast(r), .tokens = tokens[r], .ms = costs[r] };
    return .{ .rows = @intCast(best), .tokens = tokens[best], .ms = costs[best] };
}

/// Acceptance by the DSpark confidence head, `policy.Calibration`'s bins split into the first draft and the rest (a
/// deep draft's misses must not set the first draft's price), each bin drifting back to the head's own estimate
/// by `RELAX` a round, so a bin nothing teaches still recovers.
pub const Calibration = struct {
    const BINS = 12;
    const PER_UNIT: f32 = 4;
    const PRIOR_WEIGHT: f32 = 3;
    const KEEP: f32 = 0.97;
    const RELAX: f32 = 0.995;

    hit: [2][BINS]f32 = @splat(@splat(0)),
    seen: [2][BINS]f32 = @splat(@splat(0)),

    fn bin(conf: f32) usize {
        const axis = -std.math.log10(@max(1 - std.math.clamp(conf, 0, 1), 1e-3));
        return @min(BINS - 1, @as(usize, @intFromFloat(axis * PER_UNIT)));
    }

    fn class(pos: usize) usize {
        return @intFromBool(pos > 0);
    }

    pub fn probability(self: *const Calibration, pos: usize, conf: f32) f32 {
        const c = class(pos);
        const b = bin(conf);
        return (self.hit[c][b] + PRIOR_WEIGHT * std.math.clamp(conf, 0, 1)) / (self.seen[c][b] + PRIOR_WEIGHT);
    }

    pub fn observe(self: *Calibration, pos: usize, conf: f32, accepted: bool) void {
        const c = class(pos);
        const b = bin(conf);
        self.hit[c][b] = self.hit[c][b] * KEEP + @as(f32, if (accepted) 1 else 0);
        self.seen[c][b] = self.seen[c][b] * KEEP + 1;
    }

    /// One round passed.
    pub fn relax(self: *Calibration) void {
        for (&self.hit, &self.seen) |*h, *s| for (h, s) |*hb, *sb| {
            hb.* *= RELAX;
            sb.* *= RELAX;
        };
    }
};

/// Whether this round drafts at all, as `policy.Drafting` (a round whose tokens are worth less than its time in plain
/// steps loses), except that the first drafted round after a skip stretch is judged on its own, not averaged with the
/// stretch's stale estimate, and a losing stretch waits at most `MAX_PERIOD` plain rounds.
pub const Drafting = struct {
    pub const MAX_PERIOD: u32 = 8;
    const ALPHA: f32 = 0.3;
    const STALL: f32 = 2;

    gain: f32 = 0,
    wall: f32 = 0,
    owed: u32 = 0,
    period: u32 = 1,
    skipped: u64 = 0,
    probes: u64 = 0,
    after_skip: bool = false,
    probing: bool = false,

    pub fn shouldDraft(self: *Drafting) bool {
        if (self.owed > 0) {
            self.owed -= 1;
            self.skipped += 1;
            self.after_skip = true;
            return false;
        }
        self.probing = self.after_skip;
        if (self.after_skip) self.probes += 1;
        self.after_skip = false;
        return true;
    }

    /// A drafted round: `gain` = the ms of plain rounds its tokens would have cost, `took` = what it took.
    pub fn record(self: *Drafting, gain: f32, took: f32) void {
        const wall = @min(took, STALL * if (self.wall > 0) self.wall else gain);
        if (self.wall <= 0 or self.probing) {
            self.gain = gain;
            self.wall = wall;
        } else {
            self.gain += ALPHA * (gain - self.gain);
            self.wall += ALPHA * (wall - self.wall);
        }
        self.probing = false;
        if (self.gain >= self.wall) {
            self.owed = 0;
            self.period = @max(1, self.period / 2);
        } else {
            self.owed = self.period;
            self.period = @min(self.period * 2, MAX_PERIOD);
        }
    }
};

/// The policy's state for one request.
pub const Policy = struct {
    cost: RoundCost = .{},
    rate: Rate = .{},
    calib: Calibration = .{},
    drafting: Drafting = .{},
    /// First unverified drafts the next token decided, and how many of those it took: the switch's engagement line.
    free_seen: u64 = 0,
    free_hit: u64 = 0,
    /// Time is the prior ladder in plain steps, never a stopwatch: the widths depend on the drafts and verdicts alone.
    modeled: bool = false,

    /// The unit time is priced in: a plain step under `modeled`, else ms.
    fn unit(self: *const Policy, plain_ms: f32) f32 {
        return if (self.modeled) 1 else plain_ms;
    }

    /// Drafts to verify of `conf_logits` (the confidence head's logits, one per draft).
    pub fn verifyDrafts(self: *const Policy, conf_logits: []const f32, plain_ms: f32) usize {
        var p: [MAX_DRAFTS]f32 = undefined;
        const n = @min(conf_logits.len, p.len);
        for (conf_logits[0..n], p[0..n], 0..) |c, *o, k| o.* = self.calib.probability(k, sigmoid(c));
        const costs = self.cost.costs(self.unit(plain_ms));
        return chooseRows(p[0..n], &costs, self.rate.perMs(self.unit(plain_ms))).rows - 1;
    }

    /// A drafted round verified `verified` of the drafts behind `conf_logits` and kept `accepted`. `next_hit`, when the
    /// round kept every verified draft and the next token decides the first one it did not verify (greedy), says whether
    /// that draft was the next token.
    pub fn learn(self: *Policy, conf_logits: []const f32, verified: usize, accepted: usize, next_hit: ?bool) void {
        self.calib.relax();
        for (conf_logits[0..@min(verified, conf_logits.len)], 0..) |c, k| {
            if (k > accepted) break;
            self.calib.observe(k, sigmoid(c), k < accepted);
        }
        const hit = next_hit orelse return;
        if (accepted != verified or verified >= conf_logits.len) return;
        self.calib.observe(verified, sigmoid(conf_logits[verified]), hit);
        self.free_seen += 1;
        self.free_hit += @intFromBool(hit);
    }

    /// A drafted round of `rows` rows committed `tokens` in `took` ms.
    pub fn roundDone(self: *Policy, plain_ms: f32, rows: usize, tokens: usize, took: f32) void {
        if (self.modeled) {
            const steps = priorMs(1, @min(@max(rows, 1), MAX_ROWS));
            self.rate.add(@floatFromInt(tokens), steps);
            self.drafting.record(@floatFromInt(tokens), steps);
            return;
        }
        if (!std.math.isFinite(took) or took <= 0) return;
        const guess = self.cost.ms(plain_ms, @min(@max(rows, 1), MAX_ROWS));
        self.cost.sample(plain_ms, rows, took);
        self.rate.add(@floatFromInt(tokens), std.math.clamp(took, 0.5 * guess, 2 * guess));
        self.drafting.record(@as(f32, @floatFromInt(tokens)) * plain_ms, took);
    }

    /// A plain tick took `took` ms.
    pub fn plainDone(self: *Policy, plain_ms: f32, took: f32) void {
        self.rate.add(1, if (self.modeled) 1 else std.math.clamp(took, 0.5 * plain_ms, 2 * plain_ms));
    }
};

pub fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}

const testing = std.testing;

fn logit(p: f32) f32 {
    return @log(p / (1 - p));
}

test "policy2 RoundCost: the prior is DeepSeek-V4.1's ladder, the measured rounds take over, unvisited widths follow" {
    var c = RoundCost{};
    const prior = c.costs(17.7);
    try testing.expectApproxEqAbs(@as(f32, 24.41), prior[1], 0.05);
    try testing.expectApproxEqAbs(@as(f32, 44.32), prior[6], 0.1);
    // Its second row costs 1.16x a drafted one-row round (GLM-5.3's prior: 1.44x the trunk row).
    try testing.expect(prior[2] / prior[1] < 1.2);
    for (2..MAX_ROWS + 1) |r| try testing.expect(prior[r] >= prior[r - 1]);
    // A machine 20% slower at rows 3 and 4 only: those widths reach it, the others move with the shared scale.
    for (0..300) |_| {
        c.sample(17.7, 3, 1.2 * priorMs(17.7, 3));
        c.sample(17.7, 4, 1.2 * priorMs(17.7, 4));
    }
    try testing.expectApproxEqRel(1.2 * priorMs(17.7, 3), c.ms(17.7, 3), 0.02);
    try testing.expect(c.ms(17.7, 6) > 1.1 * priorMs(17.7, 6));
    // A stall and a poisoned sample move nothing much.
    const before = c.ms(17.7, 3);
    c.sample(17.7, 3, 5000);
    c.sample(17.7, 3, std.math.nan(f32));
    try testing.expect(c.ms(17.7, 3) < before * 1.03);
}

test "policy2 chooseRows: a draft is worth its row when its acceptance beats the request's rate times the row's ms" {
    const costs = (RoundCost{}).costs(17.7);
    const rate: f32 = 0.081; // tokens per ms: 81 tok/s
    const step = costs[2] - costs[1]; // ~4.0 ms
    // The bar is rate * step, ~0.32: a 0.45 first draft pays, a 0.30 one does not.
    try testing.expectEqual(@as(u32, 2), chooseRows(&[_]f32{0.45}, &costs, rate).rows);
    try testing.expectEqual(@as(u32, 1), chooseRows(&[_]f32{0.30}, &costs, rate).rows);
    try testing.expect(rate * step > 0.29 and rate * step < 0.36);
    // A sure chain verifies the block; a dead draft cuts the chain there.
    const sure: [5]f32 = @splat(0.95);
    try testing.expectEqual(@as(u32, 6), chooseRows(&sure, &costs, rate).rows);
    try testing.expectEqual(@as(u32, 2), chooseRows(&[_]f32{ 0.9, 0.05, 0.99, 0.99 }, &costs, rate).rows);
    // A slower request (its own rate lower) buys rows a faster one does not.
    try testing.expectEqual(@as(u32, 2), chooseRows(&[_]f32{0.32}, &costs, 0.06).rows);
    // A NaN is no acceptance, and no input leaves it without an answer.
    try testing.expectEqual(@as(u32, 1), chooseRows(&[_]f32{ std.math.nan(f32), 0.9 }, &costs, rate).rows);
    try testing.expectEqual(@as(u32, 1), chooseRows(&.{}, &costs, rate).rows);
}

test "policy2 Rate: the prior until rounds land, then the request's own tokens per ms" {
    var r = Rate{};
    try testing.expectApproxEqRel(@as(f32, 2.6 / (1.92 * 17.7)), r.perMs(17.7), 1e-4);
    for (0..300) |_| r.add(2.0, 40);
    try testing.expectApproxEqRel(@as(f32, 0.05), r.perMs(17.7), 0.06);
    r.add(1, std.math.nan(f32));
    try testing.expect(std.math.isFinite(r.perMs(17.7)));
}

test "policy2 Calibration: the first draft's bins are its own, and a bin no round teaches drifts back to the head" {
    var c = Calibration{};
    try testing.expectApproxEqAbs(@as(f32, 0.6), c.probability(0, 0.6), 1e-4);
    for (0..30) |_| c.observe(3, 0.6, false);
    try testing.expect(c.probability(3, 0.6) < 0.15);
    try testing.expectApproxEqAbs(@as(f32, 0.6), c.probability(0, 0.6), 1e-4);
    for (0..1000) |_| c.relax();
    try testing.expect(c.probability(3, 0.6) > 0.5);
}

/// The GLM-5.3-priced policy (`dspark.dsparkVerifyDrafts`).
fn oldVerifyDrafts(conf_logits: []const f32, calib: *const policy.Calibration) usize {
    var p: [5]f32 = undefined;
    for (conf_logits, &p) |c, *o| o.* = calib.probability(sigmoid(c));
    const costs = policy.priorCosts(17.7);
    return policy.chooseRows(&p, &costs).rows - 1;
}

/// A simulated request: every draft's head says `conf` and lands at `rate`; the first `misses` rounds' first drafts
/// miss (an unlucky start). Returns the drafts verified over rounds 100..300, `old` = the GLM-5.3 policy.
fn starvedRun(old: bool, conf: f32, rate: f32, misses: usize, labels: bool) usize {
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    var calib = policy.Calibration{ .head = true };
    var p2 = Policy{};
    const conf_logits: [5]f32 = @splat(logit(conf));
    var late: usize = 0;
    for (0..300) |round| {
        // Force the first rounds to verify draft 0 and miss it, as an unlucky start does.
        const forced = round < misses;
        const n: usize = if (forced) 1 else if (old) oldVerifyDrafts(&conf_logits, &calib) else p2.verifyDrafts(&conf_logits, 17.7);
        var accepted: usize = 0;
        while (accepted < n and !forced and r.float(f32) < rate) accepted += 1;
        // Under greedy the next token is draft `n` with the draft's own acceptance.
        const next_hit: ?bool = if (labels and accepted == n and n < 5) (!forced and r.float(f32) < rate) else null;
        if (old) {
            for (0..n) |k| {
                if (k > accepted) break;
                calib.observe(sigmoid(conf_logits[k]), k < accepted);
            }
        } else {
            p2.learn(&conf_logits, n, accepted, next_hit);
            p2.roundDone(17.7, n + 1, accepted + 1, (RoundCost{}).ms(17.7, n + 1));
        }
        if (round >= 100) late += n;
    }
    return late;
}

test "policy2: a first-draft bin two unlucky misses froze stays frozen under the old policy and recovers under policy2" {
    // Drafts the head rates 0.55 land 65% of the time: worth their row (bar ~0.33) under either cost model.
    // Old policy: one early miss puts the bin at 0.41, under its ~0.47 bar, and nothing teaches it again.
    try testing.expectEqual(@as(usize, 0), starvedRun(true, 0.55, 0.65, 2, false));
    // Policy2: the next token decides the declined draft every round, and the bin is back within 200 rounds.
    try testing.expect(starvedRun(false, 0.55, 0.65, 2, true) > 200);
    // Without labels (a sampled request), the per-round drift alone brings it back.
    try testing.expect(starvedRun(false, 0.55, 0.65, 2, false) > 100);
    // And a bin that truly does not pay is not bought: 5% drafts stay unverified after their misses.
    try testing.expect(starvedRun(false, 0.55, 0.05, 2, true) < 40);
}

test "policy2: the starved bin recovers within 20 rounds once its drafts land" {
    var p2 = Policy{};
    const conf_logits: [5]f32 = @splat(logit(0.55));
    for (0..3) |_| p2.learn(&conf_logits, 1, 0, null);
    try testing.expectEqual(@as(usize, 0), p2.verifyDrafts(&conf_logits, 17.7));
    var rounds: usize = 0;
    while (p2.verifyDrafts(&conf_logits, 17.7) == 0 and rounds < 100) : (rounds += 1) p2.learn(&conf_logits, 0, 0, rounds % 3 != 2);
    try testing.expect(rounds <= 20);
    try testing.expect(p2.free_seen == rounds and p2.free_hit > 0);
}

test "policy2 Drafting: a cold start's stalls owe a few plain rounds, never a latch; a real loss still skips" {
    // Five compile stalls in a row, then rounds worth 2.8 plain steps for 1.9 plain steps of time.
    var d = Drafting{};
    var skipped_after: u64 = 0;
    var drafted: usize = 0;
    for (0..400) |i| {
        if (!d.shouldDraft()) {
            if (i > 40) skipped_after += 1;
            continue;
        }
        drafted += 1;
        d.record(2.8 * 17.7, if (drafted <= 5) 2000 else 1.9 * 17.7);
    }
    try testing.expect(skipped_after == 0);
    // A stretch where drafting loses still skips, at most MAX_PERIOD plain rounds a probe.
    var lose = Drafting{};
    var run: u32 = 0;
    var longest: u32 = 0;
    for (0..200) |_| {
        if (!lose.shouldDraft()) {
            run += 1;
            longest = @max(longest, run);
            continue;
        }
        run = 0;
        lose.record(1.0 * 17.7, 1.27 * 17.7);
    }
    try testing.expectEqual(Drafting.MAX_PERIOD, longest);
    // The old backoff on the same stalls: its probes average into the stale estimate, each loss doubles the wait to 32.
    var old = policy.Drafting{};
    var old_skipped: u64 = 0;
    drafted = 0;
    for (0..400) |i| {
        if (!old.shouldDraft()) {
            if (i > 40) old_skipped += 1;
            continue;
        }
        drafted += 1;
        old.record(2.8 * 17.7, if (drafted <= 5) 2000 else 1.9 * 17.7);
    }
    try testing.expect(old_skipped > 100);
}

/// The widths a request of 300 rounds verifies, each round's time `took(rows) * jitter` on a `plain_ms` machine.
fn widthTrace(modeled: bool, plain_ms: f32, slow_wide: f32, seed: u64) [300]u8 {
    var prng = std.Random.DefaultPrng.init(42);
    const r = prng.random();
    var jit = std.Random.DefaultPrng.init(seed);
    const j = jit.random();
    var p2 = Policy{ .modeled = modeled };
    var out: [300]u8 = undefined;
    for (&out) |*w| {
        var conf: [5]f32 = undefined;
        for (&conf) |*c| c.* = -1 + 5 * r.float(f32);
        const n = p2.verifyDrafts(&conf, plain_ms);
        var accepted: usize = 0;
        while (accepted < n and r.float(f32) < sigmoid(conf[accepted])) accepted += 1;
        const next_hit: ?bool = if (accepted == n and n < 5) r.float(f32) < sigmoid(conf[n]) else null;
        p2.learn(&conf, n, accepted, next_hit);
        const wide: f32 = if (n + 1 >= 3) slow_wide else 1;
        p2.roundDone(plain_ms, n + 1, accepted + 1, priorMs(plain_ms, n + 1) * wide * (0.9 + 0.2 * j.float(f32)));
        if (r.float(f32) < 0.1) p2.plainDone(plain_ms, plain_ms * (0.9 + 0.2 * j.float(f32)));
        w.* = @intCast(n);
    }
    return out;
}

test "policy2 modeled: the same drafts and verdicts give the same widths whatever the round timers and the plain step" {
    // Jittered timers, another machine's plain step, a slower wide verify: one trace.
    const base = widthTrace(true, 17.7, 1, 1);
    try testing.expectEqualSlices(u8, &base, &widthTrace(true, 17.7, 1, 2));
    try testing.expectEqualSlices(u8, &base, &widthTrace(true, 15.0, 1.25, 3));
    // The timed policy follows its timers: a slower wide verify (3 rows and up) buys fewer wide rounds.
    var timed: usize = 0;
    var slow: usize = 0;
    for (widthTrace(false, 17.7, 1, 1), widthTrace(false, 17.7, 1.25, 1)) |a, b| {
        timed += @intFromBool(a >= 2);
        slow += @intFromBool(b >= 2);
    }
    try testing.expect(slow < timed);
}
