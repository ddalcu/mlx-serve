const std = @import("std");
const testing = std.testing;

pub const MAX_ROWS = 8;
pub const MAX_DEPTH = 16;
pub const MIN_SAMPLES = 3;

pub const History = struct {
    trials: [MAX_DEPTH]u16 = @splat(0),
    accepted: [MAX_DEPTH]u16 = @splat(0),
    rounds: u32 = 0,

    pub fn observe(self: *History, drafted: u32, retained_drafts: u32) void {
        const count = @min(drafted, MAX_DEPTH);
        if (count == 0) return;
        const retained = @min(retained_drafts, count);
        for (0..count) |j| {
            if (self.trials[j] >= 64) {
                self.trials[j] /= 2;
                self.accepted[j] /= 2;
            }
            self.trials[j] += 1;
            if (j < retained) self.accepted[j] += 1;
        }
        self.rounds +|= 1;
    }

    pub fn expected(self: *const History, depth: u8) f32 {
        var result: f32 = 1;
        var survival: f32 = 1;
        for (0..@min(depth, MAX_DEPTH)) |j| {
            const probability = (@as(f32, @floatFromInt(self.accepted[j])) + 0.5) / (@as(f32, @floatFromInt(self.trials[j])) + 1);
            survival = @min(survival, probability);
            result += survival;
        }
        return result;
    }
};

pub const Row = struct {
    history: *const History,
    cap: u8 = 4,
    latency_ms: f32 = 100,
};

pub const Price = struct {
    ms: f32,
    variance: f32 = 0,
    samples: u32 = MIN_SAMPLES,
    max_gap_ms: f32 = 0,
    /// Throughput estimates the mean; variance still bounds an individual round.
    mean_upper_ms: ?f32 = null,
    mean_lower_ms: ?f32 = null,
};

pub const Decision = struct {
    widths: [MAX_ROWS]u8 = @splat(0),
    n: usize = 0,
    rate: f32 = 0,
    plain_rate: f32 = 0,
    upper_ms: f32 = 0,
};

fn validPrice(price: Price) bool {
    if (price.mean_upper_ms) |bound| if (!std.math.isFinite(bound) or bound < price.ms) return false;
    if (price.mean_lower_ms) |bound| if (!std.math.isFinite(bound) or bound > price.ms) return false;
    return price.samples >= MIN_SAMPLES and std.math.isFinite(price.ms) and price.ms > 0 and
        std.math.isFinite(price.variance) and price.variance >= 0 and
        std.math.isFinite(price.max_gap_ms) and price.max_gap_ms >= 0;
}

fn consider(rows: []const Row, provider: anytype, widths: []const u8, baseline: f32, decision: *Decision) void {
    var tokens: f32 = 0;
    var speculative = false;
    for (rows, widths) |row, width| {
        if (width > row.cap or width > MAX_DEPTH) return;
        if (width > 0) {
            speculative = true;
            if (row.history.trials[width - 1] < MIN_SAMPLES) return;
        }
        tokens += row.history.expected(width);
    }
    if (!speculative) return;
    const price = provider.price(widths) orelse return;
    if (!validPrice(price)) return;
    const upper = price.ms + 2 * @sqrt(price.variance);
    const delay = @max(upper, price.max_gap_ms);
    for (rows) |row| if (delay > row.latency_ms) return;
    const rate = 1000 * tokens / (price.mean_upper_ms orelse upper);
    if (rate <= baseline * 1.05 or rate <= decision.rate) return;
    @memcpy(decision.widths[0..rows.len], widths);
    decision.rate = rate;
    decision.upper_ms = upper;
}

pub fn choose(rows: []const Row, provider: anytype) Decision {
    if (rows.len == 0 or rows.len > MAX_ROWS) return .{};
    var decision = Decision{ .n = rows.len };
    for (rows) |row| if (!std.math.isFinite(row.latency_ms) or row.latency_ms <= 0) return decision;
    const plain = provider.price(decision.widths[0..rows.len]) orelse return decision;
    if (!validPrice(plain)) return decision;
    const lower = plain.mean_lower_ms orelse (plain.ms - 2 * @sqrt(plain.variance));
    if (lower <= 0) return decision;
    decision.plain_rate = 1000 * @as(f32, @floatFromInt(rows.len)) / plain.ms;
    decision.rate = decision.plain_rate;
    decision.upper_ms = plain.ms + 2 * @sqrt(plain.variance);
    const baseline = 1000 * @as(f32, @floatFromInt(rows.len)) / lower;
    var max_cap: u8 = 0;
    for (rows) |row| max_cap = @max(max_cap, row.cap);
    for ([_]u8{ 1, 2, 4, 8, 16 }) |width| {
        var base: [MAX_ROWS]u8 = @splat(0);
        for (rows, 0..) |row, i| base[i] = @min(width, row.cap);
        consider(rows, provider, base[0..rows.len], baseline, &decision);
        for (rows, 0..) |row, i| {
            for ([_]u8{ base[i] -| 1, @min(base[i] + 1, row.cap), 0 }) |near| {
                var candidate = base;
                candidate[i] = near;
                consider(rows, provider, candidate[0..rows.len], baseline, &decision);
            }
        }
        for ([_]bool{ false, true }) |drop| {
            var scores: [MAX_ROWS]f32 = @splat(0);
            var order: [MAX_ROWS]usize = undefined;
            for (rows, 0..) |row, i| {
                order[i] = i;
                scores[i] = if (drop) 1 - row.history.expected(base[i]) else row.history.expected(@min(width * 2, @min(row.cap, MAX_DEPTH))) - row.history.expected(base[i]);
            }
            std.sort.insertion(usize, order[0..rows.len], scores, struct {
                fn before(values: [MAX_ROWS]f32, a: usize, b: usize) bool {
                    return values[a] > values[b];
                }
            }.before);
            var candidate = base;
            for (order[0 .. rows.len - 1]) |i| {
                candidate[i] = if (drop) 0 else @min(width * 2, @min(rows[i].cap, MAX_DEPTH));
                consider(rows, provider, candidate[0..rows.len], baseline, &decision);
            }
        }
        if (width >= max_cap) break;
    }
    return decision;
}

const Fixture = struct {
    mode: enum { ordinary, noisy, marginal, ragged, zero, unknown } = .ordinary,
    calls: usize = 0,

    fn price(self: *Fixture, widths: []const u8) ?Price {
        self.calls += 1;
        var plain = true;
        for (widths) |width| plain = plain and width == 0;
        if (plain) return .{ .ms = 20, .samples = 100 };
        if (self.mode == .unknown) return .{ .ms = 10, .samples = 2 };
        if (widths.len == 4) {
            if (std.mem.eql(u8, widths, &.{ 0, 0, 2, 2 }) and self.mode == .zero) return .{ .ms = 31 };
            if (std.mem.eql(u8, widths, &.{ 1, 1, 2, 2 })) return .{ .ms = 40 };
            if (std.mem.eql(u8, widths, &.{ 1, 1, 1, 1 })) return .{ .ms = 45 };
            if (std.mem.eql(u8, widths, &.{ 2, 2, 2, 2 })) return .{ .ms = 60 };
            return null;
        }
        if (std.mem.eql(u8, widths, &.{ 1, 1 })) return switch (self.mode) {
            .noisy => .{ .ms = 35, .variance = 25 },
            .marginal => .{ .ms = 38 },
            else => .{ .ms = 21 },
        };
        if (std.mem.eql(u8, widths, &.{ 2, 2 })) return .{ .ms = 70 };
        if (std.mem.eql(u8, widths, &.{ 4, 4 })) return .{ .ms = 100 };
        return null;
    }
};

fn healthy() History {
    var history: History = .{};
    for (0..100) |_| history.observe(4, 4);
    return history;
}

test "L11 acceptance history is per request and width zero emits one token" {
    var a = healthy();
    const b: History = .{};
    try testing.expectEqual(@as(f32, 1), a.expected(0));
    try testing.expectEqual(@as(f32, 3), b.expected(4));
    try testing.expect(a.expected(4) > 4.9);
    for (0..100) |_| a.observe(4, 0);
    try testing.expect(a.expected(4) < 3.1);
    try testing.expectEqual(@as(f32, 3), b.expected(4));
}

test "L11 prices a shared round once and selects a profitable narrower width" {
    const history = healthy();
    const rows = [_]Row{ .{ .history = &history }, .{ .history = &history } };
    var fixture: Fixture = .{};
    const decision = choose(&rows, &fixture);
    try testing.expectEqualSlices(u8, &.{ 1, 1 }, decision.widths[0..2]);
    try testing.expect(decision.rate > decision.plain_rate);
    try testing.expect(fixture.calls <= 256);
}

test "L11 poor acceptance selects the real plain candidate" {
    var history: History = .{};
    for (0..100) |_| history.observe(4, 0);
    const rows = [_]Row{ .{ .history = &history }, .{ .history = &history } };
    var fixture: Fixture = .{};
    const decision = choose(&rows, &fixture);
    try testing.expectEqualSlices(u8, &.{ 0, 0 }, decision.widths[0..2]);
    try testing.expectEqual(@as(f32, 100), decision.plain_rate);
}

test "L11 rejects a mean win inside measured noise or the switching margin" {
    const history = healthy();
    const rows = [_]Row{ .{ .history = &history }, .{ .history = &history } };
    var fixture = Fixture{ .mode = .noisy };
    var decision = choose(&rows, &fixture);
    try testing.expectEqualSlices(u8, &.{ 0, 0 }, decision.widths[0..2]);
    fixture.mode = .marginal;
    decision = choose(&rows, &fixture);
    try testing.expectEqualSlices(u8, &.{ 0, 0 }, decision.widths[0..2]);
    fixture.mode = .ordinary;
    decision = choose(&rows, &fixture);
    try testing.expectEqualSlices(u8, &.{ 1, 1 }, decision.widths[0..2]);
}

test "L11 applies each user's latency constraint to an aggregate winner" {
    const history = healthy();
    var rows = [_]Row{ .{ .history = &history, .latency_ms = 20 }, .{ .history = &history } };
    var fixture: Fixture = .{};
    var decision = choose(&rows, &fixture);
    try testing.expectEqualSlices(u8, &.{ 0, 0 }, decision.widths[0..2]);
    rows[0].latency_ms = 22;
    decision = choose(&rows, &fixture);
    try testing.expectEqualSlices(u8, &.{ 1, 1 }, decision.widths[0..2]);
}

test "mean cost confidence can confirm a win without narrowing the latency envelope" {
    const Provider = struct {
        bound: f32 = 37,
        samples: u32 = 100,
        pub fn price(self: *@This(), widths: []const u8) ?Price {
            if (std.mem.allEqual(u8, widths, 0)) return .{ .ms = 20, .samples = 100 };
            if (std.mem.allEqual(u8, widths, 1)) return .{ .ms = 35, .variance = 25, .samples = self.samples, .mean_upper_ms = self.bound };
            return null;
        }
    };
    const history = healthy();
    var rows = [_]Row{ .{ .history = &history }, .{ .history = &history } };
    var provider: Provider = .{};
    var decision = choose(&rows, &provider);
    try testing.expectEqualSlices(u8, &.{ 1, 1 }, decision.widths[0..2]);
    try testing.expectEqual(@as(f32, 45), decision.upper_ms);
    rows[0].latency_ms = 44;
    decision = choose(&rows, &provider);
    try testing.expectEqualSlices(u8, &.{ 0, 0 }, decision.widths[0..2]);
    rows[0].latency_ms = 100;
    provider.samples = 2;
    decision = choose(&rows, &provider);
    try testing.expectEqualSlices(u8, &.{ 0, 0 }, decision.widths[0..2]);
    provider.samples = 100;
    for ([_]f32{ 34, std.math.nan(f32), std.math.inf(f32) }) |bad| {
        provider.bound = bad;
        decision = choose(&rows, &provider);
        try testing.expectEqualSlices(u8, &.{ 0, 0 }, decision.widths[0..2]);
    }
}

test "mean cost compares against the optimistic ordinary mean rather than a single fast round" {
    const Provider = struct {
        lower: f32 = 19,
        pub fn price(self: *@This(), widths: []const u8) ?Price {
            if (std.mem.allEqual(u8, widths, 0)) return .{ .ms = 20, .variance = 4, .samples = 100, .mean_lower_ms = self.lower };
            if (std.mem.allEqual(u8, widths, 1)) return .{ .ms = 34, .variance = 4, .samples = 100, .mean_upper_ms = 35 };
            return null;
        }
    };
    const history = healthy();
    var rows = [_]Row{ .{ .history = &history }, .{ .history = &history } };
    var provider: Provider = .{};
    var decision = choose(&rows, &provider);
    try testing.expectEqualSlices(u8, &.{ 1, 1 }, decision.widths[0..2]);
    rows[0].latency_ms = 37;
    decision = choose(&rows, &provider);
    try testing.expectEqualSlices(u8, &.{ 0, 0 }, decision.widths[0..2]);
    rows[0].latency_ms = 100;
    for ([_]f32{ -1, 0, 21, std.math.nan(f32), std.math.inf(f32) }) |bad| {
        provider.lower = bad;
        decision = choose(&rows, &provider);
        try testing.expectEqualSlices(u8, &.{ 0, 0 }, decision.widths[0..2]);
    }
}

test "L11 considers bounded width cohorts and zero-width members" {
    var histories: [4]History = @splat(.{});
    var rows: [4]Row = undefined;
    for (&histories, 0..) |*history, i| {
        for (0..100) |_| history.observe(4, if (i < 2) 1 else 4);
        rows[i] = .{ .history = history, .cap = 2 };
    }
    var fixture = Fixture{ .mode = .ragged };
    var decision = choose(&rows, &fixture);
    try testing.expectEqualSlices(u8, &.{ 1, 1, 2, 2 }, decision.widths[0..4]);
    try testing.expect(fixture.calls <= 256);
    fixture = .{ .mode = .zero };
    decision = choose(&rows, &fixture);
    try testing.expectEqualSlices(u8, &.{ 0, 0, 2, 2 }, decision.widths[0..4]);
    try testing.expect(fixture.calls <= 256);
}

test "L11 declines unmeasured rounds and row counts above eight" {
    const history = healthy();
    const rows: [9]Row = @splat(.{ .history = &history });
    var fixture = Fixture{ .mode = .unknown };
    const decision = choose(rows[0..2], &fixture);
    try testing.expectEqualSlices(u8, &.{ 0, 0 }, decision.widths[0..2]);
    fixture.calls = 0;
    _ = choose(&rows, &fixture);
    try testing.expectEqual(@as(usize, 0), fixture.calls);
}

pub const Action = enum { plain, prime, speculate };

pub fn action(width: u8, hidden_stale: bool) Action {
    return if (width == 0) .plain else if (hidden_stale) .prime else .speculate;
}

pub fn probeWidth(rounds: u8, remaining: u32, cap: u8) ?u8 {
    if (remaining < 96 or rounds >= 16) return null;
    const width: u8 = if (rounds < 4) 1 else if (rounds < 12) 2 else 4;
    return if (width <= cap) width else null;
}

pub fn shouldProbe(decision: Decision, rounds: u8) bool {
    if (rounds < 12) return true;
    for (decision.widths[0..decision.n]) |width| if (width > 0) return false;
    return true;
}

pub const Recovery = struct {
    last_spec_tokens: u32 = 0,
    rounds: u8 = 0,
    burst: u8 = 0,

    pub fn width(self: Recovery, completed: u32, remaining: u32, cap: u8, probes: u8) ?u8 {
        if (probes < 12 or remaining < 96 or cap == 0 or self.rounds >= 8) return null;
        if (self.burst == 0 and completed -| self.last_spec_tokens < 48) return null;
        return @min(2, cap);
    }

    pub fn observe(self: *Recovery, completed: u32, recovering: bool) void {
        self.last_spec_tokens = completed;
        if (recovering) self.rounds +|= 1;
        // Four rounds leave three recurring observations after the transition.
        self.burst = if (recovering) (self.burst + 1) % 4 else 0;
    }
};

pub var mean_cost_override: ?bool = null;

pub fn meanCostEnabled() bool {
    return mean_cost_override orelse true;
}

test "L11 zero dispatches plain and stale hidden must prime before speculation" {
    try testing.expectEqual(Action.plain, action(0, false));
    try testing.expectEqual(Action.plain, action(0, true));
    try testing.expectEqual(Action.prime, action(2, true));
    try testing.expectEqual(Action.speculate, action(2, false));
}

test "L11 calibration is bounded and declines near the token budget" {
    try testing.expectEqual(@as(?u8, 1), probeWidth(0, 300, 4));
    try testing.expectEqual(@as(?u8, 1), probeWidth(3, 300, 4));
    try testing.expectEqual(@as(?u8, 2), probeWidth(4, 300, 4));
    try testing.expectEqual(@as(?u8, 2), probeWidth(11, 300, 4));
    try testing.expectEqual(@as(?u8, 4), probeWidth(12, 300, 4));
    try testing.expect(probeWidth(16, 300, 4) == null);
    try testing.expect(probeWidth(0, 95, 4) == null);
    try testing.expect(probeWidth(4, 300, 1) == null);
    try testing.expect(probeWidth(12, 300, 2) == null);
}

test "L11 calibration reaches width two then stops before a losing wider probe" {
    var decision = Decision{ .n = 4 };
    try testing.expect(shouldProbe(decision, 3));
    decision.widths[0] = 2;
    try testing.expect(shouldProbe(decision, 11));
    try testing.expect(!shouldProbe(decision, 12));
    decision.widths[0] = 0;
    try testing.expect(shouldProbe(decision, 12));
}

test "group recovery retries narrow depths after ordinary output and bounds its work" {
    var recovery: Recovery = .{};
    recovery.observe(30, false);
    try testing.expect(recovery.width(77, 200, 4, 12) == null);
    try testing.expectEqual(@as(?u8, 2), recovery.width(78, 200, 4, 12));
    for (0..4) |round| {
        const completed: u32 = 78 + @as(u32, @intCast(round)) * 2;
        try testing.expectEqual(@as(?u8, 2), recovery.width(completed, 200, 4, 12));
        recovery.observe(completed + 2, true);
    }
    try testing.expect(recovery.width(86, 200, 4, 12) == null);
    try testing.expectEqual(@as(?u8, 2), recovery.width(134, 160, 4, 12));
    for (0..4) |round| recovery.observe(136 + @as(u32, @intCast(round)) * 2, true);
    try testing.expect(recovery.width(200, 100, 4, 12) == null);
}

test "group recovery respects calibration caps and remaining output" {
    const recovery: Recovery = .{};
    try testing.expect(recovery.width(100, 200, 4, 11) == null);
    try testing.expect(recovery.width(100, 95, 4, 12) == null);
    try testing.expect(recovery.width(100, 200, 0, 12) == null);
    try testing.expectEqual(@as(?u8, 1), recovery.width(100, 96, 1, 12));
    try testing.expectEqual(@as(?u8, 2), recovery.width(100, 96, 2, 12));
}

test "sustained speculation ends a recovery burst and restarts the idle requirement" {
    var recovery: Recovery = .{};
    recovery.observe(100, true);
    try testing.expectEqual(@as(?u8, 2), recovery.width(101, 200, 4, 12));
    recovery.observe(103, false);
    try testing.expect(recovery.width(104, 200, 4, 12) == null);
    try testing.expectEqual(@as(?u8, 2), recovery.width(151, 200, 4, 12));
    try testing.expectEqual(@as(u8, 1), recovery.rounds);
}

pub const DEFAULT_ENABLED = true;
pub var enabled_override: ?bool = null;
var enabled_cache: ?bool = null;

pub fn enabledFromEnv(raw: ?[*:0]const u8) bool {
    const value = raw orelse return DEFAULT_ENABLED;
    return value[0] != '0';
}

pub fn enabled() bool {
    if (enabled_override) |value| return value;
    if (enabled_cache) |value| return value;
    const value = enabledFromEnv(std.c.getenv("MLX_SERVE_MTP_GROUP_PLANNER"));
    enabled_cache = value;
    return value;
}

test "group planner defaults on and explicit zero disables it" {
    try testing.expect(enabledFromEnv(null));
    try testing.expect(!enabledFromEnv("0"));
    try testing.expect(enabledFromEnv("1"));
    try testing.expect(enabledFromEnv("on"));
}

pub fn latencyMs() f32 {
    const raw = std.c.getenv("MLX_SERVE_MTP_GROUP_LATENCY_MS") orelse return 100;
    const value = std.fmt.parseFloat(f32, std.mem.span(raw)) catch return 100;
    return if (std.math.isFinite(value) and value > 0) value else 100;
}

pub const Execution = struct {
    n: usize = 0,
    prime: bool = false,
    plain: [MAX_ROWS]usize = undefined,
    speculative: [MAX_ROWS]usize = undefined,
    plain_n: usize = 0,
    speculative_n: usize = 0,
    capture: [MAX_ROWS]bool = @splat(false),
};

pub fn execution(widths: []const u8, stale: []const bool) Execution {
    if (widths.len == 0 or widths.len > MAX_ROWS or stale.len != widths.len) return .{};
    var result = Execution{ .n = widths.len };
    for (widths, stale) |width, hidden_stale| {
        if (width > MAX_DEPTH) return .{};
        result.prime = result.prime or action(width, hidden_stale) == .prime;
    }
    for (widths, 0..) |width, row| {
        if (result.prime or width == 0) {
            result.plain[result.plain_n] = row;
            result.plain_n += 1;
            result.capture[row] = result.prime and width > 0;
        } else {
            result.speculative[result.speculative_n] = row;
            result.speculative_n += 1;
        }
    }
    return result;
}

test "L11 execution keeps zero rows in one plain tick and positive rows in one verify group" {
    const plan = execution(&.{ 0, 2, 0, 4 }, &.{ true, false, true, false });
    try testing.expectEqual(@as(usize, 2), plan.plain_n);
    try testing.expectEqual(@as(usize, 2), plan.speculative_n);
    try testing.expectEqualSlices(usize, &.{ 0, 2 }, plan.plain[0..plan.plain_n]);
    try testing.expectEqualSlices(usize, &.{ 1, 3 }, plan.speculative[0..plan.speculative_n]);
    try testing.expect(!plan.prime);
    try testing.expectEqualSlices(bool, &.{ false, false, false, false }, plan.capture[0..4]);
}

test "L11 execution primes fresh hidden before any selected row speculates" {
    const plan = execution(&.{ 0, 2, 4 }, &.{ false, true, false });
    try testing.expect(plan.prime);
    try testing.expectEqual(@as(usize, 0), plan.speculative_n);
    try testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, plan.plain[0..plan.plain_n]);
    try testing.expectEqualSlices(bool, &.{ false, true, true }, plan.capture[0..3]);
    const overflow: [9]u8 = @splat(1);
    const stale: [9]bool = @splat(false);
    try testing.expectEqual(@as(usize, 0), execution(&overflow, &stale).n);
}

pub const Entry = enum { wait_predraft, drain, choose };

pub fn entry(pending_pipeline: bool, pending_draft: bool) Entry {
    return if (pending_draft) .wait_predraft else if (pending_pipeline) .drain else .choose;
}

test "L11 entry drains a returning plain pipeline so new MTP rows can regroup" {
    try testing.expectEqual(Entry.drain, entry(true, false));
    try testing.expectEqual(Entry.wait_predraft, entry(false, true));
    try testing.expectEqual(Entry.wait_predraft, entry(true, true));
    try testing.expectEqual(Entry.choose, entry(false, false));
}
