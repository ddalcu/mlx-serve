//! Measured depth for a qwen4 grouped MTP round, one picker per group size.

const std = @import("std");
const testing = std.testing;

pub const MAX_DEPTH: u8 = 4;
/// Indexed by depth; slot 0 is unused.
pub const ARMS = MAX_DEPTH + 1;
pub const MAX_GROUP: usize = 8;

/// Tokens one slot publishes per round at `depth`, from its acceptance per draft position
/// (conditional on the drafts before it). Positions deeper than the slot drafts now keep
/// their figures from when it last drafted there: a trial or a switch refreshes them.
pub fn expectedTokens(accept: []const f32, depth: u8) f64 {
    var e: f64 = 1;
    var run: f64 = 1;
    for (accept[0..depth]) |a| {
        run *= a;
        e += run;
    }
    return e;
}

/// Runs the depth with the most tokens per ns. Cost comes from each depth's own measured
/// ticks, which barely move with content but not monotonically with depth (verify tiles).
/// Tokens come from the live acceptance (`expect`), fresh only up to the running depth: the
/// picker may move shallower at any tick, deeper only after a `TRIAL`-tick trial published
/// `MARGIN` more tokens per ns. Every depth in bounds is tried once, the most promising deeper
/// one again when the running depth's tokens rose by `EASIER`, a neighbour every `PERIOD`
/// ticks. The running depth gets `TRIAL` clean ticks after a switch or a trial before the
/// next one. The first tick after a switch is never a sample: it pays the previous depth's
/// in-flight work and a new compile.
pub const Picker = struct {
    pub const PERIOD: u32 = 256;
    pub const TRIAL: u32 = 4;
    pub const MARGIN: f64 = 1.04;
    pub const EASIER: f64 = 1.15;
    const BETA: f64 = 0.125;

    ns: [ARMS]f64 = @splat(0),
    priced: u8 = 0,
    tried: u8 = 0,
    home: u8 = 1,
    home_n: u32 = 0,
    depth: u8 = 1,
    fresh: bool = true,
    left: u32 = 0,
    since: u32 = 0,
    up: bool = false,
    trial_ns: f64 = 0,
    trial_tok: f64 = 0,
    trial_n: u32 = 0,
    /// The running depth's tokens when a deeper one was last tried.
    easier: f64 = 0,

    /// `expect[d]`: tokens a tick at depth d would publish, 0 where d is out of bounds.
    pub fn choose(p: *Picker, expect: [ARMS]f64) u8 {
        const prev = p.depth;
        if (p.left > 0) {
            p.left -= 1;
        } else {
            if (p.depth != p.home) p.conclude(expect);
            if (expect[p.home] == 0) {
                p.moveTo(lowest(expect), expect);
            } else if (p.home_n >= TRIAL) p.moveTo(p.shallower(expect), expect);
            p.depth = p.home;
            p.since +|= 1;
            if (p.home_n >= TRIAL) if (p.trialDepth(expect)) |d| {
                p.depth = d;
                p.left = TRIAL - 1;
                p.since = 0;
                p.home_n = 0;
                p.tried |= bit(d);
                p.easier = expect[p.home];
                p.trial_ns = 0;
                p.trial_tok = 0;
                p.trial_n = 0;
            };
        }
        p.fresh = p.depth != prev;
        return p.depth;
    }

    /// The tick `choose` picked published `tokens` in `ns`.
    pub fn observe(p: *Picker, tokens: f64, ns: u64, clean: bool) void {
        const fresh = p.fresh;
        p.fresh = false;
        if (!clean or fresh or ns == 0) return;
        const d: f64 = @floatFromInt(ns);
        if (p.depth != p.home) {
            p.trial_ns += d;
            p.trial_tok += tokens;
            p.trial_n += 1;
            return;
        }
        p.ns[p.home] = if (p.isPriced(p.home)) p.ns[p.home] + BETA * (d - p.ns[p.home]) else d;
        p.priced |= bit(p.home);
        p.home_n +|= 1;
    }

    fn conclude(p: *Picker, expect: [ARMS]f64) void {
        if (p.trial_n < TRIAL / 2) return;
        p.ns[p.depth] = p.trial_ns / @as(f64, @floatFromInt(p.trial_n));
        p.priced |= bit(p.depth);
        if (p.depth > p.home and p.trial_tok / p.trial_ns > p.rate(p.home, expect) * MARGIN) p.moveTo(p.depth, expect);
    }

    fn moveTo(p: *Picker, d: u8, expect: [ARMS]f64) void {
        if (d == p.home) return;
        p.home = d;
        p.home_n = 0;
        p.easier = expect[d];
    }

    fn rate(p: *const Picker, d: u8, expect: [ARMS]f64) f64 {
        return if (p.isPriced(d)) expect[d] / p.ns[d] else 0;
    }

    /// The best of the running depth and the shallower ones, whose positions are all fresh.
    fn shallower(p: *const Picker, expect: [ARMS]f64) u8 {
        var b = p.home;
        var br = p.rate(p.home, expect) * MARGIN;
        for (1..p.home) |i| {
            const d: u8 = @intCast(i);
            const r = p.rate(d, expect);
            if (expect[d] > 0 and r > br) {
                b = d;
                br = r;
            }
        }
        return b;
    }

    fn trialDepth(p: *Picker, expect: [ARMS]f64) ?u8 {
        for (1..ARMS) |i| if (expect[i] > 0 and p.tried & bit(@intCast(i)) == 0) return @intCast(i);
        if (expect[p.home] > p.easier * EASIER) {
            var deeper: ?u8 = null;
            for (p.home + 1..ARMS) |i| {
                const d: u8 = @intCast(i);
                if (expect[d] > 0 and (deeper == null or p.rate(d, expect) > p.rate(deeper.?, expect))) deeper = d;
            }
            if (deeper) |d| return d;
        }
        if (p.since < PERIOD) return null;
        const above: ?u8 = if (p.home < MAX_DEPTH and expect[p.home + 1] > 0) p.home + 1 else null;
        const below: ?u8 = if (p.home > 1 and expect[p.home - 1] > 0) p.home - 1 else null;
        p.up = !p.up;
        return if (p.up) above orelse below else below orelse above;
    }

    fn isPriced(p: *const Picker, d: u8) bool {
        return p.priced & bit(d) != 0;
    }

    fn bit(d: u8) u8 {
        return @as(u8, 1) << @intCast(d);
    }

    fn lowest(expect: [ARMS]f64) u8 {
        for (1..ARMS) |d| if (expect[d] > 0) return @intCast(d);
        return 1;
    }
};

/// One model's pickers, by group size.
pub const Pickers = struct {
    by_size: [MAX_GROUP + 1]Picker = @splat(.{}),
    /// The depth the model's current tick runs; interleaved ticks reuse it.
    tick: u8 = 1,
    /// Group size of the last picked tick (0 = none).
    last_size: u8 = 0,
    /// Slots that rode a grouped round, so a tick can tell whether all of it did.
    grouped_rows: u64 = 0,
};

/// `ticks` ticks of four slots with acceptance `accept` at every position, depth d costing
/// `ms[d]`; depths outside lo..hi refused.
fn drive(p: *Picker, ms: [ARMS]f64, accept: f32, lo: u8, hi: u8, ticks: u32) !void {
    const a: [MAX_DEPTH]f32 = @splat(accept);
    for (0..ticks) |_| {
        var expect: [ARMS]f64 = @splat(0);
        for (lo..hi + 1) |d| expect[d] = 4 * expectedTokens(&a, @intCast(d));
        const d = p.choose(expect);
        try testing.expect(d >= lo and d <= hi);
        p.observe(expect[d], @intFromFloat(ms[d] * std.time.ns_per_ms), true);
    }
}

const M4_N4 = [ARMS]f64{ 0, 60, 84, 101, 120 };

test "depth picker: stays shallow where a deeper round costs more than it drafts" {
    var p: Picker = .{};
    try drive(&p, M4_N4, 0.85, 1, 4, 100);
    try testing.expectEqual(@as(u8, 1), p.home);
    try drive(&p, M4_N4, 0.58, 1, 4, 100);
    try testing.expectEqual(@as(u8, 1), p.home);
}

test "depth picker: climbs while acceptance pays for depth, and comes back down within a dwell" {
    var p: Picker = .{};
    try drive(&p, M4_N4, 0.97, 1, 4, 100);
    try testing.expect(p.home >= 3);
    try drive(&p, M4_N4, 0.58, 1, 4, Picker.TRIAL + 1);
    try testing.expectEqual(@as(u8, 1), p.home);
}

test "depth picker: finds a cheap depth past an expensive one" {
    // A verify tile that fits depth 3's rows makes it cheaper per row than depth 2.
    var p: Picker = .{};
    try drive(&p, .{ 0, 62, 89, 95, 125 }, 0.85, 1, 4, 100);
    try testing.expectEqual(@as(u8, 3), p.home);
}

test "depth picker: depths stay inside the bounds" {
    var p: Picker = .{};
    try drive(&p, M4_N4, 0.97, 2, 3, 100);
    try testing.expectEqual(@as(u8, 3), p.home);
}

test "depth picker: dirty ticks price nothing" {
    var p: Picker = .{};
    const a: [MAX_DEPTH]f32 = @splat(0.97);
    for (0..100) |_| {
        var expect: [ARMS]f64 = @splat(0);
        for (1..ARMS) |d| expect[d] = 4 * expectedTokens(&a, @intCast(d));
        const d = p.choose(expect);
        p.observe(expect[d], @intFromFloat(M4_N4[d] * std.time.ns_per_ms), d == 1);
    }
    try testing.expectEqual(@as(u8, 1), p.home);
}

test "depth picker: re-prices a deeper depth when the content gets easier" {
    // Two slots; a position's acceptance is only refreshed while the group drafts it.
    const ms = [ARMS]f64{ 0, 38, 48, 58, 70 };
    var p: Picker = .{};
    var seen: [MAX_DEPTH]f32 = @splat(0.5);
    for (0..250) |t| {
        const truth: f32 = if (t < 200) 0.58 else 0.85;
        var expect: [ARMS]f64 = @splat(0);
        for (1..ARMS) |d| expect[d] = 2 * expectedTokens(&seen, @intCast(d));
        const d = p.choose(expect);
        for (seen[0..d]) |*a| a.* += 0.15 * (truth - a.*);
        const real: [MAX_DEPTH]f32 = @splat(truth);
        p.observe(2 * expectedTokens(&real, d), @intFromFloat(ms[d] * std.time.ns_per_ms), true);
        if (t == 199) try testing.expectEqual(@as(u8, 1), p.home);
    }
    try testing.expect(p.home >= 2);
}

test "depth picker: a trial that never gets a clean tick hands back to the running depth" {
    var p: Picker = .{};
    const a: [MAX_DEPTH]f32 = @splat(0.85);
    var home_ticks: u32 = 0;
    for (0..200) |_| {
        var expect: [ARMS]f64 = @splat(0);
        for (1..ARMS) |d| expect[d] = 4 * expectedTokens(&a, @intCast(d));
        const d = p.choose(expect);
        if (d == 1) home_ticks += 1;
        p.observe(expect[d], @intFromFloat(M4_N4[d] * std.time.ns_per_ms), d == 1);
    }
    try testing.expect(home_ticks >= 180);
}
