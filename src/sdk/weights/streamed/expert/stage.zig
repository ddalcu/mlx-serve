//! Staging: a wide prompt call's predicted transient loads read while its layer's attention runs, into the transient
//! windows its routes will plan them into.
//!
//! Without it, P1's read-ahead reads the layer's predicted experts into its persistent rows only (about 50 a layer at
//! 16K rows: the rows the usage seed and the prediction leave free); the call's stream groups (its experts past the
//! seed ranks, about 240 a layer, every one a transient miss once the seed holds every persistent row) are read by
//! their own routes, each group's while the groups before it compute, so the call waits on the drive (1.5 s of host
//! waits over a 16K prompt's encoder) while the drive idles under the attention (the prompt timeline,
//! w-pio-20261101-0650).
//!
//! After P1 the stream reads the prediction's stream misses (past the predicted seed ranks, not resident),
//! hottest first, into the windows the call's first stream groups take (`layout`: group g of a call takes window g mod
//! depth, the seed groups first). The stream's ticket ring doubles (2048) so the staged reads' tickets stay unclaimed
//! beside P1's and every ticket the call's routes take before they land. The call adopts them (`adopt`, `Bounds`): each
//! of its first stream groups is the experts it routes of the window's staged group, in staged order, filled up to a
//! route's width from the rest in hot order; the rest follow as before. A route plans its transient misses into the
//! rows of its window that hold their records (`LayerPolicy.planWith`'s `held_rows`) and skips their reads; a route
//! with a transient load lands its window's staged reads first, so each window's reads stay in flight until the group
//! that uses it routes. Exact: a record is the same bytes in any row, and a wave's rows and the combine's fold do not
//! depend on the groups' composition.
//!
//! Read on the M5 Max (st1, w-pio-20261101-0821: a throwaway warm server, then off / on / on / off, each a cold
//! code-ctx16384 card request and a new-text 16K one): cold prefill 1136.0 and 1120.5 tok/s off, 1171.4 and 1178.4 on
//! (+4.1%); new text 1119.1 and 1136.1 off, 1204.2 and 1202.6 on (+6.7%). Over the cold request's encoder the moe
//! stage took 4470 and 4484 ms against 5476 and 5590, the host's waits on bytes 666 and 679 against 1516 and 1650; 4722
//! records staged, 4079 adopted (86%). The texts are the same in every arm (the cold one md5 efeb6f4ecd).

const std = @import("std");

/// A row that holds no staged record.
pub const no_expert: u16 = std.math.maxInt(u16);
/// The stream's windows (`expert_stream.max_wide_depth`) and rows a window (`max_route_ids`).
pub const max_depth = 5;
pub const max_width = 48;

/// The window the call's stream group `j` takes: group g of a call takes window g mod `depth`, its `sg` seed groups
/// first.
pub fn windowOf(sg: usize, depth: usize, j: usize) usize {
    return (sg + j) % depth;
}

/// One call's staging: per window the experts staged in its first rows (`no_expert` past them).
pub const Layout = struct {
    rows: [max_depth][max_width]u16 = @splat(@splat(no_expert)),
    n: [max_depth]u8 = @splat(0),
    records: usize = 0,

    pub fn window(l: *const Layout, w: usize) []const u16 {
        return l.rows[w][0..l.n[w]];
    }
};

/// The predicted stream misses `stream` (hottest first) in groups of `width`, at most `depth` of them: group j into the
/// window the call's stream group j takes (`windowOf`), from its first row.
pub fn layout(stream: []const u16, sg: usize, depth: usize, width: usize, out: *Layout) void {
    std.debug.assert(depth >= 1 and depth <= max_depth and width >= 1 and width <= max_width);
    out.* = .{};
    var next: usize = 0;
    var j: usize = 0;
    while (j < depth and next < stream.len) : (j += 1) {
        const w = windowOf(sg, depth, j);
        const take = @min(width, stream.len - next);
        @memcpy(out.rows[w][0..take], stream[next..][0..take]);
        out.n[w] = @intCast(take);
        out.records += take;
        next += take;
    }
}

/// The call's stream part `d` (its experts past the seed ranks, hottest first, at `base` in the call's order; `pos[e]`
/// each one's index there, negative when the call does not route it) reordered so its first groups are the staged
/// `groups` (in the call's order, `windowOf`): each one's experts that `d` holds, in staged order, then the rest's next
/// experts in hot order up to `width`; then the rest in hot order. `pos` follows. Returns how many groups lead, their
/// cumulative ends (relative to `base`) in `ends`; the leading groups stop at one that finds no expert (the rest is
/// spent: the experts of later staged groups follow in hot order). `scratch`: at least `d.len`.
pub fn adopt(d: []u16, pos: []i32, base: usize, groups: []const []const u16, width: usize, scratch: []u16, ends: []u32) usize {
    std.debug.assert(scratch.len >= d.len and ends.len >= groups.len);
    const lo: i32 = @intCast(base);
    // Routed and staged (the fill passes it by), then taken.
    const staged: i32 = -2;
    const taken: i32 = -3;
    for (groups) |grp| for (grp) |e| if (pos[e] >= lo) {
        pos[e] = staged;
    };
    var n: usize = 0;
    var rest: usize = 0;
    var led: usize = 0;
    for (groups) |grp| {
        const before = n;
        for (grp) |e| if (pos[e] == staged) {
            scratch[n] = e;
            n += 1;
            pos[e] = taken;
        };
        while (n - before < width and rest < d.len) : (rest += 1) {
            const e = d[rest];
            if (pos[e] < lo) continue;
            scratch[n] = e;
            n += 1;
            pos[e] = taken;
        }
        if (n == before) break;
        ends[led] = @intCast(n);
        led += 1;
    }
    for (d) |e| if (pos[e] >= lo or pos[e] == staged) {
        scratch[n] = e;
        n += 1;
    };
    std.debug.assert(n == d.len);
    @memcpy(d, scratch[0..n]);
    for (d, 0..) |e, i| pos[e] = lo + @as(i32, @intCast(i));
    return led;
}

/// A call's groups over its `n` distinct experts: `seed_groups` groups of `width` over the seed ranks [0, sa), then the
/// adopted staged groups (`ends`, cumulative from `sa`), then groups of `width` over the rest.
pub const Bounds = struct {
    n: usize,
    sa: usize,
    seed_groups: usize,
    width: usize,
    ends: []const u32 = &.{},

    fn stagedEnd(b: Bounds) usize {
        return if (b.ends.len == 0) 0 else b.ends[b.ends.len - 1];
    }
    pub fn count(b: Bounds) usize {
        const rest = b.n - b.sa - b.stagedEnd();
        return b.seed_groups + b.ends.len + (rest + b.width - 1) / b.width;
    }
    pub fn start(b: Bounds, gi: usize) usize {
        if (gi < b.seed_groups) return gi * b.width;
        const s = gi - b.seed_groups;
        if (s < b.ends.len) return b.sa + (if (s == 0) 0 else b.ends[s - 1]);
        return b.sa + b.stagedEnd() + (s - b.ends.len) * b.width;
    }
    pub fn end(b: Bounds, gi: usize) usize {
        if (gi < b.seed_groups) return @min(gi * b.width + b.width, b.sa);
        const s = gi - b.seed_groups;
        if (s < b.ends.len) return b.sa + b.ends[s];
        return @min(b.start(gi) + b.width, b.n);
    }
    /// The group of the expert at `fi` (`start`'s inverse).
    pub fn index(b: Bounds, fi: usize) usize {
        if (fi < b.sa) return fi / b.width;
        const r = fi - b.sa;
        for (b.ends, 0..) |e, s| if (r < e) return b.seed_groups + s;
        return b.seed_groups + b.ends.len + (r - b.stagedEnd()) / b.width;
    }
};

const testing = std.testing;

test "stage layout: stream group j in window (sg + j) mod depth from its first row, at most depth groups" {
    var l: Layout = undefined;
    // 2 seed groups, depth 3, 4 rows a window: stream group 0 takes window 2, 1 window 0, 2 window 1; 2 misses left
    // unstaged.
    layout(&.{ 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33 }, 2, 3, 4, &l);
    try testing.expectEqualSlices(u16, &.{ 20, 21, 22, 23 }, l.window(2));
    try testing.expectEqualSlices(u16, &.{ 24, 25, 26, 27 }, l.window(0));
    try testing.expectEqualSlices(u16, &.{ 28, 29, 30, 31 }, l.window(1));
    try testing.expectEqual(@as(usize, 12), l.records);
    // Fewer misses than rows: the last group short, the windows after it empty.
    layout(&.{ 1, 2, 3 }, 0, 2, 2, &l);
    try testing.expectEqualSlices(u16, &.{ 1, 2 }, l.window(0));
    try testing.expectEqualSlices(u16, &.{3}, l.window(1));
    try testing.expectEqual(no_expert, l.rows[1][1]);
    try testing.expectEqual(@as(usize, 2), windowOf(5, 3, 0));
}

test "stage adopt: staged groups lead in staged order, filled from the rest in hot order; the rest after them" {
    // The call's order: seed [7, 8], stream [1, 2, 3, 4, 5, 6] (pos = index), width 3.
    var pos: [16]i32 = @splat(-1);
    const order = [_]u16{ 7, 8, 1, 2, 3, 4, 5, 6 };
    for (order, 0..) |e, i| pos[e] = @intCast(i);
    var d = [_]u16{ 1, 2, 3, 4, 5, 6 };
    var scratch: [6]u16 = undefined;
    var ends: [3]u32 = undefined;
    // Group 0 [5, 9 (unrouted), 2] fills with 1; group 1 [7 (a seed expert), 6] fills with 3 (4 is group 2's), then the
    // rest is spent: group 1 ends short and group 2 [4] still leads, in its own window.
    const n = adopt(&d, &pos, 2, &.{ &.{ 5, 9, 2 }, &.{ 7, 6 }, &.{4} }, 3, &scratch, &ends);
    try testing.expectEqual(@as(usize, 3), n);
    try testing.expectEqualSlices(u32, &.{ 3, 5, 6 }, ends[0..3]);
    try testing.expectEqualSlices(u16, &.{ 5, 2, 1, 6, 3, 4 }, &d);
    for (d, 0..) |e, i| try testing.expectEqual(@as(i32, @intCast(2 + i)), pos[e]);
    try testing.expectEqual(@as(i32, 0), pos[7]);
    try testing.expectEqual(@as(i32, -1), pos[9]);
    // A group with no routed expert once the rest is spent ends the lead; a later group's experts follow in hot order.
    var pos2: [16]i32 = @splat(-1);
    for (order, 0..) |e, i| pos2[e] = @intCast(i);
    var d2 = [_]u16{ 1, 2, 3, 4, 5, 6 };
    const n2 = adopt(&d2, &pos2, 2, &.{ &.{ 1, 2, 3, 4, 5 }, &.{9}, &.{6} }, 5, &scratch, &ends);
    try testing.expectEqual(@as(usize, 1), n2);
    try testing.expectEqual(@as(u32, 5), ends[0]);
    try testing.expectEqualSlices(u16, &.{ 1, 2, 3, 4, 5, 6 }, &d2);
}

test "stage bounds: seed groups, then the staged groups' own sizes, then full groups; index is start's inverse" {
    const b: Bounds = .{ .n = 20, .sa = 5, .seed_groups = 2, .width = 4, .ends = &.{ 3, 5 } };
    try testing.expectEqual(@as(usize, 2 + 2 + 3), b.count());
    const want = [_][2]usize{ .{ 0, 4 }, .{ 4, 5 }, .{ 5, 8 }, .{ 8, 10 }, .{ 10, 14 }, .{ 14, 18 }, .{ 18, 20 } };
    for (want, 0..) |w, gi| {
        try testing.expectEqual(w[0], b.start(gi));
        try testing.expectEqual(w[1], b.end(gi));
        for (w[0]..w[1]) |fi| try testing.expectEqual(gi, b.index(fi));
    }
    // No staged groups: the stock chunks.
    const s: Bounds = .{ .n = 10, .sa = 3, .seed_groups = 1, .width = 4 };
    try testing.expectEqual(@as(usize, 3), s.count());
    try testing.expectEqual(@as(usize, 7), s.start(2));
    try testing.expectEqual(@as(usize, 10), s.end(2));
    try testing.expectEqual(@as(usize, 1), s.index(6));
}
