//! Conformance helpers for `kv`'s lanes (the proposal's KV truncate / rewind equivalence): a row-id backend and the
//! lane-equivalence check every lane of a module-owned decode state must pass. CPU only (no device arrays).

const std = @import("std");
const mlx = @import("sdk").mlx;
const kv = @import("kv.zig");
const testing = std.testing;


/// A lane array's shape on `RowOps`: `[1, rows, 3]` (the feature axis elided), the layout the lanes slice on axis 1.
pub const RowShape = struct {
    n: u8 = 0,
    d: [kv.max_dims]c_int = @splat(0),

    pub fn of(dims: []const c_int) RowShape {
        var s: RowShape = .{ .n = @intCast(dims.len) };
        @memcpy(s.d[0..dims.len], dims);
        return s;
    }

    pub fn slice(self: *const RowShape) []const c_int {
        return self.d[0..self.n];
    }

    pub fn dim(self: RowShape, axis: c_int) c_int {
        return self.d[@intCast(if (axis < 0) axis + @as(c_int, self.n) else axis)];
    }
};

/// Arrays of row ids (`[1, n]` rows, the feature axis elided): enough of the
/// op surface for the lanes, so every append / compaction / view / trim is
/// checked against the absolute positions it must hold.
pub const RowOps = struct {
    pub const T = u32;
    gpa: std.mem.Allocator,
    arrays: std.ArrayList(std.ArrayList(i64)) = .empty,
    kept: std.ArrayList(u32) = .empty,
    allocs: u32 = 0,

    pub fn deinit(g: *RowOps) void {
        for (g.arrays.items) |*a| a.deinit(g.gpa);
        g.arrays.deinit(g.gpa);
        g.kept.deinit(g.gpa);
    }

    pub fn new(g: *RowOps, ids: []const i64) !u32 {
        var a: std.ArrayList(i64) = .empty;
        try a.appendSlice(g.gpa, ids);
        try g.arrays.append(g.gpa, a);
        return @intCast(g.arrays.items.len - 1);
    }

    pub fn range(g: *RowOps, lo: i64, hi: i64) !u32 {
        var buf: [4096]i64 = undefined;
        for (0..@intCast(hi - lo)) |i| buf[i] = lo + @as(i64, @intCast(i));
        return g.new(buf[0..@intCast(hi - lo)]);
    }

    pub fn rows(g: *RowOps, x: u32) []const i64 {
        return g.arrays.items[x].items;
    }

    pub fn shapeOf(g: *RowOps, x: u32) RowShape {
        return RowShape.of(&.{ 1, @intCast(g.arrays.items[x].items.len), 3 });
    }
    pub fn dtypeOf(_: *RowOps, _: u32) mlx.mlx_dtype {
        return .float32;
    }
    pub fn keep(g: *RowOps, x: u32) u32 {
        g.kept.append(g.gpa, x) catch unreachable;
        return x;
    }
    pub fn release(g: *RowOps, x: u32) void {
        const i = std.mem.indexOfScalar(u32, g.kept.items, x) orelse unreachable;
        _ = g.kept.swapRemove(i);
    }
    pub fn zeros(g: *RowOps, shape: []const c_int, _: mlx.mlx_dtype) !u32 {
        g.allocs += 1;
        var buf: [4096]i64 = @splat(-1);
        return g.new(buf[0..@intCast(shape[1])]);
    }
    pub fn hostArray(g: *RowOps, bytes: []const u8, _: []const c_int, _: mlx.mlx_dtype) !u32 {
        const s = std.mem.bytesAsSlice(i32, bytes);
        return g.new(&.{s[1]});
    }
    pub fn sliceUpdateDyn(g: *RowOps, buf: u32, upd: u32, starts: u32) !u32 {
        const row: usize = @intCast(g.rows(starts)[0]);
        var out: [4096]i64 = undefined;
        const b = g.rows(buf);
        @memcpy(out[0..b.len], b);
        const u = g.rows(upd);
        if (row + u.len > b.len) return error.SliceUpdateBounds;
        @memcpy(out[row..][0..u.len], u);
        return g.new(out[0..b.len]);
    }
    pub fn slice(g: *RowOps, x: u32, start: []const c_int, stop: []const c_int, _: []const c_int) !u32 {
        const r = g.rows(x);
        if (stop[1] > r.len) return error.SliceBounds;
        var out: [4096]i64 = undefined;
        const lo: usize = @intCast(start[1]);
        const hi: usize = @intCast(stop[1]);
        @memcpy(out[0 .. hi - lo], r[lo..hi]);
        return g.new(out[0 .. hi - lo]);
    }
    pub fn concat(g: *RowOps, xs: []const u32, _: c_int) !u32 {
        var out: [4096]i64 = undefined;
        var n: usize = 0;
        for (xs) |x| {
            const r = g.rows(x);
            @memcpy(out[n..][0..r.len], r);
            n += r.len;
        }
        return g.new(out[0..n]);
    }
};

/// One step of a lane script: append the next `n` positions, then roll back `trim` of them.
pub const LaneStep = struct { n: u32, trim: u32 = 0 };

/// The proposal's KV truncate / rewind equivalence for `sdk.kv`'s lanes: `lane` (a `kv.Lanes(RowOps)` Store or Window)
/// runs `steps` beside the concatenated store. After every append and every trim its view holds exactly the reference's
/// rows it keeps: a store every row, a ring the suffix from its drop offset, which covers the `reach` rows before each
/// append's first row. A rollback a ring cannot honour is refused by name (`RingRollbackTooDeep`), the lane unchanged.
pub fn expectLaneEquivalence(g: *RowOps, lane: anytype, steps: []const LaneStep, reach: u32) !void {
    const K = @TypeOf(lane.*);
    var reference: kv.Lanes(RowOps).Concat = .{};
    defer reference.deinit(g);
    var pos: u32 = 0;
    for (steps) |sp| {
        const x = try g.range(pos, pos + sp.n);
        try lane.append(g, x);
        try reference.append(g, x);
        pos += sp.n;
        try expectLaneHolds(g, lane, &reference, pos);
        const drop = if (@hasDecl(K, "dropOffset")) lane.dropOffset() else 0;
        if (drop > 0) try testing.expect(pos - sp.n >= drop + reach);
        if (sp.trim == 0) continue;
        const to = pos - sp.trim;
        if (!@hasDecl(K, "canTruncateTo") or lane.canTruncateTo(to)) {
            if (@hasDecl(K, "truncateTo")) try lane.truncateTo(g, to) else try lane.truncate(g, to);
            try reference.truncate(g, to);
            pos = to;
        } else if (@hasDecl(K, "truncateTo")) {
            try testing.expectError(error.RingRollbackTooDeep, lane.truncateTo(g, to));
        }
        try expectLaneHolds(g, lane, &reference, pos);
    }
}

fn expectLaneHolds(g: *RowOps, lane: anytype, reference: *const kv.Lanes(RowOps).Concat, pos: u32) !void {
    const drop = if (@hasDecl(@TypeOf(lane.*), "dropOffset")) lane.dropOffset() else 0;
    const got: []const i64 = if (try lane.view(g)) |v| g.rows(v) else &.{};
    const want: []const i64 = if (reference.view()) |v| g.rows(v)[drop..pos] else &.{};
    try testing.expectEqualSlices(i64, want, got);
}

test "sdk testing: every kv lane keeps the concatenated store's rows through appends, rollbacks and a refused deep one" {
    const L = kv.Lanes(RowOps);
    var steps: [36]LaneStep = @splat(.{ .n = 6, .trim = 2 });
    steps[0..6].* = .{ .{ .n = 300 }, .{ .n = 40 }, .{ .n = 1 }, .{ .n = 6, .trim = 4 }, .{ .n = 1 }, .{ .n = 6, .trim = 5 } };
    for ([_]L.Store{ .{ .concat = .{} }, .{ .grow = L.Grow.init(256, null) }, .{ .grow = L.Grow.init(708, 708) } }) |s0| {
        var g: RowOps = .{ .gpa = testing.allocator };
        defer g.deinit();
        var lane = s0;
        defer lane.deinit(&g);
        try expectLaneEquivalence(&g, &lane, &steps, 0);
    }
    var g: RowOps = .{ .gpa = testing.allocator };
    defer g.deinit();
    var ring: L.Window = .{ .ring = L.Ring.init(128, .{ .route = .window_ring }) };
    defer ring.deinit(&g);
    var ring_steps: [steps.len + 1]LaneStep = undefined;
    @memcpy(ring_steps[0..steps.len], &steps);
    ring_steps[steps.len] = .{ .n = 1, .trim = 200 };
    try expectLaneEquivalence(&g, &ring, &ring_steps, 128 - 1);
}

test "sdk kv: the lanes' logical rows (a ring's dropped rows included) and a full rollback of each lane" {
    const L = kv.Lanes(RowOps);
    var g: RowOps = .{ .gpa = testing.allocator };
    defer g.deinit();
    var store: L.Window = .{ .store = .{ .concat = .{} } };
    defer store.deinit(&g);
    try store.append(&g, try g.range(0, 5));
    try testing.expect(store.rows() == 5 and store.dropOffset() == 0 and store.canTruncateTo(0));
    try store.store.truncate(&g, 0);
    try testing.expect(store.rows() == 0 and try store.view(&g) == null);
    var ring: L.Window = .{ .ring = L.Ring.init(4, .{ .route = .window_ring, .max_verify = 1, .slack = 1, .headroom = 2 }) };
    defer ring.deinit(&g);
    var pos: i64 = 0;
    while (pos < 40) : (pos += 4) try ring.append(&g, try g.range(pos, pos + 4));
    const r = ring.ring;
    try testing.expect(r.drop > 0 and ring.rows() == 40 and r.logicalLen() == r.drop + r.len and ring.dropOffset() == r.drop);
    // a rollback to the start is always honoured; it keeps no row
    try testing.expect(ring.canTruncateTo(0) and !ring.canTruncateTo(1));
    try ring.ring.truncateToLength(0);
    try testing.expect(ring.rows() == 0);
}
