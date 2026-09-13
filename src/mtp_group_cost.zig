const std = @import("std");
const testing = std.testing;

pub const MAX_WIDTH: u32 = 16;
pub const MAX_GROUP_ROWS: u32 = 8;
const N_BUCKETS = 9;
pub const MIN_SAMPLES: u32 = 3;
pub const BETA: f32 = 0.10;
pub const RESEED_GAP: u32 = 64;
pub const RESEED_WEIGHT: f32 = 0.5;
pub const Verdict = enum { folded, reseeded, bad_sample, out_of_range, wall_mismatch };
pub const GroupWall = enum(u8) { unset, round, verify };

pub const Cell = struct {
    ms: f32 = 0,
    tok: f32 = 0,
    n: u32 = 0,
    /// `Table.seq` at the last fold — the reseed clock.
    last_seen: u32 = 0,
    wall: GroupWall = .unset,

    pub fn msPerTok(self: Cell) ?f32 {
        if (self.n == 0 or self.tok <= 0) return null;
        return self.ms / self.tok;
    }
};

pub const GroupShape = struct {
    n: u8 = 0,
    kv_format: u64 = 0,
    head_format: u64 = 0,
    padding: u16 = 0,
    attention: u8 = 0,
    moe: u8 = 0,
    head_calls: u16 = 0,
    rows: [MAX_GROUP_ROWS]u64 = @splat(0),

    pub fn samplingMode(rerank: bool, confidence: bool, proposal: bool, stochastic: bool) u8 {
        return @as(u8, @intFromBool(rerank)) | (@as(u8, @intFromBool(confidence)) << 1) |
            (@as(u8, @intFromBool(proposal)) << 2) | (@as(u8, @intFromBool(stochastic)) << 3);
    }

    pub fn row(verify: u8, head: u8, base: u8, kv: u8, head_kv: u8, history: u8, mode: u8) u64 {
        var value: u64 = 0;
        for ([_]u8{ mode, history, head_kv, kv, base, head, verify }) |byte| value = (value << 8) | byte;
        return value;
    }

    pub fn part(code: u64, shift: u6) u8 {
        return @truncate(code >> shift);
    }

    pub fn valid(self: GroupShape) bool {
        if (self.n < 1 or self.n > MAX_GROUP_ROWS or self.padding > MAX_GROUP_ROWS * (MAX_WIDTH + 1)) return false;
        for (self.rows[0..self.n]) |code| {
            if (code >> 56 != 0 or part(code, 0) > MAX_WIDTH or part(code, 8) > MAX_WIDTH or part(code, 16) > part(code, 8)) return false;
            if (part(code, 24) >= N_BUCKETS or part(code, 32) >= N_BUCKETS or part(code, 40) > MAX_WIDTH + 2 or part(code, 48) > 15) return false;
        }
        return true;
    }

    pub fn canonical(self: GroupShape) GroupShape {
        var out = self;
        std.sort.pdq(u64, out.rows[0..out.n], {}, std.sort.asc(u64));
        @memset(out.rows[out.n..], 0);
        return out;
    }

    pub fn width(self: GroupShape) u32 {
        var result: u32 = 0;
        for (self.rows[0..self.n]) |code| result = @max(result, part(code, 0));
        return result;
    }

    fn same(a: GroupShape, b: GroupShape) bool {
        return std.meta.eql(a, b);
    }
};

pub const ShapeCell = struct {
    shape: GroupShape = .{},
    cell: Cell = .{},
    variance: f32 = 0,
    // A restored estimate starts conservatively as one observation.
    mean_weight_sq: f32 = 1,
    emitted: [MAX_GROUP_ROWS]u32 = @splat(0),
    gap_ms: [MAX_GROUP_ROWS]f32 = @splat(0),
};

pub const Table = struct {
    shapes: [128]ShapeCell = @splat(.{}),
    shape_seq: u32 = 0,

    pub fn observeShape(self: *Table, input: GroupShape, ms: f32, emitted: []const u32, gaps: []const f32, wall: GroupWall) Verdict {
        if (!input.valid() or emitted.len != input.n or gaps.len != input.n or wall == .unset) return .out_of_range;
        if (!std.math.isFinite(ms) or ms <= 0) return .bad_sample;
        var tokens: u32 = 0;
        for (emitted, gaps) |count, gap| {
            if (count == 0 or count > MAX_WIDTH + 1 or !std.math.isFinite(gap) or gap < 0) return .bad_sample;
            tokens += count;
        }
        const key = input.canonical();
        var victim: usize = 0;
        var oldest: u32 = std.math.maxInt(u32);
        for (self.shapes, 0..) |entry, i| {
            if (entry.cell.n > 0 and GroupShape.same(entry.shape, key)) {
                victim = i;
                break;
            }
            if (entry.cell.n == 0) {
                victim = i;
                break;
            }
            if (entry.cell.last_seen < oldest) {
                oldest = entry.cell.last_seen;
                victim = i;
            }
        }
        var entry = &self.shapes[victim];
        if (entry.cell.n == 0 or !GroupShape.same(entry.shape, key)) entry.* = .{ .shape = key };
        if (entry.cell.n > 0 and entry.cell.wall != wall) return .wall_mismatch;
        self.shape_seq +%= 1;
        const old = entry.cell;
        const result = foldInto(&entry.cell, ms, @floatFromInt(tokens), self.shape_seq);
        entry.cell.wall = wall;
        const beta: f32 = if (old.n == 0) 1 else if (self.shape_seq -% old.last_seen > RESEED_GAP) RESEED_WEIGHT else if (old.n < MIN_SAMPLES) 1.0 / @as(f32, @floatFromInt(old.n + 1)) else BETA;
        const delta = ms - old.ms;
        entry.variance = if (old.n == 0) 0 else (1 - beta) * (entry.variance + beta * delta * delta);
        entry.mean_weight_sq = if (old.n == 0) 1 else (1 - beta) * (1 - beta) * entry.mean_weight_sq + beta * beta;
        var order: [MAX_GROUP_ROWS]usize = undefined;
        for (0..input.n) |i| order[i] = i;
        std.sort.insertion(usize, order[0..input.n], input, struct {
            fn less(shape: GroupShape, a: usize, b: usize) bool {
                return shape.rows[a] < shape.rows[b];
            }
        }.less);
        for (order[0..input.n], 0..) |row_index, i| {
            entry.emitted[i] = emitted[row_index];
            entry.gap_ms[i] = gaps[row_index];
        }
        return result;
    }

    pub fn shapeCell(self: *const Table, input: GroupShape) ?ShapeCell {
        if (!input.valid()) return null;
        const key = input.canonical();
        for (self.shapes) |entry| if (entry.cell.n > 0 and GroupShape.same(entry.shape, key)) return entry;
        return null;
    }

    fn foldInto(cell: *Cell, ms: f32, tokens: f32, clock: u32) Verdict {
        defer cell.last_seen = clock;
        if (cell.n == 0) {
            cell.ms = ms;
            cell.tok = tokens;
            cell.n = 1;
            return .reseeded;
        }
        // The first MIN_SAMPLES are a running MEAN (an EMA seeded from
        // sample 1 is still sample 1 at n=3); the EMA takes over after.
        const stale = clock -% cell.last_seen > RESEED_GAP;
        const beta: f32 = if (stale) RESEED_WEIGHT else if (cell.n < MIN_SAMPLES) 1.0 / @as(f32, @floatFromInt(cell.n + 1)) else BETA;
        cell.ms += beta * (ms - cell.ms);
        cell.tok += beta * (tokens - cell.tok);
        cell.n += 1;
        return if (stale) .reseeded else .folded;
    }
};

test "group cost geometry separates equal-max-width vectors and execution arms" {
    var table: Table = .{};
    var a = GroupShape{ .n = 3, .kv_format = 8, .head_format = 8 };
    a.rows[0] = GroupShape.row(1, 1, 1, 2, 1, 2, 1);
    a.rows[1] = GroupShape.row(4, 4, 4, 2, 1, 2, 1);
    a.rows[2] = a.rows[1];
    var b = a;
    b.rows[0] = GroupShape.row(2, 2, 2, 2, 1, 2, 1);
    b.rows[1] = GroupShape.row(3, 3, 3, 2, 1, 2, 1);
    _ = table.observeShape(a, 30, &.{ 1, 2, 3 }, &.{ 30, 30, 30 }, .round);
    _ = table.observeShape(b, 60, &.{ 2, 2, 3 }, &.{ 60, 60, 60 }, .round);
    try testing.expectEqual(@as(f32, 30), table.shapeCell(a).?.cell.ms);
    try testing.expectEqual(@as(f32, 60), table.shapeCell(b).?.cell.ms);
    inline for (.{ "kv_format", "head_format", "padding", "attention", "moe", "head_calls" }) |field| {
        var other = a;
        @field(other, field) += 1;
        try testing.expect(table.shapeCell(other) == null);
    }
    var reversed = a;
    std.mem.reverse(u64, reversed.rows[0..reversed.n]);
    try testing.expectEqual(@as(f32, 30), table.shapeCell(reversed).?.cell.ms);
    try testing.expectEqual(Verdict.wall_mismatch, table.observeShape(a, 50, &.{ 1, 2, 3 }, &.{ 30, 30, 30 }, .verify));
}

test "mean cost weights track EMA memory and lose confidence after a reseed" {
    var table: Table = .{};
    var shape = GroupShape{ .n = 1 };
    shape.rows[0] = GroupShape.row(2, 2, 2, 1, 0, 3, 1);
    for (0..3) |_| _ = table.observeShape(shape, 75, &.{3}, &.{75}, .round);
    try testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), table.shapeCell(shape).?.mean_weight_sq, 0.00001);
    for (0..200) |_| _ = table.observeShape(shape, 75, &.{3}, &.{75}, .round);
    const mature = table.shapeCell(shape).?.mean_weight_sq;
    try testing.expectApproxEqAbs(BETA / (2 - BETA), mature, 0.00001);
    table.shape_seq +%= RESEED_GAP + 1;
    _ = table.observeShape(shape, 75, &.{3}, &.{75}, .round);
    try testing.expectApproxEqAbs((1 - RESEED_WEIGHT) * (1 - RESEED_WEIGHT) * mature + RESEED_WEIGHT * RESEED_WEIGHT, table.shapeCell(shape).?.mean_weight_sq, 0.00001);
}
