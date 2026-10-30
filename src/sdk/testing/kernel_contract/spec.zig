//! The kernel contract's registration: what a kernel computes in, where it is served, what it may touch and what it
//! replaces, and the error bound its outputs must meet against float64. Follows TensorFold's kernel contract (MIT).

const std = @import("std");

/// The accumulator a kernel declares for its sums; none: it only compares or moves values.
pub const Accum = enum { f32, f16, bf16, none };

/// An accumulator's unit roundoff: half its ulp at 1.
pub fn unit(a: Accum) f64 {
    return switch (a) {
        .f32 => 0x1p-24,
        .f16 => 0x1p-11,
        .bf16 => 0x1p-8,
        .none => 0,
    };
}

/// An output element as the device writes it; every output of one kernel shares it. `u32` an index, `i8` a quantized
/// code and `bool` a flag: all exact (their bound is zero). A bool reads 0 or 1; any other byte is the rig's poison
/// (`metal.Rig`), an element never written, and reads as NaN.
pub const Elem = enum { f32, bf16, f16, u32, i8, bool };

pub fn size(e: Elem) usize {
    return switch (e) {
        .f32, .u32 => 4,
        .bf16, .f16 => 2,
        .i8, .bool => 1,
    };
}

/// Element `i` of `bytes` read as `e` (little-endian).
pub fn read(e: Elem, bytes: []const u8, i: usize) f64 {
    return switch (e) {
        .f32 => @as(f32, @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little))),
        .bf16 => fromBf16(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)),
        .f16 => @as(f16, @bitCast(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little))),
        .u32 => @floatFromInt(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little)),
        .i8 => @floatFromInt(@as(i8, @bitCast(bytes[i]))),
        .bool => switch (bytes[i]) {
            0 => 0,
            1 => 1,
            else => std.math.nan(f64),
        },
    };
}

/// fp32 to bf16, round to nearest even (NaN stays NaN).
pub fn toBf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @truncate((u >> 16) | 0x40);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

pub fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// `v` rounded to `e`, as the device stores it (the reference's own rounding points).
pub fn roundTo(e: Elem, v: f64) f64 {
    return switch (e) {
        .f32 => @as(f32, @floatCast(v)),
        .f16 => @as(f16, @floatCast(v)),
        .bf16 => fromBf16(toBf16(@floatCast(v))),
        .u32, .i8 => v,
        .bool => if (v != 0) 1 else 0,
    };
}

/// A value's class: the non-finite gate compares classes element by element, tolerance-free.
pub const Class = enum { finite, nan, pos_inf, neg_inf };

pub fn classOf(v: f64) Class {
    if (std.math.isNan(v)) return .nan;
    if (std.math.isInf(v)) return if (v > 0) .pos_inf else .neg_inf;
    return .finite;
}

/// A call's size: input rows, outputs a row, and the inputs each output's sums reduce over.
pub const Shape = struct { rows: usize, n: usize, k: usize };

/// One dimension's edges: its tile, the step a valid size moves by, and the smallest valid size (size 0: no edges).
pub const Tile = struct { size: usize = 0, step: usize = 1, min: usize = 1 };

/// The five gates, in the order a kernel meets them.
pub const Gate = enum { nonfinite, repeat, shapes, sums, limits };

/// A shipped kernel's reported failure: that gate must keep failing, and says KNOWN until a fix lands.
pub const Known = struct { gate: Gate, finding: []const u8 };

/// How a kernel treats NaN and infinities: as float64 arithmetic on its formula does, or by a rule it states.
pub const Nan = union(enum) { propagate, stated: []const u8 };

/// Gate 4: the least factor by which its inputs must make fp16 and bf16 sums miss (0: not provable; `why` says why).
pub const Sums = struct { f16: f64 = 100, bf16: f64 = 100, why: []const u8 = "", exact: bool = true };

/// A kernel's registration: what it computes in, what it may touch, where it is served and what it replaces.
pub const Spec = struct {
    name: []const u8,
    accum: Accum,
    /// The longest chain of roundings any output's sums pass through at a shape.
    depth: *const fn (Shape) usize,
    /// fp32 roundings after the sums (scales, rsqrt, exp): each adds 2^-24 of the value to the bound.
    ops: f64 = 0,
    /// Roundings to the output's own type before its last one (a bf16 product of bf16 values).
    mids: f64 = 0,
    out: Elem,
    /// Outputs of different types, in output order: each one's element and count (the last takes the rest); empty:
    /// every output is `out`.
    outs: []const Part = &.{},
    /// It may write its inputs (the repeat gate then checks only their guard bytes).
    in_place: bool = false,
    /// A replacement kernel's outputs must equal this kernel's (or op chain's) bits on the same inputs.
    replaces: ?[]const u8 = null,
    nan: Nan = .propagate,
    /// Edges in rows, outputs a row and the reduction length.
    tiles: [3]Tile = .{ .{}, .{}, .{} },
    /// Shapes the engine calls it at: the shapes gate runs each, and the largest bounds the edges.
    served: []const Shape,
    sums: Sums = .{},
    known: ?Known = null,
    /// It runs on the tensor units: the engine calls it only where `naxAvailable()`, so elsewhere it is not gated.
    nax: bool = false,
};

/// One output of a kernel whose outputs differ in type (`Spec.outs`).
pub const Part = struct { elem: Elem, count: *const fn (Shape) usize };

/// Output `i` of `bytes` (every output, in order) at `shape`: its element and value.
pub fn readOut(s: Spec, shape: Shape, bytes: []const u8, i: usize) struct { elem: Elem, v: f64 } {
    var j = i;
    var off: usize = 0;
    for (s.outs, 0..) |p, n| {
        const c = p.count(shape);
        if (j < c or n + 1 == s.outs.len) return .{ .elem = p.elem, .v = read(p.elem, bytes[off..], j) };
        j -= c;
        off += c * size(p.elem);
    }
    return .{ .elem = s.out, .v = read(s.out, bytes, i) };
}

/// Half an output ulp relative to the value: the output's own rounding.
fn outUnit(e: Elem) f64 {
    return switch (e) {
        .f32 => 0x1p-24,
        .bf16 => 0x1p-8,
        .f16 => 0x1p-11,
        .u32, .i8, .bool => 0,
    };
}

/// The smallest step an output can take near zero: a floor no rounding can go under.
fn outFloor(e: Elem) f64 {
    return switch (e) {
        .f32 => 0x1p-149,
        .bf16 => 0x1p-133,
        .f16 => 0x1p-24,
        .u32, .i8, .bool => 0,
    };
}

/// gamma(d) = d u / (1 - d u): the worst relative error of a d-deep chain of roundings at unit roundoff u.
pub fn gamma(d: f64, u: f64) f64 {
    return if (d * u < 0.5) d * u / (1 - d * u) else 1;
}

/// The allowed |got - want| of an output of `e`: the sums' gamma(depth) of their terms' magnitude, the ops, then the
/// output's rounding. An index is exact.
pub fn bound(s: Spec, e: Elem, shape: Shape, want: f64, mag: f64) f64 {
    if (e == .u32 or e == .i8 or e == .bool) return 0;
    const work = gamma(@floatFromInt(s.depth(shape)), unit(s.accum)) * mag + (s.ops * 0x1p-24 + s.mids * outUnit(e)) * @abs(want);
    return work + outUnit(e) * (@abs(want) + work) + outFloor(e);
}

/// The allowed |got - want| when every sum is exact in fp32 in any order (gate 4's inputs): the ops and the rounding.
pub fn boundExact(s: Spec, e: Elem, want: f64) f64 {
    if (e == .u32 or e == .i8 or e == .bool) return 0;
    const work = (s.ops * 0x1p-24 + s.mids * outUnit(e)) * @abs(want);
    return work + outUnit(e) * (@abs(want) + work) + outFloor(e);
}

/// The served shapes, then per tiled dimension of the first: least, tile - step, tile, tile + step, largest served.
pub fn shapes(a: std.mem.Allocator, s: Spec) ![]Shape {
    var out: std.ArrayList(Shape) = .empty;
    errdefer out.deinit(a);
    for (s.served) |x| try add(a, &out, x);
    const base = s.served[0];
    for (s.tiles, 0..) |t, d| {
        if (t.size == 0) continue;
        var top: usize = 0;
        for (s.served) |x| top = @max(top, dim(x, d));
        const want = [_]usize{ t.min, t.size -| t.step, t.size, t.size + t.step, top };
        for (want) |v| {
            if (v < t.min or v > top or v % t.step != 0) continue;
            var x = base;
            set(&x, d, v);
            try add(a, &out, x);
        }
    }
    return out.toOwnedSlice(a);
}

fn dim(x: Shape, d: usize) usize {
    return switch (d) {
        0 => x.rows,
        1 => x.n,
        else => x.k,
    };
}

fn set(x: *Shape, d: usize, v: usize) void {
    switch (d) {
        0 => x.rows = v,
        1 => x.n = v,
        else => x.k = v,
    }
}

fn add(a: std.mem.Allocator, out: *std.ArrayList(Shape), x: Shape) !void {
    for (out.items) |y| if (std.meta.eql(x, y)) return;
    try out.append(a, x);
}

/// What a gate asks the inputs to hold: ordinary values, NaN and infinities at known places, or sums that test fp32.
pub const Fill = enum { random, nonfinite, stress };

/// The order a simulated accumulator adds terms in: one chain, contiguous halves, or 32 lanes then their halves.
pub const Order = enum { sequential, pairwise, strided };

/// `t`'s terms summed in `acc` (each term and each add rounded to it) in `order`.
pub fn sumIn(acc: Accum, order: Order, t: []const f64) f64 {
    return switch (order) {
        .sequential => chain(acc, t, 1, 0),
        .pairwise => halves(acc, t),
        .strided => blk: {
            var lanes: [32]f64 = @splat(0);
            for (&lanes, 0..) |*l, i| l.* = chain(acc, t, 32, i);
            break :blk halves(acc, &lanes);
        },
    };
}

fn roundAcc(acc: Accum, v: f64) f64 {
    return switch (acc) {
        .f32 => roundTo(.f32, v),
        .f16 => roundTo(.f16, v),
        .bf16 => roundTo(.bf16, v),
        .none => v,
    };
}

fn chain(acc: Accum, t: []const f64, stride: usize, first: usize) f64 {
    var s: f64 = 0;
    var i = first;
    while (i < t.len) : (i += stride) s = roundAcc(acc, s + roundAcc(acc, t[i]));
    return s;
}

fn halves(acc: Accum, t: []const f64) f64 {
    if (t.len == 0) return 0;
    if (t.len == 1) return roundAcc(acc, t[0]);
    const mid = t.len / 2;
    return roundAcc(acc, halves(acc, t[0..mid]) + halves(acc, t[mid..]));
}

// ── tests ──

fn flat(_: Shape) usize {
    return 1;
}

fn perRow(x: Shape) usize {
    return x.rows;
}

test "kernel contract: outputs of different types read each at its own element and offset" {
    const s = Spec{ .name = "t", .accum = .none, .depth = flat, .out = .f32, .outs = &.{ .{ .elem = .u32, .count = perRow }, .{ .elem = .bf16, .count = perRow } }, .served = &.{.{ .rows = 2, .n = 1, .k = 1 }} };
    var b: [12]u8 = undefined;
    std.mem.writeInt(u32, b[0..4], 7, .little);
    std.mem.writeInt(u32, b[4..8], 9, .little);
    std.mem.writeInt(u16, b[8..10], toBf16(1.5), .little);
    std.mem.writeInt(u16, b[10..12], toBf16(-2), .little);
    const shape: Shape = .{ .rows = 2, .n = 1, .k = 1 };
    for ([_]f64{ 7, 9, 1.5, -2 }, [_]Elem{ .u32, .u32, .bf16, .bf16 }, 0..) |v, e, i| {
        const got = readOut(s, shape, &b, i);
        try std.testing.expectEqual(e, got.elem);
        try std.testing.expectEqual(v, got.v);
    }
}

test "kernel contract: classes are exact and the bound grows with depth, magnitude and the output's rounding" {
    try std.testing.expectEqual(Class.nan, classOf(std.math.nan(f64)));
    try std.testing.expectEqual(Class.neg_inf, classOf(-std.math.inf(f64)));
    try std.testing.expectEqual(Class.finite, classOf(-0.0));
    const s = Spec{ .name = "t", .accum = .f32, .depth = flat, .out = .bf16, .served = &.{.{ .rows = 1, .n = 1, .k = 1 }} };
    const b = bound(s, .bf16, .{ .rows = 1, .n = 1, .k = 64 }, 1, 1);
    try std.testing.expect(b > 0x1p-8 and b < 0x1p-8 + 0x1p-20);
    try std.testing.expect(bound(s, .f32, .{ .rows = 1, .n = 1, .k = 1 }, 1, 1) < 0x1p-22);
    try std.testing.expectEqual(@as(f64, 0), bound(s, .u32, .{ .rows = 1, .n = 1, .k = 1 }, 3, 1));
    try std.testing.expectEqual(@as(f64, 0x1p-149), boundExact(s, .f32, 0));
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf16(1.00390625)); // halfway rounds to even
    try std.testing.expect(std.math.isNan(fromBf16(toBf16(std.math.nan(f32)))));
}

test "kernel contract: edges: the least, the tile less a step, the tile, the tile plus a step and the largest served" {
    const s = Spec{
        .name = "t",
        .accum = .f32,
        .depth = flat,
        .out = .f32,
        .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 64, .step = 32, .min = 32 } },
        .served = &.{ .{ .rows = 4, .n = 8, .k = 128 }, .{ .rows = 32, .n = 8, .k = 256 } },
    };
    const got = try shapes(std.testing.allocator, s);
    defer std.testing.allocator.free(got);
    var rows: [8]usize = undefined;
    var nr: usize = 0;
    for (got[2..]) |x| if (x.k == 128) {
        rows[nr] = x.rows;
        nr += 1;
    };
    try std.testing.expectEqualSlices(usize, &.{ 1, 15, 16, 17, 32 }, rows[0..nr]);
    var ks: [8]usize = undefined;
    var nk: usize = 0;
    for (got[2..]) |x| if (x.rows == 4) {
        ks[nk] = x.k;
        nk += 1;
    };
    try std.testing.expectEqualSlices(usize, &.{ 32, 64, 96, 256 }, ks[0..nk]); // 128 is served already
}

test "kernel contract: a bool output reads 0 or 1, its poison NaN, an i8 its signed code, and the bounds are zero" {
    try std.testing.expectEqual(@as(f64, 1), read(.bool, &.{ 0, 1, 0xaa }, 1));
    try std.testing.expectEqual(@as(f64, 0), read(.bool, &.{ 0, 1, 0xaa }, 0));
    try std.testing.expect(std.math.isNan(read(.bool, &.{ 0, 1, 0xaa }, 2)));
    try std.testing.expectEqual(@as(f64, -127), read(.i8, &.{ 0, 0x81 }, 1));
    const s = Spec{ .name = "t", .accum = .none, .depth = flat, .out = .bool, .served = &.{.{ .rows = 1, .n = 1, .k = 1 }} };
    try std.testing.expectEqual(@as(f64, 0), bound(s, .bool, .{ .rows = 1, .n = 1, .k = 1 }, 1, 1));
    try std.testing.expectEqual(@as(f64, 0), bound(s, .i8, .{ .rows = 1, .n = 1, .k = 1 }, 1, 1));
}

test "kernel contract: summed in fp16 or bf16, mirrored integer rows miss; in fp32 they are exact in every order" {
    var t: [4096]f64 = undefined;
    var prng = std.Random.DefaultPrng.init(3);
    const rng = prng.random();
    for (0..2048) |i| {
        const v: f64 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
        t[i] = v;
        t[2048 + i] = -v + @as(f64, if (i < 8) 1 else 0);
    }
    for ([_]Order{ .sequential, .pairwise, .strided }) |o| {
        try std.testing.expectEqual(@as(f64, 8), sumIn(.f32, o, &t));
        try std.testing.expect(sumIn(.f16, o, &t) != 8 and sumIn(.bf16, o, &t) != 8);
    }
}
