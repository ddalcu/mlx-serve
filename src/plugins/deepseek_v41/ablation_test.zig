//! The directional ablation's spec (`ablation.zig`): the forward's table read slot by slot, the ops that take the
//! direction out of a write against host math, the armed arrays, the batching and resume rules, and the calibration
//! tap's records and file header.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const dsv41 = @import("deepseek_v41.zig");
const ablation = @import("ablation.zig");

const testing = std.testing;
const A = mlx.mlx_array;
const Scope = dsv41.Scope;

/// `data` as an f32 array of `shape`, kept by `sc`.
fn f32s(sc: *Scope, data: []const f32, shape: []const c_int) !A {
    return sc.keep(mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32));
}

/// An armed ablation along the unit vector `r` at strength `s`, its arrays kept by `sc`.
fn along(sc: *Scope, r: []const f32, s: f32) !ablation.Ablation {
    const n: c_int = @intCast(r.len);
    return .{ .r = try f32s(sc, r, &.{n}), .col = try f32s(sc, r, &.{ n, 1 }), .s = try sc.f(s) };
}

/// `y - s (y . r) r` on the host, into `out`.
fn ablatedHost(y: []const f32, r: []const f32, s: f32, out: []f32) void {
    var dot: f32 = 0;
    for (y, r) |a, b| dot += a * b;
    for (out, y, r) |*o, a, b| o.* = a - s * dot * b;
}

fn expectClose(want: []const f32, got: []const f32) !void {
    try testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try testing.expectApproxEqAbs(w, g, 1e-4);
}

test "deepseek_v41 ablation: a forward without the field, a null one, or strength 0 is no ablation; bad tables refuse" {
    if (comptime !ablation.supported) return error.SkipZigTest;
    const one = [_]u32{0};
    var f: abi.Forward = .{ .mode = .decode, .rows = 1, .tokens = .{}, .slots = .of(&one) };
    try testing.expect((try ablation.of(&f)) == null);
    const zero = [_]f32{0};
    var ab: abi.Ablation = .{ .strengths = .of(&zero) };
    f.ablation = &ab;
    try testing.expect((try ablation.of(&f)) == null);
    const half = [_]f32{0.5};
    ab.strengths = .of(&half);
    const got = (try ablation.of(&f)) orelse return error.TestExpectedAblation;
    try testing.expectEqual(@as(f32, 0.5), got.strength);
    const big = [_]f32{1.5};
    ab.strengths = .of(&big);
    try testing.expectError(error.AblationStrength, ablation.of(&f));
    const two = [_]f32{ 1, 1 };
    ab.strengths = .of(&two);
    try testing.expectError(error.AblationSlots, ablation.of(&f));
    // A host whose `Forward` ends before the field: no ablation, whatever the bytes past its end hold.
    ab.strengths = .of(&half);
    f.size = @offsetOf(abi.Forward, "ablation");
    try testing.expect((try ablation.of(&f)) == null);
}

test "deepseek_v41 ablation: a batch reads each slot's own row, strength and key; a slot at strength 0 runs plain" {
    if (comptime !ablation.supported) return error.SkipZigTest;
    const slots = [_]u32{ 3, 4 };
    var f: abi.Forward = .{ .mode = .batched_rows, .rows = 1, .tokens = .{}, .slots = .of(&slots) };
    const strengths = [_]f32{ 0, 0.25 };
    const keys = [_]u64{ 0, 0xab1e };
    var ab: abi.Ablation = .{ .strengths = .of(&strengths), .keys = .of(&keys) };
    f.ablation = &ab;
    try testing.expect((try ablation.ofSlot(&f, 0)) == null);
    const second = (try ablation.ofSlot(&f, 1)) orelse return error.TestExpectedAblation;
    try testing.expectEqual(@as(usize, 1), second.row);
    try testing.expectEqual(@as(f32, 0.25), second.strength);
    try testing.expectEqual(@as(u64, 0xab1e), second.key);
    // A bad strength in any slot refuses the batch, whichever slot asks.
    const bad = [_]f32{ 0, 2 };
    ab.strengths = .of(&bad);
    try testing.expectError(error.AblationStrength, ablation.ofSlot(&f, 0));
}

test "deepseek_v41 ablation: a write loses s times its component along r, in its own dtype" {
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const r = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    const y = [_]f32{ 1, 2, 3, 4 };
    const ab = try along(&sc, &r, 0.5);
    // y . r = 5: every element loses 0.5 * 5 * 0.5.
    var want: [4]f32 = undefined;
    ablatedHost(&y, &r, 0.5, &want);
    try testing.expectApproxEqAbs(@as(f32, -0.25), want[0], 1e-6);
    const got = try dsv41.hostF32(try ablation.ablateWith(&sc, ab, try f32s(&sc, &y, &.{ 1, 4 })));
    defer testing.allocator.free(got);
    try expectClose(&want, got);
    const yb = try sc.astype(try f32s(&sc, &y, &.{ 1, 4 }), .bfloat16);
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(try ablation.ablateWith(&sc, ab, yb)));
}

test "deepseek_v41 ablation: a stream the FFN output was expanded into loses that output's share along r only" {
    // x2 = post y + comb^T x1 over one row, hc = 2 copies of dim 4; the old stream x1's share along r stays.
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const r = [_]f32{ 0.5, -0.5, 0.5, -0.5 };
    const y = [_]f32{ 2, -1, 0.5, 3 };
    const x1 = [_]f32{ 1, 0, 2, -1, 0.5, 1.5, -2, 1 }; // [hc, dim]
    const post = [_]f32{ 0.75, 1.25 };
    const comb = [_]f32{ 0.5, 0.25, -0.5, 1 }; // [j, k]
    var y_ablated: [4]f32 = undefined;
    ablatedHost(&y, &r, 1, &y_ablated);
    var x2: [8]f32 = undefined;
    var want: [8]f32 = undefined;
    for (0..2) |k| for (0..4) |d| {
        var mix: f32 = 0;
        for (0..2) |j| mix += comb[j * 2 + k] * x1[j * 4 + d];
        x2[k * 4 + d] = post[k] * y[d] + mix;
        want[k * 4 + d] = post[k] * y_ablated[d] + mix;
    };
    const ab = try along(&sc, &r, 1);
    const out = try ablation.ablateExpandedWith(&sc, ab, try f32s(&sc, &x1, &.{ 1, 2, 4 }), try f32s(&sc, &x2, &.{ 1, 2, 4 }), try f32s(&sc, &comb, &.{ 1, 2, 2 }));
    const got = try dsv41.hostF32(out);
    defer testing.allocator.free(got);
    try expectClose(&want, got);
}

test "deepseek_v41 ablation: an Engram row keeps its per-copy keys and loses the value's share along r" {
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const r = [_]f32{ 0, 0.6, 0, 0.8 };
    // hc = 2 keys of 4, then the value.
    const kv = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 3, 1, -2, 2 };
    var want = kv;
    ablatedHost(kv[8..12], &r, 1, want[8..12]);
    const ab = try along(&sc, &r, 1);
    const got = try dsv41.hostF32(try ablation.ablateValueWith(&sc, ab, try f32s(&sc, &kv, &.{ 1, 12 }), 8));
    defer testing.allocator.free(got);
    try expectClose(&want, got);
}

test "deepseek_v41 ablation: arming takes the slot's row of the table as r and as a column, and its strength" {
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    // [slots, dim]: slot 1's direction is (0, 0.6, 0, 0.8).
    const table = [_]f32{ 1, 0, 0, 0, 0, 0.6, 0, 0.8 };
    const dirs = try f32s(&sc, &table, &.{ 2, 4 });
    const spec: ablation.Spec = .{ .directions = .{ .ctx = dirs.ctx }, .row = 1, .strength = 0.75, .key = 9 };
    const ab = try ablation.armed(spec, 4, sc.s);
    defer ab.deinit();
    try testing.expectEqual(@as(u64, 9), ab.key);
    const r = try dsv41.hostF32(ab.r);
    defer testing.allocator.free(r);
    try expectClose(table[4..8], r);
    try testing.expectEqualSlices(c_int, &.{ 4, 1 }, mlx.getShape(ab.col));
    const s = try dsv41.hostF32(ab.s);
    defer testing.allocator.free(s);
    try expectClose(&.{0.75}, s);
    // A table that is not `[slots, dim]` f32 refuses.
    try testing.expectError(error.AblationShape, ablation.armed(spec, 8, sc.s));
}

test "deepseek_v41 ablation: states batch together only unablated or under one keyed ablation" {
    const none: A = .{ .ctx = null };
    const k7: ablation.Ablation = .{ .r = none, .col = none, .s = none, .key = 7 };
    const k8: ablation.Ablation = .{ .r = none, .col = none, .s = none, .key = 8 };
    const unkeyed: ablation.Ablation = .{ .r = none, .col = none, .s = none };
    try testing.expect(ablation.same(null, null));
    try testing.expect(ablation.same(k7, k7));
    try testing.expect(!ablation.same(k7, k8));
    try testing.expect(!ablation.same(k7, null));
    try testing.expect(!ablation.same(null, k7));
    try testing.expect(!ablation.same(unkeyed, unkeyed));
}

test "deepseek_v41 ablation: a state takes a forward of its own variant; a fresh or disk-tier state takes any" {
    try testing.expectEqual(@as(u64, 0), ablation.variantOf(null));
    try testing.expectEqual(@as(u64, 7), ablation.variantOf(.{ .directions = .{}, .strength = 1, .key = 7 }));
    try testing.expectEqual(ablation.unkeyed, ablation.variantOf(.{ .directions = .{}, .strength = 1 }));
    // Nothing computed yet.
    try testing.expect(ablation.fits(0, 7, 0));
    // Read back from the disk tier: the host keyed the entry by its variant.
    try testing.expect(ablation.fits(12, null, 7));
    try testing.expect(ablation.fits(12, 7, 7));
    // An unablated request on an ablated state, and the reverse.
    try testing.expect(!ablation.fits(12, 7, 0));
    try testing.expect(!ablation.fits(12, 0, 7));
}

test "deepseek_v41 ablation: the stream tap keeps the hc mean of a pass's last rows across its spans, only while on" {
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    var tap = try ablation.StreamTap.init(testing.allocator, 1, 3);
    defer tap.deinit();
    // A pass of two spans, [2, hc = 2, dim = 2] and [1, 2, 2]: its last three rows, each row's copies averaged.
    const a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]f32{ 9, 10, 11, 12 };
    const spans = [_]A{ try f32s(&sc, &a, &.{ 2, 2, 2 }), try f32s(&sc, &b, &.{ 1, 2, 2 }) };
    try tap.record(sc.s, 0, &spans);
    try testing.expectEqual(@as(usize, 0), tap.sites[0].items.len);
    tap.on = true;
    try tap.record(sc.s, 0, &spans);
    try tap.record(sc.s, 1, spans[1..]);
    try testing.expectEqual(@as(usize, 1), tap.sites[0].items.len);
    try testing.expectEqual(@as(usize, 1), tap.sites[1].items.len);
    const got = try dsv41.hostF32(tap.sites[0].items[0]);
    defer testing.allocator.free(got);
    try expectClose(&.{ 2, 3, 6, 7, 10, 11 }, got);
    try testing.expectEqualSlices(c_int, &.{ 1, 2 }, mlx.getShape(tap.sites[1].items[0]));
    tap.clear();
    try testing.expectEqual(@as(usize, 0), tap.sites[0].items.len);
}

test "deepseek_v41 ablation: the npy header is a 64-byte multiple that names the shape" {
    var buf: [256]u8 = undefined;
    const h = try ablation.npyHeader(&buf, 44, 5, 7168);
    try testing.expectEqual(@as(usize, 0), h.len % 64);
    try testing.expect(std.mem.startsWith(u8, h, "\x93NUMPY\x01\x00"));
    try testing.expect(std.mem.indexOf(u8, h, "'shape': (44, 5, 7168)") != null);
    try testing.expectEqual(@as(u8, '\n'), h[h.len - 1]);
    try testing.expectEqual(@as(u16, @intCast(h.len - 10)), std.mem.readInt(u16, h[8..10], .little));
}
