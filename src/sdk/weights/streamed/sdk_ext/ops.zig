//! Host-side helpers more than one plugin needs (pure functions, no MLX calls).

const std = @import("std");

/// `out` = the elements of `src`, laid out by `shape` / `strides` (in elements), in row-major
/// order, each converted to `D`. A row-major layout (dims of one element aside) is one pass; any
/// other (a view: a slice or a transpose) walks the logical index.
pub fn copyStrided(comptime S: type, comptime D: type, src: [*]const S, shape: []const c_int, strides: []const usize, out: []D) void {
    var row_major = true;
    var n: usize = 1;
    var d = shape.len;
    while (d > 0) {
        d -= 1;
        if (shape[d] != 1 and strides[d] != n) row_major = false;
        n *= @intCast(shape[d]);
    }
    std.debug.assert(n == out.len);
    if (row_major) {
        if (S == D) {
            @memcpy(out, src[0..n]);
        } else {
            for (out, src[0..n]) |*o, v| o.* = @intCast(v);
        }
        return;
    }
    var idx: [8]usize = @splat(0);
    std.debug.assert(shape.len <= idx.len);
    for (out) |*o| {
        var off: usize = 0;
        for (idx[0..shape.len], strides) |i, st| off += i * st;
        o.* = if (S == D) src[off] else @intCast(src[off]);
        var k = shape.len;
        while (k > 0) {
            k -= 1;
            idx[k] += 1;
            if (idx[k] < @as(usize, @intCast(shape[k]))) break;
            idx[k] = 0;
        }
    }
}

const testing = std.testing;

test "sdk ops: a row-major source is copied in one pass, converted when the types differ; unit dims ignore their strides" {
    const src = [_]u16{ 1, 2, 3, 4, 5, 6 };
    var same: [6]u16 = undefined;
    copyStrided(u16, u16, &src, &.{ 2, 3 }, &.{ 3, 1 }, &same);
    try testing.expectEqualSlices(u16, &src, &same);
    var wide: [6]u64 = undefined;
    copyStrided(u16, u64, &src, &.{ 2, 3 }, &.{ 3, 1 }, &wide);
    try testing.expectEqualSlices(u64, &.{ 1, 2, 3, 4, 5, 6 }, &wide);
    // a dim of one element carries any stride (MLX reports 0 or the full extent there)
    var unit: [6]u32 = undefined;
    copyStrided(u16, u32, &src, &.{ 1, 2, 1, 3 }, &.{ 999, 3, 0, 1 }, &unit);
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6 }, &unit);
}

test "sdk ops: a view (a transpose, a strided slice, a broadcast) is walked in logical row-major order" {
    // [[1, 2, 3], [4, 5, 6]] read transposed: shape [3, 2], strides [1, 3]
    const src = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var t: [6]i32 = undefined;
    copyStrided(i32, i32, &src, &.{ 3, 2 }, &.{ 1, 3 }, &t);
    try testing.expectEqualSlices(i32, &.{ 1, 4, 2, 5, 3, 6 }, &t);
    // every other column of a [2, 4] row-major buffer: shape [2, 2], strides [4, 2], converted
    const buf = [_]u8{ 10, 11, 12, 13, 20, 21, 22, 23 };
    var cols: [4]u32 = undefined;
    copyStrided(u8, u32, &buf, &.{ 2, 2 }, &.{ 4, 2 }, &cols);
    try testing.expectEqualSlices(u32, &.{ 10, 12, 20, 22 }, &cols);
    // a row broadcast down a stride-0 axis
    var b: [6]i32 = undefined;
    copyStrided(i32, i32, &src, &.{ 2, 3 }, &.{ 0, 1 }, &b);
    try testing.expectEqualSlices(i32, &.{ 1, 2, 3, 1, 2, 3 }, &b);
    // rank 0: one element
    var one: [1]i32 = undefined;
    copyStrided(i32, i32, &src, &.{}, &.{}, &one);
    try testing.expectEqual(@as(i32, 1), one[0]);
}
