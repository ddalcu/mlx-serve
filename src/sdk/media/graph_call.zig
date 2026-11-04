//! An sdk Graph block called from a backend's raw-array code: the result escapes to the caller (who frees it), every
//! intermediate dies with the call. How the large media engines share the blocks without moving onto `ops.Graph`.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");

/// `f(&g, args...)` on stream `s`.
pub fn call(s: mlx.mlx_stream, comptime f: anytype, args: anytype) !mlx.mlx_array {
    var g = ops.Graph.init(std.heap.c_allocator, s);
    defer g.deinit();
    return g.escape(try @call(.auto, f, .{&g} ++ args));
}

test "graph_call: a block's result outlives its intermediates" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const x = mlx.mlx_array_new_float(3);
    defer _ = mlx.mlx_array_free(x);
    const y = try call(s, ops.Graph.addScalar, .{ x, 2.0 });
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_array_eval(y));
    try std.testing.expectEqual(@as(f32, 5), mlx.mlx_array_data_float32(y).?[0]);
}
