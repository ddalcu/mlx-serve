//! sdk/blocks/norms/grouped_rms.
//! Owns groupedRmsNorm.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");

// ── Transformer ──

/// K2-Horizon's `K2HorizonRMSNorm`: rms over each of `groups` equal channel
/// groups of the last axis, then the full-width weight. `ones_cache` holds the
/// weight-less rms_norm's ones vector across calls.
pub fn groupedRmsNorm(x: mlx.mlx_array, w: mlx.mlx_array, groups: u32, eps: f32, ones_cache: *?mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(x);
    const width = shape[shape.len - 1];
    const gw: c_int = @divExact(width, @as(c_int, @intCast(groups)));
    if (ones_cache.* == null) {
        var ones = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{gw}, 1, mlx.mlx_array_dtype(x), s));
        ones_cache.* = ones;
    }
    var g_shape: [8]c_int = undefined;
    @memcpy(g_shape[0 .. shape.len - 1], shape[0 .. shape.len - 1]);
    g_shape[shape.len - 1] = @intCast(groups);
    g_shape[shape.len] = gw;
    var grouped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(grouped);
    try mlx.check(mlx.mlx_reshape(&grouped, x, g_shape[0 .. shape.len + 1].ptr, shape.len + 1, s));
    var normed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(normed);
    try mlx.check(mlx.mlx_fast_rms_norm(&normed, grouped, ones_cache.*.?, eps, s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, normed, shape.ptr, shape.len, s));
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&result, flat, w, s));
    return result;
}

// ── tests ──

test "groupedRmsNorm normalizes each channel group on its own rms" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const xs = [_]f32{ 1, 1, 1, 1, 3, 3, 3, 3 };
    const ws = [_]f32{ 2, 2, 2, 2, 1, 1, 1, 1 };
    const x_shape = [_]c_int{ 1, 1, 8 };
    const w_shape = [_]c_int{8};
    const x = mlx.mlx_array_new_data(&xs, &x_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(x);
    const w = mlx.mlx_array_new_data(&ws, &w_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(w);
    var ones_cache: ?mlx.mlx_array = null;
    defer if (ones_cache) |o| {
        _ = mlx.mlx_array_free(o);
    };
    const out = try groupedRmsNorm(x, w, 2, 1e-6, &ones_cache, s);
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    const got = mlx.mlx_array_data_float32(out).?[0..8];
    const want = [_]f32{ 2, 2, 2, 2, 1, 1, 1, 1 };
    for (got, want) |g, e| try std.testing.expectApproxEqAbs(e, g, 1e-4);
}
