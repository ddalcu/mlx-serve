//! One picture as the video backends condition on it (keyframes, references): PNG/JPEG bytes cover-cropped to the
//! canvas (`image_pixels.decodeCover`), mapped to [-1, 1] as `x*2 - 1`, shaped [1, 3, 1, H, W] in `dt`, evaluated.

const mlx = @import("../../mlx/ffi.zig");
const std = @import("std");
const image_pixels = @import("image_pixels.zig");

/// Null when the bytes do not decode. Caller frees.
pub fn pm1(a: std.mem.Allocator, bytes: []const u8, h: u32, w: u32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) ?mlx.mlx_array {
    const unit = image_pixels.decodeCover(a, bytes, h, w) orelse return null;
    defer _ = mlx.mlx_array_free(unit);
    return scaled(unit, h, w, dt, s) catch null;
}

fn scaled(unit: mlx.mlx_array, h: u32, w: u32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    const two = mlx.mlx_array_new_float(2.0);
    defer _ = mlx.mlx_array_free(two);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_multiply(&x2, unit, two, s));
    var m1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m1);
    try mlx.check(mlx.mlx_subtract(&m1, x2, one, s));
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_reshape(&r, m1, &[_]c_int{ 1, 3, 1, @intCast(h), @intCast(w) }, 5, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, r, dt, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

test "pm1 crops a first frame of another shape instead of stretching it" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // 100x50 source, white only in its centered square: a 50x50 frame covers it in white.
    const W = 100;
    const H = 50;
    var rgb: [W * H * 3]u8 = undefined;
    for (0..H) |y| for (0..W) |x| {
        const v: u8 = if (x >= 25 and x < 75) 255 else 0;
        @memset(rgb[(y * W + x) * 3 ..][0..3], v);
    };
    const png_bytes = try @import("png.zig").encodeRgb(a, &rgb, W, H);
    defer a.free(png_bytes);
    const arr = pm1(a, png_bytes, 50, 50, .bfloat16, s) orelse return error.DecodeFailed;
    defer _ = mlx.mlx_array_free(arr);
    try std.testing.expectEqualSlices(c_int, &.{ 1, 3, 1, 50, 50 }, mlx.getShape(arr));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    _ = mlx.mlx_array_eval(f);
    const d = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    var mean: f64 = 0;
    for (0..3 * 50 * 50) |i| mean += d[i];
    mean /= 3 * 50 * 50;
    try std.testing.expect(mean > 0.9); // a stretch gives ~0
}
