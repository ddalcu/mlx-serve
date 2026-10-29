//! sdk/blocks/attention/masks.
//! Owns encoderBandMask.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const testing = std.testing;
const bf16Scalar = @import("../../../mlx/array.zig").bf16Scalar;

// ── Mask helpers ──

/// Additive symmetric band mask for bidirectional sliding-window layers:
/// [1, 1, L, L] bf16, 0 where |i-j| < window, -inf outside — the causal
/// sliding semantic (lookback window-1 + self) mirrored in both directions.
/// Broadcasts additively against a [B, 1, 1, L] key-pad mask.
pub fn encoderBandMask(allocator: std.mem.Allocator, seq_len: usize, window: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const buf = try allocator.alloc(f32, seq_len * seq_len);
    defer allocator.free(buf);
    for (0..seq_len) |i| {
        for (0..seq_len) |j| {
            const d = if (i > j) i - j else j - i;
            buf[i * seq_len + j] = if (d < window) 0 else -std.math.inf(f32);
        }
    }
    const shape = [_]c_int{ 1, 1, @intCast(seq_len), @intCast(seq_len) };
    const f32_mask = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_mask);
    var mask = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&mask, f32_mask, .bfloat16, s));
    return mask;
}

// ── tests ──

test "encoderBandMask: symmetric |i-j| < window band, -inf outside" {
    const s = mlx.gpuStream();
    const mask = try encoderBandMask(testing.allocator, 4, 2, s);
    defer _ = mlx.mlx_array_free(mask);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, mask, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const vals = mlx.mlx_array_data_float32(f).?;
    // Row 0: [0, 0, -inf, -inf]; row 2: [-inf, 0, 0, 0].
    try testing.expectEqual(@as(f32, 0), vals[0]);
    try testing.expectEqual(@as(f32, 0), vals[1]);
    try testing.expect(std.math.isNegativeInf(vals[2]));
    try testing.expect(std.math.isNegativeInf(vals[3]));
    try testing.expect(std.math.isNegativeInf(vals[2 * 4 + 0]));
    try testing.expectEqual(@as(f32, 0), vals[2 * 4 + 1]);
    try testing.expectEqual(@as(f32, 0), vals[2 * 4 + 2]);
    try testing.expectEqual(@as(f32, 0), vals[2 * 4 + 3]);
    const shape = mlx.getShape(mask);
    try testing.expectEqual(@as(c_int, 1), shape[0]);
    try testing.expectEqual(@as(c_int, 1), shape[1]);
    try testing.expectEqual(@as(c_int, 4), shape[2]);
    try testing.expectEqual(@as(c_int, 4), shape[3]);
}
