//! The legacy `Transformer`'s methods over `sdk/blocks/attention/masks.zig` (what reads the engine's model state).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const bf16Scalar = @import("../../../mlx/array.zig").bf16Scalar;
const Transformer = @import("../transformer.zig").Transformer;

pub fn createCausalMask(self: *const Transformer, q_len: c_int, kv_len: c_int) !mlx.mlx_array {
    const offset_val = kv_len - q_len;
    const shape = [_]c_int{ q_len, kv_len };
    var ones = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones);
    try mlx.check(mlx.mlx_full(&ones, &shape, 2, self.one, .bfloat16, self.s));
    var upper = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(upper);
    try mlx.check(mlx.mlx_triu(&upper, ones, offset_val + 1, self.s));
    var bool_upper = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bool_upper);
    try mlx.check(mlx.mlx_astype(&bool_upper, upper, .bool_, self.s));
    const zero = bf16Scalar(0.0, self.s);
    defer _ = mlx.mlx_array_free(zero);
    const neg_inf = bf16Scalar(-std.math.inf(f32), self.s);
    defer _ = mlx.mlx_array_free(neg_inf);
    var mask = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_where(&mask, bool_upper, neg_inf, zero, self.s));
    const mask_shape = [_]c_int{ 1, 1, q_len, kv_len };
    var mask_4d = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&mask_4d, mask, &mask_shape, 4, self.s));
    _ = mlx.mlx_array_free(mask);
    return mask_4d;
}

pub fn createSlidingWindowDecodeMask(self: *const Transformer, kv_len: c_int, window: c_int) !mlx.mlx_array {
    var positions = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(positions);
    try mlx.check(mlx.mlx_arange(&positions, 0, @floatFromInt(kv_len), 1, .int32, self.s));
    const window_start = mlx.mlx_array_new_int(kv_len - window);
    defer _ = mlx.mlx_array_free(window_start);
    var too_old = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(too_old);
    try mlx.check(mlx.mlx_less(&too_old, positions, window_start, self.s));
    const zero = bf16Scalar(0.0, self.s);
    defer _ = mlx.mlx_array_free(zero);
    const neg_inf = bf16Scalar(-std.math.inf(f32), self.s);
    defer _ = mlx.mlx_array_free(neg_inf);
    var sw_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sw_mask);
    try mlx.check(mlx.mlx_where(&sw_mask, too_old, neg_inf, zero, self.s));
    const mask_shape = [_]c_int{ 1, 1, 1, kv_len };
    var mask_4d = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&mask_4d, sw_mask, &mask_shape, 4, self.s));
    return mask_4d;
}

pub fn createSlidingWindowMask(self: *const Transformer, q_len: c_int, kv_len: c_int, window: c_int) !mlx.mlx_array {
    const causal = try self.createCausalMask(q_len, kv_len);
    defer _ = mlx.mlx_array_free(causal);
    const offset_val = kv_len - q_len;
    var row_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row_idx);
    try mlx.check(mlx.mlx_arange(&row_idx, @floatFromInt(offset_val), @floatFromInt(offset_val + q_len), 1, .int32, self.s));
    var col_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(col_idx);
    try mlx.check(mlx.mlx_arange(&col_idx, 0, @floatFromInt(kv_len), 1, .int32, self.s));
    const row_shape = [_]c_int{ q_len, 1 };
    const col_shape = [_]c_int{ 1, kv_len };
    var row_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row_r);
    var col_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(col_r);
    try mlx.check(mlx.mlx_reshape(&row_r, row_idx, &row_shape, 2, self.s));
    try mlx.check(mlx.mlx_reshape(&col_r, col_idx, &col_shape, 2, self.s));
    var dist = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dist);
    try mlx.check(mlx.mlx_subtract(&dist, row_r, col_r, self.s));
    const window_arr = mlx.mlx_array_new_int(window);
    defer _ = mlx.mlx_array_free(window_arr);
    var too_far = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(too_far);
    try mlx.check(mlx.mlx_greater_equal(&too_far, dist, window_arr, self.s));
    const neg_inf = bf16Scalar(-std.math.inf(f32), self.s);
    defer _ = mlx.mlx_array_free(neg_inf);
    const zero = bf16Scalar(0.0, self.s);
    defer _ = mlx.mlx_array_free(zero);
    var sw_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sw_mask);
    try mlx.check(mlx.mlx_where(&sw_mask, too_far, neg_inf, zero, self.s));
    const mask_shape = [_]c_int{ 1, 1, q_len, kv_len };
    var sw_4d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sw_4d);
    try mlx.check(mlx.mlx_reshape(&sw_4d, sw_mask, &mask_shape, 4, self.s));
    var combined = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&combined, causal, sw_4d, self.s));
    return combined;
}
