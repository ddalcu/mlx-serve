//! `batched.zig`'s tests, out of its import graph: a plugin unit that reaches `batched.zig` never compiles them.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const testRandWeightBf16 = @import("../../testing/parity.zig").testRandWeightBf16;
const DenseKVView = @import("kv_view.zig").DenseKVView;
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;

test "batched decode mask at verify width: row j of slot n sees its own causal prefix only" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    var xfm: Transformer = undefined;
    xfm.rht = null;
    xfm.s = mlx.gpuStream();
    xfm.allocator = t.allocator;
    // Two slots, three verify rows each; kv_len is the length AFTER the rows were appended.
    const kv_lens = [_]i32{ 5, 8 };
    const mask = try xfm.buildBatchedDecodeMask(&kv_lens, 8, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(mask);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, mask, .float32, xfm.s));
    try mlx.check(mlx.mlx_array_eval(f));
    const d = mlx.mlx_array_data_float32(f) orelse return error.MlxArrayDataNull;
    const sh = mlx.getShape(f);
    try t.expectEqual(@as(c_int, 2), sh[0]);
    try t.expectEqual(@as(c_int, 3), sh[2]);
    try t.expectEqual(@as(c_int, 8), sh[3]);
    for (0..2) |n| for (0..3) |j| for (0..8) |pos| {
        const visible = @as(i32, @intCast(pos)) < kv_lens[n] - 3 + 1 + @as(i32, @intCast(j));
        const v = d[(n * 3 + j) * 8 + pos];
        if (visible) try t.expectEqual(@as(f32, 0.0), v) else try t.expect(v == -std.math.inf(f32));
    };
}

test "per-slot batched attention equals the stacked arm on skewed slot lengths" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    var xfm: Transformer = undefined;
    xfm.rht = null;
    xfm.s = mlx.gpuStream();
    xfm.allocator = t.allocator;
    var prng = std.Random.DefaultPrng.init(0x51075);
    const rnd = prng.random();
    const lens = [_]c_int{ 37, 1500 };
    var views: [2]DenseKVView = undefined;
    var made: usize = 0;
    defer for (views[0..made]) |dv| {
        _ = mlx.mlx_array_free(dv.k);
        _ = mlx.mlx_array_free(dv.v);
    };
    for (lens, 0..) |len, i| {
        views[i] = .{
            .k = try testRandWeightBf16(rnd, &[_]c_int{ 1, 4, len, 256 }, xfm.s),
            .v = try testRandWeightBf16(rnd, &[_]c_int{ 1, 4, len, 256 }, xfm.s),
            .owned = false,
        };
        made += 1;
    }
    for ([_]c_int{ 1, 4 }) |seq| {
        const q = try testRandWeightBf16(rnd, &[_]c_int{ 2, 24, seq, 256 }, xfm.s);
        defer _ = mlx.mlx_array_free(q);
        const stacked = try xfm.stackedBatchedAttn(q, &views, seq, 0.0625, .{ .ctx = null });
        defer _ = mlx.mlx_array_free(stacked);
        const fused = [_]bool{ false, false };
        const per_slot = try xfm.perSlotBatchedAttn(q, &views, &fused, seq, 0.0625);
        defer _ = mlx.mlx_array_free(per_slot);
        try t.expectEqualSlices(c_int, mlx.getShape(stacked), mlx.getShape(per_slot));
        var diff = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(diff);
        try mlx.check(mlx.mlx_subtract(&diff, stacked, per_slot, xfm.s));
        var mag = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mag);
        try mlx.check(mlx.mlx_abs(&mag, diff, xfm.s));
        var mx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mx);
        try mlx.check(mlx.mlx_max(&mx, mag, false, xfm.s));
        var mx32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mx32);
        try mlx.check(mlx.mlx_astype(&mx32, mx, .float32, xfm.s));
        try mlx.check(mlx.mlx_array_eval(mx32));
        var worst: f32 = 0;
        try mlx.check(mlx.mlx_array_item_float32(&worst, mx32));
        // Values are U(-0.5, 0.5) averages; two kernels agree to bf16 rounding.
        try t.expect(worst < 5e-3);
    }
}

test "batched decode mask is built in the query dtype: an f16 batch attends through it" {
    // A bf16 mask against f16 queries is an MLX error mid-generation (sdpa
    // requires the mask to promote to the output dtype).
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var xfm: Transformer = undefined;
    xfm.rht = null;
    xfm.s = mlx.gpuStream();
    const kv_lens = [_]i32{ 5, 8 };
    const mask = try xfm.buildBatchedDecodeMask(&kv_lens, 8, 3, .float16);
    defer _ = mlx.mlx_array_free(mask);
    try std.testing.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(mask));
    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    var kv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kv);
    try mlx.check(mlx.mlx_ones(&q, &[_]c_int{ 2, 4, 3, 64 }, 4, .float16, xfm.s));
    try mlx.check(mlx.mlx_ones(&kv, &[_]c_int{ 2, 2, 8, 64 }, 4, .float16, xfm.s));
    var out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, kv, kv, 0.125, "array", mask, .{ .ctx = null }, false, xfm.s));
    try mlx.check(mlx.mlx_array_eval(out));
    try std.testing.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(out));
}
