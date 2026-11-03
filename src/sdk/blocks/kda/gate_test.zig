//! `gate.zig`'s tests, out of its import graph: a plugin unit that reaches `gate.zig` never compiles them.

const gate = @import("gate.zig");
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const testing = std.testing;
const qwen4ReadF32 = @import("../../testing/parity.zig").qwen4ReadF32;
const kdaGateChain = gate.kdaGateChain;
const model_mod = @import("../../config/model.zig");

test "kdaGateChain is the bounded-sigmoid gate, not a clamped softplus" {
    // g = exp(bound * sigmoid(exp(A_log) * (a + dt_bias))). Two properties
    // that a "clamp the softplus form at bound" misreading would break:
    //   1. the decay is bounded BELOW by exp(bound) but never reaches it;
    //   2. it is MONOTONICALLY DECREASING in `a` (large positive a ⇒ maximum
    //      forgetting), where the softplus form's magnitude is unbounded.
    const s = mlx.gpuStream();
    const bound: f32 = -5.0;
    const a_vals = [_]f32{ -8.0, 0.0, 4.0 };
    const shape = [_]c_int{3};
    const a_arr = mlx.mlx_array_new_data(&a_vals, &shape, 1, .float32);
    defer _ = mlx.mlx_array_free(a_arr);
    const zeros = [_]f32{ 0.0, 0.0, 0.0 };
    const dt = mlx.mlx_array_new_data(&zeros, &shape, 1, .float32);
    defer _ = mlx.mlx_array_free(dt);
    const a_log_vals = [_]f32{ 0.0, 0.0, 0.0 }; // exp(A_log) = 1
    const a_log = mlx.mlx_array_new_data(&a_log_vals, &shape, 1, .float32);
    defer _ = mlx.mlx_array_free(a_log);

    const g = try kdaGateChain(a_log, a_arr, dt, bound, s);
    defer _ = mlx.mlx_array_free(g);
    var g32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g32);
    try mlx.check(mlx.mlx_astype(&g32, g, .float32, s));
    try mlx.check(mlx.mlx_array_eval(g32));
    const gd = mlx.mlx_array_data_float32(g32) orelse return error.InvalidDtype;

    const floor = @exp(bound); // ~0.0067
    for (0..3) |i| {
        try testing.expect(gd[i] > floor);
        // Open above too: a very negative `a` decays by 0.9983, which a bf16 gate rounds to 1.0.
        try testing.expect(gd[i] < 1.0);
    }
    try testing.expect(gd[0] > gd[1]);
    try testing.expect(gd[1] > gd[2]);
    // a = 0 ⇒ sigmoid = 0.5 ⇒ g = exp(bound/2). bf16 round-trip tolerance.
    try testing.expect(@abs(gd[1] - @exp(bound / 2.0)) < 5e-3);
}

test "glm5_next KDA q/k norm equals fla's l2norm on keys as small as the checkpoint's" {
    // Bar: the prework's rms_norm(x, gdnQkNormEps) / sqrt(dk) is x / sqrt(sum(x^2) + 1e-6) within 1e-4.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const cfg: model_mod.ModelConfig = .{ .model_type = "glm5_next", .linear_key_head_dim = 128 };
    const dk: usize = 128;
    var prng = std.Random.DefaultPrng.init(7);
    var xv: [4 * dk]f32 = undefined;
    for (&xv) |*x| x.* = 0.005 * prng.random().floatNorm(f32);
    const x = mlx.mlx_array_new_data(&xv, &[_]c_int{ 4, @intCast(dk) }, 2, .float32);
    defer _ = mlx.mlx_array_free(x);
    var ones = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones);
    try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{@intCast(dk)}, 1, .float32, s));
    var n = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(n);
    try mlx.check(mlx.mlx_fast_rms_norm(&n, x, ones, cfg.gdnQkNormEps(), s));
    const got = try qwen4ReadF32(testing.allocator, n, s);
    defer testing.allocator.free(got);
    for (0..4) |r| {
        var sum: f64 = 0;
        for (xv[r * dk ..][0..dk]) |v| sum += @as(f64, v) * v;
        const inv = 1.0 / @sqrt(sum + 1e-6);
        for (0..dk) |i| try testing.expectApproxEqRel(xv[r * dk + i] * inv, got[r * dk + i] / @sqrt(@as(f64, dk)), 1e-4);
    }
}
