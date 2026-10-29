//! `gelu_ple.zig`'s tests, out of its import graph: a plugin unit that reaches `gelu_ple.zig` never compiles them.

const gelu_ple = @import("gelu_ple.zig");
const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const fusedGeluPleProj = gelu_ple.fusedGeluPleProj;
const fused_activations = @import("../../sdk/blocks/mlp/fused_activations.zig");
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const bf16Scalar = @import("../../mlx/array.zig").bf16Scalar;
const qmatmulBits = @import("../../sdk/blocks/linear/qmatmul.zig").qmatmulBits;
const attn256MaxDiff = @import("../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const gdnParityRand = @import("../../sdk/blocks/gdn/decode_fused.zig").gdnParityRand;

test "fused gelu-ple projection is bit-identical to gelu table -> multiply -> quantized_matmul" {
    const s = mlx.gpuStream();
    const allocator = std.testing.allocator;
    fused_activations.swiglu_fused_override = true;
    defer fused_activations.swiglu_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0x9E1E);
    const rnd = prng.random();
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.compiled_gelu = null;
    xfm.three = bf16Scalar(3.0, s);
    defer _ = mlx.mlx_array_free(xfm.three.?);
    xfm.gelu_inner = bf16Scalar(0.044715, s);
    defer _ = mlx.mlx_array_free(xfm.gelu_inner.?);
    xfm.gelu_coeff = bf16Scalar(0.7978845608028654, s);
    defer _ = mlx.mlx_array_free(xfm.gelu_coeff.?);
    xfm.one = bf16Scalar(1.0, s);
    defer _ = mlx.mlx_array_free(xfm.one);
    xfm.half = bf16Scalar(0.5, s);
    defer _ = mlx.mlx_array_free(xfm.half);
    const table = (try xfm.geluTanhTable(.bfloat16)) orelse return error.NoTable;
    const HID: c_int = 2560;
    const PD: c_int = 256;
    for ([_]f32{ 1.0, 12.0, 60.0 }) |scale| {
        const wb = try gdnParityRand(rnd, &.{ HID, PD }, 1.0, .bfloat16, s);
        defer _ = mlx.mlx_array_free(wb);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
        var pw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pw);
        var ps = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ps);
        var pb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pb);
        try mlx.check(mlx.mlx_vector_array_get(&pw, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&ps, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&pb, triple, 2));
        const g = try gdnParityRand(rnd, &.{ 1, 1, PD }, scale, .bfloat16, s);
        defer _ = mlx.mlx_array_free(g);
        const ple = try gdnParityRand(rnd, &.{ 1, 1, PD }, 4.0, .bfloat16, s);
        defer _ = mlx.mlx_array_free(ple);
        const act = try xfm.gelu(g);
        defer _ = mlx.mlx_array_free(act);
        var gated = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gated);
        try mlx.check(mlx.mlx_multiply(&gated, act, ple, s));
        const ref = try qmatmulBits(gated, pw, ps, pb, 4, 64, .affine, s);
        defer _ = mlx.mlx_array_free(ref);
        const got = (try fusedGeluPleProj(s, g, ple, table, pw, ps, pb, .{ .bits = 4, .group_size = 64, .mode = .affine })) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(got);
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got, ref, s));
    }
    _ = allocator;
}
