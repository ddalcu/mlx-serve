//! The fp8 / fp4 encoders for kernels that quantize inside themselves (`kernels/common/fp_encode.metal`): MLX's
//! fp8_e4m3 / fp8_e8m0 / fp4_e2m1 constructors and mx_scale_round_up, copied, with the NaN-keeping max / min / clip
//! of MLX's ops. `HEADER` carries the scale reads (`fp4.SCALE_HEADER`) it builds on.
const std = @import("std");
const kernelSource = @import("../../mlx/kernels.zig").source;
const fp4 = @import("fp4.zig");
const mlx = @import("../../mlx/ffi.zig");

pub const HEADER = fp4.SCALE_HEADER ++ kernelSource("kernels/common/fp_encode.metal");

// ── tests ──

/// One launch of `source` (with `HEADER`) over `grid` threads in threadgroups of `tg`, its outputs as uint8 arrays of
/// `out_n` elements each; `gs` the template GS.
fn run(name: [*:0]const u8, ins: []const [*:0]const u8, args: []const mlx.mlx_array, outs: []const [*:0]const u8, out_n: []const c_int, source: [*:0]const u8, grid: c_int, tg: c_int, gs: c_int, res: []mlx.mlx_array) !void {
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, HEADER, true, false);
    defer _ = mlx.mlx_fast_metal_kernel_free(k);
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    for (out_n) |n| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{n}, 1, .uint8));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, grid, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "GS", gs));
    const av = mlx.mlx_vector_array_new_data(args.ptr, args.len);
    defer _ = mlx.mlx_vector_array_free(av);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, k, av, c, mlx.gpuStream()));
    for (res, 0..) |*r, i| {
        r.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(r, o, i));
        try mlx.check(mlx.mlx_array_eval(r.*));
    }
}

test "fp encode: e4m3 encode is MLX's to_fp8 on every float32 bit pattern" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const chunk: u64 = 1 << 26;
    var base: u64 = 0;
    while (base < 1 << 32) : (base += chunk) {
        var bits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bits);
        try mlx.check(mlx.mlx_arange(&bits, @floatFromInt(base), @floatFromInt(base + chunk), 1, .uint32, s));
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_view(&x, bits, .float32, s));
        var theirs = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(theirs);
        try mlx.check(mlx.mlx_to_fp8(&theirs, x, s));
        var ours: [1]mlx.mlx_array = undefined;
        try run("mlxserve_fp_encode_e4m3", &.{"x"}, &.{x}, &.{"y"}, &.{@intCast(chunk)}, "const uint i = thread_position_in_grid.x;\ny[i] = uchar(mlxserve_fp8_e4m3_bits(x[i]));\n", @intCast(chunk), 256, 32, &ours);
        defer _ = mlx.mlx_array_free(ours[0]);
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_equal(&eq, ours[0], theirs, s));
        var all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(all);
        try mlx.check(mlx.mlx_all(&all, eq, false, s));
        try mlx.check(mlx.mlx_array_eval(all));
        var ok: bool = false;
        try mlx.check(mlx.mlx_array_item_bool(&ok, all));
        if (!ok) std.debug.print("e4m3 encode differs from MLX's in bit patterns [0x{x}, 0x{x})\n", .{ base, base + chunk });
        try std.testing.expect(ok);
    }
}

test "fp encode: mx_scale_round_up's bit form and clip's clamp are MLX's steps on every float32 bit pattern" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    // y = 1 where ours differs from MLX's own steps (fp8.h's round of log2 and bump; minimum(maximum(a, lo), hi)).
    const src =
        \\const uint i = thread_position_in_grid.x;
        \\const float x = x_in[i];
        \\uint e = mlxserve_fp8_e8m0_bits(x);
        \\if (e < 0xFEu && mlxserve_fp4_e8m0(e) < x) e += 1u;
        \\bool bad = as_type<uint>(mlxserve_mx_scale_round_up(x)) != as_type<uint>(mlxserve_fp4_e8m0(e));
        \\bad = bad || as_type<uint>(mlxserve_clip(x, -448.0f, 448.0f)) != as_type<uint>(mlxserve_minimum(mlxserve_maximum(x, -448.0f), 448.0f));
        \\bad = bad || as_type<uint>(mlxserve_clip(x, -6.0f, 6.0f)) != as_type<uint>(mlxserve_minimum(mlxserve_maximum(x, -6.0f), 6.0f));
        \\y[i] = uchar(bad);
        \\
    ;
    const chunk: u64 = 1 << 26;
    var base: u64 = 0;
    while (base < 1 << 32) : (base += chunk) {
        var bits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bits);
        try mlx.check(mlx.mlx_arange(&bits, @floatFromInt(base), @floatFromInt(base + chunk), 1, .uint32, s));
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_view(&x, bits, .float32, s));
        var ours: [1]mlx.mlx_array = undefined;
        try run("mlxserve_fp_encode_scale_clip", &.{"x_in"}, &.{x}, &.{"y"}, &.{@intCast(chunk)}, src, @intCast(chunk), 256, 32, &ours);
        defer _ = mlx.mlx_array_free(ours[0]);
        var same = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(same);
        try mlx.check(mlx.mlx_logical_not(&same, ours[0], s));
        var all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(all);
        try mlx.check(mlx.mlx_all(&all, same, false, s));
        try mlx.check(mlx.mlx_array_eval(all));
        var ok: bool = false;
        try mlx.check(mlx.mlx_array_item_bool(&ok, all));
        if (!ok) std.debug.print("scale or clip differs from MLX's steps in bit patterns [0x{x}, 0x{x})\n", .{ base, base + chunk });
        try std.testing.expect(ok);
    }
}

/// MLX's fp_quantize (fp_quantized.h) on our encoders: the block max over the simdgroup (16-lane halves for NVFP4),
/// over 6, the e8m0 scale rounded up (MXFP4) or the e4m3 one (NVFP4), each value times the scale's inverse to e2m1.
const quantize_src =
    \\const uint i = thread_position_in_grid.x;
    \\const uint lane = thread_index_in_simdgroup;
    \\const float wt = w[i];
    \\float amax;
    \\if (GS == 32) {
    \\  amax = mlxserve_simd_max(metal::abs(wt));
    \\} else {
    \\  const float l = mlxserve_simd_max(lane < 16 ? metal::abs(wt) : 0.0f);
    \\  const float r = mlxserve_simd_max(lane >= 16 ? metal::abs(wt) : 0.0f);
    \\  amax = lane < 16 ? l : r;
    \\}
    \\float sd = amax / 6.0f;
    \\uint sb;
    \\if (GS == 32) {
    \\  sb = mlxserve_fp8_e8m0_bits(mlxserve_mx_scale_round_up(sd));
    \\} else {
    \\  sb = mlxserve_fp8_e4m3_bits(sd);
    \\}
    \\const float sf = mlxserve_fp4_scale_of<GS>(sb);
    \\const float enc = sf == 0 ? 0.0f : 1.0f / sf;
    \\codes[i] = uchar(mlxserve_fp4_e2m1_bits(wt * enc));
    \\if (i % GS == 0) scales[i / GS] = uchar(sb);
    \\
;

test "fp encode: e8m0 / e4m3 scales and e2m1 codes are MLX's mxfp4 / nvfp4 quantize, across the exponent range" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    // Blocks of 32 with maxima from 2^-140 to 2^127, each value random below the max, a code threshold times the
    // block's power of two, a zero or a negative zero; block 0 all zero.
    const blocks = 4096;
    const n = blocks * 32;
    const w = try a.alloc(f32, n);
    defer a.free(w);
    var prng = std.Random.DefaultPrng.init(11);
    const rng = prng.random();
    const ties = [_]f32{ 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6 };
    for (0..blocks) |b| {
        const e: i32 = -140 + @as(i32, @intCast(b % 268));
        for (w[b * 32 ..][0..32], 0..) |*v, j| {
            const sign: f32 = if (rng.boolean()) -1 else 1;
            v.* = switch (rng.uintLessThan(u8, 4)) {
                0 => sign * std.math.ldexp(ties[j % ties.len], e - 2),
                1 => if (rng.boolean()) 0.0 else -0.0,
                else => sign * std.math.ldexp(rng.float(f32) * 2, e),
            };
        }
        if (b == 0) @memset(w[0..32], 0);
    }
    const wa = mlx.mlx_array_new_data(w.ptr, &[_]c_int{ blocks, 32 }, 2, .float32);
    defer _ = mlx.mlx_array_free(wa);
    for ([_]fp4.Format{ .mxfp4, .nvfp4 }) |f| {
        const gs: usize = f.group();
        var q = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(q);
        try mlx.check(mlx.mlx_quantize(&q, wa, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(4), f.mode().cstr(), .{ .ctx = null }, s));
        var wq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wq);
        try mlx.check(mlx.mlx_vector_array_get(&wq, q, 0));
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        try mlx.check(mlx.mlx_vector_array_get(&sc, q, 1));
        try mlx.check(mlx.mlx_array_eval(wq));
        try mlx.check(mlx.mlx_array_eval(sc));
        var ours: [2]mlx.mlx_array = undefined;
        try run("mlxserve_fp_encode_quantize", &.{"w"}, &.{wa}, &.{ "codes", "scales" }, &.{ n, @intCast(n / gs) }, quantize_src, n, 32, @intCast(gs), &ours);
        defer for (ours) |o| {
            _ = mlx.mlx_array_free(o);
        };
        const packed_bytes: [*]const u8 = @ptrCast(mlx.mlx_array_data_uint32(wq) orelse return error.NoData);
        const their_scales = mlx.mlx_array_data_uint8(sc) orelse return error.NoData;
        const codes = mlx.mlx_array_data_uint8(ours[0]) orelse return error.NoData;
        const scales = mlx.mlx_array_data_uint8(ours[1]) orelse return error.NoData;
        var bad: usize = 0;
        for (0..n / gs) |g| if (scales[g] != their_scales[g]) {
            if (bad < 8) std.debug.print("{t} block {d}: scale 0x{x:0>2}, MLX 0x{x:0>2}\n", .{ f, g, scales[g], their_scales[g] });
            bad += 1;
        };
        for (0..n) |i| {
            const theirs = (packed_bytes[i / 2] >> @intCast(4 * (i % 2))) & 0xF;
            if (codes[i] != theirs) {
                if (bad < 8) std.debug.print("{t} value {d} ({e}): code 0x{x}, MLX 0x{x}\n", .{ f, i, w[i], codes[i], theirs });
                bad += 1;
            }
        }
        try std.testing.expectEqual(@as(usize, 0), bad);
    }
}
