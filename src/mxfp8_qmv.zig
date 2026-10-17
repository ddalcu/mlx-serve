//! MXFP8 linears at decode and verify widths (1..8 rows): one dispatch, the weight row read
//! once and applied to every activation row. MLX's fp kernel decodes e4m3 per element; here two
//! bytes become a half2 by bit placement.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

pub const MAX_ROWS: c_int = 8;
const SGS: c_int = 4;

/// Bytes 0 and 2 of `w` as a half2 (value x 2^-8), the e4m3 fields shifted into half's.
const HEADER =
    \\inline float2 mlxserve_e4m3x2(uint w) {
    \\  const uint v = w & 0x00FF00FFu;
    \\  return float2(as_type<half2>(((v & 0x007F007Fu) << 7) | ((v & 0x00800080u) << 8)));
    \\}
    \\
;

var kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;

/// y [.., N] = x [.., K] . W^T, or null outside the kernel (caller keeps MLX's matmul).
/// `w` [N, K/4] u32 e4m3 words, `sc` [N, K/32] e8m0.
pub fn linear(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    // Measured on M4 (g16) and the M5 Ultra (g17); other generations keep MLX's kernel until someone measures them.
    const g = @import("transformer.zig").deviceGeneration();
    if ((g.gen != 16 and g.gen != 17) or g.phone) return null;
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .bfloat16 and dt != .float16) return null;
    const wsh = mlx.getShape(w);
    const ssh = mlx.getShape(sc);
    if (wsh.len != 2 or ssh.len != 2 or mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(sc) != .uint8) return null;
    const N = wsh[0];
    const K = wsh[1] * 4;
    if (ssh[0] != N or ssh[1] * 32 != K) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len == 0 or xsh.len > 8 or xsh[xsh.len - 1] != K) return null;
    var rows: c_int = 1;
    for (xsh[0 .. xsh.len - 1]) |d| rows *= d;
    if (rows < 1 or rows > MAX_ROWS) return null;

    if (kernel == null) {
        const ins = [_][*:0]const u8{ "x", "w_q", "scales" };
        const outs = [_][*:0]const u8{"y"};
        const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const k = mlx.mlx_fast_metal_kernel_new("mlxserve_mxfp8_qmv", in_vec, out_vec, @embedFile("kernels/mxfp8_qmv.metal"), HEADER, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        kernel = k;
    }
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    var out_shape: [8]c_int = undefined;
    @memcpy(out_shape[0..xsh.len], xsh);
    out_shape[xsh.len - 1] = N;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, xsh.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32, N, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, SGS, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
    inline for (.{ .{ "K", K }, .{ "N", N }, .{ "M", rows } }) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t[0], t[1]));
    const inputs = [_]mlx.mlx_array{ x, w, sc };
    const v = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, kernel.?, v, c, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, o, 0));
    if (!engaged) {
        engaged = true;
        log.info("[mxfp8] qmv kernel engaged: rows={d} N={d} K={d}\n", .{ rows, N, K });
    }
    return y;
}

const testing = std.testing;

fn randBf16(rnd: std.Random, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(buf);
    for (buf) |*val| val.* = (rnd.float(f32) - 0.5) * 2.0;
    const f = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

test "mxfp8 qmv: 1..8 rows no worse than MLX's matmul against the fp32 truth (K a whole step and not)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const xfm = @import("transformer.zig");
    xfm.device_gen_override = .{ .gen = 16, .phone = false }; // the gate is a perf choice; the numerics are checked on every Mac
    defer xfm.device_gen_override = null;
    // 320: a tensor-parallel rank's half of a 640-wide expert.
    for ([_][2]c_int{ .{ 640, 2560 }, .{ 2560, 640 }, .{ 320, 2560 }, .{ 2560, 320 } }) |nk| try mxfp8Case(nk[0], nk[1]);
}

test "mxfp8 qmv: only the measured GPU generations (M4, M5) take the kernel" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const xfm = @import("transformer.zig");
    defer xfm.device_gen_override = null;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_zeros(&w, &[_]c_int{ 640, 640 }, 2, .uint32, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_zeros(&sc, &[_]c_int{ 640, 80 }, 2, .uint8, s));
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_zeros(&x, &[_]c_int{ 1, 2560 }, 2, .bfloat16, s));
    for ([_]u32{ 13, 15, 18 }) |gen| {
        xfm.device_gen_override = .{ .gen = gen, .phone = false };
        try std.testing.expect((try linear(x, w, sc, s)) == null);
    }
    for ([_]u32{ 16, 17 }) |gen| {
        xfm.device_gen_override = .{ .gen = gen, .phone = false };
        const y = (try linear(x, w, sc, s)) orelse return error.KernelDeclined;
        _ = mlx.mlx_array_free(y);
    }
}

fn mxfp8Case(N: c_int, K: c_int) !void {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x8E4);
    const rnd = prng.random();
    const w = try randBf16(rnd, &.{ N, K }, s);
    defer _ = mlx.mlx_array_free(w);
    var t = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(t);
    try mlx.check(mlx.mlx_quantize(&t, w, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", .{}, s));
    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_vector_array_get(&q, t, 0));
    try mlx.check(mlx.mlx_vector_array_get(&sc, t, 1));
    var deq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(deq);
    try mlx.check(mlx.mlx_dequantize(&deq, q, sc, .{ .ctx = null }, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
    var deq_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(deq_t);
    try mlx.check(mlx.mlx_transpose(&deq_t, deq, s));
    for ([_]c_int{ 1, 3, 5, 8 }) |rows| {
        const x = try randBf16(rnd, &.{ 1, rows, K }, s);
        defer _ = mlx.mlx_array_free(x);
        var x32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x32);
        try mlx.check(mlx.mlx_astype(&x32, x, .float32, s));
        var truth = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(truth);
        try mlx.check(mlx.mlx_matmul(&truth, x32, deq_t, s));
        var stock = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stock);
        try mlx.check(mlx.mlx_quantized_matmul(&stock, x, q, sc, .{ .ctx = null }, true, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", s));
        const ours = (try linear(x, q, sc, s)) orelse return error.KernelDeclined;
        defer _ = mlx.mlx_array_free(ours);
        try testing.expectEqualSlices(c_int, mlx.getShape(stock), mlx.getShape(ours));
        var errs: [2]f64 = undefined;
        for ([_]mlx.mlx_array{ stock, ours }, &errs) |got, *e| {
            var g32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(g32);
            try mlx.check(mlx.mlx_astype(&g32, got, .float32, s));
            try mlx.check(mlx.mlx_array_eval(g32));
            try mlx.check(mlx.mlx_array_eval(truth));
            const pg = mlx.mlx_array_data_float32(g32).?;
            const pt = mlx.mlx_array_data_float32(truth).?;
            var se: f64 = 0;
            var st: f64 = 0;
            for (0..@intCast(rows * N)) |i| {
                se += (pg[i] - pt[i]) * (pg[i] - pt[i]);
                st += pt[i] * pt[i];
            }
            e.* = @sqrt(se / st);
        }
        std.debug.print("[mxfp8] rows={d}: rel rms vs fp32 truth stock {d:.6} ours {d:.6}\n", .{ rows, errs[0], errs[1] });
        try testing.expect(errs[1] <= errs[0] * 1.05 + 1e-6);
    }
}
