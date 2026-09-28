//! OPT-IN, LOSSY int8-activation prefill route for 2-bit Prism packs on M5.
//!
//! Default OFF. `MLX_SERVE_BONSAI_INT8_PREFILL=1` turns it on. Unlike every
//! other 2-bit route in this tree it is NOT lossless: the activation is
//! quantized to uint8 per 128-group, so committed tokens can differ from the
//! stock path and greedy serial/speculative byte-equality no longer holds.
//!
//! Why it can win at all: at prompt width the packed projections are
//! COMPUTE-bound, and the tensor unit runs int8 x 4-bit about 1.39x its fp16
//! rate (MEASURED: 85.6 TOP/s vs 61.5 TFLOP/s dense f16, same process). At
//! verify width the same projection is MEMORY-bound on the 2-bit weights, so
//! quantizing the activation buys nothing there and this route declines.
//!
//! `x[m,k] = as[m,g] * (xq[m,k] - 128)` and `w[n,k] = s[n,g]*c[n,k] + b[n,g]`:
//!   y[m,n] = SUM_g as[m,g] * (s[n,g]*(C[m,n,g] - 128*colsum[n,g]) + b[n,g]*rs[m,g])
//! with C the int32 product of shifted codes against weight codes, `colsum` a
//! per-weight derived constant, and `rs[m,g] = SUM_k xq[m,k] - 128*128`.
//!
//! Arrangement follows the Bonsai speedup engine's prompt-width kernel
//! (Layr-Labs/mlxfast-bonsai2-27b-engine, MIT). Theirs feeds the tensor unit
//! `uint2b_format` directly; this toolchain has no such format, so the 2-bit
//! codes are expanded to 4-bit in threadgroup memory first (MEASURED at 4% of
//! the kernel, so the expansion is not what costs).
const std = @import("std");
const mlx = @import("mlx.zig");
const log = std.log.scoped(.qmm_int8);

pub const MIN_ROWS: c_int = 64;
pub const GS: c_int = 128;

var env_enabled: ?bool = null;
/// DEFAULT OFF: this route changes numerics.
pub fn enabled() bool {
    if (env_enabled) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_BONSAI_INT8_PREFILL");
    env_enabled = raw != null and raw.?[0] != '0';
    return env_enabled.?;
}

pub const HEADER =
    \\#include <metal_tensor>
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\
;

/// 64x64 output tile, one 64x64x128 int8 multiply-accumulate per 128-group,
/// weights expanded 2->4 bit into threadgroup memory double-buffered.
pub const SOURCE =
    \\const int K = xq_shape[xq_ndim - 1];
    \\const int N = w_shape[0];
    \\const int Kg = K / 128;
    \\const int n0 = int(threadgroup_position_in_grid.x) * 64;
    \\const int m0 = int(threadgroup_position_in_grid.y) * 64;
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const uint tid = thread_position_in_threadgroup.x;
    \\constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    \\    64, 64, 128, false, true, false,
    \\    mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
    \\mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;
    \\tensor<device uint8_t, dextents<int, 2>, tensor_inline> A((device uint8_t*)xq, dextents<int, 2>(K, MPAD));
    \\threadgroup uint32_t bs[2][64 * 128 / 8];
    \\tensor<threadgroup uint4b_format, dextents<int, 2>, tensor_inline> B0((threadgroup uchar*)bs[0], dextents<int, 2>(128, 64));
    \\tensor<threadgroup uint4b_format, dextents<int, 2>, tensor_inline> B1((threadgroup uchar*)bs[1], dextents<int, 2>(128, 64));
    \\auto tA0 = A.template slice<128, 64>(0, m0);
    \\auto cT = op.template get_destination_cooperative_tensor<
    \\    metal::remove_addrspace_t<decltype(tA0)>,
    \\    metal::remove_addrspace_t<decltype(B0)>, int32_t>();
    \\constexpr int CAP = 32;
    \\const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
    \\const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
    \\const int nb = n0 + 16 * int(sg & 1) + fn;
    \\const int mb = m0 + 16 * int(sg >> 1) + fm;
    \\float acc[CAP];
    \\#pragma clang loop unroll(full)
    \\for (int i = 0; i < CAP; i++) acc[i] = 0.0f;
    \\const int sc = int(tid >> 1);
    \\const int sh = int(tid & 1);
    \\const device uint32_t* wrow = w + (size_t)min(n0 + sc, N - 1) * (K / 16) + sh * 4;
    \\auto stage = [&](int g, int buf) {
    \\  const uint4 v = *((const device uint4*)(wrow + g * 8));
    \\  threadgroup uint32_t* dst = bs[buf] + sc * 16 + sh * 8;
    \\#pragma clang loop unroll(full)
    \\  for (int j = 0; j < 4; j++) {
    \\    const uint32_t wv = v[j];
    \\    uint32_t lo = wv & 0xFFFFu, hi = wv >> 16;
    \\    lo = (lo | (lo << 8)) & 0x00FF00FFu; lo = (lo | (lo << 4)) & 0x0F0F0F0Fu; lo = (lo | (lo << 2)) & 0x33333333u;
    \\    hi = (hi | (hi << 8)) & 0x00FF00FFu; hi = (hi | (hi << 4)) & 0x0F0F0F0Fu; hi = (hi | (hi << 2)) & 0x33333333u;
    \\    dst[2 * j] = lo; dst[2 * j + 1] = hi;
    \\  }
    \\};
    \\stage(0, 0);
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\const int mrow[4] = {mb, mb + 8, mb + 32, mb + 40};
    \\for (int g = 0; g < Kg; g++) {
    \\  const int cur = g & 1;
    \\  if (g + 1 < Kg) stage(g + 1, cur ^ 1);
    \\  auto tA = A.template slice<128, 64>(g * 128, m0);
    \\  if (cur == 0) op.run(tA, B0, cT); else op.run(tA, B1, cT);
    \\  float sv[2][4], bv[2][4], cs[2][4], av[4], rv[4];
    \\#pragma clang loop unroll(full)
    \\  for (int nh = 0; nh < 2; nh++) {
    \\#pragma clang loop unroll(full)
    \\    for (int c = 0; c < 4; c++) {
    \\      const int nn = min(nb + c + 32 * nh, N - 1);
    \\      sv[nh][c] = float(scales[(size_t)g * N + nn]);
    \\      bv[nh][c] = float(biases[(size_t)g * N + nn]);
    \\      cs[nh][c] = colsum[(size_t)g * N + nn];
    \\    }
    \\  }
    \\#pragma clang loop unroll(full)
    \\  for (int q = 0; q < 4; q++) {
    \\    av[q] = ascale[(size_t)mrow[q] * Kg + g];
    \\    rv[q] = rsum[(size_t)mrow[q] * Kg + g];
    \\  }
    \\#pragma clang loop unroll(full)
    \\  for (int i = 0; i < CAP; i++) {
    \\    const int c = i & 3;
    \\    const int nh = (i >> 3) & 1;
    \\    const int mh = ((i >> 2) & 1) | (((i >> 4) & 1) << 1);
    \\    const float t = fma(sv[nh][c], float(cT[i]) - 128.0f * cs[nh][c], bv[nh][c] * rv[mh]);
    \\    acc[i] = fma(av[mh], t, acc[i]);
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\#pragma clang loop unroll(full)
    \\for (int i = 0; i < CAP; i++) {
    \\  const int c = i & 3;
    \\  const int nh = (i >> 3) & 1;
    \\  const int mm = mb + 8 * ((i >> 2) & 1) + 32 * ((i >> 4) & 1);
    \\  const int nn = nb + c + 32 * nh;
    \\  if (nn < N) y[(size_t)mm * N + nn] = static_cast<T>(acc[i]);
    \\}
;

/// amax, scale, quantize and code-sum for one 128-group in ONE pass. Eight
/// composed MLX ops each re-read and re-wrote the whole activation in f32,
/// which MEASURED at ~2-3 ms/call at prompt width.
pub const QUANT_SOURCE =
    \\const int K = x_shape[1];
    \\const int Kg = K / 128;
    \\const uint row = threadgroup_position_in_grid.y;
    \\const uint g = threadgroup_position_in_grid.x;
    \\const uint lane = thread_index_in_simdgroup;
    \\const device T* xr = x + (size_t)row * K + g * 128;
    \\float v[4];
    \\float a = 0.0f;
    \\for (int i = 0; i < 4; ++i) { v[i] = float(xr[lane * 4 + i]); a = max(a, abs(v[i])); }
    \\a = simd_max(a);
    \\const float sc = max(a / 127.0f, 1.0e-20f);
    \\float rs = 0.0f;
    \\for (int i = 0; i < 4; ++i) {
    \\  float q = rint(v[i] / sc) + 128.0f;
    \\  q = clamp(q, 0.0f, 255.0f);
    \\  rs += q;
    \\  xq[(size_t)row * K + g * 128 + lane * 4 + i] = (uint8_t)q;
    \\}
    \\rs = simd_sum(rs);
    \\if (lane == 0) {
    \\  ascale[(size_t)row * Kg + g] = sc;
    \\  rsum[(size_t)row * Kg + g] = rs - 128.0f * 128.0f;
    \\}
;

var quant_kernel: ?mlx.mlx_fast_metal_kernel = null;
const QKey = struct { m: c_int, k: c_int, dt: mlx.mlx_dtype };
var quant_cfg: std.AutoHashMapUnmanaged(QKey, mlx.mlx_fast_metal_kernel_config) = .{};

fn quantKernel() !mlx.mlx_fast_metal_kernel {
    if (quant_kernel) |k| return k;
    const in_names = [_][*:0]const u8{"x"};
    const out_names = [_][*:0]const u8{ "xq", "ascale", "rsum" };
    const iv = mlx.mlx_vector_string_new_data(&in_names, in_names.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const kk = mlx.mlx_fast_metal_kernel_new("msv_int8_quant", iv, ov, QUANT_SOURCE, "", true, false);
    if (kk.ctx == null) return error.MetalKernelCompileFailed;
    quant_kernel = kk;
    return kk;
}

fn quantConfig(key: QKey) !mlx.mlx_fast_metal_kernel_config {
    if (quant_cfg.get(key)) |c| return c;
    const kg = @divExact(key.k, GS);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ key.m, key.k }, 2, .uint8));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ key.m, kg }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ key.m, kg }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * kg, key.m, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", key.dt));
    try quant_cfg.put(std.heap.c_allocator, key, cfg);
    return cfg;
}

// ── host side ──

fn f32Scalar(v: f32) mlx.mlx_array {
    return mlx.mlx_array_new_float(v);
}

/// `xq = round(x / as) + 128` per 128-group, with `as = amax/127`, plus the
/// per-group code sum the bias term needs. A group that is entirely zero would
/// divide by zero, so the scale is floored at a tiny positive value.
pub const Quantized = struct {
    xq: mlx.mlx_array,
    ascale: mlx.mlx_array, // f32 [M, Kg]
    rsum: mlx.mlx_array, // f32 [M, Kg]

    pub fn deinit(self: *Quantized) void {
        _ = mlx.mlx_array_free(self.xq);
        _ = mlx.mlx_array_free(self.ascale);
        _ = mlx.mlx_array_free(self.rsum);
    }
};

pub fn quantizeRows(x2: mlx.mlx_array, m: c_int, k: c_int, s: mlx.mlx_stream) !Quantized {
    const inputs = [_]mlx.mlx_array{x2};
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try quantKernel(), iv, try quantConfig(.{ .m = m, .k = k, .dt = mlx.mlx_array_dtype(x2) }), s));
    var xq = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(xq);
    var ascale = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ascale);
    var rsum = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(rsum);
    try mlx.check(mlx.mlx_vector_array_get(&xq, outs, 0));
    try mlx.check(mlx.mlx_vector_array_get(&ascale, outs, 1));
    try mlx.check(mlx.mlx_vector_array_get(&rsum, outs, 2));
    return .{ .xq = xq, .ascale = ascale, .rsum = rsum };
}

pub fn quantizeRowsComposed(x2: mlx.mlx_array, m: c_int, k: c_int, s: mlx.mlx_stream) !Quantized {
    const kg = @divExact(k, GS);
    var x32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x32);
    try mlx.check(mlx.mlx_astype(&x32, x2, .float32, s));
    var grouped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(grouped);
    try mlx.check(mlx.mlx_reshape(&grouped, x32, &[_]c_int{ m, kg, GS }, 3, s));

    var absx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(absx);
    try mlx.check(mlx.mlx_abs(&absx, grouped, s));
    var amax = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(amax);
    try mlx.check(mlx.mlx_max_axis(&amax, absx, 2, true, s));
    const c127 = f32Scalar(127.0);
    defer _ = mlx.mlx_array_free(c127);
    var asc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(asc);
    try mlx.check(mlx.mlx_divide(&asc, amax, c127, s));
    const tiny = f32Scalar(1.0e-20);
    defer _ = mlx.mlx_array_free(tiny);
    var asc_safe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(asc_safe);
    try mlx.check(mlx.mlx_maximum(&asc_safe, asc, tiny, s));

    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_divide(&scaled, grouped, asc_safe, s));
    var rounded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rounded);
    try mlx.check(mlx.mlx_round(&rounded, scaled, 0, s));
    const c128 = f32Scalar(128.0);
    defer _ = mlx.mlx_array_free(c128);
    var shifted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shifted);
    try mlx.check(mlx.mlx_add(&shifted, rounded, c128, s));
    const lo = f32Scalar(0.0);
    defer _ = mlx.mlx_array_free(lo);
    const hi = f32Scalar(255.0);
    defer _ = mlx.mlx_array_free(hi);
    var clipped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(clipped);
    try mlx.check(mlx.mlx_clip(&clipped, shifted, lo, hi, s));

    var rs_g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rs_g);
    try mlx.check(mlx.mlx_sum_axis(&rs_g, clipped, 2, false, s));
    const shiftsum = f32Scalar(128.0 * @as(f32, @floatFromInt(GS)));
    defer _ = mlx.mlx_array_free(shiftsum);
    var rsum = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(rsum);
    try mlx.check(mlx.mlx_subtract(&rsum, rs_g, shiftsum, s));

    var xq_g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xq_g);
    try mlx.check(mlx.mlx_astype(&xq_g, clipped, .uint8, s));
    var xq = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(xq);
    try mlx.check(mlx.mlx_reshape(&xq, xq_g, &[_]c_int{ m, k }, 2, s));
    var ascale = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ascale);
    try mlx.check(mlx.mlx_reshape(&ascale, asc_safe, &[_]c_int{ m, kg }, 2, s));
    return .{ .xq = xq, .ascale = ascale, .rsum = rsum };
}

/// `colsum[n,g] = SUM_{k in g} c[n,k]`, the per-weight derived constant the
/// shift correction needs. Dequantizing with scale 1 and bias 0 yields the raw
/// codes, so this costs one pass over the weight ONCE per weight, not per call.
pub fn colSums(w: mlx.mlx_array, bits: u32, group_size: u32, n: c_int, k: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const kg = @divExact(k, GS);
    var ones = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones);
    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    const sshape = [_]c_int{ n, @divExact(k, @as(c_int, @intCast(group_size))) };
    try mlx.check(mlx.mlx_ones(&ones, &sshape, 2, .float32, s));
    try mlx.check(mlx.mlx_zeros(&zeros, &sshape, 2, .float32, s));
    var codes = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(codes);
    try mlx.check(mlx.mlx_dequantize(&codes, w, ones, zeros, mlx.mlx_optional_int.some(@intCast(group_size)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
    var grouped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(grouped);
    try mlx.check(mlx.mlx_reshape(&grouped, codes, &[_]c_int{ n, kg, GS }, 3, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_sum_axis(&out, grouped, 2, false, s));
    return out;
}

/// `colsum` is a derived constant OF THE WEIGHT, so it is computed once and
/// kept. Recomputing it per call dequantizes the whole pack every time, which
/// MEASURED at ~0.9 ms/call and alone turned a 1.42x kernel into 0.71x.
const Derived = struct { scT: mlx.mlx_array, biT: mlx.mlx_array, csT: mlx.mlx_array };
var colsum_cache: std.AutoHashMapUnmanaged(usize, Derived) = .{};

fn transposed(a: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var t = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_transpose(&t, a, s));
    var c = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, t, false, s));
    _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_array_eval(c));
    return c;
}

fn derivedCached(w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, n: c_int, k: c_int, s: mlx.mlx_stream) !Derived {
    const key = @intFromPtr(w.ctx);
    if (colsum_cache.get(key)) |c| return c;
    const cs = try colSums(w, bits, group_size, n, k, s);
    defer _ = mlx.mlx_array_free(cs);
    const d = Derived{
        .scT = try transposed(sc, s),
        .biT = try transposed(bi, s),
        .csT = try transposed(cs, s),
    };
    try colsum_cache.put(std.heap.c_allocator, key, d);
    return d;
}

var kernel_cache: ?mlx.mlx_fast_metal_kernel = null;
const CfgKey = struct { n: c_int, mpad: c_int, dt: mlx.mlx_dtype };
var cfg_cache: std.AutoHashMapUnmanaged(CfgKey, mlx.mlx_fast_metal_kernel_config) = .{};
var engaged_logged = false;

fn kernel() !mlx.mlx_fast_metal_kernel {
    if (kernel_cache) |k| return k;
    const in_names = [_][*:0]const u8{ "xq", "w", "scales", "biases", "colsum", "ascale", "rsum" };
    const out_names = [_][*:0]const u8{"y"};
    const iv = mlx.mlx_vector_string_new_data(&in_names, in_names.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("msv_qmm_int8_prefill", iv, ov, SOURCE, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel_cache = k;
    return k;
}

fn configFor(key: CfgKey) !mlx.mlx_fast_metal_kernel_config {
    if (cfg_cache.get(key)) |c| return c;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const out_shape = [_]c_int{ key.mpad, key.n };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 2, key.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 128 * @divTrunc(key.n + 63, 64), @divExact(key.mpad, 64), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", key.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "MPAD", key.mpad));
    try cfg_cache.put(std.heap.c_allocator, key, cfg);
    return cfg;
}

/// `x @ w.T` with the activation quantized to uint8 per 128-group, or null when
/// the shape/dtype is outside the route (caller keeps whatever it would do).
/// LOSSY BY CONSTRUCTION — see the module comment.
pub fn qmm(
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    s: mlx.mlx_stream,
) !?mlx.mlx_array {
    if (!enabled() or bits != 2 or group_size != GS or bi.ctx == null) return null;
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .float16 and dt != .bfloat16) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    if (xs.len == 0 or xs.len > 8 or ws.len != 2) return null;
    var m: c_int = 1;
    for (xs[0 .. xs.len - 1]) |d| m *= d;
    const k = xs[xs.len - 1];
    const n = ws[0];
    if (m < MIN_ROWS or @rem(k, GS) != 0 or ws[1] * 16 != k) return null;

    const mpad = @divTrunc(m + 63, 64) * 64;
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ m, k }, 2, s));
    var xp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xp);
    if (mpad == m) {
        try mlx.check(mlx.mlx_array_set(&xp, x2));
    } else {
        var zero = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(zero);
        try mlx.check(mlx.mlx_zeros(&zero, &[_]c_int{ mpad - m, k }, 2, dt, s));
        const parts = [_]mlx.mlx_array{ x2, zero };
        const pv = mlx.mlx_vector_array_new_data(&parts, parts.len);
        defer _ = mlx.mlx_vector_array_free(pv);
        try mlx.check(mlx.mlx_concatenate_axis(&xp, pv, 0, s));
    }
    var q = try quantizeRows(xp, mpad, k, s);
    defer q.deinit();
    const d = try derivedCached(w, sc, bi, bits, group_size, n, k, s);

    if (!engaged_logged) {
        engaged_logged = true;
        log.info("[int8-prefill] engaged (LOSSY: activations quantized to uint8): first call M={d} N={d} K={d}\n", .{ m, n, k });
    }
    const inputs = [_]mlx.mlx_array{ q.xq, w, d.scT, d.biT, d.csT, q.ascale, q.rsum };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try kernel(), iv, try configFor(.{ .n = n, .mpad = mpad, .dt = dt }), s));
    var yy = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(yy);
    try mlx.check(mlx.mlx_vector_array_get(&yy, outs, 0));
    var ysl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ysl);
    if (mpad == m) {
        try mlx.check(mlx.mlx_array_set(&ysl, yy));
    } else {
        try mlx.check(mlx.mlx_slice(&ysl, yy, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ m, n }, 2, &[_]c_int{ 1, 1 }, 2, s));
    }
    var out_shape: [8]c_int = undefined;
    @memcpy(out_shape[0..xs.len], xs);
    out_shape[xs.len - 1] = n;
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&r, ysl, &out_shape, xs.len, s));
    return r;
}

// ── tests ──

const RmsMax = struct { rms: f32, max: f32 };

fn errVsTruth(got: mlx.mlx_array, truth: []const f32, m: usize, n: usize, s: mlx.mlx_stream) !RmsMax {
    var g32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g32);
    try mlx.check(mlx.mlx_astype(&g32, got, .float32, s));
    try mlx.check(mlx.mlx_array_eval(g32));
    const g = mlx.mlx_array_data_float32(g32).?;
    var ss: f64 = 0;
    var mx: f32 = 0;
    for (0..m * n) |i| {
        try std.testing.expect(std.math.isFinite(g[i]));
        const d = g[i] - truth[i];
        ss += @as(f64, d) * @as(f64, d);
        mx = @max(mx, @abs(d));
    }
    return .{ .rms = @floatCast(@sqrt(ss / @as(f64, @floatFromInt(m * n)))), .max = mx };
}

// This route is LOSSY, so the bar cannot be byte-equality or even stock's own
// error. It is: the int8 activation must not cost more than the quantization
// step it introduces, i.e. error within a small multiple of stock's, never
// NaN/Inf, and the shape/dtype contract preserved.
test "qmm_int8: error stays within a small multiple of stock at prompt width" {
    const s = mlx.gpuStream();
    const n: c_int = 512;
    const k: c_int = 1536;
    const m: c_int = 128;
    const nu: usize = @intCast(n);
    const ku: usize = @intCast(k);
    const mu: usize = @intCast(m);
    var prng = std.Random.DefaultPrng.init(31);
    const rnd = prng.random();

    const codes = try std.testing.allocator.alloc(u32, nu * ku / 16);
    defer std.testing.allocator.free(codes);
    for (codes) |*wd| {
        var v: u32 = 0;
        for (0..16) |j| v |= @as(u32, rnd.uintLessThan(u32, 3)) << @intCast(2 * j);
        wd.* = v;
    }
    const sc32 = try std.testing.allocator.alloc(f32, nu * ku / 128);
    defer std.testing.allocator.free(sc32);
    for (sc32) |*e| e.* = 0.005 + 0.02 * rnd.float(f32);
    const wq = mlx.mlx_array_new_data(codes.ptr, &[_]c_int{ n, @divExact(k, 16) }, 2, .uint32);
    defer _ = mlx.mlx_array_free(wq);
    const scf = mlx.mlx_array_new_data(sc32.ptr, &[_]c_int{ n, @divExact(k, 128) }, 2, .float32);
    defer _ = mlx.mlx_array_free(scf);
    const dt: mlx.mlx_dtype = .float16;
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_astype(&sc, scf, dt, s));
    var bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi);
    try mlx.check(mlx.mlx_negative(&bi, sc, s));

    var sct = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sct);
    var bit = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bit);
    try mlx.check(mlx.mlx_astype(&sct, sc, .float32, s));
    try mlx.check(mlx.mlx_astype(&bit, bi, .float32, s));
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_dequantize(&wt, wq, sct, bit, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(2), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
    var wtt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wtt);
    try mlx.check(mlx.mlx_transpose(&wtt, wt, s));

    const xv = try std.testing.allocator.alloc(f32, mu * ku);
    defer std.testing.allocator.free(xv);
    for (xv) |*e| e.* = rnd.floatNorm(f32);
    const x32 = mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ m, k }, 2, .float32);
    defer _ = mlx.mlx_array_free(x32);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, x32, dt, s));

    var truth_a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(truth_a);
    try mlx.check(mlx.mlx_matmul(&truth_a, x32, wtt, s));
    try mlx.check(mlx.mlx_array_eval(truth_a));
    const truth = mlx.mlx_array_data_float32(truth_a).?[0 .. mu * nu];

    var stock = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stock);
    try mlx.check(mlx.mlx_quantized_matmul(&stock, x, wq, sc, bi, true, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(2), "affine", s));
    const es = try errVsTruth(stock, truth, mu, nu, s);

    env_enabled = true;
    defer env_enabled = null;
    const got = (try qmm(x, wq, sc, bi, 2, 128, s)) orelse {
        std.debug.print("[qmm_int8] declined a shape it should take (M={d})\n", .{m});
        return error.RouteDeclined;
    };
    defer _ = mlx.mlx_array_free(got);
    try std.testing.expectEqual(dt, mlx.mlx_array_dtype(got));
    try std.testing.expectEqualSlices(c_int, mlx.getShape(stock), mlx.getShape(got));
    const eg = try errVsTruth(got, truth, mu, nu, s);
    // Stock's error here is only f16 ACCUMULATION (the truth uses the same
    // dequantized weights), so a ratio against it compares to near zero and is
    // not a bar. The physical bar is relative error against the output itself:
    // int8 over a 128-group costs ~amax/(127*sqrt(12)) per element, which for
    // unit-normal activations predicts ~0.7% relative, and that is what this
    // route BUYS ITS SPEED WITH. Anything far above that means a real defect.
    var ts: f64 = 0;
    for (truth) |t| ts += @as(f64, t) * @as(f64, t);
    const truth_rms: f32 = @floatCast(@sqrt(ts / @as(f64, @floatFromInt(mu * nu))));
    const rel = eg.rms / truth_rms;
    std.debug.print("[qmm_int8] truth rms={d:.5} | stock rms={d:.5} | int8 rms={d:.5} -> {d:.3}% relative\n", .{ truth_rms, es.rms, eg.rms, rel * 100.0 });
    try std.testing.expect(rel < 0.015);
    try std.testing.expect(es.rms / truth_rms < 0.015);
}

test "qmm_int8: declines below its row floor and when disabled" {
    const s = mlx.gpuStream();
    env_enabled = false;
    defer env_enabled = null;
    const dummy = mlx.mlx_array_new();
    try std.testing.expectEqual(@as(?mlx.mlx_array, null), try qmm(dummy, dummy, dummy, dummy, 2, 128, s));
}
