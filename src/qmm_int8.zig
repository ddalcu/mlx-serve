//! OPT-IN, LOSSY int8-activation prefill route for 2-bit Prism packs on M5.
//!
//! Default OFF. `MLX_SERVE_BONSAI_INT8_PREFILL=1` turns it on. Unlike every
//! other 2-bit route in this tree it is NOT lossless: the activation is
//! quantized to int8 per 128-group, so committed tokens can differ from the
//! stock path and greedy serial/speculative byte-equality no longer holds.
//!
//! Why it can win at all: at prompt width the packed projections are
//! COMPUTE-bound, and the tensor unit's int8 x int8 rate is about twice its
//! f16 rate on M5.
//!
//! `x[m,k] = as[m,g] * xq[m,k]` (signed) and `w[n,k] = s[n,g]*c[n,k] + b[n,g]`:
//!   y[m,n] = SUM_g as[m,g] * (s[n,g]*C[m,n,g] + b[n,g]*rs[m,g])
//! with C the int32 product of activation codes against weight codes and
//! `rs[m,g] = SUM_k xq[m,k]`.
//!
//! Arrangement follows the Bonsai speedup engine's prompt-width kernel
//! (Layr-Labs/mlxfast-bonsai2-27b-engine, MIT). Theirs feeds the tensor unit
//! `uint2b_format` directly where it exists; this toolchain has no such format,
//! so the 2-bit codes are expanded to int8 straight into the tensor op's right
//! operand in registers, as their register-staged form does.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = std.log.scoped(.qmm_int8);

pub const MIN_ROWS: c_int = 64;
pub const GS: c_int = 128;

var env_enabled: ?bool = null;
/// The process default (a model's `int8_prefill` setting overrides it).
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

/// Four simdgroups, each one 32x32x128 int8 multiply per 128-group over a
/// 32-row x 32-column tile. The right operand is built in REGISTERS from the
/// row-major 2-bit words: lane l holds columns nl + 8c and, from every word,
/// the plane `(w >> 2kq) & 0x03030303` (codes kq, kq+4, kq+8, kq+12). That is a
/// fixed 4x4 transpose of K inside each 16-block, which the quantizer applies
/// to the activation too, so the integer products are unchanged.
pub const SOURCE =
    \\const int K = xq_shape[xq_ndim - 1];
    \\const int N = w_shape[0];
    \\const int Kg = K / 128;
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const int ms = int(threadgroup_position_in_grid.y) * 32;
    \\const int ns = int(threadgroup_position_in_grid.x) * 128 + 32 * int(sg);
    \\constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    \\    32, 32, 128, false, true, false,
    \\    mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
    \\mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    \\tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq, dextents<int, 2>(K, MPAD));
    \\auto bT = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    \\thread uint32_t* bw = (thread uint32_t*)&bT;
    \\auto tA0 = A.template slice<128, 32>(0, ms);
    \\auto cT = op.template get_destination_cooperative_tensor<
    \\    metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(bT)>, int32_t>();
    \\constexpr int CAP = 32;
    \\const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
    \\const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
    \\const int nb = ns + fn;
    \\const int mb = ms + fm;
    \\float acc[CAP];
    \\#pragma clang loop unroll(full)
    \\for (int i = 0; i < CAP; i++) acc[i] = 0.0f;
    \\const int nl = int(((lane >> 1) & 3) + 4 * ((lane >> 4) & 1));
    \\const uint sh = 2 * ((lane & 1) + 2 * ((lane >> 3) & 1));
    \\const device uint4* wc[4];
    \\#pragma clang loop unroll(full)
    \\for (int c = 0; c < 4; c++) wc[c] = (const device uint4*)(w + (size_t)min(ns + nl + 8 * c, N - 1) * (K / 16));
    \\const int mrow[4] = {mb, mb + 8, mb + 16, mb + 24};
    \\for (int g = 0; g < Kg; g++) {
    \\  uint4 wv[8];
    \\#pragma clang loop unroll(full)
    \\  for (int c = 0; c < 4; c++) { wv[2 * c] = wc[c][2 * g]; wv[2 * c + 1] = wc[c][2 * g + 1]; }
    \\#pragma clang loop unroll(full)
    \\  for (int c = 0; c < 4; c++) {
    \\    const uint s2 = sh;
    \\    const uint4 lo = wv[2 * c];
    \\    const uint4 hi = wv[2 * c + 1];
    \\    bw[c + 0] = (lo.x >> s2) & 0x03030303u; bw[c + 4] = (lo.y >> s2) & 0x03030303u;
    \\    bw[c + 8] = (lo.z >> s2) & 0x03030303u; bw[c + 12] = (lo.w >> s2) & 0x03030303u;
    \\    bw[c + 16] = (hi.x >> s2) & 0x03030303u; bw[c + 20] = (hi.y >> s2) & 0x03030303u;
    \\    bw[c + 24] = (hi.z >> s2) & 0x03030303u; bw[c + 28] = (hi.w >> s2) & 0x03030303u;
    \\  }
    \\  auto tA = A.template slice<128, 32>(g * 128, ms);
    \\  op.run(tA, bT, cT);
    \\  float4 sv[2], bv[2];
    \\  float av[4], rv[4];
    \\#pragma clang loop unroll(full)
    \\  for (int h = 0; h < 2; h++) {
    \\    if (FULL) {
    \\      sv[h] = float4(*(const device vec<ST, 4>*)(scales + (size_t)g * N + nb + 16 * h));
    \\      bv[h] = NEG ? float4(0.0f) : float4(*(const device vec<ST, 4>*)(biases + (size_t)g * N + nb + 16 * h));
    \\    } else {
    \\#pragma clang loop unroll(full)
    \\      for (int c = 0; c < 4; c++) {
    \\        const size_t e = (size_t)g * N + min(nb + c + 16 * h, N - 1);
    \\        sv[h][c] = float(scales[e]);
    \\        bv[h][c] = NEG ? 0.0f : float(biases[e]);
    \\      }
    \\    }
    \\  }
    \\#pragma clang loop unroll(full)
    \\  for (int q = 0; q < 4; q++) {
    \\    av[q] = ascale[(size_t)mrow[q] * Kg + g];
    \\    rv[q] = rsum[(size_t)mrow[q] * Kg + g];
    \\    if (NEG) rv[q] *= av[q];
    \\  }
    \\  // Ternary packs (bias == -scale): acc += s * (as * C - as * rs).
    \\#pragma clang loop unroll(full)
    \\  for (int i = 0; i < CAP; i++) {
    \\    const int c = i & 3;
    \\    const int nh = (i >> 3) & 1;
    \\    const int mh = ((i >> 2) & 1) | (((i >> 4) & 1) << 1);
    \\    if (NEG) {
    \\      acc[i] = fma(sv[nh][c], fma(av[mh], float(cT[i]), -rv[mh]), acc[i]);
    \\    } else {
    \\      const float t = fma(sv[nh][c], float(cT[i]), bv[nh][c] * rv[mh]);
    \\      acc[i] = fma(av[mh], t, acc[i]);
    \\    }
    \\  }
    \\}
    \\#pragma clang loop unroll(full)
    \\for (int i = 0; i < CAP; i += 4) {
    \\  const int nn = nb + 16 * ((i >> 3) & 1);
    \\  const int mm = mb + 8 * ((i >> 2) & 1) + 16 * ((i >> 4) & 1);
    \\  if (FULL) {
    \\    *(device vec<T, 4>*)(y + (size_t)mm * N + nn) = vec<T, 4>(acc[i], acc[i + 1], acc[i + 2], acc[i + 3]);
    \\  } else {
    \\#pragma clang loop unroll(full)
    \\    for (int c = 0; c < 4; c++) if (nn + c < N) y[(size_t)mm * N + nn + c] = static_cast<T>(acc[i + c]);
    \\  }
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
    \\// Rows past the input's are the tile's padding: they quantize as zeros.
    \\const bool live = int(row) < x_shape[0];
    \\const device T* xr = x + (size_t)(live ? row : 0) * K + g * 128;
    \\float v[4];
    \\float a = 0.0f;
    \\for (int i = 0; i < 4; ++i) { v[i] = live ? float(xr[lane * 4 + i]) : 0.0f; a = max(a, abs(v[i])); }
    \\a = simd_max(a);
    \\const float sc = max(a / 127.0f, 1.0e-20f);
    \\float rs = 0.0f;
    \\for (int i = 0; i < 4; ++i) {
    \\  const float q = clamp(rint(v[i] / sc), -127.0f, 127.0f);
    \\  rs += q;
    \\  // Kernel K order: position 4b + kq of a 16-block is stored at 4kq + b.
    \\  const uint p = lane * 4 + i;
    \\  xq[(size_t)row * K + g * 128 + (p & ~15u) + (p & 3u) * 4 + ((p >> 2) & 3u)] = (int8_t)q;
    \\}
    \\rs = simd_sum(rs);
    \\if (lane == 0) {
    \\  ascale[(size_t)row * Kg + g] = sc;
    \\  rsum[(size_t)row * Kg + g] = rs;
    \\}
;

/// `H(signs * x)` over one 1024 block (rht.zig's butterfly, rounded through T
/// where the unfused route stores it), then `QUANT_SOURCE` on its eight
/// 128-groups: the int8 route's input without the rotated activation in memory.
pub const ROTQ_SOURCE =
    \\constexpr int S = 4;
    \\constexpr int P = 8;
    \\threadgroup float tg[1024];
    \\const int K = x_shape[1];
    \\const int Kg = K / 128;
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const uint col = threadgroup_position_in_grid.x * 1024;
    \\const uint row = threadgroup_position_in_grid.y;
    \\const bool live = int(row) < x_shape[0];
    \\const device T* xr = x + (size_t)(live ? row : 0) * K + col;
    \\float v[P];
    \\for (int p = 0; p < P; ++p) {
    \\  const uint i = (sg * P + p) * 32 + lane;
    \\  v[p] = live ? float(xr[i]) * float(signs[col + i]) : 0.0f;
    \\}
    \\for (int h = 1; h < P; h <<= 1) {
    \\  for (int p = 0; p < P; ++p) {
    \\    if ((p & h) == 0) { float a = v[p], b = v[p + h]; v[p] = a + b; v[p + h] = a - b; }
    \\  }
    \\}
    \\for (uint m = 1; m < 32; m <<= 1) {
    \\  float sgn = (lane & m) ? -1.0f : 1.0f;
    \\  for (int p = 0; p < P; ++p) v[p] = fma(sgn, v[p], simd_shuffle_xor(v[p], m));
    \\}
    \\for (int p = 0; p < P; ++p) tg[(sg * P + p) * 32 + lane] = v[p];
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\const float scale = rsqrt(1024.0f);
    \\float o[2][S];
    \\for (int t = 0; t < 2; ++t) {
    \\  const uint p = sg + S * t;
    \\  float w[S];
    \\  for (int q = 0; q < S; ++q) w[q] = tg[(q * P + p) * 32 + lane];
    \\  float a0 = w[0] + w[1], a1 = w[0] - w[1], a2 = w[2] + w[3], a3 = w[2] - w[3];
    \\  w[0] = a0 + a2; w[2] = a0 - a2; w[1] = a1 + a3; w[3] = a1 - a3;
    \\  for (int q = 0; q < S; ++q) o[t][q] = float(static_cast<T>(w[q] * scale));
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\for (int t = 0; t < 2; ++t)
    \\  for (int q = 0; q < S; ++q) tg[(q * P + sg + S * t) * 32 + lane] = o[t][q];
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\for (int gg = 0; gg < 2; ++gg) {
    \\  const uint gl = sg * 2 + gg;
    \\  float e[4];
    \\  float a = 0.0f;
    \\  for (int i = 0; i < 4; ++i) { e[i] = tg[gl * 128 + lane * 4 + i]; a = max(a, abs(e[i])); }
    \\  a = simd_max(a);
    \\  const float sc = max(a / 127.0f, 1.0e-20f);
    \\  float rs = 0.0f;
    \\  const size_t gb = (size_t)row * K + col + gl * 128;
    \\  for (int i = 0; i < 4; ++i) {
    \\    const float q = clamp(rint(e[i] / sc), -127.0f, 127.0f);
    \\    rs += q;
    \\    const uint p = lane * 4 + i;
    \\    xq[gb + (p & ~15u) + (p & 3u) * 4 + ((p >> 2) & 3u)] = (int8_t)q;
    \\  }
    \\  rs = simd_sum(rs);
    \\  if (lane == 0) {
    \\    const size_t gi = (size_t)row * Kg + col / 128 + gl;
    \\    ascale[gi] = sc;
    \\    rsum[gi] = rs;
    \\  }
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
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ key.m, key.k }, 2, .int8));
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

/// `xq = round(x / as)` per 128-group, with `as = amax/127`, plus the
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

/// Quantize launches, for the memo test.
pub var quantize_calls: u64 = 0;

/// `rows` may exceed x2's row count: the extra rows are the kernel tile's padding.
pub fn quantizeRows(x2: mlx.mlx_array, rows: c_int, k: c_int, s: mlx.mlx_stream) !Quantized {
    quantize_calls += 1;
    const inputs = [_]mlx.mlx_array{x2};
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try quantKernel(), iv, try quantConfig(.{ .m = rows, .k = k, .dt = mlx.mlx_array_dtype(x2) }), s));
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

fn retained(a: mlx.mlx_array) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&r, a));
    return r;
}

fn retainedQ(q: Quantized) !Quantized {
    return .{ .xq = try retained(q.xq), .ascale = try retained(q.ascale), .rsum = try retained(q.rsum) };
}

/// Sibling projections (gate/up, q/k/v/z) read ONE rotated activation, so its
/// quantization is kept for the next call on the same underlying array
/// (keyed like `rht.Registry`'s memo: the shape storage a retained handle pins).
var memo_x: mlx.mlx_array = .{ .ctx = null };
var memo_signs: ?*anyopaque = null;
var memo_q: ?Quantized = null;

var rotq_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// `quantizeRows(H(signs * x2))` in one dispatch (block 1024 only).
pub fn rotateQuantizeRows(x2: mlx.mlx_array, signs: mlx.mlx_array, rows: c_int, k: c_int, s: mlx.mlx_stream) !Quantized {
    quantize_calls += 1;
    if (rotq_kernel == null) {
        const in_names = [_][*:0]const u8{ "x", "signs" };
        const out_names = [_][*:0]const u8{ "xq", "ascale", "rsum" };
        const iv = mlx.mlx_vector_string_new_data(&in_names, in_names.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const kk = mlx.mlx_fast_metal_kernel_new("msv_int8_rotate_quant", iv, ov, ROTQ_SOURCE, "", true, false);
        if (kk.ctx == null) return error.MetalKernelCompileFailed;
        rotq_kernel = kk;
    }
    const kg = @divExact(k, GS);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, k }, 2, .int8));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, kg }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, kg }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(k, 1024) * 128, rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", mlx.mlx_array_dtype(x2)));
    const inputs = [_]mlx.mlx_array{ x2, signs };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, rotq_kernel.?, iv, cfg, s));
    var res: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (res) |r| {
        _ = mlx.mlx_array_free(r);
    };
    for (&res, 0..) |*r, i| try mlx.check(mlx.mlx_vector_array_get(r, outs, i));
    return .{ .xq = res[0], .ascale = res[1], .rsum = res[2] };
}

/// The rotation to fuse into the quantizer: `H(signs * x)` per 1024 block.
pub const Rotation = struct { signs: mlx.mlx_array };

fn quantizedFor(x: mlx.mlx_array, x2: mlx.mlx_array, rot: ?Rotation, rows: c_int, k: c_int, s: mlx.mlx_stream) !Quantized {
    const sig: ?*anyopaque = if (rot) |r| r.signs.ctx else null;
    if (memo_q) |mq| {
        // Identity, not shape: the shape pointer names the array itself, and the memo
        // retains `memo_x`, so its address cannot be reused by another activation.
        if (mlx.mlx_array_shape(memo_x) == mlx.mlx_array_shape(x) and memo_signs == sig) return retainedQ(mq);
    }
    var q = if (rot) |r| try rotateQuantizeRows(x2, r.signs, rows, k, s) else try quantizeRows(x2, rows, k, s);
    errdefer q.deinit();
    if (memo_q) |*mq| mq.deinit();
    if (memo_x.ctx != null) _ = mlx.mlx_array_free(memo_x);
    memo_q = null;
    memo_x = try retained(x);
    memo_signs = sig;
    memo_q = try retainedQ(q);
    return q;
}

/// The kernel reads scales and biases group-major; the transposed copies are
/// derived constants of the weight, made once and kept.
const Derived = struct { scT: mlx.mlx_array, biT: mlx.mlx_array };

/// The route's per-weight constants, owned by the model whose weights they
/// derive from: the owner frees them in its `deinit` (on the inference thread),
/// so a key can never outlive the weight it names.
pub const Cache = struct {
    map: std.AutoHashMapUnmanaged(usize, Derived) = .{},

    pub fn deinit(self: *Cache) void {
        var it = self.map.valueIterator();
        while (it.next()) |d| {
            _ = mlx.mlx_array_free(d.scT);
            _ = mlx.mlx_array_free(d.biT);
        }
        self.map.deinit(std.heap.c_allocator);
        self.* = .{};
        dropMemo();
    }
};

/// Frees the memoized quantization (it retains the last activation).
pub fn dropMemo() void {
    if (memo_q) |*mq| mq.deinit();
    if (memo_x.ctx != null) _ = mlx.mlx_array_free(memo_x);
    memo_q = null;
    memo_x = .{ .ctx = null };
    memo_signs = null;
}

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

fn derivedCached(cache: *Cache, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bneg: bool, s: mlx.mlx_stream) !Derived {
    const key = @intFromPtr(w.ctx);
    if (cache.map.get(key)) |c| return c;
    const scT = try transposed(sc, s);
    // The factored epilogue never reads the biases of a bias == -scale pack.
    const biT = if (bneg) try retained(scT) else try transposed(bi, s);
    const d = Derived{ .scT = scT, .biT = biT };
    try cache.map.put(std.heap.c_allocator, key, d);
    return d;
}

var kernel_cache: ?mlx.mlx_fast_metal_kernel = null;
const CfgKey = struct { n: c_int, mpad: c_int, dt: mlx.mlx_dtype, st: mlx.mlx_dtype, neg: bool };
var cfg_cache: std.AutoHashMapUnmanaged(CfgKey, mlx.mlx_fast_metal_kernel_config) = .{};
var engaged_logged = false;

fn kernel() !mlx.mlx_fast_metal_kernel {
    if (kernel_cache) |k| return k;
    const in_names = [_][*:0]const u8{ "xq", "w", "scales", "biases", "ascale", "rsum" };
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
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 128 * @divTrunc(key.n + 127, 128), @divExact(key.mpad, 32), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", key.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "MPAD", key.mpad));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "NEG", key.neg));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "ST", key.st));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "FULL", @rem(key.n, 128) == 0));
    try cfg_cache.put(std.heap.c_allocator, key, cfg);
    return cfg;
}

/// `x @ w.T` with the activation quantized to int8 per 128-group, or null when
/// the shape/dtype is outside the route (caller keeps whatever it would do).
/// LOSSY BY CONSTRUCTION — see the module comment.
pub fn qmm(
    cache: *Cache,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    /// biases == -scales (Prism's ternary codec): the epilogue drops the bias.
    bneg: bool,
    s: mlx.mlx_stream,
) !?mlx.mlx_array {
    return qmmImpl(cache, x, null, w, sc, bi, bits, group_size, bneg, s);
}

/// `qmm(H(signs * x), ...)` with the rotation fused into the quantizer; null
/// (caller rotates and takes `qmm`) outside it, e.g. a block other than 1024.
pub fn qmmRotated(
    cache: *Cache,
    x: mlx.mlx_array,
    signs: mlx.mlx_array,
    block: c_int,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    bneg: bool,
    s: mlx.mlx_stream,
) !?mlx.mlx_array {
    const xs = mlx.getShape(x);
    if (block != 1024 or xs.len == 0 or @rem(xs[xs.len - 1], 1024) != 0) return null;
    return qmmImpl(cache, x, .{ .signs = signs }, w, sc, bi, bits, group_size, bneg, s);
}

fn qmmImpl(
    cache: *Cache,
    x: mlx.mlx_array,
    rot: ?Rotation,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    bneg: bool,
    s: mlx.mlx_stream,
) !?mlx.mlx_array {
    if (bits != 2 or group_size != GS or bi.ctx == null or !mlx.streamIsGpu(s)) return null;
    if (!@import("transformer.zig").naxAvailable()) return null;
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

    const mpad = @divTrunc(m + 31, 32) * 32;
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ m, k }, 2, s));
    var q = try quantizedFor(x, x2, rot, mpad, k, s);
    defer q.deinit();
    const d = try derivedCached(cache, w, sc, bi, bneg, s);

    if (!engaged_logged) {
        engaged_logged = true;
        log.info("[int8-prefill] engaged (LOSSY: activations quantized to int8): first call M={d} N={d} K={d}\n", .{ m, n, k });
    }
    const inputs = [_]mlx.mlx_array{ q.xq, w, d.scT, d.biT, q.ascale, q.rsum };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try kernel(), iv, try configFor(.{ .n = n, .mpad = mpad, .dt = dt, .st = mlx.mlx_array_dtype(d.scT), .neg = bneg }), s));
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
/// The route runs only on a NAX GPU; elsewhere it declines by design.
fn requireNax() !void {
    if (!@import("transformer.zig").naxAvailable()) return error.SkipZigTest;
}

/// A kernel failure is its test's: name it and drop its latch, or the next test inherits it.
fn dropOwnLatch() void {
    var buf: [512]u8 = undefined;
    if (mlx.takeError(&buf)) |msg| std.debug.print("[qmm_int8] mlx: {s}\n", .{msg});
}

test "qmm_int8: error stays within a small multiple of stock at prompt width" {
    try requireNax();
    try expectWithinBar(512, 1536, 128);
}

test "qmm_int8: a failing kernel test leaves no latch for the next test" {
    try requireNax();
    mlx.armLatchingFaultForTest(1);
    defer mlx.armLatchingFaultForTest(0);
    try std.testing.expectError(error.MlxError, expectWithinBar(512, 1536, 128));
    try std.testing.expect(!mlx.errorPending());
}

test "qmm_int8: a width that is not a whole tile keeps the bar (GDN b/a projections)" {
    try requireNax();
    try expectWithinBar(48, 1024, 96);
}

test "qmm_int8: a row count that is not a whole tile keeps the bar" {
    try requireNax();
    try expectWithinBar(256, 1024, 100);
}

fn expectWithinBar(n: c_int, k: c_int, m: c_int) !void {
    errdefer dropOwnLatch();
    const s = mlx.gpuStream();
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

    var cache: Cache = .{};
    defer cache.deinit();
    const got = (try qmm(&cache, x, wq, sc, bi, 2, 128, true, s)) orelse {
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

test "qmm_int8: sibling projections of one activation quantize it once" {
    try requireNax();
    errdefer dropOwnLatch();
    const s = mlx.gpuStream();
    const n: c_int = 128;
    const k: c_int = 256;
    const m: c_int = 64;
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_random_uniform(&x, mlx.mlx_array_new_float(-1.0), mlx.mlx_array_new_float(1.0), &[_]c_int{ m, k }, 2, .float16, .{ .ctx = null }, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_zeros(&w, &[_]c_int{ n, @divExact(k, 16) }, 2, .uint32, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_ones(&sc, &[_]c_int{ n, @divExact(k, 128) }, 2, .float16, s));
    var cache: Cache = .{};
    defer cache.deinit();
    const before = quantize_calls;
    for (0..2) |_| {
        const y = (try qmm(&cache, x, w, sc, sc, 2, 128, false, s)) orelse return error.RouteDeclined;
        _ = mlx.mlx_array_free(y);
    }
    try std.testing.expectEqual(before + 1, quantize_calls);
}

test "qmm_int8: a CPU-stream call declines" {
    const s = mlx.mlx_default_cpu_stream_new();
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_zeros(&x, &[_]c_int{ MIN_ROWS, 256 }, 2, .float16, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_zeros(&w, &[_]c_int{ 128, 16 }, 2, .uint32, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_ones(&sc, &[_]c_int{ 128, 2 }, 2, .float16, s));
    var cache: Cache = .{};
    defer cache.deinit();
    try std.testing.expectEqual(@as(?mlx.mlx_array, null), try qmm(&cache, x, w, sc, sc, 2, 128, false, s));
}

test "qmm_int8: declines below its row floor" {
    const s = mlx.gpuStream();
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_zeros(&x, &[_]c_int{ MIN_ROWS - 1, 256 }, 2, .float16, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_zeros(&w, &[_]c_int{ 128, 16 }, 2, .uint32, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_ones(&sc, &[_]c_int{ 128, 2 }, 2, .float16, s));
    var cache: Cache = .{};
    defer cache.deinit();
    try std.testing.expectEqual(@as(?mlx.mlx_array, null), try qmm(&cache, x, w, sc, sc, 2, 128, false, s));
}

test "qmm_int8: the fused rotation equals rotating first, bit for bit" {
    try requireNax();
    errdefer dropOwnLatch();
    const s = mlx.gpuStream();
    const n: c_int = 256;
    const k: c_int = 2048;
    const m: c_int = 100;
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_random_normal(&x, &[_]c_int{ m, k }, 2, .float16, 0.0, 1.0, .{ .ctx = null }, s));
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_random_uniform(&u, mlx.mlx_array_new_float(0.0), mlx.mlx_array_new_float(1.0), &[_]c_int{k}, 1, .float32, .{ .ctx = null }, s));
    var half = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(half);
    try mlx.check(mlx.mlx_greater(&half, u, mlx.mlx_array_new_float(0.5), s));
    var signs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(signs);
    try mlx.check(mlx.mlx_where(&signs, half, mlx.mlx_array_new_float(1.0), mlx.mlx_array_new_float(-1.0), s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_random_bits(&w, &[_]c_int{ n, @divExact(k, 16) }, 2, 4, .{ .ctx = null }, s));
    const wu = w;
    try std.testing.expectEqual(mlx.mlx_dtype.uint32, mlx.mlx_array_dtype(wu));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_random_uniform(&sc, mlx.mlx_array_new_float(0.005), mlx.mlx_array_new_float(0.02), &[_]c_int{ n, @divExact(k, 128) }, 2, .float16, .{ .ctx = null }, s));
    var bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi);
    try mlx.check(mlx.mlx_negative(&bi, sc, s));

    const xr = try @import("rht.zig").transform(x, signs, 1024, false, s);
    defer _ = mlx.mlx_array_free(xr);
    var cache: Cache = .{};
    defer cache.deinit();
    const want = (try qmm(&cache, xr, wu, sc, bi, 2, 128, true, s)) orelse return error.RouteDeclined;
    defer _ = mlx.mlx_array_free(want);
    const got = (try qmmRotated(&cache, x, signs, 1024, wu, sc, bi, 2, 128, true, s)) orelse return error.RouteDeclined;
    defer _ = mlx.mlx_array_free(got);
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, want, got, false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var ok: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&ok, eq));
    try std.testing.expect(ok);
}

// The bar: a weight's constants come from the cache that owns them, so a new
// model (a fresh cache) never reads another model's scales off a reused handle.
test "qmm_int8: a fresh cache derives its own constants for a reused weight handle" {
    try requireNax();
    errdefer dropOwnLatch();
    const s = mlx.gpuStream();
    const n: c_int = 256;
    const k: c_int = 512;
    const m: c_int = 64;
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_random_uniform(&x, mlx.mlx_array_new_float(-1.0), mlx.mlx_array_new_float(1.0), &[_]c_int{ m, k }, 2, .float16, .{ .ctx = null }, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_random_bits(&w, &[_]c_int{ n, @divExact(k, 16) }, 2, 4, .{ .ctx = null }, s));
    var sc1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc1);
    try mlx.check(mlx.mlx_ones(&sc1, &[_]c_int{ n, @divExact(k, 128) }, 2, .float16, s));
    var sc2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc2);
    var sc_twice = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc_twice);
    try mlx.check(mlx.mlx_add(&sc_twice, sc1, sc1, s));
    try mlx.check(mlx.mlx_add(&sc2, sc_twice, sc1, s));
    var y: [2]mlx.mlx_array = undefined;
    for ([_]mlx.mlx_array{ sc1, sc2 }, 0..) |sc, i| {
        var cache: Cache = .{};
        defer cache.deinit();
        y[i] = (try qmm(&cache, x, w, sc, sc, 2, 128, false, s)) orelse return error.RouteDeclined;
    }
    defer for (y) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var y0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y0);
    var y1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y1);
    try mlx.check(mlx.mlx_astype(&y0, y[0], .float32, s));
    try mlx.check(mlx.mlx_astype(&y1, y[1], .float32, s));
    try mlx.check(mlx.mlx_array_eval(y0));
    try mlx.check(mlx.mlx_array_eval(y1));
    const v0 = mlx.mlx_array_data_float32(y0).?;
    const v1 = mlx.mlx_array_data_float32(y1).?;
    // Scales tripled, so every output triples (to f16 rounding).
    for (0..@intCast(m * n)) |i| try std.testing.expectApproxEqAbs(3.0 * v0[i], v1[i], 0.02 * @abs(v1[i]) + 0.05);
}
