//! Single-row (decode) GEMV for MLX affine 2-bit, group 128, bf16 activations.
//! Same layout as MLX's qmv_fast (2 simdgroups x 4 rows, 16 codes per lane)
//! but the decode is a half2 magic-number trick: one AND+OR turns codes i and
//! i+8 of a word into exact half2 integers (the OR plants 2^(10-2j) so code j
//! sits at unit ulp), so a code pair costs a sub + fma instead of and+cvt+fma
//! per code. qmv_fast is ALU co-bound at 2 bits; this runs at the load floor.
//! The 8-term per-word partial accumulates in half2 with x pre-scaled by 2^-6
//! (|x| up to ~4e6 stays finite); everything else is f32.
const std = @import("std");
const mlx = @import("mlx.zig");

const HEADER =
    \\inline float h2dot(uint u, thread const half2* xh) {
    \\  half2 acc = half2(0);
    \\  uint u6 = u >> 6;
    \\  acc = fma(as_type<half2>((u  & 0x00030003u) | 0x64006400u) - half2(1024.0h), xh[0], acc);
    \\  acc = fma(as_type<half2>((u  & 0x000C000Cu) | 0x5C005C00u) - half2(256.0h), xh[1], acc);
    \\  acc = fma(as_type<half2>((u  & 0x00300030u) | 0x54005400u) - half2(64.0h), xh[2], acc);
    \\  acc = fma(as_type<half2>((u  & 0x00C000C0u) | 0x4C004C00u) - half2(16.0h), xh[3], acc);
    \\  acc = fma(as_type<half2>((u  & 0x03000300u) | 0x44004400u) - half2(4.0h), xh[4], acc);
    \\  acc = fma(as_type<half2>((u6 & 0x00300030u) | 0x54005400u) - half2(64.0h), xh[5], acc);
    \\  acc = fma(as_type<half2>((u6 & 0x00C000C0u) | 0x4C004C00u) - half2(16.0h), xh[6], acc);
    \\  acc = fma(as_type<half2>((u6 & 0x03000300u) | 0x44004400u) - half2(4.0h), xh[7], acc);
    \\  return float(acc.x) + float(acc.y);
    \\}
;

// The clamp is load-bearing: without it the compiler rewrites half(a*c) as
// half(a)*c and an |x| above 65504 turns inf, then NaN.
const SOURCE =
    \\const int K = x_shape[x_ndim - 1];
    \\const int KW = K / 16, KG = K / 128;
    \\uint lane = thread_index_in_simdgroup;
    \\int row0 = (threadgroup_position_in_grid.y * 2 + simdgroup_index_in_threadgroup) * 4;
    \\const device T* xp = x + lane * 16;
    \\const device uint* wq = w + row0 * KW + lane;
    \\const device T* sp = scales + row0 * KG + lane / 8;
    \\const device T* bp = biases + row0 * KG + lane / 8;
    \\float res[4] = {0.f, 0.f, 0.f, 0.f};
    \\for (int k = 0; k < K; k += 512) {
    \\  half2 xh[8];
    \\  float sm = 0.f;
    \\  for (int i = 0; i < 8; ++i) {
    \\    float a = xp[i], b = xp[i + 8];
    \\    sm += a + b;
    \\    xh[i] = half2(half(clamp(a * 0.015625f, -65504.0f, 65504.0f)), half(clamp(b * 0.015625f, -65504.0f, 65504.0f)));
    \\  }
    \\  sm *= 0.015625f;
    \\  for (int r = 0; r < 4; ++r) res[r] += float(sp[r * KG]) * h2dot(wq[r * KW], xh) + float(bp[r * KG]) * sm;
    \\  wq += 32; sp += 4; bp += 4; xp += 512;
    \\}
    \\for (int r = 0; r < 4; ++r) {
    \\  float v = simd_sum(res[r]) * 64.0f;
    \\  if (lane == 0) y[row0 + r] = static_cast<T>(v);
    \\}
;

/// Verify widths (M rows): each weight word is decoded once into 8 half2 and
/// reused by all M rows; x is staged per threadgroup as 2^-6-prescaled half2
/// pairs (x_i, x_i+8), double-buffered. BNEG (biases == -scales, the Prism
/// ternary layout) decodes q-1 directly and drops the bias term.
const ROWS_HEADER =
    \\inline void h2dec(uint u, thread half2* q, half o) {
    \\  uint u6 = u >> 6;
    \\  q[0] = as_type<half2>((u  & 0x00030003u) | 0x64006400u) - half2(1024.0h + o);
    \\  q[1] = as_type<half2>((u  & 0x000C000Cu) | 0x5C005C00u) - half2(256.0h + o);
    \\  q[2] = as_type<half2>((u  & 0x00300030u) | 0x54005400u) - half2(64.0h + o);
    \\  q[3] = as_type<half2>((u  & 0x00C000C0u) | 0x4C004C00u) - half2(16.0h + o);
    \\  q[4] = as_type<half2>((u  & 0x03000300u) | 0x44004400u) - half2(4.0h + o);
    \\  q[5] = as_type<half2>((u6 & 0x00300030u) | 0x54005400u) - half2(64.0h + o);
    \\  q[6] = as_type<half2>((u6 & 0x00C000C0u) | 0x4C004C00u) - half2(16.0h + o);
    \\  q[7] = as_type<half2>((u6 & 0x03000300u) | 0x44004400u) - half2(4.0h + o);
    \\}
;

const ROWS_SOURCE =
    \\constexpr int R = 4, G = 8, NL = 32, NT = 32 * G;
    \\threadgroup half2 xs[2][M][NL][8];
    \\threadgroup float sms[2][M][NL];
    \\const int K = x_shape[x_ndim - 1];
    \\const int N = w_shape[0];
    \\const int KW = K / 16, KG = K / 128;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sgi = simdgroup_index_in_threadgroup;
    \\uint tid = sgi * 32 + lane;
    \\int grow = (threadgroup_position_in_grid.y * G + sgi) * R;
    \\const device uint* wq = w + grow * KW + lane;
    \\const device T* sp = scales + grow * KG + lane / 8;
    \\const device T* bp = biases + grow * KG + lane / 8;
    \\auto stage = [&](int k, int buf) {
    \\  for (int t = tid; t < M * NL; t += NT) {
    \\    int m = t / NL, l = t % NL;
    \\    const device uint4* xr = (const device uint4*)(x + m * K + k + l * 16);
    \\    uint4 A = xr[0], B = xr[1];
    \\    float sm = 0;
    \\    for (int i = 0; i < 8; ++i) {
    \\      uint a = A[i / 2], b = B[i / 2];
    \\      float fa = as_type<float>((i & 1) ? (a & 0xFFFF0000u) : (a << 16));
    \\      float fb = as_type<float>((i & 1) ? (b & 0xFFFF0000u) : (b << 16));
    \\      sm += fa + fb;
    \\      xs[buf][m][l][i] = half2(half(clamp(fa * 0.015625f, -65504.0f, 65504.0f)), half(clamp(fb * 0.015625f, -65504.0f, 65504.0f)));
    \\    }
    \\    if (!BNEG) sms[buf][m][l] = sm * 0.015625f;
    \\  }
    \\};
    \\float res[R][M];
    \\for (int r = 0; r < R; ++r) for (int m = 0; m < M; ++m) res[r][m] = 0;
    \\stage(0, 0);
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\int buf = 0;
    \\const half off = BNEG ? 1.0h : 0.0h;
    \\for (int k = 0; k < K; k += 512) {
    \\  uint wv[R]; float sc[R], bs[R];
    \\  for (int r = 0; r < R; ++r) { wv[r] = wq[r * KW]; sc[r] = float(sp[r * KG]); bs[r] = BNEG ? 0.f : float(bp[r * KG]); }
    \\  if (k + 512 < K) stage(k + 512, buf ^ 1);
    \\  half2 q[R][8];
    \\  for (int r = 0; r < R; ++r) h2dec(wv[r], q[r], off);
    \\  for (int m = 0; m < M; ++m) {
    \\    const threadgroup float4* xv = (const threadgroup float4*)&xs[buf][m][lane][0];
    \\    float4 v0 = xv[0], v1 = xv[1];
    \\    half2 xh[8] = {as_type<half2>(v0.x), as_type<half2>(v0.y), as_type<half2>(v0.z), as_type<half2>(v0.w),
    \\                   as_type<half2>(v1.x), as_type<half2>(v1.y), as_type<half2>(v1.z), as_type<half2>(v1.w)};
    \\    float sm = BNEG ? 0.f : sms[buf][m][lane];
    \\    for (int r = 0; r < R; ++r) {
    \\      half2 acc = half2(0);
    \\      for (int j = 0; j < 8; ++j) acc = fma(q[r][j], xh[j], acc);
    \\      res[r][m] += sc[r] * (float(acc.x) + float(acc.y)) + bs[r] * sm;
    \\    }
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  buf ^= 1; wq += 32; sp += 4; bp += 4;
    \\}
    \\for (int r = 0; r < R; ++r) for (int m = 0; m < M; ++m) {
    \\  float v = simd_sum(res[r][m]) * 64.0f;
    \\  if (lane == 0) y[m * N + grow + r] = static_cast<T>(v);
    \\}
;

var rows_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// Launch configs keyed on the FULL shape (the output shape is baked in, and
/// a rebuild per call is CPU tax on ~450 matmuls a token). A model has ~11
/// (n, k) shapes x 5 verify widths, so the set is bounded and never evicted.
const RowsKey = struct { n: c_int, k: c_int, m: c_int, bneg: bool };
var rows_cfg_cache: std.AutoHashMapUnmanaged(RowsKey, mlx.mlx_fast_metal_kernel_config) = .{};

fn rowsConfig(n: c_int, k: c_int, m: c_int, bneg: bool) !mlx.mlx_fast_metal_kernel_config {
    const key = RowsKey{ .n = n, .k = k, .m = m, .bneg = bneg };
    if (rows_cfg_cache.get(key)) |c| return c;
    const config = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const out_shape = [_]c_int{ m, n };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 256, @divExact(n, 32), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "M", m));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "BNEG", bneg));
    try rows_cfg_cache.put(std.heap.c_allocator, key, config);
    return config;
}

/// `x @ w.T` for 2..6 activation rows (spec-verify widths), or null. `bneg`
/// asserts the checkpoint's biases are exactly -scales.
pub fn qmvRows(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, bneg: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (bits != 2 or group_size != 128 or bi.ctx == null or !enabled()) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    if (xs.len == 0 or xs.len > 8 or ws.len != 2) return null;
    var m: c_int = 1;
    for (xs[0 .. xs.len - 1]) |d| m *= d;
    const k = xs[xs.len - 1];
    const n = ws[0];
    if (m < 2 or m > 6 or @rem(k, 512) != 0 or ws[1] * 16 != k or @rem(n, 32) != 0) return null;
    if (rows_kernel == null) {
        const in_names = [_][*:0]const u8{ "x", "w", "scales", "biases" };
        const out_names = [_][*:0]const u8{"y"};
        const in_vec = mlx.mlx_vector_string_new_data(&in_names, in_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const kk = mlx.mlx_fast_metal_kernel_new("msv_qmv2_rows", in_vec, out_vec, ROWS_SOURCE, ROWS_HEADER, true, false);
        if (kk.ctx == null) return error.MetalKernelCompileFailed;
        rows_kernel = kk;
    }
    const inputs = [_]mlx.mlx_array{ x, w, sc, bi };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, rows_kernel.?, in_vec, try rowsConfig(n, k, m, bneg), s));
    var y = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
    var out_shape: [8]c_int = undefined;
    @memcpy(out_shape[0..xs.len], xs);
    out_shape[xs.len - 1] = n;
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&r, y, &out_shape, xs.len, s));
    return r;
}

var kernel_cache: ?mlx.mlx_fast_metal_kernel = null;

const CfgKey = struct { n: c_int, k: c_int };
var cfg_cache: std.AutoHashMapUnmanaged(CfgKey, mlx.mlx_fast_metal_kernel_config) = .{};

var env_enabled: ?bool = null;
fn enabled() bool {
    if (env_enabled) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_QMV_H2");
    env_enabled = raw == null or raw.?[0] != '0';
    return env_enabled.?;
}

fn kernel() !mlx.mlx_fast_metal_kernel {
    if (kernel_cache) |k| return k;
    const in_names = [_][*:0]const u8{ "x", "w", "scales", "biases" };
    const out_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&in_names, in_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new("msv_qmv2_h2", in_vec, out_vec, SOURCE, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel_cache = k;
    return k;
}

fn configFor(n: c_int, k: c_int) !mlx.mlx_fast_metal_kernel_config {
    const key = CfgKey{ .n = n, .k = k };
    if (cfg_cache.get(key)) |c| return c;
    const config = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const out_shape = [_]c_int{n};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, 1, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 64, @divExact(n, 8), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16));
    try cfg_cache.put(std.heap.c_allocator, key, config);
    return config;
}

/// `x @ w.T` for one activation row, or null when the call is outside the
/// kernel (caller keeps stock qmm).
pub fn qmv(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    const nk = eligible(x, w, sc, bi, bits, group_size) orelse return null;
    return try launch(try kernel(), &.{ x, w, sc, bi }, x, nk[0], nk[1], s);
}

fn eligible(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32) ?[2]c_int {
    if (bits != 2 or group_size != 128 or bi.ctx == null or !enabled()) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    if (xs.len == 0 or xs.len > 8 or ws.len != 2) return null;
    var rows: c_int = 1;
    for (xs[0 .. xs.len - 1]) |d| rows *= d;
    const k = xs[xs.len - 1];
    const n = ws[0];
    if (rows != 1 or @rem(k, 512) != 0 or ws[1] * 16 != k or @rem(n, 8) != 0) return null;
    return .{ n, k };
}

fn launch(k: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, x: mlx.mlx_array, n: c_int, kdim: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, k, in_vec, try configFor(n, kdim), s));
    var y = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
    const xs = mlx.getShape(x);
    var out_shape: [8]c_int = undefined;
    @memcpy(out_shape[0..xs.len], xs);
    out_shape[xs.len - 1] = n;
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&r, y, &out_shape, xs.len, s));
    return r;
}

test "qmv2: half2 decode matches stock quantized_matmul, outliers included" {
    const s = mlx.gpuStream();
    const n: c_int = 1024;
    const k: c_int = 1536;
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    const wv = try std.testing.allocator.alloc(f32, @intCast(n * k));
    defer std.testing.allocator.free(wv);
    for (wv) |*e| e.* = rnd.floatNorm(f32) * 0.02;
    var xv: [@intCast(k)]f32 = undefined;
    for (&xv, 0..) |*e, i| e.* = if (i % 97 == 0) 3.0e4 * rnd.floatNorm(f32) else rnd.floatNorm(f32);

    const w32 = mlx.mlx_array_new_data(wv.ptr, &[_]c_int{ n, k }, 2, .float32);
    defer _ = mlx.mlx_array_free(w32);
    var wb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wb);
    try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, s));
    var q = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(q);
    try mlx.check(mlx.mlx_quantize(&q, wb, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(2), "affine", .{}, s));
    var wq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    var bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi);
    try mlx.check(mlx.mlx_vector_array_get(&wq, q, 0));
    try mlx.check(mlx.mlx_vector_array_get(&sc, q, 1));
    try mlx.check(mlx.mlx_vector_array_get(&bi, q, 2));

    const x32 = mlx.mlx_array_new_data(&xv, &[_]c_int{ 1, 1, k }, 3, .float32);
    defer _ = mlx.mlx_array_free(x32);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    try mlx.check(mlx.mlx_quantized_matmul(&want, x, wq, sc, bi, true, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(2), "affine", s));
    const got = (try qmv(x, wq, sc, bi, 2, 128, s)).?;
    defer _ = mlx.mlx_array_free(got);
    try std.testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));

    var wf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wf);
    var gf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gf);
    try mlx.check(mlx.mlx_astype(&wf, want, .float32, s));
    try mlx.check(mlx.mlx_astype(&gf, got, .float32, s));
    try mlx.check(mlx.mlx_array_eval(wf));
    try mlx.check(mlx.mlx_array_eval(gf));
    const a = mlx.mlx_array_data_float32(wf).?;
    const b = mlx.mlx_array_data_float32(gf).?;
    var peak: f32 = 0;
    var worst: f32 = 0;
    for (0..@intCast(n)) |i| {
        try std.testing.expect(std.math.isFinite(b[i]));
        peak = @max(peak, @abs(a[i]));
        worst = @max(worst, @abs(a[i] - b[i]));
    }
    try std.testing.expect(worst <= 0.015 * peak); // ~2 bf16 ulp at the peak
}


test "qmv2: verify-width rows match stock quantized_matmul (generic affine and bias == -scale)" {
    const s = mlx.gpuStream();
    const n: c_int = 512;
    const k: c_int = 1024;
    var prng = std.Random.DefaultPrng.init(9);
    const rnd = prng.random();
    const wv = try std.testing.allocator.alloc(f32, @intCast(n * k));
    defer std.testing.allocator.free(wv);
    for (wv) |*e| e.* = rnd.floatNorm(f32) * 0.02;
    const w32 = mlx.mlx_array_new_data(wv.ptr, &[_]c_int{ n, k }, 2, .float32);
    defer _ = mlx.mlx_array_free(w32);
    var wb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wb);
    try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, s));
    var q = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(q);
    try mlx.check(mlx.mlx_quantize(&q, wb, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(2), "affine", .{}, s));
    var wq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    var bi_affine = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi_affine);
    try mlx.check(mlx.mlx_vector_array_get(&wq, q, 0));
    try mlx.check(mlx.mlx_vector_array_get(&sc, q, 1));
    try mlx.check(mlx.mlx_vector_array_get(&bi_affine, q, 2));
    var bi_neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi_neg);
    try mlx.check(mlx.mlx_negative(&bi_neg, sc, s));

    for ([_]bool{ false, true }) |bneg| {
        const bi = if (bneg) bi_neg else bi_affine;
        var m: c_int = 2;
        while (m <= 6) : (m += 1) {
            const xv = try std.testing.allocator.alloc(f32, @intCast(m * k));
            defer std.testing.allocator.free(xv);
            for (xv, 0..) |*e, i| e.* = if (i % 89 == 0) 2.0e4 * rnd.floatNorm(f32) else rnd.floatNorm(f32);
            const x32 = mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ 1, m, k }, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
            var want = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(want);
            try mlx.check(mlx.mlx_quantized_matmul(&want, x, wq, sc, bi, true, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(2), "affine", s));
            const got = (try qmvRows(x, wq, sc, bi, 2, 128, bneg, s)).?;
            defer _ = mlx.mlx_array_free(got);
            try std.testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
            var wf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wf);
            var gf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gf);
            try mlx.check(mlx.mlx_astype(&wf, want, .float32, s));
            try mlx.check(mlx.mlx_astype(&gf, got, .float32, s));
            try mlx.check(mlx.mlx_array_eval(wf));
            try mlx.check(mlx.mlx_array_eval(gf));
            const a = mlx.mlx_array_data_float32(wf).?;
            const b = mlx.mlx_array_data_float32(gf).?;
            for (0..@intCast(m)) |row| {
                var peak: f32 = 0;
                var worst: f32 = 0;
                for (0..@intCast(n)) |c| {
                    const i = row * @as(usize, @intCast(n)) + c;
                    try std.testing.expect(std.math.isFinite(b[i]));
                    peak = @max(peak, @abs(a[i]));
                    worst = @max(worst, @abs(a[i] - b[i]));
                }
                try std.testing.expect(worst <= 0.015 * peak);
            }
        }
    }
}

test "qmv2: verify-width rows vs stock qmm per M (MLX_SERVE_QMV2_BENCH=1)" {
    if (std.c.getenv("MLX_SERVE_QMV2_BENCH") == null) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const n: c_int = 17408;
    const k: c_int = 5120;
    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();
    const wv = try std.testing.allocator.alloc(f32, @intCast(n * k));
    defer std.testing.allocator.free(wv);
    for (wv) |*e| e.* = rnd.floatNorm(f32) * 0.02;
    const w32 = mlx.mlx_array_new_data(wv.ptr, &[_]c_int{ n, k }, 2, .float32);
    defer _ = mlx.mlx_array_free(w32);
    var wb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wb);
    try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, s));
    var q = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(q);
    try mlx.check(mlx.mlx_quantize(&q, wb, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(2), "affine", .{}, s));
    var wq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    var bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi);
    try mlx.check(mlx.mlx_vector_array_get(&wq, q, 0));
    try mlx.check(mlx.mlx_vector_array_get(&sc, q, 1));
    try mlx.check(mlx.mlx_vector_array_get(&bi, q, 2));
    var m: c_int = 2;
    while (m <= 6) : (m += 1) {
        const xv = try std.testing.allocator.alloc(f32, @intCast(m * k));
        defer std.testing.allocator.free(xv);
        for (xv) |*e| e.* = rnd.floatNorm(f32);
        const x32 = mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ 1, m, k }, 3, .float32);
        defer _ = mlx.mlx_array_free(x32);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(x));
        var us: [2]f64 = .{ 0, 0 };
        for (0..2) |arm| {
            // 32 launches per eval: one launch per eval measures the barrier.
            for (0..5) |rep| {
                const reps: usize = 32;
                const io = std.Io.Threaded.global_single_threaded.io();
                const t0 = std.Io.Timestamp.now(io, .boot);
                const vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(vec);
                for (0..reps) |_| {
                    var y = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(y);
                    if (arm == 0) {
                        try mlx.check(mlx.mlx_quantized_matmul(&y, x, wq, sc, bi, true, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(2), "affine", s));
                    } else y = (try qmvRows(x, wq, sc, bi, 2, 128, false, s)).?;
                    try mlx.check(mlx.mlx_vector_array_append_value(vec, y));
                }
                try mlx.check(mlx.mlx_eval(vec));
                const ns: u64 = @intCast(t0.untilNow(io, .boot).nanoseconds);
                const el = @as(f64, @floatFromInt(ns)) / 1000.0 / @as(f64, @floatFromInt(reps));
                if (rep == 0 or el < us[arm]) us[arm] = el;
            }
        }
        std.debug.print("qmv2 rows M={d}: stock {d:.1} us  h2 {d:.1} us  ({d:.2}x)\n", .{ m, us[0], us[1], us[0] / us[1] });
    }
}
