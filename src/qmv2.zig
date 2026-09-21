//! 2-bit, group-128 affine GEMV for decode (one activation row), bf16 or f16.
//! Same layout as MLX's qmv_fast (2 simdgroups x 4 rows, 16 codes per lane)
//! but the codes decode with a half2 magic-number trick: one AND+OR turns
//! codes j and j+8 of a word into exact half2 integers (the OR plants
//! 2^(10-2j) so the code sits at unit ulp). Products and sums are f32 from
//! exact inputs, so the error is stock's; qmv_fast is ALU co-bound at 2 bits
//! and this is not (1.45x on the 5120x17408 MLP shapes, M4 Max).
const std = @import("std");
const mlx = @import("mlx.zig");

/// Codes (j, j+8) of `u` as exact half2 integers, minus `o`.
const DECODE =
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
    \\inline float f2dot(thread const half2* q, thread const float2* xf) {
    \\  float2 acc = float2(0);
    \\  for (int j = 0; j < 8; ++j) acc = fma(float2(q[j]), xf[j], acc);
    \\  return acc.x + acc.y;
    \\}
;

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
    \\  float2 xf[8];
    \\  float sm = 0.f;
    \\  for (int i = 0; i < 8; ++i) {
    \\    xf[i] = float2(float(xp[i]), float(xp[i + 8]));
    \\    sm += xf[i].x + xf[i].y;
    \\  }
    \\  for (int r = 0; r < 4; ++r) {
    \\    half2 q[8];
    \\    h2dec(wq[r * KW], q, 0.0h);
    \\    res[r] += float(sp[r * KG]) * f2dot(q, xf) + float(bp[r * KG]) * sm;
    \\  }
    \\  wq += 32; sp += 4; bp += 4; xp += 512;
    \\}
    \\for (int r = 0; r < 4; ++r) {
    \\  float v = simd_sum(res[r]);
    \\  if (lane == 0) y[row0 + r] = static_cast<T>(v);
    \\}
;

var kernel_cache: ?mlx.mlx_fast_metal_kernel = null;

const CfgKey = struct { n: c_int, k: c_int, dt: mlx.mlx_dtype };
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
    const k = mlx.mlx_fast_metal_kernel_new("msv_qmv2_h2", in_vec, out_vec, SOURCE, DECODE, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel_cache = k;
    return k;
}

fn configFor(n: c_int, k: c_int, dt: mlx.mlx_dtype) !mlx.mlx_fast_metal_kernel_config {
    const key = CfgKey{ .n = n, .k = k, .dt = dt };
    if (cfg_cache.get(key)) |c| return c;
    const config = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const out_shape = [_]c_int{n};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, 1, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 64, @divExact(n, 8), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
    try cfg_cache.put(std.heap.c_allocator, key, config);
    return config;
}

/// `x @ w.T` for one activation row, or null when the call is outside the
/// kernel (caller keeps stock qmm).
pub fn qmv(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    const nk = eligible(x, w, sc, bi, bits, group_size) orelse return null;
    return try launch(try kernel(), &.{ x, w, sc, bi }, x, nk[0], nk[1], s);
}

fn supportedDtypes(dt: mlx.mlx_dtype, sc: mlx.mlx_array, bi: mlx.mlx_array) bool {
    return (dt == .bfloat16 or dt == .float16) and mlx.mlx_array_dtype(sc) == dt and mlx.mlx_array_dtype(bi) == dt;
}

fn eligible(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32) ?[2]c_int {
    if (bits != 2 or group_size != 128 or bi.ctx == null or !enabled()) return null;
    if (!supportedDtypes(mlx.mlx_array_dtype(x), sc, bi)) return null;
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
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, k, in_vec, try configFor(n, kdim, mlx.mlx_array_dtype(x)), s));
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

/// Verify widths M = 2..3 (MTP depth 2), f16 activations over the Prism
/// ternary layout (biases == -scales, so codes decode to q-1 and the bias term
/// drops). Codes and x stay half2 in registers and widen at the FMA, so the
/// math is f32 FMAs over exact inputs (stock's error) at half the registers.
/// x is read straight from device: it is M x K halves and cache-resident.
const ROWS_SOURCE =
    \\constexpr int R = 4, G = 8;
    \\const int K = x_shape[x_ndim - 1];
    \\const int N = w_shape[0];
    \\const int KW = K / 16, KG = K / 128;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sgi = simdgroup_index_in_threadgroup;
    \\int grow = (threadgroup_position_in_grid.y * G + sgi) * R;
    \\const device uint* wq = w + grow * KW + lane;
    \\const device T* sp = scales + grow * KG + lane / 8;
    \\float res[R][M];
    \\for (int r = 0; r < R; ++r) for (int m = 0; m < M; ++m) res[r][m] = 0;
    \\for (int k = 0; k < K; k += 512) {
    \\  uint wv[R]; float sc[R];
    \\  for (int r = 0; r < R; ++r) { wv[r] = wq[r * KW]; sc[r] = float(sp[r * KG]); }
    \\  half2 q[R][8];
    \\  for (int r = 0; r < R; ++r) h2dec(wv[r], q[r], 1.0h);
    \\  for (int m = 0; m < M; ++m) {
    \\    const device T* xr = x + m * K + k + lane * 16;
    \\    half2 xh[8];
    \\    for (int j = 0; j < 8; ++j) xh[j] = half2(xr[j], xr[j + 8]);
    \\    for (int r = 0; r < R; ++r) {
    \\      float2 acc = float2(q[r][0]) * float2(xh[0]);
    \\      for (int j = 1; j < 8; ++j) acc = fma(float2(q[r][j]), float2(xh[j]), acc);
    \\      res[r][m] += sc[r] * (acc.x + acc.y);
    \\    }
    \\  }
    \\  wq += 32; sp += 4;
    \\}
    \\for (int r = 0; r < R; ++r) for (int m = 0; m < M; ++m) {
    \\  float v = simd_sum(res[r][m]);
    \\  if (lane == 0) y[m * N + grow + r] = static_cast<T>(v);
    \\}
;

var rows_kernel: ?mlx.mlx_fast_metal_kernel = null;

const RowsKey = struct { n: c_int, k: c_int, m: c_int };
var rows_cfg_cache: std.AutoHashMapUnmanaged(RowsKey, mlx.mlx_fast_metal_kernel_config) = .{};

fn rowsConfig(n: c_int, k: c_int, m: c_int) !mlx.mlx_fast_metal_kernel_config {
    const key = RowsKey{ .n = n, .k = k, .m = m };
    if (rows_cfg_cache.get(key)) |c| return c;
    const config = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const out_shape = [_]c_int{ m, n };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, 2, .float16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 256, @divExact(n, 32), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .float16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "M", m));
    try rows_cfg_cache.put(std.heap.c_allocator, key, config);
    return config;
}

/// `x @ w.T` for 2..3 f16 activation rows whose weight biases are exactly
/// -scales (`bneg`), or null (caller keeps stock qmm).
pub fn qmvRows(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, bneg: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!bneg or bits != 2 or group_size != 128 or bi.ctx == null or !enabled()) return null;
    if (mlx.mlx_array_dtype(x) != .float16 or !supportedDtypes(.float16, sc, bi)) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    if (xs.len == 0 or xs.len > 8 or ws.len != 2) return null;
    var m: c_int = 1;
    for (xs[0 .. xs.len - 1]) |d| m *= d;
    const k = xs[xs.len - 1];
    const n = ws[0];
    if (m < 2 or m > 3 or @rem(k, 512) != 0 or ws[1] * 16 != k or @rem(n, 32) != 0) return null;
    if (rows_kernel == null) {
        const in_names = [_][*:0]const u8{ "x", "w", "scales" };
        const out_names = [_][*:0]const u8{"y"};
        const in_vec = mlx.mlx_vector_string_new_data(&in_names, in_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const kk = mlx.mlx_fast_metal_kernel_new("msv_qmv2_rows", in_vec, out_vec, ROWS_SOURCE, DECODE, true, false);
        if (kk.ctx == null) return error.MetalKernelCompileFailed;
        rows_kernel = kk;
    }
    const inputs = [_]mlx.mlx_array{ x, w, sc };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, rows_kernel.?, in_vec, try rowsConfig(n, k, m), s));
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

const RmsMax = struct { rms: f32, max: f32 };

/// Per-row RMS and worst error of `got` against the f32 `truth` [m, n].
fn errVsTruth(got: mlx.mlx_array, truth: []const f32, m: usize, n: usize, s: mlx.mlx_stream) ![]RmsMax {
    var g32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g32);
    try mlx.check(mlx.mlx_astype(&g32, got, .float32, s));
    try mlx.check(mlx.mlx_array_eval(g32));
    const g = mlx.mlx_array_data_float32(g32).?;
    const out = try std.testing.allocator.alloc(RmsMax, m);
    for (0..m) |r| {
        var ss: f64 = 0;
        var mx: f32 = 0;
        for (0..n) |c| {
            const d = g[r * n + c] - truth[r * n + c];
            try std.testing.expect(std.math.isFinite(g[r * n + c]));
            ss += d * d;
            mx = @max(mx, @abs(d));
        }
        out[r] = .{ .rms = @floatCast(@sqrt(ss / @as(f64, @floatFromInt(n)))), .max = mx };
    }
    return out;
}

test "qmv2: no worse than stock quantized_matmul against f32 truth (bf16 + f16, M 1..3, both bias layouts)" {
    // Ternary codes with bias == -scale, as the Hadamard packs ship them, plus
    // a generic affine bias; x carries outliers that stress the half2 range.
    const s = mlx.gpuStream();
    const n: c_int = 1024;
    const k: c_int = 1536;
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    const nu: usize = @intCast(n);
    const ku: usize = @intCast(k);

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
    const sc_f = mlx.mlx_array_new_data(sc32.ptr, &[_]c_int{ n, @divExact(k, 128) }, 2, .float32);
    defer _ = mlx.mlx_array_free(sc_f);

    for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |dt| {
        for ([_]bool{ true, false }) |bneg| {
            var sc = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sc);
            try mlx.check(mlx.mlx_astype(&sc, sc_f, dt, s));
            var bi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bi);
            if (bneg) {
                try mlx.check(mlx.mlx_negative(&bi, sc, s));
            } else {
                const f = mlx.mlx_array_new_float(-1.5);
                defer _ = mlx.mlx_array_free(f);
                var half_s = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(half_s);
                try mlx.check(mlx.mlx_astype(&half_s, f, dt, s));
                try mlx.check(mlx.mlx_multiply(&bi, sc, half_s, s));
            }
            // f32 weights from the SAME dt-rounded scales and biases: the truth.
            var sc_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sc_t);
            var bi_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bi_t);
            try mlx.check(mlx.mlx_astype(&sc_t, sc, .float32, s));
            try mlx.check(mlx.mlx_astype(&bi_t, bi, .float32, s));
            var w_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(w_t);
            try mlx.check(mlx.mlx_dequantize(&w_t, wq, sc_t, bi_t, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(2), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
            var w_tt = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(w_tt);
            try mlx.check(mlx.mlx_transpose(&w_tt, w_t, s));

            var m: c_int = 1;
            while (m <= 3) : (m += 1) {
                const mu: usize = @intCast(m);
                const xv = try std.testing.allocator.alloc(f32, mu * ku);
                defer std.testing.allocator.free(xv);
                for (xv, 0..) |*e, i| e.* = if (i % 97 == 0) 3.0e3 * rnd.floatNorm(f32) else if (i % 13 == 0) 1e-3 * rnd.floatNorm(f32) else rnd.floatNorm(f32);
                const x32 = mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ 1, m, k }, 3, .float32);
                defer _ = mlx.mlx_array_free(x32);
                var x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x);
                try mlx.check(mlx.mlx_astype(&x, x32, dt, s));
                var xt = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(xt);
                try mlx.check(mlx.mlx_astype(&xt, x, .float32, s));
                var truth_a = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(truth_a);
                try mlx.check(mlx.mlx_matmul(&truth_a, xt, w_tt, s));
                try mlx.check(mlx.mlx_array_eval(truth_a));
                const truth = mlx.mlx_array_data_float32(truth_a).?[0 .. mu * nu];

                var stock = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(stock);
                try mlx.check(mlx.mlx_quantized_matmul(&stock, x, wq, sc, bi, true, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(2), "affine", s));
                const got = (if (m == 1) try qmv(x, wq, sc, bi, 2, 128, s) else try qmvRows(x, wq, sc, bi, 2, 128, bneg, s)) orelse {
                    try std.testing.expect(m > 1 and (dt != .float16 or !bneg));
                    continue;
                };
                defer _ = mlx.mlx_array_free(got);
                try std.testing.expectEqual(dt, mlx.mlx_array_dtype(got));
                try std.testing.expectEqualSlices(c_int, mlx.getShape(stock), mlx.getShape(got));

                const es = try errVsTruth(stock, truth, mu, nu, s);
                defer std.testing.allocator.free(es);
                const eg = try errVsTruth(got, truth, mu, nu, s);
                defer std.testing.allocator.free(eg);
                for (es, eg) |a, b| {
                    // The output is rounded to dt either way; the bar is stock's own error.
                    try std.testing.expect(b.rms <= 1.05 * a.rms);
                    try std.testing.expect(b.max <= 1.05 * a.max);
                }
            }
        }
    }
}
