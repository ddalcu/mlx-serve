//! Fused elementwise glue for the MiniMax-H3 DiT step. Each kernel replaces an op chain that
//! makes several full passes over [S, hidden]-sized tensors (S ~ 15k rows): q/k RMSNorm + partial
//! RoPE + the head-major transposes feeding SDPA, and SwiGLU over the fused fc1 output.
//!
//! Outputs are BIT-EQUAL to the chain they replace (same f32 arithmetic, same bf16 rounding
//! points), so the chain stays both the fallback and the oracle. `MINIMAX_H3_GLUE_FUSED=0`
//! turns every kernel here off.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

var enabled_cache: ?bool = null;
pub var disabled_for_test = false; // lets a test run the chain these kernels replace

pub fn enabled() bool {
    if (disabled_for_test) return false;
    if (enabled_cache) |v| return v;
    const raw = std.c.getenv("MINIMAX_H3_GLUE_FUSED");
    const on = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
    enabled_cache = on;
    return on;
}

fn eligibleDtype(dt: mlx.mlx_dtype) bool {
    return dt == .bfloat16 or dt == .float16;
}

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, src, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

/// Runs `kernel`; `outs` receives one handle per declared output.
fn apply(kernel: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, arrs: []const mlx.mlx_array, outs: []mlx.mlx_array, s: mlx.mlx_stream) !void {
    const v = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, kernel, v, cfg, s));
    if (mlx.mlx_vector_array_size(o) != outs.len) return error.MetalKernelBadOutputCount;
    var got: usize = 0;
    errdefer for (outs[0..got]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (outs, 0..) |*slot, i| {
        slot.* = mlx.mlx_array_new();
        got = i + 1;
        try mlx.check(mlx.mlx_vector_array_get(slot, o, i));
    }
}

// ── q/k RMSNorm + partial RoPE + head-major layout ──────────────────────────
//
// One simdgroup per (row, head): lane l owns elements [4l, 4l+4) of the 128-wide head. The RMS half
// mirrors MLX's `rms_single_row` at axis 128 (f32 squares in index order, `simd_sum`,
// `precise::rsqrt`, `w * T(x*inv)`); the rotation is the chain's own five-op bf16 sequence
// (x1*cos, x2*sin, subtract, x1*sin, x2*cos, add), each product rounded to T. Pair p couples
// elements p and p+HALF, so the partner lane is HALF/4 away; the tail past 2*HALF passes through.
// v is only transposed. Output [1, H, S, 128] for q, k and v.
const QKV_SOURCE =
    \\constexpr uint HD = 128;
    \\const uint slot = threadgroup_position_in_grid.x;
    \\const uint row = threadgroup_position_in_grid.y;
    \\const uint lane = thread_position_in_threadgroup.x;
    \\const uint which = slot / uint(H);
    \\const uint head = slot - which * uint(H);
    \\const uint seq = uint(COS_shape[0]);
    \\const device T* src = QKV + size_t(row) * (3 * uint(H) * HD) + size_t(slot) * HD;
    \\const size_t o = (size_t(head) * seq + row) * HD;
    \\const uint base = lane * 4;
    \\if (which == 2) {
    \\    for (uint i = 0; i < 4; ++i) OV[o + base + i] = src[base + i];
    \\    return;
    \\}
    \\float acc = 0.0f;
    \\for (uint i = 0; i < 4; ++i) {
    \\    float v = float(src[base + i]);
    \\    acc += v * v;
    \\}
    \\acc = simd_sum(acc);
    \\const float inv = metal::precise::rsqrt(acc / float(HD) + eps);
    \\const device T* w = which == 0 ? QW : KW;
    \\T nrm[4];
    \\for (uint i = 0; i < 4; ++i) nrm[i] = w[base + i] * T(float(src[base + i]) * inv);
    \\constexpr uint QL = uint(HALF) / 4;
    \\const uint pl = lane < QL ? lane + QL : (lane < 2 * QL ? lane - QL : lane);
    \\device T* dst = which == 0 ? OQ : OK;
    \\const size_t tab = size_t(row) * uint(HALF);
    \\for (uint i = 0; i < 4; ++i) {
    \\    const uint j = base + i;
    \\    const float pv = simd_shuffle(float(nrm[i]), pl);
    \\    T res;
    \\    if (j < uint(HALF)) {
    \\        const T a1 = T(float(nrm[i]) * float(COS[tab + j]));
    \\        const T a2 = T(pv * float(SIN[tab + j]));
    \\        res = T(float(a1) - float(a2));
    \\    } else if (j < 2 * uint(HALF)) {
    \\        const uint jj = j - uint(HALF);
    \\        const T b1 = T(pv * float(SIN[tab + jj]));
    \\        const T b2 = T(float(nrm[i]) * float(COS[tab + jj]));
    \\        res = T(float(b1) + float(b2));
    \\    } else {
    \\        res = nrm[i];
    \\    }
    \\    dst[o + j] = res;
    \\}
;

var qkv_kernel: ?mlx.mlx_fast_metal_kernel = null;
const QkvKey = struct { h: c_int, half: c_int, seq: c_int, dt: mlx.mlx_dtype };
var qkv_key: ?QkvKey = null;
var qkv_cfg: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null };
var qkv_engaged = false;

pub const Qkv = struct { q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array };

/// qkv [S, 3*H*128] (q | k | v thirds, head-major inside each); cos/sin [.., S, half] (any shape
/// of S*half elements). Null outside the envelope — the caller keeps the chain.
pub fn qkvPrep(qkv: mlx.mlx_array, q_w: mlx.mlx_array, k_w: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, eps: f32, heads: c_int, half: c_int, s: mlx.mlx_stream) !?Qkv {
    if (!enabled() or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(qkv);
    if (!eligibleDtype(dt)) return null;
    for ([_]mlx.mlx_array{ q_w, k_w, cos, sin }) |a| if (mlx.mlx_array_dtype(a) != dt) return null;
    const sh = mlx.getShape(qkv);
    if (sh.len != 2 or sh[1] != 3 * heads * 128) return null;
    if (half <= 0 or @rem(half, 4) != 0 or half * 2 > 128) return null;
    const seq = sh[0];
    if (mlx.mlx_array_size(cos) != @as(usize, @intCast(seq)) * @as(usize, @intCast(half)) or mlx.mlx_array_size(sin) != mlx.mlx_array_size(cos)) return null;

    const tab_shape = [_]c_int{ seq, half };
    var cos2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cos2);
    try mlx.check(mlx.mlx_reshape(&cos2, cos, &tab_shape, 2, s));
    var sin2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sin2);
    try mlx.check(mlx.mlx_reshape(&sin2, sin, &tab_shape, 2, s));

    if (qkv_kernel == null) qkv_kernel = try makeKernel("msv_h3_qkv_prep", &[_][*:0]const u8{ "QKV", "QW", "KW", "COS", "SIN", "eps" }, &[_][*:0]const u8{ "OQ", "OK", "OV" }, QKV_SOURCE, "");
    const key = QkvKey{ .h = heads, .half = half, .seq = seq, .dt = dt };
    if (qkv_key == null or !std.meta.eql(qkv_key.?, key)) {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const out_shape = [_]c_int{ 1, heads, seq, 128 };
        for (0..3) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * 3 * heads, seq, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "H", heads));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "HALF", half));
        if (qkv_cfg.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(qkv_cfg);
        qkv_cfg = c;
        qkv_key = key;
    }
    // A 0-dim f32 binds `constant float&`; a [1] array would bind a pointer.
    const eps_a = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_a);
    var outs: [3]mlx.mlx_array = undefined;
    try apply(qkv_kernel.?, qkv_cfg, &[_]mlx.mlx_array{ qkv, q_w, k_w, cos2, sin2, eps_a }, &outs, s);
    if (!qkv_engaged) {
        qkv_engaged = true;
        log.info("[minimax-h3] fused q/k norm+rope+layout engaged: rows={d} heads={d} rope={d}\n", .{ seq, heads, half * 2 });
    }
    return .{ .q = outs[0], .k = outs[1], .v = outs[2] };
}

// ── SwiGLU over the fused fc1 output ────────────────────────────────────────
//
// y [S, 2F] = gate | up; out = silu(gate) * up, with silu = x * sigmoid(x) rounded like the chain's
// sigmoid -> multiply -> multiply (each result rounded to T). `sigm` is MLX's own expression.
const SWIGLU_HEADER =
    \\template <typename T>
    \\T msv_sigmoid(T x) {
    \\    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    \\    return (x < 0) ? y : 1 - y;
    \\}
;
const SWIGLU_SOURCE =
    \\const uint idx = thread_position_in_grid.x;
    \\const uint f4 = uint(F) / 4;
    \\const uint row = idx / f4;
    \\const uint c = (idx - row * f4) * 4;
    \\const size_t g0 = size_t(row) * (2 * uint(F)) + c;
    \\for (uint i = 0; i < 4; ++i) {
    \\    const T g = Y[g0 + i];
    \\    const T u = Y[g0 + uint(F) + i];
    \\    const T sg = msv_sigmoid<T>(g);
    \\    const T gs = g * sg;
    \\    OUT[size_t(row) * uint(F) + c + i] = gs * u;
    \\}
;

var swiglu_kernel: ?mlx.mlx_fast_metal_kernel = null;
const SwigluKey = struct { rows: c_int, f: c_int, dt: mlx.mlx_dtype };
var swiglu_key: ?SwigluKey = null;
var swiglu_cfg: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null };
var swiglu_engaged = false;

/// y [S, 2F] -> [S, F]. Null outside the envelope.
pub fn swiglu(y: mlx.mlx_array, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!enabled() or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(y);
    if (!eligibleDtype(dt)) return null;
    const sh = mlx.getShape(y);
    if (sh.len != 2 or @rem(sh[1], 8) != 0 or @as(i64, sh[0]) * sh[1] > std.math.maxInt(c_int)) return null;
    const rows = sh[0];
    const f = @divExact(sh[1], 2);
    if (swiglu_kernel == null) swiglu_kernel = try makeKernel("msv_h3_swiglu", &[_][*:0]const u8{"Y"}, &[_][*:0]const u8{"OUT"}, SWIGLU_SOURCE, SWIGLU_HEADER);
    const key = SwigluKey{ .rows = rows, .f = f, .dt = dt };
    if (swiglu_key == null or !std.meta.eql(swiglu_key.?, key)) {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const out_shape = [_]c_int{ rows, f };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 2, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(rows * f, 4), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "F", f));
        if (swiglu_cfg.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(swiglu_cfg);
        swiglu_cfg = c;
        swiglu_key = key;
    }
    var outs: [1]mlx.mlx_array = undefined;
    try apply(swiglu_kernel.?, swiglu_cfg, &[_]mlx.mlx_array{y}, &outs, s);
    if (!swiglu_engaged) {
        swiglu_engaged = true;
        log.info("[minimax-h3] fused swiglu engaged: rows={d} ffn={d}\n", .{ rows, f });
    }
    return outs[0];
}
