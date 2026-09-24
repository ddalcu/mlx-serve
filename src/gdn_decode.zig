//! Fused single-token GatedDeltaNet step for Hadamard packs (B=1, S=1, swish
//! output gate; activations T, recurrent state StT): two dispatches instead of
//! prework + recurrence + norm-gate + rotation. K1 runs one head over SPLIT threadgroups, each
//! recomputing the conv/silu/q-k norm prework for its head (cheaper than a
//! barrier between kernels) before its slice of the recurrence rows. K2 does
//! the per-head gated RMS norm for a 1024 block (8 heads) and rotates it for
//! out_proj. Bit-identical to the composed chain.
const std = @import("std");
const mlx = @import("mlx.zig");

const HEADER =
    \\inline float msv_log1p(float x) {
    \\    float xp1 = 1.0f + x;
    \\    if (xp1 == metal::numeric_limits<float>::max()) { return metal::numeric_limits<float>::max(); }
    \\    if (xp1 == 1.0f) { return x; }
    \\    return x * (metal::log(xp1) / (xp1 - 1.0f));
    \\}
;

const K1_SOURCE =
    \\constexpr int NSG = NT / 32;
    \\constexpr int RB = DV / SPLIT;       // dv rows per threadgroup
    \\constexpr int R = RB / NSG;          // dv rows per simdgroup
    \\constexpr int GRP = HV / HK;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint hv = threadgroup_position_in_grid.x / SPLIT;
    \\uint part = threadgroup_position_in_grid.x % SPLIT;
    \\uint hk = hv / GRP;
    \\threadgroup float qs[DK], ks[DK], vs[DV];
    \\threadgroup float gb[2];
    \\uint row0 = part * RB + sg * R;
    \\float st[R][4];
    \\for (int j = 0; j < R; ++j) {
    \\  uint base = (hv * DV + row0 + j) * DK + lane * 4;
    \\  for (int i = 0; i < 4; ++i) st[j][i] = float(state_in[base + i]);
    \\}
    \\if (sg < 3) {
    \\  uint cb = sg == 0 ? hk * DK : (sg == 1 ? HK * DK + hk * DK : 2 * HK * DK + hv * DV);
    \\  T act[4];
    \\  float sumsq = 0.0f;
    \\  for (int i = 0; i < 4; ++i) {
    \\    uint ch = cb + lane * 4 + i;
    \\    float acc = 0.0f;
    \\    for (int tap = 0; tap < 3; ++tap) acc += float(conv_state[tap * C + ch]) * float(conv_w[ch * 4 + tap]);
    \\    acc += float(qkv[ch]) * float(conv_w[ch * 4 + 3]);
    \\    const T conv = T(acc);
    \\    T sy = T(1) / (T(1) + metal::exp(metal::abs(conv))); T sig = conv < T(0) ? sy : T(1) - sy;
    \\    act[i] = conv * sig;
    \\    float v = float(act[i]);
    \\    sumsq += v * v;
    \\  }
    \\  if (sg < 2) {
    \\    sumsq = simd_sum(sumsq);
    \\    float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f);
    \\    const T scale = sg == 0 ? q_scale : k_scale;
    \\    threadgroup float* dst = sg == 0 ? qs : ks;
    \\    for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = float(scale * T(1) * T(float(act[i]) * inv));
    \\  } else {
    \\    for (int i = 0; i < 4; ++i) vs[lane * 4 + i] = float(act[i]);
    \\  }
    \\  if (part == 0 && (sg == 2 || hv % GRP == 0)) {
    \\    for (int i = 0; i < 4; ++i) {
    \\      uint ch = cb + lane * 4 + i;
    \\      conv_out[ch] = conv_state[C + ch];
    \\      conv_out[C + ch] = conv_state[2 * C + ch];
    \\      conv_out[2 * C + ch] = qkv[ch];
    \\    }
    \\  }
    \\}
    \\if (sg == (NSG > 3 ? 3 : 0) && lane == 31) {
    \\  const T bv = b_in[hv];
    \\  T by = T(1) / (T(1) + metal::exp(metal::abs(bv))); T bsig = bv < T(0) ? by : T(1) - by;
    \\  gb[1] = float(bsig);
    \\  const T apd = T(float(a_in[hv]) + float(dt_bias[hv]));
    \\  float sp = msv_log1p(metal::precise::exp(float(apd)));
    \\  float ea = metal::precise::exp(float(A_log[hv]));
    \\  gb[0] = float(T(metal::precise::exp(-(ea * sp))));
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float kk[4], qq[4];
    \\for (int i = 0; i < 4; ++i) { kk[i] = ks[lane * 4 + i]; qq[i] = qs[lane * 4 + i]; }
    \\const float g = gb[0], beta = gb[1];
    \\for (int j = 0; j < R; ++j) {
    \\  uint dv = row0 + j;
    \\  float kv_mem = 0.0f;
    \\  for (int i = 0; i < 4; ++i) { st[j][i] = st[j][i] * g; kv_mem += st[j][i] * kk[i]; }
    \\  kv_mem = simd_sum(kv_mem);
    \\  float delta = (vs[dv] - kv_mem) * beta;
    \\  float out = 0.0f;
    \\  for (int i = 0; i < 4; ++i) { st[j][i] = st[j][i] + kk[i] * delta; out += st[j][i] * qq[i]; }
    \\  out = simd_sum(out);
    \\  uint base = (hv * DV + dv) * DK + lane * 4;
    \\  for (int i = 0; i < 4; ++i) state_out[base + i] = static_cast<StT>(st[j][i]);
    \\  if (lane == 0) y[hv * DV + dv] = static_cast<T>(out);
    \\}
;

const K2_SOURCE =
    \\constexpr int J = 32, S = 4, P = J / S;
    \\threadgroup float tg[1024];
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint base = threadgroup_position_in_grid.x * 1024;
    \\for (int hh = 0; hh < 2; ++hh) {
    \\  uint hb = base + (sg * 2 + hh) * DV;
    \\  float xs[4];
    \\  float sumsq = 0.0f;
    \\  for (int i = 0; i < 4; ++i) { xs[i] = float(y[hb + lane * 4 + i]); sumsq += xs[i] * xs[i]; }
    \\  sumsq = simd_sum(sumsq);
    \\  float inv = metal::precise::rsqrt(sumsq / float(DV) + eps);
    \\  for (int i = 0; i < 4; ++i) {
    \\    const T normed = norm_w[lane * 4 + i] * T(xs[i] * inv);
    \\    const T zv = z[hb + lane * 4 + i];
    \\    T sy = T(1) / (T(1) + metal::exp(metal::abs(zv))); T sig = zv < T(0) ? sy : T(1) - sy;
    \\    tg[hb - base + lane * 4 + i] = float((zv * sig) * normed) * signs[hb + lane * 4 + i];
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float v[P];
    \\for (int p = 0; p < P; ++p) v[p] = tg[(sg * P + p) * 32 + lane];
    \\for (int h = 1; h < P; h <<= 1)
    \\  for (int p = 0; p < P; ++p)
    \\    if ((p & h) == 0) { float a = v[p], b = v[p + h]; v[p] = a + b; v[p + h] = a - b; }
    \\for (uint m = 1; m < 32; m <<= 1) {
    \\  float sgn = (lane & m) ? -1.0f : 1.0f;
    \\  for (int p = 0; p < P; ++p) v[p] = fma(sgn, v[p], simd_shuffle_xor(v[p], m));
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\for (int p = 0; p < P; ++p) tg[(sg * P + p) * 32 + lane] = v[p];
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\const float scale = rsqrt(1024.0f);
    \\for (uint p = sg; p < P; p += S) {
    \\  float w[S];
    \\  for (int q = 0; q < S; ++q) w[q] = tg[(q * P + p) * 32 + lane];
    \\  float a0 = w[0] + w[1], a1 = w[0] - w[1], a2 = w[2] + w[3], a3 = w[2] - w[3];
    \\  w[0] = a0 + a2; w[2] = a0 - a2; w[1] = a1 + a3; w[3] = a1 - a3;
    \\  for (int q = 0; q < S; ++q) rot[base + (q * P + p) * 32 + lane] = static_cast<T>(w[q] * scale);
    \\}
;

// K1 over TL tokens (verify widths): the same per-token prework and recurrence,
// token after token, plus the per-step state capture MTP rollback reads
// (state_seq[TL-1] is never written, as in the stock capture kernel).
const K1S_SOURCE =
    \\constexpr int NSG = NT / 32;
    \\constexpr int RB = DV / SPLIT;
    \\constexpr int R = RB / NSG;
    \\constexpr int GRP = HV / HK;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint hv = threadgroup_position_in_grid.x / SPLIT;
    \\uint part = threadgroup_position_in_grid.x % SPLIT;
    \\uint hk = hv / GRP;
    \\threadgroup float qs[TL][DK], ks[TL][DK], vs[TL][DV];
    \\threadgroup float gb[TL][2];
    \\uint row0 = part * RB + sg * R;
    \\float st[R][4];
    \\for (int j = 0; j < R; ++j) {
    \\  uint base = (hv * DV + row0 + j) * DK + lane * 4;
    \\  for (int i = 0; i < 4; ++i) st[j][i] = float(state_in[base + i]);
    \\}
    \\if (sg < 3) {
    \\  uint cb = sg == 0 ? hk * DK : (sg == 1 ? HK * DK + hk * DK : 2 * HK * DK + hv * DV);
    \\  for (int t = 0; t < TL; ++t) {
    \\    T act[4];
    \\    float sumsq = 0.0f;
    \\    for (int i = 0; i < 4; ++i) {
    \\      uint ch = cb + lane * 4 + i;
    \\      float acc = 0.0f;
    \\      for (int tap = 0; tap < 4; ++tap) {
    \\        int w = t + tap;
    \\        const T xv = w < 3 ? conv_state[w * C + ch] : qkv[(w - 3) * C + ch];
    \\        acc += float(xv) * float(conv_w[ch * 4 + tap]);
    \\      }
    \\      const T conv = T(acc);
    \\      T sy = T(1) / (T(1) + metal::exp(metal::abs(conv))); T sig = conv < T(0) ? sy : T(1) - sy;
    \\      act[i] = conv * sig;
    \\      float v = float(act[i]);
    \\      sumsq += v * v;
    \\    }
    \\    if (sg < 2) {
    \\      sumsq = simd_sum(sumsq);
    \\      float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f);
    \\      const T scale = sg == 0 ? q_scale : k_scale;
    \\      threadgroup float* dst = sg == 0 ? qs[t] : ks[t];
    \\      for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = float(scale * T(1) * T(float(act[i]) * inv));
    \\    } else {
    \\      for (int i = 0; i < 4; ++i) vs[t][lane * 4 + i] = float(act[i]);
    \\    }
    \\  }
    \\  if (part == 0 && (sg == 2 || hv % GRP == 0)) {
    \\    for (int i = 0; i < 4; ++i) {
    \\      uint ch = cb + lane * 4 + i;
    \\      for (int j = 0; j < 3; ++j) {
    \\        int w = TL + j;
    \\        conv_out[j * C + ch] = w < 3 ? conv_state[w * C + ch] : qkv[(w - 3) * C + ch];
    \\      }
    \\    }
    \\  }
    \\}
    \\if (sg == (NSG > 3 ? 3 : 0) && lane == 31) {
    \\  float ea = metal::precise::exp(float(A_log[hv]));
    \\  for (int t = 0; t < TL; ++t) {
    \\    const T bv = b_in[t * HV + hv];
    \\    T by = T(1) / (T(1) + metal::exp(metal::abs(bv))); T bsig = bv < T(0) ? by : T(1) - by;
    \\    gb[t][1] = float(bsig);
    \\    const T apd = T(float(a_in[t * HV + hv]) + float(dt_bias[hv]));
    \\    float sp = msv_log1p(metal::precise::exp(float(apd)));
    \\    gb[t][0] = float(T(metal::precise::exp(-(ea * sp))));
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\for (int t = 0; t < TL; ++t) {
    \\  float kk[4], qq[4];
    \\  for (int i = 0; i < 4; ++i) { kk[i] = ks[t][lane * 4 + i]; qq[i] = qs[t][lane * 4 + i]; }
    \\  const float g = gb[t][0], beta = gb[t][1];
    \\  for (int j = 0; j < R; ++j) {
    \\    uint dv = row0 + j;
    \\    float kv_mem = 0.0f;
    \\    for (int i = 0; i < 4; ++i) { st[j][i] = st[j][i] * g; kv_mem += st[j][i] * kk[i]; }
    \\    kv_mem = simd_sum(kv_mem);
    \\    float delta = (vs[t][dv] - kv_mem) * beta;
    \\    float out = 0.0f;
    \\    for (int i = 0; i < 4; ++i) { st[j][i] = st[j][i] + kk[i] * delta; out += st[j][i] * qq[i]; }
    \\    out = simd_sum(out);
    \\    if (lane == 0) y[(t * HV + hv) * DV + dv] = static_cast<T>(out);
    \\    if (t + 1 < TL) {
    \\      uint sbase = t * (HV * DV * DK) + (hv * DV + dv) * DK + lane * 4;
    \\      for (int i = 0; i < 4; ++i) state_seq[sbase + i] = static_cast<StT>(st[j][i]);
    \\    }
    \\  }
    \\}
    \\for (int j = 0; j < R; ++j) {
    \\  uint base = (hv * DV + row0 + j) * DK + lane * 4;
    \\  for (int i = 0; i < 4; ++i) state_out[base + i] = static_cast<StT>(st[j][i]);
    \\}
;

const SPLIT: c_int = 4;
const NT: c_int = 256; // 4 dv rows per simdgroup

var k1_cache: ?mlx.mlx_fast_metal_kernel = null;
var k2_cache: ?mlx.mlx_fast_metal_kernel = null;

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

pub const Geometry = struct { hk: c_int, hv: c_int, dk: c_int, dv: c_int };

pub const Inputs = struct {
    qkv: mlx.mlx_array, // [1,1,C]
    z: mlx.mlx_array, // [1,1,Hv*Dv]
    a: mlx.mlx_array, // [1,1,Hv]
    b: mlx.mlx_array, // [1,1,Hv]
    conv_state: mlx.mlx_array, // [1,3,C]
    ssm_state: mlx.mlx_array, // [1,Hv,Dv,Dk]
    conv_w: mlx.mlx_array,
    A_log: mlx.mlx_array,
    dt_bias: mlx.mlx_array,
    q_scale: mlx.mlx_array, // 0-dim bf16
    k_scale: mlx.mlx_array, // 0-dim bf16
    norm_w: mlx.mlx_array,
    eps: mlx.mlx_array, // 0-dim f32
    signs: mlx.mlx_array, // [Hv*Dv] f32, out_proj's
};

pub const Outputs = struct { rot: mlx.mlx_array, conv_state: mlx.mlx_array, ssm_state: mlx.mlx_array };

const CfgKey = struct { g: Geometry, dt: mlx.mlx_dtype, st: mlx.mlx_dtype };
var cfg_key: ?CfgKey = null;
var cfg1: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null };
var cfg2: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null };

fn buildConfigs(g: Geometry, dt: mlx.mlx_dtype, st: mlx.mlx_dtype) !void {
    const c = 2 * g.hk * g.dk + g.hv * g.dv;
    const vd = g.hv * g.dv;
    const c1 = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c1);
    const y_shape = [_]c_int{ 1, 1, g.hv, g.dv };
    const cs_shape = [_]c_int{ 1, 3, c };
    const st_shape = [_]c_int{ 1, g.hv, g.dv, g.dk };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c1, &y_shape, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c1, &cs_shape, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c1, &st_shape, 4, st));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c1, g.hv * SPLIT * NT, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c1, NT, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c1, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c1, "StT", st));
    inline for (.{ .{ "HK", g.hk }, .{ "HV", g.hv }, .{ "DK", g.dk }, .{ "DV", g.dv }, .{ "C", c }, .{ "NT", NT }, .{ "SPLIT", SPLIT } }) |kv|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c1, kv[0], kv[1]));
    const c2 = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c2);
    const rot_shape = [_]c_int{ 1, 1, vd };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c2, &rot_shape, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c2, @divExact(vd, 1024) * 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c2, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c2, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c2, "DV", g.dv));
    if (cfg1.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(cfg1);
    if (cfg2.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(cfg2);
    cfg1 = c1;
    cfg2 = c2;
    cfg_key = .{ .g = g, .dt = dt, .st = st };
}

pub const Recur = struct { y: mlx.mlx_array, conv_state: mlx.mlx_array, ssm_state: mlx.mlx_array };

/// K1 alone: prework + recurrence, y as [1,1,Hv,Dv]; z, norm_w, eps and signs
/// are not read. Null when the geometry or dtypes are outside the kernel.
pub fn recur(g: Geometry, in: Inputs, s: mlx.mlx_stream) !?Recur {
    if (g.dk != 128 or g.dv != 128 or @rem(g.hv, g.hk) != 0 or @rem(g.hv * g.dv, 1024) != 0) return null;
    const dt = mlx.mlx_array_dtype(in.qkv);
    if (dt != .bfloat16 and dt != .float16) return null;
    for ([_]mlx.mlx_array{ in.a, in.b, in.conv_state, in.conv_w, in.A_log, in.dt_bias, in.q_scale, in.k_scale }) |arr|
        if (mlx.mlx_array_dtype(arr) != dt) return null;
    const st = mlx.mlx_array_dtype(in.ssm_state);
    if (st != dt and st != .float32) return null;
    if (k1_cache == null) k1_cache = try makeKernel("msv_gdn_decode_recur", &.{ "qkv", "a_in", "b_in", "conv_state", "state_in", "conv_w", "A_log", "dt_bias", "q_scale", "k_scale" }, &.{ "y", "conv_out", "state_out" }, K1_SOURCE, HEADER);
    const key = CfgKey{ .g = g, .dt = dt, .st = st };
    if (cfg_key == null or !std.meta.eql(cfg_key.?, key)) try buildConfigs(g, dt, st);

    const in1 = [_]mlx.mlx_array{ in.qkv, in.a, in.b, in.conv_state, in.ssm_state, in.conv_w, in.A_log, in.dt_bias, in.q_scale, in.k_scale };
    const v1 = mlx.mlx_vector_array_new_data(&in1, in1.len);
    defer _ = mlx.mlx_vector_array_free(v1);
    var o1 = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o1);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o1, k1_cache.?, v1, cfg1, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    var conv_out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(conv_out);
    var state_out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(state_out);
    try mlx.check(mlx.mlx_vector_array_get(&y, o1, 0));
    try mlx.check(mlx.mlx_vector_array_get(&conv_out, o1, 1));
    try mlx.check(mlx.mlx_vector_array_get(&state_out, o1, 2));
    return .{ .y = y, .conv_state = conv_out, .ssm_state = state_out };
}

/// Null when the geometry or dtypes are outside the kernels (caller keeps the chain).
pub fn step(g: Geometry, in: Inputs, s: mlx.mlx_stream) !?Outputs {
    for ([_]mlx.mlx_array{ in.z, in.norm_w }) |arr|
        if (mlx.mlx_array_dtype(arr) != mlx.mlx_array_dtype(in.qkv)) return null;
    const r = (try recur(g, in, s)) orelse return null;
    defer _ = mlx.mlx_array_free(r.y);
    errdefer _ = mlx.mlx_array_free(r.conv_state);
    errdefer _ = mlx.mlx_array_free(r.ssm_state);
    if (k2_cache == null) k2_cache = try makeKernel("msv_gdn_decode_normgate_rot", &.{ "y", "z", "norm_w", "eps", "signs" }, &.{"rot"}, K2_SOURCE, "");

    const in2 = [_]mlx.mlx_array{ r.y, in.z, in.norm_w, in.eps, in.signs };
    const v2 = mlx.mlx_vector_array_new_data(&in2, in2.len);
    defer _ = mlx.mlx_vector_array_free(v2);
    var o2 = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o2);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o2, k2_cache.?, v2, cfg2, s));
    var rot = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&rot, o2, 0));
    return .{ .rot = rot, .conv_state = r.conv_state, .ssm_state = r.ssm_state };
}

pub const RecurSeq = struct { y: mlx.mlx_array, conv_state: mlx.mlx_array, ssm_state: mlx.mlx_array, state_seq: mlx.mlx_array };

pub const MAX_SEQ: c_int = 8;
var k1s_cache: ?mlx.mlx_fast_metal_kernel = null;
var seq_cfgs: [MAX_SEQ + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var seq_cfg_key: ?CfgKey = null;

fn buildSeqConfig(g: Geometry, t_len: c_int, dt: mlx.mlx_dtype, st: mlx.mlx_dtype) !mlx.mlx_fast_metal_kernel_config {
    const c = 2 * g.hk * g.dk + g.hv * g.dv;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, t_len, g.hv, g.dv }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 3, c }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, g.hv, g.dv, g.dk }, 4, st));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ t_len, 1, g.hv, g.dv, g.dk }, 5, st));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, g.hv * SPLIT * NT, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, NT, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "StT", st));
    inline for (.{ .{ "HK", g.hk }, .{ "HV", g.hv }, .{ "DK", g.dk }, .{ "DV", g.dv }, .{ "C", c }, .{ "NT", NT }, .{ "SPLIT", SPLIT } }) |kv|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TL", t_len));
    return cfg;
}

/// K1 over t_len tokens (2..MAX_SEQ) with per-step state capture: y as
/// [1,T,Hv,Dv], the next conv state, the final state and state_seq
/// ([T,1,Hv,Dv,Dk], row T-1 unwritten). Null outside the kernel's geometry.
pub fn recurSeq(g: Geometry, t_len: c_int, in: Inputs, s: mlx.mlx_stream) !?RecurSeq {
    if (t_len < 2 or t_len > MAX_SEQ) return null;
    if (g.dk != 128 or g.dv != 128 or @rem(g.hv, g.hk) != 0) return null;
    const dt = mlx.mlx_array_dtype(in.qkv);
    if (dt != .bfloat16 and dt != .float16) return null;
    for ([_]mlx.mlx_array{ in.a, in.b, in.conv_state, in.conv_w, in.A_log, in.dt_bias, in.q_scale, in.k_scale }) |arr|
        if (mlx.mlx_array_dtype(arr) != dt) return null;
    const st = mlx.mlx_array_dtype(in.ssm_state);
    if (st != dt and st != .float32) return null;
    if (k1s_cache == null) k1s_cache = try makeKernel("msv_gdn_decode_recur_seq", &.{ "qkv", "a_in", "b_in", "conv_state", "state_in", "conv_w", "A_log", "dt_bias", "q_scale", "k_scale" }, &.{ "y", "conv_out", "state_out", "state_seq" }, K1S_SOURCE, HEADER);
    const key = CfgKey{ .g = g, .dt = dt, .st = st };
    if (seq_cfg_key == null or !std.meta.eql(seq_cfg_key.?, key)) {
        for (&seq_cfgs) |*slot| if (slot.*) |c| {
            _ = mlx.mlx_fast_metal_kernel_config_free(c);
            slot.* = null;
        };
        seq_cfg_key = key;
    }
    const idx: usize = @intCast(t_len);
    if (seq_cfgs[idx] == null) seq_cfgs[idx] = try buildSeqConfig(g, t_len, dt, st);

    const in1 = [_]mlx.mlx_array{ in.qkv, in.a, in.b, in.conv_state, in.ssm_state, in.conv_w, in.A_log, in.dt_bias, in.q_scale, in.k_scale };
    const v1 = mlx.mlx_vector_array_new_data(&in1, in1.len);
    defer _ = mlx.mlx_vector_array_free(v1);
    var o1 = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o1);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o1, k1s_cache.?, v1, seq_cfgs[idx].?, s));
    var out: [4]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (out) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&out, 0..) |*a, i| try mlx.check(mlx.mlx_vector_array_get(a, o1, i));
    return .{ .y = out[0], .conv_state = out[1], .ssm_state = out[2], .state_seq = out[3] };
}
