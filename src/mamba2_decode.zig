//! Fused Mamba2 mixer step for short windows (decode and spec verify): the
//! conv window, depthwise conv + bias + SiLU, dt softplus/clamp, the SSM
//! recurrence, the D skip and the SiLU(z) gate run as ONE dispatch over R
//! consecutive rows; the grouped RMS norm is a second. Same structure as
//! TensorFold's `mamba_step`/`group_norm` (MIT), arithmetic kept to the op
//! chain's: f32 math, bf16 where the chain stores bf16. The SSM state stays
//! f32; `y` comes back in x's dtype.
const std = @import("std");
const mlx = @import("mlx.zig");

pub const MAX_ROWS: c_int = 16;
const TGY: c_int = 8;

// grid (32, Dh, B*H), threadgroup (32, TGY, 1): one simdgroup per (b, h, d),
// each lane owning Ds/32 state columns. B/C of the head's group are conv'd
// once per threadgroup into shared memory.
const STEP_SOURCE =
    \\const uint lane = thread_position_in_threadgroup.x;
    \\const uint d = thread_position_in_grid.y;
    \\const uint n = thread_position_in_grid.z;
    \\const uint h = n % H;
    \\const uint b = n / H;
    \\const uint g = h / (H / NG);
    \\const int R = rows;
    \\constexpr int NS = DS / 32;
    \\constexpr int CD = XD + 2 * NG * DS;
    \\constexpr int XOFF = XD;
    \\constexpr int DTOFF = XD + CD;
    \\const int cx = int(h) * DH + int(d);
    \\const device T* Pb = P + size_t(b) * R * PROJ;
    \\const device T* CSb = CS_IN + size_t(b) * (KC - 1) * CD;
    \\float st[NS];
    \\const size_t sbase = size_t(n) * DH * DS + size_t(d) * DS + size_t(lane) * NS;
    \\for (int i = 0; i < NS; i++) st[i] = S_IN[sbase + i];
    \\const float A = -float(T(metal::exp(float(A_LOG[h]))));
    \\const float dtb = float(DT_BIAS[h]);
    \\#define TAP(ch, pos) ((pos) < 0 ? float(CSb[((pos) + KC - 1) * CD + (ch)]) : float(Pb[(pos) * PROJ + XOFF + (ch)]))
    \\#define CONV(ch, rr, out) { \
    \\    float a_ = float(CB[ch]); \
    \\    for (int k_ = 0; k_ < KC; k_++) a_ = metal::fma(float(CW[(ch) * KC + k_]), TAP(ch, (rr) - (KC - 1) + k_), a_); \
    \\    const float cv_ = float(T(a_)); \
    \\    out = float(T(cv_ / (1.0f + metal::exp(-cv_)))); }
    \\threadgroup float bc[MAXR * 2 * DS];
    \\const uint tid = thread_position_in_threadgroup.y * 32 + lane;
    \\for (int rr = 0; rr < R; rr++) {
    \\    for (uint c = tid; c < 2 * DS; c += 32 * TGY) {
    \\        const int ch = c < DS ? XD + int(g) * DS + int(c) : XD + NG * DS + int(g) * DS + int(c - DS);
    \\        float v; CONV(ch, rr, v);
    \\        bc[rr * 2 * DS + c] = v;
    \\    }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\for (int rr = 0; rr < R; rr++) {
    \\    float xv = 0.0f;
    \\    if (lane == 0) { CONV(cx, rr, xv); }
    \\    xv = simd_broadcast(xv, 0);
    \\    float dt = float(Pb[rr * PROJ + DTOFF + int(h)]) + dtb;
    \\    dt = metal::max(dt, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(dt)));
    \\    dt = metal::clamp(dt, dt_lo, dt_hi);
    \\    const float dA = metal::exp(A * dt);
    \\    const float xdt = xv * dt;
    \\    float acc = 0.0f;
    \\    for (int i = 0; i < NS; i++) {
    \\        const float s = dA * st[i] + xdt * bc[rr * 2 * DS + int(lane) * NS + i];
    \\        st[i] = s;
    \\        acc += s * bc[rr * 2 * DS + DS + int(lane) * NS + i];
    \\    }
    \\    acc = simd_sum(acc);
    \\    if (lane == 0) {
    \\        const float dx = float(T(T(xv) * DSKIP[h]));
    \\        const float y = float(T(acc + dx));
    \\        const float z = float(Pb[rr * PROJ + cx]);
    \\        const float sz = float(T(z / (1.0f + metal::exp(-z))));
    \\        Y[(size_t(b) * R + rr) * XD + cx] = T(sz * y);
    \\    }
    \\    if (ALLROWS || rr == R - 1) {
    \\        const size_t so = ALLROWS ? size_t(rr) * B * H * DH * DS : 0;
    \\        for (int i = 0; i < NS; i++) S_OUT[so + sbase + i] = st[i];
    \\    }
    \\}
    \\// conv state after the last row: its last KC-1 inputs, one thread per channel.
    \\if (lane == 0) {
    \\    for (int k = 0; k < KC - 1; k++) {
    \\        const int pos = R - 1 - (KC - 2) + k;
    \\        CS_OUT[(size_t(b) * (KC - 1) + k) * CD + cx] = pos < 0 ? CSb[(pos + KC - 1) * CD + cx] : Pb[pos * PROJ + XOFF + cx];
    \\    }
    \\}
    \\if ((h % (H / NG)) == 0 && d == 0) {
    \\    for (int c = int(lane); c < 2 * DS; c += 32) {
    \\        const int ch = c < DS ? XD + int(g) * DS + c : XD + NG * DS + int(g) * DS + (c - DS);
    \\        for (int k = 0; k < KC - 1; k++) {
    \\            const int pos = R - 1 - (KC - 2) + k;
    \\            CS_OUT[(size_t(b) * (KC - 1) + k) * CD + ch] = pos < 0 ? CSb[(pos + KC - 1) * CD + ch] : Pb[pos * PROJ + XOFF + ch];
    \\        }
    \\    }
    \\}
;

// grid (GS/4, NG, rows): one threadgroup per (row, group), 4 elements a thread.
const GNORM_SOURCE =
    \\const uint t = thread_position_in_threadgroup.x;
    \\const uint grp = threadgroup_position_in_grid.y;
    \\const uint r = threadgroup_position_in_grid.z;
    \\constexpr int TN = GS / 4;
    \\threadgroup float partial[TN / 32];
    \\const size_t base = size_t(r) * XD + size_t(grp) * GS + size_t(t) * 4;
    \\float v[4];
    \\float ss = 0.0f;
    \\for (int i = 0; i < 4; i++) { v[i] = float(X[base + i]); ss = metal::fma(v[i], v[i], ss); }
    \\ss = simd_sum(ss);
    \\if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = ss;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float total = 0.0f;
    \\for (int s = 0; s < TN / 32; s++) total += partial[s];
    \\const float scale = metal::rsqrt(total / float(GS) + eps);
    \\for (int i = 0; i < 4; i++) {
    \\    const int c = int(grp) * GS + int(t) * 4 + i;
    \\    OUT[base + i] = T(float(W[c]) * float(T(v[i] * scale)));
    \\}
;

pub const Geometry = struct { batch: c_int, rows: c_int, heads: c_int, groups: c_int, head_dim: c_int, state: c_int, conv_kernel: c_int, proj: c_int };

pub const Inputs = struct {
    proj: mlx.mlx_array, // [B,R,PROJ] in_proj output: z | x B C | dt
    conv_state: mlx.mlx_array, // [B,KC-1,CD]
    ssm_state: mlx.mlx_array, // [B,H,Dh,Ds] f32
    conv_w: mlx.mlx_array, // [CD,KC,1]
    conv_b: mlx.mlx_array, // [CD]
    A_log: mlx.mlx_array, // [H]
    dt_bias: mlx.mlx_array, // [H]
    D: mlx.mlx_array, // [H]
    dt_lo: mlx.mlx_array, // 0-dim f32 (MLX binds a scalar input by value)
    dt_hi: mlx.mlx_array, // 0-dim f32
    rows: mlx.mlx_array, // 0-dim int32
    all_rows: bool, // state after EVERY row (spec capture) or only the last
};

pub const Outputs = struct {
    y: mlx.mlx_array, // [B,R,H*Dh] gated, x dtype
    ssm_state: mlx.mlx_array, // [R,B,H,Dh,Ds] f32 when all_rows, else [1,B,H,Dh,Ds]
    conv_state: mlx.mlx_array, // [B,KC-1,CD]
};

var step_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gnorm_kernel: ?mlx.mlx_fast_metal_kernel = null;
const StepKey = struct { g: Geometry, dt: mlx.mlx_dtype, all_rows: bool };
var step_key: ?StepKey = null;
var step_cfg: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null };
const GnormKey = struct { rows: c_int, xd: c_int, groups: c_int, dt: mlx.mlx_dtype };
var gnorm_key: ?GnormKey = null;
var gnorm_cfg: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null };

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn tmplInt(c: mlx.mlx_fast_metal_kernel_config, name: [*:0]const u8, v: c_int) !void {
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name, v));
}

fn buildStepConfig(g: Geometry, dt: mlx.mlx_dtype, all_rows: bool) !void {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    const xd = g.heads * g.head_dim;
    const cd = xd + 2 * g.groups * g.state;
    const srows: c_int = if (all_rows) g.rows else 1;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ g.batch, g.rows, xd }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ srows, g.batch, g.heads, g.head_dim, g.state }, 5, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ g.batch, g.conv_kernel - 1, cd }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32, g.head_dim, g.batch * g.heads));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, TGY, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
    try tmplInt(c, "XD", xd);
    try tmplInt(c, "DH", g.head_dim);
    try tmplInt(c, "DS", g.state);
    try tmplInt(c, "H", g.heads);
    try tmplInt(c, "NG", g.groups);
    try tmplInt(c, "KC", g.conv_kernel);
    try tmplInt(c, "PROJ", g.proj);
    try tmplInt(c, "B", g.batch);
    try tmplInt(c, "MAXR", MAX_ROWS);
    try tmplInt(c, "TGY", TGY);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(c, "ALLROWS", all_rows));
    if (step_cfg.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(step_cfg);
    step_cfg = c;
    step_key = .{ .g = g, .dt = dt, .all_rows = all_rows };
}

/// Null when the geometry, dtypes or stream are outside the kernel (the
/// caller keeps the op chain).
pub fn step(g: Geometry, in: Inputs, s: mlx.mlx_stream) !?Outputs {
    if (!mlx.streamIsGpu(s)) return null;
    if (g.rows < 1 or g.rows > MAX_ROWS) return null;
    if (@rem(g.state, 32) != 0 or @rem(g.head_dim, TGY) != 0 or @rem(g.heads, g.groups) != 0 or g.conv_kernel < 2) return null;
    const xd = g.heads * g.head_dim;
    if (g.proj != 2 * xd + 2 * g.groups * g.state + g.heads) return null;
    const dt = mlx.mlx_array_dtype(in.proj);
    if (dt != .bfloat16 and dt != .float16) return null;
    for ([_]mlx.mlx_array{ in.conv_state, in.conv_w, in.conv_b, in.A_log, in.dt_bias, in.D }) |arr|
        if (arr.ctx == null or mlx.mlx_array_dtype(arr) != dt) return null;
    if (mlx.mlx_array_dtype(in.ssm_state) != .float32) return null;
    if (step_kernel == null) {
        const ins = [_][*:0]const u8{ "P", "CS_IN", "S_IN", "CW", "CB", "A_LOG", "DT_BIAS", "DSKIP", "dt_lo", "dt_hi", "rows" };
        const outs = [_][*:0]const u8{ "Y", "S_OUT", "CS_OUT" };
        step_kernel = try makeKernel("msv_mamba2_step", &ins, &outs, STEP_SOURCE);
    }
    const key = StepKey{ .g = g, .dt = dt, .all_rows = in.all_rows };
    if (step_key == null or !std.meta.eql(step_key.?, key)) try buildStepConfig(g, dt, in.all_rows);

    const arrs = [_]mlx.mlx_array{ in.proj, in.conv_state, in.ssm_state, in.conv_w, in.conv_b, in.A_log, in.dt_bias, in.D, in.dt_lo, in.dt_hi, in.rows };
    const v = mlx.mlx_vector_array_new_data(&arrs, arrs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, step_kernel.?, v, step_cfg, s));
    var out: Outputs = .{ .y = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .conv_state = mlx.mlx_array_new() };
    errdefer {
        _ = mlx.mlx_array_free(out.y);
        _ = mlx.mlx_array_free(out.ssm_state);
        _ = mlx.mlx_array_free(out.conv_state);
    }
    try mlx.check(mlx.mlx_vector_array_get(&out.y, o, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.ssm_state, o, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.conv_state, o, 2));
    return out;
}

/// Grouped RMS norm times the norm weight: x [B,R,XD] over XD/groups-wide
/// groups. Null outside the kernel's set.
pub fn groupNorm(x: mlx.mlx_array, w: mlx.mlx_array, eps: mlx.mlx_array, groups: c_int, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const sh = mlx.getShape(x);
    if (sh.len != 3) return null;
    const rows = sh[0] * sh[1];
    const xd = sh[2];
    if (@rem(xd, groups) != 0) return null;
    const gs = @divExact(xd, groups);
    if (@rem(gs, 128) != 0 or gs > 4096) return null; // GS/4 threads, whole simdgroups
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .bfloat16 and dt != .float16) return null;
    if (mlx.mlx_array_dtype(w) != dt) return null;
    if (gnorm_kernel == null) {
        const ins = [_][*:0]const u8{ "X", "W", "eps" };
        const outs = [_][*:0]const u8{"OUT"};
        gnorm_kernel = try makeKernel("msv_mamba2_gnorm", &ins, &outs, GNORM_SOURCE);
    }
    const key = GnormKey{ .rows = rows, .xd = xd, .groups = groups, .dt = dt };
    if (gnorm_key == null or !std.meta.eql(gnorm_key.?, key)) {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, sh.ptr, sh.len, dt));
        const tn = @divExact(gs, 4);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, tn, groups, rows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, tn, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
        try tmplInt(c, "XD", xd);
        try tmplInt(c, "GS", gs);
        if (gnorm_cfg.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(gnorm_cfg);
        gnorm_cfg = c;
        gnorm_key = key;
    }
    const arrs = [_]mlx.mlx_array{ x, w, eps };
    const v = mlx.mlx_vector_array_new_data(&arrs, arrs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, gnorm_kernel.?, v, gnorm_cfg, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, o, 0));
    return y;
}
