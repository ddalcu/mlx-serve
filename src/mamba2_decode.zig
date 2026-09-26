//! Fused single-token Mamba2 SSM step (mlx-lm's `ssm_kernel` with its
//! `compute_dt` folded in): one dispatch per layer instead of the per-timestep
//! op chain. The state stays f32 (the checkpoint's `mamba_ssm_cache_dtype`);
//! `y` comes back in x's dtype. Roundings the chain performs in the activation
//! dtype (`-exp(A_log)`, `x * D`) are reproduced so the two agree to reduction
//! order.
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

// grid (32, Dh, B*H), threadgroup (32, 8, 1): one simdgroup per (row n, d),
// each lane owning Ds/32 state columns.
const SOURCE =
    \\constexpr int n_per_t = Ds / 32;
    \\uint n = thread_position_in_grid.z;
    \\uint h_idx = n % H;
    \\uint g_idx = n / G;
    \\uint ds_idx = thread_position_in_threadgroup.x;
    \\uint d_idx = thread_position_in_grid.y;
    \\const device T* x = X + n * Dh;
    \\const device float* i_state = state_in + n * Dh * Ds;
    \\device float* o_state = state_out + n * Dh * Ds;
    \\const device T* B_ = B + g_idx * Ds;
    \\const device T* C_ = C + g_idx * Ds;
    \\float dt_ = msv_log1p(metal::exp(float(dt_raw[n]) + float(dt_bias[h_idx])));
    \\dt_ = metal::clamp(dt_, dt_lo, dt_hi);
    \\T a_act = T(metal::exp(float(A_log[h_idx])));
    \\float dA = metal::exp(-float(a_act) * dt_);
    \\float x_ = float(x[d_idx]);
    \\float xdt = x_ * dt_;
    \\float acc = 0.0f;
    \\for (int i = 0; i < n_per_t; ++i) {
    \\    int s_idx = n_per_t * ds_idx + i;
    \\    int idx = d_idx * Ds + s_idx;
    \\    float st = dA * i_state[idx] + xdt * float(B_[s_idx]);
    \\    o_state[idx] = st;
    \\    acc += st * float(C_[s_idx]);
    \\}
    \\acc = simd_sum(acc);
    \\if (thread_index_in_simdgroup == 0) {
    \\    T dx = x[d_idx] * D[h_idx];
    \\    out[n * Dh + d_idx] = T(acc + float(dx));
    \\}
;

pub const Geometry = struct { batch: c_int, heads: c_int, groups: c_int, head_dim: c_int, state: c_int };

pub const Inputs = struct {
    x: mlx.mlx_array, // [B,1,H,Dh] activation dtype (post conv+silu)
    B: mlx.mlx_array, // [B,1,G,Ds]
    C: mlx.mlx_array, // [B,1,G,Ds]
    dt_raw: mlx.mlx_array, // [B,1,H] pre-softplus
    A_log: mlx.mlx_array, // [H]
    dt_bias: mlx.mlx_array, // [H]
    D: mlx.mlx_array, // [H]
    state: mlx.mlx_array, // [B,H,Dh,Ds] f32
    dt_lo: mlx.mlx_array, // 0-dim f32 (MLX binds a scalar input by value)
    dt_hi: mlx.mlx_array, // 0-dim f32
};

pub const Outputs = struct { y: mlx.mlx_array, state: mlx.mlx_array };

var kernel_cache: ?mlx.mlx_fast_metal_kernel = null;
const CfgKey = struct { g: Geometry, dt: mlx.mlx_dtype };
var cfg_key: ?CfgKey = null;
var cfg: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null };

fn buildConfig(g: Geometry, dt: mlx.mlx_dtype) !void {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ g.batch, 1, g.heads, g.head_dim }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ g.batch, g.heads, g.head_dim, g.state }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32, g.head_dim, g.batch * g.heads));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 8, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
    inline for (.{ .{ "Dh", g.head_dim }, .{ "Ds", g.state }, .{ "H", g.heads }, .{ "G", @divExact(g.heads, g.groups) } }) |kv|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, kv[0], kv[1]));
    if (cfg.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    cfg = c;
    cfg_key = .{ .g = g, .dt = dt };
}

/// Null when the geometry, dtypes or stream are outside the kernel (the
/// caller keeps the op chain).
pub fn step(g: Geometry, in: Inputs, s: mlx.mlx_stream) !?Outputs {
    if (!mlx.streamIsGpu(s)) return null;
    if (@rem(g.state, 32) != 0 or @rem(g.head_dim, 8) != 0 or @rem(g.heads, g.groups) != 0) return null;
    const dt = mlx.mlx_array_dtype(in.x);
    if (dt != .bfloat16 and dt != .float16) return null;
    for ([_]mlx.mlx_array{ in.B, in.C, in.dt_raw, in.A_log, in.dt_bias, in.D }) |arr|
        if (mlx.mlx_array_dtype(arr) != dt) return null;
    if (mlx.mlx_array_dtype(in.state) != .float32) return null;
    if (kernel_cache == null) {
        const ins = [_][*:0]const u8{ "X", "B", "C", "dt_raw", "A_log", "dt_bias", "D", "state_in", "dt_lo", "dt_hi" };
        const outs = [_][*:0]const u8{ "out", "state_out" };
        const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const k = mlx.mlx_fast_metal_kernel_new("msv_mamba2_decode_step", in_vec, out_vec, SOURCE, HEADER, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        kernel_cache = k;
    }
    const key = CfgKey{ .g = g, .dt = dt };
    if (cfg_key == null or !std.meta.eql(cfg_key.?, key)) try buildConfig(g, dt);

    const arrs = [_]mlx.mlx_array{ in.x, in.B, in.C, in.dt_raw, in.A_log, in.dt_bias, in.D, in.state, in.dt_lo, in.dt_hi };
    const v = mlx.mlx_vector_array_new_data(&arrs, arrs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, kernel_cache.?, v, cfg, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    var state = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(state);
    try mlx.check(mlx.mlx_vector_array_get(&y, o, 0));
    try mlx.check(mlx.mlx_vector_array_get(&state, o, 1));
    return .{ .y = y, .state = state };
}
