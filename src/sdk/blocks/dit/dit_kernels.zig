//! Fused glue for image/video DiT blocks (Krea, Qwen-Image).
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");

/// q/k/v of one fused projection `y` [L, NT] (columns `q_off`, `k_off`, `v_off`,
/// 128 per head), per-head RMS norm and complex-pair RoPE on q and k, written
/// heads-major for SDPA: q [1, heads, L, 128], k and v [1, kv_heads, L, 128].
/// One launch for what was three column copies, two norms, two RoPE chains and
/// three transposes. `round_norm` rounds the normalized value to y's dtype
/// before RoPE, as an unfused chain in that dtype does.
pub const QkvHeads = struct { q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array };

pub const QKV_SOURCE = kernelSource("kernels/dit/dit_qkv.metal");

var qkv_kernel: ?mlx.mlx_fast_metal_kernel = null;

pub const QkvLayout = struct { heads: c_int, kv_heads: c_int, q_off: c_int, k_off: c_int, v_off: c_int };

/// `y` [L, NT] row-contiguous in a float dtype; `qn`/`kn` [128] (any float);
/// `cos`/`sin` f32 holding [L, 64] contiguously.
pub fn qkvHeads(y: mlx.mlx_array, lay: QkvLayout, qn: mlx.mlx_array, kn: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, eps: f32, round_norm: bool, s: mlx.mlx_stream) !QkvHeads {
    const L = mlx.getShape(y)[0];
    const dt = mlx.mlx_array_dtype(y);
    if (qkv_kernel == null) {
        const ins = [_][*:0]const u8{ "y", "qn", "kn", "cs", "sn", "eps" };
        const outs = [_][*:0]const u8{ "q", "k", "vo" };
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const kk = mlx.mlx_fast_metal_kernel_new("msv_dit_qkv_heads", iv, ov, QKV_SOURCE, "", true, false);
        if (kk.ctx == null) return error.MetalKernelCompileFailed;
        qkv_kernel = kk;
    }
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, lay.heads, L, 128 }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, lay.kv_heads, L, 128 }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, lay.kv_heads, L, 128 }, 4, dt));
    const slots = lay.heads + 2 * lay.kv_heads;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 256 * @divTrunc(slots + 7, 8), L, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "W", mlx.mlx_array_dtype(qn)));
    inline for (.{ .{ "H", lay.heads }, .{ "KV", lay.kv_heads }, .{ "QO", lay.q_off }, .{ "KO", lay.k_off }, .{ "VO", lay.v_off } }) |kv|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "ROUND", round_norm));
    const e = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(e);
    const inputs = [_]mlx.mlx_array{ y, qn, kn, cos, sin, e };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, qkv_kernel.?, iv, cfg, s));
    var r: [3]mlx.mlx_array = undefined;
    for (&r, 0..) |*a, i| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(a, outs, i));
    }
    return .{ .q = r[0], .k = r[1], .v = r[2] };
}
