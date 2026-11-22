//! A wide layer's FFN-side hc mixes on the simdgroup matrix units (`MLX_SERVE_DSV41_HC_MIX_TILED`,
//! `kernels/hc_mix_mma.metal`): hc_mix's z in f32 with one read of each stream row for all 24 mixes and one read of
//! `fn` per 32 rows, both staged through threadgroup memory in whole-row loads, where hc_mix reads a row once per group
//! of 6 mixes and `fn` (24 x 20480 f32) once per 4 rows. The same sums in another order. The attention side keeps the
//! fused mix and norm (`hcMixNormFused`).

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const dsv41 = @import("../deepseek_v41.zig");

const A = mlx.mlx_array;

pub const source = kernelSource("plugins/deepseek_v41/kernels/hc_mix_mma.metal");
pub const ins = [_][*:0]const u8{ "x", "fn", "scale", "base", "eps" };
/// 8-row tiles a threadgroup, one a simdgroup; the columns a staging step.
pub const RT = 4;
pub const KC = 64;

var kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged_logged: std.atomic.Value(bool) = .init(false);

/// Whether the kernel takes mixes `[mix, K]` f32 over rows of `K` values: whole 8-wide mix tiles (at most 3), K in
/// whole staging steps.
pub fn fits(fn_w: A, x: A) bool {
    const fs = mlx.getShape(fn_w);
    if (fs.len != 2 or mlx.mlx_array_dtype(fn_w) != .float32) return false;
    const nmix = fs[0];
    const K = fs[1];
    if (@rem(nmix, 8) != 0 or @divExact(nmix, 8) > 3 or @rem(K, KC) != 0) return false;
    const xs = mlx.getShape(x);
    return xs.len > 0 and xs[0] > 0 and mlx.mlx_array_size(x) == @as(usize, @intCast(xs[0])) * @as(usize, @intCast(K));
}

/// `hcMixWide`'s z `[L, mix]` f32 of the stream `x` `[L, ...]`; null where the switch is off or the shapes do not fit.
pub fn mix(sc: *dsv41.Scope, x: A, hc: *const dsv41.Hc, eps: A) !?A {
    if (!env.on(.MLX_SERVE_DSV41_HC_MIX_TILED) or !fits(hc.fn_w, x)) return null;
    return try launch(sc, x, hc, eps);
}

/// The launch itself (the switch unread), for the tests and the contract.
pub fn launch(sc: *dsv41.Scope, x: A, hc: *const dsv41.Hc, eps: A) !A {
    const L: usize = @intCast(mlx.getShape(x)[0]);
    const nmix = mlx.getShape(hc.fn_w)[0];
    if (kernel == null) kernel = try dsv41.buildKernel("dsv41_hc_mix_mma", &ins, &.{"z"}, source, "");
    if (!engaged_logged.swap(true, .monotonic)) log.info("[dsv41] HC_MIX_TILED engaged (rows {d})\n", .{L});
    const rows = RT * 8;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ @intCast(L), nmix }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast((L + rows - 1) / rows * RT * 32), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, RT * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RT", RT));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "MT", @divExact(nmix, 8)));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ x, hc.fn_w, hc.scale, hc.base, eps }, 5);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel.?, iv, cfg, sc.s));
    var z = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&z, ov, 0), &z);
}
