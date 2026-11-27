//! A resident wide layer's FFN tail in one launch (`MLX_SERVE_DSV41_MOE_TAIL`, `kernels/moe_tail.metal`): the routed
//! MoE's combine (unsort, k-sum, shared add) and the hc expand of the new streams read the down rows and the old streams
//! once and write the new streams, so the combined `[L, d]` rows are neither written nor read back. The same words as
//! `moePrefill` then `hcExpand`.

const std = @import("std");
const env = @import("../../../env.zig");
const lev = @import("../levers.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const dsv41 = @import("../deepseek_v41.zig");

const A = mlx.mlx_array;

pub const source = kernelSource("plugins/deepseek_v41/kernels/moe_tail.metal");
pub const ins = [_][*:0]const u8{ "dn", "inv", "sh", "res", "post", "comb" };

var kernel: ?mlx.mlx_fast_metal_kernel = null;
/// The tail reading its down rows in pair order (`MoeParts.tok`).
var kernel_tok: ?mlx.mlx_fast_metal_kernel = null;
var engaged_logged: std.atomic.Value(bool) = .init(false);

/// The new streams `[L, hc, d]` after layer `li`'s MoE over the normed rows `x` (`img` its image rows) on the streams
/// `x1` with the split `sf`; null where the switch is off or the layer's experts are not the trunk's resident banks.
pub fn tail(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, li: usize, x: A, img: A, x1: A, sf: *const dsv41.Split) !?A {
    if (!lev.onUnlessZero(.MOE_TAIL)) return null;
    if (li >= m.n_layers or m.streamed != null or m.tp != null or mlx.getShape(x)[0] <= dsv41.region_rows) return null;
    const ly = &m.layers[li];
    if (ly.experts.ext != null or mlx.mlx_array_dtype(x) != m.act or mlx.mlx_array_dtype(x1) != m.act) return null;
    const r = try dsv41.route(sc, m, ly, x, img);
    const p = try dsv41.moePrefillParts(sc, m, ly, x, r.idx, r.wts, try dsv41.shared(sc, m, ly, x));
    return try launch(sc, p, x1, sf);
}

/// The tail kernel over the MoE's parts `p` and the streams `x1` `[L, hc, d]` (the switch unread).
pub fn launch(sc: *dsv41.Scope, p: dsv41.MoeParts, x1: A, sf: *const dsv41.Split) !A {
    const sh = mlx.getShape(x1);
    const L = sh[0];
    const d = sh[2];
    const k = @divExact(@as(c_int, @intCast(mlx.mlx_array_size(p.inv))), L);
    if (kernel == null) kernel = try dsv41.buildKernel("dsv41_moe_tail", &ins, &.{"out"}, source, "");
    if (p.tok and kernel_tok == null) kernel_tok = try dsv41.buildKernel("dsv41_moe_tail_tok", &ins, &.{"out"}, source, "#define TOK 1\n");
    if (!engaged_logged.swap(true, .monotonic)) log.info(lev.line(.MOE_TAIL) ++ " (rows {d})\n", .{L});
    const dt = mlx.mlx_array_dtype(x1);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, sh.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(d, 4), L, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(256, @divTrunc(d, 4)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", sh[1]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", d));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", k));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ p.dn, p.inv, p.sh, x1, sf.post, sf.comb }, 6);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, if (p.tok) kernel_tok.? else kernel.?, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}
