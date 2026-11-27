//! A prompt call's routed pairs sorted by expert as a counting sort (`MLX_SERVE_DSV41_PF_MOE_EPI`): the SDK's prefill
//! route pack (histogram, scan, scatter over 256-pair blocks, `verify_kernels.moePrefillRoutePack`) gives each pair's
//! place, the sorted experts and their rows in three launches, and one scatter puts the routing weights in that order.
//! `moePrefill` otherwise makes them with two argsorts (a merge sort of about a dozen narrow launches each), a divide
//! and two takes. Ties keep pair order, as MLX's argsort does, so the outputs are the same words.
//! `MLX_SERVE_DSV41_PF_ROUTE_PACK` (on by default): the router's top-6 on a simdgroup a row (`routeRows`) and the pack's scan on a
//! whole threadgroup (`packFast`), the same words.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const env = @import("../../../env.zig");
const lev = @import("../levers.zig");
const log = @import("../../../sdk/util/log.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const vk = @import("../../../sdk/blocks/moe/verify_kernels.zig");
const dsv41 = @import("../deepseek_v41.zig");

const A = mlx.mlx_array;

/// The pairs in expert order: `inv[p]` is pair p's place, `rows` each place's token row, `sidx` its expert, `w` its
/// f32 routing weight.
pub const Sorted = struct { inv: A, rows: A, sidx: A, w: A };

var engaged_logged: std.atomic.Value(bool) = .init(false);

/// `flat` `[n]` uint32 expert ids of `n / k` rows (`k` a row), `wts` `[n]` f32, ids under `experts`; null where the
/// pack declines (decode widths, more than 1024 experts). The caller reads the switch.
pub fn pack(sc: *dsv41.Scope, flat: A, wts: A, k: usize, experts: usize) !?Sorted {
    var p = (try vk.moePrefillRoutePack(sc.s, flat, @intCast(k), @intCast(experts))) orelse return null;
    const inv = sc.keep(p.inverse) catch |e| {
        p.inverse = .{};
        p.deinit();
        return e;
    };
    p.inverse = .{};
    const sidx = sc.keep(p.sorted) catch |e| {
        p.sorted = .{};
        p.deinit();
        return e;
    };
    p.sorted = .{};
    return try placed(sc, flat, wts, inv, try sc.keep(p.lhs), sidx, experts);
}

/// Each pair's weight to its place: `inv` is a permutation, so every place is written once.
fn placed(sc: *dsv41.Scope, flat: A, wts: A, inv: A, rows: A, sidx: A, experts: usize) !Sorted {
    const n: c_int = @intCast(mlx.mlx_array_size(flat));
    var o = mlx.mlx_array_new();
    const w = try sc.res(mlx.mlx_put_along_axis(&o, try sc.zeros(&.{n}, .float32), inv, wts, 0, sc.s), &o);
    if (!engaged_logged.swap(true, .monotonic)) log.info(lev.line(.PF_MOE_EPI) ++ " (pairs {d}, experts {d})\n", .{ n, experts });
    return .{ .inv = inv, .rows = rows, .sidx = sidx, .w = w };
}

pub fn on() bool {
    return lev.onUnlessZero(.PF_ROUTE_PACK);
}

var pack_logged: std.atomic.Value(bool) = .init(false);
/// `MLX_SERVE_DSV41_PF_ROUTE_PACK`'s calls served (the tests' engagement count): top-6 and scan each count one.
pub var served = std.atomic.Value(usize).init(0);

fn logged(rows: usize, experts: usize) void {
    _ = served.fetchAdd(1, .monotonic);
    if (!pack_logged.swap(true, .monotonic)) log.info(lev.line(.PF_ROUTE_PACK) ++ " (rows {d}, experts {d})\n", .{ rows, experts });
}

pub const ROWS_SOURCE = kernelSource("plugins/deepseek_v41/kernels/pf_route_rows.metal");
pub const SCAN_SOURCE = kernelSource("plugins/deepseek_v41/kernels/pf_route_scan.metal");
pub const ROWS_INS = [_][*:0]const u8{ "logits", "bias", "scal", "dsc" };
pub const ROWS_OUTS = [_][*:0]const u8{ "idx", "wts" };
pub const SCAN_INS = [_][*:0]const u8{"counts"};
pub const SCAN_OUTS = [_][*:0]const u8{"offsets"};
pub const HIST_INS = [_][*:0]const u8{"ids"};
pub const HIST_OUTS = [_][*:0]const u8{"counts"};
pub const SCATTER_SOURCE = kernelSource("plugins/deepseek_v41/kernels/pf_route_scatter.metal");
pub const SCATTER_INS = [_][*:0]const u8{ "ids", "offsets", "wts" };
pub const SCATTER_OUTS = [_][*:0]const u8{ "inverse", "sorted", "lhs", "w" };
/// Rows (simdgroups) a threadgroup of the top-6.
pub const rows_sg = 8;
/// The route pack's pair block (the SDK's).
pub const block = 256;
pub const scan_threads = 1024;

var rows_kernel: ?mlx.mlx_fast_metal_kernel = null;
var scan_kernel: ?mlx.mlx_fast_metal_kernel = null;
var hist_kernel: ?mlx.mlx_fast_metal_kernel = null;
var scatter_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn kern(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    slot.* = try dsv41.buildKernel(name, ins, outs, src, "");
    return slot.*.?;
}

/// Where the top-6 kernel serves: a prompt's rows (past verify widths), whole float4 runs of experts, at most 16 picks.
pub fn rowsServes(rows: usize, experts: usize, k: usize) bool {
    return rows > 32 and experts % 128 == 0 and experts <= 1024 and k >= 1 and k <= 16;
}

/// route.metal's picks and weights of `logits` `[L, E]` f32 (`bias` `[1 or L, E]`, `scal` `[1]`) on a simdgroup a
/// row, each weight times its expert's down scale `dsc` `[E]` where given (`routeScaled`); null where `rowsServes`
/// declines. `plant` (tests) puts the last pick first.
pub fn routeRows(sc: *dsv41.Scope, logits: A, bias: A, scal: A, dsc: ?A, k: usize, norm: bool, plant: bool) !?[2]A {
    const sh = mlx.getShape(logits);
    if (sh.len != 2) return null;
    const L: usize = @intCast(sh[0]);
    const E: usize = @intCast(sh[1]);
    if (!rowsServes(L, E, k) or mlx.mlx_array_dtype(logits) != .float32 or mlx.mlx_array_dtype(bias) != .float32) return null;
    const kr = try kern(&rows_kernel, "dsv41_pf_route_rows", &ROWS_INS, &ROWS_OUTS, ROWS_SOURCE);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(k) }, 2, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(k) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci((L + rows_sg - 1) / rows_sg * 32 * rows_sg), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32 * rows_sg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "E", ci(E)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", ci(k)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "NORM", norm));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SG", rows_sg));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "PLANT", @intFromBool(plant)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "DS", @intFromBool(dsc != null)));
    const outs = try apply(sc, kr, &.{ logits, bias, scal, dsc orelse scal }, cfg, 2);
    logged(L, E);
    return outs[0..2].*;
}

/// `pack`'s outputs with the scan on a whole threadgroup, and the weights put by the scatter: the SDK's histogram,
/// `pf_route_scan`, `pf_route_scatter`.
/// `plant` (tests) adds one to the first offset.
pub fn packFast(sc: *dsv41.Scope, flat: A, wts: A, k: usize, experts: usize, plant: bool) !?Sorted {
    const n: usize = mlx.mlx_array_size(flat);
    if (mlx.mlx_array_ndim(flat) != 1 or mlx.mlx_array_dtype(flat) != .uint32 or k == 0 or n % k != 0 or n / k <= 32) return null;
    if (experts % 128 != 0 or experts > 1024) return null;
    const nb = (n + block - 1) / block;
    const table = [_]c_int{ci(nb * experts)};
    const tmpl = [_]struct { [*:0]const u8, usize }{ .{ "E", experts }, .{ "TOPK", k } };
    const counts = blk: {
        const kh = try kern(&hist_kernel, "prefill_route_hist", &HIST_INS, &HIST_OUTS, vk.PREFILL_ROUTE_HIST_SOURCE);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &table, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(nb * block), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, block, 1, 1));
        for (tmpl) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], ci(t[1])));
        break :blk (try apply(sc, kh, &.{flat}, cfg, 1))[0];
    };
    const offsets = try scan(sc, counts, experts, plant);
    const ks = try kern(&scatter_kernel, "dsv41_pf_route_scatter", &SCATTER_INS, &SCATTER_OUTS, SCATTER_SOURCE);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (0..3) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ci(n)}, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ci(n)}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(nb * block), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, block, 1, 1));
    for (tmpl) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], ci(t[1])));
    if (mlx.mlx_array_dtype(wts) != .float32) return null;
    const o = try apply(sc, ks, &.{ flat, offsets, wts }, cfg, 4);
    logged(n / k, experts);
    return .{ .inv = o[0], .rows = o[2], .sidx = o[1], .w = o[3] };
}

/// The offsets of the per-block histograms `counts` `[nb * experts]`: prefill_route_scan.metal's words.
pub fn scan(sc: *dsv41.Scope, counts: A, experts: usize, plant: bool) !A {
    const kc = try kern(&scan_kernel, "dsv41_pf_route_scan", &SCAN_INS, &SCAN_OUTS, SCAN_SOURCE);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ci(mlx.mlx_array_size(counts))}, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, scan_threads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, scan_threads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "E", ci(experts)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "PLANT", @intFromBool(plant)));
    return (try apply(sc, kc, &.{counts}, cfg, 1))[0];
}

fn apply(sc: *dsv41.Scope, k: mlx.mlx_fast_metal_kernel, ins: []const A, cfg: mlx.mlx_fast_metal_kernel_config, comptime n: usize) ![n]A {
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, sc.s));
    var out: [n]A = undefined;
    for (&out, 0..) |*o, i| {
        var a = mlx.mlx_array_new();
        o.* = try sc.res(mlx.mlx_vector_array_get(&a, ov, i), &a);
    }
    return out;
}

fn ci(v: usize) c_int {
    return @intCast(v);
}
