//! GroupNorm over channels-last [B, H, W, C], two reduction layouts that are not bit-equal:
//! - `groupNormFlat`: f32 statistics over each group flattened to one axis (transpose then
//!   one mean), f32 affine, cast to `out`. FLUX.2 VAE (bf16), Hunyuan3D paint VAE and UNet (f16).
//! - `groupNormAxes`: statistics over (H·W, C/G) as two axes in the input dtype, affine in it.
//!   MOSS-Image's f32 VAE; mage_flow's VAE is the same after its f32 cast.
//! - `groupNormSilu`: `groupNormFlat` then SiLU in three passes on the GPU (opt-in: the FLUX.2 VAE encoder).
//! ltx and minimax_h3 normalize 5-D video per (frame, group) and pixal3d nests its means:
//! different reductions, kept in their backends.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");

const Graph = ops.Graph;
const Array = ops.Array;
const Dtype = ops.Dtype;

pub fn groupNormFlat(parent: *Graph, x: Array, w: Array, b: Array, groups: c_int, eps: f32, out: Dtype) !Array {
    var g = parent.scope();
    defer g.deinit();
    const sh = ops.shape(x);
    const B, const H, const W, const C = .{ sh[0], sh[1], sh[2], sh[3] };
    const cg = @divExact(C, groups);
    const grouped = try g.transpose(try g.reshape(try g.astype(x, .float32), &.{ B, H * W, groups, cg }), &.{ 0, 2, 1, 3 });
    const flat = try g.reshape(grouped, &.{ B, groups, H * W * cg });
    const xc = try g.sub(flat, try g.mean(flat, -1, true));
    const v = try g.mean(try g.mul(xc, xc), -1, true);
    const n = try g.mul(xc, try g.rsqrt(try g.add(v, try g.scalar(eps, .float32))));
    const back = try g.reshape(try g.transpose(try g.reshape(n, &.{ B, groups, H * W, cg }), &.{ 0, 2, 1, 3 }), &.{ B, H, W, C });
    const y = try g.add(try g.mul(back, try g.astype(w, .float32)), try g.astype(b, .float32));
    return parent.keep(try g.astype(y, out));
}

pub const STATS_SOURCE = kernelSource("kernels/norms/group_norm_stats.metal");
pub const SILU_SOURCE = kernelSource("kernels/norms/group_norm_silu.metal");
/// Pixels per threadgroup of the statistics pass.
pub const PIX: c_int = 1024;
var stats_kernel: ?mlx.mlx_fast_metal_kernel = null;
var silu_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// `silu(groupNormFlat(x))` in three passes on the GPU: per-chunk sums shifted by each group's first value, the
/// groups' mean and rstd from them, then the affine and SiLU in f32 rounded once to `out` (the chain rounds the norm to
/// `out` before its SiLU). x `[B, H, W, C]` f16, bf16 or f32, C / groups a multiple of 4, C at most 1024; null where
/// the shape or stream is outside the kernels.
pub fn groupNormSilu(parent: *Graph, x: Array, w: Array, b: Array, groups: c_int, eps: f32, out: Dtype) !?Array {
    if (!mlx.streamIsGpu(parent.s)) return null;
    const dt = ops.dtype(x);
    if (dt != .float16 and dt != .bfloat16 and dt != .float32) return null;
    if (out != .float16 and out != .bfloat16 and out != .float32) return null;
    const sh = ops.shape(x);
    if (sh.len != 4 or groups <= 0 or @rem(sh[3], groups) != 0) return null;
    const bn, const hw, const c = .{ sh[0], sh[1] * sh[2], sh[3] };
    const cg = @divExact(c, groups);
    if (@rem(cg, 4) != 0 or c > 1024) return null;
    var g = parent.scope();
    defer g.deinit();
    const xc = try g.contiguous(x);
    // Each group's first value, the pivot its sums are taken about.
    const pv = try g.reshape(try g.astype(try g.call(mlx.mlx_slice, .{ xc, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ bn, 1, 1, c }, 4, &[_]c_int{ 1, 1, 1, cg }, 4 }), .float32), &.{ bn, groups });
    const sums = try g.sum(try statsPass(&g, xc, pv, groups), 1, false);
    const n = try g.scalar(@floatFromInt(hw * cg), .float32);
    const m1 = try g.div(try g.sliceAxis(sums, 2, 0, 1), n);
    const var_ = try g.maximum(try g.sub(try g.div(try g.sliceAxis(sums, 2, 1, 2), n), try g.mul(m1, m1)), try g.scalar(0, .float32));
    const mean = try g.add(try g.reshape(pv, &.{ bn, groups, 1 }), m1);
    const rstd = try g.rsqrt(try g.add(var_, try g.scalar(eps, .float32)));
    const st = try g.contiguous(try g.concat(&.{ mean, rstd }, 2));
    return try parent.keep(try siluPass(&g, xc, st, try g.astype(w, .float32), try g.astype(b, .float32), groups, out));
}

/// Pass 1 (`STATS_SOURCE`): `[B, chunks, groups, 2]` f32 per-chunk sums of x - pv and their squares; x contiguous.
pub fn statsPass(g: *Graph, x: Array, pv: Array, groups: c_int) !Array {
    const sh = ops.shape(x);
    const nch = @divTrunc(sh[1] * sh[2] + PIX - 1, PIX);
    if (stats_kernel == null) stats_kernel = try newKernel("msv_group_norm_stats", &.{ "X", "PV" }, &.{"OP"}, STATS_SOURCE);
    return launch(g, stats_kernel.?, &.{ x, pv }, &.{ sh[0], nch, groups, 2 }, .float32, .{ nch * 256, sh[0] }, 256, &.{ .{ "T", .{ .dtype = ops.dtype(x) } }, .{ "GROUPS", .{ .int = groups } }, .{ "PIX", .{ .int = PIX } } });
}

/// Pass 2 (`SILU_SOURCE`): silu((x - mean) * rstd * w + b) from `st` `[B, groups, 2]` f32, `w` and `b` f32.
pub fn siluPass(g: *Graph, x: Array, st: Array, w: Array, b: Array, groups: c_int, out: Dtype) !Array {
    const sh = ops.shape(x);
    if (silu_kernel == null) silu_kernel = try newKernel("msv_group_norm_silu", &.{ "X", "ST", "W", "BI" }, &.{"O"}, SILU_SOURCE);
    const total: c_int = @intCast(mlx.mlx_array_size(x));
    return launch(g, silu_kernel.?, &.{ x, st, w, b }, sh, out, .{ @divExact(total, 4), 1 }, 256, &.{ .{ "T", .{ .dtype = ops.dtype(x) } }, .{ "OT", .{ .dtype = out } }, .{ "GROUPS", .{ .int = groups } } });
}

fn newKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

const TArg = union(enum) { dtype: Dtype, int: c_int };

/// One output of `kernel` over a `grid` of threads (x, y) in threadgroups of `tg`, owned by `g`.
fn launch(g: *Graph, kernel: mlx.mlx_fast_metal_kernel, inputs: []const Array, shape: []const c_int, dt: Dtype, grid: [2]c_int, tg: c_int, targs: []const struct { [*:0]const u8, TArg }) !Array {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, shape.ptr, shape.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, grid[0], grid[1], 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1));
    for (targs) |t| switch (t[1]) {
        .dtype => |d| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, t[0], d)),
        .int => |v| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], v)),
    };
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, g.s));
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_vector_array_get(&o, ov, 0));
    return g.adopt(o);
}

pub fn groupNormAxes(parent: *Graph, x: Array, w: Array, b: Array, groups: c_int, eps: f32) !Array {
    var g = parent.scope();
    defer g.deinit();
    const sh = ops.shape(x);
    const grouped = try g.reshape(x, &.{ sh[0], sh[1] * sh[2], groups, @divExact(sh[3], groups) });
    const axes = [_]c_int{ 1, 3 };
    const xc = try g.sub(grouped, try g.call(mlx.mlx_mean_axes, .{ grouped, &axes, axes.len, true }));
    const v = try g.call(mlx.mlx_mean_axes, .{ try g.mul(xc, xc), &axes, axes.len, true });
    const n = try g.reshape(try g.mul(xc, try g.rsqrt(try g.add(v, try g.scalar(eps, .float32)))), sh);
    return parent.keep(try g.add(try g.mul(n, w), b));
}

// ── tests ──

const testing = std.testing;
const kit = @import("../dit/testkit.zig");
const scope_mod = @import("../../../mlx/scope.zig");
const S = kit.S;
const A = kit.A;
const astype = kit.astype;
const reshape = kit.reshape;
const transpose = kit.transpose;
const subA = kit.subA;
const mulA = kit.mulA;
const addA = kit.addA;
const meanAxes = kit.meanAxes;
const scalarF = kit.scalarF;

/// flux.zig `groupNorm` (eps 1e-6, bf16 out).
fn fluxGroupNorm(x: mlx.mlx_array, weight: mlx.mlx_array, bias: mlx.mlx_array, groups: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [1,H,W,C]
    const H = sh[1];
    const Wd = sh[2];
    const C = sh[3];
    const cg = @divExact(C, groups);
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const r1 = try reshape(xf, &[_]c_int{ 1, H * Wd, groups, cg }, s);
    defer _ = mlx.mlx_array_free(r1);
    const t1 = try transpose(r1, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(t1);
    const flat = try reshape(t1, &[_]c_int{ 1, groups, H * Wd * cg }, s);
    defer _ = mlx.mlx_array_free(flat);
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axis(&mean, flat, -1, true, s));
    const xc = try subA(flat, mean, s);
    defer _ = mlx.mlx_array_free(xc);
    const sq = try mulA(xc, xc, s);
    defer _ = mlx.mlx_array_free(sq);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_mean_axis(&v, sq, -1, true, s));
    const epsa = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(epsa);
    var ve = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ve);
    try mlx.check(mlx.mlx_add(&ve, v, epsa, s));
    var rsq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rsq);
    try mlx.check(mlx.mlx_rsqrt(&rsq, ve, s));
    const norm = try mulA(xc, rsq, s);
    defer _ = mlx.mlx_array_free(norm);
    const b1 = try reshape(norm, &[_]c_int{ 1, groups, H * Wd, cg }, s);
    defer _ = mlx.mlx_array_free(b1);
    const b2 = try transpose(b1, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(b2);
    const b3 = try reshape(b2, &[_]c_int{ 1, H, Wd, C }, s);
    defer _ = mlx.mlx_array_free(b3);
    const wf = try astype(weight, .float32, s);
    defer _ = mlx.mlx_array_free(wf);
    const bf = try astype(bias, .float32, s);
    defer _ = mlx.mlx_array_free(bf);
    const sc = try mulA(b3, wf, s);
    defer _ = mlx.mlx_array_free(sc);
    const out = try addA(sc, bf, s);
    defer _ = mlx.mlx_array_free(out);
    return astype(out, .bfloat16, s);
}

/// hunyuan3d paint_unet.zig `groupNorm` (paint.zig's is this with B = 1).
fn paintUnetGroupNorm(x: mlx.mlx_array, weight: mlx.mlx_array, bias: mlx.mlx_array, groups: c_int, eps: f32, s: S) !mlx.mlx_array {
    const rel = struct {
        fn f(a: mlx.mlx_array) void {
            _ = mlx.mlx_array_free(a);
        }
    }.f;
    const sh = mlx.getShape(x);
    const B = sh[0];
    const H = sh[1];
    const Wd = sh[2];
    const C = sh[3];
    const cg = @divExact(C, groups);
    const xf = try astype(x, .float32, s);
    defer rel(xf);
    const r1 = try reshape(xf, &[_]c_int{ B, H * Wd, groups, cg }, s);
    defer rel(r1);
    const t1 = try transpose(r1, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer rel(t1);
    const flat = try reshape(t1, &[_]c_int{ B, groups, H * Wd * cg }, s);
    defer rel(flat);
    var mean = mlx.mlx_array_new();
    defer rel(mean);
    try mlx.check(mlx.mlx_mean_axis(&mean, flat, -1, true, s));
    const xc = try subA(flat, mean, s);
    defer rel(xc);
    const sq = try mulA(xc, xc, s);
    defer rel(sq);
    var v = mlx.mlx_array_new();
    defer rel(v);
    try mlx.check(mlx.mlx_mean_axis(&v, sq, -1, true, s));
    const epsa = mlx.mlx_array_new_float(eps);
    defer rel(epsa);
    var ve = mlx.mlx_array_new();
    defer rel(ve);
    try mlx.check(mlx.mlx_add(&ve, v, epsa, s));
    var rsq = mlx.mlx_array_new();
    defer rel(rsq);
    try mlx.check(mlx.mlx_rsqrt(&rsq, ve, s));
    const norm = try mulA(xc, rsq, s);
    defer rel(norm);
    const b1 = try reshape(norm, &[_]c_int{ B, groups, H * Wd, cg }, s);
    defer rel(b1);
    const b2 = try transpose(b1, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer rel(b2);
    const b3 = try reshape(b2, &[_]c_int{ B, H, Wd, C }, s);
    defer rel(b3);
    const wf = try astype(weight, .float32, s);
    defer rel(wf);
    const bf = try astype(bias, .float32, s);
    defer rel(bf);
    const sc = try mulA(b3, wf, s);
    defer rel(sc);
    const out = try addA(sc, bf, s);
    defer rel(out);
    return astype(out, .float16, s);
}

/// moss_image.zig `Vae.groupNorm` (32 groups, eps 1e-6, Scope).
fn mossGroupNorm(sc: *scope_mod.Scope, x: A, w: A, b: A) !A {
    const sh = mlx.getShape(x);
    const g = try sc.reshape(x, &.{ sh[0], sh[1] * sh[2], 32, @divExact(sh[3], 32) });
    const axes = [_]c_int{ 1, 3 };
    var om = mlx.mlx_array_new();
    const mean = try sc.res(mlx.mlx_mean_axes(&om, g, &axes, 2, true, sc.s), &om);
    const xc = try sc.sub(g, mean);
    var ov = mlx.mlx_array_new();
    const variance = try sc.res(mlx.mlx_mean_axes(&ov, try sc.mul(xc, xc), &axes, 2, true, sc.s), &ov);
    var orr = mlx.mlx_array_new();
    const inv = try sc.res(mlx.mlx_rsqrt(&orr, try sc.addS(variance, 1e-6), sc.s), &orr);
    return sc.add(try sc.mul(try sc.reshape(try sc.mul(xc, inv), sh), w), b);
}

/// mage_flow.zig `groupNorm`.
fn mageGroupNorm(x: mlx.mlx_array, weight: mlx.mlx_array, bias: mlx.mlx_array, num_groups: c_int, eps: f32, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [B,H,W,C]
    const B = sh[0];
    const H = sh[1];
    const Wd = sh[2];
    const C = sh[3];
    const cg = @divExact(C, num_groups);
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const grouped = try reshape(xf, &[_]c_int{ B, H, Wd, num_groups, cg }, s);
    defer _ = mlx.mlx_array_free(grouped);
    const red = [_]c_int{ 1, 2, 4 };
    const mean = try meanAxes(grouped, &red, true, s);
    defer _ = mlx.mlx_array_free(mean);
    const centered = try subA(grouped, mean, s);
    defer _ = mlx.mlx_array_free(centered);
    const sq = try mulA(centered, centered, s);
    defer _ = mlx.mlx_array_free(sq);
    const variance = try meanAxes(sq, &red, true, s);
    defer _ = mlx.mlx_array_free(variance);
    const epsA = scalarF(eps);
    defer _ = mlx.mlx_array_free(epsA);
    const vpe = try addA(variance, epsA, s);
    defer _ = mlx.mlx_array_free(vpe);
    var rstd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rstd);
    try mlx.check(mlx.mlx_rsqrt(&rstd, vpe, s));
    const normed = try mulA(centered, rstd, s);
    defer _ = mlx.mlx_array_free(normed);
    const back = try reshape(normed, &[_]c_int{ B, H, Wd, C }, s);
    defer _ = mlx.mlx_array_free(back);
    const scaled = try mulA(back, weight, s);
    defer _ = mlx.mlx_array_free(scaled);
    return addA(scaled, bias, s);
}

test "media blocks: groupNormFlat / groupNormAxes are bit-equal to the backends' GroupNorms" {
    const s = kit.cpu();
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    for ([_]Dtype{ .bfloat16, .float16, .float32 }, 0..) |dt, i| {
        const x = try kit.randn(&g, &.{ 1, 6, 5, 64 }, 50 + i, dt, 2);
        const w = try kit.randn(&g, &.{64}, 60 + i, dt, 1);
        const b = try kit.randn(&g, &.{64}, 70 + i, dt, 1);
        try ops.expectBitEqual(try groupNormFlat(&g, x, w, b, 32, 1e-6, .bfloat16), try kit.own(&g, try fluxGroupNorm(x, w, b, 32, s)));
        try ops.expectBitEqual(try groupNormFlat(&g, x, w, b, 32, 1e-5, .float16), try kit.own(&g, try paintUnetGroupNorm(x, w, b, 32, 1e-5, s)));
        const x2 = try kit.randn(&g, &.{ 2, 4, 3, 64 }, 80 + i, dt, 2);
        try ops.expectBitEqual(try groupNormFlat(&g, x2, w, b, 16, 1e-5, .float16), try kit.own(&g, try paintUnetGroupNorm(x2, w, b, 16, 1e-5, s)));
    }
    // The axes layout runs in its input dtype: MOSS's VAE and mage_flow's are f32.
    const x = try kit.randn(&g, &.{ 2, 6, 5, 64 }, 90, .float32, 2);
    const w = try kit.randn(&g, &.{64}, 91, .float32, 1);
    const b = try kit.randn(&g, &.{64}, 92, .float32, 1);
    const axes = try groupNormAxes(&g, x, w, b, 32, 1e-6);
    {
        var sc = scope_mod.Scope.init(testing.allocator, s);
        defer sc.deinit();
        try ops.expectBitEqual(axes, try mossGroupNorm(&sc, x, w, b));
    }
    try ops.expectBitEqual(axes, try kit.own(&g, try mageGroupNorm(x, w, b, 32, 1e-6, s)));
}

test "group norm silu: the fused passes match silu(groupNormFlat) to the output's rounding, over chunks and images" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    // 37 x 29 pixels: two chunks, the second ragged; a mean far from zero tests the pivot.
    for ([_]struct { c: c_int, dt: Dtype, out: Dtype, bar: f32 }{
        .{ .c = 128, .dt = .bfloat16, .out = .bfloat16, .bar = 4e-3 },
        .{ .c = 512, .dt = .float32, .out = .float32, .bar = 2e-6 },
    }, 0..) |case, i| {
        const x = try g.addScalar(try kit.randn(&g, &.{ 2, 37, 29, case.c }, 100 + i, case.dt, 2), 30);
        const w = try kit.randn(&g, &.{case.c}, 110 + i, case.dt, 1);
        const b = try kit.randn(&g, &.{case.c}, 120 + i, case.dt, 1);
        const fused = (try groupNormSilu(&g, x, w, b, 32, 1e-6, case.out)).?;
        try testing.expectEqual(case.out, ops.dtype(fused));
        const chain = try g.silu(try groupNormFlat(&g, x, w, b, 32, 1e-6, .float32));
        const d = try g.sub(try g.astype(fused, .float32), chain);
        const err = try g.sqrt(try g.div(try g.sum(try g.mul(d, d), -1, false), try g.sum(try g.mul(chain, chain), -1, false)));
        const worst = try g.call(mlx.mlx_max, .{ err, false });
        try ops.eval(worst);
        var v: f32 = 0;
        try mlx.check(mlx.mlx_array_item_float32(&v, worst));
        try testing.expect(v < case.bar);
    }
}
