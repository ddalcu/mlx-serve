//! LayerNorm over the last axis computed by hand in f32 (mean, centred square mean, + eps,
//! rsqrt), optional affine in f32, cast to `out`. FLUX.2's `layerNormNA` (out bf16) and
//! mage_flow's `layerNormLast` (out = the input dtype). `mlx_fast_layer_norm` with null or
//! all-ones weights (qwen_image, ltx, trellis2) is a different kernel and stays separate.

const std = @import("std");
const ops = @import("../../../mlx/ops.zig");

const Graph = ops.Graph;
const Array = ops.Array;
const Dtype = ops.Dtype;

pub fn layerNormF32(parent: *Graph, x: Array, w: ?Array, b: ?Array, eps: f32, out: Dtype) !Array {
    var g = parent.scope();
    defer g.deinit();
    const xf = try g.astype(x, .float32);
    const xc = try g.sub(xf, try g.mean(xf, -1, true));
    const v = try g.mean(try g.mul(xc, xc), -1, true);
    var y = try g.mul(xc, try g.rsqrt(try g.add(v, try g.scalar(eps, .float32))));
    if (w) |ww| {
        y = try g.mul(y, ww);
        if (b) |bb| y = try g.add(y, bb);
    }
    return parent.keep(try g.astype(y, out));
}

// ── tests ──

const testing = std.testing;
const mlx = @import("../../../mlx/ffi.zig");
const kit = @import("../dit/testkit.zig");
const S = kit.S;
const astype = kit.astype;
const meanLast = kit.meanLast;
const meanAxes = kit.meanAxes;
const subA = kit.subA;
const mulA = kit.mulA;
const addA = kit.addA;
const scalarF = kit.scalarF;

/// flux.zig `layerNormNA`.
fn fluxLayerNormNA(x: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const m = try meanLast(xf, s);
    defer _ = mlx.mlx_array_free(m);
    const xc = try subA(xf, m, s);
    defer _ = mlx.mlx_array_free(xc);
    const sq = try mulA(xc, xc, s);
    defer _ = mlx.mlx_array_free(sq);
    const v = try meanLast(sq, s);
    defer _ = mlx.mlx_array_free(v);
    const epsa = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(epsa);
    var ve = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ve);
    try mlx.check(mlx.mlx_add(&ve, v, epsa, s));
    var rs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rs);
    try mlx.check(mlx.mlx_rsqrt(&rs, ve, s));
    const out = try mulA(xc, rs, s);
    defer _ = mlx.mlx_array_free(out);
    return astype(out, .bfloat16, s);
}

/// mage_flow.zig `layerNormLast`.
fn mageLayerNormLast(x: mlx.mlx_array, weight: ?mlx.mlx_array, bias: ?mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    const in_dt = mlx.mlx_array_dtype(x);
    const nd = mlx.getShape(x).len;
    const last = [_]c_int{@intCast(nd - 1)};
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const mean = try meanAxes(xf, &last, true, s);
    defer _ = mlx.mlx_array_free(mean);
    const centered = try subA(xf, mean, s);
    defer _ = mlx.mlx_array_free(centered);
    const sq = try mulA(centered, centered, s);
    defer _ = mlx.mlx_array_free(sq);
    const variance = try meanAxes(sq, &last, true, s);
    defer _ = mlx.mlx_array_free(variance);
    const epsA = scalarF(eps);
    defer _ = mlx.mlx_array_free(epsA);
    const vpe = try addA(variance, epsA, s);
    defer _ = mlx.mlx_array_free(vpe);
    var rstd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rstd);
    try mlx.check(mlx.mlx_rsqrt(&rstd, vpe, s));
    var out = try mulA(centered, rstd, s); // f32
    if (weight) |wgt| {
        const scaled = try mulA(out, wgt, s);
        _ = mlx.mlx_array_free(out);
        out = scaled;
        if (bias) |b| {
            const shifted = try addA(out, b, s);
            _ = mlx.mlx_array_free(out);
            out = shifted;
        }
    }
    if (in_dt == .float32) return out;
    defer _ = mlx.mlx_array_free(out);
    return astype(out, in_dt, s);
}

test "media blocks: layerNormF32 is bit-equal to flux layerNormNA and mage_flow layerNormLast" {
    const s = kit.cpu();
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    for ([_]Dtype{ .bfloat16, .float32 }, 0..) |dt, i| {
        const x = try kit.randn(&g, &.{ 2, 9, 48 }, 20 + i, dt, 2);
        const w = try kit.randn(&g, &.{48}, 30 + i, dt, 1);
        const b = try kit.randn(&g, &.{48}, 40 + i, dt, 1);
        try ops.expectBitEqual(try layerNormF32(&g, x, null, null, 1e-6, .bfloat16), try kit.own(&g, try fluxLayerNormNA(x, 1e-6, s)));
        try ops.expectBitEqual(try layerNormF32(&g, x, null, null, 1e-6, dt), try kit.own(&g, try mageLayerNormLast(x, null, null, 1e-6, s)));
        try ops.expectBitEqual(try layerNormF32(&g, x, w, null, 1e-5, dt), try kit.own(&g, try mageLayerNormLast(x, w, null, 1e-5, s)));
        try ops.expectBitEqual(try layerNormF32(&g, x, w, b, 1e-5, dt), try kit.own(&g, try mageLayerNormLast(x, w, b, 1e-5, s)));
    }
}
