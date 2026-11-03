//! Channels-last 2-D VAE convolutions: the input made contiguous first (mlx's conv2d has
//! miscomputed on lazy strided views), weight [out, kh, kw, in], optional bias added after.
//! FLUX.2, Krea, mage_flow and Hunyuan3D paint (f16 inputs: hunyuan's UNet casts first);
//! MOSS-Image's `convB` skips the copy, the same bits on a contiguous input.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");

const Graph = ops.Graph;
const Array = ops.Array;

pub fn conv2d(parent: *Graph, x: Array, w: Array, bias: ?Array, stride: c_int, pad: c_int, groups: c_int) !Array {
    var g = parent.scope();
    defer g.deinit();
    const y = try g.conv2d(try g.contiguous(x), w, stride, pad, 1, groups);
    return parent.keep(if (bias) |b| try g.add(y, b) else y);
}

/// Nearest 2x on H and W (repeat), then a 3x3 conv.
pub fn upsample2x(parent: *Graph, x: Array, w: Array, b: Array) !Array {
    var g = parent.scope();
    defer g.deinit();
    return parent.keep(try conv2d(&g, try g.repeat(try g.repeat(x, 2, 1), 2, 2), w, b, 1, 1, 1));
}

/// diffusers' Downsample2D: zero pad one row/column at the end, then a stride-2 conv.
pub fn downsample2x(parent: *Graph, x: Array, w: Array, b: Array) !Array {
    var g = parent.scope();
    defer g.deinit();
    const zero = try g.scalar(0, .float32);
    const p = try g.call(mlx.mlx_pad, .{ x, &[_]c_int{ 1, 2 }, @as(usize, 2), &[_]c_int{ 0, 0 }, @as(usize, 2), &[_]c_int{ 1, 1 }, @as(usize, 2), zero, "constant" });
    return parent.keep(try conv2d(&g, p, w, b, 2, 0, 1));
}

// ── tests ──

const testing = std.testing;
const kit = @import("../dit/testkit.zig");
const scope_mod = @import("../../../mlx/scope.zig");
const S = kit.S;
const A = kit.A;
const addA = kit.addA;
const contig = kit.contig;

/// flux.zig `conv2d` (krea.zig's and hunyuan3d paint.zig's are this sequence).
fn fluxConv2d(x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array, pad: c_int, s: S) !mlx.mlx_array {
    var xc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xc);
    try mlx.check(mlx.mlx_contiguous(&xc, x, false, s));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, xc, w, 1, 1, pad, pad, 1, 1, 1, s));
    const r = try addA(o, bias, s);
    _ = mlx.mlx_array_free(o);
    return r;
}

/// mage_flow.zig `conv2d`.
fn mageConv2d(x: mlx.mlx_array, w: mlx.mlx_array, bias: ?mlx.mlx_array, stride: c_int, pad: c_int, groups: c_int, s: S) !mlx.mlx_array {
    const xc = try contig(x, s);
    defer _ = mlx.mlx_array_free(xc);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, xc, w, stride, stride, pad, pad, 1, 1, groups, s));
    if (bias) |b| {
        defer _ = mlx.mlx_array_free(o);
        return addA(o, b, s);
    }
    return o;
}

/// flux.zig `Vae.upsample` (krea.zig `upsample`, paint.zig `upsampleConv`).
fn fluxUpsample(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var r1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r1);
    try mlx.check(mlx.mlx_repeat_axis(&r1, x, 2, 1, s));
    var r2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r2);
    try mlx.check(mlx.mlx_repeat_axis(&r2, r1, 2, 2, s));
    return fluxConv2d(r2, w, b, 1, s);
}

/// flux.zig `conv2dDown` (krea.zig `downsampleSpatial`, paint.zig `conv2dDown`).
fn fluxConv2dDown(x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array, s: S) !mlx.mlx_array {
    const axes = [_]c_int{ 1, 2 };
    const low = [_]c_int{ 0, 0 };
    const high = [_]c_int{ 1, 1 };
    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p);
    try mlx.check(mlx.mlx_pad(&p, x, &axes, 2, &low, 2, &high, 2, zero, "constant", s));
    var pc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pc);
    try mlx.check(mlx.mlx_contiguous(&pc, p, false, s));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, pc, w, 2, 2, 0, 0, 1, 1, 1, s));
    const r = try addA(o, bias, s);
    _ = mlx.mlx_array_free(o);
    return r;
}

/// moss_image.zig `Vae.convB` (Scope).
fn mossConvB(sc: *scope_mod.Scope, x: A, w: A, b: A) !A {
    const pad: c_int = @divExact(mlx.getShape(w)[1] - 1, 2);
    return sc.add(try sc.conv2d(x, w, 1, pad), b);
}

test "media blocks: VAE conv2d / upsample2x / downsample2x are bit-equal to the backends' copies" {
    const s = kit.cpu();
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    for ([_]ops.Dtype{ .bfloat16, .float16, .float32 }, 0..) |dt, i| {
        const x = try kit.randn(&g, &.{ 1, 6, 7, 8 }, 300 + i, dt, 1);
        // A strided view: the contiguous copy is what keeps it right.
        const xt = try g.transpose(try kit.randn(&g, &.{ 1, 7, 6, 8 }, 310 + i, dt, 1), &.{ 0, 2, 1, 3 });
        const w3 = try kit.randn(&g, &.{ 12, 3, 3, 8 }, 320 + i, dt, 0.2);
        const w1 = try kit.randn(&g, &.{ 12, 1, 1, 8 }, 330 + i, dt, 0.2);
        const b = try kit.randn(&g, &.{12}, 340 + i, dt, 0.2);
        for ([_]A{ x, xt }) |in| {
            try ops.expectBitEqual(try conv2d(&g, in, w3, b, 1, 1, 1), try kit.own(&g, try fluxConv2d(in, w3, b, 1, s)));
            try ops.expectBitEqual(try conv2d(&g, in, w1, b, 1, 0, 1), try kit.own(&g, try fluxConv2d(in, w1, b, 0, s)));
            try ops.expectBitEqual(try conv2d(&g, in, w3, b, 2, 1, 1), try kit.own(&g, try mageConv2d(in, w3, b, 2, 1, 1, s)));
            try ops.expectBitEqual(try conv2d(&g, in, w3, null, 1, 1, 1), try kit.own(&g, try mageConv2d(in, w3, null, 1, 1, 1, s)));
            try ops.expectBitEqual(try upsample2x(&g, in, w3, b), try kit.own(&g, try fluxUpsample(in, w3, b, s)));
            try ops.expectBitEqual(try downsample2x(&g, in, w3, b), try kit.own(&g, try fluxConv2dDown(in, w3, b, s)));
        }
        const wg = try kit.randn(&g, &.{ 8, 3, 3, 2 }, 350 + i, dt, 0.2);
        try ops.expectBitEqual(try conv2d(&g, x, wg, null, 1, 1, 4), try kit.own(&g, try mageConv2d(x, wg, null, 1, 1, 4, s)));
        var sc = scope_mod.Scope.init(testing.allocator, s);
        defer sc.deinit();
        try ops.expectBitEqual(try conv2d(&g, x, w3, b, 1, 1, 1), try mossConvB(&sc, x, w3, b));
    }
}
