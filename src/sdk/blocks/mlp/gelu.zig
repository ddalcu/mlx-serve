//! GELU for the media backends, on the op layer.
//! `geluTanh` takes the dtype its constants are made in: the operand's own keeps a bf16
//! activation bf16 (mage_flow, qwen_image, trellis2, stable_audio3, minimax_h3 audio), f32
//! promotes it to f32 (krea, ltx). `geluErf` always uses f32 constants (hunyuan3d shape +
//! paint UNet, qwen3_tts); mlx/scope.zig's `geluErf` is a different association and stays.
//! The √(2/π) constant is passed: stable_audio3 and trellis2 spell it `@sqrt(2.0 / std.math.pi)`
//! (`SQRT_2_PI_LOW`), one f32 ulp below the literal the others use (`SQRT_2_PI`); the two agree
//! once rounded to bf16/f16.

const std = @import("std");
const ops = @import("../../../mlx/ops.zig");

const Graph = ops.Graph;
const Array = ops.Array;
const Dtype = ops.Dtype;

pub const SQRT_2_PI: f32 = 0.7978845608028654;
pub const SQRT_2_PI_LOW: f32 = @sqrt(2.0 / std.math.pi);

/// 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))), the constants in `sdt`.
pub fn geluTanh(parent: *Graph, x: Array, sdt: Dtype, sqrt_2_pi: f32) !Array {
    var g = parent.scope();
    defer g.deinit();
    const x3 = try g.mul(try g.mul(x, x), x);
    const inner = try g.mul(try g.add(x, try g.mul(x3, try g.scalar(0.044715, sdt))), try g.scalar(sqrt_2_pi, sdt));
    const opt = try g.add(try g.tanh(inner), try g.scalar(1.0, sdt));
    return parent.keep(try g.mul(try g.mul(x, try g.scalar(0.5, sdt)), opt));
}

/// Exact GELU 0.5·x·(1 + erf(x/√2)) with f32 constants.
pub fn geluErf(parent: *Graph, x: Array) !Array {
    var g = parent.scope();
    defer g.deinit();
    const opt = try g.add(try g.erf(try g.mul(x, try g.scalar(0.7071067811865476, .float32))), try g.scalar(1.0, .float32));
    return parent.keep(try g.mul(try g.mul(x, try g.scalar(0.5, .float32)), opt));
}

// ── tests ──

const testing = std.testing;
const mlx = @import("../../../mlx/ffi.zig");
const kit = @import("../dit/testkit.zig");
const scope_mod = @import("../../../mlx/scope.zig");
const S = kit.S;
const A = kit.A;
const addA = kit.addA;
const mulA = kit.mulA;
const astype = kit.astype;
const scalarF = kit.scalarF;

// Oracles: each backend's private GELU as it stands (helper names mapped to the kit's).

/// krea.zig `geluTanh`: f32 scalars.
fn kreaGeluTanh(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const c: f32 = 0.7978845608028654; // sqrt(2/pi)
    const k = scalarF(0.044715);
    defer _ = mlx.mlx_array_free(k);
    const x2 = try mulA(x, x, s);
    defer _ = mlx.mlx_array_free(x2);
    const x3 = try mulA(x2, x, s);
    defer _ = mlx.mlx_array_free(x3);
    const kx3 = try mulA(x3, k, s);
    defer _ = mlx.mlx_array_free(kx3);
    const inner = try addA(x, kx3, s);
    defer _ = mlx.mlx_array_free(inner);
    const ca = scalarF(c);
    defer _ = mlx.mlx_array_free(ca);
    const cin = try mulA(inner, ca, s);
    defer _ = mlx.mlx_array_free(cin);
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_tanh(&t, cin, s));
    const one = scalarF(1.0);
    defer _ = mlx.mlx_array_free(one);
    const opt = try addA(t, one, s);
    defer _ = mlx.mlx_array_free(opt);
    const half = scalarF(0.5);
    defer _ = mlx.mlx_array_free(half);
    const hx = try mulA(x, half, s);
    defer _ = mlx.mlx_array_free(hx);
    return mulA(hx, opt, s);
}

/// mage_flow.zig `geluTanh`: scalars in x's dtype (qwen_image `geluApprox` and minimax_h3
/// audio `geluTanh` are this sequence with the same scalars).
fn mageGeluTanh(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const scalarLike = kit.scalarLike;
    const c: f32 = 0.7978845608028654;
    const k = try scalarLike(0.044715, x, s);
    defer _ = mlx.mlx_array_free(k);
    const x2 = try mulA(x, x, s);
    defer _ = mlx.mlx_array_free(x2);
    const x3 = try mulA(x2, x, s);
    defer _ = mlx.mlx_array_free(x3);
    const kx3 = try mulA(x3, k, s);
    defer _ = mlx.mlx_array_free(kx3);
    const inner = try addA(x, kx3, s);
    defer _ = mlx.mlx_array_free(inner);
    const ca = try scalarLike(c, x, s);
    defer _ = mlx.mlx_array_free(ca);
    const cin = try mulA(inner, ca, s);
    defer _ = mlx.mlx_array_free(cin);
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_tanh(&t, cin, s));
    const one = try scalarLike(1.0, x, s);
    defer _ = mlx.mlx_array_free(one);
    const opt = try addA(t, one, s);
    defer _ = mlx.mlx_array_free(opt);
    const half = try scalarLike(0.5, x, s);
    defer _ = mlx.mlx_array_free(half);
    const hx = try mulA(x, half, s);
    defer _ = mlx.mlx_array_free(hx);
    return mulA(hx, opt, s);
}

/// trellis2.zig `geluTanh` (Scope).
fn trellisGeluTanh(sc: *scope_mod.Scope, x: A) !A {
    const x3 = try sc.mul(try sc.mul(x, x), x);
    const inner = try sc.mulLike(try sc.add(x, try sc.mulLike(x3, 0.044715)), @sqrt(2.0 / std.math.pi));
    return sc.mul(try sc.mulLike(x, 0.5), try sc.add(try sc.tanh(inner), try sc.scalarLike(x, 1)));
}

/// stable_audio.zig `geluTanh` (Graph).
fn stableAudioGeluTanh(parent: *Graph, x: A) !A {
    var g = parent.scope();
    defer g.deinit();
    const x3 = try g.mul(try g.mul(x, x), x);
    const inner = try g.mulScalar(try g.add(x, try g.mulScalar(x3, 0.044715)), @sqrt(2.0 / std.math.pi));
    return parent.keep(try g.mul(try g.mulScalar(x, 0.5), try g.addScalar(try g.tanh(inner), 1)));
}

/// ltx video.zig `geluApprox`: f32 scalars, raw calls.
fn ltxGeluApprox(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const c0 = mlx.mlx_array_new_float(0.7978845608028654);
    defer _ = mlx.mlx_array_free(c0);
    const c1 = mlx.mlx_array_new_float(0.044715);
    defer _ = mlx.mlx_array_free(c1);
    const half = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(half);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_multiply(&x2, x, x, s));
    var x3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x3);
    try mlx.check(mlx.mlx_multiply(&x3, x2, x, s));
    var t1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t1);
    try mlx.check(mlx.mlx_multiply(&t1, x3, c1, s));
    var inner = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inner);
    try mlx.check(mlx.mlx_add(&inner, x, t1, s));
    var inner_s = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inner_s);
    try mlx.check(mlx.mlx_multiply(&inner_s, inner, c0, s));
    var th = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(th);
    try mlx.check(mlx.mlx_tanh(&th, inner_s, s));
    var thp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(thp);
    try mlx.check(mlx.mlx_add(&thp, th, one, s));
    var hx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hx);
    try mlx.check(mlx.mlx_multiply(&hx, x, half, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, hx, thp, s));
    return out;
}

/// hunyuan3d.zig `geluErf` (paint_unet.zig's and qwen3_tts `gelu` are this sequence).
fn hunyuanGeluErf(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const inv_sqrt2 = scalarF(0.7071067811865476);
    defer _ = mlx.mlx_array_free(inv_sqrt2);
    const xs = try mulA(x, inv_sqrt2, s);
    defer _ = mlx.mlx_array_free(xs);
    var e = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(e);
    try mlx.check(mlx.mlx_erf(&e, xs, s));
    const one = scalarF(1.0);
    defer _ = mlx.mlx_array_free(one);
    const opt = try addA(e, one, s);
    defer _ = mlx.mlx_array_free(opt);
    const half = scalarF(0.5);
    defer _ = mlx.mlx_array_free(half);
    const hx = try mulA(x, half, s);
    defer _ = mlx.mlx_array_free(hx);
    return mulA(hx, opt, s);
}

/// qwen3_tts tts.zig `gelu`.
fn ttsGelu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const inv_sqrt2 = mlx.mlx_array_new_float(0.7071067811865476);
    defer _ = mlx.mlx_array_free(inv_sqrt2);
    const half = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(half);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, x, inv_sqrt2, s));
    var erf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(erf);
    try mlx.check(mlx.mlx_erf(&erf, scaled, s));
    var one_plus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(one_plus);
    try mlx.check(mlx.mlx_add(&one_plus, erf, one, s));
    var hx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hx);
    try mlx.check(mlx.mlx_multiply(&hx, x, half, s));
    return mulA(hx, one_plus, s);
}

test "media blocks: gelu: geluTanh / geluErf are bit-equal to every backend's private copy" {
    const s = kit.cpu();
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    for ([_]Dtype{ .bfloat16, .float32, .float16 }, 0..) |dt, i| {
        const x = try kit.randn(&g, &.{ 3, 7, 33 }, 10 + i, dt, 3);
        // Operand-dtype constants.
        const like = try geluTanh(&g, x, dt, SQRT_2_PI);
        try ops.expectBitEqual(like, try kit.own(&g, try mageGeluTanh(x, s)));
        const low = try geluTanh(&g, x, dt, SQRT_2_PI_LOW);
        try ops.expectBitEqual(low, try stableAudioGeluTanh(&g, x));
        {
            var sc = scope_mod.Scope.init(testing.allocator, s);
            defer sc.deinit();
            try ops.expectBitEqual(low, try trellisGeluTanh(&sc, x));
        }
        if (dt != .float32) try ops.expectBitEqual(low, like);
        // f32 constants.
        const f32c = try geluTanh(&g, x, .float32, SQRT_2_PI);
        try ops.expectBitEqual(f32c, try kit.own(&g, try kreaGeluTanh(x, s)));
        try ops.expectBitEqual(f32c, try kit.own(&g, try ltxGeluApprox(x, s)));
        const e = try geluErf(&g, x);
        try ops.expectBitEqual(e, try kit.own(&g, try hunyuanGeluErf(x, s)));
        try ops.expectBitEqual(e, try kit.own(&g, try ttsGelu(x, s)));
    }
}
