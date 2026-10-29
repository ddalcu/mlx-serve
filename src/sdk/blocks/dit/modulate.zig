//! adaLN modulation x·(scale + 1) + shift, the 1 made in `one_dt`: f32 in FLUX.2, Krea, LTX and
//! mage_flow's VAE decoder (a bf16 activation promotes), the scale's dtype in mage_flow's DiT,
//! the modulation's in TRELLIS.2. Callers reshape scale/shift to broadcast first. MOSS-Image
//! folds the 1 into its modulation rows once per step: that is x·scale + shift, a plain FMA.

const std = @import("std");
const ops = @import("../../../mlx/ops.zig");

const Graph = ops.Graph;
const Array = ops.Array;
const Dtype = ops.Dtype;

pub fn modulate(parent: *Graph, x: Array, scale: Array, shift: Array, one_dt: Dtype) !Array {
    var g = parent.scope();
    defer g.deinit();
    return parent.keep(try g.add(try g.mul(x, try g.add(scale, try g.scalar(1.0, one_dt))), shift));
}

// ── tests ──

const testing = std.testing;
const mlx = @import("../../../mlx/ffi.zig");
const kit = @import("testkit.zig");
const scope_mod = @import("../../../mlx/scope.zig");
const S = kit.S;
const A = kit.A;
const mulA = kit.mulA;
const addA = kit.addA;
const reshape = kit.reshape;
const scalarF = kit.scalarF;
const scalarLike = kit.scalarLike;

/// flux.zig `modulate` (krea.zig's and ltx video.zig's are this sequence).
fn fluxModulate(ln: mlx.mlx_array, scale: mlx.mlx_array, shift: mlx.mlx_array, s: S) !mlx.mlx_array {
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var sp1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sp1);
    try mlx.check(mlx.mlx_add(&sp1, scale, one, s));
    const m = try mulA(ln, sp1, s);
    defer _ = mlx.mlx_array_free(m);
    return addA(m, shift, s);
}

/// mage_flow.zig `modulateSeqNoGate`.
fn mageModulateSeqNoGate(hidden: mlx.mlx_array, scale: mlx.mlx_array, shift: mlx.mlx_array, s: S) !mlx.mlx_array {
    const C = mlx.getShape(scale)[1];
    const B = mlx.getShape(scale)[0];
    const sc3 = try reshape(scale, &[_]c_int{ B, 1, C }, s);
    defer _ = mlx.mlx_array_free(sc3);
    const sh3 = try reshape(shift, &[_]c_int{ B, 1, C }, s);
    defer _ = mlx.mlx_array_free(sh3);
    const one = try scalarLike(1.0, scale, s);
    defer _ = mlx.mlx_array_free(one);
    const onep = try addA(sc3, one, s);
    defer _ = mlx.mlx_array_free(onep);
    const scaled = try mulA(hidden, onep, s);
    defer _ = mlx.mlx_array_free(scaled);
    return addA(scaled, sh3, s);
}

/// mage_flow.zig `modulate4d` (with its `reshape4`).
fn mageModulate4d(x: mlx.mlx_array, shift: mlx.mlx_array, scale: mlx.mlx_array, s: S) !mlx.mlx_array {
    const reshape4 = struct {
        fn f(v: mlx.mlx_array, st: S) !mlx.mlx_array {
            const c = mlx.getShape(v)[1];
            return reshape(v, &[_]c_int{ 1, 1, 1, c }, st);
        }
    }.f;
    const sh4 = try reshape4(shift, s);
    defer _ = mlx.mlx_array_free(sh4);
    const sc4 = try reshape4(scale, s);
    defer _ = mlx.mlx_array_free(sc4);
    const one = scalarF(1.0);
    defer _ = mlx.mlx_array_free(one);
    const one_p = try addA(sc4, one, s);
    defer _ = mlx.mlx_array_free(one_p);
    const xs = try mulA(x, one_p, s);
    defer _ = mlx.mlx_array_free(xs);
    return addA(xs, sh4, s);
}

/// trellis2.zig `modulate` (its `one` is `scalarLike(m, 1)` of the modulation).
fn trellisModulate(sc: *scope_mod.Scope, x: A, shift: A, scale: A, one: A) !A {
    return sc.add(try sc.mul(x, try sc.add(one, scale)), shift);
}

test "media blocks: modulate is bit-equal to the FLUX.2/Krea/LTX, mage_flow and TRELLIS.2 copies" {
    const s = kit.cpu();
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    for ([_]ops.Dtype{ .bfloat16, .float32 }, 0..) |dt, i| {
        const x = try kit.randn(&g, &.{ 2, 7, 32 }, 100 + i, dt, 2);
        const sc2 = try kit.randn(&g, &.{ 2, 32 }, 110 + i, dt, 0.5);
        const sh2 = try kit.randn(&g, &.{ 2, 32 }, 120 + i, dt, 0.5);
        const sc3 = try g.reshape(sc2, &.{ 2, 1, 32 });
        const sh3 = try g.reshape(sh2, &.{ 2, 1, 32 });
        try ops.expectBitEqual(try modulate(&g, x, sc3, sh3, .float32), try kit.own(&g, try fluxModulate(x, sc3, sh3, s)));
        try ops.expectBitEqual(try modulate(&g, x, sc3, sh3, dt), try kit.own(&g, try mageModulateSeqNoGate(x, sc2, sh2, s)));
        {
            var sc = scope_mod.Scope.init(testing.allocator, s);
            defer sc.deinit();
            const one = try sc.scalarLike(sc3, 1);
            try ops.expectBitEqual(try modulate(&g, x, sc3, sh3, dt), try trellisModulate(&sc, x, sh3, sc3, one));
        }
        const img = try kit.randn(&g, &.{ 1, 4, 5, 32 }, 130 + i, dt, 2);
        const sc1 = try g.sliceAxis(sc2, 0, 0, 1);
        const sh1 = try g.sliceAxis(sh2, 0, 0, 1);
        try ops.expectBitEqual(
            try modulate(&g, img, try g.reshape(sc1, &.{ 1, 1, 1, 32 }), try g.reshape(sh1, &.{ 1, 1, 1, 32 }), .float32),
            try kit.own(&g, try mageModulate4d(img, sh1, sc1, s)),
        );
    }
}
