//! Snake activations of the audio VAEs/vocoders, channels last (alpha/beta broadcast on C).
//! - `snakeBetaLog`: x + sin(x·eᵅ)² / (eᵝ + 1e-9), parameters stored as logs, an f32 epsilon
//!   (YuE2's Oobleck decoder; ACE-Step's is this on its f32 cast of x, cast back).
//! - `snake`: x + sin(α·x)² / (α + 1e-9), the epsilon made in `eps_dt` (minimax_h3 audio: f32;
//!   music3: α's dtype). Kokoro (no epsilon, `square`) and LTX (`square`, 1/(eᵝ+ε) multiplied)
//!   are other op sequences and stay in their backends.

const std = @import("std");
const ops = @import("../../../mlx/ops.zig");

const Graph = ops.Graph;
const Array = ops.Array;
const Dtype = ops.Dtype;

pub fn snakeBetaLog(parent: *Graph, x: Array, log_alpha: Array, log_beta: Array) !Array {
    var g = parent.scope();
    defer g.deinit();
    const sn = try g.sin(try g.mul(x, try g.exp(log_alpha)));
    return parent.keep(try g.add(x, try g.div(try g.mul(sn, sn), try g.add(try g.exp(log_beta), try g.scalar(1e-9, .float32)))));
}

pub fn snake(parent: *Graph, x: Array, alpha: Array, eps_dt: Dtype) !Array {
    var g = parent.scope();
    defer g.deinit();
    const sn = try g.sin(try g.mul(x, alpha));
    return parent.keep(try g.add(x, try g.div(try g.mul(sn, sn), try g.add(alpha, try g.scalar(1e-9, eps_dt)))));
}

// ── tests ──

const testing = std.testing;
const mlx = @import("../../../mlx/ffi.zig");
const kit = @import("../dit/testkit.zig");
const S = kit.S;
const A = kit.A;
const astype = kit.astype;
const mulA = kit.mulA;
const addA = kit.addA;
const divA = kit.divA;
const reshape = kit.reshape;

/// yue2.zig `Engine.snake` with its weights passed in.
fn yueSnake(parent: *Graph, x: A, a: A, b: A) !A {
    var g = parent.scope();
    defer g.deinit();
    const alpha = try g.exp(a);
    const beta = try g.exp(b);
    const sn = try g.sin(try g.mul(x, alpha));
    return parent.keep(try g.add(x, try g.div(try g.mul(sn, sn), try g.add(beta, try g.scalar(1e-9, .float32)))));
}

/// acestep.zig `snake` with its weights passed in.
fn acestepSnake(x: mlx.mlx_array, alpha: A, beta: A, s: S) !mlx.mlx_array {
    const in_dtype = mlx.mlx_array_dtype(x);
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    var ea = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ea);
    try mlx.check(mlx.mlx_exp(&ea, alpha, s));
    var eb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eb);
    try mlx.check(mlx.mlx_exp(&eb, beta, s));
    const ax = try mulA(xf, ea, s);
    defer _ = mlx.mlx_array_free(ax);
    var sn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sn);
    try mlx.check(mlx.mlx_sin(&sn, ax, s));
    const sq = try mulA(sn, sn, s);
    defer _ = mlx.mlx_array_free(sq);
    const eps_c = mlx.mlx_array_new_float(1e-9);
    defer _ = mlx.mlx_array_free(eps_c);
    const beps = try addA(eb, eps_c, s);
    defer _ = mlx.mlx_array_free(beps);
    var recip = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(recip);
    try mlx.check(mlx.mlx_divide(&recip, sq, beps, s));
    const out_f = try addA(xf, recip, s);
    defer _ = mlx.mlx_array_free(out_f);
    return astype(out_f, in_dtype, s);
}

/// minimax_h3 audio.zig `snake1d` with alpha [C] passed in.
fn h3Snake1d(alpha_raw: A, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const ch: c_int = @intCast(mlx.mlx_array_size(alpha_raw));
    const alpha = try reshape(alpha_raw, &[_]c_int{ 1, 1, ch }, s);
    defer _ = mlx.mlx_array_free(alpha);
    const ax = try mulA(x, alpha, s);
    defer _ = mlx.mlx_array_free(ax);
    var sn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sn);
    try mlx.check(mlx.mlx_sin(&sn, ax, s));
    const sq = try mulA(sn, sn, s);
    defer _ = mlx.mlx_array_free(sq);
    const eps = mlx.mlx_array_new_float(1e-9);
    defer _ = mlx.mlx_array_free(eps);
    const den = try addA(alpha, eps, s);
    defer _ = mlx.mlx_array_free(den);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_divide(&scaled, sq, den, s));
    return addA(x, scaled, s);
}

/// music3.zig `snakeA` with alpha passed in (`scalarLike(alpha, ..)` = alpha's dtype).
fn music3SnakeA(x: mlx.mlx_array, alpha: A, s: S) !mlx.mlx_array {
    const ax = try mulA(x, alpha, s);
    defer _ = mlx.mlx_array_free(ax);
    var sn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sn);
    try mlx.check(mlx.mlx_sin(&sn, ax, s));
    const sq = try mulA(sn, sn, s);
    defer _ = mlx.mlx_array_free(sq);
    const eps_c = try kit.scalarLike(1e-9, alpha, s);
    defer _ = mlx.mlx_array_free(eps_c);
    const aeps = try addA(alpha, eps_c, s);
    defer _ = mlx.mlx_array_free(aeps);
    const frac = try divA(sq, aeps, s);
    defer _ = mlx.mlx_array_free(frac);
    return addA(x, frac, s);
}

test "media blocks: snake / snakeBetaLog are bit-equal to YuE2, ACE-Step, minimax_h3 and music3" {
    const s = kit.cpu();
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    const C = 24;
    for ([_]Dtype{ .float32, .bfloat16 }, 0..) |dt, i| {
        const x = try kit.randn(&g, &.{ 1, 50, C }, 600 + i, dt, 2);
        const la = try kit.randn(&g, &.{C}, 610 + i, dt, 0.3);
        const lb = try kit.randn(&g, &.{C}, 620 + i, dt, 0.3);
        try ops.expectBitEqual(try snakeBetaLog(&g, x, la, lb), try yueSnake(&g, x, la, lb));
        const xf = try g.astype(x, .float32);
        try ops.expectBitEqual(try g.astype(try snakeBetaLog(&g, xf, la, lb), dt), try kit.own(&g, try acestepSnake(x, la, lb, s)));
        const alpha = try g.abs(try kit.randn(&g, &.{C}, 630 + i, dt, 1));
        try ops.expectBitEqual(try snake(&g, x, try g.reshape(alpha, &.{ 1, 1, C }), .float32), try kit.own(&g, try h3Snake1d(alpha, x, s)));
        try ops.expectBitEqual(try snake(&g, x, alpha, dt), try kit.own(&g, try music3SnakeA(x, alpha, s)));
    }
}
