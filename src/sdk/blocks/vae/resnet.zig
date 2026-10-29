//! diffusers' ResnetBlock2D and the VAE mid-block's single-head attention, channels-last,
//! over the norm a backend's VAE uses (`Norm`): FLUX.2 and Hunyuan3D paint (flat GroupNorm),
//! MOSS-Image (axes GroupNorm), Krea and Qwen-Image's 3D causal VAE per frame (RMS, eps 1e-12).
//! mage_flow's resnet is this without a shortcut. The mid attention comes with separate q/k/v/o
//! projections (`midAttention`: FLUX.2, Hunyuan3D paint, MOSS) or one 1x1 qkv conv and a 1x1
//! out conv (`midAttentionQkv`: Krea, Qwen-Image).

const std = @import("std");
const ops = @import("../../../mlx/ops.zig");
const gn = @import("../norms/group_norm.zig");
const conv = @import("conv.zig");

const Graph = ops.Graph;
const Array = ops.Array;

pub const Norm = union(enum) {
    flat: struct { groups: c_int = 32, eps: f32, out: ops.Dtype },
    axes: struct { groups: c_int = 32, eps: f32 },
    /// mlx's fused RMS norm over channels (no bias).
    rms: f32,

    pub fn apply(n: Norm, parent: *Graph, x: Array, w: Array, b: ?Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        return parent.keep(switch (n) {
            .flat => |f| try gn.groupNormFlat(&g, x, w, b.?, f.groups, f.eps, f.out),
            .axes => |a| try gn.groupNormAxes(&g, x, w, b.?, a.groups, a.eps),
            .rms => |eps| try g.rmsNorm(x, w, eps),
        });
    }
};

/// Conv weights [out, kh, kw, in]; the 1x1 shortcut only where channels change.
pub const Resnet = struct {
    n1w: Array,
    n1b: ?Array = null,
    c1w: Array,
    c1b: Array,
    n2w: Array,
    n2b: ?Array = null,
    c2w: Array,
    c2b: Array,
    sw: ?Array = null,
    sb: ?Array = null,

    pub fn forward(r: *const Resnet, parent: *Graph, norm: Norm, x: Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        const h1 = try conv.conv2d(&g, try g.silu(try norm.apply(&g, x, r.n1w, r.n1b)), r.c1w, r.c1b, 1, 1, 1);
        const h2 = try conv.conv2d(&g, try g.silu(try norm.apply(&g, h1, r.n2w, r.n2b)), r.c2w, r.c2b, 1, 1, 1);
        const skip = if (r.sw) |sw| try conv.conv2d(&g, x, sw, r.sb.?, 1, 0, 1) else x;
        return parent.keep(try g.add(h2, skip));
    }
};

/// x + o(attend(q(n), k(n), v(n))) over all H·W positions as one head, n = norm(x).
/// `proj` holds q, k, v, o with `forward(*Graph, Array)`, applied to [1, H, W, C].
pub fn midAttention(parent: *Graph, norm: Norm, x: Array, nw: Array, nb: ?Array, proj: anytype) !Array {
    var g = parent.scope();
    defer g.deinit();
    const sh = ops.shape(x);
    const H, const W, const C = .{ sh[1], sh[2], sh[3] };
    const n = try norm.apply(&g, x, nw, nb);
    const flat = [_]c_int{ 1, 1, H * W, C };
    const q = try g.reshape(try proj.q.forward(&g, n), &flat);
    const k = try g.reshape(try proj.k.forward(&g, n), &flat);
    const v = try g.reshape(try proj.v.forward(&g, n), &flat);
    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(C)));
    const at = try g.reshape(try g.sdpa(q, k, v, scale, .none), &.{ 1, H, W, C });
    return parent.keep(try g.add(x, try proj.o.forward(&g, at)));
}

/// x + proj(attend(split3(qkv(n)))) over all H·W positions per batch row, n = norm(x); the
/// projections are 1x1 convs (weights [out, 1, 1, in]).
pub fn midAttentionQkv(parent: *Graph, norm: Norm, x: Array, nw: Array, nb: ?Array, qkv_w: Array, qkv_b: Array, proj_w: Array, proj_b: Array) !Array {
    var g = parent.scope();
    defer g.deinit();
    const sh = ops.shape(x);
    const C = sh[3];
    const qkv = try conv.conv2d(&g, try norm.apply(&g, x, nw, nb), qkv_w, qkv_b, 1, 0, 1);
    const flat = try g.reshape(qkv, &.{ sh[0], 1, sh[1] * sh[2], 3 * C });
    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(C)));
    const o = try g.sdpa(try g.sliceAxis(flat, 3, 0, C), try g.sliceAxis(flat, 3, C, 2 * C), try g.sliceAxis(flat, 3, 2 * C, 3 * C), scale, .none);
    const grid = try g.reshape(o, &.{ sh[0], sh[1], sh[2], C });
    return parent.keep(try g.add(x, try conv.conv2d(&g, grid, proj_w, proj_b, 1, 0, 1)));
}

// ── tests ──

const testing = std.testing;
const mlx = @import("../../../mlx/ffi.zig");
const kit = @import("../dit/testkit.zig");
const Linear = @import("../dit/linear.zig").Linear;
const scope_mod = @import("../../../mlx/scope.zig");
const S = kit.S;
const A = kit.A;
const reshape = kit.reshape;
const addA = kit.addA;
const silu = kit.silu;

// Oracles: each backend's block as written; their GroupNorm and conv leaves are the sdk
// blocks those leaves were proven bit-equal to (group_norm.zig, conv.zig).

/// flux.zig `Resnet.forward` (hunyuan3d paint.zig's is this with its eps and f16).
fn fluxResnet(r: *const Resnet, n: Norm, x: mlx.mlx_array, g: *Graph) !mlx.mlx_array {
    const s = g.s;
    const h0 = try n.apply(g, x, r.n1w, r.n1b);
    const a0 = try silu(h0, s);
    defer _ = mlx.mlx_array_free(a0);
    const c1 = try conv.conv2d(g, a0, r.c1w, r.c1b, 1, 1, 1);
    const h1 = try n.apply(g, c1, r.n2w, r.n2b);
    const a1 = try silu(h1, s);
    defer _ = mlx.mlx_array_free(a1);
    const c2 = try conv.conv2d(g, a1, r.c2w, r.c2b, 1, 1, 1);
    if (r.sw) |sw| {
        const sc = try conv.conv2d(g, x, sw, r.sb.?, 1, 0, 1);
        return addA(c2, sc, s);
    }
    return addA(c2, x, s);
}

/// moss_image.zig `Vae.resForward` (Scope; groupNorm/convB as their proven sdk twins).
fn mossResForward(sc: *scope_mod.Scope, g: *Graph, r: *const Resnet, x: A) !A {
    const n: Norm = .{ .axes = .{ .eps = 1e-6 } };
    const h = try conv.conv2d(g, try sc.silu(try n.apply(g, x, r.n1w, r.n1b)), r.c1w, r.c1b, 1, 1, 1);
    const h2 = try conv.conv2d(g, try sc.silu(try n.apply(g, h, r.n2w, r.n2b)), r.c2w, r.c2b, 1, 1, 1);
    const skip = if (r.sw) |sw| try conv.conv2d(g, x, sw, r.sb.?, 1, 0, 1) else x;
    return sc.add(h2, skip);
}

const TestProj = struct { q: Linear, k: Linear, v: Linear, o: Linear };

/// flux.zig `VaeAttn.forward` (paint.zig's is this sequence).
fn fluxVaeAttn(p: *const TestProj, n: Norm, nw: A, nb: A, x: mlx.mlx_array, g: *Graph) !mlx.mlx_array {
    const s = g.s;
    const sh = mlx.getShape(x);
    const H = sh[1];
    const Wd = sh[2];
    const C = sh[3];
    const normed = try n.apply(g, x, nw, nb);
    const q = try p.q.forward(g, normed);
    const k = try p.k.forward(g, normed);
    const v = try p.v.forward(g, normed);
    const qr = try reshape(q, &[_]c_int{ 1, 1, H * Wd, C }, s);
    defer _ = mlx.mlx_array_free(qr);
    const kr = try reshape(k, &[_]c_int{ 1, 1, H * Wd, C }, s);
    defer _ = mlx.mlx_array_free(kr);
    const vr = try reshape(v, &[_]c_int{ 1, 1, H * Wd, C }, s);
    defer _ = mlx.mlx_array_free(vr);
    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(C)));
    var attn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn);
    const null_a = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qr, kr, vr, scale, "", null_a, null_a, false, s));
    const ar = try reshape(attn, &[_]c_int{ 1, H, Wd, C }, s);
    defer _ = mlx.mlx_array_free(ar);
    const ao = try p.o.forward(g, ar);
    return addA(x, ao, s);
}

/// moss_image.zig `Vae.decode`'s mid attention (projections on the flattened [1,1,HW,C]).
fn mossMidAttn(sc: *scope_mod.Scope, g: *Graph, p: *const TestProj, nw: A, nb: A, m0: A) !A {
    const hs = mlx.getShape(m0);
    const c = hs[3];
    const n = try (Norm{ .axes = .{ .eps = 1e-6 } }).apply(g, m0, nw, nb);
    const flat = try sc.reshape(n, &.{ 1, 1, hs[1] * hs[2], c });
    const q = try p.q.forward(g, flat);
    const k = try p.k.forward(g, flat);
    const v = try p.v.forward(g, flat);
    const at = try sc.sdpa(q, k, v, 1.0 / @sqrt(@as(f32, @floatFromInt(c))));
    return sc.add(m0, try sc.reshape(try p.o.forward(g, at), hs));
}

/// krea.zig `ResBlock3D.forward` (its `rmsChannels` is `g.rmsNorm(.., 1e-12)`, its conv2d the
/// sdk conv2d proven equal in conv.zig; Qwen-Image's `Res.forward` unbanded is this sequence).
fn kreaResBlock(r: *const Resnet, x: mlx.mlx_array, g: *Graph) !mlx.mlx_array {
    const s = g.s;
    const h0 = try g.rmsNorm(x, r.n1w, 1e-12);
    const a0 = try silu(h0, s);
    defer _ = mlx.mlx_array_free(a0);
    const c1 = try conv.conv2d(g, a0, r.c1w, r.c1b, 1, 1, 1);
    const h1 = try g.rmsNorm(c1, r.n2w, 1e-12);
    const a1 = try silu(h1, s);
    defer _ = mlx.mlx_array_free(a1);
    const c2 = try conv.conv2d(g, a1, r.c2w, r.c2b, 1, 1, 1);
    if (r.sw) |skw| {
        const sc = try conv.conv2d(g, x, skw, r.sb.?, 1, 0, 1);
        return addA(c2, sc, s);
    }
    return addA(c2, x, s);
}

/// krea.zig `AttnBlock3D.forward` (B = 1).
fn kreaAttnBlock(norm: A, qkv_w: A, qkv_b: A, proj_w: A, proj_b: A, x: mlx.mlx_array, g: *Graph) !mlx.mlx_array {
    const s = g.s;
    const sh = mlx.getShape(x); // [1,H,W,C]
    const H = sh[1];
    const Wd = sh[2];
    const C = sh[3];
    const normed = try g.rmsNorm(x, norm, 1e-12);
    const qkv = try conv.conv2d(g, normed, qkv_w, qkv_b, 1, 0, 1); // (1,H,W,3C)
    const flat = try reshape(qkv, &[_]c_int{ 1, 1, H * Wd, 3 * C }, s);
    defer _ = mlx.mlx_array_free(flat);
    const q = try kit.sliceAxis(flat, 3, 0, C, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try kit.sliceAxis(flat, 3, C, 2 * C, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try kit.sliceAxis(flat, 3, 2 * C, 3 * C, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(C)));
    var attn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn);
    const null_a = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, q, k, v, scale, "", null_a, null_a, false, s)); // (1,1,HW,C)
    const ar = try reshape(attn, &[_]c_int{ 1, H, Wd, C }, s);
    defer _ = mlx.mlx_array_free(ar);
    const pj = try conv.conv2d(g, ar, proj_w, proj_b, 1, 0, 1);
    return addA(pj, x, s);
}

/// qwen_image.zig `Mid.attn` (any batch; its slices are made contiguous).
fn qwenMidAttn(norm: A, qkv_w: A, qkv_b: A, proj_w: A, proj_b: A, x: A, g: *Graph) !A {
    const s = g.s;
    const sh = mlx.getShape(x); // [B,H,W,C]
    const C = sh[3];
    const n = try g.rmsNorm(x, norm, 1e-12);
    const qkv = try conv.conv2d(g, n, qkv_w, qkv_b, 1, 0, 1);
    const flat = try reshape(qkv, &[_]c_int{ sh[0], 1, sh[1] * sh[2], 3 * C }, s);
    defer _ = mlx.mlx_array_free(flat);
    var parts: [3]A = undefined;
    for (&parts, 0..) |*p, i| {
        const raw = try kit.sliceAxis(flat, 3, @as(c_int, @intCast(i)) * C, @as(c_int, @intCast(i + 1)) * C, s);
        defer _ = mlx.mlx_array_free(raw);
        p.* = try kit.contig(raw, s);
    }
    defer for (parts) |p| {
        _ = mlx.mlx_array_free(p);
    };
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(C)));
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    const none = A{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, parts[0], parts[1], parts[2], scale, "", none, none, false, s));
    const grid = try reshape(o, &[_]c_int{ sh[0], sh[1], sh[2], C }, s);
    defer _ = mlx.mlx_array_free(grid);
    const proj = try conv.conv2d(g, grid, proj_w, proj_b, 1, 0, 1);
    return addA(proj, x, s);
}

fn randResnet(g: *Graph, cin: c_int, cout: c_int, dt: ops.Dtype, seed: u64) !Resnet {
    var r: Resnet = .{
        .n1w = try kit.randn(g, &.{cin}, seed, dt, 1),
        .n1b = try kit.randn(g, &.{cin}, seed + 1, dt, 1),
        .c1w = try kit.randn(g, &.{ cout, 3, 3, cin }, seed + 2, dt, 0.1),
        .c1b = try kit.randn(g, &.{cout}, seed + 3, dt, 0.1),
        .n2w = try kit.randn(g, &.{cout}, seed + 4, dt, 1),
        .n2b = try kit.randn(g, &.{cout}, seed + 5, dt, 1),
        .c2w = try kit.randn(g, &.{ cout, 3, 3, cout }, seed + 6, dt, 0.1),
        .c2b = try kit.randn(g, &.{cout}, seed + 7, dt, 0.1),
    };
    if (cin != cout) {
        r.sw = try kit.randn(g, &.{ cout, 1, 1, cin }, seed + 8, dt, 0.1);
        r.sb = try kit.randn(g, &.{cout}, seed + 9, dt, 0.1);
    }
    return r;
}

test "media blocks: VAE Resnet and mid attention are bit-equal to FLUX.2, MOSS-Image, Krea and Qwen-Image" {
    const s = kit.cpu();
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    const flat: Norm = .{ .flat = .{ .eps = 1e-6, .out = .bfloat16 } };
    const x = try kit.randn(&g, &.{ 1, 5, 6, 32 }, 400, .bfloat16, 1);
    for ([_]c_int{ 32, 64 }, 0..) |cout, i| {
        const r = try randResnet(&g, 32, cout, .bfloat16, 410 + 10 * i);
        try ops.expectBitEqual(try r.forward(&g, flat, x), try kit.own(&g, try fluxResnet(&r, flat, x, &g)));
    }
    const xf = try kit.randn(&g, &.{ 1, 5, 6, 64 }, 450, .float32, 1);
    for ([_]c_int{ 64, 32 }, 0..) |cout, i| {
        const r = try randResnet(&g, 64, cout, .float32, 460 + 10 * i);
        var sc = scope_mod.Scope.init(testing.allocator, s);
        defer sc.deinit();
        try ops.expectBitEqual(try r.forward(&g, .{ .axes = .{ .eps = 1e-6 } }, xf), try mossResForward(&sc, &g, &r, xf));
    }

    // RMS-normed (Krea / Qwen-Image 3D causal VAE, per frame).
    const xr = try kit.randn(&g, &.{ 1, 5, 6, 32 }, 560, .bfloat16, 1);
    for ([_]c_int{ 32, 64 }, 0..) |cout, i| {
        var r = try randResnet(&g, 32, cout, .bfloat16, 570 + 10 * i);
        r.n1b = null;
        r.n2b = null;
        try ops.expectBitEqual(try r.forward(&g, .{ .rms = 1e-12 }, xr), try kit.own(&g, try kreaResBlock(&r, xr, &g)));
    }
    const an = try kit.randn(&g, &.{32}, 590, .bfloat16, 1);
    const aqw = try kit.randn(&g, &.{ 96, 1, 1, 32 }, 591, .bfloat16, 0.1);
    const aqb = try kit.randn(&g, &.{96}, 592, .bfloat16, 0.1);
    const apw = try kit.randn(&g, &.{ 32, 1, 1, 32 }, 593, .bfloat16, 0.1);
    const apb = try kit.randn(&g, &.{32}, 594, .bfloat16, 0.1);
    try ops.expectBitEqual(try midAttentionQkv(&g, .{ .rms = 1e-12 }, xr, an, null, aqw, aqb, apw, apb), try kit.own(&g, try kreaAttnBlock(an, aqw, aqb, apw, apb, xr, &g)));
    const xr2 = try kit.randn(&g, &.{ 2, 4, 3, 32 }, 595, .bfloat16, 1);
    try ops.expectBitEqual(try midAttentionQkv(&g, .{ .rms = 1e-12 }, xr2, an, null, aqw, aqb, apw, apb), try kit.own(&g, try qwenMidAttn(an, aqw, aqb, apw, apb, xr2, &g)));

    var p: TestProj = undefined;
    inline for (.{ "q", "k", "v", "o" }, 0..) |f, j| @field(p, f) = .{
        .w = try kit.randn(&g, &.{ 32, 32 }, 500 + j, .bfloat16, 0.1),
        .bias = try kit.randn(&g, &.{32}, 510 + j, .bfloat16, 0.1),
    };
    const nw = try kit.randn(&g, &.{32}, 520, .bfloat16, 1);
    const nb = try kit.randn(&g, &.{32}, 521, .bfloat16, 1);
    try ops.expectBitEqual(try midAttention(&g, flat, x, nw, nb, &p), try kit.own(&g, try fluxVaeAttn(&p, flat, nw, nb, x, &g)));
    // MOSS projects the flattened rows: the same bits on the f32 VAE.
    var pf: TestProj = undefined;
    inline for (.{ "q", "k", "v", "o" }, 0..) |f, j| @field(pf, f) = .{
        .w = try kit.randn(&g, &.{ 64, 64 }, 530 + j, .float32, 0.1),
        .bias = try kit.randn(&g, &.{64}, 540 + j, .float32, 0.1),
    };
    const nwf = try kit.randn(&g, &.{64}, 550, .float32, 1);
    const nbf = try kit.randn(&g, &.{64}, 551, .float32, 1);
    var sc = scope_mod.Scope.init(testing.allocator, s);
    defer sc.deinit();
    try ops.expectBitEqual(try midAttention(&g, .{ .axes = .{ .eps = 1e-6 } }, xf, nwf, nbf, &pf), try mossMidAttn(&sc, &g, &pf, nwf, nbf, xf));
}
