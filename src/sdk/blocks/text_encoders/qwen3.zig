//! One Qwen3 decoder layer as the media pipelines run their prompt encoders: pre-norm GQA
//! attention with q/k RMS norm (optional additive mask), SwiGLU MLP, both residual. RoPE is
//! the variant point:
//! - `fast`: mlx's kernel; FLUX.2 klein runs the SDPA in f32 and casts back (`attn_f32`),
//!   ACE-Step's encoder does not;
//! - `table`: rotate-half over full-width tables (Krea: a bf16 -1; mage_flow and Qwen-Image's
//!   8B encoder: an f32 -1);
//! - `half`: half-width tables (minimax_h3's Qwen encoder).
//! `w` holds input_ln, q, k, v, o, q_norm, k_norm, post_ln, gate, up, down; each projection
//! has `forward(*Graph, Array)`.

const std = @import("std");
const ops = @import("../../../mlx/ops.zig");
const rope_mod = @import("../dit/rope.zig");

const Graph = ops.Graph;
const Array = ops.Array;

pub const Cfg = struct { heads: c_int, kv_heads: c_int, head_dim: c_int, eps: f32 };

pub const Rope = union(enum) {
    /// mlx_fast_rope over the whole head (base θ); with `attn_f32` the attention runs on f32
    /// casts of q/k/v and its output is cast to that dtype.
    fast: struct { theta: f32, attn_f32: ?ops.Dtype = null },
    /// cos/sin [1, 1, L, head_dim] (full width); the -1 of rotate_half in `neg`.
    table: struct { cos: Array, sin: Array, neg: ops.Dtype },
    /// cos/sin [.., L, head_dim/2] (half width).
    half: struct { cos: Array, sin: Array },
};

pub fn layerForward(parent: *Graph, w: anytype, cfg: Cfg, rope: Rope, x: Array, mask: ?Array) !Array {
    var g = parent.scope();
    defer g.deinit();
    const b = ops.shape(x)[0];
    const seq = ops.shape(x)[1];
    const xn = try g.rmsNorm(x, w.input_ln, cfg.eps);
    const perm = [_]c_int{ 0, 2, 1, 3 };
    const qt = try g.transpose(try g.rmsNorm(try g.reshape(try w.q.forward(&g, xn), &.{ b, seq, cfg.heads, cfg.head_dim }), w.q_norm, cfg.eps), &perm);
    const kt = try g.transpose(try g.rmsNorm(try g.reshape(try w.k.forward(&g, xn), &.{ b, seq, cfg.kv_heads, cfg.head_dim }), w.k_norm, cfg.eps), &perm);
    const vt = try g.transpose(try g.reshape(try w.v.forward(&g, xn), &.{ b, seq, cfg.kv_heads, cfg.head_dim }), &perm);
    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    const m: ops.Mask = if (mask) |mm| .{ .array = mm } else .none;
    const attn = switch (rope) {
        .fast => |f| blk: {
            const qr = try g.rope(qt, cfg.head_dim, f.theta, 0);
            const kr = try g.rope(kt, cfg.head_dim, f.theta, 0);
            const out = f.attn_f32 orelse break :blk try g.sdpa(qr, kr, vt, scale, m);
            break :blk try g.astype(try g.sdpa(try g.astype(qr, .float32), try g.astype(kr, .float32), try g.astype(vt, .float32), scale, m), out);
        },
        .table => |t| try g.sdpa(try rope_mod.ropeRotateHalf(&g, qt, t.cos, t.sin, t.neg), try rope_mod.ropeRotateHalf(&g, kt, t.cos, t.sin, t.neg), vt, scale, m),
        .half => |t| try g.sdpa(
            try rope_mod.ropeSplitHalf(&g, qt, t.cos, t.sin, @divExact(cfg.head_dim, 2)),
            try rope_mod.ropeSplitHalf(&g, kt, t.cos, t.sin, @divExact(cfg.head_dim, 2)),
            vt,
            scale,
            m,
        ),
    };
    const af = try g.reshape(try g.transpose(attn, &perm), &.{ b, seq, cfg.heads * cfg.head_dim });
    const h1 = try g.add(x, try w.o.forward(&g, af));
    const hn = try g.rmsNorm(h1, w.post_ln, cfg.eps);
    const gu = try g.mul(try g.silu(try w.gate.forward(&g, hn)), try w.up.forward(&g, hn));
    return parent.keep(try g.add(h1, try w.down.forward(&g, gu)));
}

// ── tests ──

const testing = std.testing;
const mlx = @import("../../../mlx/ffi.zig");
const kit = @import("../dit/testkit.zig");
const Linear = @import("../dit/linear.zig").Linear;
const S = kit.S;
const A = kit.A;
const rms = kit.rms;
const reshape = kit.reshape;
const transpose = kit.transpose;
const astype = kit.astype;
const addA = kit.addA;
const mulA = kit.mulA;
const silu = kit.silu;

/// The backends' projections as `forward(x, s)` over a dense weight, plus the Graph form.
const Proj = struct {
    raw: kit.RawLin,
    pub fn forward(self: *const Proj, g: *Graph, x: Array) !Array {
        return (Linear{ .w = self.raw.w }).forward(g, x);
    }
};

const TeLayer = struct { input_ln: A, post_ln: A, q_norm: A, k_norm: A, q: Proj, k: Proj, v: Proj, o: Proj, gate: Proj, up: Proj, down: Proj };

/// flux.zig `TextEncoder.layerForward` (projections through `.raw`).
fn fluxLayer(layer: *const TeLayer, c: Cfg, theta: f32, x: mlx.mlx_array, mask: mlx.mlx_array, seq: c_int, s: S) !mlx.mlx_array {
    const eps = c.eps;
    const heads = c.heads;
    const kv = c.kv_heads;
    const hd = c.head_dim;
    const xn = try rms(x, layer.input_ln, eps, s);
    defer _ = mlx.mlx_array_free(xn);
    const q = try layer.q.raw.forward(xn, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try layer.k.raw.forward(xn, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try layer.v.raw.forward(xn, s);
    defer _ = mlx.mlx_array_free(v);
    const q4 = try reshape(q, &[_]c_int{ 1, seq, heads, hd }, s);
    defer _ = mlx.mlx_array_free(q4);
    const qn = try rms(q4, layer.q_norm, eps, s);
    defer _ = mlx.mlx_array_free(qn);
    const qt = try transpose(qn, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(qt);
    const k4 = try reshape(k, &[_]c_int{ 1, seq, kv, hd }, s);
    defer _ = mlx.mlx_array_free(k4);
    const kn = try rms(k4, layer.k_norm, eps, s);
    defer _ = mlx.mlx_array_free(kn);
    const kt = try transpose(kn, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(kt);
    const v4 = try reshape(v, &[_]c_int{ 1, seq, kv, hd }, s);
    defer _ = mlx.mlx_array_free(v4);
    const vt = try transpose(v4, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(vt);
    const base = mlx.mlx_optional_float.some(theta);
    var qr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qr);
    try mlx.check(mlx.mlx_fast_rope(&qr, qt, hd, false, base, 1.0, 0, .{ .ctx = null }, s));
    var kr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kr);
    try mlx.check(mlx.mlx_fast_rope(&kr, kt, hd, false, base, 1.0, 0, .{ .ctx = null }, s));
    const qf = try astype(qr, .float32, s);
    defer _ = mlx.mlx_array_free(qf);
    const kf = try astype(kr, .float32, s);
    defer _ = mlx.mlx_array_free(kf);
    const vf = try astype(vt, .float32, s);
    defer _ = mlx.mlx_array_free(vf);
    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(c.head_dim)));
    var attn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn);
    const null_sink = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qf, kf, vf, scale, "array", mask, null_sink, false, s));
    const attn_bf = try astype(attn, .bfloat16, s);
    defer _ = mlx.mlx_array_free(attn_bf);
    const at = try transpose(attn_bf, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(at);
    const af = try reshape(at, &[_]c_int{ 1, seq, heads * hd }, s);
    defer _ = mlx.mlx_array_free(af);
    const o = try layer.o.raw.forward(af, s);
    defer _ = mlx.mlx_array_free(o);
    const h1 = try addA(x, o, s);
    defer _ = mlx.mlx_array_free(h1);
    const hn = try rms(h1, layer.post_ln, eps, s);
    defer _ = mlx.mlx_array_free(hn);
    const g = try layer.gate.raw.forward(hn, s);
    defer _ = mlx.mlx_array_free(g);
    const sg = try silu(g, s);
    defer _ = mlx.mlx_array_free(sg);
    const u = try layer.up.raw.forward(hn, s);
    defer _ = mlx.mlx_array_free(u);
    const gu = try mulA(sg, u, s);
    defer _ = mlx.mlx_array_free(gu);
    const dn = try layer.down.raw.forward(gu, s);
    defer _ = mlx.mlx_array_free(dn);
    return addA(h1, dn, s);
}

/// krea.zig `Conditioner.layerForward` (mage_flow `teLayerForward` is this with an f32 -1);
/// its `applyRopeEncoder` is the sdk `ropeRotateHalf` it was proven equal to (rope.zig).
fn kreaLayer(gr: *Graph, layer: *const TeLayer, c: Cfg, neg: ops.Dtype, x: mlx.mlx_array, mask: mlx.mlx_array, rope_cos: A, rope_sin: A, seq: c_int, s: S) !mlx.mlx_array {
    const eps = c.eps;
    const heads = c.heads;
    const kv = c.kv_heads;
    const hd = c.head_dim;
    const xn = try rms(x, layer.input_ln, eps, s);
    defer _ = mlx.mlx_array_free(xn);
    const q = try layer.q.raw.forward(xn, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try layer.k.raw.forward(xn, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try layer.v.raw.forward(xn, s);
    defer _ = mlx.mlx_array_free(v);
    const q4 = try reshape(q, &[_]c_int{ 1, seq, heads, hd }, s);
    defer _ = mlx.mlx_array_free(q4);
    const qn = try rms(q4, layer.q_norm, eps, s);
    defer _ = mlx.mlx_array_free(qn);
    const qt = try transpose(qn, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(qt);
    const k4 = try reshape(k, &[_]c_int{ 1, seq, kv, hd }, s);
    defer _ = mlx.mlx_array_free(k4);
    const kn = try rms(k4, layer.k_norm, eps, s);
    defer _ = mlx.mlx_array_free(kn);
    const kt = try transpose(kn, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(kt);
    const v4 = try reshape(v, &[_]c_int{ 1, seq, kv, hd }, s);
    defer _ = mlx.mlx_array_free(v4);
    const vt = try transpose(v4, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(vt);
    const qr = try rope_mod.ropeRotateHalf(gr, qt, rope_cos, rope_sin, neg);
    const kr = try rope_mod.ropeRotateHalf(gr, kt, rope_cos, rope_sin, neg);
    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(c.head_dim)));
    var attn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn);
    const null_sink = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qr, kr, vt, scale, "array", mask, null_sink, false, s));
    const at = try transpose(attn, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(at);
    const af = try reshape(at, &[_]c_int{ 1, seq, heads * hd }, s);
    defer _ = mlx.mlx_array_free(af);
    const o = try layer.o.raw.forward(af, s);
    defer _ = mlx.mlx_array_free(o);
    const h1 = try addA(x, o, s);
    defer _ = mlx.mlx_array_free(h1);
    const hn = try rms(h1, layer.post_ln, eps, s);
    defer _ = mlx.mlx_array_free(hn);
    const g = try layer.gate.raw.forward(hn, s);
    defer _ = mlx.mlx_array_free(g);
    const sg = try silu(g, s);
    defer _ = mlx.mlx_array_free(sg);
    const u = try layer.up.raw.forward(hn, s);
    defer _ = mlx.mlx_array_free(u);
    const gu = try mulA(sg, u, s);
    defer _ = mlx.mlx_array_free(gu);
    const dn = try layer.down.raw.forward(gu, s);
    defer _ = mlx.mlx_array_free(dn);
    return addA(h1, dn, s);
}

/// acestep.zig `encoderLayer` (weights by field; `lin`'s dense arm is `.raw.forward`).
fn acestepLayer(layer: *const TeLayer, c: Cfg, theta: f32, h_in: mlx.mlx_array, sliding_mask: ?mlx.mlx_array, s: S) !mlx.mlx_array {
    const splitHeads = struct {
        fn f(xx: mlx.mlx_array, heads: c_int, hd: c_int, st: S) !mlx.mlx_array {
            const sh = mlx.getShape(xx);
            const x4 = try reshape(xx, &[_]c_int{ sh[0], sh[1], heads, hd }, st);
            defer _ = mlx.mlx_array_free(x4);
            return transpose(x4, &[_]c_int{ 0, 2, 1, 3 }, st);
        }
    }.f;
    const mergeHeads = struct {
        fn f(xx: mlx.mlx_array, st: S) !mlx.mlx_array {
            const sh = mlx.getShape(xx);
            const t = try transpose(xx, &[_]c_int{ 0, 2, 1, 3 }, st);
            defer _ = mlx.mlx_array_free(t);
            return reshape(t, &[_]c_int{ sh[0], sh[2], sh[1] * sh[3] }, st);
        }
    }.f;
    const ropeInPlace = struct {
        fn f(xx: mlx.mlx_array, hd: c_int, th: f32, st: S) !mlx.mlx_array {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_fast_rope(&o, xx, hd, false, mlx.mlx_optional_float.some(th), 1.0, 0, .{ .ctx = null }, st));
            return o;
        }
    }.f;
    const sdpa = struct {
        fn f(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, mask: ?mlx.mlx_array, mode: [*:0]const u8, st: S) !mlx.mlx_array {
            var o = mlx.mlx_array_new();
            const null_a = mlx.mlx_array{ .ctx = null };
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, mode, mask orelse null_a, null_a, false, st));
            return o;
        }
    }.f;
    const nh = c.heads;
    const nkv = c.kv_heads;
    const hd = c.head_dim;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
    const x = try rms(h_in, layer.input_ln, c.eps, s);
    defer _ = mlx.mlx_array_free(x);
    const q = try layer.q.raw.forward(x, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try layer.k.raw.forward(x, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try layer.v.raw.forward(x, s);
    defer _ = mlx.mlx_array_free(v);
    const qh = try splitHeads(q, nh, hd, s);
    defer _ = mlx.mlx_array_free(qh);
    const kh = try splitHeads(k, nkv, hd, s);
    defer _ = mlx.mlx_array_free(kh);
    const vh = try splitHeads(v, nkv, hd, s);
    defer _ = mlx.mlx_array_free(vh);
    const qn = try rms(qh, layer.q_norm, c.eps, s);
    defer _ = mlx.mlx_array_free(qn);
    const kn = try rms(kh, layer.k_norm, c.eps, s);
    defer _ = mlx.mlx_array_free(kn);
    const qr = try ropeInPlace(qn, hd, theta, s);
    defer _ = mlx.mlx_array_free(qr);
    const kr = try ropeInPlace(kn, hd, theta, s);
    defer _ = mlx.mlx_array_free(kr);
    const attn = if (sliding_mask) |m|
        try sdpa(qr, kr, vh, scale, m, "array", s)
    else
        try sdpa(qr, kr, vh, scale, null, "", s);
    defer _ = mlx.mlx_array_free(attn);
    const merged = try mergeHeads(attn, s);
    defer _ = mlx.mlx_array_free(merged);
    const o = try layer.o.raw.forward(merged, s);
    defer _ = mlx.mlx_array_free(o);
    const h1 = try addA(h_in, o, s);
    defer _ = mlx.mlx_array_free(h1);
    const xm = try rms(h1, layer.post_ln, c.eps, s);
    defer _ = mlx.mlx_array_free(xm);
    const gate = try layer.gate.raw.forward(xm, s);
    defer _ = mlx.mlx_array_free(gate);
    const up = try layer.up.raw.forward(xm, s);
    defer _ = mlx.mlx_array_free(up);
    const gact = try silu(gate, s);
    defer _ = mlx.mlx_array_free(gact);
    const gu = try mulA(gact, up, s);
    defer _ = mlx.mlx_array_free(gu);
    const down = try layer.down.raw.forward(gu, s);
    defer _ = mlx.mlx_array_free(down);
    return addA(h1, down, s);
}

/// minimax_h3 h3.zig `TextEncoder.layerForward`; its `ropeHalf` is the sdk `ropeSplitHalf`
/// it was proven equal to (rope.zig), `forward(x, null, s)` is `.raw.forward(x, s)`.
fn h3Layer(gr: *Graph, layer: *const TeLayer, cfg: Cfg, x: mlx.mlx_array, mask: mlx.mlx_array, cos: A, sin: A, seq: c_int, s: S) !mlx.mlx_array {
    const xn = try rms(x, layer.input_ln, cfg.eps, s);
    defer _ = mlx.mlx_array_free(xn);
    var qkv: [3]mlx.mlx_array = undefined;
    const heads = [3]c_int{ cfg.heads, cfg.kv_heads, cfg.kv_heads };
    inline for (.{ &layer.q, &layer.k, &layer.v }, 0..) |lw, i| {
        const p = try lw.raw.forward(xn, s);
        defer _ = mlx.mlx_array_free(p);
        const r4 = try reshape(p, &[_]c_int{ 1, seq, heads[i], cfg.head_dim }, s);
        if (i == 2) {
            defer _ = mlx.mlx_array_free(r4);
            const t = try transpose(r4, &[_]c_int{ 0, 2, 1, 3 }, s);
            defer _ = mlx.mlx_array_free(t);
            qkv[i] = try kit.contig(t, s);
        } else {
            defer _ = mlx.mlx_array_free(r4);
            const nw = if (i == 0) layer.q_norm else layer.k_norm;
            const nn = try rms(r4, nw, cfg.eps, s);
            defer _ = mlx.mlx_array_free(nn);
            const t = try transpose(nn, &[_]c_int{ 0, 2, 1, 3 }, s);
            defer _ = mlx.mlx_array_free(t);
            const tc = try kit.contig(t, s);
            defer _ = mlx.mlx_array_free(tc);
            qkv[i] = gr.escape(try rope_mod.ropeSplitHalf(gr, tc, cos, sin, @divExact(cfg.head_dim, 2)));
        }
    }
    defer for (&qkv) |*p| {
        _ = mlx.mlx_array_free(p.*);
    };
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    var attn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn);
    const null_sink = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qkv[0], qkv[1], qkv[2], scale, "array", mask, null_sink, false, s));
    const at = try transpose(attn, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(at);
    const af = try reshape(at, &[_]c_int{ 1, seq, cfg.heads * cfg.head_dim }, s);
    defer _ = mlx.mlx_array_free(af);
    const o = try layer.o.raw.forward(af, s);
    defer _ = mlx.mlx_array_free(o);
    const h1 = try addA(x, o, s);
    defer _ = mlx.mlx_array_free(h1);
    const hn = try rms(h1, layer.post_ln, cfg.eps, s);
    defer _ = mlx.mlx_array_free(hn);
    const g = try layer.gate.raw.forward(hn, s);
    defer _ = mlx.mlx_array_free(g);
    const sg = try silu(g, s);
    defer _ = mlx.mlx_array_free(sg);
    const u = try layer.up.raw.forward(hn, s);
    defer _ = mlx.mlx_array_free(u);
    const gu = try mulA(sg, u, s);
    defer _ = mlx.mlx_array_free(gu);
    const d = try layer.down.raw.forward(gu, s);
    defer _ = mlx.mlx_array_free(d);
    return addA(h1, d, s);
}

test "media blocks: the Qwen3 encoder layer is bit-equal to FLUX.2's, Krea's, mage_flow's, ACE-Step's and minimax_h3's" {
    const s = kit.cpu();
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    const cfg: Cfg = .{ .heads = 4, .kv_heads = 2, .head_dim = 16, .eps = 1e-6 };
    const D = 48;
    const F = 80;
    const L = 7;
    const bf = ops.Dtype.bfloat16;
    var seed: u64 = 800;
    const R = struct {
        fn p(gg: *Graph, sd: *u64, o: c_int, i: c_int) !Proj {
            sd.* += 1;
            return .{ .raw = .{ .w = try kit.randn(gg, &.{ o, i }, sd.*, .bfloat16, 0.15) } };
        }
    };
    var w: TeLayer = undefined;
    w.input_ln = try kit.randn(&g, &.{D}, 790, bf, 1);
    w.post_ln = try kit.randn(&g, &.{D}, 791, bf, 1);
    w.q_norm = try kit.randn(&g, &.{16}, 792, bf, 1);
    w.k_norm = try kit.randn(&g, &.{16}, 793, bf, 1);
    w.q = try R.p(&g, &seed, 4 * 16, D);
    w.k = try R.p(&g, &seed, 2 * 16, D);
    w.v = try R.p(&g, &seed, 2 * 16, D);
    w.o = try R.p(&g, &seed, D, 4 * 16);
    w.gate = try R.p(&g, &seed, F, D);
    w.up = try R.p(&g, &seed, F, D);
    w.down = try R.p(&g, &seed, D, F);
    const x = try kit.randn(&g, &.{ 1, L, D }, 850, bf, 1);
    // Additive causal-plus-padding mask, as the encoders build it.
    var mbuf: [L * L]f32 = undefined;
    for (0..L) |r| for (0..L) |c| {
        mbuf[r * L + c] = if (c > r or c == 5) -1e9 else 0;
    };
    const mask = try g.astype(try g.fromSlice(f32, &mbuf, &.{ L, L }), bf);
    const mask32 = try g.fromSlice(f32, &mbuf, &.{ L, L });

    try ops.expectBitEqual(
        try layerForward(&g, &w, cfg, .{ .fast = .{ .theta = 1e6, .attn_f32 = bf } }, x, mask32),
        try kit.own(&g, try fluxLayer(&w, cfg, 1e6, x, mask32, L, s)),
    );
    const cos = try kit.randn(&g, &.{ 1, 1, L, 16 }, 860, bf, 1);
    const sin = try kit.randn(&g, &.{ 1, 1, L, 16 }, 861, bf, 1);
    for ([_]ops.Dtype{ bf, .float32 }) |neg| {
        try ops.expectBitEqual(
            try layerForward(&g, &w, cfg, .{ .table = .{ .cos = cos, .sin = sin, .neg = neg } }, x, mask),
            try kit.own(&g, try kreaLayer(&g, &w, cfg, neg, x, mask, cos, sin, L, s)),
        );
    }
    for ([_]?A{ null, mask }) |mm| {
        try ops.expectBitEqual(
            try layerForward(&g, &w, cfg, .{ .fast = .{ .theta = 1e6 } }, x, mm),
            try kit.own(&g, try acestepLayer(&w, cfg, 1e6, x, mm, s)),
        );
    }
    const ch = try kit.randn(&g, &.{ 1, 1, L, 8 }, 862, bf, 1);
    const sh = try kit.randn(&g, &.{ 1, 1, L, 8 }, 863, bf, 1);
    try ops.expectBitEqual(
        try layerForward(&g, &w, cfg, .{ .half = .{ .cos = ch, .sin = sh } }, x, mask),
        try kit.own(&g, try h3Layer(&g, &w, cfg, x, mask, ch, sh, L, s)),
    );
}
