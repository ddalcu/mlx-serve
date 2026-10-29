//! RoPE applied from precomputed cos/sin tables (the media DiTs build their own multi-axis
//! tables), three layouts. `cos`/`sin` must already broadcast against the rotated half.
//! - `ropePairs`: interleaved (real, imag) pairs, computed in f32, cast to `out`
//!   (FLUX.2 + Krea DiT [B,H,L,D], mage_flow DiT [B,L,H,D]).
//! - `ropeRotateHalf`: x·cos + rotate_half(x)·sin over full-width tables, the -1 made in
//!   `neg_dt` (Krea text encoder bf16, mage_flow text encoder f32).
//! - `ropeSplitHalf`: first/second half against half-width tables, an optional untouched tail
//!   past `2·half` (MOSS-Image, minimax_h3 `ropeHalf`/`applyRopePub`, LTX `applyRopeSplit`).

const std = @import("std");
const ops = @import("../../../mlx/ops.zig");

const Graph = ops.Graph;
const Array = ops.Array;
const Dtype = ops.Dtype;

pub fn ropePairs(parent: *Graph, x: Array, cos: Array, sin: Array, out: Dtype) !Array {
    var g = parent.scope();
    defer g.deinit();
    const sh = ops.shape(x);
    const n = sh.len;
    var pair: [8]c_int = undefined;
    @memcpy(pair[0..n], sh);
    pair[n - 1] = @divExact(sh[n - 1], 2);
    pair[n] = 2;
    const x5 = try g.reshape(try g.astype(x, .float32), pair[0 .. n + 1]);
    const re = try g.reshape(try g.sliceAxis(x5, n, 0, 1), pair[0..n]);
    const im = try g.reshape(try g.sliceAxis(x5, n, 1, 2), pair[0..n]);
    const r = try g.sub(try g.mul(re, cos), try g.mul(im, sin));
    const i = try g.add(try g.mul(im, cos), try g.mul(re, sin));
    pair[n] = 1;
    const st = try g.concat(&.{ try g.reshape(r, pair[0 .. n + 1]), try g.reshape(i, pair[0 .. n + 1]) }, @intCast(n));
    return parent.keep(try g.astype(try g.reshape(st, sh), out));
}

pub fn ropeRotateHalf(parent: *Graph, x: Array, cos: Array, sin: Array, neg_dt: Dtype) !Array {
    var g = parent.scope();
    defer g.deinit();
    const sh = ops.shape(x);
    const last = sh.len - 1;
    const half = @divExact(sh[last], 2);
    const x1 = try g.sliceAxis(x, last, 0, half);
    const x2 = try g.sliceAxis(x, last, half, sh[last]);
    const rh = try g.concat(&.{ try g.mul(x2, try g.scalar(-1.0, neg_dt)), x1 }, @intCast(last));
    return parent.keep(try g.add(try g.mul(x, cos), try g.mul(rh, sin)));
}

pub fn ropeSplitHalf(parent: *Graph, x: Array, cos: Array, sin: Array, half: c_int) !Array {
    var g = parent.scope();
    defer g.deinit();
    const sh = ops.shape(x);
    const last = sh.len - 1;
    const x1 = try g.sliceAxis(x, last, 0, half);
    const x2 = try g.sliceAxis(x, last, half, 2 * half);
    const lo = try g.sub(try g.mul(x1, cos), try g.mul(x2, sin));
    const hi = try g.add(try g.mul(x2, cos), try g.mul(x1, sin));
    if (2 * half == sh[last]) return parent.keep(try g.concat(&.{ lo, hi }, @intCast(last)));
    return parent.keep(try g.concat(&.{ lo, hi, try g.sliceAxis(x, last, 2 * half, sh[last]) }, @intCast(last)));
}

// ── tests ──

const testing = std.testing;
const mlx = @import("../../../mlx/ffi.zig");
const kit = @import("testkit.zig");
const scope_mod = @import("../../../mlx/scope.zig");
const S = kit.S;
const A = kit.A;
const astype = kit.astype;
const reshape = kit.reshape;
const mulA = kit.mulA;
const addA = kit.addA;
const subA = kit.subA;
const concat = kit.concat;
const sliceAxis = kit.sliceAxis;
const scalarF = kit.scalarF;

fn sliceLast2(x: mlx.mlx_array, idx: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [1,H,seq,64,2]
    var lo = [_]c_int{ 0, 0, 0, 0, idx };
    var hi = [_]c_int{ sh[0], sh[1], sh[2], sh[3], idx + 1 };
    const st = [_]c_int{ 1, 1, 1, 1, 1 };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &lo, 5, &hi, 5, &st, 5, s));
    const sq = try reshape(o, &[_]c_int{ sh[0], sh[1], sh[2], sh[3] }, s);
    _ = mlx.mlx_array_free(o);
    return sq;
}

/// flux.zig `Dit.applyRope` (krea.zig `applyRope` is this sequence).
fn fluxApplyRope(q: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, seq: c_int, heads: c_int, s: S) !mlx.mlx_array {
    const qf = try astype(q, .float32, s);
    defer _ = mlx.mlx_array_free(qf);
    const q5 = try reshape(qf, &[_]c_int{ 1, heads, seq, 64, 2 }, s); // [.,64,2]
    defer _ = mlx.mlx_array_free(q5);
    const real = try sliceLast2(q5, 0, s);
    defer _ = mlx.mlx_array_free(real);
    const imag = try sliceLast2(q5, 1, s);
    defer _ = mlx.mlx_array_free(imag);
    const cos_b = try reshape(cos, &[_]c_int{ 1, 1, seq, 64 }, s);
    defer _ = mlx.mlx_array_free(cos_b);
    const sin_b = try reshape(sin, &[_]c_int{ 1, 1, seq, 64 }, s);
    defer _ = mlx.mlx_array_free(sin_b);
    const rc = try mulA(real, cos_b, s);
    defer _ = mlx.mlx_array_free(rc);
    const is_ = try mulA(imag, sin_b, s);
    defer _ = mlx.mlx_array_free(is_);
    const out0 = try subA(rc, is_, s);
    defer _ = mlx.mlx_array_free(out0);
    const ic = try mulA(imag, cos_b, s);
    defer _ = mlx.mlx_array_free(ic);
    const rs2 = try mulA(real, sin_b, s);
    defer _ = mlx.mlx_array_free(rs2);
    const out1 = try addA(ic, rs2, s);
    defer _ = mlx.mlx_array_free(out1);
    const o0e = try reshape(out0, &[_]c_int{ 1, heads, seq, 64, 1 }, s);
    defer _ = mlx.mlx_array_free(o0e);
    const o1e = try reshape(out1, &[_]c_int{ 1, heads, seq, 64, 1 }, s);
    defer _ = mlx.mlx_array_free(o1e);
    const st = try concat(&[_]mlx.mlx_array{ o0e, o1e }, 4, s);
    defer _ = mlx.mlx_array_free(st);
    const flat = try reshape(st, &[_]c_int{ 1, heads, seq, 128 }, s);
    defer _ = mlx.mlx_array_free(flat);
    return astype(flat, .bfloat16, s);
}

/// mage_flow.zig `applyRope` (its `splitEqual` inlined).
fn mageApplyRope(x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, s: S) !mlx.mlx_array {
    const in_dt = mlx.mlx_array_dtype(x);
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const sh = mlx.getShape(xf); // [B,L,H,128]
    const B = sh[0];
    const L = sh[1];
    const Hh = sh[2];
    const D = sh[3];
    const pairs = try reshape(xf, &[_]c_int{ B, L, Hh, @divExact(D, 2), 2 }, s);
    defer _ = mlx.mlx_array_free(pairs);
    var ri: [2]mlx.mlx_array = undefined;
    {
        var vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_split(&vec, pairs, 2, 4, s));
        for (0..2) |i| {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&o, vec, i));
            ri[i] = o;
        }
    }
    defer for (ri) |x2| {
        _ = mlx.mlx_array_free(x2);
    };
    const real = try reshape(ri[0], &[_]c_int{ B, L, Hh, @divExact(D, 2) }, s);
    defer _ = mlx.mlx_array_free(real);
    const imag = try reshape(ri[1], &[_]c_int{ B, L, Hh, @divExact(D, 2) }, s);
    defer _ = mlx.mlx_array_free(imag);
    const cos4 = try reshape(cos, &[_]c_int{ 1, L, 1, @divExact(D, 2) }, s);
    defer _ = mlx.mlx_array_free(cos4);
    const sin4 = try reshape(sin, &[_]c_int{ 1, L, 1, @divExact(D, 2) }, s);
    defer _ = mlx.mlx_array_free(sin4);
    const rc = try mulA(real, cos4, s);
    defer _ = mlx.mlx_array_free(rc);
    const is_ = try mulA(imag, sin4, s);
    defer _ = mlx.mlx_array_free(is_);
    const rot_r = try subA(rc, is_, s);
    defer _ = mlx.mlx_array_free(rot_r);
    const rs = try mulA(real, sin4, s);
    defer _ = mlx.mlx_array_free(rs);
    const ic = try mulA(imag, cos4, s);
    defer _ = mlx.mlx_array_free(ic);
    const rot_i = try addA(rs, ic, s);
    defer _ = mlx.mlx_array_free(rot_i);
    const rr5 = try reshape(rot_r, &[_]c_int{ B, L, Hh, @divExact(D, 2), 1 }, s);
    defer _ = mlx.mlx_array_free(rr5);
    const ri5 = try reshape(rot_i, &[_]c_int{ B, L, Hh, @divExact(D, 2), 1 }, s);
    defer _ = mlx.mlx_array_free(ri5);
    const stacked = try concat(&.{ rr5, ri5 }, 4, s);
    defer _ = mlx.mlx_array_free(stacked);
    const flat = try reshape(stacked, &[_]c_int{ B, L, Hh, D }, s); // f32
    if (in_dt == .float32) return flat;
    defer _ = mlx.mlx_array_free(flat);
    return astype(flat, in_dt, s);
}

/// krea.zig `applyRopeEncoder` (bf16 -1).
fn kreaApplyRopeEncoder(x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [1,H,L,hd]
    const L = sh[2];
    const hd = sh[3];
    const half = @divExact(hd, 2);
    const cos_b = try reshape(cos, &[_]c_int{ 1, 1, L, hd }, s);
    defer _ = mlx.mlx_array_free(cos_b);
    const sin_b = try reshape(sin, &[_]c_int{ 1, 1, L, hd }, s);
    defer _ = mlx.mlx_array_free(sin_b);
    const xc = try mulA(x, cos_b, s);
    defer _ = mlx.mlx_array_free(xc);
    const x1 = try sliceAxis(x, 3, 0, half, s);
    defer _ = mlx.mlx_array_free(x1);
    const x2 = try sliceAxis(x, 3, half, hd, s);
    defer _ = mlx.mlx_array_free(x2);
    const neg1 = try astype(scalarF(-1.0), .bfloat16, s);
    defer _ = mlx.mlx_array_free(neg1);
    const nx2 = try mulA(x2, neg1, s);
    defer _ = mlx.mlx_array_free(nx2);
    const rh = try concat(&[_]mlx.mlx_array{ nx2, x1 }, 3, s);
    defer _ = mlx.mlx_array_free(rh);
    const rs = try mulA(rh, sin_b, s);
    defer _ = mlx.mlx_array_free(rs);
    return addA(xc, rs, s);
}

/// mage_flow.zig `applyRopeHalf` (f32 -1).
fn mageApplyRopeHalf(x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [1,H,L,hd]
    const L = sh[2];
    const hd = sh[3];
    const half = @divExact(hd, 2);
    const cos_b = try reshape(cos, &[_]c_int{ 1, 1, L, hd }, s);
    defer _ = mlx.mlx_array_free(cos_b);
    const sin_b = try reshape(sin, &[_]c_int{ 1, 1, L, hd }, s);
    defer _ = mlx.mlx_array_free(sin_b);
    const xc = try mulA(x, cos_b, s);
    defer _ = mlx.mlx_array_free(xc);
    const x1 = try sliceAxis(x, 3, 0, half, s);
    defer _ = mlx.mlx_array_free(x1);
    const x2 = try sliceAxis(x, 3, half, hd, s);
    defer _ = mlx.mlx_array_free(x2);
    const neg1 = scalarF(-1.0);
    defer _ = mlx.mlx_array_free(neg1);
    const nx2 = try mulA(x2, neg1, s);
    defer _ = mlx.mlx_array_free(nx2);
    const rh = try concat(&.{ nx2, x1 }, 3, s);
    defer _ = mlx.mlx_array_free(rh);
    const rs = try mulA(rh, sin_b, s);
    defer _ = mlx.mlx_array_free(rs);
    return addA(xc, rs, s);
}

/// moss_image.zig `rotate` (Scope).
fn mossRotate(sc: *scope_mod.Scope, x: A, cos: A, sin: A) !A {
    const d = mlx.getShape(x)[3];
    const x1 = try sc.slice(x, 3, 0, @divExact(d, 2));
    const x2 = try sc.slice(x, 3, @divExact(d, 2), d);
    const lo = try sc.sub(try sc.mul(x1, cos), try sc.mul(x2, sin));
    const hi = try sc.add(try sc.mul(x2, cos), try sc.mul(x1, sin));
    return sc.concat(&.{ lo, hi }, 3);
}

/// minimax_h3 h3.zig `applyRopePub` (`ropeHalf` is this without the tail; sliceLast4 = slice + contiguous).
fn h3ApplyRopePub(x: mlx.mlx_array, cos: A, sin: A, half: c_int, head_dim: c_int, s: S) !mlx.mlx_array {
    const sliceLast4 = struct {
        fn f(xx: mlx.mlx_array, lo: c_int, hi: c_int, st: S) !mlx.mlx_array {
            const o = try sliceAxis(xx, 3, lo, hi, st);
            defer _ = mlx.mlx_array_free(o);
            return kit.contig(o, st);
        }
    }.f;
    const x1 = try sliceLast4(x, 0, half, s);
    defer _ = mlx.mlx_array_free(x1);
    const x2 = try sliceLast4(x, half, half * 2, s);
    defer _ = mlx.mlx_array_free(x2);
    const a1 = try mulA(x1, cos, s);
    defer _ = mlx.mlx_array_free(a1);
    const a2 = try mulA(x2, sin, s);
    defer _ = mlx.mlx_array_free(a2);
    var o1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o1);
    try mlx.check(mlx.mlx_subtract(&o1, a1, a2, s));
    const b1 = try mulA(x1, sin, s);
    defer _ = mlx.mlx_array_free(b1);
    const b2 = try mulA(x2, cos, s);
    defer _ = mlx.mlx_array_free(b2);
    const o2 = try addA(b1, b2, s);
    defer _ = mlx.mlx_array_free(o2);
    if (half * 2 == head_dim) return concat(&[_]mlx.mlx_array{ o1, o2 }, 3, s);
    const tail = try sliceLast4(x, half * 2, head_dim, s);
    defer _ = mlx.mlx_array_free(tail);
    return concat(&[_]mlx.mlx_array{ o1, o2, tail }, 3, s);
}

test "media blocks: the three RoPE layouts are bit-equal to the backends' copies" {
    const s = kit.cpu();
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    const L = 6;
    const cos64 = try kit.randn(&g, &.{ L, 64 }, 200, .float32, 1);
    const sin64 = try kit.randn(&g, &.{ L, 64 }, 201, .float32, 1);
    for ([_]Dtype{ .bfloat16, .float32 }, 0..) |dt, i| {
        // Heads-first [1, H, L, 128] (FLUX.2 / Krea).
        const q = try kit.randn(&g, &.{ 1, 3, L, 128 }, 210 + i, dt, 1);
        try ops.expectBitEqual(
            try ropePairs(&g, q, try g.reshape(cos64, &.{ 1, 1, L, 64 }), try g.reshape(sin64, &.{ 1, 1, L, 64 }), .bfloat16),
            try kit.own(&g, try fluxApplyRope(q, cos64, sin64, L, 3, s)),
        );
        // Sequence-first [B, L, H, 128] (mage_flow).
        const xs = try kit.randn(&g, &.{ 2, L, 3, 128 }, 220 + i, dt, 1);
        try ops.expectBitEqual(
            try ropePairs(&g, xs, try g.reshape(cos64, &.{ 1, L, 1, 64 }), try g.reshape(sin64, &.{ 1, L, 1, 64 }), dt),
            try kit.own(&g, try mageApplyRope(xs, cos64, sin64, s)),
        );
        // Full-width rotate-half tables.
        const xe = try kit.randn(&g, &.{ 1, 3, L, 32 }, 230 + i, dt, 1);
        const cf = try kit.randn(&g, &.{ L, 32 }, 240 + i, dt, 1);
        const sf = try kit.randn(&g, &.{ L, 32 }, 250 + i, dt, 1);
        const cb = try g.reshape(cf, &.{ 1, 1, L, 32 });
        const sb = try g.reshape(sf, &.{ 1, 1, L, 32 });
        try ops.expectBitEqual(try ropeRotateHalf(&g, xe, cb, sb, .bfloat16), try kit.own(&g, try kreaApplyRopeEncoder(xe, cf, sf, s)));
        try ops.expectBitEqual(try ropeRotateHalf(&g, xe, cb, sb, .float32), try kit.own(&g, try mageApplyRopeHalf(xe, cf, sf, s)));
        // Half tables [1, L, 1, D/2], with and without a pass-through tail.
        const xm = try kit.randn(&g, &.{ 1, L, 3, 40 }, 260 + i, dt, 1);
        const ch = try kit.randn(&g, &.{ 1, L, 1, 16 }, 270 + i, dt, 1);
        const shh = try kit.randn(&g, &.{ 1, L, 1, 16 }, 280 + i, dt, 1);
        try ops.expectBitEqual(try ropeSplitHalf(&g, xm, ch, shh, 16), try kit.own(&g, try h3ApplyRopePub(xm, ch, shh, 16, 40, s)));
        const xm2 = try g.sliceAxis(xm, 3, 0, 32);
        const split = try ropeSplitHalf(&g, xm2, ch, shh, 16);
        try ops.expectBitEqual(split, try kit.own(&g, try h3ApplyRopePub(xm2, ch, shh, 16, 32, s)));
        var sc = scope_mod.Scope.init(testing.allocator, s);
        defer sc.deinit();
        try ops.expectBitEqual(split, try mossRotate(&sc, xm2, ch, shh));
    }
}
