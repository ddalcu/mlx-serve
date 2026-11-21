//! One visit's self-attention (mlx-lm `inkling2/layers.py` `Attention`, `SlidingAttention`, `FullAttention`):
//! q/k/v/relative-state projections, the k/v short convs, the L2 clips, the value norm, the global layers'
//! log-scaling `tau`, the hidden-conditioned relative bias, the KV append and band view, attention with per-row sinks
//! (fused for a decode row, composed for a prompt chunk), the sigmoid gate and `o_proj`.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const block = @import("../../sdk/bases/block.zig");
const kv = @import("../../sdk/bases/kv.zig");
const sliding = @import("../../sdk/bases/blocks/sliding.zig");
const sdpaSinks = @import("../../sdk/bases/blocks/decoder_attention.zig").sdpaSinks;
const Linear = @import("../../sdk/bases/blocks/linear.zig").Linear;
const conv = @import("conv.zig");
const kernels = @import("kernels.zig");

const Array = ops.Array;
const Graph = ops.Graph;

/// Sliding layers: `conv1d.weight` `[C, K, 1]`; global layers: the dynamic conv.
pub const KConv = union(enum) { short: Array, dynamic: conv.Dyn };

pub const Weights = struct {
    q: Linear,
    k: Linear,
    v: Linear,
    r: Linear,
    o: Linear,
    gate: Linear,
    sink: Linear,
    /// `[H, hd]` (`self_attn.q_l2_clip.scale`).
    q_clip: Array,
    /// `[H, dr]` (`self_attn.rel_l2_clip.scale`).
    rel_clip: Array,
    /// `[dr, R]` (`self_attn.rel_logits_proj.proj`).
    rel_proj: Array,
    k_conv: KConv,
    v_conv: KConv,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    d_rel: u32,
    /// 0 = global.
    window: u32,
    /// Global layers scale q and rel by `tau` past `log_floor` positions.
    has_tau: bool,
    log_floor: u32,
    log_alpha: f32,
    qk_clip: f32,
    rel_clip_thr: f32,
    eps: f32,
    /// f32 `[hd]` of ones, shared, owned by the plugin.
    ones_hd: Array,
};

pub const Out = struct { y: Array, k_tail: Array, v_tail: Array, k_window: Array, v_window: Array };

/// The visit's attention of `q_in` over the keys of `k_in`/`v_in` appended to the cache at `c.layer` (the visit
/// index). `prev_k`/`prev_v`: the k/v convs' previous f32 windows (null = zeros).
pub fn forward(w: *const Weights, c: *block.Ctx, q_in: Array, k_in: Array, v_in: Array, prev_k: ?Array, prev_v: ?Array) !Out {
    const g = c.g;
    const L = c.rows;
    const off = c.offset();
    const total: c_int = @as(c_int, @intCast(off)) + L;
    const H: c_int = @intCast(w.heads);
    const Hk: c_int = @intCast(w.kv_heads);
    const hd: c_int = @intCast(w.head_dim);
    const dr: c_int = @intCast(w.d_rel);

    const gate = try w.gate.forward(g, q_in);
    const sinks = try w.sink.forward(g, q_in);
    var q = try g.reshape(try w.q.forward(g, q_in), &.{ 1, L, H, hd });
    var rel = try g.reshape(try w.r.forward(g, q_in), &.{ 1, L, H, dr });
    const ck = try kvConv(g, &w.k_conv, try w.k.forward(g, k_in), k_in, prev_k);
    const cv = try kvConv(g, &w.v_conv, try w.v.forward(g, v_in), v_in, prev_v);
    const k = try l2Clip(g, try g.reshape(ck.y, &.{ 1, L, Hk, hd }), null, w.qk_clip);
    const v_raw = try g.reshape(cv.y, &.{ 1, L, Hk, hd });
    const v = try g.astype(try g.rmsNorm(try g.astype(v_raw, .float32), w.ones_hd, w.eps), ops.dtype(v_raw));
    q = try l2Clip(g, q, w.q_clip, w.qk_clip);
    rel = try l2Clip(g, rel, w.rel_clip, w.rel_clip_thr);
    // At or below the floor tau is exactly 1, so skipping it is bit-exact.
    if (w.has_tau and total > @as(c_int, @intCast(w.log_floor))) {
        const tau = try logTau(g, off, L, w.log_floor, w.log_alpha);
        q = try byTau(g, q, tau);
        rel = try byTau(g, rel, tau);
    }
    const pb = try g.matmul(rel, try g.astype(w.rel_proj, ops.dtype(rel)));

    const perm = [_]c_int{ 0, 2, 1, 3 };
    const qt = try g.transpose(q, &perm);
    const keep: u32 = if (w.window > 0) sliding.view(w.window, hd, total, L).span else 0;
    var view = try kv.append(c, try g.transpose(k, &perm), try g.transpose(v, &perm), keep);
    defer kv.release(c, &view);
    const dv = kv.dense(view);
    const S = ops.shape(dv.k)[2];
    const kv_start = total - S;
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(hd))));
    const att = if (L == 1)
        try decodeAttention(g, qt, dv.k, dv.v, pb, sinks, @intCast(off), kv_start, w.window, scale)
    else
        try kernels.relativeAttention(g, qt, dv.k, dv.v, pb, sinks, scale, w.window, S - total) orelse
            try referenceAttention(g, qt, dv.k, dv.v, pb, sinks, off, @intCast(kv_start), w.window, scale);

    const flat = try g.astype(try g.reshape(att, &.{ 1, L, H * hd }), .float32);
    const gated = try g.astype(try g.mul(flat, try g.sigmoid(try g.astype(gate, .float32))), ops.dtype(gate));
    return .{ .y = try w.o.forward(g, gated), .k_tail = ck.tail, .v_tail = cv.tail, .k_window = ck.window, .v_window = cv.window };
}

/// The composed prefill: q `[1,H,L,hd]`, k/v `[1,Hk,S,hd]`, pb `[1,L,H,R]`, sinks `[1,L,H]`; query i sits at
/// `off + i`, key j at `kv_start + j`. Returns `[1,L,H,hd]` in v's dtype.
// ceiling: O(H*L*S) memory per chunk; upgrade path: kernels.relativeAttention
pub fn referenceAttention(g: *Graph, q: Array, k: Array, v: Array, pb: Array, sinks: Array, off: u32, kv_start: u32, window: u32, scale: f32) !Array {
    var t = g.scope();
    defer t.deinit();
    const s = &t;
    const H = ops.shape(q)[1];
    const L = ops.shape(q)[2];
    const hd = ops.shape(q)[3];
    const Hk = ops.shape(k)[1];
    const S = ops.shape(k)[2];
    const R = ops.shape(pb)[3];
    const G = @divExact(H, Hk);
    // Serial unpadded rows have no key before position 0, so mlx-lm's `key_positions >= 0` term is dropped; a
    // batched decode must restore it.
    const d = try distances(s, @intCast(off), L, @intCast(kv_start), S);
    const idx = try s.broadcast(try clipInt(s, d, R - 1), &.{ 1, H, L, S });
    var bias = try s.takeAlong(try s.transpose(pb, &.{ 0, 2, 1, 3 }), idx, 3);
    bias = try s.where(try s.less(d, try int(s, R)), try s.astype(bias, .float32), try s.scalar(0, .float32));
    const neg = try s.scalar(-std.math.inf(f32), .float32);
    const mask = try s.reshape(try s.where(try visible(s, d, window), bias, neg), &.{ 1, Hk, G, L, S });
    const qf = try s.reshape(try s.astype(q, .float32), &.{ 1, Hk, G, L, hd });
    const kf = try s.transpose(try s.expandDims(try s.astype(k, .float32), 2), &.{ 0, 1, 2, 4, 3 });
    const lg = try s.add(try s.mul(try s.matmul(qf, kf), try s.scalar(scale, .float32)), mask);
    const sk = try s.reshape(try s.transpose(try s.astype(sinks, .float32), &.{ 0, 2, 1 }), &.{ 1, Hk, G, L, 1 });
    const p = try s.sliceAxis(try s.softmax(try s.concat(&.{ lg, sk }, 4), 4, false), 4, 0, S);
    const o = try s.matmul(try s.astype(p, ops.dtype(v)), try s.expandDims(v, 2));
    return g.keep(try s.transpose(try s.reshape(o, &.{ 1, H, L, hd }), &.{ 0, 2, 1, 3 }));
}

/// One query row over the view's keys through the fused attention with sinks: the relative bias and the band as its
/// additive mask. q `[1,H,1,hd]`; returns `[1,1,H,hd]`.
fn decodeAttention(g: *Graph, q: Array, k: Array, v: Array, pb: Array, sinks: Array, off: c_int, kv_start: c_int, window: u32, scale: f32) !Array {
    var t = g.scope();
    defer t.deinit();
    const s = &t;
    const H = ops.shape(pb)[2];
    const R = ops.shape(pb)[3];
    const S = ops.shape(k)[2];
    const d = try s.reshape(try distances(s, off, 1, kv_start, S), &.{S});
    var bias = try s.take(try s.reshape(pb, &.{ H, R }), try clipInt(s, d, R - 1), 1);
    bias = try s.where(try s.less(d, try int(s, R)), bias, try s.scalar(0, ops.dtype(bias)));
    const neg = try s.scalar(-std.math.inf(f32), ops.dtype(bias));
    const mask = try s.astype(try s.where(try visible(s, d, window), bias, neg), ops.dtype(q));
    const sk = try s.astype(try s.reshape(sinks, &.{H}), ops.dtype(q));
    const att = try sdpaSinks(s, q, k, v, scale, "array", try s.reshape(mask, &.{ 1, H, 1, S }), sk, false);
    return g.keep(try s.transpose(att, &.{ 0, 2, 1, 3 }));
}

fn kvConv(g: *Graph, kc: *const KConv, x: Array, dyn_in: Array, prev: ?Array) !conv.ConvOut {
    return switch (kc.*) {
        .short => |w| conv.shortConv(g, x, w, prev),
        .dynamic => |*d| conv.dynamicConv(g, x, dyn_in, d, prev),
    };
}

/// `x * min(thr / (|x * scale| + 1e-6), 1)` over the last axis in f32 (scale null = 1), back in x's dtype.
fn l2Clip(g: *Graph, x: Array, scale: ?Array, thr: f32) !Array {
    var t = g.scope();
    defer t.deinit();
    const s = &t;
    var st = try s.astype(x, .float32);
    if (scale) |sc| st = try s.mul(st, try s.astype(sc, .float32));
    const n = try s.sqrt(try s.sum(try s.mul(st, st), 3, true));
    const f = try s.minimum(try s.div(try s.scalar(thr, .float32), try s.addScalar(n, 1e-6)), try s.scalar(1, .float32));
    return g.keep(try s.astype(try s.mul(st, f), ops.dtype(x)));
}

/// `1 + alpha * log(max((pos + 1) / floor, 1))` at positions `off..off+rows`, f32 `[1, rows, 1, 1]`.
fn logTau(g: *Graph, off: u32, rows: c_int, floor: u32, alpha: f32) !Array {
    var t = g.scope();
    defer t.deinit();
    const s = &t;
    const start: f64 = @floatFromInt(off);
    const pos = try s.arange(start, start + @as(f64, @floatFromInt(rows)), 1, .float32);
    const r = try s.div(try s.addScalar(pos, 1), try s.scalar(@floatFromInt(floor), .float32));
    const tau = try s.addScalar(try s.mulScalar(try s.log(try s.maximum(r, try s.scalar(1, .float32))), alpha), 1);
    return g.keep(try s.reshape(tau, &.{ 1, rows, 1, 1 }));
}

/// `x` scaled by `tau` in f32, back in x's dtype.
fn byTau(g: *Graph, x: Array, tau: Array) !Array {
    var t = g.scope();
    defer t.deinit();
    return g.keep(try t.astype(try t.mul(try t.astype(x, .float32), tau), ops.dtype(x)));
}

/// `(off + i) - (kv_start + j)`, int32 `[rows, keys]`.
fn distances(g: *Graph, off: c_int, rows: c_int, kv_start: c_int, keys: c_int) !Array {
    const qp = try g.reshape(try g.arange(@floatFromInt(off), @floatFromInt(off + rows), 1, .int32), &.{ rows, 1 });
    const kp = try g.reshape(try g.arange(@floatFromInt(kv_start), @floatFromInt(kv_start + keys), 1, .int32), &.{ 1, keys });
    return g.sub(qp, kp);
}

/// A key at distance `d` is visible when it is not ahead of the query and, on a sliding layer, inside the window.
fn visible(g: *Graph, d: Array, window: u32) !Array {
    const ahead_ok = try g.call(mlx.mlx_greater_equal, .{ d, try int(g, 0) });
    if (window == 0) return ahead_ok;
    return g.call(mlx.mlx_logical_and, .{ ahead_ok, try g.less(d, try int(g, @intCast(window))) });
}

fn clipInt(g: *Graph, d: Array, hi: c_int) !Array {
    return g.minimum(try g.maximum(d, try int(g, 0)), try int(g, hi));
}

fn int(g: *Graph, v: c_int) !Array {
    return g.adopt(mlx.mlx_array_new_int(v));
}

// ── tests ──

const testing = std.testing;
const abi = @import("../../abi/plugin.zig");
const parity = @import("../../sdk/testing/parity.zig");
const test_cache = @import("../../sdk/testing/cache_ops.zig");

const T = struct {
    const D = 16;
    const H = 4;
    const Hk = 2;
    const hd = 8;
    const dr = 4;
    const K = 4;

    fn randn(g: *Graph, sh: []const c_int, seed: u64, sd: f32) !Array {
        const key = try g.call(mlx.mlx_random_key, .{seed});
        return g.call(mlx.mlx_random_normal, .{ sh.ptr, sh.len, ops.Dtype.float32, @as(f32, 0), sd, key });
    }

    fn lin(g: *Graph, in: c_int, out: c_int, seed: u64) !Linear {
        return .{ .w = try randn(g, &.{ in, out }, seed, 0.3) };
    }

    /// Random f32 weights; a sliding layer of window 4 (R 4), or a global one of R 3 with tau past 4 positions.
    fn weights(g: *Graph, global: bool) !Weights {
        const R: c_int = if (global) 3 else 4;
        const C = Hk * hd;
        var kv_conv: [2]KConv = undefined;
        for (&kv_conv, 0..) |*kc, i| kc.* = if (global)
            .{ .dynamic = .{ .static = try randn(g, &.{ K, C }, 40 + i, 0.3), .proj = try lin(g, D, Hk * K, 50 + i), .heads = Hk } }
        else
            .{ .short = try randn(g, &.{ C, K, 1 }, 40 + i, 0.3) };
        return .{
            .q = try lin(g, D, H * hd, 1),
            .k = try lin(g, D, C, 2),
            .v = try lin(g, D, C, 3),
            .r = try lin(g, D, H * dr, 4),
            .o = try lin(g, H * hd, D, 5),
            .gate = try lin(g, D, H * hd, 6),
            .sink = try lin(g, D, H, 7),
            .q_clip = try g.addScalar(try randn(g, &.{ H, hd }, 8, 0.2), 1),
            .rel_clip = try g.addScalar(try randn(g, &.{ H, dr }, 9, 0.2), 1),
            .rel_proj = try randn(g, &.{ dr, R }, 10, 1),
            .k_conv = kv_conv[0],
            .v_conv = kv_conv[1],
            .heads = H,
            .kv_heads = Hk,
            .head_dim = hd,
            .d_rel = dr,
            .window = if (global) 0 else 4,
            .has_tau = global,
            .log_floor = 4,
            .log_alpha = 0.5,
            .qk_clip = 2,
            .rel_clip_thr = 1.5,
            .eps = 1e-6,
            .ones_hd = try g.ones(&.{hd}, .float32),
        };
    }

    /// Rows `[from, to)` of the three inputs through `forward` on `cache`.
    fn run(g: *Graph, w: *const Weights, cache: *test_cache.TestCache, shared: *const block.Shared, xs: [3]Array, from: c_int, to: c_int, prev: ?[2]Array) !Out {
        const h = [_]abi.Handle{cache.handle()};
        const pos = [_]u32{@intCast(from)};
        var c: block.Ctx = .{ .g = g, .mode = if (to - from == 1) .decode else .prefill, .rows = to - from, .positions = &pos, .caches = &h, .cache_ops = &test_cache.ops, .shared = shared };
        var in: [3]Array = undefined;
        for (&in, xs) |*o, x| o.* = try g.sliceAxis(x, 1, from, to);
        return forward(w, &c, in[0], in[1], in[2], if (prev) |p| p[0] else null, if (prev) |p| p[1] else null);
    }
};

// Bar: the last of 9 prompt rows equals 8 prompt rows then one decode row (the band dropping old keys on the
// sliding layer, tau and a relative extent shorter than the prompt on the global one): cos > 0.9999.
test "inkling2 attention: a one-row decode equals the prefill reference" {
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const N = 9;
    for ([_]bool{ false, true }) |global| {
        var g = Graph.init(a, s);
        defer g.deinit();
        var shared = block.Shared.init(1e-6, 1e-6);
        defer shared.deinit();
        const w = try T.weights(&g, global);
        var xs: [3]Array = undefined;
        for (&xs, 0..) |*x, i| x.* = try T.randn(&g, &.{ 1, N, T.D }, 100 + i, 1);

        var full_cache = try test_cache.TestCache.init(a, 1);
        defer full_cache.deinit();
        const full = try T.run(&g, &w, &full_cache, &shared, xs, 0, N, null);
        const want = try parity.evalToF32(a, try g.sliceAxis(full.y, 1, N - 1, N), T.D, s);
        defer a.free(want);

        var step_cache = try test_cache.TestCache.init(a, 1);
        defer step_cache.deinit();
        const pre = try T.run(&g, &w, &step_cache, &shared, xs, 0, N - 1, null);
        const dec = try T.run(&g, &w, &step_cache, &shared, xs, N - 1, N, .{ pre.k_tail, pre.v_tail });
        const got = try parity.evalToF32(a, dec.y, T.D, s);
        defer a.free(got);
        try testing.expect(parity.qwen4CompareRows(got, want, 1, T.D).min_cos > 0.9999);
    }
}

// Bar: bit equality at or below the floor (so the forward may skip tau there); past it tau changes the values.
test "inkling2 attention: tau at or below the log-scaling floor leaves q and rel bit-equal" {
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(a, s);
    defer g.deinit();
    const floor = 6;
    const q = try g.astype(try T.randn(&g, &.{ 1, floor, T.H, T.hd }, 1, 1), .bfloat16);
    const rel = try g.astype(try T.randn(&g, &.{ 1, floor, T.H, T.dr }, 2, 1), .bfloat16);
    const at = try logTau(&g, 0, floor, floor, 0.5);
    try ops.expectBitEqual(try byTau(&g, q, at), q);
    try ops.expectBitEqual(try byTau(&g, rel, at), rel);
    const past = try logTau(&g, floor, floor, floor, 0.5);
    try testing.expect(!try parity.bf16BitEqual(try byTau(&g, q, past), q, s));
}
