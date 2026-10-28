//! The LFM trunks LFM3's speech stack runs over a whole sequence with no cache: the speech encoder's (LFM2 layers,
//! causal over a window), the depth decoder's (LFM3 attention layers, bidirectional) and the codec decoder's (LFM3
//! layers with sandwich norms, causal over a window). Each layer is a gated short conv (`out(C * conv(B * x))`, causal)
//! or q/k-normed attention under full RoPE, optionally gated per head, then a SwiGLU; the final norm closes it.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const reader = @import("../../sdk/bases/reader.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Linear = bases.blocks.linear.Linear;
const gpa = std.heap.c_allocator;

pub const Mask = union(enum) {
    /// Every row reads every row (the depth decoder's 12 codebook slots).
    bidirectional,
    /// A row reads itself and the `window - 1` before it.
    causal: u32,
};

const ConvLayer = struct { in_proj: Linear, conv: Array, out_proj: Linear };
const Attn = struct { q: Linear, k: Linear, v: Linear, o: Linear, q_norm: Array, k_norm: Array, gate: Linear = .{}, head_dim: c_int };

const Layer = struct {
    operator_norm: Array,
    ffn_norm: Array,
    /// The sandwich: each block's output normed before the residual (empty = none).
    post_operator_norm: Array = .{},
    post_ffn_norm: Array = .{},
    mixer: union(enum) { conv: ConvLayer, attn: Attn },
    w1: Linear,
    w2: Linear,
    w3: Linear,
};

pub const Trunk = struct {
    layers: []Layer,
    final_norm: Array,
    hidden: c_int,
    eps: f32,
    theta: f32,
    mask: Mask,

    /// `<prefix>.layers.<i>.*` and `<prefix>.embedding_norm.weight`; `attention[i]` says which layers attend.
    pub fn load(r: *reader.Reader, owned: *reader.Owned, prefix: []const u8, attention: []const bool, hidden: u32, eps: f32, theta: f32, mask: Mask) !Trunk {
        var t: Trunk = .{
            .layers = try gpa.alloc(Layer, attention.len),
            .final_norm = undefined,
            .hidden = @intCast(hidden),
            .eps = eps,
            .theta = theta,
            .mask = mask,
        };
        errdefer gpa.free(t.layers);
        var nb: [192]u8 = undefined;
        t.final_norm = try arr(r, owned, try std.fmt.bufPrint(&nb, "{s}.embedding_norm.weight", .{prefix}));
        for (t.layers, attention, 0..) |*L, attends, i| {
            const At = struct {
                p: []const u8,
                i: usize,
                fn at(s: @This(), b: *[192]u8, leaf: []const u8) ![]const u8 {
                    return std.fmt.bufPrint(b, "{s}.layers.{d}.{s}", .{ s.p, s.i, leaf });
                }
            };
            const n: At = .{ .p = prefix, .i = i };
            const w1 = try lin(r, owned, try n.at(&nb, "feed_forward.w1"), hidden);
            L.* = .{
                .operator_norm = try arr(r, owned, try n.at(&nb, "operator_norm.weight")),
                .ffn_norm = try arr(r, owned, try n.at(&nb, "ffn_norm.weight")),
                .post_operator_norm = try opt(r, owned, try n.at(&nb, "post_operator_norm.weight")),
                .post_ffn_norm = try opt(r, owned, try n.at(&nb, "post_ffn_norm.weight")),
                .mixer = if (attends) .{ .attn = try attnAt(r, owned, n, &nb, hidden) } else .{ .conv = .{
                    .in_proj = try lin(r, owned, try n.at(&nb, "conv.in_proj"), hidden),
                    .conv = blk: {
                        const w = try reader.depthwiseConv(try r.need(try n.at(&nb, "conv.conv.weight")), r.s);
                        try owned.own(w);
                        break :blk w;
                    },
                    .out_proj = try lin(r, owned, try n.at(&nb, "conv.out_proj"), hidden),
                } },
                .w1 = w1,
                .w2 = try lin(r, owned, try n.at(&nb, "feed_forward.w2"), @intCast(ops.shape(w1.w)[0])),
                .w3 = try lin(r, owned, try n.at(&nb, "feed_forward.w3"), hidden),
            };
        }
        return t;
    }

    /// LFM2 spells the attention's norms `q_layernorm` and its output `out_proj`, LFM3 `q_norm` and `o_proj` (and
    /// gates its heads); the head width is the norm's, the head counts the projections' outputs.
    fn attnAt(r: *reader.Reader, owned: *reader.Owned, n: anytype, nb: *[192]u8, hidden: u32) !Attn {
        const lfm2 = r.has(try n.at(nb, "self_attn.q_layernorm.weight"));
        const q_norm = try arr(r, owned, try n.at(nb, if (lfm2) "self_attn.q_layernorm.weight" else "self_attn.q_norm.weight"));
        const hd = ops.shape(q_norm)[0];
        var a: Attn = .{
            .q = try lin(r, owned, try n.at(nb, "self_attn.q_proj"), hidden),
            .k = try lin(r, owned, try n.at(nb, "self_attn.k_proj"), hidden),
            .v = try lin(r, owned, try n.at(nb, "self_attn.v_proj"), hidden),
            .o = try lin(r, owned, try n.at(nb, if (lfm2) "self_attn.out_proj" else "self_attn.o_proj"), hidden),
            .q_norm = q_norm,
            .k_norm = try arr(r, owned, try n.at(nb, if (lfm2) "self_attn.k_layernorm.weight" else "self_attn.k_norm.weight")),
            .head_dim = hd,
        };
        if (r.has(try n.at(nb, "self_attn.gate_proj.weight"))) a.gate = try lin(r, owned, try n.at(nb, "self_attn.gate_proj"), hidden);
        return a;
    }

    pub fn deinit(t: *Trunk) void {
        gpa.free(t.layers);
    }

    /// `x` `[1, n, hidden]` at positions `offset..` through every layer and the final norm.
    pub fn forward(t: *const Trunk, g: *Graph, x0: Array, offset: c_int) !Array {
        const n: c_int = ops.shape(x0)[1];
        const mask: ops.Mask = switch (t.mask) {
            .bidirectional => .none,
            .causal => |w| .{ .array = try windowMask(g, n, @intCast(w)) },
        };
        var x = x0;
        for (t.layers) |*L| {
            var lg = g.scope();
            defer lg.deinit();
            const h = try lg.rmsNorm(x, L.operator_norm, t.eps);
            var mixed = switch (L.mixer) {
                .conv => |*c| try convLayer(&lg, c, h, t.hidden),
                .attn => |*a| try attnLayer(&lg, a, h, mask, t.theta, offset, t.eps),
            };
            if (L.post_operator_norm.ctx != null) mixed = try lg.rmsNorm(mixed, L.post_operator_norm, t.eps);
            const mid = try lg.add(x, mixed);
            const f = try lg.rmsNorm(mid, L.ffn_norm, t.eps);
            var ff = try L.w2.forward(&lg, try lg.mul(try lg.silu(try L.w1.forward(&lg, f)), try L.w3.forward(&lg, f)));
            if (L.post_ffn_norm.ctx != null) ff = try lg.rmsNorm(ff, L.post_ffn_norm, t.eps);
            x = try g.keep(try lg.add(mid, ff));
        }
        return g.rmsNorm(x, t.final_norm, t.eps);
    }
};

/// A row reads itself and the `window - 1` before it.
fn windowMask(g: *Graph, n: c_int, window: c_int) !Array {
    const i = try g.reshape(try g.arange(0, @floatFromInt(n), 1, .float32), &.{ n, 1 });
    const j = try g.reshape(try g.arange(0, @floatFromInt(n), 1, .float32), &.{ 1, n });
    const causal = try g.greater(try g.addScalar(i, 1), j);
    const near = try g.greater(j, try g.addScalar(i, @floatFromInt(-window)));
    return g.call(mlx.mlx_logical_and, .{ causal, near });
}

/// `out(C * conv(B * x))` over the whole sequence, the conv causal (left-padded).
fn convLayer(g: *Graph, c: *const ConvLayer, x: Array, h: c_int) !Array {
    const kernel = ops.shape(c.conv)[1];
    const proj = try c.in_proj.forward(g, x);
    const b = try g.sliceAxis(proj, 2, 0, h);
    const gate = try g.sliceAxis(proj, 2, h, 2 * h);
    const v = try g.sliceAxis(proj, 2, 2 * h, 3 * h);
    const bx = try g.padAxis(try g.mul(b, v), 1, kernel - 1, 0);
    return c.out_proj.forward(g, try g.mul(gate, try g.conv1d(bx, c.conv, 1, 0, 1, h)));
}

fn attnLayer(g: *Graph, a: *const Attn, x: Array, mask: ops.Mask, theta: f32, offset: c_int, eps: f32) !Array {
    const n = ops.shape(x)[1];
    const hd = a.head_dim;
    const perm = [_]c_int{ 0, 2, 1, 3 };
    const q = try g.transpose(try g.rmsNorm(try g.reshape(try a.q.forward(g, x), &.{ 1, n, -1, hd }), a.q_norm, eps), &perm);
    const k = try g.transpose(try g.rmsNorm(try g.reshape(try a.k.forward(g, x), &.{ 1, n, -1, hd }), a.k_norm, eps), &perm);
    const v = try g.transpose(try g.reshape(try a.v.forward(g, x), &.{ 1, n, -1, hd }), &perm);
    const o = try g.sdpa(try g.rope(q, hd, theta, offset), try g.rope(k, hd, theta, offset), v, 1.0 / @sqrt(@as(f32, @floatFromInt(hd))), mask);
    var rows = try g.transpose(o, &perm);
    // LFM3's per-head sigmoid gate on the attention output, before o_proj.
    if (a.gate.w.ctx != null) rows = try g.mul(rows, try g.expandDims(try g.sigmoid(try a.gate.forward(g, x)), 3));
    return a.o.forward(g, try g.reshape(rows, &.{ 1, n, -1 }));
}

fn lin(r: *reader.Reader, owned: *reader.Owned, base: []const u8, in_dim: u32) !Linear {
    const l = try r.proj(base, in_dim);
    try owned.linear(l);
    return l;
}

fn arr(r: *reader.Reader, owned: *reader.Owned, full: []const u8) !Array {
    const a = try r.need(full);
    try owned.own(a);
    return a;
}

fn opt(r: *reader.Reader, owned: *reader.Owned, full: []const u8) !Array {
    const a = try r.share(full);
    if (a.ctx != null) try owned.own(a);
    return a;
}
