//! LFM3-VL's tower (`lfm3_vl_vision`) and projector, mlx-vlm's `lfm3_vl/vision.py`: a patchifier and its norm, then
//! layers that are a gated depthwise conv over the 2-D patch grid or q/k-normed attention under axial RoPE, each with a
//! SwiGLU; the projector LayerNorms, folds every merge block into one row and runs a GELU MLP to the text width. Each
//! temporal group of a clip runs alone; their rows concatenate.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const reader = @import("../../sdk/bases/reader.zig");
const qwen_vision = @import("../../sdk/vision/qwen_vision.zig");
const config = @import("config.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Linear = bases.blocks.linear.Linear;
const gpa = std.heap.c_allocator;

const Conv = struct { in_proj: Linear, spatial: Array, out_proj: Linear };
const Attn = struct { q: Linear, k: Linear, v: Linear, o: Linear, q_norm: Array, k_norm: Array };

const Layer = struct {
    operator_norm: Array,
    ffn_norm: Array,
    mixer: union(enum) { conv: Conv, attn: Attn },
    w1: Linear,
    w2: Linear,
    w3: Linear,
};

pub const Tower = struct {
    cfg: config.Vision,
    patchifier: Linear,
    embedding_norm: Array,
    layers: []Layer,
    proj_norm_w: Array,
    proj_norm_b: Array,
    linear_1: Linear,
    linear_2: Linear,
    s: mlx.mlx_stream,

    /// The tower's and the projector's tensors, owned by `owned`; null when the pack ships no tower.
    pub fn load(ctx: *const abi.LoadCtx, cfg: config.Vision, block_json: []const u8, owned: *reader.Owned) !?Tower {
        var r = try reader.Reader.open(ctx, block_json, "vision_tower");
        defer r.close();
        if (!r.has("vision_tower.patchifier.weight")) return null;
        var nb: [160]u8 = undefined;
        const h = cfg.hidden;
        const t: Tower = .{
            .cfg = cfg,
            .patchifier = try lin(&r, owned, "vision_tower.patchifier", 3 * cfg.temporal * cfg.patch * cfg.patch),
            .embedding_norm = try arr(&r, owned, "vision_tower.embedding_norm.weight"),
            .layers = try gpa.alloc(Layer, cfg.layers),
            .proj_norm_w = try arr(&r, owned, "multi_modal_projector.layer_norm.weight"),
            .proj_norm_b = try arr(&r, owned, "multi_modal_projector.layer_norm.bias"),
            .linear_1 = try biased(&r, owned, "multi_modal_projector.linear_1", h * cfg.merge * cfg.merge),
            .linear_2 = try biased(&r, owned, "multi_modal_projector.linear_2", cfg.proj_hidden),
            .s = r.s,
        };
        errdefer gpa.free(t.layers);
        for (t.layers, 0..) |*L, i| {
            const at = struct {
                fn f(b: *[160]u8, li: usize, leaf: []const u8) ![]const u8 {
                    return std.fmt.bufPrint(b, "vision_tower.layers.{d}.{s}", .{ li, leaf });
                }
            }.f;
            const hd = h / cfg.heads;
            L.* = .{
                .operator_norm = try arr(&r, owned, try at(&nb, i, "operator_norm.weight")),
                .ffn_norm = try arr(&r, owned, try at(&nb, i, "ffn_norm.weight")),
                .mixer = if (cfg.attention[i]) .{ .attn = .{
                    .q = try lin(&r, owned, try at(&nb, i, "self_attn.q_proj"), h),
                    .k = try lin(&r, owned, try at(&nb, i, "self_attn.k_proj"), h),
                    .v = try lin(&r, owned, try at(&nb, i, "self_attn.v_proj"), h),
                    .o = try lin(&r, owned, try at(&nb, i, "self_attn.out_proj"), cfg.heads * hd),
                    .q_norm = try arr(&r, owned, try at(&nb, i, "self_attn.q_layernorm.weight")),
                    .k_norm = try arr(&r, owned, try at(&nb, i, "self_attn.k_layernorm.weight")),
                } } else .{ .conv = .{
                    .in_proj = try lin(&r, owned, try at(&nb, i, "conv.in_proj"), h),
                    .spatial = try depthwise2d(&r, owned, try at(&nb, i, "conv.spatial_conv.weight")),
                    .out_proj = try lin(&r, owned, try at(&nb, i, "conv.out_proj"), h),
                } },
                .w1 = try lin(&r, owned, try at(&nb, i, "feed_forward.w1"), h),
                .w2 = try lin(&r, owned, try at(&nb, i, "feed_forward.w2"), cfg.intermediate),
                .w3 = try lin(&r, owned, try at(&nb, i, "feed_forward.w3"), h),
            };
        }
        return t;
    }

    pub fn deinit(t: *Tower) void {
        gpa.free(t.layers);
    }

    /// The item's rows `[groups * gh * gw / merge^2, text_hidden]` (caller owns) from its pixel values
    /// `[groups * gh * gw, 3 * temporal * patch^2]` in merge-block order.
    pub fn forward(t: *const Tower, pixels: Array, groups: u32, gh: u32, gw: u32) !mlx.mlx_array {
        var g = Graph.init(gpa, t.s);
        defer g.deinit();
        const cfg = &t.cfg;
        const n: c_int = @intCast(gh * gw);
        const hd = cfg.hidden / cfg.heads;
        const rope = try qwen_vision.visionRopeTables(gpa, gh, gw, cfg.merge, hd, cfg.theta);
        const cos = try g.adopt(.{ .ctx = rope.cos.ctx });
        const sin = try g.adopt(.{ .ctx = rope.sin.ctx });
        var frames: std.ArrayList(Array) = .empty;
        defer frames.deinit(gpa);
        for (0..groups) |k| {
            var fg = g.scope();
            defer fg.deinit();
            const off: c_int = @intCast(k * @as(usize, @intCast(n)));
            const x0 = try fg.astype(try fg.sliceAxis(pixels, 0, off, off + n), .bfloat16);
            var x = try fg.rmsNorm(try t.patchifier.forward(&fg, x0), t.embedding_norm, cfg.eps);
            for (t.layers) |*L| {
                var lg = fg.scope();
                defer lg.deinit();
                const h = try lg.rmsNorm(x, L.operator_norm, cfg.eps);
                const mixed = switch (L.mixer) {
                    .conv => |*c| try convForward(&lg, c, h, gh, gw, cfg),
                    .attn => |*a| try attnForward(&lg, a, h, cos, sin, cfg),
                };
                const mid = try lg.add(x, mixed);
                const f = try lg.rmsNorm(mid, L.ffn_norm, cfg.eps);
                const up = try lg.mul(try lg.silu(try L.w1.forward(&lg, f)), try L.w3.forward(&lg, f));
                x = try fg.keep(try lg.add(mid, try L.w2.forward(&lg, up)));
            }
            try frames.append(gpa, try g.keep(x));
        }
        const all = if (frames.items.len == 1) frames.items[0] else try g.concat(frames.items, 0);
        return g.escape(try t.project(&g, all));
    }

    /// LayerNorm, one row per merge block, then `linear_2(gelu(linear_1(x)))`.
    fn project(t: *const Tower, g: *Graph, x: Array) !Array {
        const m2: c_int = @intCast(t.cfg.merge * t.cfg.merge);
        const normed = try g.layerNorm(x, t.proj_norm_w, t.proj_norm_b, t.cfg.proj_eps);
        const folded = try g.reshape(normed, &.{ -1, m2 * @as(c_int, @intCast(t.cfg.hidden)) });
        const h = try t.linear_1.forward(g, folded);
        // nn.gelu: the erf form.
        const gelu = try g.mul(try g.mulScalar(h, 0.5), try g.addScalar(try g.erf(try g.mulScalar(h, std.math.sqrt1_2)), 1.0));
        return t.linear_2.forward(g, gelu);
    }
};

/// `out(c * conv(b * v))`: the gate's product laid out as the 2-D grid (merge-block order back to rows), a depthwise
/// conv with same padding, and back.
fn convForward(g: *Graph, c: *const Conv, x: Array, gh: u32, gw: u32, cfg: *const config.Vision) !Array {
    const h: c_int = @intCast(cfg.hidden);
    const m: c_int = @intCast(cfg.merge);
    const H: c_int = @intCast(gh);
    const W: c_int = @intCast(gw);
    const proj = try c.in_proj.forward(g, x);
    const b = try g.sliceAxis(proj, 1, 0, h);
    const gate = try g.sliceAxis(proj, 1, h, 2 * h);
    const v = try g.sliceAxis(proj, 1, 2 * h, 3 * h);
    const blocks = try g.reshape(try g.mul(b, v), &.{ @divExact(H, m), @divExact(W, m), m, m, h });
    const grid = try g.reshape(try g.transpose(blocks, &.{ 0, 2, 1, 3, 4 }), &.{ 1, H, W, h });
    const pad: c_int = @intCast(cfg.conv_kernel / 2);
    const conv = try g.conv2d(grid, c.spatial, 1, pad, 1, h);
    const back = try g.transpose(try g.reshape(conv, &.{ @divExact(H, m), m, @divExact(W, m), m, h }), &.{ 0, 2, 1, 3, 4 });
    return c.out_proj.forward(g, try g.mul(gate, try g.reshape(back, &.{ H * W, h })));
}

/// Bidirectional attention over the frame's patches, q and k normed per head and rotated in f32 by the axial tables.
fn attnForward(g: *Graph, a: *const Attn, x: Array, cos: Array, sin: Array, cfg: *const config.Vision) !Array {
    const hd: c_int = @intCast(cfg.hidden / cfg.heads);
    const n = ops.shape(x)[0];
    const heads: c_int = @intCast(cfg.heads);
    const kv: c_int = @intCast(cfg.kv_heads);
    const q = try g.rmsNorm(try g.reshape(try a.q.forward(g, x), &.{ n, heads, hd }), a.q_norm, cfg.eps);
    const k = try g.rmsNorm(try g.reshape(try a.k.forward(g, x), &.{ n, kv, hd }), a.k_norm, cfg.eps);
    const v = try g.reshape(try a.v.forward(g, x), &.{ n, kv, hd });
    const qr = try rotate(g, q, cos, sin, hd);
    const kr = try rotate(g, k, cos, sin, hd);
    const perm = [_]c_int{ 1, 0, 2 };
    const o = try g.sdpa(
        try g.expandDims(try g.transpose(qr, &perm), 0),
        try g.expandDims(try g.transpose(kr, &perm), 0),
        try g.expandDims(try g.transpose(v, &perm), 0),
        1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
        .none,
    );
    const rows = try g.reshape(try g.transpose(try g.reshape(o, &.{ heads, n, hd }), &perm), &.{ n, heads * hd });
    return a.o.forward(g, rows);
}

/// `t * cos + rotate_half(t) * sin` in f32 (`[n, heads, hd]` against `[n, 1, hd]` tables), back in t's dtype.
fn rotate(g: *Graph, t: Array, cos: Array, sin: Array, hd: c_int) !Array {
    const dt = mlx.mlx_array_dtype(.{ .ctx = t.ctx });
    const f = try g.astype(t, .float32);
    const half = @divExact(hd, 2);
    const lo = try g.sliceAxis(f, 2, 0, half);
    const hi = try g.sliceAxis(f, 2, half, hd);
    const rot = try g.concat(&.{ try g.neg(hi), lo }, 2);
    return g.astype(try g.add(try g.mul(f, cos), try g.mul(rot, sin)), dt);
}

fn lin(r: *reader.Reader, owned: *reader.Owned, base: []const u8, in_dim: u32) !Linear {
    const l = try r.proj(base, in_dim);
    try owned.linear(l);
    return l;
}

fn biased(r: *reader.Reader, owned: *reader.Owned, base: []const u8, in_dim: u32) !Linear {
    var l = try lin(r, owned, base, in_dim);
    var nb: [160]u8 = undefined;
    l.bias = try r.need(try std.fmt.bufPrint(&nb, "{s}.bias", .{base}));
    try owned.own(l.bias);
    return l;
}

fn arr(r: *reader.Reader, owned: *reader.Owned, full: []const u8) !Array {
    const a = try r.need(full);
    try owned.own(a);
    return a;
}

/// A depthwise conv weight as MLX's grouped conv2d reads it, `[C, kh, kw, 1]` (a torch export's `[C, 1, kh, kw]`
/// transposed).
fn depthwise2d(r: *reader.Reader, owned: *reader.Owned, full: []const u8) !Array {
    const w = try r.need(full);
    const sh = ops.shape(w);
    if (sh.len != 4) return error.MissingWeight;
    if (sh[1] != 1 or sh[3] == 1) {
        try owned.own(w);
        return w;
    }
    defer _ = mlx.mlx_array_free(w);
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_transpose_axes(&t, w, &[_]c_int{ 0, 2, 3, 1 }, 4, r.s));
    var c = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, t, false, r.s));
    try owned.own(c);
    return c;
}
