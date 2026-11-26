//! Inkling 2's image tower and merger, mlx-lm's `models/inkling2/vision.py`: a ViT over an image's native-resolution
//! patches (one linear patch embedding, a learned position table read bilinearly at each patch's place, a class
//! token), pre-norm layers of full attention under a 2-D rotary with layer-scaled residuals, then the merger: every
//! `merge` x `merge` block of patches folded into one row and a GELU MLP to the text width. Each tensor is read by its
//! mlx-lm name (`model.vision_tower.*`) or the release's (`model.visual.*`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const reader = @import("../../sdk/bases/reader.zig");
const config = @import("config.zig");
const vk = @import("vision_kernels.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Linear = bases.blocks.linear.Linear;
const gpa = std.heap.c_allocator;

const Layer = struct {
    norm_1: [2]Array,
    norm_2: [2]Array,
    q: Linear,
    k: Linear,
    v: Linear,
    /// Dense Q/K/V weights concatenated `[hidden, 3*hidden]` with bias held separately for `vision_qkv_rope`.
    qkv: Linear = .{},
    qkv_bias: Array = .{},
    o: Linear,
    fc1: Linear,
    fc2: Linear,
    /// The residuals' per-channel scales.
    scale_1: Array,
    scale_2: Array,
};

/// Patch `i`'s row and column in the image's patch grid: the patches go merge block by merge block, the blocks in
/// raster order (mlx-lm's `_grid_positions`).
pub fn gridPosition(i: u32, grid_w: u32, merge: u32) [2]u32 {
    const blocks_w = grid_w / merge;
    return .{
        i / (merge * merge * blocks_w) * merge + i / merge % merge,
        i / (merge * merge) % blocks_w * merge + i % merge,
    };
}

/// The two taps and their weights reading a table of `side` entries at entry `index` of `size` (mlx-lm's
/// `_bilinear_taps`: half-pixel centres, clamped at the edges).
pub const Taps = struct { taps: [2]i32, weights: [2]f32 };

pub fn bilinearTaps(index: u32, size: u32, side: u32) Taps {
    const src = (@as(f32, @floatFromInt(index)) + 0.5) * @as(f32, @floatFromInt(side)) / @as(f32, @floatFromInt(size)) - 0.5;
    const lo = @floor(src);
    var out: Taps = undefined;
    for (0..2) |k| {
        const tap = lo + @as(f32, @floatFromInt(k));
        out.weights[k] = @max(1.0 - @abs(src - tap), 0.0);
        out.taps[k] = @intFromFloat(std.math.clamp(tap, 0, @as(f32, @floatFromInt(side - 1))));
    }
    return out;
}

/// What a forward of a `gh` x `gw` patch grid reads beside the weights: each patch's four position-table entries and
/// their weights, and the rotary's cos and sin for the class token (at the origin) and each patch (one past its row
/// and column): `hd` values a token, the column's half then the row's, each frequency twice.
pub const Tables = struct {
    taps: []i32,
    weights: []f32,
    cos: []f32,
    sin: []f32,

    pub fn init(a: std.mem.Allocator, v: *const config.Vision, gh: u32, gw: u32) !Tables {
        const n: usize = @as(usize, gh) * gw;
        const hd: usize = v.hidden / v.heads;
        const t: Tables = .{ .taps = try a.alloc(i32, n * 4), .weights = try a.alloc(f32, n * 4), .cos = try a.alloc(f32, (n + 1) * hd), .sin = try a.alloc(f32, (n + 1) * hd) };
        const quarter = hd / 4;
        const side: i32 = @intCast(v.grid);
        for (0..n + 1) |tok| {
            // The class token sits at (0, 0); patch i at one past its place.
            var at = [2]u32{ 0, 0 };
            if (tok > 0) {
                const i: u32 = @intCast(tok - 1);
                const rc = gridPosition(i, gw, v.merge);
                at = .{ rc[0] + 1, rc[1] + 1 };
                const rows = bilinearTaps(rc[0], gh, v.grid);
                const cols = bilinearTaps(rc[1], gw, v.grid);
                for (0..2) |r| for (0..2) |c| {
                    t.taps[i * 4 + r * 2 + c] = rows.taps[r] * side + cols.taps[c];
                    t.weights[i * 4 + r * 2 + c] = rows.weights[r] * cols.weights[c];
                };
            }
            for (0..quarter) |j| {
                const inv = 1.0 / std.math.pow(f32, v.theta, @as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(hd / 2)));
                for ([_]f32{ @as(f32, @floatFromInt(at[1])) * inv, @as(f32, @floatFromInt(at[0])) * inv }, 0..) |angle, half| for (0..2) |twice| {
                    t.cos[tok * hd + half * 2 * quarter + 2 * j + twice] = @cos(angle);
                    t.sin[tok * hd + half * 2 * quarter + 2 * j + twice] = @sin(angle);
                };
            }
        }
        return t;
    }

    pub fn deinit(t: *const Tables, a: std.mem.Allocator) void {
        a.free(t.taps);
        a.free(t.weights);
        a.free(t.cos);
        a.free(t.sin);
    }
};

pub const Tower = struct {
    cfg: config.Vision,
    s: mlx.mlx_stream,
    owned: reader.Owned = .{},
    patch: Linear,
    /// The position table `[grid * grid, hidden]` f32.
    positions: Array,
    /// The class token (its embedding plus its position's) `[1, hidden]`.
    class: Array,
    pre_norm: [2]Array,
    layers: []Layer,
    merge_1: Linear,
    merge_2: Linear,

    /// The tower's tensors; null when the pack ships none.
    pub fn load(ctx: *const abi.LoadCtx, cfg: config.Vision, block_json: []const u8) !?Tower {
        var r = try reader.Reader.open(ctx, block_json, "");
        defer r.close();
        const release = !r.has("model.vision_tower.embeddings.patch_embedding.weight");
        if (release and !r.has("model.visual.backbone.embeddings.patch_embedding.weight")) return null;
        var t: Tower = .{ .cfg = cfg, .s = r.s, .patch = .{}, .positions = .{}, .class = .{}, .pre_norm = undefined, .layers = &.{}, .merge_1 = .{}, .merge_2 = .{} };
        errdefer t.deinit();
        var at: At = .{ .r = &r, .o = &t.owned, .release = release };
        const h = cfg.hidden;
        t.patch = try at.linear("embeddings.patch_embedding", "backbone.embeddings.patch_embedding", cfg.channels * cfg.patch * cfg.patch, false);
        t.pre_norm = .{ try at.need("pre_layernorm.weight", "backbone.ln_pre.weight"), try at.need("pre_layernorm.bias", "backbone.ln_pre.bias") };
        {
            var g = Graph.init(gpa, r.s);
            defer g.deinit();
            const table = try g.adopt(try at.raw("embeddings.position_embedding.weight", "backbone.embeddings.position_embedding"));
            t.positions = try at.keep(&g, try g.astype(table, .float32));
            const class = try g.add(try g.adopt(try at.raw("embeddings.class_embedding", "backbone.embeddings.class_embedding")), try g.adopt(try at.raw("embeddings.cls_position_embedding", "backbone.embeddings.cls_position_embedding")));
            t.class = try at.keep(&g, try g.reshape(class, &.{ 1, @intCast(h) }));
        }
        t.layers = try gpa.alloc(Layer, cfg.layers);
        for (t.layers, 0..) |*L, i| {
            at.layer = @intCast(i);
            var q = try at.rawLinear("self_attn.q_proj", "self_attn.q_proj", h, true);
            var k = try at.rawLinear("self_attn.k_proj", "self_attn.k_proj", h, true);
            var v = try at.rawLinear("self_attn.v_proj", "self_attn.v_proj", h, true);
            var qkv: Linear = .{};
            var qkv_bias: Array = .{};
            if (canJoinDenseQkv(&q, &k, &v)) {
                var g = Graph.init(gpa, r.s);
                defer g.deinit();
                const qw = try at.keep(&g, try g.concat(&.{ q.w, k.w, v.w }, 1));
                const qb = try at.keep(&g, try g.concat(&.{ q.bias, k.bias, v.bias }, 0));
                freeLinear(&q);
                freeLinear(&k);
                freeLinear(&v);
                const hc: c_int = @intCast(h);
                q = .{ .w = try at.keep(&g, try g.sliceAxis(qw, 1, 0, hc)), .bias = try at.keep(&g, try g.sliceAxis(qb, 0, 0, hc)) };
                k = .{ .w = try at.keep(&g, try g.sliceAxis(qw, 1, hc, 2 * hc)), .bias = try at.keep(&g, try g.sliceAxis(qb, 0, hc, 2 * hc)) };
                v = .{ .w = try at.keep(&g, try g.sliceAxis(qw, 1, 2 * hc, 3 * hc)), .bias = try at.keep(&g, try g.sliceAxis(qb, 0, 2 * hc, 3 * hc)) };
                qkv = .{ .w = qw };
                qkv_bias = qb;
            } else {
                try t.owned.linear(q);
                try t.owned.linear(k);
                try t.owned.linear(v);
            }
            L.* = .{
                .norm_1 = .{ try at.need("layernorm_before.weight", "ln_1.weight"), try at.need("layernorm_before.bias", "ln_1.bias") },
                .norm_2 = .{ try at.need("layernorm_after.weight", "ln_2.weight"), try at.need("layernorm_after.bias", "ln_2.bias") },
                .q = q,
                .k = k,
                .v = v,
                .qkv = qkv,
                .qkv_bias = qkv_bias,
                .o = try at.linear("self_attn.out_proj", "self_attn.out_proj", h, true),
                .fc1 = try at.linear("mlp.fc1", "mlp.c_fc", h, true),
                .fc2 = try at.linear("mlp.fc2", "mlp.c_proj", cfg.intermediate(), true),
                .scale_1 = try at.need("lambda_1", "ls_1.gamma"),
                .scale_2 = try at.need("lambda_2", "ls_2.gamma"),
            };
        }
        at.layer = null;
        t.merge_1 = try at.linear("merger.linear_1", "proj_in", h * cfg.merge * cfg.merge, cfg.proj_bias);
        t.merge_2 = try at.linear("merger.linear_2", "proj_out", cfg.proj_hidden, cfg.proj_bias);
        return t;
    }

    pub fn deinit(t: *Tower) void {
        gpa.free(t.layers);
        t.owned.deinit();
    }

    /// An image's rows `[gh * gw / merge^2, text_hidden]` (caller owns) from its patches `[gh * gw, 3 * patch^2]` f32
    /// in merge-block order.
    pub fn forward(t: *const Tower, pixels: Array, gh: u32, gw: u32) !mlx.mlx_array {
        var g = Graph.init(gpa, t.s);
        defer g.deinit();
        const cfg = &t.cfg;
        const n: c_int = @intCast(gh * gw);
        const hidden: c_int = @intCast(cfg.hidden);
        const heads: c_int = @intCast(cfg.heads);
        const hd = @divExact(hidden, heads);
        const dt = ops.dtype(t.pre_norm[0]);
        const tabs = try Tables.init(gpa, cfg, gh, gw);
        defer tabs.deinit(gpa);

        const at = try g.take(t.positions, try g.fromSlice(i32, tabs.taps, &.{ n, 4 }), 0);
        const placed = try g.sum(try g.mul(at, try g.fromSlice(f32, tabs.weights, &.{ n, 4, 1 })), 1, false);
        const patches = try g.add(try t.patch.forward(&g, try g.astype(pixels, dt)), try g.astype(placed, dt));
        var x = try g.layerNorm(try g.concat(&.{ try g.astype(t.class, dt), patches }, 0), t.pre_norm[0], t.pre_norm[1], cfg.eps);

        if (vk.runs(t.s, dt) and t.layers.len > 0) {
            return g.escape(try t.forwardFused(&g, x, &tabs, n, hidden, heads, hd, dt));
        }

        const cos = try g.fromSlice(f32, tabs.cos, &.{ n + 1, 1, hd });
        const sin = try g.fromSlice(f32, tabs.sin, &.{ n + 1, 1, hd });
        const perm = [_]c_int{ 1, 0, 2 };
        for (t.layers) |*L| {
            var lg = g.scope();
            defer lg.deinit();
            const h = try lg.layerNorm(x, L.norm_1[0], L.norm_1[1], cfg.eps);
            const q = try rotate(&lg, try lg.reshape(try L.q.forward(&lg, h), &.{ n + 1, heads, hd }), cos, sin);
            const k = try rotate(&lg, try lg.reshape(try L.k.forward(&lg, h), &.{ n + 1, heads, hd }), cos, sin);
            const v = try lg.reshape(try L.v.forward(&lg, h), &.{ n + 1, heads, hd });
            const read = try lg.sdpa(
                try lg.expandDims(try lg.transpose(q, &perm), 0),
                try lg.expandDims(try lg.transpose(k, &perm), 0),
                try lg.expandDims(try lg.transpose(v, &perm), 0),
                1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
                .none,
            );
            const rows = try lg.reshape(try lg.transpose(try lg.reshape(read, &.{ heads, n + 1, hd }), &perm), &.{ n + 1, hidden });
            const mid = try lg.add(x, try lg.mul(L.scale_1, try L.o.forward(&lg, rows)));
            const mlp = try L.fc2.forward(&lg, try gelu(&lg, try L.fc1.forward(&lg, try lg.layerNorm(mid, L.norm_2[0], L.norm_2[1], cfg.eps))));
            x = try g.keep(try lg.add(mid, try lg.mul(L.scale_2, mlp)));
        }
        const m2: c_int = @intCast(cfg.merge * cfg.merge);
        const blocks = try g.reshape(try g.sliceAxis(x, 0, 1, n + 1), &.{ @divExact(n, m2), m2 * hidden });
        return g.escape(try t.merge_2.forward(&g, try gelu(&g, try t.merge_1.forward(&g, blocks))));
    }

    /// Fused Metal path for the ViT blocks and merger: joined QKV + fused 2-D RoPE, NAX bidirectional attention
    /// (falling back to SDPA off NAX or outside head width 64), fused bias + layer-scaled residual + next LayerNorm,
    /// and fused bias + erf-GELU.
    fn forwardFused(t: *const Tower, g: *Graph, x_in: Array, tabs: *const Tables, n: c_int, hidden: c_int, heads: c_int, hd: c_int, dt: ops.Dtype) !Array {
        var fg = g.scope();
        defer fg.deinit();
        const cfg = &t.cfg;
        const seq = n + 1;
        const cos2d = try fg.fromSlice(f32, tabs.cos, &.{ seq, hd });
        const sin2d = try fg.fromSlice(f32, tabs.sin, &.{ seq, hd });
        const scale_f32: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const scale_arr = try fg.scalar(scale_f32, .float32);
        const eps_arr = try fg.scalar(cfg.eps, .float32);
        const use_nax = vk.attnNaxOk(t.s, dt, hd);
        const perm = [_]c_int{ 1, 0, 2 };

        var x = x_in;
        var h = try fg.layerNorm(x, t.layers[0].norm_1[0], t.layers[0].norm_1[1], cfg.eps);
        for (t.layers, 0..) |*L, idx| {
            var lg = fg.scope();
            defer lg.deinit();
            var q: Array = undefined;
            var k: Array = undefined;
            var v: Array = undefined;
            if (L.qkv.w.ctx != null) {
                const qkv_raw = try L.qkv.forward(&lg, h);
                const split = try vk.qkvRope(&lg, qkv_raw, L.qkv_bias, cos2d, sin2d, heads, hd);
                q = split[0];
                k = split[1];
                v = split[2];
            } else {
                const cos3 = try lg.reshape(cos2d, &.{ seq, 1, hd });
                const sin3 = try lg.reshape(sin2d, &.{ seq, 1, hd });
                const qr = try rotate(&lg, try lg.reshape(try L.q.forward(&lg, h), &.{ seq, heads, hd }), cos3, sin3);
                const kr = try rotate(&lg, try lg.reshape(try L.k.forward(&lg, h), &.{ seq, heads, hd }), cos3, sin3);
                const vr = try lg.reshape(try L.v.forward(&lg, h), &.{ seq, heads, hd });
                q = try lg.expandDims(try lg.transpose(qr, &perm), 0);
                k = try lg.expandDims(try lg.transpose(kr, &perm), 0);
                v = try lg.expandDims(try lg.transpose(vr, &perm), 0);
            }
            const rows = if (use_nax)
                try vk.attnNax(&lg, q, k, v, scale_arr)
            else blk: {
                const read = try lg.sdpa(q, k, v, scale_f32, .none);
                break :blk try lg.reshape(try lg.transpose(try lg.reshape(read, &.{ heads, seq, hd }), &perm), &.{ seq, hidden });
            };
            const o_raw = try noBias(L.o).forward(&lg, rows);
            const rn1 = try vk.resNorm(&lg, x, o_raw, L.o.bias, L.scale_1, L.norm_2[0], L.norm_2[1], eps_arr, true);
            const fc1_raw = try noBias(L.fc1).forward(&lg, rn1[1]);
            const act = try vk.biasGelu(&lg, fc1_raw, L.fc1.bias);
            const fc2_raw = try noBias(L.fc2).forward(&lg, act);
            const has_next = idx + 1 < t.layers.len;
            const nw = if (has_next) t.layers[idx + 1].norm_1[0] else L.norm_2[0];
            const nb = if (has_next) t.layers[idx + 1].norm_1[1] else L.norm_2[1];
            const rn2 = try vk.resNorm(&lg, rn1[0], fc2_raw, L.fc2.bias, L.scale_2, nw, nb, eps_arr, has_next);
            const old_x = x;
            const old_h = h;
            x = try fg.keep(rn2[0]);
            if (has_next) h = try fg.keep(rn2[1]);
            if (old_x.ctx != x_in.ctx) fg.drop(old_x);
            fg.drop(old_h);
        }
        const m2: c_int = @intCast(cfg.merge * cfg.merge);
        const blocks = try fg.reshape(try fg.sliceAxis(x, 0, 1, seq), &.{ @divExact(n, m2), m2 * hidden });
        const m1_raw = try noBias(t.merge_1).forward(&fg, blocks);
        const m1_act = try vk.biasGelu(&fg, m1_raw, t.merge_1.bias);
        return g.keep(try t.merge_2.forward(&fg, m1_act));
    }
};

fn noBias(l: Linear) Linear {
    var out = l;
    out.bias = .{};
    return out;
}

fn canJoinDenseQkv(q: *const Linear, k: *const Linear, v: *const Linear) bool {
    inline for (.{ q, k, v }) |p| {
        if (p.s.ctx != null or p.rht != null or p.ext != null or p.bias.ctx == null) return false;
    }
    const wdt = ops.dtype(q.w);
    const bdt = ops.dtype(q.bias);
    return ops.dtype(k.w) == wdt and ops.dtype(v.w) == wdt and ops.dtype(k.bias) == bdt and ops.dtype(v.bias) == bdt;
}

fn freeLinear(l: *const Linear) void {
    for ([_]Array{ l.w, l.s, l.b, l.bias }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
}

/// The rotary over `x` `[n, heads, hd]` in f32, back in its dtype: each pair (a, b) turns to (a cos - b sin, b cos +
/// a sin).
fn rotate(g: *Graph, x: Array, cos: Array, sin: Array) !Array {
    const sh = ops.shape(x);
    const f = try g.astype(x, .float32);
    const pairs = try g.reshape(f, &.{ sh[0], sh[1], @divExact(sh[2], 2), 2 });
    const turned = try g.reshape(try g.concat(&.{ try g.neg(try g.sliceAxis(pairs, 3, 1, 2)), try g.sliceAxis(pairs, 3, 0, 1) }, 3), sh);
    return g.astype(try g.add(try g.mul(f, cos), try g.mul(turned, sin)), ops.dtype(x));
}

/// `nn.gelu` (the erf form) in f32, back in `x`'s dtype.
fn gelu(g: *Graph, x: Array) !Array {
    const f = try g.astype(x, .float32);
    return g.astype(try g.mul(try g.mulScalar(f, 0.5), try g.addScalar(try g.erf(try g.mulScalar(f, std.math.sqrt1_2)), 1.0)), ops.dtype(x));
}

/// Reads a tensor by its mlx-lm leaf under `model.vision_tower` or the release's under `model.visual`; a layer's
/// leaves sit under `layers.N` (the release's under `backbone.layers.N`).
const At = struct {
    r: *reader.Reader,
    o: *reader.Owned,
    release: bool,
    layer: ?u32 = null,

    fn name(a: At, buf: []u8, leaf: []const u8, rel: []const u8) ![]const u8 {
        const tail = if (a.release) rel else leaf;
        if (a.layer) |i| return std.fmt.bufPrint(buf, "{s}layers.{d}.{s}", .{ if (a.release) "model.visual.backbone." else "model.vision_tower.", i, tail });
        return std.fmt.bufPrint(buf, "{s}{s}", .{ if (a.release) "model.visual." else "model.vision_tower.", tail });
    }

    /// An unowned handle.
    fn raw(a: At, leaf: []const u8, rel: []const u8) !Array {
        var buf: [192]u8 = undefined;
        return a.r.need(try a.name(&buf, leaf, rel));
    }

    fn need(a: At, leaf: []const u8, rel: []const u8) !Array {
        const x = try a.raw(leaf, rel);
        try a.o.own(x);
        return x;
    }

    fn rawLinear(a: At, leaf: []const u8, rel: []const u8, in_dim: u32, biased: bool) !Linear {
        var buf: [192]u8 = undefined;
        var l = try a.r.proj(try a.name(&buf, leaf, rel), in_dim);
        if (biased) {
            var bb: [192]u8 = undefined;
            const base = try a.name(&buf, leaf, rel);
            l.bias = try a.r.need(try std.fmt.bufPrint(&bb, "{s}.bias", .{base}));
        }
        return l;
    }

    fn linear(a: At, leaf: []const u8, rel: []const u8, in_dim: u32, biased: bool) !Linear {
        const l = try a.rawLinear(leaf, rel, in_dim, biased);
        try a.o.linear(l);
        return l;
    }

    /// `x` evaluated, owned by the tower.
    fn keep(a: At, g: *Graph, x: Array) !Array {
        try ops.eval(x);
        const out = g.escape(x);
        try a.o.own(out);
        return out;
    }
};

// ── tests ──

const testing = std.testing;

test "inkling2 vision: patches go merge block by merge block, the blocks in raster order" {
    // A 4 x 6 patch grid in 2 x 2 blocks: block (0, 0), (0, 1), (0, 2), then the second row of blocks.
    const want = [_][2]u32{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 }, .{ 0, 2 }, .{ 0, 3 }, .{ 1, 2 }, .{ 1, 3 }, .{ 0, 4 }, .{ 0, 5 }, .{ 1, 4 }, .{ 1, 5 }, .{ 2, 0 }, .{ 2, 1 }, .{ 3, 0 } };
    for (want, 0..) |w, i| try testing.expectEqual(w, gridPosition(@intCast(i), 6, 2));
}

test "inkling2 vision: the position table is read at half-pixel centres, clamped at its edges" {
    // As many patches as entries: each reads its own.
    for (0..8) |i| {
        const t = bilinearTaps(@intCast(i), 8, 8);
        try testing.expectEqual(@as(i32, @intCast(i)), t.taps[0]);
        try testing.expectApproxEqAbs(@as(f32, 1), t.weights[0], 1e-6);
        try testing.expectApproxEqAbs(@as(f32, 0), t.weights[1], 1e-6);
    }
    // Two patches over four entries sit between entries 0|1 and 2|3.
    const a = bilinearTaps(0, 2, 4);
    try testing.expectEqual([2]i32{ 0, 1 }, a.taps);
    try testing.expectApproxEqAbs(@as(f32, 0.5), a.weights[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), a.weights[1], 1e-6);
    // Eight patches over four entries: the first sits a quarter before entry 0 and clamps onto it.
    const b = bilinearTaps(0, 8, 4);
    try testing.expectEqual([2]i32{ 0, 0 }, b.taps);
    try testing.expectApproxEqAbs(@as(f32, 0.25), b.weights[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.75), b.weights[1], 1e-6);
}

test "inkling2 vision: the rotary turns the column's half then the row's, the class token not at all" {
    const v: config.Vision = .{ .hidden = 16, .heads = 2, .merge = 2, .grid = 4, .theta = 100 };
    const t = try Tables.init(testing.allocator, &v, 2, 4);
    defer t.deinit(testing.allocator);
    const hd = 8;
    for (0..hd) |d| {
        try testing.expectEqual(@as(f32, 1), t.cos[d]);
        try testing.expectEqual(@as(f32, 0), t.sin[d]);
    }
    // Patch 5 is row 0, column 3 (the second block's second patch): the rotary reads row 1, column 4.
    const tok = 6;
    const inv1: f32 = 1.0 / std.math.pow(f32, 100, 2.0 / 4.0);
    for ([_]struct { usize, f32 }{ .{ 0, 4 }, .{ 1, 4 }, .{ 2, 4 * inv1 }, .{ 3, 4 * inv1 }, .{ 4, 1 }, .{ 5, 1 }, .{ 6, inv1 }, .{ 7, inv1 } }) |c| {
        try testing.expectApproxEqAbs(@cos(c[1]), t.cos[tok * hd + c[0]], 1e-6);
        try testing.expectApproxEqAbs(@sin(c[1]), t.sin[tok * hd + c[0]], 1e-6);
    }
}

fn buildSyntheticTower(a: std.mem.Allocator, s: mlx.mlx_stream, cfg: config.Vision, dt: ops.Dtype, seed: u64) !Tower {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const Fill = struct {
        a: std.mem.Allocator,
        s: mlx.mlx_stream,
        o: *reader.Owned,
        rng: std.Random,
        fn arr(f: @This(), shape: []const c_int, target_dt: ops.Dtype, scale: f32, offset: f32) !Array {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const buf = try f.a.alloc(f32, n);
            defer f.a.free(buf);
            for (buf) |*x| x.* = (f.rng.float(f32) * 2.0 - 1.0) * scale + offset;
            var g = Graph.init(f.a, f.s);
            defer g.deinit();
            const raw = try g.fromSlice(f32, buf, shape);
            const out = if (target_dt == .float32) raw else try g.astype(raw, target_dt);
            try ops.eval(out);
            const h = g.escape(out);
            try f.o.own(h);
            return h;
        }
        fn lin(f: @This(), in_d: u32, out_d: u32, target_dt: ops.Dtype, biased: bool) !Linear {
            const w = try f.arr(&.{ @intCast(in_d), @intCast(out_d) }, target_dt, 0.15, 0.0);
            const b = if (biased) try f.arr(&.{@intCast(out_d)}, target_dt, 0.05, 0.0) else Array{};
            return .{ .w = w, .bias = b };
        }
    };
    var t: Tower = .{ .cfg = cfg, .s = s, .patch = .{}, .positions = .{}, .class = .{}, .pre_norm = undefined, .layers = &.{}, .merge_1 = .{}, .merge_2 = .{} };
    errdefer t.deinit();
    const f: Fill = .{ .a = a, .s = s, .o = &t.owned, .rng = rng };
    const h = cfg.hidden;
    const inter = cfg.intermediate();
    t.patch = try f.lin(cfg.channels * cfg.patch * cfg.patch, h, dt, false);
    t.positions = try f.arr(&.{ @intCast(cfg.grid * cfg.grid), @intCast(h) }, .float32, 0.2, 0.0);
    t.class = try f.arr(&.{ 1, @intCast(h) }, dt, 0.2, 0.0);
    t.pre_norm = .{ try f.arr(&.{@intCast(h)}, dt, 0.1, 1.0), try f.arr(&.{@intCast(h)}, dt, 0.05, 0.0) };
    t.layers = try gpa.alloc(Layer, cfg.layers);
    for (t.layers) |*L| {
        const q = try f.lin(h, h, dt, true);
        const k = try f.lin(h, h, dt, true);
        const v = try f.lin(h, h, dt, true);
        var g = Graph.init(a, s);
        defer g.deinit();
        const qw = try g.concat(&.{ q.w, k.w, v.w }, 1);
        const qb = try g.concat(&.{ q.bias, k.bias, v.bias }, 0);
        try ops.evalMany(&.{ qw, qb });
        const qw_h = g.escape(qw);
        const qb_h = g.escape(qb);
        try t.owned.own(qw_h);
        try t.owned.own(qb_h);
        L.* = .{
            .norm_1 = .{ try f.arr(&.{@intCast(h)}, dt, 0.1, 1.0), try f.arr(&.{@intCast(h)}, dt, 0.05, 0.0) },
            .norm_2 = .{ try f.arr(&.{@intCast(h)}, dt, 0.1, 1.0), try f.arr(&.{@intCast(h)}, dt, 0.05, 0.0) },
            .q = q,
            .k = k,
            .v = v,
            .qkv = .{ .w = qw_h },
            .qkv_bias = qb_h,
            .o = try f.lin(h, h, dt, true),
            .fc1 = try f.lin(h, inter, dt, true),
            .fc2 = try f.lin(inter, h, dt, true),
            .scale_1 = try f.arr(&.{@intCast(h)}, dt, 0.2, 0.8),
            .scale_2 = try f.arr(&.{@intCast(h)}, dt, 0.2, 0.8),
        };
    }
    t.merge_1 = try f.lin(h * cfg.merge * cfg.merge, cfg.proj_hidden, dt, cfg.proj_bias);
    t.merge_2 = try f.lin(cfg.proj_hidden, 96, dt, cfg.proj_bias);
    return t;
}

// Bar: fused Metal vision forward agrees with the reference op chain on bf16, f16 (including NAX attention at hd=64)
// and f32 across single-tile and multi-tile patch grids.
test "inkling2 vision: fused Metal path agrees with the reference chain across dtypes" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.gpuStream();
    const env = @import("../../env.zig");
    const parity = @import("../../sdk/testing/parity.zig");
    const cfg: config.Vision = .{ .hidden = 128, .layers = 2, .heads = 2, .patch = 4, .merge = 2, .grid = 4, .proj_hidden = 128, .proj_bias = false, .mlp_ratio = 2.0 };
    const feat: usize = cfg.channels * cfg.patch * cfg.patch;
    inline for (.{ ops.Dtype.bfloat16, ops.Dtype.float16, ops.Dtype.float32 }) |dt| {
        var t = try buildSyntheticTower(a, s, cfg, dt, 101 + @intFromEnum(dt));
        defer t.deinit();
        for ([_][2]u32{ .{ 4, 6 }, .{ 12, 12 } }) |grid| {
            const n: usize = @as(usize, grid[0]) * grid[1];
            const px_buf = try a.alloc(f32, n * feat);
            defer a.free(px_buf);
            var prng = std.Random.DefaultPrng.init(77 + n);
            for (px_buf) |*x| x.* = prng.random().float(f32) * 2.0 - 1.0;
            const pixels = mlx.mlx_array_new_data(px_buf.ptr, &[_]c_int{ @intCast(n), @intCast(feat) }, 2, .float32);
            defer _ = mlx.mlx_array_free(pixels);

            const ref_raw = blk: {
                const off = env.override(.MLX_SERVE_INKLING2_FUSED, "0");
                defer off.restore();
                break :blk try t.forward(pixels, grid[0], grid[1]);
            };
            defer _ = mlx.mlx_array_free(ref_raw);
            const got_raw = blk: {
                const on = env.override(.MLX_SERVE_INKLING2_FUSED, "1");
                defer on.restore();
                break :blk try t.forward(pixels, grid[0], grid[1]);
            };
            defer _ = mlx.mlx_array_free(got_raw);

            var g = Graph.init(a, s);
            defer g.deinit();
            const ref32 = try g.contiguous(try g.astype(ref_raw, .float32));
            const got32 = try g.contiguous(try g.astype(got_raw, .float32));
            try ops.evalMany(&.{ ref32, got32 });
            const total: usize = @intCast(mlx.mlx_array_size(ref32));
            const cols: usize = @intCast(ops.shape(ref32)[1]);
            const rows_n = total / cols;
            const ref_sl = mlx.mlx_array_data_float32(ref32).?[0..total];
            const got_sl = mlx.mlx_array_data_float32(got32).?[0..total];
            const cmp = parity.qwen4CompareRows(got_sl, ref_sl, rows_n, cols);
            var sum_r: f64 = 0;
            var sum_g: f64 = 0;
            for (ref_sl, got_sl) |rv, gv| {
                sum_r += @as(f64, rv) * rv;
                sum_g += @as(f64, gv) * gv;
            }
            const rms_ratio = @sqrt(sum_g / sum_r);
            const min_cos: f64 = if (dt == .bfloat16) 0.9995 else 0.99999;
            try testing.expect(cmp.min_cos >= min_cos);
            try testing.expect(@abs(rms_ratio - 1.0) < 0.005);
        }
    }
}

// Bar: real Inkling-2-Mini vision tower fused vs unfused parity and timing when INKLING2_PACK is set.
test "inkling2 vision: real tower fused vs unfused parity and timing (INKLING2_PACK)" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const env = @import("../../env.zig");
    const dir = std.mem.span(env.named("INKLING2_PACK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = testing.io;
    const resident = @import("../../sdk/weights/resident.zig");
    const parity = @import("../../sdk/testing/parity.zig");
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ar = arena.allocator();

    var pack_dir = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer pack_dir.close(io);
    const Index = struct { weight_map: std.json.ArrayHashMap([]const u8) };
    const idx_bytes = try pack_dir.readFileAlloc(io, "model.safetensors.index.json", ar, .limited(1 << 24));
    const idx = try std.json.parseFromSliceLeaky(Index, ar, idx_bytes, .{ .ignore_unknown_fields = true });
    const shard = idx.weight_map.map.get("model.vision_tower.pre_layernorm.weight") orelse return error.SkipZigTest;

    const cfg_bytes = try pack_dir.readFileAlloc(io, "config.json", ar, .limited(1 << 20));
    var parsed: abi.Parsed = .{};
    const mem: @import("../../sdk/testing/fixture.zig").MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = cfg_bytes }} };
    const pdir = mem.packDir();
    try testing.expect(config.parse(&pdir, abi.Str.of(cfg_bytes), .{}, null, true, &parsed).isOk());
    defer config.freeParsed(&parsed);
    const vcfg = config.of(&parsed).vision orelse return error.SkipZigTest;

    const w = try resident.loadWeightsFile(a, dir, shard);
    var res = resident.Resident.adopt(w);
    defer res.deinit();
    var src = res.source();
    const s = mlx.gpuStream();
    var load_ctx: abi.LoadCtx = .{ .model_dir = abi.Str.of(dir), .dir = &pdir, .parsed = &parsed, .stream = .{ .ctx = s.ctx }, .weights = &src };
    var t = (try Tower.load(&load_ctx, vcfg, config.of(&parsed).block_json)).?;
    defer t.deinit();

    const feat: usize = vcfg.channels * vcfg.patch * vcfg.patch;
    std.debug.print("\n[inkling2 vision] real tower dtype={s} layers={d} hidden={d} heads={d} nax={any}\n", .{
        @tagName(ops.dtype(t.pre_norm[0])), vcfg.layers, vcfg.hidden, vcfg.heads, vk.attnNaxOk(s, ops.dtype(t.pre_norm[0]), @intCast(vcfg.hidden / vcfg.heads)),
    });
    for ([_][2]u32{ .{ 28, 28 }, .{ 48, 48 }, .{ 76, 76 } }) |grid| {
        const n: usize = @as(usize, grid[0]) * grid[1];
        const px_buf = try a.alloc(f32, n * feat);
        defer a.free(px_buf);
        var prng = std.Random.DefaultPrng.init(42 + n);
        for (px_buf) |*x| x.* = prng.random().float(f32) * 2.0 - 1.0;
        const pixels = mlx.mlx_array_new_data(px_buf.ptr, &[_]c_int{ @intCast(n), @intCast(feat) }, 2, .float32);
        defer _ = mlx.mlx_array_free(pixels);

        // Warmup both arms (compiles Metal kernels once).
        for ([_][*:0]const u8{ "0", "1" }) |flag| {
            const ov = env.override(.MLX_SERVE_INKLING2_FUSED, flag);
            defer ov.restore();
            const out = try t.forward(pixels, grid[0], grid[1]);
            try ops.eval(out);
            _ = mlx.mlx_array_free(out);
        }

        // Paired A/B/B/A (3 repeats per arm) + parity check.
        const reps = 3;
        var ms_off: [reps]f64 = undefined;
        var ms_on: [reps]f64 = undefined;
        var ref_arr: Array = .{};
        var got_arr: Array = .{};
        defer if (ref_arr.ctx != null) {
            _ = mlx.mlx_array_free(ref_arr);
        };
        defer if (got_arr.ctx != null) {
            _ = mlx.mlx_array_free(got_arr);
        };
        for (0..reps) |r| {
            inline for (.{ .{ false, "0" }, .{ true, "1" } }) |arm| {
                const ov = env.override(.MLX_SERVE_INKLING2_FUSED, arm[1]);
                defer ov.restore();
                const t0 = std.Io.Clock.awake.now(io);
                const out = try t.forward(pixels, grid[0], grid[1]);
                try ops.eval(out);
                const elapsed_ns = t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds;
                const ms = @as(f64, @floatFromInt(elapsed_ns)) / 1e6;
                if (arm[0]) {
                    ms_on[r] = ms;
                    if (r == 0) got_arr = out else _ = mlx.mlx_array_free(out);
                } else {
                    ms_off[r] = ms;
                    if (r == 0) ref_arr = out else _ = mlx.mlx_array_free(out);
                }
            }
        }
        std.mem.sort(f64, &ms_off, {}, std.sort.asc(f64));
        std.mem.sort(f64, &ms_on, {}, std.sort.asc(f64));

        var g = Graph.init(a, s);
        defer g.deinit();
        const ref32 = try g.contiguous(try g.astype(ref_arr, .float32));
        const got32 = try g.contiguous(try g.astype(got_arr, .float32));
        try ops.evalMany(&.{ ref32, got32 });
        const total: usize = @intCast(mlx.mlx_array_size(ref32));
        const cols: usize = @intCast(ops.shape(ref32)[1]);
        const ref_sl = mlx.mlx_array_data_float32(ref32).?[0..total];
        const got_sl = mlx.mlx_array_data_float32(got32).?[0..total];
        const rows_n = total / cols;
        const cmp = parity.qwen4CompareRows(got_sl, ref_sl, rows_n, cols);
        var sum_r: f64 = 0;
        var sum_g: f64 = 0;
        var dot_all: f64 = 0;
        for (ref_sl, got_sl) |rv, gv| {
            const rf: f64 = rv;
            const gf: f64 = gv;
            sum_r += rf * rf;
            sum_g += gf * gf;
            dot_all += rf * gf;
        }
        const rms_ratio = @sqrt(sum_g / sum_r);
        const global_cos = dot_all / @sqrt(sum_r * sum_g);
        std.debug.print("[inkling2 vision] grid {d}x{d} ({d} tokens): unfused={d:.2}ms fused={d:.2}ms speedup={d:.2}x cos={d:.6} min_cos={d:.6} rms_ratio={d:.6}\n", .{
            grid[0], grid[1], n + 1, ms_off[reps / 2], ms_on[reps / 2], ms_off[reps / 2] / ms_on[reps / 2], global_cos, cmp.min_cos, rms_ratio,
        });
        try testing.expect(global_cos >= 0.999);
        try testing.expect(cmp.min_cos >= 0.998);
        try testing.expect(@abs(rms_ratio - 1.0) < 0.005);
    }
}
