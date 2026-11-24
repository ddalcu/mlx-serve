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
            L.* = .{
                .norm_1 = .{ try at.need("layernorm_before.weight", "ln_1.weight"), try at.need("layernorm_before.bias", "ln_1.bias") },
                .norm_2 = .{ try at.need("layernorm_after.weight", "ln_2.weight"), try at.need("layernorm_after.bias", "ln_2.bias") },
                .q = try at.linear("self_attn.q_proj", "self_attn.q_proj", h, true),
                .k = try at.linear("self_attn.k_proj", "self_attn.k_proj", h, true),
                .v = try at.linear("self_attn.v_proj", "self_attn.v_proj", h, true),
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
        const cos = try g.fromSlice(f32, tabs.cos, &.{ n + 1, 1, hd });
        const sin = try g.fromSlice(f32, tabs.sin, &.{ n + 1, 1, hd });

        const at = try g.take(t.positions, try g.fromSlice(i32, tabs.taps, &.{ n, 4 }), 0);
        const placed = try g.sum(try g.mul(at, try g.fromSlice(f32, tabs.weights, &.{ n, 4, 1 })), 1, false);
        const patches = try g.add(try t.patch.forward(&g, try g.astype(pixels, dt)), try g.astype(placed, dt));
        var x = try g.layerNorm(try g.concat(&.{ try g.astype(t.class, dt), patches }, 0), t.pre_norm[0], t.pre_norm[1], cfg.eps);
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
};

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

    fn linear(a: At, leaf: []const u8, rel: []const u8, in_dim: u32, biased: bool) !Linear {
        var buf: [192]u8 = undefined;
        var l = try a.r.proj(try a.name(&buf, leaf, rel), in_dim);
        if (biased) {
            var bb: [192]u8 = undefined;
            const base = try a.name(&buf, leaf, rel);
            l.bias = try a.r.need(try std.fmt.bufPrint(&bb, "{s}.bias", .{base}));
        }
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
