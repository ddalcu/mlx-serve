//! Ming-Image's picture reader for edits: the release processor's prep (`BailingMM2ImageProcessor`: a bicubic
//! smart-resize to a fixed pixel budget, CLIP normalization, 2-frame 14x14 patches in 2x2 merge order), the
//! Qwen2.5-VL tower (`qwen2_5_vit.py`: windowed attention, full at `fullatt_block_indexes`, the 2x2 merger) and
//! `extract_image_feature` (`linear_proj`, then an L2 norm). One row per merged 2x2 patch block, MLLM-wide.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const weights_load = @import("../../sdk/weights/load.zig");
const qv = @import("../../sdk/vision/qwen_vision.zig");
const gelu = @import("../../sdk/blocks/mlp/gelu.zig");
const rope_blk = @import("../../sdk/blocks/dit/rope.zig");
const Linear = @import("../../sdk/blocks/dit/linear.zig").Linear;
const llm = @import("llm.zig");

const Weights = weights_load.Weights;
const Graph = ops.Graph;
const Array = ops.Array;
const S = mlx.mlx_stream;
const DT = llm.DT;

const MAX_FULL = 8;

pub const Config = struct {
    depth: u32,
    hidden: u32,
    heads: u32,
    patch: u32,
    temporal: u32,
    merge: u32,
    window: u32,
    full: [MAX_FULL]u32 = undefined,
    n_full: usize = 0,
    min_pixels: u32,
    max_pixels: u32,
    mean: [3]f32,
    std: [3]f32,

    pub fn parse(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !Config {
        var mllm = try llm.readJson(io, a, model_dir, "mllm/config.json");
        defer mllm.deinit();
        const v = mllm.value.object.get("vision_config") orelse return error.MingImageNoVision;
        var pre = try llm.readJson(io, a, model_dir, "mllm/preprocessor_config.json");
        defer pre.deinit();
        const p = pre.value;
        var c: Config = .{
            .depth = try llm.int(v, "depth"),
            .hidden = try llm.int(v, "hidden_size"),
            .heads = try llm.int(v, "num_heads"),
            .patch = try llm.int(v, "patch_size"),
            .temporal = try llm.int(v, "temporal_patch_size"),
            .merge = try llm.int(v, "spatial_merge_size"),
            .window = try llm.int(v, "window_size"),
            .min_pixels = try llm.int(p, "min_pixels"),
            .max_pixels = try llm.int(p, "max_pixels"),
            .mean = undefined,
            .std = undefined,
        };
        const full = v.object.get("fullatt_block_indexes") orelse return error.MingImageBadConfig;
        if (full != .array or full.array.items.len > MAX_FULL) return error.MingImageBadConfig;
        for (full.array.items) |it| {
            if (it != .integer or it.integer < 0) return error.MingImageBadConfig;
            c.full[c.n_full] = @intCast(it.integer);
            c.n_full += 1;
        }
        inline for (.{ .{ "image_mean", &c.mean }, .{ "image_std", &c.std } }) |f| {
            const arr = p.object.get(f[0]) orelse return error.MingImageBadConfig;
            if (arr != .array or arr.array.items.len != 3) return error.MingImageBadConfig;
            for (f[1], arr.array.items) |*o, it| o.* = @floatCast(switch (it) {
                .float => |x| x,
                .integer => |x| @as(f64, @floatFromInt(x)),
                else => return error.MingImageBadConfig,
            });
        }
        if (c.heads == 0 or c.hidden % c.heads != 0 or c.merge != 2 or c.window % (c.merge * c.patch) != 0) return error.MingImageBadConfig;
        return c;
    }

    fn isFull(c: Config, block: usize) bool {
        for (c.full[0..c.n_full]) |f| if (f == block) return true;
        return false;
    }

    /// One flattened patch's width: channels x frames x patch².
    fn patchDim(c: Config) u32 {
        return 3 * c.temporal * c.patch * c.patch;
    }
};

/// A picture as the tower reads it: `rows` patches of `patchDim` floats, on a `gh` x `gw` patch grid.
pub const Prepared = struct {
    pixels: []f32,
    gh: u32,
    gw: u32,

    pub fn deinit(p: Prepared, a: std.mem.Allocator) void {
        a.free(p.pixels);
    }

    /// The MLLM tokens the picture becomes (one per merged 2x2 block).
    pub fn tokens(p: Prepared) u32 {
        return (p.gh / 2) * (p.gw / 2);
    }
};

/// Pillow's resize of an RGBA picture: premultiplied for the resampling (`RGBA -> RGBa`, `MULDIV255`), then
/// divided back (integer, clipped; alpha 0 and 255 pass through), as `Image.resize` does for any filter but NEAREST.
/// The release resizes the edit's reference (bilinear) and its result (LANCZOS) this way.
pub fn resizeRgba(a: std.mem.Allocator, src: []const u8, h: u32, w: u32, th: u32, tw: u32, filter: qv.Filter) ![]u8 {
    const pre = try a.dupe(u8, src);
    defer a.free(pre);
    var i: usize = 0;
    while (i < pre.len) : (i += 4) for (0..3) |c| {
        const t: u32 = @as(u32, pre[i + c]) * pre[i + 3] + 128;
        pre[i + c] = @intCast(((t >> 8) + t) >> 8);
    };
    const out = try qv.resizeInterleavedPil(a, pre, h, w, th, tw, 4, filter);
    i = 0;
    while (i < out.len) : (i += 4) {
        const al: u32 = out[i + 3];
        if (al == 0 or al == 255) continue;
        for (0..3) |c| out[i + c] = @intCast(@min(255, (255 * @as(u32, out[i + c])) / al));
    }
    return out;
}

/// The MLLM tokens a `w` x `h` picture becomes (without reading it).
pub fn tokensFor(c: Config, h: u32, w: u32) u32 {
    const r = qv.smartResizeImage(h, w, c.patch * c.merge, c.min_pixels, c.max_pixels);
    return (r.h / c.patch / c.merge) * (r.w / c.patch / c.merge);
}

/// The release processor on an interleaved RGBA picture (alpha dropped, as PIL's RGB conversion does).
pub fn prepare(a: std.mem.Allocator, c: Config, rgba: []const u8, h: u32, w: u32) !Prepared {
    const rgb = try a.alloc(u8, @as(usize, h) * w * 3);
    defer a.free(rgb);
    for (0..@as(usize, h) * w) |i| @memcpy(rgb[i * 3 ..][0..3], rgba[i * 4 ..][0..3]);
    const factor = c.patch * c.merge;
    const r = qv.smartResizeImage(h, w, factor, c.min_pixels, c.max_pixels);
    const img = try qv.resizeInterleavedPil(a, rgb, h, w, r.h, r.w, 3, .bicubic);
    defer a.free(img);
    const gh = r.h / c.patch;
    const gw = r.w / c.patch;
    const pd = c.patchDim();
    const out = try a.alloc(f32, @as(usize, gh) * gw * pd);
    errdefer a.free(out);
    // Rows in 2x2 merge order; within a row (channel, frame, py, px), both frames the same picture.
    var row: usize = 0;
    var bh: u32 = 0;
    while (bh < gh / 2) : (bh += 1) {
        var bw: u32 = 0;
        while (bw < gw / 2) : (bw += 1) for (0..2) |mh| for (0..2) |mw| {
            const py0 = (bh * 2 + mh) * c.patch;
            const px0 = (bw * 2 + mw) * c.patch;
            var o = out[row * pd ..][0..pd];
            var k: usize = 0;
            for (0..3) |ch| for (0..c.temporal) |_| for (0..c.patch) |py| for (0..c.patch) |px| {
                const v: f32 = @floatFromInt(img[((py0 + py) * r.w + px0 + px) * 3 + ch]);
                o[k] = (v / 255.0 - c.mean[ch]) / c.std[ch];
                k += 1;
            };
            row += 1;
        };
    }
    return .{ .pixels = out, .gh = gh, .gw = gw };
}

/// The tower's window order over merged blocks (`get_window_index`) and each window's length, in merged blocks.
fn windows(a: std.mem.Allocator, c: Config, gh: u32, gw: u32) !struct { index: []u32, lens: []u32 } {
    const mh = gh / c.merge;
    const mw = gw / c.merge;
    const ws = c.window / c.merge / c.patch;
    const nh = (mh + ws - 1) / ws;
    const nw = (mw + ws - 1) / ws;
    const index = try a.alloc(u32, mh * mw);
    errdefer a.free(index);
    var lens: std.ArrayList(u32) = .empty;
    errdefer lens.deinit(a);
    var at: usize = 0;
    for (0..nh) |wy| for (0..nw) |wx| {
        const y1 = @min(mh, (wy + 1) * ws);
        const x1 = @min(mw, (wx + 1) * ws);
        for (wy * ws..y1) |y| for (wx * ws..x1) |x| {
            index[at] = @intCast(y * mw + x);
            at += 1;
        };
        try lens.append(a, @intCast((y1 - wy * ws) * (x1 - wx * ws)));
    };
    return .{ .index = index, .lens = try lens.toOwnedSlice(a) };
}

const Block = struct {
    norm1: Array,
    norm2: Array,
    qkv: Linear,
    proj: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
};

pub const Tower = struct {
    cfg: Config,
    w: Weights,
    blocks: []Block,
    patch_w: Array,
    ln_q: Array,
    m0: Linear,
    m2: Linear,
    p0: Linear,
    p2: Linear,
    a: std.mem.Allocator,

    pub fn load(self: *Tower, io: std.Io, a: std.mem.Allocator, s: S, model_dir: []const u8, cfg: Config) !void {
        self.cfg = cfg;
        self.a = a;
        self.w = try llm.loadDir(io, a, model_dir, "mllm", .{ .skip = notVision });
        errdefer self.w.deinit();
        if (self.w.get("vision.patch_embed.proj.weight") == null) return error.MingImageNoVision;
        try llm.castFloats(&self.w, DT, s);
        const w = &self.w;
        const d = cfg.hidden;
        self.blocks = try a.alloc(Block, cfg.depth);
        errdefer a.free(self.blocks);
        for (self.blocks, 0..) |*b, i| {
            const inter: u32 = @intCast(ops.shape((try llm.get(w, "vision.blocks.{d}.mlp.gate_proj.weight", .{i})))[0]);
            b.* = .{
                .norm1 = try llm.get(w, "vision.blocks.{d}.norm1.weight", .{i}),
                .norm2 = try llm.get(w, "vision.blocks.{d}.norm2.weight", .{i}),
                .qkv = try llm.lin(w, "vision.blocks.{d}.attn.qkv", .{i}, d),
                .proj = try llm.lin(w, "vision.blocks.{d}.attn.proj", .{i}, d),
                .gate = try llm.lin(w, "vision.blocks.{d}.mlp.gate_proj", .{i}, d),
                .up = try llm.lin(w, "vision.blocks.{d}.mlp.up_proj", .{i}, d),
                .down = try llm.lin(w, "vision.blocks.{d}.mlp.down_proj", .{i}, inter),
            };
        }
        // Conv3d [hidden, 3, frames, p, p] over non-overlapping patches = a matmul over the flattened patch.
        var pw = mlx.mlx_array_new();
        const shp = [_]c_int{ @intCast(d), @intCast(cfg.patchDim()) };
        try mlx.check(mlx.mlx_reshape(&pw, try llm.get(w, "vision.patch_embed.proj.weight", .{}), &shp, 2, s));
        try w.put("vision.patch_embed.flat", pw);
        self.patch_w = pw;
        try @import("../../mlx/scope.zig").evalAll(w);
        const md = d * cfg.merge * cfg.merge;
        self.ln_q = try llm.get(w, "vision.merger.ln_q.weight", .{});
        self.m0 = try llm.lin(w, "vision.merger.mlp.0", .{}, md);
        self.m2 = try llm.lin(w, "vision.merger.mlp.2", .{}, md);
        const emb: u32 = @intCast(ops.shape(self.m2.w)[0]);
        self.p0 = try llm.lin(w, "linear_proj.0", .{}, emb);
        self.p2 = try llm.lin(w, "linear_proj.2", .{}, @intCast(ops.shape(self.p0.w)[0]));
    }

    pub fn deinit(self: *Tower) void {
        self.a.free(self.blocks);
        self.w.deinit();
    }

    /// `[tokens, mllm hidden]` in DT for a prepared picture, in the merged blocks' raster order.
    pub fn forward(self: *const Tower, parent: *Graph, p: Prepared) !Array {
        var g = parent.scope();
        defer g.deinit();
        const c = self.cfg;
        const a = g.a;
        const n: usize = @as(usize, p.gh) * p.gw;
        const unit: usize = c.merge * c.merge;
        const hd = c.hidden / c.heads;
        const win = try windows(a, c, p.gh, p.gw);
        defer a.free(win.index);
        defer a.free(win.lens);

        // Patch rows reordered window by window (whole merged blocks move together); the rotary rows with them.
        const order = try a.alloc(u32, n);
        defer a.free(order);
        for (win.index, 0..) |m, i| for (0..unit) |u| {
            order[i * unit + u] = @intCast(m * unit + u);
        };
        const ord = try g.fromSlice(u32, order, &.{@intCast(n)});
        const px = try g.fromSlice(f32, p.pixels, &.{ @intCast(n), @intCast(c.patchDim()) });
        var x = try g.take(try g.matmul(try g.astype(px, DT), try g.transpose(self.patch_w, &.{ 1, 0 })), ord, 0);
        const rt = try self.rotary(&g, p, order);

        // Block-diagonal mask of the windows (merged-block lengths x 4 patch rows each).
        const mask = try a.alloc(bool, n * n);
        defer a.free(mask);
        @memset(mask, false);
        var at: usize = 0;
        for (win.lens) |len| {
            const r = len * unit;
            for (at..at + r) |i| @memset(mask[i * n + at ..][0..r], true);
            at += r;
        }
        const wmask = try g.fromSlice(bool, mask, &.{ 1, 1, @intCast(n), @intCast(n) });

        for (self.blocks, 0..) |*b, i| {
            var step = g.scope();
            defer step.deinit();
            const m: ops.Mask = if (c.isFull(i)) .none else .{ .array = wmask };
            const next = try g.keep(try self.block(&step, b, x, rt.cos, rt.sin, m, hd));
            try mlx.check(mlx.mlx_array_eval(next));
            x = next;
        }

        // Merger: per merged block (4 consecutive rows), back to raster order.
        const md: c_int = @intCast(c.hidden * unit);
        const merged = try g.reshape(try g.rmsNorm(x, self.ln_q, 1e-6), &.{ @intCast(n / unit), md });
        const y = try self.m2.forward(&g, try gelu.geluErf(&g, try self.m0.forward(&g, merged)));
        const inv = try a.alloc(u32, win.index.len);
        defer a.free(inv);
        for (win.index, 0..) |m, i| inv[m] = @intCast(i);
        const raster = try g.take(y, try g.fromSlice(u32, inv, &.{@intCast(inv.len)}), 0);
        const e = try self.p2.forward(&g, try gelu.geluErf(&g, try self.p0.forward(&g, raster)));
        const ef = try g.astype(e, .float32);
        const norm = try g.sqrt(try g.sum(try g.mul(ef, ef), -1, true));
        return parent.keep(try g.astype(try g.div(ef, try g.maximum(norm, try g.scalar(1e-12, .float32))), DT));
    }

    /// The 2-D rotary tables (h positions in the first half of each rotary pair set, w in the second), one row per
    /// patch in window order, `[1, 1, n, head_dim]` f32.
    fn rotary(self: *const Tower, g: *Graph, p: Prepared, order: []const u32) !struct { cos: Array, sin: Array } {
        const c = self.cfg;
        const hd = c.hidden / c.heads;
        const quarter = hd / 4; // inv_freq over head_dim/2 at stride 2
        const n = order.len;
        const cos = try g.a.alloc(f32, n * hd);
        defer g.a.free(cos);
        const sin = try g.a.alloc(f32, n * hd);
        defer g.a.free(sin);
        const mw = p.gw / 2;
        for (order, 0..) |src, i| {
            // A patch row in merge order: block (by, bx), then (my, mx) inside it.
            const blk = src / 4;
            const by = blk / mw;
            const bx = blk % mw;
            const hpos: f64 = @floatFromInt(by * 2 + (src % 4) / 2);
            const wpos: f64 = @floatFromInt(bx * 2 + src % 2);
            for (0..hd) |j| {
                const half = j % (hd / 2);
                const k = half % quarter;
                const pos = if (half < quarter) hpos else wpos;
                const inv: f32 = @floatCast(1.0 / std.math.pow(f64, 10000.0, @as(f64, @floatFromInt(2 * k)) / @as(f64, @floatFromInt(hd / 2))));
                const ang: f32 = @floatCast(pos * @as(f64, inv));
                cos[i * hd + j] = @cos(ang);
                sin[i * hd + j] = @sin(ang);
            }
        }
        const sh = [_]c_int{ 1, 1, @intCast(n), @intCast(hd) };
        return .{ .cos = try g.fromSlice(f32, cos, &sh), .sin = try g.fromSlice(f32, sin, &sh) };
    }

    fn block(self: *const Tower, parent: *Graph, b: *const Block, x: Array, cos: Array, sin: Array, m: ops.Mask, hd: u32) !Array {
        var g = parent.scope();
        defer g.deinit();
        const c = self.cfg;
        const n = ops.shape(x)[0];
        const nh: c_int = @intCast(c.heads);
        const qkv = try g.reshape(try b.qkv.forward(&g, try g.rmsNorm(x, b.norm1, 1e-6)), &.{ n, 3, nh, @intCast(hd) });
        const perm = [_]c_int{ 1, 0, 2 }; // [n, heads, hd] -> [heads, n, hd]
        var heads: [3]Array = undefined;
        for (&heads, 0..) |*h, i| {
            const t = try g.transpose(try g.reshape(try g.sliceAxis(qkv, 1, @intCast(i), @intCast(i + 1)), &.{ n, nh, @intCast(hd) }), &perm);
            h.* = try g.expandDims(t, 0);
        }
        const q = try g.astype(try rope_blk.ropeRotateHalf(&g, try g.astype(heads[0], .float32), cos, sin, .float32), DT);
        const k = try g.astype(try rope_blk.ropeRotateHalf(&g, try g.astype(heads[1], .float32), cos, sin, .float32), DT);
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(hd)));
        const at = try g.sdpa(q, k, heads[2], scale, m);
        const o = try g.reshape(try g.transpose(try g.reshape(at, &.{ nh, n, @intCast(hd) }), &perm), &.{ n, @intCast(c.hidden) });
        const h1 = try g.add(x, try b.proj.forward(&g, o));
        const hn = try g.rmsNorm(h1, b.norm2, 1e-6);
        const mlp = try b.down.forward(&g, try g.mul(try g.silu(try b.gate.forward(&g, hn)), try b.up.forward(&g, hn)));
        return parent.keep(try g.add(h1, mlp));
    }
};

fn notVision(key: []const u8) bool {
    return !(std.mem.startsWith(u8, key, "vision.") or std.mem.startsWith(u8, key, "linear_proj."));
}

const testing = std.testing;

test "ming_image vision: windows cover every merged block once, edge windows partial" {
    const c: Config = .{ .depth = 1, .hidden = 32, .heads = 2, .patch = 14, .temporal = 2, .merge = 2, .window = 112, .min_pixels = 0, .max_pixels = 0, .mean = .{ 0, 0, 0 }, .std = .{ 1, 1, 1 } };
    const w = try windows(testing.allocator, c, 10, 12); // merged 5 x 6, windows of 4 x 4
    defer testing.allocator.free(w.index);
    defer testing.allocator.free(w.lens);
    try testing.expectEqualSlices(u32, &.{ 16, 8, 4, 2 }, w.lens);
    var seen: [30]bool = @splat(false);
    for (w.index) |m| seen[m] = true;
    for (seen) |s| try testing.expect(s);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 6, 7 }, w.index[0..6]);
}
