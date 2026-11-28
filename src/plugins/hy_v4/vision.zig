//! Hy4-flash's SigLIP-style vision tower (`HYV4VLVisionModel`): a 27-block ViT over 16x16 patches,
//! learned 128x128 positional grid bilinearly resampled to the request's patch grid, full bidirectional
//! attention per image, then a 2x2 patch merger that appends one `image_newline` row per merged row.
//!
//! The tower reads the pack's `vit.*` tensors (bf16, dense): `vit.embeddings.patch_embedding`,
//! `vit.embeddings.position_embedding`, `vit.layers.*`, and the projector `vit.perceive.*`
//! (`norm` = the pre-merge RMSNorm, `proj.0` = the 2x2 conv, `proj.2` = the 1x1 conv, `image_newline`).
//!
//! Preprocessing follows the pack's `HYV4VLImageProcessor`: smart resize on the patch*merge grid within
//! [min_pixels, max_pixels], Torch's antialiased bicubic, rescale 1/255, mean/std 0.5, merge-major patchify.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const abi = @import("../../abi/plugin.zig");
const qvis = @import("../../sdk/vision/qwen_vision.zig");
const weights_load = @import("../../sdk/weights/load.zig");
const config = @import("config.zig");
const log = @import("../../sdk/util/log.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;

pub const Vision = struct {
    /// Conv weights laid out channels-last `[out, kh, kw, in]`; linears pre-transposed to `[in, out]`.
    patch_w: Array,
    patch_b: Array,
    pos_host: []const f32,
    pos_edge: u32,
    blocks: []Block,
    norm_w: Array,
    conv0_w: Array,
    conv0_b: Array,
    conv1_w: Array,
    conv1_b: Array,
    newline: Array,
    owned: std.ArrayList(Array),
    hidden: u32,
    heads: u32,
    head_dim: u32,
    inter: u32,
    out_hidden: u32,
    patch: u32,
    merge: u32,
    eps: f32,
    min_pixels: u32,
    max_pixels: u32,
    s: mlx.mlx_stream,

    pub fn deinit(self: *Vision) void {
        gpa.free(self.pos_host);
        gpa.free(self.blocks);
        for (self.owned.items) |x| _ = mlx.mlx_array_free(x);
        self.owned.deinit(gpa);
        gpa.destroy(self);
    }
};

const Block = struct {
    norm1_w: Array,
    norm1_b: Array,
    q_w: Array,
    q_b: Array,
    k_w: Array,
    k_b: Array,
    v_w: Array,
    v_b: Array,
    o_w: Array,
    o_b: Array,
    norm2_w: Array,
    norm2_b: Array,
    fc1_w: Array,
    fc1_b: Array,
    fc2_w: Array,
    fc2_b: Array,
};

fn own(owned: *std.ArrayList(Array), a: Array) !Array {
    try owned.append(gpa, a);
    return a;
}

fn transposeTo(x: Array, perm: []const c_int) !Array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_transpose_axes(&out, x, perm.ptr, @intCast(perm.len), mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

fn get(w: *const weights_load.Weights, comptime fmt: []const u8, args: anytype) !Array {
    const n = try std.fmt.allocPrint(gpa, fmt, args);
    defer gpa.free(n);
    return w.get(n) orelse error.MissingVisionWeight;
}

fn denseLinear(w: *const weights_load.Weights, owned: *std.ArrayList(Array), comptime fmt: []const u8, args: anytype) !struct { Array, Array } {
    const wt = try get(w, fmt ++ ".weight", args);
    const sh = ops.shape(wt);
    if (sh.len != 2) return error.InvalidVisionWeight;
    // Pack stores `[out, in]`; the forward reads `[in, out]`.
    const t = try transposeTo(wt, &.{ 1, 0 });
    try owned.append(gpa, t);
    if (sh[0] > (1 << 20) or sh[1] > (1 << 20)) return error.InvalidVisionWeight;
    const b = try get(w, fmt ++ ".bias", args);
    return .{ t, b };
}

pub fn load(c: *const config.Config, w: *const weights_load.Weights) !*Vision {
    var owned: std.ArrayList(Array) = .empty;
    errdefer {
        for (owned.items) |x| _ = mlx.mlx_array_free(x);
        owned.deinit(gpa);
    }
    const v = try gpa.create(Vision);
    errdefer gpa.destroy(v);
    const hidden: u32 = c.vision_hidden_size;
    const heads: u32 = c.vision_num_heads;
    const merge: u32 = c.vision_merge_size;
    const patch: u32 = c.vision_patch_size;
    if (hidden == 0 or heads == 0 or hidden % heads != 0) return error.InvalidVisionConfig;
    if (merge == 0 or patch == 0) return error.InvalidVisionConfig;

    // Patch embedding conv [hidden, 3, patch, patch] -> MLX [hidden, patch, patch, 3].
    const pw = try get(w, "vit.embeddings.patch_embedding.weight", .{});
    const pw_t = try transposeTo(pw, &.{ 0, 2, 3, 1 });
    try owned.append(gpa, pw_t);
    const pb = try get(w, "vit.embeddings.patch_embedding.bias", .{});

    // Learned 128x128 grid (row 0 is the unused cls row): keep an f32 host copy for bilinear resampling.
    const pos = try get(w, "vit.embeddings.position_embedding.weight", .{});
    try mlx.check(mlx.mlx_array_eval(pos));
    var pos32 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(pos32);
    try mlx.check(mlx.mlx_astype(&pos32, pos, .float32, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(pos32));
    defer _ = mlx.mlx_array_free(pos32);
    const pos_shape = ops.shape(pos32);
    const edge: u32 = c.vision_max_image_size / patch;
    if (pos_shape.len != 2 or pos_shape[0] != @as(c_int, @intCast(edge * edge + 1)) or pos_shape[1] != @as(c_int, @intCast(hidden)))
        return error.InvalidVisionPositionTable;
    const src = mlx.mlx_array_data_float32(pos32) orelse return error.NoData;
    const host = try gpa.alloc(f32, @as(usize, edge) * edge * hidden);
    errdefer gpa.free(host);
    // Skip the cls row: the tower's patch grid starts at row 1.
    @memcpy(host, src[@as(usize, hidden) .. @as(usize, hidden) + host.len]);

    const blocks = try gpa.alloc(Block, c.vision_depth);
    errdefer gpa.free(blocks);
    for (blocks, 0..) |*b, i| {
        const n: u32 = @intCast(i);
        const q = try denseLinear(w, &owned, "vit.layers.{d}.self_attn.q_proj", .{n});
        const k = try denseLinear(w, &owned, "vit.layers.{d}.self_attn.k_proj", .{n});
        const vv = try denseLinear(w, &owned, "vit.layers.{d}.self_attn.v_proj", .{n});
        const o = try denseLinear(w, &owned, "vit.layers.{d}.self_attn.o_proj", .{n});
        const fc1 = try denseLinear(w, &owned, "vit.layers.{d}.mlp.dense_h_to_4h", .{n});
        const fc2 = try denseLinear(w, &owned, "vit.layers.{d}.mlp.dense_4h_to_h", .{n});
        b.* = .{
            .norm1_w = try get(w, "vit.layers.{d}.input_layernorm.weight", .{n}),
            .norm1_b = try get(w, "vit.layers.{d}.input_layernorm.bias", .{n}),
            .q_w = q[0],
            .q_b = q[1],
            .k_w = k[0],
            .k_b = k[1],
            .v_w = vv[0],
            .v_b = vv[1],
            .o_w = o[0],
            .o_b = o[1],
            .norm2_w = try get(w, "vit.layers.{d}.post_attention_layernorm.weight", .{n}),
            .norm2_b = try get(w, "vit.layers.{d}.post_attention_layernorm.bias", .{n}),
            .fc1_w = fc1[0],
            .fc1_b = fc1[1],
            .fc2_w = fc2[0],
            .fc2_b = fc2[1],
        };
    }

    // Projector: `vit.perceive.norm` (pre-merge RMSNorm), `proj.0` (2x2 conv), `proj.2` (1x1 conv).
    const c0 = try get(w, "vit.perceive.proj.0.weight", .{});
    const c0_t = try transposeTo(c0, &.{ 0, 2, 3, 1 }); // [out, in, 2, 2] -> [out, 2, 2, in]
    try owned.append(gpa, c0_t);
    const c1 = try get(w, "vit.perceive.proj.2.weight", .{});
    const c1_t = try transposeTo(c1, &.{ 0, 2, 3, 1 }); // [out, in, 1, 1] -> [out, 1, 1, in]
    try owned.append(gpa, c1_t);

    v.* = .{
        .patch_w = pw_t,
        .patch_b = pb,
        .pos_host = host,
        .pos_edge = edge,
        .blocks = blocks,
        .norm_w = try get(w, "vit.perceive.norm.weight", .{}),
        .conv0_w = c0_t,
        .conv0_b = try get(w, "vit.perceive.proj.0.bias", .{}),
        .conv1_w = c1_t,
        .conv1_b = try get(w, "vit.perceive.proj.2.bias", .{}),
        .newline = try get(w, "vit.perceive.image_newline", .{}),
        .owned = owned,
        .hidden = hidden,
        .heads = heads,
        .head_dim = hidden / heads,
        .inter = c.vision_intermediate_size,
        .out_hidden = c.vision_out_hidden_size,
        .patch = patch,
        .merge = merge,
        .eps = c.vision_norm_eps,
        .min_pixels = c.vision_min_pixels,
        .max_pixels = c.vision_max_pixels,
        .s = mlx.gpuStream(),
    };
    log.info("Vision encoder: Hy4 ViT (depth={d}, hidden={d}, heads={d}, patch={d}, merge={d}, out_hidden={d})\n", .{
        c.vision_depth, hidden, heads, patch, merge, c.vision_out_hidden_size,
    });
    return v;
}

/// Bilinear resample of the learned 128x128 grid to the request's patch grid, Torch's align_corners=False
/// with the reference's `scale_factor = (size + 0.1) / edge` (effective source scale `edge / (size + 0.1)`).
fn interpolatePos(v: *const Vision, gh: u32, gw: u32) ![]f32 {
    const out = try gpa.alloc(f32, @as(usize, gh) * gw * v.hidden);
    errdefer gpa.free(out);
    const edge: usize = v.pos_edge;
    const hidden: usize = v.hidden;
    const sh = @as(f64, @floatFromInt(edge)) / (@as(f64, @floatFromInt(gh)) + 0.1);
    const sw = @as(f64, @floatFromInt(edge)) / (@as(f64, @floatFromInt(gw)) + 0.1);
    for (0..gh) |y| {
        const cy = (@as(f64, @floatFromInt(y)) + 0.5) * sh - 0.5;
        const y0: f64 = std.math.clamp(cy, 0, @as(f64, @floatFromInt(edge - 1)));
        const iy: usize = @intFromFloat(@floor(y0));
        const fy: f32 = @floatCast(y0 - @floor(y0));
        const iy1 = @min(iy + 1, edge - 1);
        for (0..gw) |x| {
            const cx = (@as(f64, @floatFromInt(x)) + 0.5) * sw - 0.5;
            const x0: f64 = std.math.clamp(cx, 0, @as(f64, @floatFromInt(edge - 1)));
            const ix: usize = @intFromFloat(@floor(x0));
            const fx: f32 = @floatCast(x0 - @floor(x0));
            const ix1 = @min(ix + 1, edge - 1);
            const dst = (y * gw + x) * hidden;
            const r00 = (iy * edge + ix) * hidden;
            const r01 = (iy * edge + ix1) * hidden;
            const r10 = (iy1 * edge + ix) * hidden;
            const r11 = (iy1 * edge + ix1) * hidden;
            for (0..hidden) |k| {
                const top = v.pos_host[r00 + k] * (1 - fx) + v.pos_host[r01 + k] * fx;
                const bot = v.pos_host[r10 + k] * (1 - fx) + v.pos_host[r11 + k] * fx;
                out[dst + k] = top * (1 - fy) + bot * fy;
            }
        }
    }
    return out;
}

fn linear(g: *Graph, x: Array, w: Array, b: Array) !Array {
    return g.add(try g.matmul(x, w), b);
}

fn gelu(g: *Graph, x: Array) !Array {
    const half = try g.scalarLike(x, 0.5);
    const inv_sqrt2 = try g.scalarLike(x, 1.0 / @sqrt(2.0));
    const erf = try g.erf(try g.mul(x, inv_sqrt2));
    return g.mul(try g.mul(x, half), try g.add(erf, try g.scalarLike(x, 1.0)));
}

fn attention(v: *const Vision, g: *Graph, x: Array, b: *const Block, n: c_int) !Array {
    var t = g.scope();
    defer t.deinit();
    const heads: c_int = @intCast(v.heads);
    const hd: c_int = @intCast(v.head_dim);
    const q = try linear(&t, x, b.q_w, b.q_b);
    const k = try linear(&t, x, b.k_w, b.k_b);
    const vv = try linear(&t, x, b.v_w, b.v_b);
    const qh = try t.transpose(try t.reshape(q, &.{ 1, n, heads, hd }), &.{ 0, 2, 1, 3 });
    const kh = try t.transpose(try t.reshape(k, &.{ 1, n, heads, hd }), &.{ 0, 2, 1, 3 });
    const vh = try t.transpose(try t.reshape(vv, &.{ 1, n, heads, hd }), &.{ 0, 2, 1, 3 });
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(v.head_dim)));
    const att = try t.sdpa(qh, kh, vh, scale, .none);
    const merged = try t.reshape(try t.transpose(att, &.{ 0, 2, 1, 3 }), &.{ n, heads * hd });
    return g.adopt(t.escape(try linear(&t, merged, b.o_w, b.o_b)));
}

/// The tower forward over `patches` `[gh*gw, 3*patch*patch]`: features `[rows, out_hidden]` with
/// `rows = (gh/merge) * (gw/merge + 1)` (one newline row per merged row).
pub fn forward(v: *Vision, patches: Array, gh: u32, gw: u32) !Array {
    if (gh % v.merge != 0 or gw % v.merge != 0) return error.UnalignedPatchGrid;
    const n: c_int = @intCast(gh * gw);
    const hidden: c_int = @intCast(v.hidden);
    const patch: c_int = @intCast(v.patch);
    var g = Graph.init(gpa, v.s);
    defer g.deinit();

    // Patch embedding conv (kernel=stride=patch), bias, and the interpolated learned positions.
    const p2 = try g.reshape(patches, &.{ n, 3, patch, patch });
    const nhwc = try g.transpose(p2, &.{ 0, 2, 3, 1 }); // [n, patch, patch, 3]
    const conv = try g.conv2d(nhwc, v.patch_w, patch, 0, 1, 1); // [n, 1, 1, hidden]
    var x = try g.reshape(conv, &.{ n, hidden });
    x = try g.add(x, v.patch_b);
    const pos = try interpolatePos(v, gh, gw);
    defer gpa.free(pos);
    const pos_arr = try g.fromSlice(f32, pos, &.{ n, hidden });
    x = try g.add(x, try g.astype(pos_arr, ops.dtype(x)));

    // One scope per block: a large image's attention scores are hundreds of MB, and 27 blocks held together
    // (the parent graph) blew Metal's resource limit. Each block's intermediates die at its own scope end.
    for (v.blocks) |*b| {
        var t = g.scope();
        defer t.deinit();
        const normed = try t.layerNorm(x, b.norm1_w, b.norm1_b, v.eps);
        const att = try attention(v, &t, normed, b, n);
        const h1 = try t.add(x, att);
        const normed2 = try t.layerNorm(h1, b.norm2_w, b.norm2_b, v.eps);
        const hh = try linear(&t, normed2, b.fc1_w, b.fc1_b);
        const act = try gelu(&t, hh);
        const y = try linear(&t, act, b.fc2_w, b.fc2_b);
        const next = try t.add(h1, y);
        x = try g.adopt(t.escape(next));
    }

    // Projector: RMSNorm -> 2x2 conv -> gelu -> 1x1 conv -> one newline row appended per merged row.
    const normed = try g.rmsNorm(x, v.norm_w, v.eps);
    const grid = try g.reshape(normed, &.{ 1, @intCast(gh), @intCast(gw), hidden });
    const down = try g.add(try g.conv2d(grid, v.conv0_w, 2, 0, 1, 1), v.conv0_b);
    const act = try gelu(&g, down);
    const out = try g.add(try g.conv2d(act, v.conv1_w, 1, 0, 1, 1), v.conv1_b); // [1, gh/2, gw/2, out]
    const mh: c_int = @intCast(gh / v.merge);
    const mw: c_int = @intCast(gw / v.merge);
    const out_h: c_int = @intCast(v.out_hidden);
    const nl = try g.broadcast(try g.reshape(v.newline, &.{ 1, 1, 1, out_h }), &.{ 1, mh, 1, out_h });
    const joined = try g.concat(&.{ out, nl }, 2); // [1, mh, mw+1, out]
    const final = try g.reshape(joined, &.{ mh * (mw + 1), out_h });
    return g.escape(final);
}

// ── media hooks ──

const C = config.Config;

fn cfgOf(p: *const abi.Parsed) *const C {
    return config.of(p);
}

/// `HYV4VLImageProcessor`: smart resize factor patch*merge within the pixel bounds, Torch antialiased
/// bicubic, rescale 1/255, mean/std 0.5, merge-major patchify (standard row-major patches, channel-major
/// features).
pub fn prepare(p: *const abi.Parsed, in: *const abi.MediaInput, budget: u64, out: *abi.MediaItem) callconv(.c) Verdict {
    const c = cfgOf(p);
    if (!c.has_vision) return Verdict.refuse(.unsupported, "hy_v4_flash: this pack has no vision tower");
    if (in.kind != .image) return Verdict.refuse(.unsupported, "hy_v4_flash: video input is not supported");
    if (in.width == 0 or in.height == 0 or in.rgb.len != @as(usize, in.width) * in.height * 3)
        return Verdict.refuse(.failed, "hy_v4_flash: an image is RGB8 pixels of width x height");
    const factor: u32 = c.vision_patch_size * c.vision_merge_size;
    const max_pixels: u32 = @intCast(@min(@as(u64, c.vision_max_pixels), budget));
    const size = qvis.smartResizeImage(in.height, in.width, factor, c.vision_min_pixels, max_pixels);
    const plane: usize = @as(usize, size.h) * size.w;
    const chw = gpa.alloc(f32, plane * 3) catch return Verdict.refuse(.failed, "hy_v4_flash: out of memory");
    defer gpa.free(chw);
    qvis.resizeRgbBicubicFloatChw(gpa, chw, in.rgb.slice(), in.height, in.width, size.h, size.w) catch
        return Verdict.refuse(.failed, "hy_v4_flash: image resize failed");
    const gh: u32 = size.h / c.vision_patch_size;
    const gw: u32 = size.w / c.vision_patch_size;
    const fpat: usize = 3 * c.vision_patch_size * c.vision_patch_size;
    const patches = gpa.alloc(f32, @as(usize, gh) * gw * fpat) catch return Verdict.refuse(.failed, "hy_v4_flash: out of memory");
    defer gpa.free(patches);
    const ps: usize = c.vision_patch_size;
    for (0..gh) |ph| for (0..gw) |pw| {
        const row = (ph * gw + pw) * fpat;
        for (0..3) |ch| for (0..ps) |py| for (0..ps) |px| {
            const src = ch * plane + (ph * ps + py) * @as(usize, size.w) + pw * ps + px;
            // rescale 1/255 then (x - 0.5)/0.5.
            patches[row + ch * ps * ps + py * ps + px] = chw[src] / 127.5 - 1.0;
        };
    };
    const arr = mlx.mlx_array_new_data(patches.ptr, &[_]c_int{ @intCast(gh * gw), @intCast(fpat) }, 2, .float32);
    if (arr.ctx == null) return Verdict.refuse(.failed, "hy_v4_flash: pixel upload failed");
    out.* = .{
        .kind = .image,
        .rows = (gh / c.vision_merge_size) * (gw / c.vision_merge_size + 1),
        .grid = .{ 1, gh, gw },
        .pixels = .{ .ctx = arr.ctx },
        .frames = 1,
    };
    return .ok;
}

/// The tower forward for one prepared image.
pub fn tower(m: *anyopaque, item: *const abi.MediaItem, out: *abi.Array) callconv(.c) Verdict {
    const wm: *const weights.Model = @ptrCast(@alignCast(m));
    const v = wm.trunk.vision orelse return Verdict.refuse(.unsupported, "hy_v4_flash: vision tower not loaded");
    const f = forward(v, .{ .ctx = item.pixels.ctx }, item.grid[1], item.grid[2]) catch |e| {
        log.err("hy_v4: vision tower failed: {t}\n", .{e});
        return Verdict.named(.failed, @errorName(e));
    };
    out.* = .{ .ctx = f.ctx };
    return .ok;
}

const weights = @import("weights.zig");
