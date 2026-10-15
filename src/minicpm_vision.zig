//! MiniCPM-V 4.7 (`minicpmv4_7`) vision: preprocessing and tower.
//!
//! Port of transformers' `MiniCPMV4_6ImageProcessorPil` (the processor the checkpoint maps to),
//! `MiniCPMV4_7VisionModel` and `MiniCPMV4_7Merger`. A SigLIP2 ViT at any aspect ratio (nearest
//! bucket lookup into a 70x70 position table, no rotary) whose encoder layer `insert_layer_id`
//! is followed by a 2x2 window-attention merger, then the LLM-width merger MLP. `16x` runs both
//! merges (4x4 patches per soft token), `4x` skips the window merger.
//!
//! The processor slices a large image into a thumbnail plus a grid of detail crops; every crop is
//! its own encoder call (its own patch grid), so one image is several pieces. Their soft tokens
//! splice at `<|image_pad|>` runs that `server.appendMiniCpmImageRun` lays out, and
//! `minicpm_canvas.zig` places those runs on the M-RoPE canvas.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;
const qwen_vision = @import("qwen_vision.zig");
const glm_vision = @import("glm_vision.zig");
const transformer = @import("transformer.zig");
const log = @import("log.zig");
const chat_mod = @import("chat.zig");

const Lin = glm_vision.Lin;
const replace = glm_vision.replace;

/// The processor's `scale_resolution`: the per-view pixel budget a crop is fitted to.
pub const SCALE_RESOLUTION: u32 = 448;
/// Two successive 2x2 merges (window merger, merger MLP) divide each side by 4.
const SIZE_QUANTUM_PATCHES: u32 = 4;
const CHANNELS: usize = 3;
/// `image_mean`/`image_std` of the checkpoint's preprocessor config.
const MEAN: f32 = 0.5;
const STD: f32 = 0.5;

pub const Params = struct {
    patch: u32 = 14,
    max_slice_nums: u32 = 9,
    scale_resolution: u32 = SCALE_RESOLUTION,
};

fn ensureDivide(length: f64, divisor: u32) u32 {
    const rounded = qwen_vision.roundHalfEven(length / @as(f64, @floatFromInt(divisor))) * @as(f64, @floatFromInt(divisor));
    return @max(@as(u32, @intFromFloat(rounded)), divisor);
}

const Size = struct { h: u32, w: u32 };

/// `find_best_resize`: sizes are floats because `get_refine_size` passes per-crop quotients.
fn findBestResize(height_in: f64, width_in: f64, scale_resolution: u32, patch: u32, allow_upscale: bool) Size {
    var height = height_in;
    var width = width_in;
    const scale: f64 = @floatFromInt(scale_resolution);
    if (height * width > scale * scale or allow_upscale) {
        const aspect = width / height;
        height = @trunc(scale / @sqrt(aspect));
        width = @trunc(height * aspect);
    }
    return .{ .h = ensureDivide(height, patch * SIZE_QUANTUM_PATCHES), .w = ensureDivide(width, patch * SIZE_QUANTUM_PATCHES) };
}

fn refineSize(height: u32, width: u32, rows: u32, cols: u32, p: Params) Size {
    const refine_w = ensureDivide(@floatFromInt(width), cols);
    const refine_h = ensureDivide(@floatFromInt(height), rows);
    const best = findBestResize(@as(f64, @floatFromInt(refine_h)) / @as(f64, @floatFromInt(rows)), @as(f64, @floatFromInt(refine_w)) / @as(f64, @floatFromInt(cols)), p.scale_resolution, p.patch, true);
    return .{ .h = best.h * rows, .w = best.w * cols };
}

pub const Grid = struct { rows: u32, cols: u32 };

/// `get_sliced_grid`: null when the image fits one view.
pub fn slicedGrid(height: u32, width: u32, max_slice_nums: u32, scale_resolution: u32) ?Grid {
    const h: f64 = @floatFromInt(height);
    const w: f64 = @floatFromInt(width);
    const scale: f64 = @floatFromInt(scale_resolution);
    const log_ratio = @log(w / h);
    const ratio = w * h / (scale * scale);
    const multiple: u32 = @intFromFloat(@min(@ceil(ratio), @as(f64, @floatFromInt(max_slice_nums))));
    if (multiple <= 1) return null;

    var best = Grid{ .rows = 1, .cols = 1 };
    var min_error = std.math.inf(f64);
    for ([_]u32{ multiple - 1, multiple, multiple + 1 }) |num_slices| {
        if (num_slices == 1 or num_slices > max_slice_nums) continue;
        for (1..num_slices + 1) |rows| {
            if (num_slices % rows != 0) continue;
            const cols = num_slices / rows;
            const err = @abs(log_ratio - @log(@as(f64, @floatFromInt(cols)) / @as(f64, @floatFromInt(rows))));
            if (err < min_error) {
                best = .{ .rows = @intCast(rows), .cols = @intCast(cols) };
                min_error = err;
            } else if (err == min_error and rows > best.rows) {
                best = .{ .rows = @intCast(rows), .cols = @intCast(cols) };
            }
        }
    }
    return best;
}

/// One encoder call: `bytes` is f32 `[grid_h*grid_w, 3*patch*patch]`, patches in raster order,
/// each flattened `[channel, row, col]` (the order the patch-embed conv weight reads).
pub const Piece = struct { bytes: []u8, h: u32, w: u32, grid_h: u32, grid_w: u32 };

pub const Sliced = struct {
    /// Thumbnail first, then the crops row-major.
    pieces: []Piece,
    /// 0x0 when the image fits one view.
    grid: Grid,

    pub fn deinit(self: Sliced, allocator: std.mem.Allocator) void {
        for (self.pieces) |piece| allocator.free(piece.bytes);
        allocator.free(self.pieces);
    }
};

fn normalizePixel(v: u8) f32 {
    const x: f32 = @floatCast(@as(f64, @floatFromInt(v)) * (1.0 / 255.0));
    return (x - MEAN) / STD;
}

/// Normalize a `rows x cols` window of `rgb` (HWC u8, row stride `stride` pixels) into the piece layout.
fn patchify(allocator: std.mem.Allocator, rgb: []const u8, stride: u32, y0: u32, x0: u32, h: u32, w: u32, patch: u32) !Piece {
    const gh = h / patch;
    const gw = w / patch;
    const feat: usize = CHANNELS * patch * patch;
    const bytes = try allocator.alloc(u8, @as(usize, gh) * gw * feat * 4);
    errdefer allocator.free(bytes);
    const out = @as([*]f32, @ptrCast(@alignCast(bytes.ptr)))[0 .. @as(usize, gh) * gw * feat];
    for (0..gh) |i| for (0..gw) |j| {
        const base = (i * gw + j) * feat;
        for (0..CHANNELS) |c| for (0..patch) |py| for (0..patch) |px| {
            const src = ((y0 + i * patch + py) * stride + x0 + j * patch + px) * CHANNELS + c;
            out[base + (c * patch + py) * patch + px] = normalizePixel(rgb[src]);
        };
    };
    return .{ .bytes = bytes, .h = h, .w = w, .grid_h = gh, .grid_w = gw };
}

fn resizeTo(allocator: std.mem.Allocator, rgb: []const u8, src_h: u32, src_w: u32, size: Size) ![]u8 {
    if (size.h == src_h and size.w == src_w) return allocator.dupe(u8, rgb);
    return qwen_vision.resizeInterleavedPil(allocator, rgb, src_h, src_w, size.h, size.w, 3, .bicubic);
}

/// The processor's `_preprocess` for one image: a thumbnail resized from the source, plus
/// the detail crops cut from a second, refined resize when the image needs slicing.
pub fn preprocess(allocator: std.mem.Allocator, rgb: []const u8, src_h: u32, src_w: u32, p: Params) !Sliced {
    if (src_h == 0 or src_w == 0) return error.InvalidImageDimensions;
    const grid = slicedGrid(src_h, src_w, p.max_slice_nums, p.scale_resolution);
    const thumb_size = findBestResize(@floatFromInt(src_h), @floatFromInt(src_w), p.scale_resolution, p.patch, grid == null);

    var pieces = std.ArrayList(Piece).empty;
    errdefer {
        for (pieces.items) |piece| allocator.free(piece.bytes);
        pieces.deinit(allocator);
    }
    {
        const thumb = try resizeTo(allocator, rgb, src_h, src_w, thumb_size);
        defer allocator.free(thumb);
        try pieces.append(allocator, try patchify(allocator, thumb, thumb_size.w, 0, 0, thumb_size.h, thumb_size.w, p.patch));
    }
    if (grid) |g| {
        const refine = refineSize(src_h, src_w, g.rows, g.cols, p);
        const refined = try resizeTo(allocator, rgb, src_h, src_w, refine);
        defer allocator.free(refined);
        const crop_h = refine.h / g.rows;
        const crop_w = refine.w / g.cols;
        for (0..g.rows) |row| for (0..g.cols) |col| {
            try pieces.append(allocator, try patchify(allocator, refined, refine.w, @intCast(row * crop_h), @intCast(col * crop_w), crop_h, crop_w, p.patch));
        };
    }
    return .{ .pieces = try pieces.toOwnedSlice(allocator), .grid = grid orelse .{ .rows = 0, .cols = 0 } };
}

/// Each piece as the server's `ImageData`: `tile_*` carry the slice grid, `tile_index` 0 is
/// the thumbnail and 1.. the crops. Moves ownership of the piece buffers into `list`.
pub fn appendImageData(allocator: std.mem.Allocator, list: *std.ArrayList(chat_mod.ImageData), sliced: Sliced) !void {
    const before = list.items.len;
    errdefer list.shrinkRetainingCapacity(before);
    for (sliced.pieces, 0..) |piece, k| {
        try list.append(allocator, .{
            .pixels = piece.bytes,
            .width = piece.w,
            .height = piece.h,
            .grid_h = piece.grid_h,
            .grid_w = piece.grid_w,
            .tile_rows = @intCast(sliced.grid.rows),
            .tile_cols = @intCast(sliced.grid.cols),
            .tile_index = @intCast(k),
        });
    }
}

/// `get_vision_nearest_position_ids`: each patch's row/column maps to the nearest bucket of a
/// `side`x`side` table through the reference's f32 boundaries and f32 fractional coordinates.
pub fn nearestPositionIds(allocator: std.mem.Allocator, grid_h: u32, grid_w: u32, side: u32) ![]i32 {
    const step: f64 = 1.0 / @as(f64, @floatFromInt(side));
    const count: usize = @intFromFloat(@ceil((1.0 - step) / step));
    const bounds = try allocator.alloc(f32, count);
    defer allocator.free(bounds);
    for (bounds, 0..) |*b, i| b.* = @floatCast(step + @as(f64, @floatFromInt(i)) * step);

    const rows = try allocator.alloc(i32, grid_h);
    defer allocator.free(rows);
    const cols = try allocator.alloc(i32, grid_w);
    defer allocator.free(cols);
    for (rows, 0..) |*r, i| r.* = bucket(bounds, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(grid_h)));
    for (cols, 0..) |*c, i| c.* = bucket(bounds, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(grid_w)));

    const ids = try allocator.alloc(i32, @as(usize, grid_h) * grid_w);
    for (rows, 0..) |r, i| for (cols, 0..) |c, j| {
        ids[i * grid_w + j] = r * @as(i32, @intCast(side)) + c;
    };
    return ids;
}

/// `torch.bucketize(v, bounds, right=True)`: how many boundaries are <= v.
fn bucket(bounds: []const f32, v: f32) i32 {
    var n: i32 = 0;
    for (bounds) |b| {
        if (b <= v) n += 1;
    }
    return n;
}

const Block = struct {
    ln1_w: mlx.mlx_array,
    ln1_b: mlx.mlx_array,
    ln2_w: mlx.mlx_array,
    ln2_b: mlx.mlx_array,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    fc1: Lin,
    fc2: Lin,
};

pub const MiniCpmVision = struct {
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,
    hidden: u32,
    heads: u32,
    head_dim: u32,
    eps: f32,
    pos_side: u32,
    insert_layer: u32,
    window_merge: bool,
    patch_w: mlx.mlx_array,
    patch_b: mlx.mlx_array,
    pos_table: mlx.mlx_array,
    blocks: []Block,
    post_w: mlx.mlx_array,
    post_b: mlx.mlx_array,
    /// The window-attention merger inserted after `insert_layer`.
    wm: Block,
    wm_pre_w: mlx.mlx_array,
    wm_pre_b: mlx.mlx_array,
    wm_1: Lin,
    wm_2: Lin,
    /// The LLM-width merger MLP (`merger_times` = 1).
    mg_pre_w: mlx.mlx_array,
    mg_pre_b: mlx.mlx_array,
    mg_1: Lin,
    mg_2: Lin,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !MiniCpmVision {
        const s = mlx.gpuStream();
        const hidden = config.vision_hidden_size;
        const inter = config.vision_intermediate_size;
        const ctx = NameCtx{ .weights = weights };

        const conv = try ctx.must("vision_tower.embeddings.patch_embedding.weight", .{});
        var patch_w = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(patch_w);
        {
            // OIHW in the HF checkpoint, OHWI in MLX packs; the patch rows are read [channel, row, col].
            const cs = mlx.getShape(conv);
            const p: c_int = @intCast(config.vision_patch_size);
            const oihw = cs.len == 4 and cs[1] == 3 and cs[2] == p and cs[3] == p;
            const ohwi = cs.len == 4 and cs[3] == 3 and cs[1] == p and cs[2] == p;
            if (!oihw and !ohwi) return error.InvalidVisionWeights;
            var t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(t);
            try mlx.check(mlx.mlx_transpose_axes(&t, conv, if (ohwi) &[_]c_int{ 0, 3, 1, 2 } else &[_]c_int{ 0, 1, 2, 3 }, 4, s));
            try mlx.check(mlx.mlx_reshape(&patch_w, t, &[_]c_int{ cs[0], 3 * p * p }, 2, s));
        }

        const blocks = try allocator.alloc(Block, config.vision_num_layers);
        errdefer allocator.free(blocks);
        for (blocks, 0..) |*b, i| b.* = try ctx.block("vision_tower.encoder.layers.{d}", .{i}, hidden, inter);

        const wm_prefix = "vit_merger";
        const window_hidden = hidden * 4;
        const pos_table = try ctx.positionTable(hidden, s);
        errdefer _ = mlx.mlx_array_free(pos_table);
        const pos_rows: u32 = @intCast(mlx.getShape(pos_table)[0]);
        const side = std.math.sqrt(pos_rows);
        if (side * side != pos_rows) return error.InvalidVisionWeights;

        var self = MiniCpmVision{
            .s = s,
            .allocator = allocator,
            .hidden = hidden,
            .heads = config.vision_num_heads,
            .head_dim = hidden / config.vision_num_heads,
            .eps = config.mcv_eps,
            .pos_side = side,
            .insert_layer = config.mcv_insert_layer,
            .window_merge = !config.mcv_downsample_4x,
            .patch_w = patch_w,
            .patch_b = try ctx.must("vision_tower.embeddings.patch_embedding.bias", .{}),
            .pos_table = pos_table,
            .blocks = blocks,
            .post_w = try ctx.must("vision_tower.post_layernorm.weight", .{}),
            .post_b = try ctx.must("vision_tower.post_layernorm.bias", .{}),
            .wm = try ctx.attnNorm(wm_prefix, hidden),
            .wm_pre_w = try ctx.must(wm_prefix ++ ".pre_norm.weight", .{}),
            .wm_pre_b = try ctx.must(wm_prefix ++ ".pre_norm.bias", .{}),
            .wm_1 = try ctx.lin(wm_prefix ++ ".linear_1", .{}, window_hidden),
            .wm_2 = try ctx.lin(wm_prefix ++ ".linear_2", .{}, inter * 4),
            .mg_pre_w = try ctx.must("merger.mlp.0.pre_norm.weight", .{}),
            .mg_pre_b = try ctx.must("merger.mlp.0.pre_norm.bias", .{}),
            .mg_1 = try ctx.lin("merger.mlp.0.linear_1", .{}, window_hidden),
            .mg_2 = try ctx.lin("merger.mlp.0.linear_2", .{}, window_hidden),
        };
        patch_w = .{ .ctx = null };
        try self.evalWeights();
        log.info("Vision encoder: MiniCPM-V 4.7 SigLIP2 ViT (layers={d}, hidden={d}, heads={d}, window merger after layer {d}, downsample={s})\n", .{
            config.vision_num_layers, hidden, config.vision_num_heads, config.mcv_insert_layer, if (config.mcv_downsample_4x) "4x" else "16x",
        });
        return self;
    }

    fn evalWeights(self: *MiniCpmVision) !void {
        var list = std.ArrayList(mlx.mlx_array).empty;
        defer list.deinit(self.allocator);
        try list.appendSlice(self.allocator, &.{ self.patch_w, self.patch_b, self.pos_table, self.post_w, self.post_b, self.wm_pre_w, self.wm_pre_b, self.mg_pre_w, self.mg_pre_b });
        for ([_]Lin{ self.wm_1, self.wm_2, self.mg_1, self.mg_2 }) |l| try glm_vision.appendLin(self.allocator, &list, l);
        for (self.blocks) |b| try appendBlock(self.allocator, &list, b);
        try appendBlock(self.allocator, &list, self.wm);
        const vec = mlx.mlx_vector_array_new_data(list.items.ptr, list.items.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
    }

    pub fn deinit(self: *MiniCpmVision) void {
        _ = mlx.mlx_array_free(self.patch_w);
        _ = mlx.mlx_array_free(self.pos_table);
        self.allocator.free(self.blocks);
    }

    /// Encode one piece: `patches` is `[gh*gw, 3*patch*patch]`; returns `[1, tokens, text_hidden]`.
    pub fn forward(self: *MiniCpmVision, patches: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        var gh = grid_h;
        var gw = grid_w;
        var x = try self.embed(patches, gh, gw);
        defer _ = mlx.mlx_array_free(x);
        for (self.blocks, 0..) |*b, i| {
            replace(&x, try self.block(x, b, 1));
            if (self.window_merge and i == self.insert_layer) {
                replace(&x, try self.windowMerge(x, gh, gw));
                gh /= 2;
                gw /= 2;
            }
        }
        replace(&x, try self.layerNorm(x, self.post_w, self.post_b));
        const feats = try self.merger(x, gh, gw);
        defer _ = mlx.mlx_array_free(feats);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_expand_dims(&out, feats, 0, self.s));
        return out;
    }

    /// Patch conv as a matmul, plus the nearest-bucket position embedding.
    pub fn embed(self: *MiniCpmVision, patches: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, patches, .bfloat16, self.s));
        replace(&x, try glm_vision.linear(x, .{ .w = self.patch_w, .bias = self.patch_b }, self.s));

        const ids = try nearestPositionIds(self.allocator, grid_h, grid_w, self.pos_side);
        defer self.allocator.free(ids);
        const idx = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(ids.len)}, 1, .int32);
        defer _ = mlx.mlx_array_free(idx);
        var pos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pos);
        try mlx.check(mlx.mlx_take_axis(&pos, self.pos_table, idx, 0, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&out, x, pos, self.s));
        return out;
    }

    /// One encoder layer over `groups` independent sequences of `x.len / groups` tokens.
    pub fn block(self: *MiniCpmVision, x: mlx.mlx_array, b: *const Block, groups: c_int) !mlx.mlx_array {
        var h = try self.layerNorm(x, b.ln1_w, b.ln1_b);
        defer _ = mlx.mlx_array_free(h);
        replace(&h, try self.attention(h, b, groups));
        var x1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x1);
        try mlx.check(mlx.mlx_add(&x1, x, h, self.s));

        replace(&h, try self.layerNorm(x1, b.ln2_w, b.ln2_b));
        replace(&h, try glm_vision.linear(h, b.fc1, self.s));
        replace(&h, try geluTanh(h, self.s));
        replace(&h, try glm_vision.linear(h, b.fc2, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&out, x1, h, self.s));
        return out;
    }

    /// Bidirectional attention inside each of `groups` equal sequences; `x` is `[groups*n, hidden]`.
    fn attention(self: *MiniCpmVision, x: mlx.mlx_array, b: *const Block, groups: c_int) !mlx.mlx_array {
        const total: c_int = mlx.getShape(x)[0];
        const n = @divExact(total, groups);
        const heads: c_int = @intCast(self.heads);
        const hd: c_int = @intCast(self.head_dim);
        var qkv: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (qkv) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for ([_]Lin{ b.q, b.k, b.v }, 0..) |l, j| {
            const proj = try glm_vision.linear(x, l, self.s);
            defer _ = mlx.mlx_array_free(proj);
            var r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(r);
            try mlx.check(mlx.mlx_reshape(&r, proj, &[_]c_int{ groups, n, heads, hd }, 4, self.s));
            try mlx.check(mlx.mlx_transpose_axes(&qkv[j], r, &[_]c_int{ 0, 2, 1, 3 }, 4, self.s));
        }
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const none = mlx.mlx_array{ .ctx = null };
        var ctx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ctx);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ctx, qkv[0], qkv[1], qkv[2], scale, "", none, none, false, self.s));
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_transpose_axes(&t, ctx, &[_]c_int{ 0, 2, 1, 3 }, 4, self.s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, t, &[_]c_int{ total, @intCast(self.hidden) }, 2, self.s));
        return glm_vision.linear(flat, b.o, self.s);
    }

    /// `[gh*gw, C]` raster rows -> `[gh/2*gw/2, 4, C]`: one row per 2x2 window, taps top-left first.
    fn windows(self: *MiniCpmVision, x: mlx.mlx_array, gh: u32, gw: u32) !mlx.mlx_array {
        const c: c_int = @intCast(self.hidden);
        var r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r);
        try mlx.check(mlx.mlx_reshape(&r, x, &[_]c_int{ @intCast(gh / 2), 2, @intCast(gw / 2), 2, c }, 5, self.s));
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_transpose_axes(&t, r, &[_]c_int{ 0, 2, 1, 3, 4 }, 5, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&out, t, &[_]c_int{ @intCast((gh / 2) * (gw / 2)), 4, c }, 3, self.s));
        return out;
    }

    /// `MiniCPMV4_7ViTWindowAttentionMerger`: attention inside each 2x2 window, then the
    /// window's four rows through an MLP plus their mean -> `[gh/2*gw/2, C]`.
    pub fn windowMerge(self: *MiniCpmVision, x: mlx.mlx_array, gh: u32, gw: u32) !mlx.mlx_array {
        if (gh % 2 != 0 or gw % 2 != 0) return error.UnalignedPatchGrid;
        const c: c_int = @intCast(self.hidden);
        const n_windows: c_int = @intCast((gh / 2) * (gw / 2));
        const normed = try self.layerNorm(x, self.wm.ln1_w, self.wm.ln1_b);
        defer _ = mlx.mlx_array_free(normed);
        const nw = try self.windows(normed, gh, gw);
        defer _ = mlx.mlx_array_free(nw);
        var rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rows);
        try mlx.check(mlx.mlx_reshape(&rows, nw, &[_]c_int{ n_windows * 4, c }, 2, self.s));
        const attn = try self.attention(rows, &self.wm, n_windows);
        defer _ = mlx.mlx_array_free(attn);
        const res = try self.windows(x, gh, gw);
        defer _ = mlx.mlx_array_free(res);
        var res_rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(res_rows);
        try mlx.check(mlx.mlx_reshape(&res_rows, res, &[_]c_int{ n_windows * 4, c }, 2, self.s));
        var summed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(summed);
        try mlx.check(mlx.mlx_add(&summed, res_rows, attn, self.s));

        var grouped = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(grouped);
        try mlx.check(mlx.mlx_reshape(&grouped, summed, &[_]c_int{ n_windows, 4, c }, 3, self.s));
        var mean = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mean);
        try mlx.check(mlx.mlx_mean_axis(&mean, grouped, 1, false, self.s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, summed, &[_]c_int{ n_windows, 4 * c }, 2, self.s));

        var h = try self.layerNorm(flat, self.wm_pre_w, self.wm_pre_b);
        defer _ = mlx.mlx_array_free(h);
        replace(&h, try glm_vision.linear(h, self.wm_1, self.s));
        replace(&h, try geluTanh(h, self.s));
        replace(&h, try glm_vision.linear(h, self.wm_2, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&out, h, mean, self.s));
        return out;
    }

    /// `MiniCPMV4_7Merger` with one round: 2x2 merge of the final grid, then the MLP.
    pub fn merger(self: *MiniCpmVision, x: mlx.mlx_array, gh: u32, gw: u32) !mlx.mlx_array {
        if (gh % 2 != 0 or gw % 2 != 0) return error.UnalignedPatchGrid;
        const grouped = try self.windows(x, gh, gw);
        defer _ = mlx.mlx_array_free(grouped);
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, grouped, &[_]c_int{ @intCast((gh / 2) * (gw / 2)), @intCast(4 * self.hidden) }, 2, self.s));
        var h = try self.layerNorm(flat, self.mg_pre_w, self.mg_pre_b);
        defer _ = mlx.mlx_array_free(h);
        replace(&h, try glm_vision.linear(h, self.mg_1, self.s));
        replace(&h, try glm_vision.geluErf(h, self.s));
        return glm_vision.linear(h, self.mg_2, self.s);
    }

    fn layerNorm(self: *MiniCpmVision, x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_layer_norm(&out, x, w, b, self.eps, self.s));
        return out;
    }

};

fn appendBlock(allocator: std.mem.Allocator, list: *std.ArrayList(mlx.mlx_array), b: Block) !void {
    for ([_]mlx.mlx_array{ b.ln1_w, b.ln1_b, b.ln2_w, b.ln2_b }) |a| {
        if (a.ctx != null) try list.append(allocator, a);
    }
    for ([_]Lin{ b.q, b.k, b.v, b.o, b.fc1, b.fc2 }) |l| {
        if (l.w.ctx != null) try glm_vision.appendLin(allocator, list, l);
    }
}

/// `gelu_pytorch_tanh`: `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`.
fn geluTanh(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(x);
    const k = try transformer.scalarOf(0.7978845608028654, dt, s);
    defer _ = mlx.mlx_array_free(k);
    const c = try transformer.scalarOf(0.044715, dt, s);
    defer _ = mlx.mlx_array_free(c);
    const one = try transformer.scalarOf(1.0, dt, s);
    defer _ = mlx.mlx_array_free(one);
    const half = try transformer.scalarOf(0.5, dt, s);
    defer _ = mlx.mlx_array_free(half);
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_multiply(&t, x, x, s));
    try mlx.check(mlx.mlx_multiply(&t, t, x, s));
    try mlx.check(mlx.mlx_multiply(&t, t, c, s));
    try mlx.check(mlx.mlx_add(&t, x, t, s));
    try mlx.check(mlx.mlx_multiply(&t, t, k, s));
    try mlx.check(mlx.mlx_tanh(&t, t, s));
    try mlx.check(mlx.mlx_add(&t, t, one, s));
    try mlx.check(mlx.mlx_multiply(&t, x, t, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, t, half, s));
    return out;
}

/// Weight lookup; handles are BORROWED from the weights map.
const NameCtx = struct {
    weights: *const Weights,

    fn get(self: NameCtx, comptime fmt: []const u8, args: anytype) ?mlx.mlx_array {
        var buf: [160]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, fmt, args) catch return null;
        return self.weights.get(key);
    }

    fn must(self: NameCtx, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
        return self.get(fmt, args) orelse {
            log.warn("MISSING MINICPM VISION WEIGHT: " ++ fmt ++ "\n", args);
            return error.MissingVisionWeights;
        };
    }

    /// The position table as a dense array the caller owns: mlx-vlm quantizes it like any embedding.
    fn positionTable(self: NameCtx, hidden: u32, s: mlx.mlx_stream) !mlx.mlx_array {
        const name = "vision_tower.embeddings.position_embedding";
        const w = try self.must(name ++ ".weight", .{});
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        const scales = self.get(name ++ ".scales", .{}) orelse {
            try mlx.check(mlx.mlx_copy(&out, w, s));
            return out;
        };
        const biases = self.get(name ++ ".biases", .{}) orelse mlx.mlx_array{ .ctx = null };
        const qp = transformer.quantParamsFromGeometry(w, scales, biases.ctx != null, hidden) orelse return error.InvalidVisionWeights;
        try mlx.check(mlx.mlx_dequantize(&out, w, scales, biases, mlx.mlx_optional_int.some(@intCast(qp.group_size)), mlx.mlx_optional_int.some(@intCast(qp.bits)), qp.mode.cstr(), .{}, .{ .value = .bfloat16, .has_value = true }, s));
        return out;
    }

    fn lin(self: NameCtx, comptime fmt: []const u8, args: anytype, in_dim: u32) !Lin {
        var l = Lin{ .w = try self.must(fmt ++ ".weight", args), .bias = self.get(fmt ++ ".bias", args) orelse .{ .ctx = null } };
        if (self.get(fmt ++ ".scales", args)) |sc| {
            l.scales = sc;
            l.biases = self.get(fmt ++ ".biases", args) orelse .{ .ctx = null };
            l.qp = transformer.quantParamsFromGeometry(l.w, sc, l.biases.ctx != null, in_dim) orelse {
                log.warn("MINICPM VISION: unreadable quantization on " ++ fmt ++ "\n", args);
                return error.InvalidVisionWeights;
            };
        }
        return l;
    }

    /// `layer_norm1`, `self_attn.{q,k,v,out}_proj`; the MLP half is left empty (the window merger has its own).
    fn attnNorm(self: NameCtx, comptime prefix: []const u8, hidden: u32) !Block {
        return .{
            .ln1_w = try self.must(prefix ++ ".layer_norm1.weight", .{}),
            .ln1_b = try self.must(prefix ++ ".layer_norm1.bias", .{}),
            .ln2_w = .{ .ctx = null },
            .ln2_b = .{ .ctx = null },
            .q = try self.lin(prefix ++ ".self_attn.q_proj", .{}, hidden),
            .k = try self.lin(prefix ++ ".self_attn.k_proj", .{}, hidden),
            .v = try self.lin(prefix ++ ".self_attn.v_proj", .{}, hidden),
            .o = try self.lin(prefix ++ ".self_attn.out_proj", .{}, hidden),
            .fc1 = .{ .w = .{ .ctx = null } },
            .fc2 = .{ .w = .{ .ctx = null } },
        };
    }

    fn block(self: NameCtx, comptime fmt: []const u8, args: anytype, hidden: u32, inter: u32) !Block {
        return .{
            .ln1_w = try self.must(fmt ++ ".layer_norm1.weight", args),
            .ln1_b = try self.must(fmt ++ ".layer_norm1.bias", args),
            .ln2_w = try self.must(fmt ++ ".layer_norm2.weight", args),
            .ln2_b = try self.must(fmt ++ ".layer_norm2.bias", args),
            .q = try self.lin(fmt ++ ".self_attn.q_proj", args, hidden),
            .k = try self.lin(fmt ++ ".self_attn.k_proj", args, hidden),
            .v = try self.lin(fmt ++ ".self_attn.v_proj", args, hidden),
            .o = try self.lin(fmt ++ ".self_attn.out_proj", args, hidden),
            .fc1 = try self.lin(fmt ++ ".mlp.fc1", args, hidden),
            .fc2 = try self.lin(fmt ++ ".mlp.fc2", args, inter),
        };
    }
};

// ── Tests ──

const testing = std.testing;

test "slicing and crop sizes equal the reference processor" {
    // MiniCPMV4_6ImageProcessorPil at the checkpoint's settings (patch 14, 448 views, up to 9 slices).
    const Case = struct { h: u32, w: u32, grid: ?Grid, pieces: []const [2]u32 };
    const cases = [_]Case{
        .{ .h = 487, .w = 730, .grid = .{ .rows = 1, .cols = 2 }, .pieces = &.{ .{ 28, 40 }, .{ 36, 28 }, .{ 36, 28 } } },
        .{ .h = 61, .w = 97, .grid = null, .pieces = &.{.{ 24, 40 }} },
        .{ .h = 900, .w = 2600, .grid = .{ .rows = 2, .cols = 4 }, .pieces = &.{ .{ 20, 56 }, .{ 28, 40 }, .{ 28, 40 }, .{ 28, 40 }, .{ 28, 40 }, .{ 28, 40 }, .{ 28, 40 }, .{ 28, 40 }, .{ 28, 40 } } },
    };
    const p = Params{};
    for (cases) |c| {
        const grid = slicedGrid(c.h, c.w, p.max_slice_nums, p.scale_resolution);
        try testing.expectEqual(c.grid, grid);
        const thumb = findBestResize(@floatFromInt(c.h), @floatFromInt(c.w), p.scale_resolution, p.patch, grid == null);
        try testing.expectEqual(c.pieces[0][0], thumb.h / p.patch);
        try testing.expectEqual(c.pieces[0][1], thumb.w / p.patch);
        if (grid) |g| {
            const refine = refineSize(c.h, c.w, g.rows, g.cols, p);
            try testing.expectEqual(c.pieces[1][0], refine.h / g.rows / p.patch);
            try testing.expectEqual(c.pieces[1][1], refine.w / g.cols / p.patch);
            try testing.expectEqual(1 + g.rows * g.cols, c.pieces.len);
        }
    }
}

test "a rounding tie in the crop size goes to the even multiple" {
    // Python's round(): 126 / 56 = 2.25 -> 2, 140 / 56 = 2.5 -> 2, 168 / 56 = 3.0.
    try testing.expectEqual(@as(u32, 112), ensureDivide(126, 56));
    try testing.expectEqual(@as(u32, 112), ensureDivide(140, 56));
    try testing.expectEqual(@as(u32, 168), ensureDivide(168, 56));
    try testing.expectEqual(@as(u32, 56), ensureDivide(3, 56));
}

test "a pack without the tower disables vision instead of failing the load" {
    var weights = Weights.init(testing.allocator);
    defer weights.deinit();
    const config = ModelConfig{ .minicpm_vision = true, .vision_num_layers = 27 };
    try testing.expectError(error.MissingVisionWeights, MiniCpmVision.init(testing.allocator, config, &weights));
}

test "nearest position ids bucket a grid onto the 70x70 table" {
    const a = testing.allocator;
    // One row of 70 patches indexes the table's columns one to one.
    const ids = try nearestPositionIds(a, 1, 70, 70);
    defer a.free(ids);
    for (ids, 0..) |id, i| try testing.expectEqual(@as(i32, @intCast(i)), id);
    // A 2x2 grid hits the four quadrants' nearest buckets.
    const small = try nearestPositionIds(a, 2, 2, 70);
    defer a.free(small);
    try testing.expectEqualSlices(i32, &.{ 0, 35, 35 * 70, 35 * 70 + 35 }, small);
}

// Preprocessing and tower parity vs the EXECUTED reference (tests/dump_minicpmv4_7_vision_fixtures.py):
//
//   MINICPM_VISION_MODEL=<pack dir> \
//   MINICPM_VISION_FIXTURE=~/claude-tmp/minicpm-vision/minicpm_vision_fixture.safetensors \
//   zig build test -Doptimize=ReleaseFast -Dtest-filter="minicpm vision"
fn fixtureEnv() ?[]const u8 {
    const raw = std.c.getenv("MINICPM_VISION_FIXTURE") orelse return null;
    const path = std.mem.sliceTo(raw, 0);
    return if (path.len == 0) null else path;
}

fn evalGet(fx: *const Weights, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
    var kb: [96]u8 = undefined;
    const arr = fx.get(try std.fmt.bufPrint(&kb, fmt, args)) orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(arr));
    return arr;
}

test "minicpm vision: preprocessing equals the reference processor's pixel_values" {
    const fix_path = fixtureEnv() orelse return error.SkipZigTest;
    const a = testing.allocator;
    var fx = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer fx.deinit();
    for ([_][]const u8{ "house", "noise", "wide" }) |case| {
        const rgb_arr = try evalGet(&fx, "{s}_rgb", .{case});
        const shape = mlx.getShape(rgb_arr);
        const h: u32 = @intCast(shape[0]);
        const w: u32 = @intCast(shape[1]);
        const rgb = @as([*]const u8, @ptrCast(mlx.mlx_array_data_uint8(rgb_arr)))[0 .. @as(usize, h) * w * 3];
        const sliced = try preprocess(a, rgb, h, w, .{});
        defer sliced.deinit(a);

        const grid = try evalGet(&fx, "{s}_grid", .{case});
        const g: [*]const i32 = @ptrCast(@alignCast(mlx.mlx_array_data_int32(grid)));
        try testing.expectEqual(@as(u32, @intCast(g[0])), sliced.grid.rows);
        try testing.expectEqual(@as(u32, @intCast(g[1])), sliced.grid.cols);
        const sizes = try evalGet(&fx, "{s}_target_sizes", .{case});
        const ts: [*]const i32 = @ptrCast(@alignCast(mlx.mlx_array_data_int32(sizes)));
        try testing.expectEqual(sliced.pieces.len * 2, mlx.mlx_array_size(sizes));

        var patches: usize = 0;
        for (sliced.pieces, 0..) |piece, k| {
            try testing.expectEqual(@as(u32, @intCast(ts[2 * k])), piece.grid_h);
            try testing.expectEqual(@as(u32, @intCast(ts[2 * k + 1])), piece.grid_w);
            patches += @as(usize, piece.grid_h) * piece.grid_w;
        }
        // Reference layout [1, 3, 14, patches*14]: patch g owns columns g*14 .. g*14+14.
        const want_arr = try evalGet(&fx, "{s}_pixel_values", .{case});
        try testing.expectEqual(3 * 14 * patches * 14, mlx.mlx_array_size(want_arr));
        const want = @as([*]const f32, @ptrCast(@alignCast(mlx.mlx_array_data_float32(want_arr))))[0 .. 3 * 14 * patches * 14];
        var worst: f32 = 0;
        var at: usize = 0;
        for (sliced.pieces) |piece| {
            const got = @as([*]const f32, @ptrCast(@alignCast(piece.bytes.ptr)))[0 .. piece.bytes.len / 4];
            const n = @as(usize, piece.grid_h) * piece.grid_w;
            for (0..n) |k| for (0..3) |c| for (0..14) |py| for (0..14) |px| {
                const x = got[k * 588 + (c * 14 + py) * 14 + px];
                const y = want[(c * 14 + py) * (patches * 14) + (at + k) * 14 + px];
                worst = @max(worst, @abs(x - y));
            };
            at += n;
        }
        std.debug.print("[minicpm-vision] {s} {d}x{d}: {d} pieces, {d} patches, pixel_values max|diff|={e}\n", .{ case, h, w, sliced.pieces.len, patches, worst });
        try testing.expect(worst < 1e-5);
    }
}

fn concat0(parts: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new_data(parts.ptr, parts.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 0, s));
    return out;
}

test "minicpm vision: tower stages and merger match the reference" {
    const fix_path = fixtureEnv() orelse return error.SkipZigTest;
    const raw_model = std.c.getenv("MINICPM_VISION_MODEL") orelse return error.SkipZigTest;
    const model_dir = std.mem.sliceTo(raw_model, 0);
    const a = testing.allocator;

    var config = try model_mod.parseConfig(testing.io, a, model_dir);
    defer config.deinit(a);
    var weights = try model_mod.loadWeightsWithVision(testing.io, a, model_dir);
    defer weights.deinit();
    var fx = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer fx.deinit();
    var mv = try MiniCpmVision.init(a, config, &weights);
    defer mv.deinit();
    const s = mv.s;

    for ([_][]const u8{ "noise", "house" }) |case| {
        const rgb_arr = try evalGet(&fx, "{s}_rgb", .{case});
        const shape = mlx.getShape(rgb_arr);
        const rgb = @as([*]const u8, @ptrCast(mlx.mlx_array_data_uint8(rgb_arr)))[0 .. @as(usize, @intCast(shape[0])) * @as(usize, @intCast(shape[1])) * 3];
        const sliced = try preprocess(a, rgb, @intCast(shape[0]), @intCast(shape[1]), .{});
        defer sliced.deinit(a);

        for ([_][]const u8{ "16x", "4x" }) |mode| {
            const merged = std.mem.eql(u8, mode, "16x");
            var stages: [5]std.ArrayList(mlx.mlx_array) = @splat(.empty);
            defer for (&stages) |*list| {
                for (list.items) |arr| _ = mlx.mlx_array_free(arr);
                list.deinit(a);
            };
            for (sliced.pieces) |piece| {
                const px = mlx.mlx_array_new_data(piece.bytes.ptr, &[_]c_int{ @intCast(piece.grid_h * piece.grid_w), 588 }, 2, .float32);
                defer _ = mlx.mlx_array_free(px);
                var gh = piece.grid_h;
                var gw = piece.grid_w;
                var x = try mv.embed(px, gh, gw);
                try stages[0].append(a, try dup(x, s));
                for (mv.blocks, 0..) |*b, i| {
                    replace(&x, try mv.block(x, b, 1));
                    if (i == mv.insert_layer) {
                        try stages[1].append(a, try dup(x, s));
                        if (merged) {
                            replace(&x, try mv.windowMerge(x, gh, gw));
                            gh /= 2;
                            gw /= 2;
                            try stages[2].append(a, try dup(x, s));
                        }
                    }
                }
                replace(&x, try mv.layerNorm(x, mv.post_w, mv.post_b));
                try stages[3].append(a, try dup(x, s));
                try stages[4].append(a, try mv.merger(x, gh, gw));
                _ = mlx.mlx_array_free(x);
            }
            const names = [_][]const u8{ "embeddings", "layer_insert", "vit_merger", "post_ln", "features" };
            for (names, 0..) |name, si| {
                if (si == 2 and !merged) continue;
                const got = try concat0(stages[si].items, s);
                defer _ = mlx.mlx_array_free(got);
                const want = try evalGet(&fx, "{s}_{s}_{s}", .{ case, mode, name });
                const cos = try glm_vision.cosineSim(got, want, s);
                const ratio = try glm_vision.rmsRatio(got, want, s);
                std.debug.print("[minicpm-vision] {s} {s} {s}: cos={d:.6} rms_ratio={d:.4}\n", .{ case, mode, name, cos, ratio });
                // bf16 serving against an fp32 reference through 27 layers.
                try testing.expect(cos > 0.99 and ratio > 0.97 and ratio < 1.03);
            }
        }
    }
}

fn dup(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_copy(&out, x, s));
    return out;
}
