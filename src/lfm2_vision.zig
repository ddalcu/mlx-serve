//! LFM2-VL vision tower + preprocessing math.
//!
//! Port of `transformers/models/lfm2_vl/` (`modeling_lfm2_vl.py`'s projector +
//! the `siglip2_vision_model` tower it wraps, `image_processing_lfm2_vl.py`).
//! A vanilla pre-LN SigLIP2 ViT in NaFlex form: the patch embedder is a plain
//! Linear over flattened patches (no conv), the learned 16x16 position table is
//! resampled to each image's own patch grid, and attention is unmasked full
//! attention — we encode one image at a time, so the reference's packing mask
//! never applies. The projector 2x2-unshuffles the feature map and runs
//! Linear -> GELU -> Linear into language-model space.
//!
//! Feature layout is the trap: `convert_image_to_patches` emits `[py, px, c]`
//! with CHANNEL INNERMOST, where Qwen and Muse both put channel outermost.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;
const qwen_vision = @import("qwen_vision.zig");
const log = @import("log.zig");

pub const Resized = qwen_vision.Resized;

/// A tile grid: `cols` x `rows` tiles of `tile_size` pixels each.
pub const TileGrid = struct { cols: u32, rows: u32 };

/// `smart_resize` (image_processing_lfm2_vl.py): snap each side to a multiple
/// of `patch * downsample` so the unshuffle needs no padding, then rescale into
/// the [min_tokens, max_tokens] budget preserving aspect ratio. Token counts
/// are POST-unshuffle, so the pixel bounds carry a `downsample²` factor.
///
/// Nearly Qwen's `_smart_resize_image` but not quite: the initial snap is
/// floored at one full factor, which changes the BRANCH for an image thinner
/// than one patch in either axis. Kept separate rather than shared for that.
pub fn smartResize(
    height: u32,
    width: u32,
    patch: u32,
    downsample: u32,
    min_tokens: u32,
    max_tokens: u32,
) Resized {
    const ff: f64 = @floatFromInt(patch * downsample);
    const fh: f64 = @floatFromInt(height);
    const fw: f64 = @floatFromInt(width);
    const px: f64 = @floatFromInt(patch * patch * downsample * downsample);
    const min_pixels: f64 = @as(f64, @floatFromInt(min_tokens)) * px;
    const max_pixels: f64 = @as(f64, @floatFromInt(max_tokens)) * px;

    var h_bar = @max(ff, qwen_vision.roundHalfEven(fh / ff) * ff);
    var w_bar = @max(ff, qwen_vision.roundHalfEven(fw / ff) * ff);
    if (h_bar * w_bar > max_pixels) {
        const beta = @sqrt((fh * fw) / max_pixels);
        h_bar = @max(ff, std.math.floor(fh / beta / ff) * ff);
        w_bar = @max(ff, std.math.floor(fw / beta / ff) * ff);
    } else if (h_bar * w_bar < min_pixels) {
        const beta = @sqrt(min_pixels / (fh * fw));
        h_bar = std.math.ceil(fh * beta / ff) * ff;
        w_bar = std.math.ceil(fw * beta / ff) * ff;
    }
    return .{ .h = @intFromFloat(h_bar), .w = @intFromFloat(w_bar) };
}

/// `_is_image_too_large`: whether the source exceeds the single-tile token
/// budget by more than `tolerance`, i.e. whether it gets SPLIT into tiles.
/// The snap here floors at ONE PATCH (not one full factor) — the reference's
/// own asymmetry with `smart_resize`, and it decides the branch for thin images.
pub fn isImageTooLarge(
    height: u32,
    width: u32,
    patch: u32,
    downsample: u32,
    max_tokens: u32,
    tolerance: f64,
) bool {
    const ff: f64 = @floatFromInt(patch * downsample);
    const fp: f64 = @floatFromInt(patch);
    const h_bar = @max(fp, qwen_vision.roundHalfEven(@as(f64, @floatFromInt(height)) / ff) * ff);
    const w_bar = @max(fp, qwen_vision.roundHalfEven(@as(f64, @floatFromInt(width)) / ff) * ff);
    const budget: f64 = @as(f64, @floatFromInt(max_tokens)) *
        @as(f64, @floatFromInt(patch * patch * downsample * downsample)) * tolerance;
    return h_bar * w_bar > budget;
}

/// `_get_grid_layout` -> `find_closest_aspect_ratio`: pick the tile grid whose
/// aspect is closest to the image's, over every (w, h) with
/// `min_tiles <= w*h <= max_tiles`. Ties go to the LARGER target area when the
/// image covers more than half of it — the reference's own tiebreak, which is
/// order-dependent in Python and reproduced here by iterating the same way
/// (ascending w*h, then w, then h).
pub fn gridLayout(height: u32, width: u32, min_tiles: u32, max_tiles: u32, tile_size: u32) TileGrid {
    const aspect: f64 = @as(f64, @floatFromInt(width)) / @as(f64, @floatFromInt(height));
    const area: f64 = @as(f64, @floatFromInt(width)) * @as(f64, @floatFromInt(height));
    const ts: f64 = @floatFromInt(tile_size);

    var best = TileGrid{ .cols = 1, .rows = 1 };
    var best_diff: f64 = std.math.inf(f64);
    // The reference builds a sorted set keyed on w*h, so equal-product ratios
    // arrive in the order Python's `sorted` leaves them; a stable pass over
    // (product, w, h) reproduces it.
    var product: u32 = min_tiles;
    while (product <= max_tiles) : (product += 1) {
        var w: u32 = 1;
        while (w <= max_tiles) : (w += 1) {
            var h: u32 = 1;
            while (h <= max_tiles) : (h += 1) {
                if (w * h != product) continue;
                const target = @as(f64, @floatFromInt(w)) / @as(f64, @floatFromInt(h));
                const diff = @abs(aspect - target);
                if (diff < best_diff) {
                    best_diff = diff;
                    best = .{ .cols = w, .rows = h };
                } else if (diff == best_diff) {
                    const target_area = ts * ts * @as(f64, @floatFromInt(w * h));
                    if (area > 0.5 * target_area) best = .{ .cols = w, .rows = h };
                }
            }
        }
    }
    return best;
}

/// Build the tower's `pixel_values` [gh*gw, patch*patch*C] from a normalized
/// CHW image. Row-major grid order; the per-patch feature runs `[py, px, c]`
/// with CHANNEL INNERMOST (`convert_image_to_patches`) — the opposite of every
/// other tower we serve, and invisible to a cosine test on a grey image.
pub fn buildPixelValues(out: []f32, img_chw: []const f32, C: u32, rh: u32, rw: u32, patch: u32) void {
    buildPixelValuesRegion(out, img_chw, C, rh, rw, patch, 0, 0, rh, rw);
}

/// `buildPixelValues` over a sub-rectangle of `img_chw`. A tiled image is
/// resized ONCE onto the full tile canvas and then read tile by tile, so a tile
/// is a window into that canvas rather than its own resample — which is what
/// `split_to_tiles` does, and re-resizing per tile would not be.
pub fn buildPixelValuesRegion(
    out: []f32,
    img_chw: []const f32,
    C: u32,
    rh: u32,
    rw: u32,
    patch: u32,
    y0: u32,
    x0: u32,
    region_h: u32,
    region_w: u32,
) void {
    const gh = region_h / patch;
    const gw = region_w / patch;
    const feat = C * patch * patch;
    std.debug.assert(out.len == @as(usize, gh) * gw * feat);
    std.debug.assert(y0 + region_h <= rh and x0 + region_w <= rw);
    const plane: usize = @as(usize, rh) * rw;

    for (0..gh) |row| {
        for (0..gw) |col| {
            const base = (row * gw + col) * feat;
            var f: usize = 0;
            for (0..patch) |py| {
                const y = y0 + row * patch + py;
                for (0..patch) |px| {
                    const x = x0 + col * patch + px;
                    for (0..C) |c| {
                        out[base + f] = img_chw[c * plane + y * rw + x];
                        f += 1;
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ViT encoder. One still image per call.
// forward(pixel_values [gh*gw, C*ps*ps], gh, gw) → [1, gh*gw/downsample², text_hidden].
// ─────────────────────────────────────────────────────────────────────────────

/// A tower linear. LiquidAI's MLX packs quantize only the language model, so
/// every tensor here is dense bf16 today — but `.scales` presence is decided
/// PER TENSOR so a pack that quantizes the tower loads too.
const Lin = struct {
    w: mlx.mlx_array,
    scales: mlx.mlx_array = .{ .ctx = null },
    biases: mlx.mlx_array = .{ .ctx = null },
    bias: mlx.mlx_array = .{ .ctx = null },
    bits: u32 = 0,
    group: u32 = 0,
    mode: [*:0]const u8 = "affine",
};

const Block = struct {
    ln1_w: mlx.mlx_array,
    ln1_b: mlx.mlx_array,
    ln2_w: mlx.mlx_array,
    ln2_b: mlx.mlx_array,
    q: Lin,
    k: Lin,
    v: Lin,
    out: Lin,
    fc1: Lin,
    fc2: Lin,
};

pub const Lfm2Vision = struct {
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,

    hidden: u32,
    heads: u32,
    head_dim: u32,
    pos_side: u32,
    downsample: u32,
    out_hidden: u32,
    ln_eps: f32,

    patch: Lin,
    pos_table: mlx.mlx_array,
    blocks: []Block,
    post_ln_w: mlx.mlx_array,
    post_ln_b: mlx.mlx_array,
    proj1: Lin,
    proj2: Lin,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !Lfm2Vision {
        const s = mlx.gpuStream();
        // LiquidAI's MLX packs ship the tower bare (`vision_tower.…`); the HF
        // original nests it under `model.vision_tower.vision_model.` and puts
        // the projector one level up from that. Probe the pair, don't guess.
        const Nesting = struct { tower: []const u8, proj: []const u8 };
        const nesting: Nesting = for ([_]Nesting{
            .{ .tower = "vision_tower.", .proj = "" },
            .{ .tower = "model.vision_tower.", .proj = "model." },
            .{ .tower = "model.vision_tower.vision_model.", .proj = "model." },
        }) |candidate| {
            var probe: [200]u8 = undefined;
            const key = std.fmt.bufPrint(&probe, "{s}embeddings.patch_embedding.weight", .{candidate.tower}) catch unreachable;
            if (weights.get(key) != null) break candidate;
        } else {
            log.warn("MISSING LFM2 VISION WEIGHT: vision_tower.embeddings.patch_embedding.weight\n", .{});
            return error.MissingVisionWeights;
        };

        var tower_buf: [200]u8 = undefined;
        var proj_buf: [200]u8 = undefined;
        const mode = config.quant_mode.cstr();
        const ctx = NameCtx{ .weights = weights, .prefix = nesting.tower, .buf = &tower_buf, .mode = mode };
        const pctx = NameCtx{ .weights = weights, .prefix = nesting.proj, .buf = &proj_buf, .mode = mode };

        const hidden = config.vision_hidden_size;
        const inter = config.vision_intermediate_size;
        const patch_dim = 3 * config.vision_patch_size * config.vision_patch_size;

        var blocks = try allocator.alloc(Block, config.vision_num_layers);
        errdefer allocator.free(blocks);
        for (0..config.vision_num_layers) |i| {
            blocks[i] = .{
                .ln1_w = try ctx.must("encoder.layers.{d}.layer_norm1.weight", .{i}),
                .ln1_b = try ctx.must("encoder.layers.{d}.layer_norm1.bias", .{i}),
                .ln2_w = try ctx.must("encoder.layers.{d}.layer_norm2.weight", .{i}),
                .ln2_b = try ctx.must("encoder.layers.{d}.layer_norm2.bias", .{i}),
                .q = try ctx.lin("encoder.layers.{d}.self_attn.q_proj", .{i}, hidden),
                .k = try ctx.lin("encoder.layers.{d}.self_attn.k_proj", .{i}, hidden),
                .v = try ctx.lin("encoder.layers.{d}.self_attn.v_proj", .{i}, hidden),
                .out = try ctx.lin("encoder.layers.{d}.self_attn.out_proj", .{i}, hidden),
                .fc1 = try ctx.lin("encoder.layers.{d}.mlp.fc1", .{i}, hidden),
                .fc2 = try ctx.lin("encoder.layers.{d}.mlp.fc2", .{i}, inter),
            };
        }

        const heads = config.vision_num_heads;
        log.info("Vision encoder: LFM2-VL SigLIP2-NaFlex ViT (depth={d}, hidden={d}, heads={d}, patch={d}, downsample={d}, out_hidden={d})\n", .{
            config.vision_num_layers, hidden, heads, config.vision_patch_size, config.lv_downsample, config.hidden_size,
        });
        return .{
            .s = s,
            .allocator = allocator,
            .hidden = hidden,
            .heads = heads,
            .head_dim = hidden / heads,
            .pos_side = config.lv_pos_side,
            .downsample = config.lv_downsample,
            .out_hidden = config.hidden_size,
            .ln_eps = config.lv_ln_eps,
            .patch = try ctx.lin("embeddings.patch_embedding", .{}, patch_dim),
            .pos_table = try ctx.must("embeddings.position_embedding.weight", .{}),
            .blocks = blocks,
            .post_ln_w = try ctx.must("post_layernorm.weight", .{}),
            .post_ln_b = try ctx.must("post_layernorm.bias", .{}),
            .proj1 = try pctx.lin("multi_modal_projector.linear_1", .{}, hidden * config.lv_downsample * config.lv_downsample),
            .proj2 = try pctx.lin("multi_modal_projector.linear_2", .{}, config.lv_projector_hidden),
        };
    }

    pub fn deinit(self: *Lfm2Vision) void {
        self.allocator.free(self.blocks);
    }

    /// Encode one image. `patches` is [gh*gw, C*ps*ps] in the reference's own
    /// `[py, px, c]` feature order; the result is [1, tokens, text_hidden],
    /// ready to splice at the image-token positions.
    pub fn forward(self: *Lfm2Vision, patches: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const hidden = try self.towerHidden(patches, grid_h, grid_w);
        defer _ = mlx.mlx_array_free(hidden);
        return self.project(hidden, grid_h, grid_w);
    }

    /// The tower alone, through `post_layernorm` → [gh*gw, hidden]. Split out
    /// from `forward` so a parity failure names the tower or the projector
    /// rather than "the vision path".
    pub fn towerHidden(self: *Lfm2Vision, patches: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        if (grid_h % self.downsample != 0 or grid_w % self.downsample != 0) return error.UnalignedPatchGrid;
        const n: c_int = @intCast(grid_h * grid_w);

        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&x, patches, .bfloat16, self.s));
        defer _ = mlx.mlx_array_free(x);

        replace(&x, try self.linear(x, self.patch));
        {
            const pos = try self.posEmbed(grid_h, grid_w);
            defer _ = mlx.mlx_array_free(pos);
            var sum = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&sum, x, pos, self.s));
            replace(&x, sum);
        }

        var dt = mlx.DtypeTrace.begin("lfm2-vision", x, if (self.blocks.len > 0) self.blocks[0].ln1_w else null);
        for (self.blocks, 0..) |blk, i| {
            {
                const normed = try self.layerNorm(x, blk.ln1_w, blk.ln1_b);
                defer _ = mlx.mlx_array_free(normed);
                const attn = try self.attention(normed, blk, n);
                defer _ = mlx.mlx_array_free(attn);
                var h = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h, x, attn, self.s));
                replace(&x, h);
            }
            {
                const normed = try self.layerNorm(x, blk.ln2_w, blk.ln2_b);
                defer _ = mlx.mlx_array_free(normed);
                const up = try self.linear(normed, blk.fc1);
                defer _ = mlx.mlx_array_free(up);
                // The encoder MLP is `gelu_pytorch_tanh`; the projector below is
                // plain erf `gelu`. Two acts, one checkpoint.
                const act = try self.geluTanh(up);
                defer _ = mlx.mlx_array_free(act);
                const down = try self.linear(act, blk.fc2);
                defer _ = mlx.mlx_array_free(down);
                var h = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h, x, down, self.s));
                replace(&x, h);
            }
            dt.layer(x, i);
        }
        dt.end(x);
        replace(&x, try self.layerNorm(x, self.post_ln_w, self.post_ln_b));

        const out = x;
        x = mlx.mlx_array_new();
        return out;
    }

    /// `Lfm2VlMultiModalProjector` — unshuffle, then Linear/GELU/Linear into
    /// language-model space. [gh*gw, hidden] → [1, tokens, text_hidden].
    fn project(self: *Lfm2Vision, hidden: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        var x = try self.pixelUnshuffle(hidden, grid_h, grid_w);
        defer _ = mlx.mlx_array_free(x);
        replace(&x, try self.linear(x, self.proj1));
        replace(&x, try self.geluErf(x));
        replace(&x, try self.linear(x, self.proj2));

        const tokens: c_int = @intCast((grid_h / self.downsample) * (grid_w / self.downsample));
        var out = mlx.mlx_array_new();
        const oshape = [_]c_int{ 1, tokens, @intCast(self.out_hidden) };
        try mlx.check(mlx.mlx_reshape(&out, x, &oshape, 3, self.s));
        return out;
    }

    /// `Lfm2VlMultiModalProjector.pixel_unshuffle`: fold each `downsample`x
    /// `downsample` block of patches into one row, channel-minor within the
    /// block. [N, D] → [N/f², f²·D], row-major over the reduced grid.
    fn pixelUnshuffle(self: *Lfm2Vision, x: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const f = self.downsample;
        const d: c_int = @intCast(self.hidden);
        const gh: c_int = @intCast(grid_h);
        const fh: c_int = @intCast(grid_h / f);
        const fw: c_int = @intCast(grid_w / f);
        const fi: c_int = @intCast(f);

        var cur = mlx.mlx_array_new();
        {
            const shape = [_]c_int{ gh, fw, fi * d };
            try mlx.check(mlx.mlx_reshape(&cur, x, &shape, 3, self.s));
        }
        const swap = [_]c_int{ 1, 0, 2 };
        {
            var t = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_transpose_axes(&t, cur, &swap, 3, self.s));
            replace(&cur, t);
        }
        {
            const shape = [_]c_int{ fw, fh, fi * fi * d };
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&r, cur, &shape, 3, self.s));
            replace(&cur, r);
        }
        {
            var t = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_transpose_axes(&t, cur, &swap, 3, self.s));
            replace(&cur, t);
        }
        {
            const shape = [_]c_int{ fh * fw, fi * fi * d };
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&r, cur, &shape, 2, self.s));
            replace(&cur, r);
        }
        return cur;
    }

    /// The learned `pos_side`x`pos_side` table resampled onto this image's patch
    /// grid, row-major → [gh*gw, hidden]. Separable, as two f32 matmuls against
    /// PIL-convention triangle weights — the reference's
    /// `interpolate(mode="bilinear", antialias=True)`, whose footprint widening
    /// is load-bearing whenever a grid axis is SHORTER than `pos_side`.
    fn posEmbed(self: *Lfm2Vision, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const side: c_int = @intCast(self.pos_side);
        const d: c_int = @intCast(self.hidden);

        var table = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(table);
        {
            var f32_table = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f32_table);
            try mlx.check(mlx.mlx_astype(&f32_table, self.pos_table, .float32, self.s));
            const shape = [_]c_int{ side, side * d };
            try mlx.check(mlx.mlx_reshape(&table, f32_table, &shape, 2, self.s));
        }

        // Height axis: [gh, side] @ [side, side*D] → [gh, side*D].
        var cur = try self.resampleAxis(table, grid_h, @intCast(side));
        errdefer _ = mlx.mlx_array_free(cur);
        {
            const shape = [_]c_int{ @intCast(grid_h), side, d };
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&r, cur, &shape, 3, self.s));
            replace(&cur, r);
        }
        // Width axis: move it to the front so the same matmul applies.
        const swap = [_]c_int{ 1, 0, 2 };
        {
            var t = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_transpose_axes(&t, cur, &swap, 3, self.s));
            replace(&cur, t);
            const shape = [_]c_int{ side, @as(c_int, @intCast(grid_h)) * d };
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&r, cur, &shape, 2, self.s));
            replace(&cur, r);
        }
        {
            const next = try self.resampleAxis(cur, grid_w, @intCast(side));
            replace(&cur, next);
            const shape = [_]c_int{ @intCast(grid_w), @intCast(grid_h), d };
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&r, cur, &shape, 3, self.s));
            replace(&cur, r);
            var t = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_transpose_axes(&t, cur, &swap, 3, self.s));
            replace(&cur, t);
        }
        {
            const shape = [_]c_int{ @as(c_int, @intCast(grid_h)) * @as(c_int, @intCast(grid_w)), d };
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&r, cur, &shape, 2, self.s));
            replace(&cur, r);
        }
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&out, cur, .bfloat16, self.s));
        _ = mlx.mlx_array_free(cur);
        return out;
    }

    /// `[out_len, in_len] @ x` with PIL-convention triangle weights.
    fn resampleAxis(self: *Lfm2Vision, x: mlx.mlx_array, out_len: u32, in_len: u32) !mlx.mlx_array {
        const host = try self.allocator.alloc(f32, @as(usize, out_len) * in_len);
        defer self.allocator.free(host);
        try qwen_vision.resampleWeightMatrix(self.allocator, host, in_len, out_len, .bilinear);
        const shape = [_]c_int{ @intCast(out_len), @intCast(in_len) };
        const w = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(w);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&out, w, x, self.s));
        return out;
    }

    fn attention(self: *Lfm2Vision, x: mlx.mlx_array, blk: Block, n: c_int) !mlx.mlx_array {
        const hd: c_int = @intCast(self.head_dim);
        const heads: c_int = @intCast(self.heads);

        // [N, D] → [1, heads, N, hd] for the fused SDPA. No mask: we encode one
        // image per call, so the reference's packing mask is all-ones.
        var bhnd: [3]mlx.mlx_array = undefined;
        var built: usize = 0;
        errdefer for (bhnd[0..built]) |arr| {
            _ = mlx.mlx_array_free(arr);
        };
        inline for (.{ blk.q, blk.k, blk.v }, 0..) |l, i| {
            const flat = try self.linear(x, l);
            defer _ = mlx.mlx_array_free(flat);
            const shape = [_]c_int{ n, heads, hd };
            var r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(r);
            try mlx.check(mlx.mlx_reshape(&r, flat, &shape, 3, self.s));
            const perm = [_]c_int{ 1, 0, 2 };
            var t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(t);
            try mlx.check(mlx.mlx_transpose_axes(&t, r, &perm, 3, self.s));
            var b = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&b, t, &[_]c_int{ 1, heads, n, hd }, 4, self.s));
            bhnd[i] = b;
            built += 1;
        }
        defer for (bhnd) |arr| {
            _ = mlx.mlx_array_free(arr);
        };

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const none = mlx.mlx_array{ .ctx = null };
        var ctx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ctx);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ctx, bhnd[0], bhnd[1], bhnd[2], scale, "", none, none, false, self.s));

        const back = [_]c_int{ 0, 2, 1, 3 };
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_transpose_axes(&t, ctx, &back, 4, self.s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, t, &[_]c_int{ n, @intCast(self.hidden) }, 2, self.s));
        return self.linear(flat, blk.out);
    }

    fn layerNorm(self: *Lfm2Vision, x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_layer_norm(&out, x, w, b, self.ln_eps, self.s));
        return out;
    }

    /// `gelu_pytorch_tanh` — the encoder MLP's activation.
    fn geluTanh(self: *Lfm2Vision, x: mlx.mlx_array) !mlx.mlx_array {
        const k = bf16Scalar(0.7978845608028654, self.s); // sqrt(2/pi)
        defer _ = mlx.mlx_array_free(k);
        const c = bf16Scalar(0.044715, self.s);
        defer _ = mlx.mlx_array_free(c);
        const one = bf16Scalar(1.0, self.s);
        defer _ = mlx.mlx_array_free(one);
        const half = bf16Scalar(0.5, self.s);
        defer _ = mlx.mlx_array_free(half);

        var x3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x3);
        try mlx.check(mlx.mlx_multiply(&x3, x, x, self.s));
        var cube = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cube);
        try mlx.check(mlx.mlx_multiply(&cube, x3, x, self.s));
        var scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_multiply(&scaled, cube, c, self.s));
        var inner = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inner);
        try mlx.check(mlx.mlx_add(&inner, x, scaled, self.s));
        var arg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(arg);
        try mlx.check(mlx.mlx_multiply(&arg, inner, k, self.s));
        var th = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(th);
        try mlx.check(mlx.mlx_tanh(&th, arg, self.s));
        var onep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(onep);
        try mlx.check(mlx.mlx_add(&onep, one, th, self.s));
        var xt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xt);
        try mlx.check(mlx.mlx_multiply(&xt, x, onep, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&out, xt, half, self.s));
        return out;
    }

    /// Exact erf `gelu` — `projector_hidden_act`.
    fn geluErf(self: *Lfm2Vision, x: mlx.mlx_array) !mlx.mlx_array {
        const inv_sqrt2 = bf16Scalar(0.7071067811865476, self.s);
        defer _ = mlx.mlx_array_free(inv_sqrt2);
        const one = bf16Scalar(1.0, self.s);
        defer _ = mlx.mlx_array_free(one);
        const half = bf16Scalar(0.5, self.s);
        defer _ = mlx.mlx_array_free(half);
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_multiply(&t, x, inv_sqrt2, self.s));
        var e = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(e);
        try mlx.check(mlx.mlx_erf(&e, t, self.s));
        var onep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(onep);
        try mlx.check(mlx.mlx_add(&onep, one, e, self.s));
        var xt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xt);
        try mlx.check(mlx.mlx_multiply(&xt, x, onep, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&out, xt, half, self.s));
        return out;
    }

    fn linear(self: *Lfm2Vision, x: mlx.mlx_array, l: Lin) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        if (l.scales.ctx != null) {
            try mlx.check(mlx.mlx_quantized_matmul(
                &out,
                x,
                l.w,
                l.scales,
                l.biases,
                true,
                mlx.mlx_optional_int.some(@intCast(l.group)),
                mlx.mlx_optional_int.some(@intCast(l.bits)),
                l.mode,
                self.s,
            ));
        } else {
            var wt = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wt);
            try mlx.check(mlx.mlx_transpose(&wt, l.w, self.s));
            try mlx.check(mlx.mlx_matmul(&out, x, wt, self.s));
        }
        if (l.bias.ctx != null) {
            var biased = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&biased, out, l.bias, self.s));
            _ = mlx.mlx_array_free(out);
            out = biased;
        }
        return out;
    }
};

fn replace(dst: *mlx.mlx_array, next: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(dst.*);
    dst.* = next;
}

fn bf16Scalar(v: f32, s: mlx.mlx_stream) mlx.mlx_array {
    const f = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&out, f, .bfloat16, s);
    return out;
}

/// Weight lookup under the checkpoint's nesting. Handles are BORROWED from the
/// weights map — the tower and the projector nest differently, so each gets its
/// own instance rather than one context guessing between them.
const NameCtx = struct {
    weights: *const Weights,
    prefix: []const u8,
    buf: *[200]u8,
    mode: [*:0]const u8,

    fn key(self: NameCtx, comptime fmt: []const u8, args: anytype) []const u8 {
        const body = std.fmt.bufPrint(self.buf[100..], fmt, args) catch unreachable;
        return std.fmt.bufPrint(self.buf[0..100], "{s}{s}", .{ self.prefix, body }) catch unreachable;
    }

    fn opt(self: NameCtx, comptime fmt: []const u8, args: anytype) ?mlx.mlx_array {
        return self.weights.get(self.key(fmt, args));
    }

    fn must(self: NameCtx, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
        return self.opt(fmt, args) orelse {
            log.warn("MISSING LFM2 VISION WEIGHT: {s}\n", .{self.key(fmt, args)});
            return error.MissingVisionWeights;
        };
    }

    fn lin(self: NameCtx, comptime fmt: []const u8, args: anytype, in_features: u32) !Lin {
        const w = try self.must(fmt ++ ".weight", args);
        var l = Lin{ .w = w, .bias = self.opt(fmt ++ ".bias", args) orelse .{ .ctx = null }, .mode = self.mode };
        if (self.opt(fmt ++ ".scales", args)) |sc| {
            l.scales = sc;
            l.biases = self.opt(fmt ++ ".biases", args) orelse .{ .ctx = null };
            const w_cols: u32 = @intCast(mlx.getShape(w)[1]);
            const s_cols: u32 = @intCast(mlx.getShape(sc)[1]);
            l.bits = @divExact(32 * w_cols, in_features);
            l.group = @divExact(in_features, s_cols);
        }
        return l;
    }
};

// Vision-tower parity vs the EXECUTED reference
// (tests/dump_lfm2_vision_fixtures.py runs transformers' own Siglip2VisionModel
// plus LFM2-VL's projector on OUR pack's weights — which ship dense bf16 in
// every quant width — so a diff is a layout/math bug, never quantization error).
//
//   LFM2_VISION_MODEL="/Volumes/G Drive SSD/models-dl/LiquidAI/LFM2.5-VL-3B-MLX-4bit" \
//   LFM2_VISION_FIXTURE=~/claude-tmp/lfm2-vision/lfm2_vision_fixture.safetensors \
//   zig build test -Doptimize=ReleaseFast -Dtest-filter="lfm2 vision live"
test "lfm2 vision live: tower parity vs the executed reference" {
    const raw_model = std.c.getenv("LFM2_VISION_MODEL") orelse return error.SkipZigTest;
    const raw_fix = std.c.getenv("LFM2_VISION_FIXTURE") orelse return error.SkipZigTest;
    const model_dir = std.mem.sliceTo(raw_model, 0);
    const fix_path = std.mem.sliceTo(raw_fix, 0);
    if (model_dir.len == 0 or fix_path.len == 0) return error.SkipZigTest;
    const a = testing.allocator;

    const config = try model_mod.parseConfig(std.testing.io, a, model_dir);
    var weights = try model_mod.loadWeightsWithVision(std.testing.io, a, model_dir);
    defer weights.deinit();
    var fx = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer fx.deinit();

    var lv = try Lfm2Vision.init(a, config, &weights);
    defer lv.deinit();
    const s = lv.s;

    // The position resample runs before every block, so a diff here makes
    // nothing downstream attributable. Both directions are covered: 32x32 only
    // upsamples the stored 16x16 table, 32x8 and 8x32 DOWNSAMPLE one axis,
    // which is the only place the anti-aliasing footprint changes the answer
    // (mlx-vlm's bicubic scores cos 0.99 / rms 1.13 against this).
    for ([_][2]u32{ .{ 14, 20 }, .{ 32, 32 }, .{ 32, 8 }, .{ 8, 32 }, .{ 26, 36 }, .{ 16, 64 } }) |g| {
        var name_buf: [32]u8 = undefined;
        const key = try std.fmt.bufPrint(&name_buf, "pos_{d}x{d}", .{ g[0], g[1] });
        const want = fx.get(key) orelse continue;
        const pos = try lv.posEmbed(g[0], g[1]);
        defer _ = mlx.mlx_array_free(pos);
        const c = try cosineSim(pos, want, s);
        const r = try rmsRatio(pos, want, s);
        std.debug.print("[lfm2-vit] {s} cos={d:.6} rms_ratio={d:.4}\n", .{ key, c, r });
        try testing.expect(c > 0.9999 and r > 0.995 and r < 1.005);
    }

    for ([_][]const u8{ "a", "b" }) |case| {
        var kb: [32]u8 = undefined;
        const grid_arr = fx.get(try std.fmt.bufPrint(&kb, "{s}_grid", .{case})) orelse return error.MissingFixtureTensor;
        try mlx.check(mlx.mlx_array_eval(grid_arr));
        const g: [*]const i32 = @ptrCast(@alignCast(mlx.mlx_array_data_int32(grid_arr)));
        const gh: u32 = @intCast(g[0]);
        const gw: u32 = @intCast(g[1]);
        const pv = fx.get(try std.fmt.bufPrint(&kb, "{s}_pixel_values", .{case})) orelse return error.MissingFixtureTensor;

        const hidden = try lv.towerHidden(pv, gh, gw);
        defer _ = mlx.mlx_array_free(hidden);
        {
            const want = fx.get(try std.fmt.bufPrint(&kb, "{s}_hidden", .{case})) orelse return error.MissingFixtureTensor;
            const c = try cosineSim(hidden, want, s);
            const r = try rmsRatio(hidden, want, s);
            std.debug.print("[lfm2-vit] {s} grid {d}x{d} hidden cos={d:.6} rms_ratio={d:.4}\n", .{ case, gh, gw, c, r });
            try testing.expect(c > 0.99 and r > 0.98 and r < 1.02);
        }

        const feats = try lv.project(hidden, gh, gw);
        defer _ = mlx.mlx_array_free(feats);
        const want = fx.get(try std.fmt.bufPrint(&kb, "{s}_features", .{case})) orelse return error.MissingFixtureTensor;
        const c = try cosineSim(feats, want, s);
        // MAGNITUDE too: these rows are spliced into the token stream, which is
        // exactly where a scale error hides from a cosine.
        const r = try rmsRatio(feats, want, s);
        std.debug.print("[lfm2-vit] {s} features cos={d:.6} rms_ratio={d:.4}\n", .{ case, c, r });
        try testing.expect(r > 0.98 and r < 1.02);
        // Ours serves bf16 against an fp32 reference through 27 blocks, so the
        // bar is "same features", not bit equality — a layout bug lands far below.
        try testing.expect(c > 0.99);
    }
}

fn cosineSim(a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    const dot = try sumSq(a_arr, b_arr, s);
    const na = try sumSq(a_arr, a_arr, s);
    const nb = try sumSq(b_arr, b_arr, s);
    if (!std.math.isFinite(dot) or na <= 0 or nb <= 0) return std.math.nan(f32);
    return dot / (@sqrt(na) * @sqrt(nb));
}

fn rmsRatio(a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    const na = try sumSq(a_arr, a_arr, s);
    const nb = try sumSq(b_arr, b_arr, s);
    if (!std.math.isFinite(na) or nb <= 0) return std.math.nan(f32);
    return @sqrt(na) / @sqrt(nb);
}

/// sum(a*b) in fp32 over flattened inputs. NaN propagates: `NaN > threshold` is
/// false, so an all-NaN candidate can never pass the comparisons above.
fn sumSq(a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var af = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(af);
    try mlx.check(mlx.mlx_astype(&af, a_arr, .float32, s));
    var bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bf);
    try mlx.check(mlx.mlx_astype(&bf, b_arr, .float32, s));
    const n = [_]c_int{@intCast(mlx.mlx_array_size(af))};
    var a1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a1);
    try mlx.check(mlx.mlx_reshape(&a1, af, &n, 1, s));
    var b1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b1);
    try mlx.check(mlx.mlx_reshape(&b1, bf, &n, 1, s));
    var prod = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prod);
    try mlx.check(mlx.mlx_multiply(&prod, a1, b1, s));
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_sum(&o, prod, false, s));
    try mlx.check(mlx.mlx_array_eval(o));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, o));
    return v;
}

// ── Tests ──

const testing = std.testing;

test "smartResize matches the reference token-budget table" {
    // Values from `Lfm2VlImageProcessor.smart_resize` (transformers 5.15) at the
    // 3B pack's processor settings: patch 16, downsample 2, 64..256 tokens.
    const cases = [_]struct { h: u32, w: u32, th: u32, tw: u32 }{
        .{ .h = 64, .w = 64, .th = 256, .tw = 256 }, // grid 16x16, 64 tokens (the floor)
        .{ .h = 33, .w = 47, .th = 224, .tw = 320 }, // grid 14x20 — SHORTER than the stored 16 in h
        .{ .h = 224, .w = 224, .th = 256, .tw = 256 },
        .{ .h = 512, .w = 512, .th = 512, .tw = 512 }, // grid 32x32, 256 tokens (the ceiling)
        .{ .h = 300, .w = 1200, .th = 256, .tw = 1024 },
        .{ .h = 200, .w = 50, .th = 512, .tw = 128 },
        .{ .h = 101, .w = 97, .th = 288, .tw = 256 },
        // Sub-patch and extreme-aspect sources: the branch these take is the
        // one difference from Qwen's otherwise-identical resize.
        .{ .h = 1, .w = 1, .th = 256, .tw = 256 },
        .{ .h = 2000, .w = 17, .th = 2784, .tw = 32 }, // round(62.5) must be EVEN (62), not 63
        .{ .h = 17, .w = 2000, .th = 32, .tw = 2784 },
    };
    for (cases) |c| {
        const r = smartResize(c.h, c.w, 16, 2, 64, 256);
        testing.expectEqual(c.th, r.h) catch |e| {
            std.debug.print("smartResize({d}x{d}).h = {d}, want {d}\n", .{ c.h, c.w, r.h, c.th });
            return e;
        };
        testing.expectEqual(c.tw, r.w) catch |e| {
            std.debug.print("smartResize({d}x{d}).w = {d}, want {d}\n", .{ c.h, c.w, r.w, c.tw });
            return e;
        };
    }
}

test "isImageTooLarge and gridLayout match the reference split decisions" {
    const cases = [_]struct { h: u32, w: u32, large: bool, cols: u32, rows: u32 }{
        .{ .h = 768, .w = 1024, .large = true, .cols = 3, .rows = 2 },
        .{ .h = 1080, .w = 1920, .large = true, .cols = 4, .rows = 2 },
        .{ .h = 3000, .w = 4000, .large = true, .cols = 3, .rows = 2 },
        .{ .h = 4000, .w = 1000, .large = true, .cols = 1, .rows = 4 },
        // Just under the 2.0 tolerance — a photo this size is still ONE tile.
        .{ .h = 512, .w = 512, .large = false, .cols = 0, .rows = 0 },
        .{ .h = 600, .w = 600, .large = false, .cols = 0, .rows = 0 },
        .{ .h = 700, .w = 700, .large = false, .cols = 0, .rows = 0 },
        .{ .h = 1200, .w = 300, .large = false, .cols = 0, .rows = 0 },
    };
    for (cases) |c| {
        const large = isImageTooLarge(c.h, c.w, 16, 2, 256, 2.0);
        testing.expectEqual(c.large, large) catch |e| {
            std.debug.print("isImageTooLarge({d}x{d}) = {}, want {}\n", .{ c.h, c.w, large, c.large });
            return e;
        };
        if (!c.large) continue;
        const g = gridLayout(c.h, c.w, 2, 10, 512);
        testing.expectEqual(c.cols, g.cols) catch |e| {
            std.debug.print("gridLayout({d}x{d}).cols = {d}, want {d}\n", .{ c.h, c.w, g.cols, c.cols });
            return e;
        };
        testing.expectEqual(c.rows, g.rows) catch |e| {
            std.debug.print("gridLayout({d}x{d}).rows = {d}, want {d}\n", .{ c.h, c.w, g.rows, c.rows });
            return e;
        };
    }
}

test "buildPixelValues emits row-major patches with CHANNEL innermost" {
    // 2x1 patch grid at patch=2, so patch (0,0) covers x 0..1 and (0,1) x 2..3.
    // Values encode c*100 + y*10 + x so a transposed axis is unmistakable.
    const C: u32 = 3;
    const rh: u32 = 2;
    const rw: u32 = 4;
    const patch: u32 = 2;
    var img: [C * rh * rw]f32 = undefined;
    for (0..C) |c| for (0..rh) |y| for (0..rw) |x| {
        img[c * rh * rw + y * rw + x] = @floatFromInt(c * 100 + y * 10 + x);
    };

    var out: [2 * C * patch * patch]f32 = undefined;
    buildPixelValues(&out, &img, C, rh, rw, patch);

    // Patch 0, feature order [py, px, c]: (0,0,r) (0,0,g) (0,0,b) (0,1,r) ...
    const want0 = [_]f32{ 0, 100, 200, 1, 101, 201, 10, 110, 210, 11, 111, 211 };
    try testing.expectEqualSlices(f32, &want0, out[0..12]);
    // Patch 1 starts at x=2.
    const want1 = [_]f32{ 2, 102, 202, 3, 103, 203, 12, 112, 212, 13, 113, 213 };
    try testing.expectEqualSlices(f32, &want1, out[12..24]);

    // A tile is a WINDOW into the shared canvas, so reading the region at
    // (0, 2) must give patch 1's bytes exactly — same pixels, not a second
    // resample of that area. An off-by-one origin here shifts every tile.
    var tile: [C * patch * patch]f32 = undefined;
    buildPixelValuesRegion(&tile, &img, C, rh, rw, patch, 0, 2, patch, patch);
    try testing.expectEqualSlices(f32, &want1, &tile);
}
