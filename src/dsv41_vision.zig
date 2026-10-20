//! DeepSeek-V4.1-Flash vision tower + preprocessing, the release's own layout
//! (`inference/vision.py`, `inference/image_processor.py`; the MLX packs keep
//! its tensor names: `vision.*`, `aligner.*`, `image_start/end/newline`).
//!
//! A 32-block ViT with 2-D RoPE and full bidirectional attention over one
//! image's 14 px patches, then the aligner: a 3x3 unfold into the text width
//! through `w2(gelu(w1(x)))`. The trunk sees an image as
//!
//!     [start] + ([row of n_w aligner tokens] + [newline]) * n_h + [end]
//!
//! every position of which carries `image_token_id`, so `forward` returns the
//! WHOLE span (the three delimiters are learned embeddings) and the generic
//! splice lands it row by row.
//!
//! The processor letterboxes (Pillow `ImageOps.pad`, bicubic, gray 127) into an
//! aspect-preserving canvas whose token grid fits `max_image_tokens`; a tiny
//! image is first scaled up to `min_pixels`.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;
const qwen_vision = @import("qwen_vision.zig");
const glm_vision = @import("glm_vision.zig");
const transformer = @import("transformer.zig");
const log = @import("log.zig");

const Lin = glm_vision.Lin;
const linear = glm_vision.linear;
const replace = glm_vision.replace;

/// `image_processor.plan_image_grid` + `load_image`: the canvas in pixels, the ViT patch grid and the
/// aligner (LLM token) grid, and whether the image is stretched into the canvas (the `max_wh_ratio` arm)
/// rather than letterboxed.
pub const Plan = struct { h: u32, w: u32, vit_h: u32, vit_w: u32, llm_h: u32, llm_w: u32, stretch: bool };

/// LLM tokens one image costs: its aligner grid, a newline per row, start and end.
pub fn spanTokens(llm_h: u32, llm_w: u32) usize {
    return @as(usize, llm_h) * (@as(usize, llm_w) + 1) + 2;
}

fn ceilDiv(a: u64, b: u64) u64 {
    return (a + b - 1) / b;
}

fn llmGrid(best_h: u64, best_w: u64, patch: u64, ds: u64) [2]u64 {
    return .{ ceilDiv(best_h / patch, ds), ceilDiv(best_w / patch, ds) };
}

/// `plan_image_grid` for an image of `width` x `height` pixels. Null only for a degenerate config.
pub fn planGrid(width: u32, height: u32, patch: u32, ds: u32, max_tokens: u32, min_pixels: u32, max_wh_ratio: u32) ?Plan {
    if (width == 0 or height == 0 or patch == 0 or ds == 0 or max_tokens < 4) return null;
    const p: u64 = patch;
    var w: u64 = width;
    var h: u64 = height;
    const stretch = max_wh_ratio != 0 and w >= @as(u64, max_wh_ratio) * h;
    if (max_wh_ratio != 0 and w > @as(u64, max_wh_ratio) * h) w = @as(u64, max_wh_ratio) * h;
    if (w * h < min_pixels) {
        const ratio = @sqrt(@as(f64, @floatFromInt(min_pixels)) / @as(f64, @floatFromInt(w * h)));
        w = @intFromFloat(@trunc(@as(f64, @floatFromInt(w)) * ratio));
        h = @intFromFloat(@trunc(@as(f64, @floatFromInt(h)) * ratio));
        if (w == 0 or h == 0) return null;
    }
    var best_w = ceilDiv(w, p) * p;
    var best_h = ceilDiv(h, p) * p;
    var grid = llmGrid(best_h, best_w, p, ds);
    if (spanTokens(@intCast(grid[0]), @intCast(grid[1])) > max_tokens) {
        // `solve_resize_ratio`: the largest aspect-preserving size whose grid fits the budget.
        const r = @as(f64, @floatFromInt(h)) / @as(f64, @floatFromInt(w));
        const max_w_f = @sqrt(@as(f64, @floatFromInt(max_tokens - 2)) / r + 0.25) - 0.5;
        const max_h_f = max_w_f * r;
        const cell = p * ds;
        if (max_w_f < 1.0) {
            best_h = (@as(u64, max_tokens) - 2) / 2 * cell;
            best_w = cell;
        } else if (max_h_f < 1.0) {
            best_h = cell;
            best_w = (@as(u64, max_tokens) - 3) * cell;
        } else {
            const cf: f64 = @floatFromInt(cell);
            const beta = @min(@floor(max_w_f) * cf / @as(f64, @floatFromInt(w)), @floor(max_h_f) * cf / @as(f64, @floatFromInt(h)));
            best_h = @as(u64, @intFromFloat(@floor(@as(f64, @floatFromInt(h)) * beta / @as(f64, @floatFromInt(p))))) * p;
            best_w = @as(u64, @intFromFloat(@floor(@as(f64, @floatFromInt(w)) * beta / @as(f64, @floatFromInt(p))))) * p;
        }
        grid = llmGrid(best_h, best_w, p, ds);
        if (best_h == 0 or best_w == 0 or spanTokens(@intCast(grid[0]), @intCast(grid[1])) > max_tokens) return null;
    }
    return .{
        .h = @intCast(best_h),
        .w = @intCast(best_w),
        .vit_h = @intCast(best_h / p),
        .vit_w = @intCast(best_w / p),
        .llm_h = @intCast(grid[0]),
        .llm_w = @intCast(grid[1]),
        .stretch = stretch,
    };
}

/// Python's `round`: half to even.
fn pyRound(x: f64) u32 {
    const f = @floor(x);
    const d = x - f;
    const r: f64 = if (d > 0.5) f + 1 else if (d < 0.5) f else if (@mod(f, 2.0) == 0) f else f + 1;
    return @intFromFloat(r);
}

/// Pillow `ImageOps.pad(image, (w, h), BICUBIC, 127)`: `contain` picks the resized size, the rest is
/// the fill, centered (half-to-even on an odd margin).
pub const Letterbox = struct { rw: u32, rh: u32, x: u32, y: u32 };

pub fn letterbox(src_w: u32, src_h: u32, w: u32, h: u32) Letterbox {
    const im_ratio = @as(f64, @floatFromInt(src_w)) / @as(f64, @floatFromInt(src_h));
    const dest_ratio = @as(f64, @floatFromInt(w)) / @as(f64, @floatFromInt(h));
    var rw = w;
    var rh = h;
    if (im_ratio != dest_ratio) {
        if (im_ratio > dest_ratio) {
            rh = pyRound(@as(f64, @floatFromInt(src_h)) / @as(f64, @floatFromInt(src_w)) * @as(f64, @floatFromInt(w)));
        } else {
            rw = pyRound(@as(f64, @floatFromInt(src_w)) / @as(f64, @floatFromInt(src_h)) * @as(f64, @floatFromInt(h)));
        }
    }
    return .{
        .rw = rw,
        .rh = rh,
        .x = if (rw != w) pyRound(@as(f64, @floatFromInt(w - rw)) * 0.5) else 0,
        .y = if (rw == w and rh != h) pyRound(@as(f64, @floatFromInt(h - rh)) * 0.5) else 0,
    };
}

/// `(x / 255 - 0.5) / 0.5`, the processor's normalization.
fn normalizePixel(v: u8) f32 {
    return (@as(f32, @floatFromInt(v)) / 255.0 - 0.5) / 0.5;
}

/// One image's ViT patches: f32 bytes of `[vit_h*vit_w, 3*patch*patch]` in `(c, py, px)` order, row-major
/// over the grid. `grid_h/grid_w` is the ViT grid; the aligner grid follows from it.
pub const Pixels = struct { bytes: []u8, h: u32, w: u32, grid_h: u32, grid_w: u32 };

/// `load_image` for `rgb` (HWC u8): plan, letterbox (or stretch), normalize, patchify. Caller owns `bytes`.
pub fn preprocess(allocator: std.mem.Allocator, rgb: []const u8, src_h: u32, src_w: u32, patch: u32, ds: u32, max_tokens: u32, min_pixels: u32, max_wh_ratio: u32) !Pixels {
    const plan = planGrid(src_w, src_h, patch, ds, max_tokens, min_pixels, max_wh_ratio) orelse return error.InvalidImageDimensions;
    const box: Letterbox = if (plan.stretch) .{ .rw = plan.w, .rh = plan.h, .x = 0, .y = 0 } else letterbox(src_w, src_h, plan.w, plan.h);
    const resized: ?[]u8 = if (box.rw == src_w and box.rh == src_h)
        null
    else
        try qwen_vision.resizeInterleavedPil(allocator, rgb, src_h, src_w, box.rh, box.rw, 3, .bicubic);
    defer if (resized) |r| allocator.free(r);
    const content = resized orelse rgb;

    const n: usize = @as(usize, plan.vit_h) * plan.vit_w;
    const feat: usize = 3 * @as(usize, patch) * patch;
    const bytes = try allocator.alloc(u8, n * feat * 4);
    const out = @as([*]f32, @ptrCast(@alignCast(bytes.ptr)))[0 .. n * feat];
    const fill = normalizePixel(127);
    for (0..plan.vit_h) |vh| for (0..plan.vit_w) |vw| {
        const row = out[(vh * plan.vit_w + vw) * feat ..][0..feat];
        for (0..3) |c| for (0..patch) |py| for (0..patch) |px| {
            const y = vh * patch + py;
            const x = vw * patch + px;
            const inside = y >= box.y and y < box.y + box.rh and x >= box.x and x < box.x + box.rw;
            row[(c * patch + py) * patch + px] = if (inside) normalizePixel(content[((y - box.y) * box.rw + (x - box.x)) * 3 + c]) else fill;
        };
    };
    return .{ .bytes = bytes, .h = plan.h, .w = plan.w, .grid_h = plan.vit_h, .grid_w = plan.vit_w };
}

const Block = struct {
    norm1: mlx.mlx_array,
    norm2: mlx.mlx_array,
    wqkv: Lin,
    wo: Lin,
    w1: Lin, // [2*inter, hidden]: gate rows then up rows
    w2: Lin,
};

pub const Dsv41Vision = struct {
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,
    hidden: u32,
    heads: u32,
    head_dim: u32,
    ds: u32,
    out_hidden: u32,
    rope_theta: f64,
    eps: f32 = 1e-6,

    patch: Lin,
    blocks: []Block,
    norm: mlx.mlx_array,
    a_w1: Lin,
    a_w2: Lin,
    image_start: mlx.mlx_array,
    image_end: mlx.mlx_array,
    image_newline: mlx.mlx_array,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !Dsv41Vision {
        const ctx = NameCtx{ .weights = weights };
        if (ctx.get("vision.patch_embed.proj.weight", .{}) == null) {
            log.warn("MISSING DEEPSEEK VISION WEIGHT: vision.patch_embed.proj.weight\n", .{});
            return error.MissingVisionWeights;
        }
        const hidden = config.vision_hidden_size;
        const inter = config.vision_intermediate_size;
        const blocks = try allocator.alloc(Block, config.vision_num_layers);
        errdefer allocator.free(blocks);
        for (blocks, 0..) |*b, i| b.* = .{
            .norm1 = try ctx.must("vision.blocks.{d}.norm1.weight", .{i}),
            .norm2 = try ctx.must("vision.blocks.{d}.norm2.weight", .{i}),
            .wqkv = try ctx.lin("vision.blocks.{d}.attn.wqkv", .{i}, hidden),
            .wo = try ctx.lin("vision.blocks.{d}.attn.wo", .{i}, hidden),
            // The MLX packs carry convert.py's `ffn`; packs in the release's own names keep vision.py's `mlp`.
            .w1 = ctx.lin("vision.blocks.{d}.ffn.w1", .{i}, hidden) catch try ctx.lin("vision.blocks.{d}.mlp.w1", .{i}, hidden),
            .w2 = ctx.lin("vision.blocks.{d}.ffn.w2", .{i}, inter) catch try ctx.lin("vision.blocks.{d}.mlp.w2", .{i}, inter),
        };
        var self = Dsv41Vision{
            .s = mlx.gpuStream(),
            .allocator = allocator,
            .hidden = hidden,
            .heads = config.vision_num_heads,
            .head_dim = hidden / config.vision_num_heads,
            .ds = config.dv_downsample,
            .out_hidden = config.hidden_size,
            .rope_theta = config.vision_rope_theta,
            .patch = try ctx.lin("vision.patch_embed.proj", .{}, 3 * config.vision_patch_size * config.vision_patch_size),
            .blocks = blocks,
            .norm = try ctx.must("vision.norm.weight", .{}),
            .a_w1 = try ctx.lin("aligner.w1", .{}, hidden * config.dv_downsample * config.dv_downsample),
            .a_w2 = try ctx.lin("aligner.w2", .{}, config.hidden_size),
            .image_start = try ctx.must("image_start", .{}),
            .image_end = try ctx.must("image_end", .{}),
            .image_newline = try ctx.must("image_newline", .{}),
        };
        // Resident now, so the load is billed for the tower rather than the first image.
        try self.evalWeights();
        log.info("Vision encoder: DeepSeek-V4.1 ViT (depth={d}, hidden={d}, heads={d}, patch={d}, downsample={d})\n", .{
            config.vision_num_layers, hidden, self.heads, config.vision_patch_size, self.ds,
        });
        return self;
    }

    fn evalWeights(self: *Dsv41Vision) !void {
        var list = std.ArrayList(mlx.mlx_array).empty;
        defer list.deinit(self.allocator);
        try list.appendSlice(self.allocator, &.{ self.norm, self.image_start, self.image_end, self.image_newline });
        for ([_]Lin{ self.patch, self.a_w1, self.a_w2 }) |l| try glm_vision.appendLin(self.allocator, &list, l);
        for (self.blocks) |b| {
            try list.appendSlice(self.allocator, &.{ b.norm1, b.norm2 });
            for ([_]Lin{ b.wqkv, b.wo, b.w1, b.w2 }) |l| try glm_vision.appendLin(self.allocator, &list, l);
        }
        const vec = mlx.mlx_vector_array_new_data(list.items.ptr, list.items.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
    }

    pub fn deinit(self: *Dsv41Vision) void {
        self.allocator.free(self.blocks);
    }

    /// Encode one image: `patches` is `[gh*gw, 3*p*p]`. Returns the trunk's whole span, `[1, spanTokens, out_hidden]`.
    pub fn forward(self: *Dsv41Vision, patches: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const hidden = try self.towerHidden(patches, grid_h, grid_w);
        defer _ = mlx.mlx_array_free(hidden);
        const feats = try self.aligner(hidden, grid_h, grid_w);
        defer _ = mlx.mlx_array_free(feats);
        return self.span(feats, grid_h, grid_w);
    }

    /// The ViT through its final norm: `[gh*gw, hidden]`.
    pub fn towerHidden(self: *Dsv41Vision, patches: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const n: c_int = @intCast(grid_h * grid_w);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, patches, .bfloat16, self.s));
        replace(&x, try linear(x, self.patch, self.s));

        // The release's 2-D rope: row angles in the first half of each rotate_half pair, column angles in the
        // second, which is `fillVisionRope` at merge 1 (plain row-major tokens).
        const rope = try qwen_vision.visionRopeTables(self.allocator, grid_h, grid_w, 1, self.head_dim, self.rope_theta);
        defer _ = mlx.mlx_array_free(rope.cos);
        defer _ = mlx.mlx_array_free(rope.sin);

        var dt = mlx.DtypeTrace.begin("dsv41-vision", x, if (self.blocks.len > 0) self.blocks[0].wqkv.w else null);
        for (self.blocks, 0..) |*b, i| {
            {
                const normed = try self.rmsNorm(x, b.norm1);
                defer _ = mlx.mlx_array_free(normed);
                const attn = try self.attention(normed, b, rope.cos, rope.sin, n);
                defer _ = mlx.mlx_array_free(attn);
                var h = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h, x, attn, self.s));
                replace(&x, h);
            }
            {
                const normed = try self.rmsNorm(x, b.norm2);
                defer _ = mlx.mlx_array_free(normed);
                const mlp = try self.swiglu(normed, b);
                defer _ = mlx.mlx_array_free(mlp);
                var h = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h, x, mlp, self.s));
                replace(&x, h);
            }
            dt.layer(x, i);
        }
        dt.end(x);
        return self.rmsNorm(x, self.norm);
    }

    /// `Aligner.forward`: the grid padded (zeros) to multiples of `ds`, every `ds x ds` block unfolded
    /// channel-major into one row, then `w2(gelu(w1(x)))` -> `[llm_h*llm_w, out_hidden]`.
    pub fn aligner(self: *Dsv41Vision, hidden: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const r: c_int = @intCast(self.ds);
        const c: c_int = @intCast(self.hidden);
        const gh: c_int = @intCast(grid_h);
        const gw: c_int = @intCast(grid_w);
        const lh: c_int = @intCast(ceilDiv(grid_h, self.ds));
        const lw: c_int = @intCast(ceilDiv(grid_w, self.ds));
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_reshape(&x, hidden, &[_]c_int{ gh, gw, c }, 3, self.s));
        if (lh * r != gh or lw * r != gw) {
            const zero = mlx.mlx_array_new_float(0.0);
            defer _ = mlx.mlx_array_free(zero);
            var padded = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_pad(&padded, x, &[_]c_int{ 0, 1 }, 2, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ lh * r - gh, lw * r - gw }, 2, zero, "constant", self.s));
            replace(&x, padded);
        }
        // [lh, r, lw, r, C] -> [lh, lw, C, r, r]: torch's unfold orders a block's features (c, ky, kx).
        try mlx.check(mlx.mlx_reshape(&x, x, &[_]c_int{ lh, r, lw, r, c }, 5, self.s));
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_transpose_axes(&t, x, &[_]c_int{ 0, 2, 4, 1, 3 }, 5, self.s));
        var rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rows);
        try mlx.check(mlx.mlx_reshape(&rows, t, &[_]c_int{ lh * lw, c * r * r }, 2, self.s));
        var h1 = try linear(rows, self.a_w1, self.s);
        defer _ = mlx.mlx_array_free(h1);
        replace(&h1, try glm_vision.geluErf(h1, self.s));
        return linear(h1, self.a_w2, self.s);
    }

    /// The trunk's span: start, each aligner row followed by newline, end -> `[1, spanTokens, out_hidden]`.
    fn span(self: *Dsv41Vision, feats: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const lh: c_int = @intCast(ceilDiv(grid_h, self.ds));
        const lw: c_int = @intCast(ceilDiv(grid_w, self.ds));
        const d: c_int = @intCast(self.out_hidden);
        const dt = mlx.mlx_array_dtype(feats);
        var grid = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(grid);
        try mlx.check(mlx.mlx_reshape(&grid, feats, &[_]c_int{ lh, lw, d }, 3, self.s));
        var nl = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(nl);
        try mlx.check(mlx.mlx_astype(&nl, self.image_newline, dt, self.s));
        try mlx.check(mlx.mlx_reshape(&nl, nl, &[_]c_int{ 1, 1, d }, 3, self.s));
        try mlx.check(mlx.mlx_broadcast_to(&nl, nl, &[_]c_int{ lh, 1, d }, 3, self.s));
        var rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rows);
        {
            const parts = [_]mlx.mlx_array{ grid, nl };
            const vec = mlx.mlx_vector_array_new_data(&parts, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&rows, vec, 1, self.s));
        }
        try mlx.check(mlx.mlx_reshape(&rows, rows, &[_]c_int{ lh * (lw + 1), d }, 2, self.s));
        var start = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(start);
        try mlx.check(mlx.mlx_astype(&start, self.image_start, dt, self.s));
        try mlx.check(mlx.mlx_reshape(&start, start, &[_]c_int{ 1, d }, 2, self.s));
        var end = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(end);
        try mlx.check(mlx.mlx_astype(&end, self.image_end, dt, self.s));
        try mlx.check(mlx.mlx_reshape(&end, end, &[_]c_int{ 1, d }, 2, self.s));
        var all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(all);
        {
            const parts = [_]mlx.mlx_array{ start, rows, end };
            const vec = mlx.mlx_vector_array_new_data(&parts, 3);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&all, vec, 0, self.s));
        }
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_expand_dims(&out, all, 0, self.s));
        return out;
    }

    /// `w2(silu(gate) * up)` with gate and up the two halves of `w1`'s output.
    fn swiglu(self: *Dsv41Vision, x: mlx.mlx_array, b: *const Block) !mlx.mlx_array {
        const gu = try linear(x, b.w1, self.s);
        defer _ = mlx.mlx_array_free(gu);
        const shape = mlx.getShape(gu);
        const inter = @divExact(shape[1], 2);
        var gate = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate);
        try mlx.check(mlx.mlx_slice(&gate, gu, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ shape[0], inter }, 2, &[_]c_int{ 1, 1 }, 2, self.s));
        var up = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up);
        try mlx.check(mlx.mlx_slice(&up, gu, &[_]c_int{ 0, inter }, 2, &[_]c_int{ shape[0], shape[1] }, 2, &[_]c_int{ 1, 1 }, 2, self.s));
        var sg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sg);
        try mlx.check(mlx.mlx_sigmoid(&sg, gate, self.s));
        var act = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act);
        try mlx.check(mlx.mlx_multiply(&act, gate, sg, self.s));
        try mlx.check(mlx.mlx_multiply(&act, act, up, self.s));
        return linear(act, b.w2, self.s);
    }

    /// Full bidirectional attention over one image's patches; q and k roped in f32.
    fn attention(self: *Dsv41Vision, x: mlx.mlx_array, b: *const Block, cos: mlx.mlx_array, sin: mlx.mlx_array, n: c_int) !mlx.mlx_array {
        const heads: c_int = @intCast(self.heads);
        const hd: c_int = @intCast(self.head_dim);
        const qkv = try linear(x, b.wqkv, self.s);
        defer _ = mlx.mlx_array_free(qkv);
        var r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r);
        try mlx.check(mlx.mlx_reshape(&r, qkv, &[_]c_int{ n, 3, heads, hd }, 4, self.s));
        var bhnd: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (bhnd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        inline for (0..3) |j| {
            var part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(part);
            try mlx.check(mlx.mlx_slice(&part, r, &[_]c_int{ 0, j, 0, 0 }, 4, &[_]c_int{ n, j + 1, heads, hd }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, self.s));
            try mlx.check(mlx.mlx_reshape(&part, part, &[_]c_int{ n, heads, hd }, 3, self.s));
            if (j < 2) replace(&part, try glm_vision.rope2d(part, cos, sin, self.s));
            var t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(t);
            try mlx.check(mlx.mlx_transpose_axes(&t, part, &[_]c_int{ 1, 0, 2 }, 3, self.s));
            try mlx.check(mlx.mlx_expand_dims(&bhnd[j], t, 0, self.s));
        }
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const none = mlx.mlx_array{ .ctx = null };
        var ctx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ctx);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ctx, bhnd[0], bhnd[1], bhnd[2], scale, "", none, none, false, self.s));
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_transpose_axes(&t, ctx, &[_]c_int{ 0, 2, 1, 3 }, 4, self.s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, t, &[_]c_int{ n, @intCast(self.hidden) }, 2, self.s));
        return linear(flat, b.wo, self.s);
    }

    fn rmsNorm(self: *Dsv41Vision, x: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, self.eps, self.s));
        return out;
    }
};

/// Weight lookup by the release's names; handles are BORROWED from the weights map.
const NameCtx = struct {
    weights: *const Weights,

    fn get(self: NameCtx, comptime fmt: []const u8, args: anytype) ?mlx.mlx_array {
        var buf: [128]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, fmt, args) catch return null;
        return self.weights.get(key);
    }

    fn must(self: NameCtx, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
        return self.get(fmt, args) orelse {
            log.warn("MISSING DEEPSEEK VISION WEIGHT: " ++ fmt ++ "\n", args);
            return error.MissingVisionWeights;
        };
    }

    fn lin(self: NameCtx, comptime fmt: []const u8, args: anytype, in_dim: u32) !Lin {
        var l = Lin{ .w = try self.must(fmt ++ ".weight", args), .bias = self.get(fmt ++ ".bias", args) orelse .{ .ctx = null } };
        if (self.get(fmt ++ ".scales", args)) |sc| {
            l.scales = sc;
            l.biases = self.get(fmt ++ ".biases", args) orelse .{ .ctx = null };
            l.qp = transformer.quantParamsFromGeometry(l.w, sc, l.biases.ctx != null, in_dim) orelse {
                log.warn("DEEPSEEK VISION: unreadable quantization on " ++ fmt ++ "\n", args);
                return error.InvalidVisionWeights;
            };
        }
        return l;
    }
};

// ── Tests ──

const testing = std.testing;

test "dsv41 vision: the resize plan and letterbox match the release's image processor" {
    // Rows from inference/image_processor.py (plan_image_grid, load_image) and Pillow's ImageOps.pad at
    // the release config (patch 14, downsample 3, 1024 tokens, min 544x544 pixels, no wh cap).
    const cases = [_]struct { w: u32, h: u32, bw: u32, bh: u32, lh: u32, lw: u32, rw: u32, rh: u32, x: u32, y: u32 }{
        .{ .w = 730, .h = 487, .bw = 742, .bh = 490, .lh = 12, .lw = 18, .rw = 734, .rh = 490, .x = 4, .y = 0 }, // fits: aligned up, letterboxed
        .{ .w = 97, .h = 61, .bw = 686, .bh = 434, .lh = 11, .lw = 17, .rw = 686, .rh = 431, .x = 0, .y = 2 }, // under min_pixels: scaled up
        .{ .w = 7000, .h = 1000, .bw = 3528, .bh = 504, .lh = 12, .lw = 84, .rw = 3528, .rh = 504, .x = 0, .y = 0 }, // over the budget
        .{ .w = 4000, .h = 3000, .bw = 1512, .bh = 1134, .lh = 27, .lw = 36, .rw = 1512, .rh = 1134, .x = 0, .y = 0 },
        .{ .w = 14, .h = 14, .bw = 546, .bh = 546, .lh = 13, .lw = 13, .rw = 546, .rh = 546, .x = 0, .y = 0 },
        .{ .w = 1, .h = 1, .bw = 546, .bh = 546, .lh = 13, .lw = 13, .rw = 546, .rh = 546, .x = 0, .y = 0 },
        .{ .w = 5000, .h = 50, .bw = 5446, .bh = 56, .lh = 2, .lw = 130, .rw = 5446, .rh = 54, .x = 0, .y = 1 },
        .{ .w = 50, .h = 5000, .bw = 56, .bh = 5446, .lh = 130, .lw = 2, .rw = 54, .rh = 5446, .x = 1, .y = 0 },
        .{ .w = 1920, .h = 1080, .bw = 1708, .bh = 966, .lh = 23, .lw = 41, .rw = 1708, .rh = 961, .x = 0, .y = 2 },
        .{ .w = 640, .h = 640, .bw = 644, .bh = 644, .lh = 16, .lw = 16, .rw = 644, .rh = 644, .x = 0, .y = 0 },
        .{ .w = 1000, .h = 7000, .bw = 462, .bh = 3234, .lh = 77, .lw = 11, .rw = 462, .rh = 3234, .x = 0, .y = 0 },
        .{ .w = 300, .h = 2900, .bw = 308, .bh = 2912, .lh = 70, .lw = 8, .rw = 301, .rh = 2912, .x = 4, .y = 0 },
    };
    for (cases) |c| {
        const p = planGrid(c.w, c.h, 14, 3, 1024, 295936, 0).?;
        const box = letterbox(c.w, c.h, p.w, p.h);
        testing.expectEqual(Plan{ .h = c.bh, .w = c.bw, .vit_h = c.bh / 14, .vit_w = c.bw / 14, .llm_h = c.lh, .llm_w = c.lw, .stretch = false }, p) catch |e| {
            std.debug.print("{d}x{d}\n", .{ c.w, c.h });
            return e;
        };
        testing.expectEqual(Letterbox{ .rw = c.rw, .rh = c.rh, .x = c.x, .y = c.y }, box) catch |e| {
            std.debug.print("{d}x{d} letterbox\n", .{ c.w, c.h });
            return e;
        };
        try testing.expect(spanTokens(p.llm_h, p.llm_w) <= 1024);
    }
    // A wh cap stretches a wider image into the capped canvas instead of letterboxing it.
    const wide = planGrid(4000, 1000, 14, 3, 1024, 295936, 2).?;
    try testing.expectEqual(Plan{ .h = 924, .w = 1848, .vit_h = 66, .vit_w = 132, .llm_h = 22, .llm_w = 44, .stretch = true }, wide);
    try testing.expect(planGrid(0, 10, 14, 3, 1024, 0, 0) == null);
}

test "dsv41 vision: patches hold the letterboxed image in (c, py, px) order and gray elsewhere" {
    const a = testing.allocator;
    // A 28x14 image already at the canvas size with min_pixels off: patch 0 is the left half, patch 1 the right.
    var rgb: [28 * 14 * 3]u8 = undefined;
    for (0..14) |y| for (0..28) |x| {
        const v: u8 = if (x < 14) 255 else 0;
        rgb[(y * 28 + x) * 3 + 0] = v;
        rgb[(y * 28 + x) * 3 + 1] = 51;
        rgb[(y * 28 + x) * 3 + 2] = v;
    };
    const px = try preprocess(a, &rgb, 14, 28, 14, 3, 1024, 0, 0);
    defer a.free(px.bytes);
    try testing.expectEqual(@as(u32, 1), px.grid_h);
    try testing.expectEqual(@as(u32, 2), px.grid_w);
    const f = @as([*]const f32, @ptrCast(@alignCast(px.bytes.ptr)))[0 .. px.bytes.len / 4];
    try testing.expectEqual(@as(usize, 2 * 588), f.len);
    try testing.expectApproxEqAbs(@as(f32, 1.0), f[0], 1e-6); // patch 0, R
    try testing.expectApproxEqAbs(@as(f32, 51.0 / 255.0 * 2.0 - 1.0), f[196], 1e-6); // patch 0, G plane
    try testing.expectApproxEqAbs(@as(f32, -1.0), f[588], 1e-6); // patch 1, R

    // A 28x7 image lands in a 28x14 canvas centered (rows 4..10): the rows above and below are the gray fill.
    const small = try preprocess(a, rgb[0 .. 28 * 7 * 3], 7, 28, 14, 3, 1024, 0, 0);
    defer a.free(small.bytes);
    const g = @as([*]const f32, @ptrCast(@alignCast(small.bytes.ptr)))[0 .. small.bytes.len / 4];
    try testing.expectEqual(@as(u32, 2), small.grid_w);
    try testing.expectApproxEqAbs(normalizePixel(127), g[0], 1e-6); // row 0 of patch 0: fill
    try testing.expectApproxEqAbs(@as(f32, 1.0), g[4 * 14], 1e-6); // row 4: content
    try testing.expectApproxEqAbs(normalizePixel(127), g[11 * 14], 1e-6); // row 11: fill
}

// Parity vs the EXECUTED reference: the release's vision.py + image_processor.py (fp32, CPU) on the
// pack's dense tower weights (tests/dump_dsv41_vision_fixtures.py).
//
//   DSV41_VISION_MODEL=/path/to/pack DSV41_VISION_FIXTURE=~/claude-tmp/dsv41-vision/fixture.safetensors \
//   zig build test -Doptimize=ReleaseFast -Dtest-filter="dsv41 vision"
fn fixtureEnv() ?[]const u8 {
    const raw = std.c.getenv("DSV41_VISION_FIXTURE") orelse return null;
    const path = std.mem.sliceTo(raw, 0);
    return if (path.len == 0) null else path;
}

test "dsv41 vision: preprocessing equals the reference processor's patches" {
    const fix_path = fixtureEnv() orelse return error.SkipZigTest;
    const a = testing.allocator;
    var fx = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer fx.deinit();
    for ([_][]const u8{ "house", "noise", "wide" }) |case| {
        var kb: [64]u8 = undefined;
        const rgb_arr = fx.get(try std.fmt.bufPrint(&kb, "{s}_rgb", .{case})) orelse return error.MissingFixtureTensor;
        try mlx.check(mlx.mlx_array_eval(rgb_arr));
        const shape = mlx.getShape(rgb_arr);
        const h: u32 = @intCast(shape[0]);
        const w: u32 = @intCast(shape[1]);
        const rgb = @as([*]const u8, @ptrCast(mlx.mlx_array_data_uint8(rgb_arr)))[0 .. @as(usize, h) * w * 3];
        const px = try preprocess(a, rgb, h, w, 14, 3, 1024, 295936, 0);
        defer a.free(px.bytes);
        const grid = fx.get(try std.fmt.bufPrint(&kb, "{s}_grid", .{case})) orelse return error.MissingFixtureTensor;
        try mlx.check(mlx.mlx_array_eval(grid));
        const g: [*]const i32 = @ptrCast(@alignCast(mlx.mlx_array_data_int32(grid)));
        try testing.expectEqual(@as(u32, @intCast(g[0])), px.grid_h);
        try testing.expectEqual(@as(u32, @intCast(g[1])), px.grid_w);
        const want_arr = fx.get(try std.fmt.bufPrint(&kb, "{s}_patches", .{case})) orelse return error.MissingFixtureTensor;
        try mlx.check(mlx.mlx_array_eval(want_arr));
        const got = @as([*]const f32, @ptrCast(@alignCast(px.bytes.ptr)))[0 .. px.bytes.len / 4];
        try testing.expectEqual(mlx.mlx_array_size(want_arr), got.len);
        const want = @as([*]const f32, @ptrCast(@alignCast(mlx.mlx_array_data_float32(want_arr))))[0..got.len];
        var worst: f32 = 0;
        for (got, want) |x, y| worst = @max(worst, @abs(x - y));
        std.debug.print("[dsv41-vision] {s} {d}x{d} -> grid {d}x{d} patches max|diff|={e}\n", .{ case, w, h, px.grid_h, px.grid_w, worst });
        // The reference's patches are bf16; Pillow's resize is reproduced exactly.
        try testing.expect(worst < 1e-2);
    }
}

test "dsv41 vision: tower, aligner and span match the reference" {
    const fix_path = fixtureEnv() orelse return error.SkipZigTest;
    const raw_model = std.c.getenv("DSV41_VISION_MODEL") orelse return error.SkipZigTest;
    const model_dir = std.mem.sliceTo(raw_model, 0);
    const a = testing.allocator;
    var config = try model_mod.parseConfig(testing.io, a, model_dir);
    defer config.deinit(a);
    var weights = try model_mod.loadWeightsWithVision(testing.io, a, model_dir);
    defer weights.deinit();
    var fx = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer fx.deinit();
    var dv = try Dsv41Vision.init(a, config, &weights);
    defer dv.deinit();
    for ([_][]const u8{ "noise", "house" }) |case| {
        var kb: [64]u8 = undefined;
        const grid = fx.get(try std.fmt.bufPrint(&kb, "{s}_grid", .{case})) orelse return error.MissingFixtureTensor;
        try mlx.check(mlx.mlx_array_eval(grid));
        const g: [*]const i32 = @ptrCast(@alignCast(mlx.mlx_array_data_int32(grid)));
        const gh: u32 = @intCast(g[0]);
        const gw: u32 = @intCast(g[1]);
        const pv = fx.get(try std.fmt.bufPrint(&kb, "{s}_patches", .{case})) orelse return error.MissingFixtureTensor;
        const hidden = try dv.towerHidden(pv, gh, gw);
        defer _ = mlx.mlx_array_free(hidden);
        const feats = try dv.aligner(hidden, gh, gw);
        defer _ = mlx.mlx_array_free(feats);
        const whole = try dv.forward(pv, gh, gw);
        defer _ = mlx.mlx_array_free(whole);
        const Stage = struct { name: []const u8, got: mlx.mlx_array };
        for ([_]Stage{ .{ .name = "tower", .got = hidden }, .{ .name = "features", .got = feats }, .{ .name = "span", .got = whole } }) |stage| {
            const want = fx.get(try std.fmt.bufPrint(&kb, "{s}_{s}", .{ case, stage.name })) orelse return error.MissingFixtureTensor;
            const c = try glm_vision.cosineSim(stage.got, want, dv.s);
            const r = try glm_vision.rmsRatio(stage.got, want, dv.s);
            std.debug.print("[dsv41-vision] {s} grid {d}x{d} {s} cos={d:.6} rms_ratio={d:.4}\n", .{ case, gh, gw, stage.name, c, r });
            // bf16 serving against an fp32 reference through 32 blocks; the span's scale is what the trunk reads.
            try testing.expect(c > 0.99 and r > 0.98 and r < 1.02);
        }
        try testing.expectEqual(@as(c_int, @intCast(spanTokens(@intCast(ceilDiv(gh, 3)), @intCast(ceilDiv(gw, 3))))), mlx.getShape(whole)[1]);
    }
}
