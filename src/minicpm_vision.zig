//! MiniCPM-V 4.6 vision tower + processor math.
//!
//! Port of mlx-vlm `models/minicpmv4_6/` (`vision.py` tower + mergers,
//! `processing_minicpmv4_6.py` slicing). The tower is a stock pre-LN SigLIP
//! (patch-14, learned 70x70 position table, `gelu_pytorch_tanh` MLP) with one
//! family twist: at `insert_layer_id` the `vit_merger` folds each 2x2 patch
//! window into one token — per-window 4-token self-attention, then
//! Linear(4*hidden → 4*intermediate) → GELU(erf) → Linear(→ hidden) around a
//! window-mean residual. After the tower a second 2x2 merge (`merger`) projects
//! into language-model space, so 16 patches become one token ("16x";
//! `downsample_mode: "4x"` skips the vit_merger and halves the compression).
//!
//! Position ids are BUCKETED, not resampled: row i of an n-row grid takes
//! floor(i * side / n) of the stored `side`x`side` table — integer math, no
//! interpolation. This is the one NaFlex tower that never resamples weights.
//!
//! Feature layout matches LFM2-VL: `buildPixelValues` (in lfm2_vision.zig)
//! already emits the reference's per-patch `[py, px, c]` channel-innermost
//! order, so the patch embed is a plain Linear over the flattened patch.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;
const qwen_vision = @import("qwen_vision.zig");
const log = @import("log.zig");

pub const Resized = qwen_vision.Resized;

// ─────────────────────────────────────────────────────────────────────────────
// Processor math (processing_minicpmv4_6.py). All sizes operate on (w, h)
// PAIRS in PIL order; `patch` is 14 and the merge factor is patch*4 = 56 —
// every resize target is a multiple of 56 so both mergers divide evenly.
// ─────────────────────────────────────────────────────────────────────────────

/// `_ensure_divide`: snap to the nearest multiple (at least one). Python's
/// `round` is banker's — an exact .5 quotient rounds EVEN, and the reference
/// leans on that at the merge-factor boundary.
pub fn ensureDivide(length: u32, grid: u32) u32 {
    const q = qwen_vision.roundHalfEven(@as(f64, @floatFromInt(length)) / @as(f64, @floatFromInt(grid)));
    const snapped: u32 = @intFromFloat(q * @as(f64, @floatFromInt(grid)));
    return @max(snapped, grid);
}

fn ensureDivideF(length: f64, grid: f64) f64 {
    return @max(qwen_vision.roundHalfEven(length / grid) * grid, grid);
}

/// `_find_best_resize`: fit into `scale_resolution`² (upscaling only when
/// allowed), then snap both sides to a multiple of `patch * 4`.
pub fn findBestResize(width: u32, height: u32, scale_resolution: u32, patch: u32, allow_upscale: bool) Resized {
    var w: f64 = @floatFromInt(width);
    var h: f64 = @floatFromInt(height);
    const res: f64 = @floatFromInt(scale_resolution);
    if (w * h > res * res or allow_upscale) {
        const ratio = w / @max(h, 1.0);
        h = res / @sqrt(@max(ratio, 1e-6));
        w = h * ratio;
    }
    const merge_factor: f64 = @floatFromInt(patch * 4);
    return .{
        .w = @intFromFloat(ensureDivideF(w, merge_factor)),
        .h = @intFromFloat(ensureDivideF(h, merge_factor)),
    };
}

/// `get_sliced_grid`: how many slices an image splits into. `multiple` is the
/// area ratio rounded up (capped at `max_slice_nums`); candidate grids are the
/// factor pairs of `multiple - 1 .. multiple + 1`, and the grid whose aspect
/// best matches the image wins (ties → smallest factor first, the reference's
/// strict-< scan order). Null = serve the whole image as one view.
pub fn slicedGrid(width: u32, height: u32, max_slice_nums: u32, scale_resolution: u32) ?[2]u32 {
    const res: f64 = @floatFromInt(scale_resolution);
    const ratio = @as(f64, @floatFromInt(width)) * @as(f64, @floatFromInt(height)) / (res * res);
    const multiple: u32 = @min(@as(u32, @intFromFloat(@ceil(ratio))), max_slice_nums);
    if (multiple <= 1) return null;

    var best: ?[2]u32 = null;
    var min_error: f64 = std.math.inf(f64);
    const log_ratio = std.math.log(f64, std.math.e, @as(f64, @floatFromInt(width)) / @max(@as(f64, @floatFromInt(height)), 1.0));
    var grid_num = multiple -| 1;
    const upper = @min(multiple + 1, max_slice_nums);
    while (grid_num <= upper) : (grid_num += 1) {
        if (grid_num <= 1) continue;
        var factor: u32 = 1;
        while (factor <= grid_num) : (factor += 1) {
            if (grid_num % factor != 0) continue;
            const grid = [2]u32{ factor, grid_num / factor };
            const err = @abs(log_ratio - std.math.log(f64, std.math.e, @as(f64, @floatFromInt(grid[0])) / @as(f64, @floatFromInt(grid[1]))));
            if (err < min_error) {
                min_error = err;
                best = grid;
            }
        }
    }
    return best;
}

/// `_get_refine_size`: the per-slice canvas. Each cell is resized to ~
/// `scale_resolution`² (upscaling allowed) and snapped to the merge factor,
/// then multiplied back out — the canvas is resized ONCE and read cell by cell.
pub fn refineSize(width: u32, height: u32, cols: u32, rows: u32, scale_resolution: u32, patch: u32) Resized {
    const refine_w = ensureDivide(width, cols);
    const refine_h = ensureDivide(height, rows);
    const cell = findBestResize(refine_w / cols, refine_h / rows, scale_resolution, patch, true);
    return .{ .w = cell.w * cols, .h = cell.h * rows };
}

/// Per-view pixel budget sanity: a view's patch grid must be even on both
/// axes or the 2x2 mergers cannot tile it. Views come from `findBestResize` /
/// `refineSize`, which snap to a multiple of 56 px = 4 patches, so this only
/// fires on a hand-built caller.
pub fn gridIsMergeable(grid_h: u32, grid_w: u32) bool {
    return grid_h % 2 == 0 and grid_w % 2 == 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// ViT encoder with the mid-tower downsample. One view per call.
// forward(pixel_values [gh*gw, patch*patch*C], gh, gw) → [1, gh*gw/16, text_hidden].
// ─────────────────────────────────────────────────────────────────────────────

/// A tower linear. The MiniCPM-V pack quantizes only the language model, so
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

pub const MinicpmVision = struct {
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,

    hidden: u32, // 1152
    heads: u32, // 16 (tower AND vit_merger attention)
    head_dim: u32, // 72
    layers: u32, // 27
    insert_layer: u32, // 6
    use_vit_merger: bool,
    vit_intermediate: u32, // 17216
    out_hidden: u32, // trunk hidden, 1024
    ln_eps: f32,
    pos_side: u32, // 70 (from the stored table)

    patch: Lin, // conv weight flattened to [hidden, patch*patch*C]
    pos_table: mlx.mlx_array, // [side*side, hidden]
    blocks: []Block,
    post_ln_w: mlx.mlx_array,
    post_ln_b: mlx.mlx_array,
    // vit_merger: window attention + 2x2 fold, fires at `insert_layer`.
    vit_ln1_w: mlx.mlx_array,
    vit_ln1_b: mlx.mlx_array,
    vit_q: Lin,
    vit_k: Lin,
    vit_v: Lin,
    vit_out: Lin,
    vit_pre_norm_w: mlx.mlx_array,
    vit_pre_norm_b: mlx.mlx_array,
    vit_up: Lin, // [4*hidden → 4*intermediate]
    vit_down: Lin, // [4*intermediate → hidden]
    // merger: the final 2x2 fold into language-model space.
    mg_pre_norm_w: mlx.mlx_array,
    mg_pre_norm_b: mlx.mlx_array,
    mg_up: Lin, // [4*hidden → 4*hidden]
    mg_down: Lin, // [4*hidden → out_hidden]

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !MinicpmVision {
        const s = mlx.gpuStream();
        const mode = config.quant_mode.cstr();
        var name_buf: [256]u8 = undefined;
        var ctx = NameCtx{ .weights = weights, .buf = &name_buf, .mode = mode };

        // Probe before allocating anything: a pack quantized without the tower
        // opts out via MissingVisionWeights and serves text-only.
        _ = ctx.must("vision_tower.embeddings.patch_embedding.weight", .{}) catch {
            log.warn("MISSING MINICPM VISION WEIGHT: vision_tower.embeddings.patch_embedding.weight\n", .{});
            return error.MissingVisionWeights;
        };

        const hidden = config.vision_hidden_size;
        const patch = config.vision_patch_size;

        // The checkpoint stores the conv kernel [hidden, kh, kw, C]; flatten it
        // once to [hidden, patch*patch*C] so the Linear path applies. The
        // per-patch feature order ([py, px, c], channel innermost) matches the
        // kernel's row-major layout, so the reshape is the conv itself.
        var patch_2d = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(patch_2d);
        {
            const w = try ctx.must("vision_tower.embeddings.patch_embedding.weight", .{});
            const kside: c_int = @intCast(patch);
            const channels: c_int = 3;
            const shape = [_]c_int{ @intCast(hidden), kside, kside, channels };
            const flat = [_]c_int{ @intCast(hidden), kside * kside * channels };
            var shaped = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(shaped);
            try mlx.check(mlx.mlx_reshape(&shaped, w, &shape, 4, s));
            try mlx.check(mlx.mlx_reshape(&patch_2d, shaped, &flat, 2, s));
        }

        var blocks = try allocator.alloc(Block, config.vision_num_layers);
        errdefer allocator.free(blocks);
        for (0..config.vision_num_layers) |i| {
            blocks[i] = .{
                .ln1_w = try ctx.must("vision_tower.encoder.layers.{d}.layer_norm1.weight", .{i}),
                .ln1_b = try ctx.must("vision_tower.encoder.layers.{d}.layer_norm1.bias", .{i}),
                .ln2_w = try ctx.must("vision_tower.encoder.layers.{d}.layer_norm2.weight", .{i}),
                .ln2_b = try ctx.must("vision_tower.encoder.layers.{d}.layer_norm2.bias", .{i}),
                .q = try ctx.lin("vision_tower.encoder.layers.{d}.self_attn.q_proj", .{i}, hidden),
                .k = try ctx.lin("vision_tower.encoder.layers.{d}.self_attn.k_proj", .{i}, hidden),
                .v = try ctx.lin("vision_tower.encoder.layers.{d}.self_attn.v_proj", .{i}, hidden),
                .out = try ctx.lin("vision_tower.encoder.layers.{d}.self_attn.out_proj", .{i}, hidden),
                .fc1 = try ctx.lin("vision_tower.encoder.layers.{d}.mlp.fc1", .{i}, hidden),
                .fc2 = try ctx.lin("vision_tower.encoder.layers.{d}.mlp.fc2", .{i}, config.vision_intermediate_size),
            };
        }

        const pos_table = try ctx.must("vision_tower.embeddings.position_embedding.weight", .{});
        const rows: u32 = @intCast(mlx.getShape(pos_table)[0]);
        const side: u32 = std.math.sqrt(rows);
        if (side * side != rows) return error.InvalidVisionPositionTable;
        const vit_inter = if (config.cp_vit_intermediate > 0) config.cp_vit_intermediate else config.vision_intermediate_size * 4;

        log.info("Vision encoder: MiniCPM-V 4.6 SigLIP2 (depth={d}, hidden={d}, heads={d}, patch={d}, vit_merger@{d}, out_hidden={d})\n", .{
            config.vision_num_layers, hidden, config.vision_num_heads, patch, config.cp_insert_layer, config.hidden_size,
        });
        return .{
            .s = s,
            .allocator = allocator,
            .hidden = hidden,
            .heads = config.vision_num_heads,
            .head_dim = hidden / config.vision_num_heads,
            .layers = config.vision_num_layers,
            .insert_layer = config.cp_insert_layer,
            .use_vit_merger = config.cp_16x,
            .vit_intermediate = vit_inter,
            .out_hidden = config.hidden_size,
            .ln_eps = config.cp_ln_eps,
            .pos_side = side,
            .patch = .{ .w = patch_2d, .bias = try ctx.must("vision_tower.embeddings.patch_embedding.bias", .{}), .mode = mode },
            .pos_table = pos_table,
            .blocks = blocks,
            .post_ln_w = try ctx.must("vision_tower.post_layernorm.weight", .{}),
            .post_ln_b = try ctx.must("vision_tower.post_layernorm.bias", .{}),
            .vit_ln1_w = try ctx.must("vit_merger.layer_norm1.weight", .{}),
            .vit_ln1_b = try ctx.must("vit_merger.layer_norm1.bias", .{}),
            .vit_q = try ctx.lin("vit_merger.self_attn.q_proj", .{}, hidden),
            .vit_k = try ctx.lin("vit_merger.self_attn.k_proj", .{}, hidden),
            .vit_v = try ctx.lin("vit_merger.self_attn.v_proj", .{}, hidden),
            .vit_out = try ctx.lin("vit_merger.self_attn.out_proj", .{}, hidden),
            .vit_pre_norm_w = try ctx.must("vit_merger.pre_norm.weight", .{}),
            .vit_pre_norm_b = try ctx.must("vit_merger.pre_norm.bias", .{}),
            .vit_up = try ctx.lin("vit_merger.linear_1", .{}, hidden * 4),
            .vit_down = try ctx.lin("vit_merger.linear_2", .{}, vit_inter),
            .mg_pre_norm_w = try ctx.must("merger.mlp.0.pre_norm.weight", .{}),
            .mg_pre_norm_b = try ctx.must("merger.mlp.0.pre_norm.bias", .{}),
            .mg_up = try ctx.lin("merger.mlp.0.linear_1", .{}, hidden * 4),
            .mg_down = try ctx.lin("merger.mlp.0.linear_2", .{}, hidden * 4),
        };
    }

    pub fn deinit(self: *MinicpmVision) void {
        self.allocator.free(self.blocks);
        _ = mlx.mlx_array_free(self.patch.w);
    }

    /// Encode one view. `patches` is [gh*gw, patch*patch*C] in the reference's
    /// `[py, px, c]` order (lfm2_vision.buildPixelValues emits it); the result
    /// is [1, gh*gw/divisor, out_hidden], ready to splice at the image-token
    /// positions. `divisor` is 16 with the vit_merger armed, else 4.
    pub fn forward(self: *MinicpmVision, patches: mlx.mlx_array, grid_h_in: u32, grid_w_in: u32) !mlx.mlx_array {
        var grid_h = grid_h_in;
        var grid_w = grid_w_in;
        const hidden = try self.towerHidden(patches, &grid_h, &grid_w);
        defer _ = mlx.mlx_array_free(hidden);
        return self.project(hidden, grid_h, grid_w);
    }

    /// The tower alone: patch embed + bucketed positions + blocks (with the
    /// vit_merger fold at `insert_layer`) + post_layernorm. The grid shrinks
    /// through `grid_h`/`grid_w` when the vit_merger fires. Split from
    /// `forward` so a parity failure names the tower or the merger rather
    /// than "the vision path".
    pub fn towerHidden(self: *MinicpmVision, patches: mlx.mlx_array, grid_h: *u32, grid_w: *u32) !mlx.mlx_array {
        if (!gridIsMergeable(grid_h.*, grid_w.*)) return error.UnalignedPatchGrid;

        var x = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, patches, .bfloat16, self.s));

        replace(&x, try self.linear(x, self.patch));
        {
            const pos = try self.posEmbed(grid_h.*, grid_w.*);
            defer _ = mlx.mlx_array_free(pos);
            var sum = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&sum, x, pos, self.s));
            replace(&x, sum);
        }

        for (self.blocks, 0..) |blk, i| {
            {
                const normed = try self.layerNorm(x, blk.ln1_w, blk.ln1_b);
                defer _ = mlx.mlx_array_free(normed);
                const attn = try self.attention(normed, blk.q, blk.k, blk.v, blk.out);
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
                // The encoder MLP is `gelu_pytorch_tanh`; both merger MLPs
                // below are plain erf `gelu`. Two acts, one checkpoint.
                const act = try self.geluTanh(up);
                defer _ = mlx.mlx_array_free(act);
                const down = try self.linear(act, blk.fc2);
                defer _ = mlx.mlx_array_free(down);
                var h = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h, x, down, self.s));
                replace(&x, h);
            }
            // `downsample_mode: "16x"`: fold 2x2 windows into single tokens
            // mid-tower. The grid shrinks HERE, so later layers attend over
            // the merged sequence.
            if (self.use_vit_merger and i == self.insert_layer) {
                const merged = try self.vitMerger(x, grid_h, grid_w);
                _ = mlx.mlx_array_free(x);
                x = merged;
            }
        }
        replace(&x, try self.layerNorm(x, self.post_ln_w, self.post_ln_b));
        const out = x;
        x = mlx.mlx_array_new();
        return out;
    }

    /// `Merger` (the final 2x2 fold into language-model space) + a [1, tokens,
    /// out_hidden] reshape. [gh*gw, hidden] -> [1, gh*gw/4, out_hidden].
    fn project(self: *MinicpmVision, hidden: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        var x = try self.merger(hidden, grid_h, grid_w);
        errdefer _ = mlx.mlx_array_free(x);
        const tokens: c_int = @intCast((grid_h / 2) * (grid_w / 2));
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        const oshape = [_]c_int{ 1, tokens, @intCast(self.out_hidden) };
        try mlx.check(mlx.mlx_reshape(&out, x, &oshape, 3, self.s));
        _ = mlx.mlx_array_free(x);
        x = mlx.mlx_array_new();
        return out;
    }

    /// `VitMerger.__call__`: per 2x2 window of patches, run 4-token
    /// self-attention (residual), then Linear → GELU(erf) → Linear around a
    /// window-mean residual. [gh*gw, hidden] → [gh/2*gw/2, hidden]. The input
    /// `x_in` stays owned by the caller.
    fn vitMerger(self: *MinicpmVision, x_in: mlx.mlx_array, grid_h: *u32, grid_w: *u32) !mlx.mlx_array {
        const mgh = grid_h.* / 2;
        const mgw = grid_w.* / 2;

        var x = try self.mergeWindows(x_in, grid_h.*, grid_w.*, false); // [W, 4, hidden]
        errdefer _ = mlx.mlx_array_free(x);
        {
            const normed = try self.layerNorm(x, self.vit_ln1_w, self.vit_ln1_b);
            defer _ = mlx.mlx_array_free(normed);
            const attn = try self.attention(normed, self.vit_q, self.vit_k, self.vit_v, self.vit_out);
            defer _ = mlx.mlx_array_free(attn);
            var h = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h, x, attn, self.s));
            replace(&x, h);
        }

        // Residual is the window MEAN — not the first token, not the CLS.
        var residual = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(residual);
        {
            var sum = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sum);
            try mlx.check(mlx.mlx_sum_axis(&sum, x, 1, false, self.s));
            const quarter = bf16Scalar(0.25, self.s);
            defer _ = mlx.mlx_array_free(quarter);
            try mlx.check(mlx.mlx_multiply(&residual, sum, quarter, self.s));
        }

        {
            const d: c_int = @intCast(self.hidden);
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const shape = [_]c_int{ @intCast(mgh * mgw), 4 * d };
            try mlx.check(mlx.mlx_reshape(&flat, x, &shape, 2, self.s));
            replace(&x, try self.layerNorm(flat, self.vit_pre_norm_w, self.vit_pre_norm_b));
        }
        replace(&x, try self.linear(x, self.vit_up));
        replace(&x, try self.geluErf(x));
        replace(&x, try self.linear(x, self.vit_down));
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_add(&out, x, residual, self.s));
        // Success: hand `out` to the caller. Leave fresh empties in the two
        // owner handles so the errdefers above can never double-free a
        // wrapper that `out`'s graph still references.
        _ = mlx.mlx_array_free(x);
        x = mlx.mlx_array_new();
        _ = mlx.mlx_array_free(residual);
        residual = mlx.mlx_array_new();

        grid_h.* = mgh;
        grid_w.* = mgw;
        return out;
    }

    /// `Merger.__call__` (one round, kernel 2x2): fold 2x2 windows and project
    /// into language-model space. [gh*gw, hidden] → [gh/2*gw/2, out_hidden].
    /// The input `x_in` stays owned by the caller.
    fn merger(self: *MinicpmVision, x_in: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        var x = try self.mergeWindows(x_in, grid_h, grid_w, true); // [N/4, 4*hidden]
        errdefer _ = mlx.mlx_array_free(x);
        replace(&x, try self.layerNorm(x, self.mg_pre_norm_w, self.mg_pre_norm_b));
        replace(&x, try self.linear(x, self.mg_up));
        replace(&x, try self.geluErf(x));
        replace(&x, try self.linear(x, self.mg_down));
        const out = x;
        x = mlx.mlx_array_new();
        return out;
    }

    /// Reshape a [gh*gw, d] token map into 2x2 windows: [gh/2*gw/2, 4, d],
    /// flattened to [gh/2*gw/2, 4*d] when `flat`. Window tokens run [py, px] —
    /// the reference's transpose(0,2,1,3,4) order.
    fn mergeWindows(self: *MinicpmVision, x: mlx.mlx_array, grid_h: u32, grid_w: u32, flat: bool) !mlx.mlx_array {
        const d: c_int = @intCast(self.hidden);
        const mgh: c_int = @intCast(grid_h / 2);
        const mgw: c_int = @intCast(grid_w / 2);

        var cur = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(cur);
        {
            const shape = [_]c_int{ mgh, 2, mgw, 2, d };
            try mlx.check(mlx.mlx_reshape(&cur, x, &shape, 5, self.s));
        }
        const swap = [_]c_int{ 0, 2, 1, 3, 4 };
        {
            // `replace` moves the wrapper WITHOUT a retain, so `t` must go
            // out of scope un-freed once cur owns it — a defer-free here
            // would delete the wrapper cur points at.
            var t = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_transpose_axes(&t, cur, &swap, 5, self.s));
            replace(&cur, t);
        }
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        if (flat) {
            const shape = [_]c_int{ mgh * mgw, 4 * d };
            try mlx.check(mlx.mlx_reshape(&out, cur, &shape, 2, self.s));
        } else {
            const shape = [_]c_int{ mgh * mgw, 4, d };
            try mlx.check(mlx.mlx_reshape(&out, cur, &shape, 3, self.s));
        }
        _ = mlx.mlx_array_free(cur);
        return out;
    }

    /// Bucketed position embedding: row i of an n-row grid reads table row
    /// `floor(i * side / n)` — `SiglipVisionEmbeddings._build_position_buckets`
    /// with the boundaries folded into integer math. [gh*gw, hidden].
    fn posEmbed(self: *MinicpmVision, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const n: usize = grid_h * grid_w;
        const side = self.pos_side;
        const ids = try self.allocator.alloc(i32, n);
        defer self.allocator.free(ids);
        for (0..grid_h) |row| {
            const bh = row * side / grid_h;
            for (0..grid_w) |col| {
                const bw = col * side / grid_w;
                ids[row * grid_w + col] = @intCast(bh * side + bw);
            }
        }
        const shape = [_]c_int{@intCast(n)};
        const idx = mlx.mlx_array_new_data(ids.ptr, &shape, 1, .int32);
        defer _ = mlx.mlx_array_free(idx);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_take_axis(&out, self.pos_table, idx, 0, self.s));
        return out;
    }

    /// Full attention over one view's tokens (no mask — views encode alone),
    /// shared by the tower blocks and the vit_merger window attention. The
    /// input is [N, hidden] for a tower block or [W, 4, hidden] for the
    /// vit_merger (batch = W windows, sequence = 4 tokens per window) — the
    /// batch dim must be honored or the head reshape reads out of bounds.
    fn attention(self: *MinicpmVision, x: mlx.mlx_array, q_l: Lin, k_l: Lin, v_l: Lin, out_l: Lin) !mlx.mlx_array {
        const sh = mlx.getShape(x);
        const batched = sh.len > 2;
        const batch: c_int = if (batched) @intCast(sh[0]) else 1;
        const n: c_int = @intCast(sh[sh.len - 2]);
        const hd: c_int = @intCast(self.head_dim);
        const heads: c_int = @intCast(self.heads);

        var bhnd: [3]mlx.mlx_array = undefined;
        var built: usize = 0;
        // Frees the built handles on BOTH paths (defer evaluates `built` at
        // scope exit) — the SDPA output holds its own graph references.
        defer for (bhnd[0..built]) |arr| {
            _ = mlx.mlx_array_free(arr);
        };
        inline for (.{ q_l, k_l, v_l }, 0..) |l, i| {
            const flat = try self.linear(x, l);
            defer _ = mlx.mlx_array_free(flat);
            const shape = [_]c_int{ batch, n, heads, hd };
            var r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(r);
            try mlx.check(mlx.mlx_reshape(&r, flat, &shape, 4, self.s));
            const perm = [_]c_int{ 0, 2, 1, 3 };
            var b = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_transpose_axes(&b, r, &perm, 4, self.s));
            bhnd[i] = b;
            built += 1;
        }

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
        try mlx.check(mlx.mlx_reshape(&flat, t, &[_]c_int{ batch, n, @intCast(self.hidden) }, 3, self.s));
        if (batched) return self.linear(flat, out_l);
        // Tower blocks run batchless: squeeze back to [N, hidden] so the
        // residual add sees the input's rank. `flat` stays owned by the defer.
        var seq = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(seq);
        try mlx.check(mlx.mlx_reshape(&seq, flat, &[_]c_int{ n, @intCast(self.hidden) }, 2, self.s));
        return self.linear(seq, out_l);
    }

    fn layerNorm(self: *MinicpmVision, x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_layer_norm(&out, x, w, b, self.ln_eps, self.s));
        return out;
    }

    /// `gelu_pytorch_tanh` — the encoder MLP's activation.
    fn geluTanh(self: *MinicpmVision, x: mlx.mlx_array) !mlx.mlx_array {
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

    /// Exact erf `gelu` — both merger MLPs' activation.
    fn geluErf(self: *MinicpmVision, x: mlx.mlx_array) !mlx.mlx_array {
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

    fn linear(self: *MinicpmVision, x: mlx.mlx_array, l: Lin) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
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

/// Weight lookup. Handles are BORROWED from the weights map.
const NameCtx = struct {
    weights: *const Weights,
    buf: *[256]u8,
    mode: [*:0]const u8,

    fn key(self: NameCtx, comptime fmt: []const u8, args: anytype) []const u8 {
        return std.fmt.bufPrint(self.buf, fmt, args) catch unreachable;
    }

    fn opt(self: NameCtx, comptime fmt: []const u8, args: anytype) ?mlx.mlx_array {
        return self.weights.get(self.key(fmt, args));
    }

    fn must(self: NameCtx, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
        return self.opt(fmt, args) orelse {
            log.warn("MISSING MINICPM VISION WEIGHT: {s}\n", .{self.key(fmt, args)});
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

// ── Tests ──

const testing = std.testing;

test "ensureDivide and findBestResize match the reference resize table" {
    // Values from MiniCPMVImageProcessor at the pack's settings:
    // scale_resolution 448, patch 14 (merge factor 56).
    const cases = [_]struct { w: u32, h: u32, upscale: bool, tw: u32, th: u32 }{
        .{ .w = 448, .h = 448, .upscale = false, .tw = 448, .th = 448 },
        // Just over the budget: fit, then snap to 56.
        .{ .w = 800, .h = 600, .upscale = false, .tw = 504, .th = 392 },
        .{ .w = 1920, .h = 1080, .upscale = false, .tw = 616, .th = 336 },
        // Tiny images stay put without upscaling (still snapped UP to 56).
        .{ .w = 100, .h = 80, .upscale = false, .tw = 112, .th = 56 },
        .{ .w = 224, .h = 224, .upscale = true, .tw = 448, .th = 448 },
    };
    for (cases) |c| {
        const r = findBestResize(c.w, c.h, 448, 14, c.upscale);
        testing.expectEqual(c.tw, r.w) catch |e| {
            std.debug.print("findBestResize({d}x{d}).w = {d}, want {d}\n", .{ c.w, c.h, r.w, c.tw });
            return e;
        };
        testing.expectEqual(c.th, r.h) catch |e| {
            std.debug.print("findBestResize({d}x{d}).h = {d}, want {d}\n", .{ c.w, c.h, r.h, c.th });
            return e;
        };
    }
}

test "slicedGrid picks the aspect-matching factor pair" {
    // 800x600: area ratio ceil(800*600/448^2)=3 -> candidates {2,3,4}; the
    // factor pairs of 2/3/4 scan against log(4/3)=0.288: (2,2) is closest
    // (error 0.288 vs (2,1)'s 0.405).
    try testing.expectEqual([2]u32{ 2, 2 }, slicedGrid(800, 600, 9, 448).?);
    // 1920x1080 (16:9): area ratio ceil(1920*1080/448^2)=11 CAPS at
    // max_slice_nums 9 -> candidates {8,9}; log(16/9)=0.575 and (4,2)'s
    // log(2)=0.693 is the closest match (error 0.118 vs (9,1)'s 0.523) — MORE
    // columns than rows, the image is wider than tall.
    try testing.expectEqual([2]u32{ 4, 2 }, slicedGrid(1920, 1080, 9, 448).?);
    // Small image: no slicing.
    try testing.expectEqual(@as(?[2]u32, null), slicedGrid(300, 300, 9, 448));
    // Extremely wide: the grid goes flat before it goes tall.
    try testing.expectEqual([2]u32{ 4, 1 }, slicedGrid(2000, 300, 9, 448).?);
}

test "refineSize snaps the cell, then multiplies back out" {
    // 1920x1080 on the (4,2) grid slicedGrid picks: refine canvas (1920, 1080)
    // (both sides already multiples of the grid), cells (480, 540) resize
    // (upscale allowed) to a snapped (448, 448); the canvas is resized ONCE to
    // 448*4 x 448*2 and read cell by cell.
    const r = refineSize(1920, 1080, 4, 2, 448, 14);
    try testing.expectEqual(@as(u32, 1792), r.w);
    try testing.expectEqual(@as(u32, 896), r.h);
}

// Vision-tower live check against the real pack (mirrors the lfm2 live test):
//   MINICPM_VISION_MODEL=~/models/mlx-community/MiniCPM-V-4.6-4bit \
//   zig build test -Doptimize=ReleaseFast -Dtest-filter="minicpm vision live"
// Run FILTERED, never as part of an env-var-carrying full suite: the
// fault-injection sweeps share a process-global op counter, so concurrent
// GPU tests make every mlx.check see countdown leaks that are not theirs.
test "minicpm vision live: tower forward on the real pack" {
    const raw_model = std.c.getenv("MINICPM_VISION_MODEL") orelse return error.SkipZigTest;
    const model_dir = std.mem.sliceTo(raw_model, 0);
    if (model_dir.len == 0) return error.SkipZigTest;
    const a = testing.allocator;

    const config = try model_mod.parseConfig(std.testing.io, a, model_dir);
    var weights = try model_mod.loadWeightsWithVision(std.testing.io, a, model_dir);
    defer weights.deinit();

    var mv = try MinicpmVision.init(a, config, &weights);
    defer mv.deinit();

    // posEmbed buckets: every id in range, row-major.
    {
        const pos = try mv.posEmbed(32, 24);
        defer _ = mlx.mlx_array_free(pos);
        try mlx.check(mlx.mlx_array_eval(pos));
        const sh = mlx.getShape(pos);
        try testing.expectEqual(@as(i64, 32 * 24), sh[0]);
    }

    // Full forward, both grids a real request produces: the 448px source view
    // (32x32) and the slice cells of a 1024x1024 image (40x28). 32x32 ->
    // 16x16 -> 8x8 = 64 tokens; 40x28 -> 20x14 -> 10x7 = 70 tokens. Values
    // must be FINITE: a wrong reshape view keeps the shape but corrupts the
    // data (and segfaults later), so shape asserts alone prove nothing.
    for ([_][2]u32{ .{ 32, 32 }, .{ 40, 28 } }) |g| {
        const gh = g[0];
        const gw = g[1];
        const n = gh * gw;
        const feat = 3 * 14 * 14;
        const host = try a.alloc(f32, n * feat);
        defer a.free(host);
        for (host, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 251)) / 255.0 - 0.5;
        const shape = [_]c_int{ @intCast(n), @intCast(feat) };
        const pv = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(pv);

        // Two entries back to back: cross-entry state (a stale stream or a
        // freed handle) must not corrupt the second encode.
        for (0..2) |pass| {
            const out = try mv.forward(pv, gh, gw);
            defer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_array_eval(out));
            const sh = mlx.getShape(out);
            const want_tokens: i64 = @intCast((gh / 4) * (gw / 4));
            try testing.expectEqual(@as(i64, 1), sh[0]);
            try testing.expectEqual(want_tokens, sh[1]);
            try testing.expectEqual(@as(i64, @intCast(config.hidden_size)), sh[2]);
            var probe = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(probe);
            const start = [_]c_int{ 0, 0, 0 };
            const stop = [_]c_int{ 1, 2, @intCast(config.hidden_size) };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&probe, out, &start, 3, &stop, 3, &strides, 3, mv.s));
            try mlx.check(mlx.mlx_array_eval(probe));
            const data: [*]const f32 = @ptrCast(@alignCast(mlx.mlx_array_data_float32(probe)));
            var acc: f32 = 0;
            for (0..2 * @as(usize, @intCast(config.hidden_size))) |i| {
                acc += data[i];
                try testing.expect(std.math.isFinite(data[i]));
            }
            std.debug.print("[minicpm-vit] {d}x{d} pass {d}: rows sum {d:.4}\n", .{ gh, gw, pass, acc });
            try testing.expect(acc != 0);
        }
    }
}

// Tower parity vs the EXECUTED reference (tests/dump_minicpm_vision_fixtures.py
// runs mlx-vlm's own minicpmv4_6 modules on OUR pack's weights — the math the
// conversion was made with). A diff here is a layout/ordering bug in the port:
// the window order inside a 2x2 vit_merger block or a swapped fold residual
// keeps every SHAPE identical while corrupting the values, so the assertions
// are on values (cosine + rms ratio), never on shape alone.
//
//   MINICPM_VISION_MODEL=~/models/mlx-community/MiniCPM-V-4.6-4bit \
//   MINICPM_VISION_FIXTURE=/tmp/minicpm_vision_fixture.safetensors \
//   zig build test -Doptimize=ReleaseFast -Dtest-filter="minicpm vision parity"
// Filter-run only — same process-global fault-counter caveat as the live test.
test "minicpm vision parity: staged tower vs the executed reference" {
    const raw_model = std.c.getenv("MINICPM_VISION_MODEL") orelse return error.SkipZigTest;
    const raw_fix = std.c.getenv("MINICPM_VISION_FIXTURE") orelse return error.SkipZigTest;
    const model_dir = std.mem.sliceTo(raw_model, 0);
    const fix_path = std.mem.sliceTo(raw_fix, 0);
    if (model_dir.len == 0 or fix_path.len == 0) return error.SkipZigTest;
    const a = testing.allocator;

    const config = try model_mod.parseConfig(std.testing.io, a, model_dir);
    var weights = try model_mod.loadWeightsWithVision(std.testing.io, a, model_dir);
    defer weights.deinit();
    var fx = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer fx.deinit();

    var mv = try MinicpmVision.init(a, config, &weights);
    defer mv.deinit();

    for ([_][]const u8{ "a", "b", "c" }) |case| {
        var kb: [32]u8 = undefined;
        const pv = fx.get(try std.fmt.bufPrint(&kb, "{s}_patches", .{case})) orelse return error.MissingFixtureTensor;
        const grid_arr = fx.get(try std.fmt.bufPrint(&kb, "{s}_grid", .{case})) orelse return error.MissingFixtureTensor;
        try mlx.check(mlx.mlx_array_eval(grid_arr));
        const g: [*]const i32 = @ptrCast(@alignCast(mlx.mlx_array_data_int32(grid_arr)));
        var gh: u32 = @intCast(g[0] * 4); // fixture grid is POST-vit_merger
        var gw: u32 = @intCast(g[1] * 4);

        const hidden = try mv.towerHidden(pv, &gh, &gw);
        defer _ = mlx.mlx_array_free(hidden);
        try testing.expectEqual(@as(u32, @intCast(g[0])), gh);
        try testing.expectEqual(@as(u32, @intCast(g[1])), gw);
        {
            const want = fx.get(try std.fmt.bufPrint(&kb, "{s}_hidden", .{case})) orelse return error.MissingFixtureTensor;
            const c = try cosineSim(hidden, want, mv.s);
            const r = try rmsRatio(hidden, want, mv.s);
            std.debug.print("[minicpm-parity] {s} tower hidden cos={d:.6} rms_ratio={d:.4}\n", .{ case, c, r });
            try testing.expect(c > 0.999 and r > 0.995 and r < 1.005);
        }

        const feats = try mv.project(hidden, gh, gw);
        defer _ = mlx.mlx_array_free(feats);
        const want = fx.get(try std.fmt.bufPrint(&kb, "{s}_features", .{case})) orelse return error.MissingFixtureTensor;
        const c = try cosineSim(feats, want, mv.s);
        const r = try rmsRatio(feats, want, mv.s);
        std.debug.print("[minicpm-parity] {s} features cos={d:.6} rms_ratio={d:.4}\n", .{ case, c, r });
        // Magnitude too: these rows are spliced into the token stream, which
        // is exactly where a scale error hides from a cosine.
        try testing.expect(c > 0.99 and r > 0.98 and r < 1.02);
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

/// sum(a*b) in fp32 over flattened inputs. NaN propagates: `NaN > threshold`
/// is false, so an all-NaN candidate can never pass the comparisons above.
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
