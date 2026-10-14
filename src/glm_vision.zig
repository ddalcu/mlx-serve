//! GLM-5.3-Flash (glm5_next) vision tower + preprocessing.
//!
//! Port of transformers' `Glm5NextVisionModel` and `Glm5NextImageProcessorPil`;
//! mlx-vlm's `glm5_next/vision.py` is the same model in the packs' layout. A
//! Qwen2-VL-shaped ViT (Conv3d patch embed, 2-D rotary, RMSNorm with per-head
//! q/k norms, clamped SwiGLU), then a 2x2 Conv2d downsample into the text width
//! and a LayerNorm -> GELU -> clamped-SwiGLU merger. No learned position table,
//! no window attention, no deepstack. Its rows splice at `<|image|>` with plain
//! 1-D positions: the text trunk is NoPE.
//!
//! The processor never distorts: the image is scaled into a canvas aligned to
//! `patch * merge` under a token budget, and the rest of the canvas is padded
//! black BEFORE normalization.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;
const qwen_vision = @import("qwen_vision.zig");
const transformer = @import("transformer.zig");
const log = @import("log.zig");

/// OPENAI_CLIP_MEAN / OPENAI_CLIP_STD, the processor's normalization.
const MEAN = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
const STD = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };
/// The vision rotary's base when `vision_config` names none (no GLM pack does).
const ROPE_THETA = 10000.0;

/// The aligned canvas and the box the resized image fills in its top-left corner.
pub const Geometry = struct { h: u32, w: u32, content_h: u32, content_w: u32 };

fn alignUp(v: u64, f: u64) u64 {
    return (v + f - 1) / f * f;
}

fn floorU64(x: f64) u64 {
    return @intFromFloat(@floor(x));
}

/// `smart_resize` plus the processor's content box (`Glm5NextImageProcessorPil.resize`).
/// A still image counts as `tps` frames; token bounds are MERGED tokens. Null when
/// the budget cannot hold one aligned patch (the reference raises).
pub fn resizeGeometry(height: u32, width: u32, factor: u32, tps: u32, min_tokens: u32, max_tokens: u32) ?Geometry {
    if (height == 0 or width == 0 or factor == 0 or tps == 0) return null;
    const f: u64 = factor;
    const frames: u64 = tps;
    const h: u64 = height;
    const w: u64 = width;
    const per_token = frames * f * f;
    const min_px = @as(u64, min_tokens) * per_token;
    const max_px = @as(u64, max_tokens) * per_token;

    var th = alignUp(h, f);
    var tw = alignUp(w, f);
    if (frames * th * tw < min_px) {
        const scale = @sqrt(@as(f64, @floatFromInt(min_px)) / @as(f64, @floatFromInt(frames * h * w)));
        th = alignUp(@max(1, @as(u64, @intFromFloat(@ceil(@as(f64, @floatFromInt(h)) * scale)))), f);
        tw = alignUp(@max(1, @as(u64, @intFromFloat(@ceil(@as(f64, @floatFromInt(w)) * scale)))), f);
    }
    if (frames * th * tw > max_px) {
        if (max_px < frames * f * f) return null;
        // The tallest content height whose aligned box fits the budget.
        var lo: u64 = 1;
        var hi: u64 = h;
        th = f;
        tw = f;
        while (lo <= hi) {
            const ch = (lo + hi) / 2;
            const cw = @max(1, floorU64(@as(f64, @floatFromInt(w * ch)) / @as(f64, @floatFromInt(h))));
            const ah = alignUp(ch, f);
            const aw = alignUp(cw, f);
            if (frames * ah * aw <= max_px) {
                th = ah;
                tw = aw;
                lo = ch + 1;
            } else {
                hi = ch - 1;
            }
        }
    }

    const fh: f64 = @floatFromInt(h);
    const fw: f64 = @floatFromInt(w);
    var scale = @min(@as(f64, @floatFromInt(th)) / fh, @as(f64, @floatFromInt(tw)) / fw);
    if (frames * h * w >= per_token * min_tokens) scale = @min(1.0, scale);
    return .{
        .h = @intCast(th),
        .w = @intCast(tw),
        .content_h = @intCast(@max(1, @min(th, floorU64(fh * scale)))),
        .content_w = @intCast(@max(1, @min(tw, floorU64(fw * scale)))),
    };
}

/// `rescale` (f64, then f32) followed by `normalize` (f32), as the processor does.
fn normalizePixel(v: u8, c: usize) f32 {
    const x: f32 = @floatCast(@as(f64, @floatFromInt(v)) * (1.0 / 255.0));
    return (x - MEAN[c]) / STD[c];
}

/// The processor's `pixel_values` for one image: f32 bytes of `[grid_h*grid_w, 3*tps*patch*patch]`
/// in merge-block order, owned by the caller.
pub const Pixels = struct { bytes: []u8, h: u32, w: u32, grid_h: u32, grid_w: u32 };

/// Resize `rgb` (HWC u8) into its content box with Pillow's bicubic, pad the canvas black,
/// normalize, and patchify.
pub fn preprocess(allocator: std.mem.Allocator, rgb: []const u8, src_h: u32, src_w: u32, patch: u32, tps: u32, merge: u32, min_tokens: u32, max_tokens: u32) !Pixels {
    const g = resizeGeometry(src_h, src_w, patch * merge, tps, min_tokens, max_tokens) orelse return error.InvalidImageDimensions;
    const resized: ?[]u8 = if (g.content_h == src_h and g.content_w == src_w)
        null
    else
        try qwen_vision.resizeInterleavedPil(allocator, rgb, src_h, src_w, g.content_h, g.content_w, 3, .bicubic);
    defer if (resized) |r| allocator.free(r);
    const content = resized orelse rgb;

    const plane: usize = @as(usize, g.h) * g.w;
    const chw = try allocator.alloc(f32, 3 * plane);
    defer allocator.free(chw);
    for (0..3) |c| {
        const pad = normalizePixel(0, c);
        for (0..g.h) |y| for (0..g.w) |x| {
            chw[c * plane + y * g.w + x] = if (y < g.content_h and x < g.content_w)
                normalizePixel(content[(y * g.content_w + x) * 3 + c], c)
            else
                pad;
        };
    }

    const gh = g.h / patch;
    const gw = g.w / patch;
    const values: usize = @as(usize, gh) * gw * 3 * tps * patch * patch;
    const bytes = try allocator.alloc(u8, values * 4);
    const out = @as([*]f32, @ptrCast(@alignCast(bytes.ptr)))[0..values];
    qwen_vision.buildPixelValues(out, chw, 3, g.h, g.w, patch, tps, merge);
    return .{ .bytes = bytes, .h = g.h, .w = g.w, .grid_h = gh, .grid_w = gw };
}

/// A linear: dense `w [out, in]`, or quantized with its params solved from geometry.
pub const Lin = struct {
    w: mlx.mlx_array,
    scales: mlx.mlx_array = .{ .ctx = null },
    biases: mlx.mlx_array = .{ .ctx = null },
    bias: mlx.mlx_array = .{ .ctx = null },
    qp: ?transformer.QuantParams = null,
};

const Block = struct {
    norm1: mlx.mlx_array,
    norm2: mlx.mlx_array,
    qkv: Lin,
    proj: Lin,
    q_norm: mlx.mlx_array,
    k_norm: mlx.mlx_array,
    gate: Lin,
    up: Lin,
    down: Lin,
};

pub const GlmVision = struct {
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,

    hidden: u32,
    heads: u32,
    head_dim: u32,
    merge: u32,
    out_hidden: u32,
    eps: f32,
    limit: f32,

    /// Both convs flattened at load to the linear they are at stride == kernel: the
    /// patch embed to `[hidden, C*T*p*p]` in pixel_values' `[C, T, py, px]` order, the
    /// downsample to `[out, merge*merge*hidden]` in merge-block `(row, col, channel)` order.
    patch_w: mlx.mlx_array,
    patch_b: mlx.mlx_array,
    blocks: []Block,
    post_norm: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_b: mlx.mlx_array,
    m_proj: Lin,
    m_norm_w: mlx.mlx_array,
    m_norm_b: mlx.mlx_array,
    m_gate: Lin,
    m_up: Lin,
    m_down: Lin,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !GlmVision {
        const s = mlx.gpuStream();
        const prefix = qwen_vision.resolveVisionPrefix(weights) orelse {
            log.warn("MISSING GLM VISION WEIGHT: vision_tower.patch_embed.proj.weight\n", .{});
            return error.MissingVisionWeights;
        };
        const ctx = NameCtx{ .weights = weights, .prefix = prefix };
        const hidden = config.qv_hidden;
        const inter = config.qv_intermediate;
        const merge = config.qv_merge;
        const out_hidden = config.qv_out_hidden;

        var flat = std.ArrayList(mlx.mlx_array).empty;
        defer flat.deinit(allocator);
        errdefer for (flat.items) |a| {
            _ = mlx.mlx_array_free(a);
        };

        const conv3d = try ctx.must("patch_embed.proj.weight", .{});
        const cshape = mlx.getShape(conv3d);
        const layout = qwen_vision.patchProjLayout(cshape, config.qv_temporal_patch, config.qv_patch) orelse return error.InvalidVisionWeights;
        var patch_w = mlx.mlx_array_new();
        try flat.append(allocator, patch_w);
        {
            var t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(t);
            const perm: []const c_int = if (layout == .channels_last) &.{ 0, 4, 1, 2, 3 } else &.{ 0, 1, 2, 3, 4 };
            try mlx.check(mlx.mlx_transpose_axes(&t, conv3d, perm.ptr, 5, s));
            try mlx.check(mlx.mlx_reshape(&patch_w, t, &[_]c_int{ cshape[0], cshape[1] * cshape[2] * cshape[3] * cshape[4] }, 2, s));
            flat.items[0] = patch_w;
        }

        // Conv2d stored channels-last by MLX (`[out, kh, kw, C]`), channels-first by torch.
        const conv2d = try ctx.must("downsample.weight", .{});
        const dshape = mlx.getShape(conv2d);
        const m: c_int = @intCast(merge);
        const hd_c: c_int = @intCast(hidden);
        const channels_last = dshape.len == 4 and dshape[1] == m and dshape[2] == m and dshape[3] == hd_c;
        if (!channels_last and !(dshape.len == 4 and dshape[1] == hd_c and dshape[2] == m and dshape[3] == m)) return error.InvalidVisionWeights;
        var down_w = mlx.mlx_array_new();
        try flat.append(allocator, down_w);
        {
            var t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(t);
            const perm: []const c_int = if (channels_last) &.{ 0, 1, 2, 3 } else &.{ 0, 2, 3, 1 };
            try mlx.check(mlx.mlx_transpose_axes(&t, conv2d, perm.ptr, 4, s));
            try mlx.check(mlx.mlx_reshape(&down_w, t, &[_]c_int{ dshape[0], m * m * hd_c }, 2, s));
            flat.items[1] = down_w;
        }

        const blocks = try allocator.alloc(Block, config.qv_depth);
        errdefer allocator.free(blocks);
        for (blocks, 0..) |*b, i| b.* = .{
            .norm1 = try ctx.must("blocks.{d}.norm1.weight", .{i}),
            .norm2 = try ctx.must("blocks.{d}.norm2.weight", .{i}),
            .qkv = try ctx.lin("blocks.{d}.attn.qkv", .{i}, hidden),
            .proj = try ctx.lin("blocks.{d}.attn.proj", .{i}, hidden),
            .q_norm = try ctx.must("blocks.{d}.attn.q_norm.weight", .{i}),
            .k_norm = try ctx.must("blocks.{d}.attn.k_norm.weight", .{i}),
            .gate = try ctx.lin("blocks.{d}.mlp.gate_proj", .{i}, hidden),
            .up = try ctx.lin("blocks.{d}.mlp.up_proj", .{i}, hidden),
            .down = try ctx.lin("blocks.{d}.mlp.down_proj", .{i}, inter),
        };

        var self = GlmVision{
            .s = s,
            .allocator = allocator,
            .hidden = hidden,
            .heads = config.qv_heads,
            .head_dim = config.qv_head_dim,
            .merge = merge,
            .out_hidden = out_hidden,
            .eps = config.glmv_eps,
            .limit = config.glmv_swiglu_limit,
            .patch_w = patch_w,
            .patch_b = try ctx.must("patch_embed.proj.bias", .{}),
            .blocks = blocks,
            .post_norm = try ctx.must("post_layernorm.weight", .{}),
            .down_w = down_w,
            .down_b = try ctx.must("downsample.bias", .{}),
            .m_proj = try ctx.lin("merger.proj", .{}, out_hidden),
            .m_norm_w = try ctx.must("merger.post_projection_norm.weight", .{}),
            .m_norm_b = try ctx.must("merger.post_projection_norm.bias", .{}),
            .m_gate = try ctx.lin("merger.gate_proj", .{}, out_hidden),
            .m_up = try ctx.lin("merger.up_proj", .{}, out_hidden),
            .m_down = try ctx.lin("merger.down_proj", .{}, config.glmv_proj_intermediate),
        };
        // Resident now, so the load is billed for the tower rather than the first image.
        try self.evalWeights();
        flat.clearRetainingCapacity();
        log.info("Vision encoder: GLM-5.3-Flash ViT (depth={d}, hidden={d}, heads={d}, patch={d}, merge={d}, out_hidden={d})\n", .{
            config.qv_depth, hidden, config.qv_heads, config.qv_patch, merge, out_hidden,
        });
        return self;
    }

    fn evalWeights(self: *GlmVision) !void {
        var list = std.ArrayList(mlx.mlx_array).empty;
        defer list.deinit(self.allocator);
        try list.appendSlice(self.allocator, &.{ self.patch_w, self.patch_b, self.post_norm, self.down_w, self.down_b, self.m_norm_w, self.m_norm_b });
        const lins = [_]Lin{ self.m_proj, self.m_gate, self.m_up, self.m_down };
        for (lins) |l| try appendLin(self.allocator, &list, l);
        for (self.blocks) |b| {
            try list.appendSlice(self.allocator, &.{ b.norm1, b.norm2, b.q_norm, b.k_norm });
            for ([_]Lin{ b.qkv, b.proj, b.gate, b.up, b.down }) |l| try appendLin(self.allocator, &list, l);
        }
        const vec = mlx.mlx_vector_array_new_data(list.items.ptr, list.items.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
    }

    pub fn deinit(self: *GlmVision) void {
        _ = mlx.mlx_array_free(self.patch_w);
        _ = mlx.mlx_array_free(self.down_w);
        self.allocator.free(self.blocks);
    }

    /// Encode one image: `patches` is pixel_values `[gh*gw, C*T*p*p]` in merge-block
    /// order. Returns `[1, gh*gw/merge², out_hidden]`.
    pub fn forward(self: *GlmVision, patches: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const hidden = try self.towerHidden(patches, grid_h, grid_w);
        defer _ = mlx.mlx_array_free(hidden);
        const down = try self.downsample(hidden);
        defer _ = mlx.mlx_array_free(down);
        const feats = try self.merger(down);
        defer _ = mlx.mlx_array_free(feats);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_expand_dims(&out, feats, 0, self.s));
        return out;
    }

    /// The blocks through `post_layernorm` -> `[gh*gw, hidden]`.
    pub fn towerHidden(self: *GlmVision, patches: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        if (grid_h % self.merge != 0 or grid_w % self.merge != 0) return error.UnalignedPatchGrid;
        const n: c_int = @intCast(grid_h * grid_w);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, patches, .bfloat16, self.s));
        replace(&x, try self.dense(x, self.patch_w, self.patch_b));

        const rope = try qwen_vision.visionRopeTables(self.allocator, grid_h, grid_w, self.merge, self.head_dim, ROPE_THETA);
        defer _ = mlx.mlx_array_free(rope.cos);
        defer _ = mlx.mlx_array_free(rope.sin);

        var dt = mlx.DtypeTrace.begin("glm-vision", x, if (self.blocks.len > 0) self.blocks[0].qkv.w else null);
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
                const mlp = try self.swiglu(normed, b.gate, b.up, b.down);
                defer _ = mlx.mlx_array_free(mlp);
                var h = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h, x, mlp, self.s));
                replace(&x, h);
            }
            dt.layer(x, i);
        }
        dt.end(x);
        return self.rmsNorm(x, self.post_norm);
    }

    /// Each merge-block's `merge²` consecutive rows through the 2x2 Conv2d -> `[N/merge², out_hidden]`.
    fn downsample(self: *GlmVision, hidden: mlx.mlx_array) !mlx.mlx_array {
        const rows: c_int = @intCast(@divExact(mlx.getShape(hidden)[0], @as(c_int, @intCast(self.merge * self.merge))));
        var grouped = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(grouped);
        try mlx.check(mlx.mlx_reshape(&grouped, hidden, &[_]c_int{ rows, @intCast(self.merge * self.merge * self.hidden) }, 2, self.s));
        return self.dense(grouped, self.down_w, self.down_b);
    }

    fn merger(self: *GlmVision, x_in: mlx.mlx_array) !mlx.mlx_array {
        var x = try linear(x_in, self.m_proj, self.s);
        defer _ = mlx.mlx_array_free(x);
        {
            var t = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_fast_layer_norm(&t, x, self.m_norm_w, self.m_norm_b, 1e-5, self.s));
            replace(&x, t);
        }
        replace(&x, try geluErf(x, self.s));
        return self.swiglu(x, self.m_gate, self.m_up, self.m_down);
    }

    /// `down(silu(min(gate, limit)) * clip(up, ±limit))`.
    fn swiglu(self: *GlmVision, x: mlx.mlx_array, gate: Lin, up: Lin, down: Lin) !mlx.mlx_array {
        const g = try linear(x, gate, self.s);
        defer _ = mlx.mlx_array_free(g);
        const u = try linear(x, up, self.s);
        defer _ = mlx.mlx_array_free(u);
        const act = (try transformer.fusedClampedSwiGLU(self.s, g, u, self.limit)) orelse
            try transformer.clampedSwiGLU(self.s, g, u, self.limit);
        defer _ = mlx.mlx_array_free(act);
        return linear(act, down, self.s);
    }

    /// Full bidirectional attention over one image's patches.
    fn attention(self: *GlmVision, x: mlx.mlx_array, b: *const Block, cos: mlx.mlx_array, sin: mlx.mlx_array, n: c_int) !mlx.mlx_array {
        const heads: c_int = @intCast(self.heads);
        const hd: c_int = @intCast(self.head_dim);
        const qkv = try linear(x, b.qkv, self.s);
        defer _ = mlx.mlx_array_free(qkv);
        var r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r);
        try mlx.check(mlx.mlx_reshape(&r, qkv, &[_]c_int{ n, 3, heads, hd }, 4, self.s));

        // [N, 3, heads, hd] -> q, k, v as [1, heads, N, hd]; q/k normed per head, then roped.
        var bhnd: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (bhnd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        inline for (0..3) |j| {
            var part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(part);
            try mlx.check(mlx.mlx_slice(&part, r, &[_]c_int{ 0, j, 0, 0 }, 4, &[_]c_int{ n, j + 1, heads, hd }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, self.s));
            try mlx.check(mlx.mlx_reshape(&part, part, &[_]c_int{ n, heads, hd }, 3, self.s));
            if (j < 2) {
                const normed = try self.rmsNorm(part, if (j == 0) b.q_norm else b.k_norm);
                defer _ = mlx.mlx_array_free(normed);
                replace(&part, try rope2d(normed, cos, sin, self.s));
            }
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
        return linear(flat, b.proj, self.s);
    }

    fn rmsNorm(self: *GlmVision, x: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, self.eps, self.s));
        return out;
    }

    fn dense(self: *GlmVision, x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array) !mlx.mlx_array {
        return linear(x, .{ .w = w, .bias = bias }, self.s);
    }
};

/// A dense or quantized `x @ w.T`, plus the bias when the linear has one.
pub fn linear(x: mlx.mlx_array, l: Lin, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    if (l.qp) |qp| {
        try mlx.check(mlx.mlx_quantized_matmul(&out, x, l.w, l.scales, l.biases, true, mlx.mlx_optional_int.some(@intCast(qp.group_size)), mlx.mlx_optional_int.some(@intCast(qp.bits)), qp.mode.cstr(), s));
    } else {
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        try mlx.check(mlx.mlx_transpose(&wt, l.w, s));
        try mlx.check(mlx.mlx_matmul(&out, x, wt, s));
    }
    if (l.bias.ctx != null) {
        var biased = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&biased, out, l.bias, s));
        replace(&out, biased);
    }
    return out;
}

/// `x*cos + rotate_half(x)*sin` in f32, back in `x`'s dtype. `x` is `[N, heads, hd]`, tables `[N, 1, hd]`.
fn rope2d(x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(x);
    const half = @divExact(shape[2], 2);
    var x32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x32);
    try mlx.check(mlx.mlx_astype(&x32, x, .float32, s));
    var x1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x1);
    try mlx.check(mlx.mlx_slice(&x1, x32, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ shape[0], shape[1], half }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_slice(&x2, x32, &[_]c_int{ 0, 0, half }, 3, &[_]c_int{ shape[0], shape[1], shape[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    try mlx.check(mlx.mlx_negative(&x2, x2, s));
    var rot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rot);
    {
        const parts = [_]mlx.mlx_array{ x2, x1 };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&rot, vec, -1, s));
    }
    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_multiply(&a, x32, cos, s));
    try mlx.check(mlx.mlx_multiply(&rot, rot, sin, s));
    try mlx.check(mlx.mlx_add(&a, a, rot, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, a, mlx.mlx_array_dtype(x), s));
    return out;
}

/// Exact `gelu`: `x * (1 + erf(x / sqrt(2))) / 2`.
pub fn geluErf(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(x);
    const inv_sqrt2 = try transformer.scalarOf(0.7071067811865476, dt, s);
    defer _ = mlx.mlx_array_free(inv_sqrt2);
    const one = try transformer.scalarOf(1.0, dt, s);
    defer _ = mlx.mlx_array_free(one);
    const half = try transformer.scalarOf(0.5, dt, s);
    defer _ = mlx.mlx_array_free(half);
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_multiply(&t, x, inv_sqrt2, s));
    try mlx.check(mlx.mlx_erf(&t, t, s));
    try mlx.check(mlx.mlx_add(&t, t, one, s));
    try mlx.check(mlx.mlx_multiply(&t, x, t, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, t, half, s));
    return out;
}

pub fn replace(dst: *mlx.mlx_array, next: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(dst.*);
    dst.* = next;
}

pub fn appendLin(allocator: std.mem.Allocator, list: *std.ArrayList(mlx.mlx_array), l: Lin) !void {
    for ([_]mlx.mlx_array{ l.w, l.scales, l.biases, l.bias }) |a| {
        if (a.ctx != null) try list.append(allocator, a);
    }
}

/// Weight lookup under the tower's prefix; handles are BORROWED from the weights map.
const NameCtx = struct {
    weights: *const Weights,
    prefix: []const u8,

    fn get(self: NameCtx, comptime fmt: []const u8, args: anytype) ?mlx.mlx_array {
        var buf: [160]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}" ++ fmt, .{self.prefix} ++ args) catch return null;
        return self.weights.get(key);
    }

    fn must(self: NameCtx, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
        return self.get(fmt, args) orelse {
            log.warn("MISSING GLM VISION WEIGHT: {s}" ++ fmt ++ "\n", .{self.prefix} ++ args);
            return error.MissingVisionWeights;
        };
    }

    fn lin(self: NameCtx, comptime fmt: []const u8, args: anytype, in_dim: u32) !Lin {
        var l = Lin{ .w = try self.must(fmt ++ ".weight", args), .bias = self.get(fmt ++ ".bias", args) orelse .{ .ctx = null } };
        if (self.get(fmt ++ ".scales", args)) |sc| {
            l.scales = sc;
            l.biases = self.get(fmt ++ ".biases", args) orelse .{ .ctx = null };
            l.qp = transformer.quantParamsFromGeometry(l.w, sc, l.biases.ctx != null, in_dim) orelse {
                log.warn("GLM VISION: unreadable quantization on {s}" ++ fmt ++ "\n", .{self.prefix} ++ args);
                return error.InvalidVisionWeights;
            };
        }
        return l;
    }
};

// ── Tests ──

const testing = std.testing;

test "resizeGeometry matches the reference processor's canvas and content box" {
    // `smart_resize` + the content box of transformers' Glm5NextImageProcessorPil at the
    // pack's settings (patch 14, merge 2, tps 2, 16..8000 tokens).
    const cases = [_]struct { h: u32, w: u32, th: u32, tw: u32, ch: u32, cw: u32 }{
        .{ .h = 487, .w = 730, .th = 504, .tw = 756, .ch = 487, .cw = 730 }, // fits: padded, never resized
        .{ .h = 61, .w = 97, .th = 112, .tw = 168, .ch = 105, .cw = 168 }, // under the floor: upscaled
        .{ .h = 1000, .w = 7000, .th = 952, .tw = 6580, .ch = 940, .cw = 6580 }, // over the cap
        .{ .h = 1, .w = 1, .th = 112, .tw = 112, .ch = 112, .cw = 112 },
        .{ .h = 10, .w = 2000, .th = 28, .tw = 2016, .ch = 10, .cw = 2000 },
        .{ .h = 2000, .w = 10, .th = 2016, .tw = 28, .ch = 2000, .cw = 10 },
        .{ .h = 448, .w = 448, .th = 448, .tw = 448, .ch = 448, .cw = 448 },
        .{ .h = 28, .w = 28, .th = 112, .tw = 112, .ch = 112, .cw = 112 },
        .{ .h = 1080, .w = 1920, .th = 1092, .tw = 1932, .ch = 1080, .cw = 1920 },
        .{ .h = 2160, .w = 3840, .th = 1876, .tw = 3332, .ch = 1874, .cw = 3332 },
        .{ .h = 3000, .w = 4000, .th = 2156, .tw = 2884, .ch = 2156, .cw = 2874 },
        .{ .h = 100, .w = 100, .th = 112, .tw = 112, .ch = 112, .cw = 112 },
        .{ .h = 112, .w = 113, .th = 112, .tw = 140, .ch = 112, .cw = 113 },
        .{ .h = 5000, .w = 50, .th = 5012, .tw = 56, .ch = 5000, .cw = 50 },
        .{ .h = 767, .w = 1023, .th = 784, .tw = 1036, .ch = 767, .cw = 1023 },
    };
    for (cases) |c| {
        const g = resizeGeometry(c.h, c.w, 28, 2, 16, 8000).?;
        testing.expectEqual(Geometry{ .h = c.th, .w = c.tw, .content_h = c.ch, .content_w = c.cw }, g) catch |e| {
            std.debug.print("{d}x{d}\n", .{ c.h, c.w });
            return e;
        };
    }
    // A budget below one aligned patch is the reference's ValueError.
    try testing.expect(resizeGeometry(100, 100, 28, 2, 0, 0) == null);
}

// Preprocessing and tower parity vs the EXECUTED reference: transformers' own processor and
// Glm5NextVisionModel (fp32, CPU) on the pack's dense tower weights
// (tests/dump_glm5_next_vision_fixtures.py), so a diff is a layout/math bug.
//
//   GLM5_VISION_MODEL=~/.mlx-serve/models/TensorFold/GLM-5.3-Flash-MLX-oQ4-MTP \
//   GLM5_VISION_FIXTURE=~/claude-tmp/glm-vision/glm_vision_fixture.safetensors \
//   zig build test -Doptimize=ReleaseFast -Dtest-filter="glm vision"
fn fixtureEnv() ?[]const u8 {
    const raw = std.c.getenv("GLM5_VISION_FIXTURE") orelse return null;
    const path = std.mem.sliceTo(raw, 0);
    return if (path.len == 0) null else path;
}

test "glm vision: preprocessing equals the reference processor's pixel_values" {
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
        const px = try preprocess(a, rgb, h, w, 14, 2, 2, 16, 8000);
        defer a.free(px.bytes);

        const grid = fx.get(try std.fmt.bufPrint(&kb, "{s}_grid", .{case})) orelse return error.MissingFixtureTensor;
        try mlx.check(mlx.mlx_array_eval(grid));
        const g: [*]const i32 = @ptrCast(@alignCast(mlx.mlx_array_data_int32(grid)));
        try testing.expectEqual(@as(u32, @intCast(g[0])), px.grid_h);
        try testing.expectEqual(@as(u32, @intCast(g[1])), px.grid_w);

        const want_arr = fx.get(try std.fmt.bufPrint(&kb, "{s}_pixel_values", .{case})) orelse return error.MissingFixtureTensor;
        try mlx.check(mlx.mlx_array_eval(want_arr));
        const got = @as([*]const f32, @ptrCast(@alignCast(px.bytes.ptr)))[0 .. px.bytes.len / 4];
        const want = @as([*]const f32, @ptrCast(@alignCast(mlx.mlx_array_data_float32(want_arr))))[0..got.len];
        try testing.expectEqual(mlx.mlx_array_size(want_arr), got.len);
        var worst: f32 = 0;
        for (got, want) |x, y| worst = @max(worst, @abs(x - y));
        std.debug.print("[glm-vision] {s} {d}x{d} -> grid {d}x{d} pixel_values max|diff|={e}\n", .{ case, h, w, px.grid_h, px.grid_w, worst });
        // Pillow's fixed-point resize is reproduced exactly; only f32 rounding may differ.
        try testing.expect(worst < 1e-5);
    }
}

test "glm vision: tower and merger match the reference" {
    const fix_path = fixtureEnv() orelse return error.SkipZigTest;
    const raw_model = std.c.getenv("GLM5_VISION_MODEL") orelse return error.SkipZigTest;
    const model_dir = std.mem.sliceTo(raw_model, 0);
    const a = testing.allocator;

    var config = try model_mod.parseConfig(testing.io, a, model_dir);
    defer config.deinit(a);
    var weights = try model_mod.loadWeightsWithVision(testing.io, a, model_dir);
    defer weights.deinit();
    var fx = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer fx.deinit();
    var gv = try GlmVision.init(a, config, &weights);
    defer gv.deinit();
    const s = gv.s;

    for ([_][]const u8{ "noise", "house" }) |case| {
        var kb: [64]u8 = undefined;
        const grid = fx.get(try std.fmt.bufPrint(&kb, "{s}_grid", .{case})) orelse return error.MissingFixtureTensor;
        try mlx.check(mlx.mlx_array_eval(grid));
        const g: [*]const i32 = @ptrCast(@alignCast(mlx.mlx_array_data_int32(grid)));
        const gh: u32 = @intCast(g[0]);
        const gw: u32 = @intCast(g[1]);
        const pv = fx.get(try std.fmt.bufPrint(&kb, "{s}_pixel_values", .{case})) orelse return error.MissingFixtureTensor;

        const hidden = try gv.towerHidden(pv, gh, gw);
        defer _ = mlx.mlx_array_free(hidden);
        const down = try gv.downsample(hidden);
        defer _ = mlx.mlx_array_free(down);
        const feats = try gv.forward(pv, gh, gw);
        defer _ = mlx.mlx_array_free(feats);
        // Every stage names itself, so a failure says which one; scale too, since these
        // rows are spliced into the token stream where a cosine cannot see magnitude.
        const Stage = struct { name: []const u8, got: mlx.mlx_array };
        for ([_]Stage{ .{ .name = "post_ln", .got = hidden }, .{ .name = "downsample", .got = down }, .{ .name = "features", .got = feats } }) |stage| {
            const want = fx.get(try std.fmt.bufPrint(&kb, "{s}_{s}", .{ case, stage.name })) orelse return error.MissingFixtureTensor;
            const c = try cosineSim(stage.got, want, s);
            const r = try rmsRatio(stage.got, want, s);
            std.debug.print("[glm-vision] {s} grid {d}x{d} {s} cos={d:.6} rms_ratio={d:.4}\n", .{ case, gh, gw, stage.name, c, r });
            // bf16 serving against an fp32 reference through 24 blocks.
            try testing.expect(c > 0.99 and r > 0.98 and r < 1.02);
        }
    }
}

pub fn cosineSim(a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    const dot = try sumProd(a_arr, b_arr, s);
    const na = try sumProd(a_arr, a_arr, s);
    const nb = try sumProd(b_arr, b_arr, s);
    if (!std.math.isFinite(dot) or na <= 0 or nb <= 0) return std.math.nan(f32);
    return dot / (@sqrt(na) * @sqrt(nb));
}

pub fn rmsRatio(a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    const na = try sumProd(a_arr, a_arr, s);
    const nb = try sumProd(b_arr, b_arr, s);
    if (!std.math.isFinite(na) or nb <= 0) return std.math.nan(f32);
    return @sqrt(na) / @sqrt(nb);
}

/// sum(a*b) in f32 over the flattened inputs; NaN propagates so it can never pass a bar.
fn sumProd(a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    const n = [_]c_int{@intCast(mlx.mlx_array_size(a_arr))};
    var af = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(af);
    try mlx.check(mlx.mlx_astype(&af, a_arr, .float32, s));
    try mlx.check(mlx.mlx_reshape(&af, af, &n, 1, s));
    var bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bf);
    try mlx.check(mlx.mlx_astype(&bf, b_arr, .float32, s));
    try mlx.check(mlx.mlx_reshape(&bf, bf, &n, 1, s));
    var prod = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prod);
    try mlx.check(mlx.mlx_multiply(&prod, af, bf, s));
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_sum(&o, prod, false, s));
    try mlx.check(mlx.mlx_array_eval(o));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, o));
    return v;
}
