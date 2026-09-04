//! Mistral3 (Mistral Small 3.1/3.2) vision tower + preprocessing math.
//!
//! Port of `transformers/models/pixtral/` (`modeling_pixtral.py`,
//! `image_processing_pixtral.py`) and Mistral3's own patch-merger/projector
//! (`modeling_mistral3.py`'s `Mistral3PatchMerger`/`Mistral3MultiModalProjector`).
//! No CLS token, no learned absolute position embedding — a bias-less
//! non-overlapping Conv2d patch embed (implemented as patchify+matmul, same
//! trick the other patch-grid towers use) followed by 2D RoPE. One image per
//! `forward` call (see `vision.zig:forwardPatches`), so — unlike the
//! reference, which batches multiple images with a block-diagonal attention
//! mask — no cross-image masking is needed here.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;
const qwen_vision = @import("qwen_vision.zig");
const log = @import("log.zig");

pub const Resized = qwen_vision.Resized;

/// Pixtral's `image_processing_pixtral.py` resize: scale down (never up) so
/// neither dimension exceeds `longest_edge`, floor-rounding the scaled
/// dimensions ("we use floor to ensure the image is always smaller than the
/// given 'longest_edge'"), then round UP to the nearest patch multiple via
/// ceil-div (`(dim - 1) // patch + 1`).
pub fn resizePixtral(height: u32, width: u32, longest_edge: u32, patch: u32) Resized {
    const fh: f64 = @floatFromInt(height);
    const fw: f64 = @floatFromInt(width);
    const cap: f64 = @floatFromInt(longest_edge);
    const ratio = @max(fh / cap, fw / cap);

    var h = height;
    var w = width;
    if (ratio > 1.0) {
        h = @intFromFloat(@floor(fh / ratio));
        w = @intFromFloat(@floor(fw / ratio));
        if (h < 1) h = 1;
        if (w < 1) w = 1;
    }
    const gh = (h - 1) / patch + 1;
    const gw = (w - 1) / patch + 1;
    return .{ .h = gh * patch, .w = gw * patch };
}

/// CLIP mean/std (Pixtral's own normalization — NOT the 0.5/0.5
/// `qwen_vision.normalizeQwenPixel` the other towers share).
const clip_mean = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
const clip_std = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };

/// Bicubic resize of interleaved RGB into CLIP-normalized CHW f32, reusing
/// `qwen_vision`'s separable Pillow-compatible resample coefficients (same
/// filter, different final normalization).
pub fn resizeRgbBicubicClipNormalizedChw(
    allocator: std.mem.Allocator,
    dst: []f32,
    rgb: []const u8,
    source_h: u32,
    source_w: u32,
    target_h: u32,
    target_w: u32,
) !void {
    const source_plane: usize = @as(usize, source_h) * source_w;
    const target_plane: usize = @as(usize, target_h) * target_w;
    if (source_h == 0 or source_w == 0 or target_h == 0 or target_w == 0)
        return error.InvalidImageDimensions;
    if (rgb.len != source_plane * 3 or dst.len != target_plane * 3)
        return error.InvalidImageBuffer;

    var horizontal_owned: ?[]u8 = null;
    defer if (horizontal_owned) |buffer| allocator.free(buffer);
    const horizontal: []const u8 = if (target_w != source_w) resize: {
        const horizontal_len = try std.math.mul(usize, @as(usize, source_h) * target_w, 3);
        const buffer = try allocator.alloc(u8, horizontal_len);
        horizontal_owned = buffer;

        var coefficients = try qwen_vision.buildResampleCoefficients(allocator, source_w, target_w, .bicubic);
        defer coefficients.deinit(allocator);
        for (0..source_h) |y| {
            for (0..target_w) |x| {
                const bound = coefficients.bounds[x];
                const weights = coefficients.weights[x * coefficients.kernel_size ..][0..bound.len];
                inline for (0..3) |channel| {
                    var sum: i64 = qwen_vision.RESAMPLE_ROUNDING_BIAS;
                    for (weights, 0..) |weight, source_offset| {
                        const source = (y * source_w + bound.start + source_offset) * 3 + channel;
                        sum += @as(i64, rgb[source]) * @as(i64, weight);
                    }
                    buffer[(y * target_w + x) * 3 + channel] = qwen_vision.clipFixedResample(sum);
                }
            }
        }
        break :resize buffer;
    } else rgb;

    if (target_h != source_h) {
        var coefficients = try qwen_vision.buildResampleCoefficients(allocator, source_h, target_h, .bicubic);
        defer coefficients.deinit(allocator);
        for (0..target_h) |y| {
            const bound = coefficients.bounds[y];
            const weights = coefficients.weights[y * coefficients.kernel_size ..][0..bound.len];
            for (0..target_w) |x| {
                const destination = y * target_w + x;
                inline for (0..3) |channel| {
                    var sum: i64 = qwen_vision.RESAMPLE_ROUNDING_BIAS;
                    for (weights, 0..) |weight, source_offset| {
                        const source = ((bound.start + source_offset) * target_w + x) * 3 + channel;
                        sum += @as(i64, horizontal[source]) * @as(i64, weight);
                    }
                    dst[channel * target_plane + destination] = normalizeClipPixel(qwen_vision.clipFixedResample(sum), channel);
                }
            }
        }
    } else {
        for (0..target_h) |y| {
            for (0..target_w) |x| {
                const source = (y * target_w + x) * 3;
                const destination = y * target_w + x;
                inline for (0..3) |channel| {
                    dst[channel * target_plane + destination] = normalizeClipPixel(horizontal[source + channel], channel);
                }
            }
        }
    }
}

fn normalizeClipPixel(value: u8, channel: usize) f32 {
    const x = @as(f32, @floatFromInt(value)) / 255.0;
    return (x - clip_mean[channel]) / clip_std[channel];
}

/// Build the patch_conv-equivalent linear input `[gh*gw, 3*patch*patch]` from
/// a normalized CHW image, row-major grid order (h outer, w inner — matches
/// the RoPE position-id convention `h*max_side+w`) and CHANNEL-major feature
/// layout `[c, py, px]`, matching `vision_tower.patch_conv.weight`'s natural
/// PyTorch flatten order `(in_channels, kH, kW)` so the patch embed is a
/// plain matmul against the reshaped conv weight.
pub fn buildPixelValues(out: []f32, img_chw: []const f32, C: u32, rh: u32, rw: u32, patch: u32) void {
    const gh = rh / patch;
    const gw = rw / patch;
    const feat = C * patch * patch;
    std.debug.assert(out.len == @as(usize, gh) * gw * feat);
    const plane: usize = @as(usize, rh) * rw;

    for (0..gh) |row| {
        for (0..gw) |col| {
            const base = (row * gw + col) * feat;
            var f: usize = 0;
            for (0..C) |c| {
                for (0..patch) |py| {
                    const y = row * patch + py;
                    for (0..patch) |px| {
                        const x = col * patch + px;
                        out[base + f] = img_chw[c * plane + y * rw + x];
                        f += 1;
                    }
                }
            }
        }
    }
}

const Lin = struct {
    w: mlx.mlx_array,
    scales: mlx.mlx_array = .{ .ctx = null },
    biases: mlx.mlx_array = .{ .ctx = null },
    bias: mlx.mlx_array = .{ .ctx = null },
    bits: u32 = 0,
    group: u32 = 0,
    mode: [*:0]const u8 = "affine",
};

const Layer = struct {
    attn_norm_w: mlx.mlx_array,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    ffn_norm_w: mlx.mlx_array,
    gate: Lin,
    up: Lin,
    down: Lin,
};

pub const PixtralVision = struct {
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,

    hidden: u32,
    heads: u32,
    head_dim: u32,
    merge: u32,
    max_side: u32, // max_patches_per_side = pv_image_size / patch
    rms_eps: f32, // vision tower's own norm eps (Pixtral reference default 1e-5)
    proj_rms_eps: f32, // projector's norm — the TEXT config's rms_norm_eps

    patch_w: Lin, // [hidden, C*patch*patch], dense
    ln_pre_w: mlx.mlx_array,
    layers: []Layer,

    proj_norm_w: mlx.mlx_array,
    merger: Lin, // [hidden, hidden*merge*merge]
    linear1: Lin,
    linear2: Lin,

    rope_cos: mlx.mlx_array, // [max_side*max_side, head_dim] bf16
    rope_sin: mlx.mlx_array,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !PixtralVision {
        const s = mlx.gpuStream();
        var name_buf: [160]u8 = undefined;
        const ctx = NameCtx{ .weights = weights, .buf = &name_buf, .mode = config.quant_mode.cstr() };

        const hidden = config.vision_hidden_size;
        const patch = config.vision_patch_size;
        const c_in: u32 = 3;

        const patch_w_raw = try ctx.must("vision_tower.patch_conv.weight", .{});
        // [hidden, C, patch, patch] → [hidden, C*patch*patch]; contiguous
        // reshape, matches the conv weight's natural PyTorch flatten order.
        var patch_w = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&patch_w, patch_w_raw, &[_]c_int{ @intCast(hidden), @intCast(c_in * patch * patch) }, 2, s));

        var layers = try allocator.alloc(Layer, config.vision_num_layers);
        errdefer allocator.free(layers);
        for (0..config.vision_num_layers) |i| {
            layers[i] = .{
                .attn_norm_w = try ctx.must("vision_tower.transformer.layers.{d}.attention_norm.weight", .{i}),
                .q = try ctx.lin("vision_tower.transformer.layers.{d}.attention.q_proj", .{i}, hidden),
                .k = try ctx.lin("vision_tower.transformer.layers.{d}.attention.k_proj", .{i}, hidden),
                .v = try ctx.lin("vision_tower.transformer.layers.{d}.attention.v_proj", .{i}, hidden),
                .o = try ctx.lin("vision_tower.transformer.layers.{d}.attention.o_proj", .{i}, hidden),
                .ffn_norm_w = try ctx.must("vision_tower.transformer.layers.{d}.ffn_norm.weight", .{i}),
                .gate = try ctx.lin("vision_tower.transformer.layers.{d}.feed_forward.gate_proj", .{i}, hidden),
                .up = try ctx.lin("vision_tower.transformer.layers.{d}.feed_forward.up_proj", .{i}, hidden),
                .down = try ctx.lin("vision_tower.transformer.layers.{d}.feed_forward.down_proj", .{i}, config.vision_intermediate_size),
            };
        }

        const merge = config.pv_spatial_merge;
        const max_side = if (config.vision_patch_size > 0) config.pv_image_size / config.vision_patch_size else 0;
        const rope = try buildRope(allocator, max_side, config.vision_head_dim, @floatCast(config.vision_rope_theta), s);

        log.info("Vision encoder: Pixtral ViT (layers={d}, hidden={d}, heads={d}, merge={d})\n", .{
            config.vision_num_layers, hidden, config.vision_num_heads, merge,
        });

        return .{
            .s = s,
            .allocator = allocator,
            .hidden = hidden,
            .heads = config.vision_num_heads,
            .head_dim = config.vision_head_dim,
            .merge = merge,
            .max_side = max_side,
            .rms_eps = 1e-5,
            .proj_rms_eps = config.rms_norm_eps,
            .patch_w = .{ .w = patch_w },
            .ln_pre_w = try ctx.must("vision_tower.ln_pre.weight", .{}),
            .layers = layers,
            .proj_norm_w = try ctx.must("multi_modal_projector.norm.weight", .{}),
            .merger = try ctx.lin("multi_modal_projector.patch_merger.merging_layer", .{}, hidden * merge * merge),
            .linear1 = try ctx.lin("multi_modal_projector.linear_1", .{}, hidden * merge * merge),
            .linear2 = try ctx.lin("multi_modal_projector.linear_2", .{}, config.hidden_size),
            .rope_cos = rope.cos,
            .rope_sin = rope.sin,
        };
    }

    pub fn deinit(self: *PixtralVision) void {
        _ = mlx.mlx_array_free(self.rope_cos);
        _ = mlx.mlx_array_free(self.rope_sin);
        self.allocator.free(self.layers);
    }

    pub fn forward(self: *PixtralVision, patches: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const n: c_int = @intCast(grid_h * grid_w);
        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&x, patches, .bfloat16, self.s));
        defer _ = mlx.mlx_array_free(x);

        replace(&x, try self.linear(x, self.patch_w));
        replace(&x, try self.rmsNorm(x, self.ln_pre_w, self.rms_eps));

        // Position ids for THIS image's grid, row-major (h outer, w inner —
        // matches buildPixelValues), gathered from the precomputed full
        // max_side×max_side table.
        const pos = try self.allocator.alloc(i32, grid_h * grid_w);
        defer self.allocator.free(pos);
        for (0..grid_h) |h| {
            for (0..grid_w) |w| {
                pos[h * grid_w + w] = @intCast(h * self.max_side + w);
            }
        }
        const cos, const sin = try self.gatherRope(pos);
        defer _ = mlx.mlx_array_free(cos);
        defer _ = mlx.mlx_array_free(sin);

        var dt = mlx.DtypeTrace.begin("pixtral-vision", x, if (self.layers.len > 0) self.layers[0].attn_norm_w else null);
        for (self.layers, 0..) |layer, i| {
            {
                const normed = try self.rmsNorm(x, layer.attn_norm_w, self.rms_eps);
                defer _ = mlx.mlx_array_free(normed);
                const attn = try self.attention(normed, layer, cos, sin, n);
                defer _ = mlx.mlx_array_free(attn);
                var h = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h, x, attn, self.s));
                replace(&x, h);
            }
            {
                const normed = try self.rmsNorm(x, layer.ffn_norm_w, self.rms_eps);
                defer _ = mlx.mlx_array_free(normed);
                const gate_out = try self.linear(normed, layer.gate);
                defer _ = mlx.mlx_array_free(gate_out);
                const up_out = try self.linear(normed, layer.up);
                defer _ = mlx.mlx_array_free(up_out);
                const act = try self.silu(gate_out);
                defer _ = mlx.mlx_array_free(act);
                var gated = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(gated);
                try mlx.check(mlx.mlx_multiply(&gated, act, up_out, self.s));
                const down_out = try self.linear(gated, layer.down);
                defer _ = mlx.mlx_array_free(down_out);
                var h = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h, x, down_out, self.s));
                replace(&x, h);
            }
            dt.layer(x, i);
        }
        dt.end(x);

        // Mistral3MultiModalProjector: norm (TEXT rms_norm_eps) → patch merger
        // → linear_1 → GELU → linear_2.
        replace(&x, try self.rmsNorm(x, self.proj_norm_w, self.proj_rms_eps));
        replace(&x, try self.patchMerge(x, grid_h, grid_w));
        replace(&x, try self.linear(x, self.merger));
        replace(&x, try self.linear(x, self.linear1));
        replace(&x, try self.gelu(x));
        replace(&x, try self.linear(x, self.linear2));

        const bh = grid_h / self.merge;
        const bw = grid_w / self.merge;
        var out = mlx.mlx_array_new();
        const oshape = [_]c_int{ 1, @intCast(bh * bw), @intCast(mlx.getShape(x)[1]) };
        try mlx.check(mlx.mlx_reshape(&out, x, &oshape, 3, self.s));
        return out;
    }

    /// Space-to-depth merge of `merge × merge` neighboring patches into one
    /// wider token: `(gh,gw,C) → (Bh,k,Bw,k,C) → transpose → (Bh,Bw,C,k,k) →
    /// (Bh*Bw, C*k*k)`. For the non-overlapping stride==kernel_size case this
    /// is EXACTLY PyTorch's `F.unfold(kernel=k,stride=k)` used by
    /// `Mistral3PatchMerger` — the flattened feature index is
    /// `c*(k*k)+kh*k+kw` (channel-major, then within-block row-major
    /// position), confirmed from the reference `unfold` helper. Getting this
    /// axis order wrong silently scrambles the image.
    ///
    /// `resizePixtral` only rounds the grid up to a multiple of `patch`, not
    /// `patch*merge`, so an odd grid (e.g. 110×83) is routine. `unfold`
    /// silently drops a trailing row/col that doesn't fill a full `k×k`
    /// block — crop to `(bh*k, bw*k)` before the reshape to match.
    fn patchMerge(self: *PixtralVision, x: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const k = self.merge;
        const bh = grid_h / k;
        const bw = grid_w / k;
        const C: c_int = @intCast(self.hidden);

        var src = x;
        var cropped = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cropped);
        if (bh * k != grid_h or bw * k != grid_w) {
            var g = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(g);
            try mlx.check(mlx.mlx_reshape(&g, x, &[_]c_int{ @intCast(grid_h), @intCast(grid_w), C }, 3, self.s));

            var sl = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sl);
            const start = [_]c_int{ 0, 0, 0 };
            const stop = [_]c_int{ @intCast(bh * k), @intCast(bw * k), C };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&sl, g, &start, 3, &stop, 3, &strides, 3, self.s));

            var contiguous = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(contiguous);
            try mlx.check(mlx.mlx_contiguous(&contiguous, sl, false, self.s));
            try mlx.check(mlx.mlx_reshape(&cropped, contiguous, &[_]c_int{ @intCast(bh * k * bw * k), C }, 2, self.s));
            src = cropped;
        }

        var r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r);
        try mlx.check(mlx.mlx_reshape(&r, src, &[_]c_int{ @intCast(bh), @intCast(k), @intCast(bw), @intCast(k), C }, 5, self.s));

        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_transpose_axes(&t, r, &[_]c_int{ 0, 2, 4, 1, 3 }, 5, self.s));

        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&out, t, &[_]c_int{ @intCast(bh * bw), @intCast(self.hidden * k * k) }, 2, self.s));
        return out;
    }

    fn attention(self: *PixtralVision, x: mlx.mlx_array, layer: Layer, cos: mlx.mlx_array, sin: mlx.mlx_array, n: c_int) !mlx.mlx_array {
        const hd: c_int = @intCast(self.head_dim);
        const heads: c_int = @intCast(self.heads);
        var qkv: [3]mlx.mlx_array = undefined;
        inline for (.{ layer.q, layer.k, layer.v }, 0..) |l, i| {
            const flat = try self.linear(x, l);
            defer _ = mlx.mlx_array_free(flat);
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&r, flat, &[_]c_int{ n, heads, hd }, 3, self.s));
            qkv[i] = if (i < 2) blk: {
                defer _ = mlx.mlx_array_free(r);
                break :blk try self.applyRope(r, cos, sin, n, hd);
            } else r;
        }
        defer for (qkv) |a| {
            _ = mlx.mlx_array_free(a);
        };

        // [N, heads, hd] → [1, heads, N, hd] for the fused SDPA.
        var bhnd: [3]mlx.mlx_array = undefined;
        const perm = [_]c_int{ 1, 0, 2 };
        const bshape = [_]c_int{ 1, heads, n, hd };
        inline for (qkv, 0..) |a, i| {
            var t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(t);
            try mlx.check(mlx.mlx_transpose_axes(&t, a, &perm, 3, self.s));
            var b = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&b, t, &bshape, 4, self.s));
            bhnd[i] = b;
        }
        defer for (bhnd) |a| {
            _ = mlx.mlx_array_free(a);
        };

        // One image per call — full (unmasked) attention, no block-diagonal
        // cross-image mask needed (see file doc comment).
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const none = mlx.mlx_array{ .ctx = null };
        var ctx_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ctx_out);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ctx_out, bhnd[0], bhnd[1], bhnd[2], scale, "", none, none, false, self.s));

        // [1, heads, N, hd] → [N, heads*hd]
        const back = [_]c_int{ 0, 2, 1, 3 };
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_transpose_axes(&t, ctx_out, &back, 4, self.s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, t, &[_]c_int{ n, @intCast(self.hidden) }, 2, self.s));
        return self.linear(flat, layer.o);
    }

    /// x·cos + rotate_half(x)·sin over [N, heads, hd]; cos/sin are [N, 1, hd].
    fn applyRope(self: *PixtralVision, x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, n: c_int, hd: c_int) !mlx.mlx_array {
        const heads: c_int = @intCast(self.heads);
        const half = @divExact(hd, 2);
        const strides = [_]c_int{ 1, 1, 1 };
        var x1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x1);
        try mlx.check(mlx.mlx_slice(&x1, x, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ n, heads, half }, 3, &strides, 3, self.s));
        var x2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x2);
        try mlx.check(mlx.mlx_slice(&x2, x, &[_]c_int{ 0, 0, half }, 3, &[_]c_int{ n, heads, hd }, 3, &strides, 3, self.s));
        var neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(neg);
        try mlx.check(mlx.mlx_negative(&neg, x2, self.s));
        const arrs = [_]mlx.mlx_array{ neg, x1 };
        const vec = mlx.mlx_vector_array_new_data(&arrs, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        var rot = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rot);
        try mlx.check(mlx.mlx_concatenate_axis(&rot, vec, -1, self.s));

        var xc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xc);
        try mlx.check(mlx.mlx_multiply(&xc, x, cos, self.s));
        var rs = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rs);
        try mlx.check(mlx.mlx_multiply(&rs, rot, sin, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&out, xc, rs, self.s));
        return out;
    }

    /// Gather rows `pos` from the precomputed [max_side*max_side, hd] table,
    /// reshaped to [N,1,hd] for elementwise broadcast against [N,heads,hd].
    fn gatherRope(self: *PixtralVision, pos: []const i32) !struct { mlx.mlx_array, mlx.mlx_array } {
        const idx = hostI32(pos);
        defer _ = mlx.mlx_array_free(idx);
        const n: c_int = @intCast(pos.len);
        const hd: c_int = @intCast(self.head_dim);

        var cos_g = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_take_axis(&cos_g, self.rope_cos, idx, 0, self.s));
        var cos = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&cos, cos_g, &[_]c_int{ n, 1, hd }, 3, self.s));
        _ = mlx.mlx_array_free(cos_g);

        var sin_g = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_take_axis(&sin_g, self.rope_sin, idx, 0, self.s));
        var sin = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&sin, sin_g, &[_]c_int{ n, 1, hd }, 3, self.s));
        _ = mlx.mlx_array_free(sin_g);

        return .{ cos, sin };
    }

    fn rmsNorm(self: *PixtralVision, x: mlx.mlx_array, w: mlx.mlx_array, eps: f32) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, eps, self.s));
        return out;
    }

    /// x * sigmoid(x).
    fn silu(self: *PixtralVision, x: mlx.mlx_array) !mlx.mlx_array {
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        try mlx.check(mlx.mlx_sigmoid(&sig, x, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&out, x, sig, self.s));
        return out;
    }

    /// nn.functional.gelu (exact erf form) — `projector_hidden_act` is "gelu".
    fn gelu(self: *PixtralVision, x: mlx.mlx_array) !mlx.mlx_array {
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

    fn linear(self: *PixtralVision, x: mlx.mlx_array, l: Lin) !mlx.mlx_array {
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

/// Precompute the full 2D RoPE table [max_side*max_side, head_dim], per
/// mlx-vlm's `PixtralRotaryEmbedding`: `freqs[i] = theta^(-2i/dim)` for
/// `i in [0, dim/2)`; `freqs_h[k] = freqs[2k]`, `freqs_w[k] = freqs[2k+1]`
/// for `k in [0, dim/4)`. Row `t=(h,w)` (position id `h*max_side+w`) holds
/// `[h*freqs_h[0..nfreq), w*freqs_w[0..nfreq), h*freqs_h[..], w*freqs_w[..]]`
/// — i.e. concat(freqs_h-block, freqs_w-block) duplicated to fill `head_dim`
/// (CLAUDE.md documents Muse's own 2D RoPE as the SAME concat-then-duplicate
/// shape but with the two blocks in [w,h,w,h] order; Pixtral's is [h,w,h,w]).
fn buildRope(allocator: std.mem.Allocator, max_side: u32, head_dim: u32, theta: f64, s: mlx.mlx_stream) !struct { cos: mlx.mlx_array, sin: mlx.mlx_array } {
    const hd: usize = head_dim;
    const nfreq = hd / 4;
    const n = @as(usize, max_side) * max_side;

    const freqs_h = try allocator.alloc(f64, nfreq);
    defer allocator.free(freqs_h);
    const freqs_w = try allocator.alloc(f64, nfreq);
    defer allocator.free(freqs_w);
    for (0..nfreq) |k| {
        freqs_h[k] = std.math.pow(f64, theta, -@as(f64, @floatFromInt(4 * k)) / @as(f64, @floatFromInt(hd)));
        freqs_w[k] = std.math.pow(f64, theta, -@as(f64, @floatFromInt(4 * k + 2)) / @as(f64, @floatFromInt(hd)));
    }

    const cos_buf = try allocator.alloc(f32, n * hd);
    defer allocator.free(cos_buf);
    const sin_buf = try allocator.alloc(f32, n * hd);
    defer allocator.free(sin_buf);

    const half = hd / 2;
    for (0..max_side) |h| {
        for (0..max_side) |w| {
            const t = h * max_side + w;
            const o = t * hd;
            const fh: f64 = @floatFromInt(h);
            const fw: f64 = @floatFromInt(w);
            inline for (.{ 0, half }) |base| {
                for (0..nfreq) |k| {
                    const ah = fh * freqs_h[k];
                    const aw = fw * freqs_w[k];
                    cos_buf[o + base + k] = @floatCast(@cos(ah));
                    cos_buf[o + base + nfreq + k] = @floatCast(@cos(aw));
                    sin_buf[o + base + k] = @floatCast(@sin(ah));
                    sin_buf[o + base + nfreq + k] = @floatCast(@sin(aw));
                }
            }
        }
    }

    const shape = [_]c_int{ @intCast(n), @intCast(hd) };
    const cf = mlx.mlx_array_new_data(cos_buf.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(cf);
    const sf = mlx.mlx_array_new_data(sin_buf.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(sf);
    var cos = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&cos, cf, .bfloat16, s));
    var sin = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&sin, sf, .bfloat16, s));
    return .{ .cos = cos, .sin = sin };
}

fn replace(dst: *mlx.mlx_array, next: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(dst.*);
    dst.* = next;
}

fn hostI32(v: []const i32) mlx.mlx_array {
    const shape = [_]c_int{@intCast(v.len)};
    return mlx.mlx_array_new_data(v.ptr, &shape, 1, .int32);
}

fn bf16Scalar(v: f32, s: mlx.mlx_stream) mlx.mlx_array {
    const f = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&out, f, .bfloat16, s);
    return out;
}

/// Weight lookup under the checkpoint's flat `vision_tower.`/
/// `multi_modal_projector.` prefixes (confirmed against a real Mistral3
/// safetensors index — no nested "model." root, unlike Muse). Handles are
/// BORROWED from the weights map; the encoder owns none of them.
const NameCtx = struct {
    weights: *const Weights,
    buf: *[160]u8,
    mode: [*:0]const u8,

    fn key(self: NameCtx, comptime fmt: []const u8, args: anytype) []const u8 {
        return std.fmt.bufPrint(self.buf, fmt, args) catch unreachable;
    }

    fn opt(self: NameCtx, comptime fmt: []const u8, args: anytype) ?mlx.mlx_array {
        return self.weights.get(self.key(fmt, args));
    }

    fn must(self: NameCtx, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
        return self.opt(fmt, args) orelse {
            log.warn("MISSING PIXTRAL VISION WEIGHT: {s}\n", .{self.key(fmt, args)});
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

const testing = std.testing;

test "resizePixtral: already patch-aligned image is unchanged" {
    const r = resizePixtral(560, 420, 1540, 14);
    try testing.expectEqual(@as(u32, 560), r.h);
    try testing.expectEqual(@as(u32, 420), r.w);
}

test "resizePixtral: downscales when longer than longest_edge, floor then ceil-div" {
    // 2000 > 1540 → ratio = 2000/1540 ≈ 1.2987; H=floor(2000/ratio)=1540,
    // W=floor(1500/ratio)=floor(1155.0)=1155 → ceil-div to patch 14:
    // gh=(1540-1)/14+1=110 → 1540; gw=(1155-1)/14+1=83 (1162/14=83.0) → 1162.
    const r = resizePixtral(2000, 1500, 1540, 14);
    try testing.expectEqual(@as(u32, 1540), r.h);
    try testing.expectEqual(@as(u32, 1162), r.w);
}

test "resizePixtral: non-patch-aligned dims round UP to the next patch multiple" {
    // 100x100 is under the cap, but 100 is not a multiple of 14:
    // ceil-div(100,14) = 8 → 112.
    const r = resizePixtral(100, 100, 1540, 14);
    try testing.expectEqual(@as(u32, 112), r.h);
    try testing.expectEqual(@as(u32, 112), r.w);
}

test "patchMerge: 2x2 merge orders features channel-major then within-block row-major" {
    // Synthetic 2x2 grid (gh=gw=2), C=2 channels, merge=2 → ONE output token
    // whose 8 features must be [c0@(0,0), c0@(0,1), c0@(1,0), c0@(1,1),
    // c1@(0,0), c1@(0,1), c1@(1,0), c1@(1,1)] — i.e. index c*(k*k)+kh*k+kw.
    // Patch order in `x` is row-major (h outer, w inner): (0,0),(0,1),(1,0),(1,1).
    const s = mlx.gpuStream();
    const data = [_]f32{
        // patch(0,0): c0=1, c1=10
        1, 10,
        // patch(0,1): c0=2, c1=20
        2, 20,
        // patch(1,0): c0=3, c1=30
        3, 30,
        // patch(1,1): c0=4, c1=40
        4, 40,
    };
    const shape = [_]c_int{ 4, 2 };
    const raw = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    var x = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&x, raw, .bfloat16, s));
    _ = mlx.mlx_array_free(raw);
    defer _ = mlx.mlx_array_free(x);

    var pv: PixtralVision = undefined;
    pv.s = s;
    pv.allocator = testing.allocator;
    pv.hidden = 2;
    pv.merge = 2;

    const merged = try pv.patchMerge(x, 2, 2);
    defer _ = mlx.mlx_array_free(merged);

    var f32_out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&f32_out, merged, .float32, s));
    defer _ = mlx.mlx_array_free(f32_out);
    _ = mlx.mlx_eval(mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{f32_out}, 1));
    const out_shape = mlx.getShape(f32_out);
    try testing.expectEqual(@as(c_int, 1), out_shape[0]);
    try testing.expectEqual(@as(c_int, 8), out_shape[1]);

    const ptr = mlx.mlx_array_data_float32(f32_out).?;
    const got = ptr[0..8];
    const want = [_]f32{ 1, 2, 3, 4, 10, 20, 30, 40 };
    for (want, 0..) |w, i| {
        try testing.expectApproxEqAbs(w, got[i], 1e-2);
    }
}

test "patchMerge: an odd grid crops the trailing row/col before merging" {
    // resizePixtral(2000, 1500, 1540, 14) yields a 110x83-patch grid — 83 is
    // odd, so merge=2 must drop the trailing column exactly like PyTorch's
    // F.unfold(kernel=2,stride=2) would. Minimal repro: gh=2, gw=3 (only the
    // width is odd), C=1, values = h*10+w so the dropped column is visible.
    const s = mlx.gpuStream();
    const data = [_]f32{
        0,  1,  2,
        10, 11, 12,
    };
    const shape = [_]c_int{ 6, 1 };
    const raw = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    var x = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&x, raw, .bfloat16, s));
    _ = mlx.mlx_array_free(raw);
    defer _ = mlx.mlx_array_free(x);

    var pv: PixtralVision = undefined;
    pv.s = s;
    pv.allocator = testing.allocator;
    pv.hidden = 1;
    pv.merge = 2;

    // gh=2 (already a multiple of 2), gw=3 (bw = 3/2 = 1, drops column 2).
    const merged = try pv.patchMerge(x, 2, 3);
    defer _ = mlx.mlx_array_free(merged);

    var f32_out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&f32_out, merged, .float32, s));
    defer _ = mlx.mlx_array_free(f32_out);
    _ = mlx.mlx_eval(mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{f32_out}, 1));
    const out_shape = mlx.getShape(f32_out);
    try testing.expectEqual(@as(c_int, 1), out_shape[0]);
    try testing.expectEqual(@as(c_int, 4), out_shape[1]);

    const ptr = mlx.mlx_array_data_float32(f32_out).?;
    const got = ptr[0..4];
    // c*(k*k)+kh*k+kw with C=1: [patch(0,0), patch(0,1), patch(1,0), patch(1,1)].
    const want = [_]f32{ 0, 1, 10, 11 };
    for (want, 0..) |w, i| {
        try testing.expectApproxEqAbs(w, got[i], 1e-2);
    }
}

test "buildRope: table shape and determinism" {
    const s = mlx.gpuStream();
    const rope = try buildRope(testing.allocator, 4, 8, 10000.0, s);
    defer _ = mlx.mlx_array_free(rope.cos);
    defer _ = mlx.mlx_array_free(rope.sin);
    const shape = mlx.getShape(rope.cos);
    try testing.expectEqual(@as(c_int, 16), shape[0]); // 4*4
    try testing.expectEqual(@as(c_int, 8), shape[1]);

    var cos_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cos_f32);
    try mlx.check(mlx.mlx_astype(&cos_f32, rope.cos, .float32, s));
    _ = mlx.mlx_eval(mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{cos_f32}, 1));
    const got = mlx.mlx_array_data_float32(cos_f32).?;

    const rope2 = try buildRope(testing.allocator, 4, 8, 10000.0, s);
    defer _ = mlx.mlx_array_free(rope2.cos);
    defer _ = mlx.mlx_array_free(rope2.sin);
    var cos2_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cos2_f32);
    try mlx.check(mlx.mlx_astype(&cos2_f32, rope2.cos, .float32, s));
    _ = mlx.mlx_eval(mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{cos2_f32}, 1));
    const got2 = mlx.mlx_array_data_float32(cos2_f32).?;

    for (0..16 * 8) |i| try testing.expectEqual(got[i], got2[i]);
    // Position (0,0) is the zero-angle row: cos == 1 everywhere.
    for (0..8) |i| try testing.expectApproxEqAbs(@as(f32, 1.0), got[i], 1e-6);
}
