//! Qwen3.5/3.6 (Qwen3-VL) vision tower + preprocessing math for mlx-serve.
//!
//! Mirrors mlx-vlm `qwen3_vl/vision.py` (the ViT) and `qwen3_vl/processing_qwen3_vl.py`
//! (`_smart_resize_image`). This file currently holds the pure preprocessing math
//! (`smartResizeImage`); the MLX ViT encoder (`QwenVision`) lands alongside it and
//! is dispatched from `vision.VisionEncoder` when `config.qwen_vision` is set.
//!
//! Image tokens per image = `(grid_h/merge) * (grid_w/merge)` where
//! `grid_h = resized_H/patch`, `grid_w = resized_W/patch` — a DYNAMIC count, unlike
//! Gemma 4's fixed 280.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;
const log = @import("log.zig");

/// mlx-vlm `Qwen3VLImageProcessor` defaults (processing_qwen3_vl.py:96-98). The
/// config.json `size.{shortest,longest}_edge` do NOT bind — these processor
/// defaults are what actually clamp the resized area.
pub const FACTOR: u32 = 32; // patch_size(16) * merge_size(2)
pub const MIN_PIXELS: u32 = 56 * 56; // 3136
pub const MAX_PIXELS: u32 = 14 * 14 * 4 * 1280; // 1003520

pub const Resized = struct { h: u32, w: u32 };

/// Python `round()` is round-half-to-EVEN (banker's rounding); Zig's
/// `std.math.round` is round-half-away-from-zero. `_smart_resize_image` uses the
/// Python builtin, so we must match it for byte-faithful grids. Inputs here are
/// always positive (image dimensions / factor).
fn roundHalfEven(x: f64) f64 {
    const fl = std.math.floor(x);
    const frac = x - fl;
    if (frac < 0.5) return fl;
    if (frac > 0.5) return fl + 1.0;
    // Exactly .5 → choose the even neighbour.
    return if (@mod(fl, 2.0) == 0.0) fl else fl + 1.0;
}

/// Faithful port of `_smart_resize_image` (processing_qwen3_vl.py:93-116): snap
/// each side to a multiple of `factor`, then rescale into [min_pixels, max_pixels]
/// preserving aspect ratio. Returns the resized (H, W) in pixels.
pub fn smartResizeImage(height: u32, width: u32, factor: u32, min_pixels: u32, max_pixels: u32) Resized {
    const fh: f64 = @floatFromInt(height);
    const fw: f64 = @floatFromInt(width);
    const ff: f64 = @floatFromInt(factor);
    const fmin: f64 = @floatFromInt(min_pixels);
    const fmax: f64 = @floatFromInt(max_pixels);

    var h_bar = roundHalfEven(fh / ff) * ff;
    var w_bar = roundHalfEven(fw / ff) * ff;
    if (h_bar * w_bar > fmax) {
        const beta = @sqrt((fh * fw) / fmax);
        h_bar = @max(ff, std.math.floor(fh / beta / ff) * ff);
        w_bar = @max(ff, std.math.floor(fw / beta / ff) * ff);
    } else if (h_bar * w_bar < fmin) {
        const beta = @sqrt(fmin / (fh * fw));
        h_bar = std.math.ceil(fh * beta / ff) * ff;
        w_bar = std.math.ceil(fw * beta / ff) * ff;
    }
    return .{ .h = @intFromFloat(h_bar), .w = @intFromFloat(w_bar) };
}

/// Convenience wrapper using the Qwen3VL processor defaults.
pub fn smartResizeDefault(height: u32, width: u32) Resized {
    return smartResizeImage(height, width, FACTOR, MIN_PIXELS, MAX_PIXELS);
}

/// Number of LLM image-pad tokens an image of resized (H,W) expands to.
pub fn imageTokenCount(resized: Resized, patch: u32, merge: u32) u32 {
    const gh = resized.h / patch;
    const gw = resized.w / patch;
    return (gh / merge) * (gw / merge);
}

/// Build the processor's `pixel_values` [N, C*tps*ps*ps] from a normalized CHW
/// image, in merge-block token order with feature layout [C, tps, py, px] — the
/// exact ordering of mlx-vlm's `_process_one` transpose. `img_chw` is
/// `[C, rh, rw]` already rescaled+normalized ((x/255−0.5)/0.5). The temporal
/// axis duplicates the single frame `tps` times. Caller owns the result.
pub fn buildPixelValues(
    out: []f32,
    img_chw: []const f32,
    C: u32,
    rh: u32,
    rw: u32,
    patch: u32,
    tps: u32,
    merge: u32,
) void {
    const gh = rh / patch;
    const gw = rw / patch;
    const mh = gh / merge;
    const mw = gw / merge;
    const feat = C * tps * patch * patch;
    std.debug.assert(out.len == @as(usize, gh) * gw * feat);

    const plane: usize = @as(usize, rh) * rw;
    var token: usize = 0;
    var bh: u32 = 0;
    while (bh < mh) : (bh += 1) {
        var bw: u32 = 0;
        while (bw < mw) : (bw += 1) {
            var ir: u32 = 0;
            while (ir < merge) : (ir += 1) {
                var ic: u32 = 0;
                while (ic < merge) : (ic += 1) {
                    const row = bh * merge + ir;
                    const col = bw * merge + ic;
                    const base = token * feat;
                    var f: usize = 0;
                    var c: u32 = 0;
                    while (c < C) : (c += 1) {
                        var tt: u32 = 0;
                        while (tt < tps) : (tt += 1) {
                            var py: u32 = 0;
                            while (py < patch) : (py += 1) {
                                const y = row * patch + py;
                                var px: u32 = 0;
                                while (px < patch) : (px += 1) {
                                    const x = col * patch + px;
                                    // Temporal axis duplicates the single frame.
                                    out[base + f] = img_chw[@as(usize, c) * plane + @as(usize, y) * rw + x];
                                    f += 1;
                                }
                            }
                        }
                    }
                    token += 1;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Qwen3-VL ViT encoder. Mirrors mlx-vlm qwen3_vl/vision.py for a SINGLE still
// image (grid_thw = [[1, grid_h, grid_w]]): full bidirectional attention over all
// patches (cu_seqlens trivial), no windowing, no DeepStack. Dense bf16 — even on
// 4-bit checkpoints the vision tower ships bf16, so no quantized-linear path.
//
// forward(patches [N, C*tps*ps*ps], grid_h, grid_w) → [1, N/merge², out_hidden].
// `patches` is the processor's `pixel_values`, already in merge-block token order
// with feature layout [C, tps, py, px] (see _process_one). The text trunk then
// splices these embeddings at the image-pad token positions.
// ─────────────────────────────────────────────────────────────────────────────

const QBlock = struct {
    norm1_w: mlx.mlx_array,
    norm1_b: mlx.mlx_array,
    norm2_w: mlx.mlx_array,
    norm2_b: mlx.mlx_array,
    qkv_w: mlx.mlx_array,
    qkv_b: mlx.mlx_array,
    proj_w: mlx.mlx_array,
    proj_b: mlx.mlx_array,
    fc1_w: mlx.mlx_array,
    fc1_b: mlx.mlx_array,
    fc2_w: mlx.mlx_array,
    fc2_b: mlx.mlx_array,
};

pub const QwenVision = struct {
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,

    depth: u32,
    hidden: u32,
    heads: u32,
    head_dim: u32,
    merge: u32,
    num_grid_per_side: u32, // = sqrt(num_position_embeddings), e.g. 48
    out_hidden: u32,
    rope_theta: f64 = 10000.0, // VisionRotaryEmbedding default

    patch_w: mlx.mlx_array, // [hidden, C*tps*ps*ps] (transposed from conv layout)
    patch_b: mlx.mlx_array, // [hidden]
    pos_embed: mlx.mlx_array, // [num_position_embeddings, hidden]
    blocks: []QBlock,
    merger_norm_w: mlx.mlx_array,
    merger_norm_b: mlx.mlx_array,
    merger_fc1_w: mlx.mlx_array,
    merger_fc1_b: mlx.mlx_array,
    merger_fc2_w: mlx.mlx_array,
    merger_fc2_b: mlx.mlx_array,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !QwenVision {
        const s = mlx.mlx_default_gpu_stream_new();
        var buf: [256]u8 = undefined;

        const must = struct {
            fn f(w: *const Weights, b: *[256]u8, name: []const u8) !mlx.mlx_array {
                return getWeightLocal(w, b, name) orelse {
                    log.warn("MISSING QWEN VISION WEIGHT: {s}\n", .{name});
                    return error.MissingVisionWeights;
                };
            }
        }.f;

        // patch_embed.proj.weight is stored MLX-conv layout [out, kT, ps, ps, Cin].
        // Transpose to [out, Cin, kT, ps, ps] then flatten to [out, Cin*kT*ps*ps]
        // so it matches the processor's [C, tps, py, px] pixel_values feature order,
        // turning the full-window Conv3d into a plain Linear.
        const conv_w = try must(weights, &buf, "vision_tower.patch_embed.proj.weight");
        const cw_shape = mlx.getShape(conv_w);
        std.debug.assert(cw_shape.len == 5);
        const out_c = cw_shape[0];
        const flat_in = cw_shape[1] * cw_shape[2] * cw_shape[3] * cw_shape[4];
        var patch_w = mlx.mlx_array_new();
        {
            const perm = [_]c_int{ 0, 4, 1, 2, 3 }; // [out,kT,ps,ps,C] → [out,C,kT,ps,ps]
            var transposed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(transposed);
            try mlx.check(mlx.mlx_transpose_axes(&transposed, conv_w, &perm, 5, s));
            const flat_shape = [_]c_int{ out_c, flat_in };
            try mlx.check(mlx.mlx_reshape(&patch_w, transposed, &flat_shape, 2, s));
        }
        const patch_b = try must(weights, &buf, "vision_tower.patch_embed.proj.bias");
        const pos_embed = try must(weights, &buf, "vision_tower.pos_embed.weight");

        const depth = config.qv_depth;
        var blocks = try allocator.alloc(QBlock, depth);
        errdefer allocator.free(blocks);
        for (0..depth) |i| {
            blocks[i] = .{
                .norm1_w = try must(weights, &buf, fmtLayer(&buf, i, "norm1.weight")),
                .norm1_b = try must(weights, &buf, fmtLayer(&buf, i, "norm1.bias")),
                .norm2_w = try must(weights, &buf, fmtLayer(&buf, i, "norm2.weight")),
                .norm2_b = try must(weights, &buf, fmtLayer(&buf, i, "norm2.bias")),
                .qkv_w = try must(weights, &buf, fmtLayer(&buf, i, "attn.qkv.weight")),
                .qkv_b = try must(weights, &buf, fmtLayer(&buf, i, "attn.qkv.bias")),
                .proj_w = try must(weights, &buf, fmtLayer(&buf, i, "attn.proj.weight")),
                .proj_b = try must(weights, &buf, fmtLayer(&buf, i, "attn.proj.bias")),
                .fc1_w = try must(weights, &buf, fmtLayer(&buf, i, "mlp.linear_fc1.weight")),
                .fc1_b = try must(weights, &buf, fmtLayer(&buf, i, "mlp.linear_fc1.bias")),
                .fc2_w = try must(weights, &buf, fmtLayer(&buf, i, "mlp.linear_fc2.weight")),
                .fc2_b = try must(weights, &buf, fmtLayer(&buf, i, "mlp.linear_fc2.bias")),
            };
        }

        return QwenVision{
            .s = s,
            .allocator = allocator,
            .depth = depth,
            .hidden = config.qv_hidden,
            .heads = config.qv_heads,
            .head_dim = config.qv_head_dim,
            .merge = config.qv_merge,
            .num_grid_per_side = @intFromFloat(@sqrt(@as(f64, @floatFromInt(config.qv_num_pos_emb)))),
            .out_hidden = config.qv_out_hidden,
            .patch_w = patch_w,
            .patch_b = patch_b,
            .pos_embed = pos_embed,
            .blocks = blocks,
            .merger_norm_w = try must(weights, &buf, "vision_tower.merger.norm.weight"),
            .merger_norm_b = try must(weights, &buf, "vision_tower.merger.norm.bias"),
            .merger_fc1_w = try must(weights, &buf, "vision_tower.merger.linear_fc1.weight"),
            .merger_fc1_b = try must(weights, &buf, "vision_tower.merger.linear_fc1.bias"),
            .merger_fc2_w = try must(weights, &buf, "vision_tower.merger.linear_fc2.weight"),
            .merger_fc2_b = try must(weights, &buf, "vision_tower.merger.linear_fc2.bias"),
        };
    }

    pub fn deinit(self: *QwenVision) void {
        _ = mlx.mlx_array_free(self.patch_w);
        self.allocator.free(self.blocks);
    }

    fn bf16Scalar(self: *QwenVision, v: f32) mlx.mlx_array {
        const f = mlx.mlx_array_new_float(v);
        defer _ = mlx.mlx_array_free(f);
        var out = mlx.mlx_array_new();
        _ = mlx.mlx_astype(&out, f, .bfloat16, self.s);
        return out;
    }

    /// y = x · Wᵀ (+ bias). W is [out, in], x is [..., in].
    fn denseLinear(self: *QwenVision, x: mlx.mlx_array, w: mlx.mlx_array, b: ?mlx.mlx_array) !mlx.mlx_array {
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        try mlx.check(mlx.mlx_transpose(&wt, w, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&out, x, wt, self.s));
        if (b) |bv| {
            var biased = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&biased, out, bv, self.s));
            _ = mlx.mlx_array_free(out);
            out = biased;
        }
        return out;
    }

    fn layerNorm6(self: *QwenVision, x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_layer_norm(&out, x, w, b, 1e-6, self.s));
        return out;
    }

    /// nn.GELU(approx="tanh"): 0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³))).
    fn geluTanh(self: *QwenVision, x: mlx.mlx_array) !mlx.mlx_array {
        const c_coeff = self.bf16Scalar(0.7978845608028654);
        defer _ = mlx.mlx_array_free(c_coeff);
        const c_inner = self.bf16Scalar(0.044715);
        defer _ = mlx.mlx_array_free(c_inner);
        const c_three = self.bf16Scalar(3.0);
        defer _ = mlx.mlx_array_free(c_three);
        const c_one = self.bf16Scalar(1.0);
        defer _ = mlx.mlx_array_free(c_one);
        const c_half = self.bf16Scalar(0.5);
        defer _ = mlx.mlx_array_free(c_half);

        var x3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x3);
        try mlx.check(mlx.mlx_power(&x3, x, c_three, self.s));
        var inner = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inner);
        try mlx.check(mlx.mlx_multiply(&inner, c_inner, x3, self.s));
        var sum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_add(&sum, x, inner, self.s));
        var scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_multiply(&scaled, c_coeff, sum, self.s));
        var th = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(th);
        try mlx.check(mlx.mlx_tanh(&th, scaled, self.s));
        var onep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(onep);
        try mlx.check(mlx.mlx_add(&onep, c_one, th, self.s));
        var xt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xt);
        try mlx.check(mlx.mlx_multiply(&xt, x, onep, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&out, xt, c_half, self.s));
        return out;
    }

    /// nn.GELU() exact: 0.5·x·(1+erf(x/√2)). Used by the PatchMerger.
    fn geluExact(self: *QwenVision, x: mlx.mlx_array) !mlx.mlx_array {
        const inv_sqrt2 = self.bf16Scalar(0.7071067811865476);
        defer _ = mlx.mlx_array_free(inv_sqrt2);
        const c_one = self.bf16Scalar(1.0);
        defer _ = mlx.mlx_array_free(c_one);
        const c_half = self.bf16Scalar(0.5);
        defer _ = mlx.mlx_array_free(c_half);
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_multiply(&t, x, inv_sqrt2, self.s));
        var e = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(e);
        try mlx.check(mlx.mlx_erf(&e, t, self.s));
        var onep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(onep);
        try mlx.check(mlx.mlx_add(&onep, c_one, e, self.s));
        var xt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xt);
        try mlx.check(mlx.mlx_multiply(&xt, x, onep, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&out, xt, c_half, self.s));
        return out;
    }

    /// rotate_half over the last dim (split at dim/2): concat(-x2, x1).
    fn rotateHalf(self: *QwenVision, x: mlx.mlx_array, n: c_int, hd: c_int) !mlx.mlx_array {
        const half = @divExact(hd, 2);
        const strides = [_]c_int{ 1, 1, 1 };
        const start1 = [_]c_int{ 0, 0, half };
        const stop1 = [_]c_int{ n, self.headsC(), hd };
        var x2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x2);
        try mlx.check(mlx.mlx_slice(&x2, x, &start1, 3, &stop1, 3, &strides, 3, self.s));
        const start0 = [_]c_int{ 0, 0, 0 };
        const stop0 = [_]c_int{ n, self.headsC(), half };
        var x1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x1);
        try mlx.check(mlx.mlx_slice(&x1, x, &start0, 3, &stop0, 3, &strides, 3, self.s));
        var neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(neg);
        try mlx.check(mlx.mlx_negative(&neg, x2, self.s));
        const arrs = [_]mlx.mlx_array{ neg, x1 };
        const vec = mlx.mlx_vector_array_new_data(&arrs, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, -1, self.s));
        return out;
    }

    inline fn headsC(self: *QwenVision) c_int {
        return @intCast(self.heads);
    }

    /// Apply vision 2D RoPE: out = x·cos + rotate_half(x)·sin. x is [N, heads, hd],
    /// cos/sin are [N, 1, hd] (broadcast over heads).
    fn applyVisionRope(self: *QwenVision, x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, n: c_int, hd: c_int) !mlx.mlx_array {
        var xcos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xcos);
        try mlx.check(mlx.mlx_multiply(&xcos, x, cos, self.s));
        const rh = try self.rotateHalf(x, n, hd);
        defer _ = mlx.mlx_array_free(rh);
        var rsin = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rsin);
        try mlx.check(mlx.mlx_multiply(&rsin, rh, sin, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&out, xcos, rsin, self.s));
        return out;
    }

    /// Build per-token vision-RoPE cos/sin tables [N, 1, head_dim] (bf16) for a
    /// single image, in merge-block token order. VisionRotaryEmbedding(head_dim/2):
    /// 16 freqs; each token gets [h_emb(16) ‖ w_emb(16)], tiled ×2 over head_dim.
    fn buildVisionRope(self: *QwenVision, grid_h: u32, grid_w: u32) !struct { cos: mlx.mlx_array, sin: mlx.mlx_array } {
        const hd: usize = self.head_dim;
        const half = hd / 2; // 32
        const nfreq = half / 2; // 16
        const merge = self.merge;
        const mh = grid_h / merge;
        const mw = grid_w / merge;
        const n: usize = grid_h * grid_w;

        var inv_freq = try self.allocator.alloc(f64, nfreq);
        defer self.allocator.free(inv_freq);
        for (0..nfreq) |k| {
            const exp = -@as(f64, @floatFromInt(2 * k)) / @as(f64, @floatFromInt(half));
            inv_freq[k] = std.math.pow(f64, self.rope_theta, exp);
        }

        var cos_buf = try self.allocator.alloc(f32, n * hd);
        defer self.allocator.free(cos_buf);
        var sin_buf = try self.allocator.alloc(f32, n * hd);
        defer self.allocator.free(sin_buf);

        var token: usize = 0;
        var bh: usize = 0;
        while (bh < mh) : (bh += 1) {
            var bw: usize = 0;
            while (bw < mw) : (bw += 1) {
                var ir: usize = 0;
                while (ir < merge) : (ir += 1) {
                    var ic: usize = 0;
                    while (ic < merge) : (ic += 1) {
                        const row: f64 = @floatFromInt(bh * merge + ir);
                        const col: f64 = @floatFromInt(bw * merge + ic);
                        const o = token * hd;
                        for (0..nfreq) |k| {
                            const ah = row * inv_freq[k];
                            const aw = col * inv_freq[k];
                            // emb = [h(16), w(16)] then tile ×2 over head_dim.
                            cos_buf[o + k] = @floatCast(@cos(ah));
                            cos_buf[o + nfreq + k] = @floatCast(@cos(aw));
                            cos_buf[o + half + k] = @floatCast(@cos(ah));
                            cos_buf[o + half + nfreq + k] = @floatCast(@cos(aw));
                            sin_buf[o + k] = @floatCast(@sin(ah));
                            sin_buf[o + nfreq + k] = @floatCast(@sin(aw));
                            sin_buf[o + half + k] = @floatCast(@sin(ah));
                            sin_buf[o + half + nfreq + k] = @floatCast(@sin(aw));
                        }
                        token += 1;
                    }
                }
            }
        }

        const shape = [_]c_int{ @intCast(n), 1, @intCast(hd) };
        const cos_f = mlx.mlx_array_new_data(cos_buf.ptr, &shape, 3, .float32);
        defer _ = mlx.mlx_array_free(cos_f);
        const sin_f = mlx.mlx_array_new_data(sin_buf.ptr, &shape, 3, .float32);
        defer _ = mlx.mlx_array_free(sin_f);
        var cos = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&cos, cos_f, .bfloat16, self.s));
        var sin = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&sin, sin_f, .bfloat16, self.s));
        return .{ .cos = cos, .sin = sin };
    }

    /// Interpolated learned position embeddings, merge-block order → [N, hidden] bf16.
    fn posEmbedInterpolate(self: *QwenVision, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const G = self.num_grid_per_side;
        const merge = self.merge;
        const mh = grid_h / merge;
        const mw = grid_w / merge;
        const n: usize = grid_h * grid_w;

        // Per-axis bilinear endpoints from linspace(0, G-1, grid).
        const Axis = struct { floor: []i32, ceil: []i32, frac: []f64 };
        const mkAxis = struct {
            fn f(a: std.mem.Allocator, len: u32, gside: u32) !Axis {
                const fl = try a.alloc(i32, len);
                const cl = try a.alloc(i32, len);
                const fr = try a.alloc(f64, len);
                const last: f64 = @floatFromInt(gside - 1);
                for (0..len) |i| {
                    const v: f64 = if (len == 1) 0.0 else last * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(len - 1));
                    const flo: i32 = @intFromFloat(v); // truncation toward 0 == floor for v>=0
                    fl[i] = flo;
                    cl[i] = @min(flo + 1, @as(i32, @intCast(gside - 1)));
                    fr[i] = v - @as(f64, @floatFromInt(flo));
                }
                return .{ .floor = fl, .ceil = cl, .frac = fr };
            }
        }.f;

        const ha = try mkAxis(self.allocator, grid_h, G);
        defer {
            self.allocator.free(ha.floor);
            self.allocator.free(ha.ceil);
            self.allocator.free(ha.frac);
        }
        const wa = try mkAxis(self.allocator, grid_w, G);
        defer {
            self.allocator.free(wa.floor);
            self.allocator.free(wa.ceil);
            self.allocator.free(wa.frac);
        }

        // Four corner index arrays + weights, emitted directly in merge-block order.
        var idx: [4][]i32 = undefined;
        var wgt: [4][]f32 = undefined;
        inline for (0..4) |c| {
            idx[c] = try self.allocator.alloc(i32, n);
            wgt[c] = try self.allocator.alloc(f32, n);
        }
        defer inline for (0..4) |c| {
            self.allocator.free(idx[c]);
            self.allocator.free(wgt[c]);
        };

        var token: usize = 0;
        var bh: usize = 0;
        while (bh < mh) : (bh += 1) {
            var bw: usize = 0;
            while (bw < mw) : (bw += 1) {
                var ir: usize = 0;
                while (ir < merge) : (ir += 1) {
                    var ic: usize = 0;
                    while (ic < merge) : (ic += 1) {
                        const row = bh * merge + ir;
                        const col = bw * merge + ic;
                        const hf = ha.floor[row];
                        const hc = ha.ceil[row];
                        const wf = wa.floor[col];
                        const wc = wa.ceil[col];
                        const dh = ha.frac[row];
                        const dw = wa.frac[col];
                        const gi: i32 = @intCast(G);
                        idx[0][token] = hf * gi + wf;
                        idx[1][token] = hf * gi + wc;
                        idx[2][token] = hc * gi + wf;
                        idx[3][token] = hc * gi + wc;
                        wgt[0][token] = @floatCast((1.0 - dh) * (1.0 - dw));
                        wgt[1][token] = @floatCast((1.0 - dh) * dw);
                        wgt[2][token] = @floatCast(dh * (1.0 - dw));
                        wgt[3][token] = @floatCast(dh * dw);
                        token += 1;
                    }
                }
            }
        }

        const idx_shape = [_]c_int{@intCast(n)};
        const wshape = [_]c_int{ @intCast(n), 1 };
        var acc = mlx.mlx_array_new();
        var first = true;
        inline for (0..4) |c| {
            const ix = mlx.mlx_array_new_data(idx[c].ptr, &idx_shape, 1, .int32);
            defer _ = mlx.mlx_array_free(ix);
            var gathered = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gathered);
            try mlx.check(mlx.mlx_take_axis(&gathered, self.pos_embed, ix, 0, self.s)); // [N, hidden]
            const wf = mlx.mlx_array_new_data(wgt[c].ptr, &wshape, 2, .float32);
            defer _ = mlx.mlx_array_free(wf);
            var wbf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wbf);
            try mlx.check(mlx.mlx_astype(&wbf, wf, .bfloat16, self.s));
            var weighted = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(weighted);
            try mlx.check(mlx.mlx_multiply(&weighted, gathered, wbf, self.s));
            if (first) {
                // astype-copy so `acc` owns an array independent of `weighted`'s defer.
                try mlx.check(mlx.mlx_astype(&acc, weighted, .bfloat16, self.s));
                first = false;
            } else {
                var sum = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&sum, acc, weighted, self.s));
                _ = mlx.mlx_array_free(acc);
                acc = sum;
            }
        }
        return acc;
    }

    /// Self-attention over all patches (single image → full bidirectional).
    fn attention(self: *QwenVision, normed: mlx.mlx_array, blk: QBlock, cos: mlx.mlx_array, sin: mlx.mlx_array, n: c_int) !mlx.mlx_array {
        const hd: c_int = @intCast(self.head_dim);
        const heads: c_int = self.headsC();
        const qkv = try self.denseLinear(normed, blk.qkv_w, blk.qkv_b); // [N, 3*hidden]
        defer _ = mlx.mlx_array_free(qkv);
        // [N, 3, heads, hd]
        var r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r);
        const rshape = [_]c_int{ n, 3, heads, hd };
        try mlx.check(mlx.mlx_reshape(&r, qkv, &rshape, 4, self.s));

        const sl_strides = [_]c_int{ 1, 1, 1, 1 };
        var parts: [3]mlx.mlx_array = undefined;
        inline for (0..3) |j| {
            const start = [_]c_int{ 0, @intCast(j), 0, 0 };
            const stop = [_]c_int{ n, @intCast(j + 1), heads, hd };
            var sliced = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sliced);
            try mlx.check(mlx.mlx_slice(&sliced, r, &start, 4, &stop, 4, &sl_strides, 4, self.s));
            const flat = [_]c_int{ n, heads, hd };
            var part = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&part, sliced, &flat, 3, self.s));
            parts[j] = part; // [N, heads, hd]
        }
        defer for (parts) |p| {
            _ = mlx.mlx_array_free(p);
        };

        const q_rope = try self.applyVisionRope(parts[0], cos, sin, n, hd);
        defer _ = mlx.mlx_array_free(q_rope);
        const k_rope = try self.applyVisionRope(parts[1], cos, sin, n, hd);
        defer _ = mlx.mlx_array_free(k_rope);

        // [N, heads, hd] → [heads, N, hd], in fp32 to mirror the reference's
        // fused SDPA (bf16 in/out, fp32 internal accumulation). Computing the
        // score/softmax/context core in bf16 leaves sparse ~0.9 outliers on
        // high-variance channels; fp32 internals match the reference far closer.
        const perm = [_]c_int{ 1, 0, 2 };
        var qh = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qh);
        try mlx.check(mlx.mlx_transpose_axes(&qh, q_rope, &perm, 3, self.s));
        try mlx.check(mlx.mlx_astype(&qh, qh, .float32, self.s));
        var kh = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kh);
        try mlx.check(mlx.mlx_transpose_axes(&kh, k_rope, &perm, 3, self.s));
        try mlx.check(mlx.mlx_astype(&kh, kh, .float32, self.s));
        var vh = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(vh);
        try mlx.check(mlx.mlx_transpose_axes(&vh, parts[2], &perm, 3, self.s));
        try mlx.check(mlx.mlx_astype(&vh, vh, .float32, self.s));

        // scores = q @ k^T * scale
        const kperm = [_]c_int{ 0, 2, 1 };
        var kt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kt);
        try mlx.check(mlx.mlx_transpose_axes(&kt, kh, &kperm, 3, self.s));
        var scores = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scores);
        try mlx.check(mlx.mlx_matmul(&scores, qh, kt, self.s));
        const scale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim))));
        defer _ = mlx.mlx_array_free(scale);
        var scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_multiply(&scaled, scores, scale, self.s));
        var probs = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(probs);
        try mlx.check(mlx.mlx_softmax_axis(&probs, scaled, -1, true, self.s));
        var ctx32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ctx32);
        try mlx.check(mlx.mlx_matmul(&ctx32, probs, vh, self.s)); // [heads, N, hd] f32
        var ctx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ctx);
        try mlx.check(mlx.mlx_astype(&ctx, ctx32, .bfloat16, self.s));

        // [heads, N, hd] → [N, heads*hd]
        const operm = [_]c_int{ 1, 0, 2 };
        var ctt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ctt);
        try mlx.check(mlx.mlx_transpose_axes(&ctt, ctx, &operm, 3, self.s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const fshape = [_]c_int{ n, @intCast(self.hidden) };
        try mlx.check(mlx.mlx_reshape(&flat, ctt, &fshape, 2, self.s));
        return self.denseLinear(flat, blk.proj_w, blk.proj_b);
    }

    /// Encode one image. `patches` = pixel_values [N, C*tps*ps*ps] (merge order).
    pub fn forward(self: *QwenVision, patches: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const n: c_int = @intCast(grid_h * grid_w);
        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&x, patches, .bfloat16, self.s));

        // patch_embed (Linear) + interpolated pos_embed.
        {
            const pe = try self.denseLinear(x, self.patch_w, self.patch_b);
            _ = mlx.mlx_array_free(x);
            x = pe;
        }
        {
            const pos = try self.posEmbedInterpolate(grid_h, grid_w);
            defer _ = mlx.mlx_array_free(pos);
            var added = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&added, x, pos, self.s));
            _ = mlx.mlx_array_free(x);
            x = added;
        }

        const rope = try self.buildVisionRope(grid_h, grid_w);
        defer _ = mlx.mlx_array_free(rope.cos);
        defer _ = mlx.mlx_array_free(rope.sin);

        for (self.blocks) |blk| {
            // Attention residual.
            {
                const normed = try self.layerNorm6(x, blk.norm1_w, blk.norm1_b);
                defer _ = mlx.mlx_array_free(normed);
                const attn = try self.attention(normed, blk, rope.cos, rope.sin, n);
                defer _ = mlx.mlx_array_free(attn);
                var h = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h, x, attn, self.s));
                _ = mlx.mlx_array_free(x);
                x = h;
            }
            // MLP residual.
            {
                const normed = try self.layerNorm6(x, blk.norm2_w, blk.norm2_b);
                defer _ = mlx.mlx_array_free(normed);
                const fc1 = try self.denseLinear(normed, blk.fc1_w, blk.fc1_b);
                defer _ = mlx.mlx_array_free(fc1);
                const act = try self.geluTanh(fc1);
                defer _ = mlx.mlx_array_free(act);
                const fc2 = try self.denseLinear(act, blk.fc2_w, blk.fc2_b);
                defer _ = mlx.mlx_array_free(fc2);
                var h = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h, x, fc2, self.s));
                _ = mlx.mlx_array_free(x);
                x = h;
            }
        }

        // Merger: LayerNorm(hidden) → reshape [N/merge², hidden·merge²] → fc2(gelu(fc1)).
        const normed = try self.layerNorm6(x, self.merger_norm_w, self.merger_norm_b);
        _ = mlx.mlx_array_free(x);
        defer _ = mlx.mlx_array_free(normed);
        const merge2: c_int = @intCast(self.merge * self.merge);
        const n_merged = @divExact(n, merge2);
        var grouped = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(grouped);
        const gshape = [_]c_int{ n_merged, @intCast(self.hidden * self.merge * self.merge) };
        try mlx.check(mlx.mlx_reshape(&grouped, normed, &gshape, 2, self.s));
        const m1 = try self.denseLinear(grouped, self.merger_fc1_w, self.merger_fc1_b);
        defer _ = mlx.mlx_array_free(m1);
        const ma = try self.geluExact(m1);
        defer _ = mlx.mlx_array_free(ma);
        const m2 = try self.denseLinear(ma, self.merger_fc2_w, self.merger_fc2_b);
        defer _ = mlx.mlx_array_free(m2);
        // [N_merged, out_hidden] → [1, N_merged, out_hidden].
        var out = mlx.mlx_array_new();
        const oshape = [_]c_int{ 1, n_merged, @intCast(self.out_hidden) };
        try mlx.check(mlx.mlx_reshape(&out, m2, &oshape, 3, self.s));
        return out;
    }
};

fn getWeightLocal(weights: *const Weights, buf: *[256]u8, name: []const u8) ?mlx.mlx_array {
    _ = buf;
    return weights.get(name);
}

fn fmtLayer(buf: *[256]u8, layer: usize, suffix: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "vision_tower.blocks.{d}.{s}", .{ layer, suffix }) catch unreachable;
}

test "qwen smart_resize matches reference table" {
    const cases = [_]struct { h: u32, w: u32, eh: u32, ew: u32 }{
        .{ .h = 768, .w = 768, .eh = 768, .ew = 768 }, // in-range, 48x48 grid → 576 tokens
        .{ .h = 1024, .w = 768, .eh = 1024, .ew = 768 }, // 64x48 → 768 tokens
        .{ .h = 480, .w = 640, .eh = 480, .ew = 640 }, // 30x40 → 300 tokens
        .{ .h = 4000, .w = 3000, .eh = 1152, .ew = 864 }, // > max → 72x54 → 972 tokens
        .{ .h = 100, .w = 100, .eh = 96, .ew = 96 }, // round(3.125)=3 → 96, 6x6 grid
        .{ .h = 40, .w = 40, .eh = 64, .ew = 64 }, // < min → upscaled to 64x64
    };
    for (cases) |c| {
        const r = smartResizeDefault(c.h, c.w);
        try std.testing.expectEqual(c.eh, r.h);
        try std.testing.expectEqual(c.ew, r.w);
    }
}

test "qwen buildPixelValues merge-order + [C,tps,py,px] feature layout" {
    const a = std.testing.allocator;
    // 2x2 image, patch=1, merge=2, tps=2, C=3. img[c,y,x] = c*100 + y*10 + x.
    const C: u32 = 3;
    const rh: u32 = 2;
    const rw: u32 = 2;
    var img: [3 * 2 * 2]f32 = undefined;
    for (0..C) |c| for (0..rh) |y| for (0..rw) |x| {
        img[c * 4 + y * 2 + x] = @floatFromInt(c * 100 + y * 10 + x);
    };
    const pv = try a.alloc(f32, 4 * 6);
    defer a.free(pv);
    buildPixelValues(pv, &img, C, rh, rw, 1, 2, 2);
    // 4 tokens (merge-block order over a single 2x2 block) × feat 6 (=C*tps*1*1).
    const expect = [_]f32{
        0,   0,   100, 100, 200, 200, // token0 row0col0
        1,   1,   101, 101, 201, 201, // token1 row0col1
        10,  10,  110, 110, 210, 210, // token2 row1col0
        11,  11,  111, 111, 211, 211, // token3 row1col1
    };
    try std.testing.expectEqualSlices(f32, &expect, pv);
}

test "qwen image token count" {
    try std.testing.expectEqual(@as(u32, 576), imageTokenCount(.{ .h = 768, .w = 768 }, 16, 2));
    try std.testing.expectEqual(@as(u32, 972), imageTokenCount(.{ .h = 1152, .w = 864 }, 16, 2));
    try std.testing.expectEqual(@as(u32, 300), imageTokenCount(.{ .h = 480, .w = 640 }, 16, 2));
}

test "qwen roundHalfEven matches python banker's rounding" {
    try std.testing.expectEqual(@as(f64, 0.0), roundHalfEven(0.5)); // → even 0
    try std.testing.expectEqual(@as(f64, 2.0), roundHalfEven(1.5)); // → even 2
    try std.testing.expectEqual(@as(f64, 2.0), roundHalfEven(2.5)); // → even 2
    try std.testing.expectEqual(@as(f64, 4.0), roundHalfEven(3.5)); // → even 4
    try std.testing.expectEqual(@as(f64, 3.0), roundHalfEven(3.125));
    try std.testing.expectEqual(@as(f64, 94.0), roundHalfEven(93.75));
}

fn readBinF32(io: std.Io, alloc: std.mem.Allocator, path: []const u8) ![]f32 {
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var rs = file.reader(io, &buf);
    const bytes = try rs.interface.allocRemaining(alloc, .limited(256 * 1024 * 1024));
    // Reinterpret the raw little-endian float32 payload.
    const n = bytes.len / 4;
    const out = try alloc.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 4]);
    alloc.free(bytes);
    return out;
}

// Live parity vs the mlx-vlm reference vision tower. Feeds the REFERENCE's own
// pixel_values straight into QwenVision (isolating the ViT math from the
// preprocessing), then compares post-merger embeddings. Build the fixture first:
//   python3 tests/build_qwen_vision_fixture.py --model <dir> --image <img> --out <fix>
// then run via tests/test_qwen_vision_parity.sh (sets the env vars below).
test "qwen vision parity vs reference embeddings (QWEN_VISION_TEST_MODEL)" {
    const model_raw = std.c.getenv("QWEN_VISION_TEST_MODEL") orelse return error.SkipZigTest;
    const fix_raw = std.c.getenv("QWEN_VISION_FIXTURE") orelse return error.SkipZigTest;
    const gh_raw = std.c.getenv("QV_GH") orelse return error.SkipZigTest;
    const gw_raw = std.c.getenv("QV_GW") orelse return error.SkipZigTest;
    const dir = std.mem.sliceTo(model_raw, 0);
    const fix = std.mem.sliceTo(fix_raw, 0);
    if (dir.len == 0 or fix.len == 0) return error.SkipZigTest;
    const grid_h = try std.fmt.parseInt(u32, std.mem.sliceTo(gh_raw, 0), 10);
    const grid_w = try std.fmt.parseInt(u32, std.mem.sliceTo(gw_raw, 0), 10);

    const alloc = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    const config = try model_mod.parseConfig(io, alloc, dir);
    try std.testing.expect(config.qwen_vision);
    var weights = try model_mod.loadWeightsWithVision(io, alloc, dir);
    defer weights.deinit();
    var qv = try QwenVision.init(alloc, config, &weights);
    defer qv.deinit();

    // Reference pixel_values [N, C*tps*ps*ps] and post-merger embeddings.
    const px_path = try std.fmt.allocPrint(alloc, "{s}/pixel_values.bin", .{fix});
    defer alloc.free(px_path);
    const ref_path = try std.fmt.allocPrint(alloc, "{s}/ref_embeds.bin", .{fix});
    defer alloc.free(ref_path);
    const px = try readBinF32(io, alloc, px_path);
    defer alloc.free(px);
    const ref = try readBinF32(io, alloc, ref_path);
    defer alloc.free(ref);

    const n = grid_h * grid_w;
    const feat = px.len / n;
    const px_shape = [_]c_int{ @intCast(n), @intCast(feat) };
    const px_arr = mlx.mlx_array_new_data(px.ptr, &px_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(px_arr);

    const out = try qv.forward(px_arr, grid_h, grid_w);
    defer _ = mlx.mlx_array_free(out);
    var out_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out_f32);
    try mlx.check(mlx.mlx_astype(&out_f32, out, .float32, qv.s));
    try mlx.check(mlx.mlx_array_eval(out_f32));
    const od = mlx.mlx_array_data_float32(out_f32) orelse return error.TestUnexpectedNullData;

    try std.testing.expectEqual(ref.len, n / 4 * config.qv_out_hidden);
    const dim = config.qv_out_hidden;
    var max_abs: f32 = 0;
    var max_idx: usize = 0;
    var sum_abs: f64 = 0;
    var gt_010: usize = 0;
    var gt_030: usize = 0;
    // Per-token max diff to see clustering (a structural bug clusters by token).
    var worst_tok_diff: f64 = 0;
    var worst_tok: usize = 0;
    var tok_sum: f64 = 0;
    var cur_tok: usize = 0;
    for (0..ref.len) |i| {
        const d = @abs(od[i] - ref[i]);
        if (d > max_abs) {
            max_abs = d;
            max_idx = i;
        }
        if (d > 0.10) gt_010 += 1;
        if (d > 0.30) gt_030 += 1;
        sum_abs += d;
        const tok = i / dim;
        if (tok != cur_tok) {
            if (tok_sum > worst_tok_diff) {
                worst_tok_diff = tok_sum;
                worst_tok = cur_tok;
            }
            tok_sum = 0;
            cur_tok = tok;
        }
        tok_sum += d;
    }
    const mean_abs = sum_abs / @as(f64, @floatFromInt(ref.len));
    std.debug.print("\nqwen vision parity: max_abs={d:.5} mean_abs={d:.6} (N_merged={d}, dim={d})\n", .{ max_abs, mean_abs, n / 4, dim });
    std.debug.print("  max@ token={d} chan={d} ours={d:.4} ref={d:.4}\n", .{ max_idx / dim, max_idx % dim, od[max_idx], ref[max_idx] });
    std.debug.print("  diffs>0.10: {d}/{d}  diffs>0.30: {d}  worst-token meanrow-diff token={d} sum={d:.3}\n", .{ gt_010, ref.len, gt_030, worst_tok, worst_tok_diff / @as(f64, @floatFromInt(dim)) });
    // bf16 ViT through 12 blocks with different reduction orders than the
    // reference: tolerate accumulated rounding + a handful of outliers, catch
    // real bugs (a structural bug blows up mean_abs by 10-100x).
    try std.testing.expect(mean_abs < 0.02);
    try std.testing.expect(gt_030 < ref.len / 1000); // <0.1% of elements may exceed 0.30
}
