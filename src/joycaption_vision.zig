//! JoyCaption / standard HF `LlavaForConditionalGeneration` vision tower.
//!
//! A FIXED-resolution `siglip_vision_model` encoder: patch_embedding (a
//! Conv2d, implemented here as a linear layer on flattened patches — stride
//! equals kernel size, so the two are equivalent), a learned
//! `position_embedding` added directly (no CLS token, no interpolation —
//! the checkpoint's position count always equals our fixed patch grid),
//! then standard pre-norm transformer blocks (2 layernorms, biased q/k/v/out
//! attention, biased fc1/fc2 MLP with `gelu_pytorch_tanh`, no RoPE, no
//! gating). `vision_feature_layer` (-2 for JoyCaption) selects a
//! hidden_states index BEFORE the final encoder layer, so that layer and
//! `post_layernorm` are never run at all. `multi_modal_projector` (linear →
//! erf `gelu` → linear, both biased) bridges vision_hidden → text_hidden.
//! Reference: transformers `SiglipVisionModel` + `LlavaForConditionalGeneration`.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");

const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;

const Lin = struct {
    w: mlx.mlx_array,
    b: mlx.mlx_array,
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

/// Weight lookup under the checkpoint's fixed nesting. Handles are BORROWED
/// from the weights map (owned by `Weights`, freed there).
const NameCtx = struct {
    weights: *const Weights,
    prefix: []const u8,
    buf: *[200]u8,

    fn key(self: NameCtx, comptime fmt: []const u8, args: anytype) []const u8 {
        const body = std.fmt.bufPrint(self.buf[100..], fmt, args) catch unreachable;
        return std.fmt.bufPrint(self.buf[0..100], "{s}{s}", .{ self.prefix, body }) catch unreachable;
    }

    fn opt(self: NameCtx, comptime fmt: []const u8, args: anytype) ?mlx.mlx_array {
        return self.weights.get(self.key(fmt, args));
    }

    fn must(self: NameCtx, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
        return self.opt(fmt, args) orelse {
            log.warn("MISSING JOYCAPTION VISION WEIGHT: {s}\n", .{self.key(fmt, args)});
            return error.MissingVisionWeights;
        };
    }

    fn lin(self: NameCtx, comptime fmt: []const u8, args: anytype) !Lin {
        return .{
            .w = try self.must(fmt ++ ".weight", args),
            .b = try self.must(fmt ++ ".bias", args),
        };
    }
};

pub const JoycaptionVision = struct {
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,

    hidden: u32,
    heads: u32,
    head_dim: u32,
    patch: u32,
    ln_eps: f32,
    out_hidden: u32,

    // Conv2d [hidden, 3, ps, ps] weight, pre-transposed + flattened to
    // [hidden, ps*ps*3] to match `patchify`'s (py, px, c) feature order —
    // an owned array (everything else below is borrowed from `weights`).
    patch_w: mlx.mlx_array,
    patch_b: mlx.mlx_array,
    pos_emb: mlx.mlx_array, // [num_positions, hidden]

    // Only `run_layers` of the checkpoint's `vision_num_layers` blocks ever
    // run — see the module doc comment on `vision_feature_layer`.
    blocks: []Block,

    proj1: Lin, // multi_modal_projector.linear_1: vision_hidden -> text_hidden
    proj2: Lin, // multi_modal_projector.linear_2: text_hidden -> text_hidden

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !JoycaptionVision {
        const s = mlx.mlx_default_gpu_stream_new();
        var tbuf: [200]u8 = undefined;
        var pbuf: [200]u8 = undefined;
        const ctx = NameCtx{ .weights = weights, .prefix = "vision_tower.vision_model.", .buf = &tbuf };
        const pctx = NameCtx{ .weights = weights, .prefix = "multi_modal_projector.", .buf = &pbuf };

        const hidden = config.vision_hidden_size;
        const patch = config.vision_patch_size;

        const patch_w_raw = try ctx.must("embeddings.patch_embedding.weight", .{});
        const patch_b = try ctx.must("embeddings.patch_embedding.bias", .{});
        const pos_emb = try ctx.must("embeddings.position_embedding.weight", .{});

        // [hidden, 3, ps, ps] -> [hidden, ps, ps, 3] -> [hidden, ps*ps*3],
        // matching `patchify`'s (py, px, c) flatten order below.
        var patch_w = mlx.mlx_array_new();
        {
            const perm = [_]c_int{ 0, 2, 3, 1 };
            var wt = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wt);
            try mlx.check(mlx.mlx_transpose_axes(&wt, patch_w_raw, &perm, 4, s));
            const flat_shape = [_]c_int{ @intCast(hidden), @intCast(3 * patch * patch) };
            try mlx.check(mlx.mlx_reshape(&patch_w, wt, &flat_shape, 2, s));
        }

        const num_layers = config.vision_num_layers;
        // hidden_states[k] = the state after k encoder layers (index 0 = the
        // raw embeddings); `vision_feature_layer` indexes that HF list.
        const target_idx: i32 = if (config.vision_feature_layer < 0)
            @as(i32, @intCast(num_layers)) + 1 + config.vision_feature_layer
        else
            config.vision_feature_layer;
        if (target_idx < 0 or target_idx > @as(i32, @intCast(num_layers))) return error.MissingVisionWeights;
        const run_layers: u32 = @intCast(target_idx);

        var blocks = try allocator.alloc(Block, run_layers);
        errdefer allocator.free(blocks);
        for (0..run_layers) |i| {
            blocks[i] = .{
                .ln1_w = try ctx.must("encoder.layers.{d}.layer_norm1.weight", .{i}),
                .ln1_b = try ctx.must("encoder.layers.{d}.layer_norm1.bias", .{i}),
                .ln2_w = try ctx.must("encoder.layers.{d}.layer_norm2.weight", .{i}),
                .ln2_b = try ctx.must("encoder.layers.{d}.layer_norm2.bias", .{i}),
                .q = try ctx.lin("encoder.layers.{d}.self_attn.q_proj", .{i}),
                .k = try ctx.lin("encoder.layers.{d}.self_attn.k_proj", .{i}),
                .v = try ctx.lin("encoder.layers.{d}.self_attn.v_proj", .{i}),
                .out = try ctx.lin("encoder.layers.{d}.self_attn.out_proj", .{i}),
                .fc1 = try ctx.lin("encoder.layers.{d}.mlp.fc1", .{i}),
                .fc2 = try ctx.lin("encoder.layers.{d}.mlp.fc2", .{i}),
            };
        }

        const proj1 = try pctx.lin("linear_1", .{});
        const proj2 = try pctx.lin("linear_2", .{});

        // Batch-eval everything once.
        {
            var eval_list = std.ArrayList(mlx.mlx_array).empty;
            defer eval_list.deinit(allocator);
            const heads = [_]mlx.mlx_array{ patch_w, patch_b, pos_emb, proj1.w, proj1.b, proj2.w, proj2.b };
            try eval_list.appendSlice(allocator, &heads);
            for (blocks) |b| {
                const arrs = [_]mlx.mlx_array{
                    b.ln1_w, b.ln1_b, b.ln2_w, b.ln2_b,
                    b.q.w,   b.q.b,   b.k.w,    b.k.b,
                    b.v.w,   b.v.b,   b.out.w,  b.out.b,
                    b.fc1.w, b.fc1.b, b.fc2.w,  b.fc2.b,
                };
                try eval_list.appendSlice(allocator, &arrs);
            }
            const vec = mlx.mlx_vector_array_new_data(eval_list.items.ptr, eval_list.items.len);
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_eval(vec);
        }

        log.info("Vision encoder: JoyCaption SigLIP ({d}/{d} layers run, hidden={d}, heads={d}, patch={d}, image_size={d})\n", .{
            run_layers, num_layers, hidden, config.vision_num_heads, patch, config.vision_image_size,
        });

        return .{
            .allocator = allocator,
            .s = s,
            .hidden = hidden,
            .heads = config.vision_num_heads,
            .head_dim = hidden / config.vision_num_heads,
            .patch = patch,
            .ln_eps = config.vision_layer_norm_eps,
            .out_hidden = config.hidden_size,
            .patch_w = patch_w,
            .patch_b = patch_b,
            .pos_emb = pos_emb,
            .blocks = blocks,
            .proj1 = proj1,
            .proj2 = proj2,
        };
    }

    pub fn deinit(self: *JoycaptionVision) void {
        _ = mlx.mlx_array_free(self.patch_w);
        self.allocator.free(self.blocks);
        _ = mlx.mlx_stream_free(self.s);
    }

    /// Encode one image. `pixels` is `[1, 3, H, W]` float32 (H == W ==
    /// `config.vision_image_size`, the fixed SigLIP resize target) →
    /// `[1, tokens, text_hidden]`, ready to splice at image-token positions.
    pub fn forward(self: *JoycaptionVision, pixels: mlx.mlx_array) !mlx.mlx_array {
        const pix_shape = mlx.getShape(pixels);
        const batch = pix_shape[0];
        if (batch != 1) return error.UnsupportedBatchSize;
        const height = pix_shape[2];
        const width = pix_shape[3];
        const ps: c_int = @intCast(self.patch);
        const gh = @divExact(height, ps);
        const gw = @divExact(width, ps);
        const n: c_int = gh * gw;

        const patches_raw = try self.patchify(pixels, batch, gh, gw, ps);
        defer _ = mlx.mlx_array_free(patches_raw);
        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&x, patches_raw, .bfloat16, self.s));
        {
            // [1, N, patch_dim] -> [N, patch_dim]: every block below works in 2D.
            const shape = [_]c_int{ n, @intCast(3 * self.patch * self.patch) };
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&r, x, &shape, 2, self.s));
            replace(&x, r);
        }

        replace(&x, try self.linear(x, self.patch_w, self.patch_b));
        {
            var summed = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&summed, x, self.pos_emb, self.s));
            replace(&x, summed);
        }

        var dt = mlx.DtypeTrace.begin("joycaption-vision", x, if (self.blocks.len > 0) self.blocks[0].ln1_w else null);
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
                const up = try self.linear(normed, blk.fc1.w, blk.fc1.b);
                defer _ = mlx.mlx_array_free(up);
                // The encoder MLP is `gelu_pytorch_tanh`; the projector below is
                // plain erf `gelu` — two acts, one checkpoint.
                const act = try self.geluTanh(up);
                defer _ = mlx.mlx_array_free(act);
                const down = try self.linear(act, blk.fc2.w, blk.fc2.b);
                defer _ = mlx.mlx_array_free(down);
                var h = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h, x, down, self.s));
                replace(&x, h);
            }
            dt.layer(x, i);
        }
        dt.end(x);
        // No post_layernorm: `vision_feature_layer` selected a hidden state
        // before the final encoder layer, which was never run either.

        return self.project(x, n);
    }

    /// `LlavaMultiModalProjector`: Linear -> erf GELU -> Linear.
    /// [N, vision_hidden] -> [1, N, text_hidden].
    fn project(self: *JoycaptionVision, hidden: mlx.mlx_array, n: c_int) !mlx.mlx_array {
        var x = try self.linear(hidden, self.proj1.w, self.proj1.b);
        replace(&x, try self.geluExact(x));
        replace(&x, try self.linear(x, self.proj2.w, self.proj2.b));

        var out = mlx.mlx_array_new();
        const oshape = [_]c_int{ 1, n, @intCast(self.out_hidden) };
        try mlx.check(mlx.mlx_reshape(&out, x, &oshape, 3, self.s));
        _ = mlx.mlx_array_free(x);
        return out;
    }

    fn attention(self: *JoycaptionVision, x: mlx.mlx_array, blk: Block, n: c_int) !mlx.mlx_array {
        const hd: c_int = @intCast(self.head_dim);
        const heads: c_int = @intCast(self.heads);

        // [N, D] -> [1, heads, N, hd] for the fused SDPA. No mask: one image
        // per call, full bidirectional attention (SigLIP has no CLS/packing).
        var bhnd: [3]mlx.mlx_array = undefined;
        var built: usize = 0;
        errdefer for (bhnd[0..built]) |arr| {
            _ = mlx.mlx_array_free(arr);
        };
        inline for (.{ blk.q, blk.k, blk.v }, 0..) |l, i| {
            const flat = try self.linear(x, l.w, l.b);
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
        var ctx_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ctx_out);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ctx_out, bhnd[0], bhnd[1], bhnd[2], scale, "", none, none, false, self.s));

        const back = [_]c_int{ 0, 2, 1, 3 };
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_transpose_axes(&t, ctx_out, &back, 4, self.s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, t, &[_]c_int{ n, @intCast(self.hidden) }, 2, self.s));
        return self.linear(flat, blk.out.w, blk.out.b);
    }

    /// `[B,3,H,W]` -> `[B, gh*gw, ps*ps*3]`, feature order (py, px, c).
    fn patchify(self: *JoycaptionVision, pixels: mlx.mlx_array, batch: c_int, gh: c_int, gw: c_int, ps: c_int) !mlx.mlx_array {
        const reshape6 = [_]c_int{ batch, 3, gh, ps, gw, ps };
        var r1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r1);
        try mlx.check(mlx.mlx_reshape(&r1, pixels, &reshape6, 6, self.s));
        const perm = [_]c_int{ 0, 2, 4, 3, 5, 1 };
        var r2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r2);
        try mlx.check(mlx.mlx_transpose_axes(&r2, r1, &perm, 6, self.s));
        const num_patches = gh * gw;
        const patch_dim = 3 * ps * ps;
        const reshape3 = [_]c_int{ batch, num_patches, patch_dim };
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&out, r2, &reshape3, 3, self.s));
        return out;
    }

    /// y = x · Wᵀ + b. W is [out, in] bf16 dense (this tower never ships quantized).
    fn linear(self: *JoycaptionVision, x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array) !mlx.mlx_array {
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        try mlx.check(mlx.mlx_transpose(&wt, w, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&out, x, wt, self.s));
        var biased = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&biased, out, b, self.s));
        _ = mlx.mlx_array_free(out);
        return biased;
    }

    fn layerNorm(self: *JoycaptionVision, x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_layer_norm(&out, x, w, b, self.ln_eps, self.s));
        return out;
    }

    /// `gelu_pytorch_tanh`: 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3))). SigLIP's
    /// own MLP activation.
    fn geluTanh(self: *JoycaptionVision, x: mlx.mlx_array) !mlx.mlx_array {
        const c_coeff = bf16Scalar(0.7978845608028654, self.s);
        defer _ = mlx.mlx_array_free(c_coeff);
        const c_inner = bf16Scalar(0.044715, self.s);
        defer _ = mlx.mlx_array_free(c_inner);
        const c_three = bf16Scalar(3.0, self.s);
        defer _ = mlx.mlx_array_free(c_three);
        const c_one = bf16Scalar(1.0, self.s);
        defer _ = mlx.mlx_array_free(c_one);
        const c_half = bf16Scalar(0.5, self.s);
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

    /// nn.GELU() exact: 0.5·x·(1+erf(x/√2)). `multi_modal_projector`'s
    /// `projector_hidden_act: "gelu"`.
    fn geluExact(self: *JoycaptionVision, x: mlx.mlx_array) !mlx.mlx_array {
        const inv_sqrt2 = bf16Scalar(0.7071067811865476, self.s);
        defer _ = mlx.mlx_array_free(inv_sqrt2);
        const c_one = bf16Scalar(1.0, self.s);
        defer _ = mlx.mlx_array_free(c_one);
        const c_half = bf16Scalar(0.5, self.s);
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
