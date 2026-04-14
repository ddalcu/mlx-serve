const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");

const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;

// ── Clipped Linear Weight ──
// Gemma 4 vision uses "clipped linears": bf16 dense weights with input/output clamping.
// Forward: clip(x, in_min, in_max) → matmul(x, w.T) → clip(y, out_min, out_max)

const ClippedLinearWeight = struct {
    weight: mlx.mlx_array, // [out_dim, in_dim] bf16
    input_min: mlx.mlx_array,
    input_max: mlx.mlx_array,
    output_min: mlx.mlx_array,
    output_max: mlx.mlx_array,
};

// ── Vision Layer Weights ──

const VisionLayerWeights = struct {
    input_layernorm: mlx.mlx_array,
    post_attention_layernorm: mlx.mlx_array,
    pre_feedforward_layernorm: mlx.mlx_array,
    post_feedforward_layernorm: mlx.mlx_array,

    // Attention
    q_proj: ClippedLinearWeight,
    k_proj: ClippedLinearWeight,
    v_proj: ClippedLinearWeight,
    out_proj: ClippedLinearWeight,
    q_norm: mlx.mlx_array,
    k_norm: mlx.mlx_array,

    // MLP
    gate_proj: ClippedLinearWeight,
    up_proj: ClippedLinearWeight,
    down_proj: ClippedLinearWeight,
};

// ── Vision Encoder ──
// Matches mlx-vlm's Gemma4 VisionModel: SigLIP encoder with 2D RoPE,
// clipped linears, V-norm, position-based pooling, post-projection norm.

pub const VisionEncoder = struct {
    config: ModelConfig,
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,

    // Patch embedding
    patch_proj_w: mlx.mlx_array, // [hidden, patch_dim] = [768, 768]
    position_embedding: mlx.mlx_array, // [2, position_embedding_size, hidden]

    // Transformer layers
    layers: []VisionLayerWeights,

    // Embedding projection (quantized int4 linear: vision_hidden → text_hidden)
    proj_w: mlx.mlx_array,
    proj_s: mlx.mlx_array,
    proj_b: mlx.mlx_array,
    proj_quant_bits: u32,
    proj_quant_group_size: u32,

    // Constants
    rms_eps: f32,
    half: mlx.mlx_array,
    one: mlx.mlx_array,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !VisionEncoder {
        const s = mlx.mlx_default_gpu_stream_new();

        const num_layers = config.vision_num_layers;
        var layers = try allocator.alloc(VisionLayerWeights, num_layers);
        errdefer allocator.free(layers);

        var name_buf: [256]u8 = undefined;

        for (0..num_layers) |i| {
            layers[i] = .{
                .input_layernorm = getVisionWeight(weights, &name_buf, i, "input_layernorm.weight"),
                .post_attention_layernorm = getVisionWeight(weights, &name_buf, i, "post_attention_layernorm.weight"),
                .pre_feedforward_layernorm = getVisionWeight(weights, &name_buf, i, "pre_feedforward_layernorm.weight"),
                .post_feedforward_layernorm = getVisionWeight(weights, &name_buf, i, "post_feedforward_layernorm.weight"),
                .q_proj = loadClippedLinear(weights, &name_buf, i, "self_attn.q_proj"),
                .k_proj = loadClippedLinear(weights, &name_buf, i, "self_attn.k_proj"),
                .v_proj = loadClippedLinear(weights, &name_buf, i, "self_attn.v_proj"),
                .out_proj = loadClippedLinear(weights, &name_buf, i, "self_attn.o_proj"),
                .q_norm = getVisionWeight(weights, &name_buf, i, "self_attn.q_norm.weight"),
                .k_norm = getVisionWeight(weights, &name_buf, i, "self_attn.k_norm.weight"),
                .gate_proj = loadClippedLinear(weights, &name_buf, i, "mlp.gate_proj"),
                .up_proj = loadClippedLinear(weights, &name_buf, i, "mlp.up_proj"),
                .down_proj = loadClippedLinear(weights, &name_buf, i, "mlp.down_proj"),
            };
        }

        // Patch embedder
        const patch_w = getWeight(weights, &name_buf, "vision_tower.patch_embedder.input_proj.weight") orelse {
            log.err("MISSING WEIGHT: vision_tower.patch_embedder.input_proj.weight\n", .{});
            return error.MissingVisionWeights;
        };
        const pos_emb = getWeight(weights, &name_buf, "vision_tower.patch_embedder.position_embedding_table") orelse {
            log.err("MISSING WEIGHT: vision_tower.patch_embedder.position_embedding_table\n", .{});
            return error.MissingVisionWeights;
        };

        // Embedding projection: embed_vision.embedding_projection
        const proj_w = getWeight(weights, &name_buf, "embed_vision.embedding_projection.weight") orelse
            getWeight(weights, &name_buf, "multi_modal_projector.weight") orelse {
            log.err("MISSING WEIGHT: embed_vision.embedding_projection.weight\n", .{});
            return error.MissingVisionWeights;
        };
        const proj_s = getWeight(weights, &name_buf, "embed_vision.embedding_projection.scales") orelse
            getWeight(weights, &name_buf, "multi_modal_projector.scales") orelse mlx.mlx_array_new();
        const proj_b = getWeight(weights, &name_buf, "embed_vision.embedding_projection.biases") orelse
            getWeight(weights, &name_buf, "multi_modal_projector.biases") orelse mlx.mlx_array_new();

        const proj_bits = detectProjBits(proj_w, proj_s, config.quant_group_size);

        // Batch eval all vision weights
        {
            var eval_list = std.ArrayList(mlx.mlx_array).empty;
            defer eval_list.deinit(allocator);
            try eval_list.append(allocator, patch_w);
            try eval_list.append(allocator, pos_emb);
            try eval_list.append(allocator, proj_w);
            if (mlx.mlx_array_ndim(proj_s) > 0) try eval_list.append(allocator, proj_s);
            if (mlx.mlx_array_ndim(proj_b) > 0) try eval_list.append(allocator, proj_b);
            for (layers) |lw| {
                try eval_list.append(allocator, lw.input_layernorm);
                try eval_list.append(allocator, lw.post_attention_layernorm);
                try eval_list.append(allocator, lw.pre_feedforward_layernorm);
                try eval_list.append(allocator, lw.post_feedforward_layernorm);
                try eval_list.append(allocator, lw.q_proj.weight);
                try eval_list.append(allocator, lw.k_proj.weight);
                try eval_list.append(allocator, lw.v_proj.weight);
                try eval_list.append(allocator, lw.out_proj.weight);
                try eval_list.append(allocator, lw.q_norm);
                try eval_list.append(allocator, lw.k_norm);
                try eval_list.append(allocator, lw.gate_proj.weight);
                try eval_list.append(allocator, lw.up_proj.weight);
                try eval_list.append(allocator, lw.down_proj.weight);
            }
            const vec = mlx.mlx_vector_array_new_data(eval_list.items.ptr, eval_list.items.len);
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_eval(vec);
        }

        log.info("Vision encoder: {d} layers, hidden={d}, heads={d}, pool→{d} tokens\n", .{
            num_layers, config.vision_hidden_size, config.vision_num_heads, config.vision_soft_tokens,
        });

        return .{
            .config = config,
            .s = s,
            .allocator = allocator,
            .patch_proj_w = patch_w,
            .position_embedding = pos_emb,
            .layers = layers,
            .proj_w = proj_w,
            .proj_s = proj_s,
            .proj_b = proj_b,
            .proj_quant_bits = proj_bits,
            .proj_quant_group_size = config.quant_group_size,
            .rms_eps = 1e-6,
            .half = bf16Scalar(0.5, s),
            .one = bf16Scalar(1.0, s),
        };
    }

    pub fn deinit(self: *VisionEncoder) void {
        _ = mlx.mlx_array_free(self.half);
        _ = mlx.mlx_array_free(self.one);
        self.allocator.free(self.layers);
        _ = mlx.mlx_stream_free(self.s);
    }

    /// Forward pass: pixel data [B, 3, H, W] float32 → vision embeddings [B, N_out, text_hidden_size].
    /// N_out is the number of valid pooled tokens (e.g. 25 for 224x224 with kernel=3).
    /// The caller must cycle these to fill the expected image_seq_length (280) token positions.
    pub fn forward(self: *VisionEncoder, pixels: mlx.mlx_array) !mlx.mlx_array {
        const cfg = &self.config;
        const ps: c_int = @intCast(cfg.vision_patch_size);
        const hidden: c_int = @intCast(cfg.vision_hidden_size);
        const num_heads: c_int = @intCast(cfg.vision_num_heads);
        const head_dim: c_int = @intCast(cfg.vision_head_dim);
        const kernel: u32 = cfg.vision_pooling_kernel;

        // pixels: [B, 3, H, W]
        const pix_shape = mlx.getShape(pixels);
        const batch = pix_shape[0];
        const height = pix_shape[2];
        const width = pix_shape[3];
        const grid_h = @divExact(height, ps);
        const grid_w = @divExact(width, ps);
        const num_patches = grid_h * grid_w;

        // 1. Patchify: [B, 3, H, W] → [B, num_patches, 3*ps*ps]
        // Reference: patches = 2 * (patches - 0.5) then matmul
        const patches_raw = try self.patchify(pixels, batch, grid_h, grid_w, ps);
        defer _ = mlx.mlx_array_free(patches_raw);

        // Cast to bf16
        var patches_bf16 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(patches_bf16);
        try mlx.check(mlx.mlx_astype(&patches_bf16, patches_raw, .bfloat16, self.s));

        // Normalize: 2*(x - 0.5) = 2x - 1
        const two = bf16Scalar(2.0, self.s);
        defer _ = mlx.mlx_array_free(two);
        var scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_multiply(&scaled, patches_bf16, two, self.s));
        var normed_patches = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normed_patches);
        try mlx.check(mlx.mlx_subtract(&normed_patches, scaled, self.one, self.s));

        // Project patches: matmul with input_proj weight
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        try mlx.check(mlx.mlx_transpose(&wt, self.patch_proj_w, self.s));
        var h = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&h, normed_patches, wt, self.s));

        // 2. Position embeddings via one_hot @ table
        //    Build 2D position grid [B, num_patches, 2] then one_hot matmul
        {
            const pos_emb = try self.computePositionEmbeddings(batch, grid_h, grid_w, num_patches, hidden);
            defer _ = mlx.mlx_array_free(pos_emb);
            var h_new = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_new, h, pos_emb, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_new;
        }

        // 3. Build 2D position indices for RoPE: [B, num_patches, 2]
        const positions = try self.buildPositionGrid(batch, grid_h, grid_w, num_patches);
        defer _ = mlx.mlx_array_free(positions);

        // 4. Transformer layers
        for (self.layers) |lw| {
            h = try self.transformerLayer(h, lw, positions, batch, num_patches, num_heads, head_dim);
        }

        // 5. Pooling: reduce patches via position-based averaging
        //    For 14x14 with kernel=3: 5x5 = 25 output tokens
        const k: c_int = @intCast(kernel);
        const out_h = @divTrunc(grid_h + k - 1, k); // ceil division
        const out_w = @divTrunc(grid_w + k - 1, k);
        const n_pooled = out_h * out_w;

        const pooled = try self.positionPool(h, grid_h, grid_w, k, batch, hidden);
        _ = mlx.mlx_array_free(h);

        // Scale by sqrt(hidden_size) — reference: hidden_states * self.root_hidden_size
        const root_hidden = bf16Scalar(@sqrt(@as(f32, @floatFromInt(cfg.vision_hidden_size))), self.s);
        defer _ = mlx.mlx_array_free(root_hidden);
        var pooled_scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&pooled_scaled, pooled, root_hidden, self.s));
        _ = mlx.mlx_array_free(pooled);

        // 6. Embedding projection: vision_hidden → text_hidden
        var projected: mlx.mlx_array = undefined;
        if (mlx.mlx_array_ndim(self.proj_s) > 0) {
            projected = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_quantized_matmul(
                &projected, pooled_scaled, self.proj_w, self.proj_s, self.proj_b,
                true, mlx.mlx_optional_int.some(@intCast(self.proj_quant_group_size)),
                mlx.mlx_optional_int.some(@intCast(self.proj_quant_bits)), "affine", self.s,
            ));
        } else {
            var proj_wt = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(proj_wt);
            try mlx.check(mlx.mlx_transpose(&proj_wt, self.proj_w, self.s));
            projected = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_matmul(&projected, pooled_scaled, proj_wt, self.s));
        }
        _ = mlx.mlx_array_free(pooled_scaled);

        // 7. Post-projection norm: parameter-free RMS norm (no learned weight)
        //    Reference: RMSNormNoScale — norm = x / sqrt(mean(x^2) + eps)
        const post_normed = try self.rmsNormNoScale(projected);
        _ = mlx.mlx_array_free(projected);

        // Output: [B, n_pooled, text_hidden]
        // The caller inserts exactly n_pooled image tokens into the prompt.
        // For 768x768: 48/3=16 → 16*16=256 tokens. For 224x224: 14/3≈5 → 5*5=25 tokens.
        _ = n_pooled;
        return post_normed;
    }

    // ── Patch Extraction ──

    fn patchify(self: *VisionEncoder, pixels: mlx.mlx_array, batch: c_int, grid_h: c_int, grid_w: c_int, ps: c_int) !mlx.mlx_array {
        // [B, 3, H, W] → [B, 3, grid_h, ps, grid_w, ps]
        const reshape6 = [_]c_int{ batch, 3, grid_h, ps, grid_w, ps };
        var r1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r1);
        try mlx.check(mlx.mlx_reshape(&r1, pixels, &reshape6, 6, self.s));

        // → [B, grid_h, grid_w, ps, ps, 3] (reference: transpose(0,2,4,3,5,1))
        const perm = [_]c_int{ 0, 2, 4, 3, 5, 1 };
        var r2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r2);
        try mlx.check(mlx.mlx_transpose_axes(&r2, r1, &perm, 6, self.s));

        // → [B, num_patches, 3*ps*ps]
        const num_patches = grid_h * grid_w;
        const patch_dim = 3 * ps * ps;
        const reshape3 = [_]c_int{ batch, num_patches, patch_dim };
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&result, r2, &reshape3, 3, self.s));
        return result;
    }

    // ── Position Embeddings (one-hot matmul) ──

    fn computePositionEmbeddings(self: *VisionEncoder, batch: c_int, grid_h: c_int, grid_w: c_int, num_patches: c_int, hidden: c_int) !mlx.mlx_array {
        const pos_size: c_int = @intCast(self.config.vision_position_embedding_size);
        _ = hidden;

        // Build position grid: [num_patches, 2] with (x, y) coords
        const positions = try self.buildPositionGrid(batch, grid_h, grid_w, num_patches);
        defer _ = mlx.mlx_array_free(positions);

        // One-hot encode: [B, num_patches, 2] → [B, num_patches, 2, pos_size]
        //   one_hot(indices, num_classes) = (expand_dims(indices, -1) == arange(num_classes))
        var arange_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(arange_arr);
        try mlx.check(mlx.mlx_arange(&arange_arr, 0.0, @floatFromInt(pos_size), 1.0, .int32, self.s));

        // Expand positions: [B, num_patches, 2, 1]
        const exp_shape = [_]c_int{ batch, num_patches, 2, 1 };
        var pos_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pos_exp);
        try mlx.check(mlx.mlx_reshape(&pos_exp, positions, &exp_shape, 4, self.s));

        // Broadcast equal: [B, num_patches, 2, 1] == [pos_size] → [B, num_patches, 2, pos_size]
        var oh_bool = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(oh_bool);
        try mlx.check(mlx.mlx_equal(&oh_bool, pos_exp, arange_arr, self.s));

        // Cast to bf16
        var oh = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(oh);
        try mlx.check(mlx.mlx_astype(&oh, oh_bool, .bfloat16, self.s));

        // Transpose: [B, num_patches, 2, pos_size] → [B, 2, num_patches, pos_size]
        const oh_perm = [_]c_int{ 0, 2, 1, 3 };
        var oh_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(oh_t);
        try mlx.check(mlx.mlx_transpose_axes(&oh_t, oh, &oh_perm, 4, self.s));

        // Matmul: [B, 2, num_patches, pos_size] @ [2, pos_size, hidden] → [B, 2, num_patches, hidden]
        var pos_emb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pos_emb);
        try mlx.check(mlx.mlx_matmul(&pos_emb, oh_t, self.position_embedding, self.s));

        // Sum over dim 1 (the 2 spatial dimensions): [B, 2, num_patches, hidden] → [B, num_patches, hidden]
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sum_axis(&result, pos_emb, 1, false, self.s));

        return result;
    }

    fn buildPositionGrid(self: *VisionEncoder, batch: c_int, grid_h: c_int, grid_w: c_int, num_patches: c_int) !mlx.mlx_array {
        _ = self;
        _ = batch;
        // Build [num_patches, 2] position grid matching reference meshgrid(xy)
        // positions[i] = (x, y) where x = i % grid_w, y = i / grid_w
        const n: usize = @intCast(num_patches);
        const gw: usize = @intCast(grid_w);
        const gh: usize = @intCast(grid_h);
        _ = gh;
        var pos_data = try std.heap.page_allocator.alloc(i32, n * 2);
        defer std.heap.page_allocator.free(pos_data);
        for (0..n) |i| {
            pos_data[i * 2] = @intCast(i % gw); // x
            pos_data[i * 2 + 1] = @intCast(i / gw); // y
        }
        // Create [1, num_patches, 2] array
        const shape = [_]c_int{ 1, @intCast(num_patches), 2 };
        return mlx.mlx_array_new_data(pos_data.ptr, &shape, 3, .int32);
    }

    // ── Transformer Layer (correct residual pattern) ──

    fn transformerLayer(
        self: *VisionEncoder,
        input: mlx.mlx_array,
        lw: VisionLayerWeights,
        positions: mlx.mlx_array,
        batch: c_int,
        seq_len: c_int,
        num_heads: c_int,
        head_dim: c_int,
    ) !mlx.mlx_array {
        // Reference pattern:
        //   normed = input_layernorm(x)
        //   attn_out = self_attn(normed)
        //   attn_out = post_attention_layernorm(attn_out)  ← post-norm on OUTPUT only
        //   h = x + attn_out                                ← residual AFTER post-norm
        //
        //   normed_h = pre_feedforward_layernorm(h)
        //   ffw_out = mlp(normed_h)
        //   ffw_out = post_feedforward_layernorm(ffw_out)   ← post-norm on OUTPUT only
        //   return h + ffw_out                               ← residual AFTER post-norm

        var h = input;

        // Attention block
        {
            const normed = try self.rmsNorm(h, lw.input_layernorm);
            defer _ = mlx.mlx_array_free(normed);

            const attn_out = try self.selfAttention(normed, lw, positions, batch, seq_len, num_heads, head_dim);
            defer _ = mlx.mlx_array_free(attn_out);

            // Post-norm on attention output only (NOT residual)
            const attn_normed = try self.rmsNorm(attn_out, lw.post_attention_layernorm);
            defer _ = mlx.mlx_array_free(attn_normed);

            // Residual: h = x + post_norm(attn_out)
            var h_new = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_new, h, attn_normed, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_new;
        }

        // FFN block
        {
            const normed = try self.rmsNorm(h, lw.pre_feedforward_layernorm);
            defer _ = mlx.mlx_array_free(normed);

            const ffw_out = try self.mlpForward(normed, lw);
            defer _ = mlx.mlx_array_free(ffw_out);

            // Post-norm on FFN output only
            const ffw_normed = try self.rmsNorm(ffw_out, lw.post_feedforward_layernorm);
            defer _ = mlx.mlx_array_free(ffw_normed);

            // Residual: h = h + post_norm(ffw_out)
            var h_new = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_new, h, ffw_normed, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_new;
        }

        return h;
    }

    // ── Self-Attention with 2D RoPE and V-norm ──

    fn selfAttention(
        self: *VisionEncoder,
        x: mlx.mlx_array,
        lw: VisionLayerWeights,
        positions: mlx.mlx_array,
        batch: c_int,
        seq_len: c_int,
        num_heads: c_int,
        head_dim: c_int,
    ) !mlx.mlx_array {
        // Q, K, V projections (clipped linear)
        const q_raw = try self.clippedLinear(x, lw.q_proj);
        defer _ = mlx.mlx_array_free(q_raw);
        const k_raw = try self.clippedLinear(x, lw.k_proj);
        defer _ = mlx.mlx_array_free(k_raw);
        const v_raw = try self.clippedLinear(x, lw.v_proj);
        defer _ = mlx.mlx_array_free(v_raw);

        // Reshape to [B, seq, heads, head_dim]
        const qkv_shape = [_]c_int{ batch, seq_len, num_heads, head_dim };

        var q = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q);
        try mlx.check(mlx.mlx_reshape(&q, q_raw, &qkv_shape, 4, self.s));

        var k = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k);
        try mlx.check(mlx.mlx_reshape(&k, k_raw, &qkv_shape, 4, self.s));

        var v = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v);
        try mlx.check(mlx.mlx_reshape(&v, v_raw, &qkv_shape, 4, self.s));

        // QK-norm (per-head RMS norm with learned weight)
        const q_normed = try self.headwiseRmsNorm(q, lw.q_norm, batch, seq_len, num_heads, head_dim);
        defer _ = mlx.mlx_array_free(q_normed);
        const k_normed = try self.headwiseRmsNorm(k, lw.k_norm, batch, seq_len, num_heads, head_dim);
        defer _ = mlx.mlx_array_free(k_normed);

        // V-norm: parameter-free RMS norm on values
        const v_normed = try self.rmsNormNoScalePerHead(v, batch, seq_len, num_heads, head_dim);
        defer _ = mlx.mlx_array_free(v_normed);

        // Apply 2D RoPE to Q and K (after norms, before attention)
        // positions: [B, seq, 2]
        const q_rope = try self.applyMultidimensionalRope(q_normed, positions, batch, seq_len, num_heads, head_dim);
        defer _ = mlx.mlx_array_free(q_rope);
        const k_rope = try self.applyMultidimensionalRope(k_normed, positions, batch, seq_len, num_heads, head_dim);
        defer _ = mlx.mlx_array_free(k_rope);

        // Transpose to [B, heads, seq, head_dim]
        const perm = [_]c_int{ 0, 2, 1, 3 };
        var q_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_t);
        try mlx.check(mlx.mlx_transpose_axes(&q_t, q_rope, &perm, 4, self.s));

        var k_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_t);
        try mlx.check(mlx.mlx_transpose_axes(&k_t, k_rope, &perm, 4, self.s));

        var v_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_t);
        try mlx.check(mlx.mlx_transpose_axes(&v_t, v_normed, &perm, 4, self.s));

        // Scaled dot-product attention (scale=1.0, no mask — bidirectional)
        // Reference uses scale=1.0 because QK-norm already normalizes
        const kt_perm = [_]c_int{ 0, 1, 3, 2 };
        var k_tp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_tp);
        try mlx.check(mlx.mlx_transpose_axes(&k_tp, k_t, &kt_perm, 4, self.s));

        var attn_weights = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_weights);
        try mlx.check(mlx.mlx_matmul(&attn_weights, q_t, k_tp, self.s));
        // No scaling (scale=1.0)

        var attn_probs = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_probs);
        try mlx.check(mlx.mlx_softmax_axis(&attn_probs, attn_weights, -1, false, self.s));

        var attn_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_out);
        try mlx.check(mlx.mlx_matmul(&attn_out, attn_probs, v_t, self.s));

        // Transpose back: [B, heads, seq, head_dim] → [B, seq, heads*head_dim]
        const out_perm = [_]c_int{ 0, 2, 1, 3 };
        var attn_tp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_tp);
        try mlx.check(mlx.mlx_transpose_axes(&attn_tp, attn_out, &out_perm, 4, self.s));

        const out_hidden = num_heads * head_dim;
        const out_shape = [_]c_int{ batch, seq_len, out_hidden };
        var attn_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_flat);
        try mlx.check(mlx.mlx_reshape(&attn_flat, attn_tp, &out_shape, 3, self.s));

        return self.clippedLinear(attn_flat, lw.out_proj);
    }

    // ── 2D RoPE ──

    fn applyMultidimensionalRope(self: *VisionEncoder, input: mlx.mlx_array, positions: mlx.mlx_array, batch: c_int, seq_len: c_int, num_heads: c_int, head_dim: c_int) !mlx.mlx_array {
        // Reference: split head_dim into 2 parts (one per spatial dim), apply RoPE independently
        // channels_per_dim = 2 * (head_dim / (2*ndim)) where ndim=2
        // For head_dim=64: channels_per_dim=32, half_per_dim=16
        const ndim: c_int = 2;
        const channels_per_dim = 2 * @divExact(head_dim, 2 * ndim); // 32
        const half_per_dim = @divExact(channels_per_dim, 2); // 16
        const base: f32 = self.config.vision_rope_theta;

        // Input: [B, seq, heads, head_dim], positions: [B, seq, 2]
        var parts: [2]mlx.mlx_array = undefined;

        for (0..2) |d| {
            const d_i: c_int = @intCast(d);
            const start = d_i * channels_per_dim;
            const end = start + channels_per_dim;

            // Slice input[..., start:end]: [B, seq, heads, channels_per_dim]
            const sl_start = [_]c_int{ 0, 0, 0, start };
            const sl_stop = [_]c_int{ batch, seq_len, num_heads, end };
            const sl_strides = [_]c_int{ 1, 1, 1, 1 };
            var x_part = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&x_part, input, &sl_start, 4, &sl_stop, 4, &sl_strides, 4, self.s));

            // Compute frequencies for this dimension
            // freq_exponents = (2/channels_per_dim) * arange(half_per_dim)
            // timescale = base^freq_exponents
            // sinusoid_inp = positions[..., d:d+1] / timescale
            const pos_start = [_]c_int{ 0, 0, d_i };
            const pos_stop = [_]c_int{ batch, seq_len, d_i + 1 };
            const pos_strides = [_]c_int{ 1, 1, 1 };
            var pos_d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(pos_d);
            try mlx.check(mlx.mlx_slice(&pos_d, positions, &pos_start, 3, &pos_stop, 3, &pos_strides, 3, self.s));

            // pos_d: [B, seq, 1] float
            var pos_f = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(pos_f);
            try mlx.check(mlx.mlx_astype(&pos_f, pos_d, .float32, self.s));

            // freq_exponents: [half_per_dim]
            var arange_f = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(arange_f);
            try mlx.check(mlx.mlx_arange(&arange_f, 0.0, @floatFromInt(half_per_dim), 1.0, .float32, self.s));

            const cpd_inv = mlx.mlx_array_new_float(2.0 / @as(f32, @floatFromInt(channels_per_dim)));
            defer _ = mlx.mlx_array_free(cpd_inv);
            var freq_exp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(freq_exp);
            try mlx.check(mlx.mlx_multiply(&freq_exp, arange_f, cpd_inv, self.s));

            // timescale = base^freq_exp
            const base_arr = mlx.mlx_array_new_float(base);
            defer _ = mlx.mlx_array_free(base_arr);
            var timescale = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(timescale);
            try mlx.check(mlx.mlx_power(&timescale, base_arr, freq_exp, self.s));

            // sinusoid_inp = pos / timescale: [B, seq, 1] / [half_per_dim] → [B, seq, half_per_dim]
            var sin_inp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sin_inp);
            try mlx.check(mlx.mlx_divide(&sin_inp, pos_f, timescale, self.s));

            // cos, sin
            var cos_val = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cos_val);
            try mlx.check(mlx.mlx_cos(&cos_val, sin_inp, self.s));
            var sin_val = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sin_val);
            try mlx.check(mlx.mlx_sin(&sin_val, sin_inp, self.s));

            // Duplicate: [B, seq, half] → [B, seq, channels_per_dim]
            const cat_arrs_c = [_]mlx.mlx_array{ cos_val, cos_val };
            const cat_vec_c = mlx.mlx_vector_array_new_data(&cat_arrs_c, 2);
            defer _ = mlx.mlx_vector_array_free(cat_vec_c);
            var cos_dup = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cos_dup);
            try mlx.check(mlx.mlx_concatenate_axis(&cos_dup, cat_vec_c, -1, self.s));

            const cat_arrs_s = [_]mlx.mlx_array{ sin_val, sin_val };
            const cat_vec_s = mlx.mlx_vector_array_new_data(&cat_arrs_s, 2);
            defer _ = mlx.mlx_vector_array_free(cat_vec_s);
            var sin_dup = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sin_dup);
            try mlx.check(mlx.mlx_concatenate_axis(&sin_dup, cat_vec_s, -1, self.s));

            // Cast to input dtype and add head dimension
            var cos_bf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cos_bf);
            try mlx.check(mlx.mlx_astype(&cos_bf, cos_dup, .bfloat16, self.s));
            var sin_bf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sin_bf);
            try mlx.check(mlx.mlx_astype(&sin_bf, sin_dup, .bfloat16, self.s));

            // Expand dims for head broadcasting: [B, seq, 1, channels_per_dim]
            const exp_shape = [_]c_int{ batch, seq_len, 1, channels_per_dim };
            var cos_exp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cos_exp);
            try mlx.check(mlx.mlx_reshape(&cos_exp, cos_bf, &exp_shape, 4, self.s));
            var sin_exp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sin_exp);
            try mlx.check(mlx.mlx_reshape(&sin_exp, sin_bf, &exp_shape, 4, self.s));

            // rotate_half: [-x2, x1]
            const half_ch = @divExact(channels_per_dim, 2);
            const rh_start1 = [_]c_int{ 0, 0, 0, half_ch };
            const rh_stop1 = [_]c_int{ batch, seq_len, num_heads, channels_per_dim };
            var x2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x2);
            try mlx.check(mlx.mlx_slice(&x2, x_part, &rh_start1, 4, &rh_stop1, 4, &sl_strides, 4, self.s));

            const rh_start0 = [_]c_int{ 0, 0, 0, 0 };
            const rh_stop0 = [_]c_int{ batch, seq_len, num_heads, half_ch };
            var x1 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x1);
            try mlx.check(mlx.mlx_slice(&x1, x_part, &rh_start0, 4, &rh_stop0, 4, &sl_strides, 4, self.s));

            var neg_x2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(neg_x2);
            try mlx.check(mlx.mlx_negative(&neg_x2, x2, self.s));

            const rot_arrs = [_]mlx.mlx_array{ neg_x2, x1 };
            const rot_vec = mlx.mlx_vector_array_new_data(&rot_arrs, 2);
            defer _ = mlx.mlx_vector_array_free(rot_vec);
            var rotated = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(rotated);
            try mlx.check(mlx.mlx_concatenate_axis(&rotated, rot_vec, -1, self.s));

            // y = x * cos + rotate_half(x) * sin
            var xcos = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xcos);
            try mlx.check(mlx.mlx_multiply(&xcos, x_part, cos_exp, self.s));
            _ = mlx.mlx_array_free(x_part);

            var rsin = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(rsin);
            try mlx.check(mlx.mlx_multiply(&rsin, rotated, sin_exp, self.s));

            var y_part = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&y_part, xcos, rsin, self.s));
            parts[d] = y_part;
        }

        // Concatenate parts along head_dim axis
        defer _ = mlx.mlx_array_free(parts[0]);
        defer _ = mlx.mlx_array_free(parts[1]);
        const cat_vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(cat_vec);
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&result, cat_vec, -1, self.s));
        return result;
    }

    // ── MLP ──

    /// MLP: down_proj(gelu(gate_proj(x)) * up_proj(x))
    fn mlpForward(self: *VisionEncoder, x: mlx.mlx_array, lw: VisionLayerWeights) !mlx.mlx_array {
        const gate = try self.clippedLinear(x, lw.gate_proj);
        defer _ = mlx.mlx_array_free(gate);

        const gate_act = try self.geluApprox(gate);
        defer _ = mlx.mlx_array_free(gate_act);

        const up = try self.clippedLinear(x, lw.up_proj);
        defer _ = mlx.mlx_array_free(up);

        var gated = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gated);
        try mlx.check(mlx.mlx_multiply(&gated, gate_act, up, self.s));

        return self.clippedLinear(gated, lw.down_proj);
    }

    // ── Core Operations ──

    fn clippedLinear(self: *VisionEncoder, x: mlx.mlx_array, cl: ClippedLinearWeight) !mlx.mlx_array {
        var clipped_in = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(clipped_in);
        try mlx.check(mlx.mlx_maximum(&clipped_in, x, cl.input_min, self.s));
        var clipped_in2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(clipped_in2);
        try mlx.check(mlx.mlx_minimum(&clipped_in2, clipped_in, cl.input_max, self.s));

        var w_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w_t);
        try mlx.check(mlx.mlx_transpose(&w_t, cl.weight, self.s));
        var y = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&y, clipped_in2, w_t, self.s));

        var clipped_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(clipped_out);
        try mlx.check(mlx.mlx_maximum(&clipped_out, y, cl.output_min, self.s));
        _ = mlx.mlx_array_free(y);

        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_minimum(&result, clipped_out, cl.output_max, self.s));
        return result;
    }

    fn rmsNorm(self: *VisionEncoder, x: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_rms_norm(&result, x, w, self.rms_eps, self.s));
        return result;
    }

    /// Parameter-free RMS norm (no learned weight): x / sqrt(mean(x^2) + eps)
    fn rmsNormNoScale(self: *VisionEncoder, x: mlx.mlx_array) !mlx.mlx_array {
        // Create ones weight of appropriate size
        const x_shape = mlx.getShape(x);
        const last_dim = x_shape[x_shape.len - 1];
        const ones_shape = [_]c_int{last_dim};
        var ones = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ones);
        try mlx.check(mlx.mlx_ones(&ones, &ones_shape, 1, .float32, self.s));
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_rms_norm(&result, x, ones, self.rms_eps, self.s));
        return result;
    }

    /// Per-head RMS norm with learned weight. Input: [B, seq, heads, head_dim]
    fn headwiseRmsNorm(self: *VisionEncoder, x: mlx.mlx_array, w: mlx.mlx_array, batch: c_int, seq_len: c_int, num_heads: c_int, head_dim: c_int) !mlx.mlx_array {
        const flat_shape = [_]c_int{ batch * seq_len * num_heads, head_dim };
        var x_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_flat);
        try mlx.check(mlx.mlx_reshape(&x_flat, x, &flat_shape, 2, self.s));

        var normed_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normed_flat);
        try mlx.check(mlx.mlx_fast_rms_norm(&normed_flat, x_flat, w, self.rms_eps, self.s));

        const orig_shape = [_]c_int{ batch, seq_len, num_heads, head_dim };
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&result, normed_flat, &orig_shape, 4, self.s));
        return result;
    }

    /// Per-head parameter-free RMS norm. Input: [B, seq, heads, head_dim]
    fn rmsNormNoScalePerHead(self: *VisionEncoder, x: mlx.mlx_array, batch: c_int, seq_len: c_int, num_heads: c_int, head_dim: c_int) !mlx.mlx_array {
        const flat_shape = [_]c_int{ batch * seq_len * num_heads, head_dim };
        var x_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_flat);
        try mlx.check(mlx.mlx_reshape(&x_flat, x, &flat_shape, 2, self.s));

        const ones_shape = [_]c_int{head_dim};
        var ones = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ones);
        try mlx.check(mlx.mlx_ones(&ones, &ones_shape, 1, .float32, self.s));

        var normed_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normed_flat);
        try mlx.check(mlx.mlx_fast_rms_norm(&normed_flat, x_flat, ones, self.rms_eps, self.s));

        const orig_shape = [_]c_int{ batch, seq_len, num_heads, head_dim };
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&result, normed_flat, &orig_shape, 4, self.s));
        return result;
    }

    /// GELU (tanh approximation): 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
    fn geluApprox(self: *VisionEncoder, x: mlx.mlx_array) !mlx.mlx_array {
        const three = bf16Scalar(3.0, self.s);
        defer _ = mlx.mlx_array_free(three);
        const gelu_coeff = bf16Scalar(0.7978845608028654, self.s);
        defer _ = mlx.mlx_array_free(gelu_coeff);
        const gelu_inner = bf16Scalar(0.044715, self.s);
        defer _ = mlx.mlx_array_free(gelu_inner);

        var x3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x3);
        try mlx.check(mlx.mlx_power(&x3, x, three, self.s));
        var inner = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inner);
        try mlx.check(mlx.mlx_multiply(&inner, gelu_inner, x3, self.s));
        var sum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_add(&sum, x, inner, self.s));
        var scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_multiply(&scaled, gelu_coeff, sum, self.s));
        var tanh_val = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tanh_val);
        try mlx.check(mlx.mlx_tanh(&tanh_val, scaled, self.s));
        var one_plus = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(one_plus);
        try mlx.check(mlx.mlx_add(&one_plus, self.one, tanh_val, self.s));
        var x_times = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_times);
        try mlx.check(mlx.mlx_multiply(&x_times, x, one_plus, self.s));
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&result, x_times, self.half, self.s));
        return result;
    }

    // ── Position-based Pooling ──

    fn positionPool(self: *VisionEncoder, h: mlx.mlx_array, grid_h: c_int, grid_w: c_int, k: c_int, batch: c_int, hidden: c_int) !mlx.mlx_array {
        // Simple block-average pooling: group patches by floor(pos/k)
        // For 14x14 with k=3: output is 5x5 = 25 tokens
        const out_h = @divTrunc(grid_h + k - 1, k);
        const out_w = @divTrunc(grid_w + k - 1, k);
        const n_out = out_h * out_w;
        const n_in: usize = @intCast(grid_h * grid_w);

        // Build weight matrix [n_in, n_out] where w[i,j] = 1/k^2 if patch i belongs to group j
        const n_out_u: usize = @intCast(n_out);
        const k_u: usize = @intCast(k);
        const gw: usize = @intCast(grid_w);
        const ow: usize = @intCast(out_w);
        const k_sq_inv: f32 = 1.0 / @as(f32, @floatFromInt(k * k));

        var weight_data = try self.allocator.alloc(f32, n_in * n_out_u);
        defer self.allocator.free(weight_data);
        @memset(weight_data, 0.0);

        for (0..n_in) |i| {
            const x_pos = i % gw;
            const y_pos = i / gw;
            const gx = x_pos / k_u;
            const gy = y_pos / k_u;
            const group = gx + gy * ow;
            if (group < n_out_u) {
                weight_data[i * n_out_u + group] = k_sq_inv;
            }
        }

        // Create weight array [n_in, n_out]
        const w_shape = [_]c_int{ @intCast(n_in), n_out };
        const w_arr = mlx.mlx_array_new_data(weight_data.ptr, &w_shape, 2, .float32);
        defer _ = mlx.mlx_array_free(w_arr);

        var w_bf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w_bf);
        try mlx.check(mlx.mlx_astype(&w_bf, w_arr, .bfloat16, self.s));

        // h: [B, n_in, hidden], w: [n_in, n_out]
        // output = h^T @ w: need einsum "bLd,Ll->bld" = transpose + matmul
        // Transpose h: [B, hidden, n_in]
        const h_perm = [_]c_int{ 0, 2, 1 };
        var h_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h_t);
        try mlx.check(mlx.mlx_transpose_axes(&h_t, h, &h_perm, 3, self.s));

        // [B, hidden, n_in] @ [n_in, n_out] → [B, hidden, n_out]
        var out_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out_t);
        try mlx.check(mlx.mlx_matmul(&out_t, h_t, w_bf, self.s));

        // Transpose back: [B, n_out, hidden]
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_transpose_axes(&result, out_t, &h_perm, 3, self.s));

        _ = batch;
        _ = hidden;
        return result;
    }

    // ── Cycle Fill ──

    fn cycleFill(self: *VisionEncoder, src: mlx.mlx_array, batch: c_int, n_src: c_int, n_target: c_int) !mlx.mlx_array {
        // Tile the source to fill target length: repeat src ceil(n_target/n_src) times, then slice
        const repeats = @divTrunc(n_target + n_src - 1, n_src);
        _ = batch;

        // Use tile along seq dim
        var parts: [16]mlx.mlx_array = undefined;
        const r: usize = @intCast(@min(repeats, 16));
        for (0..r) |i| {
            parts[i] = src;
        }
        const cat_vec = mlx.mlx_vector_array_new_data(&parts, r);
        defer _ = mlx.mlx_vector_array_free(cat_vec);
        var tiled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tiled);
        try mlx.check(mlx.mlx_concatenate_axis(&tiled, cat_vec, 1, self.s));

        // Slice to target length
        const t_shape = mlx.getShape(tiled);
        const sl_start = [_]c_int{ 0, 0, 0 };
        const sl_stop = [_]c_int{ t_shape[0], n_target, t_shape[2] };
        const sl_strides = [_]c_int{ 1, 1, 1 };
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&result, tiled, &sl_start, 3, &sl_stop, 3, &sl_strides, 3, self.s));
        return result;
    }
};

// ── Weight Loading Helpers ──

fn getVisionWeight(weights: *const Weights, buf: *[256]u8, layer: usize, suffix: []const u8) mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "vision_tower.encoder.layers.{d}.{s}", .{ layer, suffix }) catch unreachable;
    return weights.get(name) orelse {
        log.err("MISSING VISION WEIGHT: {s}\n", .{name});
        unreachable;
    };
}

fn getWeight(weights: *const Weights, buf: *[256]u8, name: []const u8) ?mlx.mlx_array {
    const n = std.fmt.bufPrint(buf, "{s}", .{name}) catch unreachable;
    return weights.get(n);
}

fn loadClippedLinear(weights: *const Weights, buf: *[256]u8, layer: usize, prefix: []const u8) ClippedLinearWeight {
    return .{
        .weight = getClippedWeight(weights, buf, layer, prefix, ".linear.weight"),
        .input_min = getClippedWeight(weights, buf, layer, prefix, ".input_min"),
        .input_max = getClippedWeight(weights, buf, layer, prefix, ".input_max"),
        .output_min = getClippedWeight(weights, buf, layer, prefix, ".output_min"),
        .output_max = getClippedWeight(weights, buf, layer, prefix, ".output_max"),
    };
}

fn getClippedWeight(weights: *const Weights, buf: *[256]u8, layer: usize, prefix: []const u8, suffix: []const u8) mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "vision_tower.encoder.layers.{d}.{s}{s}", .{ layer, prefix, suffix }) catch unreachable;
    return weights.get(name) orelse {
        log.err("MISSING VISION WEIGHT: {s}\n", .{name});
        unreachable;
    };
}

fn bf16Scalar(val: f32, s: mlx.mlx_stream) mlx.mlx_array {
    const f32_arr = mlx.mlx_array_new_float(val);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf16_arr = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s);
    return bf16_arr;
}

fn detectProjBits(w: mlx.mlx_array, sc: mlx.mlx_array, group_size: u32) u32 {
    if (mlx.mlx_array_ndim(sc) < 2) return 4;
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    if (w_shape.len < 2 or s_shape.len < 2) return 4;
    const w_cols: u32 = @intCast(w_shape[w_shape.len - 1]);
    const s_cols: u32 = @intCast(s_shape[s_shape.len - 1]);
    if (s_cols == 0) return 4;
    return (w_cols * 32) / (s_cols * group_size);
}

// ── Tests ──

const testing = std.testing;

test "VisionEncoder clipped linear concept" {
    const low: f32 = -1.0;
    const high: f32 = 1.0;
    const val: f32 = 2.5;
    const clipped = @min(@max(val, low), high);
    try testing.expectEqual(@as(f32, 1.0), clipped);
}

test "patch grid arithmetic" {
    const image_size: u32 = 224;
    const patch_size: u32 = 16;
    const grid = image_size / patch_size;
    try testing.expectEqual(@as(u32, 14), grid);
    try testing.expectEqual(@as(u32, 196), grid * grid);

    // Pooling: ceil(14/3) = 5, 5*5 = 25 output tokens
    const kernel: u32 = 3;
    const out_dim = (grid + kernel - 1) / kernel;
    try testing.expectEqual(@as(u32, 5), out_dim);
    try testing.expectEqual(@as(u32, 25), out_dim * out_dim);
}

test "patch normalization" {
    // 2*(x-0.5) maps [0,1] → [-1,1]
    try testing.expectEqual(@as(f32, -1.0), 2.0 * (0.0 - 0.5));
    try testing.expectEqual(@as(f32, 0.0), 2.0 * (0.5 - 0.5));
    try testing.expectEqual(@as(f32, 1.0), 2.0 * (1.0 - 0.5));
}
