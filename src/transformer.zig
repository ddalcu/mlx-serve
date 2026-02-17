const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");

const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;

/// Per-layer KV cache entry.
const KVCacheEntry = struct {
    keys: mlx.mlx_array,
    values: mlx.mlx_array,
    initialized: bool,
};

/// KV cache for all layers.
pub const KVCache = struct {
    entries: []KVCacheEntry,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, num_layers: u32) !KVCache {
        const entries = try allocator.alloc(KVCacheEntry, num_layers);
        for (entries) |*e| {
            e.* = .{
                .keys = mlx.mlx_array_new(),
                .values = mlx.mlx_array_new(),
                .initialized = false,
            };
        }
        return .{ .entries = entries, .allocator = allocator };
    }

    pub fn deinit(self: *KVCache) void {
        for (self.entries) |*e| {
            _ = mlx.mlx_array_free(e.keys);
            _ = mlx.mlx_array_free(e.values);
        }
        self.allocator.free(self.entries);
    }

    /// Append new keys/values and return the full accumulated K and V.
    /// max_seq > 0: trim to at most max_seq entries (sliding window for local layers).
    pub fn update(self: *KVCache, layer: u32, new_k: mlx.mlx_array, new_v: mlx.mlx_array, s: mlx.mlx_stream, max_seq: u32) !struct { mlx.mlx_array, mlx.mlx_array } {
        const entry = &self.entries[layer];

        if (!entry.initialized) {
            try mlx.check(mlx.mlx_array_set(&entry.keys, new_k));
            try mlx.check(mlx.mlx_array_set(&entry.values, new_v));
            entry.initialized = true;
            return .{ entry.keys, entry.values };
        }

        const k_arr = [_]mlx.mlx_array{ entry.keys, new_k };
        const v_arr = [_]mlx.mlx_array{ entry.values, new_v };
        const k_vec = mlx.mlx_vector_array_new_data(&k_arr, 2);
        defer _ = mlx.mlx_vector_array_free(k_vec);
        const v_vec = mlx.mlx_vector_array_new_data(&v_arr, 2);
        defer _ = mlx.mlx_vector_array_free(v_vec);

        var new_keys = mlx.mlx_array_new();
        var new_values = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&new_keys, k_vec, 2, s));
        try mlx.check(mlx.mlx_concatenate_axis(&new_values, v_vec, 2, s));

        _ = mlx.mlx_array_free(entry.keys);
        _ = mlx.mlx_array_free(entry.values);
        entry.keys = new_keys;
        entry.values = new_values;

        // Trim to sliding window: keep only the last max_seq entries on axis 2
        if (max_seq > 0) {
            const shape = mlx.getShape(entry.keys);
            const current_seq = shape[2];
            const max_s: c_int = @intCast(max_seq);
            if (current_seq > max_s) {
                const trim_start = current_seq - max_s;
                const start = [_]c_int{ 0, 0, trim_start, 0 };
                const stop = [_]c_int{ shape[0], shape[1], current_seq, shape[3] };
                const strides = [_]c_int{ 1, 1, 1, 1 };

                var trimmed_k = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice(&trimmed_k, entry.keys, &start, 4, &stop, 4, &strides, 4, s));
                _ = mlx.mlx_array_free(entry.keys);
                entry.keys = trimmed_k;

                var trimmed_v = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice(&trimmed_v, entry.values, &start, 4, &stop, 4, &strides, 4, s));
                _ = mlx.mlx_array_free(entry.values);
                entry.values = trimmed_v;
            }
        }

        return .{ entry.keys, entry.values };
    }

    pub fn seqLen(self: *const KVCache, layer: u32) usize {
        const entry = &self.entries[layer];
        if (!entry.initialized) return 0;
        const shape = mlx.getShape(entry.keys);
        if (shape.len < 3) return 0;
        return @intCast(shape[2]);
    }
};

/// Precomputed per-layer weight references (avoids hash lookups during forward pass).
const LayerWeights = struct {
    // Norm weights (precomputed for Gemma: 1+weight; direct for Llama-family)
    input_norm: mlx.mlx_array,
    post_attn_norm: mlx.mlx_array,
    pre_ff_norm: ?mlx.mlx_array, // null for Llama-family (only 2 norms)
    post_ff_norm: ?mlx.mlx_array, // null for Llama-family
    q_norm: ?mlx.mlx_array, // null for models without Q/K norms
    k_norm: ?mlx.mlx_array, // null for models without Q/K norms
    // Attention projection weights (quantized: weight, scales, biases)
    q_w: mlx.mlx_array,
    q_s: mlx.mlx_array,
    q_b: mlx.mlx_array,
    k_w: mlx.mlx_array,
    k_s: mlx.mlx_array,
    k_b: mlx.mlx_array,
    v_w: mlx.mlx_array,
    v_s: mlx.mlx_array,
    v_b: mlx.mlx_array,
    o_w: mlx.mlx_array,
    o_s: mlx.mlx_array,
    o_b: mlx.mlx_array,
    // MLP weights (quantized)
    gate_w: mlx.mlx_array,
    gate_s: mlx.mlx_array,
    gate_b: mlx.mlx_array,
    up_w: mlx.mlx_array,
    up_s: mlx.mlx_array,
    up_b: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_s: mlx.mlx_array,
    down_b: mlx.mlx_array,
};

/// Full transformer state for inference.
pub const Transformer = struct {
    config: ModelConfig,
    cache: KVCache,
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,

    // Quantized embedding references (not owned — Weights map owns these)
    emb_w: mlx.mlx_array,
    emb_s: mlx.mlx_array,
    emb_b: mlx.mlx_array,
    emb_scale: ?mlx.mlx_array, // sqrt(hidden_size) for Gemma, null for others
    final_norm: mlx.mlx_array, // precomputed norm weight
    lm_head_w: mlx.mlx_array,
    lm_head_s: mlx.mlx_array,
    lm_head_b: mlx.mlx_array,
    layers: []LayerWeights,

    // Whether we own lm_head arrays (false when tied to embeddings)
    owns_lm_head: bool,
    // Whether we own norm arrays (true when we computed 1+weight)
    owns_norms: bool,

    // Constant arrays reused every step
    gelu_coeff: ?mlx.mlx_array, // sqrt(2/pi) for GELU approx
    gelu_inner: ?mlx.mlx_array, // 0.044715 for GELU approx
    half: mlx.mlx_array, // 0.5
    one: mlx.mlx_array, // 1.0
    three: ?mlx.mlx_array, // 3.0 for GELU approx (x^3)
    neg_one: ?mlx.mlx_array, // -1.0 for SiLU: sigmoid = 1/(1+exp(-x))

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !Transformer {
        const cache = try KVCache.init(allocator, config.num_hidden_layers);
        const s = mlx.mlx_default_gpu_stream_new();
        const prefix = config.weight_prefix;

        // Embedding weights
        var name_buf: [256]u8 = undefined;
        const emb_w = getWeightFmt(weights, &name_buf, "{s}.embed_tokens.weight", prefix);
        const emb_s_arr = getWeightFmt(weights, &name_buf, "{s}.embed_tokens.scales", prefix);
        const emb_b_arr = getWeightFmt(weights, &name_buf, "{s}.embed_tokens.biases", prefix);

        // Embedding scale (Gemma only)
        const emb_scale: ?mlx.mlx_array = if (config.scale_embeddings)
            bf16Scalar(@sqrt(@as(f32, @floatFromInt(config.hidden_size))), s)
        else
            null;

        // Final norm weight
        const final_norm_raw = getWeightFmt(weights, &name_buf, "{s}.norm.weight", prefix);
        const final_norm = if (config.norm_has_offset) try addOne(final_norm_raw, s) else final_norm_raw;
        if (config.norm_has_offset) try mlx.check(mlx.mlx_array_eval(final_norm));

        // LM head weights: separate or tied to embeddings
        var lm_head_w: mlx.mlx_array = undefined;
        var lm_head_s: mlx.mlx_array = undefined;
        var lm_head_b: mlx.mlx_array = undefined;
        var owns_lm_head = false;

        if (config.tie_word_embeddings) {
            // Check if separate lm_head weights exist anyway (some quantized models have them)
            const sep_prefix = if (std.mem.eql(u8, prefix, "language_model.model")) "language_model" else prefix;
            const maybe_lm_w = getWeightFmtOpt(weights, &name_buf, "{s}.lm_head.weight", sep_prefix);
            if (maybe_lm_w != null) {
                lm_head_w = maybe_lm_w.?;
                lm_head_s = getWeightFmt(weights, &name_buf, "{s}.lm_head.scales", sep_prefix);
                lm_head_b = getWeightFmt(weights, &name_buf, "{s}.lm_head.biases", sep_prefix);
            } else {
                // Truly tied: reuse embedding weights
                lm_head_w = emb_w;
                lm_head_s = emb_s_arr;
                lm_head_b = emb_b_arr;
            }
        } else {
            // Separate lm_head
            const lm_prefix = if (std.mem.eql(u8, prefix, "language_model.model")) "language_model" else prefix;
            lm_head_w = getWeightFmt(weights, &name_buf, "{s}.lm_head.weight", lm_prefix);
            lm_head_s = getWeightFmt(weights, &name_buf, "{s}.lm_head.scales", lm_prefix);
            lm_head_b = getWeightFmt(weights, &name_buf, "{s}.lm_head.biases", lm_prefix);
            owns_lm_head = true;
        }

        // Precompute per-layer weights
        std.debug.print("Precomputing layer weights...\n", .{});
        const layers = try allocator.alloc(LayerWeights, config.num_hidden_layers);

        for (0..config.num_hidden_layers) |i| {
            const li: u32 = @intCast(i);
            const lw = &layers[i];

            // Norm weights
            const input_norm_raw = getLayerWeight(weights, &name_buf, prefix, li, "input_layernorm.weight");
            lw.input_norm = if (config.norm_has_offset) try addOne(input_norm_raw, s) else input_norm_raw;

            const post_attn_raw = getLayerWeight(weights, &name_buf, prefix, li, "post_attention_layernorm.weight");
            lw.post_attn_norm = if (config.norm_has_offset) try addOne(post_attn_raw, s) else post_attn_raw;

            // Pre-FF / Post-FF norms (Gemma only: 4 norms per layer)
            if (config.has_pre_ff_norm) {
                const pre_ff_raw = getLayerWeight(weights, &name_buf, prefix, li, "pre_feedforward_layernorm.weight");
                lw.pre_ff_norm = if (config.norm_has_offset) try addOne(pre_ff_raw, s) else pre_ff_raw;
                const post_ff_raw = getLayerWeight(weights, &name_buf, prefix, li, "post_feedforward_layernorm.weight");
                lw.post_ff_norm = if (config.norm_has_offset) try addOne(post_ff_raw, s) else post_ff_raw;
            } else {
                lw.pre_ff_norm = null;
                lw.post_ff_norm = null;
            }

            // Q/K norms (Gemma and Qwen3 have these)
            if (config.has_qk_norm) {
                const q_norm_raw = getLayerWeight(weights, &name_buf, prefix, li, "self_attn.q_norm.weight");
                lw.q_norm = if (config.norm_has_offset) try addOne(q_norm_raw, s) else q_norm_raw;
                const k_norm_raw = getLayerWeight(weights, &name_buf, prefix, li, "self_attn.k_norm.weight");
                lw.k_norm = if (config.norm_has_offset) try addOne(k_norm_raw, s) else k_norm_raw;
            } else {
                lw.q_norm = null;
                lw.k_norm = null;
            }

            // Attention projection weights
            lw.q_w = getLayerWeight(weights, &name_buf, prefix, li, "self_attn.q_proj.weight");
            lw.q_s = getLayerWeight(weights, &name_buf, prefix, li, "self_attn.q_proj.scales");
            lw.q_b = getLayerWeight(weights, &name_buf, prefix, li, "self_attn.q_proj.biases");
            lw.k_w = getLayerWeight(weights, &name_buf, prefix, li, "self_attn.k_proj.weight");
            lw.k_s = getLayerWeight(weights, &name_buf, prefix, li, "self_attn.k_proj.scales");
            lw.k_b = getLayerWeight(weights, &name_buf, prefix, li, "self_attn.k_proj.biases");
            lw.v_w = getLayerWeight(weights, &name_buf, prefix, li, "self_attn.v_proj.weight");
            lw.v_s = getLayerWeight(weights, &name_buf, prefix, li, "self_attn.v_proj.scales");
            lw.v_b = getLayerWeight(weights, &name_buf, prefix, li, "self_attn.v_proj.biases");
            lw.o_w = getLayerWeight(weights, &name_buf, prefix, li, "self_attn.o_proj.weight");
            lw.o_s = getLayerWeight(weights, &name_buf, prefix, li, "self_attn.o_proj.scales");
            lw.o_b = getLayerWeight(weights, &name_buf, prefix, li, "self_attn.o_proj.biases");

            // MLP weights
            lw.gate_w = getLayerWeight(weights, &name_buf, prefix, li, "mlp.gate_proj.weight");
            lw.gate_s = getLayerWeight(weights, &name_buf, prefix, li, "mlp.gate_proj.scales");
            lw.gate_b = getLayerWeight(weights, &name_buf, prefix, li, "mlp.gate_proj.biases");
            lw.up_w = getLayerWeight(weights, &name_buf, prefix, li, "mlp.up_proj.weight");
            lw.up_s = getLayerWeight(weights, &name_buf, prefix, li, "mlp.up_proj.scales");
            lw.up_b = getLayerWeight(weights, &name_buf, prefix, li, "mlp.up_proj.biases");
            lw.down_w = getLayerWeight(weights, &name_buf, prefix, li, "mlp.down_proj.weight");
            lw.down_s = getLayerWeight(weights, &name_buf, prefix, li, "mlp.down_proj.scales");
            lw.down_b = getLayerWeight(weights, &name_buf, prefix, li, "mlp.down_proj.biases");
        }

        // Batch eval ALL weights in one call to force GPU transfer upfront.
        {
            var eval_timer = std.time.Timer.start() catch unreachable;
            const all_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(all_vec);

            for (layers) |lw| {
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.input_norm);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.post_attn_norm);
                if (lw.pre_ff_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                if (lw.post_ff_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                if (lw.q_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                if (lw.k_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
            }

            for (layers) |lw| {
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.q_w);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.q_s);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.q_b);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.k_w);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.k_s);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.k_b);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.v_w);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.v_s);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.v_b);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.o_w);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.o_s);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.o_b);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.gate_w);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.gate_s);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.gate_b);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.up_w);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.up_s);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.up_b);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.down_w);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.down_s);
                _ = mlx.mlx_vector_array_append_value(all_vec, lw.down_b);
            }

            _ = mlx.mlx_vector_array_append_value(all_vec, emb_w);
            _ = mlx.mlx_vector_array_append_value(all_vec, emb_s_arr);
            _ = mlx.mlx_vector_array_append_value(all_vec, emb_b_arr);
            _ = mlx.mlx_vector_array_append_value(all_vec, lm_head_w);
            _ = mlx.mlx_vector_array_append_value(all_vec, lm_head_s);
            _ = mlx.mlx_vector_array_append_value(all_vec, lm_head_b);

            try mlx.check(mlx.mlx_eval(all_vec));
            const eval_ms = eval_timer.read() / std.time.ns_per_ms;
            std.debug.print("Batch eval all weights: {d}ms\n", .{eval_ms});
        }

        // Activation constants
        const need_gelu = config.hidden_act == .gelu_approx;
        const need_silu = config.hidden_act == .silu;

        return .{
            .config = config,
            .cache = cache,
            .s = s,
            .allocator = allocator,
            .emb_w = emb_w,
            .emb_s = emb_s_arr,
            .emb_b = emb_b_arr,
            .emb_scale = emb_scale,
            .final_norm = final_norm,
            .lm_head_w = lm_head_w,
            .lm_head_s = lm_head_s,
            .lm_head_b = lm_head_b,
            .layers = layers,
            .owns_lm_head = owns_lm_head,
            .owns_norms = config.norm_has_offset,
            .gelu_coeff = if (need_gelu) bf16Scalar(0.7978845608028654, s) else null,
            .gelu_inner = if (need_gelu) bf16Scalar(0.044715, s) else null,
            .half = bf16Scalar(0.5, s),
            .one = bf16Scalar(1.0, s),
            .three = if (need_gelu) bf16Scalar(3.0, s) else null,
            .neg_one = if (need_silu) bf16Scalar(-1.0, s) else null,
        };
    }

    pub fn deinit(self: *Transformer) void {
        self.cache.deinit();
        if (self.emb_scale) |es| _ = mlx.mlx_array_free(es);
        if (self.owns_norms) _ = mlx.mlx_array_free(self.final_norm);
        if (self.gelu_coeff) |g| _ = mlx.mlx_array_free(g);
        if (self.gelu_inner) |g| _ = mlx.mlx_array_free(g);
        _ = mlx.mlx_array_free(self.half);
        _ = mlx.mlx_array_free(self.one);
        if (self.three) |t| _ = mlx.mlx_array_free(t);
        if (self.neg_one) |n| _ = mlx.mlx_array_free(n);
        for (self.layers) |lw| {
            if (self.owns_norms) {
                _ = mlx.mlx_array_free(lw.input_norm);
                _ = mlx.mlx_array_free(lw.post_attn_norm);
                if (lw.pre_ff_norm) |n| _ = mlx.mlx_array_free(n);
                if (lw.post_ff_norm) |n| _ = mlx.mlx_array_free(n);
                if (lw.q_norm) |n| _ = mlx.mlx_array_free(n);
                if (lw.k_norm) |n| _ = mlx.mlx_array_free(n);
            }
        }
        self.allocator.free(self.layers);
        _ = mlx.mlx_stream_free(self.s);
    }

    /// Quantized matmul with pre-resolved weight references.
    inline fn qmatmul(self: *const Transformer, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array) !mlx.mlx_array {
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_quantized_matmul(
            &result,
            x,
            w,
            sc,
            bi,
            true,
            mlx.mlx_optional_int.some(@intCast(self.config.quant_group_size)),
            mlx.mlx_optional_int.some(@intCast(self.config.quant_bits)),
            "affine",
            self.s,
        ));
        return result;
    }

    /// Quantized embedding lookup: gather rows then dequantize (saves ~1.9GB vs full table).
    fn embedding(self: *const Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
        const id_shape = mlx.getShape(token_ids);
        const batch = id_shape[0];
        const seq_len = id_shape[1];

        // Flatten to 1D for take
        const flat_shape = [_]c_int{batch * seq_len};
        var flat_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat_ids);
        try mlx.check(mlx.mlx_reshape(&flat_ids, token_ids, &flat_shape, 1, self.s));

        // Gather quantized rows for just the needed tokens
        var taken_w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken_w);
        try mlx.check(mlx.mlx_take_axis(&taken_w, self.emb_w, flat_ids, 0, self.s));

        var taken_s = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken_s);
        try mlx.check(mlx.mlx_take_axis(&taken_s, self.emb_s, flat_ids, 0, self.s));

        var taken_b = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken_b);
        try mlx.check(mlx.mlx_take_axis(&taken_b, self.emb_b, flat_ids, 0, self.s));

        // Dequantize only the gathered rows
        var emb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(emb);
        try mlx.check(mlx.mlx_dequantize(
            &emb,
            taken_w,
            taken_s,
            taken_b,
            mlx.mlx_optional_int.some(@intCast(self.config.quant_group_size)),
            mlx.mlx_optional_int.some(@intCast(self.config.quant_bits)),
            "affine",
            .{ .value = .bfloat16, .has_value = true },
            self.s,
        ));

        // Reshape to [batch, seq_len, hidden_size]
        const out_shape = [_]c_int{ batch, seq_len, @intCast(self.config.hidden_size) };
        var reshaped = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(reshaped);
        try mlx.check(mlx.mlx_reshape(&reshaped, emb, &out_shape, 3, self.s));

        // Scale embeddings (Gemma: * sqrt(hidden_size))
        if (self.emb_scale) |scale| {
            var scaled = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&scaled, reshaped, scale, self.s));
            return scaled;
        }
        // No scaling — return a copy (caller expects to own the result)
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&result, reshaped));
        return result;
    }

    /// RMSNorm with precomputed weight (1+weight for Gemma, raw weight for Llama-family).
    inline fn rmsNorm(self: *const Transformer, x: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_rms_norm(&result, x, w, self.config.rms_norm_eps, self.s));
        return result;
    }

    /// GELU approximate: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    fn gelu(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        var x3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x3);
        try mlx.check(mlx.mlx_power(&x3, x, self.three.?, self.s));

        var inner = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inner);
        try mlx.check(mlx.mlx_multiply(&inner, self.gelu_inner.?, x3, self.s));

        var sum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_add(&sum, x, inner, self.s));

        var scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_multiply(&scaled, self.gelu_coeff.?, sum, self.s));

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

    /// SiLU (Swish): x * sigmoid(x) = x / (1 + exp(-x))
    fn silu(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        // -x
        var neg_x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(neg_x);
        try mlx.check(mlx.mlx_multiply(&neg_x, x, self.neg_one.?, self.s));

        // exp(-x)
        var exp_neg_x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(exp_neg_x);
        try mlx.check(mlx.mlx_exp(&exp_neg_x, neg_x, self.s));

        // 1 + exp(-x)
        var denom = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(denom);
        try mlx.check(mlx.mlx_add(&denom, self.one, exp_neg_x, self.s));

        // x / (1 + exp(-x))
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_divide(&result, x, denom, self.s));
        return result;
    }

    /// Apply the configured MLP activation function.
    inline fn mlpActivation(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        return switch (self.config.hidden_act) {
            .gelu_approx => self.gelu(x),
            .silu => self.silu(x),
        };
    }

    /// How many layers to process before forcing evaluation to cap memory.
    const EVAL_EVERY_N_LAYERS: u32 = 48;

    /// Forward pass: token_ids [batch, seq_len] -> logits [batch, seq_len, vocab_size]
    pub fn forward(self: *Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
        const offset = self.cache.seqLen(0);
        const h_count = self.config.num_attention_heads;
        const kv_h = self.config.num_key_value_heads;
        const hd = self.config.head_dim;
        const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.config.query_pre_attn_scalar)));

        // Embedding
        var h = try self.embedding(token_ids);

        const x_shape = mlx.getShape(h);
        const batch: c_int = x_shape[0];
        const seq_len: c_int = x_shape[1];

        const is_prefill = seq_len > 1;

        // Shapes for reshape ops (constant for all layers)
        const q_shape = [_]c_int{ batch, seq_len, @intCast(h_count), @intCast(hd) };
        const kv_shape = [_]c_int{ batch, seq_len, @intCast(kv_h), @intCast(hd) };
        const perm = [_]c_int{ 0, 2, 1, 3 };
        const perm_back = [_]c_int{ 0, 2, 1, 3 };
        const out_shape = [_]c_int{ batch, seq_len, @intCast(h_count * hd) };

        const none_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(none_mask);

        // Pre-compute sliding window masks (only for models that use them)
        const total_kv: c_int = @as(c_int, @intCast(offset)) + seq_len;
        var local_prefill_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(local_prefill_mask);
        var local_decode_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(local_decode_mask);

        if (self.config.has_sliding_window) {
            const sw: c_int = @intCast(self.config.sliding_window);
            if (is_prefill) {
                local_prefill_mask = try self.createSlidingWindowMask(seq_len, total_kv, sw);
            }
            const local_total_kv: c_int = @min(total_kv, sw);
            if (!is_prefill and local_total_kv > sw) {
                local_decode_mask = try self.createSlidingWindowDecodeMask(local_total_kv, sw);
            }
        }

        // Process each layer
        for (0..self.config.num_hidden_layers) |layer_idx| {
            const li: u32 = @intCast(layer_idx);
            const lw = &self.layers[layer_idx];
            const is_global = self.config.isGlobalLayer(li);

            // Pre-attention RMSNorm
            const normed = try self.rmsNorm(h, lw.input_norm);
            defer _ = mlx.mlx_array_free(normed);

            // Q/K/V projections
            const q = try self.qmatmul(normed, lw.q_w, lw.q_s, lw.q_b);
            defer _ = mlx.mlx_array_free(q);
            const k = try self.qmatmul(normed, lw.k_w, lw.k_s, lw.k_b);
            defer _ = mlx.mlx_array_free(k);
            const v = try self.qmatmul(normed, lw.v_w, lw.v_s, lw.v_b);
            defer _ = mlx.mlx_array_free(v);

            // Reshape to [B, S, H, D]
            var q_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_r);
            var k_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_r);
            var v_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_r);
            try mlx.check(mlx.mlx_reshape(&q_r, q, &q_shape, 4, self.s));
            try mlx.check(mlx.mlx_reshape(&k_r, k, &kv_shape, 4, self.s));
            try mlx.check(mlx.mlx_reshape(&v_r, v, &kv_shape, 4, self.s));

            // Q/K RMSNorm (optional)
            const q_normed: ?mlx.mlx_array = if (lw.q_norm) |qn| try self.rmsNorm(q_r, qn) else null;
            defer {
                if (q_normed) |qn| _ = mlx.mlx_array_free(qn);
            }
            const k_normed: ?mlx.mlx_array = if (lw.k_norm) |kn| try self.rmsNorm(k_r, kn) else null;
            defer {
                if (k_normed) |kn| _ = mlx.mlx_array_free(kn);
            }

            const q_for_rope = q_normed orelse q_r;
            const k_for_rope = k_normed orelse k_r;

            // Transpose to [B, H, S, D]
            var q_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_t);
            var k_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_t);
            var v_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_t);
            try mlx.check(mlx.mlx_transpose_axes(&q_t, q_for_rope, &perm, 4, self.s));
            try mlx.check(mlx.mlx_transpose_axes(&k_t, k_for_rope, &perm, 4, self.s));
            try mlx.check(mlx.mlx_transpose_axes(&v_t, v_r, &perm, 4, self.s));

            // RoPE
            const rope_base: f32 = if (is_global) self.config.rope_theta else self.config.rope_local_base_freq;
            const rope_scale: f32 = if (is_global) (1.0 / self.config.rope_scaling_factor) else 1.0;

            var q_rope = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_rope);
            var k_rope = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_rope);

            try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, @intCast(hd), false, mlx.mlx_optional_float.some(rope_base), rope_scale, @intCast(offset), .{ .ctx = null }, self.s));
            try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, @intCast(hd), false, mlx.mlx_optional_float.some(rope_base), rope_scale, @intCast(offset), .{ .ctx = null }, self.s));

            // KV cache update (trim local layers to sliding window)
            const max_kv: u32 = if (is_global) 0 else if (self.config.has_sliding_window) self.config.sliding_window else 0;
            const kv = try self.cache.update(li, k_rope, v_t, self.s, max_kv);
            const full_k = kv[0];
            const full_v = kv[1];

            // Attention
            var attn_out = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_out);

            if (!self.config.has_sliding_window) {
                // No sliding window — simple causal for prefill, none for decode
                if (is_prefill) {
                    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, self.s));
                } else {
                    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, self.s));
                }
            } else {
                // Sliding window model (Gemma)
                const sw: c_int = @intCast(self.config.sliding_window);
                if (is_prefill and is_global) {
                    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, self.s));
                } else if (is_prefill) {
                    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "array", local_prefill_mask, .{ .ctx = null }, self.s));
                } else if (is_global or @as(c_int, @intCast(self.cache.seqLen(li))) <= sw) {
                    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, self.s));
                } else {
                    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "array", local_decode_mask, .{ .ctx = null }, self.s));
                }
            }

            // Transpose back [B,H,S,D] -> [B,S,H,D] -> [B,S,H*D]
            var attn_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_t);
            try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm_back, 4, self.s));
            var attn_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_flat);
            try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &out_shape, 3, self.s));

            // Output projection
            const o_out = try self.qmatmul(attn_flat, lw.o_w, lw.o_s, lw.o_b);
            defer _ = mlx.mlx_array_free(o_out);

            if (self.config.has_pre_ff_norm) {
                // Gemma path: 4 norms per layer
                // post_attn_norm on attention output, add residual
                const attn_normed = try self.rmsNorm(o_out, lw.post_attn_norm);
                defer _ = mlx.mlx_array_free(attn_normed);
                var h_new = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_new, h, attn_normed, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_new;

                // pre_ff_norm -> MLP -> post_ff_norm -> add residual
                const ff_normed = try self.rmsNorm(h, lw.pre_ff_norm.?);
                defer _ = mlx.mlx_array_free(ff_normed);

                const gate_raw = try self.qmatmul(ff_normed, lw.gate_w, lw.gate_s, lw.gate_b);
                defer _ = mlx.mlx_array_free(gate_raw);
                const gate = try self.mlpActivation(gate_raw);
                defer _ = mlx.mlx_array_free(gate);
                const up = try self.qmatmul(ff_normed, lw.up_w, lw.up_s, lw.up_b);
                defer _ = mlx.mlx_array_free(up);
                var gate_up = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(gate_up);
                try mlx.check(mlx.mlx_multiply(&gate_up, gate, up, self.s));
                const down = try self.qmatmul(gate_up, lw.down_w, lw.down_s, lw.down_b);
                defer _ = mlx.mlx_array_free(down);

                const mlp_normed = try self.rmsNorm(down, lw.post_ff_norm.?);
                defer _ = mlx.mlx_array_free(mlp_normed);
                var h_next = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_next, h, mlp_normed, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_next;
            } else {
                // Llama-family path: 2 norms per layer
                // Add attention output directly to residual
                var h_new = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_new, h, o_out, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_new;

                // post_attention_layernorm serves as pre-MLP norm
                const ff_normed = try self.rmsNorm(h, lw.post_attn_norm);
                defer _ = mlx.mlx_array_free(ff_normed);

                const gate_raw = try self.qmatmul(ff_normed, lw.gate_w, lw.gate_s, lw.gate_b);
                defer _ = mlx.mlx_array_free(gate_raw);
                const gate = try self.mlpActivation(gate_raw);
                defer _ = mlx.mlx_array_free(gate);
                const up = try self.qmatmul(ff_normed, lw.up_w, lw.up_s, lw.up_b);
                defer _ = mlx.mlx_array_free(up);
                var gate_up = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(gate_up);
                try mlx.check(mlx.mlx_multiply(&gate_up, gate, up, self.s));
                const down = try self.qmatmul(gate_up, lw.down_w, lw.down_s, lw.down_b);
                defer _ = mlx.mlx_array_free(down);

                // Add MLP output to residual
                var h_next = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_next, h, down, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_next;
            }

            // During prefill, periodically eval to prevent the lazy graph from
            // exhausting unified memory (which causes an unrecoverable SoC crash).
            if (is_prefill and (layer_idx + 1) % EVAL_EVERY_N_LAYERS == 0) {
                try mlx.check(mlx.mlx_array_eval(h));
            }
        }

        // Final norm + LM head
        const final_normed = try self.rmsNorm(h, self.final_norm);
        _ = mlx.mlx_array_free(h);

        const logits = try self.qmatmul(final_normed, self.lm_head_w, self.lm_head_s, self.lm_head_b);
        _ = mlx.mlx_array_free(final_normed);

        return logits;
    }

    /// Causal mask [1, 1, q_len, kv_len] for prefill (bfloat16).
    fn createCausalMask(self: *const Transformer, q_len: c_int, kv_len: c_int) !mlx.mlx_array {
        const offset_val = kv_len - q_len;
        const shape = [_]c_int{ q_len, kv_len };

        var ones = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ones);
        try mlx.check(mlx.mlx_full(&ones, &shape, 2, self.one, .bfloat16, self.s));

        var upper = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(upper);
        try mlx.check(mlx.mlx_triu(&upper, ones, offset_val + 1, self.s));

        var bool_upper = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bool_upper);
        try mlx.check(mlx.mlx_astype(&bool_upper, upper, .bool_, self.s));

        const zero = bf16Scalar(0.0, self.s);
        defer _ = mlx.mlx_array_free(zero);
        const neg_inf = bf16Scalar(-std.math.inf(f32), self.s);
        defer _ = mlx.mlx_array_free(neg_inf);

        var mask = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_where(&mask, bool_upper, neg_inf, zero, self.s));

        const mask_shape = [_]c_int{ 1, 1, q_len, kv_len };
        var mask_4d = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&mask_4d, mask, &mask_shape, 4, self.s));
        _ = mlx.mlx_array_free(mask);
        return mask_4d;
    }

    /// Sliding window decode mask [1, 1, 1, kv_len] (bfloat16).
    fn createSlidingWindowDecodeMask(self: *const Transformer, kv_len: c_int, window: c_int) !mlx.mlx_array {
        var positions = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(positions);
        try mlx.check(mlx.mlx_arange(&positions, 0, @floatFromInt(kv_len), 1, .int32, self.s));

        const window_start = mlx.mlx_array_new_int(kv_len - window);
        defer _ = mlx.mlx_array_free(window_start);

        var too_old = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(too_old);
        try mlx.check(mlx.mlx_less(&too_old, positions, window_start, self.s));

        const zero = bf16Scalar(0.0, self.s);
        defer _ = mlx.mlx_array_free(zero);
        const neg_inf = bf16Scalar(-std.math.inf(f32), self.s);
        defer _ = mlx.mlx_array_free(neg_inf);

        var sw_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sw_mask);
        try mlx.check(mlx.mlx_where(&sw_mask, too_old, neg_inf, zero, self.s));

        const mask_shape = [_]c_int{ 1, 1, 1, kv_len };
        var mask_4d = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&mask_4d, sw_mask, &mask_shape, 4, self.s));
        return mask_4d;
    }

    /// Sliding window + causal mask for prefill.
    fn createSlidingWindowMask(self: *const Transformer, q_len: c_int, kv_len: c_int, window: c_int) !mlx.mlx_array {
        const causal = try self.createCausalMask(q_len, kv_len);
        defer _ = mlx.mlx_array_free(causal);

        const offset_val = kv_len - q_len;
        var row_idx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row_idx);
        try mlx.check(mlx.mlx_arange(&row_idx, @floatFromInt(offset_val), @floatFromInt(offset_val + q_len), 1, .int32, self.s));
        var col_idx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(col_idx);
        try mlx.check(mlx.mlx_arange(&col_idx, 0, @floatFromInt(kv_len), 1, .int32, self.s));

        const row_shape = [_]c_int{ q_len, 1 };
        const col_shape = [_]c_int{ 1, kv_len };
        var row_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row_r);
        var col_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(col_r);
        try mlx.check(mlx.mlx_reshape(&row_r, row_idx, &row_shape, 2, self.s));
        try mlx.check(mlx.mlx_reshape(&col_r, col_idx, &col_shape, 2, self.s));

        var dist = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dist);
        try mlx.check(mlx.mlx_subtract(&dist, row_r, col_r, self.s));

        const window_arr = mlx.mlx_array_new_int(window);
        defer _ = mlx.mlx_array_free(window_arr);
        var too_far = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(too_far);
        try mlx.check(mlx.mlx_greater_equal(&too_far, dist, window_arr, self.s));

        const neg_inf = bf16Scalar(-std.math.inf(f32), self.s);
        defer _ = mlx.mlx_array_free(neg_inf);
        const zero = bf16Scalar(0.0, self.s);
        defer _ = mlx.mlx_array_free(zero);
        var sw_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sw_mask);
        try mlx.check(mlx.mlx_where(&sw_mask, too_far, neg_inf, zero, self.s));

        const mask_shape = [_]c_int{ 1, 1, q_len, kv_len };
        var sw_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sw_4d);
        try mlx.check(mlx.mlx_reshape(&sw_4d, sw_mask, &mask_shape, 4, self.s));

        var combined = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&combined, causal, sw_4d, self.s));
        return combined;
    }
};

/// Get a weight by formatted name with a string prefix. Returns the weight or panics.
fn getWeightFmt(weights: *const Weights, buf: *[256]u8, comptime fmt: []const u8, prefix: []const u8) mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, fmt, .{prefix}) catch unreachable;
    return weights.get(name) orelse {
        std.debug.print("MISSING WEIGHT: {s}\n", .{name});
        unreachable;
    };
}

/// Get a weight by formatted name, returning null if not found.
fn getWeightFmtOpt(weights: *const Weights, buf: *[256]u8, comptime fmt: []const u8, prefix: []const u8) ?mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, fmt, .{prefix}) catch unreachable;
    return weights.get(name);
}

/// Get a per-layer weight by prefix, layer index, and suffix.
fn getLayerWeight(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8) mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ prefix, layer, suffix }) catch unreachable;
    return weights.get(name) orelse {
        std.debug.print("MISSING WEIGHT: {s}\n", .{name});
        unreachable;
    };
}

/// Create a bfloat16 scalar constant.
fn bf16Scalar(val: f32, s: mlx.mlx_stream) mlx.mlx_array {
    const f32_arr = mlx.mlx_array_new_float(val);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf16_arr = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s);
    return bf16_arr;
}

/// Compute 1.0 + arr in bfloat16 (caller owns result).
fn addOne(arr: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const one = bf16Scalar(1.0, s);
    defer _ = mlx.mlx_array_free(one);
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&result, one, arr, s));
    return result;
}
