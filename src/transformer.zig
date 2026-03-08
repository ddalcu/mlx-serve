const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");

const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;

// ── KV Cache (standard attention) ──

const KVCacheEntry = struct {
    keys: mlx.mlx_array,
    values: mlx.mlx_array,
    initialized: bool,
};

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

// ── SSM Cache (for GatedDeltaNet linear attention layers) ──

const SSMCacheEntry = struct {
    conv_state: mlx.mlx_array, // [B, kernel-1, conv_dim]
    ssm_state: mlx.mlx_array, // [B, Hv, Dv, Dk]
    initialized: bool,
};

// ── Prompt Cache (snapshot of KV + SSM state for prefix reuse) ──

pub const PrefillCache = struct {
    tokens: []u32,
    kv_entries: []KVCacheEntry,
    ssm_entries: ?[]SSMCacheEntry,
    moe_seq_offset: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *PrefillCache) void {
        self.allocator.free(self.tokens);
        for (self.kv_entries) |*e| {
            _ = mlx.mlx_array_free(e.keys);
            _ = mlx.mlx_array_free(e.values);
        }
        self.allocator.free(self.kv_entries);
        if (self.ssm_entries) |entries| {
            for (entries) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
            }
            self.allocator.free(entries);
        }
    }
};

// ── Standard model per-layer weights ──

const LayerWeights = struct {
    input_norm: mlx.mlx_array,
    post_attn_norm: mlx.mlx_array,
    pre_ff_norm: ?mlx.mlx_array,
    post_ff_norm: ?mlx.mlx_array,
    q_norm: ?mlx.mlx_array,
    k_norm: ?mlx.mlx_array,
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

// ── MoE model per-layer weights ──

const FullAttnWeights = struct {
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
    q_norm: mlx.mlx_array,
    k_norm: mlx.mlx_array,
};

const LinearAttnWeights = struct {
    // For separate projections (qwen3_5_moe): qkv=QKV, z=Z, a=A, b=B
    // For combined projections (qwen3_next): qkv=QKVZ, b=BA, z/a unused
    combined_proj: bool = false,
    qkv_w: mlx.mlx_array,
    qkv_s: mlx.mlx_array,
    qkv_b: mlx.mlx_array,
    z_w: mlx.mlx_array,
    z_s: mlx.mlx_array,
    z_b: mlx.mlx_array,
    a_w: mlx.mlx_array,
    a_s: mlx.mlx_array,
    a_b: mlx.mlx_array,
    b_w: mlx.mlx_array,
    b_s: mlx.mlx_array,
    b_b: mlx.mlx_array,
    conv1d_w: mlx.mlx_array,
    A_log: mlx.mlx_array,
    dt_bias: mlx.mlx_array,
    norm_w: mlx.mlx_array,
    out_w: mlx.mlx_array,
    out_s: mlx.mlx_array,
    out_b: mlx.mlx_array,
};

const MoeMlpWeights = struct {
    router_w: mlx.mlx_array,
    router_s: mlx.mlx_array,
    router_b: mlx.mlx_array,
    switch_gate_w: mlx.mlx_array,
    switch_gate_s: mlx.mlx_array,
    switch_gate_b: mlx.mlx_array,
    switch_up_w: mlx.mlx_array,
    switch_up_s: mlx.mlx_array,
    switch_up_b: mlx.mlx_array,
    switch_down_w: mlx.mlx_array,
    switch_down_s: mlx.mlx_array,
    switch_down_b: mlx.mlx_array,
    shared_gate_w: mlx.mlx_array,
    shared_gate_s: mlx.mlx_array,
    shared_gate_b: mlx.mlx_array,
    shared_up_w: mlx.mlx_array,
    shared_up_s: mlx.mlx_array,
    shared_up_b: mlx.mlx_array,
    shared_down_w: mlx.mlx_array,
    shared_down_s: mlx.mlx_array,
    shared_down_b: mlx.mlx_array,
    shared_expert_gate_w: mlx.mlx_array,
    shared_expert_gate_s: mlx.mlx_array,
    shared_expert_gate_b: mlx.mlx_array,
};

const MoeLayerWeights = struct {
    input_norm: mlx.mlx_array,
    post_attn_norm: mlx.mlx_array,
    is_linear: bool,
    attn: union(enum) { full: FullAttnWeights, linear: LinearAttnWeights },
    mlp: MoeMlpWeights,
};

// ── Transformer ──

pub const Transformer = struct {
    config: ModelConfig,
    cache: KVCache,
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,

    emb_w: mlx.mlx_array,
    emb_s: mlx.mlx_array,
    emb_b: mlx.mlx_array,
    emb_scale: ?mlx.mlx_array,
    final_norm: mlx.mlx_array,
    lm_head_w: mlx.mlx_array,
    lm_head_s: mlx.mlx_array,
    lm_head_b: mlx.mlx_array,
    layers: []LayerWeights,

    owns_lm_head: bool,
    owns_norms: bool,

    gelu_coeff: ?mlx.mlx_array,
    gelu_inner: ?mlx.mlx_array,
    half: mlx.mlx_array,
    one: mlx.mlx_array,
    three: ?mlx.mlx_array,
    neg_one: ?mlx.mlx_array,

    // MoE-specific (null/empty for standard models)
    moe_layers: ?[]MoeLayerWeights,
    ssm_entries: ?[]SSMCacheEntry,
    moe_seq_offset: usize,

    // Prompt cache for prefix reuse across requests
    prompt_cache: ?PrefillCache,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !Transformer {
        const s = mlx.mlx_default_gpu_stream_new();
        const prefix = config.weight_prefix;

        var name_buf: [256]u8 = undefined;
        const emb_w = getWeightFmt(weights, &name_buf, "{s}.embed_tokens.weight", prefix);
        const emb_s_arr = getWeightFmt(weights, &name_buf, "{s}.embed_tokens.scales", prefix);
        const emb_b_arr = getWeightFmt(weights, &name_buf, "{s}.embed_tokens.biases", prefix);

        const emb_scale: ?mlx.mlx_array = if (config.scale_embeddings)
            bf16Scalar(@sqrt(@as(f32, @floatFromInt(config.hidden_size))), s)
        else
            null;

        const final_norm_raw = getWeightFmt(weights, &name_buf, "{s}.norm.weight", prefix);
        const final_norm = if (config.norm_has_offset) try addOne(final_norm_raw, s) else final_norm_raw;
        if (config.norm_has_offset) try mlx.check(mlx.mlx_array_eval(final_norm));

        var lm_head_w: mlx.mlx_array = undefined;
        var lm_head_s: mlx.mlx_array = undefined;
        var lm_head_b: mlx.mlx_array = undefined;
        var owns_lm_head = false;

        {
            // lm_head prefix: "language_model.model" -> "language_model", "model" -> try root, else -> prefix
            const lm_prefix = if (std.mem.eql(u8, prefix, "language_model.model")) "language_model" else prefix;
            const maybe_lm_w = getWeightFmtOpt(weights, &name_buf, "{s}.lm_head.weight", lm_prefix);
            if (maybe_lm_w) |w| {
                lm_head_w = w;
                lm_head_s = getWeightFmt(weights, &name_buf, "{s}.lm_head.scales", lm_prefix);
                lm_head_b = getWeightFmt(weights, &name_buf, "{s}.lm_head.biases", lm_prefix);
                owns_lm_head = !config.tie_word_embeddings;
            } else if (weights.get("lm_head.weight")) |w| {
                lm_head_w = w;
                lm_head_s = weights.get("lm_head.scales") orelse emb_s_arr;
                lm_head_b = weights.get("lm_head.biases") orelse emb_b_arr;
                owns_lm_head = !config.tie_word_embeddings;
            } else if (config.tie_word_embeddings) {
                lm_head_w = emb_w;
                lm_head_s = emb_s_arr;
                lm_head_b = emb_b_arr;
            } else {
                std.debug.print("MISSING WEIGHT: lm_head.weight\n", .{});
                unreachable;
            }
        }

        // Cache for KV (standard models use all entries, MoE only uses full-attn layers)
        const cache = try KVCache.init(allocator, config.num_hidden_layers);

        const need_gelu = config.hidden_act == .gelu_approx;
        const need_silu = config.hidden_act == .silu;

        // Load architecture-specific layer weights
        var layers: []LayerWeights = &.{};
        var moe_layers: ?[]MoeLayerWeights = null;
        var ssm_entries: ?[]SSMCacheEntry = null;

        if (config.isMoe()) {
            const ml = try initMoeLayers(allocator, config, weights, &name_buf, s);
            moe_layers = ml.moe_layers;
            ssm_entries = ml.ssm_entries;
        } else {
            layers = try initStandardLayers(allocator, config, weights, &name_buf, s);
        }

        // Batch eval all weights
        {
            var eval_timer = std.time.Timer.start() catch unreachable;
            const all_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(all_vec);

            _ = mlx.mlx_vector_array_append_value(all_vec, emb_w);
            _ = mlx.mlx_vector_array_append_value(all_vec, emb_s_arr);
            _ = mlx.mlx_vector_array_append_value(all_vec, emb_b_arr);
            _ = mlx.mlx_vector_array_append_value(all_vec, lm_head_w);
            _ = mlx.mlx_vector_array_append_value(all_vec, lm_head_s);
            _ = mlx.mlx_vector_array_append_value(all_vec, lm_head_b);

            if (moe_layers) |ml| {
                for (ml) |lw| {
                    _ = mlx.mlx_vector_array_append_value(all_vec, lw.input_norm);
                    _ = mlx.mlx_vector_array_append_value(all_vec, lw.post_attn_norm);
                    appendMoeWeights(all_vec, &lw.mlp);
                    switch (lw.attn) {
                        .full => |fa| appendFullAttnWeights(all_vec, &fa),
                        .linear => |la| appendLinearAttnWeights(all_vec, &la),
                    }
                }
            } else {
                for (layers) |lw| {
                    _ = mlx.mlx_vector_array_append_value(all_vec, lw.input_norm);
                    _ = mlx.mlx_vector_array_append_value(all_vec, lw.post_attn_norm);
                    if (lw.pre_ff_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                    if (lw.post_ff_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                    if (lw.q_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
                    if (lw.k_norm) |n| _ = mlx.mlx_vector_array_append_value(all_vec, n);
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
            }

            try mlx.check(mlx.mlx_eval(all_vec));
            const eval_ms = eval_timer.read() / std.time.ns_per_ms;
            std.debug.print("Batch eval all weights: {d}ms\n", .{eval_ms});
        }

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
            .moe_layers = moe_layers,
            .ssm_entries = ssm_entries,
            .moe_seq_offset = 0,
            .prompt_cache = null,
        };
    }

    /// Reset all caches for a new request (KV cache + SSM state for MoE).
    pub fn resetCache(self: *Transformer) !void {
        self.cache.deinit();
        self.cache = try KVCache.init(self.allocator, self.config.num_hidden_layers);
        if (self.ssm_entries) |entries| {
            for (entries) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                e.conv_state = mlx.mlx_array_new();
                e.ssm_state = mlx.mlx_array_new();
                e.initialized = false;
            }
        }
        self.moe_seq_offset = 0;
    }

    /// Try to restore state from prompt cache if the cached tokens are an exact
    /// prefix of new_ids. Returns the number of matched (restored) tokens, or 0
    /// if the cache missed and a full reset was performed.
    pub fn tryRestoreCache(self: *Transformer, new_ids: []const u32) !usize {
        const pc = self.prompt_cache orelse {
            try self.resetCache();
            return 0;
        };

        const match_limit = @min(pc.tokens.len, new_ids.len);
        var matched: usize = 0;
        while (matched < match_limit) : (matched += 1) {
            if (pc.tokens[matched] != new_ids[matched]) break;
        }

        if (matched < pc.tokens.len or matched >= new_ids.len) {
            try self.resetCache();
            return 0;
        }

        // Full prefix match with tokens remaining — restore cached state.
        self.cache.deinit();
        self.cache = try KVCache.init(self.allocator, self.config.num_hidden_layers);
        for (pc.kv_entries, 0..) |src, i| {
            if (src.initialized) {
                try mlx.check(mlx.mlx_array_set(&self.cache.entries[i].keys, src.keys));
                try mlx.check(mlx.mlx_array_set(&self.cache.entries[i].values, src.values));
                self.cache.entries[i].initialized = true;
            }
        }

        if (pc.ssm_entries) |ssm_src| {
            if (self.ssm_entries) |ssm_dst| {
                for (ssm_src, ssm_dst) |src, *dst| {
                    _ = mlx.mlx_array_free(dst.conv_state);
                    _ = mlx.mlx_array_free(dst.ssm_state);
                    dst.conv_state = mlx.mlx_array_new();
                    dst.ssm_state = mlx.mlx_array_new();
                    if (src.initialized) {
                        try mlx.check(mlx.mlx_array_set(&dst.conv_state, src.conv_state));
                        try mlx.check(mlx.mlx_array_set(&dst.ssm_state, src.ssm_state));
                        dst.initialized = true;
                    } else {
                        dst.initialized = false;
                    }
                }
            }
        }

        self.moe_seq_offset = pc.moe_seq_offset;
        return matched;
    }

    /// Snapshot the current KV cache + SSM state so the next request can reuse
    /// them if its prompt starts with the same token prefix.
    pub fn savePromptCache(self: *Transformer, prompt_ids: []const u32) void {
        if (self.prompt_cache) |*pc| pc.deinit();
        self.prompt_cache = null;

        const tokens = self.allocator.dupe(u32, prompt_ids) catch return;
        const num_layers = self.cache.entries.len;
        const kv = self.allocator.alloc(KVCacheEntry, num_layers) catch {
            self.allocator.free(tokens);
            return;
        };
        for (self.cache.entries, kv) |src, *dst| {
            dst.keys = mlx.mlx_array_new();
            dst.values = mlx.mlx_array_new();
            if (src.initialized) {
                _ = mlx.mlx_array_set(&dst.keys, src.keys);
                _ = mlx.mlx_array_set(&dst.values, src.values);
            }
            dst.initialized = src.initialized;
        }

        var ssm: ?[]SSMCacheEntry = null;
        if (self.ssm_entries) |entries| {
            const ssm_copy = self.allocator.alloc(SSMCacheEntry, entries.len) catch return;
            for (entries, ssm_copy) |src, *dst| {
                dst.conv_state = mlx.mlx_array_new();
                dst.ssm_state = mlx.mlx_array_new();
                if (src.initialized) {
                    _ = mlx.mlx_array_set(&dst.conv_state, src.conv_state);
                    _ = mlx.mlx_array_set(&dst.ssm_state, src.ssm_state);
                }
                dst.initialized = src.initialized;
            }
            ssm = ssm_copy;
        }

        self.prompt_cache = .{
            .tokens = tokens,
            .kv_entries = kv,
            .ssm_entries = ssm,
            .moe_seq_offset = self.moe_seq_offset,
            .allocator = self.allocator,
        };
    }

    pub fn deinit(self: *Transformer) void {
        if (self.prompt_cache) |*pc| pc.deinit();
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
        if (self.ssm_entries) |entries| {
            for (entries) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
            }
            self.allocator.free(entries);
        }
        if (self.moe_layers) |ml| self.allocator.free(ml);
        _ = mlx.mlx_stream_free(self.s);
    }

    // ── Core ops ──

    inline fn qmatmul(self: *const Transformer, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array) !mlx.mlx_array {
        return qmatmulBits(x, w, sc, bi, self.config.quant_bits, self.config.quant_group_size, self.s);
    }

    inline fn rmsNorm(self: *const Transformer, x: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_rms_norm(&result, x, w, self.config.rms_norm_eps, self.s));
        return result;
    }

    fn embedding(self: *const Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
        const id_shape = mlx.getShape(token_ids);
        const batch = id_shape[0];
        const seq_len = id_shape[1];

        const flat_shape = [_]c_int{batch * seq_len};
        var flat_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat_ids);
        try mlx.check(mlx.mlx_reshape(&flat_ids, token_ids, &flat_shape, 1, self.s));

        var taken_w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken_w);
        try mlx.check(mlx.mlx_take_axis(&taken_w, self.emb_w, flat_ids, 0, self.s));
        var taken_s = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken_s);
        try mlx.check(mlx.mlx_take_axis(&taken_s, self.emb_s, flat_ids, 0, self.s));
        var taken_b = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken_b);
        try mlx.check(mlx.mlx_take_axis(&taken_b, self.emb_b, flat_ids, 0, self.s));

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

        const out_shape = [_]c_int{ batch, seq_len, @intCast(self.config.hidden_size) };
        var reshaped = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(reshaped);
        try mlx.check(mlx.mlx_reshape(&reshaped, emb, &out_shape, 3, self.s));

        if (self.emb_scale) |scale| {
            var scaled = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&scaled, reshaped, scale, self.s));
            return scaled;
        }
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&result, reshaped));
        return result;
    }

    // ── Activation functions ──

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

    fn silu(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        try mlx.check(mlx.mlx_sigmoid(&sig, x, self.s));
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&result, x, sig, self.s));
        return result;
    }

    inline fn mlpActivation(self: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        return switch (self.config.hidden_act) {
            .gelu_approx => self.gelu(x),
            .silu => self.silu(x),
        };
    }

    /// SwiGLU: silu(gate) * x
    fn swiglu(self: *const Transformer, gate: mlx.mlx_array, x: mlx.mlx_array) !mlx.mlx_array {
        const activated = try self.silu(gate);
        defer _ = mlx.mlx_array_free(activated);
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&result, activated, x, self.s));
        return result;
    }

    // ── Forward dispatch ──

    const EVAL_EVERY_N_LAYERS: u32 = 48;
    const MOE_EVAL_EVERY_N_LAYERS: u32 = 4;
    const RECURRENCE_EVAL_INTERVAL: usize = 32;

    pub fn forward(self: *Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
        if (self.config.isMoe()) return self.forwardMoe(token_ids);
        return self.forwardStandard(token_ids);
    }

    // ── Standard forward pass (Gemma / Llama / Qwen3) ──

    fn forwardStandard(self: *Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
        const offset = self.cache.seqLen(0);
        const h_count = self.config.num_attention_heads;
        const kv_h = self.config.num_key_value_heads;
        const hd = self.config.head_dim;
        const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.config.query_pre_attn_scalar)));

        var h = try self.embedding(token_ids);

        const x_shape = mlx.getShape(h);
        const batch: c_int = x_shape[0];
        const seq_len: c_int = x_shape[1];
        const is_prefill = seq_len > 1;

        const q_shape = [_]c_int{ batch, seq_len, @intCast(h_count), @intCast(hd) };
        const kv_shape = [_]c_int{ batch, seq_len, @intCast(kv_h), @intCast(hd) };
        const perm = [_]c_int{ 0, 2, 1, 3 };
        const perm_back = [_]c_int{ 0, 2, 1, 3 };
        const out_shape = [_]c_int{ batch, seq_len, @intCast(h_count * hd) };

        const none_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(none_mask);

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

        for (0..self.config.num_hidden_layers) |layer_idx| {
            const li: u32 = @intCast(layer_idx);
            const lw = &self.layers[layer_idx];
            const is_global = self.config.isGlobalLayer(li);

            const normed = try self.rmsNorm(h, lw.input_norm);
            defer _ = mlx.mlx_array_free(normed);

            const q = try self.qmatmul(normed, lw.q_w, lw.q_s, lw.q_b);
            defer _ = mlx.mlx_array_free(q);
            const k = try self.qmatmul(normed, lw.k_w, lw.k_s, lw.k_b);
            defer _ = mlx.mlx_array_free(k);
            const v = try self.qmatmul(normed, lw.v_w, lw.v_s, lw.v_b);
            defer _ = mlx.mlx_array_free(v);

            var q_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_r);
            var k_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_r);
            var v_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_r);
            try mlx.check(mlx.mlx_reshape(&q_r, q, &q_shape, 4, self.s));
            try mlx.check(mlx.mlx_reshape(&k_r, k, &kv_shape, 4, self.s));
            try mlx.check(mlx.mlx_reshape(&v_r, v, &kv_shape, 4, self.s));

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

            var q_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_t);
            var k_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_t);
            var v_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_t);
            try mlx.check(mlx.mlx_transpose_axes(&q_t, q_for_rope, &perm, 4, self.s));
            try mlx.check(mlx.mlx_transpose_axes(&k_t, k_for_rope, &perm, 4, self.s));
            try mlx.check(mlx.mlx_transpose_axes(&v_t, v_r, &perm, 4, self.s));

            const rope_base: f32 = if (is_global) self.config.rope_theta else self.config.rope_local_base_freq;
            const rope_scale: f32 = if (is_global) (1.0 / self.config.rope_scaling_factor) else 1.0;

            var q_rope = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_rope);
            var k_rope = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_rope);

            try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, @intCast(hd), false, mlx.mlx_optional_float.some(rope_base), rope_scale, @intCast(offset), .{ .ctx = null }, self.s));
            try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, @intCast(hd), false, mlx.mlx_optional_float.some(rope_base), rope_scale, @intCast(offset), .{ .ctx = null }, self.s));

            const max_kv: u32 = if (is_global) 0 else if (self.config.has_sliding_window) self.config.sliding_window else 0;
            const kv = try self.cache.update(li, k_rope, v_t, self.s, max_kv);
            const full_k = kv[0];
            const full_v = kv[1];

            var attn_out = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_out);

            if (!self.config.has_sliding_window) {
                if (is_prefill) {
                    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, self.s));
                } else {
                    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, self.s));
                }
            } else {
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

            var attn_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_t);
            try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm_back, 4, self.s));
            var attn_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_flat);
            try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &out_shape, 3, self.s));

            const o_out = try self.qmatmul(attn_flat, lw.o_w, lw.o_s, lw.o_b);
            defer _ = mlx.mlx_array_free(o_out);

            if (self.config.has_pre_ff_norm) {
                const attn_normed = try self.rmsNorm(o_out, lw.post_attn_norm);
                defer _ = mlx.mlx_array_free(attn_normed);
                var h_new = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_new, h, attn_normed, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_new;

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
                var h_new = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_new, h, o_out, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_new;

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

                var h_next = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_next, h, down, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_next;
            }

            if (is_prefill and (layer_idx + 1) % EVAL_EVERY_N_LAYERS == 0) {
                try mlx.check(mlx.mlx_array_eval(h));
            }
        }

        const final_normed = try self.rmsNorm(h, self.final_norm);
        _ = mlx.mlx_array_free(h);
        const logits = try self.qmatmul(final_normed, self.lm_head_w, self.lm_head_s, self.lm_head_b);
        _ = mlx.mlx_array_free(final_normed);
        return logits;
    }

    // ── MoE forward pass (Qwen3.5) ──

    fn forwardMoe(self: *Transformer, token_ids: mlx.mlx_array) !mlx.mlx_array {
        const ml = self.moe_layers.?;
        const offset = self.moe_seq_offset;
        const cfg = &self.config;

        var h = try self.embedding(token_ids);

        const x_shape = mlx.getShape(h);
        const batch: c_int = x_shape[0];
        const seq_len: c_int = x_shape[1];
        const is_prefill = seq_len > 1;

        for (0..cfg.num_hidden_layers) |layer_idx| {
            const li: u32 = @intCast(layer_idx);
            const lw = &ml[layer_idx];

            const normed = try self.rmsNorm(h, lw.input_norm);
            defer _ = mlx.mlx_array_free(normed);

            // Attention: linear (GatedDeltaNet) or full
            const attn_out = switch (lw.attn) {
                .linear => |la| try self.gatedDeltaNet(normed, &la, &self.ssm_entries.?[layer_idx], batch, seq_len),
                .full => |fa| try self.gatedFullAttn(normed, &fa, li, @intCast(offset), batch, seq_len, is_prefill),
            };
            defer _ = mlx.mlx_array_free(attn_out);

            // h = h + attn_out
            var h_new = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_new, h, attn_out, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_new;

            // MoE MLP
            const ff_normed = try self.rmsNorm(h, lw.post_attn_norm);
            defer _ = mlx.mlx_array_free(ff_normed);
            const mlp_out = try self.moeMLP(ff_normed, &lw.mlp);
            defer _ = mlx.mlx_array_free(mlp_out);

            // h = h + mlp_out
            var h_next = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_next, h, mlp_out, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_next;

            if (is_prefill and (layer_idx + 1) % MOE_EVAL_EVERY_N_LAYERS == 0) {
                try mlx.check(mlx.mlx_array_eval(h));
            }
        }

        self.moe_seq_offset += @intCast(seq_len);

        const final_normed = try self.rmsNorm(h, self.final_norm);
        _ = mlx.mlx_array_free(h);
        const logits = try self.qmatmul(final_normed, self.lm_head_w, self.lm_head_s, self.lm_head_b);
        _ = mlx.mlx_array_free(final_normed);
        return logits;
    }

    // ── Full Attention for MoE models (with optional output gate) ──

    fn gatedFullAttn(
        self: *Transformer,
        x: mlx.mlx_array,
        fa: *const FullAttnWeights,
        layer: u32,
        offset: c_int,
        batch: c_int,
        seq_len: c_int,
        is_prefill: bool,
    ) !mlx.mlx_array {
        const cfg = &self.config;
        const h_count: c_int = @intCast(cfg.num_attention_heads);
        const kv_h: c_int = @intCast(cfg.num_key_value_heads);
        const hd: c_int = @intCast(cfg.head_dim);
        const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar)));
        const rope_dims: c_int = @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor);
        const flat_shape = [_]c_int{ batch, seq_len, h_count * hd };

        // Q projection
        const q_proj = try self.qmatmul(x, fa.q_w, fa.q_s, fa.q_b);
        defer _ = mlx.mlx_array_free(q_proj);

        // With output gate: q_proj outputs [B, S, 2*H*D], split into queries + gate
        // Without: q_proj outputs [B, S, H*D], used directly as queries
        var queries: mlx.mlx_array = undefined;
        defer _ = mlx.mlx_array_free(queries);
        var gate: mlx.mlx_array = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate);

        if (cfg.attn_output_gate) {
            const q_gate_shape = [_]c_int{ batch, seq_len, h_count, hd * 2 };
            var q_gate_r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_gate_r);
            try mlx.check(mlx.mlx_reshape(&q_gate_r, q_proj, &q_gate_shape, 4, self.s));

            const strides4 = [_]c_int{ 1, 1, 1, 1 };
            const q_start = [_]c_int{ 0, 0, 0, 0 };
            const q_stop = [_]c_int{ batch, seq_len, h_count, hd };
            queries = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&queries, q_gate_r, &q_start, 4, &q_stop, 4, &strides4, 4, self.s));

            const g_start = [_]c_int{ 0, 0, 0, hd };
            const g_stop = [_]c_int{ batch, seq_len, h_count, hd * 2 };
            var gate_4d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate_4d);
            try mlx.check(mlx.mlx_slice(&gate_4d, q_gate_r, &g_start, 4, &g_stop, 4, &strides4, 4, self.s));

            try mlx.check(mlx.mlx_reshape(&gate, gate_4d, &flat_shape, 3, self.s));
        } else {
            const q_shape = [_]c_int{ batch, seq_len, h_count, hd };
            queries = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&queries, q_proj, &q_shape, 4, self.s));
        }

        // K, V projections
        const k_proj = try self.qmatmul(x, fa.k_w, fa.k_s, fa.k_b);
        defer _ = mlx.mlx_array_free(k_proj);
        const v_proj = try self.qmatmul(x, fa.v_w, fa.v_s, fa.v_b);
        defer _ = mlx.mlx_array_free(v_proj);

        const kv_shape = [_]c_int{ batch, seq_len, kv_h, hd };
        var k_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_r);
        var v_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_r);
        try mlx.check(mlx.mlx_reshape(&k_r, k_proj, &kv_shape, 4, self.s));
        try mlx.check(mlx.mlx_reshape(&v_r, v_proj, &kv_shape, 4, self.s));

        // Q/K norms
        const q_normed = try self.rmsNorm(queries, fa.q_norm);
        defer _ = mlx.mlx_array_free(q_normed);
        const k_normed = try self.rmsNorm(k_r, fa.k_norm);
        defer _ = mlx.mlx_array_free(k_normed);

        // Transpose to [B, H, S, D]
        const perm = [_]c_int{ 0, 2, 1, 3 };
        var q_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_t);
        var k_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_t);
        var v_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_t);
        try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed, &perm, 4, self.s));
        try mlx.check(mlx.mlx_transpose_axes(&k_t, k_normed, &perm, 4, self.s));
        try mlx.check(mlx.mlx_transpose_axes(&v_t, v_r, &perm, 4, self.s));

        // Partial RoPE
        var q_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_rope);
        var k_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_rope);
        try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, rope_dims, false, mlx.mlx_optional_float.some(self.config.rope_theta), 1.0, offset, .{ .ctx = null }, self.s));
        try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, rope_dims, false, mlx.mlx_optional_float.some(self.config.rope_theta), 1.0, offset, .{ .ctx = null }, self.s));

        // KV cache update
        const kv = try self.cache.update(layer, k_rope, v_t, self.s, 0);
        const full_k = kv[0];
        const full_v = kv[1];

        // Attention
        var attn_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_out);
        const none_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(none_mask);

        if (is_prefill) {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, self.s));
        } else {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, self.s));
        }

        // Transpose back [B,H,S,D] -> [B,S,H*D]
        const perm_back = [_]c_int{ 0, 2, 1, 3 };
        var attn_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_t);
        try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm_back, 4, self.s));
        var attn_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_flat);
        try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, self.s));

        // Optional output gating
        if (cfg.attn_output_gate) {
            var gate_sig = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate_sig);
            try mlx.check(mlx.mlx_sigmoid(&gate_sig, gate, self.s));
            var gated = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gated);
            try mlx.check(mlx.mlx_multiply(&gated, attn_flat, gate_sig, self.s));
            return self.qmatmul(gated, fa.o_w, fa.o_s, fa.o_b);
        }

        return self.qmatmul(attn_flat, fa.o_w, fa.o_s, fa.o_b);
    }

    // ── GatedDeltaNet (linear attention layers) ──

    fn gatedDeltaNet(
        self: *Transformer,
        x: mlx.mlx_array,
        la: *const LinearAttnWeights,
        ssm: *SSMCacheEntry,
        batch: c_int,
        seq_len: c_int,
    ) !mlx.mlx_array {
        const cfg = &self.config;
        const num_k_heads: c_int = @intCast(cfg.linear_num_key_heads);
        const num_v_heads: c_int = @intCast(cfg.linear_num_value_heads);
        const dk: c_int = @intCast(cfg.linear_key_head_dim);
        const dv: c_int = @intCast(cfg.linear_value_head_dim);
        const key_dim: c_int = dk * num_k_heads;
        const value_dim: c_int = dv * num_v_heads;
        const conv_dim: c_int = key_dim * 2 + value_dim;
        const kernel: c_int = @intCast(cfg.linear_conv_kernel_dim);

        // Projections: combined (qkvz+ba) or separate (qkv+z+a+b)
        var qkv: mlx.mlx_array = undefined;
        var z_proj: mlx.mlx_array = undefined;
        var a_proj: mlx.mlx_array = undefined;
        var b_proj: mlx.mlx_array = undefined;
        defer _ = mlx.mlx_array_free(qkv);
        defer _ = mlx.mlx_array_free(z_proj);
        defer _ = mlx.mlx_array_free(a_proj);
        defer _ = mlx.mlx_array_free(b_proj);

        if (la.combined_proj) {
            // Combined QKVZ: output is interleaved by key-head groups.
            // Reshape to [B, S, nk, per_head], split per-head into q/k/v/z, then flatten back.
            const vph = @divExact(num_v_heads, num_k_heads); // value heads per key head group
            const qkvz_raw = try self.qmatmul(x, la.qkv_w, la.qkv_s, la.qkv_b);
            defer _ = mlx.mlx_array_free(qkvz_raw);
            const per_head = dk + dk + vph * dv + vph * dv;
            const gh_shape = [_]c_int{ batch, seq_len, num_k_heads, per_head };
            var qkvz_g = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(qkvz_g);
            try mlx.check(mlx.mlx_reshape(&qkvz_g, qkvz_raw, &gh_shape, 4, self.s));

            const strides4 = [_]c_int{ 1, 1, 1, 1 };
            // q: [B,S,nk,dk]
            var q_g = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_g);
            try mlx.check(mlx.mlx_slice(&q_g, qkvz_g, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ batch, seq_len, num_k_heads, dk }, 4, &strides4, 4, self.s));
            // k: [B,S,nk,dk]
            var k_g = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_g);
            try mlx.check(mlx.mlx_slice(&k_g, qkvz_g, &[_]c_int{ 0, 0, 0, dk }, 4, &[_]c_int{ batch, seq_len, num_k_heads, dk * 2 }, 4, &strides4, 4, self.s));
            // v: [B,S,nk,vph*dv]
            const v_off = dk * 2;
            const v_end = v_off + vph * dv;
            var v_g = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_g);
            try mlx.check(mlx.mlx_slice(&v_g, qkvz_g, &[_]c_int{ 0, 0, 0, v_off }, 4, &[_]c_int{ batch, seq_len, num_k_heads, v_end }, 4, &strides4, 4, self.s));
            // z: [B,S,nk,vph*dv]
            var z_g = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(z_g);
            try mlx.check(mlx.mlx_slice(&z_g, qkvz_g, &[_]c_int{ 0, 0, 0, v_end }, 4, &[_]c_int{ batch, seq_len, num_k_heads, per_head }, 4, &strides4, 4, self.s));

            // Flatten: q/k -> [B,S,key_dim], v/z -> [B,S,value_dim]
            const flat3_qk = [_]c_int{ batch, seq_len, key_dim };
            const flat3_vz = [_]c_int{ batch, seq_len, value_dim };
            var q_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_flat);
            var k_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_flat);
            var v_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_flat);
            try mlx.check(mlx.mlx_reshape(&q_flat, q_g, &flat3_qk, 3, self.s));
            try mlx.check(mlx.mlx_reshape(&k_flat, k_g, &flat3_qk, 3, self.s));
            try mlx.check(mlx.mlx_reshape(&v_flat, v_g, &flat3_vz, 3, self.s));
            z_proj = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&z_proj, z_g, &flat3_vz, 3, self.s));

            // Concatenate [q, k, v] -> qkv [B,S,conv_dim]
            const qkv_arr = [_]mlx.mlx_array{ q_flat, k_flat, v_flat };
            const qkv_vec = mlx.mlx_vector_array_new_data(&qkv_arr, 3);
            defer _ = mlx.mlx_vector_array_free(qkv_vec);
            qkv = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_concatenate_axis(&qkv, qkv_vec, 2, self.s));

            // Combined BA: interleaved by key-head groups
            const ba_raw = try self.qmatmul(x, la.b_w, la.b_s, la.b_b);
            defer _ = mlx.mlx_array_free(ba_raw);
            const ba_per_head = vph * 2;
            const ba_shape = [_]c_int{ batch, seq_len, num_k_heads, ba_per_head };
            var ba_g = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ba_g);
            try mlx.check(mlx.mlx_reshape(&ba_g, ba_raw, &ba_shape, 4, self.s));
            var b_g = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(b_g);
            var a_g = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(a_g);
            try mlx.check(mlx.mlx_slice(&b_g, ba_g, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ batch, seq_len, num_k_heads, vph }, 4, &strides4, 4, self.s));
            try mlx.check(mlx.mlx_slice(&a_g, ba_g, &[_]c_int{ 0, 0, 0, vph }, 4, &[_]c_int{ batch, seq_len, num_k_heads, ba_per_head }, 4, &strides4, 4, self.s));
            const flat3_ba = [_]c_int{ batch, seq_len, num_v_heads };
            b_proj = mlx.mlx_array_new();
            a_proj = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&b_proj, b_g, &flat3_ba, 3, self.s));
            try mlx.check(mlx.mlx_reshape(&a_proj, a_g, &flat3_ba, 3, self.s));
        } else {
            qkv = try self.qmatmul(x, la.qkv_w, la.qkv_s, la.qkv_b);
            z_proj = try self.qmatmul(x, la.z_w, la.z_s, la.z_b);
            a_proj = try self.qmatmul(x, la.a_w, la.a_s, la.a_b);
            b_proj = try self.qmatmul(x, la.b_w, la.b_s, la.b_b);
        }

        // Conv1d: prepend conv_state, apply depthwise conv, then silu
        var conv_input: mlx.mlx_array = undefined;
        defer _ = mlx.mlx_array_free(conv_input);
        if (ssm.initialized) {
            const arr = [_]mlx.mlx_array{ ssm.conv_state, qkv };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            conv_input = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, self.s));
        } else {
            // First call: prepend zeros
            const zero_shape = [_]c_int{ batch, kernel - 1, conv_dim };
            var zero_state = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(zero_state);
            try mlx.check(mlx.mlx_zeros(&zero_state, &zero_shape, 3, .bfloat16, self.s));
            const arr = [_]mlx.mlx_array{ zero_state, qkv };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            conv_input = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, self.s));
        }

        // Update conv_state: keep last (kernel-1) positions
        {
            const ci_shape = mlx.getShape(conv_input);
            const ci_len = ci_shape[1];
            const keep_start = ci_len - (kernel - 1);
            const start = [_]c_int{ 0, keep_start, 0 };
            const stop = [_]c_int{ batch, ci_len, conv_dim };
            const strides = [_]c_int{ 1, 1, 1 };
            var new_conv_state = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&new_conv_state, conv_input, &start, 3, &stop, 3, &strides, 3, self.s));
            _ = mlx.mlx_array_free(ssm.conv_state);
            ssm.conv_state = new_conv_state;
        }

        // Depthwise conv1d + silu
        var conv_out_raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(conv_out_raw);
        try mlx.check(mlx.mlx_conv1d(&conv_out_raw, conv_input, la.conv1d_w, 1, 0, 1, conv_dim, self.s));
        const conv_out = try self.silu(conv_out_raw); // [B, S, conv_dim]
        defer _ = mlx.mlx_array_free(conv_out);

        // Split conv output into Q, K, V
        // Q: [B, S, key_dim] → [B, S, num_k_heads, dk]
        // K: [B, S, key_dim] → [B, S, num_k_heads, dk]
        // V: [B, S, value_dim] → [B, S, num_v_heads, dv]
        const strides3 = [_]c_int{ 1, 1, 1 };
        var q_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_flat);
        {
            const start = [_]c_int{ 0, 0, 0 };
            const stop = [_]c_int{ batch, seq_len, key_dim };
            try mlx.check(mlx.mlx_slice(&q_flat, conv_out, &start, 3, &stop, 3, &strides3, 3, self.s));
        }
        var k_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_flat);
        {
            const start = [_]c_int{ 0, 0, key_dim };
            const stop = [_]c_int{ batch, seq_len, key_dim * 2 };
            try mlx.check(mlx.mlx_slice(&k_flat, conv_out, &start, 3, &stop, 3, &strides3, 3, self.s));
        }
        var v_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_flat);
        {
            const start = [_]c_int{ 0, 0, key_dim * 2 };
            const stop = [_]c_int{ batch, seq_len, key_dim * 2 + value_dim };
            try mlx.check(mlx.mlx_slice(&v_flat, conv_out, &start, 3, &stop, 3, &strides3, 3, self.s));
        }

        // Reshape to head dims
        const q_shape = [_]c_int{ batch, seq_len, num_k_heads, dk };
        const k_shape = [_]c_int{ batch, seq_len, num_k_heads, dk };
        const v_shape = [_]c_int{ batch, seq_len, num_v_heads, dv };
        var q_heads = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_heads);
        var k_heads = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_heads);
        var v_heads = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_heads);
        try mlx.check(mlx.mlx_reshape(&q_heads, q_flat, &q_shape, 4, self.s));
        try mlx.check(mlx.mlx_reshape(&k_heads, k_flat, &k_shape, 4, self.s));
        try mlx.check(mlx.mlx_reshape(&v_heads, v_flat, &v_shape, 4, self.s));

        // Q/K normalization: q = (1/dk) * rms_norm(q, null), k = (1/sqrt(dk)) * rms_norm(k, null)
        const inv_scale = 1.0 / @as(f32, @floatFromInt(cfg.linear_key_head_dim));
        const inv_sqrt_scale = @sqrt(inv_scale);
        const inv_scale_sq = bf16Scalar(inv_scale, self.s);
        defer _ = mlx.mlx_array_free(inv_scale_sq);
        const inv_sqrt_sc = bf16Scalar(inv_sqrt_scale, self.s);
        defer _ = mlx.mlx_array_free(inv_sqrt_sc);

        var q_norm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_norm);
        try mlx.check(mlx.mlx_fast_rms_norm(&q_norm, q_heads, .{ .ctx = null }, 1e-6, self.s));
        var q_scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_scaled);
        try mlx.check(mlx.mlx_multiply(&q_scaled, q_norm, inv_scale_sq, self.s));

        var k_norm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_norm);
        try mlx.check(mlx.mlx_fast_rms_norm(&k_norm, k_heads, .{ .ctx = null }, 1e-6, self.s));
        var k_scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_scaled);
        try mlx.check(mlx.mlx_multiply(&k_scaled, k_norm, inv_sqrt_sc, self.s));

        // Compute gating: g = exp(-exp(A_log) * softplus(a + dt_bias))
        // Cast A_log to float32 for stability
        var A_log_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(A_log_f32);
        try mlx.check(mlx.mlx_astype(&A_log_f32, la.A_log, .float32, self.s));
        var exp_A = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(exp_A);
        try mlx.check(mlx.mlx_exp(&exp_A, A_log_f32, self.s));

        // softplus(a + dt_bias) = log(1 + exp(a + dt_bias))
        var a_plus_dt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(a_plus_dt);
        try mlx.check(mlx.mlx_add(&a_plus_dt, a_proj, la.dt_bias, self.s));
        var a_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(a_f32);
        try mlx.check(mlx.mlx_astype(&a_f32, a_plus_dt, .float32, self.s));
        var exp_a = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(exp_a);
        try mlx.check(mlx.mlx_exp(&exp_a, a_f32, self.s));
        var sp_inner = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sp_inner);
        try mlx.check(mlx.mlx_log1p(&sp_inner, exp_a, self.s));

        // -exp(A_log) * softplus(...)
        var neg_decay = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(neg_decay);
        try mlx.check(mlx.mlx_multiply(&neg_decay, exp_A, sp_inner, self.s));
        var neg_neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(neg_neg);
        try mlx.check(mlx.mlx_negative(&neg_neg, neg_decay, self.s));
        var g_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g_f32);
        try mlx.check(mlx.mlx_exp(&g_f32, neg_neg, self.s)); // [B, S, Hv]
        var g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g);
        try mlx.check(mlx.mlx_astype(&g, g_f32, .bfloat16, self.s));

        // beta = sigmoid(b)
        var beta = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(beta);
        try mlx.check(mlx.mlx_sigmoid(&beta, b_proj, self.s)); // [B, S, Hv]

        // Repeat Q, K heads from num_k_heads to num_v_heads for the recurrence
        const repeat_factor = @divExact(cfg.linear_num_value_heads, cfg.linear_num_key_heads);
        var q_rep = q_scaled;
        var k_rep = k_scaled;
        var q_rep_owned = false;
        var k_rep_owned = false;
        if (repeat_factor > 1) {
            var q_tmp = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_repeat_axis(&q_tmp, q_scaled, @intCast(repeat_factor), 2, self.s));
            q_rep = q_tmp;
            q_rep_owned = true;
            var k_tmp = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_repeat_axis(&k_tmp, k_scaled, @intCast(repeat_factor), 2, self.s));
            k_rep = k_tmp;
            k_rep_owned = true;
        }
        defer {
            if (q_rep_owned) _ = mlx.mlx_array_free(q_rep);
            if (k_rep_owned) _ = mlx.mlx_array_free(k_rep);
        }

        // Initialize SSM state if needed
        if (!ssm.initialized) {
            const state_shape = [_]c_int{ batch, num_v_heads, dv, dk };
            _ = mlx.mlx_array_free(ssm.ssm_state);
            ssm.ssm_state = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&ssm.ssm_state, &state_shape, 4, .bfloat16, self.s));
            ssm.initialized = true;
        }

        // Delta recurrence: loop over timesteps
        const T: usize = @intCast(seq_len);
        const out_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(out_vec);

        for (0..T) |t| {
            const ti: c_int = @intCast(t);
            // Extract timestep slices: [B, 1, ...] → squeeze to [B, ...]
            const qt = try sliceTimestep4(q_rep, batch, num_v_heads, dk, ti, self.s);
            defer _ = mlx.mlx_array_free(qt);
            const kt = try sliceTimestep4(k_rep, batch, num_v_heads, dk, ti, self.s);
            defer _ = mlx.mlx_array_free(kt);
            const vt = try sliceTimestep4(v_heads, batch, num_v_heads, dv, ti, self.s);
            defer _ = mlx.mlx_array_free(vt);
            const gt = try sliceTimestep3(g, batch, num_v_heads, ti, self.s);
            defer _ = mlx.mlx_array_free(gt);
            const bt = try sliceTimestep3(beta, batch, num_v_heads, ti, self.s);
            defer _ = mlx.mlx_array_free(bt);

            // state = state * g[..., None, None]
            var g_exp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(g_exp);
            {
                var g_e1 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(g_e1);
                try mlx.check(mlx.mlx_expand_dims(&g_e1, gt, 2, self.s));
                try mlx.check(mlx.mlx_expand_dims(&g_exp, g_e1, 3, self.s));
            }
            var decayed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(decayed);
            try mlx.check(mlx.mlx_multiply(&decayed, ssm.ssm_state, g_exp, self.s));

            // kv_mem = sum(state * k[..., None, :], axis=-1) → [B, Hv, Dv]
            // k[..., None, :] expands k [B,H,Dk] to [B,H,1,Dk]
            var k_exp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_exp);
            try mlx.check(mlx.mlx_expand_dims(&k_exp, kt, 2, self.s)); // [B,H,1,Dk]
            var state_k = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(state_k);
            try mlx.check(mlx.mlx_multiply(&state_k, decayed, k_exp, self.s)); // [B,H,Dv,Dk]
            var kv_mem = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(kv_mem);
            try mlx.check(mlx.mlx_sum_axis(&kv_mem, state_k, -1, false, self.s)); // [B,H,Dv]

            // delta = (v - kv_mem) * beta[..., None]
            var v_minus = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_minus);
            try mlx.check(mlx.mlx_subtract(&v_minus, vt, kv_mem, self.s));
            var b_exp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(b_exp);
            try mlx.check(mlx.mlx_expand_dims(&b_exp, bt, 2, self.s)); // [B,H,1]
            var delta = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(delta);
            try mlx.check(mlx.mlx_multiply(&delta, v_minus, b_exp, self.s)); // [B,H,Dv]

            // state = decayed + k[..., None, :] * delta[..., None]
            var d_exp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(d_exp);
            try mlx.check(mlx.mlx_expand_dims(&d_exp, delta, 3, self.s)); // [B,H,Dv,1]
            var kd = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(kd);
            try mlx.check(mlx.mlx_multiply(&kd, k_exp, d_exp, self.s)); // [B,H,Dv,Dk]
            var new_state = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&new_state, decayed, kd, self.s));
            _ = mlx.mlx_array_free(ssm.ssm_state);
            ssm.ssm_state = new_state;

            // y = sum(state * q[..., None, :], axis=-1) → [B, Hv, Dv]
            var q_exp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_exp);
            try mlx.check(mlx.mlx_expand_dims(&q_exp, qt, 2, self.s)); // [B,H,1,Dk]
            var state_q = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(state_q);
            try mlx.check(mlx.mlx_multiply(&state_q, ssm.ssm_state, q_exp, self.s));
            var y_t = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_sum_axis(&y_t, state_q, -1, false, self.s)); // [B,Hv,Dv]
            _ = mlx.mlx_vector_array_append_value(out_vec, y_t);
            _ = mlx.mlx_array_free(y_t);

            if ((t + 1) % RECURRENCE_EVAL_INTERVAL == 0) {
                try mlx.check(mlx.mlx_array_eval(ssm.ssm_state));
            }
        }

        // Stack outputs: [B, Hv, Dv] * T → [T, B, Hv, Dv] → transpose to [B, T, Hv, Dv]
        var stacked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stacked);
        try mlx.check(mlx.mlx_stack_axis(&stacked, out_vec, 0, self.s));

        // Transpose [T, B, Hv, Dv] → [B, T, Hv, Dv]
        const perm_tbhd = [_]c_int{ 1, 0, 2, 3 };
        var y_bthd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y_bthd);
        try mlx.check(mlx.mlx_transpose_axes(&y_bthd, stacked, &perm_tbhd, 4, self.s));

        // Reshape z to [B, S, Hv, Dv]
        const z_shape = [_]c_int{ batch, seq_len, num_v_heads, dv };
        var z_heads = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(z_heads);
        try mlx.check(mlx.mlx_reshape(&z_heads, z_proj, &z_shape, 4, self.s));

        // RMSNormGated: swiglu(z, rms_norm(y, norm_weight))
        const y_normed = try self.rmsNorm(y_bthd, la.norm_w);
        defer _ = mlx.mlx_array_free(y_normed);
        const out_gated = try self.swiglu(z_heads, y_normed);
        defer _ = mlx.mlx_array_free(out_gated);

        // Flatten [B, S, Hv, Dv] → [B, S, value_dim]
        const out_flat_shape = [_]c_int{ batch, seq_len, value_dim };
        var out_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out_flat);
        try mlx.check(mlx.mlx_reshape(&out_flat, out_gated, &out_flat_shape, 3, self.s));

        return self.qmatmul(out_flat, la.out_w, la.out_s, la.out_b);
    }

    // ── Sparse MoE MLP ──

    fn moeMLP(self: *Transformer, x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
        const cfg = &self.config;
        const k: c_int = @intCast(cfg.num_experts_per_tok);
        const bits = cfg.quant_bits;
        const gs = cfg.quant_group_size;

        // Router: softmax(gate(x)) → [B, S, num_experts]
        // Detect router bits from weight/scales shapes
        const router_bits = detectQuantBits(mw.router_w, mw.router_s, gs);
        const router_logits = try qmatmulBits(x, mw.router_w, mw.router_s, mw.router_b, router_bits, gs, self.s);
        defer _ = mlx.mlx_array_free(router_logits);
        var scores = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scores);
        try mlx.check(mlx.mlx_softmax_axis(&scores, router_logits, -1, true, self.s));

        // Top-K: argpartition then slice last K
        var partitioned = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(partitioned);
        try mlx.check(mlx.mlx_argpartition_axis(&partitioned, scores, -k, -1, self.s));

        // Slice [..., -K:] from partitioned
        const p_shape = mlx.getShape(partitioned);
        const p_last = p_shape[p_shape.len - 1];
        var inds: mlx.mlx_array = undefined;
        defer _ = mlx.mlx_array_free(inds);
        {
            var start_arr: [4]c_int = undefined;
            var stop_arr: [4]c_int = undefined;
            var strides_arr: [4]c_int = undefined;
            for (0..p_shape.len) |d| {
                start_arr[d] = if (d == p_shape.len - 1) p_last - k else 0;
                stop_arr[d] = p_shape[d];
                strides_arr[d] = 1;
            }
            inds = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&inds, partitioned, &start_arr, p_shape.len, &stop_arr, p_shape.len, &strides_arr, p_shape.len, self.s));
        }

        // Gather top-K scores and normalize
        var top_scores = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(top_scores);
        try mlx.check(mlx.mlx_take_along_axis(&top_scores, scores, inds, -1, self.s));
        var score_sum_raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(score_sum_raw);
        try mlx.check(mlx.mlx_sum_axis(&score_sum_raw, top_scores, -1, false, self.s));
        var score_sum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(score_sum);
        try mlx.check(mlx.mlx_expand_dims(&score_sum, score_sum_raw, -1, self.s));
        var norm_scores = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(norm_scores);
        try mlx.check(mlx.mlx_divide(&norm_scores, top_scores, score_sum, self.s));

        // Expert computation using gather_qmm
        // x: [B, S, D] → add K and M dims → [B, S, 1, 1, D]
        // batch dims [B, S, 1] broadcast with gathered w batch dims [B, S, K]
        var x_e1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_e1);
        try mlx.check(mlx.mlx_expand_dims(&x_e1, x, -1, self.s)); // [B, S, D, 1]
        var x_e2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_e2);
        try mlx.check(mlx.mlx_expand_dims(&x_e2, x_e1, -1, self.s)); // [B, S, D, 1, 1]

        // Transpose to [B, S, 1, 1, D]
        const x_shape = mlx.getShape(x);
        const D = x_shape[x_shape.len - 1];
        const exp_shape = [_]c_int{ x_shape[0], x_shape[1], 1, 1, D };
        var x_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_exp);
        try mlx.check(mlx.mlx_reshape(&x_exp, x, &exp_shape, 5, self.s));

        const no_idx = mlx.mlx_array{ .ctx = null };

        // gate_out: [B, S, K, 1, intermediate] → squeeze M → [B, S, K, intermediate]
        var gate_out_5d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_out_5d);
        try mlx.check(mlx.mlx_gather_qmm(&gate_out_5d, x_exp, mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, no_idx, inds, true, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", false, self.s));
        var gate_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_out);
        try mlx.check(mlx.mlx_squeeze(&gate_out, gate_out_5d, self.s));

        // up_out: [B, S, K, intermediate]
        var up_out_5d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up_out_5d);
        try mlx.check(mlx.mlx_gather_qmm(&up_out_5d, x_exp, mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, no_idx, inds, true, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", false, self.s));
        var up_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up_out);
        try mlx.check(mlx.mlx_squeeze(&up_out, up_out_5d, self.s));

        // SwiGLU: silu(gate) * up → [B, S, K, intermediate]
        const expert_act = try self.swiglu(gate_out, up_out);
        defer _ = mlx.mlx_array_free(expert_act);

        // down_out: expert_act [B,S,K,intermediate] → expand M → [B,S,K,1,intermediate]
        var act_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act_exp);
        try mlx.check(mlx.mlx_expand_dims(&act_exp, expert_act, -2, self.s));

        var down_out_5d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(down_out_5d);
        try mlx.check(mlx.mlx_gather_qmm(&down_out_5d, act_exp, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, no_idx, inds, true, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", false, self.s));
        var down_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(down_out);
        try mlx.check(mlx.mlx_squeeze(&down_out, down_out_5d, self.s));

        // Weight by scores: down_out * norm_scores[..., None] → sum over K
        var scores_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scores_exp);
        try mlx.check(mlx.mlx_expand_dims(&scores_exp, norm_scores, -1, self.s)); // [B, S, K, 1]
        var weighted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(weighted);
        try mlx.check(mlx.mlx_multiply(&weighted, down_out, scores_exp, self.s));
        var expert_sum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(expert_sum);
        try mlx.check(mlx.mlx_sum_axis(&expert_sum, weighted, -2, false, self.s)); // [B, S, hidden]

        // Shared expert: silu(gate_proj(x)) * up_proj(x) → down_proj(...)
        const sh_gate = try self.qmatmul(x, mw.shared_gate_w, mw.shared_gate_s, mw.shared_gate_b);
        defer _ = mlx.mlx_array_free(sh_gate);
        const sh_up = try self.qmatmul(x, mw.shared_up_w, mw.shared_up_s, mw.shared_up_b);
        defer _ = mlx.mlx_array_free(sh_up);
        const sh_act = try self.swiglu(sh_gate, sh_up);
        defer _ = mlx.mlx_array_free(sh_act);
        const sh_down = try self.qmatmul(sh_act, mw.shared_down_w, mw.shared_down_s, mw.shared_down_b);
        defer _ = mlx.mlx_array_free(sh_down);

        // Shared expert gate: sigmoid(gate(x)) * shared_output
        const seg_bits = detectQuantBits(mw.shared_expert_gate_w, mw.shared_expert_gate_s, gs);
        const sh_gate_logit = try qmatmulBits(x, mw.shared_expert_gate_w, mw.shared_expert_gate_s, mw.shared_expert_gate_b, seg_bits, gs, self.s);
        defer _ = mlx.mlx_array_free(sh_gate_logit);
        var sh_gate_sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sh_gate_sig);
        try mlx.check(mlx.mlx_sigmoid(&sh_gate_sig, sh_gate_logit, self.s));
        var shared_gated = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(shared_gated);
        try mlx.check(mlx.mlx_multiply(&shared_gated, sh_gate_sig, sh_down, self.s));

        // Combine: expert_sum + shared_gated
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&result, expert_sum, shared_gated, self.s));
        return result;
    }

    // ── Mask helpers ──

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

// ── Init helpers ──

fn initStandardLayers(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights, name_buf: *[256]u8, s: mlx.mlx_stream) ![]LayerWeights {
    std.debug.print("Precomputing layer weights...\n", .{});
    const prefix = config.weight_prefix;
    const layers = try allocator.alloc(LayerWeights, config.num_hidden_layers);

    for (0..config.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        const lw = &layers[i];

        const input_norm_raw = getLayerWeight(weights, name_buf, prefix, li, "input_layernorm.weight");
        lw.input_norm = if (config.norm_has_offset) try addOne(input_norm_raw, s) else input_norm_raw;
        const post_attn_raw = getLayerWeight(weights, name_buf, prefix, li, "post_attention_layernorm.weight");
        lw.post_attn_norm = if (config.norm_has_offset) try addOne(post_attn_raw, s) else post_attn_raw;

        if (config.has_pre_ff_norm) {
            const pre_ff_raw = getLayerWeight(weights, name_buf, prefix, li, "pre_feedforward_layernorm.weight");
            lw.pre_ff_norm = if (config.norm_has_offset) try addOne(pre_ff_raw, s) else pre_ff_raw;
            const post_ff_raw = getLayerWeight(weights, name_buf, prefix, li, "post_feedforward_layernorm.weight");
            lw.post_ff_norm = if (config.norm_has_offset) try addOne(post_ff_raw, s) else post_ff_raw;
        } else {
            lw.pre_ff_norm = null;
            lw.post_ff_norm = null;
        }

        if (config.has_qk_norm) {
            const q_norm_raw = getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_norm.weight");
            lw.q_norm = if (config.norm_has_offset) try addOne(q_norm_raw, s) else q_norm_raw;
            const k_norm_raw = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_norm.weight");
            lw.k_norm = if (config.norm_has_offset) try addOne(k_norm_raw, s) else k_norm_raw;
        } else {
            lw.q_norm = null;
            lw.k_norm = null;
        }

        lw.q_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.weight");
        lw.q_s = getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.scales");
        lw.q_b = getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.biases");
        lw.k_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.weight");
        lw.k_s = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.scales");
        lw.k_b = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.biases");
        lw.v_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.v_proj.weight");
        lw.v_s = getLayerWeight(weights, name_buf, prefix, li, "self_attn.v_proj.scales");
        lw.v_b = getLayerWeight(weights, name_buf, prefix, li, "self_attn.v_proj.biases");
        lw.o_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.o_proj.weight");
        lw.o_s = getLayerWeight(weights, name_buf, prefix, li, "self_attn.o_proj.scales");
        lw.o_b = getLayerWeight(weights, name_buf, prefix, li, "self_attn.o_proj.biases");

        lw.gate_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.gate_proj.weight");
        lw.gate_s = getLayerWeight(weights, name_buf, prefix, li, "mlp.gate_proj.scales");
        lw.gate_b = getLayerWeight(weights, name_buf, prefix, li, "mlp.gate_proj.biases");
        lw.up_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.up_proj.weight");
        lw.up_s = getLayerWeight(weights, name_buf, prefix, li, "mlp.up_proj.scales");
        lw.up_b = getLayerWeight(weights, name_buf, prefix, li, "mlp.up_proj.biases");
        lw.down_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.down_proj.weight");
        lw.down_s = getLayerWeight(weights, name_buf, prefix, li, "mlp.down_proj.scales");
        lw.down_b = getLayerWeight(weights, name_buf, prefix, li, "mlp.down_proj.biases");
    }
    return layers;
}

fn initMoeLayers(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights, name_buf: *[256]u8, _: mlx.mlx_stream) !struct { moe_layers: []MoeLayerWeights, ssm_entries: []SSMCacheEntry } {
    std.debug.print("Precomputing MoE layer weights...\n", .{});
    const prefix = config.weight_prefix;
    const moe_layers = try allocator.alloc(MoeLayerWeights, config.num_hidden_layers);
    const ssm_entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);

    for (0..config.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        const lw = &moe_layers[i];
        const is_linear = config.isLinearLayer(li);

        lw.input_norm = getLayerWeight(weights, name_buf, prefix, li, "input_layernorm.weight");
        lw.post_attn_norm = getLayerWeight(weights, name_buf, prefix, li, "post_attention_layernorm.weight");
        lw.is_linear = is_linear;

        if (is_linear) {
            // Detect combined (qkvz+ba) vs separate (qkv+z+a+b) projections
            const combined = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkvz.weight") != null;
            if (combined) {
                lw.attn = .{ .linear = .{
                    .combined_proj = true,
                    .qkv_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_qkvz.weight"),
                    .qkv_s = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_qkvz.scales"),
                    .qkv_b = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_qkvz.biases"),
                    .z_w = mlx.mlx_array_new(),
                    .z_s = mlx.mlx_array_new(),
                    .z_b = mlx.mlx_array_new(),
                    .a_w = mlx.mlx_array_new(),
                    .a_s = mlx.mlx_array_new(),
                    .a_b = mlx.mlx_array_new(),
                    .b_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_ba.weight"),
                    .b_s = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_ba.scales"),
                    .b_b = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_ba.biases"),
                    .conv1d_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.conv1d.weight"),
                    .A_log = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.A_log"),
                    .dt_bias = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.dt_bias"),
                    .norm_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.norm.weight"),
                    .out_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.out_proj.weight"),
                    .out_s = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.out_proj.scales"),
                    .out_b = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.out_proj.biases"),
                } };
            } else {
                lw.attn = .{ .linear = .{
                    .qkv_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_qkv.weight"),
                    .qkv_s = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_qkv.scales"),
                    .qkv_b = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_qkv.biases"),
                    .z_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_z.weight"),
                    .z_s = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_z.scales"),
                    .z_b = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_z.biases"),
                    .a_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_a.weight"),
                    .a_s = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_a.scales"),
                    .a_b = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_a.biases"),
                    .b_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_b.weight"),
                    .b_s = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_b.scales"),
                    .b_b = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_b.biases"),
                    .conv1d_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.conv1d.weight"),
                    .A_log = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.A_log"),
                    .dt_bias = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.dt_bias"),
                    .norm_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.norm.weight"),
                    .out_w = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.out_proj.weight"),
                    .out_s = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.out_proj.scales"),
                    .out_b = getLayerWeight(weights, name_buf, prefix, li, "linear_attn.out_proj.biases"),
                } };
            }
        } else {
            lw.attn = .{ .full = .{
                .q_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.weight"),
                .q_s = getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.scales"),
                .q_b = getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.biases"),
                .k_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.weight"),
                .k_s = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.scales"),
                .k_b = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.biases"),
                .v_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.v_proj.weight"),
                .v_s = getLayerWeight(weights, name_buf, prefix, li, "self_attn.v_proj.scales"),
                .v_b = getLayerWeight(weights, name_buf, prefix, li, "self_attn.v_proj.biases"),
                .o_w = getLayerWeight(weights, name_buf, prefix, li, "self_attn.o_proj.weight"),
                .o_s = getLayerWeight(weights, name_buf, prefix, li, "self_attn.o_proj.scales"),
                .o_b = getLayerWeight(weights, name_buf, prefix, li, "self_attn.o_proj.biases"),
                .q_norm = getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_norm.weight"),
                .k_norm = getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_norm.weight"),
            } };
        }

        lw.mlp = .{
            .router_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.gate.weight"),
            .router_s = getLayerWeight(weights, name_buf, prefix, li, "mlp.gate.scales"),
            .router_b = getLayerWeight(weights, name_buf, prefix, li, "mlp.gate.biases"),
            .switch_gate_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.switch_mlp.gate_proj.weight"),
            .switch_gate_s = getLayerWeight(weights, name_buf, prefix, li, "mlp.switch_mlp.gate_proj.scales"),
            .switch_gate_b = getLayerWeight(weights, name_buf, prefix, li, "mlp.switch_mlp.gate_proj.biases"),
            .switch_up_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.switch_mlp.up_proj.weight"),
            .switch_up_s = getLayerWeight(weights, name_buf, prefix, li, "mlp.switch_mlp.up_proj.scales"),
            .switch_up_b = getLayerWeight(weights, name_buf, prefix, li, "mlp.switch_mlp.up_proj.biases"),
            .switch_down_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.switch_mlp.down_proj.weight"),
            .switch_down_s = getLayerWeight(weights, name_buf, prefix, li, "mlp.switch_mlp.down_proj.scales"),
            .switch_down_b = getLayerWeight(weights, name_buf, prefix, li, "mlp.switch_mlp.down_proj.biases"),
            .shared_gate_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.weight"),
            .shared_gate_s = getLayerWeight(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.scales"),
            .shared_gate_b = getLayerWeight(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.biases"),
            .shared_up_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.weight"),
            .shared_up_s = getLayerWeight(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.scales"),
            .shared_up_b = getLayerWeight(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.biases"),
            .shared_down_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.weight"),
            .shared_down_s = getLayerWeight(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.scales"),
            .shared_down_b = getLayerWeight(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.biases"),
            .shared_expert_gate_w = getLayerWeight(weights, name_buf, prefix, li, "mlp.shared_expert_gate.weight"),
            .shared_expert_gate_s = getLayerWeight(weights, name_buf, prefix, li, "mlp.shared_expert_gate.scales"),
            .shared_expert_gate_b = getLayerWeight(weights, name_buf, prefix, li, "mlp.shared_expert_gate.biases"),
        };

        ssm_entries[i] = .{
            .conv_state = mlx.mlx_array_new(),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = false,
        };
    }

    return .{ .moe_layers = moe_layers, .ssm_entries = ssm_entries };
}

fn appendFullAttnWeights(vec: mlx.mlx_vector_array, fa: *const FullAttnWeights) void {
    inline for (std.meta.fields(FullAttnWeights)) |field| {
        _ = mlx.mlx_vector_array_append_value(vec, @field(fa, field.name));
    }
}

fn appendLinearAttnWeights(vec: mlx.mlx_vector_array, la: *const LinearAttnWeights) void {
    inline for (std.meta.fields(LinearAttnWeights)) |field| {
        if (field.type != mlx.mlx_array) continue;
        if (comptime std.mem.startsWith(u8, field.name, "z_") or std.mem.startsWith(u8, field.name, "a_")) {
            if (!la.combined_proj)
                _ = mlx.mlx_vector_array_append_value(vec, @field(la, field.name));
        } else {
            _ = mlx.mlx_vector_array_append_value(vec, @field(la, field.name));
        }
    }
}

fn appendMoeWeights(vec: mlx.mlx_vector_array, mw: *const MoeMlpWeights) void {
    inline for (std.meta.fields(MoeMlpWeights)) |field| {
        _ = mlx.mlx_vector_array_append_value(vec, @field(mw, field.name));
    }
}

// ── Utility functions ──

/// Detect quantization bits from weight and scales shapes: bits = w_cols * 32 / (s_cols * group_size)
fn detectQuantBits(w: mlx.mlx_array, sc: mlx.mlx_array, group_size: u32) u32 {
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    if (w_shape.len < 2 or s_shape.len < 2) return 4;
    const w_cols: u32 = @intCast(w_shape[w_shape.len - 1]);
    const s_cols: u32 = @intCast(s_shape[s_shape.len - 1]);
    if (s_cols == 0) return 4;
    return (w_cols * 32) / (s_cols * group_size);
}

fn qmatmulBits(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_quantized_matmul(
        &result,
        x,
        w,
        sc,
        bi,
        true,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        s,
    ));
    return result;
}

/// Extract timestep t from a [B, T, H, D] tensor → [B, H, D]
fn sliceTimestep4(arr: mlx.mlx_array, batch: c_int, heads: c_int, dim: c_int, t: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const start = [_]c_int{ 0, t, 0, 0 };
    const stop = [_]c_int{ batch, t + 1, heads, dim };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    try mlx.check(mlx.mlx_slice(&sliced, arr, &start, 4, &stop, 4, &strides, 4, s));
    const out_shape = [_]c_int{ batch, heads, dim };
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&result, sliced, &out_shape, 3, s));
    return result;
}

/// Extract timestep t from a [B, T, H] tensor → [B, H]
fn sliceTimestep3(arr: mlx.mlx_array, batch: c_int, heads: c_int, t: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const start = [_]c_int{ 0, t, 0 };
    const stop = [_]c_int{ batch, t + 1, heads };
    const strides = [_]c_int{ 1, 1, 1 };
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    try mlx.check(mlx.mlx_slice(&sliced, arr, &start, 3, &stop, 3, &strides, 3, s));
    const out_shape = [_]c_int{ batch, heads };
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&result, sliced, &out_shape, 2, s));
    return result;
}

fn getWeightFmt(weights: *const Weights, buf: *[256]u8, comptime fmt: []const u8, prefix: []const u8) mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, fmt, .{prefix}) catch unreachable;
    return weights.get(name) orelse {
        std.debug.print("MISSING WEIGHT: {s}\n", .{name});
        unreachable;
    };
}

fn getWeightFmtOpt(weights: *const Weights, buf: *[256]u8, comptime fmt: []const u8, prefix: []const u8) ?mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, fmt, .{prefix}) catch unreachable;
    return weights.get(name);
}

fn getLayerWeightOpt(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8) ?mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ prefix, layer, suffix }) catch unreachable;
    return weights.get(name);
}

fn getLayerWeight(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8) mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ prefix, layer, suffix }) catch unreachable;
    return weights.get(name) orelse {
        std.debug.print("MISSING WEIGHT: {s}\n", .{name});
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

fn addOne(arr: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const one = bf16Scalar(1.0, s);
    defer _ = mlx.mlx_array_free(one);
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&result, one, arr, s));
    return result;
}
