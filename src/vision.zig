const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");
const qwen_vision = @import("qwen_vision.zig");
const muse_vision = @import("muse_vision.zig");
const lfm2_vision = @import("lfm2_vision.zig");

const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;

// ── Linear Weight (optionally clipped) ──
// Gemma 4 E4B vision uses "clipped linears": bf16 dense weights with input/output clamping.
// Forward: clip(x, in_min, in_max) → matmul(x, w.T) → clip(y, out_min, out_max)
// Gemma 4 26B/31B use plain linears (no clipping): matmul(x, w.T)
// When `has_clip` is false, the min/max arrays are unused placeholders.

const LinearWeight = struct {
    weight: mlx.mlx_array, // [out_dim, in_dim] bf16
    has_clip: bool,
    input_min: mlx.mlx_array, // unused when has_clip == false
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
    q_proj: LinearWeight,
    k_proj: LinearWeight,
    v_proj: LinearWeight,
    out_proj: LinearWeight,
    q_norm: mlx.mlx_array,
    k_norm: mlx.mlx_array,

    // MLP
    gate_proj: LinearWeight,
    up_proj: LinearWeight,
    down_proj: LinearWeight,
};

// ── Gemma 4 12B "unified" (encoder-free) embedder ──
// Vision and audio inputs are projected straight into language-model space —
// there is no SigLIP transformer tower and no conformer. Vision: raw 48×48 RGB
// patches → LayerNorm → Dense → LayerNorm → +factorized 2D posemb → LayerNorm →
// RMSNorm → Linear. Audio: raw 640-sample frames → RMSNorm → Linear. Reference:
// transformers Gemma4UnifiedVisionEmbedder / Gemma4UnifiedMultimodalEmbedder.
const UnifiedWeights = struct {
    // Patch embedder (nn.LayerNorm has weight+bias; eps 1e-5).
    patch_ln1_w: mlx.mlx_array, // [6912]
    patch_ln1_b: mlx.mlx_array,
    patch_dense_w: mlx.mlx_array, // quant linear 6912 → mm_embed_dim
    patch_dense_s: mlx.mlx_array,
    patch_dense_b: mlx.mlx_array,
    patch_dense_bias: mlx.mlx_array, // [mm_embed_dim] additive bias
    patch_dense_bits: u32,
    patch_ln2_w: mlx.mlx_array, // [mm_embed_dim]
    patch_ln2_b: mlx.mlx_array,
    pos_embedding: mlx.mlx_array, // [posemb_size, 2, mm_embed_dim]
    pos_norm_w: mlx.mlx_array, // [mm_embed_dim]
    pos_norm_b: mlx.mlx_array,
    // embed_vision projection (RMSNorm-noscale → Linear, no bias).
    ev_w: mlx.mlx_array,
    ev_s: mlx.mlx_array,
    ev_b: mlx.mlx_array,
    ev_bits: u32,
    // embed_audio projection (RMSNorm-noscale → Linear, no bias). Optional —
    // present only when the checkpoint ships audio weights and --no-vision off.
    ea_w: ?mlx.mlx_array,
    ea_s: mlx.mlx_array,
    ea_b: mlx.mlx_array,
    ea_bits: u32,
    model_patch_size: c_int, // 48
};

// ── Vision Encoder ──
// Matches mlx-vlm's Gemma4 VisionModel: SigLIP encoder with 2D RoPE,
// clipped linears, V-norm, position-based pooling, post-projection norm.

/// Encoder outputs by input hash, so a conversation's history media is encoded
/// once per load, not on every turn (20 history screenshots re-encoded cost
/// Flash-Next ~7 s per turn). Inference thread only, like every mlx call.
pub const EmbeddingCache = struct {
    const Entry = struct { key: u64, emb: mlx.mlx_array, bytes: usize, last_used: u64 };
    pub const CAP_BYTES: usize = 256 << 20;

    entries: std.ArrayList(Entry) = .empty,
    bytes: usize = 0,
    clock: u64 = 0,

    /// A new handle to the cached rows (caller frees), or null.
    pub fn get(self: *EmbeddingCache, key: u64) ?mlx.mlx_array {
        for (self.entries.items) |*e| {
            if (e.key != key) continue;
            self.clock += 1;
            e.last_used = self.clock;
            var out = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&out, e.emb);
            return out;
        }
        return null;
    }

    /// Mark the start of a request: entries it touches from here on are not
    /// evicted by its own puts.
    pub fn beginRequest(self: *const EmbeddingCache) u64 {
        return self.clock;
    }

    /// Evaluate `emb` and keep it under `key`, evicting least recently used
    /// entries past `cap` but never one used since `epoch`: every turn reads
    /// the whole history in order, and plain LRU on that scan evicts exactly
    /// the entry the next turn needs first. When nothing is evictable the new
    /// entry is not kept. Best effort: a failure only skips the cache.
    pub fn put(self: *EmbeddingCache, allocator: std.mem.Allocator, key: u64, emb: mlx.mlx_array, cap: usize, epoch: u64) void {
        if (mlx.mlx_array_eval(emb) != 0) return;
        const bytes = mlx.mlx_array_size(emb) * mlx.mlx_array_itemsize(emb);
        if (bytes > cap) return;
        while (self.bytes + bytes > cap) {
            var lru: ?usize = null;
            for (self.entries.items, 0..) |e, i| {
                if (e.last_used > epoch) continue;
                if (lru == null or e.last_used < self.entries.items[lru.?].last_used) lru = i;
            }
            const victim = lru orelse return;
            const gone = self.entries.swapRemove(victim);
            _ = mlx.mlx_array_free(gone.emb);
            self.bytes -= gone.bytes;
        }
        var held = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&held, emb);
        self.clock += 1;
        self.entries.append(allocator, .{ .key = key, .emb = held, .bytes = bytes, .last_used = self.clock }) catch {
            _ = mlx.mlx_array_free(held);
            return;
        };
        self.bytes += bytes;
    }

    /// Drop everything: the bytes go back to a request that needs them.
    pub fn clear(self: *EmbeddingCache) void {
        for (self.entries.items) |e| _ = mlx.mlx_array_free(e.emb);
        self.entries.clearRetainingCapacity();
        self.bytes = 0;
    }

    pub fn deinit(self: *EmbeddingCache, allocator: std.mem.Allocator) void {
        self.clear();
        self.entries.deinit(allocator);
        self.* = .{};
    }
};

pub const VisionEncoder = struct {
    config: ModelConfig,
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,
    emb_cache: EmbeddingCache = .{},

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

    // Gemma-4 26B/31B: learned pre-encoder standardization (scale * h + bias on last dim).
    // null for E4B which uses clipped linears and no standardization.
    std_scale: ?mlx.mlx_array,
    std_bias: ?mlx.mlx_array,

    // Constants
    rms_eps: f32,
    half: mlx.mlx_array,
    one: mlx.mlx_array,

    // Gemma 4 12B encoder-free path. When set, `forward`/`forwardAudio` use the
    // unified embedder and all SigLIP fields above are unused sentinels.
    unified: ?UnifiedWeights = null,

    // Patch-grid ViT paths (Qwen3-VL / Muse-Glimmer). When either is set,
    // `forwardPatches` drives the tower and all SigLIP fields above are unused
    // sentinels. See src/qwen_vision.zig / src/muse_vision.zig.
    qwen: ?qwen_vision.QwenVision = null,
    muse: ?muse_vision.MuseVision = null,
    lfm2: ?lfm2_vision.Lfm2Vision = null,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !VisionEncoder {
        if (config.is_gemma4_unified) return initUnified(allocator, config, weights);
        if (config.qwen_vision) return initQwen(allocator, config, weights);
        if (config.muse_vision) return initMuse(allocator, config, weights);
        if (config.lfm2_vision) return initLfm2(allocator, config, weights);
        const s = mlx.mlx_default_gpu_stream_new();

        var name_buf: [256]u8 = undefined;

        // Check for essential vision weights before allocating layers.
        // Models like Qwen 3.5 have vision_config in config.json but ship
        // without vision weights when quantized text-only.
        const patch_w = getWeight(weights, &name_buf, "vision_tower.patch_embedder.input_proj.weight") orelse {
            log.warn("MISSING WEIGHT: vision_tower.patch_embedder.input_proj.weight\n", .{});
            return error.MissingVisionWeights;
        };
        const pos_emb = getWeight(weights, &name_buf, "vision_tower.patch_embedder.position_embedding_table") orelse {
            log.warn("MISSING WEIGHT: vision_tower.patch_embedder.position_embedding_table\n", .{});
            return error.MissingVisionWeights;
        };

        // Vision weights confirmed present — load all layers
        const num_layers = config.vision_num_layers;
        var layers = try allocator.alloc(VisionLayerWeights, num_layers);
        errdefer allocator.free(layers);

        const clipped = config.vision_use_clipped_linears;

        for (0..num_layers) |i| {
            layers[i] = .{
                .input_layernorm = try getVisionWeight(weights, &name_buf, i, "input_layernorm.weight"),
                .post_attention_layernorm = try getVisionWeight(weights, &name_buf, i, "post_attention_layernorm.weight"),
                .pre_feedforward_layernorm = try getVisionWeight(weights, &name_buf, i, "pre_feedforward_layernorm.weight"),
                .post_feedforward_layernorm = try getVisionWeight(weights, &name_buf, i, "post_feedforward_layernorm.weight"),
                .q_proj = try loadLinearWeight(weights, &name_buf, i, "self_attn.q_proj", clipped),
                .k_proj = try loadLinearWeight(weights, &name_buf, i, "self_attn.k_proj", clipped),
                .v_proj = try loadLinearWeight(weights, &name_buf, i, "self_attn.v_proj", clipped),
                .out_proj = try loadLinearWeight(weights, &name_buf, i, "self_attn.o_proj", clipped),
                .q_norm = try getVisionWeight(weights, &name_buf, i, "self_attn.q_norm.weight"),
                .k_norm = try getVisionWeight(weights, &name_buf, i, "self_attn.k_norm.weight"),
                .gate_proj = try loadLinearWeight(weights, &name_buf, i, "mlp.gate_proj", clipped),
                .up_proj = try loadLinearWeight(weights, &name_buf, i, "mlp.up_proj", clipped),
                .down_proj = try loadLinearWeight(weights, &name_buf, i, "mlp.down_proj", clipped),
            };
        }

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

        // Gemma-4 26B/31B learned standardization (per-channel scale + bias).
        // Applied once after patch + position embedding, before the transformer stack.
        const std_scale_opt = getWeight(weights, &name_buf, "vision_tower.std_scale");
        const std_bias_opt = getWeight(weights, &name_buf, "vision_tower.std_bias");

        // Batch eval all vision weights
        {
            var eval_list = std.ArrayList(mlx.mlx_array).empty;
            defer eval_list.deinit(allocator);
            try eval_list.append(allocator, patch_w);
            try eval_list.append(allocator, pos_emb);
            try eval_list.append(allocator, proj_w);
            // Dense bf16 projector has null-ctx scales/biases; mlx_array_ndim
            // throws on a null handle (it does not return 0), so gate on .ctx.
            if (proj_s.ctx != null) try eval_list.append(allocator, proj_s);
            if (proj_b.ctx != null) try eval_list.append(allocator, proj_b);
            if (std_scale_opt) |a| try eval_list.append(allocator, a);
            if (std_bias_opt) |a| try eval_list.append(allocator, a);
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

        log.info("Vision encoder: {d} layers, hidden={d}, heads={d}, pool→{d} tokens{s}{s}\n", .{
            num_layers, config.vision_hidden_size, config.vision_num_heads, config.vision_soft_tokens,
            if (clipped) ", clipped" else "",
            if (std_scale_opt != null) ", std" else "",
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
            .std_scale = std_scale_opt,
            .std_bias = std_bias_opt,
            .rms_eps = 1e-6,
            .half = bf16Scalar(0.5, s),
            .one = bf16Scalar(1.0, s),
        };
    }

    pub fn deinit(self: *VisionEncoder) void {
        self.emb_cache.deinit(self.allocator);
        _ = mlx.mlx_array_free(self.half);
        _ = mlx.mlx_array_free(self.one);
        if (self.qwen) |*q| q.deinit();
        if (self.muse) |*m| m.deinit();
        if (self.lfm2) |*l| l.deinit();
        self.allocator.free(self.layers);
        _ = mlx.mlx_stream_free(self.s);
    }

    /// True when this encoder can embed audio — i.e. the Gemma 4 12B unified
    /// checkpoint shipped `embed_audio.*` weights. Drives the `audio` capability
    /// reported by /v1/models so the app only offers the mic on audio models.
    pub fn supportsAudio(self: *const VisionEncoder) bool {
        return if (self.unified) |u| u.ea_w != null else false;
    }

    // ── Gemma 4 12B unified (encoder-free) ──

    /// Build a VisionEncoder that holds only the unified patch/audio embedders.
    /// SigLIP fields are sentinels; `unified` drives forward().
    fn initUnified(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !VisionEncoder {
        const s = mlx.mlx_default_gpu_stream_new();
        var nb: [256]u8 = undefined;

        const dense_w = getWeight(weights, &nb, "vision_embedder.patch_dense.weight") orelse {
            log.err("MISSING WEIGHT: vision_embedder.patch_dense.weight\n", .{});
            return error.MissingVisionWeights;
        };
        const dense_s = getWeight(weights, &nb, "vision_embedder.patch_dense.scales") orelse mlx.mlx_array_new();
        const dense_b = getWeight(weights, &nb, "vision_embedder.patch_dense.biases") orelse mlx.mlx_array_new();
        const ev_w = getWeight(weights, &nb, "embed_vision.embedding_projection.weight") orelse {
            log.err("MISSING WEIGHT: embed_vision.embedding_projection.weight\n", .{});
            return error.MissingVisionWeights;
        };
        const ev_s = getWeight(weights, &nb, "embed_vision.embedding_projection.scales") orelse mlx.mlx_array_new();
        const ev_b = getWeight(weights, &nb, "embed_vision.embedding_projection.biases") orelse mlx.mlx_array_new();
        const ea_w = getWeight(weights, &nb, "embed_audio.embedding_projection.weight");
        const ea_s = getWeight(weights, &nb, "embed_audio.embedding_projection.scales") orelse mlx.mlx_array_new();
        const ea_b = getWeight(weights, &nb, "embed_audio.embedding_projection.biases") orelse mlx.mlx_array_new();

        const gs = config.quant_group_size;
        const unified = UnifiedWeights{
            .patch_ln1_w = getWeight(weights, &nb, "vision_embedder.patch_ln1.weight").?,
            .patch_ln1_b = getWeight(weights, &nb, "vision_embedder.patch_ln1.bias").?,
            .patch_dense_w = dense_w,
            .patch_dense_s = dense_s,
            .patch_dense_b = dense_b,
            .patch_dense_bias = getWeight(weights, &nb, "vision_embedder.patch_dense.bias").?,
            .patch_dense_bits = detectProjBits(dense_w, dense_s, gs),
            .patch_ln2_w = getWeight(weights, &nb, "vision_embedder.patch_ln2.weight").?,
            .patch_ln2_b = getWeight(weights, &nb, "vision_embedder.patch_ln2.bias").?,
            .pos_embedding = getWeight(weights, &nb, "vision_embedder.pos_embedding").?,
            .pos_norm_w = getWeight(weights, &nb, "vision_embedder.pos_norm.weight").?,
            .pos_norm_b = getWeight(weights, &nb, "vision_embedder.pos_norm.bias").?,
            .ev_w = ev_w,
            .ev_s = ev_s,
            .ev_b = ev_b,
            .ev_bits = detectProjBits(ev_w, ev_s, gs),
            .ea_w = ea_w,
            .ea_s = ea_s,
            .ea_b = ea_b,
            .ea_bits = if (ea_w) |w| detectProjBits(w, ea_s, gs) else 4,
            .model_patch_size = @intCast(if (config.vision_model_patch_size > 0) config.vision_model_patch_size else 48),
        };

        log.info("Vision encoder: Gemma 4 12B unified (encoder-free), patch={d}px, mm_embed_dim={d}, posemb={d}{s}\n", .{
            unified.model_patch_size, config.vision_mm_embed_dim, config.vision_mm_posemb_size,
            if (ea_w != null) ", +audio" else "",
        });

        return .{
            .config = config,
            .s = s,
            .allocator = allocator,
            .patch_proj_w = mlx.mlx_array_new(),
            .position_embedding = mlx.mlx_array_new(),
            .layers = &.{},
            .proj_w = mlx.mlx_array_new(),
            .proj_s = mlx.mlx_array_new(),
            .proj_b = mlx.mlx_array_new(),
            .proj_quant_bits = 4,
            .proj_quant_group_size = gs,
            .std_scale = null,
            .std_bias = null,
            .rms_eps = 1e-6, // embed_{vision,audio} pre-projection RMSNorm eps
            .half = bf16Scalar(0.5, s),
            .one = bf16Scalar(1.0, s),
            .unified = unified,
        };
    }

    /// Build a VisionEncoder wrapping the Qwen3-VL ViT. SigLIP fields are
    /// sentinels; `qwen` drives `forwardQwen`. The ViT owns its own GPU stream.
    fn initQwen(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !VisionEncoder {
        const s = mlx.mlx_default_gpu_stream_new();
        const qv = try qwen_vision.QwenVision.init(allocator, config, weights);
        log.info("Vision encoder: Qwen3-VL ViT (depth={d}, hidden={d}, heads={d}, merge={d}, out_hidden={d})\n", .{
            config.qv_depth, config.qv_hidden, config.qv_heads, config.qv_merge, config.qv_out_hidden,
        });
        return .{
            .config = config,
            .s = s,
            .allocator = allocator,
            .patch_proj_w = mlx.mlx_array_new(),
            .position_embedding = mlx.mlx_array_new(),
            .layers = &.{},
            .proj_w = mlx.mlx_array_new(),
            .proj_s = mlx.mlx_array_new(),
            .proj_b = mlx.mlx_array_new(),
            .proj_quant_bits = 4,
            .proj_quant_group_size = config.quant_group_size,
            .std_scale = null,
            .std_bias = null,
            .rms_eps = 1e-6,
            .half = bf16Scalar(0.5, s),
            .one = bf16Scalar(1.0, s),
            .qwen = qv,
        };
    }

    /// Build a VisionEncoder wrapping the Muse-Glimmer ViT. Same sentinel shape
    /// as `initQwen`; `muse` drives `forwardPatches`.
    fn initMuse(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !VisionEncoder {
        const s = mlx.mlx_default_gpu_stream_new();
        const mv = try muse_vision.MuseVision.init(allocator, config, weights);
        return .{
            .config = config,
            .s = s,
            .allocator = allocator,
            .patch_proj_w = mlx.mlx_array_new(),
            .position_embedding = mlx.mlx_array_new(),
            .layers = &.{},
            .proj_w = mlx.mlx_array_new(),
            .proj_s = mlx.mlx_array_new(),
            .proj_b = mlx.mlx_array_new(),
            .proj_quant_bits = 4,
            .proj_quant_group_size = config.quant_group_size,
            .std_scale = null,
            .std_bias = null,
            .rms_eps = 1e-6,
            .half = bf16Scalar(0.5, s),
            .one = bf16Scalar(1.0, s),
            .muse = mv,
        };
    }

    /// Build a VisionEncoder wrapping the LFM2-VL SigLIP2-NaFlex tower. Same
    /// sentinel shape as `initQwen`; `lfm2` drives `forwardPatches`.
    fn initLfm2(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !VisionEncoder {
        const s = mlx.mlx_default_gpu_stream_new();
        const lv = try lfm2_vision.Lfm2Vision.init(allocator, config, weights);
        return .{
            .config = config,
            .s = s,
            .allocator = allocator,
            .patch_proj_w = mlx.mlx_array_new(),
            .position_embedding = mlx.mlx_array_new(),
            .layers = &.{},
            .proj_w = mlx.mlx_array_new(),
            .proj_s = mlx.mlx_array_new(),
            .proj_b = mlx.mlx_array_new(),
            .proj_quant_bits = 4,
            .proj_quant_group_size = config.quant_group_size,
            .std_scale = null,
            .std_bias = null,
            .rms_eps = 1e-6,
            .half = bf16Scalar(0.5, s),
            .one = bf16Scalar(1.0, s),
            .lfm2 = lv,
        };
    }

    /// Encode one image on a patch-grid tower: `patches` is the processor's
    /// pixel_values [N, feat]; `grid_h/grid_w` is the full patch grid.
    /// Returns [1, N/merge², out_hidden].
    pub fn forwardPatches(self: *VisionEncoder, patches: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        if (self.muse) |*mv| return mv.forward(patches, grid_h, grid_w);
        if (self.qwen) |*qv| return qv.forward(patches, grid_h, grid_w);
        if (self.lfm2) |*lv| return lv.forward(patches, grid_h, grid_w);
        return error.NoPatchGridEncoder;
    }

    /// Encode one VIDEO: `patches` holds `grid_t` temporal-patch groups'
    /// pixel_values concatenated (see `qwen_vision.buildPixelValuesVideo`).
    /// Only Qwen3-VL-family checkpoints declare `video_token_id` — muse and
    /// lfm2 have no video path.
    pub fn forwardVideoPatches(self: *VisionEncoder, patches: mlx.mlx_array, grid_t: u32, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        if (self.qwen) |*qv| return qv.forwardVideo(patches, grid_t, grid_h, grid_w);
        if (self.config.isEmbeddingGemma2()) return self.forwardFrames(patches, grid_t, grid_h, grid_w);
        return error.NoVideoEncoder;
    }

    /// A video on EmbeddingGemma 2's tower is its frames, each encoded as an image of its own. `frames` is the
    /// buffer the scheduler views as patch rows; its bytes are `grid_t` float32 CHW frames back to back, each
    /// `grid_h x grid_w` patches. Returns every frame's soft tokens in order, [1, grid_t * n, hidden].
    fn forwardFrames(self: *VisionEncoder, frames: mlx.mlx_array, grid_t: u32, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const ps: c_int = @intCast(self.config.vision_patch_size);
        const h: c_int = @as(c_int, @intCast(grid_h)) * ps;
        const w: c_int = @as(c_int, @intCast(grid_w)) * ps;
        const shape = [_]c_int{ @intCast(grid_t), 3, h, w };
        var chw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(chw);
        try mlx.check(mlx.mlx_reshape(&chw, frames, &shape, 4, self.s));

        var parts = std.ArrayList(mlx.mlx_array).empty;
        defer {
            for (parts.items) |p| _ = mlx.mlx_array_free(p);
            parts.deinit(self.allocator);
        }
        for (0..grid_t) |t| {
            const start = [_]c_int{ @intCast(t), 0, 0, 0 };
            const stop = [_]c_int{ @intCast(t + 1), 3, h, w };
            const strides = [_]c_int{ 1, 1, 1, 1 };
            var frame = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(frame);
            try mlx.check(mlx.mlx_slice(&frame, chw, &start, 4, &stop, 4, &strides, 4, self.s));
            const tokens = try self.forward(frame);
            errdefer _ = mlx.mlx_array_free(tokens);
            // One frame's activations at a time: the tower's graph is large and none of it is reused.
            try mlx.check(mlx.mlx_array_eval(tokens));
            try parts.append(self.allocator, tokens);
        }
        const vec = mlx.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, self.s));
        return out;
    }

    /// Apply a quantized (or dense) Linear y = x · Wᵀ (+ optional bias). `sc`
    /// non-empty (ndim>0) selects the quantized path; otherwise a plain matmul.
    fn quantLinear(self: *VisionEncoder, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, qb: mlx.mlx_array, bits: u32, bias: ?mlx.mlx_array) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        // Quantized weight has scales (non-null ctx); dense bf16 weight ships a
        // null-ctx `mlx_array_new()` placeholder. Gate on .ctx, not ndim, because
        // mlx_array_ndim throws on a null handle.
        if (sc.ctx != null) {
            try mlx.check(mlx.mlx_quantized_matmul(
                &out, x, w, sc, qb, true,
                mlx.mlx_optional_int.some(@intCast(self.proj_quant_group_size)),
                mlx.mlx_optional_int.some(@intCast(bits)), self.config.quant_mode.cstr(), self.s,
            ));
        } else {
            var wt = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wt);
            try mlx.check(mlx.mlx_transpose(&wt, w, self.s));
            try mlx.check(mlx.mlx_matmul(&out, x, wt, self.s));
        }
        if (bias) |bvec| {
            var biased = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&biased, out, bvec, self.s));
            _ = mlx.mlx_array_free(out);
            out = biased;
        }
        return out;
    }

    /// nn.LayerNorm (weight + bias) over the last dim, eps 1e-5.
    fn layerNorm(self: *VisionEncoder, x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_layer_norm(&out, x, w, b, 1e-5, self.s));
        return out;
    }

    /// Gemma 4 12B unified vision forward: pixels [1,3,H,W] (H,W multiples of
    /// model_patch_size, rescaled to ~[0,1]) → soft tokens [1, L, mm_embed_dim],
    /// L = (H/P)·(W/P). Caller inserts exactly L image placeholder tokens.
    fn forwardUnified(self: *VisionEncoder, u: *const UnifiedWeights, pixels: mlx.mlx_array) !mlx.mlx_array {
        const P = u.model_patch_size;
        const pix_shape = mlx.getShape(pixels);
        const batch = pix_shape[0];
        const height = pix_shape[2];
        const width = pix_shape[3];
        const grid_h = @divExact(height, P);
        const grid_w = @divExact(width, P);
        const L = grid_h * grid_w;

        // 1. Patchify into 48×48 (H,W,C)-ordered patches: [1, L, 3*P*P].
        const patches = try self.patchify(pixels, batch, grid_h, grid_w, P);
        defer _ = mlx.mlx_array_free(patches);
        var h = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&h, patches, .bfloat16, self.s));

        // 2. Patch embed: LN1 → Dense(+bias) → LN2.
        {
            const t = try self.layerNorm(h, u.patch_ln1_w, u.patch_ln1_b);
            _ = mlx.mlx_array_free(h);
            h = t;
        }
        {
            const t = try self.quantLinear(h, u.patch_dense_w, u.patch_dense_s, u.patch_dense_b, u.patch_dense_bits, u.patch_dense_bias);
            _ = mlx.mlx_array_free(h);
            h = t;
        }

        {
            const t = try self.layerNorm(h, u.patch_ln2_w, u.patch_ln2_b);
            _ = mlx.mlx_array_free(h);
            h = t;
        }

        // 3. Add factorized 2D position embeddings, then LN.
        {
            const pos = try self.factorizedPosEmb(u, grid_h, grid_w, batch);
            defer _ = mlx.mlx_array_free(pos);
            var hp = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&hp, h, pos, self.s));
            _ = mlx.mlx_array_free(h);
            h = hp;
        }
        {
            const t = try self.layerNorm(h, u.pos_norm_w, u.pos_norm_b);
            _ = mlx.mlx_array_free(h);
            h = t;
        }

        // 4. embed_vision: parameter-free RMSNorm → Linear (no bias).
        {
            const normed = try self.rmsNormNoScale(h);
            _ = mlx.mlx_array_free(h);
            defer _ = mlx.mlx_array_free(normed);
            h = try self.quantLinear(normed, u.ev_w, u.ev_s, u.ev_b, u.ev_bits, null);
        }
        _ = L;
        return h;
    }

    /// Build factorized 2D position embeddings for an L=grid_h·grid_w patch grid.
    /// pos[i] = pos_embedding[x_i, 0, :] + pos_embedding[y_i, 1, :], where
    /// i = row·grid_w + col, x=col, y=row. Returns [batch, L, mm_embed_dim].
    fn factorizedPosEmb(self: *VisionEncoder, u: *const UnifiedWeights, grid_h: c_int, grid_w: c_int, batch: c_int) !mlx.mlx_array {
        const L: usize = @intCast(grid_h * grid_w);
        const gw: usize = @intCast(grid_w);
        // Host-build x (col) and y (row) index arrays.
        var xs = try self.allocator.alloc(i32, L);
        defer self.allocator.free(xs);
        var ys = try self.allocator.alloc(i32, L);
        defer self.allocator.free(ys);
        for (0..L) |i| {
            xs[i] = @intCast(i % gw);
            ys[i] = @intCast(i / gw);
        }
        const idx_shape = [_]c_int{@intCast(L)};
        const x_idx = mlx.mlx_array_new_data(xs.ptr, &idx_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(x_idx);
        const y_idx = mlx.mlx_array_new_data(ys.ptr, &idx_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(y_idx);

        // pos_embedding: [posemb_size, 2, mm_embed_dim]. Split axis-1 into the
        // x-table (index 0) and y-table (index 1), each [posemb_size, mm_embed_dim].
        const pe_shape = mlx.getShape(u.pos_embedding);
        const pos_size = pe_shape[0];
        const dim = pe_shape[2];
        const table_x = try sliceAxis1(self, u.pos_embedding, 0, pos_size, dim);
        defer _ = mlx.mlx_array_free(table_x);
        const table_y = try sliceAxis1(self, u.pos_embedding, 1, pos_size, dim);
        defer _ = mlx.mlx_array_free(table_y);

        var px = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(px);
        try mlx.check(mlx.mlx_take_axis(&px, table_x, x_idx, 0, self.s)); // [L, dim]
        var py = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(py);
        try mlx.check(mlx.mlx_take_axis(&py, table_y, y_idx, 0, self.s)); // [L, dim]
        var pos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pos);
        try mlx.check(mlx.mlx_add(&pos, px, py, self.s)); // [L, dim]

        // Reshape to [batch, L, dim] (batch is 1 for the per-image call).
        const out_shape = [_]c_int{ batch, @intCast(L), dim };
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&out, pos, &out_shape, 3, self.s));
        return out;
    }

    /// Slice `[N, 2, D]` at axis-1 index `i` → `[N, D]`.
    fn sliceAxis1(self: *VisionEncoder, a: mlx.mlx_array, i: c_int, n: c_int, d: c_int) !mlx.mlx_array {
        const start = [_]c_int{ 0, i, 0 };
        const stop = [_]c_int{ n, i + 1, d };
        const strides = [_]c_int{ 1, 1, 1 };
        var sl = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sl);
        try mlx.check(mlx.mlx_slice(&sl, a, &start, 3, &stop, 3, &strides, 3, self.s));
        const flat = [_]c_int{ n, d };
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&out, sl, &flat, 2, self.s));
        return out;
    }

    /// Gemma 4 12B unified audio forward: frames [1, N, 640] (raw 16 kHz samples,
    /// 640/token) → soft tokens [1, N, mm_embed_dim] via RMSNorm → Linear.
    /// Returns error.AudioNotSupported if the checkpoint shipped no audio weights.
    pub fn forwardAudio(self: *VisionEncoder, frames: mlx.mlx_array) !mlx.mlx_array {
        const u = self.unified orelse return error.AudioNotSupported;
        const ea_w = u.ea_w orelse return error.AudioNotSupported;
        var h = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&h, frames, .bfloat16, self.s));
        const normed = try self.rmsNormNoScale(h);
        _ = mlx.mlx_array_free(h);
        defer _ = mlx.mlx_array_free(normed);
        return self.quantLinear(normed, ea_w, u.ea_s, u.ea_b, u.ea_bits, null);
    }

    /// Forward pass: pixel data [B, 3, H, W] float32 → vision embeddings [B, N_out, text_hidden_size].
    /// N_out is the number of valid pooled tokens (e.g. 25 for 224x224 with kernel=3).
    /// The caller must cycle these to fill the expected image_seq_length (280) token positions.
    pub fn forward(self: *VisionEncoder, pixels: mlx.mlx_array) !mlx.mlx_array {
        if (self.unified) |u| return self.forwardUnified(&u, pixels);
        // A patch-grid tower's SigLIP fields are null sentinels, so a gridless
        // (Gemma-shaped) buffer reaching here is a client/preprocessing
        // mismatch, not something to dereference. Backstop for the parse-level
        // refusal in `parseImageUrlContent`.
        if (self.qwen != null or self.muse != null or self.lfm2 != null) return error.PixelsMissingPatchGrid;
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

        // 2c. Pad the patch sequence out to `max_patches` with zero embeddings so transformer
        //     blocks see the fixed shape they were trained on. Reference (vision.py:459-464).
        //     max_patches = default_output_length * pooling_kernel^2. For Gemma-4: 280 * 9 = 2520.
        // Gemma 4 pads every image to its one token budget. EmbeddingGemma 2 sizes each image to a budget of its
        // own, up to 1120 tokens, which can exceed `default_output_length`: its grid is the sequence.
        const max_patches: c_int = if (cfg.isEmbeddingGemma2()) num_patches else @intCast(cfg.vision_soft_tokens * kernel * kernel);
        const num_padding: c_int = max_patches - num_patches;
        if (num_padding > 0) {
            const pad_shape = [_]c_int{ batch, num_padding, hidden };
            var pad_emb = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(pad_emb);
            try mlx.check(mlx.mlx_zeros(&pad_emb, &pad_shape, 3, .bfloat16, self.s));
            const arrs = [_]mlx.mlx_array{ h, pad_emb };
            const vec = mlx.mlx_vector_array_new_data(&arrs, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            var h_padded = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_concatenate_axis(&h_padded, vec, 1, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_padded;
        }

        // 3. Build 2D position indices for RoPE: [B, max_patches, 2]
        //    Real patches get (x, y) on the grid; padded patches get (-1, -1) per reference.
        const positions = try self.buildPositionGridPadded(batch, grid_h, grid_w, num_patches, max_patches);
        defer _ = mlx.mlx_array_free(positions);

        // 4. Attention mask [B, 1, L, L]: 0 where both Q and K are valid, -inf otherwise.
        //    Broadcast on head dim. Matches reference vision.py:466-473.
        const attn_mask_opt: ?mlx.mlx_array = if (num_padding > 0)
            try self.buildAttentionMask(batch, max_patches, num_patches)
        else
            null;
        defer {
            if (attn_mask_opt) |m| _ = mlx.mlx_array_free(m);
        }

        // 5. Transformer layers (now over `max_patches` with optional mask)
        const seq_len_for_transformer = max_patches;
        var dt = mlx.DtypeTrace.begin("siglip-vision", h, if (self.layers.len > 0) self.layers[0].q_proj.weight else null);
        for (self.layers, 0..) |lw, layer_idx| {
            h = try self.transformerLayer(h, lw, positions, attn_mask_opt, batch, seq_len_for_transformer, num_heads, head_dim);
            dt.layer(h, layer_idx);
        }
        dt.end(h);

        // 6. Slice back to real patches before pooling. The padded rows are zero (masked in
        //    attention throughout), and our positionPool below operates on the real grid;
        //    this avoids a second, more complex einsum-style pool path.
        if (num_padding > 0) {
            const s_start = [_]c_int{ 0, 0, 0 };
            const s_stop = [_]c_int{ batch, num_patches, hidden };
            const s_strides = [_]c_int{ 1, 1, 1 };
            var h_real = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&h_real, h, &s_start, 3, &s_stop, 3, &s_strides, 3, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_real;
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
        // (done inside VisionPooler in the Python reference; we do it right after pooling).
        const root_hidden = bf16Scalar(@sqrt(@as(f32, @floatFromInt(cfg.vision_hidden_size))), self.s);
        defer _ = mlx.mlx_array_free(root_hidden);
        var pooled_scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&pooled_scaled, pooled, root_hidden, self.s));
        _ = mlx.mlx_array_free(pooled);

        // 5b. Gemma-4 26B/31B: learned standardization applied AFTER the sqrt(hidden) scaling.
        //     Reference (mlx-vlm vision.py:502): `hidden_states = (hidden_states - std_bias) * std_scale`.
        //     Verified against dumped weights: std_bias holds large (~train mean) values while
        //     std_scale is tiny (~1/train_std), confirming the "subtract-then-scale" formula.
        var post_std: mlx.mlx_array = pooled_scaled;
        if (self.std_scale) |scale| {
            const bias = self.std_bias orelse unreachable;
            var centered = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_subtract(&centered, post_std, bias, self.s));
            _ = mlx.mlx_array_free(post_std);
            var scaled_out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&scaled_out, centered, scale, self.s));
            _ = mlx.mlx_array_free(centered);
            post_std = scaled_out;
        }

        // 6. Pre-projection parameter-free RMS norm → embedding projection.
        //    Reference (mlx-vlm gemma4.py:32-34, language.py:18-26):
        //      normed = mx.fast.rms_norm(x, None, eps)   # RMSNormNoScale, no learned weight
        //      return embedding_projection(normed)        # dense linear, no bias
        //    Historical port did projection-then-norm; that's wrong (different magnitudes because
        //    RMS-normalizing a 2560/2816/5376-dim text-space vector is not the same as
        //    normalizing the 1152-dim vision-space vector).
        const pre_proj_normed = try self.rmsNormNoScale(post_std);
        _ = mlx.mlx_array_free(post_std);

        var post_normed: mlx.mlx_array = undefined;
        // .ctx (not ndim) — dense bf16 projector's proj_s is a null-ctx handle.
        if (self.proj_s.ctx != null) {
            post_normed = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_quantized_matmul(
                &post_normed, pre_proj_normed, self.proj_w, self.proj_s, self.proj_b,
                true, mlx.mlx_optional_int.some(@intCast(self.proj_quant_group_size)),
                mlx.mlx_optional_int.some(@intCast(self.proj_quant_bits)), self.config.quant_mode.cstr(), self.s,
            ));
        } else {
            var proj_wt = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(proj_wt);
            try mlx.check(mlx.mlx_transpose(&proj_wt, self.proj_w, self.s));
            post_normed = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_matmul(&post_normed, pre_proj_normed, proj_wt, self.s));
        }
        _ = mlx.mlx_array_free(pre_proj_normed);

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

    /// Like `buildPositionGrid`, but pads the tail with (-1, -1) positions up to `max_patches`.
    /// Matches reference vision.py:432 — padding coords are -1.
    fn buildPositionGridPadded(self: *VisionEncoder, batch: c_int, grid_h: c_int, grid_w: c_int, num_patches: c_int, max_patches: c_int) !mlx.mlx_array {
        _ = self;
        _ = batch;
        _ = grid_h;
        const n_real: usize = @intCast(num_patches);
        const n_total: usize = @intCast(max_patches);
        const gw: usize = @intCast(grid_w);
        var pos_data = try std.heap.page_allocator.alloc(i32, n_total * 2);
        defer std.heap.page_allocator.free(pos_data);
        for (0..n_real) |i| {
            pos_data[i * 2] = @intCast(i % gw);
            pos_data[i * 2 + 1] = @intCast(i / gw);
        }
        for (n_real..n_total) |i| {
            pos_data[i * 2] = -1;
            pos_data[i * 2 + 1] = -1;
        }
        const shape = [_]c_int{ 1, max_patches, 2 };
        return mlx.mlx_array_new_data(pos_data.ptr, &shape, 3, .int32);
    }

    /// Build an attention mask [1, 1, 1, L] with 0 for valid keys (j < num_real) and -inf for
    /// padding keys. Broadcasts across heads AND queries — each query row then has a non-empty
    /// set of valid keys, so softmax never produces NaN (a symmetric Q×K mask would make
    /// padding-row softmax sum to zero → NaN, which corrupts the slice-back into real tokens).
    /// Padding queries' output becomes a weighted sum over valid keys; it's ignored after the
    /// slice-back to real patches, so the values don't matter.
    fn buildAttentionMask(self: *VisionEncoder, batch: c_int, max_patches: c_int, num_real: c_int) !mlx.mlx_array {
        _ = batch;
        const L: usize = @intCast(max_patches);
        const R: usize = @intCast(num_real);
        var mask_data = try std.heap.page_allocator.alloc(f32, L);
        defer std.heap.page_allocator.free(mask_data);
        const neg_inf: f32 = -std.math.inf(f32);
        for (0..L) |j| {
            mask_data[j] = if (j < R) 0.0 else neg_inf;
        }
        const shape = [_]c_int{ 1, 1, 1, max_patches };
        const arr = mlx.mlx_array_new_data(mask_data.ptr, &shape, 4, .float32);
        defer _ = mlx.mlx_array_free(arr);
        var bf = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&bf, arr, .bfloat16, self.s));
        return bf;
    }

    // ── Transformer Layer (correct residual pattern) ──

    fn transformerLayer(
        self: *VisionEncoder,
        input: mlx.mlx_array,
        lw: VisionLayerWeights,
        positions: mlx.mlx_array,
        mask: ?mlx.mlx_array,
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

            const attn_out = try self.selfAttention(normed, lw, positions, mask, batch, seq_len, num_heads, head_dim);
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
        mask: ?mlx.mlx_array,
        batch: c_int,
        seq_len: c_int,
        num_heads: c_int,
        head_dim: c_int,
    ) !mlx.mlx_array {
        // Q, K, V projections
        const q_raw = try self.linearForward(x, lw.q_proj);
        defer _ = mlx.mlx_array_free(q_raw);
        const k_raw = try self.linearForward(x, lw.k_proj);
        defer _ = mlx.mlx_array_free(k_raw);
        const v_raw = try self.linearForward(x, lw.v_proj);
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

        // Scaled dot-product attention (scale=1.0, bidirectional). Reference uses scale=1.0 because QK-norm
        // already normalizes.
        var attn_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_out);
        if (mask == null) {
            // An unpadded grid (EmbeddingGemma 2) has nothing to mask, and the fused kernel never builds the
            // [heads, L, L] score matrix: at the largest budget (10080 patches) that is 2.4 GB per layer in bf16.
            const no_mask = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(no_mask);
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_t, k_t, v_t, 1.0, "", no_mask, .{ .ctx = null }, false, self.s));
        } else {
            const kt_perm = [_]c_int{ 0, 1, 3, 2 };
            var k_tp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_tp);
            try mlx.check(mlx.mlx_transpose_axes(&k_tp, k_t, &kt_perm, 4, self.s));

            var attn_weights = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_weights);
            try mlx.check(mlx.mlx_matmul(&attn_weights, q_t, k_tp, self.s));

            // Padding keys carry -inf; the mask is [1, 1, 1, L] and broadcasts along heads and queries.
            var masked = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(masked);
            try mlx.check(mlx.mlx_add(&masked, attn_weights, mask.?, self.s));

            var attn_probs = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn_probs);
            try mlx.check(mlx.mlx_softmax_axis(&attn_probs, masked, -1, false, self.s));
            try mlx.check(mlx.mlx_matmul(&attn_out, attn_probs, v_t, self.s));
        }

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

        return self.linearForward(attn_flat, lw.out_proj);
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
        const gate = try self.linearForward(x, lw.gate_proj);
        defer _ = mlx.mlx_array_free(gate);

        const gate_act = try self.geluApprox(gate);
        defer _ = mlx.mlx_array_free(gate_act);

        const up = try self.linearForward(x, lw.up_proj);
        defer _ = mlx.mlx_array_free(up);

        var gated = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gated);
        try mlx.check(mlx.mlx_multiply(&gated, gate_act, up, self.s));

        return self.linearForward(gated, lw.down_proj);
    }

    // ── Core Operations ──

    fn linearForward(self: *VisionEncoder, x: mlx.mlx_array, cl: LinearWeight) !mlx.mlx_array {
        var w_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w_t);
        try mlx.check(mlx.mlx_transpose(&w_t, cl.weight, self.s));

        if (!cl.has_clip) {
            // Plain linear: matmul(x, w.T)
            var result = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_matmul(&result, x, w_t, self.s));
            return result;
        }

        // Clipped linear: clip(x) → matmul → clip(y)
        var clipped_in = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(clipped_in);
        try mlx.check(mlx.mlx_maximum(&clipped_in, x, cl.input_min, self.s));
        var clipped_in2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(clipped_in2);
        try mlx.check(mlx.mlx_minimum(&clipped_in2, clipped_in, cl.input_max, self.s));

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

// ── Soft-token-budget processor (EmbeddingGemma 2) ──

pub const Resized = qwen_vision.Resized;

/// `Gemma4ImageProcessor`'s resize target: the largest size with the image's aspect ratio whose patch grid
/// fits `max_tokens` pooled soft tokens and whose sides are multiples of `merge * patch`. A source thinner than
/// one multiple takes one multiple on its short side. Null for a degenerate request.
pub fn budgetSize(height: u32, width: u32, patch: u32, merge: u32, max_tokens: u32) ?Resized {
    if (height == 0 or width == 0 or patch == 0 or merge == 0 or max_tokens == 0) return null;
    const side: u64 = @as(u64, merge) * patch;
    const max_patches: u64 = @as(u64, max_tokens) * merge * merge;
    const target_px: f64 = @floatFromInt(max_patches * patch * patch);
    const factor = @sqrt(target_px / (@as(f64, @floatFromInt(height)) * @as(f64, @floatFromInt(width))));
    const side_f: f64 = @floatFromInt(side);
    var h: u64 = @as(u64, @intFromFloat(@floor(factor * @as(f64, @floatFromInt(height)) / side_f))) * side;
    var w: u64 = @as(u64, @intFromFloat(@floor(factor * @as(f64, @floatFromInt(width)) / side_f))) * side;
    if (h == 0 and w == 0) return null;
    const max_side: u64 = (max_patches / (@as(u64, merge) * merge)) * side;
    if (h == 0) {
        h = side;
        w = @min(@as(u64, width / height) * side, max_side);
    } else if (w == 0) {
        w = side;
        h = @min(@as(u64, height / width) * side, max_side);
    }
    if (@as(f64, @floatFromInt(h * w)) > target_px) return null;
    return .{ .h = @intCast(h), .w = @intCast(w) };
}

/// The i-th kept frame when a client sends `n` frames and the processor keeps `max`: `np.linspace(0, n-1, max,
/// dtype=int)`, i.e. an even spread that always keeps the first and last frame.
pub fn sampledFrame(i: usize, n: usize, max: usize) usize {
    if (n <= max) return i;
    if (max < 2) return 0;
    return i * (n - 1) / (max - 1);
}

/// A resized image as the tower reads it: float32 CHW in [0, 1], as bytes.
pub const UnitImage = struct { pixels: []u8, h: u32, w: u32 };

/// Bicubic antialiased resize (Pillow's, which torchvision's uint8 kernel reproduces to within one level) of
/// interleaved RGB8 to `[3, th, tw]` float32 in [0, 1]. Caller frees `pixels`.
pub fn resizeToUnitChw(allocator: std.mem.Allocator, rgb: []const u8, sh: u32, sw: u32, th: u32, tw: u32) !UnitImage {
    const resized: ?[]u8 = if (th == sh and tw == sw) null else try qwen_vision.resizeInterleavedPil(allocator, rgb, sh, sw, th, tw, 3, .bicubic);
    defer if (resized) |r| allocator.free(r);
    const src = resized orelse rgb;
    const plane: usize = @as(usize, th) * tw;
    if (src.len != plane * 3) return error.InvalidImageBuffer;
    const bytes = try allocator.alloc(u8, plane * 3 * @sizeOf(f32));
    const out = @as([*]f32, @ptrCast(@alignCast(bytes.ptr)))[0 .. plane * 3];
    // The processor multiplies by the float32 of 1/255 rather than dividing.
    const rescale: f32 = 1.0 / 255.0;
    for (0..plane) |i| inline for (0..3) |c| {
        out[c * plane + i] = @as(f32, @floatFromInt(src[i * 3 + c])) * rescale;
    };
    return .{ .pixels = bytes, .h = th, .w = tw };
}

/// `Gemma4ImageProcessor` on one decoded image: resize to the budget's target size and rescale.
pub fn budgetPixels(allocator: std.mem.Allocator, rgb: []const u8, h: u32, w: u32, patch: u32, merge: u32, max_tokens: u32) !UnitImage {
    const size = budgetSize(h, w, patch, merge, max_tokens) orelse return error.InvalidImageDimensions;
    return resizeToUnitChw(allocator, rgb, h, w, size.h, size.w);
}

// ── Weight Loading Helpers ──

fn getVisionWeight(weights: *const Weights, buf: *[256]u8, layer: usize, suffix: []const u8) error{MissingWeight}!mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "vision_tower.encoder.layers.{d}.{s}", .{ layer, suffix }) catch unreachable;
    return weights.get(name) orelse {
        log.err("MISSING VISION WEIGHT: {s}\n", .{name});
        return error.MissingWeight;
    };
}

fn getWeight(weights: *const Weights, buf: *[256]u8, name: []const u8) ?mlx.mlx_array {
    const n = std.fmt.bufPrint(buf, "{s}", .{name}) catch unreachable;
    return weights.get(n);
}

fn loadLinearWeight(weights: *const Weights, buf: *[256]u8, layer: usize, prefix: []const u8, clipped: bool) error{MissingWeight}!LinearWeight {
    const weight = try getClippedWeight(weights, buf, layer, prefix, ".linear.weight");
    if (clipped) {
        return .{
            .weight = weight,
            .has_clip = true,
            .input_min = try getClippedWeight(weights, buf, layer, prefix, ".input_min"),
            .input_max = try getClippedWeight(weights, buf, layer, prefix, ".input_max"),
            .output_min = try getClippedWeight(weights, buf, layer, prefix, ".output_min"),
            .output_max = try getClippedWeight(weights, buf, layer, prefix, ".output_max"),
        };
    }
    return .{
        .weight = weight,
        .has_clip = false,
        .input_min = mlx.mlx_array_new(),
        .input_max = mlx.mlx_array_new(),
        .output_min = mlx.mlx_array_new(),
        .output_max = mlx.mlx_array_new(),
    };
}

fn getClippedWeight(weights: *const Weights, buf: *[256]u8, layer: usize, prefix: []const u8, suffix: []const u8) error{MissingWeight}!mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "vision_tower.encoder.layers.{d}.{s}{s}", .{ layer, prefix, suffix }) catch unreachable;
    return weights.get(name) orelse {
        log.err("MISSING VISION WEIGHT: {s}\n", .{name});
        return error.MissingWeight;
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
    // A dense bf16 projector has null-ctx scales; mlx_array_ndim throws on a
    // null handle, so short-circuit first. The returned bits are unused in that
    // case (quantLinear takes the dense matmul path), so any value is fine.
    if (sc.ctx == null or mlx.mlx_array_ndim(sc) < 2) return 4;
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

test "unified patchify produces [1, gh*gw, 3*P*P] model patches" {
    // Gemma 4 12B patchifies directly at the 48px model-patch size. Verify the
    // reshape arithmetic: a 96×48 image at P=48 → gh=2, gw=1 → 2 patches of
    // 3*48*48 = 6912 dims each. (Byte-level ordering vs the reference 16px+merge
    // pipeline is proven by the Python equivalence harness.)
    var enc: VisionEncoder = undefined;
    enc.s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(enc.s);

    const P: c_int = 48;
    const pix_shape = [_]c_int{ 1, 3, 96, 48 };
    var pixels = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pixels);
    try mlx.check(mlx.mlx_zeros(&pixels, &pix_shape, 4, .float32, enc.s));

    const out = try enc.patchify(pixels, 1, 2, 1, P);
    defer _ = mlx.mlx_array_free(out);
    const shape = mlx.getShape(out);
    try testing.expectEqual(@as(usize, 3), shape.len);
    try testing.expectEqual(@as(c_int, 1), shape[0]);
    try testing.expectEqual(@as(c_int, 2), shape[1]); // gh*gw
    try testing.expectEqual(@as(c_int, 3 * 48 * 48), shape[2]); // 6912
}

test "detectProjBits returns dense default for null-ctx scales" {
    // A fully-dense bf16 multimodal projector (e.g. gemma-4-E2B-it-qat-bf16)
    // ships `embed_vision.embedding_projection.weight` with NO `.scales`, so the
    // scales handle is a null-ctx `mlx_array_new()`. mlx-c 0.6.0's
    // `mlx_array_ndim` THROWS "expected a non-empty mlx_array" on a null-ctx
    // array (it does not return 0), which used to abort `VisionEncoder.init`.
    // detectProjBits must short-circuit on the null handle and return the dense
    // default (4) — the value is unused because quantLinear takes its dense path.
    const dense_scales = mlx.mlx_array_new(); // null-ctx
    const w = mlx.mlx_array_new(); // weight irrelevant on the dense short-circuit
    try testing.expectEqual(@as(u32, 4), detectProjBits(w, dense_scales, 64));
}

test "quantLinear dense fallback computes x @ Wᵀ for a bf16 projector" {
    // The dense projector path (scales absent) must transpose the [out, in]
    // weight and matmul — identical x @ Wᵀ semantics to the quantized
    // transpose=true path. Guards both the null-ctx ndim short-circuit (line
    // 328) and the matmul orientation. x[1,2] @ W[3,2]ᵀ = [1,3].
    var enc: VisionEncoder = undefined;
    enc.s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(enc.s);
    enc.proj_quant_group_size = 64; // unused on the dense branch

    var x_buf = [_]f32{ 1.0, 2.0 };
    const x_shape = [_]c_int{ 1, 2 };
    const x = mlx.mlx_array_new_data(&x_buf, &x_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(x);

    // W is [out=3, in=2]: rows [1,0],[0,1],[1,1].
    var w_buf = [_]f32{ 1.0, 0.0, 0.0, 1.0, 1.0, 1.0 };
    const w_shape = [_]c_int{ 3, 2 };
    const w = mlx.mlx_array_new_data(&w_buf, &w_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(w);

    const out = try enc.quantLinear(x, w, mlx.mlx_array_new(), mlx.mlx_array_new(), 4, null);
    defer _ = mlx.mlx_array_free(out);

    const oshape = mlx.getShape(out);
    try testing.expectEqual(@as(usize, 2), oshape.len);
    try testing.expectEqual(@as(c_int, 1), oshape[0]);
    try testing.expectEqual(@as(c_int, 3), oshape[1]);

    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    const fshape = [_]c_int{3};
    try mlx.check(mlx.mlx_reshape(&flat, out, &fshape, 1, enc.s));
    const ev = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ev);
    _ = mlx.mlx_vector_array_append_value(ev, flat);
    try mlx.check(mlx.mlx_eval(ev));
    const ptr = mlx.mlx_array_data_float32(flat) orelse return error.NullData;
    try testing.expectApproxEqAbs(@as(f32, 1.0), ptr[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 2.0), ptr[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 3.0), ptr[2], 1e-4);
}

test "EmbeddingCache keeps what the current request uses: a history past the cap still hits" {
    // Every turn reads the whole history in the same order; plain LRU on that
    // scan evicts exactly the entry the next turn needs first, so nothing hits.
    var cache: EmbeddingCache = .{};
    defer cache.deinit(testing.allocator);
    var buf = [_]f32{ 1, 2, 3, 4 };
    const shape = [_]c_int{ 1, 4 };
    var arrays: [3]mlx.mlx_array = undefined;
    for (&arrays) |*a| a.* = mlx.mlx_array_new_data(&buf, &shape, 2, .float32);
    defer for (arrays) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const cap = 2 * 16; // two 16-byte entries, a three-item history
    for (0..3) |turn| {
        const epoch = cache.beginRequest();
        var hits: usize = 0;
        for (arrays, 0..) |a, key| {
            if (cache.get(key)) |hit| {
                _ = mlx.mlx_array_free(hit);
                hits += 1;
            } else cache.put(testing.allocator, key, a, cap, epoch);
        }
        try testing.expectEqual(@as(usize, if (turn == 0) 0 else 2), hits);
    }
    const kept = cache.get(0) orelse return error.Miss;
    defer _ = mlx.mlx_array_free(kept);
    try testing.expectEqual(@as(f32, 3), mlx.mlx_array_data_float32(kept).?[2]);
    try testing.expectEqual(@as(usize, cap), cache.bytes);
}

// Reference: `get_aspect_ratio_preserving_size` from transformers (patch 16, pooling kernel 3), rows of
// { source h, source w, soft-token budget, target h, target w }.
const SIZE_CASES = [_][5]u32{
    .{ 5, 9, 70, 288, 528 },         .{ 16, 2000, 70, 48, 3360 },     .{ 2000, 16, 70, 3360, 48 },       .{ 48, 48, 70, 384, 384 },
    .{ 47, 49, 70, 384, 384 },       .{ 1, 1, 70, 384, 384 },         .{ 300, 500, 70, 288, 480 },       .{ 200, 360, 70, 288, 528 },
    .{ 1080, 1920, 70, 288, 528 },   .{ 3000, 4000, 70, 336, 432 },   .{ 50, 70, 70, 336, 432 },         .{ 16, 20000, 70, 48, 3360 },
    .{ 20000, 16, 70, 3360, 48 },    .{ 2, 3, 70, 288, 480 },         .{ 5, 9, 140, 384, 720 },          .{ 16, 2000, 140, 48, 6336 },
    .{ 47, 49, 140, 528, 576 },      .{ 300, 500, 140, 432, 720 },    .{ 200, 360, 140, 384, 720 },      .{ 3000, 4000, 140, 480, 624 },
    .{ 5, 9, 280, 576, 1056 },       .{ 16, 2000, 280, 48, 8976 },    .{ 2000, 16, 280, 8976, 48 },      .{ 48, 48, 280, 768, 768 },
    .{ 47, 49, 280, 768, 816 },      .{ 1, 1, 280, 768, 768 },        .{ 300, 500, 280, 576, 1008 },     .{ 200, 360, 280, 576, 1056 },
    .{ 1080, 1920, 280, 576, 1056 }, .{ 3000, 4000, 280, 672, 912 },  .{ 16, 20000, 280, 48, 13440 },    .{ 20000, 16, 280, 13440, 48 },
    .{ 2, 3, 280, 624, 960 },        .{ 5, 9, 560, 816, 1488 },       .{ 16, 2000, 560, 96, 12672 },     .{ 300, 500, 560, 864, 1440 },
    .{ 3000, 4000, 560, 960, 1296 }, .{ 16, 20000, 560, 48, 26880 },  .{ 5, 9, 1120, 1152, 2112 },       .{ 16, 2000, 1120, 96, 17952 },
    .{ 47, 49, 1120, 1536, 1632 },   .{ 300, 500, 1120, 1200, 2064 }, .{ 1080, 1920, 1120, 1200, 2112 }, .{ 16, 20000, 1120, 48, 53760 },
};

test "budgetSize: the largest aspect-preserving size that fits the soft-token budget, as the processor computes it" {
    for (SIZE_CASES) |c| {
        const got = budgetSize(c[0], c[1], 16, 3, c[2]) orelse {
            std.debug.print("{d}x{d} @ {d}: refused\n", .{ c[0], c[1], c[2] });
            return error.Refused;
        };
        testing.expect(got.h == c[3] and got.w == c[4]) catch |e| {
            std.debug.print("{d}x{d} @ {d}: got {d}x{d}, want {d}x{d}\n", .{ c[0], c[1], c[2], got.h, got.w, c[3], c[4] });
            return e;
        };
        // Whatever the shape, the grid pools evenly and fits the budget.
        try testing.expect(got.h % 48 == 0 and got.w % 48 == 0);
        try testing.expect((got.h / 16) * (got.w / 16) <= c[2] * 9);
    }
    try testing.expect(budgetSize(0, 10, 16, 3, 280) == null);
    try testing.expect(budgetSize(10, 0, 16, 3, 280) == null);
}

test "sampledFrame: more frames than the processor keeps are cut to an even spread, first and last included" {
    // np.linspace(0, n - 1, 32, dtype=int) for n = 33, 40, 100.
    const want_33 = [_]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32 };
    const want_40 = [_]usize{ 0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28, 30, 31, 32, 33, 35, 36, 37, 39 };
    const want_100 = [_]usize{ 0, 3, 6, 9, 12, 15, 19, 22, 25, 28, 31, 35, 38, 41, 44, 47, 51, 54, 57, 60, 63, 67, 70, 73, 76, 79, 83, 86, 89, 92, 95, 99 };
    inline for (.{ .{ 33, want_33 }, .{ 40, want_40 }, .{ 100, want_100 } }) |c| {
        for (c[1], 0..) |want, i| try testing.expectEqual(want, sampledFrame(i, c[0], 32));
    }
    // Within the cap every frame is kept, in order; a single kept frame is the first.
    for (0..20) |i| try testing.expectEqual(i, sampledFrame(i, 20, 32));
    try testing.expectEqual(@as(usize, 0), sampledFrame(0, 50, 1));
}

/// The closed-form source image of the fixture's resize cases.
fn patternRgb(allocator: std.mem.Allocator, h: usize, w: usize) ![]u8 {
    const out = try allocator.alloc(u8, h * w * 3);
    for (0..h) |y| for (0..w) |x| for (0..3) |c| {
        out[(y * w + x) * 3 + c] = @intCast((x * 7 + y * 13 + c * 61 + ((x * y) % 29) * 5) % 256);
    };
    return out;
}

test "resizeToUnitChw matches torchvision's bicubic antialias on the reference's own cases" {
    const a = testing.allocator;
    const Case = struct { src: [2]u32, dst: [2]u32, rgb: []const u8 };
    const Expected = struct { media: struct { resize: []const Case } };
    const parsed = try std.json.parseFromSlice(Expected, a, @embedFile("fixtures/embedding_gemma2_tiny_expected.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    for (parsed.value.media.resize, 0..) |c, case| {
        const src = try patternRgb(a, c.src[0], c.src[1]);
        defer a.free(src);
        const img = try resizeToUnitChw(a, src, c.src[0], c.src[1], c.dst[0], c.dst[1]);
        defer a.free(img.pixels);
        try testing.expectEqual(c.dst[0], img.h);
        try testing.expectEqual(c.dst[1], img.w);

        const want = try a.alloc(u8, c.dst[0] * c.dst[1] * 3);
        defer a.free(want);
        try std.base64.standard.Decoder.decode(want, c.rgb);
        const got: []const f32 = @as([*]const f32, @ptrCast(@alignCast(img.pixels.ptr)))[0..want.len];
        const plane = @as(usize, c.dst[0]) * c.dst[1];
        var off_by_one: usize = 0;
        var worst: u32 = 0;
        for (0..plane) |i| for (0..3) |ch| {
            const level: i32 = @intFromFloat(@round(got[ch * plane + i] * 255.0));
            const diff = @abs(level - @as(i32, want[i * 3 + ch]));
            worst = @max(worst, diff);
            off_by_one += @intFromBool(diff == 1);
        };
        // Torchvision's integer kernel and Pillow's differ in the last bit now and then, never more.
        testing.expect(worst <= 1 and off_by_one * 20 <= plane * 3) catch |e| {
            std.debug.print("case {d}: worst {d} levels, {d} of {d} values off by one\n", .{ case, worst, off_by_one, plane * 3 });
            return e;
        };
    }
}
