const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const tokenizer_mod = @import("tokenizer.zig");

pub const HiddenAct = enum { gelu_approx, silu, relu_sq };

/// MLX quantization mode from config.json's `quantization.mode`. All
/// non-affine modes store NO `.biases` tensors (per-group fp8-encoded uint8
/// scales only) but share the packed-u32 weight layout, so supporting them is
/// a matter of skipping the biases fetch and passing the right mode string to
/// the mlx quantized ops. Tag names match the mlx-c mode strings exactly.
pub const QuantMode = enum {
    affine,
    nvfp4,
    mxfp4,
    mxfp8,

    pub fn fromString(name: []const u8) ?QuantMode {
        return std.meta.stringToEnum(QuantMode, name);
    }

    /// Mode string for mlx_quantized_matmul / mlx_gather_qmm / mlx_dequantize.
    pub fn cstr(self: QuantMode) [*:0]const u8 {
        return switch (self) {
            .affine => "affine",
            .nvfp4 => "nvfp4",
            .mxfp4 => "mxfp4",
            .mxfp8 => "mxfp8",
        };
    }

    /// Affine is the only mode whose checkpoints carry per-group biases.
    pub fn hasBiases(self: QuantMode) bool {
        return self == .affine;
    }
};

pub const LayerBlockType = enum { attention, gated_conv, mamba2, mlp, moe };

pub const ModelConfig = struct {
    // Architecture identity
    model_type: []const u8 = "gemma3",
    weight_prefix: []const u8 = "language_model.model",

    // Core dimensions
    vocab_size: u32 = 262208,
    hidden_size: u32 = 3840,
    intermediate_size: u32 = 15360,
    num_hidden_layers: u32 = 48,
    num_attention_heads: u32 = 16,
    num_key_value_heads: u32 = 8,
    head_dim: u32 = 256,
    rms_norm_eps: f32 = 1e-6,

    // RoPE
    rope_theta: f32 = 1000000.0,
    rope_local_base_freq: f32 = 10000.0,
    rope_scaling_factor: f32 = 1.0,
    rope_proportional: bool = false, // Gemma 4: full attention uses proportional RoPE
    rope_proportional_factor: f32 = 1.0,

    // Sliding window attention
    has_sliding_window: bool = true,
    sliding_window: u32 = 1024,
    sliding_window_pattern: u32 = 6,

    // Quantization. 0 = dense bf16 (config.json has no "quantization" key);
    // quantized checkpoints always set this from that key (see parseConfig).
    quant_bits: u32 = 0,
    quant_group_size: u32 = 64,
    quant_mode: QuantMode = .affine,

    // Attention scale: 1/sqrt(query_pre_attn_scalar) for Gemma, 1/sqrt(head_dim) for others
    query_pre_attn_scalar: u32 = 256,

    // Architectural differences between model families
    tie_word_embeddings: bool = false,
    hidden_act: HiddenAct = .gelu_approx,
    norm_has_offset: bool = true,
    scale_embeddings: bool = true,
    has_pre_ff_norm: bool = true,
    has_qk_norm: bool = true,

    // MoE
    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    moe_intermediate_size: u32 = 0,
    shared_expert_intermediate_size: u32 = 0,

    // Linear attention (GatedDeltaNet)
    linear_num_key_heads: u32 = 0,
    linear_num_value_heads: u32 = 0,
    linear_key_head_dim: u32 = 128,
    linear_value_head_dim: u32 = 128,
    linear_conv_kernel_dim: u32 = 4,

    // Hybrid attention
    full_attention_interval: u32 = 0,
    partial_rotary_factor: f32 = 1.0,
    attn_output_gate: bool = false,

    // BERT encoder-only
    is_encoder_only: bool = false,
    layer_norm_eps: f32 = 1e-12,
    type_vocab_size: u32 = 0,

    // Context length from config.json (0 = unknown)
    max_position_embeddings: u32 = 0,

    // Stop tokens (populated from config.json)
    eos_token_ids: [8]u32 = .{0} ** 8,
    num_eos_tokens: u32 = 0,

    // Model-author sampling recommendations from generation_config.json
    // (e.g. Qwen 3.6: temp 1.0 / top_p 0.95 / top_k 20; Gemma 4: top_k 64).
    // null = the file or key is absent. Used as defaults for request fields
    // the client OMITTED — Claude Code sends no sampling params at all, and
    // pre-2026-06 it sampled the full untruncated distribution at temp 1.0,
    // well outside the model card's intended envelope.
    gen_temperature: ?f32 = null,
    gen_top_p: ?f32 = null,
    gen_top_k: ?u32 = null,

    // Gemma 4: explicit layer type map (bit = 1 means full/global attention)
    has_explicit_layer_types: bool = false,
    layer_is_global: [128]bool = .{false} ** 128,

    // Vision encoder (Gemma 4 SigLIP)
    has_vision: bool = false,
    vision_hidden_size: u32 = 768,
    vision_num_layers: u32 = 16,
    vision_num_heads: u32 = 12,
    vision_head_dim: u32 = 64,
    vision_intermediate_size: u32 = 3072,
    vision_patch_size: u32 = 16,
    vision_pooling_kernel: u32 = 3,
    vision_soft_tokens: u32 = 280,
    vision_position_embedding_size: u32 = 10240,
    vision_rope_theta: f32 = 100.0,
    vision_use_clipped_linears: bool = true,
    image_token_id: u32 = 0, // 0 = no image token
    boi_token_id: u32 = 0, // beginning of image
    eoi_token_id: u32 = 0, // end of image

    // Gemma 4 12B "unified" (encoder-free) multimodal. Instead of the SigLIP
    // transformer tower, vision is a single patch embedder
    // (LN → Dense → LN → +factorized 2D posemb → LN → RMSNorm → Linear) and
    // audio is raw 640-sample frames projected straight to text space. Set
    // when model_type is gemma4_unified*. See src/vision.zig (UnifiedEmbedder).
    is_gemma4_unified: bool = false,
    vision_mm_embed_dim: u32 = 0, // unified: mm_embed_dim (3840 for 12B = text hidden)
    vision_model_patch_size: u32 = 0, // unified: 48px merged "model patch" (16px teacher × 3 pool)
    vision_mm_posemb_size: u32 = 0, // unified: factorized position table size per axis (1120)
    // Audio (gemma4_unified). embed_audio projects audio_embed_dim → text hidden.
    audio_token_id: u32 = 0, // 0 = no audio token
    boa_token_id: u32 = 0, // beginning of audio
    eoa_token_id: u32 = 0, // end of audio
    audio_embed_dim: u32 = 0, // unified: raw samples per token (640)
    audio_samples_per_token: u32 = 640, // 40ms @ 16kHz

    // Token IDs that mark the start of a user turn in the rendered prompt.
    // Populated at startup by encoding a chat-template-specific prefix string
    // (e.g. "<|turn>user\n" for Gemma 4, "<|im_start|>user\n" for Qwen ChatML).
    // Used by insertImageTokens to locate the latest user turn — a hard-coded
    // ID search would silently break across architectures and quantizations.
    user_turn_marker_ids: [16]u32 = .{0} ** 16,
    user_turn_marker_len: u8 = 0,

    // Gemma 4: dual head dimensions and KV sharing
    global_head_dim: u32 = 0, // 0 = same as head_dim
    num_global_key_value_heads: u32 = 0, // 0 = same as num_key_value_heads
    num_kv_shared_layers: u32 = 0,
    final_logit_softcapping: f32 = 0.0, // 0 = disabled
    hidden_size_per_layer_input: u32 = 0, // >0 enables PLE
    partial_rotary_factor_global: f32 = 1.0, // for global/full attention layers
    has_v_norm: bool = false, // parameter-free RMS norm on values
    // Gemma 4 (31B): full_attention layers share V with K (no v_proj stored)
    attention_k_eq_v: bool = false,

    // Block diffusion (DiffusionGemma). canvas_length > 0 marks a diffusion
    // checkpoint: generation runs the canvas-denoising loop in
    // src/diffusion.zig instead of autoregressive decode. The knobs mirror
    // the checkpoint's embedded `generation_config` object; defaults match
    // google/diffusiongemma-26B-A4B-it.
    canvas_length: u32 = 0,
    diffusion_max_steps: u32 = 48,
    diffusion_t_min: f32 = 0.4,
    diffusion_t_max: f32 = 0.8,
    diffusion_entropy_bound: f32 = 0.1,
    diffusion_confidence_threshold: f32 = 0.005,
    diffusion_stability_threshold: u32 = 1,
    diffusion_pad_token: u32 = 0,

    // Hybrid layers (LFM2, Nemotron-H): per-layer type dispatch
    has_hybrid_layers: bool = false,
    layer_block_types: [128]LayerBlockType = .{.attention} ** 128,
    has_embedding_norm: bool = false, // LFM2: RMS norm applied to embeddings
    has_final_norm: bool = true, // false for LFM2 (no model.norm.weight)

    // LFM2 gated convolution
    lfm_conv_kernel: u32 = 3,
    lfm_conv_dim: u32 = 0, // 0 = hidden_size

    // Mamba2 SSM (Nemotron-H)
    mamba_num_heads: u32 = 0,
    mamba_head_dim: u32 = 0,
    mamba_n_groups: u32 = 8,
    ssm_state_size: u32 = 128,
    mamba_conv_kernel: u32 = 4,
    mamba_expand: u32 = 2,
    time_step_min: f32 = 0.0,
    time_step_max: f32 = std.math.inf(f32),
    mamba_chunk_size: u32 = 256,
    mamba_mlp_act: HiddenAct = .relu_sq, // Nemotron-H MLP uses ReLU^2

    pub fn isGlobalLayer(self: ModelConfig, layer_idx: u32) bool {
        if (!self.has_sliding_window) return true;
        if (self.has_explicit_layer_types and layer_idx < 128) {
            return self.layer_is_global[layer_idx];
        }
        // HF/mlx-lm convention (Gemma 3): the GLOBAL layer closes each group —
        // global when `(idx + 1) % pattern == 0` (layers 5, 11, … for pattern 6).
        return (layer_idx % self.sliding_window_pattern) == self.sliding_window_pattern - 1;
    }

    /// For Gemma 4 KV sharing: get the source layer index for a shared layer.
    /// Returns null if the layer computes its own KV (not shared).
    pub fn getKVSourceLayer(self: ModelConfig, layer_idx: u32) ?u32 {
        if (self.num_kv_shared_layers == 0) return null;
        const first_shared = self.num_hidden_layers - self.num_kv_shared_layers;
        if (layer_idx < first_shared) return null;
        const is_global = self.isGlobalLayer(layer_idx);
        // Find last concrete layer of the same type (scanning downward)
        var j: u32 = first_shared;
        while (j > 0) {
            j -= 1;
            if (self.isGlobalLayer(j) == is_global) return j;
        }
        return null;
    }

    /// Get effective head_dim for a layer (global layers may use global_head_dim).
    pub fn layerHeadDim(self: ModelConfig, layer_idx: u32) u32 {
        if (self.global_head_dim > 0 and self.isGlobalLayer(layer_idx)) {
            return self.global_head_dim;
        }
        return self.head_dim;
    }

    /// Get effective num_kv_heads for a layer.
    pub fn layerKVHeads(self: ModelConfig, layer_idx: u32) u32 {
        if (self.num_global_key_value_heads > 0 and self.isGlobalLayer(layer_idx)) {
            return self.num_global_key_value_heads;
        }
        return self.num_key_value_heads;
    }

    pub fn isLinearLayer(self: ModelConfig, layer_idx: u32) bool {
        if (self.full_attention_interval == 0) return false;
        return ((layer_idx + 1) % self.full_attention_interval) != 0;
    }

    pub fn isMoe(self: *const ModelConfig) bool {
        return self.num_experts > 0;
    }

    /// Block-diffusion checkpoint (DiffusionGemma): generation is the canvas
    /// denoising loop, not autoregressive decode.
    pub fn isDiffusion(self: *const ModelConfig) bool {
        return self.canvas_length > 0;
    }

    /// True when the trunk uses the Gemma 4 layer structure (dual FFN with
    /// shared-expert branch, sigma-MoE router, 7 norms, layer_scalar, v_norm,
    /// proportional RoPE on full layers). DiffusionGemma reuses the Gemma 4
    /// 26B-A4B decoder verbatim, so transformer.zig's gemma4 forward/binding
    /// paths key on this rather than on the model_type string.
    pub fn isGemma4Layers(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "gemma4") or
            std.mem.eql(u8, self.model_type, "diffusion_gemma");
    }

    pub fn addEosToken(self: *ModelConfig, id: u32) void {
        if (self.num_eos_tokens < self.eos_token_ids.len) {
            self.eos_token_ids[self.num_eos_tokens] = id;
            self.num_eos_tokens += 1;
        }
    }

    pub fn isEosToken(self: *const ModelConfig, id: u32) bool {
        for (self.eos_token_ids[0..self.num_eos_tokens]) |eos| {
            if (id == eos) return true;
        }
        return false;
    }

    pub fn eosTokenSlice(self: *const ModelConfig) []const u32 {
        return self.eos_token_ids[0..self.num_eos_tokens];
    }

    pub fn userTurnMarkerSlice(self: *const ModelConfig) []const u32 {
        return self.user_turn_marker_ids[0..self.user_turn_marker_len];
    }

    /// Encode the architecture-appropriate user-turn prefix and store the IDs
    /// on the config. Selects the prefix by matching marker tokens that appear
    /// in `chat_template`, so a model that ships an unusual template still
    /// gets the right tokenization. No-op (leaves length=0) when no known
    /// pattern matches — insertImageTokens then falls back to its end-anchored
    /// heuristic.
    pub fn populateUserTurnMarker(
        self: *ModelConfig,
        allocator: std.mem.Allocator,
        tok: *const tokenizer_mod.Tokenizer,
        chat_template: []const u8,
    ) !void {
        const prefix = pickUserTurnPrefix(chat_template) orelse return;
        const ids = try tok.encode(allocator, prefix);
        defer allocator.free(ids);
        const cap = self.user_turn_marker_ids.len;
        if (ids.len == 0 or ids.len > cap) {
            log.warn("user turn marker '{s}' encoded to {d} tokens (cap {d}); skipping\n", .{ prefix, ids.len, cap });
            return;
        }
        @memcpy(self.user_turn_marker_ids[0..ids.len], ids);
        self.user_turn_marker_len = @intCast(ids.len);
        log.info("User turn marker: \"{s}\" -> {d} tokens\n", .{ prefix, ids.len });
    }
};

/// Pick the user-turn prefix string for a model based on what its chat template
/// emits at the start of a user turn. Order matters — the more specific Gemma 4
/// `<|turn>` is checked before the older `<start_of_turn>` so a tokenizer that
/// happens to register both still picks the one its template actually uses.
pub fn pickUserTurnPrefix(chat_template: []const u8) ?[]const u8 {
    if (std.mem.indexOf(u8, chat_template, "<|turn>") != null) {
        return "<|turn>user\n"; // Gemma 4
    }
    if (std.mem.indexOf(u8, chat_template, "<start_of_turn>") != null) {
        return "<start_of_turn>user\n"; // Gemma 3
    }
    if (std.mem.indexOf(u8, chat_template, "<|im_start|>") != null) {
        return "<|im_start|>user\n"; // Qwen / generic ChatML
    }
    if (std.mem.indexOf(u8, chat_template, "<|start_header_id|>") != null) {
        return "<|start_header_id|>user<|end_header_id|>\n\n"; // Llama 3
    }
    return null;
}

pub fn parseConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !ModelConfig {
    const path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
    defer allocator.free(path);

    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var reader_state = file.reader(io, &read_buf);
    const content = try reader_state.interface.allocRemaining(allocator, .limited(10 * 1024 * 1024));
    defer allocator.free(content);

    var config = try parseConfigFromJson(allocator, content);

    // Model-author sampling recommendations ride in a sibling file. Optional —
    // any failure (missing file, bad JSON) leaves the fields null.
    const gen_path = try std.fmt.allocPrint(allocator, "{s}/generation_config.json", .{model_dir});
    defer allocator.free(gen_path);
    if (std.Io.Dir.openFileAbsolute(io, gen_path, .{})) |gen_file| {
        defer gen_file.close(io);
        var gen_buf: [4096]u8 = undefined;
        var gen_reader = gen_file.reader(io, &gen_buf);
        if (gen_reader.interface.allocRemaining(allocator, .limited(1024 * 1024))) |gen_content| {
            defer allocator.free(gen_content);
            const gd = parseGenerationDefaultsFromJson(gen_content);
            config.gen_temperature = gd.temperature;
            config.gen_top_p = gd.top_p;
            config.gen_top_k = gd.top_k;
        } else |_| {}
    } else |_| {}

    return config;
}

/// Sampling recommendations parsed out of a model's generation_config.json.
pub const GenerationDefaults = struct {
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    top_k: ?u32 = null,
};

/// Pure parser for generation_config.json content. Total: malformed JSON or
/// out-of-range values yield nulls — a corrupt config must never pin
/// sampling to an extreme. (`do_sample` is deliberately ignored: HF uses it
/// for greedy-vs-sample mode selection, which the request's own temperature
/// already expresses.)
pub fn parseGenerationDefaultsFromJson(content: []const u8) GenerationDefaults {
    var buf: [16 * 1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const parsed = std.json.parseFromSlice(std.json.Value, fba.allocator(), content, .{}) catch return .{};
    defer parsed.deinit();
    if (parsed.value != .object) return .{};
    const root = parsed.value.object;

    var gd = GenerationDefaults{};
    if (root.get("temperature")) |v| {
        const t: ?f32 = switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => null,
        };
        if (t) |tv| {
            if (tv >= 0.0 and tv <= 2.0) gd.temperature = tv;
        }
    }
    if (root.get("top_p")) |v| {
        const p: ?f32 = switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => null,
        };
        if (p) |pv| {
            if (pv > 0.0 and pv <= 1.0) gd.top_p = pv;
        }
    }
    if (root.get("top_k")) |v| {
        switch (v) {
            .integer => |i| if (i > 0 and i <= 1000) {
                gd.top_k = @intCast(i);
            },
            else => {},
        }
    }
    return gd;
}

/// I/O-free variant for unit tests and for callers that already have the
/// config.json bytes in memory. The full I/O-bound `parseConfig` delegates here.
pub fn parseConfigFromJson(allocator: std.mem.Allocator, content: []const u8) !ModelConfig {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();

    const root = parsed.value.object;
    var config = ModelConfig{};

    // Detect model_type from top-level (always present)
    const model_type = if (root.get("model_type")) |v| v.string else "gemma3";

    // Determine which object to read config from: text_config (nested) or root (flat)
    const cfg_obj = if (root.get("text_config")) |tc_val| tc_val.object else root;

    // Parse common fields
    if (cfg_obj.get("vocab_size")) |v| config.vocab_size = @intCast(v.integer);
    if (cfg_obj.get("hidden_size")) |v| config.hidden_size = @intCast(v.integer);
    if (cfg_obj.get("intermediate_size")) |v| config.intermediate_size = @intCast(v.integer);
    if (cfg_obj.get("num_hidden_layers")) |v| config.num_hidden_layers = @intCast(v.integer);
    if (cfg_obj.get("num_attention_heads")) |v| config.num_attention_heads = @intCast(v.integer);
    if (cfg_obj.get("num_key_value_heads")) |v| config.num_key_value_heads = @intCast(v.integer);
    if (cfg_obj.get("head_dim")) |v| config.head_dim = @intCast(v.integer);
    if (cfg_obj.get("max_position_embeddings")) |v| config.max_position_embeddings = @intCast(v.integer);
    if (cfg_obj.get("rms_norm_eps")) |v| config.rms_norm_eps = jsonFloat(v);
    if (cfg_obj.get("rope_theta")) |v| config.rope_theta = jsonFloat(v);
    if (cfg_obj.get("query_pre_attn_scalar")) |v| config.query_pre_attn_scalar = @intCast(v.integer);

    // MoE fields (guard against JSON null values)
    if (cfg_obj.get("num_experts")) |v| { if (v == .integer) config.num_experts = @intCast(v.integer); }
    if (cfg_obj.get("num_experts_per_tok")) |v| { if (v == .integer) config.num_experts_per_tok = @intCast(v.integer); }
    if (cfg_obj.get("top_k_experts")) |v| { if (v == .integer) config.num_experts_per_tok = @intCast(v.integer); }
    if (cfg_obj.get("moe_intermediate_size")) |v| { if (v == .integer) config.moe_intermediate_size = @intCast(v.integer); }
    if (cfg_obj.get("shared_expert_intermediate_size")) |v| { if (v == .integer) config.shared_expert_intermediate_size = @intCast(v.integer); }

    // Linear attention (GatedDeltaNet) fields
    if (cfg_obj.get("linear_num_key_heads")) |v| config.linear_num_key_heads = @intCast(v.integer);
    if (cfg_obj.get("linear_num_value_heads")) |v| config.linear_num_value_heads = @intCast(v.integer);
    if (cfg_obj.get("linear_key_head_dim")) |v| config.linear_key_head_dim = @intCast(v.integer);
    if (cfg_obj.get("linear_value_head_dim")) |v| config.linear_value_head_dim = @intCast(v.integer);
    if (cfg_obj.get("linear_conv_kernel_dim")) |v| config.linear_conv_kernel_dim = @intCast(v.integer);

    // Hybrid attention
    if (cfg_obj.get("full_attention_interval")) |v| config.full_attention_interval = @intCast(v.integer);
    if (cfg_obj.get("attn_output_gate")) |v| {
        if (v == .bool) config.attn_output_gate = v.bool;
    }

    // Rope parameters (nested for Qwen3.5)
    if (cfg_obj.get("rope_parameters")) |rp_val| {
        if (rp_val == .object) {
            if (rp_val.object.get("rope_theta")) |v| config.rope_theta = jsonFloat(v);
            if (rp_val.object.get("partial_rotary_factor")) |v| config.partial_rotary_factor = jsonFloat(v);
        }
    }

    // Sliding window
    if (cfg_obj.get("sliding_window")) |v| {
        if (v == .null) {
            config.has_sliding_window = false;
        } else {
            config.sliding_window = @intCast(v.integer);
            config.has_sliding_window = true;
        }
    }
    if (cfg_obj.get("sliding_window_pattern")) |v| config.sliding_window_pattern = @intCast(v.integer);

    // Gemma-specific: dual RoPE bases
    if (cfg_obj.get("rope_local_base_freq")) |v| config.rope_local_base_freq = jsonFloat(v);
    if (cfg_obj.get("rope_scaling")) |rs_val| {
        if (rs_val == .object) {
            if (rs_val.object.get("factor")) |v| config.rope_scaling_factor = jsonFloat(v);
        }
    }

    // Gemma 4: explicit layer_types array
    if (cfg_obj.get("layer_types")) |lt_val| {
        if (lt_val == .array) {
            config.has_explicit_layer_types = true;
            for (lt_val.array.items, 0..) |item, i| {
                if (i >= 128) break;
                if (item == .string) {
                    config.layer_is_global[i] = std.mem.eql(u8, item.string, "full_attention");
                }
            }
        }
    }

    // Gemma 4: dual head dimensions and KV sharing
    if (cfg_obj.get("global_head_dim")) |v| {
        if (v == .integer) config.global_head_dim = @intCast(v.integer);
    }
    if (cfg_obj.get("num_global_key_value_heads")) |v| {
        if (v == .integer) config.num_global_key_value_heads = @intCast(v.integer);
    }
    if (cfg_obj.get("num_kv_shared_layers")) |v| {
        if (v == .integer) config.num_kv_shared_layers = @intCast(v.integer);
    }
    if (cfg_obj.get("attention_k_eq_v")) |v| {
        if (v == .bool) config.attention_k_eq_v = v.bool;
    }
    if (cfg_obj.get("final_logit_softcapping")) |v| {
        config.final_logit_softcapping = jsonFloat(v);
    }
    if (cfg_obj.get("hidden_size_per_layer_input")) |v| {
        if (v == .integer) config.hidden_size_per_layer_input = @intCast(v.integer);
    }

    // Gemma 4: nested rope_parameters with per-attention-type config
    if (cfg_obj.get("rope_parameters")) |rp_val| {
        if (rp_val == .object) {
            // Gemma 4 style: { "full_attention": {...}, "sliding_attention": {...} }
            if (rp_val.object.get("full_attention")) |fa| {
                if (fa == .object) {
                    if (fa.object.get("rope_theta")) |v| config.rope_theta = jsonFloat(v);
                    if (fa.object.get("partial_rotary_factor")) |v| config.partial_rotary_factor_global = jsonFloat(v);
                    if (fa.object.get("rope_type")) |v| {
                        if (v == .string and std.mem.eql(u8, v.string, "proportional")) {
                            config.rope_proportional = true;
                            if (fa.object.get("factor")) |fv| config.rope_proportional_factor = jsonFloat(fv);
                        }
                    }
                }
            }
            if (rp_val.object.get("sliding_attention")) |sa| {
                if (sa == .object) {
                    if (sa.object.get("rope_theta")) |v| config.rope_local_base_freq = jsonFloat(v);
                }
            }
            // Qwen3.5 style: { "rope_theta": ..., "partial_rotary_factor": ... }
            if (rp_val.object.get("rope_theta")) |v| config.rope_theta = jsonFloat(v);
            if (rp_val.object.get("partial_rotary_factor")) |v| config.partial_rotary_factor = jsonFloat(v);
        }
    }

    // Tie word embeddings
    if (root.get("tie_word_embeddings")) |v| {
        if (v == .bool) config.tie_word_embeddings = v.bool;
    }
    if (cfg_obj.get("tie_word_embeddings")) |v| {
        if (v == .bool) config.tie_word_embeddings = v.bool;
    }

    // Check root level for max_position_embeddings (may not be in text_config)
    if (config.max_position_embeddings == 0) {
        if (root.get("max_position_embeddings")) |v| {
            if (v == .integer) config.max_position_embeddings = @intCast(v.integer);
        }
    }

    // Parse quantization from top level
    if (root.get("quantization")) |q_val| {
        const q = q_val.object;
        if (q.get("bits")) |v| config.quant_bits = @intCast(v.integer);
        if (q.get("group_size")) |v| config.quant_group_size = @intCast(v.integer);
        if (q.get("mode")) |v| {
            if (v == .string) {
                config.quant_mode = QuantMode.fromString(v.string) orelse {
                    log.err("unsupported quantization mode '{s}' (supported: affine, nvfp4, mxfp4, mxfp8)\n", .{v.string});
                    return error.UnsupportedQuantMode;
                };
            }
        }
    }

    // EOS tokens
    if (root.get("eos_token_id")) |v| {
        switch (v) {
            .integer => |i| config.addEosToken(@intCast(i)),
            .array => |arr| {
                for (arr.items) |item| {
                    if (item == .integer) config.addEosToken(@intCast(item.integer));
                }
            },
            else => {},
        }
    }

    // Vision config (Gemma 4 SigLIP)
    if (root.get("vision_config")) |vc_val| {
        if (vc_val == .object) {
            config.has_vision = true;
            const vc = vc_val.object;
            if (vc.get("hidden_size")) |v| { if (v == .integer) config.vision_hidden_size = @intCast(v.integer); }
            if (vc.get("num_hidden_layers")) |v| { if (v == .integer) config.vision_num_layers = @intCast(v.integer); }
            if (vc.get("num_attention_heads")) |v| { if (v == .integer) config.vision_num_heads = @intCast(v.integer); }
            if (vc.get("head_dim")) |v| { if (v == .integer) config.vision_head_dim = @intCast(v.integer); }
            if (vc.get("global_head_dim")) |v| { if (v == .integer) config.vision_head_dim = @intCast(v.integer); }
            if (vc.get("intermediate_size")) |v| { if (v == .integer) config.vision_intermediate_size = @intCast(v.integer); }
            if (vc.get("patch_size")) |v| { if (v == .integer) config.vision_patch_size = @intCast(v.integer); }
            if (vc.get("pooling_kernel_size")) |v| { if (v == .integer) config.vision_pooling_kernel = @intCast(v.integer); }
            if (vc.get("default_output_length")) |v| { if (v == .integer) config.vision_soft_tokens = @intCast(v.integer); }
            if (vc.get("position_embedding_size")) |v| { if (v == .integer) config.vision_position_embedding_size = @intCast(v.integer); }
            if (vc.get("rope_parameters")) |rp| {
                if (rp == .object) {
                    if (rp.object.get("rope_theta")) |v| config.vision_rope_theta = jsonFloat(v);
                }
            }
            if (vc.get("use_clipped_linears")) |v| { if (v == .bool) config.vision_use_clipped_linears = v.bool; }
            // vision_config.standardize is presence-only — the actual `std_scale`/`std_bias`
            // safetensors presence drives behavior in `VisionEncoder.init`, so the config
            // flag needs no field.

            // Gemma 4 12B unified (encoder-free) vision fields. Distinct names
            // from the SigLIP tower: mm_embed_dim (vs hidden_size),
            // model_patch_size (48px merged patch vs 16px teacher patch_size),
            // num_soft_tokens (vs default_output_length), mm_posemb_size.
            if (vc.get("mm_embed_dim")) |v| { if (v == .integer) config.vision_mm_embed_dim = @intCast(v.integer); }
            if (vc.get("model_patch_size")) |v| { if (v == .integer) config.vision_model_patch_size = @intCast(v.integer); }
            if (vc.get("num_soft_tokens")) |v| { if (v == .integer) config.vision_soft_tokens = @intCast(v.integer); }
            if (vc.get("mm_posemb_size")) |v| { if (v == .integer) config.vision_mm_posemb_size = @intCast(v.integer); }
        }
    }
    // Audio config (Gemma 4 12B unified — raw-waveform projection, no conformer)
    if (root.get("audio_config")) |ac_val| {
        if (ac_val == .object) {
            const ac = ac_val.object;
            if (ac.get("audio_embed_dim")) |v| { if (v == .integer) config.audio_embed_dim = @intCast(v.integer); }
            // audio_samples_per_token lives in processor_config, not config.json;
            // default 640 (40ms @ 16kHz) matches the only shipped unified checkpoint.
            if (ac.get("audio_samples_per_token")) |v| { if (v == .integer) config.audio_samples_per_token = @intCast(v.integer); }
        }
    }
    if (root.get("audio_token_id")) |v| {
        if (v == .integer) config.audio_token_id = @intCast(v.integer);
    }
    if (root.get("boa_token_id")) |v| {
        if (v == .integer) config.boa_token_id = @intCast(v.integer);
    }
    // eoa lives under `eoa_token_index` in the unified config.json.
    if (root.get("eoa_token_index")) |v| {
        if (v == .integer) config.eoa_token_id = @intCast(v.integer);
    }
    if (root.get("eoa_token_id")) |v| {
        if (v == .integer and config.eoa_token_id == 0) config.eoa_token_id = @intCast(v.integer);
    }
    // Image token ID (top-level or in mm_tokens_per_image config)
    if (root.get("image_token_id")) |v| {
        if (v == .integer) config.image_token_id = @intCast(v.integer);
    }
    if (root.get("image_token_index")) |v| {
        if (v == .integer and config.image_token_id == 0) config.image_token_id = @intCast(v.integer);
    }
    if (root.get("boi_token_id")) |v| {
        if (v == .integer) config.boi_token_id = @intCast(v.integer);
    }
    if (root.get("eoi_token_id")) |v| {
        if (v == .integer) config.eoi_token_id = @intCast(v.integer);
    }

    // Set model-family defaults based on model_type
    if (std.mem.eql(u8, model_type, "gemma3")) {
        config.model_type = "gemma3";
        // gemma-3-4b-it's text_config omits num_attention_heads / num_key_value_heads
        // / head_dim and leans on the HF Gemma3TextConfig defaults (8 q-heads,
        // 4 kv-heads, head_dim 256). Our struct defaults are the 12b/27b shape
        // (16/8), so apply the HF defaults explicitly when the config is silent —
        // otherwise the Q projection (8*256) reshapes against 16 heads and the
        // model crashes at warmup (issue #43). The 12b ships these fields, so its
        // values are read at lines 458-460 and these fills never fire.
        if (cfg_obj.get("num_attention_heads") == null) config.num_attention_heads = 8;
        if (cfg_obj.get("num_key_value_heads") == null) config.num_key_value_heads = 4;
        if (cfg_obj.get("head_dim") == null) config.head_dim = 256;
        config.weight_prefix = "language_model.model";
        config.hidden_act = .gelu_approx;
        config.norm_has_offset = true;
        config.scale_embeddings = true;
        config.has_pre_ff_norm = true;
        config.has_qk_norm = true;
        if (config.rope_scaling_factor == 1.0) {
            if (cfg_obj.get("rope_scaling")) |rs_val| {
                if (rs_val == .object) {
                    if (rs_val.object.get("factor")) |_| {} else {
                        config.rope_scaling_factor = 8.0;
                    }
                } else {
                    config.rope_scaling_factor = 8.0;
                }
            } else {
                config.rope_scaling_factor = 8.0;
            }
        }
        if (config.num_eos_tokens == 0) {
            config.addEosToken(1);
            config.addEosToken(106);
        }
    } else if (std.mem.eql(u8, model_type, "gemma4") or
        std.mem.eql(u8, model_type, "gemma4_text") or
        std.mem.eql(u8, model_type, "gemma4_unified") or
        std.mem.eql(u8, model_type, "gemma4_unified_text"))
    {
        // Per the Gemma 4 12B developer guide, `gemma4_unified` "contains the
        // same advanced decoder structure as the Gemma 4 31B Dense model" —
        // the unified-ness lives in a tiny vision/audio embedder we don't
        // wire here. So we collapse the internal tag onto plain "gemma4" so
        // every downstream model_type comparison in transformer.zig/drafter.zig
        // (attn_scale gate, recommendedBlockSize, etc.) treats it identically
        // to 31B Dense without per-arch fan-out.
        // Detect the unified (12B encoder-free multimodal) variant before we
        // collapse the tag — drives vision_embedder/embed_audio weight loading
        // and the encoder-free forward in src/vision.zig.
        config.is_gemma4_unified = std.mem.indexOf(u8, model_type, "unified") != null;
        config.model_type = "gemma4";
        config.weight_prefix = "language_model.model";
        config.hidden_act = .gelu_approx;
        config.norm_has_offset = false; // Gemma 4 norms have NO offset (plain weight, not 1+weight)
        config.scale_embeddings = true;
        config.has_pre_ff_norm = true;
        config.has_qk_norm = true;
        config.has_v_norm = true; // Parameter-free RMS norm on values
        config.rope_scaling_factor = 1.0; // No scaling, uses proportional RoPE via theta
        if (config.num_eos_tokens == 0) {
            config.addEosToken(1); // eos_token_id
        }
        // Note on `attention_k_eq_v` for the 12B unified checkpoint: the
        // weights ship separate v_proj for SLIDING layers and omit them for
        // FULL_ATTENTION layers — i.e. K==V alias is a per-layer choice
        // keyed on global-layer-ness. The existing bindModelWeights logic
        // (transformer.zig:5677) already encodes that:
        //   `k_eq_v = config.attention_k_eq_v and isGlobalLayer(li)`.
        // So we leave the parsed flag intact.
    } else if (std.mem.eql(u8, model_type, "diffusion_gemma")) {
        // DiffusionGemma (block diffusion, June 2026). The trunk is the
        // Gemma 4 26B-A4B MoE decoder verbatim — same dual-FFN layer
        // structure, sigma-MoE router, v_norm, dual head geometry,
        // proportional RoPE — under weight prefix `model.decoder`. The
        // model_type stays distinct because GENERATION is different: a
        // bidirectional canvas-denoising loop (src/diffusion.zig), not
        // autoregressive decode.
        config.model_type = "diffusion_gemma";
        config.weight_prefix = "model.decoder";
        config.hidden_act = .gelu_approx;
        config.norm_has_offset = false;
        config.scale_embeddings = true;
        config.has_pre_ff_norm = true;
        config.has_qk_norm = true;
        config.has_v_norm = true;
        config.rope_scaling_factor = 1.0;
        // Full-attention layers ship NO v_proj — V is the param-free-normed
        // k_proj output. Same per-layer alias the Gemma 4 31B/12B binder uses.
        config.attention_k_eq_v = true;
        // canvas_length: top-level; presence is what flags diffusion.
        if (root.get("canvas_length")) |v| {
            if (v == .integer) config.canvas_length = @intCast(v.integer);
        }
        if (config.canvas_length == 0) config.canvas_length = 256;
        // Diffusion knobs from the embedded generation_config object.
        if (root.get("generation_config")) |gc_val| {
            if (gc_val == .object) {
                const gc = gc_val.object;
                if (gc.get("max_denoising_steps")) |v| { if (v == .integer) config.diffusion_max_steps = @intCast(v.integer); }
                if (gc.get("t_min")) |v| config.diffusion_t_min = jsonFloat(v);
                if (gc.get("t_max")) |v| config.diffusion_t_max = jsonFloat(v);
                if (gc.get("confidence_threshold")) |v| config.diffusion_confidence_threshold = jsonFloat(v);
                if (gc.get("stability_threshold")) |v| { if (v == .integer) config.diffusion_stability_threshold = @intCast(v.integer); }
                if (gc.get("pad_token_id")) |v| { if (v == .integer) config.diffusion_pad_token = @intCast(v.integer); }
                if (gc.get("sampler_config")) |sc_val| {
                    if (sc_val == .object) {
                        if (sc_val.object.get("entropy_bound")) |v| config.diffusion_entropy_bound = jsonFloat(v);
                    }
                }
                // EOS may also live here (mirrors the top-level list).
                if (config.num_eos_tokens == 0) {
                    if (gc.get("eos_token_id")) |v| {
                        switch (v) {
                            .integer => |i| config.addEosToken(@intCast(i)),
                            .array => |arr| for (arr.items) |item| {
                                if (item == .integer) config.addEosToken(@intCast(item.integer));
                            },
                            else => {},
                        }
                    }
                }
            }
        }
        // The model.encoder.vision_tower is not wired yet (dropped at load by
        // shouldKeepWeightKey) — never advertise vision for this arch.
        config.has_vision = false;
        if (config.num_eos_tokens == 0) {
            config.addEosToken(1);
            config.addEosToken(106);
        }
    } else if (std.mem.eql(u8, model_type, "qwen3_5_moe") or
        std.mem.eql(u8, model_type, "qwen3_5") or
        std.mem.eql(u8, model_type, "qwen3_5_moe_text") or
        std.mem.eql(u8, model_type, "qwen3_5_text"))
    {
        config.model_type = "qwen3_5_moe";
        config.weight_prefix = "language_model.model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.has_sliding_window = false;
        config.attn_output_gate = true;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        if (cfg_obj.get("query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
    } else if (std.mem.eql(u8, model_type, "qwen3_moe") or
        std.mem.eql(u8, model_type, "qwen3_moe_text"))
    {
        // Qwen3-30B-A3B / Qwen3-Coder-30B-A3B. Shares qwen3_5_moe's weight
        // layout (`mlp.gate` router + stacked `mlp.switch_mlp.*` experts) and
        // its MoE forward, but differs in three ways that make it its OWN
        // model_type rather than a remap onto qwen3_5_moe:
        //   1. No GatedDeltaNet — every layer is full attention
        //      (full_attention_interval stays 0 ⇒ isLinearLayer == false).
        //   2. No attention output gate (attn_output_gate stays false; the
        //      qwen3_5 split-Q path would mis-shape the projection here).
        //   3. No shared expert (shared_expert_intermediate_size: 0, no
        //      mlp.shared_expert.* weights). The MoE binding in
        //      transformer.zig loads those optionally and the forward skips
        //      the shared branch when shared_expert_gate_w is null.
        // weight_prefix is plain "model" (no language_model nesting).
        config.model_type = "qwen3_moe";
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.has_sliding_window = false;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        if (cfg_obj.get("query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
    } else if (std.mem.eql(u8, model_type, "qwen3_next")) {
        config.model_type = "qwen3_next";
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.has_sliding_window = false;
        config.attn_output_gate = true;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        if (cfg_obj.get("partial_rotary_factor")) |v| config.partial_rotary_factor = jsonFloat(v);
        if (cfg_obj.get("query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
    } else if (std.mem.eql(u8, model_type, "lfm2") or std.mem.startsWith(u8, model_type, "lfm2")) {
        config.model_type = "lfm2";
        // VL variant nests text weights under language_model.model (like Gemma 4)
        config.weight_prefix = if (root.get("text_config") != null) "language_model.model" else "model";
        config.hidden_act = .silu;
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.has_sliding_window = false;
        config.has_hybrid_layers = true;
        config.has_embedding_norm = false;
        config.has_final_norm = true; // embedding_norm IS the final norm
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        if (config.head_dim == 256) { // default from gemma3, override
            config.head_dim = config.hidden_size / config.num_attention_heads;
        }
        config.query_pre_attn_scalar = config.head_dim;
        // tie_embedding (LFM2 name) -> tie_word_embeddings; default true for LFM2
        config.tie_word_embeddings = true;
        if (cfg_obj.get("tie_embedding")) |v| {
            if (v == .bool) config.tie_word_embeddings = v.bool;
        }
        if (cfg_obj.get("norm_eps")) |v| config.rms_norm_eps = jsonFloat(v);
        if (cfg_obj.get("conv_dim")) |v| config.lfm_conv_dim = switch (v) {
            .integer => |i| @intCast(i),
            else => 0,
        };
        // Parse layer_types array: ["conv", "full_attention", ...]
        if (cfg_obj.get("layer_types")) |lt_val| {
            if (lt_val == .array) {
                for (lt_val.array.items, 0..) |item, i| {
                    if (i >= 128) break;
                    if (item == .string) {
                        config.layer_block_types[i] = if (std.mem.eql(u8, item.string, "conv"))
                            .gated_conv
                        else
                            .attention;
                    }
                }
            }
        }
        if (config.num_eos_tokens == 0) {
            if (cfg_obj.get("eos_token_id")) |v| {
                if (v == .integer) config.addEosToken(@intCast(v.integer));
            }
        }
    } else if (std.mem.eql(u8, model_type, "nemotron_h")) {
        config.model_type = "nemotron_h";
        config.weight_prefix = "backbone";
        config.hidden_act = .silu;
        config.mamba_mlp_act = .relu_sq;
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false;
        config.has_sliding_window = false;
        config.has_hybrid_layers = true;
        config.has_final_norm = true;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        config.query_pre_attn_scalar = config.head_dim;
        if (cfg_obj.get("rms_norm_eps")) |v| {
            config.rms_norm_eps = jsonFloat(v);
        } else if (cfg_obj.get("layer_norm_epsilon")) |v| {
            config.rms_norm_eps = jsonFloat(v);
        }
        // Mamba2-specific config
        if (cfg_obj.get("mamba_num_heads")) |v| config.mamba_num_heads = switch (v) { .integer => |i| @intCast(i), else => 0 };
        if (cfg_obj.get("mamba_head_dim")) |v| config.mamba_head_dim = switch (v) { .integer => |i| @intCast(i), else => 0 };
        if (cfg_obj.get("n_groups")) |v| config.mamba_n_groups = switch (v) { .integer => |i| @intCast(i), else => 8 };
        if (cfg_obj.get("ssm_state_size")) |v| config.ssm_state_size = switch (v) { .integer => |i| @intCast(i), else => 128 };
        if (cfg_obj.get("conv_kernel")) |v| config.mamba_conv_kernel = switch (v) { .integer => |i| @intCast(i), else => 4 };
        if (cfg_obj.get("expand")) |v| config.mamba_expand = switch (v) { .integer => |i| @intCast(i), else => 2 };
        // time_step_limit: Python defaults to (0.0, inf) if not in config.
        // config.json may have time_step_min/time_step_max fields but Python ignores them
        // for SSM clipping — only time_step_limit (a 2-element array) is used.
        if (cfg_obj.get("time_step_limit")) |v| {
            if (v == .array) {
                const items = v.array.items;
                if (items.len >= 2) {
                    config.time_step_min = jsonFloat(items[0]);
                    config.time_step_max = jsonFloat(items[1]);
                }
            }
        }
        if (cfg_obj.get("chunk_size")) |v| config.mamba_chunk_size = switch (v) { .integer => |i| @intCast(i), else => 256 };
        // Parse hybrid_override_pattern: "M-M-M-MM-M-M*-..."
        if (cfg_obj.get("hybrid_override_pattern")) |v| {
            if (v == .string) {
                for (v.string, 0..) |ch, i| {
                    if (i >= 128) break;
                    config.layer_block_types[i] = switch (ch) {
                        'M' => .mamba2,
                        '-' => .mlp,
                        '*' => .attention,
                        'E' => .moe,
                        else => .attention,
                    };
                }
            }
        }
        if (config.num_eos_tokens == 0) {
            if (cfg_obj.get("eos_token_id")) |v| {
                if (v == .integer) config.addEosToken(@intCast(v.integer));
            }
        }
    } else if (std.mem.eql(u8, model_type, "deepseek_v4")) {
        // MLX-format DSV4 is not supported in this build. Users should load
        // the GGUF checkpoint via the ds4 engine (`*.gguf` early-branch in
        // main.zig / Swift app). Fall through to the unknown-arch error
        // path so the failure message points at the right thing.
        return error.UnsupportedDsv4MlxFormat;
    } else if (std.mem.eql(u8, model_type, "bert")) {
        config.model_type = "bert";
        config.is_encoder_only = true;
        config.weight_prefix = "";
        config.hidden_act = .gelu_approx;
        config.tie_word_embeddings = true;
        config.has_sliding_window = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false;
        config.scale_embeddings = false;
        config.norm_has_offset = false;
        config.head_dim = config.hidden_size / config.num_attention_heads;
        config.num_key_value_heads = config.num_attention_heads;
        config.query_pre_attn_scalar = config.head_dim;
        if (cfg_obj.get("layer_norm_eps")) |v| config.layer_norm_eps = jsonFloat(v);
        if (cfg_obj.get("type_vocab_size")) |v| config.type_vocab_size = switch (v) {
            .integer => |i| @intCast(i),
            else => 2,
        };
    } else {
        // Llama-family defaults (qwen3, llama, mistral, etc.)
        if (std.mem.eql(u8, model_type, "qwen3")) {
            config.model_type = "qwen3";
        } else if (std.mem.eql(u8, model_type, "qwen2")) {
            // Qwen2.5 family: dense Llama-style attention, NO QK-norm (left
            // false below), additive qkv-projection biases applied in the
            // forward when present.
            config.model_type = "qwen2";
        } else if (std.mem.eql(u8, model_type, "llama")) {
            config.model_type = "llama";
        } else if (std.mem.eql(u8, model_type, "mistral")) {
            config.model_type = "mistral";
        } else {
            config.model_type = "unknown";
        }
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        // Llama-family models (qwen2, llama, mistral) usually omit `head_dim`;
        // the HF default is hidden_size / num_attention_heads. Without this the
        // stale 256 sentinel (line 53) would corrupt attention for any such
        // checkpoint that doesn't ship an explicit head_dim (e.g. Qwen2.5).
        // qwen3 ships an explicit head_dim, so this leaves it untouched.
        if (cfg_obj.get("head_dim") == null) {
            config.head_dim = config.hidden_size / config.num_attention_heads;
        }

        if (cfg_obj.get("hidden_act")) |v| {
            if (v == .string) {
                if (std.mem.eql(u8, v.string, "silu")) {
                    config.hidden_act = .silu;
                } else if (std.mem.eql(u8, v.string, "gelu_pytorch_tanh")) {
                    config.hidden_act = .gelu_approx;
                }
            }
        }

        if (cfg_obj.get("query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }

        if (std.mem.eql(u8, model_type, "qwen3")) {
            config.has_qk_norm = true;
        }
    }

    return config;
}

fn jsonFloat(v: std.json.Value) f32 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => 0.0,
    };
}

/// Holds all loaded weights as mlx arrays, keyed by name.
pub const Weights = struct {
    map: std.StringHashMap(mlx.mlx_array),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Weights {
        return .{
            .map = std.StringHashMap(mlx.mlx_array).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Weights) void {
        var it = self.map.iterator();
        while (it.next()) |entry| {
            _ = mlx.mlx_array_free(entry.value_ptr.*);
            self.allocator.free(entry.key_ptr.*);
        }
        self.map.deinit();
    }

    pub fn get(self: *const Weights, name: []const u8) ?mlx.mlx_array {
        return self.map.get(name);
    }

    pub fn count(self: *const Weights) u32 {
        return @intCast(self.map.count());
    }
};

/// Load all safetensors files from model_dir.
/// When `load_vision` is true, vision_tower and multi_modal_projector weights are included.
pub fn loadWeights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Weights {
    return loadWeightsOpt(io, allocator, model_dir, false);
}

pub fn loadWeightsWithVision(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Weights {
    return loadWeightsOpt(io, allocator, model_dir, true);
}

fn loadWeightsOpt(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, load_vision: bool) !Weights {
    var weights = Weights.init(allocator);
    errdefer weights.deinit();

    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true });
    defer dir.close(io);

    var file_count: u32 = 0;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;

        const path_slice = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, entry.name });
        defer allocator.free(path_slice);
        const path = try allocator.dupeZ(u8, path_slice);
        defer allocator.free(path);

        log.info("Loading {s}...\n", .{entry.name});
        try loadSafetensorsFile(allocator, &weights, path, s, load_vision);
        file_count += 1;
    }

    log.info("Loaded {d} weights from {d} file(s)\n", .{ weights.count(), file_count });
    return weights;
}

fn loadSafetensorsFile(
    allocator: std.mem.Allocator,
    weights: *Weights,
    path: [*:0]const u8,
    s: mlx.mlx_stream,
    load_vision: bool,
) !void {
    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);

    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);

    try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, path, s));

    const iter = mlx.mlx_map_string_to_array_iterator_new(tensor_map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);

    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();

        const ret = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iter);
        if (ret != 0 or key == null) {
            _ = mlx.mlx_array_free(value);
            break;
        }

        const key_str = std.mem.span(key.?);

        if (!shouldKeepWeightKey(key_str, load_vision)) {
            _ = mlx.mlx_array_free(value);
            continue;
        }

        const owned_key = try allocator.dupe(u8, key_str);
        try weights.map.put(owned_key, value);
    }
}

/// True if the safetensors weight `key` should be retained for the text
/// forward pass. Audio is always dropped; vision is dropped unless
/// `load_vision` is set. MTP-style head tensors (`*.mtp.*`) on Qwen3.5/3.6
/// checkpoints are kept (the binder ignores them, but the loader doesn't
/// need to know that).
pub fn shouldKeepWeightKey(key: []const u8, load_vision: bool) bool {
    // Gemma 4 12B `gemma4_unified` is encoder-free: it ships a tiny vision
    // patch embedder (`vision_embedder.*` + `embed_vision.*`) and a raw-waveform
    // audio projection (`embed_audio.*`) instead of the SigLIP vision tower and
    // conformer audio tower of earlier Gemma 4 variants. Those embedders are
    // wired in src/vision.zig (UnifiedEmbedder), so keep them under the same
    // `load_vision` gate as the SigLIP weights (`--no-vision` → text only).
    const is_vision = std.mem.startsWith(u8, key, "vision_tower.") or
        std.mem.startsWith(u8, key, "embed_vision.") or
        std.mem.startsWith(u8, key, "vision_embedder.") or
        std.mem.startsWith(u8, key, "embed_audio.") or
        std.mem.startsWith(u8, key, "multi_modal_projector.") or
        std.mem.startsWith(u8, key, "language_model.multi_modal_projector.");
    // The heavy SigLIP-era conformer audio tower is still not wired — drop it.
    const is_audio_tower = std.mem.startsWith(u8, key, "audio_tower.") or
        std.mem.startsWith(u8, key, "language_model.audio_multi_modal_projector.");
    if (is_audio_tower) return false;
    // DiffusionGemma nests its (not-yet-wired) vision tower under
    // model.encoder.* — always drop it so a 26B text load doesn't carry
    // ~1 GB of dead tower weights. The encoder LAYER SCALARS
    // (model.encoder.language_model.layers.N.layer_scalar) must survive:
    // they're the only untied encoder text params and the causal encoder
    // pass multiplies by them instead of the decoder's layer_scalar.
    if (std.mem.startsWith(u8, key, "model.encoder.vision_tower.") or
        std.mem.startsWith(u8, key, "model.encoder.embed_vision.")) return false;
    if (is_vision and !load_vision) return false;
    return true;
}

// ── Tests ──

const testing = std.testing;

test "ModelConfig defaults" {
    const config = ModelConfig{};
    try testing.expectEqual(@as(u32, 0), config.num_eos_tokens);
    try testing.expectEqual(@as(u32, 0), config.max_position_embeddings);
    try testing.expectEqual(@as(u32, 0), config.quant_bits); // 0 = dense bf16 (no "quantization" key)
    try testing.expectEqual(@as(u32, 64), config.quant_group_size);
    try testing.expect(!config.tie_word_embeddings);
}

test "ModelConfig addEosToken" {
    var config = ModelConfig{};
    config.addEosToken(1);
    config.addEosToken(106);
    try testing.expectEqual(@as(u32, 2), config.num_eos_tokens);
    try testing.expect(config.isEosToken(1));
    try testing.expect(config.isEosToken(106));
    try testing.expect(!config.isEosToken(42));
}

test "ModelConfig addEosToken max capacity" {
    var config = ModelConfig{};
    // Fill all 8 slots
    for (0..8) |i| {
        config.addEosToken(@intCast(i + 100));
    }
    try testing.expectEqual(@as(u32, 8), config.num_eos_tokens);
    // 9th should be silently dropped
    config.addEosToken(999);
    try testing.expectEqual(@as(u32, 8), config.num_eos_tokens);
    try testing.expect(!config.isEosToken(999));
}

test "EOS merge: chat-terminator added even when config already provided an eos" {
    // Regression for the Qwen2.5-Coder-7B leak: its config.json sets
    // eos_token_id=<|endoftext|> (151643), but its chat template ends turns
    // with <|im_end|> (151645). The load path (main.zig / scheduler doLoad)
    // must ALWAYS merge the tokenizer's chat-terminator EOS — additively and
    // dedup-guarded — not only when config provided none; otherwise <|im_end|>
    // is never a stop token and leaks into the output (broke structured JSON /
    // tool calling). This pins the merge invariant those call sites implement.
    var config = ModelConfig{};
    config.addEosToken(151643); // from config.json eos_token_id
    try testing.expectEqual(@as(u32, 1), config.num_eos_tokens);

    // Merge step the fix performs: add the chat terminator if absent.
    const chat_eos: u32 = 151645; // <|im_end|>, from tokenizer_config eos_token
    if (!config.isEosToken(chat_eos)) config.addEosToken(chat_eos);

    try testing.expect(config.isEosToken(151645)); // now stops on <|im_end|>
    try testing.expect(config.isEosToken(151643)); // original preserved
    try testing.expectEqual(@as(u32, 2), config.num_eos_tokens);

    // Idempotent: re-running the merge must not duplicate.
    if (!config.isEosToken(chat_eos)) config.addEosToken(chat_eos);
    try testing.expectEqual(@as(u32, 2), config.num_eos_tokens);
}

test "ModelConfig eosTokenSlice" {
    var config = ModelConfig{};
    config.addEosToken(10);
    config.addEosToken(20);
    const slice = config.eosTokenSlice();
    try testing.expectEqual(@as(usize, 2), slice.len);
    try testing.expectEqual(@as(u32, 10), slice[0]);
    try testing.expectEqual(@as(u32, 20), slice[1]);
}

test "pickUserTurnPrefix Gemma 4 wins over older patterns" {
    // Gemma 4 templates also contain "<start_of_turn>" inside fallback comments
    // in some checkpoints — make sure we still pick the Gemma 4 marker first.
    const tmpl = "{{- '<|turn>' + role + '\n' }} {# legacy: <start_of_turn> #}";
    try testing.expectEqualStrings("<|turn>user\n", pickUserTurnPrefix(tmpl).?);
}

test "pickUserTurnPrefix Gemma 3" {
    const tmpl = "<start_of_turn>user\n{{ message['content'] }}<end_of_turn>";
    try testing.expectEqualStrings("<start_of_turn>user\n", pickUserTurnPrefix(tmpl).?);
}

test "pickUserTurnPrefix Qwen ChatML" {
    const tmpl = "<|im_start|>user\n{{ message['content'] }}<|im_end|>";
    try testing.expectEqualStrings("<|im_start|>user\n", pickUserTurnPrefix(tmpl).?);
}

test "pickUserTurnPrefix Llama 3" {
    const tmpl = "<|start_header_id|>user<|end_header_id|>\n\n{{ content }}<|eot_id|>";
    try testing.expectEqualStrings("<|start_header_id|>user<|end_header_id|>\n\n", pickUserTurnPrefix(tmpl).?);
}

test "pickUserTurnPrefix unknown template returns null" {
    try testing.expect(pickUserTurnPrefix("[INST] {{ content }} [/INST]") == null);
    try testing.expect(pickUserTurnPrefix("") == null);
}

test "ModelConfig userTurnMarkerSlice respects length" {
    var config = ModelConfig{};
    config.user_turn_marker_ids[0] = 105;
    config.user_turn_marker_ids[1] = 2364;
    config.user_turn_marker_ids[2] = 107;
    config.user_turn_marker_len = 3;
    const slice = config.userTurnMarkerSlice();
    try testing.expectEqual(@as(usize, 3), slice.len);
    try testing.expectEqual(@as(u32, 105), slice[0]);
    try testing.expectEqual(@as(u32, 2364), slice[1]);
    try testing.expectEqual(@as(u32, 107), slice[2]);
}

test "ModelConfig isGlobalLayer with sliding window" {
    // Gemma 3 convention (HF + mlx-lm): every Nth layer is global, with the
    // pattern anchored at the END of each group — global when
    // `(idx + 1) % pattern == 0`, i.e. layers 5, 11, 17… for pattern 6.
    // The old `% pattern == 0` phase made layer 0 global and layer 5 local —
    // every layer got the wrong RoPE base/scale and attention scope, which
    // surfaced as fluent-but-wrong output (spaced digits, broken arithmetic)
    // on gemma-3-12b. Gemma 4 ships explicit layer_types and never hits this
    // fallback.
    var config = ModelConfig{};
    config.has_sliding_window = true;
    config.sliding_window_pattern = 6;
    try testing.expect(!config.isGlobalLayer(0));
    try testing.expect(!config.isGlobalLayer(1));
    try testing.expect(config.isGlobalLayer(5));
    try testing.expect(!config.isGlobalLayer(6));
    try testing.expect(config.isGlobalLayer(11));
    try testing.expect(!config.isGlobalLayer(12));
}

test "ModelConfig isGlobalLayer without sliding window" {
    var config = ModelConfig{};
    config.has_sliding_window = false;
    // All layers should be global
    try testing.expect(config.isGlobalLayer(0));
    try testing.expect(config.isGlobalLayer(1));
    try testing.expect(config.isGlobalLayer(5));
}

test "ModelConfig isLinearLayer" {
    var config = ModelConfig{};
    config.full_attention_interval = 4;
    // Layer 0: (0+1) % 4 == 1 != 0 → linear
    try testing.expect(config.isLinearLayer(0));
    // Layer 3: (3+1) % 4 == 0 → NOT linear (full attention)
    try testing.expect(!config.isLinearLayer(3));
    // Layer 7: (7+1) % 4 == 0 → NOT linear
    try testing.expect(!config.isLinearLayer(7));
    // Layer 4: (4+1) % 4 == 1 → linear
    try testing.expect(config.isLinearLayer(4));
}

test "ModelConfig isLinearLayer disabled" {
    var config = ModelConfig{};
    config.full_attention_interval = 0;
    try testing.expect(!config.isLinearLayer(0));
    try testing.expect(!config.isLinearLayer(5));
}

test "ModelConfig isMoe" {
    var config = ModelConfig{};
    try testing.expect(!config.isMoe());
    config.num_experts = 8;
    try testing.expect(config.isMoe());
}

test "jsonFloat converts integer" {
    const val = std.json.Value{ .integer = 42 };
    try testing.expectApproxEqAbs(@as(f32, 42.0), jsonFloat(val), 0.001);
}

test "jsonFloat converts float" {
    const val = std.json.Value{ .float = 3.14 };
    try testing.expectApproxEqAbs(@as(f32, 3.14), jsonFloat(val), 0.01);
}

test "ModelConfig isGlobalLayer with explicit layer_types" {
    var config = ModelConfig{};
    config.has_sliding_window = true;
    config.has_explicit_layer_types = true;
    // Set layer 4 and 9 as global (like Gemma 4 E2B pattern)
    config.layer_is_global[4] = true;
    config.layer_is_global[9] = true;
    try testing.expect(!config.isGlobalLayer(0));
    try testing.expect(!config.isGlobalLayer(3));
    try testing.expect(config.isGlobalLayer(4));
    try testing.expect(!config.isGlobalLayer(5));
    try testing.expect(config.isGlobalLayer(9));
}

test "ModelConfig getKVSourceLayer" {
    var config = ModelConfig{};
    config.num_hidden_layers = 35;
    config.num_kv_shared_layers = 20;
    config.has_sliding_window = true;
    config.has_explicit_layer_types = true;
    // E2B pattern: every 5th layer starting from 4 is global
    for (0..35) |i| {
        config.layer_is_global[i] = (i % 5 == 4);
    }
    // Layers 0-14 are concrete (no source)
    try testing.expect(config.getKVSourceLayer(0) == null);
    try testing.expect(config.getKVSourceLayer(14) == null);
    // Layer 15 (sliding) -> should map to layer 13 (last concrete sliding)
    try testing.expectEqual(@as(?u32, 13), config.getKVSourceLayer(15));
    // Layer 19 (full) -> should map to layer 14 (last concrete full)
    try testing.expectEqual(@as(?u32, 14), config.getKVSourceLayer(19));
    // Layer 20 (sliding) -> should also map to layer 13
    try testing.expectEqual(@as(?u32, 13), config.getKVSourceLayer(20));
}

test "ModelConfig layerHeadDim" {
    var config = ModelConfig{};
    config.head_dim = 256;
    config.global_head_dim = 512;
    config.has_sliding_window = true;
    config.has_explicit_layer_types = true;
    config.layer_is_global[4] = true;
    try testing.expectEqual(@as(u32, 256), config.layerHeadDim(0));
    try testing.expectEqual(@as(u32, 512), config.layerHeadDim(4));
}

test "ModelConfig BERT defaults" {
    var config = ModelConfig{};
    config.is_encoder_only = true;
    config.model_type = "bert";
    config.hidden_size = 384;
    config.num_attention_heads = 12;
    config.head_dim = 384 / 12;
    config.num_key_value_heads = 12;

    try testing.expect(config.is_encoder_only);
    try testing.expectEqual(@as(u32, 32), config.head_dim);
    try testing.expectEqual(@as(u32, 12), config.num_key_value_heads);
    try testing.expectApproxEqAbs(@as(f32, 1e-12), config.layer_norm_eps, 1e-15);
}

test "ModelConfig BERT is not MoE" {
    var config = ModelConfig{};
    config.is_encoder_only = true;
    config.model_type = "bert";
    try testing.expect(!config.isMoe());
}

test "ModelConfig BERT has no sliding window" {
    var config = ModelConfig{};
    config.is_encoder_only = true;
    config.has_sliding_window = false;
    try testing.expect(config.isGlobalLayer(0));
    try testing.expect(config.isGlobalLayer(5));
}

test "shouldKeepWeightKey accepts orphan MTP head weights on Qwen3.5/3.6 checkpoints" {
    // Some Qwen3.5/3.6 checkpoints embed `*.mtp.*` tensors in the MAIN
    // shards (the sidecar-based MTP head in src/mtp.zig loads separately).
    // The safetensors iterator must let them through (they're neither vision
    // nor audio) so the model loads cleanly; the trunk binder ignores them.
    try testing.expect(shouldKeepWeightKey("language_model.model.mtp.0.eh_proj.weight", true));
    try testing.expect(shouldKeepWeightKey("language_model.model.mtp.0.eh_proj.weight", false));
    try testing.expect(shouldKeepWeightKey("model.mtp.0.shared_head.head.weight", false));
}

test "shouldKeepWeightKey filters audio and gated vision weights" {
    // Regression: the existing filter should still reject audio and reject
    // vision when load_vision is false.
    try testing.expect(!shouldKeepWeightKey("audio_tower.encoder.layer.0.weight", true));
    try testing.expect(!shouldKeepWeightKey("vision_tower.encoder.layer.0.weight", false));
    try testing.expect(shouldKeepWeightKey("vision_tower.encoder.layer.0.weight", true));
    try testing.expect(shouldKeepWeightKey("language_model.model.layers.0.self_attn.q_proj.weight", false));
}

test "shouldKeepWeightKey keeps Gemma 4 12B unified embedder weights when vision enabled" {
    // gemma4_unified is encoder-free: vision_embedder.* (patch embedder),
    // embed_vision.* and embed_audio.* (raw projections) ARE wired in
    // src/vision.zig (UnifiedEmbedder), so they must be kept under load_vision.
    // The heavy SigLIP-era conformer audio_tower.* stays dropped.
    try testing.expect(shouldKeepWeightKey("vision_embedder.patch_dense.weight", true));
    try testing.expect(shouldKeepWeightKey("embed_vision.embedding_projection.weight", true));
    try testing.expect(shouldKeepWeightKey("embed_audio.embedding_projection.weight", true));
    // Gated off by --no-vision.
    try testing.expect(!shouldKeepWeightKey("vision_embedder.patch_dense.weight", false));
    try testing.expect(!shouldKeepWeightKey("embed_audio.embedding_projection.weight", false));
    // The conformer audio tower is never wired — always dropped.
    try testing.expect(!shouldKeepWeightKey("audio_tower.encoder.layer.0.weight", true));
}

test "ModelConfig parses gemma4_unified text_config" {
    // Gemma 4 12B base ships `model_type: gemma4_unified` with text_config
    // carrying the language tower. The dispatch arm must:
    //   - tag as gemma4_unified
    //   - inherit the gemma4 weight prefix + norm/scale flags
    //   - pass attention_k_eq_v through unchanged so the per-layer binder
    //     (transformer.zig:5677) aliases V to K on global layers but uses
    //     the shipped v_proj on sliding layers
    //   - pass through gemma4 fields like global_head_dim, final_logit_softcapping.
    const json =
        \\{
        \\  "model_type": "gemma4_unified",
        \\  "text_config": {
        \\    "model_type": "gemma4_unified_text",
        \\    "hidden_size": 3840,
        \\    "intermediate_size": 15360,
        \\    "num_hidden_layers": 48,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 8,
        \\    "head_dim": 256,
        \\    "global_head_dim": 512,
        \\    "num_global_key_value_heads": 8,
        \\    "num_kv_shared_layers": 0,
        \\    "hidden_size_per_layer_input": 0,
        \\    "layer_types": ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
        \\    "rope_parameters": {
        \\      "full_attention": {"rope_theta": 1000000.0, "rope_type": "proportional", "factor": 1.0},
        \\      "sliding_attention": {"rope_theta": 10000.0}
        \\    },
        \\    "attention_k_eq_v": true,
        \\    "final_logit_softcapping": 30.0,
        \\    "hidden_activation": "gelu_pytorch_tanh",
        \\    "rms_norm_eps": 1e-06,
        \\    "max_position_embeddings": 8192,
        \\    "sliding_window": 1024
        \\  },
        \\  "quantization": {"bits": 4, "group_size": 64}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    // Collapsed onto "gemma4" so downstream code paths (attn_scale gate,
    // recommendedBlockSize) match the 31B Dense decoder it inherits.
    try testing.expectEqualStrings("gemma4", config.model_type);
    try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    try testing.expect(config.has_v_norm);
    try testing.expect(config.has_pre_ff_norm);
    try testing.expect(config.has_qk_norm);
    try testing.expect(!config.norm_has_offset);
    try testing.expect(config.scale_embeddings);
    // 12B's k_proj/v_proj layout is mixed: full_attention layers omit v_proj
    // (K=V alias), sliding layers ship it. The existing per-layer binder
    // (transformer.zig:5677) keys the alias on isGlobalLayer; we must keep
    // the config flag intact so that AND-clause fires on global layers only.
    try testing.expect(config.attention_k_eq_v);
    try testing.expectEqual(@as(u32, 512), config.global_head_dim);
    try testing.expectEqual(@as(u32, 0), config.hidden_size_per_layer_input);
    try testing.expectApproxEqAbs(@as(f32, 30.0), config.final_logit_softcapping, 0.001);
    try testing.expectEqual(@as(u32, 3840), config.hidden_size);
    try testing.expectEqual(@as(u32, 48), config.num_hidden_layers);
    // Unified flag drives the encoder-free vision/audio embedder path.
    try testing.expect(config.is_gemma4_unified);
}

test "ModelConfig fills HF gemma3 defaults when text_config omits head counts" {
    // gemma-3-4b-it-4bit's text_config carries hidden_size/num_hidden_layers but
    // OMITS num_attention_heads/num_key_value_heads/head_dim, relying on the HF
    // Gemma3TextConfig defaults (8 q-heads / 4 kv-heads / head_dim 256). Our
    // struct defaults are the 12b/27b shape (16 q / 8 kv), so without an explicit
    // fill the Q projection (8*256=2048) gets reshaped against 16 heads and the
    // model crashes at warmup with "Cannot reshape array of size 2048 into shape
    // (1,1,16,256)" (issue #43). The 12b config ships these fields explicitly, so
    // it was never affected.
    const json =
        \\{
        \\  "model_type": "gemma3",
        \\  "text_config": {
        \\    "model_type": "gemma3_text",
        \\    "hidden_size": 2560,
        \\    "intermediate_size": 10240,
        \\    "num_hidden_layers": 34,
        \\    "sliding_window": 1024,
        \\    "rms_norm_eps": 1e-06
        \\  },
        \\  "vision_config": {"hidden_size": 1152},
        \\  "quantization": {"bits": 4, "group_size": 32}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("gemma3", config.model_type);
    // HF Gemma3TextConfig defaults — NOT our 12b-shaped struct defaults (16/8).
    try testing.expectEqual(@as(u32, 8), config.num_attention_heads);
    try testing.expectEqual(@as(u32, 4), config.num_key_value_heads);
    try testing.expectEqual(@as(u32, 256), config.head_dim);
    // Fields the 4b text_config DOES carry must still win.
    try testing.expectEqual(@as(u32, 2560), config.hidden_size);
    try testing.expectEqual(@as(u32, 34), config.num_hidden_layers);
}

test "ModelConfig keeps explicit gemma3 head counts (12b)" {
    // Regression guard for the fix above: a gemma3 text_config that DOES ship
    // head counts must keep them, never get clobbered by the HF-default fill.
    const json =
        \\{
        \\  "model_type": "gemma3",
        \\  "text_config": {
        \\    "model_type": "gemma3_text",
        \\    "hidden_size": 3840,
        \\    "num_hidden_layers": 48,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 8,
        \\    "head_dim": 256,
        \\    "sliding_window": 1024
        \\  },
        \\  "quantization": {"bits": 4, "group_size": 32}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(@as(u32, 16), config.num_attention_heads);
    try testing.expectEqual(@as(u32, 8), config.num_key_value_heads);
    try testing.expectEqual(@as(u32, 256), config.head_dim);
}

test "ModelConfig parses gemma4_unified vision + audio multimodal fields" {
    // The 12B unified config.json carries top-level vision_config/audio_config
    // with encoder-free dims plus image/audio/boi/boa/eoi/eoa token ids. These
    // drive the UnifiedEmbedder forward and placeholder insertion.
    const json =
        \\{
        \\  "model_type": "gemma4_unified",
        \\  "image_token_id": 258880,
        \\  "audio_token_id": 258881,
        \\  "boi_token_id": 255999,
        \\  "eoi_token_id": 258882,
        \\  "boa_token_id": 256000,
        \\  "eoa_token_index": 258883,
        \\  "vision_config": {
        \\    "model_type": "gemma4_unified_vision",
        \\    "mm_embed_dim": 3840,
        \\    "mm_posemb_size": 1120,
        \\    "model_patch_size": 48,
        \\    "patch_size": 16,
        \\    "pooling_kernel_size": 3,
        \\    "num_soft_tokens": 280,
        \\    "output_proj_dims": 3840,
        \\    "rms_norm_eps": 1e-06
        \\  },
        \\  "audio_config": {
        \\    "model_type": "gemma4_unified_audio",
        \\    "audio_embed_dim": 640,
        \\    "output_proj_dims": 640,
        \\    "rms_norm_eps": 1e-06
        \\  },
        \\  "text_config": {
        \\    "hidden_size": 3840,
        \\    "num_hidden_layers": 48,
        \\    "head_dim": 256
        \\  },
        \\  "quantization": {"bits": 4, "group_size": 64}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(config.is_gemma4_unified);
    try testing.expect(config.has_vision);
    // Vision (encoder-free) dims.
    try testing.expectEqual(@as(u32, 3840), config.vision_mm_embed_dim);
    try testing.expectEqual(@as(u32, 48), config.vision_model_patch_size);
    try testing.expectEqual(@as(u32, 1120), config.vision_mm_posemb_size);
    try testing.expectEqual(@as(u32, 280), config.vision_soft_tokens);
    try testing.expectEqual(@as(u32, 3), config.vision_pooling_kernel);
    // Audio.
    try testing.expectEqual(@as(u32, 640), config.audio_embed_dim);
    // Multimodal token ids.
    try testing.expectEqual(@as(u32, 258880), config.image_token_id);
    try testing.expectEqual(@as(u32, 258881), config.audio_token_id);
    try testing.expectEqual(@as(u32, 255999), config.boi_token_id);
    try testing.expectEqual(@as(u32, 258882), config.eoi_token_id);
    try testing.expectEqual(@as(u32, 256000), config.boa_token_id);
    try testing.expectEqual(@as(u32, 258883), config.eoa_token_id);
}

test "parseConfigFromJson dense bf16 qwen3_5_moe → quant_bits 0" {
    // A fully-dense bf16 checkpoint (e.g. Qwen3.6-35B-A3B-bf16) has NO
    // "quantization" key. quant_bits must stay 0 so the loader skips every
    // .scales/.biases fetch and the forward pass dispatches to plain matmul.
    const json =
        \\{
        \\  "model_type": "qwen3_5_moe",
        \\  "text_config": {
        \\    "hidden_size": 2048,
        \\    "head_dim": 256,
        \\    "num_hidden_layers": 40,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 2,
        \\    "num_experts": 256,
        \\    "num_experts_per_tok": 8,
        \\    "moe_intermediate_size": 512,
        \\    "attn_output_gate": true,
        \\    "tie_word_embeddings": false
        \\  }
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(@as(u32, 0), config.quant_bits);
    try testing.expectEqualStrings("qwen3_5_moe", config.model_type);
    try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    try testing.expect(config.attn_output_gate);
    try testing.expect(config.isMoe());
    try testing.expectEqual(@as(u32, 256), config.num_experts);
}

test "parseConfigFromJson quantized qwen3_5_moe → quant_bits from key" {
    // Same arch but with a "quantization" block: quant_bits must reflect it so
    // the mandatory scale/bias fetches still fire (a missing scale is a clear
    // MISSING WEIGHT error, not a silent dense fallback). Guards the default flip.
    const json =
        \\{
        \\  "model_type": "qwen3_5_moe",
        \\  "text_config": {"hidden_size": 2048, "num_experts": 256},
        \\  "quantization": {"bits": 4, "group_size": 64}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(@as(u32, 4), config.quant_bits);
    try testing.expectEqual(@as(u32, 64), config.quant_group_size);
    try testing.expectEqual(QuantMode.affine, config.quant_mode);
}

test "parseConfigFromJson nvfp4 quantization mode" {
    // NVFP4 checkpoints (issue #24): {"group_size": 16, "bits": 4, "mode": "nvfp4"}.
    // The mode must land on config.quant_mode so the loader skips the .biases
    // fetches (nvfp4 stores no biases tensors) and the matmul call sites pass
    // "nvfp4" to mlx instead of "affine".
    const json =
        \\{
        \\  "model_type": "qwen3",
        \\  "hidden_size": 1024,
        \\  "quantization": {"group_size": 16, "bits": 4, "mode": "nvfp4"}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(QuantMode.nvfp4, config.quant_mode);
    try testing.expectEqual(@as(u32, 4), config.quant_bits);
    try testing.expectEqual(@as(u32, 16), config.quant_group_size);
    try testing.expect(!config.quant_mode.hasBiases());
}

test "parseConfigFromJson explicit affine mode keeps biases" {
    const json =
        \\{
        \\  "model_type": "qwen3",
        \\  "hidden_size": 1024,
        \\  "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(QuantMode.affine, config.quant_mode);
    try testing.expect(config.quant_mode.hasBiases());
}

test "parseConfigFromJson unknown quantization mode → error" {
    // An unrecognized mode must fail loudly at config parse — not crash later
    // in the weight loader with a misleading MISSING WEIGHT error.
    const json =
        \\{
        \\  "model_type": "qwen3",
        \\  "hidden_size": 1024,
        \\  "quantization": {"group_size": 32, "bits": 4, "mode": "fp99"}
        \\}
    ;
    try testing.expectError(error.UnsupportedQuantMode, parseConfigFromJson(testing.allocator, json));
}

test "parseConfigFromJson qwen3_moe (Qwen3-30B-A3B) → MoE, no shared expert, no output gate" {
    // Qwen3-Coder-30B-A3B / Qwen3-30B-A3B ship model_type "qwen3_moe": a pure
    // full-attention MoE (no GatedDeltaNet) that DROPPED the shared expert that
    // Qwen2-MoE / Qwen3.5-MoE carry (shared_expert_intermediate_size: 0, no
    // mlp.shared_expert.* weights). It must NOT be remapped onto qwen3_5_moe
    // (which assumes a shared expert and an attention output gate) — doing so
    // crashed at load with "MISSING WEIGHT: ...mlp.shared_expert.gate_proj.weight".
    const json =
        \\{
        \\  "model_type": "qwen3_moe",
        \\  "hidden_size": 2048,
        \\  "head_dim": 128,
        \\  "num_hidden_layers": 48,
        \\  "num_attention_heads": 32,
        \\  "num_key_value_heads": 4,
        \\  "num_experts": 128,
        \\  "num_experts_per_tok": 8,
        \\  "moe_intermediate_size": 768,
        \\  "shared_expert_intermediate_size": 0,
        \\  "use_qk_norm": true,
        \\  "use_sliding_window": false,
        \\  "rope_theta": 10000000,
        \\  "tie_word_embeddings": false,
        \\  "quantization": {"bits": 8, "group_size": 64}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("qwen3_moe", config.model_type);
    try testing.expectEqualStrings("model", config.weight_prefix);
    try testing.expect(config.isMoe());
    try testing.expectEqual(@as(u32, 128), config.num_experts);
    try testing.expectEqual(@as(u32, 8), config.num_experts_per_tok);
    // qwen3 attention: QK-norm on, NO output gate (that's a qwen3_5 thing).
    try testing.expect(config.has_qk_norm);
    try testing.expect(!config.attn_output_gate);
    // Full attention everywhere — no GatedDeltaNet/linear layers.
    try testing.expect(!config.isLinearLayer(0));
    try testing.expect(!config.isLinearLayer(3));
    try testing.expect(!config.has_hybrid_layers);
    try testing.expect(!config.has_sliding_window);
    try testing.expectEqual(@as(u32, 8), config.quant_bits);
}

test "parseConfigFromJson qwen2 (Qwen2.5) → dense, no QK-norm, silu" {
    // Qwen2.5-Coder / Qwen2.5-Instruct ship model_type "qwen2": a dense
    // full-attention Llama-family arch that, unlike qwen3, has NO QK-norm and
    // DOES carry additive qkv-projection biases (q/k/v_proj.bias). The forward
    // applies those biases when present; here we pin the config classification.
    const json =
        \\{
        \\  "model_type": "qwen2",
        \\  "hidden_size": 5120,
        \\  "num_hidden_layers": 64,
        \\  "num_attention_heads": 40,
        \\  "num_key_value_heads": 8,
        \\  "intermediate_size": 27648,
        \\  "rms_norm_eps": 1e-6,
        \\  "rope_theta": 1000000.0,
        \\  "hidden_act": "silu",
        \\  "tie_word_embeddings": false,
        \\  "quantization": {"bits": 8, "group_size": 64}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("qwen2", config.model_type);
    try testing.expectEqualStrings("model", config.weight_prefix);
    try testing.expect(!config.isMoe());
    // KEY difference from qwen3: no QK-norm.
    try testing.expect(!config.has_qk_norm);
    try testing.expect(!config.has_pre_ff_norm);
    try testing.expect(!config.scale_embeddings);
    try testing.expectEqual(HiddenAct.silu, config.hidden_act);
    try testing.expectEqual(@as(u32, 128), config.head_dim); // 5120 / 40
    try testing.expectEqual(@as(u32, 8), config.quant_bits);
}

test "ModelConfig parses diffusion_gemma (DiffusionGemma 26B-A4B block diffusion)" {
    // Faithful subset of mlx-community/diffusiongemma-26B-A4B-it-4bit's
    // config.json. The trunk is the Gemma 4 26B-A4B MoE decoder (dual FFN,
    // sigma-MoE router, v_norm, K=V alias on full layers, proportional RoPE)
    // under weight prefix `model.decoder`; the diffusion-specific knobs ride
    // in the embedded `generation_config` object plus top-level canvas_length.
    const json =
        \\{
        \\  "model_type": "diffusion_gemma",
        \\  "canvas_length": 256,
        \\  "eos_token_id": [1, 106, 50],
        \\  "tie_word_embeddings": true,
        \\  "generation_config": {
        \\    "confidence_threshold": 0.005,
        \\    "max_denoising_steps": 48,
        \\    "pad_token_id": 0,
        \\    "sampler_config": {"_cls_name": "EntropyBoundSamplerConfig", "entropy_bound": 0.1},
        \\    "stability_threshold": 1,
        \\    "t_max": 0.8,
        \\    "t_min": 0.4
        \\  },
        \\  "text_config": {
        \\    "model_type": "diffusion_gemma_text",
        \\    "vocab_size": 262144,
        \\    "hidden_size": 2816,
        \\    "intermediate_size": 2112,
        \\    "moe_intermediate_size": 704,
        \\    "num_experts": 128,
        \\    "top_k_experts": 8,
        \\    "num_hidden_layers": 30,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 8,
        \\    "num_global_key_value_heads": 2,
        \\    "head_dim": 256,
        \\    "global_head_dim": 512,
        \\    "final_logit_softcapping": 30.0,
        \\    "hidden_activation": "gelu_pytorch_tanh",
        \\    "rms_norm_eps": 1e-06,
        \\    "max_position_embeddings": 262144,
        \\    "sliding_window": 1024,
        \\    "tie_word_embeddings": true,
        \\    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        \\    "rope_parameters": {
        \\      "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
        \\      "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
        \\    },
        \\    "use_bidirectional_attention": "vision"
        \\  },
        \\  "vision_config": {"model_type": "gemma4_vision", "hidden_size": 1152, "num_hidden_layers": 27},
        \\  "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    // model_type stays distinct — it drives diffusion generation dispatch —
    // but the trunk inherits every gemma4 layer-structure flag.
    try testing.expectEqualStrings("diffusion_gemma", config.model_type);
    try testing.expectEqualStrings("model.decoder", config.weight_prefix);
    try testing.expect(config.isDiffusion());
    try testing.expect(config.has_v_norm);
    try testing.expect(config.has_pre_ff_norm);
    try testing.expect(config.has_qk_norm);
    try testing.expect(!config.norm_has_offset);
    try testing.expect(config.scale_embeddings);
    try testing.expect(config.tie_word_embeddings);
    // Full-attention layers ship no v_proj: V = param-free-norm(k_proj out).
    try testing.expect(config.attention_k_eq_v);
    // MoE trunk
    try testing.expect(config.isMoe());
    try testing.expectEqual(@as(u32, 128), config.num_experts);
    try testing.expectEqual(@as(u32, 8), config.num_experts_per_tok);
    try testing.expectEqual(@as(u32, 704), config.moe_intermediate_size);
    // Dual head geometry + per-type RoPE
    try testing.expectEqual(@as(u32, 512), config.global_head_dim);
    try testing.expectEqual(@as(u32, 2), config.num_global_key_value_heads);
    try testing.expect(config.rope_proportional);
    try testing.expectApproxEqAbs(@as(f32, 0.25), config.partial_rotary_factor_global, 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 1000000.0), config.rope_theta, 0.5);
    try testing.expectApproxEqAbs(@as(f32, 10000.0), config.rope_local_base_freq, 0.5);
    try testing.expect(config.has_explicit_layer_types);
    try testing.expect(config.isGlobalLayer(5));
    try testing.expect(!config.isGlobalLayer(4));
    try testing.expectEqual(@as(u32, 1024), config.sliding_window);
    try testing.expectApproxEqAbs(@as(f32, 30.0), config.final_logit_softcapping, 0.001);
    // EOS set {eos, end_of_turn, +1}
    try testing.expectEqual(@as(u32, 3), config.num_eos_tokens);
    try testing.expect(config.isEosToken(1));
    try testing.expect(config.isEosToken(106));
    try testing.expect(config.isEosToken(50));
    // Diffusion generation knobs from the embedded generation_config
    try testing.expectEqual(@as(u32, 256), config.canvas_length);
    try testing.expectEqual(@as(u32, 48), config.diffusion_max_steps);
    try testing.expectApproxEqAbs(@as(f32, 0.4), config.diffusion_t_min, 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 0.8), config.diffusion_t_max, 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 0.1), config.diffusion_entropy_bound, 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 0.005), config.diffusion_confidence_threshold, 0.000001);
    try testing.expectEqual(@as(u32, 1), config.diffusion_stability_threshold);
    try testing.expectEqual(@as(u32, 0), config.diffusion_pad_token);
    // Vision tower (model.encoder.vision_tower.*) is not wired yet — the
    // diffusion arm must NOT advertise vision, or image requests would splice
    // embeddings into a tower-less forward.
    try testing.expect(!config.has_vision);
    try testing.expectEqual(@as(u32, 4), config.quant_bits);
}

test "shouldKeepWeightKey drops DiffusionGemma encoder vision tower (text-only v1)" {
    // DiffusionGemma nests its vision tower under model.encoder.* — distinct
    // from the bare vision_tower.* prefixes of earlier checkpoints. Until the
    // tower is wired, those tensors must be dropped even with load_vision on,
    // and ALWAYS dropped when vision is off.
    try testing.expect(!shouldKeepWeightKey("model.encoder.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight", false));
    try testing.expect(!shouldKeepWeightKey("model.encoder.embed_vision.embedding_projection.weight", false));
    // Trunk + diffusion weights always survive.
    try testing.expect(shouldKeepWeightKey("model.decoder.layers.0.experts.gate_up_proj.weight", false));
    try testing.expect(shouldKeepWeightKey("model.decoder.self_conditioning.gate_proj.weight", false));
    try testing.expect(shouldKeepWeightKey("model.encoder.language_model.layers.0.layer_scalar", false));
}

test "parseGenerationDefaultsFromJson: reads model sampling recommendations" {
    // Verbatim shape of Qwen3.6 / Gemma 4 checkpoints' generation_config.json.
    const json =
        \\{"bos_token_id": 248044, "do_sample": true, "temperature": 1.0, "top_k": 20, "top_p": 0.95}
    ;
    const gd = parseGenerationDefaultsFromJson(json);
    try testing.expectEqual(@as(?f32, 1.0), gd.temperature);
    try testing.expectEqual(@as(?f32, 0.95), gd.top_p);
    try testing.expectEqual(@as(?u32, 20), gd.top_k);
}

test "parseGenerationDefaultsFromJson: missing keys and malformed input give nulls" {
    const partial = parseGenerationDefaultsFromJson("{\"eos_token_id\": [1, 2]}");
    try testing.expectEqual(@as(?f32, null), partial.temperature);
    try testing.expectEqual(@as(?f32, null), partial.top_p);
    try testing.expectEqual(@as(?u32, null), partial.top_k);

    const broken = parseGenerationDefaultsFromJson("not json at all");
    try testing.expectEqual(@as(?f32, null), broken.temperature);

    // Out-of-range values are dropped, not clamped — a corrupt config must
    // not silently pin sampling to an extreme.
    const insane = parseGenerationDefaultsFromJson("{\"temperature\": 99.0, \"top_p\": 7.0, \"top_k\": -5}");
    try testing.expectEqual(@as(?f32, null), insane.temperature);
    try testing.expectEqual(@as(?f32, null), insane.top_p);
    try testing.expectEqual(@as(?u32, null), insane.top_k);
}
