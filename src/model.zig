const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const tokenizer_mod = @import("tokenizer.zig");

pub const HiddenAct = enum { gelu_approx, silu, relu_sq };

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

    // Quantization
    quant_bits: u32 = 4,
    quant_group_size: u32 = 64,

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

    // MTP (Multi-Token Prediction) self-speculative decoding head.
    // Populated only for architectures that ship MTP weights (Qwen3.5+, Qwen3-Next);
    // other model families ignore `num_nextn_predict_layers` even if present in JSON.
    num_mtp_predict_layers: u32 = 0,
    has_mtp: bool = false,

    pub fn isGlobalLayer(self: ModelConfig, layer_idx: u32) bool {
        if (!self.has_sliding_window) return true;
        if (self.has_explicit_layer_types and layer_idx < 128) {
            return self.layer_is_global[layer_idx];
        }
        return (layer_idx % self.sliding_window_pattern) == 0;
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

    return parseConfigFromJson(allocator, content);
}

/// Read MTP layer count from a Qwen-style config object. Accepts both
/// `mtp_num_hidden_layers` (HF Qwen3.5 actual) and `num_nextn_predict_layers`
/// (Qwen3-Next spec; sometimes used). Sets `has_mtp = true` only when >0.
fn readMtpLayerCount(cfg_obj: std.json.ObjectMap, config: *ModelConfig) void {
    const candidate_keys = [_][]const u8{ "mtp_num_hidden_layers", "num_nextn_predict_layers" };
    for (candidate_keys) |key| {
        if (cfg_obj.get(key)) |v| {
            if (v == .integer and v.integer > 0) {
                config.num_mtp_predict_layers = @intCast(v.integer);
                config.has_mtp = true;
                return;
            }
        }
    }
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
        }
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
    } else if (std.mem.eql(u8, model_type, "gemma4") or std.mem.eql(u8, model_type, "gemma4_text")) {
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
        // MTP head metadata: HF Qwen3.5 ships `mtp_num_hidden_layers`; the spec
        // also documents `num_nextn_predict_layers` (Qwen3-Next style). Accept both.
        // Note: on MLX-converted checkpoints the field may be present as
        // metadata even though the MTP weight tensors were stripped during
        // conversion; weight-binder will fail loudly via getMtpWeight in that case.
        readMtpLayerCount(cfg_obj, &config);
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
        readMtpLayerCount(cfg_obj, &config);
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

/// True if the safetensors weight `key` should be retained for the text/MTP
/// forward pass. Audio is always dropped; vision is dropped unless
/// `load_vision` is set. MTP keys (`*.mtp.*`) are always retained — they live
/// under the standard text prefix and are needed for self-speculative decoding.
pub fn shouldKeepWeightKey(key: []const u8, load_vision: bool) bool {
    const is_vision = std.mem.startsWith(u8, key, "vision_tower.") or
        std.mem.startsWith(u8, key, "embed_vision.") or
        std.mem.startsWith(u8, key, "multi_modal_projector.") or
        std.mem.startsWith(u8, key, "language_model.multi_modal_projector.");
    const is_audio = std.mem.startsWith(u8, key, "audio_tower.") or
        std.mem.startsWith(u8, key, "language_model.audio_multi_modal_projector.");
    if (is_audio) return false;
    if (is_vision and !load_vision) return false;
    return true;
}

// ── Tests ──

const testing = std.testing;

test "ModelConfig defaults" {
    const config = ModelConfig{};
    try testing.expectEqual(@as(u32, 0), config.num_eos_tokens);
    try testing.expectEqual(@as(u32, 0), config.max_position_embeddings);
    try testing.expectEqual(@as(u32, 4), config.quant_bits);
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
    var config = ModelConfig{};
    config.has_sliding_window = true;
    config.sliding_window_pattern = 6;
    // Layer 0: 0 % 6 == 0 → global
    try testing.expect(config.isGlobalLayer(0));
    // Layer 1: 1 % 6 != 0 → local
    try testing.expect(!config.isGlobalLayer(1));
    // Layer 6: 6 % 6 == 0 → global
    try testing.expect(config.isGlobalLayer(6));
    // Layer 12: 12 % 6 == 0 → global
    try testing.expect(config.isGlobalLayer(12));
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

// ── MTP (Multi-Token Prediction) ──────────────────────────────────────────
//
// Qwen3.5 / Qwen3.6 / Qwen3-Next ship a native MTP head in their safetensors.
// `num_nextn_predict_layers` in config.json gives the count (always 1 in shipped
// checkpoints today). The head is a single transformer block trained to predict
// token N+2 from `(hidden_state_N, embed(token_N+1))`. Used for self-speculative
// decoding: draft with the MTP head, verify with a length-2 main forward.

test "ModelConfig MTP defaults" {
    const config = ModelConfig{};
    try testing.expectEqual(@as(u32, 0), config.num_mtp_predict_layers);
    try testing.expect(!config.has_mtp);
}

test "parseConfigFromJson reads mtp_num_hidden_layers (HF Qwen3.5 actual field)" {
    // Real Qwen3.5 checkpoints from Hugging Face use `mtp_num_hidden_layers`
    // inside text_config, not the spec-style `num_nextn_predict_layers`. Both
    // must be honored.
    const json =
        \\{
        \\  "model_type": "qwen3_5",
        \\  "text_config": {
        \\    "model_type": "qwen3_5_text",
        \\    "vocab_size": 248320,
        \\    "hidden_size": 2560,
        \\    "intermediate_size": 9216,
        \\    "num_hidden_layers": 32,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 4,
        \\    "head_dim": 256,
        \\    "mtp_num_hidden_layers": 1
        \\  }
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(@as(u32, 1), config.num_mtp_predict_layers);
    try testing.expect(config.has_mtp);
}

test "parseConfigFromJson reads num_nextn_predict_layers for qwen3_5_moe" {
    const json =
        \\{
        \\  "model_type": "qwen3_5_moe",
        \\  "vocab_size": 151936,
        \\  "hidden_size": 2048,
        \\  "intermediate_size": 6144,
        \\  "num_hidden_layers": 48,
        \\  "num_attention_heads": 16,
        \\  "num_key_value_heads": 2,
        \\  "head_dim": 128,
        \\  "num_nextn_predict_layers": 1
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(@as(u32, 1), config.num_mtp_predict_layers);
    try testing.expect(config.has_mtp);
    try testing.expectEqualStrings("qwen3_5_moe", config.model_type);
}

test "parseConfigFromJson reads num_nextn_predict_layers for qwen3_next" {
    const json =
        \\{
        \\  "model_type": "qwen3_next",
        \\  "vocab_size": 151936,
        \\  "hidden_size": 2048,
        \\  "intermediate_size": 5120,
        \\  "num_hidden_layers": 48,
        \\  "num_attention_heads": 16,
        \\  "num_key_value_heads": 2,
        \\  "head_dim": 128,
        \\  "num_nextn_predict_layers": 1
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(@as(u32, 1), config.num_mtp_predict_layers);
    try testing.expect(config.has_mtp);
    try testing.expectEqualStrings("qwen3_next", config.model_type);
}

test "parseConfigFromJson defaults num_nextn_predict_layers to 0 when absent" {
    const json =
        \\{
        \\  "model_type": "qwen3_5_moe",
        \\  "vocab_size": 151936,
        \\  "hidden_size": 2048,
        \\  "intermediate_size": 6144,
        \\  "num_hidden_layers": 48,
        \\  "num_attention_heads": 16,
        \\  "num_key_value_heads": 2,
        \\  "head_dim": 128
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(@as(u32, 0), config.num_mtp_predict_layers);
    try testing.expect(!config.has_mtp);
}

test "parseConfigFromJson does not enable MTP for non-Qwen architectures" {
    // num_nextn_predict_layers in a llama config (synthetic) should be ignored
    // — only Qwen variants currently have working MTP weights.
    const json =
        \\{
        \\  "model_type": "llama",
        \\  "vocab_size": 32000,
        \\  "hidden_size": 4096,
        \\  "intermediate_size": 11008,
        \\  "num_hidden_layers": 32,
        \\  "num_attention_heads": 32,
        \\  "num_key_value_heads": 32,
        \\  "head_dim": 128,
        \\  "num_nextn_predict_layers": 1
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(!config.has_mtp);
}

test "shouldKeepWeightKey accepts MTP head weights" {
    // MTP head weights live under `{prefix}.mtp.{layer_idx}.{name}` in HF
    // safetensors (e.g. `language_model.model.mtp.0.eh_proj.weight`). The
    // safetensors iterator filter must let them through (they're neither
    // vision nor audio) regardless of `load_vision`.
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
