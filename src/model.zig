const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

pub const HiddenAct = enum { gelu_approx, silu };

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

    // Context length from config.json (0 = unknown)
    max_position_embeddings: u32 = 0,

    // Stop tokens (populated from config.json)
    eos_token_ids: [8]u32 = .{0} ** 8,
    num_eos_tokens: u32 = 0,

    pub fn isGlobalLayer(self: ModelConfig, layer_idx: u32) bool {
        if (!self.has_sliding_window) return true;
        return (layer_idx % self.sliding_window_pattern) == 0;
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
};

pub fn parseConfig(allocator: std.mem.Allocator, model_dir: []const u8) !ModelConfig {
    const path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
    defer allocator.free(path);

    const file = try std.fs.openFileAbsolute(path, .{});
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 10 * 1024 * 1024);
    defer allocator.free(content);

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

    // MoE fields
    if (cfg_obj.get("num_experts")) |v| config.num_experts = @intCast(v.integer);
    if (cfg_obj.get("num_experts_per_tok")) |v| config.num_experts_per_tok = @intCast(v.integer);
    if (cfg_obj.get("moe_intermediate_size")) |v| config.moe_intermediate_size = @intCast(v.integer);
    if (cfg_obj.get("shared_expert_intermediate_size")) |v| config.shared_expert_intermediate_size = @intCast(v.integer);

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
    } else if (std.mem.eql(u8, model_type, "qwen3_5_moe") or
        std.mem.eql(u8, model_type, "qwen3_5") or
        std.mem.eql(u8, model_type, "qwen3_5_moe_text"))
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
pub fn loadWeights(allocator: std.mem.Allocator, model_dir: []const u8) !Weights {
    var weights = Weights.init(allocator);
    errdefer weights.deinit();

    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    var dir = try std.fs.openDirAbsolute(model_dir, .{ .iterate = true });
    defer dir.close();

    var file_count: u32 = 0;
    var it = dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;

        const path_slice = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, entry.name });
        defer allocator.free(path_slice);
        const path = try allocator.dupeZ(u8, path_slice);
        defer allocator.free(path);

        log.info("Loading {s}...\n", .{entry.name});
        try loadSafetensorsFile(allocator, &weights, path, s);
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

        // Skip non-text-model weights (vision_tower, multi_modal_projector, etc.)
        if (std.mem.startsWith(u8, key_str, "vision_tower.") or
            std.mem.startsWith(u8, key_str, "multi_modal_projector."))
        {
            _ = mlx.mlx_array_free(value);
            continue;
        }

        const owned_key = try allocator.dupe(u8, key_str);
        try weights.map.put(owned_key, value);
    }
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
