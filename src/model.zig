const std = @import("std");
const mlx = @import("mlx.zig");

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
    norm_has_offset: bool = true, // Gemma: (1+weight)*rms_norm(x), Llama-family: weight*rms_norm(x)
    scale_embeddings: bool = true, // Gemma: multiply by sqrt(hidden_size)
    has_pre_ff_norm: bool = true, // Gemma has 4 norms per layer, Llama-family has 2
    has_qk_norm: bool = true,

    // Stop tokens (populated from config.json)
    eos_token_ids: [8]u32 = .{0} ** 8,
    num_eos_tokens: u32 = 0,

    pub fn isGlobalLayer(self: ModelConfig, layer_idx: u32) bool {
        if (!self.has_sliding_window) return true;
        return (layer_idx % self.sliding_window_pattern) == 0;
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
    if (cfg_obj.get("rms_norm_eps")) |v| config.rms_norm_eps = jsonFloat(v);
    if (cfg_obj.get("rope_theta")) |v| config.rope_theta = jsonFloat(v);
    if (cfg_obj.get("query_pre_attn_scalar")) |v| config.query_pre_attn_scalar = @intCast(v.integer);

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
            // Gemma default: read from rope_scaling.factor, but if not found, use 8.0
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
        // Gemma default stop tokens
        if (config.num_eos_tokens == 0) {
            config.addEosToken(1); // EOS
            config.addEosToken(106); // END_OF_TURN
        }
    } else {
        // Llama-family defaults (qwen3, llama, mistral, etc.)
        // Use string literals to avoid use-after-free (model_type points into parsed JSON)
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
        config.rope_local_base_freq = config.rope_theta; // no dual RoPE

        // hidden_act
        if (cfg_obj.get("hidden_act")) |v| {
            if (v == .string) {
                if (std.mem.eql(u8, v.string, "silu")) {
                    config.hidden_act = .silu;
                } else if (std.mem.eql(u8, v.string, "gelu_pytorch_tanh")) {
                    config.hidden_act = .gelu_approx;
                }
            }
        }

        // query_pre_attn_scalar defaults to head_dim for non-Gemma
        if (cfg_obj.get("query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }

        // Qwen3 has Q/K norms
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

        std.debug.print("Loading {s}...\n", .{entry.name});
        try loadSafetensorsFile(allocator, &weights, path, s);
        file_count += 1;
    }

    std.debug.print("Loaded {d} weights from {d} file(s)\n", .{ weights.count(), file_count });
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
