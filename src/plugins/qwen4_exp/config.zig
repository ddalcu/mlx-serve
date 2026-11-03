//! Qwen3.8-Flash-Next (`qwen4_exp`, `qwen4_exp_text`): the claim, and the parse into `Parsed` plus the family's
//! `Config`, read with the engine's semantics. MLX-free (discovery runs it). The fields carry the engine's
//! `ModelConfig` names, so `model_config.zig` hands the trunk module its config one to one.

const std = @import("std");
const env = @import("../../env.zig");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/plugin.zig");
const generation = @import("../../sdk/config/generation.zig");
const ple_table = @import("ple_table.zig");
const mrope = @import("../../sdk/vision/mrope.zig");

pub const model_types = [_][]const u8{ "qwen4_exp", "qwen4_exp_text" };

pub const claims = sdk.claimsByType(&model_types);

pub const plugin = sdk.Plugin{
    .name = "qwen4_exp",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw"},
};

pub const QuantMode = enum { affine, nvfp4, mxfp4, mxfp8, gguf };
pub const NormConvention = enum { delta, folded };

/// `NgramHash`'s fixed arrays (`qwen4_exp.MAX_NGRAM_SIZE`, `MAX_HEADS`); pinned against them in `model_config.zig`.
pub const max_ngram_size = 8;
pub const max_heads = 32;

/// The family's parsed config. Defaults are the engine's, so a key a pack omits reads as it always has.
pub const Config = struct {
    vocab_size: u32 = 262208,
    hidden_size: u32 = 3840,
    intermediate_size: u32 = 15360,
    intermediate_size_declared: bool = false,
    num_hidden_layers: u32 = 48,
    num_attention_heads: u32 = 16,
    num_key_value_heads: u32 = 8,
    head_dim: u32 = 256,
    max_position_embeddings: u32 = 0,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 1000000.0,
    query_pre_attn_scalar: u32 = 256,
    partial_rotary_factor: f32 = 1.0,
    sliding_window: u32 = 1024,
    sliding_window_pattern: u32 = 6,
    has_explicit_layer_types: bool = false,
    layer_is_global: [128]bool = @splat(false),
    tie_word_embeddings: bool = false,
    bos_token_id: ?u32 = null,
    eos_token_ids: [8]u32 = @splat(0),
    num_eos_tokens: u32 = 0,

    quant_bits: u32 = 0,
    quant_group_size: u32 = 64,
    quant_mode: QuantMode = .affine,
    /// An NVIDIA ModelOpt pack served as published.
    modelopt: bool = false,

    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    /// A Sushi pack: routed experts in EXL3 (`expert_quant`, read by `toModelConfig`).
    expert_quant: bool = false,
    moe_intermediate_size: u32 = 0,
    shared_expert_intermediate_size: u32 = 0,

    linear_num_key_heads: u32 = 0,
    linear_num_value_heads: u32 = 0,
    linear_key_head_dim: u32 = 128,
    linear_value_head_dim: u32 = 128,
    linear_conv_kernel_dim: u32 = 4,
    full_attention_interval: u32 = 0,
    /// The GDN state and decay gate in f32 (`MLX_SERVE_GDN_STATE_F32`, default on).
    ssm_state_f32: bool = false,

    rope_yarn: bool = false,
    yarn_factor: f32 = 1.0,
    yarn_orig_max_pos: u32 = 0,
    yarn_beta_fast: f32 = 32.0,
    yarn_beta_slow: f32 = 1.0,
    yarn_attention_factor: f32 = 1.0,
    yarn_truncate: bool = true,

    hc_count: u32 = 0,
    hc_lowrank: u32 = 0,
    ple_layer_idx: i32 = -1,
    ple_embed_dim: u32 = 0,
    ple_conv_kernel: u32 = 4,
    ngram_size: u32 = 3,
    heads_per_ngram: u32 = 8,
    ngram_vocab_base: u64 = 20_000_000,
    ngram_vocab_divisor: u32 = 128,
    ngram_seed: u64 = 1234,
    split_ngram_parts: u32 = 128,
    qwen4_norm_convention: ?NormConvention = null,
    indexer_n_heads: u32 = 0,
    indexer_head_dim: u32 = 0,
    indexer_budget: u32 = 0,
    indexer_compress_ratio: u32 = 0,
    /// The text config's first eos: the n-gram hash's segment reset token.
    ngram_eos: u32 = 0,

    // The generic `vision_config` block the engine reads for every arch (Gemma's names).
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
    vision_mm_embed_dim: u32 = 0,
    vision_model_patch_size: u32 = 0,
    vision_mm_posemb_size: u32 = 0,
    image_token_id: u32 = 0,
    boi_token_id: u32 = 0,
    eoi_token_id: u32 = 0,

    // The Qwen3-VL tower and its M-RoPE.
    qwen_vision: bool = false,
    qv_depth: u32 = 0,
    qv_hidden: u32 = 0,
    qv_heads: u32 = 0,
    qv_head_dim: u32 = 0,
    qv_intermediate: u32 = 0,
    qv_patch: u32 = 16,
    qv_temporal_patch: u32 = 2,
    qv_merge: u32 = 2,
    qv_num_pos_emb: u32 = 0,
    qv_out_hidden: u32 = 0,
    qv_min_pixels: u32 = 0,
    qv_max_pixels: u32 = 0,
    mv_max_image_tokens: u32 = 0,
    mrope_section: [3]u32 = .{ 0, 0, 0 },
    mrope_interleaved: bool = false,
    video_token_id: u32 = 0,
    vision_start_token_id: u32 = 0,
    vision_end_token_id: u32 = 0,

    gen_temperature: ?f32 = null,
    gen_top_p: ?f32 = null,
    gen_top_k: ?u32 = null,
    gen_min_p: ?f32 = null,
    gen_enable_thinking: ?bool = null,

    /// The pack's index names n-gram shards beside the weights (oMLX, ModelOpt) instead of `ngram_table.bin`.
    embedded_ngram: bool = false,
    /// The M-RoPE spectrum `Parsed.mrope` points at (the YaRN table when the rope scales).
    mrope_inv_freq: [64]f64 = @splat(0),

    fn addEos(self: *Config, id: u32) void {
        if (self.num_eos_tokens < self.eos_token_ids.len) {
            self.eos_token_ids[self.num_eos_tokens] = id;
            self.num_eos_tokens += 1;
        }
    }

    fn mergeEos(self: *Config, id: u32) void {
        if (std.mem.indexOfScalar(u32, self.eos_token_ids[0..self.num_eos_tokens], id) == null) self.addEos(id);
    }

    /// Full attention on layer `li`, else a GatedDeltaNet layer: the interval decides (the engine's
    /// `isLinearLayer`; `layer_types` is not read on this arch).
    pub fn isFullLayer(self: *const Config, li: u32) bool {
        if (self.full_attention_interval == 0) return true;
        return (li + 1) % self.full_attention_interval == 0;
    }

    pub fn attnLayers(self: *const Config) u32 {
        var n: u32 = 0;
        for (0..self.num_hidden_layers) |li| {
            if (self.isFullLayer(@intCast(li))) n += 1;
        }
        return n;
    }

    /// The longest sequence the rope resolves: `original_max_position_embeddings x factor` under YaRN.
    pub fn contextCap(self: *const Config) u32 {
        const declared = self.max_position_embeddings;
        if (!self.rope_yarn) return declared;
        const scaled: f64 = @floor(@as(f64, @floatFromInt(self.yarn_orig_max_pos)) * @as(f64, @floatCast(self.yarn_factor)));
        const max_u32: f64 = @floatFromInt(std.math.maxInt(u32));
        const window: u32 = if (scaled >= max_u32) std.math.maxInt(u32) else @intFromFloat(scaled);
        return if (declared == 0) window else @min(window, declared);
    }
};

pub const ParseError = error{
    InvalidConfigField,
    UnsupportedQuantMode,
    UnsupportedQuantBits,
    YarnRopeNeedsOriginalMaxPos,
    InvalidRopeScalingFactor,
    InvalidQwen4NormConvention,
    InvalidQwen4ConfigField,
    InvalidQwen4PleLayer,
    InvalidQwen4NgramSize,
    InvalidQwen4NgramHeads,
    InvalidQwen4NgramVocab,
    InvalidQwen4PleGeometry,
    InvalidQwen4Indexer,
    /// A shape this plugin does not serve yet (the legacy path keeps it).
    Unsupported,
    OutOfMemory,
};

// Readers with the engine parser's semantics: a JSON null reads as an omitted key.

fn field(obj: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    const v = obj.get(key) orelse return null;
    return if (v == .null) null else v;
}

fn u32Of(v: std.json.Value) ParseError!u32 {
    if (v != .integer) return error.InvalidConfigField;
    return std.math.cast(u32, v.integer) orelse error.InvalidConfigField;
}

fn f32Of(v: std.json.Value) ParseError!f32 {
    const f: f32 = switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |x| @floatCast(x),
        else => 0.0,
    };
    if (!std.math.isFinite(f)) return error.InvalidConfigField;
    return f;
}

/// `key` into `out` when present (any non-integer refused).
fn readU32(obj: std.json.ObjectMap, key: []const u8, out: *u32) ParseError!void {
    if (field(obj, key)) |v| out.* = try u32Of(v);
}

/// `key` into `out` when it is an integer; another type leaves the field as it was.
fn readU32If(obj: std.json.ObjectMap, key: []const u8, out: *u32) ParseError!void {
    if (field(obj, key)) |v| if (v == .integer) {
        out.* = try u32Of(v);
    };
}

fn readF32(obj: std.json.ObjectMap, key: []const u8, out: *f32) ParseError!void {
    if (field(obj, key)) |v| out.* = try f32Of(v);
}

fn readBool(obj: std.json.ObjectMap, key: []const u8, out: *bool) void {
    if (field(obj, key)) |v| if (v == .bool) {
        out.* = v.bool;
    };
}

/// A qwen4 key read strictly: a non-negative integer or a named refusal.
fn q4U64(obj: std.json.ObjectMap, key: []const u8) ParseError!?u64 {
    const v = field(obj, key) orelse return null;
    if (v != .integer or v.integer < 0) return error.InvalidQwen4ConfigField;
    return @intCast(v.integer);
}

fn q4U32(obj: std.json.ObjectMap, key: []const u8) ParseError!?u32 {
    const v = try q4U64(obj, key) orelse return null;
    return std.math.cast(u32, v) orelse error.InvalidQwen4ConfigField;
}

/// Keys of another family's shapes the engine's generic block would read into fields this arch never serves.
const foreign_keys = [_][]const u8{
    "use_bidirectional_attention", "global_head_dim", "num_global_key_value_heads", "num_kv_shared_layers",
    "attention_k_eq_v",            "final_logit_softcapping", "hidden_size_per_layer_input", "top_k_experts",
};
const foreign_root_keys = [_][]const u8{ "pooling_mode", "audio_config", "audio_token_id", "boa_token_id", "eoa_token_index", "eoa_token_id" };

/// config.json into `c`, as the engine's generic block and its qwen4_exp arm read it.
pub fn parseJson(c: *Config, outer: std.json.ObjectMap) ParseError!void {
    const root = outer;
    const cfg = if (field(root, "text_config")) |tc| (if (tc == .object) tc.object else return error.InvalidConfigField) else root;
    for (foreign_keys) |k| if (field(cfg, k) != null) return error.Unsupported;
    for (foreign_root_keys) |k| if (field(root, k) != null) return error.Unsupported;

    try readU32(cfg, "vocab_size", &c.vocab_size);
    try readU32(cfg, "hidden_size", &c.hidden_size);
    if (field(cfg, "intermediate_size")) |v| {
        c.intermediate_size = try u32Of(v);
        c.intermediate_size_declared = true;
    }
    try readU32(cfg, "num_hidden_layers", &c.num_hidden_layers);
    try readU32(cfg, "num_attention_heads", &c.num_attention_heads);
    try readU32(cfg, "num_key_value_heads", &c.num_key_value_heads);
    try readU32(cfg, "head_dim", &c.head_dim);
    try readU32(cfg, "max_position_embeddings", &c.max_position_embeddings);
    try readF32(cfg, "rms_norm_eps", &c.rms_norm_eps);
    try readF32(cfg, "rope_theta", &c.rope_theta);
    try readU32(cfg, "query_pre_attn_scalar", &c.query_pre_attn_scalar);
    try readU32If(cfg, "num_experts", &c.num_experts);
    try readU32If(cfg, "num_experts_per_tok", &c.num_experts_per_tok);
    c.expert_quant = field(root, "expert_quant") != null;
    try readU32If(cfg, "moe_intermediate_size", &c.moe_intermediate_size);
    try readU32If(cfg, "shared_expert_intermediate_size", &c.shared_expert_intermediate_size);
    try readU32(cfg, "linear_num_key_heads", &c.linear_num_key_heads);
    try readU32(cfg, "linear_num_value_heads", &c.linear_num_value_heads);
    try readU32(cfg, "linear_key_head_dim", &c.linear_key_head_dim);
    try readU32(cfg, "linear_value_head_dim", &c.linear_value_head_dim);
    try readU32(cfg, "linear_conv_kernel_dim", &c.linear_conv_kernel_dim);
    try readU32(cfg, "full_attention_interval", &c.full_attention_interval);
    if (field(cfg, "bos_token_id")) |v| if (v == .integer and v.integer >= 0) {
        c.bos_token_id = try u32Of(v);
    };
    if (field(cfg, "rope_parameters")) |rp| if (rp == .object) {
        if (field(rp.object, "full_attention") != null or field(rp.object, "sliding_attention") != null) return error.Unsupported;
        try readF32(rp.object, "rope_theta", &c.rope_theta);
        try readF32(rp.object, "partial_rotary_factor", &c.partial_rotary_factor);
    };
    // The engine reads the window even though this arch never slides (`has_sliding_window` is forced off).
    if (cfg.get("sliding_window")) |v| if (v != .null) {
        c.sliding_window = try u32Of(v);
    };
    try readU32(cfg, "sliding_window_pattern", &c.sliding_window_pattern);
    if (field(cfg, "rope_scaling")) |rs| if (rs == .object) {
        var factor: f32 = 1.0;
        try readF32(rs.object, "factor", &factor);
    };
    if (field(cfg, "rope_local_base_freq")) |v| _ = try f32Of(v);
    if (field(cfg, "layer_types")) |lt| if (lt == .array) {
        c.has_explicit_layer_types = true;
        for (lt.array.items, 0..) |item, i| {
            if (i >= 128) break;
            if (item == .string) c.layer_is_global[i] = std.mem.eql(u8, item.string, "full_attention");
        }
    };
    readBool(root, "tie_word_embeddings", &c.tie_word_embeddings);
    readBool(cfg, "tie_word_embeddings", &c.tie_word_embeddings);
    if (c.max_position_embeddings == 0) try readU32If(root, "max_position_embeddings", &c.max_position_embeddings);

    try parseQuantization(c, outer);

    if (field(root, "eos_token_id")) |v| switch (v) {
        .integer => |i| c.addEos(std.math.cast(u32, i) orelse return error.InvalidConfigField),
        .array => |arr| for (arr.items) |item| if (item == .integer) c.addEos(try u32Of(item)),
        else => {},
    };

    try parseVisionConfig(c, root);
    try readU32If(root, "image_token_id", &c.image_token_id);
    if (field(root, "image_token_index")) |v| if (v == .integer and c.image_token_id == 0) {
        c.image_token_id = try u32Of(v);
    };
    try readU32If(root, "boi_token_id", &c.boi_token_id);
    try readU32If(root, "eoi_token_id", &c.eoi_token_id);

    // The qwen4_exp arm.
    if (root.get("qwen4_norm_convention")) |v| {
        if (v != .string) return error.InvalidQwen4NormConvention;
        c.qwen4_norm_convention = std.meta.stringToEnum(NormConvention, v.string) orelse return error.InvalidQwen4NormConvention;
    }
    c.ssm_state_f32 = gdnStateF32();
    if (field(cfg, "query_pre_attn_scalar") == null) c.query_pre_attn_scalar = c.head_dim;
    c.hc_count = 4;
    c.hc_lowrank = 320;
    c.ple_embed_dim = c.hidden_size;
    try parseQwenVision(c, root, cfg);
    try parseYarn(c, cfg);
    if (try q4U32(cfg, "hc_count")) |v| c.hc_count = v;
    if (try q4U32(cfg, "hc_lowrank")) |v| c.hc_lowrank = v;
    {
        const v = field(cfg, "ple_layer_ids") orelse return error.InvalidQwen4PleLayer;
        if (v != .array or v.array.items.len != 1 or v.array.items[0] != .integer) return error.InvalidQwen4PleLayer;
        const id = v.array.items[0].integer;
        if (id < 1 or id > @as(i64, c.num_hidden_layers)) return error.InvalidQwen4PleLayer;
        c.ple_layer_idx = std.math.cast(i32, id - 1) orelse return error.InvalidQwen4PleLayer;
    }
    if (try q4U32(cfg, "ple_embed_dim")) |v| c.ple_embed_dim = v;
    if (try q4U32(cfg, "ple_conv_kernel_size")) |v| c.ple_conv_kernel = v;
    if (try q4U32(cfg, "ngram_size")) |v| c.ngram_size = v;
    if (try q4U32(cfg, "heads_per_ngram")) |v| c.heads_per_ngram = v;
    if (try q4U64(cfg, "ngram_vocab_size_base")) |v| c.ngram_vocab_base = v;
    if (try q4U32(cfg, "make_ngram_vocab_size_divisible_by")) |v| c.ngram_vocab_divisor = v;
    if (try q4U64(cfg, "seed")) |v| c.ngram_seed = v;
    if (try q4U32(cfg, "split_ngram_parts")) |v| c.split_ngram_parts = v;
    if (try q4U32(cfg, "indexer_n_heads")) |v| c.indexer_n_heads = v;
    if (try q4U32(cfg, "indexer_head_dim")) |v| c.indexer_head_dim = v;
    if (try q4U32(cfg, "indexer_budget")) |v| c.indexer_budget = v;
    if (try q4U32(cfg, "indexer_compress_ratio")) |v| c.indexer_compress_ratio = v;
    if (field(cfg, "eos_token_id")) |v| {
        switch (v) {
            .integer => |i| c.ngram_eos = std.math.cast(u32, i) orelse return error.InvalidConfigField,
            .array => |arr| if (arr.items.len > 0 and arr.items[0] == .integer) {
                c.ngram_eos = try u32Of(arr.items[0]);
            },
            else => {},
        }
        if (c.num_eos_tokens == 0) c.addEos(c.ngram_eos);
    }
    try validate(c);

    if (field(cfg, "linear_num_key_heads") != null or field(cfg, "linear_num_value_heads") != null or
        field(cfg, "linear_key_head_dim") != null or field(cfg, "linear_value_head_dim") != null or
        field(cfg, "linear_conv_kernel_dim") != null)
    {
        if (c.linear_num_key_heads == 0 or c.linear_num_value_heads == 0 or c.linear_num_value_heads % c.linear_num_key_heads != 0)
            return error.InvalidConfigField;
    }
}

fn parseQuantization(c: *Config, outer: std.json.ObjectMap) ParseError!void {
    if (field(outer, "quantization")) |qv| {
        if (qv != .object) return error.InvalidConfigField;
        const q = qv.object;
        try readU32(q, "bits", &c.quant_bits);
        try readU32(q, "group_size", &c.quant_group_size);
        if (field(q, "mode")) |m| if (m == .string) {
            c.quant_mode = std.meta.stringToEnum(QuantMode, m.string) orelse return error.UnsupportedQuantMode;
        };
        // Per-weight overrides carry their own mode: an unknown one is refused by name.
        var it = q.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.* != .object) continue;
            const mode = kv.value_ptr.object.get("mode") orelse continue;
            if (mode == .string and std.meta.stringToEnum(QuantMode, mode.string) == null) return error.UnsupportedQuantMode;
        }
        if (c.quant_mode == .affine and c.quant_bits != 0) switch (c.quant_bits) {
            2, 3, 4, 5, 6, 8 => {},
            else => return error.UnsupportedQuantBits,
        };
    }
    if (field(outer, "quantization_config")) |qc| if (qc == .object) if (qc.object.get("quant_method")) |m| {
        if (m == .string and std.mem.eql(u8, m.string, "modelopt")) {
            c.modelopt = true;
            c.quant_bits = 4;
            c.quant_group_size = 16;
            c.quant_mode = .nvfp4;
        }
    };
}

/// The generic `vision_config` block (every arch's): its presence marks a tower.
fn parseVisionConfig(c: *Config, root: std.json.ObjectMap) ParseError!void {
    const vv = field(root, "vision_config") orelse return;
    if (vv != .object) return;
    c.has_vision = true;
    const vc = vv.object;
    const ints = .{
        .{ "hidden_size", "vision_hidden_size" },          .{ "num_hidden_layers", "vision_num_layers" },
        .{ "num_attention_heads", "vision_num_heads" },    .{ "head_dim", "vision_head_dim" },
        .{ "global_head_dim", "vision_head_dim" },          .{ "intermediate_size", "vision_intermediate_size" },
        .{ "patch_size", "vision_patch_size" },             .{ "pooling_kernel_size", "vision_pooling_kernel" },
        .{ "default_output_length", "vision_soft_tokens" }, .{ "position_embedding_size", "vision_position_embedding_size" },
    };
    inline for (ints) |kf| try readU32If(vc, kf[0], &@field(c, kf[1]));
    if (field(vc, "rope_parameters")) |rp| if (rp == .object) try readF32(rp.object, "rope_theta", &c.vision_rope_theta);
    readBool(vc, "use_clipped_linears", &c.vision_use_clipped_linears);
    try readU32If(vc, "mm_embed_dim", &c.vision_mm_embed_dim);
    try readU32If(vc, "model_patch_size", &c.vision_model_patch_size);
    try readU32If(vc, "num_soft_tokens", &c.vision_soft_tokens);
    try readU32If(vc, "mm_posemb_size", &c.vision_mm_posemb_size);
}

/// The Qwen3-VL tower geometry, the interleaved M-RoPE sections and the vision token ids.
fn parseQwenVision(c: *Config, root: std.json.ObjectMap, cfg: std.json.ObjectMap) ParseError!void {
    if (field(root, "vision_config")) |vv| if (vv == .object) {
        c.qwen_vision = true;
        const vc = vv.object;
        const ints = .{
            .{ "depth", "qv_depth" },                       .{ "hidden_size", "qv_hidden" },          .{ "num_heads", "qv_heads" },
            .{ "intermediate_size", "qv_intermediate" },    .{ "patch_size", "qv_patch" },            .{ "temporal_patch_size", "qv_temporal_patch" },
            .{ "spatial_merge_size", "qv_merge" },          .{ "num_position_embeddings", "qv_num_pos_emb" }, .{ "out_hidden_size", "qv_out_hidden" },
        };
        inline for (ints) |kf| try readU32If(vc, kf[0], &@field(c, kf[1]));
        if (c.qv_heads != 0) c.qv_head_dim = c.qv_hidden / c.qv_heads;
        if (c.qv_out_hidden == 0) c.qv_out_hidden = c.hidden_size;
    };
    if (field(cfg, "rope_parameters")) |rp| if (rp == .object) {
        readBool(rp.object, "mrope_interleaved", &c.mrope_interleaved);
        if (field(rp.object, "mrope_section")) |v| if (v == .array) for (v.array.items, 0..) |item, i| {
            if (i >= 3) break;
            if (item == .integer) c.mrope_section[i] = try u32Of(item);
        };
    };
    try readU32If(root, "video_token_id", &c.video_token_id);
    try readU32If(root, "vision_start_token_id", &c.vision_start_token_id);
    try readU32If(root, "vision_end_token_id", &c.vision_end_token_id);
}

/// HF's default YaRN mscale for a scaling `factor`.
fn yarnMscale(factor: f32) f32 {
    if (factor <= 1.0) return 1.0;
    return 0.1 * @log(@as(f32, factor)) + 1.0;
}

/// A flat `rope_parameters` block of `rope_type: yarn`: one rotary table for attention, the indexer and the head.
fn parseYarn(c: *Config, cfg: std.json.ObjectMap) ParseError!void {
    const rpv = field(cfg, "rope_parameters") orelse return;
    if (rpv != .object) return;
    const rp = rpv.object;
    const rt = field(rp, "rope_type") orelse return;
    if (!(rt == .string and std.mem.eql(u8, rt.string, "yarn"))) return;
    try readU32If(rp, "original_max_position_embeddings", &c.yarn_orig_max_pos);
    if (c.yarn_orig_max_pos == 0) return error.YarnRopeNeedsOriginalMaxPos;
    try readF32(rp, "factor", &c.yarn_factor);
    try readF32(rp, "beta_fast", &c.yarn_beta_fast);
    try readF32(rp, "beta_slow", &c.yarn_beta_slow);
    readBool(rp, "truncate", &c.yarn_truncate);
    if (c.yarn_factor <= 0.0) return error.InvalidRopeScalingFactor;
    if (field(rp, "factor") == null and c.max_position_embeddings > c.yarn_orig_max_pos)
        c.yarn_factor = @as(f32, @floatFromInt(c.max_position_embeddings)) / @as(f32, @floatFromInt(c.yarn_orig_max_pos));
    if (field(rp, "attention_factor")) |v| {
        c.yarn_attention_factor = try f32Of(v);
    } else if (field(rp, "attn_factor")) |v| {
        c.yarn_attention_factor = yarnMscale(c.yarn_factor) * try f32Of(v);
    } else {
        c.yarn_attention_factor = yarnMscale(c.yarn_factor);
    }
    if (!std.math.isFinite(c.yarn_attention_factor)) return error.InvalidConfigField;
    c.rope_yarn = true;
}

/// Every bound the forward indexes a fixed array with or divides by, refused by name.
fn validate(c: *const Config) ParseError!void {
    if (c.ngram_size < 2 or c.ngram_size > max_ngram_size) return error.InvalidQwen4NgramSize;
    if (c.heads_per_ngram == 0) return error.InvalidQwen4NgramHeads;
    if (@as(u64, c.ngram_size - 1) * c.heads_per_ngram > max_heads) return error.InvalidQwen4NgramHeads;
    if (c.ngram_vocab_divisor == 0 or c.ngram_vocab_base < 2) return error.InvalidQwen4NgramVocab;
    if (c.ple_embed_dim == 0 or c.ple_embed_dim % ((c.ngram_size - 1) * c.heads_per_ngram) != 0 or c.split_ngram_parts == 0)
        return error.InvalidQwen4PleGeometry;
    if (c.indexer_n_heads > 0) {
        if (c.indexer_head_dim == 0 or c.indexer_compress_ratio == 0) return error.InvalidQwen4Indexer;
        if (c.indexer_budget < c.indexer_compress_ratio) return error.InvalidQwen4Indexer;
    }
    if (c.ple_layer_idx < 0 or c.ple_layer_idx >= c.num_hidden_layers) return error.InvalidQwen4PleLayer;
}

var gdn_state_f32_env: ?bool = null;

/// The reference keeps the GDN state f32 whatever `mamba_ssm_dtype` says; `MLX_SERVE_GDN_STATE_F32=0` = bf16.
fn gdnStateF32() bool {
    if (gdn_state_f32_env) |v| return v;
    const raw = env.get(.MLX_SERVE_GDN_STATE_F32);
    const on = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
    gdn_state_f32_env = on;
    return on;
}

// ── the pack's other files ──

fn positive(v: ?std.json.Value) ?u32 {
    const x = v orelse return null;
    return switch (x) {
        .integer => |i| if (i > 0 and i <= std.math.maxInt(u32)) @intCast(i) else null,
        else => null,
    };
}

/// `generation_config.json`: the sampling recommendations and its eos ids (merged after config.json's).
fn applyGeneration(c: *Config, dir: *const abi.PackDir, buf: []u8) void {
    var why: abi.Verdict = .{};
    const bytes = dir.readFile("generation_config.json", buf, &why) orelse return;
    var shell: abi.Shell = .{};
    generation.applyJson(bytes, &shell);
    const g = shell.gen;
    c.gen_temperature = if (std.math.isNan(g.temperature)) null else g.temperature;
    c.gen_top_p = if (std.math.isNan(g.top_p)) null else g.top_p;
    c.gen_top_k = if (g.top_k < 0) null else @intCast(g.top_k);
    c.gen_min_p = if (std.math.isNan(g.min_p)) null else g.min_p;
    c.gen_enable_thinking = switch (g.enable_thinking) {
        .unset => null,
        .on => true,
        .off => false,
    };
    for (shell.eos[0..shell.eos_count]) |id| c.mergeEos(id);
}

/// The Qwen processor's pixel bounds (`processor_config.json`, then `preprocessor_config.json`).
fn applyProcessor(c: *Config, dir: *const abi.PackDir, buf: []u8) void {
    var lo: ?u32 = null;
    var hi: ?u32 = null;
    var tokens: ?u32 = null;
    for ([_][]const u8{ "processor_config.json", "preprocessor_config.json" }) |name| {
        var why: abi.Verdict = .{};
        const bytes = dir.readFile(name, buf, &why) orelse continue;
        var tree: [16 * 1024]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&tree);
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, fba.allocator(), bytes, .{}) catch continue;
        if (parsed != .object) continue;
        const p = if (parsed.object.get("image_processor")) |v| (if (v == .object) v.object else parsed.object) else parsed.object;
        var min = positive(p.get("min_pixels"));
        var max = positive(p.get("max_pixels"));
        const max_tokens = positive(p.get("max_image_tokens"));
        if (p.get("size")) |sz| if (sz == .object) {
            if (min == null) min = positive(sz.object.get("shortest_edge"));
            if (max == null) max = positive(sz.object.get("longest_edge"));
        };
        if (min != null and max != null and min.? > max.?) continue;
        if (lo == null) lo = min;
        if (hi == null) hi = max;
        if (tokens == null) tokens = max_tokens;
    }
    if (lo != null and hi != null and lo.? > hi.?) {
        lo = null;
        hi = null;
        tokens = null;
    }
    c.qv_min_pixels = lo orelse 0;
    c.qv_max_pixels = hi orelse 0;
    c.mv_max_image_tokens = tokens orelse 4096;
}

/// Whether the index names n-gram shards (the table rides in the weights instead of `ngram_table.bin`).
fn indexNamesNgramShards(dir: *const abi.PackDir) bool {
    const buf = std.heap.page_allocator.alloc(u8, 64 << 20) catch return false;
    defer std.heap.page_allocator.free(buf);
    var why: abi.Verdict = .{};
    const bytes = dir.readFile("model.safetensors.index.json", buf, &why) orelse return false;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const v = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), bytes, .{}) catch return false;
    if (v != .object) return false;
    const wm = v.object.get("weight_map") orelse return false;
    if (wm != .object) return false;
    var it = wm.object.iterator();
    while (it.next()) |e| if (ple_table.embeddedTensorName(e.key_ptr.*)) return true;
    return false;
}

// ── the table's parse ──

const gpa = std.heap.c_allocator;

pub fn of(p: *const abi.Parsed) *const Config {
    return @ptrCast(@alignCast(p.private.?));
}

/// config.json (and, unless `stub`, the pack's sidecars) into `c`.
pub fn parseInto(c: *Config, dir: *const abi.PackDir, config_json: []const u8, stub: bool) ParseError!void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json, .{}) catch return error.InvalidConfigField;
    if (doc != .object) return error.InvalidConfigField;
    const mt = field(doc.object, "model_type") orelse return error.Unsupported;
    if (mt != .string or (!std.mem.eql(u8, mt.string, model_types[0]) and !std.mem.eql(u8, mt.string, model_types[1]))) return error.Unsupported;
    try parseJson(c, doc.object);
    // A sentence-transformers pooling sidecar makes the pack an embedding model.
    if (dir.has("1_Pooling/config.json")) return error.Unsupported;
    if (stub) return;
    const buf = try gpa.alloc(u8, 1 << 20);
    defer gpa.free(buf);
    applyGeneration(c, dir, buf);
    if (c.gen_top_k == null) c.gen_top_k = 20;
    if (c.gen_top_p == null) c.gen_top_p = 0.95;
    if (c.qwen_vision) applyProcessor(c, dir, buf);
    c.embedded_ngram = indexNamesNgramShards(dir);
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    if (span != null) return abi.Verdict.refuse(.unsupported, "qwen4_exp: a layer span (the cluster split) is served by the legacy path");
    const c = gpa.create(Config) catch return abi.Verdict.refuse(.failed, "qwen4_exp: out of memory");
    c.* = .{};
    parseInto(c, dir, config_json.slice(), stub) catch |e| {
        gpa.destroy(c);
        return switch (e) {
            error.Unsupported => abi.Verdict.refuse(.unsupported, "qwen4_exp: a shape this plugin does not serve yet"),
            error.OutOfMemory => abi.Verdict.refuse(.failed, "qwen4_exp: out of memory"),
            inline else => |x| abi.Verdict.refuse(.failed, "qwen4_exp: " ++ @errorName(x)),
        };
    };
    out.* = facts(c);
    return .ok;
}

pub fn freeParsed(p: *abi.Parsed) callconv(.c) void {
    if (p.private) |ptr| gpa.destroy(@as(*Config, @ptrCast(@alignCast(ptr))));
    p.private = null;
}

/// What the host reads off a parsed pack.
pub fn facts(c: *Config) abi.Parsed {
    var p: abi.Parsed = .{ .private = c };
    p.caps = .{
        .batches_decode = true,
        // Each slot's QSA keys are read per slot inside one stacked forward; the per-slot attention arm declines.
        .batch_per_slot_attn = false,
        .shared_readonly_state = true,
        .long_ctx = true,
        .moe = c.num_experts > 0,
        .spec_rollback = true,
        .images = c.qwen_vision,
        .video = c.qwen_vision,
    };
    const s = &p.shell;
    s.* = .{
        .vocab_size = c.vocab_size,
        .hidden_size = c.hidden_size,
        .intermediate_size = c.intermediate_size,
        .num_hidden_layers = c.num_hidden_layers,
        .num_attention_heads = c.num_attention_heads,
        .num_key_value_heads = c.num_key_value_heads,
        .head_dim = c.head_dim,
        .num_experts_per_tok = c.num_experts_per_tok,
        .moe_intermediate_size = c.moe_intermediate_size,
        .max_position_embeddings = c.max_position_embeddings,
        .context_cap = c.contextCap(),
        .quant = .{ .bits = @intCast(@min(c.quant_bits, 255)), .group_size = c.quant_group_size, .mode = if (c.quant_bits == 0) .none else switch (c.quant_mode) {
            inline else => |m| @field(abi.QuantMode, @tagName(m)),
        } },
        .gen = .{
            .temperature = c.gen_temperature orelse std.math.nan(f32),
            .top_p = c.gen_top_p orelse std.math.nan(f32),
            .min_p = c.gen_min_p orelse std.math.nan(f32),
            .top_k = if (c.gen_top_k) |k| @intCast(k) else -1,
            .enable_thinking = if (c.gen_enable_thinking) |t| (if (t) .on else .off) else .unset,
        },
        .bos = if (c.bos_token_id) |b| b else -1,
        .act_dtype = .bf16,
        .state_dtype = if (c.ssm_state_f32) .f32 else .bf16,
        .model_type = abi.Str.of(model_types[0]),
    };
    @memcpy(s.eos[0..c.num_eos_tokens], c.eos_token_ids[0..c.num_eos_tokens]);
    s.eos_count = c.num_eos_tokens;
    p.chat = .{
        .tool_dialect = .qwen_xml,
        .sampling_defaults = .{ .top_k = 20, .top_p = 0.95 },
    };
    p.cache = .{
        .layout_namespace = if (c.ssm_state_f32) abi.Str.of("gdn-f32-state-v1") else .{},
        .ssm_state = .{
            .entries = c.num_hidden_layers,
            .dtype = if (c.ssm_state_f32) .f32 else .bf16,
            // The aux buffers (PLE window, QSA keys) are written in place.
            .snapshot_policy = .in_place,
            .checkpoint_bytes = ssmCheckpointBytes(c),
        },
    };
    p.spec = .{
        .mtp_layout = .in_checkpoint,
        .head_module_owned = true,
        // Module archs run only their own MTP head: PLD and drafters are unmeasured here. Flash Next drafts best at
        // the target's own temperature.
        .policy = .{ .drafter_ok = false, .mtp_group_solo = true, .draft_temperature = 1.0 },
    };
    p.placeholder_ids = .{
        .image = c.image_token_id,
        .video = c.video_token_id,
        .vision_start = c.vision_start_token_id,
        .vision_end = c.vision_end_token_id,
    };
    const rd: u32 = @intFromFloat(@as(f32, @floatFromInt(c.head_dim)) * c.partial_rotary_factor);
    if (c.mrope_section[0] + c.mrope_section[1] + c.mrope_section[2] > 0 and rd / 2 <= c.mrope_inv_freq.len) {
        const inv = c.mrope_inv_freq[0 .. rd / 2];
        if (c.rope_yarn) (mrope.Yarn{
            .theta = c.rope_theta,
            .factor = c.yarn_factor,
            .orig_max = c.yarn_orig_max_pos,
            .rotary_dim = rd,
            .beta_fast = c.yarn_beta_fast,
            .beta_slow = c.yarn_beta_slow,
            .truncate = c.yarn_truncate,
        }).invFreq(inv) else mrope.computeInvFreq(inv, rd, c.rope_theta);
        p.mrope = .{ .sections = c.mrope_section, .rope_dims = rd, .inv_freq = .of(inv), .mscale = if (c.rope_yarn) c.yarn_attention_factor else 1 };
    }
    p.weights = .{
        .prefix = abi.Str.of("language_model.model"),
        .quant_layout = if (c.modelopt) abi.Str.of("modelopt") else .{},
    };
    return p;
}

/// Bytes one SSM checkpoint holds: the recurrent state and conv window of every linear layer.
pub fn ssmCheckpointBytes(c: *const Config) u64 {
    if (c.linear_num_value_heads == 0) return 0;
    const linear: u64 = @as(u64, c.num_hidden_layers) -| c.attnLayers();
    if (linear == 0) return 0;
    const elem: u64 = if (c.ssm_state_f32) 4 else 2;
    const state: u64 = @as(u64, c.linear_num_value_heads) * c.linear_value_head_dim * c.linear_key_head_dim * elem;
    const conv_dim: u64 = 2 * @as(u64, c.linear_num_key_heads) * c.linear_key_head_dim + @as(u64, c.linear_num_value_heads) * c.linear_value_head_dim;
    return linear * (state + (@as(u64, c.linear_conv_kernel_dim) -| 1) * conv_dim * 2);
}

// ── tests ──

const testing = std.testing;
const fixture = @import("../../sdk/testing/fixture.zig");

fn parseOk(json: []const u8) !*Config {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    const v = parse(&dir, .of(json), .{}, null, false, &p);
    if (!v.isOk()) {
        std.debug.print("parse: {s}\n", .{v.why.slice()});
        return error.Refused;
    }
    return @ptrCast(@alignCast(p.private.?));
}

pub const tiny_text =
    \\{"model_type":"qwen4_exp_text","vocab_size":128,"hidden_size":64,"num_hidden_layers":4,
    \\ "num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"full_attention_interval":2,
    \\ "linear_num_key_heads":2,"linear_num_value_heads":4,"linear_key_head_dim":16,"linear_value_head_dim":16,
    \\ "num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":32,"shared_expert_intermediate_size":32,
    \\ "ple_layer_ids":[2],"ple_embed_dim":64,"ngram_size":3,"heads_per_ngram":2,"ngram_vocab_size_base":100,
    \\ "make_ngram_vocab_size_divisible_by":8,"split_ngram_parts":2,"indexer_n_heads":2,"indexer_head_dim":16,
    \\ "indexer_budget":8,"indexer_compress_ratio":4,"eos_token_id":1,"max_position_embeddings":4096}
;

test "qwen4_exp config: claims both spellings and nothing else" {
    const none: fixture.MemDir = .{ .files = &.{} };
    const d = none.packDir();
    try testing.expectEqual(abi.Claim.by_type, claims(.of("qwen4_exp"), .of("{}"), &d));
    try testing.expectEqual(abi.Claim.by_type, claims(.of("qwen4_exp_text"), .of("{}"), &d));
    try testing.expectEqual(abi.Claim.no, claims(.of("qwen3_5_moe"), .of("{}"), &d));
}

test "qwen4_exp config: the flat text pack parses with the family defaults and its facts" {
    const c = try parseOk(tiny_text);
    defer gpa.destroy(c);
    try testing.expectEqual(@as(u32, 4), c.hc_count);
    try testing.expectEqual(@as(i32, 1), c.ple_layer_idx);
    try testing.expectEqual(@as(u32, 32), c.query_pre_attn_scalar);
    try testing.expectEqual(@as(u32, 1), c.ngram_eos);
    try testing.expectEqual(@as(?u32, 20), c.gen_top_k);
    try testing.expect(!c.qwen_vision and !c.isFullLayer(0) and c.isFullLayer(1) and c.attnLayers() == 2);
    const p = facts(c);
    try testing.expect(p.caps.batches_decode and p.caps.long_ctx and p.caps.moe and !p.caps.images and !p.caps.batch_per_slot_attn);
    try testing.expectEqual(@as(u32, 1), p.shell.eos_count);
    try testing.expectEqual(@as(f32, 1.0), p.spec.policy.draft_temperature);
}

test "qwen4_exp config: a bad bound is refused by name, another family's key is unsupported" {
    const mem: fixture.MemDir = .{ .files = &.{} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    const no_ple = "{\"model_type\":\"qwen4_exp\",\"num_hidden_layers\":2}";
    const v = parse(&dir, .of(no_ple), .{}, null, true, &p);
    try testing.expect(v.code == .failed and std.mem.indexOf(u8, v.why.slice(), "InvalidQwen4PleLayer") != null and p.private == null);
    const audio = "{\"model_type\":\"qwen4_exp\",\"audio_config\":{}}";
    try testing.expectEqual(abi.Verdict.Code.unsupported, parse(&dir, .of(audio), .{}, null, true, &p).code);
    try testing.expectEqual(abi.Verdict.Code.unsupported, parse(&dir, .of(tiny_text), .{}, &.{ .first = 0, .count = 2 }, true, &p).code);
}
