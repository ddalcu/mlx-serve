//! GLM-5.3-Flash's claims and parse, and the KDA-hybrid family config it shares with Ling 3.0 (`bailing_hybrid`
//! imports `Config`, the readers and `readCommon`). MLX-free (discovery runs it). Fields are named as the engine's
//! `ModelConfig` names them, with its defaults, so the modules that still read `ModelConfig` get them one to one
//! (`model_config.zig`). A shape this plugin does not serve exactly as the engine did is refused `unsupported`.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const generation = @import("../../sdk/config/generation.zig");
const fp8_grid = @import("../../sdk/quant/fp8_block128_gemv.zig");

/// `glm_moe_dsa` (the streamed GLM-5.3 pack) is claimed here and declined: lib/mlx-stream serves it.
pub const model_types = [_][]const u8{ "glm5_next", "glm5_next_text", "glm_moe_dsa" };

pub const claims = sdk.claimsByType(&model_types);

pub const plugin = sdk.Plugin{
    .name = "glm5_next",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"ddalcu/GLM-5.3-Flash-MLX-Serve-iQ-MLX-5.1bpw"},
};

pub const QuantMode = enum { affine, nvfp4, mxfp4, mxfp8, gguf };

/// The family's config: every field the engine's parse sets for a GLM-5.3-Flash or Ling 3.0 pack.
pub const Config = struct {
    model_type: []const u8 = "glm5_next",
    weight_prefix: []const u8 = "model",
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
    rope_local_base_freq: f32 = 10000.0,
    rope_scaling_factor: f32 = 1.0,
    partial_rotary_factor: f32 = 1.0,
    rope_interleaved_pairs: bool = false,
    query_pre_attn_scalar: u32 = 256,
    has_sliding_window: bool = false,
    sliding_window: u32 = 1024,
    tie_word_embeddings: bool = false,
    bos_token_id: ?u32 = null,
    eos_token_ids: [8]u32 = @splat(0),
    num_eos_tokens: u32 = 0,
    quant_bits: u32 = 0,
    quant_group_size: u32 = 64,
    quant_mode: QuantMode = .affine,
    /// NVIDIA ModelOpt FP4 (`quantization_config.quant_method: modelopt`).
    modelopt: bool = false,
    attn_output_gate: bool = false,
    has_explicit_layer_types: bool = false,
    layer_is_global: [128]bool = @splat(false),

    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    moe_intermediate_size: u32 = 0,
    shared_expert_intermediate_size: u32 = 0,
    moe_sigmoid_router: bool = true,
    moe_route_norm: bool = true,
    router_scaling_factor: f32 = 1.0,
    moe_n_group: u32 = 1,
    moe_topk_group: u32 = 1,
    first_k_dense_replace: u32 = 0,
    swiglu_limit: f32 = 0.0,
    swiglu_clamp: bool = false,

    /// Every `full_attention_interval`-th layer is attention, the rest KDA.
    full_attention_interval: u32 = 0,
    linear_num_key_heads: u32 = 0,
    linear_num_value_heads: u32 = 0,
    linear_key_head_dim: u32 = 128,
    linear_value_head_dim: u32 = 128,
    linear_conv_kernel_dim: u32 = 4,
    kda_vector_gate: bool = true,
    kda_gate_lower_bound: f32 = 0.0,
    kda_sigmoid_out_gate: bool = true,

    // GLM-5.3: DeepSeek sparse attention (absorbed NoPE MLA over a latent, pooled indexer) inside mHC.
    dsa_q_lora_rank: u32 = 0,
    dsa_kv_lora_rank: u32 = 0,
    dsa_head_dim: u32 = 0,
    dsa_index_topk: u32 = 0,
    dsa_index_kpool: u32 = 0,
    dsa_index_heads: u32 = 0,
    dsa_index_head_dim: u32 = 0,
    dsv4_hc_mult: u32 = 0,
    dsv4_hc_sinkhorn_iters: u32 = 0,
    dsv4_hc_eps: f32 = 1e-6,

    // Ling 3.0: Multi-head Latent Attention.
    mla_q_lora_rank: u32 = 0,
    mla_kv_lora_rank: u32 = 0,
    mla_qk_nope_head_dim: u32 = 0,
    mla_qk_rope_head_dim: u32 = 0,
    mla_v_head_dim: u32 = 0,
    mla_head_gate: bool = false,

    has_vision: bool = false,
    vision_hidden_size: u32 = 768,
    vision_num_layers: u32 = 16,
    vision_num_heads: u32 = 12,
    vision_head_dim: u32 = 64,
    vision_intermediate_size: u32 = 3072,
    vision_patch_size: u32 = 16,
    vision_rope_theta: f32 = 100.0,
    image_token_id: u32 = 0,
    boi_token_id: u32 = 0,
    eoi_token_id: u32 = 0,
    glm_vision: bool = false,
    /// What discovery advertises: no images for a `_text` pack's vestigial tower, video for a `video_token_id`
    /// beside the tower (which the engine never reads for GLM).
    advertises_images: bool = false,
    advertises_video: bool = false,
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
    glmv_proj_intermediate: u32 = 10240,
    glmv_eps: f32 = 1e-5,
    glmv_swiglu_limit: f32 = 10.0,
    glmv_min_image_tokens: u32 = 16,
    glmv_max_image_tokens: u32 = 8000,

    /// A Sushi pack's `expert_quant` block as JSON (EXL3 routed experts); empty = none. Read by the engine side.
    expert_quant_json: []const u8 = "",
    /// The routed experts stream from an EXL3 bank beside the shards (`experts.bin`, `expert_streaming`).
    streamed: bool = false,

    /// Pack-owned sampling recommendations (`generation_config.json`).
    gen: abi.GenDefaults = .{},

    pub fn isGlm5(self: *const Config) bool {
        return std.mem.eql(u8, self.model_type, "glm5_next");
    }

    pub fn isLinearLayer(self: *const Config, li: u32) bool {
        if (self.full_attention_interval == 0) return false;
        return (li + 1) % self.full_attention_interval != 0;
    }

    pub fn attnLayerCount(self: *const Config) u32 {
        var n: u32 = 0;
        for (0..self.num_hidden_layers) |i| n += @intFromBool(!self.isLinearLayer(@intCast(i)));
        return n;
    }

    /// Bytes one SSM checkpoint holds: every KDA layer's recurrence (`state_bytes` an element) and conv window.
    pub fn ssmCheckpointBytes(self: *const Config, state_bytes: u64) u64 {
        const linear: u64 = @as(u64, self.num_hidden_layers) - self.attnLayerCount();
        const state: u64 = @as(u64, self.linear_num_value_heads) * self.linear_value_head_dim * self.linear_key_head_dim * state_bytes;
        const conv_dim: u64 = 2 * @as(u64, self.linear_num_key_heads) * self.linear_key_head_dim + @as(u64, self.linear_num_value_heads) * self.linear_value_head_dim;
        return linear * (state + (@as(u64, self.linear_conv_kernel_dim) -| 1) * conv_dim * 2);
    }

    fn addEos(self: *Config, id: u32) void {
        if (self.num_eos_tokens < self.eos_token_ids.len) {
            self.eos_token_ids[self.num_eos_tokens] = id;
            self.num_eos_tokens += 1;
        }
    }
};

pub const ParseError = error{ InvalidConfigField, Unsupported };

// Readers with the engine parser's semantics: a JSON null reads as an omitted key.

pub fn field(obj: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    const v = obj.get(key) orelse return null;
    return if (v == .null) null else v;
}

pub fn value(comptime tag: std.meta.Tag(std.json.Value), v: std.json.Value) ParseError!@TypeOf(@field(@as(std.json.Value, undefined), @tagName(tag))) {
    if (v != tag) return error.InvalidConfigField;
    return @field(v, @tagName(tag));
}

pub fn u32v(v: std.json.Value) ParseError!u32 {
    return std.math.cast(u32, try value(.integer, v)) orelse error.InvalidConfigField;
}

/// A non-numeric value reads as 0, as the engine's reader does.
pub fn f32v(v: std.json.Value) ParseError!f32 {
    const f: f32 = switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |x| @floatCast(x),
        else => 0.0,
    };
    if (!std.math.isFinite(f)) return error.InvalidConfigField;
    return f;
}

/// The integer at `key`, when the value is one; another type leaves `out` as it was.
pub fn intIf(obj: std.json.ObjectMap, key: []const u8, out: *u32) ParseError!void {
    if (field(obj, key)) |v| if (v == .integer) {
        out.* = try u32v(v);
    };
}

fn positive(v: ?std.json.Value) ?u32 {
    const x = v orelse return null;
    return if (x == .integer and x.integer > 0 and x.integer <= std.math.maxInt(u32)) @intCast(x.integer) else null;
}

/// Keys the engine's common block reads for other families; a pack carrying one is not modelled here.
const foreign_keys = [_][]const u8{
    "top_k_experts",          "use_bidirectional_attention", "pooling_mode",
    "sliding_window_pattern", "global_head_dim",             "num_global_key_value_heads",  "num_kv_shared_layers",
    "attention_k_eq_v",       "final_logit_softcapping",     "hidden_size_per_layer_input", "rope_local_base_freq",
    "audio_config",           "audio_token_id",              "boa_token_id",                "eoa_token_index",
    "eoa_token_id",           "image_token_index",           "llm_config",                  "thinker_config",
    "linear_num_key_heads",   "linear_num_value_heads",      "linear_key_head_dim",         "linear_value_head_dim",
    "linear_conv_kernel_dim", "full_attention_interval",
};

pub const Objects = struct {
    root: std.json.ObjectMap,
    /// The text model: `text_config` when nested, else the root.
    cfg: std.json.ObjectMap,
};

/// The engine's common block for this family's packs: shape, rope, layer types, quantization, eos, the generic
/// vision geometry and the media ids. `error.Unsupported` for a key another family's arm reads.
pub fn readCommon(c: *Config, o: Objects) ParseError!void {
    const root = o.root;
    const cfg = o.cfg;
    // A release that also spells its `linear_attn_config` flat (`linear_conv_kernel_dim`) is read from the block.
    const lac = field(cfg, "linear_attn_config") != null;
    for (foreign_keys) |k| if (field(cfg, k) != null or field(root, k) != null) {
        if (!(lac and std.mem.eql(u8, k, "linear_conv_kernel_dim"))) return error.Unsupported;
    };
    if (field(cfg, "vocab_size")) |v| c.vocab_size = try u32v(v);
    if (field(cfg, "hidden_size")) |v| c.hidden_size = try u32v(v);
    if (field(cfg, "intermediate_size")) |v| {
        c.intermediate_size = try u32v(v);
        c.intermediate_size_declared = true;
    }
    if (field(cfg, "num_hidden_layers")) |v| c.num_hidden_layers = try u32v(v);
    if (field(cfg, "num_attention_heads")) |v| c.num_attention_heads = try u32v(v);
    if (field(cfg, "num_key_value_heads")) |v| c.num_key_value_heads = try u32v(v);
    if (field(cfg, "head_dim")) |v| c.head_dim = try u32v(v);
    if (field(cfg, "max_position_embeddings")) |v| c.max_position_embeddings = try u32v(v);
    if (field(cfg, "rms_norm_eps")) |v| c.rms_norm_eps = try f32v(v);
    if (field(cfg, "rope_theta")) |v| c.rope_theta = try f32v(v);
    if (field(cfg, "query_pre_attn_scalar")) |v| c.query_pre_attn_scalar = try u32v(v);
    try intIf(cfg, "num_experts", &c.num_experts);
    try intIf(cfg, "num_experts_per_tok", &c.num_experts_per_tok);
    try intIf(cfg, "moe_intermediate_size", &c.moe_intermediate_size);
    try intIf(cfg, "shared_expert_intermediate_size", &c.shared_expert_intermediate_size);
    if (field(cfg, "attn_output_gate")) |v| if (v == .bool) {
        c.attn_output_gate = v.bool;
    };
    if (field(cfg, "bos_token_id")) |v| if (v == .integer and v.integer >= 0) {
        c.bos_token_id = try u32v(v);
    };
    if (field(cfg, "rope_parameters")) |rp| if (rp == .object) {
        if (field(rp.object, "full_attention") != null or field(rp.object, "sliding_attention") != null) return error.Unsupported;
        if (field(rp.object, "rope_theta")) |v| c.rope_theta = try f32v(v);
        if (field(rp.object, "partial_rotary_factor")) |v| c.partial_rotary_factor = try f32v(v);
    };
    if (cfg.get("sliding_window")) |v| if (v != .null) {
        c.sliding_window = try u32v(v);
    };
    if (field(cfg, "layer_types")) |lt| if (lt == .array) {
        c.has_explicit_layer_types = true;
        for (lt.array.items, 0..) |item, i| {
            if (i >= 128) break;
            if (item == .string) c.layer_is_global[i] = std.mem.eql(u8, item.string, "full_attention");
        }
    };
    if (field(root, "tie_word_embeddings")) |v| if (v == .bool) {
        c.tie_word_embeddings = v.bool;
    };
    if (field(cfg, "tie_word_embeddings")) |v| if (v == .bool) {
        c.tie_word_embeddings = v.bool;
    };
    if (c.max_position_embeddings == 0) if (field(root, "max_position_embeddings")) |v| if (v == .integer) {
        c.max_position_embeddings = try u32v(v);
    };
    if (field(root, "quantization")) |q_val| {
        const q = try value(.object, q_val);
        if (field(q, "bits")) |v| c.quant_bits = try u32v(v);
        if (field(q, "group_size")) |v| c.quant_group_size = try u32v(v);
        if (field(q, "mode")) |v| if (v == .string) {
            c.quant_mode = std.meta.stringToEnum(QuantMode, v.string) orelse return error.Unsupported;
            if (c.quant_mode == .gguf) return error.Unsupported;
        };
        var it = q.iterator();
        while (it.next()) |kv| if (kv.value_ptr.* == .object) if (kv.value_ptr.object.get("mode")) |m| if (m == .string) {
            // A release's e4m3 projection on a 64 x 128 grid is read by its dtypes (`fp8_block128`).
            if (fp8_grid.isEntry(kv.value_ptr.object)) continue;
            const mode = std.meta.stringToEnum(QuantMode, m.string) orelse return error.Unsupported;
            if (mode == .gguf) return error.Unsupported;
        };
        if (c.quant_mode == .affine and c.quant_bits != 0) switch (c.quant_bits) {
            2, 3, 4, 5, 6, 8 => {},
            else => return error.Unsupported,
        };
    }
    if (field(root, "quantization_config")) |qc| if (qc == .object) if (qc.object.get("quant_method")) |m| {
        if (m == .string and std.mem.eql(u8, m.string, "modelopt")) {
            c.modelopt = true;
            c.quant_bits = 4;
            c.quant_group_size = 16;
            c.quant_mode = .nvfp4;
        }
    };
    if (field(root, "eos_token_id")) |v| switch (v) {
        .integer => |i| c.addEos(std.math.cast(u32, i) orelse return error.InvalidConfigField),
        .array => |arr| for (arr.items) |item| if (item == .integer) c.addEos(try u32v(item)),
        else => {},
    };
    if (field(root, "vision_config")) |vc_val| if (vc_val == .object) {
        c.has_vision = true;
        const vc = vc_val.object;
        inline for (.{ "pooling_kernel_size", "default_output_length", "position_embedding_size", "use_clipped_linears", "mm_embed_dim", "model_patch_size", "num_soft_tokens", "mm_posemb_size", "global_head_dim" }) |k|
            if (field(vc, k) != null) return error.Unsupported;
        try intIf(vc, "hidden_size", &c.vision_hidden_size);
        try intIf(vc, "num_hidden_layers", &c.vision_num_layers);
        try intIf(vc, "num_attention_heads", &c.vision_num_heads);
        try intIf(vc, "head_dim", &c.vision_head_dim);
        try intIf(vc, "intermediate_size", &c.vision_intermediate_size);
        try intIf(vc, "patch_size", &c.vision_patch_size);
        if (field(vc, "rope_parameters")) |rp| if (rp == .object) if (field(rp.object, "rope_theta")) |v| {
            c.vision_rope_theta = try f32v(v);
        };
    };
    try intIf(root, "image_token_id", &c.image_token_id);
    try intIf(root, "boi_token_id", &c.boi_token_id);
    try intIf(root, "eoi_token_id", &c.eoi_token_id);
}

/// The Qwen-VL `vision_config` geometry keys into `qv_*`; GLM-5.3-Flash's tower names them the same.
pub fn readQvGeometry(c: *Config, vc: std.json.ObjectMap) ParseError!void {
    try intIf(vc, "depth", &c.qv_depth);
    try intIf(vc, "hidden_size", &c.qv_hidden);
    try intIf(vc, "num_heads", &c.qv_heads);
    try intIf(vc, "intermediate_size", &c.qv_intermediate);
    try intIf(vc, "patch_size", &c.qv_patch);
    try intIf(vc, "temporal_patch_size", &c.qv_temporal_patch);
    try intIf(vc, "spatial_merge_size", &c.qv_merge);
    try intIf(vc, "num_position_embeddings", &c.qv_num_pos_emb);
    try intIf(vc, "out_hidden_size", &c.qv_out_hidden);
    if (c.qv_heads != 0) c.qv_head_dim = c.qv_hidden / c.qv_heads;
    if (c.qv_out_hidden == 0) c.qv_out_hidden = c.hidden_size;
}

/// The family arm of the engine's parse for `glm5_next` / `glm5_next_text`.
fn readGlm(c: *Config, o: Objects) ParseError!void {
    const root = o.root;
    const cfg = o.cfg;
    c.model_type = "glm5_next";
    c.weight_prefix = if (field(root, "text_config") != null) "language_model.model" else "model";
    c.has_sliding_window = false;
    c.rope_scaling_factor = 1.0;
    c.rope_local_base_freq = c.rope_theta;
    c.moe_sigmoid_router = true;
    if (field(cfg, "n_routed_experts")) |v| c.num_experts = try u32v(v);
    if (field(cfg, "norm_topk_prob")) |v| if (v == .bool) {
        c.moe_route_norm = v.bool;
    };
    if (field(cfg, "routed_scaling_factor")) |v| c.router_scaling_factor = try f32v(v);
    if (field(cfg, "n_group")) |v| c.moe_n_group = try u32v(v);
    if (field(cfg, "topk_group")) |v| c.moe_topk_group = try u32v(v);
    if (field(cfg, "n_shared_experts")) |v| if (v == .integer) {
        c.shared_expert_intermediate_size = std.math.mul(u32, try u32v(v), c.moe_intermediate_size) catch return error.InvalidConfigField;
    };
    if (field(cfg, "swiglu_limit")) |v| c.swiglu_limit = try f32v(v);
    c.swiglu_clamp = true;

    if (field(cfg, "qk_rope_head_dim")) |v| if (v == .integer and v.integer != 0) return error.Unsupported;
    if (field(cfg, "q_lora_rank")) |v| c.dsa_q_lora_rank = try u32v(v);
    if (field(cfg, "kv_lora_rank")) |v| c.dsa_kv_lora_rank = try u32v(v);
    if (field(cfg, "qk_nope_head_dim")) |v| c.dsa_head_dim = try u32v(v);
    if (field(cfg, "v_head_dim")) |v| if (try u32v(v) != c.dsa_head_dim) return error.Unsupported;
    c.head_dim = c.dsa_head_dim;
    c.num_key_value_heads = c.num_attention_heads;
    if (field(cfg, "index_topk")) |v| c.dsa_index_topk = try u32v(v);
    if (field(cfg, "index_kpool")) |v| c.dsa_index_kpool = try u32v(v);
    if (field(cfg, "index_n_heads")) |v| c.dsa_index_heads = try u32v(v);
    if (field(cfg, "index_head_dim")) |v| c.dsa_index_head_dim = try u32v(v);
    inline for (.{ "index_kpool_compress", "index_kpool_always_select_tail", "mla_use_nope" }) |key|
        if (field(cfg, key)) |v| if (v == .bool and !v.bool) return error.Unsupported;
    if (c.dsa_index_kpool == 0 or c.dsa_index_topk % c.dsa_index_kpool != 0) return error.Unsupported;
    if (field(cfg, "indexer_types")) |v| for ((try value(.array, v)).items) |t| {
        if (!std.mem.eql(u8, try value(.string, t), "full")) return error.Unsupported;
    };

    c.full_attention_interval = 4;
    if (field(cfg, "layer_types")) |v| for ((try value(.array, v)).items, 0..) |t, i| {
        const name = try value(.string, t);
        const sparse = std.mem.eql(u8, name, "deepseek_sparse_attention") or std.mem.eql(u8, name, "full_attention");
        if (sparse == c.isLinearLayer(@intCast(i))) return error.Unsupported;
    };
    if (field(cfg, "mlp_layer_types")) |v| {
        c.first_k_dense_replace = 0;
        for ((try value(.array, v)).items, 0..) |t, i| if (std.mem.eql(u8, try value(.string, t), "dense")) {
            if (c.first_k_dense_replace != i) return error.Unsupported;
            c.first_k_dense_replace += 1;
        };
    } else if (field(cfg, "first_k_dense_replace")) |v| c.first_k_dense_replace = try u32v(v);

    c.linear_num_key_heads = c.num_attention_heads;
    c.linear_key_head_dim = 128;
    if (field(cfg, "linear_attn_config")) |lac_v| {
        const lac = try value(.object, lac_v);
        if (field(lac, "num_heads")) |v| c.linear_num_key_heads = try u32v(v);
        if (field(lac, "head_dim")) |v| c.linear_key_head_dim = try u32v(v);
        if (field(lac, "short_conv_kernel_size")) |v| c.linear_conv_kernel_dim = try u32v(v);
        if (field(lac, "gate_lower_bound")) |v| c.kda_gate_lower_bound = try f32v(v);
    }
    if (c.kda_gate_lower_bound == 0.0) c.kda_gate_lower_bound = -5.0;
    c.linear_num_value_heads = c.linear_num_key_heads;
    c.linear_value_head_dim = c.linear_key_head_dim;
    c.kda_vector_gate = true;
    c.kda_sigmoid_out_gate = true;

    c.dsv4_hc_mult = 4;
    if (field(cfg, "hc_mult")) |v| c.dsv4_hc_mult = try u32v(v);
    if (c.dsv4_hc_mult != 4) return error.Unsupported;
    c.dsv4_hc_sinkhorn_iters = 20;
    if (field(cfg, "hc_sinkhorn_iters")) |v| c.dsv4_hc_sinkhorn_iters = try u32v(v);
    if (field(cfg, "hc_eps")) |v| c.dsv4_hc_eps = try f32v(v);
    if (c.has_vision) {
        const vc = field(root, "vision_config").?.object;
        c.glm_vision = true;
        const mt = field(root, "model_type");
        c.advertises_images = !(mt != null and mt.? == .string and std.mem.endsWith(u8, mt.?.string, "_text"));
        if (c.advertises_images) for ([_]std.json.ObjectMap{ cfg, root }) |obj| if (field(obj, "video_token_id")) |v| if (v == .integer and v.integer > 0) {
            c.advertises_video = true;
        };
        try readQvGeometry(c, vc);
        if (field(vc, "projection_intermediate_size")) |v| c.glmv_proj_intermediate = try u32v(v);
        if (field(vc, "rms_norm_eps")) |v| c.glmv_eps = try f32v(v);
        if (field(vc, "swiglu_limit")) |v| c.glmv_swiglu_limit = try f32v(v);
        if (c.qv_heads == 0 or c.qv_hidden % c.qv_heads != 0 or c.qv_head_dim % 4 != 0 or
            c.qv_patch == 0 or c.qv_merge == 0 or c.qv_temporal_patch == 0 or c.qv_out_hidden != c.hidden_size)
            return error.InvalidConfigField;
    }
}

/// The processor's image budget (`processor_config.json`, then `preprocessor_config.json`), as the engine reads it.
pub fn readProcessor(c: *Config, dir: *const abi.PackDir) void {
    var min_pixels: ?u32 = null;
    var max_pixels: ?u32 = null;
    var max_tokens: ?u32 = null;
    var min_tokens: ?u32 = null;
    var buf: [1 << 20]u8 = undefined;
    for ([_][]const u8{ "processor_config.json", "preprocessor_config.json" }) |name| {
        var why: abi.Verdict = .{};
        const bytes = dir.readFile(name, &buf, &why) orelse continue;
        var tree: [16 * 1024]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&tree);
        const parsed = std.json.parseFromSlice(std.json.Value, fba.allocator(), bytes, .{}) catch continue;
        if (parsed.value != .object) continue;
        const root = parsed.value.object;
        const p = if (root.get("image_processor")) |v| (if (v == .object) v.object else root) else root;
        var lo = positive(p.get("min_pixels"));
        var hi = positive(p.get("max_pixels"));
        if (p.get("size")) |s| if (s == .object) {
            if (lo == null) lo = positive(s.object.get("shortest_edge"));
            if (hi == null) hi = positive(s.object.get("longest_edge"));
        };
        var tok_hi = positive(p.get("max_image_tokens"));
        var tok_lo = positive(p.get("min_image_tokens"));
        if (lo != null and hi != null and lo.? > hi.?) {
            lo = null;
            hi = null;
            tok_hi = null;
            tok_lo = null;
        }
        if (min_pixels == null) min_pixels = lo;
        if (max_pixels == null) max_pixels = hi;
        if (max_tokens == null) max_tokens = tok_hi;
        if (min_tokens == null) min_tokens = tok_lo;
    }
    if (min_pixels != null and max_pixels != null and min_pixels.? > max_pixels.?) {
        min_pixels = null;
        max_pixels = null;
        max_tokens = null;
        min_tokens = null;
    }
    c.qv_min_pixels = min_pixels orelse 0;
    c.qv_max_pixels = max_pixels orelse 0;
    c.mv_max_image_tokens = max_tokens orelse 4096;
    const lo = min_tokens orelse c.glmv_min_image_tokens;
    const hi = max_tokens orelse c.glmv_max_image_tokens;
    if (lo <= hi) {
        c.glmv_min_image_tokens = lo;
        c.glmv_max_image_tokens = hi;
    }
}

const gpa = std.heap.c_allocator;

/// The parse both plugins run: config.json into `Config` through `arm`, then the `Parsed` facts `facts` fills.
pub fn parseWith(
    comptime name: []const u8,
    comptime types: []const []const u8,
    comptime arm: fn (*Config, Objects) ParseError!void,
    comptime facts: fn (*const Config, *abi.Parsed) void,
    dir: *const abi.PackDir,
    config_json: abi.Str,
    span: ?*const abi.LayerSpan,
    stub: bool,
    out: *abi.Parsed,
) abi.Verdict {
    if (span != null) return abi.Verdict.refuse(.unsupported, name ++ ": a layer span (the cluster split)");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json.slice(), .{}) catch
        return abi.Verdict.refuse(.failed, name ++ ": config.json does not parse");
    if (doc != .object) return abi.Verdict.refuse(.failed, name ++ ": config.json is not an object");
    const root = doc.object;
    const mt = field(root, "model_type") orelse return abi.Verdict.refuse(.unsupported, name ++ ": no model_type");
    if (mt != .string) return abi.Verdict.refuse(.unsupported, name ++ ": no model_type");
    for (types) |t| {
        if (std.mem.eql(u8, t, mt.string)) break;
    } else return abi.Verdict.refuse(.unsupported, name ++ ": a model_type this plugin does not serve");
    const streamed = field(root, "expert_streaming");
    if (streamed) |es| if (!isExl3Bank(es)) return abi.Verdict.refuse(.unsupported, name ++ ": an expert_streaming block other than the EXL3 K=3 mul1 bank");
    if (field(root, "residents_format")) |rf| if (rf != .string or !std.mem.eql(u8, rf.string, "fp8_block128"))
        return abi.Verdict.refuse(.unsupported, name ++ ": a residents_format other than fp8_block128");
    const tc = field(root, "text_config");
    if (tc != null and tc.? != .object) return abi.Verdict.refuse(.failed, name ++ ": text_config is not an object");
    const o: Objects = .{ .root = root, .cfg = if (tc) |t| t.object else root };
    var c: Config = .{ .streamed = streamed != null };
    readCommon(&c, o) catch |e| return refusal(name, e);
    arm(&c, o) catch |e| return refusal(name, e);
    if (field(root, "expert_quant")) |eq| {
        var w: std.Io.Writer.Allocating = .init(gpa);
        std.json.Stringify.value(eq, .{}, &w.writer) catch return abi.Verdict.refuse(.failed, name ++ ": out of memory");
        c.expert_quant_json = w.toOwnedSlice() catch return abi.Verdict.refuse(.failed, name ++ ": out of memory");
    }
    // Discovery's stub parse stops at config.json, as the engine's `parseConfigFromJson` does.
    if (c.glm_vision and !stub) readProcessor(&c, dir);

    var p: abi.Parsed = .{};
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
        .context_cap = c.max_position_embeddings,
        .quant = .{ .bits = std.math.cast(u8, c.quant_bits) orelse 0, .group_size = c.quant_group_size, .mode = if (c.quant_bits == 0) .none else std.meta.stringToEnum(abi.QuantMode, @tagName(c.quant_mode)).? },
        .bos = if (c.bos_token_id) |b| b else -1,
        .act_dtype = .bf16,
        .model_type = abi.Str.of(c.model_type),
    };
    for (c.eos_token_ids[0..c.num_eos_tokens]) |e| generation.addEos(s, e);
    if (!stub) {
        generation.apply(dir, s);
        c.num_eos_tokens = @min(s.eos_count, c.eos_token_ids.len);
        @memcpy(c.eos_token_ids[0..c.num_eos_tokens], s.eos[0..c.num_eos_tokens]);
    }
    c.gen = s.gen;
    facts(&c, &p);

    const cp = gpa.create(Config) catch {
        if (c.expert_quant_json.len > 0) gpa.free(c.expert_quant_json);
        return abi.Verdict.refuse(.failed, name ++ ": out of memory");
    };
    cp.* = c;
    p.private = cp;
    out.* = p;
    return .ok;
}

/// mlx-stream's EXL3 K=3 mul1 bank (`experts.bin` + its v2 and runtime manifests), the one streamed format served.
fn isExl3Bank(v: std.json.Value) bool {
    if (v != .object) return false;
    const want = [_][2][]const u8{ .{ "family", "mlx-stream" }, .{ "format", "mtplx-expert-manifest-v2" }, .{ "manifest", "expert-manifest-v2.json" }, .{ "runtime_manifest", "expert-manifest.json" }, .{ "quant", "exl3-k3-mul1" } };
    for (want) |kv| {
        const f = v.object.get(kv[0]) orelse return false;
        if (f != .string or !std.mem.eql(u8, f.string, kv[1])) return false;
    }
    return true;
}

fn refusal(comptime name: []const u8, e: ParseError) abi.Verdict {
    return switch (e) {
        error.Unsupported => abi.Verdict.refuse(.unsupported, name ++ ": a shape this plugin does not serve (the engine's parse names it)"),
        error.InvalidConfigField => abi.Verdict.refuse(.failed, name ++ ": a config field has the wrong type or range"),
    };
}

/// GLM-5.3-Flash's `Parsed` facts beside the shell.
fn glmFacts(c: *const Config, p: *abi.Parsed) void {
    p.caps = .{
        .batches_decode = true,
        // Each slot's DSA cache is advanced by its own call: a batched tick never pads or stacks KV.
        .batch_pads_kv = false,
        .moe = c.num_experts > 0,
        .dflash_capture = true,
        .dflash_policy = true,
        .images = c.advertises_images,
        .video = c.advertises_video,
        .streamed_weights = c.streamed,
    };
    // A streamed bank's decode lane takes a few rows a call: one request decodes at a time.
    if (c.streamed) p.caps.batches_decode = false;
    p.shell.state_dtype = .f32;
    p.chat.thinking_default = .on;
    p.chat.effort_vocab = .glm5;
    // A pool's completing indexer row stores the pool's key in its gate half; v2: the KDA q/k norm epsilon.
    p.cache.layout_namespace = abi.Str.of("glm5-pooled-keys-v2");
    p.cache.ssm_state = .{ .entries = c.num_hidden_layers, .dtype = .f32 };
    // Its head drafts one slot at a time: MTP rounds never group.
    p.spec = .{ .mtp_default_depth = 3, .mtp_layout = .in_checkpoint, .head_reads_final_norm = true, .policy = .{ .mtp_group_solo = true } };
    p.placeholder_ids.image = c.image_token_id;
    p.weights.prefix = abi.Str.of(c.weight_prefix);
    if (c.modelopt) p.weights.quant_layout = abi.Str.of("modelopt") else if (c.expert_quant_json.len > 0) p.weights.quant_layout = abi.Str.of("exl3");
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    return parseWith("glm5_next", &.{ "glm5_next", "glm5_next_text" }, readGlm, glmFacts, dir, config_json, span, stub, out);
}

pub fn freeParsed(p: *abi.Parsed) callconv(.c) void {
    if (p.private) |ptr| {
        const c: *Config = @ptrCast(@alignCast(ptr));
        if (c.expert_quant_json.len > 0) gpa.free(c.expert_quant_json);
        gpa.destroy(c);
    }
    p.private = null;
}

pub fn of(p: *const abi.Parsed) *const Config {
    return @ptrCast(@alignCast(p.private.?));
}

// ── tests ──

const testing = std.testing;

pub fn parseJson(json: []const u8, stub: bool, out: *abi.Parsed) abi.Verdict {
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(json), .{}, null, stub, out);
}

test "glm5_next config: the mlx-community pack parses into the shell, its facts and the family config" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(@embedFile("../../fixtures/model-configs/glm5_next-mlx-community-4bit.json"), true, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expectEqual(@as(u32, 45), p.shell.num_hidden_layers);
    try testing.expectEqual(@as(u32, 256), p.shell.head_dim);
    try testing.expectEqual(@as(u32, 11), c.attnLayerCount());
    try testing.expect(p.caps.batches_decode and !p.caps.batch_pads_kv and p.caps.moe and p.caps.images);
    try testing.expectEqualStrings("glm5-pooled-keys-v2", p.cache.layout_namespace.slice());
    try testing.expectEqualStrings("language_model.model", c.weight_prefix);
    try testing.expectEqual(@as(f32, -5.0), c.kda_gate_lower_bound);
}

test "glm5_next config: an EXL3 streaming pack with fp8 residents parses streamed and decodes serially" {
    const base = @embedFile("../../fixtures/model-configs/glm5_next-mlx-community-4bit.json");
    const bank = "{\"residents_format\":\"fp8_block128\",\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mtplx-expert-manifest-v2\",\"manifest\":\"expert-manifest-v2.json\",\"runtime_manifest\":\"expert-manifest.json\",\"quant\":\"exl3-k3-mul1\"},";
    const json = try std.mem.concat(testing.allocator, u8, &.{ bank, std.mem.trimStart(u8, base, " \n{") });
    defer testing.allocator.free(json);
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(json, true, &p).isOk());
    defer freeParsed(&p);
    try testing.expect(of(&p).streamed and p.caps.streamed_weights and !p.caps.batches_decode);
}

test "glm5_next config: a streamed pack, a RoPE'd sparse attention and broken JSON are refused by name" {
    inline for (.{
        "{\"model_type\":\"glm_moe_dsa\",\"num_hidden_layers\":2}",
        "{\"model_type\":\"glm5_next\",\"expert_streaming\":{},\"num_hidden_layers\":2}",
        "{\"model_type\":\"glm5_next\",\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mlx-stream-nvfp4-expert-bank-v1\",\"manifest\":\"expert-manifest.json\"},\"num_hidden_layers\":2}",
        "{\"model_type\":\"glm5_next_text\",\"num_hidden_layers\":2,\"kv_lora_rank\":64,\"index_kpool\":4,\"index_topk\":8,\"qk_rope_head_dim\":64}",
        "{\"model_type\":\"glm5_next_text\",\"num_hidden_layers\":2,\"index_kpool\":4,\"index_topk\":8,\"top_k_experts\":4}",
    }) |json| {
        var p: abi.Parsed = .{};
        const v = parseJson(json, true, &p);
        try testing.expect(v.code == .unsupported and p.private == null);
    }
    var p: abi.Parsed = .{};
    try testing.expectEqual(abi.Verdict.Code.failed, parseJson("{", true, &p).code);
}
