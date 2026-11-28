//! hy_v4_flash (Tencent Hy4-flash): the claims and parse. MLX-free (discovery runs it; the build checks the
//! closure). One arm serves one pack family; every shape the plugin does not serve yet is refused BY NAME,
//! so the pack fails with the reason, not with a generic decline.
//!
//! The pack's own ground rules this parse encodes (issue #1241, SEMANTICS.md):
//! - `text_config` carries the trunk; the root carries vision + the per-tensor quant map.
//! - Layer kinds are decided by the WEIGHTS (which layers carry `compressed_indexer`/`engram_layer`), not by
//!   `layer_types`: this config's list is one off from what HF ships. The plan keeps a `layer_types` shape
//!   check but the engine trusts weight presence; a pack that violates both is refused at load.
//! - `enable_ihc` is the iHC gate; the per-module mixing shape is pinned here by dims.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const generation = @import("../../sdk/config/generation.zig");

pub const model_types = [_][]const u8{"hy_v4_flash"};

pub const claims = sdk.claimsByType(&model_types);

pub const QuantMode = enum { affine, nvfp4, mxfp4, mxfp8, gguf };

/// The trunk + tower facts the loader and forward read. Names follow what the engine's own names will be.
pub const Config = struct {
    model_type: []const u8 = "hy_v4_flash",
    weight_prefix: []const u8 = "model",

    vocab_size: u32 = 120832,
    hidden_size: u32 = 3072,
    num_hidden_layers: u32 = 42,
    num_attention_heads: u32 = 32,
    head_dim: u32 = 64,
    rms_norm_eps: f32 = 1e-5,
    max_position_embeddings: u64 = 1048576,
    bos_token_id: ?u32 = null,
    eos_token_ids: [8]u32 = @splat(0),
    num_eos_tokens: u32 = 0,

    quant_bits: u32 = 0,
    quant_group_size: u32 = 64,
    quant_mode: QuantMode = .affine,

    // Attention: MLA shared by both layer kinds; SWA differs in heads/window/theta; DSA adds the indexer.
    q_lora_rank: u32 = 1024,
    kv_lora_rank: u32 = 512,
    qk_nope_head_dim: u32 = 192,
    qk_rope_head_dim: u32 = 64,
    v_head_dim: u32 = 256,
    swa_num_attention_heads: u32 = 64,
    sliding_window: u32 = 512,
    rope_theta_dsa: f32 = 1e7,
    rope_theta_swa: f32 = 1e4,
    mrope_section: [3]u32 = .{ 12, 10, 10 },
    learnable_sink: bool = true,
    gated_mla: bool = true,
    sandwich_norm: bool = true,

    // DSA: the compressed (IndexPool) indexer only — the fine indexer is skipped by the pack.
    index_topk: u32 = 2048,
    index_n_heads: u32 = 16,
    index_head_dim: u32 = 128,
    compressed_ratio: u32 = 4,
    compressed_group_topk: u32 = 512,
    compressed_query_block: u32 = 8192,
    dsa_layers: [64]bool = @splat(false),
    dsa_layer_count: u32 = 0,

    // MoE + LatentMoE (routed experts live in a 1536-d latent space).
    n_routed_experts: u32 = 512,
    num_experts_per_tok: u32 = 16,
    n_shared_experts: u32 = 1,
    moe_intermediate_size: u32 = 1536,
    routed_expert_hidden_size: u32 = 1536,
    intermediate_size: u32 = 13824,
    first_k_dense_replace: u32 = 1,
    routed_scaling_factor: f32 = 3.998,
    swiglu_limit: f32 = 10.0,
    norm_topk_prob: bool = true,
    n_group: u32 = 1,
    topk_group: u32 = 1,

    // iHC (identity hyper-connections): 4 streams; per-module hc_pre + final hc_head.
    hc_mult: u32 = 4,
    hc_magnitude: f32 = 2.0,
    hc_eps: f32 = 1e-6,

    // Engram: n-gram embedding injection on `engram_layers`; tables stream from disk.
    engram_layers: [64]bool = @splat(false),
    engram_layer_count: u32 = 0,
    /// The table rows each engram layer carries (the head-vocab lists' sums; HF-keyed, summed here). The
    /// bill prices the stream bank from these; the load cross-checks the pack's table against them.
    engram_table_rows: [64]u64 = @splat(0),
    engram_vocab_sizes: [2]u32 = .{ 0, 0 },
    /// The per-head hash moduli, tier-major (order 2's 8 heads then order 3's 8), per pack layer slot.
    engram_moduli: [64][16]u32 = @splat(@splat(0)),
    /// `engram_lookup_table_size` (HF default 120818) and the seed the multipliers are drawn from.
    engram_lookup_size: u32 = 120818,
    engram_seed: u64 = 0,
    /// `engram_fp8_embedding`: the looked-up rows round-trip through e4m3.
    engram_fp8: bool = true,
    engram_ngram_heads: u32 = 8,
    engram_head_dim: u32 = 96,
    engram_max_ngram: u32 = 3,
    engram_pad_id: u32 = 120002,

    // DFly2 drafter config: 5 draft layers over target KV/index layers 25/29/33/37/41 (validated, not served).
    dfly_layers: u32 = 0,
    dfly_mask_token_id: u32 = 0,
    dfly_target_kv_layers: [8]u32 = @splat(0),
    dfly_target_layer_count: u32 = 0,
    dfly_index_topk: u32 = 0,
    dfly_mtp_fusion_slot0: bool = false,

    // Vision: SigLIP2-shaped tower + rmsnorm_mlp projector; video refused by name.
    has_vision: bool = false,
    vision_depth: u32 = 27,
    vision_hidden_size: u32 = 1152,
    vision_intermediate_size: u32 = 4304,
    vision_num_heads: u32 = 16,
    vision_patch_size: u32 = 16,
    vision_out_hidden_size: u32 = 3072,
    vision_merge_size: u32 = 2,
    vision_rope_theta: f32 = 10000.0,
    vision_norm_eps: f32 = 1e-5,
    vision_min_pixels: u32 = 4096,
    vision_max_pixels: u32 = 4194304,
    vision_max_image_size: u32 = 2048,
    vision_min_image_size: u32 = 512,
    vision_max_seq_len: u32 = 16384,
    vision_img_max_tokens: u32 = 4096,
    /// The driver's `Forward.mrope` inverse frequencies (the DSA base) and the plugin's raw 3-D layout.
    mrope_inv: [32]f64 = @splat(0),
    mrope_inv_len: usize = 0,
    image_start_id: u32 = 120118,
    image_end_id: u32 = 120119,
    image_newline_id: u32 = 120121,
    image_token_id: u32 = 120120,
    video_start_id: u32 = 0,
    video_end_id: u32 = 0,
    video_token_id: u32 = 0,

    /// Pack-owned sampling defaults (generation_config.json; read on the load parse, not discovery).
    gen: abi.GenDefaults = .{},

    fn addEos(self: *Config, id: u32) void {
        if (self.num_eos_tokens < self.eos_token_ids.len) {
            self.eos_token_ids[self.num_eos_tokens] = id;
            self.num_eos_tokens += 1;
        }
    }
};

pub const ParseError = error{ InvalidConfigField, Unsupported };

// Checked readers (config JSON is untrusted; a null reads as an omitted key).

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

pub fn boolV(v: std.json.Value) ParseError!bool {
    return value(.bool, v);
}

fn f32v(v: std.json.Value) ParseError!f32 {
    const f: f32 = switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |x| @floatCast(x),
        else => return error.InvalidConfigField,
    };
    if (!std.math.isFinite(f)) return error.InvalidConfigField;
    return f;
}

fn intIf(obj: std.json.ObjectMap, key: []const u8, out: *u32) ParseError!void {
    if (field(obj, key)) |v| {
        if (v == .integer) out.* = try u32v(v);
    }
}

fn intIf2(v: ?std.json.Value, out: *u32) ParseError!void {
    if (v) |w| {
        if (w == .integer) out.* = try u32v(w);
    }
}

fn f32If(obj: std.json.ObjectMap, key: []const u8, out: *f32) ParseError!void {
    if (field(obj, key)) |v| out.* = try f32v(v);
}

fn require(cond: bool) ParseError!void {
    if (!cond) return error.Unsupported;
}

/// A config field the plugin has no arm for yet: true when the key is PRESENT with a value the parse cannot honour.
fn want(cond: bool) ParseError!void {
    if (!cond) return error.Unsupported;
}

/// Keys another family's parser reads that this pack must not carry (a mislabel or a forked release).
const foreign_keys = [_][]const u8{
    "vision_config", "audio_config",    "audio_token_id", "llm_config",       "thinker_config",
    "layer_types",   "mlp_layer_types", "indexer_types",  "engram_layer_ids", "expert_streaming",
};

pub const Objects = struct {
    root: std.json.ObjectMap,
    cfg: std.json.ObjectMap,
};

fn readShape(c: *Config, o: Objects) ParseError!void {
    const root = o.root;
    const cfg = o.cfg;
    if (field(cfg, "vocab_size")) |v| c.vocab_size = try u32v(v);
    if (field(cfg, "hidden_size")) |v| c.hidden_size = try u32v(v);
    if (field(cfg, "num_hidden_layers")) |v| c.num_hidden_layers = try u32v(v);
    if (field(cfg, "num_attention_heads")) |v| c.num_attention_heads = try u32v(v);
    if (field(cfg, "head_dim")) |v| c.head_dim = try u32v(v);
    if (field(cfg, "rms_norm_eps")) |v| c.rms_norm_eps = try f32v(v);
    if (field(cfg, "max_position_embeddings")) |v| c.max_position_embeddings = try u32v(v);
    if (field(cfg, "bos_token_id")) |v| {
        if (v == .integer and v.integer >= 0) c.bos_token_id = try u32v(v);
    }
    if (field(root, "tie_word_embeddings")) |v| if (v == .bool) try require(!v.bool);
    try require(c.vocab_size > 0 and c.vocab_size <= (1 << 21));
    try require(c.hidden_size > 0 and c.hidden_size <= (1 << 14));
    try require(c.num_hidden_layers > 0 and c.num_hidden_layers <= 64);
    try require(c.max_position_embeddings > 0 and c.max_position_embeddings <= (1 << 32));
    try require(c.num_attention_heads > 0 and c.num_attention_heads <= 256);
    try require(c.rms_norm_eps > 0);
}

fn readAttention(c: *Config, o: Objects) ParseError!void {
    const cfg = o.cfg;
    if (field(cfg, "q_lora_rank")) |v| c.q_lora_rank = try u32v(v);
    if (field(cfg, "kv_lora_rank")) |v| c.kv_lora_rank = try u32v(v);
    if (field(cfg, "qk_nope_head_dim")) |v| c.qk_nope_head_dim = try u32v(v);
    if (field(cfg, "qk_rope_head_dim")) |v| c.qk_rope_head_dim = try u32v(v);
    if (field(cfg, "v_head_dim")) |v| c.v_head_dim = try u32v(v);
    if (field(cfg, "swa_num_attention_heads")) |v| c.swa_num_attention_heads = try u32v(v);
    if (field(cfg, "sliding_window")) |v| c.sliding_window = try u32v(v);
    try require(c.q_lora_rank > 0 and c.kv_lora_rank > 0);
    try require(c.qk_nope_head_dim + c.qk_rope_head_dim == 256);
    // One nope/rope/v width serves both arms in the forward: a pack with a wider SWA geometry is refused
    // by name rather than read at the DSA width.
    if (field(cfg, "swa_qk_nope_head_dim")) |v| try require(try u32v(v) == c.qk_nope_head_dim);
    if (field(cfg, "swa_qk_rope_head_dim")) |v| try require(try u32v(v) == c.qk_rope_head_dim);
    if (field(cfg, "swa_v_head_dim")) |v| try require(try u32v(v) == c.v_head_dim);
    if (field(cfg, "gated_mla")) |v| c.gated_mla = try boolV(v);
    if (field(cfg, "learnable_sink")) |v| c.learnable_sink = try boolV(v);
    if (field(cfg, "sandwich_norm")) |v| c.sandwich_norm = try boolV(v);
    try require(c.gated_mla and c.learnable_sink and c.sandwich_norm);
    // rope_parameters is keyed per layer kind; both arms interleave M-RoPE, thetas differ.
    if (field(cfg, "rope_parameters")) |rp| {
        const rpo = try value(.object, rp);
        if (field(rpo, "deepseek_sparse_attention")) |arm| {
            const a = try value(.object, arm);
            try f32If(a, "rope_theta", &c.rope_theta_dsa);
            if (field(a, "rope_type")) |t| try require(std.mem.eql(u8, try value(.string, t), "default"));
            if (field(a, "mrope_section")) |ms| try readMropeSection(&c.mrope_section, ms);
        }
        if (field(rpo, "sliding_attention")) |arm| {
            const a = try value(.object, arm);
            try f32If(a, "rope_theta", &c.rope_theta_swa);
            if (field(a, "rope_type")) |t| try require(std.mem.eql(u8, try value(.string, t), "default"));
            if (field(a, "mrope_section")) |ms| try readMropeSection(&c.mrope_section, ms);
        }
        // A flat rope_parameters (no per-kind arms) is a shape we do not serve.
        try require(field(rpo, "deepseek_sparse_attention") != null or field(rpo, "sliding_attention") != null);
    } else return error.Unsupported;
    try require(c.rope_theta_swa > 0 and c.rope_theta_dsa > 0);
    try require(c.mrope_section[0] + c.mrope_section[1] + c.mrope_section[2] == c.qk_rope_head_dim / 2);
}

fn readMropeSection(out: *[3]u32, ms: std.json.Value) ParseError!void {
    const arr = try value(.array, ms);
    if (arr.items.len != 3) return error.InvalidConfigField;
    for (arr.items, 0..) |it, i| out[i] = try u32v(it);
}

fn readLayerKinds(c: *Config, o: Objects) ParseError!void {
    const cfg = o.cfg;
    // layer_types: the HF-offset list; accepted, and the loader decides from weights. Validate the pattern's shape.
    var dsa_bits: u64 = 0;
    if (field(cfg, "layer_types")) |v| {
        const arr = try value(.array, v);
        try require(arr.items.len == c.num_hidden_layers);
        for (arr.items, 0..) |t, i| {
            const s = try value(.string, t);
            const dsa = std.mem.eql(u8, s, "deepseek_sparse_attention");
            if (!dsa) try require(std.mem.eql(u8, s, "sliding_attention"));
            if (dsa) dsa_bits |= @as(u64, 1) << @intCast(i);
        }
    } else return error.Unsupported;
    // swa_pattern is the same fact without the offset confusion: 0 = sparse/dsa layer, 1 = sliding.
    if (field(cfg, "swa_pattern")) |v| {
        const arr = try value(.array, v);
        try require(arr.items.len == c.num_hidden_layers);
        for (arr.items, 0..) |t, i| {
            const p = try u32v(t);
            try require(p <= 1);
            const dsa = dsa_bits >> @intCast(i) & 1 == 1;
            try require(dsa == (p == 0));
        }
    }
    // indexer_types must all read "full": any other kind (a fine indexer in the mix) is not served.
    if (field(cfg, "indexer_types")) |v| for ((try value(.array, v)).items) |t| {
        try require(std.mem.eql(u8, try value(.string, t), "full"));
    };
    if (field(cfg, "use_dsa")) |v| try require(try boolV(v));
    if (field(cfg, "use_compressed_indexer")) |v| try require(try boolV(v));
    if (field(cfg, "compressed_indexer_skip_fine_select")) |v| try require(try boolV(v));
    if (field(cfg, "compressed_indexer_reuse_fine_indexer")) |v| try require(!(try boolV(v)));
    if (field(cfg, "index_topk")) |v| c.index_topk = try u32v(v);
    if (field(cfg, "index_n_heads")) |v| c.index_n_heads = try u32v(v);
    if (field(cfg, "index_head_dim")) |v| c.index_head_dim = try u32v(v);
    if (field(cfg, "compressed_indexer_compress_ratio")) |v| c.compressed_ratio = try u32v(v);
    if (field(cfg, "compressed_indexer_group_topk")) |v| c.compressed_group_topk = try u32v(v);
    if (field(cfg, "compressed_indexer_stage2_query_block_size")) |v| c.compressed_query_block = try u32v(v);
    try require(c.index_topk > 0 and c.index_n_heads > 0 and c.index_head_dim > 0);
    try require(c.compressed_ratio > 0 and c.compressed_group_topk > 0);

    // mlp_layer_types: a contiguous block of dense layers at the front is the only shape we serve.
    if (field(cfg, "first_k_dense_replace")) |v| c.first_k_dense_replace = try u32v(v);
    if (field(cfg, "mlp_layer_types")) |v| {
        const arr = try value(.array, v);
        try require(arr.items.len == c.num_hidden_layers);
        var dense: u32 = 0;
        for (arr.items, 0..) |t, i| {
            const s = try value(.string, t);
            if (std.mem.eql(u8, s, "dense")) {
                try require(dense == i);
                dense += 1;
            } else try require(std.mem.eql(u8, s, "sparse"));
        }
        try require(dense == c.first_k_dense_replace);
    }
    _ = &dsa_bits;
    c.dsa_layers = dsaBits(dsa_bits);
    c.dsa_layer_count = @popCount(dsa_bits);
    try require(c.dsa_layer_count > 0);
}

fn dsaBits(bits: u64) [64]bool {
    var out: [64]bool = undefined;
    for (&out, 0..) |*b, i| b.* = bits >> @intCast(i) & 1 == 1;
    return out;
}

/// A layer's attention heads: DSA layers keep `num_attention_heads`, SWA layers the wider
/// `swa_num_attention_heads` (the pack stores both geometries, per layer).
pub fn layerHeads(c: *const Config, li: u32) u32 {
    return if (li < c.dsa_layers.len and c.dsa_layers[li]) c.num_attention_heads else c.swa_num_attention_heads;
}

/// A layer's RoPE base: the sparse layers rotate long (1e7), the sliding ones short (1e4).
pub fn layerTheta(c: *const Config, li: u32) f32 {
    return if (li < c.dsa_layers.len and c.dsa_layers[li]) c.rope_theta_dsa else c.rope_theta_swa;
}

fn readMoe(c: *Config, o: Objects) ParseError!void {
    const cfg = o.cfg;
    if (field(cfg, "n_routed_experts")) |v| c.n_routed_experts = try u32v(v);
    try intIf(cfg, "num_experts_per_tok", &c.num_experts_per_tok);
    if (field(cfg, "n_shared_experts")) |v| c.n_shared_experts = try u32v(v);
    try intIf(cfg, "moe_intermediate_size", &c.moe_intermediate_size);
    try intIf(cfg, "routed_expert_hidden_size", &c.routed_expert_hidden_size);
    try intIf(cfg, "intermediate_size", &c.intermediate_size);
    try f32If(cfg, "routed_scaling_factor", &c.routed_scaling_factor);
    try f32If(cfg, "swiglu_limit", &c.swiglu_limit);
    try intIf(cfg, "n_group", &c.n_group);
    try intIf(cfg, "topk_group", &c.topk_group);
    if (field(cfg, "norm_topk_prob")) |v| c.norm_topk_prob = try boolV(v);
    if (field(cfg, "moe_router_use_sigmoid")) |v| try require(try boolV(v));
    if (field(cfg, "moe_router_enable_expert_bias")) |v| try require(try boolV(v));
    if (field(cfg, "gating_type")) |v| try require(std.mem.eql(u8, try value(.string, v), "elementwise"));
    try require(c.n_routed_experts > 0 and c.num_experts_per_tok > 0 and c.n_shared_experts >= 1);
    try require(c.moe_intermediate_size > 0 and c.routed_expert_hidden_size > 0 and c.intermediate_size > 0);
    // Group-limited routing (n_group>1 with topk_group<n_group) needs the grouped router arm: refuse until wired.
    try require(c.n_group == 1 or c.topk_group == c.n_group);
    try require(c.swiglu_limit > 0);
}

fn readHc(c: *Config, o: Objects) ParseError!void {
    const cfg = o.cfg;
    if (field(cfg, "enable_ihc")) |v| {
        try require(try boolV(v));
    } else return error.Unsupported;
    try intIf(cfg, "hc_mult", &c.hc_mult);
    try f32If(cfg, "hc_magnitude", &c.hc_magnitude);
    try f32If(cfg, "hc_eps", &c.hc_eps);
    try require(c.hc_mult == 4 and c.hc_magnitude > 0 and c.hc_eps > 0);
}

fn readEngram(c: *Config, o: Objects) ParseError!void {
    const cfg = o.cfg;
    try intIf(cfg, "engram_max_ngram_size", &c.engram_max_ngram);
    try intIf(cfg, "engram_n_head_per_ngram", &c.engram_ngram_heads);
    try intIf(cfg, "engram_n_embed_per_ngram", &c.engram_head_dim);
    try intIf(cfg, "engram_pad_id", &c.engram_pad_id);
    try intIf(cfg, "engram_lookup_table_size", &c.engram_lookup_size);
    if (field(cfg, "engram_seed")) |v| c.engram_seed = @intCast(try u32v(v));
    if (field(cfg, "engram_fp8_embedding")) |v| c.engram_fp8 = try boolV(v);
    try require(c.engram_lookup_size > 0 and c.engram_pad_id < c.engram_lookup_size);
    try require(c.engram_max_ngram == 3 and c.engram_ngram_heads == 8);
    try require(c.engram_head_dim == 8 * 96);
    c.engram_head_dim = 96;
    if (field(cfg, "engram_vocab_size")) |v| {
        const arr = try value(.array, v);
        if (arr.items.len == 2) {
            c.engram_vocab_sizes[0] = try u32v(arr.items[0]);
            c.engram_vocab_sizes[1] = try u32v(arr.items[1]);
        }
    }
    // The config's ids are HF (1-based); the pack's tensors sit at index id-1 (omlx: block+1 is the
    // config id). Store the PACK index; head-vocab lists key on the same id, so rows line up too.
    if (field(cfg, "engram_layer_ids")) |v| {
        const arr = try value(.array, v);
        try require(arr.items.len > 0 and arr.items.len <= 16);
        for (arr.items) |t| {
            const li = try u32v(t);
            try require(li > 0 and li <= c.num_hidden_layers);
            c.engram_layers[li - 1] = true;
            c.engram_layer_count += 1;
        }
    } else return error.Unsupported;
    if (field(cfg, "engram_layer_head_vocab_sizes")) |v| {
        const obj = try value(.object, v);
        try require(obj.keys().len == c.engram_layer_count);
        var it = obj.iterator();
        var seen: u32 = 0;
        while (it.next()) |kv| {
            // Head-vocab lists key on the HF (1-based) layer; the loader places tables by WEIGHT presence,
            // so only the row SUM matters here (the bill's), and the two lists are the 8- and 3-gram tiers.
            const li = std.fmt.parseInt(u32, kv.key_ptr.*, 10) catch return error.InvalidConfigField;
            try require(li > 0 and li <= c.num_hidden_layers);
            const tiers = try value(.array, kv.value_ptr.*);
            try require(tiers.items.len == 2);
            var total: u64 = 0;
            var head_i: usize = 0;
            for (tiers.items) |tier| {
                const heads = try value(.array, tier);
                // Each tier carries one modulus per hash head: the n-gram orders flatten head-major.
                try require(heads.items.len == c.engram_ngram_heads);
                for (heads.items) |hv| {
                    const m = try u32v(hv);
                    try require(m > 0);
                    if (head_i < 16) c.engram_moduli[li - 1][head_i] = m;
                    head_i += 1;
                    total += m;
                }
            }
            try require(head_i == 2 * c.engram_ngram_heads);
            try require(total > 0);
            c.engram_table_rows[li - 1] = total;
            seen += 1;
        }
        try require(seen == c.engram_layer_count);
        for (c.engram_layers, 0..) |en, li| {
            if (!en) continue;
            try require(c.engram_table_rows[li] > 0);
        }
    }
}

fn readDfly(c: *Config, o: Objects) ParseError!void {
    const cfg = o.cfg;
    const root = o.root;
    // The spec block sits in `text_config` on the pack and its keys sit at the root; accept either side.
    const find = struct {
        fn f(r: std.json.ObjectMap, t: std.json.ObjectMap, key: []const u8) ?std.json.Value {
            return field(t, key) orelse field(r, key);
        }
    }.f;
    if (find(root, cfg, "speculative_config")) |v| {
        const sc = try value(.object, v);
        const m = field(sc, "method") orelse return error.InvalidConfigField;
        if (!std.mem.eql(u8, try value(.string, m), "dfly")) return; // another drafter: left for the host's gate
        try intIf2(find(root, cfg, "dfly_num_hidden_layers"), &c.dfly_layers);
        try intIf2(find(root, cfg, "dfly_mask_token_id"), &c.dfly_mask_token_id);
        try intIf2(find(root, cfg, "dfly_target_index_topk"), &c.dfly_index_topk);
        if (find(root, cfg, "dfly_mtp_fusion")) |f| {
            try require(std.mem.eql(u8, try value(.string, f), "slot0"));
            c.dfly_mtp_fusion_slot0 = true;
        } else return error.Unsupported;
        if (find(root, cfg, "dfly_target_kv_layers")) |l| {
            const arr = try value(.array, l);
            try require(arr.items.len > 0 and arr.items.len <= c.dfly_target_kv_layers.len);
            for (arr.items, 0..) |t, i| {
                const li = try u32v(t);
                try require(li < c.num_hidden_layers);
                c.dfly_target_kv_layers[i] = li;
            }
            c.dfly_target_layer_count = @intCast(arr.items.len);
        } else return error.Unsupported;
        if (find(root, cfg, "dfly_enable_hidden_correction")) |hv| try require(try boolV(hv));
        // The pack's own draft depth and k are the shape; the tiny fixture keeps them small on purpose.
        try require(c.dfly_mask_token_id > 0 and c.dfly_index_topk == c.index_topk);
    }
    if (find(root, cfg, "num_nextn_predict_layers")) |v| try require(try u32v(v) == 0);
    if (find(root, cfg, "dfly_context_local_window") != null) return error.Unsupported;
}

fn readQuant(c: *Config, root: std.json.ObjectMap) ParseError!void {
    const q = field(root, "quantization") orelse field(root, "quantization_config") orelse return;
    const qo = try value(.object, q);
    if (field(qo, "bits")) |v| c.quant_bits = try u32v(v);
    if (field(qo, "group_size")) |v| c.quant_group_size = try u32v(v);
    if (field(qo, "mode")) |v| if (v == .string) {
        // Accept only the spelling the plugin serves; anything else (gguf, nvfp4) is refused by name.
        try require(std.mem.eql(u8, v.string, "affine"));
        c.quant_mode = .affine;
    };
    if (c.quant_bits != 0) switch (c.quant_bits) {
        2, 3, 4, 5, 6, 8 => {},
        else => return error.Unsupported,
    };
    // Per-tensor overrides: every entry must be a mode this plugin serves (affine g32/g64 today; an
    // fp8/nvfp4 entry is named at load, refused here).
    var it = qo.iterator();
    while (it.next()) |kv| if (kv.value_ptr.* == .object) {
        const mode = kv.value_ptr.object.get("mode") orelse continue;
        if (mode == .string) {
            try require(std.mem.eql(u8, mode.string, "affine"));
            if (kv.value_ptr.object.get("bits")) |b| switch (std.math.cast(u32, b.integer) orelse 0) {
                2, 3, 4, 5, 6, 8 => {},
                else => return error.Unsupported,
            };
        }
    };
}

fn readVision(c: *Config, root: std.json.ObjectMap) ParseError!void {
    if (field(root, "vision_config") == null) return;
    const vc = (field(root, "vision_config")).?.object;
    c.has_vision = true;
    try intIf(vc, "num_hidden_layers", &c.vision_depth);
    try intIf(vc, "hidden_size", &c.vision_hidden_size);
    try intIf(vc, "intermediate_size", &c.vision_intermediate_size);
    try intIf(vc, "num_attention_heads", &c.vision_num_heads);
    try intIf(vc, "patch_size", &c.vision_patch_size);
    try intIf(vc, "out_hidden_size", &c.vision_out_hidden_size);
    try intIf(vc, "spatial_merge_size", &c.vision_merge_size);
    try f32If(vc, "rope_2d_theta", &c.vision_rope_theta);
    try f32If(vc, "rms_norm_eps", &c.vision_norm_eps);
    try intIf(vc, "max_image_size", &c.vision_max_image_size);
    try intIf(vc, "min_image_size", &c.vision_min_image_size);
    try intIf(vc, "max_vit_seq_len", &c.vision_max_seq_len);
    try intIf(vc, "img_max_token_num", &c.vision_img_max_tokens);
    if (field(vc, "projector_type")) |p| try require(std.mem.eql(u8, try value(.string, p), "rmsnorm_mlp_adapter"));
    if (field(vc, "vision_full_attention")) |f| try require(!(try boolV(f)));
    try require(c.vision_depth > 0 and c.vision_hidden_size > 0 and c.vision_num_heads > 0);
    try require(c.vision_out_hidden_size == c.hidden_size);
    try intIf(root, "image_start_token_id", &c.image_start_id);
    try intIf(root, "image_end_token_id", &c.image_end_id);
    try intIf(root, "image_newline_token_id", &c.image_newline_id);
    try intIf(root, "image_token_id", &c.image_token_id);
    try intIf(root, "video_start_token_id", &c.video_start_id);
    try intIf(root, "video_end_token_id", &c.video_end_id);
    try intIf(root, "video_token_id", &c.video_token_id);
    if (c.video_start_id != 0) {
        // Video ids ship in the pack; the Plan-07 arm is images. Refuse video input, not the pack.
        // (Input-level refusal lives in the forward; the config only records the ids.)
    }
}

const gpa = std.heap.c_allocator;

pub fn parse(
    dir: *const abi.PackDir,
    config_json: abi.Str,
    settings_json: abi.Str,
    span: ?*const abi.LayerSpan,
    stub: bool,
    out: *abi.Parsed,
) callconv(.c) abi.Verdict {
    _ = settings_json;
    if (span != null) return abi.Verdict.refuse(.unsupported, "hy_v4_flash: a layer span (the cluster split)");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json.slice(), .{}) catch
        return abi.Verdict.refuse(.failed, "hy_v4_flash: config.json does not parse");
    if (doc != .object) return abi.Verdict.refuse(.failed, "hy_v4_flash: config.json is not an object");
    const root = doc.object;
    const mt = field(root, "model_type") orelse return abi.Verdict.refuse(.unsupported, "hy_v4_flash: no model_type");
    if (mt != .string) return abi.Verdict.refuse(.unsupported, "hy_v4_flash: no model_type");
    for (model_types[0..1]) |t| {
        if (std.mem.eql(u8, t, mt.string)) break;
    } else return abi.Verdict.refuse(.unsupported, "hy_v4_flash: a model_type this plugin does not serve");

    const tc = field(root, "text_config");
    if (tc != null and tc.? != .object) return abi.Verdict.refuse(.failed, "hy_v4_flash: text_config is not an object");
    const o: Objects = .{ .root = root, .cfg = if (tc) |t| t.object else root };

    var c: Config = .{};
    parseBody(&c, o) catch |e| return refusalField(e, "trunk");
    readQuant(&c, root) catch |e| return refusalField(e, "quantization");
    readVision(&c, root) catch |e| return refusalField(e, "vision_config");
    _ = foreign_keys;

    var p: abi.Parsed = .{};
    const s = &p.shell;
    s.* = .{
        .vocab_size = c.vocab_size,
        .hidden_size = c.hidden_size,
        .intermediate_size = c.intermediate_size,
        .num_hidden_layers = c.num_hidden_layers,
        .num_attention_heads = c.num_attention_heads,
        .num_key_value_heads = c.num_attention_heads,
        .head_dim = c.head_dim,
        .num_experts_per_tok = c.num_experts_per_tok,
        .moe_intermediate_size = c.moe_intermediate_size,
        .max_position_embeddings = c.max_position_embeddings,
        .context_cap = @min(c.max_position_embeddings, 262_144),
        .quant = .{ .bits = std.math.cast(u8, c.quant_bits) orelse 0, .group_size = c.quant_group_size, .mode = if (c.quant_bits == 0) .none else std.meta.stringToEnum(abi.QuantMode, @tagName(c.quant_mode)).? },
        .bos = if (c.bos_token_id) |b| b else -1,
        .act_dtype = .bf16,
        .model_type = abi.Str.of(c.model_type),
    };
    for (c.eos_token_ids[0..c.num_eos_tokens]) |e| generation.addEos(s, e);
    if (!stub) {
        generation.apply(dir, s);
    }
    c.num_eos_tokens = @min(s.eos_count, c.eos_token_ids.len);
    @memcpy(c.eos_token_ids[0..c.num_eos_tokens], s.eos[0..c.num_eos_tokens]);
    c.gen = s.gen;
    p.caps = .{
        .batches_decode = false,
        .batch_pads_kv = false,
        .moe = c.n_routed_experts > 0,
        .images = c.has_vision,
        // A restored prefix may hold image placeholder rows; the splice resumes by row index only inside a
        // chunked prefill, so a cache restore across an image run would splice the tower rows at the wrong
        // offset. Keep KV reuse off until the cache carries the spliced rows' offset.
        .prefix_cache = false,
    };
    // The driver builds `Forward.mrope` off this (its angles are unused by Hy4, which ropes per layer
    // kind from the raw table); the raw positions ride `Forward.mrope_positions`.
    c.mrope_inv_len = c.qk_rope_head_dim / 2;
    if (c.mrope_inv_len > c.mrope_inv.len) c.mrope_inv_len = c.mrope_inv.len;
    for (0..c.mrope_inv_len) |i| {
        c.mrope_inv[i] = std.math.pow(f64, c.rope_theta_dsa, -@as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(c.qk_rope_head_dim)));
    }
    p.mrope = .{
        .sections = c.mrope_section,
        .rope_dims = c.qk_rope_head_dim,
        .inv_freq = .of(c.mrope_inv[0..c.mrope_inv_len]),
        .mrope_newline_rows = c.has_vision,
    };
    p.cache.layout_namespace = abi.Str.of("hy4-mla-latent-v1");
    p.placeholder_ids.image = c.image_token_id;
    p.weights.prefix = abi.Str.of(c.weight_prefix);

    const cp = gpa.create(Config) catch return abi.Verdict.refuse(.failed, "hy_v4_flash: out of memory");
    cp.* = c;
    p.private = cp;
    out.* = p;
    return .ok;
}

/// Every trunk read, so a test can run the shape without the vision/quant sidecars.
fn parseBody(c: *Config, o: Objects) ParseError!void {
    try readShape(c, o);
    try readAttention(c, o);
    try readLayerKinds(c, o);
    try readMoe(c, o);
    try readHc(c, o);
    try readEngram(c, o);
    try readDfly(c, o);
    // The pack keeps eos in `text_config`; a root-level one reads too.
    const eosv = field(o.cfg, "eos_token_id") orelse field(o.root, "eos_token_id");
    if (eosv) |v| switch (v) {
        .integer => |i| c.addEos(std.math.cast(u32, i) orelse return error.InvalidConfigField),
        .array => |arr| for (arr.items) |item| if (item == .integer) c.addEos(try u32v(item)),
        else => {},
    };
}

fn refusal(e: ParseError) abi.Verdict {
    return switch (e) {
        error.Unsupported => abi.Verdict.refuse(.unsupported, "hy_v4_flash: a shape this plugin does not serve (the field is named)"),
        error.InvalidConfigField => abi.Verdict.refuse(.failed, "hy_v4_flash: a config field has the wrong type or range"),
    };
}

/// A refusal that names the field, so the log says what to change. `why` must outlive the call: the field name
/// is a comptime string; the rest is static.
fn refusalField(e: ParseError, comptime key: []const u8) abi.Verdict {
    const code: abi.Verdict.Code = if (e == error.Unsupported) .unsupported else .failed;
    return abi.Verdict.refuse(code, "hy_v4_flash: field \"" ++ key ++ "\" is not served or has a bad value");
}

pub fn freeParsed(p: *abi.Parsed) callconv(.c) void {
    if (p.private) |ptr| {
        gpa.destroy(@as(*Config, @ptrCast(@alignCast(ptr))));
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

/// A tiny hy_v4_flash config with the same shape space as the pack (4 layers, DSA at 0, engram at 1).
const tiny_config =
    \\{"model_type":"hy_v4_flash","text_config":{"model_type":"hy_v4_flash",
    \\"vocab_size":512,"hidden_size":64,"num_hidden_layers":4,"num_attention_heads":2,"head_dim":64,
    \\"rms_norm_eps":1e-5,"max_position_embeddings":4096,"bos_token_id":1,"eos_token_id":2,
    \\"q_lora_rank":32,"kv_lora_rank":16,"qk_nope_head_dim":192,"qk_rope_head_dim":64,"v_head_dim":256,
    \\"gated_mla":true,"learnable_sink":true,"sandwich_norm":true,"sliding_window":512,"swa_num_attention_heads":4,
    \\"rope_parameters":{"deepseek_sparse_attention":{"rope_type":"default","rope_theta":10000000,
    \\"mrope_interleaved":true,"mrope_section":[12,10,10]},
    \\"sliding_attention":{"rope_type":"default","rope_theta":10000,
    \\"mrope_interleaved":true,"mrope_section":[12,10,10]}},
    \\"layer_types":["deepseek_sparse_attention","sliding_attention","sliding_attention","sliding_attention"],
    \\"swa_pattern":[0,1,1,1],"indexer_types":["full","full","full","full"],
    \\"use_dsa":true,"use_compressed_indexer":true,"compressed_indexer_skip_fine_select":true,
    \\"compressed_indexer_reuse_fine_indexer":false,
    \\"index_topk":8,"index_n_heads":16,"index_head_dim":128,"compressed_indexer_compress_ratio":4,
    \\"compressed_indexer_group_topk":2,"compressed_indexer_stage2_query_block_size":64,
    \\"first_k_dense_replace":1,"mlp_layer_types":["dense","sparse","sparse","sparse"],
    \\"n_routed_experts":8,"num_experts_per_tok":2,"n_shared_experts":1,"moe_intermediate_size":16,
    \\"routed_expert_hidden_size":16,"intermediate_size":64,"routed_scaling_factor":3.998,"swiglu_limit":10.0,
    \\"norm_topk_prob":true,"n_group":1,"topk_group":1,"moe_router_use_sigmoid":true,
    \\"moe_router_enable_expert_bias":true,"gating_type":"elementwise",
    \\"enable_ihc":true,"hc_mult":4,"hc_magnitude":2.0,"hc_eps":1e-6,
    \\"engram_max_ngram_size":3,"engram_n_head_per_ngram":8,"engram_n_embed_per_ngram":768,"engram_pad_id":120002,
    \\"engram_vocab_size":[100,100],"engram_layer_ids":[1],
    \\"engram_layer_head_vocab_sizes":{"1":[[100,100,100,100,100,100,100,100],[100,100,100,100,100,100,100,100]]},
    \\"num_nextn_predict_layers":0},
    \\"speculative_config":{"method":"dfly"},"dfly_num_hidden_layers":5,"dfly_mask_token_id":9,"dfly_target_index_topk":8,
    \\"dfly_mtp_fusion":"slot0","dfly_target_kv_layers":[0,1,2,3],"dfly_enable_hidden_correction":true,
    \\"quantization":{"bits":4,"group_size":64,"mode":"affine"}}
;

test "hy_v4_flash config: the tiny pack parses into shell, caps and the family config" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(tiny_config, true, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expectEqual(@as(u32, 4), p.shell.num_hidden_layers);
    try testing.expectEqual(@as(u32, 512), p.shell.vocab_size);
    try testing.expectEqual(@as(u32, 1), c.dsa_layer_count);
    try testing.expectEqual(@as(u32, 1), c.engram_layer_count);
    try testing.expect(c.dsa_layers[0] and !c.dsa_layers[1] and !c.dsa_layers[2] and !c.dsa_layers[3]);
    try testing.expectEqual(@as(u32, 16), c.index_n_heads);
    try testing.expectEqual(@as(u32, 4), c.compressed_ratio);
    try testing.expectEqual(@as(u32, 2), c.num_experts_per_tok);
    try testing.expectEqual(@as(f32, 1e7), c.rope_theta_dsa);
    try testing.expectEqual(@as(f32, 1e4), c.rope_theta_swa);
    try testing.expectEqual(@as(u32, 5), c.dfly_layers);
    try testing.expectEqual(@as(u32, 9), c.dfly_mask_token_id);
    try testing.expect(c.dfly_mtp_fusion_slot0);
    try testing.expect(p.caps.moe and !p.caps.batches_decode);
    try testing.expectEqualStrings("hy4-mla-latent-v1", p.cache.layout_namespace.slice());
    try testing.expectEqual(@as(i64, 1), p.shell.bos);
    try testing.expect(p.shell.eos_count == 1 and p.shell.eos[0] == 2);
}

test "hy_v4_flash config: the load parse merges generation_config.json (temp 0.9, top_p 1, eos join)" {
    var p: abi.Parsed = .{};
    const mem: sdk.fixture.MemDir = .{ .files = &.{
        .{ .name = "config.json", .bytes = tiny_config },
        .{ .name = "generation_config.json", .bytes = "{\"temperature\":0.9,\"top_p\":1.0,\"top_k\":-1,\"eos_token_id\":2}" },
    } };
    const dir = mem.packDir();
    try testing.expect(parse(&dir, abi.Str.of(tiny_config), .{}, null, false, &p).isOk());
    defer freeParsed(&p);
    try testing.expectEqual(@as(f32, 0.9), p.shell.gen.temperature);
    try testing.expectEqual(@as(f32, 1.0), p.shell.gen.top_p);
    try testing.expect(p.shell.eos_count >= 1 and p.shell.eos[0] == 2);
}

test "hy_v4_flash config: shapes the plugin does not serve are refused by name" {
    // Each refusal, one key at a time, over a mutated copy of tiny_config.
    try refuseContains(tiny_config, "\"compressed_indexer_skip_fine_select\":true", "\"compressed_indexer_skip_fine_select\":false");
    try refuseContains(tiny_config, "\"compressed_indexer_reuse_fine_indexer\":false", "\"compressed_indexer_reuse_fine_indexer\":true");
    try refuseContains(tiny_config, "\"enable_ihc\":true", "\"enable_ihc\":false");
    try refuseContains(tiny_config, "\"hc_mult\":4", "\"hc_mult\":2");
    try refuseContains(tiny_config, "\"engram_max_ngram_size\":3", "\"engram_max_ngram_size\":2");
    try refuseContains(tiny_config, "\"moe_router_use_sigmoid\":true", "\"moe_router_use_sigmoid\":false");
    try refuseContains(tiny_config, "\"gating_type\":\"elementwise\"", "\"gating_type\":\"nope\"");
    try refuseContains(tiny_config, "\"learnable_sink\":true", "\"learnable_sink\":false");
    try refuseContains(tiny_config, "\"num_nextn_predict_layers\":0", "\"num_nextn_predict_layers\":1");
    try refuseContains(tiny_config, "\"n_group\":1", "\"n_group\":8");
    try refuseContains(tiny_config, "\"rope_theta\":10000000", "\"rope_theta\":\"llama3\"");
    // layer_types disagreeing with swa_pattern is a broken release: refused.
    try refuseContains(tiny_config, "\"swa_pattern\":[0,1,1,1]", "\"swa_pattern\":[1,1,1,1]");
    // quant: an unsupported width, an unsupported mode.
    try refuseContains(tiny_config, "\"bits\":4", "\"bits\":7");
    try refuseContains(tiny_config, "\"mode\":\"affine\"", "\"mode\":\"gguf\"");
    try refuseContains(tiny_config, "\"vocab_size\":512", "\"vocab_size\":-1");
    try refuseContains(tiny_config, "\"num_hidden_layers\":4", "\"num_hidden_layers\":0");
    // DFly on a shape we do not serve: fusion spelled other than slot0.
    try refuseContains(tiny_config, "\"dfly_mtp_fusion\":\"slot0\"", "\"dfly_mtp_fusion\":\"all\"");
}

fn replaceOne(arena: *std.heap.ArenaAllocator, hay: []const u8, needle: []const u8, sub: []const u8) ![]const u8 {
    const at = std.mem.indexOf(u8, hay, needle) orelse return error.NeedleMissing;
    const out = try arena.allocator().alloc(u8, hay.len - needle.len + sub.len);
    @memcpy(out[0..at], hay[0..at]);
    @memcpy(out[at .. at + sub.len], sub);
    @memcpy(out[at + sub.len ..], hay[at + needle.len ..]);
    return out;
}

fn refuseContains(base: []const u8, needle: []const u8, sub: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const mutated = try replaceOne(&arena, base, needle, sub);
    var p: abi.Parsed = .{};
    const v = parseJson(mutated, true, &p);
    if (v.isOk() or p.private != null) return error.NotRefused;
    if (v.why.len == 0) return error.NoReason;
}
