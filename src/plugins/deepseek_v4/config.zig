//! DeepSeek-V4-Flash's claims and parse, and the family's config both DeepSeek plugins share (V4.1 extends it).
//! MLX-free (discovery runs it): fields are named as the engine's `ModelConfig` names them, so the module that still
//! reads `ModelConfig` gets them one to one (`model_config.zig`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const generation = @import("../../sdk/config/generation.zig");
const sdk = @import("../../sdk/root.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "deepseek_v4",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"ddalcu/DeepSeek-V4-Flash-MLX-Serve"},
};

pub const model_types = [_][]const u8{"deepseek_v4"};

pub const claims = sdk.claimsByType(&model_types);

pub const QuantMode = enum { affine, nvfp4, mxfp4, mxfp8, gguf };

/// The family's config. Defaults are the engine's, so a key a pack omits reads as it always has.
pub const Config = struct {
    model_type: []const u8 = "deepseek_v4",
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
    rope_scaling_factor: f32 = 1.0,
    has_sliding_window: bool = true,
    sliding_window: u32 = 1024,
    tie_word_embeddings: bool = false,
    bos_token_id: ?u32 = null,
    eos_token_ids: [8]u32 = @splat(0),
    num_eos_tokens: u32 = 0,
    quant_bits: u32 = 0,
    quant_group_size: u32 = 64,
    quant_mode: QuantMode = .affine,

    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    moe_intermediate_size: u32 = 0,
    shared_expert_intermediate_size: u32 = 0,
    router_scaling_factor: f32 = 1.0,
    moe_route_norm: bool = true,

    rope_yarn: bool = false,
    yarn_factor: f32 = 1.0,
    yarn_orig_max_pos: u32 = 0,
    yarn_beta_fast: f32 = 32.0,
    yarn_beta_slow: f32 = 1.0,

    dsv4_q_lora_rank: u32 = 0,
    dsv4_o_lora_rank: u32 = 0,
    dsv4_o_groups: u32 = 0,
    dsv4_rope_head_dim: u32 = 0,
    dsv4_hash_layers: u32 = 0,
    dsv4_index_n_heads: u32 = 0,
    dsv4_index_head_dim: u32 = 0,
    dsv4_index_topk: u32 = 0,
    dsv4_hc_mult: u32 = 0,
    dsv4_hc_sinkhorn_iters: u32 = 0,
    dsv4_hc_eps: f32 = 1e-6,
    dsv4_swiglu_limit: f32 = 0.0,
    dsv4_compress_rope_theta: f32 = 0.0,
    /// Per-layer compression ratio (0 = pure sliding window); entries past the trunk are the DSpark stages.
    dsv4_compress_ratios: [128]u8 = @splat(0),
    dsv4_n_compress_ratios: u32 = 0,
    dsv4_mtp_layers: u32 = 0,
    dsv4_dspark_block_size: u32 = 0,
    dsv4_dspark_noise_token_id: u32 = 0,
    dsv4_dspark_markov_rank: u32 = 0,
    dsv4_dspark_target_layers: [8]u8 = @splat(0),
    dsv4_n_dspark_target_layers: u32 = 0,

    // V4.1 only (`deepseek_v41/config.zig`): bit i = layer i.
    dsv41_kv_sources: u64 = 0,
    dsv41_index_sources: u64 = 0,
    dsv41_candidate_source: i32 = -1,
    dsv41_candidate_topk_blocks: u32 = 0,
    dsv41_candidate_block_size: u32 = 0,
    dsv41_engram_layers: [4]u8 = @splat(0),
    dsv41_engram_rows: [4]u64 = @splat(0),
    dsv41_n_engram_layers: u32 = 0,
    dsv41_engram_max_ngram: u32 = 0,
    dsv41_engram_vocab_size: u64 = 0,
    dsv41_engram_n_heads: u32 = 0,
    dsv41_engram_head_dim: u32 = 0,
    dsv41_engram_pad_id: u32 = 0,
    dsv41_engram_compressed_vocab: u32 = 0,
    dsv41_dspark_experts: u32 = 0,
    dsv41_dspark_top_k: u32 = 0,

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
    dsv41_vision: bool = false,
    /// V4.1's routed experts streamed (`deepseek_v41/plugin.zig`): from an `experts.bin` EXL3 bank
    /// (`dsv41_exl3_bank`), else from the MLX pack's own shards (the `stream_experts` setting).
    dsv41_streamed: bool = false,
    dsv41_exl3_bank: bool = false,
    dv_downsample: u32 = 3,
    dv_max_image_tokens: u32 = 1024,
    dv_min_pixels: u32 = 295936,
    dv_max_wh_ratio: u32 = 0,

    /// Pack-owned sampling recommendations (`generation_config.json`), resolved as the host reads them.
    gen: abi.GenDefaults = .{},

    pub fn isV41(self: *const Config) bool {
        return std.mem.eql(u8, self.model_type, "deepseek_v41");
    }

    fn addEos(self: *Config, id: u32) void {
        if (self.num_eos_tokens < self.eos_token_ids.len) {
            self.eos_token_ids[self.num_eos_tokens] = id;
            self.num_eos_tokens += 1;
        }
    }

    /// DSpark stages the pack declares: ratio-table entries past the trunk, gated on a block size.
    pub fn dsparkStages(self: *const Config) u32 {
        if (self.dsv4_dspark_block_size == 0 or self.dsv4_n_compress_ratios <= self.num_hidden_layers) return 0;
        return self.dsv4_n_compress_ratios - self.num_hidden_layers;
    }

    /// The longest sequence the rope resolves: `original_max_position_embeddings x factor` under YaRN.
    pub fn contextCap(self: *const Config) u64 {
        const declared: u64 = self.max_position_embeddings;
        if (!self.rope_yarn) return declared;
        const scaled: f64 = @floor(@as(f64, @floatFromInt(self.yarn_orig_max_pos)) * @as(f64, @floatCast(self.yarn_factor)));
        const max_u32: f64 = @floatFromInt(std.math.maxInt(u32));
        const window: u64 = if (scaled >= max_u32) std.math.maxInt(u32) else @intFromFloat(scaled);
        return if (declared == 0) window else @min(window, declared);
    }
};

pub const ParseError = error{ InvalidConfigField, UnsupportedQuantMode, UnsupportedQuantBits, UnsupportedDsv4Config, UnsupportedDsv41Config, Dsv4Preview };

/// config.json's objects the parse reads: `outer` (quantization), `root` (eos, vision, image ids), `cfg` (the text
/// model: `text_config` when nested, else the root).
pub const Json = struct {
    outer: std.json.ObjectMap,
    root: std.json.ObjectMap,
    cfg: std.json.ObjectMap,

    pub fn of(v: std.json.Value) ParseError!Json {
        const outer = try value(.object, v);
        const cfg = if (field(outer, "text_config")) |tc| try value(.object, tc) else outer;
        return .{ .outer = outer, .root = outer, .cfg = cfg };
    }
};

// Readers with the engine parser's semantics: a JSON null reads as an omitted key.

pub fn field(obj: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    const v = obj.get(key) orelse return null;
    return if (v == .null) null else v;
}

pub fn value(comptime tag: std.meta.Tag(std.json.Value), v: std.json.Value) ParseError!@TypeOf(@field(@as(std.json.Value, undefined), @tagName(tag))) {
    if (v != tag) return error.InvalidConfigField;
    return @field(v, @tagName(tag));
}

pub fn int(comptime T: type, v: std.json.Value) ParseError!T {
    return std.math.cast(T, try value(.integer, v)) orelse error.InvalidConfigField;
}

pub fn float(v: std.json.Value) ParseError!f32 {
    const f: f32 = switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |x| @floatCast(x),
        else => 0.0,
    };
    if (!std.math.isFinite(f)) return error.InvalidConfigField;
    return f;
}

/// An integer key's value when it is one; another type leaves the field as it was.
fn intIf(obj: std.json.ObjectMap, key: []const u8, out: *u32) ParseError!void {
    if (field(obj, key)) |v| if (v == .integer) {
        out.* = try int(u32, v);
    };
}

/// The keys every DeepSeek config shares and the V4 family arm; `v41` names the release whose refusals they are.
pub fn parseShared(c: *Config, j: Json, v41: bool) ParseError!void {
    const o = j.cfg;
    c.model_type = if (v41) "deepseek_v41" else "deepseek_v4";
    if (field(o, "vocab_size")) |v| c.vocab_size = try int(u32, v);
    if (field(o, "hidden_size")) |v| c.hidden_size = try int(u32, v);
    if (field(o, "intermediate_size")) |v| {
        c.intermediate_size = try int(u32, v);
        c.intermediate_size_declared = true;
    }
    if (field(o, "num_hidden_layers")) |v| c.num_hidden_layers = try int(u32, v);
    if (field(o, "num_attention_heads")) |v| c.num_attention_heads = try int(u32, v);
    if (field(o, "num_key_value_heads")) |v| c.num_key_value_heads = try int(u32, v);
    if (field(o, "head_dim")) |v| c.head_dim = try int(u32, v);
    if (field(o, "max_position_embeddings")) |v| c.max_position_embeddings = try int(u32, v);
    if (field(o, "rms_norm_eps")) |v| c.rms_norm_eps = try float(v);
    if (field(o, "rope_theta")) |v| c.rope_theta = try float(v);
    if (field(o, "query_pre_attn_scalar")) |v| c.query_pre_attn_scalar = try int(u32, v);
    try intIf(o, "num_experts", &c.num_experts);
    try intIf(o, "num_experts_per_tok", &c.num_experts_per_tok);
    try intIf(o, "top_k_experts", &c.num_experts_per_tok);
    try intIf(o, "moe_intermediate_size", &c.moe_intermediate_size);
    try intIf(o, "shared_expert_intermediate_size", &c.shared_expert_intermediate_size);
    if (field(o, "bos_token_id")) |v| if (v == .integer and v.integer >= 0) {
        c.bos_token_id = try int(u32, v);
    };
    if (field(o, "rope_parameters")) |rp| if (rp == .object) {
        if (field(rp.object, "rope_theta")) |v| c.rope_theta = try float(v);
    };
    if (o.get("sliding_window")) |v| {
        if (v == .null) c.has_sliding_window = false else {
            c.sliding_window = try int(u32, v);
            c.has_sliding_window = true;
        }
    }
    if (field(o, "rope_scaling")) |rs| if (rs == .object) {
        if (field(rs.object, "factor")) |v| c.rope_scaling_factor = try float(v);
    };
    inline for (.{ j.root, o }) |obj| if (field(obj, "tie_word_embeddings")) |v| if (v == .bool) {
        c.tie_word_embeddings = v.bool;
    };
    if (c.max_position_embeddings == 0) try intIf(j.root, "max_position_embeddings", &c.max_position_embeddings);
    try parseQuantization(c, j.outer);
    if (field(j.root, "eos_token_id")) |v| switch (v) {
        .integer => |i| c.addEos(std.math.cast(u32, i) orelse return error.InvalidConfigField),
        .array => |arr| for (arr.items) |item| if (item == .integer) c.addEos(try int(u32, item)),
        else => {},
    };
    if (field(j.root, "vision_config")) |vc| if (vc == .object) {
        c.has_vision = true;
        const x = vc.object;
        try intIf(x, "hidden_size", &c.vision_hidden_size);
        try intIf(x, "num_hidden_layers", &c.vision_num_layers);
        try intIf(x, "num_attention_heads", &c.vision_num_heads);
        try intIf(x, "head_dim", &c.vision_head_dim);
        try intIf(x, "global_head_dim", &c.vision_head_dim);
        try intIf(x, "intermediate_size", &c.vision_intermediate_size);
        try intIf(x, "patch_size", &c.vision_patch_size);
        if (field(x, "rope_parameters")) |rp| if (rp == .object) {
            if (field(rp.object, "rope_theta")) |v| c.vision_rope_theta = try float(v);
        };
    };
    try intIf(j.root, "image_token_id", &c.image_token_id);
    if (field(j.root, "image_token_index")) |v| if (v == .integer and c.image_token_id == 0) {
        c.image_token_id = try int(u32, v);
    };
    try intIf(j.root, "boi_token_id", &c.boi_token_id);
    try intIf(j.root, "eoi_token_id", &c.eoi_token_id);
    try parseFamilyArm(c, o, v41);
}

fn parseQuantization(c: *Config, outer: std.json.ObjectMap) ParseError!void {
    const q = try value(.object, field(outer, "quantization") orelse return);
    if (field(q, "bits")) |v| c.quant_bits = try int(u32, v);
    if (field(q, "group_size")) |v| c.quant_group_size = try int(u32, v);
    if (field(q, "mode")) |v| if (v == .string) {
        c.quant_mode = std.meta.stringToEnum(QuantMode, v.string) orelse return error.UnsupportedQuantMode;
    };
    // A per-weight override's mode is refused by name too (it would fail at the first matmul).
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

fn parseFamilyArm(c: *Config, o: std.json.ObjectMap, v41: bool) ParseError!void {
    const unsupported: ParseError = if (v41) error.UnsupportedDsv41Config else error.UnsupportedDsv4Config;
    try intIf(o, "n_routed_experts", &c.num_experts);
    try intIf(o, "num_hash_layers", &c.dsv4_hash_layers);
    if (field(o, "routed_scaling_factor")) |v| c.router_scaling_factor = try float(v);
    if (field(o, "norm_topk_prob")) |v| if (v == .bool) {
        c.moe_route_norm = v.bool;
    };
    try intIf(o, "q_lora_rank", &c.dsv4_q_lora_rank);
    try intIf(o, "o_lora_rank", &c.dsv4_o_lora_rank);
    try intIf(o, "o_groups", &c.dsv4_o_groups);
    try intIf(o, "qk_rope_head_dim", &c.dsv4_rope_head_dim);
    try intIf(o, "index_n_heads", &c.dsv4_index_n_heads);
    try intIf(o, "index_head_dim", &c.dsv4_index_head_dim);
    try intIf(o, "index_topk", &c.dsv4_index_topk);
    try intIf(o, "hc_mult", &c.dsv4_hc_mult);
    try intIf(o, "hc_sinkhorn_iters", &c.dsv4_hc_sinkhorn_iters);
    if (field(o, "hc_eps")) |v| c.dsv4_hc_eps = try float(v);
    if (field(o, "swiglu_limit")) |v| c.dsv4_swiglu_limit = try float(v);
    if (field(o, "compress_rope_theta")) |v| c.dsv4_compress_rope_theta = try float(v);
    try intIf(o, "num_nextn_predict_layers", &c.dsv4_mtp_layers);
    try intIf(o, "dspark_block_size", &c.dsv4_dspark_block_size);
    try intIf(o, "dspark_noise_token_id", &c.dsv4_dspark_noise_token_id);
    try intIf(o, "dspark_markov_rank", &c.dsv4_dspark_markov_rank);
    if (field(o, "dspark_target_layer_ids")) |v| if (v == .array) {
        const items = v.array.items;
        for (items, 0..) |item, i| {
            if (i >= c.dsv4_dspark_target_layers.len) break;
            if (item == .integer) c.dsv4_dspark_target_layers[i] = std.math.cast(u8, try int(u32, item)) orelse return error.InvalidConfigField;
        }
        c.dsv4_n_dspark_target_layers = @intCast(@min(items.len, c.dsv4_dspark_target_layers.len));
    };
    if (field(o, "compress_ratios")) |v| if (v == .array) {
        const items = v.array.items;
        for (items, 0..) |item, i| {
            if (i >= c.dsv4_compress_ratios.len) break;
            if (item == .integer) c.dsv4_compress_ratios[i] = std.math.cast(u8, try int(u32, item)) orelse return error.InvalidConfigField;
        }
        c.dsv4_n_compress_ratios = @intCast(@min(items.len, c.dsv4_compress_ratios.len));
    };
    // YaRN on compressed layers only, with no mscale (the reference applies none).
    if (field(o, "rope_scaling")) |rs| if (rs == .object) {
        c.rope_yarn = true;
        if (field(rs.object, "factor")) |x| c.yarn_factor = try float(x);
        if (field(rs.object, "beta_fast")) |x| c.yarn_beta_fast = try float(x);
        if (field(rs.object, "beta_slow")) |x| c.yarn_beta_slow = try float(x);
        try intIf(rs.object, "original_max_position_embeddings", &c.yarn_orig_max_pos);
    };
    // The forward serves exactly sqrt(softplus) scoring with selection-only bias, one shared expert and one KV latent.
    if (field(o, "scoring_func")) |v| if (v == .string and !std.mem.eql(u8, v.string, "sqrtsoftplus")) return unsupported;
    if (field(o, "topk_method")) |v| if (v == .string and !std.mem.eql(u8, v.string, "noaux_tc")) return unsupported;
    if (field(o, "n_shared_experts")) |v| if (v == .integer and v.integer != 1) return unsupported;
    if (c.num_key_value_heads != 1) return unsupported;
    // The withdrawn preview declares MTP layers with no DSpark descriptor.
    if (c.dsv4_mtp_layers > 0 and c.dsv4_dspark_block_size == 0) return error.Dsv4Preview;
}

/// The pack's `generation_config.json` read as the host reads it (`sdk/config/generation`): ranged sampling values,
/// extra eos ids, the template's thinking default.
pub fn applyGenerationConfig(c: *Config, bytes: []const u8) void {
    var sh: abi.Shell = .{ .gen = c.gen, .eos_count = c.num_eos_tokens };
    @memcpy(sh.eos[0..c.num_eos_tokens], c.eos_token_ids[0..c.num_eos_tokens]);
    generation.applyJson(bytes, &sh);
    c.gen = sh.gen;
    c.num_eos_tokens = @min(sh.eos_count, c.eos_token_ids.len);
    @memcpy(c.eos_token_ids[0..c.num_eos_tokens], sh.eos[0..c.num_eos_tokens]);
}

const gpa = std.heap.c_allocator;

/// Parses config.json (and, for a load, `generation_config.json`) into `out`; `v41` selects the release.
pub fn parseInto(dir: *const abi.PackDir, config_json: abi.Str, stub: bool, v41: bool, stream_pack: bool, out: *abi.Parsed, extend: ?*const fn (*Config, Json) ParseError!void) abi.Verdict {
    const exl3_bank = v41 and dir.has("experts.bin");
    const tree = std.json.parseFromSlice(std.json.Value, gpa, config_json.slice(), .{}) catch
        return abi.Verdict.refuse(.failed, "deepseek: config.json does not parse");
    defer tree.deinit();
    const cfg = gpa.create(Config) catch return abi.Verdict.refuse(.failed, "deepseek: out of memory");
    cfg.* = .{};
    const j = Json.of(tree.value) catch |e| return refuseFree(cfg, e);
    parseShared(cfg, j, v41) catch |e| return refuseFree(cfg, e);
    cfg.dsv41_streamed = exl3_bank or (v41 and stream_pack);
    cfg.dsv41_exl3_bank = exl3_bank;
    if (extend) |f| f(cfg, j) catch |e| return refuseFree(cfg, e);
    if (!stub) {
        var buf: [64 * 1024]u8 = undefined;
        var why: abi.Verdict = .{};
        if (dir.readFile("generation_config.json", &buf, &why)) |bytes| applyGenerationConfig(cfg, bytes);
        // V4 releases ship the wild 1.0/1.0 signature their own generate.py ignores (its default is 0.6).
        if (!v41 and cfg.gen.temperature == 1.0 and cfg.gen.top_p == 1.0) cfg.gen.temperature = 0.6;
    }
    out.* = facts(cfg);
    return .ok;
}

fn refuseFree(cfg: *Config, e: ParseError) abi.Verdict {
    gpa.destroy(cfg);
    return switch (e) {
        error.InvalidConfigField => abi.Verdict.refuse(.failed, "deepseek: a config.json field is out of range or of the wrong type"),
        error.UnsupportedQuantMode => abi.Verdict.refuse(.unsupported, "deepseek: unsupported quantization mode"),
        error.UnsupportedQuantBits => abi.Verdict.refuse(.unsupported, "deepseek: unsupported affine width (2, 3, 4, 5, 6, 8)"),
        error.UnsupportedDsv4Config => abi.Verdict.refuse(.unsupported, "deepseek_v4: a config the forward does not serve (sqrtsoftplus, noaux_tc, one shared expert, one KV latent, a DSpark release)"),
        error.UnsupportedDsv41Config => abi.Verdict.refuse(.unsupported, "deepseek_v41: a config the forward does not serve"),
        error.Dsv4Preview => abi.Verdict.refuse(.unsupported, "deepseek: the superseded preview checkpoint (MTP layers, no dspark_* config); use DeepSeek-V4-Flash-0731 or later"),
    };
}

/// The host's facts for a parsed config: module-owned serial decode, no prefix cache, the family's chat facts.
fn facts(c: *const Config) abi.Parsed {
    const v41 = c.isV41();
    var p: abi.Parsed = .{
        .caps = .{
            .owns_decode_state = true,
            .prefix_cache = false,
            .moe = c.num_experts > 0,
            .images = c.dsv41_vision,
            // V4.1 re-reads a Thunderbolt pair's plan at load (`deepseek_v41/plugin.zig`).
            .cluster_plan = v41,
            // A streamed bank's prompt runs in the module's own wide calls (each layer's experts read once a call).
            .whole_prompt_prefill = c.dsv41_streamed,
            .streamed_weights = c.dsv41_streamed,
            // V4.1 keeps a state per slot, and plain decode tokens of several run as one batch.
            .slot_state = v41 and !c.dsv41_streamed,
            .batches_decode = v41 and !c.dsv41_streamed,
        },
        .chat = .{
            .thinking_default = if (v41) .on else .off,
            .effort_vocab = if (v41) .dsv41 else .dsv4,
            .template_family = abi.Str.of(c.model_type),
        },
        .spec = .{
            .native_stages = c.dsparkStages(),
            .native_block = c.dsv4_dspark_block_size,
            .policy = .{ .drafter_ok = false },
        },
        .placeholder_ids = .{ .image = if (c.dsv41_vision) c.image_token_id else 0 },
        .weights = .{ .prefix = .{} },
        .private = @constCast(c),
    };
    p.shell = .{
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
        .quant = .{ .bits = @intCast(@min(c.quant_bits, 255)), .group_size = c.quant_group_size, .mode = switch (c.quant_mode) {
            .affine => if (c.quant_bits == 0) .none else .affine,
            .nvfp4 => .nvfp4,
            .mxfp4 => .mxfp4,
            .mxfp8 => .mxfp8,
            .gguf => .gguf,
        } },
        .gen = c.gen,
        .bos = if (c.bos_token_id) |b| b else -1,
        .act_dtype = .bf16,
        .model_type = abi.Str.of(c.model_type),
    };
    @memcpy(p.shell.eos[0..c.num_eos_tokens], c.eos_token_ids[0..c.num_eos_tokens]);
    p.shell.eos_count = c.num_eos_tokens;
    return p;
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    _ = span;
    return parseInto(dir, config_json, stub, false, false, out, null);
}

pub fn freeParsed(p: *abi.Parsed) callconv(.c) void {
    if (p.private) |c| gpa.destroy(@as(*Config, @ptrCast(@alignCast(c))));
    p.private = null;
}

pub fn of(p: *const abi.Parsed) *const Config {
    return @ptrCast(@alignCast(p.private.?));
}

// ── tests ──

const testing = std.testing;
const fixture = sdk.fixture;

/// The release's config (0731 and later): three DSpark stages past 43 trunk layers, a per-weight quant override.
pub const release_config =
    \\{"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4", "bos_token_id": 0, "eos_token_id": 1,
    \\ "head_dim": 512, "hidden_act": "silu", "hidden_size": 4096, "index_head_dim": 128, "index_n_heads": 64,
    \\ "index_topk": 512, "max_position_embeddings": 1048576, "moe_intermediate_size": 2048, "n_routed_experts": 256,
    \\ "n_shared_experts": 1, "norm_topk_prob": true, "num_attention_heads": 64, "num_experts_per_tok": 6,
    \\ "num_hidden_layers": 43, "num_hash_layers": 3, "num_key_value_heads": 1, "num_nextn_predict_layers": 3,
    \\ "dspark_block_size": 5, "dspark_noise_token_id": 128799, "dspark_target_layer_ids": [40, 41, 42],
    \\ "dspark_markov_rank": 256, "o_groups": 8, "o_lora_rank": 1024, "q_lora_rank": 1024, "qk_rope_head_dim": 64,
    \\ "hc_eps": 1e-06, "hc_mult": 4, "hc_sinkhorn_iters": 20, "rms_norm_eps": 1e-06,
    \\ "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 16, "original_max_position_embeddings": 65536, "type": "yarn"},
    \\ "rope_theta": 10000, "routed_scaling_factor": 1.5, "scoring_func": "sqrtsoftplus", "sliding_window": 128,
    \\ "swiglu_limit": 10.0, "tie_word_embeddings": false, "topk_method": "noaux_tc", "torch_dtype": "bfloat16",
    \\ "vocab_size": 129280, "compress_rope_theta": 160000,
    \\ "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0],
    \\ "quantization": {"group_size": 64, "bits": 8, "mode": "affine",
    \\   "layers.0.ffn.experts.w1": {"group_size": 64, "bits": 2, "mode": "affine"}}}
;

fn parseFiles(files: []const fixture.File, stub: bool, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = files };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(files[0].bytes), .{}, null, stub, out);
}

test "deepseek_v4 config: the release parses into the family config and the host's facts" {
    var p: abi.Parsed = .{};
    try testing.expect(parseFiles(&.{.{ .name = "config.json", .bytes = release_config }}, true, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expectEqual(@as(u32, 3), c.dsparkStages());
    try testing.expectEqual(@as(u8, 128), c.dsv4_compress_ratios[3]);
    try testing.expectEqual(@as(u64, 1 << 20), p.shell.context_cap);
    try testing.expect(p.caps.owns_decode_state and !p.caps.prefix_cache and !p.caps.batches_decode and p.caps.moe);
    try testing.expectEqual(abi.EffortVocab.dsv4, p.chat.effort_vocab);
    try testing.expectEqualStrings("deepseek_v4", p.chat.template_family.slice());
    try testing.expectEqual(@as(u32, 5), p.spec.native_block);
    try testing.expectEqual(@as(i64, 0), p.shell.bos);
    try testing.expectEqual(@as(u32, 1), p.shell.eos_count);
}

test "deepseek_v4 config: the wild 1.0/1.0 generation signature resolves to the reference's 0.6; a tuned one stays" {
    inline for (.{ .{ "{\"temperature\":1.0,\"top_p\":1.0}", 0.6 }, .{ "{\"temperature\":0.9,\"top_p\":1.0}", 0.9 } }) |case| {
        var p: abi.Parsed = .{};
        try testing.expect(parseFiles(&.{ .{ .name = "config.json", .bytes = release_config }, .{ .name = "generation_config.json", .bytes = case[0] } }, false, &p).isOk());
        defer freeParsed(&p);
        try testing.expectEqual(@as(f32, case[1]), p.shell.gen.temperature);
    }
}

test "deepseek_v4 config: mlx-community's 0731 pack parses; its withdrawn preview pack is refused by name" {
    var p: abi.Parsed = .{};
    try testing.expect(parseFiles(&.{.{ .name = "config.json", .bytes = @embedFile("fixtures/real/mlx-community_DeepSeek-V4-Flash-0731-2.4bit-mixed.json") }}, true, &p).isOk());
    freeParsed(&p);
    const v = parseFiles(&.{.{ .name = "config.json", .bytes = @embedFile("fixtures/real/mlx-community_DeepSeek-V4-Flash-4bit.json") }}, true, &p);
    try testing.expectEqualStrings("deepseek: the superseded preview checkpoint (MTP layers, no dspark_* config); use DeepSeek-V4-Flash-0731 or later", v.why.slice());
}

test "deepseek_v4 config: what the forward does not serve is refused by name, leaving nothing to free" {
    inline for (.{
        "{\"model_type\":\"deepseek_v4\",\"num_key_value_heads\":1,\"scoring_func\":\"softmax\"}",
        "{\"model_type\":\"deepseek_v4\",\"num_key_value_heads\":2}",
        "{\"model_type\":\"deepseek_v4\",\"num_key_value_heads\":1,\"num_nextn_predict_layers\":1}",
        "{\"model_type\":\"deepseek_v4\",\"num_key_value_heads\":1,\"compress_ratios\":[256]}",
        "{\"model_type\":\"deepseek_v4\",\"num_key_value_heads\":1,\"quantization\":{\"bits\":7}}",
        "{\"model_type\":\"deepseek_v4\",\"num_key_value_heads\":1,\"quantization\":{\"x\":{\"mode\":\"iq2_xxs\"}}}",
    }) |json| {
        var p: abi.Parsed = .{};
        const v = parseFiles(&.{.{ .name = "config.json", .bytes = json }}, true, &p);
        try testing.expect(!v.isOk() and v.why.len > 0 and p.private == null);
    }
}
