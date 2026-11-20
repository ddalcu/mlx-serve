//! GLM-5.3-Flash's claims and parse over the KDA-hybrid family config (`sdk/bases/kda_hybrid/config.zig`, which
//! `bailing_hybrid` shares). MLX-free (discovery runs it). A shape this plugin does not serve exactly as the engine did
//! is refused `unsupported`.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const family = @import("../../sdk/bases/kda_hybrid/config.zig");

pub const Config = family.Config;
pub const of = family.of;
pub const freeParsed = family.freeParsed;
const Objects = family.Objects;
const ParseError = family.ParseError;
const bankFacts = family.bankFacts;
const parseWith = family.parseWith;
const field = family.field;
const value = family.value;
const u32v = family.u32v;
const f32v = family.f32v;
const intIf = family.intIf;

pub const model_types = [_][]const u8{ "glm5_next", "glm5_next_text" };

pub const claims = sdk.claimsByType(&model_types);

pub const plugin = sdk.Plugin{
    .name = "glm5_next",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"ddalcu/GLM-5.3-Flash-MLX-Serve-iQ-MLX-5.1bpw"},
};

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
        try readQvGeometry(c, vc);
        if (field(vc, "projection_intermediate_size")) |v| c.glmv_proj_intermediate = try u32v(v);
        if (field(vc, "rms_norm_eps")) |v| c.glmv_eps = try f32v(v);
        if (field(vc, "swiglu_limit")) |v| c.glmv_swiglu_limit = try f32v(v);
        if (c.qv_heads == 0 or c.qv_hidden % c.qv_heads != 0 or c.qv_head_dim % 4 != 0 or
            c.qv_patch == 0 or c.qv_merge == 0 or c.qv_temporal_patch == 0 or c.qv_out_hidden != c.hidden_size)
            return error.InvalidConfigField;
    }
}

/// GLM-5.3-Flash's `Parsed` facts beside the shell.
fn glmFacts(c: *const Config, p: *abi.Parsed) void {
    p.caps = .{
        .batches_decode = true,
        // Each slot's DSA cache is advanced by its own call: a batched tick never pads or stacks KV.
        .batch_pads_kv = false,
        .moe = c.num_experts > 0,
        .experts_stream = c.num_experts > 0,
        .dflash_capture = true,
        .dflash_policy = true,
        .images = c.advertises_images,
    };
    if (c.streamed_bank) bankFacts(p);
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

test "glm5_next config: a streaming pack beside its bank parses streamed, serially decoded, without images" {
    const base = @embedFile("../../fixtures/model-configs/glm5_next-mlx-community-4bit.json");
    const bank = "{\"expert_streaming\":{\"family\":\"turboexl3\",\"manifest\":\"expert-manifest-v2.json\"},";
    const json = try std.mem.concat(testing.allocator, u8, &.{ bank, std.mem.trimStart(u8, base, " \n{") });
    defer testing.allocator.free(json);
    const mem: sdk.fixture.MemDir = .{ .files = &.{ .{ .name = "config.json", .bytes = json }, .{ .name = "experts.bin", .bytes = "" } } };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(parse(&dir, abi.Str.of(json), .{}, null, true, &p).isOk());
    defer freeParsed(&p);
    try testing.expect(of(&p).streamed_bank and p.caps.streamed_weights and !p.caps.batches_decode and !p.caps.images);
    // Ling's facts stream the bank too (bailing_hybrid shares this trunk): the same block parses streamed.
    const ling_cfg = @import("../bailing_hybrid/config.zig");
    const ling_base = @embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.0-tiny-4bit.json");
    const ling = try std.mem.concat(testing.allocator, u8, &.{ bank, std.mem.trimStart(u8, ling_base, " \n{") });
    defer testing.allocator.free(ling);
    var q: abi.Parsed = .{};
    try testing.expect(ling_cfg.parse(&dir, abi.Str.of(ling), .{}, null, true, &q).isOk());
    defer ling_cfg.freeParsed(&q);
    try testing.expect(of(&q).streamed_bank and q.caps.streamed_weights and !q.caps.batches_decode and !q.caps.images);
}

test "glm5_next config: a streamed pack, a RoPE'd sparse attention and broken JSON are refused by name" {
    inline for (.{
        "{\"model_type\":\"glm_moe_dsa\",\"num_hidden_layers\":2}",
        "{\"model_type\":\"glm5_next\",\"expert_streaming\":{},\"num_hidden_layers\":2}",
        "{\"model_type\":\"glm5_next\",\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mtplx-expert-manifest-v2\",\"manifest\":\"expert-manifest-v2.json\",\"runtime_manifest\":\"expert-manifest.json\",\"quant\":\"exl3-k3-mul1\"},\"num_hidden_layers\":2}",
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
