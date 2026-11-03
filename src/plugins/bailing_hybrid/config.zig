//! Ling 3.0's (`bailing_hybrid`, BailingMoeV3) claim and parse onto the KDA-hybrid family config GLM-5.3-Flash
//! defines (`glm5_next/config.zig`). MLX-free. Three KDA layers per Multi-head Latent Attention layer, grouped
//! sigmoid routing with an expert bias, one ungated shared expert.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const family = @import("../glm5_next/config.zig");

const Config = family.Config;
const Objects = family.Objects;
const ParseError = family.ParseError;
const field = family.field;
const u32v = family.u32v;
const f32v = family.f32v;
const intIf = family.intIf;

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "bailing_hybrid",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"rapid-mlx/Ling-3.0-tiny-MLX-4bit"},
};

pub const model_types = [_][]const u8{"bailing_hybrid"};

pub const claims = sdk.claimsByType(&model_types);

/// The family arm of the engine's parse for `bailing_hybrid`.
fn readLing(c: *Config, o: Objects) ParseError!void {
    const cfg = o.cfg;
    c.model_type = "bailing_hybrid";
    c.weight_prefix = "model";
    c.has_sliding_window = false;
    c.rope_scaling_factor = 1.0;
    c.rope_local_base_freq = c.rope_theta;
    try intIf(cfg, "layer_group_size", &c.full_attention_interval);

    c.linear_num_key_heads = c.num_attention_heads;
    c.linear_num_value_heads = c.num_attention_heads;
    c.linear_key_head_dim = c.head_dim;
    c.linear_value_head_dim = c.head_dim;
    try intIf(cfg, "short_conv_kernel_size", &c.linear_conv_kernel_dim);
    if (field(cfg, "num_kv_heads_for_linear_attn")) |v|
        if (v == .integer and v.integer != 0 and v.integer != @as(i64, c.num_attention_heads)) return error.Unsupported;
    c.kda_vector_gate = true;
    c.kda_sigmoid_out_gate = true;
    if (field(cfg, "kda_lower_bound")) |v| {
        const lb = try f32v(v);
        if (lb >= 0.0) return error.Unsupported;
        c.kda_gate_lower_bound = lb;
    }
    if (field(cfg, "no_kda_lora")) |v| if (v == .bool and !v.bool) return error.Unsupported;
    if (field(cfg, "use_kda_lora")) |v| if (v == .bool and v.bool) return error.Unsupported;

    try intIf(cfg, "q_lora_rank", &c.mla_q_lora_rank);
    try intIf(cfg, "kv_lora_rank", &c.mla_kv_lora_rank);
    try intIf(cfg, "qk_nope_head_dim", &c.mla_qk_nope_head_dim);
    try intIf(cfg, "qk_rope_head_dim", &c.mla_qk_rope_head_dim);
    try intIf(cfg, "v_head_dim", &c.mla_v_head_dim);
    if (c.mla_v_head_dim == 0) c.mla_v_head_dim = c.head_dim;
    if (c.mla_kv_lora_rank == 0) return error.Unsupported;
    const qk_head_dim = std.math.add(u32, c.mla_qk_nope_head_dim, c.mla_qk_rope_head_dim) catch return error.InvalidConfigField;
    if (field(cfg, "qk_head_dim")) |v| if (v == .integer and try u32v(v) != qk_head_dim) return error.Unsupported;
    c.query_pre_attn_scalar = qk_head_dim;
    if (field(cfg, "gated_attention_proj_granularity_type")) |v| if (v == .string) {
        if (!std.mem.eql(u8, v.string, "head_wise")) return error.Unsupported;
        c.mla_head_gate = true;
    };
    if (field(cfg, "rope_interleave")) |v| if (v == .bool) {
        c.rope_interleaved_pairs = v.bool;
    };
    if (field(cfg, "use_mla_nope")) |v| if (v == .bool and v.bool) return error.Unsupported;
    c.partial_rotary_factor = 1.0;
    inline for (.{ "value_norm", "up_proj_norm", "use_nGPT" }) |key|
        if (field(cfg, key)) |v| if (v == .bool and v.bool) return error.Unsupported;
    if (field(cfg, "linear_silu")) |v| if (v == .bool and !v.bool) return error.Unsupported;

    c.moe_sigmoid_router = true;
    try intIf(cfg, "first_k_dense_replace", &c.first_k_dense_replace);
    try intIf(cfg, "n_group", &c.moe_n_group);
    try intIf(cfg, "topk_group", &c.moe_topk_group);
    if (field(cfg, "norm_topk_prob")) |v| if (v == .bool) {
        c.moe_route_norm = v.bool;
    };
    if (field(cfg, "routed_scaling_factor")) |v| c.router_scaling_factor = try f32v(v);
    if (c.shared_expert_intermediate_size == 0) {
        var width: u32 = c.moe_intermediate_size;
        try intIf(cfg, "moe_shared_expert_intermediate_size", &width);
        var n_shared: u32 = 0;
        try intIf(cfg, "num_shared_experts", &n_shared);
        c.shared_expert_intermediate_size = std.math.mul(u32, width, n_shared) catch return error.InvalidConfigField;
    }
    if (field(cfg, "score_function")) |v| if (v == .string and !std.mem.eql(u8, v.string, "sigmoid")) return error.Unsupported;
}

/// Ling 3.0's `Parsed` facts beside the shell.
fn lingFacts(c: *const Config, p: *abi.Parsed) void {
    p.caps = .{ .moe = c.num_experts > 0, .dflash_capture = true };
    p.chat.thinking_default = .on;
    p.cache.ssm_state = .{ .entries = c.num_hidden_layers, .dtype = .bf16 };
    p.weights.prefix = abi.Str.of(c.weight_prefix);
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    return family.parseWith("bailing_hybrid", &model_types, readLing, lingFacts, dir, config_json, span, stub, out);
}

pub const freeParsed = family.freeParsed;
pub const of = family.of;

// ── tests ──

const testing = std.testing;

fn parseJson(json: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(json), .{}, null, true, out);
}

test "bailing_hybrid config: Ling 3.0 tiny parses into the shell and the family config" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(@embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.0-tiny-4bit.json"), &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expectEqual(@as(u32, 24), p.shell.num_hidden_layers);
    try testing.expectEqual(@as(u32, 6), c.attnLayerCount());
    try testing.expectEqual(@as(u32, 192), c.query_pre_attn_scalar);
    try testing.expect(c.mla_head_gate and c.rope_interleaved_pairs and !p.caps.batches_decode);
    try testing.expectEqualSlices(u32, &.{156895}, p.shell.eos[0..p.shell.eos_count]);
}

test "bailing_hybrid config: a NoPE MLA, low-rank KDA gates and a non-sigmoid router are refused unsupported" {
    inline for (.{
        "{\"model_type\":\"bailing_hybrid\",\"kv_lora_rank\":64,\"use_mla_nope\":true}",
        "{\"model_type\":\"bailing_hybrid\",\"kv_lora_rank\":64,\"use_kda_lora\":true}",
        "{\"model_type\":\"bailing_hybrid\",\"kv_lora_rank\":64,\"score_function\":\"softmax\"}",
        "{\"model_type\":\"bailing_hybrid\"}",
    }) |json| {
        var p: abi.Parsed = .{};
        const v = parseJson(json, &p);
        try testing.expect(v.code == .unsupported and p.private == null);
    }
}
