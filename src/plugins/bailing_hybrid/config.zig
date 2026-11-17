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
    // Ling-3.1's clamps after the SiLU, one entry a trunk layer (0 = none).
    try limitList(cfg, "expert_swiglu_limit_list", &c.expert_limit_out);
    try limitList(cfg, "share_expert_swiglu_limit_list", &c.shared_limit_out);
    try intIf(cfg, "num_nextn_predict_layers", &c.nextn_layers);

    // A Ling streaming pack (the EXL3 repack beside its `experts.bin`) keeps the release's config.json, so it names no
    // `expert_streaming`: its EXL3 `quantization_config` marks it. The family parse refuses it without `experts.bin`, and
    // the outside quant plugin that claims the block runs its projections and streams the bank.
    if (field(o.root, "quantization_config")) |qc| if (qc == .object) if (qc.object.get("quant_method")) |m|
        if (m == .string and std.mem.eql(u8, m.string, "exl3")) {
            c.streamed_bank = true;
        };
}

/// A per-layer SwiGLU limit list (`key`) into `out`: whole numbers 0 to 255, at most `out.len` layers.
fn limitList(cfg: std.json.ObjectMap, key: []const u8, out: *[128]u8) ParseError!void {
    const v = field(cfg, key) orelse return;
    if (v != .array) return error.InvalidConfigField;
    if (v.array.items.len > out.len) return error.Unsupported;
    for (v.array.items, 0..) |item, i| {
        const n = try u32v(item);
        out[i] = std.math.cast(u8, n) orelse return error.Unsupported;
    }
}

/// Ling 3.0's `Parsed` facts beside the shell.
fn lingFacts(c: *const Config, p: *abi.Parsed) void {
    p.caps = .{ .moe = c.num_experts > 0, .dflash_capture = true, .experts_stream = c.num_experts > 0 };
    if (c.streamed_bank) {
        family.bankFacts(p);
        // The outside plugin's head computes a prompt row as the whole prompt's call does (glm5_next/host.zig `finish`).
        p.caps.prompt_last_row = true;
    }
    p.chat.thinking_default = .on;
    // A streaming pack's NextN layer drafts one token a round over the trunk's final-normed hidden (`head.zig`).
    if (c.streamed_bank and c.nextn_layers > 0) p.spec = .{ .mtp_default_depth = 1, .mtp_layout = .in_checkpoint, .head_reads_final_norm = true };
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

test "bailing_hybrid config: Ling-3.1-flash's EXL3 repack is a streaming pack by its quantization_config, beside experts.bin" {
    const json = @embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.1-flash-exl3-2.60bpw.json");
    // The repack keeps the release's config.json: no `expert_streaming`, an EXL3 `quantization_config`.
    try testing.expect(std.mem.indexOf(u8, json, "expert_streaming") == null);
    const mem: sdk.fixture.MemDir = .{ .files = &.{ .{ .name = "config.json", .bytes = json }, .{ .name = "experts.bin", .bytes = "" } } };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(parse(&dir, abi.Str.of(json), .{}, null, true, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expect(c.streamed_bank and p.caps.streamed_weights and !p.caps.batches_decode and !p.caps.images);
    try testing.expectEqual(@as(u32, 72), p.shell.num_hidden_layers);
    // Without its bank the same config is refused by name.
    const bare: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const bare_dir = bare.packDir();
    var q: abi.Parsed = .{};
    const v = parse(&bare_dir, abi.Str.of(json), .{}, null, true, &q);
    try testing.expect(v.code == .unsupported and q.private == null);
}

test "bailing_hybrid config: Ling-3.1-flash's SwiGLU limit lists, routed (layers 65-71) and shared (64-71), clamp after the SiLU at 10" {
    const json = @embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.1-flash-exl3-2.60bpw.json");
    const mem: sdk.fixture.MemDir = .{ .files = &.{ .{ .name = "config.json", .bytes = json }, .{ .name = "experts.bin", .bytes = "" } } };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(parse(&dir, abi.Str.of(json), .{}, null, true, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    for (0..72) |i| {
        try testing.expectEqual(@as(u8, if (i >= 65) 10 else 0), c.expert_limit_out[i]);
        try testing.expectEqual(@as(u8, if (i >= 64) 10 else 0), c.shared_limit_out[i]);
    }
    for (c.expert_limit_out[72..]) |l| try testing.expectEqual(@as(u8, 0), l);
    // Ling 3.0 lists none.
    var q: abi.Parsed = .{};
    try testing.expect(parseJson(@embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.0-tiny-4bit.json"), &q).isOk());
    defer freeParsed(&q);
    for (of(&q).shared_limit_out) |l| try testing.expectEqual(@as(u8, 0), l);
}

test "bailing_hybrid config: a limit list past 128 layers, past 255 or not a list is refused" {
    inline for (.{
        "{\"model_type\":\"bailing_hybrid\",\"kv_lora_rank\":64,\"expert_swiglu_limit_list\":[256]}",
        "{\"model_type\":\"bailing_hybrid\",\"kv_lora_rank\":64,\"share_expert_swiglu_limit_list\":7}",
    }) |json| {
        var p: abi.Parsed = .{};
        const v = parseJson(json, &p);
        try testing.expect(!v.isOk() and p.private == null);
    }
    // 129 layers
    var long: std.ArrayList(u8) = .empty;
    defer long.deinit(testing.allocator);
    try long.appendSlice(testing.allocator, "{\"model_type\":\"bailing_hybrid\",\"kv_lora_rank\":64,\"share_expert_swiglu_limit_list\":[0");
    for (0..128) |_| try long.appendSlice(testing.allocator, ",0");
    try long.appendSlice(testing.allocator, "]}");
    var p: abi.Parsed = .{};
    const v = parseJson(long.items, &p);
    try testing.expect(!v.isOk() and p.private == null);
}

test "bailing_hybrid config: a streaming pack's NextN layer declares the MTP head (one draft a round over the final-normed hidden); an MLX pack none" {
    const json = @embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.1-flash-exl3-2.60bpw.json");
    const mem: sdk.fixture.MemDir = .{ .files = &.{ .{ .name = "config.json", .bytes = json }, .{ .name = "experts.bin", .bytes = "" } } };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(parse(&dir, abi.Str.of(json), .{}, null, true, &p).isOk());
    defer freeParsed(&p);
    try testing.expectEqual(@as(u32, 1), of(&p).nextn_layers);
    try testing.expect(p.spec.mtp_layout == .in_checkpoint and p.spec.mtp_default_depth == 1 and p.spec.head_reads_final_norm);
    // MLX packs keep their facts: no head, a NextN layer or not.
    inline for (.{ "bailing_hybrid-ling-3.0-tiny-4bit.json", "bailing_hybrid-ling-3.0-flash-oq4e-mtp.json" }) |name| {
        var q: abi.Parsed = .{};
        try testing.expect(parseJson(@embedFile("../../fixtures/model-configs/" ++ name), &q).isOk());
        defer freeParsed(&q);
        try testing.expect(q.spec.mtp_layout == .none and q.spec.mtp_default_depth == 0 and !of(&q).streamed_bank);
    }
}
