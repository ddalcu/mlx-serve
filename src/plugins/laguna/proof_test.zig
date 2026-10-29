//! The laguna plugin against the legacy layered forward on tiny random Laguna packs: per-layer heads, YaRN full layers,
//! sliding layers, the softplus gate, a dense layer 0 then routed experts (selection bias, one layer without it,
//! an ungated shared expert). 4-bit, and bf16 with decode-attn-quant off and on (its decode side copies).

const std = @import("std");
const plugin = @import("plugin.zig");
const both = @import("../../engines/mlx/plugin/both.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const transformer = @import("../../engines/mlx/transformer.zig");
const decode_attn_quant = @import("../../sdk/blocks/attention/decode_attn_quant.zig");

const H = 128;
const HD = 128;
const E = 8;
const MI = 64;
const INTER = 256;
const VOCAB = 256;
const heads = [_]c_int{ 2, 4, 4, 2 };

fn configJson(comptime quantized: bool) []const u8 {
    return "{\"model_type\":\"laguna\",\"hidden_size\":128,\"intermediate_size\":256,\"num_hidden_layers\":4," ++
        "\"num_attention_heads\":4,\"num_key_value_heads\":1,\"head_dim\":128,\"rms_norm_eps\":1e-6,\"vocab_size\":256," ++
        "\"max_position_embeddings\":4096,\"tie_word_embeddings\":false,\"gating\":true,\"sliding_window\":4," ++
        "\"num_experts\":8,\"num_experts_per_tok\":2,\"moe_intermediate_size\":64,\"shared_expert_intermediate_size\":64," ++
        "\"moe_routed_scaling_factor\":2.5,\"num_attention_heads_per_layer\":[2,4,4,2]," ++
        "\"layer_types\":[\"full_attention\",\"sliding_attention\",\"sliding_attention\",\"full_attention\"]," ++
        "\"rope_parameters\":{\"full_attention\":{\"rope_theta\":500000.0,\"rope_type\":\"yarn\",\"factor\":32.0," ++
        "\"original_max_position_embeddings\":8192,\"beta_slow\":1.0,\"beta_fast\":32.0,\"partial_rotary_factor\":0.5}," ++
        "\"sliding_attention\":{\"rope_type\":\"default\",\"rope_theta\":10000.0,\"partial_rotary_factor\":1.0}}" ++
        (if (quantized) ",\"quantization\":{\"group_size\":64,\"bits\":4}" else "") ++ "}";
}

fn at(buf: []u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ layer, name });
}

fn writePack(p: *fixture.Pack, q: bool) !void {
    return writePackAt(p, q, H, VOCAB);
}

/// The pack at hidden width `h` and vocabulary `vocab` (the DFlash proof's tiny assistant reads 64 and 128).
fn writePackAt(p: *fixture.Pack, q: bool, comptime h: c_int, comptime vocab: c_int) !void {
    var buf: [160]u8 = undefined;
    try p.weight("model.embed_tokens", vocab, h, q);
    try p.norm("model.norm.weight", h);
    try p.weight("lm_head", vocab, h, q);
    for (heads, 0..) |nh, i| {
        try p.norm(try at(&buf, i, "input_layernorm.weight"), h);
        try p.norm(try at(&buf, i, "post_attention_layernorm.weight"), h);
        try p.weight(try at(&buf, i, "self_attn.q_proj"), nh * HD, h, q);
        try p.weight(try at(&buf, i, "self_attn.k_proj"), HD, h, q);
        try p.weight(try at(&buf, i, "self_attn.v_proj"), HD, h, q);
        try p.weight(try at(&buf, i, "self_attn.o_proj"), h, nh * HD, q);
        try p.norm(try at(&buf, i, "self_attn.q_norm.weight"), HD);
        try p.norm(try at(&buf, i, "self_attn.k_norm.weight"), HD);
        try p.weight(try at(&buf, i, "self_attn.g_proj"), nh, h, q);
        if (i == 0) {
            try p.weight(try at(&buf, i, "mlp.gate_proj"), INTER, h, q);
            try p.weight(try at(&buf, i, "mlp.up_proj"), INTER, h, q);
            try p.weight(try at(&buf, i, "mlp.down_proj"), h, INTER, q);
            continue;
        }
        try p.weight(try at(&buf, i, "mlp.gate"), E, h, false);
        if (i != 3) try p.denseAs(try at(&buf, i, "mlp.gate.e_score_correction_bias"), &.{E}, 0.1, 0, .float32);
        try p.tensor(try at(&buf, i, "mlp.switch_mlp.gate_proj"), &.{ E, MI, h }, q);
        try p.tensor(try at(&buf, i, "mlp.switch_mlp.up_proj"), &.{ E, MI, h }, q);
        try p.tensor(try at(&buf, i, "mlp.switch_mlp.down_proj"), &.{ E, h, MI }, q);
        try p.weight(try at(&buf, i, "mlp.shared_expert.gate_proj"), MI, h, q);
        try p.weight(try at(&buf, i, "mlp.shared_expert.up_proj"), MI, h, q);
        try p.weight(try at(&buf, i, "mlp.shared_expert.down_proj"), h, MI, q);
    }
}

fn case(comptime quantized: bool, comptime seed: u64) !void {
    return caseOn(quantized, seed, false);
}


fn openCase(p: *fixture.Pack, json: []const u8, comptime legacy_config: bool, comptime seed: u64) !both.Both {
    const name = std.fmt.comptimePrint("fixtures/{x}{s}.tape", .{ seed, if (legacy_config) "-lc" else "" });
    return both.Both.replay(&plugin.table, p, json, legacy_config, @embedFile(name));
}

fn caseOn(comptime quantized: bool, comptime seed: u64, comptime legacy_config: bool) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writePack(&p, quantized);
    var b = try openCase(&p, configJson(quantized), legacy_config, seed);
    defer b.close();
    try b.expectSteps(&both.long_steps);
    try b.expectGreedy();
    try b.expectDone();
}

// ── tests ──

test "laguna plugin: a 4-bit pack (dense layer 0, routed experts, YaRN) is bit-equal to the legacy forward" {
    try case(true, 0x1a91);
}

test "laguna plugin: a bf16 pack is bit-equal to the legacy forward with decode-attn-quant off" {
    const was = decode_attn_quant.decode_attn_quant_override;
    defer decode_attn_quant.decode_attn_quant_override = was;
    decode_attn_quant.decode_attn_quant_override = false;
    try case(false, 0x1a92);
}

test "laguna plugin: a bf16 pack with decode-attn-quant on (int8 and nvfp4 side copies at decode) is bit-equal" {
    const was = decode_attn_quant.decode_attn_quant_override;
    defer decode_attn_quant.decode_attn_quant_override = was;
    decode_attn_quant.decode_attn_quant_override = true;
    try case(false, 0x1a93);
}

test "laguna plugin: a 4-bit pack is bit-equal on the legacy parse's ModelConfig (how a served load builds the shell)" {
    try caseOn(true, 0x1e9c, true);
}

test "laguna plugin: a pack the plugin declines is refused by name (no legacy forward to fall back to)" {
    try both.expectNoLegacyFallback(configJson(true), "expert_quant");
}

test "laguna plugin: mlx-lm's quantization_config copy of the quant block serves through the plugin" {
    const json = try std.mem.concat(std.testing.allocator, u8, &.{ "{\"quantization_config\":{\"group_size\":64,\"bits\":4},", configJson(true)[1..] });
    defer std.testing.allocator.free(json);
    var cfg = try both.servedUnderOverrides(json);
    defer cfg.deinit(std.testing.allocator);
}

test "laguna plugin: under a launch override it serves through the plugin, the override in the engine config and its parse" {
    const model = @import("../../sdk/config/model.zig");
    model.setConfigOverrides("{\"rms_norm_eps\":3e-5,\"max_position_embeddings\":8192}");
    defer model.setConfigOverrides(null);
    var cfg = try both.servedUnderOverrides(configJson(true));
    defer cfg.deinit(std.testing.allocator);
    try std.testing.expect(cfg.max_position_embeddings == 8192 and cfg.rms_norm_eps == 3e-5);
    try std.testing.expectEqual(@as(f32, 3e-5), @import("config.zig").of(&cfg.plugin.?.parsed).rms_norm_eps);
    // An override the forward does not read leaves it bit-equal.
    model.setConfigOverrides("{\"max_position_embeddings\":8192}");
    try caseOn(true, 0x1e9c, true);
}

/// A released pack's config (`fixtures/mlx_lm/`, `poolside_Laguna-XS-2.1-NVFP4-mlx`) at this proof pack's geometry, on the legacy parse's
/// ModelConfig: every key the released config adds, and its own values of the ones listed, reach both forwards.
fn realCase(comptime seed: u64) !void {
    const real = @embedFile("../../engines/mlx/plugin/fixtures/mlx_lm/poolside_Laguna-XS-2.1-NVFP4-mlx.json");
    const json = try both.atGeometry(real, configJson(true), &.{ "rope_parameters", "gating", "moe_routed_scaling_factor", "norm_topk_prob", "max_position_embeddings", "eos_token_id" }, &.{ "mlp_only_layers", "mlp_layer_types", "gating_types" });
    defer std.testing.allocator.free(json);
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writePack(&p, true);
    var b = try openCase(&p, json, true, seed);
    defer b.close();
    try b.expectSteps(&both.long_steps);
    try b.expectGreedy();
    try b.expectDone();
}

test "laguna plugin: the released Laguna-XS-2.1-NVFP4-mlx config at the proof geometry is bit-equal to the legacy forward" {
    try realCase(0x1aa1);
}

test "laguna plugin: every shape its parse declines is refused by name, or fails the legacy parse as it always did" {
    try both.expectDeclines(configJson(true), &.{
        .{ .key = "expert_quant", .value = "{\"bits\":4}", .legacy = .dropped },
        .{ .key = "rope_scaling", .value = "{\"type\":\"linear\",\"factor\":2.0}", .legacy = .dropped },
        .{ .key = "rope_scaling", .value = "{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":1024}", .legacy = .dropped },
        .{ .key = "num_kv_shared_layers", .value = "2", .legacy = .dropped },
        .{ .key = "global_head_dim", .value = "64", .legacy = .dropped },
        .{ .key = "use_bidirectional_attention", .value = "true", .legacy = .failed },
        .{ .key = "linear_num_key_heads", .value = "4", .legacy = .parse },
        .{ .key = "full_attention_interval", .value = "4", .legacy = .failed },
        .{ .key = "hidden_size_per_layer_input", .value = "16", .legacy = .failed },
        .{ .key = "swiglu_limit", .value = "7.0", .legacy = .dropped },
        .{ .key = "attention_k_eq_v", .value = "true", .legacy = .dropped },
        .{ .key = "rope_parameters", .value = "{\"full_attention\":{\"rope_type\":\"llama3\",\"rope_theta\":10000.0,\"factor\":8.0,\"low_freq_factor\":1.0,\"high_freq_factor\":4.0,\"original_max_position_embeddings\":1024}}", .legacy = .dropped },
        .{ .key = "hidden_size", .value = null, .legacy = .failed },
        .{ .key = "num_hidden_layers", .value = null, .legacy = .failed },
        .{ .key = "num_attention_heads", .value = null, .legacy = .dropped },
        .{ .key = "num_key_value_heads", .value = null, .legacy = .failed },
        .{ .key = "head_dim", .value = null, .legacy = .failed },
        .{ .key = "sliding_window", .value = null, .legacy = .dropped },
        .{ .key = "layer_types", .value = null, .legacy = .dropped },
        .{ .key = "num_experts", .value = null, .legacy = .failed },
        .{ .key = "num_experts_per_tok", .value = null, .legacy = .dropped },
        .{ .key = "rope_parameters", .value = null, .legacy = .dropped },
        .{ .key = "layer_types", .value = "[\"sliding_attention\"]", .legacy = .dropped },
        .{ .key = "layer_types", .value = "[1,2,3,4]", .legacy = .dropped },
        .{ .key = "rms_norm_eps", .value = "\"x\"", .legacy = .dropped },
        .{ .key = "rope_theta", .value = "\"x\"", .legacy = .dropped },
        .{ .key = "rope_local_base_freq", .value = "\"x\"", .legacy = .dropped },
        .{ .key = "query_pre_attn_scalar", .value = "0", .legacy = .failed },
        .{ .key = "query_pre_attn_scalar", .value = "\"x\"", .legacy = .parse },
        .{ .key = "head_dim", .value = "127", .legacy = .failed },
        .{ .key = "vocab_size", .value = "\"x\"", .legacy = .parse },
        .{ .key = "eos_token_id", .value = "99999999999", .legacy = .parse },
        .{ .key = "quantization", .value = "{\"group_size\":64,\"bits\":7}", .legacy = .parse },
        .{ .key = "quantization", .value = "{\"group_size\":64,\"bits\":4,\"mode\":\"bogus\"}", .legacy = .parse },
        .{ .key = "quantization", .value = "4", .legacy = .parse },
        .{ .key = "moe_router_logit_softcapping", .value = "30.0", .legacy = .parse },
        .{ .key = "gating", .value = "\"per-channel\"", .legacy = .parse },
        .{ .key = "num_attention_heads_per_layer", .value = "4", .legacy = .failed },
        .{ .key = "moe_routed_scaling_factor", .value = "\"x\"", .legacy = .dropped },
        .{ .key = "rope_parameters", .value = "{\"full_attention\":{\"rope_theta\":500000.0,\"rope_type\":\"yarn\",\"factor\":32.0,\"beta_slow\":1.0,\"beta_fast\":32.0,\"partial_rotary_factor\":0.5},\"sliding_attention\":{\"rope_type\":\"default\",\"rope_theta\":10000.0,\"partial_rotary_factor\":1.0}}", .legacy = .dropped },
        .{ .key = "rope_parameters", .value = "{\"full_attention\":{\"rope_theta\":500000.0,\"rope_type\":\"linear\",\"factor\":32.0,\"original_max_position_embeddings\":8192,\"beta_slow\":1.0,\"beta_fast\":32.0,\"partial_rotary_factor\":0.5},\"sliding_attention\":{\"rope_type\":\"default\",\"rope_theta\":10000.0,\"partial_rotary_factor\":1.0}}", .legacy = .dropped },
    });
}

// ── an in-dir DFlash sidecar ──

/// This pack at the DFlash assistant's geometry (`dflash.TinyFix`: hidden 64, vocab 128).
fn dflashJson(a: std.mem.Allocator) ![]u8 {
    const h = try std.mem.replaceOwned(u8, a, configJson(true), "\"hidden_size\":128", "\"hidden_size\":64");
    defer a.free(h);
    return std.mem.replaceOwned(u8, a, h, "\"vocab_size\":256", "\"vocab_size\":128");
}

// Recorded on the pre-deletion tree, where the plugin's drafted stream, embedding rows and draft head were also
// bit-equal to the legacy forward's.
test "laguna plugin: a pack with a DFlash sidecar serves through the plugin and drafts the legacy forward's stream" {
    const a = std.testing.allocator;
    const json = try dflashJson(a);
    defer a.free(json);
    try both.expectServedWithDrafter(json);
    var p = fixture.Pack.init(0xdf01);
    defer p.deinit();
    try writePackAt(&p, true, 64, 128);
    var b = try both.Both.openAs(&plugin.table, &p, json, true, .none);
    defer b.close();
    try both.expectDflashRecorded(&b.xfm, "applegpu_g17d", &.{ 61, 76, 58, 71, 89, 126, 10, 72, 50, 35, 4, 120, 28, 28, 53, 126 }, &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 });
}

test "laguna plugin: a served pack reports its dense attention projections to /props as the legacy load does" {
    inline for (.{ false, true }) |quantized| {
        var p = fixture.Pack.init(0xda01);
        defer p.deinit();
        try writePack(&p, quantized);
        var b = try both.Both.openAs(&plugin.table, &p, configJson(quantized), true, .none);
        defer b.close();
        try std.testing.expectEqual(!quantized, b.xfm.dense_attn_proj);
    }
}
