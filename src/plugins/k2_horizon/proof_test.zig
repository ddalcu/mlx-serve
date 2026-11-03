//! The k2_horizon plugin against the legacy dense forward on tiny random K2 packs (grouped norms; 4-bit and bf16):
//! steps, greedy decode and batched rows bit-equal, and the engine's config of both parses equal.

const std = @import("std");
const plugin = @import("plugin.zig");
const both = @import("../../engines/mlx/plugin/both.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");

const H = 128;
const INTER = 256;
const VOCAB = 256;

fn configJson(comptime quantized: bool) []const u8 {
    return "{\"model_type\":\"k2_horizon\",\"hidden_size\":128,\"intermediate_size\":256,\"num_hidden_layers\":2," ++
        "\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":32,\"vocab_size\":256,\"rms_norm_eps\":1e-6," ++
        "\"hidden_act\":\"silu\",\"layernorm_num_groups\":4,\"query_key_norm\":false,\"attention_gate_func\":null," ++
        "\"num_experts\":0,\"sliding_window\":null,\"tie_word_embeddings\":false,\"max_position_embeddings\":524288," ++
        "\"rope_parameters\":{\"rope_theta\":10000000.0,\"rope_type\":\"default\"}" ++
        (if (quantized) ",\"quantization\":{\"group_size\":64,\"bits\":4}" else "") ++ "}";
}

fn at(buf: []u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ layer, name });
}

fn writePack(p: *fixture.Pack, quantized: bool) !void {
    var buf: [128]u8 = undefined;
    try p.weight("model.embed_tokens", VOCAB, H, quantized);
    try p.norm("model.norm.weight", H);
    try p.weight("lm_head", VOCAB, H, quantized);
    for (0..2) |i| {
        try p.norm(try at(&buf, i, "input_layernorm.weight"), H);
        try p.norm(try at(&buf, i, "post_attention_layernorm.weight"), H);
        try p.weight(try at(&buf, i, "self_attn.q_proj"), 128, H, quantized);
        try p.weight(try at(&buf, i, "self_attn.k_proj"), 64, H, quantized);
        try p.weight(try at(&buf, i, "self_attn.v_proj"), 64, H, quantized);
        try p.weight(try at(&buf, i, "self_attn.o_proj"), H, 128, quantized);
        try p.weight(try at(&buf, i, "mlp.gate_proj"), INTER, H, quantized);
        try p.weight(try at(&buf, i, "mlp.up_proj"), INTER, H, quantized);
        try p.weight(try at(&buf, i, "mlp.down_proj"), H, INTER, quantized);
    }
}

/// Fields only the plugin's forward reads; the adapter leaves their defaults.
const owned = [_][]const u8{ "rms_norm_eps", "rope_theta", "rope_local_base_freq", "rope_llama3", "tie_word_embeddings", "query_pre_attn_scalar", "norm_groups" };

fn case(comptime quantized: bool, comptime seed: u64) !void {
    return caseOn(quantized, seed, false);
}

/// The legacy forward's values, recorded to `fixtures/`.
fn caseOn(comptime quantized: bool, comptime seed: u64, comptime legacy_config: bool) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writePack(&p, quantized);
    const name = std.fmt.comptimePrint("fixtures/{x}{s}.tape", .{ seed, if (legacy_config) "-lc" else "" });
    var b = try both.Both.replay(&plugin.table, &p, configJson(quantized), legacy_config, @embedFile(name));
    defer b.close();
    try b.expectSteps(&both.steps);
    try b.expectGreedy();
    try b.expectBatched();
    try b.expectDone();
}

// ── tests ──

test "k2_horizon plugin: a 4-bit pack with grouped norms is bit-equal to the legacy forward" {
    try case(true, 0x42a1);
}

test "k2_horizon plugin: a bf16 pack with grouped norms is bit-equal to the legacy forward" {
    try case(false, 0x42a2);
}

test "k2_horizon plugin: the served config equals the legacy parse" {
    try both.expectConfigEqual(&plugin.table, configJson(true), &owned);
}

test "k2_horizon plugin: a 4-bit pack is bit-equal on the legacy parse's ModelConfig (how a served load builds the shell)" {
    try caseOn(true, 0x1e9c, true);
}
