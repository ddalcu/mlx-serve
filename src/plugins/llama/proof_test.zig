//! The llama plugin against the legacy dense forward's values (recorded to `fixtures/`) on tiny random Qwen2 (q/k/v
//! biases, tied head) and Mistral packs: steps, greedy decode and batched rows bit-equal, and the engine's config of
//! both parses equal.

const std = @import("std");
const plugin = @import("plugin.zig");
const both = @import("../../engines/mlx/plugin/both.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");

const testing = std.testing;

const H = 128;
const HEADS = 4;
const KV_HEADS = 2;
const HD = 32;
const INTER = 256;
const VOCAB = 256;

fn configJson(comptime model_type: []const u8, comptime quantized: bool, comptime tied: bool, comptime extra: []const u8) []const u8 {
    return "{\"model_type\":\"" ++ model_type ++ "\",\"hidden_size\":128,\"intermediate_size\":256,\"num_hidden_layers\":2," ++
        "\"num_attention_heads\":4,\"num_key_value_heads\":2,\"vocab_size\":256,\"rms_norm_eps\":1e-6," ++
        "\"rope_theta\":1000000.0,\"max_position_embeddings\":32768,\"hidden_act\":\"silu\",\"bos_token_id\":1,\"eos_token_id\":2," ++
        "\"tie_word_embeddings\":" ++ (if (tied) "true" else "false") ++ extra ++
        (if (quantized) ",\"quantization\":{\"group_size\":64,\"bits\":4}" else "") ++ "}";
}

fn at(buf: []u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ layer, name });
}

fn writePack(p: *fixture.Pack, quantized: bool, tied: bool, biases: bool) !void {
    var buf: [128]u8 = undefined;
    try p.weight("model.embed_tokens", VOCAB, H, quantized);
    try p.norm("model.norm.weight", H);
    if (!tied) try p.weight("lm_head", VOCAB, H, quantized);
    for (0..2) |i| {
        try p.norm(try at(&buf, i, "input_layernorm.weight"), H);
        try p.norm(try at(&buf, i, "post_attention_layernorm.weight"), H);
        try p.weight(try at(&buf, i, "self_attn.q_proj"), HEADS * HD, H, quantized);
        try p.weight(try at(&buf, i, "self_attn.k_proj"), KV_HEADS * HD, H, quantized);
        try p.weight(try at(&buf, i, "self_attn.v_proj"), KV_HEADS * HD, H, quantized);
        if (biases) {
            try p.dense(try at(&buf, i, "self_attn.q_proj.bias"), &.{HEADS * HD}, 0.5, 0);
            try p.dense(try at(&buf, i, "self_attn.k_proj.bias"), &.{KV_HEADS * HD}, 0.5, 0);
            try p.dense(try at(&buf, i, "self_attn.v_proj.bias"), &.{KV_HEADS * HD}, 0.5, 0);
        }
        try p.weight(try at(&buf, i, "self_attn.o_proj"), H, HEADS * HD, quantized);
        try p.weight(try at(&buf, i, "mlp.gate_proj"), INTER, H, quantized);
        try p.weight(try at(&buf, i, "mlp.up_proj"), INTER, H, quantized);
        try p.weight(try at(&buf, i, "mlp.down_proj"), H, INTER, quantized);
    }
}

/// The legacy forward's values, recorded to `fixtures/<name>.tape`.
fn open(p: *fixture.Pack, json: []const u8, comptime legacy_config: bool, comptime name: []const u8) !both.Both {
    return both.Both.replay(&plugin.table, p, json, legacy_config, @embedFile("fixtures/" ++ name ++ ".tape"));
}

fn expectAll(b: *both.Both) !void {
    try b.expectSteps(&both.steps);
    try b.expectGreedy();
    try b.expectBatched();
    try b.expectDone();
}

/// Fields only the plugin's forward reads; the adapter leaves their defaults.
const owned = [_][]const u8{ "rms_norm_eps", "rope_theta", "rope_local_base_freq", "rope_llama3", "tie_word_embeddings", "query_pre_attn_scalar" };

const qwen2_json = configJson("qwen2", true, true, ",\"use_sliding_window\":false,\"sliding_window\":32768,\"max_window_layers\":2");
const mistral_json = configJson("mistral", true, false, ",\"head_dim\":null,\"sliding_window\":4096");

// ── tests ──

test "llama plugin: a Qwen2 4-bit pack (q/k/v biases, tied head) is bit-equal to the legacy forward" {
    var p = fixture.Pack.init(0x0e2a);
    defer p.deinit();
    try writePack(&p, true, true, true);
    var b = try open(&p, qwen2_json, false, "0e2a");
    defer b.close();
    try expectAll(&b);
}

test "llama plugin: a Qwen2 bf16 pack with biases is bit-equal to the legacy forward" {
    var p = fixture.Pack.init(0x0e2b);
    defer p.deinit();
    try writePack(&p, false, true, true);
    var b = try open(&p, configJson("qwen2", false, true, ""), false, "0e2b");
    defer b.close();
    try expectAll(&b);
}

test "llama plugin: a Mistral 4-bit pack is bit-equal to the legacy forward" {
    var p = fixture.Pack.init(0x3157);
    defer p.deinit();
    try writePack(&p, true, false, false);
    var b = try open(&p, mistral_json, false, "3157");
    defer b.close();
    try expectAll(&b);
}

test "llama plugin: Qwen2 and Mistral 4-bit packs are bit-equal on the legacy parse's ModelConfig" {
    inline for (.{ qwen2_json, mistral_json }, .{ true, false }, .{ "1e9c-lc-qwen2", "1e9c-lc-mistral" }) |json, qwen, name| {
        var p = fixture.Pack.init(0x1e9c);
        defer p.deinit();
        try writePack(&p, true, qwen, qwen);
        var b = try open(&p, json, true, name);
        defer b.close();
        try expectAll(&b);
    }
}

test "llama plugin: the served config of Qwen2 and Mistral packs equals the legacy parse" {
    try both.expectConfigEqual(&plugin.table, qwen2_json, &owned);
    try both.expectConfigEqual(&plugin.table, mistral_json, &owned);
    try both.expectConfigEqual(&plugin.table, configJson("qwen2", false, true, ""), &owned);
}

test "llama plugin: a launch override the forward reads (rope_theta, the context) is bit-equal to the legacy forward under it" {
    const model = @import("../../sdk/config/model.zig");
    model.setConfigOverrides("{\"rope_theta\":12345.0,\"max_position_embeddings\":8192}");
    defer model.setConfigOverrides(null);
    var p = fixture.Pack.init(0x0e2c);
    defer p.deinit();
    try writePack(&p, true, true, true);
    var b = try open(&p, qwen2_json, true, "0e2c-lc");
    defer b.close();
    try testing.expect(b.cfg.rope_theta == 12345.0 and b.cfg.max_position_embeddings == 8192);
    try testing.expectEqual(@as(f32, 12345.0), plugin.config.of(&b.cfg.plugin.?.parsed).rope_theta);
    try expectAll(&b);
    var served = try both.servedUnderOverrides(qwen2_json);
    defer served.deinit(testing.allocator);
    try testing.expect(served.rope_theta == 12345.0 and served.max_position_embeddings == 8192);
}

test "llama plugin: lm_head_mxfp8 (LOSSY) quantizes a bf16 Mistral pack's head as the legacy load does" {
    const cli = &@import("../../engines/mlx/transformer.zig").cli().lm_head_mxfp8_cli;
    const was = cli.*;
    defer cli.* = was;
    cli.* = true;
    var p = fixture.Pack.init(0x3158);
    defer p.deinit();
    try writePack(&p, false, false, false);
    var b = try open(&p, configJson("mistral", false, false, ",\"head_dim\":null,\"sliding_window\":4096"), false, "mxfp8-3158");
    defer b.close();
    try testing.expect(b.xfm.lm_head_s.ctx != null);
    try expectAll(&b);
}
