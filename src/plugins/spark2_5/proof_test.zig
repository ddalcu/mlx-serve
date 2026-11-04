//! The spark2_5 plugin against the legacy dense forward on tiny random Spark packs at the real head width (hd 256, so
//! the fused RoPE decode kernel runs): three sliding layers and a full one with 25% rotary, the per-head gate, the
//! split q_k_v_proj; 4-bit and bf16. Steps crossing the window, greedy decode and batched rows bit-equal.

const std = @import("std");
const plugin = @import("plugin.zig");
const both = @import("../../engines/mlx/plugin/both.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");

const H = 128;
const HEADS = 2;
const KV_HEADS = 1;
const HD = 256;
const INTER = 256;
const VOCAB = 256;

fn configJson(comptime quantized: bool) []const u8 {
    return "{\"model_type\":\"spark2_5\",\"hidden_size\":128,\"intermediate_size\":256,\"num_hidden_layers\":4," ++
        "\"num_attention_heads\":2,\"num_key_value_heads\":1,\"head_dim\":256,\"vocab_size\":256,\"rms_norm_eps\":1e-6," ++
        "\"sliding_window\":4,\"layer_types\":[\"sliding_attention\",\"sliding_attention\",\"sliding_attention\",\"full_attention\"]," ++
        "\"rope_parameters\":{\"full_attention\":{\"rope_theta\":5000000.0,\"partial_rotary_factor\":0.25}," ++
        "\"sliding_attention\":{\"rope_theta\":10000.0}},\"tie_word_embeddings\":true,\"max_position_embeddings\":4096" ++
        (if (quantized) ",\"quantization\":{\"group_size\":64,\"bits\":4}" else "") ++ "}";
}

fn at(buf: []u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ layer, name });
}

fn writePack(p: *fixture.Pack, quantized: bool) !void {
    return writePackAt(p, quantized, H, VOCAB);
}

/// The pack at hidden width `h` and vocabulary `vocab` (the DFlash proof's tiny assistant reads 64 and 128).
fn writePackAt(p: *fixture.Pack, quantized: bool, comptime h: c_int, comptime vocab: c_int) !void {
    var buf: [128]u8 = undefined;
    try p.weight("model.embedding", vocab, h, quantized);
    try p.norm("model.norm.weight", h);
    for (0..4) |i| {
        try p.norm(try at(&buf, i, "input_layernorm.weight"), h);
        try p.norm(try at(&buf, i, "post_attention_layernorm.weight"), h);
        try p.weight(try at(&buf, i, "self_attn.q_k_v_proj"), (HEADS + 2 * KV_HEADS) * HD, h, quantized);
        try p.weight(try at(&buf, i, "self_attn.out_proj"), h, HEADS * HD, quantized);
        try p.weight(try at(&buf, i, "self_attn.g_proj"), HEADS, h, quantized);
        try p.weight(try at(&buf, i, "mlp.gate_proj"), INTER, h, quantized);
        try p.weight(try at(&buf, i, "mlp.up_proj"), INTER, h, quantized);
        try p.weight(try at(&buf, i, "mlp.down_proj"), h, INTER, quantized);
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
    try b.expectBatched();
    try b.expectDone();
}

// ── tests ──

test "spark2_5 plugin: a 4-bit pack at hd 256 is bit-equal to the legacy forward" {
    try case(true, 0x5a25);
}

test "spark2_5 plugin: a bf16 pack at hd 256 is bit-equal to the legacy forward" {
    try case(false, 0x5a26);
}

test "spark2_5 plugin: a 4-bit pack is bit-equal on the legacy parse's ModelConfig (how a served load builds the shell)" {
    try caseOn(true, 0x1e9c, true);
}

test "spark2_5 plugin: a pack the plugin declines is refused by name (no legacy forward to fall back to)" {
    try both.expectNoLegacyFallback(configJson(true), "expert_quant");
}

test "spark2_5 plugin: under a launch override it serves through the plugin, the override in the engine config and its parse" {
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

/// A released pack's config (`fixtures/mlx_lm/`, `abenzerps_Spark-X2.5-4B-MLX-4bit`) at this proof pack's geometry, on the legacy parse's
/// ModelConfig: every key the released config adds, and its own values of the ones listed, reach both forwards.
fn realCase(comptime seed: u64) !void {
    const real = @embedFile("../../engines/mlx/plugin/fixtures/mlx_lm/abenzerps_Spark-X2.5-4B-MLX-4bit.json");
    const json = try both.atGeometry(real, configJson(true), &.{ "rope_parameters", "tie_word_embeddings", "max_position_embeddings", "eos_token_id" }, &.{});
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

test "spark2_5 plugin: the released Spark-X2.5-4B-MLX-4bit config at the proof geometry is bit-equal to the legacy forward" {
    try realCase(0x5aa1);
}

test "spark2_5 plugin: every shape its parse declines is refused by name, or fails the legacy parse as it always did" {
    try both.expectDeclines(configJson(true), &.{
        .{ .key = "expert_quant", .value = "{\"bits\":4}", .legacy = .dropped },
        .{ .key = "rope_scaling", .value = "{\"type\":\"linear\",\"factor\":2.0}", .legacy = .dropped },
        .{ .key = "rope_scaling", .value = "{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":1024}", .legacy = .dropped },
        .{ .key = "num_kv_shared_layers", .value = "2", .legacy = .dropped },
        .{ .key = "global_head_dim", .value = "64", .legacy = .failed },
        .{ .key = "num_experts", .value = "8", .legacy = .failed },
        .{ .key = "num_local_experts", .value = "8", .legacy = .dropped },
        .{ .key = "rope_parameters", .value = "{\"rope_theta\":10000.0,\"partial_rotary_factor\":0.5}", .legacy = .dropped },
        .{ .key = "rope_parameters", .value = "{\"full_attention\":{\"rope_type\":\"llama3\",\"rope_theta\":10000.0,\"factor\":8.0,\"low_freq_factor\":1.0,\"high_freq_factor\":4.0,\"original_max_position_embeddings\":1024}}", .legacy = .dropped },
        .{ .key = "rope_parameters", .value = "{\"full_attention\":{\"rope_type\":\"yarn\",\"rope_theta\":10000.0,\"factor\":4.0,\"original_max_position_embeddings\":1024}}", .legacy = .dropped },
        .{ .key = "hidden_size", .value = null, .legacy = .failed },
        .{ .key = "num_hidden_layers", .value = null, .legacy = .failed },
        .{ .key = "num_attention_heads", .value = null, .legacy = .failed },
        .{ .key = "num_key_value_heads", .value = null, .legacy = .failed },
        .{ .key = "head_dim", .value = null, .legacy = .dropped },
        .{ .key = "sliding_window", .value = null, .legacy = .dropped },
        .{ .key = "layer_types", .value = null, .legacy = .dropped },
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
        .{ .key = "rope_parameters", .value = "{\"full_attention\":{\"rope_theta\":5000000.0,\"partial_rotary_factor\":0.25,\"rope_type\":\"yarn\"},\"sliding_attention\":{\"rope_theta\":10000.0}}", .legacy = .dropped },
        .{ .key = "rope_parameters", .value = "{\"full_attention\":{\"rope_theta\":5000000.0,\"partial_rotary_factor\":0.25,\"rope_type\":\"linear\"},\"sliding_attention\":{\"rope_theta\":10000.0}}", .legacy = .dropped },
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
test "spark2_5 plugin: a pack with a DFlash sidecar serves through the plugin and drafts the legacy forward's stream" {
    const a = std.testing.allocator;
    const json = try dflashJson(a);
    defer a.free(json);
    try both.expectServedWithDrafter(json);
    var p = fixture.Pack.init(0xdf04);
    defer p.deinit();
    try writePackAt(&p, true, 64, 128);
    var b = try both.Both.openAs(&plugin.table, &p, json, true, .none);
    defer b.close();
    try both.expectDflashRecorded(&b.xfm, "applegpu_g17d", &.{ 125, 125, 109, 109, 109, 109, 56, 52, 52, 52, 52, 52, 52, 52, 52, 52 }, &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
}
