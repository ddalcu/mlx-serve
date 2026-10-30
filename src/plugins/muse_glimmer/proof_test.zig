//! The muse_glimmer plugin against the legacy dense forward on tiny random Muse text packs at the real head width
//! (hd 128, GQA): normed embeddings, centred sandwich norms, the weight-less q/k norm and scale factor, the channel
//! gate, NoPE full layers, sliding layers, the output multiplier and the compiled softcap. 4-bit under a
//! `text_config` and bf16 flat (`muse_glimmer_text`); steps, greedy decode and batched rows.

const std = @import("std");
const plugin = @import("plugin.zig");
const both = @import("../../engines/mlx/plugin/both.zig");
const plugin_config = @import("../../engines/mlx/plugin/config.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const model = @import("../../sdk/config/model.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const ops = @import("../../mlx/ops.zig");
const dflash_mod = @import("../../engines/mlx/spec/dflash/dflash.zig");
const Generator = @import("../../engines/mlx/generate/generator.zig").Generator;
const Tokenizer = @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const testing = std.testing;

const H = 128;
const HEADS = 4;
const KV = 2;
const HD = 128;
const INTER = 256;
const VOCAB = 256;

const text =
    "\"hidden_size\":128,\"intermediate_size\":256,\"num_hidden_layers\":4,\"num_attention_heads\":4," ++
    "\"num_key_value_heads\":2,\"head_dim\":128,\"hidden_activation\":\"silu\",\"rms_norm_eps\":1e-05," ++
    "\"post_norm_eps\":1e-08,\"qk_scale_factor\":3.87,\"output_multiplier\":0.19611613513818404," ++
    "\"final_logit_softcapping\":20.0,\"vocab_size\":256,\"max_position_embeddings\":4096,\"tie_word_embeddings\":false," ++
    "\"eos_token_id\":200001,\"sliding_window\":4,\"layer_types\":[\"sliding_attention\",\"sliding_attention\"," ++
    "\"sliding_attention\",\"full_attention\"],\"layer_rope_theta\":[500000.0,500000.0,500000.0,0]," ++
    "\"rope_parameters\":{\"rope_theta\":500000.0,\"rope_type\":\"default\"}";

fn configJson(comptime nested: bool, comptime quantized: bool) []const u8 {
    const q = if (quantized) ",\"quantization\":{\"group_size\":64,\"bits\":4}" else "";
    return if (nested)
        "{\"model_type\":\"muse_glimmer\",\"text_config\":{\"model_type\":\"muse_glimmer_text\"," ++ text ++ "}" ++ q ++ "}"
    else
        "{\"model_type\":\"muse_glimmer_text\"," ++ text ++ q ++ "}";
}

fn at(buf: []u8, prefix: []const u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ prefix, layer, name });
}

fn writePack(p: *fixture.Pack, prefix: []const u8, q: bool) !void {
    return writePackAt(p, prefix, q, H, VOCAB);
}

fn writePackAt(p: *fixture.Pack, prefix: []const u8, q: bool, comptime h: c_int, comptime vocab: c_int) !void {
    var buf: [160]u8 = undefined;
    var nb: [96]u8 = undefined;
    try p.weight(try std.fmt.bufPrint(&nb, "{s}.embed_tokens", .{prefix}), vocab, h, q);
    try p.norm(try std.fmt.bufPrint(&nb, "{s}.norm.weight", .{prefix}), h);
    try p.weight("lm_head", vocab, h, q);
    for (0..4) |i| {
        for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight" }) |n|
            try p.norm(try at(&buf, prefix, i, n), h);
        try p.weight(try at(&buf, prefix, i, "self_attn.q_proj"), HEADS * HD, h, q);
        try p.weight(try at(&buf, prefix, i, "self_attn.k_proj"), KV * HD, h, q);
        try p.weight(try at(&buf, prefix, i, "self_attn.v_proj"), KV * HD, h, q);
        try p.weight(try at(&buf, prefix, i, "self_attn.gate_proj"), HEADS * HD, h, q);
        try p.weight(try at(&buf, prefix, i, "self_attn.o_proj"), h, HEADS * HD, q);
        try p.weight(try at(&buf, prefix, i, "mlp.gate_proj"), INTER, h, q);
        try p.weight(try at(&buf, prefix, i, "mlp.up_proj"), INTER, h, q);
        try p.weight(try at(&buf, prefix, i, "mlp.down_proj"), h, INTER, q);
    }
}

fn case(comptime nested: bool, comptime quantized: bool, comptime seed: u64) !void {
    return caseOn(nested, quantized, seed, false);
}


fn openCase(p: *fixture.Pack, json: []const u8, comptime legacy_config: bool, comptime seed: u64) !both.Both {
    const name = std.fmt.comptimePrint("fixtures/{x}{s}.tape", .{ seed, if (legacy_config) "-lc" else "" });
    return both.Both.replay(&plugin.table, p, json, legacy_config, @embedFile(name));
}

fn caseOn(comptime nested: bool, comptime quantized: bool, comptime seed: u64, comptime legacy_config: bool) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writePack(&p, if (nested) "model.language_model" else "model", quantized);
    var b = try openCase(&p, configJson(nested, quantized), legacy_config, seed);
    defer b.close();
    // The served legacy load compiles the softcap; the plugin carries the same closure.
    try b.expectSteps(&both.long_steps);
    try b.expectGreedy();
    try b.expectBatched();
    try b.expectDone();
}

// ── tests ──

test "muse_glimmer plugin: a 4-bit pack under a text_config is bit-equal to the legacy forward, batched too" {
    try case(true, true, 0x3e5e);
}

test "muse_glimmer plugin: a bf16 muse_glimmer_text pack is bit-equal to the legacy forward, batched too" {
    try case(false, false, 0x3e5f);
}

// ── a DFlash drafter (the `-assistant` sidecar) on the served trunk ──

test "muse_glimmer plugin: a DFlash drafter reads the bare table and head and emits the legacy stream" {
    const a = testing.allocator;
    // The tiny assistant's geometry (`dflash.TinyFix`): hidden 64, vocab 128, targets layers 0 and 2.
    const json = try std.mem.replaceOwned(u8, a, configJson(true, true), "\"hidden_size\":128", "\"hidden_size\":64");
    defer a.free(json);
    const json2 = try std.mem.replaceOwned(u8, a, json, "\"vocab_size\":256", "\"vocab_size\":128");
    defer a.free(json2);
    var p = fixture.Pack.init(0x3e60);
    defer p.deinit();
    try writePackAt(&p, "model.language_model", true, 64, 128);
    var b = try openCase(&p, json2, false, 0x3e60);
    defer b.close();
    try testing.expect(b.xfm.supportsLayerCapture());
    {
        var g = ops.Graph.init(a, b.xfm.s);
        defer g.deinit();
        const ids = try fixture.tokens(&g, &.{ 4, 90, 127, 0, 33 });
        const raw = try g.adopt(try b.xfm.rawEmbedding(ids));
        try b.check(raw, false);
        const hidden = try g.reshape(raw, &.{ 1, 5, 64 });
        try b.check(try g.adopt(try b.xfm.lmHeadForDraft(hidden)), false);
    }
    try b.expectDone();
    // Recorded from the legacy forward, whose drafted stream the plugin's matched.
    try both.expectDflashRecorded(&b.xfm, "applegpu_g17d", &.{ 19, 30, 58, 1, 30, 122, 46, 83, 60, 116, 119, 107, 60, 3, 32, 126 }, &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
}

/// A pack with its vision tower declared and image id 250.
fn visionJson(a: std.mem.Allocator) ![]u8 {
    return std.mem.replaceOwned(u8, a, configJson(false, false), "{\"model_type\"", "{\"image_token_id\":250,\"vision_config\":{\"model_type\":\"muse_glimmer_vision\"},\"model_type\"");
}

test "muse_glimmer plugin: a vision pack's media rows at the root's image id splice as the legacy forward's (across a chunk boundary)" {
    const a = testing.allocator;
    const json = try visionJson(a);
    defer a.free(json);
    var p = fixture.Pack.init(0x3e61);
    defer p.deinit();
    try writePack(&p, "model", false);
    var b = try openCase(&p, json, true, 0x3e61);
    defer b.close();
    try testing.expect(b.cfg.has_vision);
    try b.expectMedia(250, H);
    try b.expectDone();
}

test "muse_glimmer plugin: a vision pack is served by the plugin (the engine loads its tower beside it)" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const json = try visionJson(a);
    defer a.free(json);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
    var path_buf: [512]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const bind = (try plugin_config.bindServed(io, a, dir, json)) orelse return error.NotServed;
    defer bind.destroy(a);
    try testing.expect(bind.table == &plugin.table and bind.parsed.caps.images);
}

test "muse_glimmer plugin: a 4-bit pack is bit-equal on the legacy parse's ModelConfig (how a served load builds the shell)" {
    try caseOn(true, true, 0x1e9c, true);
}

test "muse_glimmer plugin: a served vision pack's tower (nested under model.) loads into the engine's map, not the plugin's" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const json = try visionJson(a);
    defer a.free(json);
    var p = fixture.Pack.init(0x3e62);
    defer p.deinit();
    try writePack(&p, "model", false);
    try p.norm("model.vision_tower.ln_pre.weight", 32);
    try p.norm("model.vision_adapter.fc1.weight", 32);
    try p.norm("model.vision_projection.weight", 32);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
    var path_buf: [512]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    try p.save(dir);
    var cfg = try model.parseConfig(io, a, dir);
    defer cfg.deinit(a);
    try testing.expect(cfg.plugin != null and cfg.has_vision);
    var tower = model.Weights.init(a);
    defer tower.deinit();
    var x = try driver.loadWith(io, a, &cfg, dir, fixture.stream(), &tower);
    defer x.deinit();
    try testing.expectEqual(@as(u32, 3), tower.map.count());
    try testing.expect(tower.map.contains("model.vision_tower.ln_pre.weight") and tower.map.contains("model.vision_projection.weight"));
}

test "muse_glimmer plugin: under a launch override it serves through the plugin, the override in the engine config and its parse" {
    model.setConfigOverrides("{\"rms_norm_eps\":3e-5,\"max_position_embeddings\":8192}");
    defer model.setConfigOverrides(null);
    var cfg = try both.servedUnderOverrides(configJson(false, true));
    defer cfg.deinit(std.testing.allocator);
    try std.testing.expect(cfg.max_position_embeddings == 8192 and cfg.rms_norm_eps == 3e-5);
    try std.testing.expectEqual(@as(f32, 3e-5), @import("config.zig").of(&cfg.plugin.?.parsed).rms_norm_eps);
    // An override the forward does not read leaves it bit-equal.
    model.setConfigOverrides("{\"max_position_embeddings\":8192}");
    try caseOn(true, true, 0x1e9c, true);
}

// The released `ddalcu/Muse-Glimmer-30B-MLX-Serve-4bit` config (`engines/mlx/plugin/fixtures/mlx_lm/`) at this proof
// pack's geometry, on the legacy parse's ModelConfig: every key the released config adds (its vision tower declared,
// none in the pack), and its own values of the ones listed, reach both forwards.
test "muse_glimmer plugin: the released Muse-Glimmer-30B-MLX-Serve-4bit config at the proof geometry is bit-equal to the legacy forward" {
    const real = @embedFile("../../engines/mlx/plugin/fixtures/mlx_lm/ddalcu_Muse-Glimmer-30B-MLX-Serve-4bit.json");
    const keep = [_][]const u8{ "rope_parameters", "qk_scale_factor", "output_multiplier", "post_norm_eps", "final_logit_softcapping", "rms_norm_eps", "hidden_activation", "eos_token_id", "bos_token_id", "max_position_embeddings" };
    const json = try both.atGeometry(real, configJson(true, true), &keep, &.{});
    defer testing.allocator.free(json);
    var p = fixture.Pack.init(0x3ea1);
    defer p.deinit();
    try writePack(&p, "model.language_model", true);
    var b = try openCase(&p, json, true, 0x3ea1);
    defer b.close();
    try b.expectSteps(&both.long_steps);
    try b.expectGreedy();
    try b.expectDone();
}

test "muse_glimmer plugin: a pack the plugin declines is refused by name (no legacy forward to fall back to)" {
    try both.expectNoLegacyFallback(configJson(false, true), "rope_scaling");
}

test "muse_glimmer plugin: every shape its parse declines is refused by name, or fails the legacy parse as it always did" {
    try both.expectDeclines(configJson(false, true), &.{
        .{ .key = "rope_scaling", .value = "{\"type\":\"linear\",\"factor\":2.0}", .legacy = .dropped },
        .{ .key = "rope_scaling", .value = "{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":1024}", .legacy = .dropped },
        .{ .key = "num_kv_shared_layers", .value = "2", .legacy = .dropped },
        .{ .key = "global_head_dim", .value = "64", .legacy = .failed },
        .{ .key = "use_bidirectional_attention", .value = "true", .legacy = .dropped },
        .{ .key = "attention_k_eq_v", .value = "true", .legacy = .dropped },
        .{ .key = "num_experts", .value = "8", .legacy = .failed },
        .{ .key = "rope_parameters", .value = "{\"full_attention\":{\"rope_type\":\"llama3\",\"rope_theta\":10000.0,\"factor\":8.0,\"low_freq_factor\":1.0,\"high_freq_factor\":4.0,\"original_max_position_embeddings\":1024}}", .legacy = .dropped },
        .{ .key = "rope_parameters", .value = "{\"full_attention\":{\"rope_type\":\"yarn\",\"rope_theta\":10000.0,\"factor\":4.0,\"original_max_position_embeddings\":1024}}", .legacy = .dropped },
        .{ .key = "hidden_size", .value = null, .legacy = .failed },
        .{ .key = "num_hidden_layers", .value = null, .legacy = .failed },
        .{ .key = "num_attention_heads", .value = null, .legacy = .failed },
        .{ .key = "num_key_value_heads", .value = null, .legacy = .failed },
        .{ .key = "head_dim", .value = null, .legacy = .failed },
        .{ .key = "sliding_window", .value = null, .legacy = .dropped },
        .{ .key = "layer_types", .value = null, .legacy = .dropped },
        .{ .key = "layer_types", .value = "[\"sliding_attention\"]", .legacy = .dropped },
        .{ .key = "layer_types", .value = "[1,2,3,4]", .legacy = .dropped },
        .{ .key = "rms_norm_eps", .value = "\"x\"", .legacy = .dropped },
        .{ .key = "rope_theta", .value = "\"x\"", .legacy = .dropped },
        .{ .key = "rope_local_base_freq", .value = "\"x\"", .legacy = .dropped },
        .{ .key = "head_dim", .value = "127", .legacy = .failed },
        .{ .key = "vocab_size", .value = "\"x\"", .legacy = .parse },
        .{ .key = "eos_token_id", .value = "99999999999", .legacy = .parse },
        .{ .key = "quantization", .value = "{\"group_size\":64,\"bits\":7}", .legacy = .parse },
        .{ .key = "quantization", .value = "{\"group_size\":64,\"bits\":4,\"mode\":\"bogus\"}", .legacy = .parse },
        .{ .key = "quantization", .value = "4", .legacy = .parse },
        .{ .key = "rope_parameters", .value = "{\"full_attention\":{\"rope_theta\":500000.0},\"sliding_attention\":{\"rope_theta\":10000.0}}", .legacy = .dropped },
        .{ .key = "rope_parameters", .value = "{\"rope_type\":\"llama3\",\"rope_theta\":500000.0,\"factor\":8.0,\"low_freq_factor\":1.0,\"high_freq_factor\":4.0,\"original_max_position_embeddings\":1024}", .legacy = .dropped },
        .{ .key = "rope_parameters", .value = "{\"rope_type\":\"default\",\"rope_theta\":\"x\"}", .legacy = .dropped },
        .{ .key = "qk_scale_factor", .value = "0", .legacy = .dropped },
        .{ .key = "output_multiplier", .value = "0", .legacy = .dropped },
        .{ .key = "post_norm_eps", .value = "0", .legacy = .dropped },
        .{ .key = "qk_scale_factor", .value = "\"x\"", .legacy = .dropped },
        .{ .key = "layer_rope_theta", .value = "3", .legacy = .dropped },
        .{ .key = "layer_rope_theta", .value = "[\"a\",\"b\",\"c\",\"d\"]", .legacy = .dropped },
    });
}
