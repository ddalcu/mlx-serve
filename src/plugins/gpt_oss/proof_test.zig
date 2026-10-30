//! The gpt_oss plugin against the legacy layered forward on tiny random gpt-oss packs: sinks, additive attention
//! biases, YaRN on every layer, alternating sliding/full layers, routed experts with router and per-expert biases
//! under the clamped SwiGLU. 4-bit, and bf16 with decode-attn-quant off; steps crossing the window and greedy decode.

const std = @import("std");
const plugin = @import("plugin.zig");
const both = @import("../../engines/mlx/plugin/both.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const transformer = @import("../../engines/mlx/transformer.zig");

const H = 128;
const HEADS = 4;
const KV = 2;
const HD = 64;
const E = 8;
const I = 128;
const VOCAB = 256;

fn configJson(comptime quantized: bool) []const u8 {
    return "{\"model_type\":\"gpt_oss\",\"hidden_size\":128,\"intermediate_size\":128,\"num_hidden_layers\":4," ++
        "\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":64,\"num_local_experts\":8,\"experts_per_token\":2," ++
        "\"rms_norm_eps\":1e-05,\"layer_types\":[\"sliding_attention\",\"full_attention\",\"sliding_attention\",\"full_attention\"]," ++
        "\"max_position_embeddings\":131072,\"rope_scaling\":{\"beta_fast\":32.0,\"beta_slow\":1.0,\"factor\":32.0," ++
        "\"original_max_position_embeddings\":4096,\"rope_type\":\"yarn\",\"truncate\":false},\"rope_theta\":150000," ++
        "\"sliding_window\":4,\"swiglu_limit\":7.0,\"tie_word_embeddings\":false,\"vocab_size\":256,\"eos_token_id\":200002" ++
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
    for (0..4) |i| {
        try p.norm(try at(&buf, i, "input_layernorm.weight"), h);
        try p.norm(try at(&buf, i, "post_attention_layernorm.weight"), h);
        inline for (.{ .{ "q_proj", HEADS * HD, h }, .{ "k_proj", KV * HD, h }, .{ "v_proj", KV * HD, h }, .{ "o_proj", h, HEADS * HD } }) |t| {
            try p.weight(try at(&buf, i, "self_attn." ++ t[0]), t[1], t[2], q);
            try p.denseAs(try at(&buf, i, "self_attn." ++ t[0] ++ ".bias"), &.{t[1]}, 0.1, 0, .bfloat16);
        }
        try p.denseAs(try at(&buf, i, "self_attn.sinks"), &.{HEADS}, 1.0, 0, .bfloat16);
        try p.weight(try at(&buf, i, "mlp.router"), E, h, q);
        try p.denseAs(try at(&buf, i, "mlp.router.bias"), &.{E}, 0.1, 0, .bfloat16);
        inline for (.{ .{ "gate_proj", I, h }, .{ "up_proj", I, h }, .{ "down_proj", h, I } }) |t| {
            try p.tensor(try at(&buf, i, "mlp.experts." ++ t[0]), &.{ E, t[1], t[2] }, q);
            try p.denseAs(try at(&buf, i, "mlp.experts." ++ t[0] ++ ".bias"), &.{ E, t[1] }, 0.1, 0, .bfloat16);
        }
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

test "gpt_oss plugin: a 4-bit pack (sinks, biases, YaRN, biased experts) is bit-equal to the legacy forward" {
    try case(true, 0x9055);
}

test "gpt_oss plugin: a bf16 pack is bit-equal to the legacy forward with decode-attn-quant off" {
    const was = decode_attn_quant.decode_attn_quant_override;
    defer decode_attn_quant.decode_attn_quant_override = was;
    decode_attn_quant.decode_attn_quant_override = false;
    try case(false, 0x9056);
}

test "gpt_oss plugin: a bf16 pack with decode-attn-quant on (its decode side copies) is bit-equal to the legacy forward" {
    const was = decode_attn_quant.decode_attn_quant_override;
    defer decode_attn_quant.decode_attn_quant_override = was;
    decode_attn_quant.decode_attn_quant_override = true;
    try case(false, 0x9057);
}

test "gpt_oss plugin: a 4-bit pack is bit-equal on the legacy parse's ModelConfig (how a served load builds the shell)" {
    try caseOn(true, 0x1e9c, true);
}

test "gpt_oss plugin: a pack the plugin declines is refused by name (no legacy forward to fall back to)" {
    try both.expectNoLegacyFallback(configJson(true), "swiglu_clamp");
}

test "gpt_oss plugin: under a launch override it serves through the plugin, the override in the engine config and its parse" {
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

// ── a `--mlx-gguf` file ──

const mlx = @import("../../mlx/ffi.zig");
const gguf_kernels = @import("../../sdk/quant/gguf/kernels.zig");
const GgmlType = @import("../../sdk/quant/gguf/quants.zig").GgmlType;
const decode_attn_quant = @import("../../sdk/blocks/attention/decode_attn_quant.zig");

/// `<base>.weight` and its sentinel as the GGUF loader leaves a block-quantized tensor (`sdk/quant/gguf/weights.zig`):
/// valid random q8_0 (an f16 scale, 32 int8) or mxfp4 (an e8m0 scale, 16 nibble bytes) blocks, split rows packed as the
/// kernels read them. `shape` is `[..., rows, in]`.
fn ggufTensor(p: *fixture.Pack, rng: std.Random, base: []const u8, shape: []const c_int, comptime ty: GgmlType) !void {
    const a = std.testing.allocator;
    const in: usize = @intCast(shape[shape.len - 1]);
    var rows: usize = 1;
    for (shape[0 .. shape.len - 1]) |d| rows *= @intCast(d);
    const raw_row = in / ty.blockElems() * ty.blockBytes();
    const raw = try a.alloc(u8, rows * raw_row);
    defer a.free(raw);
    rng.bytes(raw);
    var blk: usize = 0;
    while (blk < raw.len) : (blk += ty.blockBytes()) switch (ty) {
        .q8_0 => std.mem.writeInt(u16, raw[blk..][0..2], @bitCast(@as(f16, @floatCast(0.002 + 0.01 * rng.float(f32)))), .little),
        .mxfp4 => raw[blk] = 118 + rng.uintLessThan(u8, 4),
        else => unreachable,
    };
    const rb = gguf_kernels.rowBytes(ty, in);
    const packed_rows = try a.alloc(u8, rows * rb);
    defer a.free(packed_rows);
    if (gguf_kernels.isSplit(ty, in)) gguf_kernels.packRows(ty, in, packed_rows, raw) else @memcpy(packed_rows, raw);
    var sh: [4]c_int = undefined;
    @memcpy(sh[0..shape.len], shape);
    sh[shape.len - 1] = @intCast(rb);
    var nb: [192]u8 = undefined;
    const w = mlx.mlx_array_new_data(packed_rows.ptr, &sh, @intCast(shape.len), .uint8);
    defer _ = mlx.mlx_array_free(w);
    try p.put(try std.fmt.bufPrint(&nb, "{s}.weight", .{base}), w);
    const sc = try gguf_kernels.sentinel(.{ .ty = ty }, fixture.stream());
    defer _ = mlx.mlx_array_free(sc);
    try p.put(try std.fmt.bufPrint(&nb, "{s}.scales", .{base}), sc);
}

/// The weights a gpt-oss GGUF loads into: q8_0 table, head and attention, mxfp4 expert banks, the router and every
/// norm, bias and sink as bf16 (the loader's float tensors).
fn writeGgufPack(p: *fixture.Pack, seed: u64) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    var buf: [160]u8 = undefined;
    try ggufTensor(p, rng, "model.embed_tokens", &.{ VOCAB, H }, .q8_0);
    try p.norm("model.norm.weight", H);
    try ggufTensor(p, rng, "lm_head", &.{ VOCAB, H }, .q8_0);
    for (0..4) |i| {
        try p.norm(try at(&buf, i, "input_layernorm.weight"), H);
        try p.norm(try at(&buf, i, "post_attention_layernorm.weight"), H);
        inline for (.{ .{ "q_proj", HEADS * HD, H }, .{ "k_proj", KV * HD, H }, .{ "v_proj", KV * HD, H }, .{ "o_proj", H, HEADS * HD } }) |t| {
            try ggufTensor(p, rng, try at(&buf, i, "self_attn." ++ t[0]), &.{ t[1], t[2] }, .q8_0);
            try p.denseAs(try at(&buf, i, "self_attn." ++ t[0] ++ ".bias"), &.{t[1]}, 0.1, 0, .bfloat16);
        }
        try p.denseAs(try at(&buf, i, "self_attn.sinks"), &.{HEADS}, 1.0, 0, .bfloat16);
        try p.weight(try at(&buf, i, "mlp.router"), E, H, false);
        try p.denseAs(try at(&buf, i, "mlp.router.bias"), &.{E}, 0.1, 0, .bfloat16);
        inline for (.{ .{ "gate_proj", I, H }, .{ "up_proj", I, H }, .{ "down_proj", H, I } }) |t| {
            try ggufTensor(p, rng, try at(&buf, i, "mlp.experts." ++ t[0]), &.{ E, t[1], t[2] }, .mxfp4);
            try p.denseAs(try at(&buf, i, "mlp.experts." ++ t[0] ++ ".bias"), &.{ E, t[1] }, 0.1, 0, .bfloat16);
        }
    }
}

/// The config `sdk/quant/gguf/arch/gpt_oss.zig` rebuilds from a GGUF's metadata, at this pack's geometry.
const gguf_json = "{\"architectures\":[\"GptOssForCausalLM\"],\"model_type\":\"gpt_oss\",\"attention_bias\":true," ++
    "\"eos_token_id\":200002,\"experts_per_token\":2,\"head_dim\":64,\"hidden_act\":\"silu\",\"hidden_size\":128," ++
    "\"initial_context_length\":4096,\"intermediate_size\":128,\"layer_types\":[\"sliding_attention\",\"full_attention\"," ++
    "\"sliding_attention\",\"full_attention\"],\"max_position_embeddings\":131072,\"num_attention_heads\":4," ++
    "\"num_experts_per_tok\":2,\"num_hidden_layers\":4,\"num_key_value_heads\":2,\"num_local_experts\":8," ++
    "\"rms_norm_eps\":1e-05,\"rope_scaling\":{\"beta_fast\":32.0,\"beta_slow\":1.0,\"factor\":32.0," ++
    "\"original_max_position_embeddings\":4096,\"rope_type\":\"yarn\",\"truncate\":false},\"rope_theta\":150000," ++
    "\"sliding_window\":4,\"swiglu_limit\":7.0,\"tie_word_embeddings\":false,\"vocab_size\":256," ++
    "\"quantization\":{\"group_size\":32,\"bits\":4,\"mode\":\"gguf\"}}";

fn ggufCase(comptime seed: u64, comptime legacy_config: bool) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writeGgufPack(&p, seed);
    const name = std.fmt.comptimePrint("fixtures/gguf-{x}{s}.tape", .{ seed, if (legacy_config) "-lc" else "" });
    var b = try both.Both.replay(&plugin.table, &p, gguf_json, legacy_config, @embedFile(name));
    defer b.close();
    try b.expectSteps(&both.long_steps);
    try b.expectGreedy();
    try b.expectDone();
}


test "gpt_oss plugin: a --mlx-gguf pack (q8_0 attention, table and head, mxfp4 experts, bf16 router) is bit-equal to the legacy forward" {
    try ggufCase(0x9a1, false);
}

test "gpt_oss plugin: a --mlx-gguf pack is bit-equal on the legacy parse's ModelConfig" {
    try ggufCase(0x9a2, true);
}

/// A released pack's config (`fixtures/mlx_lm/`, `mlx-community_gpt-oss-20b-MXFP4-Q8`) at this proof pack's geometry, on the legacy parse's
/// ModelConfig: every key the released config adds, and its own values of the ones listed, reach both forwards.
fn realCase(comptime seed: u64) !void {
    const real = @embedFile("../../engines/mlx/plugin/fixtures/mlx_lm/mlx-community_gpt-oss-20b-MXFP4-Q8.json");
    const json = try both.atGeometry(real, configJson(true), &.{ "rope_scaling", "swiglu_limit", "max_position_embeddings", "eos_token_id" }, &.{});
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

test "gpt_oss plugin: the released gpt-oss-20b-MXFP4-Q8 config at the proof geometry is bit-equal to the legacy forward" {
    try realCase(0x90a1);
}

test "gpt_oss plugin: every shape its parse declines is refused by name, or fails the legacy parse as it always did" {
    try both.expectDeclines(configJson(true), &.{
        .{ .key = "rope_scaling", .value = "{\"type\":\"linear\",\"factor\":2.0}", .legacy = .dropped },
        .{ .key = "swiglu_clamp", .value = "7.0", .legacy = .dropped },
        .{ .key = "moe_intermediate_size", .value = "64", .legacy = .failed },
        .{ .key = "rope_parameters", .value = "{\"rope_type\":\"default\",\"rope_theta\":10000.0}", .legacy = .dropped },
        .{ .key = "rope_parameters", .value = "{\"rope_theta\":10000.0,\"partial_rotary_factor\":0.5}", .legacy = .failed },
        .{ .key = "rope_parameters", .value = "{\"full_attention\":{\"rope_type\":\"llama3\",\"rope_theta\":10000.0,\"factor\":8.0,\"low_freq_factor\":1.0,\"high_freq_factor\":4.0,\"original_max_position_embeddings\":1024}}", .legacy = .dropped },
        .{ .key = "rope_parameters", .value = "{\"full_attention\":{\"rope_type\":\"yarn\",\"rope_theta\":10000.0,\"factor\":4.0,\"original_max_position_embeddings\":1024}}", .legacy = .dropped },
        .{ .key = "hidden_size", .value = null, .legacy = .failed },
        .{ .key = "num_hidden_layers", .value = null, .legacy = .failed },
        .{ .key = "num_attention_heads", .value = null, .legacy = .failed },
        .{ .key = "num_key_value_heads", .value = null, .legacy = .failed },
        .{ .key = "head_dim", .value = null, .legacy = .failed },
        .{ .key = "sliding_window", .value = null, .legacy = .dropped },
        .{ .key = "layer_types", .value = null, .legacy = .dropped },
        .{ .key = "num_local_experts", .value = null, .legacy = .failed },
        .{ .key = "experts_per_token", .value = null, .legacy = .dropped },
        .{ .key = "swiglu_limit", .value = null, .legacy = .dropped },
        .{ .key = "layer_types", .value = "[\"sliding_attention\"]", .legacy = .dropped },
        .{ .key = "layer_types", .value = "[1,2,3,4]", .legacy = .dropped },
        .{ .key = "rms_norm_eps", .value = "\"x\"", .legacy = .dropped },
        .{ .key = "rope_theta", .value = "\"x\"", .legacy = .dropped },
        .{ .key = "query_pre_attn_scalar", .value = "0", .legacy = .failed },
        .{ .key = "query_pre_attn_scalar", .value = "\"x\"", .legacy = .parse },
        .{ .key = "head_dim", .value = "127", .legacy = .failed },
        .{ .key = "vocab_size", .value = "\"x\"", .legacy = .parse },
        .{ .key = "eos_token_id", .value = "99999999999", .legacy = .parse },
        .{ .key = "quantization", .value = "{\"group_size\":64,\"bits\":7}", .legacy = .parse },
        .{ .key = "quantization", .value = "{\"group_size\":64,\"bits\":4,\"mode\":\"bogus\"}", .legacy = .parse },
        .{ .key = "quantization", .value = "4", .legacy = .parse },
        .{ .key = "rope_scaling", .value = "{\"rope_type\":\"linear\",\"factor\":2.0}", .legacy = .dropped },
        .{ .key = "rope_scaling", .value = "{\"rope_type\":\"yarn\",\"factor\":32.0}", .legacy = .dropped },
        .{ .key = "rope_scaling", .value = "{\"factor\":32.0,\"original_max_position_embeddings\":4096}", .legacy = .dropped },
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
test "gpt_oss plugin: a pack with a DFlash sidecar serves through the plugin and drafts the legacy forward's stream" {
    const a = std.testing.allocator;
    const json = try dflashJson(a);
    defer a.free(json);
    try both.expectServedWithDrafter(json);
    var p = fixture.Pack.init(0xdf05);
    defer p.deinit();
    try writePackAt(&p, true, 64, 128);
    var b = try both.Both.openAs(&plugin.table, &p, json, true, .none);
    defer b.close();
    try both.expectDflashRecorded(&b.xfm, "applegpu_g17d", &.{ 48, 54, 99, 45, 69, 54, 54, 29, 11, 31, 54, 122, 91, 122, 120, 44 }, &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
}
