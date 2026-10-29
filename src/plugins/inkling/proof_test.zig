//! The inkling plugin against the legacy layered forward on tiny random Inkling packs: the relative bias past its
//! extent, short convs carried across chunks in the slot's conv state, sliding and global layers with log-scaling
//! past the floor, a dense layer with its gain, routed + shared experts, the embedding norm, muP and the padded vocab.
//! 4-bit, and bf16 with decode-attn-quant off and on.

const std = @import("std");
const plugin = @import("plugin.zig");
const both = @import("../../engines/mlx/plugin/both.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const transformer = @import("../../engines/mlx/transformer.zig");
const decode_attn_quant = @import("../../sdk/blocks/attention/decode_attn_quant.zig");

const H = 128;
const HD = 64;
const HEADS = 2;
const D_REL = 8;
const EXTENT = 16;
const KERNEL = 4;
const E = 8;
const NS = 2;
const MI = 64;
const INTER = 256;
const VOCAB = 256;

fn configJson(comptime quantized: bool) []const u8 {
    return "{\"model_type\":\"inkling_mm_model\",\"eos_token_id\":2,\"text_config\":{\"model_max_length\":4096," ++
        "\"hidden_size\":128,\"num_hidden_layers\":4,\"vocab_size\":256,\"unpadded_vocab_size\":250," ++
        "\"num_attention_heads\":2,\"num_key_value_heads\":1,\"head_dim\":64,\"d_rel\":8,\"rel_extent\":16," ++
        "\"log_scaling_n_floor\":8,\"log_scaling_alpha\":0.1,\"rms_norm_eps\":1e-6,\"use_embed_norm\":true," ++
        "\"local_layer_ids\":[0,1,3],\"dense_mlp_idx\":1,\"use_sconv\":true,\"sconv_kernel_size\":4," ++
        "\"logits_mup_width_multiplier\":4.0,\"sliding_window_size\":4,\"n_routed_experts\":8,\"num_experts_per_tok\":2," ++
        "\"n_shared_experts\":2,\"shared_expert_sink\":true,\"dense_intermediate_size\":256,\"intermediate_size\":64," ++
        "\"route_scale\":2.0,\"gate_activation\":\"sigmoid\"}" ++
        (if (quantized) ",\"quantization\":{\"group_size\":64,\"bits\":4}" else "") ++ "}";
}

fn at(buf: []u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "model.llm.layers.{d}.{s}", .{ layer, name });
}

fn writePack(p: *fixture.Pack, q: bool) !void {
    var buf: [160]u8 = undefined;
    try p.weight("model.llm.embed", VOCAB, H, q);
    try p.weight("model.llm.unembed", VOCAB, H, q);
    try p.norm("model.llm.embed_norm.weight", H);
    try p.norm("model.llm.norm.weight", H);
    for (0..4) |i| {
        try p.norm(try at(&buf, i, "attn_norm.weight"), H);
        try p.norm(try at(&buf, i, "mlp_norm.weight"), H);
        try p.weight(try at(&buf, i, "attn.wq_du"), HEADS * HD, H, q);
        try p.weight(try at(&buf, i, "attn.wk_dv"), HD, H, q);
        try p.weight(try at(&buf, i, "attn.wv_dv"), HD, H, q);
        try p.weight(try at(&buf, i, "attn.wr_du"), HEADS * D_REL, H, q);
        try p.weight(try at(&buf, i, "attn.wo_ud"), H, HEADS * HD, q);
        try p.norm(try at(&buf, i, "attn.q_norm.weight"), HD);
        try p.norm(try at(&buf, i, "attn.k_norm.weight"), HD);
        try p.dense(try at(&buf, i, "attn.rel_logits_proj.proj"), &.{ D_REL, EXTENT }, 0.5, 0);
        try p.dense(try at(&buf, i, "attn.k_sconv.weight"), &.{ HD, KERNEL, 1 }, 0.3, 0);
        try p.dense(try at(&buf, i, "attn.v_sconv.weight"), &.{ HD, KERNEL, 1 }, 0.3, 0);
        try p.dense(try at(&buf, i, "attn_sconv.weight"), &.{ H, KERNEL, 1 }, 0.3, 0);
        try p.dense(try at(&buf, i, "mlp_sconv.weight"), &.{ H, KERNEL, 1 }, 0.3, 0);
        if (i == 0) {
            try p.weight(try at(&buf, i, "mlp.gate_proj"), INTER, H, q);
            try p.weight(try at(&buf, i, "mlp.up_proj"), INTER, H, q);
            try p.weight(try at(&buf, i, "mlp.down_proj"), H, INTER, q);
            try p.dense(try at(&buf, i, "mlp.global_scale"), &.{1}, 0.2, 1.0);
            continue;
        }
        try p.weight(try at(&buf, i, "mlp.gate"), E + NS, H, false);
        try p.denseAs(try at(&buf, i, "mlp.gate.bias"), &.{E}, 0.1, 0, .float32);
        try p.denseAs(try at(&buf, i, "mlp.gate.global_scale"), &.{1}, 0.2, 1.0, .float32);
        for ([_][]const u8{ "mlp.experts", "mlp.shared_experts" }, [_]c_int{ E, NS }) |container, n| {
            var nb: [96]u8 = undefined;
            try p.tensor(try at(&buf, i, try std.fmt.bufPrint(&nb, "{s}.gate_proj", .{container})), &.{ n, MI, H }, q);
            try p.tensor(try at(&buf, i, try std.fmt.bufPrint(&nb, "{s}.up_proj", .{container})), &.{ n, MI, H }, q);
            try p.tensor(try at(&buf, i, try std.fmt.bufPrint(&nb, "{s}.down_proj", .{container})), &.{ n, H, MI }, q);
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

test "inkling plugin: a 4-bit pack is bit-equal to the legacy forward" {
    try case(true, 0x1a1);
}

test "inkling plugin: a bf16 pack is bit-equal to the legacy forward with decode-attn-quant off" {
    const was = decode_attn_quant.decode_attn_quant_override;
    defer decode_attn_quant.decode_attn_quant_override = was;
    decode_attn_quant.decode_attn_quant_override = false;
    try case(false, 0x1a2);
}

test "inkling plugin: a bf16 pack with decode-attn-quant on (its decode side copies) is bit-equal to the legacy forward" {
    const was = decode_attn_quant.decode_attn_quant_override;
    defer decode_attn_quant.decode_attn_quant_override = was;
    decode_attn_quant.decode_attn_quant_override = true;
    try case(false, 0x1a3);
}

test "inkling plugin: a 4-bit pack is bit-equal on the legacy parse's ModelConfig (how a served load builds the shell)" {
    try caseOn(true, 0x1a4, true);
}

// ── an in-dir DFlash sidecar ──

// The legacy forward cannot draft Inkling (its verify has no conv-state rollback: `SpecRollbackUnavailable`), and spec
// is off for it; a pack carrying a sidecar still served. The plugin binds with drafters off (no `dflash_capture`).
test "inkling plugin: a pack with a DFlash sidecar serves through the plugin, drafters off" {
    try both.expectServedWithDrafter(configJson(true));
}

// The released `mlx-community/Inkling-Small-mlx-2bit` config (`engines/mlx/plugin/fixtures/mlx_lm/`) at this proof
// pack's geometry, on the legacy parse's ModelConfig: every key the released config adds, and its own values of the
// ones listed, reach both forwards.
test "inkling plugin: the released Inkling-Small-mlx-2bit config at the proof geometry is bit-equal to the legacy forward" {
    const real = @embedFile("../../engines/mlx/plugin/fixtures/mlx_lm/mlx-community_Inkling-Small-mlx-2bit.json");
    const keep = [_][]const u8{ "route_scale", "logits_mup_width_multiplier", "log_scaling_alpha", "use_embed_norm", "gate_activation", "eos_token_id", "model_max_length" };
    const json = try both.atGeometry(real, configJson(true), &keep, &.{ "swa_head_dim", "swa_num_attention_heads", "swa_num_key_value_heads" });
    defer std.testing.allocator.free(json);
    var p = fixture.Pack.init(0x1aa2);
    defer p.deinit();
    try writePack(&p, true);
    var b = try openCase(&p, json, true, 0x1aa2);
    defer b.close();
    try b.expectSteps(&both.long_steps);
    try b.expectGreedy();
    try b.expectDone();
}

test "inkling plugin: a pack the plugin declines is refused by name (no legacy forward to fall back to)" {
    try both.expectNoLegacyFallback(configJson(true), "expert_quant");
}

test "inkling plugin: every shape its parse declines is refused by name, or fails the legacy parse as it always did" {
    try both.expectDeclines(configJson(true), &.{
        .{ .key = "expert_quant", .value = "{\"bits\":4}", .legacy = .dropped },
        .{ .key = "text_config.hidden_size", .value = null, .legacy = .failed },
        .{ .key = "text_config.num_hidden_layers", .value = null, .legacy = .failed },
        .{ .key = "text_config.num_attention_heads", .value = null, .legacy = .failed },
        .{ .key = "text_config.num_key_value_heads", .value = null, .legacy = .failed },
        .{ .key = "text_config.head_dim", .value = null, .legacy = .failed },
        .{ .key = "text_config.num_experts_per_tok", .value = null, .legacy = .dropped },
        .{ .key = "text_config.rms_norm_eps", .value = "\"x\"", .legacy = .dropped },
        .{ .key = "text_config.vocab_size", .value = "\"x\"", .legacy = .parse },
        .{ .key = "eos_token_id", .value = "99999999999", .legacy = .parse },
        .{ .key = "quantization", .value = "{\"group_size\":64,\"bits\":7}", .legacy = .parse },
        .{ .key = "quantization", .value = "{\"group_size\":64,\"bits\":4,\"mode\":\"bogus\"}", .legacy = .parse },
        .{ .key = "quantization", .value = "4", .legacy = .parse },
        .{ .key = "text_config.d_rel", .value = null, .legacy = .failed },
        .{ .key = "text_config.n_routed_experts", .value = null, .legacy = .failed },
        .{ .key = "text_config.n_shared_experts", .value = null, .legacy = .failed },
        .{ .key = "text_config.sliding_window_size", .value = null, .legacy = .dropped },
        .{ .key = "text_config.local_layer_ids", .value = null, .legacy = .dropped },
        .{ .key = "text_config.use_sconv", .value = "false", .legacy = .dropped },
        .{ .key = "text_config.shared_expert_sink", .value = "false", .legacy = .dropped },
        .{ .key = "text_config.gate_activation", .value = "\"softmax\"", .legacy = .parse },
        .{ .key = "text_config.swa_head_dim", .value = "32", .legacy = .parse },
        .{ .key = "text_config.local_layer_ids", .value = "3", .legacy = .dropped },
        .{ .key = "text_config.dense_mlp_idx", .value = "\"x\"", .legacy = .failed },
        .{ .key = "text_config.unpadded_vocab_size", .value = "\"x\"", .legacy = .dropped },
        .{ .key = "text_config.num_experts_per_tok", .value = "\"x\"", .legacy = .dropped },
    });
}
