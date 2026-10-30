//! Parity proof for the hy_v3 plugin: tiny random packs (4-bit; bf16, whose dense attention decodes through the
//! requantized side copies) with a dense first layer, experts under both container names, one layer without a
//! selection bias and an ungated shared expert, served bit-equal to the legacy `forwardMoeWith`'s values (recorded to
//! `fixtures/`) at every step, and `parse` equal to the engine's parse.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const proof = @import("../lfm2/proof.zig");
const both = @import("../../engines/mlx/plugin/both.zig");
const transformer = @import("../../engines/mlx/transformer.zig");
const hy3 = @import("plugin.zig");
const config = @import("config.zig");
const plugin_config = @import("../../engines/mlx/plugin/config.zig");
const decode_attn_quant = @import("../../sdk/blocks/attention/decode_attn_quant.zig");

const testing = std.testing;

const H: c_int = 256;
const HEADS: c_int = 4;
const KV_HEADS: c_int = 2;
const HD: c_int = 128;
const INTER: c_int = 512;
const VOCAB: c_int = 256;
const EXPERTS: c_int = 8;
const MOE_INTER: c_int = 128;
const LAYERS = 3;

const base_keys = "\"model_type\":\"hy_v3\",\"vocab_size\":256,\"hidden_size\":256,\"intermediate_size\":512," ++
    "\"num_hidden_layers\":3,\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":128," ++
    "\"rms_norm_eps\":1e-5,\"rope_theta\":11158840.0,\"max_position_embeddings\":4096,\"num_experts\":8," ++
    "\"num_experts_per_tok\":2,\"expert_hidden_dim\":128,\"num_shared_experts\":1,\"router_scaling_factor\":2.5," ++
    "\"first_k_dense_replace\":1,\"tie_word_embeddings\":false,\"eos_token_id\":[2,120020]";

pub fn configJson(comptime quantized: bool) []const u8 {
    return configAs(if (quantized) .q4 else .bf16, "");
}

/// The config of a `q` pack, `extra` keys appended.
fn configAs(comptime q: Quant, comptime extra: []const u8) []const u8 {
    return "{" ++ base_keys ++ switch (q) {
        .bf16 => "",
        .q4 => ",\"quantization\":{\"group_size\":64,\"bits\":4}",
        .q2, .ternary => ",\"quantization\":{\"group_size\":128,\"bits\":2}",
    } ++ extra ++ "}";
}

fn at(buf: []u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ layer, name });
}

pub fn writePack(p: *fixture.Pack, q: bool) !void {
    return writePackAs(p, if (q) .q4 else .bf16, true);
}

/// How a proof pack stores its projections: bf16, 4-bit group 64, 2-bit group 128, or 2-bit group 128 with every
/// bias exactly -scale (a ternary pack).
pub const Quant = enum { bf16, q4, q2, ternary };

fn tensorAs(p: *fixture.Pack, base: []const u8, shape: []const c_int, q: Quant) !void {
    return switch (q) {
        .bf16, .q4 => p.tensor(base, shape, q == .q4),
        .q2 => p.affine(base, shape, 2, 128),
        .ternary => p.ternary(base, shape, 128),
    };
}

pub fn writePackAs(p: *fixture.Pack, q: Quant, qk_norm: bool) !void {
    var buf: [160]u8 = undefined;
    var nb: [96]u8 = undefined;
    try tensorAs(p, "model.embed_tokens", &.{ VOCAB, H }, q);
    try p.norm("model.norm.weight", H);
    try tensorAs(p, "lm_head", &.{ VOCAB, H }, q);
    for (0..LAYERS) |i| {
        try p.norm(try at(&buf, i, "input_layernorm.weight"), H);
        try p.norm(try at(&buf, i, "post_attention_layernorm.weight"), H);
        try tensorAs(p, try at(&buf, i, "self_attn.q_proj"), &.{ HEADS * HD, H }, q);
        try tensorAs(p, try at(&buf, i, "self_attn.k_proj"), &.{ KV_HEADS * HD, H }, q);
        try tensorAs(p, try at(&buf, i, "self_attn.v_proj"), &.{ KV_HEADS * HD, H }, q);
        try tensorAs(p, try at(&buf, i, "self_attn.o_proj"), &.{ H, HEADS * HD }, q);
        if (qk_norm) {
            try p.norm(try at(&buf, i, "self_attn.q_norm.weight"), HD);
            try p.norm(try at(&buf, i, "self_attn.k_norm.weight"), HD);
        }
        if (i == 0) {
            try tensorAs(p, try at(&buf, i, "mlp.gate_proj"), &.{ INTER, H }, q);
            try tensorAs(p, try at(&buf, i, "mlp.up_proj"), &.{ INTER, H }, q);
            try tensorAs(p, try at(&buf, i, "mlp.down_proj"), &.{ H, INTER }, q);
            continue;
        }
        // mlx-lm's container on layer 1, the ox-ox one on layer 2; only layer 1 ships its selection bias.
        const ex = if (i == 1) "switch_mlp" else "experts";
        try tensorAs(p, try at(&buf, i, "mlp.router.gate"), &.{ EXPERTS, H }, q);
        if (i == 1) try p.denseAs(try at(&buf, i, "mlp.expert_bias"), &.{EXPERTS}, 0.3, 0, .float32);
        try tensorAs(p, try at(&buf, i, try std.fmt.bufPrint(&nb, "mlp.{s}.gate_proj", .{ex})), &.{ EXPERTS, MOE_INTER, H }, q);
        try tensorAs(p, try at(&buf, i, try std.fmt.bufPrint(&nb, "mlp.{s}.up_proj", .{ex})), &.{ EXPERTS, MOE_INTER, H }, q);
        try tensorAs(p, try at(&buf, i, try std.fmt.bufPrint(&nb, "mlp.{s}.down_proj", .{ex})), &.{ EXPERTS, H, MOE_INTER }, q);
        try tensorAs(p, try at(&buf, i, "mlp.shared_mlp.gate_proj"), &.{ MOE_INTER, H }, q);
        try tensorAs(p, try at(&buf, i, "mlp.shared_mlp.up_proj"), &.{ MOE_INTER, H }, q);
        try tensorAs(p, try at(&buf, i, "mlp.shared_mlp.down_proj"), &.{ H, MOE_INTER }, q);
    }
}

/// After `proof.steps`: a wider chunk (the sorted expert gather), then more decode.
const wide_steps = [_]proof.Step{
    .{ .mode = .prefill, .ids = &.{ 3, 17, 42, 99, 7, 250, 1, 8, 66, 31, 120, 5, 77, 200, 9, 33, 77, 140, 2, 18, 19 } },
    .{ .mode = .decode, .ids = &.{88} },
    .{ .mode = .verify, .ids = &.{ 5, 61, 200 } },
    .{ .mode = .decode, .ids = &.{64} },
};

/// The legacy forward's values on the legacy parse's config, recorded to `fixtures/<seed>.tape`.
fn open(p: *fixture.Pack, json: []const u8, comptime seed: u64) !both.Both {
    const name = std.fmt.comptimePrint("fixtures/{x}.tape", .{seed});
    return both.Both.replay(&hy3.table, p, json, true, @embedFile(name));
}

fn case(comptime quantized: bool, comptime seed: u64) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writePack(&p, quantized);
    var b = try open(&p, configJson(quantized), seed);
    defer b.close();
    try b.expectSteps(&(proof.steps ++ wide_steps));
    try b.expectDone();
}

/// Served: the driver's shell on the legacy parse's config, steps past a chunk and the pipelined greedy decode.
fn servedCase(comptime quantized: bool, comptime seed: u64) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writePack(&p, quantized);
    var b = try open(&p, configJson(quantized), seed);
    defer b.close();
    try b.expectSteps(&both.long_steps);
    try b.expectGreedy();
    try b.expectDone();
}

/// A shape the plugin once declined (`extra` joins the config's keys) binds to the plugin and is served bit-equal on
/// the legacy config: steps past a chunk, greedy decode, and image rows spliced at token 250 under a `vision_config`.
fn shapeCase(comptime q: Quant, qk_norm: bool, comptime extra: []const u8, comptime seed: u64) !void {
    const json = configAs(q, extra);
    try expectBinds(json);
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writePackAs(&p, q, qk_norm);
    var b = try open(&p, json, seed);
    defer b.close();
    // The 2-bit arm: qmv2's kernels for a ternary pack, stock qmm otherwise.
    const m: *const hy3.Loaded = @ptrCast(@alignCast(b.xfm.plugin.?.model));
    try testing.expectEqual(q == .ternary, m.layers[0].mixer.q.lin.ternary and m.base.weights.head.ternary);
    try b.expectSteps(&both.long_steps);
    try b.expectGreedy();
    if (std.mem.indexOf(u8, extra, "vision_config") != null) try b.expectMedia(250, H);
    try b.expectDone();
}

fn expectBinds(json: []const u8) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
    var path_buf: [512]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const bound = (try plugin_config.bindServed(io, a, dir, json)) orelse return error.TestPackNotServed;
    bound.destroy(a);
}

fn expectParse(json: []const u8) !void {
    var want = try proof.legacyConfig(json);
    defer want.deinit(testing.allocator);
    var p: abi.Parsed = .{};
    try testing.expect(config.parseJson(json, &p).isOk());
    defer config.freeParsed(&p);
    try proof.expectShell(&p.shell, &want);
    const c = config.of(&p);
    try testing.expectEqual(want.rms_norm_eps, c.rms_norm_eps);
    try testing.expectEqual(want.rope_theta, c.rope_theta);
    try testing.expectEqual(want.partial_rotary_factor, c.partial_rotary_factor);
    try testing.expectEqual(want.query_pre_attn_scalar, c.query_pre_attn_scalar);
    try testing.expectEqual(want.tie_word_embeddings, c.tie_word_embeddings);
    try testing.expectEqual(want.num_experts, c.experts);
    try testing.expectEqual(want.num_experts_per_tok, c.top_k);
    try testing.expectEqual(want.moe_intermediate_size, c.moe_intermediate);
    try testing.expectEqual(want.shared_expert_intermediate_size, c.shared_intermediate);
    try testing.expectEqual(want.moe_route_norm, c.route_norm);
    try testing.expectEqual(want.router_scaling_factor, c.route_scale);
    try testing.expectEqual(want.first_k_dense_replace, c.first_k_dense);
    try testing.expectEqual(want.isMoe(), p.caps.moe);
}

// ── tests ──

test "hy_v3 plugin: a 4-bit pack is bit-equal to forwardMoeWith at every step" {
    try case(true, 0x4a3);
}

test "hy_v3 plugin: a bf16 pack (decode reads the attention side copies) is bit-equal" {
    try case(false, 0x4a4);
}

test "hy_v3 plugin: parse equals the engine's parse on the shell and the plugin's own fields" {
    try expectParse(configJson(true));
    try expectParse(config.hy3_295b);
}

test "hy_v3 served: a 4-bit pack through the driver on the legacy config is bit-equal, greedy decode too" {
    try servedCase(true, 0x4a5);
}

test "hy_v3 served: a bf16 pack with decode-attn-quant on (its side copies) is bit-equal" {
    const was = decode_attn_quant.decode_attn_quant_override;
    defer decode_attn_quant.decode_attn_quant_override = was;
    decode_attn_quant.decode_attn_quant_override = true;
    try servedCase(false, 0x4a6);
}

test "hy_v3 served: lm_head_mxfp8 (LOSSY) quantizes a bf16 pack's head as the legacy load does, steps equal" {
    const cli = &transformer.cli().lm_head_mxfp8_cli;
    const was = cli.*;
    defer cli.* = was;
    cli.* = true;
    var p = fixture.Pack.init(0x4a7);
    defer p.deinit();
    try writePack(&p, false);
    var b = try both.Both.replay(&hy3.table, &p, configJson(false), true, @embedFile("fixtures/mxfp8-4a7.tape"));
    defer b.close();
    try testing.expect(b.xfm.lm_head_s.ctx != null);
    try b.expectSteps(&both.long_steps);
    try b.expectGreedy();
    try b.expectDone();
}

test "hy_v3 served: a 2-bit group-128 pack with mlx-lm's quantization_config copy (stock qmm) is bit-equal" {
    try shapeCase(.q2, true, ",\"quantization_config\":{\"group_size\":128,\"bits\":2}", 0x4b1);
}

test "hy_v3 served: a ternary 2-bit pack (every bias -scale) reads qmv2's kernels on both sides, bit-equal" {
    try shapeCase(.ternary, true, "", 0x4b2);
}

test "hy_v3 served: a pack without q/k norms (qk_norm false) is bit-equal" {
    try shapeCase(.q4, false, ",\"qk_norm\":false", 0x4b3);
}

test "hy_v3 served: a vision_config splices image rows as the engine does" {
    try shapeCase(.q4, true, ",\"vision_config\":{},\"image_token_id\":250", 0x4b4);
}
