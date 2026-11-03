//! Parity proof for Sushi's MiMo packs on the mimo_v2 plugin: a tiny pack in Sushi's source layout (the fused e4m3
//! `qkv_proj` with its 128x128 tile scales over four tensor-parallel slabs, an e4m3 dense first MLP, EXL3 routed
//! experts under `expert_quant`, a bf16 rest) loaded by the engine's loader (`sushi_pack.adapt` splits the QKV into
//! row-scaled e4m3 parts and decodes the MLP), served by the table bit-equal to the legacy forward at every step, and
//! through the driver on the legacy config: steps, greedy and batched decode. The legacy forward's values are replayed
//! from `fixtures/*.tape`.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const exl3 = @import("../../sdk/quant/exl3/format.zig");
const sushi_pack = @import("../../sdk/quant/exl3/sushi_pack.zig");
const trunk = @import("proof_test.zig");
const both = @import("../../engines/mlx/plugin/both.zig");
const mimo = @import("plugin.zig");
const heads_test = @import("mtp_head_test.zig");

const testing = std.testing;

const H: c_int = 256;
const HEADS: c_int = 8;
const KV: c_int = 2;
const HD: c_int = 192;
const VD: c_int = 128;
const INTER: c_int = 512;
const VOCAB: c_int = 256;
const EXPERTS: c_int = 8;
const MOE_INTER: c_int = 128;
const LAYERS = 4;
const TP = 4;
/// Halfwords per 16x16 tile at k = 3.
const TILE_WORDS: c_int = 48;

const config_json =
    "{\"model_type\":\"mimo_v2\",\"vocab_size\":256,\"hidden_size\":256,\"intermediate_size\":512," ++
    "\"num_hidden_layers\":4,\"num_attention_heads\":8,\"num_key_value_heads\":2,\"swa_num_key_value_heads\":2," ++
    "\"head_dim\":192,\"v_head_dim\":128,\"swa_head_dim\":192,\"swa_v_head_dim\":128,\"swa_num_attention_heads\":8," ++
    "\"layernorm_epsilon\":1e-6,\"rope_theta\":10000000.0,\"swa_rope_theta\":10000.0," ++
    "\"rope_parameters\":{\"partial_rotary_factor\":0.334,\"rope_theta\":10000000.0,\"rope_type\":\"default\"}," ++
    "\"sliding_window\":8,\"hybrid_layer_pattern\":[0,1,1,0],\"moe_layer_freq\":[0,1,1,1]," ++
    "\"max_position_embeddings\":4096,\"n_routed_experts\":8,\"num_experts_per_tok\":2,\"moe_intermediate_size\":128," ++
    "\"n_shared_experts\":null,\"norm_topk_prob\":true,\"routed_scaling_factor\":null,\"n_group\":1,\"topk_group\":1," ++
    "\"attention_value_scale\":0.707,\"tie_word_embeddings\":false,\"eos_token_id\":[2,151645]," ++
    "\"quantization_config\":{\"quant_method\":\"fp8\",\"weight_block_size\":[128,128]}," ++
    "\"expert_quant\":{\"format\":\"exl3\",\"k\":3,\"codebook\":\"mcg\"}}";

fn at(buf: []u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ layer, name });
}

fn put(p: *fixture.Pack, name: []const u8, ptr: *const anyopaque, shape: []const c_int, dt: mlx.mlx_dtype) !void {
    const a = mlx.mlx_array_new_data(ptr, shape.ptr, @intCast(shape.len), dt);
    defer _ = mlx.mlx_array_free(a);
    try p.put(name, a);
}

/// e4m3 codes `[rows, cols]` below 1.0 in magnitude, either sign, and their f32 scales per 128x128 tile.
fn fp8(p: *fixture.Pack, base: []const u8, rows: c_int, cols: c_int, tile_rows: c_int) !void {
    const a = testing.allocator;
    var nb: [160]u8 = undefined;
    const codes = try a.alloc(u8, @intCast(rows * cols));
    defer a.free(codes);
    const r = p.rnd.random();
    for (codes) |*c| c.* = r.uintLessThan(u8, 0x38) | (if (r.boolean()) @as(u8, 0x80) else 0);
    try put(p, try std.fmt.bufPrint(&nb, "{s}.weight", .{base}), codes.ptr, &.{ rows, cols }, .uint8);
    const scales = try a.alloc(f32, @intCast(tile_rows * @divExact(cols, 128)));
    defer a.free(scales);
    for (scales) |*s| s.* = 0.02 + r.float(f32) * 0.04;
    try put(p, try std.fmt.bufPrint(&nb, "{s}.weight_scale_inv", .{base}), scales.ptr, &.{ tile_rows, @divExact(cols, 128) }, .float32);
}

/// One EXL3 bank `<base>.{trellis,suh,svh}` over `in` -> `out`.
fn bank(p: *fixture.Pack, base: []const u8, in: c_int, out: c_int) !void {
    const a = testing.allocator;
    var nb: [160]u8 = undefined;
    const r = p.rnd.random();
    const words = try a.alloc(u16, @intCast(EXPERTS * @divExact(in, 16) * @divExact(out, 16) * TILE_WORDS));
    defer a.free(words);
    for (words) |*v| v.* = r.int(u16);
    try put(p, try std.fmt.bufPrint(&nb, "{s}.trellis", .{base}), words.ptr, &.{ EXPERTS, @divExact(in, 16), @divExact(out, 16), TILE_WORDS }, .uint16);
    for ([_]struct { leaf: []const u8, dim: c_int }{ .{ .leaf = "suh", .dim = in }, .{ .leaf = "svh", .dim = out } }) |side| {
        const h = try a.alloc(u16, @intCast(EXPERTS * side.dim));
        defer a.free(h);
        for (h) |*v| v.* = exl3.sushi.format.f32ToF16Bits(if (r.boolean()) 0.125 else -0.125);
        try put(p, try std.fmt.bufPrint(&nb, "{s}.{s}", .{ base, side.leaf }), h.ptr, &.{ EXPERTS, side.dim }, .float16);
    }
}

fn writePack(p: *fixture.Pack) !void {
    var buf: [160]u8 = undefined;
    try p.weight("model.embed_tokens", VOCAB, H, false);
    try p.norm("model.norm.weight", H);
    try p.weight("lm_head", VOCAB, H, false);
    const qkv_rows = HEADS * HD + KV * HD + KV * VD;
    for (0..LAYERS) |i| {
        try p.norm(try at(&buf, i, "input_layernorm.weight"), H);
        try p.norm(try at(&buf, i, "post_attention_layernorm.weight"), H);
        // Four slabs of 544 rows: five tile rows each.
        try fp8(p, try at(&buf, i, "self_attn.qkv_proj"), qkv_rows, H, TP * 5);
        try p.weight(try at(&buf, i, "self_attn.o_proj"), H, HEADS * VD, false);
        if (i == 1 or i == 2) try p.dense(try at(&buf, i, "self_attn.attention_sink_bias"), &.{HEADS}, 0.5, 0);
        if (i == 0) {
            try fp8(p, try at(&buf, i, "mlp.gate_proj"), INTER, H, 4);
            try fp8(p, try at(&buf, i, "mlp.up_proj"), INTER, H, 4);
            try fp8(p, try at(&buf, i, "mlp.down_proj"), H, INTER, 2);
            continue;
        }
        try p.dense(try at(&buf, i, "mlp.gate.weight"), &.{ EXPERTS, H }, 0.1, 0);
        try p.denseAs(try at(&buf, i, "mlp.gate.e_score_correction_bias"), &.{EXPERTS}, 0.3, 0, .float32);
        try bank(p, try at(&buf, i, "mlp.switch_mlp.gate_proj"), H, MOE_INTER);
        try bank(p, try at(&buf, i, "mlp.switch_mlp.up_proj"), H, MOE_INTER);
        try bank(p, try at(&buf, i, "mlp.switch_mlp.down_proj"), MOE_INTER, H);
    }
}

/// Three MTP heads stored as the trunk: fused e4m3 QKV (a sliding layer's geometry) and an e4m3 MLP.
fn writeHeads(p: *fixture.Pack) !void {
    var b: [160]u8 = undefined;
    for (0..3) |k| {
        const n = struct {
            fn f(buf: []u8, kk: usize, leaf: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, "model.mtp.layers.{d}.{s}", .{ kk, leaf });
            }
        }.f;
        for ([_][]const u8{ "enorm.weight", "hnorm.weight", "input_layernorm.weight", "pre_mlp_layernorm.weight", "final_layernorm.weight" }) |leaf| try p.norm(try n(&b, k, leaf), H);
        try p.dense(try n(&b, k, "self_attn.attention_sink_bias"), &.{HEADS}, 0.5, 0);
        try p.weight(try n(&b, k, "eh_proj"), H, 2 * H, false);
        try fp8(p, try n(&b, k, "self_attn.qkv_proj"), HEADS * HD + KV * HD + KV * VD, H, TP * 5);
        try p.weight(try n(&b, k, "self_attn.o_proj"), H, HEADS * VD, false);
        try fp8(p, try n(&b, k, "mlp.gate_proj"), INTER, H, 4);
        try fp8(p, try n(&b, k, "mlp.up_proj"), INTER, H, 4);
        try fp8(p, try n(&b, k, "mlp.down_proj"), H, INTER, 2);
    }
}

// ── tests ──

test "mimo_v2 Sushi pack: e4m3 QKV parts, the decoded dense MLP and EXL3 experts are bit-equal to the legacy forward" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var p = fixture.Pack.init(0x5a51);
    defer p.deinit();
    try writePack(&p);
    var b = try trunk.openCase(&p, config_json, true, 0x5a51);
    defer b.close();
    try b.expectSteps(&both.steps);
    try b.expectDone();
}

test "mimo_v2 Sushi served: steps, greedy and batched decode through the driver on the legacy config, the MTP heads too" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var p = fixture.Pack.init(0x5a52);
    defer p.deinit();
    try writePack(&p);
    try writeHeads(&p);
    const json = "{\"num_nextn_predict_layers\":3," ++ config_json[1..];
    var b = try trunk.openCase(&p, json, true, 0x5a52);
    defer b.close();
    try b.expectSteps(&both.long_steps);
    try b.expectGreedy();
    try b.expectBatched();
    try heads_test.expectServedHeads(&b, 0x5a53);
    try b.expectDone();
}

test "mimo_v2 resident bill: a Sushi pack holds QKV as stored, the dense MLP decoded, the MTP heads, nothing unserved" {
    const header =
        \\{"__metadata__":{"format":"pt"},
        \\"model.layers.0.self_attn.qkv_proj.weight":{"dtype":"F8_E4M3","shape":[1,1],"data_offsets":[0,1000]},
        \\"model.layers.0.self_attn.qkv_proj.weight_scale_inv":{"dtype":"F32","shape":[1,1],"data_offsets":[1000,1100]},
        \\"model.layers.0.mlp.gate_proj.weight":{"dtype":"F8_E4M3","shape":[1,1],"data_offsets":[1100,1300]},
        \\"model.layers.0.mlp.gate_proj.weight_scale_inv":{"dtype":"F32","shape":[1,1],"data_offsets":[1300,1400]},
        \\"model.layers.0.mlp.switch_mlp.up_proj.trellis":{"dtype":"U16","shape":[1,1],"data_offsets":[1400,1900]},
        \\"model.mtp.layers.0.eh_proj.weight":{"dtype":"BF16","shape":[1,1],"data_offsets":[1900,2900]},
        \\"visual.blocks.0.attn.proj.weight":{"dtype":"BF16","shape":[1,1],"data_offsets":[2900,3900]}}
    ;
    // qkv 1000 + its scales 100 x 128 + gate 200 held twice + trellis 500 + the MTP head's 1000.
    const got = try sushi_pack.headerResidentBytes(testing.allocator, header, "a.safetensors", null, mimo.SushiBill{ .layers = 48 });
    try testing.expectEqual(@as(u64, 1000 + 100 * 128 + 400 + 500 + 1000), got);
}
