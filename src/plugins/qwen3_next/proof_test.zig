//! Parity proof for the qwen3_next plugin: the table's forward over its own load of a tiny random Qwen3-Next pack
//! (combined q|k|v|z and b|a projections, two GDN + two gated attention layers, routed experts with a gated shared
//! expert) is bit-equal to the legacy layered forward on logits and every recurrent state and verify capture.

const std = @import("std");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const twin = @import("../qwen3/twin_test.zig");
const qwen3_next = @import("plugin.zig");

/// A tiny Qwen3-Next pack: 4 layers, GDN at even layers, gated attention at odd ones, routed experts with a gated
/// shared expert.
pub const Geo = struct {
    hidden: c_int = 256,
    vocab: c_int = 256,
    heads: c_int = 4,
    kv_heads: c_int = 2,
    head_dim: c_int = 256,
    hk: c_int = 2,
    hv: c_int = 8,
    dk: c_int = 128,
    dv: c_int = 128,
    experts: c_int = 8,
    moe_inter: c_int = 128,
    quantized: bool = true,
};

pub fn config(a: std.mem.Allocator, g: Geo) ![]u8 {
    return std.fmt.allocPrint(a, "{{\"model_type\":\"qwen3_next\",\"hidden_size\":{d},\"intermediate_size\":512,\"num_hidden_layers\":4," ++
        "\"num_attention_heads\":{d},\"num_key_value_heads\":{d},\"head_dim\":{d},\"vocab_size\":{d},\"rms_norm_eps\":1e-6," ++
        "\"rope_theta\":100000.0,\"partial_rotary_factor\":0.25,\"full_attention_interval\":2,\"linear_num_key_heads\":{d}," ++
        "\"linear_num_value_heads\":{d},\"linear_key_head_dim\":{d},\"linear_value_head_dim\":{d},\"linear_conv_kernel_dim\":4," ++
        "\"num_experts\":{d},\"num_experts_per_tok\":2,\"moe_intermediate_size\":{d},\"shared_expert_intermediate_size\":{d}," ++
        "\"max_position_embeddings\":4096,\"hidden_act\":\"silu\",\"tie_word_embeddings\":false{s}}}", .{
        g.hidden,                                                                    g.heads, g.kv_heads, g.head_dim, g.vocab, g.hk, g.hv, g.dk, g.dv, g.experts, g.moe_inter, g.moe_inter,
        if (g.quantized) ",\"quantization\":{\"group_size\":64,\"bits\":4}" else "",
    });
}

fn at(buf: []u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ layer, name });
}

pub fn writePack(p: *fixture.Pack, g: Geo) !void {
    var buf: [160]u8 = undefined;
    const q = g.quantized;
    const H = g.hidden;
    const conv = 2 * g.hk * g.dk + g.hv * g.dv;
    try p.weight("model.embed_tokens", g.vocab, H, q);
    try p.norm("model.norm.weight", H);
    try p.weight("lm_head", g.vocab, H, q);
    for (0..4) |i| {
        try p.norm(try at(&buf, i, "input_layernorm.weight"), H);
        try p.norm(try at(&buf, i, "post_attention_layernorm.weight"), H);
        try p.weight(try at(&buf, i, "mlp.gate"), g.experts, H, q);
        try p.tensor(try at(&buf, i, "mlp.switch_mlp.gate_proj"), &.{ g.experts, g.moe_inter, H }, q);
        try p.tensor(try at(&buf, i, "mlp.switch_mlp.up_proj"), &.{ g.experts, g.moe_inter, H }, q);
        try p.tensor(try at(&buf, i, "mlp.switch_mlp.down_proj"), &.{ g.experts, H, g.moe_inter }, q);
        try p.weight(try at(&buf, i, "mlp.shared_expert.gate_proj"), g.moe_inter, H, q);
        try p.weight(try at(&buf, i, "mlp.shared_expert.up_proj"), g.moe_inter, H, q);
        try p.weight(try at(&buf, i, "mlp.shared_expert.down_proj"), H, g.moe_inter, q);
        try p.weight(try at(&buf, i, "mlp.shared_expert_gate"), 1, H, false);
        if (i % 2 == 0) {
            try p.weight(try at(&buf, i, "linear_attn.in_proj_qkvz"), conv + g.hv * g.dv, H, q);
            try p.weight(try at(&buf, i, "linear_attn.in_proj_ba"), 2 * g.hv, H, q);
            try p.dense(try at(&buf, i, "linear_attn.conv1d.weight"), &.{ conv, 4, 1 }, 0.5, 0);
            try p.dense(try at(&buf, i, "linear_attn.A_log"), &.{g.hv}, 0.5, 0.2);
            try p.dense(try at(&buf, i, "linear_attn.dt_bias"), &.{g.hv}, 0.5, 0);
            try p.norm(try at(&buf, i, "linear_attn.norm.weight"), g.dv);
            try p.weight(try at(&buf, i, "linear_attn.out_proj"), H, g.hv * g.dv, q);
        } else {
            try p.weight(try at(&buf, i, "self_attn.q_proj"), 2 * g.heads * g.head_dim, H, q);
            try p.weight(try at(&buf, i, "self_attn.k_proj"), g.kv_heads * g.head_dim, H, q);
            try p.weight(try at(&buf, i, "self_attn.v_proj"), g.kv_heads * g.head_dim, H, q);
            try p.weight(try at(&buf, i, "self_attn.o_proj"), H, g.heads * g.head_dim, q);
            try p.norm(try at(&buf, i, "self_attn.q_norm.weight"), g.head_dim);
            try p.norm(try at(&buf, i, "self_attn.k_norm.weight"), g.head_dim);
        }
    }
}

const steps = [_]twin.Step{
    .{ .mode = .prefill, .ids = &twin.seq(20, 3) },
    .{ .mode = .prefill, .ids = &twin.seq(70, 11) },
    .{ .mode = .decode, .ids = &.{88} },
    .{ .mode = .verify, .ids = &.{ 5, 61, 200 } },
    .{ .mode = .decode, .ids = &.{12} },
    .{ .mode = .verify, .ids = &.{ 9, 33, 77, 140, 2 } },
    .{ .mode = .decode, .ids = &.{64} },
    .{ .mode = .prefill, .ids = &twin.seq(6, 5) },
    .{ .mode = .decode, .ids = &.{17} },
};

fn case(comptime quantized: bool, comptime seed: u64) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    const g: Geo = .{ .quantized = quantized };
    try writePack(&p, g);
    const json = try config(std.testing.allocator, g);
    defer std.testing.allocator.free(json);
    const T = twin.Twin(qwen3_next);
    const name = std.fmt.comptimePrint("fixtures/twin-{x}.tape", .{seed});
    const t = try T.open(&p, json, &.{}, @embedFile(name));
    defer t.close();
    const l = try t.lane();
    defer l.deinit();
    for (steps) |st| try t.step(l, st);
    try t.expectDone();
}

test "qwen3_next plugin: a 4-bit pack through the table is bit-equal to the legacy layered forward" {
    try case(true, 0x3a01);
}

test "qwen3_next plugin: a bf16 pack (decode attention side copies) is bit-equal" {
    try case(false, 0x3a02);
}

test "qwen3_next served: steps, captured verifies, rollbacks and the lookup loop through the driver equal the legacy shell" {
    const served = @import("../qwen3_5/served_test.zig");
    var pk = fixture.Pack.init(0x3a03);
    defer pk.deinit();
    try writePack(&pk, .{});
    const json = try config(std.testing.allocator, .{});
    defer std.testing.allocator.free(json);
    var p: served.Pair = undefined;
    try p.open(&qwen3_next.table, &pk, json, try served.oracle("qwen3_next", "served-3a03"));
    defer p.close();
    try served.expectSolo(&p);
}

test "qwen3_next served: a DFlash drafter on the served trunk reads its embedding and head and emits the legacy stream" {
    const a = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const served = @import("../qwen3_5/served_test.zig");
    const dflash_mod = @import("../../engines/mlx/spec/dflash/dflash.zig");
    // The tiny assistant's geometry (`dflash.TinyFix`): hidden 64, vocab 128, targets layers 0 and 2 (both GDN here).
    const g: Geo = .{ .hidden = 64, .vocab = 128, .head_dim = 64, .hv = 4, .dk = 64, .dv = 64, .moe_inter = 64 };
    var pk = fixture.Pack.init(0x3a04);
    defer pk.deinit();
    try writePack(&pk, g);
    const json = try config(a, g);
    defer a.free(json);
    var p: served.Pair = undefined;
    try p.open(&qwen3_next.table, &pk, json, try served.oracle("qwen3_next", "dflash-3a04"));
    defer p.close();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const asst = buf[0..try tmp.dir.realPath(io, &buf)];
    try dflash_mod.TinyFix.writeAssistant(io, tmp.dir, asst, fixture.stream());
    try served.expectDflash(&p, asst, 64, "lmHeadForDraft");
}
