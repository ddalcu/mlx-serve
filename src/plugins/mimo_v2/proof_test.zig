//! Parity proof for the mimo_v2 plugin: tiny random packs (4-bit; bf16, whose dense attention decodes and verifies
//! through the requantized side copies) at MiMo's own head shape (K 192 / V 128, RoPE on 64 channels, V scaled), a
//! dense global first layer, sliding layers with sinks under an 8-row band, a global MoE layer, the router under both
//! of its names and one layer without a selection bias, served by the table bit-equal to the legacy `forwardMoeWith`
//! at every step (prefill chunks past the window and the fused prefill width, decode, speculative verify), and
//! `parse` equal to the engine's parse. The deleted legacy forward's values are replayed from `fixtures/*.tape`.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const cache_ops = @import("../../engines/mlx/kv/cache_ops.zig");
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const proof = @import("../lfm2/proof.zig");
const both = @import("../../engines/mlx/plugin/both.zig");
const mimo = @import("plugin.zig");
const config = @import("config.zig");

const testing = std.testing;

pub const H: c_int = 256;
const HEADS: c_int = 8;
const KV_GLOBAL: c_int = 1;
const KV_SWA: c_int = 2;
const HD: c_int = 192;
const VD: c_int = 128;
const INTER: c_int = 512;
pub const VOCAB: c_int = 256;
pub const EXPERTS: c_int = 8;
pub const MOE_INTER: c_int = 128;
const LAYERS = 4;
/// 0 = global, 1 = sliding.
const pattern = [LAYERS]u8{ 0, 1, 1, 0 };

pub fn configJson(comptime quantized: bool) []const u8 {
    return "{\"model_type\":\"mimo_v2\",\"vocab_size\":256,\"hidden_size\":256,\"intermediate_size\":512," ++
        "\"num_hidden_layers\":4,\"num_attention_heads\":8,\"num_key_value_heads\":1,\"swa_num_key_value_heads\":2," ++
        "\"head_dim\":192,\"v_head_dim\":128,\"swa_head_dim\":192,\"swa_v_head_dim\":128,\"swa_num_attention_heads\":8," ++
        "\"layernorm_epsilon\":1e-6,\"rope_theta\":10000000.0,\"swa_rope_theta\":10000.0," ++
        "\"rope_parameters\":{\"partial_rotary_factor\":0.334,\"rope_theta\":10000000.0,\"rope_type\":\"default\"}," ++
        "\"sliding_window\":8,\"hybrid_layer_pattern\":[0,1,1,0],\"moe_layer_freq\":[0,1,1,1]," ++
        "\"max_position_embeddings\":4096,\"n_routed_experts\":8,\"num_experts_per_tok\":2,\"moe_intermediate_size\":128," ++
        "\"n_shared_experts\":null,\"norm_topk_prob\":true,\"routed_scaling_factor\":null,\"n_group\":1,\"topk_group\":1," ++
        "\"attention_value_scale\":0.707,\"tie_word_embeddings\":false,\"eos_token_id\":[2,151645]" ++
        (if (quantized) ",\"quantization\":{\"group_size\":64,\"bits\":4}}" else "}");
}

fn at(buf: []u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ layer, name });
}

pub fn writePack(p: *fixture.Pack, q: bool) !void {
    var buf: [160]u8 = undefined;
    try p.weight("model.embed_tokens", VOCAB, H, q);
    try p.norm("model.norm.weight", H);
    try p.weight("lm_head", VOCAB, H, q);
    for (0..LAYERS) |i| {
        const kvh: c_int = if (pattern[i] == 0) KV_GLOBAL else KV_SWA;
        try p.norm(try at(&buf, i, "input_layernorm.weight"), H);
        try p.norm(try at(&buf, i, "post_attention_layernorm.weight"), H);
        try p.weight(try at(&buf, i, "self_attn.q_proj"), HEADS * HD, H, q);
        try p.weight(try at(&buf, i, "self_attn.k_proj"), kvh * HD, H, q);
        try p.weight(try at(&buf, i, "self_attn.v_proj"), kvh * VD, H, q);
        try p.weight(try at(&buf, i, "self_attn.o_proj"), H, HEADS * VD, q);
        try p.dense(try at(&buf, i, "self_attn.q_proj.bias"), &.{HEADS * HD}, 0.1, 0);
        try p.dense(try at(&buf, i, "self_attn.k_proj.bias"), &.{kvh * HD}, 0.1, 0);
        try p.dense(try at(&buf, i, "self_attn.v_proj.bias"), &.{kvh * VD}, 0.1, 0);
        // Sinks on the sliding layers (`add_swa_attention_sink_bias`), under both of their names.
        if (i == 1) try p.dense(try at(&buf, i, "self_attn.attention_sink_bias"), &.{HEADS}, 0.5, 0);
        if (i == 2) try p.dense(try at(&buf, i, "self_attn.sinks"), &.{HEADS}, 0.5, 0);
        if (i == 0) {
            try p.weight(try at(&buf, i, "mlp.gate_proj"), INTER, H, q);
            try p.weight(try at(&buf, i, "mlp.up_proj"), INTER, H, q);
            try p.weight(try at(&buf, i, "mlp.down_proj"), H, INTER, q);
            continue;
        }
        // The router as `mlp.gate` (bf16, widened to f32 at load) and as HF-native `mlp.gate.proj`; layer 3 has no bias.
        try p.dense(try at(&buf, i, if (i == 2) "mlp.gate.proj.weight" else "mlp.gate.weight"), &.{ EXPERTS, H }, 0.1, 0);
        if (i != 3) try p.denseAs(try at(&buf, i, "mlp.gate.e_score_correction_bias"), &.{EXPERTS}, 0.3, 0, .float32);
        try p.tensor(try at(&buf, i, "mlp.switch_mlp.gate_proj"), &.{ EXPERTS, MOE_INTER, H }, q);
        try p.tensor(try at(&buf, i, "mlp.switch_mlp.up_proj"), &.{ EXPERTS, MOE_INTER, H }, q);
        try p.tensor(try at(&buf, i, "mlp.switch_mlp.down_proj"), &.{ EXPERTS, H, MOE_INTER }, q);
    }
}

/// After `proof.steps`: a chunk at the fused prefill width, decode past the window, an 8-row verify (the split-K
/// kernel's widest), a 9-row verify (past the side copies), then decode.
const wide_steps = [_]proof.Step{
    .{ .mode = .prefill, .ids = &.{ 3, 17, 42, 99, 7, 250, 1, 8, 66, 31, 120, 5, 77, 200, 9, 33, 77, 140, 2, 18, 19 } },
    .{ .mode = .decode, .ids = &.{88} },
    .{ .mode = .verify, .ids = &.{ 5, 61, 200, 4, 11, 90, 13, 7 } },
    .{ .mode = .verify, .ids = &.{ 6, 62, 201, 5, 12, 91, 14, 8, 21 } },
    .{ .mode = .decode, .ids = &.{64} },
};

/// `p` served by the table against the tape `fixtures/<seed>.tape`, recorded from the deleted legacy forward.
pub fn openCase(p: *fixture.Pack, json: []const u8, comptime legacy_config: bool, comptime seed: u64) !both.Both {
    return both.Both.replay(&mimo.table, p, json, legacy_config, @embedFile(std.fmt.comptimePrint("fixtures/{x}.tape", .{seed})));
}

fn case(comptime quantized: bool, comptime seed: u64) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writePack(&p, quantized);
    var b = try openCase(&p, configJson(quantized), false, seed);
    defer b.close();
    try b.expectSteps(&(proof.steps ++ wide_steps));
    // A drafter reads the trunk's table and head as loaded, with their geometry.
    for ([_]abi.TrunkWeight{ .embed, .head }) |which| {
        var w: @import("../../abi/quant.zig").Weight = .{};
        try testing.expect(mimo.table.spec.?.trunk_weight.?(b.xfm.plugin.?.model, which, &w).isOk());
        try testing.expect(w.w.ctx != null and (w.scales.ctx != null) == quantized);
        try testing.expectEqual(@as(u8, if (quantized) 4 else 0), w.bits);
    }
    try b.expectDone();
}

/// One slot of the table: its KV cache and position.
pub const Side = struct {
    kv: KVCache,
    off: usize = 0,
    slot: cache_ops.Slot = undefined,

    pub fn init(s: *Side) !void {
        s.* = .{ .kv = try KVCache.init(testing.allocator, LAYERS) };
        s.slot = .{ .kv = &s.kv };
    }

    pub fn deinit(s: *Side) void {
        s.kv.deinit();
    }
};

/// The table's logits for one forward, owned by `g`.
pub fn tableLogits(g: *ops.Graph, m: *anyopaque, f: abi.Forward) !mlx.mlx_array {
    var out: abi.ForwardOut = .{};
    var fw = f;
    fw.logits = .all_rows;
    fw.cache_ops = &cache_ops.ops;
    const v = mimo.table.forward(m, &fw, &out);
    if (!v.isOk()) {
        std.debug.print("forward: {s}\n", .{v.why.slice()});
        return error.PluginForward;
    }
    if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
    return g.adopt(.{ .ctx = out.logits.ctx });
}

/// Two slots of different lengths (one past the window) prefilled on their own, then batched decode ticks and a
/// batched verify: the table's batched rows bit-equal to the recorded legacy `forwardMoeBatchedRows`.
fn batchedCase(comptime quantized: bool, comptime seed: u64) !void {
    const a = testing.allocator;
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writePack(&p, quantized);
    var b = try openCase(&p, configJson(quantized), false, seed);
    defer b.close();
    const m = b.xfm.plugin.?.model;
    var sides: [2]Side = undefined;
    for (&sides) |*sd| try sd.init();
    defer for (&sides) |*sd| sd.deinit();
    const prompts = [2][]const i32{ &.{ 3, 17, 42, 99, 7, 250, 1, 8, 66, 31, 120, 5, 77 }, &.{ 9, 33, 77, 140, 2, 18 } };
    for (0..2) |i| {
        const pr = prompts[i];
        var g = ops.Graph.init(a, b.xfm.s);
        defer g.deinit();
        var host: [32]u32 = undefined;
        for (pr, 0..) |id, k| host[k] = @intCast(id);
        const handles = [_]abi.Handle{sides[i].slot.handle()};
        const got = try tableLogits(&g, m, .{ .mode = .prefill, .rows = @intCast(pr.len), .tokens = .of(host[0..pr.len]), .positions = .of(&.{0}), .caches = .of(&handles) });
        try b.check(got, false);
    }
    const ticks = [_][2][]const i32{
        .{ &.{88}, &.{12} },
        .{ &.{64}, &.{200} },
        .{ &.{ 5, 61, 200 }, &.{ 4, 11, 90 } },
        .{ &.{7}, &.{21} },
    };
    for (ticks) |tk| {
        var g = ops.Graph.init(a, b.xfm.s);
        defer g.deinit();
        const w = tk[0].len;
        var host: [8]u32 = undefined;
        for (0..2) |i| for (tk[i], 0..) |id, k| {
            host[i * w + k] = @intCast(id);
        };
        const offs = [_]u32{ @intCast(sides[0].off), @intCast(sides[1].off) };
        const handles = [_]abi.Handle{ sides[0].slot.handle(), sides[1].slot.handle() };
        const got = try tableLogits(&g, m, .{ .mode = .batched_rows, .rows = @intCast(w), .tokens = .of(host[0 .. 2 * w]), .positions = .of(&offs), .caches = .of(&handles) });
        try b.check(try g.reshape(got, &.{ @intCast(2 * w), VOCAB }), false);
        for (&sides) |*sd| sd.off += w;
    }
    try b.expectDone();
}

/// Prompt chunks past the ring (window + 2048 rows), then decode and a verify: the sliding layers' rings trim on both
/// sides.
fn ringCase(comptime seed: u64) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writePack(&p, true);
    var b = try openCase(&p, configJson(true), false, seed);
    defer b.close();
    var long: [2 * 1100]i32 = undefined;
    for (&long, 0..) |*id, i| id.* = @intCast((i * 7919 + 13) % VOCAB);
    const seq = [_]proof.Step{
        .{ .mode = .prefill, .ids = long[0..1100] },
        .{ .mode = .prefill, .ids = long[1100..] },
        .{ .mode = .decode, .ids = &.{88} },
        .{ .mode = .verify, .ids = &.{ 5, 61, 200 } },
        .{ .mode = .decode, .ids = &.{64} },
    };
    // Logits only.
    var side: Side = undefined;
    try side.init();
    defer side.deinit();
    const host = try testing.allocator.alloc(u32, long.len);
    defer testing.allocator.free(host);
    for (seq) |st| {
        var g = ops.Graph.init(testing.allocator, b.xfm.s);
        defer g.deinit();
        for (st.ids, host[0..st.ids.len]) |id, *h| h.* = @intCast(id);
        const handles = [_]abi.Handle{side.slot.handle()};
        const got = try tableLogits(&g, b.xfm.plugin.?.model, .{ .mode = st.mode, .rows = @intCast(st.ids.len), .tokens = .of(host[0..st.ids.len]), .positions = .of(&.{@intCast(side.off)}), .caches = .of(&handles) });
        try b.check(got, false);
        side.off += st.ids.len;
    }
    try b.expectDone();
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
    try testing.expectEqual(want.rope_local_base_freq, c.swa_rope_theta);
    try testing.expectEqual(want.query_pre_attn_scalar, c.head_dim);
    try testing.expectEqual(want.attention_value_scale, c.value_scale);
    try testing.expectEqual(want.tie_word_embeddings, c.tie_word_embeddings);
    try testing.expectEqual(want.num_experts, c.experts);
    try testing.expectEqual(want.num_experts_per_tok, c.top_k);
    try testing.expectEqual(want.moe_intermediate_size, c.moe_intermediate);
    try testing.expectEqual(want.moe_route_norm, c.route_norm);
    try testing.expectEqual(want.router_scaling_factor, c.route_scale);
    try testing.expectEqual(want.first_k_dense_replace, c.first_k_dense);
    try testing.expectEqual(if (want.has_sliding_window) want.sliding_window else 0, c.sliding_window);
    try testing.expectEqual(want.slidingKeepRows(), c.ring());
    for (0..c.num_hidden_layers) |i| {
        const l: u32 = @intCast(i);
        try testing.expectEqual(want.isGlobalLayer(l), c.isGlobal(l));
        try testing.expectEqual(want.layerKVHeads(l), c.kvHeads(l));
        try testing.expectEqual(want.layerVHeadDim(l), c.v_head_dim);
        try testing.expectEqual(want.layerRopeDims(l), c.ropeDims());
    }
    try testing.expectEqual(want.isMoe(), p.caps.moe);
}

// ── tests ──

test "mimo_v2 plugin: a 4-bit pack is bit-equal to forwardMoeWith at every step" {
    try case(true, 0x3130);
}

test "mimo_v2 plugin: a bf16 pack (decode and short verify read the attention side copies) is bit-equal" {
    try case(false, 0x3131);
}

test "mimo_v2 plugin: batched rows (decode ticks and a verify, slots of different lengths) are bit-equal to forwardMoeBatchedRows" {
    try batchedCase(true, 0x3132);
}

test "mimo_v2 plugin: bf16 batched rows are too" {
    try batchedCase(false, 0x3133);
}

test "mimo_v2 plugin: past the sliding ring (2200 rows in two chunks), then decode, is bit-equal" {
    try ringCase(0x3134);
}

test "mimo_v2 plugin: parse equals the engine's parse on the shell and the plugin's own fields" {
    try expectParse(configJson(true));
    try expectParse(config.pack_json);
}
