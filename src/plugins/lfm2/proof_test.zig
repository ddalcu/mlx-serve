//! Parity proof for the lfm2 plugin: tiny random packs (4-bit; bf16 nested under a `text_config` with torch-layout
//! conv taps and a tied head; lfm2_moe with dense and sparse layers) served by the table bit-equal to the legacy
//! `forwardHybridWith` at every step, and `parse` equal to the engine's parse on every field the plugin owns.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const proof = @import("proof.zig");
const both = @import("../../engines/mlx/plugin/both.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const model = @import("../../sdk/config/model.zig");
const lfm2 = @import("plugin.zig");
const config = @import("config.zig");

const testing = std.testing;

const H: c_int = 128;
const HEADS: c_int = 4;
const KV_HEADS: c_int = 2;
const HD: c_int = 32;
const INTER: c_int = 256;
const VOCAB: c_int = 256;
const EXPERTS: c_int = 8;
const MOE_INTER: c_int = 64;

const Shape = struct {
    quantized: bool = true,
    nested: bool = false,
    torch_conv: bool = false,
    tied: bool = true,
    moe: bool = false,
    /// An LFM2-VL config (`vision_config`, `image_token_index`) on the nested pack; the tower's rows come from the test.
    vision: bool = false,
    /// Quantized at 2 bits, group 128 (not with `moe`: the expert width is 64); ternary = every bias -scale.
    two_bit: enum { no, q2, ternary } = .no,
    /// The nested pack under mlx-community's `lfm2-vl` spelling.
    hyphen: bool = false,
};

fn tensorAs(p: *fixture.Pack, base: []const u8, shape: []const c_int, sh: Shape) !void {
    return switch (sh.two_bit) {
        .no => p.tensor(base, shape, sh.quantized),
        .q2 => p.affine(base, shape, 2, 128),
        .ternary => p.ternary(base, shape, 128),
    };
}

fn configJson(buf: []u8, sh: Shape) ![]const u8 {
    var body_buf: [1024]u8 = undefined;
    const body = try std.fmt.bufPrint(&body_buf,
        \\"vocab_size":256,"hidden_size":128,"intermediate_size":256,"num_hidden_layers":4,"num_attention_heads":4,
        \\"num_key_value_heads":2,"norm_eps":1e-5,"rope_theta":1000000.0,"max_position_embeddings":4096,
        \\"conv_L_cache":3,"layer_types":["conv","full_attention","conv","full_attention"],"tie_embedding":{s},
        \\"eos_token_id":7{s}
    , .{ if (sh.tied) "true" else "false", if (sh.moe) ",\"num_experts\":8,\"num_experts_per_tok\":2,\"moe_intermediate_size\":64,\"num_dense_layers\":1,\"norm_topk_prob\":true,\"routed_scaling_factor\":1.5" else "" });
    const q = if (sh.two_bit != .no) ",\"quantization\":{\"group_size\":128,\"bits\":2}" else if (sh.quantized) ",\"quantization\":{\"group_size\":64,\"bits\":4}" else "";
    const mt = if (sh.moe) "lfm2_moe" else if (sh.hyphen) "lfm2-vl" else if (sh.nested) "lfm2_vl" else "lfm2";
    const vl = if (sh.vision) ",\"vision_config\":{\"hidden_size\":64},\"image_token_index\":250" else "";
    if (sh.nested) return std.fmt.bufPrint(buf, "{{\"model_type\":\"{s}\",\"text_config\":{{{s}}},\"eos_token_id\":7{s}{s}}}", .{ mt, body, q, vl });
    return std.fmt.bufPrint(buf, "{{\"model_type\":\"{s}\",{s}{s}}}", .{ mt, body, q });
}

fn at(buf: []u8, prefix: []const u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ prefix, layer, name });
}

fn writePack(p: *fixture.Pack, sh: Shape) !void {
    var buf: [160]u8 = undefined;
    const m = if (sh.nested) "language_model.model" else "model";
    try tensorAs(p, try std.fmt.bufPrint(&buf, "{s}.embed_tokens", .{m}), &.{ VOCAB, H }, sh);
    try p.norm(try std.fmt.bufPrint(&buf, "{s}.embedding_norm.weight", .{m}), H);
    if (!sh.tied) try tensorAs(p, if (sh.nested) "language_model.lm_head" else "lm_head", &.{ VOCAB, H }, sh);
    for (0..4) |i| {
        try p.norm(try at(&buf, m, i, "operator_norm.weight"), H);
        try p.norm(try at(&buf, m, i, "ffn_norm.weight"), H);
        if (sh.moe and i >= 1) {
            try tensorAs(p, try at(&buf, m, i, "feed_forward.gate"), &.{ EXPERTS, H }, sh);
            try tensorAs(p, try at(&buf, m, i, "feed_forward.switch_mlp.gate_proj"), &.{ EXPERTS, MOE_INTER, H }, sh);
            try tensorAs(p, try at(&buf, m, i, "feed_forward.switch_mlp.up_proj"), &.{ EXPERTS, MOE_INTER, H }, sh);
            try tensorAs(p, try at(&buf, m, i, "feed_forward.switch_mlp.down_proj"), &.{ EXPERTS, H, MOE_INTER }, sh);
            // One layer ships its selection bias, the others take the zero default.
            if (i == 2) try p.denseAs(try at(&buf, m, i, "feed_forward.expert_bias"), &.{EXPERTS}, 0.3, 0, .float32);
        } else {
            const keys = if (sh.moe) [3][]const u8{ "gate_proj", "up_proj", "down_proj" } else [3][]const u8{ "w1", "w3", "w2" };
            var kb: [64]u8 = undefined;
            try tensorAs(p, try at(&buf, m, i, try std.fmt.bufPrint(&kb, "feed_forward.{s}", .{keys[0]})), &.{ INTER, H }, sh);
            try tensorAs(p, try at(&buf, m, i, try std.fmt.bufPrint(&kb, "feed_forward.{s}", .{keys[1]})), &.{ INTER, H }, sh);
            try tensorAs(p, try at(&buf, m, i, try std.fmt.bufPrint(&kb, "feed_forward.{s}", .{keys[2]})), &.{ H, INTER }, sh);
        }
        if (i % 2 == 0) {
            try tensorAs(p, try at(&buf, m, i, "conv.in_proj"), &.{ 3 * H, H }, sh);
            const taps: []const c_int = if (sh.torch_conv) &.{ H, 1, 3 } else &.{ H, 3, 1 };
            try p.dense(try at(&buf, m, i, "conv.conv.weight"), taps, 0.5, 0);
            try tensorAs(p, try at(&buf, m, i, "conv.out_proj"), &.{ H, H }, sh);
        } else {
            try tensorAs(p, try at(&buf, m, i, "self_attn.q_proj"), &.{ HEADS * HD, H }, sh);
            try tensorAs(p, try at(&buf, m, i, "self_attn.k_proj"), &.{ KV_HEADS * HD, H }, sh);
            try tensorAs(p, try at(&buf, m, i, "self_attn.v_proj"), &.{ KV_HEADS * HD, H }, sh);
            try tensorAs(p, try at(&buf, m, i, "self_attn.out_proj"), &.{ H, HEADS * HD }, sh);
            try p.norm(try at(&buf, m, i, "self_attn.q_layernorm.weight"), HD);
            try p.norm(try at(&buf, m, i, "self_attn.k_layernorm.weight"), HD);
        }
    }
}

/// After `proof.steps`: a wider chunk (the sorted MoE gather), a conv window carried across prompt chunks.
const wide_steps = [_]proof.Step{
    .{ .mode = .prefill, .ids = &.{ 3, 17, 42, 99, 7, 250, 1, 8, 66, 31, 120, 5, 77, 200 } },
    .{ .mode = .decode, .ids = &.{88} },
    .{ .mode = .verify, .ids = &.{ 5, 61, 200 } },
    .{ .mode = .prefill, .ids = &.{ 9, 33, 77, 140, 2, 18, 19, 20, 21, 22, 23 } },
    .{ .mode = .decode, .ids = &.{64} },
};


/// `sh`'s pack served by the driver on the legacy parse's config (recurrent entries per slot), against its oracle.
fn openCase(p: *fixture.Pack, sh: Shape, comptime seed: u64) !both.Both {
    try writePack(p, sh);
    var cbuf: [2048]u8 = undefined;
    const json = try configJson(&cbuf, sh);
    const name = std.fmt.comptimePrint("fixtures/{x}.tape", .{seed});
    return both.Both.replay(&lfm2.table, p, json, true, @embedFile(name));
}

/// Prefill, decode and verify past a chunk and a conv window carried across chunks, then the pipelined greedy decode.
fn servedCase(sh: Shape, comptime seed: u64) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    var b = try openCase(&p, sh, seed);
    defer b.close();
    try b.expectSteps(&(proof.steps ++ wide_steps));
    try b.expectSteps(&both.long_steps);
    try b.expectGreedy();
    try b.expectDone();
}

/// `parse` against the engine's parse of the same bytes: the shell and every field the plugin keeps.
fn expectParse(json: []const u8) !void {
    var want = try proof.legacyConfig(json);
    defer want.deinit(testing.allocator);
    const mem: @import("../../sdk/root.zig").fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(config.parse(&dir, .of(json), .{}, null, true, &p).isOk());
    defer config.freeParsed(&p);
    try proof.expectShell(&p.shell, &want);
    const c = config.of(&p);
    try testing.expectEqualStrings(want.weight_prefix, c.prefix);
    try testing.expectEqual(want.rms_norm_eps, c.rms_norm_eps);
    try testing.expectEqual(want.rope_theta, c.rope_theta);
    try testing.expectEqual(want.lfm_conv_kernel, c.conv_kernel);
    try testing.expectEqual(want.tie_word_embeddings, c.tie_word_embeddings);
    try testing.expectEqual(want.lfm2_moe, c.moe != null);
    for (0..c.num_hidden_layers) |i| try testing.expectEqual(want.layer_block_types[i] == .gated_conv, c.layers[i] == .conv);
    if (c.moe) |m| {
        try testing.expectEqual(want.num_experts, m.experts);
        try testing.expectEqual(want.num_experts_per_tok, m.top_k);
        try testing.expectEqual(want.moe_intermediate_size, m.intermediate);
        try testing.expectEqual(want.num_dense_layers, m.dense_layers);
        try testing.expectEqual(want.moe_route_norm, m.norm);
        try testing.expectEqual(want.router_scaling_factor, m.scale);
        try testing.expectEqual(want.num_experts_per_tok, p.shell.num_experts_per_tok);
        try testing.expectEqual(want.moe_intermediate_size, p.shell.moe_intermediate_size);
    }
    try testing.expectEqual(want.isMoe(), p.caps.moe);
}

// ── tests ──

test "lfm2 served: a 4-bit pack is bit-equal to the engine's forward" {
    try servedCase(.{}, 0x1f3a);
}

test "lfm2 served: a bf16 pack nested under text_config, torch-layout conv taps and an untied head is bit-equal" {
    try servedCase(.{ .quantized = false, .nested = true, .torch_conv = true, .tied = false }, 0x1f3b);
}

test "lfm2 served: an lfm2_moe pack (dense first layer, sigmoid-routed experts, absent selection bias) is bit-equal" {
    try servedCase(.{ .moe = true }, 0x1f3c);
}

test "lfm2 plugin: parse equals the engine's parse on the shell and the plugin's own fields" {
    var buf: [2048]u8 = undefined;
    try expectParse(config.lfm25_text);
    try expectParse(try configJson(&buf, .{}));
    try expectParse(try configJson(&buf, .{ .moe = true }));
    try expectParse(try configJson(&buf, .{ .quantized = false, .tied = false }));
}

test "lfm2 served: an LFM2-VL pack splices media rows bit-equal across a chunk boundary" {
    var p = fixture.Pack.init(0x1f3d);
    defer p.deinit();
    var b = try openCase(&p, .{ .nested = true, .vision = true }, 0x1f3d);
    defer b.close();
    try b.expectSteps(&both.long_steps);
    try b.expectMedia(250, H);
    try b.expectGreedy();
    try b.expectDone();
}

test "lfm2 served: an LFM2-VL pack's tower (either nesting) loads into the engine's map" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const sh: Shape = .{ .nested = true, .vision = true };
    var cbuf: [2048]u8 = undefined;
    const json = try configJson(&cbuf, sh);
    for ([_][2][]const u8{
        .{ "vision_tower.embeddings.patch_embedding.weight", "multi_modal_projector.linear_1.weight" },
        .{ "model.vision_tower.vision_model.embeddings.patch_embedding.weight", "model.multi_modal_projector.linear_1.weight" },
    }) |keys| {
        var tp = fixture.Pack.init(0x1f3e);
        defer tp.deinit();
        try writePack(&tp, sh);
        for (keys) |k| try tp.norm(k, 32);
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
        var path_buf: [512]u8 = undefined;
        const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
        try tp.save(dir);
        var cfg = try model.parseConfig(io, a, dir);
        defer cfg.deinit(a);
        try testing.expect(cfg.plugin != null);
        var tower = model.Weights.init(a);
        defer tower.deinit();
        var x = try driver.loadWith(io, a, &cfg, dir, fixture.stream(), &tower);
        defer x.deinit();
        try testing.expectEqual(@as(u32, 2), tower.map.count());
        for (keys) |k| try testing.expect(tower.map.contains(k));
    }
}

test "lfm2 served: 2-bit group-128 packs, plain (stock qmm) and ternary (qmv2's kernels on both sides), are bit-equal" {
    inline for (.{ .q2, .ternary }, .{ 0x1f41, 0x1f42 }) |two, seed| {
        var p = fixture.Pack.init(seed);
        defer p.deinit();
        var b = try openCase(&p, .{ .two_bit = two, .tied = false }, seed);
        defer b.close();
        const m: *const lfm2.Loaded = @ptrCast(@alignCast(b.xfm.plugin.?.model));
        try testing.expectEqual(two == .ternary, m.base.weights.head.ternary and m.layers[0].op.first.in_proj.ternary);
        try b.expectSteps(&both.long_steps);
        try b.expectGreedy();
        try b.expectDone();
    }
}

test "lfm2 config: an LFM2-VL config without its image id parses, the id left for the driver (the tokenizer's <image>)" {
    var cbuf: [2048]u8 = undefined;
    const vl = try configJson(&cbuf, .{ .nested = true, .vision = true });
    var jbuf: [2048]u8 = undefined;
    const needle = ",\"image_token_index\":250";
    try testing.expectEqual(@as(usize, 1), std.mem.replace(u8, vl, needle, "", &jbuf));
    const plain = jbuf[0 .. vl.len - needle.len];
    const mem: @import("../../sdk/root.zig").fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = plain }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(config.parse(&dir, .of(plain), .{}, null, true, &p).isOk());
    defer config.freeParsed(&p);
    try testing.expect(p.caps.images and p.placeholder_ids.image == 0);
}

test "lfm2 served: mlx-community's `lfm2-vl` spelling binds as the text model the engine reads it as" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const json = @embedFile("../../engines/mlx/plugin/fixtures/mlx_lm/mlx-community_LFM2-VL-1.6B-4bit.json");
    var want = try proof.legacyConfig(json);
    defer want.deinit(testing.allocator);
    try testing.expect(!want.lfm2_vision);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    const bound = (try @import("../../engines/mlx/plugin/config.zig").bindServed(io, testing.allocator, dir, json)) orelse return error.TestPackNotServed;
    defer bound.destroy(testing.allocator);
    try testing.expect(!bound.parsed.caps.images);
    try expectParse(json);
    try servedCase(.{ .nested = true, .vision = true, .hyphen = true }, 0x1f43);
}

test "lfm2 served: a D1 decision pack (an LFM2-VL config naming modeling_d1) is refused on the LM's served path" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var cbuf: [2048]u8 = undefined;
    const vl = try configJson(&cbuf, .{ .nested = true, .vision = true });
    var jbuf: [2200]u8 = undefined;
    const json = try std.fmt.bufPrint(&jbuf, "{s},\"auto_map\":{{\"AutoModel\":\"modeling_d1.D1Model\"}}}}", .{vl[0 .. vl.len - 1]});
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    try testing.expectError(error.PluginDeclinedNoLegacy, @import("../../engines/mlx/plugin/config.zig").bindServed(io, testing.allocator, dir, json));
}

// ── a `--mlx-gguf` file ──

const gguf_arch = @import("../../sdk/quant/gguf/arch.zig");

const GH: c_int = 256;
const G_INTER: c_int = 512;
const G_EXPERTS: c_int = 4;

/// The config `sdk/quant/gguf/arch/lfm2.zig` rebuilds from a GGUF's metadata, at this pack's geometry.
fn ggufConfig(moe: bool) ![]u8 {
    var hp = std.mem.zeroes(gguf_arch.lfm2.Hparams);
    hp = .{ .moe = moe, .n_layers = 4, .hidden = GH, .ffn = G_INTER, .n_heads = 4, .n_kv_heads = 2, .attention = hp.attention, .conv_cache = 3, .rope_theta = 1e6, .rms_eps = 1e-5, .ctx_len = 4096, .vocab = VOCAB, .eos = 7, .n_experts = if (moe) G_EXPERTS else 0, .n_experts_used = if (moe) 2 else 0, .expert_ffn = if (moe) GH else 0, .dense_layers = if (moe) 1 else 0 };
    hp.attention[1] = true;
    hp.attention[3] = true;
    return gguf_arch.lfm2.configJson(testing.allocator, hp);
}

/// The weights an LFM2 GGUF loads into: k-quant, legacy and i-quant blocks across the projections (q6_k table and
/// down projections), one float projection, the router and every norm, bias and conv tap as bf16.
fn writeGgufPack(p: *fixture.Pack, moe: bool) !void {
    var buf: [160]u8 = undefined;
    try p.gguf("model.embed_tokens", &.{ VOCAB, GH }, .q6_k);
    try p.norm("model.embedding_norm.weight", GH);
    for (0..4) |i| {
        try p.norm(try at(&buf, "model", i, "operator_norm.weight"), GH);
        try p.norm(try at(&buf, "model", i, "ffn_norm.weight"), GH);
        if (moe and i >= 1) {
            try p.weight(try at(&buf, "model", i, "feed_forward.gate"), G_EXPERTS, GH, false);
            try p.gguf(try at(&buf, "model", i, "feed_forward.switch_mlp.gate_proj"), &.{ G_EXPERTS, GH, GH }, .q4_k);
            try p.gguf(try at(&buf, "model", i, "feed_forward.switch_mlp.up_proj"), &.{ G_EXPERTS, GH, GH }, .q4_k);
            try p.gguf(try at(&buf, "model", i, "feed_forward.switch_mlp.down_proj"), &.{ G_EXPERTS, GH, GH }, .q6_k);
            try p.denseAs(try at(&buf, "model", i, "feed_forward.expert_bias"), &.{G_EXPERTS}, 0.3, 0, .bfloat16);
        } else {
            try p.gguf(try at(&buf, "model", i, "feed_forward.w1"), &.{ G_INTER, GH }, .q4_k);
            try p.gguf(try at(&buf, "model", i, "feed_forward.w3"), &.{ G_INTER, GH }, .q4_k);
            try p.gguf(try at(&buf, "model", i, "feed_forward.w2"), &.{ GH, G_INTER }, .q6_k);
        }
        if (i % 2 == 0) {
            try p.gguf(try at(&buf, "model", i, "conv.in_proj"), &.{ 3 * GH, GH }, .q4_k);
            try p.dense(try at(&buf, "model", i, "conv.conv.weight"), &.{ GH, 3, 1 }, 0.5, 0);
            try p.gguf(try at(&buf, "model", i, "conv.out_proj"), &.{ GH, GH }, .q5_0);
        } else {
            try p.gguf(try at(&buf, "model", i, "self_attn.q_proj"), &.{ 4 * 64, GH }, .q4_k);
            if (i == 3) try p.weight(try at(&buf, "model", i, "self_attn.k_proj"), 2 * 64, GH, false) else try p.gguf(try at(&buf, "model", i, "self_attn.k_proj"), &.{ 2 * 64, GH }, .q4_k);
            try p.gguf(try at(&buf, "model", i, "self_attn.v_proj"), &.{ 2 * 64, GH }, .q8_0);
            try p.gguf(try at(&buf, "model", i, "self_attn.out_proj"), &.{ GH, 4 * 64 }, .iq4_xs);
            try p.norm(try at(&buf, "model", i, "self_attn.q_layernorm.weight"), 64);
            try p.norm(try at(&buf, "model", i, "self_attn.k_layernorm.weight"), 64);
        }
    }
}


fn ggufCase(moe: bool, comptime seed: u64) !void {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    try writeGgufPack(&p, moe);
    const json = try ggufConfig(moe);
    defer testing.allocator.free(json);
    const name = std.fmt.comptimePrint("fixtures/gguf-{x}.tape", .{seed});
    var b = try both.Both.replay(&lfm2.table, &p, json, true, @embedFile(name));
    defer b.close();
    const m: *const lfm2.Loaded = @ptrCast(@alignCast(b.xfm.plugin.?.model));
    try testing.expect(m.layers[0].op.first.in_proj.mode == .gguf and m.base.weights.embed.table.mode == .gguf);
    try b.expectSteps(&(proof.steps ++ wide_steps));
    try b.expectSteps(&both.long_steps);
    try b.expectGreedy();
    try b.expectDone();
}

test "lfm2 served: a --mlx-gguf file (k-quant, legacy and i-quant blocks, a float projection, a tied q6_k table) is bit-equal" {
    try ggufCase(false, 0x1f44);
}

test "lfm2 served: a --mlx-gguf lfm2_moe file (gguf expert banks, a float router) is bit-equal" {
    try ggufCase(true, 0x1f45);
}

test "lfm2 plugin: a --mlx-gguf file's rebuilt config parses as the engine's" {
    for ([_]bool{ false, true }) |moe| {
        const json = try ggufConfig(moe);
        defer testing.allocator.free(json);
        try expectParse(json);
    }
}

/// A real LFM2 GGUF file (`lfm2` or `lfm2moe`) at the `ggufConfig` geometry, as llama.cpp names and lays out its
/// tensors: the engine's own GGUF loader reads it on both sides.
fn writeGgufFile(g: *proof.Gguf, moe: bool, seed: u64) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const arch = if (moe) "lfm2moe" else "lfm2";
    var kb: [96]u8 = undefined;
    try g.string("general.architecture", arch);
    const k = struct {
        fn f(b: []u8, a: []const u8, s: []const u8) ![]const u8 {
            return std.fmt.bufPrint(b, "{s}.{s}", .{ a, s });
        }
    }.f;
    try g.int(try k(&kb, arch, "block_count"), 4);
    try g.int(try k(&kb, arch, "embedding_length"), GH);
    try g.int(try k(&kb, arch, "feed_forward_length"), G_INTER);
    try g.int(try k(&kb, arch, "attention.head_count"), 4);
    try g.ints(try k(&kb, arch, "attention.head_count_kv"), &.{ 0, 2, 0, 2 });
    try g.int(try k(&kb, arch, "shortconv.l_cache"), 3);
    try g.float(try k(&kb, arch, "rope.freq_base"), 1e6);
    try g.float(try k(&kb, arch, "attention.layer_norm_rms_epsilon"), 1e-5);
    try g.int(try k(&kb, arch, "context_length"), 4096);
    if (moe) {
        try g.int(try k(&kb, arch, "expert_count"), G_EXPERTS);
        try g.int(try k(&kb, arch, "expert_used_count"), 2);
        try g.int(try k(&kb, arch, "expert_feed_forward_length"), GH);
        try g.int(try k(&kb, arch, "leading_dense_block_count"), 1);
    }
    try g.vocab(VOCAB, 7);
    const W: u64 = GH;
    try g.quant(r, "token_embd.weight", &.{ VOCAB, W }, .q6_k);
    try g.f32s(r, "token_embd_norm.weight", &.{W}, 0.2, 1);
    var nb: [96]u8 = undefined;
    for (0..4) |i| {
        const at_ = struct {
            fn f(b: []u8, l: usize, leaf: []const u8) ![]const u8 {
                return std.fmt.bufPrint(b, "blk.{d}.{s}", .{ l, leaf });
            }
        }.f;
        try g.f32s(r, try at_(&nb, i, "attn_norm.weight"), &.{W}, 0.2, 1);
        try g.f32s(r, try at_(&nb, i, "ffn_norm.weight"), &.{W}, 0.2, 1);
        if (moe and i >= 1) {
            try g.f32s(r, try at_(&nb, i, "ffn_gate_inp.weight"), &.{ W, G_EXPERTS }, 0.06, 0);
            try g.quant(r, try at_(&nb, i, "ffn_gate_exps.weight"), &.{ G_EXPERTS, W, W }, .q4_k);
            try g.quant(r, try at_(&nb, i, "ffn_up_exps.weight"), &.{ G_EXPERTS, W, W }, .q4_k);
            try g.quant(r, try at_(&nb, i, "ffn_down_exps.weight"), &.{ G_EXPERTS, W, W }, .q6_k);
            try g.f32s(r, try at_(&nb, i, "exp_probs_b.bias"), &.{G_EXPERTS}, 0.3, 0);
        } else {
            try g.quant(r, try at_(&nb, i, "ffn_gate.weight"), &.{ G_INTER, W }, .q4_k);
            try g.quant(r, try at_(&nb, i, "ffn_up.weight"), &.{ G_INTER, W }, .q4_k);
            try g.quant(r, try at_(&nb, i, "ffn_down.weight"), &.{ W, G_INTER }, .q6_k);
        }
        if (i % 2 == 0) {
            try g.quant(r, try at_(&nb, i, "shortconv.in_proj.weight"), &.{ 3 * W, W }, .q4_k);
            try g.f32s(r, try at_(&nb, i, "shortconv.conv.weight"), &.{ 3, W }, 0.5, 0);
            try g.quant(r, try at_(&nb, i, "shortconv.out_proj.weight"), &.{ W, W }, .q5_0);
        } else {
            try g.quant(r, try at_(&nb, i, "attn_q.weight"), &.{ 4 * 64, W }, .q4_k);
            try g.f32s(r, try at_(&nb, i, "attn_k.weight"), &.{ W, 2 * 64 }, 0.06, 0);
            try g.quant(r, try at_(&nb, i, "attn_v.weight"), &.{ 2 * 64, W }, .q8_0);
            try g.quant(r, try at_(&nb, i, "attn_output.weight"), &.{ W, 4 * 64 }, .iq4_xs);
            try g.f32s(r, try at_(&nb, i, "attn_q_norm.weight"), &.{64}, 0.2, 1);
            try g.f32s(r, try at_(&nb, i, "attn_k_norm.weight"), &.{64}, 0.2, 1);
        }
    }
}

fn ggufFileCase(moe: bool, by_file: bool, comptime seed: u64) !void {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    var g: proof.Gguf = .{};
    defer g.deinit();
    try writeGgufFile(&g, moe, seed);
    const name = std.fmt.comptimePrint("fixtures/gguf-file-{x}.tape", .{seed});
    var b = try proof.openGguf(&lfm2.table, &g, by_file, .{ .tape = @embedFile(name) });
    defer b.close();
    try b.expectSteps(&(proof.steps ++ wide_steps));
    try b.expectSteps(&both.long_steps);
    try b.expectGreedy();
    try b.expectDone();
}

test "lfm2 served: a GGUF file the engine loads (--mlx-gguf), addressed by the file, is bit-equal" {
    try ggufFileCase(false, true, 0x1f46);
}

test "lfm2 served: an lfm2moe GGUF file the engine loads (--mlx-gguf), addressed by its folder, is bit-equal" {
    try ggufFileCase(true, false, 0x1f47);
}

test "lfm2 plugin: a pack the plugin declines is refused by name (no legacy forward to fall back to)" {
    var buf: [2048]u8 = undefined;
    try both.expectNoLegacyFallback(try configJson(&buf, .{}), "expert_quant");
    try both.expectNoLegacyFallback(try configJson(&buf, .{ .moe = true }), "modules");
}
