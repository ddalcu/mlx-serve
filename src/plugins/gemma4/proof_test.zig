//! The Gemma 4 plugin proven against the legacy engine on tiny random packs (Gemma 3's `Both` harness): an E-series
//! shape (4-bit; per-layer embeddings at width 256, so the fused gelu*ple decode tail runs; a KV-shared tail; K=V and
//! a wider head on full layers; proportional partial rope; layer scalars; tied head; softcap) and a 31B-like bf16 pack
//! (no PLE, K=V, untied head). Steps, greedy pipelined decode and batched rows bit-equal; the parse equals the legacy
//! parse field for field.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const quant_abi = @import("../../abi/quant.zig");
const ops = @import("../../mlx/ops.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const proof = @import("../gemma3/proof_test.zig");
const gemma4 = @import("plugin.zig");
const drafter_mod = @import("assistant_drafter.zig");
const plugin_config = @import("../../engines/mlx/plugin/config.zig");
const Generator = @import("../../engines/mlx/generate/generator.zig").Generator;
const Tokenizer = @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;

const testing = std.testing;


fn oracle(comptime name: []const u8) proof.Oracle {
    const rel = "fixtures/" ++ name ++ ".tape";
    return .{ .tape = @embedFile(rel) };
}

pub const H = 128;
const VOCAB = 256;
pub const INTER = 256;

fn at(buf: []u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "language_model.model.layers.{d}.{s}", .{ layer, name });
}

/// Six layers (sliding, sliding, full, sliding, then two reading layers 3 and 2): hd 256 sliding over 1 KV head, hd
/// 512 full layers over `gkv` heads; `pd` > 0 adds per-layer embeddings; `k_eq_v` full layers project no v.
fn writeGemma4(p: *fixture.Pack, q: ?Quant, pd: c_int, shared: usize, k_eq_v: bool, head: bool) !void {
    var buf: [192]u8 = undefined;
    const m = "language_model.model.";
    const n = 6;
    try put(p, m ++ "embed_tokens", &.{ VOCAB, H }, q);
    try p.norm(m ++ "norm.weight", H);
    if (head) try put(p, "language_model.lm_head", &.{ VOCAB, H }, q);
    if (pd > 0) {
        try put(p, m ++ "embed_tokens_per_layer", &.{ VOCAB, n * pd }, q);
        try put(p, m ++ "per_layer_model_projection", &.{ n * pd, H }, q);
        try p.norm(m ++ "per_layer_projection_norm.weight", pd);
    }
    for (0..n) |i| {
        const global = i == 2 or i == 5;
        const hd: c_int = if (global) 512 else 256;
        for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight" }) |nm|
            try p.norm(try at(&buf, i, nm), H);
        try p.dense(try at(&buf, i, "layer_scalar"), &.{1}, 0.2, 0.9);
        try put(p, try at(&buf, i, "self_attn.q_proj"), &.{ 2 * hd, H }, q);
        try put(p, try at(&buf, i, "self_attn.o_proj"), &.{ H, 2 * hd }, q);
        try p.norm(try at(&buf, i, "self_attn.q_norm.weight"), hd);
        if (i < n - shared) {
            try put(p, try at(&buf, i, "self_attn.k_proj"), &.{ hd, H }, q);
            if (!(global and k_eq_v)) try put(p, try at(&buf, i, "self_attn.v_proj"), &.{ hd, H }, q);
            try p.norm(try at(&buf, i, "self_attn.k_norm.weight"), hd);
        }
        try put(p, try at(&buf, i, "mlp.gate_proj"), &.{ INTER, H }, q);
        try put(p, try at(&buf, i, "mlp.up_proj"), &.{ INTER, H }, q);
        try put(p, try at(&buf, i, "mlp.down_proj"), &.{ H, INTER }, q);
        if (pd > 0) {
            try put(p, try at(&buf, i, "per_layer_input_gate"), &.{ pd, H }, q);
            try put(p, try at(&buf, i, "per_layer_projection"), &.{ H, pd }, q);
            try p.norm(try at(&buf, i, "post_per_layer_input_norm.weight"), H);
        }
    }
}

const layer_types =
    \\"layer_types":["sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","full_attention"],
;

const e_series =
    \\{"model_type":"gemma4","text_config":{"model_type":"gemma4_text","hidden_size":128,"intermediate_size":256,
    \\"num_hidden_layers":6,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":256,"global_head_dim":512,
    \\"num_kv_shared_layers":2,"attention_k_eq_v":true,"hidden_size_per_layer_input":256,"vocab_size":256,
    \\"rms_norm_eps":1e-6,"sliding_window":4,"tie_word_embeddings":true,
++ layer_types ++
    \\"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"proportional","partial_rotary_factor":0.25},
    \\"sliding_attention":{"rope_theta":10000.0}},"final_logit_softcapping":30.0,"max_position_embeddings":4096},
    \\"eos_token_id":[1,106],"quantization":{"group_size":64,"bits":4}}
;

const dense31b =
    \\{"model_type":"gemma4_text","hidden_size":128,"intermediate_size":256,"num_hidden_layers":6,
    \\"num_attention_heads":2,"num_key_value_heads":1,"head_dim":256,"global_head_dim":512,"num_global_key_value_heads":1,
    \\"attention_k_eq_v":true,"vocab_size":256,"rms_norm_eps":1e-6,"sliding_window":4,"tie_word_embeddings":false,
++ layer_types ++
    \\"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"proportional","partial_rotary_factor":0.25},
    \\"sliding_attention":{"rope_theta":10000.0}},"final_logit_softcapping":30.0,"max_position_embeddings":4096,
    \\"eos_token_id":1}
;

/// The drafter hook hands out the trunk's own embedding table and head: the legacy's bytes, one array when tied.
fn expectTrunk(b: *proof.Both, tied: bool) !void {
    const m = b.xfm.plugin.?.model;
    const hook = gemma4.table.spec.?.trunk_weight.?;
    var e: quant_abi.Weight = .{};
    var h: quant_abi.Weight = .{};
    try testing.expect(hook(m, .embed, &e).isOk() and hook(m, .head, &h).isOk());
    try testing.expectEqual(tied, e.w.ctx == h.w.ctx);
    try b.check(.{ .ctx = e.w.ctx }, false);
    try b.check(.{ .ctx = h.w.ctx }, false);
}

test "gemma4 plugin: an E-series 4-bit pack (PLE, KV-shared tail, K=V, layer scalars) is bit-equal to the legacy forward" {
    var p = fixture.Pack.init(0x9e4a);
    defer p.deinit();
    try writeGemma4(&p, .{}, 256, 2, true, false);
    var b = try proof.Both.openAs(&gemma4.table, &p, e_series, oracle("e-series"));
    defer b.close();
    try expectTrunk(&b, true);
    try proof.expectSteps(&b);
    try proof.expectGreedy(&b);
    try proof.expectBatched(&b);
    try b.expectDone();
    try @import("../../engines/mlx/plugin/both.zig").expectEmbeddingBatch(&b.xfm);
}

/// `json` with a tower and its placeholder (250), the engine's way of marking a vision pack.
fn withVision(json: []const u8) ![]u8 {
    return std.mem.replaceOwned(u8, testing.allocator, json, "{\"model_type\":\"gemma4\",", "{\"model_type\":\"gemma4\",\"image_token_id\":250,\"vision_config\":{\"hidden_size\":64},");
}

test "gemma4 plugin: image rows splice bit-equal on an E-series pack (per-layer inputs read id 0 there) and a 26B-A4B pack" {
    {
        var p = fixture.Pack.init(0x9e4e);
        defer p.deinit();
        try writeGemma4(&p, .{}, 256, 2, true, false);
        const json = try withVision(e_series);
        defer testing.allocator.free(json);
        var b = try proof.Both.openAs(&gemma4.table, &p, json, oracle("e-series-vision"));
        defer b.close();
        try testing.expectEqual(@as(u32, 250), b.cfg.image_token_id);
        try proof.expectMedia(&b, 250);
        try b.expectDone();
    }
    var p = fixture.Pack.init(0x9e4f);
    defer p.deinit();
    try writeGemma4Moe(&p, .{}, "language_model.model", false);
    const json = try withVision(a4b);
    defer testing.allocator.free(json);
    var b = try proof.Both.openAs(&gemma4.table, &p, json, oracle("a4b-vision"));
    defer b.close();
    try proof.expectMedia(&b, 250);
    try b.expectDone();
}

test "gemma4 plugin: a unified pack's image and audio rows (its encoder-free embedder's) splice bit-equal" {
    var p = fixture.Pack.init(0x9e52);
    defer p.deinit();
    try writeGemma4(&p, .{}, 256, 2, true, false);
    const vision = try withVision(e_series);
    defer testing.allocator.free(vision);
    const json = try std.mem.replaceOwned(u8, testing.allocator, vision, "{\"model_type\":\"gemma4\",", "{\"model_type\":\"gemma4_unified\",\"audio_token_id\":251,\"audio_config\":{\"hidden_size\":64},");
    defer testing.allocator.free(json);
    var b = try proof.Both.openAs(&gemma4.table, &p, json, oracle("unified"));
    defer b.close();
    try testing.expect(b.cfg.is_gemma4_unified and b.xfm.plugin != null);
    try proof.expectMedia(&b, 250);
    try proof.expectMedia(&b, 251);
    try b.expectDone();
}

test "gemma4 plugin: a 31B-like bf16 pack (no PLE, K=V full layers, untied head) is bit-equal to the legacy forward" {
    var p = fixture.Pack.init(0x9e4b);
    defer p.deinit();
    try writeGemma4(&p, null, 0, 0, true, true);
    var b = try proof.Both.openAs(&gemma4.table, &p, dense31b, oracle("31b"));
    defer b.close();
    try expectTrunk(&b, false);
    try proof.expectSteps(&b);
    try proof.expectGreedy(&b);
    try proof.expectBatched(&b);
    try b.expectDone();
}

test "gemma4 plugin: lm_head_mxfp8 (LOSSY) quantizes the 31B-like pack's untied head as the legacy load does" {
    const cli = &@import("../../engines/mlx/transformer.zig").cli().lm_head_mxfp8_cli;
    const was = cli.*;
    defer cli.* = was;
    cli.* = true;
    var p = fixture.Pack.init(0x9e4c);
    defer p.deinit();
    try writeGemma4(&p, null, 0, 0, true, true);
    var b = try proof.Both.openAs(&gemma4.table, &p, dense31b, oracle("31b-mxfp8"));
    defer b.close();
    try testing.expect(b.xfm.lm_head_s.ctx != null);
    try proof.expectSteps(&b);
    try proof.expectGreedy(&b);
    try b.expectDone();
}

/// A 26B-A4B-shaped layer stack: the attention of `writeGemma4` (no PLE, K=V full layers), a shared GeGLU MLP and
/// `E` experts of width `MI` under the sigma-MoE router, the two extra feed-forward norms, layer scalars.
const E = 8;
const MI = 128;
pub const Quant = proof.Quant;
const put = proof.put;

pub fn writeGemma4Moe(p: *fixture.Pack, q: ?Quant, prefix: []const u8, packed_gu: bool) !void {
    var buf: [192]u8 = undefined;
    var nb: [96]u8 = undefined;
    const n = 4;
    try put(p, try std.fmt.bufPrint(&nb, "{s}.embed_tokens", .{prefix}), &.{ VOCAB, H }, q);
    try p.norm(try std.fmt.bufPrint(&nb, "{s}.norm.weight", .{prefix}), H);
    for (0..n) |i| {
        const global = i == 2;
        const hd: c_int = if (global) 512 else 256;
        const L = struct {
            fn at(b: []u8, pre: []const u8, li: usize, leaf: []const u8) ![]const u8 {
                return std.fmt.bufPrint(b, "{s}.layers.{d}.{s}", .{ pre, li, leaf });
            }
        };
        for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "pre_feedforward_layernorm_2.weight", "post_feedforward_layernorm_1.weight", "post_feedforward_layernorm_2.weight", "router.scale" }) |nm|
            try p.norm(try L.at(&buf, prefix, i, nm), H);
        try p.dense(try L.at(&buf, prefix, i, "layer_scalar"), &.{1}, 0.2, 0.9);
        try p.dense(try L.at(&buf, prefix, i, "router.per_expert_scale"), &.{E}, 0.2, 0.9);
        try put(p, try L.at(&buf, prefix, i, "self_attn.q_proj"), &.{ 2 * hd, H }, q);
        try put(p, try L.at(&buf, prefix, i, "self_attn.o_proj"), &.{ H, 2 * hd }, q);
        try p.norm(try L.at(&buf, prefix, i, "self_attn.q_norm.weight"), hd);
        try put(p, try L.at(&buf, prefix, i, "self_attn.k_proj"), &.{ hd, H }, q);
        if (!global) try put(p, try L.at(&buf, prefix, i, "self_attn.v_proj"), &.{ hd, H }, q);
        try p.norm(try L.at(&buf, prefix, i, "self_attn.k_norm.weight"), hd);
        try put(p, try L.at(&buf, prefix, i, "mlp.gate_proj"), &.{ INTER, H }, q);
        try put(p, try L.at(&buf, prefix, i, "mlp.up_proj"), &.{ INTER, H }, q);
        try put(p, try L.at(&buf, prefix, i, "mlp.down_proj"), &.{ H, INTER }, q);
        // mlx-lm keeps the router 8-bit affine in a float-quantized pack.
        const router_q: ?Quant = if (q) |x| (if (std.mem.eql(u8, std.mem.span(x.mode), "affine")) x else .{ .bits = 8 }) else null;
        try put(p, try L.at(&buf, prefix, i, "router.proj"), &.{ E, H }, router_q);
        if (packed_gu) {
            try put(p, try L.at(&buf, prefix, i, "experts.gate_up_proj"), &.{ E, 2 * MI, H }, q);
            try put(p, try L.at(&buf, prefix, i, "experts.down_proj"), &.{ E, H, MI }, q);
        } else {
            try put(p, try L.at(&buf, prefix, i, "experts.switch_glu.gate_proj"), &.{ E, MI, H }, q);
            try put(p, try L.at(&buf, prefix, i, "experts.switch_glu.up_proj"), &.{ E, MI, H }, q);
            try put(p, try L.at(&buf, prefix, i, "experts.switch_glu.down_proj"), &.{ E, H, MI }, q);
        }
    }
}

const moe_layer_types =
    \\"layer_types":["sliding_attention","sliding_attention","full_attention","sliding_attention"],
;

const a4b =
    \\{"model_type":"gemma4","text_config":{"model_type":"gemma4_text","hidden_size":128,"intermediate_size":256,
    \\"num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":256,"global_head_dim":512,
    \\"num_global_key_value_heads":1,"attention_k_eq_v":true,"vocab_size":256,"rms_norm_eps":1e-6,"sliding_window":4,
    \\"num_experts":8,"top_k_experts":2,"moe_intermediate_size":128,"enable_moe_block":true,"tie_word_embeddings":true,
++ moe_layer_types ++
    \\"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"proportional","partial_rotary_factor":0.25},
    \\"sliding_attention":{"rope_theta":10000.0}},"final_logit_softcapping":30.0,"max_position_embeddings":4096},
    \\"eos_token_id":[1,106],"quantization":{"group_size":64,"bits":4}}
;

test "gemma4 plugin: a 26B-A4B-shaped 4-bit pack (shared MLP + sigma-MoE experts) is bit-equal to the legacy forward" {
    var p = fixture.Pack.init(0x9e4c);
    defer p.deinit();
    try writeGemma4Moe(&p, .{}, "language_model.model", false);
    var b = try proof.Both.openAs(&gemma4.table, &p, a4b, oracle("a4b"));
    defer b.close();
    try proof.expectSteps(&b);
    try proof.expectGreedy(&b);
    try b.expectDone();
}

test "gemma4 plugin: a 26B-A4B-shaped bf16 pack is bit-equal to the legacy forward" {
    var p = fixture.Pack.init(0x9e4d);
    defer p.deinit();
    try writeGemma4Moe(&p, null, "language_model.model", false);
    const json = try std.mem.replaceOwned(u8, testing.allocator, a4b, ",\"quantization\":{\"group_size\":64,\"bits\":4}", "");
    defer testing.allocator.free(json);
    var b = try proof.Both.openAs(&gemma4.table, &p, json, oracle("a4b-bf16"));
    defer b.close();
    try proof.expectSteps(&b);
    try proof.expectGreedy(&b);
    try b.expectDone();
}

test "gemma4 plugin: the parse equals the legacy parse on the proof packs" {
    for ([_][]const u8{ e_series, dense31b, a4b }) |json|
        try proof.expectParseEqual(gemma4.config.Config, &gemma4.table, json, &proof.flip_waived);
}

// ── the Gemma 4 assistant drafter on the served trunk ──

/// A two-layer `gemma4_assistant` (sliding, full) over a trunk of width `H`: hidden 64, the trunk's head widths and
/// vocabulary, bf16, written to `dir`.
fn writeDrafter(dir: std.testing.TmpDir, path: []const u8) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    try dir.dir.writeFile(io, .{ .sub_path = "config.json", .data =
        \\{"model_type":"gemma4_assistant","backbone_hidden_size":128,"tie_word_embeddings":true,
        \\"text_config":{"hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":2,
        \\"head_dim":256,"global_head_dim":512,"sliding_window":4,"vocab_size":256,"rms_norm_eps":1e-6,
        \\"layer_types":["sliding_attention","full_attention"],
        \\"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"proportional","partial_rotary_factor":0.25},
        \\"sliding_attention":{"rope_theta":10000.0}}}}
    });
    var d = fixture.Pack.init(0xd4af);
    defer d.deinit();
    var buf: [128]u8 = undefined;
    try d.weight("model.embed_tokens", VOCAB, 64, false);
    try d.norm("model.norm.weight", 64);
    try d.weight("pre_projection", 64, 2 * H, false);
    try d.weight("post_projection", H, 64, false);
    for (0..2) |i| {
        const hd: c_int = if (i == 1) 512 else 256;
        const L = struct {
            fn at(b: []u8, li: usize, leaf: []const u8) ![]const u8 {
                return std.fmt.bufPrint(b, "model.layers.{d}.{s}", .{ li, leaf });
            }
        };
        for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight" }) |nm|
            try d.norm(try L.at(&buf, i, nm), 64);
        try d.norm(try L.at(&buf, i, "self_attn.q_norm.weight"), hd);
        try d.dense(try L.at(&buf, i, "layer_scalar"), &.{1}, 0.2, 0.9);
        try d.weight(try L.at(&buf, i, "self_attn.q_proj"), 2 * hd, 64, false);
        try d.weight(try L.at(&buf, i, "self_attn.o_proj"), 64, 2 * hd, false);
        try d.weight(try L.at(&buf, i, "mlp.gate_proj"), 128, 64, false);
        try d.weight(try L.at(&buf, i, "mlp.up_proj"), 128, 64, false);
        try d.weight(try L.at(&buf, i, "mlp.down_proj"), 64, 128, false);
    }
    const file = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/model.safetensors", .{path}, 0);
    defer testing.allocator.free(file);
    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try mlx.check(mlx.mlx_save_safetensors(file.ptr, d.map, meta));
}

/// A greedy drafter run on `xfm`: every emitted token, each round's accepted count, and the drafter's first logits
/// of each round (its read of the trunk's table, hidden and KV cache).
fn drafterRun(xfm: *Transformer, dir: []const u8, tokens: *std.ArrayList(u32), accepted: *std.ArrayList(u32), logits: *std.ArrayList(mlx.mlx_array)) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var dm = try drafter_mod.loadDrafter(io, a, fixture.stream(), dir);
    defer dm.deinit();
    try dm.bind(xfm);
    var tok_dummy: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17, 2, 28, 99, 140 };
    var gen = try Generator.initWithOptions(io, a, xfm, &tok_dummy, &prompt, 24, .{ .temperature = 0.0 }, &.{}, .{
        .drafter_enabled = true,
        .drafter = &dm,
        .drafter_block_size = 4,
    });
    defer gen.deinit(a);
    while (!gen.done) {
        const out = try drafter_mod.step(&dm, xfm, gen.ctx.cache, gen.next_token_id, gen.last_hidden, @intCast(gen.ctx.cache.step));
        _ = mlx.mlx_array_free(out.h_prev_next);
        try mlx.check(mlx.mlx_array_eval(out.logits));
        try logits.append(a, out.logits);
        const res = (try gen.nextDrafter(a)) orelse break;
        defer a.free(res.tokens);
        try tokens.appendSlice(a, res.tokens);
        try accepted.append(a, res.accepted_tokens);
    }
    try testing.expect(gen.drafter_attempted > 0);
}

/// The drafter over `b`'s legacy trunk and its served shell: tokens, per-round acceptance and the drafter's first
/// logits of each round bit-equal.
fn expectDrafter(b: *proof.Both) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    try testing.expect(b.xfm.plugin.?.drafters);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    try writeDrafter(tmp, dir);
    var tokens: [2]std.ArrayList(u32) = .{ .empty, .empty };
    var accepted: [2]std.ArrayList(u32) = .{ .empty, .empty };
    var logits: [2]std.ArrayList(mlx.mlx_array) = .{ .empty, .empty };
    defer for (&tokens, &accepted, &logits) |*t, *c, *l| {
        t.deinit(a);
        c.deinit(a);
        for (l.items) |x| _ = mlx.mlx_array_free(x);
        l.deinit(a);
    };
    var xb: [2]*Transformer = undefined;
    const xs = b.sides(&xb);
    for (xs, 0..) |x, k| try drafterRun(x, dir, &tokens[k], &accepted[k], &logits[k]);
    const k = xs.len - 1;
    try b.checkIds(tokens[k].items);
    try b.checkIds(accepted[k].items);
    if (k == 1) try testing.expectEqual(logits[0].items.len, logits[1].items.len);
    for (logits[k].items) |got| try b.check(got, false);
}

test "gemma4 plugin: the assistant drafter over a served trunk (E-series, 26B-A4B) emits the legacy tokens, acceptance and draft logits" {
    {
        var p = fixture.Pack.init(0x9e50);
        defer p.deinit();
        try writeGemma4(&p, .{}, 256, 2, true, false);
        var b = try proof.Both.openAs(&gemma4.table, &p, e_series, oracle("drafter-e-series"));
        defer b.close();
        try expectDrafter(&b);
        try b.expectDone();
    }
    var p = fixture.Pack.init(0x9e51);
    defer p.deinit();
    try writeGemma4Moe(&p, .{}, "language_model.model", false);
    var b = try proof.Both.openAs(&gemma4.table, &p, a4b, oracle("drafter-a4b"));
    defer b.close();
    try expectDrafter(&b);
    try b.expectDone();
}

// ── float-quantized packs ──

/// `json` stored in MLX's float `mode` (a pack with no `quantization` block gets one).
fn withQuant(json: []const u8, q: Quant) ![]u8 {
    var buf: [96]u8 = undefined;
    const spec = try std.fmt.bufPrint(&buf, "\"quantization\":{{\"group_size\":{d},\"bits\":{d},\"mode\":\"{s}\"}}", .{ q.group, q.bits, std.mem.span(q.mode) });
    const affine = "\"quantization\":{\"group_size\":64,\"bits\":4}";
    if (std.mem.indexOf(u8, json, affine) != null) return std.mem.replaceOwned(u8, testing.allocator, json, affine, spec);
    return std.fmt.allocPrint(testing.allocator, "{s},{s}}}", .{ json[0 .. json.len - 1], spec });
}

const mxfp4: Quant = .{ .mode = "mxfp4", .bits = 4, .group = 32 };
const nvfp4: Quant = .{ .mode = "nvfp4", .bits = 4, .group = 16 };
const mxfp8: Quant = .{ .mode = "mxfp8", .bits = 8, .group = 32 };

test "gemma4 plugin: float-quantized packs (mxfp4 31B-like, nvfp4 E-series, mxfp8 26B-A4B) are bit-equal to the legacy forward" {
    {
        var p = fixture.Pack.init(0x9e61);
        defer p.deinit();
        try writeGemma4(&p, mxfp4, 0, 0, true, true);
        const json = try withQuant(dense31b, mxfp4);
        defer testing.allocator.free(json);
        var b = try proof.Both.openAs(&gemma4.table, &p, json, oracle("31b-mxfp4"));
        defer b.close();
        try proof.expectSteps(&b);
        try proof.expectGreedy(&b);
        try proof.expectBatched(&b);
        try b.expectDone();
    }
    {
        var p = fixture.Pack.init(0x9e62);
        defer p.deinit();
        try writeGemma4(&p, nvfp4, 256, 2, true, false);
        const json = try withQuant(e_series, nvfp4);
        defer testing.allocator.free(json);
        var b = try proof.Both.openAs(&gemma4.table, &p, json, oracle("e-series-nvfp4"));
        defer b.close();
        try proof.expectSteps(&b);
        try proof.expectGreedy(&b);
        try b.expectDone();
    }
    var p = fixture.Pack.init(0x9e63);
    defer p.deinit();
    try writeGemma4Moe(&p, mxfp8, "language_model.model", false);
    const json = try withQuant(a4b, mxfp8);
    defer testing.allocator.free(json);
    var b = try proof.Both.openAs(&gemma4.table, &p, json, oracle("a4b-mxfp8"));
    defer b.close();
    try proof.expectSteps(&b);
    try proof.expectGreedy(&b);
    try b.expectDone();
}

/// fp4 experts: a prefill and the verify after it bit-equal to the legacy forward (its sorted gather); a decode step
/// equal to the plugin's own verify row at that position. The legacy's one- and two-row fp4 expert kernel applies
/// SwiGLU, not Gemma's GeGLU, so its decode is no oracle.
fn expectFp4Experts(b: *proof.Both) !void {
    const n = b.cfg.num_hidden_layers;
    var xb: [2]*Transformer = undefined;
    const xs = b.sides(&xb);
    var g = ops.Graph.init(testing.allocator, b.xfm.s);
    defer g.deinit();
    const prompt = [_]i32{ 3, 17, 42, 99, 7, 250, 1, 8, 66, 31, 4, 5 };
    var outs: [2][2]mlx.mlx_array = undefined;
    for (xs, 0..) |x, k| {
        var s: proof.Slot = undefined;
        try s.init(n);
        defer s.deinit();
        outs[k][0] = try g.adopt(try x.forwardWith(&s.ctx, try fixture.tokens(&g, &prompt)));
        s.ctx.spec_verify = true;
        outs[k][1] = try g.adopt(try x.forwardWith(&s.ctx, try fixture.tokens(&g, &.{ 88, 5, 61 })));
    }
    const got = outs[xs.len - 1];
    for (0..2) |i| try b.check(got[i], false);
    var s: proof.Slot = undefined;
    try s.init(n);
    defer s.deinit();
    _ = try g.adopt(try b.xfm.forwardWith(&s.ctx, try fixture.tokens(&g, &prompt)));
    const dec = try g.adopt(try b.xfm.forwardWith(&s.ctx, try fixture.tokens(&g, &.{88})));
    try ops.expectBitEqual(dec, try g.sliceAxis(got[1], 1, 0, 1));
}

test "gemma4 plugin: ternary 2-bit packs (31B-like, E-series, 26B-A4B) arm qmv2's kernels and are bit-equal to the legacy forward" {
    inline for (.{ .{ "31b", 0x9e66 }, .{ "e-series", 0x9e67 }, .{ "a4b", 0x9e68 } }) |cs| {
        var p = fixture.Pack.init(cs[1]);
        defer p.deinit();
        const kind = cs[0];
        const base = if (comptime std.mem.eql(u8, kind, "31b")) dense31b else if (comptime std.mem.eql(u8, kind, "e-series")) e_series else a4b;
        if (comptime std.mem.eql(u8, kind, "31b")) try writeGemma4(&p, proof.ternary, 0, 0, true, true) else if (comptime std.mem.eql(u8, kind, "e-series")) try writeGemma4(&p, proof.ternary, 256, 2, true, false) else try writeGemma4Moe(&p, proof.ternary, "language_model.model", false);
        const json = if (std.mem.indexOf(u8, base, proof.q4_json) != null)
            try std.mem.replaceOwned(u8, testing.allocator, base, proof.q4_json, proof.q2_json)
        else
            try std.fmt.allocPrint(testing.allocator, "{s},{s}}}", .{ base[0 .. base.len - 1], proof.q2_json });
        defer testing.allocator.free(json);
        var b = try proof.Both.openAs(&gemma4.table, &p, json, oracle(kind ++ "-ternary"));
        defer b.close();
        const m: *const gemma4.Loaded = @ptrCast(@alignCast(b.xfm.plugin.?.model));
        try testing.expect(m.layers[0].attn.o.ternary);
        try proof.expectSteps(&b);
        try proof.expectGreedy(&b);
        try b.expectDone();
    }
}

test "gemma4 plugin: fp4 experts (mxfp4, nvfp4 26B-A4B) prefill and verify as the legacy forward, decode as their own verify row" {
    inline for (.{ .{ mxfp4, "a4b-mxfp4", 0x9e64 }, .{ nvfp4, "a4b-nvfp4", 0x9e65 } }) |c| {
        var p = fixture.Pack.init(c[2]);
        defer p.deinit();
        try writeGemma4Moe(&p, c[0], "language_model.model", false);
        const json = try withQuant(a4b, c[0]);
        defer testing.allocator.free(json);
        var b = try proof.Both.openAs(&gemma4.table, &p, json, oracle(c[1]));
        defer b.close();
        try expectFp4Experts(&b);
        try b.expectDone();
    }
}

// ── declines the legacy path refused too ──

test "gemma4 plugin: an NVIDIA ModelOpt NVFP4 pack is declined, and the legacy load refuses it by name" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var p = fixture.Pack.init(0x9e70);
    defer p.deinit();
    try writeGemma4(&p, null, 0, 0, true, true);
    // nvidia/Gemma-4-31B-IT-NVFP4's MLP layout: packed codes, e4m3 group scales and the per-tensor weight_scale_2.
    const m = "model.language_model.layers.0.mlp.gate_proj.";
    try p.denseAs(m ++ "weight", &.{ INTER, H / 2 }, 100, 120, .uint8);
    try p.denseAs(m ++ "weight_scale", &.{ INTER, H / 16 }, 10, 60, .uint8);
    try p.denseAs(m ++ "weight_scale_2", &.{1}, 0.001, 0.002, .float32);
    const json = try std.fmt.allocPrint(a, "{s},\"quantization_config\":{{\"quant_method\":\"modelopt\",\"quant_algo\":\"NVFP4\"}}}}", .{dense31b[0 .. dense31b.len - 1]});
    defer a.free(json);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    const out = try plugin_config.resolve(io, a, dir, json);
    try testing.expectEqualStrings("gemma: a quantization_config naming a quant_method", out.declined);
}

// Bar: LTX 2.5's text encoder (a left-padded prompt under its additive mask, every layer's residual captured) gives
// the states the legacy forward recorded in `fixtures/ltx-mask.tape`, bit for bit.
test "gemma4 plugin: LTX 2.5's text encoder states over a left-padded prompt replay the legacy forward's" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const tape_mod = @import("../../sdk/testing/tape.zig");
    var p = fixture.Pack.init(0x9e53);
    defer p.deinit();
    try writeGemma4(&p, null, 0, 0, true, true);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    try p.save(dir);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = dense31b });
    const ids = [_]i32{ 0, 0, 0, 0, 7, 19, 42, 5, 88, 3, 61, 200 };
    const states = try @import("../ltx/video.zig").gemmaCapture4(io, a, dir, &ids, 0, fixture.stream());
    defer {
        for (states) |x| _ = mlx.mlx_array_free(x);
        a.free(states);
    }
    var tape = try tape_mod.Tape.player(@embedFile("fixtures/ltx-mask.tape"));
    for (states) |x| try tape.expectNext(x, false);
    try tape.expectDone();
}
