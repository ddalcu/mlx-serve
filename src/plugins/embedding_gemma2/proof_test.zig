//! EmbeddingGemma 2 proven against the legacy engine on tiny packs (4-bit and bf16; sliding layers narrower than the
//! sequence, wider full layers, per-layer inputs, layer scalars, the projection head), through a served shell: the
//! hidden rows of one sequence, of a padded batch under its key mask, of a prompt with image rows spliced at their
//! placeholders, the projection of pooled rows and `/v1/embeddings`' vectors, bit-equal to the legacy encoder's
//! (replayed from `fixtures/*.tape`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const served = @import("../embedding_gemma/proof_test.zig");
const gemma_proof = @import("../gemma3/proof_test.zig");
const Both = gemma_proof.Both;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const embedding_gemma2 = @import("plugin.zig");

const testing = std.testing;


fn oracle(comptime name: []const u8) gemma_proof.Oracle {
    const rel = "fixtures/" ++ name ++ ".tape";
    return .{ .tape = @embedFile(rel) };
}

const H = 128;
const INTER = 256;
const PD = 64;
const LAYERS = 4;
const E = 128;

const config_json =
    \\{"model_type":"embedding_gemma2","image_token_id":250,"video_token_id":251,"audio_token_id":252,
    \\"text_config":{"model_type":"embedding_gemma2_text","vocab_size":256,"hidden_size":128,"intermediate_size":256,
    \\"num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":256,"max_position_embeddings":64,
    \\"rms_norm_eps":1e-6,"eos_token_id":1,"bos_token_id":2,"sliding_window":2,"hidden_size_per_layer_input":64,
    \\"embedding_dim":128,"layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],
    \\"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"default"},
    \\"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}},
    \\"per_layer_config":{"01":{"head_dim":512,"num_key_value_heads":1},"03":{"head_dim":512,"num_key_value_heads":1}}},
    \\"vision_config":{"model_type":"gemma4_vision","hidden_size":24},
    \\"quantization":{"group_size":64,"bits":4}}
;

fn writePack(p: *fixture.Pack, q: ?gemma_proof.Quant) !void {
    var nb: [128]u8 = undefined;
    const pre = "language_model.";
    try gemma_proof.put(p, pre ++ "embed_tokens", &.{ 256, H }, q);
    try p.norm(pre ++ "norm.weight", H);
    try gemma_proof.put(p, pre ++ "embedding_projection", &.{ E, H }, q);
    try gemma_proof.put(p, pre ++ "ple.per_layer_model_projection", &.{ LAYERS * PD, H }, q);
    try p.norm(pre ++ "ple.per_layer_projection_norm.weight", PD);
    for (0..LAYERS) |i| {
        const at = struct {
            fn f(b: []u8, li: usize, leaf: []const u8) ![]const u8 {
                return std.fmt.bufPrint(b, "language_model.layers.{d}.{s}", .{ li, leaf });
            }
        }.f;
        const hd: c_int = if (i % 2 == 1) 512 else 256;
        for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "ple_block.post_per_layer_input_norm.weight" }) |nm|
            try p.norm(try at(&nb, i, nm), H);
        try p.dense(try at(&nb, i, "layer_scalar"), &.{1}, 0.2, 0.9);
        try p.norm(try at(&nb, i, "self_attn.q_norm.weight"), hd);
        try p.norm(try at(&nb, i, "self_attn.k_norm.weight"), hd);
        try gemma_proof.put(p, try at(&nb, i, "self_attn.q_proj"), &.{ 2 * hd, H }, q);
        try gemma_proof.put(p, try at(&nb, i, "self_attn.k_proj"), &.{ hd, H }, q);
        try gemma_proof.put(p, try at(&nb, i, "self_attn.v_proj"), &.{ hd, H }, q);
        try gemma_proof.put(p, try at(&nb, i, "self_attn.o_proj"), &.{ H, 2 * hd }, q);
        try gemma_proof.put(p, try at(&nb, i, "mlp.gate_proj"), &.{ INTER, H }, q);
        try gemma_proof.put(p, try at(&nb, i, "mlp.up_proj"), &.{ INTER, H }, q);
        try gemma_proof.put(p, try at(&nb, i, "mlp.down_proj"), &.{ H, INTER }, q);
        try gemma_proof.put(p, try at(&nb, i, "ple_block.per_layer_input_gate"), &.{ PD, H }, q);
        try gemma_proof.put(p, try at(&nb, i, "ple_block.per_layer_projection"), &.{ H, PD }, q);
    }
}

/// A prompt with three image rows at the placeholder (250) through the served shell, bit-equal to the legacy splice.
fn expectImageRows(b: *Both) !void {
    const a = testing.allocator;
    var xb: [2]*Transformer = undefined;
    const xs = b.sides(&xb);
    var g = ops.Graph.init(a, b.xfm.s);
    defer g.deinit();
    const toks = [_]u32{ 2, 17, 250, 250, 250, 42, 99, 7, 1 };
    var vals: [3 * H]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0x1a6e);
    for (&vals) |*v| v.* = prng.random().float(f32) - 0.5;
    const rows = try g.astype(try g.fromSlice(f32, &vals, &.{ 1, 3, H }), .bfloat16);
    const ids = try g.astype(try g.fromSlice(u32, &toks, &.{ 1, toks.len }), .int32);
    var outs: [2]mlx.mlx_array = undefined;
    for (xs, 0..) |x, k| outs[k] = try g.adopt(try x.forwardEmbeddingMasked(ids, null, rows));
    const got = outs[xs.len - 1];
    try b.check(got, false);
    // The rows reach the forward: without them the same prompt reads the placeholder's embedding.
    const plain = try g.adopt(try b.xfm.forwardEmbeddingMasked(ids, null, null));
    const eq = try g.call(mlx.mlx_array_equal, .{ plain, got, false });
    try mlx.check(mlx.mlx_array_eval(eq));
    var same = true;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    try testing.expect(!same);
}

const seq = [_]u32{ 2, 17, 42, 99, 7, 250, 31, 8, 66, 9, 4, 1 };
const batch = [_]i32{ 2, 17, 42, 99, 7, 200, 31, 8, 66, 1, 2, 9, 33, 77, 1, 0, 0, 0, 0, 0 };

test "embedding_gemma2 plugin: a 4-bit pack through a served shell (one sequence, a padded batch, image rows, the head, the vectors) equals the legacy encoder" {
    var p = fixture.Pack.init(0xe9e2);
    defer p.deinit();
    try writePack(&p, .{});
    var b = try Both.openAs(&embedding_gemma2.table, &p, config_json, oracle("4bit"));
    defer b.close();
    try testing.expect(b.cfg.gv_budgeted and b.xfm.plugin != null);
    // The short row's padding runs past its band: the floored mask keeps it finite on both paths.
    try served.expectServed(&b, &seq, &batch, &.{ 10, 5 });
    try expectImageRows(&b);
    try b.expectDone();
}

test "embedding_gemma2 plugin: a bf16 pack through a served shell equals the legacy encoder" {
    var p = fixture.Pack.init(0xe9e3);
    defer p.deinit();
    try writePack(&p, null);
    const json = try std.mem.replaceOwned(u8, testing.allocator, config_json, ",\n\"quantization\":{\"group_size\":64,\"bits\":4}", "");
    defer testing.allocator.free(json);
    try testing.expect(json.len < config_json.len);
    var b = try Both.openAs(&embedding_gemma2.table, &p, json, oracle("bf16"));
    defer b.close();
    try served.expectServed(&b, &seq, &batch, &.{ 10, 5 });
    try b.expectDone();
}

test "embedding_gemma2 plugin: a pack the plugin declines is refused by name (no legacy forward to fall back to)" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const json = try std.mem.replaceOwned(u8, a, config_json, "\"embedding_dim\":128,", "\"embedding_dim\":128,\"num_kv_shared_layers\":2,");
    defer a.free(json);
    try testing.expect(json.len > config_json.len);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    try testing.expectError(error.PluginDeclinedNoLegacy, @import("../../engines/mlx/plugin/config.zig").bindServed(io, a, dir, json));
}
