//! EmbeddingGemma proven against the legacy engine on a tiny 4-bit pack (the Gemma 3 trunk, a band narrower than the
//! sequence, a two-layer projection head), through a served shell: the hidden rows of one sequence and of a padded
//! batch under its key mask, the projection of pooled rows and `/v1/embeddings`' vectors, bit-equal to the legacy
//! encoder's (replayed from `fixtures/*.tape`); the parse equal to the engine's on the shell and the pooling contract.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const pooling = @import("../../engines/mlx/embed/pooling.zig");
const proof = @import("../lfm2/proof.zig");
const gemma_proof = @import("../gemma3/proof_test.zig");
const Both = gemma_proof.Both;
const Transformer = @import("../../engines/mlx/transformer.zig").Transformer;
const encoder = @import("../gemma4/bidirectional.zig");
const embedding_gemma = @import("plugin.zig");

const testing = std.testing;
const Array = mlx.mlx_array;


fn oracle(comptime name: []const u8) gemma_proof.Oracle {
    const rel = "fixtures/" ++ name ++ ".tape";
    return .{ .tape = @embedFile(rel) };
}

pub const H = 128;
const INTER = 256;
const HD = 256;

const config_json =
    \\{"model_type":"gemma3_text","use_bidirectional_attention":true,"hidden_size":128,"intermediate_size":256,
    \\"num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":256,"vocab_size":256,
    \\"sliding_window":4,"sliding_window_pattern":2,"query_pre_attn_scalar":256,"rms_norm_eps":1e-6,
    \\"max_position_embeddings":64,"bos_token_id":2,"eos_token_id":1,"pooling_mode":"mean",
    \\"quantization":{"group_size":64,"bits":4}}
;

fn writePack(p: *fixture.Pack, q: ?gemma_proof.Quant) !void {
    var nb: [128]u8 = undefined;
    try gemma_proof.put(p, "model.embed_tokens", &.{ 256, H }, q);
    try p.norm("model.norm.weight", H);
    try gemma_proof.put(p, "dense.0", &.{ 2 * H, H }, q);
    try gemma_proof.put(p, "dense.1", &.{ H, 2 * H }, q);
    for (0..4) |i| {
        const at = struct {
            fn f(b: []u8, li: usize, leaf: []const u8) ![]const u8 {
                return std.fmt.bufPrint(b, "model.layers.{d}.{s}", .{ li, leaf });
            }
        }.f;
        for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight" }) |nm|
            try p.norm(try at(&nb, i, nm), H);
        try p.norm(try at(&nb, i, "self_attn.q_norm.weight"), HD);
        try p.norm(try at(&nb, i, "self_attn.k_norm.weight"), HD);
        try gemma_proof.put(p, try at(&nb, i, "self_attn.q_proj"), &.{ 2 * HD, H }, q);
        try gemma_proof.put(p, try at(&nb, i, "self_attn.k_proj"), &.{ HD, H }, q);
        try gemma_proof.put(p, try at(&nb, i, "self_attn.v_proj"), &.{ HD, H }, q);
        try gemma_proof.put(p, try at(&nb, i, "self_attn.o_proj"), &.{ H, 2 * HD }, q);
        try gemma_proof.put(p, try at(&nb, i, "mlp.gate_proj"), &.{ INTER, H }, q);
        try gemma_proof.put(p, try at(&nb, i, "mlp.up_proj"), &.{ INTER, H }, q);
        try gemma_proof.put(p, try at(&nb, i, "mlp.down_proj"), &.{ H, INTER }, q);
    }
}

/// The served embedding path against the legacy encoder: the hidden rows of one sequence and of a padded batch under
/// its key mask, the projection of the batch's mean-pooled rows, `/v1/embeddings`' vectors (pooling and
/// `EmbedHooks.project` included); a decode step refused.
pub fn expectServed(b: *Both, seq: []const u32, batch: []const i32, lengths: []const usize) !void {
    const a = testing.allocator;
    var xb: [2]*Transformer = undefined;
    const xs = b.sides(&xb);
    const k = xs.len - 1;
    var g = ops.Graph.init(a, b.xfm.s);
    defer g.deinit();
    const t: c_int = @intCast(batch.len / lengths.len);
    const one = try g.astype(try g.fromSlice(u32, seq, &.{ 1, @intCast(seq.len) }), .int32);
    const ids = try g.fromSlice(i32, batch, &.{ @intCast(lengths.len), t });
    const mask = try g.adopt(try pooling.buildKeyPadMask(a, lengths, @intCast(t), b.xfm.s));
    var outs: [2][3]Array = undefined;
    for (xs, 0..) |x, i| {
        outs[i][0] = try g.adopt(try x.forwardEmbeddingMasked(one, null, null));
        outs[i][1] = try g.adopt(try x.forwardEmbeddingMasked(ids, mask, null));
        const pooled = try g.adopt(try pooling.maskedMeanPool(a, outs[i][1], lengths, b.xfm.s));
        outs[i][2] = try g.adopt(try x.embedProjection(pooled));
    }
    for (0..3) |j| try b.check(outs[k][j], false);
    const pb = b.xfm.plugin.?;
    var refused: abi.ForwardOut = .{};
    try testing.expectEqual(abi.Verdict.Code.unsupported, pb.table.forward(pb.model, &.{ .mode = .decode, .rows = 1, .tokens = .of(seq[0..1]), .positions = .of(&.{0}) }, &refused).code);
    const seqs = [_][]const u32{ &.{ 2, 17, 42, 99, 7, 250, 31, 8, 66, 1 }, &.{ 2, 9, 33, 77, 140, 5, 61, 1 } };
    var vecs: [2][][]f32 = undefined;
    for (xs, 0..) |x, i| vecs[i] = try pooling.computeEmbeddingsBatch(a, x, &seqs);
    defer for (vecs[0..xs.len]) |v| {
        for (v) |r| a.free(r);
        a.free(v);
    };
    for (vecs[k]) |got| try b.checkSlice(f32, got, false);
}

test "embedding_gemma plugin: a 4-bit pack's hidden rows (one sequence, a padded batch), projection and vectors through a served shell equal the legacy encoder's" {
    var p = fixture.Pack.init(0xe3b0);
    defer p.deinit();
    try writePack(&p, .{});
    var b = try Both.openAs(&embedding_gemma.table, &p, config_json, oracle("4bit"));
    defer b.close();
    try testing.expect(b.cfg.use_bidirectional_attention and b.xfm.plugin != null and b.xfm.hasEmbedProjection());
    // The short row's padding stays inside the band of its last real key: without the floor (EmbeddingGemma 2's) a
    // padded query seeing only padding is NaN, and so is the row, on both paths.
    try expectServed(&b, &.{ 2, 17, 42, 99, 7, 250, 31, 8, 66, 9, 4, 1 }, &.{ 2, 17, 42, 99, 7, 250, 31, 8, 66, 1, 2, 9, 33, 77, 140, 5, 61, 1, 0, 0 }, &.{ 10, 8 });
    try b.expectDone();
}

test "embedding_gemma plugin: the parse equals the engine's on the shell, the encoder flag and the pooling contract" {
    var want = try proof.legacyConfig(config_json);
    defer want.deinit(testing.allocator);
    var p: abi.Parsed = .{};
    try testing.expect(embedding_gemma.config.parseJson(config_json, &p).isOk());
    defer embedding_gemma.config.freeParsed(&p);
    try testing.expect(want.is_encoder_only and p.caps.embed);
    try testing.expectEqualStrings(@tagName(want.pooling_mode.?), @tagName(p.embed.pooling));
    try testing.expectEqual(want.bos_token_id.?, p.shell.bos);
    try testing.expectEqual(want.rope_scaling_factor, embedding_gemma.config.of(&p).rope_scaling_factor);
}

test "embedding_gemma plugin: a ternary 2-bit pack arms qmv2's kernels and equals the legacy encoder through a served shell" {
    var p = fixture.Pack.init(0xe3b2);
    defer p.deinit();
    try writePack(&p, gemma_proof.ternary);
    const json = try std.mem.replaceOwned(u8, testing.allocator, config_json, gemma_proof.q4_json, gemma_proof.q2_json);
    defer testing.allocator.free(json);
    var b = try Both.openAs(&embedding_gemma.table, &p, json, oracle("ternary"));
    defer b.close();
    const m: *const encoder.Loaded = @ptrCast(@alignCast(b.xfm.plugin.?.model));
    try testing.expect(m.layers[0].attn.o.ternary and m.base.weights.projection.?.first.ternary);
    try expectServed(&b, &.{ 2, 17, 42, 99, 7, 250, 31, 8, 66, 9, 4, 1 }, &.{ 2, 17, 42, 99, 7, 250, 31, 8, 66, 1, 2, 9, 33, 77, 140, 5, 61, 1, 0, 0 }, &.{ 10, 8 });
    try b.expectDone();
}
