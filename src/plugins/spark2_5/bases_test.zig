//! The bases `DecoderAttention(.{ .sliding, .gate = .head })` block (Spark's per-head sigmoid gate) on a hd-32 Spark
//! pack, built from this plugin's loaded weights, against a tape recorded from the legacy dense forward (now deleted):
//! steps across the window and batched rows.
//! copied_from: sdk/bases/proof/attention_test.zig (`Spark`, `spark_config`, `writeSpark`, `batchedCase`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const both = @import("../../engines/mlx/plugin/both.zig");
const cache_ops = @import("../../engines/mlx/kv/cache_ops.zig");
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const plugin = @import("plugin.zig");

const testing = std.testing;
const Array = mlx.mlx_array;
const b = bases.blocks;

pub const Spark = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = b.decoder_attention.DecoderAttention(.{ .sliding = true, .gate = .head }),
    .mlp = b.extras.GatedGelu(.erf),
    .seam = b.seam.ResidualTail,
    .head = b.head.LmHead,
});

const H: c_int = 128;
const VOCAB: c_int = 256;
const INTER: c_int = 256;

/// Three sliding layers and a full one with 25% rotary; one joined q|k|v projection split at load.
const spark_config =
    \\{"model_type":"spark2_5","hidden_size":128,"intermediate_size":256,"num_hidden_layers":4,"num_attention_heads":4,
    \\"num_key_value_heads":2,"head_dim":32,"vocab_size":256,"rms_norm_eps":1e-6,"sliding_window":4,
    \\"layer_types":["sliding_attention","sliding_attention","sliding_attention","full_attention"],
    \\"rope_parameters":{"full_attention":{"rope_theta":5000000.0,"partial_rotary_factor":0.25},
    \\"sliding_attention":{"rope_theta":10000.0}},"tie_word_embeddings":true,"max_position_embeddings":4096,
    \\"quantization":{"group_size":64,"bits":4}}
;

fn writeSpark(p: *fixture.Pack) !void {
    var buf: [160]u8 = undefined;
    try p.weight("model.embedding", VOCAB, H, true);
    try p.norm("model.norm.weight", H);
    for (0..4) |i| {
        for ([_]struct { []const u8, c_int, c_int }{
            .{ "self_attn.q_k_v_proj", 4 * 32 + 2 * 2 * 32, H },
            .{ "self_attn.out_proj", H, 4 * 32 },
            .{ "self_attn.g_proj", 4, H },
            .{ "mlp.gate_proj", INTER, H },
            .{ "mlp.up_proj", INTER, H },
            .{ "mlp.down_proj", H, INTER },
        }) |t| {
            if (std.mem.eql(u8, t[0], "self_attn.q_k_v_proj")) {
                try p.norm(try std.fmt.bufPrint(&buf, "model.layers.{d}.input_layernorm.weight", .{i}), H);
                try p.norm(try std.fmt.bufPrint(&buf, "model.layers.{d}.post_attention_layernorm.weight", .{i}), H);
            }
            try p.weight(try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}", .{ i, t[0] }), t[1], t[2], true);
        }
    }
}

/// The plugin's loaded layers as the bases block's (the same projections, joins and geometry).
fn sparkWeights(m: *const plugin.Loaded, layers: []Spark.Layer) Spark.Weights {
    const pw = m.base.weights;
    for (layers, pw.layers) |*L, *pl| {
        const a = &pl.mixer;
        L.* = .{
            .input_norm = pl.input_norm,
            .mixer = .{ .q = a.q, .k = a.k, .v = a.v, .o = a.o, .qkv = a.qkv, .geo = a.geo, .window = a.window, .out_gate = a.out_gate },
            .post_norm = pl.post_norm,
            .mlp = pl.mlp,
        };
    }
    return .{ .embed = pw.embed, .layers = layers, .final_norm = pw.final_norm, .head = pw.head };
}

fn open(p: *fixture.Pack, comptime seed: u64) !both.Both {
    const name = std.fmt.comptimePrint("fixtures/bases-{x}.tape", .{seed});
    return both.Both.openAs(&plugin.table, p, spark_config, false, .{ .tape = @embedFile(name) });
}

test "spark2_5 bases: the per-head sigmoid gate and sliding layers are bit-equal to the recorded forwardStandardWith" {
    const a = testing.allocator;
    var p = fixture.Pack.init(0x5a2c);
    defer p.deinit();
    try writeSpark(&p);
    var bt = try open(&p, 0x5a2c);
    defer bt.close();
    const m: *const plugin.Loaded = @ptrCast(@alignCast(bt.xfm.plugin.?.model));
    var layers: [4]Spark.Layer = undefined;
    const w = sparkWeights(m, &layers);
    var kvc = try KVCache.init(a, 4);
    defer kvc.deinit();
    var slot: cache_ops.Slot = .{ .kv = &kvc };
    const handles = [_]abi.Handle{slot.handle()};
    var pos: u32 = 0;
    for (both.long_steps) |st| {
        var g = ops.Graph.init(a, bt.xfm.s);
        defer g.deinit();
        const ids = try fixture.tokens(&g, st.ids);
        const got = try Spark.forward(&w, &m.base.shared, &g, ids, .{ .mode = st.mode, .rows = @intCast(st.ids.len), .positions = &.{pos}, .caches = &handles, .cache_ops = &cache_ops.ops, .logits = .all_rows });
        try bt.check(got.logits, false);
        pos += @intCast(st.ids.len);
    }
    try bt.expectDone();
}

test "spark2_5 bases: batched rows are bit-equal to the recorded forwardBatchedDecode" {
    const a = testing.allocator;
    var p = fixture.Pack.init(0x5a2d);
    defer p.deinit();
    try writeSpark(&p);
    var bt = try open(&p, 0x5a2d);
    defer bt.close();
    const m: *const plugin.Loaded = @ptrCast(@alignCast(bt.xfm.plugin.?.model));
    var layers: [4]Spark.Layer = undefined;
    const w = sparkWeights(m, &layers);
    var base_kv: [2]KVCache = .{ try KVCache.init(a, 4), try KVCache.init(a, 4) };
    defer for (&base_kv) |*c| c.deinit();
    var slots: [2]cache_ops.Slot = .{ .{ .kv = &base_kv[0] }, .{ .kv = &base_kv[1] } };
    const handles = [_]abi.Handle{ slots[0].handle(), slots[1].handle() };
    const prompts = [_][]const i32{ &.{ 4, 8, 15, 16, 23, 42, 9 }, &.{ 42, 101, 7 } };
    var positions: [2]u32 = undefined;
    for (prompts, 0..) |pr, i| {
        var g = ops.Graph.init(a, bt.xfm.s);
        defer g.deinit();
        const ids = try fixture.tokens(&g, pr);
        const got = try Spark.forward(&w, &m.base.shared, &g, ids, .{ .mode = .prefill, .rows = @intCast(pr.len), .positions = &.{0}, .caches = handles[i .. i + 1], .cache_ops = &cache_ops.ops, .logits = .all_rows });
        try bt.check(got.logits, false);
        positions[i] = @intCast(pr.len);
    }
    for ([_][2]u32{ .{ 77, 3 }, .{ 190, 64 }, .{ 11, 12 }, .{ 13, 250 } }) |next| {
        var g = ops.Graph.init(a, bt.xfm.s);
        defer g.deinit();
        const ids = try g.fromSlice(i32, &.{ @intCast(next[0]), @intCast(next[1]) }, &.{ 2, 1 });
        const got = try Spark.forward(&w, &m.base.shared, &g, ids, .{ .mode = .batched_rows, .rows = 1, .slots = 2, .positions = &positions, .caches = &handles, .cache_ops = &cache_ops.ops, .logits = .last_row });
        for (0..2) |i| try bt.check(try g.sliceAxis(got.logits, 0, @intCast(i), @intCast(i + 1)), false);
        for (&positions) |*pp| pp.* += 1;
    }
    try bt.expectDone();
}
