//! Qwen3-ASR's trunk served through the plugin driver against the legacy shell over one tiny pack: a
//! transcription-shaped prompt whose audio rows splice at the placeholders across a chunk boundary, and the aligner's read (hidden rows, the head the shell borrows) bit-equal.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const q3proof = @import("../qwen3/proof_test.zig");
const served = @import("../qwen3_5/served_test.zig");
const qwen3_asr = @import("plugin.zig");

const testing = std.testing;
const Array = mlx.mlx_array;

const AUDIO: i32 = 250;

fn asrJson(a: std.mem.Allocator, g: q3proof.Geo) ![]u8 {
    const text = try q3proof.configJson(a, g);
    defer a.free(text);
    // The pack's quantization sits at the outer root, the trunk under the thinker.
    const quant = std.mem.indexOf(u8, text, ",\"quantization\"");
    const trunk = if (quant) |i| try std.mem.concat(a, u8, &.{ text[0..i], "}" }) else try a.dupe(u8, text);
    defer a.free(trunk);
    return std.mem.concat(a, u8, &.{
        "{\"model_type\":\"qwen3_asr\",",
        if (quant) |i| text[i + 1 .. text.len - 1] else "",
        if (quant != null) "," else "",
        "\"thinker_config\":{\"model_type\":\"qwen3_asr\",\"audio_token_id\":250,",
        "\"audio_config\":{\"d_model\":64,\"encoder_layers\":1,\"encoder_attention_heads\":2,\"encoder_ffn_dim\":64,",
        "\"num_mel_bins\":16,\"downsample_hidden_size\":8,\"n_window\":2,\"n_window_infer\":4,\"output_dim\":128},",
        "\"text_config\":",
        trunk,
        "}}",
    });
}

fn open(p: *served.Pair, g: q3proof.Geo, comptime seed: u64) !void {
    var pk = fixture.Pack.init(seed);
    defer pk.deinit();
    try q3proof.writePack(&pk, g);
    const json = try asrJson(testing.allocator, g);
    defer testing.allocator.free(json);
    try p.open(&qwen3_asr.table, &pk, json, try served.oracle("qwen3_asr", std.fmt.comptimePrint("served-{x}", .{seed})));
}


test "qwen3_asr served: audio rows spliced across a chunk boundary, then the aligner's hidden rows and head read" {
    const a = testing.allocator;
    var p: served.Pair = undefined;
    try open(&p, .{ .quantized = true }, 0xa5a3);
    defer p.close();
    const s = fixture.stream();
    var g0 = ops.Graph.init(a, s);
    defer g0.deinit();
    const rows = try g0.astype(try g0.adopt(try served.fixtureRows(5, 128, 7)), .bfloat16);
    const tokens = [_]i32{ 3, 17, AUDIO, AUDIO, AUDIO, AUDIO, AUDIO, 42, 9, 11 };
    const xs = p.shells();
    var lane_buf: [2]served.Lane = undefined;
    const lanes = lane_buf[0..xs.len];
    for (lanes, xs) |*l, x| {
        try l.init(x);
        l.ctx.vision_embeddings = rows;
    }
    defer for (lanes) |*l| l.deinit();
    const chunks = [_]struct { ids: []const i32, offset: usize = 0 }{
        .{ .ids = tokens[0..4] },
        .{ .ids = tokens[4..], .offset = 2 },
        .{ .ids = &.{88} },
    };
    for (chunks) |ch| {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        const t = try fixture.tokens(&g, ch.ids);
        var outs: [2]Array = undefined;
        for (lanes, xs, outs[0..xs.len]) |*l, x, *o| {
            l.ctx.vision_splice_offset = ch.offset;
            o.* = try g.adopt(try x.forwardWith(&l.ctx, t));
        }
        try p.check(outs[0..xs.len]);
    }
    // The aligner's pass: one prompt, hidden rows (no head), the head over picked rows (`forced_align.timestampClasses`).
    var g = ops.Graph.init(a, s);
    defer g.deinit();
    const t = try fixture.tokens(&g, &tokens);
    var outs: [2]Array = undefined;
    for (xs, outs[0..xs.len]) |x, *o| {
        try x.resetCache();
        var ctx = x.defaultCtx();
        ctx.vision_embeddings = rows;
        ctx.skip_lm_head = true;
        const hidden = try g.adopt(try x.forwardWith(&ctx, t));
        const picked = try g.sliceAxis(hidden, 1, 7, 10);
        o.* = try g.adopt(try x.lmHeadForDraft(picked));
    }
    try p.check(outs[0..xs.len]);
    try p.oracle.done();
}
