//! Nemotron 3 Diarization on the media table: no route of its own; the host's `diarized_json` transcription hands it
//! the whole clip (`inputs[0]`, f32 mono PCM at 16 kHz) once, so a voice keeps its letter across the 180 s pieces.
//! The result is the speaker count (u32 little-endian) then f32 probabilities [frames][speakers], one row per 10 ms.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const mlx = @import("../../mlx/ffi.zig");
const diarization = @import("diarization.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.MediaTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .required_file = abi.Str.of(config.required_file),
    .modality = .diarization,
    .caps = &config.caps,
    .bill_load = &config.billLoad,
    .bill_request = &config.billRequest,
    .load = &load,
    .unload = &unload,
    .step = &step,
    .encode = &encode,
};

const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;

const Engine = struct {
    d: *diarization.Engine,
    stream: mlx.mlx_stream,
    owns_stream: bool,
    result: []u8 = &.{},
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    const owns = ctx.stream.ctx == null;
    const s: mlx.mlx_stream = if (owns) mlx.mlx_default_gpu_stream_new() else .{ .ctx = ctx.stream.ctx };
    const d = diarization.Engine.loadOn(io, gpa, ctx.model_dir.slice(), s) catch |err| {
        if (owns) _ = mlx.mlx_stream_free(s);
        out.* = Verdict.named(.failed, @errorName(err));
        return null;
    };
    const e = gpa.create(Engine) catch {
        d.deinit();
        if (owns) _ = mlx.mlx_stream_free(s);
        out.* = Verdict.refuse(.failed, "nemotron_diarization: out of memory");
        return null;
    };
    e.* = .{ .d = d, .stream = s, .owns_stream = owns };
    return e;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const e: *Engine = @ptrCast(@alignCast(ep));
    e.d.deinit();
    if (e.owns_stream) _ = mlx.mlx_stream_free(e.stream);
    gpa.free(e.result);
    gpa.destroy(e);
}

fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    const inputs = req.inputs.slice();
    if (inputs.len != 1 or inputs[0].len % @sizeOf(f32) != 0) return Verdict.refuse(.unsupported, "diarization takes one input: the clip as f32 mono PCM at 16 kHz");
    // The upload's bytes carry no alignment promise.
    const pcm = gpa.alloc(f32, inputs[0].len / @sizeOf(f32)) catch return Verdict.refuse(.busy, "out of memory");
    defer gpa.free(pcm);
    @memcpy(std.mem.sliceAsBytes(pcm), inputs[0].slice());
    if (progress.boundary(progress.ctx)) return job.client_left;
    const probs = e.d.diarize(gpa, pcm) catch |err| return job.failure(err, "speaker diarization failed");
    defer gpa.free(probs);
    progress.report(progress.ctx, .of("diarize"), 1, 1);
    const result = gpa.alloc(u8, 4 + probs.len * @sizeOf(f32)) catch return Verdict.refuse(.busy, "out of memory");
    std.mem.writeInt(u32, result[0..4], e.d.cfg.speakers, .little);
    @memcpy(result[4..], std.mem.sliceAsBytes(probs));
    gpa.free(e.result);
    e.result = result;
    return .ok;
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    if (e.result.len == 0) return Verdict.refuse(.failed, "nemotron_diarization: no finished job to encode");
    out.* = .of(e.result);
    return .ok;
}

/// A tiny random diarizer: one encoder layer at 16 over 8 mels stacked two to a row, two speakers, chunks of four
/// rows with a FIFO of two and a speaker cache of six, so a 400-sample clip (hop 4) runs 13 chunks and compresses the
/// cache.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const d = 16;
    const mels = 8;
    const stack = 2;
    const tf = 8;
    const spk = 2;
    const n_fft = 16;
    const win = 12;
    var t = sdk.tensors.Tensors.init(a, 61);
    inline for (.{ "norm1", "norm2" }) |n| {
        try t.fill("encoder.layers.0." ++ n ++ ".weight", .{}, &.{d}, .F32, 1);
        try t.fill("encoder.layers.0." ++ n ++ ".bias", .{}, &.{d}, .F32, 0);
    }
    inline for (.{ .{ "encoder.layers.0.attn.w_qkv", 3 * d, d }, .{ "encoder.layers.0.attn.out_proj", d, d }, .{ "encoder.layers.0.ffn.linear1", 2 * d, d }, .{ "encoder.layers.0.ffn.linear2", d, 2 * d }, .{ "encoder.pre_encode.proj", d, mels * stack }, .{ "sortformer_modules.encoder_proj", tf, d }, .{ "sortformer_modules.first_hidden_to_hidden", tf, tf }, .{ "sortformer_modules.single_hidden_to_spks", spk, tf } }) |l| {
        try t.rand(l[0] ++ ".weight", .{}, &.{ l[1], l[2] }, .F32, 0.3);
        try t.rand(l[0] ++ ".bias", .{}, &.{l[1]}, .F32, 0.1);
    }
    inline for (.{ "embed_norm", "final_norm" }) |n| {
        try t.fill("encoder." ++ n ++ ".weight", .{}, &.{d}, .F32, 1);
        try t.fill("encoder." ++ n ++ ".bias", .{}, &.{d}, .F32, 0);
    }
    try t.rand("sortformer_modules.subpixel_upsample.weight", .{}, &.{ stack * tf, 3, tf }, .F32, 0.3);
    try t.rand("sortformer_modules.subpixel_upsample.bias", .{}, &.{stack * tf}, .F32, 0.1);
    try t.rand("sortformer_modules.learnable_sil_emb", .{}, &.{d}, .F32, 0.3);
    var fb: [mels * (n_fft / 2 + 1)]f32 = undefined;
    for (&fb, 0..) |*x, i| x.* = 0.05 + @as(f32, @floatFromInt(i % 7)) * 0.03;
    try t.values("preprocessor.fb", .{}, &.{fb.len}, .F32, &fb);
    var window: [win]f32 = undefined;
    for (&window, 0..) |*x, i| x.* = 0.5 - 0.5 * @cos(2 * std.math.pi * @as(f32, @floatFromInt(i)) / win);
    try t.values("preprocessor.window", .{}, &.{win}, .F32, &window);

    const pcm = try a.alloc(f32, 400);
    for (pcm, 0..) |*x, i| {
        const f: f32 = @floatFromInt(i);
        x.* = 0.4 * @sin(f * 0.21) + (if (i >= 200) 0.3 * @sin(f * 0.9) else 0);
    }
    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes = 
            \\{"model_type":"nemotron_diarization","num_speakers":2,"output_subsampling_factor":1,
            \\"encoder_config":{"feat_in":8,"d_model":16,"n_layers":1,"n_heads":2,"subsampling_factor":2,"ff_expansion":2.0},
            \\"modules_config":{"tf_d_model":8,"chunk_len":4,"fifo_len":2,"spkcache_len":6,"spkcache_update_period":2,
            \\"chunk_right_context":1,"spkcache_sil_frames_per_spk":1,"pred_score_threshold":0.25,"scores_boost_latest":0.05,
            \\"strong_boost_rate":0.75,"weak_boost_rate":1.5,"min_pos_scores_rate":0.5},
            \\"processor_config":{"feature_size":8,"sampling_rate":16000,"hop_length":4,"n_fft":16,"win_length":12,"preemphasis":0.97}}
            },
            .{ .name = "model.safetensors", .bytes = try t.bytes() },
        }),
        .inputs = try a.dupe([]const u8, &.{std.mem.sliceAsBytes(pcm)}),
    };
}

// ── tests ──

const testing = std.testing;

test "nemotron_diarization plugin: a clip on the tiny pack is the speaker count then a row per hop; no clip is refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    const clip = [_]abi.Str{.of(pack.fx.inputs[0])};
    const first = try arena.allocator().dupe(u8, (try pack.runInputs("{}", &clip)).ok);
    try testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, first[0..4], .little));
    try testing.expectEqual(@as(usize, 4 + 100 * 2 * 4), first.len);
    const probs = std.mem.bytesAsSlice(f32, first[4..]);
    for (probs) |p| try testing.expect(p >= 0 and p <= 1);
    try testing.expectEqualSlices(u8, first, (try pack.runInputs("{}", &clip)).ok);
    try testing.expectEqualStrings("diarization takes one input: the clip as f32 mono PCM at 16 kHz", (try pack.run("{}")).refused.why.slice());
}

const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack(@This());
