//! Each served media backend's tiny pack answers through its table with the bytes its legacy gen.zig arm wrote:
//! results, streamed events, refusals, wrong-route hints and the ready /v1/models row, recorded per plugin in
//! `fixtures/legacy_http.golden` before those arms were deleted (a change there is a change in served bytes).
//! Also: the admission reserve legacy set (no `GenRequest.peak_bytes`: the model's own estimate) and each
//! type's completeness marker.

const std = @import("std");
const testing = std.testing;
const mlx = @import("../../../mlx/ffi.zig");
const gen = @import("../gen.zig");
const http = @import("http.zig");
const served = @import("served.zig");
const Engine = @import("engine.zig").Engine;
const harness = @import("flip_harness.zig");

const Fixture = harness.Fixture;
const capture = harness.capture;
const pluginRoute = harness.pluginRoute;
const expectDecisions = harness.expectDecisions;

test "flip: stable_audio3 answers its legacy arm's bytes" {
    var f: Fixture(@import("../../../plugins/stable_audio3/plugin.zig"), @embedFile("../../../plugins/stable_audio3/fixtures/legacy_http.golden")) = .{};
    try f.open(false);
    defer f.close();
    try f.expectAudio(&.{ f.p.fx.body, try f.p.streamed(), "{\"prompt\":\"rain\",\"steps\":0}", "{\"duration_seconds\":1}" });
    // Control: the golden sees the audio (another seed differs from the recorded bytes).
    const other = try capture(testing.allocator, pluginRoute, .{ testing.allocator, "{\"prompt\":\"rain\",\"duration_seconds\":1,\"steps\":2,\"seed\":4}", f.pe, gen.GenRoute.sound });
    defer testing.allocator.free(other);
    try testing.expectError(error.GoldenMismatch, f.g.compare("sound 0", other, false));
}

test "flip: kokoro answers its legacy arm's bytes" {
    var f: Fixture(@import("../../../plugins/kokoro/plugin.zig"), @embedFile("../../../plugins/kokoro/fixtures/legacy_http.golden")) = .{};
    try f.open(false);
    defer f.close();
    try f.expectAudio(&.{ f.p.fx.body, try f.p.streamed(), "{}", "{\"input\":\"hi\",\"voice\":\"nobody\"}", "{\"input\":\"hi\",\"ref_audio\":\"AAAA\"}" });
}

test "flip: qwen3_tts answers its legacy arm's bytes" {
    var f: Fixture(@import("../../../plugins/qwen3_tts/plugin.zig"), @embedFile("../../../plugins/qwen3_tts/fixtures/legacy_http.golden")) = .{};
    try f.open(false);
    defer f.close();
    try f.expectAudio(&.{ f.p.fx.body, try f.p.streamed(), "{}", "{\"input\":\"\"}", "{\"input\":\"hi\",\"voice\":\"default\"}" });
}

test "flip: acestep answers its legacy arm's bytes" {
    var f: Fixture(@import("../../../plugins/acestep/plugin.zig"), @embedFile("../../../plugins/acestep/fixtures/legacy_http.golden")) = .{};
    try f.open(false);
    defer f.close();
    try f.expectAudio(&.{ f.p.fx.body, try f.p.streamed(), "{}", "{\"prompt\":\"x\",\"instrumental\":true,\"lyrics\":\"la\"}", "{\"prompt\":\"x\",\"steps\":0}" });
}

test "flip: music3 answers its legacy arm's bytes" {
    var f: Fixture(@import("../../../plugins/music3/plugin.zig"), @embedFile("../../../plugins/music3/fixtures/legacy_http.golden")) = .{};
    try f.open(false);
    defer f.close();
    try f.expectAudio(&.{ f.p.fx.body, try f.p.streamed(), "{}", "{\"prompt\":\"x\",\"instrumental\":true,\"lyrics\":\"la\"}" });
}

test "flip: yue2 answers its legacy arm's bytes, the planned score beside the streamed WAV" {
    var f: Fixture(@import("../../../plugins/yue2/plugin.zig"), @embedFile("../../../plugins/yue2/fixtures/legacy_http.golden")) = .{};
    try f.open(false);
    defer f.close();
    const melody = "{\"stream\":true,\"prompt\":\"pop\",\"lyrics\":\"[verse]\\nla la\",\"cot\":\"melody\",\"duration_seconds\":5,\"steps\":2,\"seed\":3}";
    f.music_needle = .{ .i = 2, .text = "\"type\":\"complete\",\"format\":\"wav\",\"abc\":\"" };
    try f.expectAudio(&.{ f.p.fx.body, try f.p.streamed(), melody, "{}", "{\"prompt\":\"p\",\"bpm\":120}" });
    try testing.expect(f.music_needle == null);
}

test "flip: ltx answers its legacy arm's bytes" {
    var f: Fixture(@import("../../../plugins/ltx/plugin.zig"), @embedFile("../../../plugins/ltx/fixtures/legacy_http.golden")) = .{};
    try f.open(false);
    defer f.close();
    try f.expectRoute(.video, &.{ f.p.fx.body, try f.p.streamed(), "{}", "{\"prompt\":\"x\",\"pipeline\":\"two_stage\"}", "{\"prompt\":\"x\",\"decoder\":\"diffusion\"}" });
    try f.expectRow(.video);
}

test "flip: minimax_h3 answers its legacy arm's bytes" {
    var f: Fixture(@import("../../../plugins/minimax_h3/plugin.zig"), @embedFile("../../../plugins/minimax_h3/fixtures/legacy_http.golden")) = .{};
    try f.open(true); // recorded from the legacy arm, which generated on the GPU stream
    defer f.close();
    try f.expectRoute(.video, &.{ f.p.fx.body, try f.p.streamed(), "{}", "{\"prompt\":\"x\",\"width\":70}", "{\"prompt\":\"x\",\"ref_images\":[\"AAAA\"]}" });
    try f.expectRow(.video);
}

test "flip: laya answers its legacy arm's bytes" {
    try expectDecisions(Fixture(@import("../../../plugins/laya/plugin.zig"), @embedFile("../../../plugins/laya/fixtures/legacy_http.golden")), &.{});
}

test "flip: kev answers its legacy arm's bytes" {
    try expectDecisions(Fixture(@import("../../../plugins/kev/plugin.zig"), @embedFile("../../../plugins/kev/fixtures/legacy_http.golden")), &.{});
}

test "flip: d1 answers its legacy arm's bytes" {
    try expectDecisions(Fixture(@import("../../../plugins/d1/plugin.zig"), @embedFile("../../../plugins/d1/fixtures/legacy_http.golden")), &.{});
}

test "plugin arm: a stable_audio3 pack is the plugin's, billed as the legacy arm billed it" {
    const a = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const sa_plugin = @import("../../../plugins/stable_audio3/plugin.zig");
    const fx = try sa_plugin.conformanceFixture(arena.allocator());
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "t5gemma-b-b-ul2");
    for (fx.files) |f| try tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const path = pbuf[0..try tmp.dir.realPath(io, &pbuf)];

    const mt = gen.peekModelType(io, a, path).?;
    defer a.free(mt);
    try testing.expectEqualStrings("stable_audio3", mt);
    try testing.expectEqual(gen.Modality.audio, gen.detectModality(io, a, path).?);
    const legacy_bill = gen.sumSafetensorsAt(io, path);
    try testing.expect(legacy_bill > 0);
    try testing.expectEqual(legacy_bill, gen.estimatePeakResidentBytes(io, path, mt));

    const pe = try Engine.load(io, a, path, served.byName(mt).?, mlx.mlx_default_cpu_stream_new());
    defer pe.deinit();
    try testing.expectEqual(@as(u64, 0), pe.requestPeak(fx.body));
}

test "flip: nemotron_diarization gives its legacy engine's speakers and probabilities" {
    const a = testing.allocator;
    var f: Fixture(@import("../../../plugins/nemotron_diarization/plugin.zig"), @embedFile("../../../plugins/nemotron_diarization/fixtures/legacy_http.golden")) = .{};
    try f.open(false);
    defer f.close();
    const pcm = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(f.p.fx.inputs[0])));
    const got = http.diarize(f.pe, a, pcm).?;
    defer a.free(got.probs);
    try f.g.check("probs", std.mem.sliceAsBytes(got.probs));
    var kb: [8]u8 = undefined;
    try f.g.check("speakers", try std.fmt.bufPrint(&kb, "{d}", .{got.speakers}));
    try f.expectRow(.diarization);
}

test "flip: each served type keeps its legacy completeness marker (none where legacy had none)" {
    const discovery = @import("../../registry/discovery.zig");
    const want = [_]struct { []const u8, ?[]const u8 }{
        .{ "AudioVideo", "connector.safetensors" },                .{ "minimax_h3", "transformer.safetensors" },
        .{ "minimax_music3", "vocoder.safetensors" },              .{ "acestep", "text_encoder/model.safetensors" },
        .{ "stable_audio3", "t5gemma-b-b-ul2/model.safetensors" }, .{ "yue2", "vae.safetensors" },
        .{ "kokoro", null },                                       .{ "qwen3_tts", null },
        .{ "laya", null },                                         .{ "kev", null },
        .{ "d1", null },                                           .{ "clef", null },
        .{ "nemotron_diarization", null },
    };
    for (want) |w| {
        try testing.expect(served.byName(w[0]) != null);
        const got = discovery.requiredMediaMarker(w[0]);
        if (w[1]) |m| try testing.expectEqualStrings(m, got.?) else try testing.expect(got == null);
    }
}
