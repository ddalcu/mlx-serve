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
const test_conn = @import("test_conn.zig");
const registry_mod = @import("../../registry/registry.zig");
const models_api = @import("../../api/models.zig");
const Conn = @import("../../http/server.zig").Conn;

const capture = test_conn.capture;

/// The fixture written to a temp dir; `path` is its absolute path.
const Pack = struct {
    arena: std.heap.ArenaAllocator,
    tmp: testing.TmpDir,
    fx: @import("../../../sdk/testing/fixture.zig").Fixture,
    pbuf: [std.fs.max_path_bytes]u8 = undefined,
    path: []const u8 = "",

    fn open(self: *Pack, comptime P: type) !void {
        self.arena = std.heap.ArenaAllocator.init(testing.allocator);
        self.tmp = testing.tmpDir(.{ .iterate = true });
        self.fx = try P.conformanceFixture(self.arena.allocator());
        const io = testing.io;
        for (self.fx.files) |f| {
            if (std.fs.path.dirname(f.name)) |d| try self.tmp.dir.createDirPath(io, d);
            try self.tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
        }
        self.path = self.pbuf[0..try self.tmp.dir.realPath(io, &self.pbuf)];
    }

    fn close(self: *Pack) void {
        self.tmp.cleanup();
        self.arena.deinit();
    }

    /// The fixture's body with `"stream":true`.
    fn streamed(self: *Pack) ![]const u8 {
        return std.fmt.allocPrint(self.arena.allocator(), "{{\"stream\":true,{s}", .{self.fx.body[1..]});
    }
};

/// Every golden was recorded on one machine; computed values (and a RAM-sized context) differ on another, where a
/// differing record is reported, not failed, as proof tapes do with greedy runs.
const recorded_on = "applegpu_g17d";

/// Records of `== <key> <len> <sha256 hex>\n`, each followed by its bytes and a newline when they are at most
/// 2 KiB. Bytes are compared normalized: the pack's temp path reads `<pack>`, a row's `created` time is zeroed.
const Golden = struct {
    text: []const u8,
    pack: []const u8,

    fn check(self: Golden, key: []const u8, bytes: []const u8) !void {
        return self.compare(key, bytes, true);
    }

    /// `off_device`: a differing record off the recording machine is reported, not failed.
    fn compare(self: Golden, key: []const u8, bytes: []const u8, off_device: bool) !void {
        const got = try self.norm(bytes);
        defer testing.allocator.free(got);
        var h: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(got, &h, .{});
        var hb: [64]u8 = undefined;
        const hex = std.fmt.bufPrint(&hb, "{x}", .{h}) catch unreachable;
        var pos: usize = 0;
        while (pos < self.text.len) {
            const eol = std.mem.indexOfScalarPos(u8, self.text, pos, '\n').?;
            const head = self.text[pos + 3 .. eol]; // past "== "
            const sp2 = std.mem.lastIndexOfScalar(u8, head, ' ').?;
            const sp1 = std.mem.lastIndexOfScalar(u8, head[0..sp2], ' ').?;
            const len = try std.fmt.parseInt(usize, head[sp1 + 1 .. sp2], 10);
            const stored = if (len <= 2048) self.text[eol + 1 ..][0..len] else null;
            pos = eol + 1 + if (stored != null) len + 1 else 0;
            if (!std.mem.eql(u8, head[0..sp1], key)) continue;
            const same = len == got.len and std.mem.eql(u8, head[sp2 + 1 ..], hex);
            var dbuf: [64]u8 = undefined;
            const here = @import("../../../sdk/testing/tape.zig").deviceName(&dbuf);
            if (!same and off_device and !std.mem.eql(u8, here, recorded_on)) {
                std.debug.print("golden '{s}' differs on {s} (recorded on {s}): not compared\n", .{ key, here, recorded_on });
                return;
            }
            if (stored) |want| try testing.expectEqualStrings(want, got);
            if (!same) {
                std.debug.print("golden '{s}': want {d} bytes {s}, got {d} bytes {s}\n", .{ key, len, head[sp2 + 1 ..], got.len, hex });
                return error.GoldenMismatch;
            }
            return;
        }
        std.debug.print("golden '{s}': no such record\n", .{key});
        return error.GoldenMissing;
    }

    fn norm(self: Golden, bytes: []const u8) ![]u8 {
        const out = try std.mem.replaceOwned(u8, testing.allocator, bytes, self.pack, "<pack>");
        const tag = "\"created\":";
        const i = (std.mem.indexOf(u8, out, tag) orelse return out) + tag.len;
        var j = i;
        while (j < out.len and std.ascii.isDigit(out[j])) j += 1;
        @memset(out[i..j], '0');
        return out;
    }
};

/// One plugin's tiny pack, served by its table, and its golden.
fn Fixture(comptime P: type, comptime golden: []const u8) type {
    return struct {
        p: Pack = undefined,
        s: mlx.mlx_stream = undefined,
        pe: *Engine = undefined,
        g: Golden = undefined,

        fn open(self: *@This(), gpu: bool) !void {
            try self.p.open(P);
            errdefer self.p.close();
            const t = served.claimPath(testing.io, self.p.path) orelse return error.NotServed;
            self.s = if (gpu) mlx.gpuStream() else mlx.mlx_default_cpu_stream_new();
            errdefer _ = mlx.mlx_stream_free(self.s);
            self.pe = try Engine.load(testing.io, testing.allocator, self.p.path, t, self.s);
            self.g = .{ .text = golden, .pack = self.p.path };
        }

        fn close(self: *@This()) void {
            self.pe.deinit();
            _ = mlx.mlx_stream_free(self.s);
            self.p.close();
        }

        /// Each body on `route`, reserved as legacy reserved it (nothing beyond the model's estimate).
        fn expectRoute(self: *@This(), route: gen.GenRoute, bodies: []const []const u8) !void {
            const a = testing.allocator;
            for (bodies, 0..) |body, i| {
                try testing.expectEqual(@as(u64, 0), self.pe.requestPeak(body));
                const got = try capture(a, pluginRoute, .{ a, body, self.pe, route });
                defer a.free(got);
                var kb: [32]u8 = undefined;
                try self.g.check(try std.fmt.bufPrint(&kb, "{s} {d}", .{ @tagName(route), i }), got);
            }
        }

        /// Every body on every audio route, then the ready row.
        fn expectAudio(self: *@This(), bodies: []const []const u8) !void {
            for ([_]gen.GenRoute{ .speech, .music, .sound }) |route| try self.expectRoute(route, bodies);
            try self.expectRow(.audio);
        }

        /// The /v1/models row of the ready model.
        fn expectRow(self: *@This(), m: gen.Modality) !void {
            const a = testing.allocator;
            const io = testing.io;
            const mt = gen.peekModelType(io, a, self.p.path).?;
            defer a.free(mt);
            var stub = try gen.buildStubCpuState(a, m);
            defer gen.freeStubCpuState(a, &stub);
            var entry = std.mem.zeroInit(registry_mod.LoadedModel, .{ .allocator = a, .id = "m", .path = self.p.path, .arch_hint = mt, .state = .ready, .config = stub.config, .chat_config = stub.chat_config, .refcount = std.atomic.Value(u32).init(0) });
            entry.media_plugin = self.pe;
            const got = try models_api.renderModelEntry(a, io, &entry);
            defer a.free(got);
            try self.g.check("row", got);
        }
    };
}

fn pluginRoute(conn: *Conn, a: std.mem.Allocator, body: []const u8, e: *Engine, route: gen.GenRoute) !void {
    return http.handle(a, conn, body, e, route, "m");
}

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
    try f.expectAudio(&.{ f.p.fx.body, try f.p.streamed(), "{}", "{\"input\":\"\"}", "{\"input\":\"hi\",\"voice\":\"af_heart\"}" });
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
    try f.expectAudio(&.{ f.p.fx.body, try f.p.streamed(), melody, "{}", "{\"prompt\":\"p\",\"bpm\":120}" });
    const scored = try capture(testing.allocator, pluginRoute, .{ testing.allocator, melody, f.pe, gen.GenRoute.music });
    defer testing.allocator.free(scored);
    try testing.expect(std.mem.indexOf(u8, scored, "\"type\":\"complete\",\"format\":\"wav\",\"abc\":\"") != null);
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

fn pluginDecision(conn: *Conn, a: std.mem.Allocator, body: []const u8, e: *Engine) !void {
    http.handleMany(e, "m", &.{.{ .allocator = a, .conn = conn, .body = body }});
}

/// The fixture's question, `extra`, then malformed bodies; then the row.
fn expectDecisions(comptime F: type, extra: []const []const u8) !void {
    const a = testing.allocator;
    var f: F = .{};
    try f.open(false);
    defer f.close();
    var bodies: std.ArrayList([]const u8) = .empty;
    defer bodies.deinit(a);
    try bodies.append(a, f.p.fx.body);
    try bodies.appendSlice(a, extra);
    try bodies.appendSlice(a, &.{ "not json", "[]", "{}", "{\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"nope\"}}}" });
    for (bodies.items, 0..) |body, i| {
        try testing.expectEqual(@as(u64, 0), f.pe.requestPeak(body));
        const got = try capture(a, pluginDecision, .{ a, body, f.pe });
        defer a.free(got);
        var kb: [32]u8 = undefined;
        try f.g.check(try std.fmt.bufPrint(&kb, "decision {d}", .{i}), got);
    }
    try f.expectRow(.decision);
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

test "flip: clef answers its legacy arm's bytes" {
    try expectDecisions(Fixture(@import("../../../plugins/clef/plugin.zig"), @embedFile("../../../plugins/clef/fixtures/legacy_http.golden")), &.{"{\"truncate\":1}"});
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
