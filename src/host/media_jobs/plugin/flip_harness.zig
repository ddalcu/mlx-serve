//! Shared golden-fixture harness for `flip_test.zig` (pure-SDK media plugins) and `clef_flip_test.zig` (clef),
//! carrying no plugin imports of its own.

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

pub const capture = test_conn.capture;

/// The fixture written to a temp dir; `path` is its absolute path.
pub const Pack = struct {
    arena: std.heap.ArenaAllocator,
    tmp: testing.TmpDir,
    fx: @import("../../../sdk/testing/fixture.zig").Fixture,
    pbuf: [std.fs.max_path_bytes]u8 = undefined,
    path: []const u8 = "",

    pub fn open(self: *Pack, comptime P: type) !void {
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

    pub fn close(self: *Pack) void {
        self.tmp.cleanup();
        self.arena.deinit();
    }

    /// The fixture's body with `"stream":true`.
    pub fn streamed(self: *Pack) ![]const u8 {
        return std.fmt.allocPrint(self.arena.allocator(), "{{\"stream\":true,{s}", .{self.fx.body[1..]});
    }
};

/// Every golden was recorded on one machine; computed values (and a RAM-sized context) differ on another, where a
/// differing record is reported, not failed, as proof tapes do with greedy runs.
const recorded_on = "applegpu_g17d";

/// Records of `== <key> <len> <sha256 hex>\n`, each followed by its bytes and a newline when they are at most
/// 2 KiB. Bytes are compared normalized: the pack's temp path reads `<pack>`, a row's `created` time is zeroed.
pub const Golden = struct {
    text: []const u8,
    pack: []const u8,

    pub fn check(self: Golden, key: []const u8, bytes: []const u8) !void {
        return self.compare(key, bytes, true);
    }

    /// `off_device`: a differing record off the recording machine is reported, not failed.
    pub fn compare(self: Golden, key: []const u8, bytes: []const u8, off_device: bool) !void {
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
pub fn Fixture(comptime P: type, comptime golden: []const u8) type {
    return struct {
        p: Pack = undefined,
        s: mlx.mlx_stream = undefined,
        pe: *Engine = undefined,
        g: Golden = undefined,
        /// Text the music route's answer to body `i` must hold on every device; cleared once seen.
        music_needle: ?struct { i: usize, text: []const u8 } = null,

        pub fn open(self: *@This(), gpu: bool) !void {
            try self.p.open(P);
            errdefer self.p.close();
            const t = served.claimPath(testing.io, self.p.path) orelse return error.NotServed;
            self.s = if (gpu) mlx.gpuStream() else mlx.mlx_default_cpu_stream_new();
            errdefer _ = mlx.mlx_stream_free(self.s);
            self.pe = try Engine.load(testing.io, testing.allocator, self.p.path, t, self.s);
            self.g = .{ .text = golden, .pack = self.p.path };
        }

        pub fn close(self: *@This()) void {
            self.pe.deinit();
            _ = mlx.mlx_stream_free(self.s);
            self.p.close();
        }

        /// Each body on `route`, reserved as legacy reserved it (nothing beyond the model's estimate).
        pub fn expectRoute(self: *@This(), route: gen.GenRoute, bodies: []const []const u8) !void {
            const a = testing.allocator;
            for (bodies, 0..) |body, i| {
                try testing.expectEqual(@as(u64, 0), self.pe.requestPeak(body));
                const got = try capture(a, pluginRoute, .{ a, body, self.pe, route });
                defer a.free(got);
                var kb: [32]u8 = undefined;
                try self.g.check(try std.fmt.bufPrint(&kb, "{s} {d}", .{ @tagName(route), i }), got);
                if (self.music_needle) |n| if (route == .music and n.i == i) {
                    try testing.expect(std.mem.indexOf(u8, got, n.text) != null);
                    self.music_needle = null;
                };
            }
        }

        /// Every body on every audio route, then the ready row.
        pub fn expectAudio(self: *@This(), bodies: []const []const u8) !void {
            for ([_]gen.GenRoute{ .speech, .music, .sound }) |route| try self.expectRoute(route, bodies);
            try self.expectRow(.audio);
        }

        /// The /v1/models row of the ready model.
        pub fn expectRow(self: *@This(), m: gen.Modality) !void {
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

pub fn pluginRoute(conn: *Conn, a: std.mem.Allocator, body: []const u8, e: *Engine, route: gen.GenRoute) !void {
    return http.handle(a, conn, body, e, route, "m");
}

fn pluginDecision(conn: *Conn, a: std.mem.Allocator, body: []const u8, e: *Engine) !void {
    http.handleMany(e, "m", &.{.{ .allocator = a, .conn = conn, .body = body }});
}

/// The fixture's question, `extra`, then malformed bodies; then the row.
pub fn expectDecisions(comptime F: type, extra: []const []const u8) !void {
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
