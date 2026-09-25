//! Per-model settings (`~/.mlx-serve/model-settings.json`): context size, KV
//! quant and MTP that follow the MODEL, applied at every load construction
//! site. Keyed by the model's absolute path (dir, or the `.gguf` file).
//! The app and `POST /v1/steering` edit the file; the server owns applying
//! it. A malformed file is logged and treated as empty: a settings typo must
//! never stop a load.
const std = @import("std");
const kv_quant = @import("kv_quant.zig");
const log = @import("log.zig");
const mtp_acceptance = @import("mtp_acceptance.zig");
const steering = @import("steering.zig");

pub const Override = struct {
    ctx_size: ?u32 = null,
    kv_quant: ?kv_quant.KVQuantConfig = null,
    mtp: ?bool = null,
    mtp_acceptance: ?mtp_acceptance.Mode = null,
    /// Absent key = follow the launch flags; `null` = off; object = configured.
    steering: steering.Setting = .{},

    pub fn isEmpty(o: Override) bool {
        return o.ctx_size == null and o.kv_quant == null and o.mtp == null and o.mtp_acceptance == null and o.steering.state == .inherit;
    }
};

pub const Settings = struct {
    parsed: ?std.json.Parsed(std.json.Value) = null,

    pub fn deinit(self: *Settings) void {
        if (self.parsed) |*p| p.deinit();
        self.parsed = null;
    }

    pub fn lookup(self: *const Settings, model_path: []const u8) Override {
        const p = self.parsed orelse return .{};
        const root = switch (p.value) {
            .object => |o| o,
            else => return .{},
        };
        const want = trimSlash(model_path);
        var it = root.iterator();
        while (it.next()) |kv| {
            if (!std.mem.eql(u8, trimSlash(kv.key_ptr.*), want)) continue;
            return fromValue(kv.value_ptr.*);
        }
        return .{};
    }
};

fn trimSlash(p: []const u8) []const u8 {
    var s = p;
    while (s.len > 1 and s[s.len - 1] == '/') s = s[0 .. s.len - 1];
    return s;
}

fn fromValue(v: std.json.Value) Override {
    const obj = switch (v) {
        .object => |o| o,
        else => return .{},
    };
    var o: Override = .{};
    if (obj.get("ctx_size")) |c| switch (c) {
        .integer => |i| if (i > 0 and i <= std.math.maxInt(u32)) {
            o.ctx_size = @intCast(i);
        },
        else => {},
    };
    if (obj.get("kv_quant")) |k| o.kv_quant = kv_quant.KVQuantConfig.fromJsonValue(k);
    if (obj.get("mtp")) |m| switch (m) {
        .bool => |b| o.mtp = b,
        else => {},
    };
    if (obj.get("mtp_acceptance")) |a| switch (a) {
        .string => |name| o.mtp_acceptance = mtp_acceptance.fromName(name),
        else => {},
    };
    if (obj.get("steering")) |sv| {
        if (steering.Setting.fromJsonValue(sv)) |st| {
            o.steering = st;
        } else log.warn("[model-settings] malformed steering ignored; the launch flags apply\n", .{});
    }
    return o;
}

/// Serializes every read-modify-write of the file across models: two
/// `/v1/steering` calls for different models hold different bank locks but
/// share this one file. Never held while taking another lock.
pub var file_mu: std.Io.Mutex = .init;

/// An f32 widened to f64 prints as 0.10000000149011612 in a file people hand-edit, and
/// the sheet then reads the re-typed 0.1 back as a change. Six digits is past f32's
/// precision and round-trips every scale the server accepts.
fn roundScale(v: f32) f64 {
    return @round(@as(f64, v) * 1_000_000.0) / 1_000_000.0;
}

/// Rewrite ONLY `model_path`'s `steering` key (`.inherit` removes it), keeping
/// every other key. A missing file or entry is created; a malformed file is
/// refused rather than clobbered.
pub fn writeSteering(alloc: std.mem.Allocator, io: std.Io, file_path: []const u8, model_path: []const u8, setting: steering.Setting) !void {
    file_mu.lockUncancelable(io);
    defer file_mu.unlock(io);
    const body = std.Io.Dir.cwd().readFileAlloc(io, file_path, alloc, .limited(1 << 20)) catch |err| switch (err) {
        error.FileNotFound => try alloc.dupe(u8, "{}"),
        else => return err,
    };
    defer alloc.free(body);
    var parsed = std.json.parseFromSlice(std.json.Value, alloc, body, .{}) catch return error.SettingsMalformed;
    defer parsed.deinit();
    if (parsed.value != .object) return error.SettingsMalformed;
    const arena = parsed.arena.allocator();
    const root = &parsed.value.object;
    const want = trimSlash(model_path);
    var it = root.iterator();
    const entry: *std.json.Value = while (it.next()) |kv| {
        if (std.mem.eql(u8, trimSlash(kv.key_ptr.*), want)) break kv.value_ptr;
    } else blk: {
        try root.put(arena, try arena.dupe(u8, model_path), .{ .object = .empty });
        break :blk root.getPtr(model_path).?;
    };
    if (entry.* != .object) entry.* = .{ .object = .empty };
    const obj = &entry.object;
    switch (setting.state) {
        .inherit => _ = obj.orderedRemove("steering"),
        .off => try obj.put(arena, "steering", .null),
        .configured => {
            const c = setting.cfg;
            var o: std.json.ObjectMap = .empty;
            try o.put(arena, "name", .{ .string = try arena.dupe(u8, c.name()) });
            try o.put(arena, "ffn", .{ .float = roundScale(c.ffn) });
            try o.put(arena, "attn", .{ .float = roundScale(c.attn) });
            try obj.put(arena, "steering", .{ .object = o });
        },
    }
    const out = try std.json.Stringify.valueAlloc(alloc, parsed.value, .{ .whitespace = .indent_2 });
    defer alloc.free(out);
    if (std.fs.path.dirname(file_path)) |dir| try std.Io.Dir.cwd().createDirPath(io, dir);
    const tmp = try std.fmt.allocPrint(alloc, "{s}.tmp", .{file_path});
    defer alloc.free(tmp);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = tmp, .data = out });
    try std.Io.Dir.renameAbsolute(tmp, file_path, io);
}

pub fn parse(alloc: std.mem.Allocator, body: []const u8) !Settings {
    return .{ .parsed = try std.json.parseFromSlice(std.json.Value, alloc, body, .{}) };
}

/// Missing file = empty. Unreadable or malformed = empty, logged.
pub fn load(alloc: std.mem.Allocator, io: std.Io, path: []const u8) Settings {
    const body = std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1 << 20)) catch |err| {
        if (err != error.FileNotFound) log.warn("[model-settings] {s}: unreadable ({s}), ignored\n", .{ path, @errorName(err) });
        return .{};
    };
    defer alloc.free(body);
    return parse(alloc, body) catch |err| {
        log.warn("[model-settings] {s}: malformed ({s}), ignored\n", .{ path, @errorName(err) });
        return .{};
    };
}

pub fn defaultPath(buf: []u8) []const u8 {
    const home = std.mem.span(std.c.getenv("HOME") orelse "/tmp");
    return std.fmt.bufPrint(buf, "{s}/.mlx-serve/model-settings.json", .{home}) catch "";
}

/// The one call load sites make: read the default file, look the model up, log a hit.
pub fn overrideFor(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8) Override {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    var s = load(alloc, io, defaultPath(&buf));
    defer s.deinit();
    const o = s.lookup(model_path);
    if (!o.isEmpty()) log.info("[model-settings] {s}: ctx={d} kv={s} mtp={s} accept={s} steering={s}\n", .{
        model_path,
        o.ctx_size orelse 0,
        if (o.kv_quant) |k| k.wireName() else "default",
        if (o.mtp) |m| (if (m) "on" else "off") else "default",
        if (o.mtp_acceptance) |a| mtp_acceptance.name(a) else "default",
        switch (o.steering.state) {
            .inherit => "default",
            .off => "off",
            .configured => o.steering.cfg.name(),
        },
    });
    return o;
}

test "model_settings: parse + lookup with and without trailing slash" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a/": {"ctx_size": 65536, "kv_quant": "8", "mtp": false}, "/m/b": {"kv_quant": 4}}
    );
    defer s.deinit();
    const a = s.lookup("/m/a");
    try std.testing.expectEqual(@as(?u32, 65536), a.ctx_size);
    try std.testing.expectEqual(@as(u8, 8), a.kv_quant.?.bits);
    try std.testing.expectEqual(@as(?bool, false), a.mtp);
    const b = s.lookup("/m/b/");
    try std.testing.expectEqual(@as(?u32, null), b.ctx_size);
    try std.testing.expectEqual(@as(u8, 4), b.kv_quant.?.bits);
    try std.testing.expectEqual(@as(?bool, null), b.mtp);
    try std.testing.expect(s.lookup("/m/c").isEmpty());
}

test "model_settings: mtp_acceptance names a mode at its default threshold" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a": {"mtp_acceptance": "typical"}, "/m/b": {"mtp_acceptance": "tokenv3"}, "/m/c": {"mtp_acceptance": "exact"}, "/m/d": {"mtp_acceptance": "fast"}}
    );
    defer s.deinit();
    try std.testing.expectEqual(@as(f32, 0.2), s.lookup("/m/a").mtp_acceptance.?.typical.delta);
    try std.testing.expectEqual(@as(f32, 0.95), s.lookup("/m/b").mtp_acceptance.?.tokenv3);
    try std.testing.expect(s.lookup("/m/c").mtp_acceptance.? == .exact);
    try std.testing.expect(s.lookup("/m/d").isEmpty());
}

test "model_settings: bad values ignored, bad JSON = empty" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a": {"ctx_size": 0, "kv_quant": "16", "mtp": "yes", "future": 1}}
    );
    defer s.deinit();
    try std.testing.expect(s.lookup("/m/a").isEmpty());
    try std.testing.expectError(error.SyntaxError, parse(std.testing.allocator, "{nope"));
    var empty = load(std.testing.allocator, std.testing.io, "/nonexistent/model-settings.json");
    defer empty.deinit();
    try std.testing.expect(empty.lookup("/m/a").isEmpty());
}

test "model_settings: steering has three states and a malformed object is ignored" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a": {"steering": null}, "/m/b": {"steering": {"name": "terse", "ffn": -1}}, "/m/c": {"ctx_size": 4096}, "/m/d": {"steering": {"ffn": 1}}}
    );
    defer s.deinit();
    const a = s.lookup("/m/a");
    try std.testing.expect(a.steering.state == .off);
    try std.testing.expect(!a.isEmpty());
    const b = s.lookup("/m/b").steering.cfg;
    try std.testing.expectEqualStrings("terse", b.name());
    try std.testing.expectEqual(@as(f32, -1), b.ffn);
    try std.testing.expectEqual(@as(f32, 0), b.attn);
    try std.testing.expect(s.lookup("/m/c").steering.state == .inherit);
    const d = s.lookup("/m/d");
    try std.testing.expect(d.steering.state == .inherit);
    try std.testing.expect(d.isEmpty());
}

test "model_settings: writeSteering rewrites only the model's steering key" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &pbuf);
    const path = try std.fmt.allocPrint(a, "{s}/sub/model-settings.json", .{pbuf[0..root_len]});
    defer a.free(path);

    // Missing file (and dir) = created.
    try writeSteering(a, io, path, "/m/c", steering.Setting.off);
    try tmp.dir.writeFile(io, .{ .sub_path = "sub/model-settings.json", .data =
        \\{"/m/a": {"ctx_size": 4096, "future": 1}, "/m/b/": {"mtp": false}}
    });
    const cfg = try steering.Configured.init("terse", -1, 0.25);
    try writeSteering(a, io, path, "/m/a", steering.Setting.configured(cfg));
    try writeSteering(a, io, path, "/m/b", steering.Setting.off); // trailing-slash key matched, not duplicated
    {
        var s = load(a, io, path);
        defer s.deinit();
        const ma = s.lookup("/m/a");
        try std.testing.expectEqual(@as(?u32, 4096), ma.ctx_size);
        try std.testing.expectEqualStrings("terse", ma.steering.cfg.name());
        try std.testing.expectEqual(@as(f32, 0.25), ma.steering.cfg.attn);
        const mb = s.lookup("/m/b");
        try std.testing.expectEqual(@as(?bool, false), mb.mtp);
        try std.testing.expect(mb.steering.state == .off);
        try std.testing.expectEqual(@as(usize, 2), s.parsed.?.value.object.count());
        const raw = try tmp.dir.readFileAlloc(io, "sub/model-settings.json", a, .limited(1 << 16));
        defer a.free(raw);
        try std.testing.expect(std.mem.indexOf(u8, raw, "\"future\"") != null);
    }
    // Inherit removes the key and nothing else; a malformed file is refused.
    try writeSteering(a, io, path, "/m/a", steering.Setting.inherit);
    {
        var s = load(a, io, path);
        defer s.deinit();
        try std.testing.expect(s.lookup("/m/a").steering.state == .inherit);
        try std.testing.expectEqual(@as(?u32, 4096), s.lookup("/m/a").ctx_size);
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "sub/model-settings.json", .data = "{nope" });
    try std.testing.expectError(error.SettingsMalformed, writeSteering(a, io, path, "/m/a", steering.Setting.off));
}
