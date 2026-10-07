//! Per-model settings (`~/.mlx-serve/model-settings.json`): a request alias, context size, KV
//! quant and MTP that follow the MODEL, applied at every load construction
//! site. Keyed by the model's absolute path (dir, or the `.gguf` file).
//! The app edits the file; the server owns applying it. A malformed file is
//! logged and treated as empty: a settings typo must never stop a load.
const std = @import("std");
const kv_quant = @import("kv_quant.zig");
const log = @import("log.zig");
const mtp_acceptance = @import("mtp_acceptance.zig");
const generation = @import("generation_settings.zig");

pub const Override = struct {
    ctx_size: ?u32 = null,
    kv_quant: ?kv_quant.KVQuantConfig = null,
    mtp: ?bool = null,
    mtp_acceptance: ?mtp_acceptance.Mode = null,
    mtp_greedy_tail: ?bool = null,
    /// LOSSY int8-activation prefill for 2-bit Prism packs (`qmm_int8`).
    int8_prefill: ?bool = null,
    /// Extra template variables as a JSON object (vLLM/llama.cpp
    /// `chat_template_kwargs`), e.g. `{"preserve_thinking": true}`. Owned.
    chat_template_kwargs: ?[]const u8 = null,
    /// The speculation sidecar: "off", "auto" (the pack's own `drafter/`) or an
    /// absolute path. Owned.
    drafter: ?[]const u8 = null,

    pub fn isEmpty(o: Override) bool {
        return o.ctx_size == null and o.kv_quant == null and o.mtp == null and o.mtp_acceptance == null and
            o.mtp_greedy_tail == null and o.int8_prefill == null and o.chat_template_kwargs == null and o.drafter == null;
    }

    pub fn deinit(o: *Override, alloc: std.mem.Allocator) void {
        if (o.chat_template_kwargs) |k| alloc.free(k);
        if (o.drafter) |d| alloc.free(d);
        o.chat_template_kwargs = null;
        o.drafter = null;
    }
};

pub const Settings = struct {
    parsed: ?std.json.Parsed(std.json.Value) = null,

    pub fn deinit(self: *Settings) void {
        if (self.parsed) |*p| p.deinit();
        self.parsed = null;
    }

    /// The returned Override owns its strings (`deinit`).
    pub fn lookup(self: *const Settings, alloc: std.mem.Allocator, model_path: []const u8) Override {
        const p = self.parsed orelse return .{};
        const root = switch (p.value) {
            .object => |o| o,
            else => return .{},
        };
        const want = trimSlash(model_path);
        var it = root.iterator();
        while (it.next()) |kv| {
            if (!std.mem.eql(u8, trimSlash(kv.key_ptr.*), want)) continue;
            return fromValue(alloc, kv.value_ptr.*);
        }
        return .{};
    }

    /// The model path whose `alias` is `name`; a name claimed twice goes to the
    /// lowest path. Borrowed from the parsed file.
    pub fn pathForAlias(self: *const Settings, name: []const u8) ?[]const u8 {
        const root = self.rootObject() orelse return null;
        var best: ?[]const u8 = null;
        var it = root.iterator();
        while (it.next()) |kv| {
            const a = aliasOf(kv.value_ptr.*) orelse continue;
            if (!std.mem.eql(u8, a, name)) continue;
            const p = trimSlash(kv.key_ptr.*);
            if (best == null or std.mem.lessThan(u8, p, best.?)) best = p;
        }
        return best;
    }

    /// The valid alias set for `model_path`, borrowed from the parsed file.
    pub fn aliasForPath(self: *const Settings, model_path: []const u8) ?[]const u8 {
        const root = self.rootObject() orelse return null;
        const want = trimSlash(model_path);
        var it = root.iterator();
        while (it.next()) |kv| {
            if (std.mem.eql(u8, trimSlash(kv.key_ptr.*), want)) return aliasOf(kv.value_ptr.*);
        }
        return null;
    }

    pub const Duplicate = struct { alias: []const u8, a: []const u8, b: []const u8 };

    /// The first alias that two model paths both claim, if any.
    pub fn duplicateAlias(self: *const Settings) ?Duplicate {
        const root = self.rootObject() orelse return null;
        const keys = root.keys();
        const vals = root.values();
        for (vals, 0..) |v, i| {
            const a = aliasOf(v) orelse continue;
            for (vals[i + 1 ..], keys[i + 1 ..]) |w, k| {
                const b = aliasOf(w) orelse continue;
                if (std.mem.eql(u8, a, b)) return .{ .alias = a, .a = trimSlash(keys[i]), .b = trimSlash(k) };
            }
        }
        return null;
    }

    fn rootObject(self: *const Settings) ?std.json.ObjectMap {
        const p = self.parsed orelse return null;
        return switch (p.value) {
            .object => |o| o,
            else => null,
        };
    }
};

/// An alias is a bare request name: `@` would route it to a LAN peer or a
/// provider, `/` reads as an org/repo id, and "mlx-serve" means the default.
pub fn validAlias(name: []const u8) bool {
    if (name.len == 0 or name.len > 128 or std.mem.eql(u8, name, "mlx-serve")) return false;
    for (name) |c| if (c <= ' ' or c == '@' or c == '/' or c == '"' or c == '\\' or c == 0x7f) return false;
    return true;
}

fn aliasOf(v: std.json.Value) ?[]const u8 {
    if (v != .object) return null;
    const a = v.object.get("alias") orelse return null;
    if (a != .string or !validAlias(a.string)) return null;
    return a.string;
}

fn trimSlash(p: []const u8) []const u8 {
    var s = p;
    while (s.len > 1 and s[s.len - 1] == '/') s = s[0 .. s.len - 1];
    return s;
}

fn fromValue(alloc: std.mem.Allocator, v: std.json.Value) Override {
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
    if (obj.get("int8_prefill")) |m| switch (m) {
        .bool => |b| o.int8_prefill = b,
        else => {},
    };
    if (obj.get("mtp_greedy_tail")) |g| switch (g) {
        .bool => |b| o.mtp_greedy_tail = b,
        else => {},
    };
    if (obj.get("drafter")) |d| if (d == .string) {
        const name = d.string;
        if (std.mem.eql(u8, name, "off") or std.mem.eql(u8, name, "auto") or std.fs.path.isAbsolute(name))
            o.drafter = alloc.dupe(u8, name) catch null;
    };
    if (obj.get("chat_template_kwargs")) |k| if (k == .object) {
        o.chat_template_kwargs = std.json.Stringify.valueAlloc(alloc, k, .{}) catch null;
    };
    return o;
}

/// A model entry's generation rules: its `generation_defaults`, plus the `enable_thinking` /
/// `reasoning_effort` of its `chat_template_kwargs` where no rule names them.
fn generationProfile(obj: std.json.ObjectMap) generation.Profile {
    var p = if (obj.get("generation_defaults")) |v| generation.parseProfile(v) catch |err| blk: {
        log.warn("[model-settings] generation_defaults ignored ({s})\n", .{@errorName(err)});
        break :blk generation.Profile{};
    } else generation.Profile{};
    const kw = obj.get("chat_template_kwargs") orelse return p;
    if (kw != .object) return p;
    if (p.rules.get(.enable_thinking) == null) if (kw.object.get("enable_thinking")) |v| if (v == .bool)
        p.set(.enable_thinking, .{ .boolean = v.bool }, false);
    if (p.rules.get(.reasoning_effort) == null) if (kw.object.get("reasoning_effort")) |v| if (v == .string)
        if (std.meta.stringToEnum(generation.Effort, v.string)) |e| p.set(.reasoning_effort, .{ .effort = e }, false);
    return p;
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
    return homeFile(buf, "model-settings.json");
}

/// `~/.mlx-serve/<name>`; "" when it does not fit `buf`.
pub fn homeFile(buf: []u8, name: []const u8) []const u8 {
    const home = std.mem.span(std.c.getenv("HOME") orelse "/tmp");
    return std.fmt.bufPrint(buf, "{s}/.mlx-serve/{s}", .{ home, name }) catch "";
}

/// The parsed file kept for the request path: re-stats at most once per
/// `recheck_ms` and re-parses only when size, mtime or inode moved (the app
/// saves atomically, so a save is a new inode).
pub const Cache = struct {
    path: []const u8,
    /// Owns every parse; set with `path`. Null (or an empty path) = never reads.
    alloc: ?std.mem.Allocator = null,
    recheck_ms: i64 = 1000,
    mutex: std.Io.Mutex = .init,
    settings: Settings = .{},
    stamp: ?Stamp = null,
    checked_ms: ?i64 = null,

    const Stamp = struct { size: u64, mtime: i96, inode: u64 };

    pub fn deinit(self: *Cache) void {
        self.settings.deinit();
    }

    /// The model path answering to `name`, copied into `buf`.
    pub fn pathForAlias(self: *Cache, io: std.Io, name: []const u8, buf: []u8) ?[]const u8 {
        if (!validAlias(name)) return null;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.refresh(io);
        return copyInto(buf, self.settings.pathForAlias(name));
    }

    /// The alias of `model_path`, copied into `buf`.
    pub fn aliasForPath(self: *Cache, io: std.Io, model_path: []const u8, buf: []u8) ?[]const u8 {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.refresh(io);
        return copyInto(buf, self.settings.aliasForPath(model_path));
    }

    /// The generation rules of `model_path`'s entry; null reads the whole file as one
    /// profile (`generation-settings.json`). Read per request, so an edit applies to the next one.
    pub fn generationDefaults(self: *Cache, io: std.Io, model_path: ?[]const u8) generation.Profile {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.refresh(io);
        const parsed = self.settings.parsed orelse return .{};
        const path = model_path orelse return generation.parseProfile(parsed.value) catch |err| {
            log.warn("[generation-settings] {s} ignored ({s})\n", .{ self.path, @errorName(err) });
            return .{};
        };
        if (parsed.value != .object) return .{};
        var it = parsed.value.object.iterator();
        while (it.next()) |kv| {
            if (!std.mem.eql(u8, trimSlash(kv.key_ptr.*), trimSlash(path))) continue;
            return if (kv.value_ptr.* == .object) generationProfile(kv.value_ptr.object) else .{};
        }
        return .{};
    }

    /// Re-read the file if it changed; lookups call it, and a caller that must
    /// not do I/O later (under another lock) can call it first.
    pub fn refreshNow(self: *Cache, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.refresh(io);
    }

    fn refresh(self: *Cache, io: std.Io) void {
        const alloc = self.alloc orelse return;
        if (self.path.len == 0) return;
        const now = std.Io.Timestamp.now(io, .boot).toMilliseconds();
        if (self.checked_ms) |c| if (now - c < self.recheck_ms) return;
        self.checked_ms = now;
        const stamp: ?Stamp = if (std.Io.Dir.cwd().statFile(io, self.path, .{})) |st|
            .{ .size = st.size, .mtime = st.mtime.nanoseconds, .inode = @intCast(st.inode) }
        else |_|
            null;
        if (std.meta.eql(stamp, self.stamp)) return;
        self.stamp = stamp;
        self.settings.deinit();
        self.settings = load(alloc, io, self.path);
        if (self.settings.duplicateAlias()) |d|
            log.warn("[model-settings] alias \"{s}\" claimed by {s} and {s}; the lower path wins\n", .{ d.alias, d.a, d.b });
    }

    fn copyInto(buf: []u8, s: ?[]const u8) ?[]const u8 {
        const v = s orelse return null;
        if (v.len > buf.len) return null;
        @memcpy(buf[0..v.len], v);
        return buf[0..v.len];
    }
};

/// The one call load sites make: read the default file, look the model up, log
/// a hit. The caller owns the result (`Override.deinit`).
pub fn overrideFor(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8) Override {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    var s = load(alloc, io, defaultPath(&buf));
    defer s.deinit();
    var o = s.lookup(alloc, model_path);
    if (o.drafter) |d| if (std.fs.path.isAbsolute(d)) {
        var cfg_buf: [std.fs.max_path_bytes]u8 = undefined;
        const cfg = std.fmt.bufPrint(&cfg_buf, "{s}/config.json", .{d}) catch "";
        std.Io.Dir.cwd().access(io, cfg, .{}) catch {
            log.warn("[model-settings] {s}: drafter {s} has no config.json, using auto\n", .{ model_path, d });
            alloc.free(d);
            o.drafter = null;
        };
    };
    if (!o.isEmpty()) log.info("[model-settings] {s}: ctx={d} kv={s} mtp={s} accept={s} greedy_tail={s} int8={s} drafter={s} kwargs={s}\n", .{
        model_path,
        o.ctx_size orelse 0,
        if (o.kv_quant) |k| k.wireName() else "default",
        if (o.mtp) |m| (if (m) "on" else "off") else "default",
        if (o.mtp_acceptance) |a| mtp_acceptance.name(a) else "default",
        if (o.mtp_greedy_tail) |g| (if (g) "on" else "off") else "default",
        if (o.int8_prefill) |b| (if (b) "on" else "off") else "default",
        o.drafter orelse "auto",
        o.chat_template_kwargs orelse "none",
    });
    return o;
}

test "model_settings: parse + lookup with and without trailing slash" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a/": {"ctx_size": 65536, "kv_quant": "8", "mtp": false}, "/m/b": {"kv_quant": 4}}
    );
    defer s.deinit();
    const t = std.testing.allocator;
    const a = s.lookup(t, "/m/a");
    try std.testing.expectEqual(@as(?u32, 65536), a.ctx_size);
    try std.testing.expectEqual(@as(u8, 8), a.kv_quant.?.bits);
    try std.testing.expectEqual(@as(?bool, false), a.mtp);
    const b = s.lookup(t, "/m/b/");
    try std.testing.expectEqual(@as(?u32, null), b.ctx_size);
    try std.testing.expectEqual(@as(u8, 4), b.kv_quant.?.bits);
    try std.testing.expectEqual(@as(?bool, null), b.mtp);
    try std.testing.expect(s.lookup(t, "/m/c").isEmpty());
}

test "model_settings: chat_template_kwargs is an object carried verbatim, anything else is unset" {
    const t = std.testing.allocator;
    var s = try parse(t,
        \\{"/m/a": {"chat_template_kwargs": {"preserve_thinking": true, "x": [1]}}, "/m/b": {"chat_template_kwargs": "yes"},
        \\ "/m/c": {"chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "high"}}}
    );
    defer s.deinit();
    var a = s.lookup(t, "/m/a");
    defer a.deinit(t);
    try std.testing.expectEqualStrings("{\"preserve_thinking\":true,\"x\":[1]}", a.chat_template_kwargs.?);
    try std.testing.expect(s.lookup(t, "/m/b").isEmpty());
}

test "model_settings: generation rules re-read per request; kwargs thinking keys fill rules the entry leaves unset" {
    const t = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(io, &dir_buf)];
    const path = try std.fs.path.join(t, &.{ dir, "model-settings.json" });
    defer t.free(path);
    var c: Cache = .{ .path = path, .alloc = t, .recheck_ms = 0 };
    defer c.deinit();
    try tmp.dir.writeFile(io, .{ .sub_path = "model-settings.json", .data =
        \\{"/m/a/": {"chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "high"},
        \\  "generation_defaults": {"reasoning_effort": {"value": "low", "ignore_client": true}}}}
    });
    var p = c.generationDefaults(io, "/m/a");
    try std.testing.expectEqual(true, p.value(bool, .enable_thinking).?);
    try std.testing.expectEqual(generation.Effort.low, p.value(generation.Effort, .reasoning_effort).?);
    try std.testing.expect(p.forced(.reasoning_effort));
    try tmp.dir.writeFile(io, .{ .sub_path = "model-settings.json", .data =
        \\{"/m/a": {"generation_defaults": {"temperature": {"value": 0.5}, "top_k": {"value": -1}}}}
    });
    p = c.generationDefaults(io, "/m/a"); // a bad rule drops the profile, never the request
    try std.testing.expect(p.rules.get(.temperature) == null);
    try tmp.dir.writeFile(io, .{ .sub_path = "model-settings.json", .data = "{\"max_tokens\": {\"value\": 12, \"ignore_client\": true}}" });
    p = c.generationDefaults(io, null);
    try std.testing.expectEqual(@as(u32, 12), p.resolve(u32, .max_tokens, 64, 0));
}

test "model_settings: mtp_acceptance names a mode at its default threshold" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a": {"mtp_acceptance": "typical"}, "/m/b": {"mtp_acceptance": "tokenv3"}, "/m/c": {"mtp_acceptance": "exact"}, "/m/d": {"mtp_acceptance": "fast"}}
    );
    defer s.deinit();
    const t = std.testing.allocator;
    try std.testing.expectEqual(@as(f32, 0.2), s.lookup(t, "/m/a").mtp_acceptance.?.typical.delta);
    try std.testing.expectEqual(@as(f32, 0.95), s.lookup(t, "/m/b").mtp_acceptance.?.tokenv3);
    try std.testing.expect(s.lookup(t, "/m/c").mtp_acceptance.? == .exact);
    try std.testing.expect(s.lookup(t, "/m/d").isEmpty());
}

test "model_settings: the greedy tail (mtp_greedy_tail) is a bool, anything else is unset" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a": {"mtp_greedy_tail": true}, "/m/b": {"mtp_greedy_tail": false}, "/m/c": {"mtp_greedy_tail": "on"}}
    );
    defer s.deinit();
    const t = std.testing.allocator;
    try std.testing.expectEqual(@as(?bool, true), s.lookup(t, "/m/a").mtp_greedy_tail);
    try std.testing.expectEqual(@as(?bool, false), s.lookup(t, "/m/b").mtp_greedy_tail);
    try std.testing.expect(!s.lookup(t, "/m/b").isEmpty());
    try std.testing.expect(s.lookup(t, "/m/c").isEmpty());
}

test "model_settings: int8_prefill is a boolean, anything else is unset" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a": {"int8_prefill": true}, "/m/b": {"int8_prefill": false}, "/m/c": {"int8_prefill": "yes"}}
    );
    defer s.deinit();
    const t = std.testing.allocator;
    try std.testing.expectEqual(@as(?bool, true), s.lookup(t, "/m/a").int8_prefill);
    try std.testing.expectEqual(@as(?bool, false), s.lookup(t, "/m/b").int8_prefill);
    try std.testing.expect(!s.lookup(t, "/m/b").isEmpty());
    try std.testing.expect(s.lookup(t, "/m/c").isEmpty());
}

test "model_settings: bad values ignored, bad JSON = empty" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a": {"ctx_size": 0, "kv_quant": "16", "mtp": "yes", "future": 1}}
    );
    defer s.deinit();
    try std.testing.expect(s.lookup(std.testing.allocator, "/m/a").isEmpty());
    try std.testing.expectError(error.SyntaxError, parse(std.testing.allocator, "{nope"));
    var empty = load(std.testing.allocator, std.testing.io, "/nonexistent/model-settings.json");
    defer empty.deinit();
    try std.testing.expect(empty.lookup(std.testing.allocator, "/m/a").isEmpty());
}

test "model_settings: drafter is off, auto or an absolute path; anything else is unset" {
    const t = std.testing.allocator;
    var s = try parse(t,
        \\{"/m/a": {"drafter": "off"}, "/m/b": {"drafter": "/d/x"}, "/m/c": {"drafter": "auto"}, "/m/d": {"drafter": "rel/x"}, "/m/e": {"drafter": true}}
    );
    defer s.deinit();
    inline for (.{ .{ "/m/a", "off" }, .{ "/m/b", "/d/x" }, .{ "/m/c", "auto" } }) |c| {
        var o = s.lookup(t, c[0]);
        defer o.deinit(t);
        try std.testing.expectEqualStrings(c[1], o.drafter.?);
        try std.testing.expect(!o.isEmpty());
    }
    try std.testing.expect(s.lookup(t, "/m/d").isEmpty());
    try std.testing.expect(s.lookup(t, "/m/e").isEmpty());
}

test "model_settings: an alias names its model path; an invalid alias names nothing" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a/": {"alias": "qwen"}, "/m/b": {"alias": "mlx-serve"}, "/m/c": {"alias": "x@peer"},
        \\ "/m/d": {"alias": "org/x"}, "/m/e": {"alias": ""}, "/m/f": {"alias": 3}, "/m/g": {"alias": "two words"}}
    );
    defer s.deinit();
    try std.testing.expectEqualStrings("/m/a", s.pathForAlias("qwen").?);
    try std.testing.expectEqualStrings("qwen", s.aliasForPath("/m/a/").?);
    for ([_][]const u8{ "mlx-serve", "x@peer", "org/x", "", "3", "two words", "Qwen" }) |name|
        try std.testing.expect(s.pathForAlias(name) == null);
    for ([_][]const u8{ "/m/b", "/m/c", "/m/d", "/m/e", "/m/f", "/m/g" }) |p|
        try std.testing.expect(s.aliasForPath(p) == null);
    // An alias is a name, not a load setting.
    try std.testing.expect(s.lookup(std.testing.allocator, "/m/a").isEmpty());
}

test "model_settings: an alias claimed twice resolves to the lowest path" {
    var s = try parse(std.testing.allocator,
        \\{"/m/z": {"alias": "q"}, "/m/b": {"alias": "q"}, "/m/k": {"alias": "q"}}
    );
    defer s.deinit();
    try std.testing.expectEqualStrings("/m/b", s.pathForAlias("q").?);
}

test "model_settings: the alias cache rereads the file only after it changes, at most once per interval" {
    const t = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(io, &dir_buf)];
    const path = try std.fs.path.join(t, &.{ dir, "model-settings.json" });
    defer t.free(path);

    try tmp.dir.writeFile(io, .{ .sub_path = "model-settings.json", .data = "{\"/m/a\": {\"alias\": \"q\"}}" });
    var c: Cache = .{ .path = path, .alloc = t, .recheck_ms = std.math.maxInt(i64) };
    defer c.deinit();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    try std.testing.expectEqualStrings("/m/a", c.pathForAlias(io, "q", &buf).?);

    try tmp.dir.writeFile(io, .{ .sub_path = "model-settings.json", .data = "{\"/m/bb\": {\"alias\": \"q\"}}" });
    // Inside the interval the parsed copy answers; no stat, no read.
    try std.testing.expectEqualStrings("/m/a", c.pathForAlias(io, "q", &buf).?);
    c.recheck_ms = 0;
    try std.testing.expectEqualStrings("/m/bb", c.pathForAlias(io, "q", &buf).?);
    try std.testing.expectEqualStrings("q", c.aliasForPath(io, "/m/bb/", &buf).?);
    try std.testing.expect(c.aliasForPath(io, "/m/a", &buf) == null);
}

test "model_settings: an alias two models claim is named with both paths" {
    var s = try parse(std.testing.allocator,
        \\{"/m/z": {"alias": "q"}, "/m/b": {"alias": "q"}, "/m/k": {"alias": "k"}}
    );
    defer s.deinit();
    const d = s.duplicateAlias().?;
    try std.testing.expectEqualStrings("q", d.alias);
    try std.testing.expect(!std.mem.eql(u8, d.a, d.b));
    var one = try parse(std.testing.allocator,
        \\{"/m/z": {"alias": "q"}, "/m/k": {"alias": "k"}}
    );
    defer one.deinit();
    try std.testing.expect(one.duplicateAlias() == null);
}
