//! Per-model settings (`~/.mlx-serve/model-settings.json`): context size, KV
//! quant and MTP that follow the MODEL, applied at every load construction
//! site. Keyed by the model's absolute path (dir, or the `.gguf` file).
//! The app edits the file; the server owns applying it. A malformed file is
//! logged and treated as empty: a settings typo must never stop a load.
const std = @import("std");
const kv_quant = @import("kv_quant.zig");
const log = @import("log.zig");
const mtp_acceptance = @import("mtp_acceptance.zig");

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
    /// Its `enable_thinking` / `reasoning_effort`: defaults for a request that
    /// names neither. Effort owned.
    enable_thinking: ?bool = null,
    reasoning_effort: ?[]const u8 = null,
    /// The speculation sidecar: "off", "auto" (the pack's own `drafter/`) or an
    /// absolute path. Owned.
    drafter: ?[]const u8 = null,

    pub fn isEmpty(o: Override) bool {
        return o.ctx_size == null and o.kv_quant == null and o.mtp == null and o.mtp_acceptance == null and
            o.mtp_greedy_tail == null and o.int8_prefill == null and o.chat_template_kwargs == null and o.drafter == null;
    }

    pub fn deinit(o: *Override, alloc: std.mem.Allocator) void {
        if (o.chat_template_kwargs) |k| alloc.free(k);
        if (o.reasoning_effort) |e| alloc.free(e);
        if (o.drafter) |d| alloc.free(d);
        o.chat_template_kwargs = null;
        o.reasoning_effort = null;
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
};

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
        if (k.object.get("enable_thinking")) |e| if (e == .bool) {
            o.enable_thinking = e.bool;
        };
        if (k.object.get("reasoning_effort")) |e| if (e == .string) {
            o.reasoning_effort = alloc.dupe(u8, e.string) catch null;
        };
    };
    return o;
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

/// The one call load sites make: read the default file, look the model up, log
/// a hit. The caller owns the result (`Override.deinit`).
pub fn overrideFor(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8) Override {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    var s = load(alloc, io, defaultPath(&buf));
    defer s.deinit();
    var o = s.lookup(alloc, model_path);
    if (o.drafter) |d| if (std.fs.path.isAbsolute(d)) {
        const cfg = std.fs.path.join(alloc, &.{ d, "config.json" }) catch "";
        defer alloc.free(cfg);
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
    try std.testing.expectEqual(@as(?bool, null), a.enable_thinking);
    // The thinking keys are typed out: they are request defaults, not template text.
    var c = s.lookup(t, "/m/c");
    defer c.deinit(t);
    try std.testing.expectEqual(@as(?bool, true), c.enable_thinking);
    try std.testing.expectEqualStrings("high", c.reasoning_effort.?);
    try std.testing.expect(s.lookup(t, "/m/b").isEmpty());
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
