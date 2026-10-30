//! Outside plugins: the dylibs `mlx-serve plugin install` put in `~/.mlx-serve/plugins/`, loaded once at startup
//! when `plugins.json` switches them on. Each must match its install receipt's hash before it is opened, then is
//! negotiated (`abi.dylib.negotiate`) and joins the claims (`outsideWinner`). A refusal names the file and the
//! reason; the server serves on without it. A dylib is never unloaded (it may hold MLX arrays and threads).

const std = @import("std");
const env = @import("../../env.zig");
const builtin = @import("builtin");
const common = @import("../../abi/common.zig");
const dylib_abi = @import("../../abi/dylib.zig");
const sdk_plugin = @import("../../sdk/plugin.zig");
const log = @import("../../sdk/util/log.zig");
const FsPackDir = @import("../registry/pack_dir.zig").FsPackDir;

const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

/// Outside plugins load on the macOS CLI and app; the Mac App Store build and iOS stay static (sandbox).
pub const supported = builtin.os.tag == .macos and !@import("build_options").mas;

/// `~/.mlx-serve/plugins.json`: the Settings switch (off unless the user turned it on) and the plugins turned off.
pub const Settings = struct {
    enabled: bool = false,
    disabled: []const []const u8 = &.{},

    pub fn isDisabled(self: Settings, name: []const u8) bool {
        for (self.disabled) |d| if (std.mem.eql(u8, d, name)) return true;
        return false;
    }
};

/// `plugins.json` at `path`; a missing or broken file is the default (off).
pub fn readSettings(io: std.Io, a: Allocator, path: []const u8) Settings {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 16)) catch return .{};
    return std.json.parseFromSliceLeaky(Settings, a, bytes, .{ .ignore_unknown_fields = true }) catch .{};
}

/// What `plugin install` records beside `<name>.dylib` as `<name>.json` (the app's Settings lists these).
pub const Receipt = struct {
    name: []const u8,
    kind: []const u8,
    /// The plugin's own version (its root's `version`), "" when it declares none.
    version: []const u8 = "",
    sha256: []const u8,
    /// Who signed the file, as `codesign` reports it (`signatureOf`).
    signature: []const u8 = "",
    source: []const u8 = "",
    /// The `model_type`s it declares: the app lists their packs as the plugin's.
    model_types: []const []const u8 = &.{},
};

/// The signer of a code-signed file from `codesign -dvv`'s report: its first `Authority=` (with the team), "ad-hoc", or
/// "unsigned".
pub fn signatureFromReport(report: []const u8) []const u8 {
    var it = std.mem.splitScalar(u8, report, '\n');
    while (it.next()) |line| {
        if (std.mem.startsWith(u8, line, "Authority=")) return line["Authority=".len..];
        if (std.mem.eql(u8, line, "Signature=adhoc")) return "ad-hoc";
    }
    return "unsigned";
}

/// The signer of the file at `path` (`signatureFromReport`); "unknown" where `codesign` cannot run. Use an arena: the
/// result is in `a` or static.
pub fn signatureOf(io: std.Io, a: Allocator, path: []const u8) []const u8 {
    if (builtin.os.tag != .macos) return "unknown";
    const r = std.process.run(a, io, .{ .argv = &.{ "/usr/bin/codesign", "-dvv", path } }) catch return "unknown";
    // codesign reports on stderr.
    return signatureFromReport(r.stderr);
}

pub const Plugin = struct {
    /// The installed file.
    path: []const u8,
    /// Lives in the dylib, which stays loaded for the process.
    manifest: *const dylib_abi.Manifest,
    sha256: [Sha256.digest_length * 2]u8,

    pub fn name(self: Plugin) []const u8 {
        return self.manifest.name.slice();
    }
    pub fn version(self: Plugin) []const u8 {
        return self.manifest.plugin_version.slice();
    }
    pub fn entry(self: Plugin) sdk_plugin.Entry {
        const m = self.manifest;
        var e: sdk_plugin.Entry = .{ .plugin = .{ .name = m.name.slice(), .kind = switch (m.kind) {
            .model => .model,
            .media => .media,
            .quant => .quant,
        } } };
        switch (m.kind) {
            .model => e.model = @ptrCast(@alignCast(m.table)),
            .media => e.media = @ptrCast(@alignCast(m.table)),
            .quant => e.quant = @ptrCast(@alignCast(m.table)),
        }
        return e;
    }
};

pub const Refusal = struct { path: []const u8, why: []const u8 };

pub const Set = struct {
    plugins: std.ArrayList(Plugin) = .empty,
    refused: std.ArrayList(Refusal) = .empty,
    /// One entry per plugin, in `plugins` order.
    entries: std.ArrayList(sdk_plugin.Entry) = .empty,

    /// Frees the lists and their text; the dylibs stay loaded.
    pub fn deinit(self: *Set, a: Allocator) void {
        for (self.plugins.items) |p| a.free(p.path);
        for (self.refused.items) |r| {
            a.free(r.path);
            a.free(r.why);
        }
        self.plugins.deinit(a);
        self.refused.deinit(a);
        self.entries.deinit(a);
    }
};

/// The hex SHA-256 of `bytes`.
pub fn sha256Hex(bytes: []const u8) [Sha256.digest_length * 2]u8 {
    var d: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

pub const Opened = union(enum) {
    ok: Plugin,
    /// Allocated in the caller's allocator.
    refused: []const u8,
};

/// Opens the dylib at `path` (now: every mlx-c symbol it names must exist in this process) and negotiates its
/// manifest against this build's MLX `host_mlx`. `sha256` is the file's, checked by the caller.
pub fn open(a: Allocator, path: []const u8, sha256: [Sha256.digest_length * 2]u8, host_mlx: []const u8) !Opened {
    if (!supported) return .{ .refused = try a.dupe(u8, "does not load: outside plugins are unsupported on this build") };
    const z = try a.dupeSentinel(u8, path, 0);
    defer a.free(z);
    const mode: std.c.RTLD = if (builtin.os.tag == .macos) .{ .NOW = true, .LOCAL = true } else .{ .NOW = true };
    const handle = std.c.dlopen(z, mode) orelse
        return .{ .refused = try std.fmt.allocPrint(a, "does not load: {s}", .{if (std.c.dlerror()) |e| std.mem.span(e) else "dlopen failed"}) };
    const sym = std.c.dlsym(handle, dylib_abi.symbol) orelse
        return .{ .refused = try a.dupe(u8, "exports no `" ++ dylib_abi.symbol ++ "` (not built with the plugin kit)") };
    const m: *const dylib_abi.Manifest = @ptrCast(@alignCast(sym));
    const v = dylib_abi.negotiate(m, host_mlx);
    if (!v.isOk()) return .{ .refused = try a.dupe(u8, v.why.slice()) };
    return .{ .ok = .{ .path = try a.dupe(u8, path), .manifest = m, .sha256 = sha256 } };
}

/// Loads every installed dylib in `dir` that `settings` leaves on. Plugins load in name order.
pub fn loadDir(io: std.Io, a: Allocator, dir_path: []const u8, settings: Settings, host_mlx: []const u8) !Set {
    var set: Set = .{};
    errdefer set.deinit(a);
    var dir = std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return set;
    defer dir.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    defer {
        for (names.items) |n| a.free(n);
        names.deinit(a);
    }
    var it = dir.iterate();
    while (try it.next(io)) |e| if (e.kind != .directory and std.mem.endsWith(u8, e.name, ".dylib")) try names.append(a, try a.dupe(u8, e.name));
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    for (names.items) |file| {
        const stem = file[0 .. file.len - ".dylib".len];
        if (settings.isDisabled(stem)) continue;
        const path = try std.fs.path.join(a, &.{ dir_path, file });
        defer a.free(path);
        const got = try loadInstalled(io, a, dir, stem, path, host_mlx);
        switch (got) {
            .ok => |p| {
                try set.plugins.append(a, p);
                try set.entries.append(a, p.entry());
            },
            .refused => |why| try set.refused.append(a, .{ .path = try a.dupe(u8, path), .why = why }),
        }
    }
    return set;
}

/// One installed dylib: its receipt's hash first (nothing runs before it matches), then `open`.
fn loadInstalled(io: std.Io, a: Allocator, dir: std.Io.Dir, stem: []const u8, path: []const u8, host_mlx: []const u8) !Opened {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    const receipt_bytes = dir.readFileAlloc(io, try std.fmt.allocPrint(aa, "{s}.json", .{stem}), aa, .limited(1 << 16)) catch
        return .{ .refused = try a.dupe(u8, "has no install receipt: install it with `mlx-serve plugin install`") };
    const receipt = std.json.parseFromSliceLeaky(Receipt, aa, receipt_bytes, .{ .ignore_unknown_fields = true }) catch
        return .{ .refused = try a.dupe(u8, "its install receipt does not parse") };
    const bytes = dir.readFileAlloc(io, try std.fmt.allocPrint(aa, "{s}.dylib", .{stem}), aa, .limited(1 << 30)) catch
        return .{ .refused = try a.dupe(u8, "cannot be read") };
    const hash = sha256Hex(bytes);
    if (!std.mem.eql(u8, &hash, receipt.sha256)) return .{ .refused = try a.dupe(u8, "changed since it was installed (hash differs from its receipt)") };
    const got = try open(a, path, hash, host_mlx);
    if (got == .ok and !std.mem.eql(u8, got.ok.name(), stem)) {
        a.free(got.ok.path);
        return .{ .refused = try a.dupe(u8, "its manifest names another plugin than its file") };
    }
    return got;
}

/// Which outside plugin serves a pack over the built-ins, as an index into `outside`; null leaves it to the built-ins.
/// An outside plugin with a built-in's name replaces that built-in; otherwise the higher claim rank wins and a
/// built-in wins a tie. Two outside plugins tied at the top are ambiguous: neither wins.
pub fn outsideWinner(builtins: []const sdk_plugin.Entry, outside: []const sdk_plugin.Entry, model_type: []const u8, config_json: []const u8, dir: *const common.PackDir) ?usize {
    if (outside.len == 0 or outside.len > 64) return null;
    var out_ranks: [64]common.Claim = undefined;
    const got = sdk_plugin.claimsOf(outside, model_type, config_json, dir, out_ranks[0..outside.len]);
    const i = (common.pick(got) catch return null) orelse return null;
    var top: u8 = 0;
    for (builtins) |e| {
        if (replaced(e.plugin.name, outside)) continue;
        var one: [1]common.Claim = undefined;
        top = @max(top, @backingInt(sdk_plugin.claimsOf(&.{e}, model_type, config_json, dir, &one)[0]));
    }
    return if (@backingInt(got[i]) > top) i else null;
}

fn replaced(name: []const u8, outside: []const sdk_plugin.Entry) bool {
    for (outside) |o| if (std.mem.eql(u8, o.plugin.name, name)) return true;
    return false;
}

// ── the process's set ──

const State = struct { set: Set = .{}, builtins: []const sdk_plugin.Entry = &.{} };
/// Set once at startup, before any server thread; read-only after.
pub const process = @import("../../sdk/util/shared.zig").Shared(State, "outside_plugins", .{});

/// The outside plugins this process loaded.
pub fn loaded() []const Plugin {
    return process.get().set.plugins.items;
}

/// Their registry entries, in `loaded()` order.
pub fn entries() []const sdk_plugin.Entry {
    return process.get().set.entries.items;
}

/// The loaded outside plugin that serves a pack over the built-ins (`outsideWinner`), or null (at once when none
/// is loaded).
pub fn winner(model_type: []const u8, config_json: []const u8, dir: *const common.PackDir) ?*const Plugin {
    const st = process.get();
    const i = outsideWinner(st.builtins, st.set.entries.items, model_type, config_json, dir) orelse return null;
    return &st.set.plugins.items[i];
}

/// The name of the loaded outside plugin serving the pack at absolute `path` (its `/v1/models` badge), or null.
pub fn servingName(io: std.Io, path: []const u8) ?[]const u8 {
    if (loaded().len == 0) return null;
    var fs = FsPackDir.open(io, path) catch return null;
    defer fs.close();
    const dir = fs.packDir();
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const n = dir.fileBytes("config.json");
    var why: common.Verdict = .{};
    const cfg: []const u8 = if (n == 0 or n > 4 << 20) "" else dir.readFile("config.json", arena.allocator().alloc(u8, @intCast(n)) catch return null, &why) orelse "";
    const mt = sdk_plugin.modelTypeOf(arena.allocator(), cfg) catch return null;
    return (winner(mt, cfg, &dir) orelse return null).name();
}

/// The loaded outside plugin whose table is `table`, or null (a built-in's).
pub fn ownerOf(table: *const anyopaque) ?*const Plugin {
    for (process.get().set.plugins.items) |*p| if (p.manifest.table == table) return p;
    return null;
}

/// `~/.mlx-serve/<rel>`.
pub fn homePath(a: Allocator, rel: []const u8) ![]u8 {
    const home = std.mem.span(env.get(.HOME) orelse return error.NoHome);
    return std.fs.path.join(a, &.{ home, ".mlx-serve", rel });
}

/// At startup: publishes the process's `Host` table, which every model load hands its plugin, then loads the installed
/// plugins when `plugins.json` switches them on and `--no-plugins` was not given, logging each plugin loaded or refused
/// and each built-in an outside plugin replaces.
pub fn loadAtStartup(io: std.Io, a: Allocator, builtins: []const sdk_plugin.Entry, cli_off: bool, host_mlx: []const u8) void {
    _ = @import("../plugin_host_impl.zig").processHost();
    if (!supported) return;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const settings_path = homePath(arena.allocator(), "plugins.json") catch return;
    const settings = readSettings(io, arena.allocator(), settings_path);
    if (!settings.enabled or cli_off) {
        log.debug("[plugins] outside plugins off ({s})\n", .{if (cli_off) "--no-plugins" else "plugins.json"});
        return;
    }
    const dir = homePath(arena.allocator(), "plugins") catch return;
    const st = process.get();
    st.builtins = builtins;
    st.set = loadDir(io, a, dir, settings, host_mlx) catch |e| {
        log.warn("[plugins] {s}: {t}\n", .{ dir, e });
        return;
    };
    for (st.set.refused.items) |r| log.warn("[plugins] refused {s}: {s}\n", .{ r.path, r.why });
    for (st.set.plugins.items) |p| {
        log.info("[plugins] loaded {s} ({t}, sha256 {s}) from {s}\n", .{ p.name(), p.manifest.kind, p.sha256[0..12], p.path });
        for (builtins) |e| if (std.mem.eql(u8, e.plugin.name, p.name())) log.info("[plugins] {s} replaces the built-in plugin of that name\n", .{p.name()});
    }
}

// ── tests ──

const testing = std.testing;
const fixture = @import("../../sdk/testing/fixture.zig");
const model_abi = @import("../../abi/plugin.zig");

test "outside plugins: open refuses unsupported builds before loading a library" {
    if (supported) return error.SkipZigTest;
    const result = try open(testing.allocator, "missing.dylib", @splat(0), "pin");
    switch (result) {
        .refused => |why| {
            defer testing.allocator.free(why);
            try testing.expectEqualStrings("does not load: outside plugins are unsupported on this build", why);
        },
        .ok => return error.TestUnexpectedResult,
    }
}

test {
    _ = @import("cli.zig");
    _ = @import("kit_test.zig");
    _ = dylib_abi;
    _ = @import("../../sdk/dylib.zig");
}

fn fakeModel(comptime name: []const u8, comptime types: []const []const u8) model_abi.ModelTable {
    return .{ .name = .of(name), .claims = sdk_plugin.claimsByType(types), .parse = undefined, .free_parsed = undefined, .bill = undefined, .load = undefined, .unload = undefined, .forward = undefined, .cache = undefined, .weights = undefined };
}

fn markerClaims(_: common.Str, config_json: common.Str, _: *const common.PackDir) callconv(.c) common.Claim {
    return if (std.mem.indexOf(u8, config_json.slice(), "\"toy_marker\"") != null) .by_marker else .no;
}

test "outside plugins: a built-in wins a tie, a higher rank or a replaced name hands the pack to the outside plugin" {
    const base = comptime fakeModel("base", &.{"base"});
    const toy = comptime fakeModel("toy", &.{"base"});
    const toy_marker: model_abi.ModelTable = comptime blk: {
        var t = fakeModel("toy_m", &.{});
        t.claims = &markerClaims;
        break :blk t;
    };
    const fixed_base = comptime fakeModel("base", &.{"base"});
    const e = struct {
        fn of(comptime name: []const u8, t: *const model_abi.ModelTable) sdk_plugin.Entry {
            return .{ .plugin = .{ .name = name, .kind = .model }, .model = t };
        }
    };
    const builtins = [_]sdk_plugin.Entry{e.of("base", &base)};
    const mem: fixture.MemDir = .{ .files = &.{} };
    const dir = mem.packDir();
    const cfg = "{\"model_type\":\"base\"}";
    try testing.expectEqual(@as(?usize, null), outsideWinner(&builtins, &.{e.of("toy", &toy)}, "base", cfg, &dir));
    try testing.expectEqual(@as(?usize, 0), outsideWinner(&builtins, &.{e.of("toy_m", &toy_marker)}, "base", "{\"model_type\":\"base\",\"toy_marker\":1}", &dir));
    try testing.expectEqual(@as(?usize, null), outsideWinner(&builtins, &.{e.of("toy_m", &toy_marker)}, "base", cfg, &dir));
    try testing.expectEqual(@as(?usize, 0), outsideWinner(&builtins, &.{e.of("base", &fixed_base)}, "base", cfg, &dir));
    // Two outside plugins tied at the top: neither.
    try testing.expectEqual(@as(?usize, null), outsideWinner(&.{}, &.{ e.of("toy", &toy), e.of("toy2", &toy) }, "base", cfg, &dir));
    try testing.expectEqual(@as(?usize, 1), outsideWinner(&.{}, &.{ e.of("toy", &toy), e.of("toy_m", &toy_marker) }, "base", "{\"toy_marker\":1}", &dir));
}

test "outside plugins: settings default off; a broken file is off; disabled names are skipped" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(testing.io, &buf)];
    const path = try std.fs.path.join(a, &.{ root, "plugins.json" });
    try testing.expect(!readSettings(testing.io, a, path).enabled);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "plugins.json", .data = "{\"enabled\":" });
    try testing.expect(!readSettings(testing.io, a, path).enabled);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "plugins.json", .data = "{\"enabled\":true,\"disabled\":[\"toy\"],\"later\":1}" });
    const s = readSettings(testing.io, a, path);
    try testing.expect(s.enabled and s.isDisabled("toy") and !s.isDisabled("toy2"));
}

test "outside plugins: a dylib without a receipt, with a changed hash, or not a plugin is refused by name" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(testing.io, &buf)];
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "a.dylib", .data = "not a dylib" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "b.dylib", .data = "not a dylib" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "b.json", .data = "{\"name\":\"b\",\"kind\":\"model\",\"sha256\":\"00\"}" });
    const h = sha256Hex("not a dylib");
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "c.dylib", .data = "not a dylib" });
    var receipt_buf: [256]u8 = undefined;
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "c.json", .data = try std.fmt.bufPrint(&receipt_buf, "{{\"name\":\"c\",\"kind\":\"model\",\"sha256\":\"{s}\"}}", .{h}) });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "d.dylib", .data = "off" });
    var set = try loadDir(testing.io, testing.allocator, root, .{ .enabled = true, .disabled = &.{"d"} }, "pin");
    defer set.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), set.plugins.items.len);
    try testing.expectEqual(@as(usize, 3), set.refused.items.len);
    try testing.expectEqualStrings("has no install receipt: install it with `mlx-serve plugin install`", set.refused.items[0].why);
    try testing.expectEqualStrings("changed since it was installed (hash differs from its receipt)", set.refused.items[1].why);
    try testing.expect(std.mem.startsWith(u8, set.refused.items[2].why, "does not load: "));
}

test "plugin signature: codesign's report reads as its first authority, ad-hoc or unsigned" {
    try testing.expectEqualStrings("Developer ID Application: Jane Doe (ABCDE12345)", signatureFromReport("Executable=/x\nIdentifier=x\nAuthority=Developer ID Application: Jane Doe (ABCDE12345)\nAuthority=Developer ID Certification Authority\nTeamIdentifier=ABCDE12345\n"));
    try testing.expectEqualStrings("ad-hoc", signatureFromReport("Executable=/x\nSignature=adhoc\nTeamIdentifier=not set\n"));
    try testing.expectEqualStrings("unsigned", signatureFromReport("/x: code object is not signed at all\n"));
}
