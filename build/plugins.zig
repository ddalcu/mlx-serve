//! Configure-time plugin registration: scans `src/plugins/*/plugin.zig`, checks that each plugin's `config.zig` reaches
//! no MLX (its import closure), and writes `src/plugins/registry.zig` (generated, git-ignored). Folders starting with
//! `_` are templates: compiled and tested, never registered.

const std = @import("std");

const plugins_dir = "src/plugins";
const configs_dir = "src/fixtures/model-configs";
const registry_path = "src/plugins/registry.zig";

/// What a `config.zig` may never reach: the MLX layer and everything built on it. Anything outside `src/` too.
const denied = [_][]const u8{ "src/mlx/", "src/engines/", "src/host/", "src/kernels/" };

/// Zig caches the configure phase: every input the scan reads is declared, so a new plugin folder, a changed
/// `config.zig` closure, a new fixture or a deleted registry re-runs it. Returns the registered plugins' names.
pub fn generate(b: *std.Build) []const []const u8 {
    const a = b.allocator;
    const io = b.graph.io;
    const root = b.root.root_dir.handle;
    b.dependOnDirectoryContents(b.path(plugins_dir));
    b.dependOnDirectoryContents(b.path(configs_dir));
    const folders = listDir(a, io, root, plugins_dir, .directory) catch |e| std.process.fatal("plugins: cannot list {s}: {t}", .{ plugins_dir, e });
    var names: std.ArrayList([]const u8) = .empty;
    for (folders) |f| {
        b.dependOnDirectoryContents(b.path(b.fmt("{s}/{s}", .{ plugins_dir, f })));
        // A folder with its own build.zig is an outside plugin built from the kit (build/kit.zig), never compiled in.
        if (root.access(io, b.fmt("{s}/{s}/build.zig", .{ plugins_dir, f }), .{})) |_| continue else |_| {}
        root.access(io, b.fmt("{s}/{s}/plugin.zig", .{ plugins_dir, f }), .{}) catch continue;
        const cfg = b.fmt("{s}/{s}/config.zig", .{ plugins_dir, f });
        root.access(io, cfg, .{}) catch std.process.fatal("plugin {s}: plugin.zig without the MLX-free config.zig", .{f});
        const fs_reader: FsReader = .{ .b = b, .dir = root, .io = io };
        if (firstViolation(a, fs_reader, cfg) catch |e| std.process.fatal("plugin {s}: reading its imports: {t}", .{ f, e })) |msg|
            std.process.fatal("plugin {s}: {s}", .{ f, msg });
        names.append(a, f) catch @panic("OOM");
    }
    const configs = listDir(a, io, root, configs_dir, .file) catch |e| std.process.fatal("plugins: cannot list {s}: {t}", .{ configs_dir, e });
    var jsons: std.ArrayList([]const u8) = .empty;
    for (configs) |c| if (std.mem.endsWith(u8, c, ".json")) jsons.append(a, c) catch @panic("OOM");
    const split = splitTemplates(a, names.items) catch @panic("OOM");
    const in_config = a.alloc(bool, split.plugins.len) catch @panic("OOM");
    for (split.plugins, in_config) |n, *f| {
        const text = root.readFileAlloc(io, b.fmt("{s}/{s}/config.zig", .{ plugins_dir, n }), a, .limited(1 << 20)) catch "";
        f.* = std.mem.indexOf(u8, text, "\npub const plugin = ") != null;
    }
    const src = render(a, split.plugins, in_config, split.templates, jsons.items) catch @panic("OOM");
    writeIfChanged(io, root, registry_path, src) catch |e| std.process.fatal("plugins: cannot write {s}: {t}", .{ registry_path, e });
    b.dependOnFileContents(b.path(registry_path));
    return split.plugins;
}

/// `-Dunits`: one root per plugin group, `src/unit_<g>.zig` (generated, git-ignored), importing only its own plugins so
/// an edit to one plugin recompiles its group alone. Returns the roots' paths.
pub fn writeUnitRoots(b: *std.Build, groups: []const []const []const u8) []const []const u8 {
    const a = b.allocator;
    const paths = a.alloc([]const u8, groups.len) catch @panic("OOM");
    for (groups, paths, 0..) |names, *path, g| {
        var out: std.ArrayList(u8) = .empty;
        out.appendSlice(a,
            \\//! GENERATED at configure time by build/plugins.zig (`-Dunits`): one plugin group as its own compile unit, exporting
            \\//! its plugins' tables for the host to take by symbol. Never edit, never commit.
            \\
            \\comptime {
            \\    @import("plugin_host_check.zig").reexports(@This());
            \\    @import("sdk/plugin.zig").exportAll(&.{
            \\
        ) catch @panic("OOM");
        for (names) |n| out.print(a, "        @import(\"plugins/{s}/plugin.zig\"),\n", .{n}) catch @panic("OOM");
        out.appendSlice(a,
            \\    });
            \\}
            \\
            \\/// Marks a plugin unit: `Shared` state resolves to the host's copy (sdk/util/shared.zig).
            \\pub const mlx_serve_unit = true;
            \\
            \\// What the absorbed modules read from their host root (`mlx_host`), as `main.zig` declares them.
            \\pub const env = @import("env.zig");
            \\pub const mlx = @import("mlx/ffi.zig");
            \\pub const naxAvailable = @import("mlx/device.zig").naxAvailable;
            \\pub const io_util = @import("sdk/util/io_util.zig");
            \\pub const shared = @import("sdk/util/shared.zig");
            \\pub const mtp_acceptance = @import("sdk/spec/mtp_acceptance.zig");
            \\pub const exl3 = @import("sdk/quant/exl3/format.zig");
            \\pub const dsa_nax = @import("plugins/glm5_next/dsa_nax.zig");
            \\pub const fp8_block128_gemv = @import("sdk/quant/fp8_block128_gemv.zig");
            \\pub const log = @import("sdk/util/log.zig");
            \\pub const plugin_profile: bool = @import("build_options").plugin_profile;
            \\
        ) catch @panic("OOM");
        path.* = b.fmt("src/unit_{d}.zig", .{g});
        writeIfChanged(b.graph.io, b.root.root_dir.handle, path.*, out.items) catch |e| std.process.fatal("units: cannot write {s}: {t}", .{ path.*, e });
    }
    return paths;
}

/// The build's own tests of this file (the scan, the closure check, the rendering), on the host.
pub fn addTests(b: *std.Build, step: *std.Build.Step) void {
    const t = b.addTest(.{ .name = "build-plugins-test", .root_module = b.createModule(.{
        .root_source_file = b.path("build/plugins.zig"),
        .target = b.graph.host,
    }) });
    step.dependOn(&b.addRunArtifact(t).step);
}

fn listDir(a: std.mem.Allocator, io: std.Io, root: std.Io.Dir, sub: []const u8, kind: std.Io.File.Kind) ![]const []const u8 {
    var dir = try root.openDir(io, sub, .{ .iterate = true });
    defer dir.close(io);
    var out: std.ArrayList([]const u8) = .empty;
    var it = dir.iterate();
    while (try it.next(io)) |e| if (e.kind == kind) try out.append(a, try a.dupe(u8, e.name));
    std.mem.sort([]const u8, out.items, {}, lessThan);
    return out.items;
}

fn lessThan(_: void, x: []const u8, y: []const u8) bool {
    return std.mem.lessThan(u8, x, y);
}

const Split = struct { plugins: []const []const u8, templates: []const []const u8 };

fn splitTemplates(a: std.mem.Allocator, names: []const []const u8) !Split {
    var p: std.ArrayList([]const u8) = .empty;
    var t: std.ArrayList([]const u8) = .empty;
    for (names) |n| try (if (n.len > 0 and n[0] == '_') &t else &p).append(a, n);
    return .{ .plugins = p.items, .templates = t.items };
}

/// The `@import` strings of a Zig source, line comments excluded.
fn importsOf(a: std.mem.Allocator, src: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var lines = std.mem.splitScalar(u8, src, '\n');
    while (lines.next()) |full| {
        var line = full;
        if (std.mem.indexOf(u8, line, "//")) |c| line = line[0..c];
        const key = "@import(\"";
        var at: usize = 0;
        while (std.mem.indexOfPos(u8, line, at, key)) |i| {
            const start = i + key.len;
            const end = std.mem.indexOfScalarPos(u8, line, start, '"') orelse break;
            try out.append(a, line[start..end]);
            at = end;
        }
    }
    return out.items;
}

/// Reads a source for the closure check and declares it a configure input.
pub const FsReader = struct {
    b: *std.Build,
    dir: std.Io.Dir,
    io: std.Io,

    fn read(self: FsReader, path: []const u8) ![]const u8 {
        self.b.dependOnFileContents(self.b.path(path));
        return self.dir.readFileAlloc(self.io, path, self.b.allocator, .limited(16 << 20));
    }
};

/// The first import in `start`'s closure that leaves the MLX-free set, as the chain that reaches it; null = clean.
fn firstViolation(a: std.mem.Allocator, reader: anytype, start: []const u8) !?[]const u8 {
    return switch (try closure(a, reader, start)) {
        .files => null,
        .violation => |v| v,
    };
}

pub const Closure = union(enum) { files: []const []const u8, violation: []const u8 };

/// `start`'s import closure (itself included, sorted), or the first import that leaves the MLX-free set.
pub fn closure(a: std.mem.Allocator, reader: anytype, start: []const u8) !Closure {
    var parent: std.StringHashMapUnmanaged([]const u8) = .empty;
    var queue: std.ArrayList([]const u8) = .empty;
    try parent.put(a, start, "");
    try queue.append(a, start);
    while (queue.pop()) |path| {
        for (try importsOf(a, try reader.read(path))) |imp| {
            if (std.mem.eql(u8, imp, "std") or std.mem.eql(u8, imp, "builtin")) continue;
            if (std.mem.endsWith(u8, imp, ".zon")) continue;
            if (!std.mem.endsWith(u8, imp, ".zig"))
                return .{ .violation = try std.fmt.allocPrint(a, "{s} imports the module \"{s}\" (a config.zig reaches std and src/ files only)", .{ try chain(a, &parent, path), imp }) };
            const next = try std.fs.path.resolveAllocPosix(a, &.{ std.fs.path.dirnamePosix(path) orelse ".", imp });
            if (deniedPath(next))
                return .{ .violation = try std.fmt.allocPrint(a, "{s} -> {s}: a config.zig must not reach MLX", .{ try chain(a, &parent, path), next }) };
            const gop = try parent.getOrPut(a, next);
            if (gop.found_existing) continue;
            gop.value_ptr.* = path;
            try queue.append(a, next);
        }
    }
    var files: std.ArrayList([]const u8) = .empty;
    var it = parent.keyIterator();
    while (it.next()) |k| try files.append(a, k.*);
    std.mem.sort([]const u8, files.items, {}, lessThan);
    return .{ .files = files.items };
}

fn deniedPath(path: []const u8) bool {
    if (!std.mem.startsWith(u8, path, "src/")) return true;
    for (denied) |d| if (std.mem.startsWith(u8, path, d)) return true;
    return false;
}

/// `start -> ... -> path`, from the parent links.
fn chain(a: std.mem.Allocator, parent: *const std.StringHashMapUnmanaged([]const u8), path: []const u8) ![]const u8 {
    var hops: std.ArrayList([]const u8) = .empty;
    var p = path;
    while (p.len > 0) : (p = parent.get(p).?) try hops.append(a, p);
    std.mem.reverse([]const u8, hops.items);
    return std.mem.join(a, " -> ", hops.items);
}

/// `in_config[i]`: plugin i declares its metadata (`plugin`) in its `config.zig`.
/// `units`: the `-Dunits` form, metadata only (each `config.zig`) and the extern tables the unit objects export, so the
/// host's closure holds no plugin's code and a plugin edit rebuilds its unit alone. Tests need the full form.
fn render(a: std.mem.Allocator, plugins: []const []const u8, in_config: []const bool, templates: []const []const u8, configs: []const []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a,
        \\//! GENERATED at configure time by build/plugins.zig from src/plugins/*/plugin.zig: never edit, never commit.
        \\//! Folders starting with `_` are templates: compiled and tested, never registered.
        \\
        \\const sdk = @import("../sdk/plugin.zig");
        \\const File = @import("../sdk/testing/fixture.zig").File;
        \\
        \\
    );
    inline for (.{ .{ "plugins", plugins }, .{ "templates", templates } }) |list| {
        try out.print(a, "pub const {s} = [_]type{{\n", .{list[0]});
        for (list[1]) |n| try out.print(a, "    @import(\"{s}/plugin.zig\"),\n", .{n});
        try out.print(a, "}};\npub const {s}_names = [_][]const u8{{\n", .{list[0]});
        for (list[1]) |n| try out.print(a, "    \"{s}\",\n", .{n});
        try out.appendSlice(a, "};\n");
    }
    try renderMetas(a, &out, plugins, in_config, configs);
    try out.appendSlice(a,
        \\
        \\/// `-Dunits`: every plugin is its own compile unit (`src/unit.zig`) exporting its table; the host links them by name.
        \\const units = @import("build_options").units and !@import("builtin").is_test;
        \\
        \\pub const entries = if (units) sdk.externEntries(&unit_metas) else sdk.entries(&plugins, &plugins_names);
        \\
        \\comptime {
        \\    _ = sdk.entries(&templates, &templates_names);
        \\    if (!units) sdk.exportAll(&plugins);
        \\}
        \\
        \\test "plugin registry: every template and plugin passes conformance; one plugin claims each fixture pack" {
        \\    try @import("../sdk/testing/conformance.zig").expectRegistry(@This());
        \\}
        \\
    );
    return out.items;
}

fn renderMetas(a: std.mem.Allocator, out: *std.ArrayList(u8), plugins: []const []const u8, in_config: []const bool, configs: []const []const u8) !void {
    try out.appendSlice(a, "\n/// Each plugin's metadata as `-Dunits` reads it: from its `config.zig` (no plugin code), else its `plugin.zig`.\nconst unit_metas = [_]sdk.Plugin{\n");
    for (plugins, in_config) |n, c| try out.print(a, "    @import(\"{s}/{s}.zig\").plugin,\n", .{ n, if (c) "config" else "plugin" });
    try out.appendSlice(a, "};\n");
    try out.appendSlice(a, "\n/// Every fixture config.json: the packs the claims test runs through every plugin.\npub const model_configs = [_]File{\n");
    for (configs) |c| try out.print(a, "    .{{ .name = \"{s}\", .bytes = @embedFile(\"../fixtures/model-configs/{s}\") }},\n", .{ c, c });
    try out.appendSlice(a, "};\n");
}

fn writeIfChanged(io: std.Io, root: std.Io.Dir, path: []const u8, bytes: []const u8) !void {
    var buf: [1 << 16]u8 = undefined;
    if (root.readFile(io, path, &buf)) |old| {
        if (std.mem.eql(u8, old, bytes)) return;
    } else |_| {}
    var tmp_buf: [256]u8 = undefined;
    var rnd: [8]u8 = undefined;
    io.random(&rnd);
    const tmp = try std.fmt.bufPrint(&tmp_buf, "{s}.{x}.tmp", .{ path, std.mem.readInt(u64, &rnd, .little) });
    try root.writeFile(io, .{ .sub_path = tmp, .data = bytes });
    try root.rename(tmp, root, path, io);
}

// ── tests ──

const testing = std.testing;

const MapReader = struct {
    files: []const [2][]const u8,
    fn read(self: MapReader, path: []const u8) ![]const u8 {
        for (self.files) |f| if (std.mem.eql(u8, f[0], path)) return f[1];
        return error.FileNotFound;
    }
};

test "build plugins: imports are read from code, never from line comments" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const got = try importsOf(arena.allocator(),
        \\const std = @import("std");
        \\// const mlx = @import("../../mlx/ffi.zig");
        \\const a = @import("a.zig"); const b = @import("b.zig"); // @import("c.zig")
    );
    try testing.expectEqual(@as(usize, 3), got.len);
    try testing.expectEqualStrings("b.zig", got[2]);
}

test "build plugins: a config.zig whose closure reaches MLX or a module is refused with the chain" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const clean: MapReader = .{ .files = &.{
        .{ "src/plugins/fx/config.zig", "const abi = @import(\"../../abi/plugin.zig\");\nconst sdk = @import(\"../../sdk/root.zig\");" },
        .{ "src/abi/plugin.zig", "const std = @import(\"std\");\nconst c = @import(\"common.zig\");" },
        .{ "src/abi/common.zig", "const std = @import(\"std\");" },
        .{ "src/sdk/root.zig", "pub const plugin = @import(\"plugin.zig\");" },
        .{ "src/sdk/plugin.zig", "const abi = @import(\"../abi/plugin.zig\");" },
    } };
    try testing.expectEqual(@as(?[]const u8, null), try firstViolation(a, clean, "src/plugins/fx/config.zig"));

    const deep: MapReader = .{ .files = &.{
        .{ "src/plugins/fx/config.zig", "const h = @import(\"helpers.zig\");" },
        .{ "src/plugins/fx/helpers.zig", "const b = @import(\"../../sdk/blocks/norm.zig\");" },
        .{ "src/sdk/blocks/norm.zig", "const mlx = @import(\"../../mlx/ffi.zig\");" },
    } };
    try testing.expectEqualStrings(
        "src/plugins/fx/config.zig -> src/plugins/fx/helpers.zig -> src/sdk/blocks/norm.zig -> src/mlx/ffi.zig: a config.zig must not reach MLX",
        (try firstViolation(a, deep, "src/plugins/fx/config.zig")).?,
    );
    const module: MapReader = .{ .files = &.{.{ "src/plugins/fx/config.zig", "const m = @import(\"mlx_stream\");" }} };
    try testing.expect(std.mem.indexOf(u8, (try firstViolation(a, module, "src/plugins/fx/config.zig")).?, "the module \"mlx_stream\"") != null);
    const outside: MapReader = .{ .files = &.{.{ "src/plugins/fx/config.zig", "const m = @import(\"../../../lib/x.zig\");" }} };
    try testing.expect((try firstViolation(a, outside, "src/plugins/fx/config.zig")) != null);
}

test "build plugins: the closure lists every file an MLX-free root reaches" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const r: MapReader = .{ .files = &.{
        .{ "src/kit.zig", "const a = @import(\"abi/plugin.zig\");\nconst s = @import(\"sdk/root.zig\");" },
        .{ "src/abi/plugin.zig", "const c = @import(\"common.zig\");" },
        .{ "src/abi/common.zig", "const std = @import(\"std\");" },
        .{ "src/sdk/root.zig", "const p = @import(\"../abi/plugin.zig\");" },
    } };
    const got = (try closure(arena.allocator(), r, "src/kit.zig")).files;
    try testing.expectEqual(@as(usize, 4), got.len);
    try testing.expectEqualStrings("src/abi/common.zig", got[0]);
    try testing.expectEqualStrings("src/sdk/root.zig", got[3]);
}

test "build plugins: templates compile beside the registry but are never registered or exported" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const split = try splitTemplates(a, &.{ "_template", "flint", "_template_media", "laguna" });
    try testing.expectEqual(@as(usize, 2), split.plugins.len);
    try testing.expectEqualStrings("laguna", split.plugins[1]);
    const in_config = try a.alloc(bool, split.plugins.len);
    @memset(in_config, false);
    const src = try render(a, split.plugins, in_config, split.templates, &.{"laguna.json"});
    try testing.expect(std.mem.indexOf(u8, src, "pub const plugins = [_]type{\n    @import(\"flint/plugin.zig\"),\n    @import(\"laguna/plugin.zig\"),\n};") != null);
    try testing.expect(std.mem.indexOf(u8, src, "pub const templates = [_]type{\n    @import(\"_template/plugin.zig\"),\n    @import(\"_template_media/plugin.zig\"),\n};") != null);
    try testing.expect(std.mem.indexOf(u8, src, "@embedFile(\"../fixtures/model-configs/laguna.json\")") != null);
    try testing.expect(std.mem.indexOf(u8, src, "sdk.exportAll(&plugins);") != null);
    // An empty registry still renders valid arrays.
    const none = try render(a, &.{}, &.{}, &.{}, &.{});
    try testing.expect(std.mem.indexOf(u8, none, "pub const plugins = [_]type{\n};") != null);
}

test "build plugins: the -Dunits metadata reads each plugin's config.zig where it declares one" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const src = try render(a, &.{ "flint", "laguna" }, &.{ true, false }, &.{"_template"}, &.{"laguna.json"});
    try testing.expect(std.mem.indexOf(u8, src, "@import(\"flint/config.zig\").plugin,") != null);
    try testing.expect(std.mem.indexOf(u8, src, "@import(\"laguna/plugin.zig\").plugin,") != null);
    try testing.expect(std.mem.indexOf(u8, src, "sdk.externEntries(&unit_metas)") != null);
}
