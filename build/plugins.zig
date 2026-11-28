//! Configure-time plugin registration: scans `src/plugins/*/plugin.zig`, checks that each plugin's `config.zig` reaches
//! no MLX (its import closure), and writes `src/plugins/registry.zig` and `registry_full.zig` (generated, git-ignored). Folders starting with
//! `_` are templates: compiled and tested, never registered.

const std = @import("std");

const plugins_dir = "src/plugins";
const configs_dir = "src/fixtures/model-configs";
const registry_path = "src/plugins/registry.zig";
const registry_full_path = "src/plugins/registry_full.zig";
const registry_test_core_path = "src/plugins/registry_test_core.zig";
const registry_test_media_path = "src/plugins/registry_test_media.zig";
const registry_test_stubs_path = "src/plugins/registry_test_stubs.zig";

/// What a `config.zig` may never reach: the MLX layer and everything built on it. Anything outside `src/` too.
const denied = [_][]const u8{ "src/mlx/", "src/engines/", "src/host/", "src/kernels/" };

/// Zig caches the configure phase: every input the scan reads is declared, so a new plugin folder, a changed
/// `config.zig` closure, a new fixture or a deleted registry re-runs it. `units`: the main graph builds `-Dunits`.
/// Returns the registered plugins' names.
pub fn generate(b: *std.Build, units: bool) []const []const u8 {
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
    const is_media = a.alloc(bool, split.plugins.len) catch @panic("OOM");
    for (split.plugins, in_config, is_media) |n, *f, *m| {
        const text = root.readFileAlloc(io, b.fmt("{s}/{s}/config.zig", .{ plugins_dir, n }), a, .limited(1 << 20)) catch "";
        f.* = std.mem.indexOf(u8, text, "\npub const plugin = ") != null;
        m.* = std.mem.indexOf(u8, text, ".kind = .media") != null;
    }
    const full = renderFull(a, split.plugins, split.templates, jsons.items) catch @panic("OOM");
    const facade = renderFacade(a, units, split.plugins, in_config, jsons.items) catch @panic("OOM");
    const test_core = renderTestCore(a, split.plugins, is_media, split.templates) catch @panic("OOM");
    const test_media = renderTestMedia(a, split.plugins, is_media) catch @panic("OOM");
    const test_stubs = renderTestStubs(a, split.plugins, is_media) catch @panic("OOM");
    for ([_][2][]const u8{
        .{ registry_full_path, full },
        .{ registry_path, facade },
        .{ registry_test_core_path, test_core },
        .{ registry_test_media_path, test_media },
        .{ registry_test_stubs_path, test_stubs },
    }) |w| {
        writeIfChanged(io, root, w[0], w[1]) catch |e| std.process.fatal("plugins: cannot write {s}: {t}", .{ w[0], e });
        b.dependOnFileContents(b.path(w[0]));
    }
    return split.plugins;
}

/// Pure-SDK media plugins compiled in `test-media` instead of `test-core` (every `.media` plugin except `clef`,
/// whose forward reaches `engines/mlx`).
pub fn pureMediaNames(b: *std.Build, plugin_names: []const []const u8) []const []const u8 {
    const a = b.allocator;
    const io = b.graph.io;
    const root = b.root.root_dir.handle;
    var out: std.ArrayList([]const u8) = .empty;
    for (plugin_names) |n| {
        if (std.mem.eql(u8, n, "clef")) continue;
        const text = root.readFileAlloc(io, b.fmt("{s}/{s}/config.zig", .{ plugins_dir, n }), a, .limited(1 << 20)) catch "";
        if (std.mem.indexOf(u8, text, ".kind = .media") != null) out.append(a, n) catch @panic("OOM");
    }
    return out.items;
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
            \\pub const io_util = @import("sdk/util/io_util.zig");
            \\pub const log = @import("sdk/util/log.zig");
            \\
        ) catch @panic("OOM");
        path.* = b.fmt("src/unit_{d}.zig", .{g});
        writeIfChanged(b.graph.io, b.root.root_dir.handle, path.*, out.items) catch |e| std.process.fatal("units: cannot write {s}: {t}", .{ path.*, e });
    }
    return paths;
}

/// `-Dunits` plugin groups packed so plugins that compile the same SDK code share an object, each at most as costly as
/// the largest round-robin group for 8, 4 and 3 groups: same longest compile, about half the code compiled in total.
/// Regenerate with `scripts/unit_groups.py` when plugins come or go; a test names any plugin the table misses.
const packed_groups = [_][]const []const []const u8{
    &.{
        &.{ "clef", "d1", "hy_v3", "kev", "laguna", "lfm2", "llama", "moss_image", "nemotron_h", "nemotron_transcribe", "prism_hadamard_qwen35", "qwen3", "qwen3_5", "qwen3_asr", "qwen3_next", "qwen4_exp" },
        &.{ "bailing_hybrid", "bert", "embedding_gemma", "embedding_gemma2", "flint", "gemma3", "gemma4", "glm5_next", "gpt_oss", "inkling", "inkling2", "k2_horizon", "kolibri1", "lfm3", "ltx", "mimo_v2", "minimax_h3", "minimax_h3_image", "muse_glimmer", "spark2_5" },
        &.{ "acestep", "deepseek_v41", "flux2", "hunyuan3d", "idol", "krea", "laya", "mage_flow", "ming_tts", "music3", "nemotron_diarization", "pixal3d", "qwen3_tts", "qwen_image", "stable_audio3", "trellis2" },
        &.{ "deepseek_v4", "kokoro", "yue2" },
    },
    &.{
        &.{ "bailing_hybrid", "clef", "d1", "flint", "gemma3", "gemma4", "glm5_next", "gpt_oss", "hy_v3", "k2_horizon", "kev", "kolibri1", "laguna", "lfm2", "lfm3", "llama", "ltx", "mimo_v2", "moss_image", "muse_glimmer", "nemotron_h", "nemotron_transcribe", "prism_hadamard_qwen35", "qwen3", "qwen3_5", "qwen3_asr", "qwen3_next", "qwen4_exp", "spark2_5" },
        &.{ "acestep", "bert", "deepseek_v41", "embedding_gemma", "embedding_gemma2", "flux2", "inkling", "inkling2", "krea", "mage_flow", "ming_tts", "minimax_h3", "minimax_h3_image", "music3", "qwen3_tts", "qwen_image" },
        &.{ "deepseek_v4", "hunyuan3d", "idol", "kokoro", "laya", "nemotron_diarization", "pixal3d", "stable_audio3", "trellis2", "yue2" },
    },
    &.{
        &.{ "bailing_hybrid", "bert", "clef", "d1", "embedding_gemma", "embedding_gemma2", "flint", "gemma3", "gemma4", "glm5_next", "gpt_oss", "hy_v3", "inkling", "inkling2", "k2_horizon", "kev", "kolibri1", "laguna", "lfm2", "lfm3", "llama", "ltx", "mimo_v2", "minimax_h3", "moss_image", "muse_glimmer", "nemotron_h", "nemotron_transcribe", "prism_hadamard_qwen35", "qwen3", "qwen3_5", "qwen3_asr", "qwen3_next", "qwen4_exp", "spark2_5" },
        &.{ "acestep", "deepseek_v4", "deepseek_v41", "flux2", "hunyuan3d", "idol", "kokoro", "krea", "laya", "mage_flow", "ming_tts", "minimax_h3_image", "music3", "nemotron_diarization", "pixal3d", "qwen3_tts", "qwen_image", "stable_audio3", "trellis2", "yue2" },
    },
};

/// `names` in at most `max_groups` plugin groups: one plugin each when that many fit (what `scripts/unit_groups.py`
/// measures), else the largest packed table that fits, a plugin it misses dealt round-robin after the packed ones, or
/// one group when only one fits.
pub fn assignGroups(a: std.mem.Allocator, names: []const []const u8, max_groups: usize) ![]const []const []const u8 {
    if (max_groups >= names.len) {
        const solo = try a.alloc([]const []const u8, names.len);
        for (solo, names) |*s, *n| s.* = n[0..1];
        return solo;
    }
    const table: []const []const []const u8 = for (packed_groups) |t| {
        if (t.len <= max_groups) break t;
    } else &.{names};
    const groups = try a.alloc(std.ArrayList([]const u8), table.len);
    for (groups) |*g| g.* = .empty;
    var spill: usize = 0;
    for (names) |n| {
        const gi = for (table, 0..) |t, i| {
            if (contains(t, n)) break i;
        } else blk: {
            spill += 1;
            break :blk (spill - 1) % table.len;
        };
        try groups[gi].append(a, n);
    }
    var out: std.ArrayList([]const []const u8) = .empty;
    for (groups) |g| if (g.items.len > 0) try out.append(a, g.items);
    return out.items;
}

/// The `-Dunits` plugin groups for a build at `optimize`: packed for Debug, where the tables were measured; dealt
/// round-robin otherwise, where the packed media group compiles longer at -O3 than the engine object.
pub fn unitGroups(a: std.mem.Allocator, names: []const []const u8, max_groups: usize, optimize: std.builtin.OptimizeMode) ![]const []const []const u8 {
    if (optimize == .debug) return assignGroups(a, names, max_groups);
    const n = @min(names.len, @max(max_groups, 1));
    const groups = try a.alloc([]const []const u8, n);
    for (groups, 0..) |*g, gi| {
        var list: std.ArrayList([]const u8) = .empty;
        var i = gi;
        while (i < names.len) : (i += n) try list.append(a, names[i]);
        g.* = list.items;
    }
    return groups;
}

fn contains(list: []const []const u8, name: []const u8) bool {
    for (list) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
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
pub fn importsOf(a: std.mem.Allocator, src: []const u8) ![]const []const u8 {
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

    pub fn read(self: FsReader, path: []const u8) ![]const u8 {
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

const header =
    \\//! GENERATED at configure time by build/plugins.zig from src/plugins/*/plugin.zig: never edit, never commit.
    \\//! Folders starting with `_` are templates: compiled and tested, never registered.
    \\
    \\const sdk = @import("../sdk/plugin.zig");
    \\const File = @import("../sdk/testing/fixture.zig").File;
    \\
    \\
;

/// `registry_full.zig`: every plugin's and template's code, compiled into one unit. The tests, the iOS library and every
/// build without `-Dunits` serve it.
fn renderFull(a: std.mem.Allocator, plugins: []const []const u8, templates: []const []const u8, configs: []const []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, header);
    inline for (.{ .{ "plugins", plugins }, .{ "templates", templates } }) |list| {
        try out.print(a, "pub const {s} = [_]type{{\n", .{list[0]});
        for (list[1]) |n| try out.print(a, "    @import(\"{s}/plugin.zig\"),\n", .{n});
        try out.print(a, "}};\npub const {s}_names = [_][]const u8{{\n", .{list[0]});
        for (list[1]) |n| try out.print(a, "    \"{s}\",\n", .{n});
        try out.appendSlice(a, "};\n");
    }
    try renderConfigs(a, &out, configs);
    try out.appendSlice(a,
        \\
        \\pub const entries = sdk.entries(&plugins, &plugins_names);
        \\
        \\comptime {
        \\    _ = sdk.entries(&templates, &templates_names);
        \\    sdk.exportAll(&plugins);
        \\}
        \\
        \\test "plugin registry: every template and plugin passes conformance; one plugin claims each fixture pack" {
        \\    try @import("../sdk/testing/conformance.zig").expectRegistry(@This());
        \\}
        \\
    );
    return out.items;
}

/// `registry.zig`, what the host and the engine import. Without `-Dunits` it is `registry_full.zig`. Under `-Dunits` it
/// names no `plugin.zig` (Zig keys an object on every file its sources import, so the host, engine and modules objects
/// would rebuild on any plugin edit): each `config.zig`'s metadata and the extern tables the unit objects export. The
/// tests and the iOS library import `registry_full.zig`, whose `exportAll` defines the same tables in their binary.
/// `in_config[i]`: plugin i declares its metadata (`plugin`) in its `config.zig`.
fn renderFacade(a: std.mem.Allocator, units: bool, plugins: []const []const u8, in_config: []const bool, configs: []const []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    if (!units) {
        try out.appendSlice(a,
            \\//! GENERATED at configure time by build/plugins.zig: never edit, never commit. Built without `-Dunits`: the
            \\//! full registry.
            \\
            \\const full = @import("registry_full.zig");
            \\pub const entries = full.entries;
            \\pub const model_configs = full.model_configs;
            \\
            \\test {
            \\    _ = full;
            \\}
            \\
        );
        return out.items;
    }
    try out.appendSlice(a, header);
    try out.appendSlice(a, "/// Each plugin's metadata as `-Dunits` reads it: from its `config.zig` (no plugin code), else its `plugin.zig`.\nconst unit_metas = [_]sdk.Plugin{\n");
    for (plugins, in_config) |n, c| try out.print(a, "    @import(\"{s}/{s}.zig\").plugin,\n", .{ n, if (c) "config" else "plugin" });
    try out.appendSlice(a, "};\n");
    try renderConfigs(a, &out, configs);
    try out.appendSlice(a,
        \\
        \\/// The tables the unit objects export, or `registry_full.zig` exports where a root imports it (tests, iOS).
        \\pub const entries = sdk.externEntries(&unit_metas);
        \\
    );
    return out.items;
}

fn renderConfigs(a: std.mem.Allocator, out: *std.ArrayList(u8), configs: []const []const u8) !void {
    try out.appendSlice(a, "\n/// Every fixture config.json: the packs the claims test runs through every plugin.\npub const model_configs = [_]File{\n");
    for (configs) |c| try out.print(a, "    .{{ .name = \"{s}\", .bytes = @embedFile(\"../fixtures/model-configs/{s}\") }},\n", .{ c, c });
    try out.appendSlice(a, "};\n");
}

fn isPureMedia(name: []const u8, is_media: bool) bool {
    return is_media and !std.mem.eql(u8, name, "clef");
}

/// `registry_test_core.zig`: model plugins + clef + templates compiled in, pure-SDK media plugins exported as
/// config-only tables so `test-core` links `registry.entries` without compiling media plugin bodies.
fn renderTestCore(a: std.mem.Allocator, plugins: []const []const u8, is_media: []const bool, templates: []const []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, header);
    try out.appendSlice(a, "pub const plugins = [_]type{\n");
    for (plugins, is_media) |n, m| if (!isPureMedia(n, m)) try out.print(a, "    @import(\"{s}/plugin.zig\"),\n", .{n});
    try out.appendSlice(a, "};\npub const templates = [_]type{\n");
    for (templates) |n| try out.print(a, "    @import(\"{s}/plugin.zig\"),\n", .{n});
    try out.appendSlice(a, "};\npub const templates_names = [_][]const u8{\n");
    for (templates) |n| try out.print(a, "    \"{s}\",\n", .{n});
    try out.appendSlice(a, "};\nconst media_stubs = [_]type{\n");
    for (plugins, is_media) |n, m| if (isPureMedia(n, m)) try out.print(a, "    @import(\"{s}/config.zig\"),\n", .{n});
    try out.appendSlice(a,
        \\};
        \\pub const claimant_entries = @import("registry.zig").entries;
        \\pub const model_configs = @import("registry.zig").model_configs;
        \\
        \\comptime {
        \\    _ = sdk.entries(&templates, &templates_names);
        \\    sdk.exportAll(&plugins);
        \\    sdk.exportMediaConfigStubs(&media_stubs);
        \\}
        \\
        \\test "plugin registry: every template and model plugin passes conformance; one plugin claims each fixture pack" {
        \\    try @import("../sdk/testing/conformance.zig").expectRegistrySlice(@This());
        \\}
        \\
    );
    return out.items;
}

/// `registry_test_media.zig`: pure-SDK media plugins compiled in, model plugins + clef exported as config-only tables
/// so `test-media` links `registry.entries` without compiling LLM plugin bodies.
fn renderTestMedia(a: std.mem.Allocator, plugins: []const []const u8, is_media: []const bool) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, header);
    try out.appendSlice(a, "pub const plugins = [_]type{\n");
    for (plugins, is_media) |n, m| if (isPureMedia(n, m)) try out.print(a, "    @import(\"{s}/plugin.zig\"),\n", .{n});
    try out.appendSlice(a, "};\nconst media_stubs = [_]type{\n");
    for (plugins, is_media) |n, m| if (m and !isPureMedia(n, m)) try out.print(a, "    @import(\"{s}/config.zig\"),\n", .{n});
    try out.appendSlice(a, "};\nconst model_stubs = [_]type{\n");
    for (plugins, is_media) |n, m| if (!m) try out.print(a, "    @import(\"{s}/config.zig\"),\n", .{n});
    try out.appendSlice(a,
        \\};
        \\
        \\comptime {
        \\    sdk.exportAll(&plugins);
        \\    sdk.exportMediaConfigStubs(&media_stubs);
        \\    sdk.exportModelConfigStubs(&model_stubs);
        \\}
        \\
        \\test "plugin registry: every media plugin passes conformance" {
        \\    try @import("../sdk/testing/conformance.zig").expectRegistrySlice(@This());
        \\}
        \\
    );
    return out.items;
}

/// `registry_test_stubs.zig`: every plugin exported as a config-only table so `test-contracts` resolves
/// `registry.entries` without compiling any plugin body.
fn renderTestStubs(a: std.mem.Allocator, plugins: []const []const u8, is_media: []const bool) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, header);
    try out.appendSlice(a, "const media_stubs = [_]type{\n");
    for (plugins, is_media) |n, m| if (m) try out.print(a, "    @import(\"{s}/config.zig\"),\n", .{n});
    try out.appendSlice(a, "};\nconst model_stubs = [_]type{\n");
    for (plugins, is_media) |n, m| if (!m) try out.print(a, "    @import(\"{s}/config.zig\"),\n", .{n});
    try out.appendSlice(a,
        \\};
        \\
        \\comptime {
        \\    sdk.exportMediaConfigStubs(&media_stubs);
        \\    sdk.exportModelConfigStubs(&model_stubs);
        \\}
        \\
    );
    return out.items;
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

/// A source tree in memory (path, text), for the closure tests here and in `build/kit.zig`.
pub const MapReader = struct {
    files: []const [2][]const u8,
    pub fn read(self: MapReader, path: []const u8) ![]const u8 {
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
    const src = try renderFull(a, split.plugins, split.templates, &.{"laguna.json"});
    try testing.expect(std.mem.indexOf(u8, src, "pub const plugins = [_]type{\n    @import(\"flint/plugin.zig\"),\n    @import(\"laguna/plugin.zig\"),\n};") != null);
    try testing.expect(std.mem.indexOf(u8, src, "pub const templates = [_]type{\n    @import(\"_template/plugin.zig\"),\n    @import(\"_template_media/plugin.zig\"),\n};") != null);
    try testing.expect(std.mem.indexOf(u8, src, "@embedFile(\"../fixtures/model-configs/laguna.json\")") != null);
    try testing.expect(std.mem.indexOf(u8, src, "sdk.exportAll(&plugins);") != null);
    // An empty registry still renders valid arrays.
    const none = try renderFull(a, &.{}, &.{}, &.{});
    try testing.expect(std.mem.indexOf(u8, none, "pub const plugins = [_]type{\n};") != null);
}

test "build plugins: the -Dunits registry names no plugin.zig a config.zig can stand in for" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const src = try renderFacade(a, true, &.{ "flint", "laguna" }, &.{ true, false }, &.{"laguna.json"});
    try testing.expect(std.mem.indexOf(u8, src, "@import(\"flint/config.zig\").plugin,") != null);
    try testing.expect(std.mem.indexOf(u8, src, "@import(\"laguna/plugin.zig\").plugin,") != null);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, src, "/plugin.zig\").plugin"));
    try testing.expect(std.mem.indexOf(u8, src, "[_]type") == null);
    try testing.expect(std.mem.indexOf(u8, src, "pub const entries = sdk.externEntries(&unit_metas);") != null);
    try testing.expect(std.mem.indexOf(u8, src, "@embedFile(\"../fixtures/model-configs/laguna.json\")") != null);
}

test "build plugins: without -Dunits the registry is the full one" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const src = try renderFacade(arena.allocator(), false, &.{"flint"}, &.{true}, &.{});
    try testing.expect(std.mem.indexOf(u8, src, "const full = @import(\"registry_full.zig\");") != null);
    try testing.expect(std.mem.indexOf(u8, src, "/plugin.zig\")") == null);
}

test "build plugins: every packed table holds every registered plugin once (else rerun scripts/unit_groups.py)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = testing.io;
    const root = std.Io.Dir.cwd();
    var registered: std.ArrayList([]const u8) = .empty;
    for (try listDir(a, io, root, plugins_dir, .directory)) |f| {
        if (f[0] == '_') continue;
        if (root.access(io, try std.fmt.allocPrint(a, "{s}/{s}/build.zig", .{ plugins_dir, f }), .{})) |_| continue else |_| {}
        root.access(io, try std.fmt.allocPrint(a, "{s}/{s}/plugin.zig", .{ plugins_dir, f }), .{}) catch continue;
        try registered.append(a, f);
    }
    try testing.expect(registered.items.len > 0);
    for (packed_groups) |table| {
        var count: usize = 0;
        for (table) |g| count += g.len;
        for (registered.items) |n| {
            var seen: usize = 0;
            for (table) |g| seen += @intFromBool(contains(g, n));
            if (seen != 1) std.debug.print("plugin {s} is in {d} groups of a {d}-group packed table: rerun scripts/unit_groups.py\n", .{ n, seen, table.len });
            try testing.expectEqual(@as(usize, 1), seen);
        }
        try testing.expectEqual(registered.items.len, count);
    }
}

test "build plugins: assignGroups takes the largest packed table that fits and deals unknown plugins round-robin" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const names = [_][]const u8{ "kokoro", "qwen3", "llama", "krea", "deepseek_v4", "zz_new_a", "zz_new_b" };
    const four = try assignGroups(a, &names, 4);
    try testing.expectEqual(@as(usize, 4), four.len);
    var total: usize = 0;
    for (four) |g| total += g.len;
    try testing.expectEqual(names.len, total);
    try testing.expect(contains(four[0], "qwen3") and contains(four[0], "llama"));
    try testing.expect(contains(four[3], "kokoro") and contains(four[3], "deepseek_v4"));
    try testing.expectEqual(@as(usize, 2), (try assignGroups(a, &names, 2)).len);
    try testing.expectEqual(names.len, (try assignGroups(a, &names, names.len)).len);
    const one = try assignGroups(a, &names, 1);
    try testing.expectEqual(@as(usize, 1), one.len);
    try testing.expectEqual(names.len, one[0].len);
    try testing.expectEqual(@as(usize, 0), (try assignGroups(a, &.{}, 4)).len);
}

test "build plugins: only Debug builds take the packed groups; release builds deal round-robin" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const names = [_][]const u8{ "kokoro", "qwen3", "llama", "krea", "deepseek_v4" };
    const debug = try unitGroups(a, &names, 4, .debug);
    try testing.expect(contains(debug[0], "qwen3") and contains(debug[0], "llama"));
    const fast = try unitGroups(a, &names, 2, .ReleaseFast);
    try testing.expectEqual(@as(usize, 2), fast.len);
    try testing.expectEqualStrings("kokoro", fast[0][0]);
    try testing.expectEqualStrings("llama", fast[0][1]);
    try testing.expectEqualStrings("qwen3", fast[1][0]);
    try testing.expectEqual(@as(usize, 1), (try unitGroups(a, &names, 0, .ReleaseFast)).len);
}

test "build plugins: split test registries partition every plugin without overlap" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const names = [_][]const u8{ "clef", "flux2", "kokoro", "qwen3" };
    const media = [_]bool{ true, true, true, false };
    const core = try renderTestCore(a, &names, &media, &.{"_template"});
    const med = try renderTestMedia(a, &names, &media);
    // Core compiles model plugins + clef, stubs pure-SDK media configs.
    try testing.expect(std.mem.indexOf(u8, core, "@import(\"qwen3/plugin.zig\")") != null);
    try testing.expect(std.mem.indexOf(u8, core, "@import(\"clef/plugin.zig\")") != null);
    try testing.expect(std.mem.indexOf(u8, core, "@import(\"flux2/plugin.zig\")") == null);
    try testing.expect(std.mem.indexOf(u8, core, "@import(\"flux2/config.zig\")") != null);
    try testing.expect(std.mem.indexOf(u8, core, "@import(\"kokoro/config.zig\")") != null);
    // Media compiles pure-SDK media plugins, stubs model + clef configs.
    try testing.expect(std.mem.indexOf(u8, med, "@import(\"flux2/plugin.zig\")") != null);
    try testing.expect(std.mem.indexOf(u8, med, "@import(\"kokoro/plugin.zig\")") != null);
    try testing.expect(std.mem.indexOf(u8, med, "@import(\"qwen3/plugin.zig\")") == null);
    try testing.expect(std.mem.indexOf(u8, med, "@import(\"clef/plugin.zig\")") == null);
    try testing.expect(std.mem.indexOf(u8, med, "@import(\"qwen3/config.zig\")") != null);
    try testing.expect(std.mem.indexOf(u8, med, "@import(\"clef/config.zig\")") != null);
}
