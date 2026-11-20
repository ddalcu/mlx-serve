//! The plugin kit: `zig build kit` stages `build/kit/build.zig`, `src/kit.zig`'s import closure, mlx-c's headers,
//! the C header of the tables (`src/kit_header.zig`), the MLX pin, the pinned Zig's fetch script and the example
//! plugins as `zig-out/kit/mlx-serve-plugin-kit-<version>/` and its tarball. Under `zig build test`, each example is
//! built FROM that staged kit by its own `zig build` into a dylib the unit tests install and load
//! (`MLX_SERVE_KIT_PLUGINS`), and the C header compiles as C11 and C++17 with its layout asserts, so the kit and the
//! server cannot drift.

const std = @import("std");
const plugins = @import("plugins.zig");

/// The example plugins (`src/plugins/<name>/`, each with its own build.zig): `plugin new` copies them.
pub const examples = [_][]const u8{ "_example", "_example_media", "_example_quant", "_example_engine" };

/// The kit's own check plugins (`build/kit/<folder>/`, shipped as `checks/<folder>`): built from the staged kit for the
/// unit tests beside the examples, never copied by `plugin new`. `_outside_probe` opens its pack through the outside
/// quant table the host names (`host/plugins/kit_test.zig`).
const check_plugins = [_]struct { folder: []const u8, name: []const u8 }{.{ .folder = "outside_probe", .name = "_outside_probe" }};

pub fn add(b: *std.Build, version: []const u8, test_runs: []const *std.Build.Step.Run) void {
    const name = b.fmt("mlx-serve-plugin-kit-{s}", .{version});
    const gen = b.addRunArtifact(b.addExecutable(.{ .name = "kit_header", .root_module = b.createModule(.{
        .root_source_file = b.path("src/kit_header.zig"),
        .target = b.graph.host,
    }) }));
    const header = gen.addOutputFileArg("mlx_serve_plugin.h");
    const wf = b.addWriteFiles();
    stage(b, wf, name, version, header);
    const dir = wf.getDirectory();

    const step = b.step("kit", "Stage the plugin kit into zig-out/kit/ (a folder and its tarball)");
    step.dependOn(&b.addInstallDirectory(.{ .source_dir = dir, .install_dir = .prefix, .install_subdir = "kit" }).step);
    const tar = b.addSystemCommand(&.{ "tar", "-czf" });
    const tgz = tar.addOutputFileArg(b.fmt("{s}.tar.gz", .{name}));
    tar.addArg("-C");
    tar.addDirectoryArg(dir);
    tar.addArg(name);
    step.dependOn(&b.addInstallFileWithDir(tgz, .prefix, b.fmt("kit/{s}.tar.gz", .{name})).step);

    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0 .. b.root.root_dir.handle.realPath(b.graph.io, &root_buf) catch |e| std.process.fatal("kit: build root: {t}", .{e})];
    const out = b.fmt("{s}/.zig-cache/kit-plugins", .{root});
    var built: [examples.len + check_plugins.len]struct { file: []const u8, name: []const u8 } = undefined;
    for (examples, built[0..examples.len]) |ex, *x| x.* = .{ .file = b.fmt("{s}/examples/{s}/build.zig", .{ name, ex }), .name = ex };
    for (check_plugins, built[examples.len..]) |c, *x| x.* = .{ .file = b.fmt("{s}/checks/{s}/build.zig", .{ name, c.folder }), .name = c.name };
    for (built) |x| {
        const build = b.addSystemCommand(&.{ b.graph.zig_exe, "build", "--build-file" });
        build.addFileArg(dir.path(b, x.file));
        build.addArgs(&.{ "--prefix", b.fmt("{s}/{s}", .{ out, x.name }), "--cache-dir", b.fmt("{s}/.zig-cache/kit-nested", .{root}) });
        // The nested build is incremental itself; the tests rerun only when a dylib's bytes change.
        build.has_side_effects = true;
        for (test_runs) |r| {
            r.step.dependOn(&build.step);
            r.addFileInput(.{ .cwd_relative = b.fmt("{s}/{s}/lib/lib{s}.dylib", .{ out, x.name, x.name }) });
        }
    }
    // Every public function of `kit.blocks` builds from the staged kit.
    const blocks = b.addSystemCommand(&.{ b.graph.zig_exe, "build", "--build-file" });
    blocks.addFileArg(dir.path(b, b.fmt("{s}/checks/blocks/build.zig", .{name})));
    blocks.addArgs(&.{ "--prefix", b.fmt("{s}/.zig-cache/kit-blocks-check", .{root}), "--cache-dir", b.fmt("{s}/.zig-cache/kit-nested", .{root}) });
    blocks.has_side_effects = true;
    for (test_runs) |r| r.step.dependOn(&blocks.step);
    for (test_runs) |r| r.setEnvironmentVariable("MLX_SERVE_KIT_PLUGINS", out);
    // The header's static asserts are the layout check: a C or C++ compiler that lays a table out otherwise fails.
    for ([_][]const []const u8{ &.{ "cc", "-x", "c", "-std=c11" }, &.{ "c++", "-x", "c++", "-std=c++17" } }) |lang| {
        const cc = b.addSystemCommand(&.{b.graph.zig_exe});
        cc.addArgs(lang);
        cc.addArgs(&.{ "-Wall", "-Wextra", "-Werror", "-c" });
        cc.addFileArg(header);
        _ = cc.addPrefixedOutputFileArg("-o", b.fmt("layout-{s}.o", .{lang[0]}));
        for (test_runs) |r| r.step.dependOn(&cc.step);
    }
}

/// The build's own tests of this file (the kit's closure and what it refuses), on the host. They read the tree from its
/// root, which the run's cache key does not see: they rerun every time.
pub fn addTests(b: *std.Build, step: *std.Build.Step) void {
    const t = b.addTest(.{ .name = "build-kit-test", .root_module = b.createModule(.{
        .root_source_file = b.path("build/kit.zig"),
        .target = b.graph.host,
    }) });
    const run = b.addRunArtifact(t);
    run.setCwd(b.path("."));
    run.has_side_effects = true;
    step.dependOn(&run.step);
}

fn stage(b: *std.Build, wf: *std.Build.Step.WriteFile, name: []const u8, version: []const u8, header: std.Build.LazyPath) void {
    const reader: plugins.FsReader = .{ .b = b, .dir = b.root.root_dir.handle, .io = b.graph.io };
    const staged = switch (kitClosure(b.allocator, reader, "src/kit.zig") catch |e| std.process.fatal("kit: reading src/kit.zig's imports: {t}", .{e})) {
        .staged => |s| s,
        .refused => |why| std.process.fatal("kit: {s}", .{why}),
    };
    for (staged.files) |f| {
        const dst = b.fmt("{s}/{s}", .{ name, f });
        if (stagedInstead(f)) |text| _ = wf.add(dst, text) else _ = wf.addCopyFile(b.path(f), dst);
    }
    _ = wf.add(b.fmt("{s}/src/kernels/digests.txt", .{name}), staged.digests);
    _ = wf.addCopyFile(b.path("build/kit/build.zig"), b.fmt("{s}/build.zig", .{name}));
    // `kit.streamed`'s read pool, which a streaming plugin compiles in (`build.zig` `streamSources`).
    for ([_][]const u8{ "q3_lookahead4_exl3.c", "q3_lookahead4.h" }) |f|
        _ = wf.addCopyFile(b.path(b.fmt("src/sdk/weights/streamed/csrc/{s}", .{f})), b.fmt("{s}/src/sdk/weights/streamed/csrc/{s}", .{ name, f }));
    _ = wf.add(b.fmt("{s}/build.zig.zon", .{name}), b.fmt(
        \\.{{
        \\    .name = .mlx_serve_kit,
        \\    .version = "{s}",
        \\    .fingerprint = 0xf5cd6b8abae7f101,
        \\    .minimum_zig_version = "0.17.0",
        \\    .paths = .{{ "build.zig", "build.zig.zon", "mlx.version", "src", "lib", "include", "examples", "scripts", "LICENSE", "LICENSE-APACHE-2.0", "NOTICE" }},
        \\}}
        \\
    , .{version}));
    _ = wf.addCopyFile(b.path("lib/mlx/.version"), b.fmt("{s}/mlx.version", .{name}));
    // `kit.blocks.qmm` reuses MLX's steel tile matmul in its own kernels: the four headers, as the `mlx_steel_sources` module
    // (with MLX's gemv.h, for a plugin's own kernel built on MLX's matrix-vector kernel).
    _ = wf.addCopyFile(b.path("lib/mlx_steel_sources.zig"), b.fmt("{s}/lib/mlx_steel_sources.zig", .{name}));
    for ([_][]const u8{ "defines.h", "utils/type_traits.h", "utils/integral_constant.h", "gemm/nax.h" }) |h|
        _ = wf.addCopyFile(b.path(b.fmt("lib/mlx-src/mlx/backend/metal/kernels/steel/{s}", .{h})), b.fmt("{s}/lib/mlx-src/mlx/backend/metal/kernels/steel/{s}", .{ name, h }));
    _ = wf.addCopyFile(b.path("lib/mlx-src/mlx/backend/metal/kernels/gemv.h"), b.fmt("{s}/lib/mlx-src/mlx/backend/metal/kernels/gemv.h", .{name}));
    _ = wf.addCopyFile(b.path("lib/mlx-src/LICENSE"), b.fmt("{s}/lib/mlx-src/LICENSE", .{name}));
    _ = wf.addCopyDirectory(b.path("lib/mlx/include/mlx/c"), b.fmt("{s}/include/mlx/c", .{name}), .{ .include_extensions = &.{".h"} });
    _ = wf.addCopyFile(b.path("lib/mlxc-src/LICENSE"), b.fmt("{s}/include/mlx/LICENSE", .{name}));
    _ = wf.addCopyFile(header, b.fmt("{s}/include/mlx_serve_plugin.h", .{name}));
    // `scripts/fetch-zig.sh` stages the pinned Zig at the kit's `.zig-toolchain/`.
    _ = wf.addCopyFile(b.path("scripts/fetch-zig.sh"), b.fmt("{s}/scripts/fetch-zig.sh", .{name}));
    for ([_][]const u8{ "LICENSE", "LICENSE-APACHE-2.0", "NOTICE" }) |f| _ = wf.addCopyFile(b.path(f), b.fmt("{s}/{s}", .{ name, f }));
    for (examples) |ex| _ = wf.addCopyDirectory(b.path(b.fmt("src/plugins/{s}", .{ex})), b.fmt("{s}/examples/{s}", .{ name, ex }), .{ .include_extensions = &.{ ".zig", ".zon" } });
    _ = wf.addCopyDirectory(b.path("build/kit/blocks_check"), b.fmt("{s}/checks/blocks", .{name}), .{});
    for (check_plugins) |c| _ = wf.addCopyDirectory(b.path(b.fmt("build/kit/{s}", .{c.folder})), b.fmt("{s}/checks/{s}", .{ name, c.folder }), .{ .include_extensions = &.{ ".zig", ".zon" } });
}

/// What the kit's module may never reach: the engine, the host and the plugins. `mlx/` and `kernels/` ride along for
/// `kit.blocks`: the blocks write their layers on `mlx/ops.zig`, whose externs resolve at load against the server's libmlxc.
const kit_denied = [_][]const u8{ "src/engines/", "src/host/", "src/plugins/" };

const Staged = struct {
    files: []const []const u8,
    /// `src/kernels/digests.txt` cut to the kernel files the staged sources name: `mlx/kernels.zig` embeds every file
    /// the list names, so the kit ships the list of what it carries.
    digests: []const u8,
};

/// What `kitClosure` stages, or the first edge it refuses and why (`zig build kit` stops on it).
const Closure = union(enum) { staged: Staged, refused: []const u8 };

/// `start`'s import closure for the kit: `@import`ed sources, `@embedFile`d files and every kernel file a source names
/// by path. Refuses a source that leaves `src/` or reaches `kit_denied`, and a kernel under `kit_denied`.
/// `reader.read(path)` gives a source's text.
fn kitClosure(a: std.mem.Allocator, reader: anytype, start: []const u8) !Closure {
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    var queue: std.ArrayList([]const u8) = .empty;
    // A kernel's path from `src/`, and the first source that names it.
    var kernels: std.StringHashMapUnmanaged([]const u8) = .empty;
    try seen.put(a, start, {});
    try queue.append(a, start);
    while (queue.pop()) |path| {
        const src = if (stagedInstead(path)) |text| text else try reader.read(path);
        try namedKernels(a, src, path, &kernels);
        for (try plugins.importsOf(a, src)) |imp| {
            if (std.mem.eql(u8, imp, "std") or std.mem.eql(u8, imp, "builtin") or std.mem.eql(u8, imp, "root")) continue;
            if (isKitModule(imp)) continue;
            if (!std.mem.endsWith(u8, imp, ".zig")) return refuse(a, "{s} imports the module \"{s}\"; the kit's module reaches src/ files only", .{ path, imp });
            const next = try std.fs.path.resolveAllocPosix(a, &.{ std.fs.path.dirnamePosix(path) orelse ".", imp });
            if (!std.mem.startsWith(u8, next, "src/")) return refuse(a, "{s} -> {s} leaves src/", .{ path, next });
            if (kitDenied(next)) return refuse(a, "{s} -> {s}: the kit must not reach the engine, the host or the plugins", .{ path, next });
            if ((try seen.getOrPut(a, next)).found_existing) continue;
            try queue.append(a, next);
        }
    }
    var files: std.ArrayList([]const u8) = .empty;
    var it = seen.keyIterator();
    while (it.next()) |k| try files.append(a, k.*);
    var digests: std.ArrayList(u8) = .empty;
    var lines = std.mem.tokenizeScalar(u8, try reader.read("src/kernels/digests.txt"), '\n');
    while (lines.next()) |line| {
        const by = kernels.get(line[66..]) orelse continue;
        const file = try std.fmt.allocPrint(a, "src/{s}", .{line[66..]});
        if (kitDenied(file)) return refuse(a, "{s} names the kernel {s}: the kit must not ship the engine's, the host's or the plugins' files", .{ by, file });
        try files.append(a, file);
        try digests.print(a, "{s}\n", .{line});
    }
    std.mem.sort([]const u8, files.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    return .{ .staged = .{ .files = files.items, .digests = digests.items } };
}

fn refuse(a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !Closure {
    return .{ .refused = try std.fmt.allocPrint(a, fmt, args) };
}

fn kitDenied(path: []const u8) bool {
    for (kit_denied) |d| if (std.mem.startsWith(u8, path, d)) return true;
    return false;
}

/// Every string literal in `src` that names a kernel file (`.metal` / `.cu`, optionally `.fmt.`), as a path from `src/`,
/// mapped to the first source (`by`) that names it.
fn namedKernels(a: std.mem.Allocator, src: []const u8, by: []const u8, out: *std.StringHashMapUnmanaged([]const u8)) !void {
    var at: usize = 0;
    while (std.mem.indexOfScalarPos(u8, src, at, '"')) |open| {
        const close = std.mem.indexOfScalarPos(u8, src, open + 1, '"') orelse return;
        at = close + 1;
        const lit = src[open + 1 .. close];
        if (std.mem.indexOfScalar(u8, lit, '\n') != null) continue;
        if (!std.mem.endsWith(u8, lit, ".metal") and !std.mem.endsWith(u8, lit, ".cu")) continue;
        const gop = try out.getOrPut(a, lit);
        if (gop.found_existing) continue;
        gop.key_ptr.* = try a.dupe(u8, lit);
        gop.value_ptr.* = by;
    }
}

/// Files the kit stages with other text than the server's: the server's own foreign formats (GGUF, fp8) are not the
/// blocks' to carry, so the kit's list of them is empty and a plugin registers its own with `foreign.register`.
fn stagedInstead(path: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, path, "src/sdk/quant/foreign_builtin.zig")) return
    \\//! The kit's list of the server's foreign formats: none. A plugin registers its own with `foreign.register`.
    \\
    \\const Format = @import("foreign.zig").Format;
    \\pub const formats = [_]Format{};
    \\
    ;
    return null;
}

/// Modules a staged source may import beyond `std`: `mlx_steel_sources` ships with the kit (`lib/`); `build_options` is
/// named by test blocks and by code a plugin only reaches if it asks for it, and a plugin that does fails to compile
/// there by name.
fn isKitModule(name: []const u8) bool {
    for ([_][]const u8{ "mlx_steel_sources", "build_options" }) |m| if (std.mem.eql(u8, name, m)) return true;
    return false;
}

// ── tests ──

const testing = std.testing;

/// The tree as `zig build kit` reads it, from the repository root (the test's working directory).
const TreeReader = struct {
    a: std.mem.Allocator,
    fn read(self: TreeReader, path: []const u8) ![]const u8 {
        return std.Io.Dir.cwd().readFileAlloc(testing.io, path, self.a, .limited(16 << 20));
    }
};

fn stagedOf(c: Closure) !Staged {
    switch (c) {
        .staged => |s| return s,
        .refused => |why| {
            std.debug.print("kit: {s}\n", .{why});
            return error.KitRefused;
        },
    }
}

fn refusalOf(c: Closure) ![]const u8 {
    return switch (c) {
        .staged => error.NotRefused,
        .refused => |why| why,
    };
}

test "build kit: the kit stages the blocks, bills and kernels the Ling and GLM runners reach, the KDA-hybrid base, and the kernel contract" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const staged = try stagedOf(try kitClosure(a, TreeReader{ .a = a }, "src/kit.zig"));
    // The KDA-hybrid base (sdk/bases/kda_hybrid/: trunk, kda.zig, moe.zig) and glm5_next's table_forward.zig reach these
    // (both runners); bailing_hybrid/plugin.zig bills its prompt pass with prefill.zig (Ling); a plugin's kernels gate on
    // the contract in its own test step.
    for ([_][]const u8{
        "src/sdk/bases/blocks/compiled.zig",
        "src/sdk/bases/blocks/conv1d.zig",
        "src/sdk/bases/blocks/gdn.zig",
        "src/sdk/bases/blocks/moe.zig",
        "src/sdk/bases/blocks/state.zig",
        "src/sdk/bases/kda_hybrid/kda_hybrid.zig",
        "src/sdk/bases/kda_hybrid/trunk.zig",
        "src/sdk/bill/prefill.zig",
        "src/sdk/blocks/kda/recurrence.zig",
        "src/sdk/quant/outside_model.zig",
        "src/sdk/testing/kernel_contract/root.zig",
        "src/kernels/kda/kda_decode.metal",
        "src/kernels/kda/kda_percore.metal",
        "src/kernels/kda/qmv_header.metal",
    }) |want| {
        for (staged.files) |f| {
            if (std.mem.eql(u8, f, want)) break;
        } else {
            std.debug.print("the kit does not stage {s}\n", .{want});
            return error.NotStaged;
        }
    }
    for (staged.files) |f| try testing.expect(std.mem.startsWith(u8, f, "src/") and !kitDenied(f));
    // The KDA-hybrid base ships without its Sushi format hook, which reaches the `sushi_exl3` module.
    for (staged.files) |f| try testing.expect(!std.mem.eql(u8, f, "src/sdk/bases/kda_hybrid/builtin_formats.zig"));
    // A staged kernel ships with its digest line (`mlx/kernels.zig` embeds every file the list names).
    try testing.expect(std.mem.indexOf(u8, staged.digests, "  kernels/kda/qmv_header.metal\n") != null);
}

test "build kit: a source that reaches the plugins, the engine, the host or a module is refused, with the edge" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const deep: plugins.MapReader = .{ .files = &.{
        .{ "src/kit.zig", "pub const t = @import(\"sdk/trunk.zig\");" },
        .{ "src/sdk/trunk.zig", "const std = @import(\"std\");\nconst m = @import(\"../plugins/glm5_next/moe.zig\");" },
        .{ "src/kernels/digests.txt", "" },
    } };
    try testing.expectEqualStrings(
        "src/sdk/trunk.zig -> src/plugins/glm5_next/moe.zig: the kit must not reach the engine, the host or the plugins",
        try refusalOf(try kitClosure(a, deep, "src/kit.zig")),
    );
    const engine: plugins.MapReader = .{ .files = &.{.{ "src/kit.zig", "const d = @import(\"engines/mlx/plugin/driver.zig\");" }} };
    try testing.expectEqualStrings(
        "src/kit.zig -> src/engines/mlx/plugin/driver.zig: the kit must not reach the engine, the host or the plugins",
        try refusalOf(try kitClosure(a, engine, "src/kit.zig")),
    );
    // `sdk/quant/exl3/format.zig` is such a source: Sushi is a module the kit does not ship.
    const module: plugins.MapReader = .{ .files = &.{.{ "src/kit.zig", "pub const sushi = @import(\"sushi_exl3\");" }} };
    try testing.expectEqualStrings(
        "src/kit.zig imports the module \"sushi_exl3\"; the kit's module reaches src/ files only",
        try refusalOf(try kitClosure(a, module, "src/kit.zig")),
    );
    const outside: plugins.MapReader = .{ .files = &.{.{ "src/kit.zig", "const x = @import(\"../lib/x.zig\");" }} };
    try testing.expectEqualStrings("src/kit.zig -> lib/x.zig leaves src/", try refusalOf(try kitClosure(a, outside, "src/kit.zig")));
}

test "build kit: a kernel file under the plugins is refused, with the source that names it; a shared one is staged" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const decode_line = "a701fbad42356f0b01ab7a76d143ebe8d94fdf8f4c4e63514acd223fe00cb6fc  kernels/kda/kda_decode.metal\n";
    const digests = "78fac0ed60016ff8b92fb6354288449dccefa2640960006bd214158636bdd147  plugins/glm5_next/kernels/qmv_header.metal\n" ++ decode_line;
    const plugin_kernel: plugins.MapReader = .{ .files = &.{
        .{ "src/kit.zig", "pub const r = @import(\"sdk/blocks/kda/recurrence.zig\");" },
        .{ "src/sdk/blocks/kda/recurrence.zig", "const d = kernelSource(\"kernels/kda/kda_decode.metal\");\nconst h = kernelSource(\"plugins/glm5_next/kernels/qmv_header.metal\");" },
        .{ "src/kernels/digests.txt", digests },
    } };
    try testing.expectEqualStrings(
        "src/sdk/blocks/kda/recurrence.zig names the kernel src/plugins/glm5_next/kernels/qmv_header.metal: the kit must not ship the engine's, the host's or the plugins' files",
        try refusalOf(try kitClosure(a, plugin_kernel, "src/kit.zig")),
    );
    const shared: plugins.MapReader = .{ .files = &.{
        .{ "src/kit.zig", "pub const r = @import(\"sdk/blocks/kda/recurrence.zig\");" },
        .{ "src/sdk/blocks/kda/recurrence.zig", "const d = kernelSource(\"kernels/kda/kda_decode.metal\");" },
        .{ "src/kernels/digests.txt", digests },
    } };
    const staged = try stagedOf(try kitClosure(a, shared, "src/kit.zig"));
    try testing.expectEqual(@as(usize, 3), staged.files.len);
    try testing.expectEqualStrings("src/kernels/kda/kda_decode.metal", staged.files[0]);
    try testing.expectEqualStrings(decode_line, staged.digests);
}
