//! The plugin kit: `zig build kit` stages `build/kit/build.zig`, `src/kit.zig`'s import closure, mlx-c's headers,
//! the C header of the tables (`src/kit_header.zig`), the MLX pin, the pinned Zig's fetch script and the example
//! plugins as `zig-out/kit/mlx-serve-plugin-kit-<version>/` and its tarball. Under `zig build test`, each example is
//! built FROM that staged kit by its own `zig build` into a dylib the unit tests install and load
//! (`MLX_SERVE_KIT_PLUGINS`), and the C header compiles as C11 and C++17 with its layout asserts, so the kit and the
//! server cannot drift.

const std = @import("std");
const plugins = @import("plugins.zig");

/// The example plugins (`src/plugins/<name>/`, each with its own build.zig): `plugin new` copies them.
pub const examples = [_][]const u8{ "_example", "_example_media", "_example_quant" };

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
    for (examples) |ex| {
        const build = b.addSystemCommand(&.{ b.graph.zig_exe, "build", "--build-file" });
        build.addFileArg(dir.path(b, b.fmt("{s}/examples/{s}/build.zig", .{ name, ex })));
        build.addArgs(&.{ "--prefix", b.fmt("{s}/{s}", .{ out, ex }), "--cache-dir", b.fmt("{s}/.zig-cache/kit-nested", .{root}) });
        // The nested build is incremental itself; the tests rerun only when a dylib's bytes change.
        build.has_side_effects = true;
        for (test_runs) |r| {
            r.step.dependOn(&build.step);
            r.addFileInput(.{ .cwd_relative = b.fmt("{s}/{s}/lib/lib{s}.dylib", .{ out, ex, ex }) });
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

fn stage(b: *std.Build, wf: *std.Build.Step.WriteFile, name: []const u8, version: []const u8, header: std.Build.LazyPath) void {
    const reader: plugins.FsReader = .{ .b = b, .dir = b.root.root_dir.handle, .io = b.graph.io };
    const staged = kitClosure(b.allocator, reader, "src/kit.zig") catch |e| std.process.fatal("kit: reading src/kit.zig's imports: {t}", .{e});
    for (staged.files) |f| {
        const dst = b.fmt("{s}/{s}", .{ name, f });
        if (stagedInstead(f)) |text| _ = wf.add(dst, text) else _ = wf.addCopyFile(b.path(f), dst);
    }
    _ = wf.add(b.fmt("{s}/src/kernels/digests.txt", .{name}), staged.digests);
    _ = wf.addCopyFile(b.path("build/kit/build.zig"), b.fmt("{s}/build.zig", .{name}));
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
    // `kit.blocks.qmm` reuses MLX's steel tile matmul in its own kernels: the four headers, as the `mlx_steel_sources` module.
    _ = wf.addCopyFile(b.path("lib/mlx_steel_sources.zig"), b.fmt("{s}/lib/mlx_steel_sources.zig", .{name}));
    for ([_][]const u8{ "defines.h", "utils/type_traits.h", "utils/integral_constant.h", "gemm/nax.h" }) |h|
        _ = wf.addCopyFile(b.path(b.fmt("lib/mlx-src/mlx/backend/metal/kernels/steel/{s}", .{h})), b.fmt("{s}/lib/mlx-src/mlx/backend/metal/kernels/steel/{s}", .{ name, h }));
    _ = wf.addCopyFile(b.path("lib/mlx-src/LICENSE"), b.fmt("{s}/lib/mlx-src/LICENSE", .{name}));
    _ = wf.addCopyDirectory(b.path("lib/mlx/include/mlx/c"), b.fmt("{s}/include/mlx/c", .{name}), .{ .include_extensions = &.{".h"} });
    _ = wf.addCopyFile(b.path("lib/mlxc-src/LICENSE"), b.fmt("{s}/include/mlx/LICENSE", .{name}));
    _ = wf.addCopyFile(header, b.fmt("{s}/include/mlx_serve_plugin.h", .{name}));
    // `scripts/fetch-zig.sh` stages the pinned Zig at the kit's `.zig-toolchain/`.
    _ = wf.addCopyFile(b.path("scripts/fetch-zig.sh"), b.fmt("{s}/scripts/fetch-zig.sh", .{name}));
    for ([_][]const u8{ "LICENSE", "LICENSE-APACHE-2.0", "NOTICE" }) |f| _ = wf.addCopyFile(b.path(f), b.fmt("{s}/{s}", .{ name, f }));
    for (examples) |ex| _ = wf.addCopyDirectory(b.path(b.fmt("src/plugins/{s}", .{ex})), b.fmt("{s}/examples/{s}", .{ name, ex }), .{ .include_extensions = &.{ ".zig", ".zon" } });
    _ = wf.addCopyDirectory(b.path("build/kit/blocks_check"), b.fmt("{s}/checks/blocks", .{name}), .{});
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

/// `start`'s import closure for the kit: `@import`ed sources, `@embedFile`d files and every kernel file a source names
/// by path. Refuses a file that leaves `src/` or reaches `kit_denied`.
fn kitClosure(a: std.mem.Allocator, reader: plugins.FsReader, start: []const u8) !Staged {
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    var queue: std.ArrayList([]const u8) = .empty;
    var kernels: std.StringHashMapUnmanaged(void) = .empty;
    try seen.put(a, start, {});
    try queue.append(a, start);
    while (queue.pop()) |path| {
        const src = if (stagedInstead(path)) |text| text else try reader.read(path);
        try namedKernels(a, src, &kernels);
        for (try plugins.importsOf(a, src)) |imp| {
            if (std.mem.eql(u8, imp, "std") or std.mem.eql(u8, imp, "builtin") or std.mem.eql(u8, imp, "root")) continue;
            if (isKitModule(imp)) continue;
            if (!std.mem.endsWith(u8, imp, ".zig")) std.process.fatal("kit: {s} imports the module \"{s}\"; the kit's module reaches src/ files only", .{ path, imp });
            const next = try std.fs.path.resolveAllocPosix(a, &.{ std.fs.path.dirnamePosix(path) orelse ".", imp });
            if (!std.mem.startsWith(u8, next, "src/")) std.process.fatal("kit: {s} -> {s} leaves src/", .{ path, next });
            for (kit_denied) |d| if (std.mem.startsWith(u8, next, d)) std.process.fatal("kit: {s} -> {s}: the kit must not reach the engine, the host or the plugins", .{ path, next });
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
        if (!kernels.contains(line[66..])) continue;
        try files.append(a, try std.fmt.allocPrint(a, "src/{s}", .{line[66..]}));
        try digests.print(a, "{s}\n", .{line});
    }
    std.mem.sort([]const u8, files.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    return .{ .files = files.items, .digests = digests.items };
}

/// Every string literal in `src` that names a kernel file (`.metal` / `.cu`, optionally `.fmt.`), as a path from `src/`.
fn namedKernels(a: std.mem.Allocator, src: []const u8, out: *std.StringHashMapUnmanaged(void)) !void {
    var at: usize = 0;
    while (std.mem.indexOfScalarPos(u8, src, at, '"')) |open| {
        const close = std.mem.indexOfScalarPos(u8, src, open + 1, '"') orelse return;
        at = close + 1;
        const lit = src[open + 1 .. close];
        if (std.mem.indexOfScalar(u8, lit, '\n') != null) continue;
        if (std.mem.endsWith(u8, lit, ".metal") or std.mem.endsWith(u8, lit, ".cu")) try out.put(a, try a.dupe(u8, lit), {});
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

/// Modules a staged source may import beyond `std`: `mlx_steel_sources` ships with the kit (`lib/`); `build_options` and
/// `mlx_stream` are named by test blocks and by code a plugin only reaches if it asks for it (`moe_fp4`'s streamed
/// weights), and a plugin that does fails to compile there by name.
fn isKitModule(name: []const u8) bool {
    for ([_][]const u8{ "mlx_steel_sources", "build_options", "mlx_stream" }) |m| if (std.mem.eql(u8, name, m)) return true;
    return false;
}
