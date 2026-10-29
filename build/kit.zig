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
    const files = switch (plugins.closure(b.allocator, reader, "src/kit.zig") catch |e| std.process.fatal("kit: reading src/kit.zig's imports: {t}", .{e})) {
        .files => |f| f,
        .violation => |v| std.process.fatal("kit: {s}", .{v}),
    };
    for (files) |f| _ = wf.addCopyFile(b.path(f), b.fmt("{s}/{s}", .{ name, f }));
    _ = wf.addCopyFile(b.path("build/kit/build.zig"), b.fmt("{s}/build.zig", .{name}));
    _ = wf.add(b.fmt("{s}/build.zig.zon", .{name}), b.fmt(
        \\.{{
        \\    .name = .mlx_serve_kit,
        \\    .version = "{s}",
        \\    .fingerprint = 0xf5cd6b8abae7f101,
        \\    .minimum_zig_version = "0.17.0",
        \\    .paths = .{{ "build.zig", "build.zig.zon", "mlx.version", "src", "include", "examples", "scripts", "LICENSE", "LICENSE-APACHE-2.0", "NOTICE" }},
        \\}}
        \\
    , .{version}));
    _ = wf.addCopyFile(b.path("lib/mlx/.version"), b.fmt("{s}/mlx.version", .{name}));
    _ = wf.addCopyDirectory(b.path("lib/mlx/include/mlx/c"), b.fmt("{s}/include/mlx/c", .{name}), .{ .include_extensions = &.{".h"} });
    _ = wf.addCopyFile(b.path("lib/mlxc-src/LICENSE"), b.fmt("{s}/include/mlx/LICENSE", .{name}));
    _ = wf.addCopyFile(header, b.fmt("{s}/include/mlx_serve_plugin.h", .{name}));
    // `scripts/fetch-zig.sh` stages the pinned Zig at the kit's `.zig-toolchain/`.
    _ = wf.addCopyFile(b.path("scripts/fetch-zig.sh"), b.fmt("{s}/scripts/fetch-zig.sh", .{name}));
    for ([_][]const u8{ "LICENSE", "LICENSE-APACHE-2.0", "NOTICE" }) |f| _ = wf.addCopyFile(b.path(f), b.fmt("{s}/{s}", .{ name, f }));
    for (examples) |ex| _ = wf.addCopyDirectory(b.path(b.fmt("src/plugins/{s}", .{ex})), b.fmt("{s}/examples/{s}", .{ name, ex }), .{ .include_extensions = &.{ ".zig", ".zon" } });
}
