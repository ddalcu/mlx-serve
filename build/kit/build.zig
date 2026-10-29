//! The mlx-serve plugin kit. A plugin is a folder holding `plugin.zig` and `config.zig` (the same root contract as
//! a built-in plugin: `plugin`, `table`, `config`, `conformanceFixture`), a `build.zig.zon` naming this kit as the
//! dependency `mlx_serve_kit`, and a `build.zig` calling `plugin` (`mlx-serve plugin new` writes all four).
//!
//! The plugin imports `mlx_serve` (the ABI tables and the MLX-free SDK) and `mlx_c` (mlx-c's headers). It links no
//! MLX: mlx-c's symbols resolve to the server's own libmlxc when the server loads the dylib, so one MLX runs in the
//! process and the manifest's MLX pin (`mlx.version`) must equal the server's.

const std = @import("std");

pub fn build(b: *std.Build) void {
    _ = b;
}

pub const Options = struct {
    /// The plugin's name: its folder's, its declaration's, and the dylib's (`lib<name>.dylib`).
    name: []const u8,
};

/// Adds the plugin's dylib (installed to `lib/`) and a `test` step over its MLX-free `config.zig`.
pub fn plugin(b: *std.Build, opts: Options) *std.Build.Step.Compile {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const kit = b.dependency("mlx_serve_kit", .{});
    const mlx_serve = b.createModule(.{ .root_source_file = kit.path("src/kit.zig"), .target = target, .optimize = optimize });
    const tc = b.addTranslateC(.{ .root_source_file = kit.path("include/mlx/c/mlx.h"), .target = target, .optimize = optimize });
    tc.addIncludePath(kit.path("include"));
    const imports: []const std.Build.Module.Import = &.{
        .{ .name = "mlx_serve", .module = mlx_serve },
        .{ .name = "mlx_c", .module = tc.createModule() },
    };
    const root_mod = b.createModule(.{ .root_source_file = b.path("plugin.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = imports });

    const wf = b.addWriteFiles();
    _ = wf.addCopyFile(kit.path("mlx.version"), "mlx.version");
    const entry = wf.add("dylib.zig",
        \\comptime {
        \\    @import("mlx_serve").dylib.exportManifest(@import("plugin"), @embedFile("mlx.version"));
        \\}
        \\
    );
    const lib = b.addLibrary(.{ .name = opts.name, .linkage = .dynamic, .root_module = b.createModule(.{
        .root_source_file = entry,
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "mlx_serve", .module = mlx_serve }, .{ .name = "plugin", .module = root_mod } },
    }) });
    // mlx-c resolves at load, against the server's libmlxc: never a second MLX in the process.
    lib.linker_allow_shlib_undefined = true;
    b.installArtifact(lib);

    const tests = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("config.zig"), .target = target, .optimize = optimize, .imports = imports }) });
    b.step("test", "Run the plugin's MLX-free tests").dependOn(&b.addRunArtifact(tests).step);
    return lib;
}
