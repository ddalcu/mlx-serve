const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // vibe-jinja dependency for chat template rendering
    const vibe_jinja_dep = b.dependency("vibe_jinja", .{
        .target = target,
        .optimize = optimize,
    });

    const mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "vibe_jinja", .module = vibe_jinja_dep.module("vibe_jinja") },
        },
    });

    // mlx-c include/lib paths (homebrew)
    mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
    mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    mod.linkSystemLibrary("mlxc", .{});

    const exe = b.addExecutable(.{
        .name = "mlx-serve",
        .root_module = mod,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run mlx-serve");
    run_step.dependOn(&run_cmd.step);
}
