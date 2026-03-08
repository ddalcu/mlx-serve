const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
    });

    // jinja.cpp (C++ Jinja2 engine compiled by Zig)
    mod.addCSourceFiles(.{
        .files = &.{"lib/jinja_cpp/jinja_wrapper.cpp"},
        .flags = &.{ "-std=c++11", "-DNDEBUG" },
    });
    mod.addIncludePath(b.path("lib/jinja_cpp"));
    mod.addIncludePath(b.path("lib/jinja_cpp/third_party"));

    // mlx-c include/lib paths (homebrew)
    mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
    mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    mod.linkSystemLibrary("mlxc", .{});

    const exe = b.addExecutable(.{
        .name = "mlx-serve",
        .root_module = mod,
    });

    exe.linkFramework("IOKit");
    exe.linkFramework("CoreFoundation");

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run mlx-serve");
    run_step.dependOn(&run_cmd.step);
}
