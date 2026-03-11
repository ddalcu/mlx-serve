const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Version from build option or default
    const version = b.option([]const u8, "version", "Version string") orelse "0.1.0-dev";

    const build_options = b.addOptions();
    build_options.addOption([]const u8, "version", version);

    const mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
        },
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

    // Unit tests — reuses the same module config (mlx-c, jinja_cpp, etc.)
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
        },
    });

    test_mod.addCSourceFiles(.{
        .files = &.{"lib/jinja_cpp/jinja_wrapper.cpp"},
        .flags = &.{ "-std=c++11", "-DNDEBUG" },
    });
    test_mod.addIncludePath(b.path("lib/jinja_cpp"));
    test_mod.addIncludePath(b.path("lib/jinja_cpp/third_party"));
    test_mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
    test_mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    test_mod.linkSystemLibrary("mlxc", .{});

    const unit_tests = b.addTest(.{
        .root_module = test_mod,
    });
    unit_tests.linkFramework("IOKit");
    unit_tests.linkFramework("CoreFoundation");

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
