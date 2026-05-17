const std = @import("std");
const builtin = @import("builtin");

comptime {
    const required: std.SemanticVersion = .{ .major = 0, .minor = 16, .patch = 0 };
    if (builtin.zig_version.order(required) == .lt) {
        @compileError(std.fmt.comptimePrint(
            "mlx-serve requires Zig {d}.{d}.{d} or newer (have {d}.{d}.{d}). Run `brew upgrade zig`.",
            .{ required.major, required.minor, required.patch, builtin.zig_version.major, builtin.zig_version.minor, builtin.zig_version.patch },
        ));
    }
}

pub fn build(b: *std.Build) void {
    // Pin LC_BUILD_VERSION minos to macOS 14 (Sonoma) so binaries built on newer
    // runners (macos-26 in CI) still load on Sonoma. dyld refuses any image whose
    // minos is newer than the running OS.
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .os_version_min = .{ .semver = .{ .major = 14, .minor = 0, .patch = 0 } },
        },
    });
    const optimize = b.standardOptimizeOption(.{});

    // Setting any non-default target field disables Zig's native macOS SDK detection,
    // so we resolve the SDK path ourselves and surface its frameworks dir.
    const macos_sdk_frameworks: ?[]const u8 = blk: {
        if (target.result.os.tag != .macos) break :blk null;
        var code: u8 = undefined;
        const stdout = b.runAllowFail(
            &.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" },
            &code,
            .inherit,
        ) catch break :blk null;
        const sdk = std.mem.trim(u8, stdout, " \n\r\t");
        if (sdk.len == 0) break :blk null;
        break :blk b.fmt("{s}/System/Library/Frameworks", .{sdk});
    };

    if (target.result.os.tag == .macos) {
        verifyBrewDeps(b);
    }

    // Version from build option or default
    const version = b.option([]const u8, "version", "Version string") orelse "0.1.0-dev";

    const build_options = b.addOptions();
    build_options.addOption([]const u8, "version", version);

    // ds4 Metal kernel sources embedded via @embedFile and exposed as a
    // named module so src/arch/ds4.zig can import them with `@import("ds4_metal_sources")`
    // without traversing the project root.
    const ds4_metal_sources = b.createModule(.{
        .root_source_file = b.path("lib/ds4_metal_sources.zig"),
        .target = target,
        .optimize = optimize,
    });

    const mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
            .{ .name = "ds4_metal_sources", .module = ds4_metal_sources },
        },
    });

    // Jinja2 template engine (from llama.cpp's common/jinja + nlohmann/json).
    // Pre-compiled as a static library with system clang++ (C++17 requires system libc++).
    // Rebuild with: cd lib/jinja_cpp && for f in jinja_wrapper caps lexer parser runtime jinja_string value; do clang++ -std=c++17 -O2 -DNDEBUG -I . -c $f.cpp -o obj/$f.o; done && ar rcs libjinja.a obj/*.o
    mod.addObjectFile(b.path("lib/jinja_cpp/libjinja.a"));
    mod.addIncludePath(b.path("lib/jinja_cpp"));

    // stb_image for JPEG/PNG decoding in the vision pipeline
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    mod.addIncludePath(b.path("lib"));

    // ds4 inference engine for DSV4-Flash (Metal backend, macOS only). See
    // `lib/ds4/` submodule pinned at 613e9b2 and `src/arch/ds4.zig`. Kernel
    // sources are embedded via `lib/ds4_metal_sources.zig` and extracted at
    // runtime to ~/.mlx-serve/ds4-metal/<hash>/.
    addDs4Sources(b, mod);
    mod.addIncludePath(b.path("lib/ds4"));

    // mlx-c include/lib paths (homebrew)
    mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
    mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    mod.linkSystemLibrary("mlxc", .{});
    mod.linkSystemLibrary("webp", .{});

    if (macos_sdk_frameworks) |fw_path| {
        mod.addFrameworkPath(.{ .cwd_relative = fw_path });
    }
    mod.linkFramework("IOKit", .{});
    mod.linkFramework("CoreFoundation", .{});
    mod.linkFramework("Foundation", .{});
    mod.linkFramework("Metal", .{});

    const exe = b.addExecutable(.{
        .name = "mlx-serve",
        .root_module = mod,
    });

    // Ensure Mach-O header has room for install_name_tool path changes (app bundling)
    exe.headerpad_max_install_names = true;

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
            .{ .name = "ds4_metal_sources", .module = ds4_metal_sources },
        },
    });

    test_mod.addObjectFile(b.path("lib/jinja_cpp/libjinja.a"));
    test_mod.addIncludePath(b.path("lib/jinja_cpp"));
    test_mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    test_mod.addIncludePath(b.path("lib"));
    addDs4Sources(b, test_mod);
    test_mod.addIncludePath(b.path("lib/ds4"));
    test_mod.linkSystemLibrary("c++", .{});
    test_mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
    test_mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    test_mod.linkSystemLibrary("mlxc", .{});
    test_mod.linkSystemLibrary("webp", .{});

    if (macos_sdk_frameworks) |fw_path| {
        test_mod.addFrameworkPath(.{ .cwd_relative = fw_path });
    }
    test_mod.linkFramework("IOKit", .{});
    test_mod.linkFramework("CoreFoundation", .{});
    test_mod.linkFramework("Foundation", .{});
    test_mod.linkFramework("Metal", .{});

    const unit_tests = b.addTest(.{
        .root_module = test_mod,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}

fn addDs4Sources(b: *std.Build, module: *std.Build.Module) void {
    // Match ds4's Makefile flags (lib/ds4/Makefile lines 10–11). We drop
    // `-mcpu=native` so the produced binary stays portable across Apple
    // Silicon generations — ds4 itself ships portable IR for its Metal
    // kernels, and the C host code is not perf-critical compared to the GPU
    // path. `-Wno-unused-parameter` + `-Wno-unused-variable` keep upstream's
    // warnings from breaking our build without patching the submodule.
    const c_flags = &[_][]const u8{
        "-O3",
        "-ffast-math",
        "-std=c99",
        "-Wno-unused-parameter",
        "-Wno-unused-variable",
        "-Wno-unused-but-set-variable",
        "-Wno-unused-function",
        "-Wno-deprecated-declarations",
    };
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4.c"), .flags = c_flags });

    const objc_flags = &[_][]const u8{
        "-O3",
        "-ffast-math",
        "-fobjc-arc",
        "-Wno-unused-parameter",
        "-Wno-unused-variable",
        "-Wno-unused-but-set-variable",
        "-Wno-unused-function",
        "-Wno-deprecated-declarations",
    };
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_metal.m"), .flags = objc_flags });
}

const BrewDep = struct { name: []const u8, min: std.SemanticVersion };

const required_brew_deps = [_]BrewDep{
    .{ .name = "mlx", .min = .{ .major = 0, .minor = 31, .patch = 2 } },
    .{ .name = "mlx-c", .min = .{ .major = 0, .minor = 6, .patch = 0 } },
    .{ .name = "webp", .min = .{ .major = 1, .minor = 6, .patch = 0 } },
};

fn verifyBrewDeps(b: *std.Build) void {
    for (required_brew_deps) |dep| {
        var code: u8 = undefined;
        const stdout = b.runAllowFail(
            &.{ "brew", "list", "--versions", dep.name },
            &code,
            .inherit,
        ) catch {
            std.debug.print(
                "\n[mlx-serve] missing Homebrew dependency '{s}' (>= {d}.{d}.{d}). Install with: brew install mlx-c webp\n\n",
                .{ dep.name, dep.min.major, dep.min.minor, dep.min.patch },
            );
            std.process.exit(1);
        };
        const trimmed = std.mem.trim(u8, stdout, " \n\r\t");
        const space = std.mem.indexOfScalar(u8, trimmed, ' ') orelse {
            std.debug.print("[mlx-serve] cannot parse `brew list --versions {s}` output: {s}\n", .{ dep.name, trimmed });
            std.process.exit(1);
        };
        var ver_str = trimmed[space + 1 ..];
        // Strip Homebrew revision suffix (e.g., "0.6.0_2" -> "0.6.0").
        if (std.mem.indexOfScalar(u8, ver_str, '_')) |us| ver_str = ver_str[0..us];
        const have = std.SemanticVersion.parse(ver_str) catch {
            std.debug.print("[mlx-serve] cannot parse '{s}' version '{s}'\n", .{ dep.name, ver_str });
            std.process.exit(1);
        };
        if (have.order(dep.min) == .lt) {
            std.debug.print(
                "\n[mlx-serve] Homebrew '{s}' is {d}.{d}.{d}; need >= {d}.{d}.{d}. Run: brew upgrade {s}\n\n",
                .{ dep.name, have.major, have.minor, have.patch, dep.min.major, dep.min.minor, dep.min.patch, dep.name },
            );
            std.process.exit(1);
        }
    }
}
