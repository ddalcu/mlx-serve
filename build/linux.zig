//! `zig build` on/for Linux: the full mlx-serve HTTP server against the
//! Linux MLX (Vulkan backend) + mlx-c pair staged into lib/mlx by
//! scripts/build-mlx-linux.sh. Mirrors the macOS graph minus everything
//! Apple-specific:
//!   - ds4 on its CUDA backend (scripts/build-ds4-linux.sh) and llama.cpp's
//!     CUDA release (scripts/fetch-llama.sh) instead of Metal; no ANE;
//!   - no IOKit/Metal frameworks, no metallib fingerprint (round_cost mixes
//!     exe bytes only when git_sha is empty, and mixMlxArtifacts no-ops
//!     without dyld);
//!   - system libwebp instead of Homebrew;
//!   - libjinja built for the host by the same staging script (the committed
//!     libjinja.a holds Mach-O objects).
//! Windows reuses this graph without ds4, libwebp or Bonjour (src/windows_stubs.c), linking the
//! MSVC-built mlx + mlx-c staged in lib/mlx-win/{include,lib} and llama.cpp's release DLLs.

const std = @import("std");
const builtin = @import("builtin");
const config = @import("config.zig");
const libs = @import("libs.zig");

/// `install`, `run`, `test`.
pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    const windows = target.result.os.tag == .windows;
    if (!windows) verifyMlxStageLinux(b);

    const version = b.option([]const u8, "version", "Version string") orelse config.readAppVersion(b) orelse "0.0.0-dev";
    const mlx_c_version = b.option([]const u8, "mlx-c-version", "Pinned mlx-c version") orelse config.readMlxcPin(b) orelse "unknown";

    const build_options = b.addOptions();
    build_options.addOption([]const u8, "version", version);
    build_options.addOption(bool, "mas", false);
    build_options.addOption([]const u8, "mlx_c_version", mlx_c_version);
    build_options.addOption([]const u8, "ds4_commit", b.option([]const u8, "ds4-commit", "Pinned ds4 submodule short commit") orelse config.readDs4Commit(b) orelse "unknown");
    build_options.addOption([]const u8, "llama_tag", config.readLlamaTag(b) orelse "unknown");
    build_options.addOption([]const u8, "git_sha", "");
    build_options.addOption(bool, "ios", false);
    build_options.addOption(bool, "units", false);
    build_options.addOption(bool, "ds4_engine", !windows);
    build_options.addOption(bool, "llama_engine", true);
    build_options.addOption(bool, "mlx_stream", false);
    build_options.addOption(bool, "plugin_profile", false);
    // The corpus replay and benchmark tests run tens of seconds in Debug (#639), so
    // `zig build test` skips them and `zig build test -Dslow-tests` runs them.
    build_options.addOption(bool, "slow_tests", b.option(bool, "slow-tests", "Also run the slow corpus-replay and benchmark tests") orelse false);

    const embedded: libs.Embedded = .init(b, target, optimize);

    // The server and its unit tests (`zig build test`, root src/tests.zig) share one module setup.
    const mlx_stamp = config.mlxStampModule(b, if (windows) "lib/mlx-win/.version" else "lib/mlx/.version");
    var mods: [2]*std.Build.Module = undefined;
    for (&mods, [_][]const u8{ "src/main.zig", "src/tests.zig" }) |*slot, root| {
        const mod = b.createModule(.{
            .root_source_file = b.path(root),
            .target = target,
            .optimize = optimize,
            .link_libcpp = true,
            .imports = &.{
                .{ .name = "build_options", .module = build_options.createModule() },
                .{ .name = "mlx_stamp", .module = mlx_stamp },
                .{ .name = "mlx_steel_sources", .module = embedded.mlx_steel_sources },
                .{ .name = "opencode2_plugin", .module = embedded.opencode2_plugin },
                .{ .name = "agent_skills", .module = embedded.agent_skills },
                .{ .name = "jinja_c", .module = libs.addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), target, optimize, "") },
                .{ .name = "stb", .module = libs.addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), target, optimize, "") },
                .{ .name = "webp", .module = if (windows)
                    libs.addCHeaderModule(b, b.path("src/webp_stub.h"), b.path("src"), target, optimize, "")
                else
                    libs.addCHeaderModule(b, .{ .cwd_relative = "/usr/include/webp/decode.h" }, .{ .cwd_relative = "/usr/include" }, target, optimize, "") },
            },
        });

        // Jinja2 template engine — same vendored sources as the macOS graph, built
        // as an ELF static lib by scripts/build-mlx-linux.sh (zig c++).
        _ = libs.addExl3Module(b, mod, target, optimize);
        if (windows) {
            mod.addCSourceFiles(.{
                .root = b.path("lib/jinja_cpp"),
                .files = &.{ "caps.cpp", "jinja_string.cpp", "jinja_wrapper.cpp", "lexer.cpp", "parser.cpp", "runtime.cpp", "value.cpp" },
                .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" },
            });
        } else mod.addObjectFile(b.path("lib/jinja_cpp/libjinja-linux.a"));
        mod.addIncludePath(b.path("lib/jinja_cpp"));
        libs.addVendoredSources(b, mod);

        // ANE offload C ABI → unavailable stubs on Linux (src/ane_stub.c); ane.zig
        // compiles unchanged and gates itself off via available() == false.
        mod.addCSourceFile(.{ .file = b.path("src/ane_stub.c"), .flags = &.{"-O2"} });

        // mlx (Vulkan fork) + mlx-c, staged in lib/mlx — same link shape as macOS.
        if (windows) {
            mod.addIncludePath(b.path("lib/mlx-win/include"));
            mod.addLibraryPath(b.path("lib/mlx-win/lib"));
            mod.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });
            mod.addCSourceFile(.{ .file = b.path("src/windows_stubs.c"), .flags = &.{"-O2"} });
            libs.addLlamaLib(b, mod);
            slot.* = mod;
            continue;
        }
        libs.addMlxLib(b, mod);
        libs.addDs4LinuxLib(b, mod);
        libs.addLlamaLib(b, mod);
        // ELF has no @loader_path: the Mach-O rpaths emitted above are inert here,
        // so the loader never finds libmlxc.so. Mirror them in $ORIGIN form.
        for ([_][]const u8{ "mlx/lib", "llama/lib", "ds4-linux" }) |dir| {
            mod.addRPath(.{ .cwd_relative = b.fmt("$ORIGIN/../../lib/{s}", .{dir}) });
            mod.addRPath(.{ .cwd_relative = b.fmt("$ORIGIN/../../../lib/{s}", .{dir}) });
        }

        // System libwebp for the vision pipeline (pkg-config resolves -lwebp).
        mod.linkSystemLibrary("webp", .{});

        // Bonjour/mDNS peer discovery (src/host/lan/lan.zig) via Avahi's dns_sd compat lib
        // (Arch: avahi ships /usr/lib/libdns_sd.so; Debian: libavahi-compat-libdnssd-dev).
        mod.linkSystemLibrary("dns_sd", .{ .use_pkg_config = .no });
        slot.* = mod;
    }

    const exe = b.addExecutable(.{
        .name = "mlx-serve",
        .root_module = mods[0],
        .win32_manifest = if (windows) b.path("src/windows/mlx-serve.manifest") else null,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    const run_step = b.step("run", "Run mlx-serve");
    run_step.dependOn(&run_cmd.step);

    const test_filter = b.option([]const u8, "test-filter", "Only run tests whose name contains this substring");
    const unit_tests = b.addTest(.{
        .root_module = mods[1],
        .filters = if (test_filter) |f| &.{f} else &.{},
    });
    const test_step = b.step("test", "Run unit tests");
    if (builtin.os.tag != .windows) test_step.dependOn(&b.top_level_steps.get("lint-test").?.step); // no python3 there
    if (builtin.os.tag != .windows) test_step.dependOn(&b.top_level_steps.get("unit-closure").?.step);
    const run_unit_tests = b.addRunArtifact(unit_tests);
    // Rerun every time when filtered, as on macOS (env-gated tests).
    if (test_filter != null) run_unit_tests.has_side_effects = true;
    test_step.dependOn(&run_unit_tests.step);
}

/// Linux counterpart of verifyMlxStage: fail loudly when scripts/
/// build-mlx-linux.sh has not staged the Linux mlx/mlx-c pair.
fn verifyMlxStageLinux(b: *std.Build) void {
    const root = config.buildRootHandle(b);
    const stage_ok = blk: {
        root.access(b.graph.io, "lib/mlx/lib/libmlxc.so", .{}) catch break :blk false;
        root.access(b.graph.io, "lib/mlx/lib/libmlx.so", .{}) catch break :blk false;
        root.access(b.graph.io, "lib/mlx/.version", .{}) catch break :blk false;
        break :blk true;
    };
    if (!stage_ok) {
        std.debug.print(
            "\n[mlx-serve] lib/mlx is not staged for Linux. Build it with:\n" ++
                "  git submodule update --init lib/mlxc-src && MLX_SOURCE=<staged Linux mlx tree> ./scripts/build-mlx-linux.sh\n\n",
            .{},
        );
        std.process.exit(1);
    }
    // The server embeds files from the opencode2 submodule
    // (lib/opencode2_plugin.zig @embedFile). A missing checkout surfaces as a
    // cryptic FileNotFound mid-compile, so check it at configure time.
    root.access(b.graph.io, "lib/opencode2-mlx-serve/LICENSE", .{}) catch {
        std.debug.print(
            "\n[mlx-serve] lib/opencode2-mlx-serve is not checked out. Run:\n" ++
                "  git submodule update --init lib/opencode2-mlx-serve\n\n",
            .{},
        );
        std.process.exit(1);
    };
    // The embedded engines (ds4 on CUDA, llama.cpp's CUDA release).
    for ([_][2][]const u8{
        .{ "lib/ds4-linux/libds4.so", "git submodule update --init lib/ds4 && ./scripts/build-ds4-linux.sh" },
        .{ "lib/llama/lib/libllama.so", "./scripts/fetch-llama.sh" },
    }) |stage| {
        root.access(b.graph.io, stage[0], .{}) catch {
            std.debug.print("\n[mlx-serve] {s} is not staged. Run:\n  {s}\n\n", .{ stage[0], stage[1] });
            std.process.exit(1);
        };
    }
}
