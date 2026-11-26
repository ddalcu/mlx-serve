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
//! `-Dunits` (ELF only) compiles the host, the MLX engine and the plugin groups as parallel objects, as on macOS
//! (`macos.addServer`); `src/exe.zig` and `src/host_unit.zig` read POSIX argv/envp, so Windows stays one unit.

const std = @import("std");
const builtin = @import("builtin");
const config = @import("config.zig");
const libs = @import("libs.zig");
const plugins = @import("plugins.zig");

/// `install`, `run`, `test`, `units`.
pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, units: bool, plugin_names: []const []const u8) void {
    const windows = target.result.os.tag == .windows;
    if (windows and units) {
        std.debug.print("\n[mlx-serve] -Dunits is not wired for Windows (src/exe.zig reads POSIX argv); build without it.\n\n", .{});
        std.process.exit(1);
    }
    if (!windows) verifyMlxStageLinux(b);
    // CI passes -Dstrip: without debug info LLVM has less to emit, and the PR gate's binary is never shipped.
    const strip = b.option(bool, "strip", "Omit debug info and Zig symbols from the mlx-serve executable");

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
    build_options.addOption(bool, "units", units);
    build_options.addOption(bool, "ds4_engine", !windows);
    build_options.addOption(bool, "llama_engine", true);
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
            .strip = strip,
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
    const units_step = b.step("units", "Compile every unit (-Dunits: host, engine, plugin groups) without linking");
    if (units) {
        addUnits(b, mods[0], target, optimize, strip, plugin_names, units_step);
    } else {
        units_step.dependOn(&b.addFail("pass -Dunits=true").step);
    }
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
        .test_runner = .{ .path = b.path("build/test_runner.zig"), .mode = .server },
    });
    const test_step = b.step("test", "Run unit tests");
    if (builtin.os.tag != .windows) test_step.dependOn(&b.top_level_steps.get("lint-test").?.step); // no python3 there
    if (builtin.os.tag != .windows) test_step.dependOn(&b.top_level_steps.get("unit-closure").?.step);
    const run_unit_tests = b.addRunArtifact(unit_tests);
    // Rerun every time when filtered, as on macOS (env-gated tests).
    if (test_filter != null) run_unit_tests.has_side_effects = true;
    test_step.dependOn(&run_unit_tests.step);
}

/// `-Dunits`: the host, the MLX engine, the module archs and the plugin groups become objects of their own, compiled in
/// parallel and linked into `mod`, whose root becomes `src/exe.zig`. Mirrors `macos.addServer`.
fn addUnits(
    b: *std.Build,
    mod: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    strip: ?bool,
    plugin_names: []const []const u8,
    units_step: *std.Build.Step,
) void {
    // Taken before any unit joins `mod`: an added object becomes one of its include sources, which would chain the units.
    const includes = b.allocator.dupe(std.Build.Module.IncludeDir, mod.include_dirs.items) catch @panic("OOM");
    for ([_][2][]const u8{
        .{ "host", "src/host_unit.zig" },
        .{ "engine-mlx", "src/engine_unit.zig" },
        .{ "engine-modules", "src/modules_unit.zig" },
    }) |u| addUnit(b, mod, units_step, unitModule(b, mod, includes, target, optimize, strip, u[1]), u[0], u[0]);
    mod.root_source_file = b.path("src/exe.zig");
    const n_groups = @min(plugin_names.len, b.option(usize, "unit-groups", "Plugin objects under -Dunits (default 8)") orelse 8);
    const groups = b.allocator.alloc([]const []const u8, n_groups) catch @panic("OOM");
    for (groups, 0..) |*names, gi| {
        var list: std.ArrayList([]const u8) = .empty;
        var i = gi;
        while (i < plugin_names.len) : (i += n_groups) list.append(b.allocator, plugin_names[i]) catch @panic("OOM");
        names.* = list.items;
    }
    for (plugins.writeUnitRoots(b, groups), 0..) |root, gi| {
        addUnit(b, mod, units_step, unitModule(b, mod, includes, target, optimize, strip, root), b.fmt("unit-{d}", .{gi}), b.fmt("{d}", .{gi}));
    }
}

fn addUnit(b: *std.Build, mod: *std.Build.Module, units_step: *std.Build.Step, um: *std.Build.Module, name: []const u8, file: []const u8) void {
    const obj = b.addObject(.{ .name = name, .root_module = um });
    mod.addObject(obj);
    units_step.dependOn(&b.addInstallFile(obj.getEmittedBin(), b.fmt("units/{s}.o", .{file})).step);
}

/// A unit's module rooted at `root`: the exe module's imports and include dirs, its own absorbed EXL3 module, and no C
/// sources or system libraries (the exe links those once).
fn unitModule(
    b: *std.Build,
    exe_mod: *std.Build.Module,
    includes: []const std.Build.Module.IncludeDir,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    strip: ?bool,
    root: []const u8,
) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path(root), .target = target, .optimize = optimize, .strip = strip, .link_libcpp = true });
    var it = exe_mod.import_table.iterator();
    while (it.next()) |e| if (!std.mem.eql(u8, e.key_ptr.*, "sushi_exl3")) m.addImport(e.key_ptr.*, e.value_ptr.*);
    m.include_dirs.appendSlice(b.allocator, includes) catch @panic("OOM");
    libs.addMlxLib(b, m);
    _ = libs.addExl3Module(b, m, target, optimize);
    return m;
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
