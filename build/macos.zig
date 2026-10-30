//! The macOS graph: the engine module (the server's root and the unit tests' share one setup), the executable and its
//! `-Dunits` objects, the unit tests run as shards, the artifacts Zig collects tests from on their own (Sushi's EXL3,
//! mlx-stream), and the type check.

const std = @import("std");
const config = @import("config.zig");
const libs = @import("libs.zig");
const plugins = @import("plugins.zig");

pub const Graph = struct {
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    host: config.Host,
    version: []const u8,
    units: bool,
    /// The server's module (`src/main.zig`; `src/exe.zig` under `-Dunits`) and the unit tests' (`src/tests.zig`).
    exe_mod: *std.Build.Module,
    test_mod: *std.Build.Module,

    pub fn init(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, host: config.Host, units: bool) Graph {
        // App version. Release builds pass it explicitly (app/build.sh computes the
        // next CalVer from the GitHub releases and stamps it into app/Info.plist;
        // the release workflow passes the tag). A plain `zig build` used to fall
        // back to a literal "0.1.0-dev", which then showed up as the version in
        // `--version` AND on the console page — so a dev build reported a version
        // that exists nowhere. Default to the checked-in Info.plist stamp instead:
        // one source of truth, already in the repo, and the same string the last
        // real build shipped. Same pattern as the engine pins below.
        const version = b.option([]const u8, "version", "Version string") orelse config.readAppVersion(b) orelse "0.0.0-dev";

        const mas = b.option(bool, "mas", "MAS build (no curl/model-pull subprocess)") orelse false;

        // Engine-version pins surfaced by `mlx-serve --version` (the macOS app spawns
        // it and parses the output — see src/version.zig). These are the versions
        // that have NO runtime query API (MLX + ggml report themselves at runtime):
        //   --mlx-c-version  pinned mlx-c submodule version; defaults from the
        //                    lib/mlx/.version stamp (written by scripts/build-mlx.sh)
        //   --ds4-commit     pinned ds4 submodule short commit; defaults from its checked-out
        //                    commit (readDs4Commit), so the release build reports it too
        //   --llama-tag      llama.cpp release tag; defaults from lib/llama/.version
        //                    (written by scripts/fetch-llama.sh) so a plain dev build
        //                    still reports it. app/build.sh passes all three.
        const mlx_c_version = b.option([]const u8, "mlx-c-version", "Pinned mlx-c version") orelse config.readMlxcPin(b) orelse "unknown";
        const ds4_commit = b.option([]const u8, "ds4-commit", "Pinned ds4 submodule short commit") orelse config.readDs4Commit(b) orelse "unknown";
        const llama_tag = b.option([]const u8, "llama-tag", "llama.cpp release tag (bNNNN)") orelse config.readLlamaTag(b) orelse "unknown";

        const build_options = b.addOptions();
        build_options.addOption([]const u8, "version", version);
        build_options.addOption(bool, "mas", mas);
        build_options.addOption([]const u8, "mlx_c_version", mlx_c_version);
        build_options.addOption([]const u8, "ds4_commit", ds4_commit);
        build_options.addOption([]const u8, "llama_tag", llama_tag);
        const git_sha = b.option([]const u8, "git-sha", "Engine build id for the round-cost table: a release sha stands for the executable bytes, which are then not hashed; the MLX dylib and metallib fingerprints are always mixed in") orelse "";
        build_options.addOption([]const u8, "git_sha", git_sha);
        // false for the macOS exe/tests; the iOS static-lib step (`zig build ios-lib`)
        // builds its own options with ios=true so the engine swaps the embedded
        // ds4 + llama.cpp engines for no-op stubs (iOS serves MLX safetensors only).
        build_options.addOption(bool, "ios", false);
        build_options.addOption(bool, "units", units);
        // True when the embedded engines (ds4, libllama) are actually linked:
        // macOS (Metal) and Linux (CUDA) exe = yes; the iOS static lib = no, and the
        // host leaves their tables out (`src/host/engines/served.zig`). Read this
        // option, NOT `ios` — `ios` keeps its own meaning (low-mem policy, sandboxing).
        build_options.addOption(bool, "ds4_engine", true);
        build_options.addOption(bool, "llama_engine", true);
        // DeepSeek-V4.1's EXL3 repack through mlx-stream (macOS only; Linux and iOS build the stub).
        build_options.addOption(bool, "mlx_stream", true);
        // `-Dplugin-profile`: the plugin's profile probes (its sdk reads `mlx_host.plugin_profile`) and their C shims.
        build_options.addOption(bool, "plugin_profile", libs.pluginProfile(b));
        // The corpus replay and benchmark tests run tens of seconds in Debug (#639), so
        // `zig build test` skips them and `zig build test -Dslow-tests` runs them.
        build_options.addOption(bool, "slow_tests", b.option(bool, "slow-tests", "Also run the slow corpus-replay and benchmark tests") orelse false);

        const shared: Shared = .{
            .build_options = build_options,
            .mlx_stamp = config.mlxStampModule(b, "lib/mlx/.version"),
            // ds4 Metal kernel sources embedded via @embedFile and exposed as a
            // named module so src/engines/ds4/bridge.zig can import them with `@import("ds4_metal_sources")`
            // without traversing the project root.
            .ds4_metal_sources = b.createModule(.{
                .root_source_file = b.path("lib/ds4_metal_sources.zig"),
                .target = target,
                .optimize = optimize,
            }),
            .embedded = .init(b, target, optimize),
        };
        return .{
            .b = b,
            .target = target,
            .optimize = optimize,
            .host = host,
            .version = version,
            .units = units,
            .exe_mod = blk: {
                const m = shared.engineModule(b, target, optimize, host, "src/main.zig");
                // CI and release pass -Dstrip: debug info is most of the ReleaseFast compile's peak memory, which
                // otherwise outgrows a GitHub macOS runner's 7 GB. A stripped binary's crash traces lose Zig names.
                m.strip = b.option(bool, "strip", "Omit debug info and Zig symbols from the mlx-serve executable");
                break :blk m;
            },
            .test_mod = shared.engineModule(b, target, optimize, host, "src/tests.zig"),
        };
    }
};

/// What the server's and the unit tests' modules import alike.
const Shared = struct {
    build_options: *std.Build.Step.Options,
    mlx_stamp: *std.Build.Module,
    ds4_metal_sources: *std.Build.Module,
    embedded: libs.Embedded,

    fn engineModule(s: Shared, b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, host: config.Host, root: []const u8) *std.Build.Module {
        const mod = b.createModule(.{
            .root_source_file = b.path(root),
            .target = target,
            .optimize = optimize,
            .link_libcpp = true,
            .imports = &.{
                .{ .name = "build_options", .module = s.build_options.createModule() },
                .{ .name = "mlx_stamp", .module = s.mlx_stamp },
                .{ .name = "ds4_metal_sources", .module = s.ds4_metal_sources },
                .{ .name = "mlx_steel_sources", .module = s.embedded.mlx_steel_sources },
                .{ .name = "opencode2_plugin", .module = s.embedded.opencode2_plugin },
                .{ .name = "agent_skills", .module = s.embedded.agent_skills },
                .{ .name = "jinja_c", .module = libs.addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), target, optimize, "") },
                .{ .name = "stb", .module = libs.addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), target, optimize, "") },
                .{ .name = "webp", .module = libs.addCHeaderModule(b, .{ .cwd_relative = b.fmt("{s}/include/webp/decode.h", .{host.brew_prefix}) }, .{ .cwd_relative = b.fmt("{s}/include", .{host.brew_prefix}) }, target, optimize, "") },
            },
        });

        // Jinja2 template engine (from llama.cpp's common/jinja + nlohmann/json).
        // Pre-compiled as a static library with system clang++ (C++17 requires system libc++).
        // Rebuild with: cd lib/jinja_cpp && for f in jinja_wrapper caps lexer parser runtime jinja_string value; do clang++ -std=c++17 -O2 -DNDEBUG -I . -c $f.cpp -o obj/$f.o; done && ar rcs libjinja.a obj/*.o
        mod.addObjectFile(b.path("lib/jinja_cpp/libjinja.a"));
        mod.addIncludePath(b.path("lib/jinja_cpp"));
        libs.addVendoredSources(b, mod);

        // ds4 inference engine for DSV4-Flash (Metal backend, macOS only). See
        // `lib/ds4/` submodule pinned at 9139e2a and `src/engines/ds4/bridge.zig`. Kernel
        // sources are embedded via `lib/ds4_metal_sources.zig` and extracted at
        // runtime to ~/.mlx-serve/ds4-metal/<hash>/.
        libs.addDs4Sources(b, mod);
        mod.addIncludePath(b.path("lib/ds4"));

        // ANE prefill-MLP offload (perf-plan-aug-17 P5): objc bridge to the
        // private AppleNeuralEngine framework (dlopen'd at runtime — the probe
        // returns unavailable on machines/OSes without it) + the per-layer MLP
        // MIL program builder. See lib/ane/ + src/sdk/ane/offload.zig; provenance in NOTICE.
        libs.addAneSources(b, mod);

        // llama.cpp libllama for generic GGUF models (Metal backend, macOS only).
        // Staged by `scripts/fetch-llama.sh` into lib/llama/ (a single self-contained
        // dylib + headers extracted from the pinned XCFramework). See src/engines/llama/bridge.zig.
        libs.addLlamaLib(b, mod);
        _ = libs.addExl3Module(b, mod, target, optimize);
        libs.addMlxStreamModule(b, mod, target, optimize);

        // mlx + mlx-c: self-built from the pinned submodules (lib/mlx-src,
        // lib/mlxc-src) into lib/mlx by scripts/build-mlx.sh, with NAX kernels
        // enabled (the Homebrew bottle ships without them). MUST come before the
        // /opt/homebrew lib path so a leftover brew mlx-c can never win the link.
        libs.addMlxLib(b, mod);
        // webp include/lib paths (homebrew)
        mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{host.brew_prefix}) });
        mod.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{host.brew_prefix}) });
        mod.linkSystemLibrary("webp", .{});
        // A copied prefix's dylibs carry @rpath install names (Homebrew's carry its absolute path): the artifact finds
        // them in the copy at run time, the bundle in its own lib/.
        if (host.webp_prefix) |dir| mod.addRPath(.{ .cwd_relative = b.fmt("{s}/lib", .{dir}) });

        if (host.frameworks) |fw_path| mod.addFrameworkPath(.{ .cwd_relative = fw_path });
        for ([_][]const u8{ "IOKit", "CoreFoundation", "Foundation", "Metal", "IOSurface" }) |f| mod.linkFramework(f, .{});
        return mod;
    }
};

/// `zig build` (the executable), `units`, `units-check`, `run`.
pub fn addServer(g: *const Graph, plugin_names: []const []const u8) void {
    const b = g.b;
    const mod = g.exe_mod;
    const exe = b.addExecutable(.{
        .name = "mlx-serve",
        .root_module = mod,
    });
    // `-Dunits`: each plugin is an object of its own, compiled in parallel with the host and linked in.
    const units_step = b.step("units", "Compile every unit (-Dunits: host, engine, plugin groups) without linking");
    // Every unit compiles its own copy of the globals it reaches: scripts/unit_globals.py fails on one in a plugin unit
    // or in two of the host, the engine and the module archs that is not `Shared` (names survive in a Debug build only).
    const units_check = b.step("units-check", "Check no unit holds a copy of another's state (-Dunits, Debug)");
    const check_run = b.addSystemCommand(&.{ "python3", "scripts/unit_globals.py" });
    check_run.has_side_effects = true;
    if (g.units) units_check.dependOn(&check_run.step) else for ([_]*std.Build.Step{ units_step, units_check }) |s| s.dependOn(&b.addFail("pass -Dunits=true").step);
    // Taken before any unit joins `mod`: an added object becomes one of its include sources, which would chain the units.
    const unit_includes = b.allocator.dupe(std.Build.Module.IncludeDir, mod.include_dirs.items) catch @panic("OOM");
    if (g.units) {
        // The host is an object too, so it compiles beside the units; the exe's root only calls into it.
        const host_obj = b.addObject(.{ .name = "host", .root_module = unitModule(g, unit_includes, "src/host_unit.zig") });
        mod.addObject(host_obj);
        units_step.dependOn(&b.addInstallFile(host_obj.getEmittedBin(), "units/host.o").step);
        check_run.addFileArg(host_obj.getEmittedBin());
        // The MLX engine (the host reaches it only through its table, `host/engines/mlx.zig`), and the module archs it
        // calls through tables (`engines/mlx/modules/split.zig`).
        for ([_][2][]const u8{ .{ "engine-mlx", "src/engine_unit.zig" }, .{ "engine-modules", "src/modules_unit.zig" } }) |u| {
            const obj = b.addObject(.{ .name = u[0], .root_module = unitModule(g, unit_includes, u[1]) });
            mod.addObject(obj);
            units_step.dependOn(&b.addInstallFile(obj.getEmittedBin(), b.fmt("units/{s}.o", .{u[0]})).step);
            check_run.addFileArg(obj.getEmittedBin());
        }
        mod.root_source_file = b.path("src/exe.zig");
        // Plugins dealt round-robin into `unit-groups` objects: each compiles the shared code it reaches once, so fewer
        // groups cost less CPU, more groups more parallelism.
        const n_groups = @min(plugin_names.len, b.option(usize, "unit-groups", "Plugin objects under -Dunits (default 8)") orelse 8);
        const groups = b.allocator.alloc([]const []const u8, n_groups) catch @panic("OOM");
        for (groups, 0..) |*names, gi| {
            var list: std.ArrayList([]const u8) = .empty;
            var i = gi;
            while (i < plugin_names.len) : (i += n_groups) list.append(b.allocator, plugin_names[i]) catch @panic("OOM");
            names.* = list.items;
        }
        for (plugins.writeUnitRoots(b, groups), 0..) |root, gi| {
            const um = unitModule(g, unit_includes, root);
            const obj = b.addObject(.{ .name = b.fmt("unit-{d}", .{gi}), .root_module = um });
            mod.addObject(obj);
            units_step.dependOn(&b.addInstallFile(obj.getEmittedBin(), b.fmt("units/{d}.o", .{gi})).step);
            check_run.addFileArg(obj.getEmittedBin());
        }
    }

    // Ensure Mach-O header has room for install_name_tool path changes (app bundling)
    exe.headerpad_max_install_names = true;

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();

    const run_step = b.step("run", "Run mlx-serve");
    run_step.dependOn(&run_cmd.step);
}

/// A `-Dunits` object's module rooted at `root`: the exe module's imports and include dirs, its own pair of absorbed
/// modules (they import their host root), and no C sources (the exe links those once).
fn unitModule(g: *const Graph, includes: []const std.Build.Module.IncludeDir, root: []const u8) *std.Build.Module {
    const b = g.b;
    const m = b.createModule(.{ .root_source_file = b.path(root), .target = g.target, .optimize = g.optimize, .link_libcpp = true });
    var it = g.exe_mod.import_table.iterator();
    while (it.next()) |e| if (!std.mem.eql(u8, e.key_ptr.*, "sushi_exl3") and !std.mem.eql(u8, e.key_ptr.*, "mlx_stream")) m.addImport(e.key_ptr.*, e.value_ptr.*);
    m.include_dirs.appendSlice(b.allocator, includes) catch @panic("OOM");
    libs.addMlxLib(b, m);
    _ = libs.addExl3Module(b, m, g.target, g.optimize);
    m.addImport("mlx_stream", libs.mlxStreamPkg(b, m, "src/mlx_stream.zig", g.target, g.optimize, false, false));
    return m;
}

pub const Tests = struct {
    /// `zig build test`.
    step: *std.Build.Step,
    /// `zig build test-build`.
    build: *std.Build.Step,
    /// The unit-test binary's shard runs.
    runs: []const *std.Build.Step.Run,
    filter: ?[]const u8,
};

/// `test-build` and `test`: the unit tests, plus the plugins' own tests, the lint's and the import closure check.
pub fn addUnitTests(g: *const Graph) Tests {
    const b = g.b;
    const test_filter = b.option([]const u8, "test-filter", "Only run tests whose name contains this substring");
    const qwen_preprocess_fixture = b.option(
        []const u8,
        "qwen-preprocess-fixture",
        "CPU reference fixture for the gated Qwen preprocessing parity test",
    );
    const unit_tests = b.addTest(.{
        .root_module = g.test_mod,
        .filters = if (test_filter) |f| &.{f} else &.{},
        .test_runner = .{ .path = b.path("build/test_runner.zig"), .mode = .server },
    });

    const test_build = b.step("test-build", "Compile unit tests without running them");
    test_build.dependOn(&b.addInstallArtifact(unit_tests, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);

    // The one test binary runs as `test-shards` processes in parallel, each every n-th test (build/test_runner.zig);
    // a filtered run is one process.
    const shards: u32 = if (test_filter != null) 1 else b.option(u32, "test-shards", "Unit-test processes run in parallel (default 8)") orelse 8;
    const test_runs = b.allocator.alloc(*std.Build.Step.Run, shards) catch @panic("OOM");
    const test_step = b.step("test", "Run unit tests");
    for (test_runs, 0..) |*run, k| {
        run.* = b.addRunArtifact(unit_tests);
        if (shards > 1) {
            run.*.setEnvironmentVariable("MLX_SERVE_TEST_SHARD", b.fmt("{d}/{d}", .{ k, shards }));
            run.*.setName(b.fmt("run test shard {d}/{d}", .{ k, shards }));
        }
        // A filtered run is a targeted one, often gated on an env var (ANE_TEST_PACK, QWEN4_TEST_MODEL)
        // that the run's cache key does not see: it reruns every time.
        if (test_filter != null) run.*.has_side_effects = true;
        if (qwen_preprocess_fixture) |fixture| {
            run.*.setEnvironmentVariable("QWEN_PREPROCESS_FIXTURE", fixture);
            run.*.addFileInput(.{ .cwd_relative = b.fmt("{s}/manifest.json", .{fixture}) });
            run.*.addFileInput(.{ .cwd_relative = b.fmt("{s}/source_rgb.bin", .{fixture}) });
            run.*.addFileInput(.{ .cwd_relative = b.fmt("{s}/pixel_values.bin", .{fixture}) });
        }
        test_step.dependOn(&run.*.step);
    }
    plugins.addTests(b, test_step);
    test_step.dependOn(&b.top_level_steps.get("lint-test").?.step);
    test_step.dependOn(&b.top_level_steps.get("unit-closure").?.step);
    return .{ .step = test_step, .build = test_build, .runs = test_runs, .filter = test_filter };
}

/// `exl3-test`, `mlx-stream-test`, `mlx-stream-conformance`: Zig collects tests from an artifact's root module only, so
/// the modules over the plugin host surface (`src/plugin_host.zig`) are artifacts of their own.
pub fn addPluginTests(g: *const Graph, tests: Tests) void {
    const b = g.b;
    {
        // The EXL3 module's CPU references run for hours unoptimized, so Debug runs it as ReleaseSafe.
        const opt: std.builtin.OptimizeMode = if (g.optimize == .debug) .safe else g.optimize;
        const host = b.createModule(.{ .root_source_file = b.path("src/plugin_host.zig"), .target = g.target, .optimize = opt, .link_libc = true, .link_libcpp = true });
        libs.addMlxLib(b, host);
        if (g.host.frameworks) |fw| host.addFrameworkPath(.{ .cwd_relative = fw });
        for ([_][]const u8{ "IOKit", "CoreFoundation", "Foundation", "Metal", "IOSurface" }) |f| host.linkFramework(f, .{});
        const t = b.addTest(.{ .name = "exl3-test", .root_module = libs.addExl3Module(b, host, g.target, opt), .filters = if (tests.filter) |f| &.{f} else &.{} });
        tests.build.dependOn(&b.addInstallArtifact(t, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);
        // ~13 min and 14 GB: its own step, run at phase gates, outside `zig build test`.
        b.step("exl3-test", "Run the in-tree EXL3/STQ (sushi) tests").dependOn(&b.addRunArtifact(t).step);
    }
    {
        const t = addMlxStreamTests(g, tests.filter, "src/mlx_stream_tests.zig", "mlx-stream-test");
        tests.build.dependOn(&b.addInstallArtifact(t, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);
        const run = b.addRunArtifact(t);
        tests.step.dependOn(&run.step);
        b.step("mlx-stream-test", "Run the mlx-stream plugin's tests").dependOn(&run.step);
        // Its own binary (the last test checks that nothing in it created a Metal device), outside `zig build test`:
        // it recompiles every plugin file's tests.
        const conf = b.addRunArtifact(addMlxStreamTests(g, null, "src/mlx_stream_conformance.zig", "mlx-stream-conformance"));
        conf.setEnvironmentVariable("MLX_DEFAULT_DEVICE", "cpu");
        b.step("mlx-stream-conformance", "Run the mlx-stream plugin's conformance suite (CPU lane, no device)").dependOn(&conf.step);
    }
}

/// One of the plugin's own test roots (src/tests.zig, src/conformance.zig) over the host surface plugins import
/// (`src/plugin_host.zig`), with the read pool's scripted faults.
fn addMlxStreamTests(g: *const Graph, filter: ?[]const u8, root: []const u8, name: []const u8) *std.Build.Step.Compile {
    const b = g.b;
    const host = b.createModule(.{ .root_source_file = b.path("src/plugin_host.zig"), .target = g.target, .optimize = g.optimize, .link_libc = true, .link_libcpp = true });
    libs.addMlxLib(b, host);
    _ = libs.addExl3Module(b, host, g.target, g.optimize);
    if (g.host.frameworks) |fw| host.addFrameworkPath(.{ .cwd_relative = fw });
    for ([_][]const u8{ "IOKit", "CoreFoundation", "Foundation", "Metal", "IOSurface" }) |f| host.linkFramework(f, .{});
    const m = libs.mlxStreamPkg(b, host, root, g.target, g.optimize, true, true);
    return b.addTest(.{ .name = name, .root_module = m, .filters = if (filter) |f| &.{f} else &.{} });
}

/// `zig build check`: type-check the engine and its tests without generating
/// code. The build system passes `-fno-emit-bin` when no step asks for the
/// binary, so this is AstGen + Sema alone — the whole tree in seconds, where
/// emitting it costs minutes (LLVM, single-threaded, ~4x more again at
/// ReleaseFast). For the edit loop; a real build still gates every change.
pub fn addCheck(g: *const Graph) void {
    const b = g.b;
    const check = b.step("check", "Type-check everything, emitting no binary");
    check.dependOn(&b.addExecutable(.{ .name = "mlx-serve", .root_module = g.exe_mod }).step);
    check.dependOn(&b.addTest(.{ .root_module = g.test_mod }).step);
}
