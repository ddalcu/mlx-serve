//! The macOS graph: the engine module (the server's root and the unit tests' share one setup), the executable and its
//! `-Dunits` objects, the unit tests run as shards, the artifact Zig collects tests from on its own (Sushi's EXL3), and
//! the type check.

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
    /// The server's module (`src/main.zig`; `src/exe.zig` under `-Dunits`) and the unit tests' modules.
    exe_mod: *std.Build.Module,
    test_mod: *std.Build.Module,
    test_core_mod: *std.Build.Module,
    test_llm_mod: *std.Build.Module,
    test_media_mod: *std.Build.Module,
    test_contracts_mod: *std.Build.Module,

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
        const strip = b.option(bool, "strip", "Omit debug info and Zig symbols from mlx-serve and its main test binary");
        const exe_mod = shared.engineModule(b, target, optimize, host, "src/main.zig");
        const test_mod = shared.engineModule(b, target, optimize, host, "src/tests.zig");
        const test_core_mod = shared.engineModule(b, target, optimize, host, "src/tests_core.zig");
        const test_llm_mod = shared.engineModule(b, target, optimize, host, "src/tests_llm.zig");
        const test_media_mod = shared.engineModule(b, target, optimize, host, "src/tests_media.zig");
        const test_contracts_mod = shared.engineModule(b, target, optimize, host, "src/tests_contracts.zig");
        // CI and release strip mlx-serve because ReleaseFast debug info can exceed the macOS runner's 7 GB.
        // Tests keep Zig frames in failure traces unless the caller opts into stripping.
        exe_mod.strip = strip;
        test_mod.strip = strip;
        test_core_mod.strip = strip;
        test_llm_mod.strip = strip;
        test_media_mod.strip = strip;
        test_contracts_mod.strip = strip;
        return .{
            .b = b,
            .target = target,
            .optimize = optimize,
            .host = host,
            .version = version,
            .units = units,
            .exe_mod = exe_mod,
            .test_mod = test_mod,
            .test_core_mod = test_core_mod,
            .test_llm_mod = test_llm_mod,
            .test_media_mod = test_media_mod,
            .test_contracts_mod = test_contracts_mod,
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
        libs.addStreamSources(b, mod, std.mem.startsWith(u8, root, "src/tests"));

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
        // Plugins in at most `unit-groups` objects (`plugins.unitGroups`): each compiles the shared code it reaches once,
        // so fewer groups cost less CPU, more groups more parallelism. The default is one per core, 2 to 8.
        const cores = std.Thread.getCpuCount() catch 8;
        const default_groups = std.math.clamp(cores, 2, 8);
        const max_groups = b.option(usize, "unit-groups", "Most plugin objects under -Dunits (default: one per core, 2 to 8)") orelse default_groups;
        const groups = plugins.unitGroups(b.allocator, plugin_names, max_groups, g.optimize) catch @panic("OOM");
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
    const m = b.createModule(.{ .root_source_file = b.path(root), .target = g.target, .optimize = g.optimize, .link_libcpp = true, .strip = g.exe_mod.strip });
    var it = g.exe_mod.import_table.iterator();
    while (it.next()) |e| if (!std.mem.eql(u8, e.key_ptr.*, "sushi_exl3")) m.addImport(e.key_ptr.*, e.value_ptr.*);
    m.include_dirs.appendSlice(b.allocator, includes) catch @panic("OOM");
    libs.addMlxLib(b, m);
    _ = libs.addExl3Module(b, m, g.target, g.optimize);
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
pub fn addUnitTests(g: *const Graph, plugin_names: []const []const u8) Tests {
    const b = g.b;
    const test_filter = b.option([]const u8, "test-filter", "Only run tests whose name contains this substring");
    const qwen_preprocess_fixture = b.option(
        []const u8,
        "qwen-preprocess-fixture",
        "CPU reference fixture for the gated Qwen preprocessing parity test",
    );
    const filters: []const []const u8 = if (test_filter) |f| &.{f} else &.{};
    const test_build = b.step("test-build", "Compile unit tests without running them");
    const test_step = b.step("test", "Run unit tests");
    // The test binaries run in parallel shards (`MLX_SERVE_TEST_SHARD`, build/test_runner.zig); a filtered run is one process.
    const shards: u32 = if (test_filter != null) 1 else b.option(u32, "test-shards", "Unit-test processes run in parallel (default: one per 16 GiB of RAM past the first 16, 1 to 8)") orelse defaultShards();
    // A Run step given an env var copies the WHOLE configure-time environment into the cached configuration, so the
    // tests' own variables (MLX_SERVE_TAPE_RECORD, *_TEST_MODEL) would stay frozen at the last reconfigure.
    b.graph.poisonCache();

    const test_runs: []*std.Build.Step.Run = if (!g.units) blk: {
        const unit_tests = b.addTest(.{
            .root_module = g.test_mod,
            .filters = filters,
            .test_runner = .{ .path = b.path("build/test_runner.zig"), .mode = .server },
        });
        test_build.dependOn(&b.addInstallArtifact(unit_tests, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);
        break :blk addShardedRuns(b, unit_tests, test_step, null, "", shards, test_filter != null, qwen_preprocess_fixture);
    } else blk: {
        const pure_media = plugins.pureMediaNames(b, plugin_names);
        const media_csv = std.mem.join(b.allocator, ",", pure_media) catch @panic("OOM");
        const core_filters = if (test_filter != null) filters else coreTestFilters();
        const llm_filters = if (test_filter != null) filters else llmTestFilters(b, plugin_names, pure_media);
        const media_filters = if (test_filter != null) filters else mediaTestFilters(b, pure_media);
        const contract_filters = if (test_filter != null) filters else contractTestFilters();
        const core_bin = b.addTest(.{
            .name = "test-core",
            .root_module = g.test_core_mod,
            .filters = core_filters,
            .test_runner = .{ .path = b.path("build/test_runner.zig"), .mode = .server },
        });
        const llm_bin = b.addTest(.{
            .name = "test-llm",
            .root_module = g.test_llm_mod,
            .filters = llm_filters,
            .test_runner = .{ .path = b.path("build/test_runner.zig"), .mode = .server },
        });
        const media_bin = b.addTest(.{
            .name = "test-media",
            .root_module = g.test_media_mod,
            .filters = media_filters,
            .test_runner = .{ .path = b.path("build/test_runner.zig"), .mode = .server },
        });
        const contracts_bin = b.addTest(.{
            .name = "test-contracts",
            .root_module = g.test_contracts_mod,
            .filters = contract_filters,
            .test_runner = .{ .path = b.path("build/test_runner.zig"), .mode = .server },
        });
        for ([_]*std.Build.Step.Compile{ core_bin, llm_bin, media_bin, contracts_bin }) |bin| {
            test_build.dependOn(&b.addInstallArtifact(bin, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);
        }
        const core_step = b.step("test-core", "Run core (host, engines, SDK, ABI) unit tests");
        const llm_step = b.step("test-llm", "Run LLM model plugin proof, served and conformance tests");
        const media_step = b.step("test-media", "Run media/3D/audio/image plugin unit tests");
        const contracts_step = b.step("test-contracts", "Run kernel contract and SDK block/quant/weight unit tests");
        const llm_shards: u32 = if (test_filter != null) 1 else @min(shards, if (shards <= 4) @as(u32, 2) else 3);
        const media_shards: u32 = if (test_filter != null or shards <= 4) 1 else @min(shards, 2);
        const contract_shards: u32 = if (test_filter != null or shards <= 4) 1 else @min(shards, 5);
        const core_runs = addShardedRuns(b, core_bin, core_step, "core", media_csv, shards, test_filter != null, qwen_preprocess_fixture);
        _ = addShardedRuns(b, llm_bin, llm_step, "llm", media_csv, llm_shards, test_filter != null, null);
        _ = addShardedRuns(b, media_bin, media_step, "media", media_csv, media_shards, test_filter != null, null);
        _ = addShardedRuns(b, contracts_bin, contracts_step, "contracts", media_csv, contract_shards, test_filter != null, null);
        test_step.dependOn(core_step);
        test_step.dependOn(llm_step);
        test_step.dependOn(media_step);
        test_step.dependOn(contracts_step);
        break :blk core_runs;
    };

    plugins.addTests(b, test_step);
    config.addTests(b, test_step);
    test_step.dependOn(&b.top_level_steps.get("lint-test").?.step);
    test_step.dependOn(&b.top_level_steps.get("unit-closure").?.step);
    return .{ .step = test_step, .build = test_build, .runs = test_runs, .filter = test_filter };
}

fn addShardedRuns(
    b: *std.Build,
    bin: *std.Build.Step.Compile,
    parent: *std.Build.Step,
    suite: ?[]const u8,
    media_csv: []const u8,
    shards: u32,
    filtered: bool,
    qwen_preprocess_fixture: ?[]const u8,
) []*std.Build.Step.Run {
    const runs = b.allocator.alloc(*std.Build.Step.Run, shards) catch @panic("OOM");
    for (runs, 0..) |*run, k| {
        run.* = b.addRunArtifact(bin);
        if (suite) |s| {
            run.*.setEnvironmentVariable("MLX_SERVE_TEST_SUITE", s);
            run.*.setEnvironmentVariable("MLX_SERVE_TEST_MEDIA_PLUGINS", media_csv);
        }
        if (shards > 1) {
            run.*.setEnvironmentVariable("MLX_SERVE_TEST_SHARD", b.fmt("{d}/{d}", .{ k, shards }));
            run.*.setName(if (suite) |s| b.fmt("run test-{s} shard {d}/{d}", .{ s, k, shards }) else b.fmt("run test shard {d}/{d}", .{ k, shards }));
        } else if (suite) |s| {
            run.*.setName(b.fmt("run test-{s}", .{s}));
        }
        if (filtered) run.*.has_side_effects = true;
        if (qwen_preprocess_fixture) |fixture| {
            run.*.setEnvironmentVariable("QWEN_PREPROCESS_FIXTURE", fixture);
            run.*.addFileInput(.{ .cwd_relative = b.fmt("{s}/manifest.json", .{fixture}) });
            run.*.addFileInput(.{ .cwd_relative = b.fmt("{s}/source_rgb.bin", .{fixture}) });
            run.*.addFileInput(.{ .cwd_relative = b.fmt("{s}/pixel_values.bin", .{fixture}) });
        }
        parent.dependOn(&run.*.step);
    }
    return runs;
}

/// Compile-time `--test-filter` prefixes for `test-media` so Zig skips Sema/LLVM codegen for transitively-reached
/// `host/` and `sdk/` tests owned by the other suites.
fn mediaTestFilters(b: *std.Build, pure_media: []const []const u8) []const []const u8 {
    var list: std.ArrayList([]const u8) = .empty;
    list.appendSlice(b.allocator, &.{
        "plugins.registry_test_media.",
        "sdk.media.",
        "sdk.mesh.",
        "sdk.vision.dinov3.",
        "sdk.bases.dit.",
        "sdk.blocks.dit.linear.",
        "sdk.blocks.dit.mixed_linear.",
        "sdk.blocks.dit.modulate.",
        "sdk.blocks.dit.timestep.",
        "sdk.blocks.mlp.gelu.",
        "sdk.blocks.norms.layer_norm_f32.",
        "sdk.blocks.schedulers.",
        "sdk.blocks.text_encoders.",
        "sdk.blocks.vae.",
        "host.media_jobs.plugin.flip_test.",
        "host.media_jobs.plugin.flip_image_test.",
    }) catch @panic("OOM");
    for (pure_media) |m| list.append(b.allocator, b.fmt("plugins.{s}.", .{m})) catch @panic("OOM");
    return list.items;
}

/// Compile-time `--test-filter` prefixes for `test-core`: host, engines, ABI, MLX and non-contract/non-media SDK.
fn coreTestFilters() []const []const u8 {
    return &.{
        "host.",
        "engines.",
        "abi.",
        "mlx.",
        "testing.",
        "env.",
        "version.",
        "kit_header.",
        "sdk.config.",
        "sdk.dylib.",
        "sdk.quant.mxfp8.",
        "sdk.tokenizer.",
        "sdk.grammar.",
        "sdk.spec.",
        "sdk.util.",
        "sdk.ane.",
        "sdk.kv.",
        "sdk.request.",
        "sdk.vision.",
        "sdk.bases.",
        "sdk.bill.",
        "sdk.diag.",
        "sdk.plugin.",
        "sdk.root.",
        "sdk.testing.conformance.",
        "sdk.testing.fixture.",
        "sdk.testing.media_pack.",
        "sdk.testing.parity.",
        "sdk.testing.tape.",
    };
}

/// Compile-time `--test-filter` prefixes for `test-llm`: model plugins, clef, proof/served harnesses and conformance.
fn llmTestFilters(b: *std.Build, plugin_names: []const []const u8, pure_media: []const []const u8) []const []const u8 {
    var list: std.ArrayList([]const u8) = .empty;
    list.appendSlice(b.allocator, &.{
        "plugins.registry.",
        "plugins.registry_test_core.",
        "plugins._template",
        "engines.mlx.plugin.proof",
        "engines.mlx.plugin.mlx_lm_configs_test.",
        "engines.mlx.plugin.deepseek_test.",
        "host.media_jobs.plugin.clef_flip_test.",
    }) catch @panic("OOM");
    for (plugin_names) |n| {
        const is_pm = for (pure_media) |m| {
            if (std.mem.eql(u8, m, n)) break true;
        } else false;
        if (!is_pm) list.append(b.allocator, b.fmt("plugins.{s}.", .{n})) catch @panic("OOM");
    }
    return list.items;
}

fn contractTestFilters() []const []const u8 {
    return &.{
        ".contract.",
        "sdk.testing.kernel_contract.",
        "sdk.testing.shard.",
        "sdk.blocks.",
        "sdk.quant.",
        "sdk.weights.",
        "plugins.deepseek_v41.streamed_test.",
        "plugins.gemma4.gelu_ple_test.",
        "plugins.qwen4_exp.trunk.warmup.",
    };
}

/// Every shard holds its own GPU tests' memory, and the heaviest (dsv41 moe_down) wants ~10 GB: a 16 GB Mac hung with
/// two shards (a Metal command buffer that fails for memory never signals, so its test waits forever).
fn defaultShards() u32 {
    const gib = (std.process.totalSystemMemory() catch return 8) >> 30;
    return @intCast(std.math.clamp((gib -| 16) / 16, 1, 8));
}

/// `exl3-test`: Zig collects tests from an artifact's root module only, so the module over the plugin host surface
/// (`src/plugin_host.zig`) is an artifact of its own.
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
}

/// `zig build check`: type-check the engine and its tests without generating
/// code. The build system passes `-fno-emit-bin` when no step asks for the
/// binary, so this is AstGen + Sema alone — the whole tree in seconds, where
/// emitting it costs minutes (LLVM, single-threaded, ~4x more again at
/// ReleaseFast). For the edit loop; a real build still gates every change.
pub fn addCheck(g: *const Graph, plugin_names: []const []const u8) void {
    const b = g.b;
    const check = b.step("check", "Type-check everything, emitting no binary");
    check.dependOn(&b.addExecutable(.{ .name = "mlx-serve", .root_module = g.exe_mod }).step);
    if (g.units) {
        const pure_media = plugins.pureMediaNames(b, plugin_names);
        check.dependOn(&b.addTest(.{ .name = "test-core", .root_module = g.test_core_mod, .filters = coreTestFilters() }).step);
        check.dependOn(&b.addTest(.{ .name = "test-llm", .root_module = g.test_llm_mod, .filters = llmTestFilters(b, plugin_names, pure_media) }).step);
        check.dependOn(&b.addTest(.{ .name = "test-media", .root_module = g.test_media_mod, .filters = mediaTestFilters(b, pure_media) }).step);
        check.dependOn(&b.addTest(.{ .name = "test-contracts", .root_module = g.test_contracts_mod, .filters = contractTestFilters() }).step);
    } else {
        check.dependOn(&b.addTest(.{ .root_module = g.test_mod }).step);
    }
}
