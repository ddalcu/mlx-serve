const std = @import("std");
const builtin = @import("builtin");

comptime {
    // 0.17.0 isn't tagged stable yet (homebrew still ships 0.16.0) — a nightly
    // build from ziglang.org/download is required until it is. 0.16.0's
    // bundled libc++ fails to compile against the macOS 27 beta SDK
    // (`use of undeclared identifier 'INFINITY'` in its vendored <random>);
    // fixed upstream by 0.17.0-dev, which is why the floor moved.
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 17) {
        @compileError(std.fmt.comptimePrint(
            "mlx-serve requires Zig 0.17 (nightly until 0.17.0 stable ships) (have {d}.{d}.{d}). Grab a nightly from https://ziglang.org/download/.",
            .{ builtin.zig_version.major, builtin.zig_version.minor, builtin.zig_version.patch },
        ));
    }
}

pub fn build(b: *std.Build) void {
    // Pin LC_BUILD_VERSION minos to macOS 26.2 — the honest floor: the linked
    // libmlx is built at deployment target 26.2 (NAX kernels, scripts/
    // build-mlx.sh), so on older macOS the binary can't run anyway; failing at
    // the binary with a clear dyld version error beats "loading" and dying on
    // the dylib. Matches app LSMinimumSystemVersion + Package.swift. Guard:
    // tests/test_mlx_staged_nax.sh (binary minos check).
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .os_version_min = .{ .semver = .{ .major = 26, .minor = 2, .patch = 0 } },
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
        verifyMlxStage(b);
    }

    // Version from build option or default
    const version = b.option([]const u8, "version", "Version string") orelse "0.1.0-dev";

    const mas = b.option(bool, "mas", "MAS build (no curl/model-pull subprocess)") orelse false;

    // Engine-version pins surfaced by `mlx-serve --version` (the macOS app spawns
    // it and parses the output — see src/version.zig). These are the versions
    // that have NO runtime query API (MLX + ggml report themselves at runtime):
    //   --mlx-c-version  pinned mlx-c submodule version; defaults from the
    //                    lib/mlx/.version stamp (written by scripts/build-mlx.sh)
    //   --ds4-commit     pinned ds4 submodule short commit (build.sh: `git rev-parse`)
    //   --llama-tag      llama.cpp release tag; defaults from lib/llama/.version
    //                    (written by scripts/fetch-llama.sh) so a plain dev build
    //                    still reports it. app/build.sh passes all three.
    const mlx_c_version = b.option([]const u8, "mlx-c-version", "Pinned mlx-c version") orelse readMlxcPin(b) orelse "unknown";
    const ds4_commit = b.option([]const u8, "ds4-commit", "Pinned ds4 submodule short commit") orelse "unknown";
    const llama_tag = b.option([]const u8, "llama-tag", "llama.cpp release tag (bNNNN)") orelse readLlamaTag(b) orelse "unknown";

    const build_options = b.addOptions();
    build_options.addOption([]const u8, "version", version);
    build_options.addOption(bool, "mas", mas);
    build_options.addOption([]const u8, "mlx_c_version", mlx_c_version);
    build_options.addOption([]const u8, "ds4_commit", ds4_commit);
    build_options.addOption([]const u8, "llama_tag", llama_tag);
    // false for the macOS exe/tests; the iOS static-lib step (`zig build ios-lib`)
    // builds its own options with ios=true so the engine swaps the macOS-only
    // ds4 + llama.cpp engines for no-op stubs (iOS serves MLX safetensors only).
    build_options.addOption(bool, "ios", false);

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
            .{ .name = "jinja_c", .module = addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), target, optimize) },
            .{ .name = "stb", .module = addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), target, optimize) },
            .{ .name = "webp", .module = addCHeaderModule(b, .{ .cwd_relative = "/opt/homebrew/include/webp/decode.h" }, .{ .cwd_relative = "/opt/homebrew/include" }, target, optimize) },
        },
    });

    // Jinja2 template engine (from llama.cpp's common/jinja + nlohmann/json).
    // Pre-compiled as a static library with system clang++ (C++17 requires system libc++).
    // Rebuild with: cd lib/jinja_cpp && for f in jinja_wrapper caps lexer parser runtime jinja_string value; do clang++ -std=c++17 -O2 -DNDEBUG -I . -c $f.cpp -o obj/$f.o; done && ar rcs libjinja.a obj/*.o
    mod.addObjectFile(b.path("lib/jinja_cpp/libjinja.a"));
    mod.addIncludePath(b.path("lib/jinja_cpp"));

    // stb_image for JPEG/PNG decoding in the vision pipeline
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    // stb_image_write for PNG encoding (native image-generation endpoint)
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_write_impl.c"), .flags = &.{"-O2"} });
    mod.addIncludePath(b.path("lib"));

    // xatlas UV unwrapping (MIT, vendored amalgamation) + C shim for the
    // Hunyuan3D texture paint stage. See lib/xatlas/xatlas_shim.h + src/uvwrap.zig.
    mod.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas_shim.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addIncludePath(b.path("lib/xatlas"));

    // ds4 inference engine for DSV4-Flash (Metal backend, macOS only). See
    // `lib/ds4/` submodule pinned at 613e9b2 and `src/arch/ds4.zig`. Kernel
    // sources are embedded via `lib/ds4_metal_sources.zig` and extracted at
    // runtime to ~/.mlx-serve/ds4-metal/<hash>/.
    addDs4Sources(b, mod);
    mod.addIncludePath(b.path("lib/ds4"));

    // llama.cpp libllama for generic GGUF models (Metal backend, macOS only).
    // Staged by `scripts/fetch-llama.sh` into lib/llama/ (a single self-contained
    // dylib + headers extracted from the pinned XCFramework). See src/arch/llama.zig.
    addLlamaLib(b, mod);

    // mlx + mlx-c: self-built from the pinned submodules (lib/mlx-src,
    // lib/mlxc-src) into lib/mlx by scripts/build-mlx.sh, with NAX kernels
    // enabled (the Homebrew bottle ships without them). MUST come before the
    // /opt/homebrew lib path so a leftover brew mlx-c can never win the link.
    addMlxLib(b, mod);
    // webp include/lib paths (homebrew)
    mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
    mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
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
    run_cmd.addPassthruArgs();

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
            .{ .name = "jinja_c", .module = addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), target, optimize) },
            .{ .name = "stb", .module = addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), target, optimize) },
            .{ .name = "webp", .module = addCHeaderModule(b, .{ .cwd_relative = "/opt/homebrew/include/webp/decode.h" }, .{ .cwd_relative = "/opt/homebrew/include" }, target, optimize) },
        },
    });

    test_mod.addObjectFile(b.path("lib/jinja_cpp/libjinja.a"));
    test_mod.addIncludePath(b.path("lib/jinja_cpp"));
    test_mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    test_mod.addCSourceFile(.{ .file = b.path("lib/stb_image_write_impl.c"), .flags = &.{"-O2"} });
    test_mod.addIncludePath(b.path("lib"));
    test_mod.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    test_mod.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas_shim.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    test_mod.addIncludePath(b.path("lib/xatlas"));
    addDs4Sources(b, test_mod);
    test_mod.addIncludePath(b.path("lib/ds4"));
    addLlamaLib(b, test_mod);
    test_mod.linkSystemLibrary("c++", .{});
    addMlxLib(b, test_mod);
    test_mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
    test_mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    test_mod.linkSystemLibrary("webp", .{});

    if (macos_sdk_frameworks) |fw_path| {
        test_mod.addFrameworkPath(.{ .cwd_relative = fw_path });
    }
    test_mod.linkFramework("IOKit", .{});
    test_mod.linkFramework("CoreFoundation", .{});
    test_mod.linkFramework("Foundation", .{});
    test_mod.linkFramework("Metal", .{});

    const test_filter = b.option([]const u8, "test-filter", "Only run tests whose name contains this substring");
    const qwen_preprocess_fixture = b.option(
        []const u8,
        "qwen-preprocess-fixture",
        "CPU reference fixture for the gated Qwen preprocessing parity test",
    );
    const unit_tests = b.addTest(.{
        .root_module = test_mod,
        .filters = if (test_filter) |f| &.{f} else &.{},
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    if (qwen_preprocess_fixture) |fixture| {
        run_unit_tests.setEnvironmentVariable("QWEN_PREPROCESS_FIXTURE", fixture);
        run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/manifest.json", .{fixture}) });
        run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/source_rgb.bin", .{fixture}) });
        run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/pixel_values.bin", .{fixture}) });
    }
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // ── vz-agent: the Agent Sandbox's guest-side binary.
    //
    // A standalone static aarch64-linux-musl ELF (~200 KB) that the app injects
    // into the guest rootfs before boot, exactly like `/.vz-init`. It serves the
    // vsock exec protocol (`src/vz_agent.zig`), replacing the hvc1 console shell.
    //
    // It is NOT imported by main.zig — it links nothing but libc and never runs
    // on macOS. Its tests do, though: `serveConnection` is OS-agnostic, so the
    // whole request → spawn → stream → exit path is exercised over a socketpair
    // here on the host. Wire them into `zig build test` explicitly, since the
    // main test module's root never reaches this file.
    addVzAgent(b, target, optimize, test_step);

    // ── iOS on-device engine: a static library (libmlxserve.a) linking the
    //    MLX-only decode path. ds4 + llama.cpp are stubbed (build_options.ios =
    //    true). Two slices: `zig build ios-lib` (device, arm64-iphoneos) and
    //    `zig build ios-lib-sim` (arm64 iphonesimulator). Driven by the iPhone
    //    app project's build scripts (../mlx-iphone/scripts/build-zig-ios.sh),
    //    which supply the matching --sysroot and copy the artifact out of
    //    zig-out/ios/<sdk>/lib. `-Dios-include=<dir>` points at the iOS dist's
    //    include dir for third-party headers (webp); defaults to Homebrew's,
    //    whose versions are pinned identical by verifyBrewDeps.
    const ios_include = b.option([]const u8, "ios-include", "Include dir for webp/stb headers when cross-compiling the iOS lib") orelse "/opt/homebrew/include";
    addIosLib(b, version, ios_include, .{ .step = "ios-lib", .abi = .none, .sdk = "iphoneos" });
    addIosLib(b, version, ios_include, .{ .step = "ios-lib-sim", .abi = .simulator, .sdk = "iphonesimulator" });
}

/// `zig build vz-agent` → `zig-out/guest/vz-agent` (static aarch64 Linux ELF),
/// plus the host-side unit tests wired into `zig build test`.
fn addVzAgent(
    b: *std.Build,
    host_target: std.Build.ResolvedTarget,
    host_optimize: std.builtin.OptimizeMode,
    test_step: *std.Build.Step,
) void {
    // Guest binary. musl + static so it runs on ANY base image — the bundled
    // Debian rootfs for the App Store build, or a user-chosen alpine/slim image.
    const guest_target = b.resolveTargetQuery(.{
        .cpu_arch = .aarch64,
        .os_tag = .linux,
        .abi = .musl,
    });
    const guest = b.addExecutable(.{
        .name = "vz-agent",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/vz_agent.zig"),
            .target = guest_target,
            // Size, not speed: it shuttles bytes between a socket and a pipe.
            .optimize = .ReleaseSmall,
            .link_libc = true,
        }),
    });
    const install = b.addInstallArtifact(guest, .{
        .dest_dir = .{ .override = .{ .custom = "guest" } },
    });
    const step = b.step("vz-agent", "Build the Agent Sandbox guest binary (static aarch64-linux)");
    step.dependOn(&install.step);

    // Host-side tests of the same source.
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/vz_agent.zig"),
            .target = host_target,
            .optimize = host_optimize,
            .link_libc = true,
        }),
    });
    test_step.dependOn(&b.addRunArtifact(tests).step);

    // A macOS-native build of the SAME source, which listens on a unix socket
    // instead of vsock. `GuestExecInteropTests` (Swift) drives it, so the host
    // frame driver and the guest agent are proven against each other without a
    // VM — the golden-byte tests alone can't catch a streaming bug.
    const host_agent = b.addExecutable(.{
        .name = "vz-agent-host",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/vz_agent.zig"),
            .target = host_target,
            .optimize = host_optimize,
            .link_libc = true,
        }),
    });
    const host_step = b.step("vz-agent-host", "Build vz-agent natively (unix-socket mode, for interop tests)");
    host_step.dependOn(&b.addInstallArtifact(host_agent, .{}).step);
}

const IosSlice = struct { step: []const u8, abi: std.Target.Abi, sdk: []const u8 };

fn addIosLib(b: *std.Build, version: []const u8, ios_include: []const u8, slice: IosSlice) void {
    // Min 18.0 to match the MLX metallib (Metal 3.2). abi=.none → device,
    // abi=.simulator → iOS Simulator slice.
    const ios_target = b.resolveTargetQuery(.{
        .cpu_arch = .aarch64,
        .os_tag = .ios,
        .os_version_min = .{ .semver = .{ .major = 18, .minor = 0, .patch = 0 } },
        .abi = slice.abi,
    });

    const ios_options = b.addOptions();
    ios_options.addOption([]const u8, "version", version);
    ios_options.addOption(bool, "ios", true);

    const mod = b.createModule(.{
        .root_source_file = b.path("src/ios_lib.zig"),
        .target = ios_target,
        .optimize = .ReleaseFast,
        .link_libc = true,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = ios_options.createModule() },
        },
    });

    // Apple cross-compiles don't auto-resolve the SDK's libc/frameworks from
    // --sysroot alone, so wire them explicitly (resolved per slice via xcrun).
    //
    // NO iOS SDK → register NOTHING (the `ios-lib` steps just don't exist in
    // this environment) instead of failing the whole configure: app/build.sh
    // pins DEVELOPER_DIR to the CommandLineTools for the macOS link, and CLT
    // ships no iOS SDKs — a @panic here aborted every macOS app build even
    // though nobody asked for an iOS step.
    var code: u8 = undefined;
    const sdk_path = b.runAllowFail(
        &.{ "xcrun", "--sdk", slice.sdk, "--show-sdk-path" },
        &code,
        .ignore, // silent when absent — CLT environments hit this on purpose
    ) catch return;
    const ios_sdk = std.mem.trim(u8, sdk_path, " \n\r\t");
    if (ios_sdk.len == 0) return;
    mod.addSystemIncludePath(.{ .cwd_relative = b.fmt("{s}/usr/include", .{ios_sdk}) });
    mod.addFrameworkPath(.{ .cwd_relative = b.fmt("{s}/System/Library/Frameworks", .{ios_sdk}) });

    // Headers for the @import("jinja_c")/@import("stb") sites (jinja_wrapper.h,
    // stb_image.h, webp/decode.h). The matching static archives are linked by
    // Xcode at final app-link time.
    mod.addIncludePath(b.path("lib/jinja_cpp"));
    mod.addIncludePath(b.path("lib"));
    mod.addIncludePath(.{ .cwd_relative = ios_include });
    mod.addImport("jinja_c", addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), ios_target, .ReleaseFast));
    mod.addImport("stb", addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), ios_target, .ReleaseFast));
    mod.addImport("webp", addCHeaderModule(b, .{ .cwd_relative = b.fmt("{s}/webp/decode.h", .{ios_include}) }, .{ .cwd_relative = ios_include }, ios_target, .ReleaseFast));
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_write_impl.c"), .flags = &.{"-O2"} });
    // xatlas UV unwrapping (C++), used by the Hunyuan3D texture paint stage via
    // src/uvwrap.zig extern decls — compiled into the lib like the macOS exe.
    mod.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas_shim.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addIncludePath(b.path("lib/xatlas"));

    const lib = b.addLibrary(.{
        .name = "mlxserve",
        .root_module = mod,
        .linkage = .static,
    });
    lib.bundle_compiler_rt = true;

    const install = b.addInstallArtifact(lib, .{
        .dest_dir = .{ .override = .{ .custom = b.fmt("ios/{s}/lib", .{slice.sdk}) } },
    });
    const step = b.step(slice.step, b.fmt("Build the iOS engine static lib ({s})", .{slice.sdk}));
    step.dependOn(&install.step);
}

/// Translates a single C header into an importable module (`@import("name")`
/// at the call site) via `addTranslateC`, replacing an inline `@cImport` —
/// removed as a language builtin in 0.17.0-dev.
fn addCHeaderModule(
    b: *std.Build,
    header_path: std.Build.LazyPath,
    include_dir: std.Build.LazyPath,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    const translate = b.addTranslateC(.{
        .root_source_file = header_path,
        .target = target,
        .optimize = optimize,
    });
    translate.addIncludePath(include_dir);
    return translate.createModule();
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
    // ds4.c #includes ds4_distributed.h; the engine/session path links its impl.
    // ds4_gpu.h is implemented in ds4_metal.m; ds4_kvstore/web/help/agent.c and
    // ds4_gpu_args.c are CLI/server-only and not part of the library path
    // mlx-serve embeds (upstream Makefile CORE_OBJS is the authority).
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_distributed.c"), .flags = c_flags });
    // SSD weight-streaming (issue #39): ds4_ssd.c is a standalone TU (#includes
    // only ds4_ssd.h) implementing the streaming expert cache the engine_options
    // ssd_streaming_* fields drive. Added upstream after the previous pin.
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_ssd.c"), .flags = c_flags });
    // Two-machine tensor parallelism + multi-GPU layer placement (pin efdadd4):
    // ds4.c references ds4_tp_* and ds4_compute_layer_placement/ds4_layer_pack_print
    // unconditionally, so both TUs must link even though we never enable TP.
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_tp.c"), .flags = c_flags });
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_layer_pack.c"), .flags = c_flags });
    // Our own shim: exports sizeof/offsetof of the real C structs so the
    // ds4_ffi.zig layout test catches mirror drift (mid-struct-insert class).
    module.addCSourceFile(.{ .file = b.path("src/ds4_layout_check.c"), .flags = c_flags });

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

fn buildRootHandle(b: *std.Build) std.Io.Dir {
    return b.root.root_dir.handle;
}

/// The llama.cpp tag staged by scripts/fetch-llama.sh (it writes LLAMA_TAG to
/// `lib/llama/.version`). Read at configure time so a plain `zig build` reports
/// the real tag without app/build.sh having to pass `--llama-tag`. Returns null
/// (→ "unknown") when llama hasn't been fetched yet.
fn readLlamaTag(b: *std.Build) ?[]const u8 {
    const bytes = buildRootHandle(b).readFileAlloc(
        b.graph.io,
        "lib/llama/.version",
        b.allocator,
        .limited(256),
    ) catch return null;
    const trimmed = std.mem.trim(u8, bytes, " \t\r\n");
    return if (trimmed.len == 0) null else b.dupe(trimmed);
}

fn addLlamaLib(b: *std.Build, module: *std.Build.Module) void {
    // Link the prebuilt libllama staged by scripts/fetch-llama.sh. The dylib's
    // install-name is @rpath/libllama.dylib; we add an rpath to its build-tree
    // location so `zig build run` / unit tests resolve it in dev. The app bundle
    // and CLI tarball rewrite that reference to @executable_path/... and re-sign
    // with the Developer ID (see release.yml / app/build.sh).
    module.addIncludePath(b.path("lib/llama/include"));
    module.addLibraryPath(b.path("lib/llama/lib"));
    // use_pkg_config = .no: a Homebrew `llama.cpp` install ships a llama.pc that
    // would otherwise hijack this link (pulling in /opt/homebrew's version + its
    // separate libggml). We want exactly the pinned dylib staged in lib/llama/lib.
    module.linkSystemLibrary("llama", .{ .use_pkg_config = .no });
    module.addRPath(b.path("lib/llama/lib"));

    // Our clean C shim over llama.h (src/llama_ffi.zig mirrors lib/llama_shim/llama_shim.h).
    // C11 for pthread_once-based one-time backend init.
    module.addIncludePath(b.path("lib/llama_shim"));
    module.addCSourceFile(.{
        .file = b.path("lib/llama_shim/llama_shim.c"),
        .flags = &.{ "-O2", "-std=c11", "-Wno-unused-parameter" },
    });
}

/// Link the self-built mlx + mlx-c staged in lib/mlx by scripts/build-mlx.sh
/// (pinned submodules lib/mlx-src + lib/mlxc-src, deployment target 26.2 so
/// MLX's NAX kernels are compiled in — the Homebrew bottle ships without them
/// and hard-wires is_nax_available() false even on M5). Install names are
/// @rpath/...; the build-tree rpath resolves them in dev, release.yml /
/// app/build.sh rewrite to @executable_path and re-sign for bundles.
/// Guard test: tests/test_mlx_staged_nax.sh.
fn addMlxLib(b: *std.Build, module: *std.Build.Module) void {
    module.addIncludePath(b.path("lib/mlx/include"));
    module.addLibraryPath(b.path("lib/mlx/lib"));
    // use_pkg_config = .no: a leftover Homebrew mlx-c must never hijack this
    // link — we want exactly the staged NAX-enabled pair (same class as the
    // llama.pc hijack above).
    module.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });
    module.addRPath(b.path("lib/mlx/lib"));
}

/// Configure-time check that scripts/build-mlx.sh has staged the pinned
/// mlx/mlx-c build. Mirrors verifyBrewDeps: fail loudly with the fix, never
/// let the linker produce a confusing -lmlxc error (or silently pick up a
/// leftover brew copy from /opt/homebrew/lib).
fn verifyMlxStage(b: *std.Build) void {
    const stage_ok = blk: {
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/libmlxc.dylib", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/mlx.metallib", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/.version", .{}) catch break :blk false;
        break :blk true;
    };
    if (!stage_ok) {
        std.debug.print(
            "\n[mlx-serve] lib/mlx is not staged (self-built mlx + mlx-c). Run:\n" ++
                "  git submodule update --init lib/mlx-src lib/mlxc-src && ./scripts/build-mlx.sh\n\n",
            .{},
        );
        std.process.exit(1);
    }
}

/// The pinned mlx-c revision from lib/mlx/.version (written by
/// scripts/build-mlx.sh as "mlx=<sha> mlxc=<sha> target=<ver>"), surfaced in
/// `mlx-serve --version`. Returns null (→ "unknown") when not staged yet.
fn readMlxcPin(b: *std.Build) ?[]const u8 {
    const bytes = buildRootHandle(b).readFileAlloc(
        b.graph.io,
        "lib/mlx/.version",
        b.allocator,
        .limited(256),
    ) catch return null;
    var it = std.mem.tokenizeScalar(u8, std.mem.trim(u8, bytes, " \t\r\n"), ' ');
    while (it.next()) |tok| {
        if (std.mem.startsWith(u8, tok, "mlxc=")) return b.dupe(tok["mlxc=".len..]);
    }
    return null;
}

const BrewDep = struct { name: []const u8, min: std.SemanticVersion };

const required_brew_deps = [_]BrewDep{
    // mlx + mlx-c are NOT brew deps anymore: they are pinned submodules built
    // by scripts/build-mlx.sh (see addMlxLib) so the NAX kernels ship enabled.
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
                "\n[mlx-serve] missing Homebrew dependency '{s}' (>= {d}.{d}.{d}). Install with: brew install webp\n\n",
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
