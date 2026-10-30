//! What a graph compiles in and links: the vendored C/C++ sources, the embedded engines (ds4, llama.cpp, ANE), the
//! staged mlx + mlx-c, the modules the engine imports by name (C headers, embedded sources, Sushi's EXL3, mlx-stream).

const std = @import("std");

/// stb_image_write's JPEG bit-packer relies on WRAPPING left shifts of a signed
/// int (`bitBuf <<= 8`), which is UB by the letter of C and which Zig's C
/// frontend TRAPS under UBSan — so a Debug build of any graph that encodes a
/// JPEG aborts inside libc. One flag list, used at every site that compiles it,
/// because the flag is about the C source and not about which graph it is in
/// (`zig build test` is Debug and ran 6 crashed tests without it).
pub const stb_write_flags: []const []const u8 = &.{ "-O2", "-fno-sanitize=undefined" };

/// stb_image (JPEG/PNG decode) + stb_image_write (PNG encode), xatlas UV unwrapping + FQMS decimation (MIT, vendored)
/// with C shims for the Hunyuan3D texture paint stage (lib/{xatlas,fqms}, src/sdk/mesh/uvwrap.zig): portable C/C++.
pub fn addVendoredSources(b: *std.Build, mod: *std.Build.Module) void {
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_write_impl.c"), .flags = stb_write_flags });
    mod.addIncludePath(b.path("lib"));
    mod.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addCSourceFile(.{ .file = b.path("lib/xatlas/xatlas_shim.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addIncludePath(b.path("lib/xatlas"));
    mod.addCSourceFile(.{ .file = b.path("lib/fqms/fqms_shim.cpp"), .flags = &.{ "-std=c++17", "-O2", "-DNDEBUG" } });
    mod.addIncludePath(b.path("lib/fqms"));
}

/// The embedded files the engine imports by name on every platform: MLX's steel NAX tile headers, the opencode2
/// plugin and the agent skill.
pub const Embedded = struct {
    mlx_steel_sources: *std.Build.Module,
    opencode2_plugin: *std.Build.Module,
    agent_skills: *std.Build.Module,

    pub fn init(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) Embedded {
        return .{
            .mlx_steel_sources = b.createModule(.{ .root_source_file = b.path("lib/mlx_steel_sources.zig"), .target = target, .optimize = optimize }),
            .opencode2_plugin = b.createModule(.{ .root_source_file = b.path("lib/opencode2_plugin.zig"), .target = target, .optimize = optimize }),
            .agent_skills = b.createModule(.{ .root_source_file = b.path("skills/agent_skills.zig"), .target = target, .optimize = optimize }),
        };
    }
};

/// Translates a single C header into an importable module (`@import("name")`
/// at the call site) via `addTranslateC`, replacing an inline `@cImport` —
/// removed as a language builtin in 0.17.0-dev.
pub fn addCHeaderModule(
    b: *std.Build,
    header_path: std.Build.LazyPath,
    include_dir: std.Build.LazyPath,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    ios_sdk: []const u8,
) *std.Build.Module {
    const translate = b.addTranslateC(.{
        .root_source_file = header_path,
        .target = target,
        .optimize = optimize,
    });
    translate.addIncludePath(include_dir);
    // iOS cross-compile: addTranslateC (0.17's @cImport replacement) does NOT
    // inherit the parent module's SDK include search the way inline @cImport
    // used to, so a header that pulls in <stdio.h>/<inttypes.h> can't find the
    // Apple libc and translation fails. Wire the SDK's system include here.
    // Host builds pass "" (their toolchain resolves the system headers).
    if (ios_sdk.len > 0)
        translate.addSystemIncludePath(.{ .cwd_relative = b.fmt("{s}/usr/include", .{ios_sdk}) });
    return translate.createModule();
}

pub fn addDs4Sources(b: *std.Build, module: *std.Build.Module) void {
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
    // Two-machine tensor parallelism + multi-GPU layer placement (pin 9139e2a):
    // ds4.c references ds4_tp_* and ds4_compute_layer_placement/ds4_layer_pack_print
    // unconditionally, so both TUs must link even though we never enable TP.
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_tp.c"), .flags = c_flags });
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_layer_pack.c"), .flags = c_flags });
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_image.c"), .flags = c_flags });
    module.addCSourceFile(.{ .file = b.path("lib/ds4/ds4_engram.c"), .flags = c_flags });
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

/// ds4 on its CUDA backend: scripts/build-ds4-linux.sh links upstream's library objects
/// (nvcc) into lib/ds4-linux/libds4.so, with its GNU C++ runtime and CUDA libraries inside.
pub fn addDs4LinuxLib(b: *std.Build, module: *std.Build.Module) void {
    module.addIncludePath(b.path("lib/ds4"));
    module.addLibraryPath(b.path("lib/ds4-linux"));
    module.linkSystemLibrary("ds4", .{ .use_pkg_config = .no });
    module.addCSourceFile(.{ .file = b.path("src/ds4_layout_check.c"), .flags = &.{"-std=c99"} });
}

/// ANE prefill offload sources (lib/ane): the private-framework bridge and
/// the per-layer MLP program builder, both ARC objc. Runtime-probed —
/// compiling them in costs nothing on machines without the framework.
pub fn addAneSources(b: *std.Build, module: *std.Build.Module) void {
    const objc_flags = &[_][]const u8{
        "-O3",
        "-fobjc-arc",
        "-Wno-deprecated-declarations",
    };
    module.addCSourceFile(.{ .file = b.path("lib/ane/ane_bridge.m"), .flags = objc_flags });
    module.addCSourceFile(.{ .file = b.path("lib/ane/ane_mlp.m"), .flags = objc_flags });
    module.addIncludePath(b.path("lib/ane"));
}

pub fn addLlamaLib(b: *std.Build, module: *std.Build.Module) void {
    // Link the prebuilt libllama staged by scripts/fetch-llama.sh. The dylib's
    // install-name is @rpath/libllama.dylib; we add an rpath to its build-tree
    // location so `zig build run` / unit tests resolve it in dev. The app bundle
    // and CLI tarball rewrite that reference to @executable_path/... and re-sign
    // with the Developer ID (see release.yml / app/build.sh).
    // Windows: the release's DLLs through import libraries, staged by
    // `LLAMA_TARGET=windows scripts/fetch-llama.sh`; no rpaths (DLLs ship beside the exe).
    const os = module.resolved_target.?.result.os.tag;
    const stage = if (os == .windows) "lib/llama-win" else "lib/llama";
    module.addIncludePath(b.path(b.fmt("{s}/include", .{stage})));
    module.addLibraryPath(b.path(b.fmt("{s}/lib", .{stage})));
    // use_pkg_config = .no: a Homebrew `llama.cpp` install ships a llama.pc that
    // would otherwise hijack this link (pulling in /opt/homebrew's version + its
    // separate libggml). We want exactly the pinned dylib staged in lib/llama/lib.
    module.linkSystemLibrary("llama", .{ .use_pkg_config = .no });
    // The macOS dylib merges ggml in; the Linux and Windows releases keep it separate.
    if (os == .linux or os == .windows) {
        module.linkSystemLibrary("ggml", .{ .use_pkg_config = .no });
        module.linkSystemLibrary("ggml-base", .{ .use_pkg_config = .no });
    }
    // @loader_path resolves against the BINARY's own location at launch,
    // not the launching process's cwd. A bare relative string here (e.g.
    // "lib/llama/lib") gets baked verbatim into LC_RPATH when `zig build`
    // runs with cwd == build root (b.build_root.path is null in that case,
    // so b.path() can't make it absolute) and dyld then resolves that
    // relative string against argv[0]'s cwd, breaking any launch from
    // outside the repo root. An absolute path avoids that but bakes in a
    // machine-specific location, so the build isn't relocatable. This is
    // relative to the binary itself, so it stays correct from any launch
    // cwd and survives copying the whole zig-out + lib tree elsewhere.
    //
    // This module backs two different binaries at two different depths
    // under the build root, so one entry can't serve both: the installed
    // exe lands at zig-out/bin/mlx-serve (@loader_path = zig-out/bin/, 2
    // levels up to root), while `zig build test` runs straight out of
    // .zig-cache/o/<hash>/test (@loader_path = that dir, 3 levels up to
    // root). dyld tries every LC_RPATH entry in order and silently skips
    // ones that don't resolve, so listing both depths here is safe — each
    // binary finds its own and ignores the other.
    if (os != .windows) {
        module.addRPath(.{ .cwd_relative = "@loader_path/../../lib/llama/lib" });
        module.addRPath(.{ .cwd_relative = "@loader_path/../../../lib/llama/lib" });
    }

    // Our clean C shim over llama.h (src/engines/llama/ffi.zig mirrors lib/llama_shim/llama_shim.h).
    // C++ for llama.cpp's C++-linked NextN staging calls (MTP).
    module.addIncludePath(b.path("lib/llama_shim"));
    module.addCSourceFile(.{
        .file = b.path("lib/llama_shim/llama_shim.cpp"),
        .flags = &.{ "-O2", "-std=c++17", "-Wno-unused-parameter" },
    });
}

/// Link the self-built mlx + mlx-c staged in lib/mlx by scripts/build-mlx.sh
/// (pinned submodules lib/mlx-src + lib/mlxc-src, deployment target 26.2 so
/// MLX's NAX kernels are compiled in — the Homebrew bottle ships without them
/// and hard-wires is_nax_available() false even on M5). Install names are
/// @rpath/...; the build-tree rpath resolves them in dev, release.yml /
/// app/build.sh rewrite to @executable_path and re-sign for bundles.
/// Guard test: tests/test_mlx_staged_nax.sh.
pub fn addMlxLib(b: *std.Build, module: *std.Build.Module) void {
    module.addIncludePath(b.path("lib/mlx/include"));
    module.addLibraryPath(b.path("lib/mlx/lib"));
    // use_pkg_config = .no: a leftover Homebrew mlx-c must never hijack this
    // link — we want exactly the staged NAX-enabled pair (same class as the
    // llama.pc hijack above).
    module.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });
    // See addLlamaLib above: @loader_path is relative to the binary itself,
    // so this stays correct regardless of the launching process's cwd and
    // stays relocatable across machines. Two entries for the same reason —
    // the installed exe and the `zig build test` binary sit at different
    // depths under the build root.
    module.addRPath(.{ .cwd_relative = "@loader_path/../../lib/mlx/lib" });
    module.addRPath(.{ .cwd_relative = "@loader_path/../../../lib/mlx/lib" });
}

/// `src/sdk/quant/exl3/sushi/`: Sushi's EXL3 + STQ module (`sushi_exl3`), kept as upstream ships it. It reaches
/// mlx, log and io_util through `mlx_host`, so the host root exposes them.
pub fn addExl3Module(b: *std.Build, host: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const exl3 = b.createModule(.{
        .root_source_file = b.path("src/sdk/quant/exl3/sushi/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "mlx_host", .module = host }},
    });
    host.addImport("sushi_exl3", exl3);
    return exl3;
}

/// mlx-stream (absorbed from davidtai/mlx-stream): the streamed packs' archs (`src/plugins/<model>/stream/`), their
/// EXL3 lanes (`src/sdk/quant/exl3/stream/`) and the expert streamer (`src/sdk/weights/streamed/`), one module rooted at
/// `src/mlx_stream.zig` (`src/engines/mlx/modules/stream_glue.zig` binds it). Its contract types are its own `sdk` module;
/// both reach the host through `mlx_host`.
pub fn addMlxStreamModule(b: *std.Build, host: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    host.addImport("mlx_stream", mlxStreamPkg(b, host, "src/mlx_stream.zig", target, optimize, false, true));
}

pub fn mlxStreamPkg(b: *std.Build, host: *std.Build.Module, root: []const u8, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, inject: bool, c_sources: bool) *std.Build.Module {
    const sdk = b.createModule(.{
        .root_source_file = b.path("src/sdk/weights/streamed/sdk/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "mlx_host", .module = host }},
    });
    const m = b.createModule(.{
        .root_source_file = b.path(root),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "mlx_host", .module = host }, .{ .name = "sdk", .module = sdk } },
    });
    // The read pool (pthreads, pread into slot rows), its MTLSharedEvent signal (non-ARC objc), and the MLX
    // event / alloc shims, which include the staged MLX's private headers and link libmlx.
    const csrc = "src/sdk/weights/streamed/csrc";
    // A profile server build (`-Dplugin-profile=true`, never the tests) also compiles the decode profile's command-buffer
    // timeline shims and the read pool's event-signal log (`-DQ3LD_EVSIG`), which the plugin's profile probes link.
    const profile = !inject and pluginProfile(b);
    const c: []const []const u8 = if (inject)
        &.{ "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread", "-DQ3LD_INJECT" }
    else if (profile)
        &.{ "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread", "-DQ3LD_EVSIG" }
    else
        &.{ "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread" };
    const cxx: []const []const u8 = &.{ "-std=c++20", "-O2", "-D_METAL_", "-DACCELERATE_NEW_LAPACK", "-fno-sanitize=all", "-Wall", "-Wno-unused-parameter", "-Wno-deprecated-declarations" };
    if (c_sources) m.addCSourceFile(.{ .file = b.path(b.pathJoin(&.{ csrc, "q3_lookahead4_exl3.c" })), .flags = c });
    if (c_sources) m.addCSourceFile(.{ .file = b.path(b.pathJoin(&.{ csrc, "q3_event_shim.mm" })), .flags = &.{ "-O2", "-Wall", "-Wextra", "-Werror", "-fno-objc-arc" } });
    if (c_sources) m.addCSourceFile(.{ .file = b.path(b.pathJoin(&.{ csrc, "mlx_event_shim.cpp" })), .flags = cxx });
    if (c_sources) m.addCSourceFile(.{ .file = b.path(b.pathJoin(&.{ csrc, "mlx_alloc_shim.cpp" })), .flags = cxx });
    if (profile) {
        const objc: []const []const u8 = &.{ "-O2", "-Wall", "-Wextra", "-Werror", "-fno-objc-arc" };
        if (c_sources) m.addCSourceFile(.{ .file = b.path(b.pathJoin(&.{ csrc, "dsv41_cb_timeline.mm" })), .flags = objc });
        if (c_sources) m.addCSourceFile(.{ .file = b.path(b.pathJoin(&.{ csrc, "dsv41_newbuffer_count.mm" })), .flags = objc });
        if (c_sources) m.addCSourceFile(.{ .file = b.path(b.pathJoin(&.{ csrc, "dsv41_tl_mlx.cpp" })), .flags = cxx });
    }
    m.addIncludePath(b.path(csrc));
    m.addIncludePath(b.path("lib/mlx/include"));
    m.addIncludePath(b.path("lib/mlx/include/metal_cpp"));
    m.addIncludePath(b.path("lib/mlxc-src"));
    m.linkSystemLibrary("mlx", .{ .use_pkg_config = .no });
    // The shims make the server load @rpath/libmlx.dylib itself: the app bundle keeps it in Contents/Frameworks/, the
    // CLI zip in lib/ beside the binary.
    m.addRPath(.{ .cwd_relative = "@executable_path/../Frameworks" });
    m.addRPath(.{ .cwd_relative = "@executable_path/lib" });
    return m;
}

/// `-Dplugin-profile`: compile the plugin's profile probes and their shims in; off in every served build. `b.option`
/// may be declared once; the graphs share this answer.
var plugin_profile_opt: ?bool = null;
pub fn pluginProfile(b: *std.Build) bool {
    if (plugin_profile_opt == null) plugin_profile_opt = b.option(bool, "plugin-profile", "Compile the mlx-stream plugin's profile probes in (profile builds only)") orelse false;
    return plugin_profile_opt.?;
}
