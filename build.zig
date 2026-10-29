//! The build's steps, in the order `zig build --help` lists them; each graph's logic lives in `build/`.

const std = @import("std");
const builtin = @import("builtin");
const config = @import("build/config.zig");
const ios = @import("build/ios.zig");
const kit = @import("build/kit.zig");
const lint = @import("build/lint.zig");
const linux = @import("build/linux.zig");
const macos = @import("build/macos.zig");
const plugins = @import("build/plugins.zig");
const standalone = @import("build/standalone.zig");

comptime {
    // 0.16.0's bundled libc++ fails to compile against the macOS 27 SDK
    // (`use of undeclared identifier 'INFINITY'` in its vendored <random>);
    // fixed in 0.17.0, which is why the floor moved.
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 17) {
        @compileError(std.fmt.comptimePrint(
            "mlx-serve requires Zig 0.17 (have {d}.{d}.{d}). Run ./scripts/fetch-zig.sh or grab it from https://ziglang.org/download/.",
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
    // Host-gated: on a Linux box a semver 26.2 min would be read as a glibc
    // floor (glibc has no 26.2) and break the native Linux graph.
    const target = b.standardTargetOptions(.{
        .default_target = if (builtin.os.tag == .macos) .{
            .os_version_min = .{ .semver = .{ .major = 26, .minor = 2, .patch = 0 } },
        } else .{},
    });
    const optimize = b.standardOptimizeOption(.{});

    // `-Dunits` (the macOS default): the host, the MLX engine and every plugin group compile as their own objects, in
    // parallel, none of them needing the whole tree's compiler memory. `-Dunits=false` builds one unit.
    const units = b.option(bool, "units", "Compile the host, the engine and each plugin group as their own units, in parallel") orelse
        (builtin.os.tag == .macos);
    // A plugin registers by its folder: the scan writes src/plugins/registry.zig, every graph compiles it.
    const plugin_names = plugins.generate(b);

    // `lint`, `lint-test`, `unit-closure`.
    lint.add(b);
    // The macOS SDK and webp; on a macOS target, webp's version and the staged mlx are checked here, for every step.
    const host = config.Host.init(b, target);
    // `preview-test`: the hermetic JPEG / Latent2RGB tests, the only graph a Linux Cloud Agent can build. Native query:
    // it does not inherit the macOS 26.2 minos default_target.
    standalone.addPreviewTest(b, b.resolveTargetQuery(.{}));

    // `install`, `run`, `test` for Linux and Windows, gated on the TARGET so the Linux exe can be built on any host
    // zig can target; the host gate below still guards the macOS graph.
    if (target.result.os.tag == .linux or target.result.os.tag == .windows) return linux.add(b, target, optimize);
    if (builtin.os.tag != .macos) return;

    const graph: macos.Graph = .init(b, target, optimize, host, units);
    // `install`, `units`, `units-check`, `run`.
    macos.addServer(&graph, plugin_names);
    // `test-build`, `test`.
    const tests = macos.addUnitTests(&graph);
    // `kit`, and the kit's example plugins built from it as dylibs the unit tests load.
    kit.add(b, graph.version, tests.runs);
    // `exl3-test`, `mlx-stream-test`, `mlx-stream-conformance`.
    macos.addPluginTests(&graph, tests);
    // `check`.
    macos.addCheck(&graph);
    // `vz-agent`, `vz-agent-host`; its host-side tests join `test`.
    standalone.addVzAgent(b, target, optimize, tests.step);
    // `ios-lib`, `ios-lib-sim`.
    ios.add(b, graph.version, host.brew_prefix);
}
