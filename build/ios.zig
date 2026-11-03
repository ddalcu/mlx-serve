//! iOS on-device engine: a static library (libmlxserve.a) linking the
//! MLX-only decode path. ds4 + llama.cpp are stubbed (build_options.ios =
//! true). Two slices: `zig build ios-lib` (device, arm64-iphoneos) and
//! `zig build ios-lib-sim` (arm64 iphonesimulator). Driven by the iPhone
//! app project's build scripts (../mlx-iphone/scripts/build-zig-ios.sh),
//! which supply the matching --sysroot and copy the artifact out of
//! zig-out/ios/<sdk>/lib. `-Dios-include=<dir>` points at the iOS dist's
//! include dir for third-party headers (webp); defaults to Homebrew's,
//! whose versions are pinned identical by verifyBrewDeps.

const std = @import("std");
const libs = @import("libs.zig");

/// `ios-lib`, `ios-lib-sim`.
pub fn add(b: *std.Build, version: []const u8, brew_prefix: []const u8) void {
    const ios_include = b.option([]const u8, "ios-include", "Include dir for webp/stb headers when cross-compiling the iOS lib") orelse b.fmt("{s}/include", .{brew_prefix});
    addLib(b, version, ios_include, .{ .step = "ios-lib", .abi = .none, .sdk = "iphoneos" });
    addLib(b, version, ios_include, .{ .step = "ios-lib-sim", .abi = .simulator, .sdk = "iphonesimulator" });
}

const Slice = struct { step: []const u8, abi: std.Target.Abi, sdk: []const u8 };

fn addLib(b: *std.Build, version: []const u8, ios_include: []const u8, slice: Slice) void {
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
    ios_options.addOption(bool, "units", false);
    // Mirror the build options the shared engine sources read (server.zig,
    // scheduler.zig). iOS is sandboxed (no curl/model-pull subprocess), so
    // mas=true; ds4 + llama.cpp are stubbed here, so their version pins are
    // unreported. Without these the iOS lib fails to compile ("options has no
    // member named 'mas'/...").
    ios_options.addOption(bool, "mas", true);
    ios_options.addOption(bool, "ds4_engine", false);
    ios_options.addOption(bool, "llama_engine", false);
    ios_options.addOption(bool, "mlx_stream", false);
    ios_options.addOption(bool, "plugin_profile", false);
    ios_options.addOption([]const u8, "mlx_c_version", "unknown");
    ios_options.addOption([]const u8, "ds4_commit", "unknown");
    ios_options.addOption([]const u8, "llama_tag", "unknown");
    ios_options.addOption([]const u8, "git_sha", "");

    const embedded: libs.Embedded = .init(b, ios_target, .ReleaseFast);
    const mod = b.createModule(.{
        .root_source_file = b.path("src/ios_lib.zig"),
        .target = ios_target,
        .optimize = .ReleaseFast,
        .link_libc = true,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = ios_options.createModule() },
            .{ .name = "mlx_steel_sources", .module = embedded.mlx_steel_sources },
            .{ .name = "opencode2_plugin", .module = embedded.opencode2_plugin },
            .{ .name = "agent_skills", .module = embedded.agent_skills },
        },
    });
    _ = libs.addExl3Module(b, mod, ios_target, .ReleaseFast);

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
    mod.addIncludePath(.{ .cwd_relative = ios_include });
    mod.addImport("jinja_c", libs.addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), ios_target, .ReleaseFast, ios_sdk));
    mod.addImport("stb", libs.addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), ios_target, .ReleaseFast, ios_sdk));
    mod.addImport("webp", libs.addCHeaderModule(b, .{ .cwd_relative = b.fmt("{s}/webp/decode.h", .{ios_include}) }, .{ .cwd_relative = ios_include }, ios_target, .ReleaseFast, ios_sdk));
    libs.addVendoredSources(b, mod);

    const lib = b.addLibrary(.{
        .name = "mlxserve",
        .root_module = mod,
        .linkage = .static,
        .zig_lib_dir = patchedZigLib(b),
    });
    lib.bundle_compiler_rt = true;

    const install = b.addInstallArtifact(lib, .{
        .dest_dir = .{ .override = .{ .custom = b.fmt("ios/{s}/lib", .{slice.sdk}) } },
    });
    const step = b.step(slice.step, b.fmt("Build the iOS engine static lib ({s})", .{slice.sdk}));
    step.dependOn(&install.step);
}

/// Zig 0.17.0's std does not compile for iOS: `Io.Threaded`'s process replace reads `null_file.fd`, a field its iOS
/// `NullFile` lacks. The iOS slices compile against the toolchain's lib dir with iOS moved to the `NullFile` arm that
/// has one (the spawn it serves is never called there); the step fails once a toolchain no longer needs the patch.
fn patchedZigLib(b: *std.Build) std.Build.LazyPath {
    const lib = b.pathJoin(&.{ std.fs.path.dirname(b.graph.zig_exe).?, "lib" });
    const script =
        \\set -e
        \\lib=$1 out=$2 arm='    .wasi, .ios, .tvos, .visionos, .watchos => struct {'
        \\grep -qxF "$arm" "$lib/std/Io/Threaded.zig" || { echo "std's iOS NullFile changed: drop patchedZigLib" >&2; exit 1; }
        \\for e in "$lib"/*; do [ "$(basename "$e")" = std ] || ln -sf "$e" "$out/"; done
        \\rm -rf "$out/std" && cp -R "$lib/std" "$out/std"
        \\sed "s/^    \.wasi, \.ios, \.tvos,/    .wasi, .tvos,/" "$lib/std/Io/Threaded.zig" > "$out/std/Io/Threaded.zig"
    ;
    const patch = b.addSystemCommand(&.{ "sh", "-c", script, "sh", lib });
    const out = patch.addOutputDirectoryArg("zig-lib");
    patch.addFileInput(.{ .cwd_relative = b.pathJoin(&.{ lib, "std/Io/Threaded.zig" }) });
    return out;
}
