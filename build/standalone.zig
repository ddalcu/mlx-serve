//! The graphs that link no MLX: the hermetic preview tests and the Agent Sandbox's guest binary.

const std = @import("std");
const libs = @import("libs.zig");

/// Hermetic Latent2RGB + JPEG tests. No MLX, no Homebrew — the only `zig build`
/// graph that is valid on Linux (issue #208). The ARTIFACT is hermetic, the
/// STEP is not a way around a missing mlx: `verifyBrewDeps` + `verifyMlxStage`
/// run at configure time for every step, so on a Mac `lib/mlx/` must be staged
/// before this builds. UBSan is off for stb (its bit shifts trip the sanitizer).
pub fn addPreviewTest(b: *std.Build, target: std.Build.ResolvedTarget) void {
    const mod = b.createModule(.{
        .root_source_file = b.path("src/preview_tests.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .link_libc = true,
    });
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_write_impl.c"), .flags = libs.stb_write_flags });
    mod.addIncludePath(b.path("lib"));
    const tests = b.addTest(.{
        .name = "preview-test",
        .root_module = mod,
    });
    const run = b.addRunArtifact(tests);
    const step = b.step("preview-test", "Hermetic JPEG / Latent2RGB preview tests (no MLX)");
    step.dependOn(&run.step);
}

/// `zig build vz-agent` → `zig-out/guest/vz-agent` (static aarch64 Linux ELF),
/// plus the host-side unit tests wired into `zig build test`.
///
/// A standalone static aarch64-linux-musl ELF (~200 KB) that the app injects
/// into the guest rootfs before boot, exactly like `/.vz-init`. It serves the
/// vsock exec protocol (`src/host/sandbox/vz_agent.zig`), replacing the hvc1 console shell.
///
/// It is NOT imported by main.zig — it links nothing but libc and never runs
/// on macOS. Its tests do, though: `serveConnection` is OS-agnostic, so the
/// whole request → spawn → stream → exit path is exercised over a socketpair
/// here on the host. Wire them into `zig build test` explicitly, since the
/// main test module's root never reaches this file.
pub fn addVzAgent(
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
            .root_source_file = b.path("src/host/sandbox/vz_agent.zig"),
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
            .root_source_file = b.path("src/host/sandbox/vz_agent.zig"),
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
            .root_source_file = b.path("src/host/sandbox/vz_agent.zig"),
            .target = host_target,
            .optimize = host_optimize,
            .link_libc = true,
        }),
    });
    const host_step = b.step("vz-agent-host", "Build vz-agent natively (unix-socket mode, for interop tests)");
    host_step.dependOn(&b.addInstallArtifact(host_agent, .{}).step);
}
