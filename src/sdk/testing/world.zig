//! The test world: the one owner of what a unit test starts from. The test runner enters it before every test
//! (`build/test_runner.zig` through `msv_test_enter` in `src/tests.zig`), so whatever a test changes ends with it:
//! - every registered variable at its value when the process started (`env.worldRestore`), with
//!   `MLX_SERVE_TEST_FLIP`'s lever at its other arm when that is set (the flip proof, `scripts/flip_proof.py`);
//! - MLX's buffer-pool and wired limits as at start;
//! - the machine a machine-keyed table reads (`device.machineId`) at `reference`, whatever box runs the tests;
//! - the latched state `src/tests.zig` lists, at its initial value.
//! A test leaves the reference only through `with`, and reads the GPU's capabilities through `nax`, `tf32` and
//! `gpuGen`, so a test whose coverage depends on the machine says so where it does (lint rule machine-read-in-test).

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const device = @import("../../mlx/device.zig");

pub const Machine = device.MachineId;

/// The machine every unit test plans for: the M5 Ultra with 512 GiB, the box the defaults are tuned on.
pub const reference: Machine = .{ .chip = "Apple M5 Ultra", .mem = 512 << 30 };
/// A machine with no measured knee (the memory-sized widths).
pub const no_knee: Machine = .{ .chip = "Apple M5 Max", .mem = 128 << 30 };

var cache_limit: usize = 0;
var wired_limit: usize = 0;
var limits_known = false;

/// Once, before the first test: apply `MLX_SERVE_TEST_FLIP`, record the environment and MLX's limits.
pub fn processInit() void {
    if (env.span(.MLX_SERVE_TEST_FLIP)) |spec| {
        const f = env.flipArm(spec) catch |e| std.debug.panic("MLX_SERVE_TEST_FLIP={s}: {t} (a registered name; a non-flag needs NAME=value)", .{ spec, e });
        _ = env.setForeign(f.name.ptr, f.value.ptr);
        std.debug.print("[test world] flip: {s}={s}\n", .{ f.name, f.value });
    }
    env.worldSnapshot();
    var buf: [2048]u8 = undefined;
    const set = env.worldLevers(&buf);
    if (set.len > 0) std.debug.print("[test world] every test runs with the levers this process started with: {s}\n", .{set});
    if (!mlx.noGpuBackend()) {
        var tmp: usize = 0;
        _ = mlx.mlx_set_cache_limit(&cache_limit, std.math.maxInt(usize));
        _ = mlx.mlx_set_cache_limit(&tmp, cache_limit);
        _ = mlx.mlx_set_wired_limit(&wired_limit, 0);
        _ = mlx.mlx_set_wired_limit(&tmp, wired_limit);
        limits_known = true;
    }
}

/// Before each test: the environment, MLX's limits and the machine back at the reference.
pub fn enter() void {
    _ = env.worldRestore();
    if (limits_known) {
        var prev: usize = 0;
        _ = mlx.mlx_set_cache_limit(&prev, cache_limit);
        _ = mlx.mlx_set_wired_limit(&prev, wired_limit);
    }
    device.machine_id_override = reference;
    device.device_gen_override = null;
}

/// What a test declares in place of the reference; the next `enter` undoes it.
pub const With = struct { machine: ?Machine = null };

pub fn with(w: With) void {
    if (w.machine) |m| device.machine_id_override = m;
}

/// The GPU's NAX, as a test whose arms or skip depend on it reads it.
pub fn nax() bool {
    return device.naxAvailable();
}

/// Whether MLX rounds f32 matmuls to TF32 here, as a test whose bound depends on it reads it.
pub fn tf32() bool {
    return device.tf32Matmul();
}

/// The GPU generation, as a test whose arms or skip depend on it reads it.
pub fn gpuGen() u32 {
    return device.deviceGeneration().gen;
}

test "test world: enter puts a variable back, the machine at the reference, and a declared machine lasts until then" {
    const t = std.testing;
    const was: ?[]u8 = if (env.get(.MLX_SERVE_DSV41_PF_PASS_CEIL)) |v| try t.allocator.dupe(u8, std.mem.span(v)) else null;
    defer if (was) |w| t.allocator.free(w);
    env.set(.MLX_SERVE_DSV41_PF_PASS_CEIL, "7");
    with(.{ .machine = no_knee });
    try t.expectEqualStrings(no_knee.chip, device.machineId().chip);
    enter();
    const now = env.get(.MLX_SERVE_DSV41_PF_PASS_CEIL);
    if (was) |w| try t.expectEqualStrings(w, std.mem.span(now.?)) else try t.expect(now == null);
    try t.expectEqualStrings(reference.chip, device.machineId().chip);
    try t.expectEqual(reference.mem, device.machineId().mem);
}

test "test world: a flip names a flag's other arm, takes an explicit value, and refuses what it cannot flip" {
    const t = std.testing;
    const on = try env.flipArm("PF_PASS_CEIL");
    try t.expectEqualStrings("MLX_SERVE_DSV41_PF_PASS_CEIL", on.name);
    try t.expectEqualStrings("0", on.value);
    try t.expectEqualStrings("1", (try env.flipArm("FLASH_PSUM")).value);
    try t.expectEqualStrings("3", (try env.flipArm("PF_PASS_CEIL=3")).value);
    try t.expectError(error.UnknownFlip, env.flipArm("NO_SUCH_LEVER"));
    try t.expectError(error.NoFlipArm, env.flipArm("MLX_SERVE_CACHE_LIMIT"));
}
