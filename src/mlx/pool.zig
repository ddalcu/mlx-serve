//! MLX's buffer-pool cap, moved for one scope and put back.

const std = @import("std");
const mlx = @import("ffi.zig");
const rule = @import("../sdk/media/pool.zig");

/// The cap a media job runs under: lowered on a small-RAM Mac (`rule.SMALL_RAM_CAP`), else left alone.
pub fn forMediaJob(total_ram: u64) Cap {
    return if (rule.smallRam(total_ram)) Cap.lower(rule.SMALL_RAM_CAP) else .{};
}

pub const Cap = struct {
    prev: usize = 0,
    moved: bool = false,
    raised: bool = false,

    /// Lowers the cap to `bytes`; a tighter one stays.
    pub fn lower(bytes: usize) Cap {
        return set(bytes, false);
    }

    /// Raises the cap to `bytes`; a looser one stays.
    pub fn raise(bytes: usize) Cap {
        return set(bytes, true);
    }

    fn set(bytes: usize, up: bool) Cap {
        var prev: usize = 0;
        if (mlx.mlx_set_cache_limit(&prev, bytes) != 0) return .{};
        if (if (up) prev >= bytes else prev <= bytes) {
            var t: usize = 0;
            _ = mlx.mlx_set_cache_limit(&t, prev);
            return .{};
        }
        return .{ .prev = prev, .moved = true, .raised = up };
    }

    /// Puts the cap back; a raised pool's buffers go back to the OS.
    pub fn restore(self: Cap) void {
        if (!self.moved) return;
        var t: usize = 0;
        _ = mlx.mlx_set_cache_limit(&t, self.prev);
        if (self.raised) _ = mlx.mlx_clear_cache();
    }
};

// ── tests ──

const testing = std.testing;
const GB: u64 = 1 << 30;

fn capNow() usize {
    var c: usize = 0;
    _ = mlx.mlx_set_cache_limit(&c, 0);
    var t: usize = 0;
    _ = mlx.mlx_set_cache_limit(&t, c);
    return c;
}

fn capSet(v: usize) void {
    var t: usize = 0;
    _ = mlx.mlx_set_cache_limit(&t, v);
}

test "pool cap: lower and raise move the cap one way only, restore puts it back" {
    const orig = capNow();
    defer capSet(orig);
    capSet(8 << 30);
    const up = Cap.raise(32 << 30);
    try testing.expectEqual(@as(usize, 32 << 30), capNow());
    up.restore();
    try testing.expectEqual(@as(usize, 8 << 30), capNow());
    const kept = Cap.raise(4 << 30);
    try testing.expectEqual(@as(usize, 8 << 30), capNow());
    kept.restore();
    const down = Cap.lower(1 << 30);
    try testing.expectEqual(@as(usize, 1 << 30), capNow());
    down.restore();
    try testing.expectEqual(@as(usize, 8 << 30), capNow());
    const tighter = Cap.lower(16 << 30);
    try testing.expectEqual(@as(usize, 8 << 30), capNow());
    tighter.restore();
    try testing.expectEqual(@as(usize, 8 << 30), capNow());
}

test "pool cap: a lift inside a small-RAM media job leaves the job's cap" {
    const orig = capNow();
    defer capSet(orig);
    capSet(2 << 30);
    const job = forMediaJob(16 * GB);
    try testing.expectEqual(@as(usize, rule.SMALL_RAM_CAP), capNow());
    const lift = Cap.raise(rule.liftBytes(30 * GB, 16 * GB, 30 * GB));
    try testing.expectEqual(@as(usize, rule.SMALL_RAM_CAP), capNow());
    lift.restore();
    job.restore();
    try testing.expectEqual(@as(usize, 2 << 30), capNow());
    const roomy = forMediaJob(64 * GB);
    try testing.expectEqual(@as(usize, 2 << 30), capNow());
    roomy.restore();
}

test "pool cap: restoring a raised cap returns its buffers to the OS" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const orig = capNow();
    defer capSet(orig);
    capSet(0);
    _ = mlx.mlx_clear_cache();
    const up = Cap.raise(1 << 30);
    const s = mlx.gpuStream();
    var x = mlx.mlx_array_new();
    _ = mlx.mlx_arange(&x, 0, 16 << 20, 1, mlx.mlx_dtype.float32, s);
    _ = mlx.mlx_array_eval(x);
    _ = mlx.mlx_array_free(x);
    _ = mlx.mlx_synchronize(s);
    var cached: usize = 0;
    _ = mlx.mlx_get_cache_memory(&cached);
    try testing.expect(cached >= 64 << 20);
    up.restore();
    _ = mlx.mlx_get_cache_memory(&cached);
    try testing.expectEqual(@as(usize, 0), cached);
    try testing.expectEqual(@as(usize, 0), capNow());
}
