//! MLX allocator cache limit

const std = @import("std");
const env = @import("../env.zig");
const mlx = @import("ffi.zig");
const log = @import("../sdk/util/log.zig");
const metrics = @import("../sdk/util/system.zig");
const testing = std.testing;

/// PURE (unit-testable): the cap to put on MLX's reclaimable buffer pool for a
/// machine with `total_ram` bytes of physical memory. 0 when the RAM query
/// failed — never clamp on bad data.
///
/// MLX's own default is `min(1.5 x working_set, 0.95 x RAM)` with a GC limit at
/// `0.95 x working_set` (`backend/metal/allocator.cpp`) — ~121 GB / ~91 GB on a
/// 128 GB Mac, i.e. it does not trim until the machine is already dead. Freed
/// buffers are parked in that pool instead of going back to the OS, so anything
/// that frees never-repeating sizes in a loop (KV growth, decode transients)
/// grows the process footprint without moving `active_bytes` at all. Issue #110:
/// 81.4 GB in Activity Monitor against 19.6 GB in the panel.
///
/// RAM/16 is big enough for the step-to-step buffer reuse the pool exists for
/// (8 GB on a 128 GB Mac) and small enough that it can never be the footprint.
/// The 2 GB floor keeps 8/16 GB machines from thrashing the allocator.
pub fn mlxCacheLimitBytes(total_ram: u64) u64 {
    if (total_ram == 0) return 0;
    const GB: u64 = 1 << 30;
    return @max(2 * GB, @min(8 * GB, total_ram / 16));
}

/// PURE: resolve the cap from `MLX_SERVE_CACHE_LIMIT` (bytes) over the
/// RAM-proportional default. `0` means "leave MLX's default alone" — the
/// same-boot A/B off-switch. Anything unparseable falls through to the default
/// rather than silently disabling the cap.
pub fn mlxCacheLimitFromEnv(raw: ?[]const u8, total_ram: u64) u64 {
    if (raw) |v| {
        const trimmed = std.mem.trim(u8, v, " \t\r\n");
        if (std.fmt.parseInt(u64, trimmed, 10) catch null) |n| return n;
    }
    return mlxCacheLimitBytes(total_ram);
}

/// Impure wrapper: apply the cap to the live MLX allocator. Called ONCE from
/// `main()`, above every subcommand branch — not per serve path, because a
/// hand-rolled per-path config is exactly how `runHeadlessServe` (the mode the
/// app always launches) came to silently eat the `--pld*` flags.
///
/// Never RAISES a tighter existing cap: `scheduler.runGenRequests` drops the pool
/// to 1 GB on small-RAM machines during media gen, and iOS boots at 384 MB.
pub fn applyMlxCacheLimit() void {
    const cap = mlxCacheLimitFromEnv(env.span(.MLX_SERVE_CACHE_LIMIT), metrics.getTotalMemBytes());
    if (cap == 0) return;
    var prev: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev, @intCast(cap));
    if (prev < cap) {
        var tmp: usize = 0;
        _ = mlx.mlx_set_cache_limit(&tmp, prev);
        return;
    }
    log.info("[mem] MLX buffer-pool cap {d} MB (was {d} MB)\n", .{ cap >> 20, prev >> 20 });
}

// ── tests ──

test "mlxCacheLimitBytes: RAM-proportional cap, 2 GB floor, 8 GB ceiling" {
    const GB: u64 = 1 << 30;
    // MLX's own default is min(1.5 x working_set, 0.95 x RAM) — ~121 GB on a
    // 128 GB Mac (`backend/metal/allocator.cpp`), i.e. it will not trim until
    // the machine is already dead. RAM/16 keeps the pool big enough for
    // step-to-step buffer reuse without letting it become the footprint.
    try testing.expectEqual(8 * GB, mlxCacheLimitBytes(128 * GB));
    try testing.expectEqual(4 * GB, mlxCacheLimitBytes(64 * GB));
    try testing.expectEqual(2 * GB, mlxCacheLimitBytes(32 * GB)); // 2 GB exactly
    try testing.expectEqual(2 * GB, mlxCacheLimitBytes(16 * GB)); // floor
    try testing.expectEqual(2 * GB, mlxCacheLimitBytes(8 * GB)); // floor
    // Ceiling holds above 128 GB (Mac Studio / M3 Ultra 512 GB).
    try testing.expectEqual(8 * GB, mlxCacheLimitBytes(512 * GB));
    // A failed `hw.memsize` read is 0 — never clamp on bad data.
    try testing.expectEqual(@as(u64, 0), mlxCacheLimitBytes(0));
}

test "mlxCacheLimitFromEnv: explicit bytes win, 0 disables, garbage falls through" {
    const GB: u64 = 1 << 30;
    // The A/B off-switch: MLX_SERVE_CACHE_LIMIT=0 leaves MLX's default in
    // place so the pre-#110 behavior stays reachable in a same-boot A/B.
    try testing.expectEqual(@as(u64, 0), mlxCacheLimitFromEnv("0", 128 * GB));
    try testing.expectEqual(@as(u64, 3 * GB), mlxCacheLimitFromEnv("3221225472", 128 * GB));
    // Unset or unparseable → the RAM-proportional default.
    try testing.expectEqual(8 * GB, mlxCacheLimitFromEnv(null, 128 * GB));
    try testing.expectEqual(8 * GB, mlxCacheLimitFromEnv("lots", 128 * GB));
    try testing.expectEqual(8 * GB, mlxCacheLimitFromEnv("", 128 * GB));
}
