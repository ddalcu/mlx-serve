//! When a media job may hold MLX's buffer pool above or below the server's cap. MLX-free, so a bill prices the same
//! pool the job runs under (`mlx/pool.zig` moves the cap).

const std = @import("std");

/// A media job on a Mac with at most this much RAM caps the pool at `SMALL_RAM_CAP`: a denoise plus a VAE pass would
/// otherwise park GBs of one-off transients against a ~12 GB Metal working set. 1 GB still covers step-to-step reuse.
pub const SMALL_RAM_BYTES: u64 = 17 << 30;
pub const SMALL_RAM_CAP: usize = 1 << 30;

pub fn smallRam(total_ram: u64) bool {
    return total_ram > 0 and total_ram <= SMALL_RAM_BYTES;
}

/// What a stage that reuses `want` bytes between passes may raise the pool to: nothing where a media job caps it for
/// small RAM (or RAM is unknown), and never more than an eighth of RAM.
pub fn liftBytes(want: u64, total_ram: u64) u64 {
    if (total_ram == 0 or smallRam(total_ram)) return 0;
    return @min(want, total_ram / 8);
}

test "media pool lift: none on small or unknown RAM, at most an eighth of RAM" {
    const GB: u64 = 1 << 30;
    try std.testing.expectEqual(@as(u64, 0), liftBytes(30 * GB, 16 * GB));
    try std.testing.expectEqual(@as(u64, 0), liftBytes(30 * GB, 0));
    try std.testing.expectEqual(@as(u64, 3 * GB), liftBytes(30 * GB, 24 * GB));
    try std.testing.expectEqual(@as(u64, 16 * GB), liftBytes(30 * GB, 128 * GB));
    try std.testing.expectEqual(@as(u64, 30 * GB), liftBytes(30 * GB, 256 * GB));
}
