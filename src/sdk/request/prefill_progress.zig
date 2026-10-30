//! Live prefill progress one streaming request reports to its client. The engine writes it (once at
//! prefill start, then once per chunk) and the connection thread reads it between chunks, so both
//! sides share the layout by importing this type; the pointer lives in the host's `Stream`.
const std = @import("std");

pub const PrefillProgress = extern struct {
    /// Absolute prompt position forwarded so far: the restored prefix plus the tail's chunks.
    pos: std.atomic.Value(u32) = .init(0),
    /// The restored prefix this prefill reuses: the tokens the prompt skips.
    cached: std.atomic.Value(u32) = .init(0),
    /// The whole prompt as the engine counts it. 0 until the engine starts the prefill.
    total: std.atomic.Value(u32) = .init(0),

    pub fn snapshot(self: *const PrefillProgress) Snapshot {
        return .{
            .pos = self.pos.load(.acquire),
            .cached = self.cached.load(.acquire),
            .total = self.total.load(.acquire),
        };
    }
};

pub const Snapshot = struct { pos: u32, cached: u32, total: u32 };

// ── tests ──

const testing = std.testing;

test "prefill progress: a snapshot reads the three counters; zero total means the engine has not started" {
    var p: PrefillProgress = .{};
    const idle = p.snapshot();
    try testing.expectEqual(@as(u32, 0), idle.total);
    p.cached.store(30000, .release);
    p.pos.store(31744, .release);
    p.total.store(@intCast(32768), .release);
    const s = p.snapshot();
    try testing.expectEqual(@as(u32, 31744), s.pos);
    try testing.expectEqual(@as(u32, 30000), s.cached);
    try testing.expectEqual(@as(u32, 32768), s.total);
}
