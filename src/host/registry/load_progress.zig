//! What a model load is doing, for `GET /v1/load-progress`: the loading thread writes it, any HTTP
//! thread reads it. A GPU load reads weights (bytes, measured as the process's footprint growth: the
//! one meter every arch's loader moves), a Neural Engine load builds programs (counted as each is
//! compiled or restored from the compile cache); both end warming up.

const std = @import("std");

pub const Phase = enum(u8) {
    starting,
    reading_weights,
    ane_programs,
    warming_up,
};

pub const Progress = struct {
    phase: std.atomic.Value(u8) = .init(0),
    done: std.atomic.Value(u64) = .init(0),
    total: std.atomic.Value(u64) = .init(0),
    /// The footprint (MiB) when reading began; while reading, `done` is the growth since.
    base_mb: std.atomic.Value(u32) = .init(0),
    /// A Neural Engine program missed the compile cache (a first load on this Mac).
    compiling: std.atomic.Value(bool) = .init(false),
    started_ms: std.atomic.Value(i64) = .init(0),
    phase_ms: std.atomic.Value(i64) = .init(0),

    /// A new load: no phase measured yet.
    pub fn start(self: *Progress, now_ms: i64) void {
        self.compiling.store(false, .monotonic);
        self.started_ms.store(now_ms, .monotonic);
        self.begin(.starting, 0, now_ms);
    }

    /// Enter `phase` with `total` units (bytes or programs; 0 = no count).
    pub fn begin(self: *Progress, phase: Phase, total: u64, now_ms: i64) void {
        self.done.store(0, .monotonic);
        self.total.store(total, .monotonic);
        self.phase_ms.store(now_ms, .monotonic);
        self.phase.store(@intFromEnum(phase), .release);
    }

    /// Reading weights from a footprint of `base_mb`.
    pub fn beginReading(self: *Progress, total_bytes: u64, base_mb: u32, now_ms: i64) void {
        self.base_mb.store(base_mb, .monotonic);
        self.begin(.reading_weights, total_bytes, now_ms);
    }

    pub fn add(self: *Progress, n: u64) void {
        _ = self.done.fetchAdd(n, .monotonic);
    }

    /// The load as of `now_ms`, with the process footprint at `footprint_mb`.
    pub fn snapshot(self: *const Progress, now_ms: i64, footprint_mb: u32) Snapshot {
        const phase: Phase = @enumFromInt(self.phase.load(.acquire));
        const total = self.total.load(.monotonic);
        // Reading never reports done before it is: the footprint also holds what the load allocates.
        const done = if (phase == .reading_weights)
            @min(@as(u64, footprint_mb -| self.base_mb.load(.monotonic)) << 20, total -| total / 100)
        else
            @min(self.done.load(.monotonic), total);
        const phase_ms = now_ms - self.phase_ms.load(.monotonic);
        return .{
            .phase = phase,
            .done = done,
            .total = total,
            .compiling = self.compiling.load(.monotonic),
            .elapsed_s = @intCast(@max(0, now_ms - self.started_ms.load(.monotonic)) / 1000),
            .eta_s = eta(done, total, phase_ms),
        };
    }
};

pub const Snapshot = struct {
    phase: Phase,
    done: u64,
    total: u64,
    compiling: bool,
    elapsed_s: u64,
    eta_s: ?u64,

    /// `{"phase":…,"done":…,"total":…,"compiling":…,"elapsed_s":…,"eta_s":…|null}`.
    pub fn writeJson(self: Snapshot, w: *std.Io.Writer) !void {
        try w.print("{{\"phase\":\"{s}\",\"done\":{d},\"total\":{d},\"compiling\":{},\"elapsed_s\":{d},\"eta_s\":", .{ @tagName(self.phase), self.done, self.total, self.compiling, self.elapsed_s });
        if (self.eta_s) |e| try w.print("{d}}}", .{e}) else try w.writeAll("null}");
    }
};

/// Seconds left at the rate this phase has run so far; null until there is a rate.
pub fn eta(done: u64, total: u64, phase_ms: i64) ?u64 {
    if (done == 0 or total == 0 or phase_ms <= 0) return null;
    const left = total -| done;
    return @intCast((@as(u128, left) * @as(u128, @intCast(phase_ms))) / (@as(u128, done) * 1000));
}

// ── tests ──

const testing = std.testing;

test "eta: the phase's own rate carried over what is left" {
    try testing.expectEqual(@as(?u64, null), eta(0, 100, 5000));
    try testing.expectEqual(@as(?u64, null), eta(10, 0, 5000));
    // A quarter done in 10 s: 30 s left.
    try testing.expectEqual(@as(?u64, 30), eta(25, 100, 10_000));
    try testing.expectEqual(@as(?u64, 0), eta(100, 100, 10_000));
}

test "reading counts footprint growth, never past 99% of the weights" {
    var p: Progress = .{};
    p.start(1_000);
    p.beginReading(10 << 30, 2_000, 2_000);
    var s = p.snapshot(4_000, 2_000 + 4 * 1024);
    try testing.expectEqual(Phase.reading_weights, s.phase);
    try testing.expectEqual(@as(u64, 4 << 30), s.done);
    // 4 of 10 GiB in 2 s: 3 s left.
    try testing.expectEqual(@as(?u64, 3), s.eta_s);
    s = p.snapshot(9_000, 2_000 + 20 * 1024);
    try testing.expect(s.done < s.total);
    // A footprint that shrank reads as nothing read yet.
    try testing.expectEqual(@as(u64, 0), p.snapshot(9_000, 1_000).done);
}

test "Neural Engine programs count up; a phase change resets the count" {
    var p: Progress = .{};
    p.start(0);
    p.begin(.ane_programs, 28, 0);
    p.add(7);
    p.compiling.store(true, .monotonic);
    const s = p.snapshot(7_000, 0);
    try testing.expectEqual(@as(u64, 7), s.done);
    try testing.expect(s.compiling);
    try testing.expectEqual(@as(?u64, 21), s.eta_s);
    p.begin(.warming_up, 0, 7_000);
    try testing.expectEqual(@as(u64, 0), p.snapshot(8_000, 0).done);
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try p.snapshot(8_000, 0).writeJson(&w);
    try testing.expectEqualStrings("{\"phase\":\"warming_up\",\"done\":0,\"total\":0,\"compiling\":true,\"elapsed_s\":8,\"eta_s\":null}", w.buffered());
}
