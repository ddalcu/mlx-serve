//! Small helpers around `std.Io.Timestamp` to ease the Zig 0.15 -> 0.16 migration.
//!
//! In 0.16 the legacy `std.time.Timer`, `std.time.timestamp()`, and
//! `std.time.milliTimestamp()` are gone — all clocks live under `std.Io` and
//! require an `Io` parameter. These helpers wrap those calls so the rest of the
//! codebase reads naturally.

const std = @import("std");

/// `io` over this unit's own vtable. Each compile unit numbers Zig errors on its own and an `Io` returns the numbers of
/// the unit that built its vtable, so an `Io` handed in by another unit must be rebuilt before its errors are compared
/// or named (the host's `FileNotFound` read here as `HostUnreachable`). Every `Io` the host hands out is a `Threaded`.
pub fn local(io: std.Io) std.Io {
    if (comptime !@hasDecl(@import("root"), "mlx_serve_unit")) return io;
    const t: *std.Io.Threaded = @ptrCast(@alignCast(io.userdata.?));
    return t.io();
}

/// Seconds since the Unix epoch (wall-clock).
pub fn nowSecs(io: std.Io) i64 {
    return std.Io.Timestamp.now(io, .real).toSeconds();
}

/// Milliseconds since the Unix epoch (wall-clock).
pub fn nowMs(io: std.Io) i64 {
    return std.Io.Timestamp.now(io, .real).toMilliseconds();
}

/// Milliseconds on the MONOTONIC `boot` clock (arbitrary epoch — only
/// differences are meaningful). Use for deadlines/intervals, never for
/// timestamps a client sees: an NTP step or a manual clock change must not be
/// able to stall (or spam) a timer. `.boot` counts across pmset sleep, so a
/// deadline that expired while the lid was shut fires immediately on wake.
pub fn nowMsMonotonic(io: std.Io) i64 {
    return std.Io.Timestamp.now(io, .boot).toMilliseconds();
}

/// `nowMsMonotonic` in microseconds.
pub fn nowUsMonotonic(io: std.Io) i64 {
    return std.Io.Timestamp.now(io, .boot).toMicroseconds();
}

/// `nowMsMonotonic` as a value, for code that takes its clock as a parameter so a test can step it.
pub const MonotonicClock = struct {
    io: std.Io,
    pub fn nowMs(self: MonotonicClock) i64 {
        return nowMsMonotonic(self.io);
    }
};

/// Drop-in replacement for `std.time.Timer` — uses the `boot` clock, which
/// counts wall-time across pmset sleep (vs `.awake`, which stops during
/// suspend). Important for long-running stopwatches that span a lid close.
/// Resolution is set by the `Io` implementation.
pub const Stopwatch = struct {
    io: std.Io,
    started_at: std.Io.Timestamp,

    pub fn init(io: std.Io) Stopwatch {
        return .{ .io = io, .started_at = std.Io.Timestamp.now(io, .boot) };
    }

    /// Nanoseconds since `init` (or last `reset`).
    pub fn read(s: Stopwatch) u64 {
        return @intCast(s.started_at.untilNow(s.io, .boot).nanoseconds);
    }

    /// Reset the start point to "now".
    pub fn reset(s: *Stopwatch) void {
        s.started_at = std.Io.Timestamp.now(s.io, .boot);
    }
};
