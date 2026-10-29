//! Leveled logging with an optional persistent file sink.
//!
//! stderr is always written (the macOS app scrapes it into an in-memory
//! rolling buffer). When a file sink is open every emitted line is ALSO
//! appended to disk, so a server that ran hours ago can still be diagnosed —
//! the rolling buffer is gone the moment the app quits, and a crashed or
//! restarted server takes its whole history with it.
//!
//! The sink uses raw libc `write(2)` rather than `std.Io`: log calls come from
//! the accept loop, every connection thread and the inference thread, none of
//! which carry an `Io` handle. A mutex serializes writes + rotation; a failed
//! write is swallowed (logging must never take the server down).

const std = @import("std");
const os_mutex = @import("os_mutex.zig");
const crt = @import("crt.zig");
const rtfmt = @import("rtfmt.zig");
const builtin = @import("builtin");

pub const Level = enum {
    err,
    warn,
    info,
    debug,

    pub fn fromString(s: []const u8) ?Level {
        if (std.mem.eql(u8, s, "error")) return .err;
        if (std.mem.eql(u8, s, "warn")) return .warn;
        if (std.mem.eql(u8, s, "info")) return .info;
        if (std.mem.eql(u8, s, "debug")) return .debug;
        return null;
    }
};

/// Process-wide under `-Dunits` (sdk/util/shared.zig): one copy in the host.
const SinksState = struct {
    current_level: Level = .info,
    /// Read outside the mutex on every log call, so keep it atomic. When false the
    /// sink costs one relaxed load and nothing else.
    sink_active: bool = false,
    sink_bytes: u64 = 0,
    sink_fd: c_int = -1,
    sink_max_bytes: u64 = default_max_bytes,
    /// A pthread mutex, not `std.Io.Mutex`: locking the latter needs an `Io`
    /// handle, and log calls arrive from threads that don't carry one.
    sink_mutex: os_mutex.Mutex = .{},
    sink_path_buf: [1024]u8 = undefined,
    sink_path_len: usize = 0,
    /// Server builds log to stderr; TEST builds do not.
    ///
    /// A unit test that exercises a logging code path (the tool-call parse layer
    /// alone emits hundreds of lines) sprays them into the suite's stderr, where
    /// they do two kinds of damage: they bury the one line that matters when a test
    /// actually fails, and Zig's build runner echoes any test stderr back tagged
    /// with a `failed command: …/test --listen=-` line — which READS AS A FAILURE
    /// even though the build exits 0 and every test passed. That false signal cost
    /// real debugging time on both sides of a genuine CI break. Test
    /// failures themselves are unaffected: they surface through the test runner and
    /// through each test's own `std.debug.print`, neither of which goes through here.
    ///
    /// Flip it locally (see the file-sink test) when a test must assert on logging.
    stderr_enabled: bool = !builtin.is_test,
};
pub const sinks_shared = @import("shared.zig").Shared(SinksState, "log_state", .{});
inline fn sinks() *SinksState {
    return sinks_shared.get();
}

pub fn setLevel(level: Level) void {
    sinks().current_level = level;
}

pub fn isDebug() bool {
    return @backingInt(sinks().current_level) >= @backingInt(Level.debug);
}

// ── File sink ──

/// Rotate to `<path>.1` once the live file passes this. Two files, so the
/// worst case on disk is bounded at 2x. `--log-level debug` on an agent
/// workload writes a few MB an hour, so this holds days of history.
pub const default_max_bytes: u64 = 32 * 1024 * 1024;

/// Longest single log line written to stderr and the sink. The biggest real line is the
/// debug raw-tool-parse dump (4 KB of model output plus a prefix); anything
/// past this is truncated with a visible marker so a clipped line can never be
/// mistaken for what the model actually emitted.
const line_buf_len = 16 * 1024;
const truncation_marker = "…[mlx-serve: log line truncated]\n";

fn lockSink() void {
    sinks().sink_mutex.lock();
}
fn unlockSink() void {
    sinks().sink_mutex.unlock();
}

/// Build the default log path for a server listening on `port`:
/// `<home>/.mlx-serve/logs/mlx-serve-<port>.log`. Per-port so the app's server
/// and a test server never interleave into one file. Returns the slice written
/// into `buf`.
pub fn defaultLogPath(buf: []u8, home: []const u8, port: u16) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/.mlx-serve/logs/mlx-serve-{d}.log", .{ home, port });
}

/// Pure rotation policy: would appending `incoming` bytes push the live file
/// past its cap? An empty file never rotates, so a single line larger than the
/// cap is written rather than looping forever.
pub fn shouldRotate(current_bytes: u64, incoming: usize, max_bytes: u64) bool {
    if (max_bytes == 0) return false;
    if (current_bytes == 0) return false;
    return current_bytes + incoming > max_bytes;
}

/// `mkdir -p` for the directory holding `path`. Existing components (EEXIST)
/// are fine; a genuinely unwritable path surfaces later as OpenLogFileFailed.
fn makeParentDirs(path: []const u8) void {
    const last_slash = std.mem.lastIndexOfAny(u8, path, "/\\") orelse return;
    var buf: [1024]u8 = undefined;
    var i: usize = 1; // skip a leading '/'
    while (i <= last_slash) : (i += 1) {
        if (i == last_slash or std.fs.path.isSep(path[i])) {
            @memcpy(buf[0..i], path[0..i]);
            buf[i] = 0;
            crt.mkdir(@ptrCast(&buf));
        }
    }
}

/// Open (create/append) a log file, creating parent directories as needed.
/// `max_bytes` of 0 disables rotation. Errors are returned so the caller can
/// warn — a failure here is never fatal to the server.
pub fn openFile(path: []const u8, max_bytes: u64) !void {
    if (path.len == 0 or path.len >= sinks().sink_path_buf.len) return error.InvalidLogPath;

    makeParentDirs(path);

    var path_z: [1024]u8 = undefined;
    @memcpy(path_z[0..path.len], path);
    path_z[path.len] = 0;

    const fd = crt.openWrite(@ptrCast(&path_z), false);
    if (fd < 0) return error.OpenLogFileFailed;

    lockSink();
    defer unlockSink();
    if (sinks().sink_fd >= 0) crt.close(sinks().sink_fd);
    sinks().sink_fd = fd;
    sinks().sink_max_bytes = max_bytes;
    // Start the rotation counter at the file's current size.
    const end = crt.seekEnd(fd);
    sinks().sink_bytes = if (end > 0) @intCast(end) else 0;
    @memcpy(sinks().sink_path_buf[0..path.len], path);
    sinks().sink_path_len = path.len;
    @atomicStore(bool, &sinks().sink_active, true, .release);
}

pub fn closeFile() void {
    lockSink();
    defer unlockSink();
    @atomicStore(bool, &sinks().sink_active, false, .release);
    if (sinks().sink_fd >= 0) crt.close(sinks().sink_fd);
    sinks().sink_fd = -1;
    sinks().sink_bytes = 0;
    sinks().sink_path_len = 0;
}

/// Path of the live log file, or null when no sink is open.
pub fn filePath() ?[]const u8 {
    if (!@atomicLoad(bool, &sinks().sink_active, .acquire)) return null;
    return sinks().sink_path_buf[0..sinks().sink_path_len];
}

/// Caller holds `sinks().sink_mutex`.
fn rotateLocked() void {
    if (sinks().sink_path_len == 0) return;
    var live: [1024]u8 = undefined;
    var prev: [1024]u8 = undefined;
    @memcpy(live[0..sinks().sink_path_len], sinks().sink_path_buf[0..sinks().sink_path_len]);
    live[sinks().sink_path_len] = 0;
    @memcpy(prev[0..sinks().sink_path_len], sinks().sink_path_buf[0..sinks().sink_path_len]);
    @memcpy(prev[sinks().sink_path_len..][0..2], ".1");
    prev[sinks().sink_path_len + 2] = 0;

    if (sinks().sink_fd >= 0) crt.close(sinks().sink_fd);
    _ = crt.rename(@ptrCast(&live), @ptrCast(&prev));
    sinks().sink_fd = crt.openWrite(@ptrCast(&live), true);
    sinks().sink_bytes = 0;
    if (sinks().sink_fd < 0) @atomicStore(bool, &sinks().sink_active, false, .release);
}

fn writeToSink(bytes: []const u8) void {
    lockSink();
    defer unlockSink();
    if (sinks().sink_fd < 0) return;
    if (shouldRotate(sinks().sink_bytes, bytes.len, sinks().sink_max_bytes)) rotateLocked();
    if (sinks().sink_fd < 0) return;

    var off: usize = 0;
    while (off < bytes.len) {
        const n = crt.write(sinks().sink_fd, bytes[off..]);
        if (n <= 0) return; // ENOSPC / EIO: drop the line, never kill the server
        off += @intCast(n);
    }
    sinks().sink_bytes += bytes.len;
}

/// Env-gated live-test harnesses (generation runs driven through the test
/// binary) opt back into stderr logging: the phase/step lines ARE the
/// harness's observable output, and a profile run without them is blind.
pub fn enableStderr() void {
    sinks().stderr_enabled = true;
}

/// Format once, fan out to stderr and (when open) the file. A format `rtfmt` accepts is walked at run time, so a
/// call site costs its packed arguments, not its own instance of the formatter.
fn emit(comptime fmt: []const u8, args: anytype) void {
    if (!sinks().stderr_enabled and !@atomicLoad(bool, &sinks().sink_active, .acquire)) return;
    var buf: [line_buf_len]u8 = undefined;
    // Format into everything but the marker's reserved tail, so an overlong line stays ONE `write(2)`: two writes
    // could interleave with another thread's line between them.
    var w = std.Io.Writer.fixed(buf[0 .. buf.len - truncation_marker.len]);
    const whole = if (comptime rtfmt.supported(fmt, @TypeOf(args))) blk: {
        const packed_args = rtfmt.pack(args);
        break :blk if (rtfmt.print(&w, fmt, &packed_args)) |_| true else |_| false;
    } else if (w.print(fmt, args)) |_| true else |_| false;
    writeLine(&buf, w.buffered().len, whole);
}

fn writeLine(buf: *[line_buf_len]u8, len: usize, whole: bool) void {
    var n = len;
    if (!whole) {
        @memcpy(buf[n..][0..truncation_marker.len], truncation_marker);
        n += truncation_marker.len;
    }
    if (sinks().stderr_enabled) std.debug.print("{s}", .{buf[0..n]});
    if (@atomicLoad(bool, &sinks().sink_active, .acquire)) writeToSink(buf[0..n]);
}

pub fn info(comptime fmt: []const u8, args: anytype) void {
    if (@backingInt(sinks().current_level) >= @backingInt(Level.info)) {
        emit(fmt, args);
    }
}

pub fn warn(comptime fmt: []const u8, args: anytype) void {
    if (@backingInt(sinks().current_level) >= @backingInt(Level.warn)) {
        emit(fmt, args);
    }
}

pub fn err(comptime fmt: []const u8, args: anytype) void {
    if (@backingInt(sinks().current_level) >= @backingInt(Level.err)) {
        emit(fmt, args);
    }
}

pub fn debug(comptime fmt: []const u8, args: anytype) void {
    if (@backingInt(sinks().current_level) >= @backingInt(Level.debug)) {
        emit(fmt, args);
    }
}

/// Would a line at `level` be written? Ask first when the arguments are costly to build.
pub fn enabled(level: Level) bool {
    return @backingInt(sinks().current_level) >= @backingInt(level);
}

/// `info`/`debug`/`warn` with the level chosen at runtime.
pub fn atLevel(level: Level, comptime fmt: []const u8, args: anytype) void {
    if (enabled(level)) emit(fmt, args);
}

// ── Tests ──

const testing = std.testing;

test "Level.fromString valid levels" {
    try testing.expectEqual(Level.err, Level.fromString("error").?);
    try testing.expectEqual(Level.warn, Level.fromString("warn").?);
    try testing.expectEqual(Level.info, Level.fromString("info").?);
    try testing.expectEqual(Level.debug, Level.fromString("debug").?);
}

test "Level.fromString invalid returns null" {
    try testing.expect(Level.fromString("verbose") == null);
    try testing.expect(Level.fromString("") == null);
    try testing.expect(Level.fromString("INFO") == null);
}

test "Level ordering" {
    // err < warn < info < debug
    try testing.expect(@backingInt(Level.err) < @backingInt(Level.warn));
    try testing.expect(@backingInt(Level.warn) < @backingInt(Level.info));
    try testing.expect(@backingInt(Level.info) < @backingInt(Level.debug));
}

test "setLevel changes current level" {
    const original = sinks().current_level;
    defer setLevel(original); // restore

    setLevel(.debug);
    try testing.expectEqual(Level.debug, sinks().current_level);
    try testing.expect(isDebug());
    setLevel(.err);
    try testing.expectEqual(Level.err, sinks().current_level);
    try testing.expect(!isDebug());
}

test "defaultLogPath is per-port under ~/.mlx-serve/logs" {
    var buf: [256]u8 = undefined;
    const p = try defaultLogPath(&buf, "/Users/x", 11234);
    try testing.expectEqualStrings("/Users/x/.mlx-serve/logs/mlx-serve-11234.log", p);
    // Per-port: the app's server and a test server never share a file.
    const q = try defaultLogPath(&buf, "/Users/x", 8098);
    try testing.expectEqualStrings("/Users/x/.mlx-serve/logs/mlx-serve-8098.log", q);
}

test "shouldRotate: caps growth, never loops on an oversized first line" {
    try testing.expect(!shouldRotate(0, 100, 1000)); // empty file: always accept
    try testing.expect(!shouldRotate(900, 100, 1000)); // exactly at the cap
    try testing.expect(shouldRotate(901, 100, 1000)); // one past
    try testing.expect(!shouldRotate(5000, 100, 0)); // 0 = rotation disabled
    // A single line larger than the whole cap is written to an empty file
    // rather than rotating forever.
    try testing.expect(!shouldRotate(0, 999_999, 1000));
}

// NOTE: the sink is a process-global singleton, so every assertion about it
// lives in ONE test — the build runner executes tests in parallel and two
// tests calling `openFile` would fight over `sinks().sink_fd`.
test "file sink: writes lines, honors level, survives reopen, rotates at the cap" {
    // A private dir per run: a fixed /tmp path collided when two test binaries ran at once.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    defer closeFile();
    var root_buf: [512]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(std.Io.Threaded.global_single_threaded.io(), &root_buf)];
    var path_buf: [600]u8 = undefined;
    var rot_buf: [600]u8 = undefined;
    var deep_buf: [600]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buf, "{s}/s.log", .{root}, 0);
    const rotated = try std.fmt.bufPrintSentinel(&rot_buf, "{s}/s.log.1", .{root}, 0);
    const deep = try std.fmt.bufPrintSentinel(&deep_buf, "{s}/a/b/c/deep.log", .{root}, 0);

    const original = sinks().current_level;
    defer setLevel(original);
    setLevel(.info);
    const original_stderr = sinks().stderr_enabled;
    sinks().stderr_enabled = false; // this test drives info/debug for real
    defer sinks().stderr_enabled = original_stderr;

    // No sink yet -> nothing on disk, and filePath reports that.
    try testing.expect(filePath() == null);

    try openFile(path, 0); // rotation disabled
    try testing.expectEqualStrings(path, filePath().?);
    info("hello {d}\n", .{42});
    info("second line\n", .{});
    // Below the threshold: stderr and disk both stay quiet.
    debug("this must not reach disk\n", .{});
    closeFile();
    try testing.expect(filePath() == null);

    var buf: [4096]u8 = undefined;
    const n = readFileForTest(path, &buf);
    try testing.expect(std.mem.indexOf(u8, buf[0..n], "hello 42\n") != null);
    try testing.expect(std.mem.indexOf(u8, buf[0..n], "second line\n") != null);
    try testing.expect(std.mem.indexOf(u8, buf[0..n], "must not reach disk") == null);

    // Reopening appends rather than truncating: history survives a restart.
    try openFile(path, 0);
    info("after restart\n", .{});
    closeFile();
    const n2 = readFileForTest(path, &buf);
    try testing.expect(std.mem.indexOf(u8, buf[0..n2], "hello 42\n") != null);
    try testing.expect(std.mem.indexOf(u8, buf[0..n2], "after restart\n") != null);

    // A tiny cap forces a rotation; the old bytes land in `.log.1`.
    try openFile(path, 64);
    var i: usize = 0;
    while (i < 40) : (i += 1) info("filler line {d}\n", .{i});
    closeFile();
    const live = readFileForTest(path, &buf);
    try testing.expect(live > 0);
    try testing.expect(live <= 64 + 32); // bounded: the last line may overshoot
    var buf2: [4096]u8 = undefined;
    try testing.expect(readFileForTest(rotated, &buf2) > 0); // rotated backup exists

    // Nested parents are created: the real default path is
    // ~/.mlx-serve/logs/… and BOTH components can be missing on a fresh HOME.
    try openFile(deep, 0);
    info("deep\n", .{});
    closeFile();
    try testing.expect(readFileForTest(deep, &buf2) > 0);

    // An overlong line is clipped and SAYS SO — a silently-clipped model dump
    // must never read as the model's actual output.
    crt.unlink(deep);
    try openFile(deep, 0);
    const huge: [line_buf_len + 500]u8 = @splat('x');
    info("{s}\n", .{huge});
    closeFile();
    var big: [line_buf_len * 2]u8 = undefined;
    const wrote = readFileForTest(deep, &big);
    try testing.expectEqual(@as(usize, line_buf_len), wrote); // never exceeds the buffer
    try testing.expect(std.mem.endsWith(u8, big[0..wrote], truncation_marker));
}

fn readFileForTest(path: [*:0]const u8, buf: []u8) usize {
    const fd = crt.openRead(path);
    if (fd < 0) return 0;
    defer crt.close(fd);
    const n = crt.read(fd, buf);
    return if (n > 0) @intCast(n) else 0;
}

test "enabled/atLevel: the level check a caller can ask BEFORE building its arguments" {
    const original = sinks().current_level;
    defer sinks().current_level = original;
    const original_stderr = sinks().stderr_enabled;
    sinks().stderr_enabled = false; // this test drives atLevel for real
    defer sinks().stderr_enabled = original_stderr;

    sinks().current_level = .info;
    try testing.expect(enabled(.err));
    try testing.expect(enabled(.warn));
    try testing.expect(enabled(.info));
    try testing.expect(!enabled(.debug));

    sinks().current_level = .warn;
    try testing.expect(enabled(.warn));
    try testing.expect(!enabled(.info));
    try testing.expect(!enabled(.debug));

    sinks().current_level = .debug;
    inline for (.{ Level.err, Level.warn, Level.info, Level.debug }) |lv| {
        try testing.expect(enabled(lv));
    }

    sinks().current_level = .info;
    atLevel(.debug, "suppressed {d}\n", .{1});
    atLevel(.info, "written {d}\n", .{2});
}
