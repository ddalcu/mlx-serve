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

var current_level: Level = .info;

pub fn setLevel(level: Level) void {
    current_level = level;
}

pub fn isDebug() bool {
    return @intFromEnum(current_level) >= @intFromEnum(Level.debug);
}

// ── File sink ──

/// Rotate to `<path>.1` once the live file passes this. Two files, so the
/// worst case on disk is bounded at 2x. `--log-level debug` on an agent
/// workload writes a few MB an hour, so this holds days of history.
pub const default_max_bytes: u64 = 32 * 1024 * 1024;

/// Longest single log line written to the sink. The biggest real line is the
/// debug raw-tool-parse dump (4 KB of model output plus a prefix); anything
/// past this is truncated with a visible marker so a clipped line can never be
/// mistaken for what the model actually emitted.
const line_buf_len = 16 * 1024;
const truncation_marker = "…[mlx-serve: log line truncated]\n";

/// A pthread mutex, not `std.Io.Mutex`: locking the latter needs an `Io`
/// handle, and log calls arrive from threads that don't carry one.
var sink_mutex: std.c.pthread_mutex_t = .{};

fn lockSink() void {
    _ = std.c.pthread_mutex_lock(&sink_mutex);
}
fn unlockSink() void {
    _ = std.c.pthread_mutex_unlock(&sink_mutex);
}

var sink_fd: c_int = -1;
var sink_bytes: u64 = 0;
var sink_max_bytes: u64 = default_max_bytes;
var sink_path_buf: [1024]u8 = undefined;
var sink_path_len: usize = 0;
/// Read outside the mutex on every log call, so keep it atomic. When false the
/// sink costs one relaxed load and nothing else.
var sink_active: bool = false;

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

fn openAppend(path_z: [*:0]const u8, truncate: bool) c_int {
    return std.c.open(path_z, .{
        .ACCMODE = .WRONLY,
        .CREAT = true,
        .APPEND = !truncate,
        .TRUNC = truncate,
    }, @as(std.c.mode_t, 0o644));
}

/// `mkdir -p` for the directory holding `path`. Existing components (EEXIST)
/// are fine; a genuinely unwritable path surfaces later as OpenLogFileFailed.
fn makeParentDirs(path: []const u8) void {
    const last_slash = std.mem.lastIndexOfScalar(u8, path, '/') orelse return;
    var buf: [1024]u8 = undefined;
    var i: usize = 1; // skip a leading '/'
    while (i <= last_slash) : (i += 1) {
        if (i == last_slash or path[i] == '/') {
            @memcpy(buf[0..i], path[0..i]);
            buf[i] = 0;
            _ = std.c.mkdir(@ptrCast(&buf), @as(std.c.mode_t, 0o755));
        }
    }
}

/// Open (create/append) a log file, creating parent directories as needed.
/// `max_bytes` of 0 disables rotation. Errors are returned so the caller can
/// warn — a failure here is never fatal to the server.
pub fn openFile(path: []const u8, max_bytes: u64) !void {
    if (path.len == 0 or path.len >= sink_path_buf.len) return error.InvalidLogPath;

    makeParentDirs(path);

    var path_z: [1024]u8 = undefined;
    @memcpy(path_z[0..path.len], path);
    path_z[path.len] = 0;

    const fd = openAppend(@ptrCast(&path_z), false);
    if (fd < 0) return error.OpenLogFileFailed;

    lockSink();
    defer unlockSink();
    if (sink_fd >= 0) _ = std.c.close(sink_fd);
    sink_fd = fd;
    sink_max_bytes = max_bytes;
    // Start the rotation counter at the file's current size.
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    sink_bytes = if (end > 0) @intCast(end) else 0;
    @memcpy(sink_path_buf[0..path.len], path);
    sink_path_len = path.len;
    @atomicStore(bool, &sink_active, true, .release);
}

pub fn closeFile() void {
    lockSink();
    defer unlockSink();
    @atomicStore(bool, &sink_active, false, .release);
    if (sink_fd >= 0) _ = std.c.close(sink_fd);
    sink_fd = -1;
    sink_bytes = 0;
    sink_path_len = 0;
}

/// Path of the live log file, or null when no sink is open.
pub fn filePath() ?[]const u8 {
    if (!@atomicLoad(bool, &sink_active, .acquire)) return null;
    return sink_path_buf[0..sink_path_len];
}

/// Caller holds `sink_mutex`.
fn rotateLocked() void {
    if (sink_path_len == 0) return;
    var live: [1024]u8 = undefined;
    var prev: [1024]u8 = undefined;
    @memcpy(live[0..sink_path_len], sink_path_buf[0..sink_path_len]);
    live[sink_path_len] = 0;
    @memcpy(prev[0..sink_path_len], sink_path_buf[0..sink_path_len]);
    @memcpy(prev[sink_path_len..][0..2], ".1");
    prev[sink_path_len + 2] = 0;

    if (sink_fd >= 0) _ = std.c.close(sink_fd);
    _ = std.c.rename(@ptrCast(&live), @ptrCast(&prev));
    sink_fd = openAppend(@ptrCast(&live), true);
    sink_bytes = 0;
    if (sink_fd < 0) @atomicStore(bool, &sink_active, false, .release);
}

fn writeToSink(bytes: []const u8) void {
    lockSink();
    defer unlockSink();
    if (sink_fd < 0) return;
    if (shouldRotate(sink_bytes, bytes.len, sink_max_bytes)) rotateLocked();
    if (sink_fd < 0) return;

    var off: usize = 0;
    while (off < bytes.len) {
        const n = std.c.write(sink_fd, bytes.ptr + off, bytes.len - off);
        if (n <= 0) return; // ENOSPC / EIO: drop the line, never kill the server
        off += @intCast(n);
    }
    sink_bytes += bytes.len;
}

/// Server builds log to stderr; TEST builds do not.
///
/// A unit test that exercises a logging code path (the tool-call parse layer
/// alone emits hundreds of lines) sprays them into the suite's stderr, where
/// they do two kinds of damage: they bury the one line that matters when a test
/// actually fails, and Zig's build runner echoes any test stderr back tagged
/// with a `failed command: …/test --listen=-` line — which READS AS A FAILURE
/// even though the build exits 0 and every test passed. That false signal cost
/// real debugging time on both sides of a genuine CI break (2026-07-14). Test
/// failures themselves are unaffected: they surface through the test runner and
/// through each test's own `std.debug.print`, neither of which goes through here.
///
/// Flip it locally (see the file-sink test) when a test must assert on logging.
var stderr_enabled: bool = !builtin.is_test;

/// Env-gated live-test harnesses (generation runs driven through the test
/// binary) opt back into stderr logging: the phase/step lines ARE the
/// harness's observable output, and a profile run without them is blind.
pub fn enableStderr() void {
    stderr_enabled = true;
}

/// Format once, fan out to stderr and (when open) the file.
fn emit(comptime fmt: []const u8, args: anytype) void {
    if (stderr_enabled) std.debug.print(fmt, args);
    if (!@atomicLoad(bool, &sink_active, .acquire)) return;

    var buf: [line_buf_len]u8 = undefined;
    // Format into everything but the marker's reserved tail, so an overlong
    // line stays ONE `write(2)` — two writes could interleave with another
    // thread's line between them.
    var w = std.Io.Writer.fixed(buf[0 .. buf.len - truncation_marker.len]);
    if (w.print(fmt, args)) |_| {
        writeToSink(w.buffered());
    } else |_| {
        const kept = w.buffered().len;
        @memcpy(buf[kept..][0..truncation_marker.len], truncation_marker);
        writeToSink(buf[0 .. kept + truncation_marker.len]);
    }
}

pub fn info(comptime fmt: []const u8, args: anytype) void {
    if (@intFromEnum(current_level) >= @intFromEnum(Level.info)) {
        emit(fmt, args);
    }
}

pub fn warn(comptime fmt: []const u8, args: anytype) void {
    if (@intFromEnum(current_level) >= @intFromEnum(Level.warn)) {
        emit(fmt, args);
    }
}

pub fn err(comptime fmt: []const u8, args: anytype) void {
    if (@intFromEnum(current_level) >= @intFromEnum(Level.err)) {
        emit(fmt, args);
    }
}

pub fn debug(comptime fmt: []const u8, args: anytype) void {
    if (@intFromEnum(current_level) >= @intFromEnum(Level.debug)) {
        emit(fmt, args);
    }
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
    try testing.expect(@intFromEnum(Level.err) < @intFromEnum(Level.warn));
    try testing.expect(@intFromEnum(Level.warn) < @intFromEnum(Level.info));
    try testing.expect(@intFromEnum(Level.info) < @intFromEnum(Level.debug));
}

test "setLevel changes current level" {
    const original = current_level;
    defer setLevel(original); // restore

    setLevel(.debug);
    try testing.expectEqual(Level.debug, current_level);
    try testing.expect(isDebug());
    setLevel(.err);
    try testing.expectEqual(Level.err, current_level);
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
// tests calling `openFile` would fight over `sink_fd`.
test "file sink: writes lines, honors level, survives reopen, rotates at the cap" {
    const dir = "/tmp/mlx-serve-logtest";
    const path = dir ++ "/s.log";
    _ = std.c.mkdir(dir, @as(std.c.mode_t, 0o755));
    _ = std.c.unlink(path);
    _ = std.c.unlink(path ++ ".1");
    defer {
        closeFile();
        _ = std.c.unlink(path);
        _ = std.c.unlink(path ++ ".1");
    }

    const original = current_level;
    defer setLevel(original);
    setLevel(.info);
    const original_stderr = stderr_enabled;
    stderr_enabled = false; // this test drives info/debug for real
    defer stderr_enabled = original_stderr;

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
    try testing.expect(readFileForTest(path ++ ".1", &buf2) > 0); // rotated backup exists

    // Nested parents are created: the real default path is
    // ~/.mlx-serve/logs/… and BOTH components can be missing on a fresh HOME.
    const deep = "/tmp/mlx-serve-logtest/a/b/c/deep.log";
    defer {
        closeFile();
        _ = std.c.unlink(deep);
        _ = std.c.rmdir("/tmp/mlx-serve-logtest/a/b/c");
        _ = std.c.rmdir("/tmp/mlx-serve-logtest/a/b");
        _ = std.c.rmdir("/tmp/mlx-serve-logtest/a");
    }
    try openFile(deep, 0);
    info("deep\n", .{});
    closeFile();
    try testing.expect(readFileForTest(deep, &buf2) > 0);

    // An overlong line is clipped and SAYS SO — a silently-clipped model dump
    // must never read as the model's actual output.
    _ = std.c.unlink(deep);
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
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return 0;
    defer _ = std.c.close(fd);
    const n = std.c.read(fd, buf.ptr, buf.len);
    return if (n > 0) @intCast(n) else 0;
}
