//! OS calls for threads that carry no `Io` handle (log sink, pollers): C-runtime integer
//! file descriptors, sleep and a monotonic clock. std.c on POSIX; on Windows the CRT's
//! `_open` family (std.c's file calls take a HANDLE there) and kernel32.
const std = @import("std");
const env = @import("../../env.zig");
const builtin = @import("builtin");
const is_windows = builtin.os.tag == .windows;

const win = struct {
    extern "c" fn _open(path: [*:0]const u8, flags: c_int, ...) c_int;
    extern "c" fn _close(fd: c_int) c_int;
    extern "c" fn _write(fd: c_int, buf: [*]const u8, n: c_uint) c_int;
    extern "c" fn _read(fd: c_int, buf: [*]u8, n: c_uint) c_int;
    extern "c" fn _lseeki64(fd: c_int, offset: i64, whence: c_int) i64;
    extern "c" fn _mkdir(path: [*:0]const u8) c_int;
    extern "c" fn _unlink(path: [*:0]const u8) c_int;

    const O_WRONLY = 0x1;
    const O_APPEND = 0x8;
    const O_NOINHERIT = 0x80; // a spawned child (curl) never holds our files open
    const O_CREAT = 0x100;
    const O_TRUNC = 0x200;
    const O_BINARY = 0x8000;
    const S_IREAD = 0x100;
    const S_IWRITE = 0x80;
    const SEEK_END = 2;
};

/// Create-or-open for writing: appends, or truncates when `truncate`. -1 on failure.
pub fn openWrite(path: [*:0]const u8, truncate: bool) c_int {
    if (is_windows) {
        const mode: c_int = if (truncate) win.O_TRUNC else win.O_APPEND;
        return win._open(path, win.O_WRONLY | win.O_CREAT | win.O_BINARY | win.O_NOINHERIT | mode, @as(c_int, win.S_IREAD | win.S_IWRITE));
    }
    return std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = !truncate, .TRUNC = truncate }, @as(std.c.mode_t, 0o644));
}

pub fn openRead(path: [*:0]const u8) c_int {
    if (is_windows) return win._open(path, win.O_BINARY | win.O_NOINHERIT);
    return std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
}

pub fn close(fd: c_int) void {
    _ = if (is_windows) win._close(fd) else std.c.close(fd);
}

/// Bytes written, or <= 0 on error.
pub fn write(fd: c_int, bytes: []const u8) isize {
    if (is_windows) return win._write(fd, bytes.ptr, @intCast(@min(bytes.len, std.math.maxInt(c_int))));
    return std.c.write(fd, bytes.ptr, bytes.len);
}

/// Bytes read, or <= 0 on EOF/error.
pub fn read(fd: c_int, buf: []u8) isize {
    if (is_windows) return win._read(fd, buf.ptr, @intCast(@min(buf.len, std.math.maxInt(c_int))));
    return std.c.read(fd, buf.ptr, buf.len);
}

/// Size of the open file (seeks to its end); <= 0 on error.
pub fn seekEnd(fd: c_int) i64 {
    if (is_windows) return win._lseeki64(fd, 0, win.SEEK_END);
    return std.c.lseek(fd, 0, std.c.SEEK.END);
}

/// Existing directories are fine; failures surface at the next open.
pub fn mkdir(path: [*:0]const u8) void {
    _ = if (is_windows) win._mkdir(path) else std.c.mkdir(path, @as(std.c.mode_t, 0o755));
}

extern "kernel32" fn MoveFileExA(old: [*:0]const u8, new: [*:0]const u8, flags: u32) callconv(.winapi) i32;

/// Atomically replaces an existing `new` on every OS (the Windows CRT's rename refuses to).
pub fn rename(old: [*:0]const u8, new: [*:0]const u8) c_int {
    if (is_windows) return if (MoveFileExA(old, new, 0x1) != 0) 0 else -1; // MOVEFILE_REPLACE_EXISTING
    return std.c.rename(old, new);
}

pub fn unlink(path: [*:0]const u8) void {
    _ = if (is_windows) win._unlink(path) else std.c.unlink(path);
}

const Overlapped = extern struct { internal: usize = 0, internal_high: usize = 0, offset: u32, offset_high: u32, event: ?*anyopaque = null };
extern "c" fn _get_osfhandle(fd: c_int) isize;
extern "kernel32" fn ReadFile(h: isize, buf: [*]u8, n: u32, got: *u32, ov: *Overlapped) callconv(.winapi) i32;

/// Positioned read (std.c.pread's argument order); on Windows an offset ReadFile on the fd's handle.
pub fn pread(fd: c_int, buf: [*]u8, len: usize, off: i64) isize {
    if (!is_windows) return std.c.pread(fd, buf, len, off);
    const at: u64 = @intCast(off);
    var ov: Overlapped = .{ .offset = @truncate(at), .offset_high = @truncate(at >> 32) };
    var got: u32 = 0;
    if (ReadFile(_get_osfhandle(fd), buf, @intCast(@min(len, std.math.maxInt(u32))), &got, &ov) == 0) return -1;
    return got;
}

/// The HOME to default to: Windows has none, and every `~/.mlx-serve` path reads it.
pub fn homeFallback(home: ?[]const u8, profile: ?[]const u8) ?[]const u8 {
    return if (home == null) profile else null;
}

/// Windows: HOME = USERPROFILE when unset, before anything reads it.
pub fn exportHomeDefault() void {
    if (!is_windows) return;
    const profile = env.get(.USERPROFILE);
    if (homeFallback(env.span(.HOME), if (profile) |p| std.mem.span(p) else null) != null)
        env.set(.HOME, profile.?);
}

extern "kernel32" fn Sleep(ms: u32) callconv(.winapi) void;
extern "kernel32" fn GetTickCount64() callconv(.winapi) u64;

pub fn sleepMs(ms: u32) void {
    if (is_windows) return Sleep(ms);
    const ts = std.c.timespec{ .sec = @intCast(ms / 1000), .nsec = @intCast(@as(u64, ms % 1000) * 1_000_000) };
    _ = std.c.nanosleep(&ts, null);
}

pub fn monotonicMs() i64 {
    if (is_windows) return @intCast(GetTickCount64());
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i64, @intCast(ts.sec)) * 1000 + @divTrunc(@as(i64, @intCast(ts.nsec)), 1_000_000);
}

test "homeFallback: Windows takes USERPROFILE only when HOME is unset" {
    try std.testing.expectEqualStrings("C:\\Users\\d", homeFallback(null, "C:\\Users\\d").?);
    try std.testing.expect(homeFallback("C:\\h", "C:\\Users\\d") == null);
    try std.testing.expect(homeFallback(null, null) == null);
}

test "crt: write, reopen to append, read back, rename over an existing file" {
    const a = "crt_test_a.tmp";
    const b = "crt_test_b.tmp";
    defer unlink(a);
    defer unlink(b);

    var fd = openWrite(a, true);
    try std.testing.expect(fd >= 0);
    try std.testing.expectEqual(@as(isize, 3), write(fd, "abc"));
    close(fd);
    fd = openWrite(a, false);
    try std.testing.expectEqual(@as(i64, 3), seekEnd(fd));
    _ = write(fd, "de");
    close(fd);

    fd = openWrite(b, true);
    close(fd);
    try std.testing.expectEqual(@as(c_int, 0), rename(a, b));

    fd = openRead(b);
    defer close(fd);
    var buf: [16]u8 = undefined;
    const n = read(fd, &buf);
    try std.testing.expectEqualStrings("abcde", buf[0..@intCast(n)]);
    try std.testing.expectEqual(@as(isize, 2), pread(fd, &buf, 2, 3));
    try std.testing.expectEqualStrings("de", buf[0..2]);
}
