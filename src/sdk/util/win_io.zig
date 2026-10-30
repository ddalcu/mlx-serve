//! Waiting on Windows handles with a timeout. std.Io's sockets there are AFD endpoint handles, which
//! WSAPoll does not accept, so readiness is asked of AFD itself (IOCTL_AFD_POLL, the request libuv and
//! wepoll use); a child's pipe is read overlapped, waiting on the read's own event.
const std = @import("std");
const builtin = @import("builtin");
const windows = std.os.windows;

pub const RECEIVE: u32 = 0x1;
pub const DISCONNECT: u32 = 0x8;
pub const ABORT: u32 = 0x10;
pub const LOCAL_CLOSE: u32 = 0x20;
pub const ACCEPT: u32 = 0x80;
/// The peer is gone: a FIN, a reset, or our own close.
pub const GONE: u32 = DISCONNECT | ABORT | LOCAL_CLOSE;

const HandleInfo = extern struct { handle: windows.HANDLE, events: u32, status: windows.NTSTATUS };
const PollInfo = extern struct { timeout: i64, number_of_handles: u32, exclusive: u32, handles: [1]HandleInfo };

extern "kernel32" fn CreateEventW(attrs: ?*anyopaque, manual_reset: i32, initial_state: i32, name: ?[*:0]const u16) callconv(.winapi) ?windows.HANDLE;
extern "kernel32" fn WaitForSingleObject(handle: windows.HANDLE, ms: u32) callconv(.winapi) u32;
extern "kernel32" fn ReadFile(handle: windows.HANDLE, buf: [*]u8, len: u32, got: ?*u32, ov: *Overlapped) callconv(.winapi) i32;
extern "kernel32" fn GetOverlappedResult(handle: windows.HANDLE, ov: *Overlapped, got: *u32, wait: i32) callconv(.winapi) i32;
extern "kernel32" fn CancelIoEx(handle: windows.HANDLE, ov: *Overlapped) callconv(.winapi) i32;
extern "kernel32" fn GetLastError() callconv(.winapi) u32;

const Overlapped = extern struct { internal: usize = 0, internal_high: usize = 0, offset: u32 = 0, offset_high: u32 = 0, event: windows.HANDLE };
const WAIT_TIMEOUT: u32 = 0x102;
const ERROR_BROKEN_PIPE: u32 = 109;
const ERROR_IO_PENDING: u32 = 997;

/// The subset of `events` that holds for `handle` within `timeout_ms` (0 probes); 0 on timeout.
pub fn poll(handle: windows.HANDLE, events: u32, timeout_ms: i32) error{PollFailed}!u32 {
    var info = PollInfo{
        .timeout = -@as(i64, @max(timeout_ms, 0)) * 10_000, // relative, in 100 ns units
        .number_of_handles = 1,
        .exclusive = 0,
        .handles = .{.{ .handle = handle, .events = events, .status = .SUCCESS }},
    };
    const event = CreateEventW(null, 1, 0, null) orelse return error.PollFailed;
    defer windows.CloseHandle(event);
    var iosb: windows.IO_STATUS_BLOCK = undefined;
    var status = windows.ntdll.NtDeviceIoControlFile(handle, event, null, null, &iosb, windows.IOCTL.AFD.POLL, &info, @sizeOf(PollInfo), &info, @sizeOf(PollInfo));
    if (status == .PENDING) {
        _ = WaitForSingleObject(event, 0xFFFFFFFF);
        status = iosb.u.Status;
    }
    if (status == .TIMEOUT) return 0; // AFD's timeout is a status, not an empty success
    if (status != .SUCCESS) return error.PollFailed;
    return if (info.number_of_handles == 0) 0 else info.handles[0].events & events;
}

/// One read from an overlapped `handle` (a child's pipe), as soon as bytes arrive; while none do,
/// `stop(ctx)` is asked every second and true cancels the read (error.Stopped). 0 is EOF.
pub fn readPolled(handle: windows.HANDLE, buf: []u8, ctx: anytype, comptime stop: fn (@TypeOf(ctx)) bool) error{ ReadFailed, Stopped }!usize {
    const event = CreateEventW(null, 1, 0, null) orelse return error.ReadFailed;
    defer windows.CloseHandle(event);
    var ov = Overlapped{ .event = event };
    if (ReadFile(handle, buf.ptr, @intCast(@min(buf.len, std.math.maxInt(u32))), null, &ov) == 0) switch (GetLastError()) {
        ERROR_IO_PENDING => while (WaitForSingleObject(event, 1000) == WAIT_TIMEOUT) {
            if (!stop(ctx)) continue;
            _ = CancelIoEx(handle, &ov);
            var got: u32 = 0;
            _ = GetOverlappedResult(handle, &ov, &got, 1); // the read is ours until it settles
            return if (got > 0) got else error.Stopped;
        },
        ERROR_BROKEN_PIPE => return 0,
        else => return error.ReadFailed,
    };
    var got: u32 = 0;
    if (GetOverlappedResult(handle, &ov, &got, 1) == 0) return if (GetLastError() == ERROR_BROKEN_PIPE) 0 else error.ReadFailed;
    return got;
}

test "win_io: a listener times out with no client, then reports the connection to accept" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;
    const io = std.testing.io;
    const addr: std.Io.net.IpAddress = .{ .ip4 = std.Io.net.Ip4Address.loopback(0) };
    var server = try addr.listen(io, .{});
    defer server.deinit(io);
    try std.testing.expectEqual(@as(u32, 0), try poll(server.socket.handle, ACCEPT, 50));

    var client = try server.socket.address.connect(io, .{ .mode = .stream });
    try std.testing.expect(try poll(server.socket.handle, ACCEPT, 2000) & ACCEPT != 0);

    var conn = try server.accept(io);
    defer conn.close(io);
    try std.testing.expectEqual(@as(u32, 0), try poll(conn.socket.handle, RECEIVE | GONE, 0));
    client.close(io);
    try std.testing.expect(try poll(conn.socket.handle, GONE, 2000) & GONE != 0);
}

test "win_io: readPolled hands over a child's output then EOF, and stops a silent one when asked" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;
    const io = std.testing.io;
    var buf: [64]u8 = undefined;
    const Flags = struct {
        fn never(_: void) bool {
            return false;
        }
        fn always(_: void) bool {
            return true;
        }
    };
    var talker = try std.process.spawn(io, .{ .argv = &.{ "cmd.exe", "/c", "echo hi" }, .stdin = .ignore, .stdout = .pipe, .stderr = .ignore });
    defer talker.kill(io);
    var n = try readPolled(talker.stdout.?.handle, &buf, {}, Flags.never);
    try std.testing.expect(std.mem.startsWith(u8, buf[0..n], "hi"));
    while (n != 0) n = try readPolled(talker.stdout.?.handle, &buf, {}, Flags.never);

    var quiet = try std.process.spawn(io, .{ .argv = &.{ "cmd.exe", "/c", "ping -n 6 127.0.0.1 >nul" }, .stdin = .ignore, .stdout = .pipe, .stderr = .ignore });
    defer quiet.kill(io);
    try std.testing.expectError(error.Stopped, readPolled(quiet.stdout.?.handle, &buf, {}, Flags.always));
}
