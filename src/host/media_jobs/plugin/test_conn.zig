//! Test connections: everything a handler writes, read off the other end of a socketpair.

const std = @import("std");
const testing = std.testing;
const Conn = @import("../../http/server.zig").Conn;
const sse = @import("../sse.zig");
const media_progress = @import("../../../sdk/media/progress.zig");

/// The peer a handler sees: loopback, as the app's own client (handlers read it, e.g. LoRA paths are loopback-only).
pub const loopback: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 0 } };

/// Everything `run` writes to a connection, read off the other end of a socketpair.
pub fn capture(a: std.mem.Allocator, comptime run: anytype, args: anytype) ![]u8 {
    var sv: [2]std.posix.fd_t = undefined;
    try testing.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
    const Drain = struct {
        fn f(fd: std.posix.fd_t, out: *std.ArrayList(u8), al: std.mem.Allocator) void {
            var buf: [1 << 16]u8 = undefined;
            while (true) {
                const n = std.c.recv(fd, &buf, buf.len, 0);
                if (n <= 0) return;
                out.appendSlice(al, buf[0..@intCast(n)]) catch return;
            }
        }
    };
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    const t = try std.Thread.spawn(.{}, Drain.f, .{ sv[1], &out, a });
    var conn: Conn = undefined;
    conn.init(.{ .socket = .{ .handle = sv[0], .address = loopback } }, testing.io);
    // The job's keepalive registration outlives it here (the scheduler clears it in production): drop it with the conn.
    const r = @call(.auto, run, .{&conn} ++ args);
    media_progress.streamKeepalive().* = null;
    _ = std.c.close(sv[0]);
    t.join();
    _ = std.c.close(sv[1]);
    try r;
    return out.toOwnedSlice(a);
}

/// `capture` over several connections at once (a merged pass answers each).
pub fn captureMany(a: std.mem.Allocator, comptime n: usize, comptime run: anytype, args: anytype) ![n][]u8 {
    var sv: [n][2]std.posix.fd_t = undefined;
    var conns: [n]Conn = undefined;
    var outs: [n]std.ArrayList(u8) = @splat(.empty);
    var threads: [n]std.Thread = undefined;
    const Drain = struct {
        fn f(fd: std.posix.fd_t, out: *std.ArrayList(u8), al: std.mem.Allocator) void {
            var buf: [1 << 16]u8 = undefined;
            while (true) {
                const k = std.c.recv(fd, &buf, buf.len, 0);
                if (k <= 0) return;
                out.appendSlice(al, buf[0..@intCast(k)]) catch return;
            }
        }
    };
    for (&sv, &conns, &outs, &threads) |*p, *c, *o, *t| {
        try testing.expect(std.c.socketpair(1, 1, 0, p) == 0);
        c.init(.{ .socket = .{ .handle = p[0], .address = loopback } }, testing.io);
        t.* = try std.Thread.spawn(.{}, Drain.f, .{ p[1], o, a });
    }
    @call(.auto, run, .{&conns} ++ args);
    media_progress.streamKeepalive().* = null;
    var got: [n][]u8 = undefined;
    for (&sv, &outs, &threads, &got) |p, *o, t, *g| {
        _ = std.c.close(p[0]);
        t.join();
        _ = std.c.close(p[1]);
        g.* = try o.toOwnedSlice(a);
    }
    return got;
}

