//! listenExclusive(Linux), parseIp4, StartupResponder

const std = @import("std");
const win_io = @import("../../sdk/util/win_io.zig");
const builtin = @import("builtin");
const model_registry_mod = @import("../registry/registry.zig");
const ModelRegistry = model_registry_mod.ModelRegistry;
const testing = std.testing;

const Conn = @import("conn.zig").Conn;
const isRoute = @import("router.zig").isRoute;
const readRequest = @import("request.zig").readRequest;
const loadProgressBody = @import("../api/models.zig").loadProgressBody;
const sendResponse = @import("respond.zig").sendResponse;
const peerIsLoopback = @import("auth.zig").peerIsLoopback;
const sendErrorResponse = @import("errors.zig").sendErrorResponse;

/// Listen with SO_REUSEADDR (a restart can rebind over TIME_WAIT) but WITHOUT
/// SO_REUSEPORT, which std sets beside it: with it a second instance binds the
/// same port and silently takes the connections. The kernel checks the flag on
/// the socket already bound, so clearing it after bind is enough.
pub fn listenExclusive(io: std.Io, addr: std.Io.net.IpAddress) !std.Io.net.Server {
    if (comptime builtin.os.tag == .linux) return listenExclusiveLinux(addr);
    // Windows' SO_REUSEADDR lets another socket bind over a live listener, so it stays off. std.Io binds
    // inside listen, so SO_EXCLUSIVEADDRUSE cannot be set first: a process that sets SO_REUSEADDR can still share the port.
    if (comptime builtin.os.tag == .windows) return addr.listen(io, .{});
    var server = try addr.listen(io, .{ .reuse_address = true });
    errdefer server.deinit(io);
    try std.posix.setsockopt(server.socket.handle, std.posix.SOL.SOCKET, std.posix.SO.REUSEPORT, &std.mem.toBytes(@as(c_int, 0)));
    return server;
}

/// Whether `handle` turns readable (a connection to accept, bytes, or EOF) within `timeout_ms`.
pub fn socketReadable(handle: std.Io.net.Socket.Handle, timeout_ms: i32) !bool {
    if (comptime builtin.os.tag == .windows)
        return try win_io.poll(handle, win_io.RECEIVE | win_io.ACCEPT | win_io.GONE, timeout_ms) != 0;
    var fds = [_]std.posix.pollfd{.{ .fd = handle, .events = std.posix.POLL.IN, .revents = 0 }};
    return try std.posix.poll(&fds, timeout_ms) > 0;
}

/// Linux joins a socket to its SO_REUSEPORT group at bind, so clearing the flag afterwards still
/// let a second server bind the port and take half its connections. Only SO_REUSEADDR is set:
/// a restart binds over TIME_WAIT connections, a live listener refuses it.
fn listenExclusiveLinux(addr: std.Io.net.IpAddress) !std.Io.net.Server {
    const linux = std.os.linux;
    var sa: extern union { in: linux.sockaddr.in, in6: linux.sockaddr.in6, any: linux.sockaddr } = undefined;
    var len: linux.socklen_t = undefined;
    const family: c_uint = switch (addr) {
        .ip4 => |a| blk: {
            sa = .{ .in = .{ .port = std.mem.nativeToBig(u16, a.port), .addr = @bitCast(a.bytes) } };
            len = @sizeOf(linux.sockaddr.in);
            break :blk linux.AF.INET;
        },
        .ip6 => |a| blk: {
            sa = .{ .in6 = .{ .port = std.mem.nativeToBig(u16, a.port), .flowinfo = a.flow, .addr = a.bytes, .scope_id = 0 } };
            len = @sizeOf(linux.sockaddr.in6);
            break :blk linux.AF.INET6;
        },
    };
    const fd = std.c.socket(family, linux.SOCK.STREAM | linux.SOCK.CLOEXEC, 0);
    if (fd < 0) return error.SystemResources;
    errdefer _ = std.c.close(fd);
    const one: c_int = 1;
    if (std.c.setsockopt(fd, linux.SOL.SOCKET, linux.SO.REUSEADDR, &one, @sizeOf(c_int)) != 0) return error.Unexpected;
    switch (std.posix.errno(std.c.bind(fd, &sa.any, len))) {
        .SUCCESS => {},
        .ADDRINUSE => return error.AddressInUse,
        .ACCES => return error.AccessDenied,
        else => return error.Unexpected,
    }
    if (std.c.listen(fd, std.Io.net.default_kernel_backlog) != 0) return error.Unexpected;
    if (std.c.getsockname(fd, &sa.any, &len) != 0) return error.Unexpected;
    var bound = addr;
    switch (bound) {
        .ip4 => |*a| a.port = std.mem.bigToNative(u16, sa.in.port),
        .ip6 => |*a| a.port = std.mem.bigToNative(u16, sa.in6.port),
    }
    return .{ .socket = .{ .handle = fd, .address = bound }, .options = {} };
}

/// Start the HTTP server on the given host and port.
///
/// `cfg` carries all process-wide defaults (context size, timeouts, PLD
/// defaults, reasoning budget). It's copied into the module-level
/// `server_config` before the listen loop starts.
/// Dotted-decimal IPv4 (`0.0.0.0` and anything unparsable read as zeros).
pub fn parseIp4(host: []const u8) [4]u8 {
    var out: [4]u8 = .{ 0, 0, 0, 0 };
    if (std.mem.eql(u8, host, "0.0.0.0")) return out;
    var parts = std.mem.splitScalar(u8, host, '.');
    var idx: usize = 0;
    while (parts.next()) |part| {
        if (idx >= 4) break;
        out[idx] = std.fmt.parseInt(u8, part, 10) catch 0;
        idx += 1;
    }
    return out;
}

/// Answers on the bound socket while the startup load runs, before the accept loop takes it:
/// `/health` is 503 `loading` (clients keep waiting), a loopback client reads `/v1/load-progress`,
/// anything else is 503. One connection at a time; nothing here touches the scheduler.
pub const StartupResponder = struct {
    server: *std.Io.net.Server,
    registry: *ModelRegistry,
    allocator: std.mem.Allocator,
    io: std.Io,
    done: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    pub fn start(self: *StartupResponder) void {
        self.thread = std.Thread.spawn(.{}, run, .{self}) catch null;
    }

    pub fn stop(self: *StartupResponder) void {
        self.done.store(true, .release);
        if (self.thread) |t| t.join();
        self.thread = null;
    }

    fn run(self: *StartupResponder) void {
        while (!self.done.load(.acquire)) {
            const ready = socketReadable(self.server.socket.handle, 100) catch continue;
            // Once done, a waiting connection is the accept loop's.
            if (!ready or self.done.load(.acquire)) continue;
            const accepted = self.server.accept(self.io) catch continue;
            var conn: Conn = undefined;
            Conn.init(&conn, accepted, self.io);
            defer conn.close();
            self.answer(&conn) catch {};
        }
    }

    fn answer(self: *StartupResponder, conn: *Conn) !void {
        const raw = (try readRequest(self.allocator, conn)) orelse return;
        defer self.allocator.free(raw.buf);
        const line_end = std.mem.indexOf(u8, raw.buf[0..raw.len], "\r\n") orelse return;
        var words = std.mem.splitScalar(u8, raw.buf[0..line_end], ' ');
        _ = words.next();
        const raw_path = words.next() orelse return;
        const path = if (std.mem.indexOfScalar(u8, raw_path, '?')) |q| raw_path[0..q] else raw_path;
        if (isRoute(path, "/health")) return sendResponse(conn, "503 Service Unavailable", "application/json", "{\"status\":\"loading\"}");
        if (isRoute(path, "/v1/load-progress") and peerIsLoopback(conn)) {
            const body = try loadProgressBody(self.allocator, self.io, self.registry);
            defer self.allocator.free(body);
            return sendResponse(conn, "200 OK", "application/json", body);
        }
        return sendErrorResponse(self.allocator, conn, "503 Service Unavailable", "loading", "The model is still loading", 503);
    }
};

// ── tests ──

/// `request` through `StartupResponder.answer` over a socketpair from `peer`; the reply's bytes.
fn startupReply(r: *StartupResponder, request: []const u8, peer: std.Io.net.IpAddress, out: []u8) ![]const u8 {
    var sv: [2]std.posix.fd_t = undefined;
    try testing.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
    defer _ = std.c.close(sv[1]);
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = peer } }, testing.io);
    defer conn.close();
    try testing.expectEqual(@as(isize, @intCast(request.len)), std.c.send(sv[1], request.ptr, request.len, 0));
    _ = std.c.shutdown(sv[1], 1);
    try r.answer(&conn);
    try conn.flush();
    const got = std.c.recv(sv[1], out.ptr, out.len, std.c.MSG.DONTWAIT);
    return out[0..@intCast(@max(got, 0))];
}

test "startup: /health is 503 loading, a loopback client reads the load, the rest is 503" {
    var reg = try ModelRegistry.init(testing.allocator, testing.io, null, 3, 0, null);
    defer reg.deinit();
    const stub = try reg.registerStub("m", "/models/m", 10);
    reg.mutex.lockUncancelable(testing.io);
    try testing.expect(reg.tryBeginLoadLocked(stub));
    reg.mutex.unlock(testing.io);
    var r: StartupResponder = .{ .server = undefined, .registry = reg, .allocator = testing.allocator, .io = testing.io };
    const local: std.Io.net.IpAddress = .{ .ip4 = std.Io.net.Ip4Address.loopback(0) };
    const lan: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 192, 168, 1, 5 }, .port = 0 } };
    var buf: [4096]u8 = undefined;

    var reply = try startupReply(&r, "GET /health HTTP/1.1\r\n\r\n", local, &buf);
    try testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 503"));
    try testing.expect(std.mem.endsWith(u8, reply, "{\"status\":\"loading\"}"));
    reply = try startupReply(&r, "GET /v1/load-progress HTTP/1.1\r\n\r\n", local, &buf);
    try testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 200"));
    try testing.expect(std.mem.indexOf(u8, reply, "\"path\":\"/models/m\",\"progress\":{\"phase\":\"starting\"") != null);
    reply = try startupReply(&r, "GET /v1/load-progress HTTP/1.1\r\n\r\n", lan, &buf);
    try testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 503"));
    reply = try startupReply(&r, "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}", local, &buf);
    try testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 503"));
}

test "listenExclusive: a second server cannot bind a port that is already listening" {
    var first = try listenExclusive(testing.io, .{ .ip4 = std.Io.net.Ip4Address.loopback(0) });
    defer first.deinit(testing.io);
    try testing.expectError(error.AddressInUse, listenExclusive(testing.io, first.socket.address));
}
