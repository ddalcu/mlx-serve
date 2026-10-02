//! Linux UDP multicast helpers for the mDNS transport. TCP stays in lan.zig.
const std = @import("std");

pub const Socket = i32;
pub const invalid_socket: Socket = -1;
const IpMreq = extern struct { multiaddr: [4]u8, interface: [4]u8 };

pub fn close(s: Socket) void {
    if (s >= 0) _ = std.c.close(s);
}

pub fn monoMs() i64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i64, @intCast(ts.sec)) * 1000 + @divTrunc(@as(i64, @intCast(ts.nsec)), 1_000_000);
}

fn setOpt(s: Socket, level: i32, opt: u32, bytes: []const u8) void {
    _ = std.c.setsockopt(s, level, opt, bytes.ptr, @intCast(bytes.len));
}

pub fn waitReadable(s: Socket, timeout_ms: i32) bool {
    var fds = [_]std.posix.pollfd{.{ .fd = s, .events = std.posix.POLL.IN, .revents = 0 }};
    const ready = std.posix.poll(&fds, timeout_ms) catch return false;
    return ready > 0 and fds[0].revents & std.posix.POLL.IN != 0;
}

pub fn udpBindShared(port: u16) !Socket {
    const s = std.c.socket(std.posix.AF.INET, std.posix.SOCK.DGRAM, 0);
    if (s < 0) return error.SocketFailed;
    errdefer close(s);
    const on: c_int = 1;
    setOpt(s, std.posix.SOL.SOCKET, std.posix.SO.REUSEADDR, std.mem.asBytes(&on));
    setOpt(s, std.posix.SOL.SOCKET, std.posix.SO.REUSEPORT, std.mem.asBytes(&on));
    var sa = std.posix.sockaddr.in{ .port = std.mem.nativeToBig(u16, port), .addr = 0 };
    if (std.c.bind(s, @ptrCast(&sa), @sizeOf(std.posix.sockaddr.in)) != 0) return error.SocketFailed;
    return s;
}

pub fn joinMulticast(s: Socket, group: [4]u8, iface: [4]u8) void {
    const mreq = IpMreq{ .multiaddr = group, .interface = iface };
    setOpt(s, 0, 35, std.mem.asBytes(&mreq)); // IP_ADD_MEMBERSHIP
}

pub fn setMulticastInterface(s: Socket, iface: [4]u8) void {
    setOpt(s, 0, 32, &iface); // IP_MULTICAST_IF
}

pub fn setMulticastTtl(s: Socket, ttl: u8) void {
    const value: c_int = ttl;
    setOpt(s, 0, 33, std.mem.asBytes(&value)); // IP_MULTICAST_TTL
}

pub fn setMulticastLoop(s: Socket, on: bool) void {
    const value: c_int = if (on) 1 else 0;
    setOpt(s, 0, 34, std.mem.asBytes(&value)); // IP_MULTICAST_LOOP
}

pub fn sendTo(s: Socket, ip4: [4]u8, port: u16, data: []const u8) void {
    var sa = std.posix.sockaddr.in{ .port = std.mem.nativeToBig(u16, port), .addr = @bitCast(ip4) };
    _ = std.c.sendto(s, data.ptr, data.len, 0, @ptrCast(&sa), @sizeOf(std.posix.sockaddr.in));
}

pub const Datagram = struct { len: usize, from_ip4: [4]u8 };

pub fn recvFrom(s: Socket, buf: []u8) ?Datagram {
    var sa: std.posix.sockaddr.in = undefined;
    var len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr.in);
    const n = std.c.recvfrom(s, buf.ptr, buf.len, 0, @ptrCast(&sa), &len);
    if (n <= 0) return null;
    return .{ .len = @intCast(n), .from_ip4 = @bitCast(sa.addr) };
}

pub fn localIp4Addresses(out: [][4]u8) [][4]u8 {
    var ifap: ?*Ifaddrs = null;
    if (getifaddrs(&ifap) != 0) return out[0..0];
    defer freeifaddrs(ifap);
    var n: usize = 0;
    var it = ifap;
    while (it) |ia| : (it = ia.next) {
        const addr = ia.addr orelse continue;
        if (addr.family != std.posix.AF.INET or ia.flags & 1 == 0) continue;
        if (n == out.len) break;
        const sin: *const std.posix.sockaddr.in = @ptrCast(@alignCast(addr));
        out[n] = @bitCast(sin.addr);
        n += 1;
    }
    return out[0..n];
}

const Ifaddrs = extern struct {
    next: ?*Ifaddrs,
    name: [*:0]u8,
    flags: c_uint,
    addr: ?*std.posix.sockaddr,
    netmask: ?*std.posix.sockaddr,
    dstaddr: ?*std.posix.sockaddr,
    data: ?*anyopaque,
};
extern "c" fn getifaddrs(out: *?*Ifaddrs) c_int;
extern "c" fn freeifaddrs(p: ?*Ifaddrs) void;
