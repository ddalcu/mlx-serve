//! Minimal RFC 6455 WebSocket framing + handshake.
//!
//! Server-side only — no masking on outbound frames; rejects fragmented
//! control frames per spec; tear down on payloads >16 MiB.
//!
//! Designed to layer on top of `server.Conn`: the caller drives the read/write
//! loop and decides what to do with text frames.

const std = @import("std");

/// Server-imposed cap on a single message's reassembled payload.
pub const max_message_bytes: usize = 16 * 1024 * 1024;

/// Magic GUID per RFC 6455 §1.3.
const ws_magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

pub const Opcode = enum(u4) {
    continuation = 0x0,
    text = 0x1,
    binary = 0x2,
    close = 0x8,
    ping = 0x9,
    pong = 0xA,
    _,
};

pub const Message = struct {
    opcode: Opcode,
    /// Owned by the caller's allocator (free with the same allocator after use).
    payload: []u8,
};

/// Minimal `Conn`-like interface used by this module. Keeps `ws.zig`
/// independent of `server.zig`. Any caller that exposes `read(buf) -> usize`
/// and `writeAll([]const u8) -> !void` works.
pub fn WsConn(comptime ConnT: type) type {
    return struct {
        const Self = @This();
        conn: *ConnT,
        /// Set when a control or fragmented data message has been partially
        /// reassembled. When non-null, any incoming frame with opcode != 0
        /// (continuation) and != control is a protocol violation.
        in_message: ?std.ArrayList(u8) = null,
        in_opcode: Opcode = .continuation,

        pub fn init(conn: *ConnT) Self {
            return .{ .conn = conn };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            if (self.in_message) |*buf| buf.deinit(allocator);
        }

        /// Read a complete WebSocket message (handles fragmentation). The
        /// returned `payload` is owned by the caller and must be freed with
        /// the same allocator. On close/ping/pong, the corresponding opcode
        /// is returned with its payload — the caller decides how to respond.
        pub fn readMessage(self: *Self, allocator: std.mem.Allocator) !Message {
            while (true) {
                const frame = try readFrame(self.conn, allocator);
                errdefer allocator.free(frame.payload);

                // Control frames (close/ping/pong) cannot be fragmented and
                // can interleave inside a fragmented data message. Bubble
                // them up immediately to the caller.
                if (isControlOpcode(frame.opcode)) {
                    if (!frame.fin or frame.payload.len > 125) return error.WsProtocol;
                    return .{ .opcode = frame.opcode, .payload = frame.payload };
                }

                // Data frame: handle fragmentation.
                if (self.in_message == null) {
                    if (frame.opcode == .continuation) return error.WsProtocol;
                    if (frame.fin) {
                        // Single-frame message — return directly.
                        return .{ .opcode = frame.opcode, .payload = frame.payload };
                    }
                    // First frame of a fragmented message.
                    var buf: std.ArrayList(u8) = .empty;
                    errdefer buf.deinit(allocator);
                    try buf.appendSlice(allocator, frame.payload);
                    allocator.free(frame.payload);
                    self.in_message = buf;
                    self.in_opcode = frame.opcode;
                } else {
                    if (frame.opcode != .continuation) return error.WsProtocol;
                    var buf = &self.in_message.?;
                    if (buf.items.len + frame.payload.len > max_message_bytes) {
                        allocator.free(frame.payload);
                        return error.WsTooLarge;
                    }
                    try buf.appendSlice(allocator, frame.payload);
                    allocator.free(frame.payload);
                    if (frame.fin) {
                        const owned = try buf.toOwnedSlice(allocator);
                        const op = self.in_opcode;
                        self.in_message = null;
                        return .{ .opcode = op, .payload = owned };
                    }
                }
            }
        }

        pub fn writeText(self: *Self, payload: []const u8) !void {
            try writeFrame(self.conn, .text, true, payload);
        }

        pub fn writePong(self: *Self, payload: []const u8) !void {
            try writeFrame(self.conn, .pong, true, payload);
        }

        pub fn writeClose(self: *Self, code: u16, reason: []const u8) !void {
            var buf: [125]u8 = undefined;
            const n = @min(reason.len, buf.len - 2);
            std.mem.writeInt(u16, buf[0..2], code, .big);
            @memcpy(buf[2 .. 2 + n], reason[0..n]);
            try writeFrame(self.conn, .close, true, buf[0 .. 2 + n]);
        }
    };
}

fn isControlOpcode(op: Opcode) bool {
    return switch (op) {
        .close, .ping, .pong => true,
        else => false,
    };
}

fn isReservedOpcode(op: Opcode) bool {
    return switch (op) {
        .continuation, .text, .binary, .close, .ping, .pong => false,
        else => true,
    };
}

const RawFrame = struct {
    opcode: Opcode,
    fin: bool,
    payload: []u8,
};

fn readExact(conn: anytype, buf: []u8) !void {
    var off: usize = 0;
    while (off < buf.len) {
        const n = try conn.read(buf[off..]);
        if (n == 0) return error.WsClosed;
        off += n;
    }
}

fn readFrame(conn: anytype, allocator: std.mem.Allocator) !RawFrame {
    var hdr: [2]u8 = undefined;
    try readExact(conn, &hdr);

    const fin = (hdr[0] & 0x80) != 0;
    const rsv = (hdr[0] & 0x70) != 0;
    if (rsv) return error.WsProtocol;
    const op_raw: u4 = @intCast(hdr[0] & 0x0F);
    const opcode: Opcode = @enumFromInt(op_raw);
    if (isReservedOpcode(opcode)) return error.WsProtocol;
    const masked = (hdr[1] & 0x80) != 0;
    if (!masked) return error.WsProtocol; // Client→server MUST be masked.
    const len7: u8 = hdr[1] & 0x7F;

    var payload_len: u64 = len7;
    if (len7 == 126) {
        var ext: [2]u8 = undefined;
        try readExact(conn, &ext);
        payload_len = std.mem.readInt(u16, &ext, .big);
    } else if (len7 == 127) {
        var ext: [8]u8 = undefined;
        try readExact(conn, &ext);
        payload_len = std.mem.readInt(u64, &ext, .big);
    }
    if (payload_len > max_message_bytes) return error.WsTooLarge;

    var mask_key: [4]u8 = undefined;
    try readExact(conn, &mask_key);

    const len_usize: usize = @intCast(payload_len);
    const payload = try allocator.alloc(u8, len_usize);
    errdefer allocator.free(payload);
    if (len_usize > 0) try readExact(conn, payload);
    for (payload, 0..) |*b, i| b.* ^= mask_key[i & 3];

    return .{ .opcode = opcode, .fin = fin, .payload = payload };
}

fn writeFrame(conn: anytype, opcode: Opcode, fin: bool, payload: []const u8) !void {
    var hdr_buf: [10]u8 = undefined;
    var off: usize = 0;
    hdr_buf[off] = (if (fin) @as(u8, 0x80) else 0) | @as(u8, @intFromEnum(opcode));
    off += 1;
    if (payload.len <= 125) {
        hdr_buf[off] = @intCast(payload.len);
        off += 1;
    } else if (payload.len <= 0xFFFF) {
        hdr_buf[off] = 126;
        off += 1;
        std.mem.writeInt(u16, hdr_buf[off..][0..2], @intCast(payload.len), .big);
        off += 2;
    } else {
        hdr_buf[off] = 127;
        off += 1;
        std.mem.writeInt(u64, hdr_buf[off..][0..8], payload.len, .big);
        off += 8;
    }
    try conn.writeAllNoFlush(hdr_buf[0..off]);
    if (payload.len > 0) try conn.writeAllNoFlush(payload);
    try conn.flush();
}

// ─── handshake ────────────────────────────────────────────────────────────

/// Compute the `Sec-WebSocket-Accept` value for a given client key.
pub fn computeAccept(out: *[28]u8, key: []const u8) void {
    var hasher = std.crypto.hash.Sha1.init(.{});
    hasher.update(key);
    hasher.update(ws_magic);
    var digest: [20]u8 = undefined;
    hasher.final(&digest);
    const enc = std.base64.standard.Encoder;
    _ = enc.encode(out, &digest);
}

/// Find a header value (case-insensitive name). Returns the value substring
/// (trimmed of leading/trailing whitespace) or null.
pub fn findHeader(headers: []const u8, name: []const u8) ?[]const u8 {
    var lines = std.mem.splitSequence(u8, headers, "\r\n");
    while (lines.next()) |line| {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const lname = line[0..colon];
        if (lname.len != name.len) continue;
        var match = true;
        for (lname, name) |a, b| {
            if (std.ascii.toLower(a) != std.ascii.toLower(b)) {
                match = false;
                break;
            }
        }
        if (!match) continue;
        return std.mem.trim(u8, line[colon + 1 ..], " \t");
    }
    return null;
}

/// Returns true if the given headers represent a WebSocket upgrade request.
pub fn isUpgrade(headers: []const u8) bool {
    const upg = findHeader(headers, "Upgrade") orelse return false;
    const conn = findHeader(headers, "Connection") orelse return false;
    if (std.ascii.findIgnoreCase(upg, "websocket") == null) return false;
    if (std.ascii.findIgnoreCase(conn, "upgrade") == null) return false;
    return true;
}

/// Perform the server-side handshake. Writes the 101 response.
pub fn handshake(conn: anytype, headers: []const u8) !void {
    const key = findHeader(headers, "Sec-WebSocket-Key") orelse return error.WsHandshake;
    var accept_buf: [28]u8 = undefined;
    computeAccept(&accept_buf, key);

    var resp_buf: [256]u8 = undefined;
    const resp = try std.fmt.bufPrint(&resp_buf, "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {s}\r\n\r\n", .{accept_buf});
    try conn.writeAll(resp);
}

// ─── tests ────────────────────────────────────────────────────────────────

const testing = std.testing;

test "computeAccept matches RFC 6455 §1.3 example" {
    var out: [28]u8 = undefined;
    computeAccept(&out, "dGhlIHNhbXBsZSBub25jZQ==");
    try testing.expectEqualStrings("s3pPLMBiTxaQ9kYGzzhZRbK+xOo=", &out);
}

test "findHeader case insensitive" {
    const hdrs = "Host: example.com\r\nUpgrade: websocket\r\nSec-WebSocket-Key: abc123\r\n";
    try testing.expectEqualStrings("websocket", findHeader(hdrs, "Upgrade").?);
    try testing.expectEqualStrings("websocket", findHeader(hdrs, "upgrade").?);
    try testing.expectEqualStrings("abc123", findHeader(hdrs, "sec-websocket-key").?);
    try testing.expect(findHeader(hdrs, "missing") == null);
}

test "isUpgrade detects upgrade headers" {
    try testing.expect(isUpgrade("Upgrade: websocket\r\nConnection: Upgrade\r\n"));
    try testing.expect(isUpgrade("Upgrade: WebSocket\r\nConnection: keep-alive, Upgrade\r\n"));
    try testing.expect(!isUpgrade("Upgrade: h2c\r\nConnection: Upgrade\r\n"));
    try testing.expect(!isUpgrade("Connection: keep-alive\r\n"));
}

// MockConn — exercises the framing path without a real socket.
const MockConn = struct {
    read_data: []const u8,
    read_pos: usize = 0,
    write_buf: std.ArrayList(u8) = .empty,
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator, read_data: []const u8) MockConn {
        return .{ .read_data = read_data, .allocator = allocator };
    }

    fn deinit(self: *MockConn) void {
        self.write_buf.deinit(self.allocator);
    }

    pub fn read(self: *MockConn, buf: []u8) !usize {
        const remaining = self.read_data.len - self.read_pos;
        if (remaining == 0) return 0;
        // Simulate short reads to exercise the readExact loop.
        const n = @min(buf.len, @min(remaining, 3));
        @memcpy(buf[0..n], self.read_data[self.read_pos .. self.read_pos + n]);
        self.read_pos += n;
        return n;
    }

    pub fn writeAll(self: *MockConn, data: []const u8) !void {
        try self.write_buf.appendSlice(self.allocator, data);
    }

    pub fn writeAllNoFlush(self: *MockConn, data: []const u8) !void {
        try self.write_buf.appendSlice(self.allocator, data);
    }

    pub fn flush(_: *MockConn) !void {}
};

fn buildClientFrame(allocator: std.mem.Allocator, opcode: Opcode, fin: bool, payload: []const u8) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    try buf.append(allocator, (if (fin) @as(u8, 0x80) else 0) | @as(u8, @intFromEnum(opcode)));
    if (payload.len <= 125) {
        try buf.append(allocator, 0x80 | @as(u8, @intCast(payload.len)));
    } else if (payload.len <= 0xFFFF) {
        try buf.append(allocator, 0x80 | 126);
        var ext: [2]u8 = undefined;
        std.mem.writeInt(u16, &ext, @intCast(payload.len), .big);
        try buf.appendSlice(allocator, &ext);
    } else {
        try buf.append(allocator, 0x80 | 127);
        var ext: [8]u8 = undefined;
        std.mem.writeInt(u64, &ext, payload.len, .big);
        try buf.appendSlice(allocator, &ext);
    }
    const mask_key = [_]u8{ 0xAA, 0xBB, 0xCC, 0xDD };
    try buf.appendSlice(allocator, &mask_key);
    const start = buf.items.len;
    try buf.appendSlice(allocator, payload);
    for (buf.items[start..], 0..) |*b, i| b.* ^= mask_key[i & 3];
    return try buf.toOwnedSlice(allocator);
}

test "round-trip a small text frame" {
    const a = testing.allocator;
    const frame_bytes = try buildClientFrame(a, .text, true, "hello");
    defer a.free(frame_bytes);

    var mock = MockConn.init(a, frame_bytes);
    defer mock.deinit();
    var ws = WsConn(MockConn).init(&mock);
    defer ws.deinit(a);

    const msg = try ws.readMessage(a);
    defer a.free(msg.payload);
    try testing.expectEqual(Opcode.text, msg.opcode);
    try testing.expectEqualStrings("hello", msg.payload);
}

test "fragmented text reassembles" {
    const a = testing.allocator;
    const f1 = try buildClientFrame(a, .text, false, "hel");
    defer a.free(f1);
    const f2 = try buildClientFrame(a, .continuation, true, "lo!");
    defer a.free(f2);
    const all = try std.mem.concat(a, u8, &.{ f1, f2 });
    defer a.free(all);

    var mock = MockConn.init(a, all);
    defer mock.deinit();
    var ws = WsConn(MockConn).init(&mock);
    defer ws.deinit(a);

    const msg = try ws.readMessage(a);
    defer a.free(msg.payload);
    try testing.expectEqual(Opcode.text, msg.opcode);
    try testing.expectEqualStrings("hello!", msg.payload);
}

test "ping interleaved during fragmented data is reported separately" {
    const a = testing.allocator;
    const f1 = try buildClientFrame(a, .text, false, "ab");
    defer a.free(f1);
    const ping = try buildClientFrame(a, .ping, true, "p");
    defer a.free(ping);
    const f2 = try buildClientFrame(a, .continuation, true, "cd");
    defer a.free(f2);
    const all = try std.mem.concat(a, u8, &.{ f1, ping, f2 });
    defer a.free(all);

    var mock = MockConn.init(a, all);
    defer mock.deinit();
    var ws = WsConn(MockConn).init(&mock);
    defer ws.deinit(a);

    const m1 = try ws.readMessage(a);
    defer a.free(m1.payload);
    try testing.expectEqual(Opcode.ping, m1.opcode);
    try testing.expectEqualStrings("p", m1.payload);

    const m2 = try ws.readMessage(a);
    defer a.free(m2.payload);
    try testing.expectEqual(Opcode.text, m2.opcode);
    try testing.expectEqualStrings("abcd", m2.payload);
}

test "writeText emits unmasked frame with correct length encoding" {
    const a = testing.allocator;
    var mock = MockConn.init(a, "");
    defer mock.deinit();
    var ws = WsConn(MockConn).init(&mock);
    defer ws.deinit(a);

    try ws.writeText("hi");
    try testing.expectEqual(@as(usize, 4), mock.write_buf.items.len);
    try testing.expectEqual(@as(u8, 0x81), mock.write_buf.items[0]);
    try testing.expectEqual(@as(u8, 2), mock.write_buf.items[1]);
    try testing.expectEqualStrings("hi", mock.write_buf.items[2..]);
}

test "writeText with 16-bit length boundary" {
    const a = testing.allocator;
    var mock = MockConn.init(a, "");
    defer mock.deinit();
    var ws = WsConn(MockConn).init(&mock);
    defer ws.deinit(a);

    const payload = try a.alloc(u8, 200);
    defer a.free(payload);
    @memset(payload, 'x');
    try ws.writeText(payload);
    try testing.expectEqual(@as(u8, 0x81), mock.write_buf.items[0]);
    try testing.expectEqual(@as(u8, 126), mock.write_buf.items[1]);
    const len = std.mem.readInt(u16, mock.write_buf.items[2..4], .big);
    try testing.expectEqual(@as(u16, 200), len);
}

test "unmasked client frame is rejected" {
    const a = testing.allocator;
    // FIN=1, text, no mask bit, len=2, payload "hi"
    const bytes = [_]u8{ 0x81, 0x02, 'h', 'i' };
    var mock = MockConn.init(a, &bytes);
    defer mock.deinit();
    var ws = WsConn(MockConn).init(&mock);
    defer ws.deinit(a);

    try testing.expectError(error.WsProtocol, ws.readMessage(a));
}

test "fragmented control frame is rejected" {
    const a = testing.allocator;
    // ping with FIN=0
    const bytes = try buildClientFrame(a, .ping, false, "x");
    defer a.free(bytes);
    var mock = MockConn.init(a, bytes);
    defer mock.deinit();
    var ws = WsConn(MockConn).init(&mock);
    defer ws.deinit(a);

    try testing.expectError(error.WsProtocol, ws.readMessage(a));
}
