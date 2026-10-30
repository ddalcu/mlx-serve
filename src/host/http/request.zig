//! connection thread, JSON depth, readRequest; header helpers

const std = @import("std");
const log = @import("../../sdk/util/log.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const nowMsMonotonic = io_util.nowMsMonotonic;
const testing = std.testing;
const http_server = @import("server.zig");

const Conn = @import("conn.zig").Conn;
const max_request_bytes = @import("router.zig").max_request_bytes;
const maxRequestBytesFor = @import("router.zig").maxRequestBytesFor;
const payloadTooLargeMessage = @import("router.zig").payloadTooLargeMessage;
const handleConnection = @import("router.zig").handleConnection;
const logSafePath = @import("respond.zig").logSafePath;
const sendErrorResponse = @import("errors.zig").sendErrorResponse;

/// Per-connection thread arguments. Heap-allocated so the spawned thread
/// owns its lifetime; freed in `handleConnectionThread` after the handler
/// returns. The accepted stream is moved into a stack-allocated `Conn`
/// inside the thread.
pub const ConnThreadArgs = struct {
    allocator: std.mem.Allocator,
    accepted_stream: std.Io.net.Stream,
    io: std.Io,
};

pub fn handleConnectionThread(args: *ConnThreadArgs) void {
    // Decrement LAST (declared first → runs after every other defer), so the
    // shutdown drain in `serve` only sees this thread leave once it has fully
    // returned from `complete()` and closed its socket.
    defer _ = http_server.active_conn_threads.fetchSub(1, .acq_rel);
    var conn: Conn = undefined;
    Conn.init(&conn, args.accepted_stream, args.io);
    defer conn.close();
    handleConnection(args.allocator, &conn) catch |err| {
        switch (err) {
            error.WriteFailed, error.ReadFailed => {
                // error.WriteFailed/ReadFailed collapses BrokenPipe + ConnectionResetByPeer
                // + other low-level errors. Surface the actual cause from write_state.err /
                // read_state.err so debug logs distinguish "client hung up" from real bugs.
                if (err == error.WriteFailed) {
                    if (conn.write_state.err) |we| {
                        log.debug("  -> client disconnected (write: {s})\n", .{@errorName(we)});
                    } else {
                        log.debug("  -> client disconnected (write)\n", .{});
                    }
                } else {
                    if (conn.read_state.err) |re| {
                        log.debug("  -> client disconnected (read: {s})\n", .{@errorName(re)});
                    } else {
                        log.debug("  -> client disconnected (read)\n", .{});
                    }
                }
            },
            else => log.err("  -> error: {}\n", .{err}),
        }
    };
    args.allocator.destroy(args);
}

/// Deepest JSON nesting a request may carry. Tool schemas and template kwargs are walked and
/// re-serialized recursively on a connection thread's stack: 50000 levels overflowed it.
pub const MAX_JSON_DEPTH = 4096;

/// Deepest `[`/`{` nesting in `body`, brackets inside strings skipped. A plain byte scan, so it
/// also bounds bodies the JSON parser would reject.
pub fn jsonNestingDepth(body: []const u8) usize {
    var depth: usize = 0;
    var max: usize = 0;
    var in_str = false;
    var i: usize = 0;
    while (i < body.len) : (i += 1) {
        const c = body[i];
        if (in_str) {
            if (c == '\\') i += 1 else if (c == '"') in_str = false;
            continue;
        }
        switch (c) {
            '"' => in_str = true,
            '[', '{' => {
                depth += 1;
                max = @max(max, depth);
            },
            ']', '}' => depth -|= 1,
            else => {},
        }
    }
    return max;
}

const RawRequest = struct {
    /// Owned; `buf[0..len]` is the head plus as much of the body as arrived.
    buf: []u8,
    len: usize,
    header_end: usize,
};

/// Reads one request: the head (up to 16 KB), then its `Content-Length` body. Null when no
/// complete head arrived, the client stalled, or the body is over the route's cap (the 413 is
/// already sent).
/// Bytes past the declared body are dropped: one request per connection.
pub fn readRequest(allocator: std.mem.Allocator, stream: *Conn) !?RawRequest {
    return readRequestWithin(allocator, stream, REQUEST_READ_LIMIT_MS);
}

/// A client gets this long between bytes of its request, and in all for its head: an idle or
/// dripping connection otherwise holds its thread forever.
const REQUEST_READ_LIMIT_MS: i64 = 30_000;

fn readRequestWithin(allocator: std.mem.Allocator, stream: *Conn, limit_ms: i64) !?RawRequest {
    const head_deadline = nowMsMonotonic(stream.io) + limit_ms;
    var hdr_buf: [16 * 1024]u8 = undefined;
    var total_read: usize = 0;
    var content_length: ?usize = null;
    var header_end_pos: usize = 0;

    while (total_read < hdr_buf.len) {
        if (!stream.awaitReadable(head_deadline - nowMsMonotonic(stream.io))) return null;
        const n = try stream.read(hdr_buf[total_read..]);
        if (n == 0) break;
        total_read += n;

        if (std.mem.indexOf(u8, hdr_buf[0..total_read], "\r\n\r\n")) |he| {
            header_end_pos = he + 4;
            content_length = findContentLength(hdr_buf[0..he]);
            break;
        }
    }
    if (header_end_pos == 0) return null;

    const total_size = header_end_pos +| (content_length orelse 0);
    // The cap is per ROUTE, so peek the path off the request line we already
    // have — media bodies carry base64 frames and dwarf any JSON chat body.
    const req_path = blk: {
        const line_end = std.mem.indexOf(u8, hdr_buf[0..header_end_pos], "\r\n") orelse break :blk "";
        var it = std.mem.splitScalar(u8, hdr_buf[0..line_end], ' ');
        _ = it.next();
        break :blk it.next() orelse "";
    };
    const max_request_size = maxRequestBytesFor(req_path);
    if (total_size > max_request_size) {
        var msg_buf: [128]u8 = undefined;
        const msg = payloadTooLargeMessage(&msg_buf, total_size, max_request_size);
        log.warn("[http] 413 {s}: {s}\n", .{ logSafePath(req_path), msg });
        try sendErrorResponse(allocator, stream, "413 Payload Too Large", "invalid_request_error", msg, 413);
        // Closing with body bytes unread resets the connection, and the client
        // can lose the 413 before reading it: read what is left, up to the chat cap.
        var left = @min(total_size, max_request_bytes) -| total_read;
        while (left > 0 and stream.awaitReadable(limit_ms)) {
            const n = stream.read(hdr_buf[0..@min(left, hdr_buf.len)]) catch break;
            if (n == 0) break;
            left -= n;
        }
        return null;
    }

    // curl asks before a body over 1 MB and otherwise waits a second for the answer.
    if (total_read < total_size) if (findHeaderValue(hdr_buf[0 .. header_end_pos - 4], "expect")) |v| {
        if (std.ascii.eqlIgnoreCase(v, "100-continue")) {
            try stream.writeAll("HTTP/1.1 100 Continue\r\n\r\n");
            try stream.flush();
        }
    };

    const buf = try allocator.alloc(u8, total_size);
    errdefer allocator.free(buf);
    total_read = @min(total_read, total_size);
    @memcpy(buf[0..total_read], hdr_buf[0..total_read]);
    while (total_read < total_size) {
        if (!stream.awaitReadable(limit_ms)) {
            allocator.free(buf);
            return null;
        }
        const n = try stream.read(buf[total_read..total_size]);
        // The client hung up mid-body (an aborted upload): there is no request to answer.
        if (n == 0) {
            allocator.free(buf);
            return null;
        }
        total_read += n;
    }
    return .{ .buf = buf, .len = total_read, .header_end = header_end_pos };
}

fn findContentLength(headers: []const u8) ?usize {
    var lines = std.mem.splitSequence(u8, headers, "\r\n");
    while (lines.next()) |line| {
        const lower = "content-length: ";
        if (line.len >= lower.len) {
            var match = true;
            for (0..lower.len) |j| {
                if (std.ascii.toLower(line[j]) != lower[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return std.fmt.parseInt(usize, std.mem.trim(u8, line[lower.len..], " "), 10) catch null;
            }
        }
    }
    return null;
}

/// Case-insensitive header lookup returning the trimmed VALUE (`Content-Type`
/// carries the multipart boundary, so `/v1/images/edits` can't be parsed from
/// the body alone like every other endpoint).
pub fn findHeaderValue(headers: []const u8, comptime name_lower: []const u8) ?[]const u8 {
    var lines = std.mem.splitSequence(u8, headers, "\r\n");
    while (lines.next()) |line| {
        if (line.len <= name_lower.len or line[name_lower.len] != ':') continue;
        var match = true;
        for (0..name_lower.len) |j| {
            if (std.ascii.toLower(line[j]) != name_lower[j]) {
                match = false;
                break;
            }
        }
        if (match) return std.mem.trim(u8, line[name_lower.len + 1 ..], " \t");
    }
    return null;
}

// ── API-key auth helpers (used only when --api-key / g_api_key is set) ──

// ── tests ──

/// Sends `request` (then EOF) over a socketpair and runs `readRequest` on the server end;
/// returns the body it read, or null, plus whatever the server wrote back.
pub fn readRequestOver(request: []const u8, body_out: []u8, reply_out: []u8) !struct { body: ?[]const u8, reply: []const u8 } {
    var sv: [2]std.posix.fd_t = undefined;
    try testing.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
    defer _ = std.c.close(sv[1]);
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, testing.io);
    defer conn.close();
    try testing.expectEqual(@as(isize, @intCast(request.len)), std.c.send(sv[1], request.ptr, request.len, 0));
    _ = std.c.shutdown(sv[1], 1);

    const raw = try readRequest(testing.allocator, &conn);
    try conn.flush();
    const got = std.c.recv(sv[1], reply_out.ptr, reply_out.len, std.c.MSG.DONTWAIT);
    const reply = reply_out[0..@intCast(@max(got, 0))];
    const r = raw orelse return .{ .body = null, .reply = reply };
    defer testing.allocator.free(r.buf);
    const body = r.buf[r.header_end..r.len];
    @memcpy(body_out[0..body.len], body);
    return .{ .body = body_out[0..body.len], .reply = reply };
}

/// Runs `readRequestWithin(limit_ms)` against a client that sends `sent`, then one byte per
/// `drip_ms` (0 = nothing more), and never finishes; true when the reader gave up on its own.
fn readerGivesUp(sent: []const u8, limit_ms: i64, drip_ms: u64) !bool {
    var sv: [2]std.posix.fd_t = undefined;
    try testing.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, testing.io);
    defer conn.close();
    _ = std.c.send(sv[1], sent.ptr, sent.len, 0);

    var done = std.atomic.Value(bool).init(false);
    const reader = try std.Thread.spawn(.{}, struct {
        fn run(c: *Conn, limit: i64, d: *std.atomic.Value(bool)) void {
            if (readRequestWithin(testing.allocator, c, limit) catch null) |r| testing.allocator.free(r.buf);
            d.store(true, .release);
        }
    }.run, .{ &conn, limit_ms, &done });
    var waited: u64 = 0;
    while (!done.load(.acquire) and waited < 2000) : (waited += 10) {
        std.Io.sleep(testing.io, .fromMilliseconds(10), .real) catch {};
        if (drip_ms > 0 and waited % drip_ms == 0) _ = std.c.send(sv[1], "x", 1, 0);
    }
    const gave_up = done.load(.acquire);
    _ = std.c.close(sv[1]);
    reader.join();
    return gave_up;
}

test "readRequest: a client that stalls its head or body, or drips its head, is dropped, not held forever" {
    const head = "POST /v1/chat/completions HTTP/1.1\r\n";
    try testing.expect(try readerGivesUp(head, 200, 0));
    try testing.expect(try readerGivesUp(head, 300, 50));
    try testing.expect(try readerGivesUp(head ++ "Content-Length: 100\r\n\r\n{}", 200, 0));
}

test "readRequest: bytes past Content-Length are dropped, a short body is no request, a wrapping Content-Length is a 413" {
    var body_buf: [64]u8 = undefined;
    var reply_buf: [1024]u8 = undefined;
    const extra: [4000]u8 = @splat('x');
    const short = try readRequestOver("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}" ++ extra, &body_buf, &reply_buf);
    try testing.expectEqualStrings("{}", short.body.?);

    const none = try readRequestOver("GET /health HTTP/1.1\r\n\r\n" ++ extra, &body_buf, &reply_buf);
    try testing.expectEqualStrings("", none.body.?);

    // A client that hangs up mid-body sent no request: nothing is parsed, nothing answered.
    const cut = try readRequestOver("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 100\r\n\r\n{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hel", &body_buf, &reply_buf);
    try testing.expect(cut.body == null);
    try testing.expectEqualStrings("", cut.reply);

    const huge = try readRequestOver("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 18446744073709551615\r\n\r\n{}", &body_buf, &reply_buf);
    try testing.expect(huge.body == null);
    try testing.expect(std.mem.startsWith(u8, huge.reply, "HTTP/1.1 413"));
}

test "jsonNestingDepth: brackets inside strings never count" {
    try testing.expectEqual(@as(usize, 0), jsonNestingDepth("42"));
    try testing.expectEqual(@as(usize, 3), jsonNestingDepth("{\"a\":[{\"b\":1}]}"));
    try testing.expectEqual(@as(usize, 1), jsonNestingDepth("{\"a\":\"[[[{{{\"}"));
    try testing.expectEqual(@as(usize, 1), jsonNestingDepth("[\"\\\"[[\"]")); // escaped quote stays in the string
    const open: [5000]u8 = @splat('[');
    const close: [5000]u8 = @splat(']');
    try testing.expectEqual(@as(usize, 5000), jsonNestingDepth(&(open ++ close)));
}

test "findContentLength parses header" {
    try testing.expectEqual(@as(?usize, 42), findContentLength("Host: localhost\r\nContent-Length: 42\r\nAccept: */*"));
}

test "findContentLength case insensitive" {
    try testing.expectEqual(@as(?usize, 100), findContentLength("content-length: 100"));
    try testing.expectEqual(@as(?usize, 100), findContentLength("Content-Length: 100"));
    try testing.expectEqual(@as(?usize, 100), findContentLength("CONTENT-LENGTH: 100"));
}

test "findContentLength returns null when missing" {
    try testing.expect(findContentLength("Host: localhost\r\nAccept: */*") == null);
    try testing.expect(findContentLength("") == null);
}
