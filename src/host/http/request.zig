//! connection thread, JSON depth, readRequest; header helpers

const std = @import("std");
const log = @import("../../sdk/util/log.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const nowMsMonotonic = io_util.nowMsMonotonic;
const testing = std.testing;
const http_server = @import("server.zig");
const mlx = @import("../../mlx/ffi.zig");
const memory_ceiling = @import("../registry/memory_ceiling.zig");

const Conn = @import("conn.zig").Conn;
const max_request_bytes = @import("router.zig").max_request_bytes;
const maxRequestBytesFor = @import("router.zig").maxRequestBytesFor;
const payloadTooLargeMessage = @import("router.zig").payloadTooLargeMessage;
const handleConnection = @import("router.zig").handleConnection;
const logSafePath = @import("respond.zig").logSafePath;
const sendErrorResponse = @import("errors.zig").sendErrorResponse;
const sendErrorResponseWithReason = @import("errors.zig").sendErrorResponseWithReason;
const auth = @import("auth.zig");

pub var import_headroom_override: ?u64 = null;

/// Per-connection thread arguments. Heap-allocated so the spawned thread
/// owns its lifetime; freed in `handleConnectionThread` after the handler
/// returns. The accepted stream is moved into a stack-allocated `Conn`
/// inside the thread.
pub const ConnThreadArgs = struct {
    allocator: std.mem.Allocator,
    accepted_stream: std.Io.net.Stream,
    io: std.Io,
};

pub const RawRequest = struct {
    buf: []u8,
    len: usize,
    header_end: usize,
    header_read_ms: i64,

    pub fn deinit(self: RawRequest, allocator: std.mem.Allocator) void {
        const line_end = std.mem.indexOf(u8, self.buf[0..self.header_end], "\r\n") orelse self.header_end;
        var it = std.mem.splitScalar(u8, self.buf[0..line_end], ' ');
        _ = it.next();
        const target = it.next() orelse "";
        const path = target[0 .. std.mem.indexOfScalar(u8, target, '?') orelse target.len];
        if (std.mem.eql(u8, path, "/v1/cache/import")) {
            http_server.releaseCacheImportBytes(self.len - self.header_end);
        }
        allocator.free(self.buf);
    }
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

/// A request's head, read before anything is allocated for its body.
pub const RequestHead = struct {
    /// `buf[0..len]` is the head plus whatever body bytes arrived with it.
    buf: [16 * 1024]u8 = undefined,
    len: usize = 0,
    end: usize = 0,
    content_length: usize = 0,
    header_read_ms: i64 = 0,

    pub fn bytes(self: *const RequestHead) []const u8 {
        return self.buf[0..self.end];
    }

    fn totalSize(self: *const RequestHead) usize {
        return self.end +| self.content_length;
    }
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
pub const REQUEST_READ_LIMIT_MS: i64 = 30_000;

fn readRequestWithin(allocator: std.mem.Allocator, stream: *Conn, limit_ms: i64) !?RawRequest {
    var head: RequestHead = .{};
    if (!try readHead(stream, &head, limit_ms)) return null;
    return readBody(allocator, stream, &head, limit_ms);
}

/// False when no complete head arrived in time.
pub fn readHead(stream: *Conn, head: *RequestHead, limit_ms: i64) !bool {
    const head_deadline = nowMsMonotonic(stream.io) + limit_ms;
    while (head.len < head.buf.len) {
        if (!stream.awaitReadable(head_deadline - nowMsMonotonic(stream.io))) return false;
        const n = try stream.read(head.buf[head.len..]);
        if (n == 0) break;
        head.len += n;

        if (std.mem.indexOf(u8, head.buf[0..head.len], "\r\n\r\n")) |he| {
            head.end = he + 4;
            head.header_read_ms = nowMsMonotonic(stream.io);
            head.content_length = findContentLength(head.buf[0..he]) orelse 0;
            return true;
        }
    }
    return false;
}

/// The rest of `head`'s request. Null when the client stalled or hung up, or the body is over
/// the route's cap (the 413 is already sent).
pub fn readBody(allocator: std.mem.Allocator, stream: *Conn, head: *RequestHead, limit_ms: i64) !?RawRequest {
    const header_end_pos = head.end;
    const total_size = head.totalSize();
    // The cap is per ROUTE, so peek the path off the request line we already
    // have — media bodies carry base64 frames and dwarf any JSON chat body.
    const req_line: struct { method: []const u8, path: []const u8 } = blk: {
        const line_end = std.mem.indexOf(u8, head.bytes(), "\r\n") orelse break :blk .{ .method = "", .path = "" };
        var it = std.mem.splitScalar(u8, head.buf[0..line_end], ' ');
        const method = it.next() orelse "";
        const path = it.next() orelse "";
        break :blk .{ .method = method, .path = path };
    };
    const req_method = req_line.method;
    const req_path = req_line.path;
    const max_request_size = maxRequestBytesFor(req_path);
    const cache_import = isCacheImportPath(req_path);
    const headers = head.buf[0 .. header_end_pos - 4];
    const cache_export = if (findHeaderValue(headers, "x-cache-export")) |value| std.mem.eql(u8, value, "1") else false;
    if (cache_import or cache_export) {
        const verdict = auth.checkCacheTransferAuth(http_server.g_cache_transfer, stream.stream.socket.address, headers, req_path);
        if (verdict != .allowed) {
            try auth.sendCacheTransferDenial(allocator, stream, verdict, cache_import);
            return null;
        }
        if (cache_import and !std.mem.eql(u8, req_method, "POST")) {
            try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "Unknown endpoint", 404);
            return null;
        }
    }
    const content_bytes = head.content_length;
    const transfer_body_too_large = cache_import and content_bytes > @import("router.zig").max_cache_transfer_body_bytes;
    if (total_size > max_request_size or transfer_body_too_large) {
        var msg_buf: [128]u8 = undefined;
        const got = if (transfer_body_too_large) content_bytes else total_size;
        const cap = if (transfer_body_too_large) @import("router.zig").max_cache_transfer_body_bytes else max_request_size;
        const msg = payloadTooLargeMessage(&msg_buf, got, cap);
        log.warn("[http] 413 {s}: {s}\n", .{ logSafePath(req_path), msg });
        if (cache_import) {
            try sendErrorResponseWithReason(allocator, stream, "413 Payload Too Large", "invalid_request_error", msg, 413, "cache_transfer_too_large");
        } else {
            try sendErrorResponse(allocator, stream, "413 Payload Too Large", "invalid_request_error", msg, 413);
        }
        if (!transfer_body_too_large) drainBody(stream, head, limit_ms);
        return null;
    }

    const reserved_bytes: u64 = if (cache_import) @intCast(head.content_length) else 0;
    if (reserved_bytes > 0 and !http_server.reserveCacheImportBytes(reserved_bytes)) {
        try sendErrorResponseWithReason(allocator, stream, "503 Service Unavailable", "insufficient_memory", "cache import refused: inbound transfer byte budget is full", 503, "cache_import_busy");
        return null;
    }
    var reservation_in_reader = reserved_bytes > 0;
    defer if (reservation_in_reader) http_server.releaseCacheImportBytes(reserved_bytes);

    if (cache_import) {
        if (findHeaderValue(headers, "x-cache-import-peak")) |value| {
            const peak = Conn.parsePositiveDecimal(value) orelse {
                try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "X-Cache-Import-Peak must be a positive decimal integer", 400, "cache_import_peak_invalid");
                return null;
            };
            var qmodel_buf: [512]u8 = undefined;
            const qmodel = @import("../api/request_model.zig").queryModel(&qmodel_buf, req_path);
            if (!cacheImportPeakFits(stream.io, peak, reserved_bytes, qmodel)) {
                try sendErrorResponseWithReason(allocator, stream, "503 Service Unavailable", "insufficient_memory", "cache import refused: source planned peak exceeds current headroom", 503, "cache_import_peak_no_memory");
                return null;
            }
        }
    }

    // curl asks before a body over 1 MB and otherwise waits a second for the answer.
    if (head.len < total_size) if (findHeaderValue(head.buf[0 .. header_end_pos - 4], "expect")) |v| {
        if (std.ascii.eqlIgnoreCase(v, "100-continue")) {
            try stream.writeAll("HTTP/1.1 100 Continue\r\n\r\n");
            try stream.flush();
        }
    };

    const buf = allocator.alloc(u8, total_size) catch |err| {
        if (cache_import) try sendErrorResponseWithReason(allocator, stream, "503 Service Unavailable", "insufficient_memory", "cache import refused: request body allocation failed", 503, "cache_import_no_memory");
        return err;
    };
    errdefer allocator.free(buf);
    var total_read = @min(head.len, total_size);
    @memcpy(buf[0..total_read], head.buf[0..total_read]);
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
    reservation_in_reader = false;
    return .{ .buf = buf, .len = total_read, .header_end = header_end_pos, .header_read_ms = head.header_read_ms };
}

fn isCacheImportPath(target: []const u8) bool {
    const path = target[0 .. std.mem.indexOfScalar(u8, target, '?') orelse target.len];
    return std.mem.eql(u8, path, "/v1/cache/import");
}

pub fn cacheImportPeakFits(io: std.Io, peak: u64, already_reserved: u64, requested_model: ?[]const u8) bool {
    if (import_headroom_override) |headroom| return peak <= headroom;
    const scheduler = http_server.launch().global_scheduler orelse return true;
    const reg = scheduler.registry;
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    const ceiling: u64 = blk: {
        const entry = if (requested_model) |req_id|
            (reg.resolveEntry(req_id) catch null)
        else
            null;
        reg.mutex.lockUncancelable(io);
        defer reg.mutex.unlock(io);
        const model = entry orelse scheduler.current_model orelse return true;
        if (model.state != .ready) return true;
        const config = model.config orelse return true;
        break :blk memory_ceiling.currentGpuMemoryCeiling(config, active_mem);
    };
    const other_reserved = http_server.cacheTransferReservedBytes() -| already_reserved;
    const headroom = ceiling -| active_mem -| other_reserved;
    const reclaimable = scheduler.resident_hot_cache_bytes.load(.monotonic);
    return peak <= headroom +| reclaimable;
}

/// Closing with body bytes unread resets the connection, and the client can lose a refusal
/// before reading it: read what is left of the body into nothing, up to the chat cap.
pub fn drainBody(stream: *Conn, head: *const RequestHead, limit_ms: i64) void {
    var scratch: [16 * 1024]u8 = undefined;
    var left = @min(head.totalSize(), max_request_bytes) -| head.len;
    while (left > 0 and stream.awaitReadable(limit_ms)) {
        const n = stream.read(scratch[0..@min(left, scratch.len)]) catch break;
        if (n == 0) break;
        left -= n;
    }
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

pub const ReadOverResult = struct { body: ?[]const u8, reply: []const u8 };

/// Sends `request` (then EOF) over a socketpair and runs `readRequest` on the server end;
/// returns the body it read, or null, plus whatever the server wrote back.
pub fn readRequestOver(request: []const u8, body_out: []u8, reply_out: []u8) !ReadOverResult {
    return readRequestOverFrom(request, body_out, reply_out, .{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 0 } });
}

fn readRequestOverFrom(
    request: []const u8,
    body_out: []u8,
    reply_out: []u8,
    peer: std.Io.net.IpAddress,
) !ReadOverResult {
    return readRequestOverFromWithAllocator(request, body_out, reply_out, peer, testing.allocator);
}

fn readRequestOverFromWithAllocator(
    request: []const u8,
    body_out: []u8,
    reply_out: []u8,
    peer: std.Io.net.IpAddress,
    allocator: std.mem.Allocator,
) !ReadOverResult {
    var sv: [2]std.posix.fd_t = undefined;
    try testing.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
    defer _ = std.c.close(sv[1]);
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = peer } }, testing.io);
    defer conn.close();
    try testing.expectEqual(@as(isize, @intCast(request.len)), std.c.send(sv[1], request.ptr, request.len, 0));
    _ = std.c.shutdown(sv[1], 1);

    const raw = try readRequest(allocator, &conn);
    try conn.flush();
    const got = std.c.recv(sv[1], reply_out.ptr, reply_out.len, std.c.MSG.DONTWAIT);
    const reply = reply_out[0..@intCast(@max(got, 0))];
    const r = raw orelse return .{ .body = null, .reply = reply };
    defer r.deinit(allocator);
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
            if (readRequestWithin(testing.allocator, c, limit) catch null) |r| r.deinit(testing.allocator);
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

test "cache import disabled returns 404 before body allocation" {
    const prev_ct = http_server.g_cache_transfer;
    defer http_server.g_cache_transfer = prev_ct;
    http_server.g_cache_transfer = .disabled;
    var body_buf: [1]u8 = undefined;
    var reply_buf: [1024]u8 = undefined;
    var allocation_buf: [64 * 1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&allocation_buf);
    const lo: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 0 } };
    const result = try readRequestOverFromWithAllocator(
        "POST /v1/cache/import HTTP/1.1\r\nContent-Length: 1073741824\r\nExpect: 100-continue\r\n\r\n",
        &body_buf,
        &reply_buf,
        lo,
        fba.allocator(),
    );
    try testing.expect(result.body == null);
    try testing.expect(std.mem.startsWith(u8, result.reply, "HTTP/1.1 404 Not Found"));
    try testing.expect(std.mem.indexOf(u8, result.reply, "100 Continue") == null);
    try testing.expectEqual(@as(u64, 0), http_server.serverState().cache_transfer_inbound_reserved.load(.acquire));
}

test "cache import above 4 GiB is refused at the header before body allocation" {
    const prev_ct = http_server.g_cache_transfer;
    defer http_server.g_cache_transfer = prev_ct;
    http_server.g_cache_transfer = .loopback;
    var body_buf: [1]u8 = undefined;
    var reply_buf: [1024]u8 = undefined;
    const result = try readRequestOver("POST /v1/cache/import HTTP/1.1\r\nContent-Length: 4294967297\r\nExpect: 100-continue\r\n\r\n", &body_buf, &reply_buf);
    try testing.expect(result.body == null);
    try testing.expect(std.mem.startsWith(u8, result.reply, "HTTP/1.1 413"));
    try testing.expect(std.mem.indexOf(u8, result.reply, "cache_transfer_too_large") != null);
    try testing.expect(std.mem.indexOf(u8, result.reply, "100 Continue") == null);
    try testing.expectEqual(@as(u64, 0), http_server.serverState().cache_transfer_inbound_reserved.load(.acquire));
}

test "cache import trust: an untrusted peer is refused before a 1 GiB reservation" {
    const prev_ct = http_server.g_cache_transfer;
    defer http_server.g_cache_transfer = prev_ct;
    http_server.g_cache_transfer = .loopback;
    var body_buf: [1]u8 = undefined;
    var reply_buf: [1024]u8 = undefined;
    var allocation_buf: [64 * 1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&allocation_buf);
    const peer: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 192, 168, 1, 9 }, .port = 0 } };
    const result = try readRequestOverFromWithAllocator(
        "POST /v1/cache/import HTTP/1.1\r\nContent-Length: 1073741824\r\nExpect: 100-continue\r\n\r\n",
        &body_buf,
        &reply_buf,
        peer,
        fba.allocator(),
    );
    try testing.expect(result.body == null);
    try testing.expect(std.mem.startsWith(u8, result.reply, "HTTP/1.1 403 Forbidden"));
    try testing.expect(std.mem.indexOf(u8, result.reply, "loopback peer or --cache-transfer-key-file") != null);
    try testing.expect(std.mem.indexOf(u8, result.reply, "100 Continue") == null);
    try testing.expectEqual(@as(u64, 0), http_server.serverState().cache_transfer_inbound_reserved.load(.acquire));
}

test "cache import trust: key mode refuses wrong key before reservation and accepts right key from any IP" {
    const prev_ct = http_server.g_cache_transfer;
    defer http_server.g_cache_transfer = prev_ct;
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("correct-transfer-key-01234567890123", &digest, .{});
    http_server.g_cache_transfer = .{ .key = digest };
    var body_buf: [1]u8 = undefined;
    var reply_buf: [1024]u8 = undefined;
    var allocation_buf: [64 * 1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&allocation_buf);
    const peer: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 192, 168, 1, 9 }, .port = 0 } };
    const refused = try readRequestOverFromWithAllocator(
        "POST /v1/cache/import HTTP/1.1\r\nContent-Length: 1073741824\r\nAuthorization: Bearer wrong-key\r\nExpect: 100-continue\r\n\r\n",
        &body_buf,
        &reply_buf,
        peer,
        fba.allocator(),
    );
    try testing.expect(refused.body == null);
    try testing.expect(std.mem.startsWith(u8, refused.reply, "HTTP/1.1 401 Unauthorized"));
    try testing.expect(std.mem.indexOf(u8, refused.reply, "100 Continue") == null);
    try testing.expectEqual(@as(u64, 0), http_server.serverState().cache_transfer_inbound_reserved.load(.acquire));

    const ok = try readRequestOverFrom(
        "POST /v1/cache/import HTTP/1.1\r\nContent-Length: 1\r\nAuthorization: Bearer correct-transfer-key-01234567890123\r\nExpect: 100-continue\r\n\r\n",
        &body_buf,
        &reply_buf,
        peer,
    );
    try testing.expect(ok.body == null);
    try testing.expect(std.mem.startsWith(u8, ok.reply, "HTTP/1.1 100 Continue"));
    try testing.expectEqual(@as(u64, 0), http_server.serverState().cache_transfer_inbound_reserved.load(.acquire));
}

test "cache export trust: an untrusted peer is refused before chat body allocation" {
    const prev_ct = http_server.g_cache_transfer;
    defer http_server.g_cache_transfer = prev_ct;
    http_server.g_cache_transfer = .loopback;
    var body_buf: [1]u8 = undefined;
    var reply_buf: [1024]u8 = undefined;
    var allocation_buf: [64 * 1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&allocation_buf);
    const peer: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 192, 168, 1, 9 }, .port = 0 } };
    const result = try readRequestOverFromWithAllocator(
        "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 1048576\r\nX-Cache-Export: 1\r\nExpect: 100-continue\r\n\r\n",
        &body_buf,
        &reply_buf,
        peer,
        fba.allocator(),
    );
    try testing.expect(result.body == null);
    try testing.expect(std.mem.startsWith(u8, result.reply, "HTTP/1.1 403 Forbidden"));
    try testing.expect(std.mem.indexOf(u8, result.reply, "100 Continue") == null);
}

test "cache import byte reservations reject an aggregate overflow before allocating the next body" {
    const prev_ct = http_server.g_cache_transfer;
    defer http_server.g_cache_transfer = prev_ct;
    http_server.g_cache_transfer = .loopback;
    const held: u64 = http_server.CACHE_TRANSFER_BUDGET_BYTES - (1 << 30);
    try testing.expect(http_server.reserveCacheImportBytes(held));
    defer http_server.releaseCacheImportBytes(held);

    var body_buf: [1]u8 = undefined;
    var reply_buf: [1024]u8 = undefined;
    const result = try readRequestOver("POST /v1/cache/import HTTP/1.1\r\nContent-Length: 1073741825\r\nExpect: 100-continue\r\n\r\n", &body_buf, &reply_buf);
    try testing.expect(result.body == null);
    try testing.expect(std.mem.startsWith(u8, result.reply, "HTTP/1.1 503"));
    try testing.expect(std.mem.indexOf(u8, result.reply, "cache_import_busy") != null);
    try testing.expect(std.mem.indexOf(u8, result.reply, "100 Continue") == null);
    try testing.expectEqual(held, http_server.serverState().cache_transfer_inbound_reserved.load(.acquire));
}

test "cache import source planned peak refuses before body allocation" {
    const prev_ct = http_server.g_cache_transfer;
    defer http_server.g_cache_transfer = prev_ct;
    http_server.g_cache_transfer = .loopback;
    import_headroom_override = 100;
    defer import_headroom_override = null;
    var body_buf: [1]u8 = undefined;
    var reply_buf: [1024]u8 = undefined;
    const result = try readRequestOver("POST /v1/cache/import HTTP/1.1\r\nContent-Length: 1\r\nX-Cache-Import-Peak: 101\r\nExpect: 100-continue\r\n\r\n", &body_buf, &reply_buf);
    try testing.expect(result.body == null);
    try testing.expect(std.mem.startsWith(u8, result.reply, "HTTP/1.1 503"));
    try testing.expect(std.mem.indexOf(u8, result.reply, "cache_import_peak_no_memory") != null);
    try testing.expect(std.mem.indexOf(u8, result.reply, "100 Continue") == null);
    try testing.expectEqual(@as(u64, 0), http_server.serverState().cache_transfer_inbound_reserved.load(.acquire));
}

test "cache import reservation returns to zero after success and a mid-body disconnect" {
    const prev_ct = http_server.g_cache_transfer;
    defer http_server.g_cache_transfer = prev_ct;
    http_server.g_cache_transfer = .loopback;
    try testing.expectEqual(@as(u64, 0), http_server.serverState().cache_transfer_inbound_reserved.load(.acquire));
    var body_buf: [16]u8 = undefined;
    var reply_buf: [1024]u8 = undefined;
    const complete = try readRequestOver("POST /v1/cache/import HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}", &body_buf, &reply_buf);
    try testing.expectEqualStrings("{}", complete.body.?);
    try testing.expectEqual(@as(u64, 0), http_server.serverState().cache_transfer_inbound_reserved.load(.acquire));

    const disconnected = try readRequestOver("POST /v1/cache/import HTTP/1.1\r\nContent-Length: 100\r\n\r\n{}", &body_buf, &reply_buf);
    try testing.expect(disconnected.body == null);
    try testing.expectEqual(@as(u64, 0), http_server.serverState().cache_transfer_inbound_reserved.load(.acquire));
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
