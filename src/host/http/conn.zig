//! StreamHeartbeat, Conn

const std = @import("std");
const win_io = @import("../../sdk/util/win_io.zig");
const builtin = @import("builtin");
const instr = @import("../metrics/metrics.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const ollama_mod = @import("../api/ollama/translate.zig");
const nowMsMonotonic = io_util.nowMsMonotonic;
const testing = std.testing;
const http_server = @import("server.zig");

const WsBridge = @import("../api/responses/ws.zig").WsBridge;
const socketReadable = @import("listen.zig").socketReadable;
const readRequest = @import("request.zig").readRequest;
const sendResponseFramed = @import("respond.zig").sendResponseFramed;
const SSE_ALLOW_HEADERS_DEFAULT = @import("respond.zig").SSE_ALLOW_HEADERS_DEFAULT;
const sendSseHeaders = @import("respond.zig").sendSseHeaders;
const findHeaderValue = @import("request.zig").findHeaderValue;
const readRequestOver = @import("request.zig").readRequestOver;

/// Tracks how long a streaming connection has been SILENT so the token loops
/// can emit a protocol-level keepalive before a client's idle-body timeout
/// fires. The mirror image of `generate.StallClock`: that one protects the
/// server from a wedged model, this one protects the client from a wedged-
/// looking socket.
///
/// It exists because every streaming surface buffers tokens while it might be
/// looking at a tool call or an unclosed thinking block — during a large
/// tool call (an agent one-shotting a whole file into `write_file`) the
/// handler produces no SSE output at all, sometimes for minutes. Node's
/// `fetch`/undici drops such a body after 300 s (`TypeError: terminated`).
pub const StreamHeartbeat = struct {
    /// MONOTONIC milliseconds (`io_util.nowMsMonotonic`) — an NTP step must not
    /// be able to stall the keepalive, nor spam one per token.
    last_write_ms: i64 = 0,
    interval_ms: i64 = Conn.STREAM_KEEPALIVE_MS,

    pub fn noteWrite(self: *StreamHeartbeat, now_ms: i64) void {
        self.last_write_ms = now_ms;
    }

    /// True once `interval_ms` has elapsed with no bytes handed to the socket.
    pub fn due(self: *const StreamHeartbeat, now_ms: i64) bool {
        return now_ms - self.last_write_ms >= self.interval_ms;
    }
};

/// Connection wrapper bundling a TCP stream with its `Io` and per-connection
/// reader/writer buffers. Replaces `std.net.Stream` in 0.16 — methods that took
/// a bare stream now take `*Conn` so the IO interface and buffers travel together.
pub const Conn = struct {
    stream: std.Io.net.Stream,
    io: std.Io,
    write_buf: [16 * 1024]u8,
    read_buf: [16 * 1024]u8,
    write_state: std.Io.net.Stream.Writer,
    read_state: std.Io.net.Stream.Reader,
    /// Silence tracker for the streaming keepalive. Stamped by every write
    /// path below, so it measures client-visible silence rather than "time
    /// since the last token" — a buffered tool call keeps producing tokens
    /// while emitting nothing. (`writeAllNoFlush` stamps too: every SSE emit
    /// path flushes before returning, so a stamp can't outrun the socket by
    /// more than one event.)
    heartbeat: StreamHeartbeat = .{},
    /// Non-null when this connection is bridged onto a WebSocket. Output that
    /// would otherwise be HTTP/SSE is reshaped into WS text frames at the
    /// `sendResponse` / `sendAnthropicEvent` chokepoints.
    ws_mode: ?*WsBridge = null,
    /// True once this connection has written a `text/event-stream` head: past it a generation
    /// failure can only be an SSE `error` event. Set only in `sendSseHeaders`.
    sse_headers_sent: bool = false,
    /// True once this connection has written a response head with a `Content-Length`: the body
    /// ends by its length, not at the close, so `close` need not wait for the peer.
    length_framed: bool = false,
    /// Non-null while an Ollama /api/* handler runs an inner /v1 handler:
    /// every write the inner handler makes is fed to the sink (SSE → NDJSON
    /// re-framing) instead of the socket. The sink writes its translated
    /// output through `writer()` directly, bypassing this hook — same
    /// interception pattern as `ws_mode`. See src/host/api/ollama/translate.zig.
    ollama_sink: ?*ollama_mod.Sink = null,
    /// Agent named by the current request's User-Agent.
    client: instr.Client = .other,

    pub fn init(c: *Conn, stream: std.Io.net.Stream, io: std.Io) void {
        c.stream = stream;
        c.io = io;
        c.client = .other;
        c.write_state = stream.writer(io, &c.write_buf);
        c.read_state = stream.reader(io, &c.read_buf);
        c.ws_mode = null;
        c.ollama_sink = null;
        c.sse_headers_sent = false;
        c.length_framed = false;
        c.heartbeat = .{ .last_write_ms = nowMsMonotonic(io) };
    }

    /// Replaces the previous request's client, so a request without a User-Agent never inherits it.
    pub fn noteRequestHeaders(c: *Conn, headers: []const u8) void {
        c.client = instr.Client.fromUserAgent(findHeaderValue(headers, "user-agent"));
    }

    /// True when the connection has been silent long enough that a streaming
    /// keepalive is due.
    pub fn keepaliveDue(c: *Conn) bool {
        return c.heartbeat.due(nowMsMonotonic(c.io));
    }

    pub fn writer(c: *Conn) *std.Io.Writer {
        return &c.write_state.interface;
    }

    pub fn reader(c: *Conn) *std.Io.Reader {
        return &c.read_state.interface;
    }

    pub fn writeAll(c: *Conn, data: []const u8) !void {
        c.heartbeat.noteWrite(nowMsMonotonic(c.io));
        if (c.ollama_sink) |s| return s.feed(data);
        try c.writer().writeAll(data);
        try c.writer().flush();
    }

    pub fn writeAllNoFlush(c: *Conn, data: []const u8) !void {
        c.heartbeat.noteWrite(nowMsMonotonic(c.io));
        if (c.ollama_sink) |s| return s.feed(data);
        try c.writer().writeAll(data);
    }

    pub fn flush(c: *Conn) !void {
        c.heartbeat.noteWrite(nowMsMonotonic(c.io));
        if (c.ollama_sink != null) return;
        try c.writer().flush();
    }

    /// Read up to buf.len bytes; may return fewer (HTTP-style short read). Returns 0 on EOF.
    /// Uses `readVec` because `readSliceShort` blocks until the buffer is full or EOF —
    /// wrong semantics for HTTP request parsing where we read until headers terminate.
    pub fn read(c: *Conn, buf: []u8) !usize {
        var bufs: [1][]u8 = .{buf};
        return c.reader().readVec(&bufs) catch |err| switch (err) {
            error.EndOfStream => 0,
            else => |e| e,
        };
    }

    /// Waits up to `ms` for bytes to read; false when none came. Bytes the reader already
    /// buffered count as readable.
    pub fn awaitReadable(c: *Conn, ms: i64) bool {
        if (c.reader().bufferedLen() > 0) return true;
        if (ms <= 0) return false;
        return socketReadable(c.stream.socket.handle, @intCast(@min(ms, std.math.maxInt(i32)))) catch true;
    }

    /// Longest `close` waits for the peer to hang up before releasing the socket anyway.
    const CLOSE_WAIT_MS: i64 = 5 * 60 * 1000;

    /// A close-delimited body (SSE, NDJSON: no `Content-Length`) ends at the close, so this sends
    /// our FIN and holds the socket until the peer closes. A socket closed under a reader that
    /// still lags is dropped by the kernel with an RST once its FIN_WAIT_2 timer runs out (60 s
    /// on macOS), so the reader gets the whole body and then ECONNRESET, not EOF. A length-framed
    /// response closes at once: its reader knows where the body ends.
    pub fn close(c: *Conn) void {
        c.flush() catch {};
        if (c.length_framed) return c.stream.close(c.io);
        if (comptime builtin.os.tag == .windows) return c.closeAfterPeerWindows();
        const fd = c.stream.socket.handle;
        _ = std.posix.system.shutdown(fd, std.posix.SHUT.WR);
        const deadline = nowMsMonotonic(c.io) + CLOSE_WAIT_MS;
        var fds = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        var sink: [256]u8 = undefined;
        while (nowMsMonotonic(c.io) < deadline and !http_server.shutdown_requested.load(.acquire)) {
            const n = std.posix.poll(&fds, 500) catch break;
            if (n == 0) continue;
            // EOF or reset ends the wait; a byte from a pipelining client is dropped.
            if (std.c.recv(fd, &sink, sink.len, std.posix.MSG.DONTWAIT) <= 0) break;
        }
        c.stream.close(c.io);
    }

    /// Non-blocking probe: has the peer closed the connection? TCP send
    /// buffers absorb hundreds of SSE writes after the client FIN/RST so
    /// `writeAll` only fails seconds late; this surfaces the disconnect
    /// promptly so a long-running decode can be cancelled before the GPU
    /// burns more cycles.
    ///
    /// Uses `poll(timeout=0)` for HUP/ERR, and a zero-byte `recv(MSG_PEEK)`
    /// to disambiguate `POLLIN`-with-data (still alive, client sending) from
    /// `POLLIN`-with-FIN (peer closed, `recv` returns 0). Stray bytes from a
    /// pipelined client are not expected on an SSE response stream — if we
    /// see them, we conservatively treat the connection as live.
    /// Idle interval (ms) between keepalive probes while a streaming
    /// request waits on its first tokens (prefill). Short enough that an
    /// abandoned prefill is noticed and cancelled quickly and that clients
    /// never hit stream-idle timeouts; tiny traffic cost.
    pub const STREAM_KEEPALIVE_MS: i64 = 5000;

    /// `close`'s peer-hangup wait over AFD readiness (Windows sockets take no poll(2)).
    fn closeAfterPeerWindows(c: *Conn) void {
        const h = c.stream.socket.handle;
        c.stream.shutdown(c.io, .send) catch {};
        const deadline = nowMsMonotonic(c.io) + CLOSE_WAIT_MS;
        while (nowMsMonotonic(c.io) < deadline and !http_server.shutdown_requested.load(.acquire)) {
            const ev = win_io.poll(h, win_io.RECEIVE | win_io.GONE, 500) catch break;
            if (ev & win_io.GONE != 0) break;
            // A byte from a pipelining client is dropped.
            if (ev != 0) _ = c.reader().discard(.limited(256)) catch break;
        }
        c.stream.close(c.io);
    }

    pub fn peerClosed(c: *Conn) bool {
        if (comptime builtin.os.tag == .windows)
            return (win_io.poll(c.stream.socket.handle, win_io.GONE, 0) catch return false) != 0;
        const fd = c.stream.socket.handle;
        var fds = [_]std.posix.pollfd{.{
            .fd = fd,
            .events = std.posix.POLL.IN,
            .revents = 0,
        }};
        const n = std.posix.poll(&fds, 0) catch return false;
        if (n == 0) return false;
        const revents = fds[0].revents;
        if ((revents & (std.posix.POLL.HUP | std.posix.POLL.ERR | std.posix.POLL.NVAL)) != 0) return true;
        if ((revents & std.posix.POLL.IN) == 0) return false;
        var peek_buf: [1]u8 = undefined;
        const flags: c_int = std.posix.MSG.PEEK | std.posix.MSG.DONTWAIT;
        const r = std.c.recv(fd, &peek_buf, peek_buf.len, flags);
        if (r == 0) return true; // FIN: peer closed cleanly
        return false; // negative (EAGAIN) or data available → assume alive
    }
};

// ── tests ──

test "Conn.peerClosed: alive socket returns false, closed peer returns true" {
    // Create a connected socket pair (AF_UNIX SOCK_STREAM via socketpair).
    var sv: [2]std.posix.fd_t = undefined;
    const AF_UNIX: c_uint = 1;
    const SOCK_STREAM: c_uint = 1;
    const rc = std.c.socketpair(AF_UNIX, SOCK_STREAM, 0, &sv);
    try testing.expect(rc == 0);

    const server_fd = sv[0];
    const client_fd = sv[1];

    // Build a Conn that wraps the server-side fd. We only need
    // `stream.socket.handle` for peerClosed, so the Reader/Writer state
    // can stay zeroed.
    var conn: Conn = undefined;
    conn.stream = .{ .socket = .{ .handle = server_fd, .address = undefined } };

    // Healthy connection: no data pending, no FIN → peerClosed returns false.
    try testing.expect(!conn.peerClosed());

    // Client closes its side → server should observe FIN/HUP.
    _ = std.c.close(client_fd);

    // socketpair() returns connected sockets in the kernel; close-of-peer
    // is observable immediately on the other side without delay.
    const closed = conn.peerClosed();
    _ = std.c.close(server_fd);
    try testing.expect(closed);
}

/// Runs `Conn.close` after one response, with a peer that stays connected, and checks whether it
/// held the socket (`close_delimited`) or returned at once.
fn expectCloseWaitsForPeer(close_delimited: bool) !void {
    var sv: [2]std.posix.fd_t = undefined;
    const AF_UNIX: c_uint = 1;
    const SOCK_STREAM: c_uint = 1;
    try testing.expect(std.c.socketpair(AF_UNIX, SOCK_STREAM, 0, &sv) == 0);

    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, testing.io);
    const tail: []const u8 = if (close_delimited) "data: [DONE]\n\n" else "{}";
    if (close_delimited) {
        try sendSseHeaders(&conn, "test", SSE_ALLOW_HEADERS_DEFAULT);
        try conn.writeAll(tail);
    } else {
        try sendResponseFramed(&conn, "200 OK", "application/json", tail);
    }

    var closed = std.atomic.Value(bool).init(false);
    const closer = try std.Thread.spawn(.{}, struct {
        fn run(c: *Conn, done: *std.atomic.Value(bool)) void {
            c.close();
            done.store(true, .release);
        }
    }.run, .{ &conn, &closed });
    var client_open = true;
    defer closer.join();
    defer if (client_open) {
        _ = std.c.close(sv[1]);
    };

    // The reader gets the whole response and then EOF ...
    var buf: [512]u8 = undefined;
    var got: usize = 0;
    while (true) {
        const n = std.c.recv(sv[1], &buf[got], buf.len - got, 0);
        if (n <= 0) break;
        got += @intCast(n);
    }
    try testing.expect(std.mem.endsWith(u8, buf[0..got], tail));

    // ... and only a close-delimited body keeps the server's end open, so the kernel cannot
    // reset a lagging reader.
    var waited: u32 = 0;
    while (!closed.load(.acquire) and waited < 50) : (waited += 1) {
        std.Io.sleep(testing.io, .fromMilliseconds(10), .real) catch {};
    }
    try testing.expectEqual(!close_delimited, closed.load(.acquire));

    _ = std.c.close(sv[1]);
    client_open = false;
    waited = 0;
    while (!closed.load(.acquire) and waited < 300) : (waited += 1) {
        std.Io.sleep(testing.io, .fromMilliseconds(10), .real) catch {};
    }
    try testing.expect(closed.load(.acquire));
}

test "Conn.close keeps the socket until the peer hangs up, after a close-delimited body" {
    try expectCloseWaitsForPeer(true);
}

test "Conn.close does not wait for the peer after a Content-Length response" {
    try expectCloseWaitsForPeer(false);
}

test "readRequest: a head that expects 100-continue gets it before the body is read" {
    var body_buf: [64]u8 = undefined;
    var reply_buf: [256]u8 = undefined;
    // curl sends this for bodies over 1 MB and waits a second for the 100 before sending them.
    var sv: [2]std.posix.fd_t = undefined;
    try testing.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, testing.io);
    defer conn.close();
    const client = try std.Thread.spawn(.{}, struct {
        fn run(fd: std.posix.fd_t, reply: []u8) void {
            const head = "POST /v1/chat/completions HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 10\r\n\r\n";
            _ = std.c.send(fd, head, head.len, 0);
            const got = std.c.recv(fd, reply.ptr, reply.len, 0); // the body waits for the 100
            if (got > 0) _ = std.c.send(fd, "0123456789", 10, 0);
            _ = std.c.shutdown(fd, 1);
        }
    }.run, .{ sv[1], &reply_buf });
    const raw = try readRequest(testing.allocator, &conn);
    client.join();
    _ = std.c.close(sv[1]);
    const r = raw orelse return error.TestUnexpectedResult;
    defer testing.allocator.free(r.buf);
    try testing.expectEqualStrings("0123456789", r.buf[r.header_end..r.len]);
    try testing.expect(std.mem.startsWith(u8, &reply_buf, "HTTP/1.1 100 Continue\r\n\r\n"));
    const whole = try readRequestOver("POST /v1/chat/completions HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 2\r\n\r\n{}", &body_buf, &reply_buf);
    try testing.expectEqualStrings("", whole.reply); // the body already arrived
}

test "StreamHeartbeat: a write resets the deadline, silence expires it" {
    var hb = StreamHeartbeat{ .last_write_ms = 1_000, .interval_ms = 5_000 };

    // Silent, but not yet for a full interval.
    try testing.expect(!hb.due(1_000));
    try testing.expect(!hb.due(5_999));
    // Exactly the interval is due (the client's clock is not ours to trust).
    try testing.expect(hb.due(6_000));
    try testing.expect(hb.due(60_000));

    // Any byte written to the socket pushes the deadline out.
    hb.noteWrite(60_000);
    try testing.expect(!hb.due(60_001));
    try testing.expect(hb.due(65_000));
}

test "StreamHeartbeat: buffered tokens do NOT reset the deadline" {
    // The regression this guards: the old code beat only when NO token was
    // available (`.idle`). A tool call buffers tokens continuously while
    // writing nothing, so token arrival must not be mistaken for liveness —
    // only `noteWrite` (bytes on the socket) counts.
    var hb = StreamHeartbeat{ .last_write_ms = 0, .interval_ms = 5_000 };
    var now: i64 = 0;
    // 300 tokens arrive over 30s. Not one is written out (buffered tool call).
    var i: usize = 0;
    while (i < 300) : (i += 1) now += 100;
    try testing.expectEqual(@as(i64, 30_000), now);
    try testing.expect(hb.due(now));
}

test "Conn names the client from each request's User-Agent and keeps no header text" {
    const peer: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 10, 0, 0, 9 }, .port = 4242 } };
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = -1, .address = peer } }, std.Io.Threaded.global_single_threaded.io());
    try std.testing.expectEqual(instr.Client.other, conn.client);

    conn.noteRequestHeaders("POST /v1/messages HTTP/1.1\r\nUser-Agent: opencode/1.4\r\n\r\n");
    try std.testing.expectEqual(instr.Client.opencode, conn.client);
    conn.noteRequestHeaders("POST /v1/messages HTTP/1.1\r\nHost: x\r\n\r\n");
    try std.testing.expectEqual(instr.Client.other, conn.client);
}
