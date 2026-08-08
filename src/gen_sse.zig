//! Shared Server-Sent-Events progress plumbing for the native media-gen servers
//! (flux image / tts audio / ltx video). Opt-in per request via `"stream": true`:
//! the server replies `text/event-stream` and pushes `progress` events during
//! generation, then a `complete` (or `error`) event. Non-stream requests keep
//! their single-response shape.

const std = @import("std");
const server_mod = @import("server.zig");
const Conn = server_mod.Conn;

/// Erased progress callback handed into the model code (flux/tts/ltx) so the
/// inner loops can report step/total without importing the HTTP layer.
pub const Progress = struct {
    ctx: *anyopaque,
    cb: *const fn (ctx: *anyopaque, stage: []const u8, step: u32, total: u32) void,
    /// Optional "has the client gone away?" probe. Long inner loops poll it
    /// each step and abort with error.Cancelled — without it a cancelled
    /// request burns the GPU to completion and queues everything behind it.
    cancelled_cb: ?*const fn (ctx: *anyopaque) bool = null,
    pub fn emit(self: Progress, stage: []const u8, step: u32, total: u32) void {
        self.cb(self.ctx, stage, step, total);
    }
    pub fn cancelled(self: Progress) bool {
        const f = self.cancelled_cb orelse return false;
        return f(self.ctx);
    }
};

/// SSE response headers (no Content-Length — the body is an event stream).
pub const headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n";

/// A `Progress` sink that writes one `data: {…progress…}` SSE event per call
/// (`total == 0` signals an indeterminate bar, e.g. audio of unknown length).
/// A failed write (client hung up) latches `cancelled` so the generating loop
/// can abort instead of finishing a video nobody will receive.
pub const StreamCtx = struct {
    conn: *Conn,
    /// False for a non-streaming request. Such a job still needs the
    /// CANCELLATION half — a media generation runs for minutes on the
    /// inference thread with every other request queued behind it, so a
    /// client that hangs up must not leave the GPU finishing a video nobody
    /// will receive — but its response is one JSON body, and an SSE event
    /// spliced into that is unparseable. So `cb` no-ops and only the probe
    /// runs.
    stream: bool = true,
    cancelled: bool = false,
    pub fn cb(ptr: *anyopaque, stage: []const u8, step: u32, total: u32) void {
        const self: *StreamCtx = @ptrCast(@alignCast(ptr));
        if (!self.stream) return;
        var buf: [256]u8 = undefined;
        const ev = std.fmt.bufPrint(&buf, "data: {{\"type\":\"progress\",\"stage\":\"{s}\",\"step\":{d},\"total\":{d}}}\n\n", .{ stage, step, total }) catch return;
        self.conn.writeAll(ev) catch {
            self.cancelled = true;
        };
    }
    fn cancelledCb(ptr: *anyopaque) bool {
        const self: *StreamCtx = @ptrCast(@alignCast(ptr));
        if (self.cancelled) return true;
        // A failed write is a LATE signal even when streaming: TCP send
        // buffers absorb hundreds of events after the client's FIN. The
        // socket probe is prompt, and for a non-streaming job it is the only
        // signal there is.
        if (self.conn.peerClosed()) {
            self.cancelled = true;
            return true;
        }
        return false;
    }
    pub fn progress(self: *StreamCtx) Progress {
        return .{ .ctx = self, .cb = StreamCtx.cb, .cancelled_cb = StreamCtx.cancelledCb };
    }
};

/// Escape a message for embedding in a JSON string, TRUNCATING to fit `out`.
/// Media-gen error bodies are built into fixed buffers from hand-written text,
/// and both failure modes shipped: a message quoting a field value (`mode:"edit"`)
/// emitted a body with raw quotes — invalid JSON a client can't parse — and an
/// over-long one made `bufPrint` fail so the handler sent NO body at all.
/// Truncation stops on a UTF-8 boundary so the tail can't be a broken sequence.
pub fn jsonEscapeMessage(out: []u8, msg: []const u8) []const u8 {
    var n: usize = 0;
    var truncated = false;
    for (msg) |c| {
        var one: [1]u8 = .{if (c < 0x20) ' ' else c};
        const esc: []const u8 = switch (c) {
            '"' => "\\\"",
            '\\' => "\\\\",
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            else => one[0..1],
        };
        if (n + esc.len > out.len) {
            truncated = true;
            break;
        }
        @memcpy(out[n..][0..esc.len], esc);
        n += esc.len;
    }
    if (truncated) {
        while (n > 0 and (out[n - 1] & 0xC0) == 0x80) n -= 1; // continuation bytes
        if (n > 0 and out[n - 1] >= 0x80) n -= 1; // the lead byte they belonged to
    }
    return out[0..n];
}

/// Write a terminal `error` SSE event.
pub fn sendError(conn: *Conn, msg: []const u8) void {
    var esc_buf: [512]u8 = undefined;
    const esc = jsonEscapeMessage(&esc_buf, msg);
    var buf: [640]u8 = undefined;
    const ev = std.fmt.bufPrint(&buf, "data: {{\"type\":\"error\",\"message\":\"{s}\"}}\n\n", .{esc}) catch return;
    conn.writeAll(ev) catch {};
}

/// Tri-state bool: null when the key is absent or unreadable, else its value.
pub fn bodyBool(body: []const u8, key: []const u8) ?bool {
    var pat_buf: [64]u8 = undefined;
    const pat = std.fmt.bufPrint(&pat_buf, "\"{s}\"", .{key}) catch return null;
    const ki = std.mem.indexOf(u8, body, pat) orelse return null;
    var i = ki + pat.len;
    while (i < body.len and (body[i] == ' ' or body[i] == ':' or body[i] == '\t')) i += 1;
    if (std.mem.startsWith(u8, body[i..], "true")) return true;
    if (std.mem.startsWith(u8, body[i..], "false")) return false;
    return null;
}

test "bodyBool is tri-state (absent != false)" {
    try std.testing.expectEqual(@as(?bool, null), bodyBool("{}", "fast"));
    try std.testing.expectEqual(@as(?bool, true), bodyBool("{\"fast\": true}", "fast"));
    try std.testing.expectEqual(@as(?bool, false), bodyBool("{\"fast\":false}", "fast"));
    try std.testing.expectEqual(@as(?bool, null), bodyBool("{\"fast\": \"yes\"}", "fast"));
}

/// True if the JSON body contains `"key": true`.
pub fn bodyWantsTrue(body: []const u8, key: []const u8) bool {
    var pat_buf: [64]u8 = undefined;
    const pat = std.fmt.bufPrint(&pat_buf, "\"{s}\"", .{key}) catch return false;
    const ki = std.mem.indexOf(u8, body, pat) orelse return false;
    var i = ki + pat.len;
    while (i < body.len and (body[i] == ' ' or body[i] == ':' or body[i] == '\t')) i += 1;
    return std.mem.startsWith(u8, body[i..], "true");
}

test "jsonEscapeMessage keeps an error body parseable (live: the mode:\"edit\" 400)" {
    const a = std.testing.allocator;
    var buf: [512]u8 = undefined;

    // The exact message that shipped an INVALID body: raw quotes inside a JSON
    // string. Escaped, the body must round-trip through a real JSON parser.
    const live = "instruction editing (mode:\"edit\") requires a FLUX.2 or Mage-Flow-Edit model";
    const body = try std.fmt.allocPrint(a, "{{\"error\":{{\"message\":\"{s}\"}}}}", .{jsonEscapeMessage(&buf, live)});
    defer a.free(body);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, body, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings(live, parsed.value.object.get("error").?.object.get("message").?.string);

    // Backslashes, control bytes and newlines can't break out either.
    const nasty = "a\\b\"c\nd\te";
    const b2 = try std.fmt.allocPrint(a, "{{\"m\":\"{s}\"}}", .{jsonEscapeMessage(&buf, nasty)});
    defer a.free(b2);
    var p2 = try std.json.parseFromSlice(std.json.Value, a, b2, .{});
    defer p2.deinit();
    try std.testing.expectEqualStrings(nasty, p2.value.object.get("m").?.string);

    // Over-long input truncates instead of vanishing, and never leaves a torn
    // UTF-8 tail (bufPrint's `catch return` used to drop the whole response).
    var small: [8]u8 = undefined;
    const cut = jsonEscapeMessage(&small, "ééééééé");
    try std.testing.expect(cut.len <= small.len);
    try std.testing.expect(std.unicode.utf8ValidateSlice(cut));
    // A trailing escape is emitted whole or not at all.
    var tiny: [1]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 0), jsonEscapeMessage(&tiny, "\"").len);
}

test "Progress.cancelled defaults false, reads the probe when set" {
    const H = struct {
        flag: bool,
        fn emitCb(ctx: *anyopaque, stage: []const u8, step: u32, total: u32) void {
            _ = ctx;
            _ = stage;
            _ = step;
            _ = total;
        }
        fn cancelCb(ctx: *anyopaque) bool {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            return self.flag;
        }
    };
    var h = H{ .flag = false };
    var p = Progress{ .ctx = &h, .cb = H.emitCb };
    try std.testing.expect(!p.cancelled()); // no probe → never cancelled
    p.cancelled_cb = H.cancelCb;
    try std.testing.expect(!p.cancelled());
    h.flag = true;
    try std.testing.expect(p.cancelled());
}

test "a NON-streaming job still gets a cancellation probe, and writes no SSE" {
    // A media generation runs for MINUTES on the inference thread while the
    // connection thread blocks in `Scheduler.runGeneration`, so a client that
    // hangs up leaves the GPU producing a video nobody will receive — and
    // every queued request waits behind it. Streaming requests latch on a
    // failed progress write; a non-streaming one has no write to fail, so it
    // used to have no cancellation at all.
    var sv: [2]std.posix.fd_t = undefined;
    try std.testing.expect(std.c.socketpair(1, 1, 0, &sv) == 0); // AF_UNIX, SOCK_STREAM
    var conn: Conn = undefined;
    conn.stream = .{ .socket = .{ .handle = sv[0], .address = undefined } };
    conn.ollama_sink = null;

    var sctx = StreamCtx{ .conn = &conn, .stream = false };
    const p = sctx.progress();
    try std.testing.expect(!p.cancelled());

    // Emitting must put NOTHING on the wire: this response is a single JSON
    // body, and an SSE event spliced into it is unparseable.
    p.emit("Generating", 1, 15);
    var peek: [1]u8 = undefined;
    const n = std.c.recv(sv[1], &peek, peek.len, std.posix.MSG.PEEK | std.posix.MSG.DONTWAIT);
    try std.testing.expect(n <= 0);

    _ = std.c.close(sv[1]); // client hangs up mid-generation
    const cancelled = p.cancelled();
    _ = std.c.close(sv[0]);
    try std.testing.expect(cancelled);
}
