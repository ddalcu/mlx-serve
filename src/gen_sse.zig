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
    cancelled: bool = false,
    pub fn cb(ptr: *anyopaque, stage: []const u8, step: u32, total: u32) void {
        const self: *StreamCtx = @ptrCast(@alignCast(ptr));
        var buf: [256]u8 = undefined;
        const ev = std.fmt.bufPrint(&buf, "data: {{\"type\":\"progress\",\"stage\":\"{s}\",\"step\":{d},\"total\":{d}}}\n\n", .{ stage, step, total }) catch return;
        self.conn.writeAll(ev) catch {
            self.cancelled = true;
        };
    }
    fn cancelledCb(ptr: *anyopaque) bool {
        const self: *StreamCtx = @ptrCast(@alignCast(ptr));
        return self.cancelled;
    }
    pub fn progress(self: *StreamCtx) Progress {
        return .{ .ctx = self, .cb = StreamCtx.cb, .cancelled_cb = StreamCtx.cancelledCb };
    }
};

/// Write a terminal `error` SSE event.
pub fn sendError(conn: *Conn, msg: []const u8) void {
    var buf: [256]u8 = undefined;
    const ev = std.fmt.bufPrint(&buf, "data: {{\"type\":\"error\",\"message\":\"{s}\"}}\n\n", .{msg}) catch return;
    conn.writeAll(ev) catch {};
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
