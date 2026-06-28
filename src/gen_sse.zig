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
    pub fn emit(self: Progress, stage: []const u8, step: u32, total: u32) void {
        self.cb(self.ctx, stage, step, total);
    }
};

/// SSE response headers (no Content-Length — the body is an event stream).
pub const headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n";

/// A `Progress` sink that writes one `data: {…progress…}` SSE event per call
/// (`total == 0` signals an indeterminate bar, e.g. audio of unknown length).
pub const StreamCtx = struct {
    conn: *Conn,
    pub fn cb(ptr: *anyopaque, stage: []const u8, step: u32, total: u32) void {
        const self: *StreamCtx = @ptrCast(@alignCast(ptr));
        var buf: [256]u8 = undefined;
        const ev = std.fmt.bufPrint(&buf, "data: {{\"type\":\"progress\",\"stage\":\"{s}\",\"step\":{d},\"total\":{d}}}\n\n", .{ stage, step, total }) catch return;
        self.conn.writeAll(ev) catch {};
    }
    pub fn progress(self: *StreamCtx) Progress {
        return .{ .ctx = self, .cb = StreamCtx.cb };
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
