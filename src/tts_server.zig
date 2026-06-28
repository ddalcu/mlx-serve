//! Standalone HTTP server for native Qwen3-TTS text-to-speech.
//!
//! Like the ds4 / llama embedded engines, TTS is a serial, request→file engine
//! that doesn't fit the MLX transformer scheduler (no token streaming, no KV
//! batching). Rather than half-wire it through the scheduler, it gets a small
//! self-contained serve loop: load the `tts.Synthesizer` once on this thread
//! (so the mlx GPU stream is thread-local and stable), then accept connections
//! serially and answer `POST /v1/audio/speech` with WAV bytes.
//!
//! `main.zig` routes here when `config.json`'s `model_type == "qwen3_tts"`.

const std = @import("std");
const mlx = @import("mlx.zig");
const tts = @import("tts.zig");
const log = @import("log.zig");
const server_mod = @import("server.zig");

const Conn = server_mod.Conn;

pub fn runTtsServe(
    io: std.Io,
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    host: []const u8,
    port: u16,
) !void {
    log.info("mlx-serve (native Qwen3-TTS engine)\n", .{});
    log.info("[args] model: {s}\n", .{model_dir});

    // Load the synthesizer on THIS thread; the mlx GPU stream it captures is
    // used for every subsequent synthesis on this (single) serve thread.
    const s = mlx.mlx_default_gpu_stream_new();
    var synth = try tts.Synthesizer.load(io, allocator, s, model_dir);
    defer synth.deinit();
    log.info("[tts] synthesizer ready (sample_rate={d})\n", .{synth.model.cfg.sample_rate});

    const model_id = basename(model_dir);

    // Listener (mirrors server.zig).
    var ip4_bytes: [4]u8 = .{ 0, 0, 0, 0 };
    {
        var parts = std.mem.splitScalar(u8, host, '.');
        var idx: usize = 0;
        while (parts.next()) |part| {
            if (idx >= 4) break;
            ip4_bytes[idx] = std.fmt.parseInt(u8, part, 10) catch 0;
            idx += 1;
        }
    }
    const ip_addr: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = ip4_bytes, .port = port } };
    var server = try ip_addr.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);
    log.info("\nTTS server listening on http://{s}:{d}\n", .{ host, port });
    log.info("  POST /v1/audio/speech  {{\"model\":\"...\",\"input\":\"text to speak\"}}\n", .{});

    while (true) {
        const accepted = server.accept(io) catch |err| {
            log.err("accept error: {}\n", .{err});
            continue;
        };
        var conn: Conn = undefined;
        conn.init(accepted, io);
        handleOne(allocator, &synth, &conn, model_id) catch |err| {
            log.warn("[tts] request error: {}\n", .{err});
        };
        conn.flush() catch {};
        conn.close();
    }
}

fn handleOne(allocator: std.mem.Allocator, synth: *tts.Synthesizer, conn: *Conn, model_id: []const u8) !void {
    // Read headers, then body (mirrors server.handleConnection).
    var hdr_buf: [16 * 1024]u8 = undefined;
    var total_read: usize = 0;
    var content_length: ?usize = null;
    var header_end_pos: usize = 0;
    while (total_read < hdr_buf.len) {
        const n = try conn.read(hdr_buf[total_read..]);
        if (n == 0) break;
        total_read += n;
        if (std.mem.indexOf(u8, hdr_buf[0..total_read], "\r\n\r\n")) |he| {
            header_end_pos = he + 4;
            content_length = findContentLength(hdr_buf[0..he]);
            break;
        }
    }
    if (header_end_pos == 0) return;

    const cl = content_length orelse 0;
    const total_size = header_end_pos + cl;
    if (total_size > 16 * 1024 * 1024) return sendError(conn, 413, "request too large");
    const buf = try allocator.alloc(u8, total_size);
    defer allocator.free(buf);
    @memcpy(buf[0..total_read], hdr_buf[0..total_read]);
    while (total_read < total_size) {
        const n = try conn.read(buf[total_read..total_size]);
        if (n == 0) break;
        total_read += n;
    }
    const request = buf[0..total_read];
    const first_line_end = std.mem.indexOf(u8, request, "\r\n") orelse return;
    const first_line = request[0..first_line_end];
    const body = if (header_end_pos <= total_read) request[header_end_pos..total_read] else "";

    // Method + path.
    var it = std.mem.tokenizeScalar(u8, first_line, ' ');
    const method = it.next() orelse return;
    const path = it.next() orelse return;

    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/health")) {
        return sendJson(conn, "{\"status\":\"ok\"}");
    }
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/v1/models")) {
        var b: std.ArrayList(u8) = .empty;
        defer b.deinit(allocator);
        try b.appendSlice(allocator, "{\"object\":\"list\",\"data\":[{\"id\":\"");
        try b.appendSlice(allocator, model_id);
        try b.appendSlice(allocator, "\",\"object\":\"model\",\"capabilities\":[\"audio\"]}]}");
        return sendJson(conn, b.items);
    }
    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/audio/speech")) {
        const input = extractJsonString(body, "input") orelse extractJsonString(body, "text") orelse {
            return sendError(conn, 400, "missing 'input'");
        };
        // Unescape minimal JSON (\n \t \" \\) so the spoken text is literal.
        const text = try jsonUnescape(allocator, input);
        defer allocator.free(text);
        if (text.len == 0) return sendError(conn, 400, "empty 'input'");

        log.info("[tts] synthesizing {d} chars\n", .{text.len});
        const wav = synth.synthesizeWav(text, 2048) catch |err| {
            log.err("[tts] synthesis failed: {}\n", .{err});
            return sendError(conn, 500, "synthesis failed");
        };
        defer allocator.free(wav);
        log.info("[tts] -> {d} WAV bytes\n", .{wav.len});
        return sendBytes(conn, allocator, "audio/wav", wav);
    }
    return sendError(conn, 404, "not found");
}

// ── HTTP response helpers ──

fn sendBytes(conn: *Conn, allocator: std.mem.Allocator, content_type: []const u8, payload: []const u8) !void {
    var hdr: std.ArrayList(u8) = .empty;
    defer hdr.deinit(allocator);
    try hdr.appendSlice(allocator, "HTTP/1.1 200 OK\r\nContent-Type: ");
    try hdr.appendSlice(allocator, content_type);
    try hdr.appendSlice(allocator, "\r\nContent-Length: ");
    var num: [20]u8 = undefined;
    const ns = std.fmt.bufPrint(&num, "{d}", .{payload.len}) catch unreachable;
    try hdr.appendSlice(allocator, ns);
    try hdr.appendSlice(allocator, "\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n");
    try conn.writeAllNoFlush(hdr.items);
    try conn.writeAll(payload);
}

fn sendJson(conn: *Conn, json: []const u8) !void {
    var hdr: [256]u8 = undefined;
    const head = std.fmt.bufPrint(&hdr, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n", .{json.len}) catch return;
    try conn.writeAllNoFlush(head);
    try conn.writeAll(json);
}

fn sendError(conn: *Conn, code: u16, msg: []const u8) !void {
    var body_buf: [256]u8 = undefined;
    const body = std.fmt.bufPrint(&body_buf, "{{\"error\":{{\"message\":\"{s}\"}}}}", .{msg}) catch return;
    var hdr: [256]u8 = undefined;
    const head = std.fmt.bufPrint(&hdr, "HTTP/1.1 {d} Error\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{ code, body.len }) catch return;
    try conn.writeAllNoFlush(head);
    try conn.writeAll(body);
}

// ── Minimal parsing helpers ──

fn findContentLength(headers: []const u8) ?usize {
    // Case-insensitive search for "content-length:".
    var lower_buf: [16 * 1024]u8 = undefined;
    const n = @min(headers.len, lower_buf.len);
    for (headers[0..n], 0..) |c, i| lower_buf[i] = std.ascii.toLower(c);
    const key = "content-length:";
    const idx = std.mem.indexOf(u8, lower_buf[0..n], key) orelse return null;
    var rest = headers[idx + key.len ..];
    var sp: usize = 0;
    while (sp < rest.len and (rest[sp] == ' ' or rest[sp] == '\t')) sp += 1;
    rest = rest[sp..];
    const end = std.mem.indexOfAny(u8, rest, "\r\n ") orelse rest.len;
    return std.fmt.parseInt(usize, rest[0..end], 10) catch null;
}

/// Extract the (raw, still-escaped) string value of top-level JSON key `key`.
fn extractJsonString(body: []const u8, key: []const u8) ?[]const u8 {
    var key_pat_buf: [64]u8 = undefined;
    const key_pat = std.fmt.bufPrint(&key_pat_buf, "\"{s}\"", .{key}) catch return null;
    const ki = std.mem.indexOf(u8, body, key_pat) orelse return null;
    var i = ki + key_pat.len;
    // skip whitespace + ':'
    while (i < body.len and (body[i] == ' ' or body[i] == ':' or body[i] == '\t')) i += 1;
    if (i >= body.len or body[i] != '"') return null;
    i += 1;
    const start = i;
    while (i < body.len) : (i += 1) {
        if (body[i] == '\\') {
            i += 1;
            continue;
        }
        if (body[i] == '"') return body[start..i];
    }
    return null;
}

/// Unescape a JSON string body (handles \n \t \r \" \\ \/ and \uXXXX→byte for ASCII).
fn jsonUnescape(allocator: std.mem.Allocator, raw: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < raw.len) : (i += 1) {
        if (raw[i] != '\\') {
            try out.append(allocator, raw[i]);
            continue;
        }
        i += 1;
        if (i >= raw.len) break;
        switch (raw[i]) {
            'n' => try out.append(allocator, '\n'),
            't' => try out.append(allocator, '\t'),
            'r' => try out.append(allocator, '\r'),
            '"' => try out.append(allocator, '"'),
            '\\' => try out.append(allocator, '\\'),
            '/' => try out.append(allocator, '/'),
            'u' => {
                if (i + 4 < raw.len) {
                    const cp = std.fmt.parseInt(u21, raw[i + 1 .. i + 5], 16) catch 0;
                    var b: [4]u8 = undefined;
                    const len = std.unicode.utf8Encode(cp, &b) catch 0;
                    try out.appendSlice(allocator, b[0..len]);
                    i += 4;
                }
            },
            else => try out.append(allocator, raw[i]),
        }
    }
    return out.toOwnedSlice(allocator);
}

fn basename(path: []const u8) []const u8 {
    var p = path;
    while (p.len > 0 and p[p.len - 1] == '/') p = p[0 .. p.len - 1];
    if (std.mem.lastIndexOfScalar(u8, p, '/')) |i| return p[i + 1 ..];
    return p;
}

test "extractJsonString + jsonUnescape" {
    const body = "{\"model\":\"x\",\"input\":\"Hello\\nworld\"}";
    const raw = extractJsonString(body, "input").?;
    try std.testing.expectEqualStrings("Hello\\nworld", raw);
    const un = try jsonUnescape(std.testing.allocator, raw);
    defer std.testing.allocator.free(un);
    try std.testing.expectEqualStrings("Hello\nworld", un);
}

test "findContentLength case-insensitive" {
    try std.testing.expectEqual(@as(?usize, 42), findContentLength("Host: x\r\nContent-Length: 42\r\n"));
    try std.testing.expectEqual(@as(?usize, 7), findContentLength("content-length:7"));
}
