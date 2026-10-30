//! responses, HTTP logging, SSE headers

const std = @import("std");
const log = @import("../../sdk/util/log.zig");
const lan_mod = @import("../lan/lan.zig");
const testing = std.testing;

const Conn = @import("conn.zig").Conn;
const handleLanProxy = @import("../lan/http.zig").handleLanProxy;
const queryParamValue = @import("auth.zig").queryParamValue;
const loginFormKey = @import("auth.zig").loginFormKey;
const readRequestOver = @import("request.zig").readRequestOver;

pub fn sendResponse(stream: *Conn, status: []const u8, content_type: []const u8, body: []const u8) !void {
    logHttpResponse(status, content_type, body);
    try sendResponseFramed(stream, status, content_type, body);
}

/// Like `sendResponse` but does NOT dump the response body to the debug log —
/// only a one-line status summary. For high-frequency, self-describing bodies
/// where the full dump is pure noise: the Prometheus scrape (`/metrics`, hit
/// every couple seconds) and the panel's JSON feed (`/metrics.json`, ~1 Hz).
pub fn sendResponseQuiet(stream: *Conn, status: []const u8, content_type: []const u8, body: []const u8) !void {
    if (log.isDebug()) log.debug("[http] <- {s} {s} body={d}b (scrape; body not logged)\n", .{ status, content_type, body.len });
    try sendResponseFramed(stream, status, content_type, body);
}

pub fn sendResponseFramed(stream: *Conn, status: []const u8, content_type: []const u8, body: []const u8) !void {
    if (stream.ws_mode) |bridge| {
        // WS transport: skip HTTP framing, send body as a single text frame.
        // Compliance suite expects errors as `{"type":"error", ...}` text frames.
        if (body.len > 0) try bridge.sendText(body);
        return;
    }

    var hdr_buf: [512]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, GET, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\n\r\n", .{
        status,
        content_type,
        body.len,
    }) catch return error.Overflow;
    try stream.writeAll(hdr);
    stream.length_framed = true;
    if (body.len > 0) try stream.writeAll(body);
}

pub fn logHttpRequest(method: []const u8, path: []const u8, body: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] -> {s} {s} body={d}b\n", .{ method, path, body.len });
    logHttpBody("[http] request body", body);
}

pub fn logHttpResponse(status: []const u8, content_type: []const u8, body: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] <- {s} {s} body={d}b\n", .{ status, content_type, body.len });
    if (responseBodyLogged(content_type)) logHttpBody("[http] response body", body);
}

/// The path as logged: a query carrying the API key is dropped.
pub fn logSafePath(raw_path: []const u8) []const u8 {
    if (queryParamValue(raw_path, "api_key") == null and queryParamValue(raw_path, "key") == null) return raw_path;
    return raw_path[0 .. std.mem.indexOfScalar(u8, raw_path, '?') orelse raw_path.len];
}

/// The body as logged: the login form's POST carries the API key.
pub fn logSafeBody(method: []const u8, path: []const u8, content_type: []const u8, body: []const u8) []const u8 {
    return if (loginFormKey(method, path, content_type, body) != null) "" else body;
}

/// The console page is half a megabyte of our own HTML and says nothing about a request;
/// API responses (JSON, text) are what a post-mortem reads.
fn responseBodyLogged(content_type: []const u8) bool {
    return !std.mem.startsWith(u8, content_type, "text/html");
}

/// The `Access-Control-Allow-Headers` value each SSE surface advertises.
pub const SSE_ALLOW_HEADERS_DEFAULT = "Content-Type, Authorization";

pub const SSE_ALLOW_HEADERS_ANTHROPIC = "Content-Type, Authorization, x-api-key, anthropic-version";

/// The one place a `text/event-stream` response head is written. Past this call a generation
/// failure is an SSE `error` event; before it the stream answers like its non-streaming twin.
pub fn sendSseHeaders(stream: *Conn, kind: []const u8, allow_headers: []const u8) !void {
    try stream.writeAllNoFlush("HTTP/1.1 200 OK\r\n" ++
        "Content-Type: text/event-stream\r\n" ++
        "Cache-Control: no-cache\r\n" ++
        "Connection: close\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n" ++
        "Access-Control-Allow-Headers: ");
    try stream.writeAllNoFlush(allow_headers);
    try stream.writeAll("\r\n\r\n");
    stream.sse_headers_sent = true;
    logHttpStreamStart(kind);
}

fn logHttpStreamStart(kind: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] <- 200 OK text/event-stream ({s})\n", .{kind});
}

pub fn logHttpSseEvent(event_name: []const u8, data: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] <- sse event={s} data={d}b\n", .{ event_name, data.len });
    logHttpBody("[http] sse data", data);
}

pub fn logHttpSseData(data: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] <- sse data={d}b\n", .{data.len});
    logHttpBody("[http] sse data", data);
}

/// Cap on how much of a BINARY body reaches the debug log.
const BODY_LOG_LIMIT = 4096;

/// True when every byte is printable or ordinary whitespace. Multibyte UTF-8
/// (≥0x80) counts as text, so an emoji in a chat body doesn't demote it.
fn bodyIsText(body: []const u8) bool {
    for (body) |c| {
        if (c == '\n' or c == '\r' or c == '\t') continue;
        if (c < 0x20 or c == 0x7f) return false;
    }
    return true;
}

/// Bounded, strictly-printable view of a body, copied into `out`.
fn bodyPreview(out: []u8, body: []const u8) []const u8 {
    const n = @min(@min(body.len, out.len), BODY_LOG_LIMIT);
    for (body[0..n], 0..) |c, i| {
        out[i] = if (c == '\n' or c == '\r' or c == '\t' or (c >= 0x20 and c < 0x7f)) c else '.';
    }
    return out[0..n];
}

/// A run of base64 this long is an embedded image or clip, not text anyone reads out of a log.
const LOG_BASE64_RUN: usize = 4096;

fn isBase64Byte(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '+' or c == '/' or c == '=' or c == '-' or c == '_';
}

/// `body` with every base64 run of `LOG_BASE64_RUN` bytes or more replaced by `<N base64 chars>`.
fn elideLongBase64(allocator: std.mem.Allocator, body: []const u8) ![]u8 {
    var out = try std.ArrayList(u8).initCapacity(allocator, @min(body.len, 1 << 20));
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < body.len) {
        var j = i;
        while (j < body.len and isBase64Byte(body[j])) j += 1;
        if (j - i >= LOG_BASE64_RUN) {
            try out.print(allocator, "<{d} base64 chars>", .{j - i});
        } else {
            try out.appendSlice(allocator, body[i..j]);
        }
        if (j < body.len) {
            try out.append(allocator, body[j]);
            j += 1;
        }
        i = j;
    }
    return out.toOwnedSlice(allocator);
}

fn logHttpBody(label: []const u8, body: []const u8) void {
    if (body.len == 0) return;
    // Text bodies go out WHOLE — reading a full request body out of the debug
    // log is the documented way to reproduce a tool-calling bug — minus embedded
    // images: a vision agent's history is megabytes of base64 per request.
    if (bodyIsText(body)) {
        if (body.len > LOG_BASE64_RUN) if (elideLongBase64(std.heap.page_allocator, body)) |shown| {
            defer std.heap.page_allocator.free(shown);
            log.debug("{s} ({d}b):\n{s}\n", .{ label, body.len, shown });
            return;
        } else |_| {};
        log.debug("{s} ({d}b):\n{s}\n", .{ label, body.len, body });
        return;
    }
    // Binary is a different story. `/v1/images/edits` is multipart, so one image
    // upload wrote raw PNG bytes into the log, and a body near the request
    // cap blew through the 32 MB rotation — taking the post-mortem file with it.
    var buf: [BODY_LOG_LIMIT]u8 = undefined;
    const preview = bodyPreview(&buf, body);
    log.debug("{s} ({d}b binary, first {d} sanitized):\n{s}…\n", .{ label, body.len, preview.len, preview });
}

// ── tests ──

test "a logged request never carries the API key" {
    try testing.expectEqualStrings("/metrics", logSafePath("/metrics?api_key=s3cret"));
    try testing.expectEqualStrings("/v1/models", logSafePath("/v1/models?x=1&key=s3cret"));
    try testing.expectEqualStrings("/v1/messages?beta=true", logSafePath("/v1/messages?beta=true"));
    const ct = "application/x-www-form-urlencoded";
    try testing.expectEqualStrings("", logSafeBody("POST", "/", ct, "api_key=s3cret"));
    try testing.expectEqualStrings("{\"a\":1}", logSafeBody("POST", "/v1/chat/completions", "application/json", "{\"a\":1}"));
}

test "the default-level log lines never carry the API key either" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [512]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(std.Io.Threaded.global_single_threaded.io(), &root_buf)];
    var path_buf: [600]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/s.log", .{root});
    try log.openFile(path, 0);
    defer log.closeFile();

    // The 413 for an oversized body (warn).
    var body_buf: [64]u8 = undefined;
    var reply_buf: [1024]u8 = undefined;
    _ = try readRequestOver("POST /v1/chat/completions?api_key=s3cret HTTP/1.1\r\nContent-Length: 18446744073709551615\r\n\r\n{}", &body_buf, &reply_buf);

    // The LAN proxy line (info), to a peer that accepts nothing.
    const a = testing.allocator;
    var l = lan_mod.Lan{ .alloc = a, .port = 0, .discover = true, .peers = .init(a), .known = .init(a) };
    defer {
        var it = l.peers.valueIterator();
        while (it.next()) |p| p.deinit(a);
        l.peers.deinit();
        l.known.deinit();
    }
    const models = try a.alloc(lan_mod.PeerModel, 1);
    models[0] = .{ .id = try a.dupe(u8, "gemma"), .entry_json = try a.dupe(u8, "{}") };
    l.installPeer("studio", .{ 127, 0, 0, 1 }, 1, models, false, .{});
    var sv: [2]std.posix.fd_t = undefined;
    try testing.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, testing.io);
    defer conn.close();
    defer _ = std.c.close(sv[1]);
    try handleLanProxy(a, &conn, &l, "POST", "/v1/chat/completions?key=s3cret", "{\"model\":\"gemma@studio\"}", null, "gemma@studio");

    const logged = try tmp.dir.readFileAlloc(std.Io.Threaded.global_single_threaded.io(), "s.log", a, .limited(1 << 20));
    defer a.free(logged);
    try testing.expect(std.mem.indexOf(u8, logged, "[lan] proxy") != null); // both lines were written
    try testing.expect(std.mem.indexOf(u8, logged, "413") != null);
    try testing.expect(std.mem.indexOf(u8, logged, "s3cret") == null);
}

test "responseBodyLogged: API bodies are logged, the console page is not" {
    try testing.expect(responseBodyLogged("application/json"));
    try testing.expect(responseBodyLogged("text/plain; charset=utf-8"));
    try testing.expect(!responseBodyLogged("text/html; charset=utf-8"));
}

test "debug body log: an embedded image is elided and the text around it is kept whole" {
    const gpa = std.testing.allocator;
    var b64: [5000]u8 = undefined;
    @memset(&b64, 'A');
    const body = try std.fmt.allocPrint(gpa, "{{\"role\":\"user\",\"image\":\"data:image/png;base64,{s}\",\"text\":\"keep this\"}}", .{&b64});
    defer gpa.free(body);
    const out = try elideLongBase64(gpa, body);
    defer gpa.free(out);
    try std.testing.expect(out.len < 200);
    try std.testing.expect(std.mem.indexOf(u8, out, "<5000 base64 chars>") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"role\":\"user\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "keep this") != null);
    // A short run (a token, a hash, a small id) is ordinary text.
    const short = "{\"id\":\"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\"}";
    const kept = try elideLongBase64(gpa, short);
    defer gpa.free(kept);
    try std.testing.expectEqualStrings(short, kept);
}

test "debug body log bounds BINARY but never truncates text (multipart PNG class)" {
    // `/v1/images/edits` is the first binary body we serve. `logHttpBody` dumped
    // whatever it got verbatim, so at --log-level debug one image upload wrote
    // raw PNG bytes into the log — and a body near the request cap blew
    // straight through the 32 MB rotation, taking the post-mortem file with it.
    //
    // Text bodies must stay WHOLE: reading a full request body out of the debug
    // log is the documented way to reproduce a tool-calling bug.
    try testing.expect(bodyIsText("{\"model\":\"x\",\"prompt\":\"hi\"}"));
    try testing.expect(bodyIsText("multi\nline\twith\r\nwhitespace"));
    try testing.expect(bodyIsText("utf8 stays text: é ✓ 🎉")); // multibyte, not control
    try testing.expect(!bodyIsText("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"));
    // The real shape: text multipart framing wrapped around a binary payload.
    try testing.expect(!bodyIsText("--B\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\n\x89PNG\x00\x01\r\n--B--"));

    // A binary preview is bounded AND strictly printable ASCII.
    var buf: [64]u8 = undefined;
    const png = "\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\xffbinary tail";
    const p = bodyPreview(&buf, png);
    try testing.expect(p.len <= buf.len);
    for (p) |c| try testing.expect(c == '\n' or c == '\r' or c == '\t' or (c >= 0x20 and c < 0x7f));

    // Larger than the sink → truncated, never streamed.
    const big = try testing.allocator.alloc(u8, 1024 * 1024);
    defer testing.allocator.free(big);
    @memset(big, 0);
    try testing.expectEqual(@as(usize, buf.len), bodyPreview(&buf, big).len);
    // The cap is the SMALLER of the caller's buffer and the byte limit, so a
    // generous buffer can't reintroduce the megabyte dump.
    var huge: [BODY_LOG_LIMIT * 2]u8 = undefined;
    try testing.expectEqual(@as(usize, BODY_LOG_LIMIT), bodyPreview(&huge, big).len);
}
