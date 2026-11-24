//! loopback, api-key gate; constTimeEql, basic auth, cookie, login page + tests

const std = @import("std");
const http_server = @import("server.zig");
const log = @import("../../sdk/util/log.zig");
const ablation = @import("../api/ablation.zig");

const Conn = @import("conn.zig").Conn;
const isRoute = @import("router.zig").isRoute;
const queryDecoded = @import("../api/request_model.zig").queryDecoded;
const pairValue = @import("../api/request_model.zig").pairValue;
const percentDecoded = @import("../api/request_model.zig").percentDecoded;
const logHttpResponse = @import("respond.zig").logHttpResponse;
const sendErrorResponse = @import("errors.zig").sendErrorResponse;
const UNAUTHORIZED_ALLOW_HEADERS = @import("respond.zig").UNAUTHORIZED_ALLOW_HEADERS;

/// True if the connection's peer is a loopback address (127.0.0.0/8, ::1, or an
/// IPv4-mapped-IPv6 loopback). Loopback is the local machine → trusted and
/// exempt from the API key, so the same-host app never needs credentials.
fn ipIsLoopback(addr: std.Io.net.IpAddress) bool {
    return switch (addr) {
        .ip4 => |a4| a4.bytes[0] == 127, // 127.0.0.0/8
        .ip6 => |a6| blk: {
            const b = a6.bytes; // big-endian 16 bytes
            const zeros10: [10]u8 = @splat(0);
            // ::1
            const zeros15: [15]u8 = @splat(0);
            if (std.mem.eql(u8, b[0..15], &zeros15) and b[15] == 1) break :blk true;
            // ::ffff:127.x.x.x  (IPv4-mapped loopback)
            if (std.mem.eql(u8, b[0..10], &zeros10) and b[10] == 0xff and b[11] == 0xff and b[12] == 127) break :blk true;
            break :blk false;
        },
    };
}

/// Whether the API-key auth gate covers this peer at all: a key must be set,
/// and the peer must be non-loopback — unless `--api-key-strict` removed the
/// loopback exemption. Pure so the truth table is hermetically testable.
pub fn apiKeyGateApplies(key_set: bool, strict: bool, peer_loopback: bool) bool {
    return key_set and (!peer_loopback or strict);
}

pub fn peerIsLoopback(conn: *const Conn) bool {
    return ipIsLoopback(conn.stream.socket.address);
}

/// Whether this connection is trusted to tag committed KV-cache entries with `X-Session-Key`.
pub fn sessionTagAllowed(conn: *const Conn) bool {
    return switch (http_server.g_cache_transfer) {
        .disabled => false,
        .loopback => peerIsLoopback(conn),
        .key => !@import("../lan/http.zig").lanGateApplies(conn),
    };
}

/// Access policy for KV-cache transfer endpoints (`POST /v1/cache/import`, `X-Cache-Export: 1`).
pub const CacheTransferConfig = union(enum) {
    /// Default: cache transfer endpoints are disabled.
    disabled,
    /// `--cache-transfer`: enabled for loopback peers only (`127.0.0.0/8`, `::1`, `::ffff:127.0.0.0/104`).
    loopback,
    /// `--cache-transfer-key-file <path>`: enabled from any peer presenting `Authorization: Bearer <key>`
    /// matching this SHA-256 digest (0600 file, >= 32 bytes, distinct from `--api-key`, never loopback-exempt).
    key: [32]u8,

    pub fn enabled(self: CacheTransferConfig) bool {
        return self != .disabled;
    }
};

pub const CacheTransferVerdict = enum {
    allowed,
    disabled,
    forbidden_peer,
    unauthorized,
};

pub const CacheTransferLaunchError = error{
    CacheTransferKeyRefused,
    CacheTransferKeyIsServingKey,
};

/// Resolves `--cache-transfer` / `--cache-transfer-key-file` at launch. A key file follows the same
/// ownership and 0600 mode rules as `--ablation-key-file` and must not reuse `--api-key`.
pub fn launchCacheTransfer(
    gpa: std.mem.Allocator,
    loopback_flag: bool,
    key_file: ?[]const u8,
    serving_key: ?[]const u8,
) CacheTransferLaunchError!CacheTransferConfig {
    if (key_file) |path| {
        const digest = ablation.readKey(gpa, path) catch |e| {
            log.err("[cache-transfer] key file {s} refused: {s}\n", .{ path, switch (e) {
                error.NotRegular => "not a regular file",
                error.NotOwner => "not owned by the server's user",
                error.GroupOrOtherAccess => "mode must be 0600 (no group or other access)",
                error.TooShort => "the key must be at least 32 bytes",
                error.Unreadable, error.OutOfMemory => "unreadable",
                error.Unsupported => "--cache-transfer-key-file needs a POSIX host",
            } });
            return error.CacheTransferKeyRefused;
        };
        if (serving_key) |s| if (ablation.Set.digestMatches(digest, s)) {
            log.err("[cache-transfer] --cache-transfer-key-file holds the serving key; cache transfer needs a key of its own\n", .{});
            return error.CacheTransferKeyIsServingKey;
        };
        log.info("[cache-transfer] enabled: authenticated peers with key from {s}\n", .{path});
        return .{ .key = digest };
    }
    if (loopback_flag) {
        log.info("[cache-transfer] enabled: loopback peers only\n", .{});
        return .loopback;
    }
    return .disabled;
}

/// Evaluates whether a peer + request headers are allowed to invoke a cache-transfer operation.
pub fn checkCacheTransferAuth(
    cfg: CacheTransferConfig,
    peer: std.Io.net.IpAddress,
    raw_headers: []const u8,
    raw_path: []const u8,
) CacheTransferVerdict {
    switch (cfg) {
        .disabled => return .disabled,
        .loopback => {
            if (!ipIsLoopback(peer)) return .forbidden_peer;
            if (apiKeyGateApplies(http_server.g_api_key != null, http_server.g_api_key_strict, true) and
                !apiKeyAuthorized(raw_headers, raw_path))
            {
                return .unauthorized;
            }
            return .allowed;
        },
        .key => |digest| {
            const auth_hdr = findHeaderValueCI(raw_headers, "authorization") orelse return .unauthorized;
            if (!std.mem.startsWith(u8, auth_hdr, "Bearer ")) return .unauthorized;
            const presented = std.mem.trim(u8, auth_hdr["Bearer ".len..], " \t");
            if (!ablation.Set.digestMatches(digest, presented)) return .unauthorized;
            return .allowed;
        },
    }
}

/// Sends the HTTP refusal for a non-`.allowed` `CacheTransferVerdict`.
/// `is_dedicated_route` distinguishes dedicated cache-transfer routes (`/v1/cache/import`, which 404
/// when disabled) from overloaded inference routes (`X-Cache-Export: 1` on `/v1/chat/completions`,
/// which 403 when disabled).
pub fn sendCacheTransferDenial(
    allocator: std.mem.Allocator,
    stream: *Conn,
    verdict: CacheTransferVerdict,
    is_dedicated_route: bool,
) !void {
    switch (verdict) {
        .allowed => {},
        .disabled => {
            if (is_dedicated_route) {
                try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "Cache transfer is disabled; pass --cache-transfer or --cache-transfer-key-file", 404);
            } else {
                try sendErrorResponse(allocator, stream, "403 Forbidden", "forbidden", "Cache transfer is disabled; pass --cache-transfer or --cache-transfer-key-file", 403);
            }
        },
        .forbidden_peer => {
            try sendErrorResponse(allocator, stream, "403 Forbidden", "forbidden", "Cache transfer requires loopback peer or --cache-transfer-key-file", 403);
        },
        .unauthorized => {
            if (http_server.g_cache_transfer == .key) {
                try sendErrorResponse(allocator, stream, "401 Unauthorized", "authentication_error", "missing or invalid cache transfer key", 401);
            } else {
                try sendUnauthorized(stream);
            }
        },
    }
}

/// Case-insensitive HTTP header lookup in the raw header block. `name_lower`
/// must be lowercase (e.g. "authorization"). Returns the trimmed value, or null.
pub fn findHeaderValueCI(headers: []const u8, name_lower: []const u8) ?[]const u8 {
    var lines = std.mem.splitSequence(u8, headers, "\r\n");
    while (lines.next()) |line| {
        if (line.len <= name_lower.len + 1) continue;
        var match = true;
        for (0..name_lower.len) |j| {
            if (std.ascii.toLower(line[j]) != name_lower[j]) {
                match = false;
                break;
            }
        }
        if (!match) continue;
        if (line[name_lower.len] != ':') continue;
        return std.mem.trim(u8, line[name_lower.len + 1 ..], " \t");
    }
    return null;
}

/// Constant-time equality — avoids a timing oracle on the API key.
fn constTimeEql(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    var diff: u8 = 0;
    for (a, b) |x, y| diff |= x ^ y;
    return diff == 0;
}

/// Extract a query-string parameter from the RAW (un-stripped) request path,
/// e.g. `queryParamValue("/metrics?api_key=abc", "api_key") == "abc"`.
pub fn queryParamValue(raw_path: []const u8, name: []const u8) ?[]const u8 {
    const qpos = std.mem.indexOfScalar(u8, raw_path, '?') orelse return null;
    var it = std.mem.splitScalar(u8, raw_path[qpos + 1 ..], '&');
    while (it.next()) |pair| {
        const eq = std.mem.indexOfScalar(u8, pair, '=') orelse continue;
        if (std.mem.eql(u8, pair[0..eq], name)) return pair[eq + 1 ..];
    }
    return null;
}

/// True if an `Authorization: Basic base64(user:pass)` value carries the key as
/// its password (browser index/metrics pages send this after the Basic prompt).
/// A user-only entry (no colon) matching the key is also accepted.
fn basicAuthMatches(enc: []const u8, key: []const u8) bool {
    var dbuf: [1024]u8 = undefined;
    const dec_len = std.base64.standard.Decoder.calcSizeForSlice(enc) catch return false;
    if (dec_len > dbuf.len) return false;
    std.base64.standard.Decoder.decode(dbuf[0..dec_len], enc) catch return false;
    const creds = dbuf[0..dec_len];
    const pass = if (std.mem.indexOfScalar(u8, creds, ':')) |c| creds[c + 1 ..] else creds;
    return constTimeEql(pass, key);
}

/// True if the request is authorized for the configured `g_api_key` (or no key
/// is set). Accepts, in order: `Authorization: Bearer <key>`, HTTP Basic (key as
/// the password), `x-api-key: <key>` (Anthropic), and `?api_key=` / `?key=`.
pub fn apiKeyAuthorized(raw_headers: []const u8, raw_path: []const u8) bool {
    const key = http_server.g_api_key orelse return true;
    if (findHeaderValueCI(raw_headers, "authorization")) |auth| {
        if (std.mem.startsWith(u8, auth, "Bearer ")) {
            if (constTimeEql(std.mem.trim(u8, auth["Bearer ".len..], " \t"), key)) return true;
        } else if (std.mem.startsWith(u8, auth, "Basic ")) {
            if (basicAuthMatches(std.mem.trim(u8, auth["Basic ".len..], " \t"), key)) return true;
        }
    }
    if (findHeaderValueCI(raw_headers, "x-api-key")) |xk| {
        if (constTimeEql(xk, key)) return true;
    }
    if (findHeaderValueCI(raw_headers, "cookie")) |c| {
        if (pairValue(c, ';', KEY_COOKIE)) |v| if (encodedKeyMatches(v, key)) return true;
    }
    var qbuf: [512]u8 = undefined;
    inline for (.{ "api_key", "key" }) |name| {
        if (queryDecoded(&qbuf, raw_path, name)) |q| if (constTimeEql(q, key)) return true;
    }
    return false;
}

/// The console's login cookie: the key as the form percent-encoded it. HttpOnly, so page
/// scripts never read it; the browser sends it on the page load and on the console's own calls.
const KEY_COOKIE = "mlx_serve_key";

pub fn encodedKeyMatches(encoded: []const u8, key: []const u8) bool {
    var buf: [512]u8 = undefined;
    return constTimeEql(percentDecoded(&buf, encoded) orelse return false, key);
}

/// Form encoding's own alphabet: nothing that could end a header or a cookie.
pub fn cookieSafe(v: []const u8) bool {
    if (v.len == 0 or v.len > 512) return false;
    for (v) |c| if (!std.ascii.isAlphanumeric(c) and std.mem.indexOfScalar(u8, "%+-._*~", c) == null) return false;
    return true;
}

/// The still-encoded `api_key` of the login form's POST, or null when this is not that POST.
pub fn loginFormKey(method: []const u8, path: []const u8, content_type: []const u8, body: []const u8) ?[]const u8 {
    if (!std.mem.eql(u8, method, "POST") or !isRoute(path, "/")) return null;
    if (!std.mem.startsWith(u8, content_type, "application/x-www-form-urlencoded")) return null;
    const v = pairValue(body, '&', "api_key") orelse return null;
    return if (cookieSafe(v)) v else null;
}

/// Trade an accepted key for the cookie and send the browser to the bare page, so the key
/// never sits in the URL (`./` keeps a reverse proxy's path prefix).
pub fn sendKeyCookie(stream: *Conn, encoded: []const u8) !void {
    logHttpResponse("303 See Other", "", "");
    var hdr_buf: [1024]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 303 See Other\r\nLocation: ./\r\nSet-Cookie: " ++ KEY_COOKIE ++ "={s}; Path=/; HttpOnly; SameSite=Strict\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", .{encoded}) catch return error.Overflow;
    try stream.writeAll(hdr);
    stream.length_framed = true;
}

/// A browser opening the page (`GET /` that accepts HTML) without the key gets
/// the login form: a Basic prompt is easy to miss and never returns once cancelled.
pub fn wantsLoginPage(method: []const u8, path: []const u8, raw_headers: []const u8) bool {
    if (!std.mem.eql(u8, method, "GET") or !isRoute(path, "/")) return false;
    const accept = findHeaderValueCI(raw_headers, "accept") orelse return false;
    return std.mem.indexOf(u8, accept, "text/html") != null;
}

/// The login form, saying so when a key was offered and refused. Script-free:
/// it POSTs the key back to `/` (`loginFormKey`), which answers with the cookie.
pub fn loginPage(raw_headers: []const u8, raw_path: []const u8) []const u8 {
    const cookie_key = if (findHeaderValueCI(raw_headers, "cookie")) |c| pairValue(c, ';', KEY_COOKIE) else null;
    const tried = findHeaderValueCI(raw_headers, "authorization") != null or
        findHeaderValueCI(raw_headers, "x-api-key") != null or
        cookie_key != null or
        queryParamValue(raw_path, "api_key") != null or
        queryParamValue(raw_path, "key") != null;
    return if (tried) loginPageHtml("<p class=err>That key was not accepted. Try again.</p>") else loginPageHtml("");
}

pub fn loginPageHtml(comptime note: []const u8) []const u8 {
    return
    \\<!doctype html>
    \\<html lang="en"><head><meta charset="utf-8">
    \\<meta name="viewport" content="width=device-width,initial-scale=1">
    \\<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'">
    \\<title>MLX Serve · API key required</title>
    \\<style>
    \\:root{color-scheme:light dark;font-family:system-ui,sans-serif}
    \\body{margin:0;min-height:100vh;display:grid;place-items:center;background:Canvas;color:CanvasText}
    \\main{width:min(22rem,calc(100vw - 2rem));padding:1.75rem;border:1px solid color-mix(in srgb,CanvasText 15%,transparent);border-radius:.75rem}
    \\h1{font-size:1.15rem;margin:0 0 .5rem}p{margin:0 0 1rem;font-size:.9rem;opacity:.8}
    \\.err{color:#d33;opacity:1}
    \\input,button{box-sizing:border-box;width:100%;font:inherit;padding:.6rem .7rem;border-radius:.5rem}
    \\input{border:1px solid color-mix(in srgb,CanvasText 30%,transparent);background:Canvas;color:CanvasText;margin-bottom:.75rem}
    \\button{border:0;background:#0a6cff;color:#fff;cursor:pointer}
    \\</style></head><body><main>
    \\<h1>API key required</h1>
    \\<p>This MLX Serve server requires an API key. Enter the key the server was started with.</p>
    \\
++ note ++
    \\<form method="post">
    \\<input type="password" name="api_key" placeholder="API key" autocomplete="current-password" required autofocus>
    \\<button type="submit">Continue</button>
    \\</form>
    \\</main></body></html>
    \\
    ;
}

/// Send a 401 with a Basic-auth challenge so browsers prompt for the key on the
/// index + metrics pages; API clients read the JSON error body.
pub fn sendUnauthorized(stream: *Conn) !void {
    const body = "{\"error\":{\"message\":\"missing or invalid API key\",\"type\":\"authentication_error\"}}";
    logHttpResponse("401 Unauthorized", "application/json", body);
    if (stream.ws_mode) |bridge| {
        if (body.len > 0) try bridge.sendText(body);
        return;
    }
    var hdr_buf: [512]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nWWW-Authenticate: Basic realm=\"mlx-serve\"\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: " ++ UNAUTHORIZED_ALLOW_HEADERS ++ "\r\n\r\n", .{body.len}) catch return error.Overflow;
    try stream.writeAll(hdr);
    stream.length_framed = true;
    try stream.writeAll(body);
}

// ── tests ──

test "apiKeyGateApplies: strict removes exactly the loopback exemption" {
    // No key: fully open regardless of strictness or peer.
    try std.testing.expect(!apiKeyGateApplies(false, false, true));
    try std.testing.expect(!apiKeyGateApplies(false, true, false));
    // Key, default: loopback exempt, network gated.
    try std.testing.expect(!apiKeyGateApplies(true, false, true));
    try std.testing.expect(apiKeyGateApplies(true, false, false));
    // Key, strict: everyone gated.
    try std.testing.expect(apiKeyGateApplies(true, true, true));
    try std.testing.expect(apiKeyGateApplies(true, true, false));
}

test "apiKeyAuthorized accepts Bearer, x-api-key, Basic, and query param" {
    const prev = http_server.g_api_key;
    defer http_server.g_api_key = prev;
    http_server.g_api_key = "s3cret";

    // Bearer
    try std.testing.expect(apiKeyAuthorized("Authorization: Bearer s3cret\r\n", "/v1/chat/completions"));
    // x-api-key (Anthropic)
    try std.testing.expect(apiKeyAuthorized("x-api-key: s3cret\r\n", "/v1/messages"));
    // Basic — base64("user:s3cret") = "dXNlcjpzM2NyZXQ="
    try std.testing.expect(apiKeyAuthorized("Authorization: Basic dXNlcjpzM2NyZXQ=\r\n", "/"));
    // Query param (browser fallback)
    try std.testing.expect(apiKeyAuthorized("", "/metrics.json?api_key=s3cret"));
    try std.testing.expect(apiKeyAuthorized("", "/metrics?key=s3cret"));

    // Wrong / missing key is rejected
    try std.testing.expect(!apiKeyAuthorized("Authorization: Bearer wrong\r\n", "/"));
    try std.testing.expect(!apiKeyAuthorized("x-api-key: nope\r\n", "/"));
    try std.testing.expect(!apiKeyAuthorized("", "/v1/models"));

    // The login form percent-encodes the key it sends back (a base64 key has + / =).
    http_server.g_api_key = "a+b/c=";
    try std.testing.expect(apiKeyAuthorized("", "/?api_key=a%2Bb%2Fc%3D"));
    try std.testing.expect(!apiKeyAuthorized("", "/?api_key=a%2Bb"));

    // No key configured ⇒ always authorized (open mode)
    http_server.g_api_key = null;
    try std.testing.expect(apiKeyAuthorized("", "/v1/chat/completions"));
}

test "a browser opening the page without the key gets a login form; API clients do not" {
    const html = "Accept: text/html,application/xhtml+xml,*/*;q=0.8\r\n";
    try std.testing.expect(wantsLoginPage("GET", "/", html));
    try std.testing.expect(!wantsLoginPage("GET", "/", "Accept: */*\r\n"));
    try std.testing.expect(!wantsLoginPage("GET", "/v1/models", html));
    try std.testing.expect(!wantsLoginPage("POST", "/", html));

    const first = loginPage("", "/");
    try std.testing.expect(std.mem.indexOf(u8, first, "name=\"api_key\"") != null);
    // POST, so the key never lands in the URL, history or a Referer (#887).
    try std.testing.expect(std.mem.indexOf(u8, first, "method=\"post\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, first, "<script") == null);
    try std.testing.expect(std.mem.indexOf(u8, first, "not accepted") == null);
    // A key that was tried and failed says so, whichever way it came.
    try std.testing.expect(std.mem.indexOf(u8, loginPage("", "/?api_key=bad"), "not accepted") != null);
    try std.testing.expect(std.mem.indexOf(u8, loginPage("Authorization: Basic dTpi\r\n", "/"), "not accepted") != null);
}

test "the login form's key comes back as a cookie that authorizes the page and its API calls (#887)" {
    const prev = http_server.g_api_key;
    defer http_server.g_api_key = prev;
    http_server.g_api_key = "a+b/c=";
    // The form body is percent-encoded; its value is what the cookie stores.
    try std.testing.expectEqualStrings("a%2Bb%2Fc%3D", loginFormKey("POST", "/", "application/x-www-form-urlencoded", "api_key=a%2Bb%2Fc%3D").?);
    try std.testing.expect(loginFormKey("GET", "/", "application/x-www-form-urlencoded", "api_key=x") == null);
    try std.testing.expect(loginFormKey("POST", "/v1/models", "application/x-www-form-urlencoded", "api_key=x") == null);
    try std.testing.expect(loginFormKey("POST", "/", "application/json", "{}") == null);
    try std.testing.expect(apiKeyAuthorized("Cookie: theme=dark; " ++ KEY_COOKIE ++ "=a%2Bb%2Fc%3D\r\n", "/v1/models"));
    try std.testing.expect(!apiKeyAuthorized("Cookie: " ++ KEY_COOKIE ++ "=a%2Bb\r\n", "/v1/models"));
    try std.testing.expect(!apiKeyAuthorized("Cookie: x" ++ KEY_COOKIE ++ "=a%2Bb%2Fc%3D\r\n", "/"));
}

test "ipIsLoopback exempts local addresses only" {
    // IPv4 loopback (whole 127.0.0.0/8) is exempt; a LAN address is not.
    try std.testing.expect(ipIsLoopback(.{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 0 } }));
    try std.testing.expect(ipIsLoopback(.{ .ip4 = .{ .bytes = .{ 127, 3, 2, 1 }, .port = 0 } }));
    try std.testing.expect(!ipIsLoopback(.{ .ip4 = .{ .bytes = .{ 192, 168, 1, 42 }, .port = 0 } }));
    try std.testing.expect(!ipIsLoopback(.{ .ip4 = .{ .bytes = .{ 10, 0, 0, 5 }, .port = 0 } }));
    // IPv6 ::1 is exempt; a routable v6 address is not.
    try std.testing.expect(ipIsLoopback(.{ .ip6 = .{ .bytes = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, .port = 0 } }));
    // IPv4-mapped loopback ::ffff:127.0.0.1
    try std.testing.expect(ipIsLoopback(.{ .ip6 = .{ .bytes = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 127, 0, 0, 1 }, .port = 0 } }));
    try std.testing.expect(!ipIsLoopback(.{ .ip6 = .{ .bytes = .{ 0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, .port = 0 } }));
}

test "checkCacheTransferAuth: disabled, loopback-only, and key-file policies" {
    const prev_key = http_server.g_api_key;
    const prev_strict = http_server.g_api_key_strict;
    defer {
        http_server.g_api_key = prev_key;
        http_server.g_api_key_strict = prev_strict;
    }
    http_server.g_api_key = null;
    http_server.g_api_key_strict = false;

    const lo4: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 0 } };
    const lo6: std.Io.net.IpAddress = .{ .ip6 = .{ .bytes = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, .port = 0 } };
    const lan: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 192, 168, 1, 10 }, .port = 0 } };
    const private10: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 198, 51, 100, 2 }, .port = 0 } };

    // Disabled rejects every peer.
    try std.testing.expectEqual(CacheTransferVerdict.disabled, checkCacheTransferAuth(.disabled, lo4, "", "/v1/cache/import"));
    try std.testing.expectEqual(CacheTransferVerdict.disabled, checkCacheTransferAuth(.disabled, lan, "", "/v1/cache/import"));

    // Loopback mode admits only loopback peers (no private-subnet exemption).
    try std.testing.expectEqual(CacheTransferVerdict.allowed, checkCacheTransferAuth(.loopback, lo4, "", "/v1/cache/import"));
    try std.testing.expectEqual(CacheTransferVerdict.allowed, checkCacheTransferAuth(.loopback, lo6, "", "/v1/cache/import"));
    try std.testing.expectEqual(CacheTransferVerdict.forbidden_peer, checkCacheTransferAuth(.loopback, lan, "", "/v1/cache/import"));
    try std.testing.expectEqual(CacheTransferVerdict.forbidden_peer, checkCacheTransferAuth(.loopback, private10, "", "/v1/cache/import"));

    // Loopback mode respects --api-key-strict when --api-key is set.
    http_server.g_api_key = "serving-secret";
    http_server.g_api_key_strict = false;
    try std.testing.expectEqual(CacheTransferVerdict.allowed, checkCacheTransferAuth(.loopback, lo4, "", "/v1/cache/import"));
    http_server.g_api_key_strict = true;
    try std.testing.expectEqual(CacheTransferVerdict.unauthorized, checkCacheTransferAuth(.loopback, lo4, "", "/v1/cache/import"));
    try std.testing.expectEqual(CacheTransferVerdict.allowed, checkCacheTransferAuth(.loopback, lo4, "Authorization: Bearer serving-secret\r\n", "/v1/cache/import"));

    // Key mode requires Bearer matching its own SHA-256 digest from any IP (including loopback) and rejects the serving key.
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("transfer-secret-0123456789abcdef0123", &digest, .{});
    const keyed: CacheTransferConfig = .{ .key = digest };
    try std.testing.expectEqual(CacheTransferVerdict.unauthorized, checkCacheTransferAuth(keyed, lo4, "", "/v1/cache/import"));
    try std.testing.expectEqual(CacheTransferVerdict.unauthorized, checkCacheTransferAuth(keyed, lan, "Authorization: Bearer serving-secret\r\n", "/v1/cache/import"));
    try std.testing.expectEqual(CacheTransferVerdict.unauthorized, checkCacheTransferAuth(keyed, lan, "x-api-key: transfer-secret-0123456789abcdef0123\r\n", "/v1/cache/import"));
    try std.testing.expectEqual(CacheTransferVerdict.allowed, checkCacheTransferAuth(keyed, lo4, "Authorization: Bearer transfer-secret-0123456789abcdef0123\r\n", "/v1/cache/import"));
    try std.testing.expectEqual(CacheTransferVerdict.allowed, checkCacheTransferAuth(keyed, lan, "Authorization: Bearer transfer-secret-0123456789abcdef0123\r\n", "/v1/cache/import"));
    try std.testing.expectEqual(CacheTransferVerdict.allowed, checkCacheTransferAuth(keyed, private10, "Authorization: Bearer transfer-secret-0123456789abcdef0123\r\n", "/v1/cache/import"));
}

test "launchCacheTransfer rejects a key file that reuses --api-key and accepts a distinct 0600 key file" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    const a = std.testing.allocator;
    const path = try std.fs.path.join(a, &.{ root, "transfer.key" });
    defer a.free(path);
    const path_z = try a.dupeSentinel(u8, path, 0);
    defer a.free(path_z);
    const key = "cache-transfer-key-0123456789-0123456789";
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "transfer.key", .data = key ++ "\n" });
    _ = std.c.chmod(path_z, 0o600);

    try std.testing.expectEqual(CacheTransferConfig.disabled, try launchCacheTransfer(a, false, null, null));
    try std.testing.expectEqual(CacheTransferConfig.loopback, try launchCacheTransfer(a, true, null, null));
    try std.testing.expectError(error.CacheTransferKeyIsServingKey, launchCacheTransfer(a, false, path, key));

    const cfg = try launchCacheTransfer(a, false, path, "different-serving-key");
    try std.testing.expect(cfg == .key);
    try std.testing.expect(ablation.Set.digestMatches(cfg.key, key));
}
