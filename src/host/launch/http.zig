//! handleLaunchScript

const std = @import("std");
const launch_mod = @import("launch.zig");
const testing = std.testing;
const http_server = @import("../http/server.zig");

const Conn = @import("../http/conn.zig").Conn;
const queryModel = @import("../api/request_model.zig").queryModel;
const queryDecoded = @import("../api/request_model.zig").queryDecoded;
const renderModelsBody = @import("../api/models.zig").renderModelsBody;
const sendResponse = @import("../http/respond.zig").sendResponse;
const apiKeyGateApplies = @import("../http/auth.zig").apiKeyGateApplies;
const peerIsLoopback = @import("../http/auth.zig").peerIsLoopback;
const findHeaderValueCI = @import("../http/auth.zig").findHeaderValueCI;
const queryParamValue = @import("../http/auth.zig").queryParamValue;
const sendErrorResponse = @import("../http/errors.zig").sendErrorResponse;

/// `GET /launch?agent=<name>[&model=<id>]`: a shell script that configures a
/// coding agent for this server and starts it (`curl -fsSL … | sh`, from the
/// console's Code Launcher). The agent targets the host the request reached.
pub fn handleLaunchScript(allocator: std.mem.Allocator, stream: *Conn, raw_path: []const u8, raw_headers: []const u8, lan_filtered: bool) !void {
    if (launchRefusal(apiKeyGateApplies(http_server.g_api_key != null, http_server.g_api_key_strict, peerIsLoopback(stream)))) |reason|
        return sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
    const kind = launch_mod.AgentKind.fromName(queryParamValue(raw_path, "agent") orelse "");
    if (kind == null or !launch_mod.webLaunchable(kind.?))
        return sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "agent must be one of: claude, pi, omp, opencode2, grok", 400);
    var url_buf: [300]u8 = undefined;
    var base_buf: [300]u8 = undefined;
    const base_url = launchBaseUrl(&url_buf, queryDecoded(&base_buf, raw_path, "base"), findHeaderValueCI(raw_headers, "host") orelse "") orelse
        return sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "base must be http(s)://host[:port][/path] and the Host header a plain host[:port]", 400);
    const body = renderModelsBody(allocator, stream.io, lan_filtered) catch |err| switch (err) {
        error.RegistryNotReady => return sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Registry not ready", 503),
        else => return err,
    };
    defer allocator.free(body);
    var models = try launch_mod.parseChatEntries(allocator, body);
    defer models.deinit();
    var model_buf: [1024]u8 = undefined;
    const entry = launch_mod.pickEntry(models.entries, queryModel(&model_buf, raw_path)) orelse
        return sendErrorResponse(allocator, stream, "404 Not Found", "invalid_request_error", "no such chat model on this server", 404);
    const script = try launch_mod.remoteScript(allocator, kind.?, base_url, entry.id, entry.budget, models.entries);
    defer allocator.free(script);
    try sendResponse(stream, "200 OK", "text/plain; charset=utf-8", script);
}

/// The agents' configs carry a placeholder key, so a script fetched past the `--api-key`
/// gate would start an agent that 401s on every request.
fn launchRefusal(key_gated: bool) ?[]const u8 {
    return if (key_gated) "the Code Launcher does not carry this server's --api-key: run `mlx-serve launch <agent>` with the key on that machine" else null;
}

/// The script's server URL: the console's own `base` (scheme and path prefix kept, the
/// reverse-proxy case), else `http://<Host>`. Both are held to bytes that cannot break
/// out of the script's quoting.
fn launchBaseUrl(buf: []u8, base: ?[]const u8, host: []const u8) ?[]const u8 {
    if (base) |b| {
        const rest = if (std.mem.startsWith(u8, b, "https://")) b["https://".len..] else if (std.mem.startsWith(u8, b, "http://")) b["http://".len..] else return null;
        const trimmed = std.mem.trimEnd(u8, rest, "/");
        if (trimmed.len == 0) return null;
        for (trimmed) |c| if (!(std.ascii.isAlphanumeric(c) or std.mem.indexOfScalar(u8, ".-:[]/_~", c) != null)) return null;
        return std.fmt.bufPrint(buf, "{s}{s}", .{ b[0 .. b.len - rest.len], trimmed }) catch null;
    }
    if (host.len == 0) return null;
    for (host) |c| if (!(std.ascii.isAlphanumeric(c) or std.mem.indexOfScalar(u8, ".-:[]", c) != null)) return null;
    return std.fmt.bufPrint(buf, "http://{s}", .{host}) catch null;
}

// ── tests ──

test "launchBaseUrl: a served launch script targets the Host the request reached, never a quote-breaking one" {
    var buf: [300]u8 = undefined;
    try testing.expectEqualStrings("http://192.168.1.20:11234", launchBaseUrl(&buf, null, "192.168.1.20:11234").?);
    try testing.expectEqualStrings("http://[::1]:8080", launchBaseUrl(&buf, null, "[::1]:8080").?);
    for ([_][]const u8{ "", "h'; rm -rf ~;'", "h:1/x", "h $(id)", "h\"" }) |bad| try testing.expect(launchBaseUrl(&buf, null, bad) == null);
}

test "launchBaseUrl: the console's own base keeps its scheme and path prefix, and is held to the same quoting bar" {
    var buf: [300]u8 = undefined;
    try testing.expectEqualStrings("https://mac.tail1.ts.net/mlx-serve", launchBaseUrl(&buf, "https://mac.tail1.ts.net/mlx-serve/", "ignored").?);
    try testing.expectEqualStrings("http://10.0.0.5:11234", launchBaseUrl(&buf, "http://10.0.0.5:11234", "").?);
    for ([_][]const u8{ "", "ftp://h", "h:1", "https://h/'$(id)'", "http://h/a b", "https://h\"", "http://" }) |bad|
        try testing.expect(launchBaseUrl(&buf, bad, "h:1") == null);
}

test "launchRefusal: under --api-key a gated request is refused by name, a loopback one is served" {
    try testing.expect(launchRefusal(true) != null);
    try testing.expect(launchRefusal(false) == null);
}
