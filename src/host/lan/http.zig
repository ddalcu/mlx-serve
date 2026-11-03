//! LAN gate/proxy, provider proxy

const std = @import("std");
const crt = @import("../../sdk/util/crt.zig");
const log = @import("../../sdk/util/log.zig");
const model_registry_mod = @import("../registry/registry.zig");
const ModelRegistry = model_registry_mod.ModelRegistry;
const io_util = @import("../../sdk/util/io_util.zig");
const lan_mod = @import("lan.zig");
const providers_mod = @import("../api/providers.zig");
const nowMsMonotonic = io_util.nowMsMonotonic;
const http_server = @import("../http/server.zig");
const request_model = @import("../api/request_model.zig");

const Conn = @import("../http/conn.zig").Conn;
const isRoute = @import("../http/router.zig").isRoute;
const resolveRequestModelId = @import("../api/request_model.zig").resolveRequestModelId;
const parseModelFromRequest = @import("../api/request_model.zig").parseModelFromRequest;
const sendResponse = @import("../http/respond.zig").sendResponse;
const logSafePath = @import("../http/respond.zig").logSafePath;
const peerIsLoopback = @import("../http/auth.zig").peerIsLoopback;
const findHeaderValueCI = @import("../http/auth.zig").findHeaderValueCI;
const sendErrorResponse = @import("../http/errors.zig").sendErrorResponse;
const jsonEscape = @import("../api/format.zig").jsonEscape;

/// True when the LAN-share gate governs this request: sharing on, keyless
/// mode (a configured --api-key already gated non-loopback traffic), and the
/// client is not local.
pub fn lanGateApplies(stream: *const Conn) bool {
    const l = http_server.g_lan orelse return false;
    return l.sharing() and http_server.g_api_key == null and !peerIsLoopback(stream);
}

/// True when this request arrived through another mlx-serve's LAN tunnel
/// (`lan.tunnel` stamps `X-MLX-LAN: 1` on every request it forwards).
/// Tunneled requests are never proxied again — THAT is the multi-hop bound
/// (depth 1 by construction), so proxying no longer keys on loopback-ness:
/// any direct client (the local app, the agent-sandbox VM arriving over the
/// NAT interface, a phone on the LAN) may initiate the single hop.
pub fn isTunneledRequest(raw_headers: []const u8) bool {
    return findHeaderValueCI(raw_headers, "x-mlx-lan") != null;
}

/// LAN-share gate decision for one non-loopback request; null = allowed.
/// The effective model resolves exactly like dispatch will (unknown/absent
/// ids fall back to the default model), so the gate can never disagree with
/// what would actually run.
pub fn lanShareDenial(l: *lan_mod.Lan, registry: *ModelRegistry, method: []const u8, path: []const u8, body: []const u8, content_type: []const u8, tunneled: bool) ?[]const u8 {
    switch (lan_mod.routeClass(method, path)) {
        .open => return null,
        .denied => return "This endpoint is host-local; LAN sharing exposes inference on shared models only",
        .model_gated => {},
    }
    var mid_buf: [512]u8 = undefined;
    const mid = lan_mod.unescapeJsonSlashes(
        &mid_buf,
        parseModelFromRequest(body, content_type) orelse "",
    );
    if (lan_mod.splitRemoteId(mid)) |rid| if (registry.peek(mid) == null) {
        // A provider is the host owner's paid key: never reachable from the LAN.
        if (http_server.g_providers) |p| if (p.isProvider(rid.peer)) return "Provider models are host-local";
    };
    if (lan_mod.splitRemoteId(mid) != null and registry.peek(mid) == null) {
        // A remote (@peer) id from a DIRECT client is allowed — dispatch
        // proxies exactly one hop and the peer's own gate governs its model
        // (the old blanket deny also 403'd the agent-sandbox guest, which is
        // non-loopback by construction; live). A request that
        // arrived through a peer's tunnel never hops again.
        if (tunneled) return "Remote (@peer) model ids cannot be proxied onward — ask that peer directly";
        return null;
    }
    const effective = switch (resolveRequestModelId(registry, &request_model.g_model_aliases, mid, path)) {
        .id => |id| if (id.len > 0 and !std.mem.eql(u8, id, "mlx-serve")) id else registry.default_id,
        .unknown_name => registry.default_id,
        // Dispatch answers 404 and runs nothing.
        .unknown_path => return null,
    };
    if (!l.sharedAllows(effective)) return "Model not shared on this host";
    return null;
}

/// How long a proxied request waits for discovery to converge before the
/// honest "peer offline" 404. Covers a local restart's cold peer table AND a
/// peer Mac mid-reboot/redeploy: the peer needs to boot, advertise, and be
/// re-fetched — seconds, not milliseconds. A genuinely dead peer costs the
/// client this long once; an unlisted model or discovery-off never waits.
const LAN_PEER_WAIT_MS: i64 = 15_000;

/// Proxy a request naming `<bare>@<peer>` to that peer (lan.tunnel). The
/// failure modes are deliberately distinct — the live bite was one instant
/// "peer offline" 404 covering all three:
///   • discovery off on THIS server → say so (a share-only boot can never
///     resolve a peer, and "offline" sent the user debugging the wrong Mac);
///   • peer known but model unlisted → fail fast, honestly;
///   • peer unknown → poke discovery and wait up to LAN_PEER_WAIT_MS
///     (client disconnect abandons the wait), then 404.
/// 502 when the peer resolves but stops accepting. Never a silent fallback
/// to the local default model.
/// `full_id` is canonical (`\/` collapsed) and `raw_model` the body's value as written: Swift/PHP
/// clients send `ddalcu\/gemma…@peer`, the peer table stores `ddalcu/gemma…`, and the canonical
/// bare id replaces the raw bytes, or the peer's own scanner would miss it and serve its default.
pub fn handleLanProxy(allocator: std.mem.Allocator, stream: *Conn, l: *lan_mod.Lan, method: []const u8, raw_path: []const u8, body: []const u8, raw_model: ?[]const u8, full_id: []const u8) !void {
    const rid = lan_mod.splitRemoteId(full_id).?;
    if (!l.discover) {
        try sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "This id names a LAN peer's model, but LAN discovery is off on this server — start it with --lan-discover (app: Settings > LAN Sharing > Use models shared by other Macs)", 404);
        return;
    }
    const deadline = nowMsMonotonic(stream.io) + LAN_PEER_WAIT_MS;
    const remote: lan_mod.Remote = remote: while (true) {
        switch (l.lookupRemote(full_id)) {
            .found => |r| break :remote r,
            .model_unlisted => {
                try sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "The LAN peer no longer shares this model", 404);
                return;
            },
            .peer_unknown => {
                if (nowMsMonotonic(stream.io) >= deadline) {
                    try sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "LAN peer for this model is offline (waited 15 s for discovery)", 404);
                    return;
                }
                if (stream.peerClosed()) return; // client gave up while we waited
                l.pokeDiscovery();
                crt.sleepMs(250);
            },
        }
    };
    const rewritten = try proxiedBody(allocator, body, raw_model, rid.bare);
    defer allocator.free(rewritten);
    const key = l.peerKey(allocator, rid.peer);
    defer if (key) |k| allocator.free(k);
    log.info("[lan] proxy {s} {s} -> \"{s}\" @ {d}.{d}.{d}.{d}:{d}\n", .{ method, logSafePath(raw_path), rid.peer, remote.ip4[0], remote.ip4[1], remote.ip4[2], remote.ip4[3], remote.port });
    lan_mod.tunnel(remote, method, raw_path, rewritten, key, stream) catch {
        try sendErrorResponse(allocator, stream, "502 Bad Gateway", "lan_peer_unreachable", "LAN peer did not accept the connection", 502);
    };
}

/// Proxy `<bare>@<provider>` to the provider's /v1/chat/completions. Only that
/// surface exists upstream, so every other model-gated route is a named 400
/// rather than a request the provider would reject in its own words.
/// `body` to forward with its model value replaced by `bare`: `raw` is that value as the client
/// wrote it (a slice of `body`); a model named in the query string leaves the body as is.
fn proxiedBody(allocator: std.mem.Allocator, body: []const u8, raw: ?[]const u8, bare: []const u8) ![]u8 {
    const r = raw orelse return allocator.dupe(u8, body);
    const bare_json = try jsonEscape(allocator, bare); // canonical again, so a backslash is re-escaped
    defer allocator.free(bare_json);
    return lan_mod.rewriteModelValue(allocator, body, r, bare_json[1 .. bare_json.len - 1]);
}

pub fn handleProviderProxy(allocator: std.mem.Allocator, stream: *Conn, prov: *providers_mod.Providers, path: []const u8, body: []const u8, raw_model: ?[]const u8, full_id: []const u8) !void {
    if (!isRoute(path, "/v1/chat/completions")) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Provider models are served on /v1/chat/completions only", 400);
        return;
    }
    var up = prov.lookup(allocator, full_id) orelse {
        try sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "No such provider", 404);
        return;
    };
    defer up.deinit(allocator);
    const rewritten = try proxiedBody(allocator, body, raw_model, up.bare);
    defer allocator.free(rewritten);
    log.info("[providers] proxy \"{s}\" -> {s}/chat/completions\n", .{ up.bare, up.url });
    providers_mod.proxyChat(allocator, stream.io, up, rewritten, stream) catch {
        try sendErrorResponse(allocator, stream, "502 Bad Gateway", "provider_unreachable", "Provider did not answer", 502);
    };
}

pub fn handleProvidersReload(allocator: std.mem.Allocator, stream: *Conn) !void {
    const p = http_server.g_providers orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "providers_unavailable", "Providers are disabled on this server", 503);
        return;
    };
    const n = p.reload() catch |err| {
        const msg = try std.fmt.allocPrint(allocator, "providers.json could not be read: {s}", .{@errorName(err)});
        defer allocator.free(msg);
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, 400);
        return;
    };
    const body = try std.fmt.allocPrint(allocator, "{{\"providers\":{d}}}", .{n});
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

// ── tests ──

test "lanShareDenial: shared inference surface only, resolved like dispatch" {
    const a = std.testing.allocator;
    const reg = try ModelRegistry.init(a, std.Io.Threaded.global_single_threaded.io(), null, 8, 0, null);
    defer reg.deinit();
    const shared_entry = try reg.registerStub("gemma-4-e4b-it-4bit", "/m/g", 1);
    _ = try reg.registerStub("qwen3.6-27b", "/m/q", 1);
    reg.default_id = shared_entry.id;

    var l = lan_mod.Lan{
        .alloc = a,
        .port = 0,
        .discover = false,
        .peers = .init(a),
        .known = .init(a),
        .share = try lan_mod.SharedSet.parse(a, "gemma-4-e4b-it-4bit"),
    };
    defer {
        l.share.?.deinit(a);
        l.peers.deinit();
        l.known.deinit();
    }

    // Open routes pass with no model check; host-local ones are denied.
    try std.testing.expect(lanShareDenial(&l, reg, "GET", "/health", "", "application/json", false) == null);
    try std.testing.expect(lanShareDenial(&l, reg, "GET", "/v1/models", "", "application/json", false) == null);
    try std.testing.expect(lanShareDenial(&l, reg, "OPTIONS", "/v1/messages", "", "application/json", false) == null);
    try std.testing.expect(lanShareDenial(&l, reg, "POST", "/v1/load-model", "{}", "application/json", false) != null);
    try std.testing.expect(lanShareDenial(&l, reg, "GET", "/metrics", "", "application/json", false) != null);
    try std.testing.expect(lanShareDenial(&l, reg, "GET", "/", "", "application/json", false) != null);

    // Shared model allowed; unshared denied — on chat AND media surfaces.
    try std.testing.expect(lanShareDenial(&l, reg, "POST", "/v1/chat/completions", "{\"model\":\"gemma-4-e4b-it-4bit\"}", "application/json", false) == null);
    try std.testing.expect(lanShareDenial(&l, reg, "POST", "/v1/chat/completions", "{\"model\":\"qwen3.6-27b\"}", "application/json", false) != null);
    try std.testing.expect(lanShareDenial(&l, reg, "POST", "/v1/images/generations", "{\"model\":\"qwen3.6-27b\"}", "application/json", false) != null);

    // Omitted / unknown ids resolve to the default model exactly like
    // dispatch will — here the default is shared, so both pass.
    try std.testing.expect(lanShareDenial(&l, reg, "POST", "/v1/chat/completions", "{}", "application/json", false) == null);
    try std.testing.expect(lanShareDenial(&l, reg, "POST", "/v1/messages", "{\"model\":\"gpt-4\"}", "application/json", false) == null);
    // A PATH names its own entry, never the default: the unshared model's dir is denied.
    try std.testing.expect(lanShareDenial(&l, reg, "POST", "/v1/chat/completions", "{\"model\":\"/m/q\"}", "application/json", false) != null);

    // @peer ids: a DIRECT client (not tunneled) may initiate the single hop —
    // the old blanket deny also 403'd the agent-sandbox guest, which reaches
    // this host over the VM NAT interface. A request that
    // ARRIVED through a peer's tunnel never hops again (the multi-hop bound).
    try std.testing.expect(lanShareDenial(&l, reg, "POST", "/v1/messages", "{\"model\":\"x@peer\"}", "application/json", false) == null);
    try std.testing.expect(lanShareDenial(&l, reg, "POST", "/v1/messages", "{\"model\":\"x@peer\"}", "application/json", true) != null);

    // `/v1/images/edits` is model_gated but its body is multipart, so a
    // JSON-only scan reads NO id and every edit silently gets default-model
    // semantics — the gate would then disagree with what dispatch runs, which
    // is the one thing this function promises it never does.
    const mp_ct = "multipart/form-data; boundary=B";
    const mp_unshared = "--B\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nqwen3.6-27b\r\n--B--\r\n";
    const mp_shared = "--B\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ngemma-4-e4b-it-4bit\r\n--B--\r\n";
    try std.testing.expect(lanShareDenial(&l, reg, "POST", "/v1/images/edits", mp_unshared, mp_ct, false) != null);
    try std.testing.expect(lanShareDenial(&l, reg, "POST", "/v1/images/edits", mp_shared, mp_ct, false) == null);
}

test "isTunneledRequest keys on the lan.tunnel marker header, case-insensitive" {
    try std.testing.expect(isTunneledRequest("X-MLX-LAN: 1\r\n"));
    try std.testing.expect(isTunneledRequest("x-mlx-lan: 1\r\n"));
    try std.testing.expect(!isTunneledRequest("Content-Type: application/json\r\n"));
    try std.testing.expect(!isTunneledRequest(""));
}

test "proxiedBody: a Windows path id is spliced back escaped, so the forwarded body stays JSON" {
    const a = std.testing.allocator;
    const body = "{\"model\":\"C:\\\\m@local\"}";
    const raw = body[10 .. body.len - 2];
    const out = try proxiedBody(a, body, raw, "C:\\m");
    defer a.free(out);
    try std.testing.expectEqualStrings("{\"model\":\"C:\\\\m\"}", out);
}
