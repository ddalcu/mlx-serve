//! ROUTE_PATHS, routeExists, request size limits; handleConnection (route dispatch); isStatusRoute, isTextGenRoute

const request_options = @import("../api/request_options.zig");
const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const media_mod = @import("../media_jobs/gen.zig");
const instr = @import("../metrics/metrics.zig");
const lan_mod = @import("../lan/lan.zig");
const multipart = @import("multipart.zig");
const ws_mod = @import("ws.zig");
const http_server = @import("server.zig");
const request_model = @import("../api/request_model.zig");

const Conn = @import("conn.zig").Conn;
const resolveRequestModelId = @import("../api/request_model.zig").resolveRequestModelId;
const queryModel = @import("../api/request_model.zig").queryModel;
const parseModelFromRequest = @import("../api/request_model.zig").parseModelFromRequest;
const MAX_JSON_DEPTH = @import("request.zig").MAX_JSON_DEPTH;
const jsonNestingDepth = @import("request.zig").jsonNestingDepth;
const readRequest = @import("request.zig").readRequest;
const sendOllamaError = @import("../api/ollama_routes.zig").sendOllamaError;
const handleOllamaEarly = @import("../api/ollama_routes.zig").handleOllamaEarly;
const handleOllamaChat = @import("../api/ollama_routes.zig").handleOllamaChat;
const handleOllamaGenerate = @import("../api/ollama_routes.zig").handleOllamaGenerate;
const handleOllamaEmbed = @import("../api/ollama_routes.zig").handleOllamaEmbed;
const admission = @import("../api/admission.zig");
const ModelRegistry = @import("../registry/registry.zig").ModelRegistry;
const TextGenTarget = @import("../api/media_gate.zig").TextGenTarget;
const textGenRejectReason = @import("../api/media_gate.zig").textGenRejectReason;
const textGenTargetOf = @import("../api/media_gate.zig").textGenTargetOf;
const handleModels = @import("../api/models.zig").handleModels;
const handleLaunchScript = @import("../launch/http.zig").handleLaunchScript;
const renderLoadProgressBody = @import("../api/models.zig").renderLoadProgressBody;
const handleModelsRescan = @import("../api/models.zig").handleModelsRescan;
const handleLoadModelStrict = @import("../api/models.zig").handleLoadModelStrict;
const handleTranscriptions = @import("../api/audio.zig").handleTranscriptions;
const handleSpeech = @import("../api/audio.zig").handleSpeech;
const handleGen = @import("../media_jobs/http.zig").handleGen;
const handleUnloadModelStrict = @import("../api/models.zig").handleUnloadModelStrict;
const handleProps = @import("../api/props.zig").handleProps;
const handlePropsNoModel = @import("../api/props.zig").handlePropsNoModel;
const handleStatusPage = @import("../api/props.zig").handleStatusPage;
const handleEmbeddings = @import("../api/embeddings.zig").handleEmbeddings;
const handleTokenize = @import("../api/tokenize.zig").handleTokenize;
const handleDetokenize = @import("../api/tokenize.zig").handleDetokenize;
const handleChatCompletions = @import("../api/chat_completions.zig").handleChatCompletions;
const handleCompletions = @import("../api/completions.zig").handleCompletions;
const sendResponse = @import("respond.zig").sendResponse;
const sendResponseQuiet = @import("respond.zig").sendResponseQuiet;
const logHttpRequest = @import("respond.zig").logHttpRequest;
const logSafePath = @import("respond.zig").logSafePath;
const logSafeBody = @import("respond.zig").logSafeBody;
const findHeaderValue = @import("request.zig").findHeaderValue;
const apiKeyGateApplies = @import("auth.zig").apiKeyGateApplies;
const peerIsLoopback = @import("auth.zig").peerIsLoopback;
const lanGateApplies = @import("../lan/http.zig").lanGateApplies;
const isTunneledRequest = @import("../lan/http.zig").isTunneledRequest;
const lanShareDenial = @import("../lan/http.zig").lanShareDenial;
const handleLanProxy = @import("../lan/http.zig").handleLanProxy;
const handleProviderProxy = @import("../lan/http.zig").handleProviderProxy;
const handleProvidersReload = @import("../lan/http.zig").handleProvidersReload;
const queryParamValue = @import("auth.zig").queryParamValue;
const apiKeyAuthorized = @import("auth.zig").apiKeyAuthorized;
const encodedKeyMatches = @import("auth.zig").encodedKeyMatches;
const cookieSafe = @import("auth.zig").cookieSafe;
const loginFormKey = @import("auth.zig").loginFormKey;
const sendKeyCookie = @import("auth.zig").sendKeyCookie;
const wantsLoginPage = @import("auth.zig").wantsLoginPage;
const loginPage = @import("auth.zig").loginPage;
const loginPageHtml = @import("auth.zig").loginPageHtml;
const sendUnauthorized = @import("auth.zig").sendUnauthorized;
const sendEnsureLoadedError = @import("errors.zig").sendEnsureLoadedError;
const sendErrorResponse = @import("errors.zig").sendErrorResponse;
const sendAnthropicError = @import("../api/anthropic.zig").sendAnthropicError;
const handleAnthropicMessages = @import("../api/anthropic.zig").handleAnthropicMessages;
const handleResponsesGet = @import("../api/responses/handler.zig").handleResponsesGet;
const handleResponsesDelete = @import("../api/responses/handler.zig").handleResponsesDelete;
const handleResponsesCompact = @import("../api/responses/handler.zig").handleResponsesCompact;
const handleResponses = @import("../api/responses/handler.zig").handleResponses;
const handleResponsesWebSocket = @import("../api/responses/ws.zig").handleResponsesWebSocket;

/// Every path `handleConnection` dispatches. Kept beside the chain rather than
/// derived from it because one question has to be answerable BEFORE a model is
/// resolved: does this endpoint exist at all?
///
/// Resolution runs ahead of dispatch, so without this an unknown path on a
/// server with no default model returned 503 "No default model configured"
/// instead of 404 — and a path's existence has nothing to do with model state.
/// Every dispatch compare goes through `isRoute`, which refuses to compile a path missing here.
const ROUTE_PATHS = [_][]const u8{
    "/",
    "/api/chat",
    "/api/embed",
    "/api/embeddings",
    "/api/generate",
    "/api/ps",
    "/api/pull",
    "/api/show",
    "/api/tags",
    "/api/version",
    "/detokenize",
    "/health",
    "/launch",
    "/metrics",
    "/metrics.json",
    "/props",
    "/tokenize",
    "/v1/3d/generations",
    "/v1/audio/music-generations",
    "/v1/audio/sound-generations",
    "/v1/audio/speech",
    "/v1/audio/transcriptions",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/decisions",
    "/v1/systemone",
    "/v1/embeddings",
    "/v1/images/edits",
    "/v1/images/generations",
    "/v1/load-model",
    "/v1/load-progress",
    "/v1/messages",
    "/v1/models",
    "/v1/models/rescan",
    "/v1/providers",
    "/v1/providers/reload",
    "/v1/responses",
    "/v1/responses/compact",
    "/v1/unload-model",
    "/v1/video/generations",
};

/// Is `path` an endpoint this server serves at all (any method)?
fn routeExists(path: []const u8) bool {
    // `/v1/responses/{id}` (GET/DELETE) is matched by prefix, not by a literal.
    if (std.mem.startsWith(u8, path, "/v1/responses/")) return true;
    for (ROUTE_PATHS) |p| if (std.mem.eql(u8, path, p)) return true;
    return false;
}

/// `path == p` for a dispatched literal `p`; a `p` missing from ROUTE_PATHS does not compile.
pub fn isRoute(path: []const u8, comptime p: []const u8) bool {
    comptime {
        @setEvalBranchQuota(20_000);
        if (!routeExists(p)) @compileError("dispatched path missing from ROUTE_PATHS: " ++ p);
    }
    return std.mem.eql(u8, path, p);
}

pub const max_media_request_bytes: usize = 512 * 1024 * 1024;

/// A coding agent keeps its screenshots in the history, as base64: chat takes what media takes.
pub const max_request_bytes: usize = max_media_request_bytes;

/// A decision prompt keeps at most `max_len` state tokens, so a larger body only
/// buys tokenizer work on the inference thread.
pub const max_decision_request_bytes: usize = 4 * 1024 * 1024;

/// Per-route request-body cap. Media bodies are base64 payloads — a single
/// ref2va reference video is ~100 MB of JPEG frames, three plus full-res
/// reference images approach 500 MB (issue #151) — and a vision agent's chat
/// history carries its screenshots the same way. Only a decision prompt is small.
pub fn maxRequestBytesFor(target: []const u8) usize {
    const path = target[0 .. std.mem.indexOfScalar(u8, target, '?') orelse target.len];
    for ([_][]const u8{ "/v1/images/", "/v1/video/", "/v1/audio/", "/v1/3d/" }) |p|
        if (std.mem.startsWith(u8, path, p)) return max_media_request_bytes;
    if (isRoute(path, "/v1/decisions") or isRoute(path, "/v1/systemone")) return max_decision_request_bytes;
    return max_request_bytes;
}

/// The 413 names BOTH numbers it compared (context-overflow-400 class): a
/// refusal that only names the cap reads as "this should have worked".
pub fn payloadTooLargeMessage(buf: []u8, got: usize, cap: usize) []const u8 {
    const mb = 1024 * 1024;
    return std.fmt.bufPrint(buf, "Request body too large: {d} MB exceeds this endpoint's {d} MB limit", .{
        got / mb + @intFromBool(got % mb != 0), cap / mb,
    }) catch "Request body too large";
}

pub fn handleConnection(
    allocator: std.mem.Allocator,
    stream: *Conn,
) !void {
    // Plan 05: resolve which model this request targets. The registry was
    // set up in `serve()`; per-POST routing happens after we read the body
    // and parse out the optional `"model"` field. For Phase C we use the
    // default model for everything (only one is loaded), but the plumbing
    // is in place for Phase D's hot-load path.
    const registry = http_server.global_registry orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Server not ready", 503);
        return;
    };
    const raw = (try readRequest(allocator, stream)) orelse return;
    defer allocator.free(raw.buf);
    const total_read = raw.len;
    const header_end_pos = raw.header_end;
    const request = raw.buf[0..total_read];
    const first_line_end = std.mem.indexOf(u8, request, "\r\n") orelse return;
    const first_line = request[0..first_line_end];

    var line_iter = std.mem.splitScalar(u8, first_line, ' ');
    const method = line_iter.next() orelse return;
    const raw_path = line_iter.next() orelse return;
    // Strip query string for route matching (e.g. /v1/messages?beta=true -> /v1/messages)
    const path = if (std.mem.indexOf(u8, raw_path, "?")) |qpos| raw_path[0..qpos] else raw_path;
    const request_body = if (total_read > header_end_pos) request[header_end_pos..total_read] else "";
    // Needed before model resolution, not just at the handler: `/v1/images/edits`
    // carries its model in a multipart FIELD, which only the content-type's
    // boundary lets us find (`parseModelFromRequest`).
    const request_content_type = findHeaderValue(request[0..header_end_pos], "content-type") orelse "";
    logHttpRequest(method, logSafePath(raw_path), logSafeBody(method, path, request_content_type, request_body));
    if (multipart.boundaryFromContentType(request_content_type) == null and jsonNestingDepth(request_body) > MAX_JSON_DEPTH) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request JSON nests deeper than 4096 levels", 400);
        return;
    }
    stream.noteRequestHeaders(request[0..header_end_pos]);

    // ── API-key auth gate. When --api-key is set, every NON-LOOPBACK request
    //    requires the key (the OpenAI/Anthropic/Ollama APIs AND the index page
    //    + metrics panel/feed) — EXCEPT `/health` and CORS preflight, which
    //    load balancers and browsers must reach unauthenticated. Loopback is
    //    trusted (the local app connects via 127.0.0.1), so it's exempt: the app
    //    + a local browser never need credentials, and the key protects network
    //    exposure. Browser-facing pages get a `WWW-Authenticate: Basic`
    //    challenge so the index + metrics panel prompt for the key; API clients
    //    pass Bearer / x-api-key. null key ⇒ fully open. See `apiKeyAuthorized`.
    if (apiKeyGateApplies(http_server.g_api_key != null, http_server.g_api_key_strict, peerIsLoopback(stream)) and
        !std.mem.eql(u8, method, "OPTIONS") and
        !(std.mem.eql(u8, method, "GET") and isRoute(path, "/health")))
    {
        const raw_headers = request[0..header_end_pos];
        if (loginFormKey(method, path, request_content_type, request_body)) |k| {
            if (encodedKeyMatches(k, http_server.g_api_key.?)) return sendKeyCookie(stream, k);
            return sendResponse(stream, "401 Unauthorized", "text/html; charset=utf-8", loginPageHtml("<p class=err>That key was not accepted. Try again.</p>"));
        }
        if (!apiKeyAuthorized(raw_headers, raw_path)) {
            log.debug("{s} {s} -> 401 (missing/invalid API key)\n", .{ method, path });
            if (wantsLoginPage(method, path, raw_headers))
                try sendResponse(stream, "401 Unauthorized", "text/html; charset=utf-8", loginPage(raw_headers, raw_path))
            else
                try sendUnauthorized(stream);
            return;
        }
        // A page opened with `?api_key=` trades it for the cookie, so the key leaves the address bar.
        if (wantsLoginPage(method, path, raw_headers)) {
            if (queryParamValue(raw_path, "api_key") orelse queryParamValue(raw_path, "key")) |k| {
                if (cookieSafe(k) and encodedKeyMatches(k, http_server.g_api_key.?)) return sendKeyCookie(stream, k);
            }
        }
    }

    // ── LAN sharing gate. With sharing ON and no --api-key set, a non-loopback
    //    client gets exactly the shared inference surface: allowlisted routes
    //    (lan.routeClass) on shared models only. With a key set, unauthorized
    //    non-loopback requests already died above and key-holders keep full
    //    access — so this gate only exists in keyless mode.
    if (lanGateApplies(stream)) {
        if (lanShareDenial(http_server.g_lan.?, registry, method, path, request_body, request_content_type, isTunneledRequest(request[0..header_end_pos]))) |denial| {
            log.debug("{s} {s} -> 403 (lan: {s})\n", .{ method, path, denial });
            try sendErrorResponse(allocator, stream, "403 Forbidden", "forbidden", denial, 403);
            return;
        }
    }

    // ── Plan 05: routes that don't depend on a loaded model (connectivity
    //    probes + CORS preflight + listing endpoints). Handle these BEFORE
    //    `scheduler.ensureLoaded` so they don't trigger a cold load of the
    //    default model just to read metadata. `/v1/models` and the GET-side
    //    of the Responses API are pure registry/store reads — no model
    //    needed.
    if (std.mem.eql(u8, method, "HEAD") and isRoute(path, "/")) {
        log.debug("HEAD / -> 200\n", .{});
        try sendResponse(stream, "200 OK", "text/plain", "");
        return;
    }
    if (std.mem.eql(u8, method, "GET") and isRoute(path, "/health")) {
        if (mlx.gpuContextLost()) return sendResponse(stream, "503 Service Unavailable", "application/json", "{\"status\":\"cuda context lost\"}");
        log.debug("GET  /health -> 200\n", .{});
        try sendResponse(stream, "200 OK", "application/json", "{\"status\":\"ok\"}");
        return;
    }
    // The console. It belongs here, above resolution, for the same reason
    // /v1/models does: it IS the model picker, so it has to render before
    // anything is loaded. Dispatched after resolution it rendered one
    // *LoadedModel and a headless boot (`mlx-serve serve`, and every
    // app-launched server) answered 503 "No default model configured" at the
    // root — the first page a person opens. Everything it renders from
    // the model it now fetches from /v1/models + /props client-side.
    if (std.mem.eql(u8, method, "GET") and isRoute(path, "/")) {
        log.debug("GET  / -> 200 (console)\n", .{});
        try handleStatusPage(stream);
        return;
    }
    // Prometheus scrape endpoint. 503 when --metrics is off. Behind the global
    // API-key gate above when --api-key is set (auth already enforced here).
    if (std.mem.eql(u8, method, "GET") and isRoute(path, "/metrics")) {
        if (http_server.g_metrics) |m| {
            var out: std.Io.Writer.Allocating = .init(allocator);
            defer out.deinit();
            try instr.renderPrometheus(m, &out.writer);
            // Quiet: don't dump the ~3 KB scrape body every 2 s at debug level.
            try sendResponseQuiet(stream, "200 OK", "text/plain; version=0.0.4; charset=utf-8", out.written());
        } else {
            try sendResponse(stream, "503 Service Unavailable", "text/plain", "metrics not enabled (start with --metrics)");
        }
        return;
    }
    // JSON feed — drives the console's Monitoring pane. Behind the
    // global API-key gate above when --api-key is set (same-origin browser
    // fetch inherits the page's Basic credentials).
    if (std.mem.eql(u8, method, "GET") and isRoute(path, "/metrics.json")) {
        if (http_server.g_metrics) |m| {
            var out: std.Io.Writer.Allocating = .init(allocator);
            defer out.deinit();
            var sessions: [2 * instr.MAX_SESSIONS]instr.Session = undefined;
            try instr.renderJson(m, liveSessions(registry, &sessions), &out.writer);
            // Quiet: the index panel polls this ~1 Hz — don't log the body.
            try sendResponseQuiet(stream, "200 OK", "application/json", out.written());
        } else {
            try sendResponse(stream, "503 Service Unavailable", "application/json", "{\"error\":\"metrics not enabled — start with --metrics\"}");
        }
        return;
    }
    if (std.mem.eql(u8, method, "OPTIONS")) {
        log.debug("OPTIONS {s} -> 204\n", .{path});
        try sendResponse(stream, "204 No Content", "text/plain", "");
        return;
    }
    if (std.mem.eql(u8, method, "GET") and isRoute(path, "/v1/models")) {
        log.debug("GET  /v1/models -> 200\n", .{});
        // A keyless LAN peer sees only shared models and never the remote
        // stubs (a mirrored entry would invite multi-hop loops). Peer
        // discovery fetches self-identify with the X-MLX-LAN marker and get
        // the SAME filtered view even over loopback — two servers on one Mac
        // resolve each other loopback-first, and the unfiltered list leaked
        // remote stubs into `@a@b` re-export chains.
        try handleModels(allocator, stream, lanGateApplies(stream) or
            (http_server.g_lan != null and isTunneledRequest(request[0..header_end_pos])));
        return;
    }
    if (std.mem.eql(u8, method, "GET") and isRoute(path, "/launch")) {
        try handleLaunchScript(allocator, stream, raw_path, request[0..header_end_pos], lanGateApplies(stream));
        return;
    }
    if (std.mem.eql(u8, method, "GET") and std.mem.startsWith(u8, path, "/v1/responses/")) {
        const id = path["/v1/responses/".len..];
        try handleResponsesGet(allocator, stream, id);
        return;
    }
    if (std.mem.eql(u8, method, "DELETE") and std.mem.startsWith(u8, path, "/v1/responses/")) {
        const id = path["/v1/responses/".len..];
        try handleResponsesDelete(allocator, stream, id);
        return;
    }
    if (std.mem.eql(u8, method, "POST") and isRoute(path, "/v1/responses/compact")) {
        // Compaction is a pure data transformation — no model load needed.
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleResponsesCompact(allocator, stream, body);
        return;
    }
    if (std.mem.eql(u8, method, "POST") and isRoute(path, "/v1/load-model")) {
        // Phase E: explicit cold-load. Keep strict semantics — unknown ids
        // get a 404 here even though the main dispatch path below falls
        // back to default for SDK compatibility.
        log.debug("POST /v1/load-model -> 200\n", .{});
        try handleLoadModelStrict(allocator, stream, request_body);
        return;
    }
    // Any method: a status route never reaches `ensureLoaded`, whatever its body names.
    if (isRoute(path, "/v1/load-progress")) {
        const body = renderLoadProgressBody(allocator, stream.io) catch |err| switch (err) {
            error.RegistryNotReady => return sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Registry not ready", 503),
            else => return err,
        };
        defer allocator.free(body);
        try sendResponse(stream, "200 OK", "application/json", body);
        return;
    }
    if (std.mem.eql(u8, method, "POST") and isRoute(path, "/v1/unload-model")) {
        // Free a model's resident GPU state (the app's load→generate→unload
        // default). The stub stays registered so it can reload. Idempotent.
        log.debug("POST /v1/unload-model -> 200\n", .{});
        try handleUnloadModelStrict(allocator, stream, request_body);
        return;
    }
    if (std.mem.eql(u8, method, "POST") and isRoute(path, "/v1/providers/reload")) {
        try handleProvidersReload(allocator, stream);
        return;
    }
    if (std.mem.eql(u8, method, "GET") and isRoute(path, "/v1/providers")) {
        const body = if (http_server.g_providers) |p| try p.statusJson(allocator) else try allocator.dupe(u8, "{\"providers\":[]}");
        defer allocator.free(body);
        try sendResponse(stream, "200 OK", "application/json", body);
        return;
    }
    if (std.mem.eql(u8, method, "POST") and isRoute(path, "/v1/models/rescan")) {
        // Absorb models downloaded AFTER boot (the Model Browser pulls while
        // the server runs; discovery only walks the roots at startup).
        try handleModelsRescan(allocator, stream);
        return;
    }

    // ── Ollama-compatible API (/api/*): endpoints that must not trigger a
    //    model load (version/tags/ps/show/pull/unsupported) are handled
    //    here; /api/chat, /api/generate and /api/embed(dings) fall through
    //    to the model-resolution path below. Glue lives at the bottom of
    //    this file; pure translation in src/host/api/ollama/translate.zig.
    if (std.mem.startsWith(u8, path, "/api/")) {
        if (try handleOllamaEarly(allocator, stream, method, path, request_body)) return;
    }

    // ── Plan 05 Phase D: resolve the request's target model via the
    //    scheduler (which delegates the fast path to registry.ensureLoaded
    //    and handles cold-load + eviction internally). Absent `model`
    //    field → default. Unknown id → 404. Unloaded id with no room and
    //    no LRU victim → 503 (NotEnoughMemory). Block-until-loaded is
    //    intentional; clients targeting an unloaded model should set a
    //    longer request timeout (or hit /v1/load-model first).
    const scheduler = http_server.launch().global_scheduler orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Scheduler not ready", 503);
        return;
    };
    // Strip whatever the client passed in `"model":"..."` — except when the
    // id literally matches one we've discovered, in which case we honor it
    // and route. The OpenAI / Anthropic ecosystem commonly sends marketing
    // names like "gpt-4" or "claude-opus-4-x" expecting the local server to
    // just respond with whatever it has loaded; the multi-model registry's
    // strict-id semantics are opt-in by sending an id we registered.
    // Swift's JSON writers escape '/' as '\\/' (Foundation does this
    // unconditionally, both JSONSerialization and JSONEncoder). That is legal
    // JSON, but the id is read out of the RAW body, so an HF-style `org/repo`
    // id arrived as `Runpod\\/FLUX.2-…`, matched nothing in the registry, and
    // silently fell through to the default model — a 400 "does not support
    // this media modality" on the gen endpoints, and a wrong-model answer on
    // chat (live from the iPhone app). Canonicalise once, here,
    // so every consumer below (proxy, peek, ensureLoaded) sees the real id.
    var model_id_buf: [512]u8 = undefined;
    var query_model_buf: [512]u8 = undefined;
    // The proxies splice over the value as written, so they also get the raw slice.
    const body_model = parseModelFromRequest(request_body, request_content_type);
    var requested_model_id = lan_mod.unescapeJsonSlashes(
        &model_id_buf,
        body_model orelse queryModel(&query_model_buf, raw_path) orelse "",
    );
    // ── LAN-discovered remote model (`<id>@<peer>`) → proxy the request to
    //    its host byte-for-byte, model field rewritten to the bare id.
    //    Any DIRECT client may initiate the hop (loopback app, the
    //    agent-sandbox VM over its NAT interface, LAN clients); a request
    //    that itself arrived through a peer's tunnel never hops again —
    //    that marker, not loopback-ness, is the multi-hop bound. A
    //    registered LOCAL id containing '@' keeps winning via the peek; an
    //    offline peer is an honest 404, never a silent local-default answer.
    // ── Configured provider (`<id>@<provider>`) → proxied to its
    //    /v1/chat/completions with the provider's key. Checked before the LAN
    //    table: a provider is a URL the user typed, a peer is discovered.
    if (http_server.g_providers) |prov| if (lan_mod.splitRemoteId(requested_model_id)) |rid| if (prov.isProvider(rid.peer) and
        !isTunneledRequest(request[0..header_end_pos]) and registry.peek(requested_model_id) == null)
    {
        try handleProviderProxy(allocator, stream, prov, path, request_body, body_model, requested_model_id);
        return;
    };
    if (http_server.g_lan != null and lan_mod.splitRemoteId(requested_model_id) != null and
        !isTunneledRequest(request[0..header_end_pos]) and registry.peek(requested_model_id) == null)
    {
        try handleLanProxy(allocator, stream, http_server.g_lan.?, method, raw_path, request_body, body_model, requested_model_id);
        return;
    }
    // Counted from here to the response's end: a burst's requests decide their spec mode before any reaches the
    // engine, so the engine's own count misses them (`request_options.requestHasCompany`).
    const text_gen = isTextGenRoute(method, path);
    if (text_gen) _ = request_options.text_gen_live.fetchAdd(1, .monotonic);
    defer if (text_gen) {
        _ = request_options.text_gen_live.fetchSub(1, .monotonic);
    };
    switch (resolveRequestModelId(registry, &request_model.g_model_aliases, requested_model_id, path)) {
        .id => |id| requested_model_id = id,
        .unknown_path => {
            try sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "No model is registered at that path; load it first with POST /v1/load-model", 404);
            return;
        },
        // Unknown name — fall back to the default model rather than 404, so
        // off-the-shelf SDK clients keep working.
        .unknown_name => requested_model_id = "",
    }
    // Text-gen route aimed at a KNOWN non-text model: reject before
    // ensureLoaded, or the request cold-loads a multi-GB media model just
    // to earn its 400. The post-load `text_gen_reject` below stays the
    // authoritative crash barrier (this peek can't see `--model` primaries
    // with no arch hint until they're resident).
    if (requested_model_id.len > 0 and isTextGenRoute(method, path)) {
        // Entry and snapshot under ONE hold — plain `peek` unlocks before it
        // returns. `arch_hint` outlives the unlock: entry-owned, freed in
        // `deinit`.
        const target: ?TextGenTarget = blk: {
            registry.mutex.lockUncancelable(stream.io);
            defer registry.mutex.unlock(stream.io);
            const peeked = registry.peekLocked(requested_model_id) orelse break :blk null;
            break :blk textGenTargetOf(peeked);
        };
        if (target) |t| {
            if (textGenRejectReason(t)) |reason| {
                if (isRoute(path, "/v1/messages")) {
                    try sendAnthropicError(allocator, stream, "invalid_request_error", reason, 400);
                } else if (std.mem.startsWith(u8, path, "/api/")) {
                    try sendOllamaError(allocator, stream, "400 Bad Request", reason);
                } else {
                    try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
                }
                return;
            }
        }
    }
    // An endpoint that doesn't exist is a 404, and it must cost NOTHING to say
    // so. Resolution runs ahead of dispatch, so this has to be answered here
    // rather than inside one of `ensureLoaded`'s error arms: with a resolvable
    // `model` in the body, resolution SUCCEEDS — cold-loading the checkpoint —
    // and the unknown path only 404s afterwards. `POST /v1/load` (the route is
    // /v1/load-model) cost 2m42s and 121 GB resident for a typo, and the same
    // one-liner pins the box for anyone who sends it. Placed below the LAN
    // proxy so a `<id>@<peer>` hop is unchanged, and above every load.
    if (!routeExists(path)) {
        try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "Unknown endpoint", 404);
        return;
    }
    // `GET /props` is a STATUS read, and falling into `ensureLoaded` makes a
    // poll cold-load the model. With idle eviction on that is a loop: the sweep
    // hands the memory back and the next poll takes it straight again (the app's
    // tray polls every 3s). Nothing resident means nothing to report — the same
    // body the `NoDefaultModel` arm below already sends.
    if (std.mem.eql(u8, method, "GET") and isRoute(path, "/props")) {
        const resident = if (registry.resolveEntry(requested_model_id)) |e| blk: {
            registry.mutex.lockUncancelable(stream.io);
            defer registry.mutex.unlock(stream.io);
            break :blk e.state == .ready;
        } else |_| false;
        if (!resident) {
            try handlePropsNoModel(allocator, stream);
            return;
        }
    }
    const lm = scheduler.ensureLoaded(requested_model_id) catch |err| {
        // Headless gen-only boot (media models resident, no default chat
        // model): /props keeps answering with the live memory counters —
        // the app's tray polls it, and a 503 here read as "0 MB".
        if (err == error.NoDefaultModel and std.mem.eql(u8, method, "GET") and isRoute(path, "/props")) {
            try handlePropsNoModel(allocator, stream);
            return;
        }
        try sendEnsureLoadedError(allocator, stream, scheduler, requested_model_id, err);
        return;
    };
    // A status read must not look like use — see `isStatusRoute`.
    const status_read = isStatusRoute(method, path);
    defer if (status_read) scheduler.releaseStatus(lm) else scheduler.release(lm);

    // A model loaded on demand freezes its own auto-context here, for the same
    // reason the `--model` primary does at startup: the number we advertise is
    // what clients budget against, and each model has its own dims. Idempotent,
    // and a no-op under `--ctx-size`.
    if (lm.config) |c| _ = admission.pinAutoContext(c, false);

    // ── Phase C: handlers take `lm` directly and extract their own
    //    locals. handleConnection only needs one decision for every
    //    text-generation route: encoder-only AND media (image/audio/video/
    //    3D) models get a clean 400 instead of reaching a prefill that has
    //    no transformer to run (the SIGSEGV class documented on
    //    textGenRejectReason).
    const text_gen_reject: ?[]const u8 = textGenRejectReason(textGenTargetOf(lm));

    // ── WebSocket upgrade for /v1/responses ──
    // Detect Upgrade: websocket BEFORE the regular route dispatch so the
    // GET method (used for the upgrade handshake) doesn't fall through to
    // the 404 branch.
    if (std.mem.eql(u8, method, "GET") and isRoute(path, "/v1/responses") and ws_mod.isUpgrade(request[0..header_end_pos])) {
        if (text_gen_reject) |reason| {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
            return;
        }
        try handleResponsesWebSocket(allocator, stream, request[0..header_end_pos], lm);
        return;
    }

    if (std.mem.eql(u8, method, "GET") and isRoute(path, "/props")) {
        log.debug("GET  /props -> 200\n", .{});
        try handleProps(allocator, stream, lm);
    } else if (std.mem.eql(u8, method, "POST") and isRoute(path, "/v1/chat/completions")) {
        if (text_gen_reject) |reason| {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
            return;
        }
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleChatCompletions(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and isRoute(path, "/v1/completions")) {
        if (text_gen_reject) |reason| {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
            return;
        }
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleCompletions(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and isRoute(path, "/v1/embeddings")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleEmbeddings(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and isRoute(path, "/v1/messages")) {
        if (text_gen_reject) |reason| {
            try sendAnthropicError(allocator, stream, "invalid_request_error", reason, 400);
            return;
        }
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleAnthropicMessages(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and isRoute(path, "/v1/responses")) {
        if (text_gen_reject) |reason| {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
            return;
        }
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleResponses(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and isRoute(path, "/tokenize")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleTokenize(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and isRoute(path, "/detokenize")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleDetokenize(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and isRoute(path, "/v1/images/generations")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleGen(allocator, stream, body, lm, .image);
    } else if (std.mem.eql(u8, method, "POST") and isRoute(path, "/v1/images/edits")) {
        // OpenAI's image-EDIT surface: multipart/form-data, not JSON. Translated
        // into the `/v1/images/generations` edit body (gen.openaiEditFormToJson)
        // and served by the identical path — no second inference route.
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const ct = findHeaderValue(request[0..header_end], "content-type") orelse "";
        const body = request[header_end + 4 .. total_read];
        const json = media_mod.openaiEditFormToJson(allocator, body, ct) catch |err| {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", media_mod.editFormErrorMessage(err), 400);
            return;
        };
        defer allocator.free(json);
        try handleGen(allocator, stream, json, lm, .image);
    } else if (std.mem.eql(u8, method, "POST") and isRoute(path, "/v1/audio/transcriptions")) {
        if (text_gen_reject) |reason| {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
            return;
        }
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const ct = findHeaderValue(request[0..header_end], "content-type") orelse "";
        try handleTranscriptions(allocator, stream, request[header_end + 4 .. total_read], ct, lm);
    } else if (std.mem.eql(u8, method, "POST") and isRoute(path, "/v1/audio/speech")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        // A language model that speaks (LFM3-Audio) answers here itself; media TTS backends go through the gen path.
        const speaks = if (lm.config) |c| c.speech() != null else false;
        if (speaks) try handleSpeech(allocator, stream, body, lm) else try handleGen(allocator, stream, body, lm, .speech);
    } else if (std.mem.eql(u8, method, "POST") and (isRoute(path, "/v1/decisions") or isRoute(path, "/v1/systemone"))) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleGen(allocator, stream, body, lm, .decisions);
    } else if (std.mem.eql(u8, method, "POST") and isRoute(path, "/v1/audio/music-generations")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleGen(allocator, stream, body, lm, .music);
    } else if (std.mem.eql(u8, method, "POST") and isRoute(path, "/v1/audio/sound-generations")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleGen(allocator, stream, body, lm, .sound);
    } else if (std.mem.eql(u8, method, "POST") and isRoute(path, "/v1/video/generations")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleGen(allocator, stream, body, lm, .video);
    } else if (std.mem.eql(u8, method, "POST") and isRoute(path, "/v1/3d/generations")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleGen(allocator, stream, body, lm, .mesh);
    } else if (std.mem.eql(u8, method, "POST") and isRoute(path, "/api/chat")) {
        if (text_gen_reject) |reason| {
            try sendOllamaError(allocator, stream, "400 Bad Request", reason);
            return;
        }
        try handleOllamaChat(allocator, stream, request_body, lm);
    } else if (std.mem.eql(u8, method, "POST") and isRoute(path, "/api/generate")) {
        if (text_gen_reject) |reason| {
            try sendOllamaError(allocator, stream, "400 Bad Request", reason);
            return;
        }
        try handleOllamaGenerate(allocator, stream, request_body, lm);
    } else if (std.mem.eql(u8, method, "POST") and (isRoute(path, "/api/embed") or isRoute(path, "/api/embeddings"))) {
        try handleOllamaEmbed(allocator, stream, request_body, lm, isRoute(path, "/api/embeddings"));
    } else {
        log.warn("{s} {s} -> 404\n", .{ method, path });
        try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "The requested endpoint does not exist", null);
    }
}

// ── Ollama-compatible API (/api/*) glue ─────────────────────────────────

// Pure translation (request shapes, SSE→NDJSON sink, renderers) lives in

// src/host/api/ollama/translate.zig; this section owns routing targets, registry access, and

// the Conn sink hook. The inner /v1 handlers are reused verbatim — an

// /api/chat request becomes a /v1/chat/completions body whose SSE output

// the Sink re-frames into Ollama NDJSON on the real socket.

/// True for a read-only route that reports on a model without using it.
/// `GET /props` is the only one: every other status endpoint either answers
/// before `ensureLoaded` (the Ollama block returns from `handleOllamaEarly`,
/// and `/health`, `/metrics`, `/v1/models` return above it) or is a real load
/// (`/v1/load-model`). `/props` alone falls through to the shared
/// `ensureLoaded`/release pair, so without this a poll stamps `last_used_ms`
/// and a client polling faster than `--idle-evict-secs` pins every model.
fn isStatusRoute(method: []const u8, path: []const u8) bool {
    return std.mem.eql(u8, method, "GET") and isRoute(path, "/props");
}

/// True for the routes textGenRejectReason protects — used for the
/// pre-load peek so naming a media model in a chat request doesn't
/// cold-load gigabytes just to earn its 400.
fn isTextGenRoute(method: []const u8, path: []const u8) bool {
    if (std.mem.eql(u8, method, "POST")) {
        const routes = [_][]const u8{
            "/v1/chat/completions", "/v1/completions", "/v1/messages",
            "/v1/responses",        "/api/chat",       "/api/generate",
            "/v1/audio/transcriptions",
        };
        for (routes) |r| if (std.mem.eql(u8, path, r)) return true;
        return false;
    }
    // WebSocket upgrade handshake for /v1/responses.
    return std.mem.eql(u8, method, "GET") and isRoute(path, "/v1/responses");
}

/// Live requests, then every hot-cache entry no live request restored from, each row stamped
/// with its model's context limit.
fn liveSessions(registry: *ModelRegistry, buf: *[2 * instr.MAX_SESSIONS]instr.Session) []instr.Session {
    const sch = http_server.launch().global_scheduler orelse return buf[0..0];
    sch.queue_mu.lockUncancelable(sch.io);
    const live = sch.live_session_count;
    @memcpy(buf[0..live], sch.live_sessions[0..live]);
    sch.queue_mu.unlock(sch.io);

    var n = live;
    sch.digest_mu.lockUncancelable(sch.io);
    for (sch.cached_sessions[0..sch.cached_session_count]) |c| {
        const in_use = for (buf[0..live]) |l| {
            if (l.entry_id == c.entry_id and std.mem.eql(u8, l.model(), c.model())) break true;
        } else false;
        if (in_use) continue;
        buf[n] = c;
        n += 1;
    }
    sch.digest_mu.unlock(sch.io);

    registry.mutex.lockUncancelable(registry.io);
    defer registry.mutex.unlock(registry.io);
    for (buf[0..n]) |*s| {
        const entry = registry.peekLocked(s.model()) orelse continue;
        if (entry.state != .ready) continue;
        if (entry.config) |cfg| s.context_length = admission.getEffectiveContextLength(cfg);
    }
    return buf[0..n];
}

// ── tests ──

test "routeExists answers endpoint existence without consulting the model" {
    // Whether a path EXISTS has nothing to do with which models are loaded, but
    // model resolution runs ahead of dispatch — so on a headless boot (`--serve
    // --model-dir`, no `--model`, how the app always launches) an unknown path
    // never reached the 404 branch and came back 503 "No default model
    // configured". Any client that maps endpoints by probing them then concludes
    // the server implements EVERYTHING (llmprobe: "server answers unknown paths
    // with HTTP 503" → every surface scored absent, live).
    try std.testing.expect(routeExists("/v1/chat/completions"));
    try std.testing.expect(routeExists("/v1/images/edits"));
    try std.testing.expect(routeExists("/v1/embeddings"));
    try std.testing.expect(routeExists("/health"));
    // `/v1/responses/{id}` is served by prefix, not by an exact literal.
    try std.testing.expect(routeExists("/v1/responses/resp_abc123"));
    try std.testing.expect(routeExists("/v1/responses/compact"));

    try std.testing.expect(!routeExists("/v1/__llmprobe_no_such_endpoint__"));
    try std.testing.expect(!routeExists("/v1/audio/translations")); // real OpenAI route we don't serve
    try std.testing.expect(!routeExists("/nope"));
    try std.testing.expect(!routeExists(""));
}

test "the index page documents every endpoint the server serves (drift guard)" {
    // The API reference is static data in app-web (`apiReference`); a route
    // added here and not there is missing from the built page. The finer check
    // (exact rows, both directions) is app-web's own test, which CI does not run.
    const page = @embedFile("../../html/index.html");
    for (ROUTE_PATHS) |p| {
        // "/" is the page itself — trivially present and not worth documenting
        // as an endpoint row.
        if (std.mem.eql(u8, p, "/")) continue;
        if (std.mem.indexOf(u8, page, p) == null) {
            std.debug.print("endpoint missing from the index page: {s}\n", .{p});
            return error.EndpointNotDocumented;
        }
    }
}

test "the mlx-serve agent skill documents every /v1 endpoint it does not deliberately leave out" {
    // The skill is what an agent reads before writing client code; an endpoint
    // missing there does not exist for it.
    const left_out = [_][]const u8{ "/v1/completions", "/v1/models/rescan", "/v1/providers", "/v1/providers/reload", "/v1/responses/compact" };
    const skills = @import("agent_skills");
    outer: for (ROUTE_PATHS) |p| {
        if (!std.mem.startsWith(u8, p, "/v1/")) continue;
        for (left_out) |l| if (std.mem.eql(u8, p, l)) continue :outer;
        for (skills.files) |f| if (std.mem.indexOf(u8, f.bytes, p) != null) continue :outer;
        std.debug.print("endpoint missing from skills/mlx-serve: {s}\n", .{p});
        return error.EndpointNotInSkill;
    }
}

test "isTextGenRoute covers exactly the guarded surfaces" {
    try std.testing.expect(isTextGenRoute("POST", "/v1/chat/completions"));
    try std.testing.expect(isTextGenRoute("POST", "/v1/completions"));
    try std.testing.expect(isTextGenRoute("POST", "/v1/messages"));
    try std.testing.expect(isTextGenRoute("POST", "/v1/responses"));
    try std.testing.expect(isTextGenRoute("POST", "/api/chat"));
    try std.testing.expect(isTextGenRoute("POST", "/api/generate"));
    try std.testing.expect(isTextGenRoute("GET", "/v1/responses")); // WS upgrade
    // Transcription decodes text on the LM trunk: a media model is refused before it loads.
    try std.testing.expect(isTextGenRoute("POST", "/v1/audio/transcriptions"));
    // Media + embedding routes must NOT be gated — they serve these models.
    try std.testing.expect(!isTextGenRoute("POST", "/v1/images/generations"));
    try std.testing.expect(!isTextGenRoute("POST", "/v1/audio/speech"));
    try std.testing.expect(!isTextGenRoute("POST", "/v1/embeddings"));
    try std.testing.expect(!isTextGenRoute("POST", "/api/embed"));
    try std.testing.expect(!isTextGenRoute("GET", "/v1/models"));
}

test "isStatusRoute: only /props reaches the shared release" {
    try std.testing.expect(isStatusRoute("GET", "/props"));
    // Generation is use.
    try std.testing.expect(!isStatusRoute("POST", "/v1/chat/completions"));
    try std.testing.expect(!isStatusRoute("POST", "/v1/completions"));
    // An explicit load is use — it is the whole point of the request.
    try std.testing.expect(!isStatusRoute("POST", "/v1/load-model"));
    // The Ollama status routes never reach the release: `handleOllamaEarly`
    // answers and returns above `ensureLoaded`, so listing them here would be
    // dead code that reads as coverage. Measured — polling each of these
    // against a resident model with a 5s window evicts on schedule.
    try std.testing.expect(!isStatusRoute("GET", "/api/ps"));
    try std.testing.expect(!isStatusRoute("GET", "/api/tags"));
    try std.testing.expect(!isStatusRoute("GET", "/api/version"));
    try std.testing.expect(!isStatusRoute("POST", "/api/show"));
    // Method matters: the path alone is not the predicate.
    try std.testing.expect(!isStatusRoute("POST", "/props"));
}

test "request body cap is per route: media bodies are base64 frame payloads" {
    // Issue #151: ONE ref2va reference video is ~100 MB of base64 JPEG frames,
    // three of them plus full-res ref images can approach 500 MB.
    for ([_][]const u8{
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1/video/generations",
        "/v1/audio/speech",
        "/v1/audio/music-generations",
        "/v1/3d/generations",
    }) |p| try std.testing.expectEqual(max_media_request_bytes, maxRequestBytesFor(p));
    for ([_][]const u8{ "/v1/chat/completions", "/v1/messages", "/api/chat", "/", "" }) |p|
        try std.testing.expectEqual(max_request_bytes, maxRequestBytesFor(p));
    try std.testing.expectEqual(max_decision_request_bytes, maxRequestBytesFor("/v1/decisions"));
    try std.testing.expectEqual(max_decision_request_bytes, maxRequestBytesFor("/v1/decisions?x=1"));
}

test "the chat body cap admits an agent session that keeps its screenshots" {
    // 100 screenshots at 4 MiB of base64 each.
    const screenshots = 100;
    const base64_png = 4 * 1024 * 1024;
    for ([_][]const u8{ "/v1/chat/completions", "/v1/messages", "/v1/responses" }) |p|
        try std.testing.expect(maxRequestBytesFor(p) >= screenshots * base64_png);
}

test "the 413 names both counts it compared" {
    var buf: [128]u8 = undefined;
    const msg = payloadTooLargeMessage(&buf, 97 * 1024 * 1024 + 1, 64 * 1024 * 1024);
    try std.testing.expect(std.mem.indexOf(u8, msg, "98 MB") != null);
    try std.testing.expect(std.mem.indexOf(u8, msg, "64 MB") != null);
}
