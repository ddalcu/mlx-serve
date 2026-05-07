const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const tokenizer_mod = @import("tokenizer.zig");
const generate_mod = @import("generate.zig");
const drafter_mod = @import("drafter.zig");
const chat_mod = @import("chat.zig");
const model_mod = @import("model.zig");
const vision_mod = @import("vision.zig");
const log = @import("log.zig");
const token_mask_mod = @import("token_mask.zig");
const responses_mod = @import("responses.zig");
const pld_index = @import("pld_index.zig");
const stb = @cImport({ @cInclude("stb_image.h"); });
const webp = @cImport({ @cInclude("webp/decode.h"); });
const metrics = @import("status.zig");

const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const Generator = generate_mod.Generator;
const VisionEncoder = vision_mod.VisionEncoder;
/// Global flag set by signal handler for graceful shutdown.
var shutdown_requested = std.atomic.Value(bool).init(false);

/// Single-slot inference gate: mutex + condition variable for request queuing.
/// Requests wait in line instead of getting 503 when the server is busy.
var inference_mutex: std.Io.Mutex = .init;
var inference_cond: std.Io.Condition = .init;
var inference_busy: bool = false;
var inference_queue_len: u32 = 0;
const max_queue_size: u32 = 32;

/// Acquire the inference slot, blocking until available.
/// Returns false if the queue is full or shutdown was requested.
fn acquireInferenceSlot(io: std.Io) bool {
    inference_mutex.lockUncancelable(io);
    defer inference_mutex.unlock(io);

    if (inference_queue_len >= max_queue_size) return false;
    inference_queue_len += 1;

    if (inference_busy) {
        log.info("  queued (position {d})\n", .{inference_queue_len});
    }

    while (inference_busy) {
        if (shutdown_requested.load(.acquire)) {
            inference_queue_len -= 1;
            return false;
        }
        inference_cond.waitUncancelable(io, &inference_mutex);
    }

    inference_queue_len -= 1;
    inference_busy = true;
    return true;
}

/// Release the inference slot and wake the next waiting request.
fn releaseInferenceSlot(io: std.Io) void {
    inference_mutex.lockUncancelable(io);
    defer inference_mutex.unlock(io);

    inference_busy = false;
    inference_cond.signal(io);
}

const io_util = @import("io_util.zig");
const ws_mod = @import("ws.zig");
const nowSecs = io_util.nowSecs;
const nowMs = io_util.nowMs;
const Stopwatch = io_util.Stopwatch;

/// Bridge that lets a Conn route OpenAI-Responses output through a WebSocket
/// transport instead of HTTP/SSE. When `Conn.ws_mode` is set, `sendResponse`
/// and `sendAnthropicEvent` send the JSON payload as a single WS text frame
/// instead of an HTTP body or SSE event line.
pub const WsBridge = struct {
    impl: *anyopaque,
    sendTextFn: *const fn (impl: *anyopaque, data: []const u8) anyerror!void,
    /// Captured response id from the first event seen this turn (for moving
    /// the entry between global and connection-local caches afterwards).
    /// Owned by `allocator`; freed by the WS handler at end of turn.
    captured_resp_id: ?[]u8 = null,
    allocator: ?std.mem.Allocator = null,
    /// Status emitted in the most recent `response.completed`-class event,
    /// or null if no terminal event was seen.
    captured_status: ?[]u8 = null,

    pub fn sendText(self: *WsBridge, data: []const u8) !void {
        if (self.captured_resp_id == null) {
            if (self.allocator) |a| {
                if (extractResponseId(a, data)) |id| self.captured_resp_id = id;
            }
        }
        if (self.allocator) |a| {
            if (extractCompletedStatus(a, data)) |st| {
                if (self.captured_status) |old| a.free(old);
                self.captured_status = st;
            }
        }
        try self.sendTextFn(self.impl, data);
    }

    pub fn reset(self: *WsBridge) void {
        if (self.allocator) |a| {
            if (self.captured_resp_id) |id| a.free(id);
            if (self.captured_status) |st| a.free(st);
        }
        self.captured_resp_id = null;
        self.captured_status = null;
    }
};

/// Pull the value of the first `"id":"resp_..."` field out of a JSON
/// payload. Returns an owned copy or null. Used by `WsBridge.sendText` to
/// learn the response id without parsing the whole envelope.
fn extractResponseId(allocator: std.mem.Allocator, data: []const u8) ?[]u8 {
    const needle = "\"id\":\"resp_";
    const start = std.mem.indexOf(u8, data, needle) orelse return null;
    const v_start = start + "\"id\":\"".len;
    const v_end = std.mem.indexOfScalarPos(u8, data, v_start, '"') orelse return null;
    return allocator.dupe(u8, data[v_start..v_end]) catch null;
}

/// Pull the value of `"status":"..."` (response-level, not item-level)
/// from a `response.completed`/`.failed`/`.incomplete` payload.
fn extractCompletedStatus(allocator: std.mem.Allocator, data: []const u8) ?[]u8 {
    // Only look in completed/failed/incomplete events.
    const is_terminal = std.mem.indexOf(u8, data, "\"type\":\"response.completed\"") != null or
        std.mem.indexOf(u8, data, "\"type\":\"response.failed\"") != null or
        std.mem.indexOf(u8, data, "\"type\":\"response.incomplete\"") != null;
    if (!is_terminal) return null;
    // Status field appears after the response object; first match is fine.
    const needle = "\"status\":\"";
    const start = std.mem.indexOf(u8, data, needle) orelse return null;
    const v_start = start + needle.len;
    const v_end = std.mem.indexOfScalarPos(u8, data, v_start, '"') orelse return null;
    return allocator.dupe(u8, data[v_start..v_end]) catch null;
}

fn startStopwatch(io: std.Io) Stopwatch {
    return Stopwatch.init(io);
}

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
    /// Non-null when this connection is bridged onto a WebSocket. Output that
    /// would otherwise be HTTP/SSE is reshaped into WS text frames at the
    /// `sendResponse` / `sendAnthropicEvent` chokepoints.
    ws_mode: ?*WsBridge = null,

    pub fn init(c: *Conn, stream: std.Io.net.Stream, io: std.Io) void {
        c.stream = stream;
        c.io = io;
        c.write_state = stream.writer(io, &c.write_buf);
        c.read_state = stream.reader(io, &c.read_buf);
        c.ws_mode = null;
    }

    pub fn writer(c: *Conn) *std.Io.Writer {
        return &c.write_state.interface;
    }

    pub fn reader(c: *Conn) *std.Io.Reader {
        return &c.read_state.interface;
    }

    pub fn writeAll(c: *Conn, data: []const u8) !void {
        try c.writer().writeAll(data);
        try c.writer().flush();
    }

    pub fn writeAllNoFlush(c: *Conn, data: []const u8) !void {
        try c.writer().writeAll(data);
    }

    pub fn flush(c: *Conn) !void {
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

    pub fn close(c: *Conn) void {
        c.flush() catch {};
        c.stream.close(c.io);
    }
};

fn signalHandler(_: std.posix.SIG) callconv(.c) void {
    shutdown_requested.store(true, .release);
}

/// Maximum context size (0 = unlimited). Set by --ctx-size flag.
var max_context_size: u32 = 0;

/// Request timeout in seconds (0 = no timeout). Set by --timeout flag.
var request_timeout_sec: u32 = 300;

/// Default reasoning budget in tokens (-1 = unlimited). Set by --reasoning-budget flag.
var default_reasoning_budget: i32 = -1;

/// Default MTP enabled state. Set by --mtp / --no-mtp flag at server startup.
/// Per-request `enable_mtp` JSON field overrides this default.
var default_enable_mtp: bool = false;

/// Default PLD (Prompt Lookup Decoding) enabled state. Set by --pld /
/// --no-pld at startup. Per-request `enable_pld` JSON field overrides.
/// MTP, when active, takes priority over PLD on a per-request basis (a model
/// with native MTP weights gets the higher-quality draft).
var default_enable_pld: bool = false;
var default_pld_draft_len: u32 = 5;
/// Adaptive spec-decode gate threshold. Per-request, we score the prompt's
/// 3-gram repetition density; if `score < spec_gate_threshold` AND the user
/// did not explicitly set the flag, PLD/drafter are disabled for this request.
///
/// Empirically tuned (Qwen3.5 BPE, 24-300 token prompts):
///   - "creative story" / "write an essay" prompts:                    0.000
///   - 82-token echo-heavy code rename:                                0.013
///   - 155-token JS-translation w/ many `add/sub/mul/div` repetitions: 0.128
/// The plan started from 0.15 but BPE tokenization fragments echo-heavy
/// content enough that even strong echo cases land in the 0.01–0.10 range.
/// 0.01 cleanly separates "any repetition at all" (PLD likely helps) from
/// "pure novel" (PLD overhead-only). Tune via bench_spec.sh --gated.
const spec_gate_threshold: f32 = 0.01;
var default_pld_key_len: u32 = 3;

/// Loaded Gemma 4 assistant drafter (null when `--drafter` flag wasn't supplied
/// or the drafter+target pair didn't validate). When non-null, requests with
/// `enable_drafter:true` (default-on while loaded) use it for speculative
/// decoding. The drafter outranks both MTP and PLD on a per-request basis —
/// for a Gemma 4 target paired with the assistant drafter, the drafter is the
/// preferred draft source. Non-Gemma targets cannot load the drafter (rejected
/// at startup by `bind`).
var default_drafter: ?*drafter_mod.DrafterModel = null;
var default_enable_drafter: bool = false;
var default_draft_block_size: u32 = 4;

fn getTimeoutNs() u64 {
    if (request_timeout_sec == 0) return 0;
    return @as(u64, request_timeout_sec) * std.time.ns_per_s;
}

/// Cached prompt IDs from the last request (for KV cache reuse).
var cached_prompt_ids: ?[]u32 = null;
var cached_has_tools: bool = false;

/// Set by generation handlers when output is pad-only (signals bad KV cache state).
var last_generation_was_pad: bool = false;

/// Vision encoder (null if model has no vision support).
var global_vision_encoder: ?*VisionEncoder = null;

/// Model identifier shown in /v1/models. Defaults to architecture family but
/// is overridden in serve() to the model directory basename when known
/// (e.g. "gemma-4-e4b-it-8bit") so clients can distinguish quantizations.
var global_model_id: []const u8 = "";

/// Port the HTTP server is bound to. Used by the landing page's curl
/// example so users can copy-paste a working command.
var global_port: u16 = 0;

/// Tokenizer-vocabulary byte table for grammar-constrained sampling. Built
/// lazily on the first JSON-schema request and reused for the lifetime of the
/// server. Single-threaded inference path means no synchronization is needed.
var global_token_bytes: ?token_mask_mod.TokenBytes = null;
var global_token_bytes_gpa: ?std.mem.Allocator = null;

fn getOrBuildTokenBytes(gpa: std.mem.Allocator, tok: *const Tokenizer) !*const token_mask_mod.TokenBytes {
    if (global_token_bytes) |*tb| return tb;
    log.info("[grammar] building token-byte table for vocab (one-time, ~50ms)\n", .{});
    const built = try token_mask_mod.build(gpa, tok);
    global_token_bytes = built;
    global_token_bytes_gpa = gpa;
    return &global_token_bytes.?;
}

fn deinitGlobalTokenBytes() void {
    if (global_token_bytes) |*tb| {
        tb.deinit();
        global_token_bytes = null;
        global_token_bytes_gpa = null;
    }
}

/// In-memory store for OpenAI Responses API state (`store: true` requests).
/// Bounded LRU; lost on restart.
var global_response_store: ?responses_mod.ResponseStore = null;
var global_response_store_gpa: ?std.mem.Allocator = null;
const RESPONSE_STORE_CAP: usize = 256;
const DEFAULT_API_MAX_TOKENS: u32 = 1024;
const DEFAULT_STRUCTURED_OUTPUT_MAX_TOKENS: u32 = 2048;

fn getOrInitResponseStore(io: std.Io, gpa: std.mem.Allocator) *responses_mod.ResponseStore {
    if (global_response_store == null) {
        global_response_store = responses_mod.ResponseStore.init(io, gpa, RESPONSE_STORE_CAP);
        global_response_store_gpa = gpa;
    }
    return &global_response_store.?;
}

fn deinitGlobalResponseStore() void {
    if (global_response_store) |*store| {
        store.deinit();
        global_response_store = null;
        global_response_store_gpa = null;
    }
}

/// Start the HTTP server on the given host and port.
pub fn serve(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    tok: *const Tokenizer,
    chat_config: *const chat_mod.ChatConfig,
    config: *const model_mod.ModelConfig,
    vision_encoder: ?*VisionEncoder,
    model_dir: []const u8,
    host: []const u8,
    port: u16,
    ctx_size: u32,
    timeout: u32,
    reasoning_budget: i32,
    enable_mtp: bool,
    enable_pld: bool,
    pld_draft_len: u32,
    pld_key_len: u32,
    drafter: ?*drafter_mod.DrafterModel,
    draft_block_size: u32,
) !void {
    global_vision_encoder = vision_encoder;
    max_context_size = ctx_size;
    request_timeout_sec = timeout;
    default_reasoning_budget = reasoning_budget;
    default_enable_mtp = enable_mtp;
    default_enable_pld = enable_pld;
    default_pld_draft_len = pld_draft_len;
    default_pld_key_len = pld_key_len;
    default_drafter = drafter;
    default_enable_drafter = drafter != null;
    default_draft_block_size = draft_block_size;
    global_port = port;
    // Use the directory basename as the public model id (e.g. "gemma-4-e4b-it-8bit").
    // Trim any trailing slash, then take everything after the final slash. Falls
    // back to the architecture family if model_dir is empty.
    global_model_id = blk: {
        var p = model_dir;
        while (p.len > 0 and p[p.len - 1] == '/') p = p[0 .. p.len - 1];
        if (p.len == 0) break :blk config.model_type;
        if (std.mem.lastIndexOfScalar(u8, p, '/')) |i| break :blk p[i + 1 ..];
        break :blk p;
    };
    // Install signal handlers for graceful shutdown
    const sigact = std.posix.Sigaction{
        .handler = .{ .handler = signalHandler },
        .mask = std.posix.sigemptyset(),
        .flags = 0,
    };
    std.posix.sigaction(std.posix.SIG.INT, &sigact, null);
    std.posix.sigaction(std.posix.SIG.TERM, &sigact, null);

    // Parse host address
    var ip4_bytes: [4]u8 = .{ 0, 0, 0, 0 };
    if (!std.mem.eql(u8, host, "0.0.0.0")) {
        // Parse dotted-decimal IP
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

    // Log context size (auto-computed or explicit)
    const safe_ctx = computeMaxSafeContext(config);
    if (max_context_size > 0) {
        log.info("Context size: {d} tokens (manual)\n", .{max_context_size});
    } else {
        log.info("Context size: {d} tokens (auto, from GPU memory)\n", .{safe_ctx});
    }

    if (request_timeout_sec > 0) {
        log.info("Request timeout: {d}s\n", .{request_timeout_sec});
    }
    const model_ctx = config.max_position_embeddings;
    if (model_ctx > 0) {
        log.info("Model context length: {d} tokens\n", .{model_ctx});
    }
    if (default_reasoning_budget >= 0) {
        log.info("Reasoning budget: {d} tokens\n", .{default_reasoning_budget});
    } else {
        log.info("Reasoning budget: unlimited\n", .{});
    }
    if (default_enable_mtp) {
        log.info("MTP speculative decoding: ENABLED (default for new requests)\n", .{});
    }
    if (default_enable_pld) {
        log.info("PLD speculative decoding: ENABLED (draft_len={d}, key_len={d}; default for new requests)\n", .{ default_pld_draft_len, default_pld_key_len });
    }
    if (default_drafter != null) {
        log.info("Drafter speculative decoding: ENABLED (block_size={d}; default for new requests)\n", .{default_draft_block_size});
    }
    log.info("\nServer listening on http://{s}:{d}\n", .{ host, port });
    log.info("  GET  /\n", .{});
    log.info("  GET  /health\n", .{});
    log.info("  GET  /props\n", .{});
    log.info("  GET  /v1/models\n", .{});
    log.info("  POST /v1/chat/completions\n", .{});
    log.info("  POST /v1/completions\n", .{});
    log.info("  POST /v1/embeddings\n", .{});
    log.info("  POST /v1/messages (Anthropic)\n", .{});
    log.info("  POST /v1/responses (OpenAI Responses)\n", .{});
    log.info("  POST /v1/responses/compact\n", .{});
    log.info("  GET  /v1/responses/{{id}}\n", .{});
    log.info("  DEL  /v1/responses/{{id}}\n", .{});
    log.info("  POST /tokenize\n", .{});
    log.info("  POST /detokenize\n\n", .{});

    // Print system metrics once at startup
    const rss = metrics.getAppRssMb();
    if (rss >= 1024) {
        log.info("RSS: {d}.{d}G  Mem: {d}%  CPU: {d}%  GPU: {d}%\n", .{
            rss / 1024, (rss % 1024) * 10 / 1024,
            metrics.getSysMemPct(), metrics.getCpuPct(), metrics.getGpuPct(),
        });
    } else {
        log.info("RSS: {d}M  Mem: {d}%  CPU: {d}%  GPU: {d}%\n", .{
            rss, metrics.getSysMemPct(), metrics.getCpuPct(), metrics.getGpuPct(),
        });
    }

    var poll_fds = [_]std.posix.pollfd{.{
        .fd = server.socket.handle,
        .events = std.posix.POLL.IN,
        .revents = 0,
    }};

    while (!shutdown_requested.load(.acquire)) {
        // Poll with 1-second timeout so we can check shutdown flag
        const poll_result = std.posix.poll(&poll_fds, 1000) catch |err| {
            if (shutdown_requested.load(.acquire)) break;
            log.err("poll error: {}\n", .{err});
            continue;
        };
        if (poll_result == 0) continue; // timeout, re-check shutdown flag
        if (shutdown_requested.load(.acquire)) break;

        const accepted_stream = server.accept(io) catch |err| {
            if (shutdown_requested.load(.acquire)) break;
            log.err("accept error: {}\n", .{err});
            continue;
        };

        // Handle connections synchronously on the listener thread. We previously spawned a thread
        // per connection so health checks could land during generation, but mlx 0.31.2 made
        // streams thread-local — the model weights are bound to the thread that loaded them, and
        // calling `mlx_eval` on a different thread fails with "no Stream(gpu, 1) in current thread".
        // Inference was already serialized by `inference_mutex`, so multi-threaded accept didn't add
        // real concurrency to the slow path; only quick endpoints (/health, /v1/models, /props) get
        // briefly delayed during generation, which is acceptable.
        var conn: Conn = undefined;
        Conn.init(&conn, accepted_stream, io);
        handleConnection(allocator, &conn, xfm, tok, chat_config, config) catch |err| {
            switch (err) {
                error.WriteFailed, error.ReadFailed => log.debug("  -> client disconnected\n", .{}),
                else => log.err("  -> error: {}\n", .{err}),
            }
        };
        conn.close();
    }

    // Free cached prompt on shutdown
    if (cached_prompt_ids) |old| {
        allocator.free(old);
        cached_prompt_ids = null;
    }
    deinitGlobalResponseStore();
    deinitGlobalTokenBytes();

    log.info("\nShutting down gracefully...\n", .{});
}

fn handleConnection(
    allocator: std.mem.Allocator,
    stream: *Conn,
    xfm: *Transformer,
    tok: *const Tokenizer,
    chat_config: *const chat_mod.ChatConfig,
    config: *const model_mod.ModelConfig,
) !void {
    // Read HTTP headers first (up to 16KB), then allocate for the full body based on Content-Length.
    var hdr_buf: [16 * 1024]u8 = undefined;
    var total_read: usize = 0;
    var content_length: ?usize = null;
    var header_end_pos: usize = 0;

    // Phase 1: Read until we have complete headers
    while (total_read < hdr_buf.len) {
        const n = try stream.read(hdr_buf[total_read..]);
        if (n == 0) break;
        total_read += n;

        if (std.mem.indexOf(u8, hdr_buf[0..total_read], "\r\n\r\n")) |he| {
            header_end_pos = he + 4;
            content_length = findContentLength(hdr_buf[0..he]);
            break;
        }
    }

    if (header_end_pos == 0) {
        // No complete headers found
        return;
    }

    // Phase 2: Allocate buffer for full request and read remaining body
    const cl = content_length orelse 0;
    const total_size = header_end_pos + cl;
    const max_request_size = 64 * 1024 * 1024; // 64MB hard limit
    if (total_size > max_request_size) {
        try sendErrorResponse(allocator, stream, "413 Payload Too Large", "invalid_request_error", "Request body too large (max 64MB)", 413);
        return;
    }

    const buf = try allocator.alloc(u8, total_size);
    defer allocator.free(buf);
    @memcpy(buf[0..total_read], hdr_buf[0..total_read]);

    while (total_read < total_size) {
        const n = try stream.read(buf[total_read..total_size]);
        if (n == 0) break;
        total_read += n;
    }

    const request = buf[0..total_read];
    const first_line_end = std.mem.indexOf(u8, request, "\r\n") orelse return;
    const first_line = request[0..first_line_end];

    var line_iter = std.mem.splitScalar(u8, first_line, ' ');
    const method = line_iter.next() orelse return;
    const raw_path = line_iter.next() orelse return;
    // Strip query string for route matching (e.g. /v1/messages?beta=true -> /v1/messages)
    const path = if (std.mem.indexOf(u8, raw_path, "?")) |qpos| raw_path[0..qpos] else raw_path;
    const request_body = if (total_read > header_end_pos) request[header_end_pos..total_read] else "";
    logHttpRequest(method, raw_path, request_body);

    // ── WebSocket upgrade for /v1/responses ──
    // Detect Upgrade: websocket BEFORE the regular route dispatch so the
    // GET method (used for the upgrade handshake) doesn't fall through to
    // the 404 branch.
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/v1/responses") and ws_mod.isUpgrade(request[0..header_end_pos])) {
        if (config.is_encoder_only) {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Encoder-only models do not support text generation. Use /v1/embeddings instead.", 400);
            return;
        }
        try handleResponsesWebSocket(allocator, stream, request[0..header_end_pos], xfm, tok, chat_config, config);
        return;
    }

    if (std.mem.eql(u8, method, "HEAD") and std.mem.eql(u8, path, "/")) {
        // Connectivity check (Claude Code sends HEAD / before any API call)
        log.debug("HEAD / -> 200\n", .{});
        try sendResponse(stream, "200 OK", "text/plain", "");
    } else if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/")) {
        log.debug("GET  / -> 200 (status page)\n", .{});
        try handleStatusPage(allocator, stream, config, chat_config);
    } else if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/health")) {
        log.debug("GET  /health -> 200\n", .{});
        try sendResponse(stream, "200 OK", "application/json", "{\"status\":\"ok\"}");
    } else if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/v1/models")) {
        log.debug("GET  /v1/models -> 200\n", .{});
        try handleModels(allocator, stream, xfm, config, chat_config);
    } else if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/props")) {
        log.debug("GET  /props -> 200\n", .{});
        try handleProps(allocator, stream, config, chat_config);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/chat/completions")) {
        if (config.is_encoder_only) {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Encoder-only models do not support text generation. Use /v1/embeddings instead.", 400);
            return;
        }
        if (!acquireInferenceSlot(stream.io)) {
            log.warn("POST /v1/chat/completions -> 503 (queue full)\n", .{});
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "server_error", "Server request queue is full. Try again shortly.", 503);
            return;
        }
        defer releaseInferenceSlot(stream.io);
        xfm.useCurrentThreadStream();
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleChatCompletions(allocator, stream, body, xfm, tok, chat_config, config);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/completions")) {
        if (config.is_encoder_only) {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Encoder-only models do not support text generation. Use /v1/embeddings instead.", 400);
            return;
        }
        if (!acquireInferenceSlot(stream.io)) {
            log.warn("POST /v1/completions -> 503 (queue full)\n", .{});
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "server_error", "Server request queue is full. Try again shortly.", 503);
            return;
        }
        defer releaseInferenceSlot(stream.io);
        xfm.useCurrentThreadStream();
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleCompletions(allocator, stream, body, xfm, tok, config);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/embeddings")) {
        if (!acquireInferenceSlot(stream.io)) {
            log.warn("POST /v1/embeddings -> 503 (queue full)\n", .{});
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "server_error", "Server request queue is full. Try again shortly.", 503);
            return;
        }
        defer releaseInferenceSlot(stream.io);
        xfm.useCurrentThreadStream();
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleEmbeddings(allocator, stream, body, xfm, tok, config);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/messages")) {
        if (config.is_encoder_only) {
            try sendAnthropicError(allocator, stream, "invalid_request_error", "Encoder-only models do not support text generation. Use /v1/embeddings instead.", 400);
            return;
        }
        if (!acquireInferenceSlot(stream.io)) {
            log.warn("POST /v1/messages -> 529 (queue full)\n", .{});
            try sendAnthropicError(allocator, stream, "overloaded_error", "Server is overloaded. Try again shortly.", 529);
            return;
        }
        defer releaseInferenceSlot(stream.io);
        xfm.useCurrentThreadStream();
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleAnthropicMessages(allocator, stream, body, xfm, tok, chat_config, config);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/responses/compact")) {
        // Compaction is a pure data transformation — no inference slot required.
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleResponsesCompact(allocator, stream, body);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/responses")) {
        if (config.is_encoder_only) {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Encoder-only models do not support text generation. Use /v1/embeddings instead.", 400);
            return;
        }
        if (!acquireInferenceSlot(stream.io)) {
            log.warn("POST /v1/responses -> 503 (queue full)\n", .{});
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "server_error", "Server request queue is full. Try again shortly.", 503);
            return;
        }
        defer releaseInferenceSlot(stream.io);
        xfm.useCurrentThreadStream();
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleResponses(allocator, stream, body, xfm, tok, chat_config, config);
    } else if (std.mem.eql(u8, method, "GET") and std.mem.startsWith(u8, path, "/v1/responses/")) {
        const id = path["/v1/responses/".len..];
        try handleResponsesGet(allocator, stream, id);
    } else if (std.mem.eql(u8, method, "DELETE") and std.mem.startsWith(u8, path, "/v1/responses/")) {
        const id = path["/v1/responses/".len..];
        try handleResponsesDelete(allocator, stream, id);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/tokenize")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleTokenize(allocator, stream, body, tok);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/detokenize")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleDetokenize(allocator, stream, body, tok);
    } else if (std.mem.eql(u8, method, "OPTIONS")) {
        log.debug("OPTIONS {s} -> 204\n", .{path});
        try sendResponse(stream, "204 No Content", "text/plain", "");
    } else {
        log.warn("{s} {s} -> 404\n", .{ method, path });
        try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "The requested endpoint does not exist", null);
    }
}

fn getEffectiveContextLength(config: *const model_mod.ModelConfig) u32 {
    if (max_context_size > 0) return max_context_size;
    // Compute safe default from available GPU memory instead of a fixed 16K cap.
    return computeMaxSafeContext(config);
}

/// Compute the maximum safe context length based on available GPU memory.
/// Solves: (heads × seq² × 4 + layers × 2 × seq × kv_heads × head_dim × 2) × 1.25 ≤ available
fn computeMaxSafeContext(config: *const model_mod.ModelConfig) u32 {
    const heads: u64 = config.num_attention_heads;
    if (heads == 0) return 16384;

    const layers: u64 = config.num_hidden_layers;
    const kv_heads: u64 = config.num_key_value_heads;
    const hdim: u64 = config.head_dim;
    const hidden: u64 = config.hidden_size;
    const ffn: u64 = @max(config.intermediate_size, config.moe_intermediate_size + config.shared_expert_intermediate_size);

    const total_limit = getMetalBufferLimit();
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    const available: u64 = if (total_limit > active_mem) total_limit - active_mem else 0;

    // Linear in seq: per_tok × seq ≤ budget. Mirrors checkAttentionMemory's model.
    // No seq² term — MLX's fused SDPA tiles over seq and never materializes [heads, seq, seq].
    //   KV cache (fp16):   layers × 2 × kv_heads × head_dim × 2 bytes per token
    //   Working (fp16):    ~8 × max(hidden, ffn) × 2 bytes per token (per-layer tensors,
    //                      bounded by EVAL_EVERY_N_LAYERS in transformer.zig)
    const kv_per_tok: u64 = layers * 2 * kv_heads * hdim * 2;
    const work_per_tok: u64 = 8 * @max(hidden, ffn) * 2;
    const per_tok: u64 = kv_per_tok + work_per_tok;
    if (per_tok == 0) return 1024;

    // 80% of available, then divide by 1.25 margin (matches checkAttentionMemory's 5/4)
    const budget: u64 = available * 4 / 5;
    const max_seq: u64 = (budget * 4 / 5) / per_tok;
    if (max_seq == 0) return 1024;

    var result: u32 = if (max_seq > std.math.maxInt(u32)) std.math.maxInt(u32) else @intCast(max_seq);
    // Cap at model's max position embeddings
    if (config.max_position_embeddings > 0) {
        result = @min(result, config.max_position_embeddings);
    }
    return result;
}

/// Estimate peak GPU memory for prefill and reject if it would exceed Metal buffer limit.
/// Metal has a max buffer size (~75% of unified memory). If this is exceeded, the Metal
/// runtime throws an uncatchable C++ exception that crashes the process.
///
/// MLX uses `mlx_fast_scaled_dot_product_attention` — a fused flash-attention-style kernel
/// that tiles over seq and never materializes the full [heads, seq, seq] attention matrix.
/// So peak memory is dominated by (a) the persistent KV cache and (b) per-layer working
/// tensors (QKV projections, MLP intermediates). There is no seq² term.
fn checkAttentionMemory(allocator: std.mem.Allocator, stream: *Conn, prompt_len: usize, config: *const model_mod.ModelConfig, is_anthropic: bool) !bool {
    const heads = config.num_attention_heads;
    if (heads == 0) return true; // unknown architecture, skip check

    const seq: u64 = @intCast(prompt_len);
    const layers: u64 = config.num_hidden_layers;
    const kv_heads: u64 = config.num_key_value_heads;
    const hdim: u64 = config.head_dim;
    const hidden: u64 = config.hidden_size;
    const ffn: u64 = @max(config.intermediate_size, config.moe_intermediate_size + config.shared_expert_intermediate_size);

    // KV cache (all layers, fp16): layers × 2(K+V) × seq × kv_heads × head_dim × 2 bytes.
    // This is persistent for the rest of the request, so it's a real hard cost.
    const kv_bytes: u64 = layers * 2 * seq * kv_heads * hdim * 2;
    // Per-layer working memory during prefill (fp16). The transformer eval()s every N layers,
    // so transient tensors from earlier layers are released. Peak is bounded by a single layer:
    //   QKV projections (~3× seq × hidden) + MLP intermediates (~3× seq × ffn) + residuals.
    // A ~8× seq × max(hidden, ffn) × 2-byte envelope captures this with headroom.
    const working_bytes: u64 = 8 * seq * @max(hidden, ffn) * 2;
    // Total estimate with 25% safety margin
    const needed: u64 = (kv_bytes + working_bytes) * 5 / 4;

    // Available = Metal limit minus current GPU usage (model weights etc.)
    const total_limit: u64 = getMetalBufferLimit();
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    const available = if (total_limit > active_mem) total_limit - active_mem else 0;

    if (needed > available) {
        const needed_mb = needed / (1024 * 1024);
        const avail_mb = available / (1024 * 1024);
        log.warn("  prompt {d} tokens needs ~{d}MB (KV+working+margin), ~{d}MB available — rejecting\n", .{ prompt_len, needed_mb, avail_mb });
        const msg = try std.fmt.allocPrint(allocator,
            "Prompt ({d} tokens) requires ~{d}MB GPU memory but only ~{d}MB available. Reduce prompt size or use a smaller model.", .{ prompt_len, needed_mb, avail_mb });
        defer allocator.free(msg);
        if (is_anthropic) {
            try sendAnthropicError(allocator, stream, "invalid_request_error", msg, 400);
        } else {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, 400);
        }
        return false;
    }
    return true;
}

extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

/// Get the Metal max buffer allocation limit (~75% of system unified memory).
fn getMetalBufferLimit() u64 {
    var mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    _ = sysctlbyname("hw.memsize", @ptrCast(&mem), &len, null, 0);
    if (mem == 0) return 8 * 1024 * 1024 * 1024; // fallback 8GB
    return mem * 75 / 100;
}

/// Clamp max_tokens so prompt + completion doesn't exceed context length.
fn clampMaxTokens(max_tokens: u32, prompt_len: usize) u32 {
    if (max_context_size == 0) return max_tokens;
    const prompt: u32 = @intCast(@min(prompt_len, max_context_size));
    if (prompt >= max_context_size) return 1; // at least 1 token
    const remaining = max_context_size - prompt;
    if (remaining < max_tokens / 4) {
        log.warn("  generation budget squeezed: {d}/{d} tokens remaining (prompt={d}, ctx={d}) — tool call arguments may be truncated\n", .{ remaining, max_tokens, prompt, max_context_size });
    }
    if (max_tokens > remaining) {
        log.debug("  max_tokens clamped: {d} -> {d} (ctx_size={d}, prompt={d})\n", .{ max_tokens, remaining, max_context_size, prompt });
        return remaining;
    }
    return max_tokens;
}

/// Heuristic: chat templates that contain a thinking-block opener indicate the
/// model can produce reasoning_content. Covers Qwen (`enable_thinking`,
/// `<think>`), Gemma 4 (`<|channel>thought`), and generic `<think>` templates.
fn chatTemplateSupportsThinking(tmpl: []const u8) bool {
    return std.mem.indexOf(u8, tmpl, "enable_thinking") != null or
        std.mem.indexOf(u8, tmpl, "<think>") != null or
        std.mem.indexOf(u8, tmpl, "thought") != null or
        std.mem.indexOf(u8, tmpl, "<|channel>") != null;
}

fn handleModels(
    allocator: std.mem.Allocator,
    stream: *Conn,
    xfm: *Transformer,
    config: *const model_mod.ModelConfig,
    chat_config: *const chat_mod.ChatConfig,
) !void {
    const ctx_len = getEffectiveContextLength(config);
    const ctx_str = if (ctx_len > 0) blk: {
        break :blk try std.fmt.allocPrint(allocator, "{d}", .{ctx_len});
    } else try std.fmt.allocPrint(allocator, "null", .{});
    defer allocator.free(ctx_str);

    // ── compute capabilities ──
    var caps = std.ArrayList(u8).empty;
    defer caps.deinit(allocator);
    try caps.append(allocator, '[');
    var n_caps: usize = 0;
    const append_cap = struct {
        fn call(a: std.mem.Allocator, b: *std.ArrayList(u8), n: *usize, name: []const u8) !void {
            if (n.* > 0) try b.append(a, ',');
            try b.append(a, '"');
            try b.appendSlice(a, name);
            try b.append(a, '"');
            n.* += 1;
        }
    }.call;

    const has_chat = !config.is_encoder_only and chat_config.chat_template.len > 0;
    const has_vision = global_vision_encoder != null;
    const has_reasoning = has_chat and chatTemplateSupportsThinking(chat_config.chat_template);

    if (has_chat) try append_cap(allocator, &caps, &n_caps, "chat");
    if (has_chat) try append_cap(allocator, &caps, &n_caps, "tool_use");
    if (has_chat) try append_cap(allocator, &caps, &n_caps, "streaming");
    if (has_vision) try append_cap(allocator, &caps, &n_caps, "vision");
    if (has_reasoning) try append_cap(allocator, &caps, &n_caps, "reasoning");
    if (has_chat) try append_cap(allocator, &caps, &n_caps, "json_schema");
    if (config.is_encoder_only) try append_cap(allocator, &caps, &n_caps, "embeddings");
    try caps.append(allocator, ']');

    // ── input modalities ──
    var mods = std.ArrayList(u8).empty;
    defer mods.deinit(allocator);
    try mods.appendSlice(allocator, "[\"text\"");
    if (has_vision) try mods.appendSlice(allocator, ",\"image\"");
    try mods.append(allocator, ']');

    // ── id falls back to architecture family if serve() didn't set it ──
    const model_id: []const u8 = if (global_model_id.len > 0) global_model_id else config.model_type;

    // `context_length` is the *effective* working ceiling (user-supplied
    // --ctx-size or memory-bounded computeMaxSafeContext). `model_max_tokens`
    // is the model's own declared `max_position_embeddings` from config.json
    // — independent of the running server's chosen context size — so UI code
    // can show the true architectural cap without it shifting around when the
    // user picks a different ctx-size.
    // `supports_mtp` reflects whether MTP weights are actually loaded — the
    // config can declare MTP layers but most MLX-converted Qwen3.5/3.6
    // checkpoints strip the `*.mtp.*` tensors at conversion time, so
    // `--mtp` would fail at first draft. Reading the bound `xfm.mtp_layers`
    // gives the truthful answer for UI gating.
    const body = try std.fmt.allocPrint(allocator,
        \\{{"object":"list","data":[{{"id":"{s}","object":"model","created":{d},"owned_by":"mlx-serve","capabilities":{s},"input_modalities":{s},"meta":{{"architecture":"{s}","vocab_size":{d},"hidden_size":{d},"num_layers":{d},"quantization":"{d}-bit","context_length":{s},"model_max_tokens":{d},"supports_mtp":{s}}}}}]}}
    , .{
        model_id,
        nowSecs(stream.io),
        caps.items,
        mods.items,
        config.model_type,
        config.vocab_size,
        config.hidden_size,
        config.num_hidden_layers,
        config.quant_bits,
        ctx_str,
        config.max_position_embeddings,
        if (xfm.mtp_layers != null) "true" else "false",
    });
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

fn handleProps(allocator: std.mem.Allocator, stream: *Conn, config: *const model_mod.ModelConfig, chat_config: *const chat_mod.ChatConfig) !void {
    const ctx_len = getEffectiveContextLength(config);
    const ctx_str = if (ctx_len > 0) blk: {
        break :blk try std.fmt.allocPrint(allocator, "{d}", .{ctx_len});
    } else try std.fmt.allocPrint(allocator, "0", .{});
    defer allocator.free(ctx_str);

    const safe_ctx = computeMaxSafeContext(config);

    // Query MLX memory usage
    var active_mem: usize = 0;
    var peak_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    _ = mlx.mlx_get_peak_memory(&peak_mem);

    // JSON-escape the chat template
    const escaped_template = try jsonEscape(allocator, chat_config.chat_template);
    defer allocator.free(escaped_template);

    const body = try std.fmt.allocPrint(allocator,
        \\{{"default_generation_settings":{{"model":"{s}","n_ctx":{s}}},"total_slots":1,"chat_template":{s},"model_info":{{"vocab_size":{d},"hidden_size":{d},"num_hidden_layers":{d},"num_attention_heads":{d},"num_key_value_heads":{d},"head_dim":{d},"quantization_bits":{d},"quantization_group_size":{d},"max_position_embeddings":{d}}},"memory":{{"active_bytes":{d},"peak_bytes":{d},"max_safe_context":{d}}}}}
    , .{
        config.model_type,        ctx_str,
        escaped_template,
        config.vocab_size,        config.hidden_size,
        config.num_hidden_layers, config.num_attention_heads,
        config.num_key_value_heads, config.head_dim,
        config.quant_bits,        config.quant_group_size,
        config.max_position_embeddings,
        active_mem,               peak_mem,
        safe_ctx,
    });
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

/// Render the human-friendly landing page at `GET /`. Lists the API
/// surface (OpenAI Chat + Responses, Anthropic Messages, embeddings,
/// utility endpoints) and the loaded model's key facts. No JS, no
/// external assets — single self-contained document.
fn handleStatusPage(
    allocator: std.mem.Allocator,
    stream: *Conn,
    config: *const model_mod.ModelConfig,
    chat_config: *const chat_mod.ChatConfig,
) !void {
    const main_mod = @import("main.zig");

    // Memory + capabilities
    var active_mem: usize = 0;
    var peak_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    _ = mlx.mlx_get_peak_memory(&peak_mem);
    const ctx_len = getEffectiveContextLength(config);
    const has_chat = !config.is_encoder_only and chat_config.chat_template.len > 0;
    const has_vision = global_vision_encoder != null;
    const has_reasoning = has_chat and chatTemplateSupportsThinking(chat_config.chat_template);

    const model_id: []const u8 = if (global_model_id.len > 0) global_model_id else config.model_type;
    const model_id_esc = try htmlEscape(allocator, model_id);
    defer allocator.free(model_id_esc);
    const arch_esc = try htmlEscape(allocator, config.model_type);
    defer allocator.free(arch_esc);
    const version_esc = try htmlEscape(allocator, main_mod.VERSION);
    defer allocator.free(version_esc);

    // Capability pills
    var caps_buf = std.ArrayList(u8).empty;
    defer caps_buf.deinit(allocator);
    if (has_chat) try caps_buf.appendSlice(allocator, "<span class=cap>chat</span>");
    if (has_chat) try caps_buf.appendSlice(allocator, "<span class=cap>tool use</span>");
    if (has_chat) try caps_buf.appendSlice(allocator, "<span class=cap>streaming</span>");
    if (has_chat) try caps_buf.appendSlice(allocator, "<span class=cap>json schema</span>");
    if (has_vision) try caps_buf.appendSlice(allocator, "<span class=cap>vision</span>");
    if (has_reasoning) try caps_buf.appendSlice(allocator, "<span class=cap>reasoning</span>");
    if (config.is_encoder_only) try caps_buf.appendSlice(allocator, "<span class=cap>embeddings</span>");

    const mem_mb: usize = active_mem / (1024 * 1024);
    const peak_mb: usize = peak_mem / (1024 * 1024);

    const body = try std.fmt.allocPrint(allocator,
        \\<!doctype html>
        \\<html lang=en>
        \\<head>
        \\<meta charset=utf-8>
        \\<meta name=viewport content="width=device-width,initial-scale=1">
        \\<title>mlx-serve — {s}</title>
        \\<style>
        \\*{{box-sizing:border-box}}
        \\body{{margin:0;font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"SF Pro Text",Inter,sans-serif;background:#0b0d10;color:#e6e9ee}}
        \\.wrap{{max-width:880px;margin:0 auto;padding:32px 20px 64px}}
        \\h1{{margin:0 0 4px;font-size:22px;font-weight:600;letter-spacing:-.01em}}
        \\h1 .ver{{color:#7d8794;font-weight:400;font-size:14px;margin-left:8px}}
        \\.sub{{color:#7d8794;margin-bottom:28px}}
        \\.dot{{display:inline-block;width:8px;height:8px;border-radius:50%;background:#10b981;margin-right:6px;vertical-align:middle;box-shadow:0 0 0 4px rgba(16,185,129,.15)}}
        \\h2{{font-size:12px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#9aa4b2;margin:24px 0 10px}}
        \\.card{{background:#13161b;border:1px solid #1f242c;border-radius:10px;padding:14px 16px;margin-bottom:10px}}
        \\.row{{display:flex;justify-content:space-between;gap:12px;padding:6px 0;border-bottom:1px solid #1a1e25}}
        \\.row:last-child{{border-bottom:0}}
        \\.row .k{{color:#9aa4b2}}
        \\.row .v{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#e6e9ee}}
        \\.caps{{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}}
        \\.cap{{background:#1a2330;color:#86b8ff;border:1px solid #1f3a5f;border-radius:999px;padding:2px 10px;font-size:12px}}
        \\.ep{{display:grid;grid-template-columns:60px 1fr;gap:12px;align-items:center;padding:8px 0;border-bottom:1px solid #1a1e25}}
        \\.ep:last-child{{border-bottom:0}}
        \\.ep .m{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;font-weight:600;letter-spacing:.04em;text-align:center;border-radius:4px;padding:2px 0}}
        \\.m.get{{background:#173d33;color:#7ee2b8}}
        \\.m.post{{background:#1c2e57;color:#86b8ff}}
        \\.m.del{{background:#3d1d23;color:#ff95a8}}
        \\.m.ws{{background:#3a2a4d;color:#caa6ff}}
        \\.ep .p{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#e6e9ee}}
        \\.ep .d{{color:#7d8794;font-size:13px;margin-top:2px}}
        \\code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#1a1e25;color:#86b8ff;padding:1px 6px;border-radius:4px;font-size:12px}}
        \\pre{{background:#13161b;border:1px solid #1f242c;border-radius:8px;padding:12px;overflow-x:auto;font-size:12px;color:#cfd6df}}
        \\a{{color:#86b8ff;text-decoration:none}}
        \\a:hover{{text-decoration:underline}}
        \\footer{{margin-top:36px;color:#5b6470;font-size:12px;text-align:center}}
        \\</style></head><body>
        \\<div class=wrap>
        \\<h1><span class=dot></span>mlx-serve<span class=ver>v{s}</span></h1>
        \\<div class=sub>Native MLX inference for Apple Silicon · No Python · OpenAI &amp; Anthropic compatible</div>
        \\
        \\<h2>Loaded model</h2>
        \\<div class=card>
        \\<div class=row><span class=k>id</span><span class=v>{s}</span></div>
        \\<div class=row><span class=k>architecture</span><span class=v>{s}</span></div>
        \\<div class=row><span class=k>quantization</span><span class=v>{d}-bit (group {d})</span></div>
        \\<div class=row><span class=k>layers · hidden · heads</span><span class=v>{d} · {d} · {d}/{d}kv</span></div>
        \\<div class=row><span class=k>head dim · vocab</span><span class=v>{d} · {d}</span></div>
        \\<div class=row><span class=k>context length</span><span class=v>{d} tokens (model max {d})</span></div>
        \\<div class=row><span class=k>GPU memory</span><span class=v>{d} MB active · {d} MB peak</span></div>
        \\<div class=caps>{s}</div>
        \\</div>
        \\
        \\<h2>OpenAI Chat Completions</h2>
        \\<div class=card>
        \\<div class=ep><span class="m post">POST</span><div><div class=p>/v1/chat/completions</div><div class=d>Streaming and non-streaming · tool calling · JSON mode · vision (when supported)</div></div></div>
        \\<div class=ep><span class="m post">POST</span><div><div class=p>/v1/completions</div><div class=d>Legacy text completions</div></div></div>
        \\</div>
        \\
        \\<h2>OpenAI Responses</h2>
        \\<div class=card>
        \\<div class=ep><span class="m post">POST</span><div><div class=p>/v1/responses</div><div class=d>Stateful responses with tool calling · stream/non-stream · vision</div></div></div>
        \\<div class=ep><span class="m post">POST</span><div><div class=p>/v1/responses/compact</div><div class=d>Compact a conversation into a round-trippable opaque blob</div></div></div>
        \\<div class=ep><span class="m get">GET</span><div><div class=p>/v1/responses/&#123;id&#125;</div><div class=d>Retrieve a stored response envelope</div></div></div>
        \\<div class=ep><span class="m del">DEL</span><div><div class=p>/v1/responses/&#123;id&#125;</div><div class=d>Delete a stored response</div></div></div>
        \\<div class=ep><span class="m ws">WS</span><div><div class=p>/v1/responses</div><div class=d>WebSocket transport · per-connection store-false cache · sequential turns</div></div></div>
        \\</div>
        \\
        \\<h2>Anthropic Messages</h2>
        \\<div class=card>
        \\<div class=ep><span class="m post">POST</span><div><div class=p>/v1/messages</div><div class=d>Claude SDK / Claude Code compatible · stream &amp; non-stream · tool use · thinking blocks</div></div></div>
        \\</div>
        \\
        \\<h2>Embeddings &amp; utilities</h2>
        \\<div class=card>
        \\<div class=ep><span class="m post">POST</span><div><div class=p>/v1/embeddings</div><div class=d>Vector embeddings (encoder-only models)</div></div></div>
        \\<div class=ep><span class="m post">POST</span><div><div class=p>/tokenize</div><div class=d>Tokenize a string</div></div></div>
        \\<div class=ep><span class="m post">POST</span><div><div class=p>/detokenize</div><div class=d>Detokenize an id sequence</div></div></div>
        \\</div>
        \\
        \\<h2>Discovery</h2>
        \\<div class=card>
        \\<div class=ep><span class="m get">GET</span><div><div class=p><a href=/v1/models>/v1/models</a></div><div class=d>OpenAI models list (id, capabilities, context length)</div></div></div>
        \\<div class=ep><span class="m get">GET</span><div><div class=p><a href=/props>/props</a></div><div class=d>llama.cpp-style server props (chat template, memory)</div></div></div>
        \\<div class=ep><span class="m get">GET</span><div><div class=p><a href=/health>/health</a></div><div class=d>Liveness probe</div></div></div>
        \\</div>
        \\
        \\<h2>Quick start</h2>
        \\<pre>curl http://localhost:{d}/v1/chat/completions \
        \\  -H 'Content-Type: application/json' \
        \\  -d '&#123;"model":"{s}","messages":[&#123;"role":"user","content":"hello"&#125;]&#125;'</pre>
        \\
        \\<footer>mlx-serve · serving <code>{s}</code></footer>
        \\</div></body></html>
    , .{
        // <title>
        model_id_esc,
        // version
        version_esc,
        // model card
        model_id_esc,
        arch_esc,
        config.quant_bits,
        config.quant_group_size,
        config.num_hidden_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.vocab_size,
        ctx_len,
        config.max_position_embeddings,
        mem_mb,
        peak_mb,
        caps_buf.items,
        // curl example
        global_port,
        model_id_esc,
        // footer
        model_id_esc,
    });
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "text/html; charset=utf-8", body);
}

/// Minimal HTML escape — covers the five chars that matter inside element
/// content + double-quoted attribute values. Caller frees.
fn htmlEscape(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    for (input) |c| switch (c) {
        '&' => try buf.appendSlice(allocator, "&amp;"),
        '<' => try buf.appendSlice(allocator, "&lt;"),
        '>' => try buf.appendSlice(allocator, "&gt;"),
        '"' => try buf.appendSlice(allocator, "&quot;"),
        '\'' => try buf.appendSlice(allocator, "&#39;"),
        else => try buf.append(allocator, c),
    };
    return try buf.toOwnedSlice(allocator);
}

fn handleEmbeddings(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    xfm: *Transformer,
    tok: *const Tokenizer,
    config: *const model_mod.ModelConfig,
) !void {
    const gen_mod = @import("generate.zig");
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", null);
        return;
    };
    defer parsed.deinit();
    const root = parsed.value.object;

    // Parse input — can be a string or array of strings
    const input_val = root.get("input") orelse {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Missing 'input' field", null);
        return;
    };

    const model_name = if (root.get("model")) |m| (if (m == .string) m.string else config.model_type) else config.model_type;

    // Collect input texts
    var texts = std.ArrayList([]const u8).empty;
    defer texts.deinit(allocator);

    switch (input_val) {
        .string => |s| try texts.append(allocator, s),
        .array => |arr| {
            for (arr.items) |item| {
                if (item == .string) {
                    try texts.append(allocator, item.string);
                }
            }
        },
        else => {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'input' must be a string or array of strings", null);
            return;
        },
    }

    if (texts.items.len == 0) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'input' must not be empty", null);
        return;
    }

    log.info("POST /v1/embeddings ({d} inputs)\n", .{texts.items.len});

    // Build response JSON
    var resp_buf = std.ArrayList(u8).empty;
    defer resp_buf.deinit(allocator);

    try resp_buf.appendSlice(allocator, "{\"object\":\"list\",\"data\":[");

    var total_tokens: usize = 0;
    for (texts.items, 0..) |text, idx| {
        // Tokenize
        const ids = try tok.encode(allocator, text);
        defer allocator.free(ids);
        total_tokens += ids.len;

        // Reset KV cache for each embedding (no carry-over state)
        try xfm.resetCache();

        // Compute embedding
        const embedding = gen_mod.computeEmbedding(allocator, xfm, ids) catch |err| {
            log.err("  embedding error: {}\n", .{err});
            try sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", "Failed to compute embedding", null);
            return;
        };
        defer allocator.free(embedding);

        if (idx > 0) try resp_buf.appendSlice(allocator, ",");

        // Format: {"object":"embedding","embedding":[...floats...],"index":N}
        const idx_str = try std.fmt.allocPrint(allocator, "{d}", .{idx});
        defer allocator.free(idx_str);
        try resp_buf.appendSlice(allocator, "{\"object\":\"embedding\",\"embedding\":[");

        for (embedding, 0..) |val, i| {
            if (i > 0) try resp_buf.appendSlice(allocator, ",");
            var buf: [32]u8 = undefined;
            const float_str = std.fmt.bufPrint(&buf, "{d:.8}", .{val}) catch "0";
            try resp_buf.appendSlice(allocator, float_str);
        }

        try resp_buf.appendSlice(allocator, "],\"index\":");
        try resp_buf.appendSlice(allocator, idx_str);
        try resp_buf.appendSlice(allocator, "}");
    }

    const total_str = try std.fmt.allocPrint(allocator, "{d}", .{total_tokens});
    defer allocator.free(total_str);
    const model_escaped = try jsonEscape(allocator, model_name);
    defer allocator.free(model_escaped);

    try resp_buf.appendSlice(allocator, "],\"model\":");
    try resp_buf.appendSlice(allocator, model_escaped);
    try resp_buf.appendSlice(allocator, ",\"usage\":{\"prompt_tokens\":");
    try resp_buf.appendSlice(allocator, total_str);
    try resp_buf.appendSlice(allocator, ",\"total_tokens\":");
    try resp_buf.appendSlice(allocator, total_str);
    try resp_buf.appendSlice(allocator, "}}");

    try sendResponse(stream, "200 OK", "application/json", resp_buf.items);
    log.info("  <- {d} embeddings ({d} tokens)\n", .{ texts.items.len, total_tokens });
}

fn handleTokenize(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    tok: *const Tokenizer,
) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON", 400);
        return;
    };
    defer parsed.deinit();
    const root = parsed.value.object;

    const content = if (root.get("content")) |v| (if (v == .string) v.string else null) else null;
    if (content == null) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'content' is required", 400);
        return;
    }

    const ids = try tok.encode(allocator, content.?);
    defer allocator.free(ids);

    var result = std.ArrayList(u8).empty;
    defer result.deinit(allocator);
    try result.appendSlice(allocator, "{\"tokens\":[");
    for (ids, 0..) |id, i| {
        if (i > 0) try result.append(allocator, ',');
        var num_buf: [12]u8 = undefined;
        const num = std.fmt.bufPrint(&num_buf, "{d}", .{id}) catch continue;
        try result.appendSlice(allocator, num);
    }
    try result.appendSlice(allocator, "]}");

    log.debug("POST /tokenize -> {d} tokens\n", .{ids.len});
    try sendResponse(stream, "200 OK", "application/json", result.items);
}

fn handleDetokenize(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    tok: *const Tokenizer,
) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON", 400);
        return;
    };
    defer parsed.deinit();
    const root = parsed.value.object;

    const tokens_val = root.get("tokens") orelse {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'tokens' is required", 400);
        return;
    };
    if (tokens_val != .array) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'tokens' must be an array", 400);
        return;
    }

    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(allocator);
    for (tokens_val.array.items) |item| {
        if (item == .integer) try ids.append(allocator, @intCast(item.integer));
    }

    const text = try tok.decode(allocator, ids.items, false);
    defer allocator.free(text);

    // JSON-escape the text
    var result = std.ArrayList(u8).empty;
    defer result.deinit(allocator);
    try result.appendSlice(allocator, "{\"content\":\"");
    for (text) |c| {
        switch (c) {
            '"' => try result.appendSlice(allocator, "\\\""),
            '\\' => try result.appendSlice(allocator, "\\\\"),
            '\n' => try result.appendSlice(allocator, "\\n"),
            '\r' => try result.appendSlice(allocator, "\\r"),
            '\t' => try result.appendSlice(allocator, "\\t"),
            else => try result.append(allocator, c),
        }
    }
    try result.appendSlice(allocator, "\"}");

    log.debug("POST /detokenize -> {d} tokens -> {d} chars\n", .{ ids.items.len, text.len });
    try sendResponse(stream, "200 OK", "application/json", result.items);
}

fn handleChatCompletions(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    xfm: *Transformer,
    tok: *const Tokenizer,
    chat_config: *const chat_mod.ChatConfig,
    config: *const model_mod.ModelConfig,
) !void {
    // Parse JSON body
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST /v1/chat/completions -> 400 (invalid JSON)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", 400);
        return;
    };
    defer parsed.deinit();

    const root = parsed.value.object;

    // Extract messages
    const messages_val = root.get("messages") orelse {
        log.warn("POST /v1/chat/completions -> 400 (missing messages)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'messages' is a required field", 400);
        return;
    };

    var messages = std.ArrayList(chat_mod.Message).empty;
    defer messages.deinit(allocator);

    // Parse tool call structs for assistant messages (stored temporarily)
    var tool_call_lists = std.ArrayList([]const chat_mod.ToolCall).empty;
    defer {
        for (tool_call_lists.items) |tcs| allocator.free(tcs);
        tool_call_lists.deinit(allocator);
    }

    for (messages_val.array.items) |msg_val| {
        const obj = msg_val.object;
        const role_val = obj.get("role") orelse continue;
        if (role_val != .string) continue;

        // Content can be null for assistant messages with tool_calls
        const content_val = obj.get("content");
        var msg_images: ?[]const chat_mod.ImageData = null;
        const content: []const u8 = if (content_val) |cv| switch (cv) {
            .string => |s| s,
            .array => |arr| blk: {
                var text_content: []const u8 = "";
                var image_list = std.ArrayList(chat_mod.ImageData).empty;
                for (arr.items) |part| {
                    if (part != .object) continue;
                    const ptype = part.object.get("type") orelse continue;
                    if (ptype != .string) continue;
                    if (std.mem.eql(u8, ptype.string, "text")) {
                        const text = part.object.get("text") orelse continue;
                        if (text == .string) text_content = text.string;
                    } else if (std.mem.eql(u8, ptype.string, "image_url")) {
                        // Parse image_url content block
                        const img_obj = part.object.get("image_url") orelse continue;
                        if (img_obj != .object) continue;
                        const url_val = img_obj.object.get("url") orelse continue;
                        if (url_val != .string) continue;
                        if (parseImageUrlContent(allocator, url_val.string)) |img| {
                            image_list.append(allocator, img) catch {
                                allocator.free(img.pixels);
                                continue;
                            };
                        }
                    }
                }
                if (image_list.items.len > 0) {
                    msg_images = image_list.toOwnedSlice(allocator) catch null;
                } else {
                    image_list.deinit(allocator);
                }
                break :blk text_content;
            },
            .null => "",
            else => "",
        } else "";

        // Parse tool_calls from assistant messages
        var msg_tool_calls: ?[]const chat_mod.ToolCall = null;
        if (std.mem.eql(u8, role_val.string, "assistant")) {
            if (obj.get("tool_calls")) |tc_val| {
                if (tc_val == .array) {
                    var tcs = std.ArrayList(chat_mod.ToolCall).empty;
                    for (tc_val.array.items) |tc_item| {
                        if (tc_item != .object) continue;
                        const tc_id = if (tc_item.object.get("id")) |v| (if (v == .string) v.string else "") else "";
                        const func = tc_item.object.get("function") orelse continue;
                        if (func != .object) continue;
                        const fn_name = if (func.object.get("name")) |v| (if (v == .string) v.string else "") else "";
                        const fn_args = if (func.object.get("arguments")) |v| (if (v == .string) v.string else "{}") else "{}";
                        try tcs.append(allocator, .{ .id = tc_id, .name = fn_name, .arguments = fn_args });
                    }
                    if (tcs.items.len > 0) {
                        const owned = try tcs.toOwnedSlice(allocator);
                        try tool_call_lists.append(allocator, owned);
                        msg_tool_calls = owned;
                    } else {
                        tcs.deinit(allocator);
                    }
                }
            }
        }

        // Parse tool_call_id from tool messages
        const tool_call_id: ?[]const u8 = if (std.mem.eql(u8, role_val.string, "tool"))
            (if (obj.get("tool_call_id")) |v| (if (v == .string) v.string else null) else null)
        else
            null;

        // Skip messages with no content, no tool_calls, and no images
        if (content.len == 0 and msg_tool_calls == null and msg_images == null and !std.mem.eql(u8, role_val.string, "tool")) continue;

        try messages.append(allocator, .{
            .role = role_val.string,
            .content = content,
            .tool_calls = msg_tool_calls,
            .tool_call_id = tool_call_id,
            .images = msg_images,
        });
    }

    if (messages.items.len == 0) {
        log.warn("POST /v1/chat/completions -> 400 (no valid messages)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "No valid messages found in request", 400);
        return;
    }

    const max_tokens: u32 = blk: {
        // Support both max_tokens and max_completion_tokens (OpenAI alias)
        const v = root.get("max_tokens") orelse root.get("max_completion_tokens");
        break :blk if (v) |val|
            switch (val) {
                .integer => |i| @intCast(i),
                else => DEFAULT_API_MAX_TOKENS,
            }
        else
            DEFAULT_API_MAX_TOKENS;
    };

    const is_stream = if (root.get("stream")) |v| v == .bool and v.bool else false;

    const temperature = parseJsonFloat(root, "temperature", 1.0, 0.0, 2.0);
    const top_p = parseJsonFloat(root, "top_p", 1.0, 0.0, 1.0);
    const top_k: u32 = if (root.get("top_k")) |v| switch (v) {
        .integer => |i| if (i > 0) @intCast(@min(i, 1000)) else 0,
        .float => |f| if (f > 0) @intFromFloat(@min(f, 1000)) else 0,
        else => 0,
    } else 0;

    const repeat_penalty: f32 = blk: {
        const rp = parseJsonFloat(root, "repeat_penalty", 0.0, 0.0, 10.0);
        if (rp > 0.0) break :blk rp;
        // Also check frequency_penalty (OpenAI format: 0-2 range, mapped to 1.0 + fp)
        const fp = parseJsonFloat(root, "frequency_penalty", 0.0, 0.0, 2.0);
        break :blk if (fp > 0.0) 1.0 + fp else 1.0;
    };

    const presence_penalty = parseJsonFloat(root, "presence_penalty", 0.0, 0.0, 2.0);

    const seed: ?u64 = if (root.get("seed")) |v| switch (v) {
        .integer => |i| @intCast(i),
        else => null,
    } else null;

    // Parse logprobs: "logprobs": true, "top_logprobs": N (0-20)
    const logprobs_n: u32 = blk: {
        const lp = root.get("logprobs") orelse break :blk 0;
        if (lp != .bool or !lp.bool) break :blk 0;
        // logprobs=true without top_logprobs defaults to 0 (just the chosen token's logprob)
        const tlp = root.get("top_logprobs") orelse break :blk 1;
        break :blk switch (tlp) {
            .integer => |i| @intCast(@min(@max(i, 0), 20)),
            else => 1,
        };
    };

    // Extract tools JSON from request body for chat template injection
    var tools_json: ?[]const u8 = null;
    var has_tools = root.get("tools") != null;
    var tool_choice_instruction: ?[]const u8 = null;
    var tool_choice_allocated = false;
    defer if (tool_choice_allocated) {
        if (tool_choice_instruction) |tci| allocator.free(tci);
    };

    if (has_tools) {
        // Parse tool_choice: "none" | "auto" | "required" | {"type":"function","function":{"name":"..."}}
        if (root.get("tool_choice")) |tc| {
            if (tc == .string) {
                if (std.mem.eql(u8, tc.string, "none")) {
                    has_tools = false; // Don't inject tools at all
                } else if (std.mem.eql(u8, tc.string, "required")) {
                    tool_choice_instruction = "\nYou MUST call one of the available functions. Do not respond with text.";
                }
                // "auto" is the default behavior
            } else if (tc == .object) {
                // Specific function: {"type":"function","function":{"name":"fn_name"}}
                if (tc.object.get("function")) |func| {
                    if (func == .object) {
                        if (func.object.get("name")) |name_val| {
                            if (name_val == .string) {
                                tool_choice_instruction = try std.fmt.allocPrint(allocator,
                                    "\nYou MUST call the function \"{s}\". Do not respond with text.", .{name_val.string});
                                tool_choice_allocated = true;
                            }
                        }
                    }
                }
            }
        }

        if (has_tools) {
            // Find the tools array in the raw JSON body and extract it
            if (extractJsonField(body, "tools")) |tools_str| {
                tools_json = tools_str;
            }
        }
    }

    // Parse stop sequences
    var stop_sequences = std.ArrayList([]const u8).empty;
    defer stop_sequences.deinit(allocator);
    if (root.get("stop")) |stop_val| {
        switch (stop_val) {
            .string => |s| try stop_sequences.append(allocator, s),
            .array => |arr| {
                for (arr.items) |item| {
                    if (item == .string) try stop_sequences.append(allocator, item.string);
                }
            },
            else => {},
        }
    }

    // Parse model name from request (use for response, fallback to config)
    const model_name = if (root.get("model")) |v|
        (if (v == .string) v.string else config.model_type)
    else
        config.model_type;

    // Track allocations from response_format injection so we can free them
    var rf_allocs = std.ArrayList([]const u8).empty;
    defer {
        for (rf_allocs.items) |a| allocator.free(a);
        rf_allocs.deinit(allocator);
    }

    // Parse response_format — inject JSON schema constraint into system message,
    // and capture the schema value for grammar-constrained sampling below.
    var grammar_schema_val: ?std.json.Value = null;
    // Backing storage for the synthetic `{"type":"object"}` schema used for
    // `response_format: {type: "json_object"}`. Lives for the request and is
    // freed at the bottom; the schema parser arena copies what it needs.
    var json_object_schema_holder: ?std.json.Parsed(std.json.Value) = null;
    defer if (json_object_schema_holder) |*p| p.deinit();
    if (root.get("response_format")) |rf| {
        if (rf == .object) {
            const rf_type = if (rf.object.get("type")) |t| (if (t == .string) t.string else "") else "";
            if (std.mem.eql(u8, rf_type, "json_schema")) {
                // Extract the schema JSON string from the raw body
                var schema_instruction = std.ArrayList(u8).empty;
                defer schema_instruction.deinit(allocator);
                try schema_instruction.appendSlice(allocator, "Respond with valid JSON only. No other text, no markdown, no explanation. ");

                if (rf.object.get("json_schema")) |js| {
                    if (js == .object) {
                        if (js.object.get("schema")) |schema_val| {
                            grammar_schema_val = schema_val;
                            var out: std.Io.Writer.Allocating = .init(allocator);
                            defer out.deinit();
                            var jws: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
                            schema_val.jsonStringify(&jws) catch {};
                            try schema_instruction.appendSlice(allocator, "Your response MUST conform to this JSON schema:\n");
                            try schema_instruction.appendSlice(allocator, out.written());
                        }
                    }
                }

                const instruction = try allocator.dupe(u8, schema_instruction.items);
                try rf_allocs.append(allocator, instruction);
                if (messages.items.len > 0 and std.mem.eql(u8, messages.items[0].role, "system")) {
                    const combined = try std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ messages.items[0].content, instruction });
                    try rf_allocs.append(allocator, combined);
                    messages.items[0].content = combined;
                } else {
                    try messages.insert(allocator, 0, .{ .role = "system", .content = instruction, .tool_calls = null, .tool_call_id = null });
                }
            } else if (std.mem.eql(u8, rf_type, "json_object")) {
                const instruction = "Respond with valid JSON only. No other text, no markdown fences (no ``` or ```json), no explanation. Begin your response with `{` or `[`.";
                if (messages.items.len > 0 and std.mem.eql(u8, messages.items[0].role, "system")) {
                    const combined = try std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ messages.items[0].content, instruction });
                    try rf_allocs.append(allocator, combined);
                    messages.items[0].content = combined;
                } else {
                    try messages.insert(allocator, 0, .{ .role = "system", .content = instruction, .tool_calls = null, .tool_call_id = null });
                }
                // Belt + braces: also constrain decoding with a permissive
                // object schema so the very first token cannot be a leading
                // backtick (Gemma 4 ignores the prompt-side instruction
                // otherwise). `additionalProperties: true` allows any keys
                // and values — `json_object` only constrains JSON-ness, not
                // a specific shape.
                if (json_object_schema_holder == null) {
                    json_object_schema_holder = std.json.parseFromSlice(std.json.Value, allocator, "{\"type\":\"object\",\"additionalProperties\":true}", .{}) catch null;
                }
                if (json_object_schema_holder) |*p| {
                    grammar_schema_val = p.value;
                }
            }
        }
    }

    // Parse stream_options
    const include_usage = if (root.get("stream_options")) |so| blk: {
        if (so != .object) break :blk false;
        if (so.object.get("include_usage")) |iu| {
            break :blk iu == .bool and iu.bool;
        }
        break :blk false;
    } else false;

    // Parse enable_thinking (default: false — strips <think> blocks from output)
    const enable_thinking = if (root.get("enable_thinking")) |v| v == .bool and v.bool else false;

    // Parse reasoning_budget_tokens: max tokens in <think> block (-1 = unlimited)
    // Per-request override, falls back to server --reasoning-budget flag
    const reasoning_budget: i32 = if (root.get("reasoning_budget_tokens")) |v| switch (v) {
        .integer => |i| @intCast(i),
        else => default_reasoning_budget,
    } else default_reasoning_budget;

    // Parse enable_mtp: per-request override of the --mtp default. Auto-disabled
    // when tools are present in the request — tool-call buffering is incompatible
    // with the multi-token verify path in v1 (see Phase 6 plan).
    var enable_mtp: bool = if (root.get("enable_mtp")) |v|
        (v == .bool and v.bool)
    else
        default_enable_mtp;
    if (enable_mtp and !xfm.config.has_mtp) {
        enable_mtp = false; // model has no MTP head; quietly run without speculation
    }
    if (enable_mtp and has_tools) {
        log.info("  mtp=disabled (tools present)\n", .{});
        enable_mtp = false;
    }
    if (enable_mtp) log.info("  mtp=enabled\n", .{});

    // Parse enable_pld: per-request override of the --pld default. Auto-disabled
    // when MTP is winning the priority race (the model has native MTP weights
    // AND MTP was selected for this request — MTP's drafts are higher-quality
    // for that one model). Also disabled when tools/logprobs/streaming would
    // be incompatible with multi-token verify (mirrors MTP's gates).
    //
    // `pld_explicit_in_json` records whether the request body had `enable_pld`
    // as an explicit field (vs falling back to the server default). The
    // adaptive gate (later, after prompt is tokenized) only disables flags
    // that came from the default — explicit user overrides are honored even
    // on novel content.
    const pld_explicit_in_json: bool = root.get("enable_pld") != null;
    var enable_pld: bool = if (root.get("enable_pld")) |v|
        (v == .bool and v.bool)
    else
        default_enable_pld;
    if (enable_pld and enable_mtp) {
        // Both eligible; prefer MTP for the one model that ships MTP weights.
        log.info("  pld=disabled (mtp takes priority for this request)\n", .{});
        enable_pld = false;
    }
    if (enable_pld and has_tools) {
        log.info("  pld=disabled (tools present)\n", .{});
        enable_pld = false;
    }
    // PLD on hybrid SSM models (LFM2.5, Nemotron-H) works once the SSM
    // snapshot/restore paths handle null ssm_state correctly — see
    // `ssmSnapshot`/`ssmRestore` in transformer.zig. The previous auto-disable
    // here was a workaround; no gating is needed now.
    if (enable_pld) log.info("  pld=enabled (draft_len={d}, key_len={d})\n", .{ default_pld_draft_len, default_pld_key_len });

    // Parse enable_drafter: per-request override of the --drafter default.
    // Auto-disabled when:
    //   - the server didn't load a drafter (`default_drafter == null`)
    //   - tools are present (multi-token verify ↔ tool-call buffering conflict)
    //   - logprobs are requested (drafter doesn't expose realized log-probs
    //     from the draft side; same constraint as MTP)
    //   - hybrid SSM architecture (same SSM-state issue as PLD; drafter would
    //     hit it on the verify forward).
    // Priority: drafter > MTP > PLD > regular. When drafter wins, force MTP
    // and PLD off so logs / state stay consistent.
    const drafter_explicit_in_json: bool = root.get("enable_drafter") != null;
    var enable_drafter: bool = if (root.get("enable_drafter")) |v|
        (v == .bool and v.bool)
    else
        default_enable_drafter;
    if (enable_drafter and default_drafter == null) {
        enable_drafter = false; // no drafter loaded; quietly fall through
    }
    if (enable_drafter and has_tools) {
        log.info("  drafter=disabled (tools present)\n", .{});
        enable_drafter = false;
    }
    if (enable_drafter and logprobs_n > 0) {
        log.info("  drafter=disabled (logprobs requested)\n", .{});
        enable_drafter = false;
    }
    if (enable_drafter and xfm.config.has_hybrid_layers) {
        log.info("  drafter=disabled (hybrid SSM architecture not yet supported for multi-token verify)\n", .{});
        enable_drafter = false;
    }
    if (enable_drafter) {
        if (enable_mtp) {
            log.info("  mtp=disabled (drafter takes priority for this request)\n", .{});
            enable_mtp = false;
        }
        if (enable_pld) {
            log.info("  pld=disabled (drafter takes priority for this request)\n", .{});
            enable_pld = false;
        }
        log.info("  drafter=enabled (block_size={d})\n", .{default_draft_block_size});
    }

    // Log the request
    const last_msg = messages.items[messages.items.len - 1];
    const preview_len = @min(last_msg.content.len, 80);

    // Compute sizes for debug info
    var system_chars: usize = 0;
    var user_chars: usize = 0;
    var tool_msg_count: usize = 0;
    for (messages.items) |msg| {
        if (std.mem.eql(u8, msg.role, "system")) {
            system_chars += msg.content.len;
        } else if (std.mem.eql(u8, msg.role, "user")) {
            user_chars += msg.content.len;
        } else if (std.mem.eql(u8, msg.role, "tool")) {
            tool_msg_count += 1;
        }
    }
    const tools_len = if (tools_json) |tj| tj.len else 0;

    log.info("POST /v1/chat/completions ({d} msgs, max_tokens={d}, temp={d:.2}, top_p={d:.2}, stream={}, thinking={}, sys={d}b, user={d}b, tools={d}b, tool_msgs={d}) \n", .{ messages.items.len, max_tokens, temperature, top_p, is_stream, enable_thinking, system_chars, user_chars, tools_len, tool_msg_count });
    log.info("  > \"{s}{s}\"\n", .{ last_msg.content[0..preview_len], if (last_msg.content.len > 80) "..." else "" });

    // Format chat template
    var prompt_ids_raw = try chat_mod.formatChat(allocator, tok, messages.items, chat_config, tools_json, tool_choice_instruction, enable_thinking);

    // Run vision encoder if any messages contain images
    var vision_prompt_ids: ?[]u32 = null;
    if (global_vision_encoder) |ve| {
        processVisionImages(allocator, ve, xfm, messages.items) catch |err| {
            log.warn("Vision encoding failed: {}\n", .{err});
        };
        // If vision embeddings were set, insert image tokens into the prompt
        // Use the actual encoder output token count
        if (xfm.vision_embeddings != null) {
            const ve_shape = mlx.getShape(xfm.vision_embeddings.?);
            const n_img_tokens: usize = @intCast(ve_shape[1]);
            vision_prompt_ids = try insertImageTokens(allocator, prompt_ids_raw, config.image_token_id, n_img_tokens, config);
            allocator.free(prompt_ids_raw);
            prompt_ids_raw = vision_prompt_ids.?;
        }
    }
    const prompt_ids = prompt_ids_raw;
    defer allocator.free(prompt_ids);
    // Clear vision embeddings after prefill (defer ensures cleanup even on error)
    defer {
        if (xfm.vision_embeddings) |ve| _ = mlx.mlx_array_free(ve);
        xfm.vision_embeddings = null;
    }

    // Enforce context size limit
    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        log.warn("POST /v1/chat/completions -> 400 (prompt {d} tokens exceeds ctx_size {d})\n", .{ prompt_ids.len, effective_ctx });
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Prompt exceeds maximum context length", 400);
        return;
    }

    // Check if attention computation would exceed GPU memory
    if (!try checkAttentionMemory(allocator, stream, prompt_ids.len, config, false)) return;

    // Clamp max_tokens to stay within context window
    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len);
    log.info("  prompt={d} tokens, max_gen={d}, ctx={d}\n", .{ prompt_ids.len, effective_max_tokens, effective_ctx });

    // ── Adaptive spec-decode gating ──
    //
    // PLD and drafter both pay a per-step overhead (lookup + verify forward
    // for PLD; drafter forwards + verify forward for drafter) that only pays
    // off on echo-heavy content. On novel content, the verify forward is wasted
    // work — the bench shows drafter at 0.55× on creative content and PLD at
    // 0.91× on LFM2.5-350M heavy-echo. To make the default-on flip safe we
    // gate per-request on the prompt's n-gram repetition score.
    //
    // Rule: if the score (= ratio of distinct 3-grams that recur in the prompt)
    // is below `spec_gate_threshold` AND the user did not explicitly set the
    // flag in the JSON, disable PLD/drafter for this request. Explicit user
    // overrides are honored — if you `enable_pld:true` on novel content, you
    // get PLD even though it's likely a perf loss; user knows best.
    if ((enable_pld and !pld_explicit_in_json) or (enable_drafter and !drafter_explicit_in_json)) {
        const score = pld_index.ngramRepeatScore(allocator, prompt_ids, 3) catch 1.0; // on error, don't gate
        log.info("  spec-gate: ngram-score={d:.3} (threshold={d:.3})\n", .{ score, spec_gate_threshold });
        if (score < spec_gate_threshold) {
            if (enable_pld and !pld_explicit_in_json) {
                log.info("  pld=disabled (ngram-score={d:.3} < gate threshold {d:.3})\n", .{ score, spec_gate_threshold });
                enable_pld = false;
            }
            if (enable_drafter and !drafter_explicit_in_json) {
                log.info("  drafter=disabled (ngram-score={d:.3} < gate threshold {d:.3})\n", .{ score, spec_gate_threshold });
                enable_drafter = false;
            }
        }
    }

    // Prompt caching: reuse KV cache for shared prefix.
    // Force invalidation when images are present — image tokens have identical IDs
    // but different vision embeddings, so prefix matching would reuse stale features.
    const has_images = xfm.vision_embeddings != null;
    if (has_images) {
        if (cached_prompt_ids) |old| {
            allocator.free(old);
            cached_prompt_ids = null;
        }
        try xfm.resetCache();
        log.info("  [cache] reset — image request\n", .{});
    }
    const cache_result = try reuseKVCache(allocator, xfm, prompt_ids, has_tools);

    const eos_slice = config.eosTokenSlice();

    var sampling = generate_mod.SamplingParams{
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .repeat_penalty = repeat_penalty,
        .presence_penalty = presence_penalty,
        .seed = seed,
    };

    // Build grammar-constrained sampling state if a JSON schema was supplied.
    // Lifetime is scoped to this request — the SchemaConstraint must NOT be
    // moved (the embedded Constraint holds pointers into it).
    var sc: generate_mod.SchemaConstraint = undefined;
    var sc_init = false;
    defer if (sc_init) sc.deinit();

    if (grammar_schema_val) |sv| {
        if (has_tools) {
            log.info("[grammar] skipped JSON schema mask while tools are available (tool calls must remain reachable)\n", .{});
        } else {
            const tb = try getOrBuildTokenBytes(allocator, tok);
            if (sc.initFromValue(allocator, sv, tb)) {
                sc_init = true;
                sampling.constraint = &sc.constraint;
                log.info("[grammar] enforcing JSON schema (vocab={d}, mask={d}b)\n", .{ tb.bytes.len, sc.mask_buf.len });
            } else |err| {
                log.warn("[grammar] schema parse failed ({s}); falling back to prompt-only enforcement\n", .{@errorName(err)});
            }
        }
    }

    if (is_stream) {
        handleStreamingGeneration(allocator, stream, xfm, tok, cache_result.new_tokens, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, include_usage, has_tools, cache_result.cached_tokens, logprobs_n, enable_thinking, reasoning_budget, enable_mtp, enable_pld, enable_drafter) catch |err| {
            log.err("  -> streaming error: {}\n", .{err});
            // Send SSE error event so the client gets a proper error instead of a dropped connection
            const err_chunk = std.fmt.allocPrint(allocator,
                \\data: {{"error":{{"message":"Internal server error: {s}","type":"server_error"}}}}
            , .{@errorName(err)}) catch return;
            defer allocator.free(err_chunk);
            stream.writeAll(err_chunk) catch {};
            stream.writeAll("\n\ndata: [DONE]\n\n") catch {};
        };
    } else {
        handleNonStreamingGeneration(allocator, stream, xfm, tok, cache_result.new_tokens, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, has_tools, cache_result.cached_tokens, logprobs_n, enable_thinking, reasoning_budget, enable_mtp, enable_pld, enable_drafter) catch |err| {
            log.err("  -> 500 ({s})\n", .{@errorName(err)});
            sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", @errorName(err), 500) catch {};
        };
    }

    // Store prompt IDs for next request's cache comparison — but NOT if generation
    // produced pad-only output, which indicates corrupted KV cache state.
    // Tool-calling requests no longer invalidate: reuseKVCache() truncates the cache
    // to the shared prefix, correctly discarding stale generated-token entries.
    if (last_generation_was_pad) {
        log.info("  [cache] invalidating cache after pad-only generation\n", .{});
        try xfm.resetCache();
        if (cached_prompt_ids) |old| {
            allocator.free(old);
            cached_prompt_ids = null;
        }
    } else {
        updateCachedPrompt(allocator, prompt_ids, has_tools);
    }
}

fn handleCompletions(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    xfm: *Transformer,
    tok: *const Tokenizer,
    config: *const model_mod.ModelConfig,
) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST /v1/completions -> 400 (invalid JSON)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", 400);
        return;
    };
    defer parsed.deinit();

    const root = parsed.value.object;

    // Extract prompt (required)
    const prompt_text = if (root.get("prompt")) |v|
        (if (v == .string) v.string else null)
    else
        null;

    if (prompt_text == null) {
        log.warn("POST /v1/completions -> 400 (missing prompt)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'prompt' is a required field", 400);
        return;
    }

    const max_tokens: u32 = blk: {
        const v = root.get("max_tokens") orelse root.get("max_completion_tokens");
        break :blk if (v) |val| switch (val) {
            .integer => |i| @intCast(i),
            else => DEFAULT_API_MAX_TOKENS,
        } else DEFAULT_API_MAX_TOKENS;
    };

    const is_stream = if (root.get("stream")) |v| v == .bool and v.bool else false;

    const temperature: f32 = if (root.get("temperature")) |v| switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => 1.0,
    } else 1.0;

    const top_p: f32 = if (root.get("top_p")) |v| switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => 1.0,
    } else 1.0;

    const top_k: u32 = if (root.get("top_k")) |v| switch (v) {
        .integer => |i| if (i > 0) @intCast(@min(i, 1000)) else 0,
        .float => |f| if (f > 0) @intFromFloat(@min(f, 1000)) else 0,
        else => 0,
    } else 0;

    const repeat_penalty: f32 = if (root.get("repeat_penalty")) |v| switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => blk: {
            break :blk if (root.get("frequency_penalty")) |fp| switch (fp) {
                .float => |f| 1.0 + @as(f32, @floatCast(f)),
                .integer => |i| 1.0 + @as(f32, @floatFromInt(i)),
                else => 1.0,
            } else 1.0;
        },
    } else 1.0;

    const presence_penalty_c: f32 = if (root.get("presence_penalty")) |v| switch (v) {
        .float => |f| @floatCast(@min(@max(f, 0.0), 2.0)),
        .integer => |i| @floatFromInt(@min(@max(i, 0), 2)),
        else => 0.0,
    } else 0.0;

    const seed: ?u64 = if (root.get("seed")) |v| switch (v) {
        .integer => |i| @intCast(i),
        else => null,
    } else null;

    // Parse stop sequences
    var stop_sequences = std.ArrayList([]const u8).empty;
    defer stop_sequences.deinit(allocator);
    if (root.get("stop")) |stop_val| {
        switch (stop_val) {
            .string => |s| try stop_sequences.append(allocator, s),
            .array => |arr| {
                for (arr.items) |item| {
                    if (item == .string) try stop_sequences.append(allocator, item.string);
                }
            },
            else => {},
        }
    }

    const model_name = if (root.get("model")) |v|
        (if (v == .string) v.string else config.model_type)
    else
        config.model_type;

    const include_usage = if (root.get("stream_options")) |so| blk: {
        if (so != .object) break :blk false;
        if (so.object.get("include_usage")) |iu| {
            break :blk iu == .bool and iu.bool;
        }
        break :blk false;
    } else false;

    // Log the request
    const preview_len = @min(prompt_text.?.len, 80);
    log.info("POST /v1/completions (max_tokens={d}, temp={d:.2}, top_p={d:.2}, stream={}) \n", .{ max_tokens, temperature, top_p, is_stream });
    log.info("  > \"{s}{s}\"\n", .{ prompt_text.?[0..preview_len], if (prompt_text.?.len > 80) "..." else "" });

    // Tokenize prompt directly (no chat template)
    const prompt_ids = try tok.encode(allocator, prompt_text.?);
    defer allocator.free(prompt_ids);

    // Enforce context size limit
    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        log.warn("POST /v1/completions -> 400 (prompt {d} tokens exceeds ctx_size {d})\n", .{ prompt_ids.len, effective_ctx });
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Prompt exceeds maximum context length", 400);
        return;
    }

    // Check if attention computation would exceed GPU memory
    if (!try checkAttentionMemory(allocator, stream, prompt_ids.len, config, false)) return;

    // Clamp max_tokens to stay within context window
    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len);

    // Prompt caching: reuse KV cache for shared prefix (completions never have tools)
    const has_tools = false;
    const cache_result = try reuseKVCache(allocator, xfm, prompt_ids, has_tools);

    const eos_slice = config.eosTokenSlice();
    const sampling = generate_mod.SamplingParams{
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .repeat_penalty = repeat_penalty,
        .presence_penalty = presence_penalty_c,
        .seed = seed,
    };

    if (is_stream) {
        handleStreamingCompletion(allocator, stream, xfm, tok, cache_result.new_tokens, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, include_usage, cache_result.cached_tokens) catch |err| {
            log.err("  -> streaming error: {}\n", .{err});
        };
    } else {
        handleNonStreamingCompletion(allocator, stream, xfm, tok, cache_result.new_tokens, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, cache_result.cached_tokens) catch |err| {
            log.err("  -> 500 ({s})\n", .{@errorName(err)});
            sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", @errorName(err), 500) catch {};
        };
    }

    // Store prompt IDs for next request's cache comparison
    updateCachedPrompt(allocator, prompt_ids, has_tools);
}

fn handleNonStreamingCompletion(
    allocator: std.mem.Allocator,
    stream: *Conn,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: generate_mod.SamplingParams,
    eos_token_ids: []const u32,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    cached_tokens: u32,
) !void {
    var timer = startStopwatch(stream.io);

    var result = try generate_mod.generate(stream.io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, getTimeoutNs(), 0);
    result.prompt_tokens += cached_tokens;
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);

    var final_text: []const u8 = result.text;
    var finish_reason = result.finish_reason;
    for (stop_sequences) |stop_seq| {
        if (std.mem.indexOf(u8, final_text, stop_seq)) |idx| {
            final_text = final_text[0..idx];
            finish_reason = "stop";
            break;
        }
    }

    const elapsed_ms = timer.read() / std.time.ns_per_ms;
    log.info("  <- {d}+{d} tokens ({d}ms) [prefill: {d:.1} tok/s, decode: {d:.1} tok/s] [{s}]\n", .{
        result.prompt_tokens, result.completion_tokens, elapsed_ms, result.prefill_tps, result.decode_tps, finish_reason,
    });

    const escaped_text = jsonEscape(allocator, final_text) catch "\"\"";
    defer if (!std.mem.eql(u8, escaped_text, "\"\"")) allocator.free(escaped_text);

    const response = try std.fmt.allocPrint(allocator,
        \\{{"id":"cmpl-{d}","object":"text_completion","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[{{"index":0,"text":{s},"finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
    , .{
        nowMs(stream.io),
        nowSecs(stream.io),
        model_name,
        escaped_text,
        finish_reason,
        result.prompt_tokens,
        result.completion_tokens,
        result.prompt_tokens + result.completion_tokens,
    });
    defer allocator.free(response);

    try sendResponse(stream, "200 OK", "application/json", response);
}

fn handleStreamingCompletion(
    allocator: std.mem.Allocator,
    stream: *Conn,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: generate_mod.SamplingParams,
    eos_token_ids: []const u32,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    include_usage: bool,
    cached_tokens: u32,
) !void {
    const cmpl_id = nowMs(stream.io);
    const created_ts = nowSecs(stream.io);
    var timer = startStopwatch(stream.io);

    var gen = try Generator.init(stream.io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids);
    gen.timeout_ns = getTimeoutNs();
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    timer.reset();

    // SSE headers
    const header =
        "HTTP/1.1 200 OK\r\n" ++
        "Content-Type: text/event-stream\r\n" ++
        "Cache-Control: no-cache\r\n" ++
        "Connection: close\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n" ++
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" ++
        "\r\n";
    try stream.writeAll(header);
    logHttpStreamStart("completions");

    var text_buf = std.ArrayList(u8).empty;
    defer text_buf.deinit(allocator);
    var stopped = false;
    var utf8_carry_c: [3]u8 = undefined;
    var utf8_carry_c_len: u8 = 0;

    while (try gen.next(allocator)) |token_id| {
        const strip = tok.tok_type == .sentencepiece_bpe;
        const raw_decoded_c = try tok.decode(allocator, &[_]u32{token_id}, strip and false);

        // Handle incomplete UTF-8 sequences across token boundaries
        const token_text = blk: {
            const with_carry = if (utf8_carry_c_len > 0) cc: {
                const combined = try allocator.alloc(u8, utf8_carry_c_len + raw_decoded_c.len);
                @memcpy(combined[0..utf8_carry_c_len], utf8_carry_c[0..utf8_carry_c_len]);
                @memcpy(combined[utf8_carry_c_len..], raw_decoded_c);
                allocator.free(raw_decoded_c);
                utf8_carry_c_len = 0;
                break :cc combined;
            } else raw_decoded_c;

            const tail = utf8TrailingIncomplete(with_carry);
            if (tail > 0) {
                @memcpy(utf8_carry_c[0..tail], with_carry[with_carry.len - tail ..]);
                utf8_carry_c_len = @intCast(tail);
            }
            if (with_carry.len == tail) {
                allocator.free(with_carry);
                continue;
            }
            if (tail > 0) {
                const trimmed = try allocator.dupe(u8, with_carry[0 .. with_carry.len - tail]);
                allocator.free(with_carry);
                break :blk trimmed;
            }
            break :blk with_carry;
        };
        defer allocator.free(token_text);

        if (stop_sequences.len > 0) {
            try text_buf.appendSlice(allocator, token_text);
            for (stop_sequences) |stop_seq| {
                if (std.mem.indexOf(u8, text_buf.items, stop_seq) != null) {
                    stopped = true;
                    break;
                }
            }
            if (stopped) break;
        }

        const escaped = try jsonEscape(allocator, token_text);
        defer allocator.free(escaped);
        const chunk = try std.fmt.allocPrint(allocator,
            \\{{"id":"cmpl-{d}","object":"text_completion.chunk","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[{{"index":0,"text":{s},"finish_reason":null}}]}}
        , .{ cmpl_id, created_ts, model_name, escaped });
        defer allocator.free(chunk);

        logHttpSseData(chunk);
        try stream.writeAll("data: ");
        try stream.writeAll(chunk);
        try stream.writeAll("\n\n");
    }

    // Final chunk with finish_reason
    const finish_reason = if (stopped) "stop" else gen.finish_reason;
    const total_prompt = gen.prompt_tokens + cached_tokens;
    const usage_str = if (include_usage) blk: {
        break :blk try std.fmt.allocPrint(allocator,
            \\,"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}
        , .{ total_prompt, gen.completion_tokens, total_prompt + gen.completion_tokens });
    } else try std.fmt.allocPrint(allocator, "", .{});
    defer allocator.free(usage_str);

    const final_chunk = try std.fmt.allocPrint(allocator,
        \\{{"id":"cmpl-{d}","object":"text_completion.chunk","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[{{"index":0,"text":"","finish_reason":"{s}"}}]{s}}}
    , .{ cmpl_id, created_ts, model_name, finish_reason, usage_str });
    defer allocator.free(final_chunk);

    logHttpSseData(final_chunk);
    try stream.writeAll("data: ");
    try stream.writeAll(final_chunk);
    try stream.writeAll("\n\n");
    logHttpSseData("[DONE]");
    try stream.writeAll("data: [DONE]\n\n");

    const decode_ns = timer.read();
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(gen.completion_tokens)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    const elapsed_ms = (prefill_ns + decode_ns) / std.time.ns_per_ms;
    log.info("  <- {d}+{d} tokens streamed ({d}ms) [prefill: {d:.1} tok/s, decode: {d:.1} tok/s] [{s}]\n", .{
        gen.prompt_tokens, gen.completion_tokens, elapsed_ms, prefill_tps, decode_tps, finish_reason,
    });
}

fn handleNonStreamingGeneration(
    allocator: std.mem.Allocator,
    stream: *Conn,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: generate_mod.SamplingParams,
    eos_token_ids: []const u32,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    has_tools: bool,
    cached_tokens: u32,
    logprobs_n: u32,
    enable_thinking: bool,
    reasoning_budget: i32,
    enable_mtp: bool,
    enable_pld: bool,
    enable_drafter: bool,
) !void {
    var timer = startStopwatch(stream.io);

    // Speculative-decoding dispatch (priority: drafter > MTP > PLD > regular).
    //   1. Drafter wins if loaded AND requested AND no logprobs (drafter
    //      cannot expose realized log-probs from the draft side).
    //   2. MTP next if model has MTP weights AND was requested AND no logprobs.
    //   3. PLD next if requested AND no logprobs AND no grammar constraint
    //      (constrained decode requires per-token state advancement).
    //   4. Otherwise the regular pipeline.
    const use_drafter = enable_drafter and logprobs_n == 0 and default_drafter != null and sampling.constraint == null;
    const use_mtp = !use_drafter and enable_mtp and logprobs_n == 0 and sampling.constraint == null and xfm.mtp_layers != null;
    const use_pld = !use_drafter and !use_mtp and enable_pld and logprobs_n == 0 and sampling.constraint == null;
    var result = if (use_drafter)
        try generate_mod.generateDrafter(stream.io, allocator, xfm, default_drafter.?, tok, prompt_ids, max_tokens, sampling, eos_token_ids, getTimeoutNs(), default_draft_block_size)
    else if (use_mtp)
        try generate_mod.generateMtp(stream.io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, getTimeoutNs())
    else if (use_pld)
        try generate_mod.generatePld(stream.io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, getTimeoutNs(), default_pld_draft_len, default_pld_key_len)
    else
        try generate_mod.generate(stream.io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, getTimeoutNs(), logprobs_n);
    result.prompt_tokens += cached_tokens; // Include cached tokens in total prompt count
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);
    defer if (result.logprobs) |lps| {
        for (lps) |*lp| allocator.free(lp.top_logprobs);
        allocator.free(lps);
    };

    // Detect pad-only output (all token IDs are 0) — signals corrupted KV cache
    if (result.token_ids.len > 0) {
        var all_pad = true;
        for (result.token_ids) |tid| {
            if (tid != 0) { all_pad = false; break; }
        }
        last_generation_was_pad = all_pad;
        if (all_pad) {
            log.info("  [cache] pad-only output detected ({d} tokens) — will invalidate cache\n", .{result.token_ids.len});
        }
    }

    // Apply stop sequences: truncate text at first match
    var final_text: []const u8 = result.text;
    var finish_reason = result.finish_reason;
    for (stop_sequences) |stop_seq| {
        if (std.mem.indexOf(u8, final_text, stop_seq)) |idx| {
            final_text = final_text[0..idx];
            finish_reason = "stop";
            break;
        }
    }

    // Apply reasoning budget: truncate reasoning by token count
    // For non-streaming, we truncate after generation since we can't interrupt mid-generation
    var budget_truncated_reasoning: ?[]const u8 = null;
    var budget_reasoning_allocated = false;
    defer if (budget_reasoning_allocated) allocator.free(budget_truncated_reasoning.?);

    if (enable_thinking and reasoning_budget >= 0) {
        const think_split = chat_mod.splitThinkBlock(final_text, true);
        if (think_split.reasoning_content) |reasoning| {
            // Count tokens in reasoning by encoding it
            const reasoning_ids = try tok.encode(allocator, reasoning);
            defer allocator.free(reasoning_ids);
            if (reasoning_ids.len > @as(usize, @intCast(reasoning_budget))) {
                // Truncate: decode only budget-many tokens
                const budget_usize: usize = @intCast(reasoning_budget);
                const truncated_ids = reasoning_ids[0..budget_usize];
                const truncated_text = try tok.decode(allocator, truncated_ids, false);
                budget_truncated_reasoning = truncated_text;
                budget_reasoning_allocated = true;
                log.info("  reasoning budget truncated ({d}/{d} tokens)\n", .{ budget_usize, reasoning_ids.len });
            }
        }
    }

    const elapsed_ms = timer.read() / std.time.ns_per_ms;

    // Check for tool calls in the output
    if (has_tools) {
        log.debug("  checking {d} bytes of generated text for tool calls\n", .{final_text.len});
        if (try chat_mod.parseToolCalls(allocator, final_text)) |tool_calls| {
            defer {
                for (tool_calls) |tc| {
                    allocator.free(tc.name);
                    allocator.free(tc.arguments);
                }
                allocator.free(tool_calls);
            }

            log.info("  <- {d}+{d} tokens ({d}ms) [prefill: {d:.1} tok/s, decode: {d:.1} tok/s] [tool_calls: {d}]\n", .{
                result.prompt_tokens, result.completion_tokens, elapsed_ms, result.prefill_tps, result.decode_tps, tool_calls.len,
            });

            // Build tool_calls JSON array
            var tc_buf = std.ArrayList(u8).empty;
            defer tc_buf.deinit(allocator);
            try tc_buf.appendSlice(allocator, "[");
            for (tool_calls, 0..) |tc, i| {
                if (i > 0) try tc_buf.appendSlice(allocator, ",");
                const tc_id = try std.fmt.allocPrint(allocator, "call_{d}_{d}", .{ nowMs(stream.io), i });
                defer allocator.free(tc_id);
                const escaped_name = try jsonEscape(allocator, tc.name);
                defer allocator.free(escaped_name);
                const escaped_args = try jsonEscape(allocator, tc.arguments);
                defer allocator.free(escaped_args);
                const tc_json = try std.fmt.allocPrint(allocator,
                    \\{{"id":"{s}","type":"function","function":{{"name":{s},"arguments":{s}}}}}
                , .{ tc_id, escaped_name, escaped_args });
                defer allocator.free(tc_json);
                try tc_buf.appendSlice(allocator, tc_json);
            }
            try tc_buf.appendSlice(allocator, "]");

            // Extract reasoning_content if thinking is enabled
            var tc_reasoning_json: []const u8 = "";
            var tc_reasoning_allocated = false;
            if (enable_thinking) {
                const tc_think_split = chat_mod.splitThinkBlock(final_text, true);
                if (tc_think_split.reasoning_content) |reasoning| {
                    const escaped_reasoning = try jsonEscape(allocator, reasoning);
                    tc_reasoning_json = try std.fmt.allocPrint(allocator, ",\"reasoning_content\":{s}", .{escaped_reasoning});
                    allocator.free(escaped_reasoning);
                    tc_reasoning_allocated = true;
                }
            }
            defer if (tc_reasoning_allocated) allocator.free(tc_reasoning_json);

            const response = try std.fmt.allocPrint(allocator,
                \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[{{"index":0,"message":{{"role":"assistant","content":null{s},"tool_calls":{s}}},"finish_reason":"tool_calls"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
            , .{
                nowMs(stream.io),
                nowSecs(stream.io),
                model_name,
                tc_reasoning_json,
                tc_buf.items,
                result.prompt_tokens,
                result.completion_tokens,
                result.prompt_tokens + result.completion_tokens,
            });
            defer allocator.free(response);
            try sendResponse(stream, "200 OK", "application/json", response);
            return;
        }
    }

    log.info("  <- {d}+{d} tokens ({d}ms) [prefill: {d:.1} tok/s, decode: {d:.1} tok/s] [{s}]\n", .{
        result.prompt_tokens, result.completion_tokens, elapsed_ms, result.prefill_tps, result.decode_tps, finish_reason,
    });

    // Split thinking content from response
    const think_split = chat_mod.splitThinkBlock(final_text, enable_thinking);
    const content_text = if (enable_thinking) think_split.content else chat_mod.stripThinkBlock(final_text);

    const escaped_text = jsonEscape(allocator, content_text) catch "\"\"";
    defer if (!std.mem.eql(u8, escaped_text, "\"\"")) allocator.free(escaped_text);

    // Build logprobs JSON if requested
    var logprobs_json: []const u8 = "null";
    var logprobs_allocated = false;
    if (result.logprobs) |lps| {
        logprobs_json = try formatLogprobsObject(allocator, tok, result.token_ids, lps);
        logprobs_allocated = true;
    }
    defer if (logprobs_allocated) allocator.free(logprobs_json);

    // Build reasoning_content field if thinking is enabled and reasoning exists
    var reasoning_json: []const u8 = "";
    var reasoning_allocated = false;
    if (enable_thinking) {
        // Use budget-truncated reasoning if available, otherwise use full reasoning
        const reasoning_text = if (budget_truncated_reasoning) |tr| tr else think_split.reasoning_content;
        if (reasoning_text) |reasoning| {
            const escaped_reasoning = try jsonEscape(allocator, reasoning);
            reasoning_json = try std.fmt.allocPrint(allocator, ",\"reasoning_content\":{s}", .{escaped_reasoning});
            allocator.free(escaped_reasoning);
            reasoning_allocated = true;
        }
    }
    defer if (reasoning_allocated) allocator.free(reasoning_json);

    const response = try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[{{"index":0,"message":{{"role":"assistant","content":{s}{s}}},"logprobs":{s},"finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
    , .{
        nowMs(stream.io),
        nowSecs(stream.io),
        model_name,
        escaped_text,
        reasoning_json,
        logprobs_json,
        finish_reason,
        result.prompt_tokens,
        result.completion_tokens,
        result.prompt_tokens + result.completion_tokens,
    });
    defer allocator.free(response);

    try sendResponse(stream, "200 OK", "application/json", response);
}

/// Token-stream adapter that drives the streaming SSE state machine the same
/// way regardless of whether speculative decoding (MTP / PLD) is on. Each
/// `next()` call yields exactly one token (or null on EOS / max_tokens), so
/// the per-token state machine in `handleStreamingGeneration` /
/// `handleAnthropicStreaming` / `handleResponses` does not need to know
/// anything about multi-token batches that MTP and PLD emit per step.
///
/// EOS-in-batch behavior matches the non-streaming `generateMtp` /
/// `generatePld`: the stop token is NOT yielded — the loop just terminates.
/// This keeps tokens leaking past EOS impossible regardless of where in a
/// 2- or N-token batch the EOS lands.
///
/// `pld_drop_buf` is a heap-owned pending buffer (PLD can yield up to
/// `1+max_draft_len`=16 tokens per step; drafter yields up to `block_size`).
/// Caller must `deinit`.
const StreamMode = enum { regular, mtp, pld, drafter };
const StreamingTokenStream = struct {
    gen: *Generator,
    mode: StreamMode,
    pld_draft_len: u32 = 0,
    pld_key_len: u32 = 0,
    eos_token_ids: []const u32,
    /// Pending tokens from a multi-token speculative step. Drained one at a
    /// time before the next call to nextMtp/nextPld/nextDrafter.
    pending_buf: std.ArrayList(u32) = .empty,
    pending_idx: usize = 0,
    finished: bool = false,

    fn init(gen: *Generator, mode: StreamMode, pld_draft_len: u32, pld_key_len: u32, eos: []const u32) StreamingTokenStream {
        return .{
            .gen = gen,
            .mode = mode,
            .pld_draft_len = pld_draft_len,
            .pld_key_len = pld_key_len,
            .eos_token_ids = eos,
        };
    }

    fn deinit(self: *StreamingTokenStream, allocator: std.mem.Allocator) void {
        self.pending_buf.deinit(allocator);
    }

    /// Yield the next decoded token id, or null if generation is complete.
    /// Mirrors the contract of `Generator.next` for the regular path.
    fn next(self: *StreamingTokenStream, allocator: std.mem.Allocator) !?u32 {
        // Drain any pending tokens from a previous speculative step FIRST,
        // before honoring `self.finished`. The PLD/drafter branches set
        // `self.finished = true` mid-batch when an EOS lands at index > 0;
        // tokens *before* that EOS were pushed to pending_buf and must still
        // flush before we terminate. Skipping the drain here would silently
        // truncate the last few output tokens (observed on Gemma 4 drafter
        // with `print(sum_two(10, 20))` losing its trailing `))`).
        if (self.pending_idx < self.pending_buf.items.len) {
            const tok = self.pending_buf.items[self.pending_idx];
            self.pending_idx += 1;
            return tok;
        }
        // Reset the pending buffer once drained so the speculative path can
        // refill it.
        if (self.pending_idx > 0) {
            self.pending_buf.clearRetainingCapacity();
            self.pending_idx = 0;
        }

        if (self.finished) return null;

        switch (self.mode) {
            .regular => return self.gen.next(allocator),
            .mtp => {
                const r = (try self.gen.nextMtp(allocator)) orelse return null;
                if (generate_mod.isEosId(r.first, self.eos_token_ids)) {
                    self.gen.done = true;
                    self.gen.finish_reason = "stop";
                    self.finished = true;
                    return null;
                }
                if (r.second) |s| {
                    if (generate_mod.isEosId(s, self.eos_token_ids)) {
                        // Emit first; mark finished so the next call returns null.
                        self.gen.done = true;
                        self.gen.finish_reason = "stop";
                        self.finished = true;
                    } else if (self.gen.completion_tokens > self.gen.max_tokens) {
                        // Full-accept advanced completion_tokens by 2; if that
                        // crosses max_tokens, suppress the second token to
                        // avoid over-emit. The pair would otherwise smuggle a
                        // single extra token past the cap.
                        self.gen.done = true;
                        self.gen.finish_reason = "length";
                        self.finished = true;
                    } else {
                        try self.pending_buf.append(allocator, s);
                    }
                }
                return r.first;
            },
            .pld => {
                const r = (try self.gen.nextPld(allocator, self.pld_draft_len, self.pld_key_len)) orelse return null;
                defer allocator.free(r.tokens);
                if (r.tokens.len == 0) {
                    self.finished = true;
                    return null;
                }
                // Walk tokens in order, stopping at the first EOS (and not
                // emitting it) — matches `generatePld`.
                var first_idx: usize = 0;
                while (first_idx < r.tokens.len and generate_mod.isEosId(r.tokens[first_idx], self.eos_token_ids)) : (first_idx += 1) {}
                if (first_idx >= r.tokens.len) {
                    self.gen.done = true;
                    self.gen.finish_reason = "stop";
                    self.finished = true;
                    return null;
                }
                const first_tok = r.tokens[first_idx];
                // Append the remaining (non-EOS-prefixed) tokens to pending,
                // stopping at the first EOS in the tail.
                var i: usize = first_idx + 1;
                while (i < r.tokens.len) : (i += 1) {
                    if (generate_mod.isEosId(r.tokens[i], self.eos_token_ids)) {
                        self.gen.done = true;
                        self.gen.finish_reason = "stop";
                        self.finished = true;
                        break;
                    }
                    try self.pending_buf.append(allocator, r.tokens[i]);
                }
                return first_tok;
            },
            .drafter => {
                // Mirror the PLD branch — `nextDrafter` returns the same
                // `{tokens, accepted_tokens}` shape (full accept yields
                // `block_size` tokens, partial accept yields `1+j`). Walk the
                // batch, stop at the first EOS, push the rest into pending.
                const r = (try self.gen.nextDrafter(allocator)) orelse return null;
                defer allocator.free(r.tokens);
                if (r.tokens.len == 0) {
                    self.finished = true;
                    return null;
                }
                var first_idx: usize = 0;
                while (first_idx < r.tokens.len and generate_mod.isEosId(r.tokens[first_idx], self.eos_token_ids)) : (first_idx += 1) {}
                if (first_idx >= r.tokens.len) {
                    self.gen.done = true;
                    self.gen.finish_reason = "stop";
                    self.finished = true;
                    return null;
                }
                const first_tok = r.tokens[first_idx];
                var i: usize = first_idx + 1;
                while (i < r.tokens.len) : (i += 1) {
                    if (generate_mod.isEosId(r.tokens[i], self.eos_token_ids)) {
                        self.gen.done = true;
                        self.gen.finish_reason = "stop";
                        self.finished = true;
                        break;
                    }
                    try self.pending_buf.append(allocator, r.tokens[i]);
                }
                return first_tok;
            },
        }
    }
};

/// Choose the speculative-decoding mode for a streaming request based on
/// the request flags and the model capabilities. Mirrors the dispatch in
/// `handleNonStreamingGeneration` so streaming and non-streaming pick the
/// same path for the same inputs.
///
/// Priority: drafter > MTP > PLD > regular (matches the non-streaming
/// dispatch in `handleNonStreamingGeneration`). The same per-mode disable
/// rules (no logprobs, no grammar constraint, no hybrid SSM for PLD/drafter)
/// apply at the request-entry parse site; `pickStreamMode` re-enforces them
/// defensively here so a missed gate at the parse site doesn't crash the
/// dispatch.
fn pickStreamMode(
    enable_mtp: bool,
    enable_pld: bool,
    enable_drafter: bool,
    drafter_loaded: bool,
    has_mtp_weights: bool,
    has_hybrid_layers: bool,
    has_constraint: bool,
    logprobs_n: u32,
) StreamMode {
    if (enable_drafter and drafter_loaded and logprobs_n == 0 and !has_constraint and !has_hybrid_layers) return .drafter;
    if (enable_mtp and logprobs_n == 0 and !has_constraint and has_mtp_weights) return .mtp;
    if (enable_pld and logprobs_n == 0 and !has_constraint and !has_hybrid_layers) return .pld;
    return .regular;
}

fn handleStreamingGeneration(
    allocator: std.mem.Allocator,
    stream: *Conn,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: generate_mod.SamplingParams,
    eos_token_ids: []const u32,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    include_usage: bool,
    has_tools: bool,
    cached_tokens: u32,
    logprobs_n: u32,
    enable_thinking: bool,
    reasoning_budget: i32,
    enable_mtp: bool,
    enable_pld: bool,
    enable_drafter: bool,
) !void {
    const chat_id = nowMs(stream.io);
    var timer = startStopwatch(stream.io);

    // Pick the speculative-decoding mode (regular / MTP / PLD / drafter).
    // The per-token state machine below is driven by `StreamingTokenStream`,
    // which feeds `next` (regular), `nextMtp` (1-2 tokens/step), `nextPld`
    // (1..1+draft_len tokens/step), or `nextDrafter` (1..block_size
    // tokens/step) through the same one-token-at-a-time interface.
    const stream_mode = pickStreamMode(enable_mtp, enable_pld, enable_drafter, default_drafter != null, xfm.mtp_layers != null, xfm.config.has_hybrid_layers, sampling.constraint != null, logprobs_n);
    if (stream_mode == .mtp) log.info("  mtp=enabled (streaming)\n", .{});
    if (stream_mode == .pld) log.info("  pld=enabled (streaming, draft_len={d}, key_len={d})\n", .{ default_pld_draft_len, default_pld_key_len });
    if (stream_mode == .drafter) log.info("  drafter=enabled (streaming, block_size={d})\n", .{default_draft_block_size});

    // Prefill + init generator. MTP/PLD/drafter need `initWithOptions` so
    // they get the right post-prefill cache state (and, for MTP+drafter,
    // the captured hidden state for the first draft).
    var gen = switch (stream_mode) {
        .mtp => try Generator.initWithOptions(stream.io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{ .mtp_enabled = true }),
        .pld => try Generator.initWithOptions(stream.io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{ .pld_enabled = true }),
        .drafter => try Generator.initWithOptions(stream.io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{
            .drafter_enabled = true,
            .drafter = default_drafter,
            .drafter_block_size = default_draft_block_size,
        }),
        .regular => try Generator.init(stream.io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids),
    };
    gen.timeout_ns = getTimeoutNs();
    gen.logprobs_n = logprobs_n;
    defer gen.deinit(allocator);

    var ts = StreamingTokenStream.init(&gen, stream_mode, default_pld_draft_len, default_pld_key_len, eos_token_ids);
    defer ts.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    timer.reset();

    // Send SSE headers (no Content-Length — we stream until done)
    const header =
        "HTTP/1.1 200 OK\r\n" ++
        "Content-Type: text/event-stream\r\n" ++
        "Cache-Control: no-cache\r\n" ++
        "Connection: close\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n" ++
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" ++
        "\r\n";
    try stream.writeAll(header);
    logHttpStreamStart("chat.completions");

    // First chunk: role announcement
    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = "assistant", .content = "" }, null, null);

    // Buffer for stop sequence and tool call detection
    var text_buf = std.ArrayList(u8).empty;
    defer text_buf.deinit(allocator);
    // When tools are present, buffer individual token texts for deferred streaming
    var token_texts = std.ArrayList([]const u8).empty;
    defer {
        for (token_texts.items) |t| allocator.free(t);
        token_texts.deinit(allocator);
    }
    var stopped = false;

    // Buffer for incomplete UTF-8 sequences split across BPE tokens
    var utf8_carry: [3]u8 = undefined;
    var utf8_carry_len: u8 = 0;

    // Thinking state for real-time streaming of reasoning_content vs content
    // Supports both <think>...</think> and Gemma 4's <|channel>thought\n...<channel|>
    var in_think_block = enable_thinking; // starts true when thinking enabled (model outputs <think> first)
    var think_buf = std.ArrayList(u8).empty; // buffer to detect close tag across token boundaries
    defer think_buf.deinit(allocator);
    var think_close_tag: []const u8 = "</think>"; // will be updated if Gemma 4 format detected
    var skipped_think_open = false; // track if we've skipped the initial think tag
    var think_tokens: i32 = 0; // count of tokens generated in think block
    var budget_exhausted = false; // true when reasoning budget hit

    // Pad-only detection: track if all generated tokens are pad (token ID 0)
    var all_pad = true;
    var gen_token_count: u32 = 0;

    // Generate tokens via the adapter — yields one decoded token id per call
    // regardless of whether the underlying decode is regular, MTP, or PLD.
    while (try ts.next(allocator)) |token_id| {
        gen_token_count += 1;
        if (token_id != 0) all_pad = false;
        const strip = tok.tok_type == .sentencepiece_bpe;
        const raw_decoded = try tok.decode(allocator, &[_]u32{token_id}, strip and false);

        // Prepend any carried-over bytes from a previous incomplete UTF-8 sequence,
        // then strip any new trailing incomplete bytes into the carry buffer.
        const token_text = blk: {
            // Step 1: prepend carry-over from previous token
            const with_carry = if (utf8_carry_len > 0) cc: {
                const combined = try allocator.alloc(u8, utf8_carry_len + raw_decoded.len);
                @memcpy(combined[0..utf8_carry_len], utf8_carry[0..utf8_carry_len]);
                @memcpy(combined[utf8_carry_len..], raw_decoded);
                allocator.free(raw_decoded);
                utf8_carry_len = 0;
                break :cc combined;
            } else raw_decoded;

            // Step 2: check for trailing incomplete UTF-8 sequence
            const tail = utf8TrailingIncomplete(with_carry);
            if (tail > 0) {
                @memcpy(utf8_carry[0..tail], with_carry[with_carry.len - tail ..]);
                utf8_carry_len = @intCast(tail);
            }

            // Step 3: if everything was incomplete, skip this iteration
            if (with_carry.len == tail) {
                allocator.free(with_carry);
                continue;
            }

            // Step 4: if we trimmed trailing bytes, reallocate to the complete prefix
            if (tail > 0) {
                const trimmed = try allocator.dupe(u8, with_carry[0 .. with_carry.len - tail]);
                allocator.free(with_carry);
                break :blk trimmed;
            }

            break :blk with_carry;
        };

        // Accumulate for stop sequence and tool call detection
        if (has_tools or stop_sequences.len > 0) {
            try text_buf.appendSlice(allocator, token_text);
        }

        // Check stop sequences
        if (stop_sequences.len > 0) {
            var hit_stop = false;
            for (stop_sequences) |stop_seq| {
                if (std.mem.indexOf(u8, text_buf.items, stop_seq)) |_| {
                    hit_stop = true;
                    break;
                }
            }
            if (hit_stop) {
                allocator.free(token_text);
                stopped = true;
                break;
            }
        }

        if (has_tools) {
            // Stream tokens until we detect a tool call pattern starting, then buffer.
            // Tool call prefixes: <tool_call>, <|tool_call>, {"name":
            try token_texts.append(allocator, token_text);
            // text_buf already updated above

            const buf = text_buf.items;
            const maybe_tool = std.mem.indexOf(u8, buf, "<tool_call") != null or
                std.mem.indexOf(u8, buf, "<|tool_call") != null or
                (buf.len > 0 and buf[0] == '{' and std.mem.indexOf(u8, buf, "\"name\"") != null) or
                // Partial prefix at the end that could become a tool call tag
                (buf.len >= 1 and buf[buf.len - 1] == '<');

            if (!maybe_tool) {
                // No tool call pattern — flush buffered tokens as streamed content.
                // But if thinking is enabled, keep buffering until the close tag arrives.
                // Many templates (Qwen 3.5/3.6) pre-inject the opener (`<think>\n`)
                // into the prompt, so the model's tokens are already inside the
                // thinking block — we won't see a literal opener in the buffer.
                // Also detect literal openers for templates that don't pre-inject.
                const has_thinking = enable_thinking or
                    std.mem.indexOf(u8, buf, "<|channel>thought") != null or
                    std.mem.indexOf(u8, buf, "<think>") != null or
                    (std.mem.startsWith(u8, buf, "<|channel>") and buf.len < 18) or
                    (std.mem.startsWith(u8, buf, "<think") and buf.len < 7);
                if (has_thinking) {
                    // Check if the thinking block is complete (has closing tag)
                    const has_close = std.mem.indexOf(u8, buf, "<channel|>") != null or
                        std.mem.indexOf(u8, buf, "</think>") != null;
                    if (!has_close) {
                        // Incomplete thinking block — keep buffering until closed
                    } else {
                        // Complete thinking block — split into reasoning + content
                        const split = chat_mod.splitThinkBlock(buf, enable_thinking);
                        for (token_texts.items) |tt| allocator.free(tt);
                        token_texts.clearRetainingCapacity();
                        text_buf.clearRetainingCapacity();
                        if (enable_thinking) {
                            if (split.reasoning_content) |rc| {
                                try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = rc }, null, null);
                            }
                        }
                        if (split.content.len > 0) {
                            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = split.content }, null, null);
                        }
                    }
                } else {
                    for (token_texts.items) |tt| {
                        defer allocator.free(tt);
                        // Skip bare channel/think tags that leak without a full block
                        if (std.mem.eql(u8, tt, "<|channel>") or std.mem.eql(u8, tt, "<channel|>") or
                            std.mem.eql(u8, tt, "<think>") or std.mem.eql(u8, tt, "</think>"))
                        {
                            continue;
                        }
                        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = tt }, null, null);
                    }
                    token_texts.clearRetainingCapacity();
                }
            }
            // Otherwise keep buffering — tool call may be in progress
        } else if (enable_thinking and in_think_block) {
            // Inside <think> block — stream as reasoning_content with </think> detection
            defer allocator.free(token_text);
            try think_buf.appendSlice(allocator, token_text);
            think_tokens += 1;

            // Skip the initial think tag prefix (<think> or <|channel>thought\n).
            // Many templates (e.g. Qwen 3.5/3.6, some Gemma 4 variants) pre-inject
            // the opener into the prompt so the model's first tokens are already
            // INSIDE the thinking block — no opener appears in the streamed text.
            if (!skipped_think_open and think_buf.items.len >= 7) {
                if (std.mem.startsWith(u8, think_buf.items, "<think>")) {
                    // Remove <think> prefix and any leading newline
                    var skip: usize = 7;
                    while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                } else if (think_buf.items.len >= 17 and std.mem.startsWith(u8, think_buf.items, "<|channel>thought")) {
                    // Gemma 4 think format — switch close tag
                    think_close_tag = "<channel|>";
                    var skip: usize = 17; // len of "<|channel>thought"
                    while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                } else if (think_buf.items.len < 17 and std.mem.startsWith(u8, "<|channel>thought", think_buf.items)) {
                    // Buffer is still a partial prefix of `<|channel>thought` —
                    // wait for more tokens before deciding.
                } else {
                    // Not a known opener — template already injected one.
                    // Stay inside the think block; close tag is detected dynamically below.
                    skipped_think_open = true;
                }
            }

            // Check if reasoning budget exhausted
            if (!budget_exhausted and reasoning_budget >= 0 and think_tokens >= reasoning_budget and skipped_think_open) {
                budget_exhausted = true;
                // Flush all buffered reasoning
                if (think_buf.items.len > 0) {
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = think_buf.items }, null, null);
                }
                think_buf.clearRetainingCapacity();
                in_think_block = false;
                log.info("  reasoning budget exhausted ({d}/{d} tokens)\n", .{ think_tokens, reasoning_budget });
                continue;
            }

            // Check for the close tag — accept whichever appears first.
            // Models with templates that pre-inject the opener (Qwen 3.5/3.6,
            // some Gemma 4 variants) don't reveal which format they use until
            // the close tag arrives, so we look for both.
            const think_pos = std.mem.indexOf(u8, think_buf.items, "</think>");
            const channel_pos = std.mem.indexOf(u8, think_buf.items, "<channel|>");
            const close_match: ?struct { pos: usize, tag: []const u8 } = blk: {
                if (think_pos == null and channel_pos == null) break :blk null;
                if (think_pos == null) break :blk .{ .pos = channel_pos.?, .tag = "<channel|>" };
                if (channel_pos == null) break :blk .{ .pos = think_pos.?, .tag = "</think>" };
                if (think_pos.? <= channel_pos.?) break :blk .{ .pos = think_pos.?, .tag = "</think>" };
                break :blk .{ .pos = channel_pos.?, .tag = "<channel|>" };
            };

            if (close_match) |m| {
                if (m.pos > 0) {
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = think_buf.items[0..m.pos] }, null, null);
                }
                const after = m.pos + m.tag.len;
                var content_after = std.mem.trimStart(u8, think_buf.items[after..], "\n ");
                // Strip Gemma 4 content channel tag: <|channel>\n or <|channel>
                if (std.mem.startsWith(u8, content_after, "<|channel>\n")) {
                    content_after = content_after[11..];
                } else if (std.mem.startsWith(u8, content_after, "<|channel>")) {
                    content_after = content_after[10..];
                }
                content_after = std.mem.trimStart(u8, content_after, "\n ");
                if (content_after.len > 0) {
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = content_after }, null, null);
                }
                think_buf.clearRetainingCapacity();
                in_think_block = false;
                think_close_tag = m.tag;
            } else if (skipped_think_open) {
                // Flush reasoning tokens that can't be part of either close tag.
                // Hold back the longest possible partial-tag suffix (max 9 bytes
                // to cover both "</think>" and "<channel|>").
                const max_partial: usize = 9;
                const safe_len = if (think_buf.items.len > max_partial)
                    think_buf.items.len - max_partial
                else
                    0;
                if (safe_len > 0) {
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = think_buf.items[0..safe_len] }, null, null);
                    const remaining = try allocator.dupe(u8, think_buf.items[safe_len..]);
                    think_buf.clearRetainingCapacity();
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                }
            }
        } else {
            defer allocator.free(token_text);
            // Skip Gemma 4 channel tags that leak after thinking blocks
            if (std.mem.eql(u8, token_text, "<|channel>") or std.mem.eql(u8, token_text, "<channel|>")) {
                continue;
            }
            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = token_text }, null, null);
        }
    }

    // Detect pad-only generation
    if (gen_token_count > 0 and all_pad) {
        last_generation_was_pad = true;
        log.info("  [cache] pad-only output detected ({d} tokens) — will invalidate cache\n", .{gen_token_count});
    } else {
        last_generation_was_pad = false;
    }

    // Flush any remaining think buffer
    if (enable_thinking and think_buf.items.len > 0) {
        if (in_think_block) {
            // Never found </think> — flush as reasoning
            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = think_buf.items }, null, null);
        } else {
            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = think_buf.items }, null, null);
        }
    }

    // After generation: check for tool calls in accumulated text
    var finish_reason: []const u8 = if (stopped) "stop" else gen.finish_reason;
    if (has_tools) {
        log.debug("  checking {d} bytes of streamed text for tool calls\n", .{text_buf.items.len});
        if (try chat_mod.parseToolCalls(allocator, text_buf.items)) |tool_calls| {
            defer {
                for (tool_calls) |tc| {
                    allocator.free(tc.name);
                    allocator.free(tc.arguments);
                }
                allocator.free(tool_calls);
            }

            // Emit reasoning_content before tool calls if thinking is enabled
            if (enable_thinking) {
                const think_split = chat_mod.splitThinkBlock(text_buf.items, true);
                if (think_split.reasoning_content) |reasoning| {
                    // Apply reasoning budget truncation if set
                    const final_reasoning = if (reasoning_budget >= 0) blk: {
                        const r_ids = try tok.encode(allocator, reasoning);
                        defer allocator.free(r_ids);
                        const budget_usize: usize = @intCast(reasoning_budget);
                        if (r_ids.len > budget_usize) {
                            const truncated = try tok.decode(allocator, r_ids[0..budget_usize], false);
                            defer allocator.free(truncated);
                            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = truncated }, null, null);
                            break :blk @as(?[]const u8, null);
                        }
                        break :blk @as(?[]const u8, reasoning);
                    } else @as(?[]const u8, reasoning);
                    if (final_reasoning) |r| {
                        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = r }, null, null);
                    }
                }
            }

            // Emit tool call deltas in OpenAI streaming format
            for (tool_calls, 0..) |tc, i| {
                const tc_id = try std.fmt.allocPrint(allocator, "call_{d}_{d}", .{ chat_id, i });
                defer allocator.free(tc_id);

                // Escape the full arguments string for embedding in JSON
                const escaped_args = try jsonEscape(allocator, tc.arguments);
                defer allocator.free(escaped_args);
                // Strip outer quotes from jsonEscape result (it wraps in "...")
                const args_inner = if (escaped_args.len >= 2 and escaped_args[0] == '"')
                    escaped_args[1 .. escaped_args.len - 1]
                else
                    escaped_args;

                // First delta: name + id + full arguments (clients accumulate these)
                const first_delta = try std.fmt.allocPrint(allocator,
                    \\[{{"index":{d},"id":"{s}","type":"function","function":{{"name":"{s}","arguments":"{s}"}}}}]
                , .{ i, tc_id, tc.name, args_inner });
                defer allocator.free(first_delta);
                try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .tool_calls_json = first_delta }, null, null);
            }
            finish_reason = "tool_calls";
        } else {
            // No tool calls found — flush buffered tokens as content
            if (enable_thinking) {
                // Concatenate all buffered tokens and split thinking from content
                var full_text = std.ArrayList(u8).empty;
                defer full_text.deinit(allocator);
                for (token_texts.items) |t| {
                    try full_text.appendSlice(allocator, t);
                }
                const think_split = chat_mod.splitThinkBlock(full_text.items, true);
                if (think_split.reasoning_content) |reasoning| {
                    // Apply reasoning budget truncation if set
                    const final_reasoning = if (reasoning_budget >= 0) blk: {
                        const r_ids = try tok.encode(allocator, reasoning);
                        defer allocator.free(r_ids);
                        const budget_usize: usize = @intCast(reasoning_budget);
                        if (r_ids.len > budget_usize) {
                            const truncated = try tok.decode(allocator, r_ids[0..budget_usize], false);
                            defer allocator.free(truncated);
                            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = truncated }, null, null);
                            break :blk @as(?[]const u8, null);
                        }
                        break :blk @as(?[]const u8, reasoning);
                    } else @as(?[]const u8, reasoning);
                    if (final_reasoning) |r| {
                        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = r }, null, null);
                    }
                }
                if (think_split.content.len > 0) {
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = think_split.content }, null, null);
                }
            } else {
                for (token_texts.items) |t| {
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = t }, null, null);
                }
            }
        }
    }

    // Final chunk with finish_reason
    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null }, finish_reason, null);

    // Usage chunk (if requested via stream_options.include_usage)
    const total_prompt = gen.prompt_tokens + cached_tokens;
    if (include_usage) {
        const usage_json = try std.fmt.allocPrint(allocator,
            \\{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}
        , .{ total_prompt, gen.completion_tokens, total_prompt + gen.completion_tokens });
        defer allocator.free(usage_json);
        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null }, finish_reason, usage_json);
    }

    // Done sentinel
    logHttpSseData("[DONE]");
    try stream.writeAll("data: [DONE]\n\n");

    const decode_ns = timer.read();
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(gen.completion_tokens)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    const elapsed_ms = (prefill_ns + decode_ns) / std.time.ns_per_ms;
    log.info("  <- {d}+{d} tokens streamed ({d}ms) [prefill: {d:.1} tok/s, decode: {d:.1} tok/s] [{s}]\n", .{
        total_prompt, gen.completion_tokens, elapsed_ms, prefill_tps, decode_tps, finish_reason,
    });
}

const DeltaFields = struct {
    role: ?[]const u8,
    content: ?[]const u8,
    reasoning_content: ?[]const u8 = null,
    tool_calls_json: ?[]const u8 = null,
};

fn sendSSEChunk(
    allocator: std.mem.Allocator,
    stream: *Conn,
    chat_id: i64,
    model_name: []const u8,
    delta: DeltaFields,
    finish_reason: ?[]const u8,
    usage_json: ?[]const u8,
) !void {
    // Build the delta JSON object
    var delta_buf = std.ArrayList(u8).empty;
    defer delta_buf.deinit(allocator);

    try delta_buf.appendSlice(allocator, "{");
    var need_comma = false;

    if (delta.role) |role| {
        try delta_buf.appendSlice(allocator, "\"role\":\"");
        try delta_buf.appendSlice(allocator, role);
        try delta_buf.appendSlice(allocator, "\"");
        need_comma = true;
    }

    if (delta.content) |content| {
        if (need_comma) try delta_buf.appendSlice(allocator, ",");
        try delta_buf.appendSlice(allocator, "\"content\":");
        const escaped = try jsonEscape(allocator, content);
        defer allocator.free(escaped);
        try delta_buf.appendSlice(allocator, escaped);
        need_comma = true;
    }

    if (delta.reasoning_content) |reasoning| {
        if (need_comma) try delta_buf.appendSlice(allocator, ",");
        try delta_buf.appendSlice(allocator, "\"reasoning_content\":");
        const escaped_r = try jsonEscape(allocator, reasoning);
        defer allocator.free(escaped_r);
        try delta_buf.appendSlice(allocator, escaped_r);
        need_comma = true;
    }

    if (delta.tool_calls_json) |tc_json| {
        if (need_comma) try delta_buf.appendSlice(allocator, ",");
        try delta_buf.appendSlice(allocator, "\"tool_calls\":");
        try delta_buf.appendSlice(allocator, tc_json);
    }

    try delta_buf.appendSlice(allocator, "}");

    // Build the finish_reason field
    var fr_buf: [64]u8 = undefined;
    const fr_str = if (finish_reason) |fr|
        std.fmt.bufPrint(&fr_buf, "\"{s}\"", .{fr}) catch "null"
    else
        "null";

    // Build usage field
    const usage_str = if (usage_json) |u| u else "null";

    // Build the full SSE chunk
    const chunk = try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[{{"index":0,"delta":{s},"finish_reason":{s}}}],"usage":{s}}}
    , .{ chat_id, nowSecs(stream.io), model_name, delta_buf.items, fr_str, usage_str });
    defer allocator.free(chunk);

    // Write as SSE event
    logHttpSseData(chunk);
    try stream.writeAll("data: ");
    try stream.writeAll(chunk);
    try stream.writeAll("\n\n");
}

// ── Shared utilities ──

/// Compare new prompt with cached prompt, truncate KV cache to shared prefix,
/// and return only the new tokens to process.
const CacheResult = struct {
    new_tokens: []const u32,
    cached_tokens: u32, // how many tokens were reused from cache
};

fn reuseKVCache(allocator: std.mem.Allocator, xfm: *Transformer, prompt_ids: []const u32, has_tools: bool) !CacheResult {
    // Invalidate cache if tool configuration changed between requests
    if (has_tools != cached_has_tools) {
        if (cached_prompt_ids) |old| {
            allocator.free(old);
            cached_prompt_ids = null;
        }
        log.info("  [cache] reset — tools config changed\n", .{});
    }

    if (cached_prompt_ids) |cached| {
        // Find shared prefix length
        const max_shared = @min(cached.len, prompt_ids.len);
        var shared: usize = 0;
        while (shared < max_shared and cached[shared] == prompt_ids[shared]) {
            shared += 1;
        }

        // Only reuse when the new prompt is a strict extension of the cached one.
        // For hybrid architectures (Qwen 3.5/3.6 GatedDeltaNet, Nemotron-H, LFM2),
        // the SSM/conv recurrent state cannot be rolled back — it has processed
        // the cached prompt AND any tokens generated afterwards. Partial truncation
        // of the KV cache leaves the SSM state stale, producing immediate EOS or
        // garbage on the next generation. Any divergence (shorter, diverged, or
        // identical-re-issue) forces a clean reset.
        if (shared < cached.len) {
            log.info("  [cache] reset — new prompt diverges from cached (shared={d} cached={d})\n", .{ shared, cached.len });
            allocator.free(cached);
            cached_prompt_ids = null;
            try xfm.resetCache();
            return .{ .new_tokens = prompt_ids, .cached_tokens = 0 };
        }

        // shared == cached.len at this point. Identical re-issue (shared == prompt_ids.len)
        // still resets: KV buffer holds stale post-gen tokens that truncate() can't scrub
        // on every architecture.
        if (shared == prompt_ids.len) {
            log.info("  [cache] reset — identical prompt re-issued (stale generation residue)\n", .{});
            allocator.free(cached);
            cached_prompt_ids = null;
            try xfm.resetCache();
            return .{ .new_tokens = prompt_ids, .cached_tokens = 0 };
        }

        // Sliding window models (e.g. Gemma 4) interleave global and local attention layers.
        // The KV cache buffers store ALL tokens regardless of window size — only the
        // *views* returned during decode are windowed. Since truncate() just updates
        // offsets and re-slices from the intact buffer, prefix reuse is safe: the cached
        // K vectors retain their original RoPE position embeddings at the correct indices.

        if (shared > 0) {
            // Truncate KV cache to the shared prefix
            try xfm.cache.truncate(shared, xfm.s);
            xfm.moe_seq_offset = shared;
            log.debug("  [cache] reusing {d}/{d} tokens from previous prompt\n", .{ shared, prompt_ids.len });
            return .{
                .new_tokens = prompt_ids[shared..],
                .cached_tokens = @intCast(shared),
            };
        }
    }

    // No cache hit — reset completely
    if (cached_prompt_ids != null) {
        log.info("  [cache] reset — no shared prefix\n", .{});
    }
    try xfm.resetCache();
    return .{ .new_tokens = prompt_ids, .cached_tokens = 0 };
}

fn updateCachedPrompt(allocator: std.mem.Allocator, prompt_ids: []const u32, has_tools: bool) void {
    if (cached_prompt_ids) |old| {
        allocator.free(old);
    }
    cached_prompt_ids = allocator.dupe(u32, prompt_ids) catch null;
    cached_has_tools = has_tools;
}

fn sendResponse(stream: *Conn, status: []const u8, content_type: []const u8, body: []const u8) !void {
    logHttpResponse(status, content_type, body);

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
    if (body.len > 0) try stream.writeAll(body);
}

fn logHttpRequest(method: []const u8, path: []const u8, body: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] -> {s} {s} body={d}b\n", .{ method, path, body.len });
    logHttpBody("[http] request body", body);
}

fn logHttpResponse(status: []const u8, content_type: []const u8, body: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] <- {s} {s} body={d}b\n", .{ status, content_type, body.len });
    logHttpBody("[http] response body", body);
}

fn logHttpStreamStart(kind: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] <- 200 OK text/event-stream ({s})\n", .{kind});
}

fn logHttpSseEvent(event_name: []const u8, data: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] <- sse event={s} data={d}b\n", .{ event_name, data.len });
    logHttpBody("[http] sse data", data);
}

fn logHttpSseData(data: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] <- sse data={d}b\n", .{data.len});
    logHttpBody("[http] sse data", data);
}

fn logHttpBody(label: []const u8, body: []const u8) void {
    if (body.len == 0) return;
    log.debug("{s} ({d}b):\n{s}\n", .{
        label,
        body.len,
        body,
    });
}

fn findContentLength(headers: []const u8) ?usize {
    var lines = std.mem.splitSequence(u8, headers, "\r\n");
    while (lines.next()) |line| {
        const lower = "content-length: ";
        if (line.len >= lower.len) {
            var match = true;
            for (0..lower.len) |j| {
                if (std.ascii.toLower(line[j]) != lower[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return std.fmt.parseInt(usize, std.mem.trim(u8, line[lower.len..], " "), 10) catch null;
            }
        }
    }
    return null;
}

/// Extract a JSON field's raw value from a JSON body string.
/// Returns the raw substring for the field value (e.g., the array or object).
fn extractJsonField(body: []const u8, field: []const u8) ?[]const u8 {
    // Search for "field": or "field" :
    var pos: usize = 0;
    while (pos < body.len) {
        const quote_pos = std.mem.indexOf(u8, body[pos..], "\"") orelse return null;
        const key_start = pos + quote_pos + 1;
        if (key_start + field.len >= body.len) return null;

        if (std.mem.eql(u8, body[key_start .. key_start + field.len], field) and
            body[key_start + field.len] == '"')
        {
            // Found the key, skip to colon
            var i = key_start + field.len + 1;
            while (i < body.len and (body[i] == ' ' or body[i] == ':' or body[i] == '\n' or body[i] == '\r' or body[i] == '\t')) {
                i += 1;
            }
            if (i >= body.len) return null;

            // Now extract the value - find matching bracket/brace
            const start = i;
            const open = body[start];
            const close: u8 = if (open == '[') ']' else if (open == '{') '}' else return null;
            var depth: usize = 1;
            var j = start + 1;
            var in_string = false;
            while (j < body.len and depth > 0) {
                if (body[j] == '\\' and in_string) {
                    j += 1; // skip escaped char
                } else if (body[j] == '"') {
                    in_string = !in_string;
                } else if (!in_string) {
                    if (body[j] == open) depth += 1;
                    if (body[j] == close) depth -= 1;
                }
                j += 1;
            }
            if (depth == 0) return body[start..j];
            return null;
        }
        pos = key_start;
    }
    return null;
}

fn sendErrorResponse(allocator: std.mem.Allocator, stream: *Conn, status: []const u8, err_type: []const u8, message: []const u8, code: ?u32) !void {
    const escaped_msg = try jsonEscape(allocator, message);
    defer allocator.free(escaped_msg);

    var code_buf: [16]u8 = undefined;
    const code_str = if (code) |c|
        std.fmt.bufPrint(&code_buf, "{d}", .{c}) catch "null"
    else
        "null";

    const body = try std.fmt.allocPrint(allocator,
        \\{{"error":{{"message":{s},"type":"{s}","param":null,"code":{s}}}}}
    , .{ escaped_msg, err_type, code_str });
    defer allocator.free(body);
    try sendResponse(stream, status, "application/json", body);
}

/// Returns the number of trailing bytes that form an incomplete UTF-8 sequence.
/// If the string ends with a complete codepoint (or is empty), returns 0.
fn utf8TrailingIncomplete(s: []const u8) usize {
    if (s.len == 0) return 0;
    // Walk backwards to find the last leading byte (one with bit pattern 11xxxxxx or 0xxxxxxx)
    var i: usize = s.len;
    // Check up to 3 trailing continuation bytes (10xxxxxx)
    var cont: usize = 0;
    while (cont < 3 and i > 0) {
        i -= 1;
        if (s[i] & 0xC0 != 0x80) break; // found a non-continuation byte
        cont += 1;
    }
    // i now points to the last leading byte (or the byte that broke the loop)
    if (i >= s.len) return 0;
    const lead = s[i];
    // Determine expected sequence length from leading byte
    const expected: usize = if (lead & 0x80 == 0) 1 // 0xxxxxxx — ASCII
    else if (lead & 0xE0 == 0xC0) 2 // 110xxxxx
    else if (lead & 0xF0 == 0xE0) 3 // 1110xxxx
    else if (lead & 0xF8 == 0xF0) 4 // 11110xxx
    else return 0; // invalid leading byte, don't buffer
    const actual = s.len - i;
    return if (actual < expected) actual else 0;
}

fn jsonEscape(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);

    try result.append(allocator, '"');
    for (input) |c| {
        switch (c) {
            '"' => try result.appendSlice(allocator, "\\\""),
            '\\' => try result.appendSlice(allocator, "\\\\"),
            '\n' => try result.appendSlice(allocator, "\\n"),
            '\r' => try result.appendSlice(allocator, "\\r"),
            '\t' => try result.appendSlice(allocator, "\\t"),
            else => {
                if (c < 0x20) {
                    var esc_buf: [6]u8 = undefined;
                    const s = std.fmt.bufPrint(&esc_buf, "\\u{x:0>4}", .{c}) catch unreachable;
                    try result.appendSlice(allocator, s);
                } else {
                    try result.append(allocator, c);
                }
            },
        }
    }
    try result.append(allocator, '"');
    return result.toOwnedSlice(allocator);
}

/// Build logprobs JSON for a single token (for both streaming and non-streaming).
/// Returns a string like: {"token":"hello","logprob":-1.23,"bytes":[104,101],"top_logprobs":[...]}
fn formatTokenLogprob(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    token_id: u32,
    logprob: f32,
    top_logprobs: []const generate_mod.TokenLogprob,
) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);

    const strip = tok.tok_type == .sentencepiece_bpe;
    const token_text = try tok.decode(allocator, &[_]u32{token_id}, strip and false);
    defer allocator.free(token_text);

    const escaped_token = try jsonEscape(allocator, token_text);
    defer allocator.free(escaped_token);

    // Build bytes array
    var bytes_buf = std.ArrayList(u8).empty;
    defer bytes_buf.deinit(allocator);
    try bytes_buf.appendSlice(allocator, "[");
    for (token_text, 0..) |b, i| {
        if (i > 0) try bytes_buf.appendSlice(allocator, ",");
        const num = try std.fmt.allocPrint(allocator, "{d}", .{b});
        defer allocator.free(num);
        try bytes_buf.appendSlice(allocator, num);
    }
    try bytes_buf.appendSlice(allocator, "]");

    // Build top_logprobs array
    var top_buf = std.ArrayList(u8).empty;
    defer top_buf.deinit(allocator);
    try top_buf.appendSlice(allocator, "[");
    for (top_logprobs, 0..) |tlp, i| {
        if (i > 0) try top_buf.appendSlice(allocator, ",");

        const tlp_text = try tok.decode(allocator, &[_]u32{tlp.token_id}, strip and false);
        defer allocator.free(tlp_text);
        const escaped_tlp = try jsonEscape(allocator, tlp_text);
        defer allocator.free(escaped_tlp);

        // Bytes for this token
        var tlp_bytes = std.ArrayList(u8).empty;
        defer tlp_bytes.deinit(allocator);
        try tlp_bytes.appendSlice(allocator, "[");
        for (tlp_text, 0..) |b, j| {
            if (j > 0) try tlp_bytes.appendSlice(allocator, ",");
            const num = try std.fmt.allocPrint(allocator, "{d}", .{b});
            defer allocator.free(num);
            try tlp_bytes.appendSlice(allocator, num);
        }
        try tlp_bytes.appendSlice(allocator, "]");

        const entry = try std.fmt.allocPrint(allocator,
            \\{{"token":{s},"logprob":{d:.6},"bytes":{s}}}
        , .{ escaped_tlp, tlp.logprob, tlp_bytes.items });
        defer allocator.free(entry);
        try top_buf.appendSlice(allocator, entry);
    }
    try top_buf.appendSlice(allocator, "]");

    const result = try std.fmt.allocPrint(allocator,
        \\{{"token":{s},"logprob":{d:.6},"bytes":{s},"top_logprobs":{s}}}
    , .{ escaped_token, logprob, bytes_buf.items, top_buf.items });

    return result;
}

/// Build the full logprobs object for a non-streaming response.
fn formatLogprobsObject(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    token_ids: []const u32,
    logprobs: []const generate_mod.LogprobResult,
) ![]const u8 {
    var content_buf = std.ArrayList(u8).empty;
    defer content_buf.deinit(allocator);

    try content_buf.appendSlice(allocator, "[");
    const count = @min(token_ids.len, logprobs.len);
    for (0..count) |i| {
        if (i > 0) try content_buf.appendSlice(allocator, ",");
        const entry = try formatTokenLogprob(allocator, tok, token_ids[i], logprobs[i].token_logprob, logprobs[i].top_logprobs);
        defer allocator.free(entry);
        try content_buf.appendSlice(allocator, entry);
    }
    try content_buf.appendSlice(allocator, "]");

    return try std.fmt.allocPrint(allocator, "{{\"content\":{s}}}", .{content_buf.items});
}

/// Parse a float from a JSON value, clamping to [min, max]. Returns default if missing/invalid.
fn parseJsonFloat(root: std.json.ObjectMap, key: []const u8, default: f32, min: f32, max: f32) f32 {
    const raw = if (root.get(key)) |v| switch (v) {
        .float => |f| @as(f32, @floatCast(f)),
        .integer => |i| @as(f32, @floatFromInt(i)),
        else => default,
    } else default;
    return std.math.clamp(raw, min, max);
}

// ── Vision Processing ──

/// Collect images from messages, run vision encoder, set embeddings on transformer.
fn processVisionImages(
    allocator: std.mem.Allocator,
    vision_enc: *VisionEncoder,
    xfm: *Transformer,
    msgs: []const chat_mod.Message,
) !void {
    // Only process images from the LAST user message.
    // Previous turns' images were already processed in their original request;
    // re-processing them wastes context and causes stale feature confusion.
    var last_user_images: ?[]const chat_mod.ImageData = null;
    var i = msgs.len;
    while (i > 0) {
        i -= 1;
        if (std.mem.eql(u8, msgs[i].role, "user")) {
            last_user_images = msgs[i].images;
            break;
        }
    }

    const images = last_user_images orelse return;
    if (images.len == 0) return;

    log.info("Vision: processing {d} image(s)\n", .{images.len});

    // Encode each image and concatenate embeddings along the token dimension.
    // Each image produces [1, N, hidden], concatenated → [1, total_tokens, hidden].
    var emb_parts = std.ArrayList(mlx.mlx_array).empty;
    defer {
        for (emb_parts.items) |e| _ = mlx.mlx_array_free(e);
        emb_parts.deinit(allocator);
    }

    for (images) |img| {
        const h: c_int = @intCast(img.height);
        const w: c_int = @intCast(img.width);
        const shape = [_]c_int{ 1, 3, h, w };
        const pixel_arr = mlx.mlx_array_new_data(img.pixels.ptr, &shape, 4, .float32);
        defer _ = mlx.mlx_array_free(pixel_arr);
        const emb = try vision_enc.forward(pixel_arr);
        try emb_parts.append(allocator, emb);
    }

    if (emb_parts.items.len == 1) {
        // Single image — use directly
        const ve_shape = mlx.getShape(emb_parts.items[0]);
        log.debug("  Vision: [{d},{d},{d}] tokens\n", .{ ve_shape[0], ve_shape[1], ve_shape[2] });
        xfm.vision_embeddings = emb_parts.items[0];
        // Prevent the defer from freeing the one we're using
        emb_parts.items[0] = mlx.mlx_array_new();
    } else {
        // Multiple images — concatenate along token dim (axis=1)
        const cat_vec = mlx.mlx_vector_array_new_data(emb_parts.items.ptr, emb_parts.items.len);
        defer _ = mlx.mlx_vector_array_free(cat_vec);
        var combined = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&combined, cat_vec, 1, vision_enc.s));
        const ve_shape = mlx.getShape(combined);
        log.info("  Vision: {d} images → [{d},{d},{d}] tokens\n", .{ emb_parts.items.len, ve_shape[0], ve_shape[1], ve_shape[2] });
        xfm.vision_embeddings = combined;
    }
}

/// Insert BOI + N×image_token + EOI into the prompt before the last user turn's content.
/// n_tokens: the expected image_seq_length (e.g. 280 from config).
fn insertImageTokens(allocator: std.mem.Allocator, prompt_ids: []const u32, image_token_id: u32, n_tokens: usize, config: *const model_mod.ModelConfig) ![]u32 {
    if (image_token_id == 0 or n_tokens == 0) return try allocator.dupe(u32, prompt_ids);

    // Find the last USER turn and insert image tokens immediately after it.
    // The marker IDs come from encoding an architecture-specific prefix
    // (e.g. "<|turn>user\n") at server startup — see populateUserTurnMarker.
    const marker = config.userTurnMarkerSlice();
    var insert_pos: usize = 0;
    var found_turn = false;
    if (marker.len > 0 and prompt_ids.len >= marker.len) {
        var i = prompt_ids.len - marker.len;
        while (true) {
            if (std.mem.eql(u32, prompt_ids[i .. i + marker.len], marker)) {
                insert_pos = i + marker.len;
                found_turn = true;
                break;
            }
            if (i == 0) break;
            i -= 1;
        }
    }
    if (!found_turn) {
        // Fallback: insert after BOS + system prompt, before last few tokens
        log.warn("insertImageTokens: user turn marker not found (marker_len={d}, prompt_len={d}); using end-anchored fallback\n", .{ marker.len, prompt_ids.len });
        insert_pos = if (prompt_ids.len > 5) prompt_ids.len - 5 else 0;
    }

    // Insert: BOI + n_tokens × image_token + EOI
    const boi: u32 = config.boi_token_id;
    const eoi: u32 = config.eoi_token_id;
    const has_boi = boi > 0;
    const has_eoi = eoi > 0;
    const extra = n_tokens + (if (has_boi) @as(usize, 1) else 0) + (if (has_eoi) @as(usize, 1) else 0);
    const new_len = prompt_ids.len + extra;
    const result = try allocator.alloc(u32, new_len);

    @memcpy(result[0..insert_pos], prompt_ids[0..insert_pos]);
    var pos = insert_pos;
    if (has_boi) {
        result[pos] = boi;
        pos += 1;
    }
    @memset(result[pos .. pos + n_tokens], image_token_id);
    pos += n_tokens;
    if (has_eoi) {
        result[pos] = eoi;
        pos += 1;
    }
    @memcpy(result[pos..], prompt_ids[insert_pos..]);

    log.info("  Inserted {s}{d} image tokens{s} at position {d} (prompt: {d} -> {d} tokens)\n", .{
        if (has_boi) "BOI + " else "", n_tokens, if (has_eoi) " + EOI" else "", insert_pos, prompt_ids.len, new_len,
    });
    return result;
}

/// Decode a JPEG/PNG image buffer to float32 CHW pixels, resized to target_size.
/// Uses stb_image for decoding, then nearest-neighbor resize + CHW conversion.
/// Decode an image content URL into preprocessed float32 CHW pixels. Supports:
///   data:image/x-mlx-pixels;base64,... (already-preprocessed float32 CHW)
///   data:image/jpeg|png|webp|...;base64,... (decoded + resized via stb_image / libwebp)
/// Returns null on any decode failure (caller treats as missing image).
fn parseImageUrlContent(allocator: std.mem.Allocator, url: []const u8) ?chat_mod.ImageData {
    const sep = std.mem.indexOf(u8, url, ";base64,") orelse return null;
    const b64_data = url[sep + 8 ..];
    const decoded_size = std.base64.standard.Decoder.calcSizeForSlice(b64_data) catch return null;
    const raw_buf = allocator.alloc(u8, decoded_size) catch return null;
    std.base64.standard.Decoder.decode(raw_buf, b64_data) catch {
        allocator.free(raw_buf);
        return null;
    };

    const mime = url[0..sep];
    if (std.mem.eql(u8, mime, "data:image/x-mlx-pixels")) {
        // Already preprocessed float32 CHW pixels
        const n_pixels = raw_buf.len / 4;
        const per_channel = n_pixels / 3;
        const side = std.math.sqrt(per_channel);
        if (side * side == per_channel and n_pixels == 3 * side * side) {
            return .{
                .pixels = raw_buf,
                .width = @intCast(side),
                .height = @intCast(side),
            };
        }
        allocator.free(raw_buf);
        return null;
    }

    if (std.mem.startsWith(u8, mime, "data:image/")) {
        // JPEG/PNG/WebP — decode + resize + convert to float32 CHW
        const img = decodeImageToPixels(allocator, raw_buf);
        allocator.free(raw_buf);
        return img;
    }

    allocator.free(raw_buf);
    return null;
}

fn decodeImageToPixels(allocator: std.mem.Allocator, encoded: []const u8) ?chat_mod.ImageData {
    const target: u32 = 768; // Gemma 4 default for square images

    // Try stb_image first (JPEG/PNG) — request 4 channels to handle transparency
    var w: c_int = 0;
    var h: c_int = 0;
    var channels: c_int = 0;
    const pixels_rgba: ?[*]u8 = stb.stbi_load_from_memory(encoded.ptr, @intCast(encoded.len), &w, &h, &channels, 4);
    var pixels: ?[*]u8 = null;
    var free_fn: enum { stb_free, webp_free, alloc_free } = .stb_free;

    // Composite RGBA onto white background → RGB
    if (pixels_rgba) |rgba| {
        const total_px: usize = @intCast(w * h);
        const rgb_buf = allocator.alloc(u8, total_px * 3) catch {
            stb.stbi_image_free(rgba);
            return null;
        };
        for (0..total_px) |i| {
            const a = @as(u16, rgba[i * 4 + 3]);
            const inv_a = 255 - a;
            rgb_buf[i * 3 + 0] = @intCast((a * @as(u16, rgba[i * 4 + 0]) + inv_a * 255) / 255);
            rgb_buf[i * 3 + 1] = @intCast((a * @as(u16, rgba[i * 4 + 1]) + inv_a * 255) / 255);
            rgb_buf[i * 3 + 2] = @intCast((a * @as(u16, rgba[i * 4 + 2]) + inv_a * 255) / 255);
        }
        stb.stbi_image_free(rgba);
        pixels = rgb_buf.ptr;
        free_fn = .alloc_free;
    }

    // If stb_image failed, try WebP
    if (pixels == null) {
        var webp_w: c_int = 0;
        var webp_h: c_int = 0;
        pixels = webp.WebPDecodeRGB(encoded.ptr, encoded.len, &webp_w, &webp_h);
        if (pixels != null) {
            w = webp_w;
            h = webp_h;
            free_fn = .webp_free;
        }
    }
    if (pixels == null) return null;
    const px = pixels.?;
    defer switch (free_fn) {
        .stb_free => stb.stbi_image_free(px),
        .webp_free => webp.WebPFree(px),
        .alloc_free => {
            const src_total: usize = @intCast(w * h * 3);
            allocator.free(px[0..src_total]);
        },
    };

    const src_w: u32 = @intCast(w);
    const src_h: u32 = @intCast(h);

    // Allocate float32 CHW output: [3, target, target]
    const out_size = 3 * target * target;
    const out_buf = allocator.alloc(u8, out_size * 4) catch return null;
    const float_buf: [*]f32 = @alignCast(@ptrCast(out_buf.ptr));

    // Bilinear resize + HWC→CHW + rescale to [0,1]
    for (0..target) |ty| {
        for (0..target) |tx| {
            // Map target pixel to source coordinates
            const sx_f: f32 = @as(f32, @floatFromInt(tx)) * @as(f32, @floatFromInt(src_w)) / @as(f32, @floatFromInt(target));
            const sy_f: f32 = @as(f32, @floatFromInt(ty)) * @as(f32, @floatFromInt(src_h)) / @as(f32, @floatFromInt(target));

            // Nearest-neighbor for simplicity (bilinear adds complexity for marginal benefit here)
            const sx: u32 = @min(@as(u32, @intFromFloat(sx_f)), src_w - 1);
            const sy: u32 = @min(@as(u32, @intFromFloat(sy_f)), src_h - 1);

            const src_idx = (sy * src_w + sx) * 3;
            const dst_idx = ty * target + tx;

            // CHW: channel 0 (R), channel 1 (G), channel 2 (B)
            float_buf[0 * target * target + dst_idx] = @as(f32, @floatFromInt(px[src_idx + 0])) / 255.0;
            float_buf[1 * target * target + dst_idx] = @as(f32, @floatFromInt(px[src_idx + 1])) / 255.0;
            float_buf[2 * target * target + dst_idx] = @as(f32, @floatFromInt(px[src_idx + 2])) / 255.0;
        }
    }

    log.info("  Decoded {d}x{d} image → {d}x{d} float32 CHW\n", .{ src_w, src_h, target, target });
    return .{ .pixels = out_buf, .width = target, .height = target };
}

// ── Anthropic Messages API ──

fn sendAnthropicError(allocator: std.mem.Allocator, stream: *Conn, err_type: []const u8, message: []const u8, status_code: u32) !void {
    const escaped_msg = try jsonEscape(allocator, message);
    defer allocator.free(escaped_msg);
    const body = try std.fmt.allocPrint(allocator,
        \\{{"type":"error","error":{{"type":"{s}","message":{s}}}}}
    , .{ err_type, escaped_msg });
    defer allocator.free(body);
    var status_buf: [32]u8 = undefined;
    const status = std.fmt.bufPrint(&status_buf, "{d} Error", .{status_code}) catch "500 Error";
    try sendResponse(stream, status, "application/json", body);
}

fn sendAnthropicEvent(stream: *Conn, event_name: []const u8, data: []const u8) !void {
    if (stream.ws_mode) |bridge| {
        // WS transport: emit only the JSON payload as a text frame; the
        // event name lives inside the JSON as `"type": "..."`. (Anthropic
        // events are not currently WS-bridged — only Responses.)
        try bridge.sendText(data);
        return;
    }
    logHttpSseEvent(event_name, data);
    try stream.writeAll("event: ");
    try stream.writeAll(event_name);
    try stream.writeAll("\ndata: ");
    try stream.writeAll(data);
    try stream.writeAll("\n\n");
}

/// Wrap a Responses-API streaming event payload with `sequence_number` (which
/// the OpenAI Responses streaming schema requires on every event), then send.
/// The `payload` is expected to be a JSON object string ending in `}`.
fn sendResponsesEvent(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    event_name: []const u8,
    payload: []const u8,
) !void {
    if (payload.len < 2 or payload[0] != '{' or payload[payload.len - 1] != '}') {
        // Malformed payload — fall back to raw send (defensive; should not happen).
        try sendAnthropicEvent(stream, event_name, payload);
        return;
    }
    var num_buf: [32]u8 = undefined;
    const num_str = try std.fmt.bufPrint(&num_buf, "{d}", .{seq.*});
    seq.* += 1;
    // Detect whether the object has any existing fields (decides leading comma).
    var has_fields = false;
    for (payload[1 .. payload.len - 1]) |c| {
        if (!std.ascii.isWhitespace(c)) {
            has_fields = true;
            break;
        }
    }
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);
    try buf.appendSlice(allocator, payload[0 .. payload.len - 1]);
    if (has_fields) try buf.append(allocator, ',');
    try buf.appendSlice(allocator, "\"sequence_number\":");
    try buf.appendSlice(allocator, num_str);
    try buf.append(allocator, '}');
    try sendAnthropicEvent(stream, event_name, buf.items);
}

fn mapFinishToStopReason(finish_reason: []const u8) []const u8 {
    if (std.mem.eql(u8, finish_reason, "stop")) return "end_turn";
    if (std.mem.eql(u8, finish_reason, "length")) return "max_tokens";
    if (std.mem.eql(u8, finish_reason, "tool_calls")) return "tool_use";
    return "end_turn";
}

/// Serialize a std.json.Value to JSON text, appending to buf.
fn serializeJsonValue(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), value: std.json.Value) !void {
    switch (value) {
        .null => try buf.appendSlice(allocator, "null"),
        .bool => |b| try buf.appendSlice(allocator, if (b) "true" else "false"),
        .integer => |i| {
            var num_buf: [24]u8 = undefined;
            const s = std.fmt.bufPrint(&num_buf, "{d}", .{i}) catch "0";
            try buf.appendSlice(allocator, s);
        },
        .float => |f| {
            var num_buf: [32]u8 = undefined;
            const s = std.fmt.bufPrint(&num_buf, "{d}", .{f}) catch "0";
            try buf.appendSlice(allocator, s);
        },
        .string => |s| {
            const escaped = try jsonEscape(allocator, s);
            defer allocator.free(escaped);
            try buf.appendSlice(allocator, escaped);
        },
        .array => |arr| {
            try buf.append(allocator, '[');
            for (arr.items, 0..) |item, idx| {
                if (idx > 0) try buf.append(allocator, ',');
                try serializeJsonValue(allocator, buf, item);
            }
            try buf.append(allocator, ']');
        },
        .object => |obj| {
            try buf.append(allocator, '{');
            var iter = obj.iterator();
            var first = true;
            while (iter.next()) |entry| {
                if (!first) try buf.append(allocator, ',');
                first = false;
                const ek = try jsonEscape(allocator, entry.key_ptr.*);
                defer allocator.free(ek);
                try buf.appendSlice(allocator, ek);
                try buf.append(allocator, ':');
                try serializeJsonValue(allocator, buf, entry.value_ptr.*);
            }
            try buf.append(allocator, '}');
        },
        .number_string => |s| try buf.appendSlice(allocator, s),
    }
}

/// Convert Anthropic tools format to OpenAI tools format for chat template compatibility.
fn buildOpenAIToolsJson(allocator: std.mem.Allocator, tools_array: std.json.Array) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    try buf.append(allocator, '[');
    for (tools_array.items, 0..) |tool_val, i| {
        if (i > 0) try buf.append(allocator, ',');
        if (tool_val != .object) continue;
        const tool = tool_val.object;
        const name = if (tool.get("name")) |v| (if (v == .string) v.string else "") else "";
        const desc = if (tool.get("description")) |v| (if (v == .string) v.string else "") else "";
        const esc_n = try jsonEscape(allocator, name);
        defer allocator.free(esc_n);
        const esc_d = try jsonEscape(allocator, desc);
        defer allocator.free(esc_d);
        try buf.appendSlice(allocator, "{\"type\":\"function\",\"function\":{\"name\":");
        try buf.appendSlice(allocator, esc_n);
        try buf.appendSlice(allocator, ",\"description\":");
        try buf.appendSlice(allocator, esc_d);
        try buf.appendSlice(allocator, ",\"parameters\":");
        if (tool.get("input_schema")) |schema_val| {
            try serializeJsonValue(allocator, &buf, schema_val);
        } else {
            try buf.appendSlice(allocator, "{}");
        }
        try buf.appendSlice(allocator, "}}");
    }
    try buf.append(allocator, ']');
    return try buf.toOwnedSlice(allocator);
}

fn handleAnthropicMessages(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    xfm: *Transformer,
    tok: *const Tokenizer,
    chat_config: *const chat_mod.ChatConfig,
    config: *const model_mod.ModelConfig,
) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST /v1/messages -> 400 (invalid JSON)\n", .{});
        try sendAnthropicError(allocator, stream, "invalid_request_error", "Invalid JSON in request body", 400);
        return;
    };
    defer parsed.deinit();
    const root = parsed.value.object;

    // max_tokens is required in Anthropic API
    const max_tokens: u32 = if (root.get("max_tokens")) |v| switch (v) {
        .integer => |i| @intCast(i),
        else => 0,
    } else 0;
    if (max_tokens == 0) {
        try sendAnthropicError(allocator, stream, "invalid_request_error", "'max_tokens' is required and must be > 0", 400);
        return;
    }

    // Parse messages array (required)
    const messages_val = root.get("messages") orelse {
        try sendAnthropicError(allocator, stream, "invalid_request_error", "'messages' is required", 400);
        return;
    };
    if (messages_val != .array) {
        try sendAnthropicError(allocator, stream, "invalid_request_error", "'messages' must be an array", 400);
        return;
    }

    var messages = std.ArrayList(chat_mod.Message).empty;
    defer messages.deinit(allocator);

    // Track allocations for serialized tool arguments (need to outlive generation)
    var arg_allocs = std.ArrayList([]const u8).empty;
    defer {
        for (arg_allocs.items) |a| allocator.free(a);
        arg_allocs.deinit(allocator);
    }
    var tool_call_lists = std.ArrayList([]const chat_mod.ToolCall).empty;
    defer {
        for (tool_call_lists.items) |tcs| allocator.free(tcs);
        tool_call_lists.deinit(allocator);
    }
    // Track allocated content strings (concatenated text)
    var content_allocs = std.ArrayList([]const u8).empty;
    defer {
        for (content_allocs.items) |s| allocator.free(s);
        content_allocs.deinit(allocator);
    }

    // System prompt (Anthropic puts it at top level)
    if (root.get("system")) |sys_val| {
        const sys_text: []const u8 = switch (sys_val) {
            .string => |s| s,
            .array => |arr| blk: {
                for (arr.items) |block| {
                    if (block != .object) continue;
                    const btype = block.object.get("type") orelse continue;
                    if (btype != .string or !std.mem.eql(u8, btype.string, "text")) continue;
                    const text = block.object.get("text") orelse continue;
                    if (text == .string) break :blk text.string;
                }
                break :blk "";
            },
            else => "",
        };
        if (sys_text.len > 0) {
            try messages.append(allocator, .{ .role = "system", .content = sys_text, .tool_calls = null, .tool_call_id = null });
        }
    }

    // Convert Anthropic messages to internal format
    for (messages_val.array.items) |msg_val| {
        if (msg_val != .object) continue;
        const msg_obj = msg_val.object;
        const role_val = msg_obj.get("role") orelse continue;
        if (role_val != .string) continue;
        const role = role_val.string;
        const content_val = msg_obj.get("content");

        if (std.mem.eql(u8, role, "user")) {
            if (content_val) |cv| switch (cv) {
                .string => |s| {
                    try messages.append(allocator, .{ .role = "user", .content = s, .tool_calls = null, .tool_call_id = null });
                },
                .array => |arr| {
                    // Process tool_result blocks first, then text+image blocks.
                    for (arr.items) |block| {
                        if (block != .object) continue;
                        const btype = if (block.object.get("type")) |t| (if (t == .string) t.string else "") else "";
                        if (!std.mem.eql(u8, btype, "tool_result")) continue;

                        const tool_use_id = if (block.object.get("tool_use_id")) |v| (if (v == .string) v.string else "") else "";
                        var result_text: []const u8 = "";
                        if (block.object.get("content")) |rc| switch (rc) {
                            .string => |s| result_text = s,
                            .array => |result_arr| {
                                for (result_arr.items) |rb| {
                                    if (rb != .object) continue;
                                    const rtype = if (rb.object.get("type")) |t| (if (t == .string) t.string else "") else "";
                                    if (std.mem.eql(u8, rtype, "text")) {
                                        if (rb.object.get("text")) |t| {
                                            if (t == .string) { result_text = t.string; break; }
                                        }
                                    }
                                }
                            },
                            else => {},
                        };
                        try messages.append(allocator, .{ .role = "tool", .content = result_text, .tool_calls = null, .tool_call_id = tool_use_id });
                    }

                    // Collect text + image blocks into a single user message so
                    // the vision encoder sees them attached to the right turn.
                    var msg_text = std.ArrayList(u8).empty;
                    defer msg_text.deinit(allocator);
                    var image_list = std.ArrayList(chat_mod.ImageData).empty;
                    errdefer {
                        for (image_list.items) |img| allocator.free(img.pixels);
                        image_list.deinit(allocator);
                    }
                    for (arr.items) |block| {
                        if (block != .object) continue;
                        const btype = if (block.object.get("type")) |t| (if (t == .string) t.string else "") else "";
                        if (std.mem.eql(u8, btype, "text")) {
                            const text = if (block.object.get("text")) |t| (if (t == .string) t.string else "") else "";
                            if (text.len > 0) {
                                if (msg_text.items.len > 0) try msg_text.append(allocator, '\n');
                                try msg_text.appendSlice(allocator, text);
                            }
                        } else if (std.mem.eql(u8, btype, "image")) {
                            // Anthropic image block: source = {type:"base64", media_type, data}
                            //                    or = {type:"url", url}
                            const src_val = block.object.get("source") orelse continue;
                            if (src_val != .object) continue;
                            const stype = if (src_val.object.get("type")) |t| (if (t == .string) t.string else "") else "";
                            const data_url = blk: {
                                if (std.mem.eql(u8, stype, "base64")) {
                                    const media = if (src_val.object.get("media_type")) |v| (if (v == .string) v.string else "image/png") else "image/png";
                                    const data = if (src_val.object.get("data")) |v| (if (v == .string) v.string else "") else "";
                                    if (data.len == 0) break :blk @as(?[]const u8, null);
                                    break :blk @as(?[]const u8, try std.fmt.allocPrint(allocator, "data:{s};base64,{s}", .{ media, data }));
                                } else if (std.mem.eql(u8, stype, "url")) {
                                    const url = if (src_val.object.get("url")) |v| (if (v == .string) v.string else "") else "";
                                    if (url.len == 0) break :blk @as(?[]const u8, null);
                                    // Pass through as-is (parseImageUrlContent handles data URLs).
                                    break :blk @as(?[]const u8, try allocator.dupe(u8, url));
                                }
                                break :blk @as(?[]const u8, null);
                            };
                            if (data_url) |du| {
                                defer allocator.free(du);
                                if (parseImageUrlContent(allocator, du)) |img| {
                                    image_list.append(allocator, img) catch {
                                        allocator.free(img.pixels);
                                        continue;
                                    };
                                }
                            }
                        }
                    }
                    if (msg_text.items.len > 0 or image_list.items.len > 0) {
                        const owned_text = if (msg_text.items.len > 0) blk: {
                            const s = try allocator.dupe(u8, msg_text.items);
                            try content_allocs.append(allocator, s);
                            break :blk s;
                        } else "";
                        const owned_images: ?[]const chat_mod.ImageData = if (image_list.items.len > 0) blk: {
                            break :blk try image_list.toOwnedSlice(allocator);
                        } else null;
                        try messages.append(allocator, .{
                            .role = "user",
                            .content = owned_text,
                            .tool_calls = null,
                            .tool_call_id = null,
                            .images = owned_images,
                        });
                    } else {
                        image_list.deinit(allocator);
                    }
                },
                else => {},
            };
        } else if (std.mem.eql(u8, role, "assistant")) {
            if (content_val) |cv| switch (cv) {
                .string => |s| {
                    try messages.append(allocator, .{ .role = "assistant", .content = s, .tool_calls = null, .tool_call_id = null });
                },
                .array => |arr| {
                    // Extract text and tool_use blocks
                    var text_content = std.ArrayList(u8).empty;
                    defer text_content.deinit(allocator);
                    var tcs = std.ArrayList(chat_mod.ToolCall).empty;

                    for (arr.items) |block| {
                        if (block != .object) continue;
                        const btype = if (block.object.get("type")) |t| (if (t == .string) t.string else "") else "";

                        if (std.mem.eql(u8, btype, "text")) {
                            const text = if (block.object.get("text")) |t| (if (t == .string) t.string else "") else "";
                            try text_content.appendSlice(allocator, text);
                        } else if (std.mem.eql(u8, btype, "tool_use")) {
                            const tc_id = if (block.object.get("id")) |v| (if (v == .string) v.string else "") else "";
                            const tc_name = if (block.object.get("name")) |v| (if (v == .string) v.string else "") else "";
                            // Serialize input object to JSON string
                            var args_buf = std.ArrayList(u8).empty;
                            if (block.object.get("input")) |input_val| {
                                try serializeJsonValue(allocator, &args_buf, input_val);
                            } else {
                                try args_buf.appendSlice(allocator, "{}");
                            }
                            const args_str = try args_buf.toOwnedSlice(allocator);
                            try arg_allocs.append(allocator, args_str);
                            try tcs.append(allocator, .{ .id = tc_id, .name = tc_name, .arguments = args_str });
                        }
                        // Skip "thinking" and "redacted_thinking" — model generates its own
                    }

                    var msg_tool_calls: ?[]const chat_mod.ToolCall = null;
                    if (tcs.items.len > 0) {
                        const owned = try tcs.toOwnedSlice(allocator);
                        try tool_call_lists.append(allocator, owned);
                        msg_tool_calls = owned;
                    } else {
                        tcs.deinit(allocator);
                    }

                    const content: []const u8 = if (text_content.items.len > 0) blk: {
                        const duped = try allocator.dupe(u8, text_content.items);
                        try content_allocs.append(allocator, duped);
                        break :blk duped;
                    } else "";
                    try messages.append(allocator, .{ .role = "assistant", .content = content, .tool_calls = msg_tool_calls, .tool_call_id = null });
                },
                else => {},
            };
        }
    }

    if (messages.items.len == 0) {
        try sendAnthropicError(allocator, stream, "invalid_request_error", "No valid messages found in request", 400);
        return;
    }

    // Sampling parameters
    const temperature = parseJsonFloat(root, "temperature", 1.0, 0.0, 2.0);
    const top_p = parseJsonFloat(root, "top_p", 1.0, 0.0, 1.0);
    const top_k: u32 = if (root.get("top_k")) |v| switch (v) {
        .integer => |i| if (i > 0) @intCast(@min(i, 1000)) else 0,
        else => 0,
    } else 0;
    const seed: ?u64 = if (root.get("seed")) |v| switch (v) {
        .integer => |i| @intCast(i),
        else => null,
    } else null;

    // Tools
    var tools_json: ?[]const u8 = null;
    var tools_json_allocated = false;
    defer if (tools_json_allocated) allocator.free(tools_json.?);
    var has_tools = false;
    var tool_choice_instruction: ?[]const u8 = null;
    var tool_choice_allocated = false;
    defer if (tool_choice_allocated) {
        if (tool_choice_instruction) |tci| allocator.free(tci);
    };

    if (root.get("tools")) |tools_val| {
        if (tools_val == .array and tools_val.array.items.len > 0) {
            has_tools = true;
            // Convert Anthropic tools to OpenAI format for chat template
            tools_json = try buildOpenAIToolsJson(allocator, tools_val.array);
            tools_json_allocated = true;

            // Parse tool_choice
            if (root.get("tool_choice")) |tc| {
                if (tc == .object) {
                    const tc_type = if (tc.object.get("type")) |t| (if (t == .string) t.string else "auto") else "auto";
                    if (std.mem.eql(u8, tc_type, "none")) {
                        has_tools = false;
                    } else if (std.mem.eql(u8, tc_type, "any")) {
                        tool_choice_instruction = "\nYou MUST call one of the available functions. Do not respond with text.";
                    } else if (std.mem.eql(u8, tc_type, "tool")) {
                        if (tc.object.get("name")) |name_val| {
                            if (name_val == .string) {
                                tool_choice_instruction = try std.fmt.allocPrint(allocator,
                                    "\nYou MUST call the function \"{s}\". Do not respond with text.", .{name_val.string});
                                tool_choice_allocated = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // Stop sequences
    var stop_sequences = std.ArrayList([]const u8).empty;
    defer stop_sequences.deinit(allocator);
    if (root.get("stop_sequences")) |stop_val| {
        if (stop_val == .array) {
            for (stop_val.array.items) |item| {
                if (item == .string) try stop_sequences.append(allocator, item.string);
            }
        }
    }

    // Thinking config
    var enable_thinking = false;
    var reasoning_budget: i32 = default_reasoning_budget;
    if (root.get("thinking")) |think_val| {
        if (think_val == .object) {
            const think_type = if (think_val.object.get("type")) |t| (if (t == .string) t.string else "") else "";
            if (std.mem.eql(u8, think_type, "enabled") or std.mem.eql(u8, think_type, "adaptive")) {
                enable_thinking = true;
            }
            if (think_val.object.get("budget_tokens")) |bt| {
                if (bt == .integer) reasoning_budget = @intCast(bt.integer);
            }
        }
    }

    const is_stream = if (root.get("stream")) |v| v == .bool and v.bool else false;
    const model_name = if (root.get("model")) |v| (if (v == .string) v.string else config.model_type) else config.model_type;

    // Per-request MTP/PLD overrides (mirror chat-completions behavior).
    // Auto-disable rules: MTP needs model-side weights and is off when tools
    // are present; PLD is off with tools / hybrid SSM models.
    var enable_mtp: bool = if (root.get("enable_mtp")) |v|
        (v == .bool and v.bool)
    else
        default_enable_mtp;
    if (enable_mtp and !xfm.config.has_mtp) enable_mtp = false;
    if (enable_mtp and has_tools) enable_mtp = false;

    const pld_explicit_in_json: bool = root.get("enable_pld") != null;
    var enable_pld: bool = if (root.get("enable_pld")) |v|
        (v == .bool and v.bool)
    else
        default_enable_pld;
    if (enable_pld and enable_mtp) enable_pld = false;
    if (enable_pld and has_tools) enable_pld = false;
    if (enable_pld and xfm.config.has_hybrid_layers) enable_pld = false;

    // Drafter: same disable rules as chat-completions parse site.
    const drafter_explicit_in_json: bool = root.get("enable_drafter") != null;
    var enable_drafter: bool = if (root.get("enable_drafter")) |v|
        (v == .bool and v.bool)
    else
        default_enable_drafter;
    if (enable_drafter and default_drafter == null) enable_drafter = false;
    if (enable_drafter and has_tools) enable_drafter = false;
    if (enable_drafter and xfm.config.has_hybrid_layers) enable_drafter = false;

    // Log request
    const last_msg = messages.items[messages.items.len - 1];
    const preview_len = @min(last_msg.content.len, 80);
    var tool_msg_count: usize = 0;
    for (messages.items) |msg| {
        if (std.mem.eql(u8, msg.role, "tool")) tool_msg_count += 1;
    }
    const tools_len = if (tools_json) |tj| tj.len else 0;
    log.info("POST /v1/messages ({d} msgs, max_tokens={d}, temp={d:.2}, stream={}, thinking={}, tools={d}b, tool_msgs={d})\n", .{
        messages.items.len, max_tokens, temperature, is_stream, enable_thinking, tools_len, tool_msg_count,
    });
    log.info("  > \"{s}{s}\"\n", .{ last_msg.content[0..preview_len], if (last_msg.content.len > 80) "..." else "" });

    // Format chat template
    var prompt_ids_raw = try chat_mod.formatChat(allocator, tok, messages.items, chat_config, if (has_tools) tools_json else null, tool_choice_instruction, enable_thinking);

    // Vision encoder: encode any images on the last user message and splice
    // image tokens into the prompt at the model's configured image_token_id.
    if (global_vision_encoder) |ve| {
        processVisionImages(allocator, ve, xfm, messages.items) catch |err| {
            log.warn("Vision encoding failed: {}\n", .{err});
        };
        if (xfm.vision_embeddings != null) {
            const ve_shape = mlx.getShape(xfm.vision_embeddings.?);
            const n_img_tokens: usize = @intCast(ve_shape[1]);
            const new_ids = try insertImageTokens(allocator, prompt_ids_raw, config.image_token_id, n_img_tokens, config);
            allocator.free(prompt_ids_raw);
            prompt_ids_raw = new_ids;
        }
    }
    const prompt_ids = prompt_ids_raw;
    defer allocator.free(prompt_ids);
    defer {
        if (xfm.vision_embeddings) |ve| _ = mlx.mlx_array_free(ve);
        xfm.vision_embeddings = null;
    }

    // Adaptive spec-decode gate (Anthropic path; mirrors chat-completions).
    if ((enable_pld and !pld_explicit_in_json) or (enable_drafter and !drafter_explicit_in_json)) {
        const score = pld_index.ngramRepeatScore(allocator, prompt_ids, 3) catch 1.0;
        if (score < spec_gate_threshold) {
            if (enable_pld and !pld_explicit_in_json) {
                log.info("  pld=disabled (ngram-score={d:.3} < gate threshold {d:.3})\n", .{ score, spec_gate_threshold });
                enable_pld = false;
            }
            if (enable_drafter and !drafter_explicit_in_json) {
                log.info("  drafter=disabled (ngram-score={d:.3} < gate threshold {d:.3})\n", .{ score, spec_gate_threshold });
                enable_drafter = false;
            }
        }
    }

    // Context size enforcement
    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        log.warn("POST /v1/messages -> 400 (prompt {d} tokens exceeds ctx_size {d})\n", .{ prompt_ids.len, effective_ctx });
        try sendAnthropicError(allocator, stream, "invalid_request_error", "Prompt exceeds maximum context length", 400);
        return;
    }

    // Check if attention computation would exceed GPU memory
    if (!try checkAttentionMemory(allocator, stream, prompt_ids.len, config, true)) return;

    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len);
    log.info("  prompt={d} tokens, max_gen={d}, ctx={d}\n", .{ prompt_ids.len, effective_max_tokens, effective_ctx });

    // KV cache
    const cache_result = try reuseKVCache(allocator, xfm, prompt_ids, has_tools);
    const eos_slice = config.eosTokenSlice();
    const sampling = generate_mod.SamplingParams{
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .repeat_penalty = 1.0,
        .presence_penalty = 0.0,
        .seed = seed,
    };

    if (is_stream) {
        handleAnthropicStreaming(allocator, stream, xfm, tok, cache_result.new_tokens, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, has_tools, cache_result.cached_tokens, enable_thinking, reasoning_budget, @intCast(prompt_ids.len), enable_mtp, enable_pld, enable_drafter) catch |err| {
            log.err("  -> streaming error: {}\n", .{err});
            const err_data = std.fmt.allocPrint(allocator,
                \\{{"type":"error","error":{{"type":"api_error","message":"Internal server error: {s}"}}}}
            , .{@errorName(err)}) catch return;
            defer allocator.free(err_data);
            sendAnthropicEvent(stream, "error", err_data) catch {};
        };
    } else {
        handleAnthropicNonStreaming(allocator, stream, xfm, tok, cache_result.new_tokens, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, has_tools, cache_result.cached_tokens, enable_thinking, reasoning_budget, @intCast(prompt_ids.len), enable_mtp, enable_pld) catch |err| {
            log.err("  -> 500 ({s})\n", .{@errorName(err)});
            sendAnthropicError(allocator, stream, "api_error", @errorName(err), 500) catch {};
        };
    }

    // Update cached prompt
    if (last_generation_was_pad) {
        log.info("  [cache] invalidating cache after pad-only generation\n", .{});
        try xfm.resetCache();
        if (cached_prompt_ids) |old| {
            allocator.free(old);
            cached_prompt_ids = null;
        }
    } else {
        updateCachedPrompt(allocator, prompt_ids, has_tools);
    }
}

fn handleAnthropicNonStreaming(
    allocator: std.mem.Allocator,
    stream: *Conn,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: generate_mod.SamplingParams,
    eos_token_ids: []const u32,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    has_tools: bool,
    cached_tokens: u32,
    enable_thinking: bool,
    reasoning_budget: i32,
    prompt_token_count: u32,
    enable_mtp: bool,
    enable_pld: bool,
) !void {
    var timer = startStopwatch(stream.io);

    // Speculative decoding dispatch — same priority as chat-completions.
    const use_mtp = enable_mtp and sampling.constraint == null and xfm.mtp_layers != null;
    const use_pld = !use_mtp and enable_pld and sampling.constraint == null and !xfm.config.has_hybrid_layers;
    var result = if (use_mtp)
        try generate_mod.generateMtp(stream.io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, getTimeoutNs())
    else if (use_pld)
        try generate_mod.generatePld(stream.io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, getTimeoutNs(), default_pld_draft_len, default_pld_key_len)
    else
        try generate_mod.generate(stream.io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, getTimeoutNs(), 0);
    result.prompt_tokens += cached_tokens;
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);

    // Pad-only detection
    if (result.token_ids.len > 0) {
        var all_pad = true;
        for (result.token_ids) |tid| {
            if (tid != 0) { all_pad = false; break; }
        }
        last_generation_was_pad = all_pad;
    }

    // Apply stop sequences
    var final_text: []const u8 = result.text;
    var finish_reason = result.finish_reason;
    for (stop_sequences) |stop_seq| {
        if (std.mem.indexOf(u8, final_text, stop_seq)) |idx| {
            final_text = final_text[0..idx];
            finish_reason = "stop";
            break;
        }
    }

    const elapsed_ms = timer.read() / std.time.ns_per_ms;

    // Build content blocks array
    var content = std.ArrayList(u8).empty;
    defer content.deinit(allocator);
    try content.append(allocator, '[');

    var block_count: u32 = 0;

    // Thinking block
    if (enable_thinking) {
        const think_split = chat_mod.splitThinkBlock(final_text, true);
        if (think_split.reasoning_content) |reasoning| {
            // Apply budget truncation
            var truncated_reasoning = reasoning;
            var trunc_allocated = false;
            defer if (trunc_allocated) allocator.free(truncated_reasoning);
            if (reasoning_budget >= 0) {
                const r_ids = try tok.encode(allocator, reasoning);
                defer allocator.free(r_ids);
                const budget_usize: usize = @intCast(reasoning_budget);
                if (r_ids.len > budget_usize) {
                    truncated_reasoning = try tok.decode(allocator, r_ids[0..budget_usize], false);
                    trunc_allocated = true;
                }
            }
            const esc_r = try jsonEscape(allocator, truncated_reasoning);
            defer allocator.free(esc_r);
            const thinking_block = try std.fmt.allocPrint(allocator,
                \\{{"type":"thinking","thinking":{s},"signature":"mlx-serve-local"}}
            , .{esc_r});
            defer allocator.free(thinking_block);
            try content.appendSlice(allocator, thinking_block);
            block_count += 1;
            final_text = think_split.content;
        } else {
            final_text = chat_mod.stripThinkBlock(final_text);
        }
    } else {
        final_text = chat_mod.stripThinkBlock(final_text);
    }

    // Check for tool calls
    if (has_tools) {
        if (try chat_mod.parseToolCalls(allocator, final_text)) |tool_calls| {
            defer {
                for (tool_calls) |tc| {
                    allocator.free(tc.name);
                    allocator.free(tc.arguments);
                }
                allocator.free(tool_calls);
            }

            log.info("  <- {d}+{d} tokens ({d}ms) [prefill: {d:.1} tok/s, decode: {d:.1} tok/s] [tool_use: {d}]\n", .{
                result.prompt_tokens, result.completion_tokens, elapsed_ms, result.prefill_tps, result.decode_tps, tool_calls.len,
            });

            // Emit tool_use content blocks
            for (tool_calls, 0..) |tc, i| {
                if (block_count > 0) try content.append(allocator, ',');
                const tc_id = try std.fmt.allocPrint(allocator, "toolu_{d}_{d}", .{ nowMs(stream.io), i });
                defer allocator.free(tc_id);
                const esc_name = try jsonEscape(allocator, tc.name);
                defer allocator.free(esc_name);
                const tc_block = try std.fmt.allocPrint(allocator,
                    \\{{"type":"tool_use","id":"{s}","name":{s},"input":{s}}}
                , .{ tc_id, esc_name, tc.arguments });
                defer allocator.free(tc_block);
                try content.appendSlice(allocator, tc_block);
                block_count += 1;
            }
            finish_reason = "tool_calls";
        } else {
            // No tool calls — emit text block
            if (block_count > 0) try content.append(allocator, ',');
            const esc_text = try jsonEscape(allocator, final_text);
            defer allocator.free(esc_text);
            const text_block = try std.fmt.allocPrint(allocator,
                \\{{"type":"text","text":{s}}}
            , .{esc_text});
            defer allocator.free(text_block);
            try content.appendSlice(allocator, text_block);
        }
    } else {
        // No tools — emit text block
        if (block_count > 0) try content.append(allocator, ',');
        const esc_text = try jsonEscape(allocator, final_text);
        defer allocator.free(esc_text);
        const text_block = try std.fmt.allocPrint(allocator,
            \\{{"type":"text","text":{s}}}
        , .{esc_text});
        defer allocator.free(text_block);
        try content.appendSlice(allocator, text_block);
    }

    try content.append(allocator, ']');

    const stop_reason = mapFinishToStopReason(finish_reason);
    log.info("  <- {d}+{d} tokens ({d}ms) [prefill: {d:.1} tok/s, decode: {d:.1} tok/s] [{s}]\n", .{
        result.prompt_tokens, result.completion_tokens, elapsed_ms, result.prefill_tps, result.decode_tps, stop_reason,
    });

    const response = try std.fmt.allocPrint(allocator,
        \\{{"id":"msg_{d}","type":"message","role":"assistant","content":{s},"model":"{s}","stop_reason":"{s}","stop_sequence":null,"usage":{{"input_tokens":{d},"output_tokens":{d}}}}}
    , .{
        nowMs(stream.io),
        content.items,
        model_name,
        stop_reason,
        prompt_token_count,
        result.completion_tokens,
    });
    defer allocator.free(response);
    try sendResponse(stream, "200 OK", "application/json", response);
}

fn handleAnthropicStreaming(
    allocator: std.mem.Allocator,
    stream: *Conn,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: generate_mod.SamplingParams,
    eos_token_ids: []const u32,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    has_tools: bool,
    cached_tokens: u32,
    enable_thinking: bool,
    reasoning_budget: i32,
    prompt_token_count: u32,
    enable_mtp: bool,
    enable_pld: bool,
    enable_drafter: bool,
) !void {
    var timer = startStopwatch(stream.io);

    // Pick speculative-decoding mode (regular / MTP / PLD / drafter). The
    // token-stream adapter below feeds the per-token Anthropic state machine
    // the same way for all four modes.
    const stream_mode = pickStreamMode(enable_mtp, enable_pld, enable_drafter, default_drafter != null, xfm.mtp_layers != null, xfm.config.has_hybrid_layers, sampling.constraint != null, 0);
    if (stream_mode == .mtp) log.info("  mtp=enabled (streaming)\n", .{});
    if (stream_mode == .pld) log.info("  pld=enabled (streaming, draft_len={d}, key_len={d})\n", .{ default_pld_draft_len, default_pld_key_len });
    if (stream_mode == .drafter) log.info("  drafter=enabled (streaming, block_size={d})\n", .{default_draft_block_size});

    var gen = switch (stream_mode) {
        .mtp => try Generator.initWithOptions(stream.io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{ .mtp_enabled = true }),
        .pld => try Generator.initWithOptions(stream.io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{ .pld_enabled = true }),
        .drafter => try Generator.initWithOptions(stream.io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{
            .drafter_enabled = true,
            .drafter = default_drafter,
            .drafter_block_size = default_draft_block_size,
        }),
        .regular => try Generator.init(stream.io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids),
    };
    gen.timeout_ns = getTimeoutNs();
    defer gen.deinit(allocator);

    var ts = StreamingTokenStream.init(&gen, stream_mode, default_pld_draft_len, default_pld_key_len, eos_token_ids);
    defer ts.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    timer.reset();

    // SSE headers
    const header =
        "HTTP/1.1 200 OK\r\n" ++
        "Content-Type: text/event-stream\r\n" ++
        "Cache-Control: no-cache\r\n" ++
        "Connection: close\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n" ++
        "Access-Control-Allow-Headers: Content-Type, Authorization, x-api-key, anthropic-version\r\n" ++
        "\r\n";
    try stream.writeAll(header);
    logHttpStreamStart("anthropic.messages");

    // message_start
    {
        const data = try std.fmt.allocPrint(allocator,
            \\{{"type":"message_start","message":{{"id":"msg_{d}","type":"message","role":"assistant","content":[],"model":"{s}","stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":{d},"output_tokens":1}}}}}}
        , .{ nowMs(stream.io), model_name, prompt_token_count });
        defer allocator.free(data);
        try sendAnthropicEvent(stream, "message_start", data);
    }
    try sendAnthropicEvent(stream, "ping", "{\"type\":\"ping\"}");

    // State
    var block_index: u32 = 0;
    var text_block_open = false;
    var thinking_block_open = false;
    var in_think_block = enable_thinking;
    var think_buf = std.ArrayList(u8).empty;
    defer think_buf.deinit(allocator);
    var think_close_tag: []const u8 = "</think>";
    var skipped_think_open = false;
    var think_tokens: i32 = 0;
    var budget_exhausted = false;

    var text_buf = std.ArrayList(u8).empty;
    defer text_buf.deinit(allocator);
    var token_texts = std.ArrayList([]const u8).empty;
    defer {
        for (token_texts.items) |t| allocator.free(t);
        token_texts.deinit(allocator);
    }
    var stopped = false;
    var utf8_carry: [3]u8 = undefined;
    var utf8_carry_len: u8 = 0;
    var all_pad = true;
    var gen_token_count: u32 = 0;

    while (try ts.next(allocator)) |token_id| {
        gen_token_count += 1;
        if (token_id != 0) all_pad = false;
        const strip = tok.tok_type == .sentencepiece_bpe;
        const raw_decoded = try tok.decode(allocator, &[_]u32{token_id}, strip and false);

        // UTF-8 carry handling
        const token_text = blk: {
            const with_carry = if (utf8_carry_len > 0) cc: {
                const combined = try allocator.alloc(u8, utf8_carry_len + raw_decoded.len);
                @memcpy(combined[0..utf8_carry_len], utf8_carry[0..utf8_carry_len]);
                @memcpy(combined[utf8_carry_len..], raw_decoded);
                allocator.free(raw_decoded);
                utf8_carry_len = 0;
                break :cc combined;
            } else raw_decoded;
            const tail = utf8TrailingIncomplete(with_carry);
            if (tail > 0) {
                @memcpy(utf8_carry[0..tail], with_carry[with_carry.len - tail ..]);
                utf8_carry_len = @intCast(tail);
            }
            if (with_carry.len == tail) { allocator.free(with_carry); continue; }
            if (tail > 0) {
                const trimmed = try allocator.dupe(u8, with_carry[0 .. with_carry.len - tail]);
                allocator.free(with_carry);
                break :blk trimmed;
            }
            break :blk with_carry;
        };

        if (has_tools or stop_sequences.len > 0) {
            try text_buf.appendSlice(allocator, token_text);
        }

        // Stop sequences
        if (stop_sequences.len > 0) {
            for (stop_sequences) |stop_seq| {
                if (std.mem.indexOf(u8, text_buf.items, stop_seq) != null) {
                    stopped = true;
                    break;
                }
            }
            if (stopped) { allocator.free(token_text); break; }
        }

        if (has_tools) {
            // Buffer for tool detection
            try token_texts.append(allocator, token_text);
            const buf = text_buf.items;
            const maybe_tool = std.mem.indexOf(u8, buf, "<tool_call") != null or
                std.mem.indexOf(u8, buf, "<|tool_call") != null or
                (buf.len > 0 and buf[0] == '{' and std.mem.indexOf(u8, buf, "\"name\"") != null) or
                (buf.len >= 1 and buf[buf.len - 1] == '<');

            if (!maybe_tool) {
                const maybe_thinking = enable_thinking and (
                    std.mem.indexOf(u8, buf, "<|channel>thought") != null or
                    std.mem.indexOf(u8, buf, "<think>") != null or
                    (std.mem.startsWith(u8, buf, "<|channel>") and buf.len < 18) or
                    (std.mem.startsWith(u8, buf, "<think") and buf.len < 7));
                if (!maybe_thinking) {
                    // Flush buffered tokens as text
                    for (token_texts.items) |tt| {
                        if (!text_block_open) {
                            const sd = try std.fmt.allocPrint(allocator,
                                \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
                            , .{block_index});
                            defer allocator.free(sd);
                            try sendAnthropicEvent(stream, "content_block_start", sd);
                            text_block_open = true;
                        }
                        try emitAnthropicTextDelta(allocator, stream, block_index, tt);
                        allocator.free(tt);
                    }
                    token_texts.clearRetainingCapacity();
                }
            }
        } else if (enable_thinking and in_think_block) {
            // Thinking block handling
            defer allocator.free(token_text);
            try think_buf.appendSlice(allocator, token_text);
            think_tokens += 1;

            if (!skipped_think_open and think_buf.items.len >= 7) {
                if (std.mem.startsWith(u8, think_buf.items, "<think>")) {
                    var skip: usize = 7;
                    while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                    if (!thinking_block_open) {
                        const sd = try std.fmt.allocPrint(allocator,
                            \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"thinking","thinking":"","signature":""}}}}
                        , .{block_index});
                        defer allocator.free(sd);
                        try sendAnthropicEvent(stream, "content_block_start", sd);
                        thinking_block_open = true;
                    }
                } else if (think_buf.items.len >= 17 and std.mem.startsWith(u8, think_buf.items, "<|channel>thought")) {
                    think_close_tag = "<channel|>";
                    var skip: usize = 17;
                    while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                    if (!thinking_block_open) {
                        const sd = try std.fmt.allocPrint(allocator,
                            \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"thinking","thinking":"","signature":""}}}}
                        , .{block_index});
                        defer allocator.free(sd);
                        try sendAnthropicEvent(stream, "content_block_start", sd);
                        thinking_block_open = true;
                    }
                } else if (think_buf.items.len < 17 and std.mem.startsWith(u8, "<|channel>thought", think_buf.items)) {
                    // Partial prefix of `<|channel>thought` — wait for more tokens.
                } else {
                    // No opener in the model's output — template injected one.
                    // Stay in the think block; close tag is detected dynamically.
                    skipped_think_open = true;
                    if (!thinking_block_open) {
                        const sd = try std.fmt.allocPrint(allocator,
                            \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"thinking","thinking":"","signature":""}}}}
                        , .{block_index});
                        defer allocator.free(sd);
                        try sendAnthropicEvent(stream, "content_block_start", sd);
                        thinking_block_open = true;
                    }
                }
            }

            // Budget check
            if (!budget_exhausted and reasoning_budget >= 0 and think_tokens >= reasoning_budget and skipped_think_open) {
                budget_exhausted = true;
                if (thinking_block_open and think_buf.items.len > 0) {
                    try emitAnthropicThinkingDelta(allocator, stream, block_index, think_buf.items);
                }
                think_buf.clearRetainingCapacity();
                if (thinking_block_open) {
                    try closeAnthropicThinkingBlock(allocator, stream, block_index);
                    thinking_block_open = false;
                    block_index += 1;
                }
                in_think_block = false;
                continue;
            }

            // Check for the close tag — accept whichever appears first.
            const think_pos = std.mem.indexOf(u8, think_buf.items, "</think>");
            const channel_pos = std.mem.indexOf(u8, think_buf.items, "<channel|>");
            const close_match: ?struct { pos: usize, tag: []const u8 } = blk: {
                if (think_pos == null and channel_pos == null) break :blk null;
                if (think_pos == null) break :blk .{ .pos = channel_pos.?, .tag = "<channel|>" };
                if (channel_pos == null) break :blk .{ .pos = think_pos.?, .tag = "</think>" };
                if (think_pos.? <= channel_pos.?) break :blk .{ .pos = think_pos.?, .tag = "</think>" };
                break :blk .{ .pos = channel_pos.?, .tag = "<channel|>" };
            };

            if (close_match) |m| {
                if (thinking_block_open and m.pos > 0) {
                    try emitAnthropicThinkingDelta(allocator, stream, block_index, think_buf.items[0..m.pos]);
                }
                if (thinking_block_open) {
                    try closeAnthropicThinkingBlock(allocator, stream, block_index);
                    thinking_block_open = false;
                    block_index += 1;
                }
                const after = m.pos + m.tag.len;
                var content_after = std.mem.trimStart(u8, think_buf.items[after..], "\n ");
                if (std.mem.startsWith(u8, content_after, "<|channel>\n")) content_after = content_after[11..];
                if (std.mem.startsWith(u8, content_after, "<|channel>")) content_after = content_after[10..];
                content_after = std.mem.trimStart(u8, content_after, "\n ");
                if (content_after.len > 0) {
                    if (!text_block_open) {
                        const sd = try std.fmt.allocPrint(allocator,
                            \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
                        , .{block_index});
                        defer allocator.free(sd);
                        try sendAnthropicEvent(stream, "content_block_start", sd);
                        text_block_open = true;
                    }
                    try emitAnthropicTextDelta(allocator, stream, block_index, content_after);
                }
                think_buf.clearRetainingCapacity();
                in_think_block = false;
                think_close_tag = m.tag;
            } else if (skipped_think_open and thinking_block_open) {
                // Hold back the longest possible partial-tag suffix (max 9 bytes
                // covers both "</think>" and "<channel|>").
                const max_partial: usize = 9;
                const safe_len = if (think_buf.items.len > max_partial)
                    think_buf.items.len - max_partial
                else
                    0;
                if (safe_len > 0) {
                    try emitAnthropicThinkingDelta(allocator, stream, block_index, think_buf.items[0..safe_len]);
                    const remaining = try allocator.dupe(u8, think_buf.items[safe_len..]);
                    think_buf.clearRetainingCapacity();
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                }
            }
        } else {
            // Regular content token
            defer allocator.free(token_text);
            if (std.mem.eql(u8, token_text, "<|channel>") or std.mem.eql(u8, token_text, "<channel|>")) continue;
            if (!text_block_open) {
                const sd = try std.fmt.allocPrint(allocator,
                    \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
                , .{block_index});
                defer allocator.free(sd);
                try sendAnthropicEvent(stream, "content_block_start", sd);
                text_block_open = true;
            }
            try emitAnthropicTextDelta(allocator, stream, block_index, token_text);
        }
    }

    // Pad-only detection
    last_generation_was_pad = gen_token_count > 0 and all_pad;

    // Flush remaining think buffer
    if (enable_thinking and thinking_block_open and think_buf.items.len > 0) {
        try emitAnthropicThinkingDelta(allocator, stream, block_index, think_buf.items);
    }
    if (thinking_block_open) {
        try closeAnthropicThinkingBlock(allocator, stream, block_index);
        block_index += 1;
    }

    // Post-generation: handle tool calls
    var finish_reason: []const u8 = if (stopped) "stop" else gen.finish_reason;

    if (has_tools) {
        if (try chat_mod.parseToolCalls(allocator, text_buf.items)) |tool_calls| {
            defer {
                for (tool_calls) |tc| { allocator.free(tc.name); allocator.free(tc.arguments); }
                allocator.free(tool_calls);
            }

            // Emit thinking from buffered text if needed
            if (enable_thinking) {
                const think_split = chat_mod.splitThinkBlock(text_buf.items, true);
                if (think_split.reasoning_content) |reasoning| {
                    const sd = try std.fmt.allocPrint(allocator,
                        \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"thinking","thinking":"","signature":""}}}}
                    , .{block_index});
                    defer allocator.free(sd);
                    try sendAnthropicEvent(stream, "content_block_start", sd);
                    try emitAnthropicThinkingDelta(allocator, stream, block_index, reasoning);
                    try closeAnthropicThinkingBlock(allocator, stream, block_index);
                    block_index += 1;
                }
            }

            if (text_block_open) {
                const sd = try std.fmt.allocPrint(allocator, "{{\"type\":\"content_block_stop\",\"index\":{d}}}", .{block_index});
                defer allocator.free(sd);
                try sendAnthropicEvent(stream, "content_block_stop", sd);
                block_index += 1;
                text_block_open = false;
            }

            for (tool_calls, 0..) |tc, i| {
                const tc_id = try std.fmt.allocPrint(allocator, "toolu_{d}_{d}", .{ nowMs(stream.io), i });
                defer allocator.free(tc_id);
                const esc_name = try jsonEscape(allocator, tc.name);
                defer allocator.free(esc_name);

                // content_block_start
                const start = try std.fmt.allocPrint(allocator,
                    \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"tool_use","id":"{s}","name":{s},"input":{{}}}}}}
                , .{ block_index, tc_id, esc_name });
                defer allocator.free(start);
                try sendAnthropicEvent(stream, "content_block_start", start);

                // input_json_delta
                const esc_args_full = try jsonEscape(allocator, tc.arguments);
                defer allocator.free(esc_args_full);
                const args_inner = esc_args_full[1 .. esc_args_full.len - 1];
                const delta = try std.fmt.allocPrint(allocator,
                    \\{{"type":"content_block_delta","index":{d},"delta":{{"type":"input_json_delta","partial_json":"{s}"}}}}
                , .{ block_index, args_inner });
                defer allocator.free(delta);
                try sendAnthropicEvent(stream, "content_block_delta", delta);

                // content_block_stop
                const stop = try std.fmt.allocPrint(allocator, "{{\"type\":\"content_block_stop\",\"index\":{d}}}", .{block_index});
                defer allocator.free(stop);
                try sendAnthropicEvent(stream, "content_block_stop", stop);
                block_index += 1;
            }
            finish_reason = "tool_calls";
        } else {
            // No tool calls — flush buffered tokens
            if (enable_thinking) {
                var full_text = std.ArrayList(u8).empty;
                defer full_text.deinit(allocator);
                for (token_texts.items) |t| try full_text.appendSlice(allocator, t);
                const think_split = chat_mod.splitThinkBlock(full_text.items, true);
                if (think_split.reasoning_content) |reasoning| {
                    const sd = try std.fmt.allocPrint(allocator,
                        \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"thinking","thinking":"","signature":""}}}}
                    , .{block_index});
                    defer allocator.free(sd);
                    try sendAnthropicEvent(stream, "content_block_start", sd);
                    try emitAnthropicThinkingDelta(allocator, stream, block_index, reasoning);
                    try closeAnthropicThinkingBlock(allocator, stream, block_index);
                    block_index += 1;
                }
                if (think_split.content.len > 0) {
                    if (!text_block_open) {
                        const sd2 = try std.fmt.allocPrint(allocator,
                            \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
                        , .{block_index});
                        defer allocator.free(sd2);
                        try sendAnthropicEvent(stream, "content_block_start", sd2);
                        text_block_open = true;
                    }
                    try emitAnthropicTextDelta(allocator, stream, block_index, think_split.content);
                }
            } else {
                for (token_texts.items) |t| {
                    if (!text_block_open) {
                        const sd = try std.fmt.allocPrint(allocator,
                            \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
                        , .{block_index});
                        defer allocator.free(sd);
                        try sendAnthropicEvent(stream, "content_block_start", sd);
                        text_block_open = true;
                    }
                    try emitAnthropicTextDelta(allocator, stream, block_index, t);
                }
            }
        }
    }

    // Close text block if open
    if (text_block_open) {
        const sd = try std.fmt.allocPrint(allocator, "{{\"type\":\"content_block_stop\",\"index\":{d}}}", .{block_index});
        defer allocator.free(sd);
        try sendAnthropicEvent(stream, "content_block_stop", sd);
    }

    // Ensure at least one content block
    if (!text_block_open and block_index == 0) {
        const sd = try std.fmt.allocPrint(allocator,
            \\{{"type":"content_block_start","index":0,"content_block":{{"type":"text","text":""}}}}
        , .{});
        defer allocator.free(sd);
        try sendAnthropicEvent(stream, "content_block_start", sd);
        const sd2 = "{\"type\":\"content_block_stop\",\"index\":0}";
        try sendAnthropicEvent(stream, "content_block_stop", sd2);
    }

    // message_delta
    const stop_reason = mapFinishToStopReason(finish_reason);
    const total_prompt = gen.prompt_tokens + cached_tokens;
    {
        const md = try std.fmt.allocPrint(allocator,
            \\{{"type":"message_delta","delta":{{"stop_reason":"{s}","stop_sequence":null}},"usage":{{"output_tokens":{d}}}}}
        , .{ stop_reason, gen.completion_tokens });
        defer allocator.free(md);
        try sendAnthropicEvent(stream, "message_delta", md);
    }
    try sendAnthropicEvent(stream, "message_stop", "{\"type\":\"message_stop\"}");

    const decode_ns = timer.read();
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(gen.completion_tokens)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    const elapsed_ms = (prefill_ns + decode_ns) / std.time.ns_per_ms;
    log.info("  <- {d}+{d} tokens streamed ({d}ms) [prefill: {d:.1} tok/s, decode: {d:.1} tok/s] [{s}]\n", .{
        total_prompt, gen.completion_tokens, elapsed_ms, prefill_tps, decode_tps, stop_reason,
    });
}

/// Emit a text_delta event for Anthropic streaming.
fn emitAnthropicTextDelta(allocator: std.mem.Allocator, stream: *Conn, index: u32, text: []const u8) !void {
    const esc = try jsonEscape(allocator, text);
    defer allocator.free(esc);
    const inner = esc[1 .. esc.len - 1];
    const data = try std.fmt.allocPrint(allocator,
        \\{{"type":"content_block_delta","index":{d},"delta":{{"type":"text_delta","text":"{s}"}}}}
    , .{ index, inner });
    defer allocator.free(data);
    try sendAnthropicEvent(stream, "content_block_delta", data);
}

/// Emit a thinking_delta event for Anthropic streaming.
fn emitAnthropicThinkingDelta(allocator: std.mem.Allocator, stream: *Conn, index: u32, thinking: []const u8) !void {
    const esc = try jsonEscape(allocator, thinking);
    defer allocator.free(esc);
    const inner = esc[1 .. esc.len - 1];
    const data = try std.fmt.allocPrint(allocator,
        \\{{"type":"content_block_delta","index":{d},"delta":{{"type":"thinking_delta","thinking":"{s}"}}}}
    , .{ index, inner });
    defer allocator.free(data);
    try sendAnthropicEvent(stream, "content_block_delta", data);
}

/// Close a thinking block with a fake signature and content_block_stop.
fn closeAnthropicThinkingBlock(allocator: std.mem.Allocator, stream: *Conn, index: u32) !void {
    const sig = try std.fmt.allocPrint(allocator,
        \\{{"type":"content_block_delta","index":{d},"delta":{{"type":"signature_delta","signature":"mlx-serve-local"}}}}
    , .{index});
    defer allocator.free(sig);
    try sendAnthropicEvent(stream, "content_block_delta", sig);
    const stop = try std.fmt.allocPrint(allocator, "{{\"type\":\"content_block_stop\",\"index\":{d}}}", .{index});
    defer allocator.free(stop);
    try sendAnthropicEvent(stream, "content_block_stop", stop);
}

// ─── /v1/responses (OpenAI Responses API) ────────────────────────────────

fn handleResponsesGet(allocator: std.mem.Allocator, stream: *Conn, id: []const u8) !void {
    const store = getOrInitResponseStore(stream.io, allocator);
    if (store.get(id)) |sr| {
        try sendResponse(stream, "200 OK", "application/json", sr.body_json);
    } else {
        try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "Response not found", 404);
    }
}

fn handleResponsesDelete(allocator: std.mem.Allocator, stream: *Conn, id: []const u8) !void {
    const store = getOrInitResponseStore(stream.io, allocator);
    if (store.delete(id)) {
        const esc_id = try jsonEscape(allocator, id);
        defer allocator.free(esc_id);
        const body = try std.fmt.allocPrint(allocator,
            \\{{"id":{s},"object":"response","deleted":true}}
        , .{esc_id});
        defer allocator.free(body);
        try sendResponse(stream, "200 OK", "application/json", body);
    } else {
        try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "Response not found", 404);
    }
}


fn responsesToolExists(tools_val: ?std.json.Value, name: []const u8) bool {
    const v = tools_val orelse return false;
    if (v != .array) return false;
    for (v.array.items) |tool_val| {
        if (tool_val != .object) continue;
        const tool = tool_val.object;
        const t = if (tool.get("type")) |tv| (if (tv == .string) tv.string else "") else "";
        if (!std.mem.eql(u8, t, "function")) continue;
        const tool_name = if (tool.get("name")) |nv| (if (nv == .string) nv.string else "") else "";
        if (std.mem.eql(u8, tool_name, name)) return true;
    }
    return false;
}

fn buildResponsesJsonInstruction(allocator: std.mem.Allocator, schema_val: ?std.json.Value, tools_active: bool) ![]const u8 {
    var instruction_buf = std.ArrayList(u8).empty;
    defer instruction_buf.deinit(allocator);

    if (tools_active) {
        try instruction_buf.appendSlice(allocator, "If you answer without calling a function, respond with valid JSON only. Do not include markdown or explanation outside the JSON. If a function call is needed, call the function instead; this JSON format applies only to final assistant messages. ");
    } else {
        try instruction_buf.appendSlice(allocator, "Respond with valid JSON only. No other text, no markdown, no explanation. ");
    }

    if (schema_val) |sv| {
        var out: std.Io.Writer.Allocating = .init(allocator);
        defer out.deinit();
        var jws: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
        sv.jsonStringify(&jws) catch {};
        try instruction_buf.appendSlice(allocator, "Your response MUST conform to this JSON schema:\n");
        try instruction_buf.appendSlice(allocator, out.written());
    }

    return try allocator.dupe(u8, instruction_buf.items);
}

fn shouldInjectResponsesJsonInstruction(wants_json: bool, tools_active: bool, tool_choice_instruction: ?[]const u8) bool {
    if (!wants_json) return false;
    if (!tools_active) return true;

    // Required/forced tool calls must keep the prompt focused on tool syntax.
    // Structured-output instructions apply after the tool result is supplied.
    return tool_choice_instruction == null;
}

fn isJsonObjectString(allocator: std.mem.Allocator, text: []const u8) bool {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, text, .{}) catch return false;
    defer parsed.deinit();
    return parsed.value == .object;
}

/// Compact a conversation into a single opaque, round-trippable item.
///
/// The OpenAI Responses spec treats `encrypted_content` as provider-defined.
/// We synthesize a base64-encoded JSON envelope of the resolved messages so
/// the returned `compaction` item can be fed back into `response.create` as
/// an `input` item — exercising the full round-trip without an LLM call.
fn handleResponsesCompact(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST /v1/responses/compact -> 400 (invalid JSON)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", 400);
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
    const root = parsed.value.object;

    // ── model (required) ──
    const model_val = root.get("model");
    const has_model = model_val != null and model_val.? == .string and model_val.?.string.len > 0;
    if (!has_model) {
        try sendErrorResponse(allocator, stream, "422 Unprocessable Entity", "invalid_request_error", "model is required", 422);
        return;
    }

    // ── input (required) ──
    const input_val = root.get("input") orelse {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'input' is a required field", 400);
        return;
    };

    // ── instructions (optional) ──
    const instructions: ?[]const u8 = if (root.get("instructions")) |v|
        (if (v == .string) v.string else null)
    else
        null;

    // ── previous_response_id (optional) ──
    const prev_id: ?[]const u8 = if (root.get("previous_response_id")) |v|
        (if (v == .string) v.string else null)
    else
        null;

    var prev_messages: ?[]const chat_mod.Message = null;
    if (prev_id) |pid| {
        const store = getOrInitResponseStore(stream.io, allocator);
        if (store.get(pid)) |sr| {
            prev_messages = sr.history;
        } else {
            try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "previous_response_id not found", 404);
            return;
        }
    }

    // ── parse → resolved message history ──
    var pi = responses_mod.parseInput(allocator, input_val, instructions, prev_messages, parseImageUrlContent) catch |err| {
        log.warn("POST /v1/responses/compact -> 400 (input parse: {s})\n", .{@errorName(err)});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Failed to parse input", 400);
        return;
    };
    defer pi.deinit();

    // ── synthesize the opaque blob ──
    const blob = try responses_mod.encodeCompactionBlob(allocator, pi.messages.items);
    defer allocator.free(blob);

    // ── mint ids ──
    const resp_id = try responses_mod.makeId(stream.io, allocator, "comp");
    defer allocator.free(resp_id);
    const item_id = try responses_mod.makeId(stream.io, allocator, "cmp");
    defer allocator.free(item_id);

    const esc_resp_id = try jsonEscape(allocator, resp_id);
    defer allocator.free(esc_resp_id);
    const esc_item_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_item_id);
    const esc_blob = try jsonEscape(allocator, blob);
    defer allocator.free(esc_blob);

    const created_ts = nowSecs(stream.io);

    const out = try std.fmt.allocPrint(allocator,
        \\{{"id":{s},"object":"response.compaction","created_at":{d},"output":[{{"type":"compaction","id":{s},"encrypted_content":{s}}}],"usage":{{"input_tokens":0,"output_tokens":0,"total_tokens":0,"input_tokens_details":{{"cached_tokens":0}},"output_tokens_details":{{"reasoning_tokens":0}}}}}}
    , .{ esc_resp_id, created_ts, esc_item_id, esc_blob });
    defer allocator.free(out);

    log.info("POST /v1/responses/compact ({d} msgs -> {d}b blob)\n", .{ pi.messages.items.len, blob.len });
    try sendResponse(stream, "200 OK", "application/json", out);
}

fn handleResponses(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    xfm: *Transformer,
    tok: *const Tokenizer,
    chat_config: *const chat_mod.ChatConfig,
    config: *const model_mod.ModelConfig,
) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST /v1/responses -> 400 (invalid JSON)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", 400);
        return;
    };
    defer parsed.deinit();
    const root = parsed.value.object;

    // ── input (required) ──
    const input_val = root.get("input") orelse {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'input' is a required field", 400);
        return;
    };

    // ── instructions ──
    const instructions: ?[]const u8 = if (root.get("instructions")) |v|
        (if (v == .string) v.string else null)
    else
        null;

    // ── previous_response_id ──
    const prev_id: ?[]const u8 = if (root.get("previous_response_id")) |v|
        (if (v == .string) v.string else null)
    else
        null;

    // ── store (default true) ──
    const should_store = if (root.get("store")) |v| (if (v == .bool) v.bool else true) else true;

    // ── streaming ──
    const is_stream = if (root.get("stream")) |v| (v == .bool and v.bool) else false;

    // ── text.format (or chat-style response_format alias) ──
    var text_format = responses_mod.parseTextFormat(root.get("text"));
    if (text_format.schema_value == null and !std.mem.eql(u8, text_format.kind, "json_schema") and !std.mem.eql(u8, text_format.kind, "json_object")) {
        // Fall back to top-level `response_format` (some clients reuse their
        // chat-completions adapter for /v1/responses).
        const alias = responses_mod.parseResponseFormatAlias(root.get("response_format"));
        if (alias.schema_value != null or std.mem.eql(u8, alias.kind, "json_schema") or std.mem.eql(u8, alias.kind, "json_object")) {
            text_format = alias;
            log.debug("[responses] using top-level response_format as text.format alias\n", .{});
        }
    }
    const grammar_schema_val: ?std.json.Value = text_format.schema_value;
    const wants_json = std.mem.eql(u8, text_format.kind, "json_schema") or std.mem.eql(u8, text_format.kind, "json_object");
    const has_current_tool_output = responses_mod.inputContainsFunctionCallOutput(input_val);

    // ── sampling params ──
    // Track whether the user explicitly supplied max_output_tokens so we can
    // echo `null` (vs. our internal default) in the response envelope.
    const req_max_output_tokens: ?u32 = blk: {
        const v = root.get("max_output_tokens") orelse root.get("max_tokens");
        break :blk if (v) |val| switch (val) {
            .integer => |i| @as(?u32, @intCast(i)),
            else => null,
        } else null;
    };
    const max_tokens: u32 = req_max_output_tokens orelse
        (if (wants_json) DEFAULT_STRUCTURED_OUTPUT_MAX_TOKENS else DEFAULT_API_MAX_TOKENS);
    const temperature = parseJsonFloat(root, "temperature", 1.0, 0.0, 2.0);
    const top_p = parseJsonFloat(root, "top_p", 1.0, 0.0, 1.0);
    const top_k: u32 = if (root.get("top_k")) |v| switch (v) {
        .integer => |i| if (i > 0) @intCast(@min(i, 1000)) else 0,
        else => 0,
    } else 0;
    const frequency_penalty = parseJsonFloat(root, "frequency_penalty", 0.0, 0.0, 2.0);
    const repeat_penalty: f32 = if (frequency_penalty > 0.0) 1.0 + frequency_penalty else 1.0;
    const presence_penalty = parseJsonFloat(root, "presence_penalty", 0.0, 0.0, 2.0);

    // ── echo fields (parsed but not consumed by generation; round-tripped
    // back into the response envelope to satisfy the OpenAI Responses schema) ──
    const top_logprobs_echo: u32 = if (root.get("top_logprobs")) |v| switch (v) {
        .integer => |i| if (i >= 0 and i <= 20) @intCast(i) else 0,
        else => 0,
    } else 0;
    const max_tool_calls_echo: ?u32 = if (root.get("max_tool_calls")) |v| switch (v) {
        .integer => |i| if (i >= 0) @as(?u32, @intCast(i)) else null,
        else => null,
    } else null;
    const truncation_echo: []const u8 = if (root.get("truncation")) |v|
        (if (v == .string and (std.mem.eql(u8, v.string, "auto") or std.mem.eql(u8, v.string, "disabled"))) v.string else "disabled")
    else
        "disabled";
    const parallel_tool_calls_echo: bool = if (root.get("parallel_tool_calls")) |v|
        (if (v == .bool) v.bool else true)
    else
        true;
    const background_echo: bool = if (root.get("background")) |v|
        (if (v == .bool) v.bool else false)
    else
        false;
    const service_tier_echo: []const u8 = if (root.get("service_tier")) |v|
        (if (v == .string) v.string else "default")
    else
        "default";
    const safety_identifier_echo: ?[]const u8 = if (root.get("safety_identifier")) |v|
        (if (v == .string) v.string else null)
    else
        null;
    const prompt_cache_key_echo: ?[]const u8 = if (root.get("prompt_cache_key")) |v|
        (if (v == .string) v.string else null)
    else
        null;
    const seed: ?u64 = if (root.get("seed")) |v| switch (v) {
        .integer => |i| @intCast(i),
        else => null,
    } else null;

    // ── stop sequences ──
    var stop_sequences = std.ArrayList([]const u8).empty;
    defer stop_sequences.deinit(allocator);
    if (root.get("stop")) |sv| switch (sv) {
        .string => |s| try stop_sequences.append(allocator, s),
        .array => |arr| for (arr.items) |it| {
            if (it == .string) try stop_sequences.append(allocator, it.string);
        },
        else => {},
    };

    // ── reasoning ──
    // reasoning.budget is parsed but not enforced post-generation in this MVP
    // pass; thinking truncation happens via finish_reason="length" if the model
    // overruns max_output_tokens.
    const reasoning_cfg = responses_mod.parseReasoning(root.get("reasoning"), default_reasoning_budget);
    const enable_thinking = reasoning_cfg.enable;
    _ = reasoning_cfg.budget;

    // ── tools ──
    var tools_json: ?[]const u8 = null;
    var tools_json_owned = false;
    defer if (tools_json_owned) {
        if (tools_json) |tj| allocator.free(tj);
    };
    var has_tools = root.get("tools") != null;

    const tool_choice = try responses_mod.parseToolChoice(allocator, root.get("tool_choice"));
    defer if (tool_choice.instruction) |ins| allocator.free(ins);
    if (!tool_choice.include_tools) has_tools = false;

    if (has_tools) {
        if (root.get("tools")) |tools_val| if (tools_val == .array) {
            const reshaped = try responses_mod.buildToolsJson(allocator, tools_val.array);
            tools_json = reshaped;
            tools_json_owned = true;
            if (reshaped.len <= 2) has_tools = false; // "[]" — no function tools
        };
    }

    // Once tool outputs are supplied for a structured-output request, this turn
    // must produce the final JSON answer. Keeping tools available lets local
    // models reinterpret schema fields as fake tool calls and loop forever.
    const final_answer_mode = wants_json and has_current_tool_output;
    const active_has_tools = has_tools and !final_answer_mode;
    const active_tools_json: ?[]const u8 = if (active_has_tools) tools_json else null;
    const active_tool_choice_instruction: ?[]const u8 = if (active_has_tools) tool_choice.instruction else null;
    if (final_answer_mode and has_tools) {
        log.info("[responses] final-answer mode - tools disabled after function_call_output\n", .{});
    }

    // ── model name ──
    const model_name = if (root.get("model")) |v|
        (if (v == .string) v.string else config.model_type)
    else
        config.model_type;

    // ── previous response — fetch stored history ──
    var prev_messages: ?[]const chat_mod.Message = null;
    if (prev_id) |pid| {
        const store = getOrInitResponseStore(stream.io, allocator);
        if (store.get(pid)) |sr| {
            prev_messages = sr.history;
        } else {
            try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "previous_response_id not found", 404);
            return;
        }
    }

    // ── parse input → messages ──
    var pi = responses_mod.parseInput(allocator, input_val, instructions, prev_messages, parseImageUrlContent) catch |err| {
        log.warn("POST /v1/responses -> 400 (input parse: {s})\n", .{@errorName(err)});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Failed to parse input", 400);
        return;
    };
    defer pi.deinit();

    if (pi.messages.items.len == 0) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "No valid messages found in 'input'", 400);
        return;
    }

    // ── inject json_schema instruction into system msg (prompt-side belt) ──
    var rf_allocs = std.ArrayList([]const u8).empty;
    defer {
        for (rf_allocs.items) |a| allocator.free(a);
        rf_allocs.deinit(allocator);
    }
    const inject_json_instruction = shouldInjectResponsesJsonInstruction(wants_json, active_has_tools, active_tool_choice_instruction);
    if (inject_json_instruction) {
        const owned = try buildResponsesJsonInstruction(allocator, grammar_schema_val, active_has_tools);
        try rf_allocs.append(allocator, owned);
        if (pi.messages.items.len > 0 and std.mem.eql(u8, pi.messages.items[0].role, "system")) {
            const combined = try std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ pi.messages.items[0].content, owned });
            try rf_allocs.append(allocator, combined);
            pi.messages.items[0].content = combined;
        } else {
            try pi.messages.insert(allocator, 0, .{ .role = "system", .content = owned });
        }
        if (active_has_tools) {
            log.info("[grammar] injected JSON schema prompt while tools are available (mask deferred for tool calls)\n", .{});
        }
    }

    log.info("POST /v1/responses ({d} msgs, max_out={d}, temp={d:.2}, stream={}, thinking={}, prev={?s})\n", .{
        pi.messages.items.len, max_tokens, temperature, is_stream, enable_thinking, prev_id,
    });

    // ── format chat template ──
    var prompt_ids_raw = try chat_mod.formatChat(allocator, tok, pi.messages.items, chat_config, active_tools_json, active_tool_choice_instruction, enable_thinking);

    // ── vision encoder ──
    if (global_vision_encoder) |ve| {
        processVisionImages(allocator, ve, xfm, pi.messages.items) catch |err| {
            log.warn("Vision encoding failed: {}\n", .{err});
        };
        if (xfm.vision_embeddings != null) {
            const ve_shape = mlx.getShape(xfm.vision_embeddings.?);
            const n_img_tokens: usize = @intCast(ve_shape[1]);
            const new_ids = try insertImageTokens(allocator, prompt_ids_raw, config.image_token_id, n_img_tokens, config);
            allocator.free(prompt_ids_raw);
            prompt_ids_raw = new_ids;
        }
    }
    const prompt_ids = prompt_ids_raw;
    defer allocator.free(prompt_ids);
    defer {
        if (xfm.vision_embeddings) |ve| _ = mlx.mlx_array_free(ve);
        xfm.vision_embeddings = null;
    }

    // ── context limit ──
    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Prompt exceeds maximum context length", 400);
        return;
    }
    if (!try checkAttentionMemory(allocator, stream, prompt_ids.len, config, false)) return;
    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len);

    // ── KV cache ──
    const has_images = xfm.vision_embeddings != null;
    if (has_images) {
        if (cached_prompt_ids) |old| {
            allocator.free(old);
            cached_prompt_ids = null;
        }
        try xfm.resetCache();
    }
    const cache_result = try reuseKVCache(allocator, xfm, prompt_ids, active_has_tools);

    // ── sampling ──
    var sampling = generate_mod.SamplingParams{
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .repeat_penalty = repeat_penalty,
        .presence_penalty = presence_penalty,
        .seed = seed,
    };
    var sc: generate_mod.SchemaConstraint = undefined;
    var sc_init = false;
    defer if (sc_init) sc.deinit();
    if (grammar_schema_val) |sv| {
        if (active_has_tools) {
            log.info("[grammar] skipped JSON schema mask while tools are available (tool calls must remain reachable)\n", .{});
        } else {
            const tb = try getOrBuildTokenBytes(allocator, tok);
            if (sc.initFromValue(allocator, sv, tb)) {
                sc_init = true;
                sampling.constraint = &sc.constraint;
                log.info("[grammar] enforcing JSON schema (vocab={d}, mask={d}b)\n", .{ tb.bytes.len, sc.mask_buf.len });
            } else |err| {
                log.warn("[grammar] schema parse failed ({s})\n", .{@errorName(err)});
            }
        }
    }

    // ── pre-allocate response id (used in streaming envelopes too) ──
    const resp_id = try responses_mod.makeId(stream.io, allocator, "resp");
    defer allocator.free(resp_id);
    const esc_resp_id = try jsonEscape(allocator, resp_id);
    defer allocator.free(esc_resp_id);
    const esc_model = try jsonEscape(allocator, model_name);
    defer allocator.free(esc_model);

    // ── pre-render request echoes (live for both the streaming skeleton
    // and the final completed envelope) ──
    const echo_tools_json = try renderResponsesToolsEcho(allocator, root.get("tools"));
    defer allocator.free(echo_tools_json);
    const echo_tool_choice_json = try renderResponsesToolChoiceEcho(allocator, root.get("tool_choice"));
    defer allocator.free(echo_tool_choice_json);
    const echo_text_json = try renderResponsesTextEcho(allocator, root);
    defer allocator.free(echo_text_json);
    const echo_reasoning_json = try renderResponsesReasoningEcho(allocator, root);
    defer allocator.free(echo_reasoning_json);
    const echo_metadata_json = try renderResponsesMetadataEcho(allocator, root);
    defer allocator.free(echo_metadata_json);

    const response_echo = ResponseEcho{
        .tools_json = echo_tools_json,
        .tool_choice_json = echo_tool_choice_json,
        .text_json = echo_text_json,
        .reasoning_json = echo_reasoning_json,
        .metadata_json = echo_metadata_json,
        .instructions = instructions,
        .truncation = truncation_echo,
        .service_tier = service_tier_echo,
        .safety_identifier = safety_identifier_echo,
        .prompt_cache_key = prompt_cache_key_echo,
        .temperature = temperature,
        .top_p = top_p,
        .presence_penalty = presence_penalty,
        .frequency_penalty = frequency_penalty,
        .top_logprobs = top_logprobs_echo,
        .parallel_tool_calls = parallel_tool_calls_echo,
        .background = background_echo,
        .max_output_tokens = req_max_output_tokens,
        .max_tool_calls = max_tool_calls_echo,
    };

    // SSE event sequence counter (required field on every Responses streaming event).
    var seq_num: u64 = 0;

    // ── streaming: send SSE headers + response.created + response.in_progress ──
    if (is_stream) {
        if (stream.ws_mode == null) {
            const sse_headers =
                "HTTP/1.1 200 OK\r\n" ++
                "Content-Type: text/event-stream\r\n" ++
                "Cache-Control: no-cache\r\n" ++
                "Connection: close\r\n" ++
                "Access-Control-Allow-Origin: *\r\n" ++
                "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n" ++
                "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" ++
                "\r\n";
            try stream.writeAll(sse_headers);
            logHttpStreamStart("responses");
        }

        // Skeleton envelope (status:in_progress, output:[])
        const skel = try buildResponsesEnvelope(
            stream.io, allocator, esc_resp_id, esc_model,
            "in_progress", "[]",
            0, 0, 0, 0,
            should_store, prev_id,
            false, false, response_echo,
        );
        defer allocator.free(skel);
        const created_payload = try std.fmt.allocPrint(allocator, "{{\"type\":\"response.created\",\"response\":{s}}}", .{skel});
        defer allocator.free(created_payload);
        try sendResponsesEvent(allocator, stream, &seq_num, "response.created", created_payload);
        const ip_payload = try std.fmt.allocPrint(allocator, "{{\"type\":\"response.in_progress\",\"response\":{s}}}", .{skel});
        defer allocator.free(ip_payload);
        try sendResponsesEvent(allocator, stream, &seq_num, "response.in_progress", ip_payload);
    }

    // ── generate (streaming path: emit deltas live; non-streaming: existing) ──
    const eos_slice = config.eosTokenSlice();

    // Streaming bookkeeping. When we live-stream a reasoning or message item,
    // we record its id + index so the post-loop block emits just the END
    // events instead of full start+delta+end via emit*Events.
    var streamed_reasoning_id: ?[]u8 = null;
    var streamed_reasoning_index: u32 = 0;
    var streamed_reasoning_started = false;
    var streamed_message_id: ?[]u8 = null;
    var streamed_message_index: u32 = 0;
    var streamed_message_started = false;
    defer if (streamed_reasoning_id) |id| allocator.free(id);
    defer if (streamed_message_id) |id| allocator.free(id);

    // Per-request MTP/PLD overrides for the Responses path. Mirror the
    // chat-completions auto-disable logic exactly so the same prompt picks
    // the same path on /v1/chat/completions and /v1/responses.
    var enable_mtp_resp: bool = if (root.get("enable_mtp")) |v|
        (v == .bool and v.bool)
    else
        default_enable_mtp;
    if (enable_mtp_resp and !xfm.config.has_mtp) enable_mtp_resp = false;
    if (enable_mtp_resp and active_has_tools) enable_mtp_resp = false;

    const pld_explicit_in_json: bool = root.get("enable_pld") != null;
    var enable_pld_resp: bool = if (root.get("enable_pld")) |v|
        (v == .bool and v.bool)
    else
        default_enable_pld;
    if (enable_pld_resp and enable_mtp_resp) enable_pld_resp = false;
    if (enable_pld_resp and active_has_tools) enable_pld_resp = false;
    if (enable_pld_resp and xfm.config.has_hybrid_layers) enable_pld_resp = false;

    // Drafter (Responses-side parsing). Same disable rules as the chat
    // and Anthropic parse sites.
    const drafter_explicit_in_json: bool = root.get("enable_drafter") != null;
    var enable_drafter_resp: bool = if (root.get("enable_drafter")) |v|
        (v == .bool and v.bool)
    else
        default_enable_drafter;
    if (enable_drafter_resp and default_drafter == null) enable_drafter_resp = false;
    if (enable_drafter_resp and active_has_tools) enable_drafter_resp = false;
    if (enable_drafter_resp and xfm.config.has_hybrid_layers) enable_drafter_resp = false;

    // Adaptive spec-decode gate (Responses path; mirrors chat-completions and
    // Anthropic). Score the full prompt's 3-gram repetition; novel content
    // (low score) skips PLD/drafter unless the user explicitly opted in.
    if ((enable_pld_resp and !pld_explicit_in_json) or (enable_drafter_resp and !drafter_explicit_in_json)) {
        const score = pld_index.ngramRepeatScore(allocator, prompt_ids, 3) catch 1.0;
        if (score < spec_gate_threshold) {
            if (enable_pld_resp and !pld_explicit_in_json) {
                log.info("  pld=disabled (ngram-score={d:.3} < gate threshold {d:.3})\n", .{ score, spec_gate_threshold });
                enable_pld_resp = false;
            }
            if (enable_drafter_resp and !drafter_explicit_in_json) {
                log.info("  drafter=disabled (ngram-score={d:.3} < gate threshold {d:.3})\n", .{ score, spec_gate_threshold });
                enable_drafter_resp = false;
            }
        }
    }

    var result: generate_mod.GenerationResult = undefined;
    if (is_stream) {
        // Pick speculative-decoding mode for the streaming Responses path.
        const stream_mode = pickStreamMode(enable_mtp_resp, enable_pld_resp, enable_drafter_resp, default_drafter != null, xfm.mtp_layers != null, xfm.config.has_hybrid_layers, sampling.constraint != null, 0);
        if (stream_mode == .mtp) log.info("  mtp=enabled (streaming responses)\n", .{});
        if (stream_mode == .pld) log.info("  pld=enabled (streaming responses, draft_len={d}, key_len={d})\n", .{ default_pld_draft_len, default_pld_key_len });
        if (stream_mode == .drafter) log.info("  drafter=enabled (streaming responses, block_size={d})\n", .{default_draft_block_size});

        var gen = switch (stream_mode) {
            .mtp => try generate_mod.Generator.initWithOptions(stream.io, allocator, xfm, tok, cache_result.new_tokens, effective_max_tokens, sampling, eos_slice, .{ .mtp_enabled = true }),
            .pld => try generate_mod.Generator.initWithOptions(stream.io, allocator, xfm, tok, cache_result.new_tokens, effective_max_tokens, sampling, eos_slice, .{ .pld_enabled = true }),
            .drafter => try generate_mod.Generator.initWithOptions(stream.io, allocator, xfm, tok, cache_result.new_tokens, effective_max_tokens, sampling, eos_slice, .{
                .drafter_enabled = true,
                .drafter = default_drafter,
                .drafter_block_size = default_draft_block_size,
            }),
            .regular => try generate_mod.Generator.init(stream.io, allocator, xfm, tok, cache_result.new_tokens, effective_max_tokens, sampling, eos_slice),
        };
        gen.timeout_ns = getTimeoutNs();
        defer gen.deinit(allocator);

        var ts = StreamingTokenStream.init(&gen, stream_mode, default_pld_draft_len, default_pld_key_len, eos_slice);
        defer ts.deinit(allocator);

        var raw_buf = std.ArrayList(u8).empty;
        defer raw_buf.deinit(allocator);
        var token_ids_buf = std.ArrayList(u32).empty;
        defer token_ids_buf.deinit(allocator);

        var utf8_carry: [3]u8 = undefined;
        var utf8_carry_len: u8 = 0;
        var stopped = false;
        var in_think_block = enable_thinking;
        var think_buf = std.ArrayList(u8).empty;
        defer think_buf.deinit(allocator);
        var skipped_think_open = false;
        var live_output_index: u32 = 0;

        while (try ts.next(allocator)) |token_id| {
            try token_ids_buf.append(allocator, token_id);
            const raw_decoded = try tok.decode(allocator, &[_]u32{token_id}, false);

            // UTF-8 carry across BPE-token boundaries (matches chat-completion).
            const token_text = blk: {
                const with_carry = if (utf8_carry_len > 0) cc: {
                    const combined = try allocator.alloc(u8, utf8_carry_len + raw_decoded.len);
                    @memcpy(combined[0..utf8_carry_len], utf8_carry[0..utf8_carry_len]);
                    @memcpy(combined[utf8_carry_len..], raw_decoded);
                    allocator.free(raw_decoded);
                    utf8_carry_len = 0;
                    break :cc combined;
                } else raw_decoded;
                const tail = utf8TrailingIncomplete(with_carry);
                if (tail > 0) {
                    @memcpy(utf8_carry[0..tail], with_carry[with_carry.len - tail ..]);
                    utf8_carry_len = @intCast(tail);
                }
                if (with_carry.len == tail) {
                    allocator.free(with_carry);
                    continue;
                }
                if (tail > 0) {
                    const trimmed = try allocator.dupe(u8, with_carry[0 .. with_carry.len - tail]);
                    allocator.free(with_carry);
                    break :blk trimmed;
                }
                break :blk with_carry;
            };
            defer allocator.free(token_text);

            try raw_buf.appendSlice(allocator, token_text);

            if (stop_sequences.items.len > 0) {
                for (stop_sequences.items) |stop_seq| {
                    if (std.mem.indexOf(u8, raw_buf.items, stop_seq) != null) {
                        stopped = true;
                        break;
                    }
                }
                if (stopped) break;
            }

            // Tool-active requests buffer entirely — we cannot emit text deltas
            // before knowing whether the output is a tool call.
            if (active_has_tools) continue;

            if (in_think_block) {
                try think_buf.appendSlice(allocator, token_text);

                // Skip a literal think opener if the template did not pre-inject one.
                if (!skipped_think_open and think_buf.items.len >= 7) {
                    if (std.mem.startsWith(u8, think_buf.items, "<think>")) {
                        var skip: usize = 7;
                        while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                        const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                        think_buf.clearAndFree(allocator);
                        try think_buf.appendSlice(allocator, remaining);
                        allocator.free(remaining);
                        skipped_think_open = true;
                    } else if (think_buf.items.len >= 17 and std.mem.startsWith(u8, think_buf.items, "<|channel>thought")) {
                        var skip: usize = 17;
                        while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                        const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                        think_buf.clearAndFree(allocator);
                        try think_buf.appendSlice(allocator, remaining);
                        allocator.free(remaining);
                        skipped_think_open = true;
                    } else if (think_buf.items.len < 17 and std.mem.startsWith(u8, "<|channel>thought", think_buf.items)) {
                        // partial prefix of channel-thought tag; wait for more
                    } else {
                        skipped_think_open = true;
                    }
                }

                // Detect close tag (</think> or <channel|>).
                const tp = std.mem.indexOf(u8, think_buf.items, "</think>");
                const cp = std.mem.indexOf(u8, think_buf.items, "<channel|>");
                const close_match: ?struct { pos: usize, tag: []const u8 } = blk: {
                    if (tp == null and cp == null) break :blk null;
                    if (tp == null) break :blk .{ .pos = cp.?, .tag = "<channel|>" };
                    if (cp == null) break :blk .{ .pos = tp.?, .tag = "</think>" };
                    if (tp.? <= cp.?) break :blk .{ .pos = tp.?, .tag = "</think>" };
                    break :blk .{ .pos = cp.?, .tag = "<channel|>" };
                };

                if (close_match) |m| {
                    const before = think_buf.items[0..m.pos];
                    if (before.len > 0) {
                        if (!streamed_reasoning_started) {
                            streamed_reasoning_id = try responses_mod.makeId(stream.io, allocator, "rs");
                            streamed_reasoning_index = live_output_index;
                            try emitResponsesReasoningStart(allocator, stream, &seq_num, streamed_reasoning_index, streamed_reasoning_id.?);
                            streamed_reasoning_started = true;
                        }
                        try emitResponsesReasoningDelta(allocator, stream, &seq_num, streamed_reasoning_index, streamed_reasoning_id.?, before);
                    }
                    if (streamed_reasoning_started) live_output_index += 1;

                    const after = m.pos + m.tag.len;
                    var content_after = std.mem.trimStart(u8, think_buf.items[after..], "\n ");
                    if (std.mem.startsWith(u8, content_after, "<|channel>\n")) {
                        content_after = content_after[11..];
                    } else if (std.mem.startsWith(u8, content_after, "<|channel>")) {
                        content_after = content_after[10..];
                    }
                    content_after = std.mem.trimStart(u8, content_after, "\n ");
                    if (content_after.len > 0) {
                        if (!streamed_message_started) {
                            streamed_message_id = try responses_mod.makeId(stream.io, allocator, "msg");
                            streamed_message_index = live_output_index;
                            try emitResponsesMessageStart(allocator, stream, &seq_num, streamed_message_index, streamed_message_id.?);
                            streamed_message_started = true;
                        }
                        try emitResponsesMessageDelta(allocator, stream, &seq_num, streamed_message_index, streamed_message_id.?, content_after);
                    }
                    think_buf.clearRetainingCapacity();
                    in_think_block = false;
                } else if (skipped_think_open) {
                    // Hold back the longest possible partial-tag suffix (max 9 bytes
                    // covers both "</think>" and "<channel|>").
                    const max_partial: usize = 9;
                    const safe_len = if (think_buf.items.len > max_partial) think_buf.items.len - max_partial else 0;
                    if (safe_len > 0) {
                        if (!streamed_reasoning_started) {
                            streamed_reasoning_id = try responses_mod.makeId(stream.io, allocator, "rs");
                            streamed_reasoning_index = live_output_index;
                            try emitResponsesReasoningStart(allocator, stream, &seq_num, streamed_reasoning_index, streamed_reasoning_id.?);
                            streamed_reasoning_started = true;
                        }
                        try emitResponsesReasoningDelta(allocator, stream, &seq_num, streamed_reasoning_index, streamed_reasoning_id.?, think_buf.items[0..safe_len]);
                        const remaining = try allocator.dupe(u8, think_buf.items[safe_len..]);
                        think_buf.clearRetainingCapacity();
                        try think_buf.appendSlice(allocator, remaining);
                        allocator.free(remaining);
                    }
                }
            } else {
                // Skip Gemma 4 channel tags that may leak after the thinking block.
                if (std.mem.eql(u8, token_text, "<|channel>") or std.mem.eql(u8, token_text, "<channel|>")) continue;
                if (!streamed_message_started) {
                    streamed_message_id = try responses_mod.makeId(stream.io, allocator, "msg");
                    streamed_message_index = live_output_index;
                    try emitResponsesMessageStart(allocator, stream, &seq_num, streamed_message_index, streamed_message_id.?);
                    streamed_message_started = true;
                }
                try emitResponsesMessageDelta(allocator, stream, &seq_num, streamed_message_index, streamed_message_id.?, token_text);
            }
        }

        // Flush any remaining think buffer (no close tag found) as reasoning.
        if (in_think_block and think_buf.items.len > 0 and !active_has_tools) {
            if (!streamed_reasoning_started) {
                streamed_reasoning_id = try responses_mod.makeId(stream.io, allocator, "rs");
                streamed_reasoning_index = live_output_index;
                try emitResponsesReasoningStart(allocator, stream, &seq_num, streamed_reasoning_index, streamed_reasoning_id.?);
                streamed_reasoning_started = true;
            }
            try emitResponsesReasoningDelta(allocator, stream, &seq_num, streamed_reasoning_index, streamed_reasoning_id.?, think_buf.items);
        }

        result = .{
            .text = try raw_buf.toOwnedSlice(allocator),
            .token_ids = try token_ids_buf.toOwnedSlice(allocator),
            .prompt_tokens = gen.prompt_tokens + cache_result.cached_tokens,
            .completion_tokens = gen.completion_tokens,
            .finish_reason = if (stopped) "stop" else gen.finish_reason,
            .prefill_tps = 0.0,
            .decode_tps = 0.0,
        };
    } else {
        // Non-streaming Responses: dispatch to MTP / PLD when applicable so
        // /v1/responses gets the same speedup as /v1/chat/completions.
        const use_mtp = enable_mtp_resp and sampling.constraint == null and xfm.mtp_layers != null;
        const use_pld = !use_mtp and enable_pld_resp and sampling.constraint == null and !xfm.config.has_hybrid_layers;
        result = if (use_mtp)
            try generate_mod.generateMtp(stream.io, allocator, xfm, tok, cache_result.new_tokens, effective_max_tokens, sampling, eos_slice, getTimeoutNs())
        else if (use_pld)
            try generate_mod.generatePld(stream.io, allocator, xfm, tok, cache_result.new_tokens, effective_max_tokens, sampling, eos_slice, getTimeoutNs(), default_pld_draft_len, default_pld_key_len)
        else
            try generate_mod.generate(stream.io, allocator, xfm, tok, cache_result.new_tokens, effective_max_tokens, sampling, eos_slice, getTimeoutNs(), 0);
        result.prompt_tokens += cache_result.cached_tokens;
    }
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);

    // pad-only detection (KV cache hygiene)
    if (result.token_ids.len > 0) {
        var all_pad = true;
        for (result.token_ids) |tid| {
            if (tid != 0) { all_pad = false; break; }
        }
        last_generation_was_pad = all_pad;
    }

    // ── apply stop sequences ──
    var final_text: []const u8 = result.text;
    var finish_reason = result.finish_reason;
    for (stop_sequences.items) |stop_seq| {
        if (std.mem.indexOf(u8, final_text, stop_seq)) |idx| {
            final_text = final_text[0..idx];
            finish_reason = "stop";
            break;
        }
    }

    const status_str: []const u8 = if (std.mem.eql(u8, finish_reason, "length")) "incomplete" else "completed";

    // ── split thinking & tool calls ──
    const think_split = chat_mod.splitThinkBlock(final_text, enable_thinking);
    const reasoning_text: ?[]const u8 = if (enable_thinking) think_split.reasoning_content else null;
    const visible_text: []const u8 = if (enable_thinking) think_split.content else chat_mod.stripThinkBlock(final_text);

    var tool_calls: ?[]chat_mod.ParsedToolCall = null;
    if (active_has_tools) {
        tool_calls = try chat_mod.parseToolCalls(allocator, final_text);
    }
    defer if (tool_calls) |tcs| {
        for (tcs) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(tcs);
    };

    // ── build output[] array (shared between streaming + non-streaming) ──
    var out_buf = std.ArrayList(u8).empty;
    defer out_buf.deinit(allocator);
    try out_buf.append(allocator, '[');
    var emitted: usize = 0;
    var output_index: u32 = 0;
    var emitted_tool_calls = std.ArrayList(chat_mod.ToolCall).empty;
    defer {
        for (emitted_tool_calls.items) |tc| allocator.free(tc.id);
        emitted_tool_calls.deinit(allocator);
    }

    if (reasoning_text) |rt| if (rt.len > 0) {
        if (emitted > 0) try out_buf.append(allocator, ',');
        if (is_stream and streamed_reasoning_started) {
            // Live deltas already streamed; emit just the closing events with
            // the canonical reasoning text from splitThinkBlock.
            try responses_mod.appendReasoningItem(allocator, &out_buf, streamed_reasoning_id.?, rt);
            try emitResponsesReasoningEnd(allocator, stream, &seq_num, streamed_reasoning_index, streamed_reasoning_id.?, rt);
        } else {
            const rid = try responses_mod.makeId(stream.io, allocator, "rs");
            defer allocator.free(rid);
            try responses_mod.appendReasoningItem(allocator, &out_buf, rid, rt);
            if (is_stream) {
                try emitResponsesReasoningEvents(allocator, stream, &seq_num, output_index, rid, rt);
            }
        }
        emitted += 1;
        output_index += 1;
    };

    if (tool_calls) |tcs| if (tcs.len > 0) {
        for (tcs) |tc| {
            if (!responsesToolExists(root.get("tools"), tc.name)) {
                log.warn("[responses] dropping undeclared tool call: {s}\n", .{tc.name});
                continue;
            }
            if (!isJsonObjectString(allocator, tc.arguments)) {
                log.warn("[responses] dropping tool call with non-object arguments: {s}\n", .{tc.name});
                continue;
            }
            const fc_id = try responses_mod.makeId(stream.io, allocator, "fc");
            defer allocator.free(fc_id);
            const call_id = try responses_mod.makeId(stream.io, allocator, "call");
            defer allocator.free(call_id);
            const stored_call_id = try allocator.dupe(u8, call_id);
            emitted_tool_calls.append(allocator, .{ .id = stored_call_id, .name = tc.name, .arguments = tc.arguments }) catch |err| {
                allocator.free(stored_call_id);
                return err;
            };
            if (emitted > 0) try out_buf.append(allocator, ',');
            try responses_mod.appendFunctionCallItem(allocator, &out_buf, fc_id, call_id, tc.name, tc.arguments);
            emitted += 1;
            if (is_stream) {
                try emitResponsesFunctionCallEvents(allocator, stream, &seq_num, output_index, fc_id, call_id, tc.name, tc.arguments);
            }
            output_index += 1;
        }
    };

    const has_tool_calls = emitted_tool_calls.items.len > 0;
    if (!has_tool_calls) {
        if (emitted > 0) try out_buf.append(allocator, ',');
        if (is_stream and streamed_message_started) {
            // Live deltas already streamed; emit just the closing events.
            try responses_mod.appendOutputTextMessage(allocator, &out_buf, streamed_message_id.?, visible_text);
            try emitResponsesMessageEnd(allocator, stream, &seq_num, streamed_message_index, streamed_message_id.?, visible_text);
        } else {
            const mid = try responses_mod.makeId(stream.io, allocator, "msg");
            defer allocator.free(mid);
            try responses_mod.appendOutputTextMessage(allocator, &out_buf, mid, visible_text);
            if (is_stream) {
                try emitResponsesMessageEvents(allocator, stream, &seq_num, output_index, mid, visible_text);
            }
        }
        emitted += 1;
        output_index += 1;
    }

    try out_buf.append(allocator, ']');

    // ── envelope ──
    const is_incomplete = std.mem.eql(u8, status_str, "incomplete");
    const is_completed_status = std.mem.eql(u8, status_str, "completed") or is_incomplete;
    const envelope = try buildResponsesEnvelope(
        stream.io,
        allocator,
        esc_resp_id,
        esc_model,
        status_str,
        out_buf.items,
        result.prompt_tokens,
        result.completion_tokens,
        cache_result.cached_tokens,
        0, // reasoning_tokens — not tracked separately yet
        should_store,
        prev_id,
        is_incomplete,
        is_completed_status,
        response_echo,
    );
    defer allocator.free(envelope);

    // ── store response ──
    if (should_store) {
        const stored_tool_calls: ?[]const chat_mod.ToolCall = if (emitted_tool_calls.items.len > 0) emitted_tool_calls.items else null;
        storeResponse(stream.io, allocator, resp_id, model_name, status_str, envelope, pi.messages.items, visible_text, reasoning_text, stored_tool_calls) catch |err| {
            log.warn("[responses] store failed: {s}\n", .{@errorName(err)});
        };
    }

    if (is_stream) {
        const completed_payload = try std.fmt.allocPrint(allocator, "{{\"type\":\"response.completed\",\"response\":{s}}}", .{envelope});
        defer allocator.free(completed_payload);
        try sendResponsesEvent(allocator, stream, &seq_num, "response.completed", completed_payload);
    } else {
        try sendResponse(stream, "200 OK", "application/json", envelope);
    }

    // KV cache update (mirrors chat-completions)
    if (last_generation_was_pad) {
        try xfm.resetCache();
        if (cached_prompt_ids) |old| {
            allocator.free(old);
            cached_prompt_ids = null;
        }
    } else {
        updateCachedPrompt(allocator, prompt_ids, active_has_tools);
    }
}

// ─── WebSocket transport for /v1/responses ────────────────────────────────

const WsConnT = ws_mod.WsConn(Conn);

/// Connection-local cache for `store: false` continuations on a WS session.
/// Each entry owns its arena (StoredResponse.deinit frees both).
const WsLocalCache = struct {
    map: std.StringHashMapUnmanaged(*responses_mod.StoredResponse) = .{},
    gpa: std.mem.Allocator,

    fn put(self: *WsLocalCache, sr: *responses_mod.StoredResponse) !void {
        if (self.map.fetchRemove(sr.id)) |kv| kv.value.deinit();
        try self.map.put(self.gpa, sr.id, sr);
    }

    fn get(self: *WsLocalCache, id: []const u8) ?*responses_mod.StoredResponse {
        return self.map.get(id);
    }

    fn evict(self: *WsLocalCache, id: []const u8) void {
        if (self.map.fetchRemove(id)) |kv| kv.value.deinit();
    }

    fn deinit(self: *WsLocalCache) void {
        var it = self.map.valueIterator();
        while (it.next()) |sr_ptr| sr_ptr.*.deinit();
        self.map.deinit(self.gpa);
    }
};

/// Returns the call_id of the first `function_call_output` input item that
/// has no matching `function_call` in the prev response's history (or in
/// the same input list, for parallel tool-call replies). Returns null if
/// every output has a matching call.
fn orphanFunctionCallOutputId(
    input_val: std.json.Value,
    prev_id: ?[]const u8,
    local: *WsLocalCache,
    global: *responses_mod.ResponseStore,
) ?[]const u8 {
    if (input_val != .array) return null;
    // Collect all known call_ids from (a) prev history and (b) current input's
    // function_call items.
    var known: std.StringHashMapUnmanaged(void) = .{};
    defer known.deinit(std.heap.page_allocator);
    if (prev_id) |pid| blk: {
        const sr = local.get(pid) orelse global.get(pid) orelse break :blk;
        for (sr.history) |m| {
            if (m.tool_calls) |tcs| for (tcs) |tc| {
                _ = known.put(std.heap.page_allocator, tc.id, {}) catch {};
            };
        }
    }
    for (input_val.array.items) |item| {
        if (item != .object) continue;
        const t_v = item.object.get("type") orelse continue;
        if (t_v != .string or !std.mem.eql(u8, t_v.string, "function_call")) continue;
        const cid_v = item.object.get("call_id") orelse continue;
        if (cid_v == .string) {
            _ = known.put(std.heap.page_allocator, cid_v.string, {}) catch {};
        }
    }
    // Now check each function_call_output.
    for (input_val.array.items) |item| {
        if (item != .object) continue;
        const t_v = item.object.get("type") orelse continue;
        if (t_v != .string or !std.mem.eql(u8, t_v.string, "function_call_output")) continue;
        const cid_v = item.object.get("call_id") orelse return "";
        if (cid_v != .string) return "";
        if (!known.contains(cid_v.string)) return cid_v.string;
    }
    return null;
}

fn wsBridgeSend(impl: *anyopaque, data: []const u8) anyerror!void {
    const ws_conn: *WsConnT = @ptrCast(@alignCast(impl));
    try ws_conn.writeText(data);
}

/// Send a JSON error frame for a single WS turn. The compliance suite's
/// frame handler treats a `{"type":"error"}` text frame as terminal —
/// emitting a trailing `[DONE]` would land in the *next* turn's bucket
/// (same bug as success-path: see `handleResponsesWebSocket`).
fn wsSendErrorTurn(allocator: std.mem.Allocator, ws_conn: *WsConnT, status: u32, code: []const u8, message: []const u8) !void {
    const esc_code = try jsonEscape(allocator, code);
    defer allocator.free(esc_code);
    const esc_msg = try jsonEscape(allocator, message);
    defer allocator.free(esc_msg);
    const err_frame = try std.fmt.allocPrint(
        allocator,
        \\{{"type":"error","status":{d},"error":{{"code":{s},"message":{s}}}}}
    ,
        .{ status, esc_code, esc_msg },
    );
    defer allocator.free(err_frame);
    try ws_conn.writeText(err_frame);
}

/// Drive a WebSocket connection that bridges to /v1/responses.
///
/// Each text frame is a `response.create`-shaped JSON message. We translate
/// it into an HTTP-like body and reuse `handleResponses` (with a
/// `Conn.ws_mode` bridge) so all SSE events become WS text frames. After
/// each turn we emit `[DONE]` and wait for the next message on the same
/// connection, supporting chained `response.create` calls and
/// `previous_response_id` continuations.
fn handleResponsesWebSocket(
    allocator: std.mem.Allocator,
    stream: *Conn,
    headers: []const u8,
    xfm: *Transformer,
    tok: *const Tokenizer,
    chat_config: *const chat_mod.ChatConfig,
    config: *const model_mod.ModelConfig,
) !void {
    ws_mod.handshake(stream, headers) catch |err| {
        log.warn("WS handshake failed: {s}\n", .{@errorName(err)});
        return;
    };
    log.info("WS /v1/responses connected\n", .{});

    var ws_conn = WsConnT.init(stream);
    defer ws_conn.deinit(allocator);

    var local_cache: WsLocalCache = .{ .gpa = allocator };
    defer local_cache.deinit();

    var bridge: WsBridge = .{ .impl = &ws_conn, .sendTextFn = &wsBridgeSend, .allocator = allocator };
    defer bridge.reset();

    // Frame loop — one iteration per inbound WS message.
    while (true) {
        const msg = ws_conn.readMessage(allocator) catch |err| switch (err) {
            error.WsClosed => return,
            error.WsProtocol => {
                ws_conn.writeClose(1002, "protocol error") catch {};
                return;
            },
            error.WsTooLarge => {
                ws_conn.writeClose(1009, "message too large") catch {};
                return;
            },
            else => return,
        };
        defer allocator.free(msg.payload);

        switch (msg.opcode) {
            .close => {
                ws_conn.writeClose(1000, "") catch {};
                return;
            },
            .ping => {
                try ws_conn.writePong(msg.payload);
                continue;
            },
            .pong => continue,
            .text => {},
            else => continue,
        }

        // Parse the request payload — must be {"type":"response.create", ...}
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, msg.payload, .{}) catch {
            try wsSendErrorTurn(allocator, &ws_conn, 400, "invalid_request_error", "Invalid JSON in request body");
            continue;
        };
        defer parsed.deinit();
        if (parsed.value != .object) {
            try wsSendErrorTurn(allocator, &ws_conn, 400, "invalid_request_error", "Request must be a JSON object");
            continue;
        }
        const root = parsed.value.object;
        const type_val = root.get("type");
        if (type_val == null or type_val.? != .string or !std.mem.eql(u8, type_val.?.string, "response.create")) {
            try wsSendErrorTurn(allocator, &ws_conn, 400, "invalid_request_error", "Expected type=response.create");
            continue;
        }
        // The plan disallows stream/stream_options/background here — WS is
        // inherently streaming and stateless (no background jobs).
        if (root.get("stream") != null or root.get("stream_options") != null or root.get("background") != null) {
            try wsSendErrorTurn(allocator, &ws_conn, 400, "invalid_request_error", "stream/stream_options/background are not allowed over WebSocket");
            continue;
        }

        // ── Resolve previous_response_id against per-conn cache first.
        //    For store:false continuations on this connection, the chain
        //    root is in `local_cache` only. For store:true responses we
        //    fall back to the global store (lets stored chains survive
        //    across connections). On miss, evict any stale local entry
        //    before reporting the error.
        const prev_id_owned: ?[]const u8 = if (root.get("previous_response_id")) |v|
            (if (v == .string) v.string else null)
        else
            null;
        var prev_in_local: bool = false;
        if (prev_id_owned) |pid| {
            if (local_cache.get(pid) != null) {
                prev_in_local = true;
            } else {
                const store = getOrInitResponseStore(stream.io, allocator);
                if (store.get(pid) == null) {
                    local_cache.evict(pid);
                    try wsSendErrorTurn(allocator, &ws_conn, 404, "previous_response_not_found", "previous_response_id not found");
                    continue;
                }
            }
        }

        // Validate any function_call_output items reference real call_ids
        // in the prev response's history. An orphan output means the user
        // is trying to feed back a tool result for a call the model never
        // made — the turn must fail and (per compliance) evict the chain
        // root from the local cache.
        if (prev_id_owned) |pid| {
            if (root.get("input")) |input_v| {
                if (orphanFunctionCallOutputId(input_v, prev_id_owned, &local_cache, getOrInitResponseStore(stream.io, allocator))) |_| {
                    local_cache.evict(pid);
                    try wsSendErrorTurn(allocator, &ws_conn, 400, "invalid_request_error", "function_call_output references a missing call_id");
                    continue;
                }
            }
        }

        // For store:false continuations on this connection, the chain root
        // lives in `local_cache` (not the global store). Move it into the
        // global store *just for this turn* so handleResponses' lookup at
        // line 4634 can find it. We restore the original location at end
        // of turn. (The kludge avoids modifying handleResponses.)
        const did_borrow_to_global = prev_id_owned != null and prev_in_local;
        if (did_borrow_to_global) {
            const sr = local_cache.map.get(prev_id_owned.?).?;
            // Remove from local without freeing.
            _ = local_cache.map.remove(prev_id_owned.?);
            const store = getOrInitResponseStore(stream.io, allocator);
            store.put(sr) catch {};
        }

        // ── Reserialize the request as an HTTP body, forcing
        //    `stream: true` (WS is inherently streaming) and `store: true`
        //    so handleResponses always persists the result. We move the
        //    entry to local_cache below if the user actually wanted
        //    `store: false`, achieving connection-scoped lifetime.
        const want_user_store: bool = if (root.get("store")) |v| (if (v == .bool) v.bool else true) else true;
        const body = try buildResponsesBodyFromWsRequest(allocator, root);
        defer allocator.free(body);

        if (!acquireInferenceSlot(stream.io)) {
            try wsSendErrorTurn(allocator, &ws_conn, 503, "server_error", "Server request queue is full. Try again shortly.");
            continue;
        }
        xfm.useCurrentThreadStream();

        // Reset sequence numbering per response per the OpenAI spec.
        // (handleResponses owns its own seq_num, fresh each call.)
        var seq: u64 = 0;
        _ = &seq;

        stream.ws_mode = &bridge;
        defer stream.ws_mode = null;

        bridge.reset();
        handleResponses(allocator, stream, body, xfm, tok, chat_config, config) catch |err| {
            log.warn("WS handleResponses error: {s}\n", .{@errorName(err)});
            // Best-effort error frame; connection may already be torn.
            wsSendErrorTurn(allocator, &ws_conn, 500, "server_error", @errorName(err)) catch {};
            releaseInferenceSlot(stream.io);
            // Restore borrowed prev entry back to local cache on failure.
            if (did_borrow_to_global and prev_id_owned != null) {
                const store = getOrInitResponseStore(stream.io, allocator);
                if (store.map.fetchRemove(prev_id_owned.?)) |kv| {
                    store.lru.remove(&kv.value.list_node);
                    local_cache.map.put(local_cache.gpa, kv.value.id, kv.value) catch {};
                }
            }
            continue;
        };
        releaseInferenceSlot(stream.io);

        // Handle the borrowed prev: move it back from global to local
        // cache. On success, also do the eviction-on-failure check —
        // if the just-completed turn ended in a non-completed status
        // (failed/incomplete), evict the chain root from local cache.
        if (did_borrow_to_global and prev_id_owned != null) {
            const store = getOrInitResponseStore(stream.io, allocator);
            if (store.map.fetchRemove(prev_id_owned.?)) |kv| {
                store.lru.remove(&kv.value.list_node);
                const turn_failed = bridge.captured_status != null and !std.mem.eql(u8, bridge.captured_status.?, "completed");
                if (turn_failed) {
                    // Compliance: a failed continuation evicts the chain root.
                    kv.value.deinit();
                } else {
                    local_cache.map.put(local_cache.gpa, kv.value.id, kv.value) catch {};
                }
            }
        }

        // For store:false on this WS, move the freshly-stored response
        // from the global store into the connection-local cache so it
        // (a) survives the user's `store: false` semantics within this
        // connection (allowing previous_response_id chains) and
        // (b) does NOT leak across reconnects (other WS connections
        // looking up this id should get previous_response_not_found).
        if (!want_user_store) {
            if (bridge.captured_resp_id) |rid| {
                const store = getOrInitResponseStore(stream.io, allocator);
                if (store.map.fetchRemove(rid)) |kv| {
                    store.lru.remove(&kv.value.list_node);
                    local_cache.map.put(local_cache.gpa, kv.value.id, kv.value) catch {};
                }
            }
        }

        // No `[DONE]` on the success path: the OpenAI Responses streaming
        // schema treats `response.completed` (or .failed/.incomplete) as
        // the per-response terminator, and the compliance suite advances to
        // the next turn the moment it sees one. A trailing `[DONE]` would
        // arrive *after* that advance and be misread as the next turn's
        // marker, killing chained sessions. We still emit `[DONE]` for
        // error fallbacks (see `wsSendErrorTurn`) where no terminal event
        // is sent.
    }
}

/// Reshape a WS `response.create` JSON object into an OpenAI-Responses HTTP
/// request body. We strip the `type` discriminator and force both
/// `stream: true` (WS is inherently streaming) and `store: true` (so
/// handleResponses persists the response in the global store, where the WS
/// handler can later move it into the connection-local cache when the user
/// requested `store: false`).
fn buildResponsesBodyFromWsRequest(allocator: std.mem.Allocator, root: std.json.ObjectMap) ![]u8 {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);
    try buf.append(allocator, '{');
    var first = true;
    var stream_seen = false;
    var store_seen = false;
    var iter = root.iterator();
    while (iter.next()) |entry| {
        const k = entry.key_ptr.*;
        if (std.mem.eql(u8, k, "type")) continue;
        if (std.mem.eql(u8, k, "stream")) stream_seen = true;
        if (std.mem.eql(u8, k, "store")) store_seen = true;
        if (!first) try buf.append(allocator, ',');
        first = false;
        const ek = try jsonEscape(allocator, k);
        defer allocator.free(ek);
        try buf.appendSlice(allocator, ek);
        try buf.append(allocator, ':');
        if (std.mem.eql(u8, k, "stream") or std.mem.eql(u8, k, "store")) {
            try buf.appendSlice(allocator, "true");
        } else {
            try responses_mod.serializeJsonValue(allocator, &buf, entry.value_ptr.*);
        }
    }
    if (!stream_seen) {
        if (!first) try buf.append(allocator, ',');
        first = false;
        try buf.appendSlice(allocator, "\"stream\":true");
    }
    if (!store_seen) {
        if (!first) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, "\"store\":true");
    }
    try buf.append(allocator, '}');
    return try buf.toOwnedSlice(allocator);
}

// ─── Response envelope echo helpers ──────────────────────────────────────
// The OpenAI Responses API ResponseResource schema requires the response to
// echo most request configuration (tools, tool_choice, text.format, reasoning,
// metadata, etc.). These helpers re-render those values from the parsed
// request JSON in the exact shape the schema demands. Caller owns the result.

fn renderResponsesToolsEcho(allocator: std.mem.Allocator, tools_val: ?std.json.Value) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    try buf.append(allocator, '[');
    var emitted: usize = 0;
    if (tools_val) |tv| if (tv == .array) {
        for (tv.array.items) |tool_val| {
            if (tool_val != .object) continue;
            const tool = tool_val.object;
            const t = if (tool.get("type")) |x| (if (x == .string) x.string else "") else "";
            if (!std.mem.eql(u8, t, "function")) continue;

            // Accept both flat (Responses API) and nested-under-"function" (chat-completions) shapes.
            var fn_obj_opt: ?std.json.ObjectMap = null;
            if (tool.get("function")) |fv| if (fv == .object) {
                fn_obj_opt = fv.object;
            };

            const name_v: ?std.json.Value = tool.get("name") orelse if (fn_obj_opt) |fo| fo.get("name") else null;
            const desc_v: ?std.json.Value = tool.get("description") orelse if (fn_obj_opt) |fo| fo.get("description") else null;
            const params_v: ?std.json.Value = tool.get("parameters") orelse if (fn_obj_opt) |fo| fo.get("parameters") else null;
            const strict_v: ?std.json.Value = tool.get("strict") orelse if (fn_obj_opt) |fo| fo.get("strict") else null;

            if (emitted > 0) try buf.append(allocator, ',');
            emitted += 1;
            try buf.appendSlice(allocator, "{\"type\":\"function\",\"name\":");
            if (name_v) |nv| if (nv == .string) {
                const e = try jsonEscape(allocator, nv.string);
                defer allocator.free(e);
                try buf.appendSlice(allocator, e);
            } else {
                try buf.appendSlice(allocator, "\"\"");
            } else {
                try buf.appendSlice(allocator, "\"\"");
            }
            try buf.appendSlice(allocator, ",\"description\":");
            if (desc_v) |dv| if (dv == .string) {
                const e = try jsonEscape(allocator, dv.string);
                defer allocator.free(e);
                try buf.appendSlice(allocator, e);
            } else {
                try buf.appendSlice(allocator, "null");
            } else {
                try buf.appendSlice(allocator, "null");
            }
            try buf.appendSlice(allocator, ",\"parameters\":");
            if (params_v) |pv| if (pv == .object) {
                try responses_mod.serializeJsonValue(allocator, &buf, pv);
            } else {
                try buf.appendSlice(allocator, "null");
            } else {
                try buf.appendSlice(allocator, "null");
            }
            try buf.appendSlice(allocator, ",\"strict\":");
            if (strict_v) |sv| if (sv == .bool) {
                try buf.appendSlice(allocator, if (sv.bool) "true" else "false");
            } else {
                try buf.appendSlice(allocator, "null");
            } else {
                try buf.appendSlice(allocator, "null");
            }
            try buf.append(allocator, '}');
        }
    };
    try buf.append(allocator, ']');
    return try buf.toOwnedSlice(allocator);
}

fn renderResponsesToolChoiceEcho(allocator: std.mem.Allocator, tc_val: ?std.json.Value) ![]const u8 {
    if (tc_val) |v| switch (v) {
        .string => |s| {
            if (std.mem.eql(u8, s, "auto") or std.mem.eql(u8, s, "none") or std.mem.eql(u8, s, "required")) {
                return try std.fmt.allocPrint(allocator, "\"{s}\"", .{s});
            }
        },
        .object => |obj| {
            const t = if (obj.get("type")) |x| (if (x == .string) x.string else "") else "";
            if (std.mem.eql(u8, t, "function")) {
                var name: []const u8 = "";
                if (obj.get("name")) |x| {
                    if (x == .string) name = x.string;
                } else if (obj.get("function")) |fv| if (fv == .object) {
                    if (fv.object.get("name")) |nv| if (nv == .string) {
                        name = nv.string;
                    };
                };
                if (name.len > 0) {
                    const esc = try jsonEscape(allocator, name);
                    defer allocator.free(esc);
                    return try std.fmt.allocPrint(allocator, "{{\"type\":\"function\",\"name\":{s}}}", .{esc});
                }
                return try allocator.dupe(u8, "{\"type\":\"function\"}");
            }
        },
        else => {},
    };
    return try allocator.dupe(u8, "\"auto\"");
}

fn renderResponsesTextEcho(allocator: std.mem.Allocator, root: std.json.ObjectMap) ![]const u8 {
    if (root.get("text")) |tv| if (tv == .object) {
        var buf = std.ArrayList(u8).empty;
        errdefer buf.deinit(allocator);
        try responses_mod.serializeJsonValue(allocator, &buf, tv);
        return try buf.toOwnedSlice(allocator);
    };
    if (root.get("response_format")) |rf| if (rf == .object) {
        var buf = std.ArrayList(u8).empty;
        errdefer buf.deinit(allocator);
        try buf.appendSlice(allocator, "{\"format\":");
        try responses_mod.serializeJsonValue(allocator, &buf, rf);
        try buf.append(allocator, '}');
        return try buf.toOwnedSlice(allocator);
    };
    return try allocator.dupe(u8, "{\"format\":{\"type\":\"text\"}}");
}

fn renderResponsesReasoningEcho(allocator: std.mem.Allocator, root: std.json.ObjectMap) ![]const u8 {
    var effort_buf: [32]u8 = undefined;
    var effort_str: []const u8 = "null";
    var summary_str: []const u8 = "null";
    if (root.get("reasoning")) |rv| if (rv == .object) {
        if (rv.object.get("effort")) |ev| if (ev == .string) {
            const s = ev.string;
            const valid = std.mem.eql(u8, s, "minimal") or
                std.mem.eql(u8, s, "low") or
                std.mem.eql(u8, s, "medium") or
                std.mem.eql(u8, s, "high");
            if (valid) {
                effort_str = std.fmt.bufPrint(&effort_buf, "\"{s}\"", .{s}) catch "null";
            }
        };
        if (rv.object.get("summary")) |sv| if (sv == .string) {
            const s = sv.string;
            const valid = std.mem.eql(u8, s, "auto") or
                std.mem.eql(u8, s, "concise") or
                std.mem.eql(u8, s, "detailed");
            if (valid) {
                // share buffer is fine since alloc happens immediately after
                if (std.mem.eql(u8, s, "auto")) summary_str = "\"auto\""
                else if (std.mem.eql(u8, s, "concise")) summary_str = "\"concise\""
                else summary_str = "\"detailed\"";
            }
        };
    };
    return try std.fmt.allocPrint(allocator, "{{\"effort\":{s},\"summary\":{s}}}", .{ effort_str, summary_str });
}

fn renderResponsesMetadataEcho(allocator: std.mem.Allocator, root: std.json.ObjectMap) ![]const u8 {
    if (root.get("metadata")) |mv| if (mv == .object) {
        var buf = std.ArrayList(u8).empty;
        errdefer buf.deinit(allocator);
        try responses_mod.serializeJsonValue(allocator, &buf, mv);
        return try buf.toOwnedSlice(allocator);
    };
    return try allocator.dupe(u8, "{}");
}

/// Echoed-back fields that round out the OpenAI Responses envelope.
/// Owned by the caller (POST handler); freed after the final envelope is built.
const ResponseEcho = struct {
    // Pre-rendered JSON fragments (raw object/array text — caller owns).
    tools_json: []const u8 = "[]",
    tool_choice_json: []const u8 = "\"auto\"",
    text_json: []const u8 = "{\"format\":{\"type\":\"text\"}}",
    reasoning_json: []const u8 = "{\"effort\":null,\"summary\":null}",
    metadata_json: []const u8 = "{}",
    // Plain values; serialized inline.
    instructions: ?[]const u8 = null,
    truncation: []const u8 = "disabled",
    service_tier: []const u8 = "default",
    safety_identifier: ?[]const u8 = null,
    prompt_cache_key: ?[]const u8 = null,
    temperature: f32 = 1.0,
    top_p: f32 = 1.0,
    presence_penalty: f32 = 0.0,
    frequency_penalty: f32 = 0.0,
    top_logprobs: u32 = 0,
    parallel_tool_calls: bool = true,
    background: bool = false,
    max_output_tokens: ?u32 = null,
    max_tool_calls: ?u32 = null,
};

/// Build the Responses envelope JSON body. Used for both the response.created
/// skeleton (in_progress, output:[]) and the final response.completed body.
/// Shape matches the OpenAI Responses API ResponseResource schema.
fn buildResponsesEnvelope(
    io: std.Io,
    allocator: std.mem.Allocator,
    esc_resp_id: []const u8,
    esc_model: []const u8,
    status_str: []const u8,
    output_json: []const u8,
    input_tokens: u32,
    output_tokens: u32,
    cached_input_tokens: u32,
    reasoning_output_tokens: u32,
    should_store: bool,
    prev_id: ?[]const u8,
    incomplete: bool,
    completed: bool,
    echo: ResponseEcho,
) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    var num_buf: [32]u8 = undefined;

    try buf.append(allocator, '{');
    try buf.appendSlice(allocator, "\"id\":");
    try buf.appendSlice(allocator, esc_resp_id);

    try buf.appendSlice(allocator, ",\"object\":\"response\",\"created_at\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{nowSecs(io)}));

    if (completed) {
        try buf.appendSlice(allocator, ",\"completed_at\":");
        try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{nowSecs(io)}));
    } else {
        try buf.appendSlice(allocator, ",\"completed_at\":null");
    }

    try buf.appendSlice(allocator, ",\"status\":\"");
    try buf.appendSlice(allocator, status_str);
    try buf.append(allocator, '"');

    if (incomplete) {
        try buf.appendSlice(allocator, ",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}");
    } else {
        try buf.appendSlice(allocator, ",\"incomplete_details\":null");
    }

    try buf.appendSlice(allocator, ",\"model\":");
    try buf.appendSlice(allocator, esc_model);

    if (prev_id) |pid| {
        const esc_pid = try jsonEscape(allocator, pid);
        defer allocator.free(esc_pid);
        try buf.appendSlice(allocator, ",\"previous_response_id\":");
        try buf.appendSlice(allocator, esc_pid);
    } else {
        try buf.appendSlice(allocator, ",\"previous_response_id\":null");
    }

    if (echo.instructions) |s| {
        const esc = try jsonEscape(allocator, s);
        defer allocator.free(esc);
        try buf.appendSlice(allocator, ",\"instructions\":");
        try buf.appendSlice(allocator, esc);
    } else {
        try buf.appendSlice(allocator, ",\"instructions\":null");
    }

    try buf.appendSlice(allocator, ",\"output\":");
    try buf.appendSlice(allocator, output_json);

    try buf.appendSlice(allocator, ",\"error\":null");

    try buf.appendSlice(allocator, ",\"tools\":");
    try buf.appendSlice(allocator, echo.tools_json);

    try buf.appendSlice(allocator, ",\"tool_choice\":");
    try buf.appendSlice(allocator, echo.tool_choice_json);

    try buf.appendSlice(allocator, ",\"truncation\":\"");
    try buf.appendSlice(allocator, echo.truncation);
    try buf.append(allocator, '"');

    try buf.appendSlice(allocator, ",\"parallel_tool_calls\":");
    try buf.appendSlice(allocator, if (echo.parallel_tool_calls) "true" else "false");

    try buf.appendSlice(allocator, ",\"text\":");
    try buf.appendSlice(allocator, echo.text_json);

    try buf.appendSlice(allocator, ",\"top_p\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{echo.top_p}));
    try buf.appendSlice(allocator, ",\"presence_penalty\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{echo.presence_penalty}));
    try buf.appendSlice(allocator, ",\"frequency_penalty\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{echo.frequency_penalty}));
    try buf.appendSlice(allocator, ",\"top_logprobs\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{echo.top_logprobs}));
    try buf.appendSlice(allocator, ",\"temperature\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{echo.temperature}));

    try buf.appendSlice(allocator, ",\"reasoning\":");
    try buf.appendSlice(allocator, echo.reasoning_json);

    try buf.appendSlice(allocator, ",\"usage\":{\"input_tokens\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{input_tokens}));
    try buf.appendSlice(allocator, ",\"output_tokens\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{output_tokens}));
    try buf.appendSlice(allocator, ",\"total_tokens\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{input_tokens + output_tokens}));
    try buf.appendSlice(allocator, ",\"input_tokens_details\":{\"cached_tokens\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{cached_input_tokens}));
    try buf.appendSlice(allocator, "},\"output_tokens_details\":{\"reasoning_tokens\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{reasoning_output_tokens}));
    try buf.appendSlice(allocator, "}}");

    if (echo.max_output_tokens) |n| {
        try buf.appendSlice(allocator, ",\"max_output_tokens\":");
        try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{n}));
    } else {
        try buf.appendSlice(allocator, ",\"max_output_tokens\":null");
    }

    if (echo.max_tool_calls) |n| {
        try buf.appendSlice(allocator, ",\"max_tool_calls\":");
        try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{n}));
    } else {
        try buf.appendSlice(allocator, ",\"max_tool_calls\":null");
    }

    try buf.appendSlice(allocator, ",\"store\":");
    try buf.appendSlice(allocator, if (should_store) "true" else "false");

    try buf.appendSlice(allocator, ",\"background\":");
    try buf.appendSlice(allocator, if (echo.background) "true" else "false");

    try buf.appendSlice(allocator, ",\"service_tier\":\"");
    try buf.appendSlice(allocator, echo.service_tier);
    try buf.append(allocator, '"');

    try buf.appendSlice(allocator, ",\"metadata\":");
    try buf.appendSlice(allocator, echo.metadata_json);

    if (echo.safety_identifier) |s| {
        const esc = try jsonEscape(allocator, s);
        defer allocator.free(esc);
        try buf.appendSlice(allocator, ",\"safety_identifier\":");
        try buf.appendSlice(allocator, esc);
    } else {
        try buf.appendSlice(allocator, ",\"safety_identifier\":null");
    }

    if (echo.prompt_cache_key) |s| {
        const esc = try jsonEscape(allocator, s);
        defer allocator.free(esc);
        try buf.appendSlice(allocator, ",\"prompt_cache_key\":");
        try buf.appendSlice(allocator, esc);
    } else {
        try buf.appendSlice(allocator, ",\"prompt_cache_key\":null");
    }

    try buf.append(allocator, '}');
    return try buf.toOwnedSlice(allocator);
}

/// Emit output_item.added (type=reasoning) + reasoning_summary_part.added.
/// Pair with `emitResponsesReasoningEnd` after deltas are streamed.
fn emitResponsesReasoningStart(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
) !void {
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    {
        const item_added = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_item.added","output_index":{d},"item":{{"type":"reasoning","id":{s},"summary":[]}}}}
        , .{ output_index, esc_id });
        defer allocator.free(item_added);
        try sendResponsesEvent(allocator, stream, seq, "response.output_item.added", item_added);
    }
    {
        const part_added = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.reasoning_summary_part.added","item_id":{s},"output_index":{d},"summary_index":0,"part":{{"type":"summary_text","text":""}}}}
        , .{ esc_id, output_index });
        defer allocator.free(part_added);
        try sendResponsesEvent(allocator, stream, seq, "response.reasoning_summary_part.added", part_added);
    }
}

/// Emit a single reasoning_summary_text.delta event with the given chunk.
fn emitResponsesReasoningDelta(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
    delta_text: []const u8,
) !void {
    if (delta_text.len == 0) return;
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    const esc_delta = try jsonEscape(allocator, delta_text);
    defer allocator.free(esc_delta);
    const delta = try std.fmt.allocPrint(allocator,
        \\{{"type":"response.reasoning_summary_text.delta","item_id":{s},"output_index":{d},"summary_index":0,"delta":{s}}}
    , .{ esc_id, output_index, esc_delta });
    defer allocator.free(delta);
    try sendResponsesEvent(allocator, stream, seq, "response.reasoning_summary_text.delta", delta);
}

/// Emit reasoning_summary_text.done + reasoning_summary_part.done + output_item.done.
fn emitResponsesReasoningEnd(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
    full_text: []const u8,
) !void {
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    const esc_text = try jsonEscape(allocator, full_text);
    defer allocator.free(esc_text);
    {
        const done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.reasoning_summary_text.done","item_id":{s},"output_index":{d},"summary_index":0,"text":{s}}}
        , .{ esc_id, output_index, esc_text });
        defer allocator.free(done);
        try sendResponsesEvent(allocator, stream, seq, "response.reasoning_summary_text.done", done);
    }
    {
        const part_done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.reasoning_summary_part.done","item_id":{s},"output_index":{d},"summary_index":0,"part":{{"type":"summary_text","text":{s}}}}}
        , .{ esc_id, output_index, esc_text });
        defer allocator.free(part_done);
        try sendResponsesEvent(allocator, stream, seq, "response.reasoning_summary_part.done", part_done);
    }
    {
        const item_done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_item.done","output_index":{d},"item":{{"type":"reasoning","id":{s},"summary":[{{"type":"summary_text","text":{s}}}]}}}}
        , .{ output_index, esc_id, esc_text });
        defer allocator.free(item_done);
        try sendResponsesEvent(allocator, stream, seq, "response.output_item.done", item_done);
    }
}

/// Emit a full reasoning output item in one shot. Used when the entire
/// reasoning text is known up-front (non-streaming generation paths).
fn emitResponsesReasoningEvents(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
    reasoning_text: []const u8,
) !void {
    try emitResponsesReasoningStart(allocator, stream, seq, output_index, item_id);
    try emitResponsesReasoningDelta(allocator, stream, seq, output_index, item_id, reasoning_text);
    try emitResponsesReasoningEnd(allocator, stream, seq, output_index, item_id, reasoning_text);
}

/// Emit the SSE event sequence for a function_call output item: output_item.added,
/// function_call_arguments.delta (single full-args delta), .done, output_item.done.
fn emitResponsesFunctionCallEvents(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    fc_id: []const u8,
    call_id: []const u8,
    name: []const u8,
    arguments_json: []const u8,
) !void {
    const esc_id = try jsonEscape(allocator, fc_id);
    defer allocator.free(esc_id);
    const esc_call = try jsonEscape(allocator, call_id);
    defer allocator.free(esc_call);
    const esc_name = try jsonEscape(allocator, name);
    defer allocator.free(esc_name);
    const esc_args = try jsonEscape(allocator, arguments_json);
    defer allocator.free(esc_args);

    {
        const item_added = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_item.added","output_index":{d},"item":{{"type":"function_call","id":{s},"call_id":{s},"name":{s},"arguments":"","status":"in_progress"}}}}
        , .{ output_index, esc_id, esc_call, esc_name });
        defer allocator.free(item_added);
        try sendResponsesEvent(allocator, stream, seq, "response.output_item.added", item_added);
    }
    {
        const delta = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.function_call_arguments.delta","item_id":{s},"output_index":{d},"delta":{s}}}
        , .{ esc_id, output_index, esc_args });
        defer allocator.free(delta);
        try sendResponsesEvent(allocator, stream, seq, "response.function_call_arguments.delta", delta);
    }
    {
        const done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.function_call_arguments.done","item_id":{s},"output_index":{d},"arguments":{s}}}
        , .{ esc_id, output_index, esc_args });
        defer allocator.free(done);
        try sendResponsesEvent(allocator, stream, seq, "response.function_call_arguments.done", done);
    }
    {
        const item_done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_item.done","output_index":{d},"item":{{"type":"function_call","id":{s},"call_id":{s},"name":{s},"arguments":{s},"status":"completed"}}}}
        , .{ output_index, esc_id, esc_call, esc_name, esc_args });
        defer allocator.free(item_done);
        try sendResponsesEvent(allocator, stream, seq, "response.output_item.done", item_done);
    }
}

/// Emit output_item.added (type=message) + content_part.added.
/// Pair with `emitResponsesMessageEnd` after output_text.delta events stream.
fn emitResponsesMessageStart(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
) !void {
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    {
        const item_added = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_item.added","output_index":{d},"item":{{"type":"message","id":{s},"role":"assistant","status":"in_progress","content":[]}}}}
        , .{ output_index, esc_id });
        defer allocator.free(item_added);
        try sendResponsesEvent(allocator, stream, seq, "response.output_item.added", item_added);
    }
    {
        const part_added = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.content_part.added","item_id":{s},"output_index":{d},"content_index":0,"part":{{"type":"output_text","text":"","annotations":[]}}}}
        , .{ esc_id, output_index });
        defer allocator.free(part_added);
        try sendResponsesEvent(allocator, stream, seq, "response.content_part.added", part_added);
    }
}

/// Emit a single output_text.delta event for an in-progress message.
fn emitResponsesMessageDelta(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
    delta_text: []const u8,
) !void {
    if (delta_text.len == 0) return;
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    const esc_delta = try jsonEscape(allocator, delta_text);
    defer allocator.free(esc_delta);
    const delta = try std.fmt.allocPrint(allocator,
        \\{{"type":"response.output_text.delta","item_id":{s},"output_index":{d},"content_index":0,"delta":{s}}}
    , .{ esc_id, output_index, esc_delta });
    defer allocator.free(delta);
    try sendResponsesEvent(allocator, stream, seq, "response.output_text.delta", delta);
}

/// Emit output_text.done + content_part.done + output_item.done.
fn emitResponsesMessageEnd(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
    full_text: []const u8,
) !void {
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    const esc_text = try jsonEscape(allocator, full_text);
    defer allocator.free(esc_text);
    {
        const done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_text.done","item_id":{s},"output_index":{d},"content_index":0,"text":{s}}}
        , .{ esc_id, output_index, esc_text });
        defer allocator.free(done);
        try sendResponsesEvent(allocator, stream, seq, "response.output_text.done", done);
    }
    {
        const part_done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.content_part.done","item_id":{s},"output_index":{d},"content_index":0,"part":{{"type":"output_text","text":{s},"annotations":[]}}}}
        , .{ esc_id, output_index, esc_text });
        defer allocator.free(part_done);
        try sendResponsesEvent(allocator, stream, seq, "response.content_part.done", part_done);
    }
    {
        const item_done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_item.done","output_index":{d},"item":{{"type":"message","id":{s},"role":"assistant","status":"completed","content":[{{"type":"output_text","text":{s},"annotations":[]}}]}}}}
        , .{ output_index, esc_id, esc_text });
        defer allocator.free(item_done);
        try sendResponsesEvent(allocator, stream, seq, "response.output_item.done", item_done);
    }
}

/// Emit a full message output item in one shot. Used when the entire visible
/// text is known up-front (non-streaming generation paths).
fn emitResponsesMessageEvents(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
    text: []const u8,
) !void {
    try emitResponsesMessageStart(allocator, stream, seq, output_index, item_id);
    try emitResponsesMessageDelta(allocator, stream, seq, output_index, item_id, text);
    try emitResponsesMessageEnd(allocator, stream, seq, output_index, item_id, text);
}

/// Persist a finished response to the in-memory store. The stored history is
/// the input messages plus the assistant turn, deep-copied into the entry's
/// arena so it stays valid across the request that produced it.
fn storeResponse(
    io: std.Io,
    gpa: std.mem.Allocator,
    resp_id: []const u8,
    model_name: []const u8,
    status_str: []const u8,
    body_json: []const u8,
    input_messages: []const chat_mod.Message,
    visible_text: []const u8,
    reasoning_text: ?[]const u8,
    tool_calls: ?[]const chat_mod.ToolCall,
) !void {
    const sr = try gpa.create(responses_mod.StoredResponse);
    errdefer gpa.destroy(sr);
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    // Build the assistant message that produced this response.
    var assistant_text_parts = std.ArrayList(u8).empty;
    defer assistant_text_parts.deinit(a);
    if (reasoning_text) |rt| {
        try assistant_text_parts.appendSlice(a, "<think>");
        try assistant_text_parts.appendSlice(a, rt);
        try assistant_text_parts.appendSlice(a, "</think>");
    }
    try assistant_text_parts.appendSlice(a, visible_text);
    const assistant_content = try a.dupe(u8, assistant_text_parts.items);

    var assistant_tool_calls: ?[]chat_mod.ToolCall = null;
    if (tool_calls) |tcs| if (tcs.len > 0) {
        const arr = try a.alloc(chat_mod.ToolCall, tcs.len);
        for (tcs, 0..) |tc, i| {
            arr[i] = .{
                .id = try a.dupe(u8, tc.id),
                .name = try a.dupe(u8, tc.name),
                .arguments = try a.dupe(u8, tc.arguments),
            };
        }
        assistant_tool_calls = arr;
    };

    // Deep-copy input messages.
    const total_msgs = input_messages.len + 1; // +1 for assistant turn
    const history = try a.alloc(chat_mod.Message, total_msgs);
    for (input_messages, 0..) |m, i| {
        history[i] = .{
            .role = try a.dupe(u8, m.role),
            .content = try a.dupe(u8, m.content),
            .tool_call_id = if (m.tool_call_id) |tid| try a.dupe(u8, tid) else null,
            .tool_calls = if (m.tool_calls) |tcs| blk: {
                const copied = try a.alloc(chat_mod.ToolCall, tcs.len);
                for (tcs, 0..) |tc, j| copied[j] = .{
                    .id = try a.dupe(u8, tc.id),
                    .name = try a.dupe(u8, tc.name),
                    .arguments = try a.dupe(u8, tc.arguments),
                };
                break :blk copied;
            } else null,
            // images: not preserved across requests (would require deep-copying pixel buffers).
            .images = null,
        };
    }
    history[total_msgs - 1] = .{
        .role = try a.dupe(u8, "assistant"),
        .content = assistant_content,
        .tool_calls = assistant_tool_calls,
    };

    sr.* = .{
        .id = try a.dupe(u8, resp_id),
        .created_at = nowSecs(io),
        .model = try a.dupe(u8, model_name),
        .status = try a.dupe(u8, status_str),
        .body_json = try a.dupe(u8, body_json),
        .history = history,
        .arena = arena,
    };
    errdefer sr.deinit();

    const store = getOrInitResponseStore(io, gpa);
    try store.put(sr);
}

// ── Tests ──

const testing = std.testing;

test "findContentLength parses header" {
    try testing.expectEqual(@as(?usize, 42), findContentLength("Host: localhost\r\nContent-Length: 42\r\nAccept: */*"));
}

test "findContentLength case insensitive" {
    try testing.expectEqual(@as(?usize, 100), findContentLength("content-length: 100"));
    try testing.expectEqual(@as(?usize, 100), findContentLength("Content-Length: 100"));
    try testing.expectEqual(@as(?usize, 100), findContentLength("CONTENT-LENGTH: 100"));
}

test "findContentLength returns null when missing" {
    try testing.expect(findContentLength("Host: localhost\r\nAccept: */*") == null);
    try testing.expect(findContentLength("") == null);
}

test "extractJsonField extracts array" {
    const body =
        \\{"messages":[{"role":"user","content":"hi"}],"temperature":0.7}
    ;
    const result = extractJsonField(body, "messages").?;
    try testing.expect(std.mem.startsWith(u8, result, "["));
    try testing.expect(std.mem.endsWith(u8, result, "]"));
}

test "extractJsonField extracts nested object" {
    const body =
        \\{"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object"}}}}
    ;
    const result = extractJsonField(body, "response_format").?;
    try testing.expect(std.mem.startsWith(u8, result, "{"));
    try testing.expect(std.mem.endsWith(u8, result, "}"));
}

test "extractJsonField returns null for missing field" {
    const body = "{\"messages\":[]}";
    try testing.expect(extractJsonField(body, "tools") == null);
}

test "extractJsonField handles escaped quotes in strings" {
    const body =
        \\{"tools":[{"type":"function","function":{"name":"say_\"hello\""}}]}
    ;
    const result = extractJsonField(body, "tools").?;
    try testing.expect(std.mem.startsWith(u8, result, "["));
    try testing.expect(std.mem.endsWith(u8, result, "]"));
}

test "jsonEscape basic string" {
    const allocator = testing.allocator;
    const result = try jsonEscape(allocator, "hello");
    defer allocator.free(result);
    try testing.expectEqualStrings("\"hello\"", result);
}

test "jsonEscape special characters" {
    const allocator = testing.allocator;
    const result = try jsonEscape(allocator, "line1\nline2\t\"quoted\"\\back");
    defer allocator.free(result);
    try testing.expectEqualStrings("\"line1\\nline2\\t\\\"quoted\\\"\\\\back\"", result);
}

test "jsonEscape control characters" {
    const allocator = testing.allocator;
    const input = &[_]u8{ 0x01, 0x02 };
    const result = try jsonEscape(allocator, input);
    defer allocator.free(result);
    try testing.expectEqualStrings("\"\\u0001\\u0002\"", result);
}

test "jsonEscape empty string" {
    const allocator = testing.allocator;
    const result = try jsonEscape(allocator, "");
    defer allocator.free(result);
    try testing.expectEqualStrings("\"\"", result);
}

test "utf8TrailingIncomplete complete ASCII" {
    try testing.expectEqual(@as(usize, 0), utf8TrailingIncomplete("hello"));
}

test "utf8TrailingIncomplete complete multibyte" {
    // 🎉 = F0 9F 8E 89 (4-byte sequence, complete)
    try testing.expectEqual(@as(usize, 0), utf8TrailingIncomplete("\xF0\x9F\x8E\x89"));
}

test "utf8TrailingIncomplete partial 4-byte" {
    // First 3 bytes of a 4-byte sequence
    try testing.expectEqual(@as(usize, 3), utf8TrailingIncomplete("\xF0\x9F\x8E"));
    // First 2 bytes
    try testing.expectEqual(@as(usize, 2), utf8TrailingIncomplete("\xF0\x9F"));
    // First 1 byte
    try testing.expectEqual(@as(usize, 1), utf8TrailingIncomplete("\xF0"));
}

test "utf8TrailingIncomplete partial after complete" {
    // "hi" + first 2 bytes of emoji
    try testing.expectEqual(@as(usize, 2), utf8TrailingIncomplete("hi\xF0\x9F"));
}

test "utf8TrailingIncomplete empty" {
    try testing.expectEqual(@as(usize, 0), utf8TrailingIncomplete(""));
}

test "parseJsonFloat returns value when present" {
    const allocator = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{\"temp\":0.7}", .{});
    defer parsed.deinit();
    const result = parseJsonFloat(parsed.value.object, "temp", 1.0, 0.0, 2.0);
    try testing.expectApproxEqAbs(@as(f32, 0.7), result, 0.001);
}

test "parseJsonFloat returns default when missing" {
    const allocator = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{}", .{});
    defer parsed.deinit();
    const result = parseJsonFloat(parsed.value.object, "temp", 1.0, 0.0, 2.0);
    try testing.expectApproxEqAbs(@as(f32, 1.0), result, 0.001);
}

test "parseJsonFloat clamps to range" {
    const allocator = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{\"temp\":5.0}", .{});
    defer parsed.deinit();
    const result = parseJsonFloat(parsed.value.object, "temp", 1.0, 0.0, 2.0);
    try testing.expectApproxEqAbs(@as(f32, 2.0), result, 0.001);
}

test "parseJsonFloat handles integer value" {
    const allocator = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{\"temp\":1}", .{});
    defer parsed.deinit();
    const result = parseJsonFloat(parsed.value.object, "temp", 0.5, 0.0, 2.0);
    try testing.expectApproxEqAbs(@as(f32, 1.0), result, 0.001);
}

test "getEffectiveContextLength uses ctx_size override" {
    const original = max_context_size;
    defer max_context_size = original;

    max_context_size = 4096;
    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 32768;
    try testing.expectEqual(@as(u32, 4096), getEffectiveContextLength(&config));
}

test "getEffectiveContextLength computes safe default from GPU memory" {
    const original = max_context_size;
    defer max_context_size = original;

    max_context_size = 0;
    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 131072;
    config.num_attention_heads = 8;
    config.num_hidden_layers = 42;
    config.num_key_value_heads = 2;
    config.head_dim = 256;
    // Should compute a value based on GPU memory, not hardcoded 16K
    const computed = getEffectiveContextLength(&config);
    try testing.expect(computed > 0);
    try testing.expect(computed <= config.max_position_embeddings);

    // Explicit --ctx-size overrides the computed default
    max_context_size = 32768;
    try testing.expectEqual(@as(u32, 32768), getEffectiveContextLength(&config));
}

test "clampMaxTokens no limit when ctx_size=0" {
    const original = max_context_size;
    defer max_context_size = original;
    max_context_size = 0;

    try testing.expectEqual(@as(u32, 1000), clampMaxTokens(1000, 500));
}

test "clampMaxTokens clamps when would exceed" {
    const original = max_context_size;
    defer max_context_size = original;
    max_context_size = 4096;

    // prompt=3000, max_tokens=2000 → clamp to 1096
    try testing.expectEqual(@as(u32, 1096), clampMaxTokens(2000, 3000));
}

test "clampMaxTokens no clamp when fits" {
    const original = max_context_size;
    defer max_context_size = original;
    max_context_size = 4096;

    // prompt=100, max_tokens=200 → fits, no clamp
    try testing.expectEqual(@as(u32, 200), clampMaxTokens(200, 100));
}

test "clampMaxTokens at boundary" {
    const original = max_context_size;
    defer max_context_size = original;
    max_context_size = 4096;

    // prompt=4096 → only 1 token allowed
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(100, 4096));
    // prompt=4095 → only 1 token remaining
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(100, 4095));
}

test "getTimeoutNs computes correctly" {
    const original = request_timeout_sec;
    defer request_timeout_sec = original;

    request_timeout_sec = 300;
    try testing.expectEqual(@as(u64, 300_000_000_000), getTimeoutNs());

    request_timeout_sec = 0;
    try testing.expectEqual(@as(u64, 0), getTimeoutNs());
}


test "responsesToolExists validates Responses function tool names" {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator,
        \\[{"type":"function","name":"smartSearch","parameters":{}},{"type":"web_search"}]
    , .{});
    defer parsed.deinit();

    try testing.expect(responsesToolExists(parsed.value, "smartSearch"));
    try testing.expect(!responsesToolExists(parsed.value, "cruise_cards"));
}

test "buildResponsesJsonInstruction scopes schema prompt when tools are active" {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator,
        \\{"type":"object","properties":{"blocks":{"type":"array"}},"required":["blocks"]}
    , .{});
    defer parsed.deinit();

    const with_tools = try buildResponsesJsonInstruction(testing.allocator, parsed.value, true);
    defer testing.allocator.free(with_tools);
    try testing.expect(std.mem.indexOf(u8, with_tools, "If you answer without calling a function") != null);
    try testing.expect(std.mem.indexOf(u8, with_tools, "function call is needed") != null);
    try testing.expect(std.mem.indexOf(u8, with_tools, "\"blocks\"") != null);

    const without_tools = try buildResponsesJsonInstruction(testing.allocator, parsed.value, false);
    defer testing.allocator.free(without_tools);
    try testing.expect(std.mem.indexOf(u8, without_tools, "Respond with valid JSON only") != null);
    try testing.expect(std.mem.indexOf(u8, without_tools, "without calling a function") == null);
}

test "shouldInjectResponsesJsonInstruction skips required tool turns" {
    try testing.expect(!shouldInjectResponsesJsonInstruction(false, false, null));
    try testing.expect(shouldInjectResponsesJsonInstruction(true, false, null));
    try testing.expect(shouldInjectResponsesJsonInstruction(true, true, null));
    try testing.expect(!shouldInjectResponsesJsonInstruction(true, true, "required"));
}

test "deinitGlobalResponseStore frees stored responses" {
    deinitGlobalResponseStore();
    defer deinitGlobalResponseStore();

    const messages = [_]chat_mod.Message{.{ .role = "user", .content = "hi" }};
    try storeResponse(testing.io, testing.allocator, "resp_test", "mlx-serve", "completed", "{}", &messages, "hello", null, null);

    if (global_response_store) |*store| {
        try testing.expectEqual(@as(usize, 1), store.map.count());
    } else {
        return error.TestUnexpectedResult;
    }

    deinitGlobalResponseStore();
    try testing.expect(global_response_store == null);
}

test "isJsonObjectString only accepts JSON objects" {
    try testing.expect(isJsonObjectString(testing.allocator, "{\"destination\":\"CARIBBEAN\"}"));
    try testing.expect(!isJsonObjectString(testing.allocator, "[]"));
    try testing.expect(!isJsonObjectString(testing.allocator, "not-json"));
}

test "insertImageTokens lands right after the user-turn marker (Gemma 4)" {
    var config = model_mod.ModelConfig{};
    // Simulate the marker that populateUserTurnMarker would store for Gemma 4:
    // <|turn>(105) user(2364) \n(107).
    config.user_turn_marker_ids[0] = 105;
    config.user_turn_marker_ids[1] = 2364;
    config.user_turn_marker_ids[2] = 107;
    config.user_turn_marker_len = 3;
    config.boi_token_id = 200;
    config.eoi_token_id = 201;

    // Prompt: BOS, system text, then a user turn followed by its content tokens
    // and the trailing model-generation prompt. The marker [105, 2364, 107]
    // appears once at the start of the user turn (positions 5-7).
    const prompt = [_]u32{
        2,   500, 501, 502, 503, // BOS + system prefix
        105, 2364, 107,          // <|turn>user\n
        900, 901, 902,           // user content
        106, 107,                // <turn|>\n
        105, 4368, 107,          // <|turn>model\n (generation prompt)
    };

    const out = try insertImageTokens(testing.allocator, &prompt, 999, 4, &config);
    defer testing.allocator.free(out);

    // Image tokens should be inserted right after position 7 (end of marker),
    // i.e., between "<|turn>user\n" and the user content.
    // Expected: prompt[0..8] + BOI + image*4 + EOI + prompt[8..]
    try testing.expectEqual(@as(usize, prompt.len + 6), out.len);
    try testing.expectEqual(@as(u32, 200), out[8]);  // BOI
    try testing.expectEqual(@as(u32, 999), out[9]);  // image
    try testing.expectEqual(@as(u32, 999), out[10]);
    try testing.expectEqual(@as(u32, 999), out[11]);
    try testing.expectEqual(@as(u32, 999), out[12]);
    try testing.expectEqual(@as(u32, 201), out[13]); // EOI
    try testing.expectEqual(@as(u32, 900), out[14]); // first user content token preserved
}

test "insertImageTokens picks the LAST user turn when multiple are present" {
    var config = model_mod.ModelConfig{};
    config.user_turn_marker_ids[0] = 105;
    config.user_turn_marker_ids[1] = 2364;
    config.user_turn_marker_ids[2] = 107;
    config.user_turn_marker_len = 3;
    config.boi_token_id = 200;
    config.eoi_token_id = 201;

    // Two user turns. Vision tokens must land inside the LATER one.
    const prompt = [_]u32{
        105, 2364, 107, 800, 801, // first user turn
        106, 107,
        105, 4368, 107, 850, 851, // first model turn
        106, 107,
        105, 2364, 107, 900,      // second user turn (the one we're answering)
    };

    const out = try insertImageTokens(testing.allocator, &prompt, 999, 1, &config);
    defer testing.allocator.free(out);

    // Marker at positions 14-16; insert after position 17.
    // Original first-user-turn content (800, 801) must be untouched.
    try testing.expectEqual(@as(u32, 800), out[3]);
    try testing.expectEqual(@as(u32, 801), out[4]);
    // BOI at position 17, image at 18, EOI at 19, then original 900 at 20.
    try testing.expectEqual(@as(u32, 200), out[17]);
    try testing.expectEqual(@as(u32, 999), out[18]);
    try testing.expectEqual(@as(u32, 201), out[19]);
    try testing.expectEqual(@as(u32, 900), out[20]);
}

test "insertImageTokens falls back gracefully when marker is unset" {
    var config = model_mod.ModelConfig{};
    // user_turn_marker_len stays 0 — simulates an architecture we don't know
    // how to detect a turn boundary for. Should still produce a valid prompt.
    config.boi_token_id = 200;
    config.eoi_token_id = 201;

    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const out = try insertImageTokens(testing.allocator, &prompt, 999, 2, &config);
    defer testing.allocator.free(out);

    // Should be original len + 2 image + 2 BOI/EOI = +4
    try testing.expectEqual(@as(usize, prompt.len + 4), out.len);
}

test "insertImageTokens is a no-op when image_token_id or n_tokens is zero" {
    var config = model_mod.ModelConfig{};
    config.user_turn_marker_ids[0] = 105;
    config.user_turn_marker_len = 1;

    const prompt = [_]u32{ 1, 2, 105, 3, 4 };

    const out_zero_id = try insertImageTokens(testing.allocator, &prompt, 0, 4, &config);
    defer testing.allocator.free(out_zero_id);
    try testing.expectEqualSlices(u32, &prompt, out_zero_id);

    const out_zero_n = try insertImageTokens(testing.allocator, &prompt, 999, 0, &config);
    defer testing.allocator.free(out_zero_n);
    try testing.expectEqualSlices(u32, &prompt, out_zero_n);
}
