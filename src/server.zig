const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const tokenizer_mod = @import("tokenizer.zig");
const generate_mod = @import("generate.zig");
const drafter_mod = @import("drafter.zig");
const chat_mod = @import("chat.zig");
const model_mod = @import("model.zig");
const qwen_vision = @import("qwen_vision.zig");
const mrope_mod = @import("mrope.zig");
const vision_mod = @import("vision.zig");
const log = @import("log.zig");
const token_mask_mod = @import("token_mask.zig");
const responses_mod = @import("responses.zig");
const pld_index = @import("pld_index.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const tokenize_cache_mod = @import("tokenize_cache.zig");
const scheduler_mod = @import("scheduler.zig");
const ds4_ffi = if (@import("build_options").ios) @import("ds4_ffi_stub.zig") else @import("ds4_ffi.zig");
const model_registry_mod = @import("model_registry.zig");
const model_discovery = @import("model_discovery.zig");
const arch_llama = if (@import("build_options").ios) @import("arch/llama_stub.zig") else @import("arch/llama.zig");
const media_mod = @import("gen.zig");
const stb = @cImport({ @cInclude("stb_image.h"); });
const webp = @cImport({ @cInclude("webp/decode.h"); });
const metrics = @import("status.zig");
const instr = @import("metrics.zig");

const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const Generator = generate_mod.Generator;
const VisionEncoder = vision_mod.VisionEncoder;
const ModelRegistry = model_registry_mod.ModelRegistry;
const LoadedModel = model_registry_mod.LoadedModel;
/// Global flag set by signal handler for graceful shutdown.
var shutdown_requested = std.atomic.Value(bool).init(false);
/// Number of live per-connection threads. Incremented before each spawn,
/// decremented when the thread's handler returns. On shutdown, `serve` waits
/// for this to reach 0 before returning (which runs `scheduler.deinit`) — a
/// detached conn thread still in `Scheduler.complete` would otherwise race
/// deinit's teardown of the slot queues into a use-after-free (SIGSEGV).
var active_conn_threads = std.atomic.Value(u32).init(0);
/// Set from main.zig before serve() is called when --metrics is on; null
/// otherwise. Gates the gauge-sampler thread and the /metrics + /metrics.json
/// routes. When null, `/metrics*` return 503 and the index page shows no panel.
pub var g_metrics: ?*instr.Metrics = null;
/// Optional global API key (`--api-key`). When set, every NON-LOOPBACK request
/// (i.e. from another machine over the network) except the `/health` probe and
/// CORS preflight requires the key — the OpenAI/Anthropic/Ollama APIs AND the
/// index page + metrics panel/feed. Loopback (127.0.0.0/8, ::1) is TRUSTED and
/// exempt, so the local app + a local browser keep working with no credentials;
/// the key protects the surface that actually matters — network exposure
/// (`--host 0.0.0.0`). Accepted as `Authorization: Bearer <key>`, `x-api-key:
/// <key>` (Anthropic), HTTP Basic (browser pages — the key is the password), or
/// an `?api_key=` / `?key=` query param. null ⇒ fully open (default, unchanged).
pub var g_api_key: ?[]const u8 = null;

/// Tool-call ARGUMENT auto-correct. When true (default), parsed tool-call
/// arguments are coerced to the types the request's tool schema declares
/// (`chat.coerceToolArgsToSchema` — e.g. Python's `False` → JSON `false`, an
/// `edits` array shipped as a string → a real array). `--no-tool-autocorrect`
/// turns it OFF: arguments pass through exactly as the model emitted them (still
/// valid JSON — the parse-repair + safety net always run; this gates ONLY the
/// schema-type coercion, the opinionated layer). Off means a weak model's
/// mistyped value reaches the client verbatim, which strict clients reject.
pub var g_tool_autocorrect: bool = true;

const io_util = @import("io_util.zig");
const ws_mod = @import("ws.zig");
const ollama_mod = @import("ollama.zig");
const cli_mod = @import("cli.zig");
const build_options = @import("build_options");
const nowSecs = io_util.nowSecs;
const nowMs = io_util.nowMs;
const nowMsMonotonic = io_util.nowMsMonotonic;
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

/// Tracks how long a streaming connection has been SILENT so the token loops
/// can emit a protocol-level keepalive before a client's idle-body timeout
/// fires. The mirror image of `generate.StallClock`: that one protects the
/// server from a wedged model, this one protects the client from a wedged-
/// looking socket.
///
/// It exists because every streaming surface buffers tokens while it might be
/// looking at a tool call or an unclosed thinking block — during a large
/// tool call (an agent one-shotting a whole file into `write_file`) the
/// handler produces no SSE output at all, sometimes for minutes. Node's
/// `fetch`/undici drops such a body after 300 s (`TypeError: terminated`).
pub const StreamHeartbeat = struct {
    /// MONOTONIC milliseconds (`io_util.nowMsMonotonic`) — an NTP step must not
    /// be able to stall the keepalive, nor spam one per token.
    last_write_ms: i64 = 0,
    interval_ms: i64 = Conn.STREAM_KEEPALIVE_MS,

    pub fn noteWrite(self: *StreamHeartbeat, now_ms: i64) void {
        self.last_write_ms = now_ms;
    }

    /// True once `interval_ms` has elapsed with no bytes handed to the socket.
    pub fn due(self: *const StreamHeartbeat, now_ms: i64) bool {
        return now_ms - self.last_write_ms >= self.interval_ms;
    }
};

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
    /// Silence tracker for the streaming keepalive. Stamped by every write
    /// path below, so it measures client-visible silence rather than "time
    /// since the last token" — a buffered tool call keeps producing tokens
    /// while emitting nothing. (`writeAllNoFlush` stamps too: every SSE emit
    /// path flushes before returning, so a stamp can't outrun the socket by
    /// more than one event.)
    heartbeat: StreamHeartbeat = .{},
    /// Non-null when this connection is bridged onto a WebSocket. Output that
    /// would otherwise be HTTP/SSE is reshaped into WS text frames at the
    /// `sendResponse` / `sendAnthropicEvent` chokepoints.
    ws_mode: ?*WsBridge = null,
    /// Non-null while an Ollama /api/* handler runs an inner /v1 handler:
    /// every write the inner handler makes is fed to the sink (SSE → NDJSON
    /// re-framing) instead of the socket. The sink writes its translated
    /// output through `writer()` directly, bypassing this hook — same
    /// interception pattern as `ws_mode`. See src/ollama.zig.
    ollama_sink: ?*ollama_mod.Sink = null,

    pub fn init(c: *Conn, stream: std.Io.net.Stream, io: std.Io) void {
        c.stream = stream;
        c.io = io;
        c.write_state = stream.writer(io, &c.write_buf);
        c.read_state = stream.reader(io, &c.read_buf);
        c.ws_mode = null;
        c.ollama_sink = null;
        c.heartbeat = .{ .last_write_ms = nowMsMonotonic(io) };
    }

    /// True when the connection has been silent long enough that a streaming
    /// keepalive is due.
    pub fn keepaliveDue(c: *Conn) bool {
        return c.heartbeat.due(nowMsMonotonic(c.io));
    }

    pub fn writer(c: *Conn) *std.Io.Writer {
        return &c.write_state.interface;
    }

    pub fn reader(c: *Conn) *std.Io.Reader {
        return &c.read_state.interface;
    }

    pub fn writeAll(c: *Conn, data: []const u8) !void {
        c.heartbeat.noteWrite(nowMsMonotonic(c.io));
        if (c.ollama_sink) |s| return s.feed(data);
        try c.writer().writeAll(data);
        try c.writer().flush();
    }

    pub fn writeAllNoFlush(c: *Conn, data: []const u8) !void {
        c.heartbeat.noteWrite(nowMsMonotonic(c.io));
        if (c.ollama_sink) |s| return s.feed(data);
        try c.writer().writeAll(data);
    }

    pub fn flush(c: *Conn) !void {
        c.heartbeat.noteWrite(nowMsMonotonic(c.io));
        if (c.ollama_sink != null) return;
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

    /// Non-blocking probe: has the peer closed the connection? TCP send
    /// buffers absorb hundreds of SSE writes after the client FIN/RST so
    /// `writeAll` only fails seconds late; this surfaces the disconnect
    /// promptly so a long-running decode can be cancelled before the GPU
    /// burns more cycles.
    ///
    /// Uses `poll(timeout=0)` for HUP/ERR, and a zero-byte `recv(MSG_PEEK)`
    /// to disambiguate `POLLIN`-with-data (still alive, client sending) from
    /// `POLLIN`-with-FIN (peer closed, `recv` returns 0). Stray bytes from a
    /// pipelined client are not expected on an SSE response stream — if we
    /// see them, we conservatively treat the connection as live.
    /// Idle interval (ms) between keepalive probes while a streaming
    /// request waits on its first tokens (prefill). Short enough that an
    /// abandoned prefill is noticed and cancelled quickly and that clients
    /// never hit stream-idle timeouts; tiny traffic cost.
    pub const STREAM_KEEPALIVE_MS: i64 = 5000;

    pub fn peerClosed(c: *Conn) bool {
        const fd = c.stream.socket.handle;
        var fds = [_]std.posix.pollfd{.{
            .fd = fd,
            .events = std.posix.POLL.IN,
            .revents = 0,
        }};
        const n = std.posix.poll(&fds, 0) catch return false;
        if (n == 0) return false;
        const revents = fds[0].revents;
        if ((revents & (std.posix.POLL.HUP | std.posix.POLL.ERR | std.posix.POLL.NVAL)) != 0) return true;
        if ((revents & std.posix.POLL.IN) == 0) return false;
        var peek_buf: [1]u8 = undefined;
        const flags: c_int = std.posix.MSG.PEEK | std.posix.MSG.DONTWAIT;
        const r = std.c.recv(fd, &peek_buf, peek_buf.len, flags);
        if (r == 0) return true; // FIN: peer closed cleanly
        return false; // negative (EAGAIN) or data available → assume alive
    }
};

fn signalHandler(_: std.posix.SIG) callconv(.c) void {
    shutdown_requested.store(true, .release);
}

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
/// "pure novel" (PLD overhead-only). Re-tune by sweeping `enable_pld` on/off
/// across an echo-heavy vs novel-content prompt set and picking the threshold
/// that maximizes (echo_speedup / novel_overhead).
const spec_gate_threshold: f32 = 0.01;

/// MTP-vs-PLD coexistence routing. When a model has BOTH a trained MTP head and
/// PLD available for a request, the DEFAULT is the MTP head — it holds ~93%+
/// per-draft on novel AND echo content and never risks PLD's runtime acceptance
/// gate degrading to plain decode. Only a DOMINANT-echo prompt (score >= this)
/// is handed to PLD, whose long n-gram drafts then reliably emit more tokens per
/// verify forward than the depth-1 head.
///
/// The bar is deliberately high. Measured Qwen3.6-27B 4-bit: at ngram-score
/// 0.055 PLD is a coin flip — a long whole-file reproduction sustains ~4.3
/// accepted/round (48 tok/s) but a shorter one collapses below the 50% runtime
/// gate and falls back to plain decode (~29 tok/s, WORSE than the head's 40).
/// The score is computed on the PROMPT and can't see which way the GENERATION
/// will go, so anything below a clearly-dominant-echo score routes to the robust
/// head (40-42 tok/s, reliable). User `enable_mtp` in the body overrides.

// Plan 05: drafter state moved to `LoadedModel` (per-model `drafter`,
// `drafter_block_size`, `drafter_path`). The previous module-level
// `default_drafter` / `default_enable_drafter` / `default_draft_block_size`
// / `global_drafter_path` singletons were removed; handlers now read
// these fields off the request's resolved `*LoadedModel` (`lm.drafter`,
// etc.). `default_enable_drafter` semantics (`drafter != null and !isMoe()`)
// is re-computed per-request as `lm.drafter != null and !config.isMoe()`.

/// Server-level configuration. Single source of truth for all process-wide
/// defaults that handlers might consult. Populated once by `serve()` from
/// its CLI args; read-only afterwards (no synchronization required, the
/// values don't change for the server's lifetime).
pub const ServerConfig = struct {
    /// Maximum context size (0 = unlimited). `--ctx-size N`.
    max_context_size: u32 = 0,
    /// Request timeout in seconds (0 = no timeout). `--timeout N`.
    request_timeout_sec: u32 = 300,
    /// Default reasoning budget in tokens (-1 = unlimited).
    /// `--reasoning-budget N`. Per-request body fields override.
    default_reasoning_budget: i32 = -1,
    /// Default PLD enabled state. Per-request `enable_pld` JSON overrides.
    default_enable_pld: bool = false,
    /// Maximum draft tokens proposed per PLD step.
    default_pld_draft_len: u32 = 5,
    /// N-gram match key length for PLD.
    default_pld_key_len: u32 = 3,
    /// Phase 2 (Plan ricky): default `kv_attn_fused` for new requests.
    /// Set by `--kv-attn-mode fused`. Per-request `kv_attn_mode` body
    /// field overrides. Only takes effect at `--kv-quant 4|8` (.affine
    /// cache scheme); other schemes ignore it.
    default_kv_attn_fused: bool = false,
    /// Defaults for sampling fields the request OMITS, set by `--temp` /
    /// `--top-p` / `--top-k` in serve mode (the macOS app passes its Settings
    /// values so external clients like Claude Code — which send no sampling
    /// params at all — inherit them). null = flag not given; resolution falls
    /// through to the model's generation_config.json recommendation, then the
    /// hardcoded fallback. Explicit request body fields always win. See
    /// `resolveSamplingDefault`.
    default_temperature: ?f32 = null,
    default_top_p: ?f32 = null,
    default_top_k: ?u32 = null,
};

/// Sampling-default resolution chain: request body > CLI launch flag >
/// model generation_config.json > hardcoded fallback. An explicit request
/// value of 0 (greedy / disabled) is a value, not an omission.
fn resolveSamplingDefault(comptime T: type, request: ?T, cli: ?T, gen_config: ?T, fallback: T) T {
    return request orelse cli orelse gen_config orelse fallback;
}

/// parseJsonFloat variant that distinguishes "omitted / wrong type" (null)
/// from an explicit value, for fields whose default comes from the
/// resolution chain above.
fn parseJsonFloatOpt(root: std.json.ObjectMap, key: []const u8, min: f32, max: f32) ?f32 {
    const v = root.get(key) orelse return null;
    const raw: f32 = switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => return null,
    };
    return std.math.clamp(raw, min, max);
}

/// Optional top_k body-field parse (positive integer, capped at 1000).
/// null = omitted or unusable; explicit 0 means "disable top-k".
fn parseJsonTopKOpt(root: std.json.ObjectMap, key: []const u8) ?u32 {
    const v = root.get(key) orelse return null;
    return switch (v) {
        .integer => |i| if (i > 0) @intCast(@min(i, 1000)) else 0,
        .float => |f| if (f > 0) @intFromFloat(@min(f, 1000)) else 0,
        else => null,
    };
}

/// Live process-wide config. Mutated by `serve()` at startup from its
/// parameters; never written from a handler after that. Public so main.zig
/// can supply defaults (notably the per-request override CLI flags don't
/// flow through `serve()` arguments).
pub var server_config: ServerConfig = .{};

fn getTimeoutNs() u64 {
    if (server_config.request_timeout_sec == 0) return 0;
    return @as(u64, server_config.request_timeout_sec) * std.time.ns_per_s;
}

/// Plan 01 Phase 2 — continuous-batching scheduler. Always non-null in
/// serve mode (set by `serve()`); null only before `serve()` runs. Every
/// inference request handler routes through this; the scheduler's
/// inference thread is the single mlx-call site.
var global_scheduler: ?*scheduler_mod.Scheduler = null;

/// Plan 05 — model registry. Always non-null in serve mode (set by
/// `serve()`); handleConnection resolves `model` body fields against this
/// per request via `ensureLoaded`/`release`.
var global_registry: ?*ModelRegistry = null;

/// Plan 05 — extract the `"model":"..."` value from a JSON request body
/// without doing a full parse. Returns a borrowed slice into `body` (valid
/// until the body buffer is freed) or null if the field is missing or
/// malformed. Used by `handleConnection` to route each POST to the right
/// LoadedModel before invoking the handler.
///
/// Cheap (single linear scan, ~tens of nanoseconds for typical bodies);
/// the per-handler full JSON parse runs immediately after and rejects any
/// truly malformed body, so robustness here is fine.
pub fn parseModelFromBody(body: []const u8) ?[]const u8 {
    var idx: usize = 0;
    while (idx < body.len) {
        const key_start = std.mem.indexOfPos(u8, body, idx, "\"model\"") orelse return null;
        // Must be JSON object key: preceded by `{` or `,` (skipping whitespace).
        var prev = key_start;
        while (prev > 0) {
            prev -= 1;
            const c = body[prev];
            if (c == ' ' or c == '\t' or c == '\n' or c == '\r') continue;
            if (c == '{' or c == ',') break;
            // Not a top-level key — keep searching after this match.
            idx = key_start + 1;
            return parseModelFromBody(body[idx..]);
        }
        // Skip past the key + the colon.
        var pos = key_start + "\"model\"".len;
        while (pos < body.len and (body[pos] == ' ' or body[pos] == '\t')) pos += 1;
        if (pos >= body.len or body[pos] != ':') return null;
        pos += 1;
        while (pos < body.len and (body[pos] == ' ' or body[pos] == '\t' or body[pos] == '\n' or body[pos] == '\r')) pos += 1;
        if (pos >= body.len) return null;
        // Accept "string" only — null/numbers/etc fall back to default.
        if (body[pos] != '"') return null;
        pos += 1;
        const val_start = pos;
        while (pos < body.len and body[pos] != '"') {
            if (body[pos] == '\\' and pos + 1 < body.len) pos += 1; // skip escape
            pos += 1;
        }
        if (pos >= body.len) return null;
        return body[val_start..pos];
    }
    return null;
}

/// Module-level capacity for plan 03 hot prefix cache. main.zig writes this
/// from `--prefix-cache-entries N` before calling `serve()`.
// Default 32: agent clients (Claude Code, the app's agent loop) interleave
// requests from several conversation roots — main thread, subagents, title
// generation — and a single-entry cache gets its long system-prompt prefix
// evicted by every interleaved request, forcing a full re-prefill each
// turn. The count cap is pure retention metadata; the byte budget
// (`--prefix-cache-mem`, default 2 GB) is what actually bounds memory,
// evicting LRU entries by size. 0 disables.
pub var prefix_cache_capacity: u32 = 32;

/// Wave 1.B — hot prefix cache memory budget. The cache evicts LRU entries on
/// commit until `current_kv_bytes + new_bytes <= prefix_cache_mem_bytes`. The
/// default (2 GB) is generous for one or two long conversations on a Gemma 4
/// E4B-sized model and tiny relative to total wired-limit budget; tune via
/// `--prefix-cache-mem <N>{GB,MB}`. 0 disables the byte budget (count cap
/// from `--prefix-cache-entries` still applies).
pub var prefix_cache_mem_bytes: u64 = 2 * 1024 * 1024 * 1024;

/// SSD tier for the hot prefix cache (`--prefix-cache-disk`). Committed KV
/// prefixes persist as chunked safetensors under
/// `~/.mlx-serve/kv-cache/<model-fingerprint>` and are restored across RAM
/// evictions AND server restarts instead of recomputed — a cold 30-50 s
/// long-context TTFT becomes a bounded SSD read. LRU-evicted to this byte
/// budget. v1 covers pure-attention archs; hybrid SSM state stays RAM-only.
///
/// DEFAULT OFF (`0`): the tier can hold gigabytes of KV on disk, so it's opt-in
/// — pass `--prefix-cache-disk <n>{KB,MB,GB}` to enable (10 GB is a sensible
/// value). The Swift app exposes this as a Settings toggle, default off.
pub var prefix_cache_disk_bytes: u64 = 0;

/// Phase 1 (performance-plan): SSM/conv state snapshot stride during prefill,
/// in tokens. Non-zero values enable multi-turn warm reuse on hybrid SSM
/// architectures (Qwen3.5/3.6 GatedDeltaNet, Nemotron-H Mamba2, LFM2.5
/// gated-conv). Set to 0 to disable (fall back to pre-Phase-1 behavior:
/// hybrid models bypass the hot prefix cache entirely). Override via
/// `--ssm-checkpoint-stride N`.
///
/// Default 256. The stride forces a prefill CHUNK boundary at every multiple
/// (to snapshot SSM/sliding-window state mid-prompt). On dense / non-MoE-hybrid
/// models (dense Gemma sliding-window, LFM2, Nemotron-H) prefill is
/// compute-bound, so a fine 256 stride is ~free and buys finer warm mid-prompt
/// reuse — measured <3% cold cost on Qwen3.5-4B. **MoE models are different**:
/// their prefill is memory-bound on the per-expert weights, and every extra
/// chunk re-streams ~all expert weights from HBM, so a fine stride silently
/// costs ~25% cold prefill on 26B/35B-class MoE (an 850-token prompt = 4
/// chunks = ~4x expert-weight traffic). To avoid that, the prefill loop
/// **coarsens this stride to >= PREFILL_CHUNK for MoE models only** (see
/// `generate.effectiveSsmCheckpointStride`), so MoE prefill is never
/// over-chunked at any prompt length while non-MoE keeps the fine stride.
/// The always-on end-of-prompt snapshot preserves append-growth multi-turn
/// reuse for MoE regardless. Raise this (e.g. 512/1024) to also coarsen the
/// non-MoE path; set 0 to disable (hybrid models then bypass the hot cache).
pub var ssm_checkpoint_stride: u32 = 256;

/// Phase 1: cap on the number of checkpoints retained per request. Snapshots
/// past this drop the oldest (front of list) to keep memory bounded on very
/// long prompts. 0 = unlimited (rely on the prefix-cache byte budget alone).
pub var ssm_checkpoint_max: u32 = 32;

/// SSM prefill checkpoints exist ONLY to feed the hot prefix cache (RAM +
/// disk tiers key their hybrid restores off them). With the cache disabled
/// (`--prefix-cache-entries 0`) every capture is a 48-layer materialize +
/// eval thrown straight away — measured ~2-4% of short-prompt prefill on
/// Qwen3.6-27B. Single chokepoint for every LoadParams builder.
pub fn effectiveSsmCheckpointStride(stride: u32, cache_capacity: u32) u32 {
    if (cache_capacity == 0) return 0;
    return stride;
}

test "effectiveSsmCheckpointStride: disabled prefix cache disables checkpoint capture" {
    try std.testing.expectEqual(@as(u32, 0), effectiveSsmCheckpointStride(256, 0));
    try std.testing.expectEqual(@as(u32, 256), effectiveSsmCheckpointStride(256, 32));
    try std.testing.expectEqual(@as(u32, 0), effectiveSsmCheckpointStride(0, 32));
}

/// Iteration 2 (perf-plan Phase 4 #3): LRU capacity of the per-LoadedModel
/// chat-template tokenize cache. 0 disables (every request re-renders +
/// re-tokenizes, restoring pre-Iteration-2 behavior). Default 4 is small
/// because real chat conversations mutate the messages list every turn;
/// the cache catches warm-reuse benches + repeated agent probes without
/// hoarding token buffers across a long session.
pub var tokenize_cache_entries: u32 = 4;

/// Image-generation NSFW content filter (Krea 2 Community License §4.2). ON by
/// default; `--no-safety` disables it process-wide, or a request can opt out with
/// `"safety": false`. The classifier (Falconsai ViT, src/nsfw.zig) is lazy-loaded
/// from `~/.mlx-serve/models`; if it's unavailable the server fails OPEN (warns +
/// passes the image through).
pub var image_safety_filter: bool = true;

/// Iteration 3-5 (perf-plan Phase 5 #1): cap on resident llama.cpp KV
/// sessions per loaded GGUF model. 1 is the legacy single-session
/// behavior (a flip between two long-doc prompts evicts the other on
/// every turn — and even sequential shared-prefix requests reported
/// cached_tokens=0). N > 1 enables the best-prefix-match LRU so
/// alternating multi-doc / agent workloads stay warm. Sessions are
/// created lazily, so unused slots cost nothing.
pub var llama_cache_entries: u32 = 4;

/// Phase 5 (performance-plan) #2: KV-cache quantization for the embedded
/// llama.cpp engine. `off` = F16 (libllama default); `q8` halves the KV
/// bytes (Q8_0, near-lossless); `q4` quarters them (Q4_0, some quality
/// impact). Non-default settings automatically enable flash attention in
/// the shim because llama.cpp's plain SDPA only supports F16/F32 KV.
/// Set via `--llama-kv-quant {off,q8,q4}`. Applies to every llama.cpp
/// session created after this is set (i.e., from the next model load).
pub var llama_kv_quant: arch_llama.LlamaKvQuant = .off;

/// Plan 01 — continuous batching: maximum concurrent in-flight requests sharing
/// the inference thread's batched-decode pass. Set via `--max-concurrent N`.
/// Default 1 = legacy single-slot behavior (no scheduler engagement, every
/// existing test bit-identical). Values >1 require Phase 2's scheduler wiring
/// (handler refactor + per-connection threads), which the `serve()` startup
/// guard checks before enabling.
pub var max_concurrent: u32 = 1;

// Plan 05: vision encoder and model id moved to `LoadedModel.vision_encoder`
// and `LoadedModel.id`. Handlers read them off `lm`. `global_vision_encoder`
// and `global_model_id` singletons were removed. The `discovered_models`
// slice was also removed — `/v1/models` iterates `registry.entries` directly.

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

/// Decode a slice of token IDs to bytes, routing through the ds4 engine when
/// the loaded model is GGUF-backed (no MLX tokenizer in that case). Used by
/// the request handlers' streaming + final-decode paths so a single call
/// site supports both backends.
fn decodeTokens(
    allocator: std.mem.Allocator,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    ids: []const u32,
    strip_leading_space: bool,
) ![]u8 {
    if (lm.ds4_engine) |engine| {
        return chat_mod.decodeViaDs4(allocator, engine, ids);
    }
    if (lm.llama_engine) |engine| {
        return chat_mod.decodeViaLlama(allocator, engine, ids);
    }
    return tok.decode(allocator, ids, strip_leading_space);
}

/// True when the rendered generation prompt ends inside a template-opened
/// think block (Qwen 3.5/3.6 render `…assistant\n<think>\n` when thinking is
/// on). Decodes the last few prompt tokens — cheap, engine-agnostic, and
/// independent of the tokenize cache. Drives the unclosed-think split policy
/// in `chat.splitThinkBlock` so a length-truncated thought never leaks into
/// visible content.
fn promptOpensThink(
    allocator: std.mem.Allocator,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
) bool {
    if (prompt_ids.len == 0) return false;
    const n = @min(prompt_ids.len, 8);
    const tail = decodeTokens(allocator, lm, tok, prompt_ids[prompt_ids.len - n ..], false) catch return false;
    defer allocator.free(tail);
    return chat_mod.promptTailOpensThink(tail);
}

/// In-memory store for OpenAI Responses API state (`store: true` requests).
/// Bounded LRU; lost on restart.
var global_response_store: ?responses_mod.ResponseStore = null;
var global_response_store_gpa: ?std.mem.Allocator = null;
const RESPONSE_STORE_CAP: usize = 256;
const DEFAULT_STRUCTURED_OUTPUT_MAX_TOKENS: u32 = 2048;

/// Default when an OpenAI-style client omits `max_tokens` /
/// `max_completion_tokens` / `max_output_tokens`. OpenAI semantics: omitted
/// means "generate until EOS, bounded by the context window" — NOT a small
/// fixed cap. (The old 256 default silently broke agent clients like pi:
/// every thinking-enabled turn hit `length` mid-reasoning, so no tool call or
/// answer ever came back.) The sentinel is huge so the downstream
/// `clampMaxTokens` resolves it to the remaining context; only when the context
/// is genuinely unknown (0) do we fall back to a finite 4096.
///
/// Takes the EFFECTIVE context, not `server_config.max_context_size`: that
/// global is set only by `--ctx-size`, so under auto-context this used to
/// return 4096 — an omitted-max_tokens client silently capped at 4096 tokens,
/// truncating any large tool call. Same class as the 256 default it replaced.
fn omittedMaxTokensDefault(effective_ctx: u32) u32 {
    return if (effective_ctx > 0) std.math.maxInt(u32) / 4 else 4096;
}

/// Resolve a request's `max_tokens` (or its aliases) to an effective cap.
/// Absent OR `<= 0` means **auto**: peg generation to the remaining context
/// window via `auto_default` (the `omittedMaxTokensDefault` sentinel, which
/// `clampMaxTokens` then reduces to `context - prompt`). The app's "Auto"
/// setting sends 0 (or omits the field); both land on the context-pegged budget
/// instead of a fixed ceiling. A positive integer is an explicit cap.
fn resolveRequestMaxTokens(v: ?std.json.Value, auto_default: u32) u32 {
    const val = v orelse return auto_default;
    return switch (val) {
        .integer => |i| if (i > 0) @intCast(i) else auto_default,
        else => auto_default,
    };
}

/// Finish reason for a response whose generated text yielded tool calls.
/// A TRUNCATED generation ("length": max_tokens or the request timeout) must
/// keep reporting "length" even when the cut-off text salvaged a partial tool
/// call (the truncated-opener recovery in chat.parseToolCalls) — clients key
/// their truncation recovery on it (the app's chunk-and-retry nudge fires on
/// finish_reason "length" + tool calls present). Overriding to "tool_calls"
/// hides the cut: the client executes a half-argument call, blames the model
/// ("you sent no content"), and the model re-emits the same doomed mega-call.
fn toolCallFinishReason(pre_parse: []const u8) []const u8 {
    return if (std.mem.eql(u8, pre_parse, "length")) "length" else "tool_calls";
}

/// Debug-only corpus-harvest aid (`MLX_SERVE_RAW_DUMP_FILE`). Appends ONE framed
/// record per tools request: the DECLARED TOOLS SCHEMA and the raw pre-parse text
/// together, written at the one site where both are in scope. Correlating them
/// any other way is unsound — the debug log truncates lines at 16 KB (so a large
/// request body never re-parses, and a harvester silently pairs a model output
/// with some EARLIER request's schema), and concurrent requests interleave.
///
/// Framing is by byte COUNT, never a delimiter: both payloads are arbitrary bytes
/// and any sentinel can occur inside them. libc `write(2)` like log.zig's sink
/// (callers here carry no `Io`); O_APPEND keeps conn threads from interleaving.
fn appendRawToolDump(path: []const u8, tools_json: ?[]const u8, text: []const u8) void {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len == 0 or path.len >= path_buf.len) return;
    @memcpy(path_buf[0..path.len], path);
    path_buf[path.len] = 0;
    const path_z: [*:0]const u8 = @ptrCast(&path_buf);

    const fd = std.c.open(path_z, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return;
    defer _ = std.c.close(fd);

    const tools = tools_json orelse "";
    var hdr: [96]u8 = undefined;
    const h = std.fmt.bufPrint(&hdr, "\n===MLX_RAW_DUMP tools={d} raw={d}===\n", .{ tools.len, text.len }) catch return;
    _ = std.c.write(fd, h.ptr, h.len);
    if (tools.len > 0) _ = std.c.write(fd, tools.ptr, tools.len);
    if (text.len > 0) _ = std.c.write(fd, text.ptr, text.len);
}

/// The ONE tool-call extraction path for every HTTP surface: parse, fall back to
/// bare-JSON inference, then coerce the arguments to the schema the request
/// declared. Every surface must go through this — a site that calls
/// `chat_mod.parseToolCalls` directly silently reintroduces the value-spelling
/// type-inference bug (a boolean param arriving as the string "False") on that
/// surface alone, which no output-equality test can see.
fn parseToolCallsForRequest(
    allocator: std.mem.Allocator,
    text: []const u8,
    tools_json: ?[]const u8,
) !?[]chat_mod.ParsedToolCall {
    var parsed_calls = try chat_mod.parseToolCalls(allocator, text);
    // Heuristically-inferred raw-JSON calls must name a DECLARED tool — a
    // truncated data object ({"name": "George Washington", …}) is not a call.
    // Deliberately NOT gated on g_tool_autocorrect: this corrects our own
    // heuristic's false positive, not the model's output.
    if (parsed_calls) |c| {
        if (tools_json) |tj| parsed_calls = try chat_mod.filterInferredBySchema(allocator, c, tj);
    }
    const calls = parsed_calls orelse
        (if (tools_json) |tj| try chat_mod.inferBareJsonToolCalls(allocator, text, tj) else null) orelse
        return null;
    if (g_tool_autocorrect) {
        if (tools_json) |tj| try chat_mod.coerceToolArgsToSchema(allocator, calls, tj);
    }
    return calls;
}

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
///
/// `cfg` carries all process-wide defaults (context size, timeouts, PLD
/// defaults, reasoning budget). It's copied into the module-level
/// `server_config` before the listen loop starts.
pub fn serve(
    io: std.Io,
    allocator: std.mem.Allocator,
    /// Phase A1: model load happens on the scheduler's inference thread. The
    /// caller (main.zig) is responsible only for CPU-side setup (config parse,
    /// tokenizer load, EOS resolution, chat-template load); everything that
    /// touches mlx is done by `Scheduler.init` before this fn proceeds.
    load_params: scheduler_mod.LoadParams,
    config: *const model_mod.ModelConfig,
    host: []const u8,
    port: u16,
    cfg: ServerConfig,
) !void {
    server_config = cfg;

    // ── Phase A1: spin up the scheduler. Its inference thread does the
    //    Transformer/vision/drafter load, JIT compile, and warmup before
    //    `Scheduler.init` returns. From this point on, mlx ops are bound
    //    to the inference thread's GPU stream.
    var scheduler = try scheduler_mod.Scheduler.init(
        allocator,
        io,
        load_params,
        max_concurrent,
    );
    defer scheduler.deinit();
    global_scheduler = scheduler;
    defer global_scheduler = null;

    // Gauge sampler: samples instantaneous system + queue state every 2 s and
    // writes the metrics gauges. Only runs when --metrics is on. Trivial cost
    // (a few non-blocking syscalls + one brief queue_mu lock per 2 s); never on
    // the per-token decode path.
    // LIFO defer ordering: `stop` is registered AFTER `join` so it runs FIRST
    // (setting the flag), letting the thread exit before join() blocks. Do NOT
    // gate on scheduler.shutdown here — deinit() (which sets that flag) is the
    // earliest defer and runs LAST, after the join, so the loop would never see
    // it; the dedicated `sampler_stop` flag avoids that LIFO deadlock.
    var sampler_thread: ?std.Thread = null;
    var sampler_stop = std.atomic.Value(bool).init(false);
    if (g_metrics) |m| {
        const ctx = GaugeSamplerCtx{
            .metrics = m,
            .scheduler = scheduler,
            .stop = &sampler_stop,
        };
        sampler_thread = try std.Thread.spawn(.{}, gaugeSamplerLoop, .{ctx});
        log.info("Prometheus metrics: ENABLED — GET http://{s}:{d}/metrics (live panel at GET /)\n", .{ host, port });
    }
    // LIFO: join deferred first → runs second. stop deferred second → runs first.
    defer if (sampler_thread) |t| t.join();
    defer sampler_stop.store(true, .monotonic);

    global_registry = scheduler.registry;
    defer global_registry = null;

    // Plan 05: the hot prefix cache lives on the LoadedModel
    // (entry.prefix_cache) and is set up by `loadModelOnInferenceThread`
    // using the per-LoadParams capacity + byte budget. Surface a friendly
    // log line so users see whether the cache engaged.
    if (scheduler.hot_prefix_cache != null) {
        const ssm_note: []const u8 = if (config.has_hybrid_layers)
            " [hybrid: SSM checkpoints]"
        else
            "";
        if (prefix_cache_mem_bytes > 0) {
            const cap_mb = @as(f64, @floatFromInt(prefix_cache_mem_bytes)) / (1024.0 * 1024.0);
            log.info("Hot prefix cache: ENABLED (capacity={d}, mem-cap={d:.1} MB){s}\n", .{ prefix_cache_capacity, cap_mb, ssm_note });
        } else {
            log.info("Hot prefix cache: ENABLED (capacity={d}, mem-cap=unlimited){s}\n", .{ prefix_cache_capacity, ssm_note });
        }
        if (config.has_hybrid_layers) {
            log.info("  ssm-checkpoint-stride={d} tokens, max={d}/entry\n", .{ ssm_checkpoint_stride, ssm_checkpoint_max });
        }
    } else if (prefix_cache_capacity > 0) {
        if (config.has_hybrid_layers and ssm_checkpoint_stride == 0) {
            log.info("Hot prefix cache: requested capacity={d} but model is hybrid and --ssm-checkpoint-stride is 0 — disabled\n", .{prefix_cache_capacity});
        } else if (config.full_attention_interval > 0) {
            log.info("Hot prefix cache: requested capacity={d} but model uses full-attention-interval — disabled\n", .{prefix_cache_capacity});
        } else {
            log.info("Hot prefix cache: requested capacity={d} but disabled by scheduler\n", .{prefix_cache_capacity});
        }
    }

    // `--max-concurrent N` is honored when the model can ride the batched
    // decode kernel (pure-attention, no SSM/MoE/encoder). Hybrid / MoE /
    // encoder architectures clamp to 1 — they need single-slot serial
    // because the batched kernel doesn't model their state. DSV4 is
    // honored (per-slot LatentKVCache landed in Plan 04 Section 4
    // Phase A+B; Section B fix 2026-05-13 excludes DSV4 from
    // `Scheduler.batchable` so two DSV4 slots fall through to per-slot
    // `runSingleDecodeTick` — sequential through the inference thread,
    // safe).
    if (max_concurrent > 1) {
        if (config.has_hybrid_layers or config.full_attention_interval > 0 or config.is_encoder_only or config.isMoe()) {
            log.info("Concurrency: requested {d} but model is hybrid/MoE/encoder; falling back to 1\n", .{max_concurrent});
            max_concurrent = 1;
        } else {
            log.info("Concurrency: --max-concurrent={d} (continuous batching enabled)\n", .{max_concurrent});
            if (prefix_cache_capacity < max_concurrent) prefix_cache_capacity = max_concurrent;
        }
    }
    global_port = port;
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

    // Freeze the auto-context NOW, at startup, while the model is freshly
    // loaded and nothing else has taken RAM. Clients read this number once
    // (pi/opencode bake it into a config file) and budget against it for the
    // whole session, so it must not drift with system load. `--ctx-size` wins.
    const pinned = pinAutoContext(@constCast(config));
    if (server_config.max_context_size > 0) {
        log.info("Context size: {d} tokens (manual)\n", .{server_config.max_context_size});
    } else {
        const memory_ctx = computeMemoryContext(config);
        const memory_allows = safeAutoContext(memory_ctx);
        if (config.max_position_embeddings > 0 and pinned >= config.max_position_embeddings) {
            // The checkpoint's own maximum binds; memory had room to spare.
            log.info("Context size: {d} tokens (auto: the model's maximum; memory would allow {d}) [pinned]\n", .{ pinned, memory_allows });
        } else {
            log.info("Context size: {d} tokens (auto: {d}% of the {d}-token memory ceiling, reserving headroom) [pinned]\n", .{ pinned, auto_ctx_safety_pct, memory_ctx });
        }
    }

    if (server_config.request_timeout_sec > 0) {
        log.info("Request timeout: {d}s\n", .{server_config.request_timeout_sec});
    }
    const model_ctx = config.max_position_embeddings;
    if (model_ctx > 0) {
        log.info("Model context length: {d} tokens\n", .{model_ctx});
    }
    if (server_config.default_reasoning_budget >= 0) {
        log.info("Reasoning budget: {d} tokens\n", .{server_config.default_reasoning_budget});
    } else {
        log.info("Reasoning budget: unlimited\n", .{});
    }
    if (server_config.default_enable_pld) {
        log.info("PLD speculative decoding: ENABLED (draft_len={d}, key_len={d}; default for new requests)\n", .{ server_config.default_pld_draft_len, server_config.default_pld_key_len });
    }
    if (scheduler.drafter != null) {
        log.info("Drafter speculative decoding: ENABLED (block_size={d}; default for new requests)\n", .{scheduler.drafter_block_size});
    }
    log.info("\nServer listening on http://{s}:{d}\n", .{ host, port });
    if (g_api_key != null) {
        log.info("API key auth: ENABLED for non-loopback requests (localhost is trusted; /health stays open)\n", .{});
    }
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

        // When the scheduler is engaged AND the model is pure-attention,
        // Spawn a per-connection thread so HTTP I/O for one request can
        // overlap with another request's generation. mlx ops (forward
        // passes, eval, etc.) live exclusively on the scheduler's inference
        // thread, so the connection thread NEVER touches `xfm.s` directly —
        // it parses HTTP, encodes the prompt, calls `scheduler.submit`,
        // reads tokens via `slot.waitNext`, and writes the response.
        const args = allocator.create(ConnThreadArgs) catch |err| {
            log.err("conn thread args alloc failed: {}\n", .{err});
            accepted_stream.close(io);
            continue;
        };
        args.* = .{
            .allocator = allocator,
            .accepted_stream = accepted_stream,
            .io = io,
        };
        // Count the thread before spawning so the shutdown drain can't miss a
        // thread that's about to start; the thread decrements on return.
        _ = active_conn_threads.fetchAdd(1, .acq_rel);
        _ = std.Thread.spawn(.{}, handleConnectionThread, .{args}) catch |err| {
            log.err("spawn conn thread failed: {}\n", .{err});
            _ = active_conn_threads.fetchSub(1, .acq_rel);
            accepted_stream.close(io);
            allocator.destroy(args);
            continue;
        };
    }

    // Shutdown ordering (prevents a SIGSEGV in Scheduler.complete): the accept
    // loop has exited. Cancel every in-flight slot so its connection thread
    // unblocks from waitNext and runs its `defer complete(...)`, then WAIT for
    // all connection threads to finish before returning — `serve`'s caller runs
    // `scheduler.deinit` on return, which frees the slot queues that an
    // in-flight `complete()` is still touching. The inference thread is still
    // alive here (deinit joins it only after we return), so cancelled slots
    // settle promptly. Bounded so a wedged thread can't hang shutdown forever.
    if (global_scheduler) |sch| sch.cancelAllInFlight();
    {
        var waited_ms: u64 = 0;
        var idle_fds = [_]std.posix.pollfd{};
        while (active_conn_threads.load(.acquire) > 0 and waited_ms < 30_000) {
            _ = std.posix.poll(&idle_fds, 20) catch {}; // empty fd set → 20ms sleep
            waited_ms += 20;
        }
        const remaining = active_conn_threads.load(.acquire);
        if (remaining > 0)
            log.warn("shutdown: {d} connection thread(s) still active after {d}ms — proceeding\n", .{ remaining, waited_ms });
    }

    deinitGlobalResponseStore();
    deinitGlobalTokenBytes();

    log.info("\nShutting down gracefully...\n", .{});
}

/// Per-connection thread arguments. Heap-allocated so the spawned thread
/// owns its lifetime; freed in `handleConnectionThread` after the handler
/// returns. The accepted stream is moved into a stack-allocated `Conn`
/// inside the thread.
const ConnThreadArgs = struct {
    allocator: std.mem.Allocator,
    accepted_stream: std.Io.net.Stream,
    io: std.Io,
};

fn handleConnectionThread(args: *ConnThreadArgs) void {
    // Decrement LAST (declared first → runs after every other defer), so the
    // shutdown drain in `serve` only sees this thread leave once it has fully
    // returned from `complete()` and closed its socket.
    defer _ = active_conn_threads.fetchSub(1, .acq_rel);
    var conn: Conn = undefined;
    Conn.init(&conn, args.accepted_stream, args.io);
    defer conn.close();
    handleConnection(args.allocator, &conn) catch |err| {
        switch (err) {
            error.WriteFailed, error.ReadFailed => {
                // error.WriteFailed/ReadFailed collapses BrokenPipe + ConnectionResetByPeer
                // + other low-level errors. Surface the actual cause from write_state.err /
                // read_state.err so debug logs distinguish "client hung up" from real bugs.
                if (err == error.WriteFailed) {
                    if (conn.write_state.err) |we| {
                        log.debug("  -> client disconnected (write: {s})\n", .{@errorName(we)});
                    } else {
                        log.debug("  -> client disconnected (write)\n", .{});
                    }
                } else {
                    if (conn.read_state.err) |re| {
                        log.debug("  -> client disconnected (read: {s})\n", .{@errorName(re)});
                    } else {
                        log.debug("  -> client disconnected (read)\n", .{});
                    }
                }
            },
            else => log.err("  -> error: {}\n", .{err}),
        }
    };
    args.allocator.destroy(args);
}

fn handleConnection(
    allocator: std.mem.Allocator,
    stream: *Conn,
) !void {
    // Plan 05: resolve which model this request targets. The registry was
    // set up in `serve()`; per-POST routing happens after we read the body
    // and parse out the optional `"model"` field. For Phase C we use the
    // default model for everything (only one is loaded), but the plumbing
    // is in place for Phase D's hot-load path.
    const registry = global_registry orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Server not ready", 503);
        return;
    };
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

    // ── API-key auth gate. When --api-key is set, every NON-LOOPBACK request
    //    requires the key (the OpenAI/Anthropic/Ollama APIs AND the index page
    //    + metrics panel/feed) — EXCEPT `/health` and CORS preflight, which
    //    load balancers and browsers must reach unauthenticated. Loopback is
    //    trusted (the local app connects via 127.0.0.1), so it's exempt: the app
    //    + a local browser never need credentials, and the key protects network
    //    exposure. Browser-facing pages get a `WWW-Authenticate: Basic`
    //    challenge so the index + metrics panel prompt for the key; API clients
    //    pass Bearer / x-api-key. null key ⇒ fully open. See `apiKeyAuthorized`.
    if (g_api_key != null and
        !peerIsLoopback(stream) and
        !std.mem.eql(u8, method, "OPTIONS") and
        !(std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/health")) and
        !apiKeyAuthorized(request[0..header_end_pos], raw_path))
    {
        log.debug("{s} {s} -> 401 (missing/invalid API key)\n", .{ method, path });
        try sendUnauthorized(stream);
        return;
    }

    // ── Plan 05: routes that don't depend on a loaded model (connectivity
    //    probes + CORS preflight + listing endpoints). Handle these BEFORE
    //    `scheduler.ensureLoaded` so they don't trigger a cold load of the
    //    default model just to read metadata. `/v1/models` and the GET-side
    //    of the Responses API are pure registry/store reads — no model
    //    needed.
    if (std.mem.eql(u8, method, "HEAD") and std.mem.eql(u8, path, "/")) {
        log.debug("HEAD / -> 200\n", .{});
        try sendResponse(stream, "200 OK", "text/plain", "");
        return;
    }
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/health")) {
        log.debug("GET  /health -> 200\n", .{});
        try sendResponse(stream, "200 OK", "application/json", "{\"status\":\"ok\"}");
        return;
    }
    // Prometheus scrape endpoint. 503 when --metrics is off. Behind the global
    // API-key gate above when --api-key is set (auth already enforced here).
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/metrics")) {
        if (g_metrics) |m| {
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
    // JSON feed — drives the live metrics panel on the index page. Behind the
    // global API-key gate above when --api-key is set (same-origin browser
    // fetch inherits the page's Basic credentials).
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/metrics.json")) {
        if (g_metrics) |m| {
            var out: std.Io.Writer.Allocating = .init(allocator);
            defer out.deinit();
            try instr.renderJson(m, &out.writer);
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
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/v1/models")) {
        log.debug("GET  /v1/models -> 200\n", .{});
        try handleModels(allocator, stream);
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
    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/responses/compact")) {
        // Compaction is a pure data transformation — no model load needed.
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleResponsesCompact(allocator, stream, body);
        return;
    }
    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/load-model")) {
        // Phase E: explicit cold-load. Keep strict semantics — unknown ids
        // get a 404 here even though the main dispatch path below falls
        // back to default for SDK compatibility.
        log.debug("POST /v1/load-model -> 200\n", .{});
        try handleLoadModelStrict(allocator, stream, request_body);
        return;
    }
    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/unload-model")) {
        // Free a model's resident GPU state (the app's load→generate→unload
        // default). The stub stays registered so it can reload. Idempotent.
        log.debug("POST /v1/unload-model -> 200\n", .{});
        try handleUnloadModelStrict(allocator, stream, request_body);
        return;
    }

    // ── Ollama-compatible API (/api/*): endpoints that must not trigger a
    //    model load (version/tags/ps/show/pull/unsupported) are handled
    //    here; /api/chat, /api/generate and /api/embed(dings) fall through
    //    to the model-resolution path below. Glue lives at the bottom of
    //    this file; pure translation in src/ollama.zig.
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
    const scheduler = global_scheduler orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Scheduler not ready", 503);
        return;
    };
    // Strip whatever the client passed in `"model":"..."` — except when the
    // id literally matches one we've discovered, in which case we honor it
    // and route. The OpenAI / Anthropic ecosystem commonly sends marketing
    // names like "gpt-4" or "claude-opus-4-x" expecting the local server to
    // just respond with whatever it has loaded; the multi-model registry's
    // strict-id semantics are opt-in by sending an id we registered.
    var requested_model_id = parseModelFromBody(request_body) orelse "";
    if (requested_model_id.len > 0 and !std.mem.eql(u8, requested_model_id, "mlx-serve")) {
        if (registry.peek(requested_model_id) == null) {
            // Ollama clients send tagged/short names ("qwen3.6:latest");
            // resolve them against registry ids before giving up. Scoped to
            // /api/ paths so /v1 fallback semantics stay pinned.
            var resolved: ?[]const u8 = null;
            if (std.mem.startsWith(u8, path, "/api/")) {
                resolved = ollamaResolveRegistryId(stream.io, registry, requested_model_id);
            }
            // Unknown id — fall back to the default model rather than 404,
            // so off-the-shelf SDK clients keep working. Multi-model
            // clients that care about routing precision pass an exact id
            // we registered (and `peek` will find it).
            requested_model_id = resolved orelse "";
        }
    }
    // Text-gen route aimed at a KNOWN non-text model: reject before
    // ensureLoaded, or the request cold-loads a multi-GB media model just
    // to earn its 400. The post-load `text_gen_reject` below stays the
    // authoritative crash barrier (this peek can't see `--model` primaries
    // with no arch hint until they're resident).
    if (requested_model_id.len > 0 and isTextGenRoute(method, path)) {
        if (registry.peek(requested_model_id)) |peeked| {
            if (textGenRejectReason(textGenTargetOf(peeked))) |reason| {
                if (std.mem.eql(u8, path, "/v1/messages")) {
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
    const lm = scheduler.ensureLoaded(requested_model_id) catch |err| switch (err) {
        error.UnknownModelId => {
            try sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "Unknown model id", 404);
            return;
        },
        error.NotLoaded => {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "model_not_loaded", "Requested model is not currently loaded", 503);
            return;
        },
        error.NoDefaultModel => {
            // Headless gen-only boot (media models resident, no default chat
            // model): /props keeps answering with the live memory counters —
            // the app's tray polls it, and a 503 here read as "0 MB".
            if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/props")) {
                try handlePropsNoModel(allocator, stream);
                return;
            }
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "no_model", "No default model configured", 503);
            return;
        },
        error.NotEnoughMemory => {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "out_of_memory", "Not enough memory to load model; retry after current requests complete", 503);
            return;
        },
        error.LoadFailed => {
            try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", "Model load failed", 500);
            return;
        },
        error.Shutdown => {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "shutting_down", "Server is shutting down", 503);
            return;
        },
        // Other errors (CPU-side preload failures like FileNotFound on a
        // missing tokenizer.json, JSON parse errors on a malformed
        // config.json, etc.) bubble out of `preloadCpuState`. Surface a
        // 500 with the error name so the client gets a clean failure
        // instead of a hung connection.
        else => {
            log.warn("  -> 500 ({s}) while resolving model\n", .{@errorName(err)});
            const msg = std.fmt.allocPrint(allocator, "Failed to load model: {s}", .{@errorName(err)}) catch {
                try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", "Failed to load model", 500);
                return;
            };
            defer allocator.free(msg);
            try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", msg, 500);
            return;
        },
    };
    defer scheduler.release(lm);

    // A model loaded on demand freezes its own auto-context here, for the same
    // reason the `--model` primary does at startup: the number we advertise is
    // what clients budget against, and each model has its own dims. Idempotent,
    // and a no-op under `--ctx-size`.
    if (lm.config) |c| _ = pinAutoContext(c);

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
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/v1/responses") and ws_mod.isUpgrade(request[0..header_end_pos])) {
        if (text_gen_reject) |reason| {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
            return;
        }
        try handleResponsesWebSocket(allocator, stream, request[0..header_end_pos], lm);
        return;
    }

    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/")) {
        log.debug("GET  / -> 200 (status page)\n", .{});
        try handleStatusPage(allocator, stream, lm);
    } else if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/props")) {
        log.debug("GET  /props -> 200\n", .{});
        try handleProps(allocator, stream, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/chat/completions")) {
        if (text_gen_reject) |reason| {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
            return;
        }
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleChatCompletions(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/completions")) {
        if (text_gen_reject) |reason| {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
            return;
        }
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleCompletions(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/embeddings")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleEmbeddings(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/messages")) {
        if (text_gen_reject) |reason| {
            try sendAnthropicError(allocator, stream, "invalid_request_error", reason, 400);
            return;
        }
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleAnthropicMessages(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/responses")) {
        if (text_gen_reject) |reason| {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
            return;
        }
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleResponses(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/tokenize")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleTokenize(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/detokenize")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleDetokenize(allocator, stream, body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/images/generations")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleGen(allocator, stream, body, lm, .image);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/audio/speech")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleGen(allocator, stream, body, lm, .speech);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/audio/music-generations")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleGen(allocator, stream, body, lm, .music);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/video/generations")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleGen(allocator, stream, body, lm, .video);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/3d/generations")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleGen(allocator, stream, body, lm, .mesh);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/api/chat")) {
        if (text_gen_reject) |reason| {
            try sendOllamaError(allocator, stream, "400 Bad Request", reason);
            return;
        }
        try handleOllamaChat(allocator, stream, request_body, lm);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/api/generate")) {
        if (text_gen_reject) |reason| {
            try sendOllamaError(allocator, stream, "400 Bad Request", reason);
            return;
        }
        try handleOllamaGenerate(allocator, stream, request_body, lm);
    } else if (std.mem.eql(u8, method, "POST") and (std.mem.eql(u8, path, "/api/embed") or std.mem.eql(u8, path, "/api/embeddings"))) {
        try handleOllamaEmbed(allocator, stream, request_body, lm, std.mem.eql(u8, path, "/api/embeddings"));
    } else {
        log.warn("{s} {s} -> 404\n", .{ method, path });
        try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "The requested endpoint does not exist", null);
    }
}

// ── Ollama-compatible API (/api/*) glue ─────────────────────────────────
// Pure translation (request shapes, SSE→NDJSON sink, renderers) lives in
// src/ollama.zig; this section owns routing targets, registry access, and
// the Conn sink hook. The inner /v1 handlers are reused verbatim — an
// /api/chat request becomes a /v1/chat/completions body whose SSE output
// the Sink re-frames into Ollama NDJSON on the real socket.

/// Sink output path: writes translated bytes DIRECTLY through the Conn's
/// writer interface, bypassing the `ollama_sink` hook in writeAll.
fn ollamaSinkOut(impl: *anyopaque, data: []const u8) anyerror!void {
    const c: *Conn = @ptrCast(@alignCast(impl));
    try c.writer().writeAll(data);
    try c.writer().flush();
}

fn ollamaSinkNowMs(impl: *anyopaque) i64 {
    const c: *Conn = @ptrCast(@alignCast(impl));
    return nowMs(c.io);
}

fn sendOllamaError(allocator: std.mem.Allocator, stream: *Conn, status: []const u8, message: []const u8) !void {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try out.writer.writeAll("{\"error\":");
    try ollama_mod.writeJsonString(&out.writer, message);
    try out.writer.writeAll("}");
    try sendResponse(stream, status, "application/json", out.written());
}

/// /api/* endpoints that must not trigger a model load. Returns true when
/// the request was fully handled.
fn handleOllamaEarly(allocator: std.mem.Allocator, stream: *Conn, method: []const u8, path: []const u8, body: []const u8) !bool {
    const is_get = std.mem.eql(u8, method, "GET");
    const is_post = std.mem.eql(u8, method, "POST");
    if ((is_get or std.mem.eql(u8, method, "HEAD")) and std.mem.eql(u8, path, "/api/version")) {
        log.debug("{s} /api/version -> 200\n", .{method});
        const vbody = try std.fmt.allocPrint(allocator, "{{\"version\":\"{s}\"}}", .{build_options.version});
        defer allocator.free(vbody);
        try sendResponse(stream, "200 OK", "application/json", if (is_get) vbody else "");
        return true;
    }
    if (is_get and std.mem.eql(u8, path, "/api/tags")) {
        log.debug("GET  /api/tags -> 200\n", .{});
        try handleOllamaTags(allocator, stream, false);
        return true;
    }
    if (is_get and std.mem.eql(u8, path, "/api/ps")) {
        log.debug("GET  /api/ps -> 200\n", .{});
        try handleOllamaTags(allocator, stream, true);
        return true;
    }
    if (is_post and std.mem.eql(u8, path, "/api/show")) {
        try handleOllamaShow(allocator, stream, body);
        return true;
    }
    if (is_post and std.mem.eql(u8, path, "/api/pull")) {
        try handleOllamaPull(allocator, stream, body);
        return true;
    }
    // Registry-mutating Ollama endpoints we deliberately don't support get
    // an explicit, actionable error instead of a bare 404.
    const unsupported = [_][]const u8{ "/api/create", "/api/copy", "/api/delete", "/api/push", "/api/blobs" };
    for (unsupported) |prefix| {
        if (std.mem.startsWith(u8, path, prefix)) {
            log.warn("{s} {s} -> 501 (unsupported ollama endpoint)\n", .{ method, path });
            try sendOllamaError(allocator, stream, "501 Not Implemented", "this Ollama endpoint is not supported by mlx-serve; manage models via /v1/load-model, /api/pull, or the MLX Core app");
            return true;
        }
    }
    return false;
}

/// Ollama-style model name → registered model id, or null. Registry ids
/// are stable for the process lifetime (unload keeps the stub), so the
/// returned slice stays valid after the mutex drops.
fn ollamaResolveRegistryId(io: std.Io, registry: *ModelRegistry, name: []const u8) ?[]const u8 {
    registry.mutex.lockUncancelable(io);
    defer registry.mutex.unlock(io);
    var ids_buf: [128][]const u8 = undefined;
    var n: usize = 0;
    var it = registry.entries.valueIterator();
    while (it.next()) |ep| {
        if (n >= ids_buf.len) break;
        ids_buf[n] = ep.*.id;
        n += 1;
    }
    const idx = ollama_mod.resolveName(name, ids_buf[0..n]) orelse return null;
    return ids_buf[idx];
}

fn ollamaQuantOf(id: []const u8) []const u8 {
    if (std.ascii.indexOfIgnoreCase(id, "4bit") != null) return "4bit";
    if (std.ascii.indexOfIgnoreCase(id, "8bit") != null) return "8bit";
    if (std.ascii.indexOfIgnoreCase(id, "bf16") != null) return "BF16";
    if (std.ascii.indexOfIgnoreCase(id, "nvfp4") != null) return "NVFP4";
    if (std.ascii.indexOfIgnoreCase(id, "q4") != null) return "Q4";
    if (std.ascii.indexOfIgnoreCase(id, "q8") != null) return "Q8";
    return "";
}

/// Snapshot one registry entry into the pure TagEntry shape. Caller holds
/// the registry mutex; id/arch_hint slices are entry-owned and stable.
fn ollamaTagEntryOf(io: std.Io, e: *LoadedModel) ollama_mod.TagEntry {
    const family: []const u8 = if (e.config) |c| c.model_type else (if (e.arch_hint.len > 0) e.arch_hint else "unknown");
    // arch_hint "gguf" covers unloaded discovery stubs whose PATH is a
    // directory of .gguf files (issue #59) — no engine yet, no .gguf suffix.
    const is_gguf = e.ds4_engine != null or e.llama_engine != null or
        std.mem.endsWith(u8, e.path, ".gguf") or std.mem.eql(u8, e.arch_hint, "gguf");
    var modified_ms: i64 = 0;
    // config.json mtime; .gguf entries fall back to 0 (epoch) rather than
    // paying a parent-dir walk. Guard the absolute-path precondition —
    // openDirAbsolute on a non-absolute path is ReleaseFast UB (CLAUDE.md).
    if (!is_gguf and e.path.len > 0 and std.fs.path.isAbsolute(e.path)) {
        if (std.Io.Dir.openDirAbsolute(io, e.path, .{})) |d| {
            var dir = d;
            defer dir.close(io);
            if (dir.statFile(io, "config.json", .{})) |st| {
                modified_ms = st.mtime.toMilliseconds();
            } else |_| {}
        } else |_| {}
    }
    return .{
        .id = e.id,
        .size_bytes = e.bytes_on_disk orelse 0,
        .modified_ms = modified_ms,
        .family = family,
        .format = if (is_gguf) "gguf" else "safetensors",
        .quant = ollamaQuantOf(e.id),
    };
}

/// GET /api/tags (`ps_only=false`: every registered model) and GET /api/ps
/// (`ps_only=true`: only GPU-resident entries, with residency bytes).
fn handleOllamaTags(allocator: std.mem.Allocator, stream: *Conn, ps_only: bool) !void {
    const registry = global_registry orelse {
        try sendOllamaError(allocator, stream, "503 Service Unavailable", "registry not ready");
        return;
    };
    var body: []u8 = undefined;
    {
        registry.mutex.lockUncancelable(stream.io);
        defer registry.mutex.unlock(stream.io);
        if (ps_only) {
            var entries = std.ArrayList(ollama_mod.PsEntry).empty;
            defer entries.deinit(allocator);
            var it = registry.entries.valueIterator();
            while (it.next()) |ep| {
                const e = ep.*;
                if (e.state != .ready) continue;
                try entries.append(allocator, .{
                    .tag = ollamaTagEntryOf(stream.io, e),
                    .resident_bytes = e.bytes_resident,
                });
            }
            body = try ollama_mod.renderPsJson(allocator, entries.items);
        } else {
            var entries = std.ArrayList(ollama_mod.TagEntry).empty;
            defer entries.deinit(allocator);
            var it = registry.entries.valueIterator();
            while (it.next()) |ep| {
                try entries.append(allocator, ollamaTagEntryOf(stream.io, ep.*));
            }
            body = try ollama_mod.renderTagsJson(allocator, entries.items);
        }
    }
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

/// POST /api/show — model metadata + capabilities.
fn handleOllamaShow(allocator: std.mem.Allocator, stream: *Conn, body: []const u8) !void {
    const registry = global_registry orelse {
        try sendOllamaError(allocator, stream, "503 Service Unavailable", "registry not ready");
        return;
    };
    var requested: []const u8 = "";
    var parsed_body: ?std.json.Parsed(std.json.Value) = null;
    defer if (parsed_body) |*p| p.deinit();
    if (std.json.parseFromSlice(std.json.Value, allocator, body, .{})) |parsed| {
        parsed_body = parsed;
        if (parsed.value == .object) {
            // "model" is current; "name" is the pre-0.5 client field.
            if (parsed.value.object.get("model")) |m| {
                if (m == .string) requested = m.string;
            } else if (parsed.value.object.get("name")) |m| {
                if (m == .string) requested = m.string;
            }
        }
    } else |_| {}
    if (requested.len == 0) {
        try sendOllamaError(allocator, stream, "400 Bad Request", "model is required");
        return;
    }

    var rendered: ?[]u8 = null;
    {
        registry.mutex.lockUncancelable(stream.io);
        defer registry.mutex.unlock(stream.io);
        var ids_buf: [128][]const u8 = undefined;
        var n: usize = 0;
        var entry_buf: [128]*LoadedModel = undefined;
        var it = registry.entries.valueIterator();
        while (it.next()) |ep| {
            if (n >= ids_buf.len) break;
            ids_buf[n] = ep.*.id;
            entry_buf[n] = ep.*;
            n += 1;
        }
        if (ollama_mod.resolveName(requested, ids_buf[0..n])) |idx| {
            const e = entry_buf[idx];
            const template: []const u8 = if (e.chat_config) |cc| cc.chat_template else "";
            const is_encoder = if (e.config) |c| c.is_encoder_only else std.mem.eql(u8, e.arch_hint, "bert");
            const has_chat = !is_encoder;
            rendered = try ollama_mod.renderShowJson(allocator, .{
                .tag = ollamaTagEntryOf(stream.io, e),
                .context_length = if (e.config) |c| getEffectiveContextLength(c) else 0,
                .template = template,
                .has_chat = has_chat,
                .has_tools = has_chat,
                .has_vision = e.vision_encoder != null,
                .has_thinking = has_chat and chatTemplateSupportsThinking(template),
                .has_embedding = is_encoder,
            });
        }
    }
    if (rendered) |r| {
        defer allocator.free(r);
        log.debug("POST /api/show -> 200 ({s})\n", .{requested});
        try sendResponse(stream, "200 OK", "application/json", r);
    } else {
        log.warn("POST /api/show -> 404 (unknown model {s})\n", .{requested});
        try sendOllamaError(allocator, stream, "404 Not Found", "model not found");
    }
}

/// Progress reporter for /api/pull: each status line becomes an Ollama
/// NDJSON `{"status":"…"}` chunk on the wire.
const OllamaPullSink = struct {
    stream: *Conn,
    allocator: std.mem.Allocator,
    quiet: bool = false,
    headers_sent: bool = false,
    write_failed: bool = false,

    fn report(impl: *anyopaque, line: []const u8) void {
        const self: *OllamaPullSink = @ptrCast(@alignCast(impl));
        if (self.quiet) return;
        self.emit("status", line) catch {
            self.write_failed = true;
        };
    }

    fn emit(self: *OllamaPullSink, key: []const u8, line: []const u8) !void {
        if (!self.headers_sent) {
            self.headers_sent = true;
            try self.stream.writeAll("HTTP/1.1 200 OK\r\n" ++
                "Content-Type: application/x-ndjson\r\n" ++
                "Cache-Control: no-cache\r\n" ++
                "Connection: close\r\n" ++
                "Access-Control-Allow-Origin: *\r\n" ++
                "\r\n");
        }
        var out: std.Io.Writer.Allocating = .init(self.allocator);
        defer out.deinit();
        try out.writer.writeAll("{\"");
        try out.writer.writeAll(key);
        try out.writer.writeAll("\":");
        try ollama_mod.writeJsonString(&out.writer, line);
        try out.writer.writeAll("}\n");
        try self.stream.writeAll(out.written());
    }
};

/// POST /api/pull — native HF download into ~/.mlx-serve/models (same
/// resolver + layout as `mlx-serve pull`), then register-by-path so the
/// model is immediately loadable by name. Streams NDJSON status lines
/// unless the client passed `stream:false`.
fn handleOllamaPull(allocator: std.mem.Allocator, stream: *Conn, body: []const u8) !void {
    // No `curl`, no arbitrary-path downloads in the App Store build — the Swift
    // app owns model downloads via URLSession into the container.
    if (@import("build_options").mas) {
        try sendOllamaError(allocator, stream, "501 Not Implemented", "model pull is unavailable in this build");
        return;
    }
    var requested: []const u8 = "";
    var wants_stream = true;
    var parsed_body: ?std.json.Parsed(std.json.Value) = null;
    defer if (parsed_body) |*p| p.deinit();
    if (std.json.parseFromSlice(std.json.Value, allocator, body, .{})) |parsed| {
        parsed_body = parsed;
        if (parsed.value == .object) {
            if (parsed.value.object.get("model")) |m| {
                if (m == .string) requested = m.string;
            } else if (parsed.value.object.get("name")) |m| {
                if (m == .string) requested = m.string;
            }
            if (parsed.value.object.get("stream")) |s| {
                wants_stream = s == .bool and s.bool;
            }
        }
    } else |_| {}
    if (requested.len == 0) {
        try sendOllamaError(allocator, stream, "400 Bad Request", "model is required");
        return;
    }
    const resolved = cli_mod.resolveShortName(requested) orelse {
        try sendOllamaError(allocator, stream, "404 Not Found", "unknown model name; use a known short name or a HuggingFace 'org/repo' id");
        return;
    };
    const home = std.mem.span(std.c.getenv("HOME") orelse "/tmp");
    const dest = try cli_mod.modelDestPath(allocator, home, resolved.repo);
    defer allocator.free(dest);

    log.info("POST /api/pull {s} -> {s}\n", .{ requested, dest });
    var sink = OllamaPullSink{ .stream = stream, .allocator = allocator, .quiet = !wants_stream };
    const reporter = cli_mod.Reporter{ .impl = &sink, .reportFn = &OllamaPullSink.report };
    if (!cli_mod.modelPresent(stream.io, dest)) {
        cli_mod.pullRepo(allocator, stream.io, resolved, dest, reporter, false) catch {
            if (sink.headers_sent) {
                sink.emit("error", "pull failed (partials kept — retry to resume)") catch {};
            } else {
                try sendOllamaError(allocator, stream, "500 Internal Server Error", "pull failed (partials kept — retry to resume)");
            }
            return;
        };
    }
    // Make it loadable by name right away. GGUF-only dirs (no config.json)
    // aren't registerable this way — they still work via --model / the app.
    if (global_registry) |registry| {
        _ = registry.registerByPath(stream.io, dest) catch {};
    }
    sink.quiet = false;
    sink.emit("status", "success") catch {};
}

fn handleOllamaChat(allocator: std.mem.Allocator, stream: *Conn, body: []const u8, lm: *LoadedModel) !void {
    var tr = ollama_mod.translateChatRequest(allocator, body) catch |err| switch (err) {
        error.InvalidRequest => {
            log.warn("POST /api/chat -> 400 (invalid request)\n", .{});
            try sendOllamaError(allocator, stream, "400 Bad Request", "invalid chat request: model and messages are required");
            return;
        },
        error.OutOfMemory => return error.OutOfMemory,
    };
    defer tr.deinit(allocator);
    log.debug("POST /api/chat (stream={any}) -> inner /v1/chat/completions\n", .{tr.wants_stream});
    var sink = ollama_mod.Sink.init(allocator, .{
        .mode = .chat,
        .wants_stream = tr.wants_stream,
        .model = tr.model,
        .out_impl = stream,
        .outFn = &ollamaSinkOut,
        .nowMsFn = &ollamaSinkNowMs,
    });
    defer sink.deinit();
    stream.ollama_sink = &sink;
    defer stream.ollama_sink = null;
    try handleChatCompletions(allocator, stream, tr.body, lm);
    stream.ollama_sink = null;
    try sink.finish();
}

fn handleOllamaGenerate(allocator: std.mem.Allocator, stream: *Conn, body: []const u8, lm: *LoadedModel) !void {
    var tr = ollama_mod.translateGenerateRequest(allocator, body) catch |err| switch (err) {
        error.InvalidRequest => {
            log.warn("POST /api/generate -> 400 (invalid request)\n", .{});
            try sendOllamaError(allocator, stream, "400 Bad Request", "invalid generate request: prompt is required");
            return;
        },
        error.OutOfMemory => return error.OutOfMemory,
    };
    defer tr.deinit(allocator);
    log.debug("POST /api/generate (stream={any}, raw={any}) -> inner {s}\n", .{ tr.wants_stream, tr.raw, if (tr.raw) "/v1/completions" else "/v1/chat/completions" });
    var sink = ollama_mod.Sink.init(allocator, .{
        .mode = .generate,
        .wants_stream = tr.wants_stream,
        .model = tr.model,
        .out_impl = stream,
        .outFn = &ollamaSinkOut,
        .nowMsFn = &ollamaSinkNowMs,
    });
    defer sink.deinit();
    stream.ollama_sink = &sink;
    defer stream.ollama_sink = null;
    if (tr.raw) {
        try handleCompletions(allocator, stream, tr.body, lm);
    } else {
        try handleChatCompletions(allocator, stream, tr.body, lm);
    }
    stream.ollama_sink = null;
    try sink.finish();
}

fn handleOllamaEmbed(allocator: std.mem.Allocator, stream: *Conn, body: []const u8, lm: *LoadedModel, legacy: bool) !void {
    var tr = ollama_mod.translateEmbedRequest(allocator, body, legacy) catch |err| switch (err) {
        error.InvalidRequest => {
            log.warn("POST /api/embed -> 400 (invalid request)\n", .{});
            try sendOllamaError(allocator, stream, "400 Bad Request", if (legacy) "invalid request: prompt is required" else "invalid request: input is required");
            return;
        },
        error.OutOfMemory => return error.OutOfMemory,
    };
    defer tr.deinit(allocator);
    var sink = ollama_mod.Sink.init(allocator, .{
        .mode = .chat, // unused for embed; finishEmbed re-renders the capture
        .wants_stream = false,
        .model = tr.model,
        .out_impl = stream,
        .outFn = &ollamaSinkOut,
        .nowMsFn = &ollamaSinkNowMs,
    });
    defer sink.deinit();
    stream.ollama_sink = &sink;
    defer stream.ollama_sink = null;
    try handleEmbeddings(allocator, stream, tr.body, lm);
    stream.ollama_sink = null;
    try sink.finishEmbed(legacy);
}

/// Percentage of the memory-derived ceiling we actually admit as context when
/// `--ctx-size` is absent.
///
/// `computeMaxSafeContext` returns the largest context that FITS right now.
/// Serving at exactly that leaves nothing for the prefix cache to grow into, a
/// second model to load beside this one, or another app to take RAM — and the
/// Metal OOM it would eventually hit is uncatchable (see the auto-context
/// gotcha). Reserve headroom instead.
const auto_ctx_safety_pct: u32 = 85;

/// PURE: apply the safety margin and round down to a 1024 boundary so the
/// number reads sanely in logs and client configs. Never returns 0.
///
/// Takes the MEMORY-derived ceiling only. The model's own `max_position_embeddings`
/// is applied afterwards, un-margined: when the checkpoint's max is the binding
/// constraint there is nothing to reserve memory headroom against, and shaving
/// 15% off a 131,072-token model that comfortably fits in RAM just throws
/// context away.
fn safeAutoContext(raw: u32) u32 {
    const scaled: u64 = (@as(u64, raw) * auto_ctx_safety_pct) / 100;
    const rounded: u64 = (scaled / 1024) * 1024;
    if (rounded == 0) return @intCast(@max(scaled, 1));
    return @intCast(rounded);
}

/// This model's auto-context: memory ceiling with headroom, then capped by what
/// the checkpoint actually supports.
fn autoContextFor(config: *const model_mod.ModelConfig) u32 {
    const with_headroom = safeAutoContext(computeMemoryContext(config));
    const max_pos = config.max_position_embeddings;
    return if (max_pos > 0) @min(with_headroom, max_pos) else with_headroom;
}

/// Freeze this model's auto-context at load time. Idempotent; a no-op (and
/// never a write) when `--ctx-size` pinned it explicitly. Returns the value
/// `getEffectiveContextLength` will report from now on.
///
/// Called once per model, at startup for the `--model` primary and right after
/// an on-demand load. The advertised context must not drift: agent CLIs read it
/// once (pi/opencode never re-read `/v1/models`) and budget their own
/// `max_tokens` against it for the rest of the session.
fn pinAutoContext(config: *model_mod.ModelConfig) u32 {
    if (server_config.max_context_size > 0) return server_config.max_context_size;
    if (config.pinned_context == 0) {
        config.pinned_context = autoContextFor(config);
    }
    return config.pinned_context;
}

fn getEffectiveContextLength(config: *const model_mod.ModelConfig) u32 {
    if (server_config.max_context_size > 0) return server_config.max_context_size;
    if (config.pinned_context > 0) return config.pinned_context;
    // Not pinned yet (a discovery stub that was never loaded): compute from
    // current GPU memory rather than a fixed 16K cap.
    return autoContextFor(config);
}

/// Metal's recommended max working-set size for the default device — the real
/// ceiling whose breach throws `[METAL] … Insufficient Memory` from the
/// command-buffer completion handler (which terminates the process: ggml's
/// global std::terminate handler prints the backtrace, but the throw is MLX's).
/// `getMetalBufferLimit()` (75% of physical RAM) over-estimates this on
/// small-RAM Macs — a 16 GB Mac reports ~11.9 GB recommended vs the 12 GB that
/// hw.memsize×0.75 yields — so budgeting against hw.memsize lets auto-context
/// oversubscribe. Falls back to `getMetalBufferLimit()` when the device query
/// is unavailable (CI / non-Metal hosts).
fn getGpuWorkingSetLimit() u64 {
    var dev = mlx.mlx_device{ .ctx = null };
    _ = mlx.mlx_get_default_device(&dev);
    var info = mlx.mlx_device_info_new();
    defer _ = mlx.mlx_device_info_free(info);
    if (mlx.mlx_device_info_get(&info, dev) == 0) {
        var max_rec: usize = 0;
        if (mlx.mlx_device_info_get_size(&max_rec, info, "max_recommended_working_set_size") == 0 and max_rec > 0) {
            return @as(u64, max_rec);
        }
    }
    return getMetalBufferLimit();
}

/// PURE (unit-testable): the real ceiling a NEW MLX allocation must fit under.
///
/// `working_set_limit` (Metal's `max_recommended_working_set_size`) is a STATIC
/// device maximum — it assumes the whole GPU working set is MLX's to claim and
/// is BLIND to memory held by anything else on the machine. `mlx_footprint`
/// (MLX active + reclaimable cache) + `free_system` (physically free RAM) is
/// what MLX can actually reach RIGHT NOW; when another process holds a big
/// chunk of unified memory the second term binds and the ceiling collapses.
///
/// #64 (2026-07): a Claude Code session on a 128 GB Mac ran a docker-compose
/// stack (firecrawl/rabbitmq/postgres/playwright) holding tens of GB. The guard
/// budgeted against the static 115 GB max, admitted a 90 K-token MoE prefill,
/// and Metal threw `Insufficient Memory` from the command-buffer completion
/// handler — an UNCATCHABLE async C++ throw that terminates the process (the
/// libllama frames in the backtrace are just its global std::terminate handler;
/// the throw is MLX's). Capping by real free RAM makes the two prefill guards
/// reject/clamp before that allocation is ever submitted.
fn physicalMemoryCeiling(working_set_limit: u64, mlx_footprint: u64, free_system: u64) u64 {
    return @min(working_set_limit, mlx_footprint +| free_system);
}

/// Impure wrapper: current GPU allocation ceiling from live MLX + system
/// counters. `mlx_footprint` = active (in-use) + cache (reclaimable) so an
/// idle machine's ceiling stays ≈ the static device max (no auto-context
/// regression), while external pressure — reflected in `getAvailableMemBytes`
/// (total − wired − compressed − internal-anon) — tightens it. See
/// `physicalMemoryCeiling`.
fn currentGpuMemoryCeiling(active_mem: u64) u64 {
    var cache_mem: usize = 0;
    _ = mlx.mlx_get_cache_memory(&cache_mem);
    return physicalMemoryCeiling(
        getGpuWorkingSetLimit(),
        active_mem +| @as(u64, cache_mem),
        metrics.getAvailableMemBytes(),
    );
}

/// PURE budget math (no MLX/Metal calls — unit-testable): the largest context
/// length whose KV cache + per-token working set fits under the GPU
/// working-set ceiling, after subtracting what's already resident
/// (`active_mem` — model weights + any resident hot-cache KV) AND the hot
/// prefix cache's FULL byte budget (`hot_cache_reserve`).
///
/// Reserving the whole cache budget — not just its current residency — budgets
/// for STEADY STATE. Over an agentic session the hot cache fills to its cap, so
/// an auto-context computed against an empty cache (24k tokens on a 16 GB Mac,
/// 2026-06-19 live qwen3_5_moe) later collides with the filled 2 GB cache plus a
/// large cold MoE prefill and the command buffer OOMs. Subtracting it up front
/// keeps the reported context stable across the session and within the ceiling
/// once the cache is full. The 0.64 factor (two ×4/5 margins) absorbs the
/// prefill compute spike that lands on top of `active_mem`.
fn safeContextForBudget(
    working_set_limit: u64,
    active_mem: u64,
    hot_cache_reserve: u64,
    per_tok: u64,
    max_pos: u32,
) u32 {
    if (per_tok == 0) return 1024;
    const spoken_for: u64 = active_mem + hot_cache_reserve;
    const usable: u64 = if (working_set_limit > spoken_for) working_set_limit - spoken_for else 0;
    const budget: u64 = usable * 4 / 5;
    const max_seq: u64 = (budget * 4 / 5) / per_tok;
    if (max_seq == 0) return 1024;

    var result: u32 = if (max_seq > std.math.maxInt(u32)) std.math.maxInt(u32) else @intCast(max_seq);
    // Cap at model's max position embeddings.
    if (max_pos > 0) result = @min(result, max_pos);
    return result;
}

/// Compute the maximum safe context length based on the GPU working-set
/// ceiling, the model's per-token footprint, and the hot prefix cache budget.
/// Linear in seq (per_tok × seq ≤ budget) — no seq² term, MLX's fused SDPA
/// tiles over seq and never materializes [heads, seq, seq]. See
/// `safeContextForBudget` for the steady-state reservation rationale.
/// `computeMemoryContext` capped by the checkpoint's `max_position_embeddings`.
/// This is the raw "largest context that fits" figure — reported in logs and
/// `/props` for diagnostics. The context we actually SERVE is `autoContextFor`,
/// which reserves headroom below it.
fn computeMaxSafeContext(config: *const model_mod.ModelConfig) u32 {
    const memory_ctx = computeMemoryContext(config);
    const max_pos = config.max_position_embeddings;
    return if (max_pos > 0) @min(memory_ctx, max_pos) else memory_ctx;
}

/// The largest context this model's per-token footprint fits into RAM right
/// now, IGNORING the checkpoint's own maximum. Reads live memory.
fn computeMemoryContext(config: *const model_mod.ModelConfig) u32 {
    const heads: u64 = config.num_attention_heads;
    if (heads == 0) return 16384;

    const layers: u64 = config.num_hidden_layers;
    const kv_heads: u64 = config.num_key_value_heads;
    const hdim: u64 = config.head_dim;
    const hidden: u64 = config.hidden_size;
    const ffn: u64 = @max(config.intermediate_size, config.moe_intermediate_size + config.shared_expert_intermediate_size);

    //   KV cache (fp16):   layers × 2 × kv_heads × head_dim × 2 bytes per token
    //   Working (fp16):    ~8 × max(hidden, ffn) × 2 bytes per token (per-layer tensors,
    //                      bounded by EVAL_EVERY_N_LAYERS in transformer.zig)
    const kv_per_tok: u64 = layers * 2 * kv_heads * hdim * 2;
    const work_per_tok: u64 = 8 * @max(hidden, ffn) * 2;
    const per_tok: u64 = kv_per_tok + work_per_tok;

    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);

    return safeContextForBudget(
        // Real reachable ceiling, not the static device max — so auto-context
        // shrinks when another process (e.g. a docker stack) holds unified
        // memory, instead of oversubscribing into an uncatchable Metal OOM (#64).
        currentGpuMemoryCeiling(active_mem),
        active_mem,
        // The hot prefix cache fills to this cap over a session — reserve it so
        // auto-context doesn't oversubscribe once the cache is full.
        prefix_cache_mem_bytes,
        per_tok,
        // 0 = do NOT clamp to the checkpoint's max here. The caller applies that
        // cap AFTER the safety margin, so a model whose own max is the binding
        // constraint keeps every token of it (see `autoContextFor`).
        0,
    );
}

/// Pure memory model behind checkAttentionMemory. All quantities in bytes,
/// 25% safety margin included. `chunk` must be the SAME prefill chunk
/// generate.zig will pick (generate_mod.effectivePrefillChunk) so the
/// admission guard and the real prefill cannot drift. Terms:
///   - kv: persistent for the request; billed at the ACTIVE kv-quant width
///     (+0.5 bit/elem group scale/bias overhead when quantized), not a
///     hardcoded fp16.
///   - scores: the composed-SDPA scratch [heads, chunk, seq] — materialized
///     only for head_dims no fused kernel covers (transformer.
///     prefillHeadDimFused: <= 128 via MLX, 256 via msv_attn_p256 — unfused
///     only under the MLX_SERVE_FUSED_256=0 kill switch or an exotic dim).
///     Bounded to ~one layer by the adaptive eval cadence
///     (transformer.prefillEvalCadence).
///   - dequant: dense-fp16 rebuild of the FULL quantized cache each layer
///     (KVCache.denseView under --kv-quant).
///   - mlp: QKV/MLP transient envelope; ~3 layers coexist between eval
///     points, and it is CHUNK-bounded, never seq-scaled — the old
///     8×seq×ffn envelope over-billed a 255K prompt by ~60 GB and rejected
///     requests that fit comfortably (PR #69).
pub fn prefillMemoryNeeded(seq: u64, heads: u64, kv_heads: u64, layers: u64, hdim: u64, hidden: u64, ffn: u64, kv_bits: u64, chunk: u64) u64 {
    const kv_bytes: u64 = if (kv_bits >= 16)
        layers * 2 * seq * kv_heads * hdim * 2
    else
        layers * 2 * seq * kv_heads * hdim * (2 * kv_bits + 1) / 16;
    const scores: u64 = if (!transformer_mod.prefillHeadDimFused(@intCast(hdim))) heads * chunk * seq * 2 else 0;
    const dequant: u64 = if (kv_bits < 16) 2 * seq * kv_heads * hdim * 2 else 0;
    const mlp: u64 = 8 * chunk * @max(hidden, ffn) * 2;
    return (kv_bytes + scores + dequant + 3 * mlp) * 5 / 4;
}

/// Estimate peak GPU memory for prefill and reject if it would exceed the Metal
/// working-set ceiling. Exceeding it throws an uncatchable C++ exception on a
/// Metal completion-handler thread and kills the process, so PREVENTION is the
/// only lever — this guard must track what the prefill actually allocates
/// (prefillMemoryNeeded above; chunk choice from generate.effectivePrefillChunk).
/// `kv_override` is the per-request `kv_quant` body field where the surface
/// parses one (chat/messages/responses); null falls back to the process default.
fn checkAttentionMemory(allocator: std.mem.Allocator, stream: *Conn, prompt_len: usize, config: *const model_mod.ModelConfig, is_anthropic: bool, kv_override: ?transformer_mod.KVQuantConfig) !bool {
    const heads = config.num_attention_heads;
    if (heads == 0) return true; // unknown architecture, skip check

    const seq: u64 = @intCast(prompt_len);
    const layers: u64 = config.num_hidden_layers;
    const kv_heads: u64 = config.num_key_value_heads;
    const hdim: u64 = config.head_dim;
    const hidden: u64 = config.hidden_size;
    // Deliberately the conservative @max: MoE checkpoints often omit
    // intermediate_size (struct default 15360 leaks in) but the envelope is
    // chunk-bounded now, so a fat ffn costs MBs of estimate, not GBs.
    const ffn: u64 = @max(config.intermediate_size, config.moe_intermediate_size + config.shared_expert_intermediate_size);

    const kv_cfg: transformer_mod.KVQuantConfig = kv_override orelse
        (if (global_scheduler) |sch| sch.kv_quant_config else transformer_mod.KVQuantConfig.dense);
    const kv_bits: u64 = if (kv_cfg.scheme == .off) 16 else kv_cfg.bits;
    const chunk: u64 = @intCast(generate_mod.effectivePrefillChunk(config.head_dim, config.num_attention_heads, prompt_len, config.has_sliding_window));
    const needed: u64 = prefillMemoryNeeded(seq, heads, kv_heads, layers, hdim, hidden, ffn, kv_bits, chunk);

    // Available = GPU allocation ceiling minus current usage (model weights,
    // resident hot-cache KV, etc.). The ceiling is the LESSER of Metal's static
    // working-set max and what's physically reachable now (currentGpuMemoryCeiling)
    // so the two prefill guards agree AND both see external memory pressure — a
    // docker stack holding tens of GB otherwise leaves this guard admitting a
    // prefill that OOMs the command buffer (#64).
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    const total_limit: u64 = currentGpuMemoryCeiling(active_mem);
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
/// Clamp a request's `max_tokens` so `prompt + generation` stays inside
/// `effective_ctx`. `effective_ctx == 0` means "unknown" and imposes no limit.
///
/// Takes the context explicitly rather than reading `server_config.max_context_size`:
/// that global is only set by `--ctx-size`, so under AUTO context this function
/// used to return `max_tokens` untouched — the server never trimmed a client's
/// budget, never emitted the "budget squeezed" warning, and generation could run
/// past the memory-safe window. Callers pass `getEffectiveContextLength(config)`,
/// which is the same number the prompt-length guard and `/v1/models` report.
fn clampMaxTokens(max_tokens: u32, prompt_len: usize, effective_ctx: u32) u32 {
    if (effective_ctx == 0) return max_tokens;
    const prompt: u32 = @intCast(@min(prompt_len, effective_ctx));
    if (prompt >= effective_ctx) return 1; // at least 1 token
    const remaining = effective_ctx - prompt;
    if (remaining < max_tokens / 4) {
        log.warn("  generation budget squeezed: {d}/{d} tokens remaining (prompt={d}, ctx={d}) — tool call arguments may be truncated\n", .{ remaining, max_tokens, prompt, effective_ctx });
    }
    if (max_tokens > remaining) {
        log.debug("  max_tokens clamped: {d} -> {d} (ctx={d}, prompt={d})\n", .{ max_tokens, remaining, effective_ctx, prompt });
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

/// Render an optional model-author sampling recommendation (from the model's
/// generation_config.json) as a JSON scalar: the number when present, the
/// literal `null` when the model ships no value. Caller owns the slice.
/// Used for the `gen_temperature`/`gen_top_p`/`gen_top_k` meta fields the
/// Swift Settings UI reads to show "model recommends" guidance pills.
fn optSamplingRecJson(allocator: std.mem.Allocator, comptime T: type, v: ?T) ![]u8 {
    if (v) |val| return std.fmt.allocPrint(allocator, "{d}", .{val});
    return allocator.dupe(u8, "null");
}

/// Render the JSON metadata fragment for one entry. For `.ready` entries
/// Capability flags for a READY registry entry, kept as plain booleans so the
/// JSON assembly below is hermetically testable without a LoadedModel. Every
/// engine slot on LoadedModel must have a flag here — a missing arm renders a
/// ready model with an empty capabilities list (the live `.mesh`/"3d" hole).
const ReadyCaps = struct {
    has_chat: bool = false,
    has_vision: bool = false,
    has_audio: bool = false,
    has_reasoning: bool = false,
    is_encoder_only: bool = false,
    has_image_engine: bool = false,
    has_audio_engine: bool = false,
    /// Audio engine's backend is the ACE-Step music generator (advertises
    /// "music" ADDITIVELY beside "audio", the ready-model "3d" precedent).
    has_music_backend: bool = false,
    has_video_engine: bool = false,
    has_mesh_engine: bool = false,
};

/// `capabilities` JSON array for a ready model. Caller deinits.
fn readyCapsJson(allocator: std.mem.Allocator, c: ReadyCaps) !std.ArrayList(u8) {
    var caps = std.ArrayList(u8).empty;
    errdefer caps.deinit(allocator);
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
    if (c.has_chat) try append_cap(allocator, &caps, &n_caps, "chat");
    if (c.has_chat) try append_cap(allocator, &caps, &n_caps, "tool_use");
    if (c.has_chat) try append_cap(allocator, &caps, &n_caps, "streaming");
    if (c.has_vision) try append_cap(allocator, &caps, &n_caps, "vision");
    if (c.has_audio) try append_cap(allocator, &caps, &n_caps, "audio");
    if (c.has_reasoning) try append_cap(allocator, &caps, &n_caps, "reasoning");
    if (c.has_chat) try append_cap(allocator, &caps, &n_caps, "json_schema");
    if (c.is_encoder_only) try append_cap(allocator, &caps, &n_caps, "embeddings");
    // Native media-generation engines (resident).
    if (c.has_image_engine) try append_cap(allocator, &caps, &n_caps, "image");
    if (c.has_audio_engine and !c.has_audio) try append_cap(allocator, &caps, &n_caps, "audio");
    if (c.has_music_backend) try append_cap(allocator, &caps, &n_caps, "music");
    if (c.has_video_engine) try append_cap(allocator, &caps, &n_caps, "video");
    if (c.has_mesh_engine) try append_cap(allocator, &caps, &n_caps, "3d");
    try caps.append(allocator, ']');
    return caps;
}

/// Facts about a request's target model that decide whether a TEXT-
/// GENERATION route (/v1/chat/completions, /v1/completions, /v1/messages,
/// /v1/responses + WS, /api/chat, /api/generate) may serve it. Extracted
/// from a LoadedModel by `textGenTargetOf`; kept as plain facts so the
/// decision is hermetically testable.
const TextGenTarget = struct {
    is_encoder_only: bool = false,
    arch_hint: []const u8 = "",
    has_image_engine: bool = false,
    has_audio_engine: bool = false,
    has_video_engine: bool = false,
    has_mesh_engine: bool = false,
    /// A text-capable LM is resident (transformer / ds4 / llama engine) —
    /// or the entry isn't loaded yet, in which case stubs default to
    /// "assume text until the arch hint or a load says otherwise".
    has_text_lm: bool = true,
};

/// Reason a text-generation route must reject this model with a 400, or
/// null when it can serve text. Crash class (live SIGSEGV 2026-07-06): a
/// chat request routed at a media model has only the gen stub CPU state —
/// the empty stub tokenizer produced 0 prompt tokens and prefill deref'd
/// `transformer == null`, killing the whole server from one request (any
/// remote client naming a media model could down it). Media entries are
/// detected BOTH pre-load (discovery arch_hint) and post-load (engine
/// slots) — either alone has gaps: `--model` primaries carry no hint,
/// engines exist only while resident.
fn textGenRejectReason(t: TextGenTarget) ?[]const u8 {
    if (t.is_encoder_only) return "Encoder-only models do not support text generation. Use /v1/embeddings instead.";
    const modality: ?media_mod.Modality = blk: {
        if (t.has_image_engine) break :blk .image;
        if (t.has_video_engine) break :blk .video;
        if (t.has_mesh_engine) break :blk .mesh;
        if (t.has_audio_engine) break :blk .audio;
        break :blk media_mod.modalityFromType(t.arch_hint);
    };
    if (modality) |m| return switch (m) {
        .image => "This is an image generation model; it cannot serve chat/text requests. Use POST /v1/images/generations instead.",
        .audio => "This is an audio generation model; it cannot serve chat/text requests. Use POST /v1/audio/speech (TTS) or /v1/audio/music-generations (music) instead.",
        .video => "This is a video generation model; it cannot serve chat/text requests. Use POST /v1/video/generations instead.",
        .mesh => "This is a 3D generation model; it cannot serve chat/text requests. Use POST /v1/3d/generations instead.",
    };
    if (!t.has_text_lm) return "This model cannot serve text generation (no language model resident).";
    return null;
}

fn textGenTargetOf(lm: *LoadedModel) TextGenTarget {
    return .{
        .is_encoder_only = if (lm.config) |c| c.is_encoder_only else false,
        .arch_hint = lm.arch_hint,
        .has_image_engine = lm.image_engine != null,
        .has_audio_engine = lm.audio_engine != null,
        .has_video_engine = lm.video_engine != null,
        .has_mesh_engine = lm.mesh_engine != null,
        .has_text_lm = lm.state != .ready or lm.transformer != null or
            lm.ds4_engine != null or lm.llama_engine != null,
    };
}

/// True for the routes textGenRejectReason protects — used for the
/// pre-load peek so naming a media model in a chat request doesn't
/// cold-load gigabytes just to earn its 400.
fn isTextGenRoute(method: []const u8, path: []const u8) bool {
    if (std.mem.eql(u8, method, "POST")) {
        const routes = [_][]const u8{
            "/v1/chat/completions", "/v1/completions", "/v1/messages",
            "/v1/responses",        "/api/chat",       "/api/generate",
        };
        for (routes) |r| if (std.mem.eql(u8, path, r)) return true;
        return false;
    }
    // WebSocket upgrade handshake for /v1/responses.
    return std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/v1/responses");
}

/// pulls full capabilities/dimensions off the resident config/chat_config;
/// for non-ready entries renders a lightweight stub with state +
/// bytes_on_disk only. Returns an allocator-owned string; caller frees.
/// Called from `handleModels` per registry entry. Registry mutex must be
/// held by the caller (entry fields are read directly).
fn renderModelEntry(
    allocator: std.mem.Allocator,
    io: std.Io,
    entry: *LoadedModel,
) ![]u8 {
    if (entry.state == .ready and entry.config != null and entry.chat_config != null) {
        const config = entry.config.?;
        const chat_config = entry.chat_config.?;
        const ctx_len = getEffectiveContextLength(config);
        const ctx_str = if (ctx_len > 0)
            try std.fmt.allocPrint(allocator, "{d}", .{ctx_len})
        else
            try std.fmt.allocPrint(allocator, "null", .{});
        defer allocator.free(ctx_str);

        const has_chat = !config.is_encoder_only and chat_config.chat_template.len > 0;
        const has_vision = entry.vision_encoder != null;
        const has_audio = if (entry.vision_encoder) |ve| ve.supportsAudio() else false;
        var caps = try readyCapsJson(allocator, .{
            .has_chat = has_chat,
            .has_vision = has_vision,
            .has_audio = has_audio,
            .has_reasoning = has_chat and chatTemplateSupportsThinking(chat_config.chat_template),
            .is_encoder_only = config.is_encoder_only,
            .has_image_engine = entry.image_engine != null,
            .has_audio_engine = entry.audio_engine != null,
            .has_music_backend = if (entry.audio_engine) |ae| ae.backend == .music else false,
            .has_video_engine = entry.video_engine != null,
            .has_mesh_engine = entry.mesh_engine != null,
        });
        defer caps.deinit(allocator);

        var mods = std.ArrayList(u8).empty;
        defer mods.deinit(allocator);
        try mods.appendSlice(allocator, "[\"text\"");
        if (has_vision) try mods.appendSlice(allocator, ",\"image\"");
        if (has_audio) try mods.appendSlice(allocator, ",\"audio\"");
        try mods.append(allocator, ']');

        const model_id: []const u8 = if (entry.id.len > 0) entry.id else config.model_type;
        const drafter_loaded = entry.drafter != null;
        const mtp_loaded = entry.mtp != null;
        const drafter_path_json = if (drafter_loaded)
            try jsonEscape(allocator, entry.drafter_path)
        else
            try allocator.dupe(u8, "null");
        defer allocator.free(drafter_path_json);
        const bytes_on_disk_str = if (entry.bytes_on_disk) |b|
            try std.fmt.allocPrint(allocator, "{d}", .{b})
        else
            try allocator.dupe(u8, "null");
        defer allocator.free(bytes_on_disk_str);

        // Model-author sampling recommendations from generation_config.json,
        // surfaced so the Swift Settings UI can show "model recommends" pills
        // next to the per-request sampling sliders. `null` when the model
        // ships no value for that field.
        const gen_temp_str = try optSamplingRecJson(allocator, f32, config.gen_temperature);
        defer allocator.free(gen_temp_str);
        const gen_top_p_str = try optSamplingRecJson(allocator, f32, config.gen_top_p);
        defer allocator.free(gen_top_p_str);
        const gen_top_k_str = try optSamplingRecJson(allocator, u32, config.gen_top_k);
        defer allocator.free(gen_top_k_str);

        return std.fmt.allocPrint(allocator,
            \\{{"id":"{s}","object":"model","created":{d},"owned_by":"mlx-serve","loaded":true,"state":"ready","bytes_resident":{d},"bytes_on_disk":{s},"capabilities":{s},"input_modalities":{s},"meta":{{"architecture":"{s}","vocab_size":{d},"hidden_size":{d},"num_layers":{d},"quantization":"{d}-bit","context_length":{s},"model_max_tokens":{d},"is_moe":{s},"drafter_loaded":{s},"drafter_path":{s},"mtp_loaded":{s},"gen_temperature":{s},"gen_top_p":{s},"gen_top_k":{s}}}}}
        , .{
            model_id,
            nowSecs(io),
            entry.bytes_resident,
            bytes_on_disk_str,
            caps.items,
            mods.items,
            config.model_type,
            config.vocab_size,
            config.hidden_size,
            config.num_hidden_layers,
            config.quant_bits,
            ctx_str,
            config.max_position_embeddings,
            if (config.isMoe()) "true" else "false",
            if (drafter_loaded) "true" else "false",
            drafter_path_json,
            if (mtp_loaded) "true" else "false",
            gen_temp_str,
            gen_top_p_str,
            gen_top_k_str,
        });
    }

    // Non-ready entry: stub form with state + bytes_on_disk.
    const state_str = switch (entry.state) {
        .unloaded => "unloaded",
        .loading => "loading",
        .ready => "ready",
        .error_state => "error",
        .evicting => "evicting",
    };
    const bytes_on_disk_str = if (entry.bytes_on_disk) |b|
        try std.fmt.allocPrint(allocator, "{d}", .{b})
    else
        try allocator.dupe(u8, "null");
    defer allocator.free(bytes_on_disk_str);
    const err_part: []const u8 = if (entry.error_name) |name| blk: {
        // Inline escape to avoid double allocation; the names we emit
        // never contain quotes/backslashes (they're @errorName output).
        break :blk try std.fmt.allocPrint(allocator,
            ",\"error\":\"{s}\"",
            .{name},
        );
    } else &[_]u8{};
    defer if (err_part.len > 0) allocator.free(err_part);
    // Stub metadata sourced from config.json (+ chat-template presence) WITHOUT
    // faulting in weights: dims, context window, MoE-ness, and capabilities.
    // `/v1/models` isn't hot, so reading a small JSON per unloaded entry is fine.
    const sm = model_discovery.readStubMeta(io, allocator, entry.path);

    // Capabilities. Encoder-only models advertise "embeddings"; chat models get
    // chat/tool_use/streaming/json_schema (gated on chat-template presence, the
    // same rule the loaded path uses) plus "vision" when a vision config is
    // present. Reasoning/audio stay load-gated (they need the live template /
    // encoder), so an unloaded stub may under-report those two.
    const is_encoder_stub = std.mem.eql(u8, entry.arch_hint, "bert");
    // GGUF discovery stub (issue #59): no config.json to read StubMeta from,
    // but the embedded llama/ds4 engines always serve chat (the GGUF's own
    // template is adopted at load), so advertise the chat capability set the
    // ready path would.
    const is_gguf_stub = std.mem.eql(u8, entry.arch_hint, "gguf");
    // Native media-gen stub (image/audio/video) — advertise the modality from
    // the discovery-peeked arch_hint so the app can find it before loading.
    const media_modality = media_mod.modalityFromType(entry.arch_hint);
    const caps_part: []const u8 = blk: {
        if (media_modality) |m| {
            // The music backend advertises "music" beside "audio" on the stub
            // too (matches the ready-path readyCapsJson additive rule).
            if (m == .audio and media_mod.audioBackendKindForType(entry.arch_hint) == .music)
                break :blk try allocator.dupe(u8, ",\"capabilities\":[\"audio\",\"music\"]");
            break :blk try std.fmt.allocPrint(allocator, ",\"capabilities\":[\"{s}\"]", .{m.capability()});
        }
        if (is_encoder_stub) break :blk try allocator.dupe(u8, ",\"capabilities\":[\"embeddings\"]");
        if (is_gguf_stub) break :blk try allocator.dupe(u8, ",\"capabilities\":[\"chat\",\"tool_use\",\"streaming\",\"json_schema\"]");
        if (!sm.found or !(sm.has_chat or sm.has_vision)) break :blk try allocator.dupe(u8, "");
        var b = std.ArrayList(u8).empty;
        errdefer b.deinit(allocator);
        try b.appendSlice(allocator, ",\"capabilities\":[");
        var n: usize = 0;
        const add = struct {
            fn call(a: std.mem.Allocator, list: *std.ArrayList(u8), cnt: *usize, name: []const u8) !void {
                if (cnt.* > 0) try list.append(a, ',');
                try list.append(a, '"');
                try list.appendSlice(a, name);
                try list.append(a, '"');
                cnt.* += 1;
            }
        }.call;
        if (sm.has_chat) {
            try add(allocator, &b, &n, "chat");
            try add(allocator, &b, &n, "tool_use");
            try add(allocator, &b, &n, "streaming");
            try add(allocator, &b, &n, "json_schema");
        }
        if (sm.has_vision) try add(allocator, &b, &n, "vision");
        try b.append(allocator, ']');
        break :blk try b.toOwnedSlice(allocator);
    };
    defer allocator.free(caps_part);

    const mods_part: []const u8 = if (sm.found and sm.has_vision)
        ",\"input_modalities\":[\"text\",\"image\"]"
    else if ((sm.found and sm.has_chat) or is_gguf_stub)
        ",\"input_modalities\":[\"text\"]"
    else
        "";

    const arch_part: []const u8 = if (entry.arch_hint.len > 0) blk: {
        // arch_hint comes from config.json's model_type via discovery — the
        // supported-type allowlist guarantees no JSON metacharacters.
        break :blk try std.fmt.allocPrint(allocator, "\"architecture\":\"{s}\",", .{entry.arch_hint});
    } else &[_]u8{};
    defer if (arch_part.len > 0) allocator.free(arch_part);

    // Dimensions/context/quant/MoE — emitted only when config.json was readable.
    const dims_part: []const u8 = if (sm.found) blk: {
        break :blk try std.fmt.allocPrint(allocator, "\"vocab_size\":{d},\"hidden_size\":{d},\"num_layers\":{d},\"quantization\":\"{d}-bit\",\"context_length\":{d},\"model_max_tokens\":{d},\"is_moe\":{s},", .{
            sm.vocab_size,
            sm.hidden_size,
            sm.num_hidden_layers,
            sm.quant_bits,
            sm.max_position_embeddings,
            sm.max_position_embeddings,
            if (sm.is_moe) "true" else "false",
        });
    } else &[_]u8{};
    defer if (dims_part.len > 0) allocator.free(dims_part);

    return std.fmt.allocPrint(allocator,
        \\{{"id":"{s}","object":"model","created":0,"owned_by":"mlx-serve","loaded":false,"state":"{s}","bytes_resident":0,"bytes_on_disk":{s}{s}{s}{s},"meta":{{{s}{s}"bytes_on_disk":{s}}}}}
    , .{ entry.id, state_str, bytes_on_disk_str, err_part, caps_part, mods_part, arch_part, dims_part, bytes_on_disk_str });
}

fn handleModels(
    allocator: std.mem.Allocator,
    stream: *Conn,
) !void {
    // Plan 05 Phase E: emit every registry entry (loaded + unloaded), not
    // just the default model + flat discovery list. Default model is sorted
    // first so single-model clients reading `data[0]` continue to work.
    // The mutex is held for the whole render so eviction can't fire under
    // us; each rendered entry is at most a few hundred bytes. This handler
    // does NOT route through `scheduler.ensureLoaded` — listing metadata
    // shouldn't trigger a cold load of the default model.
    const registry = global_registry orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Registry not ready", 503);
        return;
    };

    var entries_buf = std.ArrayList(u8).empty;
    defer entries_buf.deinit(allocator);

    // Collect entries while holding the mutex so iteration is safe.
    // Sort: default first, then by last_used_ns desc.
    var ordered = std.ArrayList(*LoadedModel).empty;
    defer ordered.deinit(allocator);
    {
        registry.mutex.lockUncancelable(stream.io);
        defer registry.mutex.unlock(stream.io);
        try ordered.ensureTotalCapacity(allocator, registry.entries.count());
        var it = registry.entries.valueIterator();
        while (it.next()) |entry_ptr| ordered.appendAssumeCapacity(entry_ptr.*);
        const default_id = registry.default_id;
        const Cmp = struct {
            fn lt(ctx: []const u8, a: *LoadedModel, b: *LoadedModel) bool {
                const a_def = std.mem.eql(u8, a.id, ctx);
                const b_def = std.mem.eql(u8, b.id, ctx);
                if (a_def != b_def) return a_def;
                return a.last_used_ns > b.last_used_ns;
            }
        };
        std.sort.pdq(*LoadedModel, ordered.items, default_id, Cmp.lt);

        for (ordered.items, 0..) |entry, idx| {
            if (idx > 0) try entries_buf.append(allocator, ',');
            const json = try renderModelEntry(allocator, stream.io, entry);
            defer allocator.free(json);
            try entries_buf.appendSlice(allocator, json);
        }
    }

    const body = try std.fmt.allocPrint(allocator,
        \\{{"object":"list","data":[{s}]}}
    , .{entries_buf.items});
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

/// Plan 05 Phase E: `POST /v1/load-model`. Renders a status payload for the
/// resolved-and-loaded `lm` (the dispatcher already called
/// `scheduler.ensureLoaded` against the body's `model` field, so the load
/// has actually happened by the time we get here). Returns the single
/// loaded entry with resident bytes + state so clients can confirm.
/// Plan 05 Phase E: explicit cold-load. Strict on unknown ids (404 — the
/// caller asked for a specific id, not "anything"). Routes through
/// scheduler.ensureLoaded so eviction/back-pressure kicks in.
fn handleLoadModelStrict(allocator: std.mem.Allocator, stream: *Conn, request_body: []const u8) !void {
    const scheduler = global_scheduler orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Scheduler not ready", 503);
        return;
    };
    // Parse the model field with std.json (NOT the quick scanner): clients
    // like Swift's JSONSerialization legally escape '/' as '\/', and the raw
    // scanner slice would read "\/Users\/…" — missing the absolute-path
    // branch below and 404ing on the mangled id (live failure 2026-06-12).
    var requested_id: []const u8 = "";
    var parsed_body: ?std.json.Parsed(std.json.Value) = null;
    defer if (parsed_body) |*p| p.deinit();
    if (std.json.parseFromSlice(std.json.Value, allocator, request_body, .{})) |parsed| {
        parsed_body = parsed;
        if (parsed.value == .object) {
            if (parsed.value.object.get("model")) |m| {
                if (m == .string) requested_id = m.string;
            }
        }
    } else |_| {
        requested_id = parseModelFromBody(request_body) orelse "";
    }
    // Register-by-path: an absolute path to a model directory OUTSIDE the
    // --model-dir scan (e.g. the app's auto-downloaded embedding encoder).
    // The dir is validated exactly like discovery (config.json, supported
    // model_type + quant mode) before anything is registered; the load then
    // proceeds under the directory-basename id.
    if (std.mem.startsWith(u8, requested_id, "/")) {
        const registry = global_registry orelse {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Registry not ready", 503);
            return;
        };
        requested_id = registry.registerByPath(stream.io, requested_id) catch |err| switch (err) {
            error.ModelDirNotFound, error.InvalidModelPath => {
                try sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "No loadable model directory at that path", 404);
                return;
            },
            error.UnsupportedArch => {
                try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Model at that path has an unsupported model_type", 400);
                return;
            },
            error.UnsupportedQuantMode => {
                try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Model at that path has an unsupported quantization mode", 400);
                return;
            },
            else => return err,
        };
    }
    const lm = scheduler.ensureLoaded(requested_id) catch |err| switch (err) {
        error.UnknownModelId => {
            try sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "Unknown model id", 404);
            return;
        },
        error.NoDefaultModel => {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "no_model", "No default model configured", 503);
            return;
        },
        error.NotEnoughMemory => {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "out_of_memory", "Not enough memory to load model; retry after current requests complete", 503);
            return;
        },
        error.LoadFailed => {
            try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", "Model load failed", 500);
            return;
        },
        else => {
            log.warn("  -> 500 ({s}) on /v1/load-model\n", .{@errorName(err)});
            const msg = std.fmt.allocPrint(allocator, "Failed to load model: {s}", .{@errorName(err)}) catch {
                try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", "Failed to load model", 500);
                return;
            };
            defer allocator.free(msg);
            try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", msg, 500);
            return;
        },
    };
    defer scheduler.release(lm);
    const config = lm.config orelse {
        try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_not_ready", "Loaded model has no parsed config", 500);
        return;
    };
    const model_id: []const u8 = if (lm.id.len > 0) lm.id else config.model_type;
    const bytes_resident = lm.bytes_resident;
    const bytes_on_disk_str = if (lm.bytes_on_disk) |b|
        try std.fmt.allocPrint(allocator, "{d}", .{b})
    else
        try allocator.dupe(u8, "null");
    defer allocator.free(bytes_on_disk_str);
    const drafter_loaded = lm.drafter != null;
    const drafter_path_json = if (drafter_loaded)
        try jsonEscape(allocator, lm.drafter_path)
    else
        try allocator.dupe(u8, "null");
    defer allocator.free(drafter_path_json);

    const body = try std.fmt.allocPrint(allocator,
        \\{{"model":{{"id":"{s}","object":"model","loaded":true,"state":"ready","bytes_resident":{d},"bytes_on_disk":{s},"drafter_loaded":{s},"drafter_path":{s}}}}}
    , .{
        model_id,
        bytes_resident,
        bytes_on_disk_str,
        if (drafter_loaded) "true" else "false",
        drafter_path_json,
    });
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

/// Media-generation job payload. Carries the connection + body + resolved
/// model into the inference thread, where `genJobRun` runs the actual
/// generation (mlx + SSE writes) on the GPU-stream-owning thread.
const GenJob = struct {
    allocator: std.mem.Allocator,
    conn: *Conn,
    body: []const u8,
    lm: *model_registry_mod.LoadedModel,
    route: media_mod.GenRoute,
};

/// Inference-thread entry point for a gen job. Dispatches on the engine slot
/// and writes the full HTTP/SSE response to the (parked) connection.
fn genJobRun(ctx: *anyopaque) void {
    const job: *GenJob = @ptrCast(@alignCast(ctx));
    const result = switch (job.route) {
        .image => if (job.lm.image_engine) |e| media_mod.handleImage(job.conn.io, job.allocator, job.conn, job.body, e) else error.WrongModality,
        .speech => if (job.lm.audio_engine) |e| media_mod.handleAudio(job.allocator, job.conn, job.body, e) else error.WrongModality,
        .music => if (job.lm.audio_engine) |e| media_mod.handleMusic(job.allocator, job.conn, job.body, e) else error.WrongModality,
        .video => if (job.lm.video_engine) |e| media_mod.handleVideo(job.conn.io, job.allocator, job.conn, job.body, e) else error.WrongModality,
        .mesh => if (job.lm.mesh_engine) |e| media_mod.handleMesh(job.allocator, job.conn, job.body, e) else error.WrongModality,
    };
    result catch |err| {
        log.warn("[gen] {s} job failed: {s}\n", .{ @tagName(job.route), @errorName(err) });
    };
}

/// Dispatch a media-generation request to the inference thread. `lm` is the
/// already-resolved + refcounted model (so it can't be evicted mid-gen). The
/// generation runs on the scheduler's inference thread (the sole mlx caller);
/// the body writes its own response. A wrong-modality target gets a clear 400.
fn handleGen(allocator: std.mem.Allocator, stream: *Conn, body: []const u8, lm: *model_registry_mod.LoadedModel, route: media_mod.GenRoute) !void {
    const scheduler = global_scheduler orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Scheduler not ready", 503);
        return;
    };
    const ok = switch (route.modality()) {
        .image => lm.image_engine != null,
        .audio => lm.audio_engine != null,
        .video => lm.video_engine != null,
        .mesh => lm.mesh_engine != null,
    };
    if (!ok) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Target model does not support this media modality. Load the matching image/audio/video/3D model and target it by id.", 400);
        return;
    }
    var job = GenJob{ .allocator = allocator, .conn = stream, .body = body, .lm = lm, .route = route };
    var req = scheduler_mod.GenRequest{ .ctx = &job, .run = genJobRun, .model = lm };
    scheduler.runGeneration(&req) catch |err| switch (err) {
        error.Shutdown => {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "shutting_down", "Server is shutting down", 503);
            return;
        },
        else => return err,
    };
    // The job body wrote the full HTTP/SSE response on the inference thread.
}

/// `POST /v1/unload-model` `{"model": id}`. Frees the model's resident GPU
/// state (the stub stays registered so it can reload). Idempotent. Returns a
/// small status payload confirming the model is unloaded.
fn handleUnloadModelStrict(allocator: std.mem.Allocator, stream: *Conn, request_body: []const u8) !void {
    const scheduler = global_scheduler orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Scheduler not ready", 503);
        return;
    };
    // Parse the model id with std.json (handles '\/'-escaped absolute paths
    // the same way /v1/load-model does), then reduce an absolute path to its
    // basename — the registry keys media + by-path models by dir basename.
    var requested_id: []const u8 = "";
    var parsed_body: ?std.json.Parsed(std.json.Value) = null;
    defer if (parsed_body) |*p| p.deinit();
    if (std.json.parseFromSlice(std.json.Value, allocator, request_body, .{})) |parsed| {
        parsed_body = parsed;
        if (parsed.value == .object) {
            if (parsed.value.object.get("model")) |m| {
                if (m == .string) requested_id = m.string;
            }
        }
    } else |_| {
        requested_id = parseModelFromBody(request_body) orelse "";
    }
    if (std.mem.startsWith(u8, requested_id, "/")) {
        var trimmed = requested_id;
        while (trimmed.len > 0 and trimmed[trimmed.len - 1] == '/') trimmed = trimmed[0 .. trimmed.len - 1];
        requested_id = std.fs.path.basename(trimmed);
    }

    scheduler.unloadModel(requested_id) catch |err| switch (err) {
        error.UnknownModelId => {
            try sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "Unknown model id", 404);
            return;
        },
        error.NoDefaultModel => {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "no_model", "No default model configured", 503);
            return;
        },
        error.Shutdown => {
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "shutting_down", "Server is shutting down", 503);
            return;
        },
        else => {
            try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_unload_failed", "Model unload failed", 500);
            return;
        },
    };

    const id_for_body = if (requested_id.len > 0) requested_id else "mlx-serve";
    const body = try std.fmt.allocPrint(allocator,
        \\{{"model":{{"id":"{s}","object":"model","loaded":false,"state":"unloaded"}}}}
    , .{id_for_body});
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

/// Pure body-builder for `GET /props`. Split out from `handleProps` so it
/// can be unit-tested without standing up a real `LoadedModel`. Caller owns
/// the returned slice.
///
/// Note: the `chat_template` field that upstream llama-server exposes here
/// is intentionally omitted. Our Swift app never read it, and stock chat
/// templates are 10–100 KB of Jinja — that turned every /props poll into
/// wasted bandwidth. Capabilities still come from `/v1/models` per entry;
/// clients that genuinely need the template can still get it from the
/// model's `config.json` / `chat_template.jinja` on disk, or by rendering
/// through `/v1/chat/completions` server-side.
fn renderPropsBody(
    allocator: std.mem.Allocator,
    config: *const model_mod.ModelConfig,
    ctx_str: []const u8,
    active_mem: usize,
    peak_mem: usize,
    available_mem: u64,
    safe_ctx: u32,
) ![]u8 {
    // `available_bytes` is free SYSTEM RAM, computed with the SAME formula the
    // model-load pre-flight uses (`metrics.getAvailableMemBytes`), so the tray's
    // "Free RAM" line can never drift from the number that gates a load. Distinct
    // axis from `active_bytes` (the MLX GPU-allocator footprint).
    return std.fmt.allocPrint(allocator,
        \\{{"default_generation_settings":{{"model":"{s}","n_ctx":{s}}},"total_slots":1,"model_info":{{"vocab_size":{d},"hidden_size":{d},"num_hidden_layers":{d},"num_attention_heads":{d},"num_key_value_heads":{d},"head_dim":{d},"quantization_bits":{d},"quantization_group_size":{d},"max_position_embeddings":{d}}},"memory":{{"active_bytes":{d},"peak_bytes":{d},"available_bytes":{d},"max_safe_context":{d}}}}}
    , .{
        config.model_type,        ctx_str,
        config.vocab_size,        config.hidden_size,
        config.num_hidden_layers, config.num_attention_heads,
        config.num_key_value_heads, config.head_dim,
        config.quant_bits,        config.quant_group_size,
        config.max_position_embeddings,
        active_mem,               peak_mem,
        available_mem,            safe_ctx,
    });
}

fn handleProps(allocator: std.mem.Allocator, stream: *Conn, lm: *LoadedModel) !void {
    const config = lm.config.?;
    const ctx_len = getEffectiveContextLength(config);
    const ctx_str = if (ctx_len > 0) blk: {
        break :blk try std.fmt.allocPrint(allocator, "{d}", .{ctx_len});
    } else try std.fmt.allocPrint(allocator, "0", .{});
    defer allocator.free(ctx_str);

    const safe_ctx = computeMaxSafeContext(config);

    // Query memory usage. The MLX path uses mlx's allocator counters; the
    // ds4 path bypasses MLX entirely (no allocator hook to query), so we
    // fall back to the GGUF on-disk size + the ds4 context-memory estimate.
    // Without this branch the Swift app's "GPU Memory" progress bar shows a
    // 0/0 indeterminate state for the entire DSV4 session.
    var active_mem: usize = 0;
    var peak_mem: usize = 0;
    if (lm.ds4_engine != null) {
        // Static estimate: GGUF mmap size (set on the entry at registry-stub
        // time in `runDs4Serve`) plus ds4's reported KV/scratch for the
        // current ctx. Falls back to ctx-only if bytes_on_disk wasn't
        // populated (shouldn't happen in practice, defensive).
        const gguf_bytes: u64 = lm.bytes_on_disk orelse 0;
        const ctx_for_estimate: c_int = @intCast(if (ctx_len > 0) ctx_len else config.max_position_embeddings);
        const ctx_mem = ds4_ffi.ds4_context_memory_estimate(.metal, ctx_for_estimate);
        const total: u64 = gguf_bytes + ctx_mem.total_bytes;
        active_mem = @intCast(total);
        peak_mem = active_mem;
    } else if (lm.llama_engine != null) {
        // llama.cpp owns its own (Metal) allocations outside MLX's counters.
        // Use the GGUF on-disk size as the headline figure so the app's GPU
        // memory bar isn't stuck at 0/0; KV/scratch isn't separately exposed.
        const gguf_bytes: u64 = lm.bytes_on_disk orelse 0;
        active_mem = @intCast(gguf_bytes);
        peak_mem = active_mem;
    } else {
        _ = mlx.mlx_get_active_memory(&active_mem);
        _ = mlx.mlx_get_peak_memory(&peak_mem);
    }

    // Free system RAM — same calc as the model-load pre-flight, so the app's
    // "Free RAM" line stays in lockstep with what gates a load.
    const available_mem = metrics.getAvailableMemBytes();

    const body = try renderPropsBody(allocator, config, ctx_str, active_mem, peak_mem, available_mem, safe_ctx);
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

/// Memory-only `/props` for a boot with no default chat model (headless
/// media-gen serving). Same `memory` object shape as `renderPropsBody` —
/// `MemoryInfo.parse` client-side reads only that key — with the model
/// fields omitted (there is no model config to describe).
fn handlePropsNoModel(allocator: std.mem.Allocator, stream: *Conn) !void {
    var active_mem: usize = 0;
    var peak_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    _ = mlx.mlx_get_peak_memory(&peak_mem);
    const available_mem = metrics.getAvailableMemBytes();
    const body = try std.fmt.allocPrint(allocator,
        \\{{"total_slots":1,"memory":{{"active_bytes":{d},"peak_bytes":{d},"available_bytes":{d},"max_safe_context":0}}}}
    , .{ active_mem, peak_mem, available_mem });
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
    lm: *LoadedModel,
) !void {
    const config = lm.config.?;
    const chat_config = lm.chat_config.?;

    // Memory + capabilities
    var active_mem: usize = 0;
    var peak_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    _ = mlx.mlx_get_peak_memory(&peak_mem);
    const ctx_len = getEffectiveContextLength(config);
    const has_chat = !config.is_encoder_only and chat_config.chat_template.len > 0;
    const has_vision = lm.vision_encoder != null;
    const has_reasoning = has_chat and chatTemplateSupportsThinking(chat_config.chat_template);

    const model_id: []const u8 = if (lm.id.len > 0) lm.id else config.model_type;
    const model_id_esc = try htmlEscape(allocator, model_id);
    defer allocator.free(model_id_esc);
    const arch_esc = try htmlEscape(allocator, config.model_type);
    defer allocator.free(arch_esc);
    const version_esc = try htmlEscape(allocator, build_options.version);
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

    // Optional live-metrics panel: a mount div + the polling script (which also
    // carries the panel markup and injects it into the mount). Rendered into the
    // `{s}` slot right after the "Loaded model" card — but ONLY when --metrics is
    // on. Off ⇒ empty string, so the page is byte-identical to before. This is a
    // runtime `{s}` arg, so the JS may contain raw `{`/`}` — std.fmt does not
    // re-parse it (which is exactly why the panel/JS can't live inline in the
    // std.fmt-formatted index.html).
    const METRICS_SECTION = "\n<div id=mlx-metrics></div>\n<script>\n" ++ @embedFile("html/metrics.js") ++ "\n</script>\n";
    const metrics_section: []const u8 = if (g_metrics != null) METRICS_SECTION else "";

    // The page template lives in src/html/index.html (@embedFile resolves
    // relative to this source file, so no build.zig change). It is a std.fmt
    // format string: `{{`/`}}` are literal braces in the <style> block; the
    // `{s}`/`{d}` slots below fill in model fields, the metrics section, the
    // curl port + model, and the footer — in this exact order.
    const body = try std.fmt.allocPrint(allocator, @embedFile("html/index.html"), .{
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
        // optional live-metrics panel (empty when --metrics is off)
        metrics_section,
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
    lm: *LoadedModel,
) !void {
    // Optional: engine-backed (GGUF/ds4) models have no MLX transformer. The
    // scheduler path doesn't need it; only the no-scheduler fallback does, and
    // it guards on this being present.
    const xfm_opt = lm.transformer;
    const tok = lm.tokenizer.?;
    const config = lm.config.?;
    const gen_mod = @import("generate.zig");
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", null);
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", null);
        return;
    }
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

    // Tokenize every input up front so the whole request rides ONE scheduler
    // round-trip; the inference thread embeds the sequences in padded,
    // key-masked GPU batches (generate.EMBED_MAX_BATCH per forward) instead
    // of one forward per text.
    var total_tokens: usize = 0;
    var seqs = std.ArrayList([]const u32).empty;
    defer {
        for (seqs.items) |ids| allocator.free(ids);
        seqs.deinit(allocator);
    }
    for (texts.items) |text| {
        const ids = try tok.encode(allocator, text);
        total_tokens += ids.len;
        try seqs.append(allocator, ids);
    }

    // Phase A: route through scheduler when available so the encoder
    // forward pass runs on the inference thread (mlx 0.31.2 thread-local
    // streams). Falls back to a direct call only in the offline path
    // where no scheduler exists. Cache reset is handled inside the
    // scheduler's `runEmbedRequest` (or here for the fallback) —
    // encoder-only embeddings carry no cross-request state.
    const embeddings = if (global_scheduler) |sch| blk: {
        var req = scheduler_mod.EmbedRequest{
            .model = lm,
            .token_seqs = seqs.items,
            .allocator = allocator,
        };
        break :blk sch.computeEmbeddings(&req) catch |err| {
            if (req.error_name) |e| {
                log.err("  embedding error: {s}\n", .{e});
                allocator.free(e);
            } else {
                log.err("  embedding error: {}\n", .{err});
            }
            try sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", "Failed to compute embedding", null);
            return;
        };
    } else fallback: {
        const xfm = xfm_opt orelse {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Embeddings require an MLX (safetensors) model; this model has no encoder", null);
            return;
        };
        try xfm.resetCache();
        break :fallback gen_mod.computeEmbeddingsBatch(allocator, xfm, seqs.items) catch |err| {
            log.err("  embedding error: {}\n", .{err});
            try sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", "Failed to compute embedding", null);
            return;
        };
    };
    defer {
        for (embeddings) |e| allocator.free(e);
        allocator.free(embeddings);
    }

    for (embeddings, 0..) |embedding, idx| {
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
    lm: *LoadedModel,
) !void {
    const tok = lm.tokenizer.?;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON", 400);
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
    const root = parsed.value.object;

    const content = if (root.get("content")) |v| (if (v == .string) v.string else null) else null;
    if (content == null) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'content' is required", 400);
        return;
    }

    const ids = if (lm.ds4_engine) |engine| blk: {
        const i32_ids = try engine.tokenizeText(allocator, content.?);
        defer allocator.free(i32_ids);
        const out = try allocator.alloc(u32, i32_ids.len);
        for (i32_ids, 0..) |t, i| out[i] = @intCast(t);
        break :blk out;
    } else if (lm.llama_engine) |engine| blk: {
        const i32_ids = try engine.tokenizeText(allocator, content.?, true);
        defer allocator.free(i32_ids);
        const out = try allocator.alloc(u32, i32_ids.len);
        for (i32_ids, 0..) |t, i| out[i] = @intCast(t);
        break :blk out;
    } else try tok.encode(allocator, content.?);
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
    lm: *LoadedModel,
) !void {
    const tok = lm.tokenizer.?;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON", 400);
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
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

    const text = try decodeTokens(allocator, lm, tok, ids.items, false);
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
    lm: *LoadedModel,
) !void {
    // NOTE: no `lm.transformer.?` here — this handler also serves engine-backed
    // models (GGUF/llama, ds4) whose `transformer` is null. The only MLX-specific
    // gate below reads `config.has_hybrid_layers` (valid for every model incl. the
    // GGUF stub), so the transformer is never needed at this level.
    const tok = lm.tokenizer.?;
    const chat_config = lm.chat_config.?;
    const config = lm.config.?;
    // Parse JSON body
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST /v1/chat/completions -> 400 (invalid JSON)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", 400);
        return;
    };
    defer parsed.deinit();

    // A valid-JSON body that isn't an object (e.g. `42` or `[1,2]`) would panic
    // on the `.object` field access below and take the whole server down, so
    // reject it as a 400 like any other malformed request.
    if (parsed.value != .object) {
        log.warn("POST /v1/chat/completions -> 400 (body is not a JSON object)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
    const root = parsed.value.object;

    // Extract messages
    const messages_val = root.get("messages") orelse {
        log.warn("POST /v1/chat/completions -> 400 (missing messages)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'messages' is a required field", 400);
        return;
    };

    // `messages` present but not an array (e.g. a string) would panic on the
    // `.array` access below. Reject rather than crash.
    if (messages_val != .array) {
        log.warn("POST /v1/chat/completions -> 400 (messages is not an array)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'messages' must be an array", 400);
        return;
    }

    var messages = std.ArrayList(chat_mod.Message).empty;
    defer messages.deinit(allocator);

    // Parse tool call structs for assistant messages (stored temporarily)
    var tool_call_lists = std.ArrayList([]const chat_mod.ToolCall).empty;
    defer {
        for (tool_call_lists.items) |tcs| allocator.free(tcs);
        tool_call_lists.deinit(allocator);
    }

    for (messages_val.array.items) |msg_val| {
        // A non-object array element (e.g. `messages:[1,2,3]`) would panic on
        // `.object`. Skip it rather than crash — consistent with how malformed
        // inner fields are already tolerated below.
        if (msg_val != .object) continue;
        const obj = msg_val.object;
        const role_val = obj.get("role") orelse continue;
        if (role_val != .string) continue;

        // Content can be null for assistant messages with tool_calls
        const content_val = obj.get("content");
        var msg_images: ?[]const chat_mod.ImageData = null;
        var msg_audio: ?[]const chat_mod.AudioData = null;
        const content: []const u8 = if (content_val) |cv| switch (cv) {
            .string => |s| s,
            .array => |arr| blk: {
                var text_content: []const u8 = "";
                var image_list = std.ArrayList(chat_mod.ImageData).empty;
                var audio_list = std.ArrayList(chat_mod.AudioData).empty;
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
                        if (parseImageUrlContent(allocator, url_val.string, visionPreprocFromConfig(config))) |img| {
                            image_list.append(allocator, img) catch {
                                allocator.free(img.pixels);
                                continue;
                            };
                        }
                    } else if (std.mem.eql(u8, ptype.string, "input_audio")) {
                        // OpenAI-style audio block. For the Gemma 4 12B unified
                        // engine the client sends raw 16 kHz mono float32-LE PCM
                        // (format "mlx_pcm_f32") base64-encoded in `data`.
                        const a_obj = part.object.get("input_audio") orelse continue;
                        if (a_obj != .object) continue;
                        const data_val = a_obj.object.get("data") orelse continue;
                        if (data_val != .string) continue;
                        if (parseAudioContent(allocator, data_val.string)) |aud| {
                            audio_list.append(allocator, aud) catch {
                                allocator.free(aud.samples);
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
                if (audio_list.items.len > 0) {
                    msg_audio = audio_list.toOwnedSlice(allocator) catch null;
                } else {
                    audio_list.deinit(allocator);
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

        // Skip messages with no content, no tool_calls, and no images/audio
        if (content.len == 0 and msg_tool_calls == null and msg_images == null and msg_audio == null and !std.mem.eql(u8, role_val.string, "tool")) continue;

        try messages.append(allocator, .{
            .role = role_val.string,
            .content = content,
            .tool_calls = msg_tool_calls,
            .tool_call_id = tool_call_id,
            .images = msg_images,
            .audio = msg_audio,
        });
    }

    if (messages.items.len == 0) {
        log.warn("POST /v1/chat/completions -> 400 (no valid messages)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "No valid messages found in request", 400);
        return;
    }

    // Support both max_tokens and max_completion_tokens (OpenAI alias). Absent
    // or <= 0 → auto (peg to remaining context).
    const max_tokens: u32 = resolveRequestMaxTokens(
        root.get("max_tokens") orelse root.get("max_completion_tokens"),
        omittedMaxTokensDefault(getEffectiveContextLength(config)),
    );

    const is_stream = if (root.get("stream")) |v| v == .bool and v.bool else false;

    const temperature = resolveSamplingDefault(f32, parseJsonFloatOpt(root, "temperature", 0.0, 2.0), server_config.default_temperature, config.gen_temperature, 1.0);
    const top_p = resolveSamplingDefault(f32, parseJsonFloatOpt(root, "top_p", 0.0, 1.0), server_config.default_top_p, config.gen_top_p, 1.0);
    const top_k = resolveSamplingDefault(u32, parseJsonTopKOpt(root, "top_k"), server_config.default_top_k, config.gen_top_k, 0);

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
        else => server_config.default_reasoning_budget,
    } else server_config.default_reasoning_budget;

    // Wave 1.A: per-request KV-quant override. When unset, falls back to the
    // process-level --kv-quant default carried on the scheduler. Cross-scheme
    // hot-prefix-cache hits never happen — entries record their scheme and
    // findBestMatch filters on it.
    const kv_quant_override = parseKvQuantOverride(root);
    if (kv_quant_override) |kq| {
        switch (kq.scheme) {
            .off => log.info("  kv-quant override: off (per-request)\n", .{}),
            .affine => log.info("  kv-quant override: affine {d}-bit (per-request)\n", .{kq.bits}),
            .turboquant_2, .turboquant_4 => log.info("  kv-quant override: turboquant {d}-bit (per-request)\n", .{kq.bits}),
        }
    }

    // Parse enable_pld: per-request override of the --pld default.
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
        server_config.default_enable_pld;
    // Tools do NOT disable PLD: tool-pattern detection operates on emitted
    // text and is agnostic to how many tokens a decode step yields, and
    // agent traffic (tool results echoed into edits) is PLD's best workload.
    // The original blanket gate predated streaming PLD + scheduler slots;
    // equivalence with tools is pinned by tests/test_pld_tools.sh.
    //
    // PLD on hybrid SSM models (LFM2.5, Nemotron-H) works once the SSM
    // snapshot/restore paths handle null ssm_state correctly — see
    // `ssmSnapshot`/`ssmRestore` in transformer.zig.
    if (enable_pld) log.info("  pld=enabled (draft_len={d}, key_len={d})\n", .{ server_config.default_pld_draft_len, server_config.default_pld_key_len });

    // Parse enable_drafter: per-request override of the --drafter default.
    // Auto-disabled when:
    //   - the server didn't load a drafter (`default_drafter == null`)
    //   - logprobs are requested (drafter doesn't expose realized log-probs)
    //   - hybrid SSM architecture (same SSM-state issue as PLD; drafter would
    //     hit it on the verify forward).
    // Tools do NOT disable the drafter (same reasoning as PLD above);
    // equivalence with tools is pinned by tests/test_drafter_tools.sh.
    // Priority: drafter > PLD > regular. When drafter wins, force PLD off
    // so logs / state stay consistent.
    const drafter_explicit_in_json: bool = root.get("enable_drafter") != null;
    // Plan 05: per-model drafter — when a drafter is loaded for this model,
    // requests default to ON unless the target is MoE (where verify-forward
    // routing penalty overwhelms the win). Per-request `enable_drafter` JSON
    // overrides either way.
    const lm_default_enable_drafter: bool = lm.drafter != null and !config.isMoe();
    var enable_drafter: bool = if (root.get("enable_drafter")) |v|
        (v == .bool and v.bool)
    else
        lm_default_enable_drafter;
    if (enable_drafter and lm.drafter == null) {
        enable_drafter = false; // no drafter loaded; quietly fall through
    }
    if (enable_drafter and logprobs_n > 0) {
        log.info("  drafter=disabled (logprobs requested)\n", .{});
        enable_drafter = false;
    }
    if (enable_drafter and config.has_hybrid_layers) {
        log.info("  drafter=disabled (hybrid SSM architecture not yet supported for multi-token verify)\n", .{});
        enable_drafter = false;
    }
    // Qwen native MTP head: defaults ON whenever the model loaded one (the
    // sidecar only loads when it binds to this trunk; MoE mirrors the
    // drafter caution). Priority MTP > drafter > PLD at dispatch. NOT
    // subject to the n-gram spec gate below — the trained head holds ~73%
    // per-draft acceptance even on fully novel content.
    var enable_mtp: bool = if (root.get("enable_mtp")) |v|
        (v == .bool and v.bool)
    else
        (lm.mtp != null and !config.isMoe());
    if (enable_mtp and lm.mtp == null) enable_mtp = false;
    if (enable_mtp and logprobs_n > 0) {
        log.info("  mtp=disabled (logprobs requested)\n", .{});
        enable_mtp = false;
    }
    if (enable_drafter) {
        if (enable_pld) {
            log.info("  pld=disabled (drafter takes priority for this request)\n", .{});
            enable_pld = false;
        }
        log.info("  drafter=enabled (block_size={d})\n", .{lm.drafter_block_size});
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

    log.info("POST /v1/chat/completions ({d} msgs, max_tokens={d}, temp={d:.2}, top_p={d:.2}, top_k={d}, stream={}, thinking={}, sys={d}b, user={d}b, tools={d}b, tool_msgs={d}) \n", .{ messages.items.len, max_tokens, temperature, top_p, top_k, is_stream, enable_thinking, system_chars, user_chars, tools_len, tool_msg_count });
    log.info("  > \"{s}{s}\"\n", .{ last_msg.content[0..preview_len], if (last_msg.content.len > 80) "..." else "" });

    // Format chat template. ds4-backed models render through the engine's
    // built-in template/tokenizer; the MLX path renders via Jinja and
    // tokenizes through the loaded BPE tokenizer. Both paths now thread
    // `tools_json` + `tool_choice_instruction` through — the ds4 helper
    // synthesizes a system-message fallback when the GGUF chat template
    // doesn't model `tools` natively (which is the DSV4 case).
    //
    // Phase 4 instrumentation + Iteration 2 cache: time the render+tokenize
    // step. The cache is engine-agnostic — same hit even when the
    // underlying call is ds4 / llama / MLX formatChat.
    var tokenize_sw = Stopwatch.init(stream.io);
    var prompt_ids_raw = try cachedFormatChat(allocator, stream.io, lm, tok, chat_config, messages.items, tools_json, tool_choice_instruction, enable_thinking);
    const tokenize_ns = tokenize_sw.read();

    // Run vision encoder if any messages contain images. Phase A8: each
    // request owns its embedding locally; we hand it off to the slot at
    // submit time. Defer frees if we don't transfer ownership.
    var local_ve: ?mlx.mlx_array = null;
    defer { if (local_ve) |arr| _ = mlx.mlx_array_free(arr); }
    if (lm.vision_encoder) |ve| {
        var n_vis: usize = 0;
        var n_aud: usize = 0;
        local_ve = processVisionImages(allocator, lm, ve, messages.items, &n_vis, &n_aud) catch |err| blk: {
            log.warn("Vision encoding failed: {}\n", .{err});
            break :blk null;
        };
        if (local_ve != null) {
            const new_ids = try insertMultimodalTokens(allocator, prompt_ids_raw, config.image_token_id, n_vis, config.audio_token_id, n_aud, config);
            allocator.free(prompt_ids_raw);
            prompt_ids_raw = new_ids;
        }
    }
    const prompt_ids = prompt_ids_raw;
    defer allocator.free(prompt_ids);

    // Qwen3-VL interleaved M-RoPE: compute the position-id table from the final
    // (image-pad-expanded) prompt + the image grids. Ownership transfers to the
    // slot at submit (mirrors the vision-embeddings handoff below).
    var local_mrope = computeQwenMrope(allocator, prompt_ids, messages.items, config) catch MropeData{};
    defer {
        if (local_mrope.pos) |p| allocator.free(p);
    }

    // Enforce context size limit
    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        log.warn("POST /v1/chat/completions -> 400 (prompt {d} tokens exceeds ctx_size {d})\n", .{ prompt_ids.len, effective_ctx });
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Prompt exceeds maximum context length", 400);
        return;
    }

    // Check if attention computation would exceed GPU memory
    if (!try checkAttentionMemory(allocator, stream, prompt_ids.len, config, false, kv_quant_override)) return;

    // Clamp max_tokens to stay within context window
    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len, effective_ctx);
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
        // NOTE (2026-07-13): the old heavy-echo MTP->PLD routing (score >=
        // 0.13 disabled MTP) was RETIRED with the verify-qmm kernels + EV
        // depth. The prompt-time n-gram score cannot separate "output will
        // echo verbatim" (PLD excels: 83 vs 75 tok/s live) from
        // "repetitive-looking agent context" (PLD flaps at ~45% per-draft,
        // runtime-disables, and lands on plain AR at 28 tok/s with the MTP
        // head idle) — live captures scored 0.334 vs 0.365, inseparable.
        // MTP now wins whenever loaded (generator priority); force PLD with
        // enable_pld:true + enable_mtp:false.
    }

    // Prompt caching: reuse KV cache for shared prefix.
    // Force invalidation when images are present — image tokens have identical IDs
    // but different vision embeddings, so prefix matching would reuse stale features.
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

    // Hand vision ownership off to the sub-handler, which transfers it to
    // the slot at submit time.
    const sub_ve = local_ve;
    local_ve = null;
    const sub_mrope = local_mrope;
    local_mrope = .{}; // ownership transferred to the sub-handler → slot
    if (is_stream) {
        handleStreamingGeneration(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, include_usage, has_tools, tools_json, logprobs_n, enable_thinking, reasoning_budget, enable_pld, enable_drafter, enable_mtp, sub_ve, sub_mrope, kv_quant_override, tokenize_ns) catch |err| {
            log.err("  -> streaming error: {}\n", .{err});
            // Send SSE error event so the client gets a proper error instead of a dropped connection
            const err_chunk = std.fmt.allocPrint(allocator,
                \\data: {{"error":{{"message":"Internal server error: {s}","type":"server_error"}}}}
            , .{@errorName(err)}) catch return;
            defer allocator.free(err_chunk);
            stream.writeAllNoFlush(err_chunk) catch {};
            stream.writeAll("\n\ndata: [DONE]\n\n") catch {};
        };
    } else {
        handleNonStreamingGeneration(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, has_tools, tools_json, logprobs_n, enable_thinking, reasoning_budget, enable_pld, enable_drafter, enable_mtp, sub_ve, sub_mrope, kv_quant_override, tokenize_ns) catch |err| {
            log.err("  -> 500 ({s})\n", .{@errorName(err)});
            sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", @errorName(err), 500) catch {};
        };
    }
}

fn handleCompletions(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    lm: *LoadedModel,
) !void {
    const tok = lm.tokenizer.?;
    const config = lm.config.?;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST /v1/completions -> 400 (invalid JSON)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", 400);
        return;
    };
    defer parsed.deinit();

    if (parsed.value != .object) {
        log.warn("POST /v1/completions -> 400 (body is not a JSON object)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
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

    const max_tokens: u32 = resolveRequestMaxTokens(
        root.get("max_tokens") orelse root.get("max_completion_tokens"),
        omittedMaxTokensDefault(getEffectiveContextLength(config)),
    );

    const is_stream = if (root.get("stream")) |v| v == .bool and v.bool else false;

    const temperature = resolveSamplingDefault(f32, parseJsonFloatOpt(root, "temperature", 0.0, 2.0), server_config.default_temperature, config.gen_temperature, 1.0);
    const top_p = resolveSamplingDefault(f32, parseJsonFloatOpt(root, "top_p", 0.0, 1.0), server_config.default_top_p, config.gen_top_p, 1.0);
    const top_k = resolveSamplingDefault(u32, parseJsonTopKOpt(root, "top_k"), server_config.default_top_k, config.gen_top_k, 0);

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

    // Spec-decode flags (mirror chat-completions). FIM / code-completion
    // prompts are echo-heavy, so the old hardcoded enable_pld/enable_drafter
    // = false at submit left real speedups unused on this endpoint
    // (tests/test_completions_spec.sh). Embedded engines (ds4/llama) ignore
    // these flags at dispatch.
    const pld_explicit_in_json: bool = root.get("enable_pld") != null;
    var enable_pld: bool = if (root.get("enable_pld")) |v|
        (v == .bool and v.bool)
    else
        server_config.default_enable_pld;
    const drafter_explicit_in_json: bool = root.get("enable_drafter") != null;
    var enable_drafter: bool = if (root.get("enable_drafter")) |v|
        (v == .bool and v.bool)
    else
        (lm.drafter != null and !config.isMoe());
    if (enable_drafter and lm.drafter == null) enable_drafter = false;
    if (enable_drafter and config.has_hybrid_layers) enable_drafter = false;
    if (enable_drafter and enable_pld) enable_pld = false;
    var enable_mtp: bool = if (root.get("enable_mtp")) |v|
        (v == .bool and v.bool)
    else
        (lm.mtp != null and !config.isMoe());
    if (enable_mtp and lm.mtp == null) enable_mtp = false;

    // Log the request
    const preview_len = @min(prompt_text.?.len, 80);
    log.info("POST /v1/completions (max_tokens={d}, temp={d:.2}, top_p={d:.2}, top_k={d}, stream={}) \n", .{ max_tokens, temperature, top_p, top_k, is_stream });
    log.info("  > \"{s}{s}\"\n", .{ prompt_text.?[0..preview_len], if (prompt_text.?.len > 80) "..." else "" });

    // Tokenize prompt directly (no chat template). ds4-backed models
    // tokenize through the engine's GGUF vocab; MLX models go through
    // the loaded BPE tokenizer.
    const prompt_ids = if (lm.ds4_engine) |engine| blk: {
        const i32_ids = try engine.tokenizeText(allocator, prompt_text.?);
        defer allocator.free(i32_ids);
        const out = try allocator.alloc(u32, i32_ids.len);
        for (i32_ids, 0..) |t, i| out[i] = @intCast(t);
        break :blk out;
    } else if (lm.llama_engine) |engine| blk: {
        const i32_ids = try engine.tokenizeText(allocator, prompt_text.?, true);
        defer allocator.free(i32_ids);
        const out = try allocator.alloc(u32, i32_ids.len);
        for (i32_ids, 0..) |t, i| out[i] = @intCast(t);
        break :blk out;
    } else try tok.encode(allocator, prompt_text.?);
    defer allocator.free(prompt_ids);

    // Enforce context size limit
    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        log.warn("POST /v1/completions -> 400 (prompt {d} tokens exceeds ctx_size {d})\n", .{ prompt_ids.len, effective_ctx });
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Prompt exceeds maximum context length", 400);
        return;
    }

    // Check if attention computation would exceed GPU memory
    // (/v1/completions has no per-request kv_quant field -> process default.)
    if (!try checkAttentionMemory(allocator, stream, prompt_ids.len, config, false, null)) return;

    // Clamp max_tokens to stay within context window
    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len, effective_ctx);

    // Adaptive spec-decode gate (mirrors chat-completions): novel prompts
    // (low 3-gram repetition) skip PLD/drafter unless explicitly requested.
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
        // Heavy-echo MTP->PLD routing retired 2026-07-13 (see the NOTE at the
        // chat-completions site): MTP wins whenever loaded.
    }

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
        handleStreamingCompletion(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, include_usage, enable_pld, enable_drafter, enable_mtp) catch |err| {
            log.err("  -> streaming error: {}\n", .{err});
        };
    } else {
        handleNonStreamingCompletion(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, enable_pld, enable_drafter, enable_mtp) catch |err| {
            log.err("  -> 500 ({s})\n", .{@errorName(err)});
            sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", @errorName(err), 500) catch {};
        };
    }
}

fn handleNonStreamingCompletion(
    allocator: std.mem.Allocator,
    stream: *Conn,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: generate_mod.SamplingParams,
    eos_token_ids: []const u32,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    enable_pld: bool,
    enable_drafter: bool,
    enable_mtp: bool,
) !void {
    var timer = Stopwatch.init(stream.io);

    // Spec dispatch (priority MTP > drafter > PLD; mirrors handleNonStreamingGeneration).
    const use_mtp = enable_mtp and lm.mtp != null and sampling.constraint == null;
    const use_drafter = !use_mtp and enable_drafter and lm.drafter != null and sampling.constraint == null;
    const use_pld = !use_mtp and !use_drafter and enable_pld and sampling.constraint == null;

    var result = nonStreamingViaScheduler(allocator, global_scheduler.?, lm, tok, prompt_ids, prompt_ids, max_tokens, sampling, eos_token_ids, 0, false, use_pld, use_drafter, use_mtp, getTimeoutNs(), null, .{}, 0, null, stream) catch |err| switch (err) {
        error.GenerationFailed => return sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", "generation failed", null),
        else => return err,
    };
    _ = &result;
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);

    // Text completion is a raw continuation: keep the first token's leading
    // space (SentencePiece `▁`) instead of the chat-style strip the scheduler
    // applies — FIM clients rely on exact indentation, and the streaming
    // handler never stripped it, so this also restores stream/non-stream
    // parity (tests/test_completions_spec.sh).
    const raw_text = try decodeTokens(allocator, lm, tok, result.token_ids, false);
    defer allocator.free(raw_text);

    var final_text: []const u8 = raw_text;
    var finish_reason = result.finish_reason;
    for (stop_sequences) |stop_seq| {
        if (std.mem.indexOf(u8, final_text, stop_seq)) |idx| {
            final_text = final_text[0..idx];
            finish_reason = "stop";
            break;
        }
    }

    const elapsed_ms = timer.read() / std.time.ns_per_ms;
    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns);
    log.info("  <- {d}+{d} tokens ({d}ms) [{s}] [{s}]\n", .{
        result.prompt_tokens, result.completion_tokens, elapsed_ms, perf, finish_reason,
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
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: generate_mod.SamplingParams,
    eos_token_ids: []const u32,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    include_usage: bool,
    enable_pld: bool,
    enable_drafter: bool,
    enable_mtp: bool,
) !void {
    const cmpl_id = nowMs(stream.io);
    const created_ts = nowSecs(stream.io);
    var timer = Stopwatch.init(stream.io);

    const config = lm.config.?;
    const stream_mode = pickStreamMode(enable_pld, enable_drafter, enable_mtp, lm.drafter != null, lm.mtp != null, config.has_hybrid_layers, sampling.constraint != null, 0);
    if (stream_mode == .pld) log.info("  pld=enabled (streaming, draft_len={d}, key_len={d})\n", .{ server_config.default_pld_draft_len, server_config.default_pld_key_len });
    if (stream_mode == .drafter) log.info("  drafter=enabled (streaming, block_size={d})\n", .{lm.drafter_block_size});
    if (stream_mode == .mtp) log.info("  mtp=enabled (streaming, depth={d})\n", .{lm.mtp_depth});

    var slot_handle: ?*scheduler_mod.Slot = null;
    defer if (slot_handle) |s| global_scheduler.?.complete(s);

    const sch = global_scheduler.?;
    slot_handle = try sch.submit(.{
        .model = lm,
        .prompt_ids = prompt_ids,
        .full_prompt = prompt_ids,
        .cached_tokens = 0,
        .has_tools = false,
        .sampling = sampling,
        .eos_token_ids = eos_token_ids,
        .max_tokens = max_tokens,
        .timeout_ns = getTimeoutNs(),
        .enable_pld = stream_mode == .pld,
        .enable_drafter = stream_mode == .drafter,
        .drafter = if (stream_mode == .drafter) lm.drafter else null,
        .drafter_block_size = lm.drafter_block_size,
        .enable_mtp = stream_mode == .mtp,
        .mtp = if (stream_mode == .mtp) lm.mtp else null,
        .mtp_depth = lm.mtp_depth,
        .pld_draft_len = server_config.default_pld_draft_len,
        .pld_key_len = server_config.default_pld_key_len,
        .kv_attn_fused = server_config.default_kv_attn_fused,
        .logprobs_n = 0,
    });
    var ts = StreamingTokenStream.initFromSlot(slot_handle.?, stream_mode, eos_token_ids);
    defer ts.deinit(allocator);

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
    var client_gone = false;

    while (true) {
        const token_id: u32 = switch (try ts.nextOrIdle(allocator, Conn.STREAM_KEEPALIVE_MS)) {
            .token => |t| t,
            .done => break,
            .idle => {
                // No tokens yet (long prefill). Probe the peer: an abandoned
                // request must cancel instead of grinding a ghost prefill
                // (Claude Code retries pile up serially otherwise), and the
                // keepalive stops clients timing out on stream silence.
                if (stream.peerClosed()) {
                    log.info("  [cancel] client disconnected while waiting for tokens — cancelling slot\n", .{});
                    slot_handle.?.cancel();
                    client_gone = true;
                    break;
                }
                sendStreamKeepalive(stream) catch {
                    log.info("  [cancel] keepalive write failed (client disconnected) — cancelling slot\n", .{});
                    slot_handle.?.cancel();
                    client_gone = true;
                    break;
                };
                continue;
            },
        };
        if (stream.peerClosed()) {
            slot_handle.?.cancel();
            client_gone = true;
            break;
        }
        const strip = tok.tok_type == .sentencepiece_bpe;
        const raw_decoded_c = try decodeTokens(allocator, lm, tok, &[_]u32{token_id}, strip and false);

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
        try stream.writeAllNoFlush("data: ");
        try stream.writeAllNoFlush(chunk);
        try stream.writeAllNoFlush("\n\n");
        try stream.flush();
    }

    // Final chunk with finish_reason
    ts.finalize();
    const finish_reason = if (client_gone) "client_disconnect" else if (stopped) "stop" else ts.finish_reason;
    const total_prompt = ts.prompt_tokens;

    if (!client_gone) {
        const usage_str = if (include_usage) blk: {
            break :blk try std.fmt.allocPrint(allocator,
                \\,"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}
            , .{ total_prompt, ts.completion_tokens, total_prompt + ts.completion_tokens });
        } else try std.fmt.allocPrint(allocator, "", .{});
        defer allocator.free(usage_str);

        const final_chunk = try std.fmt.allocPrint(allocator,
            \\{{"id":"cmpl-{d}","object":"text_completion.chunk","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[{{"index":0,"text":"","finish_reason":"{s}"}}]{s}}}
        , .{ cmpl_id, created_ts, model_name, finish_reason, usage_str });
        defer allocator.free(final_chunk);

        logHttpSseData(final_chunk);
        try stream.writeAllNoFlush("data: ");
        try stream.writeAllNoFlush(final_chunk);
        try stream.writeAllNoFlush("\n\n");
        logHttpSseData("[DONE]");
        try stream.writeAll("data: [DONE]\n\n");
    }

    const elapsed_ms = timer.read() / std.time.ns_per_ms;
    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, ts.prompt_tokens, ts.cached_tokens, ts.completion_tokens, ts.prefill_ns, ts.decode_ns);
    log.info("  <- {d}+{d} tokens streamed ({d}ms) [{s}] [{s}]\n", .{
        total_prompt, ts.completion_tokens, elapsed_ms, perf, finish_reason,
    });
}

/// Run a non-streaming generation through the scheduler. Returns the same
/// shape as `generate.generate` so the calling handler's response builder
/// is unchanged.
///
/// `vision_embeddings` (Phase A4/A8): when non-null, ownership transfers
/// into the slot — the slot's deinit frees the array. Each handler holds a
/// per-request `?mlx_array` local; it nulls its copy before passing here so
/// its own `defer` doesn't double-free.
fn nonStreamingViaScheduler(
    allocator: std.mem.Allocator,
    sch: *scheduler_mod.Scheduler,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    full_prompt: []const u32,
    max_tokens: u32,
    sampling: generate_mod.SamplingParams,
    eos_token_ids: []const u32,
    cached_tokens: u32,
    has_tools: bool,
    enable_pld: bool,
    enable_drafter: bool,
    enable_mtp: bool,
    timeout_ns: u64,
    vision_embeddings: ?mlx.mlx_array,
    mrope: MropeData,
    logprobs_n: u32,
    /// Wave 1.A: per-request KV-quant override; null = inherit scheduler default.
    kv_quant_override: ?transformer_mod.KVQuantConfig,
    /// When non-null, the peer socket is probed on idle wakeups during the
    /// wait — a vanished client cancels the slot (aborting its prefill)
    /// instead of grinding out a ghost generation nobody will read.
    conn: ?*Conn,
) !generate_mod.GenerationResult {
    var slot = try sch.submit(.{
        .model = lm,
        .prompt_ids = prompt_ids,
        .full_prompt = full_prompt,
        .cached_tokens = cached_tokens,
        .has_tools = has_tools,
        .sampling = sampling,
        .eos_token_ids = eos_token_ids,
        .max_tokens = max_tokens,
        .timeout_ns = timeout_ns,
        .enable_pld = enable_pld,
        .enable_drafter = enable_drafter and lm.drafter != null,
        .drafter = if (enable_drafter) lm.drafter else null,
        .drafter_block_size = lm.drafter_block_size,
        .enable_mtp = enable_mtp and lm.mtp != null,
        .mtp = if (enable_mtp) lm.mtp else null,
        .mtp_depth = lm.mtp_depth,
        .pld_draft_len = server_config.default_pld_draft_len,
        .pld_key_len = server_config.default_pld_key_len,
        .kv_attn_fused = server_config.default_kv_attn_fused,
        .vision_embeddings = vision_embeddings,
        .mrope_pos = mrope.pos,
        .mrope_total = mrope.total,
        .mrope_delta = mrope.delta,
        .logprobs_n = logprobs_n,
        .kv_quant_config = kv_quant_override,
    });
    defer sch.complete(slot);

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    wait: while (true) {
        const nr = slot.waitNextTimeout(Conn.STREAM_KEEPALIVE_MS) orelse {
            // Idle (long prefill). If the client is gone, cancel and serve
            // whatever accumulated — the response write will fail upstream,
            // which is fine; the win is freeing the GPU.
            if (conn) |c| {
                if (c.peerClosed()) {
                    log.info("  [cancel] client disconnected while waiting (non-stream) — cancelling slot\n", .{});
                    slot.cancel();
                    break :wait;
                }
            }
            continue :wait;
        };
        switch (nr) {
            .token => |t| try output_ids.append(allocator, t),
            .done => break :wait,
            .err => return error.GenerationFailed,
        }
    }

    // The scheduler measures prefill_ns / decode_ns per-slot directly. Pull
    // those instead of the old single-wall-clock approximation, which
    // double-counted total time into both phases.
    const prefill_tps = generate_mod.prefillTokensPerSec(slot.prompt_tokens, slot.cached_tokens, slot.prefill_ns);
    const decode_tps = generate_mod.tokensPerSec(slot.completion_tokens, slot.decode_ns);

    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try decodeTokens(allocator, lm, tok, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);

    // Phase A5: take ownership of the slot's accumulated logprobs. After
    // `toOwnedSlice` the slot's list is empty, so `Slot.deinit` doesn't try
    // to free what we just transferred to the caller.
    const logprobs_slice: ?[]generate_mod.LogprobResult = if (slot.logprobs_buf.items.len > 0)
        slot.logprobs_buf.toOwnedSlice(slot.allocator) catch null
    else
        null;

    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = slot.prompt_tokens,
        .completion_tokens = slot.completion_tokens,
        .finish_reason = slot.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .prefill_ns = slot.prefill_ns,
        .decode_ns = slot.decode_ns,
        .cached_tokens = slot.cached_tokens,
        .logprobs = logprobs_slice,
    };
}

fn handleNonStreamingGeneration(
    allocator: std.mem.Allocator,
    stream: *Conn,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: generate_mod.SamplingParams,
    eos_token_ids: []const u32,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    has_tools: bool,
    /// OpenAI-shape tools JSON (for bare-args tool-call inference); null when
    /// the request defined no tools.
    tools_json: ?[]const u8,
    logprobs_n: u32,
    enable_thinking: bool,
    reasoning_budget: i32,
    enable_pld: bool,
    enable_drafter: bool,
    enable_mtp: bool,
    vision_embeddings: ?mlx.mlx_array,
    mrope: MropeData,
    /// Wave 1.A: per-request KV-quant override; null = inherit scheduler default.
    kv_quant_override: ?transformer_mod.KVQuantConfig,
    /// Iteration 1: tokenize_ns from the parent handleChatCompletions, so
    /// the non-streaming chat response carries `timings.tokenize_ms`.
    tokenize_ns: u64,
) !void {
    // Phase A8: this handler owns the vision array on entry; ownership
    // transfers to the slot on submit (the scheduler's `Slot.deinit` frees
    // it). Nulled before transfer so the early-return defer is a no-op.
    var ve_local = vision_embeddings;
    defer { if (ve_local) |arr| _ = mlx.mlx_array_free(arr); }

    var timer = Stopwatch.init(stream.io);

    // Speculative-decoding dispatch (priority: drafter > PLD > regular).
    //   1. Drafter wins if loaded AND requested AND no logprobs (drafter
    //      cannot expose realized log-probs from the draft side).
    //   2. PLD next if requested AND no logprobs AND no grammar constraint
    //      (constrained decode requires per-token state advancement).
    //   3. Otherwise the regular pipeline.
    const use_mtp = enable_mtp and logprobs_n == 0 and lm.mtp != null and sampling.constraint == null;
    const use_drafter = !use_mtp and enable_drafter and logprobs_n == 0 and lm.drafter != null and sampling.constraint == null;
    const use_pld = !use_mtp and !use_drafter and enable_pld and logprobs_n == 0 and sampling.constraint == null;

    // Transfer vision ownership into the slot via the scheduler.
    const slot_ve: ?mlx.mlx_array = blk: {
        const v = ve_local;
        ve_local = null;
        break :blk v;
    };
    const result = nonStreamingViaScheduler(allocator, global_scheduler.?, lm, tok, prompt_ids, prompt_ids, max_tokens, sampling, eos_token_ids, 0, has_tools, use_pld, use_drafter, use_mtp, getTimeoutNs(), slot_ve, mrope, logprobs_n, kv_quant_override, stream) catch |err| switch (err) {
        error.GenerationFailed => {
            try sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", "generation failed", null);
            return;
        },
        else => return err,
    };
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);
    defer if (result.logprobs) |lps| {
        for (lps) |*lp| allocator.free(lp.top_logprobs);
        allocator.free(lps);
    };

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

    // Merge re-opened mid-text thought channels into the leading block so the
    // split/parse below never leaks raw tags (Gemma 12B re-opens channels mid-turn).
    const normalized_text = try chat_mod.normalizeEmbeddedThinkBlocks(allocator, final_text);
    defer if (normalized_text) |n| allocator.free(n);
    if (normalized_text) |n| final_text = n;

    // Template-opened think block (Qwen 3.5/3.6): unclosed output is reasoning.
    const opens_think = enable_thinking and promptOpensThink(allocator, lm, tok, prompt_ids);

    // Apply reasoning budget: truncate reasoning by token count
    // For non-streaming, we truncate after generation since we can't interrupt mid-generation
    var budget_truncated_reasoning: ?[]const u8 = null;
    var budget_reasoning_allocated = false;
    defer if (budget_reasoning_allocated) allocator.free(budget_truncated_reasoning.?);

    if (enable_thinking and reasoning_budget >= 0) {
        const think_split = chat_mod.splitThinkBlock(final_text, true, opens_think);
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
        const found_calls = try parseToolCallsForRequest(allocator, final_text, tools_json);
        if (found_calls) |tool_calls| {
            defer {
                for (tool_calls) |tc| {
                    allocator.free(tc.name);
                    allocator.free(tc.arguments);
                }
                allocator.free(tool_calls);
            }

            var tc_perf_buf: [160]u8 = undefined;
            const tc_perf = formatPerfBracket(&tc_perf_buf, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns);
            log.info("  <- {d}+{d} tokens ({d}ms) [{s}] [tool_calls: {d}]\n", .{
                result.prompt_tokens, result.completion_tokens, elapsed_ms, tc_perf, tool_calls.len,
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
                const tc_think_split = chat_mod.splitThinkBlock(final_text, true, opens_think);
                if (tc_think_split.reasoning_content) |reasoning| {
                    const escaped_reasoning = try jsonEscape(allocator, reasoning);
                    tc_reasoning_json = try std.fmt.allocPrint(allocator, ",\"reasoning_content\":{s}", .{escaped_reasoning});
                    allocator.free(escaped_reasoning);
                    tc_reasoning_allocated = true;
                }
            }
            defer if (tc_reasoning_allocated) allocator.free(tc_reasoning_json);

            const tc_timings = try formatTimingsObject(allocator, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns, tokenize_ns);
            defer allocator.free(tc_timings);
            const tc_timings_field = if (tc_timings.len > 0)
                try std.fmt.allocPrint(allocator, ",\"timings\":{s}", .{tc_timings})
            else
                try allocator.alloc(u8, 0);
            defer allocator.free(tc_timings_field);

            const response = try std.fmt.allocPrint(allocator,
                \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[{{"index":0,"message":{{"role":"assistant","content":null{s},"tool_calls":{s}}},"finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}{s}}}
            , .{
                nowMs(stream.io),
                nowSecs(stream.io),
                model_name,
                tc_reasoning_json,
                tc_buf.items,
                toolCallFinishReason(finish_reason),
                result.prompt_tokens,
                result.completion_tokens,
                result.prompt_tokens + result.completion_tokens,
                tc_timings_field,
            });
            defer allocator.free(response);
            try sendResponse(stream, "200 OK", "application/json", response);
            return;
        }
    }

    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns);
    log.info("  <- {d}+{d} tokens ({d}ms) [{s}] [{s}]\n", .{
        result.prompt_tokens, result.completion_tokens, elapsed_ms, perf, finish_reason,
    });

    // Split thinking content from response
    const think_split = chat_mod.splitThinkBlock(final_text, enable_thinking, opens_think);
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
    var usage_details_json: []const u8 = "";
    var usage_details_allocated = false;
    if (enable_thinking) {
        // Use budget-truncated reasoning if available, otherwise use full reasoning
        const reasoning_text = if (budget_truncated_reasoning) |tr| tr else think_split.reasoning_content;
        if (reasoning_text) |reasoning| {
            const escaped_reasoning = try jsonEscape(allocator, reasoning);
            reasoning_json = try std.fmt.allocPrint(allocator, ",\"reasoning_content\":{s}", .{escaped_reasoning});
            allocator.free(escaped_reasoning);
            reasoning_allocated = true;
            // usage.completion_tokens_details.reasoning_tokens (OpenAI/LM Studio
            // parity) so clients can budget visible content separately.
            if (tok.encode(allocator, reasoning)) |rids| {
                defer allocator.free(rids);
                usage_details_json = try std.fmt.allocPrint(allocator, ",\"completion_tokens_details\":{{\"reasoning_tokens\":{d}}}", .{rids.len});
                usage_details_allocated = true;
            } else |_| {}
        }
    }
    defer if (reasoning_allocated) allocator.free(reasoning_json);
    defer if (usage_details_allocated) allocator.free(usage_details_json);

    const timings_obj = try formatTimingsObject(allocator, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns, tokenize_ns);
    defer allocator.free(timings_obj);
    const timings_field = if (timings_obj.len > 0)
        try std.fmt.allocPrint(allocator, ",\"timings\":{s}", .{timings_obj})
    else
        try allocator.alloc(u8, 0);
    defer allocator.free(timings_field);

    const response = try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[{{"index":0,"message":{{"role":"assistant","content":{s}{s}}},"logprobs":{s},"finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}{s}}}{s}}}
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
        usage_details_json,
        timings_field,
    });
    defer allocator.free(response);

    try sendResponse(stream, "200 OK", "application/json", response);
}

/// Token-stream adapter that drives the streaming SSE state machine the same
/// way regardless of whether speculative decoding (PLD / drafter) is on. Each
/// `next()` call yields exactly one token (or null on EOS / max_tokens), so
/// the per-token state machine in `handleStreamingGeneration` /
/// `handleAnthropicStreaming` / `handleResponses` does not need to know
/// anything about multi-token batches that PLD and drafter emit per step.
///
/// EOS-in-batch behavior matches the non-streaming `generatePld` /
/// `generateDrafter`: the stop token is NOT yielded — the loop just
/// terminates. This keeps tokens leaking past EOS impossible regardless of
/// where in an N-token batch the EOS lands.
///
/// `pld_drop_buf` is a heap-owned pending buffer (PLD can yield up to
/// `1+max_draft_len`=16 tokens per step; drafter yields up to `block_size`).
/// Caller must `deinit`.
const StreamMode = enum { regular, pld, drafter, mtp };
/// Source of streamed tokens. Either a legacy Generator (single-slot mlx on
/// the calling thread) or a Scheduler Slot (mlx on the scheduler's inference
/// thread, this thread reads via waitNext). `next()` yields one token id at
/// a time regardless of source. Post-generation stats are surfaced via the
/// `prompt_tokens` / `completion_tokens` / `finish_reason` fields populated
/// by `finalize()`.
const StreamingTokenStream = struct {
    /// Active source. Exactly one of `gen`/`slot` is set per stream.
    gen: ?*Generator = null,
    slot: ?*scheduler_mod.Slot = null,
    mode: StreamMode,
    pld_draft_len: u32 = 0,
    pld_key_len: u32 = 0,
    eos_token_ids: []const u32,
    /// Pending tokens from a multi-token speculative step (legacy path only).
    /// Drained one at a time before the next call to nextPld / nextDrafter.
    pending_buf: std.ArrayList(u32) = .empty,
    pending_idx: usize = 0,
    finished: bool = false,

    /// Stats populated by `finalize()` after the stream ends. Consumers read
    /// these instead of touching the underlying gen/slot directly so the same
    /// post-generation code works for both legacy and scheduler paths.
    prompt_tokens: u32 = 0,
    cached_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    finish_reason: []const u8 = "stop",
    /// Wall-clock ns for prefill / decode (scheduler path only; the legacy
    /// Generator path leaves these at 0, in which case the server omits the
    /// `timings` block from the usage chunk).
    prefill_ns: u64 = 0,
    decode_ns: u64 = 0,

    fn init(gen: *Generator, mode: StreamMode, pld_draft_len: u32, pld_key_len: u32, eos: []const u32) StreamingTokenStream {
        return .{
            .gen = gen,
            .mode = mode,
            .pld_draft_len = pld_draft_len,
            .pld_key_len = pld_key_len,
            .eos_token_ids = eos,
        };
    }

    /// Phase A2-streaming: build a token stream backed by a scheduler Slot.
    /// `mode` is recorded for telemetry but doesn't drive next()'s dispatch
    /// (the scheduler's inference thread already routed PLD/drafter inside
    /// runSingleDecodeTick — this side just drains the resulting token ring).
    fn initFromSlot(slot: *scheduler_mod.Slot, mode: StreamMode, eos: []const u32) StreamingTokenStream {
        return .{
            .slot = slot,
            .mode = mode,
            .eos_token_ids = eos,
        };
    }

    fn deinit(self: *StreamingTokenStream, allocator: std.mem.Allocator) void {
        self.pending_buf.deinit(allocator);
    }

    /// Snapshot prompt/completion tokens + finish reason from the underlying
    /// source so callers can read them after the loop exits without having
    /// to know which source was active. Safe to call multiple times; idempotent.
    fn finalize(self: *StreamingTokenStream) void {
        if (self.slot) |s| {
            self.prompt_tokens = s.prompt_tokens;
            self.cached_tokens = s.cached_tokens;
            self.completion_tokens = s.completion_tokens;
            self.finish_reason = s.finish_reason;
            self.prefill_ns = s.prefill_ns;
            self.decode_ns = s.decode_ns;
        } else if (self.gen) |g| {
            self.prompt_tokens = g.prompt_tokens;
            self.completion_tokens = g.completion_tokens;
            self.finish_reason = g.finish_reason;
        }
    }

    const NextOrIdle = union(enum) { token: u32, done, idle };

    /// `next` with an idle timeout (scheduler path only): returns `.idle`
    /// after `timeout_ms` with no progress so the caller can poll the peer
    /// socket and emit SSE keepalives during long prefills. The local-
    /// generator path computes synchronously and never idles.
    fn nextOrIdle(self: *StreamingTokenStream, allocator: std.mem.Allocator, timeout_ms: i64) !NextOrIdle {
        if (self.slot) |s| {
            if (self.finished) return .done;
            const nr = s.waitNextTimeout(timeout_ms) orelse return .idle;
            switch (nr) {
                .token => |t| return .{ .token = t },
                .done => {
                    self.finished = true;
                    return .done;
                },
                .err => return error.GenerationFailed,
            }
        }
        if (try self.next(allocator)) |t| return .{ .token = t };
        return .done;
    }

    /// Yield the next decoded token id, or null if generation is complete.
    /// Mirrors the contract of `Generator.next` for the regular path.
    fn next(self: *StreamingTokenStream, allocator: std.mem.Allocator) !?u32 {
        // Scheduler path: drain the slot's output ring one token at a time.
        // The inference thread already handled regular/PLD/drafter dispatch
        // via runSingleDecodeTick; we just consume what it pushed.
        if (self.slot) |s| {
            if (self.finished) return null;
            switch (s.waitNext()) {
                .token => |t| return t,
                .done => {
                    self.finished = true;
                    return null;
                },
                .err => return error.GenerationFailed,
            }
        }
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

        const gen = self.gen.?;
        switch (self.mode) {
            .regular => return gen.next(allocator),
            .pld => {
                const r = (try gen.nextPld(allocator, self.pld_draft_len, self.pld_key_len)) orelse return null;
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
                    gen.done = true;
                    gen.finish_reason = "stop";
                    self.finished = true;
                    return null;
                }
                const first_tok = r.tokens[first_idx];
                // Append the remaining (non-EOS-prefixed) tokens to pending,
                // stopping at the first EOS in the tail.
                var i: usize = first_idx + 1;
                while (i < r.tokens.len) : (i += 1) {
                    if (generate_mod.isEosId(r.tokens[i], self.eos_token_ids)) {
                        gen.done = true;
                        gen.finish_reason = "stop";
                        self.finished = true;
                        break;
                    }
                    try self.pending_buf.append(allocator, r.tokens[i]);
                }
                return first_tok;
            },
            .mtp => {
                // Mirror the drafter branch — `nextMtp` returns the same
                // `{tokens, accepted_tokens}` shape.
                const r = (try gen.nextMtp(allocator)) orelse return null;
                defer allocator.free(r.tokens);
                if (r.tokens.len == 0) {
                    self.finished = true;
                    return null;
                }
                var first_idx: usize = 0;
                while (first_idx < r.tokens.len and generate_mod.isEosId(r.tokens[first_idx], self.eos_token_ids)) : (first_idx += 1) {}
                if (first_idx >= r.tokens.len) {
                    gen.done = true;
                    gen.finish_reason = "stop";
                    self.finished = true;
                    return null;
                }
                const first_tok = r.tokens[first_idx];
                var i: usize = first_idx + 1;
                while (i < r.tokens.len) : (i += 1) {
                    if (generate_mod.isEosId(r.tokens[i], self.eos_token_ids)) {
                        gen.done = true;
                        gen.finish_reason = "stop";
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
                const r = (try gen.nextDrafter(allocator)) orelse return null;
                defer allocator.free(r.tokens);
                if (r.tokens.len == 0) {
                    self.finished = true;
                    return null;
                }
                var first_idx: usize = 0;
                while (first_idx < r.tokens.len and generate_mod.isEosId(r.tokens[first_idx], self.eos_token_ids)) : (first_idx += 1) {}
                if (first_idx >= r.tokens.len) {
                    gen.done = true;
                    gen.finish_reason = "stop";
                    self.finished = true;
                    return null;
                }
                const first_tok = r.tokens[first_idx];
                var i: usize = first_idx + 1;
                while (i < r.tokens.len) : (i += 1) {
                    if (generate_mod.isEosId(r.tokens[i], self.eos_token_ids)) {
                        gen.done = true;
                        gen.finish_reason = "stop";
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
/// Priority: drafter > PLD > regular (matches the non-streaming dispatch in
/// `handleNonStreamingGeneration`). The same per-mode disable rules (no
/// logprobs, no grammar constraint, no hybrid SSM for drafter) apply at the
/// request-entry parse site; `pickStreamMode` re-enforces them defensively
/// here so a missed gate at the parse site doesn't crash the dispatch.
fn pickStreamMode(
    enable_pld: bool,
    enable_drafter: bool,
    enable_mtp: bool,
    drafter_loaded: bool,
    mtp_loaded: bool,
    has_hybrid_layers: bool,
    has_constraint: bool,
    logprobs_n: u32,
) StreamMode {
    // Priority: MTP > drafter > PLD. The MTP head only loads when it binds
    // to the trunk, so no extra arch gates here; the GDN/SSM rollback path
    // it needs is the same one PLD uses.
    if (enable_mtp and mtp_loaded and logprobs_n == 0 and !has_constraint) return .mtp;
    if (enable_drafter and drafter_loaded and logprobs_n == 0 and !has_constraint and !has_hybrid_layers) return .drafter;
    if (enable_pld and logprobs_n == 0 and !has_constraint) return .pld;
    return .regular;
}

fn handleStreamingGeneration(
    allocator: std.mem.Allocator,
    stream: *Conn,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: generate_mod.SamplingParams,
    eos_token_ids: []const u32,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    include_usage: bool,
    has_tools: bool,
    /// OpenAI-shape tools JSON (for bare-args tool-call inference); null when
    /// the request defined no tools.
    tools_json: ?[]const u8,
    logprobs_n: u32,
    enable_thinking: bool,
    reasoning_budget: i32,
    enable_pld: bool,
    enable_drafter: bool,
    enable_mtp: bool,
    vision_embeddings: ?mlx.mlx_array,
    mrope: MropeData,
    /// Wave 1.A: per-request KV-quant override; null = inherit scheduler default.
    kv_quant_override: ?transformer_mod.KVQuantConfig,
    /// Iteration 1: tokenize_ns measured by the request handler before
    /// dispatching here. Surfaced via `timings.tokenize_ms` on the final
    /// usage SSE chunk so streaming clients see the same metric as
    /// non-streaming.
    tokenize_ns: u64,
) !void {
    // Vision array ownership: held by this handler on entry, transfers to
    // the slot on submit (slot.deinit frees). Nulled before transfer so
    // the early-return defer is a no-op.
    var ve_local = vision_embeddings;
    defer { if (ve_local) |arr| _ = mlx.mlx_array_free(arr); }

    const config = lm.config.?;
    const chat_id = nowMs(stream.io);

    // Template-opened think block (Qwen 3.5/3.6): unclosed buffered output is
    // reasoning, never content (mirrors the non-streaming split policy).
    const opens_think = enable_thinking and promptOpensThink(allocator, lm, tok, prompt_ids);

    // Pick the speculative-decoding mode (regular / PLD / drafter). The
    // per-token state machine below is driven by `StreamingTokenStream`,
    // which feeds `next` (regular), `nextPld` (1..1+draft_len tokens/step),
    // or `nextDrafter` (1..block_size tokens/step) through the same
    // one-token-at-a-time interface.
    const stream_mode = pickStreamMode(enable_pld, enable_drafter, enable_mtp, lm.drafter != null, lm.mtp != null, config.has_hybrid_layers, sampling.constraint != null, logprobs_n);
    if (stream_mode == .pld) log.info("  pld=enabled (streaming, draft_len={d}, key_len={d})\n", .{ server_config.default_pld_draft_len, server_config.default_pld_key_len });
    if (stream_mode == .drafter) log.info("  drafter=enabled (streaming, block_size={d})\n", .{lm.drafter_block_size});
    if (stream_mode == .mtp) log.info("  mtp=enabled (streaming, depth={d})\n", .{lm.mtp_depth});

    // Scheduler's inference thread runs prefill + per-tick decode (regular
    // / PLD / drafter) and pushes generated tokens into the slot's output
    // ring; this thread reads via `slot.waitNext` through the
    // `StreamingTokenStream.initFromSlot` adapter.
    var slot_handle: ?*scheduler_mod.Slot = null;
    defer if (slot_handle) |s| global_scheduler.?.complete(s);

    // Transfer vision ownership into the slot.
    const slot_ve_s = ve_local;
    ve_local = null;
    const sch = global_scheduler.?;
    slot_handle = try sch.submit(.{
        .model = lm,
        .prompt_ids = prompt_ids,
        .full_prompt = prompt_ids,
        .cached_tokens = 0,
        .has_tools = has_tools,
        .sampling = sampling,
        .eos_token_ids = eos_token_ids,
        .max_tokens = max_tokens,
        .timeout_ns = getTimeoutNs(),
        .enable_pld = stream_mode == .pld,
        .enable_drafter = stream_mode == .drafter,
        .drafter = if (stream_mode == .drafter) lm.drafter else null,
        .drafter_block_size = lm.drafter_block_size,
        .enable_mtp = stream_mode == .mtp,
        .mtp = if (stream_mode == .mtp) lm.mtp else null,
        .mtp_depth = lm.mtp_depth,
        .pld_draft_len = server_config.default_pld_draft_len,
        .pld_key_len = server_config.default_pld_key_len,
        .kv_attn_fused = server_config.default_kv_attn_fused,
        .logprobs_n = logprobs_n,
        .vision_embeddings = slot_ve_s,
        .mrope_pos = mrope.pos,
        .mrope_total = mrope.total,
        .mrope_delta = mrope.delta,
        .kv_quant_config = kv_quant_override,
    });
    var ts = StreamingTokenStream.initFromSlot(slot_handle.?, stream_mode, eos_token_ids);
    defer ts.deinit(allocator);

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
    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = "assistant", .content = "" }, null, null, null);

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
    var client_gone = false;

    // Buffer for incomplete UTF-8 sequences split across BPE tokens
    var utf8_carry: [3]u8 = undefined;
    var utf8_carry_len: u8 = 0;

    // Thinking state for real-time streaming of reasoning_content vs content
    // Supports both <think>...</think> and Gemma 4's <|channel>thought\n...<channel|>
    var in_think_block = enable_thinking; // starts true when thinking enabled (model outputs <think> first)
    var think_closed = false; // a complete think block was already split+emitted this stream
    var think_buf = std.ArrayList(u8).empty; // buffer to detect close tag across token boundaries
    defer think_buf.deinit(allocator);
    var think_close_tag: []const u8 = "</think>"; // will be updated if Gemma 4 format detected
    var skipped_think_open = false; // track if we've skipped the initial think tag
    var think_tokens: i32 = 0; // count of tokens generated in think block
    var budget_exhausted = false; // true when reasoning budget hit

    // Generate tokens via the adapter — yields one decoded token id per call
    // regardless of whether the underlying decode is regular, PLD, or drafter.
    while (true) {
        const token_id: u32 = switch (try ts.nextOrIdle(allocator, Conn.STREAM_KEEPALIVE_MS)) {
            .token => |t| t,
            .done => break,
            .idle => {
                // No tokens yet (long prefill). Probe the peer: an abandoned
                // request must cancel instead of grinding a ghost prefill
                // (Claude Code retries pile up serially otherwise), and the
                // keepalive stops clients timing out on stream silence.
                if (stream.peerClosed()) {
                    log.info("  [cancel] client disconnected while waiting for tokens — cancelling slot\n", .{});
                    slot_handle.?.cancel();
                    client_gone = true;
                    break;
                }
                sendStreamKeepalive(stream) catch {
                    log.info("  [cancel] keepalive write failed (client disconnected) — cancelling slot\n", .{});
                    slot_handle.?.cancel();
                    client_gone = true;
                    break;
                };
                continue;
            },
        };
        if (stream.peerClosed()) {
            slot_handle.?.cancel();
            client_gone = true;
            break;
        }
        const strip = tok.tok_type == .sentencepiece_bpe;
        const raw_decoded = try decodeTokens(allocator, lm, tok, &[_]u32{token_id}, strip and false);

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
            // Detection rules live in `chat.streamShouldBufferForTools` — it
            // covers the full `<tool…>` family (including bare DSV4 `<tool>`),
            // Gemma 4 `<|tool_call`, raw JSON, plus partial-prefix growth from
            // a single `<` all the way to `<|tool_cal`. Without the partial
            // coverage, a `<tool` single-BPE-token leaks as visible content
            // (the bug surfaced on DSV4 where the model emits `<tool` then
            // gets stuck looping the partial).
            try token_texts.append(allocator, token_text);
            // text_buf already updated above

            const buf = text_buf.items;
            const maybe_tool = chat_mod.streamShouldBufferForTools(buf);

            if (!maybe_tool) {
                // No tool call pattern — ask the shared gate (chat.streamThinkGate,
                // also used by /v1/messages) whether the buffer is thinking that
                // must be held, a completed think block to split, or visible
                // prose to flush. Hermetically pinned per recorded model family
                // by the format corpus streaming-gate test.
                switch (chat_mod.streamThinkGate(buf, enable_thinking, think_closed)) {
                    .hold_thinking => {
                        // Incomplete thinking block — keep buffering until closed
                    },
                    .split_think => {
                        // Complete thinking block — split into reasoning + content
                        const split = chat_mod.splitThinkBlock(buf, enable_thinking, opens_think);
                        for (token_texts.items) |tt| allocator.free(tt);
                        token_texts.clearRetainingCapacity();
                        text_buf.clearRetainingCapacity();
                        think_closed = true;
                        if (enable_thinking) {
                            if (split.reasoning_content) |rc| {
                                try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = rc }, null, null, null);
                            }
                        }
                        if (split.content.len > 0) {
                            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = split.content }, null, null, null);
                        }
                    },
                    .flush_text => {
                        for (token_texts.items) |tt| {
                            defer allocator.free(tt);
                            // Skip bare channel/think tags that leak without a full block
                            if (std.mem.eql(u8, tt, "<|channel>") or std.mem.eql(u8, tt, "<channel|>") or
                                std.mem.eql(u8, tt, "<think>") or std.mem.eql(u8, tt, "</think>"))
                            {
                                continue;
                            }
                            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = tt }, null, null, null);
                        }
                        token_texts.clearRetainingCapacity();
                    },
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
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = think_buf.items }, null, null, null);
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
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = think_buf.items[0..m.pos] }, null, null, null);
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
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = content_after }, null, null, null);
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
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = think_buf.items[0..safe_len] }, null, null, null);
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
            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = token_text }, null, null, null);
        }

        // Every branch above may have written NOTHING (tool-call detection and
        // unclosed thinking blocks buffer silently, sometimes for minutes on a
        // one-shot whole-file `write_file`). Tokens flowing is not bytes
        // flowing — without this the client's idle-body timeout kills the
        // stream mid-generation.
        beatStreamKeepalive(stream, .sse_comment) catch {
            log.info("  [cancel] keepalive write failed (client disconnected) — cancelling slot\n", .{});
            slot_handle.?.cancel();
            client_gone = true;
            break;
        };
    }

    // Flush any remaining think buffer
    if (!client_gone and enable_thinking and think_buf.items.len > 0) {
        if (in_think_block) {
            // Never found </think> — flush as reasoning
            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = think_buf.items }, null, null, null);
        } else {
            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = think_buf.items }, null, null, null);
        }
    }

    // After generation: capture stats from whichever source was active
    // (Generator on legacy, Slot on scheduler).
    ts.finalize();

    // Check for tool calls in accumulated text
    var finish_reason: []const u8 = if (client_gone) "client_disconnect" else if (stopped) "stop" else ts.finish_reason;
    if (has_tools and !client_gone) {
        log.debug("  checking {d} bytes of streamed text for tool calls\n", .{text_buf.items.len});
        if (log.isDebug() and text_buf.items.len > 0) {
            log.debug("  raw generated text before tool parse ({d}b): {s}\n", .{ text_buf.items.len, text_buf.items[0..@min(text_buf.items.len, 4000)] });
            // Corpus-harvest aid: the inline dump caps at 4KB, useless for
            // >30KB mega-tool-calls. Set MLX_SERVE_RAW_DUMP_FILE=<abs path> to
            // APPEND the FULL pre-parse buffer of every tools request, framed so
            // a harvester can slice each record exactly (the text is arbitrary
            // bytes, so the byte count — not a delimiter — defines the record).
            if (std.c.getenv("MLX_SERVE_RAW_DUMP_FILE")) |dump_path| {
                appendRawToolDump(std.mem.span(dump_path), tools_json, text_buf.items);
            }
        }
        // Merge re-opened mid-text thought channels into the leading block so
        // the split/parse below never leaks raw tags (Gemma 12B tail behavior).
        const norm_owned = try chat_mod.normalizeEmbeddedThinkBlocks(allocator, text_buf.items);
        defer if (norm_owned) |n| allocator.free(n);
        const gen_text: []const u8 = norm_owned orelse text_buf.items;
        const found_calls = try parseToolCallsForRequest(allocator, gen_text, tools_json);
        if (found_calls) |tool_calls| {
            defer {
                for (tool_calls) |tc| {
                    allocator.free(tc.name);
                    allocator.free(tc.arguments);
                }
                allocator.free(tool_calls);
            }

            // Emit reasoning_content before tool calls if thinking is enabled
            if (enable_thinking) {
                const think_split = chat_mod.splitThinkBlock(gen_text, true, opens_think and !think_closed);
                if (think_split.reasoning_content) |reasoning| {
                    // Apply reasoning budget truncation if set
                    const final_reasoning = if (reasoning_budget >= 0) blk: {
                        const r_ids = try tok.encode(allocator, reasoning);
                        defer allocator.free(r_ids);
                        const budget_usize: usize = @intCast(reasoning_budget);
                        if (r_ids.len > budget_usize) {
                            const truncated = try tok.decode(allocator, r_ids[0..budget_usize], false);
                            defer allocator.free(truncated);
                            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = truncated }, null, null, null);
                            break :blk @as(?[]const u8, null);
                        }
                        break :blk @as(?[]const u8, reasoning);
                    } else @as(?[]const u8, reasoning);
                    if (final_reasoning) |r| {
                        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = r }, null, null, null);
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
                try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .tool_calls_json = first_delta }, null, null, null);
            }
            finish_reason = toolCallFinishReason(finish_reason);
        } else {
            // No tool calls found — flush buffered tokens as content
            if (enable_thinking) {
                // Concatenate all buffered tokens and split thinking from content
                var full_text = std.ArrayList(u8).empty;
                defer full_text.deinit(allocator);
                for (token_texts.items) |t| {
                    try full_text.appendSlice(allocator, t);
                }
                const flush_norm = try chat_mod.normalizeEmbeddedThinkBlocks(allocator, full_text.items);
                defer if (flush_norm) |n| allocator.free(n);
                const flush_text: []const u8 = flush_norm orelse full_text.items;
                const think_split = chat_mod.splitThinkBlock(flush_text, true, opens_think and !think_closed);
                if (think_split.reasoning_content) |reasoning| {
                    // Apply reasoning budget truncation if set
                    const final_reasoning = if (reasoning_budget >= 0) blk: {
                        const r_ids = try tok.encode(allocator, reasoning);
                        defer allocator.free(r_ids);
                        const budget_usize: usize = @intCast(reasoning_budget);
                        if (r_ids.len > budget_usize) {
                            const truncated = try tok.decode(allocator, r_ids[0..budget_usize], false);
                            defer allocator.free(truncated);
                            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = truncated }, null, null, null);
                            break :blk @as(?[]const u8, null);
                        }
                        break :blk @as(?[]const u8, reasoning);
                    } else @as(?[]const u8, reasoning);
                    if (final_reasoning) |r| {
                        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = r }, null, null, null);
                    }
                }
                if (think_split.content.len > 0) {
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = think_split.content }, null, null, null);
                }
            } else {
                for (token_texts.items) |t| {
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = t }, null, null, null);
                }
            }
        }
    }

    const total_prompt = ts.prompt_tokens;
    if (!client_gone) {
        // Final chunk with finish_reason
        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null }, finish_reason, null, null);

        // Usage chunk (if requested via stream_options.include_usage). Scheduler
        // accounts for any prompt-cache hits in `ts.prompt_tokens` directly.
        if (include_usage) {
            const usage_json = try std.fmt.allocPrint(allocator,
                \\{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}
            , .{ total_prompt, ts.completion_tokens, total_prompt + ts.completion_tokens });
            defer allocator.free(usage_json);
            const timings_obj = try formatTimingsObject(allocator, total_prompt, ts.cached_tokens, ts.completion_tokens, ts.prefill_ns, ts.decode_ns, tokenize_ns);
            defer allocator.free(timings_obj);
            const timings_opt: ?[]const u8 = if (timings_obj.len > 0) timings_obj else null;
            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null }, finish_reason, usage_json, timings_opt);
        }

        // Done sentinel
        logHttpSseData("[DONE]");
        try stream.writeAll("data: [DONE]\n\n");
    }

    // Per-slot timings come from the scheduler (ts.prefill_ns / ts.decode_ns,
    // populated in finalize); the bracket reports compute + any prefix reuse.
    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, ts.prompt_tokens, ts.cached_tokens, ts.completion_tokens, ts.prefill_ns, ts.decode_ns);
    log.info("  <- {d}+{d} tokens streamed [{s}] [{s}]\n", .{
        total_prompt, ts.completion_tokens, perf, finish_reason,
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
    timings_json: ?[]const u8,
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

    // Optional `timings` tail. Inserted as a top-level field next to `usage`
    // — matches the llama.cpp shape so clients that key off it work as-is.
    var timings_tail_buf = std.ArrayList(u8).empty;
    defer timings_tail_buf.deinit(allocator);
    if (timings_json) |t| {
        try timings_tail_buf.appendSlice(allocator, ",\"timings\":");
        try timings_tail_buf.appendSlice(allocator, t);
    }

    // Build the full SSE chunk
    const chunk = try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[{{"index":0,"delta":{s},"finish_reason":{s}}}],"usage":{s}{s}}}
    , .{ chat_id, nowSecs(stream.io), model_name, delta_buf.items, fr_str, usage_str, timings_tail_buf.items });
    defer allocator.free(chunk);

    // Write as SSE event
    logHttpSseData(chunk);
    try stream.writeAllNoFlush("data: ");
    try stream.writeAllNoFlush(chunk);
    try stream.writeAllNoFlush("\n\n");
    try stream.flush();
}

// ── Shared utilities ──

fn sendResponse(stream: *Conn, status: []const u8, content_type: []const u8, body: []const u8) !void {
    logHttpResponse(status, content_type, body);
    try sendResponseFramed(stream, status, content_type, body);
}

/// Like `sendResponse` but does NOT dump the response body to the debug log —
/// only a one-line status summary. For high-frequency, self-describing bodies
/// where the full dump is pure noise: the Prometheus scrape (`/metrics`, hit
/// every couple seconds) and the panel's JSON feed (`/metrics.json`, ~1 Hz).
fn sendResponseQuiet(stream: *Conn, status: []const u8, content_type: []const u8, body: []const u8) !void {
    if (log.isDebug()) log.debug("[http] <- {s} {s} body={d}b (scrape; body not logged)\n", .{ status, content_type, body.len });
    try sendResponseFramed(stream, status, content_type, body);
}

fn sendResponseFramed(stream: *Conn, status: []const u8, content_type: []const u8, body: []const u8) !void {
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

const GaugeSamplerCtx = struct {
    metrics: *instr.Metrics,
    scheduler: *scheduler_mod.Scheduler,
    /// Dedicated stop flag — set before joining the thread (LIFO defers ensure
    /// this is written before join() blocks). A separate flag (not
    /// scheduler.shutdown) avoids a LIFO deadlock: scheduler.deinit() (which
    /// sets shutdown) is the FIRST-registered defer and runs LAST, after join.
    stop: *std.atomic.Value(bool),
};

/// Sample instantaneous system/queue state into the gauges. Non-blocking
/// syscalls plus a brief scheduler-counter lock. Called by the sampler thread.
/// The live-token gauge reads the scheduler's `inflight_generated_tokens`
/// aggregate (published once per decode tick) — it NEVER touches per-slot
/// fields off-thread, so the inference path stays lock-free per token.
fn sampleGauges(ctx: GaugeSamplerCtx) void {
    // System gauges (non-blocking syscalls).
    ctx.metrics.gpu_utilization_pct.set(@as(u64, metrics.getGpuPct()));
    ctx.metrics.memory_mb.set(@as(u64, metrics.getAppMemFootprintMb()));

    // Request queue depth — brief lock to read two scheduler counters only.
    ctx.scheduler.queue_mu.lockUncancelable(ctx.scheduler.io);
    const running = @as(u64, ctx.scheduler.in_flight);
    const waiting = @as(u64, ctx.scheduler.pending.items.len);
    ctx.scheduler.queue_mu.unlock(ctx.scheduler.io);
    ctx.metrics.requests_running.set(running);
    ctx.metrics.requests_waiting.set(waiting);

    // Real-time throughput source: completed generation tokens plus the tokens
    // generated so far by still-decoding slots (the scheduler publishes that
    // aggregate atomically once per decode tick — race-free, no per-token
    // write). Non-zero even mid-request, unlike the completion-only counter.
    const inflight = ctx.scheduler.inflight_generated_tokens.load(.monotonic);
    ctx.metrics.generation_tokens_live.set(instr.liveGenerationTokens(ctx.metrics, inflight));

    // The other phase. Prompt-token counters and the prefill-time histogram
    // only advance at request COMPLETION, so a long prefill was invisible: GPU
    // pinned, decode reading 0, prefill reading "—". This gauge moves per
    // prefill chunk and returns to 0 the moment the prefill ends.
    ctx.metrics.prefill_tokens_live.set(ctx.scheduler.inflight_prefill_tokens.load(.monotonic));
    ctx.metrics.requests_prefilling.set(ctx.scheduler.requests_prefilling.load(.monotonic));
}

fn gaugeSamplerLoop(ctx: GaugeSamplerCtx) void {
    // Wake every 500 ms to check the stop flag; sample system state once per
    // 2-second interval. The 2 s cadence is what makes generation_tokens_live a
    // real-time tok/s source — the panel windows over a few of these samples.
    // Uses std.c.nanosleep (POSIX) because std.time.sleep was removed in Zig
    // 0.16 — this thread has no Io handle.
    const SAMPLE_INTERVAL_TICKS: u64 = 4; // 4 × 500 ms = 2 s
    const poll_ts = std.c.timespec{ .sec = 0, .nsec = 500_000_000 };
    // Sample once immediately so the panel / Grafana show real values from the
    // first scrape instead of zeros for the first interval.
    sampleGauges(ctx);
    var tick: u64 = 0;
    while (!ctx.stop.load(.monotonic)) {
        _ = std.c.nanosleep(&poll_ts, null);
        tick += 1;
        if (tick < SAMPLE_INTERVAL_TICKS) continue;
        tick = 0;
        sampleGauges(ctx);
    }
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

// ── API-key auth helpers (used only when --api-key / g_api_key is set) ──

/// True if the connection's peer is a loopback address (127.0.0.0/8, ::1, or an
/// IPv4-mapped-IPv6 loopback). Loopback is the local machine → trusted and
/// exempt from the API key, so the same-host app never needs credentials.
fn ipIsLoopback(addr: std.Io.net.IpAddress) bool {
    return switch (addr) {
        .ip4 => |a4| a4.bytes[0] == 127, // 127.0.0.0/8
        .ip6 => |a6| blk: {
            const b = a6.bytes; // big-endian 16 bytes
            const zeros10 = [_]u8{0} ** 10;
            // ::1
            if (std.mem.eql(u8, b[0..15], &([_]u8{0} ** 15)) and b[15] == 1) break :blk true;
            // ::ffff:127.x.x.x  (IPv4-mapped loopback)
            if (std.mem.eql(u8, b[0..10], &zeros10) and b[10] == 0xff and b[11] == 0xff and b[12] == 127) break :blk true;
            break :blk false;
        },
    };
}

fn peerIsLoopback(conn: *const Conn) bool {
    return ipIsLoopback(conn.stream.socket.address);
}

/// Case-insensitive HTTP header lookup in the raw header block. `name_lower`
/// must be lowercase (e.g. "authorization"). Returns the trimmed value, or null.
fn findHeaderValueCI(headers: []const u8, name_lower: []const u8) ?[]const u8 {
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
fn queryParamValue(raw_path: []const u8, name: []const u8) ?[]const u8 {
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
fn apiKeyAuthorized(raw_headers: []const u8, raw_path: []const u8) bool {
    const key = g_api_key orelse return true;
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
    if (queryParamValue(raw_path, "api_key")) |q| {
        if (constTimeEql(q, key)) return true;
    }
    if (queryParamValue(raw_path, "key")) |q| {
        if (constTimeEql(q, key)) return true;
    }
    return false;
}

/// Send a 401 with a Basic-auth challenge so browsers prompt for the key on the
/// index + metrics pages; API clients read the JSON error body.
fn sendUnauthorized(stream: *Conn) !void {
    const body = "{\"error\":{\"message\":\"missing or invalid API key\",\"type\":\"authentication_error\"}}";
    logHttpResponse("401 Unauthorized", "application/json", body);
    if (stream.ws_mode) |bridge| {
        if (body.len > 0) try bridge.sendText(body);
        return;
    }
    var hdr_buf: [512]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nWWW-Authenticate: Basic realm=\"mlx-serve\"\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Content-Type, Authorization, x-api-key\r\n\r\n", .{body.len}) catch return error.Overflow;
    try stream.writeAll(hdr);
    try stream.writeAll(body);
}

test "apiKeyAuthorized accepts Bearer, x-api-key, Basic, and query param" {
    const prev = g_api_key;
    defer g_api_key = prev;
    g_api_key = "s3cret";

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

    // No key configured ⇒ always authorized (open mode)
    g_api_key = null;
    try std.testing.expect(apiKeyAuthorized("", "/v1/chat/completions"));
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

/// Build a llama.cpp-style `timings` JSON object (no surrounding key) from
/// raw nanosecond counts and token totals. Caller frees. Returns an empty
/// string when `prefill_ns`, `decode_ns`, AND `tokenize_ns` are all zero
/// (legacy paths that don't measure timing) so the field can be omitted.
///
/// `tokenize_ns` is the wall-clock cost of the synchronous
/// `renderChatTemplate + tokenizer.encode` step on the request thread.
/// Phase 4 #1 of `performance-plan.md` calls for instrumentation-first:
/// before we move tokenize onto a worker thread, we need numbers per
/// engine / prompt length. Pass 0 for legacy paths that don't measure it
/// (the field is then omitted from the JSON; existing callers stay shape-
/// compatible while the new ones surface the metric).
///
/// Iteration 2: chat-template render+tokenize wrapper that consults a
/// per-LoadedModel LRU cache before calling the underlying engine's
/// encoder. On hit the returned slice is a fresh allocation owned by the
/// caller (drop-in replacement for the engine encoders). On miss it
/// stores a copy of the result in the cache.
fn cachedFormatChat(
    allocator: std.mem.Allocator,
    io: std.Io,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    chat_config: *const chat_mod.ChatConfig,
    messages: []const chat_mod.Message,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
) ![]u32 {
    const cache_ptr: ?*tokenize_cache_mod.TokenizeCache = if (lm.tokenize_cache) |*tc| tc else null;
    const key_opt: ?u64 = if (cache_ptr != null)
        tokenize_cache_mod.TokenizeCache.keyFor(messages, tools_json, tool_choice_instruction, enable_thinking)
    else
        null;
    if (cache_ptr) |cache| if (key_opt) |key| {
        if (try cache.get(io, key, allocator)) |cached| return cached;
    };
    const ids = if (lm.ds4_engine) |engine|
        try chat_mod.encodeChatViaDs4(allocator, engine, messages, tools_json, tool_choice_instruction, enable_thinking)
    else if (lm.llama_engine) |engine|
        try chat_mod.encodeChatViaLlama(allocator, engine, chat_config, messages, tools_json, tool_choice_instruction, enable_thinking)
    else
        try chat_mod.formatChat(allocator, tok, messages, chat_config, tools_json, tool_choice_instruction, enable_thinking);
    if (cache_ptr) |cache| if (key_opt) |key| {
        // Insert is best-effort; an OOM in the cache shouldn't fail the
        // request — the user already has their tokenized prompt.
        cache.put(io, key, ids) catch |err| {
            log.warn("[tokenize-cache] put failed: {s}\n", .{@errorName(err)});
        };
    };
    return ids;
}

fn formatTimingsObject(
    allocator: std.mem.Allocator,
    prompt_tokens: u32,
    cached_tokens: u32,
    completion_tokens: u32,
    prefill_ns: u64,
    decode_ns: u64,
    tokenize_ns: u64,
) ![]u8 {
    if (prefill_ns == 0 and decode_ns == 0 and tokenize_ns == 0) return try allocator.alloc(u8, 0);
    const ns_per_ms_f: f64 = @as(f64, @floatFromInt(std.time.ns_per_ms));
    const p_ms = @as(f64, @floatFromInt(prefill_ns)) / ns_per_ms_f;
    const d_ms = @as(f64, @floatFromInt(decode_ns)) / ns_per_ms_f;
    const t_ms = @as(f64, @floatFromInt(tokenize_ns)) / ns_per_ms_f;
    // prompt_per_second reflects compute: divide by the tokens actually run
    // (prompt minus the KV-cache prefix). `cached_n` exposes the reuse so a
    // bench / client can tell a warm hit from a cold prefill.
    const p_tps = generate_mod.prefillTokensPerSec(prompt_tokens, cached_tokens, prefill_ns);
    const d_tps = generate_mod.tokensPerSec(completion_tokens, decode_ns);
    // Always emit `tokenize_ms` (even at 0.0) so clients can rely on the key
    // being present — they can branch on the value, not on presence/absence.
    return try std.fmt.allocPrint(
        allocator,
        \\{{"prompt_n":{d},"cached_n":{d},"prompt_ms":{d:.3},"prompt_per_second":{d:.3},"predicted_n":{d},"predicted_ms":{d:.3},"predicted_per_second":{d:.3},"tokenize_ms":{d:.3}}}
    ,
        .{ prompt_tokens, cached_tokens, p_ms, p_tps, completion_tokens, d_ms, d_tps, t_ms },
    );
}

/// Format the "prefill: X tok/s [(C cached / P total)], decode: Y tok/s" perf
/// bracket into `buf` so every API path logs prefill compute (and any prefix
/// reuse) identically. `cached > 0` adds the cached/total hint; otherwise the
/// short form keeps the existing log shape for cold prefills. Single source of
/// truth so the format never drifts across the chat/anthropic/streaming logs.
fn formatPerfBracket(
    buf: []u8,
    prompt_tokens: u32,
    cached_tokens: u32,
    completion_tokens: u32,
    prefill_ns: u64,
    decode_ns: u64,
) []const u8 {
    const p_tps = generate_mod.prefillTokensPerSec(prompt_tokens, cached_tokens, prefill_ns);
    const d_tps = generate_mod.tokensPerSec(completion_tokens, decode_ns);
    const s = if (cached_tokens > 0)
        std.fmt.bufPrint(buf, "prefill: {d:.1} tok/s ({d} cached / {d} total), decode: {d:.1} tok/s", .{ p_tps, cached_tokens, prompt_tokens, d_tps })
    else
        std.fmt.bufPrint(buf, "prefill: {d:.1} tok/s, decode: {d:.1} tok/s", .{ p_tps, d_tps });
    return s catch buf[0..0];
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

/// Wave 1.A — parse the optional per-request `kv_quant` body field. Accepts
/// the string forms `"off"`, `"0"`, `"4"`, `"8"` and the integer forms `0`,
/// `4`, `8`. Returns null when the field is absent or unrecognized (the
/// caller falls back to the process-level `--kv-quant` default carried on
/// the scheduler). Returns `KVQuantConfig.dense` for "off"/0.
fn parseKvQuantOverride(root: std.json.ObjectMap) ?transformer_mod.KVQuantConfig {
    const v = root.get("kv_quant") orelse return null;
    switch (v) {
        .string => |s| {
            if (std.mem.eql(u8, s, "off") or std.mem.eql(u8, s, "0")) return transformer_mod.KVQuantConfig.dense;
            if (std.mem.eql(u8, s, "4")) return transformer_mod.KVQuantConfig.affine(4);
            if (std.mem.eql(u8, s, "8")) return transformer_mod.KVQuantConfig.affine(8);
            if (std.mem.eql(u8, s, "turbo2")) return transformer_mod.KVQuantConfig.turboquant(2);
            if (std.mem.eql(u8, s, "turbo4")) return transformer_mod.KVQuantConfig.turboquant(4);
            return null;
        },
        .integer => |i| {
            if (i == 0) return transformer_mod.KVQuantConfig.dense;
            if (i == 4) return transformer_mod.KVQuantConfig.affine(4);
            if (i == 8) return transformer_mod.KVQuantConfig.affine(8);
            return null;
        },
        else => return null,
    }
}

// ── Vision Processing ──

/// Collect images from messages, run vision encoder, set embeddings on transformer.
/// Encode vision images from the last user message and return the resulting
/// `[1, total_tokens, hidden]` array. Caller owns the returned array (free
/// via `mlx_array_free` if not transferred to a scheduler slot). Returns
/// `null` when there are no images on the last user turn.
///
/// Phase A8: per-request ownership. Earlier versions wrote the result into
/// `xfm.vision_embeddings` (a global field on Transformer), which raced
/// under `--max-concurrent ≥ 2`: two concurrent vision requests would
/// clobber each other's array. Returning the value lets each conn thread
/// hold its own local — no global state involved.
fn processVisionImages(
    allocator: std.mem.Allocator,
    lm: *LoadedModel,
    vision_enc: *VisionEncoder,
    msgs: []const chat_mod.Message,
    out_n_vision: *usize,
    out_n_audio: *usize,
) !?mlx.mlx_array {
    out_n_vision.* = 0;
    out_n_audio.* = 0;
    // Only process media from the LAST user message. Previous turns' images /
    // audio were already processed in their original request; re-processing
    // wastes context and causes stale feature confusion.
    var last_user_images: ?[]const chat_mod.ImageData = null;
    var last_user_audio: ?[]const chat_mod.AudioData = null;
    var i = msgs.len;
    while (i > 0) {
        i -= 1;
        if (std.mem.eql(u8, msgs[i].role, "user")) {
            last_user_images = msgs[i].images;
            last_user_audio = msgs[i].audio;
            break;
        }
    }

    const images: []const chat_mod.ImageData = last_user_images orelse &.{};
    const audio: []const chat_mod.AudioData = last_user_audio orelse &.{};
    if (images.len == 0 and audio.len == 0) return null;

    log.info("Multimodal: processing {d} image(s), {d} audio clip(s)\n", .{ images.len, audio.len });

    // Phase A4: route encoding to the scheduler's inference thread when
    // available. Conn thread only decodes pixels/PCM (CPU); the mlx ops
    // (array construction, encoder forward, concatenation) run on the
    // inference thread so we don't disturb the JIT-compiled stream binding.
    if (global_scheduler) |sch| {
        var pix_list = std.ArrayList(scheduler_mod.VisionImagePixels).empty;
        defer pix_list.deinit(allocator);
        try pix_list.ensureTotalCapacity(allocator, images.len);
        for (images) |img| {
            pix_list.appendAssumeCapacity(.{
                .pixels = img.pixels,
                .width = @intCast(img.width),
                .height = @intCast(img.height),
                .grid_h = img.grid_h,
                .grid_w = img.grid_w,
            });
        }
        var aud_list = std.ArrayList([]const u8).empty;
        defer aud_list.deinit(allocator);
        try aud_list.ensureTotalCapacity(allocator, audio.len);
        for (audio) |a| aud_list.appendAssumeCapacity(a.samples);
        var req = scheduler_mod.VisionEncodeRequest{
            .model = lm,
            .images = pix_list.items,
            .audio = aud_list.items,
            .allocator = allocator,
        };
        const arr = sch.encodeVision(&req) catch |err| {
            if (req.error_name) |e| {
                log.err("Vision encode (via scheduler) failed: {s}\n", .{e});
                allocator.free(e);
            }
            return err;
        };
        out_n_vision.* = req.n_vision_tokens;
        out_n_audio.* = req.n_audio_tokens;
        const ve_shape = mlx.getShape(arr);
        if (ve_shape.len >= 3) {
            log.info("  Multimodal: → [{d},{d},{d}] tokens ({d} vision + {d} audio)\n", .{ ve_shape[0], ve_shape[1], ve_shape[2], req.n_vision_tokens, req.n_audio_tokens });
        }
        return arr;
    }

    // Legacy path (offline / no scheduler): encode on this thread. Encode
    // each image and concatenate embeddings along the token dimension. Each
    // image produces [1, N, hidden], concatenated → [1, total_tokens, hidden].
    var emb_parts = std.ArrayList(mlx.mlx_array).empty;
    defer {
        for (emb_parts.items) |e| _ = mlx.mlx_array_free(e);
        emb_parts.deinit(allocator);
    }

    for (images) |img| {
        var emb: mlx.mlx_array = undefined;
        if (img.grid_h > 0) {
            const n: usize = @as(usize, img.grid_h) * img.grid_w;
            const feat: usize = (img.pixels.len / 4) / n;
            const shape = [_]c_int{ @intCast(n), @intCast(feat) };
            const pixel_arr = mlx.mlx_array_new_data(img.pixels.ptr, &shape, 2, .float32);
            defer _ = mlx.mlx_array_free(pixel_arr);
            emb = try vision_enc.forwardQwen(pixel_arr, img.grid_h, img.grid_w);
        } else {
            const h: c_int = @intCast(img.height);
            const w: c_int = @intCast(img.width);
            const shape = [_]c_int{ 1, 3, h, w };
            const pixel_arr = mlx.mlx_array_new_data(img.pixels.ptr, &shape, 4, .float32);
            defer _ = mlx.mlx_array_free(pixel_arr);
            emb = try vision_enc.forward(pixel_arr);
        }
        const es = mlx.getShape(emb);
        out_n_vision.* += @intCast(es[1]);
        try emb_parts.append(allocator, emb);
    }

    for (audio) |clip| {
        const n_samples = clip.samples.len / 4;
        if (n_samples == 0) continue;
        const spt: usize = if (lm.config.?.audio_samples_per_token > 0) lm.config.?.audio_samples_per_token else 640;
        const n_frames = (n_samples + spt - 1) / spt;
        const padded = n_frames * spt;
        const buf = try allocator.alloc(f32, padded);
        @memset(buf, 0);
        @memcpy(std.mem.sliceAsBytes(buf)[0..clip.samples.len], clip.samples);
        const shape = [_]c_int{ 1, @intCast(n_frames), @intCast(spt) };
        const frames_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 3, .float32);
        allocator.free(buf);
        defer _ = mlx.mlx_array_free(frames_arr);
        const emb = try vision_enc.forwardAudio(frames_arr);
        out_n_audio.* += n_frames;
        try emb_parts.append(allocator, emb);
    }

    if (emb_parts.items.len == 0) return null;
    if (emb_parts.items.len == 1) {
        // Single part — return directly. Detach so the defer-free skips it.
        const out = emb_parts.items[0];
        emb_parts.items[0] = mlx.mlx_array_new();
        return out;
    }

    // Multiple parts (vision + audio, or multiple clips) — concat along tokens.
    const cat_vec = mlx.mlx_vector_array_new_data(emb_parts.items.ptr, emb_parts.items.len);
    defer _ = mlx.mlx_vector_array_free(cat_vec);
    var combined = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&combined, cat_vec, 1, vision_enc.s));
    return combined;
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

    // Insert: BOI + n_tokens × image_token + EOI. Qwen3-VL wraps the image-pad run
    // with <|vision_start|> / <|vision_end|> instead (get_rope_index keys on a
    // vision_start immediately followed by image tokens).
    const boi: u32 = if (config.qwen_vision) config.vision_start_token_id else config.boi_token_id;
    const eoi: u32 = if (config.qwen_vision) config.vision_end_token_id else config.eoi_token_id;
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

/// Locate the byte offset just after the last user-turn marker in `prompt_ids`.
/// Mirrors insertImageTokens' search; shared by the multimodal inserter.
fn userTurnInsertPos(prompt_ids: []const u32, config: *const model_mod.ModelConfig) usize {
    const marker = config.userTurnMarkerSlice();
    if (marker.len > 0 and prompt_ids.len >= marker.len) {
        var i = prompt_ids.len - marker.len;
        while (true) {
            if (std.mem.eql(u32, prompt_ids[i .. i + marker.len], marker)) return i + marker.len;
            if (i == 0) break;
            i -= 1;
        }
    }
    return if (prompt_ids.len > 5) prompt_ids.len - 5 else 0;
}

/// Insert an image block (BOI + n_image × image_token + EOI) followed by an
/// audio block (BOA + n_audio × audio_token + EOA) at the last user turn. The
/// block order MUST match the [vision ; audio] concatenation order of the
/// soft-token embedding so the splice scatters each row into its slot.
/// Gemma 4 12B unified routes both modalities through one splice channel.
/// Qwen3-VL interleaved M-RoPE position-id table for a request. `pos` (owned) is
/// the flat [3 × total] i32 table threaded to the slot; null for non-Qwen / no
/// images. Pass-through bundle so sub-handlers thread one value, not three.
pub const MropeData = struct {
    pos: ?[]const i32 = null,
    total: usize = 0,
    delta: i32 = 0,
};

/// Compute the interleaved-M-RoPE table from the FINAL prompt_ids (after image-pad
/// expansion) + the last user turn's image grids. Returns an empty bundle when the
/// model isn't Qwen-vision or there are no images. Caller owns `pos`.
fn computeQwenMrope(allocator: std.mem.Allocator, prompt_ids: []const u32, msgs: []const chat_mod.Message, config: *const model_mod.ModelConfig) !MropeData {
    if (!config.qwen_vision) return .{};
    // Collect the last user message's image grids (full patch grid per image).
    var grids = std.ArrayList(mrope_mod.ImageGrid).empty;
    defer grids.deinit(allocator);
    var i = msgs.len;
    while (i > 0) {
        i -= 1;
        if (std.mem.eql(u8, msgs[i].role, "user")) {
            if (msgs[i].images) |imgs| for (imgs) |im| {
                if (im.grid_h > 0) try grids.append(allocator, .{ .t = 1, .h = im.grid_h, .w = im.grid_w });
            };
            break;
        }
    }
    if (grids.items.len == 0) return .{};

    var ri = mrope_mod.getRopeIndex(allocator, prompt_ids, grids.items, config.image_token_id, config.video_token_id, config.vision_start_token_id, config.qv_merge) catch |err| {
        log.warn("M-RoPE get_rope_index failed ({s}); falling back to scalar RoPE\n", .{@errorName(err)});
        return .{};
    };
    defer ri.deinit();
    const total = prompt_ids.len;
    const flat = try allocator.alloc(i32, 3 * total);
    @memcpy(flat[0..total], ri.pos[0]);
    @memcpy(flat[total .. 2 * total], ri.pos[1]);
    @memcpy(flat[2 * total .. 3 * total], ri.pos[2]);
    log.info("  M-RoPE: {d} images, position ids over {d} tokens, decode delta {d}\n", .{ grids.items.len, total, ri.delta });
    return .{ .pos = flat, .total = total, .delta = ri.delta };
}

fn insertMultimodalTokens(
    allocator: std.mem.Allocator,
    prompt_ids: []const u32,
    image_token_id: u32,
    n_image: usize,
    audio_token_id: u32,
    n_audio: usize,
    config: *const model_mod.ModelConfig,
) ![]u32 {
    const want_image = image_token_id != 0 and n_image > 0;
    const want_audio = audio_token_id != 0 and n_audio > 0;
    if (!want_image and !want_audio) return try allocator.dupe(u32, prompt_ids);

    const insert_pos = userTurnInsertPos(prompt_ids, config);

    // Qwen3-VL wraps the image-pad run with <|vision_start|>/<|vision_end|>
    // (get_rope_index keys on vision_start immediately followed by image tokens);
    // Gemma uses BOI/EOI.
    const boi = if (config.qwen_vision) config.vision_start_token_id else config.boi_token_id;
    const eoi = if (config.qwen_vision) config.vision_end_token_id else config.eoi_token_id;
    const boa = config.boa_token_id;
    const eoa = config.eoa_token_id;

    var seg = std.ArrayList(u32).empty;
    defer seg.deinit(allocator);
    if (want_image) {
        if (boi > 0) try seg.append(allocator, boi);
        try seg.appendNTimes(allocator, image_token_id, n_image);
        if (eoi > 0) try seg.append(allocator, eoi);
    }
    if (want_audio) {
        if (boa > 0) try seg.append(allocator, boa);
        try seg.appendNTimes(allocator, audio_token_id, n_audio);
        if (eoa > 0) try seg.append(allocator, eoa);
    }

    const new_len = prompt_ids.len + seg.items.len;
    const result = try allocator.alloc(u32, new_len);
    @memcpy(result[0..insert_pos], prompt_ids[0..insert_pos]);
    @memcpy(result[insert_pos .. insert_pos + seg.items.len], seg.items);
    @memcpy(result[insert_pos + seg.items.len ..], prompt_ids[insert_pos..]);

    log.info("  Inserted {d} image + {d} audio soft tokens at position {d} (prompt: {d} -> {d} tokens)\n", .{ n_image, n_audio, insert_pos, prompt_ids.len, new_len });
    return result;
}

/// Decode an `input_audio.data` payload into raw float32-LE PCM samples for the
/// Gemma 4 12B unified audio embedder. Accepts a bare base64 string or a
/// `data:audio/...;base64,...` URL. The decoded bytes are interpreted as
/// little-endian float32 mono samples at 16 kHz (the client resamples). Returns
/// null on decode failure or a non-multiple-of-4 byte length.
fn parseAudioContent(allocator: std.mem.Allocator, data: []const u8) ?chat_mod.AudioData {
    const b64 = if (std.mem.indexOf(u8, data, ";base64,")) |sep| data[sep + 8 ..] else data;
    const decoded_size = std.base64.standard.Decoder.calcSizeForSlice(b64) catch return null;
    if (decoded_size == 0 or decoded_size % 4 != 0) return null;
    const raw_buf = allocator.alloc(u8, decoded_size) catch return null;
    std.base64.standard.Decoder.decode(raw_buf, b64) catch {
        allocator.free(raw_buf);
        return null;
    };
    return .{ .samples = raw_buf };
}

/// Decode a JPEG/PNG image buffer to float32 CHW pixels, resized to target_size.
/// Uses stb_image for decoding, then nearest-neighbor resize + CHW conversion.
/// Decode an image content URL into preprocessed float32 CHW pixels. Supports:
///   data:image/x-mlx-pixels;base64,... (already-preprocessed float32 CHW)
///   data:image/jpeg|png|webp|...;base64,... (decoded + resized via stb_image / libwebp)
/// Returns null on any decode failure (caller treats as missing image).
/// Derive per-request image preprocessing params from the loaded model config.
fn visionPreprocFromConfig(config: *const model_mod.ModelConfig) chat_mod.VisionPreproc {
    if (!config.qwen_vision) return .{};
    return .{ .qwen = true, .patch = config.qv_patch, .tps = config.qv_temporal_patch, .merge = config.qv_merge };
}

fn parseImageUrlContent(allocator: std.mem.Allocator, url: []const u8, vp: chat_mod.VisionPreproc) ?chat_mod.ImageData {
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
        // JPEG/PNG/WebP — decode + resize + convert to float32 CHW (Gemma) or
        // smart-resized merge-order pixel_values (Qwen3-VL).
        const img = decodeImageToPixels(allocator, raw_buf, vp);
        allocator.free(raw_buf);
        return img;
    }

    allocator.free(raw_buf);
    return null;
}

fn decodeImageToPixels(allocator: std.mem.Allocator, encoded: []const u8, vp: chat_mod.VisionPreproc) ?chat_mod.ImageData {
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

    // Qwen3-VL: smart-resize to a multiple of patch·merge within [min,max] pixels,
    // normalize (x/255−0.5)/0.5, then emit the processor's merge-order pixel_values
    // [N, C·tps·ps·ps]. The encoder is QwenVision; `grid_h/grid_w` carry the full
    // patch grid (token count = (gh/merge)·(gw/merge)).
    if (vp.qwen) {
        const rs = qwen_vision.smartResizeDefault(src_h, src_w);
        const rh = rs.h;
        const rw = rs.w;
        const C: u32 = 3;
        const gh = rh / vp.patch;
        const gw = rw / vp.patch;
        const n: usize = @as(usize, gh) * gw;
        const feat: usize = @as(usize, C) * vp.tps * vp.patch * vp.patch;
        const plane: usize = @as(usize, rh) * rw;

        const chw = allocator.alloc(f32, @as(usize, C) * plane) catch return null;
        defer allocator.free(chw);
        for (0..rh) |ty| {
            for (0..rw) |tx| {
                const sx_f = @as(f32, @floatFromInt(tx)) * @as(f32, @floatFromInt(src_w)) / @as(f32, @floatFromInt(rw));
                const sy_f = @as(f32, @floatFromInt(ty)) * @as(f32, @floatFromInt(src_h)) / @as(f32, @floatFromInt(rh));
                const sx: usize = @min(@as(usize, @intFromFloat(sx_f)), src_w - 1);
                const sy: usize = @min(@as(usize, @intFromFloat(sy_f)), src_h - 1);
                const sidx = (sy * src_w + sx) * 3;
                const didx = ty * rw + tx;
                inline for (0..3) |c| {
                    // (x/255 − 0.5)/0.5 == x/127.5 − 1.0
                    chw[c * plane + didx] = @as(f32, @floatFromInt(px[sidx + c])) / 127.5 - 1.0;
                }
            }
        }

        const pv_bytes = allocator.alloc(u8, n * feat * 4) catch return null;
        const pv_f32 = @as([*]f32, @alignCast(@ptrCast(pv_bytes.ptr)))[0 .. n * feat];
        qwen_vision.buildPixelValues(pv_f32, chw, C, rh, rw, vp.patch, vp.tps, vp.merge);
        log.info("  Decoded {d}x{d} image → Qwen grid {d}x{d} ({d} tokens, resized {d}x{d})\n", .{ src_w, src_h, gh, gw, n / (@as(usize, vp.merge) * vp.merge), rw, rh });
        return .{ .pixels = pv_bytes, .width = rw, .height = rh, .grid_h = gh, .grid_w = gw };
    }

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

/// SSE comment keepalive for the OpenAI-style streaming surfaces. Comments
/// are SSE-spec-legal and skipped by every SSE parser. No-op on the
/// WebSocket transport — a raw comment line would corrupt WS framing, and
/// WS has protocol-level liveness of its own.
fn sendStreamKeepalive(stream: *Conn) !void {
    if (stream.ws_mode != null) return;
    try stream.writeAll(": keepalive\n\n");
}

/// Which no-op event a surface uses to prove liveness.
const KeepaliveStyle = enum { sse_comment, anthropic_ping };

/// Emit a keepalive iff the socket has been SILENT for `STREAM_KEEPALIVE_MS`.
///
/// Every streaming token loop calls this once per iteration — not just on the
/// `.idle` (waiting-for-first-token) path. Those two are different facts: a
/// surface buffering tokens for tool-call or thinking detection is receiving
/// tokens steadily while writing nothing, and only bytes on the wire hold off
/// a client's idle-body timeout (Node's `fetch`/undici: 300 s, then
/// `TypeError: terminated`). A one-shot `write_file` of a whole source file
/// buffers for exactly that long.
fn beatStreamKeepalive(stream: *Conn, style: KeepaliveStyle) !void {
    if (!stream.keepaliveDue()) return;
    switch (style) {
        .sse_comment => try sendStreamKeepalive(stream),
        .anthropic_ping => try sendAnthropicEvent(stream, "ping", "{\"type\":\"ping\"}"),
    }
    // The WS transport no-ops both senders; stamp regardless so a bridged
    // connection doesn't re-evaluate the deadline on every token.
    stream.heartbeat.noteWrite(nowMsMonotonic(stream.io));
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
    try stream.writeAllNoFlush("event: ");
    try stream.writeAllNoFlush(event_name);
    try stream.writeAllNoFlush("\ndata: ");
    try stream.writeAllNoFlush(data);
    try stream.writeAllNoFlush("\n\n");
    try stream.flush();
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
    lm: *LoadedModel,
) !void {
    // No `lm.transformer.?` — engine-backed (GGUF/ds4) models have a null
    // transformer; the only gate below uses `config.has_hybrid_layers`.
    const tok = lm.tokenizer.?;
    const chat_config = lm.chat_config.?;
    const config = lm.config.?;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST /v1/messages -> 400 (invalid JSON)\n", .{});
        try sendAnthropicError(allocator, stream, "invalid_request_error", "Invalid JSON in request body", 400);
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        log.warn("POST /v1/messages -> 400 (body is not a JSON object)\n", .{});
        try sendAnthropicError(allocator, stream, "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
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
                                if (parseImageUrlContent(allocator, du, visionPreprocFromConfig(config))) |img| {
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

    // Sampling parameters. Omitted fields resolve through CLI flags and the
    // model's generation_config.json — Claude Code omits ALL of them, and the
    // bare temp=1.0/top_p=1.0/no-top_k fallback sampled far outside Qwen's
    // intended envelope (model card wants top_k=20, top_p=0.95).
    const temperature = resolveSamplingDefault(f32, parseJsonFloatOpt(root, "temperature", 0.0, 2.0), server_config.default_temperature, config.gen_temperature, 1.0);
    const top_p = resolveSamplingDefault(f32, parseJsonFloatOpt(root, "top_p", 0.0, 1.0), server_config.default_top_p, config.gen_top_p, 1.0);
    const top_k = resolveSamplingDefault(u32, parseJsonTopKOpt(root, "top_k"), server_config.default_top_k, config.gen_top_k, 0);
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
    var reasoning_budget: i32 = server_config.default_reasoning_budget;
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

    // Wave 1.A: per-request KV-quant override (Anthropic mirror).
    const kv_quant_override = parseKvQuantOverride(root);

    // Per-request PLD override (mirror chat-completions behavior: tools and
    // hybrid SSM do not disable PLD; the adaptive ngram gate below and the
    // runtime acceptance gate handle the rest).
    const pld_explicit_in_json: bool = root.get("enable_pld") != null;
    var enable_pld: bool = if (root.get("enable_pld")) |v|
        (v == .bool and v.bool)
    else
        server_config.default_enable_pld;

    // Drafter: same disable rules as chat-completions parse site.
    const drafter_explicit_in_json: bool = root.get("enable_drafter") != null;
    const lm_default_enable_drafter: bool = lm.drafter != null and !config.isMoe();
    var enable_drafter: bool = if (root.get("enable_drafter")) |v|
        (v == .bool and v.bool)
    else
        lm_default_enable_drafter;
    if (enable_drafter and lm.drafter == null) enable_drafter = false;
    if (enable_drafter and config.has_hybrid_layers) enable_drafter = false;
    var enable_mtp: bool = if (root.get("enable_mtp")) |v|
        (v == .bool and v.bool)
    else
        (lm.mtp != null and !config.isMoe());
    if (enable_mtp and lm.mtp == null) enable_mtp = false;

    // Log request
    const last_msg = messages.items[messages.items.len - 1];
    const preview_len = @min(last_msg.content.len, 80);
    var tool_msg_count: usize = 0;
    for (messages.items) |msg| {
        if (std.mem.eql(u8, msg.role, "tool")) tool_msg_count += 1;
    }
    const tools_len = if (tools_json) |tj| tj.len else 0;
    log.info("POST /v1/messages ({d} msgs, max_tokens={d}, temp={d:.2}, top_p={d:.2}, top_k={d}, stream={}, thinking={}, tools={d}b, tool_msgs={d})\n", .{
        messages.items.len, max_tokens, temperature, top_p, top_k, is_stream, enable_thinking, tools_len, tool_msg_count,
    });
    log.info("  > \"{s}{s}\"\n", .{ last_msg.content[0..preview_len], if (last_msg.content.len > 80) "..." else "" });

    // Format chat template. Iteration 1 timing + Iteration 2 cache. The
    // `effective_tools_json` swap (`null` when has_tools is false) keeps
    // the cache key consistent with what the encoder actually sees.
    const effective_tools_json: ?[]const u8 = if (has_tools) tools_json else null;
    var tokenize_sw = Stopwatch.init(stream.io);
    var prompt_ids_raw = try cachedFormatChat(allocator, stream.io, lm, tok, chat_config, messages.items, effective_tools_json, tool_choice_instruction, enable_thinking);
    const tokenize_ns = tokenize_sw.read();

    // Vision encoder: encode any images on the last user message and splice
    // image tokens into the prompt at the model's configured image_token_id.
    // Phase A8: per-request ownership.
    var local_ve: ?mlx.mlx_array = null;
    defer { if (local_ve) |arr| _ = mlx.mlx_array_free(arr); }
    if (lm.vision_encoder) |ve| {
        var n_vis: usize = 0;
        var n_aud: usize = 0;
        local_ve = processVisionImages(allocator, lm, ve, messages.items, &n_vis, &n_aud) catch |err| blk: {
            log.warn("Vision encoding failed: {}\n", .{err});
            break :blk null;
        };
        if (local_ve != null) {
            const new_ids = try insertMultimodalTokens(allocator, prompt_ids_raw, config.image_token_id, n_vis, config.audio_token_id, n_aud, config);
            allocator.free(prompt_ids_raw);
            prompt_ids_raw = new_ids;
        }
    }
    const prompt_ids = prompt_ids_raw;
    defer allocator.free(prompt_ids);

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
        // Heavy-echo MTP->PLD routing retired 2026-07-13 (see the NOTE at the
        // chat-completions site): MTP wins whenever loaded.
    }

    // Context size enforcement
    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        log.warn("POST /v1/messages -> 400 (prompt {d} tokens exceeds ctx_size {d})\n", .{ prompt_ids.len, effective_ctx });
        try sendAnthropicError(allocator, stream, "invalid_request_error", "Prompt exceeds maximum context length", 400);
        return;
    }

    // Check if attention computation would exceed GPU memory
    if (!try checkAttentionMemory(allocator, stream, prompt_ids.len, config, true, kv_quant_override)) return;

    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len, effective_ctx);
    log.info("  prompt={d} tokens, max_gen={d}, ctx={d}\n", .{ prompt_ids.len, effective_max_tokens, effective_ctx });

    const eos_slice = config.eosTokenSlice();
    const sampling = generate_mod.SamplingParams{
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .repeat_penalty = 1.0,
        .presence_penalty = 0.0,
        .seed = seed,
    };

    // Hand vision ownership to the sub-handler (slot takes it on submit).
    const sub_ve = local_ve;
    local_ve = null;
    if (is_stream) {
        handleAnthropicStreaming(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, has_tools, tools_json, enable_thinking, reasoning_budget, @intCast(prompt_ids.len), enable_pld, enable_drafter, enable_mtp, sub_ve, kv_quant_override, tokenize_ns) catch |err| {
            log.err("  -> streaming error: {}\n", .{err});
            const err_data = std.fmt.allocPrint(allocator,
                \\{{"type":"error","error":{{"type":"api_error","message":"Internal server error: {s}"}}}}
            , .{@errorName(err)}) catch return;
            defer allocator.free(err_data);
            sendAnthropicEvent(stream, "error", err_data) catch {};
        };
    } else {
        handleAnthropicNonStreaming(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, has_tools, tools_json, enable_thinking, reasoning_budget, @intCast(prompt_ids.len), enable_pld, enable_drafter, enable_mtp, sub_ve, kv_quant_override, tokenize_ns) catch |err| {
            log.err("  -> 500 ({s})\n", .{@errorName(err)});
            sendAnthropicError(allocator, stream, "api_error", @errorName(err), 500) catch {};
        };
    }
}

fn handleAnthropicNonStreaming(
    allocator: std.mem.Allocator,
    stream: *Conn,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: generate_mod.SamplingParams,
    eos_token_ids: []const u32,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    has_tools: bool,
    /// OpenAI-shape tools JSON (for bare-args tool-call inference); null when
    /// the request defined no tools.
    tools_json: ?[]const u8,
    enable_thinking: bool,
    reasoning_budget: i32,
    prompt_token_count: u32,
    enable_pld: bool,
    enable_drafter: bool,
    enable_mtp: bool,
    vision_embeddings: ?mlx.mlx_array,
    /// Wave 1.A: per-request KV-quant override.
    kv_quant_override: ?transformer_mod.KVQuantConfig,
    /// Iteration 1 instrumentation: nanoseconds of render+tokenize measured
    /// by the parent handleAnthropicMessages. Threaded through so the
    /// non-streaming response carries `timings.tokenize_ms`.
    tokenize_ns: u64,
) !void {
    // Vision-array ownership: nulled below before scheduler.submit so the
    // early-return defer doesn't double-free.
    var ve_local = vision_embeddings;
    defer { if (ve_local) |arr| _ = mlx.mlx_array_free(arr); }

    var timer = Stopwatch.init(stream.io);

    // Speculative decoding dispatch — same priority as chat-completions
    // (drafter > PLD; PLD runs on hybrid SSM, the drafter does not).
    const config = lm.config.?;
    const use_mtp = enable_mtp and lm.mtp != null and sampling.constraint == null;
    const use_drafter = !use_mtp and enable_drafter and lm.drafter != null and sampling.constraint == null and !config.has_hybrid_layers;
    const use_pld = !use_mtp and !use_drafter and enable_pld and sampling.constraint == null;

    // Anthropic responses never carry logprobs (the API doesn't expose
    // them). Vision-bearing requests transfer ownership of `ve_local` into
    // the slot.
    const slot_ve: ?mlx.mlx_array = blk: {
        const v = ve_local;
        ve_local = null;
        break :blk v;
    };
    // M-RoPE: Anthropic path uses scalar-RoPE fallback for now (faithful M-RoPE
    // wired for /v1/chat/completions; see computeQwenMrope). Qwen image requests
    // still decode correctly — M-RoPE refines spatial grounding only.
    const result = nonStreamingViaScheduler(allocator, global_scheduler.?, lm, tok, prompt_ids, prompt_ids, max_tokens, sampling, eos_token_ids, 0, has_tools, use_pld, use_drafter, use_mtp, getTimeoutNs(), slot_ve, .{}, 0, kv_quant_override, stream) catch |err| switch (err) {
        error.GenerationFailed => return sendAnthropicError(allocator, stream, "api_error", "generation failed", 500),
        else => return err,
    };
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);

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

    // Merge re-opened mid-text thought channels into the leading block so the
    // split/parse below never leaks raw tags (Gemma 12B re-opens channels mid-turn).
    const normalized_text = try chat_mod.normalizeEmbeddedThinkBlocks(allocator, final_text);
    defer if (normalized_text) |n| allocator.free(n);
    if (normalized_text) |n| final_text = n;

    const elapsed_ms = timer.read() / std.time.ns_per_ms;

    // Build content blocks array
    var content = std.ArrayList(u8).empty;
    defer content.deinit(allocator);
    try content.append(allocator, '[');

    var block_count: u32 = 0;

    // Thinking block
    if (enable_thinking) {
        const think_split = chat_mod.splitThinkBlock(final_text, true, promptOpensThink(allocator, lm, tok, prompt_ids));
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
        const found_calls = try parseToolCallsForRequest(allocator, final_text, tools_json);
        if (found_calls) |tool_calls| {
            defer {
                for (tool_calls) |tc| {
                    allocator.free(tc.name);
                    allocator.free(tc.arguments);
                }
                allocator.free(tool_calls);
            }

            var tu_perf_buf: [160]u8 = undefined;
            const tu_perf = formatPerfBracket(&tu_perf_buf, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns);
            log.info("  <- {d}+{d} tokens ({d}ms) [{s}] [tool_use: {d}]\n", .{
                result.prompt_tokens, result.completion_tokens, elapsed_ms, tu_perf, tool_calls.len,
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
            finish_reason = toolCallFinishReason(finish_reason);
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
    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns);
    log.info("  <- {d}+{d} tokens ({d}ms) [{s}] [{s}]\n", .{
        result.prompt_tokens, result.completion_tokens, elapsed_ms, perf, stop_reason,
    });

    // Anthropic spec doesn't standardize a `timings` field; we attach one as
    // an extension (mirrors what /v1/chat/completions already does) so bench
    // tooling can read tokenize_ms / prompt_ms / predicted_ms from either
    // surface without re-implementing the SSE accumulator.
    const timings_obj = try formatTimingsObject(allocator, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns, tokenize_ns);
    defer allocator.free(timings_obj);
    const timings_field = if (timings_obj.len > 0)
        try std.fmt.allocPrint(allocator, ",\"timings\":{s}", .{timings_obj})
    else
        try allocator.alloc(u8, 0);
    defer allocator.free(timings_field);

    const response = try std.fmt.allocPrint(allocator,
        \\{{"id":"msg_{d}","type":"message","role":"assistant","content":{s},"model":"{s}","stop_reason":"{s}","stop_sequence":null,"usage":{{"input_tokens":{d},"output_tokens":{d},"cache_read_input_tokens":{d}}}{s}}}
    , .{
        nowMs(stream.io),
        content.items,
        model_name,
        stop_reason,
        prompt_token_count,
        result.completion_tokens,
        result.cached_tokens,
        timings_field,
    });
    defer allocator.free(response);
    try sendResponse(stream, "200 OK", "application/json", response);
}

fn handleAnthropicStreaming(
    allocator: std.mem.Allocator,
    stream: *Conn,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: generate_mod.SamplingParams,
    eos_token_ids: []const u32,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    has_tools: bool,
    /// OpenAI-shape tools JSON (for bare-args tool-call inference); null when
    /// the request defined no tools.
    tools_json: ?[]const u8,
    enable_thinking: bool,
    reasoning_budget: i32,
    prompt_token_count: u32,
    enable_pld: bool,
    enable_drafter: bool,
    enable_mtp: bool,
    vision_embeddings: ?mlx.mlx_array,
    /// Wave 1.A: per-request KV-quant override.
    kv_quant_override: ?transformer_mod.KVQuantConfig,
    /// Iteration 1: tokenize_ns from parent handler. Anthropic streaming
    /// doesn't currently emit `timings` over SSE (spec doesn't model it),
    /// but plumbing the value through keeps the signature consistent with
    /// non-streaming and unblocks a future message_delta extension.
    tokenize_ns: u64,
) !void {
    _ = tokenize_ns; // reserved; see doc comment
    // Vision-array ownership: held by this handler on entry, transfers to
    // the slot on submit (slot.deinit frees). Nulled before transfer.
    var ve_local = vision_embeddings;
    defer { if (ve_local) |arr| _ = mlx.mlx_array_free(arr); }

    // Pick speculative-decoding mode (regular / PLD / drafter). The token-
    // stream adapter below feeds the per-token Anthropic state machine the
    // same way for all three modes.
    const config = lm.config.?;
    const stream_mode = pickStreamMode(enable_pld, enable_drafter, enable_mtp, lm.drafter != null, lm.mtp != null, config.has_hybrid_layers, sampling.constraint != null, 0);
    if (stream_mode == .pld) log.info("  pld=enabled (streaming, draft_len={d}, key_len={d})\n", .{ server_config.default_pld_draft_len, server_config.default_pld_key_len });
    if (stream_mode == .drafter) log.info("  drafter=enabled (streaming, block_size={d})\n", .{lm.drafter_block_size});
    if (stream_mode == .mtp) log.info("  mtp=enabled (streaming, depth={d})\n", .{lm.mtp_depth});

    var slot_handle: ?*scheduler_mod.Slot = null;
    defer if (slot_handle) |s| global_scheduler.?.complete(s);

    // Transfer vision ownership into the slot.
    const slot_ve_anth = ve_local;
    ve_local = null;
    const sch = global_scheduler.?;
    slot_handle = try sch.submit(.{
        .model = lm,
        .prompt_ids = prompt_ids,
        .full_prompt = prompt_ids,
        .cached_tokens = 0,
        .has_tools = has_tools,
        .sampling = sampling,
        .eos_token_ids = eos_token_ids,
        .max_tokens = max_tokens,
        .timeout_ns = getTimeoutNs(),
        .enable_pld = stream_mode == .pld,
        .enable_drafter = stream_mode == .drafter,
        .drafter = if (stream_mode == .drafter) lm.drafter else null,
        .drafter_block_size = lm.drafter_block_size,
        .enable_mtp = stream_mode == .mtp,
        .mtp = if (stream_mode == .mtp) lm.mtp else null,
        .mtp_depth = lm.mtp_depth,
        .pld_draft_len = server_config.default_pld_draft_len,
        .pld_key_len = server_config.default_pld_key_len,
        .kv_attn_fused = server_config.default_kv_attn_fused,
        .logprobs_n = 0,
        .vision_embeddings = slot_ve_anth,
        .kv_quant_config = kv_quant_override,
    });
    var ts = StreamingTokenStream.initFromSlot(slot_handle.?, stream_mode, eos_token_ids);
    defer ts.deinit(allocator);

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
    // Template-opened think (Qwen 3.5/3.6 render `…assistant\n<think>\n` into
    // the generation prompt): the output's thinking carries NO opener tag.
    // Needed up front by the tools branch — by the time the close tag is the
    // only evidence, visible-text flushing has already leaked the thoughts.
    const opens_think = enable_thinking and promptOpensThink(allocator, lm, tok, prompt_ids);
    // Set once the buffered think block has been split + emitted (tools
    // branch). Releases the buffer hold AND tells the end-of-stream split
    // that the remaining text has no template-opened semantics.
    var think_closed = false;
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
    var client_gone = false;
    var utf8_carry: [3]u8 = undefined;
    var utf8_carry_len: u8 = 0;

    while (true) {
        const token_id: u32 = switch (try ts.nextOrIdle(allocator, Conn.STREAM_KEEPALIVE_MS)) {
            .token => |t| t,
            .done => break,
            .idle => {
                // No tokens yet (long prefill). Probe the peer: an abandoned
                // request must cancel instead of grinding a ghost prefill
                // (Claude Code retries pile up serially otherwise), and the
                // keepalive stops clients timing out on stream silence.
                if (stream.peerClosed()) {
                    log.info("  [cancel] client disconnected while waiting for tokens — cancelling slot\n", .{});
                    slot_handle.?.cancel();
                    client_gone = true;
                    break;
                }
                sendAnthropicEvent(stream, "ping", "{\"type\":\"ping\"}") catch {
                    log.info("  [cancel] keepalive write failed (client disconnected) — cancelling slot\n", .{});
                    slot_handle.?.cancel();
                    client_gone = true;
                    break;
                };
                continue;
            },
        };
        if (stream.peerClosed()) {
            slot_handle.?.cancel();
            client_gone = true;
            break;
        }
        const strip = tok.tok_type == .sentencepiece_bpe;
        const raw_decoded = try decodeTokens(allocator, lm, tok, &[_]u32{token_id}, strip and false);

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
                // Shared gate with the chat-completions stream (the two paths
                // drifted once: this one only recognized think openers present
                // in the OUTPUT, so Qwen-family template-opened thinking
                // streamed as visible text_deltas and a raw `</think>` leaked
                // into Claude Code transcripts). Hermetically pinned per
                // recorded model family by the corpus streaming-gate test.
                switch (chat_mod.streamThinkGate(buf, enable_thinking, think_closed)) {
                    .hold_thinking => {
                        // Incomplete thinking — keep buffering until closed
                    },
                    .split_think => {
                        // Complete think block — split once: thinking block
                        // first, then the visible remainder as text.
                        const split = chat_mod.splitThinkBlock(buf, enable_thinking, opens_think);
                        if (enable_thinking) {
                            if (split.reasoning_content) |rc| {
                                const sd = try std.fmt.allocPrint(allocator,
                                    \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"thinking","thinking":"","signature":""}}}}
                                , .{block_index});
                                defer allocator.free(sd);
                                try sendAnthropicEvent(stream, "content_block_start", sd);
                                try emitAnthropicThinkingDelta(allocator, stream, block_index, rc);
                                try closeAnthropicThinkingBlock(allocator, stream, block_index);
                                block_index += 1;
                            }
                        }
                        if (split.content.len > 0) {
                            if (!text_block_open) {
                                const sd = try std.fmt.allocPrint(allocator,
                                    \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
                                , .{block_index});
                                defer allocator.free(sd);
                                try sendAnthropicEvent(stream, "content_block_start", sd);
                                text_block_open = true;
                            }
                            try emitAnthropicTextDelta(allocator, stream, block_index, split.content);
                        }
                        for (token_texts.items) |tt| allocator.free(tt);
                        token_texts.clearRetainingCapacity();
                        text_buf.clearRetainingCapacity();
                        think_closed = true;
                    },
                    .flush_text => {
                        for (token_texts.items) |tt| {
                            defer allocator.free(tt);
                            // Skip bare think/channel tags that leak without a block
                            if (std.mem.eql(u8, tt, "<|channel>") or std.mem.eql(u8, tt, "<channel|>") or
                                std.mem.eql(u8, tt, "<think>") or std.mem.eql(u8, tt, "</think>"))
                            {
                                continue;
                            }
                            if (!text_block_open) {
                                const sd = try std.fmt.allocPrint(allocator,
                                    \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
                                , .{block_index});
                                defer allocator.free(sd);
                                try sendAnthropicEvent(stream, "content_block_start", sd);
                                text_block_open = true;
                            }
                            try emitAnthropicTextDelta(allocator, stream, block_index, tt);
                        }
                        token_texts.clearRetainingCapacity();
                    },
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

        // See the chat-completions loop: a buffered tool call / thinking block
        // emits no events at all, so liveness must be driven off socket
        // silence, not off token arrival.
        beatStreamKeepalive(stream, .anthropic_ping) catch {
            log.info("  [cancel] keepalive write failed (client disconnected) — cancelling slot\n", .{});
            slot_handle.?.cancel();
            client_gone = true;
            break;
        };
    }

    // Flush remaining think buffer
    if (!client_gone and enable_thinking and thinking_block_open and think_buf.items.len > 0) {
        try emitAnthropicThinkingDelta(allocator, stream, block_index, think_buf.items);
    }
    if (!client_gone and thinking_block_open) {
        try closeAnthropicThinkingBlock(allocator, stream, block_index);
        block_index += 1;
    }

    // Post-generation: snapshot stats from whichever source was active.
    ts.finalize();

    // Handle tool calls
    var finish_reason: []const u8 = if (client_gone) "client_disconnect" else if (stopped) "stop" else ts.finish_reason;

    if (has_tools and !client_gone) {
        if (log.isDebug() and text_buf.items.len > 0) {
            log.debug("  raw generated text before tool parse ({d}b): {s}\n", .{ text_buf.items.len, text_buf.items[0..@min(text_buf.items.len, 4000)] });
            if (std.c.getenv("MLX_SERVE_RAW_DUMP_FILE")) |dump_path| {
                appendRawToolDump(std.mem.span(dump_path), tools_json, text_buf.items);
            }
        }
        // Merge re-opened mid-text thought channels into the leading block so
        // the split/parse below never leaks raw tags (Gemma 12B re-opens
        // channels mid-turn — observed live via Claude Code on this surface).
        const norm_owned = try chat_mod.normalizeEmbeddedThinkBlocks(allocator, text_buf.items);
        defer if (norm_owned) |n| allocator.free(n);
        const gen_text: []const u8 = norm_owned orelse text_buf.items;
        const found_calls = try parseToolCallsForRequest(allocator, gen_text, tools_json);
        if (found_calls) |tool_calls| {
            defer {
                for (tool_calls) |tc| { allocator.free(tc.name); allocator.free(tc.arguments); }
                allocator.free(tool_calls);
            }

            // Close any open text block FIRST, at its own index. The old
            // order emitted the thinking block's start while the text block
            // still held this index, then stopped the text block at a
            // never-started index — Claude Code aborted the turn with
            // "API Error: Content block not found".
            if (text_block_open) {
                const sd = try std.fmt.allocPrint(allocator, "{{\"type\":\"content_block_stop\",\"index\":{d}}}", .{block_index});
                defer allocator.free(sd);
                try sendAnthropicEvent(stream, "content_block_stop", sd);
                block_index += 1;
                text_block_open = false;
            }

            // Emit thinking from buffered text if needed. Once the in-loop
            // split already emitted it, the remaining buffer has no
            // template-opened semantics — passing `opens_think` unguarded
            // would misfile the visible tail as reasoning.
            if (enable_thinking) {
                const think_split = chat_mod.splitThinkBlock(gen_text, true, opens_think and !think_closed);
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
            finish_reason = toolCallFinishReason(finish_reason);
        } else {
            // No tool calls — flush buffered tokens
            if (enable_thinking) {
                var full_text = std.ArrayList(u8).empty;
                defer full_text.deinit(allocator);
                for (token_texts.items) |t| try full_text.appendSlice(allocator, t);
                const flush_norm = try chat_mod.normalizeEmbeddedThinkBlocks(allocator, full_text.items);
                defer if (flush_norm) |n| allocator.free(n);
                const flush_text: []const u8 = flush_norm orelse full_text.items;
                const think_split = chat_mod.splitThinkBlock(flush_text, true, opens_think and !think_closed);
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

    const total_prompt = ts.prompt_tokens;
    const stop_reason = mapFinishToStopReason(finish_reason);

    if (!client_gone) {
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

        // message_delta. Scheduler accounts for any prompt-cache hits in `ts.prompt_tokens`.
        // cache_read_input_tokens rides here (not message_start) because the
        // prefix-cache hit count is only known after prefill; clients merge
        // message_delta usage into the final message per Anthropic semantics.
        {
            const md = try std.fmt.allocPrint(allocator,
                \\{{"type":"message_delta","delta":{{"stop_reason":"{s}","stop_sequence":null}},"usage":{{"output_tokens":{d},"cache_read_input_tokens":{d}}}}}
            , .{ stop_reason, ts.completion_tokens, ts.cached_tokens });
            defer allocator.free(md);
            try sendAnthropicEvent(stream, "message_delta", md);
        }
        try sendAnthropicEvent(stream, "message_stop", "{\"type\":\"message_stop\"}");
    }

    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, ts.prompt_tokens, ts.cached_tokens, ts.completion_tokens, ts.prefill_ns, ts.decode_ns);
    log.info("  <- {d}+{d} tokens streamed [{s}] [{s}]\n", .{
        total_prompt, ts.completion_tokens, perf, stop_reason,
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
    // Compaction drops images, so the preprocessing selector is irrelevant here.
    var pi = responses_mod.parseInput(allocator, input_val, instructions, prev_messages, parseImageUrlContent, .{}) catch |err| {
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
    lm: *LoadedModel,
) !void {
    // No `lm.transformer.?` — engine-backed (GGUF/ds4) models have a null
    // transformer; the only gates below use `config.has_hybrid_layers`.
    const tok = lm.tokenizer.?;
    const chat_config = lm.chat_config.?;
    const config = lm.config.?;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST /v1/responses -> 400 (invalid JSON)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", 400);
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        log.warn("POST /v1/responses -> 400 (body is not a JSON object)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
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
    const wants_json = std.mem.eql(u8, text_format.kind, "json_schema") or std.mem.eql(u8, text_format.kind, "json_object");
    // Belt + braces, mirroring the chat-completions path: bare `json_object`
    // carries no schema, so without a synthesized permissive one there is no
    // grammar constraint at all and prompt-ignoring models (Gemma 3 wraps the
    // object in a ```json fence — caught live by llmprobe
    // structured-json-mode-valid, 2026-06-10) emit invalid JSON.
    var json_object_schema_holder: ?std.json.Parsed(std.json.Value) = null;
    defer if (json_object_schema_holder) |*p| p.deinit();
    if (text_format.schema_value == null and std.mem.eql(u8, text_format.kind, "json_object")) {
        json_object_schema_holder = std.json.parseFromSlice(std.json.Value, allocator, "{\"type\":\"object\",\"additionalProperties\":true}", .{}) catch null;
    }
    const grammar_schema_val: ?std.json.Value = if (json_object_schema_holder) |*p|
        p.value
    else
        text_format.schema_value;
    const has_current_tool_output = responses_mod.inputContainsFunctionCallOutput(input_val);

    // ── sampling params ──
    // Track whether the user explicitly supplied max_output_tokens so we can
    // echo `null` (vs. our internal default) in the response envelope.
    const req_max_output_tokens: ?u32 = blk: {
        const v = root.get("max_output_tokens") orelse root.get("max_tokens");
        // <= 0 is treated as "auto" (null here → context-pegged default below).
        break :blk if (v) |val| switch (val) {
            .integer => |i| if (i > 0) @as(?u32, @intCast(i)) else null,
            else => null,
        } else null;
    };
    const max_tokens: u32 = req_max_output_tokens orelse
        (if (wants_json) DEFAULT_STRUCTURED_OUTPUT_MAX_TOKENS else omittedMaxTokensDefault(getEffectiveContextLength(config)));
    const temperature = resolveSamplingDefault(f32, parseJsonFloatOpt(root, "temperature", 0.0, 2.0), server_config.default_temperature, config.gen_temperature, 1.0);
    const top_p = resolveSamplingDefault(f32, parseJsonFloatOpt(root, "top_p", 0.0, 1.0), server_config.default_top_p, config.gen_top_p, 1.0);
    const top_k = resolveSamplingDefault(u32, parseJsonTopKOpt(root, "top_k"), server_config.default_top_k, config.gen_top_k, 0);
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
    const reasoning_cfg = responses_mod.parseReasoning(root.get("reasoning"), server_config.default_reasoning_budget);
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
    var pi = responses_mod.parseInput(allocator, input_val, instructions, prev_messages, parseImageUrlContent, visionPreprocFromConfig(config)) catch |err| {
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
    // Iteration 1 timing + Iteration 2 cache. Responses sees the same
    // cache as chat-completions / messages because they all hash the
    // same canonical (messages, tools, flags) tuple.
    var tokenize_sw = Stopwatch.init(stream.io);
    var prompt_ids_raw = try cachedFormatChat(allocator, stream.io, lm, tok, chat_config, pi.messages.items, active_tools_json, active_tool_choice_instruction, enable_thinking);
    const tokenize_ns = tokenize_sw.read();

    // ── vision encoder ──
    // Phase A8: per-request ownership. Defer frees if we don't end up
    // transferring the array to a scheduler slot.
    var local_ve: ?mlx.mlx_array = null;
    defer { if (local_ve) |arr| _ = mlx.mlx_array_free(arr); }
    if (lm.vision_encoder) |ve| {
        var n_vis: usize = 0;
        var n_aud: usize = 0;
        local_ve = processVisionImages(allocator, lm, ve, pi.messages.items, &n_vis, &n_aud) catch |err| blk: {
            log.warn("Vision encoding failed: {}\n", .{err});
            break :blk null;
        };
        if (local_ve != null) {
            const new_ids = try insertMultimodalTokens(allocator, prompt_ids_raw, config.image_token_id, n_vis, config.audio_token_id, n_aud, config);
            allocator.free(prompt_ids_raw);
            prompt_ids_raw = new_ids;
        }
    }
    const prompt_ids = prompt_ids_raw;
    defer allocator.free(prompt_ids);

    // ── context limit ──
    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Prompt exceeds maximum context length", 400);
        return;
    }
    const kv_quant_override = parseKvQuantOverride(root);
    if (!try checkAttentionMemory(allocator, stream, prompt_ids.len, config, false, kv_quant_override)) return;
    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len, effective_ctx);

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
        // Timings are all zero on the skeleton — the real numbers appear on
        // the final `response.completed` envelope after generation.
        const skel = try buildResponsesEnvelope(
            stream.io, allocator, esc_resp_id, esc_model,
            "in_progress", "[]",
            0, 0, 0, 0,
            should_store, prev_id,
            false, false, response_echo,
            0, 0, tokenize_ns, 0,
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

    // Wave 1.A: per-request KV-quant override (Responses path mirror).

    // Per-request PLD override for the Responses path. Mirror the
    // chat-completions auto-disable logic exactly so the same prompt picks
    // the same path on /v1/chat/completions and /v1/responses.
    const pld_explicit_in_json: bool = root.get("enable_pld") != null;
    var enable_pld_resp: bool = if (root.get("enable_pld")) |v|
        (v == .bool and v.bool)
    else
        server_config.default_enable_pld;

    // Drafter (Responses-side parsing). Same disable rules as the chat
    // and Anthropic parse sites.
    const drafter_explicit_in_json: bool = root.get("enable_drafter") != null;
    const lm_default_enable_drafter_resp: bool = lm.drafter != null and !config.isMoe();
    var enable_drafter_resp: bool = if (root.get("enable_drafter")) |v|
        (v == .bool and v.bool)
    else
        lm_default_enable_drafter_resp;
    if (enable_drafter_resp and lm.drafter == null) enable_drafter_resp = false;
    if (enable_drafter_resp and config.has_hybrid_layers) enable_drafter_resp = false;
    var enable_mtp_resp: bool = if (root.get("enable_mtp")) |v|
        (v == .bool and v.bool)
    else
        (lm.mtp != null and !config.isMoe());
    if (enable_mtp_resp and lm.mtp == null) enable_mtp_resp = false;

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
        // Heavy-echo MTP->PLD routing retired 2026-07-13 (see the NOTE at the
        // chat-completions site): MTP wins whenever loaded.
    }

    var result: generate_mod.GenerationResult = undefined;
    if (is_stream) {
        // Pick speculative-decoding mode for the streaming Responses path.
        const stream_mode = pickStreamMode(enable_pld_resp, enable_drafter_resp, enable_mtp_resp, lm.drafter != null, lm.mtp != null, config.has_hybrid_layers, sampling.constraint != null, 0);
        if (stream_mode == .pld) log.info("  pld=enabled (streaming responses, draft_len={d}, key_len={d})\n", .{ server_config.default_pld_draft_len, server_config.default_pld_key_len });
        if (stream_mode == .drafter) log.info("  drafter=enabled (streaming responses, block_size={d})\n", .{lm.drafter_block_size});
        if (stream_mode == .mtp) log.info("  mtp=enabled (streaming responses, depth={d})\n", .{lm.mtp_depth});

        var slot_handle: ?*scheduler_mod.Slot = null;
        defer if (slot_handle) |s| global_scheduler.?.complete(s);

        // Transfer vision ownership into the slot.
        const slot_ve_resp = local_ve;
        local_ve = null;
        const sch = global_scheduler.?;
        slot_handle = try sch.submit(.{
            .model = lm,
            .prompt_ids = prompt_ids,
            .full_prompt = prompt_ids,
            .cached_tokens = 0,
            .has_tools = active_has_tools,
            .sampling = sampling,
            .eos_token_ids = eos_slice,
            .max_tokens = effective_max_tokens,
            .timeout_ns = getTimeoutNs(),
            .enable_pld = stream_mode == .pld,
            .enable_drafter = stream_mode == .drafter,
            .drafter = if (stream_mode == .drafter) lm.drafter else null,
            .drafter_block_size = lm.drafter_block_size,
            .enable_mtp = stream_mode == .mtp,
            .mtp = if (stream_mode == .mtp) lm.mtp else null,
            .mtp_depth = lm.mtp_depth,
            .vision_embeddings = slot_ve_resp,
            .pld_draft_len = server_config.default_pld_draft_len,
            .pld_key_len = server_config.default_pld_key_len,
            .kv_attn_fused = server_config.default_kv_attn_fused,
            .logprobs_n = 0,
            .kv_quant_config = kv_quant_override,
        });
        var ts = StreamingTokenStream.initFromSlot(slot_handle.?, stream_mode, eos_slice);
        defer ts.deinit(allocator);

        var raw_buf = std.ArrayList(u8).empty;
        defer raw_buf.deinit(allocator);
        var token_ids_buf = std.ArrayList(u32).empty;
        defer token_ids_buf.deinit(allocator);

        var utf8_carry: [3]u8 = undefined;
        var utf8_carry_len: u8 = 0;
        var stopped = false;
        var client_gone = false;
        var in_think_block = enable_thinking;
        var think_buf = std.ArrayList(u8).empty;
        defer think_buf.deinit(allocator);
        var skipped_think_open = false;
        var live_output_index: u32 = 0;

        while (true) {
            const token_id: u32 = switch (try ts.nextOrIdle(allocator, Conn.STREAM_KEEPALIVE_MS)) {
                .token => |t| t,
                .done => break,
                .idle => {
                    // No tokens yet (long prefill). Probe the peer: an abandoned
                    // request must cancel instead of grinding a ghost prefill
                    // (Claude Code retries pile up serially otherwise), and the
                    // keepalive stops clients timing out on stream silence.
                    if (stream.peerClosed()) {
                        log.info("  [cancel] client disconnected while waiting for tokens — cancelling slot\n", .{});
                        slot_handle.?.cancel();
                        client_gone = true;
                        break;
                    }
                    sendStreamKeepalive(stream) catch {
                        log.info("  [cancel] keepalive write failed (client disconnected) — cancelling slot\n", .{});
                        slot_handle.?.cancel();
                        client_gone = true;
                        break;
                    };
                    continue;
                },
            };
            if (stream.peerClosed()) {
                slot_handle.?.cancel();
                client_gone = true;
                break;
            }
            try token_ids_buf.append(allocator, token_id);
            const raw_decoded = try decodeTokens(allocator, lm, tok, &[_]u32{token_id}, false);

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

            // Beat BEFORE the tool early-continue below: a tool-active request
            // emits nothing for its whole generation, and the thinking branch
            // holds until its close tag. Both look identical to a dead server
            // from the client's socket.
            beatStreamKeepalive(stream, .sse_comment) catch {
                log.info("  [cancel] keepalive write failed (client disconnected) — cancelling slot\n", .{});
                slot_handle.?.cancel();
                client_gone = true;
                break;
            };

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
        if (!client_gone and in_think_block and think_buf.items.len > 0 and !active_has_tools) {
            if (!streamed_reasoning_started) {
                streamed_reasoning_id = try responses_mod.makeId(stream.io, allocator, "rs");
                streamed_reasoning_index = live_output_index;
                try emitResponsesReasoningStart(allocator, stream, &seq_num, streamed_reasoning_index, streamed_reasoning_id.?);
                streamed_reasoning_started = true;
            }
            try emitResponsesReasoningDelta(allocator, stream, &seq_num, streamed_reasoning_index, streamed_reasoning_id.?, think_buf.items);
        }

        ts.finalize();

        if (client_gone) {
            // Peer disconnected mid-decode. We've already cancelled the slot;
            // tear down without spending more I/O on terminal envelope events
            // (which would just fail or pile into a dead socket buffer).
            log.info("  <- {d}+{d} tokens streamed [client_disconnect]\n", .{ ts.prompt_tokens, ts.completion_tokens });
            return;
        }

        result = .{
            .text = try raw_buf.toOwnedSlice(allocator),
            .token_ids = try token_ids_buf.toOwnedSlice(allocator),
            .prompt_tokens = ts.prompt_tokens,
            .completion_tokens = ts.completion_tokens,
            .finish_reason = if (stopped) "stop" else ts.finish_reason,
            .prefill_tps = 0.0,
            .decode_tps = 0.0,
        };
    } else {
        // Non-streaming Responses: spec-decode dispatch (drafter > PLD) so
        // /v1/responses gets the same speedup as /v1/chat/completions.
        const use_mtp = enable_mtp_resp and lm.mtp != null and sampling.constraint == null;
        const use_drafter = !use_mtp and enable_drafter_resp and lm.drafter != null and sampling.constraint == null and !config.has_hybrid_layers;
        const use_pld = !use_mtp and !use_drafter and enable_pld_resp and sampling.constraint == null;
        // Transfer vision ownership into the slot.
        const slot_ve_ns: ?mlx.mlx_array = blk: {
            const v = local_ve;
            local_ve = null;
            break :blk v;
        };
        result = nonStreamingViaScheduler(allocator, global_scheduler.?, lm, tok, prompt_ids, prompt_ids, effective_max_tokens, sampling, eos_slice, 0, active_has_tools, use_pld, use_drafter, use_mtp, getTimeoutNs(), slot_ve_ns, .{}, 0, kv_quant_override, stream) catch |err| switch (err) {
            error.GenerationFailed => return sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", "generation failed", null),
            else => return err,
        };
    }
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);

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

    // Merge re-opened mid-text thought channels into the leading block so the
    // split/parse below never leaks raw tags (Gemma 12B re-opens channels mid-turn).
    const normalized_text = try chat_mod.normalizeEmbeddedThinkBlocks(allocator, final_text);
    defer if (normalized_text) |n| allocator.free(n);
    if (normalized_text) |n| final_text = n;

    // ── split thinking & tool calls ──
    const think_split = chat_mod.splitThinkBlock(final_text, enable_thinking, enable_thinking and promptOpensThink(allocator, lm, tok, prompt_ids));
    const reasoning_text: ?[]const u8 = if (enable_thinking) think_split.reasoning_content else null;
    const visible_text: []const u8 = if (enable_thinking) think_split.content else chat_mod.stripThinkBlock(final_text);

    var tool_calls: ?[]chat_mod.ParsedToolCall = null;
    if (active_has_tools) {
        tool_calls = try parseToolCallsForRequest(allocator, final_text, active_tools_json);
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
    // reasoning token count for usage.output_tokens_details (re-encode of the
    // split reasoning text — exact modulo merge boundaries).
    const reasoning_tok_count: u32 = blk: {
        const rt = reasoning_text orelse break :blk 0;
        const rids = tok.encode(allocator, rt) catch break :blk 0;
        defer allocator.free(rids);
        break :blk @intCast(rids.len);
    };
    const envelope = try buildResponsesEnvelope(
        stream.io,
        allocator,
        esc_resp_id,
        esc_model,
        status_str,
        out_buf.items,
        result.prompt_tokens,
        result.completion_tokens,
        result.cached_tokens,
        reasoning_tok_count,
        should_store,
        prev_id,
        is_incomplete,
        is_completed_status,
        response_echo,
        result.prefill_ns,
        result.decode_ns,
        tokenize_ns,
        result.completion_tokens,
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
    lm: *LoadedModel,
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

        // Reset sequence numbering per response per the OpenAI spec.
        // (handleResponses owns its own seq_num, fresh each call.)
        var seq: u64 = 0;
        _ = &seq;

        stream.ws_mode = &bridge;
        defer stream.ws_mode = null;

        bridge.reset();
        handleResponses(allocator, stream, body, lm) catch |err| {
            log.warn("WS handleResponses error: {s}\n", .{@errorName(err)});
            // Best-effort error frame; connection may already be torn.
            wsSendErrorTurn(allocator, &ws_conn, 500, "server_error", @errorName(err)) catch {};
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
    /// Iteration 1: timings extension. The Responses spec doesn't model
    /// per-stage timings, so we attach a sibling `timings` block at the
    /// envelope root with the same shape as /v1/chat/completions. Pass
    /// zeroes to omit the field (legacy callers stay shape-compatible).
    prefill_ns: u64,
    decode_ns: u64,
    tokenize_ns: u64,
    completion_tokens_for_timings: u32,
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

    // Iteration 1 timings extension. Reuses the chat-completions
    // formatter so any future field added there appears on Responses too
    // without a second touch point.
    const timings_obj = try formatTimingsObject(allocator, input_tokens, cached_input_tokens, completion_tokens_for_timings, prefill_ns, decode_ns, tokenize_ns);
    defer allocator.free(timings_obj);
    if (timings_obj.len > 0) {
        try buf.appendSlice(allocator, ",\"timings\":");
        try buf.appendSlice(allocator, timings_obj);
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

test "Conn.peerClosed: alive socket returns false, closed peer returns true" {
    // Create a connected socket pair (AF_UNIX SOCK_STREAM via socketpair).
    var sv: [2]std.posix.fd_t = undefined;
    const AF_UNIX: c_uint = 1;
    const SOCK_STREAM: c_uint = 1;
    const rc = std.c.socketpair(AF_UNIX, SOCK_STREAM, 0, &sv);
    try testing.expect(rc == 0);

    const server_fd = sv[0];
    const client_fd = sv[1];

    // Build a Conn that wraps the server-side fd. We only need
    // `stream.socket.handle` for peerClosed, so the Reader/Writer state
    // can stay zeroed.
    var conn: Conn = undefined;
    conn.stream = .{ .socket = .{ .handle = server_fd, .address = undefined } };

    // Healthy connection: no data pending, no FIN → peerClosed returns false.
    try testing.expect(!conn.peerClosed());

    // Client closes its side → server should observe FIN/HUP.
    _ = std.c.close(client_fd);

    // socketpair() returns connected sockets in the kernel; close-of-peer
    // is observable immediately on the other side without delay.
    const closed = conn.peerClosed();
    _ = std.c.close(server_fd);
    try testing.expect(closed);
}

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

test "safeAutoContext leaves headroom below the memory ceiling and rounds to 1024" {
    // The raw ceiling is the largest context that FITS. Running at exactly that
    // leaves nothing for the prefix cache to grow into, a second model, or
    // another app — so admit only `auto_ctx_safety_pct` of it.
    try testing.expectEqual(@as(u32, 79872), safeAutoContext(94729)); // 85% = 80519 -> 1024-floor
    try testing.expectEqual(@as(u32, 27648), safeAutoContext(32768)); // 85% = 27852 -> 1024-floor
    // Never rounds down to zero on a tiny ceiling.
    try testing.expect(safeAutoContext(1000) > 0);
    try testing.expect(safeAutoContext(1) > 0);
    // Monotonic, and always strictly inside the ceiling for real-sized contexts.
    var prev: u32 = 0;
    for ([_]u32{ 4096, 8192, 16384, 32768, 65536, 94729, 262144 }) |raw| {
        const got = safeAutoContext(raw);
        try testing.expect(got <= raw);
        try testing.expect(got >= prev);
        prev = got;
    }
}

test "autoContextFor: the safety margin applies to MEMORY, never to the model's own max" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;
    server_config.max_context_size = 0;

    // A tiny model whose `max_position_embeddings` is far below anything memory
    // could constrain: it must get its FULL declared context, un-margined.
    // (Regression: applying 85% AFTER the model-max clamp shaved 15% off a
    // 131,072-token checkpoint that fits in RAM with room to spare.)
    var small = model_mod.ModelConfig{};
    small.max_position_embeddings = 4096;
    small.num_attention_heads = 8;
    small.num_hidden_layers = 4;
    small.num_key_value_heads = 2;
    small.head_dim = 64;
    small.hidden_size = 512;
    small.intermediate_size = 1024;
    try testing.expectEqual(@as(u32, 4096), autoContextFor(&small));

    // No declared max: fall back to the margined memory ceiling.
    var unbounded = small;
    unbounded.max_position_embeddings = 0;
    const got = autoContextFor(&unbounded);
    try testing.expect(got > 0);
    try testing.expectEqual(safeAutoContext(computeMemoryContext(&unbounded)), got);
}

test "getEffectiveContextLength returns the PINNED value, not a fresh memory reading" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;
    server_config.max_context_size = 0; // auto

    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 131072;
    config.num_attention_heads = 8;
    config.num_hidden_layers = 42;
    config.num_key_value_heads = 2;
    config.head_dim = 256;

    // Once pinned, the advertised context must NOT move — clients budget
    // against it, and a fresh memory reading drifts with system load.
    config.pinned_context = 12345;
    try testing.expectEqual(@as(u32, 12345), getEffectiveContextLength(&config));
    try testing.expectEqual(@as(u32, 12345), getEffectiveContextLength(&config));

    // ...but an explicit --ctx-size still wins.
    server_config.max_context_size = 4096;
    try testing.expectEqual(@as(u32, 4096), getEffectiveContextLength(&config));
}

test "pinAutoContext is idempotent and a no-op under --ctx-size" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;

    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 131072;
    config.num_attention_heads = 8;
    config.num_hidden_layers = 42;
    config.num_key_value_heads = 2;
    config.head_dim = 256;

    server_config.max_context_size = 0;
    const first = pinAutoContext(&config);
    try testing.expect(first > 0);
    try testing.expectEqual(first, config.pinned_context);
    // Second call must not re-read memory and shift the value.
    try testing.expectEqual(first, pinAutoContext(&config));

    // Explicit --ctx-size: never overwrite, always report the override.
    var manual = model_mod.ModelConfig{};
    manual.max_position_embeddings = 131072;
    server_config.max_context_size = 8192;
    try testing.expectEqual(@as(u32, 8192), pinAutoContext(&manual));
    try testing.expectEqual(@as(u32, 0), manual.pinned_context);
}

test "clampMaxTokens honors the effective context (auto-pinned, not just --ctx-size)" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;
    server_config.max_context_size = 0; // auto: the old code never clamped here

    // 4096-token window, 4000-token prompt -> only 96 tokens of generation left.
    try testing.expectEqual(@as(u32, 96), clampMaxTokens(8192, 4000, 4096));
    // Comfortable budget is passed through untouched.
    try testing.expectEqual(@as(u32, 8192), clampMaxTokens(8192, 100, 65536));
    // Prompt at/over the window still yields at least one token.
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(8192, 4096, 4096));
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(8192, 9000, 4096));
    // ctx 0 (unknown) = no limit.
    try testing.expectEqual(@as(u32, 8192), clampMaxTokens(8192, 4000, 0));
}

test "getEffectiveContextLength uses ctx_size override" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;

    server_config.max_context_size = 4096;
    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 32768;
    try testing.expectEqual(@as(u32, 4096), getEffectiveContextLength(&config));
}

test "getEffectiveContextLength computes safe default from GPU memory" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;

    server_config.max_context_size = 0;
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
    server_config.max_context_size = 32768;
    try testing.expectEqual(@as(u32, 32768), getEffectiveContextLength(&config));
}

test "safeContextForBudget reserves the hot-cache budget (2026-06-19 OOM regression)" {
    const GB: u64 = 1 << 30;
    // qwen3_5_moe footprint that OOM'd a 16 GB Mac: 32 layers, 4 kv heads, head_dim 256.
    const kv_per_tok: u64 = 32 * 2 * 4 * 256 * 2; // 131072
    const work_per_tok: u64 = 8 * 9216 * 2; //        147456
    const per_tok: u64 = kv_per_tok + work_per_tok;

    // Mid-session: ~6 GB resident (model + a partly-filled hot cache), 2 GB cache cap.
    const ws: u64 = 12 * GB; // ~Metal recommended working set on a 16 GB Mac
    const active: u64 = 6 * GB;
    const cache: u64 = 2 * GB;

    const with_reserve = safeContextForBudget(ws, active, cache, per_tok, 0);
    const without_reserve = safeContextForBudget(ws, active, 0, per_tok, 0);

    // The whole point of the fix: subtracting the hot-cache budget shrinks the
    // reported context so it still fits once the cache fills (it didn't before
    // — auto-ctx was computed against an empty cache and later collided with it).
    try testing.expect(with_reserve < without_reserve);
    // Still usable, never floored to nothing.
    try testing.expect(with_reserve > 1024);
}

test "physicalMemoryCeiling caps the static GPU max by real free RAM (#64 docker OOM)" {
    const GB: u64 = 1 << 30;
    const static_max: u64 = 115 * GB; // Metal max_recommended_working_set_size, 128 GB Mac

    // Idle: MLX holds 26 GB (active) + 5 GB (reclaimable cache), ~84 GB free.
    // 31 + 84 = 115 ⇒ the static device max still binds — NO regression when the
    // machine is uncontended (auto-context stays generous).
    try testing.expectEqual(static_max, physicalMemoryCeiling(static_max, 31 * GB, 84 * GB));

    // #64: a docker-compose stack (firecrawl/rabbitmq/postgres/playwright) holds
    // ~45 GB. MLX holds 26 GB, only ~35 GB is physically free. The static max is
    // BLIND to docker, but the ceiling must collapse to 26 + 35 = 61 GB — otherwise
    // a large MoE prefill is admitted and Metal throws an uncatchable OOM.
    try testing.expectEqual(@as(u64, 61 * GB), physicalMemoryCeiling(static_max, 26 * GB, 35 * GB));

    // Saturating add: a bogus/huge footprint reading can never wrap to a tiny ceiling.
    try testing.expectEqual(static_max, physicalMemoryCeiling(static_max, std.math.maxInt(u64), GB));
}

test "auto-context tightens under external memory pressure vs the pre-fix static ceiling (#64)" {
    const GB: u64 = 1 << 30;
    // qwen3_5_moe 35B-A3B footprint (per-token KV + working envelope).
    const per_tok: u64 = 328 * 1024;
    const static_max: u64 = 115 * GB;
    const active: u64 = 26 * GB;
    const mlx_cache: u64 = 4 * GB;
    const hot_cache_reserve: u64 = 2 * GB;
    const max_pos: u32 = 262144;

    // Pre-fix behaviour: budget straight against the static device max, blind to
    // free RAM — the exact code path that OOM'd #64.
    const prefix_ctx = safeContextForBudget(static_max, active, hot_cache_reserve, per_tok, max_pos);

    // Post-fix, idle (~84 GB free): ceiling ≈ static max ⇒ context essentially unchanged.
    const idle_ceiling = physicalMemoryCeiling(static_max, active + mlx_cache, 84 * GB);
    const idle_ctx = safeContextForBudget(idle_ceiling, active, hot_cache_reserve, per_tok, max_pos);
    try testing.expect(idle_ctx >= prefix_ctx * 9 / 10); // within 10% of the old generous value

    // Post-fix, docker holds ~45 GB (~35 GB free): ceiling collapses, so the safe
    // context is far smaller than the pre-fix static-max computation would allow.
    const busy_ceiling = physicalMemoryCeiling(static_max, active + mlx_cache, 35 * GB);
    const busy_ctx = safeContextForBudget(busy_ceiling, active, hot_cache_reserve, per_tok, max_pos);
    try testing.expect(busy_ctx < prefix_ctx / 2);
    try testing.expect(busy_ctx > 1024); // still usable, never floored to nothing
}

test "safeContextForBudget floors when memory is exhausted and caps at max_pos" {
    const GB: u64 = 1 << 30;
    const per_tok: u64 = 278528;
    // working set already below what's resident → no room → minimum floor.
    try testing.expectEqual(@as(u32, 1024), safeContextForBudget(4 * GB, 5 * GB, 0, per_tok, 0));
    // active + reserve exceeds the ceiling → floor.
    try testing.expectEqual(@as(u32, 1024), safeContextForBudget(8 * GB, 6 * GB, 4 * GB, per_tok, 0));
    // Plenty of memory, but capped at the model's max position embeddings.
    try testing.expectEqual(@as(u32, 4096), safeContextForBudget(256 * GB, 0, 0, per_tok, 4096));
    // Degenerate per_tok never divides by zero.
    try testing.expectEqual(@as(u32, 1024), safeContextForBudget(8 * GB, 0, 0, 0, 0));
}

test "omittedMaxTokensDefault: context-bound when ctx is known, finite fallback otherwise" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;

    // OpenAI semantics: omitted max_tokens = generate until EOS bounded by the
    // context window. The old fixed 256 default broke agent clients (pi) whose
    // thinking-enabled turns hit `length` mid-reasoning on EVERY request.
    server_config.max_context_size = 32768;
    const sentinel = omittedMaxTokensDefault(32768);
    // Big enough that clampMaxTokens (the downstream bound) always wins…
    try testing.expect(sentinel > 32768);
    // …and the composition resolves to exactly the remaining context.
    try testing.expectEqual(@as(u32, 32768 - 1500), clampMaxTokens(sentinel, 1500, 32768));

    // ctx unknown (max_context_size=0): clampMaxTokens won't bound anything,
    // so the default itself must stay finite.
    server_config.max_context_size = 0;
    try testing.expectEqual(@as(u32, 4096), omittedMaxTokensDefault(0));
}

test "resolveRequestMaxTokens: absent / 0 / negative / non-int → auto; positive → value" {
    const auto: u32 = 12345;
    // Absent → auto (the omitted-default path the app's "Auto" setting rides).
    try testing.expectEqual(auto, resolveRequestMaxTokens(null, auto));
    // 0 and negatives are "auto" too — a client must not be able to request a
    // 0-token (generate-nothing) response by selecting Auto.
    try testing.expectEqual(auto, resolveRequestMaxTokens(.{ .integer = 0 }, auto));
    try testing.expectEqual(auto, resolveRequestMaxTokens(.{ .integer = -5 }, auto));
    // Wrong type (e.g. a string) → auto rather than crashing.
    try testing.expectEqual(auto, resolveRequestMaxTokens(.{ .string = "999" }, auto));
    // A real positive cap is honored verbatim.
    try testing.expectEqual(@as(u32, 4096), resolveRequestMaxTokens(.{ .integer = 4096 }, auto));
}

test "StreamHeartbeat: a write resets the deadline, silence expires it" {
    var hb = StreamHeartbeat{ .last_write_ms = 1_000, .interval_ms = 5_000 };

    // Silent, but not yet for a full interval.
    try testing.expect(!hb.due(1_000));
    try testing.expect(!hb.due(5_999));
    // Exactly the interval is due (the client's clock is not ours to trust).
    try testing.expect(hb.due(6_000));
    try testing.expect(hb.due(60_000));

    // Any byte written to the socket pushes the deadline out.
    hb.noteWrite(60_000);
    try testing.expect(!hb.due(60_001));
    try testing.expect(hb.due(65_000));
}

test "StreamHeartbeat: buffered tokens do NOT reset the deadline" {
    // The regression this guards: the old code beat only when NO token was
    // available (`.idle`). A tool call buffers tokens continuously while
    // writing nothing, so token arrival must not be mistaken for liveness —
    // only `noteWrite` (bytes on the socket) counts.
    var hb = StreamHeartbeat{ .last_write_ms = 0, .interval_ms = 5_000 };
    var now: i64 = 0;
    // 300 tokens arrive over 30s. Not one is written out (buffered tool call).
    var i: usize = 0;
    while (i < 300) : (i += 1) now += 100;
    try testing.expectEqual(@as(i64, 30_000), now);
    try testing.expect(hb.due(now));
}

test "clampMaxTokens no limit when ctx_size=0" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;
    server_config.max_context_size = 0;

    try testing.expectEqual(@as(u32, 1000), clampMaxTokens(1000, 500, 0));
}

test "clampMaxTokens clamps when would exceed" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;
    server_config.max_context_size = 4096;

    // prompt=3000, max_tokens=2000 → clamp to 1096
    try testing.expectEqual(@as(u32, 1096), clampMaxTokens(2000, 3000, 4096));
}

test "clampMaxTokens no clamp when fits" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;
    server_config.max_context_size = 4096;

    // prompt=100, max_tokens=200 → fits, no clamp
    try testing.expectEqual(@as(u32, 200), clampMaxTokens(200, 100, 4096));
}

test "clampMaxTokens at boundary" {
    const original = server_config.max_context_size;
    defer server_config.max_context_size = original;
    server_config.max_context_size = 4096;

    // prompt=4096 → only 1 token allowed
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(100, 4096, 4096));
    // prompt=4095 → only 1 token remaining
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(100, 4095, 4096));
}

test "getTimeoutNs computes correctly" {
    const original = server_config.request_timeout_sec;
    defer server_config.request_timeout_sec = original;

    server_config.request_timeout_sec = 300;
    try testing.expectEqual(@as(u64, 300_000_000_000), getTimeoutNs());

    server_config.request_timeout_sec = 0;
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

test "insertMultimodalTokens lays out image block then audio block at the user turn" {
    // Gemma 4 12B unified: the image placeholder block MUST precede the audio
    // block so a single splice scatters the [vision ; audio] embedding in order.
    var config = model_mod.ModelConfig{};
    config.user_turn_marker_ids[0] = 105;
    config.user_turn_marker_ids[1] = 2364;
    config.user_turn_marker_ids[2] = 107;
    config.user_turn_marker_len = 3;
    config.boi_token_id = 200; // BOI
    config.eoi_token_id = 201; // EOI
    config.boa_token_id = 300; // BOA
    config.eoa_token_id = 301; // EOA

    const prompt = [_]u32{ 2, 500, 105, 2364, 107, 900, 901 };
    // image_token=999 ×2, audio_token=888 ×3.
    const out = try insertMultimodalTokens(testing.allocator, &prompt, 999, 2, 888, 3, &config);
    defer testing.allocator.free(out);

    // Inserted after marker (position 5): [BOI 999 999 EOI][BOA 888 888 888 EOA].
    const expected = [_]u32{
        2,   500, 105, 2364, 107,
        200, 999, 999, 201, // image block
        300, 888, 888, 888, 301, // audio block
        900, 901,
    };
    try testing.expectEqualSlices(u32, &expected, out);
}

test "insertMultimodalTokens handles audio-only and image-only" {
    var config = model_mod.ModelConfig{};
    config.user_turn_marker_ids[0] = 105;
    config.user_turn_marker_len = 1;
    config.boi_token_id = 200;
    config.eoi_token_id = 201;
    config.boa_token_id = 300;
    config.eoa_token_id = 301;
    const prompt = [_]u32{ 1, 105, 7 };

    // Audio only (n_image=0) → just the audio block.
    const ao = try insertMultimodalTokens(testing.allocator, &prompt, 999, 0, 888, 2, &config);
    defer testing.allocator.free(ao);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 105, 300, 888, 888, 301, 7 }, ao);

    // Image only (n_audio=0) → just the image block.
    const io = try insertMultimodalTokens(testing.allocator, &prompt, 999, 2, 888, 0, &config);
    defer testing.allocator.free(io);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 105, 200, 999, 999, 201, 7 }, io);

    // Neither → unchanged.
    const none = try insertMultimodalTokens(testing.allocator, &prompt, 999, 0, 888, 0, &config);
    defer testing.allocator.free(none);
    try testing.expectEqualSlices(u32, &prompt, none);
}

test "parseAudioContent decodes base64 float32 PCM and rejects bad lengths" {
    // 2 float32 samples = 8 bytes. base64 of 8 zero bytes = "AAAAAAAAAAA=".
    const eight_zeros = "AAAAAAAAAAA=";
    const aud = parseAudioContent(testing.allocator, eight_zeros) orelse return error.TestUnexpectedNull;
    defer testing.allocator.free(aud.samples);
    try testing.expectEqual(@as(usize, 8), aud.samples.len);

    // A data-URL prefix is tolerated.
    const with_prefix = "data:audio/x-mlx-pcm;base64," ++ eight_zeros;
    const aud2 = parseAudioContent(testing.allocator, with_prefix) orelse return error.TestUnexpectedNull;
    defer testing.allocator.free(aud2.samples);
    try testing.expectEqual(@as(usize, 8), aud2.samples.len);

    // 6 decoded bytes is not a whole number of float32s → rejected.
    try testing.expect(parseAudioContent(testing.allocator, "AAAAAAAA") == null);
}

// --- /props payload regression ------------------------------------------------
//
// The `chat_template` field used to be emitted by /props for llama.cpp
// server-compat clients. Our own Swift app never read it, and stock chat
// templates are 10–100 KB of Jinja — that's wasted bandwidth on every poll.
// `renderPropsBody` is the pure body-builder behind `handleProps`; these
// tests pin the schema so a future "let's add it back" can't slip through
// without flipping these assertions intentionally.

test "renderPropsBody omits chat_template" {
    var config = model_mod.ModelConfig{};
    config.vocab_size = 32000;
    config.hidden_size = 4096;
    config.num_hidden_layers = 32;
    config.num_attention_heads = 32;
    config.num_key_value_heads = 8;
    config.head_dim = 128;
    config.quant_bits = 4;
    config.quant_group_size = 64;
    config.max_position_embeddings = 8192;
    config.model_type = "gemma4";

    const body = try renderPropsBody(testing.allocator, &config, "4096", 1234, 5678, 9_000_000_000, 16384);
    defer testing.allocator.free(body);

    try testing.expect(std.mem.indexOf(u8, body, "\"chat_template\"") == null);
}

test "renderPropsBody keeps fields the Swift app + integration tests rely on" {
    var config = model_mod.ModelConfig{};
    config.model_type = "gemma4";
    config.vocab_size = 32000;
    config.hidden_size = 4096;
    config.num_hidden_layers = 32;
    config.num_attention_heads = 32;
    config.num_key_value_heads = 8;
    config.head_dim = 128;
    config.quant_bits = 4;
    config.quant_group_size = 64;
    config.max_position_embeddings = 8192;

    const body = try renderPropsBody(testing.allocator, &config, "4096", 1234, 5678, 9_000_000_000, 16384);
    defer testing.allocator.free(body);

    // Hit every field a known consumer reads.
    try testing.expect(std.mem.indexOf(u8, body, "\"n_ctx\":4096") != null);          // integration_test.sh
    try testing.expect(std.mem.indexOf(u8, body, "\"total_slots\":1") != null);       // integration_test.sh
    try testing.expect(std.mem.indexOf(u8, body, "\"model_info\"") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"memory\"") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"active_bytes\":1234") != null);   // Swift fetchProps
    try testing.expect(std.mem.indexOf(u8, body, "\"peak_bytes\":5678") != null);     // Swift fetchProps
    try testing.expect(std.mem.indexOf(u8, body, "\"available_bytes\":9000000000") != null); // Swift fetchProps (Free RAM line)
    try testing.expect(std.mem.indexOf(u8, body, "\"max_safe_context\":16384") != null); // Swift fetchProps
}


test "llama cache default keeps shared prefixes warm" {
    // With the legacy default of 1, every llama.cpp request evicted the
    // single KV session — even two SEQUENTIAL requests sharing an 8 KB
    // prefix reported cached_tokens=0 (caught live by llmprobe
    // cache-hit-reported on the E4B GGUF, 2026-06-10). 4 sessions keep
    // interleaved agent roots warm; sessions are created lazily so idle
    // slots cost nothing.
    try testing.expect(llama_cache_entries >= 4);
}

test "prefix cache default capacity covers interleaved agent flows" {
    // Claude Code-style clients interleave several conversation roots (main
    // thread, subagents, title generation). With capacity 1, every
    // interleaved request evicted the long system-prompt prefix and forced a
    // full re-prefill per turn. The byte budget (prefix_cache_mem_bytes)
    // still bounds memory.
    try testing.expect(prefix_cache_capacity >= 4);
    try testing.expect(prefix_cache_mem_bytes > 0);
}

test "parseToolCallsForRequest coerces args to the schema (server-side chokepoint wiring)" {
    // The chokepoint every HTTP surface routes through must actually invoke the
    // schema coercion — not just parse. This pins the WIRING (a mis-wire that
    // parsed but skipped coerce would reintroduce the string-"False" bug on every
    // surface at once). Uses the exact Hermes-XML `replace_all=False` shape from
    // the live Claude Code failure.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;
    const text = "<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n" ++
        "<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCallsForRequest(allocator, text, tools)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const ra = parsed.value.object.get("replace_all").?;
    try std.testing.expect(ra == .bool); // coerced from the STRING "False"
    try std.testing.expectEqual(false, ra.bool);
    // Null tools_json path must still parse (no coercion, no crash).
    const calls2 = try parseToolCallsForRequest(allocator, text, null);
    defer if (calls2) |cs| {
        for (cs) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(cs);
    };
    try std.testing.expect(calls2 != null);
}

test "parseToolCallsForRequest: --no-tool-autocorrect leaves args verbatim" {
    // The escape hatch: with g_tool_autocorrect off, the schema coercion is
    // skipped and the model's value passes through EXACTLY (here the STRING
    // "False", which is what a strict client would then reject — the user's
    // choice). Parse-repair + valid-JSON safety net still run, so args stay
    // well-formed JSON regardless.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;
    const text = "<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n" ++
        "<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>";

    g_tool_autocorrect = false;
    defer g_tool_autocorrect = true; // restore the default for other tests
    const calls = (try parseToolCallsForRequest(allocator, text, tools)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const ra = parsed.value.object.get("replace_all").?;
    try std.testing.expect(ra == .string); // NOT coerced — verbatim
    try std.testing.expectEqualStrings("False", ra.string);
}

test "parseToolCallsForRequest: truncated DATA object is not promoted to a tool call (George Washington class)" {
    // Live pi capture 2026-07-13 (Qwen3.6-35B-A3B distilled): generation hit
    // max_tokens midway through a presidents data script. The raw-JSON fallback
    // found the first balanced object — {"name": "George Washington", "num": 1,
    // …} — and the flat-shape synthesis promoted it to a TOOL CALL named
    // "George Washington" (args = every key but "name"). pi answered "Tool
    // George Washington not found", the model retried the identical mega-write,
    // and the session burned two full 16K-token turns making zero progress.
    // A HEURISTICALLY inferred call (no tag syntax — the model never said
    // "tool call") must carry a name the request actually declared; otherwise
    // the text stays visible and finish_reason="length" reaches the client
    // untouched, so its truncation recovery fires instead of a bogus tool loop.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
    ;
    const text = "Now let me build a generator script that creates all 46 president pages with real historical data:\n\n" ++
        "presidents = [\n" ++
        "  {\"name\": \"George Washington\", \"num\": 1, \"party\": \"None (Federalist-leaning)\", \"term\": \"1789\u{2013}1797\", \"vice\": \"John Adams\"},\n" ++
        "  {\"name\": \"John Adams\", \"num\": 2, \"party\": \"Federalist\",";
    const calls = try parseToolCallsForRequest(allocator, text, tools);
    defer if (calls) |cs| {
        for (cs) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(cs);
    };
    try std.testing.expect(calls == null);
}

test "parseToolCallsForRequest: raw-JSON call with a DECLARED name still parses" {
    // The counterweight to the George Washington guard: models without a
    // trained tool format (Gemma 3) emit fenced raw-JSON calls, and those must
    // keep working when the name matches a declared tool.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]
    ;
    const text = "```json\n{\"name\": \"write\", \"arguments\": {\"path\": \"a.txt\", \"content\": \"hi\"}}\n```";
    const calls = (try parseToolCallsForRequest(allocator, text, tools)) orelse
        return error.ExpectedToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try std.testing.expectEqualStrings("write", calls[0].name);
}

test "parseToolCallsForRequest: tag-format call with an UNDECLARED name is kept" {
    // An EXPLICIT tag-format call (<tool_call>…) to a name the request never
    // declared still goes to the client — "tool not found" is model-visible
    // feedback the model can correct from. Only HEURISTIC raw-JSON inference
    // gets schema-name validation; this pins the filter against over-reach.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
    ;
    const text = "<tool_call>{\"name\":\"searchWeb\",\"arguments\":{\"q\":\"zig\"}}</tool_call>";
    const calls = (try parseToolCallsForRequest(allocator, text, tools)) orelse
        return error.ExpectedToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try std.testing.expectEqualStrings("searchWeb", calls[0].name);
}

test "parseToolCallsForRequest: coercion fires across think on/off × qwen/gemma" {
    // The tool-call parse strips the leading think block BEFORE parsing args, and
    // the two families strip DIFFERENTLY (Qwen `<think>…</think>`, Gemma
    // `<|channel>thought…<channel|>`). Pin that a preceding reasoning block never
    // blocks the schema coercion, for BOTH families, WITH and WITHOUT thinking.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;
    const gtools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;

    const Case = struct { name: []const u8, text: []const u8, tools: []const u8 };
    const cases = [_]Case{
        // Qwen3.6-MoE thinking OFF — Hermes XML, no reasoning block.
        .{ .name = "qwen think-off", .tools = tools, .text = "<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>" },
        // Qwen3.6-MoE thinking ON — a `<think>…</think>` block precedes the call.
        .{ .name = "qwen think-on", .tools = tools, .text = "<think>\nThe user wants replace_all disabled, so I'll pass it false.\n</think>\n<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>" },
        // Gemma thinking OFF — custom `call:` form, no channel block.
        .{ .name = "gemma think-off", .tools = gtools, .text = "<|tool_call>call:Edit{file_path:<|\"|>/tmp/x<|\"|>,replace_all:False}<tool_call|>" },
        // Gemma thinking ON — a `<|channel>thought…<channel|>` block precedes.
        .{ .name = "gemma think-on", .tools = gtools, .text = "<|channel>thought\nI should disable replace_all for a single edit.<channel|>\n<|tool_call>call:Edit{file_path:<|\"|>/tmp/x<|\"|>,replace_all:False}<tool_call|>" },
    };

    for (cases) |c| {
        const calls = (try parseToolCallsForRequest(allocator, c.text, c.tools)) orelse {
            std.debug.print("\n[{s}] no tool call parsed\n", .{c.name});
            return error.NoToolCall;
        };
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try std.testing.expectEqualStrings("Edit", calls[0].name);
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        const ra = parsed.value.object.get("replace_all") orelse {
            std.debug.print("\n[{s}] replace_all missing in {s}\n", .{ c.name, calls[0].arguments });
            return error.ArgMissing;
        };
        if (ra != .bool or ra.bool != false) {
            std.debug.print("\n[{s}] replace_all NOT coerced to bool false: {s}\n", .{ c.name, calls[0].arguments });
            return error.NotCoerced;
        }
    }
}

test "toolCallFinishReason preserves truncation over parsed tool calls" {
    // A truncated generation ("length": max_tokens OR request timeout) whose
    // cut-off text still salvaged a partial tool call must KEEP reporting
    // "length" — clients key their truncation recovery (chunk-and-retry
    // nudges) on it. Reporting "tool_calls" hides the cut: the client
    // executes a half-argument call and tells the model IT made a mistake.
    // Live capture 2026-07-03: Qwen3.6-27B 33KB writeFile guillotined by the
    // 300s default timeout arrived as {"path":...} with finish "tool_calls".
    try std.testing.expectEqualStrings("length", toolCallFinishReason("length"));
    // Everything else stays the normal tool-call finish.
    try std.testing.expectEqualStrings("tool_calls", toolCallFinishReason("stop"));
    try std.testing.expectEqualStrings("tool_calls", toolCallFinishReason("tool_calls"));
    try std.testing.expectEqualStrings("tool_calls", toolCallFinishReason("client_disconnect"));
}

test "resolveSamplingDefault: request > CLI > generation_config > fallback" {
    // Request value always wins.
    try std.testing.expectEqual(@as(f32, 0.2), resolveSamplingDefault(f32, 0.2, 0.7, 1.0, 1.0));
    // Omitted in request -> CLI launch flag (the app passes Settings here).
    try std.testing.expectEqual(@as(f32, 0.7), resolveSamplingDefault(f32, null, 0.7, 1.0, 1.0));
    // No CLI flag -> the model's generation_config.json recommendation.
    try std.testing.expectEqual(@as(u32, 20), resolveSamplingDefault(u32, null, null, 20, 0));
    // Nothing anywhere -> hardcoded fallback (pre-existing behavior).
    try std.testing.expectEqual(@as(f32, 1.0), resolveSamplingDefault(f32, null, null, null, 1.0));
    // Explicit request 0 (greedy) must not be treated as omitted.
    try std.testing.expectEqual(@as(f32, 0.0), resolveSamplingDefault(f32, 0.0, 0.7, 1.0, 1.0));
}

test "optSamplingRecJson emits number when present, null when absent" {
    const a = std.testing.allocator;

    // Present -> bare JSON number (no quotes), so the Swift side decodes it
    // straight to Double/Int.
    const top_k = try optSamplingRecJson(a, u32, 20);
    defer a.free(top_k);
    try std.testing.expectEqualStrings("20", top_k);

    const top_p = try optSamplingRecJson(a, f32, 0.95);
    defer a.free(top_p);
    try std.testing.expectEqualStrings("0.95", top_p);

    // Absent -> JSON null literal so the model-author recommendation reads as
    // "no opinion" rather than a spurious 0.
    const none = try optSamplingRecJson(a, u32, null);
    defer a.free(none);
    try std.testing.expectEqualStrings("null", none);
}

test "textGenRejectReason: media + encoder models rejected, chat models pass (SIGSEGV class 2026-07-06)" {
    // Chat-capable targets pass — resident MLX LM, embedded engines, and
    // unloaded stubs with a chat arch hint (or none: benefit of the doubt
    // until load).
    try std.testing.expect(textGenRejectReason(.{ .arch_hint = "gemma4" }) == null);
    try std.testing.expect(textGenRejectReason(.{ .arch_hint = "gguf" }) == null);
    try std.testing.expect(textGenRejectReason(.{}) == null);

    // The live crash shape: a resident image model — engine slot set, no
    // transformer. Must reject, never reach prefill.
    {
        const r = textGenRejectReason(.{ .has_image_engine = true, .has_text_lm = false });
        try std.testing.expect(r != null);
        try std.testing.expect(std.mem.indexOf(u8, r.?, "/v1/images/generations") != null);
    }
    // Pre-load detection via the discovery arch hint — no engine resident yet.
    {
        const r = textGenRejectReason(.{ .arch_hint = "flux2-klein-4b" });
        try std.testing.expect(std.mem.indexOf(u8, r.?, "/v1/images/generations") != null);
    }
    try std.testing.expect(std.mem.indexOf(u8, textGenRejectReason(.{ .arch_hint = "hunyuan3d_2_1" }).?, "/v1/3d/generations") != null);
    try std.testing.expect(std.mem.indexOf(u8, textGenRejectReason(.{ .arch_hint = "qwen3_tts" }).?, "/v1/audio/speech") != null);
    try std.testing.expect(std.mem.indexOf(u8, textGenRejectReason(.{ .arch_hint = "acestep" }).?, "music-generations") != null);
    try std.testing.expect(std.mem.indexOf(u8, textGenRejectReason(.{ .arch_hint = "AudioVideo" }).?, "/v1/video/generations") != null);
    try std.testing.expect(std.mem.indexOf(u8, textGenRejectReason(.{ .has_mesh_engine = true }).?, "/v1/3d/generations") != null);

    // Encoder-only keeps its dedicated message (pre-existing guard, now
    // routed through the same decision).
    try std.testing.expect(std.mem.indexOf(u8, textGenRejectReason(.{ .is_encoder_only = true }).?, "/v1/embeddings") != null);

    // Catch-all: a READY entry with no LM and no engines (future modality)
    // still rejects instead of crashing.
    try std.testing.expect(textGenRejectReason(.{ .has_text_lm = false }) != null);
}

test "isTextGenRoute covers exactly the guarded surfaces" {
    try std.testing.expect(isTextGenRoute("POST", "/v1/chat/completions"));
    try std.testing.expect(isTextGenRoute("POST", "/v1/completions"));
    try std.testing.expect(isTextGenRoute("POST", "/v1/messages"));
    try std.testing.expect(isTextGenRoute("POST", "/v1/responses"));
    try std.testing.expect(isTextGenRoute("POST", "/api/chat"));
    try std.testing.expect(isTextGenRoute("POST", "/api/generate"));
    try std.testing.expect(isTextGenRoute("GET", "/v1/responses")); // WS upgrade
    // Media + embedding routes must NOT be gated — they serve these models.
    try std.testing.expect(!isTextGenRoute("POST", "/v1/images/generations"));
    try std.testing.expect(!isTextGenRoute("POST", "/v1/audio/speech"));
    try std.testing.expect(!isTextGenRoute("POST", "/v1/embeddings"));
    try std.testing.expect(!isTextGenRoute("POST", "/api/embed"));
    try std.testing.expect(!isTextGenRoute("GET", "/v1/models"));
}

test "readyCapsJson: every resident media engine surfaces its capability (mesh -> 3d)" {
    const a = std.testing.allocator;

    // A ready 3D shape model (no chat template, no encoder) must advertise
    // "3d" — the stub path already did; the READY path shipped without the
    // mesh arm and rendered [] (live 2026-07-04, test_3d_gen.sh check 1).
    var mesh = try readyCapsJson(a, .{ .has_mesh_engine = true });
    defer mesh.deinit(a);
    try std.testing.expectEqualStrings("[\"3d\"]", mesh.items);

    // The other three engine slots keep their existing capability names.
    var img = try readyCapsJson(a, .{ .has_image_engine = true });
    defer img.deinit(a);
    try std.testing.expectEqualStrings("[\"image\"]", img.items);
    var aud = try readyCapsJson(a, .{ .has_audio_engine = true });
    defer aud.deinit(a);
    try std.testing.expectEqualStrings("[\"audio\"]", aud.items);
    // A music-backend audio engine advertises "music" ADDITIVELY beside "audio".
    var mus = try readyCapsJson(a, .{ .has_audio_engine = true, .has_music_backend = true });
    defer mus.deinit(a);
    try std.testing.expectEqualStrings("[\"audio\",\"music\"]", mus.items);
    var vid = try readyCapsJson(a, .{ .has_video_engine = true });
    defer vid.deinit(a);
    try std.testing.expectEqualStrings("[\"video\"]", vid.items);

    // Chat-class flags are unaffected by the media arms.
    var chat = try readyCapsJson(a, .{ .has_chat = true, .has_reasoning = true });
    defer chat.deinit(a);
    try std.testing.expectEqualStrings(
        "[\"chat\",\"tool_use\",\"streaming\",\"reasoning\",\"json_schema\"]",
        chat.items,
    );
}

test "prefillMemoryNeeded: quantized KV is billed at its real width, not fp16" {
    const t = std.testing;
    transformer_mod.fused256_override = false;
    defer transformer_mod.fused256_override = null;
    // 26B-A4B shape at 100K ctx, chunk 512 (what the bounded chunk picks there).
    // fp16: kv 30x2x100000x8x256x2 = 24.576 GB; scores 16x512x100000x2 = 1.6384 GB;
    //       mlp 3x8x512x2816x2 = 69.2 MB; x1.25 margin.
    try t.expectEqual(
        @as(u64, 32_854_507_520),
        prefillMemoryNeeded(100_000, 16, 8, 30, 256, 2816, 2112, 16, 512),
    );
    // 4-bit: kv billed at (2*4+1)/16 bytes/elem (payload + group scale/bias)
    // = 6.912 GB, plus the 0.8192 GB per-layer dense dequant transient.
    try t.expectEqual(
        @as(u64, 11_798_507_520),
        prefillMemoryNeeded(100_000, 16, 8, 30, 256, 2816, 2112, 4, 512),
    );
    // Direction: quantized admission must be under half the fp16 bill.
    try t.expect(prefillMemoryNeeded(100_000, 16, 8, 30, 256, 2816, 2112, 4, 512) * 2 <
        prefillMemoryNeeded(100_000, 16, 8, 30, 256, 2816, 2112, 16, 512));
}

test "prefillMemoryNeeded: working set is chunk-bounded — the 255K MoE prompt is admittable" {
    const t = std.testing;
    transformer_mod.fused256_override = false;
    defer transformer_mod.fused256_override = null;
    // The PR-#69 failure case: Qwen3.6-35B-A3B shape (whose checkpoint OMITS
    // intermediate_size, so the struct default 15360 leaks into ffn) at 262144
    // tokens with 4-bit KV. kv 6.04 GB + scores 4.29 GB + dequant 0.54 GB +
    // mlp 0.377 GB, x1.25 = ~13.1 GiB — admittable on a big Mac.
    const needed = prefillMemoryNeeded(262_144, 16, 2, 40, 256, 2048, 15360, 4, 512);
    try t.expectEqual(@as(u64, 14_061_404_160), needed);
    try t.expect(needed < 16 << 30);
    // The retired seq-scaled envelope billed 8 x seq x ffn x 2 working bytes
    // (~64 GB) on top of fp16 KV (~21.5 GB) -> ~107 GB and a spurious 400.
    const old_estimate: u64 = (40 * 2 * 262_144 * 2 * 256 * 2 + 8 * 262_144 * 15360 * 2) * 5 / 4;
    try t.expect(needed < old_estimate / 7);
}

test "prefillMemoryNeeded: unfused head dims bill the materialized score scratch" {
    const t = std.testing;
    transformer_mod.fused256_override = false;
    defer transformer_mod.fused256_override = null;
    // Same shape, head_dim 128 vs 256 (kv scaled by hdim too, so compare the
    // full bills computed by hand): hd=128 fp16 -> kv 12.288 GB + 0 scores +
    // mlp 69.2 MB, x1.25.
    try t.expectEqual(
        @as(u64, 15_446_507_520),
        prefillMemoryNeeded(100_000, 16, 8, 30, 128, 2816, 2112, 16, 512),
    );
    // hd=256 adds the [heads, chunk, seq] score tensor (the guard must see
    // what the composed SDPA path actually allocates). Decomposition: the
    // hd-128 bill + the KV doubling from 128->256 (12.288 GB x1.25) + the
    // score tensor (16x512x100000x2 = 1.6384 GB x1.25).
    const with_scores = prefillMemoryNeeded(100_000, 16, 8, 30, 256, 2816, 2112, 16, 512);
    try t.expectEqual(@as(u64, 15_446_507_520 + 15_360_000_000 + 2_048_000_000), with_scores);
}

test "prefillMemoryNeeded: fused hd-256 kernel drops the score bill, keeps KV + dequant" {
    const t = std.testing;
    // Default (msv_attn_p256 active): the composed score scratch never exists,
    // so hd 256 bills exactly the hd-256 KV + mlp — the unfused bill minus the
    // 1.6384 GB x1.25 score term from the test above.
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;
    try t.expectEqual(
        @as(u64, 32_854_507_520 - 2_048_000_000),
        prefillMemoryNeeded(100_000, 16, 8, 30, 256, 2816, 2112, 16, 512),
    );
    // The quantized-KV dequant transient is a denseView property, NOT a score
    // property — it must survive the fused kernel (fires on every kv-quant
    // request regardless of head_dim).
    try t.expectEqual(
        @as(u64, 11_798_507_520 - 2_048_000_000),
        prefillMemoryNeeded(100_000, 16, 8, 30, 256, 2816, 2112, 4, 512),
    );
}
