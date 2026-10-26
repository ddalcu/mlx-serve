//! generation error mapping and wire errors; ensure-loaded/load-failed/error responses

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const Scheduler = @import("../gpu_thread/scheduler.zig").Scheduler;
const Stream = @import("../engines/stream.zig").Stream;
const mlx_engine = @import("../engines/mlx.zig");
const err_name = @import("../../sdk/util/err_name.zig");
const instr = @import("../metrics/metrics.zig");
const testing = std.testing;
const http_server = @import("server.zig");

const Conn = @import("conn.zig").Conn;
const AdmissionBill = @import("../../abi/engine.zig").Bill;
const PROMPT_LOGPROBS_UNAVAILABLE_MSG = @import("../api/completions.zig").PROMPT_LOGPROBS_UNAVAILABLE_MSG;
const sendResponse = @import("respond.zig").sendResponse;
const jsonEscape = @import("../api/format.zig").jsonEscape;
const sendAnthropicError = @import("../api/anthropic.zig").sendAnthropicError;
const sendAnthropicEvent = @import("../api/anthropic.zig").sendAnthropicEvent;
const sendResponsesEvent = @import("../api/responses/events.zig").sendResponsesEvent;
const engine_requests = @import("../engines/refusal.zig");

/// A slot that errored reports why: the MLX error latch's `error.OutOfMemory` is a memory
/// failure the operator can act on. One helper for the three slot-drain sites.
pub fn slotFailure(slot: *Stream) anyerror {
    // A pre-prefill admission refusal is not an abandoned generation: the client gets a 400.
    if (slot.errorNameIs("PrefillDoesNotFit")) {
        prefill_refusal = if (slot.refused_bill) |bill| .{ .tokens = slot.prompt_len, .needed = bill[0], .available = bill[1] } else null;
        return error.PrefillDoesNotFit;
    }
    if (slot.errorNameIs("PromptLogprobsUnavailable")) return error.PromptLogprobsUnavailable;
    if (slot.errorIsMemory()) return error.GenerationOutOfMemory;
    if (slot.errorNameIs("DistWorkerLeft")) return error.DistWorkerLeft;
    return error.GenerationFailed;
}

/// The message `error.GenerationOutOfMemory` sends on every surface and both paths.
const GEN_OOM_MSG = "The engine ran out of GPU memory during this request and it was abandoned. The server is still running. Reduce the prompt length, lower --ctx-size, or free memory on the machine.";

/// The body of the pre-flight memory 400, quoting the hot-cache credits the guard compared.
/// Caller owns the returned bytes.
pub fn memoryRefusalMessage(
    allocator: std.mem.Allocator,
    prompt_len: usize,
    needed_mb: u64,
    avail_mb: u64,
    bill: AdmissionBill,
) ![]u8 {
    return std.fmt.allocPrint(allocator, "Prompt ({d} tokens) requires ~{d}MB GPU memory but only ~{d}MB is available and the hot prefix cache holds ~{d}MB more, all of which can be reclaimed (~{d}MB — the shortfall is elsewhere), which is not enough. Reduce prompt size, lower --ctx-size, or use a smaller model.", .{ prompt_len, needed_mb, avail_mb, bill.evictable / (1024 * 1024), bill.evictionCredit() / (1024 * 1024) });
}

/// The message `error.PrefillDoesNotFit` sends: refused before its first forward by the same
/// estimator the guard uses, re-asked after the hot cache gave back everything it could.
const PREFILL_NOFIT_MSG = "This prompt does not fit in GPU memory even after freeing the prefix cache; it was refused before any work started. Reduce the prompt length, lower --ctx-size, or free memory on the machine (the server log quotes the byte counts it compared).";

/// The figures of the `PrefillDoesNotFit` this connection thread last drained, read once by
/// `mapGenerationError`: the error name alone cannot carry them past `slotFailure`.
threadlocal var prefill_refusal: ?struct { tokens: usize, needed: u64, available: u64 } = null;

/// `PREFILL_NOFIT_MSG` with the figures the refusal compared, worded like the guard's 400 so a client
/// reads both refusals the same way. The bare message when no figures were drained.
fn prefillNoFitMessage(buf: []u8) []const u8 {
    const refusal = prefill_refusal orelse return PREFILL_NOFIT_MSG;
    prefill_refusal = null;
    const mb = 1024 * 1024;
    return std.fmt.bufPrint(buf, "This prompt ({d} tokens) does not fit in GPU memory even after freeing the prefix cache: it requires ~{d}MB GPU memory but only ~{d}MB is available. It was refused before any work started. Reduce the prompt length, lower --ctx-size, or free memory on the machine.", .{ refusal.tokens, refusal.needed / mb, refusal.available / mb }) catch PREFILL_NOFIT_MSG;
}

/// A generation failure as the wire sees it: one status, one message, both dialects' type.
pub const GenErrorWire = struct {
    status_line: []const u8,
    code: u32,
    openai_type: []const u8,
    anthropic_type: []const u8,
    message: []const u8,
};

/// The mapping from a failed generation to what the client is told. It existed only on the
/// non-streaming arms; streaming arms wrote a bare `server_error` into an SSE frame, so agents
/// never saw the actionable 400/503. The unknown arm keeps the error name at a mapped 500.
pub fn mapGenerationError(err: anyerror, buf: []u8) GenErrorWire {
    // Whatever the request saw, a lost context is the cause and a restart the remedy.
    if (mlx.gpuContextLost()) return .{
        .status_line = "503 Service Unavailable",
        .code = 503,
        .openai_type = "server_error",
        .anthropic_type = "api_error",
        .message = gpu_context_lost_message,
    };
    return switch (err) {
        // The MLX working-set latch (#353): a memory 503.
        error.GenerationOutOfMemory, error.OutOfMemory => .{
            .status_line = "503 Service Unavailable",
            .code = 503,
            .openai_type = "server_error",
            .anthropic_type = "api_error",
            .message = GEN_OOM_MSG,
        },
        // Refused before the first forward: the request is too large, a named 400.
        error.PrefillDoesNotFit => .{
            .status_line = "400 Bad Request",
            .code = 400,
            .openai_type = "invalid_request_error",
            .anthropic_type = "invalid_request_error",
            .message = prefillNoFitMessage(buf),
        },
        error.EmptyPrompt => .{
            .status_line = "400 Bad Request",
            .code = 400,
            .openai_type = "invalid_request_error",
            .anthropic_type = "invalid_request_error",
            .message = "the prompt encodes to no tokens",
        },
        // Backstop for a forward that returns fewer rows than the prompt: the request's shape, not a fault.
        error.PromptLogprobsUnavailable => .{
            .status_line = "400 Bad Request",
            .code = 400,
            .openai_type = "invalid_request_error",
            .anthropic_type = "invalid_request_error",
            .message = PROMPT_LOGPROBS_UNAVAILABLE_MSG,
        },
        // What the model's engine does not declare: refused by name, not silently ignored.
        error.EngineUnsupported => .{
            .status_line = "400 Bad Request",
            .code = 400,
            .openai_type = "invalid_request_error",
            .anthropic_type = "invalid_request_error",
            .message = engine_requests.refusalMessage(buf),
        },
        error.DistWorkerLeft => .{
            .status_line = "503 Service Unavailable",
            .code = 503,
            .openai_type = "server_error",
            .anthropic_type = "overloaded_error",
            .message = "the cluster's worker Mac left during this request; the server restarts to pair again",
        },
        error.GenerationFailed => .{
            .status_line = "500 Internal Server Error",
            .code = 500,
            .openai_type = "server_error",
            .anthropic_type = "api_error",
            .message = "generation failed",
        },
        else => .{
            .status_line = "500 Internal Server Error",
            .code = 500,
            .openai_type = "server_error",
            .anthropic_type = "api_error",
            .message = std.fmt.bufPrint(buf, "Internal server error: {s}", .{@errorName(err)}) catch
                "Internal server error",
        },
    };
}

/// The terminal SSE frame for the OpenAI dialects, with a `finish_reason: "error"` choice so a
/// chunk-only client terminates.
fn buildOpenAiStreamError(allocator: std.mem.Allocator, w: GenErrorWire) ![]u8 {
    const esc = try jsonEscape(allocator, w.message);
    defer allocator.free(esc);
    return std.fmt.allocPrint(allocator,
        \\data: {{"error":{{"message":{s},"type":"{s}","param":null,"code":{d}}},"choices":[{{"index":0,"delta":{{}},"finish_reason":"error"}}]}}
    ++ "\n\n", .{ esc, w.openai_type, w.code });
}

/// The `/v1/messages` terminal `error` event payload.
fn buildAnthropicStreamError(allocator: std.mem.Allocator, w: GenErrorWire) ![]u8 {
    const esc = try jsonEscape(allocator, w.message);
    defer allocator.free(esc);
    return std.fmt.allocPrint(allocator,
        \\{{"type":"error","error":{{"type":"{s}","message":{s}}}}}
    , .{ w.anthropic_type, esc });
}

/// The `/v1/responses` terminal `error` event payload (`sendResponsesEvent` adds `sequence_number`).
fn buildResponsesStreamError(allocator: std.mem.Allocator, w: GenErrorWire) ![]u8 {
    const esc = try jsonEscape(allocator, w.message);
    defer allocator.free(esc);
    return std.fmt.allocPrint(allocator,
        \\{{"type":"error","code":"{s}","message":{s},"param":null}}
    , .{ w.openai_type, esc });
}

/// Which wire dialect a generative surface speaks its errors in.
const ErrorSurface = union(enum) {
    openai,
    anthropic,
    responses: *u64,
};

/// The error arm for every generative surface, streaming or not: before the SSE head the
/// stream gets the same response its non-streaming twin would; past it the same `type` and
/// `message` ride the surface's terminal `error` event.
/// A client that hung up mid-stream (`WriteFailed`) is not a server error.
pub fn logStreamError(err: anyerror) void {
    if (err == error.WriteFailed) log.debug("  -> client disconnected mid-stream\n", .{}) else log.err("  -> streaming error: {}\n", .{err});
}

pub fn sendGenerationError(allocator: std.mem.Allocator, stream: *Conn, err: anyerror, surface: ErrorSurface) !void {
    var msg_buf: [320]u8 = undefined;
    const w = mapGenerationError(err, &msg_buf);

    if (!stream.sse_headers_sent) {
        return switch (surface) {
            .anthropic => sendAnthropicError(allocator, stream, w.anthropic_type, w.message, w.code),
            else => sendErrorResponse(allocator, stream, w.status_line, w.openai_type, w.message, w.code),
        };
    }

    switch (surface) {
        .openai => {
            const body = try buildOpenAiStreamError(allocator, w);
            defer allocator.free(body);
            try stream.writeAllNoFlush(body);
            try stream.writeAll("data: [DONE]\n\n");
        },
        .anthropic => {
            const body = try buildAnthropicStreamError(allocator, w);
            defer allocator.free(body);
            // Anthropic streams have no `[DONE]`; the `error` event is the terminator.
            try sendAnthropicEvent(stream, "error", body);
        },
        .responses => |seq| {
            const body = try buildResponsesStreamError(allocator, w);
            defer allocator.free(body);
            try sendResponsesEvent(allocator, stream, seq, "error", body);
            // Same gate as the success path: no `[DONE]` over the WS bridge.
            if (stream.ws_mode == null) try stream.writeAll("data: [DONE]\n\n");
        },
    }
}

/// Extract a JSON field's raw value from a JSON body string.
/// Returns the raw substring for the field value (e.g., the array or object).
pub fn extractJsonField(body: []const u8, field: []const u8) ?[]const u8 {
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

pub const gpu_context_lost_message =
    "The GPU context was lost (a CUDA kernel faulted) and every GPU call now fails. Restart the server; the server log names the first error.";

/// Body text for a context-overflow 400, in a caller-owned buffer.
///
/// The COUNTS are the point. "Prompt exceeds maximum context length" tells a
/// client only that it failed; with both numbers the client can say how far
/// over it went and offer the one action that fixes it (raise the context to
/// N). Our own app renders exactly that, and the numbers are only knowable
/// here — the request is rejected before any usage is reported.
///
/// The legacy sentence stays the PREFIX: `APIError.looksLikeContextOverflow`
/// and third-party clients key on it, so appending must not move it. Output is
/// digits and spaces, so it carries no escaping hazard into the JSON sink, and
/// a bufPrint failure falls back to the bare sentence rather than sending no
/// body at all (the media-gen fixed-buffer class).
/// The `error.NotEnoughMemory` refusal, in ONE place because two identical
/// copies of it is how they drift.
///
/// The old wording ("retry after current requests complete") named concurrency,
/// which is the one thing it is never waiting for: the gate refuses against a
/// STATIC cap, so on an idle server with nothing loaded there is nothing to
/// retry after (#126). The counts are knowable only server-side and go to the
/// log; what the client needs is the knob.
///
/// ONE gate reaches this arm: the resident-model budget with no evictable
/// LRU victim (scheduler's `planEvictionsLocked` refusal). The free-memory
/// PREFLIGHT refuses as `error.InsufficientMemory` and gets its own message
/// below (#144) — blaming free memory here sends a user quitting apps when
/// the fix is unloading a model or moving the cap.
pub const not_enough_memory_message =
    "Not enough memory to load model: it would exceed the resident-model budget and no loaded model can be evicted. " ++
    "Unload the model you are chatting with (tray > Models > eject), " ++
    "or raise a cap: --max-resident-models <n>, --max-resident-mem <size>|0. " ++
    "The server log names the exact figures and the current cap.";

/// #144: the memory PREFLIGHT refusal (free RAM can't hold weights + warmup
/// headroom) is distinct from the resident-budget gate above and has a
/// different knob. Counts go to the log (#126); the client gets the remedy.
pub const insufficient_free_memory_message =
    "Not enough free memory to load model: weights + warmup headroom exceed what is currently available. " ++
    "Close other apps or models and retry (the server log names the peak estimate and available memory), or pass --skip-mem-preflight to override.";

/// The response for a failed `Scheduler.ensureLoaded(id)`.
pub fn sendEnsureLoadedError(allocator: std.mem.Allocator, stream: *Conn, sched: *Scheduler, id: []const u8, err: anyerror) !void {
    switch (err) {
        error.UnknownModelId => try sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "Unknown model id", 404),
        error.NotLoaded => try sendErrorResponse(allocator, stream, "503 Service Unavailable", "model_not_loaded", "Requested model is not currently loaded", 503),
        error.NoDefaultModel => try sendErrorResponse(allocator, stream, "503 Service Unavailable", "no_model", "No default model configured", 503),
        error.NotEnoughMemory => try sendErrorResponse(allocator, stream, "503 Service Unavailable", "out_of_memory", not_enough_memory_message, 503),
        error.InsufficientMemory => try sendErrorResponse(allocator, stream, "503 Service Unavailable", "out_of_memory", insufficient_free_memory_message, 503),
        error.LoadFailed => try sendLoadFailedResponse(allocator, stream, sched, id),
        error.Shutdown => try sendErrorResponse(allocator, stream, "503 Service Unavailable", "shutting_down", "Server is shutting down", 503),
        error.GpuContextLost => try sendErrorResponse(allocator, stream, "503 Service Unavailable", "gpu_context_lost", gpu_context_lost_message, 503),
        error.ForeignBytesReaderBusy => try sendErrorResponse(allocator, stream, "503 Service Unavailable", "model_busy", "Another DeepSeek-V4.1 or GLM-5.3 stream pack is in use; retry when its request finishes", 503),
        // Other errors (CPU-side preload failures like FileNotFound on a
        // missing tokenizer.json, JSON parse errors on a malformed
        // config.json, etc.) bubble out of `preloadCpuState`. Surface a
        // 500 with the error name so the client gets a clean failure
        // instead of a hung connection.
        else => {
            // An error the engine named without this unit knowing it arrives as `EngineFailed`, its name kept aside.
            const name = if (err == error.EngineFailed) err_name.last() else @errorName(err);
            log.warn("  -> 500 ({s}) while loading {s}\n", .{ name, id });
            const msg = std.fmt.allocPrint(allocator, "Failed to load model: {s}", .{name}) catch {
                try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", "Failed to load model", 500);
                return;
            };
            defer allocator.free(msg);
            try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", msg, 500);
        },
    }
}

/// #144: `error.LoadFailed` used to surface as a bare "Model load failed" for
/// every on-demand load failure — the inference thread's error name was
/// dropped at the thread boundary. The registry keeps it; echo it.
fn sendLoadFailedResponse(allocator: std.mem.Allocator, stream: *Conn, sched: *Scheduler, id: []const u8) !void {
    if (mlx_engine.api().load_error_name(sched, allocator, id)) |name| {
        defer allocator.free(name);
        if (std.fmt.allocPrint(allocator, "Model load failed: {s}", .{name}) catch null) |msg| {
            defer allocator.free(msg);
            try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", msg, 500);
            return;
        }
    }
    try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", "Model load failed", 500);
}

/// A request refused before it owned a slot. A null sink (metrics off) is one branch.
pub fn countRejected() void {
    if (http_server.g_metrics) |m| m.recordRejected();
}

pub fn contextOverflowMessage(buf: []u8, prompt_tokens: usize, ctx: usize) []const u8 {
    return std.fmt.bufPrint(
        buf,
        "Prompt exceeds maximum context length: {d} tokens requested, {d} available",
        .{ prompt_tokens, ctx },
    ) catch "Prompt exceeds maximum context length";
}

pub fn sendErrorResponse(allocator: std.mem.Allocator, stream: *Conn, status: []const u8, err_type: []const u8, message: []const u8, code: ?u32) !void {
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

// ── tests ──

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

test "the out-of-memory 503 names the cap's flag and never blames concurrency" {
    // #126: an idle server with zero models loaded and 21 MB RSS answered
    // "retry after current requests complete", which is unactionable advice
    // pointing at the wrong subsystem — the cause is the static
    // --max-resident-mem cap, a flag the message never mentioned and the app
    // does not pass. Both 503 sites read the one constant.
    const m = not_enough_memory_message;
    try testing.expect(std.mem.indexOf(u8, m, "--max-resident-mem") != null);
    try testing.expect(std.mem.indexOf(u8, m, "retry after current requests") == null);
    // Only the resident-budget gate raises NotEnoughMemory; the free-memory
    // preflight has its own arm (insufficient_free_memory_message). Blaming
    // free memory here sends the user quitting apps when the fix is the cap.
    try testing.expect(std.mem.indexOf(u8, m, "free memory") == null);
    try testing.expect(std.mem.indexOf(u8, m, "resident-model budget") != null);
}

test "PrefillDoesNotFit quotes the figures the inference thread refused on, once" {
    const t = std.testing;
    const mb = 1024 * 1024;
    var buf: [320]u8 = undefined;
    prefill_refusal = .{ .tokens = 12447, .needed = 1733 * mb, .available = 77 * mb };
    const w = mapGenerationError(error.PrefillDoesNotFit, &buf);
    try t.expectEqual(@as(u32, 400), w.code);
    try t.expect(std.mem.indexOf(u8, w.message, "This prompt (12447 tokens) does not fit in GPU memory") != null);
    // The guard's wording, so a client parses both refusals with one pattern.
    try t.expect(std.mem.indexOf(u8, w.message, "requires ~1733MB GPU memory but only ~77MB is available") != null);
    // Read once: the next refusal without figures on this thread gets the bare message.
    try t.expect(prefill_refusal == null);
    try t.expectEqualStrings(PREFILL_NOFIT_MSG, mapGenerationError(error.PrefillDoesNotFit, &buf).message);
}

test "mapGenerationError: after a lost GPU context any failure is the named 503" {
    const t = std.testing;
    var stale: [512]u8 = undefined;
    _ = mlx.takeError(&stale);
    defer mlx.resetGpuContextLostForTest();
    var buf: [192]u8 = undefined;
    try t.expectEqual(@as(u32, 503), mapGenerationError(error.DistWorkerLeft, &buf).code);
    try t.expectEqual(@as(u32, 500), mapGenerationError(error.GenerationFailed, &buf).code);
    mlx.latchErrorForTest("cudaGraphLaunch(graph_exec, stream_) failed: an illegal memory access was encountered");
    _ = mlx.takeError(&stale);
    const w = mapGenerationError(error.GenerationFailed, &buf);
    try t.expectEqual(@as(u32, 503), w.code);
    try t.expectEqualStrings(gpu_context_lost_message, w.message);
}

test "a streaming fault answers with the SAME mapped error a non-streaming one does" {
    // The 400/503 mapping existed only on the non-streaming arms; streaming arms wrote a bare `server_error` frame.
    const t = std.testing;
    const a = t.allocator;

    var buf: [192]u8 = undefined;
    {
        const w = mapGenerationError(error.GenerationOutOfMemory, &buf);
        try t.expectEqual(@as(u32, 503), w.code);
        try t.expectEqualStrings("503 Service Unavailable", w.status_line);
        try t.expectEqualStrings(GEN_OOM_MSG, w.message);
        try t.expectEqualStrings("server_error", w.openai_type);
        try t.expectEqualStrings("api_error", w.anthropic_type);
    }
    {
        const w = mapGenerationError(error.PrefillDoesNotFit, &buf);
        try t.expectEqual(@as(u32, 400), w.code);
        try t.expectEqualStrings(PREFILL_NOFIT_MSG, w.message);
        try t.expectEqualStrings("invalid_request_error", w.openai_type);
        try t.expectEqualStrings("invalid_request_error", w.anthropic_type);
    }
    {
        const w = mapGenerationError(error.GenerationFailed, &buf);
        try t.expectEqual(@as(u32, 500), w.code);
        try t.expectEqualStrings("generation failed", w.message);
    }
    {
        // A prefill that cannot return every prompt row is the request's shape, not a fault.
        const w = mapGenerationError(error.PromptLogprobsUnavailable, &buf);
        try t.expectEqual(@as(u32, 400), w.code);
        try t.expectEqualStrings("invalid_request_error", w.openai_type);
        try t.expectEqualStrings("invalid_request_error", w.anthropic_type);
    }
    {
        // A prompt that encodes to no tokens is refused before a slot exists.
        const w = mapGenerationError(error.EmptyPrompt, &buf);
        try t.expectEqual(@as(u32, 400), w.code);
        try t.expectEqualStrings("invalid_request_error", w.openai_type);
    }
    {
        // What a model's engine cannot honor is the request's shape, not a fault, and the message names it.
        engine_requests.refused().* = .{ .kind = .ane, .cap = "logprobs" };
        const w = mapGenerationError(error.EngineUnsupported, &buf);
        try t.expectEqual(@as(u32, 400), w.code);
        try t.expectEqualStrings("invalid_request_error", w.anthropic_type);
        try t.expect(std.mem.indexOf(u8, w.message, "Neural Engine") != null);
    }
    {
        // Anything else keeps its name in the message, at a mapped status.
        const w = mapGenerationError(error.WriteFailed, &buf);
        try t.expectEqual(@as(u32, 500), w.code);
        try t.expectEqualStrings("Internal server error: WriteFailed", w.message);
    }

    // The SSE bodies carry the same type and message, plus the surface's terminal marker.
    {
        const w = mapGenerationError(error.GenerationOutOfMemory, &buf);
        const body = try buildOpenAiStreamError(a, w);
        defer a.free(body);
        try t.expect(std.mem.indexOf(u8, body, "\"type\":\"server_error\"") != null);
        try t.expect(std.mem.indexOf(u8, body, "\"code\":503") != null);
        try t.expect(std.mem.indexOf(u8, body, GEN_OOM_MSG) != null);
        try t.expect(std.mem.indexOf(u8, body, "\"finish_reason\":\"error\"") != null);

        const anth = try buildAnthropicStreamError(a, w);
        defer a.free(anth);
        try t.expect(std.mem.indexOf(u8, anth, "\"type\":\"error\"") != null);
        try t.expect(std.mem.indexOf(u8, anth, "\"type\":\"api_error\"") != null);
        try t.expect(std.mem.indexOf(u8, anth, GEN_OOM_MSG) != null);

        const resp = try buildResponsesStreamError(a, w);
        defer a.free(resp);
        try t.expect(std.mem.indexOf(u8, resp, "\"type\":\"error\"") != null);
        try t.expect(std.mem.indexOf(u8, resp, "\"code\":\"server_error\"") != null);
        try t.expect(std.mem.indexOf(u8, resp, GEN_OOM_MSG) != null);
    }
}

test "contextOverflowMessage: the 400 names both counts so a client can act on it" {
    const t = std.testing;
    var buf: [160]u8 = undefined;

    // The counts ARE the feature: without them a client can only say "too
    // long", and our own app can't offer "raise the context to N" — which is
    // the one action that fixes it.
    const msg = contextOverflowMessage(&buf, 4108, 4096);
    try t.expectEqualStrings("Prompt exceeds maximum context length: 4108 tokens requested, 4096 available", msg);

    // Legacy prefix preserved verbatim — APIError.looksLikeContextOverflow and
    // every third-party client key on this exact phrase.
    try t.expect(std.mem.startsWith(u8, msg, "Prompt exceeds maximum context length"));

    // No quotes or control bytes, so the message is safe to drop into a JSON
    // error body without escaping (the appendJsonString class).
    for (msg) |c| try t.expect(c >= 0x20 and c != '"' and c != '\\');
}

test "contextOverflowMessage: a buffer too small falls back rather than sending nothing" {
    const t = std.testing;
    // bufPrint failing must not propagate — the media-gen 400 that sent NO body
    // at all when its fixed buffer overflowed is the class this guards.
    var tiny: [8]u8 = undefined;
    try t.expectEqualStrings("Prompt exceeds maximum context length", contextOverflowMessage(&tiny, 999999, 1));
}

test "memoryRefusalMessage: the 400 quotes the hot-cache credits the guard compared" {
    const t = std.testing;
    const MB: u64 = 1024 * 1024;

    const rich = try memoryRefusalMessage(t.allocator, 12_000, 4096, 2048, .{
        .needed = 30 * MB,
        .available = 20 * MB,
        .evictable = 1500 * MB,
        .reclaimable = 900 * MB,
    });
    defer t.allocator.free(rich);
    try t.expect(std.mem.indexOf(u8, rich, "holds ~1500MB more") != null);
    try t.expect(std.mem.indexOf(u8, rich, "(~900MB") != null);

    const cold = try memoryRefusalMessage(t.allocator, 12_000, 4096, 2048, .{ .needed = 30 * MB, .available = 20 * MB });
    defer t.allocator.free(cold);
    try t.expectEqualStrings(
        "Prompt (12000 tokens) requires ~4096MB GPU memory but only ~2048MB is available and the hot prefix cache holds ~0MB more, all of which can be reclaimed (~0MB — the shortfall is elsewhere), which is not enough. Reduce prompt size, lower --ctx-size, or use a smaller model.",
        cold,
    );
}

test "outcome row 7/9: countRejected moves the rejected counter, and does nothing with metrics off" {
    const t = std.testing;
    const saved = http_server.g_metrics;
    defer http_server.g_metrics = saved;

    http_server.g_metrics = null;
    countRejected();

    var m = instr.Metrics.init();
    http_server.g_metrics = &m;
    countRejected();
    try t.expectEqual(@as(u64, 1), m.requests_rejected_total.load());
    try t.expectEqual(@as(u64, 0), m.requests_failed_total.load());
    try t.expectEqual(@as(u64, 0), m.requests_cancelled_total.load());
}
