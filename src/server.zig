const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const tokenizer_mod = @import("tokenizer.zig");
const generate_mod = @import("generate.zig");
const chat_mod = @import("chat.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");
const metrics = @import("status.zig");

const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const Generator = generate_mod.Generator;
/// Global flag set by signal handler for graceful shutdown.
var shutdown_requested = std.atomic.Value(bool).init(false);

/// Single-slot inference gate: mutex + condition variable for request queuing.
/// Requests wait in line instead of getting 503 when the server is busy.
var inference_mutex: std.Thread.Mutex = .{};
var inference_cond: std.Thread.Condition = .{};
var inference_busy: bool = false;
var inference_queue_len: u32 = 0;
const max_queue_size: u32 = 32;

/// Acquire the inference slot, blocking until available.
/// Returns false if the queue is full or shutdown was requested.
fn acquireInferenceSlot() bool {
    inference_mutex.lock();
    defer inference_mutex.unlock();

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
        inference_cond.wait(&inference_mutex);
    }

    inference_queue_len -= 1;
    inference_busy = true;
    return true;
}

/// Release the inference slot and wake the next waiting request.
fn releaseInferenceSlot() void {
    inference_mutex.lock();
    defer inference_mutex.unlock();

    inference_busy = false;
    inference_cond.signal();
}

fn signalHandler(_: c_int) callconv(.c) void {
    shutdown_requested.store(true, .release);
}

/// Maximum context size (0 = unlimited). Set by --ctx-size flag.
var max_context_size: u32 = 0;

/// Request timeout in seconds (0 = no timeout). Set by --timeout flag.
var request_timeout_sec: u32 = 300;

/// Default reasoning budget in tokens (-1 = unlimited). Set by --reasoning-budget flag.
var default_reasoning_budget: i32 = -1;

fn getTimeoutNs() u64 {
    if (request_timeout_sec == 0) return 0;
    return @as(u64, request_timeout_sec) * std.time.ns_per_s;
}

/// Cached prompt IDs from the last request (for KV cache reuse).
var cached_prompt_ids: ?[]u32 = null;

/// Start the HTTP server on the given host and port.
pub fn serve(
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    tok: *const Tokenizer,
    chat_config: *const chat_mod.ChatConfig,
    config: *const model_mod.ModelConfig,
    host: []const u8,
    port: u16,
    ctx_size: u32,
    timeout: u32,
    reasoning_budget: i32,
) !void {
    max_context_size = ctx_size;
    request_timeout_sec = timeout;
    default_reasoning_budget = reasoning_budget;
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

    const addr = std.net.Address.initIp4(ip4_bytes, port);
    var server = try addr.listen(.{ .reuse_address = true });
    defer server.deinit();

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
    log.info("\nServer listening on http://{s}:{d}\n", .{ host, port });
    log.info("  GET  /health\n", .{});
    log.info("  GET  /props\n", .{});
    log.info("  GET  /v1/models\n", .{});
    log.info("  POST /v1/chat/completions\n", .{});
    log.info("  POST /v1/completions\n", .{});
    log.info("  POST /v1/embeddings\n", .{});
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
        .fd = server.stream.handle,
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

        const conn = server.accept() catch |err| {
            if (shutdown_requested.load(.acquire)) break;
            log.err("accept error: {}\n", .{err});
            continue;
        };

        // Spawn a thread to handle the connection so we can accept new ones immediately.
        // This allows health checks and 503 responses while generation is running.
        const thread = std.Thread.spawn(.{}, connectionThread, .{ allocator, conn.stream, xfm, tok, chat_config, config }) catch {
            // If thread spawn fails, handle synchronously
            handleConnection(allocator, conn.stream, xfm, tok, chat_config, config) catch {};
            conn.stream.close();
            continue;
        };
        thread.detach();
    }

    // Free cached prompt on shutdown
    if (cached_prompt_ids) |old| {
        allocator.free(old);
        cached_prompt_ids = null;
    }

    log.info("\nShutting down gracefully...\n", .{});
}

fn connectionThread(
    allocator: std.mem.Allocator,
    stream: std.net.Stream,
    xfm: *Transformer,
    tok: *const Tokenizer,
    chat_config: *const chat_mod.ChatConfig,
    config: *const model_mod.ModelConfig,
) void {
    defer stream.close();
    handleConnection(allocator, stream, xfm, tok, chat_config, config) catch |err| {
        switch (err) {
            error.BrokenPipe, error.ConnectionResetByPeer => {
                log.debug("  -> client disconnected\n", .{});
            },
            else => {
                log.err("  -> error: {}\n", .{err});
            },
        }
    };
}

fn handleConnection(
    allocator: std.mem.Allocator,
    stream: std.net.Stream,
    xfm: *Transformer,
    tok: *const Tokenizer,
    chat_config: *const chat_mod.ChatConfig,
    config: *const model_mod.ModelConfig,
) !void {
    // Read the full HTTP request (up to 1MB for large tool definitions / conversation history)
    var buf: [1024 * 1024]u8 = undefined;
    var total_read: usize = 0;

    while (total_read < buf.len) {
        const n = try stream.read(buf[total_read..]);
        if (n == 0) break;
        total_read += n;

        if (std.mem.indexOf(u8, buf[0..total_read], "\r\n\r\n")) |header_end| {
            const headers = buf[0..header_end];
            if (findContentLength(headers)) |cl| {
                const body_start = header_end + 4;
                const body_received = total_read - body_start;
                if (body_received >= cl) break;
            } else {
                break;
            }
        }
    }

    const request = buf[0..total_read];
    const first_line_end = std.mem.indexOf(u8, request, "\r\n") orelse return;
    const first_line = request[0..first_line_end];

    var line_iter = std.mem.splitScalar(u8, first_line, ' ');
    const method = line_iter.next() orelse return;
    const path = line_iter.next() orelse return;

    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/health")) {
        log.debug("GET  /health -> 200\n", .{});
        try sendResponse(stream, "200 OK", "application/json", "{\"status\":\"ok\"}");
    } else if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/v1/models")) {
        log.debug("GET  /v1/models -> 200\n", .{});
        try handleModels(allocator, stream, config);
    } else if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/props")) {
        log.debug("GET  /props -> 200\n", .{});
        try handleProps(allocator, stream, config, chat_config);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/chat/completions")) {
        if (config.is_encoder_only) {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Encoder-only models do not support text generation. Use /v1/embeddings instead.", 400);
            return;
        }
        if (!acquireInferenceSlot()) {
            log.warn("POST /v1/chat/completions -> 503 (queue full)\n", .{});
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "server_error", "Server request queue is full. Try again shortly.", 503);
            return;
        }
        defer releaseInferenceSlot();
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleChatCompletions(allocator, stream, body, xfm, tok, chat_config, config);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/completions")) {
        if (config.is_encoder_only) {
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Encoder-only models do not support text generation. Use /v1/embeddings instead.", 400);
            return;
        }
        if (!acquireInferenceSlot()) {
            log.warn("POST /v1/completions -> 503 (queue full)\n", .{});
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "server_error", "Server request queue is full. Try again shortly.", 503);
            return;
        }
        defer releaseInferenceSlot();
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleCompletions(allocator, stream, body, xfm, tok, config);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/embeddings")) {
        if (!acquireInferenceSlot()) {
            log.warn("POST /v1/embeddings -> 503 (queue full)\n", .{});
            try sendErrorResponse(allocator, stream, "503 Service Unavailable", "server_error", "Server request queue is full. Try again shortly.", 503);
            return;
        }
        defer releaseInferenceSlot();
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleEmbeddings(allocator, stream, body, xfm, tok, config);
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
    return config.max_position_embeddings;
}

/// Clamp max_tokens so prompt + completion doesn't exceed context length.
fn clampMaxTokens(max_tokens: u32, prompt_len: usize) u32 {
    if (max_context_size == 0) return max_tokens;
    const prompt: u32 = @intCast(@min(prompt_len, max_context_size));
    if (prompt >= max_context_size) return 1; // at least 1 token
    const remaining = max_context_size - prompt;
    if (max_tokens > remaining) {
        log.debug("  max_tokens clamped: {d} -> {d} (ctx_size={d}, prompt={d})\n", .{ max_tokens, remaining, max_context_size, prompt });
        return remaining;
    }
    return max_tokens;
}

fn handleModels(allocator: std.mem.Allocator, stream: std.net.Stream, config: *const model_mod.ModelConfig) !void {
    const ctx_len = getEffectiveContextLength(config);
    const ctx_str = if (ctx_len > 0) blk: {
        break :blk try std.fmt.allocPrint(allocator, "{d}", .{ctx_len});
    } else try std.fmt.allocPrint(allocator, "null", .{});
    defer allocator.free(ctx_str);

    const body = try std.fmt.allocPrint(allocator,
        \\{{"object":"list","data":[{{"id":"{s}","object":"model","created":{d},"owned_by":"mlx-serve","meta":{{"vocab_size":{d},"hidden_size":{d},"num_layers":{d},"quantization":"{d}-bit","context_length":{s}}}}}]}}
    , .{ config.model_type, std.time.timestamp(), config.vocab_size, config.hidden_size, config.num_hidden_layers, config.quant_bits, ctx_str });
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

fn handleProps(allocator: std.mem.Allocator, stream: std.net.Stream, config: *const model_mod.ModelConfig, chat_config: *const chat_mod.ChatConfig) !void {
    const ctx_len = getEffectiveContextLength(config);
    const ctx_str = if (ctx_len > 0) blk: {
        break :blk try std.fmt.allocPrint(allocator, "{d}", .{ctx_len});
    } else try std.fmt.allocPrint(allocator, "0", .{});
    defer allocator.free(ctx_str);

    // Query MLX memory usage
    var active_mem: usize = 0;
    var peak_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    _ = mlx.mlx_get_peak_memory(&peak_mem);

    // JSON-escape the chat template
    const escaped_template = try jsonEscape(allocator, chat_config.chat_template);
    defer allocator.free(escaped_template);

    const body = try std.fmt.allocPrint(allocator,
        \\{{"default_generation_settings":{{"model":"{s}","n_ctx":{s}}},"total_slots":1,"chat_template":{s},"model_info":{{"vocab_size":{d},"hidden_size":{d},"num_hidden_layers":{d},"num_attention_heads":{d},"num_key_value_heads":{d},"head_dim":{d},"quantization_bits":{d},"quantization_group_size":{d},"max_position_embeddings":{d}}},"memory":{{"active_bytes":{d},"peak_bytes":{d}}}}}
    , .{
        config.model_type,        ctx_str,
        escaped_template,
        config.vocab_size,        config.hidden_size,
        config.num_hidden_layers, config.num_attention_heads,
        config.num_key_value_heads, config.head_dim,
        config.quant_bits,        config.quant_group_size,
        config.max_position_embeddings,
        active_mem,               peak_mem,
    });
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

fn handleEmbeddings(
    allocator: std.mem.Allocator,
    stream: std.net.Stream,
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
    stream: std.net.Stream,
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
    stream: std.net.Stream,
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
    stream: std.net.Stream,
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
        const content: []const u8 = if (content_val) |cv| switch (cv) {
            .string => |s| s,
            .array => |arr| blk: {
                for (arr.items) |part| {
                    if (part != .object) continue;
                    const ptype = part.object.get("type") orelse continue;
                    if (ptype != .string or !std.mem.eql(u8, ptype.string, "text")) continue;
                    const text = part.object.get("text") orelse continue;
                    if (text == .string) break :blk text.string;
                }
                break :blk "";
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

        // Skip messages with no content and no tool_calls
        if (content.len == 0 and msg_tool_calls == null and !std.mem.eql(u8, role_val.string, "tool")) continue;

        try messages.append(allocator, .{
            .role = role_val.string,
            .content = content,
            .tool_calls = msg_tool_calls,
            .tool_call_id = tool_call_id,
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
                else => 256,
            }
        else
            256;
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

    // Parse response_format — inject JSON schema constraint into system message
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
                            var out: std.io.Writer.Allocating = .init(allocator);
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
                const instruction = "Respond with valid JSON only. No other text, no markdown, no explanation.";
                if (messages.items.len > 0 and std.mem.eql(u8, messages.items[0].role, "system")) {
                    const combined = try std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ messages.items[0].content, instruction });
                    try rf_allocs.append(allocator, combined);
                    messages.items[0].content = combined;
                } else {
                    try messages.insert(allocator, 0, .{ .role = "system", .content = instruction, .tool_calls = null, .tool_call_id = null });
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
    const prompt_ids = try chat_mod.formatChat(allocator, tok, messages.items, chat_config, tools_json, tool_choice_instruction, enable_thinking);
    defer allocator.free(prompt_ids);

    // Enforce context size limit
    if (max_context_size > 0 and prompt_ids.len > max_context_size) {
        log.warn("POST /v1/chat/completions -> 400 (prompt {d} tokens exceeds ctx_size {d})\n", .{ prompt_ids.len, max_context_size });
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Prompt exceeds maximum context length", 400);
        return;
    }

    // Clamp max_tokens to stay within context window
    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len);

    // Prompt caching: reuse KV cache for shared prefix
    const cache_result = try reuseKVCache(allocator, xfm, prompt_ids);

    const eos_slice = config.eosTokenSlice();

    const sampling = generate_mod.SamplingParams{
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .repeat_penalty = repeat_penalty,
        .presence_penalty = presence_penalty,
        .seed = seed,
    };

    if (is_stream) {
        handleStreamingGeneration(allocator, stream, xfm, tok, cache_result.new_tokens, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, include_usage, has_tools, cache_result.cached_tokens, logprobs_n, enable_thinking, reasoning_budget) catch |err| {
            log.err("  -> streaming error: {}\n", .{err});
        };
    } else {
        handleNonStreamingGeneration(allocator, stream, xfm, tok, cache_result.new_tokens, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, has_tools, cache_result.cached_tokens, logprobs_n, enable_thinking, reasoning_budget) catch |err| {
            log.err("  -> 500 ({s})\n", .{@errorName(err)});
            sendErrorResponse(allocator, stream, "500 Internal Server Error", "server_error", @errorName(err), 500) catch {};
        };
    }

    // Store prompt IDs for next request's cache comparison
    updateCachedPrompt(allocator, prompt_ids);
}

fn handleCompletions(
    allocator: std.mem.Allocator,
    stream: std.net.Stream,
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
            else => 256,
        } else 256;
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
    if (max_context_size > 0 and prompt_ids.len > max_context_size) {
        log.warn("POST /v1/completions -> 400 (prompt {d} tokens exceeds ctx_size {d})\n", .{ prompt_ids.len, max_context_size });
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Prompt exceeds maximum context length", 400);
        return;
    }

    // Clamp max_tokens to stay within context window
    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len);

    // Prompt caching: reuse KV cache for shared prefix
    const cache_result = try reuseKVCache(allocator, xfm, prompt_ids);

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
    updateCachedPrompt(allocator, prompt_ids);
}

fn handleNonStreamingCompletion(
    allocator: std.mem.Allocator,
    stream: std.net.Stream,
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
    var timer = try std.time.Timer.start();

    var result = try generate_mod.generate(allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, getTimeoutNs(), 0);
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
    const tps = if (elapsed_ms > 0) @as(u64, result.completion_tokens) * 1000 / elapsed_ms else 0;
    log.info("  <- {d}+{d} tokens ({d}ms, ~{d} tok/s) [{s}]\n", .{
        result.prompt_tokens, result.completion_tokens, elapsed_ms, tps, finish_reason,
    });

    const escaped_text = jsonEscape(allocator, final_text) catch "\"\"";
    defer if (!std.mem.eql(u8, escaped_text, "\"\"")) allocator.free(escaped_text);

    const response = try std.fmt.allocPrint(allocator,
        \\{{"id":"cmpl-{d}","object":"text_completion","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[{{"index":0,"text":{s},"finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
    , .{
        std.time.milliTimestamp(),
        std.time.timestamp(),
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
    stream: std.net.Stream,
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
    const cmpl_id = std.time.milliTimestamp();
    const created_ts = std.time.timestamp();
    var timer = try std.time.Timer.start();

    var gen = try Generator.init(allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids);
    gen.timeout_ns = getTimeoutNs();
    defer gen.deinit(allocator);

    // SSE headers
    const header =
        "HTTP/1.1 200 OK\r\n" ++
        "Content-Type: text/event-stream\r\n" ++
        "Cache-Control: no-cache\r\n" ++
        "Connection: keep-alive\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n" ++
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" ++
        "\r\n";
    try stream.writeAll(header);

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

    try stream.writeAll("data: ");
    try stream.writeAll(final_chunk);
    try stream.writeAll("\n\n");
    try stream.writeAll("data: [DONE]\n\n");

    const elapsed_ms = timer.read() / std.time.ns_per_ms;
    const tps = if (elapsed_ms > 0) @as(u64, gen.completion_tokens) * 1000 / elapsed_ms else 0;
    log.info("  <- {d}+{d} tokens streamed ({d}ms, ~{d} tok/s) [{s}]\n", .{
        gen.prompt_tokens, gen.completion_tokens, elapsed_ms, tps, finish_reason,
    });
}

fn handleNonStreamingGeneration(
    allocator: std.mem.Allocator,
    stream: std.net.Stream,
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
) !void {
    var timer = try std.time.Timer.start();

    var result = try generate_mod.generate(allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, getTimeoutNs(), logprobs_n);
    result.prompt_tokens += cached_tokens; // Include cached tokens in total prompt count
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
    const tps = if (elapsed_ms > 0) @as(u64, result.completion_tokens) * 1000 / elapsed_ms else 0;

    // Check for tool calls in the output
    if (has_tools) {
        if (try chat_mod.parseToolCalls(allocator, final_text)) |tool_calls| {
            defer {
                for (tool_calls) |tc| {
                    allocator.free(tc.name);
                    allocator.free(tc.arguments);
                }
                allocator.free(tool_calls);
            }

            log.info("  <- {d}+{d} tokens ({d}ms, ~{d} tok/s) [tool_calls: {d}]\n", .{
                result.prompt_tokens, result.completion_tokens, elapsed_ms, tps, tool_calls.len,
            });

            // Build tool_calls JSON array
            var tc_buf = std.ArrayList(u8).empty;
            defer tc_buf.deinit(allocator);
            try tc_buf.appendSlice(allocator, "[");
            for (tool_calls, 0..) |tc, i| {
                if (i > 0) try tc_buf.appendSlice(allocator, ",");
                const tc_id = try std.fmt.allocPrint(allocator, "call_{d}_{d}", .{ std.time.milliTimestamp(), i });
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

            const response = try std.fmt.allocPrint(allocator,
                \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","system_fingerprint":"mlx-serve","choices":[{{"index":0,"message":{{"role":"assistant","content":null,"tool_calls":{s}}},"finish_reason":"tool_calls"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
            , .{
                std.time.milliTimestamp(),
                std.time.timestamp(),
                model_name,
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

    log.info("  <- {d}+{d} tokens ({d}ms, ~{d} tok/s) [{s}]\n", .{
        result.prompt_tokens, result.completion_tokens, elapsed_ms, tps, finish_reason,
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
        std.time.milliTimestamp(),
        std.time.timestamp(),
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

fn handleStreamingGeneration(
    allocator: std.mem.Allocator,
    stream: std.net.Stream,
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
) !void {
    const chat_id = std.time.milliTimestamp();
    var timer = try std.time.Timer.start();

    // Prefill + init generator
    var gen = try Generator.init(allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids);
    gen.timeout_ns = getTimeoutNs();
    gen.logprobs_n = logprobs_n;
    defer gen.deinit(allocator);

    // Send SSE headers (no Content-Length — we stream until done)
    const header =
        "HTTP/1.1 200 OK\r\n" ++
        "Content-Type: text/event-stream\r\n" ++
        "Cache-Control: no-cache\r\n" ++
        "Connection: keep-alive\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n" ++
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" ++
        "\r\n";
    try stream.writeAll(header);

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
    var in_think_block = enable_thinking; // starts true when thinking enabled (model outputs <think> first)
    var think_buf = std.ArrayList(u8).empty; // buffer to detect </think> across token boundaries
    defer think_buf.deinit(allocator);
    const think_close_tag = "</think>";
    var skipped_think_open = false; // track if we've skipped the initial <think> tag
    var think_tokens: i32 = 0; // count of tokens generated in <think> block
    var budget_exhausted = false; // true when reasoning budget hit

    // Generate tokens
    while (try gen.next(allocator)) |token_id| {
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
            // Buffer tokens — we'll emit them after generation if no tool calls found
            try token_texts.append(allocator, token_text);
        } else if (enable_thinking and in_think_block) {
            // Inside <think> block — stream as reasoning_content with </think> detection
            defer allocator.free(token_text);
            try think_buf.appendSlice(allocator, token_text);
            think_tokens += 1;

            // Skip the initial <think>\n prefix
            if (!skipped_think_open and think_buf.items.len >= 7) {
                if (std.mem.startsWith(u8, think_buf.items, "<think>")) {
                    // Remove <think> prefix and any leading newline
                    var skip: usize = 7;
                    while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                }
                skipped_think_open = true;
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

            // Check for </think> in buffer
            if (std.mem.indexOf(u8, think_buf.items, think_close_tag)) |close_pos| {
                // Flush reasoning before </think>
                if (close_pos > 0) {
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = think_buf.items[0..close_pos] }, null, null);
                }
                // Content after </think>
                const after = close_pos + think_close_tag.len;
                const content_after = std.mem.trimLeft(u8, think_buf.items[after..], "\n ");
                if (content_after.len > 0) {
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = content_after }, null, null);
                }
                think_buf.clearRetainingCapacity();
                in_think_block = false;
            } else if (skipped_think_open) {
                // Flush reasoning tokens that can't be part of </think>
                // Keep last (think_close_tag.len - 1) bytes as they might be a partial tag
                const safe_len = if (think_buf.items.len > think_close_tag.len - 1)
                    think_buf.items.len - (think_close_tag.len - 1)
                else
                    0;
                if (safe_len > 0) {
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = think_buf.items[0..safe_len] }, null, null);
                    // Shift remaining bytes to front
                    const remaining = try allocator.dupe(u8, think_buf.items[safe_len..]);
                    think_buf.clearRetainingCapacity();
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                }
            }
        } else {
            defer allocator.free(token_text);
            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = token_text }, null, null);
        }
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
                const think_split = chat_mod.splitThinkBlock(text_buf.items, false);
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
                const first_delta = try std.fmt.allocPrint(allocator,
                    \\[{{"index":{d},"id":"{s}","type":"function","function":{{"name":"{s}","arguments":""}}}}]
                , .{ i, tc_id, tc.name });
                defer allocator.free(first_delta);
                try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .tool_calls_json = first_delta }, null, null);

                const chunk_size: usize = 20;
                var arg_pos: usize = 0;
                while (arg_pos < tc.arguments.len) {
                    const end = @min(arg_pos + chunk_size, tc.arguments.len);
                    const arg_chunk = tc.arguments[arg_pos..end];
                    const escaped_chunk = try jsonEscape(allocator, arg_chunk);
                    defer allocator.free(escaped_chunk);
                    const inner = if (escaped_chunk.len >= 2 and escaped_chunk[0] == '"')
                        escaped_chunk[1 .. escaped_chunk.len - 1]
                    else
                        escaped_chunk;
                    const arg_delta = try std.fmt.allocPrint(allocator,
                        \\[{{"index":{d},"function":{{"arguments":"{s}"}}}}]
                    , .{ i, inner });
                    defer allocator.free(arg_delta);
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .tool_calls_json = arg_delta }, null, null);
                    arg_pos = end;
                }
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
                const think_split = chat_mod.splitThinkBlock(full_text.items, false);
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
    try stream.writeAll("data: [DONE]\n\n");

    const elapsed_ms = timer.read() / std.time.ns_per_ms;
    const tps = if (elapsed_ms > 0) @as(u64, gen.completion_tokens) * 1000 / elapsed_ms else 0;
    log.info("  <- {d}+{d} tokens streamed ({d}ms, ~{d} tok/s) [{s}]\n", .{
        total_prompt, gen.completion_tokens, elapsed_ms, tps, finish_reason,
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
    stream: std.net.Stream,
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
    , .{ chat_id, std.time.timestamp(), model_name, delta_buf.items, fr_str, usage_str });
    defer allocator.free(chunk);

    // Write as SSE event
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

fn reuseKVCache(allocator: std.mem.Allocator, xfm: *Transformer, prompt_ids: []const u32) !CacheResult {
    if (cached_prompt_ids) |cached| {
        // Find shared prefix length
        const max_shared = @min(cached.len, prompt_ids.len);
        var shared: usize = 0;
        while (shared < max_shared and cached[shared] == prompt_ids[shared]) {
            shared += 1;
        }

        // Always keep at least 1 token to process (generator needs at least 1 for prefill)
        if (shared >= prompt_ids.len) {
            shared = prompt_ids.len - 1;
        }

        if (shared > 0) {
            // Truncate KV cache to the shared prefix
            try xfm.cache.truncate(shared, xfm.s);
            log.debug("  [cache] reusing {d}/{d} tokens from previous prompt\n", .{ shared, prompt_ids.len });
            return .{
                .new_tokens = prompt_ids[shared..],
                .cached_tokens = @intCast(shared),
            };
        }
    }

    // No cache hit — reset completely
    _ = allocator;
    xfm.cache.deinit();
    xfm.cache = try transformer_mod.KVCache.init(xfm.cache.allocator, xfm.config.num_hidden_layers);
    return .{ .new_tokens = prompt_ids, .cached_tokens = 0 };
}

fn updateCachedPrompt(allocator: std.mem.Allocator, prompt_ids: []const u32) void {
    if (cached_prompt_ids) |old| {
        allocator.free(old);
    }
    cached_prompt_ids = allocator.dupe(u32, prompt_ids) catch null;
}

fn sendResponse(stream: std.net.Stream, status: []const u8, content_type: []const u8, body: []const u8) !void {
    var hdr_buf: [512]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, GET, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\n\r\n", .{
        status,
        content_type,
        body.len,
    }) catch return error.Overflow;
    try stream.writeAll(hdr);
    if (body.len > 0) try stream.writeAll(body);
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

fn sendErrorResponse(allocator: std.mem.Allocator, stream: std.net.Stream, status: []const u8, err_type: []const u8, message: []const u8, code: ?u32) !void {
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

test "getEffectiveContextLength falls back to model config" {
    const original = max_context_size;
    defer max_context_size = original;

    max_context_size = 0;
    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 32768;
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
