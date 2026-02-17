const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const tokenizer_mod = @import("tokenizer.zig");
const generate_mod = @import("generate.zig");
const chat_mod = @import("chat.zig");
const model_mod = @import("model.zig");

const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const Generator = generate_mod.Generator;

/// Start the HTTP server on the given port.
pub fn serve(
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    tok: *const Tokenizer,
    chat_config: *const chat_mod.ChatConfig,
    config: *const model_mod.ModelConfig,
    port: u16,
) !void {
    const addr = std.net.Address.initIp4(.{ 0, 0, 0, 0 }, port);
    var server = try addr.listen(.{ .reuse_address = true });
    defer server.deinit();

    std.debug.print("\nServer listening on http://0.0.0.0:{d}\n", .{port});
    std.debug.print("  POST /v1/chat/completions\n", .{});
    std.debug.print("  POST /v1/chat/completions  (stream=true)\n", .{});
    std.debug.print("  GET  /v1/models\n\n", .{});

    while (true) {
        const conn = try server.accept();
        defer conn.stream.close();
        handleConnection(allocator, conn.stream, xfm, tok, chat_config, config) catch |err| {
            switch (err) {
                error.BrokenPipe, error.ConnectionResetByPeer => {
                    std.debug.print("  -> client disconnected\n", .{});
                },
                else => {
                    std.debug.print("  -> error: {}\n", .{err});
                },
            }
        };
    }
}

fn handleConnection(
    allocator: std.mem.Allocator,
    stream: std.net.Stream,
    xfm: *Transformer,
    tok: *const Tokenizer,
    chat_config: *const chat_mod.ChatConfig,
    config: *const model_mod.ModelConfig,
) !void {
    // Read the full HTTP request
    var buf: [65536]u8 = undefined;
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

    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/v1/models")) {
        std.debug.print("GET  /v1/models -> 200\n", .{});
        try handleModels(allocator, stream, config);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/chat/completions")) {
        const header_end = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
        const body = request[header_end + 4 .. total_read];
        try handleChatCompletions(allocator, stream, body, xfm, tok, chat_config, config);
    } else if (std.mem.eql(u8, method, "OPTIONS")) {
        std.debug.print("OPTIONS {s} -> 204\n", .{path});
        try sendResponse(stream, "204 No Content", "text/plain", "");
    } else {
        std.debug.print("{s} {s} -> 404\n", .{ method, path });
        try sendResponse(stream, "404 Not Found", "application/json", "{\"error\":\"not found\"}");
    }
}

fn handleModels(allocator: std.mem.Allocator, stream: std.net.Stream, config: *const model_mod.ModelConfig) !void {
    const body = try std.fmt.allocPrint(allocator,
        \\{{"object":"list","data":[{{"id":"{s}","object":"model","owned_by":"mlx-serve"}}]}}
    , .{config.model_type});
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
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
        std.debug.print("POST /v1/chat/completions -> 400 (invalid JSON)\n", .{});
        try sendResponse(stream, "400 Bad Request", "application/json", "{\"error\":\"invalid JSON\"}");
        return;
    };
    defer parsed.deinit();

    const root = parsed.value.object;

    // Extract messages
    const messages_val = root.get("messages") orelse {
        std.debug.print("POST /v1/chat/completions -> 400 (missing messages)\n", .{});
        try sendResponse(stream, "400 Bad Request", "application/json", "{\"error\":\"missing messages\"}");
        return;
    };

    var messages = std.ArrayList(chat_mod.Message).empty;
    defer messages.deinit(allocator);

    for (messages_val.array.items) |msg_val| {
        const obj = msg_val.object;
        const role_val = obj.get("role") orelse continue;
        const content_val = obj.get("content") orelse continue;

        const content = switch (content_val) {
            .string => |s| s,
            .array => |arr| blk: {
                for (arr.items) |part| {
                    if (part != .object) continue;
                    const ptype = part.object.get("type") orelse continue;
                    if (ptype != .string or !std.mem.eql(u8, ptype.string, "text")) continue;
                    const text = part.object.get("text") orelse continue;
                    if (text == .string) break :blk text.string;
                }
                continue;
            },
            else => continue,
        };

        if (role_val != .string) continue;
        try messages.append(allocator, .{
            .role = role_val.string,
            .content = content,
        });
    }

    if (messages.items.len == 0) {
        std.debug.print("POST /v1/chat/completions -> 400 (no valid messages)\n", .{});
        try sendResponse(stream, "400 Bad Request", "application/json", "{\"error\":\"no valid messages\"}");
        return;
    }

    const max_tokens: u32 = if (root.get("max_tokens")) |v|
        switch (v) {
            .integer => |i| @intCast(i),
            else => 256,
        }
    else
        256;

    const is_stream = if (root.get("stream")) |v| v == .bool and v.bool else false;

    const temperature: f32 = if (root.get("temperature")) |v| switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => 1.0,
    } else 1.0;

    // Log the request
    const last_msg = messages.items[messages.items.len - 1];
    const preview_len = @min(last_msg.content.len, 80);
    std.debug.print("POST /v1/chat/completions ({d} msgs, max_tokens={d}, temp={d:.2}, stream={}) \n", .{ messages.items.len, max_tokens, temperature, is_stream });
    std.debug.print("  > \"{s}{s}\"\n", .{ last_msg.content[0..preview_len], if (last_msg.content.len > 80) "..." else "" });

    // Format chat template
    const prompt_ids = try chat_mod.formatChat(allocator, tok, messages.items, chat_config);
    defer allocator.free(prompt_ids);

    // Reset KV cache for new request
    xfm.cache.deinit();
    xfm.cache = try transformer_mod.KVCache.init(allocator, xfm.config.num_hidden_layers);

    const eos_slice = config.eosTokenSlice();

    if (is_stream) {
        handleStreamingGeneration(allocator, stream, xfm, tok, prompt_ids, max_tokens, temperature, eos_slice) catch |err| {
            std.debug.print("  -> streaming error: {}\n", .{err});
        };
    } else {
        handleNonStreamingGeneration(allocator, stream, xfm, tok, prompt_ids, max_tokens, temperature, eos_slice) catch |err| {
            std.debug.print("  -> 500 ({s})\n", .{@errorName(err)});
            const err_body = std.fmt.allocPrint(allocator, "{{\"error\":\"{s}\"}}", .{@errorName(err)}) catch return;
            defer allocator.free(err_body);
            try sendResponse(stream, "500 Internal Server Error", "application/json", err_body);
        };
    }
}

fn handleNonStreamingGeneration(
    allocator: std.mem.Allocator,
    stream: std.net.Stream,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    temperature: f32,
    eos_token_ids: []const u32,
) !void {
    var timer = try std.time.Timer.start();

    const result = try generate_mod.generate(allocator, xfm, tok, prompt_ids, max_tokens, temperature, eos_token_ids);
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);

    const elapsed_ms = timer.read() / std.time.ns_per_ms;
    const tps = if (elapsed_ms > 0) @as(u64, result.completion_tokens) * 1000 / elapsed_ms else 0;
    std.debug.print("  <- {d}+{d} tokens ({d}ms, ~{d} tok/s) [{s}]\n", .{
        result.prompt_tokens, result.completion_tokens, elapsed_ms, tps, result.finish_reason,
    });

    const escaped_text = jsonEscape(allocator, result.text) catch "\"\"";
    defer if (!std.mem.eql(u8, escaped_text, "\"\"")) allocator.free(escaped_text);

    const response = try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion","choices":[{{"index":0,"message":{{"role":"assistant","content":{s}}},"finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
    , .{
        std.time.milliTimestamp(),
        escaped_text,
        result.finish_reason,
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
    temperature: f32,
    eos_token_ids: []const u32,
) !void {
    const chat_id = std.time.milliTimestamp();
    var timer = try std.time.Timer.start();

    // Prefill + init generator
    var gen = try Generator.init(allocator, xfm, tok, prompt_ids, max_tokens, temperature, eos_token_ids);

    // Send SSE headers (no Content-Length — we stream until done)
    const header =
        "HTTP/1.1 200 OK\r\n" ++
        "Content-Type: text/event-stream\r\n" ++
        "Cache-Control: no-cache\r\n" ++
        "Connection: keep-alive\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n" ++
        "Access-Control-Allow-Headers: Content-Type\r\n" ++
        "\r\n";
    try stream.writeAll(header);

    // First chunk: role announcement
    try sendSSEChunk(allocator, stream, chat_id, .{ .role = "assistant", .content = "" }, null);

    // Generate tokens and stream each one
    while (try gen.next()) |token_id| {
        // Decode single token to text
        const strip = tok.tok_type == .sentencepiece_bpe;
        const token_text = try tok.decode(allocator, &[_]u32{token_id}, strip and false);
        defer allocator.free(token_text);

        try sendSSEChunk(allocator, stream, chat_id, .{ .role = null, .content = token_text }, null);
    }

    // Final chunk with finish_reason
    try sendSSEChunk(allocator, stream, chat_id, .{ .role = null, .content = null }, gen.finish_reason);

    // Done sentinel
    try stream.writeAll("data: [DONE]\n\n");

    const elapsed_ms = timer.read() / std.time.ns_per_ms;
    const tps = if (elapsed_ms > 0) @as(u64, gen.completion_tokens) * 1000 / elapsed_ms else 0;
    std.debug.print("  <- {d}+{d} tokens streamed ({d}ms, ~{d} tok/s) [{s}]\n", .{
        gen.prompt_tokens, gen.completion_tokens, elapsed_ms, tps, gen.finish_reason,
    });
}

const DeltaFields = struct {
    role: ?[]const u8,
    content: ?[]const u8,
};

fn sendSSEChunk(
    allocator: std.mem.Allocator,
    stream: std.net.Stream,
    chat_id: i64,
    delta: DeltaFields,
    finish_reason: ?[]const u8,
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
    }

    try delta_buf.appendSlice(allocator, "}");

    // Build the finish_reason field
    var fr_buf: [64]u8 = undefined;
    const fr_str = if (finish_reason) |fr|
        std.fmt.bufPrint(&fr_buf, "\"{s}\"", .{fr}) catch "null"
    else
        "null";

    // Build the full SSE chunk
    const chunk = try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{s},"finish_reason":{s}}}]}}
    , .{ chat_id, delta_buf.items, fr_str });
    defer allocator.free(chunk);

    // Write as SSE event
    var line_buf: [32]u8 = undefined;
    const prefix = std.fmt.bufPrint(&line_buf, "data: ", .{}) catch unreachable;
    try stream.writeAll(prefix);
    try stream.writeAll(chunk);
    try stream.writeAll("\n\n");
}

// ── Shared utilities ──

fn sendResponse(stream: std.net.Stream, status: []const u8, content_type: []const u8, body: []const u8) !void {
    var hdr_buf: [512]u8 = undefined;
    const hdr = std.fmt.bufPrint(&hdr_buf, "HTTP/1.1 {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, GET, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\n\r\n", .{
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
