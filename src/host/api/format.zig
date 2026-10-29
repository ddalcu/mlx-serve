//! usage, timings, cachedFormatChat, jsonEscape

const std = @import("std");
const tokenizer_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const request_mod = @import("../../sdk/request/request.zig");
const chat_mod = @import("../chat/chat.zig");
const log = @import("../../sdk/util/log.zig");
const responses_mod = @import("responses/model.zig");
const tokenize_cache_mod = @import("../tokenizer/cache.zig");
const engine_prompt = @import("../engines/prompt.zig");
const Served = @import("../engines/served.zig").Served;
const model_registry_mod = @import("../registry/registry.zig");
const Tokenizer = tokenizer_mod.Tokenizer;
const LoadedModel = model_registry_mod.LoadedModel;
const ollama_mod = @import("ollama/translate.zig");
const testing = std.testing;

/// Returns the number of trailing bytes that form an incomplete UTF-8 sequence.
/// If the string ends with a complete codepoint (or is empty), returns 0.
pub fn utf8TrailingIncomplete(s: []const u8) usize {
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

/// Bytes of a streamed think buffer safe to emit now: hold back the last 9
/// bytes (the longest possible partial close tag) and cut on a UTF-8
/// boundary; the escaper turns a split character into U+FFFD.
pub fn thinkHoldBackLen(buf: []const u8) usize {
    const max_partial: usize = 9;
    var safe_len: usize = if (buf.len > max_partial) buf.len - max_partial else 0;
    if (safe_len > 0) safe_len -= utf8TrailingIncomplete(buf[0..safe_len]);
    return safe_len;
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
pub fn cachedFormatChat(
    allocator: std.mem.Allocator,
    io: std.Io,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    chat_config: *const chat_mod.ChatConfig,
    messages: []const chat_mod.Message,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
    reasoning_effort: ?[]const u8,
    /// Extend the trailing assistant message rather than answering after it.
    continue_final: bool,
) ![]u32 {
    const cache_ptr: ?*tokenize_cache_mod.TokenizeCache = if (lm.tokenize_cache) |*tc| tc else null;
    const key_opt: ?u64 = if (cache_ptr != null)
        tokenize_cache_mod.TokenizeCache.keyFor(messages, tools_json, tool_choice_instruction, enable_thinking, reasoning_effort, continue_final, chat_config.chat_template_kwargs)
    else
        null;
    if (cache_ptr) |cache| if (key_opt) |key| {
        if (try cache.get(io, key, allocator)) |cached| return cached;
    };
    // ds4 renders through the GGUF engine's own template, which never reads
    // the effort string — only the Jinja paths thread it.
    // Backstop only. Both surfaces consult `continuationRejectReason` before
    // they get here, because an error returned from this depth reaches no
    // handler that can answer it — it unwinds to `handleConnectionThread`,
    // which logs and closes the socket, so the client sees a dropped
    // connection instead of a status line. A NEW surface that wires
    // continuation without asking first gets that ugly failure rather than a
    // silently doubled assistant turn; the scan below is what stops it
    // shipping.
    const engine: ?*const Served = if (lm.engine) |e| (if (e.caps().owns_tokenizer) e else null) else null;
    if (continue_final and engine != null and engine.?.caps().owns_template) return error.ContinuationUnsupported;
    const ids = if (engine) |e|
        try engine_prompt.encodeChat(allocator, e, chat_config, messages, tools_json, tool_choice_instruction, enable_thinking, reasoning_effort, continue_final)
    else
        try chat_mod.formatChat(allocator, tok, messages, chat_config, tools_json, tool_choice_instruction, enable_thinking, reasoning_effort, continue_final);
    if (cache_ptr) |cache| if (key_opt) |key| {
        // Insert is best-effort; an OOM in the cache shouldn't fail the
        // request — the user already has their tokenized prompt.
        cache.put(io, key, ids) catch |err| {
            log.warn("[tokenize-cache] put failed: {s}\n", .{@errorName(err)});
        };
    };
    return ids;
}

/// The chat-completions `usage` object, shared by the non-stream, tool-call
/// and streaming-final-chunk emitters so the shape never drifts per path.
/// `prompt_tokens_details.cached_tokens` is ALWAYS present (0 when cold) —
/// OpenAI emits it unconditionally, and black-box conformance/cost tooling
/// (llmprobe) treats a missing field as "no prompt caching" even though the
/// engine caches. `extra_details` is a pre-formatted `,"key":{...}` fragment
/// (completion_tokens_details) or "".
pub fn formatChatUsage(
    allocator: std.mem.Allocator,
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_tokens: u32,
    extra_details: []const u8,
) ![]u8 {
    return try std.fmt.allocPrint(allocator,
        \\{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d},"prompt_tokens_details":{{"cached_tokens":{d}}}{s}}}
    , .{ prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, cached_tokens, extra_details });
}

pub fn formatAnthropicUsage(allocator: std.mem.Allocator, prompt_tokens: u32, completion_tokens: u32, cached_tokens: u32) ![]u8 {
    return try std.fmt.allocPrint(allocator,
        \\{{"input_tokens":{d},"output_tokens":{d},"cache_creation_input_tokens":0,"cache_read_input_tokens":{d}}}
    , .{ prompt_tokens -| cached_tokens, completion_tokens, @min(cached_tokens, prompt_tokens) });
}

pub fn formatTimingsObject(
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
    const p_tps = request_mod.prefillTokensPerSec(prompt_tokens, cached_tokens, prefill_ns);
    const d_tps = request_mod.tokensPerSec(completion_tokens, decode_ns);
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
pub fn formatPerfBracket(
    buf: []u8,
    prompt_tokens: u32,
    cached_tokens: u32,
    completion_tokens: u32,
    prefill_ns: u64,
    decode_ns: u64,
) []const u8 {
    const p_tps = request_mod.prefillTokensPerSec(prompt_tokens, cached_tokens, prefill_ns);
    const d_tps = request_mod.tokensPerSec(completion_tokens, decode_ns);
    const s = if (cached_tokens > 0)
        std.fmt.bufPrint(buf, "prefill: {d:.1} tok/s ({d} cached / {d} total), decode: {d:.1} tok/s", .{ p_tps, cached_tokens, prompt_tokens, d_tps })
    else
        std.fmt.bufPrint(buf, "prefill: {d:.1} tok/s, decode: {d:.1} tok/s", .{ p_tps, d_tps });
    return s catch buf[0..0];
}

/// THE JSON string escaper for every response body. Every string here is model bytes
/// somewhere and a token is a BPE fragment, so invalid UTF-8 is sanitized INSIDE the
/// escaper (one U+FFFD per bad sequence); valid input passes through byte for byte.
pub fn jsonEscape(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    if (std.unicode.utf8ValidateSlice(input)) return jsonEscapeValid(allocator, input);
    const clean = try chat_mod.utf8Sanitize(allocator, input);
    defer allocator.free(clean);
    return jsonEscapeValid(allocator, clean);
}

fn jsonEscapeValid(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
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

// ── tests ──

test "EVERY JSON string escaper survives bytes that are not valid UTF-8" {
    // Class guard over the four escapers (OpenAI/Anthropic, Responses, the prompt
    // render, Ollama NDJSON): one passing bytes through loses a whole reply.
    const a = std.testing.allocator;
    const cases = [_][]const u8{
        "near\xe8\x91\xe4\xb8\x89 in", // 3-byte lead, one continuation, then another lead
        "\xff\xfe", // never-valid lead bytes
        "tail\xf0\x9f", // first half of a 4-byte emoji, cut by max_tokens
    };
    for (cases) |bad| {
        const a1 = try jsonEscape(a, bad);
        defer a.free(a1);
        try std.testing.expect(std.unicode.utf8ValidateSlice(a1));

        const a2 = try responses_mod.jsonEscape(a, bad);
        defer a.free(a2);
        try std.testing.expect(std.unicode.utf8ValidateSlice(a2));

        var buf = std.ArrayList(u8).empty;
        defer buf.deinit(a);
        try chat_mod.appendJsonString(a, &buf, bad);
        try std.testing.expect(std.unicode.utf8ValidateSlice(buf.items));

        var out: std.Io.Writer.Allocating = .init(a);
        defer out.deinit();
        try ollama_mod.writeJsonString(&out.writer, bad);
        try std.testing.expect(std.unicode.utf8ValidateSlice(out.written()));
    }
    // Valid input is passed through byte for byte.
    const good = try jsonEscape(a, "héllo \u{1F600}");
    defer a.free(good);
    try std.testing.expectEqualStrings("\"héllo \u{1F600}\"", good);
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

test "responses think hold-back never cuts a multibyte character" {
    // A CJK thought replayed token-by-token through the hold-back: every emitted
    // prefix must be valid UTF-8, and the stream must reassemble byte-exact.
    const allocator = testing.allocator;
    const tokens = [_][]const u8{
        "\xE7\x94\xA8",
        "\xE6\x88\xB7\xE9\x97\xAE",
        "\xE7\x9A\x84 9.11",
        "\xE8\xBF\x98\xE6\x98\xAF 9.9",
        "\xEF\xBC\x8C\xE5\x85\x88\xE6\xAF\x94",
        "\xE6\x95\xB4\xE6\x95\xB0",
        "\xEF\xBC\x8C\xE5\x86\x8D",
        "<|im_",
        "end|>",
    };
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);
    var reconstituted = std.ArrayList(u8).empty;
    defer reconstituted.deinit(allocator);
    for (tokens) |tok| {
        try buf.appendSlice(allocator, tok);
        const safe_len = thinkHoldBackLen(buf.items);
        if (safe_len == 0) continue;
        // The emitted prefix must be valid UTF-8 standing alone.
        try testing.expect(std.unicode.utf8ValidateSlice(buf.items[0..safe_len]));
        try reconstituted.appendSlice(allocator, buf.items[0..safe_len]);
        const remaining = try allocator.dupe(u8, buf.items[safe_len..]);
        defer allocator.free(remaining);
        buf.clearRetainingCapacity();
        try buf.appendSlice(allocator, remaining);
    }
    try reconstituted.appendSlice(allocator, buf.items);
    const want = "\xE7\x94\xA8\xE6\x88\xB7\xE9\x97\xAE\xE7\x9A\x84 9.11\xE8\xBF\x98\xE6\x98\xAF 9.9\xEF\xBC\x8C\xE5\x85\x88\xE6\xAF\x94\xE6\x95\xB4\xE6\x95\xB0\xEF\xBC\x8C\xE5\x86\x8D<|im_end|>";
    try testing.expectEqualStrings(want, reconstituted.items);
}

test "formatChatUsage: prompt_tokens_details.cached_tokens always present (llmprobe chat caching)" {
    const t = std.testing;
    const a = t.allocator;

    // Cold request: field present with 0, matching OpenAI's unconditional emission.
    const cold = try formatChatUsage(a, 100, 5, 0, "");
    defer a.free(cold);
    try t.expectEqualStrings(
        \\{"prompt_tokens":100,"completion_tokens":5,"total_tokens":105,"prompt_tokens_details":{"cached_tokens":0}}
    , cold);

    // Warm request: cached count surfaces; cached ≤ prompt by construction
    // (scheduler folds cached into prompt_tokens).
    const warm = try formatChatUsage(a, 300, 12, 250, "");
    defer a.free(warm);
    try t.expectEqualStrings(
        \\{"prompt_tokens":300,"completion_tokens":12,"total_tokens":312,"prompt_tokens_details":{"cached_tokens":250}}
    , warm);

    // Reasoning responses append completion_tokens_details after the prompt details.
    const with_details = try formatChatUsage(a, 10, 20, 4, ",\"completion_tokens_details\":{\"reasoning_tokens\":7}");
    defer a.free(with_details);
    try t.expectEqualStrings(
        \\{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":7}}
    , with_details);
}

test "Anthropic usage: cached input is a separate bucket, unlike OpenAI" {
    const a = testing.allocator;
    for ([_]struct { prompt: u32, cached: u32, input: u32 }{
        .{ .prompt = 100, .cached = 0, .input = 100 },
        .{ .prompt = 100, .cached = 40, .input = 60 },
        .{ .prompt = 92934, .cached = 91863, .input = 1071 },
        .{ .prompt = 100, .cached = 100, .input = 0 },
        .{ .prompt = 100, .cached = 101, .input = 0 },
    }) |case| {
        const json = try formatAnthropicUsage(a, case.prompt, 126, case.cached);
        defer a.free(json);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer parsed.deinit();
        const usage = parsed.value.object;
        const input = usage.get("input_tokens").?.integer;
        const created = if (usage.get("cache_creation_input_tokens")) |v| v.integer else 0;
        const cached = usage.get("cache_read_input_tokens").?.integer;
        try testing.expectEqual(@as(i64, case.prompt), input + created + cached);
        try testing.expectEqual(@as(i64, case.input), input);
        try testing.expectEqual(@as(i64, 0), created);
        try testing.expectEqual(@as(i64, @min(case.cached, case.prompt)), cached);
        try testing.expectEqual(@as(i64, 126), usage.get("output_tokens").?.integer);

        const openai = try formatChatUsage(a, case.prompt, 126, case.cached, "");
        defer a.free(openai);
        const chat_usage = try std.json.parseFromSlice(std.json.Value, a, openai, .{});
        defer chat_usage.deinit();
        const obj = chat_usage.value.object;
        try testing.expectEqual(@as(i64, case.prompt), obj.get("prompt_tokens").?.integer);
        try testing.expectEqual(@as(i64, case.prompt) + 126, obj.get("total_tokens").?.integer);
        try testing.expectEqual(@as(i64, case.cached), obj.get("prompt_tokens_details").?.object.get("cached_tokens").?.integer);
    }
}
