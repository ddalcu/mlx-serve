//! decodeTokens; tokenize/detokenize

const std = @import("std");
const tokenizer_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const chat_mod = @import("../chat/chat.zig");
const log = @import("../../sdk/util/log.zig");
const model_registry_mod = @import("../registry/registry.zig");
const Tokenizer = tokenizer_mod.Tokenizer;
const LoadedModel = model_registry_mod.LoadedModel;
const testing = std.testing;

const Conn = @import("../http/conn.zig").Conn;
const tokenVocabSize = @import("completions.zig").tokenVocabSize;
const parseTokenIds = @import("completions.zig").parseTokenIds;
const sendResponse = @import("../http/respond.zig").sendResponse;
const sendErrorResponse = @import("../http/errors.zig").sendErrorResponse;
const jsonEscape = @import("format.zig").jsonEscape;

/// Decode a slice of token IDs to bytes through the model's own vocabulary: a GGUF engine's, else the loaded
/// tokenizer. Used by the request handlers' streaming + final-decode paths.
pub fn decodeTokens(
    allocator: std.mem.Allocator,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    ids: []const u32,
    strip_leading_space: bool,
) ![]u8 {
    if (lm.engine) |e| if (e.caps().owns_tokenizer) return e.decode(allocator, ids);
    return tok.decode(allocator, ids, strip_leading_space);
}

/// Text -> ids through the model's own vocabulary: an embedded engine's GGUF
/// vocab, else the loaded BPE tokenizer (an engine model's `tok` is an empty stub).
/// `add_special` reaches llama.cpp only.
pub fn encodeText(allocator: std.mem.Allocator, lm: *const LoadedModel, tok: *const Tokenizer, text: []const u8, add_special: bool) ![]u32 {
    if (lm.engine) |e| if (e.caps().owns_tokenizer) return e.tokenize(allocator, text, add_special);
    return tok.encode(allocator, text);
}

pub fn handleTokenize(
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

    const ids = try encodeText(allocator, lm, tok, content.?, true);
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

pub fn handleDetokenize(
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

    const ids = try allocator.alloc(u32, tokens_val.array.items.len);
    defer allocator.free(ids);
    if (!parseTokenIds(tokens_val.array.items, tokenVocabSize(lm), ids)) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'tokens' must be integers inside the model's vocabulary", 400);
        return;
    }

    const text = try decodeTokens(allocator, lm, tok, ids, false);
    defer allocator.free(text);

    const result = try detokenizeResponseJson(allocator, text);
    defer allocator.free(result);

    log.debug("POST /detokenize -> {d} tokens -> {d} chars\n", .{ ids.len, text.len });
    try sendResponse(stream, "200 OK", "application/json", result);
}

/// `{"content": <text>}` with the text escaped by the SHARED escaper.
///
/// This handler used to hand-roll a five-character escape table and pass every
/// other byte through raw, so any token whose bytes are below 0x20 produced a
/// body no JSON parser accepts — found live detokenizing single ids
/// while cross-checking a vocabulary. Same class as the control-byte rule on
/// the chat render path: a decoded token is arbitrary bytes, so it goes through
/// `appendJsonString`, never a local switch.
fn detokenizeResponseJson(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);
    try result.appendSlice(allocator, "{\"content\":");
    try chat_mod.appendJsonString(allocator, &result, text);
    try result.append(allocator, '}');
    return result.toOwnedSlice(allocator);
}

/// OpenAI `n` (choice count): this is a single-choice engine and n>1 is
/// deliberately unimplemented (YAGNI — no agent client sends it). Silently
/// returning one choice for n=2 is the silent-no-op class llmprobe flags;
/// reject with an honest 400 instead. Absent, null, 1, or 1.0 all mean
/// "one choice" and pass.
/// jsonEscape with an OOM fallback to the literal `""`. Ownership is reported
/// by PROVENANCE (`owned`), never inferred from content — escaping an empty
/// string also yields `""` but that one is allocated (2-byte leak per
/// all-reasoning request under the old content-equality defer).
const EscapedText = struct { slice: []const u8, owned: bool };

pub fn jsonEscapeOrEmpty(allocator: std.mem.Allocator, text: []const u8) EscapedText {
    const s = jsonEscape(allocator, text) catch return .{ .slice = "\"\"", .owned = false };
    return .{ .slice = s, .owned = true };
}

// ── tests ──

test "jsonEscapeOrEmpty: escaping an empty string is OWNED (leak class 2026-07-19)" {
    // The old inline pattern decided ownership by CONTENT
    // (`if (!mem.eql(escaped, "\"\"")) free`), so a legitimately-allocated
    // escape of "" — every all-reasoning generation with empty visible
    // content — matched the OOM-fallback literal and leaked 2 bytes per
    // request. Ownership keys on provenance; std.testing.allocator fails
    // this test on any leak.
    const a = std.testing.allocator;
    const empty = jsonEscapeOrEmpty(a, "");
    try std.testing.expect(empty.owned);
    try std.testing.expectEqualStrings("\"\"", empty.slice);
    if (empty.owned) a.free(empty.slice);

    const text = jsonEscapeOrEmpty(a, "hi \"there\"");
    try std.testing.expect(text.owned);
    try std.testing.expectEqualStrings("\"hi \\\"there\\\"\"", text.slice);
    if (text.owned) a.free(text.slice);
}

test "detokenizeResponseJson escapes arbitrary token bytes (control-byte class)" {
    const allocator = testing.allocator;
    // A decoded token is arbitrary bytes. The old hand-rolled escaper covered
    // five characters and passed everything else through, so a single id whose
    // bytes are below 0x20 shipped a body no JSON parser accepts — live
    // while cross-checking a vocabulary id by id.
    const nasty = "a\x01b\x1fc\"d\\e\nf\tg";
    const out = try detokenizeResponseJson(allocator, nasty);
    defer allocator.free(out);
    for (out) |c| try testing.expect(c >= 0x20);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, out, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings(nasty, parsed.value.object.get("content").?.string);

    // Ordinary text is untouched apart from the quoting.
    const plain = try detokenizeResponseJson(allocator, "hello");
    defer allocator.free(plain);
    try testing.expectEqualStrings("{\"content\":\"hello\"}", plain);
}
