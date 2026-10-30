//! continuation, joined text, cache key, echoModelName

const std = @import("std");
const chat_mod = @import("../chat/chat.zig");
const testing = std.testing;

/// `continue_final_message`: extend the trailing assistant message instead of
/// answering after it. vLLM's spelling, because a client that already knows how
/// to ask another local server for this should not have to learn a second name.
///
/// The flag alone is not enough — the conversation has to BE continuable
/// (`chat_mod.continuationRequested`). A flag set on a user-final request is
/// ignored rather than 400'd: the request is perfectly answerable as an
/// ordinary turn, and refusing it would break the obvious client that sets the
/// field once for the whole session.
pub fn continueFinalMessageRequested(root: std.json.ObjectMap, messages: []const chat_mod.Message) bool {
    const v = root.get("continue_final_message") orelse return false;
    if (v != .bool or !v.bool) return false;
    return chat_mod.continuationRequested(messages);
}

/// Why this model cannot serve a continuation, or null when it can.
///
/// ds4 renders through the embedded engine's OWN template, which lives inside
/// the engine and is unreachable from this process — there is nowhere to append
/// the partial reply, so a continuation there would silently render it as
/// history and open a second assistant turn.
///
/// ONE predicate, consulted by BOTH surfaces before they render, so a model
/// that cannot serve one is never handed to `cachedFormatChat`'s backstop —
/// which returns an error no handler can turn into a response, and therefore
/// hangs up the socket with no status line at all. What the two surfaces do
/// with the answer differs, and must: the OpenAI surface was ASKED and gets a
/// named 400, while `/v1/messages` infers the request from the message list and
/// falls back to an ordinary turn — refusing a request the client never made
/// would break every Anthropic SDK caller on an embedded-engine model.
pub fn continuationRejectReason(embedded_engine: bool) ?[]const u8 {
    if (embedded_engine) {
        return "continue_final_message is not supported by this model: its chat " ++
            "template is rendered inside the embedded GGUF engine, so there is " ++
            "nowhere to append the partial reply. Send the partial text as the " ++
            "end of your prompt instead.";
    }
    return null;
}

const JoinedText = struct { text: []const u8, owned: bool };

/// Collect every `{"type":"text"}` part of a content array into one string,
/// in order, '\n'-joined. A single (or no) text part borrows the JSON's
/// bytes (`owned=false`); 2+ parts allocate (`owned=true`, caller frees).
///
/// Multiple text parts are real traffic: opencode plan mode appends its
/// "Plan Mode - System Reminder" as a SECOND text part on the user message,
/// and last-wins here dropped the user's actual prompt — the model answered
/// "What would you like to accomplish?" to every first plan-mode message
/// (issue #195).
pub fn joinedTextParts(allocator: std.mem.Allocator, parts: []const std.json.Value) !JoinedText {
    var single: []const u8 = "";
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    var n: usize = 0;
    for (parts) |part| {
        if (part != .object) continue;
        const ptype = part.object.get("type") orelse continue;
        if (ptype != .string or !std.mem.eql(u8, ptype.string, "text")) continue;
        const tv = part.object.get("text") orelse continue;
        if (tv != .string or tv.string.len == 0) continue;
        n += 1;
        if (n == 1) {
            single = tv.string;
            continue;
        }
        if (n == 2) try buf.appendSlice(allocator, single);
        try buf.append(allocator, '\n');
        try buf.appendSlice(allocator, tv.string);
    }
    if (n <= 1) return .{ .text = single, .owned = false };
    return .{ .text = try buf.toOwnedSlice(allocator), .owned = true };
}

/// Workload key for hot-cache eviction (#378): `prompt_cache_key` (OpenAI's own
/// routing field) > `metadata.user_id` (Anthropic; Claude Code sends its session
/// id) > the system prompt (OpenAI `messages[0]`, Anthropic `system`, Responses
/// `instructions`) > 0 = anonymous. Text parts hash in order like `joinedTextParts`.
pub fn requestCacheKey(root: std.json.ObjectMap) u64 {
    if (root.get("prompt_cache_key")) |v| if (v == .string and v.string.len > 0)
        return std.hash.Wyhash.hash(1, v.string);
    if (root.get("metadata")) |mv| if (mv == .object) {
        if (mv.object.get("user_id")) |v| if (v == .string and v.string.len > 0)
            return std.hash.Wyhash.hash(2, v.string);
    };
    if (root.get("messages")) |mv| if (mv == .array and mv.array.items.len > 0) {
        const first = mv.array.items[0];
        if (first == .object) if (first.object.get("role")) |r| if (r == .string and std.mem.eql(u8, chat_mod.canonicalRole(r.string), "system")) {
            if (first.object.get("content")) |c| return hashTextValue(c) orelse 0;
        };
    };
    if (root.get("system")) |v| return hashTextValue(v) orelse 0;
    if (root.get("instructions")) |v| return hashTextValue(v) orelse 0;
    return 0;
}

fn hashTextValue(v: std.json.Value) ?u64 {
    var h = std.hash.Wyhash.init(3);
    var n: usize = 0;
    switch (v) {
        .string => |t| {
            if (t.len == 0) return null;
            h.update(t);
            n = 1;
        },
        .array => |parts| for (parts.items) |part| {
            if (part != .object) continue;
            const ptype = part.object.get("type") orelse continue;
            if (ptype != .string or !std.mem.eql(u8, ptype.string, "text")) continue;
            const tv = part.object.get("text") orelse continue;
            if (tv != .string or tv.string.len == 0) continue;
            if (n > 0) h.update("\n");
            h.update(tv.string);
            n += 1;
        },
        else => {},
    }
    return if (n == 0) null else h.final();
}

/// The `model` a response echoes: the client's own string, as SDKs expect, unless it cannot
/// sit verbatim inside a JSON string (the response templates splice it unescaped).
pub fn echoModelName(root: std.json.ObjectMap, fallback: []const u8) []const u8 {
    const v = root.get("model") orelse return fallback;
    if (v != .string or !std.unicode.utf8ValidateSlice(v.string)) return fallback;
    for (v.string) |c| if (c < 0x20 or c == '"' or c == '\\') return fallback;
    return v.string;
}

// ── tests ──

test "continuationRejectReason names the embedded engine, and only it" {
    // An MLX checkpoint renders through our own Jinja, so the prefill has
    // somewhere to go.
    try std.testing.expect(continuationRejectReason(false) == null);
    // ds4 does not, and the refusal has to SAY so — the reason is what a
    // client reads instead of the answer it asked for.
    const reason = continuationRejectReason(true) orelse return error.NoReason;
    try std.testing.expect(std.mem.indexOf(u8, reason, "continue_final_message") != null);
    try std.testing.expect(std.mem.indexOf(u8, reason, "embedded GGUF engine") != null);
}

test "content-array text parts JOIN in order — plan-mode's [prompt, reminder] shape (issue #195)" {
    // opencode plan mode appends its system-reminder as a SECOND text part on
    // the user message. Last-wins parsing kept only the reminder, so the model
    // saw no user prompt at all.
    const body =
        \\[{"type":"text","text":"analyze project"},
        \\ {"type":"image_url","image_url":{"url":"data:x"}},
        \\ {"type":"text","text":"<system-reminder>plan mode</system-reminder>"}]
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const j = try joinedTextParts(testing.allocator, parsed.value.array.items);
    defer if (j.owned) testing.allocator.free(j.text);
    try testing.expect(j.owned);
    try testing.expectEqualStrings("analyze project\n<system-reminder>plan mode</system-reminder>", j.text);
}

test "joinedTextParts: single text part borrows; empty and non-text parts ignored" {
    const body =
        \\[{"type":"text","text":""}, {"type":"input_audio"}, {"type":"text","text":"hello"}, {"nope":1}]
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const j = try joinedTextParts(testing.allocator, parsed.value.array.items);
    try testing.expect(!j.owned);
    try testing.expectEqualStrings("hello", j.text);

    const none = try joinedTextParts(testing.allocator, &.{});
    try testing.expect(!none.owned);
    try testing.expectEqualStrings("", none.text);
}

fn cacheKeyOf(body: []const u8) !u64 {
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    return requestCacheKey(parsed.value.object);
}

test "canonicalRole: developer is the system turn; every other role is itself" {
    try testing.expectEqualStrings("system", chat_mod.canonicalRole("developer"));
    try testing.expectEqualStrings("system", chat_mod.canonicalRole("system"));
    try testing.expectEqualStrings("tool", chat_mod.canonicalRole("tool"));
    try testing.expectEqualStrings("assistant", chat_mod.canonicalRole("assistant"));
}

test "requestCacheKey: prompt_cache_key > metadata.user_id > system prompt > anonymous" {
    const sys_str = try cacheKeyOf(
        \\{"messages":[{"role":"system","content":"You are S."},{"role":"user","content":"hi"}]}
    );
    const sys_parts = try cacheKeyOf(
        \\{"messages":[{"role":"system","content":[{"type":"text","text":"You are S."}]}]}
    );
    const anth = try cacheKeyOf(
        \\{"system":[{"type":"text","text":"You are S."}],"messages":[{"role":"user","content":"hi"}]}
    );
    const resp = try cacheKeyOf(
        \\{"instructions":"You are S.","input":"hi"}
    );
    try testing.expect(sys_str != 0);
    try testing.expectEqual(sys_str, sys_parts);
    try testing.expectEqual(sys_str, anth);
    try testing.expectEqual(sys_str, resp);
    // OpenAI's reasoning-model spelling of the system turn (pi sends it).
    try testing.expectEqual(sys_str, try cacheKeyOf(
        \\{"messages":[{"role":"developer","content":"You are S."},{"role":"user","content":"hi"}]}
    ));
    try testing.expect(sys_str != try cacheKeyOf(
        \\{"messages":[{"role":"system","content":"You are T."}]}
    ));

    const uid = try cacheKeyOf(
        \\{"metadata":{"user_id":"sess-1"},"messages":[{"role":"system","content":"You are S."}]}
    );
    try testing.expect(uid != 0 and uid != sys_str);
    const pck = try cacheKeyOf(
        \\{"prompt_cache_key":"batch","metadata":{"user_id":"sess-1"},"messages":[{"role":"system","content":"You are S."}]}
    );
    try testing.expect(pck != 0 and pck != uid);
    try testing.expectEqual(pck, try cacheKeyOf(
        \\{"prompt_cache_key":"batch","prompt":"doc"}
    ));

    try testing.expectEqual(@as(u64, 0), try cacheKeyOf(
        \\{"messages":[{"role":"user","content":"hi"}]}
    ));
    try testing.expectEqual(@as(u64, 0), try cacheKeyOf(
        \\{"prompt_cache_key":"","prompt":"doc"}
    ));
}

test "echoModelName: a client model name the templates cannot splice falls back to the model type" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, want: []const u8 }{
        .{ .body = "{\"model\":\"gpt-4\"}", .want = "gpt-4" },
        .{ .body = "{\"model\":\"org\\/repo\"}", .want = "org/repo" },
        .{ .body = "{\"model\":\"漢字\"}", .want = "漢字" },
        .{ .body = "{}", .want = "qwen3" },
        .{ .body = "{\"model\":5}", .want = "qwen3" },
        .{ .body = "{\"model\":\"a\\u0000b\"}", .want = "qwen3" },
        .{ .body = "{\"model\":\"a\\\"b\"}", .want = "qwen3" },
        .{ .body = "{\"model\":\"a\\\\b\"}", .want = "qwen3" },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        try std.testing.expectEqualStrings(case.want, echoModelName(parsed.value.object, "qwen3"));
    }
}
