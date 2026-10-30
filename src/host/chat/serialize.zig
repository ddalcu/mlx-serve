//! Messages and tool definitions serialized to the JSON the template reads, media content parts.

const std = @import("std");
const testing = std.testing;

const ToolCall = @import("types.zig").ToolCall;
const ImageData = @import("types.zig").ImageData;
const AudioData = @import("types.zig").AudioData;
const VideoData = @import("types.zig").VideoData;
const Message = @import("types.zig").Message;
const ChatConfig = @import("types.zig").ChatConfig;
const renderChatTemplate = @import("render.zig").renderChatTemplate;
const appendJsonString = @import("../../sdk/util/json_text.zig").appendJsonString;

/// How an assistant message with NO content should serialize.
///
/// Most templates read `message.content is none` / `if message.content`, and
/// `null` is what they expect. Harmony (gpt_oss) does BOTH of these in
/// sequence on every assistant message:
///
///     {%- if "content" in message %}
///         {%- if "<|channel|>analysis<|message|>" in message.content ... %}
///
/// The first test asks whether the KEY exists, the second indexes into the
/// VALUE — so `"content": null` passes the guard and then substring-searches
/// None. A tool-calling assistant turn is exactly the message that carries no
/// content, so every tool round-trip hit it (the reply came
/// back with reasoning and raw `<|channel|>final>` markers in the content).
/// Serializing `""` satisfies both readings — the key exists, and it is falsy
/// everywhere the template branches on it.
pub const EmptyContent = enum { null_literal, empty_string };

pub fn serializeMessagesJson(allocator: std.mem.Allocator, messages: []const Message) ![]const u8 {
    return serializeMessagesJsonOpts(allocator, messages, .null_literal);
}

pub fn serializeMessagesJsonOpts(allocator: std.mem.Allocator, messages: []const Message, empty_content: EmptyContent) ![]const u8 {
    return serializeMessagesJsonImpl(allocator, messages, empty_content, false, null, null);
}

/// K2-Horizon's template raises on an assistant turn with NO thinking field,
/// so that family gets an empty `reasoning_content` where a client sent none
/// (renders as the closed `<ifm|think>\n</ifm|think>` signature).
pub fn templateRequiresReasoningField(tpl: []const u8) bool {
    return std.mem.indexOf(u8, tpl, "'ifm|think'") != null;
}

pub fn serializeMessagesJsonFor(allocator: std.mem.Allocator, messages: []const Message, empty_content: EmptyContent, chat_config: *const ChatConfig) ![]const u8 {
    return serializeMessagesJsonImpl(allocator, messages, empty_content, templateRequiresReasoningField(chat_config.chat_template), chat_config.video_placeholder, chat_config.image_placeholder);
}

pub fn messageHasMedia(msg: Message) bool {
    return (msg.images != null and msg.images.?.len > 0) or
        (msg.videos != null and msg.videos.?.len > 0) or
        (msg.audio != null and msg.audio.?.len > 0);
}

/// An LFM2-VL tiled source is several `ImageData` entries (tiles + thumbnail);
/// the template renders ONE placeholder for the source, at its first piece.
pub fn isImageItemStart(img: ImageData) bool {
    return img.tile_rows == 0 or img.tile_index == 0;
}

/// Media as content parts, so the model's own template renders one
/// placeholder per item where the message sits: images, videos, audio (the
/// order the server encodes them), then the text.
/// `content` as one string: an `atom` per image, then the text, all joined by blank lines (DeepSeek's
/// `process_image_messages`). Video and audio have no atom on such a template.
fn appendImageAtomsText(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), msg: Message, atom: []const u8) !void {
    var text = std.ArrayList(u8).empty;
    defer text.deinit(allocator);
    for (msg.images orelse &[_]ImageData{}) |img| {
        if (!isImageItemStart(img)) continue;
        if (text.items.len > 0) try text.appendSlice(allocator, "\n\n");
        try text.appendSlice(allocator, atom);
    }
    if (msg.content.len > 0) {
        if (text.items.len > 0) try text.appendSlice(allocator, "\n\n");
        try text.appendSlice(allocator, msg.content);
    }
    try appendJsonString(allocator, buf, text.items);
}

fn appendMediaContentParts(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), msg: Message, video_placeholder: ?[]const u8) !void {
    try buf.append(allocator, '[');
    var first = true;
    for (msg.images orelse &[_]ImageData{}) |img| {
        if (!isImageItemStart(img)) continue;
        if (!first) try buf.append(allocator, ',');
        first = false;
        try buf.appendSlice(allocator, "{\"type\":\"image\"}");
    }
    for (msg.videos orelse &[_]VideoData{}) |_| {
        if (!first) try buf.append(allocator, ',');
        first = false;
        if (video_placeholder) |atom| {
            try buf.appendSlice(allocator, "{\"type\":\"text\",\"text\":");
            try appendJsonString(allocator, buf, atom);
            try buf.append(allocator, '}');
        } else try buf.appendSlice(allocator, "{\"type\":\"video\"}");
    }
    for (msg.audio orelse &[_]AudioData{}) |_| {
        if (!first) try buf.append(allocator, ',');
        first = false;
        try buf.appendSlice(allocator, "{\"type\":\"audio\"}");
    }
    if (msg.content.len > 0) {
        try buf.appendSlice(allocator, ",{\"type\":\"text\",\"text\":");
        try appendJsonString(allocator, buf, msg.content);
        try buf.append(allocator, '}');
    }
    try buf.append(allocator, ']');
}

fn serializeMessagesJsonImpl(allocator: std.mem.Allocator, messages: []const Message, empty_content: EmptyContent, reasoning_required: bool, video_placeholder: ?[]const u8, image_placeholder: ?[]const u8) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);

    try buf.append(allocator, '[');
    for (messages, 0..) |msg, i| {
        if (i > 0) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, "{\"role\":");
        try appendJsonString(allocator, &buf, msg.role);

        try buf.appendSlice(allocator, ",\"content\":");
        if (messageHasMedia(msg)) {
            if (image_placeholder) |atom| try appendImageAtomsText(allocator, &buf, msg, atom) else try appendMediaContentParts(allocator, &buf, msg, video_placeholder);
        } else if (msg.content.len > 0) {
            try appendJsonString(allocator, &buf, msg.content);
        } else switch (empty_content) {
            .null_literal => try buf.appendSlice(allocator, "null"),
            .empty_string => try buf.appendSlice(allocator, "\"\""),
        }

        if (msg.tool_calls) |tcs| {
            try buf.appendSlice(allocator, ",\"tool_calls\":[");
            for (tcs, 0..) |tc, ti| {
                if (ti > 0) try buf.append(allocator, ',');
                try buf.appendSlice(allocator, "{\"type\":\"function\",");
                if (tc.id.len > 0) {
                    // Inkling's template names a tool RESULT by matching
                    // `message.tool_call_id` against the history call's `id`
                    // (`tc.id is defined and tc.id == message.tool_call_id`) —
                    // without the id the result renders nameless. Inert for
                    // templates that never read it.
                    try buf.appendSlice(allocator, "\"id\":");
                    try appendJsonString(allocator, &buf, tc.id);
                    try buf.append(allocator, ',');
                }
                try buf.appendSlice(allocator, "\"function\":{\"name\":");
                try appendJsonString(allocator, &buf, tc.name);
                try buf.appendSlice(allocator, ",\"arguments\":");
                // Embed arguments as a JSON OBJECT when it parses cleanly — Qwen
                // 3.5/3.6 templates do `tool_call.arguments|items` which requires
                // a dict. Templates that need a string (e.g. Gemma 4 with `tojson`
                // or `string`) still get a usable value. Falls back to a string
                // for malformed arguments so downstream still sees something.
                if (std.json.parseFromSlice(std.json.Value, allocator, tc.arguments, .{})) |parsed| {
                    defer parsed.deinit();
                    if (parsed.value == .object) {
                        try buf.appendSlice(allocator, tc.arguments);
                    } else {
                        try appendJsonString(allocator, &buf, tc.arguments);
                    }
                } else |_| {
                    try appendJsonString(allocator, &buf, tc.arguments);
                }
                try buf.appendSlice(allocator, "}}");
            }
            try buf.append(allocator, ']');
        }

        if (msg.tool_call_id) |tid| {
            try buf.appendSlice(allocator, ",\"tool_call_id\":");
            try appendJsonString(allocator, &buf, tid);
            // No tool_responses field needed — Gemma 4 templates handle role:"tool"
            // natively via the content field. Adding tool_responses causes the template
            // to render duplicate content, wasting tokens.
        }

        if (msg.reasoning_content) |rc| {
            // Key omitted entirely when absent: templates gate on
            // `message.reasoning_content is string`, and an explicit null
            // would still pass `is defined`-style checks.
            try buf.appendSlice(allocator, ",\"reasoning_content\":");
            try appendJsonString(allocator, &buf, rc);
        } else if (reasoning_required and std.mem.eql(u8, msg.role, "assistant")) {
            try buf.appendSlice(allocator, ",\"reasoning_content\":\"\"");
        }

        try buf.append(allocator, '}');
    }
    try buf.append(allocator, ']');

    return buf.toOwnedSlice(allocator);
}

/// Fill the OPTIONAL tool-definition keys a template may dump blindly.
///
/// `description` is optional in OpenAI's function schema and a no-arg tool
/// needs no `parameters`, but a template that renders `fn.description | tojson`
/// (muse) or `tool | tojson` wholesale hits `tojson` on Undefined, which is a
/// jinja RAISE — i.e. a silent `fallbackFormatChat` with no tools preamble at
/// all, for a request that was perfectly legal. Returns null when every
/// function already carries both keys, so the common path re-emits nothing and
/// the caller keeps the client's own bytes (provenance, not content, decides
/// ownership).
///
/// Defaults are the empty ones: `""` says the client supplied no description,
/// and an empty-properties object is what a no-argument tool means. Neither
/// invents a capability the client did not declare.
pub fn fillOptionalToolDefKeys(allocator: std.mem.Allocator, tools_json: []const u8) !?[]u8 {
    // Own the intermediates: growing a parsed ObjectMap allocates OUTSIDE the
    // parse's arena, so mutating with the caller's allocator leaks unless the
    // caller happens to be an arena too. Only the returned bytes are theirs.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const parsed = std.json.parseFromSlice(std.json.Value, a, tools_json, .{}) catch return null;
    if (parsed.value != .array) return null;

    var changed = false;
    for (parsed.value.array.items) |*tool| {
        if (tool.* != .object) continue;
        const fn_val = tool.object.getPtr("function") orelse continue;
        if (fn_val.* != .object) continue;
        if (fn_val.object.get("description") == null) {
            try fn_val.object.put(a, "description", .{ .string = "" });
            changed = true;
        }
        if (fn_val.object.get("parameters") == null) {
            var params: std.json.ObjectMap = .empty;
            try params.put(a, "type", .{ .string = "object" });
            try params.put(a, "properties", .{ .object = .empty });
            try fn_val.object.put(a, "parameters", .{ .object = params });
            changed = true;
        }
    }
    if (!changed) return null;

    return try std.json.Stringify.valueAlloc(allocator, parsed.value, .{});
}

// ── tests ──

test "renderChatTemplate: a tool with no description/parameters still renders (no silent fallback)" {
    // OpenAI's schema makes `description` optional and a no-arg tool needs no
    // `parameters`, but muse's template dumps all three through `tojson`, and
    // `tojson` RAISES on Undefined — so a description-less tool killed the
    // render and fell back to ChatML with no tools preamble at all (found
    // by a sweep whose own test tool omitted the key). Every family
    // that dumps `tool | tojson` wholesale is one missing key from the same
    // class, so the defaults are filled for ALL templates, not just this one.
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = @embedFile("../../fixtures/muse_chat_template.jinja"),
        .bos_token = null,
        .eos_token = "<|eot|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{.{ .role = "user", .content = "hi" }};

    // No `description`, no `parameters` — both legal over the wire.
    const bare =
        \\[{"type":"function","function":{"name":"note"}}]
    ;
    const rendered = try renderChatTemplate(allocator, &messages, &config, bare, null, true, null, false);
    defer allocator.free(rendered);
    // The muse tools preamble proves the REAL template ran; the fallback emits
    // ChatML markers instead and would carry no tools text at all.
    try testing.expect(std.mem.indexOf(u8, rendered, "In this environment you have access to a set of tools") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "\"name\": \"note\"") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "<|im_start|>") == null);

    // A COMPLETE declaration re-emits nothing: null means the caller keeps the
    // client's own bytes, so no request that works today is re-serialized
    // (key order, number formatting and unicode escaping all stay theirs).
    const complete =
        \\[{"type":"function","function":{"name":"note","description":"d","parameters":{"type":"object","properties":{}}}}]
    ;
    try testing.expect(try fillOptionalToolDefKeys(allocator, complete) == null);
    // Only the missing keys are added, and only to the function that lacks them.
    const partial =
        \\[{"type":"function","function":{"name":"a","description":"d"}},{"type":"function","function":{"name":"b","description":"e","parameters":{"type":"object"}}}]
    ;
    const filled = (try fillOptionalToolDefKeys(allocator, partial)).?;
    defer allocator.free(filled);
    try testing.expect(std.mem.indexOf(u8, filled, "\"parameters\"") != null);
    try testing.expect(std.mem.indexOf(u8, filled, "\"description\":\"d\"") != null);
    try testing.expect(std.mem.indexOf(u8, filled, "\"description\":\"e\"") != null);
}

test "serializeMessagesJson basic" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "Hello" },
        .{ .role = "assistant", .content = "Hi there" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Parse it back to verify valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 2), parsed.value.array.items.len);
    try testing.expectEqualStrings("user", parsed.value.array.items[0].object.get("role").?.string);
    try testing.expectEqualStrings("Hello", parsed.value.array.items[0].object.get("content").?.string);
}

test "serializeMessagesJson: a string-content template gets each image as its atom before the text" {
    const a = std.testing.allocator;
    var imgs = [_]ImageData{ .{ .pixels = &.{}, .width = 1, .height = 1 }, .{ .pixels = &.{}, .width = 1, .height = 1 } };
    const msgs = [_]Message{.{ .role = "user", .content = "What differs?", .images = &imgs }};
    var cfg = ChatConfig{ .chat_template = "", .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = a, .image_placeholder = "<IMG>" };
    const json = try serializeMessagesJsonFor(a, &msgs, .null_literal, &cfg);
    defer a.free(json);
    try std.testing.expectEqualStrings("[{\"role\":\"user\",\"content\":\"<IMG>\\n\\n<IMG>\\n\\nWhat differs?\"}]", json);
    // Without the atom the same message renders as content parts.
    cfg.image_placeholder = null;
    const parts = try serializeMessagesJsonFor(a, &msgs, .null_literal, &cfg);
    defer a.free(parts);
    try std.testing.expect(std.mem.indexOf(u8, parts, "{\"type\":\"image\"}") != null);
}

test "serializeMessagesJson with tool_calls" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "get_weather", .arguments = "{\"location\":\"Tokyo\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Should contain tool_calls array
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_calls\"") != null);
    try testing.expect(std.mem.indexOf(u8, result, "get_weather") != null);
}

test "serializeMessagesJson carries assistant reasoning_content" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "What is 2+2?" },
        .{ .role = "assistant", .content = "4", .reasoning_content = "two plus two is four" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    const asst = parsed.value.array.items[1].object;
    try testing.expectEqualStrings("two plus two is four", asst.get("reasoning_content").?.string);
    // A message without the field must not carry the key at all — templates
    // gate on `is string`, and an explicit null would still pass `is defined`.
    try testing.expect(parsed.value.array.items[0].object.get("reasoning_content") == null);
}

test "serializeMessagesJson embeds valid-JSON arguments as object (not string)" {
    // Required by Qwen 3.5/3.6 templates that do `tool_call.arguments|items`.
    // Without this, the Jinja `items` filter fails on a string and the server
    // falls back to ChatML — losing the `<think>\n` injection that primes the
    // model's reasoning + close-tag behavior on the next turn.
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"date\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Object form: ..."arguments":{"command":"date"}...
    // String form (rejected): ..."arguments":"{\"command\":\"date\"}"...
    try testing.expect(std.mem.indexOf(u8, result, "\"arguments\":{\"command\":\"date\"}") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"arguments\":\"{") == null);
}

test "serializeMessagesJson keeps malformed arguments as string" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "not valid json" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);
    try testing.expect(std.mem.indexOf(u8, result, "\"arguments\":\"not valid json\"") != null);
}

test "serializeMessagesJson tool response has tool_call_id and content" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"ls\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "file1.txt\nfile2.txt", .tool_call_id = "call_1" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Must contain tool_call_id
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_call_id\"") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"call_1\"") != null);
    // Must contain the response content
    try testing.expect(std.mem.indexOf(u8, result, "file1.txt") != null);
    // Must NOT contain tool_responses (templates handle role:"tool" natively)
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_responses\"") == null);
}

test "serializeMessagesJson full tool calling conversation" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_99_0", .name = "shell", .arguments = "{\"command\":\"echo hello\"}" },
    };
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "Run echo hello" },
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "hello", .tool_call_id = "call_99_0" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 4), parsed.value.array.items.len);

    // Verify roles
    try testing.expectEqualStrings("system", parsed.value.array.items[0].object.get("role").?.string);
    try testing.expectEqualStrings("user", parsed.value.array.items[1].object.get("role").?.string);
    try testing.expectEqualStrings("assistant", parsed.value.array.items[2].object.get("role").?.string);
    try testing.expectEqualStrings("tool", parsed.value.array.items[3].object.get("role").?.string);

    // Verify assistant has tool_calls
    const assistant = parsed.value.array.items[2].object;
    try testing.expect(assistant.get("tool_calls") != null);

    // Verify tool message has tool_call_id and content, no tool_responses
    const tool_msg = parsed.value.array.items[3].object;
    try testing.expectEqualStrings("call_99_0", tool_msg.get("tool_call_id").?.string);
    try testing.expectEqualStrings("hello", tool_msg.get("content").?.string);
    try testing.expect(tool_msg.get("tool_responses") == null);
}

test "serializeMessagesJson multiple parallel tool responses" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "readFile", .arguments = "{\"path\":\"a.txt\"}" },
        .{ .id = "call_2", .name = "shell", .arguments = "{\"command\":\"date\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "file content", .tool_call_id = "call_1" },
        .{ .role = "tool", .content = "Fri Apr 4", .tool_call_id = "call_2" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();

    // Both tool messages should have tool_call_id and content
    const tool1 = parsed.value.array.items[1].object;
    try testing.expectEqualStrings("call_1", tool1.get("tool_call_id").?.string);
    try testing.expectEqualStrings("file content", tool1.get("content").?.string);

    const tool2 = parsed.value.array.items[2].object;
    try testing.expectEqualStrings("call_2", tool2.get("tool_call_id").?.string);
}

test "serializeMessagesJson tool response with empty content" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"mkdir test\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "", .tool_call_id = "call_1" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Should have tool_call_id but no tool_responses
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_call_id\"") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_responses\"") == null);
}

// ── Fallback formatter tests ──

test "serializeMessagesJsonOpts: harmony gets \"\" for empty content, everyone else keeps null" {
    // The harmony template guards with `if \"content\" in message` and then
    // does `\"…\" in message.content`. A null passes the first and breaks the
    // second, and the message that carries no content is precisely the
    // tool-calling assistant turn — so this is the whole tool round-trip.
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "weather?" },
        .{ .role = "assistant", .content = "", .tool_calls = &[_]ToolCall{
            .{ .id = "call_1", .name = "get_weather", .arguments = "{\"location\":\"SF\"}" },
        } },
    };

    const harmony = try serializeMessagesJsonOpts(allocator, &messages, .empty_string);
    defer allocator.free(harmony);
    const hp = try std.json.parseFromSlice(std.json.Value, allocator, harmony, .{});
    defer hp.deinit();
    const hmsg = hp.value.array.items[1].object;
    try testing.expect(hmsg.get("content").? == .string);
    try testing.expectEqualStrings("", hmsg.get("content").?.string);
    // The tool call itself still round-trips.
    try testing.expectEqualStrings(
        "get_weather",
        hmsg.get("tool_calls").?.array.items[0].object.get("function").?.object.get("name").?.string,
    );

    // Every other family keeps the null it has always been given.
    const legacy = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(legacy);
    const lp = try std.json.parseFromSlice(std.json.Value, allocator, legacy, .{});
    defer lp.deinit();
    try testing.expect(lp.value.array.items[1].object.get("content").? == .null);
}
