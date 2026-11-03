//! A chat prompt for a GGUF engine (`Caps.owns_tokenizer`): rendered by the engine's own template where it owns one
//! (ds4), else by the host's Jinja over the GGUF's embedded template (llama.cpp), tokenized by the engine.

const std = @import("std");
const log = @import("../../sdk/util/log.zig");
const Served = @import("served.zig").Served;
const types = @import("../chat/types.zig");
const render = @import("../chat/render.zig");
const Message = types.Message;
const ChatConfig = types.ChatConfig;

pub fn encodeChat(
    gpa: std.mem.Allocator,
    s: *const Served,
    chat_config: *const ChatConfig,
    messages: []const Message,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
    effort: ?[]const u8,
    continue_final: bool,
) ![]u32 {
    const ids = if (s.caps().owns_template) blk: {
        const chat = try templateChatJson(gpa, messages, tools_json, tool_choice_instruction, enable_thinking);
        defer gpa.free(chat);
        break :blk try s.render(gpa, chat);
    } else blk: {
        const rendered = try render.renderChatTemplate(gpa, messages, chat_config, tools_json, tool_choice_instruction, enable_thinking, effort, continue_final);
        defer gpa.free(rendered);
        // The template owns any BOS.
        break :blk try s.tokenize(gpa, rendered, false);
    };
    log.debug("  prompt ({s}): {d} messages -> {d} tokens (tools={s})\n", .{ @tagName(s.kind), messages.len, ids.len, if (tools_json != null) "yes" else "no" });
    return ids;
}

/// `{"messages":[{"role","content"}],"enable_thinking":bool}` for an engine's own template, which models neither
/// `tools` nor `role:"tool"`: the tool catalog goes into the system message, tool results become `<tool_response>`
/// user turns and assistant tool calls inline `<tool_call>` text, so agents keep their whole history.
pub fn templateChatJson(gpa: std.mem.Allocator, messages: []const Message, tools_json: ?[]const u8, tool_choice_instruction: ?[]const u8, enable_thinking: bool) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var folded: std.ArrayList(Message) = .empty;
    try folded.appendSlice(a, messages);
    _ = try types.foldSystemMessages(a, &folded, false);
    const has_tool_content = render.messagesHaveToolContent(folded.items);
    const effective = if (tools_json != null or has_tool_content)
        try render.synthesizeToolFallbackMessages(a, folded.items, tools_json, tool_choice_instruction, tools_json != null, has_tool_content, has_tool_content)
    else
        folded.items;
    const Turn = struct { role: []const u8, content: []const u8 };
    const turns = try a.alloc(Turn, effective.len);
    for (effective, turns) |m, *t| t.* = .{ .role = m.role, .content = m.content };
    return std.json.Stringify.valueAlloc(gpa, .{ .messages = turns, .enable_thinking = enable_thinking }, .{});
}

// ── tests ──

const testing = std.testing;
const ToolCall = types.ToolCall;

const Chat = struct {
    messages: []const struct { role: []const u8, content: []const u8 },
    enable_thinking: bool,
};

fn parsed(messages: []const Message, tools_json: ?[]const u8) !std.json.Parsed(Chat) {
    const json = try templateChatJson(testing.allocator, messages, tools_json, null, false);
    defer testing.allocator.free(json);
    return std.json.parseFromSlice(Chat, testing.allocator, json, .{ .allocate = .alloc_always });
}

test "templateChatJson: tools_json injects the catalog into a system message" {
    var p = try parsed(&.{.{ .role = "user", .content = "list /tmp" }}, "[{\"type\":\"function\",\"function\":{\"name\":\"shell\",\"description\":\"Run a command\"}}]");
    defer p.deinit();
    const m = p.value.messages;
    try testing.expectEqual(@as(usize, 2), m.len);
    try testing.expectEqualStrings("system", m[0].role);
    try testing.expect(std.mem.indexOf(u8, m[0].content, "shell") != null);
    try testing.expectEqualStrings("user", m[1].role);
    try testing.expectEqualStrings("list /tmp", m[1].content);
}

test "templateChatJson: tools_json merges into an existing system message" {
    var p = try parsed(&.{ .{ .role = "system", .content = "You are helpful." }, .{ .role = "user", .content = "hi" } }, "[{\"type\":\"function\",\"function\":{\"name\":\"webSearch\"}}]");
    defer p.deinit();
    const sys = p.value.messages[0];
    try testing.expectEqualStrings("system", sys.role);
    try testing.expect(std.mem.indexOf(u8, sys.content, "You are helpful.") != null);
    try testing.expect(std.mem.indexOf(u8, sys.content, "webSearch") != null);
}

test "templateChatJson: role:tool becomes a user turn with <tool_response>" {
    var p = try parsed(&.{ .{ .role = "user", .content = "run ls" }, .{ .role = "tool", .content = "file1.txt\nfile2.txt" } }, null);
    defer p.deinit();
    const m = p.value.messages;
    try testing.expectEqual(@as(usize, 2), m.len);
    try testing.expectEqualStrings("user", m[1].role);
    try testing.expect(std.mem.indexOf(u8, m[1].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, m[1].content, "file1.txt") != null);
}

test "templateChatJson: assistant tool_calls are inlined as <tool_call> text" {
    const calls = [_]ToolCall{.{ .id = "c1", .name = "shell", .arguments = "{\"command\":\"ls /tmp\"}" }};
    var p = try parsed(&.{
        .{ .role = "user", .content = "list tmp" },
        .{ .role = "assistant", .content = "", .tool_calls = &calls },
        .{ .role = "tool", .content = "ok", .tool_call_id = "c1" },
    }, null);
    defer p.deinit();
    const m = p.value.messages;
    try testing.expectEqual(@as(usize, 3), m.len);
    try testing.expectEqualStrings("assistant", m[1].role);
    try testing.expect(std.mem.indexOf(u8, m[1].content, "<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, m[1].content, "\"name\": \"shell\"") != null);
    try testing.expect(std.mem.indexOf(u8, m[1].content, "ls /tmp") != null);
}

test "templateChatJson: no tools and no tool content pass through; the thinking switch rides along" {
    const json = try templateChatJson(testing.allocator, &.{ .{ .role = "system", .content = "be terse" }, .{ .role = "user", .content = "hi" }, .{ .role = "assistant", .content = "hello" } }, null, null, true);
    defer testing.allocator.free(json);
    try testing.expectEqualStrings(
        \\{"messages":[{"role":"system","content":"be terse"},{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}],"enable_thinking":true}
    , json);
}
