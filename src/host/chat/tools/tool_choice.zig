//! `tool_choice` enforcement: `ForcedTool`, the forced opener, the tool system prompt.

const std = @import("std");
const testing = std.testing;

/// `tool_choice` that obliges a call: any declared tool, or the named one.
pub const ForcedTool = union(enum) { any, name: []const u8 };

/// The prompt line for a forced choice; the caller frees it.
pub fn toolChoiceInstruction(allocator: std.mem.Allocator, forced: ForcedTool) ![]u8 {
    return switch (forced) {
        .any => allocator.dupe(u8, "\nYou MUST call one of the available functions. Do not respond with text."),
        .name => |n| std.fmt.allocPrint(allocator, "\nYou MUST call the function \"{s}\". Do not respond with text.", .{n}),
    };
}

/// Whether a request's `tools` array declares `name`, as `function.name`
/// (chat completions) or a top-level `name` (Anthropic, Responses).
pub fn toolsDeclare(tools: ?std.json.Value, name: []const u8) bool {
    const arr = tools orelse return false;
    if (arr != .array) return false;
    for (arr.array.items) |t| {
        if (t != .object) continue;
        const holder = if (t.object.get("function")) |f| f else t;
        if (holder != .object) continue;
        const n = holder.object.get("name") orelse continue;
        if (n == .string and std.mem.eql(u8, n.string, name)) return true;
    }
    return false;
}

/// The assistant text that commits a tool call in the template's own dialect,
/// or null for a dialect we cannot spell (the instruction line alone asks then).
pub fn forcedToolOpener(allocator: std.mem.Allocator, template: []const u8, forced: ForcedTool) !?[]u8 {
    // The pair, escaped or literal: Llama 3.1 spells a different `<function=` dialect.
    if (std.mem.indexOf(u8, template, "<tool_call>\\n<function=") == null and
        std.mem.indexOf(u8, template, "<tool_call>\n<function=") == null) return null;
    return switch (forced) {
        // Not past `=`: the pre-tokenizer joins it to the name (`=get`), so a lone `=` steers the name.
        .any => try allocator.dupe(u8, "<tool_call>\n<function"),
        .name => |n| try std.fmt.allocPrint(allocator, "<tool_call>\n<function={s}>\n", .{n}),
    };
}

/// Append tool definitions as a system prompt section.
pub fn appendToolSystemPrompt(allocator: std.mem.Allocator, result_buf: *std.ArrayList(u8), tools_json: []const u8, tool_choice_instruction: ?[]const u8) !void {
    try result_buf.appendSlice(allocator,
        \\You are a helpful assistant with access to the following functions. To call a function, respond with a JSON object in the following format:
        \\<tool_call>
        \\{"name": "function_name", "arguments": {"arg1": "value1"}}
        \\</tool_call>
        \\
        \\Available functions:
        \\
    );
    try result_buf.appendSlice(allocator, tools_json);
    if (tool_choice_instruction) |instr| {
        try result_buf.appendSlice(allocator, instr);
    }
}

// ── Tests ──

// ── tests ──

test "toolChoiceInstruction: one line per forced choice" {
    const allocator = testing.allocator;
    const any = try toolChoiceInstruction(allocator, .any);
    defer allocator.free(any);
    try testing.expectEqualStrings("\nYou MUST call one of the available functions. Do not respond with text.", any);
    const named = try toolChoiceInstruction(allocator, .{ .name = "calculator" });
    defer allocator.free(named);
    try testing.expectEqualStrings("\nYou MUST call the function \"calculator\". Do not respond with text.", named);
}

test "toolsDeclare: a name in the OpenAI, Anthropic or Responses tool shape" {
    const body =
        \\{"chat":[{"type":"function","function":{"name":"calculator"}}],"flat":[{"name":"probe_ping"}]}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const chat = parsed.value.object.get("chat");
    const flat = parsed.value.object.get("flat");
    try testing.expect(toolsDeclare(chat, "calculator"));
    try testing.expect(!toolsDeclare(chat, "bogus"));
    try testing.expect(toolsDeclare(flat, "probe_ping"));
    try testing.expect(!toolsDeclare(null, "calculator"));
}

test "forcedToolOpener: the XML dialect commits a call in the template's own spelling" {
    const allocator = testing.allocator;
    const qwen = @embedFile("../../../fixtures/qwen38_chat_template.jinja");
    const any = (try forcedToolOpener(allocator, qwen, .any)).?;
    defer allocator.free(any);
    try testing.expectEqualStrings("<tool_call>\n<function", any);
    const named = (try forcedToolOpener(allocator, qwen, .{ .name = "calculator" })).?;
    defer allocator.free(named);
    try testing.expectEqualStrings("<tool_call>\n<function=calculator>\n", named);
    try testing.expect(try forcedToolOpener(allocator, "<function=example>{}</function>", .any) == null);
    // A dialect we cannot spell keeps the instruction line only.
    try testing.expect(try forcedToolOpener(allocator, @embedFile("../../../fixtures/glm5_next_chat_template.jinja"), .any) == null);
}
