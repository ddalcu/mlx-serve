const std = @import("std");
const jinja_c = @cImport({
    @cInclude("jinja_wrapper.h");
});
const tokenizer_mod = @import("tokenizer.zig");

const Tokenizer = tokenizer_mod.Tokenizer;

pub const ToolCall = struct {
    name: []const u8,
    arguments: []const u8, // JSON string
};

pub const Message = struct {
    role: []const u8,
    content: ?[]const u8 = null,
    tool_calls: ?[]const ToolCall = null,
    tool_call_id: ?[]const u8 = null,
};

pub const ChatConfig = struct {
    chat_template: []const u8,
    bos_token: ?[]const u8,
    eos_token: ?[]const u8,
    add_bos_token: bool,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ChatConfig) void {
        self.allocator.free(self.chat_template);
        if (self.bos_token) |t| self.allocator.free(t);
        if (self.eos_token) |t| self.allocator.free(t);
    }
};

pub fn loadChatConfig(allocator: std.mem.Allocator, model_dir: []const u8) !ChatConfig {
    const path = try std.fmt.allocPrint(allocator, "{s}/tokenizer_config.json", .{model_dir});
    defer allocator.free(path);

    const file = try std.fs.openFileAbsolute(path, .{});
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 10 * 1024 * 1024);
    defer allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();

    const root = parsed.value.object;

    const chat_template: []const u8 = if (root.get("chat_template")) |v|
        try allocator.dupe(u8, v.string)
    else blk: {
        // Fall back to chat_template.jinja file (e.g. Qwen3.5 models)
        const jinja_path = try std.fmt.allocPrint(allocator, "{s}/chat_template.jinja", .{model_dir});
        defer allocator.free(jinja_path);
        if (std.fs.openFileAbsolute(jinja_path, .{})) |f| {
            defer f.close();
            break :blk try f.readToEndAlloc(allocator, 1 * 1024 * 1024);
        } else |_| {
            break :blk try allocator.dupe(u8, "");
        }
    };

    const bos_token: ?[]const u8 = if (root.get("bos_token")) |v|
        (if (v == .string) try allocator.dupe(u8, v.string) else null)
    else
        null;

    const eos_token: ?[]const u8 = if (root.get("eos_token")) |v|
        (if (v == .string) try allocator.dupe(u8, v.string) else null)
    else
        null;

    const add_bos_token = if (root.get("add_bos_token")) |v|
        (if (v == .bool) v.bool else false)
    else
        false;

    return .{
        .chat_template = chat_template,
        .bos_token = bos_token,
        .eos_token = eos_token,
        .add_bos_token = add_bos_token,
        .allocator = allocator,
    };
}

pub fn formatChat(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    messages: []const Message,
    chat_config: *const ChatConfig,
    tools: ?std.json.Value,
) ![]u32 {
    const rendered = try renderChatTemplate(allocator, messages, chat_config, tools);
    defer allocator.free(rendered);

    var ids = std.ArrayList(u32).empty;
    errdefer ids.deinit(allocator);

    if (chat_config.add_bos_token) {
        if (tok.bos_id) |bos| {
            try ids.append(allocator, bos);
        }
    }

    try encodeWithSpecialTokens(allocator, tok, rendered, &ids);
    std.debug.print("  prompt: {d} chars -> {d} tokens\n", .{ rendered.len, ids.items.len });

    return ids.toOwnedSlice(allocator);
}

fn renderChatTemplate(
    allocator: std.mem.Allocator,
    messages: []const Message,
    chat_config: *const ChatConfig,
    tools: ?std.json.Value,
) ![]const u8 {
    if (chat_config.chat_template.len == 0) {
        return fallbackFormatChat(allocator, messages, chat_config, tools);
    }

    // Serialize messages to JSON
    const messages_json = try serializeMessagesJson(allocator, messages);
    defer allocator.free(messages_json);

    // Serialize tools to JSON
    var tools_json_buf: ?[]const u8 = null;
    defer if (tools_json_buf) |t| allocator.free(t);
    if (tools) |t| {
        tools_json_buf = try std.json.Stringify.valueAlloc(allocator, t, .{});
    }

    // Build extra context (bos_token, eos_token, enable_thinking)
    const extra_json = try serializeExtraContext(allocator, chat_config);
    defer allocator.free(extra_json);

    // Null-terminate strings for C
    const tmpl_z = try allocator.dupeZ(u8, chat_config.chat_template);
    defer allocator.free(tmpl_z);
    const msgs_z = try allocator.dupeZ(u8, messages_json);
    defer allocator.free(msgs_z);
    const extra_z = try allocator.dupeZ(u8, extra_json);
    defer allocator.free(extra_z);

    var tools_z: ?[:0]const u8 = null;
    defer if (tools_z) |tz| allocator.free(tz);
    if (tools_json_buf) |tj| {
        tools_z = try allocator.dupeZ(u8, tj);
    }

    const result_ptr = jinja_c.jinja_render_chat(
        tmpl_z.ptr,
        msgs_z.ptr,
        if (tools_z) |tz| tz.ptr else null,
        extra_z.ptr,
        1,
    );

    if (result_ptr) |ptr| {
        defer jinja_c.jinja_str_free(ptr);
        return try allocator.dupe(u8, std.mem.span(ptr));
    }

    if (jinja_c.jinja_last_error()) |err| {
        std.debug.print("  jinja error: {s}, using fallback\n", .{std.mem.span(err)});
    }
    return fallbackFormatChat(allocator, messages, chat_config, tools);
}

fn serializeMessagesJson(allocator: std.mem.Allocator, messages: []const Message) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);

    try buf.append(allocator, '[');
    for (messages, 0..) |msg, i| {
        if (i > 0) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, "{\"role\":");
        try appendJsonString(allocator, &buf, msg.role);

        try buf.appendSlice(allocator, ",\"content\":");
        if (msg.content) |c| {
            try appendJsonString(allocator, &buf, c);
        } else {
            try buf.appendSlice(allocator, "null");
        }

        if (msg.tool_calls) |tcs| {
            try buf.appendSlice(allocator, ",\"tool_calls\":[");
            for (tcs, 0..) |tc, ti| {
                if (ti > 0) try buf.append(allocator, ',');
                try buf.appendSlice(allocator, "{\"type\":\"function\",\"function\":{\"name\":");
                try appendJsonString(allocator, &buf, tc.name);
                try buf.appendSlice(allocator, ",\"arguments\":");
                try buf.appendSlice(allocator, tc.arguments);
                try buf.appendSlice(allocator, "}}");
            }
            try buf.append(allocator, ']');
        }

        if (msg.tool_call_id) |tid| {
            try buf.appendSlice(allocator, ",\"tool_call_id\":");
            try appendJsonString(allocator, &buf, tid);
        }

        try buf.append(allocator, '}');
    }
    try buf.append(allocator, ']');

    return buf.toOwnedSlice(allocator);
}

fn serializeExtraContext(allocator: std.mem.Allocator, chat_config: *const ChatConfig) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);

    try buf.append(allocator, '{');
    var need_comma = false;

    if (chat_config.bos_token) |bos| {
        try buf.appendSlice(allocator, "\"bos_token\":");
        try appendJsonString(allocator, &buf, bos);
        need_comma = true;
    }
    if (chat_config.eos_token) |eos| {
        if (need_comma) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, "\"eos_token\":");
        try appendJsonString(allocator, &buf, eos);
        need_comma = true;
    }
    if (need_comma) try buf.append(allocator, ',');
    try buf.appendSlice(allocator, "\"enable_thinking\":true");

    try buf.append(allocator, '}');
    return buf.toOwnedSlice(allocator);
}

fn encodeWithSpecialTokens(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    text: []const u8,
    ids: *std.ArrayList(u32),
) !void {
    var specials = std.ArrayList(SpecialEntry).empty;
    defer specials.deinit(allocator);

    var sit = tok.special_tokens.iterator();
    while (sit.next()) |entry| {
        try specials.append(allocator, .{ .text = entry.key_ptr.*, .id = entry.value_ptr.* });
    }

    std.mem.sort(SpecialEntry, specials.items, {}, struct {
        fn lessThan(_: void, a: SpecialEntry, b: SpecialEntry) bool {
            return a.text.len > b.text.len;
        }
    }.lessThan);

    var pos: usize = 0;
    while (pos < text.len) {
        var matched = false;
        for (specials.items) |special| {
            if (pos + special.text.len <= text.len and
                std.mem.eql(u8, text[pos .. pos + special.text.len], special.text))
            {
                try ids.append(allocator, special.id);
                pos += special.text.len;
                matched = true;
                break;
            }
        }
        if (matched) continue;

        var next_special_pos: usize = text.len;
        for (specials.items) |special| {
            if (std.mem.indexOf(u8, text[pos..], special.text)) |offset| {
                const abs_pos = pos + offset;
                if (abs_pos < next_special_pos) {
                    next_special_pos = abs_pos;
                }
            }
        }

        if (next_special_pos > pos) {
            const segment = text[pos..next_special_pos];
            const segment_ids = try tok.encode(allocator, segment);
            defer allocator.free(segment_ids);
            try ids.appendSlice(allocator, segment_ids);
        }
        pos = next_special_pos;
    }
}

const SpecialEntry = struct {
    text: []const u8,
    id: u32,
};

fn fallbackFormatChat(
    allocator: std.mem.Allocator,
    messages: []const Message,
    chat_config: *const ChatConfig,
    tools: ?std.json.Value,
) ![]const u8 {
    // Serialize tools to string for fallback template
    var tools_buf: ?[]const u8 = null;
    defer if (tools_buf) |t| allocator.free(t);
    if (tools) |t| {
        if (t == .array) {
            var tj = std.ArrayList(u8).empty;
            defer tj.deinit(allocator);
            for (t.array.items, 0..) |tool, i| {
                if (i > 0) try tj.append(allocator, '\n');
                const s = std.json.Stringify.valueAlloc(allocator, tool, .{}) catch continue;
                defer allocator.free(s);
                try tj.appendSlice(allocator, s);
            }
            tools_buf = try tj.toOwnedSlice(allocator);
        }
    }

    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);

    const is_chatml = chat_config.eos_token != null and
        std.mem.indexOf(u8, chat_config.eos_token.?, "<|im_end|>") != null;

    if (is_chatml) {
        const first_is_system = messages.len > 0 and std.mem.eql(u8, messages[0].role, "system");

        if (tools_buf) |tools_str| {
            try result.appendSlice(allocator, "<|im_start|>system\n");
            try result.appendSlice(allocator,
                \\# Tools
                \\
                \\You may call one or more functions to assist with the user query.
                \\
                \\You are provided with function signatures within <tools></tools> XML tags:
                \\
            );
            try result.appendSlice(allocator, "<tools>\n");
            try result.appendSlice(allocator, tools_str);
            try result.appendSlice(allocator,
                \\
                \\</tools>
                \\
                \\For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
                \\<tool_call>
                \\{"name": "function_name", "arguments": {"arg1": "value1"}}
                \\</tool_call>
            );
            if (first_is_system) {
                if (messages[0].content) |c| {
                    if (c.len > 0) {
                        try result.appendSlice(allocator, "\n\n");
                        try result.appendSlice(allocator, c);
                    }
                }
            }
            try result.appendSlice(allocator, "<|im_end|>\n");
        } else if (first_is_system) {
            try result.appendSlice(allocator, "<|im_start|>system\n");
            if (messages[0].content) |c| try result.appendSlice(allocator, c);
            try result.appendSlice(allocator, "<|im_end|>\n");
        }

        const start_idx: usize = if (first_is_system) 1 else 0;
        for (messages[start_idx..]) |msg| {
            if (std.mem.eql(u8, msg.role, "tool")) {
                try result.appendSlice(allocator, "<|im_start|>user\n<tool_response>\n");
                if (msg.content) |c| try result.appendSlice(allocator, c);
                try result.appendSlice(allocator, "\n</tool_response><|im_end|>\n");
            } else if (std.mem.eql(u8, msg.role, "assistant")) {
                try result.appendSlice(allocator, "<|im_start|>assistant\n");
                if (msg.content) |c| try result.appendSlice(allocator, c);
                if (msg.tool_calls) |tcs| {
                    for (tcs, 0..) |tc, ti| {
                        if (ti == 0 and msg.content != null and msg.content.?.len > 0) {
                            try result.appendSlice(allocator, "\n\n");
                        }
                        try result.appendSlice(allocator, "<tool_call>\n{\"name\": \"");
                        try result.appendSlice(allocator, tc.name);
                        try result.appendSlice(allocator, "\", \"arguments\": ");
                        try result.appendSlice(allocator, tc.arguments);
                        try result.appendSlice(allocator, "}\n</tool_call>");
                    }
                }
                try result.appendSlice(allocator, "<|im_end|>\n");
            } else {
                try result.appendSlice(allocator, "<|im_start|>");
                try result.appendSlice(allocator, msg.role);
                try result.appendSlice(allocator, "\n");
                if (msg.content) |c| try result.appendSlice(allocator, c);
                try result.appendSlice(allocator, "<|im_end|>\n");
            }
        }
        try result.appendSlice(allocator, "<|im_start|>assistant\n");
        // Enable thinking mode for models that support it (e.g. Qwen3, Qwen3.5)
        if (chat_config.chat_template.len > 0 and
            std.mem.indexOf(u8, chat_config.chat_template, "enable_thinking") != null)
        {
            try result.appendSlice(allocator, "<think>\n");
        }
    } else {
        if (chat_config.bos_token) |bos| {
            try result.appendSlice(allocator, bos);
        }
        for (messages) |msg| {
            try result.appendSlice(allocator, "<start_of_turn>");
            try result.appendSlice(allocator, msg.role);
            try result.appendSlice(allocator, "\n");
            if (msg.content) |c| try result.appendSlice(allocator, c);
            try result.appendSlice(allocator, "<end_of_turn>\n");
        }
        try result.appendSlice(allocator, "<start_of_turn>model\n");
    }

    return result.toOwnedSlice(allocator);
}

/// Strip `<think>...</think>` block from model output. Returns the text after the block,
/// or the original text if no think block is present.
pub fn stripThinkBlock(text: []const u8) []const u8 {
    if (std.mem.indexOf(u8, text, "</think>")) |think_end| {
        return std.mem.trimLeft(u8, text[think_end + 8 ..], "\n ");
    }
    if (std.mem.startsWith(u8, text, "<think>")) return text[0..0];
    return text;
}

pub const ThinkSplit = struct {
    reasoning_content: ?[]const u8,
    content: []const u8,
};

/// Split model output into reasoning_content and content.
/// Handles both prompt-injected thinking (no <think> prefix) and model-generated <think>.
pub fn splitThinkBlock(text: []const u8, thinking: bool) ThinkSplit {
    if (std.mem.indexOf(u8, text, "</think>")) |end| {
        const reasoning_start: usize = if (std.mem.startsWith(u8, text, "<think>")) 7 else 0;
        const reasoning = std.mem.trim(u8, text[reasoning_start..end], "\n ");
        const content = std.mem.trimLeft(u8, text[end + 8 ..], "\n ");
        return .{
            .reasoning_content = if (reasoning.len > 0) reasoning else null,
            .content = content,
        };
    }
    if (thinking) {
        // Still inside think block (never closed), all output is reasoning
        return .{ .reasoning_content = text, .content = "" };
    }
    return .{ .reasoning_content = null, .content = text };
}

/// Parse tool calls from model output text. Returns parsed tool calls and content before them.
pub const ParsedResponse = struct {
    content: ?[]const u8,
    tool_calls: ?[]ParsedToolCall,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ParsedResponse) void {
        if (self.content) |c| self.allocator.free(c);
        if (self.tool_calls) |tcs| {
            for (tcs) |tc| {
                self.allocator.free(tc.name);
                self.allocator.free(tc.arguments);
            }
            self.allocator.free(tcs);
        }
    }
};

pub const ParsedToolCall = struct {
    name: []const u8,
    arguments: []const u8,
};

pub fn parseToolCalls(allocator: std.mem.Allocator, text: []const u8) !ParsedResponse {
    // Strip <think>...</think> block if present
    var effective_text = text;
    if (std.mem.indexOf(u8, text, "</think>")) |think_end| {
        effective_text = std.mem.trimLeft(u8, text[think_end + 8 ..], "\n ");
    }

    const tool_start = std.mem.indexOf(u8, effective_text, "<tool_call>");
    if (tool_start == null) {
        return .{
            .content = try allocator.dupe(u8, std.mem.trim(u8, effective_text, "\n ")),
            .tool_calls = null,
            .allocator = allocator,
        };
    }

    // Content before tool calls
    const before = std.mem.trim(u8, effective_text[0..tool_start.?], "\n ");
    const content: ?[]const u8 = if (before.len > 0) try allocator.dupe(u8, before) else null;

    // Parse all <tool_call> blocks
    var calls = std.ArrayList(ParsedToolCall).empty;
    errdefer {
        for (calls.items) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        calls.deinit(allocator);
    }

    var search_pos: usize = 0;
    while (std.mem.indexOf(u8, effective_text[search_pos..], "<tool_call>")) |rel_start| {
        const abs_start = search_pos + rel_start + "<tool_call>".len;
        const end = std.mem.indexOf(u8, effective_text[abs_start..], "</tool_call>") orelse break;
        const block = effective_text[abs_start .. abs_start + end];
        search_pos = abs_start + end + "</tool_call>".len;

        const trimmed_block = std.mem.trim(u8, block, " \n\t");

        // Try JSON format first: {"name": "...", "arguments": {...}}
        if (tryParseJsonToolCall(allocator, trimmed_block)) |tc| {
            try calls.append(allocator, tc);
            continue;
        }

        // Fall back to Hermes format: <function=name><parameter=name>value</parameter></function>
        const fn_start_tag = "<function=";
        const fn_start = std.mem.indexOf(u8, block, fn_start_tag) orelse continue;
        const name_start = fn_start + fn_start_tag.len;
        const name_end = std.mem.indexOf(u8, block[name_start..], ">") orelse continue;
        const fn_name = std.mem.trim(u8, block[name_start .. name_start + name_end], " \n");

        var args_map = std.ArrayList(u8).empty;
        defer args_map.deinit(allocator);
        try args_map.append(allocator, '{');

        const fn_body_start = name_start + name_end + 1;
        const fn_end = std.mem.indexOf(u8, block[fn_body_start..], "</function>") orelse block.len - fn_body_start;
        const fn_body = block[fn_body_start .. fn_body_start + fn_end];

        var param_search: usize = 0;
        var first_param = true;
        while (std.mem.indexOf(u8, fn_body[param_search..], "<parameter=")) |ps| {
            const p_name_start = param_search + ps + "<parameter=".len;
            const p_name_end = std.mem.indexOf(u8, fn_body[p_name_start..], ">") orelse break;
            const p_name = std.mem.trim(u8, fn_body[p_name_start .. p_name_start + p_name_end], " \n");
            const p_val_start = p_name_start + p_name_end + 1;
            const p_val_end = std.mem.indexOf(u8, fn_body[p_val_start..], "</parameter>") orelse break;
            const p_val = std.mem.trim(u8, fn_body[p_val_start .. p_val_start + p_val_end], "\n");

            if (!first_param) try args_map.append(allocator, ',');
            first_param = false;

            try args_map.append(allocator, '"');
            try args_map.appendSlice(allocator, p_name);
            try args_map.appendSlice(allocator, "\":");

            const trimmed_val = std.mem.trim(u8, p_val, " ");
            if (isJsonLiteral(trimmed_val)) {
                try args_map.appendSlice(allocator, trimmed_val);
            } else {
                try appendJsonString(allocator, &args_map, trimmed_val);
            }

            param_search = p_val_start + p_val_end + "</parameter>".len;
        }

        try args_map.append(allocator, '}');

        try calls.append(allocator, .{
            .name = try allocator.dupe(u8, fn_name),
            .arguments = try allocator.dupe(u8, args_map.items),
        });
    }

    return .{
        .content = content,
        .tool_calls = if (calls.items.len > 0) try calls.toOwnedSlice(allocator) else null,
        .allocator = allocator,
    };
}

fn isJsonLiteral(s: []const u8) bool {
    if (s.len == 0) return false;
    if (std.mem.eql(u8, s, "true") or std.mem.eql(u8, s, "false") or std.mem.eql(u8, s, "null")) return true;
    if (s[0] == '{' or s[0] == '[') return true;
    // Try parsing as number
    _ = std.fmt.parseFloat(f64, s) catch return false;
    return true;
}

fn appendJsonString(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), s: []const u8) !void {
    try buf.append(allocator, '"');
    for (s) |c| {
        switch (c) {
            '"' => try buf.appendSlice(allocator, "\\\""),
            '\\' => try buf.appendSlice(allocator, "\\\\"),
            '\n' => try buf.appendSlice(allocator, "\\n"),
            '\r' => try buf.appendSlice(allocator, "\\r"),
            '\t' => try buf.appendSlice(allocator, "\\t"),
            else => try buf.append(allocator, c),
        }
    }
    try buf.append(allocator, '"');
}

fn tryParseJsonToolCall(allocator: std.mem.Allocator, text: []const u8) ?ParsedToolCall {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, text, .{}) catch return null;
    defer parsed.deinit();

    if (parsed.value != .object) return null;
    const obj = parsed.value.object;

    const name_val = obj.get("name") orelse return null;
    if (name_val != .string) return null;

    const args_val = obj.get("arguments") orelse return null;
    const args_str = switch (args_val) {
        .object => std.json.Stringify.valueAlloc(allocator, args_val, .{}) catch return null,
        .string => |s| allocator.dupe(u8, s) catch return null,
        else => return null,
    };

    return .{
        .name = allocator.dupe(u8, name_val.string) catch {
            allocator.free(args_str);
            return null;
        },
        .arguments = args_str,
    };
}
