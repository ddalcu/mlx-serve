const std = @import("std");
const jinja_c = @cImport({
    @cInclude("jinja_wrapper.h");
});
const tokenizer_mod = @import("tokenizer.zig");
const log = @import("log.zig");

const Tokenizer = tokenizer_mod.Tokenizer;

pub const ToolCall = struct {
    id: []const u8,
    name: []const u8,
    arguments: []const u8, // JSON string
};

pub const Message = struct {
    role: []const u8,
    content: []const u8,
    tool_calls: ?[]const ToolCall = null,
    tool_call_id: ?[]const u8 = null,
};

/// Chat template configuration loaded from tokenizer_config.json.
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

/// Load chat template configuration from tokenizer_config.json.
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

/// Format chat messages into token IDs using the model's Jinja chat template.
pub fn formatChat(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    messages: []const Message,
    chat_config: *const ChatConfig,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
) ![]u32 {
    const rendered = try renderChatTemplate(allocator, messages, chat_config, tools_json, tool_choice_instruction, enable_thinking);
    defer allocator.free(rendered);

    var ids = std.ArrayList(u32).empty;
    errdefer ids.deinit(allocator);

    if (chat_config.add_bos_token) {
        if (tok.bos_id) |bos| {
            try ids.append(allocator, bos);
        }
    }

    try encodeWithSpecialTokens(allocator, tok, rendered, &ids);
    log.debug("  prompt: {d} chars -> {d} tokens\n", .{ rendered.len, ids.items.len });

    return ids.toOwnedSlice(allocator);
}

/// Render the Jinja chat template with the given messages.
fn renderChatTemplate(
    allocator: std.mem.Allocator,
    messages: []const Message,
    chat_config: *const ChatConfig,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
) ![]const u8 {
    if (chat_config.chat_template.len == 0) {
        return fallbackFormatChat(allocator, messages, chat_config, tools_json, tool_choice_instruction, enable_thinking);
    }

    // Serialize messages to JSON
    const messages_json = try serializeMessagesJson(allocator, messages);
    defer allocator.free(messages_json);

    // Build extra context (bos_token, eos_token, enable_thinking)
    const extra_json = try serializeExtraContext(allocator, chat_config, enable_thinking);
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
    if (tools_json) |tj| {
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

    if (jinja_c.jinja_last_error()) |e| {
        log.debug("  jinja error: {s}, using fallback\n", .{std.mem.span(e)});
    }
    return fallbackFormatChat(allocator, messages, chat_config, tools_json, tool_choice_instruction, enable_thinking);
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
        if (msg.content.len > 0) {
            try appendJsonString(allocator, &buf, msg.content);
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

fn serializeExtraContext(allocator: std.mem.Allocator, chat_config: *const ChatConfig, enable_thinking: bool) ![]const u8 {
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
    if (enable_thinking) {
        try buf.appendSlice(allocator, "\"enable_thinking\":true");
    } else {
        try buf.appendSlice(allocator, "\"enable_thinking\":false");
    }

    try buf.append(allocator, '}');
    return buf.toOwnedSlice(allocator);
}

/// Encode text that may contain special tokens (like <|im_start|>, <bos>, etc.).
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

/// Fallback chat formatting for when Jinja rendering fails.
fn fallbackFormatChat(
    allocator: std.mem.Allocator,
    messages: []const Message,
    chat_config: *const ChatConfig,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
) ![]const u8 {
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);

    const is_chatml = chat_config.eos_token != null and
        std.mem.indexOf(u8, chat_config.eos_token.?, "<|im_end|>") != null;

    const has_system = messages.len > 0 and std.mem.eql(u8, messages[0].role, "system");

    if (is_chatml) {
        // ChatML format (Qwen3, etc.)
        if (tools_json != null and !has_system) {
            try result.appendSlice(allocator, "<|im_start|>system\n");
            try appendToolSystemPrompt(allocator, &result, tools_json.?, tool_choice_instruction);
            try result.appendSlice(allocator, "<|im_end|>\n");
        }

        for (messages) |msg| {
            if (std.mem.eql(u8, msg.role, "system") and tools_json != null) {
                try result.appendSlice(allocator, "<|im_start|>system\n");
                try result.appendSlice(allocator, msg.content);
                try result.appendSlice(allocator, "\n\n");
                try appendToolSystemPrompt(allocator, &result, tools_json.?, tool_choice_instruction);
                try result.appendSlice(allocator, "<|im_end|>\n");
            } else if (std.mem.eql(u8, msg.role, "assistant") and msg.tool_calls != null) {
                try result.appendSlice(allocator, "<|im_start|>assistant\n");
                if (msg.content.len > 0) {
                    try result.appendSlice(allocator, msg.content);
                    try result.appendSlice(allocator, "\n");
                }
                for (msg.tool_calls.?) |tc| {
                    try result.appendSlice(allocator, "<tool_call>\n");
                    try result.appendSlice(allocator, "{\"name\": \"");
                    try result.appendSlice(allocator, tc.name);
                    try result.appendSlice(allocator, "\", \"arguments\": ");
                    try result.appendSlice(allocator, tc.arguments);
                    try result.appendSlice(allocator, "}\n</tool_call>");
                }
                try result.appendSlice(allocator, "<|im_end|>\n");
            } else if (std.mem.eql(u8, msg.role, "tool")) {
                try result.appendSlice(allocator, "<|im_start|>user\n");
                try result.appendSlice(allocator, "<tool_response>\n");
                try result.appendSlice(allocator, msg.content);
                try result.appendSlice(allocator, "\n</tool_response>");
                try result.appendSlice(allocator, "<|im_end|>\n");
            } else {
                try result.appendSlice(allocator, "<|im_start|>");
                try result.appendSlice(allocator, msg.role);
                try result.appendSlice(allocator, "\n");
                try result.appendSlice(allocator, msg.content);
                try result.appendSlice(allocator, "<|im_end|>\n");
            }
        }
        try result.appendSlice(allocator, "<|im_start|>assistant\n");
        if (std.mem.indexOf(u8, chat_config.chat_template, "enable_thinking") != null) {
            if (enable_thinking) {
                try result.appendSlice(allocator, "<think>\n");
            } else {
                try result.appendSlice(allocator, "<think>\n\n</think>\n\n");
            }
        }
    } else {
        // Gemma/Llama-style format
        if (chat_config.bos_token) |bos| {
            try result.appendSlice(allocator, bos);
        }

        const is_llama = std.mem.indexOf(u8, chat_config.chat_template, "start_header_id") != null;

        if (is_llama) {
            // Llama 3 format with tool support
            try result.appendSlice(allocator, "<|start_header_id|>system<|end_header_id|>\n\n");
            if (tools_json != null) {
                try result.appendSlice(allocator, "Environment: ipython\n");
            }
            if (has_system) {
                try result.appendSlice(allocator, messages[0].content);
            } else {
                try result.appendSlice(allocator, "You are a helpful assistant.");
            }
            if (tools_json != null) {
                try result.appendSlice(allocator, "\n\n");
                try appendToolSystemPrompt(allocator, &result, tools_json.?, tool_choice_instruction);
            }
            try result.appendSlice(allocator, "<|eot_id|>");

            const start_idx: usize = if (has_system) 1 else 0;
            for (messages[start_idx..]) |msg| {
                if (std.mem.eql(u8, msg.role, "assistant") and msg.tool_calls != null) {
                    try result.appendSlice(allocator, "<|start_header_id|>assistant<|end_header_id|>\n\n");
                    for (msg.tool_calls.?) |tc| {
                        try result.appendSlice(allocator, "{\"name\": \"");
                        try result.appendSlice(allocator, tc.name);
                        try result.appendSlice(allocator, "\", \"parameters\": ");
                        try result.appendSlice(allocator, tc.arguments);
                        try result.appendSlice(allocator, "}");
                    }
                    try result.appendSlice(allocator, "<|eot_id|>");
                } else if (std.mem.eql(u8, msg.role, "tool")) {
                    try result.appendSlice(allocator, "<|start_header_id|>ipython<|end_header_id|>\n\n");
                    try result.appendSlice(allocator, msg.content);
                    try result.appendSlice(allocator, "<|eot_id|>");
                } else {
                    try result.appendSlice(allocator, "<|start_header_id|>");
                    try result.appendSlice(allocator, msg.role);
                    try result.appendSlice(allocator, "<|end_header_id|>\n\n");
                    try result.appendSlice(allocator, msg.content);
                    try result.appendSlice(allocator, "<|eot_id|>");
                }
            }
            try result.appendSlice(allocator, "<|start_header_id|>assistant<|end_header_id|>\n\n");
        } else {
            // Gemma-style format
            if (tools_json != null and !has_system) {
                try result.appendSlice(allocator, "<start_of_turn>user\n");
                try appendToolSystemPrompt(allocator, &result, tools_json.?, tool_choice_instruction);
                try result.appendSlice(allocator, "<end_of_turn>\n");
            }
            for (messages) |msg| {
                if (std.mem.eql(u8, msg.role, "tool")) {
                    try result.appendSlice(allocator, "<start_of_turn>user\n");
                    try result.appendSlice(allocator, "Tool result: ");
                    try result.appendSlice(allocator, msg.content);
                    try result.appendSlice(allocator, "<end_of_turn>\n");
                } else {
                    try result.appendSlice(allocator, "<start_of_turn>");
                    try result.appendSlice(allocator, msg.role);
                    try result.appendSlice(allocator, "\n");
                    try result.appendSlice(allocator, msg.content);
                    try result.appendSlice(allocator, "<end_of_turn>\n");
                }
            }
            try result.appendSlice(allocator, "<start_of_turn>model\n");
        }
    }

    return result.toOwnedSlice(allocator);
}

/// Strip `<think>...</think>` block from model output.
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
        const start: usize = if (std.mem.startsWith(u8, text, "<think>")) 7 else 0;
        const reasoning = std.mem.trimLeft(u8, text[start..], "\n ");
        return .{ .reasoning_content = if (reasoning.len > 0) reasoning else null, .content = "" };
    }
    return .{ .reasoning_content = null, .content = text };
}

/// Parse tool calls from model output text.
pub fn parseToolCalls(allocator: std.mem.Allocator, text: []const u8) !?[]ParsedToolCall {
    // Strip <think>...</think> block if present
    var effective_text = text;
    if (std.mem.indexOf(u8, text, "</think>")) |think_end| {
        effective_text = std.mem.trimLeft(u8, text[think_end + 8 ..], "\n ");
    }

    var calls = std.ArrayList(ParsedToolCall).empty;
    errdefer {
        for (calls.items) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        calls.deinit(allocator);
    }

    // Look for <tool_call>...</tool_call> patterns
    var search_pos: usize = 0;
    while (search_pos < effective_text.len) {
        const tag_start = std.mem.indexOf(u8, effective_text[search_pos..], "<tool_call>") orelse break;
        const content_start = search_pos + tag_start + "<tool_call>".len;
        const tag_end = std.mem.indexOf(u8, effective_text[content_start..], "</tool_call>") orelse break;
        const content = std.mem.trim(u8, effective_text[content_start .. content_start + tag_end], " \t\n\r");
        search_pos = content_start + tag_end + "</tool_call>".len;

        // Try JSON format first
        if (tryParseJsonToolCall(allocator, content)) |tc| {
            try calls.append(allocator, tc);
            continue;
        }

        // Fall back to Hermes format: <function=name><parameter=name>value</parameter></function>
        if (parseHermesToolCall(allocator, content)) |tc| {
            try calls.append(allocator, tc);
        }
    }

    // If no <tool_call> tags, try to find raw JSON tool call
    if (calls.items.len == 0) {
        var trimmed = std.mem.trim(u8, effective_text, " \t\n\r");
        if (std.mem.startsWith(u8, trimmed, "</tool_call>")) {
            trimmed = std.mem.trim(u8, trimmed["</tool_call>".len..], " \t\n\r");
        }
        if (std.mem.indexOf(u8, trimmed, "{")) |brace_pos| {
            const json_start = trimmed[brace_pos..];
            if (tryParseJsonToolCall(allocator, json_start)) |tc| {
                try calls.append(allocator, tc);
            }
        }
    }

    if (calls.items.len == 0) return null;
    return try calls.toOwnedSlice(allocator);
}

pub const ParsedToolCall = struct {
    name: []const u8,
    arguments: []const u8, // JSON string
};

fn tryParseJsonToolCall(allocator: std.mem.Allocator, text: []const u8) ?ParsedToolCall {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, text, .{}) catch return null;
    defer parsed.deinit();

    if (parsed.value != .object) return null;
    const obj = parsed.value.object;

    const name_val = obj.get("name") orelse return null;
    if (name_val != .string) return null;

    const args_val = obj.get("arguments") orelse (obj.get("parameters") orelse return null);
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

fn parseHermesToolCall(allocator: std.mem.Allocator, block: []const u8) ?ParsedToolCall {
    const fn_start_tag = "<function=";
    const fn_start = std.mem.indexOf(u8, block, fn_start_tag) orelse return null;
    const name_start = fn_start + fn_start_tag.len;
    const name_end = std.mem.indexOf(u8, block[name_start..], ">") orelse return null;
    const fn_name = std.mem.trim(u8, block[name_start .. name_start + name_end], " \n");

    var args_map = std.ArrayList(u8).empty;
    defer args_map.deinit(allocator);
    args_map.append(allocator, '{') catch return null;

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

        if (!first_param) args_map.append(allocator, ',') catch return null;
        first_param = false;

        args_map.append(allocator, '"') catch return null;
        args_map.appendSlice(allocator, p_name) catch return null;
        args_map.appendSlice(allocator, "\":") catch return null;

        const trimmed_val = std.mem.trim(u8, p_val, " ");
        if (isJsonLiteral(trimmed_val)) {
            args_map.appendSlice(allocator, trimmed_val) catch return null;
        } else {
            appendJsonString(allocator, &args_map, trimmed_val) catch return null;
        }

        param_search = p_val_start + p_val_end + "</parameter>".len;
    }

    args_map.append(allocator, '}') catch return null;

    return .{
        .name = allocator.dupe(u8, fn_name) catch return null,
        .arguments = allocator.dupe(u8, args_map.items) catch return null,
    };
}

fn isJsonLiteral(s: []const u8) bool {
    if (s.len == 0) return false;
    if (std.mem.eql(u8, s, "true") or std.mem.eql(u8, s, "false") or std.mem.eql(u8, s, "null")) return true;
    if (s[0] == '{' or s[0] == '[') return true;
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

/// Append tool definitions as a system prompt section.
fn appendToolSystemPrompt(allocator: std.mem.Allocator, result_buf: *std.ArrayList(u8), tools_json: []const u8, tool_choice_instruction: ?[]const u8) !void {
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

const testing = std.testing;

test "stripThinkBlock removes think tags" {
    try testing.expectEqualStrings("Hello", stripThinkBlock("<think>reasoning</think>Hello"));
    try testing.expectEqualStrings("Hello", stripThinkBlock("<think>reasoning</think>\nHello"));
    try testing.expectEqualStrings("Hello", stripThinkBlock("<think>reasoning</think>\n\nHello"));
}

test "stripThinkBlock returns empty for open think tag" {
    try testing.expectEqualStrings("", stripThinkBlock("<think>still thinking..."));
}

test "stripThinkBlock returns text when no think tags" {
    try testing.expectEqualStrings("Hello world", stripThinkBlock("Hello world"));
}

test "splitThinkBlock with complete think block" {
    const result = splitThinkBlock("<think>reasoning here</think>answer here", false);
    try testing.expectEqualStrings("reasoning here", result.reasoning_content.?);
    try testing.expectEqualStrings("answer here", result.content);
}

test "splitThinkBlock with empty reasoning" {
    const result = splitThinkBlock("<think>\n\n</think>\n\nactual content", false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("actual content", result.content);
}

test "splitThinkBlock thinking=true no close tag" {
    const result = splitThinkBlock("<think>partial reasoning", true);
    try testing.expectEqualStrings("partial reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "splitThinkBlock thinking=false no tags" {
    const result = splitThinkBlock("just content", false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("just content", result.content);
}

test "splitThinkBlock strips think prefix in thinking mode" {
    const result = splitThinkBlock("<think>my reasoning", true);
    try testing.expectEqualStrings("my reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "parseToolCalls JSON format" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    // arguments should be valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Tokyo", parsed.value.object.get("location").?.string);
}

test "parseToolCalls multiple calls" {
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name": "func_a", "arguments": {"x": 1}}
        \\</tool_call>
        \\<tool_call>
        \\{"name": "func_b", "arguments": {"y": 2}}
        \\</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("func_a", calls[0].name);
    try testing.expectEqualStrings("func_b", calls[1].name);
}

test "parseToolCalls returns null for no tool calls" {
    const allocator = testing.allocator;
    const result = try parseToolCalls(allocator, "Hello, how can I help you?");
    try testing.expect(result == null);
}

test "parseToolCalls with think block" {
    const allocator = testing.allocator;
    const text = "<think>reasoning</think>\n<tool_call>\n{\"name\": \"search\", \"arguments\": {\"q\": \"test\"}}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("search", calls[0].name);
}

test "parseToolCalls raw JSON without tags" {
    const allocator = testing.allocator;
    const text = "{\"name\": \"get_time\", \"arguments\": {}}";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_time", calls[0].name);
}

test "parseToolCalls Hermes format" {
    const allocator = testing.allocator;
    const text = "<tool_call><function=get_weather><parameter=location>Tokyo</parameter></function></tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    // Should have {"location":"Tokyo"}
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Tokyo", parsed.value.object.get("location").?.string);
}

test "isJsonLiteral" {
    try testing.expect(isJsonLiteral("true"));
    try testing.expect(isJsonLiteral("false"));
    try testing.expect(isJsonLiteral("null"));
    try testing.expect(isJsonLiteral("{\"key\":1}"));
    try testing.expect(isJsonLiteral("[1,2,3]"));
    try testing.expect(isJsonLiteral("42"));
    try testing.expect(isJsonLiteral("3.14"));
    try testing.expect(!isJsonLiteral("hello"));
    try testing.expect(!isJsonLiteral(""));
}

test "appendJsonString escapes special chars" {
    const allocator = testing.allocator;
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);

    try appendJsonString(allocator, &buf, "hello \"world\"\nnew\\line");
    try testing.expectEqualStrings("\"hello \\\"world\\\"\\nnew\\\\line\"", buf.items);
}

test "appendJsonString empty string" {
    const allocator = testing.allocator;
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);

    try appendJsonString(allocator, &buf, "");
    try testing.expectEqualStrings("\"\"", buf.items);
}

test "serializeExtraContext with thinking enabled" {
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    _ = &config;

    const result = try serializeExtraContext(allocator, &config, true);
    defer allocator.free(result);
    // Should contain enable_thinking:true
    try testing.expect(std.mem.indexOf(u8, result, "\"enable_thinking\":true") != null);
}

test "serializeExtraContext with thinking disabled" {
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    _ = &config;

    const result = try serializeExtraContext(allocator, &config, false);
    defer allocator.free(result);
    try testing.expect(std.mem.indexOf(u8, result, "\"enable_thinking\":false") != null);
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
