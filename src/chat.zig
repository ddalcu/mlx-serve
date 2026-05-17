const std = @import("std");
const jinja_c = @cImport({
    @cInclude("jinja_wrapper.h");
});
const tokenizer_mod = @import("tokenizer.zig");
const arch_ds4 = @import("arch/ds4.zig");
const ds4_ffi = @import("ds4_ffi.zig");
const log = @import("log.zig");

const Tokenizer = tokenizer_mod.Tokenizer;

pub const ToolCall = struct {
    id: []const u8,
    name: []const u8,
    arguments: []const u8, // JSON string
};

/// Raw image pixel data for vision encoder (float32, CHW format).
pub const ImageData = struct {
    pixels: []const u8, // Raw float32 bytes [3 * H * W * 4]
    width: u32,
    height: u32,
};

pub const Message = struct {
    role: []const u8,
    content: []const u8,
    tool_calls: ?[]const ToolCall = null,
    tool_call_id: ?[]const u8 = null,
    images: ?[]const ImageData = null, // Preprocessed image data for vision
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
pub fn loadChatConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !ChatConfig {
    const path = try std.fmt.allocPrint(allocator, "{s}/tokenizer_config.json", .{model_dir});
    defer allocator.free(path);

    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var reader_state = file.reader(io, &read_buf);
    const content = try reader_state.interface.allocRemaining(allocator, .limited(10 * 1024 * 1024));
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
        if (std.Io.Dir.openFileAbsolute(io, jinja_path, .{})) |f| {
            defer f.close(io);
            var jinja_buf: [4096]u8 = undefined;
            var jinja_reader = f.reader(io, &jinja_buf);
            break :blk try jinja_reader.interface.allocRemaining(allocator, .limited(1 * 1024 * 1024));
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

/// Render + encode chat messages via the embedded ds4 engine's chat template.
/// ds4 owns its own tokenizer and chat-template renderer — we just map our
/// `Message` array to ds4's `ChatTurn` shape and append the assistant prefix.
///
/// Tool plumbing: the DSV4 chat template doesn't model the `tools` argument
/// natively, so when `tools_json` is non-null we synthesize a fallback —
/// inject the tool catalog into the system message (same machinery the Jinja
/// path uses via `synthesizeToolFallbackMessages`), and rewrite `role:"tool"`
/// turns as `<tool_response>…</tool_response>` user turns, and rewrite
/// assistant `tool_calls` as inline `<tool_call>…</tool_call>` text. Result:
/// the model sees the full tool catalog AND the live tool-result history,
/// which is what makes Agent mode + MCP tool calling work on ds4.
/// Pure prep step extracted so the tool-synthesis decision is unit-testable
/// without booting the ds4 engine. Returns the system content (or null) and
/// the post-synthesis turns ready for `Ds4Engine.encodeChatTranscript`.
/// The returned arena owns all string memory; caller must `deinit()` it.
pub const Ds4PromptPrep = struct {
    arena: std.heap.ArenaAllocator,
    system: ?[]const u8,
    turns: []arch_ds4.Ds4Engine.ChatTurn,

    pub fn deinit(self: *Ds4PromptPrep) void {
        self.arena.deinit();
    }
};

/// Pure helper: run the ds4 chat-template fallback synthesis (tool catalog
/// into system, tool-role rewrite, assistant tool_call inlining) and split
/// the result into (system, turns). Exists so the tool-plumbing decision is
/// covered by a unit test that doesn't need a live engine.
pub fn prepareDs4Prompt(
    parent_allocator: std.mem.Allocator,
    messages: []const Message,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
) !Ds4PromptPrep {
    var arena = std.heap.ArenaAllocator.init(parent_allocator);
    errdefer arena.deinit();
    const arena_alloc = arena.allocator();

    const messages_have_tool_content = messagesHaveToolContent(messages);
    var effective_messages = messages;
    if (tools_json != null or messages_have_tool_content) {
        effective_messages = try synthesizeToolFallbackMessages(
            arena_alloc,
            messages,
            tools_json,
            tool_choice_instruction,
            tools_json != null,
            messages_have_tool_content,
            messages_have_tool_content,
        );
    }

    var system_msg: ?[]const u8 = null;
    var turns_list = std.ArrayList(arch_ds4.Ds4Engine.ChatTurn).empty;
    for (effective_messages) |msg| {
        if (system_msg == null and std.mem.eql(u8, msg.role, "system")) {
            system_msg = msg.content;
            continue;
        }
        // After synthesis only system / user / assistant roles remain.
        try turns_list.append(arena_alloc, .{ .role = msg.role, .content = msg.content });
    }

    return .{
        .arena = arena,
        .system = system_msg,
        .turns = try turns_list.toOwnedSlice(arena_alloc),
    };
}

pub fn encodeChatViaDs4(
    allocator: std.mem.Allocator,
    engine: *arch_ds4.Ds4Engine,
    messages: []const Message,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
) ![]u32 {
    // ── Tool synthesis fallback. ds4's chat template doesn't reference
    //    `tools` or `role == 'tool'`, so without this rewrite:
    //      * tool definitions never reach the model (model can't call them)
    //      * tool-result messages render as empty (model loses prior context)
    //      * assistant tool_call history disappears (multi-turn loops break)
    //    Run via the pure `prepareDs4Prompt` helper so the same logic is
    //    exercised by the unit test at the bottom of this file.
    var prep = try prepareDs4Prompt(allocator, messages, tools_json, tool_choice_instruction);
    defer prep.deinit();

    const think_mode: ds4_ffi.ThinkMode = if (enable_thinking) .high else .none;
    const i32_ids = try engine.encodeChatTranscript(allocator, prep.system, prep.turns, think_mode);
    defer allocator.free(i32_ids);

    const u32_ids = try allocator.alloc(u32, i32_ids.len);
    for (i32_ids, 0..) |t, i| u32_ids[i] = @intCast(t);
    log.debug("  prompt (ds4): {d} messages -> {d} tokens (tools={s})\n", .{
        messages.len,
        u32_ids.len,
        if (tools_json != null) "yes" else "no",
    });
    return u32_ids;
}

/// Detokenize a sequence of token IDs via the ds4 engine. Mirrors
/// `Tokenizer.decode` so server handlers can switch on the LoadedModel without
/// touching the call sites for each path.
pub fn decodeViaDs4(
    allocator: std.mem.Allocator,
    engine: *arch_ds4.Ds4Engine,
    ids: []const u32,
) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    for (ids) |id| {
        const piece = try engine.detokenizeOne(allocator, @intCast(id));
        defer allocator.free(piece);
        try out.appendSlice(allocator, piece);
    }
    return out.toOwnedSlice(allocator);
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

    // Some chat templates (e.g. DeepSeek V4) don't reference `tools` or `role == 'tool'`
    // at all — Jinja silently drops the tool definitions, and tool-result messages render
    // as empty. Detect that gap and synthesize an equivalent system-prompt + user-message
    // form so the model still sees the tool context.
    const tpl = chat_config.chat_template;
    const tpl_has_tools = std.mem.indexOf(u8, tpl, "tools") != null;
    const tpl_has_tool_role = templateReferencesToolRole(tpl);
    const needs_inject_tools = tools_json != null and !tpl_has_tools;
    const needs_rewrite_tool_role = !tpl_has_tool_role and messagesHaveToolContent(messages);

    var fallback_arena: ?std.heap.ArenaAllocator = null;
    defer if (fallback_arena) |*a| a.deinit();
    var effective_messages = messages;
    var effective_tools_json = tools_json;
    if (needs_inject_tools or needs_rewrite_tool_role) {
        fallback_arena = std.heap.ArenaAllocator.init(allocator);
        const arena_alloc = fallback_arena.?.allocator();
        effective_messages = try synthesizeToolFallbackMessages(
            arena_alloc,
            messages,
            tools_json,
            tool_choice_instruction,
            needs_inject_tools,
            !tpl_has_tool_role,
            !tpl_has_tools,
        );
        if (needs_inject_tools) {
            effective_tools_json = null; // already inlined as system content
        }
    }

    // Serialize messages to JSON — Gemma 4 templates handle role:"tool" natively
    // (producing <|turn>tool in the rendered output). No transformation needed.
    const messages_json = try serializeMessagesJson(allocator, effective_messages);
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
    if (effective_tools_json) |tj| {
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
        return try collapseDoubledThinkTags(allocator, std.mem.span(ptr));
    }

    if (jinja_c.jinja_last_error()) |e| {
        log.debug("  jinja error: {s}, using fallback\n", .{std.mem.span(e)});
    }
    return fallbackFormatChat(allocator, messages, chat_config, tools_json, tool_choice_instruction, enable_thinking);
}

/// DSV4's chat template emits `</think></think>` between every user→assistant
/// pair in chat mode — the user role appends `</think>` after `<|Assistant|>`
/// and the assistant role prepends `</think>` before its content. The doubling
/// is structurally invalid (no legitimate rendered prompt should contain
/// `</think></think>`), and at 2-bit DQ it pushes the model into prompt-echo
/// / phrase-loop collapse on the second turn (a 24-token "Hello / Hi! / What
/// is 2+2?" chat reliably degenerates). The HF tokenizer renders the same
/// doubling so this is a template-side artifact, not our renderer; mlx-lm
/// must be tolerating it via reduction-order quirks we don't share. Strip
/// it here. Safe across all model families because `</think></think>` never
/// appears legitimately.
fn collapseDoubledThinkTags(allocator: std.mem.Allocator, rendered: []const u8) ![]u8 {
    const tag = "</think>";
    const needle = "</think></think>";
    if (std.mem.indexOf(u8, rendered, needle) == null) {
        return allocator.dupe(u8, rendered);
    }
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.ensureTotalCapacity(allocator, rendered.len);
    var i: usize = 0;
    while (i < rendered.len) {
        // If we see `</think></think>...`, drop the FIRST `</think>` and let
        // the next iteration handle whatever follows — that way 3x, 4x, …
        // also collapse to a single tag without a separate pass.
        if (i + needle.len <= rendered.len and
            std.mem.eql(u8, rendered[i .. i + needle.len], needle))
        {
            i += tag.len; // skip the first `</think>`; next iter writes the survivor
        } else {
            try out.append(allocator, rendered[i]);
            i += 1;
        }
    }
    return out.toOwnedSlice(allocator);
}

/// True if the template contains a branch keyed on the literal `tool` role string
/// (`'tool'` or `"tool"` as a standalone token, not `tool_calls` / `tool_call_id`).
fn templateReferencesToolRole(tpl: []const u8) bool {
    const patterns = [_][]const u8{ "'tool'", "\"tool\"" };
    for (patterns) |p| {
        if (std.mem.indexOf(u8, tpl, p) != null) return true;
    }
    return false;
}

/// True if any message has `role: "tool"` or an assistant message with tool_calls.
fn messagesHaveToolContent(messages: []const Message) bool {
    for (messages) |m| {
        if (std.mem.eql(u8, m.role, "tool")) return true;
        if (m.tool_calls != null) return true;
    }
    return false;
}

/// Synthesize a messages array suitable for templates that have no tool/role support.
/// - inject_tool_prompt: prepend (or merge into) a system message with the tool prompt
/// - rewrite_tool_role: rewrite role:"tool" messages to role:"user" with <tool_response> wrapping
/// - rewrite_tool_calls: rewrite assistant messages that carry tool_calls into plain
///   content emitting <tool_call>...</tool_call> blocks (matches what we instruct the
///   model to produce, so multi-turn context stays consistent).
fn synthesizeToolFallbackMessages(
    arena: std.mem.Allocator,
    messages: []const Message,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    inject_tool_prompt: bool,
    rewrite_tool_role: bool,
    rewrite_tool_calls: bool,
) ![]Message {
    var out = std.ArrayList(Message).empty;

    var tool_prompt: ?[]const u8 = null;
    if (inject_tool_prompt and tools_json != null) {
        var buf = std.ArrayList(u8).empty;
        try appendToolSystemPrompt(arena, &buf, tools_json.?, tool_choice_instruction);
        tool_prompt = try buf.toOwnedSlice(arena);
    }

    var injected = false;
    if (tool_prompt) |tp| {
        if (messages.len == 0 or !std.mem.eql(u8, messages[0].role, "system")) {
            try out.append(arena, .{ .role = "system", .content = tp });
            injected = true;
        }
    }

    for (messages, 0..) |msg, i| {
        if (tool_prompt) |tp| {
            if (!injected and i == 0 and std.mem.eql(u8, msg.role, "system")) {
                const merged = if (msg.content.len > 0)
                    try std.fmt.allocPrint(arena, "{s}\n\n{s}", .{ msg.content, tp })
                else
                    try arena.dupe(u8, tp);
                try out.append(arena, .{
                    .role = "system",
                    .content = merged,
                    .tool_calls = msg.tool_calls,
                    .tool_call_id = msg.tool_call_id,
                    .images = msg.images,
                });
                injected = true;
                continue;
            }
        }

        if (rewrite_tool_role and std.mem.eql(u8, msg.role, "tool")) {
            const wrapped = try std.fmt.allocPrint(arena, "<tool_response>\n{s}\n</tool_response>", .{msg.content});
            try out.append(arena, .{
                .role = "user",
                .content = wrapped,
            });
            continue;
        }

        if (rewrite_tool_calls and std.mem.eql(u8, msg.role, "assistant") and msg.tool_calls != null) {
            var buf = std.ArrayList(u8).empty;
            if (msg.content.len > 0) {
                try buf.appendSlice(arena, msg.content);
                try buf.append(arena, '\n');
            }
            for (msg.tool_calls.?) |tc| {
                try buf.appendSlice(arena, "<tool_call>\n{\"name\": \"");
                try buf.appendSlice(arena, tc.name);
                try buf.appendSlice(arena, "\", \"arguments\": ");
                try buf.appendSlice(arena, tc.arguments);
                try buf.appendSlice(arena, "}\n</tool_call>\n");
            }
            const owned = try buf.toOwnedSlice(arena);
            try out.append(arena, .{
                .role = "assistant",
                .content = std.mem.trimEnd(u8, owned, "\n"),
                .images = msg.images,
            });
            continue;
        }

        try out.append(arena, msg);
    }

    return out.toOwnedSlice(arena);
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

/// Strip `<think>...</think>` or `<|channel>thought\n...<channel|>` block from model output.
pub fn stripThinkBlock(text: []const u8) []const u8 {
    // Gemma 4 style: <|channel>thought\n...<channel|>
    if (std.mem.indexOf(u8, text, "<channel|>")) |end| {
        return std.mem.trimStart(u8, text[end + 10 ..], "\n ");
    }
    // Standard style: <think>...</think>
    if (std.mem.indexOf(u8, text, "</think>")) |think_end| {
        return std.mem.trimStart(u8, text[think_end + 8 ..], "\n ");
    }
    if (std.mem.startsWith(u8, text, "<think>")) return text[0..0];
    if (std.mem.startsWith(u8, text, "<|channel>thought")) return text[0..0];
    return text;
}

pub const ThinkSplit = struct {
    reasoning_content: ?[]const u8,
    content: []const u8,
};

/// Split model output into reasoning_content and content.
/// Handles both `<think>...</think>` and Gemma 4's `<|channel>thought\n...<channel|>`.
pub fn splitThinkBlock(text: []const u8, thinking: bool) ThinkSplit {
    // Gemma 4 style: <|channel>thought\n...<channel|>\n<|channel>\ncontent
    if (std.mem.indexOf(u8, text, "<channel|>")) |end| {
        const think_tag = "<|channel>thought\n";
        const reasoning_start: usize = if (std.mem.startsWith(u8, text, think_tag)) think_tag.len else if (std.mem.startsWith(u8, text, "<|channel>thought")) "<|channel>thought".len else 0;
        const reasoning = std.mem.trim(u8, text[reasoning_start..end], "\n ");
        var content = std.mem.trimStart(u8, text[end + 10 ..], "\n ");
        // Strip the content channel tag: <|channel>\n or <|channel>
        if (std.mem.startsWith(u8, content, "<|channel>\n")) {
            content = content[11..];
        } else if (std.mem.startsWith(u8, content, "<|channel>")) {
            content = content[10..];
        }
        content = std.mem.trimStart(u8, content, "\n ");
        return .{
            .reasoning_content = if (reasoning.len > 0) reasoning else null,
            .content = content,
        };
    }
    // Standard style: <think>...</think>
    if (std.mem.indexOf(u8, text, "</think>")) |end| {
        const reasoning_start: usize = if (std.mem.startsWith(u8, text, "<think>")) 7 else 0;
        const reasoning = std.mem.trim(u8, text[reasoning_start..end], "\n ");
        const content = std.mem.trimStart(u8, text[end + 8 ..], "\n ");
        return .{
            .reasoning_content = if (reasoning.len > 0) reasoning else null,
            .content = content,
        };
    }
    if (thinking) {
        // Unclosed think block: split policy depends on whether the model's
        // output begins with a literal opener.
        //   • Literal opener present → model definitely entered thinking but
        //     ran out of tokens / didn't close. Treat as reasoning.
        //   • No literal opener → template likely injected the opener and the
        //     model either didn't think or thought without closing. Either way,
        //     defaulting to content keeps the answer visible to the user.
        if (std.mem.startsWith(u8, text, "<think>") or std.mem.startsWith(u8, text, "<|channel>thought")) {
            const start: usize = if (std.mem.startsWith(u8, text, "<think>")) 7 else if (std.mem.startsWith(u8, text, "<|channel>thought\n")) "<|channel>thought\n".len else "<|channel>thought".len;
            const reasoning = std.mem.trimStart(u8, text[start..], "\n ");
            return .{ .reasoning_content = if (reasoning.len > 0) reasoning else null, .content = "" };
        }
        return .{ .reasoning_content = null, .content = std.mem.trimStart(u8, text, "\n ") };
    }
    return .{ .reasoning_content = null, .content = text };
}

/// Streaming-only: should the chat-completion SSE path defer flushing
/// because `buf` is on track to become a tool call?
///
/// True when:
///   * `buf` contains `<tool` followed by a valid tag terminator
///     (`>`, ` `, `\t`, `\n`, `_`, `|`) somewhere — that's any of the
///     accepted families: `<tool>`, `<tool …>`, `<tool_call…>`, `<tool_calls…>`,
///     `<tool_request…>`, `<tool_requests…>` (mirrors `parseToolCalls`).
///   * `buf` contains the Gemma 4 `<|tool_call` substring.
///   * `buf[0] == '{'` and `buf` contains `"name"` (raw JSON tool-call shape).
///   * `buf` ends with a partial prefix that could grow into one of the above
///     in the next token (`<`, `<t`, `<to`, `<too`, `<tool`, or any prefix
///     of `<|tool_call`).
///
/// Conservatively false on `<toolkit>`, `<toolbar>`, etc. — anything where
/// the char after `<tool` isn't a valid tag terminator. That keeps prose
/// that happens to mention HTML-ish tags flowing through normally.
pub fn streamShouldBufferForTools(buf: []const u8) bool {
    if (buf.len == 0) return false;

    // Raw JSON tool-call shape: starts with `{` and has `"name"` somewhere.
    if (buf[0] == '{' and std.mem.indexOf(u8, buf, "\"name\"") != null) return true;

    // Gemma 4 fully-formed canonical open.
    if (std.mem.indexOf(u8, buf, "<|tool_call") != null) return true;

    // `<tool…` family: walk every `<tool` occurrence, accept the first one
    // whose terminator is valid (mirrors parseToolCalls' acceptance rule).
    var scan: usize = 0;
    while (std.mem.indexOf(u8, buf[scan..], "<tool")) |rel| {
        const after = scan + rel + "<tool".len;
        if (after >= buf.len) return true; // truncated mid-prefix → in progress
        const c = buf[after];
        if (c == '>' or c == ' ' or c == '\t' or c == '\n' or c == '_' or c == '|') return true;
        scan = after;
    }

    // Trailing partial prefixes — the next streamed token could complete any
    // of these into a real tool open. Order doesn't matter; first endsWith
    // hit wins. Listed shortest-first for legibility.
    const tail_prefixes = [_][]const u8{
        "<",     "<t",     "<to",     "<too",     "<tool",
        "<|",    "<|t",    "<|to",    "<|too",    "<|tool",
        "<|tool_", "<|tool_c", "<|tool_ca", "<|tool_cal",
    };
    for (tail_prefixes) |p| {
        if (std.mem.endsWith(u8, buf, p)) return true;
    }
    return false;
}

/// Parse tool calls from model output text.
pub fn parseToolCalls(allocator: std.mem.Allocator, text: []const u8) !?[]ParsedToolCall {
    // Strip thinking blocks if present
    var effective_text = text;
    if (std.mem.indexOf(u8, text, "<channel|>")) |end| {
        effective_text = std.mem.trimStart(u8, text[end + 10 ..], "\n ");
    } else if (std.mem.indexOf(u8, text, "</think>")) |think_end| {
        effective_text = std.mem.trimStart(u8, text[think_end + 8 ..], "\n ");
    }

    var calls = std.ArrayList(ParsedToolCall).empty;
    errdefer {
        for (calls.items) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        calls.deinit(allocator);
    }

    // Look for `<tool…>…</tool…>` patterns. DSV4-Flash hallucinates the
    // spec aggressively; we accept the entire training-bias family:
    //   • `<tool_call>JSON</tool_call>`                      (canonical Hermes)
    //   • `<tool_call name="X">JSON</tool_call>`             (attribute style)
    //   • `<tool_call …>JSON</tool_request>`                 (mismatched close)
    //   • `<tool_calls>JSON</tool_calls>`                    (plural form)
    //   • `<tool_calls name="X">{args}</tool_calls>`         (plural + attr)
    //   • `<tool name="X">{args}</tool_calls>`               (bare-tool open, plural close — nested wrappers)
    //   • `<tool>{json}</tool>`                              (bare tag, body has name+args)
    //   • `<tool name="X" arguments="{...}"/>`               (SELF-CLOSING — DSV4 emits this routinely;
    //                                                          parseSelfClosingToolTag handles broken-quote
    //                                                          variants like `…"}'/>` too)
    // Approach: search for the prefix `<tool` (5 chars). Skip the Gemma 4
    // close marker `<tool_call|>`. Try the self-closing form first (cheap
    // attribute scan + balanced-JSON extraction; tolerates the model's
    // mismatched quotes). If that doesn't match, fall through to the
    // open/close-marker path: scan to the next `>`, optionally capturing
    // `name="X"` / `name='X'`, then locate the earliest of `</tool>`,
    // `</tool_call>`, `</tool_calls>`, `</tool_request>`, `</tool_requests>`.
    // When the body fails to parse, advance only past the OPENING `>` so any
    // inner `<tool…>` blocks get a chance — this is what makes the outer
    // `<tool_calls>` wrapper case work.
    var search_pos: usize = 0;
    while (search_pos < effective_text.len) {
        const rel = std.mem.indexOf(u8, effective_text[search_pos..], "<tool") orelse break;
        const after_tool = search_pos + rel + "<tool".len;
        if (after_tool >= effective_text.len) break;

        // Reject anything that isn't actually a tool open tag — the char
        // right after `<tool` must be `>`, whitespace, `_` (suffix coming),
        // or `|` (Gemma 4 close marker). Anything else is text like
        // `<toolkit>` or `<toolbar>`; advance past `<tool` and keep scanning.
        const next = effective_text[after_tool];
        if (next != '>' and next != ' ' and next != '\t' and next != '\n' and next != '_' and next != '|') {
            search_pos = after_tool;
            continue;
        }
        // Gemma 4 close marker: `<tool_call|>`. Detected when after `<tool`
        // we see `_call|`. Let the Gemma 4 branch below pick this up.
        if (next == '_'
            and after_tool + 6 <= effective_text.len
            and std.mem.eql(u8, effective_text[after_tool .. after_tool + 6], "_call|"))
        {
            search_pos = after_tool + 6;
            continue;
        }
        // If suffix is `_`, the only accepted continuations are
        // `_call`, `_calls`, `_request`, `_requests`. Reject other tags
        // like `<tool_undefined>` to avoid false positives.
        if (next == '_') {
            const accepted = [_][]const u8{ "_call", "_calls", "_request", "_requests" };
            var matched = false;
            for (accepted) |suf| {
                if (after_tool + suf.len > effective_text.len) continue;
                if (!std.mem.eql(u8, effective_text[after_tool .. after_tool + suf.len], suf)) continue;
                // Char right after the suffix must be tag-terminating; otherwise
                // `_call` could swallow `<tool_called_X>`.
                const post = after_tool + suf.len;
                if (post >= effective_text.len) break;
                const c = effective_text[post];
                if (c == '>' or c == ' ' or c == '\t' or c == '\n' or c == '/' or c == '|') {
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                search_pos = after_tool;
                continue;
            }
        }

        // Self-closing `<tool …/>` form first — DSV4 emits this routinely
        // and the open/close scan below won't find a `</tool*>` close, so
        // without this branch the call gets dropped silently.
        const tag_origin = after_tool - "<tool".len;
        if (parseSelfClosingToolTag(effective_text[tag_origin..])) |sc| {
            const name_owned = try allocator.dupe(u8, sc.name);
            errdefer allocator.free(name_owned);
            const args_owned = try allocator.dupe(u8, sc.arguments);
            try calls.append(allocator, .{ .name = name_owned, .arguments = args_owned });
            search_pos = tag_origin + sc.consumed;
            continue;
        }

        // Scan the opening tag to its `>`, capturing optional name attr.
        var attr_name: ?[]const u8 = null;
        var content_start: usize = 0;
        {
            var i: usize = after_tool;
            // Bound the scan so a malformed input can't run wild.
            const limit = @min(effective_text.len, after_tool + 256);
            while (i < limit and effective_text[i] != '>') : (i += 1) {
                if (effective_text[i] != 'n') continue;
                if (i + 5 > effective_text.len) continue;
                if (!std.mem.eql(u8, effective_text[i .. i + 5], "name=")) continue;
                const q_pos = i + 5;
                if (q_pos >= effective_text.len) break;
                const quote = effective_text[q_pos];
                if (quote != '"' and quote != '\'') continue;
                const val_start = q_pos + 1;
                const val_end_rel = std.mem.indexOfScalar(u8, effective_text[val_start..], quote) orelse break;
                attr_name = effective_text[val_start .. val_start + val_end_rel];
                i = val_start + val_end_rel;
            }
            if (i >= limit or effective_text[i] != '>') {
                // Unclosed opening tag — skip past `<tool` and keep going.
                search_pos = after_tool;
                continue;
            }
            content_start = i + 1;
        }
        // Find the close marker — earliest of any accepted close.
        const close_markers = [_][]const u8{
            "</tool_call>", "</tool_calls>",
            "</tool_request>", "</tool_requests>",
            "</tool>",
        };
        var close_rel: ?usize = null;
        var close_len: usize = 0;
        for (close_markers) |marker| {
            if (std.mem.indexOf(u8, effective_text[content_start..], marker)) |found| {
                if (close_rel == null or found < close_rel.?) {
                    close_rel = found;
                    close_len = marker.len;
                }
            }
        }
        if (close_rel == null) {
            // No `</tool…>` close in sight. Two ways this happens in the
            // wild: (1) the model emitted EOS mid-tool-call (`<tool_call>\n{
            // …well-formed args…}` then nothing); (2) max_tokens truncated.
            // The args object itself is usually intact — snap a balanced
            // JSON object from right after the open tag and try to parse it.
            if (balancedJsonObject(effective_text[content_start..])) |json_body| {
                const json_off = @intFromPtr(json_body.ptr) - @intFromPtr(effective_text[content_start..].ptr);
                const advance_to = content_start + json_off + json_body.len;
                // Hermes shape (top-level "name" + "arguments"): use as-is.
                if (tryParseJsonToolCall(allocator, json_body)) |tc| {
                    try calls.append(allocator, tc);
                    search_pos = advance_to;
                    continue;
                }
                // Attribute shape — body is JUST the args, take name from attr.
                if (attr_name) |an| {
                    const name_owned = try allocator.dupe(u8, an);
                    errdefer allocator.free(name_owned);
                    const args_owned = try allocator.dupe(u8, json_body);
                    try calls.append(allocator, .{ .name = name_owned, .arguments = args_owned });
                    search_pos = advance_to;
                    continue;
                }
            }
            break;
        }
        const content = std.mem.trim(u8, effective_text[content_start .. content_start + close_rel.?], " \t\n\r");

        // Pre-clean the body before the parse attempts. Two DSV4 quirks:
        //   • The args object may be wrapped in `<parameters>…</parameters>`.
        //   • The model sometimes emits an extra trailing `}` after the
        //     proper JSON close. Find the first balanced JSON object via
        //     depth tracking and use just that.
        const unwrapped = stripParametersWrapper(content);
        const balanced: []const u8 = balancedJsonObject(unwrapped) orelse unwrapped;

        // Try to extract a tool call from the body. Three shapes, tried in
        // priority order:
        //   1. Attribute form (`name="X"` captured, body is JUST args).
        //   2. Canonical Hermes JSON ({"name":"X","arguments":{...}}).
        //   3. Hermes function-tag format (parseHermesToolCall).
        var parsed_ok = false;
        if (attr_name) |an| {
            if (std.json.parseFromSlice(std.json.Value, allocator, balanced, .{})) |parsed| {
                defer parsed.deinit();
                if (parsed.value == .object) {
                    const name_owned = try allocator.dupe(u8, an);
                    errdefer allocator.free(name_owned);
                    const args_owned = try allocator.dupe(u8, balanced);
                    try calls.append(allocator, .{ .name = name_owned, .arguments = args_owned });
                    parsed_ok = true;
                }
            } else |_| {}
        }
        if (!parsed_ok) {
            if (tryParseJsonToolCall(allocator, balanced)) |tc| {
                try calls.append(allocator, tc);
                parsed_ok = true;
            }
        }
        if (!parsed_ok) {
            if (parseHermesToolCall(allocator, content)) |tc| {
                try calls.append(allocator, tc);
                parsed_ok = true;
            }
        }

        if (parsed_ok) {
            // Body consumed cleanly — advance past the close marker.
            search_pos = content_start + close_rel.? + close_len;
        } else {
            // Body wasn't a tool call. Advance only past the opening `>`
            // so inner `<tool…>` blocks (if any) still get parsed. This is
            // what makes the outer-wrapper case work: the outer
            // `<tool_calls>` has no JSON body, but its content contains
            // real `<tool name="X">` inner calls.
            search_pos = content_start;
        }
    }

    // Gemma 4 format: <|tool_call>call:name{args}<tool_call|>
    if (calls.items.len == 0) {
        search_pos = 0;
        while (search_pos < effective_text.len) {
            const tag_start = std.mem.indexOf(u8, effective_text[search_pos..], "<|tool_call>") orelse break;
            const content_start = search_pos + tag_start + "<|tool_call>".len;
            const tag_end_opt = std.mem.indexOf(u8, effective_text[content_start..], "<tool_call|>");
            const content = if (tag_end_opt) |tag_end|
                std.mem.trim(u8, effective_text[content_start .. content_start + tag_end], " \t\n\r")
            else
                // Incomplete tool call (model hit EOS before closing tag) — use rest of text
                std.mem.trim(u8, effective_text[content_start..], " \t\n\r");

            search_pos = if (tag_end_opt) |tag_end|
                content_start + tag_end + "<tool_call|>".len
            else
                effective_text.len;

            if (parseGemma4ToolCall(allocator, content)) |tc| {
                try calls.append(allocator, tc);
            } else {
                log.info("  [tool-parse] Gemma4 parse FAILED for: {s}\n", .{content[0..@min(content.len, 200)]});
            }
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

/// Strip a leading `<parameters>` / trailing `</parameters>` pair if both
/// are present. DSV4 occasionally wraps the args object in this XML
/// container — leaving it in place would defeat the JSON parser. Trims
/// surrounding whitespace.
fn stripParametersWrapper(content: []const u8) []const u8 {
    const trimmed = std.mem.trim(u8, content, " \t\n\r");
    const open = "<parameters>";
    const close = "</parameters>";
    if (trimmed.len < open.len + close.len) return trimmed;
    if (!std.mem.startsWith(u8, trimmed, open)) return trimmed;
    if (!std.mem.endsWith(u8, trimmed, close)) return trimmed;
    const inner = trimmed[open.len .. trimmed.len - close.len];
    return std.mem.trim(u8, inner, " \t\n\r");
}

/// Return the substring of `content` covering the first brace-balanced JSON
/// object, or null if no balanced object is present. Handles quoted strings
/// (so `}` inside a string doesn't close the object) and escape sequences.
/// Used to tolerate the DSV4 `{"…"}}` extra-trailing-`}` artifact.
/// Try to parse a tool tag whose body lives entirely in XML-ish
/// attributes. Two terminator shapes are accepted:
///
///   Self-closing /> :
///     <tool name="shell" arguments="{...}"/>
///     <tool name="X" arguments='{...}' />
///     <tool name="X" arguments="{...}'/>             (broken quote — DSV4)
///
///   Empty-body explicit close ></tool*> :
///     <tool_call name="cwd" arguments="{...}"></tool_call>
///     <tool name="X" arguments="{...}"></tool_calls>  (mismatched close OK)
///
/// `slice` MUST start with `<tool` (the caller has already validated the
/// terminator char after that prefix). Returns name + arguments slices into
/// `slice` plus the number of bytes consumed (up to and including the final
/// `>` or `</tool*>` close), or null if this isn't a shape we can parse.
///
/// We never trust the XML quoting around `arguments=` — the model frequently
/// uses `"` to open and `'` to close, or doesn't escape inner `"` at all.
/// Instead we expect a `{` immediately (optionally after a single opening
/// quote) and snap a balanced JSON object via `balancedJsonObject`. That
/// strategy is what makes the DSV4 fragment parse cleanly even though the
/// surrounding XML is malformed.
fn parseSelfClosingToolTag(slice: []const u8) ?struct { name: []const u8, arguments: []const u8, consumed: usize } {
    if (slice.len < "<tool".len) return null;
    if (!std.mem.startsWith(u8, slice, "<tool")) return null;
    // Bound the scan window so a runaway tag can't pin us forever.
    const limit = @min(slice.len, 8192);

    // 1. Locate `name="X"` (or `'X'`) attribute. We don't enforce ordering
    //    against `arguments=` — DSV4 sometimes interleaves them.
    var name: ?[]const u8 = null;
    var name_end: usize = "<tool".len;
    {
        var i: usize = "<tool".len;
        while (i + "name=".len < limit) : (i += 1) {
            if (slice[i] != 'n') continue;
            if (!std.mem.eql(u8, slice[i .. i + "name=".len], "name=")) continue;
            const q_pos = i + "name=".len;
            if (q_pos >= limit) break;
            const quote = slice[q_pos];
            if (quote != '"' and quote != '\'') continue;
            const val_start = q_pos + 1;
            const val_end_rel = std.mem.indexOfScalar(u8, slice[val_start..limit], quote) orelse break;
            name = slice[val_start .. val_start + val_end_rel];
            name_end = val_start + val_end_rel + 1;
            break;
        }
    }

    // 2. Locate `arguments=` and snap the next balanced JSON object as
    //    the value. We deliberately don't parse the XML quote — see
    //    function docs.
    var args: ?[]const u8 = null;
    var args_end: usize = 0;
    {
        var i: usize = "<tool".len;
        while (i + "arguments=".len < limit) : (i += 1) {
            if (slice[i] != 'a') continue;
            if (!std.mem.eql(u8, slice[i .. i + "arguments=".len], "arguments=")) continue;
            var j: usize = i + "arguments=".len;
            // Skip an opening quote if present.
            if (j < limit and (slice[j] == '"' or slice[j] == '\'')) j += 1;
            // Skip whitespace between the quote and the JSON object.
            while (j < limit and (slice[j] == ' ' or slice[j] == '\t')) j += 1;
            if (j >= limit or slice[j] != '{') break;
            const sub = slice[j..limit];
            const json = balancedJsonObject(sub) orelse break;
            args = json;
            const json_off = @intFromPtr(json.ptr) - @intFromPtr(sub.ptr);
            args_end = j + json_off + json.len;
            break;
        }
    }

    if (name == null or args == null) return null;

    // 3. Scan forward from the end of whichever attribute landed last to
    //    the terminator. Tolerate a stray closing quote (model often closes
    //    args with `'` even though the opener was `"`) and any intervening
    //    whitespace. We accept either:
    //      a) `/>` — self-closing.
    //      b) `>` followed (possibly across whitespace) by `</tool*>` —
    //         empty body, explicit close marker.
    var k: usize = @max(name_end, args_end);
    while (k < limit and (slice[k] == ' ' or slice[k] == '\t' or slice[k] == '\n' or slice[k] == '\r' or slice[k] == '"' or slice[k] == '\'')) k += 1;
    if (k >= limit) return null;

    if (slice[k] == '/') {
        k += 1;
        if (k >= limit or slice[k] != '>') return null;
        return .{ .name = name.?, .arguments = args.?, .consumed = k + 1 };
    }
    if (slice[k] == '>') {
        var w: usize = k + 1;
        while (w < slice.len and (slice[w] == ' ' or slice[w] == '\t' or slice[w] == '\n' or slice[w] == '\r')) w += 1;
        const close_markers = [_][]const u8{
            "</tool_call>",   "</tool_calls>",
            "</tool_request>", "</tool_requests>",
            "</tool>",
        };
        for (close_markers) |m| {
            if (w + m.len <= slice.len and std.mem.eql(u8, slice[w .. w + m.len], m)) {
                return .{ .name = name.?, .arguments = args.?, .consumed = w + m.len };
            }
        }
    }
    return null;
}

fn balancedJsonObject(content: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, content, " \t\n\r");
    const start = std.mem.indexOfScalar(u8, trimmed, '{') orelse return null;
    var depth: i32 = 0;
    var in_string: bool = false;
    var escape: bool = false;
    var i: usize = start;
    while (i < trimmed.len) : (i += 1) {
        const c = trimmed[i];
        if (escape) {
            escape = false;
            continue;
        }
        if (in_string) {
            if (c == '\\') {
                escape = true;
                continue;
            }
            if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '{') {
            depth += 1;
        } else if (c == '}') {
            depth -= 1;
            if (depth == 0) return trimmed[start .. i + 1];
            if (depth < 0) return null; // mismatched — give up
        }
    }
    return null;
}

fn tryParseJsonToolCall(allocator: std.mem.Allocator, text: []const u8) ?ParsedToolCall {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, text, .{}) catch blk: {
        // Strict parse failed. Try a chain of repairs for known Qwen MoE shapes:
        //   1. {"name":"shell", {"command":"ls"}}           — missing `"arguments":` key entirely
        //   2. {"name":"shell", arguments":{"command":..}}   — missing OPENING quote on `arguments`
        // Repairs are cheap and run only on the parse-failure path.
        const repaired = repairBrokenToolCallJson(allocator, text) orelse return null;
        defer allocator.free(repaired);
        const reparsed = std.json.parseFromSlice(std.json.Value, allocator, repaired, .{}) catch return null;
        break :blk reparsed;
    };
    defer parsed.deinit();

    if (parsed.value != .object) return null;
    var obj = parsed.value.object;

    // Qwen 3.5/3.6 MoE sometimes emits nested-name garbage like
    //   {"name":{"name":{"name":"write","arguments":{...}}}}
    // Walk down through up to a few levels of nested "name" objects to find the
    // leaf object that has a string name + arguments/parameters. Observed in the
    // wild with pi + Qwen3.6-35B in non-thinking mode.
    {
        var depth: u8 = 0;
        while (depth < 4) : (depth += 1) {
            const nv = obj.get("name") orelse return null;
            switch (nv) {
                .string => break,
                .object => |inner| obj = inner,
                else => return null,
            }
        }
    }

    const name_val = obj.get("name") orelse return null;
    if (name_val != .string) return null;

    const args_str: []const u8 = blk: {
        if (obj.get("arguments") orelse obj.get("parameters")) |args_val| {
            break :blk switch (args_val) {
                .object => std.json.Stringify.valueAlloc(allocator, args_val, .{}) catch return null,
                .string => |s| allocator.dupe(u8, s) catch return null,
                else => return null,
            };
        }
        // Flat shape (e.g. Qwen MoE): {"name":"shell","command":"ls"} —
        // parameters live at top level. Synthesize an arguments object from
        // every non-metadata key.
        var flat_map: std.json.ObjectMap = .empty;
        defer flat_map.deinit(allocator);
        var it = obj.iterator();
        while (it.next()) |entry| {
            const k = entry.key_ptr.*;
            if (std.mem.eql(u8, k, "name") or
                std.mem.eql(u8, k, "id") or
                std.mem.eql(u8, k, "type"))
            {
                continue;
            }
            flat_map.put(allocator, k, entry.value_ptr.*) catch return null;
        }
        if (flat_map.count() == 0) return null;
        const flat_value = std.json.Value{ .object = flat_map };
        break :blk std.json.Stringify.valueAlloc(allocator, flat_value, .{}) catch return null;
    };

    return .{
        .name = allocator.dupe(u8, name_val.string) catch {
            allocator.free(args_str);
            return null;
        },
        .arguments = args_str,
    };
}

/// Run known Qwen-MoE-broken-JSON repairs in sequence; return the first
/// successfully-repaired string (allocator-owned), or null if none match.
fn repairBrokenToolCallJson(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    if (repairFlatBraceToolCallJson(allocator, text)) |s| return s;
    if (repairUnquotedArgsKey(allocator, text)) |s| return s;
    return null;
}

/// Repair `{"name":"x", arguments":{...}}` (missing OPENING quote on
/// `arguments`/`parameters`) by injecting the quote.
fn repairUnquotedArgsKey(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\n\r");
    // Look for the unquoted-key pattern. Both `arguments` and `parameters`
    // have been observed; both are short fixed strings, so direct search is fine.
    const candidates = [_][]const u8{ ", arguments\":", ",arguments\":", ", parameters\":", ",parameters\":" };
    for (candidates) |needle| {
        if (std.mem.indexOf(u8, trimmed, needle)) |at| {
            const insert_at = at + 1; // right after the comma
            // Skip leading whitespace after the comma so the injected `"` lands on the identifier.
            var p = insert_at;
            while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t')) p += 1;
            return std.fmt.allocPrint(allocator, "{s}\"{s}", .{ trimmed[0..p], trimmed[p..] }) catch null;
        }
    }
    return null;
}

/// Repair the Qwen-MoE-broken shape `{"name":"x", {"k":"v"}}` by injecting
/// the missing `"arguments":` key between the comma and the inner object.
/// Returns an allocator-owned repaired JSON string, or null if the pattern
/// doesn't match.
fn repairFlatBraceToolCallJson(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\n\r");
    if (trimmed.len < 4 or trimmed[0] != '{') return null;

    // Locate `"name"` key.
    const name_at = std.mem.indexOf(u8, trimmed, "\"name\"") orelse return null;
    var p = name_at + 6;
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t')) p += 1;
    if (p >= trimmed.len or trimmed[p] != ':') return null;
    p += 1;
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t')) p += 1;
    if (p >= trimmed.len or trimmed[p] != '"') return null;

    // Skip the string value.
    p += 1;
    while (p < trimmed.len and trimmed[p] != '"') {
        if (trimmed[p] == '\\' and p + 1 < trimmed.len) p += 2 else p += 1;
    }
    if (p >= trimmed.len) return null;
    p += 1;

    // Expect `, {` (any whitespace).
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t' or trimmed[p] == '\n')) p += 1;
    if (p >= trimmed.len or trimmed[p] != ',') return null;
    p += 1;
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t' or trimmed[p] == '\n')) p += 1;
    if (p >= trimmed.len or trimmed[p] != '{') return null;

    return std.fmt.allocPrint(allocator, "{s}\"arguments\":{s}", .{ trimmed[0..p], trimmed[p..] }) catch null;
}

/// Parse Gemma 4 tool call format: "call:function_name{json_args}"
fn parseGemma4ToolCall(allocator: std.mem.Allocator, content: []const u8) ?ParsedToolCall {
    const prefix = "call:";
    if (!std.mem.startsWith(u8, content, prefix)) return null;
    const after_prefix = content[prefix.len..];

    // Find the opening brace
    const brace_pos = std.mem.indexOf(u8, after_prefix, "{") orelse return null;
    const name = std.mem.trim(u8, after_prefix[0..brace_pos], " \t\n\r");
    if (name.len == 0) return null;

    var args_str = after_prefix[brace_pos..];

    // Gemma 4 uses {{ }} (double braces) for literal braces in Jinja templates.
    // The model often generates {{"key":"value"}} — unwrap the outer braces.
    if (args_str.len >= 4 and std.mem.startsWith(u8, args_str, "{{") and std.mem.endsWith(u8, args_str, "}}")) {
        args_str = args_str[1 .. args_str.len - 1]; // strip one layer of braces
    }

    // Try JSON first (model sometimes outputs valid JSON arguments)
    if (std.json.parseFromSlice(std.json.Value, allocator, args_str, .{})) |parsed| {
        defer parsed.deinit();
        if (parsed.value == .object) {
            return .{
                .name = allocator.dupe(u8, name) catch return null,
                .arguments = allocator.dupe(u8, args_str) catch return null,
            };
        }
    } else |_| {}

    // Convert Gemma 4 custom format to JSON:
    // {key:<|"|>value<|"|>,key2:<|"|>value2<|"|>} → {"key":"value","key2":"value2"}
    const json = convertGemma4ArgsToJson(allocator, args_str) orelse {
        log.info("  [tool-parse] convertGemma4ArgsToJson FAILED for: {s}\n", .{args_str[0..@min(args_str.len, 200)]});
        return null;
    };
    return .{
        .name = allocator.dupe(u8, name) catch return null,
        .arguments = json,
    };
}

/// Convert Gemma 4's custom key-value format to JSON.
/// Input:  {key:<|"|>value<|"|>,key2:<|"|>value2<|"|>}
/// Output: {"key":"value","key2":"value2"}
fn convertGemma4ArgsToJson(allocator: std.mem.Allocator, input: []const u8) ?[]const u8 {
    const str_delim = "<|\"|>";
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);
    result.append(allocator, '{') catch return null;

    // Strip outer braces
    var body = input;
    if (body.len >= 2 and body[0] == '{') {
        body = body[1..];
        if (body[body.len - 1] == '}') body = body[0 .. body.len - 1];
    }

    var first = true;
    var pos: usize = 0;
    while (pos < body.len) {
        // Skip whitespace and commas
        while (pos < body.len and (body[pos] == ' ' or body[pos] == ',' or body[pos] == '\n' or body[pos] == '\t')) : ({
            pos += 1;
        }) {}
        if (pos >= body.len) break;

        // Find key (everything before ':')
        const colon = std.mem.indexOf(u8, body[pos..], ":") orelse break;
        const key_raw = std.mem.trim(u8, body[pos .. pos + colon], " \t\n\r");
        // Strip surrounding quotes if present (model sometimes quotes keys in custom format)
        const key = if (key_raw.len >= 2 and key_raw[0] == '"' and key_raw[key_raw.len - 1] == '"')
            key_raw[1 .. key_raw.len - 1]
        else
            key_raw;
        pos = pos + colon + 1;

        if (!first) result.append(allocator, ',') catch return null;
        first = false;

        // Output key
        result.append(allocator, '"') catch return null;
        result.appendSlice(allocator, key) catch return null;
        result.appendSlice(allocator, "\":") catch return null;

        // Check if value starts with <|"|> delimiter
        if (pos + str_delim.len <= body.len and std.mem.eql(u8, body[pos .. pos + str_delim.len], str_delim)) {
            pos += str_delim.len;
            // Find closing delimiter (may be missing if model output was truncated)
            const end = std.mem.indexOf(u8, body[pos..], str_delim) orelse (body.len - pos);
            const value = body[pos .. pos + end];
            pos = if (pos + end + str_delim.len <= body.len)
                pos + end + str_delim.len
            else
                body.len;

            // JSON-escape the value
            result.append(allocator, '"') catch return null;
            for (value) |c| {
                switch (c) {
                    '"' => result.appendSlice(allocator, "\\\"") catch return null,
                    '\\' => result.appendSlice(allocator, "\\\\") catch return null,
                    '\n' => result.appendSlice(allocator, "\\n") catch return null,
                    '\r' => result.appendSlice(allocator, "\\r") catch return null,
                    '\t' => result.appendSlice(allocator, "\\t") catch return null,
                    else => result.append(allocator, c) catch return null,
                }
            }
            result.append(allocator, '"') catch return null;
        } else if (pos < body.len and (body[pos] == '{' or body[pos] == '[')) {
            // Nested object or array — match braces to find the full value
            const open = body[pos];
            const close: u8 = if (open == '{') '}' else ']';
            var depth: usize = 0;
            var end: usize = pos;
            var in_str = false;
            while (end < body.len) : (end += 1) {
                if (body[end] == '"' and (end == 0 or body[end - 1] != '\\')) {
                    in_str = !in_str;
                } else if (!in_str) {
                    if (body[end] == open) depth += 1
                    else if (body[end] == close) {
                        depth -= 1;
                        if (depth == 0) { end += 1; break; }
                    }
                }
            }
            const value = body[pos..end];
            // Pass through as-is (already JSON-like structure)
            result.appendSlice(allocator, value) catch return null;
            pos = end;
        } else {
            // Bare value (number, boolean, or unquoted string) — terminates at , or }
            const val_end = std.mem.indexOfAny(u8, body[pos..], ",}") orelse body.len - pos;
            const value = std.mem.trim(u8, body[pos .. pos + val_end], " \t\n\r");
            if (isJsonLiteral(value)) {
                result.appendSlice(allocator, value) catch return null;
            } else {
                appendJsonString(allocator, &result, value) catch return null;
            }
            pos = pos + val_end;
        }
    }

    result.append(allocator, '}') catch return null;
    return result.toOwnedSlice(allocator) catch return null;
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
    // Objects/arrays are handled by brace-matching in the caller, not here
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

test "collapseDoubledThinkTags collapses 2x → 1x" {
    const out = try collapseDoubledThinkTags(testing.allocator, "<|Assistant|></think></think>Hi!");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("<|Assistant|></think>Hi!", out);
}

test "collapseDoubledThinkTags collapses triple → 1x" {
    const out = try collapseDoubledThinkTags(testing.allocator, "</think></think></think>X");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("</think>X", out);
}

test "collapseDoubledThinkTags leaves single </think> unchanged" {
    const out = try collapseDoubledThinkTags(testing.allocator, "<think>r</think>Hello");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("<think>r</think>Hello", out);
}

test "collapseDoubledThinkTags handles multiple separated doublings" {
    const out = try collapseDoubledThinkTags(testing.allocator,
        "A</think></think>B</think></think>C");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("A</think>B</think>C", out);
}

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

test "splitThinkBlock thinking=true no close tag, literal opener present" {
    // Model entered thinking but ran out of tokens before closing.
    const result = splitThinkBlock("<think>partial reasoning", true);
    try testing.expectEqualStrings("partial reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "splitThinkBlock thinking=true no close tag, no opener (template-injected)" {
    // Qwen 3.6 case: chat template injects `<think>\n` so model output starts
    // mid-block; if model finishes without `</think>`, treat as content so
    // the user actually sees the answer.
    const result = splitThinkBlock("It is currently 8:15 AM PDT.", true);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("It is currently 8:15 AM PDT.", result.content);
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

test "parseToolCalls Qwen3.6 MoE nested-name garbage (real capture)" {
    // Captured verbatim from Qwen3.6-35B-A3B-6bit generating a tool call in
    // streaming mode with enable_thinking=false and tools present.
    // parseToolCalls previously returned null here because the top-level "name"
    // is an object, not a string — tokens then leaked as plain-text content.
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name": {"name": {"name":  "write", "arguments": {"path": "/tmp/x/app.test.js", "content": "hello"}}}}
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
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/x/app.test.js", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("hello", parsed.value.object.get("content").?.string);
}

test "parseToolCalls Qwen MoE double-nested name" {
    // Two layers of nesting — defensive test for a slightly different shape.
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name":{"name":"shell","arguments":{"command":"ls"}}}
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
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
}

test "parseToolCalls flat Qwen MoE shape (no arguments wrapper)" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"shell\", \"command\": \"mkdir -p 2ddungeon\"}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mkdir -p 2ddungeon", parsed.value.object.get("command").?.string);
    try testing.expect(parsed.value.object.get("name") == null);
}

test "parseToolCalls flat shape with multiple top-level args" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"writeFile\", \"path\": \"/tmp/a\", \"content\": \"hi\"}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("writeFile", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/a", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("hi", parsed.value.object.get("content").?.string);
}

test "parseToolCalls flat shape ignores id/type metadata" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"id\": \"call_1\", \"type\": \"function\", \"name\": \"shell\", \"command\": \"ls\"}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls", parsed.value.object.get("command").?.string);
    try testing.expect(parsed.value.object.get("id") == null);
    try testing.expect(parsed.value.object.get("type") == null);
}

test "parseToolCalls repairs Qwen MoE missing-arguments-key shape" {
    const allocator = testing.allocator;
    // Real broken output observed from Qwen3.6-35B-A3B-6bit:
    // {"name": "shell",  {"command":"ls"}}  — `, {` instead of `, "arguments": {`
    const text = "<tool_call>\n{\"name\":  \"shell\",     {\"command\":\"ls -la\"}}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls -la", parsed.value.object.get("command").?.string);
}

test "parseToolCalls repairs Qwen MoE missing-opening-quote on arguments key" {
    const allocator = testing.allocator;
    // Real broken output observed from Qwen3.6-35B-A3B-6bit:
    // {"name": "shell", arguments": {"command":"mkdir -p src/app"}}
    // — missing the OPENING `"` on the `arguments` key.
    const text = "<tool_call>\n{\"name\": \"shell\", arguments\": {\"command\": \"mkdir -p src/app\"}}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mkdir -p src/app", parsed.value.object.get("command").?.string);
}

test "parseToolCalls returns null for name-only object (no real args)" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"shell\"}\n</tool_call>";
    const result = try parseToolCalls(allocator, text);
    try testing.expect(result == null);
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

// ── DSV4 training-bias tool-call shapes ──
// DSV4-Flash via ds4 emits tool calls in two non-standard variants the
// reference parser previously dropped on the floor:
//   1. Attribute form: `<tool_call name="X">{args}</tool_call>` — name as an
//      XML attribute, body is JUST the args object.
//   2. Mismatched closing tag: `<tool_call>{full json}</tool_request>` — open
//      tag is `tool_call`, close is `tool_request`. Hermes-style trained on a
//      mixed corpus.
// These tests are the regression guard. Without the fix they fail because
// the parser hard-matched on `<tool_call>…</tool_call>` exactly.

test "parseToolCalls DSV4 attribute form: <tool_call name=\"X\">{args}</tool_call>" {
    const allocator = testing.allocator;
    const text = "<tool_call name=\"webSearch\">{\"query\":\"top open source inference applications\"}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("webSearch", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("top open source inference applications", parsed.value.object.get("query").?.string);
}

test "parseToolCalls DSV4 mismatched close: <tool_call>{json}</tool_request>" {
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name": "shell", "arguments": {"command": "mkdir -p prisma src/lib"}}
        \\</tool_request>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mkdir -p prisma src/lib", parsed.value.object.get("command").?.string);
}

test "parseToolCalls DSV4 attribute + mismatched close combined" {
    // Worst-case combo observed from DSV4: attribute-style open, body is
    // just the args object (no top-level "name" key), and the closing tag
    // is `</tool_request>` instead of `</tool_call>`.
    const allocator = testing.allocator;
    const text = "<tool_call name=\"shell\">{\"command\":\"ls /tmp\"}</tool_request>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls /tmp", parsed.value.object.get("command").?.string);
}

test "parseToolCalls DSV4 attribute single-quoted: <tool_call name='X'>" {
    // Some captures used single quotes around the attribute value. Same
    // semantics — accept both quote styles.
    const allocator = testing.allocator;
    const text = "<tool_call name='browse'>{\"action\":\"navigate\",\"url\":\"https://example.com\"}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
}

test "parseToolCalls DSV4 plural-tag form: <tool_calls>{json}</tool_calls>" {
    // Real capture from DSV4-Flash in MLX Core ChatView:
    //   <tool_calls>
    //   {"name": "shell", "arguments": {"command": "df -h /"}}
    //   </tool_calls>
    // Plural open + plural close, single JSON object inside. Without the
    // fix the parser hard-matches `<tool_call` (the suffix `s>` makes the
    // `>`-scan walk past the tool name), the body never extracts, and the
    // raw XML leaks into the chat bubble.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\{"name": "shell", "arguments": {"command": "df -h /"}}
        \\</tool_calls>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("df -h /", parsed.value.object.get("command").?.string);
}

test "parseToolCalls DSV4 plural-tag with attribute: <tool_calls name=\"X\">{args}</tool_calls>" {
    const allocator = testing.allocator;
    const text = "<tool_calls name=\"webSearch\">{\"query\":\"hi\"}</tool_calls>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("webSearch", calls[0].name);
}

test "parseToolCalls DSV4 nested <tool name=\"X\"> inside <tool_calls> wrapper" {
    // Real capture from DSV4-Flash:
    //   <tool_calls>
    //   <tool name="webSearch">{"query": "top 10 ..."}</tool_calls>
    //   <tool name="webSearch">{"query": "best ..."}</tool_calls>
    //   </tool_calls>
    // Two inner calls, each opening with `<tool name="X">` and closing with
    // `</tool_calls>` (the model conflates open and close tag names). The
    // outer `<tool_calls>` wrapper has no content of its own — it's a
    // decorative parent.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\<tool name="webSearch">{"query": "top 10 open source AI inference engines 2024 2025"}</tool_calls>
        \\<tool name="webSearch">{"query": "best open source AI inference engines vllm tensorrt mlx comparison"}</tool_calls>
        \\</tool_calls>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    // Both inner calls must be recovered.
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("webSearch", calls[0].name);
    try testing.expectEqualStrings("webSearch", calls[1].name);
    const a0 = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer a0.deinit();
    try testing.expect(std.mem.indexOf(u8, a0.value.object.get("query").?.string, "top 10") != null);
    const a1 = try std.json.parseFromSlice(std.json.Value, allocator, calls[1].arguments, .{});
    defer a1.deinit();
    try testing.expect(std.mem.indexOf(u8, a1.value.object.get("query").?.string, "best") != null);
}

test "parseToolCalls bare <tool>{json}</tool>" {
    // Hypothetical minimal form — `<tool>` open with no attribute, JSON
    // body carries the name. Belt-and-suspenders for future model output.
    const allocator = testing.allocator;
    const text = "<tool>{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd\"}}</tool>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
}

test "parseToolCalls DSV4 <parameters> wrapper inside <tool name=\"X\">" {
    // Real capture from DSV4-Flash:
    //   <tool_calls>
    //   <tool name="browse"><parameters>{"action":"navigate","url":"…"}</parameters></tool>
    //   …
    //   </tool_calls>
    // The args object is wrapped in <parameters>…</parameters> before being
    // placed inside the <tool> body. Strip that wrapper before JSON-parsing.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\<tool name="browse"><parameters>{"action":"navigate","url":"https://example.com"}</parameters></tool>
        \\<tool name="browse"><parameters>{"action":"navigate","url":"https://example.org"}</parameters></tool>
        \\</tool_calls>
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
    try testing.expectEqualStrings("browse", calls[0].name);
    const a0 = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer a0.deinit();
    try testing.expectEqualStrings("navigate", a0.value.object.get("action").?.string);
    try testing.expectEqualStrings("https://example.com", a0.value.object.get("url").?.string);
    const a1 = try std.json.parseFromSlice(std.json.Value, allocator, calls[1].arguments, .{});
    defer a1.deinit();
    try testing.expectEqualStrings("https://example.org", a1.value.object.get("url").?.string);
}

test "parseToolCalls DSV4 trailing extra closing brace in args body" {
    // DSV4 occasionally emits an extra `}` after the args object — likely a
    // training-data artifact. Be tolerant: find the first balanced JSON
    // object and ignore trailing garbage.
    const allocator = testing.allocator;
    const text =
        "<tool name=\"browse\"><parameters>{\"action\":\"navigate\",\"url\":\"https://dev.to/agdex_ai/post\"}}</parameters></tool>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
    const a0 = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer a0.deinit();
    try testing.expectEqualStrings("navigate", a0.value.object.get("action").?.string);
}

test "splitThinkBlock with think block before tool call" {
    const text = "<think>I need to call the calculator</think>\n<tool_call>\n{\"name\": \"calc\", \"arguments\": {\"a\": 5}}\n</tool_call>";
    const result = splitThinkBlock(text, false);
    try testing.expectEqualStrings("I need to call the calculator", result.reasoning_content.?);
    // Content should be the tool call text
    try testing.expect(std.mem.startsWith(u8, result.content, "<tool_call>"));
}

test "splitThinkBlock with think block before regular content" {
    const text = "<think>Let me think about this</think>\n\nThe answer is 42.";
    const result = splitThinkBlock(text, false);
    try testing.expectEqualStrings("Let me think about this", result.reasoning_content.?);
    try testing.expectEqualStrings("The answer is 42.", result.content);
}

test "splitThinkBlock with empty think block" {
    const text = "<think>\n\n</think>\n\nJust content here.";
    const result = splitThinkBlock(text, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("Just content here.", result.content);
}

test "splitThinkBlock no think tags with tool call" {
    const text = "<tool_call>\n{\"name\": \"search\", \"arguments\": {}}\n</tool_call>";
    const result = splitThinkBlock(text, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings(text, result.content);
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

test "stripThinkBlock Gemma 4 channel tags" {
    try testing.expectEqualStrings("Hello", stripThinkBlock("<|channel>thought\nreasoning here<channel|>Hello"));
    try testing.expectEqualStrings("Hello", stripThinkBlock("<|channel>thought\nreasoning<channel|>\nHello"));
    try testing.expectEqualStrings("", stripThinkBlock("<|channel>thought\nstill thinking..."));
}

test "splitThinkBlock Gemma 4 channel tags" {
    const result = splitThinkBlock("<|channel>thought\nmy reasoning<channel|>answer here", false);
    try testing.expectEqualStrings("my reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("answer here", result.content);
}

test "splitThinkBlock Gemma 4 thinking in progress" {
    const result = splitThinkBlock("<|channel>thought\npartial reasoning", true);
    try testing.expectEqualStrings("partial reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "parseToolCalls Gemma 4 format" {
    const allocator = testing.allocator;
    const text = "<|tool_call>call:get_weather{\"location\": \"Tokyo\"}<tool_call|>";
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
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Tokyo", parsed.value.object.get("location").?.string);
}

test "parseToolCalls Gemma 4 with channel thinking" {
    const allocator = testing.allocator;
    const text = "<|channel>thought\nLet me check the weather<channel|>\n<|tool_call>call:get_weather{\"city\": \"Paris\"}<tool_call|>";
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
}

test "parseToolCalls Gemma 4 truncated (no closing tag)" {
    const allocator = testing.allocator;
    // Model hit EOS before generating <tool_call|>
    const text = "<|tool_call>call:browse{action:<|\"|>browse<|\"|>,url:<|\"|>https://finance.yahoo.com<|\"|>}";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
    // Should have parsed at least the action argument
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("browse", parsed.value.object.get("action").?.string);
}

test "parseToolCalls Gemma 4 truncated mid-value" {
    const allocator = testing.allocator;
    // Model stopped mid-URL, no closing <|"|> for the value
    const text = "<|tool_call>call:browse{action:<|\"|>navigate<|\"|>,url:<|\"|>https://finance.";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
    // Should have parsed the complete action and the truncated URL
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("navigate", parsed.value.object.get("action").?.string);
    // URL should be present (truncated but captured)
    try testing.expect(parsed.value.object.get("url") != null);
}

test "parseToolCalls Gemma 4 quoted keys with custom delimiters" {
    const allocator = testing.allocator;
    // Model mixes JSON-style quoted keys with <|"|> delimiters — JSON parse fails,
    // convertGemma4ArgsToJson must strip quotes from keys to produce valid JSON.
    const text =
        \\<|tool_call>call:shell{"command":<|"|>ls -la<|"|>}<tool_call|>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls -la", parsed.value.object.get("command").?.string);
}

test "parseToolCalls Gemma 4 quoted keys with bare values" {
    const allocator = testing.allocator;
    // Model uses quoted keys but bare (non-JSON, non-delimited) values
    const text = "<|tool_call>call:shell{\"command\":ls -la}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls -la", parsed.value.object.get("command").?.string);
}

test "convertGemma4ArgsToJson nested braces in value" {
    const allocator = testing.allocator;
    // Content value contains JSON-like structures (e.g., JavaScript code with objects)
    const input = "{path:<|\"|>server.js<|\"|>,content:<|\"|>const x = {a: 1, b: {c: 2}};<|\"|>}";
    const result = convertGemma4ArgsToJson(allocator, input).?;
    defer allocator.free(result);
    // Verify it produces valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("server.js", parsed.value.object.get("path").?.string);
    const content = parsed.value.object.get("content").?.string;
    try testing.expect(std.mem.indexOf(u8, content, "{a: 1") != null);
}

test "convertGemma4ArgsToJson bare array value" {
    const allocator = testing.allocator;
    // Bare array value should be preserved via brace-matching
    const input = "{stops:[\"Rome\",\"Venice\",\"Athens\"],price:1200}";
    const result = convertGemma4ArgsToJson(allocator, input).?;
    defer allocator.free(result);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 3), parsed.value.object.get("stops").?.array.items.len);
    try testing.expectEqual(@as(i64, 1200), parsed.value.object.get("price").?.integer);
}

test "convertGemma4ArgsToJson bare nested object value" {
    const allocator = testing.allocator;
    // Bare nested object should be preserved via brace-matching
    const input = "{name:test,config:{\"port\":3000,\"host\":\"localhost\"}}";
    const result = convertGemma4ArgsToJson(allocator, input).?;
    defer allocator.free(result);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    const config = parsed.value.object.get("config").?.object;
    try testing.expectEqual(@as(i64, 3000), config.get("port").?.integer);
}

test "isJsonLiteral" {
    try testing.expect(isJsonLiteral("true"));
    try testing.expect(isJsonLiteral("false"));
    try testing.expect(isJsonLiteral("null"));
    // Objects/arrays are now handled by brace-matching in convertGemma4ArgsToJson, not isJsonLiteral
    try testing.expect(!isJsonLiteral("{\"key\":1}"));
    try testing.expect(!isJsonLiteral("[1,2,3]"));
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

test "fallbackFormatChat ChatML with tool calls and responses" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"ls\"}" },
    };
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "List files" },
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "file1.txt\nfile2.txt" },
    };
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = try allocator.dupe(u8, "<|im_end|>"),
        .add_bos_token = false,
        .allocator = allocator,
    };
    defer if (config.eos_token) |t| allocator.free(t);

    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";
    const result = try fallbackFormatChat(allocator, &messages, &config, tools_json, null, false);
    defer allocator.free(result);

    // Should have <tool_call> block for assistant
    try testing.expect(std.mem.indexOf(u8, result, "<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"name\": \"shell\"") != null);
    // Should have <tool_response> block for tool result
    try testing.expect(std.mem.indexOf(u8, result, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "file1.txt") != null);
    // Should end with assistant prompt
    try testing.expect(std.mem.endsWith(u8, result, "<|im_start|>assistant\n"));
}

test "fallbackFormatChat ChatML tool response uses user role" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "tool", .content = "42" },
    };
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = try allocator.dupe(u8, "<|im_end|>"),
        .add_bos_token = false,
        .allocator = allocator,
    };
    defer if (config.eos_token) |t| allocator.free(t);

    const result = try fallbackFormatChat(allocator, &messages, &config, null, null, false);
    defer allocator.free(result);

    // Tool responses in ChatML use <|im_start|>user role
    try testing.expect(std.mem.indexOf(u8, result, "<|im_start|>user\n<tool_response>") != null);
    // Should NOT have <|im_start|>tool (invalid role)
    try testing.expect(std.mem.indexOf(u8, result, "<|im_start|>tool") == null);
}

test "fallbackFormatChat Gemma tool response uses user role" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "Run ls" },
        .{ .role = "tool", .content = "file1.txt" },
    };
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = try allocator.dupe(u8, "<eos>"),
        .add_bos_token = false,
        .allocator = allocator,
    };
    defer if (config.eos_token) |t| allocator.free(t);

    const result = try fallbackFormatChat(allocator, &messages, &config, null, null, false);
    defer allocator.free(result);

    // Gemma tool results should be in user turn
    try testing.expect(std.mem.indexOf(u8, result, "<start_of_turn>user\nTool result: file1.txt") != null);
    // Should NOT have <start_of_turn>tool (invalid role)
    try testing.expect(std.mem.indexOf(u8, result, "<start_of_turn>tool") == null);
}

test "fallbackFormatChat Llama tool response uses ipython role" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "tool", .content = "hello" },
    };
    var config = ChatConfig{
        .chat_template = "start_header_id",
        .bos_token = null,
        .eos_token = try allocator.dupe(u8, "<|eot_id|>"),
        .add_bos_token = false,
        .allocator = allocator,
    };
    defer if (config.eos_token) |t| allocator.free(t);

    const result = try fallbackFormatChat(allocator, &messages, &config, null, null, false);
    defer allocator.free(result);

    // Llama tool results use ipython role
    try testing.expect(std.mem.indexOf(u8, result, "<|start_header_id|>ipython<|end_header_id|>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "hello") != null);
}

test "fallbackFormatChat ChatML assistant with empty content and tool_calls" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"date\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
    };
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = try allocator.dupe(u8, "<|im_end|>"),
        .add_bos_token = false,
        .allocator = allocator,
    };
    defer if (config.eos_token) |t| allocator.free(t);

    const result = try fallbackFormatChat(allocator, &messages, &config, null, null, false);
    defer allocator.free(result);

    // Empty content should not produce a blank line before <tool_call>
    try testing.expect(std.mem.indexOf(u8, result, "<|im_start|>assistant\n<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"name\": \"shell\"") != null);
}

test "fallbackFormatChat multi-round tool calling" {
    const allocator = testing.allocator;
    const tc1 = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"ls\"}" },
    };
    const tc2 = [_]ToolCall{
        .{ .id = "call_2", .name = "readFile", .arguments = "{\"path\":\"main.py\"}" },
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "Read main.py" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc1 },
        .{ .role = "tool", .content = "main.py" },
        .{ .role = "assistant", .content = "Found it.", .tool_calls = &tc2 },
        .{ .role = "tool", .content = "print('hello')" },
    };
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = try allocator.dupe(u8, "<|im_end|>"),
        .add_bos_token = false,
        .allocator = allocator,
    };
    defer if (config.eos_token) |t| allocator.free(t);

    const result = try fallbackFormatChat(allocator, &messages, &config, null, null, false);
    defer allocator.free(result);

    // Should have two <tool_call> blocks
    var count: usize = 0;
    var pos: usize = 0;
    while (std.mem.indexOf(u8, result[pos..], "<tool_call>")) |offset| {
        count += 1;
        pos += offset + 11;
    }
    try testing.expectEqual(@as(usize, 2), count);

    // Should have two <tool_response> blocks
    count = 0;
    pos = 0;
    while (std.mem.indexOf(u8, result[pos..], "<tool_response>")) |offset| {
        count += 1;
        pos += offset + 15;
    }
    try testing.expectEqual(@as(usize, 2), count);

    // Second assistant should have content before tool_call
    try testing.expect(std.mem.indexOf(u8, result, "Found it.\n<tool_call>") != null);
}

// ── Tool fallback (DSV4 + any template lacking tools support) ──

test "templateReferencesToolRole detects role branches" {
    try testing.expect(templateReferencesToolRole("{% if message['role'] == 'tool' %}"));
    try testing.expect(templateReferencesToolRole("{% if message.role == \"tool\" %}"));
    try testing.expect(!templateReferencesToolRole("{% if message['role'] == 'user' %}{{ message['tool_calls'] }}{% endif %}"));
    try testing.expect(!templateReferencesToolRole(""));
}

test "messagesHaveToolContent" {
    const tc = [_]ToolCall{.{ .id = "x", .name = "n", .arguments = "{}" }};
    const m_tool = [_]Message{.{ .role = "tool", .content = "x" }};
    const m_tc = [_]Message{.{ .role = "assistant", .content = "", .tool_calls = &tc }};
    const m_plain = [_]Message{.{ .role = "user", .content = "hi" }};
    try testing.expect(messagesHaveToolContent(&m_tool));
    try testing.expect(messagesHaveToolContent(&m_tc));
    try testing.expect(!messagesHaveToolContent(&m_plain));
}

test "synthesizeToolFallbackMessages: inject tool prompt with no existing system" {
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const messages = [_]Message{
        .{ .role = "user", .content = "List files" },
    };
    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";

    const out = try synthesizeToolFallbackMessages(a, &messages, tools_json, null, true, true, true);
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqualStrings("system", out[0].role);
    // System content carries the tool prompt + the tools_json blob.
    try testing.expect(std.mem.indexOf(u8, out[0].content, "tool_call") != null);
    try testing.expect(std.mem.indexOf(u8, out[0].content, "shell") != null);
    try testing.expectEqualStrings("user", out[1].role);
    try testing.expectEqualStrings("List files", out[1].content);
}

test "synthesizeToolFallbackMessages: merge tool prompt into existing system" {
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "List files" },
    };
    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";

    const out = try synthesizeToolFallbackMessages(a, &messages, tools_json, null, true, true, true);
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqualStrings("system", out[0].role);
    // Existing system content preserved, tool prompt appended.
    try testing.expect(std.mem.startsWith(u8, out[0].content, "You are helpful."));
    try testing.expect(std.mem.indexOf(u8, out[0].content, "Available functions") != null);
    try testing.expect(std.mem.indexOf(u8, out[0].content, "shell") != null);
}

test "synthesizeToolFallbackMessages: rewrite role:tool to role:user" {
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const messages = [_]Message{
        .{ .role = "user", .content = "do it" },
        .{ .role = "tool", .content = "result: 42" },
    };
    const out = try synthesizeToolFallbackMessages(a, &messages, null, null, false, true, false);
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqualStrings("user", out[1].role);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "result: 42") != null);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "</tool_response>") != null);
}

test "synthesizeToolFallbackMessages: rewrite assistant tool_calls into <tool_call> blocks" {
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const tc = [_]ToolCall{
        .{ .id = "c1", .name = "shell", .arguments = "{\"command\":\"ls\"}" },
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "ls" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
    };
    const out = try synthesizeToolFallbackMessages(a, &messages, null, null, false, false, true);
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqualStrings("assistant", out[1].role);
    try testing.expect(out[1].tool_calls == null); // moved into content
    try testing.expect(std.mem.indexOf(u8, out[1].content, "<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "\"name\": \"shell\"") != null);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "</tool_call>") != null);
}

test "renderChatTemplate: DSV4-style template gets tool fallback applied" {
    const allocator = testing.allocator;
    // Minimal template that mirrors DSV4 — has system/user/assistant, no tools, no role:tool.
    const tpl =
        \\<bos>
        \\{%- for message in messages -%}
        \\{%- if message['role'] == 'system' -%}{{ message['content'] }}{%- elif message['role'] == 'user' -%}<U>{{ message['content'] }}</U><A>{%- elif message['role'] == 'assistant' -%}{{ message['content'] }}<EOS>{%- endif -%}
        \\{%- endfor -%}
    ;
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\",\"description\":\"run a shell command\"}}]";
    const messages = [_]Message{
        .{ .role = "user", .content = "list my files" },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false);
    defer allocator.free(rendered);
    // Tool prompt should have been injected as a system message.
    try testing.expect(std.mem.indexOf(u8, rendered, "Available functions") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "shell") != null);
    // User message still rendered after the synthesized system content.
    try testing.expect(std.mem.indexOf(u8, rendered, "list my files") != null);
}

// ── DSV4 agentic-flow compositions (round-trip + multi-turn + edge cases) ──

test "DSV4 agentic round-trip: model-generated <tool_call> parses + replays" {
    // Simulates the full agent loop step:
    //   1. Model emits a `<tool_call>` block in its generated content.
    //   2. parseToolCalls extracts the structured ToolCall(s).
    //   3. The agent builds the next prompt with role:assistant(tool_calls=...) +
    //      role:tool(content=result).
    //   4. synthesizeToolFallbackMessages rewrites both for the DSV4 template
    //      (assistant tool_calls → <tool_call> block, tool → user
    //      <tool_response>).
    //   5. The same call should round-trip: the rewritten assistant content
    //      should contain a parseable <tool_call> matching the original.
    const allocator = testing.allocator;

    // 1. Pretend the model wrote this:
    const generated =
        \\<tool_call>
        \\{"name": "shell", "arguments": {"command": "date"}}
        \\</tool_call>
    ;
    const parsed_calls = (try parseToolCalls(allocator, generated)).?;
    defer {
        for (parsed_calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(parsed_calls);
    }
    try testing.expectEqual(@as(usize, 1), parsed_calls.len);
    try testing.expectEqualStrings("shell", parsed_calls[0].name);

    // 2. Agent builds the next-turn history with the parsed call + the result.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const tc = [_]ToolCall{
        .{ .id = "tc_0", .name = parsed_calls[0].name, .arguments = parsed_calls[0].arguments },
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "what is the date?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        .{ .role = "tool", .content = "Mon May 13 2026", .tool_call_id = "tc_0" },
    };

    // 3. Run through the DSV4 fallback. inject_tool_prompt=true mirrors how
    // renderChatTemplate calls it when the template lacks tool support.
    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";
    const out = try synthesizeToolFallbackMessages(a, &messages, tools_json, null, true, true, true);

    // Should produce: system(tool-prompt), user, assistant(<tool_call>), user(<tool_response>).
    try testing.expectEqual(@as(usize, 4), out.len);
    try testing.expectEqualStrings("system", out[0].role);
    try testing.expectEqualStrings("user", out[1].role);
    try testing.expectEqualStrings("assistant", out[2].role);
    try testing.expectEqualStrings("user", out[3].role);

    // 4. The assistant's rewritten content must contain a <tool_call> block
    //    that re-parses back to the same name + arguments. This is the
    //    closed-loop test.
    try testing.expect(out[2].tool_calls == null);
    const reparsed = (try parseToolCalls(allocator, out[2].content)).?;
    defer {
        for (reparsed) |t| {
            allocator.free(t.name);
            allocator.free(t.arguments);
        }
        allocator.free(reparsed);
    }
    try testing.expectEqual(@as(usize, 1), reparsed.len);
    try testing.expectEqualStrings("shell", reparsed[0].name);
    try testing.expect(std.mem.indexOf(u8, reparsed[0].arguments, "date") != null);

    // 5. The tool-result rewrite must wrap the result in <tool_response>...
    try testing.expect(std.mem.indexOf(u8, out[3].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, out[3].content, "Mon May 13 2026") != null);
    try testing.expect(std.mem.indexOf(u8, out[3].content, "</tool_response>") != null);
}

test "DSV4 multi-turn agent history: every role correctly rewritten" {
    // Realistic 6-message agent transcript:
    //   1. user (initial question)
    //   2. assistant tool_call #1 (date)
    //   3. tool result #1
    //   4. assistant intermediate reasoning + tool_call #2 (ls)
    //   5. tool result #2
    //   6. user follow-up
    // The synthesizer must:
    //   - inject the tool-prompt system message at the top
    //   - rewrite each role:tool into role:user with <tool_response>
    //   - rewrite each role:assistant with tool_calls into <tool_call> blocks
    //   - preserve user messages untouched
    //   - preserve assistant intermediate text (msg #4 has BOTH content and tool_calls)
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const tc1 = [_]ToolCall{.{ .id = "t1", .name = "shell", .arguments = "{\"command\":\"date\"}" }};
    const tc2 = [_]ToolCall{.{ .id = "t2", .name = "shell", .arguments = "{\"command\":\"ls\"}" }};
    const messages = [_]Message{
        .{ .role = "user", .content = "what files exist as of today?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc1 },
        .{ .role = "tool", .content = "Mon May 13 2026", .tool_call_id = "t1" },
        .{ .role = "assistant", .content = "Got the date. Now listing files.", .tool_calls = &tc2 },
        .{ .role = "tool", .content = "README.md\nsrc/\ntests/", .tool_call_id = "t2" },
        .{ .role = "user", .content = "summarize" },
    };

    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";
    const out = try synthesizeToolFallbackMessages(a, &messages, tools_json, null, true, true, true);

    // Expected layout: system + 6 rewritten messages.
    try testing.expectEqual(@as(usize, 7), out.len);
    try testing.expectEqualStrings("system", out[0].role);
    try testing.expect(std.mem.indexOf(u8, out[0].content, "Available functions") != null);

    // Original index 0 (user) → out[1] user, unchanged.
    try testing.expectEqualStrings("user", out[1].role);
    try testing.expectEqualStrings("what files exist as of today?", out[1].content);

    // Original index 1 (assistant tool_call) → out[2] assistant with <tool_call> in content, no tool_calls.
    try testing.expectEqualStrings("assistant", out[2].role);
    try testing.expect(out[2].tool_calls == null);
    try testing.expect(std.mem.indexOf(u8, out[2].content, "<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, out[2].content, "\"command\":\"date\"") != null);

    // Original index 2 (tool) → out[3] user with <tool_response>.
    try testing.expectEqualStrings("user", out[3].role);
    try testing.expect(std.mem.indexOf(u8, out[3].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, out[3].content, "Mon May 13 2026") != null);

    // Original index 3 (assistant content + tool_call) → out[4] preserves intermediate text.
    try testing.expectEqualStrings("assistant", out[4].role);
    try testing.expect(out[4].tool_calls == null);
    try testing.expect(std.mem.indexOf(u8, out[4].content, "Got the date. Now listing files.") != null);
    try testing.expect(std.mem.indexOf(u8, out[4].content, "<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, out[4].content, "\"command\":\"ls\"") != null);

    // Original index 4 (tool) → out[5] user with <tool_response>.
    try testing.expectEqualStrings("user", out[5].role);
    try testing.expect(std.mem.indexOf(u8, out[5].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, out[5].content, "README.md") != null);

    // Original index 5 (user follow-up) → out[6] user, unchanged.
    try testing.expectEqualStrings("user", out[6].role);
    try testing.expectEqualStrings("summarize", out[6].content);
}

test "DSV4 fallback: null tools_json with inject_tool_prompt does not synthesize" {
    // When a request has NO tools (chat without tools param), the synthesizer
    // must NOT inject a system message even with inject_tool_prompt=true.
    // Only role:tool / assistant tool_calls rewrites still fire (in case the
    // template still lacks support and history was injected by a previous turn).
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const messages = [_]Message{
        .{ .role = "user", .content = "hello" },
    };
    // tools_json = null → no synthesis.
    const out = try synthesizeToolFallbackMessages(a, &messages, null, null, true, true, true);
    try testing.expectEqual(@as(usize, 1), out.len);
    try testing.expectEqualStrings("user", out[0].role);
    try testing.expectEqualStrings("hello", out[0].content);
}

// ── prepareDs4Prompt: end-to-end unit coverage for the tool-plumbing fix ──
// These tests guard against the regression where ds4 silently dropped
// `tools_json` / `tool_choice_instruction` / `role:"tool"` content. If the
// `encodeChatViaDs4` call sites stop threading `tools_json` through, the
// catalog won't reach `prep.system` and these tests fail.

test "prepareDs4Prompt: tools_json injects the catalog into the system message" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "list /tmp" },
    };
    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\",\"description\":\"Run a command\"}}]";

    var prep = try prepareDs4Prompt(allocator, &messages, tools_json, null);
    defer prep.deinit();

    try testing.expect(prep.system != null);
    // The synthesized system message MUST contain the tool name so the model
    // knows the tool exists. If `encodeChatViaDs4` stops threading
    // `tools_json` to `synthesizeToolFallbackMessages`, this fails.
    try testing.expect(std.mem.indexOf(u8, prep.system.?, "shell") != null);
    // User turn passes through unchanged.
    try testing.expectEqual(@as(usize, 1), prep.turns.len);
    try testing.expectEqualStrings("user", prep.turns[0].role);
    try testing.expectEqualStrings("list /tmp", prep.turns[0].content);
}

test "prepareDs4Prompt: tools_json merges into existing system message" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "hi" },
    };
    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"webSearch\"}}]";

    var prep = try prepareDs4Prompt(allocator, &messages, tools_json, null);
    defer prep.deinit();

    try testing.expect(prep.system != null);
    // Both the original system content AND the tool catalog must be present
    // in the merged system message — losing either side would silently
    // strip the agent prompt or the tool list.
    try testing.expect(std.mem.indexOf(u8, prep.system.?, "You are helpful.") != null);
    try testing.expect(std.mem.indexOf(u8, prep.system.?, "webSearch") != null);
}

test "prepareDs4Prompt: role:tool gets rewritten to user with <tool_response>" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "run ls" },
        .{ .role = "tool", .content = "file1.txt\nfile2.txt" },
    };

    var prep = try prepareDs4Prompt(allocator, &messages, null, null);
    defer prep.deinit();

    try testing.expectEqual(@as(usize, 2), prep.turns.len);
    try testing.expectEqualStrings("user", prep.turns[0].role);
    // Tool result MUST be wrapped as a user turn with <tool_response>; ds4's
    // chat template doesn't model role:"tool" natively so leaving it as
    // role:"tool" makes the model unaware of any prior tool output.
    try testing.expectEqualStrings("user", prep.turns[1].role);
    try testing.expect(std.mem.indexOf(u8, prep.turns[1].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, prep.turns[1].content, "file1.txt") != null);
}

test "prepareDs4Prompt: assistant tool_calls get inlined as <tool_call> text" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "c1", .name = "shell", .arguments = "{\"command\":\"ls /tmp\"}" },
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "list tmp" },
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "ok", .tool_call_id = "c1" },
    };

    var prep = try prepareDs4Prompt(allocator, &messages, null, null);
    defer prep.deinit();

    // Assistant turn must carry the inlined <tool_call> so multi-turn
    // history stays coherent on ds4. Without the rewrite the model sees
    // an empty assistant turn followed by a tool result with no context.
    try testing.expectEqual(@as(usize, 3), prep.turns.len);
    try testing.expectEqualStrings("assistant", prep.turns[1].role);
    try testing.expect(std.mem.indexOf(u8, prep.turns[1].content, "<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, prep.turns[1].content, "\"name\": \"shell\"") != null);
    try testing.expect(std.mem.indexOf(u8, prep.turns[1].content, "ls /tmp") != null);
}

test "prepareDs4Prompt: no tools, no tool content → passthrough" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "system", .content = "be terse" },
        .{ .role = "user", .content = "hi" },
        .{ .role = "assistant", .content = "hello" },
    };

    var prep = try prepareDs4Prompt(allocator, &messages, null, null);
    defer prep.deinit();

    // Nothing to synthesize — system should be the original verbatim, turns
    // should be the non-system messages in order.
    try testing.expect(prep.system != null);
    try testing.expectEqualStrings("be terse", prep.system.?);
    try testing.expectEqual(@as(usize, 2), prep.turns.len);
    try testing.expectEqualStrings("user", prep.turns[0].role);
    try testing.expectEqualStrings("hi", prep.turns[0].content);
    try testing.expectEqualStrings("assistant", prep.turns[1].role);
    try testing.expectEqualStrings("hello", prep.turns[1].content);
}

test "streamShouldBufferForTools: empty + plain prose" {
    try testing.expect(!streamShouldBufferForTools(""));
    try testing.expect(!streamShouldBufferForTools("hello world"));
    try testing.expect(!streamShouldBufferForTools("Let me look at this."));
}

test "streamShouldBufferForTools: trailing partial prefixes of <tool" {
    // The exact regression from the screenshot: model emits `<tool` as a
    // single BPE token, server must buffer instead of streaming it as
    // content (so the next token can either complete the open or be flushed
    // together as harmless prose).
    try testing.expect(streamShouldBufferForTools("<"));
    try testing.expect(streamShouldBufferForTools("<t"));
    try testing.expect(streamShouldBufferForTools("<to"));
    try testing.expect(streamShouldBufferForTools("<too"));
    try testing.expect(streamShouldBufferForTools("<tool"));
    try testing.expect(streamShouldBufferForTools("Let me look at the CLI help:\n<tool"));
}

test "streamShouldBufferForTools: trailing partial prefixes of <|tool_call" {
    try testing.expect(streamShouldBufferForTools("<|"));
    try testing.expect(streamShouldBufferForTools("<|t"));
    try testing.expect(streamShouldBufferForTools("<|too"));
    try testing.expect(streamShouldBufferForTools("<|tool"));
    try testing.expect(streamShouldBufferForTools("<|tool_"));
    try testing.expect(streamShouldBufferForTools("<|tool_c"));
    try testing.expect(streamShouldBufferForTools("<|tool_cal"));
}

test "streamShouldBufferForTools: bare <tool> open (DSV4)" {
    try testing.expect(streamShouldBufferForTools("<tool>"));
    try testing.expect(streamShouldBufferForTools("<tool>{\"name\":\"shell\""));
    try testing.expect(streamShouldBufferForTools("<tool name=\"shell\">"));
    try testing.expect(streamShouldBufferForTools("hello <tool>{}"));
}

test "streamShouldBufferForTools: canonical <tool_call*> opens" {
    try testing.expect(streamShouldBufferForTools("<tool_call>"));
    try testing.expect(streamShouldBufferForTools("<tool_call>{\"name\":"));
    try testing.expect(streamShouldBufferForTools("<tool_calls>"));
    try testing.expect(streamShouldBufferForTools("<tool_request>"));
    try testing.expect(streamShouldBufferForTools("prose first then <tool_call>"));
}

test "streamShouldBufferForTools: Gemma 4 <|tool_call" {
    try testing.expect(streamShouldBufferForTools("<|tool_call>"));
    try testing.expect(streamShouldBufferForTools("hello <|tool_call name=\"x\">"));
}

test "streamShouldBufferForTools: false positives — <toolkit> etc." {
    // `<toolkit>` and `<toolbar>` have `<tool` as a prefix but the char
    // after is `k` / `b` — not a tag terminator. Must NOT buffer.
    try testing.expect(!streamShouldBufferForTools("here is <toolkit>"));
    try testing.expect(!streamShouldBufferForTools("here is <toolbar>"));
    try testing.expect(!streamShouldBufferForTools("HTML: <toolset>"));
    // But a later real `<tool>` should still trip detection:
    try testing.expect(streamShouldBufferForTools("HTML: <toolset> then <tool>{"));
}

test "streamShouldBufferForTools: raw JSON shape" {
    try testing.expect(streamShouldBufferForTools("{\"name\":\"shell\",\"arguments\":{}}"));
    try testing.expect(streamShouldBufferForTools("{\"name\":\"shell\""));
    // Just `{` or JSON without `"name"` shouldn't false-trigger.
    try testing.expect(!streamShouldBufferForTools("{"));
    try testing.expect(!streamShouldBufferForTools("{\"foo\":1}"));
}

test "parseToolCalls: self-closing <tool name=... arguments=... />" {
    const allocator = testing.allocator;
    // Clean form. We use the literal byte sequence the model actually emits —
    // unescaped `"` inside the attribute value. That's XML-invalid but
    // perfectly JSON-parseable once we snap to the inner `{…}`.
    const text =
        \\<tool name="shell" arguments="{"command": "ls -la"}"/>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"command\"") != null);
}

test "parseToolCalls: DSV4 broken-quote self-closing fragment (regression repro)" {
    // Verbatim capture from the failing turn in test_agent_stop_repro.py.
    // The model opened arguments with `"`, closed with `'`, embedded
    // unescaped `"` inside the JSON, and finished with `'/>`. The parser
    // must still recover the JSON object and produce a single tool call.
    const allocator = testing.allocator;
    const text =
        "\n\n<tool_calls>\n" ++
        "<tool name=\"shell\" arguments=\"{\"command\": \"ls -la && cat package.json 2>/dev/null || echo 'no package.json yet'\"}'/>\n" ++
        "</tool_calls>";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    // Args should be the balanced JSON object — starts with `{` and ends `}`.
    try testing.expect(calls[0].arguments.len > 0);
    try testing.expectEqual(@as(u8, '{'), calls[0].arguments[0]);
    try testing.expectEqual(@as(u8, '}'), calls[0].arguments[calls[0].arguments.len - 1]);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"command\"") != null);
}

test "parseToolCalls: multiple self-closing tool tags in one response" {
    const allocator = testing.allocator;
    const text =
        \\<tool name="cwd" arguments="{"path": "src"}"/>
        \\<tool name="listFiles" arguments="{"recursive": "true"}"/>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("cwd", calls[0].name);
    try testing.expectEqualStrings("listFiles", calls[1].name);
}

test "parseToolCalls: canonical <tool_call>JSON</tool_call> still parses" {
    // Regression guard — adding the self-closing branch must not break the
    // canonical Hermes form.
    const allocator = testing.allocator;
    const text =
        \\<tool_call>{"name": "shell", "arguments": {"command": "ls"}}</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
}

test "parseToolCalls: empty body + attribute-args + explicit </tool_call> close" {
    // Verbatim capture from the user's failed chat-history.json. The model
    // emitted args via XML attribute and used `></tool_call>` (not `/>`)
    // as the terminator, with empty body. Without explicit handling this
    // shape parses as "empty body" → no JSON to extract → no tool call.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\<tool_call name="cwd" arguments="{"path": "/Users/david/.mlx-serve/workspace"}"></tool_call>
        \\</tool_calls>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("cwd", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"path\"") != null);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, ".mlx-serve/workspace") != null);
}

test "parseToolCalls: <tool_call>{JSON} truncated before </tool_call>" {
    // The model emitted `<tool_call>` + well-formed args JSON, then hit
    // EOS before completing the close marker. Output looks like:
    //   "\n\n<tool_call>\n{\"name\": \"writeFile\", \"arguments\":{...}}\n</tool_cal"
    // Previously this fell through to the orphan branch and got silently
    // dropped — that's the second failure mode the agent-stops repro
    // surfaced. We snap to a balanced JSON body and parse it.
    const allocator = testing.allocator;
    const text = "\n\n<tool_call>\n" ++
        \\{"name": "writeFile", "arguments": {"path": "vite.config.js", "content": "ok"}}
        ++ "\n</tool_cal";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("writeFile", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"path\"") != null);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"content\"") != null);
}

test "parseToolCalls: <toolkit> is not a tool tag" {
    // The self-closing scan must not false-fire on HTML-ish prose. `<toolkit>`
    // has `<tool` as a prefix but the terminator (`k`) isn't valid, so the
    // outer scan rejects it before parseSelfClosingToolTag even runs.
    const allocator = testing.allocator;
    const text = "Here's a doc about <toolkit> and <toolbar>. No actual tool call.";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}
