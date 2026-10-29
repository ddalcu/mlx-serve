//! Jinja chat template rendering, the template probes, and the fallback message shaping.

const std = @import("std");
const env = @import("../../env.zig");
const jinja_c = @import("jinja_c");
const log = @import("../../sdk/util/log.zig");
const testing = std.testing;

const ToolCall = @import("types.zig").ToolCall;
const ImageData = @import("types.zig").ImageData;
const VideoData = @import("types.zig").VideoData;
const foldSystemMessages = @import("types.zig").foldSystemMessages;
const Message = @import("types.zig").Message;
const ChatConfig = @import("types.zig").ChatConfig;
const loadChatConfig = @import("template_load.zig").loadChatConfig;
const EmptyContent = @import("serialize.zig").EmptyContent;
const serializeMessagesJsonFor = @import("serialize.zig").serializeMessagesJsonFor;
const fillOptionalToolDefKeys = @import("serialize.zig").fillOptionalToolDefKeys;
const serializeExtraContext = @import("effort.zig").serializeExtraContext;
const fallbackFormatChat = @import("fallback.zig").fallbackFormatChat;
const promptTailOpensThink = @import("reasoning/think_tags.zig").promptTailOpensThink;
const noThinkTailSuffix = @import("reasoning/think_tags.zig").noThinkTailSuffix;
const contentChannelTail = @import("reasoning/think_tags.zig").contentChannelTail;
const continuationPrefill = @import("continuation.zig").continuationPrefill;
const parseToolCalls = @import("tools/parse.zig").parseToolCalls;
const appendToolSystemPrompt = @import("tools/tool_choice.zig").appendToolSystemPrompt;

/// Render the Jinja chat template with the given messages.
pub fn renderChatTemplate(
    allocator: std.mem.Allocator,
    messages: []const Message,
    chat_config: *const ChatConfig,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
    effort: ?[]const u8,
    /// Extend the trailing assistant message instead of answering after it —
    /// the prompt ends mid-turn and generation resumes from its last word.
    continue_final: bool,
) ![]const u8 {
    // Split the partial reply off FIRST, so every transform below (tool
    // fallback synthesis, muse's reasoning drop) sees exactly the message list
    // an ordinary turn would have. The template must open the assistant turn
    // once, which it only does when the trailing assistant is not in `messages`.
    const prefill = continuationPrefill(messages, continue_final);
    const msgs = if (prefill != null) messages[0 .. messages.len - 1] else messages;

    if (chat_config.chat_template.len == 0) {
        const base = try fallbackFormatChat(allocator, msgs, chat_config, tools_json, tool_choice_instruction, enable_thinking);
        if (prefill) |partial| {
            defer allocator.free(base);
            return std.mem.concat(allocator, u8, &.{ base, partial });
        }
        return base;
    }

    // Some chat templates (e.g. DeepSeek V4) don't reference `tools` or `role == 'tool'`
    // at all — Jinja silently drops the tool definitions, and tool-result messages render
    // as empty. Detect that gap and synthesize an equivalent system-prompt + user-message
    // form so the model still sees the tool context.
    const tpl = chat_config.chat_template;
    const tpl_has_tools = std.mem.indexOf(u8, tpl, "tools") != null;
    const has_tool_content = messagesHaveToolContent(msgs);
    const tpl_has_tool_role = templateReferencesToolRole(tpl) or
        (has_tool_content and (chat_config.renders_tool_turn orelse try templateRendersToolTurn(allocator, tpl)));
    const needs_inject_tools = tools_json != null and !tpl_has_tools;
    const needs_rewrite_tool_role = !tpl_has_tool_role and has_tool_content;

    var fallback_arena: ?std.heap.ArenaAllocator = null;
    defer if (fallback_arena) |*a| a.deinit();
    var effective_messages = msgs;
    var effective_tools_json = tools_json;
    // Optional keys a template may dump blindly (see fillOptionalToolDefKeys).
    // Arena-owned, so the rewritten bytes outlive the render with no free path
    // of their own.
    if (tools_json) |tj| {
        if (fallback_arena == null) fallback_arena = std.heap.ArenaAllocator.init(allocator);
        if (try fillOptionalToolDefKeys(fallback_arena.?.allocator(), tj)) |filled| {
            effective_tools_json = filled;
        }
    }
    if (needs_inject_tools or needs_rewrite_tool_role or tool_choice_instruction != null) {
        // NEVER re-init: the tool-def fill above may already own this arena,
        // and clobbering it orphans everything allocated there (a leak here, a
        // use-after-free the moment `effective_tools_json` points into it).
        if (fallback_arena == null) fallback_arena = std.heap.ArenaAllocator.init(allocator);
        const arena_alloc = fallback_arena.?.allocator();
        effective_messages = try synthesizeToolFallbackMessages(
            arena_alloc,
            msgs,
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

    // Muse (harmony convention): reasoning belongs to the CURRENT turn only.
    // The app round-trips `reasoning_content` on assistant history, and this
    // template renders it as a to=self segment — so one bad chain (a loop-cut
    // tail) would replay in the prompt on every later turn. Drop it on
    // assistant messages before the last user message; tool rounds after it
    // keep theirs. Families whose templates NEED history reasoning (laguna,
    // inkling) are untouched — same sniff `noThinkTailSuffix` uses.
    if (std.mem.indexOf(u8, tpl, "reasoning_strength") != null) {
        if (fallback_arena == null) fallback_arena = std.heap.ArenaAllocator.init(allocator);
        if (try dropPriorTurnReasoning(fallback_arena.?.allocator(), effective_messages)) |stripped| {
            effective_messages = stripped;
        }
    }

    // Serialize messages to JSON — Gemma 4 templates handle role:"tool" natively
    // (producing <|turn>tool in the rendered output). No transformation needed.
    // Harmony (gpt_oss) indexes into `message.content` after only checking that
    // the KEY exists, so a null there breaks every tool round-trip. Sniffed off
    // the template's own channel marker, like the reasoning drop above.
    // LFM2-VL iterates any non-string content, so a null there raises on every
    // tool-call turn.
    const empty_content: EmptyContent = if (std.mem.indexOf(u8, tpl, "<|channel|>") != null or
        std.mem.indexOf(u8, tpl, "if content is not string") != null)
        .empty_string
    else
        .null_literal;
    const extra_json = try serializeExtraContext(allocator, chat_config, enable_thinking, effort);
    defer allocator.free(extra_json);
    if (hasExtraSystemMessages(effective_messages)) {
        if (fallback_arena == null) fallback_arena = std.heap.ArenaAllocator.init(allocator);
        const a = fallback_arena.?.allocator();
        var prepared = std.ArrayList(Message).empty;
        try prepared.appendSlice(a, effective_messages);
        const preserve = try templatePreservesMidSystem(a, effective_messages, chat_config, effective_tools_json, extra_json, empty_content);
        _ = try foldSystemMessages(a, &prepared, preserve);
        effective_messages = prepared.items;
    }

    if (try renderJinja(allocator, effective_messages, chat_config, effective_tools_json, extra_json, empty_content)) |rendered| {
        defer allocator.free(rendered);
        const collapsed = try collapseDoubledThinkTags(allocator, rendered);
        // A continuation commits the content channel and then hands the model
        // its own unfinished sentence. It runs INSTEAD of the thinking-off
        // tail, never beside it — both append the same channel commit, and
        // twice is a malformed turn header.
        if (prefill) |partial| {
            defer allocator.free(collapsed);
            return std.mem.concat(allocator, u8, &.{
                collapsed, contentChannelTail(tpl, collapsed, true), partial,
            });
        }
        if (noThinkTailSuffix(tpl, collapsed, enable_thinking, tools_json != null)) |suffix| {
            defer allocator.free(collapsed);
            return std.mem.concat(allocator, u8, &.{ collapsed, suffix });
        }
        return collapsed;
    }

    // WARN, not debug: a failed render silently swaps the prompt into
    // fallbackFormatChat's generic format — for families whose stop/turn
    // tokens differ (Gemma 4's <|turn> vs the fallback's <start_of_turn>)
    // that means degenerate generation, so the downgrade must be visible
    // at the default log level.
    if (jinja_c.jinja_last_error()) |e| {
        log.warn("jinja render failed ({s}), falling back to generic chat format\n", .{std.mem.span(e)});
    }
    const base = try fallbackFormatChat(allocator, msgs, chat_config, tools_json, tool_choice_instruction, enable_thinking);
    if (prefill) |partial| {
        defer allocator.free(base);
        return std.mem.concat(allocator, u8, &.{ base, partial });
    }
    return base;
}

pub fn hasExtraSystemMessages(messages: []const Message) bool {
    for (messages[@min(messages.len, 1)..]) |m| {
        if (std.mem.eql(u8, m.role, "system")) return true;
    }
    return false;
}

/// Only byte-pinned Qwen templates accept this ChatML extension. Unknown
/// revisions retain their native contract and use late-system consolidation.
fn qwenLateSystemTemplate(allocator: std.mem.Allocator, tpl: []const u8) !?[]const u8 {
    const templates = [_]struct { source: []const u8, content: []const u8 }{
        .{ .source = @embedFile("../../fixtures/qwen38_27b_chat_template.jinja"), .content = "render_content(message.content, false, true)|trim" },
        .{ .source = @embedFile("../../fixtures/qwen38_chat_template.jinja"), .content = "content" },
    };
    const branch =
        \\{%- if message.role == "system" %}
        \\        {%- if not loop.first %}
        \\            {{- raise_exception('System message must be at the beginning.') }}
        \\        {%- endif %}
    ;
    for (templates) |template| {
        if (!std.mem.eql(u8, tpl, template.source)) continue;
        if (std.mem.count(u8, tpl, branch) != 1) return null;
        const replacement = try std.mem.concat(allocator, u8, &.{
            "{%- if message.role == \"system\" %}{%- if not loop.first %}{{- '<|im_start|>system\\n' + (",
            template.content,
            ") + '<|im_end|>\\n' }}{%- endif %}",
        });
        defer allocator.free(replacement);
        return try std.mem.replaceOwned(u8, allocator, tpl, branch, replacement);
    }
    return null;
}

fn renderJinja(
    allocator: std.mem.Allocator,
    messages: []const Message,
    config: *const ChatConfig,
    tools_json: ?[]const u8,
    extra_json: []const u8,
    empty_content: EmptyContent,
) !?[]const u8 {
    const json = try serializeMessagesJsonFor(allocator, messages, empty_content, config);
    defer allocator.free(json);
    const adapted = if (hasExtraSystemMessages(messages)) try qwenLateSystemTemplate(allocator, config.chat_template) else null;
    defer if (adapted) |t| allocator.free(t);
    const tpl = try allocator.dupeSentinel(u8, adapted orelse config.chat_template, 0);
    defer allocator.free(tpl);
    const msgs = try allocator.dupeSentinel(u8, json, 0);
    defer allocator.free(msgs);
    const extra = try allocator.dupeSentinel(u8, extra_json, 0);
    defer allocator.free(extra);
    const tools = if (tools_json) |t| try allocator.dupeSentinel(u8, t, 0) else null;
    defer if (tools) |t| allocator.free(t);
    var len: usize = 0;
    const ptr = jinja_c.jinja_render_chat(tpl.ptr, msgs.ptr, if (tools) |t| t.ptr else null, extra.ptr, 1, &len) orelse return null;
    defer jinja_c.jinja_str_free(ptr);
    return try allocator.dupe(u8, ptr[0..len]);
}

/// Probe each distinct late-system boundary with the request's tools and kwargs.
/// Raising, dropping or moving the note to the front all require consolidation.
fn templatePreservesMidSystem(
    allocator: std.mem.Allocator,
    messages: []const Message,
    config: *const ChatConfig,
    tools_json: ?[]const u8,
    extra_json: []const u8,
    empty_content: EmptyContent,
) !bool {
    const before = "mlxserve_before_note_7c91";
    const note = "mlxserve_system_note_a83d";
    const after = "mlxserve_after_note_b462";
    const calls = [_]ToolCall{.{ .id = "probe_call", .name = "probe", .arguments = "{}" }};
    const Role = enum { user, assistant, tool, tail };
    var checked: u16 = 0;
    var i: usize = 0;
    while (i < messages.len) {
        if (!std.mem.eql(u8, messages[i].role, "system")) {
            i += 1;
            continue;
        }
        const start = i;
        while (i < messages.len and std.mem.eql(u8, messages[i].role, "system")) : (i += 1) {}
        if (start == 0) continue;
        const prev = messages[start - 1].role;
        const next = if (i < messages.len) messages[i].role else "tail";
        const prev_role = std.meta.stringToEnum(Role, prev) orelse return false;
        const next_role = std.meta.stringToEnum(Role, next) orelse return false;
        const bit = @as(u16, 1) << @as(u4, @intCast(@as(u8, @backingInt(prev_role)) * 4 + @backingInt(next_role)));
        if (checked & bit != 0) continue;
        checked |= bit;
        var probe = std.ArrayList(Message).empty;
        defer probe.deinit(allocator);
        try probe.append(allocator, .{ .role = "system", .content = "stable instructions" });
        if (!std.mem.eql(u8, prev, "user")) try probe.append(allocator, .{ .role = "user", .content = "question" });
        if (std.mem.eql(u8, prev, "tool")) try probe.append(allocator, .{ .role = "assistant", .content = "", .tool_calls = &calls });
        try probe.append(allocator, .{ .role = prev, .content = before, .tool_call_id = if (std.mem.eql(u8, prev, "tool")) "probe_call" else null });
        try probe.append(allocator, .{ .role = "system", .content = note });
        if (i < messages.len) try probe.append(allocator, .{ .role = next, .content = after });
        // A tail note must also survive when it becomes history next turn.
        for (0..@as(usize, if (i == messages.len) 2 else 1)) |pass| {
            if (pass == 1) try probe.append(allocator, .{ .role = "assistant", .content = after });
            const rendered = (try renderJinja(allocator, probe.items, config, tools_json, extra_json, empty_content)) orelse return false;
            defer allocator.free(rendered);
            if (!systemNoteInOrder(rendered, before, note, if (i < messages.len or pass == 1) after else null)) return false;
        }
    }
    return true;
}

fn systemNoteInOrder(rendered: []const u8, before: []const u8, note: []const u8, after: ?[]const u8) bool {
    const p = std.mem.indexOf(u8, rendered, before) orelse return false;
    const n = std.mem.indexOf(u8, rendered, note) orelse return false;
    if (n <= p) return false;
    if (after) |marker| {
        const a = std.mem.indexOf(u8, rendered, marker) orelse return false;
        if (a <= n) return false;
    }
    return true;
}

/// Null `reasoning_content` on assistant messages BEFORE the last user
/// message (harmony convention: prior-turn analysis is dropped from the
/// prompt; the current turn's tool rounds keep theirs). Returns null when
/// nothing would change, so the common no-history-reasoning case allocates
/// nothing.
fn dropPriorTurnReasoning(arena: std.mem.Allocator, messages: []const Message) !?[]const Message {
    var last_user: ?usize = null;
    for (messages, 0..) |m, i| {
        if (std.mem.eql(u8, m.role, "user")) last_user = i;
    }
    const cut = last_user orelse return null;
    var any = false;
    for (messages[0..cut]) |m| {
        if (std.mem.eql(u8, m.role, "assistant") and m.reasoning_content != null) {
            any = true;
            break;
        }
    }
    if (!any) return null;
    const out = try arena.dupe(Message, messages);
    for (out[0..cut]) |*m| {
        if (std.mem.eql(u8, m.role, "assistant")) m.reasoning_content = null;
    }
    return out;
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

/// A template that renders tool CALLS but names no 'tool' role (MiMo's generic
/// `<|im_start|>{{ role }}` header) was trained on tool turns: probe-render one
/// and keep it native when its content survives.
pub fn templateRendersToolTurn(allocator: std.mem.Allocator, tpl: []const u8) !bool {
    if (std.mem.indexOf(u8, tpl, "tool_calls") == null) return false;
    const marker = "__mlx_serve_tool_turn_probe__";
    const tpl_z = try allocator.dupeSentinel(u8, tpl, 0);
    defer allocator.free(tpl_z);
    var len: usize = 0;
    const out = jinja_c.jinja_render_chat(tpl_z.ptr, "[{\"role\":\"tool\",\"content\":\"" ++ marker ++ "\"}]", null, "{}", 0, &len) orelse return false;
    defer jinja_c.jinja_str_free(out);
    return std.mem.indexOf(u8, out[0..len], marker) != null;
}

/// True if any message has `role: "tool"` or an assistant message with tool_calls.
pub fn messagesHaveToolContent(messages: []const Message) bool {
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
pub fn synthesizeToolFallbackMessages(
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
    } else if (tool_choice_instruction) |instr| {
        // The template renders the tools itself; only the choice is ours to add.
        tool_prompt = std.mem.trim(u8, instr, "\n");
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
                .images = msg.images,
                .videos = msg.videos,
                .audio = msg.audio,
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

// ── tests ──

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
    const out = try collapseDoubledThinkTags(testing.allocator, "A</think></think>B</think></think>C");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("A</think>B</think>C", out);
}

test "renderChatTemplate: a tool-aware template with a generic role header renders tool turns natively" {
    // MiMo's template names no 'tool' literal: every non-assistant turn is
    // `<|im_start|>{{ role }}`. It renders tool CALLS, so tool results are its
    // own turns; a template that renders no tool calls keeps the rewrite.
    const a = testing.allocator;
    const generic = "{%- for m in messages -%}<|im_start|>{{ m.role }}\n{{ m.content }}" ++
        "{%- if m.tool_calls is defined and m.tool_calls -%}{%- for tc in m.tool_calls -%}<tool_call>{{ tc.function.name }}</tool_call>{%- endfor -%}{%- endif -%}<|im_end|>{%- endfor -%}";
    const calls = [_]ToolCall{.{ .id = "c1", .name = "get_weather", .arguments = "{\"city\":\"Paris\"}" }};
    const messages = [_]Message{
        .{ .role = "user", .content = "Weather?" },
        .{ .role = "assistant", .content = "", .tool_calls = &calls },
        .{ .role = "tool", .content = "18C", .tool_call_id = "c1" },
    };
    var config = ChatConfig{ .chat_template = generic, .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = a };
    const native = try renderChatTemplate(a, &messages, &config, null, null, true, null, false);
    defer a.free(native);
    try testing.expect(std.mem.indexOf(u8, native, "<|im_start|>tool\n18C<|im_end|>") != null);
    try testing.expect(std.mem.indexOf(u8, native, "<tool_response>") == null);

    config.chat_template = "{%- for m in messages -%}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>{%- endfor -%}";
    const plain = try renderChatTemplate(a, &messages, &config, null, null, true, null, false);
    defer a.free(plain);
    try testing.expect(std.mem.indexOf(u8, plain, "<|im_start|>user\n<tool_response>\n18C\n</tool_response>") != null);
}

test "renderChatTemplate: tool result with raw ANSI escapes still renders via Jinja" {
    // Regression for the pi/gemma-4-31b session: the third request
    // was the first whose history held a tool result with a raw ESC byte (the
    // interactive `npx sv create` output). The Jinja render failed on the
    // unescaped messages JSON and silently downgraded to fallbackFormatChat's
    // Gemma-2/3 `<start_of_turn>` text format — not special tokens for Gemma
    // 4, so generation never stopped at turn end. The render must SUCCEED via
    // the template (template marker present, fallback marker absent).
    const allocator = testing.allocator;
    const tpl =
        \\{%- for message in messages -%}
        \\{%- if message['role'] == 'tool' -%}<|tool_response>{{ message['content'] }}<turn|>
        \\{%- else -%}<|turn>{{ message['role'] }}
        \\{{ message['content'] }}{% if message.tool_calls %}{% for tc in message.tool_calls %}<|tool_call>{{ tc.function.name }}<tool_call|>{% endfor %}{% endif %}<turn|>
        \\{%- endif -%}
        \\{%- endfor -%}
        \\{# tools referenced so no synthesis kicks in: {{ tools }} #}
    ;
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tc = [_]ToolCall{.{ .id = "tc_0", .name = "bash", .arguments = "{\"command\": \"npx sv create .\"}" }};
    const messages = [_]Message{
        .{ .role = "user", .content = "make me a sveltekit app" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        // Verbatim shape from the live failure: hide-cursor ANSI code + prompt UI.
        .{ .role = "tool", .content = "\x1b[?25l\u{2502}\n\u{25c6}  Which template would you like?", .tool_call_id = "tc_0" },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, "[]", null, false, null, false);
    defer allocator.free(rendered);

    try testing.expect(std.mem.indexOf(u8, rendered, "<|tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "Which template") != null);
    // Fallback format must NOT have been used.
    try testing.expect(std.mem.indexOf(u8, rendered, "<start_of_turn>") == null);
}

test "renderChatTemplate: Hy3 constructs — str.format tokens, arguments.items(), tojson(ensure_ascii=False)" {
    // Hy3's chat_template.jinja builds EVERY special token via
    // `'<think{}>'.format(HYTK)` on line 1 — an engine without str.format
    // throws there, and renderChatTemplate silently downgrades to
    // fallbackFormatChat (the wrong-family prompt-format class). This pins the
    // three constructs the template leans on: positional {} format, dict
    // .items() iteration over tool-call arguments (requires args serialized as
    // an OBJECT), and `| tojson(ensure_ascii=False)` on non-string values.
    const allocator = testing.allocator;
    const tpl =
        \\{%- set HYTK = ':opensource' %}
        \\{%- set eos_token = '<eos{}>'.format(HYTK) %}
        \\{%- set user_token = '<user{}>'.format(HYTK) %}
        \\{%- for message in messages -%}
        \\{%- if message['role'] == 'user' -%}{{ user_token }}{{ message['content'] }}
        \\{%- elif message['role'] == 'assistant' -%}<asst>{{ message['content'] }}
        \\{%- if message['tool_calls'] %}{% for tool in message['tool_calls'] %}{%- set arguments = tool['function']['arguments'] %}<call>{{ tool['function']['name'] }}{% for key, value in arguments.items() %}<k>{{ key }}</k>{% if value is not string %}{%- set value = value | tojson(ensure_ascii=False) %}{% endif %}<v>{{ value }}</v>{% endfor %}</call>{% endfor %}{{ eos_token }}{% endif %}
        \\{%- endif -%}
        \\{%- endfor -%}
        \\{# {{ tools }} #}
    ;
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tc = [_]ToolCall{.{ .id = "tc_0", .name = "write_file", .arguments = "{\"path\": \"a.txt\", \"count\": 2}" }};
    const messages = [_]Message{
        .{ .role = "user", .content = "write it" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, "[]", null, false, null, false);
    defer allocator.free(rendered);

    try testing.expect(std.mem.indexOf(u8, rendered, "<user:opensource>write it") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "<eos:opensource>") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "<call>write_file") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "<k>path</k><v>a.txt</v>") != null);
    // Non-string value rendered through tojson.
    try testing.expect(std.mem.indexOf(u8, rendered, "<k>count</k><v>2</v>") != null);
}

test "renderChatTemplate: REAL MiniCPM5 chat_template.jinja multi-turn tool history renders without fallback (MINICPM5_MODEL_DIR)" {
    // Env-gated: MINICPM5_MODEL_DIR=<dir containing chat_template.jinja>.
    //
    // MiniCPM5-1B's template line 59 does
    //     {%- set min_count = [tool_calls_count, tool_sep_count]|min %}
    // inside the `{%- if message.tool_calls %}` history branch — reached ONLY
    // when a prior assistant tool call is replayed into history, i.e. the
    // SECOND turn of any tool conversation. jinja_cpp registered `min`/`max`
    // only in the EMPTY-sequence table, so a 2-element array threw
    //   "Unknown (built-in) filter 'min' for type Array"
    // and renderChatTemplate silently downgraded to fallbackFormatChat, which
    // emits Gemma-family <start_of_turn>/<end_of_turn> that MiniCPM5 never saw
    // in training — it loses its stop token and degenerates into unrelated
    // text. Single-turn requests never enter the branch, which is why the
    // MiniCPM5 V3 parser corpus (all single-turn) never caught it.
    // Same wrong-family prompt-format class as the Hy3 str.format test above.
    const dir = env.get(.MINICPM5_MODEL_DIR) orelse return error.SkipZigTest;
    const allocator = testing.allocator;
    const path = try std.fmt.allocPrint(allocator, "{s}/chat_template.jinja", .{std.mem.span(dir)});
    defer allocator.free(path);
    const io = std.Io.Threaded.global_single_threaded.io();
    const tpl = blk: {
        const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return error.SkipZigTest;
        defer f.close(io);
        var read_buf: [4096]u8 = undefined;
        var reader_state = f.reader(io, &read_buf);
        break :blk try reader_state.interface.allocRemaining(allocator, .limited(1024 * 1024));
    };
    defer allocator.free(tpl);
    // Only meaningful against a template that actually uses the filter.
    if (std.mem.indexOf(u8, tpl, "|min") == null) return error.SkipZigTest;

    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tools_json =
        \\[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
    ;
    const tc = [_]ToolCall{.{ .id = "tc_0", .name = "read_file", .arguments = "{\"path\": \"/tmp/marker.txt\"}" }};
    const messages = [_]Message{
        .{ .role = "user", .content = "Read /tmp/marker.txt" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        .{ .role = "tool", .content = "ALPHA-7719", .tool_call_id = "tc_0" },
    };

    const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false, null, false);
    defer allocator.free(rendered);

    // MiniCPM5's own family tokens survived => the Jinja render succeeded.
    try testing.expect(std.mem.indexOf(u8, rendered, "<|im_start|>") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "<|im_end|>") != null);
    // The tool round made it through the |min branch.
    try testing.expect(std.mem.indexOf(u8, rendered, "<function name=\"read_file\">") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "<tool_response>") != null);
    // The decisive assertion: NO silent downgrade to the generic fallback.
    try testing.expect(std.mem.indexOf(u8, rendered, "<start_of_turn>") == null);
    try testing.expect(std.mem.indexOf(u8, rendered, "<end_of_turn>") == null);
}

test "renderChatTemplate: |min picks the smaller element, |max the larger" {
    // Guards against a min/max implementation that just returns first/last.
    const allocator = testing.allocator;
    const tpl =
        \\{%- for message in messages -%}
        \\{{ [3, 1, 2]|min }}/{{ [3, 1, 2]|max }}
        \\{%- endfor -%}
    ;
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{.{ .role = "user", .content = "hi" }};
    const rendered = try renderChatTemplate(allocator, &messages, &config, "[]", null, false, null, false);
    defer allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "1/3") != null);
}

test "renderChatTemplate: a namespace attribute named like a dict method is the attribute" {
    // `ns.items` resolved to dict.items(), `+` raised, and the render fell back to the generic format.
    const allocator = testing.allocator;
    const tpl =
        \\{%- set ns = namespace(items=[]) -%}
        \\{%- for message in messages -%}{%- set ns.items = ns.items + [message.content] -%}{%- endfor -%}
        \\ITEMS={{ ns.items | join(',') }}
    ;
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{ .{ .role = "user", .content = "hi" }, .{ .role = "assistant", .content = "there" } };
    const rendered = try renderChatTemplate(allocator, &messages, &config, null, null, false, null, false);
    defer allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "ITEMS=hi,there") != null);
}

test "renderChatTemplate: REAL Hy3 chat_template.jinja renders without fallback (HY3_MODEL_DIR)" {
    // Env-gated: HY3_MODEL_DIR=<dir containing chat_template.jinja>. Renders
    // the actual shipped template with system + tools + a tool round and
    // pins: no silent fallback (family tokens present), template-opened think
    // (thinking on), closed think (thinking off), tool-call args rendered via
    // arguments.items(), tool_response wrapping for role:"tool".
    const dir = env.get(.HY3_MODEL_DIR) orelse return error.SkipZigTest;
    const allocator = testing.allocator;
    const path = try std.fmt.allocPrint(allocator, "{s}/chat_template.jinja", .{std.mem.span(dir)});
    defer allocator.free(path);
    const io = std.Io.Threaded.global_single_threaded.io();
    const tpl = blk: {
        const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return error.SkipZigTest;
        defer f.close(io);
        var read_buf: [4096]u8 = undefined;
        var reader_state = f.reader(io, &read_buf);
        break :blk try reader_state.interface.allocRemaining(allocator, .limited(1024 * 1024));
    };
    defer allocator.free(tpl);

    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tools_json =
        \\[{"type":"function","function":{"name":"get_time","description":"Get time","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}}]
    ;
    const tc = [_]ToolCall{.{ .id = "tc_0", .name = "get_time", .arguments = "{\"timezone\": \"UTC\"}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "What time is it?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        .{ .role = "tool", .content = "12:34 UTC", .tool_call_id = "tc_0" },
    };

    // Thinking ON: generation prompt must end inside a think block.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "hy_begin_of_sentence:opensource") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "hy_User:opensource") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<tool_call:opensource>get_time<tool_sep:opensource>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<arg_key:opensource>timezone</arg_key:opensource>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<arg_value:opensource>UTC</arg_value:opensource>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<tool_response:opensource>") != null);
        // No fallback-format markers.
        try testing.expect(std.mem.indexOf(u8, rendered, "<|im_start|>") == null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<start_of_turn>") == null);
        // Template-opened think at the generation prompt.
        try testing.expect(promptTailOpensThink(rendered));
    }
    // Thinking OFF (reasoning_effort no_think): think opened AND closed.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false, null, false);
        defer allocator.free(rendered);
        try testing.expect(!promptTailOpensThink(rendered));
        try testing.expect(std.mem.indexOf(u8, rendered, "<think:opensource></think:opensource>") != null);
    }
}

test "renderChatTemplate: REAL Inkling chat_template.jinja renders without fallback (hermetic)" {
    // src/fixtures/inkling_chat_template.jinja is the verbatim template from
    // pipenetwork/Inkling-Small-MLX-* (Thinking Machines Inkling Small). It
    // exercises jinja features no other family does — namespace(), dict/tuple
    // literals, tojson(sort_keys=true, separators=(",", ":")), `is mapping` —
    // and it RAISES on any reasoning_effort string it doesn't know (including
    // hy3's "no_think") and on tool-call arguments passed as a JSON STRING. A
    // raise = silent fallbackFormatChat = wrong-family tags, so the render is
    // pinned hermetically for both thinking arms.
    const allocator = testing.allocator;
    const tpl = @embedFile("../../fixtures/inkling_chat_template.jinja");

    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tools_json =
        \\[{"type":"function","function":{"name":"get_time","description":"Get time","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}}]
    ;
    const tc = [_]ToolCall{.{ .id = "tc_0", .name = "get_time", .arguments = "{\"timezone\": \"UTC\"}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "What time is it?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        .{ .role = "tool", .content = "12:34 UTC", .tool_call_id = "tc_0" },
    };

    // Thinking ON: effort header 0.9 (our "high"), tool declare, tool call as
    // {"name":...,"args":{...}} with args as an OBJECT (sorted, compact), tool
    // result named via tool_call_id lookup, generation prompt at the tail.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<|message_system|>tool_declare<|content_xml|>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "\"name\":\"get_time\"") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<|message_user|><|content_text|>What time is it?<|end_message|>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<|message_model|>get_time<|content_invoke_tool_json|>{\"name\":\"get_time\",\"args\":{\"timezone\":\"UTC\"}}<|end_message|>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<|content_model_end_sampling|>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<|message_tool|>get_time<|content_text|>12:34 UTC<|end_message|>") != null);
        try testing.expect(std.mem.endsWith(u8, rendered, "<|message_model|>"));
        // No fallback-format markers (a raise inside the template silently falls back).
        try testing.expect(std.mem.indexOf(u8, rendered, "<|im_start|>") == null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<start_of_turn>") == null);
    }
    // Thinking OFF: the effort header must be the template's "0" form — hy3's
    // "no_think" is UNKNOWN to this template and raises.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "Thinking effort level: 0<|end_message|>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<|im_start|>") == null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<start_of_turn>") == null);
    }
    // History reasoning round-trip (laguna class): this template PERSISTS
    // assistant reasoning as its own <|content_thinking|> message. If the
    // field ever stops reaching the render, prior turns silently lose their
    // thinking blocks while turn 1 still works.
    {
        const hist = [_]Message{
            .{ .role = "user", .content = "Why is the sky blue?" },
            .{ .role = "assistant", .content = "Rayleigh scattering.", .reasoning_content = "Shorter wavelengths scatter more." },
            .{ .role = "user", .content = "And sunsets?" },
        };
        const rendered = try renderChatTemplate(allocator, &hist, &config, null, null, true, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "<|message_model|><|content_thinking|>Shorter wavelengths scatter more.<|end_message|>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<|message_model|><|content_text|>Rayleigh scattering.<|end_message|>") != null);
        try testing.expect(std.mem.endsWith(u8, rendered, "<|message_model|>"));
    }
}

test "renderChatTemplate: REAL Qwen3.8 chat_template.jinja renders without fallback (hermetic)" {
    // src/fixtures/qwen38_chat_template.jinja is verbatim from
    // Qwen/Qwen3.8-2.4T-A95B, the first public Qwen3.8 checkpoint (the VL
    // siblings keep this body and re-add the vision macro). Qwen3.8 added two
    // raise_exception gates that make our previous extra-context values
    // unrenderable on this family:
    //   - `enable_thinking is false` -> 'Disabling thinking is not supported.'
    //   - reasoning_effort outside (xhigh, medium, low) -> raise, and we sent
    //     "high" on every thinking request, "no_think" on every off one.
    // A raise is a silent fallbackFormatChat, and this family's fallback is
    // ChatML too, so the downgrade shows up only as a missing tools preamble.
    // Both arms pinned.
    const allocator = testing.allocator;
    const tpl = @embedFile("../../fixtures/qwen38_chat_template.jinja");

    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tools_json =
        \\[{"type":"function","function":{"name":"get_time","description":"Get time","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}}]
    ;
    const tc = [_]ToolCall{.{ .id = "tc_0", .name = "get_time", .arguments = "{\"timezone\": \"UTC\"}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "What time is it?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        .{ .role = "tool", .content = "12:34 UTC", .tool_call_id = "tc_0" },
    };

    // Thinking ON, no client effort: OUR default, which is deliberately NOT
    // the checkpoint's (see qwen38EffortFor) — xhigh reasons without a bound
    // and a client that sends nothing gets no bound either.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "Reasoning effort is set to low.") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "set to xhigh") == null);
        try testing.expect(std.mem.indexOf(u8, rendered, "You have access to the following functions:") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<function=get_time>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<parameter=timezone>\nUTC\n</parameter>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<tool_response>\n12:34 UTC\n</tool_response>") != null);
        // Unconditional think opener at the generation prompt.
        try testing.expect(std.mem.endsWith(u8, rendered, "<|im_start|>assistant\n<think>\n"));
        try testing.expect(promptTailOpensThink(rendered));
    }
    // OpenAI's "high" is this family's "xhigh"; "medium" injects no preamble.
    {
        const hi = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, "high", false);
        defer allocator.free(hi);
        try testing.expect(std.mem.indexOf(u8, hi, "Reasoning effort is set to xhigh.") != null);

        const med = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, "medium", false);
        defer allocator.free(med);
        try testing.expect(std.mem.indexOf(u8, med, "Reasoning effort is set to") == null);
        try testing.expect(std.mem.indexOf(u8, med, "You have access to the following functions:") != null);
    }
    // Thinking OFF: the template REFUSES to disable thinking, so the render
    // must stay on the low-effort arm and the block the template opened is
    // closed in the prompt (noThinkTailSuffix) instead.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "Reasoning effort is set to low.") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "You have access to the following functions:") != null);
        try testing.expect(std.mem.endsWith(u8, rendered, "</think>"));
        try testing.expect(!promptTailOpensThink(rendered));
    }
}

test "renderChatTemplate: tool_choice reaches a tool-aware template's prompt" {
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = @embedFile("../../fixtures/qwen38_chat_template.jinja"),
        .bos_token = null,
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const tools_json =
        \\[{"type":"function","function":{"name":"calculator","description":"Math","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}]
    ;
    const instr = "\nYou MUST call one of the available functions. Do not respond with text.";
    const bare = [_]Message{.{ .role = "user", .content = "What is 7 times 8?" }};
    const with_system = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "What is 7 times 8?" },
    };
    for ([_][]const Message{ &bare, &with_system }) |msgs| {
        const rendered = try renderChatTemplate(allocator, msgs, &config, tools_json, instr, true, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "You have access to the following functions:") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "You MUST call one of the available functions.") != null);
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rendered, "<|im_start|>system"));
    }
}

test "renderChatTemplate: Qwen3.8-27B ACCEPTS thinking-off natively (hermetic)" {
    // Same family, DIFFERENT gates: `qwen38_chat_template.jinja` (2.4T-A95B)
    // raises on `enable_thinking is false`, while the 27B's template answers
    // that arm the 3.6 way — a CLOSED `<think>\n\n</think>\n\n` and no
    // reasoning-instructions preamble at all (its effort validation lives
    // INSIDE the thinking-on branch). Sniffing the shared `'xhigh'` literal
    // for both decisions forced every thinking-off request onto the
    // thinking-on arm at effort low, which injects a system line the
    // checkpoint never renders for that arm and leaves the block to be closed
    // by `noThinkTailSuffix` instead of by the template. The effort mapping is
    // still keyed on `'xhigh'` (both templates raise on OpenAI's "high"); only
    // the enable_thinking override is keyed on the refusal itself.
    const allocator = testing.allocator;
    const tpl = @embedFile("../../fixtures/qwen38_27b_chat_template.jinja");

    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tools_json =
        \\[{"type":"function","function":{"name":"get_time","description":"Get time","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}}]
    ;
    const messages = [_]Message{
        .{ .role = "user", .content = "What time is it?" },
    };

    // Thinking ON is unchanged: OpenAI's "high" still has to become this
    // family's "xhigh" or the render raises and silently falls back.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, "high", false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "Reasoning effort is set to xhigh.") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "You have access to the following functions:") != null);
        try testing.expect(std.mem.endsWith(u8, rendered, "<|im_start|>assistant\n<think>\n"));
        try testing.expect(promptTailOpensThink(rendered));
    }
    // Absent client effort takes OUR default (low), not the template's xhigh.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "Reasoning effort is set to low.") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "set to xhigh") == null);
    }
    // Thinking OFF renders the template's OWN no-think prompt: closed block,
    // no reasoning preamble, and nothing appended after it.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "Reasoning effort is set to") == null);
        try testing.expect(std.mem.indexOf(u8, rendered, "You have access to the following functions:") != null);
        try testing.expect(std.mem.endsWith(u8, rendered, "<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        try testing.expect(!promptTailOpensThink(rendered));
    }
}

test "assistant history reasoning_content reaches a template that reads it (laguna round-trip)" {
    // laguna's chat_template.jinja persists reasoning across turns: history
    // assistant messages render <think>{message.reasoning|reasoning_content}</think>.
    // When the parse layer drops the field, every history turn renders the
    // empty <think></think> nothink signature and the model stops thinking
    // from turn 2 of a session (live, pi agent on Laguna XS: one
    // reasoning delta on the 2-msg turn, zero across the next 13k chunks).
    // Template fragment mirrors laguna's assistant-history branch.
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = "{%- for message in messages -%}" ++
            "{%- if message.role == 'assistant' -%}" ++
            "{%- set rc = '' -%}" ++
            "{%- if message.reasoning is string -%}{%- set rc = message.reasoning -%}" ++
            "{%- elif message.reasoning_content is string -%}{%- set rc = message.reasoning_content -%}" ++
            "{%- endif -%}" ++
            "<think>{{ rc }}</think>{{ message.content }}" ++
            "{%- endif -%}{%- endfor -%}",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    _ = &config;
    const messages = [_]Message{
        .{ .role = "user", .content = "What is 2+2?" },
        .{ .role = "assistant", .content = "4", .reasoning_content = "two plus two is four" },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, false);
    defer allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "<think>two plus two is four</think>4") != null);
}

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
    const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false, null, false);
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

test "renderChatTemplate: muse_glimmer shipped template renders byte-identical to python jinja2 (hermetic)" {
    // src/fixtures/muse_chat_template.jinja is the VERBATIM template
    // meta-models/Muse-Glimmer-30B ships; muse_render_reference.txt is python
    // jinja2's render of the exact message set below (regenerate with the
    // script in the fixture header if either changes). Byte equality pins:
    // <|start|>role<|message|>…<|eot|> framing, PRIOR-turn reasoning dropped
    // before the template runs (harmony convention — the reference is jinja2's
    // render of the message set with that field already nulled; the
    // current-turn keep arm is pinned by the drops-PRIOR-turn test below),
    // tool calls as to=<fn> ATEM blocks with dict arguments
    // (string params RAW, bools bare), tool results as
    // "<|start|>tool <name><|message|><tool_output …>", the namespaced tool
    // catalog + "# Valid recipients" line, and the bare "<|start|>assistant"
    // generation prompt (NO <|message|> — the model emits its own recipient).
    const allocator = testing.allocator;
    const tpl = @embedFile("../../fixtures/muse_chat_template.jinja");
    const expected = @embedFile("../../fixtures/muse_render_reference.txt");

    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "<|begin_of_text|>",
        .eos_token = "<|end_of_text|>",
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tools_json =
        \\[{"type": "function", "function": {"name": "weather.get", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    ;
    // Alphabetical argument keys on purpose: a sorted-map JSON layer must
    // produce the same parameter order as python's insertion-ordered dicts.
    const tc = [_]ToolCall{.{ .id = "call_1", .name = "weather.get", .arguments = "{\"city\": \"Paris\", \"detailed\": false, \"units\": \"celsius\"}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "You are a helpful assistant." },
        .{ .role = "user", .content = "What's the weather in Paris?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc, .reasoning_content = "Need the weather tool." },
        .{ .role = "tool", .content = "{\"temp\": 21}", .tool_call_id = "call_1" },
        .{ .role = "user", .content = "thanks, and in kelvin?" },
    };

    const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, null, false);
    defer allocator.free(rendered);
    try testing.expectEqualStrings(expected, rendered);
}

test "renderChatTemplate: muse thinking-off no-tools commits the to=user channel" {
    // Thinking off must SKIP the reasoning pass, not generate-and-discard it:
    // spending the recipient header in the prompt makes a `to=self` segment
    // unreachable. Tools stay bare — a tool call is a `to=<fn>` header.
    const allocator = testing.allocator;
    const tpl = @embedFile("../../fixtures/muse_chat_template.jinja");
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "<|begin_of_text|>",
        .eos_token = "<|end_of_text|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{.{ .role = "user", .content = "hi" }};

    const off = try renderChatTemplate(allocator, &messages, &config, null, null, false, null, false);
    defer allocator.free(off);
    try testing.expect(std.mem.endsWith(u8, off, "<|start|>assistant to=user<|message|>"));

    const on = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, false);
    defer allocator.free(on);
    try testing.expect(std.mem.endsWith(u8, on, "<|start|>assistant"));

    const tools_json =
        \\[{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
    ;
    const off_tools = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false, null, false);
    defer allocator.free(off_tools);
    try testing.expect(std.mem.endsWith(u8, off_tools, "<|start|>assistant"));
}

test "renderChatTemplate: muse template dedups a client-supplied reasoning directive" {
    // Upstream a4e59da (#35): a system prompt already carrying a "Reasoning
    // effort/strength" line used to get the template's own "Reasoning
    // strength: high." appended after it — two conflicting directives, ours
    // stomping the client's. The template now normalizes the spelling and
    // skips its own line. Rendered through OUR jinja.cpp because the fix
    // rides the `replace`/`lower` filters and the `in` operator — a silently
    // no-op filter keeps the duplicate while the render still succeeds.
    const allocator = testing.allocator;
    const tpl = @embedFile("../../fixtures/muse_chat_template.jinja");
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "<|begin_of_text|>",
        .eos_token = "<|end_of_text|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{
        .{ .role = "system", .content = "Be terse.\nReasoning effort: low." },
        .{ .role = "user", .content = "hi" },
    };
    const r = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, false);
    defer allocator.free(r);
    try testing.expect(std.mem.indexOf(u8, r, "Reasoning strength: low.") != null);
    try testing.expect(std.mem.indexOf(u8, r, "Reasoning effort") == null);
    const first = std.mem.indexOf(u8, r, "Reasoning strength").?;
    try testing.expect(std.mem.indexOfPos(u8, r, first + 1, "Reasoning strength") == null);
}

test "renderChatTemplate: thinking-off closes a template-opened think block (LFM2.5 class)" {
    // LFM2.5's template opens `<think>` unconditionally; thinking-off used to
    // mean generating the whole reasoning pass and discarding it. Closing the
    // block in the prompt skips it. Thinking ON keeps the opener untouched.
    const allocator = testing.allocator;
    const tpl = "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n<think>{%- endif -%}";
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{.{ .role = "user", .content = "hi" }};

    const off = try renderChatTemplate(allocator, &messages, &config, null, null, false, null, false);
    defer allocator.free(off);
    try testing.expect(std.mem.endsWith(u8, off, "<think></think>"));

    const on = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, false);
    defer allocator.free(on);
    try testing.expect(std.mem.endsWith(u8, on, "<think>"));
}

test "renderChatTemplate: continuation appends the partial reply into the OPEN assistant turn" {
    // Continuing a reply means the prompt ends mid-assistant-turn. The trailing
    // assistant message is a PREFIX to extend, not history: rendering it as
    // history and then letting add_generation_prompt open a second assistant
    // turn is what the server did before this existed, and the model answers
    // the doubled turn by starting over.
    const allocator = testing.allocator;
    const tpl = "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "count to three" },
        .{ .role = "assistant", .content = "one, two," },
    };

    const cont = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, true);
    defer allocator.free(cont);
    // Ends INSIDE the turn, on the partial text — no close tag after it.
    try testing.expect(std.mem.endsWith(u8, cont, "<|im_start|>assistantone, two,"));
    // Exactly one assistant turn was opened.
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, cont, "<|im_start|>assistant"));
    // The partial appears once — as the prefill, not also as history.
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, cont, "one, two,"));

    // Same messages WITHOUT the flag stay history + a fresh turn, unchanged.
    const plain = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, false);
    defer allocator.free(plain);
    try testing.expect(std.mem.endsWith(u8, plain, "<|im_start|>assistant"));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, plain, "<|im_start|>assistant"));
}

test "renderChatTemplate: continuation resumes in the CONTENT channel, not a think block" {
    // A template that opens `<think>` unconditionally (LFM2.5 class) would
    // otherwise put the partial reply — visible prose the user already read —
    // inside a reasoning block, and the continuation would come back as
    // reasoning. The same tail that thinking-off uses closes it first.
    const allocator = testing.allocator;
    const tpl = "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n<think>{%- endif -%}";
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "hi" },
        .{ .role = "assistant", .content = "Hello ther" },
    };

    // Thinking ON is the hard case: nothing else would close the block.
    const cont = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, true);
    defer allocator.free(cont);
    try testing.expect(std.mem.endsWith(u8, cont, "<think></think>Hello ther"));
}

test "renderChatTemplate: a continuation prefill is trimmed of trailing whitespace" {
    // The tokenizer merges a trailing space with the word that follows it, so
    // a prompt ending in one splits differently than the same text mid-reply —
    // the hazard Anthropic's API rejects prefills outright for. Trim it: the
    // stored reply keeps its whitespace, only the PROMPT is trimmed.
    const allocator = testing.allocator;
    const tpl = "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "hi" },
        .{ .role = "assistant", .content = "Once upon a time \n" },
    };
    const cont = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, true);
    defer allocator.free(cont);
    try testing.expect(std.mem.endsWith(u8, cont, "Once upon a time"));
}

test "renderChatTemplate: continuation is declined when the last message is not an assistant" {
    // The flag describes the SHAPE of the request; a client that sets it on a
    // user-final conversation gets an ordinary turn rather than a prompt with
    // the user's own words committed into the assistant's mouth.
    const allocator = testing.allocator;
    const tpl = "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{.{ .role = "user", .content = "hi" }};
    const cont = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, true);
    defer allocator.free(cont);
    try testing.expect(std.mem.endsWith(u8, cont, "<|im_start|>assistant"));
}

test "renderChatTemplate: muse commits its content channel before the partial reply" {
    // Muse's turn header is `<|start|>assistant to=<recipient><|message|>`, and
    // an uncommitted header means the model picks the channel — including
    // `to=self`, which would make the continuation reasoning.
    const allocator = testing.allocator;
    const tpl = @embedFile("../../fixtures/muse_chat_template.jinja");
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "<|begin_of_text|>",
        .eos_token = "<|return|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "hi" },
        .{ .role = "assistant", .content = "Hello, I was say" },
    };
    const cont = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, true);
    defer allocator.free(cont);
    try testing.expect(std.mem.endsWith(u8, cont, " to=user<|message|>Hello, I was say"));
}

test "renderChatTemplate: muse drops PRIOR-turn reasoning, keeps the current tool round's" {
    // Harmony (muse's format ancestor) drops analysis from turns before the
    // last user message. Without that, the app's reasoning round-trip renders
    // every old chain — a loop-cut tail included — as a to=self history
    // segment the model reads back on every later turn (one
    // repetition-primed turn poisoned the rest of the chat).
    const allocator = testing.allocator;
    const tpl = @embedFile("../../fixtures/muse_chat_template.jinja");
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "<|begin_of_text|>",
        .eos_token = "<|end_of_text|>",
        .add_bos_token = false,
        .allocator = allocator,
    };

    // Prior turn: assistant reasoning BEFORE the last user message → dropped;
    // its content survives.
    const prior = [_]Message{
        .{ .role = "user", .content = "hi" },
        .{ .role = "assistant", .content = "hello there", .reasoning_content = "waffling chain" },
        .{ .role = "user", .content = "and again?" },
    };
    const r1 = try renderChatTemplate(allocator, &prior, &config, null, null, true, null, false);
    defer allocator.free(r1);
    try testing.expect(std.mem.indexOf(u8, r1, "to=self") == null);
    try testing.expect(std.mem.indexOf(u8, r1, "waffling chain") == null);
    try testing.expect(std.mem.indexOf(u8, r1, "hello there") != null);

    // Current turn: a tool round AFTER the last user message keeps its chain —
    // that is where the model needs it.
    const tools_json =
        \\[{"type": "function", "function": {"name": "weather.get", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    ;
    const tc = [_]ToolCall{.{ .id = "call_1", .name = "weather.get", .arguments = "{\"city\": \"Paris\"}" }};
    const current = [_]Message{
        .{ .role = "user", .content = "weather in Paris?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc, .reasoning_content = "Need the weather tool." },
        .{ .role = "tool", .content = "{\"temp\": 21}", .tool_call_id = "call_1" },
    };
    const r2 = try renderChatTemplate(allocator, &current, &config, tools_json, null, true, null, false);
    defer allocator.free(r2);
    try testing.expect(std.mem.indexOf(u8, r2, "<|start|>assistant to=self<|message|>Need the weather tool.<|eom|>") != null);

    // A non-muse template that persists history reasoning (laguna class) is
    // untouched — the round-trip rule stays per-family.
    const generic_tpl = "{%- for m in messages -%}{%- if m.get('reasoning_content') -%}<think>{{ m['reasoning_content'] }}</think>{%- endif -%}{{ m['role'] }}: {{ m['content'] }}\n{%- endfor -%}";
    var generic = ChatConfig{
        .chat_template = generic_tpl,
        .bos_token = "",
        .eos_token = "",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const r3 = try renderChatTemplate(allocator, &prior, &generic, null, null, true, null, false);
    defer allocator.free(r3);
    try testing.expect(std.mem.indexOf(u8, r3, "<think>waffling chain</think>") != null);
}

test "renderChatTemplate: dsv4 template renders tools + DSML history + tool_result (hermetic)" {
    // src/fixtures/dsv4_chat_template.jinja is OUR faithful transcription of
    // DeepSeek-V4's encoding_dsv4.py (the checkpoint ships none): TOOLS block
    // into the system position, assistant history tool calls as DSML
    // invoke/parameter markup (string params raw, JSON params via tojson —
    // whose default ", "/": " separators match python json.dumps), tool role
    // as <tool_result> user turns (consecutive results merge), thinking_mode
    // chat|thinking with the reference's drop-thinking rule (reasoning kept
    // only for assistant turns AFTER the last user/tool message).
    const allocator = testing.allocator;
    const tpl = @embedFile("../../fixtures/dsv4_chat_template.jinja");

    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tools_json =
        \\[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}}}]
    ;
    const tc = [_]ToolCall{.{ .id = "tc_0", .name = "get_weather", .arguments = "{\"city\": \"Paris\", \"days\": 3}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "Weather in Paris?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        .{ .role = "tool", .content = "Sunny, 22C", .tool_call_id = "tc_0" },
    };

    // Thinking OFF (chat mode): assistant turns open with the bare </think>.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "You are helpful.\n\n## Tools") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "\"<｜DSML｜tool_calls>\" block") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "\"name\": \"get_weather\"") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<｜User｜>Weather in Paris?") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<｜Assistant｜></think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "</｜DSML｜tool_calls><｜end▁of▁sentence｜>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<｜User｜><tool_result>Sunny, 22C</tool_result>") != null);
        try testing.expect(std.mem.endsWith(u8, rendered, "<｜Assistant｜></think>"));
        // No fallback-format markers (a raise inside the template silently falls back).
        try testing.expect(std.mem.indexOf(u8, rendered, "<|im_start|>") == null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<start_of_turn>") == null);
    }
    // Thinking ON with TOOLS: the reference DISABLES drop_thinking whenever
    // tools are declared (agent sessions keep reasoning on every assistant
    // turn — the laguna round-trip class), so the tool-calling assistant
    // renders <think>…</think> even though a tool result follows it.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.endsWith(u8, rendered, "<｜Assistant｜><think>"));
        try testing.expect(std.mem.indexOf(u8, rendered, "<｜Assistant｜><think></think>\n\n<｜DSML｜tool_calls>") != null);
        // 0731's reasoning-effort preamble must NOT appear at the serializer's
        // dsv4 default (the reference default is "low" = nothing). hy3's
        // blanket "high" would have prepended it to every thinking request.
        try testing.expect(std.mem.indexOf(u8, rendered, "Reasoning Effort:") == null);
    }
    // Client-supplied effort reaches the render MAPPED (dsv4EffortFor): the
    // raw CLIENT vocabulary goes in ("xhigh"), the template sees "max" and
    // prepends the max preamble. This is the end-to-end pin the unit tests
    // can't give — the string must survive renderChatTemplate →
    // serializeExtraContext → Jinja.
    {
        const user_only = [_]Message{.{ .role = "user", .content = "prove it" }};
        const r_max = try renderChatTemplate(allocator, &user_only, &config, null, null, true, "xhigh", false);
        defer allocator.free(r_max);
        try testing.expect(std.mem.indexOf(u8, r_max, "Reasoning Effort: Beyond maximum") != null);
        const r_high = try renderChatTemplate(allocator, &user_only, &config, null, null, true, "high", false);
        defer allocator.free(r_high);
        try testing.expect(std.mem.indexOf(u8, r_high, "Reasoning Effort: Absolute maximum") != null);
        // Outside thinking mode the preamble never applies, whatever the effort.
        const r_chat = try renderChatTemplate(allocator, &user_only, &config, null, null, false, "high", false);
        defer allocator.free(r_chat);
        try testing.expect(std.mem.indexOf(u8, r_chat, "Reasoning Effort:") == null);
    }
    // Old-turn reasoning DROPS (the reference's drop_thinking): an assistant
    // turn with a LATER user message renders without its reasoning.
    {
        const hist = [_]Message{
            .{ .role = "user", .content = "Why is the sky blue?" },
            .{ .role = "assistant", .content = "Rayleigh scattering.", .reasoning_content = "Shorter wavelengths scatter more." },
            .{ .role = "user", .content = "And sunsets?" },
        };
        const rendered = try renderChatTemplate(allocator, &hist, &config, null, null, true, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "<｜Assistant｜></think>Rayleigh scattering.") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "Shorter wavelengths") == null);
        try testing.expect(std.mem.endsWith(u8, rendered, "<｜Assistant｜><think>"));
    }
    // No system message + tools: the reference delivers tools through a
    // system/developer turn and renders `system_content + "\n\n" + tools`, so
    // an EMPTY system turn still emits the two newlines (measured against
    // 0731's encoder — it silently DROPS tools attached to a user message, so
    // an empty system turn is its only way to say "tools, no system prompt").
    // Clients routinely send user+tools with no system message, and dropping
    // that separator shifts every token after the bos away from what the
    // model was trained on. Pinned byte-exact by tests/dsv4_template_ab.py.
    {
        const no_sys = [_]Message{
            .{ .role = "user", .content = "hi" },
        };
        const rendered = try renderChatTemplate(allocator, &no_sys, &config, tools_json, null, false, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.startsWith(u8, rendered, "<｜begin▁of▁sentence｜>\n\n## Tools"));
        try testing.expect(std.mem.indexOf(u8, rendered, "tool name and parameter schemas to invoke tool calls.\n<｜User｜>hi") != null);
    }
}

test "renderChatTemplate: a NUL byte in message content does not truncate the prompt" {
    // Bar: content after the NUL, and the assistant header, survive the render.
    const allocator = testing.allocator;
    const tpl =
        \\{%- for message in messages -%}
        \\<|im_start|>{{ message['role'] }}
        \\{{ message['content'] }}<|im_end|>
        \\{% endfor %}{%- if add_generation_prompt -%}<|im_start|>assistant
        \\{% endif %}
    ;
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const messages = [_]Message{
        .{ .role = "user", .content = "before\x00after" },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, null, null, false, null, false);
    defer allocator.free(rendered);

    try testing.expect(std.mem.indexOf(u8, rendered, "before") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "after") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "<|im_start|>assistant") != null);
}

const mid_system_test_template =
    \\{# tools 'tool' #}
    \\{%- for m in messages -%}{{ '<|im_start|>' + m.role + '\n' + (m.content or '') + '<|im_end|>\n' }}{%- endfor -%}
    \\{{ '<|im_start|>assistant\n' }}
;

test "mid-system: supported templates preserve late notes and stable history" {
    const allocator = testing.allocator;
    const config = ChatConfig{ .chat_template = mid_system_test_template, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = allocator };
    const calls = [_]ToolCall{.{ .id = "call_1", .name = "lookup", .arguments = "{}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "stable instructions" },
        .{ .role = "user", .content = "first question" },
        .{ .role = "system", .content = "<total_tokens>15000000 tokens left</total_tokens>" },
        .{ .role = "assistant", .content = "checking", .tool_calls = &calls },
        .{ .role = "tool", .content = "result", .tool_call_id = "call_1" },
        .{ .role = "system", .content = "<total_tokens>14999999 tokens left</total_tokens>" },
    };
    const before = try renderChatTemplate(allocator, messages[0..3], &config, "[]", null, true, null, false);
    defer allocator.free(before);
    const after = try renderChatTemplate(allocator, &messages, &config, "[]", null, true, null, false);
    defer allocator.free(after);
    try testing.expect(std.mem.startsWith(u8, after, before));
    try testing.expect(std.mem.indexOf(u8, after, "result<|im_end|>\n<|im_start|>system\n<total_tokens>14999999") != null);
}

test "mid-system: stock Qwen notes preserve tool rounds without rewriting history" {
    const a = testing.allocator;
    const templates = [_][]const u8{
        @embedFile("../../fixtures/qwen38_27b_chat_template.jinja"),
        @embedFile("../../fixtures/qwen38_chat_template.jinja"),
    };
    const tools = "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"description\":\"Read a value\",\"parameters\":{\"type\":\"object\"}}}]";
    const calls = [_]ToolCall{.{ .id = "call_1", .name = "lookup", .arguments = "{\"index\":3}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "stable instructions" },
        .{ .role = "user", .content = "first question" },
        .{ .role = "system", .content = "runtime note one" },
        .{ .role = "assistant", .content = "checking", .reasoning_content = "current reasoning", .tool_calls = &calls },
        .{ .role = "tool", .content = "first result", .tool_call_id = "call_1" },
        .{ .role = "system", .content = "runtime note two" },
        .{ .role = "system", .content = "runtime note three" },
        .{ .role = "assistant", .content = "answer", .reasoning_content = "more reasoning" },
    };
    for (templates) |tpl| {
        const config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
        const first = try renderChatTemplate(a, messages[0..3], &config, tools, null, true, "low", false);
        defer a.free(first);
        const second = try renderChatTemplate(a, messages[0..6], &config, tools, null, true, "low", false);
        defer a.free(second);
        const third = try renderChatTemplate(a, &messages, &config, tools, null, true, "low", false);
        defer a.free(third);
        try testing.expectEqualStrings(first, second[0..@min(first.len, second.len)]);
        const stable_second = second[0 .. second.len - "<|im_start|>assistant\n<think>\n".len];
        try testing.expectEqualStrings(stable_second, third[0..@min(stable_second.len, third.len)]);
        try testing.expect(std.mem.indexOf(u8, third, "<|im_start|>system\nruntime note one<|im_end|>\n<|im_start|>assistant") != null);
        try testing.expect(std.mem.indexOf(u8, third, "</tool_response><|im_end|>\n<|im_start|>system\nruntime note two") != null);
        try testing.expect(std.mem.indexOf(u8, third, "<|im_start|>system\nruntime note three<|im_end|>") != null);
        try testing.expect(std.mem.indexOf(u8, third, "current reasoning") != null);
        try testing.expect(std.mem.indexOf(u8, third, "<function=lookup>\n<parameter=index>\n3\n</parameter>") != null);
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, third, "<tools>"));
    }
}

test "mid-system: real Qwen pack loaded through ChatConfig preserves tool-round prefixes" {
    const dir = env.get(.QWEN_MID_SYSTEM_MODEL_DIR) orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try loadChatConfig(io, a, std.mem.span(dir));
    defer config.deinit();
    const tools = "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";
    const calls = [_]ToolCall{.{ .id = "call_1", .name = "lookup", .arguments = "{}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "stable instructions" },
        .{ .role = "user", .content = "first question" },
        .{ .role = "system", .content = "runtime note one" },
        .{ .role = "assistant", .content = "checking", .reasoning_content = "thought", .tool_calls = &calls },
        .{ .role = "tool", .content = "result", .tool_call_id = "call_1" },
        .{ .role = "system", .content = "runtime note two" },
    };
    const first = try renderChatTemplate(a, messages[0..3], &config, tools, null, true, "low", false);
    defer a.free(first);
    const next = try renderChatTemplate(a, &messages, &config, tools, null, true, "low", false);
    defer a.free(next);
    try testing.expect(std.mem.indexOf(u8, first, "first question<|im_end|>\n<|im_start|>system\nruntime note one<|im_end|>") != null);
    try testing.expect(std.mem.startsWith(u8, next, first));
    try testing.expect(std.mem.indexOf(u8, next, "</tool_response><|im_end|>\n<|im_start|>system\nruntime note two<|im_end|>") != null);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, next, "<tools>"));
}

test "mid-system: unknown Qwen revisions consolidate without rewriting the template" {
    const a = testing.allocator;
    const messages = [_]Message{
        .{ .role = "system", .content = "stable" },
        .{ .role = "user", .content = "question" },
        .{ .role = "system", .content = "late note" },
    };
    for ([_][]const u8{
        @embedFile("../../fixtures/qwen38_27b_chat_template.jinja"),
        @embedFile("../../fixtures/qwen38_chat_template.jinja"),
    }) |tpl| {
        const revised = try std.mem.concat(a, u8, &.{ tpl, "{{- 'revision marker' }}" });
        defer a.free(revised);
        const respelled = try std.mem.replaceOwned(u8, a, tpl, "message.role == \"system\"", "message.role == 'system'");
        defer a.free(respelled);
        for ([_][]const u8{ revised, respelled }) |unknown| {
            const config = ChatConfig{ .chat_template = unknown, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
            const rendered = try renderChatTemplate(a, &messages, &config, null, null, true, "low", false);
            defer a.free(rendered);
            try testing.expect(std.mem.indexOf(u8, rendered, "stable\n\nlate note<|im_end|>\n<|im_start|>user\nquestion") != null);
            if (unknown.ptr == revised.ptr) try testing.expect(std.mem.endsWith(u8, rendered, "revision marker"));
        }
    }
}

test "mid-system: stock Qwen adapter leaves unchanged turns and media native" {
    const a = testing.allocator;
    const tpl = @embedFile("../../fixtures/qwen38_27b_chat_template.jinja");
    const config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
    const images = [_]ImageData{.{ .pixels = "", .width = 1, .height = 1 }};
    const messages = [_]Message{
        .{ .role = "system", .content = "stable" },
        .{ .role = "user", .content = "question", .images = &images },
        .{ .role = "assistant", .content = "answer", .reasoning_content = "thought" },
        .{ .role = "tool", .content = "result one" },
        .{ .role = "system", .content = "late note" },
        .{ .role = "tool", .content = "result two", .images = &images },
    };
    const adapted = (try qwenLateSystemTemplate(a, tpl)).?;
    defer a.free(adapted);
    var native = config;
    native.chat_template = adapted;
    const original = try renderChatTemplate(a, messages[0..3], &config, null, null, true, "low", false);
    defer a.free(original);
    const equivalent = try renderChatTemplate(a, messages[0..3], &native, null, null, true, "low", false);
    defer a.free(equivalent);
    try testing.expectEqualStrings(original, equivalent);
    const rendered = try renderChatTemplate(a, &messages, &config, null, null, true, "low", false);
    defer a.free(rendered);
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, rendered, "<|vision_start|><|image_pad|><|vision_end|>"));
    try testing.expect(std.mem.indexOf(u8, rendered, "</tool_response><|im_end|>\n<|im_start|>system\nlate note<|im_end|>\n<|im_start|>user\n<tool_response>") != null);
    const media_note = [_]Message{ messages[0], messages[1], .{ .role = "system", .content = "invalid", .images = &images } };
    const extra = try serializeExtraContext(a, &config, true, "low");
    defer a.free(extra);
    try testing.expect((try renderJinja(a, &media_note, &config, null, extra, .null_literal)) == null);
}

test "mid-system: pinned Qwen extension preserves native tool reasoning and media bytes" {
    const a = testing.allocator;
    const images = [_]ImageData{.{ .pixels = "", .width = 1, .height = 1 }};
    const videos = [_]VideoData{.{ .pixels = "", .grid_t = 1, .grid_h = 1, .grid_w = 1 }};
    const calls = [_]ToolCall{.{ .id = "call_1", .name = "lookup", .arguments = "{\"index\":3}" }};
    const tools = "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";
    for ([_][]const u8{
        @embedFile("../../fixtures/qwen38_27b_chat_template.jinja"),
        @embedFile("../../fixtures/qwen38_chat_template.jinja"),
    }, 0..) |tpl, index| {
        const adapted = (try qwenLateSystemTemplate(a, tpl)).?;
        defer a.free(adapted);
        const messages = [_]Message{
            .{ .role = "system", .content = "stable" },
            .{ .role = "user", .content = "question", .images = if (index == 0) &images else &.{}, .videos = if (index == 0) &videos else &.{} },
            .{ .role = "assistant", .content = "checking", .reasoning_content = "thought", .tool_calls = &calls },
            .{ .role = "tool", .content = "result", .tool_call_id = "call_1" },
            .{ .role = "assistant", .content = "answer", .reasoning_content = "more thought" },
            .{ .role = "user", .content = "next question" },
        };
        for ([_]bool{ false, true }) |preserve| {
            const config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a, .chat_template_kwargs = if (preserve) "{\"preserve_thinking\":true}" else "{\"preserve_thinking\":false}" };
            var extended = config;
            extended.chat_template = adapted;
            for ([_]?[]const u8{ null, tools }) |tool_set| {
                for ([_][]const u8{ "low", "medium", "xhigh" }) |effort| {
                    const extra = try serializeExtraContext(a, &config, true, effort);
                    defer a.free(extra);
                    for ([_][]const Message{ &messages, messages[1..] }) |history| {
                        const native = (try renderJinja(a, history, &config, tool_set, extra, .null_literal)).?;
                        defer a.free(native);
                        const rendered = (try renderJinja(a, history, &extended, tool_set, extra, .null_literal)).?;
                        defer a.free(rendered);
                        try testing.expectEqualStrings(native, rendered);
                    }
                }
            }
        }
    }
}

test "mid-system: a tail-only template uses stable consolidation across turns" {
    const allocator = testing.allocator;
    const tpl =
        \\{% for m in messages %}{% if m.role == 'system' and not loop.first and not loop.last %}{{ raise_exception('system must be first or last') }}{% endif %}{{ m.role + ':' + (m.content or '') + ';' }}{% endfor %}
    ;
    const config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = allocator };
    const messages = [_]Message{
        .{ .role = "system", .content = "stable" },
        .{ .role = "user", .content = "question" },
        .{ .role = "system", .content = "late note" },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, null, null, false, null, false);
    defer allocator.free(rendered);
    try testing.expectEqualStrings("system:stable\n\nlate note;user:question;", rendered);
}

test "mid-system: a template dropping tail notes cannot preserve them" {
    const allocator = testing.allocator;
    const tpl =
        \\{% for m in messages %}{% if m.role != 'system' or loop.first or not loop.last %}{{ m.role + ':' + (m.content or '') + ';' }}{% endif %}{% endfor %}
    ;
    const config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = allocator };
    const messages = [_]Message{
        .{ .role = "system", .content = "stable" },
        .{ .role = "user", .content = "question" },
        .{ .role = "system", .content = "late note" },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, null, null, false, null, false);
    defer allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "late note") != null);
}

test "mid-system: generic ChatML fallback injects tools only once" {
    const allocator = testing.allocator;
    const config = ChatConfig{ .chat_template = "", .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = allocator };
    const messages = [_]Message{
        .{ .role = "system", .content = "stable" },
        .{ .role = "user", .content = "question" },
        .{ .role = "system", .content = "late note" },
    };
    const tools = "[{\"type\":\"function\",\"function\":{\"name\":\"unique_tool_marker\",\"parameters\":{}}}]";
    const rendered = try renderChatTemplate(allocator, &messages, &config, tools, null, false, null, false);
    defer allocator.free(rendered);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rendered, "unique_tool_marker"));
}

test "mid-system: unsupported templates fold without generic fallback or lost notes" {
    const allocator = testing.allocator;
    const templates = [_][]const u8{
        "{% for m in messages %}{% if m.role == 'system' and not loop.first %}{{ raise_exception('system must be first') }}{% endif %}{{ m.role + ':' + (m.content or '') + ';' }}{% endfor %}",
        "{% for m in messages %}{% if m.role != 'system' or loop.first %}{{ m.role + ':' + (m.content or '') + ';' }}{% endif %}{% endfor %}",
        "{% for m in messages %}{% if m.role == 'system' %}{{ 'system:' + m.content + ';' }}{% endif %}{% endfor %}{% for m in messages %}{% if m.role != 'system' %}{{ m.role + ':' + m.content + ';' }}{% endif %}{% endfor %}",
    };
    const messages = [_]Message{
        .{ .role = "system", .content = "stable" },
        .{ .role = "system", .content = "initial hook" },
        .{ .role = "user", .content = "question" },
        .{ .role = "system", .content = "late note" },
    };
    for (templates) |tpl| {
        const config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = allocator };
        const rendered = try renderChatTemplate(allocator, &messages, &config, null, null, false, null, false);
        defer allocator.free(rendered);
        try testing.expectEqualStrings("system:stable\n\ninitial hook\n\nlate note;user:question;", rendered);
    }
}

test "media renders one template placeholder per item, a tool image inside its tool response" {
    // Every vision template we serve must place a tool message's image: one it
    // drops is a named 400 on every later turn of an agent session.
    const allocator = testing.allocator;
    const Case = struct { tpl: []const u8, eos: []const u8, ph: []const u8, tool_image: []const u8 };
    const cases = [_]Case{
        .{
            .tpl = @embedFile("../../fixtures/qwen38_27b_chat_template.jinja"),
            .eos = "<|im_end|>",
            .ph = "<|vision_start|><|image_pad|><|vision_end|>",
            .tool_image = "<tool_response>\n<|vision_start|><|image_pad|><|vision_end|>Read p3.png\n</tool_response>",
        },
        .{ .tpl = @embedFile("../../fixtures/muse_chat_template.jinja"), .eos = "<|eot|>", .ph = "<|patch|>", .tool_image = "\n<|patch|>Read p3.png" },
        .{ .tpl = @embedFile("../../fixtures/lfm2_vl_chat_template.jinja"), .eos = "<|im_end|>", .ph = "<image>", .tool_image = "<image><tool_response>\nRead p3.png" },
    };
    const img = [_]ImageData{.{ .pixels = "", .width = 1, .height = 1 }};
    const two = [_]ImageData{ img[0], img[0] };
    const tc = [_]ToolCall{.{ .id = "c1", .name = "read", .arguments = "{\"path\":\"p3.png\"}" }};
    const messages = [_]Message{
        .{ .role = "user", .content = "page one", .images = &img },
        .{ .role = "assistant", .content = "seen" },
        .{ .role = "user", .content = "pages two", .images = &two },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        .{ .role = "tool", .content = "Read p3.png", .tool_call_id = "c1", .images = &img },
    };
    for (cases) |c| {
        var config = ChatConfig{ .chat_template = c.tpl, .bos_token = null, .eos_token = c.eos, .add_bos_token = false, .allocator = allocator };
        const rendered = try renderChatTemplate(allocator, &messages, &config, null, null, false, null, false);
        defer allocator.free(rendered);
        try testing.expectEqual(@as(usize, 4), std.mem.count(u8, rendered, c.ph));
        try testing.expect(std.mem.indexOf(u8, rendered, c.tool_image) != null);
    }
}

test "renderChatTemplate: Flint's template renders tools, tool history and reasoning byte-exact with jinja2" {
    // Reference bytes: the shipped template through jinja2 with HF's apply_chat_template settings
    // (trim_blocks, lstrip_blocks); the user text's `<think>` tags are stripped by the template itself.
    const cfg = ChatConfig{ .chat_template = @embedFile("../../fixtures/flint_chat_template.jinja"), .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = testing.allocator };
    const calls = [_]ToolCall{.{ .id = "c1", .name = "shell", .arguments = "{\"command\": \"ls -la\", \"timeout\": 5}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "You are a coding agent." },
        .{ .role = "user", .content = "List the files, then say <think>hi</think> back." },
        .{ .role = "assistant", .content = "", .tool_calls = &calls, .reasoning_content = "I should run ls." },
        .{ .role = "tool", .content = "a.txt\nb.txt", .tool_call_id = "c1" },
        .{ .role = "assistant", .content = "Two files.", .reasoning_content = "Done." },
        .{ .role = "user", .content = "Thanks" },
    };
    const tools =
        \\[{"type": "function", "function": {"name": "shell", "description": "Run a command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}}}]
    ;
    const out = try renderChatTemplate(testing.allocator, &messages, &cfg, tools, null, true, null, false);
    defer testing.allocator.free(out);
    try testing.expectEqualStrings(
        "<|system|>You are a coding agent.<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" ++
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{\"name\": \"shell\", \"description\": \"Run a command\", \"parameters\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\"}, \"timeout\": {\"type\": \"integer\"}}, \"required\": [\"command\"]}}</tools>\n\n" ++
            "For each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>\n" ++
            "<|user|>List the files, then say hi back.<|assistant|><think>I should run ls.</think><tool_call>shell<arg_key>command</arg_key><arg_value>ls -la</arg_value><arg_key>timeout</arg_key><arg_value>5</arg_value></tool_call>" ++
            "<|observation|><tool_response>a.txt\nb.txt</tool_response><|assistant|><think>Done.</think>Two files.<|user|>Thanks<|assistant|>",
        out,
    );
}

test "renderChatTemplate: a `\\xHH` escape is Python's code point U+00HH, so 0x80 and up render as UTF-8" {
    const cfg = ChatConfig{ .chat_template = "{{ messages[0].content }}{{ \"\\xe9|\\x41\" }}", .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = testing.allocator };
    const messages = [_]Message{.{ .role = "user", .content = "hi" }};
    const out = try renderChatTemplate(testing.allocator, &messages, &cfg, null, null, false, null, false);
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("hi\u{e9}|A", out);
}

test "renderChatTemplate: GLM-5-Next tool history renders natively (jinja `obj.0` is `obj[0]`)" {
    // The template probes `m.content.0.output`; jinja.cpp refused a numeric attribute and the
    // whole turn fell back to the generic format, which the model answered with `<end_of_turn>`.
    const cfg = ChatConfig{ .chat_template = @embedFile("../../fixtures/glm5_next_chat_template.jinja"), .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = testing.allocator };
    const calls = [_]ToolCall{.{ .id = "a", .name = "get_weather", .arguments = "{\"city\":\"Paris\"}" }};
    const messages = [_]Message{
        .{ .role = "user", .content = "Weather in Paris?" },
        .{ .role = "assistant", .content = "", .tool_calls = &calls },
        .{ .role = "tool", .content = "{\"temp_c\": 21}", .tool_call_id = "a" },
    };
    const out = try renderChatTemplate(testing.allocator, &messages, &cfg, null, null, true, null, false);
    defer testing.allocator.free(out);
    try testing.expect(std.mem.indexOf(u8, out, "<|observation|><tool_response>{\"temp_c\": 21}</tool_response><|assistant|><think>") != null);
    try testing.expect(std.mem.indexOf(u8, out, "<start_of_turn>") == null);
}

test "chat: the embedded V4.1 template renders through jinja.cpp byte-exact with DeepSeek's encoder" {
    // Cases and reference bytes from tests/dsv41_template_ab.py (the release's
    // encoding/encoding.py); a construct jinja.cpp renders differently than
    // jinja2 shows up here, never as a silent fallback in production.
    const a = testing.allocator;
    const tpl = @embedFile("../../fixtures/dsv41_chat_template.jinja");
    const parsed = try std.json.parseFromSlice(std.json.Value, a, @embedFile("../../fixtures/dsv41_template_cases.json"), .{});
    defer parsed.deinit();
    for (parsed.value.array.items) |case| {
        const o = case.object;
        const msgs = try std.json.Stringify.valueAlloc(a, o.get("messages").?, .{});
        defer a.free(msgs);
        const tools: ?[]u8 = if (o.get("tools").? == .null) null else try std.json.Stringify.valueAlloc(a, o.get("tools").?, .{});
        defer if (tools) |t| a.free(t);
        const extra = try std.fmt.allocPrint(a, "{{\"thinking_mode\":\"{s}\",\"reasoning_effort\":\"{s}\"}}", .{ o.get("thinking_mode").?.string, o.get("reasoning_effort").?.string });
        defer a.free(extra);
        const tpl_z = try a.dupeSentinel(u8, tpl, 0);
        defer a.free(tpl_z);
        const msgs_z = try a.dupeSentinel(u8, msgs, 0);
        defer a.free(msgs_z);
        const extra_z = try a.dupeSentinel(u8, extra, 0);
        defer a.free(extra_z);
        const tools_z = if (tools) |t| try a.dupeSentinel(u8, t, 0) else null;
        defer if (tools_z) |t| a.free(t);
        var len: usize = 0;
        const ptr = jinja_c.jinja_render_chat(tpl_z.ptr, msgs_z.ptr, if (tools_z) |t| t.ptr else null, extra_z.ptr, 1, &len) orelse {
            std.debug.print("{s}: {s}\n", .{ o.get("label").?.string, if (jinja_c.jinja_last_error()) |e| std.mem.span(e) else "render failed" });
            return error.RenderFailed;
        };
        defer jinja_c.jinja_str_free(ptr);
        testing.expectEqualStrings(o.get("expected").?.string, ptr[0..len]) catch |e| {
            std.debug.print("case: {s}\n", .{o.get("label").?.string});
            return e;
        };
    }
}

test "renderChatTemplate: REAL MiniCPM-V 4.7 chat_template.jinja renders without fallback (hermetic)" {
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = @embedFile("../../fixtures/minicpmv4_7_chat_template.jinja"),
        .bos_token = null,
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const tools_json =
        \\[{"type":"function","function":{"name":"get_time","description":"Get time","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}}]
    ;
    const tc = [_]ToolCall{.{ .id = "tc_0", .name = "get_time", .arguments = "{\"timezone\": \"UTC\"}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "What time is it?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        .{ .role = "tool", .content = "12:34 UTC", .tool_call_id = "tc_0" },
    };
    const on = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, null, false);
    defer allocator.free(on);
    try testing.expect(std.mem.indexOf(u8, on, "If you choose to call a function ONLY reply") != null);
    try testing.expect(std.mem.indexOf(u8, on, "<function=get_time>\n<parameter=timezone>\nUTC\n</parameter>\n</function>") != null);
    try testing.expect(std.mem.indexOf(u8, on, "<tool_response>\n12:34 UTC\n</tool_response>") != null);
    try testing.expect(std.mem.endsWith(u8, on, "<|im_start|>assistant\n<think>\n"));
    try testing.expect(promptTailOpensThink(on));

    const off = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false, null, false);
    defer allocator.free(off);
    try testing.expect(std.mem.endsWith(u8, off, "<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    try testing.expect(!promptTailOpensThink(off));
}
