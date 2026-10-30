//! `splitThinkBlock`, `answerStopIndex`, and reasoning/content normalization.

const std = @import("std");
const testing = std.testing;

const lastUnclosedThinkOpen = @import("think_tags.zig").lastUnclosedThinkOpen;
const thinkOpenTagLenAt = @import("think_tags.zig").thinkOpenTagLenAt;
const indexOfThinkOpenTag = @import("think_tags.zig").indexOfThinkOpenTag;
const indexOfThinkCloseTag = @import("think_tags.zig").indexOfThinkCloseTag;
const trimLeakedToolMarkup = @import("../tools/markup_trim.zig").trimLeakedToolMarkup;
const trimTrailingThinkClosers = @import("think_tags.zig").trimTrailingThinkClosers;
const stripTrailingThinkOpen = @import("think_tags.zig").stripTrailingThinkOpen;
const ThinkSplit = @import("think_tags.zig").ThinkSplit;
const splitInklingChannels = @import("inkling.zig").splitInklingChannels;
const splitHarmonyChannels = @import("harmony.zig").splitHarmonyChannels;
const splitMuseChannels = @import("muse.zig").splitMuseChannels;
const streamContentLead = @import("stream_gate.zig").streamContentLead;

/// Split model output into reasoning_content and content.
/// Handles both `<think>...</think>` and Gemma 4's `<|channel>thought\n...<channel|>`.
/// `opened_by_template`: the generation prompt ended with a template-injected
/// think opener (see `promptTailOpensThink`), so output with no literal opener
/// and no close tag is still reasoning — generation started inside the block.
pub fn splitThinkBlock(text: []const u8, thinking: bool, opened_by_template: bool) ThinkSplit {
    const split = splitThinkBlockKeepingMarkup(text, thinking, opened_by_template);
    const reasoning: ?[]const u8 = if (split.reasoning_content) |r| blk: {
        const t = trimLeakedToolMarkup(r);
        break :blk if (t.len > 0) t else null;
    } else null;
    return .{ .reasoning_content = reasoning, .content = trimLeakedToolMarkup(split.content) };
}

/// First match of a client stop string at or after `from` that the split delivers as
/// answer content, never reasoning. Judged on the text up to the match only, so a
/// stream and its finished text cut at the same byte.
pub fn answerStopIndex(text: []const u8, from: usize, stop: []const u8, opened_by_template: bool) ?usize {
    var pos = from;
    while (std.mem.indexOfPos(u8, text, pos, stop)) |idx| : (pos = idx + 1) {
        const head = text[0 .. idx + stop.len];
        const c = splitThinkBlockKeepingMarkup(head, true, opened_by_template).content;
        const start = @intFromPtr(c.ptr) -% @intFromPtr(head.ptr);
        if (c.len == 0 or start > idx) continue;
        // The split trims trailing whitespace off the content; a match right after it is still answer.
        const end = start + c.len;
        if (idx <= end or std.mem.trim(u8, head[end..idx], " \t\r\n").len == 0) return idx;
    }
    return null;
}

/// The split WITHOUT the leaked-markup cut — for the one caller that feeds the
/// content back to `parseToolCalls` (the /v1/messages non-streaming path).
/// Every arm below returns raw slices; the cut is applied ONCE in the wrapper
/// above so a new arm cannot forget it.
pub fn splitThinkBlockKeepingMarkup(text: []const u8, thinking: bool, opened_by_template: bool) ThinkSplit {
    // gpt_oss / harmony channel segments (`analysis` / `final` /
    // `commentary to=functions.X`). Keyed on `<|channel|>`; must run BEFORE
    // the muse arm, which shares the `<|message|>` marker.
    if (splitHarmonyChannels(text)) |split| return split;
    // Muse-Glimmer channel segments (`to=self` / `to=user` / tool). Keyed on
    // the `<|message|>` marker, minus harmony's `<|channel|>`; runs regardless
    // of the `thinking` flag because the headers need stripping either way.
    if (splitMuseChannels(text)) |split| return split;
    // Inkling (inkling_mm_model) message channels: the model emits role-less
    // MESSAGES — `<|content_thinking|>R<|end_message|>` then
    // `<|message_model|><|content_text|>C<|end_message|>` — not a tag pair.
    // Keyed on channel openers no other family emits; runs regardless of the
    // `thinking` flag because the content marker needs stripping either way.
    if (splitInklingChannels(text)) |split| return split;
    // Gemma 4 style: <|channel>thought\n...<channel|>\n<|channel>\ncontent
    if (std.mem.indexOf(u8, text, "<channel|>")) |end| {
        const think_tag = "<|channel>thought\n";
        const reasoning_start: usize = if (std.mem.startsWith(u8, text, think_tag)) think_tag.len else if (std.mem.startsWith(u8, text, "<|channel>thought")) "<|channel>thought".len else 0;
        const reasoning = std.mem.trim(u8, text[reasoning_start..end], "\n ");
        var content = std.mem.trimStart(u8, text[end + 10 ..], "\n ");
        // Strip the content channel tag: <|channel>\n or <|channel>. A
        // re-opened THOUGHT channel (`<|channel>thought…`) is NOT a content
        // opener — stripping its `<|channel>` prefix here left a bare "thought"
        // in visible content ( live); leave it for stripTrailingThinkOpen.
        if (std.mem.startsWith(u8, content, "<|channel>\n")) {
            content = content[11..];
        } else if (std.mem.startsWith(u8, content, "<|channel>") and !std.mem.startsWith(u8, content, "<|channel>thought")) {
            content = content[10..];
        }
        content = std.mem.trimStart(u8, content, "\n ");
        return .{
            .reasoning_content = if (reasoning.len > 0) reasoning else null,
            .content = trimTrailingThinkClosers(stripTrailingThinkOpen(content)),
        };
    }
    // Standard style: <think>...</think> (or the Hy3-suffixed variant)
    if (indexOfThinkCloseTag(text, 0)) |close| {
        const reasoning_start: usize = thinkOpenTagLenAt(text) orelse 0;
        const reasoning = std.mem.trim(u8, text[reasoning_start..close.pos], "\n ");
        const content = std.mem.trimStart(u8, text[close.pos + close.len ..], "\n ");
        return .{
            .reasoning_content = if (reasoning.len > 0) reasoning else null,
            .content = trimTrailingThinkClosers(stripTrailingThinkOpen(content)),
        };
    }
    if (thinking) {
        // Unclosed think block: split policy depends on whether the model's
        // output begins with a literal opener.
        //   • Literal opener present → model definitely entered thinking but
        //     ran out of tokens / didn't close. Treat as reasoning.
        //   • No literal opener + template-injected opener → generation began
        //     INSIDE the block (Qwen 3.5/3.6 render `…assistant\n<think>\n`);
        //     an unclosed tail is truncated reasoning, never content.
        //   • No literal opener + no template opener → the model answered
        //     directly (Gemma style); keep the answer visible as content.
        if (thinkOpenTagLenAt(text) != null or std.mem.startsWith(u8, text, "<|channel>thought")) {
            const start: usize = if (thinkOpenTagLenAt(text)) |l| l else if (std.mem.startsWith(u8, text, "<|channel>thought\n")) "<|channel>thought\n".len else "<|channel>thought".len;
            const reasoning = std.mem.trimStart(u8, text[start..], "\n ");
            return .{ .reasoning_content = if (reasoning.len > 0) reasoning else null, .content = "" };
        }
        if (opened_by_template) {
            const reasoning = std.mem.trimStart(u8, text, "\n ");
            return .{ .reasoning_content = if (reasoning.len > 0) reasoning else null, .content = "" };
        }
        // Prose answer that ENDS by opening a new, unclosed thought block
        // (Gemma 12B tail behavior): the text before the opener is the
        // answer; the dangling thought is reasoning. The raw opener tag must
        // never leak into visible content.
        if (lastUnclosedThinkOpen(text)) |o| {
            const content = std.mem.trimEnd(u8, std.mem.trimStart(u8, text[0..o.pos], "\n "), "\n ");
            const reasoning = std.mem.trim(u8, text[o.after..], "\n ");
            return .{
                .reasoning_content = if (reasoning.len > 0) reasoning else null,
                .content = content,
            };
        }
        // No thought block, but the model may have emitted (or been truncated
        // right after) a dangling Gemma 4 *content* channel opener `<|channel>`
        // / `<|channel>\n`. Strip it so a cut-off reply never leaks the raw
        // control tag into visible content; whatever follows is the answer.
        var content = std.mem.trimStart(u8, text, "\n ");
        if (std.mem.startsWith(u8, content, "<|channel>\n")) {
            content = content[11..];
        } else if (std.mem.startsWith(u8, content, "<|channel>")) {
            content = content[10..];
        }
        return .{ .reasoning_content = null, .content = std.mem.trimStart(u8, content, "\n ") };
    }
    return .{ .reasoning_content = null, .content = text };
}

const ThinkOpen = struct { pos: usize, after: usize, is_think_style: bool };

/// Earliest think/thought opener at or after `from` (either tag family;
/// the think style covers both `<think>` and the Hy3-suffixed form).
pub fn nextThinkOpen(text: []const u8, from: usize) ?ThinkOpen {
    const chan = std.mem.indexOfPos(u8, text, from, "<|channel>thought");
    const think = indexOfThinkOpenTag(text, from);
    if (chan == null and think == null) return null;
    if (think == null or (chan != null and chan.? < think.?.pos)) {
        return .{ .pos = chan.?, .after = chan.? + "<|channel>thought".len, .is_think_style = false };
    }
    return .{ .pos = think.?.pos, .after = think.?.pos + think.?.len, .is_think_style = true };
}

/// Earliest close tag at or after `from` (either tag family).
fn nextThinkClose(text: []const u8, from: usize) ?ThinkOpen {
    const chan = std.mem.indexOfPos(u8, text, from, "<channel|>");
    const think = indexOfThinkCloseTag(text, from);
    if (chan == null and think == null) return null;
    if (think == null or (chan != null and chan.? < think.?.pos)) {
        return .{ .pos = chan.?, .after = chan.? + "<channel|>".len, .is_think_style = false };
    }
    return .{ .pos = think.?.pos, .after = think.?.pos + think.?.len, .is_think_style = true };
}

/// Skip an optional Gemma 4 CONTENT channel opener (`<|channel>` not followed
/// by `thought`) plus surrounding newlines, right after a thought close.
fn skipContentChannelTag(text: []const u8, start: usize) usize {
    var pos = start;
    while (pos < text.len and (text[pos] == '\n' or text[pos] == ' ')) pos += 1;
    const tag = "<|channel>";
    if (pos + tag.len <= text.len and std.mem.eql(u8, text[pos .. pos + tag.len], tag)) {
        const rest = text[pos + tag.len ..];
        if (!std.mem.startsWith(u8, rest, "thought")) {
            pos += tag.len;
            while (pos < text.len and (text[pos] == '\n' or text[pos] == ' ')) pos += 1;
        }
    }
    return pos;
}

/// Merge ALL closed think/thought blocks in `text` into one leading block.
///
/// The split/strip layer understands exactly one leading block plus an
/// optional UNCLOSED trailing opener. Gemma 4 12B, however, re-opens a thought
/// channel mid-turn and closes it again (`…content<|channel>thought\n…
/// <channel|>more content`) — observed live via Claude Code on /v1/messages,
/// where the raw pair leaked verbatim into the visible text block. This pass
/// rewrites such output to `<|channel>thought\n{all thought text}<channel|>
/// {all content}` so every downstream consumer (splitThinkBlock,
/// parseToolCalls) handles it unchanged.
///
/// Returns null when no rewrite is needed (zero or one LEADING closed block —
/// the overwhelmingly common case, zero-cost). An unclosed trailing opener is
/// left in place for the existing trailing-strip logic.
pub fn normalizeEmbeddedThinkBlocks(allocator: std.mem.Allocator, text: []const u8) !?[]u8 {
    var reasoning_parts = std.ArrayList([]const u8).empty;
    defer reasoning_parts.deinit(allocator);
    var content_parts = std.ArrayList([]const u8).empty;
    defer content_parts.deinit(allocator);

    var style_think = false;
    var style_set = false;
    var closed_blocks: usize = 0;
    var first_block_leading = false;
    var pos: usize = 0;

    // Template-opened leading block: a close tag BEFORE any opener (Qwen
    // renders `…assistant\n<think>\n` into the prompt, so output starts
    // mid-thought and contains only the close).
    if (nextThinkClose(text, 0)) |lc| {
        const open_pos = if (nextThinkOpen(text, 0)) |o| o.pos else text.len;
        if (lc.pos < open_pos) {
            try reasoning_parts.append(allocator, std.mem.trim(u8, text[0..lc.pos], "\n "));
            style_think = lc.is_think_style;
            style_set = true;
            closed_blocks += 1;
            first_block_leading = true;
            pos = skipContentChannelTag(text, lc.after);
        }
    }

    while (true) {
        const o = nextThinkOpen(text, pos) orelse {
            try content_parts.append(allocator, text[pos..]);
            break;
        };
        const close_tag: []const u8 = if (o.is_think_style) "</think>" else "<channel|>";
        const close_pos = std.mem.indexOfPos(u8, text, o.after, close_tag) orelse {
            // Unclosed trailing opener — leave verbatim for the trailing-strip
            // logic in splitThinkBlock.
            try content_parts.append(allocator, text[pos..]);
            break;
        };
        try content_parts.append(allocator, text[pos..o.pos]);
        try reasoning_parts.append(allocator, std.mem.trim(u8, text[o.after..close_pos], "\n "));
        if (!style_set) {
            style_think = o.is_think_style;
            style_set = true;
        }
        closed_blocks += 1;
        if (o.pos == 0) first_block_leading = true;
        pos = skipContentChannelTag(text, close_pos + close_tag.len);
    }

    // Rewrite only when a closed block exists beyond the single leading one.
    if (closed_blocks == 0) return null;
    if (closed_blocks == 1 and first_block_leading) return null;

    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, if (style_think) "<think>\n" else "<|channel>thought\n");
    var first = true;
    for (reasoning_parts.items) |r| {
        if (r.len == 0) continue;
        if (!first) try out.appendSlice(allocator, "\n\n");
        try out.appendSlice(allocator, r);
        first = false;
    }
    try out.appendSlice(allocator, if (style_think) "</think>\n" else "<channel|>\n");
    first = true;
    for (content_parts.items) |c| {
        const trimmed = std.mem.trim(u8, c, "\n ");
        if (trimmed.len == 0) continue;
        if (!first) try out.append(allocator, '\n');
        try out.appendSlice(allocator, trimmed);
        first = false;
    }
    return try out.toOwnedSlice(allocator);
}

/// Streaming-only: the head of a think block can no longer grow into any
/// opener the stream strips (`<…` markers, muse `assistant`/`to=` headers), so
/// a short first token is reasoning now instead of waiting for 7 bytes.
pub fn cannotOpenThink(buf: []const u8) bool {
    const rest = std.mem.trimStart(u8, buf, " \n");
    if (rest.len == 0 or rest[0] == '<') return false;
    for ([_][]const u8{ "assistant", "to=" }) |w| {
        if (std.mem.startsWith(u8, w, rest[0..@min(rest.len, w.len)])) return false;
    }
    return true;
}

// ── tests ──

test "split content: think tags never reach content" {
    try testing.expectEqualStrings("Hello", splitThinkBlock("<think>reasoning</think>Hello", true, false).content);
    try testing.expectEqualStrings("Hello", splitThinkBlock("<think>reasoning</think>\nHello", true, false).content);
    try testing.expectEqualStrings("Hello", splitThinkBlock("<think>reasoning</think>\n\nHello", true, false).content);
}

test "split content: empty for open think tag (truncated thought is never content)" {
    try testing.expectEqualStrings("", splitThinkBlock("<think>still thinking...", true, false).content);
}

test "split content: text without think tags passes through" {
    try testing.expectEqualStrings("Hello world", splitThinkBlock("Hello world", true, false).content);
}

test "splitThinkBlock with complete think block" {
    const result = splitThinkBlock("<think>reasoning here</think>answer here", false, false);
    try testing.expectEqualStrings("reasoning here", result.reasoning_content.?);
    try testing.expectEqualStrings("answer here", result.content);
}

test "splitThinkBlock with empty reasoning" {
    const result = splitThinkBlock("<think>\n\n</think>\n\nactual content", false, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("actual content", result.content);
}

test "splitThinkBlock thinking=true no close tag, literal opener present" {
    // Model entered thinking but ran out of tokens before closing.
    const result = splitThinkBlock("<think>partial reasoning", true, false);
    try testing.expectEqualStrings("partial reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "splitThinkBlock thinking=true no close tag, template-opened block" {
    // Qwen 3.6 truncated-thinking leak (regression): the chat template injects
    // `<think>\n` into the generation prompt, so the model's output starts
    // INSIDE the think block with no literal opener. If generation stops
    // (length) before `</think>`, every token so far is reasoning — it must
    // land in reasoning_content, never in content. Matches the streaming path.
    const result = splitThinkBlock("The user wants 17*23. Let me compute", true, true);
    try testing.expectEqualStrings("The user wants 17*23. Let me compute", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "splitThinkBlock thinking=true no close tag, no opener, template did NOT open" {
    // Gemma-style direct answer: thinking enabled but the template injects no
    // opener and the model answered without a thought channel. The answer must
    // stay visible as content.
    const result = splitThinkBlock("It is currently 8:15 AM PDT.", true, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("It is currently 8:15 AM PDT.", result.content);
}

test "splitThinkBlock template-opened block with close tag still splits" {
    // Normal Qwen 3.6 round: no literal opener (template-injected), close tag
    // present — reasoning before it, content after.
    const result = splitThinkBlock("compute 340+51=391</think>\n\n391.", true, true);
    try testing.expectEqualStrings("compute 340+51=391", result.reasoning_content.?);
    try testing.expectEqualStrings("391.", result.content);
}

test "splitThinkBlock thinking=false no tags" {
    const result = splitThinkBlock("just content", false, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("just content", result.content);
}

test "splitThinkBlock strips think prefix in thinking mode" {
    const result = splitThinkBlock("<think>my reasoning", true, false);
    try testing.expectEqualStrings("my reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "splitThinkBlock: Inkling message channels (thinking + text + truncation)" {
    // Real REAP25 captures: the model emits role-less MESSAGES —
    // a thinking message, then a fresh <|message_model|> text message — not a
    // tag pair. Channel markers must never leak into content or reasoning.
    {
        const out = "<|content_thinking|>Shorter wavelengths scatter more.<|end_message|><|message_model|><|content_text|>Rayleigh scattering.<|end_message|>";
        const split = splitThinkBlock(out, true, false);
        try testing.expectEqualStrings("Shorter wavelengths scatter more.", split.reasoning_content.?);
        try testing.expectEqualStrings("Rayleigh scattering.", split.content);
    }
    // Thinking-off shape (live capture: chat 2+2)
    {
        const out = "<|content_text|>4<|end_message|>";
        const split = splitThinkBlock(out, false, false);
        try testing.expect(split.reasoning_content == null);
        try testing.expectEqualStrings("4", split.content);
    }
    // Length-truncated mid-thought: reasoning, never content
    {
        const out = "<|content_thinking|>The user is asking for";
        const split = splitThinkBlock(out, true, false);
        try testing.expectEqualStrings("The user is asking for", split.reasoning_content.?);
        try testing.expectEqualStrings("", split.content);
    }
    // Raw-completion shape: leading <|end_message|><|message_model|> before
    // the first channel marker (no template context).
    {
        const out = "<|end_message|><|message_model|><|content_thinking|>Standard implementation.<|end_message|><|message_model|><|content_text|>def fib(n): ...<|end_message|>";
        const split = splitThinkBlock(out, true, false);
        try testing.expectEqualStrings("Standard implementation.", split.reasoning_content.?);
        try testing.expectEqualStrings("def fib(n): ...", split.content);
    }
}

test "cannotOpenThink decides a short think head only once no opener can follow" {
    try testing.expect(cannotOpenThink("The"));
    try testing.expect(cannotOpenThink("\nOk"));
    try testing.expect(!cannotOpenThink("<"));
    try testing.expect(!cannotOpenThink("<thi"));
    try testing.expect(!cannotOpenThink(" \n"));
    try testing.expect(!cannotOpenThink("to"));
    try testing.expect(!cannotOpenThink("assis"));
    try testing.expect(!cannotOpenThink(""));
}

test "splitThinkBlock with think block before tool call" {
    const text = "<think>I need to call the calculator</think>\n<tool_call>\n{\"name\": \"calc\", \"arguments\": {\"a\": 5}}\n</tool_call>";
    const result = splitThinkBlock(text, false, false);
    try testing.expectEqualStrings("I need to call the calculator", result.reasoning_content.?);
    // `content` is VISIBLE text, so the call block is cut: parseToolCalls runs
    // on the raw output, and whatever it leaves behind is markup the client
    // must never render.
    try testing.expectEqualStrings("", result.content);
    // The keeping-markup variant is what the one parse-from-content caller
    // (/v1/messages non-streaming) uses, and it still hands over the block.
    const raw = splitThinkBlockKeepingMarkup(text, false, false);
    try testing.expect(std.mem.startsWith(u8, raw.content, "<tool_call>"));
}

test "splitThinkBlock with think block before regular content" {
    const text = "<think>Let me think about this</think>\n\nThe answer is 42.";
    const result = splitThinkBlock(text, false, false);
    try testing.expectEqualStrings("Let me think about this", result.reasoning_content.?);
    try testing.expectEqualStrings("The answer is 42.", result.content);
}

test "splitThinkBlock with empty think block" {
    const text = "<think>\n\n</think>\n\nJust content here.";
    const result = splitThinkBlock(text, false, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("Just content here.", result.content);
}

test "splitThinkBlock no think tags with tool call" {
    const text = "<tool_call>\n{\"name\": \"search\", \"arguments\": {}}\n</tool_call>";
    const result = splitThinkBlock(text, false, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("", result.content);
    try testing.expectEqualStrings(text, splitThinkBlockKeepingMarkup(text, false, false).content);
}

test "split content: Gemma 4 channel tags never reach content" {
    try testing.expectEqualStrings("Hello", splitThinkBlock("<|channel>thought\nreasoning here<channel|>Hello", true, false).content);
    try testing.expectEqualStrings("Hello", splitThinkBlock("<|channel>thought\nreasoning<channel|>\nHello", true, false).content);
    try testing.expectEqualStrings("", splitThinkBlock("<|channel>thought\nstill thinking...", true, false).content);
}

test "splitThinkBlock Gemma 4 channel tags" {
    const result = splitThinkBlock("<|channel>thought\nmy reasoning<channel|>answer here", false, false);
    try testing.expectEqualStrings("my reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("answer here", result.content);
}

test "splitThinkBlock Gemma 4 thinking in progress" {
    const result = splitThinkBlock("<|channel>thought\npartial reasoning", true, false);
    try testing.expectEqualStrings("partial reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "splitThinkBlock truncated mid-thinking does not leak channel tag" {
    // Truncation regression: the model emitted the Gemma 4 *content* channel
    // opener (`<|channel>\n…`) directly — no thought block and no `<channel|>`
    // close — then hit the output cap. The raw `<|channel>` control tag must
    // never reach visible content.
    {
        const r = splitThinkBlock("<|channel>\nThe answer is 42.", true, false);
        try testing.expect(r.reasoning_content == null);
        try testing.expectEqualStrings("The answer is 42.", r.content);
    }
    // Bare dangling opener (cut off right after the tag) → nothing visible.
    {
        const r = splitThinkBlock("<|channel>", true, false);
        try testing.expectEqualStrings("", r.content);
    }
    {
        const r = splitThinkBlock("<|channel>\n", true, false);
        try testing.expectEqualStrings("", r.content);
    }
    // The template-injected-but-no-tags case must still pass through untouched.
    {
        const r = splitThinkBlock("It is currently 8:15 AM PDT.", true, false);
        try testing.expect(r.reasoning_content == null);
        try testing.expectEqualStrings("It is currently 8:15 AM PDT.", r.content);
    }
}

test "splitThinkBlock trailing unclosed thought opener does not leak (Gemma 12B pi regression)" {
    // Gemma 4 12B answers in prose, then opens a NEW thought channel right
    // before the turn ends (its known channel-thought tail behavior). The raw
    // `<|channel>thought` opener — and any unclosed thought text after it —
    // must never reach visible content; pi rendered the literal tag to users.
    {
        // Bare trailing opener, nothing after.
        const r = splitThinkBlock("Here is the design.\n<|channel>thought", true, false);
        try testing.expectEqualStrings("Here is the design.", r.content);
        try testing.expect(r.reasoning_content == null);
    }
    {
        // Trailing opener with unclosed thought text → thought is reasoning.
        const r = splitThinkBlock("Here is the design.\n<|channel>thought\nI should now write the file", true, false);
        try testing.expectEqualStrings("Here is the design.", r.content);
        try testing.expectEqualStrings("I should now write the file", r.reasoning_content.?);
    }
    {
        // Same shape for the <think> family.
        const r = splitThinkBlock("Done.\n<think>wait, maybe I", true, false);
        try testing.expectEqualStrings("Done.", r.content);
        try testing.expectEqualStrings("wait, maybe I", r.reasoning_content.?);
    }
}

test "split content: trailing unclosed thought opener does not leak" {
    // The visible text must be truncated at a trailing unclosed opener (the
    // tag and dangling thought are reasoning, never content).
    try testing.expectEqualStrings("Here is the design.", splitThinkBlock("Here is the design.\n<|channel>thought", true, false).content);
    try testing.expectEqualStrings("Here is the design.", splitThinkBlock("Here is the design.\n<|channel>thought\nI should now write", true, false).content);
    try testing.expectEqualStrings("Done.", splitThinkBlock("Done.\n<think>hmm", true, false).content);
}

test "in-thought tool markup never rides out as reasoning" {
    // Live (DeepSeek-V4-Flash-0731 agent session): the model
    // emitted a complete DSML call INSIDE its think block, closed the block,
    // then issued a different call in content. parseToolCalls works on the
    // post-think text, so the in-thought call is deliberately skipped — but
    // it rode out verbatim in reasoning_content, and the client round-tripped
    // it into every later prompt.
    const raw = "Let me check the directory.<｜DSML｜tool_calls>\n" ++
        "<｜DSML｜invoke name=\"listFiles\">\n" ++
        "<｜DSML｜parameter name=\"path\" string=\"true\">quake</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls></think>I'll build it.";
    const split = splitThinkBlock(raw, true, true);
    try testing.expectEqualStrings("Let me check the directory.", split.reasoning_content.?);
    try testing.expectEqualStrings("I'll build it.", split.content);

    // Reasoning that is NOTHING but markup reports no reasoning at all.
    const only_markup = splitThinkBlock("<｜DSML｜tool_calls>\n</｜DSML｜tool_calls></think>answer", true, true);
    try testing.expectEqual(@as(?[]const u8, null), only_markup.reasoning_content);
    try testing.expectEqualStrings("answer", only_markup.content);
}

test "splitThinkBlock re-opened thought opener right after close does not leak" {
    // live regression (gemma-4): the model closed its thought
    // channel and IMMEDIATELY re-opened a fresh one with nothing between, then
    // the turn ended. The leading-strip consumed the first closed block,
    // leaving the bare re-opened opener at pos 0 of the remainder — it must
    // still vanish, and `<|channel>thought` must never be mis-stripped as a
    // CONTENT channel (`<|channel>`), which leaked a glued "thought".
    {
        const r = splitThinkBlock("<|channel>thought\nLet me plan.<channel|>\n<|channel>thought\n", true, false);
        try testing.expectEqualStrings("", r.content);
        try testing.expectEqualStrings("Let me plan.", r.reasoning_content.?);
    }
    {
        // <think> family equivalent.
        const r = splitThinkBlock("<think>plan</think>\n<think>", true, false);
        try testing.expectEqualStrings("", r.content);
    }
    {
        // Content BETWEEN the close and the re-open must survive the cut.
        const r = splitThinkBlock("<|channel>thought\nPlan.<channel|>\nReady.<|channel>thought\n", true, false);
        try testing.expectEqualStrings("Ready.", r.content);
    }
}

test "split content: re-opened thought opener right after close does not leak" {
    // Same regression, content side — this is the exact form that reached
    // chat-history.json (`<|channel>thought\n` as the whole content).
    try testing.expectEqualStrings("", splitThinkBlock("<|channel>thought\nLet me plan.<channel|>\n<|channel>thought\n", true, false).content);
    try testing.expectEqualStrings("Ready.", splitThinkBlock("<|channel>thought\nPlan.<channel|>\nReady.<|channel>thought\n", true, false).content);
    try testing.expectEqualStrings("", splitThinkBlock("<think>plan</think>\n<think>", true, false).content);
}

test "split content: trailing <channel|> close-marker spam never leaks (degenerate model output)" {
    // Live soak capture (record 2151, a Gemma reasoning variant): the model emitted
    // reasoning, one <channel|> close, a scrap of content, then SPAMMED 16 more bare
    // <channel|> close markers. The leading cut takes the FIRST close, and the
    // trailing-open cut only handles unclosed OPENERS — so the 16 stray CLOSE markers
    // leaked into visible content. A close marker is never valid at the tail of
    // content; trailing closers (and openers) are trimmed so <channel|> can't leak.
    const text = "Reasoning here about the file.\n<channel|>`glob` `./game.js`\n\n" ++
        "<channel|><channel|><channel|><channel|><channel|>";
    const content = splitThinkBlock(text, true, false).content;
    try testing.expect(std.mem.indexOf(u8, content, "<channel|>") == null);
    // The scrap of real content before the spam survives.
    try testing.expect(std.mem.indexOf(u8, content, "glob") != null);
}

test "split content: trailing </think> close spam never leaks" {
    const text = "<think>plan</think>The answer.</think></think></think>";
    const content = splitThinkBlock(text, true, false).content;
    try testing.expect(std.mem.indexOf(u8, content, "</think>") == null);
    try testing.expect(std.mem.indexOf(u8, content, "The answer.") != null);
}

test "split content: orphan Gemma <tool_call|> close never leaks (no-tag-leak)" {
    // Live soak (gemma-4-26B-A4B-it-qat-4bit, tools present, a "no
    // tools needed" probe at temp 0.7): the model degenerated into a bare
    // 1-token <tool_call|> CLOSE with NO <|tool_call> opener, so parseToolCalls
    // found no call and the orphan control token leaked as the ENTIRE visible
    // content (server response content == "<tool_call|>"). A tool CLOSE marker is
    // never valid at the tail of content — same class as the <channel|> spam.
    // parseToolCalls already extracted any real call before this split runs, so
    // any residual <tool_call|> here is orphan by construction.
    try testing.expectEqualStrings("", splitThinkBlock("<tool_call|>", true, false).content);
    try testing.expectEqualStrings("Sure.", splitThinkBlock("Sure.<tool_call|>", true, false).content);
    // Legit prose (no control token) is untouched.
    try testing.expectEqualStrings("2 + 2 = 4", splitThinkBlock("2 + 2 = 4", true, false).content);
}

test "splitThinkBlock content never keeps trailing channel close spam" {
    const text = "<|channel>thought\nplan<channel|>\n<|channel>\nThe answer.<channel|><channel|><channel|>";
    const split = splitThinkBlock(text, true, false);
    try testing.expect(std.mem.indexOf(u8, split.content, "<channel|>") == null);
    try testing.expect(std.mem.indexOf(u8, split.content, "The answer.") != null);
}

// ── Hy3 (hy_v3 / Hunyuan 3) suffixed think tags + tag-format tool calls ──

test "hy3 think: splitThinkBlock splits on </think:opensource>" {
    const out = "I reason here.</think:opensource>The answer is 4.";
    const split = splitThinkBlock(out, true, true);
    try testing.expectEqualStrings("I reason here.", split.reasoning_content.?);
    try testing.expectEqualStrings("The answer is 4.", split.content);
}

test "hy3 think: literal suffixed opener + close" {
    const out = "<think:opensource>hmm</think:opensource>Answer.";
    const split = splitThinkBlock(out, true, false);
    try testing.expectEqualStrings("hmm", split.reasoning_content.?);
    try testing.expectEqualStrings("Answer.", split.content);
}

test "hy3 think: trailing suffixed close spam trimmed from content" {
    const out = "r</think:opensource>Answer.</think:opensource></think:opensource>";
    const split = splitThinkBlock(out, true, true);
    try testing.expectEqualStrings("Answer.", split.content);
}

test "hy3 think: dangling re-opened suffixed opener never leaks" {
    const out = "r</think:opensource>Answer.\n<think:opensource>\nnew thought";
    const split = splitThinkBlock(out, true, true);
    try testing.expectEqualStrings("Answer.", split.content);
    try testing.expect(std.mem.indexOf(u8, split.content, "<think") == null);
}

test "splitMuseChannels: to=self reasoning + to=user content, headers stripped" {
    // The exact shape muse emits after the prompt's `<|start|>assistant`.
    const text = " to=self<|message|>Let me think.<|eom|><|start|>assistant to=user<|message|>The answer is 4.";
    const split = splitThinkBlock(text, true, false);
    try testing.expectEqualStrings("Let me think.", split.reasoning_content.?);
    try testing.expectEqualStrings("The answer is 4.", split.content);

    // Direct answer: no reasoning segment at all.
    const direct = splitThinkBlock(" to=user<|message|>Hi there.", true, false);
    try testing.expect(direct.reasoning_content == null);
    try testing.expectEqualStrings("Hi there.", direct.content);

    // Length-truncated mid-thought: reasoning so far, never content.
    const cut = splitThinkBlock(" to=self<|message|>Half a thou", true, false);
    try testing.expectEqualStrings("Half a thou", cut.reasoning_content.?);
    try testing.expectEqualStrings("", cut.content);

    // A trailing partial header after <|eom|> is dropped, not content.
    const partial = splitThinkBlock(" to=self<|message|>R done.<|eom|><|start|>assistant to=us", true, false);
    try testing.expectEqualStrings("R done.", partial.reasoning_content.?);
    try testing.expectEqualStrings("", partial.content);

    // Non-muse text (no <|message|> marker) is never claimed.
    const plain = splitThinkBlock("just prose with to=user in it", true, false);
    try testing.expect(plain.reasoning_content == null);
    try testing.expectEqualStrings("just prose with to=user in it", plain.content);
}

test "splitMuseChannels: tool segment markup never rides out as content (wrapper cut)" {
    const text = " to=self<|message|>Need the tool.<|eom|><|start|>assistant to=weather.get<|message|><atem:function_calls>\n<atem:invoke name=\"weather.get\">\n<atem:parameter name=\"city\">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
    // Display split: the ATEM markup is CUT from content.
    const split = splitThinkBlock(text, true, false);
    try testing.expectEqualStrings("Need the tool.", split.reasoning_content.?);
    try testing.expectEqualStrings("", split.content);
    // KeepingMarkup split: the markup survives for the tool parser.
    const keeping = splitThinkBlockKeepingMarkup(text, true, false);
    try testing.expect(std.mem.indexOf(u8, keeping.content, "<atem:invoke") != null);
}

test "streaming content agrees with the non-streaming split on leading whitespace" {
    // LFM2.5 class: the template opens <think> unconditionally, so the first
    // generated token is the newline pair after the closer. The non-streaming
    // split drops it; the stream must reach the same bytes.
    const out = "\n\nOne, two, three.";
    const expected = splitThinkBlock(out, true, false).content;
    try testing.expectEqualStrings("One, two, three.", expected);

    const tokens = [_][]const u8{ "\n\n", "One", ", two", ", three." };
    var streamed: std.ArrayList(u8) = .empty;
    defer streamed.deinit(testing.allocator);
    var content_started = false;
    for (tokens) |tt| {
        const emit = streamContentLead(tt, content_started);
        if (emit.len == 0) continue;
        content_started = true;
        try streamed.appendSlice(testing.allocator, emit);
    }
    try testing.expectEqualStrings(expected, streamed.items);
}
test "answerStopIndex: a stop after trailing whitespace in the answer still cuts" {
    const t = std.testing;
    const text = "</think>\n\nHi \n\nmore";
    try t.expectEqual(@as(?usize, std.mem.indexOf(u8, text, "Hi ").? + 3), answerStopIndex(text, 0, "\n\n", true));
    const hard_break = "</think>\n\nline  \nnext";
    try t.expectEqual(@as(?usize, std.mem.indexOf(u8, hard_break, "  \n").? + 2), answerStopIndex(hard_break, 0, "\n", true));
    // Whitespace leading the answer is never delivered, so it never matches.
    try t.expectEqual(@as(?usize, null), answerStopIndex("</think>\n\nHi", 0, "\n\n", true));
}
