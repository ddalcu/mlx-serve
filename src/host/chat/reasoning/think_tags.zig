//! Think-tag scanning: openers, closers, partial tags, trailing closers, `ThinkSplit`, and the
//! prompt-tail classes (K2 openers, `noThinkTailSuffix`).

const std = @import("std");
const testing = std.testing;

const BARE_THINK_OPENER = @import("muse.zig").BARE_THINK_OPENER;
const splitThinkBlock = @import("split.zig").splitThinkBlock;
const nextThinkOpen = @import("split.zig").nextThinkOpen;

/// The FIRST unclosed think/thought opener: the earliest `<|channel>thought`
/// or `<think>` in `text` with no matching close tag anywhere after it.
/// Gemma 4 opens a NEW thought channel at the very end of a prose turn (its
/// channel-thought tail behavior) — and sometimes SEVERAL in a row, none of
/// them closed (seen live from the 26B GGUF via pi). Everything from the
/// first such opener onward is dangling thought; cutting at the LAST opener
/// instead leaks the earlier raw tags into visible content. A pos-0 opener is
/// reported too: callers (the split arms / stripTrailingThinkOpen) strip
/// their leading CLOSED block first, so an unclosed opener at the start of the
/// remainder is a genuine dangling re-open (`…<channel|>\n<|channel>thought\n`,
/// live) — excluding it leaked the bare opener into content.
const TrailingThinkOpen = struct { pos: usize, after: usize };
pub fn lastUnclosedThinkOpen(text: []const u8) ?TrailingThinkOpen {
    var from: usize = 0;
    while (nextThinkOpen(text, from)) |o| {
        const close = if (o.is_think_style)
            indexOfThinkCloseTag(text, o.after)
        else if (std.mem.indexOfPos(u8, text, o.after, "<channel|>")) |p|
            TagAt{ .pos = p, .len = "<channel|>".len }
        else
            null;
        if (close) |c| {
            // This block IS closed — keep scanning past its close.
            from = c.pos + c.len;
            continue;
        }
        return .{ .pos = o.pos, .after = o.after };
    }
    return null;
}

// ── Suffixed think/tool tag matching (Hy3 / Hunyuan 3) ──
//
// Hy3 templates suffix every control tag with a release marker:
// `<think:opensource>`, `</think:opensource>`, `<tool_call:opensource>`, …
// (HYTK in chat_template.jinja; future releases may carry a different
// suffix). The helpers below match `BASE>` or `BASE:[A-Za-z0-9_-]+>` so the
// shared think machinery handles both the canonical and suffixed families
// without hardcoding any specific suffix.

pub const TagAt = struct { pos: usize, len: usize };

fn tagSuffixChar(c: u8) bool {
    return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or
        (c >= '0' and c <= '9') or c == '_' or c == '-';
}

/// Length of a complete tag at the START of `text` whose base is `base`
/// (e.g. base "<think" matches "<think>" and "<think:opensource>").
pub fn suffixedTagLenAt(text: []const u8, comptime base: []const u8) ?usize {
    if (!std.mem.startsWith(u8, text, base)) return null;
    var i: usize = base.len;
    if (i < text.len and text[i] == '>') return i + 1;
    if (i >= text.len or text[i] != ':') return null;
    i += 1;
    const suffix_start = i;
    while (i < text.len) : (i += 1) {
        const c = text[i];
        if (c == '>') return if (i > suffix_start) i + 1 else null;
        if (!tagSuffixChar(c)) return null;
    }
    return null;
}

/// True when `text` could still GROW into a `base`-tag with more streamed
/// bytes (strict prefix: "<th", "<think", "<think:", "<think:opensou").
pub fn isPartialSuffixedTag(text: []const u8, comptime base: []const u8) bool {
    if (text.len == 0) return false;
    if (text.len <= base.len) return std.mem.startsWith(u8, base, text);
    if (!std.mem.startsWith(u8, text, base)) return false;
    if (text[base.len] != ':') return false;
    for (text[base.len + 1 ..]) |c| {
        if (!tagSuffixChar(c)) return false;
    }
    return true;
}

pub fn thinkOpenTagLenAt(text: []const u8) ?usize {
    return suffixedTagLenAt(text, "<think");
}

pub fn thinkCloseTagLenAt(text: []const u8) ?usize {
    return suffixedTagLenAt(text, "</think");
}

pub fn indexOfThinkOpenTag(text: []const u8, from: usize) ?TagAt {
    var i = from;
    while (std.mem.indexOfPos(u8, text, i, "<think")) |p| {
        if (thinkOpenTagLenAt(text[p..])) |l| return .{ .pos = p, .len = l };
        i = p + "<think".len;
    }
    return null;
}

pub fn indexOfThinkCloseTag(text: []const u8, from: usize) ?TagAt {
    var i = from;
    while (std.mem.indexOfPos(u8, text, i, "</think")) |p| {
        if (thinkCloseTagLenAt(text[p..])) |l| {
            if (!thinkCloseIsToolCallPayload(text, p)) return .{ .pos = p, .len = l };
        }
        i = p + "</think".len;
    }
    return null;
}

/// Whether a `</think>` at `pos` is an ARGUMENT VALUE rather than a block close.
///
/// The think split runs before tool parse, and this is the shared close-tag
/// scan every surface uses — so a model writing about its own prompt format
/// (`<arg_value>…closes a thought with </think>…</arg_value>`) had the split cut
/// straight through the call, losing it entirely and leaking the fragments.
/// Agent traffic writes files about prompts.
///
/// Payload ⟺ the nearest tool opener before `pos` is still OPEN there (no
/// `</tool_call` between it and `pos`) AND the block does close afterwards.
/// Both halves are load-bearing:
///   • without the first, a call the model emitted and CLOSED inside its
///     thought would make the real `</think>` after it look like payload;
///   • without the second, an unclosed opener inside a thought (the leaked-
///     markup case, which `trimLeakedToolMarkup` handles downstream) would
///     swallow the answer that follows the close.
fn thinkCloseIsToolCallPayload(text: []const u8, pos: usize) bool {
    // `streamThinkGate2` calls the close scan once per TOKEN on a growing
    // buffer, so the common case — text with no tool markup at all — must cost
    // one substring scan, not a walk per opener spelling. Both spellings share
    // this needle.
    if (std.mem.indexOf(u8, text[0..pos], "tool_call") == null) return false;
    // Nearest `<tool_call`-family opener before `pos`. Only this family is
    // considered: it is the one whose bodies carry free-form argument text.
    var open: ?usize = null;
    for ([_][]const u8{ "<tool_call", "<|tool_call" }) |m| {
        var i: usize = 0;
        while (std.mem.indexOfPos(u8, text, i, m)) |p| {
            if (p >= pos) break;
            if (open == null or p > open.?) open = p;
            i = p + 1;
        }
    }
    const o = open orelse return false;
    if (std.mem.indexOfPos(u8, text, o, "</tool_call")) |c| {
        if (c < pos) return false; // that call already closed — a real block close
        return true;
    }
    return false; // never closes — treat as leaked markup, not payload
}

/// Length of the trailing bytes of `buf` that could still GROW into a think
/// close tag with more streamed bytes — either family: `</think>`, the
/// Hy3-suffixed `</think:opensource>`, or Gemma's `<channel|>`. Streaming
/// reasoning flushes must hold this suffix back: flushing half a close tag
/// splits it across the flush boundary, so the retained remainder never
/// matches and the stream stays "inside thinking" forever (tag bytes leak
/// into reasoning_content). 0 = safe to flush everything.
pub fn partialThinkCloseSuffixLen(buf: []const u8) usize {
    // Any real close tag is well under 32 bytes; a longer '<'-run is prose.
    const window_start = if (buf.len > 32) buf.len - 32 else 0;
    var i = buf.len;
    while (i > window_start) {
        i -= 1;
        if (buf[i] != '<') continue;
        const tail = buf[i..];
        if (isPartialSuffixedTag(tail, "</think")) return tail.len;
        if (tail.len < "<channel|>".len and std.mem.startsWith(u8, "<channel|>", tail)) return tail.len;
    }
    return 0;
}

/// Tag length when `text` ENDS with a complete think-close tag.
fn endsWithThinkCloseTag(text: []const u8) ?usize {
    if (text.len == 0 or text[text.len - 1] != '>') return null;
    const lt = std.mem.lastIndexOfScalar(u8, text, '<') orelse return null;
    const l = thinkCloseTagLenAt(text[lt..]) orelse return null;
    return if (lt + l == text.len) l else null;
}

/// Tag length when `text` ENDS with a complete think-open tag.
fn endsWithThinkOpenTag(text: []const u8) ?usize {
    if (text.len == 0 or text[text.len - 1] != '>') return null;
    const lt = std.mem.lastIndexOfScalar(u8, text, '<') orelse return null;
    const l = thinkOpenTagLenAt(text[lt..]) orelse return null;
    return if (lt + l == text.len) l else null;
}

/// Trim trailing think/channel/tool CLOSE markers (and surrounding whitespace)
/// from visible content. A close marker is never valid at the tail of content —
/// but a degenerate model can spam them (live: a Gemma variant emitted 16 bare
/// `<channel|>` after its answer; the leading strip cuts the FIRST close, the
/// trailing-OPEN strip ignores closes, so they leaked). The Gemma tool CLOSE
/// `<tool_call|>` is here too: a degenerate 1-token bare close with no
/// `<|tool_call>` opener leaked as the entire content (live soak,
/// gemma-4-26B). parseToolCalls runs BEFORE this strip and extracts any real
/// call, so any residual `<tool_call|>` reaching here is orphan by construction.
/// Loops so a run of closers is fully removed. Returns a prefix slice (no alloc).
pub fn trimTrailingThinkClosers(content: []const u8) []const u8 {
    var s = content;
    while (true) {
        const t = std.mem.trimEnd(u8, s, "\n \t\r");
        if (std.mem.endsWith(u8, t, "<channel|>")) {
            s = t[0 .. t.len - "<channel|>".len];
        } else if (std.mem.endsWith(u8, t, "<tool_call|>")) {
            s = t[0 .. t.len - "<tool_call|>".len];
        } else if (endsWithThinkCloseTag(t)) |l| {
            s = t[0 .. t.len - l];
        } else {
            return t;
        }
    }
}

/// Truncate a trailing unclosed thought opener (and its dangling thought)
/// out of visible content. Used by the split arms after the leading block has
/// been handled.
pub fn stripTrailingThinkOpen(content: []const u8) []const u8 {
    if (lastUnclosedThinkOpen(content)) |o| {
        return std.mem.trimEnd(u8, content[0..o.pos], "\n ");
    }
    return content;
}

pub const ThinkSplit = struct {
    reasoning_content: ?[]const u8,
    content: []const u8,
};

/// A think block the MODEL opens when the template did not: a bare `<think>` token, or
/// Gemma 4's TWO tokens, the `<|channel>` marker (dropped from visible deltas) then the
/// word `thought`. `channel_armed` = the previous delta was that marker and nothing visible
/// has been emitted. Returns the opener to seed `think_buf` with.
pub fn modelThinkOpener(channel_armed: bool, token_text: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, std.mem.trimEnd(u8, token_text, "\n"), "<think>")) return "<think>";
    if (channel_armed and std.mem.eql(u8, token_text, "thought")) return "<|channel>thought";
    return null;
}

/// template opened (Qwen 3.5/3.6 style: `…assistant\n<think>\n`). Callers
/// decode the last few prompt tokens and pass the tail here. A thinking-off
/// render ends with a CLOSED `</think>` block and must not match.
pub fn promptTailOpensThink(tail: []const u8) bool {
    const trimmed = std.mem.trimEnd(u8, tail, "\n\r\t ");
    return endsWithThinkOpenTag(trimmed) != null or k2ThinkOpenerAt(trimmed) != null;
}

/// K2-Horizon's three RENDERED think openers (one per effort). The tokenizer
/// decodes them all to `<think>`, so generated text never carries them; only
/// the rendered prompt bytes do, and the closer must spell the same variant.
const k2_think_openers = [_][]const u8{ "<ifm|think>", "<ifm|think_fast>", "<ifm|think_faster>" };

fn k2ThinkOpenerAt(trimmed_tail: []const u8) ?[]const u8 {
    for (k2_think_openers) |opener| {
        if (std.mem.endsWith(u8, trimmed_tail, opener)) return opener;
    }
    return null;
}

/// Narrow form used by generation-side phase switches. A suffixed reasoning
/// family (for example Hy3's `<think:opensource>`) has a different close token
/// and must not be armed against Qwen's bare `</think>` boundary.
pub fn promptTailOpensBareThink(tail: []const u8) bool {
    const trimmed = std.mem.trimEnd(u8, tail, "\n\r\t ");
    return std.mem.endsWith(u8, trimmed, "<think>");
}

/// Which reasoning opener a rendered generation prompt ENDS with (trailing
/// whitespace ignored): a bare `<think>`, a complete suffixed `<think:S>`, or
/// nothing. Generation-side protocol resolution consumes this — the suffix is
/// carried into the close delimiter (`</think:S>`) so a close with a different
/// suffix is never treated as the boundary. The returned suffix borrows the
/// caller's buffer.
pub const ThinkTailClass = union(enum) {
    bare,
    suffixed: []const u8,
    none,
};

pub fn promptThinkTailClass(tail: []const u8) ThinkTailClass {
    const trimmed_tail = std.mem.trimEnd(u8, tail, "\n\r\t ");
    if (endsWithThinkOpenTag(trimmed_tail)) |l| {
        const tag = trimmed_tail[trimmed_tail.len - l ..];
        if (std.mem.eql(u8, tag, BARE_THINK_OPENER)) return .bare;
        return .{ .suffixed = tag["<think:".len .. tag.len - 1] };
    }
    return .none;
}

/// Suffix that COMMITS the no-think channel in the rendered prompt, or null.
/// Thinking-off on an always-thinking template otherwise means generating the
/// whole reasoning pass and discarding it — the prompt-side commit is what
/// makes thinking-off a real skip.
///   • Muse: the generation prompt ends bare `<|start|>assistant`; spending
///     the header (` to=user<|message|>`, byte-exactly what the template
///     renders for history content turns) makes `to=self` unreachable. Never
///     with tools — a tool call is a `to=<fn>` header.
///   • Unconditional think opener (LFM2.5 class): close the block the
///     template opened. Thinking-off Qwen-style renders end CLOSED and never
///     match; no tools gate — tool calls ride outside the think block.
pub fn noThinkTailSuffix(tpl: []const u8, rendered: []const u8, enable_thinking: bool, has_tools: bool) ?[]const u8 {
    if (enable_thinking) return null;
    const tail = contentChannelTail(tpl, rendered, !has_tools);
    return if (tail.len > 0) tail else null;
}

/// What must be appended to a rendered prompt so generation resumes in the
/// assistant's VISIBLE CONTENT channel rather than inside a reasoning block.
///
/// Two callers, one rule, deliberately: thinking-off enforces the content
/// channel so no reasoning pass runs, and a CONTINUATION needs the same thing
/// for a different reason — the partial text is prose the user has already
/// read, so a prompt that lands inside a think block would return the rest of
/// their answer as reasoning. Written twice, the muse and LFM2.5 cases would
/// drift apart the first time a third family joined.
///
/// `allow_channel_commit` is thinking-off's extra gate: with tools present it
/// declines to commit muse's `to=user`, since that would forbid the tool call
/// the model may be about to make. A continuation always passes true — it is
/// resuming prose by construction.
pub fn contentChannelTail(tpl: []const u8, rendered: []const u8, allow_channel_commit: bool) []const u8 {
    if (allow_channel_commit and std.mem.indexOf(u8, tpl, "reasoning_strength") != null and
        std.mem.endsWith(u8, rendered, "<|start|>assistant"))
    {
        return " to=user<|message|>";
    }
    const tail = std.mem.trimEnd(u8, rendered[rendered.len -| 64 ..], "\n\r\t ");
    if (k2ThinkOpenerAt(tail)) |opener| {
        inline for (k2_think_openers) |o| {
            if (std.mem.eql(u8, opener, o)) return "</" ++ o[1..];
        }
    }
    if (promptTailOpensThink(tail)) return "</think>";
    return "";
}

// ── tests ──

test "partialThinkCloseSuffixLen holds back growing close tags, ignores prose" {
    try testing.expectEqual(@as(usize, 0), partialThinkCloseSuffixLen("plain reasoning text"));
    try testing.expectEqual(@as(usize, 7), partialThinkCloseSuffixLen("thinking…</think"));
    try testing.expectEqual(@as(usize, 15), partialThinkCloseSuffixLen("thinking…</think:opensou"));
    try testing.expectEqual(@as(usize, 5), partialThinkCloseSuffixLen("thinking…<chan"));
    // A COMPLETE tag is not a partial — the scan-side indexOf owns it.
    try testing.expectEqual(@as(usize, 0), partialThinkCloseSuffixLen("done</think:opensource>"));
    // A '<' too far back is prose, not a growing tag.
    try testing.expectEqual(@as(usize, 0), partialThinkCloseSuffixLen("a < b and then lots of ordinary words follow here"));
}

test "indexOfThinkCloseTag: a close inside an OPEN tool call is argument payload, not a block close" {
    // Agent traffic writes files about prompts, and the think split runs
    // BEFORE the tool parse — so this cut used to destroy the whole call.
    const payload = "<tool_call>write\n<arg_key>content</arg_key>\n" ++
        "<arg_value>a thought ends with </think> here</arg_value>\n</tool_call>";
    try testing.expect(indexOfThinkCloseTag(payload, 0) == null);

    // A call the model emitted and CLOSED inside its thought: the `</think>`
    // after it is a REAL close and the answer stays content.
    const in_thought = "<think>let me try <tool_call>read</tool_call> hmm</think>The answer is 4.";
    const c = indexOfThinkCloseTag(in_thought, 0) orelse return error.MissingClose;
    try testing.expectEqualStrings("The answer is 4.", in_thought[c.pos + c.len ..]);
    try testing.expectEqualStrings("The answer is 4.", splitThinkBlock(in_thought, true, false).content);

    // An opener that NEVER closes is leaked markup, not a payload container:
    // the close still ends the block, and trimLeakedToolMarkup takes the tail.
    const unclosed = "<think>starting <tool_call>partial</think>Answer.";
    try testing.expect(indexOfThinkCloseTag(unclosed, 0) != null);

    // Ordinary shapes are untouched.
    try testing.expect(indexOfThinkCloseTag("<think>r</think>a", 0).?.pos == 8);
    try testing.expect(indexOfThinkCloseTag("<think>r</think>a<tool_call>x</tool_call>", 0).?.pos == 8);
}

test "promptTailOpensThink detects template-injected opener" {
    // Qwen 3.6 generation prompt tail with thinking on
    try testing.expect(promptTailOpensThink("<|im_start|>assistant\n<think>\n"));
    try testing.expect(promptTailOpensThink("<|im_start|>assistant\n<think>"));
    // Thinking off renders a CLOSED empty block — must not match
    try testing.expect(!promptTailOpensThink("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    // Gemma 4 prompt tail (no injected opener)
    try testing.expect(!promptTailOpensThink("<|turn>model\n"));
    try testing.expect(!promptTailOpensThink(""));
}

test "promptThinkTailClass resolves bare, suffixed, and plain tails" {
    try testing.expectEqual(ThinkTailClass.bare, promptThinkTailClass("<|im_start|>assistant\n<think>\n"));
    const s = promptThinkTailClass("<think:opensource>");
    try testing.expectEqualStrings("opensource", s.suffixed);
    try testing.expectEqual(ThinkTailClass.none, promptThinkTailClass("<|im_start|>assistant\n"));
    try testing.expectEqual(ThinkTailClass.none, promptThinkTailClass("<|start|>assistant"));
    try testing.expectEqual(ThinkTailClass.none, promptThinkTailClass(""));
    // A PARTIAL opener is not an opener.
    try testing.expectEqual(ThinkTailClass.none, promptThinkTailClass("<think:open"));
}

test "promptTailOpensThink does not match Gemma's channel opener" {
    // The containment boundary for the prompt-opened-think stream arm: it
    // detects the close with `</think>` ONLY, which is sound exactly because
    // the flag can only be set by a `<think`-family opener. Gemma 4 opens with
    // `<|channel>thought` and closes with `<channel|>`, so it never enters
    // that arm and its streaming is byte-unchanged.
    //
    // If `thinkOpenTagLenAt` ever grows the channel opener, this test fails —
    // and `server.zig`'s arm then needs `<channel|>` in its close check, or a
    // Gemma turn would be held forever waiting for a tag that never arrives.
    try testing.expect(!promptTailOpensThink("<|turn>model\n<|channel>thought\n"));
    try testing.expect(!promptTailOpensThink("<|turn>model\n"));
    // The `<think>` family it DOES match — bare and suffixed (hy3).
    try testing.expect(promptTailOpensThink("<|im_start|>assistant\n<think>"));
    try testing.expect(promptTailOpensThink("<|im_start|>assistant\n<think:opensource>"));
}

test "hy3 think: promptTailOpensThink recognizes the suffixed opener, not a closed no_think tail" {
    try testing.expect(promptTailOpensThink("<\xEF\xBD\x9Chy_Assistant:opensource\xEF\xBD\x9C><think:opensource>"));
    try testing.expect(!promptTailOpensThink("<\xEF\xBD\x9Chy_Assistant:opensource\xEF\xBD\x9C><think:opensource></think:opensource>"));
    try testing.expect(!promptTailOpensBareThink("<\xEF\xBD\x9Chy_Assistant:opensource\xEF\xBD\x9C><think:opensource>"));
    try testing.expect(promptTailOpensBareThink("assistant\n<think>\n"));
}

test "a model-opened think block latches from its own opener token" {
    // Both openers are dropped from visible deltas as channel markers, so without
    // this latch the whole thought streams as the answer (LFM2.5-8B-A1B, Gemma 4).
    try std.testing.expectEqualStrings("<think>", modelThinkOpener(false, "<think>").?);
    try std.testing.expectEqualStrings("<think>", modelThinkOpener(false, "<think>\n").?);
    try std.testing.expectEqualStrings("<|channel>thought", modelThinkOpener(true, "thought").?);
    try std.testing.expect(modelThinkOpener(false, "thought") == null);
    // Prose that merely BEGINS with the word is not the opener token.
    try std.testing.expect(modelThinkOpener(true, "thoughts") == null);
    try std.testing.expect(modelThinkOpener(true, "\n") == null);
    try std.testing.expect(modelThinkOpener(false, "<think>x") == null);
}

test "K2-Horizon: the rendered prompt opens one of three think tags; thinking-off closes the SAME one" {
    try testing.expect(promptTailOpensThink("<|ifm|im_start|>assistant\n<ifm|think>\n"));
    try testing.expect(promptTailOpensThink("<|ifm|im_start|>assistant\n<ifm|think_fast>\n"));
    try testing.expect(promptTailOpensThink("<|ifm|im_start|>assistant\n<ifm|think_faster>\n"));
    try testing.expect(!promptTailOpensThink("<|ifm|im_start|>assistant\n<ifm|thinker>\n"));
    try testing.expectEqualStrings("</ifm|think_fast>", contentChannelTail("", "<|ifm|im_start|>assistant\n<ifm|think_fast>\n", true));
    try testing.expectEqualStrings("</ifm|think>", contentChannelTail("", "x<ifm|think>\n", true));
    try testing.expectEqualStrings("</think>", contentChannelTail("", "x<think>\n", true));
}
