//! Inkling channel tags and the channel split.

const std = @import("std");

const ThinkSplit = @import("think_tags.zig").ThinkSplit;
const MUSE_START_TAG = @import("muse.zig").MUSE_START_TAG;
const MUSE_MSG_TAG = @import("muse.zig").MUSE_MSG_TAG;
const MUSE_EOM_TAG = @import("muse.zig").MUSE_EOM_TAG;
const MUSE_EOT_TAG = @import("muse.zig").MUSE_EOT_TAG;

/// True when a rendered generation prompt ends inside a think block the
pub const INKLING_THINKING_TAG = "<|content_thinking|>";
pub const INKLING_TEXT_TAG = "<|content_text|>";
pub const INKLING_END_TAG = "<|end_message|>";
pub const INKLING_MODEL_TAG = "<|message_model|>";
pub const INKLING_INVOKE_TAG = "<|content_invoke_tool_json|>";

/// Control-channel marker TOKENS that must never appear in a visible stream
/// delta. Every entry is a single special token in its family's vocab, so
/// exact-match filtering at the delta level is complete — a marker can't be
/// split across deltas. Shared by every streaming surface's flush path.
pub fn isChannelMarkerToken(tt: []const u8) bool {
    return std.mem.eql(u8, tt, "<|channel>") or std.mem.eql(u8, tt, "<channel|>") or
        std.mem.eql(u8, tt, "<think>") or std.mem.eql(u8, tt, "</think>") or
        std.mem.eql(u8, tt, INKLING_TEXT_TAG) or std.mem.eql(u8, tt, INKLING_END_TAG) or
        std.mem.eql(u8, tt, INKLING_MODEL_TAG) or std.mem.eql(u8, tt, INKLING_THINKING_TAG) or
        std.mem.eql(u8, tt, MUSE_START_TAG) or std.mem.eql(u8, tt, MUSE_MSG_TAG) or
        std.mem.eql(u8, tt, MUSE_EOM_TAG) or std.mem.eql(u8, tt, MUSE_EOT_TAG);
}

/// Strip a leading `<|message_model|>` (and, on raw completions, a leading
/// `<|end_message|>` closing the prompt's implicit message) so the channel
/// marker is at position 0.
pub fn inklingStripMessageHead(text: []const u8) []const u8 {
    var t = std.mem.trimStart(u8, text, "\n ");
    if (std.mem.startsWith(u8, t, INKLING_END_TAG)) t = t[INKLING_END_TAG.len..];
    if (std.mem.startsWith(u8, t, INKLING_MODEL_TAG)) t = t[INKLING_MODEL_TAG.len..];
    return t;
}

/// Content runs to its `<|end_message|>` (anything after is a re-opened
/// message the stop token cut, or trailing markers — dropped).
fn inklingTrimContentTail(content_in: []const u8) []const u8 {
    var content = content_in;
    if (std.mem.indexOf(u8, content, INKLING_END_TAG)) |em| content = content[0..em];
    return std.mem.trim(u8, content, "\n ");
}

/// The Inkling arm of splitThinkBlock: null when the text carries no Inkling
/// channel markers (every other family falls through).
pub fn splitInklingChannels(text: []const u8) ?ThinkSplit {
    const t = inklingStripMessageHead(text);
    // A reply cut right after its message opener has no channel yet: nothing to deliver, never the marker.
    if (std.mem.trim(u8, t, "\n ").len == 0 and std.mem.trim(u8, text, "\n ").len > 0) return .{ .reasoning_content = null, .content = "" };
    if (std.mem.startsWith(u8, t, INKLING_THINKING_TAG)) {
        const body = t[INKLING_THINKING_TAG.len..];
        if (std.mem.indexOf(u8, body, INKLING_END_TAG)) |em| {
            const reasoning = std.mem.trim(u8, body[0..em], "\n ");
            var content = inklingStripMessageHead(body[em + INKLING_END_TAG.len ..]);
            if (std.mem.startsWith(u8, content, INKLING_TEXT_TAG)) content = content[INKLING_TEXT_TAG.len..];
            return .{
                .reasoning_content = if (reasoning.len > 0) reasoning else null,
                .content = inklingTrimContentTail(content),
            };
        }
        // Length-truncated mid-thought: reasoning, never content.
        const reasoning = std.mem.trim(u8, body, "\n ");
        return .{ .reasoning_content = if (reasoning.len > 0) reasoning else null, .content = "" };
    }
    if (std.mem.startsWith(u8, t, INKLING_TEXT_TAG)) {
        return .{ .reasoning_content = null, .content = inklingTrimContentTail(t[INKLING_TEXT_TAG.len..]) };
    }
    // A thinking-off prompt committed the text message (`contentChannelTail`): the answer has no
    // marker and ends at its `<|end_message|>`. A tool call message is left to the tool parser.
    if (std.mem.indexOf(u8, t, INKLING_END_TAG)) |em| {
        if (std.mem.indexOf(u8, t[0..em], INKLING_INVOKE_TAG) == null)
            return .{ .reasoning_content = null, .content = inklingTrimContentTail(t) };
    }
    return null;
}

// ── tests ──

const testing = std.testing;
const modelThinkOpener = @import("think_tags.zig").modelThinkOpener;
const splitThinkBlock = @import("split.zig").splitThinkBlock;

/// What the chat stream's plain arm (no tools, no prompt-opened block) does with each token before a
/// think block opens: the visible bytes it sends, and whether a model opener latched the block.
const PlainArm = struct { visible: []const u8, latched: bool };

fn replayPlainArm(buf: []u8, tokens: []const []const u8) PlainArm {
    var n: usize = 0;
    for (tokens) |tt| {
        if (n == 0 and modelThinkOpener(false, tt) != null) return .{ .visible = buf[0..n], .latched = true };
        if (isChannelMarkerToken(tt)) continue;
        @memcpy(buf[n..][0..tt.len], tt);
        n += tt.len;
    }
    return .{ .visible = buf[0..n], .latched = false };
}

test "an answer in a committed Inkling text message streams and splits as the same content" {
    var buf: [64]u8 = undefined;
    const answer = [_][]const u8{ "The answer", " is 4.", INKLING_END_TAG };
    const arm = replayPlainArm(&buf, &answer);
    try testing.expect(!arm.latched);
    try testing.expectEqualStrings("The answer is 4.", arm.visible);
    const split = splitThinkBlock("The answer is 4.<|end_message|>", true, false);
    try testing.expect(split.reasoning_content == null);
    try testing.expectEqualStrings("The answer is 4.", split.content);

    // Cut by max_tokens before its <|end_message|>.
    const cut = replayPlainArm(&buf, answer[0..2]);
    try testing.expectEqualStrings("The answer is 4.", cut.visible);
    try testing.expectEqualStrings("The answer is 4.", splitThinkBlock("The answer is 4.", true, false).content);

    // A later message the stop token did not end is not part of the answer.
    try testing.expectEqualStrings("4", splitThinkBlock("4<|end_message|><|message_model|><|content_thinking|>x", true, false).content);
}

test "an Inkling tool call message is left to the tool parser" {
    try testing.expect(splitInklingChannels("<|message_model|>get_time<|content_invoke_tool_json|>{\"args\":{},\"name\":\"get_time\"}<|end_message|>") == null);
    try testing.expect(splitInklingChannels("get_time<|content_invoke_tool_json|>{\"args\":{}}") == null);
}
