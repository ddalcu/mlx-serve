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
    return null;
}
