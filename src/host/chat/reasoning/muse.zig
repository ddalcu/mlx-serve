//! Muse channel tags, segments, stream verdicts, and the header skip.

const std = @import("std");
const testing = std.testing;

const ThinkSplit = @import("think_tags.zig").ThinkSplit;
const HARMONY_CHANNEL_TAG = @import("harmony.zig").HARMONY_CHANNEL_TAG;
const StreamThinkGate = @import("stream_gate.zig").StreamThinkGate;

// ── Muse-Glimmer (muse_glimmer) channel format ──────────────────────────
//
// The model emits harmony-style SEGMENTS after the prompt's bare
// `<|start|>assistant`:
//   ` to=self<|message|>REASONING<|eom|>` then
//   `<|start|>assistant to=user<|message|>ANSWER<|eot|>` — or a tool segment
//   `<|start|>assistant to=<fn><|message|><atem:function_calls>…<|eot|>`.
// <|start|>/<|message|>/<|eom|>/<|eot|> are single special tokens (ids
// 200022/200023/200007/200008); the `assistant to=<recipient>` header BETWEEN
// them is ORDINARY text, which is what makes an unresolved header something
// every streaming path must hold rather than flush.
pub const MUSE_START_TAG = "<|start|>";
pub const MUSE_MSG_TAG = "<|message|>";
pub const MUSE_EOM_TAG = "<|eom|>";
pub const MUSE_EOT_TAG = "<|eot|>";

pub const BARE_THINK_OPENER = "<think>";
pub const BARE_THINK_CLOSER = "</think>";

pub fn museIsRecipientChar(c: u8) bool {
    return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or
        (c >= '0' and c <= '9') or c == '_' or c == '-' or c == '.';
}

const MuseSegment = struct { recipient: []const u8, body: []const u8 };

/// Recipient + body of the segment in text[seg_start..seg_end] (seg_start
/// sits on a `<|start|>` marker, or 0 for the headerless first segment).
/// null when the header never resolved — a growing or truncated header is
/// never content.
fn museSegmentAt(text: []const u8, seg_start: usize, seg_end: usize) ?MuseSegment {
    const seg = text[seg_start..seg_end];
    const msg_rel = std.mem.indexOf(u8, seg, MUSE_MSG_TAG) orelse return null;
    const header = seg[0..msg_rel];
    const recipient: []const u8 = if (std.mem.indexOf(u8, header, "to=")) |t|
        std.mem.trim(u8, header[t + 3 ..], " \n")
    else
        "";
    var body = seg[msg_rel + MUSE_MSG_TAG.len ..];
    if (std.mem.indexOf(u8, body, MUSE_EOM_TAG)) |e| body = body[0..e];
    if (std.mem.indexOf(u8, body, MUSE_EOT_TAG)) |e| body = body[0..e];
    return .{ .recipient = recipient, .body = body };
}

/// Muse channel split. Claims the text only when `<|message|>` is present —
/// no other family emits it. First segment of each channel wins (the template
/// renders one per turn). A tool segment's body (ATEM markup) rides out as
/// content ONLY when no user segment exists, so the KeepingMarkup callers can
/// still see it and the trim wrapper cuts it from display.
pub fn splitMuseChannels(text: []const u8) ?ThinkSplit {
    if (std.mem.indexOf(u8, text, MUSE_MSG_TAG) == null) return null;
    // `<|message|>` is NOT unique to muse: harmony (gpt_oss), muse's own
    // format ancestor, emits the same marker. Decline harmony traffic here —
    // its `analysis` header carries no `to=`, so muse's recipient rules would
    // route the model's private reasoning into user-visible content.
    if (std.mem.indexOf(u8, text, HARMONY_CHANNEL_TAG) != null) return null;
    var reasoning: ?[]const u8 = null;
    var content: ?[]const u8 = null;
    var tool_body: ?[]const u8 = null;
    var routed = false;

    var seg_start: usize = 0;
    while (true) {
        const next = std.mem.indexOfPos(u8, text, seg_start + 1, MUSE_START_TAG);
        const seg_end = next orelse text.len;
        if (museSegmentAt(text, seg_start, seg_end)) |s| {
            routed = true;
            if (std.mem.eql(u8, s.recipient, "self")) {
                if (reasoning == null) {
                    const r = std.mem.trim(u8, s.body, "\n ");
                    if (r.len > 0) reasoning = r;
                }
            } else if (s.recipient.len == 0 or std.mem.eql(u8, s.recipient, "user")) {
                if (content == null) content = std.mem.trim(u8, s.body, "\n ");
            } else {
                if (tool_body == null) tool_body = s.body;
            }
        } else if (seg_start == 0) {
            // Headerless first segment: a prompt-committed `to=user<|message|>`
            // header puts the model's first bytes straight into BODY. Claim it
            // as content — but never text that still reads as a (truncated)
            // recipient header.
            var body = text[0..seg_end];
            if (std.mem.indexOf(u8, body, MUSE_EOM_TAG)) |e| body = body[0..e];
            if (std.mem.indexOf(u8, body, MUSE_EOT_TAG)) |e| body = body[0..e];
            const c = std.mem.trim(u8, body, "\n ");
            if (c.len > 0 and museThinkOpenerAt(body) == .not_muse) {
                routed = true;
                content = c;
            }
        }
        seg_start = next orelse break;
    }
    if (!routed) return null;
    return .{
        .reasoning_content = reasoning,
        .content = content orelse (tool_body orelse ""),
    };
}

/// Muse-Glimmer streaming verdict, decided from the channel markers alone.
/// null = not muse traffic (fall through to the generic think gate). The
/// recipient header between <|start|> and <|message|> is ORDINARY text, so an
/// unresolved header always HOLDS — flushing it leaks ` to=user` fragments.
pub fn museStreamVerdict(buf: []const u8) ?StreamThinkGate {
    const has_marker = std.mem.indexOf(u8, buf, MUSE_MSG_TAG) != null or
        std.mem.indexOf(u8, buf, MUSE_START_TAG) != null or
        std.mem.indexOf(u8, buf, MUSE_EOM_TAG) != null;
    if (!has_marker) {
        // A growing bare header (` to=sel`) carries no marker yet: claim it
        // only while every byte fits the recipient grammar, so prose that
        // merely starts with "to=" is released to the generic gate. The
        // first tokens can be SHORTER than "to=" (" to" leaked live
        // ) — a strict prefix of it still holds.
        const t = std.mem.trimStart(u8, buf, " \n");
        if (t.len < "to=".len) {
            return if (t.len > 0 and std.mem.startsWith(u8, "to=", t)) .hold_thinking else null;
        }
        if (!std.mem.startsWith(u8, t, "to=")) return null;
        for (t["to=".len..]) |c| {
            if (!museIsRecipientChar(c)) return null;
        }
        return .hold_thinking;
    }
    const seg_start = std.mem.lastIndexOf(u8, buf, MUSE_START_TAG) orelse 0;
    const seg = buf[seg_start..];
    const msg_rel = std.mem.indexOf(u8, seg, MUSE_MSG_TAG) orelse return .hold_thinking;
    const header = seg[0..msg_rel];
    const recipient: []const u8 = if (std.mem.indexOf(u8, header, "to=")) |t|
        std.mem.trim(u8, header[t + 3 ..], " \n")
    else
        "";
    if (std.mem.eql(u8, recipient, "self")) {
        const closed = std.mem.indexOf(u8, seg, MUSE_EOM_TAG) != null or
            std.mem.indexOf(u8, seg, MUSE_EOT_TAG) != null;
        return if (closed) .split_think else .hold_thinking;
    }
    if (recipient.len == 0 or std.mem.eql(u8, recipient, "user")) return .split_think;
    // Tool recipient: the tools path buffers before this gate is consulted;
    // hold defensively on a no-tools request rather than leak ATEM markup.
    return .hold_thinking;
}

/// Muse streaming tool hold: true while a segment header is unresolved (the
/// next token decides whether it names a tool) or resolved to a TOOL
/// recipient (the body is ATEM payload — parse runs on the end flush).
pub fn museHeaderHoldsForTools(buf: []const u8) bool {
    var header_from: usize = 0;
    if (std.mem.lastIndexOf(u8, buf, MUSE_START_TAG)) |p| {
        header_from = p + MUSE_START_TAG.len;
    } else {
        const lead = std.mem.trimStart(u8, buf, " \n");
        // A strict prefix of "to=" could still become a header (" to" leaked
        // as a text delta live) — hold it.
        if (lead.len < "to=".len) return lead.len > 0 and std.mem.startsWith(u8, "to=", lead);
        if (!std.mem.startsWith(u8, lead, "to=")) return false;
        header_from = buf.len - lead.len;
    }
    const seg = buf[header_from..];
    if (std.mem.indexOf(u8, seg, MUSE_MSG_TAG)) |msg_rel| {
        const header = seg[0..msg_rel];
        const recipient: []const u8 = if (std.mem.indexOf(u8, header, "to=")) |t|
            std.mem.trim(u8, header[t + 3 ..], " \n")
        else
            "";
        if (recipient.len == 0) return false;
        if (std.mem.eql(u8, recipient, "self") or std.mem.eql(u8, recipient, "user")) return false;
        return true;
    }
    // Unresolved: hold only while the bytes still fit `assistant to=<r>`.
    var rest = std.mem.trimStart(u8, seg, " \n");
    if (std.mem.startsWith(u8, rest, "assistant")) rest = rest["assistant".len..];
    rest = std.mem.trimStart(u8, rest, " \n");
    if (rest.len == 0) return true;
    if (rest.len < "to=".len) return std.mem.startsWith(u8, "to=", rest);
    if (!std.mem.startsWith(u8, rest, "to=")) return false;
    for (rest["to=".len..]) |c| {
        if (!museIsRecipientChar(c)) return false;
    }
    return true;
}

/// Muse-Glimmer opener classification for the no-tools streaming think arm.
/// The model's first bytes after the prompt's `<|start|>assistant` are a
/// recipient header in ORDINARY text: ` to=self<|message|>` opens reasoning
/// (closes at <|eom|>), any other resolved header means the model went
/// straight to content. Payload sizes are the byte offset just past the
/// header's `<|message|>`.
pub const MuseThinkOpener = union(enum) { not_muse, growing, self_opened: usize, direct: usize };

pub fn museThinkOpenerAt(buf: []const u8) MuseThinkOpener {
    var rest = std.mem.trimStart(u8, buf, " \n");
    var saw_ctx = false;
    if (std.mem.startsWith(u8, rest, MUSE_START_TAG)) {
        rest = rest[MUSE_START_TAG.len..];
        saw_ctx = true;
    }
    if (std.mem.startsWith(u8, rest, "assistant")) {
        rest = rest["assistant".len..];
        rest = std.mem.trimStart(u8, rest, " \n");
        saw_ctx = true;
    } else if (saw_ctx) {
        return if (std.mem.startsWith(u8, "assistant", rest)) .growing else .not_muse;
    }
    if (std.mem.startsWith(u8, rest, MUSE_MSG_TAG)) {
        return .{ .direct = buf.len - rest.len + MUSE_MSG_TAG.len };
    }
    if (!std.mem.startsWith(u8, rest, "to=")) {
        if (rest.len < "to=".len and std.mem.startsWith(u8, "to=", rest)) return .growing;
        return .not_muse;
    }
    const after_to = rest["to=".len..];
    if (std.mem.indexOf(u8, after_to, MUSE_MSG_TAG)) |msg_rel| {
        const recipient = std.mem.trim(u8, after_to[0..msg_rel], " \n");
        const consumed = (buf.len - after_to.len) + msg_rel + MUSE_MSG_TAG.len;
        if (std.mem.eql(u8, recipient, "self")) return .{ .self_opened = consumed };
        return .{ .direct = consumed };
    }
    for (after_to) |c| {
        if (!museIsRecipientChar(c)) return .not_muse;
    }
    return .growing;
}

/// A muse prompt that ends at `<|start|>assistant` leaves the model to write its own
/// header, so the plain stream arm must start skipping as if `<|start|>` had just arrived.
pub fn promptTailOpensMuseHeader(tail: []const u8) bool {
    return std.mem.endsWith(u8, std.mem.trimEnd(u8, tail, "\n "), MUSE_START_TAG ++ "assistant");
}

/// The held header bytes between `<|start|>` and `<|message|>` name `self`: the segment is reasoning.
pub fn museHeaderOpensReasoning(head: []const u8) bool {
    var rest = std.mem.trimStart(u8, head, " \n");
    if (std.mem.startsWith(u8, rest, "assistant")) rest = std.mem.trimStart(u8, rest["assistant".len..], " \n");
    if (!std.mem.startsWith(u8, rest, "to=")) return false;
    return std.mem.eql(u8, std.mem.trim(u8, rest["to=".len..], " \n"), "self");
}

/// Length of a COMPLETE next-segment header at the start of `text`
/// (`<|start|>assistant to=<r><|message|>`), or null when text doesn't start
/// with one or the header hasn't fully arrived yet.
pub fn museContentHeaderSkip(text: []const u8) ?usize {
    if (!std.mem.startsWith(u8, text, MUSE_START_TAG)) return null;
    const seg = text[MUSE_START_TAG.len..];
    const msg_rel = std.mem.indexOf(u8, seg, MUSE_MSG_TAG) orelse return null;
    return MUSE_START_TAG.len + msg_rel + MUSE_MSG_TAG.len;
}

/// Plain-arm segment-header skip for muse streams: the `<|start|>` marker
/// token arms it, `<|message|>` disarms — everything between is the
/// role+recipient header, never content. Both markers are single special
/// tokens, so exact-match at the delta level is complete.
pub fn museHeaderSkipNext(skipping: bool, token_text: []const u8) bool {
    if (std.mem.eql(u8, token_text, MUSE_START_TAG)) return true;
    if (std.mem.eql(u8, token_text, MUSE_MSG_TAG)) return false;
    return skipping;
}

// ── tests ──

test "splitMuseChannels: headerless first segment (prompt-committed header) is content" {
    // With `to=user<|message|>` committed in the prompt, the model's first
    // bytes are BODY. If it then re-opens a segment, the leading text must
    // survive as content, not vanish.
    const split = splitMuseChannels("answer<|eom|><|start|>assistant to=self<|message|>notes").?;
    try testing.expectEqualStrings("answer", split.content);
    try testing.expectEqualStrings("notes", split.reasoning_content.?);
    // A truncated bare header is still never content.
    const hdr = splitMuseChannels(" to=self<|eom|><|start|>assistant to=user<|message|>hi").?;
    try testing.expectEqualStrings("hi", hdr.content);
    try testing.expect(hdr.reasoning_content == null);
}

test "splitMuseChannels: declines harmony traffic (<|channel|> present)" {
    // The mirror of the case above. Without this, gpt-oss reasoning routes
    // through muse's recipient rules — an `analysis` header carries no `to=`,
    // so it would land in CONTENT and the answer would be preceded by the
    // model's private reasoning.
    const harmony = "<|channel|>analysis<|message|>secret<|end|>" ++
        "<|start|>assistant<|channel|>final<|message|>hi<|return|>";
    try std.testing.expect(splitMuseChannels(harmony) == null);
}

test "museThinkOpenerAt: self opens reasoning, others answer directly, partials grow" {
    try testing.expectEqual(MuseThinkOpener.growing, museThinkOpenerAt(" to=se"));
    switch (museThinkOpenerAt(" to=self<|message|>")) {
        .self_opened => |hl| try testing.expectEqual(@as(usize, " to=self<|message|>".len), hl),
        else => return error.TestUnexpectedResult,
    }
    switch (museThinkOpenerAt(" to=user<|message|>Hi")) {
        .direct => |hl| try testing.expectEqual(@as(usize, " to=user<|message|>".len), hl),
        else => return error.TestUnexpectedResult,
    }
    try testing.expectEqual(MuseThinkOpener.not_muse, museThinkOpenerAt("Okay, the user wants X."));
    // Complete next-segment header skip; incomplete returns null.
    try testing.expectEqual(@as(usize, "<|start|>assistant to=user<|message|>".len), museContentHeaderSkip("<|start|>assistant to=user<|message|>Hello").?);
    try testing.expect(museContentHeaderSkip("<|start|>assistant to=us") == null);
}

test "a muse header the MODEL writes at position 0 is resolved, self means reasoning" {
    // The prompt ends at `<|start|>assistant`; the model's ` to=self<|message|>` used to
    // stream as content because only a `<|start|>` TOKEN armed the header skip.
    try std.testing.expect(promptTailOpensMuseHeader("...<|eot|><|start|>assistant"));
    try std.testing.expect(promptTailOpensMuseHeader("<|start|>assistant\n"));
    try std.testing.expect(!promptTailOpensMuseHeader("<|start|>assistant to=user<|message|>"));
    try std.testing.expect(museHeaderOpensReasoning(" to=self"));
    try std.testing.expect(museHeaderOpensReasoning("assistant to=self\n"));
    try std.testing.expect(!museHeaderOpensReasoning(" to=user"));
    try std.testing.expect(!museHeaderOpensReasoning(" to=selfish"));
    try std.testing.expect(!museHeaderOpensReasoning(""));
}
