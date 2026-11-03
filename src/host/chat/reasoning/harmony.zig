//! Harmony channel segments and stream verdicts.

const std = @import("std");

const ThinkSplit = @import("think_tags.zig").ThinkSplit;
const museIsRecipientChar = @import("muse.zig").museIsRecipientChar;
const StreamThinkGate = @import("stream_gate.zig").StreamThinkGate;

// ── gpt_oss / harmony channels ──
//
// After the generation prompt's bare `<|start|>assistant`, the model emits
// channel segments:
//   `<|channel|>analysis<|message|>REASONING<|end|>`
//   `<|start|>assistant<|channel|>final<|message|>ANSWER<|return|>`
//   `<|start|>assistant to=functions.NAME<|channel|>commentary <|constrain|>json`
//       `<|message|>{args}<|call|>`
//
// Muse is this format's descendant and shares `<|start|>`/`<|message|>`, so
// `<|message|>` alone CANNOT discriminate the two families. `<|channel|>` is
// harmony's and only harmony's — both routers key on it, one to claim and one
// to decline.
pub const HARMONY_START_TAG = "<|start|>";
pub const HARMONY_CHANNEL_TAG = "<|channel|>";
pub const HARMONY_MSG_TAG = "<|message|>";
pub const HARMONY_END_TAG = "<|end|>";
pub const HARMONY_RETURN_TAG = "<|return|>";
pub const HARMONY_CALL_TAG = "<|call|>";

const HarmonySegment = struct { channel: []const u8, recipient: []const u8, body: []const u8 };

/// Channel, recipient and body of the segment in text[seg_start..seg_end].
/// null when the header never resolved — a header still streaming in is never
/// content.
///
/// The recipient is searched for across the WHOLE header because harmony
/// permits it on either side of the channel name (`assistant to=functions.x
/// <|channel|>commentary` and `<|channel|>commentary to=functions.x` are the
/// same call).
fn harmonySegmentAt(text: []const u8, seg_start: usize, seg_end: usize) ?HarmonySegment {
    const seg = text[seg_start..seg_end];
    const msg_rel = std.mem.indexOf(u8, seg, HARMONY_MSG_TAG) orelse return null;
    const header = seg[0..msg_rel];
    const chan_rel = std.mem.indexOf(u8, header, HARMONY_CHANNEL_TAG) orelse return null;

    // Channel name = the identifier run right after the marker.
    const after = header[chan_rel + HARMONY_CHANNEL_TAG.len ..];
    var n: usize = 0;
    while (n < after.len and museIsRecipientChar(after[n])) n += 1;
    const channel = after[0..n];

    const recipient: []const u8 = if (std.mem.indexOf(u8, header, "to=")) |t| blk: {
        const r = header[t + 3 ..];
        var m: usize = 0;
        while (m < r.len and museIsRecipientChar(r[m])) m += 1;
        break :blk r[0..m];
    } else "";

    var body = seg[msg_rel + HARMONY_MSG_TAG.len ..];
    for ([_][]const u8{ HARMONY_END_TAG, HARMONY_RETURN_TAG, HARMONY_CALL_TAG }) |tag| {
        if (std.mem.indexOf(u8, body, tag)) |e| body = body[0..e];
    }
    return .{ .channel = channel, .recipient = recipient, .body = body };
}

/// gpt_oss channel split. Claims the text only when `<|channel|>` is present.
/// First segment of each destination wins. A tool segment's body rides out as
/// content ONLY when there is no final segment, so the KeepingMarkup callers
/// can still see the call and the trim wrapper cuts it from display — the same
/// contract muse's splitter honours.
pub fn splitHarmonyChannels(text: []const u8) ?ThinkSplit {
    if (std.mem.indexOf(u8, text, HARMONY_CHANNEL_TAG) == null) return null;
    var reasoning: ?[]const u8 = null;
    var content: ?[]const u8 = null;
    var tool_body: ?[]const u8 = null;
    var routed = false;

    var seg_start: usize = 0;
    while (true) {
        const next = std.mem.indexOfPos(u8, text, seg_start + 1, HARMONY_START_TAG);
        const seg_end = next orelse text.len;
        if (harmonySegmentAt(text, seg_start, seg_end)) |s| {
            routed = true;
            if (std.mem.startsWith(u8, s.recipient, "functions.")) {
                // A tool call, whatever channel it rode in on.
                if (tool_body == null) tool_body = s.body;
            } else if (std.mem.eql(u8, s.channel, "analysis")) {
                if (reasoning == null) {
                    const r = std.mem.trim(u8, s.body, "\n ");
                    if (r.len > 0) reasoning = r;
                }
            } else {
                // `final`, and a bare `commentary` preamble addressed to no
                // tool — both are user-visible.
                if (content == null) content = std.mem.trim(u8, s.body, "\n ");
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

/// What a harmony segment header at the START of a streaming buffer means.
/// Mirrors `museThinkOpenerAt`: the streaming loop consumes the header bytes
/// and latches a close tag, so the close matcher's pos/len model then works on
/// reasoning that really does begin at byte 0.
pub const HarmonyOpener = union(enum) {
    /// `<|channel|>analysis<|message|>` — reasoning follows, closes at <|end|>.
    /// Payload is the header length to strip.
    analysis: usize,
    /// A resolved header for any OTHER channel (`final`, bare `commentary`):
    /// the answer starts immediately, with no reasoning at all.
    direct: usize,
    /// A header still arriving token by token — decide nothing yet, or its
    /// text leaks as reasoning.
    growing,
    not_harmony,
};

pub fn harmonyThinkOpenerAt(buf: []const u8) HarmonyOpener {
    var i: usize = 0;
    if (std.mem.startsWith(u8, buf, HARMONY_START_TAG)) {
        i = HARMONY_START_TAG.len;
        while (i < buf.len and museIsRecipientChar(buf[i])) i += 1;
    } else if (HARMONY_START_TAG.len > buf.len and std.mem.startsWith(u8, HARMONY_START_TAG, buf)) {
        // A strict prefix of `<|start|>` — could still become one.
        return if (buf.len == 0) .not_harmony else .growing;
    }
    const rest = buf[i..];
    if (!std.mem.startsWith(u8, rest, HARMONY_CHANNEL_TAG)) {
        // Partial `<|channel|>` still arriving.
        if (rest.len < HARMONY_CHANNEL_TAG.len and rest.len > 0 and
            std.mem.startsWith(u8, HARMONY_CHANNEL_TAG, rest)) return .growing;
        return .not_harmony;
    }
    const after_chan = i + HARMONY_CHANNEL_TAG.len;
    const msg = std.mem.indexOfPos(u8, buf, after_chan, HARMONY_MSG_TAG) orelse return .growing;
    const hdr_len = msg + HARMONY_MSG_TAG.len;
    if (std.mem.startsWith(u8, buf[after_chan..], "analysis")) return .{ .analysis = hdr_len };
    return .{ .direct = hdr_len };
}

/// Byte length of a leading harmony ANALYSIS header
/// (`[<|start|>assistant]<|channel|>analysis<|message|>`), or null when `buf`
/// does not open with one.
///
/// The streaming split identifies a reasoning block by a close tag and emits
/// everything before it, which assumes the reasoning starts at byte 0. Harmony
/// puts a header there instead, and every byte of it is ordinary text — so
/// without this the first reasoning delta opens with a literal
/// `<|channel|>analysis<|message|>`.
pub fn harmonyAnalysisHeaderLen(buf: []const u8) ?usize {
    var i: usize = 0;
    if (std.mem.startsWith(u8, buf, HARMONY_START_TAG)) {
        // Skip `<|start|>` + the role that follows it.
        i = HARMONY_START_TAG.len;
        while (i < buf.len and museIsRecipientChar(buf[i])) i += 1;
    }
    if (!std.mem.startsWith(u8, buf[i..], HARMONY_CHANNEL_TAG)) return null;
    const after_chan = i + HARMONY_CHANNEL_TAG.len;
    if (!std.mem.startsWith(u8, buf[after_chan..], "analysis")) return null;
    const msg = std.mem.indexOfPos(u8, buf, after_chan, HARMONY_MSG_TAG) orelse return null;
    return msg + HARMONY_MSG_TAG.len;
}

/// Drop a leading harmony segment header of ANY channel — the answer arrives
/// as its own `<|start|>assistant<|channel|>final<|message|>` segment once the
/// analysis channel closes, and that header is ordinary text that would
/// otherwise ride out as the first bytes of content.
pub fn stripHarmonySegmentHeader(text: []const u8) []const u8 {
    var i: usize = 0;
    if (std.mem.startsWith(u8, text, HARMONY_START_TAG)) {
        i = HARMONY_START_TAG.len;
        while (i < text.len and museIsRecipientChar(text[i])) i += 1;
    }
    if (!std.mem.startsWith(u8, text[i..], HARMONY_CHANNEL_TAG)) return text;
    const msg = std.mem.indexOfPos(u8, text, i, HARMONY_MSG_TAG) orelse return text;
    return text[msg + HARMONY_MSG_TAG.len ..];
}

/// gpt_oss streaming verdict, decided from the channel markers alone.
/// null = not harmony traffic (fall through to the generic think gate).
///
/// The header between `<|channel|>` and `<|message|>` is ORDINARY text —
/// only the delimiters are single special tokens — so an unresolved header
/// always HOLDS. Flushing it leaks `analysis` / `commentary to=functions.…`
/// fragments into the user's stream, which is the muse ` to=user` bug in a
/// different alphabet.
pub fn harmonyStreamVerdict(buf: []const u8) ?StreamThinkGate {
    if (std.mem.indexOf(u8, buf, HARMONY_CHANNEL_TAG) == null) return null;
    const seg_start = std.mem.lastIndexOf(u8, buf, HARMONY_START_TAG) orelse 0;
    const seg = buf[seg_start..];
    const chan_rel = std.mem.indexOf(u8, seg, HARMONY_CHANNEL_TAG) orelse return .hold_thinking;
    const msg_rel = std.mem.indexOfPos(u8, seg, chan_rel, HARMONY_MSG_TAG) orelse return .hold_thinking;
    const header = seg[0..msg_rel];

    const after = header[chan_rel + HARMONY_CHANNEL_TAG.len ..];
    var n: usize = 0;
    while (n < after.len and museIsRecipientChar(after[n])) n += 1;
    const channel = after[0..n];

    const recipient: []const u8 = if (std.mem.indexOf(u8, header, "to=")) |t| blk: {
        const r = header[t + 3 ..];
        var m: usize = 0;
        while (m < r.len and museIsRecipientChar(r[m])) m += 1;
        break :blk r[0..m];
    } else "";

    // Tool payload: the tools path buffers before this gate is consulted;
    // hold defensively on a no-tools request rather than leak raw arguments.
    if (std.mem.startsWith(u8, recipient, "functions.")) return .hold_thinking;
    if (std.mem.eql(u8, channel, "analysis")) {
        const closed = std.mem.indexOf(u8, seg, HARMONY_END_TAG) != null or
            std.mem.indexOf(u8, seg, HARMONY_RETURN_TAG) != null;
        return if (closed) .split_think else .hold_thinking;
    }
    return .split_think;
}

/// gpt_oss streaming tool hold: true while a segment header is unresolved
/// (the next token decides whether it names a tool) or already resolved to a
/// `functions.` recipient (the body is call arguments — parsed on the end
/// flush, never streamed).
pub fn harmonyHeaderHoldsForTools(buf: []const u8) bool {
    const chan = std.mem.lastIndexOf(u8, buf, HARMONY_CHANNEL_TAG) orelse return false;
    // Header runs from the segment's `<|start|>` (the `to=` may sit on either
    // side of the channel marker) to `<|message|>`. No `<|message|>` yet means
    // the header is still growing — hold, the next token may name a tool.
    const msg_rel = std.mem.indexOfPos(u8, buf, chan, HARMONY_MSG_TAG) orelse return true;
    const seg_start = std.mem.lastIndexOf(u8, buf, HARMONY_START_TAG) orelse 0;
    return std.mem.indexOf(u8, buf[seg_start..msg_rel], "to=functions.") != null;
}

// ── tests ──

test "harmonyAnalysisHeaderLen: measures the header the streaming split must skip" {
    // With and without the leading <|start|>assistant (the generation prompt
    // commits that part on the FIRST segment, so the model's first bytes are
    // the bare <|channel|>).
    const bare = "<|channel|>analysis<|message|>thinking...";
    try std.testing.expectEqual(@as(?usize, "<|channel|>analysis<|message|>".len), harmonyAnalysisHeaderLen(bare));
    const full = "<|start|>assistant<|channel|>analysis<|message|>x";
    try std.testing.expectEqual(@as(?usize, "<|start|>assistant<|channel|>analysis<|message|>".len), harmonyAnalysisHeaderLen(full));
    // A header still arriving has no length yet — the split must not fire.
    try std.testing.expectEqual(@as(?usize, null), harmonyAnalysisHeaderLen("<|channel|>analysis"));
    // Only the ANALYSIS channel: `final` is content, not reasoning.
    try std.testing.expectEqual(@as(?usize, null), harmonyAnalysisHeaderLen("<|channel|>final<|message|>hi"));
    try std.testing.expectEqual(@as(?usize, null), harmonyAnalysisHeaderLen("plain text"));
}

test "stripHarmonySegmentHeader: drops a leading segment header of any channel" {
    try std.testing.expectEqualStrings(
        "The answer.",
        stripHarmonySegmentHeader("<|start|>assistant<|channel|>final<|message|>The answer."),
    );
    try std.testing.expectEqualStrings(
        "Working.",
        stripHarmonySegmentHeader("<|channel|>commentary<|message|>Working."),
    );
    // Nothing to strip — returned untouched, never truncated.
    try std.testing.expectEqualStrings("plain content", stripHarmonySegmentHeader("plain content"));
    try std.testing.expectEqualStrings(
        "<|start|>assistant<|channel|>fin",
        stripHarmonySegmentHeader("<|start|>assistant<|channel|>fin"),
    );
}

test "splitHarmonyChannels: analysis→reasoning, final→content, headers stripped" {
    // What gpt-oss emits after the generation prompt's bare `<|start|>assistant`.
    const text = "<|channel|>analysis<|message|>Let me think.<|end|>" ++
        "<|start|>assistant<|channel|>final<|message|>The answer is 4.<|return|>";
    const split = splitHarmonyChannels(text).?;
    try std.testing.expectEqualStrings("Let me think.", split.reasoning_content.?);
    try std.testing.expectEqualStrings("The answer is 4.", split.content);
}

test "splitHarmonyChannels: commentary to=functions.X is a TOOL body, never content" {
    // A tool call rides the commentary channel with a `functions.` recipient.
    // Routing it to content would print raw JSON at the user; routing it to
    // reasoning would lose the call entirely.
    const text = "<|channel|>analysis<|message|>Need weather.<|end|>" ++
        "<|start|>assistant to=functions.get_weather<|channel|>commentary <|constrain|>json<|message|>" ++
        "{\"location\":\"SF\"}<|call|>";
    const split = splitHarmonyChannels(text).?;
    try std.testing.expectEqualStrings("Need weather.", split.reasoning_content.?);
    try std.testing.expectEqualStrings("{\"location\":\"SF\"}", split.content);
}

test "splitHarmonyChannels: recipient may sit AFTER the channel name" {
    // Harmony allows `<|channel|>commentary to=functions.x`; the header is
    // scanned as a whole so both orderings resolve to the same recipient.
    const text = "<|start|>assistant<|channel|>commentary to=functions.ping<|message|>{}<|call|>";
    const split = splitHarmonyChannels(text).?;
    try std.testing.expectEqualStrings("{}", split.content);
    try std.testing.expect(split.reasoning_content == null);
}

test "splitHarmonyChannels: bare commentary (no recipient) is user-visible content" {
    const text = "<|channel|>commentary<|message|>Working on it.<|end|>";
    const split = splitHarmonyChannels(text).?;
    try std.testing.expectEqualStrings("Working on it.", split.content);
}

test "splitHarmonyChannels: an unresolved header is never content" {
    // A header still streaming in (no <|message|> yet) must not be claimed.
    try std.testing.expect(splitHarmonyChannels("<|channel|>anal") == null);
    try std.testing.expect(splitHarmonyChannels("<|start|>assistant<|channel|>fin") == null);
}

test "splitHarmonyChannels: declines muse traffic (no <|channel|> marker)" {
    // Both families emit <|message|>, so the muse router cannot be the one
    // that keys on it alone — <|channel|> is the discriminator, and harmony
    // must not claim a muse turn (or vice versa).
    try std.testing.expect(splitHarmonyChannels(" to=self<|message|>notes<|eom|>") == null);
}
