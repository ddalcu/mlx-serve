//! stop strings, StopStream, finish reasons

const std = @import("std");
const chat_mod = @import("chat.zig");

/// Where a stop sequence cuts the accumulated text and how many bytes of the arriving token
/// survive: streaming delivers the same `text[0..index]` non-streaming keeps.
const StopCut = struct { index: usize, token_keep: usize, matched: []const u8 };

fn stopSequenceCut(text: []const u8, token_len: usize, stops: []const []const u8) ?StopCut {
    const emitted = text.len - @min(token_len, text.len);
    var best: ?StopCut = null;
    for (stops) |stop_seq| {
        if (stop_seq.len == 0) continue;
        const idx = std.mem.indexOf(u8, text, stop_seq) orelse continue;
        if (best) |b| if (b.index <= idx) continue;
        best = .{ .index = idx, .token_keep = if (idx > emitted) idx - emitted else 0, .matched = stop_seq };
    }
    return best;
}

/// Bytes at the end of `text`, at most `fresh` of them, that could still begin a stop string.
fn stopPrefixTail(text: []const u8, fresh: usize, stops: []const []const u8) usize {
    var hold: usize = 0;
    for (stops) |stop| {
        var n = @min(stop.len -| 1, fresh);
        while (n > hold) : (n -= 1) {
            if (std.mem.endsWith(u8, text, stop[0..n])) {
                hold = n;
                break;
            }
        }
    }
    return hold;
}

/// Client stop strings on a stream: a tail that could still begin one is held until the next
/// token decides it, so the stream sends exactly the bytes the finished reply keeps.
pub const StopStream = struct {
    held: std.ArrayList(u8) = .empty,
    /// The held tail went out as the stream's last token.
    flushed: bool = false,

    const Fed = struct { send: []u8, stop: ?[]const u8 };

    pub fn deinit(self: *StopStream, a: std.mem.Allocator) void {
        self.held.deinit(a);
    }

    pub fn holding(self: *const StopStream) bool {
        return self.held.items.len > 0 and !self.flushed;
    }

    /// The held tail as one last token, once generation ended without a match.
    pub fn takeHeld(self: *StopStream, a: std.mem.Allocator) ![]u8 {
        self.flushed = true;
        const out = try a.dupe(u8, self.held.items);
        self.held.clearRetainingCapacity();
        return out;
    }

    /// Appends `token` to `buf` (the text sent so far) and returns what may go out now: up to
    /// a stop match (judged answer-only like `answerStopCut` when `answer_opened` is set), else
    /// everything but a tail that could still begin a stop string. The caller owns `send`.
    pub fn feed(self: *StopStream, a: std.mem.Allocator, buf: *std.ArrayList(u8), token: []const u8, stops: []const []const u8, answer_opened: ?bool) !Fed {
        const sent = buf.items.len;
        try buf.appendSlice(a, self.held.items);
        self.held.clearRetainingCapacity();
        try buf.appendSlice(a, token);
        const fresh = buf.items.len - sent;
        const cut = if (answer_opened) |o| answerStopCut(buf.items, fresh, stops, o) else stopSequenceCut(buf.items, fresh, stops);
        var end = buf.items.len;
        if (cut) |c| {
            end = @max(c.index, sent);
        } else if (!self.flushed) {
            end -= stopPrefixTail(buf.items, fresh, stops);
            try self.held.appendSlice(a, buf.items[end..]);
        }
        const send = try a.dupe(u8, buf.items[sent..end]);
        buf.shrinkRetainingCapacity(end);
        return .{ .send = send, .stop = if (cut) |c| c.matched else null };
    }
};

/// The text a stream of `text` split at `i` and `j` sends through a `StopStream`, and whether it stopped.
fn stopStreamReplay(a: std.mem.Allocator, text: []const u8, i: usize, j: usize, stops: []const []const u8) !struct { sent: []u8, stopped: bool } {
    var ss: StopStream = .{};
    defer ss.deinit(a);
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(a);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    for ([_][]const u8{ text[0..i], text[i..j], text[j..] }) |token| {
        const fed = try ss.feed(a, &buf, token, stops, null);
        defer a.free(fed.send);
        try out.appendSlice(a, fed.send);
        if (fed.stop != null) return .{ .sent = try out.toOwnedSlice(a), .stopped = true };
    }
    if (ss.holding()) {
        const tail = try ss.takeHeld(a);
        defer a.free(tail);
        const fed = try ss.feed(a, &buf, tail, stops, null);
        defer a.free(fed.send);
        try out.appendSlice(a, fed.send);
    }
    return .{ .sent = try out.toOwnedSlice(a), .stopped = false };
}

/// `stopSequenceCut` for a chat surface: a stop string ends the answer, never the reasoning.
/// Only matches ending in the arriving token are new; earlier ones were judged already.
pub fn answerStopCut(text: []const u8, token_len: usize, stops: []const []const u8, opened_by_template: bool) ?StopCut {
    const emitted = text.len - @min(token_len, text.len);
    var best: ?StopCut = null;
    for (stops) |stop_seq| {
        if (stop_seq.len == 0) continue;
        const idx = chat_mod.answerStopIndex(text, emitted -| (stop_seq.len - 1), stop_seq, opened_by_template) orelse continue;
        if (best) |b| if (b.index <= idx) continue;
        best = .{ .index = idx, .token_keep = if (idx > emitted) idx - emitted else 0, .matched = stop_seq };
    }
    return best;
}

/// A client that vanished mid-generation did not hit a token cap.
pub fn nonStreamFinishReason(client_gone: bool, slot_reason: []const u8) []const u8 {
    return if (client_gone) "client_disconnect" else slot_reason;
}

/// The text the model said before the tool markup, byte for byte as the stream flushes it.
pub fn visibleToolPreamble(content: []const u8) []const u8 {
    return if (std.mem.trim(u8, content, " \t\r\n").len > 0) content else "";
}

// ── tests ──

test "StopStream: at every token split, a stream sends what the finished reply keeps" {
    const a = std.testing.allocator;
    const Case = struct { text: []const u8, stop: []const u8, want: []const u8, stopped: bool };
    const cases = [_]Case{
        .{ .text = "9, 10, 11, 12, 13", .stop = ", 12", .want = "9, 10, 11", .stopped = true },
        .{ .text = "28, 29, 30", .stop = ", 30x", .want = "28, 29, 30", .stopped = false },
        .{ .text = "a, b, a", .stop = "a, c", .want = "a, b, a", .stopped = false },
    };
    for (cases) |c| {
        const stops = [_][]const u8{c.stop};
        for (0..c.text.len + 1) |i| for (i..c.text.len + 1) |j| {
            const r = try stopStreamReplay(a, c.text, i, j, &stops);
            defer a.free(r.sent);
            try std.testing.expectEqualStrings(c.want, r.sent);
            try std.testing.expectEqual(c.stopped, r.stopped);
        };
    }
}

test "a vanished client is not a token cap: the non-stream finish reason says so" {
    // Bar: the non-streaming path reports what the three streaming paths do.
    const t = std.testing;
    try t.expectEqualStrings("client_disconnect", nonStreamFinishReason(true, "length"));
    try t.expectEqualStrings("stop", nonStreamFinishReason(false, "stop"));
    try t.expectEqualStrings("length", nonStreamFinishReason(false, "length"));
}

test "a tool-call answer keeps the model's visible preamble" {
    // Bar: non-streamed content == what the stream flushes before the tool markup.
    const t = std.testing;
    const raw = "<think>pick a tool</think>Checking the weather.\n<tool_call>{\"name\":\"w\",\"arguments\":{}}</tool_call>";
    const split = chat_mod.splitThinkBlock(raw, true, false);
    try t.expectEqualStrings("Checking the weather.", visibleToolPreamble(split.content));
    const bare = chat_mod.splitThinkBlock("<tool_call>{\"name\":\"w\",\"arguments\":{}}</tool_call>", true, false);
    try t.expectEqualStrings("", visibleToolPreamble(bare.content));
    try t.expectEqualStrings("", visibleToolPreamble("  \n "));
}

test "a streaming stop sequence cuts at the match, not at the token boundary" {
    // Bar: the streamed bytes are the bytes non-streaming keeps (`text[0..idx]`).
    const t = std.testing;
    const stops = [_][]const u8{"\n\nHuman:"};
    // The stop opens inside the token that completes it: the leading bytes of
    // that token are answer text and must still be delivered.
    const buf = "Paris.\n\nHuman:";
    const cut = stopSequenceCut(buf, "s.\n\nHuman:".len, &stops).?;
    try t.expectEqual(@as(usize, 6), cut.index);
    try t.expectEqual(@as(usize, 2), cut.token_keep);
    try t.expectEqualStrings("\n\nHuman:", cut.matched);
    // Match entirely inside already-delivered bytes: nothing of this token survives.
    const late = stopSequenceCut(buf, 1, &stops).?;
    try t.expectEqual(@as(usize, 6), late.index);
    try t.expectEqual(@as(usize, 0), late.token_keep);
    // Earliest match wins, and a clean buffer cuts nothing.
    const many = [_][]const u8{ "END", "N" };
    try t.expectEqual(@as(usize, 1), stopSequenceCut("aNbEND", 6, &many).?.index);
    try t.expectEqual(@as(?StopCut, null), stopSequenceCut("all clear", 3, &stops));
}

test "a stop string matches the answer, never the reasoning" {
    const t = std.testing;
    const stops = [_][]const u8{"the"};
    // Template-opened thought: nothing matches until the block closes.
    try t.expectEqual(@as(?StopCut, null), answerStopCut("so the answer", 13, &stops, true));
    try t.expectEqual(@as(?StopCut, null), answerStopCut("<think>so the answer", 20, &stops, false));
    // The token straddling the close keeps its bytes up to the match in the answer.
    const buf = "so the</think>\nIn the";
    const cut = answerStopCut(buf, "</think>\nIn the".len, &stops, true).?;
    try t.expectEqual(std.mem.lastIndexOf(u8, buf, "the").?, cut.index);
    try t.expectEqual("</think>\nIn ".len, cut.token_keep);
    // No thought at all: the whole output is the answer.
    try t.expectEqual(@as(usize, 3), answerStopCut("in the", 6, &stops, false).?.index);
}
