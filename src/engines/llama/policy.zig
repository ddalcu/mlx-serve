//! Which sequence a llama.cpp prompt claims, and how many drafts an MTP round takes.

const std = @import("std");

/// Chat templates open every prompt with a fixed prologue (system header, BOS,
/// role markers: Gemma 12 tokens, Qwen 8, Llama 4). A shared prefix this short
/// is not worth taking another conversation's sequence for.
pub const min_prefix_to_claim: usize = 16;

/// The free sequence a prompt should decode on: the one sharing the longest
/// prefix past the chat-template prologue, else the least recently used. Null
/// only when every sequence is busy. `seqs` holds `LlamaSeq`s (or a test
/// double with `busy`, `resident` and `last_used_ns`).
pub fn pickSeq(seqs: anytype, prompt: []const i32) ?usize {
    var best: ?usize = null;
    var best_shared: usize = min_prefix_to_claim - 1;
    var lru: ?usize = null;
    for (seqs, 0..) |*seq, i| {
        if (seq.busy) continue;
        const shared = std.mem.indexOfDiff(i32, seq.resident.items, prompt) orelse prompt.len;
        if (shared > best_shared) {
            best_shared = shared;
            best = i;
        }
        if (lru == null or seq.last_used_ns < seqs[lru.?].last_used_ns) lru = i;
    }
    return best orelse lru;
}

/// Draft tokens one MTP round can carry (`--llama-mtp-drafts` is clamped to it).
pub const max_drafts = 8;

/// Draft tokens a llama.cpp MTP round takes this tick, 0 = plain decode. Only a
/// slot decoding alone drafts (with company, the drafts' extra batch rows cost
/// more than they save), and never past its token budget.
pub fn draftsThisTick(live: usize, mtp_drafts: u32, remaining: u32) u32 {
    if (live != 1) return 0;
    return @min(mtp_drafts, remaining, max_drafts);
}

// ── tests ──

test "draftsThisTick: a solo slot drafts within its budget, company decodes plain" {
    try std.testing.expectEqual(@as(u32, 2), draftsThisTick(1, 2, 100));
    try std.testing.expectEqual(@as(u32, 1), draftsThisTick(1, 2, 1));
    try std.testing.expectEqual(@as(u32, 0), draftsThisTick(2, 2, 100));
    try std.testing.expectEqual(@as(u32, 0), draftsThisTick(1, 0, 100));
    try std.testing.expectEqual(@as(u32, max_drafts), draftsThisTick(1, 64, 100));
}

test "pickSeq: longest shared prefix past the prologue, else least recently used, never a busy one" {
    const gpa = std.testing.allocator;
    const Seq = struct { resident: std.ArrayList(i32) = .empty, last_used_ns: i64 = 0, busy: bool = false };
    var seqs: [3]Seq = .{ .{}, .{}, .{} };
    defer for (&seqs) |*q| q.resident.deinit(gpa);
    var prompt: [40]i32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i);

    // Every sequence empty: the least recently used one.
    seqs[0].last_used_ns = 30;
    seqs[1].last_used_ns = 10;
    seqs[2].last_used_ns = 20;
    try std.testing.expectEqual(@as(?usize, 1), pickSeq(&seqs, &prompt));

    // A template-prologue-sized overlap does not claim another conversation's KV.
    try seqs[2].resident.appendSlice(gpa, prompt[0 .. min_prefix_to_claim - 1]);
    try std.testing.expectEqual(@as(?usize, 1), pickSeq(&seqs, &prompt));

    // A real shared prefix does, and the longest wins.
    try seqs[0].resident.appendSlice(gpa, prompt[0..20]);
    try seqs[2].resident.appendSlice(gpa, prompt[min_prefix_to_claim - 1 .. 30]);
    try std.testing.expectEqual(@as(?usize, 2), pickSeq(&seqs, &prompt));

    // A busy sequence is never picked, however well it matches.
    seqs[2].busy = true;
    try std.testing.expectEqual(@as(?usize, 0), pickSeq(&seqs, &prompt));
    seqs[0].busy = true;
    seqs[1].busy = true;
    try std.testing.expectEqual(@as(?usize, null), pickSeq(&seqs, &prompt));
}
