//! Prompt Lookup Decoding (PLD) n-gram index.
//!
//! Pure-data utility used by the speculative-decoding path: given a sliding
//! window of recent tokens (`key`) and the full committed token stream
//! (`prompt + generated`), find a prior occurrence of the key and return the
//! tokens that immediately followed it as a candidate "draft." The main
//! verifier model then checks that draft in a single multi-token forward.
//!
//! v1 implementation: linear scan from the end (latest match wins). For typical
//! decode contexts (few-thousand tokens) this is sub-microsecond and not on the
//! critical path. A suffix-automaton variant is reserved for v2 if profiling
//! ever shows it.

const std = @import("std");

pub const PldLookup = struct {
    committed: []const u32,
    key_len: u32,

    /// Find the most recent occurrence of `key` inside `committed[..committed.len - key_len]`
    /// (the trailing `key_len` tokens are excluded so we don't match against the
    /// query itself), and return up to `max_draft` tokens that immediately
    /// follow that occurrence. Returns `null` when:
    ///   - `key.len != self.key_len`
    ///   - `key.len == 0` or `max_draft == 0`
    ///   - `committed.len < key.len + 1` (no possible match)
    ///   - the key never appeared earlier in the committed stream
    ///
    /// The draft is naturally clipped: if a match site is near the end of the
    /// committed stream, the returned slice may be shorter than `max_draft`.
    pub fn findMatch(self: PldLookup, key: []const u32, max_draft: u32) ?[]const u32 {
        if (key.len == 0 or max_draft == 0) return null;
        if (key.len != self.key_len) return null;
        if (self.committed.len <= key.len) return null;

        // Scan from the end backwards. The latest match site is the most
        // semantically relevant — it's "what we just said we were saying."
        const last_start: usize = self.committed.len - key.len;
        var i: usize = last_start;
        while (i > 0) {
            i -= 1;
            if (std.mem.eql(u32, self.committed[i .. i + key.len], key)) {
                const draft_start = i + key.len;
                if (draft_start >= self.committed.len) return null;
                const remaining = self.committed.len - draft_start;
                const take = @min(@as(usize, max_draft), remaining);
                if (take == 0) return null;
                return self.committed[draft_start .. draft_start + take];
            }
        }
        return null;
    }
};

/// Adaptive gating helper.
///
/// Given the current request's tokenized prompt, estimate the proportion of
/// `ngram_len`-grams that recur. The score is the ratio of distinct n-grams
/// that appear at least twice to the total distinct n-grams. Range [0, 1].
///
/// Why this metric (and not raw n-gram count): repeated n-grams are exactly
/// the ones PLD's lookup will hit, and the **distinct-count** form normalizes
/// for prompt length so the threshold doesn't drift with input size. A long
/// novel-content prompt and a short novel-content prompt should both score
/// near 0.
///
/// Use case: at request entry, score the prompt + recent assistant turns. If
/// score < threshold (~0.15), disable PLD/drafter for this request — they
/// will only add overhead on novel content.
///
/// O(N * ngram_len) time, O(N) memory; called once per request, so the cost
/// is negligible. Returns 0 on inputs shorter than `ngram_len`.
pub fn ngramRepeatScore(allocator: std.mem.Allocator, tokens: []const u32, ngram_len: u32) !f32 {
    if (tokens.len < ngram_len or ngram_len == 0) return 0.0;
    if (ngram_len > 8) return error.NgramLenTooLarge; // hash uses 8 u32 max

    const N: usize = tokens.len - ngram_len + 1;
    if (N == 0) return 0.0;

    // Pack each n-gram into a u64 hash by FNV-1a — cheap, low-collision for
    // small token id alphabets and small N. We don't need cryptographic
    // strength, just enough to avoid spurious double-counting.
    const Counts = std.AutoHashMap(u64, u32);
    var counts = Counts.init(allocator);
    defer counts.deinit();
    try counts.ensureTotalCapacity(@intCast(N));

    var i: usize = 0;
    while (i < N) : (i += 1) {
        var h: u64 = 14695981039346656037;
        var j: u32 = 0;
        while (j < ngram_len) : (j += 1) {
            h ^= @as(u64, tokens[i + j]);
            h = h *% 1099511628211;
        }
        const gop = try counts.getOrPut(h);
        if (gop.found_existing) gop.value_ptr.* += 1 else gop.value_ptr.* = 1;
    }

    var distinct: u32 = 0;
    var repeated: u32 = 0;
    var it = counts.iterator();
    while (it.next()) |e| {
        distinct += 1;
        if (e.value_ptr.* >= 2) repeated += 1;
    }
    if (distinct == 0) return 0.0;
    return @as(f32, @floatFromInt(repeated)) / @as(f32, @floatFromInt(distinct));
}

/// Fraction of the last `window` positions of `committed` whose trailing
/// `key_len`-gram ALSO appears earlier in `committed` — i.e. the fraction of
/// positions where a PLD lookup would have found a candidate. Unlike
/// `ngramRepeatScore` (self-repetition within one window), this catches the
/// echo workload where generated text repeats the PROMPT: the n-grams of the
/// echoed tail match the prompt occurrence, not each other. Used by the
/// mid-request spec re-enable check. Returns 0 when the sequence is shorter
/// than key_len+1 or window is 0.
pub fn tailMatchFraction(committed: []const u32, window: usize, key_len: u32) f32 {
    const kl: usize = @intCast(key_len);
    if (kl == 0 or committed.len <= kl or window == 0) return 0.0;
    // Tail positions i (the position AFTER each key): the key for position i
    // is committed[i-kl..i]. Earliest scorable i is kl.
    const last: usize = committed.len;
    const first: usize = if (last -| window > kl) last - window else kl;
    if (first >= last) return 0.0;

    var matched: usize = 0;
    var total: usize = 0;
    var i: usize = first;
    while (i < last) : (i += 1) {
        const key = committed[i - kl .. i];
        total += 1;
        // Does this key occur anywhere strictly before its own site?
        var j: usize = 0;
        const scan_end = i - kl; // last start where the match is strictly earlier
        while (j < scan_end) : (j += 1) {
            if (std.mem.eql(u32, committed[j .. j + kl], key)) {
                matched += 1;
                break;
            }
        }
    }
    if (total == 0) return 0.0;
    return @as(f32, @floatFromInt(matched)) / @as(f32, @floatFromInt(total));
}

// ── tests ──

test "PldLookup.findMatch returns slice at latest match site" {
    const committed = [_]u32{ 0, 1, 2, 3, 1, 2, 4, 5, 6 };
    const key = [_]u32{ 1, 2 };
    const lookup = PldLookup{ .committed = &committed, .key_len = 2 };
    const draft = lookup.findMatch(&key, 3) orelse return error.ExpectedMatch;
    // Latest in-bounds match of [1,2] starts at index 4; draft = committed[6..9] = [4,5,6].
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 6 }, draft);
}

test "PldLookup.findMatch returns null when key not found" {
    const committed = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const key = [_]u32{ 99, 100 };
    const lookup = PldLookup{ .committed = &committed, .key_len = 2 };
    try std.testing.expect(lookup.findMatch(&key, 3) == null);
}

test "PldLookup.findMatch clips draft to remaining context" {
    // Match at index 2 (the trailing [1,2] at index 5 is the "self" query and
    // excluded by last_start). Tokens after index 2's match start at index 4
    // and there are 3 of them (committed[4..7] = [7, 1, 2]); requesting
    // max_draft=5 should clip to those 3.
    const committed = [_]u32{ 9, 8, 1, 2, 7, 1, 2 };
    const key = [_]u32{ 1, 2 };
    const lookup = PldLookup{ .committed = &committed, .key_len = 2 };
    const draft = lookup.findMatch(&key, 5) orelse return error.ExpectedMatch;
    try std.testing.expectEqualSlices(u32, &.{ 7, 1, 2 }, draft);
    try std.testing.expect(draft.len <= 5);
}

test "PldLookup.findMatch with key longer than committed returns null" {
    const committed = [_]u32{ 1, 2 };
    const key = [_]u32{ 1, 2, 3, 4 };
    const lookup = PldLookup{ .committed = &committed, .key_len = 4 };
    try std.testing.expect(lookup.findMatch(&key, 3) == null);
}

test "PldLookup.findMatch prefers latest match over earlier" {
    // [1,2] appears at indices 0, 4, 8 (last is the self-occurrence and excluded).
    const committed = [_]u32{ 1, 2, 100, 200, 1, 2, 50, 60, 1, 2 };
    const key = [_]u32{ 1, 2 };
    const lookup = PldLookup{ .committed = &committed, .key_len = 2 };
    const draft = lookup.findMatch(&key, 2) orelse return error.ExpectedMatch;
    // Should return tokens after the index-4 match (= [50, 60]) — NOT index 0.
    try std.testing.expectEqualSlices(u32, &.{ 50, 60 }, draft);
}

test "PldLookup.findMatch empty key returns null" {
    const committed = [_]u32{ 1, 2, 3 };
    const key = [_]u32{};
    const lookup = PldLookup{ .committed = &committed, .key_len = 0 };
    try std.testing.expect(lookup.findMatch(&key, 3) == null);
}

test "PldLookup.findMatch zero max_draft returns null" {
    const committed = [_]u32{ 1, 2, 3, 1, 2, 4 };
    const key = [_]u32{ 1, 2 };
    const lookup = PldLookup{ .committed = &committed, .key_len = 2 };
    try std.testing.expect(lookup.findMatch(&key, 0) == null);
}

test "PldLookup.findMatch key length mismatch returns null" {
    const committed = [_]u32{ 1, 2, 3, 1, 2, 4 };
    const key = [_]u32{ 1, 2 };
    // self.key_len=3 but key.len=2 → caller bug; reject defensively.
    const lookup = PldLookup{ .committed = &committed, .key_len = 3 };
    try std.testing.expect(lookup.findMatch(&key, 3) == null);
}

test "ngramRepeatScore: highly repetitive tokens score high" {
    // 6 copies of [1,2,3]. Distinct 3-grams (sliding): {1,2,3}, {2,3,1}, {3,1,2}.
    // All 3 appear multiple times → score should be 1.0.
    const tokens = [_]u32{ 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3 };
    const score = try ngramRepeatScore(std.testing.allocator, &tokens, 3);
    try std.testing.expectEqual(@as(f32, 1.0), score);
}

test "ngramRepeatScore: novel content scores low" {
    // 30 distinct tokens, every 3-gram unique → score = 0.0.
    var tokens: [30]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 1000);
    const score = try ngramRepeatScore(std.testing.allocator, &tokens, 3);
    try std.testing.expect(score < 0.05);
}

test "ngramRepeatScore: half-and-half scores in the middle" {
    // 10 tokens of [1,2,3,1,2,3,...] (high repeat) + 10 distinct novel tokens
    // (no repeat in the novel half). Roughly half the distinct n-grams should
    // recur → score in the [0.20, 0.55] range (some 3-grams span the boundary).
    var tokens: [20]u32 = undefined;
    for (tokens[0..10], 0..) |*t, i| t.* = @intCast((i % 3) + 1);
    for (tokens[10..], 0..) |*t, i| t.* = @intCast(i + 5000);
    const score = try ngramRepeatScore(std.testing.allocator, &tokens, 3);
    try std.testing.expect(score > 0.10);
    try std.testing.expect(score < 0.55);
}

test "ngramRepeatScore: returns 0 on inputs shorter than ngram_len" {
    const tokens = [_]u32{ 1, 2 };
    const score = try ngramRepeatScore(std.testing.allocator, &tokens, 3);
    try std.testing.expectEqual(@as(f32, 0.0), score);
}

test "ngramRepeatScore: rejects oversized ngram_len" {
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    try std.testing.expectError(error.NgramLenTooLarge, ngramRepeatScore(std.testing.allocator, &tokens, 9));
}

test "tailMatchFraction: tail echoing the PROMPT scores high" {
    // The re-enable regression: a prompt paragraph echoed verbatim in the
    // generated tail. The tail has no INTERNAL repeats (ngramRepeatScore ≈ 0)
    // but every tail 3-gram appears earlier — in the prompt occurrence.
    var committed: [64]u32 = undefined;
    // prompt: tokens 100..131 (a 32-token "paragraph"), then 16 novel
    // "preamble" tokens, then the paragraph echoed for the last 16 positions.
    for (committed[0..32], 0..) |*t, i| t.* = @intCast(i + 100);
    for (committed[32..48], 0..) |*t, i| t.* = @intCast(i + 9000);
    for (committed[48..64], 0..) |*t, i| t.* = @intCast(i + 100);
    const frac = tailMatchFraction(&committed, 16, 3);
    try std.testing.expect(frac > 0.7);
}

test "tailMatchFraction: novel tail scores zero" {
    var committed: [64]u32 = undefined;
    for (&committed, 0..) |*t, i| t.* = @intCast(i * 13 + 7);
    try std.testing.expectEqual(@as(f32, 0.0), tailMatchFraction(&committed, 16, 3));
}

test "tailMatchFraction: degenerate inputs return 0" {
    const short = [_]u32{ 1, 2 };
    try std.testing.expectEqual(@as(f32, 0.0), tailMatchFraction(&short, 16, 3));
    const ok = [_]u32{ 1, 2, 3, 4, 5 };
    try std.testing.expectEqual(@as(f32, 0.0), tailMatchFraction(&ok, 0, 3));
    try std.testing.expectEqual(@as(f32, 0.0), tailMatchFraction(&ok, 16, 0));
}

/// A continuation copied from earlier in the stream, backed by `match`
/// tokens that precede it there and end the stream.
pub const Copy = struct { draft: []const u32, match: usize };

/// The continuation after the site whose preceding tokens match the stream's
/// tail longest (the last `COPY_KEY` at least, counted up to `COPY_MAX_MATCH`;
/// the most recent site on a tie), up to `max_draft` tokens. Null when no
/// site is backed by `min_match` tokens.
pub fn backedCopy(committed: []const u32, max_draft: usize, min_match: usize) ?Copy {
    const n = committed.len;
    if (max_draft == 0 or n <= COPY_KEY) return null;
    var best_end: usize = 0;
    var best: usize = 0;
    var end: usize = n - 1;
    while (end >= COPY_KEY) : (end -= 1) {
        var len: usize = 0;
        while (len < COPY_MAX_MATCH and len < end and committed[end - 1 - len] == committed[n - 1 - len]) len += 1;
        if (len >= COPY_KEY and len > best) {
            best = len;
            best_end = end;
            if (len == COPY_MAX_MATCH) break;
        }
    }
    if (best < min_match) return null;
    return .{ .draft = committed[best_end..@min(n, best_end + max_draft)], .match = best };
}
const COPY_KEY = 3;
const COPY_MAX_MATCH = 64;

/// The last `buf.len` tokens of `prompt ++ gen ++ {next}`, oldest first.
pub fn tailWindow(buf: []u32, prompt: []const u32, gen: []const u32, next: u32) void {
    const total = prompt.len + gen.len + 1;
    std.debug.assert(buf.len <= total);
    for (buf, total - buf.len..) |*c, i| c.* = if (i < prompt.len) prompt[i] else if (i < prompt.len + gen.len) gen[i - prompt.len] else next;
}

test "tailWindow: the stream's tail across the prompt, the generation and the next token" {
    const prompt = [_]u32{ 1, 2, 3 };
    const gen = [_]u32{ 4, 5 };
    var buf: [6]u32 = undefined;
    tailWindow(&buf, &prompt, &gen, 9);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 9 }, &buf);
    tailWindow(buf[0..3], &prompt, &gen, 9);
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 9 }, buf[0..3]);
    tailWindow(buf[0..1], &prompt, &gen, 9);
    try std.testing.expectEqualSlices(u32, &.{9}, buf[0..1]);
    tailWindow(buf[0..2], &prompt, &.{}, 9);
    try std.testing.expectEqualSlices(u32, &.{ 3, 9 }, buf[0..2]);
}

test "backedCopy: the longest-backed site wins, the most recent on a tie" {
    // "1 2 3 4 5" then "9 2 3 4 6", tail "... 1 2 3 4": the first site is backed by 4 tokens.
    const s = [_]u32{ 1, 2, 3, 4, 5, 7, 9, 2, 3, 4, 6, 8, 1, 2, 3, 4 };
    const c = backedCopy(&s, 3, 4).?;
    try std.testing.expectEqual(@as(usize, 4), c.match);
    try std.testing.expectEqualSlices(u32, &.{ 5, 7, 9 }, c.draft);
    try std.testing.expect(backedCopy(&s, 3, 5) == null);
    // Two sites backed equally: the later one.
    const t = [_]u32{ 5, 6, 7, 1, 5, 6, 7, 2, 5, 6, 7 };
    try std.testing.expectEqualSlices(u32, &.{2}, backedCopy(&t, 1, 3).?.draft);
    // No site ends inside the tail itself, and nothing follows the last token.
    const u = [_]u32{ 4, 4, 4, 4 };
    try std.testing.expectEqualSlices(u32, &.{4}, backedCopy(&u, 5, 3).?.draft);
    try std.testing.expect(backedCopy(&[_]u32{ 1, 2, 3 }, 4, 3) == null);
}
