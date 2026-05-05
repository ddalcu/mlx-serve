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
