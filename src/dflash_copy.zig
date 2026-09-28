//! Suffix-copy proposal for DFlash rounds, after TensorFold's
//! SuffixLookupProposer/lane_engine copy integration.
//!
//! A verbatim copy of the context's own earlier span, continued from the
//! longest suffix match: when the match is long enough the copy replaces the
//! draft tree (verified as a chain — a degenerate tree), and each whole-taken
//! copy doubles the next one's window (15, 31, 63, ... rows). Wrong proposals
//! cost rows, never bytes.

const std = @import("std");

pub const MIN_MATCH: usize = 6;
pub const CONFIDENT_MATCH: usize = 24;
pub const COPY_MATCH: usize = 8;

pub const SuffixCopy = struct {
    /// Backward-match limit (TF max_extension). Matches longer than this
    /// truncate; proposals are at most `max_extension` tokens.
    max_extension: usize = 64,
    /// Consecutive all-rejected rounds before the proposer goes silent.
    silence_rounds: usize = 16,
    reject_window: usize = 4,
    min_match: usize = MIN_MATCH,
    copy_match: usize = COPY_MATCH,

    silent_for: usize = 0,
    recent_rejects: [16]bool = @splat(false),
    recent_len: usize = 0,
    /// Window-doubling level: 0 → budget, 1 → 2x, ... (TF _copy_level)
    copy_level: u32 = 0,
    pub var last_match: usize = 0; // module-level observation seam for tests
    pub var last_confident: bool = false;

    /// Longest backward match of `context[end..]`'s tail against the context
    /// tail before it: tokens at end-1, end-2, ... vs len-1, len-2, ...
    fn matchLength(self: *const SuffixCopy, context: []const u32, end: usize) usize {
        const limit = @min(self.max_extension, end);
        var length: usize = 0;
        while (length < limit and context[end - 1 - length] == context[context.len - 1 - length]) {
            length += 1;
            if (context.len - 1 - length == 0) break;
        }
        return length;
    }

    /// Find the best earlier occurrence of the context's suffix and return the
    /// copy continuation (the tokens that followed it). Caller owns nothing:
    /// the slice points into `context`. `null` = no backed proposal.
    pub fn propose(self: *SuffixCopy, context: []const u32, budget: usize) ?[]const u32 {
        SuffixCopy.last_match = 0;
        SuffixCopy.last_confident = false;
        if (budget == 0 or context.len < 4) return null;
        if (self.silent_for > 0) {
            self.silent_for -= 1;
            return null;
        }
        // Trigram key = last 3 tokens; scan occurrences latest-first, keep
        // the longest backward match (ties to the most recent).
        if (context.len < 3) return null;
        const n = context.len;
        var best_end: usize = 0;
        var best_len: usize = 0;
        var i: usize = n - 3; // candidate match START positions, walking back
        while (i > 0) {
            i -= 1;
            if (context[i] == context[n - 3] and context[i + 1] == context[n - 2] and context[i + 2] == context[n - 1]) {
                const end = i + 3;
                if (end >= n) continue;
                const length = self.matchLength(context, end);
                if (length > best_len) {
                    best_len = length;
                    best_end = end;
                    if (length >= self.max_extension) break;
                }
            }
        }
        SuffixCopy.last_match = best_len;
        if (best_len < self.min_match) return null;
        SuffixCopy.last_confident = best_len >= CONFIDENT_MATCH;
        const take = @min(budget, context.len - best_end);
        return context[best_end .. best_end + take];
    }

    /// Observe acceptance: `proposed` rows judged, `accepted` rows taken.
    pub fn observe(self: *SuffixCopy, proposed: usize, accepted: usize) void {
        if (proposed == 0) return;
        if (accepted == proposed) {
            // whole-taken copy: earn a wider window next time (TF doubling)
            self.copy_level +|= 1;
        } else {
            self.copy_level = 0;
        }
        const all_rejected = accepted == 0;
        if (self.recent_len == self.recent_rejects.len) {
            std.mem.copyForwards(bool, self.recent_rejects[0 .. self.recent_rejects.len - 1], self.recent_rejects[1..]);
            self.recent_len -= 1;
        }
        self.recent_rejects[self.recent_len] = all_rejected;
        self.recent_len += 1;
        if (self.recent_len >= self.reject_window) {
            var all = true;
            for (self.recent_rejects[0..self.recent_len]) |r| all = all and r;
            if (all) {
                self.silent_for = self.silence_rounds;
                self.recent_len = 0;
            }
        }
    }
};

test "suffix copy: longest match wins, continuation returned" {
    var sc = SuffixCopy{};
    // "2 3 4 16 17 18" occurs at 1..6 and again at 8..13; tail trigram
    // (16,17,18) matches back 6 tokens at the earlier span; the copy
    // continues with what followed it there.
    const ctx2 = [_]u32{ 5, 2, 3, 4, 16, 17, 18, 9, 2, 3, 4, 16, 17, 18 };
    const got = sc.propose(&ctx2, 8);
    try std.testing.expect(got != null);
    try std.testing.expectEqualSlices(u32, &.{ 9, 2, 3, 4, 16, 17, 18 }, got.?);
    try std.testing.expectEqual(@as(usize, 6), SuffixCopy.last_match);
}

test "suffix copy: no match under min_match returns null" {
    var sc = SuffixCopy{};
    const ctx = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    try std.testing.expect(sc.propose(&ctx, 8) == null);
    try std.testing.expectEqual(@as(usize, 0), SuffixCopy.last_match);
}

test "suffix copy: reject streak silences, then recovers" {
    var sc = SuffixCopy{ .silence_rounds = 3 };
    sc.observe(4, 0);
    sc.observe(4, 0);
    sc.observe(4, 0);
    sc.observe(4, 0);
    // silenced now
    const ctx = [_]u32{ 1, 2, 3, 4, 5, 3, 4, 5, 6, 7 };
    try std.testing.expect(sc.propose(&ctx, 8) == null);
    // after silence expires, proposals resume
    var i: usize = 0;
    while (i < 3) : (i += 1) _ = sc.propose(&ctx, 8);
    try std.testing.expect(sc.silent_for == 0);
}

test "suffix copy: window doubling on whole-taken, reset on partial" {
    var sc = SuffixCopy{};
    sc.observe(8, 8);
    try std.testing.expectEqual(@as(u32, 1), sc.copy_level);
    sc.observe(8, 8);
    try std.testing.expectEqual(@as(u32, 2), sc.copy_level);
    sc.observe(8, 3);
    try std.testing.expectEqual(@as(u32, 0), sc.copy_level);
}
