//! A wide prompt call's staging takes only the predicted stream misses that at least `floor` rows of the prediction
//! name (the stream's `stageGroups` and `stageNext`).
//!
//! Staging (stage.zig) reads the prediction's stream misses past the seed ranks, hottest first, into the windows the
//! call's first stream groups take: the coldest experts the prediction names, so its tail is the experts a few rows
//! name, the ones the call routes least often. On a cold 16K card 786 staged records were never adopted: 10.47 GB of
//! the prompt pass's 102.6 GB. The staging plan (`plan`) stops at the first expert of the ranking (votes descending)
//! under the floor; the read-ahead into the persistent rows keeps the whole ranking (the row choice sees every named
//! expert). An expert the floor takes out that the call routes is read by its own route, as before staging.
//!
//! Read on the M5 Max (a warm server, then a1 b1 b2 a2, each a cold code-ctx16384 card then a new-text 16K slice): cold
//! 16K 1419.7/1415.2 -> 1429.0/1428.0 tok/s (+0.78%), new text 1500.8/1490.4 -> 1513.7/1497.5 (+0.67%); the texts the
//! same in every arm. Read 102.25 -> 100.27 GB cold, 70.32 -> 68.52 new.
//!
//! Exact: the floor picks which records are staged, never the routes; a record is the same bytes in any row, and a
//! wave's rows and the combine's fold do not depend on the groups' composition.

const std = @import("std");

/// A staged record's least votes.
pub const floor: u32 = 8;

/// An expert's votes in a prediction counted by expert (`votes`; empty: none known, 0).
pub fn of(votes: []const u32, e: u16) u32 {
    return if (e < votes.len) votes[e] else 0;
}

/// A record whose votes are known (`v` > 0) under the floor.
pub fn under(v: u32) bool {
    return v > 0 and v < floor;
}

/// A staging plan's stream misses into `out`: `experts` (the prediction past the seed ranks, hottest first) that `pol`
/// holds no row for (`slotOf`), up to `cap` counting the misses the floor takes out (`votes`, by expert; empty: not
/// known). The misses under the floor are not staged: the ranking's votes descend, so they are its tail. Returns how
/// many are staged.
pub fn plan(pol: anytype, experts: []const u16, votes: []const u32, cap: usize, out: []u16) usize {
    var nm: usize = 0;
    var cut: usize = 0;
    for (experts) |e| {
        if (nm + cut == cap) break;
        if (pol.slotOf(e) != null) continue;
        if (under(of(votes, e))) {
            cut += 1;
            continue;
        }
        out[nm] = e;
        nm += 1;
    }
    return nm;
}

const testing = std.testing;

test "stage_votes floor: known votes under it; unknown votes never" {
    try testing.expect(under(1) and under(7));
    try testing.expect(!under(8) and !under(9) and !under(0));
    const votes = [_]u32{ 3, 0, 12 };
    try testing.expectEqual(@as(u32, 3), of(&votes, 0));
    try testing.expectEqual(@as(u32, 12), of(&votes, 2));
    try testing.expectEqual(@as(u32, 0), of(&votes, 3));
    try testing.expectEqual(@as(u32, 0), of(&.{}, 0));
}

/// A layer's rows for `plan`'s tests: the experts in `resident` hold rows.
const FakePolicy = struct {
    resident: []const u16,
    pub fn slotOf(p: *const FakePolicy, e: u16) ?u32 {
        for (p.resident, 0..) |x, i| if (x == e) return @intCast(i);
        return null;
    }
};

test "stage_votes plan: the misses under the floor taken out, the cap counting them" {
    const pol = FakePolicy{ .resident = &.{ 2, 5 } };
    // The ranking (votes descending) and each expert's votes.
    const ranked = [_]u16{ 3, 2, 7, 1, 5, 0, 6 };
    var votes: [8]u32 = @splat(0);
    for ([_][2]u32{ .{ 3, 40 }, .{ 2, 20 }, .{ 7, 9 }, .{ 1, 8 }, .{ 5, 6 }, .{ 0, 3 }, .{ 6, 1 } }) |x| votes[x[0]] = x[1];
    var out: [8]u16 = undefined;
    try testing.expectEqualSlices(u16, &.{ 3, 7, 1 }, out[0..plan(&pol, &ranked, &votes, 10, &out)]);
    // The cap counts the misses the floor takes out: 3, 7, 1 staged and 0 cut fill a cap of 4; 6 is past it.
    try testing.expectEqual(@as(usize, 3), plan(&pol, &ranked, &votes, 4, &out));
    // Votes not known: nothing is taken out.
    try testing.expectEqualSlices(u16, &.{ 3, 7, 1, 0, 6 }, out[0..plan(&pol, &ranked, &.{}, 10, &out)]);
}
