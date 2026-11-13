//! bailing_hybrid's whole-prompt plan, MLX-free: the engine's spans checked (`check`), the segments the pass runs
//! (`segmentEnd`: a span the bank's decode lane holds alone, else the run of wider spans) and each routed call's spans
//! (`callEnd`).

const std = @import("std");

/// `ends` (absolute span ends, `Forward.prompt_spans`) cover the `rows` rows from `pos0` in ascending order, and every
/// checkpoint position is one of them.
pub fn check(pos0: u64, rows: u64, ends: []const u64, cps: []const u64) !void {
    if (ends.len == 0) return error.PromptSpansEmpty;
    var prev = pos0;
    for (ends) |e| {
        if (e <= prev) return error.PromptSpansNotAscending;
        prev = e;
    }
    if (prev != pos0 + rows) return error.PromptSpansNotThePrompt;
    for (cps) |c| if (std.mem.indexOfScalar(u64, ends, c) == null) return error.PromptCheckpointOffSpan;
}

/// Whether a span of `rows` rows is a decode lane's call (`trunk.bankPhase`'s rule: at most `lane_ids` routed ids).
pub fn inLane(rows: u32, top_k: u32, lane_ids: usize) bool {
    return @as(usize, rows) * top_k <= lane_ids;
}

/// The end of the segment that starts at span `j`: the span alone when a decode lane holds it, else every wider span
/// that follows.
pub fn segmentEnd(rows: []const u32, j: usize, top_k: u32, lane_ids: usize) usize {
    if (inLane(rows[j], top_k, lane_ids)) return j + 1;
    var k = j + 1;
    while (k < rows.len and !inLane(rows[k], top_k, lane_ids)) k += 1;
    return k;
}

/// The end of the routed call that starts at span `j` of a segment's `rows`: whole spans while the call stays within
/// `cap` rows, and past it while the rows left after the call would be fewer than `tail`.
pub fn callEnd(rows: []const u32, j: usize, cap: u32, tail: u32) usize {
    var left: u64 = 0;
    for (rows[j..]) |r| left += r;
    var have: u64 = rows[j];
    var k = j + 1;
    while (k < rows.len and (have + rows[k] <= cap or left - have < tail)) : (k += 1) have += rows[k];
    return k;
}

// ── tests ──

const testing = std.testing;

test "span plan: the spans cover the prompt in order and every checkpoint ends one; anything else is refused by name" {
    try check(100, 40, &.{ 110, 130, 140 }, &.{ 110, 130 });
    try check(0, 5, &.{5}, &.{});
    try testing.expectError(error.PromptSpansEmpty, check(0, 5, &.{}, &.{}));
    try testing.expectError(error.PromptSpansNotAscending, check(100, 40, &.{ 130, 110, 140 }, &.{}));
    try testing.expectError(error.PromptSpansNotAscending, check(100, 40, &.{ 100, 140 }, &.{}));
    try testing.expectError(error.PromptSpansNotThePrompt, check(100, 40, &.{ 110, 139 }, &.{}));
    try testing.expectError(error.PromptCheckpointOffSpan, check(100, 40, &.{ 110, 140 }, &.{120}));
}

test "span plan: a span a decode lane holds is a segment alone, the wider spans between them one segment each run" {
    // A lane of 16 ids at top-2 holds 8 rows.
    const rows = [_]u32{ 12, 30, 8, 1, 20, 9 };
    var ends: [6]usize = undefined;
    var n: usize = 0;
    var j: usize = 0;
    while (j < rows.len) : (n += 1) {
        j = segmentEnd(&rows, j, 2, 16);
        ends[n] = j;
    }
    try testing.expectEqualSlices(usize, &.{ 2, 3, 4, 6 }, ends[0..n]);
    try testing.expect(inLane(8, 2, 16) and !inLane(9, 2, 16));
}

test "span plan: a routed call takes whole spans up to 16,384 rows, and a tail under 2,048 rows joins it" {
    const cap = 16_384;
    const tail = 2_048;
    // 4 x 4,096 fill the call; 2,000 left joins it, 3,000 is a call of its own.
    try testing.expectEqual(@as(usize, 5), callEnd(&.{ 4096, 4096, 4096, 4096, 2000 }, 0, cap, tail));
    const split = [_]u32{ 4096, 4096, 4096, 4096, 3000 };
    try testing.expectEqual(@as(usize, 4), callEnd(&split, 0, cap, tail));
    try testing.expectEqual(@as(usize, 5), callEnd(&split, 4, cap, tail));
    // A span past the cap is a call of its own unless what follows it is a tail.
    try testing.expectEqual(@as(usize, 1), callEnd(&.{ 20_000, 5_000 }, 0, cap, tail));
    try testing.expectEqual(@as(usize, 2), callEnd(&.{ 20_000, 100 }, 0, cap, tail));
    // The standard 18,380-token prompt's spans at 1,536 rows (stride 8,192, the tail merge, the 30-token backoff): one
    // call.
    const std18k = [_]u32{ 1536, 1536, 1536, 1536, 1536, 512, 1536, 1536, 1536, 1536, 1536, 512, 1965, 31 };
    var sum: u32 = 0;
    for (std18k) |r| sum += r;
    try testing.expectEqual(@as(u32, 18_380), sum);
    try testing.expectEqual(std18k.len, callEnd(&std18k, 0, cap, tail));
}
