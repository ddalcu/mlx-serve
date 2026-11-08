//! Free seed rows on the stream's residency policy (`policy.zig`), MLX-free.

const std = @import("std");
const policy = @import("policy.zig");
const LayerPolicy = policy.LayerPolicy;
const ReadAhead = LayerPolicy.ReadAhead;

const testing = std.testing;

/// A layer of 8 experts and 4 rows as the load's usage seed leaves it: experts 0..3 in rows 0..3 (hottest first), the
/// prompt state forgotten (`Stream.seedResidents`), and with `free` its rows marked (`Stream.freeSeed`).
fn seeded(free: bool) !LayerPolicy {
    var p = try LayerPolicy.init(testing.allocator, 8, 4);
    errdefer p.deinit(testing.allocator);
    var out: [4]ReadAhead = undefined;
    try testing.expectEqual(@as(usize, 4), p.admitReadAhead(&.{ 0, 1, 2, 3 }, &out).len);
    _ = p.forgetPrompt();
    if (free) try testing.expectEqual(@as(u32, 4), p.markSeedFree());
    return p;
}

fn residents(p: *const LayerPolicy) [4]u16 {
    var r: [4]u16 = undefined;
    for (&r, 0..) |*e, s| e.* = p.slot_to_expert[s];
    return r;
}

test "seed_free: the read-ahead takes the seed rows its prediction ranks after the expert it reads, none it ranks before" {
    // The prediction names every resident, as a wide prompt chunk's does: without the rule no row is free to take.
    const prediction = [_]u16{ 4, 0, 5, 1, 6, 2, 3 };
    var out: [8]ReadAhead = undefined;
    {
        var off = try seeded(false);
        defer off.deinit(testing.allocator);
        try testing.expectEqual(@as(usize, 0), off.admitReadAhead(&prediction, &out).len);
    }
    var p = try seeded(true);
    defer p.deinit(testing.allocator);
    const got = p.admitReadAhead(&prediction, &out);
    // 4 takes 3's row (ranked last), 5 takes 2's; 6 finds only 0 and 1, both ranked before it.
    try testing.expectEqualSlices(ReadAhead, &.{ .{ .expert = 4, .slot = 3 }, .{ .expert = 5, .slot = 2 } }, got);
    try testing.expectEqual([4]u16{ 0, 1, 5, 4 }, residents(&p));
}

test "seed_free: off, the read-ahead's victim is still the coldest resident its prediction left out" {
    var p = try seeded(false);
    defer p.deinit(testing.allocator);
    var plan: policy.Plan = undefined;
    p.plan(&.{ 0, 1, 2 }, .prefill, &plan);
    var out: [8]ReadAhead = undefined;
    const got = p.admitReadAhead(&.{ 6, 0 }, &out);
    // 3 is the one no route refreshed.
    try testing.expectEqualSlices(ReadAhead, &.{.{ .expert = 6, .slot = 3 }}, got);
}
