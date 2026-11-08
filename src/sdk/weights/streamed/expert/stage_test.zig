//! Staging on the stream's residency policy (`policy.zig`, `PlanOpts.held_rows`), MLX-free.

const std = @import("std");
const policy = @import("policy.zig");
const LayerPolicy = policy.LayerPolicy;
const no = policy.no_expert;

const testing = std.testing;

/// A layer of 16 experts and 4 rows, experts 0..3 resident and protected by the call's seed (the call routes them
/// hottest), so the call's other experts are transient misses.
fn full() !LayerPolicy {
    var p = try LayerPolicy.init(testing.allocator, 16, 4);
    errdefer p.deinit(testing.allocator);
    var ra: [4]LayerPolicy.ReadAhead = undefined;
    try testing.expectEqual(@as(usize, 4), p.admitReadAhead(&.{ 0, 1, 2, 3 }, &ra).len);
    p.prepareSeed(&.{ 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6 });
    return p;
}

fn transientSlots(plan: *const policy.Plan, out: []u32) []u32 {
    var n: usize = 0;
    for (plan.loadsOf()) |l| if (!l.persistent) {
        out[n] = l.slot;
        n += 1;
    };
    return out[0..n];
}

test "stage held rows: a transient miss takes the row holding its record, the rest the empty rows first" {
    var p = try full();
    defer p.deinit(testing.allocator);
    var plan: policy.Plan = .{};
    var slots: [8]u32 = undefined;
    // Stock: the window's rows in order (window base 8: slots 4 + 8 + k).
    p.planWith(&.{ 4, 5, 6 }, .prefill, &plan, .{ .transient_base = 8 });
    try testing.expectEqualSlices(u32, &.{ 12, 13, 14 }, transientSlots(&plan, &slots));
    // Row 1 holds 6's record and row 3 another expert's: 6 takes row 1, 4 and 5 the empty rows 0 and 2.
    p.planWith(&.{ 4, 5, 6 }, .prefill, &plan, .{ .transient_base = 8, .held_rows = &.{ no, 6, no, 9 } });
    try testing.expectEqualSlices(u32, &.{ 12, 14, 13 }, transientSlots(&plan, &slots));
    // Every row holds a record: 6 its own, the others the first rows left (the empty-first pass finds none).
    p.planWith(&.{ 4, 5, 6 }, .prefill, &plan, .{ .transient_base = 8, .held_rows = &.{ 7, 6, 8, 9 } });
    try testing.expectEqualSlices(u32, &.{ 12, 14, 13 }, transientSlots(&plan, &slots));
    // Two rows hold the same record: one takes it, the other row stays free for the rest.
    p.planWith(&.{ 5, 4 }, .prefill, &plan, .{ .transient_base = 0, .held_rows = &.{ 4, 4, no, no } });
    try testing.expectEqualSlices(u32, &.{ 6, 4 }, transientSlots(&plan, &slots));
}
