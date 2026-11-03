//! The residency policy's tests (`sdk_ext.expert.policy`), kept in the package's test root under their names.

const std = @import("std");
const env = @import("mlx_host").env;
const policy = @import("../sdk_ext.zig").expert.policy;
const Phase = policy.Phase;
const max_route_ids = policy.max_route_ids;
const no_expert = policy.no_expert;
const no_slot = policy.no_slot;
const Load = policy.Load;
const Eviction = policy.Eviction;
const Plan = policy.Plan;
const LayerPolicy = policy.LayerPolicy;
const rankHottest = policy.rankHottest;
const boundedParts = policy.boundedParts;

// ── Tests ──

const testing = std.testing;
const expert_bank = @import("bank.zig");

/// Every plan leaves a bijective residency map and serves each id exactly as
/// its hit or load says.
fn checkPlan(p: *const LayerPolicy, ids: []const u16, out: *const Plan, transient: u32) !void {
    var occupied: u32 = 0;
    for (p.slot_to_expert[0..p.n_experts], 0..) |e, s| {
        if (e == no_expert) continue;
        try testing.expect(s < p.capacity);
        try testing.expectEqual(@as(u32, @intCast(s)), p.expert_to_slot[e]);
        occupied += 1;
    }
    try testing.expectEqual(p.occupancy, occupied);
    try testing.expect(p.occupancy <= p.capacity);
    for (out.hitsOf()) |h| try testing.expect(p.expert_to_slot[h] != no_slot);
    var transient_seen: [max_route_ids]bool = @splat(false);
    for (out.loadsOf()) |l| {
        if (l.persistent) {
            try testing.expectEqual(l.slot, p.expert_to_slot[l.expert]);
        } else {
            try testing.expect(l.slot >= p.capacity and l.slot < p.capacity + transient);
            try testing.expect(!transient_seen[l.slot - p.capacity]);
            transient_seen[l.slot - p.capacity] = true;
            try testing.expectEqual(no_slot, p.expert_to_slot[l.expert]);
        }
    }
    for (out.evictionsOf()) |ev| {
        try testing.expectEqual(no_slot, p.expert_to_slot[ev.previous]);
        for (out.hitsOf()) |h| try testing.expect(h != ev.previous);
    }
    try testing.expectEqual(out.n_hits + out.n_misses, @as(u32, @intCast(countUnique(ids))));
    try testing.expectEqual(out.n_misses, out.n_loads);
    for (ids, out.slotsOf()) |e, s| {
        if (p.expert_to_slot[e] != no_slot) {
            try testing.expectEqual(p.expert_to_slot[e], s);
        } else {
            var found = false;
            for (out.loadsOf()) |l| if (l.expert == e) {
                try testing.expectEqual(l.slot, s);
                found = true;
            };
            try testing.expect(found);
        }
    }
}

fn countUnique(ids: []const u16) usize {
    var n: usize = 0;
    for (ids, 0..) |e, i| {
        if (std.mem.indexOfScalar(u16, ids[0..i], e) == null) n += 1;
    }
    return n;
}

test "dsv41 policy: decode keeps what the transitions predict" {
    // One persistent slot. After 1 -> 2 has been seen, a route of [1] finds 2
    // predicted next and serves 1 from the transient scratch instead.
    var p = try LayerPolicy.init(testing.allocator, 8, 1);
    defer p.deinit(testing.allocator);
    var out: Plan = .{};
    p.plan(&.{1}, .decode, &out);
    try testing.expectEqualSlices(Load, &.{.{ .expert = 1, .slot = 0, .persistent = true }}, out.loadsOf());
    // [2]: prediction 0 for both; window 0.2 each; recency 1/2 vs 1 -> 2 wins.
    p.plan(&.{2}, .decode, &out);
    try testing.expectEqualSlices(Load, &.{.{ .expert = 2, .slot = 0, .persistent = true }}, out.loadsOf());
    try testing.expectEqualSlices(Eviction, &.{.{ .slot = 0, .previous = 1, .next = 2 }}, out.evictionsOf());
    // [1]: 2 scores 0.7 + 0.1 + 0.05, 1 scores 0.2 + 0.1 -> 1 goes transient.
    p.plan(&.{ 1, 1 }, .decode, &out);
    try testing.expectEqualSlices(Load, &.{.{ .expert = 1, .slot = 1, .persistent = false }}, out.loadsOf());
    try testing.expectEqualSlices(u32, &.{ 1, 1 }, out.slotsOf());
    try testing.expectEqual(@as(u32, 0), out.n_evictions);
    try testing.expectEqual(@as(?u32, 0), p.slotOf(2));
}

test "dsv41 policy: decode fills empty slots in slot order and never evicts a hit" {
    var p = try LayerPolicy.init(testing.allocator, 32, 4);
    defer p.deinit(testing.allocator);
    var out: Plan = .{};
    p.plan(&.{ 7, 3, 7 }, .decode, &out);
    try testing.expectEqualSlices(Load, &.{
        .{ .expert = 7, .slot = 0, .persistent = true },
        .{ .expert = 3, .slot = 1, .persistent = true },
    }, out.loadsOf());
    try testing.expectEqualSlices(u32, &.{ 0, 1, 0 }, out.slotsOf());
    // A deterministic pseudo-random trace: every plan keeps the invariants.
    var rng = std.Random.DefaultPrng.init(41);
    const r = rng.random();
    var ids: [max_route_ids]u16 = undefined;
    for (0..400) |step| {
        const n = r.intRangeAtMost(usize, 1, 18);
        for (ids[0..n]) |*e| e.* = @intCast(r.intRangeLessThan(u32, 0, if (step % 3 == 0) 32 else 12));
        const before = p.slot_to_expert[0..4].*;
        p.plan(ids[0..n], .decode, &out);
        try checkPlan(&p, ids[0..n], &out, max_route_ids);
        for (out.hitsOf()) |h| try testing.expect(std.mem.indexOfScalar(u16, &before, h) != null);
    }
}

test "dsv41 policy: a read-ahead admits predicted experts to empty slots, unprotected; the seed keeps the ones it chose" {
    const a = testing.allocator;
    var p = try LayerPolicy.init(a, 16, 4);
    defer p.deinit(a);
    var buf: [8]LayerPolicy.ReadAhead = undefined;
    // Predicted hottest first: 5, 9, 1, 12, 7 (four slots: 7 does not fit); a repeat is skipped.
    const got = p.admitReadAhead(&.{ 5, 9, 5, 1, 12, 7 }, &buf);
    try testing.expectEqual(@as(usize, 4), got.len);
    for (got, [_]u16{ 5, 9, 1, 12 }) |r, e| {
        try testing.expectEqual(e, r.expert);
        try testing.expectEqual(r.slot, p.slotOf(e).?);
        try testing.expect(!p.protected.isSet(e));
    }
    // Full, and the prediction names every resident: nothing more is admitted (no resident it left out to replace).
    try testing.expectEqual(@as(usize, 0), p.admitReadAhead(&.{ 5, 9, 1, 12, 7, 3 }, &buf).len);
    // The call's true counts: 5 x3, 12 x2, 3 x2, 9 x1 -> seed = {5, 12, 3, 9} (capacity 4, nothing protected).
    p.prepareSeed(&.{ 5, 5, 5, 12, 12, 3, 3, 9 });
    // The read-ahead experts the seed chose are re-protected (resident: hits); 3 is to be admitted;
    // 1 (mispredicted) stays an unprotected, evictable resident.
    try testing.expect(p.protected.isSet(5) and p.protected.isSet(12) and p.protected.isSet(9));
    try testing.expect(!p.protected.isSet(1) and p.slotOf(1) != null);
    try testing.expect(p.seed.isSet(3));
    // The first prefill route: 5 and 12 hit, 3 is admitted into 1's slot (the one probationary resident).
    var plan_: Plan = .{};
    const slot1 = p.slotOf(1).?;
    p.plan(&.{ 5, 12, 3 }, .prefill, &plan_);
    try testing.expectEqual(@as(u32, 2), plan_.n_hits);
    try testing.expectEqual(@as(u32, 1), plan_.n_loads);
    try testing.expectEqual(slot1, plan_.loads[0].slot);
    try testing.expectEqual(@as(?u32, null), p.slotOf(1));
}

test "dsv41 policy: a read-ahead with no empty row replaces the coldest residents its prediction left out, never a protected, pinned or held one" {
    const a = testing.allocator;
    var p = try LayerPolicy.init(a, 16, 4);
    defer p.deinit(a);
    var buf: [8]LayerPolicy.ReadAhead = undefined;
    // A load's seed (or the last request's residents): 5, 9, 1, 12 in slots 0..3, recency in that order (5 the coldest).
    try testing.expectEqual(@as(usize, 4), p.admitReadAhead(&.{ 5, 9, 1, 12 }, &buf).len);
    try testing.expectEqual(@as(u32, 4), p.occupancy);
    // The next prompt predicts 9, 7, 3, 12: 9 and 12 are residents (hits to be, never victims); 7 takes the coldest
    // resident not predicted (5, slot 0), 3 the next (1, slot 2); 9 and 12 keep their slots.
    const got = p.admitReadAhead(&.{ 9, 7, 3, 12 }, &buf);
    try testing.expectEqualSlices(LayerPolicy.ReadAhead, &.{ .{ .expert = 7, .slot = 0 }, .{ .expert = 3, .slot = 2 } }, got);
    try testing.expectEqual(@as(u32, 4), p.occupancy);
    try testing.expectEqual(@as(?u32, null), p.slotOf(5));
    try testing.expectEqual(@as(?u32, null), p.slotOf(1));
    try testing.expectEqual(@as(?u32, 1), p.slotOf(9));
    try testing.expectEqual(@as(?u32, 3), p.slotOf(12));
    for ([_]u16{ 7, 3, 9, 12 }) |e| try testing.expect(!p.protected.isSet(e));
    // A protected resident (the call's seed) and a held slot (a live route's) are never victims: with 9 protected
    // and slot 3 (12's) held, a prediction of 2, 4, 6 replaces 7 and 3 (the coldest unprotected, unheld) and stops.
    p.prepareSeed(&.{ 9, 9, 9 });
    try testing.expect(p.protected.isSet(9));
    const got2 = p.admitReadAheadHeld(&.{ 2, 4, 6 }, &.{3}, &buf);
    try testing.expectEqual(@as(usize, 2), got2.len);
    try testing.expectEqual(@as(?u32, 1), p.slotOf(9));
    try testing.expectEqual(@as(?u32, 3), p.slotOf(12));
    try testing.expectEqual(@as(?u32, null), p.slotOf(7));
    try testing.expectEqual(@as(?u32, null), p.slotOf(3));
    try testing.expect(p.slotOf(2) != null and p.slotOf(4) != null and p.slotOf(6) == null);
    // The held mark is the admission's only: a later admission may take slot 3.
    try testing.expectEqual(@as(usize, 1), p.admitReadAhead(&.{8}, &buf).len);
    try testing.expectEqual(@as(?u32, null), p.slotOf(12));
    // Nothing of the plan scratch stays set.
    try testing.expectEqual(@as(usize, 0), p.retained.count());
    try testing.expectEqual(@as(usize, 0), p.held.count());
}

test "dsv41 policy: P1's predicted counts rank as the seed ranks (count descending, ties by id), the same from any count order" {
    const a = testing.allocator;
    // Counts: 7 x4, 2 x3, 9 x3, 4 x1, 11 x1, 0 x1; never predicted: every other expert.
    var counts: [16]u32 = @splat(0);
    for ([_]u16{ 9, 7, 2, 11, 7, 9, 4, 7, 2, 0, 9, 2, 7 }) |e| counts[e] += 1;
    var out: [16]u16 = undefined;
    const ranked = rankHottest(&counts, &out);
    try testing.expectEqualSlices(u16, &.{ 7, 2, 9, 0, 4, 11 }, ranked);
    // The same counts reached in another order rank the same (a function of the counts alone).
    var again: [16]u32 = @splat(0);
    for ([_]u16{ 0, 2, 2, 2, 4, 7, 7, 7, 7, 9, 9, 9, 11 }) |e| again[e] += 1;
    var out2: [16]u16 = undefined;
    try testing.expectEqualSlices(u16, ranked, rankHottest(&again, &out2));
    // prepareSeed chooses the ranking's head: capacity 4 seeds {7, 2, 9, 0}.
    var p = try LayerPolicy.init(a, 16, 4);
    defer p.deinit(a);
    p.prepareSeed(&.{ 9, 7, 2, 11, 7, 9, 4, 7, 2, 0, 9, 2, 7 });
    for (0..16) |e| try testing.expectEqual(std.mem.indexOfScalar(u16, ranked[0..4], @intCast(e)) != null, p.seed.isSet(e));
}

test "dsv41 policy: prefill admits the seed first and never evicts it" {
    var p = try LayerPolicy.init(testing.allocator, 16, 2);
    defer p.deinit(testing.allocator);
    // Prompt frequency: 5 x3, 9 x2, 1 x1 -> seed = {5, 9}.
    p.prepareSeed(&.{ 5, 9, 1, 5, 9, 5 });
    var out: Plan = .{};
    p.plan(&.{ 1, 9, 5 }, .prefill, &out);
    // Seed first, least frequent first; the pool is then full of protected
    // experts, so 1 overflows to the transient scratch.
    try testing.expectEqualSlices(u16, &.{ 9, 5, 1 }, out.missesOf());
    try testing.expectEqualSlices(Load, &.{
        .{ .expert = 9, .slot = 0, .persistent = true },
        .{ .expert = 5, .slot = 1, .persistent = true },
        .{ .expert = 1, .slot = 2, .persistent = false },
    }, out.loadsOf());
    try checkPlan(&p, &.{ 1, 9, 5 }, &out, max_route_ids);
    // A probationary resident is evicted before any protected one.
    var q = try LayerPolicy.init(testing.allocator, 16, 2);
    defer q.deinit(testing.allocator);
    q.prepareSeed(&.{3});
    q.plan(&.{ 3, 4 }, .prefill, &out);
    q.plan(&.{6}, .prefill, &out);
    try testing.expectEqualSlices(Eviction, &.{.{ .slot = 1, .previous = 4, .next = 6 }}, out.evictionsOf());
}

test "dsv41 policy: growth adds empty slots used before any eviction" {
    var p = try LayerPolicy.init(testing.allocator, 16, 2);
    defer p.deinit(testing.allocator);
    var out: Plan = .{};
    p.plan(&.{ 1, 2 }, .prefill, &out);
    try p.grow(4);
    try testing.expectError(error.InvalidCapacity, p.grow(3));
    p.plan(&.{ 1, 3, 4 }, .decode, &out);
    try testing.expectEqualSlices(Load, &.{
        .{ .expert = 3, .slot = 2, .persistent = true },
        .{ .expert = 4, .slot = 3, .persistent = true },
    }, out.loadsOf());
    try testing.expectEqual(@as(u32, 0), out.n_evictions);
    try testing.expectEqual(@as(?u32, 1), p.slotOf(2));
}

test "dsv41 policy: shrink forgets every expert past the new capacity; plans stay within it; it grows back" {
    var p = try LayerPolicy.init(testing.allocator, 16, 2);
    defer p.deinit(testing.allocator);
    var out: Plan = .{};
    p.plan(&.{ 1, 2 }, .prefill, &out);
    try p.grow(4);
    p.plan(&.{ 1, 3, 4 }, .decode, &out);
    try testing.expectEqual(@as(?u32, 2), p.slotOf(3));
    try testing.expectEqual(@as(?u32, 3), p.slotOf(4));
    try testing.expectError(error.InvalidCapacity, p.shrink(5));
    try p.shrink(2);
    try testing.expectEqual(@as(u32, 2), p.capacity);
    // The freed rows' experts are forgotten; the kept ones stay resident.
    try testing.expectEqual(@as(?u32, null), p.slotOf(3));
    try testing.expectEqual(@as(?u32, null), p.slotOf(4));
    try testing.expectEqual(@as(?u32, 0), p.slotOf(1));
    try testing.expectEqual(@as(?u32, 1), p.slotOf(2));
    try testing.expectEqual(@as(u32, 2), p.occupancy);
    for (p.slot_to_expert[2..4]) |e| try testing.expectEqual(no_expert, e);
    // A later prompt plans within the kept slots (a persistent load never lands on a freed row).
    p.prepareSeed(&.{ 3, 3, 5 });
    p.plan(&.{ 3, 5 }, .prefill, &out);
    try checkPlan(&p, &.{ 3, 5 }, &out, max_route_ids);
    for (out.loadsOf()) |l| if (l.persistent) try testing.expect(l.slot < 2);
    // And the phase change grows it back: the freed slots empty again, used before any eviction.
    try p.grow(4);
    p.plan(&.{ 7, 8 }, .decode, &out);
    try testing.expectEqual(@as(u32, 0), out.n_evictions);
}

test "dsv41 policy: forgetPrompt keeps the residents in their slots and clears the prompt state; the first victim is the resident decode used least recently" {
    var p = try LayerPolicy.init(testing.allocator, 16, 4);
    defer p.deinit(testing.allocator);
    var out: Plan = .{};
    // The prompt: 1, 2, 3, 4 in slots 0..3, protected by its seed.
    p.prepareSeed(&.{ 1, 2, 3, 4 });
    p.plan(&.{ 1, 2, 3, 4 }, .prefill, &out);
    try testing.expectEqual(@as(usize, 4), p.protected.count());
    // Decode at 6 rows: 5 into a grown slot; 1, 2 and 4 routed last (3 never).
    try p.grow(6);
    p.plan(&.{ 5, 1 }, .decode, &out);
    p.plan(&.{ 2, 4, 1 }, .decode, &out);
    try testing.expectEqual(@as(?u32, 4), p.slotOf(5));
    // The reverse change to 4 rows: the grown slot's resident goes, the prompt slots' stay where they are.
    try p.shrink(4);
    try testing.expectEqual(@as(u32, 4), p.forgetPrompt());
    for ([_]u16{ 1, 2, 3, 4 }, 0..) |e, s| try testing.expectEqual(@as(?u32, @intCast(s)), p.slotOf(e));
    try testing.expectEqual(@as(?u32, null), p.slotOf(5));
    // The prompt state is forgetAll's: no protection, no seed, no prompt counts.
    try testing.expectEqual(@as(usize, 0), p.protected.count());
    try testing.expectEqual(@as(usize, 0), p.seed.count());
    try testing.expectEqual(@as(u32, 0), p.seed_ranks);
    for (p.prefill_freq) |f| try testing.expectEqual(@as(u32, 0), f);
    // Recency is decode's last use: 1, 2 and 4 at the second decode route (epoch 2), 3 never.
    for ([_]u16{ 1, 2, 4 }) |e| try testing.expectEqual(@as(u64, 3), p.recency[e]);
    try testing.expectEqual(@as(u64, 0), p.recency[3]);
    try testing.expectEqual(@as(u64, 3), p.clock);
    // A continuation's prompt over 2, 4 and 7: two hits, and 7 takes 3's slot (decode never used 3), not slot 0.
    p.prepareSeed(&.{ 2, 4, 7 });
    p.plan(&.{ 2, 4, 7 }, .prefill, &out);
    try checkPlan(&p, &.{ 2, 4, 7 }, &out, max_route_ids);
    try testing.expectEqualSlices(u16, &.{ 2, 4 }, out.hitsOf());
    try testing.expectEqualSlices(u16, &.{7}, out.missesOf());
    try testing.expectEqualSlices(Eviction, &.{.{ .slot = 2, .previous = 3, .next = 7 }}, out.evictionsOf());
    try testing.expectEqual(@as(?u32, 0), p.slotOf(1));
}

test "dsv41 policy: bounded parts cut at the last physical gap" {
    var ends: [8]u32 = undefined;
    // 0-10-20 contiguous | 35-45 | 60-70.
    const off = [_]u64{ 0, 10, 20, 35, 45, 60, 70 };
    const len: [7]u64 = @splat(10);
    try testing.expectEqualSlices(u32, &.{ 3, 5, 7 }, boundedParts(&off, &len, 3, &ends));
    // Padded records never touch: plain runs of three.
    const padded: [7]u64 = @splat(9);
    try testing.expectEqualSlices(u32, &.{ 3, 6, 7 }, boundedParts(&off, &padded, 3, &ends));
    // One contiguous run longer than the bound is cut at the bound.
    const run = [_]u64{ 0, 10, 20, 30, 40 };
    try testing.expectEqualSlices(u32, &.{ 3, 5 }, boundedParts(&run, len[0..5], 3, &ends));
    try testing.expectEqual(@as(usize, 0), boundedParts(&.{}, &.{}, 3, &ends).len);
}

/// One plan as the reference runtime's dump_phase1_route_fixture.py records it from the
/// Python bank. loads: [expert, slot, persistent, physical skip];
/// evictions: [slot, previous, next]; parts: decode miss parts by expert.
pub const FixPlan = struct {
    ids: []const u16,
    slots: []const u32,
    hits: []const u16,
    misses: []const u16,
    loads: []const [4]u32,
    evictions: []const [3]u32,
    parts: []const []const u16,
};

/// A one-layer replay for the real-bank test: small rows, per-row routes.
pub const BankTrace = struct {
    layer: u32,
    seed: []const u16,
    prefill_rows: u32,
    decode_rows: u32,
    transient: u32,
    prefill: []const FixPlan,
    routes: []const FixPlan,
};

pub fn expectPlan(p: *const Plan, want: FixPlan) !void {
    try testing.expectEqualSlices(u32, want.slots, p.slotsOf());
    try testing.expectEqualSlices(u16, want.hits, p.hitsOf());
    try testing.expectEqualSlices(u16, want.misses, p.missesOf());
    try testing.expectEqual(want.loads.len, p.n_loads);
    for (want.loads, p.loadsOf()) |w, l| {
        try testing.expectEqual(w[0], l.expert);
        try testing.expectEqual(w[1], l.slot);
        try testing.expectEqual(w[2] != 0, l.persistent);
    }
    try testing.expectEqual(want.evictions.len, p.n_evictions);
    for (want.evictions, p.evictionsOf()) |w, ev| {
        try testing.expectEqual(w[0], ev.slot);
        try testing.expectEqual(w[1], ev.previous);
        try testing.expectEqual(w[2], ev.next);
    }
}

/// The decode miss parts of `p` (by expert), in the real bank's placement.
fn partsOf(p: *const Plan, table: []const expert_bank.Layer, layer: u32, out_experts: *[max_route_ids]u16, ends: *[max_route_ids]u32) []u32 {
    const n = p.n_loads;
    for (p.loadsOf(), 0..) |l, i| out_experts[i] = l.expert;
    std.sort.insertion(u16, out_experts[0..n], {}, std.sort.asc(u16));
    var offsets: [max_route_ids]u64 = undefined;
    var lengths: [max_route_ids]u64 = undefined;
    const t = table[layer];
    for (out_experts[0..n], 0..) |e, i| {
        offsets[i] = t.base_offset + @as(u64, e) * t.record_bytes;
        lengths[i] = t.logical_bytes;
    }
    return boundedParts(offsets[0..n], lengths[0..n], 3, ends);
}

/// ExpertSlotPool._prepare_load's reuse rule over physical rows: a load whose
/// row already holds (layer, expert) is not read.
const Physical = struct {
    persistent: []u16,
    experts: u32,
    transient: [max_route_ids]?[2]u32 = @splat(null),

    fn skip(ph: *Physical, layer: u32, capacity: u32, w: [4]u32) bool {
        if (w[2] != 0) {
            const o = &ph.persistent[layer * ph.experts + w[1]];
            const held = o.* == w[0];
            o.* = @intCast(w[0]);
            return held;
        }
        const o = &ph.transient[w[1] - capacity];
        const held = if (o.*) |h| h[0] == layer and h[1] == w[0] else false;
        o.* = .{ layer, w[0] };
        return held;
    }
};

// DSV41_PHASE1_ROUTE_FIXTURE=<json from the reference runtime's dump_phase1_route_fixture.py>
test "dsv41 policy: the recorded trace plans exactly like the Python bank" {
    const path = std.mem.span(env.get(.DSV41_PHASE1_ROUTE_FIXTURE) orelse return error.SkipZigTest);
    const a = testing.allocator;
    const Fixture = struct {
        layers: u32,
        experts: u32,
        transient: u32,
        records_per_part: u32,
        prefill_capacity: []const u32,
        decode_capacity: []const u32,
        resident0: []const []const u16,
        seed_plans: []const []const FixPlan,
        routes: []const FixPlan,
        final_slot_to_expert: []const []const i32,
    };
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(64 << 20));
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(Fixture, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const f = parsed.value;
    try testing.expectEqual(@as(u32, max_route_ids), f.transient);
    try testing.expectEqual(@as(u32, 3), f.records_per_part);

    var table: [40]expert_bank.Layer = undefined;
    const k3: [40]u32 = @splat(3);
    try testing.expectEqual(@as(u32, 40), f.layers);
    _ = expert_bank.layerTable(&k3, 5120, 2304, f.experts, &table).?;

    const policies = try a.alloc(LayerPolicy, f.layers);
    defer a.free(policies);
    var n_init: usize = 0;
    defer for (policies[0..n_init]) |*p| p.deinit(a);
    // The Python pool's physical owners, for its read skips.
    var phys: Physical = .{ .persistent = try a.alloc(u16, f.layers * f.experts), .experts = f.experts };
    defer a.free(phys.persistent);
    @memset(phys.persistent, no_expert);

    var out: Plan = .{};
    var n_plans: usize = 0;
    var n_reads: usize = 0;
    var n_skips: usize = 0;
    for (0..f.layers) |l| {
        policies[l] = try LayerPolicy.init(a, f.experts, f.prefill_capacity[l]);
        n_init += 1;
        const p = &policies[l];
        p.prepareSeed(f.resident0[l]);
        for (f.seed_plans[l]) |want| {
            p.plan(want.ids, .prefill, &out);
            try expectPlan(&out, want);
            for (want.loads) |w| try testing.expectEqual(w[3] != 0, phys.skip(@intCast(l), p.capacity, w));
            n_plans += 1;
        }
    }
    for (policies, f.decode_capacity) |*p, cap| try p.grow(cap);

    var parts_experts: [max_route_ids]u16 = undefined;
    var ends: [max_route_ids]u32 = undefined;
    for (f.routes, 0..) |want, i| {
        const l: u32 = @intCast(i % f.layers);
        const p = &policies[l];
        p.plan(want.ids, .decode, &out);
        expectPlan(&out, want) catch |e| {
            std.debug.print("route {d} (cycle {d}, layer {d}) differs\n", .{ i, i / f.layers, l });
            return e;
        };
        for (want.loads) |w| {
            const skip = phys.skip(l, p.capacity, w);
            try testing.expectEqual(w[3] != 0, skip);
            if (skip) n_skips += 1 else n_reads += 1;
        }
        const got = partsOf(&out, &table, l, &parts_experts, &ends);
        try testing.expectEqual(want.parts.len, got.len);
        var start: u32 = 0;
        for (want.parts, got) |wp, end| {
            try testing.expectEqualSlices(u16, wp, parts_experts[start..end]);
            start = end;
        }
        n_plans += 1;
    }
    for (policies, f.final_slot_to_expert) |*p, want| {
        try testing.expectEqual(want.len, p.capacity);
        for (want, p.slot_to_expert[0..p.capacity]) |w, e| {
            try testing.expectEqual(w, if (e == no_expert) @as(i32, -1) else @as(i32, e));
        }
    }
    std.debug.print("policy parity: {d} plans ({d} decode routes) equal the Python bank's; {d} reads, {d} skipped\n", .{ n_plans, f.routes.len, n_reads, n_skips });
}

test "dsv41 policy: a plan beside a live route never evicts its held slots and loads transient at its window" {
    var p = try LayerPolicy.init(testing.allocator, 32, 4);
    defer p.deinit(testing.allocator);
    var out: Plan = undefined;
    p.plan(&.{ 0, 1, 2, 3 }, .prefill, &out);
    try testing.expectEqual(@as(u32, 4), out.n_persistent);
    // Slots 0..3 held: the next route's misses go transient, from row 48 of the transient rows.
    p.planWith(&.{ 4, 5 }, .prefill, &out, .{ .transient_base = 48, .held = &.{ 0, 1, 2, 3 } });
    try testing.expectEqual(@as(u32, 0), out.n_persistent);
    try testing.expectEqual(@as(u32, 4 + 48), out.loads[0].slot);
    try testing.expectEqual(@as(u32, 4 + 49), out.loads[1].slot);
    // Two held: the other two are victims.
    p.planWith(&.{ 6, 7 }, .prefill, &out, .{ .held = &.{ 0, 1 } });
    try testing.expectEqual(@as(u32, 2), out.n_persistent);
    for (out.loadsOf()) |l| try testing.expect(l.slot == 2 or l.slot == 3);
}

// ── Edges and accounting identities (synthetic traces; capacity 0, 1 and full; both phases; both planners) ──


/// Totals over a trace: every plan's lookups split into hits and misses, every miss loaded once, and the residency the
/// persistent loads and evictions leave.
const Tally = struct {
    unique: u64 = 0,
    hits: u64 = 0,
    misses: u64 = 0,
    loads: u64 = 0,
    persistent: u64 = 0,
    evictions: u64 = 0,

    fn add(t: *Tally, ids: []const u16, out: *const Plan) void {
        t.unique += countUnique(ids);
        t.hits += out.n_hits;
        t.misses += out.n_misses;
        t.loads += out.n_loads;
        t.persistent += out.n_persistent;
        t.evictions += out.n_evictions;
    }

    fn expectIdentities(t: *const Tally, occupancy: u32) !void {
        try testing.expectEqual(t.unique, t.hits + t.misses);
        try testing.expectEqual(t.misses, t.loads);
        try testing.expectEqual(@as(u64, occupancy), t.persistent - t.evictions);
    }
};

test "dsv41 policy: capacity 0 serves every id from the scratch, both phases; hits never happen and nothing is resident" {
    var p = try LayerPolicy.init(testing.allocator, 16, 0);
    defer p.deinit(testing.allocator);
    var out: Plan = .{};
    var t: Tally = .{};
    p.prepareSeed(&.{ 3, 3, 5 });
    try testing.expectEqual(@as(u32, 0), p.seed_ranks);
    for ([_]Phase{ .prefill, .decode, .prefill, .decode }) |phase| {
        const ids = [_]u16{ 3, 5, 3, 9 };
        p.plan(&ids, phase, &out);
        try checkPlan(&p, &ids, &out, max_route_ids);
        t.add(&ids, &out);
        try testing.expectEqual(@as(u32, 0), out.n_hits);
        try testing.expectEqual(@as(u32, 0), out.n_persistent);
        try testing.expectEqualSlices(u32, &.{ 0, 1, 0, 2 }, out.slotsOf());
    }
    try t.expectIdentities(p.occupancy);
    try testing.expectEqual(@as(u32, 0), p.occupancy);
    var ra: [4]LayerPolicy.ReadAhead = undefined;
    try testing.expectEqual(@as(usize, 0), p.admitReadAhead(&.{ 1, 2 }, &ra).len);
}

test "dsv41 policy: capacity 1 holds one expert; every plan keeps the invariants and the trace's identities, both phases" {
    for ([_]Phase{ .prefill, .decode }) |phase| {
        var p = try LayerPolicy.init(testing.allocator, 12, 1);
        defer p.deinit(testing.allocator);
        var out: Plan = .{};
        var t: Tally = .{};
        var rng = std.Random.DefaultPrng.init(if (phase == .prefill) 7 else 8);
        const r = rng.random();
        var ids: [max_route_ids]u16 = undefined;
        for (0..300) |step| {
            const n = r.intRangeAtMost(usize, 1, 12);
            for (ids[0..n]) |*e| e.* = @intCast(r.intRangeLessThan(u32, 0, if (step % 4 == 0) 3 else 12));
            if (phase == .prefill and step % 50 == 0) p.prepareSeed(ids[0..n]);
            p.plan(ids[0..n], phase, &out);
            try checkPlan(&p, ids[0..n], &out, max_route_ids);
            try testing.expect(out.n_persistent <= 1);
            t.add(ids[0..n], &out);
        }
        try t.expectIdentities(p.occupancy);
        try testing.expectEqual(@as(u32, 1), p.occupancy);
    }
}

test "dsv41 policy: prefill traces at full capacity never evict, and with held slots never evict a held one" {
    try heldTrace(.prefill);
    var full = try LayerPolicy.init(testing.allocator, 10, 10);
    defer full.deinit(testing.allocator);
    var out: Plan = .{};
    var t: Tally = .{};
    var rng = std.Random.DefaultPrng.init(99);
    const r = rng.random();
    var ids: [max_route_ids]u16 = undefined;
    for (0..200) |_| {
        const n = r.intRangeAtMost(usize, 1, 20);
        for (ids[0..n]) |*e| e.* = @intCast(r.intRangeLessThan(u32, 0, 10));
        full.plan(ids[0..n], .prefill, &out);
        try checkPlan(&full, ids[0..n], &out, max_route_ids);
        try testing.expectEqual(@as(u32, 0), out.n_evictions);
        t.add(ids[0..n], &out);
    }
    try t.expectIdentities(full.occupancy);

}

/// A random prefill trace whose every plan holds two slots of another live route (`PlanOpts.held`): none is ever a victim.
fn heldTrace(phase: Phase) !void {
    var p = try LayerPolicy.init(testing.allocator, 40, 6);
    defer p.deinit(testing.allocator);
    var out: Plan = .{};
    var t: Tally = .{};
    var rng = std.Random.DefaultPrng.init(if (phase == .prefill) 17 else 18);
    const r = rng.random();
    var ids: [max_route_ids]u16 = undefined;
    for (0..300) |step| {
        const n = r.intRangeAtMost(usize, 1, 10);
        for (ids[0..n]) |*e| e.* = @intCast(r.intRangeLessThan(u32, 0, 40));
        if (step % 40 == 0) p.prepareSeed(ids[0..n]);
        const held = [_]u32{ @intCast(step % 6), @intCast((step + 3) % 6) };
        p.planWith(ids[0..n], phase, &out, .{ .held = &held, .transient_base = 4 });
        try checkPlan(&p, ids[0..n], &out, max_route_ids + 4);
        for (out.evictionsOf()) |ev| for (held) |h| testing.expect(ev.slot != h) catch |e| {
            std.debug.print("{t} step {d}: slot {d} held, evicted {d} -> {d}\n", .{ phase, step, h, ev.previous, ev.next });
            return e;
        };
        for (out.loadsOf()) |l| if (!l.persistent) try testing.expect(l.slot >= p.capacity + 4);
        t.add(ids[0..n], &out);
    }
    try t.expectIdentities(p.occupancy);
}


test "dsv41 policy: capacities and read-ahead admissions refuse or stop at their bounds" {
    try testing.expectError(error.InvalidCapacity, LayerPolicy.init(testing.allocator, 0, 0));
    try testing.expectError(error.InvalidCapacity, LayerPolicy.init(testing.allocator, no_expert, 1));
    try testing.expectError(error.InvalidCapacity, LayerPolicy.init(testing.allocator, 4, 5));
    var p = try LayerPolicy.init(testing.allocator, 8, 2);
    defer p.deinit(testing.allocator);
    try testing.expectError(error.InvalidCapacity, p.grow(1));
    try testing.expectError(error.InvalidCapacity, p.grow(9));
    try testing.expectError(error.InvalidCapacity, p.shrink(3));
    var ra: [3]LayerPolicy.ReadAhead = undefined;
    // An id past the experts and a repeat are skipped; the admissions stop when the slots run out (the residents are the
    // prediction's own: no victim).
    const got = p.admitReadAhead(&.{ 9, 4, 4, 6, 7 }, &ra);
    try testing.expectEqualSlices(LayerPolicy.ReadAhead, &.{ .{ .expert = 4, .slot = 0 }, .{ .expert = 6, .slot = 1 } }, got);
    try testing.expectEqual(@as(u32, 2), p.occupancy);
    // A full `out` stops them too; a grown capacity takes more; forgetting returns how many were resident.
    try p.grow(5);
    try testing.expectEqual(@as(usize, 1), p.admitReadAhead(&.{ 1, 2, 3 }, ra[0..1]).len);
    try testing.expectEqual(@as(u32, 3), p.forgetAll());
    try testing.expectEqual(@as(u32, 0), p.occupancy);
    // Invalidating a non-resident expert is a no-op.
    p.invalidate(5);
    try testing.expectEqual(@as(u32, 0), p.occupancy);
}

test "dsv41 policy: bounded parts: one per part, one whole run, no records, and a cut moved back to the last gap" {
    var ends: [16]u32 = undefined;
    try testing.expectEqual(@as(usize, 0), boundedParts(&.{}, &.{}, 4, &ends).len);
    // Contiguous records: plain cuts every `per_part`.
    try testing.expectEqualSlices(u32, &.{ 2, 4, 5 }, boundedParts(&.{ 0, 10, 20, 30, 40 }, &.{ 10, 10, 10, 10, 10 }, 2, &ends));
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, boundedParts(&.{ 0, 50, 100 }, &.{ 10, 10, 10 }, 1, &ends));
    // A gap after record 1 (10 + 10 != 30): the cut at 3 moves back to 2, the run 2..4 stays whole.
    try testing.expectEqualSlices(u32, &.{ 2, 5 }, boundedParts(&.{ 0, 10, 30, 40, 50 }, &.{ 10, 10, 10, 10, 10 }, 3, &ends));
    // rankHottest: zero counts are left out; ties by id.
    var out: [6]u16 = undefined;
    try testing.expectEqualSlices(u16, &.{ 4, 1, 3 }, rankHottest(&.{ 0, 2, 0, 2, 5, 0 }, &out));
    try testing.expectEqual(@as(usize, 0), rankHottest(&.{ 0, 0 }, &out).len);
}

fn policyInitDeinit(a: std.mem.Allocator, n_experts: u32, capacity: u32) !void {
    var p = try LayerPolicy.init(a, n_experts, capacity);
    p.deinit(a);
}

test "dsv41 policy: every allocation failure of the residency policy's construction unwinds without a leak" {
    try std.testing.checkAllAllocationFailures(testing.allocator, policyInitDeinit, .{ 16, 4 });
}

// ── The rate tally (`LayerPolicy.setTally`, `sdk_ext.expert.tally`) ──

const tally = @import("../sdk_ext.zig").expert.tally;

/// The resident experts in ascending order.
fn residentsOf(p: *const LayerPolicy, buf: []u16) []u16 {
    var n: usize = 0;
    for (p.slot_to_expert[0..p.capacity]) |e| if (e != no_expert) {
        buf[n] = e;
        n += 1;
    };
    std.sort.pdq(u16, buf[0..n], {}, std.sort.asc(u16));
    return buf[0..n];
}

fn expectPins(p: *const LayerPolicy, want: []const u16) !void {
    var n: usize = 0;
    for (0..p.n_experts) |e| if (p.isPinned(@intCast(e))) {
        try testing.expect(n < want.len and want[n] == e);
        n += 1;
    };
    try testing.expectEqual(want.len, n);
}

/// Route t of a cycle over experts 3 .. 15, three a route.
fn cycleRoute(t: usize) [3]u16 {
    return .{ @intCast(3 + (3 * t) % 13), @intCast(3 + (3 * t + 1) % 13), @intCast(3 + (3 * t + 2) % 13) };
}

test "expert policy: tally: decode never evicts the pinned top of the rate; the other rows churn as the transition window plans them" {
    // 16 experts, 6 rows, ratio 0.2: 3 pins. A window long enough that 40 routes leave the pins where the file put them.
    const a = testing.allocator;
    var p = try LayerPolicy.init(a, 16, 6);
    defer p.deinit(a);
    p.setTally(tally.pinLimit(0.2, 16), tally.alphaOf(100_000));
    var rates: [16]f32 = @splat(0.01);
    rates[0] = 0.9;
    rates[1] = 0.8;
    rates[2] = 0.7;
    p.seedTally(&rates);
    try expectPins(&p, &.{ 0, 1, 2 });
    var out: Plan = .{};
    p.plan(&.{ 0, 1, 2 }, .decode, &out);
    var buf: [16]u16 = undefined;
    for (0..40) |t| {
        const ids = cycleRoute(t);
        p.plan(&ids, .decode, &out);
        try checkPlan(&p, &ids, &out, max_route_ids);
        for (out.evictionsOf()) |ev| try testing.expect(ev.previous > 2);
        const res = residentsOf(&p, &buf);
        try testing.expectEqual(@as(usize, 6), res.len);
        try testing.expectEqualSlices(u16, &.{ 0, 1, 2 }, res[0..3]);
        // The swap rows hold this route's three.
        for (ids) |e| try testing.expect(p.slotOf(e) != null);
    }
    try expectPins(&p, &.{ 0, 1, 2 });
    // The same routes without a tally: the transition window evicts the three at the second route.
    var q = try LayerPolicy.init(a, 16, 6);
    defer q.deinit(a);
    q.plan(&.{ 0, 1, 2 }, .decode, &out);
    for (0..40) |t| {
        const ids = cycleRoute(t);
        q.plan(&ids, .decode, &out);
    }
    for ([_]u16{ 0, 1, 2 }) |e| try testing.expectEqual(@as(?u32, null), q.slotOf(e));
}

test "expert policy: tally: a pinned resident keeps its row where the transition window would give it to the miss" {
    // One row (the transition test above): without the tally, [2] evicts 1 and [1] is then served transient.
    const a = testing.allocator;
    var p = try LayerPolicy.init(a, 8, 1);
    defer p.deinit(a);
    p.setTally(1, tally.alphaOf(100_000));
    var rates: [8]f32 = @splat(0);
    rates[1] = 0.5;
    p.seedTally(&rates);
    var out: Plan = .{};
    p.plan(&.{1}, .decode, &out);
    try testing.expectEqualSlices(Load, &.{.{ .expert = 1, .slot = 0, .persistent = true }}, out.loadsOf());
    p.plan(&.{2}, .decode, &out);
    try testing.expectEqualSlices(Load, &.{.{ .expert = 2, .slot = 1, .persistent = false }}, out.loadsOf());
    try testing.expectEqual(@as(u32, 0), out.n_evictions);
    p.plan(&.{ 1, 1 }, .decode, &out);
    try testing.expectEqualSlices(u16, &.{1}, out.hitsOf());
    try testing.expectEqual(@as(u32, 0), out.n_loads);
}

test "expert policy: tally: seeded, a prefill miss that is not pinned evicts nothing and is no seed; a pinned one evicts the coldest probationer" {
    const a = testing.allocator;
    var p = try LayerPolicy.init(a, 16, 3);
    defer p.deinit(a);
    p.setTally(2, tally.alphaOf(32));
    var rates: [16]f32 = @splat(0);
    rates[5] = 0.9;
    rates[6] = 0.8;
    p.seedTally(&rates);
    try expectPins(&p, &.{ 5, 6 });
    // The rows filled as a preload leaves them (read ahead, unprotected), then the prompt's seed: 7 (3 rows), 5 (2), 1 (1).
    var ra: [4]LayerPolicy.ReadAhead = undefined;
    try testing.expectEqual(@as(usize, 3), p.admitReadAhead(&.{ 1, 2, 3 }, &ra).len);
    p.prepareSeed(&.{ 7, 7, 7, 5, 5, 1 });
    try testing.expect(p.seed.isSet(5));
    try testing.expect(!p.seed.isSet(7));
    var out: Plan = .{};
    p.plan(&.{ 7, 5 }, .prefill, &out);
    try checkPlan(&p, &.{ 7, 5 }, &out, max_route_ids);
    // 5 takes 2's row (2 is the coldest unprotected resident: 1 is the seed's, re-protected); 7 is served transient.
    try testing.expectEqualSlices(Load, &.{
        .{ .expert = 5, .slot = 1, .persistent = true },
        .{ .expert = 7, .slot = 3, .persistent = false },
    }, out.loadsOf());
    try testing.expectEqualSlices(Eviction, &.{.{ .slot = 1, .previous = 2, .next = 5 }}, out.evictionsOf());
    var buf: [16]u16 = undefined;
    try testing.expectEqualSlices(u16, &.{ 1, 3, 5 }, residentsOf(&p, &buf));
}

test "expert policy: tally: ratio 0 plans exactly as no tally; a request's end gives each expert's share of the decode routes and seeds the pins" {
    const a = testing.allocator;
    var p = try LayerPolicy.init(a, 32, 5);
    defer p.deinit(a);
    var q = try LayerPolicy.init(a, 32, 5);
    defer q.deinit(a);
    p.setTally(tally.pinLimit(0, 32), tally.alphaOf(32));
    const half: [32]f32 = @splat(0.5);
    p.seedTally(&half);
    var rng = std.Random.DefaultPrng.init(7);
    const r = rng.random();
    var ids: [max_route_ids]u16 = undefined;
    var routed: [32]u32 = @splat(0);
    var decode_routes: u32 = 0;
    var op: Plan = .{};
    var oq: Plan = .{};
    for (0..300) |step| {
        const n = r.intRangeAtMost(usize, 1, 12);
        for (ids[0..n]) |*e| e.* = @intCast(r.intRangeLessThan(u32, 0, if (step % 4 == 0) 32 else 10));
        const phase: Phase = if (step < 40) .prefill else .decode;
        if (phase == .prefill and step % 10 == 0) {
            p.prepareSeed(ids[0..n]);
            q.prepareSeed(ids[0..n]);
        }
        p.plan(ids[0..n], phase, &op);
        q.plan(ids[0..n], phase, &oq);
        try testing.expectEqualSlices(u32, oq.slotsOf(), op.slotsOf());
        try testing.expectEqualSlices(Load, oq.loadsOf(), op.loadsOf());
        try testing.expectEqualSlices(Eviction, oq.evictionsOf(), op.evictionsOf());
        if (phase == .decode) {
            decode_routes += 1;
            var seen: [32]bool = @splat(false);
            for (ids[0..n]) |e| if (!seen[e]) {
                seen[e] = true;
                routed[e] += 1;
            };
        }
    }
    var share: [32]f32 = undefined;
    try testing.expect(p.endRequest(&share));
    for (share, routed) |s, c| try testing.expectEqual(@as(f32, @floatFromInt(c)) / @as(f32, @floatFromInt(decode_routes)), s);
    // No tally on: nothing to give.
    try testing.expect(!q.endRequest(&share));
    // Unseeded, a tally counts but pins nothing; its request's end seeds it: the top 2 by rate from then on.
    var u = try LayerPolicy.init(a, 8, 3);
    defer u.deinit(a);
    u.setTally(2, tally.alphaOf(4));
    for (0..20) |_| {
        u.plan(&.{ 4, 6 }, .decode, &op);
        u.plan(&.{ 4, 1 }, .decode, &op);
    }
    try expectPins(&u, &.{});
    var s8: [8]f32 = @splat(0);
    try testing.expect(u.endRequest(&s8));
    try testing.expectEqual(@as(f32, 1), s8[4]);
    try testing.expectEqual(@as(f32, 0.5), s8[6]);
    try expectPins(&u, &.{ 1, 4 });
}

test "expert policy: tally: the settings' defaults, their keys, and the values refused by name" {
    const parse = struct {
        fn of(text: []const u8) !tally.Settings {
            var arena = std.heap.ArenaAllocator.init(testing.allocator);
            defer arena.deinit();
            const v = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), text, .{});
            var s: tally.Settings = .{};
            _ = s.apply(v.object);
            return s;
        }
    };
    const d: tally.Settings = .{};
    try testing.expectEqual(@as(f64, 0.5), d.ratio);
    try testing.expectEqual(@as(u32, 32), d.window);
    try d.check("test");
    try testing.expectEqual(@as(f64, 0.3), (try parse.of("{\"expert_resident_ratio\": 0.3}")).ratio);
    try testing.expectEqual(@as(f64, 0), (try parse.of("{\"expert_resident_ratio\": 0}")).ratio);
    try testing.expectEqual(@as(f64, 1), (try parse.of("{\"expert_resident_ratio\": 1}")).ratio);
    try testing.expectEqual(@as(f64, 0.5), (try parse.of("{\"expert_resident_ratio\": \"0.3\"}")).ratio);
    try testing.expectError(error.ResidentRatioOutOfRange, (try parse.of("{\"expert_resident_ratio\": 1.5}")).check("test"));
    try testing.expectError(error.ResidentRatioOutOfRange, (try parse.of("{\"expert_resident_ratio\": -0.1}")).check("test"));
    try testing.expectEqual(@as(u32, 64), (try parse.of("{\"expert_tally_window\": 64}")).window);
    try testing.expectError(error.TallyWindowOutOfRange, (try parse.of("{\"expert_tally_window\": 0}")).check("test"));
    try testing.expectEqual(@as(u32, 32), (try parse.of("{\"expert_tally_window\": 2.5}")).window);
    // The file's use: read and written by default, pinned (read only) or off for a cell; another name refused by name.
    try testing.expect(d.readsFile() and d.writesFile());
    const pinned = try parse.of("{\"expert_tally_file\": \"pinned\"}");
    try testing.expect(pinned.readsFile() and !pinned.writesFile());
    const off = try parse.of("{\"expert_tally_file\": \"off\"}");
    try testing.expect(!off.readsFile() and !off.writesFile() and off.enabled());
    try testing.expectError(error.TallyFileUnknown, (try parse.of("{\"expert_tally_file\": \"ro\"}")).check("test"));
    // A ratio of 0 is no tally: nothing counted, no file read or written, whatever the file's setting.
    const none = try parse.of("{\"expert_resident_ratio\": 0, \"expert_tally_file\": \"on\"}");
    try testing.expect(!none.enabled() and !none.readsFile() and !none.writesFile());
    // GLM-5.3's EXL3 bank layers: 148 and 108 experts.
    try testing.expectEqual(@as(u32, 74), tally.pinLimit(0.5, 148));
    try testing.expectEqual(@as(u32, 54), tally.pinLimit(0.5, 108));
    try testing.expectEqual(@as(u32, 0), tally.pinLimit(0, 148));
    try testing.expectEqual(@as(u32, 148), tally.pinLimit(1, 148));
    try testing.expectEqual(@as(u32, 44), tally.pinLimit(0.3, 148));
    try testing.expectApproxEqAbs(@as(f32, 0.0214), tally.alphaOf(32), 1e-4);
}

test "expert policy: tally: the file round-trips its rates; a missing file, another geometry and a short file are not read" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = testing.io;
    const root = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(root);
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/model-key/" ++ tally.file_name, .{root});
    defer testing.allocator.free(path);
    try testing.expect(try tally.read(testing.allocator, io, path, 2, 4) == .missing);
    const rates = [_]f32{ 0.5, 0, 0.25, 1, 0.125, 0.75, 0, 0 };
    try tally.write(testing.allocator, io, path, &rates, 2, 4, 32);
    const got = try tally.read(testing.allocator, io, path, 2, 4);
    defer if (got == .rates) testing.allocator.free(got.rates);
    try testing.expectEqualSlices(f32, &rates, got.rates);
    try testing.expect(try tally.read(testing.allocator, io, path, 4, 2) == .mismatched);
    try testing.expect(try tally.read(testing.allocator, io, path, 2, 8) == .mismatched);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = "MLXSTALY" });
    try testing.expect(try tally.read(testing.allocator, io, path, 2, 4) == .mismatched);
    const key = try tally.keyOf(testing.allocator, "glm_moe_dsa", "/Users/x/glm53/pack-exl3-3.42bpw/");
    defer testing.allocator.free(key);
    try testing.expectEqualStrings("glm_moe_dsa-pack-exl3-3.42bpw", key);
}
