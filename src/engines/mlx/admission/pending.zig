//! Admission of pending requests: the live-memory hold and the single-flight pending tick.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const generate_mod = @import("../generate.zig");
const log = @import("../../../sdk/util/log.zig");
const testing = std.testing;

const holdsForMemory = @import("../batching/verdict.zig").holdsForMemory;
const admissionFits = @import("../batching/verdict.zig").admissionFits;
const liveDecodingCount = @import("../batching/verdict.zig").liveDecodingCount;
const Scheduler = @import("../scheduler.zig").Scheduler;
const Slot = @import("../slot.zig").Slot;
const scheduler_mod = @import("../scheduler.zig");

/// Cold bill against live memory; the gated arch runs its own warm pass inside `runPrefill`.
pub fn slotHoldsForMemory(sch: *Scheduler, slot: *Slot) bool {
    const cfg = slot.model.config orelse return false;
    if (cfg.longCtxGated()) return false;
    if (slot.model.transformer == null) return false;
    const numbers_fn = scheduler_mod.prefill_admission_numbers orelse return false;
    // A prefill hosting a nested admission holds memory but is in no list: it is company too.
    const live = liveDecodingCount(sch) + @intFromBool(sch.boundary_outer != null);
    if (live == 0) sch.promise_ledger = .{};
    const bill = numbers_fn(cfg, slot.full_prompt.len, slot.max_tokens, slot.cache.config, slot.enable_mtp);
    var active_now: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_now);
    const fits = admissionFits(bill[0], bill[1], sch.promise_ledger.outstanding(active_now), sch.gen_reserve_bytes, live);
    if (!holdsForMemory(fits, live > 0 or sch.gen_reserve_bytes > 0)) {
        if (slot.memory_promised == 0) {
            slot.memory_promised = bill[0];
            sch.promise_ledger.admit(bill[0], active_now);
        }
        return false;
    }
    if (!slot.memory_hold_logged) {
        slot.memory_hold_logged = true;
        log.info("[admission] held: {d} tokens do not fit beside {d} live request(s) and {d} MB of media job; waiting for one to finish\n", .{ slot.full_prompt.len, live, sch.gen_reserve_bytes >> 20 });
    }
    return true;
}

/// One pending-drain candidate (or live decoding slot), reduced to what
/// admission needs: model identity, exclusivity, and an optional bounded owner count.
pub const AdmitCand = struct { model: usize, exclusive: bool, limit: usize = 0, occupied: usize = 0 };

/// FIFO admission for one drain tick. An EXCLUSIVE candidate admits only if
/// no EXCLUSIVE `active` slot (live decoding) holds its model and no earlier
/// admitted exclusive candidate claimed it this tick
/// (a same-tick sibling is not in `decoding` yet — the claim covers the
/// window). Bounded plugins also hold when their owner limit is reached. Other
/// models can still pass a held candidate. Writes admitted candidate
/// indices to `out` (queue order preserved), returns the count. Held
/// candidates stay where they are and retry next tick; no release
/// bookkeeping exists — ownership follows the scheduler's active slots.
pub fn admitPendingTick(cands: []const AdmitCand, active: []const AdmitCand, out: []usize) usize {
    var n: usize = 0;
    outer: for (cands, 0..) |c, i| {
        if (n >= out.len) break;
        if (c.limit > 0) {
            var live: usize = 0;
            for (active) |a| if (a.model == c.model) {
                live += 1;
            };
            live = @max(live, c.occupied);
            for (out[0..n]) |j| if (cands[j].model == c.model) {
                live += 1;
            };
            if (live >= c.limit) continue;
        }
        if (c.exclusive) {
            for (active) |a| if (a.exclusive and a.model == c.model) continue :outer;
            for (out[0..n]) |j| {
                if (cands[j].exclusive and cands[j].model == c.model) continue :outer;
            }
        }
        out[n] = i;
        n += 1;
    }
    return n;
}

/// The CLI narrows a bounded plugin; zero preserves existing unbounded plugin admission.
pub fn activeLimit(plugin_limit: u8, cli_limit: u32) usize {
    return if (plugin_limit == 0) 0 else @min(plugin_limit, @max(cli_limit, 1));
}

pub fn slotLimit(sch: *const Scheduler, slot: *const Slot) usize {
    const cfg = slot.model.config orelse return 0;
    return activeLimit(cfg.caps().active_slots, sch.max_concurrent);
}

/// Count owners until their release/cull, plus this pass's not-yet-completed prefills. A cancelled request still
/// owns its module state until cull; excluding it early would turn the next admission into a plugin refusal.
pub fn occupiedSlots(sch: *const Scheduler, slot: *const Slot) usize {
    var n: usize = 0;
    var outer_counted = false;
    for (sch.decoding.items) |s| {
        if (s.model == slot.model) n += 1;
        if (sch.boundary_outer == s) outer_counted = true;
    }
    for (sch.prefilling[sch.prefilling_from..sch.n_prefilling]) |s| {
        if (s.model == slot.model) n += 1;
        if (sch.boundary_outer == s) outer_counted = true;
    }
    if (!outer_counted) if (sch.boundary_outer) |s| {
        if (s.model == slot.model) n += 1;
    };
    return n;
}

// ── tests ──

test "admitPendingTick: per-slot exclusivity (qwen4 MTP slot) blocks only its own class" {
    const A: usize = 0xA0;
    var out: [16]usize = undefined;
    // An MTP candidate beside a live PLAIN slot on the same model admits.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        const active = [_]AdmitCand{.{ .model = A, .exclusive = false }};
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &active, &out));
    }
    // An MTP candidate beside a live MTP slot holds.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        const active = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 0), admitPendingTick(&cands, &active, &out));
    }
    // A plain candidate beside a live MTP slot admits.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = false }};
        const active = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &active, &out));
    }
}

test "admitPendingTick: exclusive single-flight FIFO contract" {
    const A: usize = 0xA0;
    const B: usize = 0xB0;
    var out: [16]usize = undefined;

    // Held while a live slot on the same exclusive model is decoding — the
    // dsv4 class: a second admitted slot deinit+rebuilds the module-owned
    // dec_state at cache.step==0 and both requests then interleave on it.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        const active = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 0), admitPendingTick(&cands, &active, &out));
    }
    // Admits once no live slot holds the model.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &.{}, &out));
        try testing.expectEqual(@as(usize, 0), out[0]);
    }
    // Two exclusive candidates on the SAME model in one tick: only the
    // first admits — the tick-claim covers slots not yet in `decoding`.
    {
        const cands = [_]AdmitCand{
            .{ .model = A, .exclusive = true },
            .{ .model = A, .exclusive = true },
        };
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &.{}, &out));
        try testing.expectEqual(@as(usize, 0), out[0]);
    }
    // Distinct exclusive models admit independently.
    {
        const cands = [_]AdmitCand{
            .{ .model = A, .exclusive = true },
            .{ .model = B, .exclusive = true },
        };
        try testing.expectEqual(@as(usize, 2), admitPendingTick(&cands, &.{}, &out));
        try testing.expectEqual(@as(usize, 0), out[0]);
        try testing.expectEqual(@as(usize, 1), out[1]);
    }
}

test "admitPendingTick: non-exclusive concurrency and queue order preserved" {
    const A: usize = 0xA0;
    const L: usize = 0x10;
    var out: [16]usize = undefined;

    // Non-exclusive candidates (laguna/hy3: per-slot state) admit freely
    // even beside live slots — their serial-tick interleave is safe.
    {
        const cands = [_]AdmitCand{
            .{ .model = L, .exclusive = false },
            .{ .model = L, .exclusive = false },
        };
        const active = [_]AdmitCand{.{ .model = L, .exclusive = false }};
        try testing.expectEqual(@as(usize, 2), admitPendingTick(&cands, &active, &out));
    }
    // A held exclusive candidate must not head-of-line-block a later
    // candidate on another model (requests for OTHER models keep flowing).
    {
        const cands = [_]AdmitCand{
            .{ .model = A, .exclusive = true },
            .{ .model = L, .exclusive = false },
        };
        const active = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &active, &out));
        try testing.expectEqual(@as(usize, 1), out[0]);
    }
    // `out` caps the admitted count (mirrors to_prefill's 16), in order.
    {
        const cands = [_]AdmitCand{
            .{ .model = L, .exclusive = false },
            .{ .model = L, .exclusive = false },
            .{ .model = L, .exclusive = false },
        };
        var small: [2]usize = undefined;
        try testing.expectEqual(@as(usize, 2), admitPendingTick(&cands, &.{}, &small));
        try testing.expectEqual(@as(usize, 0), small[0]);
        try testing.expectEqual(@as(usize, 1), small[1]);
    }
}

test "admitPendingTick: bounded streamed owners hold a third, including same-tick claims and cancellation" {
    const candidates = [_]AdmitCand{
        .{ .model = 1, .exclusive = false, .limit = 2 },
        .{ .model = 1, .exclusive = false, .limit = 2 },
        .{ .model = 1, .exclusive = false, .limit = 2 },
        .{ .model = 2, .exclusive = false },
    };
    var out: [4]usize = undefined;
    try testing.expectEqual(@as(usize, 3), admitPendingTick(&candidates, &.{}, &out));
    try testing.expectEqualSlices(usize, &.{ 0, 1, 3 }, out[0..3]);
    const active = [_]AdmitCand{ candidates[0], candidates[1] };
    try testing.expectEqual(@as(usize, 1), admitPendingTick(candidates[2..], &active, &out));
    try testing.expectEqual(@as(usize, 1), out[0]);
    // Capacity returns only after the cancelled owner's release removes it from the active set.
    try testing.expectEqual(@as(usize, 2), admitPendingTick(candidates[2..], active[0..1], &out));
    try testing.expectEqualSlices(usize, &.{ 0, 1 }, out[0..2]);
    const single = [_]AdmitCand{.{ .model = 1, .exclusive = false, .limit = 1 }};
    try testing.expectEqual(@as(usize, 0), admitPendingTick(&single, active[0..1], &out));
}

test "bounded plugin active limit honors CLI without changing unbounded admission" {
    try testing.expectEqual(@as(usize, 0), activeLimit(0, 1));
    try testing.expectEqual(@as(usize, 1), activeLimit(2, 1));
    try testing.expectEqual(@as(usize, 2), activeLimit(2, 2));
    try testing.expectEqual(@as(usize, 2), activeLimit(2, 32));
    try testing.expectEqual(@as(usize, 1), activeLimit(2, 0));
}

test "bounded plugin occupancy includes cancelled owners until cull and counts an outer prefill once" {
    var models: [2]@import("../../../host/registry/registry.zig").LoadedModel = undefined;
    var slots: [4]Slot = undefined;
    for (&slots, 0..) |*s, i| s.model = &models[if (i == 3) @as(usize, 1) else 0];
    slots[0].cancelled = .init(true);
    var decoding = [_]*Slot{ &slots[0], &slots[1] };
    var sch: Scheduler = undefined;
    sch.decoding = .empty;
    sch.decoding.items = &decoding;
    sch.decoding.capacity = decoding.len;
    sch.prefilling[0] = &slots[2];
    sch.prefilling[1] = &slots[3];
    sch.prefilling_from = 0;
    sch.n_prefilling = 2;
    sch.boundary_outer = &slots[2];
    try testing.expectEqual(@as(usize, 3), occupiedSlots(&sch, &slots[1]));
    try testing.expectEqual(@as(usize, 1), occupiedSlots(&sch, &slots[3]));
    sch.decoding.items = decoding[1..];
    try testing.expectEqual(@as(usize, 2), occupiedSlots(&sch, &slots[1]));
    sch.prefilling_from = 1;
    try testing.expectEqual(@as(usize, 2), occupiedSlots(&sch, &slots[1]));
    sch.boundary_outer = null;
    try testing.expectEqual(@as(usize, 1), occupiedSlots(&sch, &slots[1]));
    var out: [1]usize = undefined;
    try testing.expectEqual(@as(usize, 0), admitPendingTick(&.{.{ .model = 1, .exclusive = false, .limit = 2, .occupied = 2 }}, &.{}, &out));
}
