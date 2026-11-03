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
/// admission needs: an opaque model identity + the exclusive-decode bit.
pub const AdmitCand = struct { model: usize, exclusive: bool };

/// FIFO admission for one drain tick. An EXCLUSIVE candidate admits only if
/// no EXCLUSIVE `active` slot (live decoding) holds its model and no earlier
/// admitted exclusive candidate claimed it this tick
/// (a same-tick sibling is not in `decoding` yet — the claim covers the
/// window). Non-exclusive candidates always admit — no head-of-line
/// blocking behind a held exclusive request. Writes admitted candidate
/// indices to `out` (queue order preserved), returns the count. Held
/// candidates stay where they are and retry next tick; no release
/// bookkeeping exists — the busy signal IS presence in `decoding`.
pub fn admitPendingTick(cands: []const AdmitCand, active: []const AdmitCand, out: []usize) usize {
    var n: usize = 0;
    outer: for (cands, 0..) |c, i| {
        if (n >= out.len) break;
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
