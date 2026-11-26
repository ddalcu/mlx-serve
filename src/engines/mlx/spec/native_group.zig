//! Rounds of a served drafter (a plugin's `NativeDrafter`) for several slots of one model at once: each slot drafts and
//! opens its verify, every accept starts on the GPU together, and each slot then keeps its accepted prefix (a sampled slot
//! accepts under its own sampling, a greedy one against the trunk's picks). A slot alone, or one whose next round is not a group's, ticks solo.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const dspark = @import("dspark.zig");
const plugin_driver = @import("../plugin/driver.zig");
const Generator = @import("../generate/generator.zig").Generator;
const DrafterStepResult = @import("drafter.zig").DrafterStepResult;
const Scheduler = @import("../scheduler.zig").Scheduler;
const Slot = @import("../slot.zig").Slot;
const finishSlot = @import("../finish.zig").finishSlot;
const runSingleDecodeTick = @import("../decode.zig").runSingleDecodeTick;
const loopGuardTick = @import("../loop_stop.zig").loopGuardTick;
const publishSpeculativeBlock = @import("wiring.zig").publishSpeculativeBlock;
const slotSpecMode = @import("wiring.zig").slotSpecMode;

/// Slots one group round carries: the plugin keeps a verify open for each.
pub const MAX_SLOTS: usize = plugin_driver.max_native_group;

/// Whether `slot` ticks rounds of a served drafter that keeps a verify open a slot.
pub fn slotGroupable(slot: *const Slot) bool {
    if (!env.notOff(.MLX_SERVE_NATIVE_GROUPS)) return false;
    const gen = if (slot.legacy_gen) |*g| g else return false;
    if (gen.spec_disabled_runtime or slotSpecMode(slot, gen) != .dspark) return false;
    return plugin_driver.nativeGroups(slot.model.transformer orelse return false);
}

/// The decode tick's grouped slots: one group round per model, `MAX_SLOTS` slots at a time; a slot alone ticks solo.
pub fn runNativeGroups(sch: *Scheduler, slots: []*Slot) !void {
    std.sort.pdq(*Slot, slots, {}, struct {
        fn lt(_: void, a: *Slot, b: *Slot) bool {
            return @intFromPtr(a.model) < @intFromPtr(b.model);
        }
    }.lt);
    var start: usize = 0;
    while (start < slots.len) {
        var end = start + 1;
        while (end < slots.len and slots[end].model == slots[start].model) end += 1;
        var g0 = start;
        while (g0 < end) {
            const sub = slots[g0..@min(end, g0 + MAX_SLOTS)];
            if (sub.len == 1) try runSingleDecodeTick(sch, sub[0]) else try runGroupTick(sch, sub);
            g0 += sub.len;
        }
        start = end;
    }
}

fn runGroupTick(sch: *Scheduler, group: []*Slot) !void {
    var inner_err: ?anyerror = null;
    runGroupTickInner(sch, group) catch |e| {
        inner_err = e;
    };
    mlx.checkErrorDecode() catch |mlx_err| {
        log.err("[scheduler] group round aborted: MLX failure mid-generation ({s}): failing these requests, the server keeps serving\n", .{@errorName(mlx_err)});
        for (group) |slot| slot.markError(@errorName(mlx_err));
        return;
    };
    if (inner_err) |e| return e;
}

fn runGroupTickInner(sch: *Scheduler, group: []*Slot) !void {
    var live: [MAX_SLOTS]*Slot = undefined;
    var gens: [MAX_SLOTS]*Generator = undefined;
    var allocs: [MAX_SLOTS]std.mem.Allocator = undefined;
    var n: usize = 0;
    for (group) |slot| {
        const gen = &slot.legacy_gen.?;
        if (try loopGuardTick(sch, slot, gen)) continue;
        const ready = dspark.groupReady(gen) catch |e| {
            slot.markError(@errorName(e));
            continue;
        };
        if (!ready) {
            try runSingleDecodeTick(sch, slot);
            continue;
        }
        live[n] = slot;
        gens[n] = gen;
        allocs[n] = slot.allocator;
        n += 1;
    }
    if (n == 0) return;
    if (n == 1) return runSingleDecodeTick(sch, live[0]);
    var results: [MAX_SLOTS]DrafterStepResult = undefined;
    try dspark.nativeGroupRound(gens[0..n], allocs[0..n], results[0..n]);
    for (live[0..n], gens[0..n], results[0..n]) |slot, gen, r| {
        defer slot.allocator.free(r.tokens);
        publishSpeculativeBlock(sch, slot, gen, r.tokens);
    }
}
