//! Forced-token ticks: the thinking budget, tool_choice and committed forced tokens.

const std = @import("std");
const generate_mod = @import("generate.zig");
const log = @import("../../sdk/util/log.zig");
const Generator = generate_mod.Generator;

const finishSlot = @import("finish.zig").finishSlot;
const Scheduler = @import("scheduler.zig").Scheduler;
const Slot = @import("slot.zig").Slot;

/// A thinking budget at its limit: commit the early-stop line and the closer
/// through the model this tick; an MTP slot ramps back to speculating, any other
/// spec mode decodes regular (`spec_disable_reason = .think_bound`). True = the slot's tick is spent.
pub fn thinkBoundTick(sch: *Scheduler, slot: *Slot, gen: *Generator) !bool {
    const tb = gen.sampling.think_bound orelse return false;
    tb.observe(gen.generated_ids.items);
    if (!tb.due()) return false;
    tb.fired = true;
    if (!generate_mod.forcedBoundaryCanContinue(gen.completion_tokens, gen.max_tokens, tb.forced.len + 1)) {
        log.warn("[think-bound] budget {d} reached with no room to close the thought (max_tokens {d})\n", .{ tb.budget, gen.max_tokens });
        return false;
    }
    if (try commitForcedTick(sch, slot, gen, tb.forced, .think_bound)) {
        log.info("[think-bound] reasoning budget {d} reached at {d} generated tokens; thought closed\n", .{ tb.budget, gen.generated_ids.items.len });
    }
    return true;
}

/// `tool_choice` that obliges a call: once the thought closes, commit the
/// tool-call opener through the model, which writes the call from there.
pub fn toolForceTick(sch: *Scheduler, slot: *Slot, gen: *Generator) !bool {
    const tf = gen.sampling.tool_force orelse return false;
    const d = tf.due(gen.generated_ids.items, gen.completion_tokens, gen.max_tokens) orelse return false;
    tf.fired = true;
    // The model's own next token is published first: a call it already opened is not opened twice.
    const tokens = if (try gen.upcomingToken() == d.tokens[0]) d.tokens[1..] else d.tokens;
    if (tokens.len == 0) return false;
    if (!generate_mod.forcedBoundaryCanContinue(gen.completion_tokens, gen.max_tokens, tokens.len + 1)) {
        log.warn("[tool-choice] no room to force the call (max_tokens {d})\n", .{gen.max_tokens});
        return false;
    }
    if (try commitForcedTick(sch, slot, gen, tokens, .tool_choice)) {
        log.info("[tool-choice] call opener forced at {d} generated tokens{s}\n", .{ gen.generated_ids.items.len, if (d.closes_thought) " (thought closed for it)" else "" });
    }
    return true;
}

/// Commit `forced` through the model and publish it; an MTP or DFlash slot keeps
/// speculating, any other decodes regular. False = the pending token stopped the slot first.
pub fn commitForcedTick(sch: *Scheduler, slot: *Slot, gen: *Generator, forced: []const u32, reason: generate_mod.SpecDisableReason) !bool {
    const r = try gen.commitForcedTokens(slot.allocator, forced);
    defer slot.allocator.free(r.emitted);
    for (r.emitted) |t| {
        slot.pushToken(t);
        slot.completion_tokens += 1;
        if (t != 0) slot.was_pad_only = false;
    }
    std.debug.assert(slot.completion_tokens == gen.completion_tokens);
    if (r.stopped) {
        finishSlot(sch, slot, gen.finish_reason);
        return false;
    }
    if (!r.dflash_synced and !try gen.mtpResumeAfterForced(slot.allocator)) {
        gen.spec_disabled_runtime = true;
        gen.spec_disable_reason = reason;
    }
    return true;
}
