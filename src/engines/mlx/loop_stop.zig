//! Loop-stop guard: the cut reason, the decision and the think-block recovery.

const std = @import("std");
const env = @import("../../env.zig");
const generate_mod = @import("generate.zig");
const log = @import("../../sdk/util/log.zig");
const Generator = generate_mod.Generator;
const testing = std.testing;

const finishSlot = @import("finish.zig").finishSlot;
const thinkBoundTick = @import("logit_features.zig").thinkBoundTick;
const toolForceTick = @import("logit_features.zig").toolForceTick;
const commitForcedTick = @import("logit_features.zig").commitForcedTick;
const Scheduler = @import("scheduler.zig").Scheduler;
const Slot = @import("slot.zig").Slot;

/// Phase A6: finalize a slot. Commits to hot prefix cache (if applicable)
/// before signaling completion. The order matters: commit first while the
/// slot is alive, then markFinished — the conn thread's waitNext might
/// return immediately after the broadcast and call complete()→deinit, so
/// we cannot reach into the slot afterwards.
/// Pure decision for the degenerate-tail-loop guard: when the generated tail
/// has collapsed into a short repeating cycle, returns the finish reason to
/// cut the request with; null while generation is healthy.
///
/// The reason is "stop", not "length": this guard fired far below the requested
/// output cap, and clients such as pi treat "length" as context-overflow recovery
/// and compact unnecessarily. The server separately suppresses tool-call parsing
/// for this cause, so a cut fragment cannot be promoted to a completed call.
pub fn loopStopReason(generated_ids: []const u32) ?[]const u8 {
    const d = loopStopDecision(generated_ids) orelse return null;
    return d.finish_reason;
}

/// What a loop cut tells the rest of the server. `finish_reason` is the wire
/// value (always "stop" — see above); `finish_details` is the sibling
/// signal that names the CAUSE, and `trim_start` is where the client's copy
/// of the answer should end.
pub const LoopStop = struct {
    finish_reason: []const u8 = "stop",
    /// The `finish_details.type` value. One string for all three tiers: a
    /// client's decision ("this turn is unusable, don't feed it back") is the
    /// same whichever tier convicted, and the tier is in the log.
    finish_details: []const u8 = "repetition_loop",
    tier: generate_mod.DegenerateTail.Tier,
    trim_start: usize,
};

/// Pure decision + trim point. The three tiers live in generate.zig
/// (`degenerateTail`); this is where their verdict becomes server behaviour.
pub fn loopStopDecision(generated_ids: []const u32) ?LoopStop {
    const d = generate_mod.degenerateTail(generated_ids) orelse return null;
    return .{ .tier = d.tier, .trim_start = d.start };
}

const LoopAction = enum { cut, close_thought };

/// A loop convicted INSIDE an open think block is recoverable once per request:
/// commit the bound's early-stop line + closer and let the model answer. Anything
/// else (answer-side loop, second loop, no room, no bound) keeps the ordinary cut.
fn loopRecoveryAction(tb: ?*generate_mod.ThinkBound, ids: []const u32, completion_tokens: u32, max_tokens: u32) LoopAction {
    const b = tb orelse return .cut;
    b.observe(ids);
    if (b.fired or !b.in_think) return .cut;
    if (!generate_mod.forcedBoundaryCanContinue(completion_tokens, max_tokens, b.forced.len + 1)) return .cut;
    return .close_thought;
}


/// Every single-slot decode tick funnels through here, so a decode-time MLX failure is
/// attributed to the slot whose forward raised it instead of the next request's prefill.
/// The repetition-loop guard every decode tick runs first. True = the slot was finished here.
pub fn loopGuardTick(sch: *Scheduler, slot: *Slot, gen: *Generator) !bool {
    // Stop a runaway repetition loop before generating more. Some models (seen
    // on Gemma 4 12B after a large/confusing tool result) collapse into spamming
    // one short cycle — e.g. the thinking opener `<|channel>thought` — forever;
    // with no repeat penalty by default and a generous max_tokens, nothing else
    // halts it until the cap. Checked here, before this tick's step, so it
    // covers the regular, PLD, and drafter paths uniformly.
    const loop_guard_start = gen.loopGuardStart();
    if (loopStopDecision(gen.generated_ids.items[loop_guard_start..])) |relative_stop| {
        if (gen.hasPrePayloadConstraint()) {
            switch (try gen.forceConstraintTransition(slot.allocator)) {
                .committed => |boundary| {
                    slot.pushToken(boundary);
                    slot.completion_tokens += 1;
                    std.debug.assert(slot.completion_tokens == gen.completion_tokens);
                    if (boundary != 0) slot.was_pad_only = false;
                    log.warn("[grammar] reasoning boundary forced after repetition loop at {d} generated tokens\n", .{gen.generated_ids.items.len - 1});
                    return true;
                },
                .activated => return true,
                // The completion cap won the race with recovery: fall through
                // to the ordinary loop-stop cut.
                .refused => {},
            }
        }
        if (loopRecoveryAction(gen.sampling.think_bound, gen.generated_ids.items, gen.completion_tokens, gen.max_tokens) == .close_thought) {
            log.warn("[loop-recover] repetition loop inside the thought at {d} generated tokens; closing it\n", .{gen.generated_ids.items.len});
            const tb = gen.sampling.think_bound.?;
            tb.fired = true;
            _ = try commitForcedTick(sch, slot, gen, tb.forced, .think_bound);
            return true;
        }
        var stop = relative_stop;
        stop.trim_start += loop_guard_start;
        // Never cut silently: the php.html post-mortem took log
        // archaeology because this guard left no trace of having fired. The
        // tier and the trim point are logged too — five cuts in a row is a
        // different diagnosis from one, and the trim is what breaks the chain.
        log.warn("[loop-stop] degenerate tail loop cut after {d} generated tokens (finish_reason={s} details={s} tier={s} trim_start={d})\n", .{
            gen.generated_ids.items.len, stop.finish_reason, stop.finish_details,
            @tagName(stop.tier),         stop.trim_start,
        });
        slot.finish_details = stop.finish_details;
        slot.loop_trim_start = stop.trim_start;
        finishSlot(sch, slot, stop.finish_reason);
        return true;
    }
    if (try thinkBoundTick(sch, slot, gen)) return true;
    return toolForceTick(sch, slot, gen);
}

// ── tests ──

test "loopStopReason: a degenerate tail cut reports stop, a healthy tail is not cut" {
    // A loop guard is an intentional server stop, not exhaustion of the
    // requested output budget. Reporting "length" makes clients such as pi
    // run context-overflow recovery and compact a mostly-empty context. Tool
    // calls from this truncated buffer are suppressed separately at emission.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);

    // Healthy varied tail: never cut.
    for (0..40) |i| try ids.append(testing.allocator, @as(u32, @intCast(i * 7 + 3)));
    try testing.expect(loopStopReason(ids.items) == null);

    // Collapse into a short cycle (the php.html shape: "server-side scripting
    // language, " ≈ a 6-token cycle) past the guard's span threshold.
    for (0..generate_mod.degenerate_loop_min_span / 6 + 1) |_| {
        for ([_]u32{ 101, 202, 303, 404, 505, 606 }) |t| {
            try ids.append(testing.allocator, t);
        }
    }
    const reason = loopStopReason(ids.items) orelse return error.TestExpectedLoopCut;
    try testing.expectEqualStrings("stop", reason);
}

test "loopStopReason: a LONG-period sentence loop is cut at the second tier" {
    // The shooter wrap-up failure: a two-sentence cycle ("The game
    // is complete. Let me do a final review... Let me verify main.js...") of
    // ~58 tokens repeated 26 times sailed through the 8-token-period tier and
    // was never cut. Tier 2 scans periods 9..64 and requires 10 exact
    // repetitions — verbatim-identical long cycles at that count are
    // degeneration, not content.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);
    for (0..30) |i| try ids.append(testing.allocator, @as(u32, @intCast(i * 3 + 11)));

    // 58-token cycle, one rep short of the tier's span bar: NOT cut.
    var cycle: [58]u32 = undefined;
    for (&cycle, 0..) |*v, i| v.* = @as(u32, @intCast(1000 + i));
    for (0..generate_mod.degenerate_loop_long_min_span / cycle.len) |_| try ids.appendSlice(testing.allocator, &cycle);
    try testing.expect(loopStopReason(ids.items) == null);

    // The next repetition crosses it — cut as an intentional stop.
    try ids.appendSlice(testing.allocator, &cycle);
    const reason = loopStopReason(ids.items) orelse return error.TestExpectedLoopCut;
    try testing.expectEqualStrings("stop", reason);
}

test "loopStopDecision: the wire reason is stop and the CAUSE rides beside it" {
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);
    try ids.appendSlice(testing.allocator, &[_]u32{ 5, 6, 7 });
    for (0..generate_mod.degenerate_loop_min_span / 3 + 1) |_| {
        try ids.appendSlice(testing.allocator, &[_]u32{ 101, 102, 103 });
    }

    const stop = loopStopDecision(ids.items) orelse return error.TestExpectedLoopCut;
    try testing.expectEqualStrings("stop", stop.finish_reason);
    try testing.expectEqualStrings("repetition_loop", stop.finish_details);
    try testing.expectEqual(generate_mod.DegenerateTail.Tier.exact_cycle, stop.tier);
    // Trimmed to the honest prefix plus one copy of the cycle.
    try testing.expectEqual(@as(usize, 6), stop.trim_start);

    // Once a deferred boundary has been committed, the old reasoning loop is
    // outside the guard window and cannot stop the very next answer tick.
    try ids.append(testing.allocator, 99);
    const answer_start = ids.items.len;
    try testing.expect(loopStopDecision(ids.items[answer_start..]) == null);

    // A new loop wholly inside the constrained answer retains the existing
    // stop/repetition result, with an absolute trim point for response code.
    for (0..generate_mod.degenerate_loop_min_span / 3 + 1) |_| {
        try ids.appendSlice(testing.allocator, &[_]u32{ 7, 8, 9 });
    }
    const answer_loop = loopStopDecision(ids.items[answer_start..]) orelse return error.TestExpectedLoopCut;
    try testing.expectEqualStrings("stop", answer_loop.finish_reason);
    try testing.expect(answer_loop.trim_start + answer_start >= answer_start);

    // Healthy output decides nothing at all — no reason, and nothing to trim.
    var healthy: [512]u32 = undefined;
    for (&healthy, 0..) |*v, i| v.* = @intCast(i);
    try testing.expect(loopStopDecision(&healthy) == null);
    try testing.expect(loopStopReason(&healthy) == null);
}

test "loopStopReason: periods past the long tier stay uncut" {
    // A 70-token exact cycle (> long-tier max 64) repeated many times is
    // outside both tiers — the guard stays scoped rather than judging whole
    // repeated paragraphs.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);
    var cycle: [70]u32 = undefined;
    for (&cycle, 0..) |*v, i| v.* = @as(u32, @intCast(2000 + i));
    for (0..12) |_| try ids.appendSlice(testing.allocator, &cycle);
    try testing.expect(loopStopReason(ids.items) == null);
}

test "loopStopReason: a VARIED-phrasing restatement loop is cut at the near-repeat tier" {
    // The agent-traffic shape: the same intent restated forever in
    // slightly different words. No exact cycle exists, so tiers 1 and 2 are
    // blind and this ran to max_tokens.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);
    const phrasings = [_][]const u32{
        &[_]u32{ 40, 41, 42, 43, 44, 45, 46 },
        &[_]u32{ 40, 41, 42, 43, 44, 46 },
        &[_]u32{ 40, 41, 42, 43, 44, 45, 47, 48, 46 },
        &[_]u32{ 49, 40, 41, 42, 43, 44, 45, 46 },
        &[_]u32{ 40, 41, 42, 43, 44, 45, 50, 46 },
    };
    var i: usize = 0;
    while (ids.items.len < generate_mod.near_repeat_min_span + 32) : (i += 1) {
        try ids.appendSlice(testing.allocator, phrasings[i % phrasings.len]);
    }
    const reason = loopStopReason(ids.items) orelse return error.TestExpectedLoopCut;
    try testing.expectEqualStrings("stop", reason);

    // A long answer that keeps introducing new material is untouched, however
    // repetitive its scaffolding.
    var healthy = std.ArrayList(u32).empty;
    defer healthy.deinit(testing.allocator);
    var line: u32 = 0;
    while (healthy.items.len < generate_mod.near_repeat_window + 32) : (line += 1) {
        try healthy.appendSlice(testing.allocator, &[_]u32{ 10, 11, 12 });
        try healthy.append(testing.allocator, 1000 + line);
        try healthy.appendSlice(testing.allocator, &[_]u32{ 13, 14 });
    }
    try testing.expect(loopStopReason(healthy.items) == null);
}

test "loop recovery: a loop inside an open think block closes the thought once" {
    const OPEN: u32 = 10;
    const CLOSE: u32 = 11;
    const forced = [_]u32{ 30, 31, CLOSE, 32 };
    var tb = generate_mod.ThinkBound{ .budget = std.math.maxInt(u32), .opener_id = OPEN, .closer_id = CLOSE, .forced = &forced, .in_think = true };
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);
    try ids.appendSlice(testing.allocator, &[_]u32{ 5, 6 });
    for (0..generate_mod.degenerate_loop_min_span / 3 + 1) |_| try ids.appendSlice(testing.allocator, &[_]u32{ 101, 102, 103 });

    try testing.expectEqual(LoopAction.close_thought, loopRecoveryAction(&tb, ids.items, 400, 4096));
    // No room to close the thought: the ordinary cut.
    try testing.expectEqual(LoopAction.cut, loopRecoveryAction(&tb, ids.items, 4095, 4096));
    // Once fired, a second loop is cut.
    tb.fired = true;
    try testing.expectEqual(LoopAction.cut, loopRecoveryAction(&tb, ids.items, 400, 4096));
}

test "loop recovery: an answer-side loop and an unarmed request are still cut" {
    const OPEN: u32 = 10;
    const CLOSE: u32 = 11;
    const forced = [_]u32{ 30, CLOSE, 32 };
    var tb = generate_mod.ThinkBound{ .budget = std.math.maxInt(u32), .opener_id = OPEN, .closer_id = CLOSE, .forced = &forced, .in_think = true };
    const closed = [_]u32{ 5, 6, CLOSE, 7, 8 };
    try testing.expectEqual(LoopAction.cut, loopRecoveryAction(&tb, &closed, 5, 4096));
    try testing.expectEqual(LoopAction.cut, loopRecoveryAction(null, &closed, 5, 4096));
}
