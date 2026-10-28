//! The decode tick: grouping, single-slot ticks.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const transformer_mod = @import("transformer.zig");
const generate_mod = @import("generate.zig");
const model_registry_mod = @import("../../host/registry/registry.zig");
const log = @import("../../sdk/util/log.zig");
const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;
const Generator = generate_mod.Generator;
const testing = std.testing;

const runBatchedDecodeTick = @import("batching/tick.zig").runBatchedDecodeTick;
const MAX_BATCH_GROUP = @import("batching/verdict.zig").MAX_BATCH_GROUP;
const noteSerial = @import("batching/verdict.zig").noteSerial;
const groupKeepCount = @import("batching/verdict.zig").groupKeepCount;
const groupCtxMax = @import("batching/verdict.zig").groupCtxMax;
const groupPadsKv = @import("batching/verdict.zig").groupPadsKv;
const groupAttendsPerSlot = @import("batching/verdict.zig").groupAttendsPerSlot;
const batchedPadWaste = @import("batching/verdict.zig").batchedPadWaste;
const fillGroupPadWasteKvLens = @import("batching/verdict.zig").fillGroupPadWasteKvLens;
const finishSlot = @import("finish.zig").finishSlot;
const sumInflightGeneratedTokens = @import("prefill.zig").sumInflightGeneratedTokens;
const Scheduler = @import("scheduler.zig").Scheduler;
const Slot = @import("slot.zig").Slot;
const slotMtpGroupable = @import("spec/mtp_groups.zig").slotMtpGroupable;
const Planner = @import("spec/mtp_groups.zig").Planner;
const tryPlannerTick = @import("spec/mtp_groups.zig").tryPlannerTick;
const mtpCrowdThresholdFor = @import("spec/mtp_groups.zig").mtpCrowdThresholdFor;
const runMtpGroups = @import("spec/mtp_groups.zig").runMtpGroups;
const SpecTickMode = @import("spec/wiring.zig").SpecTickMode;
const specTickMode = @import("spec/wiring.zig").specTickMode;
const plannerOutputClock = @import("spec/wiring.zig").plannerOutputClock;
const publishSpeculativeBlock = @import("spec/wiring.zig").publishSpeculativeBlock;
const loopGuardTick = @import("loop_stop.zig").loopGuardTick;

var kv_skew_split_logged: bool = false; // one-shot log guard

pub fn runDecodeTick(sch: *Scheduler, active: []*Slot) !void {
    if (active.len == 0) return;

    // Publish the in-flight generated-token aggregate for the gauge sampler.
    // Runs after the whole tick (all decode work + any finishSlot calls) on
    // every return path — race-free (inference thread owns these slots' fields)
    // and O(active), never per token. Finished slots are excluded, so the last
    // slot's completion drives this to 0 (live == total at rest).
    defer sch.inflight_generated_tokens.store(sumInflightGeneratedTokens(active), .monotonic);

    // Contention discipline for the spec cost model's kv term: it learns
    // from realized round times, and contention only ever ADDS time. Rather
    // than try to correct for it, a busy server simply stops sampling.
    for (active) |s| {
        if (s.legacy_gen) |*g| g.spec_cost_solo = active.len == 1;
    }

    if (Planner.enabled()) {
        for (active) |slot| if (slot.legacy_gen) |*gen| {
            gen.mtp_planner_width = null;
            gen.mtp_planner_probe = false;
            gen.mtp_planner_recovering = false;
            slot.planner_plain_transition = false;
            slot.planner_force_plain = false;
        };
        if (try tryPlannerTick(sch, active)) return;
        for (active) |slot| if (slot.legacy_gen) |*gen| {
            if (gen.mtp_planner_owned) slot.planner_force_plain = true;
        };
    }
    defer if (Planner.enabled()) for (active) |slot| {
        slot.planner_force_plain = false;
        if (slot.legacy_gen) |*gen| gen.mtp_planner_pending = false;
    };

    // Phase 3 gate: at len==1, route to legacy single-slot path. Bit-identical
    // to pre-Phase-2 behavior including PLD/drafter speculative decoding.
    // Phase A7 test hook: `MLX_SERVE_FORCE_BATCHED=1` bypasses the gate so the
    // byte-equivalence test can run the batched kernel at active.len==1 and
    // assert it matches the single-slot path token-for-token.
    if (active.len == 1 and !sch.force_batched) {
        try runSingleDecodeTick(sch, active[0]);
        return;
    }

    // active.len >= 2 (or force_batched at len==1): split into batchable +
    // non-batchable. Batchable slots share one `forwardBatchedDecode` call.
    // Non-batchable (slots running PLD/drafter or grammar-constrained) fall
    // back to legacy single-slot decode this tick.
    //
    // Plan 05 Phase D: batched decode requires all participating slots to
    // share a transformer. We partition `batchable` by `slot.model` and
    // emit one batched call per model. The non-batchable bucket doesn't
    // care — each slot runs against its own `slot.model.transformer.?`.
    var batchable_buf: [MAX_BATCH_GROUP]*Slot = undefined;
    var batchable_n: usize = 0;
    var mtp_buf: [MAX_BATCH_GROUP]*Slot = undefined;
    var mtp_n: usize = 0;
    for (active) |s| {
        const why = @import("scheduler.zig").batchVerdict(sch, s);
        if (why == .ok and batchable_n < batchable_buf.len) {
            batchable_buf[batchable_n] = s;
            batchable_n += 1;
        } else if (why == .spec_active and slotMtpGroupable(s) and mtp_n < mtp_buf.len) {
            mtp_buf[mtp_n] = s;
            mtp_n += 1;
        } else {
            // legacy single-slot for spec / grammar / overflow
            noteSerial(sch, s, why);
            try runSingleDecodeTick(sch, s);
        }
    }
    // A crowded MTP group (measured past 3 slots on the 27B) beats its sub-grouped verify
    // rounds with ONE plain batched tick; those slots keep their head resumable by
    // capturing the hidden the tick forwards (`mtp_plain_tick`).
    var mtp_group_n: usize = 0;
    {
        std.sort.pdq(*Slot, mtp_buf[0..mtp_n], {}, struct {
            fn lt(_: void, a: *Slot, b: *Slot) bool {
                return @intFromPtr(a.model) < @intFromPtr(b.model);
            }
        }.lt);
        var i: usize = 0;
        while (i < mtp_n) {
            var j = i + 1;
            while (j < mtp_n and mtp_buf[j].model == mtp_buf[i].model) j += 1;
            if (j - i >= mtpCrowdThresholdFor(mtp_buf[i]) and batchable_n + (j - i) <= batchable_buf.len) {
                for (mtp_buf[i..j]) |slot| {
                    const gen = &slot.legacy_gen.?;
                    gen.mtpDetachHead(slot.allocator, true) catch |e| {
                        slot.markError(@errorName(e));
                        continue;
                    };
                    gen.mtp_group_cap = 0;
                    slot.mtp_plain_tick = true;
                    batchable_buf[batchable_n] = slot;
                    batchable_n += 1;
                }
            } else {
                for (mtp_buf[i..j]) |slot| {
                    mtp_buf[mtp_group_n] = slot;
                    mtp_group_n += 1;
                }
            }
            i = j;
        }
    }
    try runMtpGroups(sch, mtp_buf[0..mtp_group_n]);
    if (batchable_n == 0) {
        if (sch.metrics) |m| m.batched_group_size.set(0);
        return;
    }

    // Group batchable slots by model pointer (in-place partition by sort).
    // The order of slots within a model doesn't matter for batched decode;
    // we only need contiguous runs per model.
    std.sort.pdq(*Slot, batchable_buf[0..batchable_n], {}, struct {
        fn lt(_: void, a: *Slot, b: *Slot) bool {
            return @intFromPtr(a.model) < @intFromPtr(b.model);
        }
    }.lt);

    var start: usize = 0;
    while (start < batchable_n) {
        var end = start + 1;
        while (end < batchable_n and batchable_buf[end].model == batchable_buf[start].model) end += 1;
        var group = batchable_buf[start..end];
        // Cap the group by padding waste: the batched kernel pads every slot's
        // KV to the longest in the group, so one long-context stream would make
        // its short neighbours build a tensor orders of magnitude bigger than
        // they need. Sort ascending by kv_len and let `batchedKvKeepCount` say
        // how many still fit; the tail decodes serially this tick.
        if (group.len >= 2 and groupPadsKv(group[0])) {
            var kv_lens: [32]u32 = undefined;
            var ctx_max: u32 = 0;
            {
                var caches_buf: [MAX_BATCH_GROUP]*const KVCache = undefined;
                for (group, 0..) |g, i| {
                    caches_buf[i] = &g.cache;
                }
                fillGroupPadWasteKvLens(caches_buf[0..group.len], group[0].model.config, 1, kv_lens[0..group.len]);
                ctx_max = groupCtxMax(caches_buf[0..group.len]);
                // Stable insertion sort, ascending, slots and lengths moving together.
                var i: usize = 1;
                while (i < group.len) : (i += 1) {
                    const slot_i = group[i];
                    const len_i = kv_lens[i];
                    var j = i;
                    while (j > 0 and kv_lens[j - 1] > len_i) : (j -= 1) {
                        group[j] = group[j - 1];
                        kv_lens[j] = kv_lens[j - 1];
                    }
                    group[j] = slot_i;
                    kv_lens[j] = len_i;
                }
            }
            const keep = groupKeepCount(kv_lens[0..group.len], groupAttendsPerSlot(group[0]), ctx_max);
            if (keep < group.len) {
                if (!kv_skew_split_logged) {
                    kv_skew_split_logged = true;
                    log.info("[batched] pad-waste cap: kept {d} of {d} slots (waste {d:.2}x, kv_len {d}..{d}), rest serial\n", .{
                        keep,
                        group.len,
                        batchedPadWaste(kv_lens[0..group.len]),
                        kv_lens[0],
                        kv_lens[group.len - 1],
                    });
                }
                for (group[keep..]) |s| {
                    noteSerial(sch, s, .pad_waste);
                    try runSingleDecodeTick(sch, s);
                }
                group = group[0..keep];
            }
        }
        // Honor force_batched even when only one slot is batchable so the
        // test hook actually exercises forwardBatchedDecode at N=1.
        if (group.len >= 2 or (sch.force_batched and group.len == 1)) {
            if (sch.metrics) |m| m.batched_group_size.set(group.len);
            try runBatchedDecodeTick(sch, group);
        } else if (group.len == 1) {
            try runSingleDecodeTick(sch, group[0]);
        }
        start = end;
    }
}

pub fn runSingleDecodeTick(sch: *Scheduler, slot: *Slot) !void {
    var inner_err: ?anyerror = null;
    runSingleDecodeTickInner(sch, slot) catch |e| switch (e) {
        // A reasoning-protocol dead end is this request's failure; the tick
        // error path would fail every active slot.
        error.NoValidProtocolToken, error.InvalidProtocolTransition, error.InvalidProtocolPayload => {
            log.err("[grammar] reasoning protocol failed ({s}); failing this request only\n", .{@errorName(e)});
            slot.markError(@errorName(e));
        },
        else => inner_err = e,
    };
    mlx.checkErrorDecode() catch |mlx_err| {
        log.err("[scheduler] decode aborted: MLX failure mid-generation ({s}) — failing this request, the server keeps serving\n", .{@errorName(mlx_err)});
        slot.markError(@errorName(mlx_err));
        return;
    };
    if (inner_err) |e| return e;
}

fn runSingleDecodeTickInner(sch: *Scheduler, slot: *Slot) !void {
    const gen = if (slot.legacy_gen) |*g| g else {
        slot.markError("no_generator");
        return;
    };

    if (gen.speech) |st| {
        const out = try st.next(gen);
        if (out.codes.len > 0) slot.emitAudio(out.codes, out.samples, st.machine.f.sample_rate);
        if (!out.stop) slot.pushTokenWithLogprob(out.token, null);
        slot.was_pad_only = false;
        slot.completion_tokens = gen.completion_tokens;
        if (out.finished) finishSlot(sch, slot, gen.finish_reason);
        return;
    }
    if (try loopGuardTick(sch, slot, gen)) return;
    if (Planner.enabled() and slot.planner_force_plain) {
        gen.mtp_hidden_stale = true;
    }
    if (slot.model.transformer) |xfm| try xfm.ssmGroupRelease(&slot.ctx);

    // NOTE: no `!gen.spec_disabled_runtime` short-circuit here — the
    // generators handle the disabled fallback internally, and `nextPld`'s
    // disabled branch is also where the mid-request RE-ENABLE check lives
    // (bypassing it pinned PLD off for the rest of the request even when the
    // generated tail turned echo-heavy).
    const tick_mode: SpecTickMode = if (Planner.enabled() and slot.planner_force_plain) .regular else specTickMode(
        slot.enable_mtp,
        gen.mtp != null,
        slot.enable_drafter,
        gen.drafter != null,
        gen.dflash != null,
        slot.enable_pld,
        gen.pld_enabled,
        gen.dspark_enabled,
    );
    if (tick_mode == .dspark) {
        const result = try gen.nextDspark(slot.allocator);
        if (result == null) {
            finishSlot(sch, slot, gen.finish_reason);
            return;
        }
        defer slot.allocator.free(result.?.tokens);
        publishSpeculativeBlock(sch, slot, gen, result.?.tokens);
        return;
    }
    if (tick_mode == .mtp) {
        gen.mtp_group_cap = 0;
        const result = try gen.nextMtp(slot.allocator);
        if (result == null) {
            finishSlot(sch, slot, gen.finish_reason);
            return;
        }
        defer slot.allocator.free(result.?.tokens);
        publishSpeculativeBlock(sch, slot, gen, result.?.tokens);
        return;
    }

    if (tick_mode == .drafter) {
        const result = try gen.nextDrafter(slot.allocator);
        if (result == null) {
            finishSlot(sch, slot, gen.finish_reason);
            return;
        }
        defer slot.allocator.free(result.?.tokens);
        publishSpeculativeBlock(sch, slot, gen, result.?.tokens);
        return;
    }

    if (tick_mode == .dflash) {
        const result = try gen.nextDflash(slot.allocator);
        if (result == null) {
            finishSlot(sch, slot, gen.finish_reason);
            return;
        }
        defer slot.allocator.free(result.?.tokens);
        publishSpeculativeBlock(sch, slot, gen, result.?.tokens);
        return;
    }

    if (tick_mode == .pld) {
        const result = try gen.nextPld(slot.allocator, slot.pld_draft_len, slot.pld_key_len);
        if (result == null) {
            finishSlot(sch, slot, gen.finish_reason);
            return;
        }
        defer slot.allocator.free(result.?.tokens);
        publishSpeculativeBlock(sch, slot, gen, result.?.tokens);
        return;
    }

    // Regular path. A request that never armed MTP teaches the round-cost table what a plain
    // serial token costs; `observeSerialTick` owns the drop rules and `serialCellWanted`.
    const tok_opt = try gen.next(slot.allocator);
    if (tok_opt == null) {
        finishSlot(sch, slot, gen.finish_reason);
        return;
    }
    gen.observeSerialTick();
    if (Planner.enabled()) {
        if (slot.planner_force_plain) gen.mtp_planner_plain_ticks += 1;
        plannerOutputClock(slot, gen);
    }
    const t = tok_opt.?;
    // Phase A5: capture per-token logprob. `gen.last_logprob` ownership
    // transfers into slot.logprobs_buf (gen sets the field, we null it here).
    // Published in the SAME critical section as the token so a streaming
    // reader that has been handed token i can read entry i — see
    // `pushTokenWithLogprob`.
    var lp_take: ?generate_mod.LogprobResult = null;
    if (slot.logprobs_n > 0) {
        if (gen.last_logprob) |lp| {
            lp_take = lp;
            gen.last_logprob = null;
        }
    }
    // Publish the reasoning→payload boundary (if this is the token that
    // carries it) BEFORE the token itself, so a streaming reader handed token
    // i already sees its span.
    if (gen.takeConstraintSpan()) |span| {
        slot.publishConstraintSpan(span.token_index, span.byte_offset);
    }
    slot.pushTokenWithLogprob(t, lp_take);
    if (t != 0) slot.was_pad_only = false;
    slot.completion_tokens = gen.completion_tokens;
}

// ── tests ──

test "single MTP slot reaches the round entry through runDecodeTick" {
    var xfm: @import("../../plugins/qwen4_exp/trunk.zig").Trunk = undefined;
    var model: model_registry_mod.LoadedModel = undefined;
    model.transformer = null;
    var gen: Generator = undefined;
    gen.mtp_planner_owned = false;
    gen.speech = null;
    gen.generated_ids = .empty;
    gen.loop_guard_start = 0;
    gen.mtp = .{ .qwen4 = &xfm };
    gen.mtp_cache = .{ .qwen = undefined };
    gen.has_last_hidden = true;
    gen.drafter = null;
    gen.dflash = null;
    gen.pld_enabled = false;
    gen.dspark_enabled = false;
    gen.done = false;
    gen.sampling = .{};
    gen.logprobs_n = 1;
    var slot: Slot = undefined;
    slot.allocator = testing.allocator;
    slot.model = &model;
    slot.legacy_gen = gen;
    slot.enable_mtp = true;
    slot.enable_drafter = false;
    slot.enable_pld = false;
    slot.finished = false;
    slot.error_code = null;
    slot.cancelled = std.atomic.Value(bool).init(false);
    slot.completion_tokens = 0;
    var sch: Scheduler = undefined;
    sch.force_batched = false;
    sch.inflight_generated_tokens = std.atomic.Value(u64).init(0);
    var active = [_]*Slot{&slot};
    try testing.expectError(error.SpecDecodeUnsupported, runDecodeTick(&sch, &active));
    try testing.expect(slot.legacy_gen.?.spec_cost_solo);
    try testing.expectEqual(@as(u32, 0), slot.legacy_gen.?.mtp_group_cap);
    try testing.expectEqual(@as(u64, 0), sch.inflight_generated_tokens.load(.monotonic));
}
