//! The batched decode tick.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const generate_mod = @import("../generate.zig");
const log = @import("../../../sdk/util/log.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const Transformer = transformer_mod.Transformer;
const ForwardCtx = transformer_mod.ForwardCtx;
const group_cost_mod = @import("../spec/mtp_group_cost.zig");

const MAX_BATCH_GROUP = @import("verdict.zig").MAX_BATCH_GROUP;
const runSingleDecodeTick = @import("../decode.zig").runSingleDecodeTick;
const finishSlot = @import("../finish.zig").finishSlot;
const Scheduler = @import("../scheduler.zig").Scheduler;
const Slot = @import("../slot.zig").Slot;
const batchedTickAction = @import("../spec/mtp_groups.zig").batchedTickAction;
const Planner = @import("../spec/mtp_groups.zig").Planner;
const plannerShape = @import("../spec/mtp_groups.zig").plannerShape;
const plannerOutputClock = @import("../spec/wiring.zig").plannerOutputClock;

pub fn runBatchedDecodeTick(sch: *Scheduler, active: []*Slot) !void {
    var inner_err: ?anyerror = null;
    runBatchedDecodeTickInner(sch, active) catch |e| {
        inner_err = e;
    };
    mlx.checkErrorDecode() catch |mlx_err| {
        log.err("[scheduler] batched decode aborted: MLX failure ({s}) — failing all {d} slots in the group\n", .{ @errorName(mlx_err), active.len });
        for (active) |s| s.markError(@errorName(mlx_err));
        return;
    };
    if (inner_err) |e| return e;
}

fn runBatchedDecodeTickInner(sch: *Scheduler, active: []*Slot) !void {
    const N = active.len;
    if (N == 0) return;
    const allocator = sch.allocator;

    // Phase D: all batched slots must share the same model — the caller
    // (`runDecodeTick`) partitions by `slot.model` before dispatching here.
    // The first slot's transformer is authoritative; debug-assert the rest
    // match to surface partitioning bugs early.
    const xfm_ptr: *Transformer = active[0].model.transformer.?;
    if (std.debug.runtime_safety) {
        for (active) |s| std.debug.assert(s.model.transformer.? == xfm_ptr);
    }

    // The forward records its attention and MoE arms on qwen4_exp's trunk.
    const plain_trunk = xfm_ptr.qwen4Trunk();
    const observe_plain = Planner.enabled() and plain_trunk != null and N <= Planner.MAX_ROWS;
    var price_watch = io_util.Stopwatch.init(active[0].io);
    var price_key: group_cost_mod.GroupShape = .{};
    var price_clean = observe_plain;
    if (observe_plain) {
        const zeros: [Planner.MAX_ROWS]u8 = @splat(0);
        price_key = plannerShape(active, zeros[0..N]);
        plain_trunk.?.cost_trace_active = true;
        plain_trunk.?.cost_attention = 0;
        plain_trunk.?.cost_moe = 0;
        for (active) |slot| {
            const gen = &slot.legacy_gen.?;
            if (gen.has_pending_logits or gen.has_pending_token or slot.planner_plain_transition or slot.mtp_plain_tick) price_clean = false;
        }
    }
    defer if (observe_plain) {
        plain_trunk.?.cost_trace_active = false;
    };

    // Legacy→batched transition: a slot arriving from a legacy single-slot
    // tick (or fresh from prefill) carries lazy pipeline state — a lookahead
    // token ALREADY FORWARDED into its KV cache plus `pending_logits` for
    // the position after it. Consume that state via `drainPipelineForBatch`
    // (emit the lookahead, sample the new next_token_id from the pending
    // logits). Dropping it and re-forwarding `next_token_id` — the pre-fix
    // behavior — appended a duplicate cache position and re-emitted an
    // already-emitted token, corrupting any stream whose slot joined a
    // batch mid-generation (tests/test_batched_transition.sh). Slots that
    // finish during the drain are excluded from the batch.
    const live = try allocator.alloc(*Slot, N);
    defer allocator.free(live);
    var live_n: usize = 0;
    for (active) |slot| {
        const gen = if (slot.legacy_gen) |*g| g else {
            slot.markError("no_generator");
            continue;
        };
        if (gen.has_pending_logits or gen.has_pending_token) {
            const emitted = gen.drainPipelineForBatch(slot.allocator) catch |err| {
                slot.markError(@errorName(err));
                continue;
            };
            if (emitted) |tok| {
                slot.pushToken(tok);
                if (Planner.enabled() and gen.mtp_planner_owned) {
                    gen.mtp_planner_plain_ticks += 1;
                    plannerOutputClock(slot, gen);
                }
                if (tok != 0) slot.was_pad_only = false;
                slot.completion_tokens = gen.completion_tokens;
                if (generate_mod.isEosId(gen.next_token_id, slot.eos_token_ids)) {
                    finishSlot(sch, slot, "stop");
                    continue;
                }
                if (slot.completion_tokens >= slot.max_tokens) {
                    finishSlot(sch, slot, "length");
                    continue;
                }
            } else {
                // checkStop fired on the pipelined lookahead (EOS / pad-run
                // / max_tokens / timeout); nothing to emit.
                slot.completion_tokens = gen.completion_tokens;
                finishSlot(sch, slot, gen.finish_reason);
                continue;
            }
        }
        live[live_n] = slot;
        live_n += 1;
    }
    if (live_n == 0) return;
    const batch = live[0..live_n];

    // Build inputs.
    const next_tokens = try allocator.alloc(u32, live_n);
    defer allocator.free(next_tokens);
    const ctxs = try allocator.alloc(*ForwardCtx, live_n);
    defer allocator.free(ctxs);
    const rope_offsets = try allocator.alloc(u32, live_n);
    defer allocator.free(rope_offsets);

    for (batch, 0..) |slot, i| {
        const gen = &slot.legacy_gen.?;
        next_tokens[i] = gen.next_token_id;
        ctxs[i] = &gen.ctx;
    }

    // Two batched kernels: the standard one, and the GatedDeltaNet twin for
    // hybrid trunks (qwen3_5 family). `batchedGdnReady` is the runtime half of
    // the gate — a slot that has not prefilled yet carries no recurrent state
    // to merge, so that tick stays serial rather than merging a wrong width.
    const use_gdn = xfm_ptr.supportsBatchedGdnDecode() and xfm_ptr.batchedGdnReady(ctxs);
    // Position source is per PATH: a GDN trunk positions from the slot's
    // `moe_seq_offset` — `KVCache.step` only advances on layer 0, which is a
    // linear layer there, so it reads 0 forever and every batched token was
    // roped at position 0 (qwen3_5 batched diverged from serial at token 14).
    for (batch, 0..) |slot, i| rope_offsets[i] = @intCast(if (use_gdn) slot.moe_seq_offset else slot.cache.step);
    if (xfm_ptr.supportsBatchedGdnDecode() and !use_gdn) {
        // A slot with no recurrent state yet cannot join the merge. Decode the
        // group serially this tick instead of skipping it — skipping advances
        // nothing, so a group that never becomes ready would spin forever.
        for (batch) |s| try runSingleDecodeTick(sch, s);
        return;
    }
    var want_hidden = false;
    for (batch) |slot| {
        want_hidden = want_hidden or slot.mtp_plain_tick;
        if (Planner.enabled() and slot.legacy_gen.?.mtp_planner_owned) {
            const gen = &slot.legacy_gen.?;
            if (slot.mtp_plain_tick) gen.mtp_planner_prime_ticks += 1 else {
                gen.mtp_hidden_stale = true;
                gen.mtp_planner_plain_ticks += 1;
            }
        }
    }
    var hidden_rows: ?[]mlx.mlx_array = null;
    defer if (hidden_rows) |rows| {
        for (rows) |a| _ = mlx.mlx_array_free(a);
        allocator.free(rows);
    };
    const logits_arr = if (use_gdn)
        try xfm_ptr.forwardMoeBatchedDecode(next_tokens, ctxs, rope_offsets, if (want_hidden) &hidden_rows else null)
    else
        try xfm_ptr.forwardBatchedDecode(next_tokens, ctxs, rope_offsets);
    defer {
        for (logits_arr) |a| _ = mlx.mlx_array_free(a);
        allocator.free(logits_arr);
    }
    if (hidden_rows) |rows| {
        for (batch, 0..) |slot, i| {
            if (!slot.mtp_plain_tick) continue;
            slot.mtp_plain_tick = false;
            const gen = &slot.legacy_gen.?;
            if (gen.has_last_hidden) _ = mlx.mlx_array_free(gen.last_hidden);
            gen.last_hidden = rows[i];
            rows[i] = mlx.mlx_array_new();
            gen.has_last_hidden = true;
            if (Planner.enabled()) gen.mtp_hidden_stale = false;
        }
    }
    for (batch) |slot| slot.mtp_plain_tick = false;
    // The batched forward advances only its scratch offset; each slot's own
    // position moves here so a slot leaving the batch resumes serial from
    // the right place (qwen4's QSA reads it for kv length + tail rule).
    for (batch) |slot| slot.moe_seq_offset += 1;

    // `gen.sampling`, not `slot.sampling`: the Generator's copy passed
    // the initWithOptions chokepoint and carries the model's
    // reserved-token suppression mask; the slot's copy is the raw
    // request params.
    var sample_params: [MAX_BATCH_GROUP]generate_mod.SamplingParams = undefined;
    var sample_rows: [MAX_BATCH_GROUP]mlx.mlx_array = undefined;
    var sample_ids: [MAX_BATCH_GROUP]i32 = undefined;
    std.debug.assert(live_n == logits_arr.len);
    std.debug.assert(live_n <= sample_params.len);
    for (batch, 0..) |slot, i| {
        sample_params[i] = slot.legacy_gen.?.sampling;
        sample_rows[i] = logits_arr[i];
    }
    if (live_n > 0) {
        try generate_mod.sampleRows(sample_ids[0..live_n], sample_rows[0..live_n], sample_params[0..live_n], xfm_ptr.s);
        for (batch, 0..) |slot, i| slot.legacy_gen.?.sampling.draw = sample_params[i].draw;
    }

    // Sample per slot, emit prev id, set new next_token_id.
    for (batch, 0..) |slot, i| {
        const gen = &slot.legacy_gen.?;
        const act = batchedTickAction(slot.cancelled.load(.acquire));
        const sampled: ?i32 = if (act.publish) sample_ids[i] else null;

        const emit = gen.next_token_id;
        gen.generated_ids.append(slot.allocator, emit) catch |err| {
            slot.markError(@errorName(err));
            continue;
        };
        gen.advanceStep(1);
        if (emit != 0) slot.was_pad_only = false;
        const val = sampled orelse continue;
        gen.next_token_id = @intCast(val);
        if (Planner.enabled()) plannerOutputClock(slot, gen);

        // Stop checks (mirrors Generator.checkStop).
        if (generate_mod.isEosId(emit, slot.eos_token_ids)) {
            // Per existing contract, emit IS NOT yielded when it's EOS — the
            // STOP token comes BEFORE the yield. But here the cache has
            // already moved past it. The legacy path's checkStop runs on the
            // NEXT token (it's checked before emit). To preserve that
            // behavior we emit and then mark finished if `next_token_id` is
            // EOS (i.e. STOP is the next sampled token, ignored).
            slot.pushToken(emit);
            slot.completion_tokens = gen.completion_tokens;
            // not finished yet; next tick's checkStop on next_token_id ends it
        } else {
            slot.pushToken(emit);
            slot.completion_tokens = gen.completion_tokens;
        }

        if (generate_mod.isEosId(gen.next_token_id, slot.eos_token_ids)) {
            finishSlot(sch, slot, "stop");
            continue;
        }
        if (gen.next_token_id == 0) {
            gen.consecutive_pad += 1;
            if (gen.consecutive_pad >= 3) {
                finishSlot(sch, slot, "stop");
                continue;
            }
        } else {
            gen.consecutive_pad = 0;
        }
        if (slot.completion_tokens >= slot.max_tokens) {
            finishSlot(sch, slot, "length");
            continue;
        }
    }
    if (price_clean and live_n == N) {
        var gaps: [Planner.MAX_ROWS]f32 = undefined;
        for (batch, 0..) |slot, i| {
            if (slot.state != .decoding or slot.cancelled.load(.acquire)) return;
            gaps[i] = slot.mtp_publish_gap_ms;
        }
        price_key.attention = plain_trunk.?.cost_attention;
        price_key.moe = plain_trunk.?.cost_moe;
        const emitted: [Planner.MAX_ROWS]u32 = @splat(1);
        const ms = @as(f32, @floatFromInt(price_watch.read())) / std.time.ns_per_ms;
        _ = xfm_ptr.mtp_group_cost.observeShape(price_key, ms, emitted[0..N], gaps[0..N], .round);
    }
}
