//! One chat pass of the inference loop: admission, prefill, decode, live KV residency.

const std = @import("std");
const err_name = @import("../../sdk/util/err_name.zig");
const transformer_mod = @import("transformer.zig");
const prefix_cache_mod = @import("cache/prefix_cache.zig");
const metrics_mod = @import("../../host/metrics/metrics.zig");
const log = @import("../../sdk/util/log.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const sleep_inhibit = @import("runtime/sleep_inhibit.zig");
const testing = std.testing;

const slotHoldsForMemory = @import("admission/pending.zig").slotHoldsForMemory;
const AdmitCand = @import("admission/pending.zig").AdmitCand;
const admitPendingTick = @import("admission/pending.zig").admitPendingTick;
const slotExclusiveDecode = @import("batching/verdict.zig").slotExclusiveDecode;
const BUDGET_REVISE_WINDOW_NS = @import("cache/budget.zig").BUDGET_REVISE_WINDOW_NS;
const reviseHotCacheBudgets = @import("cache/budget.zig").reviseHotCacheBudgets;
const flushOwedDiskSlices = @import("cache/budget.zig").flushOwedDiskSlices;
const holdsSleepWhileParked = @import("cache/budget.zig").holdsSleepWhileParked;
const runDecodeTick = @import("decode.zig").runDecodeTick;
const finishSlot = @import("finish.zig").finishSlot;
const runPrefill = @import("prefill.zig").runPrefill;
const prefillDecodeShare = @import("prefill_interleave.zig").prefillDecodeShare;
const specYieldTick = @import("prefill_interleave.zig").specYieldTick;
const orderByTail = @import("prefill_interleave.zig").orderByTail;
const emptySlotCache = @import("prefill_interleave.zig").emptySlotCache;
const slotUncachedTail = @import("prefill_interleave.zig").slotUncachedTail;
const interleaveDecodeTick = @import("prefill_interleave.zig").interleaveDecodeTick;
const Scheduler = @import("scheduler.zig").Scheduler;
const promiseMaterialized = @import("scheduler/params.zig").promiseMaterialized;
const Slot = @import("slot.zig").Slot;
const publishFirstToken = @import("spec/wiring.zig").publishFirstToken;
const hasWorkPendingLocked = @import("runtime/loop.zig").hasWorkPendingLocked;
const drainSideQueues = @import("runtime/loop.zig").drainSideQueues;

/// Caller holds queue_mu. Recheck arrivals under the lock before the short, wakeable retry wait.
fn waitForDiskRetryLocked(sch: *Scheduler) void {
    if (hasWorkPendingLocked(sch) or sch.shutdown.load(.acquire)) return;
    sch.queue_cond.waitTimeout(sch.io, &sch.queue_mu, .{ .duration = .{ .raw = .fromMilliseconds(10), .clock = .boot } }) catch {};
}

const ChatPassMode = enum { main, yield };
pub const ChatPassResult = enum { shutdown, idle, ran };

/// What became of a slot `prefillAdmitted` took through admission and prefill.
const AdmitOutcome = enum {
    /// Prefilled and appended to `decoding`.
    decoding,
    /// Finished before or during its prefill (cancelled).
    finished,
    /// Failed; the slot carries the error.
    errored,
    /// Does not fit beside live requests yet: the caller puts it back in `pending`.
    held,
    /// After the hot-cache restore the uncached tail is over the slot's boundary limit.
    tail_too_large,
};

/// One admitted slot's cancelled check, memory hold, prefill (with the one QSA-gap retry), first-token
/// publish and move into `decoding`. The caller holds the slot's `in_pass`.
pub fn prefillAdmitted(sch: *Scheduler, slot: *Slot) AdmitOutcome {
    if (slot.cancelled.load(.acquire)) {
        // finishSlot (not raw markFinished) so the metrics sink
        // counts the cancellation; safe pre-prefill — commit
        // no-ops with legacy_gen==null.
        finishSlot(sch, slot, "cancelled");
        return .finished;
    }
    if (slotHoldsForMemory(sch, slot)) return .held;
    var prefill_sw = io_util.Stopwatch.init(sch.io);
    var qsa_gap_retried = false;
    prefill: while (true) {
        runPrefill(sch, slot) catch |err| {
            if (err == error.Cancelled) {
                log.info("[scheduler] prefill aborted: client disconnected\n", .{});
                finishSlot(sch, slot, "cancelled");
                return .finished;
            }
            if (err == error.BoundaryTailTooLarge) return .tail_too_large;
            if (err == error.QsaHistoryGap and !qsa_gap_retried) {
                if (slot.model.prefix_cache) |hc| _ = prefix_cache_mod.HotPrefixCache.dropQsaGapEntry(hc);
                emptySlotCache(slot);
                if (slot.legacy_gen) |*g| {
                    g.deinit(slot.allocator);
                    slot.legacy_gen = null;
                }
                slot.skip_prefix_cache = true;
                log.warn("[hot-cache] restored entry failed the QSA history check — dropped, cold prefill\n", .{});
                qsa_gap_retried = true;
                continue :prefill;
            }
            log.err("[scheduler] prefill failed for slot: {s}\n", .{@errorName(err)});
            slot.markError(@errorName(err));
            return .errored;
        };
        break :prefill;
    }
    if (slot.state == .errored or slot.cancelled.load(.acquire)) return .errored;
    slot.prefill_ns = prefill_sw.read() -| slot.prefill_interleaved_ns;
    slot.emitPrefilled();
    publishFirstToken(slot);
    if (slot.prefill_interleaved_ns > 0) log.debug("[interleave] prefill {d} ms, hosted decode {d} ms\n", .{
        slot.prefill_ns / std.time.ns_per_ms, slot.prefill_interleaved_ns / std.time.ns_per_ms,
    });
    // Exact time-to-first-token: elapsed from request arrival
    // (Slot.init, pre-queue-wait) to prefill completion. Captured
    // here rather than derived by subtraction in finishSlot, so a
    // slot that finishes mid-tick can't skew it (metrics TTFT fix).
    slot.first_token_ns = @intCast(slot.request_start_ts.untilNow(sch.io, .boot).nanoseconds);
    sch.queue_mu.lockUncancelable(sch.io);
    sch.decoding.append(sch.allocator, slot) catch |err| {
        sch.queue_mu.unlock(sch.io);
        slot.markError(@errorName(err));
        return .errored;
    };
    sch.queue_mu.unlock(sch.io);
    return .decoding;
}

/// One admission -> prefill -> decode tick -> cull pass. `.main` parks while there is no
/// work at all; `.yield` runs inside a media job, so it never parks and never touches the
/// load, unload or gen queues. `.ran` = something advanced.
pub fn chatPass(sch: *Scheduler, mode: ChatPassMode) ChatPassResult {
    // 1. Wait for work. Drain pending slots into a local list under lock,
    //    run prefills outside the lock.
    var ran = mode == .yield and drainSideQueues(sch);
    var to_prefill: [16]*Slot = undefined;
    var n_prefill: usize = 0;
    {
        sch.queue_mu.lockUncancelable(sch.io);
        defer sch.queue_mu.unlock(sch.io);
        if (mode == .main) {
            var disk_owed = true;
            while (!hasWorkPendingLocked(sch) and !sch.shutdown.load(.acquire)) {
                // Idle: land owed SSD slices before parking, so a later commit rarely finds one.
                if (disk_owed) {
                    sch.queue_mu.unlock(sch.io);
                    const flushed = flushOwedDiskSlices(sch);
                    sch.queue_mu.lockUncancelable(sch.io);
                    disk_owed = flushed != .done;
                    if (flushed == .busy) waitForDiskRetryLocked(sch);
                    continue;
                }
                // No later tick runs while parked, so release here.
                sleep_inhibit.setActive(holdsSleepWhileParked(sch));
                sch.queue_cond.waitUncancelable(sch.io, &sch.queue_mu);
            }
            // Hold until the loop parks again.
            if (!sch.shutdown.load(.acquire)) sleep_inhibit.setActive(true);
        }
        if (sch.shutdown.load(.acquire)) return .shutdown;

        // If only vision/embed/cleanup/load work is pending, loop back to drain it.
        if (sch.pending.items.len == 0 and sch.decoding.items.len == 0) return if (ran) .ran else .idle;

        // Single-flight admission (the dsv4 class): a model with
        // MODULE-OWNED decode state admits at most one live slot.
        // Snapshot live exclusive-model slots (same liveness predicate
        // as the step-3 active list), let `admitPendingTick` decide,
        // and leave held slots in `pending` — their conn threads keep
        // flowing SSE keepalives while they wait, and the wait
        // condition above never blocks while `pending` is non-empty,
        // so a held slot admits on the first tick after the active one
        // is culled (step 5, same mutex).
        var live_buf: [32]AdmitCand = undefined;
        var n_live: usize = 0;
        for (sch.decoding.items) |s| {
            if (s.cancelled.load(.acquire) or s.finished or s.error_code != null) continue;
            if (!slotExclusiveDecode(s)) continue;
            if (n_live >= live_buf.len) break;
            live_buf[n_live] = .{ .model = @intFromPtr(s.model), .exclusive = true };
            n_live += 1;
        }
        var cand_buf: [32]AdmitCand = undefined;
        const n_cands = @min(sch.pending.items.len, cand_buf.len);
        for (sch.pending.items[0..n_cands], 0..) |s, i| {
            cand_buf[i] = .{ .model = @intFromPtr(s.model), .exclusive = slotExclusiveDecode(s) };
        }
        var admit_idx: [to_prefill.len]usize = undefined;
        const n_admit = admitPendingTick(cand_buf[0..n_cands], live_buf[0..n_live], &admit_idx);
        for (admit_idx[0..n_admit]) |idx| {
            to_prefill[n_prefill] = sch.pending.items[idx];
            _ = to_prefill[n_prefill].in_pass.fetchAdd(1, .acq_rel);
            n_prefill += 1;
        }
        // Remove admitted entries in DESCENDING index order so the
        // earlier (ascending) indices stay valid during removal.
        var r = n_admit;
        while (r > 0) {
            r -= 1;
            _ = sch.pending.orderedRemove(admit_idx[r]);
        }
        @memcpy(sch.prefilling[0..n_prefill], to_prefill[0..n_prefill]);
        sch.prefilling_from = 0;
        sch.n_prefilling = n_prefill;
    }

    if (sch.budget_revise_sw) |sw| {
        if (sw.read() > BUDGET_REVISE_WINDOW_NS) sch.budget_revise_sw = null;
        if (n_prefill > 0) reviseHotCacheBudgets(sch, true);
    }

    // A batch admitted together prefills in order. A long cold prompt first would hold every short
    // request in the batch until it ends, and the chunk-boundary hook never sees them because
    // they already left `pending`. With a decode share, the smallest uncached tail goes first.
    if (n_prefill > 1 and prefillDecodeShare() > 0) {
        var tails: [to_prefill.len]usize = undefined;
        var order: [to_prefill.len]usize = undefined;
        for (to_prefill[0..n_prefill], 0..) |s, i| tails[i] = slotUncachedTail(s);
        orderByTail(tails[0..n_prefill], order[0..n_prefill]);
        var sorted: [to_prefill.len]*Slot = undefined;
        for (order[0..n_prefill], 0..) |src, i| sorted[i] = to_prefill[src];
        @memcpy(to_prefill[0..n_prefill], sorted[0..n_prefill]);
        // `prefilling_from` counts positions in this order.
        sch.queue_mu.lockUncancelable(sch.io);
        @memcpy(sch.prefilling[0..n_prefill], to_prefill[0..n_prefill]);
        sch.queue_mu.unlock(sch.io);
    }

    // 2. Prefill each pending slot (heavy; mlx ops on this thread).
    //    The inference thread is the sole mlx caller post-cleanup, so
    //    no per-tick stream rebind / mutex coexistence is needed.
    if (n_prefill > 0) {
        defer {
            sch.queue_mu.lockUncancelable(sch.io);
            sch.prefilling_from = 0;
            sch.n_prefilling = 0;
            sch.queue_mu.unlock(sch.io);
        }
        for (to_prefill[0..n_prefill], 0..) |slot, pi| {
            defer _ = slot.in_pass.fetchSub(1, .acq_rel);
            // Unlisted BEFORE its pass ends: `complete()` (which may free the slot) waits on
            // `in_pass`, so a listed slot is always alive for `cancelAllInFlight`.
            defer {
                sch.queue_mu.lockUncancelable(sch.io);
                sch.prefilling_from = @max(sch.prefilling_from, pi + 1);
                sch.queue_mu.unlock(sch.io);
            }
            // Between the slots of one admitted batch, tick the streams
            // that just started decoding — a single-chunk prefill exposes
            // no chunk-boundary yield, so without this every slot's first
            // token waits for the LAST slot's prefill (the TTFT
            // staircase collapse).
            if (pi > 0) _ = interleaveDecodeTick(sch);
            switch (prefillAdmitted(sch, slot)) {
                .held => {
                    // Back to the head of `pending`, in order; the decode tick below runs first.
                    sch.queue_mu.lockUncancelable(sch.io);
                    defer sch.queue_mu.unlock(sch.io);
                    sch.n_prefilling = pi; // back in `pending`, which `cancelAllInFlight` walks
                    var r = n_prefill;
                    while (r > pi) {
                        r -= 1;
                        sch.pending.insert(sch.allocator, 0, to_prefill[r]) catch {
                            to_prefill[r].markError("OutOfMemory");
                        };
                        if (r > pi) _ = to_prefill[r].in_pass.fetchSub(1, .acq_rel);
                    }
                    break;
                },
                .decoding, .finished, .errored, .tail_too_large => ran = true,
            }
        }
    }

    // 3. Build active-list snapshot (skip cancelled / finished / errored).
    var active: std.ArrayList(*Slot) = .empty;
    defer active.deinit(sch.allocator);
    {
        sch.queue_mu.lockUncancelable(sch.io);
        defer sch.queue_mu.unlock(sch.io);
        for (sch.decoding.items) |s| {
            if (s.cancelled.load(.acquire) or s.finished or s.error_code != null) continue;
            active.append(sch.allocator, s) catch break;
            _ = s.in_pass.fetchAdd(1, .acq_rel);
        }
    }
    defer for (active.items) |s| {
        _ = s.in_pass.fetchSub(1, .acq_rel);
    };

    specYieldTick(active.items);

    // 4. Decode tick. Charge the full wall-clock tick time to each
    //    participating slot — for batched ticks this matches the per-slot
    //    throughput a user actually observes (their stream advances at
    //    the tick cadence regardless of how many peers share it).
    if (active.items.len > 0) {
        ran = true;
        var decode_sw = io_util.Stopwatch.init(sch.io);
        runDecodeTick(sch, active.items) catch |err| {
            log.err("[scheduler] decode tick failed: {s}\n", .{if (err == error.EngineFailed) err_name.last() else @errorName(err)});
            for (active.items) |s| s.markError(@errorName(err));
        };
        const tick_ns = decode_sw.read();
        for (active.items) |s| s.decode_ns +|= tick_ns;
    }

    // 5. Cull finished / errored / cancelled from `decoding`. The slot
    //    still belongs to its connection thread until that thread calls
    //    `complete`; we just stop touching it.
    // A slot cancelled here by the engine (SIGTERM's `cancelAllInFlight`) tells its stream, whose reader waits on
    // the stream, not the slot; a client's cancel reached the stream first. Sent past `queue_mu` (the stream locks).
    var cancelled: [64]*Slot = undefined;
    var n_cancelled: usize = 0;
    const module_bytes = @import("cache/budget.zig").moduleStateBytes(sch);
    {
        sch.queue_mu.lockUncancelable(sch.io);
        defer sch.queue_mu.unlock(sch.io);
        var i: usize = 0;
        while (i < sch.decoding.items.len) {
            const s = sch.decoding.items[i];
            const drop = s.cancelled.load(.acquire) or s.finished or s.error_code != null;
            const grown = if (s.legacy_gen) |*g| promiseMaterialized(g.completion_tokens) else false;
            if (drop or grown) {
                sch.promise_ledger.release(s.memory_promised);
                s.memory_promised = 0;
            }
            if (drop) {
                if (!s.finished and s.error_code == null) {
                    // Past the buffer, the slot stays for the next step's cull to tell.
                    if (n_cancelled == cancelled.len) {
                        i += 1;
                        continue;
                    }
                    cancelled[n_cancelled] = s;
                    n_cancelled += 1;
                }
                _ = sch.decoding.orderedRemove(i);
            } else i += 1;
        }
        publishLiveKvResidency(sch, null, module_bytes);
    }
    for (cancelled[0..n_cancelled]) |s| {
        @import("finish.zig").releasePluginSlot(s);
        s.markFinished("cancelled");
    }
    if (n_cancelled > 0) @import("cache/budget.zig").publishHotCacheResidency(sch);
    return if (ran) .ran else .idle;
}

/// Caller holds `queue_mu`; inference thread only (it owns the slots' arrays).
/// `prefilling` is the slot mid-prefill, which is not in `decoding` yet.
/// Read `module_bytes` before taking `queue_mu`; publication does not access the registry.
pub fn publishLiveKvResidency(sch: *Scheduler, prefilling: ?*Slot, module_bytes: u64) void {
    const observe = sch.metrics != null;
    sch.live_session_count = 0;
    var bytes = module_bytes;
    if (prefilling) |p| {
        const b = slotStateBytes(p);
        bytes += b;
        if (observe) recordLiveSession(sch, p, .prefill, b);
    }
    for (sch.decoding.items) |s| {
        const b = slotStateBytes(s);
        bytes += b;
        if (observe) recordLiveSession(sch, s, .decode, b);
    }
    sch.resident_live_kv_bytes.store(bytes, .monotonic);
}

fn recordLiveSession(sch: *Scheduler, s: *const Slot, phase: metrics_mod.Session.Phase, state_bytes: u64) void {
    if (sch.live_session_count == sch.live_sessions.len) return;
    const prompt: u32 = if (phase == .prefill) @intCast(s.full_prompt.len) else s.prompt_tokens;
    sch.live_sessions[sch.live_session_count] = .init(s.model.id, phase, prompt + s.completion_tokens, s.cached_tokens, s.completion_tokens, state_bytes);
    sch.live_sessions[sch.live_session_count].entry_id = s.restored_entry;
    sch.live_sessions[sch.live_session_count].request_id = s.request_id;
    sch.live_sessions[sch.live_session_count].client = s.client;
    sch.live_session_count += 1;
}

fn slotStateBytes(s: *const Slot) u64 {
    var bytes = s.cache.residentBytes();
    if (s.ssm_entries) |ents| for (ents) |*e| {
        bytes += transformer_mod.ssmEntryBytes(e);
    };
    return bytes;
}

// ── tests ──

test "a busy disk retry parks until its bounded timeout or pending work" {
    var sch: Scheduler = undefined;
    sch.io = testing.io;
    sch.queue_mu = .init;
    sch.queue_cond = .init;
    sch.shutdown = .init(false);
    sch.pending = .empty;
    sch.decoding = .empty;
    sch.vision_queue = .empty;
    sch.embed_queue = .empty;
    sch.cleanup_queue = .empty;
    sch.load_queue = .empty;
    sch.gen_queue = .empty;
    sch.unload_queue = .empty;
    sch.queue_mu.lockUncancelable(sch.io);
    defer sch.queue_mu.unlock(sch.io);
    const before = io_util.Stopwatch.init(sch.io);
    waitForDiskRetryLocked(&sch);
    try testing.expect(before.read() >= 5 * std.time.ns_per_ms);
    sch.shutdown.store(true, .release);
    waitForDiskRetryLocked(&sch);
}
