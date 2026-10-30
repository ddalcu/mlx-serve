//! The inference loop: work-pending checks, side-queue drains, the main loop.

const std = @import("std");
const log = @import("../../../sdk/util/log.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const mlx_stream = @import("../modules/stream.zig");
const sleep_inhibit = @import("sleep_inhibit.zig");
const testing = std.testing;

const reviseHotCacheBudgets = @import("../scheduler.zig").reviseHotCacheBudgets;
const persistAtShutdown = @import("../scheduler.zig").persistAtShutdown;
const commitSlotIfApplicable = @import("../commit.zig").commitSlotIfApplicable;
const recordSlotCleanup = @import("../finish.zig").recordSlotCleanup;
const releasePluginSlot = @import("../finish.zig").releasePluginSlot;
const publishHotCacheResidency = @import("../cache/budget.zig").publishHotCacheResidency;
const doLoadOnInferenceThread = @import("../load.zig").doLoadOnInferenceThread;
const loadErrorName = @import("../../../host/media_jobs/plugin/engine.zig").loadErrorName;
const VisionEncodeRequest = @import("../media_encode.zig").VisionEncodeRequest;
const EmbedRequest = @import("../media_encode.zig").EmbedRequest;
const runVisionEncode = @import("../media_encode.zig").runVisionEncode;
const runEmbedRequest = @import("../media_encode.zig").runEmbedRequest;
const Scheduler = @import("../scheduler.zig").Scheduler;
const Slot = @import("../slot.zig").Slot;
const chatPass = @import("../step.zig").chatPass;
const runLoadRequest = @import("load_requests.zig").runLoadRequest;
const runUnloadRequest = @import("load_requests.zig").runUnloadRequest;
const LoadRequest = @import("../../../host/gpu_thread/requests.zig").LoadRequest;
const GenRequest = @import("../../../host/gpu_thread/requests.zig").GenRequest;
const MAX_MERGED_WEIGHT = @import("../../../host/gpu_thread/requests.zig").MAX_MERGED_WEIGHT;
const takeMergeable = @import("../../../host/gpu_thread/requests.zig").takeMergeable;
const UnloadRequest = @import("../../../host/gpu_thread/requests.zig").UnloadRequest;
const ThreadCtx = @import("thread.zig").ThreadCtx;
const signalStarted = @import("thread.zig").signalStarted;
const recordLoadError = @import("thread.zig").recordLoadError;
const runGenRequests = @import("media_runner.zig").runGenRequests;

pub fn hasWorkPendingLocked(sch: *const Scheduler) bool {
    return sch.pending.items.len > 0 or
        sch.decoding.items.len > 0 or
        sch.vision_queue.items.len > 0 or
        sch.embed_queue.items.len > 0 or
        sch.cleanup_queue.items.len > 0 or
        sch.load_queue.items.len > 0 or
        sch.gen_queue.items.len > 0 or
        sch.unload_queue.items.len > 0;
}

pub const idleEvictTickMs = @import("../../../host/http/server.zig").idleEvictTickMs;

pub fn inferenceLoop(ctx: ThreadCtx) void {
    const sch = ctx.scheduler;
    const params = ctx.params;
    defer sch.inference_done.store(true, .release);
    // Covers shutdown and startup-load failure.
    defer sleep_inhibit.release();

    // ── Phase A1 → Plan 05: load runs on this thread (mlx GPU stream
    //    binding). On failure, mark the entry `.error_state` in the
    //    registry AND set load_failed so `Scheduler.init`'s parent sees
    //    the same shape it always has.
    //
    // Headless boot (`no_initial_load`): start with NO primary model. The
    // server runs idle until a chat or media model is loaded on demand via
    // `/v1/load-model` (or a request targeting a discovered id). Used by the
    // app's "start headless, load gen on demand" flow.
    if (params.no_initial_load) {
        log.info("Headless: no primary model loaded; models load on demand.\n", .{});
        signalStarted(sch);
    } else {
        // Startup load runs before the wait loop can acquire.
        sleep_inhibit.setActive(true);
        // `.loading` while it runs, so the startup responder's `/v1/load-progress` lists it.
        sch.registry.mutex.lockUncancelable(sch.io);
        _ = sch.registry.tryBeginLoadLocked(params.entry);
        sch.registry.mutex.unlock(sch.io);
        if (doLoadOnInferenceThread(sch, params)) |_| {
            log.info("Model ready (loaded on inference thread).\n", .{});
            signalStarted(sch);
        } else |err| {
            recordLoadError(sch, loadErrorName(err));
            sch.registry.mutex.lockUncancelable(sch.io);
            sch.registry.markErrorLocked(params.entry, loadErrorName(err));
            sch.registry.mutex.unlock(sch.io);
            signalStarted(sch);
            return;
        }
    }

    while (!sch.shutdown.load(.acquire)) {
        // Phase D: cold-load drain. Process ONE load per tick — loading a
        // model is heavy (~seconds; weight read + JIT compile + warmup)
        // and we want the rest of the inference loop to stay responsive.
        // Other pending loads wait in queue and get picked up next tick.
        var load_req: ?*LoadRequest = null;
        // Media-gen + unload work items (one per tick, like load — both are
        // heavy and we re-check the loop between them). A gen job owns the
        // loop until it returns; chat runs between its steps (`GenYield`).
        // Decision jobs queued back to back for one model run as one pass
        // (`takeMergeable`).
        var gen_batch: [MAX_MERGED_WEIGHT]*GenRequest = undefined;
        var gen_n: usize = 0;
        var unload_req: ?*UnloadRequest = null;
        {
            sch.queue_mu.lockUncancelable(sch.io);
            defer sch.queue_mu.unlock(sch.io);
            if (sch.load_queue.items.len > 0) {
                load_req = sch.load_queue.orderedRemove(0);
            }
            if (sch.unload_queue.items.len > 0) {
                unload_req = sch.unload_queue.orderedRemove(0);
            }
            if (sch.gen_queue.items.len > 0) {
                gen_batch[0] = sch.gen_queue.orderedRemove(0);
                gen_n = takeMergeable(&sch.gen_queue, &gen_batch, 1);
                const window_us = if (gen_batch[0].merge) |m| m.window_us else 0;
                if (window_us > 0 and gen_n < gen_batch.len) {
                    const deadline = (std.Io.Timeout{ .duration = .{
                        .raw = .fromMicroseconds(window_us),
                        .clock = .awake,
                    } }).toDeadline(sch.io);
                    while (!sch.shutdown.load(.acquire)) {
                        sch.queue_cond.waitTimeout(sch.io, &sch.queue_mu, deadline) catch break;
                        gen_n = takeMergeable(&sch.gen_queue, &gen_batch, gen_n);
                    }
                    gen_n = takeMergeable(&sch.gen_queue, &gen_batch, gen_n);
                }
            }
        }
        _ = drainSideQueues(sch);
        if (load_req) |req| runLoadRequest(sch, req);
        if (unload_req) |req| runUnloadRequest(sch, req);
        if (load_req != null or unload_req != null) reviseHotCacheBudgets(sch, load_req == null);
        if (unload_req != null) sch.budget_revise_sw = io_util.Stopwatch.init(sch.io);
        if (gen_n > 0) runGenRequests(sch, gen_batch[0..gen_n]);
        if (chatPass(sch, .main) == .shutdown) break;
    }
    persistAtShutdown(sch);
    closeGlmModels(sch);
}

/// At shutdown, on the inference thread: GLM-5.3's module synchronizes its stream as it frees, and MLX finds that
/// stream only on this thread, not on the one that joins it (`Scheduler.deinit`, where every other model unloads as
/// before; "There is no Stream(gpu, 0) in current thread", mlx-c stream.cpp).
fn closeGlmModels(sch: *Scheduler) void {
    sch.registry.mutex.lockUncancelable(sch.io);
    defer sch.registry.mutex.unlock(sch.io);
    var it = sch.registry.entries.valueIterator();
    while (it.next()) |entry| {
        const x = entry.*.transformer orelse continue;
        if (!x.config.glm_stream) continue;
        const m = x.mlx_stream_ext orelse continue;
        mlx_stream.close(m);
        x.mlx_stream_ext = null;
    }
}

/// Cleanup, vision and embed work, drained on the inference thread. Returns
/// whether there was any.
pub fn drainSideQueues(sch: *Scheduler) bool {
    // 0a. Drain slots queued for cleanup. Conn threads hand finished
    //     slots here in `complete()` — we own the mlx stream binding,
    //     so freeing per-slot KVCache + vision_embeddings + ssm_entries
    //     is safe here even though those slots' arrays might trigger
    //     real GPU memory release on refcount-zero.
    var cleanup_batch: [16]*Slot = undefined;
    var cleanup_n: usize = 0;
    // 0b. Drain any pending vision/embed work. These run synchronously on
    //     behalf of conn threads waiting in `encodeVision` /
    //     `computeEmbedding`. Processed here (not concurrently with decode
    //     ticks) so they share the inference thread's mlx stream cleanly.
    var vision_batch: [4]*VisionEncodeRequest = undefined;
    var vision_n: usize = 0;
    var embed_batch: [4]*EmbedRequest = undefined;
    var embed_n: usize = 0;
    {
        sch.queue_mu.lockUncancelable(sch.io);
        defer sch.queue_mu.unlock(sch.io);
        while (cleanup_n < cleanup_batch.len and sch.cleanup_queue.items.len > 0) {
            cleanup_batch[cleanup_n] = sch.cleanup_queue.orderedRemove(0);
            cleanup_n += 1;
        }
        while (vision_n < vision_batch.len and sch.vision_queue.items.len > 0) {
            vision_batch[vision_n] = sch.vision_queue.orderedRemove(0);
            vision_n += 1;
        }
        while (embed_n < embed_batch.len and sch.embed_queue.items.len > 0) {
            embed_batch[embed_n] = sch.embed_queue.orderedRemove(0);
            embed_n += 1;
        }
    }
    for (cleanup_batch[0..cleanup_n]) |s| {
        // Decode-phase cancel: `complete()` pulled this slot straight
        // into the cleanup queue, so it never went through finishSlot
        // and its committed KV (prompt + every emitted token) would die
        // right here with the slot. Commit it first — the same guards
        // as a normal finish apply inside (pad-only / error / vision /
        // empty all decline) — then flush what was committed to the
        // SSD tier, since no finishSlot will. Normally-finished slots
        // arrive here with `finished` already set (finishSlot committed
        // them) and skip; errored slots decline via the error guard.
        // Runs on the inference thread — the sole mlx caller — which is
        // what makes the refcount-sharing snapshot legal here.
        if (s.cancelled.load(.acquire) and !s.finished and s.error_code == null) {
            commitSlotIfApplicable(sch, s);
            if (s.model.prefix_cache) |hc| {
                if (s.model.transformer) |xf| hc.flushPendingDisk(xf.s);
            }
        }
        recordSlotCleanup(sch.metrics, s);
        // Second slot-end path (a decode-phase cancel never reaches finishSlot); the
        // record must not outlive the bytes `s.deinit()` frees.
        if (s.model.prefix_cache) |hc| hc.releaseCheckout(@intFromPtr(s), "slot cleanup");
        releasePluginSlot(s);
        if (s.model.transformer) |xfm| if (xfm.qwen4Trunk()) |t| t.resetQsaPooledRope();
        s.deinit();
    }
    if (cleanup_n > 0) publishHotCacheResidency(sch);
    for (vision_batch[0..vision_n]) |req| runVisionEncode(sch, req);
    for (embed_batch[0..embed_n]) |req| runEmbedRequest(sch, req);
    return cleanup_n + vision_n + embed_n > 0;
}

// ── tests ──
