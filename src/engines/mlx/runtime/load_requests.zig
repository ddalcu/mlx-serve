//! Load, eviction and unload requests serviced on the inference thread.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const model_registry_mod = @import("../../../host/registry/registry.zig");
const log = @import("../../../sdk/util/log.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const LoadedModel = model_registry_mod.LoadedModel;

const aneWanted = @import("../../../host/engines/served.zig").aneWanted;
const engines_load = @import("engine_load.zig");
const moveHotCacheViewOff = scheduler_mod.moveHotCacheViewOff;
const publishHotCacheResidency = scheduler_mod.publishHotCacheResidency;
const doLoadOnInferenceThread = @import("../load.zig").doLoadOnInferenceThread;
const loadErrorName = @import("../../../host/media_jobs/plugin/engine.zig").loadErrorName;
const Scheduler = @import("../scheduler.zig").Scheduler;
const LoadRequest = @import("../../../host/gpu_thread/requests.zig").LoadRequest;
const UnloadRequest = @import("../../../host/gpu_thread/requests.zig").UnloadRequest;
const loadRamRoom = @import("preflight.zig").loadRamRoom;
const loadRequirementBytes = @import("preflight.zig").loadRequirementBytes;
const scheduler_mod = @import("../scheduler.zig");

/// Plan 05 Phase D: service a cold-load work item on the inference thread.
/// Conn thread has already:
///   * Marked `req.entry.state = .loading` (transitioned from .unloaded).
///   * (Optionally) marked `req.evict_entry.state = .evicting` and drained
///     its refcount to 0.
///   * Parsed CPU-only state (config/tok/chat_config) and handed pointers
///     in via the request.
/// We unload the victim (if any), run the load body, install everything on
/// `req.entry`, mark ready, and broadcast `req.done_cond` so the conn
/// thread wakes. On failure, mark `.error_state` so future ensureLoaded
/// calls fail fast; the conn thread surfaces a 500.
pub fn logWiredPolicy(r: mlx.WiredPolicyResult) void {
    if (r.target) |t| {
        log.info("[wired] mode={s} limit={d} MB\n", .{ @tagName(r.mode), t / (1024 * 1024) });
    } else {
        log.debug("[wired] mode={s} declined (no gpu / empty live set)\n", .{@tagName(r.mode)});
    }
}

/// Free `victims` (each marked `.evicting`, nobody holding it) on this thread.
fn evictResident(sch: *Scheduler, victims: []const *LoadedModel) void {
    for (victims) |victim| {
        const victim_bytes = victim.bytes_resident; // unloadResident zeroes it
        log.info("[registry] evicting model id={s} ({d:.2} GB resident)\n", .{
            victim.id,
            @as(f64, @floatFromInt(victim_bytes)) / 1_073_741_824.0,
        });
        moveHotCacheViewOff(sch, victim);
        victim.unloadResident();
        sch.registry.mutex.lockUncancelable(sch.io);
        sch.registry.accountEvictedLocked(victim_bytes);
        sch.registry.finalizeEvictionLocked(victim);
        sch.registry.mutex.unlock(sch.io);
    }
    // unloadResident freed the victims into MLX's allocator cache — clear it
    // BEFORE the load: its preflight reads OS-level availability, and the
    // parked pool would make it refuse a load that fits.
    _ = mlx.mlx_clear_cache();
}

/// Unloads idle models, oldest first, when that lets a load of `weights` fit in RAM: as few as fit
/// it, none when it already fits or when even all of them cannot (`planRamEvictionsLocked`). Runs
/// past `--skip-mem-preflight` too: an idle model reloads, an overcommitted Mac kills the server.
pub fn makeRoomInRam(sch: *Scheduler, keep_id: []const u8, weights: u64) void {
    const room = loadRamRoom() orelse return;
    const need = loadRequirementBytes(weights);
    var victims: [16]*LoadedModel = undefined;
    sch.registry.mutex.lockUncancelable(sch.io);
    const n = sch.registry.planRamEvictionsLocked(keep_id, &victims, room, need);
    sch.registry.mutex.unlock(sch.io);
    if (n == 0) return;
    log.info("[registry] {s} needs ~{d:.1} GB beside {d:.1} GB of free RAM: unloading {d} idle model(s)\n", .{
        keep_id,
        @as(f64, @floatFromInt(need)) / 1_073_741_824.0,
        @as(f64, @floatFromInt(room)) / 1_073_741_824.0,
        n,
    });
    evictResident(sch, victims[0..n]);
}

pub fn runLoadRequest(sch: *Scheduler, req: *LoadRequest) void {
    // Step 1: evict victims (if any) BEFORE the load, so peak GPU residency
    // never holds the old + new model at once. unloadResident() drops
    // mlx_arrays — same thread-stream invariant as cleanup_queue drain.
    evictResident(sch, req.evict_entries);
    req.evict_entries = &.{};

    // A Neural Engine build runs on its own thread while the GPU keeps serving.
    if (req.gguf == null and !req.ane_declined and aneWanted(req.config, req.ane)) {
        req.entry.load_progress.start(io_util.nowMsMonotonic(sch.io));
        if (engines_load.aneAdmit(sch, req)) {
            _ = sch.ane_builds.fetchAdd(1, .acq_rel);
            const t = std.Thread.spawn(.{}, aneBuildThread, .{ sch, req }) catch return aneBuildThread(sch, req);
            t.detach();
            return;
        }
        req.ane_declined = true;
    }

    // Step 2: the actual load. On error, mark .error_state and signal done
    // (conn thread frees pre-parsed CPU state — ownership stays on req on
    // the failure path).
    const had_error = mlx.errorPending();
    doLoadOnInferenceThread(sch, req) catch |err| {
        // The load's own MLX error is reported here; latched, it would fail the next request.
        mlx.dropLatchedErrorUnless(had_error);
        return failLoadRequest(sch, req, err);
    };
    // Re-apply so `fit` capacity covers everything this load brought in
    // (MTP head / drafter / vision land after the mid-load apply).
    logWiredPolicy(mlx.applyWiredPolicy());
    readyLoadRequest(sch, req);
}

/// A cold Neural Engine load's build, minutes on a first load: done, or back on the load queue
/// for the GPU when the build declines. Never calls MLX.
fn aneBuildThread(sch: *Scheduler, req: *LoadRequest) void {
    defer _ = sch.ane_builds.fetchSub(1, .acq_rel);
    const built = engines_load.aneOpen(sch, req) catch |err| return failLoadRequest(sch, req, err);
    if (built) return readyLoadRequest(sch, req);
    req.ane_declined = true;
    sch.queue_mu.lockUncancelable(sch.io);
    const queued = if (sch.shutdown.load(.acquire)) error.Shutdown else sch.load_queue.insert(sch.allocator, 0, req);
    sch.queue_cond.broadcast(sch.io);
    sch.queue_mu.unlock(sch.io);
    queued catch |err| failLoadRequest(sch, req, err);
}

fn readyLoadRequest(sch: *Scheduler, req: *LoadRequest) void {
    log.info("[registry] model id={s} ready ({d:.2} GB resident)\n", .{
        req.entry.id,
        @as(f64, @floatFromInt(req.entry.bytes_resident)) / 1_073_741_824.0,
    });
    finishLoadRequest(sch, req, null);
}

/// Mark the entry and signal the waiter (conn thread frees pre-parsed CPU state — ownership stays
/// on req on the failure path).
fn failLoadRequest(sch: *Scheduler, req: *LoadRequest, err: anyerror) void {
    log.err("[registry] load failed for model id={s}: {s}\n", .{ req.entry.id, loadErrorName(err) });
    sch.registry.mutex.lockUncancelable(sch.io);
    if (err == error.InsufficientMemory) {
        // A memory-preflight refusal is transient, not a property of the
        // checkpoint: the 503 tells the user to free memory and retry, so
        // the entry must go back to .unloaded — stuck in .error_state the
        // retry would fail fast until a server restart (#144).
        sch.registry.markUnloadedLocked(req.entry);
    } else {
        // markErrorLocked dupes the error name onto the entry; the
        // conn thread reads it back from the entry, not from req.
        sch.registry.markErrorLocked(req.entry, loadErrorName(err));
    }
    sch.registry.mutex.unlock(sch.io);
    finishLoadRequest(sch, req, loadErrorName(err));
}

fn finishLoadRequest(sch: *Scheduler, req: *LoadRequest, err_name: ?[]const u8) void {
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    if (err_name) |name| {
        if (req.error_name) |old| req.allocator.free(old);
        req.error_name = req.allocator.dupe(u8, name) catch null;
    }
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

/// Free a model's resident mlx state on the inference thread (stream-bound,
/// same invariant as the cleanup-queue drain) and finalize the eviction
/// accounting. The conn thread already marked the entry `.evicting` and
/// drained its refcount in `unloadModel`.
pub fn runUnloadRequest(sch: *Scheduler, req: *UnloadRequest) void {
    const entry = req.entry;
    const bytes = entry.bytes_resident; // unloadResident zeroes it
    log.info("[registry] unloading model id={s} ({d:.2} GB resident)\n", .{
        entry.id,
        @as(f64, @floatFromInt(bytes)) / 1_073_741_824.0,
    });
    moveHotCacheViewOff(sch, entry);
    entry.unloadResident();
    // unloadResident freed the arrays into MLX's allocator cache — clear it
    // so the unload actually returns the memory to the OS (the whole point
    // of the load→generate→unload flow).
    _ = mlx.mlx_clear_cache();
    // Drop any borrowed views that pointed at this entry so post-unload reads
    // don't dangle (gen entries leave xfm null already, but an LLM unload
    // must clear them).
    if (sch.current_model == entry) {
        sch.current_model = null;
        sch.xfm = null;
        sch.weights = null;
        sch.vision_encoder = null;
        sch.drafter = null;
        sch.dflash = null;
        if (scheduler_mod.hot_cache_budget_invalidate) |f| f();
    }
    publishHotCacheResidency(sch);
    sch.registry.mutex.lockUncancelable(sch.io);
    sch.registry.accountEvictedLocked(bytes);
    sch.registry.finalizeEvictionLocked(entry);
    sch.registry.mutex.unlock(sch.io);

    // Shrink `fit` capacity back to the surviving live set — leaving the
    // freed model's headroom in place is exactly the per-transient-commit
    // configuration the policy exists to avoid.
    logWiredPolicy(mlx.applyWiredPolicy());

    req.done_mu.lockUncancelable(sch.io);
    req.done = true;
    req.done_cond.broadcast(sch.io);
    req.done_mu.unlock(sch.io);
}
