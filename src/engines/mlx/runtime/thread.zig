//! Inference-thread start-up context and load-error plumbing.

const std = @import("std");
const gguf_meta = @import("../../../sdk/quant/gguf/engine_pick.zig");

const Scheduler = @import("../scheduler.zig").Scheduler;
const LoadParams = @import("../scheduler/params.zig").LoadParams;

pub const ThreadCtx = struct {
    scheduler: *Scheduler,
    params: LoadParams,
};

/// Signal to the parent waiting in `init()` that the inference thread is done
/// with its load (or has failed). After `started` flips, the parent reads
/// `load_failed` to decide whether to surface a startup error.
pub fn signalStarted(sch: *Scheduler) void {
    sch.started_mu.lockUncancelable(sch.io);
    defer sch.started_mu.unlock(sch.io);
    sch.started.store(true, .release);
    sch.started_cond.broadcast(sch.io);
}
/// Set the load_error_name + flip load_failed. Best-effort dupe; on OOM the
/// parent still sees `load_failed=true` and surfaces "unknown".
pub fn recordLoadError(sch: *Scheduler, err_name: []const u8) void {
    if (sch.load_error_name) |old| sch.allocator.free(old);
    sch.load_error_name = sch.allocator.dupe(u8, err_name) catch null;
    sch.load_failed.store(true, .release);
}

pub const GgufRoute = gguf_meta.Route;
