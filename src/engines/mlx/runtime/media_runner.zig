//! Media-gen jobs on the inference thread, and chat yielding between their steps.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const gen_sse = @import("../../../host/media_jobs/sse.zig");
const media_progress = @import("../../../sdk/media/progress.zig");
const model_registry_mod = @import("../../../host/registry/registry.zig");
const log = @import("../../../sdk/util/log.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const status = @import("../../../host/status.zig");
const testing = std.testing;

const decodeShareBudgetNs = @import("../prefill_interleave.zig").decodeShareBudgetNs;
const invalidateDecodeClocks = @import("../prefill_interleave.zig").invalidateDecodeClocks;
const Scheduler = @import("../scheduler.zig").Scheduler;
const ChatPassResult = @import("../step.zig").ChatPassResult;
const chatPass = @import("../step.zig").chatPass;
const GenRequest = @import("../../../host/gpu_thread/requests.zig").GenRequest;
const MAX_MERGED_WEIGHT = @import("../../../host/gpu_thread/requests.zig").MAX_MERGED_WEIGHT;
const scheduler_mod = @import("../scheduler.zig");

/// Chat's share of wall time while a media job runs: equal, so neither side starves.
const GEN_CHAT_SHARE: f32 = 0.5;

/// Chat's turn while a media job owns the inference thread (`gen_sse.Yield`): at each
/// step boundary chat runs for as long as the step took, and while a pure-CPU stage
/// runs on a worker (`gen_sse.offload`) chat runs until it is done.
const GenYield = struct {
    sch: *Scheduler,
    step_sw: io_util.Stopwatch,
    engaged: bool = false,

    fn pass(gy: *GenYield) ChatPassResult {
        const r = chatPass(gy.sch, .yield);
        if (r == .ran and !gy.engaged) {
            gy.engaged = true;
            log.debug("[gen-yield] engaged: chat runs between media steps\n", .{});
        }
        return r;
    }

    fn step(ctx: *anyopaque) void {
        const gy: *GenYield = @ptrCast(@alignCast(ctx));
        {
            gy.sch.queue_mu.lockUncancelable(gy.sch.io);
            defer gy.sch.queue_mu.unlock(gy.sch.io);
            invalidateDecodeClocks(gy.sch.decoding.items);
        }
        const budget = decodeShareBudgetNs(gy.step_sw.read(), GEN_CHAT_SHARE);
        const sw = io_util.Stopwatch.init(gy.sch.io);
        while (sw.read() < budget and gy.pass() == .ran) {}
        gy.step_sw.reset();
    }

    fn serveUntil(ctx: *anyopaque, done: *const std.atomic.Value(bool)) void {
        const gy: *GenYield = @ptrCast(@alignCast(ctx));
        const sch = gy.sch;
        while (!done.load(.acquire)) {
            if (gy.pass() == .ran) continue;
            // Idle: park until a submission or the worker wakes us. The timeout covers a
            // submission that landed between the pass and this lock.
            sch.queue_mu.lockUncancelable(sch.io);
            defer sch.queue_mu.unlock(sch.io);
            if (done.load(.acquire)) break;
            const deadline = (std.Io.Timeout{ .duration = .{ .raw = .fromMilliseconds(50), .clock = .awake } }).toDeadline(sch.io);
            sch.queue_cond.waitTimeout(sch.io, &sch.queue_mu, deadline) catch {};
        }
        gy.step_sw.reset();
    }

    fn wake(ctx: *anyopaque) void {
        const sch = @as(*GenYield, @ptrCast(@alignCast(ctx))).sch;
        sch.queue_mu.lockUncancelable(sch.io);
        defer sch.queue_mu.unlock(sch.io);
        sch.queue_cond.broadcast(sch.io);
    }
};

/// Chat interleaves with media jobs except on CUDA, where a decode inside a job faulted the context.
pub fn genYieldEnabled(cuda: bool) bool {
    return !cuda;
}

/// Run one media-generation job on the inference thread. The job body
/// (`req.run`) does all mlx work + writes the HTTP/SSE response to the parked
/// connection. We bracket it with the model's `gen_busy` flag for visibility
/// and signal `done` so the conn thread in `runGeneration` wakes. Several
/// merged jobs (`takeMergeable`) run as one `run_many` call.
pub fn runGenRequests(sch: *Scheduler, reqs: []*GenRequest) void {
    const req = reqs[0];
    req.model.gen_busy = true;
    var gy = GenYield{ .sch = sch, .step_sw = io_util.Stopwatch.init(sch.io) };
    const hook = gen_sse.Yield{ .ctx = &gy, .step = GenYield.step, .serve_until = GenYield.serveUntil, .wake = GenYield.wake };
    if (scheduler_mod.gen_yield_enabled) media_progress.genYield().* = &hook;
    // The whole estimated peak, though live memory already holds part of it: the side that
    // cannot OOM.
    sch.gen_reserve_bytes = if (req.peak_bytes > 0) req.peak_bytes else scheduler_mod.mediaPeakFor(sch, req.model);
    defer {
        media_progress.genYield().* = null;
        media_progress.streamKeepalive().* = null;
        sch.gen_reserve_bytes = 0;
        sch.queue_mu.lockUncancelable(sch.io);
        invalidateDecodeClocks(sch.decoding.items);
        sch.queue_mu.unlock(sch.io);
    }
    // On small-RAM machines (≤16 GB — mini class; also the phone), bound
    // MLX's buffer-cache growth DURING the generation: the post-request
    // clear below can't help mid-loop, and a diffusion denoise + VAE decode
    // otherwise accumulates GBs of one-off transients against a ~12 GB Metal
    // working-set ceiling. 1 GB still covers step-to-step buffer reuse.
    // Never RAISE a tighter existing cap (the iOS boot cap is 384 MB).
    const small_ram = blk: {
        const total = status.getTotalMemBytes();
        break :blk total > 0 and total <= 17 * 1024 * 1024 * 1024;
    };
    var prev_cache_limit: usize = 0;
    if (small_ram) {
        const cap: usize = 1024 * 1024 * 1024;
        _ = mlx.mlx_set_cache_limit(&prev_cache_limit, cap);
        if (prev_cache_limit < cap) {
            var tmp: usize = 0;
            _ = mlx.mlx_set_cache_limit(&tmp, prev_cache_limit);
        }
    }
    var prev_mem_limit: usize = 0;
    const gen_mem_limit = genMemoryLimit(mlx.cudaAvailable(), sch.gen_reserve_bytes);
    if (gen_mem_limit) |lim| _ = mlx.mlx_set_memory_limit(&prev_mem_limit, lim);
    runGenJobs(reqs);
    if (gen_mem_limit != null) {
        var tmp: usize = 0;
        _ = mlx.mlx_set_memory_limit(&tmp, prev_mem_limit);
    }
    if (small_ram) {
        var tmp: usize = 0;
        _ = mlx.mlx_set_cache_limit(&tmp, prev_cache_limit);
    }
    req.model.gen_busy = false;
    // Return the generation's transients to the OS. MLX parks freed buffers
    // in its allocator cache (RSS stays), and unlike chat decode (which
    // clears every 256 steps in generate.zig) a media gen frees tens of GB
    // of denoise/VAE/encoder buffers in one burst — without this, each
    // generation ratchets process RSS upward (observed ~100 GB by gen 2).
    var cached: usize = 0;
    if (req.decision) _ = mlx.mlx_get_cache_memory(&cached);
    if (shouldClearCache(req.decision, cached)) _ = mlx.mlx_clear_cache();
    for (reqs) |r| {
        r.done_mu.lockUncancelable(sch.io);
        r.done = true;
        r.done_cond.broadcast(sch.io);
        r.done_mu.unlock(sch.io);
    }
}

/// A latch the job raised is its own failure, already answered by its handler; left set, it
/// would fail the next chat request.
fn runGenJobs(reqs: []*GenRequest) void {
    const had_error = mlx.errorPending();
    defer mlx.dropLatchedErrorUnless(had_error);
    defer @import("../../../sdk/blocks/dit/w8a8.zig").dropMemo(); // the last activation (and its int8 copy) would stay pinned until the next job
    const req = reqs[0];
    if (reqs.len == 1) req.run(req.ctx) else {
        var ctxs: [MAX_MERGED_WEIGHT]*anyopaque = undefined;
        for (reqs, ctxs[0..reqs.len]) |r, *c| c.* = r.ctx;
        req.merge.?.run_many(ctxs[0..reqs.len]);
    }
}

/// MLX memory limit for a media job, null = keep MLX's own. MLX only waits on
/// in-flight work (whose graphs pin their inputs) once active memory passes this
/// limit; CUDA's default is 95% of VRAM, so a diffusion step fills the card.
fn genMemoryLimit(cuda: bool, est_peak: u64) ?usize {
    return if (cuda and est_peak > 0) @intCast(est_peak) else null;
}

/// Clear MLX's allocator cache after a job holding `cached` bytes: always for
/// media, from 256 MiB for decision jobs.
fn shouldClearCache(decision: bool, cached: usize) bool {
    return !decision or cached >= 256 << 20;
}

// ── tests ──

test "a media job on CUDA throttles MLX at its estimated peak; Metal keeps MLX's limit" {
    try testing.expectEqual(@as(?usize, 4 << 30), genMemoryLimit(true, 4 << 30));
    // No estimate (the forced aligner's pass) keeps MLX's limit: 0 would throttle every allocation.
    try testing.expectEqual(@as(?usize, null), genMemoryLimit(true, 0));
    try testing.expectEqual(@as(?usize, null), genMemoryLimit(false, 4 << 30));
}

test "a media job always clears the allocator cache, a decision job only from 256 MiB" {
    try testing.expect(shouldClearCache(false, 0));
    try testing.expect(!shouldClearCache(true, (256 << 20) - 1));
    try testing.expect(shouldClearCache(true, 256 << 20));
}

test "runGenJobs: a latch the job raised is dropped, one from before the job stays" {
    var stale: [512]u8 = undefined;
    _ = mlx.takeError(&stale); // an earlier test's latch would shadow this one's
    const job = struct {
        fn oom(_: *anyopaque) void {
            mlx.latchErrorForTest("[METAL] Command buffer execution failed: Insufficient Memory. at transforms.cpp:15");
        }
        fn ok(_: *anyopaque) void {}
    };
    var dummy: u8 = 0;
    const m: *model_registry_mod.LoadedModel = @ptrFromInt(0x1000);
    var failing: GenRequest = .{ .ctx = &dummy, .run = job.oom, .model = m };
    var reqs = [_]*GenRequest{&failing};
    runGenJobs(&reqs);
    try testing.expect(!mlx.errorPending());

    var clean: GenRequest = .{ .ctx = &dummy, .run = job.ok, .model = m };
    reqs[0] = &clean;
    mlx.latchErrorForTest("earlier failure");
    runGenJobs(&reqs);
    var buf: [64]u8 = undefined;
    try testing.expectEqualStrings("earlier failure", mlx.takeError(&buf).?);
}

test "genYieldEnabled: off on CUDA, on elsewhere" {
    try std.testing.expect(!genYieldEnabled(true));
    try std.testing.expect(genYieldEnabled(false));
}
