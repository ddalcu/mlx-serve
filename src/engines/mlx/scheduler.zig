//! Plan 01 Phase 2 — continuous-batching scheduler.
//!
//! Owns the single inference thread (the only thread that calls into mlx
//! ops; mlx 0.31.2 made GPU streams thread-local so the model weights are
//! bound to whichever thread first calls `useCurrentThreadStream` after
//! load). Connection threads parse HTTP, build prompt token ids, call
//! `submit()` and then loop on `Slot.waitNext()` to read generated tokens
//! one at a time. The state machines for tool-call detection, thinking
//! blocks, SSE streaming, etc. live on the connection thread, unchanged.
//!
//! Per-slot state (KVCache, moe_seq_offset, ssm_entries, vision_embeddings)
//! lives on `Slot` itself; each slot's `ForwardCtx` points at those fields,
//! and the `Generator` constructed for the slot stores the ctx so
//! `xfm.forwardWith(&self.ctx, ...)` routes through slot-local state. The
//! shared Transformer holds the weights only — single-writer-per-tick on
//! the inference thread guards correctness while many slots can be in
//! flight at once.
//!
//! Decode tick logic (the Phase 3 gate):
//!   * `active.len == 1` (the common single-stream case) → legacy path:
//!     `Generator.next` / `nextPld` / `nextDrafter` with the same lazy
//!     pipeline that today's serial path uses. Bit-identical to pre-Phase-2.
//!   * `active.len >= 2` → batched: `forwardBatchedDecode` produces N logits
//!     in one kernel pass, sampled per-slot. PLD / drafter are forced off in
//!     this path (the speculative paths assume a single in-flight slot's
//!     KV cache; expensive to interleave).
//!   * `cancelled` slots are skipped and culled from `decoding`.
//!
//! Output channel: each slot owns a bounded ring (`std.ArrayList(u32)` +
//! cursor + cv) the inference thread pushes into and the connection thread
//! drains. Generation end is signaled by `state == .finished` (or `.errored`)
//! plus a cv broadcast.

const std = @import("std");
const env = @import("../../env.zig");
const depth_bounds = @import("spec/mtp_depth_bounds.zig");
const mlx = @import("../../mlx/ffi.zig");
const ple_gpu = @import("../../plugins/qwen4_exp/ple_gpu.zig");
const transformer_mod = @import("transformer.zig");
const tokenizer_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const generate_mod = @import("generate.zig");
const rp_mod = @import("../../sdk/grammar/protocol.zig");
const gen_mod = @import("../../host/media_jobs/gen.zig");
const gen_sse = @import("../../host/media_jobs/sse.zig");
const drafter_mod = @import("../../plugins/gemma4/assistant_drafter.zig");
const mtp_graft = @import("../../plugins/prism_hadamard_qwen35/mtp_graft.zig");
const mtp_mod = @import("../../sdk/spec/mtp/mtp.zig");
const ane_mod = @import("../../sdk/ane/offload.zig");
const model_mod = @import("../../sdk/config/model.zig");
const sushi_pack = @import("../../sdk/quant/exl3/sushi_pack.zig");
const modelopt_pack = @import("../../sdk/quant/nvfp4/modelopt_pack.zig");
const vision_mod = @import("vision_encoder.zig");
const radio_vision = @import("../../plugins/nemotron_h/radio_vision.zig");
const chat_mod = @import("../../host/chat/chat.zig");
const prefix_cache_mod = @import("cache/prefix_cache.zig");
const restore_dump = @import("cache/restore_dump.zig");
const metrics_mod = @import("../../host/metrics/metrics.zig");
const kv_disk_cache = @import("cache/ssd/disk_tier.zig");
const tokenize_cache_mod = @import("../../host/tokenizer/cache.zig");
const model_registry_mod = @import("../../host/registry/registry.zig");
const model_settings = @import("../../host/registry/model_settings.zig");
const model_discovery = @import("../../host/registry/discovery.zig");
const gguf_meta = @import("../../sdk/quant/gguf/engine_pick.zig");
const engines_served = @import("../../host/engines/served.zig");
const engine_requests = @import("runtime/offload.zig");
const refusal = @import("../../host/engines/refusal.zig");
const engines_load = @import("runtime/engine_load.zig");
const mlx_gguf = @import("../../sdk/quant/gguf/claim.zig");
const log = @import("../../sdk/util/log.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const status = @import("../../host/status.zig");
const dsv41_mod = @import("../../plugins/deepseek_v41/deepseek_v41.zig");
const mlx_stream = @import("modules/stream.zig");
const sleep_inhibit = @import("runtime/sleep_inhibit.zig");

const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;
const SSMCacheEntry = transformer_mod.SSMCacheEntry;
const ForwardCtx = transformer_mod.ForwardCtx;
const ModelConfig = model_mod.ModelConfig;
const Tokenizer = tokenizer_mod.Tokenizer;
const Generator = generate_mod.Generator;
const SamplingParams = generate_mod.SamplingParams;
const DrafterModel = drafter_mod.DrafterModel;
const dflash_mod = @import("spec/dflash/dflash.zig");
const round_cost_mod = @import("spec/round_cost.zig");
const group_cost_mod = @import("spec/mtp_group_cost.zig");
const DflashModel = dflash_mod.DflashModel;
const VisionEncoder = vision_mod.VisionEncoder;
const Weights = model_mod.Weights;
const ChatConfig = chat_mod.ChatConfig;
const ModelRegistry = model_registry_mod.ModelRegistry;
const LoadedModel = model_registry_mod.LoadedModel;

pub const MAX_BATCH_GROUP = @import("batching/verdict.zig").MAX_BATCH_GROUP;
pub const BatchVerdict = @import("batching/verdict.zig").BatchVerdict;
pub const configBatchesDecode = @import("batching/verdict.zig").configBatchesDecode;
pub const modelBatchable = @import("batching/verdict.zig").modelBatchable;
pub const liveDecodingCount = @import("batching/verdict.zig").liveDecodingCount;
pub const BudgetRevise = @import("cache/budget.zig").BudgetRevise;
pub const reclaimableHotCacheBytesFor = @import("cache/budget.zig").reclaimableHotCacheBytesFor;
pub const armShutdownDeadline = @import("cache/budget.zig").armShutdownDeadline;
pub const reviseHotCacheBudgets = @import("cache/budget.zig").reviseHotCacheBudgets;
pub const persistAtShutdown = @import("cache/budget.zig").persistAtShutdown;
pub const moveHotCacheViewOff = @import("cache/budget.zig").moveHotCacheViewOff;
pub const publishHotCacheResidency = @import("cache/budget.zig").publishHotCacheResidency;
pub const installAdmissionHooks = @import("admission/admit.zig").installSchedulerHooks;
pub const VisionItem = @import("media_encode.zig").VisionItem;
pub const VisionEncodeRequest = @import("media_encode.zig").VisionEncodeRequest;
pub const EmbedRequest = @import("media_encode.zig").EmbedRequest;
pub const prefillDecodeShare = @import("prefill_interleave.zig").prefillDecodeShare;
pub const resolveDecodeShare = @import("prefill_interleave.zig").resolveDecodeShare;
pub const LlamaSettings = @import("scheduler/params.zig").LlamaSettings;
pub const LoadParams = @import("scheduler/params.zig").LoadParams;
pub const SubmitParams = @import("scheduler/params.zig").SubmitParams;
pub const PromiseLedger = @import("scheduler/params.zig").PromiseLedger;
pub const postEvictionPrefillChunk = @import("scheduler/params.zig").postEvictionPrefillChunk;
pub const adaptiveChunkWidthFor = @import("scheduler/params.zig").adaptiveChunkWidthFor;
pub const Slot = @import("slot.zig").Slot;
const Planner = @import("spec/mtp_groups.zig").Planner;
pub const SpecTickMode = @import("spec/wiring.zig").SpecTickMode;
pub const specTickMode = @import("spec/wiring.zig").specTickMode;
const slotSpecMode = @import("spec/wiring.zig").slotSpecMode;
pub const idleEvictTickMs = @import("runtime/loop.zig").idleEvictTickMs;
const inferenceLoop = @import("runtime/loop.zig").inferenceLoop;
pub const LoadRequest = @import("../../host/gpu_thread/requests.zig").LoadRequest;
pub const GenRequest = @import("../../host/gpu_thread/requests.zig").GenRequest;
pub const MAX_MERGED_WEIGHT = @import("../../host/gpu_thread/requests.zig").MAX_MERGED_WEIGHT;
pub const UnloadRequest = @import("../../host/gpu_thread/requests.zig").UnloadRequest;
const ThreadCtx = @import("runtime/thread.zig").ThreadCtx;
pub const genYieldEnabled = @import("runtime/media_runner.zig").genYieldEnabled;
pub const coldLoadDrafterDir = @import("runtime/cold_load.zig").coldLoadDrafterDir;
pub const coldLoadVision = @import("runtime/cold_load.zig").coldLoadVision;
const preloadCpuState = @import("../../host/registry/cpu_state.zig").preloadCpuState;
const freeCpuState = @import("../../host/registry/cpu_state.zig").freeCpuState;
pub const applyModelSettings = @import("../../host/registry/model_settings_apply.zig").applyModelSettings;
const loadExceedsMemCap = @import("runtime/preflight.zig").loadExceedsMemCap;
pub const gateEstimateBytes = @import("runtime/preflight.zig").gateEstimateBytes;
const mlxActiveBytes = @import("runtime/preflight.zig").mlxActiveBytes;
const pleTableBill = @import("runtime/preflight.zig").pleTableBill;
const residentModelDiskBytes = @import("runtime/preflight.zig").residentModelDiskBytes;
const residentGateBytes = @import("runtime/preflight.zig").residentGateBytes;
pub const loadRequirementBytes = @import("runtime/preflight.zig").loadRequirementBytes;

test {
    _ = @import("admission/pending.zig");
    _ = @import("batching/tick.zig");
    _ = @import("batching/verdict.zig");
    _ = @import("cache/budget.zig");
    _ = @import("commit.zig");
    _ = @import("decode.zig");
    _ = @import("finish.zig");
    _ = @import("load.zig");
    _ = @import("logit_features.zig");
    _ = @import("media_encode.zig");
    _ = @import("prefill.zig");
    _ = @import("prefill_interleave.zig");
    _ = @import("scheduler/params.zig");
    _ = @import("slot.zig");
    _ = @import("spec/dflash_gate.zig");
    _ = @import("spec/mtp_groups.zig");
    _ = @import("spec/wiring.zig");
    _ = @import("step.zig");
    _ = @import("loop_stop.zig");
    _ = @import("runtime/load_requests.zig");
    _ = @import("runtime/loop.zig");
    _ = @import("../../host/gpu_thread/requests.zig");
    _ = @import("runtime/thread.zig");
    _ = @import("runtime/media_load.zig");
    _ = @import("runtime/media_runner.zig");
    _ = @import("runtime/cold_load.zig");
    _ = @import("../../host/registry/cpu_state.zig");
    _ = @import("../../host/registry/model_settings_apply.zig");
    _ = @import("runtime/preflight.zig");
}

/// Set by `server.installPrefillAdmission`: does a request of this shape fit in GPU memory
/// right now? Null (unit tests, no HTTP server) disables evict-to-admit. The trailing warm
/// arguments are the restored rows, their capacity, and whether the restore checked the
/// entry out (the only restore whose rows the request will not allocate).
/// {needed, available} of the same cold bill, for the sibling ledger.
pub var prefill_admission_numbers: ?*const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, bool) [2]u64 = null;

pub var prefill_admission_fits: ?*const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, u64, u64, bool, bool) bool = null;

/// The prefill width this request should run at, chosen against live post-eviction memory
/// (`server.requestPrefillChunkNow`). Null keeps the model's load-time pin.
pub var prefill_request_chunk: ?*const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, u64, u64, bool, bool) u32 = null;

/// Asked by the prefill loop: the width of the next chunk, re-priced at every chunk boundary
/// (`server.adaptivePrefillWidthNow`). Null keeps the admitted width.
pub var prefill_chunk_adapt: ?*const fn (
    *const model_mod.ModelConfig,
    u64,
    usize,
    u32,
    u32,
    *generate_mod.AdaptiveWidthState,
    u64,
) u32 = null;

/// Whether the per-chunk adaptive width is enabled for this model. The hook above is
/// installed process-wide, so its presence is not the arch gate.
pub var prefill_chunk_adaptive_enabled: ?*const fn (*const model_mod.ModelConfig) bool = null;

/// Re-price a widen after the interleave tick (`server.adaptivePrefillWidenStillFits`). Null declines every widen.
pub var prefill_chunk_widen_ok: ?*const fn (
    *const model_mod.ModelConfig,
    u64,
    usize,
    u32,
    u64,
) bool = null;

/// Logs the numbers the estimator compared on a refusal and returns them, {needed, available} bytes.
pub var prefill_admission_refused_log: ?*const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, u64, u64, bool, bool) [2]u64 = null;

/// Invalidate the published hot-cache budget on unload/switch (`server.clearResolvedPrefixCacheMem`).
pub var hot_cache_budget_invalidate: ?*const fn () void = null;

pub const Scheduler = @import("../../host/gpu_thread/scheduler.zig").Scheduler;

/// Construct a Scheduler whose inference thread loads the model. Returns
/// only after load + (optional) warmup completes. On load failure, returns
/// `error.LoadFailed`; the inference thread has already cleaned up any
/// partially-allocated mlx state by then.
pub fn init(
    allocator: std.mem.Allocator,
    io: std.Io,
    params: LoadParams,
    max_concurrent: u32,
) !*Scheduler {
    const self = try allocator.create(Scheduler);
    errdefer allocator.destroy(self);

    const cap = if (max_concurrent == 0) 1 else max_concurrent;
    // Phase A7: force-batched test hook. The byte-equivalence test sets
    // `MLX_SERVE_FORCE_BATCHED=1` to verify that the batched-kernel
    // output matches the single-slot path token-for-token at temp=0,
    // single client. Uses libc getenv to stay allocator-free.
    const force_batched = blk: {
        const raw = env.get(.MLX_SERVE_FORCE_BATCHED);
        if (raw == null) break :blk false;
        const slice = std.mem.sliceTo(raw.?, 0);
        break :blk std.mem.eql(u8, slice, "1");
    };
    if (force_batched) {
        log.info("[scheduler] force_batched=on (MLX_SERVE_FORCE_BATCHED=1) — single-slot ticks will route through batched kernel\n", .{});
    }
    self.* = .{
        .allocator = allocator,
        .io = io,
        .registry = params.registry,
        .current_model = null,
        .xfm = null,
        .weights = null,
        .vision_encoder = null,
        .drafter = null,
        .drafter_block_size = params.draft_block_size,
        .kv_quant_config = params.kv_quant_config,
        .gguf_ctx_size = params.ctx_size,
        .prefix_cache_capacity = params.prefix_cache_capacity,
        .prefix_cache_ram_enabled = params.prefix_cache_ram_enabled,
        .prefix_cache_mem_bytes = params.prefix_cache_mem_bytes,
        .prefix_cache_mem_resolver = if (params.prefix_cache_clamp) &@import("admission/hot_cache_plan.zig").prefixCacheMemForLoad else null,
        .prefix_cache_disk_bytes = params.prefix_cache_disk_bytes,
        .ssm_checkpoint_stride = params.ssm_checkpoint_stride,
        .ssm_checkpoint_max = params.ssm_checkpoint_max,
        .mtp_enabled = params.mtp_enabled,
        .mtp_head_kv_quant = params.mtp_head_kv_quant,
        .mtp_depth = params.mtp_depth,
        .llama = params.llama,
        .ds4_mtp = params.ds4_mtp,
        .ds4_dspark = params.ds4_dspark,
        .ds4_ssd_streaming = params.ds4_ssd_streaming,
        .ane_prefill = params.ane_prefill,
        .ane = params.ane,
        .ane_slots = params.ane_slots,
        .ane_chunk_resolver = if (params.ane_resolvers) &@import("admission/prefill_chunk.zig").pinPrefillChunk else null,
        .ane_headroom_resolver = if (params.ane_resolvers) &@import("admission/bills.zig").aneGateHeadroom else null,
        .no_drafter = params.no_drafter,
        .drafter_dir = params.drafter_dir,
        .primary_model_dir = params.model_dir,
        .draft_block_size = params.draft_block_size,
        .draft_block_size_explicit = params.draft_block_size_explicit,
        .drafter_path = params.drafter_dir,
        .hot_prefix_cache = null,
        .max_concurrent = cap,
        .force_batched = force_batched,
        .queue_mu = .init,
        .queue_cond = .init,
        .pending = std.ArrayList(*Slot).empty,
        .decoding = std.ArrayList(*Slot).empty,
        .vision_queue = std.ArrayList(*VisionEncodeRequest).empty,
        .embed_queue = std.ArrayList(*EmbedRequest).empty,
        .load_queue = std.ArrayList(*LoadRequest).empty,
        .gen_queue = std.ArrayList(*GenRequest).empty,
        .unload_queue = std.ArrayList(*UnloadRequest).empty,
        .cleanup_queue = std.ArrayList(*Slot).empty,
        .metrics = params.metrics,
        .inflight_generated_tokens = std.atomic.Value(u64).init(0),
        .inflight_prefill_tokens = std.atomic.Value(u64).init(0),
        .inflight_prefill_expected = std.atomic.Value(u64).init(0),
        .requests_prefilling = std.atomic.Value(u64).init(0),
        .in_flight = 0,
        .queue_cap = cap + 32,
        .submit_cond = .init,
        .session_cond = .init,
        .inference_thread = null,
        .shutdown = std.atomic.Value(bool).init(false),
        .started = std.atomic.Value(bool).init(false),
        .started_mu = .init,
        .started_cond = .init,
        .load_failed = std.atomic.Value(bool).init(false),
        .load_error_name = null,
    };

    const ctx = ThreadCtx{ .scheduler = self, .params = params };
    self.inference_thread = try std.Thread.spawn(.{}, inferenceLoop, .{ctx});

    // Wait until inference thread has loaded the model and (optionally)
    // warmed up. submit() relies on xfm/weights being live.
    self.started_mu.lockUncancelable(io);
    defer self.started_mu.unlock(io);
    while (!self.started.load(.acquire)) {
        self.started_cond.waitUncancelable(io, &self.started_mu);
    }

    if (self.load_failed.load(.acquire)) {
        // Inference thread already exited cleanly. Join + free + bubble up.
        if (self.inference_thread) |t| t.join();
        self.inference_thread = null;
        const name = self.load_error_name orelse "unknown";
        log.err("[scheduler] model load failed: {s}\n", .{name});
        return error.LoadFailed;
    }

    return self;
}

/// The inference thread's current step gets this long past the shutdown flush's deadline.
const JOIN_GRACE_MS: i64 = 30_000;

/// When `deinit` stops waiting for the inference thread: the flush deadline (0 = unarmed) or now, plus the grace.
fn joinDeadlineMs(flush_deadline_ms: i64, now_ms: i64) i64 {
    return @max(flush_deadline_ms, now_ms) + JOIN_GRACE_MS;
}

test "scheduler: shutdown waits for the inference thread past the flush deadline, then a grace" {
    try std.testing.expectEqual(@as(i64, 1_000 + JOIN_GRACE_MS), joinDeadlineMs(0, 1_000));
    try std.testing.expectEqual(@as(i64, 46_000 + JOIN_GRACE_MS), joinDeadlineMs(46_000, 1_000));
    try std.testing.expectEqual(@as(i64, 90_000 + JOIN_GRACE_MS), joinDeadlineMs(46_000, 90_000));
}

pub fn deinit(self: *Scheduler) void {
    self.shutdown.store(true, .release);
    // Wake inference thread if it's waiting on queue_cond.
    self.queue_mu.lockUncancelable(self.io);
    self.queue_cond.broadcast(self.io);
    self.submit_cond.broadcast(self.io);
    self.queue_mu.unlock(self.io);

    if (self.inference_thread) |t| {
        // A step that never returns (a receive from a cluster worker that left) or runs for minutes (a streamed pass
        // on a slow SSD) would hold the join forever: past the flush deadline and a grace, exit without it.
        const start = io_util.nowMsMonotonic(self.io);
        const until = joinDeadlineMs(self.shutdown_deadline_ms.load(.acquire), start);
        while (!self.inference_done.load(.acquire)) {
            if (io_util.nowMsMonotonic(self.io) >= until) {
                log.err("[shutdown] the inference thread is still inside a step after {d} s; exiting without it\n", .{@divTrunc(until - start, 1000)});
                std.process.exit(1);
            }
            std.Io.sleep(self.io, .fromMilliseconds(20), .real) catch {};
        }
        t.join();
    }
    // Past the join: the cluster's last message is this process's last MLX call on the link.
    @import("cluster/distributed.zig").shutdown();
    // A build in flight still writes its entry and request.
    while (self.ane_builds.load(.acquire) > 0) std.Io.sleep(self.io, .fromMilliseconds(10), .real) catch {};

    // Freed below the join: the publisher is gone, so this is the last writer.
    if (self.hot_cache_digests.len > 0) {
        self.allocator.free(self.hot_cache_digests);
        self.hot_cache_digests = &.{};
    }

    // Drain any leftover slots — should be empty if all conn threads
    // called `complete` properly, but defensive. Inference thread has
    // already exited by now (joined above), so freeing here is safe.
    for (self.pending.items) |slot| slot.deinit();
    self.pending.deinit(self.allocator);
    for (self.decoding.items) |slot| slot.deinit();
    self.decoding.deinit(self.allocator);
    self.offloaded.deinit(self.allocator);
    for (self.cleanup_queue.items) |slot| slot.deinit();
    self.cleanup_queue.deinit(self.allocator);
    // Vision/embed queues should be empty (encodeVision/computeEmbedding
    // block until done) but guard against shutdown-mid-encode by signaling
    // done with an error.
    for (self.vision_queue.items) |req| {
        req.done_mu.lockUncancelable(self.io);
        req.error_name = self.allocator.dupe(u8, "Shutdown") catch null;
        req.done = true;
        req.done_cond.broadcast(self.io);
        req.done_mu.unlock(self.io);
    }
    self.vision_queue.deinit(self.allocator);
    for (self.embed_queue.items) |req| {
        req.done_mu.lockUncancelable(self.io);
        req.error_name = self.allocator.dupe(u8, "Shutdown") catch null;
        req.done = true;
        req.done_cond.broadcast(self.io);
        req.done_mu.unlock(self.io);
    }
    self.embed_queue.deinit(self.allocator);
    // Phase D: signal any pending cold-load requesters that the
    // server is shutting down. They roll back their entry state and
    // free pre-loaded CPU resources.
    for (self.load_queue.items) |req| {
        req.done_mu.lockUncancelable(self.io);
        req.error_name = self.allocator.dupe(u8, "Shutdown") catch null;
        req.done = true;
        req.done_cond.broadcast(self.io);
        req.done_mu.unlock(self.io);
    }
    self.load_queue.deinit(self.allocator);
    // Wake any conn threads blocked in runGeneration / unloadModel. The
    // jobs never ran (so the gen body wrote no response — the conn thread
    // surfaces a 503), but we must release them so they don't hang.
    for (self.gen_queue.items) |req| {
        req.done_mu.lockUncancelable(self.io);
        req.done = true;
        req.done_cond.broadcast(self.io);
        req.done_mu.unlock(self.io);
    }
    self.gen_queue.deinit(self.allocator);
    for (self.unload_queue.items) |req| {
        req.done_mu.lockUncancelable(self.io);
        req.done = true;
        req.done_cond.broadcast(self.io);
        req.done_mu.unlock(self.io);
    }
    self.unload_queue.deinit(self.allocator);

    // Plan 05: mlx-allocating state lives on the `LoadedModel` owned by
    // the registry. We can't free the entries here (registry teardown
    // happens later, after serve() returns), but we DO need to release
    // their mlx pieces while we still have a thread bound to the mlx
    // GPU stream — that's actually the calling thread, since the
    // existing pattern frees mlx_array refcount-zeros from
    // `Scheduler.deinit` directly. Walk EVERY .ready entry in the
    // registry (multi-model: more than just `current_model`).
    {
        self.registry.mutex.lockUncancelable(self.io);
        defer self.registry.mutex.unlock(self.io);
        var it = self.registry.entries.valueIterator();
        while (it.next()) |entry_ptr| {
            const entry = entry_ptr.*;
            if (entry.state == .ready or entry.state == .evicting) {
                entry.unloadResident();
            }
        }
    }
    self.current_model = null;
    // Clear the borrowed views so post-shutdown reads (defensive) see
    // null rather than dangling pointers.
    self.xfm = null;
    self.weights = null;
    self.vision_encoder = null;
    self.drafter = null;
    self.dflash = null;
    self.hot_prefix_cache = null;
    self.resident_hot_cache_bytes.store(0, .monotonic);
    if (hot_cache_budget_invalidate) |f| f();
    self.reclaimable_hot_cache_bytes.store(0, .monotonic);
    if (self.load_error_name) |n| self.allocator.free(n);

    self.allocator.destroy(self);
}

/// Another request is queued, prefilling or decoding right now.
pub fn decodingCount(self: *Scheduler) usize {
    self.queue_mu.lockUncancelable(self.io);
    defer self.queue_mu.unlock(self.io);
    return self.decoding.items.len;
}

/// Another request this one could batch with is live. Slots on an engine beside MLX never join a group.
pub fn hasCompany(self: *Scheduler) bool {
    self.queue_mu.lockUncancelable(self.io);
    defer self.queue_mu.unlock(self.io);
    return self.in_flight > self.offloaded.items.len;
}

/// Submit a new request. Builds a Slot, queues it, returns the handle.
/// Blocks if the queue is full (i.e. `in_flight >= queue_cap`). When
/// the scheduler is shutting down this returns `error.Shutdown`.
pub fn submit(self: *Scheduler, params: SubmitParams) !*Slot {
    // Construct the slot up front so we don't hold the queue mutex
    // through any allocation. Per-request `kv_quant_config` override (Wave
    // 1.A) wins over the process-level default carried on the scheduler.
    // Phase D fix: use the slot's target-model config (not the
    // scheduler's startup-model config) so per-slot state allocation
    // (KVCache shape, SSM entries) matches the model that will
    // actually run the request. Critical when two models with
    // different architectures (e.g. pure-attention + hybrid SSM)
    // share one scheduler.
    const slot_config: *const ModelConfig = params.model.config orelse return error.ModelNotReady;
    if (params.model.engine) |e| if (engine_requests.missingCap(e, params)) |cap| {
        refusal.refused().* = .{ .kind = e.kind, .cap = cap };
        return error.EngineUnsupported;
    };
    // Every decode path reads the prompt's last token; `""` on a BOS-less vocab encodes to none.
    if (params.prompt_ids.len == 0) return error.EmptyPrompt;
    const eff_kv_quant = params.kv_quant_config orelse slot_config.kv_quant_override orelse self.kv_quant_config;
    const slot = try Slot.init(self.allocator, self.io, slot_config, params, eff_kv_quant);
    errdefer slot.deinit();

    self.queue_mu.lockUncancelable(self.io);
    defer self.queue_mu.unlock(self.io);

    while (self.in_flight >= self.queue_cap and !self.shutdown.load(.acquire)) {
        self.submit_cond.waitUncancelable(self.io, &self.queue_mu);
    }
    if (self.shutdown.load(.acquire)) return error.Shutdown;

    // An engine beside MLX serves as many requests at once as it has sessions (ds4's one, a llama.cpp
    // sequence, a Neural Engine slot), each reusing its KV across requests. Block here until one is free,
    // then claim it (released in `complete`).
    if (params.model.sessionCapacity()) |cap| {
        while (params.model.session_claims >= cap and !self.shutdown.load(.acquire)) {
            self.session_cond.waitUncancelable(self.io, &self.queue_mu);
        }
        if (self.shutdown.load(.acquire)) return error.Shutdown;
        params.model.session_claims += 1;
        slot.holds_session = true;
    }

    if (params.model.engine) |e| {
        // The engine holds `in_pass` until its `finished` event; `complete` waits that out.
        self.offloaded.append(self.allocator, slot) catch |err| {
            if (slot.holds_session) releaseSessionLocked(self, slot);
            return err;
        };
        engine_requests.submit(slot, e) catch |err| {
            _ = self.offloaded.pop();
            if (slot.holds_session) releaseSessionLocked(self, slot);
            return err;
        };
        self.in_flight += 1;
        // A parked inference thread re-reads `holdsSleepWhileParked`.
        self.queue_cond.broadcast(self.io);
        return slot;
    }
    self.pending.append(self.allocator, slot) catch |err| {
        // Release the session claim before bubbling the error — the caller
        // never gets the slot, so `complete` won't run for it.
        if (slot.holds_session) releaseSessionLocked(self, slot);
        return err;
    };
    self.in_flight += 1;
    self.queue_cond.broadcast(self.io);
    return slot;
}

/// Give back the slot's session claim. Caller holds `queue_mu`.
fn releaseSessionLocked(self: *Scheduler, slot: *Slot) void {
    slot.model.session_claims -= 1;
    slot.holds_session = false;
    self.session_cond.broadcast(self.io);
}

/// Hand the slot off to the inference thread for cleanup, and notify any
/// submitter waiting for queue space. Must be called once per slot
/// returned by `submit`. Safe whether or not the slot finished normally.
///
/// Why not free here: `slot.deinit()` walks the per-slot KVCache and
/// calls `mlx_array_free` on each entry. Refcount-shared GPU memory
/// release queues work against the array's owning stream, which is the
/// inference thread's. Freeing from a conn thread without that stream
/// binding crashes mlx 0.31.2 ("no Stream(gpu, N) in current thread").
/// So we remove the slot from any active list (so the inference thread
/// stops touching it) and queue it for cleanup on the inference thread.
pub fn complete(self: *Scheduler, slot: *Slot) void {
    // Mark cancelled so any in-flight tick filters this slot out of its
    // active list before we remove it from `decoding`. Idempotent.
    slot.cancelled.store(true, .release);

    self.queue_mu.lockUncancelable(self.io);

    // Remove from pending (rare — only if conn thread cancels before
    // prefill) and from decoding (the common case). After this, only the
    // cleanup queue references the slot, so the next inference-thread
    // cleanup drain can safely deinit it.
    var i: usize = 0;
    while (i < self.pending.items.len) : (i += 1) {
        if (self.pending.items[i] == slot) {
            _ = self.pending.orderedRemove(i);
            break;
        }
    }
    i = 0;
    while (i < self.decoding.items.len) : (i += 1) {
        if (self.decoding.items[i] == slot) {
            _ = self.decoding.orderedRemove(i);
            break;
        }
    }
    for (self.offloaded.items, 0..) |s, j| if (s == slot) {
        _ = self.offloaded.swapRemove(j);
        engine_requests.cancel(slot);
        break;
    };

    // Release the session claim (if this slot held it) so the next queued
    // request can claim it AND reuse the KV prefix it now holds. Done before
    // enqueueing cleanup so a waiting submitter can proceed immediately.
    if (slot.holds_session) releaseSessionLocked(self, slot);

    // Out of every list, so no new pass can take it; wait out the one that has it,
    // before the cleanup queue owns (and may free) the slot.
    if (slot.in_pass.load(.acquire) != 0) {
        self.queue_mu.unlock(self.io);
        while (slot.in_pass.load(.acquire) != 0) std.Io.sleep(self.io, .fromMilliseconds(1), .real) catch {};
        self.queue_mu.lockUncancelable(self.io);
    }

    self.cleanup_queue.append(self.allocator, slot) catch {
        // OOM on the cleanup list — fall back to inline deinit. This
        // races on mlx but is strictly better than the leak; the slot
        // is no longer referenced from pending/decoding above.
        self.queue_mu.unlock(self.io);
        slot.deinit();
        self.queue_mu.lockUncancelable(self.io);
    };
    if (self.in_flight > 0) self.in_flight -= 1;
    self.queue_cond.broadcast(self.io); // wake inference thread to drain
    self.submit_cond.broadcast(self.io); // wake any blocked submitter
    self.queue_mu.unlock(self.io);
}

/// Shutdown helper: signal every in-flight slot (pending + decoding) to
/// cancel so their owning connection threads unblock from `waitNext` and
/// run their `defer complete(...)` promptly. `server.serve` calls this when
/// the accept loop exits, THEN waits for the connection threads to drain
/// before returning (which triggers `deinit`) — otherwise a thread still in
/// `complete()` races `deinit`'s teardown of `pending`/`decoding`/
/// `cleanup_queue` into a use-after-free. Does NOT remove or free slots;
/// the conn threads own that via `complete()`.
pub fn cancelAllInFlight(self: *Scheduler) void {
    self.queue_mu.lockUncancelable(self.io);
    defer self.queue_mu.unlock(self.io);
    for (self.pending.items) |slot| slot.cancel();
    if (self.boundary_slot) |slot| slot.cancel();
    if (self.prefilling_from < self.n_prefilling) {
        for (self.prefilling[self.prefilling_from..self.n_prefilling]) |slot| slot.cancel();
    }
    for (self.decoding.items) |slot| slot.cancel();
    for (self.offloaded.items) |slot| {
        slot.cancel();
        engine_requests.cancel(slot);
    }
}

/// Ends every request in flight with the error `name`, from any thread: for an inference thread that will never
/// return from its step (a cluster worker left mid-exchange), so the readers are told before the process exits.
/// The stuck pass never drops its slots' `in_pass`; it is dropped here, or each reader's `complete` waits forever.
pub fn failInFlight(self: *Scheduler, name: []const u8) void {
    self.queue_mu.lockUncancelable(self.io);
    defer self.queue_mu.unlock(self.io);
    for (self.pending.items) |slot| failSlot(slot, name);
    if (self.boundary_slot) |slot| failSlot(slot, name);
    if (self.prefilling_from < self.n_prefilling) {
        for (self.prefilling[self.prefilling_from..self.n_prefilling]) |slot| failSlot(slot, name);
    }
    for (self.decoding.items) |slot| failSlot(slot, name);
}

fn failSlot(slot: *Slot, name: []const u8) void {
    slot.markError(name);
    slot.in_pass.store(0, .release);
}

/// The media residency bill for an entry, or 0 when it is not a media
/// model. The backend type must come from the DIRECTORY, never from the
/// entry's stub config, whose `model_type` is the MODALITY static
/// ("AudioVideo" for every video backend) and would bill the sum.
/// `arch_hint` is discovery's own read of config.json — the same authority
/// `gen.peekModelType` is — so it is used when present and the peek is the
/// fallback for an entry registered without one.
pub fn mediaPeakFor(self: *Scheduler, entry: *LoadedModel) u64 {
    if (entry.arch_hint.len > 0) {
        if (gen_mod.modalityFromType(entry.arch_hint) == null) return 0;
        return gen_mod.estimatePeakResidentBytes(self.io, entry.path, entry.arch_hint);
    }
    const peeked = gen_mod.peekModelType(self.io, self.allocator, entry.path) orelse return 0;
    defer self.allocator.free(peeked);
    if (gen_mod.modalityFromType(peeked) == null) return 0;
    return gen_mod.estimatePeakResidentBytes(self.io, entry.path, peeked);
}

/// Plan 05 Phase D: resolve `id_or_empty` ("" / "mlx-serve" → default)
/// to a refcounted, ready `*LoadedModel`. Cold-loads on demand: if the
/// entry is `.unloaded`, parses CPU state, picks an LRU victim if
/// over caps, and posts a `LoadRequest` to the inference thread,
/// blocking until the load completes.
///
/// Caller MUST call `release(lm)` once done.
///
/// Errors:
///   error.UnknownModelId    — id isn't in the registry.
///   error.NoDefaultModel    — id empty AND no default set.
///   error.NotEnoughMemory   — would exceed caps and no LRU victim.
///   error.InsufficientMemory — memory preflight refused the load (free
///                             RAM can't hold weights + headroom).
///   error.LoadFailed        — inference thread reported a load failure.
///   error.Shutdown          — scheduler is shutting down.
pub fn ensureLoaded(self: *Scheduler, id_or_empty: []const u8) !*LoadedModel {
    // A lost GPU context fails every MLX call: nothing is loaded or served until a restart.
    if (mlx.gpuContextLost()) return error.GpuContextLost;
    // Fast path: ready entries. registry.ensureLoaded handles waiting
    // out .loading / .evicting transitions by other callers.
    const fast_result = self.registry.ensureLoaded(id_or_empty);
    if (fast_result) |lm| return lm else |err| switch (err) {
        error.NotLoaded => {}, // fall through to slow path
        else => return err,
    }

    // Slow path: cold load. Resolve the entry. Re-acquire mutex and
    // re-check state — between the fast-path call and now another
    // caller could have completed the load.
    const entry = try self.registry.resolveEntry(id_or_empty);

    // CPU-only pre-load: parse config / load tokenizer / load chat
    // config. Cheap (~tens of ms); kept outside the mutex so other
    // requests on other models stay unblocked. On failure, mark the
    // entry `.error_state` so /v1/models surfaces the failure (and
    // future ensureLoaded calls fail fast instead of re-tripping the
    // same parse error). FileNotFound / parse errors land here.
    var settings = model_settings.overrideFor(self.allocator, self.io, entry.path);
    defer settings.deinit(self.allocator);
    const cpu_state = preloadCpuState(self.allocator, self.io, entry.path, settings.ctx_size orelse self.gguf_ctx_size, settings.ggufEngine()) catch |err| {
        self.registry.mutex.lockUncancelable(self.io);
        self.registry.markErrorLocked(entry, @errorName(err));
        self.registry.mutex.unlock(self.io);
        return error.LoadFailed;
    };
    // Ownership: on success transfers to the entry inside the
    // inference thread's `doLoadOnInferenceThread`; on any error from
    // here on, free them ourselves before returning.
    var owned = cpu_state;
    var owned_active: bool = true;
    defer if (owned_active) freeCpuState(self.allocator, &owned);
    applyModelSettings(owned.config, owned.chat_config, &settings, self.mtp_enabled);
    // The resolved .gguf path (when this is a GGUF entry) is borrowed by
    // the LoadRequest until `done`; the engines dupe what they keep, so
    // it's released here on success AND failure.
    defer if (owned.gguf) |g| self.allocator.free(g.path);

    // Victims selected by the eviction planner (multi-victim: one load may
    // need to free several models to fit). Lives on this stack frame; the
    // slice handed to the LoadRequest stays valid while we block on `done`.
    var victims_buf: [16]*LoadedModel = undefined;
    var n_victims: usize = 0;
    // The requester's reference, taken when the load is claimed: a model that turns `.ready`
    // with no reference is another load's free eviction victim before this thread wakes.
    var pinned = false;
    errdefer if (pinned) self.registry.releaseStatus(entry);

    // Peeked OUTSIDE the registry mutex — it stats the model dir, and no
    // other load should block on our filesystem.
    const media_peak = mediaPeakFor(self, entry);
    const mlx_text = owned.gguf == null and gen_mod.modalityFromType(owned.config.model_type) == null;
    const validated_model_bytes: ?u64 = if (mlx_text)
        residentModelDiskBytes(self.allocator, self.io, entry.path, owned.config) catch |err| {
            self.registry.mutex.lockUncancelable(self.io);
            if (entry.state == .unloaded) self.registry.markErrorLocked(entry, @errorName(err));
            self.registry.mutex.unlock(self.io);
            return error.LoadFailed;
        }
    else
        null;
    const resident_model_bytes = residentGateBytes(validated_model_bytes, entry.bytes_on_disk, mlx_text);
    // A Neural Engine load holds int8 copies of the pack's 4- to 8-bit weights, and its KV.
    const ane_bytes: ?u64 = if (mlx_text and engines_served.aneWanted(owned.config, self.ane))
        engines_load.aneBytes(self.allocator, entry.path, self.ane_slots)
    else
        null;
    const gate_model_bytes: ?u64 = if (ane_bytes) |b| @max(b, resident_model_bytes orelse 0) else resident_model_bytes;

    // ── Stage 1 (registry mutex): claim .loading, plan eviction.
    {
        self.registry.mutex.lockUncancelable(self.io);
        errdefer self.registry.mutex.unlock(self.io); // bail on early returns

        // Wait out any concurrent .loading / .evicting state.
        wait_loop: while (true) {
            switch (entry.state) {
                .ready => {
                    _ = entry.refcount.fetchAdd(1, .acq_rel);
                    self.registry.mutex.unlock(self.io);
                    return entry;
                },
                .loading, .evicting => {
                    self.registry.state_cond.waitUncancelable(self.io, &self.registry.mutex);
                    continue :wait_loop;
                },
                .error_state => {
                    const load_err = ModelRegistry.loadErrorFromName(entry.error_name);
                    self.registry.mutex.unlock(self.io);
                    return load_err;
                },
                .unloaded => break :wait_loop,
            }
        }

        // One stream pack at a time reads the process's foreign-bytes value (DeepSeek-V4.1, GLM-5.3).
        const foreign_reader = try self.registry.foreignReaderVictimLocked(entry.id, owned.config);

        // Claim the slot.
        std.debug.assert(self.registry.tryBeginLoadLocked(entry));
        // From the claim: while victims unload or the queue waits, not the last load's numbers.
        entry.load_progress.start(io_util.nowMsMonotonic(self.io));
        _ = entry.refcount.fetchAdd(1, .acq_rel);
        pinned = true;

        // Estimate post-load bytes (see `gateEstimateBytes` for why a media
        // entry cannot be billed by its directory's size).
        const estimated: u64 = gateEstimateBytes(media_peak, gate_model_bytes, owned.config.num_hidden_layers, owned.config.hidden_size) +
            pleTableBill(self.io, owned.config);

        // Reserve this load's estimate BEFORE planning eviction, so a
        // concurrent loader sees the pending allocation in its own gate.
        // Without this, two loads can both read a stale resident total
        // (one's bytes not yet committed at markReady), both skip eviction,
        // and oversubscribe GPU memory → Metal OOM → process crash.
        self.registry.reserveLoadLocked(entry, estimated);

        // Evict LRU victims until both caps hold for this reservation
        // (multi-victim). On failure — every other resident model is pinned
        // by an in-flight request — roll back and surface a 503 instead of
        // loading anyway and crashing.
        const n = self.registry.planEvictionsLocked(entry.id, foreign_reader, &victims_buf, mlxActiveBytes()) orelse {
            // Name the numbers. A refusal that logs NOTHING sends the user
            // hunting for a concurrent request that does not exist: on an
            // idle server the cause is always the static cap (#126), and
            // the flag that moves it is not otherwise discoverable.
            const gb = 1024.0 * 1024.0 * 1024.0;
            if (loadExceedsMemCap(estimated, self.registry.max_resident_mem)) {
                log.err("Refusing to load {s}: needs ~{d:.2} GB, more than --max-resident-mem {d:.2} GB on its own, so unloading other models cannot help. Raise the cap.\n", .{
                    entry.id,
                    @as(f64, @floatFromInt(estimated)) / gb,
                    @as(f64, @floatFromInt(self.registry.max_resident_mem)) / gb,
                });
            } else log.err("Refusing to load {s}: needs ~{d:.2} GB with {d:.2} GB resident ({d:.2} GB live, {d:.2} GB reserved by loads in flight, this one included) across {d} model(s), and unloading the idle ones does not fit it under --max-resident-models {d} and --max-resident-mem {d:.2} GB (0 = off). Unload a model or raise a cap.\n", .{
                entry.id,
                @as(f64, @floatFromInt(estimated)) / gb,
                @as(f64, @floatFromInt(self.registry.current_resident_bytes)) / gb,
                @as(f64, @floatFromInt(mlxActiveBytes())) / gb,
                @as(f64, @floatFromInt(self.registry.reserved_bytes)) / gb,
                self.registry.countLoadedLocked(),
                self.registry.max_resident_models,
                @as(f64, @floatFromInt(self.registry.max_resident_mem)) / gb,
            });
            self.registry.markUnloadedLocked(entry); // releases the reservation
            self.registry.mutex.unlock(self.io);
            return error.NotEnoughMemory;
        };
        n_victims = n;
        // Drain readers on each victim before the inference thread frees it.
        for (victims_buf[0..n_victims]) |v| self.registry.waitForRefcountZeroLocked(v);
        self.registry.mutex.unlock(self.io);
    }

    // ── Stage 2: build + post LoadRequest, wait for completion.
    var req = LoadRequest{
        .entry = entry,
        .config = owned.config,
        .tok = owned.tok,
        .chat_config = owned.chat_config,
        .model_dir = entry.path,
        .resident_model_bytes = resident_model_bytes,
        // `--no-drafter` / `--drafter` / `--draft-block-size` reach cold
        // loads too; the path itself is scoped by `coldLoadDrafterDir`.
        .drafter_dir = coldLoadDrafterDir(self.no_drafter, self.primary_model_dir, self.drafter_dir, entry.path),
        .no_drafter = self.no_drafter,
        .load_vision = coldLoadVision(owned.config.has_vision),
        .warmup_eager = true,
        .draft_block_size = self.draft_block_size,
        .draft_block_size_explicit = self.draft_block_size_explicit,
        .kv_quant_config = self.kv_quant_config,
        // Cold loads get the SAME prefix-cache configuration as the
        // startup model — pre-plumbing these were (1, 0, stride 0),
        // which silently degraded warm reuse after every model switch.
        .prefix_cache_capacity = self.prefix_cache_capacity,
        .prefix_cache_ram_enabled = self.prefix_cache_ram_enabled,
        .prefix_cache_mem_bytes = self.prefix_cache_mem_bytes,
        .prefix_cache_disk_bytes = self.prefix_cache_disk_bytes,
        .ssm_checkpoint_stride = self.ssm_checkpoint_stride,
        .ssm_checkpoint_max = self.ssm_checkpoint_max,
        // Cold loads honor the launch-flag MTP + embedded-llama.cpp
        // settings too (same reason as prefix-cache above) — pre-plumbing
        // these were LoadRequest defaults, so --no-mtp / --mtp-max-depth /
        // --llama-* were silently dropped on every on-demand load and
        // model switch.
        .mtp_enabled = self.mtp_enabled,
        .mtp_head_kv_quant = self.mtp_head_kv_quant,
        .mtp_depth = self.mtp_depth,
        .llama = self.llama,
        .ds4_mtp = self.ds4_mtp,
        .ds4_dspark = self.ds4_dspark,
        .ds4_ssd_streaming = self.ds4_ssd_streaming,
        .ane_prefill = self.ane_prefill,
        .ane = self.ane,
        .ane_slots = self.ane_slots,
        .ane_chunk_resolver = self.ane_chunk_resolver,
        .ane_headroom_resolver = self.ane_headroom_resolver,
        .evict_entries = victims_buf[0..n_victims],
        .allocator = self.allocator,
    };
    // A GGUF entry opens on its engine (issue #59: discovered/pulled GGUF dirs cold-load on demand).
    req.gguf = owned.gguf;

    {
        self.queue_mu.lockUncancelable(self.io);
        defer self.queue_mu.unlock(self.io);
        if (self.shutdown.load(.acquire)) return error.Shutdown;
        try self.load_queue.append(self.allocator, &req);
        self.queue_cond.broadcast(self.io);
    }

    // Block until the inference thread signals done.
    req.done_mu.lockUncancelable(self.io);
    while (!req.done) req.done_cond.waitUncancelable(self.io, &req.done_mu);
    req.done_mu.unlock(self.io);

    if (req.error_name) |name| {
        // The failure crosses the thread boundary by NAME — map it back
        // to a typed error so a memory-preflight refusal surfaces as a
        // named 503, not the generic "Model load failed" 500 (#144).
        defer self.allocator.free(name);
        // On success the inference thread took ownership of cpu_state;
        // on failure it didn't, so we still hold it.
        return ModelRegistry.loadErrorFromName(name);
    }

    // Success — inference thread installed cpu_state onto the entry.
    owned_active = false;

    // Already refcounted (`pinned`), so no other load evicted it; only an explicit
    // unload can have claimed it since `markReady`, and that is a 503, not a failed load.
    self.registry.mutex.lockUncancelable(self.io);
    const ready = entry.state == .ready;
    self.registry.mutex.unlock(self.io);
    if (!ready) return error.NotLoaded;
    return entry;
}

/// Release a borrowed pointer obtained from `ensureLoaded`. Forwards
/// to the registry so the refcount decrement + LRU clock bump happen
/// under registry.mutex.
pub fn release(self: *Scheduler, lm: *LoadedModel) void {
    self.registry.release(lm);
}

/// `release` for a status read — same refcount protocol, no recency
/// stamp. See `ModelRegistry.releaseStatus`.
pub fn releaseStatus(self: *Scheduler, lm: *LoadedModel) void {
    self.registry.releaseStatus(lm);
}

/// Duped stored failure name for the id `ensureLoaded` just refused with
/// `error.LoadFailed` — feeds the "Model load failed: <name>" HTTP
/// message (#144). Caller frees.
pub fn loadErrorName(self: *Scheduler, alloc: std.mem.Allocator, id_or_empty: []const u8) ?[]u8 {
    return self.registry.loadErrorNameDupe(alloc, id_or_empty);
}

/// Run a media-generation job on the inference thread. Posts `req` to the
/// gen queue and blocks the calling (connection) thread until the
/// inference thread has run the job body to completion. The job writes its
/// own HTTP/SSE response to the connection; this just synchronizes.
pub fn runGeneration(self: *Scheduler, req: *GenRequest) !void {
    {
        self.queue_mu.lockUncancelable(self.io);
        defer self.queue_mu.unlock(self.io);
        if (self.shutdown.load(.acquire)) return error.Shutdown;
        try self.gen_queue.append(self.allocator, req);
        self.queue_cond.broadcast(self.io);
    }
    req.done_mu.lockUncancelable(self.io);
    while (!req.done) req.done_cond.waitUncancelable(self.io, &req.done_mu);
    req.done_mu.unlock(self.io);
}

/// Free a model's resident GPU state, returning its registry stub to
/// `.unloaded` so it can reload later. Idempotent — a non-resident model
/// returns immediately. Marks the entry `.evicting`, drains in-flight
/// requests (refcount → 0), then hands the mlx free to the inference
/// thread (stream-bound). Blocks until the free completes.
pub fn unloadModel(self: *Scheduler, id_or_empty: []const u8) !void {
    return unloadModelIfIdle(self, id_or_empty, null);
}

/// `unloadModel` with an optional idle precondition, re-checked under the
/// registry mutex at the moment we commit.
///
/// The sweep picks a victim, drops the mutex, then calls in, so a request
/// can arrive in that gap: refcount and age are re-checked under the mutex
/// before committing. (`planEvictionsLocked` picks and marks under one
/// hold instead; the sweep cannot, because the unload itself is slow.)
pub fn unloadModelIfIdle(self: *Scheduler, id_or_empty: []const u8, idle_window_ms: ?i64) !void {
    const entry = try self.registry.resolveEntry(id_or_empty);
    {
        self.registry.mutex.lockUncancelable(self.io);
        wait: while (true) {
            switch (entry.state) {
                // Already free (or failed-load stub) → nothing to do.
                .unloaded, .error_state => {
                    self.registry.mutex.unlock(self.io);
                    return;
                },
                // Another caller is mid load/evict — wait it out, then
                // re-check (it may end up resident or unloaded).
                .loading, .evicting => {
                    self.registry.state_cond.waitUncancelable(self.io, &self.registry.mutex);
                    continue :wait;
                },
                .ready => break :wait,
            }
        }
        if (idle_window_ms) |window_ms| {
            const now_ms = io_util.nowMsMonotonic(self.io);
            if (!model_registry_mod.ModelRegistry.idleEvictable(entry, now_ms, window_ms)) {
                self.registry.mutex.unlock(self.io);
                return;
            }
        }
        self.registry.markEvictingLocked(entry);
        self.registry.waitForRefcountZeroLocked(entry);
        self.registry.mutex.unlock(self.io);
    }

    var req = UnloadRequest{ .entry = entry };
    {
        self.queue_mu.lockUncancelable(self.io);
        defer self.queue_mu.unlock(self.io);
        // On shutdown the entry stays `.evicting`; `Scheduler.deinit`
        // unloads every `.ready`/`.evicting` entry, so it's still freed.
        if (self.shutdown.load(.acquire)) return error.Shutdown;
        try self.unload_queue.append(self.allocator, &req);
        self.queue_cond.broadcast(self.io);
    }
    req.done_mu.lockUncancelable(self.io);
    while (!req.done) req.done_cond.waitUncancelable(self.io, &req.done_mu);
    req.done_mu.unlock(self.io);
}

/// The least recently used resident model nobody is holding, other than `keep`; its id
/// outlives the unload (the entry stays registered as a stub).
pub fn idleModelToUnload(self: *Scheduler, keep: []const u8) ?[]const u8 {
    self.registry.mutex.lockUncancelable(self.io);
    defer self.registry.mutex.unlock(self.io);
    const entry = self.registry.pickLruEvictable(keep, false) orelse return null;
    return entry.id;
}

/// Synchronously compute embeddings for `req.token_seqs` using the
/// batched encoder forward pass on the inference thread. Same lifecycle
/// as `encodeVision`: post + block + return results. Caller frees the
/// returned rows + outer slice (allocated with `req.allocator`).
pub fn computeEmbeddings(self: *Scheduler, req: *EmbedRequest) ![][]f32 {
    self.queue_mu.lockUncancelable(self.io);
    self.embed_queue.append(self.allocator, req) catch |err| {
        self.queue_mu.unlock(self.io);
        return err;
    };
    self.queue_cond.broadcast(self.io);
    self.queue_mu.unlock(self.io);

    req.done_mu.lockUncancelable(self.io);
    defer req.done_mu.unlock(self.io);
    while (!req.done) {
        req.done_cond.waitUncancelable(self.io, &req.done_mu);
    }
    if (req.error_name) |_| return error.EmbedFailed;
    return req.results orelse error.EmbedFailed;
}

/// Phase A4: synchronously encode one or more images and return the
/// embedding tensor. Conn thread fills `req.images` (CHW float32 pixel
/// buffers, decoded by stb_image / libwebp on the conn thread); this
/// method posts the request to the inference thread, blocks until done,
/// and returns the resulting `mlx_array` on success. Ownership of the
/// returned array transfers to the caller — typically passed straight
/// into `submit(.{ .vision_embeddings = arr, ... })` so the slot owns
/// it and frees on `deinit`.
///
/// Returns `error.VisionEncodeFailed` if the inference thread fails.
/// The request struct must outlive this call, but since the call blocks,
/// a stack allocation in the caller works.
pub fn encodeVision(self: *Scheduler, req: *VisionEncodeRequest) !mlx.mlx_array {
    // Post + wake.
    self.queue_mu.lockUncancelable(self.io);
    self.vision_queue.append(self.allocator, req) catch |err| {
        self.queue_mu.unlock(self.io);
        return err;
    };
    self.queue_cond.broadcast(self.io);
    self.queue_mu.unlock(self.io);

    // Wait for completion.
    req.done_mu.lockUncancelable(self.io);
    defer req.done_mu.unlock(self.io);
    while (!req.done) {
        req.done_cond.waitUncancelable(self.io, &req.done_mu);
    }
    if (req.error_name) |_| return error.VisionEncodeFailed;
    return req.result orelse error.VisionEncodeFailed;
}

/// Does this slot's next decode tick actually run the regular (non-speculative)
/// path? The slot's `enable_*` flags carry the REQUEST's wish; `specTickMode` is
/// the authoritative dispatch answer, and a generator can also have turned spec
/// off at runtime. Reading the armed flags here vetoed batched decode for any
/// slot whose prompt merely n-gram-scored high enough to ARM PLD, even after
/// PLD's own yield gate had disabled itself — neither speculation nor batching
/// (measured 2.4x on concurrent GDN decode). Same class as "a guard that shapes
/// INIT options does not bind DISPATCH".
fn slotTicksRegular(slot: *const Slot) bool {
    if (Planner.enabled() and slot.planner_force_plain) return true;
    const gen = if (slot.legacy_gen) |*g| g else return !(slot.enable_pld or slot.enable_drafter or slot.enable_mtp);
    // A runtime-disabled generator is already ticking regular, so batching it
    // dispatches what it was going to dispatch anyway.
    //
    // The POLICY this encodes, which is deliberate and not free: `nextPld`'s
    // periodic re-enable check (`SPEC_REENABLE_INTERVAL`) lives on the serial
    // path, and the batched tick never calls it. So a slot whose PLD yield
    // gate disabled itself stays disabled for as long as it keeps company —
    // even if its tail later turns into the file/tool echo PLD is best at.
    // That is the right trade at N>1 (the batched kernel reads the weight set
    // once for the whole group, which beats one slot's lookup wins) and it
    // self-corrects: a SOLO slot never reaches the batched path, so the
    // re-enable check resumes the moment concurrency drops back to one.
    // Pinned by `a spec_disabled_runtime slot is batchable, and that is the
    // documented trade` below — flip either half deliberately, not by accident.
    return gen.spec_disabled_runtime or slotSpecMode(slot, gen) == .regular;
}

/// Does this slot owe a module-head release? One single-slot tick lands it.
fn slotReleasePending(slot: *const Slot) bool {
    const gen = if (slot.legacy_gen) |*g| g else return false;
    return gen.mtpReleasePending();
}

/// Active-tick gate. Decides whether a slot is eligible for the batched
/// decode kernel. Hybrid SSM / MoE / encoder / DSV4 models can't ride
/// the batched kernel (it doesn't model their state), so any slot
/// targeting such a model falls through to the single-slot path. Phase
/// D: the gate reads off the slot's own model config — multi-model
/// means the scheduler's startup config is no longer authoritative.
fn batchable(self: *const Scheduler, slot: *const Slot) bool {
    return batchVerdict(self, slot) == .ok;
}

/// Why a slot does or does not ride the batched kernel. `.ok` batches; every
/// other arm decodes serial this tick and is what the `[batched] serial` line
/// and `decode_serial_total{reason}` name.
pub fn batchVerdict(self: *const Scheduler, slot: *const Slot) BatchVerdict {
    _ = self;
    if (slot.legacy_gen != null and slot.legacy_gen.?.speech != null) return .speech;
    if (!slotTicksRegular(slot)) return .spec_active;
    // A slot whose module-head release is armed but not landed still holds the head.
    if (slotReleasePending(slot)) return .head_release_pending;
    if (slot.sampling.constraint != null) return .grammar;
    if (slot.logprobs_n > 0) return .logprobs;
    if (slot.sampling.penalized()) return .penalty;
    const cfg = slot.model.config orelse return .arch;
    if (modelBatchable(cfg)) return .ok;
    // A GatedDeltaNet trunk is rejected by the pure-config predicate (it is
    // a hybrid), but has its own batched kernel. Ask the transformer, never
    // name the arch here — same rule as `modelExclusiveDecode`.
    const t = slot.model.transformer orelse return .arch;
    return if (t.supportsBatchedGdnDecode()) .ok else .arch;
}

/// Pure: would loading `weights_bytes` of model with `avail_bytes` free RAM risk
/// a Metal OOM? Requires the weights plus ~1/12 (≈8%) + 0.25 GB headroom for the
/// warmup KV cache + compute buffers. Deliberately lean: `avail_bytes` (active +
/// wired + compressed subtracted) under-counts what macOS reclaims from file
/// cache the moment MLX allocates, so a fat headroom wrongly refuses loads that
/// fit. The guard's real job is the gross case (restart a 42 GB model into 44 GB
/// free → hard process-killing OOM), which this still catches. Returns false
/// (allow the load) when either figure is 0 — a failed memory query must never
/// block a load.
/// Set by `--skip-mem-preflight` (main.zig) to bypass the model-load memory
/// pre-flight below. A module global, not a `LoadParams` field, so it applies
/// uniformly to startup loads AND later hot-loads — matching the env var
/// (`MLX_SERVE_SKIP_MEM_PREFLIGHT`) it replaced.
pub var skip_mem_preflight: bool = false;

/// Process-wide vision opt-out (`--no-vision` / the iPhone app, which has no
/// image-input UI yet). A module global for the same reason as
/// `skip_mem_preflight`: it must apply to on-demand /v1/load-model cold loads
/// too, not just the startup `LoadParams` — the cold-load path used to
/// hardcode `load_vision = config.has_vision` and silently ignore the flag.
pub var no_vision_global: bool = false;

/// Chat interleaves with a running media job (GenYield) except on CUDA: there a decode that ran
/// inside one image job left state the next job's first step faulted on once that chat model
/// was evicted (an illegal memory access). Set at boot by `server.serve`.
pub var gen_yield_enabled = true;

/// Target fraction of wall time the decoding streams get while another slot prefills,
/// set once by `main` from `--prefill-decode-share` or MLX_SERVE_PREFILL_DECODE_SHARE
/// (`resolveDecodeShare`). 0 = one boundary's `interleaveTicksFor` ticks and the company
/// width, as before.
pub var prefill_decode_share: f32 = 0;

const testing = std.testing;

test "a live MTP slot never ticks regular, an unarmed one always does" {
    try testing.expectEqual(SpecTickMode.mtp, specTickMode(true, true, false, false, false, true, true, false));
    try testing.expectEqual(SpecTickMode.regular, specTickMode(true, false, true, false, false, true, false, false));
}

test "company: a slot on an engine beside MLX never batches, so it does not count" {
    var sch: Scheduler = undefined;
    sch.io = testing.io;
    sch.queue_mu = .init;
    sch.offloaded = .empty;
    defer sch.offloaded.deinit(testing.allocator);
    var ane: Slot = undefined;
    try sch.offloaded.append(testing.allocator, &ane);
    sch.in_flight = 1;
    try testing.expect(!hasCompany(&sch));
    sch.in_flight = 2;
    try testing.expect(hasCompany(&sch));
}

test "failInFlight: a decoding request ends with the error by name, its stuck pass released" {
    var sch: Scheduler = undefined;
    sch.io = testing.io;
    sch.queue_mu = .init;
    sch.pending = .empty;
    sch.decoding = .empty;
    sch.boundary_slot = null;
    sch.prefilling_from = 0;
    sch.n_prefilling = 0;
    var slot: Slot = undefined;
    slot.io = testing.io;
    slot.allocator = testing.allocator;
    slot.out_mu = .init;
    slot.out_cond = .init;
    slot.out_event = .unset;
    slot.finished = false;
    slot.error_code = null;
    slot.sink = null;
    slot.in_pass = .init(1);
    try sch.decoding.append(testing.allocator, &slot);
    defer sch.decoding.deinit(testing.allocator);
    failInFlight(&sch, "DistWorkerLeft");
    defer testing.allocator.free(slot.error_code.?);
    try testing.expectEqualStrings("DistWorkerLeft", slot.error_code.?);
    // The stuck pass's hold is dropped, so the reader's `complete` returns.
    try testing.expectEqual(@as(u32, 0), slot.in_pass.load(.acquire));
}

test "cancelAllInFlight: a request being prefilled at a chunk boundary is cancelled too" {
    var sch: Scheduler = undefined;
    sch.io = testing.io;
    sch.queue_mu = .init;
    sch.pending = .empty;
    sch.decoding = .empty;
    sch.offloaded = .empty;
    sch.prefilling_from = 0;
    sch.n_prefilling = 0;
    var slot: Slot = undefined;
    slot.io = testing.io;
    slot.cancelled = .init(false);
    slot.out_mu = .init;
    slot.out_cond = .init;
    slot.out_event = .unset;
    sch.boundary_slot = &slot;
    cancelAllInFlight(&sch);
    try testing.expect(slot.cancelled.load(.acquire));
}
