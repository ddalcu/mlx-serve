//! The GPU thread's shared state: the queues connection threads post to, the slots in flight, the launch settings
//! that shape every load, and the numbers `/props` and `/metrics` read. The MLX engine runs the thread and owns the
//! work (`engines/mlx/scheduler.zig`); the host reaches it through `host/engines/mlx.zig`.

const std = @import("std");
const handles = @import("../../abi/mlx_handles.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const metrics_mod = @import("../metrics/metrics.zig");
const engines_served = @import("../engines/served.zig");
const model_registry_mod = @import("../registry/registry.zig");
const requests = @import("requests.zig");
const params = @import("params.zig");
const mlx = @import("../../mlx/ffi.zig");
const mlx_engine = @import("../engines/mlx.zig");
const ModelConfig = @import("../../sdk/config/model.zig").ModelConfig;
const KVQuantConfig = @import("../../sdk/blocks/attention/kv_quant.zig").KVQuantConfig;
const Weights = @import("../../sdk/weights/resident.zig").Weights;
const LlamaSettings = @import("../../sdk/quant/gguf/llama_settings.zig").Settings;
const ModelRegistry = model_registry_mod.ModelRegistry;
const LoadedModel = model_registry_mod.LoadedModel;
const Transformer = handles.Transformer;
const VisionEncoder = handles.VisionEncoder;
const DrafterModel = handles.DrafterModel;
const DflashModel = handles.DflashModel;
const Slot = handles.Slot;
const PromiseLedger = params.PromiseLedger;
const VisionEncodeRequest = requests.VisionEncodeRequest;
const EmbedRequest = requests.EmbedRequest;
const LoadRequest = requests.LoadRequest;
const GenRequest = requests.GenRequest;
const UnloadRequest = requests.UnloadRequest;

/// Most slots one batched decode tick advances.
pub const MAX_BATCH_GROUP = 32;

/// Why a slot decodes serially beside company (`decode_serial_total{reason}`).
pub const BatchVerdict = enum {
    ok,
    spec_active,
    head_release_pending,
    grammar,
    logprobs,
    penalty,
    embedded_engine,
    arch,
    pad_waste,
    /// A spoken reply: one row a slot, its frames drawn between forwards.
    speech,
};

/// Options for re-clamping the hot-cache budgets after residency changed.
pub const BudgetRevise = struct { exclude_bytes: u64 = 0, quiet: bool = false };

/// One hot-cache entry as the connection-thread guard sees it (`Scheduler.hot_cache_digests`).
pub const HotCacheDigest = struct {
    fingerprint: u64,
    len: u32,
    kv_bytes: u64,
};

/// One per server. Created by the engine (`mlx.start`), which spawns the GPU thread that drains these queues.
pub const Scheduler = struct {
    allocator: std.mem.Allocator,
    io: std.Io,

    // ── Plan 05 — multi-model state. Source of truth for model fields is
    //    `current_model` (a borrowed *LoadedModel owned by the registry).
    //    The fields below (`xfm`, `weights`, …) are *borrowed views* into
    //    `current_model` set at load time so existing scheduler-internal
    //    code can keep reading them as fields. Phase D will refresh these
    //    views on every model swap inside `pickNextTickWork`.
    registry: *ModelRegistry,
    current_model: ?*LoadedModel,

    // ── Borrowed views (non-owning). Cleared on shutdown via
    //    `clearCurrentModelViews`. Null only before load completes or after
    //    an eviction in Phase D.
    xfm: ?*Transformer,
    weights: ?*Weights,
    vision_encoder: ?*VisionEncoder,
    drafter: ?*DrafterModel,
    dflash: ?*DflashModel = null,
    drafter_block_size: u32,
    kv_quant_config: KVQuantConfig,
    /// `LoadParams.ctx_size` — the --ctx-size launch flag, kept for sizing
    /// GGUF stub configs on cold loads (see preloadGgufCpuState).
    gguf_ctx_size: u32,
    /// Launch-flag prefix-cache settings, retained so COLD-LOADED models
    /// (`ensureLoaded` → /v1/load-model, model switches) get the same
    /// prefix-cache behavior as the `--model` primary. Pre-plumbing these
    /// were hardcoded to (1, 0, stride 0) on the cold path, which silently
    /// crippled warm reuse — and disabled it entirely on hybrids — after
    /// every model switch.
    prefix_cache_capacity: u32,
    prefix_cache_ram_enabled: bool,
    prefix_cache_mem_bytes: u64,
    prefix_cache_mem_resolver: ?*const fn (*ModelConfig, u64, BudgetRevise, *u64) u64,
    prefix_cache_disk_bytes: u64,
    ssm_checkpoint_stride: u32,
    ssm_checkpoint_max: u32,
    /// Launch-flag MTP + embedded-llama.cpp settings, retained (same rationale
    /// as the prefix-cache fields above) so COLD-LOADED models — on-demand
    /// `/v1/load-model`, model switches — honor `--no-mtp` / `--mtp-max-depth` /
    /// the `--llama-*` flags like the `--model` primary. Pre-plumbing, the
    /// cold-load `LoadRequest` used its struct defaults, silently ignoring
    /// these flags on every on-demand load and model switch.
    mtp_enabled: bool,
    mtp_head_kv_quant: bool,
    mtp_depth: u32,
    llama: LlamaSettings,
    /// Launch-flag ds4 speculative settings, retained for cold loads (same
    /// class as `mtp_enabled` above — `--no-ds4-mtp` / `--dspark` must
    /// survive an on-demand GGUF load, not just the `--model` primary).
    ds4_mtp: bool,
    ds4_dspark: bool,
    ds4_ssd_streaming: bool,
    /// `--ane-prefill`, retained for cold loads (same class as `mtp_enabled`).
    ane_prefill: bool,
    ane: bool,
    ane_slots: u32,
    ane_chunk_resolver: ?*const fn (*ModelConfig) u32,
    ane_headroom_resolver: ?*const fn (*const ModelConfig, u32) u64,
    /// Launch-flag drafter settings, retained for cold loads. `--no-drafter`
    /// became load-bearing on this path the moment `dflash.resolveInDirDrafter`
    /// started probing `<model_dir>/drafter` at load: without it here, a server
    /// launched with speculation off re-enabled it on every model switched to.
    /// `drafter_dir` is the launch `--drafter` path and `primary_model_dir` the
    /// `--model` it belongs to — see `coldLoadDrafterDir` for why it is not
    /// simply copied across.
    no_drafter: bool,
    drafter_dir: []const u8,
    primary_model_dir: []const u8,
    draft_block_size: u32,
    draft_block_size_explicit: bool,

    // ── Borrowed refs (CPU-only state owned by the LoadedModel). ──
    drafter_path: []const u8,

    /// Phase A6 → Plan 05: per-model hot prefix cache. Pre-Plan-05 this was
    /// a server-owned global; Plan 05 moves it onto `LoadedModel` so each
    /// model gets isolation by construction. Borrowed view here is the
    /// current model's cache (or null when the cache isn't applicable for
    /// the model, e.g. hybrid SSM archs).
    hot_prefix_cache: ?*handles.HotPrefixCache,
    /// Resident hot-cache bytes, published for the connection thread: `hot_prefix_cache` is
    /// inference-thread state, freed on every model switch, so the guard reads this number
    /// and never the pointer.
    resident_hot_cache_bytes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    /// KV + recurrent state the decoding slots own, published once per tick (`/props`).
    resident_live_kv_bytes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    /// Per-slot context snapshot for `/metrics.json`, published with the above under `queue_mu`.
    live_sessions: [metrics_mod.MAX_SESSIONS]metrics_mod.Session = undefined,
    live_session_count: usize = 0,
    /// Every ready model's hot-cache entries for `/metrics.json`, under `digest_mu`.
    cached_sessions: [metrics_mod.MAX_SESSIONS]metrics_mod.Session = undefined,
    cached_session_count: usize = 0,

    /// The part of the above an eviction can prove it will return (residency minus the largest entry).
    reclaimable_hot_cache_bytes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    /// Set on unload: the OS hands freed pages back lazily, so the budget revise repeats
    /// before each prefill batch while this is armed and settles as the ceiling recovers.
    budget_revise_sw: ?io_util.Stopwatch = null,
    promise_ledger: PromiseLedger = .{},
    /// Inference thread only. Greater than 0 while a prefill started at a chunk boundary runs: it may
    /// tick decoders at its own boundaries but never admits.
    boundary_depth: u8 = 0,
    /// The prefill whose chunk boundary is hosting a nested admission; counted as company for memory.
    boundary_outer: ?*Slot = null,
    /// The request being prefilled at that boundary; in no queue, so `cancelAllInFlight` reads it here.
    /// Guarded by `queue_mu`.
    boundary_slot: ?*Slot = null,
    /// A running media job's estimated peak, kept off chat admitted between its steps.
    gen_reserve_bytes: u64 = 0,

    /// Per-entry digest snapshot the connection-thread guard reads instead of the cache.
    /// Replaced under `digest_mu` by the inference thread; readers copy under the lock.
    hot_cache_digests: []HotCacheDigest = &.{},
    /// The residency the digests describe, published in the same critical section.
    digest_residency: u64 = 0,
    digest_mu: std.Io.Mutex = .init,

    max_concurrent: u32,
    /// Phase A7 test hook: when true, `runDecodeTick` forces the batched
    /// kernel even at `active.len == 1`. Set via the `MLX_SERVE_FORCE_BATCHED`
    /// environment variable (`=1` to enable). Test-only — production uses
    /// the auto-gate that drops to `runSingleDecodeTick` for single-slot
    /// requests because that path is bit-identical to legacy and supports
    /// speculative decoding (which the batched kernel doesn't).
    force_batched: bool,

    queue_mu: std.Io.Mutex,
    queue_cond: std.Io.Condition,
    pending: std.ArrayList(*Slot),
    decoding: std.ArrayList(*Slot),
    /// Slots an engine beside MLX runs on its own threads (`engines/mlx/runtime/offload.zig`), off the inference thread.
    offloaded: std.ArrayList(*Slot) = .empty,
    /// Cold Neural Engine builds running on their own threads (`aneBuildThread`).
    ane_builds: std.atomic.Value(u32) = .init(0),
    /// Those engines' live metrics; the sampler adds them to the GPU's.
    engine_gauges: engines_served.Gauges = .{},
    /// Phase A4: pending vision-encode requests. The inference thread drains
    /// these in the gap before/after each prefill+decode tick. Posting
    /// broadcasts on `queue_cond` to wake an idle inference thread.
    vision_queue: std.ArrayList(*VisionEncodeRequest),
    /// Pending embedding requests (encoder-only models). Same shape as
    /// vision_queue; serviced inline between decode ticks.
    embed_queue: std.ArrayList(*EmbedRequest),
    /// Phase D: pending cold-load requests. Conn threads post here via
    /// `scheduler.ensureLoaded`; the inference thread drains between ticks
    /// (load runs after cleanup + vision/embed, before prefill). Multiple
    /// concurrent requesters for the same id share one load via the
    /// `.loading` state on the entry — `ensureLoaded` only posts when it
    /// successfully flips the entry from `.unloaded` → `.loading`.
    load_queue: std.ArrayList(*LoadRequest),
    /// Pending media-generation jobs (image/audio/video). Conn threads post
    /// here via `runGeneration`; the inference thread runs each to completion
    /// between ticks (it owns the GPU stream — the sole mlx caller — so gen
    /// MLX ops + SSE writes to the parked connection are single-threaded).
    /// A long gen stalls chat decode for its duration; accepted tradeoff
    /// (single GPU, gen is the user's foreground action).
    gen_queue: std.ArrayList(*GenRequest),
    /// Pending model-unload jobs. Conn threads post here via `unloadModel`
    /// after marking the entry `.evicting` + draining its refcount; the
    /// inference thread frees the mlx state (stream-bound, like cleanup).
    unload_queue: std.ArrayList(*UnloadRequest),
    /// Slots awaiting cleanup. The conn thread queues a slot here in
    /// `complete()` instead of calling `slot.deinit()` directly — `deinit`
    /// frees mlx_arrays via refcount-decrement, and the underlying GPU
    /// memory release races against the inference thread's stream. The
    /// inference thread drains this queue between ticks where it owns the
    /// stream binding, so all mlx ops stay on one thread.
    cleanup_queue: std.ArrayList(*Slot),
    /// Metrics sink. Null when --metrics is off. Populated from LoadParams.
    /// Read once per REQUEST in `finishSlot` — never on the per-token path.
    metrics: ?*metrics_mod.Metrics,
    /// In-flight generated-token aggregate: the sum of `completion_tokens`
    /// over the slots still decoding, republished by the inference thread once
    /// per decode tick (O(1) at the tick boundary, NOT per token). The gauge
    /// sampler reads this race-free to derive a live tok/s — it never touches
    /// per-slot fields off-thread. Zero when nothing is decoding.
    inflight_generated_tokens: std.atomic.Value(u64),
    /// Tokens forwarded so far by the prefill currently running on the
    /// inference thread; 0 when no prefill is in flight. Mirror of
    /// `inflight_generated_tokens` for the OTHER phase — without it the metrics
    /// panel shows nothing while a multi-minute prefill pins the GPU, because
    /// prompt-token counters and prefill-time histograms only advance when the
    /// request finishes. Written per prefill CHUNK, read by the gauge sampler.
    inflight_prefill_tokens: std.atomic.Value(u64),
    /// Post-cache tail the in-flight prefill will forward, same scale as
    /// `inflight_prefill_tokens`; 0 when none is running.
    inflight_prefill_expected: std.atomic.Value(u64),
    /// Number of slots currently inside `runPrefill`. Set on entry, cleared on
    /// every exit — so the panel can say "prefilling" IMMEDIATELY, rather than
    /// waiting for the first 8192-token chunk to land (~40 s on a 27B).
    requests_prefilling: std.atomic.Value(u64),
    /// Counts `pending.len + decoding.len` for back-pressure.
    in_flight: u32,
    /// Capacity for back-pressure. `submit` waits when in_flight >= cap.
    /// `cap = max_concurrent + queue_depth`. queue_depth = 32 hardcoded
    /// (matches the legacy `max_queue_size`).
    queue_cap: u32,
    submit_cond: std.Io.Condition,
    /// Signaled when an engine session (beside MLX) is released in
    /// `complete()`, waking a `submit()` blocked waiting to claim it. Guarded by
    /// `queue_mu` together with `LoadedModel.session_claims`.
    session_cond: std.Io.Condition,

    inference_thread: ?std.Thread,
    /// Set as the inference thread returns: `deinit` joins only a thread that has.
    inference_done: std.atomic.Value(bool) = .init(false),
    shutdown: std.atomic.Value(bool),
    /// The shutdown flush's deadline on the `nowMsMonotonic` clock; 0 = off. Set before `shutdown`.
    shutdown_deadline_ms: std.atomic.Value(i64) = .init(0),
    /// The batch the inference thread took out of `pending` and is prefilling (`queue_mu`): in
    /// neither queue, so `cancelAllInFlight` reads it here or SIGTERM waits out the whole prefill.
    /// Live entries are `[prefilling_from, n_prefilling)`; each holds `in_pass` while listed.
    prefilling: [16]*Slot = undefined,
    prefilling_from: usize = 0,
    n_prefilling: usize = 0,
    started: std.atomic.Value(bool),
    started_mu: std.Io.Mutex,
    started_cond: std.Io.Condition,

    /// Set true by the inference thread if model load fails. Read by `init`
    /// after `started` is signaled to decide whether to surface a load error.
    load_failed: std.atomic.Value(bool),
    /// Owned, dupe'd error name (e.g. "MissingVisionWeights"). null on
    /// success. Freed in `deinit`.
    load_error_name: ?[]const u8,

    // ── calls into the engine (`host/engines/mlx.zig`)

    /// Stops the GPU thread and frees every request and model it holds.
    pub fn deinit(self: *Scheduler) void {
        mlx_engine.api().stop(self);
    }

    pub fn cancelAllInFlight(self: *Scheduler) void {
        mlx_engine.api().cancel_all(self);
    }

    /// Another request is queued, prefilling or decoding right now.
    pub fn hasCompany(self: *Scheduler) bool {
        return mlx_engine.api().has_company(self);
    }

    /// The model `id` (empty: the default), loaded and held for the caller until `release`.
    pub fn ensureLoaded(self: *Scheduler, id: []const u8) !*LoadedModel {
        return mlx_engine.ensureLoaded(self, id);
    }

    pub fn release(self: *Scheduler, lm: *LoadedModel) void {
        mlx_engine.api().release(self, lm);
    }

    /// Releases a hold taken only to read the model's status.
    pub fn releaseStatus(self: *Scheduler, lm: *LoadedModel) void {
        mlx_engine.api().release_status(self, lm);
    }

    pub fn loadErrorName(self: *Scheduler, alloc: std.mem.Allocator, id: []const u8) ?[]u8 {
        return mlx_engine.api().load_error_name(self, alloc, id);
    }

    pub fn unloadModel(self: *Scheduler, id: []const u8) !void {
        return mlx_engine.unloadModel(self, id);
    }

    pub fn unloadModelIfIdle(self: *Scheduler, id: []const u8, window_ms: ?i64) !void {
        return mlx_engine.unloadModelIfIdle(self, id, window_ms);
    }

    /// Runs a media job on the GPU thread; returns when it is done.
    pub fn runGeneration(self: *Scheduler, req: *GenRequest) !void {
        return mlx_engine.runGeneration(self, req);
    }

    pub fn computeEmbeddings(self: *Scheduler, req: *EmbedRequest) ![][]f32 {
        return mlx_engine.computeEmbeddings(self, req);
    }

    pub fn encodeVision(self: *Scheduler, req: *VisionEncodeRequest) !mlx.mlx_array {
        return mlx_engine.encodeVision(self, req);
    }
};
