//! Load, media-gen and unload work items posted to the inference thread, and media-job merging.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kv_quant = @import("../../sdk/blocks/attention/kv_quant.zig");
const tokenizer_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const model_mod = @import("../../sdk/config/model.zig");
const chat_mod = @import("../chat/chat.zig");
const model_registry_mod = @import("../registry/registry.zig");
const ModelConfig = model_mod.ModelConfig;
const Tokenizer = tokenizer_mod.Tokenizer;
const ChatConfig = chat_mod.ChatConfig;
const LoadedModel = model_registry_mod.LoadedModel;
const testing = std.testing;

const LlamaSettings = @import("../../sdk/quant/gguf/llama_settings.zig").Settings;
const GgufRoute = @import("../../sdk/quant/gguf/engine_pick.zig").Route;

/// Plan 05 Phase D: cold-load work item. Posted by `Scheduler.ensureLoaded`
/// when a request targets an `.unloaded` (or freshly-evicted) entry; the
/// inference thread drains the queue between ticks. The conn thread parses
/// `config.json` / tokenizer / chat_config on its own thread (CPU only,
/// no mlx ops) and hands the pre-parsed CPU state to the inference thread,
/// which does the mlx-allocating work (weights + Transformer + vision +
/// drafter + JIT + warmup) and installs everything on `entry`.
///
/// Eviction: when set, `evict_entry` is unloaded on the inference thread
/// BEFORE the new load starts. The conn thread has already marked the
/// victim `.evicting` and waited for refcount == 0, so freeing GPU memory
/// is safe.
pub const LoadRequest = struct {
    /// Target entry to populate. Already transitioned to `.loading` by
    /// the conn thread before posting; the inference thread completes the
    /// load and calls `registry.markReadyLocked(entry, bytes)` or
    /// `markErrorLocked` on failure.
    entry: *LoadedModel,
    /// Pre-parsed CPU state. Ownership transfers to `entry` on success;
    /// on failure the conn thread takes them back via the `done` cond-var
    /// and frees them.
    config: *ModelConfig,
    tok: *Tokenizer,
    chat_config: *ChatConfig,

    /// Borrowed paths. Conn thread keeps the buffers alive until `done`.
    model_dir: []const u8,
    resident_model_bytes: ?u64 = null,
    drafter_dir: []const u8 = "",
    /// `--no-drafter`: never load a drafter, including one MERGED into the
    /// checkpoint. `drafter_dir == ""` stopped meaning "off" the moment a
    /// checkpoint could carry its own, so the opt-out needs its own bit.
    no_drafter: bool = false,
    /// SSD weight-streaming for cold-loaded ds4 models (issue #39). The CLI
    /// startup path supplies this via LoadParams; cold-load defaults it off.
    ds4_ssd_streaming: bool = false,
    /// Auto-load the ds4 MTP draft head beside a cold-loaded ds4 model. Default
    /// on (a switched-to ds4 model gets speculative decode); forced off under
    /// `ds4_ssd_streaming`.
    ds4_mtp: bool = true,
    /// DSpark runtime for a cold-loaded ds4 model whose sidecar carries
    /// DSpark stages. Cold loads inherit the launch flag via the Scheduler's
    /// `ds4_dspark` (the headless flag-eater class — a LoadRequest default
    /// would silently drop `--dspark` on every on-demand GGUF load).
    ds4_dspark: bool = false,
    /// Auto-load the Qwen native MTP sidecar when the model dir ships one.
    mtp_enabled: bool = true,
    mtp_head_kv_quant: bool = false,
    /// Max MTP draft depth (CLI --mtp-max-depth; 0 = auto, resolved by
    /// generate_mod.resolveMtpDepthCap at load/Generator init).
    mtp_depth: u32 = 0,
    /// `--ane-prefill` survives cold loads (the flag-eater class).
    ane_prefill: bool = false,
    ane: bool = false,
    ane_slots: u32 = 1,
    ane_chunk_resolver: ?*const fn (*model_mod.ModelConfig) u32 = null,
    ane_headroom_resolver: ?*const fn (*const model_mod.ModelConfig, u32) u64 = null,

    load_vision: bool = false,
    warmup_eager: bool = true,
    draft_block_size: u32 = 4,
    draft_block_size_explicit: bool = false,
    kv_quant_config: kv_quant.KVQuantConfig = kv_quant.KVQuantConfig.dense,
    prefix_cache_capacity: u32 = 1,
    prefix_cache_ram_enabled: bool = true,
    prefix_cache_mem_bytes: u64 = 0,
    /// SSD tier byte budget (mirrors `LoadParams.prefix_cache_disk_bytes`).
    prefix_cache_disk_bytes: u64 = 0,
    /// Phase 1 (perf-plan): SSM/conv state snapshot stride during prefill.
    /// Zero disables (hybrid models bypass the hot prefix cache, as before).
    /// Non-zero enables multi-turn warm reuse on hybrid SSM archs. Plumbed
    /// to `HotPrefixCache.shouldUse(enable_ssm_checkpoints = stride > 0)`
    /// and to every `Generator.initWithOptions` call so the prefill loop
    /// captures snapshots.
    ssm_checkpoint_stride: u32 = 0,
    /// Phase 1: maximum checkpoints retained per request. Older ones are
    /// dropped front-first when the buffer would grow past this. 0 = no cap
    /// beyond the prefix-cache byte budget.
    ssm_checkpoint_max: u32 = 16,
    /// Iteration 2: tokenize cache LRU capacity. Mirrored on
    /// `LoadParams.tokenize_cache_entries`; both paths feed
    /// `doLoadOnInferenceThread`.
    tokenize_cache_entries: u32 = 4,
    /// Mirrors `LoadParams.llama`.
    llama: LlamaSettings = .{},

    /// Victims to evict before the load (LRU-selected by the planner). Each is
    /// already marked `.evicting` with refcount == 0 by the conn thread under
    /// registry.mutex. The inference thread calls `unloadResident()` on each to
    /// free GPU memory, drops its resident-bytes accounting via
    /// `registry.accountEvictedLocked`, then `registry.finalizeEvictionLocked`.
    /// Borrows the conn thread's stack buffer; valid until `done`.
    evict_entries: []*LoadedModel = &.{},
    /// The Neural Engine refused the model (`aneAdmit`, `aneBuild`): the GPU loads it.
    ane_declined: bool = false,

    /// Output: error name on failure (owned by `allocator`; conn thread
    /// frees). Null on success.
    error_name: ?[]const u8 = null,

    /// Conn-thread synchronization. Inference thread broadcasts when done.
    done: bool = false,
    allocator: std.mem.Allocator,
    done_mu: std.Io.Mutex = .init,
    done_cond: std.Io.Condition = .init,

    /// Mirrors `LoadParams.gguf`: set by `ensureLoaded` when the entry's path resolves to a GGUF (issue #59).
    /// Borrowed from the conn thread until `done`.
    gguf: ?GgufRoute = null,
};

/// Media-generation work item. Posted by `runGeneration` from a connection
/// thread; the inference thread invokes `run(ctx)` between ticks so all mlx
/// ops AND the SSE writes to the (parked) connection happen on the GPU-stream-
/// owning thread. Decoupled via an opaque ctx + runner so this module needs
/// no dependency on `server.zig`/`gen.zig` — `server.zig` owns the job body.
pub const GenRequest = struct {
    /// Opaque job payload (a `*GenJob` in server.zig).
    ctx: *anyopaque,
    /// Runs the job body on the inference thread. Must not return an error
    /// (it writes any failure into its own response/SSE).
    run: *const fn (ctx: *anyopaque) void,
    /// The media model. `gen_busy` is set/cleared around the run for
    /// visibility; the conn thread's refcount already pins it against eviction.
    model: *model_registry_mod.LoadedModel,
    /// A decision job: its buffers are reused by the next request (`shouldClearCache`).
    decision: bool = false,
    /// Set on jobs that can share one pass (decision requests): jobs queued
    /// back to back for the same model and `run_many` run as one call.
    merge: ?Merge = null,
    /// This job's own MLX peak when the request moves it (a 1536 cascade peaks 12 GB above a 1024 one);
    /// 0 bills the model's resident estimate.
    peak_bytes: u64 = 0,
    done: bool = false,
    done_mu: std.Io.Mutex = .init,
    done_cond: std.Io.Condition = .init,
};

pub const Merge = struct {
    run_many: *const fn (ctxs: []const *anyopaque) void,
    /// This job's share of `MAX_MERGED_WEIGHT` (its question count, at least 1).
    weight: usize,
    /// How long the inference thread waits for more jobs to merge; 0 merges
    /// only the jobs already queued.
    window_us: u32 = 0,
};

/// Bound on the work merged into one pass, so one tick cannot hold the
/// inference thread for long; the first job always runs.
pub const MAX_MERGED_WEIGHT: usize = 256;

/// Move the jobs at the head of `queue` that can run with `batch[0]` (same
/// model, same `run_many`) into `batch[n..]` while the merged weight fits.
/// Only a contiguous run merges: other gen jobs keep their place in the queue.
/// Returns the new count.
pub fn takeMergeable(queue: *std.ArrayList(*GenRequest), batch: []*GenRequest, n: usize) usize {
    const m = batch[0].merge orelse return n;
    var weight: usize = 0;
    for (batch[0..n]) |r| weight += r.merge.?.weight;
    var count = n;
    while (count < batch.len and queue.items.len > 0) {
        const next = queue.items[0];
        const nm = next.merge orelse break;
        if (next.model != batch[0].model or nm.run_many != m.run_many or weight + nm.weight > MAX_MERGED_WEIGHT) break;
        weight += nm.weight;
        batch[count] = queue.orderedRemove(0);
        count += 1;
    }
    return count;
}

/// Model-unload work item. Posted by `unloadModel` after the conn thread
/// marked the entry `.evicting` and drained its refcount. The inference thread
/// frees the entry's resident mlx state (stream-bound) and finalizes the
/// eviction accounting, then the entry returns to `.unloaded` (the stub stays
/// in the registry so it can reload later).
pub const UnloadRequest = struct {
    entry: *model_registry_mod.LoadedModel,
    done: bool = false,
    done_mu: std.Io.Mutex = .init,
    done_cond: std.Io.Condition = .init,
};

// ── tests ──

test "takeMergeable: merges the contiguous run of same-model decision jobs within the weight cap" {
    const a = testing.allocator;
    const noop = struct {
        fn run(_: *anyopaque) void {}
        // Distinct bodies: identical ones may be folded into one address.
        var calls: usize = 0;
        fn many(_: []const *anyopaque) void {
            calls += 1;
        }
        fn other(_: []const *anyopaque) void {
            calls += 2;
        }
    };
    var dummy: u8 = 0;
    const m1: *model_registry_mod.LoadedModel = @ptrFromInt(0x1000);
    const m2: *model_registry_mod.LoadedModel = @ptrFromInt(0x2000);
    const dec = struct {
        fn req(ctx: *anyopaque, model: *model_registry_mod.LoadedModel, weight: usize, many: *const fn ([]const *anyopaque) void) GenRequest {
            return .{ .ctx = ctx, .run = noop.run, .model = model, .merge = .{ .run_many = many, .weight = weight } };
        }
    };
    var r = [_]GenRequest{
        dec.req(&dummy, m1, 3, noop.many),
        dec.req(&dummy, m1, 3, noop.many),
        dec.req(&dummy, m1, 200, noop.many),
        dec.req(&dummy, m1, 60, noop.many), // over 256 with the three before
        dec.req(&dummy, m2, 3, noop.many),
    };
    var media: GenRequest = .{ .ctx = &dummy, .run = noop.run, .model = m1 };
    var q: std.ArrayList(*GenRequest) = .empty;
    defer q.deinit(a);
    for (r[1..]) |*x| try q.append(a, x);
    var batch: [MAX_MERGED_WEIGHT]*GenRequest = undefined;
    batch[0] = &r[0];
    try testing.expectEqual(@as(usize, 3), takeMergeable(&q, &batch, 1));
    try testing.expectEqual(&r[2], batch[2]);
    try testing.expectEqual(&r[3], q.items[0]); // stays queued for the next tick

    // A media job ahead of a decision job keeps its place; nothing jumps it.
    q.clearRetainingCapacity();
    try q.append(a, &media);
    try q.append(a, &r[1]);
    try testing.expectEqual(@as(usize, 1), takeMergeable(&q, &batch, 1));
    try testing.expectEqual(@as(usize, 2), q.items.len);
    // Another model or another runner does not merge; a media job merges nothing.
    q.clearRetainingCapacity();
    try q.append(a, &r[4]);
    try testing.expectEqual(@as(usize, 1), takeMergeable(&q, &batch, 1));
    var foreign = dec.req(&dummy, m1, 1, noop.other);
    q.clearRetainingCapacity();
    try q.append(a, &foreign);
    try testing.expectEqual(@as(usize, 1), takeMergeable(&q, &batch, 1));
    batch[0] = &media;
    try q.append(a, &r[1]);
    try testing.expectEqual(@as(usize, 1), takeMergeable(&q, &batch, 1));
}

/// Phase A4: pixel data for a single image, decoded by the connection
/// thread (CPU only — stb_image / libwebp). The inference thread wraps this
/// in an `mlx_array` via `mlx_array_new_data` and runs the vision encoder.
pub const VisionImagePixels = struct {
    /// Raw bytes holding float32 pixel data. Gemma: CHW (3 × H × W × 4). Qwen3-VL:
    /// merge-order pixel_values (N × C·tps·ps·ps × 4). Borrowed; must outlive the
    /// encodeVision call (which blocks until completion).
    pixels: []const u8,
    width: u32,
    height: u32,
    /// Qwen3-VL only: full patch grid (0 ⇒ Gemma CHW). Selects QwenVision.
    grid_h: u32 = 0,
    grid_w: u32 = 0,
};

/// Qwen3-VL video: pre-patchified pixel_values for ALL `grid_t` temporal-patch
/// groups, concatenated (see `qwen_vision.buildPixelValuesVideo`). Borrowed;
/// must outlive the encodeVision call.
pub const VisionVideoPixels = struct {
    pixels: []const u8,
    grid_t: u32,
    grid_h: u32,
    grid_w: u32,
    /// A plugin tower's frames: the rate the client sampled them at (0 = not given).
    fps: f32 = 0,
};

/// One encoder input. `audio` is raw float32-LE 16 kHz mono PCM, framed into
/// `audio_samples_per_token` tokens by the unified audio embedder.
pub const VisionItem = union(enum) {
    image: VisionImagePixels,
    video: VisionVideoPixels,
    audio: []const u8,
};

/// Phase A4: vision-encode work item. Conn thread fills `images` (raw pixel
/// data, CPU-only) and calls `Scheduler.encodeVision`, which posts the
/// request and blocks until the inference thread fills `result` and signals
/// `done`. Ownership of `result` transfers to the caller on success — pass
/// to `scheduler.submit(.{ .vision_embeddings = arr, ... })` and the slot's
/// `deinit` will free it.
pub const VisionEncodeRequest = struct {
    /// Plan 05 Phase D: target model whose `vision_encoder` services this
    /// request. The conn thread holds a refcount (via `ensureLoaded`) for
    /// the duration of the call.
    model: *LoadedModel,
    /// Encoder inputs in prompt order. Borrowed; must outlive the call.
    items: []const VisionItem,
    /// Each item's pixel/PCM hash, parallel to `items`: the encoder cache key.
    item_keys: []const u64,
    /// Output: soft tokens each item produced, parallel to `items`. Caller-owned.
    item_tokens: []usize,
    /// Output: every item's soft tokens, concatenated along the token axis in
    /// prompt order (the splice scatters them positionally). Ownership
    /// transfers to the caller.
    result: ?mlx.mlx_array = null,
    /// Output: error name on failure. Owned by `allocator`; caller frees.
    error_name: ?[]const u8 = null,
    /// RADIO video only, parallel to `items`. A mask is the full pre-EVS row
    /// count (true = keep). Null entries are not video. Owned by `allocator`
    /// on success; the caller frees each mask and the array.
    item_keep: ?[]?[]bool = null,
    /// Plugin towers only, parallel to `items` and allocated by the caller: each item's placeholder run, filled
    /// with `allocator`; the caller frees each run.
    item_runs: [][]u32 = &.{},
    /// Done flag (under done_mu). Caller's wait-loop drains the cond when
    /// this flips true.
    done: bool = false,
    allocator: std.mem.Allocator,
    done_mu: std.Io.Mutex = .init,
    done_cond: std.Io.Condition = .init,
};

/// Phase: embedding work item for encoder-only models. Conn thread fills
/// `token_seqs` and calls `Scheduler.computeEmbeddings`; the inference
/// thread services the request via `generate.computeEmbeddingsBatch` — one
/// padded, key-masked GPU forward per EMBED_MAX_BATCH chunk — and writes
/// the float vectors into `results` (caller frees). Mirrors the
/// VisionEncodeRequest pattern.
pub const EmbedRequest = struct {
    /// Plan 05 Phase D: target model whose `transformer` services this
    /// request. The conn thread holds a refcount for the duration.
    model: *LoadedModel,
    /// Tokenized inputs, one slice per text. Borrowed; must outlive the call.
    token_seqs: []const []const u32,
    /// The soft-token rows [1, n, hidden] of the ONE prompt's images and video, in prompt order. The request
    /// owns them: the inference thread frees them when the forward is done.
    vision_embeddings: ?mlx.mlx_array = null,
    /// Output: one pooled L2-normalized embedding per input on success.
    /// Rows + outer slice owned by `allocator`; caller frees.
    results: ?[][]f32 = null,
    /// Output: error name on failure. Owned by `allocator`; caller frees.
    error_name: ?[]const u8 = null,
    done: bool = false,
    allocator: std.mem.Allocator,
    done_mu: std.Io.Mutex = .init,
    done_cond: std.Io.Condition = .init,
};
