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
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const tokenizer_mod = @import("tokenizer.zig");
const generate_mod = @import("generate.zig");
const gen_mod = @import("gen.zig");
const drafter_mod = @import("drafter.zig");
const mtp_mod = @import("mtp.zig");
const diffusion_mod = @import("diffusion.zig");
const model_mod = @import("model.zig");
const vision_mod = @import("vision.zig");
const chat_mod = @import("chat.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const metrics_mod = @import("metrics.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const tokenize_cache_mod = @import("tokenize_cache.zig");
const model_registry_mod = @import("model_registry.zig");
const model_discovery = @import("model_discovery.zig");
const gguf_meta = @import("gguf_meta.zig");
const arch_ds4 = if (@import("build_options").ios) @import("arch/ds4_stub.zig") else @import("arch/ds4.zig");
const arch_llama = if (@import("build_options").ios) @import("arch/llama_stub.zig") else @import("arch/llama.zig");
const log = @import("log.zig");
const io_util = @import("io_util.zig");
const status = @import("status.zig");

const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;
const SSMCacheEntry = transformer_mod.SSMCacheEntry;
const ForwardCtx = transformer_mod.ForwardCtx;
const ModelConfig = model_mod.ModelConfig;
const Tokenizer = tokenizer_mod.Tokenizer;
const Generator = generate_mod.Generator;
const SamplingParams = generate_mod.SamplingParams;
const DrafterModel = drafter_mod.DrafterModel;
const VisionEncoder = vision_mod.VisionEncoder;
const Weights = model_mod.Weights;
const ChatConfig = chat_mod.ChatConfig;
const ModelRegistry = model_registry_mod.ModelRegistry;
const LoadedModel = model_registry_mod.LoadedModel;

/// Phase A1: model-load plan executed on the scheduler's inference thread.
///
/// mlx 0.31.2 uses thread-local GPU streams: every `mlx_*` op binds to the
/// stream of the calling thread, and JIT-compiled closures are tied to that
/// stream too. If main loads the model and the scheduler later calls forward,
/// mlx aborts with "no Stream(gpu, N) in current thread". Solution: the
/// scheduler's inference thread does the load itself so the stream is bound
/// on the right thread from t0.
///
/// CPU-only state (parsed config, tokenizer, chat config) is loaded by main
/// and passed in by reference. The mlx-allocating pieces (weights tensors,
/// Transformer, vision encoder, drafter, JIT compile, warmup) all run on the
/// inference thread before `init()` returns.
pub const LoadParams = struct {
    /// Registry that owns the entry to populate + provides snapshot/eviction
    /// bookkeeping. Outlives the scheduler.
    registry: *ModelRegistry,
    /// The (pre-registered) LoadedModel stub that will be promoted to
    /// `.ready` by the inference-thread load. Holds id/path on entry;
    /// `loadModelOnInferenceThread` installs weights/transformer/vision/
    /// drafter/tokenizer/chat_config/config into this slot. Lifetime
    /// matches the registry.
    entry: *LoadedModel,
    /// Heap-allocated parsed config. Ownership transfers to `entry` on
    /// successful load; caller must NOT deinit/free externally after
    /// `Scheduler.init` returns.
    config: *ModelConfig,
    /// Heap-allocated tokenizer. Ownership transfers to `entry`.
    tok: *Tokenizer,
    /// Heap-allocated chat config. Ownership transfers to `entry`.
    chat_config: *ChatConfig,
    /// Path to the model directory. Borrowed; outlive scheduler.
    model_dir: []const u8,
    /// Path to the assistant drafter checkpoint. Empty disables the drafter.
    /// Borrowed; outlive scheduler.
    drafter_dir: []const u8 = "",
    /// Auto-load the Qwen native MTP sidecar when the model dir ships one.
    mtp_enabled: bool = true,
    /// Max MTP draft depth (CLI --mtp-depth; 0 = auto, resolved by
    /// generate_mod.resolveMtpDepthCap at load/Generator init).
    mtp_depth: u32 = 0,
    /// Whether to also load vision-tower weights. Combined with
    /// `config.has_vision` — false here disables vision regardless of config.
    load_vision: bool = false,
    /// Eager warmup: fault weight pages + run a tiny forward to JIT-compile
    /// the decode path on the inference thread. Adds ~600-900 ms at boot but
    /// keeps the first user request fast.
    warmup_eager: bool = true,
    /// Drafter block size (caller computed via `drafter.recommendedBlockSize`).
    /// Ignored when `drafter_dir` is empty.
    draft_block_size: u32 = 4,
    /// Whether the user passed --draft-block-size explicitly (used for
    /// human-readable startup logging). Ignored when `drafter_dir` is empty.
    draft_block_size_explicit: bool = false,
    /// KV-cache storage backend. Defaults to dense bf16; user opts into
    /// 4/8-bit affine quantization via `--kv-quant {4,8}`. Stored on every
    /// per-slot KVCache and consulted at every read/write boundary.
    kv_quant_config: transformer_mod.KVQuantConfig = transformer_mod.KVQuantConfig.dense,
    /// Per-model hot prefix cache capacity (count). 0 disables.
    prefix_cache_capacity: u32 = 1,
    /// Per-model hot prefix cache KV-bytes budget. 0 disables the byte cap.
    prefix_cache_mem_bytes: u64 = 0,
    /// SSD tier byte budget for the hot prefix cache (`--prefix-cache-disk`).
    /// 0 disables persistence. Attached per model at load for pure-attention
    /// archs; entries live under `~/.mlx-serve/kv-cache/<fingerprint>`.
    prefix_cache_disk_bytes: u64 = 0,
    /// Phase 1 (perf-plan): SSM/conv state snapshot stride during prefill.
    /// 0 = disabled (hybrid models bypass the hot prefix cache). Non-zero
    /// enables hybrid in `HotPrefixCache.shouldUse` and triggers per-stride
    /// snapshots in the Generator's prefill loop. Default 0 here so callers
    /// that don't set it (legacy paths) preserve pre-Phase-1 behavior;
    /// `main.zig` overrides via `--ssm-checkpoint-stride` for the serve path.
    ssm_checkpoint_stride: u32 = 0,
    /// Phase 1: cap on snapshots retained per request.
    ssm_checkpoint_max: u32 = 32,
    /// Iteration 2 (perf-plan Phase 4 #3): per-LoadedModel LRU cache
    /// of chat-template render+tokenize results. 0 disables the cache
    /// (useful for ablation benches / debugging). Default 4 matches
    /// `prefix_cache_capacity` — most warm-reuse benches exercise a
    /// handful of repeated prompts, and full chat conversations bump
    /// this counter anyway via LRU as new turns arrive.
    tokenize_cache_entries: u32 = 4,
    /// Iteration 3-5 (perf-plan Phase 5 #1): maximum resident llama.cpp
    /// sessions per model. 1 = legacy single-session behavior (every
    /// llama prefill fights one KV slot). > 1 keeps the N
    /// most-recently-used prompts hot in independent contexts so
    /// alternating multi-doc agent loads don't cold-prefill every flip.
    llama_cache_entries: u32 = 4,
    /// Phase 5 #2: ggml types for the embedded llama.cpp KV cache.
    /// 0 = libllama default (F16); other values match `ggml_type` enum
    /// (Q8_0=8, Q4_0=2). Wired through `Scheduler.doLoadOnInferenceThread`
    /// to the LoadedModel; consumed at first request when the session is
    /// created in `runPrefillLlama`.
    llama_kv_type_k: i32 = 0,
    llama_kv_type_v: i32 = 0,
    /// When non-empty, the load routes through the embedded ds4 engine
    /// instead of the MLX safetensors path. `model_dir` is expected to point
    /// at a `.gguf` file (or a directory containing one); the inference
    /// thread opens a `Ds4Engine` and installs it on the entry's
    /// `ds4_engine` field. `config`/`tok`/`chat_config` are stubs (the
    /// embedded engine owns the real tokenizer + chat template); they're
    /// still moved onto the entry so server-side reads of `lm.config.?`
    /// (e.g. `eosTokenSlice`, `getEffectiveContextLength`) keep working.
    ds4_path: []const u8 = "",
    /// SSD weight-streaming for the ds4 engine (issue #39): stream experts from
    /// disk instead of requiring the full model resident in RAM.
    ds4_ssd_streaming: bool = false,
    /// Auto-load the ds4 MTP draft head (found beside the model) for speculative
    /// decode. Default on; forced off when `ds4_ssd_streaming` (ds4 refuses the
    /// combination). `--no-ds4-mtp` disables it.
    ds4_mtp: bool = true,
    /// Like `ds4_path` but for the generic llama.cpp engine (any GGUF except
    /// DeepSeek-V4-Flash). The inference thread opens a `LlamaEngine` and
    /// installs it on the entry's `llama_engine` field. Mutually exclusive with
    /// `ds4_path` and the MLX safetensors path.
    llama_path: []const u8 = "",
    /// Headless boot: skip the startup load entirely and run the inference
    /// loop idle. The registry holds discovery stubs (or nothing); the first
    /// model — chat or media — loads on demand via `/v1/load-model`. `entry`/
    /// `config`/`tok`/`chat_config` are still required (they seed the
    /// scheduler's borrowed-view fields) but are never installed on an entry.
    no_initial_load: bool = false,
    /// Optional metrics sink. Null when --metrics is off (the default).
    /// Stored on the Scheduler and read by the `finishSlot` per-request funnel.
    metrics: ?*metrics_mod.Metrics = null,
    /// The --ctx-size launch flag (0 = unset). Cold-loaded GGUF entries have
    /// no config.json to size their context from, so `preloadCpuState` sizes
    /// the stub config with this — same rule as the startup GGUF paths
    /// (llama: ctx or 8192; ds4: clampSessionCtx). MLX cold loads read their
    /// own config.json and ignore it.
    ctx_size: u32 = 0,
};

/// Submit-time parameters. `prompt_ids` and `eos_token_ids` are duped into the
/// slot so callers can free their copies immediately. `vision_embeddings`
/// ownership transfers into the slot when non-null (the slot will free on
/// deinit).
pub const SubmitParams = struct {
    prompt_ids: []const u32,
    /// Full original prompt for PLD lookup. When null, defaults to
    /// `prompt_ids` (PLD's lookup table = full_prompt + generated).
    full_prompt: ?[]const u32 = null,
    cached_tokens: u32 = 0,
    has_tools: bool = false,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    max_tokens: u32,
    timeout_ns: u64 = 0,
    enable_pld: bool = false,
    enable_drafter: bool = false,
    drafter: ?*DrafterModel = null,
    drafter_block_size: u32 = 4,
    enable_mtp: bool = false,
    mtp: ?*mtp_mod.MtpModel = null,
    /// 0 = auto (see generate_mod.resolveMtpDepthCap).
    mtp_depth: u32 = 0,
    pld_draft_len: u32 = 5,
    pld_key_len: u32 = 3,
    /// Phase 2 (Plan ricky): route SDPA through `kv_quant.quantAttention`
    /// instead of dequant + dense SDPA. No effect when the cache scheme
    /// isn't `.affine`. Default false → unchanged behavior.
    kv_attn_fused: bool = false,
    /// Vision embeddings spliced at image-token positions during prefill.
    /// Ownership transferred to the slot; freed on slot.deinit.
    vision_embeddings: ?mlx.mlx_array = null,
    /// Qwen3-VL interleaved M-RoPE: server-computed flat [3 × mrope_total] i32
    /// position-id table + decode delta. Ownership of `mrope_pos` transfers to
    /// the slot; freed on slot.deinit. Null for non-image / non-Qwen requests.
    mrope_pos: ?[]const i32 = null,
    mrope_total: usize = 0,
    mrope_delta: i32 = 0,
    logprobs_n: u32 = 0,
    /// Wave 1.A: per-request override of the process-default KV-cache quant
    /// scheme. When non-null, this slot's KVCache is constructed with this
    /// config instead of `Scheduler.kv_quant_config`. Lets one server host a
    /// single model and let clients trade accuracy for context length on a
    /// per-call basis (`{"kv_quant": "off"|4|8}` body field).
    kv_quant_config: ?transformer_mod.KVQuantConfig = null,
    /// Plan 05 Phase D: the target model for this request. The conn thread
    /// resolves this via `scheduler.ensureLoaded(id)` BEFORE submitting and
    /// keeps a refcount on it for the slot's lifetime, so the model can't
    /// be evicted mid-flight. The scheduler routes prefill/decode through
    /// `slot.model.transformer.?` (and friends) instead of `sch.xfm`, so
    /// per-tick model switching is just a pointer hop. Required field
    /// post-Phase-D; tests using the legacy path pass the default model.
    model: *model_registry_mod.LoadedModel,
};

pub const SlotState = enum { pending_prefill, decoding, finished, errored };

/// Result of `Slot.waitNext`. Driven by the inference thread; consumed by
/// the connection thread.
pub const NextResult = union(enum) {
    /// Next decoded token id.
    token: u32,
    /// Generation completed (EOS, max_tokens, timeout, etc.). `finish_reason`
    /// is set on the slot at this point.
    done: void,
    /// Generation errored. `error_code` is set on the slot; the caller
    /// should surface it to the client and call `complete(slot)`.
    err: void,
};

/// Per-request state. Owned by the Scheduler from `submit` until `complete`.
pub const Slot = struct {
    allocator: std.mem.Allocator,
    /// io reference for Stopwatch / async-eval (captured from Scheduler).
    io: std.Io,

    /// Plan 05 Phase D: target model for this request. Captured from
    /// `SubmitParams.model` and borrowed for the slot's lifetime; the conn
    /// thread holds a refcount (via `scheduler.ensureLoaded`) so the
    /// pointer stays valid until `complete()`. Prefill/decode route forward
    /// passes through `model.transformer.?` (and `model.vision_encoder`,
    /// `model.drafter`, `model.prefix_cache`). Batched decode groups slots
    /// by this field so kernels never cross model boundaries.
    model: *model_registry_mod.LoadedModel,

    // ── Per-slot model state. Owned by the slot. ──
    cache: KVCache,
    moe_seq_offset: usize,
    ssm_entries: ?[]SSMCacheEntry,
    vision_embeddings: ?mlx.mlx_array,
    /// Qwen3-VL M-RoPE position-id table (flat [3 × mrope_total]) + decode delta.
    /// Owned by the slot; `mrope_pos` freed on deinit.
    mrope_pos: ?[]const i32,
    mrope_total: usize,
    mrope_delta: i32,

    /// Forward context backed by the fields above. Initialized in `init`
    /// and aliased by the Generator's own `ctx` field at prefill time.
    ctx: ForwardCtx,

    /// Generator (constructed on inference thread post-prefill).
    legacy_gen: ?Generator,

    /// Ds4 session for this slot. Created in `runPrefillDs4` when the slot's
    /// model is `.ds4_engine`-backed; freed in `Slot.deinit`. Mutually
    /// exclusive with `legacy_gen` (the MLX `Generator` path).
    ds4_session: ?*arch_ds4.Ds4Session = null,
    /// Per-request RNG state for ds4 sampling. ds4's sampler takes the seed
    /// by pointer so we keep it on the slot.
    ds4_rng: u64 = 0,

    /// llama.cpp session for this slot. BORROWED from the slot's
    /// `model.llama_session` (a persistent per-model context reused across
    /// requests for prompt-prefix KV reuse) — NOT owned, so `Slot.deinit` must
    /// not free it. Mutually exclusive with `legacy_gen` and `ds4_session`.
    llama_session: ?*arch_llama.LlamaSession = null,
    /// DiffusionGemma canvas-denoising runner. Created in
    /// `runPrefillDiffusion` for `config.isDiffusion()` models; owns the
    /// dequantized embedding table; freed in `Slot.deinit`. Mutually
    /// exclusive with `legacy_gen` (the autoregressive MLX path).
    diffusion: ?*diffusion_mod.Runner = null,
    /// True when this slot claimed `model.llama_session_busy` in `submit`. The
    /// single persistent context serves one request at a time; the claim is
    /// released in `complete()`. Tracked per-slot so only the holder releases.
    llama_holds_session: bool = false,
    /// Per-request RNG state for llama.cpp sampling (passed by pointer, like ds4).
    llama_rng: u64 = 0,

    // ── Submission data. Owned by the slot, freed in deinit. ──
    prompt_ids: []u32,
    full_prompt: []u32,
    sampling: SamplingParams,
    eos_token_ids: []u32,
    max_tokens: u32,
    timeout_ns: u64,
    has_tools: bool,
    enable_pld: bool,
    enable_drafter: bool,
    drafter: ?*DrafterModel,
    drafter_block_size: u32,
    enable_mtp: bool,
    mtp: ?*mtp_mod.MtpModel,
    mtp_depth: u32,
    pld_draft_len: u32,
    pld_key_len: u32,
    /// Phase 2 (Plan ricky): see SubmitParams.kv_attn_fused.
    kv_attn_fused: bool,
    cached_tokens: u32,
    logprobs_n: u32,

    // ── State + output channel. ──
    state: SlotState,

    out_mu: std.Io.Mutex,
    out_cond: std.Io.Condition,
    /// Wake signal for `waitNextTimeout` — set alongside every `out_cond`
    /// broadcast. Events support timed waits (Io.Condition does not), which
    /// is what lets the conn thread poll the peer socket during long
    /// prefills instead of blocking until the first token.
    out_event: std.Io.Event,
    out_buf: std.ArrayList(u32),
    out_idx: usize,
    finished: bool,
    error_code: ?[]const u8,
    finish_reason: []const u8,
    cancelled: std.atomic.Value(bool),

    // ── Stats (filled by inference thread, safe to read after finish). ──
    prompt_tokens: u32,
    completion_tokens: u32,
    prefill_tps: f64,
    decode_tps: f64,
    /// Monotonic timestamp captured in `Slot.init`, BEFORE the queue wait.
    /// Anchors the exact time-to-first-token measurement.
    request_start_ts: std.Io.Timestamp,
    /// Wall-clock nanoseconds from `request_start_ts` (request arrival) to
    /// prefill completion = queue_wait + prefill = real time-to-first-token.
    /// Captured directly when prefill finishes (never derived by subtracting
    /// decode time), so it stays exact even if a slot finishes mid-tick. Used
    /// as `real_ttft_ns` for the metrics histogram; e2e = first_token_ns + decode_ns.
    first_token_ns: u64,
    /// Wall-clock nanoseconds spent in `runPrefill` for this slot. Includes
    /// hot-prefix-cache lookup/restore and the model forward over the
    /// uncached tail. Populated by the scheduler main loop.
    prefill_ns: u64,
    /// Wall-clock nanoseconds the slot spent in decode ticks. For batched
    /// decode the full tick wall-clock is added to every participating slot,
    /// so this matches the per-slot throughput a user actually observes
    /// (`completion_tokens / decode_ns`).
    decode_ns: u64,
    /// Actual generated ids (from legacy_gen.generated_ids). Shallow copy at
    /// completion so the connection thread can read them without locking.
    generated_ids: ?[]u32,
    /// pad-only flag: set by inference thread when the entire generation was
    /// token id 0. Server-level cache invalidation reads this after complete.
    was_pad_only: bool,
    /// Phase A5: per-token logprobs accumulated by the inference thread when
    /// `logprobs_n > 0`. The conn thread takes ownership via
    /// `nonStreamingViaScheduler` (or equivalent) at completion; if the
    /// caller doesn't consume, `Slot.deinit` frees the contents.
    logprobs_buf: std.ArrayList(generate_mod.LogprobResult),

    /// Initialize but do NOT take ownership of caches — those are allocated
    /// inside `init` from the slot's allocator.
    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        config: *const ModelConfig,
        params: SubmitParams,
        kv_quant_config: transformer_mod.KVQuantConfig,
    ) !*Slot {
        const slot = try allocator.create(Slot);
        errdefer allocator.destroy(slot);

        // Embedded-GGUF model (ds4 or llama.cpp): skip all MLX per-slot
        // allocations — the engine owns its own KV cache, vision is not
        // supported, and the forward path bypasses `ForwardCtx`. Build
        // sentinel-empty fields so `Slot.deinit` is well-defined on both paths.
        const is_embedded = params.model.ds4_engine != null or params.model.llama_engine != null;

        // Per-slot KVCache, honoring the process-level kv-quant setting.
        // TurboQuant schemes need `head_dim` at construction time for the
        // per-layer rotation matrices; other schemes ignore it. For embedded
        // slots the engine owns its own cache — we initialize a zero-layer
        // shell so `Slot.deinit` is symmetric with the MLX path.
        const slot_kv_layers: u32 = if (is_embedded) 0 else config.num_hidden_layers;
        var cache = try KVCache.initWithConfigAndHeadDim(allocator, slot_kv_layers, kv_quant_config, config.head_dim);
        errdefer cache.deinit();

        // Per-slot SSM cache. Mirror the same predicate `Transformer.init`
        // uses to allocate `xfm.ssm_entries` (transformer.zig: `has_hybrid_layers`
        // OR `full_attention_interval > 0`). Without this branch the
        // slot's `ctx.ssm_entries` is null and `forwardMoeWith`'s
        // linear-attention layers crash on `ctx.ssm_entries.?` for Qwen 3.5/3.6
        // MoE (which carries GatedDeltaNet inside its MoE structure but does
        // NOT set `has_hybrid_layers`). Pure-attention MoE (qwen3_moe, Gemma 4
        // MoE — isMoe() but interval == 0) must NOT get entries: a non-null
        // slice makes the hot prefix cache treat the model as hybrid and
        // cold-prefill every request.
        var ssm_entries: ?[]SSMCacheEntry = null;
        if (!is_embedded and (config.has_hybrid_layers or config.full_attention_interval > 0)) {
            const entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
            for (entries) |*e| {
                e.* = .{
                    .conv_state = mlx.mlx_array_new(),
                    .ssm_state = mlx.mlx_array_new(),
                    .initialized = false,
                };
            }
            ssm_entries = entries;
        }
        errdefer if (ssm_entries) |entries| {
            for (entries) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
            }
            allocator.free(entries);
        };

        // Dup owned slices.
        const prompt_owned = try allocator.dupe(u32, params.prompt_ids);
        errdefer allocator.free(prompt_owned);
        const full_prompt_src = params.full_prompt orelse params.prompt_ids;
        const full_prompt_owned = try allocator.dupe(u32, full_prompt_src);
        errdefer allocator.free(full_prompt_owned);
        const eos_owned = try allocator.dupe(u32, params.eos_token_ids);
        errdefer allocator.free(eos_owned);

        slot.* = .{
            .allocator = allocator,
            .io = io,
            .model = params.model,
            .cache = cache,
            .moe_seq_offset = 0,
            .ssm_entries = ssm_entries,
            .vision_embeddings = params.vision_embeddings,
            .mrope_pos = params.mrope_pos,
            .mrope_total = params.mrope_total,
            .mrope_delta = params.mrope_delta,
            .ctx = undefined, // set after slot is in stable storage so pointers are valid
            .legacy_gen = null,
            .ds4_session = null,
            .diffusion = null,
            .ds4_rng = @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds()),
            .llama_session = null,
            .llama_rng = @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds()),
            .prompt_ids = prompt_owned,
            .full_prompt = full_prompt_owned,
            .sampling = params.sampling,
            .eos_token_ids = eos_owned,
            .max_tokens = params.max_tokens,
            .timeout_ns = params.timeout_ns,
            .has_tools = params.has_tools,
            .enable_pld = params.enable_pld,
            // MTP/drafter do extra speculative forwards that don't yet carry
            // Qwen3-VL M-RoPE positions — disable them for image requests so the
            // spec path can't desync image-token positions. PLD is fine (it only
            // engages on text decode, which uses the scalar offset+delta path).
            .enable_drafter = params.enable_drafter and params.vision_embeddings == null,
            .drafter = params.drafter,
            .drafter_block_size = params.drafter_block_size,
            .enable_mtp = params.enable_mtp and params.vision_embeddings == null,
            .mtp = params.mtp,
            .mtp_depth = params.mtp_depth,
            .pld_draft_len = params.pld_draft_len,
            .pld_key_len = params.pld_key_len,
            .kv_attn_fused = params.kv_attn_fused,
            .cached_tokens = params.cached_tokens,
            .logprobs_n = params.logprobs_n,
            .state = .pending_prefill,
            .out_mu = .init,
            .out_cond = .init,
            .out_event = .unset,
            .out_buf = std.ArrayList(u32).empty,
            .out_idx = 0,
            .finished = false,
            .error_code = null,
            .finish_reason = "length",
            .cancelled = std.atomic.Value(bool).init(false),
            .prompt_tokens = 0,
            .completion_tokens = 0,
            .prefill_tps = 0.0,
            .decode_tps = 0.0,
            .request_start_ts = std.Io.Timestamp.now(io, .boot),
            .first_token_ns = 0,
            .prefill_ns = 0,
            .decode_ns = 0,
            .generated_ids = null,
            .was_pad_only = true,
            .logprobs_buf = .empty,
        };

        // ForwardCtx points at fields owned by `slot` — must outlive the
        // Generator. The slot is heap-allocated so addresses are stable
        // until `complete` frees it.
        slot.ctx = .{
            .cache = &slot.cache,
            .moe_seq_offset = &slot.moe_seq_offset,
            .ssm_entries = slot.ssm_entries,
            .vision_embeddings = slot.vision_embeddings,
            .mrope_pos = slot.mrope_pos,
            .mrope_total = slot.mrope_total,
            .mrope_delta = slot.mrope_delta,
            .capture_hidden = null,
            .kv_attn_fused = params.kv_attn_fused,
        };

        return slot;
    }

    /// Free everything the slot owns. Only safe to call when no thread can
    /// observe the slot anymore (i.e. after the inference thread has
    /// finished/errored it AND the connection thread has consumed the final
    /// `done`/`err` from `waitNext`).
    pub fn deinit(self: *Slot) void {
        if (self.ds4_session) |session| {
            session.free();
            self.ds4_session = null;
        }
        // llama_session is borrowed from model.llama_session (persistent across
        // requests) — do NOT free it here. The claim on it is released in
        // Scheduler.complete; the session itself is freed with the model.
        self.llama_session = null;
        if (self.diffusion) |runner| {
            runner.deinit();
            self.allocator.destroy(runner);
            self.diffusion = null;
        }
        if (self.legacy_gen) |*gen| {
            gen.deinit(self.allocator);
        }
        self.cache.deinit();
        if (self.ssm_entries) |entries| {
            for (entries) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
            }
            self.allocator.free(entries);
        }
        if (self.vision_embeddings) |ve| _ = mlx.mlx_array_free(ve);
        if (self.mrope_pos) |mp| self.allocator.free(mp);
        self.allocator.free(self.prompt_ids);
        self.allocator.free(self.full_prompt);
        self.allocator.free(self.eos_token_ids);
        if (self.error_code) |code| self.allocator.free(code);
        if (self.generated_ids) |g| self.allocator.free(g);
        // Free any logprobs the conn thread didn't claim. After
        // `nonStreamingViaScheduler` calls `toOwnedSlice`, items.len becomes
        // 0 so this is a no-op on the success path.
        for (self.logprobs_buf.items) |*lp| self.allocator.free(lp.top_logprobs);
        self.logprobs_buf.deinit(self.allocator);
        self.out_buf.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Inference thread: enqueue a generated token for the consumer.
    fn pushToken(self: *Slot, t: u32) void {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        self.out_buf.append(self.allocator, t) catch |err| {
            // Allocation failure: degrade to error.
            self.error_code = self.allocator.dupe(u8, @errorName(err)) catch null;
            self.state = .errored;
        };
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
    }

    /// Inference thread: signal normal completion. Safe to call multiple
    /// times (idempotent on `finished`).
    fn markFinished(self: *Slot, reason: []const u8) void {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        if (self.finished) return;
        self.finished = true;
        self.state = .finished;
        self.finish_reason = reason;
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
    }

    /// Inference thread: signal error. `name` is borrowed; we dupe so the
    /// connection thread can read it after the inference loop drops the slot.
    fn markError(self: *Slot, name: []const u8) void {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        if (self.error_code != null or self.finished) return;
        self.error_code = self.allocator.dupe(u8, name) catch null;
        self.state = .errored;
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
    }

    /// Connection thread: block until the next token, completion, or error.
    /// Returns `.token` for each generated id, then exactly one terminator
    /// (`.done` or `.err`).
    pub fn waitNext(self: *Slot) NextResult {
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        while (true) {
            if (self.out_idx < self.out_buf.items.len) {
                const t = self.out_buf.items[self.out_idx];
                self.out_idx += 1;
                return .{ .token = t };
            }
            if (self.error_code != null) return .{ .err = {} };
            if (self.finished) return .{ .done = {} };
            // Cancellation (client disconnect or server shutdown) must unblock a
            // blocked reader — `cancel()` only broadcasts; without this the
            // reader would sleep until the inference thread happened to finish
            // the slot, so shutdown could never drain in-flight requests and
            // raced `Scheduler.deinit` into a use-after-free (SIGSEGV in
            // `complete`). Buffered tokens above still drain first.
            if (self.cancelled.load(.acquire)) return .{ .done = {} };
            self.out_cond.waitUncancelable(self.io, &self.out_mu);
        }
    }

    /// Connection thread: like `waitNext`, but wakes with `null` (idle)
    /// after `timeout_ms` with no token or terminator. Lets the caller poll
    /// the peer socket and emit SSE keepalives during long prefills —
    /// Claude Code disconnects after ~60s of stream silence, and pre-2026-06
    /// the handler sat blocked in `waitNext` for the whole multi-minute
    /// prefill, never noticed the disconnect, and abandoned giant prefills
    /// piled up serially behind every client retry (the server looked dead
    /// while the GPU ground ghosts; observed live with Claude Code + a
    /// 40K-token MCP prompt on gemma-4-12b).
    pub fn waitNextTimeout(self: *Slot, timeout_ms: i64) ?NextResult {
        while (true) {
            self.out_mu.lockUncancelable(self.io);
            if (self.out_idx < self.out_buf.items.len) {
                const t = self.out_buf.items[self.out_idx];
                self.out_idx += 1;
                self.out_mu.unlock(self.io);
                return .{ .token = t };
            }
            if (self.error_code != null) {
                self.out_mu.unlock(self.io);
                return .{ .err = {} };
            }
            if (self.finished) {
                self.out_mu.unlock(self.io);
                return .{ .done = {} };
            }
            // Cancellation unblocks the reader promptly (see waitNext) so a
            // disconnected/shutdown request stops instead of waiting out the
            // generation — the precondition for draining conn threads before
            // teardown.
            if (self.cancelled.load(.acquire)) {
                self.out_mu.unlock(self.io);
                return .{ .done = {} };
            }
            // Arm the event under the lock: producers mutate under this lock
            // and set() before releasing it, so a set racing our reset leaves
            // the event set and the wait below returns immediately — no lost
            // wakeups. A spurious wake just reads as an early idle (benign).
            self.out_event.reset();
            self.out_mu.unlock(self.io);
            self.out_event.waitTimeout(self.io, .{ .duration = .{
                .raw = .fromMilliseconds(timeout_ms),
                .clock = .awake,
            } }) catch return null;
        }
    }

    /// Connection thread: signal cancellation. The inference thread will
    /// drop this slot at the next tick boundary.
    pub fn cancel(self: *Slot) void {
        self.cancelled.store(true, .release);
        self.out_mu.lockUncancelable(self.io);
        defer self.out_mu.unlock(self.io);
        self.out_cond.broadcast(self.io);
        self.out_event.set(self.io);
    }
};

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
    model: *model_registry_mod.LoadedModel,
    /// Per-image float32 CHW pixel buffers. Borrowed; must outlive the call.
    images: []const VisionImagePixels,
    /// Gemma 4 12B unified audio: per-clip raw float32-LE 16 kHz mono sample
    /// buffers. Borrowed; must outlive the call. The inference thread frames
    /// each into 640-sample tokens and projects them through the audio embedder.
    audio: []const []const u8 = &.{},
    /// Output: encoded embedding tensor on success — vision soft tokens followed
    /// by audio soft tokens, concatenated along the token axis. Ownership
    /// transfers to the caller.
    result: ?mlx.mlx_array = null,
    /// Output: number of vision / audio soft tokens in `result` (in that order).
    /// The caller inserts exactly this many image / audio placeholders.
    n_vision_tokens: usize = 0,
    n_audio_tokens: usize = 0,
    /// Output: error name on failure. Owned by `allocator`; caller frees.
    error_name: ?[]const u8 = null,
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
    model: *model_registry_mod.LoadedModel,
    /// Tokenized inputs, one slice per text. Borrowed; must outlive the call.
    token_seqs: []const []const u32,
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
    drafter_dir: []const u8 = "",
    /// SSD weight-streaming for cold-loaded ds4 models (issue #39). The CLI
    /// startup path supplies this via LoadParams; cold-load defaults it off.
    ds4_ssd_streaming: bool = false,
    /// Auto-load the ds4 MTP draft head beside a cold-loaded ds4 model. Default
    /// on (a switched-to ds4 model gets speculative decode); forced off under
    /// `ds4_ssd_streaming`.
    ds4_mtp: bool = true,
    /// Auto-load the Qwen native MTP sidecar when the model dir ships one.
    mtp_enabled: bool = true,
    /// Max MTP draft depth (CLI --mtp-depth; 0 = auto, resolved by
    /// generate_mod.resolveMtpDepthCap at load/Generator init).
    mtp_depth: u32 = 0,

    load_vision: bool = false,
    warmup_eager: bool = true,
    draft_block_size: u32 = 4,
    draft_block_size_explicit: bool = false,
    kv_quant_config: transformer_mod.KVQuantConfig = transformer_mod.KVQuantConfig.dense,
    prefix_cache_capacity: u32 = 1,
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
    ssm_checkpoint_max: u32 = 32,
    /// Iteration 2: tokenize cache LRU capacity. Mirrored on
    /// `LoadParams.tokenize_cache_entries`; both paths feed
    /// `doLoadOnInferenceThread`.
    tokenize_cache_entries: u32 = 4,
    /// Iteration 3-5: llama.cpp multi-session cap. Mirrors
    /// `LoadParams.llama_cache_entries`.
    llama_cache_entries: u32 = 4,
    /// Phase 5 #2: ggml types for the embedded llama.cpp KV cache. 0 keeps
    /// libllama default (F16); Q8_0=8, Q4_0=2. Threaded onto the LoadedModel
    /// at load time.
    llama_kv_type_k: i32 = 0,
    llama_kv_type_v: i32 = 0,

    /// Victims to evict before the load (LRU-selected by the planner). Each is
    /// already marked `.evicting` with refcount == 0 by the conn thread under
    /// registry.mutex. The inference thread calls `unloadResident()` on each to
    /// free GPU memory, drops its resident-bytes accounting via
    /// `registry.accountEvictedLocked`, then `registry.finalizeEvictionLocked`.
    /// Borrows the conn thread's stack buffer; valid until `done`.
    evict_entries: []*LoadedModel = &.{},

    /// Output: error name on failure (owned by `allocator`; conn thread
    /// frees). Null on success.
    error_name: ?[]const u8 = null,

    /// Conn-thread synchronization. Inference thread broadcasts when done.
    done: bool = false,
    allocator: std.mem.Allocator,
    done_mu: std.Io.Mutex = .init,
    done_cond: std.Io.Condition = .init,

    /// Mirror `LoadParams.ds4_path` / `LoadParams.llama_path`. Set by
    /// `ensureLoaded` when the entry's path resolves to a GGUF (issue #59):
    /// the resolved .gguf file routes the load through the matching
    /// embedded-engine arm in `doLoadOnInferenceThread` instead of the MLX
    /// safetensors path. Borrowed from the conn thread until `done`.
    ds4_path: []const u8 = "",
    llama_path: []const u8 = "",
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
    done: bool = false,
    done_mu: std.Io.Mutex = .init,
    done_cond: std.Io.Condition = .init,
};

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

/// Continuous-batching scheduler. One per server. Owns the inference
/// thread, the queue of in-flight slots, AND (post-A1) the loaded model
/// state — Transformer + weights + vision encoder + drafter all live here,
/// allocated on the inference thread so mlx's thread-local GPU stream is
/// bound on the right thread from the start.
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
    drafter_block_size: u32,
    kv_quant_config: transformer_mod.KVQuantConfig,
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
    prefix_cache_mem_bytes: u64,
    prefix_cache_disk_bytes: u64,
    ssm_checkpoint_stride: u32,
    ssm_checkpoint_max: u32,
    /// Launch-flag MTP + embedded-llama.cpp settings, retained (same rationale
    /// as the prefix-cache fields above) so COLD-LOADED models — on-demand
    /// `/v1/load-model`, model switches — honor `--no-mtp` / `--mtp-depth` /
    /// `--llama-cache-entries` / `--llama-kv-quant` like the `--model` primary.
    /// Pre-plumbing, the cold-load `LoadRequest` used its struct defaults
    /// (mtp on, default depth, 4 llama sessions, F16 KV), silently ignoring
    /// these flags on every on-demand load and model switch.
    mtp_enabled: bool,
    mtp_depth: u32,
    llama_cache_entries: u32,
    llama_kv_type_k: i32,
    llama_kv_type_v: i32,

    // ── Borrowed refs (CPU-only state owned by the LoadedModel). ──
    config: *const ModelConfig,
    tok: *const Tokenizer,
    chat_config: *const ChatConfig,
    drafter_path: []const u8,

    /// Phase A6 → Plan 05: per-model hot prefix cache. Pre-Plan-05 this was
    /// a server-owned global; Plan 05 moves it onto `LoadedModel` so each
    /// model gets isolation by construction. Borrowed view here is the
    /// current model's cache (or null when the cache isn't applicable for
    /// the model, e.g. hybrid SSM archs).
    hot_prefix_cache: ?*prefix_cache_mod.HotPrefixCache,

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
    /// Number of slots currently inside `runPrefill`. Set on entry, cleared on
    /// every exit — so the panel can say "prefilling" IMMEDIATELY, rather than
    /// waiting for the first 8192-token chunk to land (~40 s on a 27B). Also
    /// covers the ds4/llama engines, whose prefill never reaches the MLX chunk
    /// loop and therefore never moves `inflight_prefill_tokens`.
    requests_prefilling: std.atomic.Value(u64),
    /// Counts `pending.len + decoding.len` for back-pressure.
    in_flight: u32,
    /// Capacity for back-pressure. `submit` waits when in_flight >= cap.
    /// `cap = max_concurrent + queue_depth`. queue_depth = 32 hardcoded
    /// (matches the legacy `max_queue_size`).
    queue_cap: u32,
    submit_cond: std.Io.Condition,
    /// Signaled when a persistent engine session (llama) is released in
    /// `complete()`, waking a `submit()` blocked waiting to claim it. Guarded by
    /// `queue_mu` together with `LoadedModel.llama_session_busy`.
    session_cond: std.Io.Condition,

    inference_thread: ?std.Thread,
    shutdown: std.atomic.Value(bool),
    started: std.atomic.Value(bool),
    started_mu: std.Io.Mutex,
    started_cond: std.Io.Condition,

    /// Set true by the inference thread if model load fails. Read by `init`
    /// after `started` is signaled to decide whether to surface a load error.
    load_failed: std.atomic.Value(bool),
    /// Owned, dupe'd error name (e.g. "MissingVisionWeights"). null on
    /// success. Freed in `deinit`.
    load_error_name: ?[]const u8,

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
            const raw = std.c.getenv("MLX_SERVE_FORCE_BATCHED");
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
            .prefix_cache_mem_bytes = params.prefix_cache_mem_bytes,
            .prefix_cache_disk_bytes = params.prefix_cache_disk_bytes,
            .ssm_checkpoint_stride = params.ssm_checkpoint_stride,
            .ssm_checkpoint_max = params.ssm_checkpoint_max,
            .mtp_enabled = params.mtp_enabled,
            .mtp_depth = params.mtp_depth,
            .llama_cache_entries = params.llama_cache_entries,
            .llama_kv_type_k = params.llama_kv_type_k,
            .llama_kv_type_v = params.llama_kv_type_v,
            // Initial borrowed-view refs point at the (heap-allocated) CPU
            // state carried on LoadParams; once the inference thread
            // installs them on `entry`, the views still resolve to the
            // same addresses (we store pointers, so the moves are no-ops).
            .config = params.config,
            .tok = params.tok,
            .chat_config = params.chat_config,
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

    pub fn deinit(self: *Scheduler) void {
        self.shutdown.store(true, .release);
        // Wake inference thread if it's waiting on queue_cond.
        self.queue_mu.lockUncancelable(self.io);
        self.queue_cond.broadcast(self.io);
        self.submit_cond.broadcast(self.io);
        self.queue_mu.unlock(self.io);

        if (self.inference_thread) |t| t.join();

        // Drain any leftover slots — should be empty if all conn threads
        // called `complete` properly, but defensive. Inference thread has
        // already exited by now (joined above), so freeing here is safe.
        for (self.pending.items) |slot| slot.deinit();
        self.pending.deinit(self.allocator);
        for (self.decoding.items) |slot| slot.deinit();
        self.decoding.deinit(self.allocator);
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
        self.hot_prefix_cache = null;
        if (self.load_error_name) |n| self.allocator.free(n);

        self.allocator.destroy(self);
    }

    /// Submit a new request. Builds a Slot, queues it, returns the handle.
    /// Blocks if the queue is full (i.e. `in_flight >= queue_cap`). When
    /// the scheduler is shutting down this returns `error.Shutdown`.
    pub fn submit(self: *Scheduler, params: SubmitParams) !*Slot {
        // Construct the slot up front so we don't hold the queue mutex
        // through any allocation. Per-request `kv_quant_config` override (Wave
        // 1.A) wins over the process-level default carried on the scheduler.
        const eff_kv_quant = params.kv_quant_config orelse self.kv_quant_config;
        // Phase D fix: use the slot's target-model config (not the
        // scheduler's startup-model config) so per-slot state allocation
        // (KVCache shape, SSM entries) matches the model that will
        // actually run the request. Critical when two models with
        // different architectures (e.g. pure-attention + hybrid SSM)
        // share one scheduler.
        const slot_config: *const ModelConfig = params.model.config orelse return error.ModelNotReady;
        const slot = try Slot.init(self.allocator, self.io, slot_config, params, eff_kv_quant);
        errdefer slot.deinit();

        self.queue_mu.lockUncancelable(self.io);
        defer self.queue_mu.unlock(self.io);

        while (self.in_flight >= self.queue_cap and !self.shutdown.load(.acquire)) {
            self.submit_cond.waitUncancelable(self.io, &self.queue_mu);
        }
        if (self.shutdown.load(.acquire)) return error.Shutdown;

        // Persistent-session engines (llama) reuse one KV context across
        // requests, so only one request may drive it at a time. Block here until
        // the model's session is free, then claim it (released in `complete`).
        // ds4 keeps a per-slot session, so it isn't gated. This serializes
        // concurrent llama requests (v1 scope) without spinning the inference
        // thread, and lets the next request reuse the previous one's prompt KV.
        if (params.model.llama_engine != null) {
            while (params.model.llama_session_busy and !self.shutdown.load(.acquire)) {
                self.session_cond.waitUncancelable(self.io, &self.queue_mu);
            }
            if (self.shutdown.load(.acquire)) return error.Shutdown;
            params.model.llama_session_busy = true;
            slot.llama_holds_session = true;
        }

        self.pending.append(self.allocator, slot) catch |err| {
            // Release the session claim before bubbling the error — the caller
            // never gets the slot, so `complete` won't run for it.
            if (slot.llama_holds_session) {
                params.model.llama_session_busy = false;
                slot.llama_holds_session = false;
                self.session_cond.broadcast(self.io);
            }
            return err;
        };
        self.in_flight += 1;
        self.queue_cond.broadcast(self.io);
        return slot;
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

        // Release the persistent llama session claim (if this slot held it) so
        // the next queued llama request can claim it AND reuse the KV prefix the
        // session now holds. Done before enqueueing cleanup so a waiting
        // submitter can proceed immediately.
        if (slot.llama_holds_session) {
            slot.model.llama_session_busy = false;
            slot.llama_holds_session = false;
            self.session_cond.broadcast(self.io);
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
        for (self.decoding.items) |slot| slot.cancel();
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
    ///   error.LoadFailed        — inference thread reported a load failure.
    ///   error.Shutdown          — scheduler is shutting down.
    pub fn ensureLoaded(self: *Scheduler, id_or_empty: []const u8) !*LoadedModel {
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
        const cpu_state = preloadCpuState(self.allocator, self.io, entry.path, self.gguf_ctx_size) catch |err| {
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
        // The resolved .gguf path (when this is a GGUF entry) is borrowed by
        // the LoadRequest until `done`; the engines dupe what they keep, so
        // it's released here on success AND failure.
        defer if (owned.gguf) |g| self.allocator.free(g.path);

        // Victims selected by the eviction planner (multi-victim: one load may
        // need to free several models to fit). Lives on this stack frame; the
        // slice handed to the LoadRequest stays valid while we block on `done`.
        var victims_buf: [16]*LoadedModel = undefined;
        var n_victims: usize = 0;

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
                        self.registry.mutex.unlock(self.io);
                        return error.LoadFailed;
                    },
                    .unloaded => break :wait_loop,
                }
            }

            // Claim the slot.
            std.debug.assert(self.registry.tryBeginLoadLocked(entry));

            // Estimate post-load bytes. We use the same estimator
            // `doLoadOnInferenceThread` does (entry.bytes_on_disk; else a
            // layers × hidden fallback). Adds 10% headroom for KV / vision
            // / drafter overhead — close enough for the eviction gate.
            const estimated: u64 = blk: {
                const base: u64 = if (entry.bytes_on_disk) |b|
                    b
                else
                    @as(u64, owned.config.num_hidden_layers) * @as(u64, owned.config.hidden_size) * 4 * 4;
                break :blk base + base / 10;
            };

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
            const n = self.registry.planEvictionsLocked(entry.id, &victims_buf) orelse {
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
            .drafter_dir = "", // Phase E will wire the load-model API to set this.
            .load_vision = coldLoadVision(owned.config.has_vision),
            .warmup_eager = true,
            .draft_block_size = drafter_mod.DEFAULT_BLOCK_SIZE,
            .draft_block_size_explicit = false,
            .kv_quant_config = self.kv_quant_config,
            // Cold loads get the SAME prefix-cache configuration as the
            // startup model — pre-plumbing these were (1, 0, stride 0),
            // which silently degraded warm reuse after every model switch.
            .prefix_cache_capacity = self.prefix_cache_capacity,
            .prefix_cache_mem_bytes = self.prefix_cache_mem_bytes,
            .prefix_cache_disk_bytes = self.prefix_cache_disk_bytes,
            .ssm_checkpoint_stride = self.ssm_checkpoint_stride,
            .ssm_checkpoint_max = self.ssm_checkpoint_max,
            // Cold loads honor the launch-flag MTP + embedded-llama.cpp
            // settings too (same reason as prefix-cache above) — pre-plumbing
            // these were LoadRequest defaults, so --no-mtp / --mtp-depth /
            // --llama-cache-entries / --llama-kv-quant were silently dropped
            // on every on-demand load and model switch.
            .mtp_enabled = self.mtp_enabled,
            .mtp_depth = self.mtp_depth,
            .llama_cache_entries = self.llama_cache_entries,
            .llama_kv_type_k = self.llama_kv_type_k,
            .llama_kv_type_v = self.llama_kv_type_v,
            .evict_entries = victims_buf[0..n_victims],
            .allocator = self.allocator,
        };
        // GGUF entry: route through the matching embedded-engine arm in
        // doLoadOnInferenceThread (issue #59 — discovered/pulled GGUF dirs
        // cold-load on demand like MLX ones).
        if (owned.gguf) |g| switch (g.engine) {
            .ds4 => req.ds4_path = g.path,
            .llama => req.llama_path = g.path,
        };

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
            self.allocator.free(name);
            // On success the inference thread took ownership of cpu_state;
            // on failure it didn't, so we still hold it.
            return error.LoadFailed;
        }

        // Success — inference thread installed cpu_state onto the entry.
        owned_active = false;

        // Re-acquire under mutex and refcount the ready entry.
        self.registry.mutex.lockUncancelable(self.io);
        defer self.registry.mutex.unlock(self.io);
        if (entry.state != .ready) return error.LoadFailed;
        _ = entry.refcount.fetchAdd(1, .acq_rel);
        return entry;
    }

    /// Release a borrowed pointer obtained from `ensureLoaded`. Forwards
    /// to the registry so the refcount decrement + LRU clock bump happen
    /// under registry.mutex.
    pub fn release(self: *Scheduler, lm: *LoadedModel) void {
        self.registry.release(lm);
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

    /// Active-tick gate. Decides whether a slot is eligible for the batched
    /// decode kernel. Hybrid SSM / MoE / encoder / DSV4 models can't ride
    /// the batched kernel (it doesn't model their state), so any slot
    /// targeting such a model falls through to the single-slot path. Phase
    /// D: the gate reads off the slot's own model config — multi-model
    /// means the scheduler's startup config is no longer authoritative.
    fn batchable(self: *const Scheduler, slot: *const Slot) bool {
        _ = self;
        if (slot.enable_pld or slot.enable_drafter or slot.enable_mtp) return false;
        if (slot.sampling.constraint != null) return false;
        if (slot.logprobs_n > 0) return false;
        // Embedded-GGUF slots (ds4 / llama.cpp) have no `ForwardCtx` — they
        // always fall through to the per-slot decode path (which dispatches
        // into the engine).
        if (slot.model.ds4_engine != null or slot.model.llama_engine != null) return false;
        const cfg = slot.model.config orelse return false;
        return modelBatchable(cfg);
    }
};

/// Pure-config predicate: is this model's architecture compatible with the
/// batched-decode kernel? Used by `Scheduler.batchable` after slot-level
/// flags are checked. MoE / hybrid / encoder have shape mismatches with
/// `forwardBatchedDecode` and fall through to per-slot dispatch.
pub fn modelBatchable(cfg: *const model_mod.ModelConfig) bool {
    if (cfg.has_hybrid_layers) return false;
    if (cfg.full_attention_interval > 0) return false;
    if (cfg.is_encoder_only) return false;
    if (cfg.isMoe()) return false;
    // Block diffusion denoises whole canvases — no per-token batched decode.
    if (cfg.isDiffusion()) return false;
    return true;
}

const ThreadCtx = struct {
    scheduler: *Scheduler,
    params: LoadParams,
};

/// Signal to the parent waiting in `init()` that the inference thread is done
/// with its load (or has failed). After `started` flips, the parent reads
/// `load_failed` to decide whether to surface a startup error.
fn signalStarted(sch: *Scheduler) void {
    sch.started_mu.lockUncancelable(sch.io);
    defer sch.started_mu.unlock(sch.io);
    sch.started.store(true, .release);
    sch.started_cond.broadcast(sch.io);
}

/// Set the load_error_name + flip load_failed. Best-effort dupe; on OOM the
/// parent still sees `load_failed=true` and surfaces "unknown".
fn recordLoadError(sch: *Scheduler, err_name: []const u8) void {
    if (sch.load_error_name) |old| sch.allocator.free(old);
    sch.load_error_name = sch.allocator.dupe(u8, err_name) catch null;
    sch.load_failed.store(true, .release);
}

/// Heap-allocate `T`, run `init_fn`, return owning pointer. On `init_fn`
/// failure, the heap slot is freed before the error propagates so the
/// scheduler never holds a half-initialized struct.
fn boxInit(
    allocator: std.mem.Allocator,
    comptime T: type,
    init_fn: anytype,
    args: anytype,
) !*T {
    const ptr = try allocator.create(T);
    errdefer allocator.destroy(ptr);
    ptr.* = try @call(.auto, init_fn, args);
    return ptr;
}

/// Routing decision for a GGUF entry resolved at preload time: the actual
/// `.gguf` file (owned by the conn thread, freed after the load completes —
/// the engines dupe/copy what they keep) and which embedded engine serves it.
const GgufRoute = struct {
    path: []u8,
    engine: gguf_meta.Engine,
};

/// Plan 05 Phase D: pre-loaded CPU state bundle. Built by the conn thread
/// (CPU only — file I/O + parse, no mlx) ahead of posting a LoadRequest.
/// Ownership transfers to the entry on successful load; on failure the
/// conn thread frees via `freeCpuState`. `gguf` (when set) stays owned by
/// the conn thread either way.
const CpuState = struct {
    config: *ModelConfig,
    tok: *Tokenizer,
    chat_config: *ChatConfig,
    /// Non-null when the model is a GGUF served by an embedded engine
    /// (issue #59: discovered/pulled GGUF dirs cold-load on demand).
    gguf: ?GgufRoute = null,
};

/// Phase D: parse config.json, tokenizer, and chat config from `model_dir`
/// into heap pointers ready to hand off to `LoadRequest`. Mirrors the
/// pre-load that main.zig does for the startup model in serve mode.
///
/// Errdefer pattern: for each `try ... else error`, the `deinit` errdefer
/// is registered AFTER the successful init so a downstream failure doesn't
/// call deinit on uninitialized memory. ModelConfig has no allocator-owned
/// fields, so `allocator.destroy` is sufficient there.
fn preloadCpuState(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8, gguf_ctx_size: u32) !CpuState {
    // GGUF first — mirrors `--model` routing in main.zig, where isGgufPath
    // is checked before any config.json read ("GGUF files bypass the MLX
    // dispatch entirely"). The embedded engine owns the real tokenizer +
    // chat template, so the CPU state is a stub, like the media path below.
    if (model_discovery.isGgufModelPath(io, model_dir)) {
        return preloadGgufCpuState(allocator, io, model_dir, gguf_ctx_size);
    }

    // Media model (image/audio/video): the engine owns the real tokenizer +
    // forward path, so we hand the inference thread a minimal stub instead of
    // parsing a transformer-shaped config (which would fail on a flux2/
    // qwen3_tts/AudioVideo config.json). The gen load arm dispatches off the
    // stub's `model_type`.
    if (gen_mod.detectModality(io, allocator, model_dir)) |modality| {
        const stub = try gen_mod.buildStubCpuState(allocator, modality);
        return .{ .config = stub.config, .tok = stub.tok, .chat_config = stub.chat_config };
    }

    const config = try allocator.create(ModelConfig);
    errdefer allocator.destroy(config);
    config.* = try model_mod.parseConfig(io, allocator, model_dir);

    const tok = try allocator.create(Tokenizer);
    errdefer allocator.destroy(tok);
    tok.* = try tokenizer_mod.loadTokenizer(io, allocator, model_dir);
    errdefer tok.deinit();

    const cc = try allocator.create(ChatConfig);
    errdefer allocator.destroy(cc);
    cc.* = try chat_mod.loadChatConfig(io, allocator, model_dir);
    errdefer cc.deinit();

    // Same EOS-resolution as main.zig — merge the tokenizer's chat-terminator
    // EOS into the stop set ALWAYS, even when config.json already specified an
    // eos_token_id. Some checkpoints (e.g. Qwen2.5-Coder-7B) set config.json
    // eos_token_id to <|endoftext|> but end chat turns with <|im_end|>; gating
    // on `num_eos_tokens == 0` left <|im_end|> out of the stop set and it leaked
    // into output. Additive + dedup-guarded: only ever ADDS a declared stop.
    if (cc.eos_token) |eos_str| {
        if (tok.special_tokens.get(eos_str)) |eos_id| {
            if (!config.isEosToken(eos_id)) config.addEosToken(eos_id);
        }
    }
    if (tok.special_tokens.get("<|endoftext|>")) |eot_id| {
        if (!config.isEosToken(eot_id)) config.addEosToken(eot_id);
    }
    if (tok.special_tokens.get("<pad>")) |pad_id| {
        if (pad_id > 0 and !config.isEosToken(pad_id)) {
            config.addEosToken(pad_id);
        }
    }

    return .{ .config = config, .tok = tok, .chat_config = cc };
}

/// Frees the three CPU-state pointers. Does NOT free `s.gguf` — the
/// resolved .gguf path is borrowed by the LoadRequest until `done` and is
/// freed by ensureLoaded's own defer on both success and failure paths
/// (on success the three pointers transfer to the entry, so this function
/// is skipped, but the path must still be released).
fn freeCpuState(allocator: std.mem.Allocator, s: *CpuState) void {
    allocator.destroy(s.config);
    s.tok.deinit();
    allocator.destroy(s.tok);
    s.chat_config.deinit();
    allocator.destroy(s.chat_config);
}

/// GGUF cold-load preload: resolve the actual .gguf file, pick the embedded
/// engine from its header metadata (the issue #15 rule in gguf_meta.zig;
/// unreadable metadata defaults to llama with a log line, mirroring
/// main.zig's chooseGgufEngine), and build the stub CPU state.
fn preloadGgufCpuState(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8, ctx_size: u32) !CpuState {
    const gguf_path = model_discovery.resolveGgufFile(io, allocator, model_dir) catch |err| {
        model_discovery.logResolveGgufError(model_dir, err);
        return err;
    };
    errdefer allocator.free(gguf_path);

    const engine: gguf_meta.Engine = blk: {
        var info = gguf_meta.readFromFile(io, allocator, gguf_path) catch |err| {
            log.warn("[gguf] route: metadata read failed ({s}); defaulting to llama\n", .{@errorName(err)});
            break :blk .llama;
        };
        defer info.deinit(allocator);
        const e = gguf_meta.preferredEngine(info);
        log.info("[gguf] engine: {s} (arch={s}, ds4-lora={})\n", .{
            @tagName(e),
            info.architecture orelse "?",
            info.has_ds4_lora_rank,
        });
        break :blk e;
    };

    var state = try buildGgufStubCpuState(allocator, engine, ctx_size);
    state.gguf = .{ .path = gguf_path, .engine = engine };
    return state;
}

/// Stub CPU state for a GGUF cold load. Mirrors the stubs main.zig's
/// `runLlamaServe`/`runDs4Serve` build for the startup path: the embedded
/// engine owns the real tokenizer + chat template (the llama arm adopts the
/// GGUF's embedded template at load), so only `model_type` (echoed in
/// /v1/models + engine-arm dispatch guards) and `max_position_embeddings`
/// (session sizing in runPrefillLlama/runPrefillDs4 + the server's context
/// guard) matter. `ctx_size` is the --ctx-size launch flag; 0 → the same
/// defaults the startup paths use (8192 for llama — the GGUF's trained
/// context isn't readable until the engine opens; ds4's own default via
/// clampSessionCtx).
fn buildGgufStubCpuState(allocator: std.mem.Allocator, engine: gguf_meta.Engine, ctx_size: u32) !CpuState {
    const config = try allocator.create(ModelConfig);
    errdefer allocator.destroy(config);
    config.* = switch (engine) {
        .llama => ModelConfig{
            .model_type = "gguf",
            .weight_prefix = "model",
            .head_dim = 128,
            .max_position_embeddings = if (ctx_size > 0) ctx_size else 8192,
            .is_encoder_only = false,
        },
        .ds4 => ModelConfig{
            .model_type = "deepseek_v4",
            .weight_prefix = "model",
            .num_hidden_layers = 61,
            .hidden_size = 7168,
            .head_dim = 128,
            .num_attention_heads = 56,
            .num_key_value_heads = 56,
            .max_position_embeddings = arch_ds4.clampSessionCtx(ctx_size),
            .is_encoder_only = false,
        },
    };

    const tok = try allocator.create(Tokenizer);
    errdefer allocator.destroy(tok);
    var byte_map: [256]u21 = undefined;
    for (0..256) |b| byte_map[b] = @intCast(b);
    tok.* = .{
        .vocab = std.StringHashMap(u32).init(allocator),
        .id_to_token = std.AutoHashMap(u32, []const u8).init(allocator),
        .merge_ranks = @TypeOf(tok.merge_ranks).init(allocator),
        .allocator = allocator,
        .special_tokens = std.StringHashMap(u32).init(allocator),
        .tok_type = .byte_level_bpe,
        .byte_to_unicode = byte_map,
        .unicode_to_byte = std.AutoHashMap(u21, u8).init(allocator),
        .bos_id = null,
        .eos_id = null,
        .parsed_json = null,
    };
    errdefer tok.deinit();

    const cc = try allocator.create(ChatConfig);
    errdefer allocator.destroy(cc);
    cc.* = .{
        .chat_template = try allocator.dupe(u8, ""),
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    return .{ .config = config, .tok = tok, .chat_config = cc };
}

/// ds4 load on the inference thread. Mirrors the MLX path's "open weights
/// → install on entry → markReady" shape but works exclusively through the
/// embedded engine. The stub `config`/`tok`/`chat_config` come in via
/// `params.config`/`params.tok`/`params.chat_config` (main.zig allocates
/// them); the entry takes ownership.
/// ds4 MTP speculative-decode defaults. `draft_tokens` MUST be > 1 to engage
/// (ds4 gates on it); margin 3.0 mirrors ds4's own default acceptance margin.
/// The scratch buffer holds the verified token + up to ds4's 16-draft cap.
const DS4_MTP_DRAFT_TOKENS: c_int = 4;
const DS4_MTP_MARGIN: f32 = 3.0;
const DS4_MTP_MAX_TOKENS: usize = 17;

/// Whether a ds4 decode step should use MTP speculative decode: the engine
/// loaded a draft head (`has_mtp`), it's configured for >1 draft token, and
/// sampling is greedy (ds4's spec path is argmax-based — temp>0 falls back to
/// the normal sampler). Pure + unit-tested; mirrors ds4's own CLI gate.
fn ds4MtpShouldEngage(has_mtp: bool, draft_tokens: c_int, temperature: f32) bool {
    return has_mtp and draft_tokens > 1 and temperature <= 0.0;
}

fn doLoadDs4OnInferenceThread(sch: *Scheduler, params: anytype) !void {
    log.info("[ds4] opening engine: {s}\n", .{params.ds4_path});
    // Auto-load the MTP draft head sitting beside the model for speculative
    // decode. ds4 refuses `--mtp` together with `--ssd-streaming`, so the sidecar
    // is only sought when streaming is off; `ds4_mtp` (default on) gates opt-out.
    const mtp_path: ?[]u8 = if (params.ds4_mtp and !params.ds4_ssd_streaming)
        model_discovery.findDs4MtpSidecar(sch.io, sch.allocator, params.ds4_path)
    else
        null;
    defer if (mtp_path) |p| sch.allocator.free(p);
    if (mtp_path) |p| log.info("[ds4] MTP draft head: {s}\n", .{p});

    const engine = try arch_ds4.Ds4Engine.open(sch.allocator, params.ds4_path, .{
        .backend = .metal,
        .warm_weights = true,
        .ssd_streaming = params.ds4_ssd_streaming,
        .mtp_path = mtp_path,
        .mtp_draft_tokens = if (mtp_path != null) DS4_MTP_DRAFT_TOKENS else 0,
        .mtp_margin = DS4_MTP_MARGIN,
    });
    errdefer engine.close();
    log.info("[ds4] engine ready (EOS={d}, has_mtp={})\n", .{ engine.eosToken(), engine.hasMtp() });

    // Make sure the stub config knows about the engine's EOS token so the
    // streaming/non-streaming paths' EOS check fires correctly. addEosToken
    // is a no-op if the slot is already present.
    const eos_id: u32 = @intCast(engine.eosToken());
    params.config.addEosToken(eos_id);

    // ── Install on entry. Everything below must be infallible (mirrors
    //    the MLX path's invariant about the per-ptr errdefers above).
    const entry = params.entry;
    entry.ds4_engine = engine;
    entry.config = params.config;
    entry.tokenizer = params.tok;
    entry.chat_config = params.chat_config;
    entry.weights = null;
    entry.transformer = null;
    entry.vision_encoder = null;
    entry.drafter = null;
    entry.drafter_block_size = 0;
    entry.drafter_path = "";
    entry.prefix_cache = null;
    // Iteration 2: tokenize cache also applies on the ds4 path. The
    // MLX-branch assignment isn't reached here because we early-return.
    if (params.tokenize_cache_entries > 0) {
        entry.tokenize_cache = tokenize_cache_mod.TokenizeCache.init(
            sch.allocator,
            params.tokenize_cache_entries,
        );
    }

    // Bytes-resident is whatever main.zig handed us (typically the GGUF
    // on-disk size). ds4's `ds4_context_memory_estimate` could give a
    // tighter number; we leave that as a TODO since the registry's
    // eviction gate doesn't currently support multi-engine residency.
    const bytes_resident: u64 = if (entry.bytes_on_disk) |b| b else 0;

    sch.registry.mutex.lockUncancelable(sch.io);
    sch.registry.markReadyLocked(entry, bytes_resident);
    sch.registry.mutex.unlock(sch.io);

    // Scheduler's borrowed views: leave the MLX fields null. `runPrefill`
    // / `runSingleDecodeTick` branch on `slot.ds4_session` and never touch
    // `sch.xfm` for ds4 slots.
    sch.current_model = entry;
    sch.xfm = null;
    sch.weights = null;
    sch.vision_encoder = null;
    sch.drafter = null;
    sch.hot_prefix_cache = null;
}

/// llama.cpp load on the inference thread. Mirrors `doLoadDs4OnInferenceThread`:
/// open the embedded engine (its Metal kernels bind to this thread's GPU stream
/// from t0), install it on the entry, move the stub config/tok/chat_config over,
/// and mark ready. The stub config carries the effective context length so the
/// server's memory estimate and `runPrefillLlama` size the session correctly.
fn doLoadLlamaOnInferenceThread(sch: *Scheduler, params: anytype) !void {
    log.info("[llama] opening engine: {s}\n", .{params.llama_path});
    const engine = try arch_llama.LlamaEngine.open(sch.allocator, params.llama_path, .{});
    errdefer engine.close();
    log.info("[llama] engine ready (EOS={d}, n_vocab={d})\n", .{ engine.eosToken(), engine.nVocab() });

    // Make sure the stub config's EOS set includes the engine's EOS so the
    // streaming/non-streaming stop checks fire.
    const eos_id: u32 = @intCast(engine.eosToken());
    params.config.addEosToken(eos_id);

    // Adopt the GGUF's embedded chat template into the stub ChatConfig so
    // `chat.encodeChatViaLlama` can render it through mlx-serve's Jinja engine
    // (which also supplies the tool-synthesis fallback). The stub starts with an
    // empty allocator-owned template; swap it for the model's, freed by deinit.
    if (engine.chatTemplate()) |tmpl| {
        if (params.chat_config.allocator.dupe(u8, tmpl)) |dup| {
            params.chat_config.allocator.free(params.chat_config.chat_template);
            params.chat_config.chat_template = dup;
        } else |_| {}
    }

    const entry = params.entry;
    entry.llama_engine = engine;
    entry.config = params.config;
    entry.tokenizer = params.tok;
    entry.chat_config = params.chat_config;
    entry.weights = null;
    entry.transformer = null;
    entry.vision_encoder = null;
    entry.drafter = null;
    entry.drafter_block_size = 0;
    entry.drafter_path = "";
    entry.prefix_cache = null;
    // Iteration 2: tokenize cache also applies on the llama path — same
    // chat-template render + tokenize round-trip per request. Wire it
    // here too so `--tokenize-cache-entries N` works for GGUFs.
    if (params.tokenize_cache_entries > 0) {
        entry.tokenize_cache = tokenize_cache_mod.TokenizeCache.init(
            sch.allocator,
            params.tokenize_cache_entries,
        );
    }
    // Iteration 3-5: cap for the llama.cpp multi-session LRU. The MLX
    // load path sets this further down; for llama we exit early at the
    // top of doLoadOnInferenceThread, so it has to land here.
    entry.llama_cache_max_entries = if (params.llama_cache_entries > 0)
        params.llama_cache_entries
    else
        1;
    // Phase 5 #2: ggml KV-quant types — same reason as above; the
    // MLX path's assignment is never reached on the llama branch.
    entry.llama_kv_type_k = params.llama_kv_type_k;
    entry.llama_kv_type_v = params.llama_kv_type_v;

    const bytes_resident: u64 = if (entry.bytes_on_disk) |b| b else 0;

    sch.registry.mutex.lockUncancelable(sch.io);
    sch.registry.markReadyLocked(entry, bytes_resident);
    sch.registry.mutex.unlock(sch.io);

    sch.current_model = entry;
    sch.xfm = null;
    sch.weights = null;
    sch.vision_encoder = null;
    sch.drafter = null;
    sch.hot_prefix_cache = null;
}

/// Media-gen load on the inference thread. Mirrors `doLoadDs4OnInferenceThread`:
/// build the modality engine (its mlx ops bind to this thread's GPU stream from
/// t0), install it on the entry, move the stub config/tok/chat_config over, and
/// mark ready. The MLX/ds4/llama fields stay null; request handlers route
/// through the matching `image_engine`/`audio_engine`/`video_engine` slot.
fn doLoadGenOnInferenceThread(sch: *Scheduler, params: anytype, modality: gen_mod.Modality) !void {
    log.info("[gen] loading {s} engine: {s}\n", .{ @tagName(modality), params.model_dir });
    const entry = params.entry;

    // Build the engine FIRST. On failure the engine's own errdefer cleans up
    // its partial state, the slot stays null, and the stub config (still owned
    // by the caller, not yet installed below) is freed by the caller's
    // error path — so we must not touch `entry.config` before this succeeds.
    switch (modality) {
        .image => entry.image_engine = try gen_mod.ImageEngine.load(sch.io, sch.allocator, params.model_dir),
        .audio => entry.audio_engine = try gen_mod.AudioEngine.load(sch.io, sch.allocator, params.model_dir),
        .video => entry.video_engine = try gen_mod.VideoEngine.load(sch.io, sch.allocator, params.model_dir),
        .mesh => entry.mesh_engine = try gen_mod.MeshEngine.load(sch.io, sch.allocator, params.model_dir),
    }

    // Install stub CPU state (infallible from here, mirroring the ds4 path).
    entry.config = params.config;
    entry.tokenizer = params.tok;
    entry.chat_config = params.chat_config;
    entry.weights = null;
    entry.transformer = null;
    entry.vision_encoder = null;
    entry.drafter = null;
    entry.drafter_block_size = 0;
    entry.drafter_path = "";
    entry.prefix_cache = null;

    const bytes_resident: u64 = blk: {
        if (entry.bytes_on_disk) |b| {
            if (b > 0) break :blk b;
        }
        break :blk gen_mod.estimateResidentBytes(sch.io, params.model_dir);
    };

    sch.registry.mutex.lockUncancelable(sch.io);
    sch.registry.markReadyLocked(entry, bytes_resident);
    sch.registry.mutex.unlock(sch.io);

    sch.current_model = entry;
    sch.xfm = null;
    sch.weights = null;
    sch.vision_encoder = null;
    sch.drafter = null;
    sch.hot_prefix_cache = null;
}

/// Sum of `*.safetensors` bytes in `model_dir` — the MLX weight footprint used
/// by the load pre-flight. Returns 0 if the dir can't be read (treated as
/// "unknown" by the caller, which then skips the check).
fn modelDiskBytes(io: std.Io, model_dir: []const u8) u64 {
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true }) catch return 0;
    defer dir.close(io);
    var it = dir.iterate();
    var total: u64 = 0;
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        const st = dir.statFile(io, entry.name, .{}) catch continue;
        total += @intCast(st.size);
    }
    return total;
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

/// Should a cold load bring up the checkpoint's vision tower?
pub fn coldLoadVision(has_vision: bool) bool {
    return has_vision and !no_vision_global;
}

test "coldLoadVision honors the process-wide vision opt-out" {
    no_vision_global = false;
    try std.testing.expect(coldLoadVision(true));
    try std.testing.expect(!coldLoadVision(false));
    no_vision_global = true;
    defer no_vision_global = false;
    try std.testing.expect(!coldLoadVision(true));
}

/// Which "available memory" figure the preflight should trust. On iOS the
/// host-wide number is meaningless — the OS keeps RAM full (file cache,
/// jetsam-evictable background apps) and will evict on our behalf, so the
/// per-PROCESS jetsam headroom (`os_proc_available_memory`, nonzero only on
/// iOS) is the figure that decides whether the load survives. Live bug: an
/// 8 GB iPhone reported ~4 GB host-free and the preflight refused a 3.6 GB
/// model that fit comfortably inside the ~6.4 GB process limit.
fn effectiveAvailableBytes(host_avail: u64, proc_avail: u64) u64 {
    return if (proc_avail > 0) proc_avail else host_avail;
}

test "effectiveAvailableBytes prefers the per-process jetsam headroom when present" {
    const GB: u64 = 1024 * 1024 * 1024;
    try std.testing.expectEqual(6 * GB, effectiveAvailableBytes(4 * GB, 6 * GB)); // iOS: proc wins
    try std.testing.expectEqual(4 * GB, effectiveAvailableBytes(4 * GB, 0)); // macOS: proc query = 0 → host
    try std.testing.expectEqual(@as(u64, 0), effectiveAvailableBytes(0, 0)); // both unknown → 0 (never blocks)
}

fn memInsufficientForLoad(weights_bytes: u64, avail_bytes: u64) bool {
    if (weights_bytes == 0 or avail_bytes == 0) return false;
    // Headroom over the weights for warmup compute buffers + a baseline KV cache.
    // `avail_bytes` (status.getAvailableMemBytes) now excludes the resident anon
    // set — an already-loaded model counts as used while file cache counts as free
    // — so this margin can be generous without wrongly refusing a fresh load.
    // CAVEAT: the KV cache scales with --ctx-size, which this guard doesn't see;
    // a very large context can still exceed this margin (follow-up: plumb ctx +
    // kv_quant to size KV precisely). Bypass with --skip-mem-preflight.
    const headroom: u64 = weights_bytes / 8 + 1024 * 1024 * 1024;
    return avail_bytes < weights_bytes + headroom;
}

test "memInsufficientForLoad: headroom + unknown-query guards" {
    const GB: u64 = 1024 * 1024 * 1024;
    const MB: u64 = 1024 * 1024;
    // A 6.9 GB 4-bit model with ~10 GB genuinely available — file cache is
    // excluded from the new anon-aware available figure (computeAvailableBytes),
    // so this is what a 16 GB Mac actually reports pre-load. Needs ~8.8 GB
    // (weights + weights/8 + 1 GB for warmup + baseline KV) → loads.
    try std.testing.expect(!memInsufficientForLoad(6900 * MB, 10 * GB));
    // Restart-into-pressure: 42 GB weights, only 44 GB free → needs ~46, refuse.
    try std.testing.expect(memInsufficientForLoad(42 * GB, 44 * GB));
    // Plenty of headroom → allow.
    try std.testing.expect(!memInsufficientForLoad(42 * GB, 86 * GB));
    // Exactly weights, no headroom → refuse.
    try std.testing.expect(memInsufficientForLoad(42 * GB, 42 * GB));
    // Unknown figures (query failed / size unknown) → never block.
    try std.testing.expect(!memInsufficientForLoad(0, 44 * GB));
    try std.testing.expect(!memInsufficientForLoad(42 * GB, 0));
}

/// Phase A1 → Plan 05: do the full model load on the inference thread.
/// mlx ops here bind to this thread's GPU stream from t0; subsequent
/// forwards stay on the same thread.
///
/// `params` is duck-typed (`anytype`): both `LoadParams` (startup) and
/// `*LoadRequest` (on-demand) supply the same field set — `entry`,
/// `config`/`tok`/`chat_config` (heap pointers), `model_dir`,
/// `drafter_dir`, `load_vision`, `warmup_eager`, `draft_block_size`,
/// `draft_block_size_explicit`, `kv_quant_config`, `prefix_cache_capacity`,
/// `prefix_cache_mem_bytes`. The function reads them by name.
///
/// On any error the partial state has already been freed via errdefer; the
/// caller decides how to surface (startup → recordLoadError + signal
/// started; on-demand → req.error_name + done broadcast).
fn doLoadOnInferenceThread(sch: *Scheduler, params: anytype) !void {
    // ── ds4 fast path: when the caller passed `ds4_path`, the model is a
    //    GGUF served by the embedded ds4 engine. The MLX scaffolding
    //    (weights/Transformer/vision/drafter/JIT/warmup) is entirely
    //    irrelevant — we open the engine on this thread (Metal kernels
    //    bind to the local stream from t0), install it on the entry, and
    //    mark ready. The stub config/tok/chat_config supplied by main.zig
    //    is moved onto the entry so server-side reads of `lm.config.?`
    //    (eos slices, context length, model name) keep working.
    //
    //    `params` is anytype — either `LoadParams` (startup) or `*LoadRequest`
    //    (on-demand cold-load; `ensureLoaded` sets `ds4_path`/`llama_path`
    //    when the entry resolves to a GGUF). Read from the value type's
    //    fields after a deref-when-pointer.
    const Ty = @TypeOf(params);
    const TyInfo = @typeInfo(Ty);
    const Inner = if (TyInfo == .pointer) TyInfo.pointer.child else Ty;
    if (@hasField(Inner, "ds4_path") and params.ds4_path.len > 0) {
        try doLoadDs4OnInferenceThread(sch, params);
        return;
    }
    // ── llama.cpp fast path: same shape as ds4, for any other GGUF. Fires on
    //    startup LoadParams AND cold-load LoadRequests carrying `llama_path`.
    if (@hasField(Inner, "llama_path") and params.llama_path.len > 0) {
        try doLoadLlamaOnInferenceThread(sch, params);
        return;
    }
    // ── media-gen fast path: the (stub) config's model_type marks an image/
    //    audio/video model (FLUX / Qwen3-TTS / LTX). Build the modality engine
    //    instead of the MLX safetensors path. Works for BOTH the startup
    //    gen-primary path (main builds the stub config) and the cold-load path
    //    (preloadCpuState builds it), since both dispatch off model_type.
    if (gen_mod.modalityFromType(params.config.model_type)) |modality| {
        try doLoadGenOnInferenceThread(sch, params, modality);
        return;
    }

    // GPU-memory pre-flight (MLX path). A Metal OOM during weight load / warmup
    // is thrown by MLX as a C++ exception that can't be caught across the C ABI,
    // so it terminates the whole process. Refuse the load up front instead, with
    // an actionable error, when free RAM clearly can't hold the weights + warmup
    // headroom — catches the common "restarted before the prior server released
    // its memory" case. Bypass with --skip-mem-preflight.
    if (!skip_mem_preflight) {
        const weights_bytes = modelDiskBytes(sch.io, params.model_dir);
        const avail_bytes = effectiveAvailableBytes(status.getAvailableMemBytes(), status.getProcAvailableMemBytes());
        log.info("[preflight] weights ~{d:.2} GB, available {d:.2} GB\n", .{
            @as(f64, @floatFromInt(weights_bytes)) / (1024.0 * 1024.0 * 1024.0),
            @as(f64, @floatFromInt(avail_bytes)) / (1024.0 * 1024.0 * 1024.0),
        });
        if (memInsufficientForLoad(weights_bytes, avail_bytes)) {
            const gb = 1024.0 * 1024.0 * 1024.0;
            log.err("Insufficient memory to load model: weights ~{d:.1} GB but only {d:.1} GB free. Close other models/apps (or wait for a prior mlx-serve to fully exit) and retry; pass --skip-mem-preflight to override.\n", .{
                @as(f64, @floatFromInt(weights_bytes)) / gb,
                @as(f64, @floatFromInt(avail_bytes)) / gb,
            });
            return error.InsufficientMemory;
        }
    }

    // Allocate the drafter_path dupe up front so the post-publish step
    // (lower down) has no fallible operations — once we start assigning
    // pointers onto `params.entry`, an OOM during a dupe would leave the
    // entry holding pointers that the per-ptr errdefers would double-free.
    var drafter_path_owned: []u8 = &[_]u8{};
    errdefer if (drafter_path_owned.len > 0) sch.allocator.free(drafter_path_owned);
    if (params.drafter_dir.len > 0) {
        drafter_path_owned = try sch.allocator.dupe(u8, params.drafter_dir);
    }

    // Weights — first mlx call. Binds the stream on this thread.
    const weights_ptr = try sch.allocator.create(Weights);
    errdefer sch.allocator.destroy(weights_ptr);
    weights_ptr.* = if (params.load_vision)
        try model_mod.loadWeightsWithVision(sch.io, sch.allocator, params.model_dir)
    else
        try model_mod.loadWeights(sch.io, sch.allocator, params.model_dir);
    errdefer weights_ptr.deinit();

    // Transformer — owns the bulk of the GPU memory.
    const xfm_ptr = try sch.allocator.create(Transformer);
    errdefer sch.allocator.destroy(xfm_ptr);
    xfm_ptr.* = try Transformer.init(sch.io, sch.allocator, params.config.*, weights_ptr);
    errdefer xfm_ptr.deinit();

    // Propagate the kv-quant config to the Transformer's own cache. Slot
    // caches in serve mode honor this independently in `Slot.init`; this
    // call covers any path that still touches `xfm.cache` directly (legacy
    // single-slot fallbacks, prompt-cache reuse).
    if (params.kv_quant_config.scheme != .off) {
        xfm_ptr.cache.deinit();
        xfm_ptr.cache = try KVCache.initWithConfigAndHeadDim(sch.allocator, params.config.num_hidden_layers, params.kv_quant_config, params.config.head_dim);
    }

    // Wire model weights into GPU memory (prevents paging, matches mlx-lm).
    {
        var dev = mlx.mlx_device{ .ctx = null };
        _ = mlx.mlx_get_default_device(&dev);
        var info = mlx.mlx_device_info_new();
        if (mlx.mlx_device_info_get(&info, dev) == 0) {
            var max_rec: usize = 0;
            if (mlx.mlx_device_info_get_size(&max_rec, info, "max_recommended_working_set_size") == 0 and max_rec > 0) {
                var old_limit: usize = 0;
                _ = mlx.mlx_set_wired_limit(&old_limit, max_rec);
                log.debug("Wired limit set to {d} MB\n", .{max_rec / (1024 * 1024)});
            }
            _ = mlx.mlx_device_info_free(info);
        }
    }

    // JIT-compile activation kernels. These are bound to THIS thread's mlx
    // stream — that's exactly the point of doing them here. Skipped entirely
    // when there's no GPU backend (iOS Simulator's CPU-only MLX): mlx_compile
    // is Metal kernel fusion and requests a GPU stream that doesn't exist; the
    // runtime falls back to the uncompiled paths when compiled_* stays null.
    if (!mlx.noGpuBackend()) {
        if (params.config.hidden_act == .gelu_approx) {
            xfm_ptr.compileGelu();
            xfm_ptr.compileGeglu();
        }
        if (params.config.final_logit_softcapping > 0.0) {
            xfm_ptr.compileSoftcap();
        }
        if (xfm_ptr.moe_layers != null) {
            xfm_ptr.compileMoeRouting();
        }
        if (params.config.linear_num_key_heads > 0) {
            xfm_ptr.compileGdnGate();
        }
    }

    // Phase 2 experiment: opt-in full-forward Metal fusion via
    // MLX_SERVE_COMPILE_FORWARD=1. This wraps the entire forward pass in
    // mlx_compile so the chunked-prefill loop dispatches a fused graph
    // instead of ~hundreds of separate ops per chunk. Gated because
    // (a) the compiled closure captures `xfm.cache` / `xfm.ssm_entries`
    // as state, and any path that swaps those (multi-slot scheduler,
    // future re-entrant callers) must verify they're not racing the
    // compiled call; (b) mlx_compile with shapeless=false recompiles
    // per unique input shape, which thrashes if the prefill loop sees
    // many different chunk sizes. Tied to byte-equivalence pin in
    // tests/test_phase2_forward_equivalence.sh.
    if (std.c.getenv("MLX_SERVE_COMPILE_FORWARD") != null) {
        const raw = std.c.getenv("MLX_SERVE_COMPILE_FORWARD").?;
        const slice = std.mem.sliceTo(raw, 0);
        if (std.mem.eql(u8, slice, "1")) {
            xfm_ptr.compileForward();
        }
    }

    // Vision encoder if requested. `MissingVisionWeights` is a benign opt-out
    // (model declares vision in config but the safetensors didn't ship the
    // tower); other errors fail the whole load.
    var vision_ptr: ?*VisionEncoder = null;
    if (params.load_vision) {
        const v = try sch.allocator.create(VisionEncoder);
        if (VisionEncoder.init(sch.allocator, params.config.*, weights_ptr)) |encoder| {
            v.* = encoder;
            vision_ptr = v;
        } else |err| {
            sch.allocator.destroy(v);
            if (err == error.MissingVisionWeights) {
                log.warn("Vision weights missing — vision disabled (model may have been quantized without vision tower)\n", .{});
            } else {
                return err;
            }
        }
    }
    errdefer if (vision_ptr) |v| {
        v.deinit();
        sch.allocator.destroy(v);
    };

    // Drafter (optional). Loaded only when `drafter_dir` is non-empty.
    var drafter_ptr: ?*DrafterModel = null;
    if (params.drafter_dir.len > 0) {
        const d = try sch.allocator.create(DrafterModel);
        d.* = drafter_mod.loadDrafter(sch.io, sch.allocator, mlx.gpuStream(), params.drafter_dir) catch |err| {
            sch.allocator.destroy(d);
            log.err("Failed to load drafter at {s}: {s}\n", .{ params.drafter_dir, @errorName(err) });
            return err;
        };
        d.bind(xfm_ptr) catch |err| {
            d.deinit();
            sch.allocator.destroy(d);
            log.err(
                "Drafter checkpoint at {s} is incompatible with target: {s}\n" ++
                    "  (drafter+target must share backbone_hidden_size, vocab_size, and have\n" ++
                    "  matching layer types in the target's non-shared K/V layers)\n",
                .{ params.drafter_dir, @errorName(err) },
            );
            return err;
        };
        drafter_ptr = d;

        // Auto-detect block_size unless the user pinned it explicitly.
        if (!params.draft_block_size_explicit) {
            const auto_bs = drafter_mod.recommendedBlockSize(params.config);
            sch.drafter_block_size = auto_bs;
            log.info(
                "Drafter ready (block_size={d}, auto-detected for {s}/{d}-layer{s}).\n",
                .{
                    auto_bs,
                    params.config.model_type,
                    params.config.num_hidden_layers,
                    if (params.config.isMoe()) ",moe" else "",
                },
            );
        } else {
            log.info("Drafter ready (block_size={d}, user override).\n", .{params.draft_block_size});
        }

        if (params.config.isMoe()) {
            log.warn(
                "Drafter loaded but target is MoE ({s}); per-request " ++
                    "enable_drafter defaults to OFF — drafter+MoE regresses " ++
                    "at single-stream batch=1 (verify forward expert-routing " ++
                    "penalty). Pass enable_drafter:true per request to opt-in.\n",
                .{params.config.model_type},
            );
        }
    }
    errdefer if (drafter_ptr) |d| {
        d.deinit();
        sch.allocator.destroy(d);
    };

    // Qwen native MTP head (optional). Auto-loaded when the model dir ships
    // an `mtp/weights.safetensors` sidecar that binds to this trunk; a
    // failed load or bind only disables the head — the model still serves.
    var mtp_ptr: ?*mtp_mod.MtpModel = null;
    var mtp_cost_profile: mtp_mod.MtpCostProfile = .generic;
    if (params.mtp_enabled and mtp_mod.hasMtpSidecar(sch.io, params.model_dir)) {
        if (sch.allocator.create(mtp_mod.MtpModel)) |h| {
            if (mtp_mod.loadMtp(sch.io, sch.allocator, mlx.gpuStream(), params.model_dir)) |loaded| {
                h.* = loaded;
                if (h.bind(xfm_ptr)) {
                    mtp_ptr = h;
                    mtp_cost_profile = h.m5NaxCostProfile(xfm_ptr);
                    log.info("MTP head ready (depth={d}).\n", .{generate_mod.Generator.resolveMtpDepthCapForProfile(params.mtp_depth, mtp_cost_profile)});
                } else |bind_err| {
                    log.warn("MTP sidecar incompatible with target ({s}) — disabled.\n", .{@errorName(bind_err)});
                    h.deinit();
                    sch.allocator.destroy(h);
                }
            } else |load_err| {
                log.warn("Failed to load MTP sidecar: {s} — disabled.\n", .{@errorName(load_err)});
                sch.allocator.destroy(h);
            }
        } else |_| {}
    }
    errdefer if (mtp_ptr) |h| {
        h.deinit();
        sch.allocator.destroy(h);
    };

    // Eager warmup: faults weight pages + compiles the decode-path kernels
    // on this thread's stream. ~600-900 ms at boot but the first user request
    // skips a cold path — observed savings on Gemma 4 E4B 4-bit.
    if (params.warmup_eager) {
        const warmup_start = std.Io.Timestamp.now(sch.io, .awake);
        xfm_ptr.warmup() catch |err| {
            log.warn("Warmup failed ({s}); continuing without it — first request may be slow.\n", .{@errorName(err)});
        };
        const warmup_ns: u64 = @intCast(warmup_start.untilNow(sch.io, .awake).nanoseconds);
        log.info("Warmup complete ({d} ms).\n", .{warmup_ns / std.time.ns_per_ms});
    }

    // ── Phase 05: install everything onto the LoadedModel entry, mark
    //    ready, and update the scheduler's borrowed views. The registry
    //    mutex guards the state transition + `current_resident_bytes`
    //    accounting; `state_cond.broadcast` (inside markReadyLocked) wakes
    //    any waiter blocked in ensureLoaded.
    //
    //    Everything below this comment must be infallible — once we begin
    //    assigning to `params.entry`, the per-ptr errdefers above would
    //    double-free if we error-return. `drafter_path_owned` was alloc'd
    //    up front for exactly this reason; the per-model prefix cache
    //    init is a struct literal (no fallible alloc).
    const entry = params.entry;
    entry.weights = weights_ptr;
    entry.transformer = xfm_ptr;
    entry.vision_encoder = vision_ptr;
    entry.drafter = drafter_ptr;
    entry.drafter_block_size = sch.drafter_block_size;
    entry.mtp = mtp_ptr;
    // Resolve the auto (0) cap here so every downstream reader of
    // `lm.mtp_depth` (server log lines, slot params) sees the real value.
    entry.mtp_depth = generate_mod.Generator.resolveMtpDepthCapForProfile(params.mtp_depth, mtp_cost_profile);
    entry.drafter_path = drafter_path_owned;
    drafter_path_owned = &[_]u8{}; // disarm the errdefer
    // Transfer ownership of the heap-allocated CPU state from `params` to
    // the entry. The caller (main.zig) MUST NOT free these — `LoadedModel.deinit`
    // walks them in the same `*X` pointer form they came in.
    entry.config = params.config;
    entry.tokenizer = params.tok;
    entry.chat_config = params.chat_config;
    // Per-model hot prefix cache (Plan 03 → Plan 05 move). Hybrid recurrent
    // archs are accepted iff `ssm_checkpoint_stride > 0` (Phase 1 of the
    // performance plan): with per-stride SSM checkpoints we can rewind both
    // KV and SSM to a snapshotted prefix; without them, divergence forces a
    // full reset, so we keep the legacy single-slot path for hybrid.
    const enable_ssm_cps = params.ssm_checkpoint_stride > 0;
    if (params.prefix_cache_capacity > 0 and
        prefix_cache_mod.HotPrefixCache.shouldUse(params.config, enable_ssm_cps))
    {
        entry.prefix_cache = prefix_cache_mod.HotPrefixCache.initWithMem(
            sch.allocator,
            params.prefix_cache_capacity,
            params.prefix_cache_mem_bytes,
        );
        // SSD tier (`--prefix-cache-disk`). Phase 3 persists hybrid recurrent
        // state too: the disk tier is allowed whenever the RAM tier accepted
        // the arch — i.e. pure-attention always, hybrid iff SSM checkpoints
        // are enabled (`enable_ssm_cps`, the same gate `shouldUse` applied).
        // Every failure mode is caught: persistence silently stays off, the
        // RAM cache is unaffected.
        const has_ssm_layers = params.config.has_hybrid_layers or
            params.config.full_attention_interval > 0;
        const disk_ok = !has_ssm_layers or enable_ssm_cps;
        if (params.prefix_cache_disk_bytes > 0 and disk_ok) attach: {
            const fp = kv_disk_cache.modelFingerprint(sch.allocator, sch.io, entry.path) catch |err| {
                log.warn("[disk-cache] fingerprint failed: {s} — persistence off for this model\n", .{@errorName(err)});
                break :attach;
            };
            defer sch.allocator.free(fp);
            const base = kv_disk_cache.defaultBaseDir(sch.allocator) catch break :attach;
            defer sch.allocator.free(base);
            entry.prefix_cache.?.disk = kv_disk_cache.DiskTier.init(
                sch.allocator,
                sch.io,
                base,
                fp,
                params.prefix_cache_disk_bytes,
                kv_disk_cache.DEFAULT_CHUNK_TOKENS,
            ) catch |err| {
                log.warn("[disk-cache] init failed: {s} — persistence off for this model\n", .{@errorName(err)});
                break :attach;
            };
        }
        entry.ssm_checkpoint_stride = params.ssm_checkpoint_stride;
        entry.ssm_checkpoint_max = params.ssm_checkpoint_max;
    }
    // Iteration 2 (perf-plan Phase 4 #3): tokenize cache for warm-path
    // chat-template renders. Applies to MLX, ds4, and llama engines —
    // they all funnel through `chat_mod.formatChat` /
    // `encodeChatViaDs4` / `encodeChatViaLlama` at the handler boundary.
    // Default capacity is small (4 entries) because chat conversations
    // mutate the messages list every turn; the goal is to catch warm
    // reuse benches and repeated agent-loop probes, not to memoize a
    // full session.
    if (params.tokenize_cache_entries > 0) {
        entry.tokenize_cache = tokenize_cache_mod.TokenizeCache.init(
            sch.allocator,
            params.tokenize_cache_entries,
        );
    }
    // Iteration 3-5: cap for the llama.cpp multi-session LRU. Always
    // clamp to ≥1 so `runPrefillLlama` can grow the cache even if a
    // bug or a 0-default leaks through.
    entry.llama_cache_max_entries = if (params.llama_cache_entries > 0)
        params.llama_cache_entries
    else
        1;
    // Phase 5 #2: thread KV-quant types onto the LoadedModel so
    // runPrefillLlama uses them when creating the persistent session.
    entry.llama_kv_type_k = params.llama_kv_type_k;
    entry.llama_kv_type_v = params.llama_kv_type_v;

    // Best-effort bytes_resident estimate: prefer the disk size hint when
    // available (it's close to actual GPU resident bytes after Metal page-
    // ins), else fall back to a rough multiple of layers × hidden. The
    // value drives LRU eviction's "will the new model fit?" gate in Phase
    // D; precise accounting isn't required here.
    const bytes_resident: u64 = if (entry.bytes_on_disk) |b|
        b
    else
        @as(u64, params.config.num_hidden_layers) * @as(u64, params.config.hidden_size) * 4 * 4;

    sch.registry.mutex.lockUncancelable(sch.io);
    sch.registry.markReadyLocked(entry, bytes_resident);
    sch.registry.mutex.unlock(sch.io);

    // Set borrowed views for scheduler-internal code (and `current_model`
    // for ensureLoaded → request handlers in Phase C).
    sch.current_model = entry;
    sch.weights = weights_ptr;
    sch.xfm = xfm_ptr;
    sch.vision_encoder = vision_ptr;
    sch.drafter = drafter_ptr;
    if (entry.prefix_cache) |*hc| sch.hot_prefix_cache = hc;
}

fn inferenceLoop(ctx: ThreadCtx) void {
    const sch = ctx.scheduler;
    const params = ctx.params;

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
        if (doLoadOnInferenceThread(sch, params)) |_| {
            log.info("Model ready (loaded on inference thread).\n", .{});
            signalStarted(sch);
        } else |err| {
            recordLoadError(sch, @errorName(err));
            sch.registry.mutex.lockUncancelable(sch.io);
            sch.registry.markErrorLocked(params.entry, @errorName(err));
            sch.registry.mutex.unlock(sch.io);
            signalStarted(sch);
            return;
        }
    }

    while (!sch.shutdown.load(.acquire)) {
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
        // Phase D: cold-load drain. Process ONE load per tick — loading a
        // model is heavy (~seconds; weight read + JIT compile + warmup)
        // and we want the rest of the inference loop to stay responsive.
        // Other pending loads wait in queue and get picked up next tick.
        var load_req: ?*LoadRequest = null;
        // Media-gen + unload work items (one per tick, like load — both are
        // heavy and we re-check the loop between them). Gen runs to completion
        // synchronously, blocking decode for its duration.
        var gen_req: ?*GenRequest = null;
        var unload_req: ?*UnloadRequest = null;
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
            if (sch.load_queue.items.len > 0) {
                load_req = sch.load_queue.orderedRemove(0);
            }
            if (sch.unload_queue.items.len > 0) {
                unload_req = sch.unload_queue.orderedRemove(0);
            }
            if (sch.gen_queue.items.len > 0) {
                gen_req = sch.gen_queue.orderedRemove(0);
            }
        }
        for (cleanup_batch[0..cleanup_n]) |s| s.deinit();
        if (vision_n > 0 or embed_n > 0) {
            for (vision_batch[0..vision_n]) |req| runVisionEncode(sch, req);
            for (embed_batch[0..embed_n]) |req| runEmbedRequest(sch, req);
        }
        if (load_req) |req| runLoadRequest(sch, req);
        if (unload_req) |req| runUnloadRequest(sch, req);
        if (gen_req) |req| runGenRequest(sch, req);

        // 1. Wait for work. Drain pending slots into a local list under lock,
        //    run prefills outside the lock.
        var to_prefill: [16]*Slot = undefined;
        var n_prefill: usize = 0;
        {
            sch.queue_mu.lockUncancelable(sch.io);
            defer sch.queue_mu.unlock(sch.io);
            while (sch.pending.items.len == 0 and sch.decoding.items.len == 0 and sch.vision_queue.items.len == 0 and sch.embed_queue.items.len == 0 and sch.cleanup_queue.items.len == 0 and sch.load_queue.items.len == 0 and sch.gen_queue.items.len == 0 and sch.unload_queue.items.len == 0 and !sch.shutdown.load(.acquire)) {
                sch.queue_cond.waitUncancelable(sch.io, &sch.queue_mu);
            }
            if (sch.shutdown.load(.acquire)) break;

            // If only vision/embed/cleanup/load work is pending, loop back to drain it.
            if (sch.pending.items.len == 0 and sch.decoding.items.len == 0) continue;

            while (n_prefill < to_prefill.len and sch.pending.items.len > 0) {
                to_prefill[n_prefill] = sch.pending.orderedRemove(0);
                n_prefill += 1;
            }
        }

        // 2. Prefill each pending slot (heavy; mlx ops on this thread).
        //    The inference thread is the sole mlx caller post-cleanup, so
        //    no per-tick stream rebind / mutex coexistence is needed.
        if (n_prefill > 0) {
            for (to_prefill[0..n_prefill]) |slot| {
                if (slot.cancelled.load(.acquire)) {
                    // finishSlot (not raw markFinished) so the metrics sink
                    // counts the cancellation; safe pre-prefill — commit
                    // no-ops with legacy_gen==null.
                    finishSlot(sch, slot, "cancelled");
                    continue;
                }
                var prefill_sw = io_util.Stopwatch.init(sch.io);
                runPrefill(sch, slot) catch |err| {
                    if (err == error.Cancelled) {
                        // Client vanished mid-prefill (conn thread noticed on
                        // an idle keepalive probe and set slot.cancelled);
                        // the chunk loop aborted. A clean finish, not an error.
                        log.info("[scheduler] prefill aborted: client disconnected\n", .{});
                        finishSlot(sch, slot, "cancelled");
                        continue;
                    }
                    log.err("[scheduler] prefill failed for slot: {s}\n", .{@errorName(err)});
                    slot.markError(@errorName(err));
                    continue;
                };
                slot.prefill_ns = prefill_sw.read();
                // Exact time-to-first-token: elapsed from request arrival
                // (Slot.init, pre-queue-wait) to prefill completion. Captured
                // here rather than derived by subtraction in finishSlot, so a
                // slot that finishes mid-tick can't skew it (metrics TTFT fix).
                slot.first_token_ns = @intCast(slot.request_start_ts.untilNow(sch.io, .boot).nanoseconds);
                sch.queue_mu.lockUncancelable(sch.io);
                sch.decoding.append(sch.allocator, slot) catch |err| {
                    sch.queue_mu.unlock(sch.io);
                    slot.markError(@errorName(err));
                    continue;
                };
                sch.queue_mu.unlock(sch.io);
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
            }
        }

        // 4. Decode tick. Charge the full wall-clock tick time to each
        //    participating slot — for batched ticks this matches the per-slot
        //    throughput a user actually observes (their stream advances at
        //    the tick cadence regardless of how many peers share it).
        if (active.items.len > 0) {
            var decode_sw = io_util.Stopwatch.init(sch.io);
            runDecodeTick(sch, active.items) catch |err| {
                log.err("[scheduler] decode tick failed: {s}\n", .{@errorName(err)});
                for (active.items) |s| s.markError(@errorName(err));
            };
            const tick_ns = decode_sw.read();
            for (active.items) |s| s.decode_ns +|= tick_ns;
        }

        // 5. Cull finished / errored / cancelled from `decoding`. The slot
        //    still belongs to its connection thread until that thread calls
        //    `complete`; we just stop touching it.
        {
            sch.queue_mu.lockUncancelable(sch.io);
            defer sch.queue_mu.unlock(sch.io);
            var i: usize = 0;
            while (i < sch.decoding.items.len) {
                const s = sch.decoding.items[i];
                const drop = s.cancelled.load(.acquire) or s.finished or s.error_code != null;
                if (drop) {
                    _ = sch.decoding.orderedRemove(i);
                } else i += 1;
            }
        }
    }
}

/// Phase A4: encode one or more images on the inference thread. Mirrors the
/// existing `processVisionImages` shape but writes the result into a request
/// struct + signals done, so the conn thread (blocked in `encodeVision`)
/// gets the output. On error, sets `req.error_name` and still signals done.
/// Plan 05 Phase D: routes the encode through `req.model.vision_encoder`,
/// not the scheduler's borrowed-view singleton — each LoadedModel has its
/// own vision encoder when applicable.
fn runVisionEncode(sch: *Scheduler, req: *VisionEncodeRequest) void {
    const vision_enc = req.model.vision_encoder orelse {
        finishVisionRequest(sch, req, "VisionEncoderNotLoaded");
        return;
    };
    if (req.images.len == 0 and req.audio.len == 0) {
        finishVisionRequest(sch, req, "EmptyImages");
        return;
    }

    // Encode all soft tokens into `emb_parts`: vision first, then audio, so the
    // single splice channel scatters them in the same order as the placeholder
    // blocks the conn thread injected (image block before audio block).
    var emb_parts = std.ArrayList(mlx.mlx_array).empty;
    defer emb_parts.deinit(req.allocator);
    const failParts = struct {
        fn f(s: *Scheduler, r: *VisionEncodeRequest, parts: []mlx.mlx_array, name: []const u8) void {
            for (parts) |e| _ = mlx.mlx_array_free(e);
            finishVisionRequest(s, r, name);
        }
    }.f;

    var n_vision: usize = 0;
    for (req.images) |img| {
        var emb: mlx.mlx_array = undefined;
        if (img.grid_h > 0) {
            // Qwen3-VL: pixels hold merge-order pixel_values [N, feat]; the ViT
            // produces [1, N/merge², out_hidden].
            const n: usize = @as(usize, img.grid_h) * img.grid_w;
            const feat: usize = (img.pixels.len / 4) / n;
            const shape = [_]c_int{ @intCast(n), @intCast(feat) };
            const pixel_arr = mlx.mlx_array_new_data(img.pixels.ptr, &shape, 2, .float32);
            defer _ = mlx.mlx_array_free(pixel_arr);
            emb = vision_enc.forwardQwen(pixel_arr, img.grid_h, img.grid_w) catch |err| {
                failParts(sch, req, emb_parts.items, @errorName(err));
                return;
            };
        } else {
            const h: c_int = @intCast(img.height);
            const w: c_int = @intCast(img.width);
            const shape = [_]c_int{ 1, 3, h, w };
            const pixel_arr = mlx.mlx_array_new_data(img.pixels.ptr, &shape, 4, .float32);
            defer _ = mlx.mlx_array_free(pixel_arr);
            emb = vision_enc.forward(pixel_arr) catch |err| {
                failParts(sch, req, emb_parts.items, @errorName(err));
                return;
            };
        }
        const es = mlx.getShape(emb);
        n_vision += @intCast(es[1]);
        emb_parts.append(req.allocator, emb) catch |err| {
            _ = mlx.mlx_array_free(emb);
            failParts(sch, req, emb_parts.items, @errorName(err));
            return;
        };
    }

    // Audio: frame each clip into 640-sample tokens, project through the
    // unified audio embedder → [1, n_frames, hidden].
    var n_audio: usize = 0;
    for (req.audio) |clip| {
        const n_samples = clip.len / 4;
        if (n_samples == 0) continue;
        const cfg = req.model.config orelse {
            failParts(sch, req, emb_parts.items, "NoConfig");
            return;
        };
        const samples_per_token: usize = if (cfg.audio_samples_per_token > 0) cfg.audio_samples_per_token else 640;
        const n_frames = (n_samples + samples_per_token - 1) / samples_per_token;
        const padded_len = n_frames * samples_per_token;
        const buf = req.allocator.alloc(f32, padded_len) catch |err| {
            failParts(sch, req, emb_parts.items, @errorName(err));
            return;
        };
        @memset(buf, 0);
        @memcpy(std.mem.sliceAsBytes(buf)[0..clip.len], clip);
        const shape = [_]c_int{ 1, @intCast(n_frames), @intCast(samples_per_token) };
        const frames_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 3, .float32);
        req.allocator.free(buf); // mlx_array_new_data copies into an array-owned buffer
        defer _ = mlx.mlx_array_free(frames_arr);
        const emb = vision_enc.forwardAudio(frames_arr) catch |err| {
            failParts(sch, req, emb_parts.items, @errorName(err));
            return;
        };
        n_audio += n_frames;
        emb_parts.append(req.allocator, emb) catch |err| {
            _ = mlx.mlx_array_free(emb);
            failParts(sch, req, emb_parts.items, @errorName(err));
            return;
        };
    }

    if (emb_parts.items.len == 0) {
        finishVisionRequest(sch, req, "EmptyImages");
        return;
    }

    // Single modality/clip: pass through. Multiple: concatenate along token dim.
    var combined: mlx.mlx_array = undefined;
    if (emb_parts.items.len == 1) {
        combined = emb_parts.items[0];
        emb_parts.items[0] = mlx.mlx_array_new(); // sentinel so the deferred-free path is a no-op
    } else {
        const cat_vec = mlx.mlx_vector_array_new_data(emb_parts.items.ptr, emb_parts.items.len);
        defer _ = mlx.mlx_vector_array_free(cat_vec);
        combined = mlx.mlx_array_new();
        if (mlx.mlx_concatenate_axis(&combined, cat_vec, 1, vision_enc.s) != 0) {
            _ = mlx.mlx_array_free(combined);
            failParts(sch, req, emb_parts.items, "ConcatenateFailed");
            return;
        }
    }
    for (emb_parts.items) |e| _ = mlx.mlx_array_free(e);

    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    req.result = combined;
    req.n_vision_tokens = n_vision;
    req.n_audio_tokens = n_audio;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

fn finishVisionRequest(sch: *Scheduler, req: *VisionEncodeRequest, err_name: []const u8) void {
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    if (req.error_name) |old| req.allocator.free(old);
    req.error_name = req.allocator.dupe(u8, err_name) catch null;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

/// Service one embedding request on the inference thread. Runs the batched
/// encoder-only forward pass via `generate.computeEmbeddingsBatch(xfm, ...)`,
/// resets the global xfm.cache between requests (encoder-only does not
/// share KV state across embeddings), and wakes the conn thread.
fn runEmbedRequest(sch: *Scheduler, req: *EmbedRequest) void {
    const xfm_ptr = req.model.transformer.?;
    xfm_ptr.resetCache() catch |err| {
        finishEmbedRequest(sch, req, @errorName(err));
        return;
    };
    const results = generate_mod.computeEmbeddingsBatch(req.allocator, xfm_ptr, req.token_seqs) catch |err| {
        finishEmbedRequest(sch, req, @errorName(err));
        return;
    };
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    req.results = results;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

fn finishEmbedRequest(sch: *Scheduler, req: *EmbedRequest, err_name: []const u8) void {
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    if (req.error_name) |old| req.allocator.free(old);
    req.error_name = req.allocator.dupe(u8, err_name) catch null;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

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
fn runLoadRequest(sch: *Scheduler, req: *LoadRequest) void {
    // Step 1: evict victims (if any) BEFORE the load, so peak GPU residency
    // never holds the old + new model at once. unloadResident() drops
    // mlx_arrays — same thread-stream invariant as cleanup_queue drain.
    for (req.evict_entries) |victim| {
        const victim_bytes = victim.bytes_resident; // unloadResident zeroes it
        log.info("[registry] evicting model id={s} ({d:.2} GB resident)\n", .{
            victim.id,
            @as(f64, @floatFromInt(victim_bytes)) / 1_073_741_824.0,
        });
        victim.unloadResident();
        sch.registry.mutex.lockUncancelable(sch.io);
        sch.registry.accountEvictedLocked(victim_bytes);
        sch.registry.finalizeEvictionLocked(victim);
        sch.registry.mutex.unlock(sch.io);
    }

    // Step 2: the actual load. On error, mark .error_state and signal done
    // (conn thread frees pre-parsed CPU state — ownership stays on req on
    // the failure path).
    doLoadOnInferenceThread(sch, req) catch |err| {
        log.err("[registry] load failed for model id={s}: {s}\n", .{ req.entry.id, @errorName(err) });
        sch.registry.mutex.lockUncancelable(sch.io);
        // markErrorLocked dupes the error name onto the entry; the
        // conn thread reads it back from the entry, not from req.
        sch.registry.markErrorLocked(req.entry, @errorName(err));
        sch.registry.mutex.unlock(sch.io);
        finishLoadRequest(sch, req, @errorName(err));
        return;
    };

    log.info("[registry] model id={s} ready ({d:.2} GB resident)\n", .{
        req.entry.id,
        @as(f64, @floatFromInt(req.entry.bytes_resident)) / 1_073_741_824.0,
    });
    finishLoadRequest(sch, req, null);
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

/// Run one media-generation job on the inference thread. The job body
/// (`req.run`) does all mlx work + writes the HTTP/SSE response to the parked
/// connection. We bracket it with the model's `gen_busy` flag for visibility
/// and signal `done` so the conn thread in `runGeneration` wakes.
fn runGenRequest(sch: *Scheduler, req: *GenRequest) void {
    req.model.gen_busy = true;
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
    req.run(req.ctx);
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
    _ = mlx.mlx_clear_cache();
    req.done_mu.lockUncancelable(sch.io);
    req.done = true;
    req.done_cond.broadcast(sch.io);
    req.done_mu.unlock(sch.io);
}

/// Free a model's resident mlx state on the inference thread (stream-bound,
/// same invariant as the cleanup-queue drain) and finalize the eviction
/// accounting. The conn thread already marked the entry `.evicting` and
/// drained its refcount in `unloadModel`.
fn runUnloadRequest(sch: *Scheduler, req: *UnloadRequest) void {
    const entry = req.entry;
    const bytes = entry.bytes_resident; // unloadResident zeroes it
    log.info("[registry] unloading model id={s} ({d:.2} GB resident)\n", .{
        entry.id,
        @as(f64, @floatFromInt(bytes)) / 1_073_741_824.0,
    });
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
        sch.hot_prefix_cache = null;
    }
    sch.registry.mutex.lockUncancelable(sch.io);
    sch.registry.accountEvictedLocked(bytes);
    sch.registry.finalizeEvictionLocked(entry);
    sch.registry.mutex.unlock(sch.io);

    req.done_mu.lockUncancelable(sch.io);
    req.done = true;
    req.done_cond.broadcast(sch.io);
    req.done_mu.unlock(sch.io);
}

/// Phase A6: commit a successfully completed slot's KV cache to the hot
/// prefix cache. Called from the inference thread BEFORE `markFinished`
/// broadcasts, so the slot is still alive (the conn thread is blocked in
/// `waitNext`). Skipped for pad-only generations, vision-bearing slots
/// (stale embeddings would be reused), and slots with no generated tokens.
fn commitSlotIfApplicable(sch: *Scheduler, slot: *Slot) void {
    // Phase D: per-model prefix cache — read off the slot's LoadedModel.
    const hc: *prefix_cache_mod.HotPrefixCache = if (slot.model.prefix_cache) |*p| p else return;
    if (slot.was_pad_only) return;
    if (slot.error_code != null) return;
    if (slot.vision_embeddings != null) return;
    const gen_ptr = if (slot.legacy_gen) |*g| g else return;
    if (gen_ptr.generated_ids.items.len == 0) return;

    // Construct the full token sequence: the original prompt + everything
    // generated this turn. The cache reflects exactly this state — Generator
    // forwarded each emitted token into slot.cache as it was sampled.
    const total_len = slot.full_prompt.len + gen_ptr.generated_ids.items.len;
    const total_tokens = sch.allocator.alloc(u32, total_len) catch return;
    defer sch.allocator.free(total_tokens);
    @memcpy(total_tokens[0..slot.full_prompt.len], slot.full_prompt);
    @memcpy(total_tokens[slot.full_prompt.len..], gen_ptr.generated_ids.items);

    // Phase 1: drain any SSM checkpoints captured by the Generator's prefill
    // loop and hand them to the cache alongside the KV snapshot. For plain-
    // attn models this returns an empty slice (no allocator hit). Ownership
    // transfers to the cache via `commitWithSsm`; freeing happens on
    // eviction.
    const ssm_cps_slice = gen_ptr.takeSsmCheckpoints();
    const ssm_cps_opt: ?[]transformer_mod.SSMCheckpoint = if (ssm_cps_slice.len > 0) ssm_cps_slice else null;
    if (ssm_cps_slice.len == 0 and gen_ptr.ssm_checkpoint_alloc != null) {
        // Empty list — free the (zero-length) slice we got back so the
        // allocator's bookkeeping stays clean.
        gen_ptr.ssm_checkpoint_alloc.?.free(ssm_cps_slice);
    }
    hc.commitWithSsm(&slot.cache, total_tokens, slot.has_tools, ssm_cps_opt) catch |err| {
        log.warn("[hot-cache] commit failed: {s}\n", .{@errorName(err)});
        // Commit failed — we still own the checkpoints. Free them so they
        // don't leak.
        if (ssm_cps_opt) |cps| {
            const a = gen_ptr.ssm_checkpoint_alloc orelse sch.allocator;
            for (cps) |*cp| cp.deinit(a);
            a.free(cps);
        }
    };
}

/// Phase A6: finalize a slot. Commits to hot prefix cache (if applicable)
/// before signaling completion. The order matters: commit first while the
/// slot is alive, then markFinished — the conn thread's waitNext might
/// return immediately after the broadcast and call complete()→deinit, so
/// we cannot reach into the slot afterwards.
/// Pure decision for the degenerate-tail-loop guard: when the generated tail
/// has collapsed into a short repeating cycle, returns the finish reason to
/// cut the request with; null while generation is healthy.
///
/// The reason is "length", NOT "stop": the SERVER is truncating the generation
/// (the model didn't finish — we cut a runaway loop), and "length" is the one
/// reason server.toolCallFinishReason preserves through tool-call parsing, so
/// a call salvaged from the cut buffer reaches the client as a TRUNCATION and
/// its recovery fires. Live 2026-07-14 (plang/php.html): "stop" became
/// "tool_calls", presenting a server-cut fragment as a model-completed write.
pub fn loopStopReason(generated_ids: []const u32) ?[]const u8 {
    if (generate_mod.isDegenerateTailLoop(
        generated_ids,
        generate_mod.degenerate_loop_max_period,
        generate_mod.degenerate_loop_reps,
    )) return "length";
    return null;
}

fn finishSlot(sch: *Scheduler, slot: *Slot, reason: []const u8) void {
    // Emit the `[spec-stats]` summary (no-op for non-speculative slots).
    // The legacy generate() path logs this itself; scheduler-driven slots
    // finalize here instead.
    if (slot.legacy_gen) |*g| g.logSpecStats();
    commitSlotIfApplicable(sch, slot);
    // SSD flush runs AFTER markFinished so the client never waits on the
    // chunk-append — but everything it needs must be captured BEFORE the
    // broadcast: the conn thread may complete()+free the slot immediately.
    // The prefix cache and stream live on the registry-owned LoadedModel,
    // which outlives the slot (unload also runs on this thread).
    const hc_opt: ?*prefix_cache_mod.HotPrefixCache =
        if (slot.model.prefix_cache) |*p| p else null;
    const stream_opt: ?mlx.mlx_stream =
        if (slot.model.transformer) |x| x.s else null;
    // Record per-request metrics while slot fields are still live (before
    // markFinished broadcasts — the conn thread may complete()+free the slot
    // immediately after). Off the per-token path; a null sink (metrics off) is
    // a single per-request branch. real_ttft = first_token_ns (queue+prefill,
    // captured exactly at prefill completion); recordRequest derives
    // e2e = first_token_ns + decode_ns.
    if (sch.metrics) |m| {
        m.recordRequest(
            reason,
            slot.first_token_ns,
            slot.prefill_ns,
            slot.decode_ns,
            slot.prompt_tokens,
            slot.completion_tokens,
            slot.cached_tokens,
        );
    }
    slot.markFinished(reason);
    if (hc_opt) |hc| {
        if (stream_opt) |s| hc.flushPendingDisk(s);
    }
}

/// ds4 prefill: create a session sized to the configured ctx and sync it to
/// the full prompt. ds4 internally reuses the common prefix between its live
/// session cache and the new prompt, so the mlx-serve hot prefix cache stays
/// out of the picture. `slot.prompt_tokens` reports the full prompt length;
/// `cached_tokens` is left at 0 (ds4 doesn't expose its per-session reuse
/// count back through the FFI).
fn runPrefillDs4(sch: *Scheduler, slot: *Slot, engine: *arch_ds4.Ds4Engine) !void {
    _ = sch;
    // Convert the slot's u32 prompt to ds4's i32 view. Sized once per
    // prefill — ds4's session_sync owns the read of these IDs and the
    // buffer can be freed before decode.
    const i32_prompt = try slot.allocator.alloc(i32, slot.full_prompt.len);
    defer slot.allocator.free(i32_prompt);
    for (slot.full_prompt, 0..) |t, i| i32_prompt[i] = @intCast(t);

    // Session ctx from the user's --ctx-size, which runDs4Serve carries on the
    // stub config's max_position_embeddings. Floored at ds4's prefill chunk so
    // an under-sized ctx can't drop into the junk-output regime; 0/unset →
    // ds4's default. Larger ctx → larger KV scratch up front.
    const req_ctx: u32 = if (slot.model.config) |c| c.max_position_embeddings else 0;
    const ctx_size: i32 = @intCast(arch_ds4.clampSessionCtx(req_ctx));
    var sess = try engine.createSession(ctx_size);
    errdefer sess.free();

    try sess.sync(i32_prompt);

    slot.ds4_session = sess;
    slot.prompt_tokens = @intCast(slot.full_prompt.len);
    slot.cached_tokens = 0;
    slot.state = .decoding;
}

/// llama.cpp prefill: drive a persistent per-model session, reusing the KV from
/// the previous request's shared prompt prefix (LM-Studio-style prompt caching).
/// `submit` guarantees a single slot owns the session at a time, so the resident
/// KV is exactly the prior request's prompt+generation. `sync` diffs the new
/// prompt against it, trims the divergent tail, and decodes only the suffix.
/// `cached_tokens` reports the reused prefix length; `prompt_tokens` stays the
/// full prompt so prefill tok/s reflects only the uncached suffix.
fn runPrefillLlama(sch: *Scheduler, slot: *Slot, engine: *arch_llama.LlamaEngine) !void {
    const i32_prompt = try slot.allocator.alloc(i32, slot.full_prompt.len);
    defer slot.allocator.free(i32_prompt);
    for (slot.full_prompt, 0..) |t, i| i32_prompt[i] = @intCast(t);

    // Size to the stub config's context length (main.zig sets it from the user's
    // --ctx-size or the GGUF's trained context). 0 → libllama uses the model
    // default (its trained context).
    const ctx_size: i32 = if (slot.model.config) |c| @intCast(c.max_position_embeddings) else 0;

    // Phase 5 #1 (Iteration 3-5): pick the best matching entry out of the
    // LRU. The "best" = longest common prefix between the incoming prompt
    // and the entry's resident KV mirror; ties (including the all-zero
    // case) go to the least-recently-used entry so a brand-new prompt
    // doesn't keep clobbering the same slot.
    const max_entries = if (slot.model.llama_cache_max_entries > 0)
        slot.model.llama_cache_max_entries
    else
        1;

    // Chat templates produce a fixed leading prefix (system header, BOS,
    // role markers) that's identical across requests — for Qwen3-style
    // it's ~3-10 tokens. Treating that as a "hit" would let request B
    // claim request A's slot just to save a handful of tokens, evicting
    // A's content-bearing KV. Require a higher floor before we count a
    // resident entry as a meaningful match. The value 16 sits above
    // every chat template's pure prologue in this codebase (Gemma=12,
    // Qwen=8, Llama=4) and below any real user-message overlap.
    const min_prefix_to_claim: usize = 16;

    var best_idx: ?usize = null;
    var best_shared: usize = 0;
    var lru_idx: ?usize = null;
    var lru_used: i64 = std.math.maxInt(i64);
    for (slot.model.llama_sessions.items, 0..) |entry, i| {
        const shared = arch_llama.commonPrefixLen(entry.session.resident.items, i32_prompt);
        // Strict >: ties leave the lower-indexed entry in `best_idx`, which
        // is fine — we still need the prefix-match candidate. The
        // separately tracked `lru_idx` handles the cold-miss path.
        if (shared > best_shared) {
            best_shared = shared;
            best_idx = i;
        }
        if (entry.last_used_ns < lru_used) {
            lru_used = entry.last_used_ns;
            lru_idx = i;
        }
    }

    // Promote the best match only when it crosses the chat-template floor;
    // otherwise fall through to growth / LRU eviction.
    if (best_shared < min_prefix_to_claim) best_idx = null;

    var pick_idx: usize = undefined;
    if (best_idx) |i| {
        pick_idx = i;
    } else if (slot.model.llama_sessions.items.len < max_entries) {
        // Grow the cache — every prefill so far missed; allocate a new
        // session and append it.
        const type_k = slot.model.llama_kv_type_k;
        const type_v = slot.model.llama_kv_type_v;
        const created = if (type_k != 0 or type_v != 0)
            try engine.createSessionWithKvQuant(ctx_size, type_k, type_v)
        else
            try engine.createSession(ctx_size);
        errdefer created.free();
        try slot.model.llama_sessions.append(slot.allocator, .{ .session = created, .last_used_ns = 0 });
        pick_idx = slot.model.llama_sessions.items.len - 1;
        log.info("[llama-cache] created session #{d} (cap={d})\n", .{ pick_idx, max_entries });
    } else {
        // Full + no prefix match — evict the LRU entry by resetting its KV
        // in place. Keeps the libllama context alive (re-allocating per
        // miss would be expensive) but drops the resident-token mirror so
        // the next sync starts from zero.
        pick_idx = lru_idx.?;
        slot.model.llama_sessions.items[pick_idx].session.reset();
        log.info("[llama-cache] evicted LRU session #{d}\n", .{pick_idx});
    }

    const entry_ptr = &slot.model.llama_sessions.items[pick_idx];
    entry_ptr.last_used_ns = @intCast(std.Io.Timestamp.now(sch.io, .boot).nanoseconds);
    const sess = entry_ptr.session;

    // `syncWithFallback` does the prefix-trim + suffix decode and, on any
    // libllama transient (the "failed to find a memory slot" class — see
    // `LlamaSession.syncWithFallback`), resets the session and retries once
    // cold. Either we serve the request with a clean response or we surface
    // the error after leaving the session in a known-good state.
    const cached = sess.syncWithFallback(i32_prompt) catch |err| {
        sess.reset();
        return err;
    };

    slot.llama_session = sess;
    slot.prompt_tokens = @intCast(slot.full_prompt.len);
    slot.cached_tokens = @intCast(cached);
    slot.state = .decoding;
}

/// ds4 decode tick: argmax (temp ≤ 0) or sample, check EOS, push token,
/// `eval(token)` to extend the session, and stop on max_tokens. Each call
/// emits exactly one token (unlike PLD/drafter which can emit several).
fn runDs4DecodeTick(sch: *Scheduler, slot: *Slot, session: *arch_ds4.Ds4Session) !void {
    const engine = slot.model.ds4_engine.?;
    const next_id: i32 = if (slot.sampling.temperature <= 0.0)
        session.argmax()
    else
        session.sample(
            slot.sampling.temperature,
            @intCast(slot.sampling.top_k),
            slot.sampling.top_p,
            0.05,
            &slot.ds4_rng,
        );

    // ds4 returns an i32 token id; we treat any negative value as a sampler
    // failure rather than push it through the unsigned ring buffer.
    if (next_id < 0) {
        slot.markError("ds4_sample_failed");
        return;
    }
    const tok_u32: u32 = @intCast(next_id);

    // EOS handling — match the MLX path: do NOT emit the stop token.
    if (next_id == engine.eosToken() or generate_mod.isEosId(tok_u32, slot.eos_token_ids)) {
        finishSlot(sch, slot, "stop");
        return;
    }

    // MTP speculative decode: ONE call commits the sampled token AND drafts +
    // verifies several more, advancing ds4's KV internally (no separate eval).
    // It emits `[sampled, accepted…]`, so this tick may push several tokens.
    // Mirrors ds4's own CLI loop; engages only under greedy sampling.
    if (ds4MtpShouldEngage(engine.hasMtp(), engine.mtpDraftTokens(), slot.sampling.temperature)) {
        var spec_buf: [DS4_MTP_MAX_TOKENS]i32 = undefined;
        const done: i64 = @intCast(slot.completion_tokens);
        const cap: i64 = @intCast(slot.max_tokens);
        const remaining: i32 = @intCast(@max(@as(i64, 1), cap - done));
        const n = session.evalSpeculative(next_id, remaining, engine.eosToken(), spec_buf[0..]) catch {
            slot.markError("ds4_spec_failed");
            return;
        };
        const n_usize: usize = if (n > 0) @intCast(n) else 0;
        for (spec_buf[0..n_usize]) |t| {
            const t_u32: u32 = @intCast(t);
            // EOS may appear mid-batch — stop, and never emit it.
            if (t == engine.eosToken() or generate_mod.isEosId(t_u32, slot.eos_token_ids)) {
                finishSlot(sch, slot, "stop");
                return;
            }
            slot.pushToken(t_u32);
            if (t_u32 != 0) slot.was_pad_only = false;
            slot.completion_tokens += 1;
            if (slot.completion_tokens >= slot.max_tokens) {
                finishSlot(sch, slot, "length");
                return;
            }
        }
        return;
    }

    slot.pushToken(tok_u32);
    if (tok_u32 != 0) slot.was_pad_only = false;
    slot.completion_tokens += 1;

    // Advance ds4's KV by feeding the freshly-sampled token. After this
    // the session is in the state expected by the NEXT decode tick.
    try session.eval(next_id);

    if (slot.completion_tokens >= slot.max_tokens) {
        finishSlot(sch, slot, "length");
        return;
    }
}

test "ds4MtpShouldEngage: draft head + >1 draft tokens + greedy" {
    // Engages only greedily (ds4's spec path is argmax) with a loaded head.
    try std.testing.expect(ds4MtpShouldEngage(true, 4, 0.0));
    try std.testing.expect(ds4MtpShouldEngage(true, 2, -1.0));
    // No head, or 1 draft token, or sampling (temp>0) → regular decode.
    try std.testing.expect(!ds4MtpShouldEngage(false, 4, 0.0));
    try std.testing.expect(!ds4MtpShouldEngage(true, 1, 0.0));
    try std.testing.expect(!ds4MtpShouldEngage(true, 4, 0.7));
}

/// llama.cpp decode tick: argmax (temp < 0.01, matching the MLX greedy
/// threshold) or sample, check EOS, push token, `eval(token)` to extend the
/// session, and stop on max_tokens. One token per call.
fn runLlamaDecodeTick(sch: *Scheduler, slot: *Slot, session: *arch_llama.LlamaSession) !void {
    const engine = slot.model.llama_engine.?;
    const next_id: i32 = if (slot.sampling.temperature < 0.01)
        session.argmax()
    else
        session.sample(
            slot.sampling.temperature,
            @intCast(slot.sampling.top_k),
            slot.sampling.top_p,
            0.0, // min_p disabled — matches the MLX sampler (top_k + top_p only)
            &slot.llama_rng,
        );

    if (next_id < 0) {
        slot.markError("llama_sample_failed");
        return;
    }
    const tok_u32: u32 = @intCast(next_id);

    // EOS / end-of-generation — like the MLX path, do NOT emit the stop token.
    if (engine.isEog(next_id) or generate_mod.isEosId(tok_u32, slot.eos_token_ids)) {
        finishSlot(sch, slot, "stop");
        return;
    }

    slot.pushToken(tok_u32);
    if (tok_u32 != 0) slot.was_pad_only = false;
    slot.completion_tokens += 1;

    // Advance the KV by feeding the freshly-sampled token.
    try session.eval(next_id);

    if (slot.completion_tokens >= slot.max_tokens) {
        finishSlot(sch, slot, "length");
        return;
    }
}

/// DiffusionGemma prefill: refresh the slot ctx, build the per-slot
/// diffusion Runner (which dequantizes the embedding table for
/// self-conditioning), and run the causal ENCODER pass over the full prompt
/// to fill the slot's KV cache. The hot prefix cache is intentionally NOT
/// consulted (v1): restored snapshots leave per-layer cache VIEWS stale, and
/// the diffusion decoder reads them via denseView before any update would
/// rebuild them.
fn runPrefillDiffusion(sch: *Scheduler, slot: *Slot) !void {
    _ = sch;
    slot.ctx.cache = &slot.cache;
    slot.ctx.moe_seq_offset = &slot.moe_seq_offset;
    slot.ctx.ssm_entries = slot.ssm_entries;
    slot.ctx.vision_embeddings = null; // vision tower not wired for this arch
    slot.ctx.capture_hidden = null;
    slot.ctx.kv_attn_fused = false;

    const xfm: *Transformer = slot.model.transformer.?;
    const runner = try slot.allocator.create(diffusion_mod.Runner);
    errdefer slot.allocator.destroy(runner);
    runner.* = try diffusion_mod.Runner.init(
        slot.allocator,
        xfm,
        &slot.ctx,
        slot.sampling.temperature,
        slot.max_tokens,
    );
    errdefer runner.deinit();
    runner.cancel_flag = &slot.cancelled;

    try runner.prefill(slot.full_prompt);

    slot.diffusion = runner;
    slot.prompt_tokens = @intCast(slot.full_prompt.len);
    slot.state = .decoding;
}

/// Diffusion decode tick: denoise and commit ONE canvas (≤ 48 decoder
/// forwards), then emit its tokens through the slot — block-wise streaming
/// falls out of the normal slot machinery. EOS inside the canvas finishes
/// the request without emitting the stop token (matching the AR paths); the
/// canvas remainder after EOS is discarded. The runner checks
/// `slot.cancelled` once per denoising step.
fn runDiffusionDecodeTick(sch: *Scheduler, slot: *Slot, runner: *diffusion_mod.Runner) !void {
    const result = runner.nextCanvas(slot.allocator) catch |err| switch (err) {
        error.Cancelled => return,
        else => return err,
    };
    if (result == null) {
        finishSlot(sch, slot, "length");
        return;
    }
    defer slot.allocator.free(result.?.tokens);
    for (result.?.tokens) |t| {
        if (slot.cancelled.load(.acquire)) return;
        if (generate_mod.isEosId(t, slot.eos_token_ids)) {
            finishSlot(sch, slot, "stop");
            return;
        }
        slot.pushToken(t);
        if (t != 0) slot.was_pad_only = false;
        slot.completion_tokens += 1;
        if (slot.completion_tokens >= slot.max_tokens) {
            finishSlot(sch, slot, "length");
            return;
        }
    }
}

/// Allocate the slot's KVCache state (already done in Slot.init), construct
/// the per-slot Generator via `Generator.initWithOptions(.{ .ctx = slot.ctx,
/// .skip_lazy_preforward = true_for_regular, ... })`, and store it on the
/// slot. After return, the slot is ready for decode ticks.
fn runPrefill(sch: *Scheduler, slot: *Slot) !void {
    // Mark the phase for the whole of prefill, and clear both signals on EVERY
    // exit path (success, cancel, error). `requests_prefilling` flips at entry
    // so the panel isn't blind until the first chunk lands; the chunk loop in
    // generate.zig stores absolute progress into `inflight_prefill_tokens`.
    //
    // Gated on `--metrics`, per the observability contract: when it's off the
    // prefill path executes NO extra instruction at all (the chunk loop's hook
    // is null too — see the `prefill_progress` option below).
    const observe = sch.metrics != null;
    if (observe) _ = sch.requests_prefilling.fetchAdd(1, .monotonic);
    defer if (observe) {
        _ = sch.requests_prefilling.fetchSub(1, .monotonic);
        sch.inflight_prefill_tokens.store(0, .monotonic);
    };

    // ds4-backed model: bypass the MLX prefill path entirely. The ds4
    // engine owns the chat/tokenizer/KV stack — we just create a session,
    // sync it to the slot's full prompt (ds4 reuses common prefix against
    // its live cache internally), and mark the slot decoding.
    if (slot.model.ds4_engine) |engine| {
        return runPrefillDs4(sch, slot, engine);
    }
    if (slot.model.llama_engine) |engine| {
        return runPrefillLlama(sch, slot, engine);
    }
    // DiffusionGemma: generation is a canvas-denoising loop, not
    // autoregressive decode — no Generator. The encoder prefill fills the
    // slot's own KV cache; PLD/drafter/MTP/batching never apply.
    if (slot.model.transformer.?.config.isDiffusion()) {
        return runPrefillDiffusion(sch, slot);
    }
    const sampling = slot.sampling;
    // Refresh ctx in case slot was relocated (paranoia — slot is heap so no,
    // but cheap).
    slot.ctx.cache = &slot.cache;
    slot.ctx.moe_seq_offset = &slot.moe_seq_offset;
    slot.ctx.ssm_entries = slot.ssm_entries;
    slot.ctx.vision_embeddings = slot.vision_embeddings;
    slot.ctx.mrope_pos = slot.mrope_pos;
    slot.ctx.mrope_total = slot.mrope_total;
    slot.ctx.mrope_delta = slot.mrope_delta;
    slot.ctx.capture_hidden = null;
    slot.ctx.kv_attn_fused = slot.kv_attn_fused;

    const use_mtp = slot.enable_mtp and slot.mtp != null;
    const use_drafter = !use_mtp and slot.enable_drafter and slot.drafter != null;
    const use_pld = !use_mtp and !use_drafter and slot.enable_pld;

    // Phase A6: prefill source-of-truth is `slot.full_prompt` — the conn
    // thread's `reuseKVCache` may have trimmed `slot.prompt_ids` based on
    // `xfm.cache` (the legacy global cache), but the slot has its own cache
    // which started empty. Using `slot.prompt_ids` would cause the model to
    // attend to only the trailing portion with empty cache, producing
    // garbage. Always start from the full prompt and let the hot prefix
    // cache (if configured) trim it back via the slot's own cache state.
    //
    // Vision-bearing slots: skip the hot cache altogether. Image tokens
    // have identical IDs but the underlying vision embeddings differ
    // per-request, so prefix matching would reuse stale features.
    var prefill_tokens: []const u32 = slot.full_prompt;
    var hot_matched: u32 = 0;
    // Phase D: per-slot model — pull transformer + prefix cache off the
    // slot's LoadedModel. Both stay resident for the slot's lifetime
    // because the conn thread holds a refcount on slot.model.
    const xfm_ptr: *Transformer = slot.model.transformer.?;
    if (slot.model.prefix_cache) |*hc| {
        if (slot.vision_embeddings == null) {
            const lookup = hc.lookupAndRestore(
                &slot.cache,
                &slot.moe_seq_offset,
                slot.ssm_entries,
                xfm_ptr.s,
                slot.full_prompt,
                slot.has_tools,
            ) catch |err| blk: {
                log.warn("[hot-cache] lookup failed: {s} — proceeding with cold prefill\n", .{@errorName(err)});
                break :blk prefix_cache_mod.LookupResult{ .matched = 0, .full_match = false };
            };
            if (lookup.matched > 0 and lookup.matched <= slot.full_prompt.len) {
                hot_matched = @intCast(lookup.matched);
                prefill_tokens = slot.full_prompt[hot_matched..];
            }
        }
    }

    // Phase 1 (perf-plan): forward the SSM-checkpoint stride from the
    // LoadedModel so the prefill loop snapshots SSM state at stride-aligned
    // positions for hybrid archs. Plain-attn models have empty ssm_entries
    // and ignore the stride entirely (no-op even at stride > 0). When the
    // hot prefix cache is disabled or off, set stride to 0 to skip
    // snapshot work that would just be discarded.
    const cp_stride: u32 = if (slot.model.prefix_cache != null) slot.model.ssm_checkpoint_stride else 0;
    const cp_max: u32 = slot.model.ssm_checkpoint_max;

    var gen = try Generator.initWithOptions(
        sch.io,
        slot.allocator,
        xfm_ptr,
        slot.model.tokenizer.?,
        prefill_tokens,
        slot.max_tokens,
        sampling,
        slot.eos_token_ids,
        .{
            .pld_enabled = use_pld,
            .drafter_enabled = use_drafter,
            .drafter = if (use_drafter) slot.drafter else null,
            .drafter_block_size = slot.drafter_block_size,
            .mtp_enabled = use_mtp,
            .mtp = if (use_mtp) slot.mtp else null,
            .mtp_depth = slot.mtp_depth,
            .lookup_prompt = slot.full_prompt,
            .ctx = slot.ctx,
            // Regular path: skip the lazy preforward so cache.step lands at
            // exactly prompt_len with t1 NOT in cache. Generator.next's
            // transition shim sync-forwards [t1] on the first decode call.
            // PLD/drafter/MTP init paths already skip preforward unconditionally.
            .skip_lazy_preforward = !use_pld and !use_drafter and !use_mtp,
            .ssm_checkpoint_stride = cp_stride,
            .ssm_checkpoint_max = cp_max,
            .ssm_checkpoint_pos_offset = hot_matched,
            // Abandoned-prefill abort: the conn thread sets slot.cancelled
            // when the client disconnects; the chunk loop checks it between
            // chunks so a ghost 40K prefill stops within one chunk.
            .cancel_flag = &slot.cancelled,
            .prefill_progress = if (observe) &sch.inflight_prefill_tokens else null,
        },
    );
    gen.timeout_ns = slot.timeout_ns;
    gen.logprobs_n = slot.logprobs_n;

    slot.legacy_gen = gen;
    // The conn thread's `cached_tokens` counted against `xfm.cache` (legacy
    // global cache) which the slot doesn't use. The slot's `cached_tokens`
    // is the hot-cache match (or 0 if the hot cache missed / isn't
    // configured) — those are the only tokens actually present in slot.cache
    // before this turn's prefill. With this override, slot.prompt_tokens
    // reports the full prompt length: gen.prompt_tokens (= prefill_tokens.len)
    // covers the un-cached tail, and `hot_matched` covers the restored prefix.
    slot.cached_tokens = hot_matched;
    slot.prompt_tokens = gen.prompt_tokens + slot.cached_tokens;
    slot.state = .decoding;
}

/// Sum the in-flight generated tokens over the active slots for the live-tok/s
/// gauge, EXCLUDING any slot that already finished/cancelled/errored this tick
/// (its tokens were counted into `generation_tokens_total` by `finishSlot`, so
/// including them here would double-count — and would leave a stale non-zero
/// aggregate once the last slot finishes, breaking the "at rest ⇒ live == total"
/// invariant). Generic over the slot type so the exact filter is unit-testable
/// with a lightweight stub; the real caller passes `[]*Slot`.
fn sumInflightGeneratedTokens(active: anytype) u64 {
    var inflight: u64 = 0;
    for (active) |s| {
        if (s.finished or s.error_code != null or s.cancelled.load(.acquire)) continue;
        inflight += @as(u64, s.completion_tokens);
    }
    return inflight;
}

fn runDecodeTick(sch: *Scheduler, active: []*Slot) !void {
    if (active.len == 0) return;

    // Publish the in-flight generated-token aggregate for the gauge sampler.
    // Runs after the whole tick (all decode work + any finishSlot calls) on
    // every return path — race-free (inference thread owns these slots' fields)
    // and O(active), never per token. Finished slots are excluded, so the last
    // slot's completion drives this to 0 (live == total at rest).
    defer sch.inflight_generated_tokens.store(sumInflightGeneratedTokens(active), .monotonic);

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
    var batchable_buf: [32]*Slot = undefined;
    var batchable_n: usize = 0;
    for (active) |s| {
        if (sch.batchable(s) and batchable_n < batchable_buf.len) {
            batchable_buf[batchable_n] = s;
            batchable_n += 1;
        } else {
            // legacy single-slot for spec / grammar / overflow
            try runSingleDecodeTick(sch, s);
        }
    }
    if (batchable_n == 0) return;

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
        const group = batchable_buf[start..end];
        // Honor force_batched even when only one slot is batchable so the
        // test hook actually exercises forwardBatchedDecode at N=1.
        if (group.len >= 2 or (sch.force_batched and group.len == 1)) {
            try runBatchedDecodeTick(sch, group);
        } else if (group.len == 1) {
            try runSingleDecodeTick(sch, group[0]);
        }
        start = end;
    }
}

/// Drive one Generator step (regular / PLD / drafter) and push emitted
/// tokens into the slot's output ring. Mirrors the existing
/// `StreamingTokenStream` adapter contract: 0..N tokens per call, with EOS
/// stopping the slot but NOT being emitted.
fn runSingleDecodeTick(sch: *Scheduler, slot: *Slot) !void {
    // ds4-backed slot: drive the engine's session forward by one token. No
    // PLD / drafter / batched paths apply — ds4 has its own internal MTP
    // (see TODO: wire `evalSpeculative` when temp=0 and engine.hasMtp()).
    if (slot.ds4_session) |session| {
        return runDs4DecodeTick(sch, slot, session);
    }
    if (slot.llama_session) |session| {
        return runLlamaDecodeTick(sch, slot, session);
    }
    if (slot.diffusion) |runner| {
        return runDiffusionDecodeTick(sch, slot, runner);
    }
    const gen = if (slot.legacy_gen) |*g| g else {
        slot.markError("no_generator");
        return;
    };

    // Stop a runaway repetition loop before generating more. Some models (seen
    // on Gemma 4 12B after a large/confusing tool result) collapse into spamming
    // one short cycle — e.g. the thinking opener `<|channel>thought` — forever;
    // with no repeat penalty by default and a generous max_tokens, nothing else
    // halts it until the cap. Checked here, before this tick's step, so it
    // covers the regular, PLD, and drafter paths uniformly.
    if (loopStopReason(gen.generated_ids.items)) |reason| {
        // Never cut silently: the 2026-07-14 php.html post-mortem took log
        // archaeology because this guard left no trace of having fired.
        log.warn("[loop-stop] degenerate tail loop cut after {d} generated tokens (finish_reason={s})\n", .{
            gen.generated_ids.items.len, reason,
        });
        finishSlot(sch, slot, reason);
        return;
    }

    // NOTE: no `!gen.spec_disabled_runtime` short-circuit here — the
    // generators handle the disabled fallback internally, and `nextPld`'s
    // disabled branch is also where the mid-request RE-ENABLE check lives
    // (bypassing it pinned PLD off for the rest of the request even when the
    // generated tail turned echo-heavy).
    if (slot.enable_mtp and gen.mtp != null) {
        const result = try gen.nextMtp(slot.allocator);
        if (result == null) {
            finishSlot(sch, slot, gen.finish_reason);
            return;
        }
        defer slot.allocator.free(result.?.tokens);
        for (result.?.tokens) |t| {
            if (slot.cancelled.load(.acquire)) return;
            if (generate_mod.isEosId(t, slot.eos_token_ids)) {
                finishSlot(sch, slot, "stop");
                return;
            }
            slot.pushToken(t);
            slot.completion_tokens = gen.completion_tokens;
            if (t != 0) slot.was_pad_only = false;
            if (slot.completion_tokens >= slot.max_tokens) {
                finishSlot(sch, slot, "length");
                return;
            }
        }
        return;
    }

    if (slot.enable_drafter and gen.drafter != null) {
        const result = try gen.nextDrafter(slot.allocator);
        if (result == null) {
            finishSlot(sch, slot, gen.finish_reason);
            return;
        }
        defer slot.allocator.free(result.?.tokens);
        for (result.?.tokens) |t| {
            if (slot.cancelled.load(.acquire)) return;
            if (generate_mod.isEosId(t, slot.eos_token_ids)) {
                finishSlot(sch, slot, "stop");
                return;
            }
            slot.pushToken(t);
            slot.completion_tokens = gen.completion_tokens;
            if (t != 0) slot.was_pad_only = false;
            if (slot.completion_tokens >= slot.max_tokens) {
                finishSlot(sch, slot, "length");
                return;
            }
        }
        return;
    }

    if (slot.enable_pld) {
        const result = try gen.nextPld(slot.allocator, slot.pld_draft_len, slot.pld_key_len);
        if (result == null) {
            finishSlot(sch, slot, gen.finish_reason);
            return;
        }
        defer slot.allocator.free(result.?.tokens);
        for (result.?.tokens) |t| {
            if (slot.cancelled.load(.acquire)) return;
            if (generate_mod.isEosId(t, slot.eos_token_ids)) {
                finishSlot(sch, slot, "stop");
                return;
            }
            slot.pushToken(t);
            slot.completion_tokens = gen.completion_tokens;
            if (t != 0) slot.was_pad_only = false;
            if (slot.completion_tokens >= slot.max_tokens) {
                finishSlot(sch, slot, "length");
                return;
            }
        }
        return;
    }

    // Regular path.
    const tok_opt = try gen.next(slot.allocator);
    if (tok_opt == null) {
        finishSlot(sch, slot, gen.finish_reason);
        return;
    }
    const t = tok_opt.?;
    slot.pushToken(t);
    if (t != 0) slot.was_pad_only = false;
    slot.completion_tokens = gen.completion_tokens;
    // Phase A5: capture per-token logprob for the conn thread to consume at
    // completion. `gen.last_logprob` ownership transfers into slot.logprobs_buf
    // (gen sets the field, we null it after taking it).
    if (slot.logprobs_n > 0) {
        if (gen.last_logprob) |lp| {
            slot.logprobs_buf.append(slot.allocator, lp) catch |err| {
                slot.markError(@errorName(err));
                return;
            };
            gen.last_logprob = null;
        }
    }
}

/// Batched decode kernel for >=2 active slots. All slots must have already
/// done a non-spec prefill (`skip_lazy_preforward = true`) so cache.step is
/// at prompt_len with `next_token_id` carrying t1. We forward those N tokens
/// in one kernel pass, sample per-slot, push the OLD next_token_id (= the
/// token we just committed to cache via the forward), and load the new
/// sampled id back into next_token_id.
fn runBatchedDecodeTick(sch: *Scheduler, active: []*Slot) !void {
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
        rope_offsets[i] = @intCast(slot.cache.step);
    }

    const logits_arr = try xfm_ptr.forwardBatchedDecode(next_tokens, ctxs, rope_offsets);
    defer {
        for (logits_arr) |a| _ = mlx.mlx_array_free(a);
        allocator.free(logits_arr);
    }

    // Sample per slot, emit prev id, set new next_token_id.
    for (batch, 0..) |slot, i| {
        if (slot.cancelled.load(.acquire)) continue;
        const gen = &slot.legacy_gen.?;
        const lazy = generate_mod.sampleTokenLazy(logits_arr[i], slot.sampling, xfm_ptr.s);
        try mlx.check(mlx.mlx_array_eval(lazy));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
        _ = mlx.mlx_array_free(lazy);

        const emit = gen.next_token_id;
        gen.generated_ids.append(slot.allocator, emit) catch |err| {
            slot.markError(@errorName(err));
            continue;
        };
        gen.completion_tokens += 1;
        gen.step += 1;
        gen.next_token_id = @intCast(val);

        // Stop checks (mirrors Generator.checkStop).
        if (generate_mod.isEosId(emit, slot.eos_token_ids)) {
            // Per existing contract, emit IS NOT yielded when it's EOS — the
            // STOP token comes BEFORE the yield. But here the cache has
            // already moved past it. The legacy path's checkStop runs on the
            // NEXT token (it's checked before emit). To preserve that
            // behavior we emit and then mark finished if `next_token_id` is
            // EOS (i.e. STOP is the next sampled token, ignored).
            slot.pushToken(emit);
            if (emit != 0) slot.was_pad_only = false;
            slot.completion_tokens = gen.completion_tokens;
            // not finished yet; next tick's checkStop on next_token_id ends it
        } else {
            slot.pushToken(emit);
            if (emit != 0) slot.was_pad_only = false;
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
}

const testing = std.testing;

test "modelBatchable rejects MoE / hybrid / encoder / sliding-window" {
    {
        var cfg = std.mem.zeroes(model_mod.ModelConfig);
        cfg.has_hybrid_layers = true;
        try testing.expect(!modelBatchable(&cfg));
    }
    {
        var cfg = std.mem.zeroes(model_mod.ModelConfig);
        cfg.full_attention_interval = 6;
        try testing.expect(!modelBatchable(&cfg));
    }
    {
        var cfg = std.mem.zeroes(model_mod.ModelConfig);
        cfg.is_encoder_only = true;
        try testing.expect(!modelBatchable(&cfg));
    }
    {
        // MoE: isMoe() returns true when num_experts > 0.
        var cfg = std.mem.zeroes(model_mod.ModelConfig);
        cfg.num_experts = 8;
        try testing.expect(!modelBatchable(&cfg));
    }
}

test "modelBatchable permits pure-attention" {
    // Defaults are all zero / null → vanilla pure-attention path.
    var cfg = std.mem.zeroes(model_mod.ModelConfig);
    try testing.expect(modelBatchable(&cfg));
}

test "buildGgufStubCpuState: llama stub carries gguf model_type + ctx sizing" {
    const a = std.testing.allocator;
    // No --ctx-size → the same 8192 default runLlamaServe uses (the GGUF's
    // trained context isn't readable until the engine opens).
    var s = try buildGgufStubCpuState(a, .llama, 0);
    defer freeCpuState(a, &s);
    try testing.expectEqualStrings("gguf", s.config.model_type);
    try testing.expectEqual(@as(u32, 8192), s.config.max_position_embeddings);
    try testing.expect(s.gguf == null); // route is attached by preloadGgufCpuState
    try testing.expectEqual(@as(usize, 0), s.chat_config.chat_template.len);

    // Explicit --ctx-size flows through to session sizing + context guard.
    var s2 = try buildGgufStubCpuState(a, .llama, 4096);
    defer freeCpuState(a, &s2);
    try testing.expectEqual(@as(u32, 4096), s2.config.max_position_embeddings);
}

test "buildGgufStubCpuState: ds4 stub carries deepseek_v4 + clamped ctx" {
    const a = std.testing.allocator;
    // Under-sized ctx floors at ds4's prefill chunk, exactly like runDs4Serve.
    var s = try buildGgufStubCpuState(a, .ds4, 512);
    defer freeCpuState(a, &s);
    try testing.expectEqualStrings("deepseek_v4", s.config.model_type);
    try testing.expectEqual(arch_ds4.clampSessionCtx(512), s.config.max_position_embeddings);
}

test "sumInflightGeneratedTokens sums active slots, excludes finished/cancelled/errored" {
    // Lightweight stub carrying exactly the fields the aggregate reads — proves
    // the live-gauge filter without constructing a real (mlx-backed) Slot.
    const StubSlot = struct {
        completion_tokens: u32,
        finished: bool,
        error_code: ?[]const u8,
        cancelled: std.atomic.Value(bool),
        fn make(tok: u32) @This() {
            return .{ .completion_tokens = tok, .finished = false, .error_code = null, .cancelled = std.atomic.Value(bool).init(false) };
        }
    };
    var a = StubSlot.make(10);
    var b = StubSlot.make(20);
    var done = StubSlot.make(99);
    done.finished = true; // already counted in generation_tokens_total
    var errd = StubSlot.make(50);
    errd.error_code = "OutOfMemory";
    var cxl = StubSlot.make(7);
    cxl.cancelled.store(true, .release);

    // Two still-decoding slots (10 + 20); the finished/errored/cancelled ones
    // are excluded, so the aggregate is 30 — never double-counting the tail.
    const active = [_]*StubSlot{ &a, &b, &done, &errd, &cxl };
    try testing.expectEqual(@as(u64, 30), sumInflightGeneratedTokens(active[0..]));

    // At rest: the last decoding slot finishes ⇒ aggregate collapses to 0,
    // pinning the "live == total at rest" invariant test_metrics.sh checks.
    a.finished = true;
    b.finished = true;
    try testing.expectEqual(@as(u64, 0), sumInflightGeneratedTokens(active[0..]));
}

test "loopStopReason: a degenerate tail cut reports length, a healthy tail is not cut" {
    // The reason MUST be "length": the SERVER is truncating the generation
    // (the model didn't finish — we cut a runaway repetition loop), and
    // "length" is the one reason server.toolCallFinishReason preserves through
    // tool-call parsing, so a call salvaged from the cut buffer reaches the
    // client as a TRUNCATION and its recovery fires. Live 2026-07-14
    // (plang/php.html): the cut reported "stop" → "tool_calls", presenting a
    // server-cut fragment as a model-completed write call — pi validated
    // garbage while the actual cause stayed invisible (the cut also never
    // logged). Reverting the reason to "stop" turns this red.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);

    // Healthy varied tail: never cut.
    for (0..40) |i| try ids.append(testing.allocator, @as(u32, @intCast(i * 7 + 3)));
    try testing.expect(loopStopReason(ids.items) == null);

    // Collapse into a short cycle (the php.html shape: "server-side scripting
    // language, " ≈ a 6-token cycle) past the guard's rep threshold.
    for (0..generate_mod.degenerate_loop_reps + 1) |_| {
        for ([_]u32{ 101, 202, 303, 404, 505, 606 }) |t| {
            try ids.append(testing.allocator, t);
        }
    }
    const reason = loopStopReason(ids.items) orelse return error.TestExpectedLoopCut;
    try testing.expectEqualStrings("length", reason);
}
