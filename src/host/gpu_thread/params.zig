//! What a load and a request hand the GPU thread, and the ledger of memory promised to admitted requests.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const handles = @import("../../abi/mlx_handles.zig");
const abi = @import("../../abi/engine.zig");
const tokenizer_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const chat_mod = @import("../chat/chat.zig");
const metrics_mod = @import("../metrics/metrics.zig");
const model_registry_mod = @import("../registry/registry.zig");
const ModelConfig = @import("../../sdk/config/model.zig").ModelConfig;
const KVQuantConfig = @import("../../sdk/blocks/attention/kv_quant.zig").KVQuantConfig;
const SamplingParams = @import("../../sdk/request/sampling.zig").SamplingParams;
const MediaSpan = @import("../../sdk/kv/media_span.zig").MediaSpan;
const LlamaSettings = @import("../../sdk/quant/gguf/llama_settings.zig").Settings;
const GgufRoute = @import("../../sdk/quant/gguf/engine_pick.zig").Route;
const BudgetRevise = @import("scheduler.zig").BudgetRevise;
const Tokenizer = tokenizer_mod.Tokenizer;
const ChatConfig = chat_mod.ChatConfig;
const ModelRegistry = model_registry_mod.ModelRegistry;
const LoadedModel = model_registry_mod.LoadedModel;
const testing = std.testing;

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
    resident_model_bytes: ?u64 = null,
    /// Path to the assistant drafter checkpoint. Empty disables the drafter.
    /// Borrowed; outlive scheduler.
    drafter_dir: []const u8 = "",
    /// `--no-drafter`: never load a drafter, including one MERGED into the
    /// checkpoint. `drafter_dir == ""` stopped meaning "off" the moment a
    /// checkpoint could carry its own, so the opt-out needs its own bit.
    no_drafter: bool = false,
    /// Auto-load the Qwen native MTP sidecar when the model dir ships one.
    mtp_enabled: bool = true,
    mtp_head_kv_quant: bool = false,
    /// Max MTP draft depth (CLI --mtp-max-depth; 0 = auto, resolved by
    /// the engine's depth cap at load/Generator init).
    mtp_depth: u32 = 0,
    /// Build the ANE prefill-MLP offload at load (`--ane-prefill`,
    /// perf-plan-aug-17 P5). Opt-in, lossy by design; every refusal is a
    /// named `[ane]` line and the model serves GPU-only.
    ane_prefill: bool = false,
    /// `--ane`: serve the qwen3_5 dense packs on the Neural Engine (a model's `ane` setting wins).
    ane: bool = false,
    /// Engines on one load of each Neural Engine model (`--ane-slots`): that many requests run at once.
    ane_slots: u32 = 1,
    /// The Neural Engine build compiles fixed-shape tiles against the server's prefill-chunk pin, and gates on
    /// its headroom bill (`admission/`); the engine wires both when this is set.
    ane_resolvers: bool = false,
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
    kv_quant_config: KVQuantConfig = KVQuantConfig.dense,
    /// Zero disables all prefix reuse; RAM retention is selected separately.
    prefix_cache_capacity: u32 = 1,
    prefix_cache_ram_enabled: bool = true,
    /// Per-model hot prefix cache KV-bytes budget. 0 disables the byte cap.
    prefix_cache_mem_bytes: u64 = 0,
    /// Clamp the hot-cache byte budget against live post-load headroom (`admission.prefixCacheMemForLoad`); off
    /// keeps the raw launch budget (tests).
    prefix_cache_clamp: bool = false,
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
    ssm_checkpoint_max: u32 = 16,
    /// Iteration 2 (perf-plan Phase 4 #3): per-LoadedModel LRU cache
    /// of chat-template render+tokenize results. 0 disables the cache
    /// (useful for ablation benches / debugging). Default 4 matches
    /// `prefix_cache_capacity` — most warm-reuse benches exercise a
    /// handful of repeated prompts, and full chat conversations bump
    /// this counter anyway via LRU as new turns arrive.
    tokenize_cache_entries: u32 = 4,
    /// Embedded llama.cpp engine settings (`--llama-*`), applied when a GGUF
    /// model's context is created at load.
    llama: LlamaSettings = .{},
    /// A GGUF and the engine that serves it (`engines/mlx/runtime/engine_load.zig`) instead of the MLX safetensors path.
    /// `config`/`tok`/`chat_config` are stubs (the engine owns the real tokenizer + chat template); they still move
    /// onto the entry so server-side reads of `lm.config.?` keep working.
    gguf: ?GgufRoute = null,
    /// SSD weight-streaming for the ds4 engine (issue #39): stream experts from
    /// disk instead of requiring the full model resident in RAM.
    ds4_ssd_streaming: bool = false,
    /// Auto-load the ds4 MTP draft head (found beside the model) for speculative
    /// decode. Default on; forced off when `ds4_ssd_streaming` (ds4 refuses the
    /// combination). `--no-ds4-mtp` disables it.
    ds4_mtp: bool = true,
    /// Select ds4's DSpark runtime when the auto-found support GGUF carries
    /// DSpark stages (`--dspark`, the same flag that opts the NATIVE dsv4
    /// engine into its draft stages). Off by default — the engine loads the
    /// stages but keeps target-only decode. Requires `ds4_mtp` (the sidecar
    /// is the support model).
    ds4_dspark: bool = false,
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
    /// Where the request's events go (`abi.Event`): the host's stream. Null = the slot's own buffer only.
    sink: ?abi.Sink = null,
    /// Full original prompt for PLD lookup. When null, defaults to
    /// `prompt_ids` (PLD's lookup table = full_prompt + generated).
    full_prompt: ?[]const u32 = null,
    cached_tokens: u32 = 0,
    has_tools: bool = false,
    /// The server's final resolved thinking mode after request overrides and
    /// model defaults. DFlash economics key off this, not tool presence.
    enable_thinking: bool = false,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    max_tokens: u32,
    timeout_ns: u64 = 0,
    enable_pld: bool = false,
    enable_drafter: bool = false,
    drafter: ?*handles.DrafterModel = null,
    /// DFlash assistant for this model. Rides the SAME `enable_drafter`
    /// request switch (a model loads at most ONE of drafter/dflash);
    /// `drafter_block_size` carries the dflash-resolved block size too.
    dflash: ?*handles.DflashModel = null,
    drafter_block_size: u32 = 4,
    enable_mtp: bool = false,
    allow_batch_mtp: bool = true,
    mtp: ?*handles.MtpHeadRef = null,
    /// 0 = auto (see the engine's depth cap).
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
    /// Media items in the prompt, for the prefix-cache key. Borrowed; the slot copies it.
    media: []const MediaSpan = &.{},
    /// Workload key for hot-cache eviction (`server.requestCacheKey`, 0 = anonymous).
    cache_key: u64 = 0,
    client: metrics_mod.Client = .other,
    /// Qwen3-VL interleaved M-RoPE: server-computed flat [3 × mrope_total] i32
    /// position-id table + decode delta. Ownership of `mrope_pos` transfers to
    /// the slot; freed on slot.deinit. Null for non-image / non-Qwen requests.
    mrope_pos: ?[]const i32 = null,
    mrope_total: usize = 0,
    mrope_delta: i32 = 0,
    logprobs_n: u32 = 0,
    /// `/v1/completions` echo: score every prompt token (cold prefill, no restore).
    prompt_logprobs: bool = false,
    /// A spoken reply (the model's `SpeechFacts`); null = text only.
    speech: ?@import("../../sdk/request/speech.zig").Request = null,
    /// Wave 1.A: per-request override of the process-default KV-cache quant
    /// scheme. When non-null, this slot's KVCache is constructed with this
    /// config instead of `Scheduler.kv_quant_config`. Lets one server host a
    /// single model and let clients trade accuracy for context length on a
    /// per-call basis (`{"kv_quant": "off"|4|8}` body field).
    kv_quant_config: ?KVQuantConfig = null,
    /// Plan 05 Phase D: the target model for this request. The conn thread
    /// resolves this via `scheduler.ensureLoaded(id)` BEFORE submitting and
    /// keeps a refcount on it for the slot's lifetime, so the model can't
    /// be evicted mid-flight. The scheduler routes prefill/decode through
    /// `slot.model.transformer.?` (and friends) instead of `sch.xfm`, so
    /// per-tick model switching is just a pointer hop. Required field
    /// post-Phase-D; tests using the legacy path pass the default model.
    model: *LoadedModel,
};

/// Bytes admitted requests were promised and have not allocated yet. A restored prefix shares
/// the cached buffers until its first append, so live memory does not show a sibling's claim
/// when the next one is billed; four such requests each fit alone and overran together.
pub const PromiseLedger = struct {
    base_active: u64 = 0,
    promised: u64 = 0,

    pub fn outstanding(self: PromiseLedger, active_now: u64) u64 {
        return self.promised -| (active_now -| self.base_active);
    }

    pub fn admit(self: *PromiseLedger, needed: u64, active_now: u64) void {
        if (self.promised == 0) self.base_active = active_now;
        self.promised +|= needed;
    }

    pub fn release(self: *PromiseLedger, needed: u64) void {
        self.promised -|= needed;
    }
};

/// A slot's promise stands only until its claim shows in live memory: the restored KV is
/// copied on the first append and the verify transients are steady a few rounds in.
pub const PROMISE_MATERIALIZED_TOKENS: usize = 64;
/// Slack a sibling's bill keeps under the ceiling: a group's verify transients grow with its
/// lanes, which no per-request bill prices.
pub const SIBLING_MARGIN_BYTES: u64 = 2 << 30;

pub fn promiseMaterialized(completion_tokens: usize) bool {
    return completion_tokens >= PROMISE_MATERIALIZED_TOKENS;
}

// ── tests ──

test "promiseMaterialized: a promise is dropped once the slot is decoding steadily" {
    try testing.expect(!promiseMaterialized(0));
    try testing.expect(!promiseMaterialized(PROMISE_MATERIALIZED_TOKENS - 1));
    try testing.expect(promiseMaterialized(PROMISE_MATERIALIZED_TOKENS));
}

test "PromiseLedger: a sibling is billed against what earlier admits have not allocated yet" {
    var l: PromiseLedger = .{};
    const gb: u64 = 1 << 30;
    try testing.expectEqual(@as(u64, 0), l.outstanding(20 * gb));
    l.admit(3 * gb, 20 * gb);
    l.admit(3 * gb, 20 * gb); // restored prefixes share buffers: active has not moved
    try testing.expectEqual(6 * gb, l.outstanding(20 * gb));
    try testing.expectEqual(2 * gb, l.outstanding(24 * gb)); // 4 GB of the promise materialized
    try testing.expectEqual(@as(u64, 0), l.outstanding(30 * gb));
    l.release(3 * gb);
    l.release(3 * gb);
    l.admit(1 * gb, 30 * gb); // an empty ledger re-bases
    try testing.expectEqual(1 * gb, l.outstanding(30 * gb));
}
