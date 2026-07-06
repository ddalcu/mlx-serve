//! Plan 05 — multi-model registry. Owns the discovery list and a set of
//! `LoadedModel` entries (loaded + unloaded). Provides refcounted access via
//! `ensureLoaded`/`release` so the per-request handler in `server.zig` can
//! route to the right model without globals, and so the inference thread
//! can swap which model's weights are "current" between scheduler ticks.
//!
//! Phase A scope: skeleton — types, bookkeeping (refcount, LRU clock,
//! summed-bytes accounting), snapshot for `/v1/models`. `ensureLoaded`
//! accepts already-`.ready` entries and waits on `.loading` ones; the
//! cold-load posting path that promotes `.unloaded` → `.loading` → `.ready`
//! lives in Phase D where the scheduler hook lands. Test helpers exercise
//! the bookkeeping without touching mlx.
//!
//! Threading model: connection threads call `ensureLoaded`/`release`. The
//! inference thread (Phase D) sees `.loading` work items via the same
//! `vision_queue`/`embed_queue`-shaped channel; cold-load completion
//! broadcasts on `state_cond` so blocked callers wake.

const std = @import("std");
const model_mod = @import("model.zig");
const transformer_mod = @import("transformer.zig");
const tokenizer_mod = @import("tokenizer.zig");
const chat_mod = @import("chat.zig");
const vision_mod = @import("vision.zig");
const drafter_mod = @import("drafter.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const tokenize_cache_mod = @import("tokenize_cache.zig");
const model_discovery = @import("model_discovery.zig");
const arch_ds4 = if (@import("build_options").ios) @import("arch/ds4_stub.zig") else @import("arch/ds4.zig");
const arch_llama = if (@import("build_options").ios) @import("arch/llama_stub.zig") else @import("arch/llama.zig");
const gen_mod = @import("gen.zig");
const log = @import("log.zig");

const Transformer = transformer_mod.Transformer;
const Weights = model_mod.Weights;
const ModelConfig = model_mod.ModelConfig;
const Tokenizer = tokenizer_mod.Tokenizer;
const ChatConfig = chat_mod.ChatConfig;
const VisionEncoder = vision_mod.VisionEncoder;
const DrafterModel = drafter_mod.DrafterModel;
const mtp_mod = @import("mtp.zig");
const MtpModel = mtp_mod.MtpModel;
const HotPrefixCache = prefix_cache_mod.HotPrefixCache;
const TokenizeCache = tokenize_cache_mod.TokenizeCache;

/// One slot in `LoadedModel.llama_sessions` (Iteration 3-5 of the perf
/// plan / Phase 5 #1). Each entry wraps a libllama context. The KV
/// state and the resident-token mirror live inside the session; we add
/// `last_used_ns` for LRU eviction.
pub const LlamaSessionEntry = struct {
    session: *arch_llama.LlamaSession,
    /// Bumped on every successful pick. Lowest = LRU. Monotonic — the
    /// scheduler bumps it under the same lock that guards
    /// `llama_sessions`, so reads/writes never race.
    last_used_ns: i64 = 0,
};

/// Lifecycle of an entry. State transitions are guarded by `ModelRegistry.mutex`;
/// the inference thread writes, connection threads read under the same lock and
/// wait on `state_cond` for transitions.
pub const LoadState = enum {
    /// Discovered but never loaded — no GPU memory committed. ensureLoaded
    /// promotes to `.loading` (Phase D).
    unloaded,
    /// Inference thread is faulting weights / building the Transformer.
    /// ensureLoaded callers block on `state_cond`.
    loading,
    /// Live in GPU memory. Refcounted; safe to use for inference.
    ready,
    /// Load failed. `error_name` is populated; ensureLoaded returns
    /// `error.LoadFailed` until the entry resets.
    error_state,
    /// LRU eviction in progress. Refcount must be 0 before the inference
    /// thread enters this state.
    evicting,
};

/// One discovered + possibly-loaded model. The mlx-allocating fields
/// (weights/transformer/vision_encoder/drafter) are optional so a stub
/// entry can exist for `unloaded`/`error_state`/`loading` without faking
/// half-built mlx state.
pub const LoadedModel = struct {
    allocator: std.mem.Allocator,

    /// Identifier exposed via `/v1/models` and the request `model` field.
    /// Allocator-owned dupe of the discovery id (path basename, e.g.
    /// "gemma-4-e4b-it-4bit"). Also used as the StringHashMap key — the map
    /// stores this slice directly, so it must outlive the entry's presence
    /// in `ModelRegistry.entries`.
    id: []const u8,
    /// Full absolute path on disk. Allocator-owned dupe.
    path: []const u8,
    /// Approximate weight bytes on disk (sum of *.safetensors). Used for
    /// pre-load eviction estimates so we don't oversubscribe wired memory.
    bytes_on_disk: ?u64,
    /// `model_type` peeked from config.json at discovery time (allocator-
    /// owned dupe; empty when unknown). Lets /v1/models advertise
    /// arch-derived capabilities — e.g. "bert" → embeddings — while the
    /// entry is still an `.unloaded` stub.
    arch_hint: []const u8,

    // ── Mlx-allocating state. Non-null iff state == .ready (or transitioning
    //    out of .ready via eviction). Owned by this entry; freed in deinit.

    /// Parsed `config.json`. Heap-allocated pointer so the address is stable
    /// across LoadedModel relocations (the registry stores `*LoadedModel`
    /// so the LoadedModel itself never moves, but downstream consumers
    /// borrow `*const ModelConfig` and we want one consistent ownership
    /// pattern across config/tokenizer/chat_config).
    config: ?*ModelConfig,
    weights: ?*Weights,
    transformer: ?*Transformer,
    /// Per-model tokenizer (different vocabularies across models). Phase 05
    /// moves ownership here from main.zig so the tokenizer's lifetime
    /// matches the model's residency.
    tokenizer: ?*Tokenizer,
    /// Per-model chat config — chat templates and EOS-token strings vary
    /// across model families.
    chat_config: ?*ChatConfig,
    vision_encoder: ?*VisionEncoder,
    drafter: ?*DrafterModel,
    /// Echoed in `/v1/models` so the Swift app can show the drafter checkpoint
    /// path; empty when no drafter is loaded. Allocator-owned dupe.
    drafter_path: []const u8,
    drafter_block_size: u32,
    /// Qwen native MTP head, auto-loaded when the model dir ships an
    /// `mtp/weights.safetensors` sidecar that binds to this trunk.
    mtp: ?*MtpModel = null,
    /// Default draft depth for MTP rounds (CLI `--mtp-depth`).
    mtp_depth: u32 = mtp_mod.DEFAULT_DEPTH,
    /// Per-model hot prefix cache — plan 05 drops the module-global hot
    /// cache. `model_id`-keyed isolation falls out of "one cache per
    /// LoadedModel" by construction.
    prefix_cache: ?HotPrefixCache,
    /// Phase 1 (perf-plan): SSM/conv state snapshot stride during prefill,
    /// in tokens. 0 = disabled (hybrid models bypass the hot prefix cache,
    /// preserving legacy behavior). Non-zero enables multi-turn warm reuse
    /// on hybrid SSM archs. Set by `doLoadOnInferenceThread` from
    /// `LoadParams.ssm_checkpoint_stride`.
    ssm_checkpoint_stride: u32 = 0,
    /// Phase 1: per-request cap on snapshots retained.
    ssm_checkpoint_max: u32 = 32,

    /// Iteration 2 (perf-plan Phase 4 #3): LRU cache of chat-template
    /// render+tokenize results, keyed by a digest of (messages, tools,
    /// flags). Targets the warm-reuse path where Jinja+BPE was
    /// observed at 240 ms on a 1813-tok Gemma prompt — 7× the actual
    /// Metal prefill on a KV-cache hit. Null when --tokenize-cache-entries
    /// is 0 (caller disables for tests / debugging).
    tokenize_cache: ?TokenizeCache = null,

    /// Embedded ds4 engine (DeepSeek-V4-Flash via GGUF). When non-null,
    /// `transformer` / `weights` / `tokenizer` / `chat_config` stay null and
    /// the request handlers route through `Ds4Engine` / `Ds4Session` instead
    /// of the MLX path. Mutually exclusive with the safetensors fields.
    ds4_engine: ?*arch_ds4.Ds4Engine = null,

    /// Embedded llama.cpp engine (generic GGUF via libllama). Like `ds4_engine`,
    /// when non-null the MLX fields stay null and request handlers route through
    /// `LlamaEngine` / `LlamaSession`. Mutually exclusive with the safetensors
    /// fields and `ds4_engine` (set for every `.gguf` except DeepSeek-V4-Flash).
    llama_engine: ?*arch_llama.LlamaEngine = null,

    /// Native media-generation engines, named by MODALITY (not by the FLUX/
    /// Qwen3-TTS/LTX implementations, which are swappable internals). When one
    /// is non-null the entry is a media model: the MLX/ds4/llama fields stay
    /// null and the server routes the matching gen endpoint through this slot.
    /// Mutually exclusive with each other and with the LM engine fields. Freed
    /// FIRST in deinit/unloadResident (they own the bulk of the GPU memory).
    image_engine: ?*gen_mod.ImageEngine = null,
    audio_engine: ?*gen_mod.AudioEngine = null,
    video_engine: ?*gen_mod.VideoEngine = null,
    mesh_engine: ?*gen_mod.MeshEngine = null,
    /// Model-wide serialization gate for media generation — mirrors
    /// `llama_session_busy`. A gen runs to completion on the inference thread
    /// (the sole mlx caller), so gen-vs-gen is already serial; this flag makes
    /// the in-flight state visible (set around the gen job).
    gen_busy: bool = false,

    /// Persistent llama.cpp sessions (Iteration 3-5 / Phase 5 #1): one or
    /// more KV contexts, picked by best prompt-prefix match in
    /// `runPrefillLlama`. With `--llama-cache-entries 1` (default for
    /// backwards compat) this degenerates to the old single-session
    /// behavior — one entry, every request fights for it. With N > 1 the
    /// scheduler keeps the N most-recently-used sessions alive and
    /// dispatches each incoming prompt to the entry whose resident KV
    /// shares the longest prefix.
    ///
    /// `llama_session_busy` remains a model-wide gate — `max_concurrent=1`
    /// today means only one llama request runs at a time anyway, and
    /// adding per-entry concurrency would require an inference-thread
    /// refactor we intentionally don't ship tonight.
    ///
    /// Sessions are freed BEFORE `llama_engine` in `deinit` because each
    /// holds a context bound to the engine's model.
    llama_sessions: std.ArrayListUnmanaged(LlamaSessionEntry) = .empty,
    /// Cap on resident llama sessions. Mirrored from
    /// `LoadParams.llama_cache_entries` at load time. 0 falls back to 1
    /// for safety — every llama prefill needs at least one session.
    llama_cache_max_entries: u32 = 1,
    llama_session_busy: bool = false,
    /// Phase 5 #2: ggml types for the K and V halves of the llama.cpp KV
    /// cache. 0 = libllama default (F16). Non-zero values are pulled from
    /// `LoadParams.llama_kv_type_{k,v}` at load time and read by
    /// `runPrefillLlama` when creating the persistent session. Mutating
    /// these after a session has been created has no effect — the values
    /// are baked into the libllama context at create-time.
    llama_kv_type_k: i32 = 0,
    llama_kv_type_v: i32 = 0,

    // ── Bookkeeping. Updated under `ModelRegistry.mutex`. ──

    /// Number of in-flight callers holding a borrowed pointer. Incremented
    /// by `ensureLoaded`, decremented by `release`. Eviction is blocked
    /// while refcount > 0. Atomic so the inference thread can observe it
    /// without re-acquiring the registry mutex inside a tick.
    refcount: std.atomic.Value(u32),
    /// Monotonic clock — bumped on every `release`. Higher = more recent.
    /// LRU eviction picks the lowest among `.ready` entries with
    /// refcount == 0.
    last_used_ns: i64,
    /// Resident GPU bytes for this entry (weights + vision + drafter),
    /// summed at load time. Zero for non-`.ready` entries.
    bytes_resident: u64,
    /// Estimated bytes reserved against `ModelRegistry.reserved_bytes` while
    /// this entry is mid-load (`.loading`). Lets concurrent loaders see this
    /// pending allocation in the budget gate so two loads can't both pass and
    /// oversubscribe GPU memory. Released (zeroed) at markReady/Error/Unloaded.
    load_estimate: u64 = 0,
    state: LoadState,
    /// Allocator-owned error name when state == .error_state. Echoed in
    /// the `/v1/models` snapshot so clients can see why a load failed
    /// (e.g. "LoadFailed", "MissingVisionWeights"). Null otherwise.
    error_name: ?[]const u8,

    /// Free all owned state. Safe to call regardless of `state` — null
    /// model fields are skipped. Mlx-allocating fields are freed in
    /// drafter → vision → transformer → weights order to mirror the
    /// dependency chain in `Scheduler.deinit`. Note: mlx-allocating frees
    /// must run on the scheduler's inference thread (thread-local GPU
    /// stream); the caller arranges this via `unloadResident` invoked
    /// from the inference thread before registry teardown.
    pub fn deinit(self: *LoadedModel) void {
        for (self.llama_sessions.items) |entry| entry.session.free();
        self.llama_sessions.deinit(self.allocator);
        self.llama_session_busy = false;
        if (self.ds4_engine) |engine| {
            engine.close();
            self.ds4_engine = null;
        }
        if (self.llama_engine) |engine| {
            engine.close();
            self.llama_engine = null;
        }
        if (self.image_engine) |e| {
            e.deinit();
            self.image_engine = null;
        }
        if (self.audio_engine) |e| {
            e.deinit();
            self.audio_engine = null;
        }
        if (self.video_engine) |e| {
            e.deinit();
            self.video_engine = null;
        }
        if (self.mesh_engine) |e| {
            e.deinit();
            self.mesh_engine = null;
        }
        self.gen_busy = false;
        if (self.mtp) |h| {
            h.deinit();
            self.allocator.destroy(h);
            self.mtp = null;
        }
        if (self.drafter) |d| {
            d.deinit();
            self.allocator.destroy(d);
            self.drafter = null;
        }
        if (self.vision_encoder) |v| {
            v.deinit();
            self.allocator.destroy(v);
            self.vision_encoder = null;
        }
        if (self.transformer) |x| {
            x.deinit();
            self.allocator.destroy(x);
            self.transformer = null;
        }
        if (self.weights) |w| {
            w.deinit();
            self.allocator.destroy(w);
            self.weights = null;
        }
        if (self.tokenizer) |tok| {
            tok.deinit();
            self.allocator.destroy(tok);
            self.tokenizer = null;
        }
        if (self.chat_config) |cc| {
            cc.deinit();
            self.allocator.destroy(cc);
            self.chat_config = null;
        }
        if (self.config) |c| {
            // ModelConfig has no allocator-owned fields (all by-value, plus
            // borrowed-static `model_type`/`weight_prefix`) so a plain
            // destroy suffices.
            self.allocator.destroy(c);
            self.config = null;
        }
        if (self.prefix_cache) |*hc| {
            hc.deinit();
            self.prefix_cache = null;
        }
        if (self.tokenize_cache) |*tc| {
            tc.deinit();
            self.tokenize_cache = null;
        }
        if (self.error_name) |n| self.allocator.free(n);
        if (self.drafter_path.len > 0) self.allocator.free(self.drafter_path);
        if (self.arch_hint.len > 0) self.allocator.free(self.arch_hint);
        self.allocator.free(self.id);
        self.allocator.free(self.path);
    }

    /// Free only the mlx-allocating state (weights/transformer/vision/
    /// drafter/prefix_cache), leaving CPU-only fields (config/tokenizer/
    /// chat_config) AND the discovery stub (id/path/bytes_on_disk) intact.
    /// Used by eviction so the registry keeps the entry around as
    /// `.unloaded` for later listing/reload, AND by `Scheduler.deinit` so
    /// mlx frees happen on the inference thread.
    pub fn unloadResident(self: *LoadedModel) void {
        for (self.llama_sessions.items) |entry| entry.session.free();
        self.llama_sessions.clearRetainingCapacity();
        self.llama_session_busy = false;
        if (self.ds4_engine) |engine| {
            engine.close();
            self.ds4_engine = null;
        }
        if (self.llama_engine) |engine| {
            engine.close();
            self.llama_engine = null;
        }
        if (self.image_engine) |e| {
            e.deinit();
            self.image_engine = null;
        }
        if (self.audio_engine) |e| {
            e.deinit();
            self.audio_engine = null;
        }
        if (self.video_engine) |e| {
            e.deinit();
            self.video_engine = null;
        }
        if (self.mesh_engine) |e| {
            e.deinit();
            self.mesh_engine = null;
        }
        self.gen_busy = false;
        if (self.mtp) |h| {
            h.deinit();
            self.allocator.destroy(h);
            self.mtp = null;
        }
        if (self.drafter) |d| {
            d.deinit();
            self.allocator.destroy(d);
            self.drafter = null;
        }
        if (self.vision_encoder) |v| {
            v.deinit();
            self.allocator.destroy(v);
            self.vision_encoder = null;
        }
        if (self.transformer) |x| {
            x.deinit();
            self.allocator.destroy(x);
            self.transformer = null;
        }
        if (self.weights) |w| {
            w.deinit();
            self.allocator.destroy(w);
            self.weights = null;
        }
        if (self.prefix_cache) |*hc| {
            hc.deinit();
            self.prefix_cache = null;
        }
        if (self.drafter_path.len > 0) {
            self.allocator.free(self.drafter_path);
            self.drafter_path = "";
        }
        self.drafter_block_size = 0;
        self.bytes_resident = 0;
        self.state = .unloaded;
    }
};

/// Snapshot of a single entry, returned by `ModelRegistry.snapshot` for
/// the `/v1/models` JSON listing. All slices are borrowed from the
/// underlying entry — caller must finish reading before the snapshot slice
/// is freed (registry mutex protects entry lifetime; snapshot is taken
/// under the lock so the data won't disappear under the caller).
pub const ModelStatus = struct {
    id: []const u8,
    loaded: bool,
    bytes_resident: u64,
    bytes_on_disk: ?u64,
    last_used_ns: i64,
    state: []const u8,
    error_name: ?[]const u8,

    pub fn stateName(s: LoadState) []const u8 {
        return switch (s) {
            .unloaded => "unloaded",
            .loading => "loading",
            .ready => "ready",
            .error_state => "error",
            .evicting => "evicting",
        };
    }
};

/// Registry of all discovered + loaded models. One instance per server.
///
/// Lifecycle:
///   1. `init` is called from `serve()`, given an owned `DiscoveryResult`
///      (or null for the legacy single-model case). Pre-populates `.unloaded`
///      stubs for every discovered id so `/v1/models` and `ensureLoaded`
///      can address them by name from t0.
///   2. `serve()` chooses a default model (from `--model` or the first
///      discovered) and (Phase B) hands its load spec to the scheduler.
///      On success the entry transitions `.unloaded` → `.loading` → `.ready`.
///   3. Per-request handlers (Phase C) call `ensureLoaded(scheduler, id)`
///      and `release`. The borrowed `*LoadedModel` is valid until `release`.
///   4. `deinit` tears down every entry; the discovery result is freed.
pub const ModelRegistry = struct {
    allocator: std.mem.Allocator,
    io: std.Io,

    /// Owned discovery (or null when `--model-dir` wasn't passed and the
    /// server was started with just `--model`). When null, `entries` still
    /// holds one synthetic entry for the loaded model so the rest of the
    /// API works uniformly.
    discovery: ?model_discovery.DiscoveryResult,

    /// Map of id → entry. Keys are borrowed from `LoadedModel.id`; entries
    /// are heap-allocated and owned by the registry. Iteration order isn't
    /// guaranteed; snapshot sorts by `last_used_ns` desc for stable output.
    entries: std.StringHashMap(*LoadedModel),

    /// Default model id used when a request doesn't specify `model` (or
    /// specifies the literal "mlx-serve"). Borrowed from the corresponding
    /// entry's `id`; valid for the registry's lifetime.
    default_id: []const u8,

    /// Cap on `.ready` entries. ensureLoaded evicts before exceeding.
    max_resident_models: u32,
    /// Cap on summed bytes_resident across `.ready` entries.
    /// 0 disables the byte cap (count cap still applies).
    max_resident_mem: u64,
    /// When non-null, the inference thread's idle tick (Phase D) evicts
    /// `.ready` entries with refcount == 0 that haven't been touched
    /// within this window.
    idle_evict_secs: ?u32,

    mutex: std.Io.Mutex,
    /// Broadcast whenever an entry's `state` changes. Connection threads
    /// blocked in `ensureLoaded` (on a `.loading`/`.evicting` entry) wake
    /// here. The mutex is the predicate; the cond-var is the wake signal.
    state_cond: std.Io.Condition,

    /// Running sum of `bytes_resident` across `.ready` entries. Updated
    /// under `mutex` on every state transition.
    current_resident_bytes: u64,
    /// Sum of `load_estimate` across entries that are mid-load (`.loading`).
    /// Counted alongside `current_resident_bytes` in the eviction gate so an
    /// in-flight load is visible to a concurrent loader — without this, two
    /// loads can both read a stale (pre-commit) resident total, both skip
    /// eviction, and oversubscribe GPU memory (→ Metal OOM crash).
    reserved_bytes: u64,
    /// Monotonic counter feeding `LoadedModel.last_used_ns`. We avoid
    /// reading the wall clock under the mutex; ordering is all the LRU
    /// pick needs.
    lru_clock: i64,

    /// Create the registry. Takes ownership of `discovery` (if non-null)
    /// and pre-populates entries from it.
    ///
    /// Caller responsibilities:
    ///   * Choose `default_id` so it matches either a discovered model id
    ///     or a freshly-registered synthetic entry (`registerEntry`).
    ///   * Call `registerEntry` for the loaded `--model` *before* using the
    ///     registry if it wasn't already in `discovery`.
    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        discovery: ?model_discovery.DiscoveryResult,
        max_resident_models: u32,
        max_resident_mem: u64,
        idle_evict_secs: ?u32,
    ) !*ModelRegistry {
        const self = try allocator.create(ModelRegistry);
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .discovery = discovery,
            .entries = std.StringHashMap(*LoadedModel).init(allocator),
            .default_id = "",
            .max_resident_models = if (max_resident_models == 0) 1 else max_resident_models,
            .max_resident_mem = max_resident_mem,
            .idle_evict_secs = idle_evict_secs,
            .mutex = .init,
            .state_cond = .init,
            .current_resident_bytes = 0,
            .reserved_bytes = 0,
            .lru_clock = 0,
        };

        // Pre-populate `.unloaded` stubs for every discovered model. The
        // scheduler's load path (Phase B/D) will promote one of these to
        // `.ready` during startup, and `ensureLoaded` may promote others
        // on-demand. Allocation failures here roll back any partial
        // entries via errdefer so we don't leak strings on OOM.
        if (discovery) |d| {
            errdefer self.deinitInternal();
            for (d.models) |m| {
                _ = try self.registerStubWithArch(m.id, m.path, m.bytes_on_disk, m.model_type);
            }
        }

        return self;
    }

    /// Frees everything the registry owns. Safe regardless of `state` on
    /// each entry. Called once at server shutdown after the scheduler has
    /// joined (so the inference thread isn't mid-load).
    pub fn deinit(self: *ModelRegistry) void {
        self.deinitInternal();
        self.allocator.destroy(self);
    }

    fn deinitInternal(self: *ModelRegistry) void {
        var it = self.entries.valueIterator();
        while (it.next()) |entry_ptr| {
            const entry = entry_ptr.*;
            entry.deinit();
            self.allocator.destroy(entry);
        }
        self.entries.deinit();
        if (self.discovery) |*d| d.deinit();
    }

    /// Insert an `.unloaded` stub for a model. Returns the heap-allocated
    /// entry; ownership stays with the registry. id and path are duped so
    /// caller buffers can be transient.
    ///
    /// Used by `init` for discovery and by `serve()` to register the
    /// loaded `--model` when it wasn't in `--model-dir`.
    pub fn registerStub(self: *ModelRegistry, id: []const u8, path: []const u8, bytes_on_disk: ?u64) !*LoadedModel {
        return self.registerStubWithArch(id, path, bytes_on_disk, "");
    }

    /// `registerStub` variant carrying the discovery-peeked `model_type` so
    /// the stub can advertise arch-derived capabilities before loading.
    pub fn registerStubWithArch(self: *ModelRegistry, id: []const u8, path: []const u8, bytes_on_disk: ?u64, arch_hint: []const u8) !*LoadedModel {
        if (self.entries.get(id) != null) return error.DuplicateId;

        const stub = try self.allocator.create(LoadedModel);
        errdefer self.allocator.destroy(stub);
        const id_owned = try self.allocator.dupe(u8, id);
        errdefer self.allocator.free(id_owned);
        const path_owned = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(path_owned);
        const arch_owned: []const u8 = if (arch_hint.len > 0) try self.allocator.dupe(u8, arch_hint) else "";
        errdefer if (arch_owned.len > 0) self.allocator.free(arch_owned);

        stub.* = .{
            .allocator = self.allocator,
            .id = id_owned,
            .path = path_owned,
            .bytes_on_disk = bytes_on_disk,
            .arch_hint = arch_owned,
            .config = null,
            .weights = null,
            .transformer = null,
            .tokenizer = null,
            .chat_config = null,
            .vision_encoder = null,
            .drafter = null,
            .drafter_path = "",
            .drafter_block_size = 0,
            .prefix_cache = null,
            .ds4_engine = null,
            .refcount = std.atomic.Value(u32).init(0),
            .last_used_ns = 0,
            .bytes_resident = 0,
            .state = .unloaded,
            .error_name = null,
        };

        // putNoClobber so we surface a stronger error than overwrite — the
        // earlier `entries.get` guard makes this defensive.
        try self.entries.putNoClobber(stub.id, stub);
        return stub;
    }

    /// Register-by-path (/v1/load-model with an absolute path): validate
    /// `abs_path` exactly like discovery would (config.json, supported
    /// model_type + quant mode) and insert an `.unloaded` stub keyed by the
    /// directory basename, carrying the arch hint and weight bytes. An
    /// existing entry with that id wins — same id means same model in the
    /// registry's world, and we never re-point a live entry's path. Returns
    /// the entry's stable id slice (owned by the registry).
    ///
    /// Exists for models OUTSIDE the --model-dir scan: the app auto-downloads
    /// a small embedding encoder and registers it here no matter which org
    /// dir the chat model (and thus --model-dir) points at.
    pub fn registerByPath(self: *ModelRegistry, io: std.Io, abs_path: []const u8) ![]const u8 {
        var trimmed = abs_path;
        while (trimmed.len > 0 and trimmed[trimmed.len - 1] == '/') trimmed = trimmed[0 .. trimmed.len - 1];
        const base = std.fs.path.basename(trimmed);
        if (base.len == 0) return error.InvalidModelPath;

        // Fast path: already registered (discovered, --model, or a previous
        // register-by-path). No filesystem touch.
        {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            if (self.entries.get(base)) |existing| return existing.id;
        }

        // A discovery entry may hold this path under an org/name id whose
        // basename differs — resolve by path before probing the filesystem.
        if (self.peekByPath(trimmed)) |existing| return existing.id;

        const probe = try model_discovery.probeModelDir(io, self.allocator, trimmed);
        defer self.allocator.free(probe.model_type);

        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        // Re-check under the lock — another conn thread may have raced us.
        if (self.entries.get(base)) |existing| return existing.id;
        const stub = try self.registerStubWithArch(base, trimmed, probe.bytes_on_disk, probe.model_type);
        return stub.id;
    }

    /// Set the default model id used for requests that omit `model` or
    /// pass the literal "mlx-serve". The id must already exist (via
    /// `registerStub`); caller borrows the entry's own `id` slice so the
    /// pointer is stable for the registry's lifetime.
    pub fn setDefault(self: *ModelRegistry, id: []const u8) !void {
        const entry = self.entries.get(id) orelse return error.UnknownModelId;
        self.default_id = entry.id;
    }

    /// Look up an entry by its on-disk path (trailing slashes ignored).
    /// Same read-only contract as `peek`. Exists so the `--model` path and
    /// register-by-path flows reuse a discovery entry even when their
    /// basename-derived id differs from the discovered `org/name` id —
    /// two ids for one path would let the same weights load twice.
    pub fn peekByPath(self: *ModelRegistry, path: []const u8) ?*LoadedModel {
        var trimmed = path;
        while (trimmed.len > 0 and trimmed[trimmed.len - 1] == '/') trimmed = trimmed[0 .. trimmed.len - 1];
        if (trimmed.len == 0) return null;
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        var it = self.entries.valueIterator();
        while (it.next()) |entry_ptr| {
            var entry_path: []const u8 = entry_ptr.*.path;
            while (entry_path.len > 0 and entry_path[entry_path.len - 1] == '/') entry_path = entry_path[0 .. entry_path.len - 1];
            if (std.mem.eql(u8, entry_path, trimmed)) return entry_ptr.*;
        }
        return null;
    }

    /// Look up an entry by id without taking a refcount. Returns null if
    /// the id is unknown. Callers that intend to use the entry for
    /// inference MUST go through `ensureLoaded` instead — this is for read-
    /// only paths (e.g. `/v1/models` listing, log lines).
    pub fn peek(self: *ModelRegistry, id: []const u8) ?*LoadedModel {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.entries.get(id);
    }

    /// Resolve `id_or_empty` ("" or "mlx-serve" → default) to the entry.
    /// Pure ID resolution; does not touch state or refcounts. Returns
    /// `error.UnknownModelId` if the id isn't registered or
    /// `error.NoDefaultModel` if `id_or_empty` is empty AND no default is
    /// set. The returned pointer is borrowed; valid for the registry's
    /// lifetime.
    pub fn resolveEntry(self: *ModelRegistry, id_or_empty: []const u8) !*LoadedModel {
        const id = if (id_or_empty.len == 0 or std.mem.eql(u8, id_or_empty, "mlx-serve"))
            self.default_id
        else
            id_or_empty;
        if (id.len == 0) return error.NoDefaultModel;
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.entries.get(id) orelse error.UnknownModelId;
    }

    /// Resolve `id` (or the default when `id` is null/empty/"mlx-serve")
    /// to a refcounted `*LoadedModel`. Phase A skeleton: succeeds only on
    /// already-`.ready` entries and waits out `.loading`/`.evicting`
    /// transitions; returns `error.NotLoaded` for `.unloaded` and
    /// `error.LoadFailed` for `.error_state`. Phase D's cold-load path
    /// lives on `Scheduler.ensureLoaded` (which calls into this fast-path
    /// first; on `error.NotLoaded` it triggers a load on the inference
    /// thread and re-enters here when the entry transitions to `.ready`).
    ///
    /// Caller MUST call `release(lm)` once done with the returned pointer.
    pub fn ensureLoaded(self: *ModelRegistry, id_or_empty: []const u8) !*LoadedModel {
        const id = if (id_or_empty.len == 0 or std.mem.eql(u8, id_or_empty, "mlx-serve"))
            self.default_id
        else
            id_or_empty;

        if (id.len == 0) return error.NoDefaultModel;

        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);

        const entry = self.entries.get(id) orelse return error.UnknownModelId;

        while (true) {
            switch (entry.state) {
                .ready => {
                    _ = entry.refcount.fetchAdd(1, .acq_rel);
                    return entry;
                },
                .loading, .evicting => {
                    self.state_cond.waitUncancelable(self.io, &self.mutex);
                    continue;
                },
                .error_state => return error.LoadFailed,
                .unloaded => return error.NotLoaded,
            }
        }
    }

    /// Phase D: claim the right to perform a cold load for `entry`. Caller
    /// MUST already hold `mutex`. Transitions `.unloaded` → `.loading` and
    /// broadcasts so any other ensureLoaded callers join the wait. Returns
    /// false if the entry isn't in `.unloaded` state (someone else is
    /// already loading it, or it's already ready/error). On false, the
    /// caller should re-check state and wait or fast-path.
    pub fn tryBeginLoadLocked(self: *ModelRegistry, entry: *LoadedModel) bool {
        if (entry.state != .unloaded) return false;
        entry.state = .loading;
        self.state_cond.broadcast(self.io);
        return true;
    }

    /// Phase D: roll back a failed load attempt back to `.unloaded` so a
    /// later request can retry. Caller holds `mutex`. The entry's resident
    /// fields should already be cleaned up (or never installed) before
    /// calling.
    pub fn markUnloadedLocked(self: *ModelRegistry, entry: *LoadedModel) void {
        self.releaseReservationLocked(entry);
        entry.state = .unloaded;
        entry.bytes_resident = 0;
        if (entry.error_name) |old| self.allocator.free(old);
        entry.error_name = null;
        self.state_cond.broadcast(self.io);
    }

    /// Phase D: begin evicting `entry`. Caller holds `mutex`. Subsequent
    /// ensureLoaded callers wait on `.evicting`. Returns immediately;
    /// caller MUST call `waitForRefcountZeroLocked` to drain readers
    /// before freeing GPU memory.
    pub fn markEvictingLocked(self: *ModelRegistry, entry: *LoadedModel) void {
        std.debug.assert(entry.state == .ready);
        entry.state = .evicting;
        self.state_cond.broadcast(self.io);
    }

    /// Phase D: block until `entry.refcount == 0`. Caller holds `mutex`.
    /// Wakes when any `release` lands on a zero refcount.
    pub fn waitForRefcountZeroLocked(self: *ModelRegistry, entry: *LoadedModel) void {
        while (entry.refcount.load(.acquire) != 0) {
            self.state_cond.waitUncancelable(self.io, &self.mutex);
        }
    }

    /// Phase D: finalize an eviction after `unloadResident()` has been
    /// called on the inference thread. Caller holds `mutex`. Updates the
    /// summed-bytes accounting and flips state to `.unloaded`.
    pub fn finalizeEvictionLocked(self: *ModelRegistry, entry: *LoadedModel) void {
        // bytes_resident was zeroed by unloadResident already; we still
        // subtract its previous accounting here. Because we already
        // synced `current_resident_bytes` at markReady time, just clear.
        // Defensive: cap subtraction at zero in case of double-call.
        // (unloadResident sets bytes_resident=0 itself, so we tracked the
        // pre-eviction value externally — pass-through here is a no-op.)
        entry.state = .unloaded;
        entry.error_name = null;
        self.state_cond.broadcast(self.io);
    }

    /// Phase D: count of currently-loaded (`.ready` or `.evicting`) entries.
    /// Caller holds `mutex`. Used by the cap-check before starting a load.
    pub fn countLoadedLocked(self: *const ModelRegistry) u32 {
        var n: u32 = 0;
        var it = self.entries.valueIterator();
        while (@constCast(&it).next()) |entry_ptr| {
            const entry = entry_ptr.*;
            if (entry.state == .ready or entry.state == .evicting) n += 1;
        }
        return n;
    }

    /// Phase D: subtract `bytes` from the resident accounting. Caller holds
    /// `mutex`. Used after `unloadResident()` to keep `current_resident_bytes`
    /// in sync with the actual GPU footprint.
    pub fn accountEvictedLocked(self: *ModelRegistry, bytes: u64) void {
        if (self.current_resident_bytes >= bytes) {
            self.current_resident_bytes -= bytes;
        } else {
            self.current_resident_bytes = 0;
        }
    }

    /// Release a borrowed pointer obtained from `ensureLoaded`. Decrements
    /// the refcount and bumps `last_used_ns` so LRU picks an older entry
    /// over this one next. Wakes anyone waiting on eviction.
    pub fn release(self: *ModelRegistry, lm: *LoadedModel) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.lru_clock += 1;
        lm.last_used_ns = self.lru_clock;
        const prev = lm.refcount.fetchSub(1, .acq_rel);
        std.debug.assert(prev > 0);
        // Broadcast so an evictor waiting for refcount == 0 wakes.
        if (prev == 1) self.state_cond.broadcast(self.io);
    }

    /// Pick the least-recently-used `.ready` entry whose refcount is 0,
    /// excluding `exclude_id` (typically the entry the caller is about to
    /// load — never evict ourselves). Returns null if no evictable entry
    /// exists. Caller holds `mutex`.
    pub fn pickLruEvictable(self: *ModelRegistry, exclude_id: []const u8) ?*LoadedModel {
        var best: ?*LoadedModel = null;
        var best_used: i64 = std.math.maxInt(i64);
        var it = self.entries.valueIterator();
        while (it.next()) |entry_ptr| {
            const entry = entry_ptr.*;
            if (entry.state != .ready) continue;
            if (entry.refcount.load(.acquire) != 0) continue;
            if (std.mem.eql(u8, entry.id, exclude_id)) continue;
            if (entry.last_used_ns < best_used) {
                best_used = entry.last_used_ns;
                best = entry;
            }
        }
        return best;
    }

    /// Mark an entry as `.ready` after the inference thread finishes
    /// loading. Must be called under `mutex`; broadcasts on the cond-var
    /// so blocked `ensureLoaded` callers wake. Caller has already
    /// populated weights/transformer/etc on `entry`; this just updates
    /// the bookkeeping + state field.
    pub fn markReadyLocked(self: *ModelRegistry, entry: *LoadedModel, bytes_resident: u64) void {
        self.releaseReservationLocked(entry); // pending estimate → actual residency
        entry.bytes_resident = bytes_resident;
        entry.state = .ready;
        entry.error_name = null;
        self.lru_clock += 1;
        entry.last_used_ns = self.lru_clock;
        self.current_resident_bytes += bytes_resident;
        // Headless default promotion: a server started without --model has no
        // default, so requests addressing the "mlx-serve" alias (the app's
        // chat/avatar surfaces, Claude Code) 503 with no_model even after the
        // user loads a chat model via /v1/load-model — the live gen-first→
        // chat-later hole (2026-07-05). The FIRST chat-capable model to finish
        // loading becomes the default; media engines and embedding encoders
        // never qualify, and an existing default is never stolen.
        if (self.default_id.len == 0 and chatCapable(entry)) {
            self.default_id = entry.id;
            log.info("[registry] default model -> {s} (first chat-capable load on a headless server)\n", .{entry.id});
        }
        self.state_cond.broadcast(self.io);
    }

    /// Can this READY entry serve chat/completions? Media models (image/
    /// audio/video/mesh — identified by model_type, so no engine pointers
    /// are needed) and embedding encoders cannot.
    fn chatCapable(entry: *const LoadedModel) bool {
        const cfg = entry.config orelse return false;
        if (cfg.is_encoder_only) return false;
        if (model_discovery.isMediaModelType(cfg.model_type)) return false;
        return true;
    }

    /// Reserve `estimated` bytes against `reserved_bytes` for an entry that has
    /// just claimed `.loading`. Counted in the eviction gate so concurrent
    /// loads see this pending allocation. Released by markReady/Error/Unloaded.
    /// Caller holds `mutex`.
    pub fn reserveLoadLocked(self: *ModelRegistry, entry: *LoadedModel, estimated: u64) void {
        // Release any stale prior reservation first (defensive; should be 0).
        self.releaseReservationLocked(entry);
        entry.load_estimate = estimated;
        self.reserved_bytes += estimated;
    }

    /// Drop an entry's in-flight load reservation (no-op if it never reserved).
    /// Caller holds `mutex`.
    fn releaseReservationLocked(self: *ModelRegistry, entry: *LoadedModel) void {
        if (entry.load_estimate == 0) return;
        if (self.reserved_bytes >= entry.load_estimate) {
            self.reserved_bytes -= entry.load_estimate;
        } else {
            self.reserved_bytes = 0;
        }
        entry.load_estimate = 0;
    }

    /// Undo a `markEvictingLocked` — return a victim that was selected for
    /// eviction back to `.ready` (used when an eviction PLAN can't be fully
    /// satisfied and must roll back). Caller holds `mutex`.
    pub fn unmarkEvictingLocked(self: *ModelRegistry, entry: *LoadedModel) void {
        std.debug.assert(entry.state == .evicting);
        entry.state = .ready;
        self.state_cond.broadcast(self.io);
    }

    /// Select LRU victims to evict so that, once `entry` (already `.loading`,
    /// with its estimate reserved) becomes resident, both caps hold. Marks each
    /// chosen victim `.evicting` and writes it into `out`; returns the count,
    /// or null if the caps can't be met (no more evictable victims, or `out`
    /// too small) — in which case any victims marked here are rolled back so
    /// the registry is left untouched. Caller holds `mutex` and must drain each
    /// returned victim's refcount, then hand them to the load request to free.
    pub fn planEvictionsLocked(self: *ModelRegistry, exclude_id: []const u8, out: []*LoadedModel) ?usize {
        var n: usize = 0;
        var freed: u64 = 0;
        while (true) {
            // Resident-after-plan = current minus what these victims free, plus
            // every in-flight reservation (including this load's own estimate).
            const projected_mem = (self.current_resident_bytes -| freed) + self.reserved_bytes;
            // Count: .ready+.evicting now, minus the victims we'll drop, plus
            // this load (currently `.loading`, becomes resident).
            const projected_count = self.countLoadedLocked() - @as(u32, @intCast(n)) + 1;
            const mem_ok = self.max_resident_mem == 0 or projected_mem <= self.max_resident_mem;
            const count_ok = projected_count <= self.max_resident_models;
            if (mem_ok and count_ok) return n;

            const victim = self.pickLruEvictable(exclude_id) orelse {
                // Can't satisfy the caps — roll back every marking we made.
                for (out[0..n]) |v| self.unmarkEvictingLocked(v);
                return null;
            };
            if (n >= out.len) {
                for (out[0..n]) |v| self.unmarkEvictingLocked(v);
                return null;
            }
            self.markEvictingLocked(victim); // now `.evicting` → pickLru won't repick
            freed += victim.bytes_resident;
            out[n] = victim;
            n += 1;
        }
    }

    /// Mark an entry as `.error_state` and store `error_name` (duped).
    /// Future `ensureLoaded` calls fail with `error.LoadFailed` until the
    /// entry is reset to `.unloaded` (Phase D may add a retry path).
    pub fn markErrorLocked(self: *ModelRegistry, entry: *LoadedModel, error_name: []const u8) void {
        self.releaseReservationLocked(entry);
        if (entry.error_name) |old| self.allocator.free(old);
        entry.error_name = self.allocator.dupe(u8, error_name) catch null;
        entry.state = .error_state;
        self.state_cond.broadcast(self.io);
    }

    /// Snapshot of every entry for `/v1/models`. Sort: default first
    /// (so single-model clients reading `data[0]` keep working), then by
    /// `last_used_ns` descending so the active model floats to the top of
    /// the rest. Slice memory is owned by `result_alloc`; the embedded
    /// string slices are borrowed from the registry (snapshot is taken
    /// under the lock, so caller must finish reading before any
    /// registry-mutating operation).
    pub fn snapshot(self: *ModelRegistry, result_alloc: std.mem.Allocator) ![]ModelStatus {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);

        const out = try result_alloc.alloc(ModelStatus, self.entries.count());
        var idx: usize = 0;
        var it = self.entries.valueIterator();
        while (it.next()) |entry_ptr| {
            const entry = entry_ptr.*;
            out[idx] = .{
                .id = entry.id,
                .loaded = entry.state == .ready,
                .bytes_resident = entry.bytes_resident,
                .bytes_on_disk = entry.bytes_on_disk,
                .last_used_ns = entry.last_used_ns,
                .state = ModelStatus.stateName(entry.state),
                .error_name = entry.error_name,
            };
            idx += 1;
        }

        // Bring default to front; sort the rest by last_used_ns desc.
        const default_id = self.default_id;
        std.sort.pdq(ModelStatus, out, default_id, lessThanByDefaultThenRecent);
        return out;
    }

    fn lessThanByDefaultThenRecent(default_id: []const u8, a: ModelStatus, b: ModelStatus) bool {
        const a_def = std.mem.eql(u8, a.id, default_id);
        const b_def = std.mem.eql(u8, b.id, default_id);
        if (a_def != b_def) return a_def;
        return a.last_used_ns > b.last_used_ns;
    }
};

// ── Tests ─────────────────────────────────────────────────────────────────
//
// Phase A tests exercise the registry's bookkeeping (state transitions,
// refcount math, LRU pick, snapshot output) without invoking the real
// mlx-allocating load path. We synthesize `.ready` entries directly to
// keep tests fast and free of GPU dependencies; the real load path is
// covered by integration tests once Phase D lands.

const testing = std.testing;

fn makeReadyStub(reg: *ModelRegistry, id: []const u8, bytes: u64) !*LoadedModel {
    const stub = try reg.registerStub(id, id, bytes);
    reg.mutex.lockUncancelable(reg.io);
    defer reg.mutex.unlock(reg.io);
    reg.markReadyLocked(stub, bytes);
    return stub;
}

test "ModelRegistry: init/deinit empty" {
    var reg = try ModelRegistry.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 3, 0, null);
    defer reg.deinit();
    try testing.expectEqual(@as(usize, 0), reg.entries.count());
    try testing.expectEqual(@as(u64, 0), reg.current_resident_bytes);
}

test "ModelRegistry: registerStub + setDefault" {
    var reg = try ModelRegistry.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 3, 0, null);
    defer reg.deinit();
    _ = try reg.registerStub("foo", "/path/to/foo", 1024);
    try reg.setDefault("foo");
    try testing.expectEqualStrings("foo", reg.default_id);
    try testing.expectError(error.DuplicateId, reg.registerStub("foo", "/path/to/foo", 1024));
    try testing.expectError(error.UnknownModelId, reg.setDefault("bar"));
}

test "ModelRegistry: registerStubWithArch keeps the discovery arch hint" {
    var reg = try ModelRegistry.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 3, 0, null);
    defer reg.deinit();
    const encoder = try reg.registerStubWithArch("bge", "/path/to/bge", 64, "bert");
    try testing.expectEqualStrings("bert", encoder.arch_hint);
    // Plain registerStub keeps an empty hint (no arch known).
    const plain = try reg.registerStub("foo", "/path/to/foo", 1024);
    try testing.expectEqualStrings("", plain.arch_hint);
}

test "ModelRegistry: registerByPath reuses an existing id without touching the filesystem" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var reg = try ModelRegistry.init(testing.allocator, io, null, 3, 0, null);
    defer reg.deinit();
    const stub = try reg.registerStubWithArch("bge-x", "/models/bge-x", 64, "bert");
    // The path's parent doesn't exist — proves the fast path resolves by
    // basename before any probe.
    const id = try reg.registerByPath(io, "/nonexistent/parent/bge-x/");
    try testing.expectEqualStrings("bge-x", id);
    try testing.expectEqual(stub.id.ptr, id.ptr);
}

test "ModelRegistry: peekByPath dedupes org/name discovery ids against basename registration" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var reg = try ModelRegistry.init(testing.allocator, io, null, 3, 0, null);
    defer reg.deinit();
    // Discovery registered this model under its org/name id (two-level scan).
    const stub = try reg.registerStubWithArch("mlx-community/gemma-x", "/models/mlx-community/gemma-x", 64, "gemma4");
    // The --model path registration must find the SAME entry via its path
    // (basename "gemma-x" doesn't match the org/name id) — otherwise the
    // same weights end up registered twice and can double-load.
    const by_path = reg.peekByPath("/models/mlx-community/gemma-x");
    try testing.expect(by_path != null);
    try testing.expectEqual(stub, by_path.?);
    // Trailing slash normalizes.
    try testing.expectEqual(stub, reg.peekByPath("/models/mlx-community/gemma-x/").?);
    try testing.expect(reg.peekByPath("/models/elsewhere") == null);
    // registerByPath also resolves through the path before creating a stub
    // (basename fast path misses, path match hits, no filesystem probe).
    const id = try reg.registerByPath(io, "/models/mlx-community/gemma-x/");
    try testing.expectEqualStrings("mlx-community/gemma-x", id);
    try testing.expectEqual(@as(usize, 1), reg.entries.count());
}

test "ModelRegistry: registerByPath rejects a nonexistent directory" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var reg = try ModelRegistry.init(testing.allocator, io, null, 3, 0, null);
    defer reg.deinit();
    try testing.expectError(error.ModelDirNotFound, reg.registerByPath(io, "/nonexistent/parent/some-model"));
    try testing.expectError(error.InvalidModelPath, reg.registerByPath(io, "/"));
}

test "ModelRegistry: ensureLoaded fails on unloaded stub" {
    var reg = try ModelRegistry.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 3, 0, null);
    defer reg.deinit();
    _ = try reg.registerStub("foo", "/path/to/foo", 1024);
    try reg.setDefault("foo");
    try testing.expectError(error.NotLoaded, reg.ensureLoaded("foo"));
    try testing.expectError(error.UnknownModelId, reg.ensureLoaded("bar"));
}

test "ModelRegistry: ensureLoaded + release refcount math" {
    var reg = try ModelRegistry.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 3, 0, null);
    defer reg.deinit();
    const lm = try makeReadyStub(reg, "foo", 1024);
    try reg.setDefault("foo");

    try testing.expectEqual(@as(u32, 0), lm.refcount.load(.acquire));
    const a = try reg.ensureLoaded("foo");
    try testing.expectEqual(lm, a);
    try testing.expectEqual(@as(u32, 1), lm.refcount.load(.acquire));
    const b = try reg.ensureLoaded("foo");
    try testing.expectEqual(@as(u32, 2), lm.refcount.load(.acquire));

    reg.release(a);
    try testing.expectEqual(@as(u32, 1), lm.refcount.load(.acquire));
    reg.release(b);
    try testing.expectEqual(@as(u32, 0), lm.refcount.load(.acquire));
}

test "ModelRegistry: default routing on empty / mlx-serve" {
    var reg = try ModelRegistry.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 3, 0, null);
    defer reg.deinit();
    const lm = try makeReadyStub(reg, "foo", 1024);
    try reg.setDefault("foo");

    const a = try reg.ensureLoaded("");
    try testing.expectEqual(lm, a);
    reg.release(a);

    const b = try reg.ensureLoaded("mlx-serve");
    try testing.expectEqual(lm, b);
    reg.release(b);
}

test "ModelRegistry: ensureLoaded fails when no default and id empty" {
    var reg = try ModelRegistry.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 3, 0, null);
    defer reg.deinit();
    _ = try makeReadyStub(reg, "foo", 1024);
    // default_id never set
    try testing.expectError(error.NoDefaultModel, reg.ensureLoaded(""));
}

test "ModelRegistry: ensureLoaded reports error_state" {
    var reg = try ModelRegistry.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 3, 0, null);
    defer reg.deinit();
    const stub = try reg.registerStub("broken", "/path/to/broken", 1024);
    try reg.setDefault("broken");
    reg.mutex.lockUncancelable(reg.io);
    reg.markErrorLocked(stub, "MissingVisionWeights");
    reg.mutex.unlock(reg.io);
    try testing.expectError(error.LoadFailed, reg.ensureLoaded("broken"));
    try testing.expect(stub.error_name != null);
    try testing.expectEqualStrings("MissingVisionWeights", stub.error_name.?);
}

test "ModelRegistry: pickLruEvictable orders by last_used_ns, ignores refcounted" {
    var reg = try ModelRegistry.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 3, 0, null);
    defer reg.deinit();

    const a = try makeReadyStub(reg, "a", 100);
    const b = try makeReadyStub(reg, "b", 200);
    const c = try makeReadyStub(reg, "c", 300);

    // Touch order: a (oldest via init), then b, then c (most recent).
    // markReadyLocked already bumped lru_clock in registration order, but
    // be explicit by re-releasing each so last_used_ns reflects test intent.
    _ = a;
    _ = b;
    _ = c;
    const aa = try reg.ensureLoaded("a");
    reg.release(aa); // a now newest
    const bb = try reg.ensureLoaded("b");
    reg.release(bb); // b now newest, a oldest
    const cc = try reg.ensureLoaded("c");
    reg.release(cc); // c now newest, a oldest

    {
        reg.mutex.lockUncancelable(reg.io);
        defer reg.mutex.unlock(reg.io);
        const lru = reg.pickLruEvictable("").?;
        try testing.expectEqualStrings("a", lru.id);
    }

    // Hold a refcount on `a` — eviction should now skip it and pick `b`.
    const held = try reg.ensureLoaded("a");
    {
        reg.mutex.lockUncancelable(reg.io);
        defer reg.mutex.unlock(reg.io);
        const lru = reg.pickLruEvictable("").?;
        try testing.expectEqualStrings("b", lru.id);
    }
    reg.release(held);

    // Exclude `a` explicitly — pick `b` even with `a` released.
    {
        reg.mutex.lockUncancelable(reg.io);
        defer reg.mutex.unlock(reg.io);
        const lru = reg.pickLruEvictable("a").?;
        try testing.expectEqualStrings("b", lru.id);
    }
}

test "ModelRegistry: snapshot places default first then most-recent" {
    var reg = try ModelRegistry.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 3, 0, null);
    defer reg.deinit();

    _ = try makeReadyStub(reg, "a", 100);
    _ = try makeReadyStub(reg, "b", 200);
    const c = try makeReadyStub(reg, "c", 300);
    try reg.setDefault("b");

    // Touch c last so it floats to the top of non-default entries.
    const cc = try reg.ensureLoaded("c");
    reg.release(cc);
    _ = c;

    const snap = try reg.snapshot(testing.allocator);
    defer testing.allocator.free(snap);

    try testing.expectEqual(@as(usize, 3), snap.len);
    try testing.expectEqualStrings("b", snap[0].id);     // default first
    try testing.expectEqualStrings("c", snap[1].id);     // most-recent of rest
    try testing.expectEqualStrings("a", snap[2].id);
    try testing.expect(snap[0].loaded);
    try testing.expectEqual(@as(u64, 200), snap[0].bytes_resident);
    try testing.expectEqualStrings("ready", snap[0].state);
}

test "ModelRegistry: snapshot reports unloaded entries" {
    var reg = try ModelRegistry.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 3, 0, null);
    defer reg.deinit();

    _ = try reg.registerStub("ghost", "/path/ghost", 4096);
    _ = try makeReadyStub(reg, "live", 1024);
    try reg.setDefault("live");

    const snap = try reg.snapshot(testing.allocator);
    defer testing.allocator.free(snap);

    try testing.expectEqual(@as(usize, 2), snap.len);
    try testing.expectEqualStrings("live", snap[0].id);
    try testing.expect(snap[0].loaded);
    try testing.expectEqualStrings("ghost", snap[1].id);
    try testing.expect(!snap[1].loaded);
    try testing.expectEqualStrings("unloaded", snap[1].state);
    try testing.expectEqual(@as(?u64, 4096), snap[1].bytes_on_disk);
}

test "ModelRegistry: markReadyLocked sums resident bytes" {
    var reg = try ModelRegistry.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 3, 0, null);
    defer reg.deinit();
    _ = try makeReadyStub(reg, "a", 100);
    _ = try makeReadyStub(reg, "b", 250);
    try testing.expectEqual(@as(u64, 350), reg.current_resident_bytes);
}

test "ModelRegistry: peek does not refcount" {
    var reg = try ModelRegistry.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 3, 0, null);
    defer reg.deinit();
    const lm = try makeReadyStub(reg, "foo", 1024);
    const peeked = reg.peek("foo").?;
    try testing.expectEqual(lm, peeked);
    try testing.expectEqual(@as(u32, 0), lm.refcount.load(.acquire));
    try testing.expectEqual(@as(?*LoadedModel, null), reg.peek("nope"));
}

// ── Eviction planner + reservation accounting (oversubscription fix) ──
//
// These pin the invariant that prevented the Metal-OOM crash: a cold load
// reserves its estimate so a concurrent load sees the pending allocation, and
// the planner evicts enough LRU victims to fit (multi-victim) or fails cleanly
// (→ 503) rather than loading anyway and oversubscribing GPU memory.

/// Claim `.loading` + reserve `estimate` for a fresh stub, the way the
/// scheduler's ensureLoaded slow path does. Returns the loading entry.
fn beginLoad(reg: *ModelRegistry, id: []const u8, estimate: u64) !*LoadedModel {
    const stub = try reg.registerStub(id, id, estimate);
    reg.mutex.lockUncancelable(reg.io);
    defer reg.mutex.unlock(reg.io);
    try testing.expect(reg.tryBeginLoadLocked(stub));
    reg.reserveLoadLocked(stub, estimate);
    return stub;
}

test "planEvictions: evicts one LRU victim to fit the memory budget" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var reg = try ModelRegistry.init(testing.allocator, io, null, 10, 100, null);
    defer reg.deinit();
    _ = try makeReadyStub(reg, "a", 40); // oldest
    _ = try makeReadyStub(reg, "b", 40);
    try testing.expectEqual(@as(u64, 80), reg.current_resident_bytes);

    const c = try beginLoad(reg, "c", 40); // 80 + 40 = 120 > 100 → must evict
    try testing.expectEqual(@as(u64, 40), reg.reserved_bytes);

    reg.mutex.lockUncancelable(io);
    defer reg.mutex.unlock(io);
    var buf: [16]*LoadedModel = undefined;
    const n = reg.planEvictionsLocked(c.id, &buf).?;
    try testing.expectEqual(@as(usize, 1), n);
    try testing.expectEqualStrings("a", buf[0].id); // LRU victim
    try testing.expectEqual(LoadState.evicting, buf[0].state);
}

test "planEvictions: multi-victim — evicts as many as needed to fit" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var reg = try ModelRegistry.init(testing.allocator, io, null, 10, 100, null);
    defer reg.deinit();
    _ = try makeReadyStub(reg, "a", 40);
    _ = try makeReadyStub(reg, "b", 40);
    _ = try makeReadyStub(reg, "c", 40); // current = 120 (budget shrank under us)

    const d = try beginLoad(reg, "d", 40); // need to free 2×40 to fit 40 in 100
    reg.mutex.lockUncancelable(io);
    defer reg.mutex.unlock(io);
    var buf: [16]*LoadedModel = undefined;
    const n = reg.planEvictionsLocked(d.id, &buf).?;
    try testing.expectEqual(@as(usize, 2), n); // a and b (the two oldest)
}

test "planEvictions: returns null and rolls back when every victim is pinned" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var reg = try ModelRegistry.init(testing.allocator, io, null, 10, 100, null);
    defer reg.deinit();
    const a = try makeReadyStub(reg, "a", 80);
    _ = a.refcount.fetchAdd(1, .acq_rel); // pinned by an in-flight request

    const b = try beginLoad(reg, "b", 80); // 80 + 80 = 160 > 100 → must evict a
    reg.mutex.lockUncancelable(io);
    var buf: [16]*LoadedModel = undefined;
    const plan = reg.planEvictionsLocked(b.id, &buf);
    try testing.expectEqual(@as(?usize, null), plan); // can't evict the pinned victim
    try testing.expectEqual(LoadState.ready, a.state); // rolled back, not left .evicting
    // Scheduler then rolls back the load → releases the reservation.
    reg.markUnloadedLocked(b);
    reg.mutex.unlock(io);
    try testing.expectEqual(@as(u64, 0), reg.reserved_bytes);
    a.refcount.store(0, .release);
}

test "reservation: concurrent in-flight load is visible in the budget gate" {
    // The crash's core: load #2's gate must SEE load #1's pending bytes, even
    // though #1 hasn't reached markReady. With both reserved, the gate trips.
    const io = std.Io.Threaded.global_single_threaded.io();
    var reg = try ModelRegistry.init(testing.allocator, io, null, 10, 100, null);
    defer reg.deinit();
    _ = try beginLoad(reg, "a", 60); // in-flight, not yet ready: reserved=60
    _ = try beginLoad(reg, "b", 60); // in-flight too: reserved=120
    try testing.expectEqual(@as(u64, 120), reg.reserved_bytes);
    try testing.expectEqual(@as(u64, 0), reg.current_resident_bytes);
    // A third load sees reserved=120 (+its own) → over the 100 budget, and with
    // no `.ready` victim to evict, the plan fails (→ 503) instead of loading.
    const c = try beginLoad(reg, "c", 60);
    reg.mutex.lockUncancelable(io);
    defer reg.mutex.unlock(io);
    var buf: [16]*LoadedModel = undefined;
    try testing.expectEqual(@as(?usize, null), reg.planEvictionsLocked(c.id, &buf));
}

test "reservation: released back to zero on markReady / markUnloaded" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var reg = try ModelRegistry.init(testing.allocator, io, null, 10, 0, null);
    defer reg.deinit();
    const a = try beginLoad(reg, "a", 500);
    try testing.expectEqual(@as(u64, 500), reg.reserved_bytes);
    reg.mutex.lockUncancelable(io);
    reg.markReadyLocked(a, 480); // actual differs from estimate
    reg.mutex.unlock(io);
    try testing.expectEqual(@as(u64, 0), reg.reserved_bytes); // reservation cleared
    try testing.expectEqual(@as(u64, 480), reg.current_resident_bytes); // actual counted
    try testing.expectEqual(@as(u64, 0), a.load_estimate);
}

test "ModelRegistry: first chat-capable ready load becomes the default on a headless server" {
    // gen-first→chat-later hole (live 2026-07-05): a server started headless
    // (no --model) has no default, so requests addressing the "mlx-serve"
    // alias 503 with no_model even after the user loads a chat model via
    // /v1/load-model. The FIRST chat-capable load is promoted; media engines
    // and embedding encoders never qualify; an existing default is never
    // stolen.
    var reg = try ModelRegistry.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 8, 0, null);
    defer reg.deinit();

    // A ready MEDIA model must not become the default.
    const mesh = try reg.registerStub("hy3d", "/m/hy3d", 64);
    var mesh_cfg = model_mod.ModelConfig{};
    mesh_cfg.model_type = "hunyuan3d_2_1";
    mesh.config = &mesh_cfg;
    reg.mutex.lockUncancelable(reg.io);
    reg.markReadyLocked(mesh, 64);
    reg.mutex.unlock(reg.io);
    try testing.expectEqualStrings("", reg.default_id);

    // A ready embedding ENCODER must not become the default.
    const bge = try reg.registerStub("bge", "/m/bge", 64);
    var bge_cfg = model_mod.ModelConfig{};
    bge_cfg.model_type = "bert";
    bge_cfg.is_encoder_only = true;
    bge.config = &bge_cfg;
    reg.mutex.lockUncancelable(reg.io);
    reg.markReadyLocked(bge, 64);
    reg.mutex.unlock(reg.io);
    try testing.expectEqualStrings("", reg.default_id);

    // The first chat-capable load IS promoted...
    const chat = try reg.registerStub("gemma", "/m/gemma", 64);
    var chat_cfg = model_mod.ModelConfig{};
    chat_cfg.model_type = "gemma4";
    chat.config = &chat_cfg;
    reg.mutex.lockUncancelable(reg.io);
    reg.markReadyLocked(chat, 64);
    reg.mutex.unlock(reg.io);
    try testing.expectEqualStrings("gemma", reg.default_id);

    // ...and the alias resolves to it now.
    const via_alias = try reg.ensureLoaded("mlx-serve");
    try testing.expectEqual(chat, via_alias);
    reg.release(via_alias);

    // A LATER chat load never steals an existing default.
    const chat2 = try reg.registerStub("qwen", "/m/qwen", 64);
    var chat2_cfg = model_mod.ModelConfig{};
    chat2_cfg.model_type = "qwen3";
    chat2.config = &chat2_cfg;
    reg.mutex.lockUncancelable(reg.io);
    reg.markReadyLocked(chat2, 64);
    reg.mutex.unlock(reg.io);
    try testing.expectEqualStrings("gemma", reg.default_id);

    // The configs are STACK-allocated test doubles — detach them before
    // registry deinit tries to free entry-owned configs.
    mesh.config = null;
    bge.config = null;
    chat.config = null;
    chat2.config = null;
}
