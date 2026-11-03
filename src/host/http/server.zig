const std = @import("std");
const env = @import("../../env.zig");
const crt = @import("../../sdk/util/crt.zig");
const win_io = @import("../../sdk/util/win_io.zig");
const builtin = @import("builtin");
const mlx = @import("../../mlx/ffi.zig");
const kv_quant_mod = @import("../../sdk/blocks/attention/kv_quant.zig");
const tokenizer_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const generation_settings = @import("../api/generation_settings.zig");
const mtp_mod = @import("../../sdk/spec/mtp/mtp.zig");
const mtp_acceptance_mod = @import("../../sdk/spec/mtp_acceptance.zig");
const chat_mod = @import("../chat/chat.zig");
const rp_mod = @import("../../sdk/grammar/protocol.zig");
const token_mask = @import("../../sdk/grammar/token_mask.zig");
const model_mod = @import("../../sdk/config/model.zig");
const qwen_vision = @import("../../sdk/vision/qwen_vision.zig");
const mrope_mod = @import("../../sdk/vision/mrope.zig");
const log = @import("../../sdk/util/log.zig");
const responses_mod = @import("../api/responses/model.zig");
const pld_index = @import("../../sdk/spec/pld_index.zig");
const tokenize_cache_mod = @import("../tokenizer/cache.zig");
const Scheduler = @import("../gpu_thread/scheduler.zig").Scheduler;
const LoadParams = @import("../gpu_thread/params.zig").LoadParams;
const LlamaSettings = @import("../../sdk/quant/gguf/llama_settings.zig").Settings;
const mlx_engine = @import("../engines/mlx.zig");
const model_registry_mod = @import("../registry/registry.zig");
const model_discovery = @import("../registry/discovery.zig");
const mlx_gguf = @import("../../sdk/quant/gguf/claim.zig");
const media_mod = @import("../media_jobs/gen.zig");
const transcription_mod = @import("../api/audio/transcription.zig");
const stb = @import("stb");
const webp = @import("webp");
const metrics = @import("../status.zig");
const instr = @import("../metrics/metrics.zig");

const Tokenizer = tokenizer_mod.Tokenizer;
const ModelRegistry = model_registry_mod.ModelRegistry;
const LoadedModel = model_registry_mod.LoadedModel;
/// Global flag set by signal handler for graceful shutdown.
pub var shutdown_requested = std.atomic.Value(bool).init(false);
/// Number of live per-connection threads. Incremented before each spawn,
/// decremented when the thread's handler returns. On shutdown, `serve` waits
/// for this to reach 0 before returning (which runs `scheduler.deinit`) — a
/// detached conn thread still in `Scheduler.complete` would otherwise race
/// deinit's teardown of the slot queues into a use-after-free (SIGSEGV).
pub var active_conn_threads = std.atomic.Value(u32).init(0);
/// Set from main.zig before serve() is called when --metrics is on; null
/// otherwise. Gates the gauge-sampler thread and the /metrics + /metrics.json
/// routes. When null, `/metrics*` return 503, which the console reads as "metrics off".
pub var g_metrics: ?*instr.Metrics = null;
/// Optional global API key (`--api-key`). When set, every NON-LOOPBACK request
/// (i.e. from another machine over the network) except the `/health` probe and
/// CORS preflight requires the key — the OpenAI/Anthropic/Ollama APIs AND the
/// index page + metrics panel/feed. Loopback (127.0.0.0/8, ::1) is TRUSTED and
/// exempt, so the local app + a local browser keep working with no credentials;
/// the key protects the surface that actually matters — network exposure
/// (`--host 0.0.0.0`). Accepted as `Authorization: Bearer <key>`, `x-api-key:
/// <key>` (Anthropic), HTTP Basic (browser pages — the key is the password), or
/// an `?api_key=` / `?key=` query param. null ⇒ fully open (default, unchanged).
pub var g_api_key: ?[]const u8 = null;

/// `--api-key-strict`: drop the loopback exemption, so the key is required
/// from 127.0.0.1 too (`/health` and CORS preflight stay open). For hosts
/// where localhost is NOT one trust domain — an embedding application that
/// generates a fresh key per start wants "only the process holding the key
/// can drive inference", and loopback exemption made any local process (and
/// any browser page via a simple no-preflight POST) a key holder. No effect
/// unless `--api-key` is set.
pub var g_api_key_strict: bool = false;

/// Tool-call ARGUMENT auto-correct. When true (default), parsed tool-call
/// arguments are coerced to the types the request's tool schema declares
/// (`chat.coerceToolArgsToSchema` — e.g. Python's `False` → JSON `false`, an
/// `edits` array shipped as a string → a real array). `--no-tool-autocorrect`
/// turns it OFF: arguments pass through exactly as the model emitted them (still
/// valid JSON — the parse-repair + safety net always run; this gates ONLY the
/// schema-type coercion, the opinionated layer). Off means a weak model's
/// mistyped value reaches the client verbatim, which strict clients reject.
pub var g_tool_autocorrect: bool = true;

/// LAN model sharing (src/host/lan/lan.zig). Started by `serve()` when `--lan-share`
/// and/or `--lan-discover` are set (the three `g_lan_*` config globals below
/// are written by main.zig, mirroring the `g_api_key` pattern). When sharing
/// is on and NO api-key is configured, non-loopback requests are limited to
/// the shared inference surface (`lanShareDenial`); when discovery is on,
/// requests naming a `<id>@<peer>` model are tunneled to that peer.
pub var g_lan: ?*lan_mod.Lan = null;
pub var g_lan_share_spec: ?[]const u8 = null;
pub var g_lan_name: ?[]const u8 = null;
pub var g_lan_discover: bool = false;

/// Upstream chat providers (src/host/api/providers.zig): `~/.mlx-serve/providers.json`,
/// rows mirrored into /v1/models as `<id>@<name>`, `/v1/chat/completions`
/// proxied with the provider's key. Started by `serve()` beside the LAN.
pub var g_providers: ?*providers_mod.Providers = null;

/// Should boot print the open-bind warning? True only when serve mode is about
/// to listen on a non-loopback address the user never chose: no explicit
/// `--host` (the default is still 0.0.0.0) and no `--lan-share` (which needs
/// the wide bind). A future release flips the default to 127.0.0.1 — at which
/// point the host check silences this without a code change.
pub fn shouldWarnOpenBind(host_explicit: bool, lan_share: bool, host: []const u8) bool {
    if (host_explicit or lan_share) return false;
    return !(std.mem.startsWith(u8, host, "127.") or
        std.mem.eql(u8, host, "::1") or
        std.mem.eql(u8, host, "localhost"));
}

test "shouldWarnOpenBind: warn only on an UNCHOSEN non-loopback bind" {
    // Default bind (0.0.0.0, nobody asked) → warn: a first-launch user is
    // serving whatever network the laptop joins.
    try std.testing.expect(shouldWarnOpenBind(false, false, "0.0.0.0"));
    // Someone who CHOSE the bind is not nagged — explicit --host (any value)…
    try std.testing.expect(!shouldWarnOpenBind(true, false, "0.0.0.0"));
    // …or --lan-share, which needs the wide bind by design.
    try std.testing.expect(!shouldWarnOpenBind(false, true, "0.0.0.0"));
    // Loopback defaults never warn (the future 127.0.0.1 default).
    try std.testing.expect(!shouldWarnOpenBind(false, false, "127.0.0.1"));
    try std.testing.expect(!shouldWarnOpenBind(false, false, "localhost"));
    try std.testing.expect(!shouldWarnOpenBind(false, false, "::1"));
}

const io_util = @import("../../sdk/util/io_util.zig");
const lan_mod = @import("../lan/lan.zig");
const launch_mod = @import("../launch/launch.zig");
const providers_mod = @import("../api/providers.zig");
const multipart = @import("multipart.zig");
const ws_mod = @import("ws.zig");
const ollama_mod = @import("../api/ollama/translate.zig");
const model_settings_mod = @import("../registry/model_settings.zig");
const cli_mod = @import("../cli/cli.zig");
const build_options = @import("build_options");
const nowSecs = io_util.nowSecs;
const nowMs = io_util.nowMs;
const nowMsMonotonic = io_util.nowMsMonotonic;
const Stopwatch = io_util.Stopwatch;
const request_model = @import("../api/request_model.zig");
const sampling = @import("../api/sampling.zig");
const admission = @import("../api/admission.zig");

pub const Conn = @import("conn.zig").Conn;
pub const KvAttnMode = @import("../api/request_options.zig").KvAttnMode;
pub const mtpAllowedByDefault = @import("../api/request_options.zig").mtpAllowedByDefault;
pub const PldDefaults = @import("../api/request_options.zig").PldDefaults;
pub const launchMaxTokensDefault = @import("../api/sampling.zig").launchMaxTokensDefault;
pub const deinitGlobalResponseStore = @import("../api/responses/store.zig").deinitGlobalResponseStore;
pub const listenExclusive = @import("listen.zig").listenExclusive;
pub const socketReadable = @import("listen.zig").socketReadable;
pub const parseIp4 = @import("listen.zig").parseIp4;
pub const StartupResponder = @import("listen.zig").StartupResponder;
pub const ConnThreadArgs = @import("request.zig").ConnThreadArgs;
pub const handleConnectionThread = @import("request.zig").handleConnectionThread;
pub const WIRED_LIMIT_MARGIN_BYTES = @import("../registry/memory_ceiling.zig").WIRED_LIMIT_MARGIN_BYTES;
pub const parseWiredMarginGib = @import("../registry/memory_ceiling.zig").parseWiredMarginGib;
pub const applyMlxCacheLimit = @import("../../mlx/state.zig").applyMlxCacheLimit;
pub const staticGpuMemoryCeiling = @import("../registry/memory_ceiling.zig").staticGpuMemoryCeiling;
pub const applyGpuCeilingEnv = @import("../registry/memory_ceiling.zig").applyGpuCeilingEnv;
pub const peerIsLoopback = @import("auth.zig").peerIsLoopback;
pub const visionPreprocFromConfig = @import("../media/prepare.zig").visionPreprocFromConfig;
pub const parseImageUrlContent = @import("../media/request_media.zig").parseImageUrlContent;
pub const decodeImageToPixels = @import("../media/decode.zig").decodeImageToPixels;

fn signalHandler(_: std.posix.SIG) callconv(.c) void {
    shutdown_requested.store(true, .release);
}

pub const ServerConfig = struct {
    /// Maximum context size (0 = unlimited). `--ctx-size N`.
    max_context_size: u32 = 0,
    /// Request timeout in seconds (0 = no timeout). `--timeout N`.
    request_timeout_sec: u32 = 300,
    /// Default reasoning budget in tokens (-1 = unlimited).
    /// `--reasoning-budget N`. Per-request body fields override.
    default_reasoning_budget: i32 = -1,
    /// Default PLD enabled state. Per-request `enable_pld` JSON overrides.
    default_enable_pld: bool = false,
    /// Maximum draft tokens proposed per PLD step.
    default_pld_draft_len: u32 = 5,
    /// N-gram match key length for PLD.
    default_pld_key_len: u32 = 3,
    /// docs/kv-quant-perf.md Phase 2: how quantized-KV requests READ the
    /// cache at decode. `--kv-attn-mode dense|fused|auto`; per-request
    /// `kv_attn_mode` body field ("dense"/"fused") outranks it. Only takes
    /// effect at `--kv-quant 4|8` (.affine cache scheme); other schemes
    /// always read dense. `auto` (the default) engages fused reads when the
    /// prompt is at/above the measured crossover — see `resolveKvAttnFused`.
    kv_attn_mode: KvAttnMode = .auto,
    /// Defaults for sampling fields the request OMITS, set by `--temp` /
    /// `--top-p` / `--top-k` in serve mode (the macOS app passes its Settings
    /// values so external clients like Claude Code — which send no sampling
    /// params at all — inherit them). null = flag not given; resolution falls
    /// through to the model's generation_config.json recommendation, then the
    /// hardcoded fallback. Request fields and generation-defaults rules outrank
    /// them (`resolveSampling`).
    default_temperature: ?f32 = null,
    default_top_p: ?f32 = null,
    default_top_k: ?u32 = null,
    /// Default `max_tokens` for a request that omits the field (`--max-tokens N` in serve mode;
    /// 0 = flag not given). Agent clients send nothing, and an omitted budget is the rest of
    /// the window, which the admission bill reserves for. Read through `launchMaxTokensDefault`.
    default_max_tokens: u32 = 0,
};

/// Process-wide under `-Dunits` (sdk/util/shared.zig): one copy in the host.
const ServerState = struct {
    /// Live process-wide config. Mutated by `serve()` at startup from its
    /// parameters; never written from a handler after that. Public so main.zig
    /// can supply defaults (notably the per-request override CLI flags don't
    /// flow through `serve()` arguments).
    server_config: ServerConfig = .{},
    /// `--wired-margin-gib` in bytes: what stays unplanned under a raised wired limit.
    wired_limit_margin_bytes: u64 = WIRED_LIMIT_MARGIN_BYTES,
};
/// The launch settings the engine's admission code reads beside the host, one copy for every compile unit (`Shared`).
pub const Launch = struct {
    global_scheduler: ?*Scheduler = null,
    configured_kv_quant: ?kv_quant_mod.KVQuantConfig = null,
    prefix_cache_capacity: u32 = 32,
    prefix_cache_ram_enabled: bool = true,
    prefix_cache_mem_bytes: u64 = PREFIX_CACHE_MEM_DEFAULT,
    prefix_cache_mem_explicit: bool = false,
    prefix_cache_disk_bytes: u64 = 0,
    ssm_checkpoint_stride: u32 = 256,
    ssm_checkpoint_max: u32 = 16,
};
pub const launch_shared = @import("../../sdk/util/shared.zig").Shared(Launch, "launch", .{});
pub inline fn launch() *Launch {
    return launch_shared.get();
}

pub const serverState_shared = @import("../../sdk/util/shared.zig").Shared(ServerState, "server_config", .{});
pub inline fn serverState() *ServerState {
    return serverState_shared.get();
}

pub fn getTimeoutNs() u64 {
    if (serverState().server_config.request_timeout_sec == 0) return 0;
    return @as(u64, serverState().server_config.request_timeout_sec) * std.time.ns_per_s;
}

/// Plan 01 Phase 2 — continuous-batching scheduler. Always non-null in
/// serve mode (set by `serve()`); null only before `serve()` runs. Every
/// inference request handler routes through this; the scheduler's
/// inference thread is the single mlx-call site.

/// The boot's parsed `--kv-quant`, published by `serve()` before `Scheduler.init` so a
/// load-time bill sees it (the scheduler does not exist yet during the load; asking it billed
/// the SSD-first session at bf16, ~8.6 GB of idle allowance lost at 786k).

/// The process-wide kv-quant config at any point: the scheduler's once it exists, the boot's
/// published value during the load, dense when no flag was given.
fn configuredKvQuant() kv_quant_mod.KVQuantConfig {
    if (launch().global_scheduler) |sch| return sch.kv_quant_config;
    return launch().configured_kv_quant orelse kv_quant_mod.KVQuantConfig.dense;
}

/// The width THIS model stores at: its `model-settings.json` override, else the process default.
pub fn configuredKvQuantFor(config: *const model_mod.ModelConfig) kv_quant_mod.KVQuantConfig {
    return config.kv_quant_override orelse configuredKvQuant();
}

/// The explicit context for THIS model: its `model-settings.json` `ctx_size`, else `--ctx-size`.
/// 0 = auto. Every reader of the manual context goes through here, never `server_config.max_context_size`.
pub fn manualContext(config: *const model_mod.ModelConfig) u32 {
    return if (config.ctx_override > 0) config.ctx_override else serverState().server_config.max_context_size;
}

/// Plan 05 — model registry. Always non-null in serve mode (set by
/// `serve()`); handleConnection resolves `model` body fields against this
/// per request via `ensureLoaded`/`release`.
pub var global_registry: ?*ModelRegistry = null;

/// Module-level capacity for plan 03 hot prefix cache. main.zig writes this
/// from `--prefix-cache-entries N` before calling `serve()`.
// Default 32: agent clients (Claude Code, the app's agent loop) interleave
// requests from several conversation roots — main thread, subagents, title
// generation — and a single-entry cache gets its long system-prompt prefix
// evicted by every interleaved request, forcing a full re-prefill each
// turn. The count cap is pure retention metadata; the byte budget
// (`--prefix-cache-mem`, default 2 GB) is what actually bounds memory,
// evicting LRU entries by size. 0 disables.

/// `--no-prefix-cache-ram` keeps the disk tier eligible without idle RAM retention.

pub const PREFIX_CACHE_MEM_DEFAULT: u64 = 2 * 1024 * 1024 * 1024;

/// Wave 1.B — hot prefix cache memory budget. The cache evicts LRU entries on
/// commit until `current_kv_bytes + new_bytes <= prefix_cache_mem_bytes`. The
/// default (2 GB) is generous for one or two long conversations on a Gemma 4
/// E4B-sized model and tiny relative to total wired-limit budget; tune via
/// `--prefix-cache-mem <N>{GB,MB}`. 0 disables the byte budget (count cap
/// from `--prefix-cache-entries` still applies). On qwen4_exp's RAM-first arm an
/// unset budget grows to one session at the working context (`defaultPrefixCacheAsk`).

/// Set by `--prefix-cache-mem`: an operator's number is used as given, even when it equals the default.

/// SSD tier for the hot prefix cache (`--prefix-cache-disk`). Committed KV
/// prefixes persist as chunked safetensors under
/// `~/.mlx-serve/kv-cache/<model-fingerprint>` and are restored across RAM
/// evictions AND server restarts instead of recomputed — a cold 30-50 s
/// long-context TTFT becomes a bounded SSD read. LRU-evicted to this byte
/// budget. v1 covers pure-attention archs; hybrid SSM state stays RAM-only.
///
/// DEFAULT OFF (`0`): the tier can hold gigabytes of KV on disk, so it's opt-in
/// — pass `--prefix-cache-disk <n>{KB,MB,GB}` to enable (10 GB is a sensible
/// value). The Swift app exposes this as a Settings toggle, default off.

/// `--shutdown-persist-secs`: on SIGTERM, finish the SSD slices the idle flush has not landed,
/// for at most this long after the signal. 0 turns it off.
pub var shutdown_persist_secs: u32 = 45;

/// Phase 1 (performance-plan): SSM/conv state snapshot stride during prefill,
/// in tokens. Non-zero values enable multi-turn warm reuse on hybrid SSM
/// architectures (Qwen3.5/3.6 GatedDeltaNet, Nemotron-H Mamba2, LFM2.5
/// gated-conv). Set to 0 to disable (fall back to pre-Phase-1 behavior:
/// hybrid models bypass the hot prefix cache entirely). Override via
/// `--ssm-checkpoint-stride N`.
///
/// Default 256. The stride forces a prefill CHUNK boundary at every multiple
/// (to snapshot SSM/sliding-window state mid-prompt). On dense / non-MoE-hybrid
/// models (dense Gemma sliding-window, LFM2, Nemotron-H) prefill is
/// compute-bound, so a fine 256 stride is ~free and buys finer warm mid-prompt
/// reuse — measured <3% cold cost on Qwen3.5-4B. **MoE models are different**:
/// their prefill is memory-bound on the per-expert weights, and every extra
/// chunk re-streams ~all expert weights from HBM, so a fine stride silently
/// costs ~25% cold prefill on 26B/35B-class MoE (an 850-token prompt = 4
/// chunks = ~4x expert-weight traffic). To avoid that, the prefill loop
/// **coarsens this stride to >= PREFILL_CHUNK for MoE models only** (see
/// `generate.effectiveSsmCheckpointStride`), so MoE prefill is never
/// over-chunked at any prompt length while non-MoE keeps the fine stride.
/// The always-on end-of-prompt snapshot preserves append-growth multi-turn
/// reuse for MoE regardless. Raise this (e.g. 512/1024) to also coarsen the
/// non-MoE path; set 0 to disable (hybrid models then bypass the hot cache).

/// Phase 1: cap on the number of checkpoints retained per request. Snapshots
/// past this drop the oldest (front of list) to keep memory bounded on very
/// long prompts. 0 = unlimited (rely on the prefix-cache byte budget alone).

/// Iteration 2 (perf-plan Phase 4 #3): LRU capacity of the per-LoadedModel
/// chat-template tokenize cache. 0 disables (every request re-renders +
/// re-tokenizes, restoring pre-Iteration-2 behavior). Default 4 is small
/// because real chat conversations mutate the messages list every turn;
/// the cache catches warm-reuse benches + repeated agent probes without
/// hoarding token buffers across a long session.
pub var tokenize_cache_entries: u32 = 4;

/// Embedded llama.cpp engine settings: `--llama-cache-entries` (sequences per
/// model), `--llama-kv-quant`, `--llama-ubatch`, `--llama-mtp-drafts`. Read when
/// a GGUF model loads.
pub var llama_settings: LlamaSettings = .{};

/// Plan 01 — continuous batching: maximum concurrent in-flight requests sharing
/// the inference thread's batched-decode pass. Set via `--max-concurrent N`.
/// Default 1 = legacy single-slot behavior (no scheduler engagement, every
/// existing test bit-identical). Values >1 require Phase 2's scheduler wiring
/// (handler refactor + per-connection threads), which the `serve()` startup
/// guard checks before enabling.
pub var max_concurrent: u32 = 1;

/// Issue #117: operator ceiling on per-input embedding length, in TOKENS.
/// 0 = auto (bound only by the loaded model's declared context window). Set
/// via `--embedding-max-length N`. Over-limit inputs earn a structured 400
/// naming the input index, its token count and the effective limit — never a
/// silent truncation (the Python-server `max_length=512` class). Applies to
/// every embedding route (/v1/embeddings, /api/embed, legacy /api/embeddings
/// — they all funnel into `handleEmbeddings`).
pub var embedding_max_length: u32 = 0;

pub fn serve(
    io: std.Io,
    allocator: std.mem.Allocator,
    /// Phase A1: model load happens on the scheduler's inference thread. The
    /// caller (main.zig) is responsible only for CPU-side setup (config parse,
    /// tokenizer load, EOS resolution, chat-template load); everything that
    /// touches mlx is done by `Scheduler.init` before this fn proceeds.
    load_params: LoadParams,
    config: *const model_mod.ModelConfig,
    host: []const u8,
    port: u16,
    cfg: ServerConfig,
) !void {
    serverState().server_config = cfg;
    // Before `Scheduler.init`: that call is the model load and every load-time bill inside it
    // asks for the KV width.
    launch().configured_kv_quant = load_params.kv_quant_config;
    defer launch().configured_kv_quant = null;

    // ── Phase A1: spin up the scheduler. Its inference thread does the
    //    Transformer/vision/drafter load, JIT compile, and warmup before
    //    `Scheduler.init` returns. From this point on, mlx ops are bound
    //    to the inference thread's GPU stream.
    // Bound before the startup load, which can take minutes (a first Neural Engine compile):
    // until it ends, `StartupResponder` answers in place of the accept loop below.
    const ip_addr: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = parseIp4(host), .port = port } };
    var server = listenExclusive(io, ip_addr) catch |err| {
        if (err == error.AddressInUse) log.err("Port {d} is already in use — another mlx-serve instance may be running.\n", .{port});
        return err;
    };
    defer server.deinit(io);
    var startup: StartupResponder = .{ .server = &server, .registry = load_params.registry, .allocator = allocator, .io = io };
    if (!load_params.no_initial_load) startup.start();
    defer startup.stop();

    var scheduler = try mlx_engine.start(allocator, io, load_params, max_concurrent);
    defer scheduler.deinit();
    startup.stop();
    launch().global_scheduler = scheduler;
    defer launch().global_scheduler = null;
    mlx_engine.api().install_admission_hooks(true);
    defer mlx_engine.api().install_admission_hooks(false);

    // Gauge sampler: samples instantaneous system + queue state every 2 s and
    // writes the metrics gauges. Only runs when --metrics is on. Trivial cost
    // (a few non-blocking syscalls + one brief queue_mu lock per 2 s); never on
    // the per-token decode path.
    // LIFO defer ordering: `stop` is registered AFTER `join` so it runs FIRST
    // (setting the flag), letting the thread exit before join() blocks. Do NOT
    // gate on scheduler.shutdown here — deinit() (which sets that flag) is the
    // earliest defer and runs LAST, after the join, so the loop would never see
    // it; the dedicated `sampler_stop` flag avoids that LIFO deadlock.
    var sampler_thread: ?std.Thread = null;
    var sampler_stop = std.atomic.Value(bool).init(false);
    if (g_metrics) |m| {
        const ctx = GaugeSamplerCtx{
            .metrics = m,
            .scheduler = scheduler,
            .stop = &sampler_stop,
        };
        sampler_thread = try std.Thread.spawn(.{}, gaugeSamplerLoop, .{ctx});
        log.info("Prometheus metrics: ENABLED — GET http://{s}:{d}/metrics (live panel at GET /)\n", .{ host, port });
    }
    // LIFO: join deferred first → runs second. stop deferred second → runs first.
    defer if (sampler_thread) |t| t.join();
    defer sampler_stop.store(true, .monotonic);

    // Idle eviction (--idle-evict-secs). Same lifetime shape as the sampler,
    // and never in `Scheduler.deinit` — see `IdleEvictCtx.stop`.
    var idle_evict_thread: ?std.Thread = null;
    var idle_evict_stop = std.atomic.Value(bool).init(false);
    if (scheduler.registry.idle_evict_secs) |secs| {
        idle_evict_thread = try std.Thread.spawn(.{}, idleEvictLoop, .{IdleEvictCtx{
            .scheduler = scheduler,
            .window_ms = @as(i64, secs) * 1000,
            .stop = &idle_evict_stop,
        }});
        log.info("[registry] idle eviction ON — unloading models idle for {d}s\n", .{secs});
    }
    defer if (idle_evict_thread) |t| t.join();
    defer idle_evict_stop.store(true, .monotonic);

    global_registry = scheduler.registry;
    defer global_registry = null;
    request_model.g_model_aliases.alloc = scheduler.registry.allocator;
    request_model.g_model_aliases.path = model_settings_mod.defaultPath(&request_model.g_model_aliases_path);
    sampling.g_generation_settings.alloc = scheduler.registry.allocator;
    sampling.g_generation_settings.path = model_settings_mod.homeFile(&sampling.g_generation_settings_path, "generation-settings.json");

    // Prefix cache lives on LoadedModel and may retain idle prefixes in RAM, SSD, or both.
    if (scheduler.hot_prefix_cache != null) {
        const ssm_note: []const u8 = if (config.has_hybrid_layers)
            " [hybrid: SSM checkpoints]"
        else
            "";
        if (!launch().prefix_cache_ram_enabled and launch().prefix_cache_disk_bytes > 0) {
            log.info("Prefix cache: SSD ONLY (RAM retention disabled, disk cap={d:.1} MB){s}\n", .{
                @as(f64, @floatFromInt(launch().prefix_cache_disk_bytes)) / (1024.0 * 1024.0),
                ssm_note,
            });
        } else if (admission.resolvedPrefixCacheMem() > 0) {
            const cap_mb = @as(f64, @floatFromInt(admission.resolvedPrefixCacheMem())) / (1024.0 * 1024.0);
            log.info("Hot prefix cache: ENABLED (capacity={d}, mem-cap={d:.1} MB){s}\n", .{ launch().prefix_cache_capacity, cap_mb, ssm_note });
        } else {
            log.info("Hot prefix cache: ENABLED (capacity={d}, mem-cap=unlimited){s}\n", .{ launch().prefix_cache_capacity, ssm_note });
        }
        if (config.has_hybrid_layers) {
            log.info("  ssm-checkpoint-stride={d} tokens, max={d}/entry\n", .{ launch().ssm_checkpoint_stride, launch().ssm_checkpoint_max });
        }
    } else if (launch().prefix_cache_capacity > 0) {
        if (config.has_hybrid_layers and launch().ssm_checkpoint_stride == 0) {
            log.info("Hot prefix cache: requested capacity={d} but model is hybrid and --ssm-checkpoint-stride is 0 — disabled\n", .{launch().prefix_cache_capacity});
        } else if (config.full_attention_interval > 0) {
            log.info("Hot prefix cache: requested capacity={d} but model uses full-attention-interval — disabled\n", .{launch().prefix_cache_capacity});
        } else {
            log.info("Hot prefix cache: requested capacity={d} but disabled by scheduler\n", .{launch().prefix_cache_capacity});
        }
    }

    // `--max-concurrent N` is honored when the model can ride the batched
    // decode kernel (pure-attention, no SSM/MoE/encoder). Hybrid / MoE /
    // encoder architectures clamp to 1 — they need single-slot serial
    // because the batched kernel doesn't model their state. DSV4 is
    // honored (per-slot LatentKVCache landed in Plan 04 Section 4
    // Phase A+B; Section B fix excludes DSV4 from
    // `Scheduler.batchable` so two DSV4 slots fall through to per-slot
    // `runSingleDecodeTick` — sequential through the inference thread,
    // safe).
    // `--max-concurrent` sizes the submit queue; requests decode together at
    // any value. Whether they share ONE forward is the model's answer, printed
    // at every value so a default-1 boot does not read as "one at a time".
    if (mlx_engine.api().batches_decode(config)) {
        log.info("Concurrency: --max-concurrent={d}, batched decode on\n", .{max_concurrent});
        if (launch().prefix_cache_capacity > 0 and launch().prefix_cache_capacity < max_concurrent) launch().prefix_cache_capacity = max_concurrent;
    } else {
        log.info("Concurrency: --max-concurrent={d}, batched decode off (arch: {s}); concurrent requests interleave serially\n", .{ max_concurrent, config.model_type });
    }
    // Chat interleaves with media jobs except on CUDA, where a decode inside a job faulted the context.
    const gen_yield = !mlx.cudaAvailable();
    mlx_engine.api().set_gen_yield(gen_yield);
    if (!gen_yield)
        log.info("chat does not interleave with media jobs on CUDA: a chat waits for the running image/audio job\n", .{});
    // Install signal handlers for graceful shutdown (POSIX; on Windows Ctrl+C ends the process).
    if (comptime builtin.os.tag != .windows) {
        const sigact = std.posix.Sigaction{
            .handler = .{ .handler = signalHandler },
            .mask = std.posix.sigemptyset(),
            .flags = 0,
        };
        std.posix.sigaction(std.posix.SIG.INT, &sigact, null);
        std.posix.sigaction(std.posix.SIG.TERM, &sigact, null);
    }

    // ── LAN sharing/discovery (src/host/lan/lan.zig): started HERE — the one chokepoint
    //    every serve path (model/headless/gen/ds4/llama) flows through — so the
    //    advertised port is always the bound port. Bonjour being unavailable
    //    degrades to a warning; it must never kill the server.
    if (g_lan_share_spec != null or g_lan_discover) {
        var keys_buf: [std.fs.max_path_bytes]u8 = undefined;
        const home = std.mem.span(env.get(.HOME) orelse "/tmp");
        g_lan = lan_mod.Lan.start(allocator, .{
            .port = port,
            .share_spec = g_lan_share_spec,
            .name = g_lan_name,
            .discover = g_lan_discover,
            .keys_path = std.fmt.bufPrint(&keys_buf, "{s}/.mlx-serve/lan-keys.json", .{home}) catch null,
        }) catch |err| blk: {
            log.warn("[lan] failed to start ({s}); LAN sharing disabled\n", .{@errorName(err)});
            break :blk null;
        };
    }
    // Runs after the conn-thread drain below (LIFO), so no tunnel is mid-pump
    // and no request is mid-lookup when the peer table is freed.
    defer if (g_lan) |l| {
        g_lan = null;
        l.shutdown();
    };
    // The App Store build cannot exec curl (the providers transport), so it
    // never starts them — same limit as `mlx-serve pull`.
    if (!build_options.mas) {
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const home = std.mem.span(env.get(.HOME) orelse "/tmp");
        const path = std.fmt.bufPrint(&path_buf, "{s}/.mlx-serve/providers.json", .{home}) catch "";
        g_providers = providers_mod.Providers.start(allocator, io, path, port) catch |err| blk: {
            log.warn("[providers] failed to start ({s}); providers disabled\n", .{@errorName(err)});
            break :blk null;
        };
    }
    defer if (g_providers) |p| {
        g_providers = null;
        p.shutdown();
    };

    // Freeze the auto-context NOW, at startup, while the model is freshly
    // loaded and nothing else has taken RAM. Clients read this number once
    // (pi/opencode bake it into a config file) and budget against it for the
    // whole session, so it must not drift with system load. `--ctx-size` wins.
    _ = admission.pinAutoContext(@constCast(config), true);

    if (serverState().server_config.request_timeout_sec > 0) {
        log.info("Request timeout: {d}s\n", .{serverState().server_config.request_timeout_sec});
    }
    const model_ctx = config.contextCap();
    if (model_ctx > 0) {
        log.info("Model context length: {d} tokens\n", .{model_ctx});
    }
    if (serverState().server_config.default_reasoning_budget >= 0) {
        log.info("Reasoning budget: {d} tokens\n", .{serverState().server_config.default_reasoning_budget});
    } else {
        log.info("Reasoning budget: unlimited\n", .{});
    }
    if (launchMaxTokensDefault() > 0) {
        log.info("default max_tokens for omitted requests: {d}\n", .{launchMaxTokensDefault()});
    }
    if (serverState().server_config.default_enable_pld) {
        log.info("PLD speculative decoding: ENABLED (draft_len={d}, key_len={d}; default for new requests)\n", .{ serverState().server_config.default_pld_draft_len, serverState().server_config.default_pld_key_len });
    }
    if (scheduler.dflash != null) {
        log.info("DFlash speculative decoding: ENABLED (block_size={d}; default for new requests)\n", .{scheduler.drafter_block_size});
    } else if (scheduler.drafter != null and scheduler.dflash == null) {
        log.info("Drafter speculative decoding: ENABLED (block_size={d}; default for new requests)\n", .{scheduler.drafter_block_size});
    }
    const engine_settings = mlx_engine.settings();
    if (engine_settings.mtp_head_kv_quant) {
        log.info("MTP head KV: following --kv-quant (--mtp-head-kv-quant)\n", .{});
    }
    if (engine_settings.max_mtp_ctx != 0) {
        log.info("MTP context ceiling: {d} tokens (--max-mtp-ctx; requests past it decode serially)\n", .{engine_settings.max_mtp_ctx});
    }
    log.info("\nServer listening on http://{s}:{d}\n", .{ host, port });
    if (g_api_key != null) {
        log.info("API key auth: ENABLED for non-loopback requests (localhost is trusted; /health stays open)\n", .{});
    }
    log.info("  GET  /\n", .{});
    log.info("  GET  /health\n", .{});
    log.info("  GET  /props\n", .{});
    log.info("  GET  /v1/models\n", .{});
    log.info("  POST /v1/chat/completions\n", .{});
    log.info("  POST /v1/completions\n", .{});
    log.info("  POST /v1/embeddings\n", .{});
    log.info("  POST /v1/decisions, /v1/systemone (Laya, Kev, D1, Clef)\n", .{});
    log.info("  POST /v1/messages (Anthropic)\n", .{});
    log.info("  POST /v1/responses (OpenAI Responses)\n", .{});
    log.info("  POST /v1/responses/compact\n", .{});
    log.info("  GET  /v1/responses/{{id}}\n", .{});
    log.info("  DEL  /v1/responses/{{id}}\n", .{});
    log.info("  POST /tokenize\n", .{});
    log.info("  POST /detokenize\n\n", .{});

    // Print system metrics once at startup
    const rss = metrics.getAppRssMb();
    if (rss >= 1024) {
        log.info("RSS: {d}.{d}G  Mem: {d}%  CPU: {d}%  GPU: {d}%\n", .{
            rss / 1024,             (rss % 1024) * 10 / 1024,
            metrics.getSysMemPct(), metrics.getCpuPct(),
            metrics.getGpuPct(),
        });
    } else {
        log.info("RSS: {d}M  Mem: {d}%  CPU: {d}%  GPU: {d}%\n", .{
            rss, metrics.getSysMemPct(), metrics.getCpuPct(), metrics.getGpuPct(),
        });
    }

    while (!shutdown_requested.load(.acquire)) {
        // Poll with 1-second timeout so we can check shutdown flag
        const ready = socketReadable(server.socket.handle, 1000) catch |err| {
            if (shutdown_requested.load(.acquire)) break;
            log.err("poll error: {}\n", .{err});
            continue;
        };
        if (!ready) continue; // timeout, re-check shutdown flag
        if (shutdown_requested.load(.acquire)) break;

        const accepted_stream = server.accept(io) catch |err| {
            if (shutdown_requested.load(.acquire)) break;
            log.err("accept error: {}\n", .{err});
            continue;
        };

        // When the scheduler is engaged AND the model is pure-attention,
        // Spawn a per-connection thread so HTTP I/O for one request can
        // overlap with another request's generation. mlx ops (forward
        // passes, eval, etc.) live exclusively on the scheduler's inference
        // thread, so the connection thread NEVER touches `xfm.s` directly —
        // it parses HTTP, encodes the prompt, calls `scheduler.submit`,
        // reads tokens via `slot.waitNext`, and writes the response.
        const args = allocator.create(ConnThreadArgs) catch |err| {
            log.err("conn thread args alloc failed: {}\n", .{err});
            accepted_stream.close(io);
            continue;
        };
        args.* = .{
            .allocator = allocator,
            .accepted_stream = accepted_stream,
            .io = io,
        };
        // Count the thread before spawning so the shutdown drain can't miss a
        // thread that's about to start; the thread decrements on return.
        _ = active_conn_threads.fetchAdd(1, .acq_rel);
        const conn_thread = std.Thread.spawn(.{}, handleConnectionThread, .{args}) catch |err| {
            log.err("spawn conn thread failed: {}\n", .{err});
            _ = active_conn_threads.fetchSub(1, .acq_rel);
            accepted_stream.close(io);
            allocator.destroy(args);
            continue;
        };
        conn_thread.detach();
    }

    // Shutdown ordering (prevents a SIGSEGV in Scheduler.complete): the accept
    // loop has exited. Cancel every in-flight slot so its connection thread
    // unblocks from waitNext and runs its `defer complete(...)`, then WAIT for
    // all connection threads to finish before returning — `serve`'s caller runs
    // `scheduler.deinit` on return, which frees the slot queues that an
    // in-flight `complete()` is still touching. The inference thread is still
    // alive here (deinit joins it only after we return), so cancelled slots
    // settle promptly. Bounded so a wedged thread can't hang shutdown forever.
    // The SSD flush's deadline starts here, once. The drain keeps its own bound: cutting it short
    // saves nothing (the join below waits out the inference thread's current chunk anyway) and
    // lets a late `complete()` race the queues `deinit` frees.
    if (launch().global_scheduler) |sch| {
        armShutdownDeadline(&sch.shutdown_deadline_ms, nowMsMonotonic(io), shutdown_persist_secs);
        sch.cancelAllInFlight();
    }
    {
        var waited_ms: u64 = 0;
        while (active_conn_threads.load(.acquire) > 0 and waited_ms < 30_000) {
            crt.sleepMs(20);
            waited_ms += 20;
        }
        const remaining = active_conn_threads.load(.acquire);
        if (remaining > 0)
            log.warn("shutdown: {d} connection thread(s) still active after {d}ms — proceeding\n", .{ remaining, waited_ms });
    }

    deinitGlobalResponseStore();

    log.info("\nShutting down gracefully...\n", .{});
}

/// Arm the shutdown flush's deadline `secs` after the first call; a later call (a second
/// SIGTERM) never moves it. 0 seconds leaves it unarmed, which turns the flush off.
pub fn armShutdownDeadline(slot: *std.atomic.Value(i64), now_ms: i64, secs: u32) void {
    if (secs == 0) return;
    _ = slot.cmpxchgStrong(0, now_ms + @as(i64, secs) * 1000, .acq_rel, .acquire);
}

/// Poll interval for the idle-eviction sweep, given the configured window.
///
/// A quarter of the window, so a model is evicted within ~1.25x the configured
/// idle time rather than up to 2x it. Floored at a second so `--idle-evict-secs 1`
/// does not spin, capped at 30s so a long window still costs ~nothing.
pub fn idleEvictTickMs(window_ms: i64) i64 {
    return @max(1000, @min(@divTrunc(window_ms, 4), 30_000));
}

const GaugeSamplerCtx = struct {
    metrics: *instr.Metrics,
    scheduler: *Scheduler,
    /// Dedicated stop flag — set before joining the thread (LIFO defers ensure
    /// this is written before join() blocks). A separate flag (not
    /// scheduler.shutdown) avoids a LIFO deadlock: scheduler.deinit() (which
    /// sets shutdown) is the FIRST-registered defer and runs LAST, after join.
    stop: *std.atomic.Value(bool),
};

/// Sample instantaneous system/queue state into the gauges. Non-blocking
/// syscalls plus a brief scheduler-counter lock. Called by the sampler thread.
/// The live-token gauge reads the scheduler's `inflight_generated_tokens`
/// aggregate (published once per decode tick) — it NEVER touches per-slot
/// fields off-thread, so the inference path stays lock-free per token.
fn sampleGauges(ctx: GaugeSamplerCtx) void {
    // System gauges (non-blocking syscalls).
    ctx.metrics.gpu_utilization_pct.set(@as(u64, metrics.getGpuPct()));
    ctx.metrics.memory_mb.set(@as(u64, metrics.getAppMemFootprintMb()));
    // The two halves of the MLX allocator, so `memory_mb`'s gap has a name.
    var mlx_active: usize = 0;
    var mlx_cache: usize = 0;
    _ = mlx.mlx_get_active_memory(&mlx_active);
    _ = mlx.mlx_get_cache_memory(&mlx_cache);
    ctx.metrics.mlx_active_bytes.set(@as(u64, mlx_active));
    ctx.metrics.mlx_cache_bytes.set(@as(u64, mlx_cache));
    // ANE prefill offload totals — published by the engines themselves
    // (ane.publishLive / deinit), so this is a lock-free read.
    const eg = mlx_engine.gauges();
    ctx.metrics.ane_int8_bytes.set(eg.ane_int8_bytes);
    ctx.metrics.ane_layers.set(eg.ane_layers);
    // qwen4 n-gram table warm (F8) — same lock-free published-atomic pattern.
    ctx.metrics.ngram_warm_bytes.set(eg.ngram_warm_bytes);

    // Request queue depth — brief lock to read two scheduler counters only.
    ctx.scheduler.queue_mu.lockUncancelable(ctx.scheduler.io);
    const running = @as(u64, ctx.scheduler.in_flight);
    const waiting = @as(u64, ctx.scheduler.pending.items.len);
    ctx.scheduler.queue_mu.unlock(ctx.scheduler.io);
    ctx.metrics.requests_running.set(running);
    ctx.metrics.requests_waiting.set(waiting);

    // Real-time throughput source: completed generation tokens plus the tokens
    // generated so far by still-decoding slots (the scheduler publishes that
    // aggregate atomically once per decode tick — race-free, no per-token
    // write). Non-zero even mid-request, unlike the completion-only counter.
    const inflight = ctx.scheduler.inflight_generated_tokens.load(.monotonic) + ctx.scheduler.engine_gauges.generated.load(.monotonic);
    ctx.metrics.generation_tokens_live.set(instr.liveGenerationTokens(ctx.metrics, inflight));

    // The other phase. Prompt-token counters and the prefill-time histogram
    // only advance at request COMPLETION, so a long prefill was invisible: GPU
    // pinned, decode reading 0, prefill reading "—". This gauge moves per
    // prefill chunk and returns to 0 the moment the prefill ends.
    // Requests on the engines beside MLX count apart: the GPU's prefill stores its absolute progress.
    ctx.metrics.prefill_tokens_live.set(ctx.scheduler.inflight_prefill_tokens.load(.monotonic) + ctx.scheduler.engine_gauges.prefill_done.load(.monotonic));
    ctx.metrics.prefill_tokens_expected.set(ctx.scheduler.inflight_prefill_expected.load(.monotonic) + ctx.scheduler.engine_gauges.prefill_todo.load(.monotonic));
    ctx.metrics.requests_prefilling.set(ctx.scheduler.requests_prefilling.load(.monotonic) + ctx.scheduler.engine_gauges.prefilling.load(.monotonic));
}

const IdleEvictCtx = struct {
    scheduler: *Scheduler,
    /// `--idle-evict-secs`, in ms.
    window_ms: i64,
    /// Dedicated stop flag, for the same reason `GaugeSamplerCtx` has one:
    /// gating on `scheduler.shutdown` would never be observed, because
    /// `scheduler.deinit()` (which sets it) is the FIRST-registered defer and
    /// runs LAST, after this thread's join. Worse here than for the sampler —
    /// a sweep blocked inside `unloadModel` is waiting on the inference thread,
    /// and `deinit` only rescues orphaned unload requests AFTER joining it, so
    /// a join placed inside `deinit` would hang forever.
    stop: *std.atomic.Value(bool),
};

/// `--idle-evict-secs`: unload models that have served nothing for the window.
///
/// Goes through `Scheduler.unloadModelIfIdle`, the same path
/// `POST /v1/unload-model` uses, so the eviction itself is unchanged and
/// nothing here touches mlx directly.
fn idleEvictLoop(ctx: IdleEvictCtx) void {
    const sch = ctx.scheduler;
    const tick_ms = idleEvictTickMs(ctx.window_ms);
    // Wake every 500 ms to check the stop flag; sweep once per tick interval.
    // `crt.sleepMs` for the same reason as the gauge sampler: a blocking Io timer
    // here could outlive the stop.
    const ticks_per_sweep: u64 = @max(1, @as(u64, @intCast(@divTrunc(tick_ms, 500))));
    var tick: u64 = 0;
    while (!ctx.stop.load(.monotonic)) {
        crt.sleepMs(500);
        tick += 1;
        if (tick < ticks_per_sweep) continue;
        tick = 0;
        if (ctx.stop.load(.monotonic)) return;

        // One victim per sweep. `unloadModelIfIdle` blocks on the inference
        // thread, so holding the registry mutex across it would stall every
        // request; the next sweep takes anything else that has gone idle.
        const now_ms = io_util.nowMsMonotonic(sch.io);
        sch.registry.mutex.lockUncancelable(sch.io);
        const victim = sch.registry.pickIdleEvictable(now_ms, ctx.window_ms);
        const idle_ms = if (victim) |v| now_ms - v.last_used_ms.load(.acquire) else 0;
        const bytes = if (victim) |v| v.bytes_resident else 0;
        sch.registry.mutex.unlock(sch.io);

        if (victim) |entry| {
            log.info("[registry] idle-evicting model id={s} ({d:.2} GB resident, idle {d}s)\n", .{
                entry.id,
                @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0),
                @divTrunc(idle_ms, 1000),
            });
            // Age and refcount are re-checked under the mutex inside, so a
            // request arriving since the pick cancels the eviction instead of
            // pinning the model in `.evicting` behind a live stream.
            sch.unloadModelIfIdle(entry.id, ctx.window_ms) catch |err| {
                log.warn("[registry] idle eviction failed for {s}: {t}\n", .{ entry.id, err });
            };
        }
    }
}

fn gaugeSamplerLoop(ctx: GaugeSamplerCtx) void {
    // Wake every 500 ms to check the stop flag; sample system state once per
    // 2-second interval. The 2 s cadence is what makes generation_tokens_live a
    // real-time tok/s source — the panel windows over a few of these samples.
    // `crt.sleepMs`: this thread has no Io handle.
    const SAMPLE_INTERVAL_TICKS: u64 = 4; // 4 × 500 ms = 2 s
    // Sample once immediately so the panel / Grafana show real values from the
    // first scrape instead of zeros for the first interval.
    sampleGauges(ctx);
    var tick: u64 = 0;
    while (!ctx.stop.load(.monotonic)) {
        crt.sleepMs(500);
        tick += 1;
        if (tick < SAMPLE_INTERVAL_TICKS) continue;
        tick = 0;
        sampleGauges(ctx);
    }
}

const testing = std.testing;

test "getTimeoutNs computes correctly" {
    const original = serverState().server_config.request_timeout_sec;
    defer serverState().server_config.request_timeout_sec = original;

    serverState().server_config.request_timeout_sec = 300;
    try testing.expectEqual(@as(u64, 300_000_000_000), getTimeoutNs());

    serverState().server_config.request_timeout_sec = 0;
    try testing.expectEqual(@as(u64, 0), getTimeoutNs());
}

test "llama cache default keeps shared prefixes warm" {
    // With one sequence, interleaved requests evict each other's prompt KV:
    // even two SEQUENTIAL requests sharing an 8 KB prefix reported
    // cached_tokens=0. 4 keep interleaved agent roots warm and decoding together.
    try testing.expect(llama_settings.seqs >= 4);
}

test {
    _ = @import("../api/request_options.zig");
    _ = @import("../api/admission.zig");
    _ = @import("../api/anthropic.zig");
    _ = @import("../api/anthropic_stream.zig");
    _ = @import("../api/audio.zig");
    _ = @import("../api/chat_completions.zig");
    _ = @import("../api/chat_stream.zig");
    _ = @import("../api/completions.zig");
    _ = @import("../api/embeddings.zig");
    _ = @import("../api/format.zig");
    _ = @import("../api/logprobs.zig");
    _ = @import("../api/media_gate.zig");
    _ = @import("../api/models.zig");
    _ = @import("../api/ollama_routes.zig");
    _ = @import("../api/props.zig");
    _ = @import("../api/request_model.zig");
    _ = @import("../api/request_text.zig");
    _ = @import("../api/responses/envelope.zig");
    _ = @import("../api/responses/events.zig");
    _ = @import("../api/responses/handler.zig");
    _ = @import("../api/responses/store.zig");
    _ = @import("../api/responses/ws.zig");
    _ = @import("../api/sampling.zig");
    _ = @import("../api/submit.zig");
    _ = @import("../api/tokenize.zig");
    _ = @import("../chat/reasoning/request.zig");
    _ = @import("../chat/stops.zig");
    _ = @import("../chat/tools/pipeline.zig");
    _ = @import("auth.zig");
    _ = @import("conn.zig");
    _ = @import("errors.zig");
    _ = @import("listen.zig");
    _ = @import("request.zig");
    _ = @import("respond.zig");
    _ = @import("router.zig");
    _ = @import("sse.zig");
    _ = @import("../lan/http.zig");
    _ = @import("../launch/http.zig");
    _ = @import("../media/decode.zig");
    _ = @import("../media/encode.zig");
    _ = @import("../media/placeholders.zig");
    _ = @import("../media/prepare.zig");
    _ = @import("../media/request_media.zig");
    _ = @import("../media_jobs/http.zig");
    _ = @import("../registry/memory_ceiling.zig");
    _ = @import("../../mlx/state.zig");
    _ = @import("../../plugins/deepseek_v4/bill.zig");
    _ = @import("../../plugins/deepseek_v41/bill.zig");
    _ = @import("../../plugins/lfm2/media.zig");
    _ = @import("../../plugins/nemotron_h/media.zig");
    _ = @import("../../plugins/qwen3_5/minicpm_media.zig");
    _ = @import("../../plugins/qwen4_exp/bill.zig");
    _ = @import("../media/qwen_mrope.zig");
}

test "armShutdownDeadline: the first SIGTERM sets the deadline and a second never moves it" {
    var slot = std.atomic.Value(i64).init(0);
    armShutdownDeadline(&slot, 5_000, 45);
    try testing.expectEqual(@as(i64, 50_000), slot.load(.acquire));
    armShutdownDeadline(&slot, 9_000, 45);
    try testing.expectEqual(@as(i64, 50_000), slot.load(.acquire));
    var off = std.atomic.Value(i64).init(0);
    armShutdownDeadline(&off, 5_000, 0);
    try testing.expectEqual(@as(i64, 0), off.load(.acquire));
}

test "idleEvictTickMs: sweeps well inside the window without spinning" {
    // Bar: a quarter of the window, clamped to [1s, 30s] — eviction lands near
    // the configured time without the sweep becoming a busy loop.
    try testing.expectEqual(@as(i64, 5_000), idleEvictTickMs(20_000));
    try testing.expectEqual(@as(i64, 15_000), idleEvictTickMs(60_000));
    try testing.expectEqual(@as(i64, 30_000), idleEvictTickMs(900_000));
    try testing.expectEqual(@as(i64, 1000), idleEvictTickMs(1000));
    try testing.expectEqual(@as(i64, 1000), idleEvictTickMs(2000));
    try testing.expectEqual(@as(i64, 30_000), idleEvictTickMs(28_800_000));
    try testing.expect(idleEvictTickMs(0) >= 1000);
}
