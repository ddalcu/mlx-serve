import Foundation

/// All user-tunable mlx-serve options, persisted to UserDefaults as JSON.
///
/// Split into two groups:
/// 1. Server-launch flags: passed on the `mlx-serve --serve` CLI; require a
///    server restart to take effect.
/// 2. Per-request defaults: injected into the JSON body of every chat request
///    by APIClient; apply on the next request, no restart needed.
///
/// The Settings UI introspects these via the `serverFlagFields` /
/// `requestDefaultFields` metadata to render labels, captions and the
/// "needs restart" badge automatically — every option carries its own
/// human-readable explainer.
struct ServerOptions: Codable, Equatable {
    // MARK: Server-launch flags (require restart)
    var host: String = "0.0.0.0"
    var port: UInt16 = 11234
    var ctxSize: Int = 0                // 0 = Auto (memory-bounded safe ceiling, capped at model max)
    var noVision: Bool = false
    var logLevel: LogLevel = .info
    var requestTimeout: Int = 300       // seconds; 0 = unlimited

    // Observability (server-launch flag). When true, launches with `--metrics`,
    // exposing the Prometheus `/metrics` scrape endpoint AND a live throughput/
    // latency/GPU/memory panel on the server's index page (GET /).
    var enableMetrics: Bool = false
    /// Optional API key (`--api-key`). When non-empty, remote (non-loopback)
    /// requests to the OpenAI/Anthropic/Ollama APIs AND the index page + metrics
    /// panel require it (Authorization: Bearer / x-api-key / HTTP Basic / query
    /// param). Loopback is trusted, so this app (127.0.0.1) never needs to send
    /// it — the key protects the server when bound to the network (0.0.0.0).
    var apiKey: String = ""
    /// Tool-call argument auto-correct (`--no-tool-autocorrect` disables). When
    /// true (default, mirrors the server's `g_tool_autocorrect`), parsed
    /// tool-call arguments are coerced to the types the tool schema declares
    /// (e.g. Python `False` → JSON `false`, an `edits` array shipped as a string
    /// → a real array), fixing the class of "expected boolean, provided string"
    /// rejections weak models cause. Off = args pass through verbatim (still valid
    /// JSON; only the schema-type coercion is skipped). `toCLIArgs` emits the flag
    /// ONLY when false, so a default launch stays flag-free.
    var toolAutocorrect: Bool = true

    // Speculative decoding (server-launch flags)
    var enablePLD: Bool = true          // --pld is default-on now (CLI flips with --no-pld)
    var pldDraftLen: Int = 5
    var pldKeyLen: Int = 3
    var drafterPath: String = ""        // empty = no drafter
    var draftBlockSize: Int = 4

    /// Native multi-token prediction (Qwen 3.5/3.6 checkpoints that ship a
    /// trained `mtp/` sidecar head). Default ON, mirroring the server — the head
    /// auto-loads when present and models without one are unaffected, so
    /// `toCLIArgs` emits `--no-mtp` ONLY when the user turns it off.
    var enableMTP: Bool = true
    /// How many tokens the MTP head drafts per round. `0` = auto, which is the
    /// server's own default (`--mtp-depth 0` → the adaptive EV controller tunes
    /// depth live from the measured acceptance rate). A fixed value is a
    /// benchmarking lever; emitted only when non-zero.
    var mtpDepth: Int = 0

    // Performance (server-launch flags)
    /// Continuous batching: max in-flight chat requests batched through one
    /// forward pass. 1 = serial (default). Hybrid SSM / MoE models clamp to 1
    /// regardless. Pure-attention dense models pick up ~1.6× at 4-way.
    var maxConcurrent: Int = 1
    /// KV-cache quantization scheme. `off` = dense bf16. `int4` / `int8` apply
    /// affine quant; `turbo2` / `turbo4` add a per-layer Hadamard rotation for
    /// heavy-tailed activations.
    var kvQuant: KVQuant = .off
    /// Hot prefix cache entry count. >0 enables cross-request KV reuse for
    /// shared system prompts. 0 disables. The launcher RAM-clamps the emitted
    /// value via `ramCappedPrefixCacheEntries` — each entry on a hybrid SSM
    /// model (Qwen 3.5/3.6, etc.) retains large per-position KV + conv/SSM
    /// snapshots whose true footprint dwarfs the model, so an uncapped count
    /// fills a 16 GB Mac and starves the context window. Default 8 suits
    /// 32 GB+; a 16 GB Mac caps to 1.
    var prefixCacheEntries: Int = 8
    /// Hot prefix cache memory budget. `2GB`, `512MB`, etc. `0` or `off`
    /// disables the byte cap (count cap still applies). Empty = server default.
    var prefixCacheMem: String = "2GB"
    /// SSD tier for the prefix cache. OFF by default because it can persist
    /// gigabytes of KV under ~/.mlx-serve/kv-cache. When on, seen prefixes
    /// survive restarts + RAM evictions (turns a cold 30-50 s long-context
    /// TTFT into a bounded SSD read). Always emitted (`--prefix-cache-disk
    /// <size>` on, `off` off) so the process matches the UI regardless of the
    /// server default. See "SSD prefix cache" in CLAUDE.md.
    var enablePrefixCacheDisk: Bool = false
    /// Disk budget when `enablePrefixCacheDisk` is on. `10GB`, `2GB`, etc.
    var prefixCacheDisk: String = "10GB"
    /// When true, launch with `--skip-mem-preflight` so the MLX loader skips the
    /// free-RAM pre-flight that would otherwise refuse a model whose weights +
    /// warmup headroom look too big for current free memory. The check is
    /// deliberately conservative (macOS reclaims file cache as MLX allocates),
    /// so a load it refuses often still fits — but a genuine over-commit can
    /// hard-crash the server, so this is opt-in.
    var skipMemPreflight: Bool = false

    // MARK: GGUF-only (llama.cpp engine)
    /// KV-cache quantization for the embedded llama.cpp engine. MLX's
    /// `--kv-quant` does NOT apply to the llama path (different kernels);
    /// this is the GGUF-equivalent knob. `off` keeps F16 (libllama
    /// default); `q8` ≈ 2× compression near-lossless; `q4` ≈ 4× with
    /// some quality cost. Auto-enables flash-attn server-side when
    /// non-default.
    var llamaKvQuant: LlamaKVQuant = .off
    /// Multi-session LRU size for the embedded llama.cpp engine. 1 keeps the
    /// legacy single-session behavior (every flip between long-doc prompts
    /// evicts the other). N > 1 keeps the N most-recently-used prompts hot in
    /// independent KV contexts. Default 4 MUST match `server.zig`
    /// `llama_cache_entries` — `toCLIArgs` omits the flag at this value, so a
    /// mismatch would silently run the server's default instead (see the
    /// "ServerOptions defaults must mirror the Zig server" gotcha in CLAUDE.md).
    var llamaCacheEntries: Int = 4

    // MARK: ds4-only (DeepSeek-V4-Flash engine)
    /// When true, launch with `--ssd-streaming` so the embedded ds4 engine
    /// streams expert weights from SSD instead of holding the whole model
    /// resident (skips full residency + Metal warmup). Lets DeepSeek-V4-Flash
    /// run on a Mac whose RAM can't hold the full model. ds4-only — the MLX and
    /// llama.cpp engines ignore the flag. Default off mirrors main.zig
    /// `ds4_ssd_streaming = false`.
    var ssdStreaming: Bool = false

    // MARK: All engines
    /// Per-LoadedModel LRU cache of chat-template render + tokenize
    /// results. Applies to MLX, llama.cpp, and ds4 — same handler
    /// boundary on every path. 0 disables.
    var tokenizeCacheEntries: Int = 4

    // MARK: Per-request defaults (apply immediately, no restart)
    // 16384, not 4096: a thinking trace + agentic/code answer routinely blew
    // past 4 K and tripped finish_reason "length" mid-reply. The server clamps
    // this to the live context window, so a generous default can't overflow —
    // it just stops cutting normal answers short. Users can still tune it in
    // Settings (existing stored values are preserved on upgrade).
    var defaultMaxTokens: Int = 16384
    var defaultTemperature: Double = 0.8
    var defaultTopP: Double = 0.95
    var defaultTopK: Int = 0            // 0 = disabled
    var defaultRepeatPenalty: Double = 1.0
    var defaultPresencePenalty: Double = 0.0
    var defaultReasoningBudget: Int = -1    // -1 = unlimited
    var defaultEnableThinking: Bool = false

    // Per-request overrides for spec-decode (default = follow server default)
    var perRequestEnablePLD: TriState = .auto
    var perRequestEnableDrafter: TriState = .auto

    // MARK: Messaging bridge (app-level — NOT a server-launch flag, NOT per-request)
    /// Telegram bot bridge: message your local model from your phone. The bridge
    /// runs *inside* the app (long-polls Telegram over outbound HTTPS), so it is
    /// deliberately excluded from `serverLaunchEquals` — toggling it must never
    /// prompt a server restart.
    var telegram: TelegramConfig = TelegramConfig()

    // MARK: Agent sandbox (app-level — NOT a server-launch flag, NOT per-request)
    /// Run the agent's shell/tool commands inside an isolated Linux sandbox
    /// instead of directly on host macOS. The sandbox is a lightweight Linux
    /// guest booted on Apple's Virtualization.framework (see `VzGuest`); only
    /// the working directory is shared in, so a destructive command can't reach
    /// the rest of the Mac. It's consumed by the app-side tool executor, so —
    /// like `telegram` — it's deliberately excluded from `serverLaunchEquals`:
    /// toggling it must never prompt a server restart.
    var sandbox: SandboxConfig = SandboxConfig()

    // MARK: Voice clone (app-level — NOT a server-launch flag, NOT per-request)
    /// Absolute path to the normalized voice-clone reference clip (24 kHz mono
    /// WAV) that hands-free voice mode speaks with, via Qwen3-TTS zero-shot
    /// cloning (`/v1/audio/speech` + `ref_audio`). Empty = no clone — voice
    /// mode uses the macOS system voice. Recorded/picked in Settings ▸ Voice.
    /// Consumed app-side by `ClonedVoiceSynthesizer`, so — like `telegram` and
    /// `sandbox` — it's deliberately excluded from `serverLaunchEquals` and
    /// `toCLIArgs`: changing it must never prompt a server restart.
    var voiceClonePath: String = ""
    /// Whether voice mode actually speaks with the clone. Selecting a system
    /// voice in the voice picker flips this off WITHOUT deleting the clip, so
    /// switching back to "My voice" stays one click. Defaults true — a clip
    /// set before this toggle existed keeps cloning (legacy behavior).
    var voiceCloneEnabled: Bool = true
    /// Human-readable name for the clip shown in the voice picker (the picked
    /// file's name, or "Recorded clip") — the stored clip itself is always
    /// normalized to `voice-clone.wav`, which is useless as a display name.
    var voiceCloneLabel: String = ""
    /// Wake phrase for hands-free voice mode ("hey loki" by default). Stored
    /// as the user typed it in Settings ▸ Voice; consumers normalize through
    /// `WakeWord.normalizePhrase` (blank → default). App-side like the other
    /// voice fields — never a server-launch flag.
    var wakePhrase: String = WakeWord.defaultPhrase

    enum LogLevel: String, Codable, CaseIterable, Identifiable {
        case error, warn, info, debug
        var id: String { rawValue }
        /// Human-readable label for the Settings picker. The raw value is what
        /// the server's `--log-level` flag accepts — keep these in sync.
        var label: String {
            switch self {
            case .error: return "Error (quietest)"
            case .warn:  return "Warn"
            case .info:  return "Info (default)"
            case .debug: return "Debug (verbose)"
            }
        }
    }

    enum KVQuant: String, Codable, CaseIterable, Identifiable {
        case off
        case int4 = "4"
        case int8 = "8"
        case turbo2
        case turbo4
        var id: String { rawValue }
        /// CLI flag value (`--kv-quant <x>`); same string the server parses.
        var cliValue: String { rawValue }
        var label: String {
            switch self {
            case .off:    return "Off (dense bf16)"
            case .int4:   return "4-bit (≈4× smaller KV)"
            case .int8:   return "8-bit (≈2× smaller KV)"
            case .turbo2: return "TurboQuant 2-bit"
            case .turbo4: return "TurboQuant 4-bit"
            }
        }
    }

    /// KV-cache quantization for the embedded llama.cpp engine.
    /// Distinct from `KVQuant` because llama.cpp's quant scheme is
    /// ggml's `Q8_0` / `Q4_0`, not the MLX affine kernels. The server
    /// flag is `--llama-kv-quant` rather than `--kv-quant`.
    enum LlamaKVQuant: String, Codable, CaseIterable, Identifiable {
        case off
        case q8
        case q4
        var id: String { rawValue }
        var cliValue: String { rawValue }
        var label: String {
            switch self {
            case .off: return "Off (F16, libllama default)"
            case .q8:  return "Q8_0 (≈2× smaller, near-lossless)"
            case .q4:  return "Q4_0 (≈4× smaller, some quality cost)"
            }
        }
    }

    enum TriState: String, Codable, CaseIterable, Identifiable {
        case auto, on, off
        var id: String { rawValue }
        var label: String {
            switch self {
            case .auto: return "Auto (server default)"
            case .on:   return "Force on"
            case .off:  return "Force off"
            }
        }
        /// `nil` means leave the request body alone — the server's startup
        /// flag governs. `true`/`false` overrides per-request.
        var asOptionalBool: Bool? {
            switch self {
            case .auto: return nil
            case .on:   return true
            case .off:  return false
            }
        }
    }

    /// Telegram bot bridge settings. The user creates the bot via @BotFather
    /// (`/newbot`) and pastes the token here. The bridge long-polls Telegram
    /// (outbound HTTPS only — no public URL or port-forward, so it works behind
    /// home NAT). `allowedChatIds` locks the bot to specific chats; empty means
    /// "adopt the first chat that messages" (trust-on-first-use), so setup is
    /// just: paste token, message the bot once.
    struct TelegramConfig: Codable, Equatable {
        var enabled: Bool = false
        var botToken: String = ""
        /// false = plain chat (default, safe). true = the full agent loop with
        /// tools (shell / file / web) executing on this Mac, driven from the
        /// phone. Opt-in; the Settings UI warns about the implications.
        var agentMode: Bool = false
        /// Expose the user's enabled MCP servers to the bot (and to the tasks it
        /// creates). Opt-in; works alongside or independently of `agentMode`.
        var useMCP: Bool = false
        var enableThinking: Bool = false
        /// Telegram chat IDs allowed to use the bot. Empty = adopt the first
        /// chat that messages (TOFU); the bridge appends the adopted id here.
        var allowedChatIds: [Int64] = []

        /// True when the config is actually runnable (enabled + a non-blank token).
        var isRunnable: Bool {
            enabled && !botToken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }

        /// The bot token with surrounding whitespace stripped — what the bridge
        /// actually sends to the Telegram API (users paste with trailing newlines).
        var trimmedToken: String {
            botToken.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        /// Access decision for an incoming chat id. Pure — drives the bridge's
        /// gate and is unit-tested without any network.
        func access(forChatId chatId: Int64) -> TelegramAccess {
            if allowedChatIds.contains(chatId) { return .allowed }
            if allowedChatIds.isEmpty { return .adopt }
            return .rejected
        }
    }

    /// Outcome of the bot's per-message access check. See `TelegramConfig.access`.
    enum TelegramAccess: Equatable {
        /// Chat is on the allow-list — answer it.
        case allowed
        /// No allow-list yet — adopt this chat (add to `allowedChatIds`) then answer.
        case adopt
        /// An allow-list exists and this chat isn't on it — refuse.
        case rejected
    }

    /// Agent execution sandbox. When `enabled`, the agent's shell commands run
    /// inside a throwaway Linux guest (booted on Apple's Virtualization
    /// framework) with only the working directory shared in — so a destructive
    /// command can't touch the rest of the Mac. It costs extra RAM while active
    /// because a live VM is held for the session. Off by default.
    struct SandboxConfig: Codable, Equatable {
        /// false = tools run directly on host macOS (default, no extra RAM).
        /// true = shell commands run inside the isolated Linux sandbox.
        var enabled: Bool = false
        /// OCI/Docker base image pulled once for the sandbox rootfs. MUST have an
        /// arm64 build (the HVF guest is arm64) — an amd64-only image fails to
        /// boot. Defaults to `ddalcu/agent-shell-mlxserve`: a purpose-built arm64 agentic
        /// shell (Debian glibc + Node.js + Python3/pip + git/curl + apt, ~129 MB).
        /// Any arm64 image works; e.g. `python:3.12-slim` or `debian:stable-slim`.
        var baseImage: String = "ddalcu/agent-shell-mlxserve"
        /// Outbound network for the guest (NAT) + live guest→host port mapping:
        /// a server the agent starts on guest port N becomes reachable at
        /// `localhost:N` on this Mac. Off → the guest has NO network device at
        /// all (fully isolated). Applies on the next guest boot.
        var network: Bool = true

        /// The base image ref with surrounding whitespace stripped — what the
        /// image puller actually receives (users paste with trailing newlines).
        var trimmedBaseImage: String {
            baseImage.trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    // MARK: Restart-detection helpers

    /// Compares only fields that affect the launched mlx-serve process, ignoring
    /// per-request defaults.
    func serverLaunchEquals(_ other: ServerOptions) -> Bool {
        host == other.host &&
        port == other.port &&
        ctxSize == other.ctxSize &&
        noVision == other.noVision &&
        logLevel == other.logLevel &&
        requestTimeout == other.requestTimeout &&
        enableMetrics == other.enableMetrics &&
        apiKey == other.apiKey &&
        toolAutocorrect == other.toolAutocorrect &&
        enablePLD == other.enablePLD &&
        pldDraftLen == other.pldDraftLen &&
        pldKeyLen == other.pldKeyLen &&
        drafterPath == other.drafterPath &&
        draftBlockSize == other.draftBlockSize &&
        enableMTP == other.enableMTP &&
        mtpDepth == other.mtpDepth &&
        maxConcurrent == other.maxConcurrent &&
        kvQuant == other.kvQuant &&
        prefixCacheEntries == other.prefixCacheEntries &&
        prefixCacheMem == other.prefixCacheMem &&
        enablePrefixCacheDisk == other.enablePrefixCacheDisk &&
        prefixCacheDisk == other.prefixCacheDisk &&
        skipMemPreflight == other.skipMemPreflight &&
        llamaKvQuant == other.llamaKvQuant &&
        llamaCacheEntries == other.llamaCacheEntries &&
        ssdStreaming == other.ssdStreaming &&
        tokenizeCacheEntries == other.tokenizeCacheEntries &&
        // Sampling defaults are ALSO launch flags (server-side defaults for
        // clients that omit sampling, e.g. Claude Code) — changing them must
        // trip the restart detector. The app's own chats still pick them up
        // immediately via request bodies.
        defaultTemperature == other.defaultTemperature &&
        defaultTopP == other.defaultTopP &&
        defaultTopK == other.defaultTopK
    }

    // MARK: CLI args builder

    /// Translate to the `mlx-serve` CLI flags. The leading `--model <path>` is
    /// passed by ServerManager (since the model path comes from AppState).
    ///
    /// Plan 05 Phase G — `modelDirOverride` lets the caller inject
    /// `--model-dir <path>` so the registry sees siblings (enables
    /// hot-switch). ServerManager passes the parent directory of the
    /// selected model.
    /// RAM-derived ceiling for the hot prefix cache entry count. On a hybrid
    /// SSM model each entry pins large per-position KV + conv/SSM state, so a
    /// generous count (the server's own default is 32) fills a small Mac and,
    /// as resident memory climbs, the auto-context ceiling shrinks until
    /// prompts no longer fit ("Prompt exceeds maximum context length"). Capping
    /// the entry count is the reliable lever — the byte cap under-counts the
    /// true retained allocation. An explicit 0 (disable) is preserved.
    ///   ≤18 GB (16 GB Macs): 1   ≤36 GB (24/32 GB): 8   else: uncapped.
    static func ramCappedPrefixCacheEntries(_ requested: Int, physicalMemoryBytes: UInt64) -> Int {
        if requested <= 0 { return requested }
        let gib = physicalMemoryBytes / 1_073_741_824
        let ceiling: Int
        if gib <= 18 { ceiling = 1 }
        else if gib <= 36 { ceiling = 8 }
        else { return requested }
        return min(requested, ceiling)
    }

    func toCLIArgs(modelDirOverride: String? = nil,
                   physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory) -> [String] {
        // The host field is free text in Settings — a cleared field must not
        // launch `--host ""` (the server would fail to bind).
        let trimmedHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
        var args: [String] = [
            "--serve",
            "--port", "\(port)",
            "--host", trimmedHost.isEmpty ? "0.0.0.0" : trimmedHost,
            "--log-level", logLevel.rawValue,
        ]
        if let dir = modelDirOverride, !dir.isEmpty {
            args += ["--model-dir", dir]
        }
        if ctxSize > 0 {
            args += ["--ctx-size", "\(ctxSize)"]
        }
        if noVision {
            args += ["--no-vision"]
        }
        if requestTimeout != 300 {
            args += ["--timeout", "\(requestTimeout)"]
        }
        if enableMetrics {
            args += ["--metrics"]
        }
        let trimmedApiKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedApiKey.isEmpty {
            args += ["--api-key", trimmedApiKey]
        }
        // Emit ONLY when disabled — the server default is on, so a default launch
        // stays flag-free (guarded by testDefaultLaunchOmitsAllMatchDefaultFlags).
        if !toolAutocorrect {
            args += ["--no-tool-autocorrect"]
        }
        // Spec-decode: explicit flags either way so the server's CLI defaults
        // can't drift out from under the UI.
        args += [enablePLD ? "--pld" : "--no-pld"]
        args += ["--pld-draft-len", "\(pldDraftLen)"]
        args += ["--pld-key-len", "\(pldKeyLen)"]
        if !drafterPath.isEmpty {
            args += ["--drafter", drafterPath,
                     "--draft-block-size", "\(draftBlockSize)"]
        }
        // MTP: the server auto-loads a checkpoint's `mtp/` head and defaults
        // depth to auto, so a default launch emits NOTHING here (guarded by
        // testDefaultLaunchOmitsAllMatchDefaultFlags).
        if !enableMTP {
            args += ["--no-mtp"]
        }
        if mtpDepth > 0 {
            args += ["--mtp-depth", "\(mtpDepth)"]
        }
        // Performance: only emit non-default flags so the CLI tail stays
        // readable in log lines and `ps`. Server defaults are 1 / off / 2GB.
        if maxConcurrent > 1 {
            args += ["--max-concurrent", "\(maxConcurrent)"]
        }
        if kvQuant != .off {
            args += ["--kv-quant", kvQuant.cliValue]
        }
        // ALWAYS emit — the server's prefix-cache default is 32, NOT 1, so
        // omitting this silently launched a 32-entry cache that filled small
        // Macs. Emit the RAM-clamped value so the entry count stays bounded.
        let cappedEntries = Self.ramCappedPrefixCacheEntries(prefixCacheEntries, physicalMemoryBytes: physicalMemoryBytes)
        args += ["--prefix-cache-entries", "\(cappedEntries)"]
        let trimmedPrefixMem = prefixCacheMem.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedPrefixMem.isEmpty && trimmedPrefixMem != "2GB" {
            args += ["--prefix-cache-mem", trimmedPrefixMem]
        }
        // ALWAYS emit — the SSD tier can persist gigabytes of KV, so the app is
        // authoritative: `off` when the toggle is off (regardless of any server
        // default), the chosen size when on. Mirrors the prefix-cache-entries
        // always-emit discipline.
        let trimmedPrefixDisk = prefixCacheDisk.trimmingCharacters(in: .whitespacesAndNewlines)
        if enablePrefixCacheDisk && !trimmedPrefixDisk.isEmpty {
            args += ["--prefix-cache-disk", trimmedPrefixDisk]
        } else {
            args += ["--prefix-cache-disk", "off"]
        }
        // GGUF-only performance knobs. Emitted unconditionally when not
        // the default — the server silently ignores them on the MLX path
        // (it never opens an llama session). Keeping them in argv for
        // every launch means the user gets consistent behavior whether
        // they're switching MLX↔GGUF via the menu picker or restarting.
        if llamaKvQuant != .off {
            args += ["--llama-kv-quant", llamaKvQuant.cliValue]
        }
        if llamaCacheEntries != 4 {  // 4 = server default (server.zig llama_cache_entries)
            args += ["--llama-cache-entries", "\(llamaCacheEntries)"]
        }
        // All-engines knob — applies to MLX, llama, and ds4 alike.
        if tokenizeCacheEntries != 4 {
            args += ["--tokenize-cache-entries", "\(tokenizeCacheEntries)"]
        }
        // Sampling defaults double as server-launch flags so third-party
        // clients that omit sampling params (Claude Code sends none at all)
        // inherit the Settings values. Per-request body fields always win.
        // Top-k 0 = "no opinion": OMIT the flag so the model's own
        // generation_config.json recommendation (Qwen 3.6: 20, Gemma 4: 64)
        // stays in effect rather than being force-disabled.
        // %g: slider arithmetic leaves float dirt (0.8 - 0.1 stepped to
        // 0.7000000000000001) that "\(Double)" would print verbatim into argv.
        args += ["--temp", String(format: "%g", defaultTemperature)]
        args += ["--top-p", String(format: "%g", defaultTopP)]
        if defaultTopK > 0 {
            args += ["--top-k", "\(defaultTopK)"]
        }
        // Opt-in escape hatch — omitted by default so the load pre-flight runs.
        if skipMemPreflight {
            args += ["--skip-mem-preflight"]
        }
        // ds4-only opt-in: stream DeepSeek-V4-Flash experts from SSD. The MLX
        // and llama.cpp engines ignore the flag, so it's safe to leave in argv
        // across engine switches; omitted by default to keep full residency.
        if ssdStreaming {
            args += ["--ssd-streaming"]
        }
        return args
    }

    // MARK: Settings-field validation helpers

    /// Parse the Settings port text field. Accepts exactly what a TCP listen
    /// can bind: an all-digit string in 1...65535 (after trimming). Returns
    /// nil for anything else — including 0, which the kernel would map to a
    /// random ephemeral port the rest of the app isn't watching.
    static func parsePort(_ raw: String) -> UInt16? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              trimmed.allSatisfy(\.isNumber),
              let value = UInt16(trimmed),
              value > 0 else { return nil }
        return value
    }

    // MARK: Persistence (UserDefaults JSON)

    private static let storageKey = "serverOptions"

    static func load() -> ServerOptions {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let opts = try? JSONDecoder().decode(ServerOptions.self, from: data) else {
            return ServerOptions()
        }
        return opts
    }

    func save() {
        guard let data = try? JSONEncoder().encode(self) else { return }
        UserDefaults.standard.set(data, forKey: Self.storageKey)
    }
}

// MARK: - Migration-safe decoding

extension ServerOptions {
    /// Forward/backward-compatible decode: every key is optional, falling back
    /// to the `ServerOptions()` default when absent.
    ///
    /// The COMPILER-SYNTHESIZED `init(from:)` throws `keyNotFound` the moment a
    /// stored blob is missing ANY key — which is exactly what happens to an
    /// existing user's saved options every time a new field ships. Because
    /// `load()` wraps the decode in `try?` and falls back to `ServerOptions()`,
    /// that throw silently RESETS the user's entire tuning to defaults on
    /// upgrade. Decoding key-by-key with `decodeIfPresent` (defaults supplied by
    /// delegating to the no-arg init first) keeps old blobs valid and makes
    /// every future field addition migration-safe automatically. Declared in an
    /// extension so the memberwise / no-arg initializers stay synthesized.
    /// `encode(to:)` and `CodingKeys` remain compiler-synthesized, so round-trip
    /// + Equatable are unchanged (see `testRoundTripCodable`).
    init(from decoder: Decoder) throws {
        self.init()   // every property seeded with its default
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let v = try c.decodeIfPresent(String.self, forKey: .host) { host = v }
        if let v = try c.decodeIfPresent(UInt16.self, forKey: .port) { port = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .ctxSize) { ctxSize = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .noVision) { noVision = v }
        if let v = try c.decodeIfPresent(LogLevel.self, forKey: .logLevel) { logLevel = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .requestTimeout) { requestTimeout = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .enableMetrics) { enableMetrics = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .apiKey) { apiKey = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .toolAutocorrect) { toolAutocorrect = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .enablePLD) { enablePLD = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .pldDraftLen) { pldDraftLen = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .pldKeyLen) { pldKeyLen = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .drafterPath) { drafterPath = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .draftBlockSize) { draftBlockSize = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .enableMTP) { enableMTP = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .mtpDepth) { mtpDepth = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .maxConcurrent) { maxConcurrent = v }
        if let v = try c.decodeIfPresent(KVQuant.self, forKey: .kvQuant) { kvQuant = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .prefixCacheEntries) { prefixCacheEntries = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .prefixCacheMem) { prefixCacheMem = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .enablePrefixCacheDisk) { enablePrefixCacheDisk = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .prefixCacheDisk) { prefixCacheDisk = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .skipMemPreflight) { skipMemPreflight = v }
        if let v = try c.decodeIfPresent(LlamaKVQuant.self, forKey: .llamaKvQuant) { llamaKvQuant = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .llamaCacheEntries) { llamaCacheEntries = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .ssdStreaming) { ssdStreaming = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .tokenizeCacheEntries) { tokenizeCacheEntries = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .defaultMaxTokens) { defaultMaxTokens = v }
        if let v = try c.decodeIfPresent(Double.self, forKey: .defaultTemperature) { defaultTemperature = v }
        if let v = try c.decodeIfPresent(Double.self, forKey: .defaultTopP) { defaultTopP = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .defaultTopK) { defaultTopK = v }
        if let v = try c.decodeIfPresent(Double.self, forKey: .defaultRepeatPenalty) { defaultRepeatPenalty = v }
        if let v = try c.decodeIfPresent(Double.self, forKey: .defaultPresencePenalty) { defaultPresencePenalty = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .defaultReasoningBudget) { defaultReasoningBudget = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .defaultEnableThinking) { defaultEnableThinking = v }
        if let v = try c.decodeIfPresent(TriState.self, forKey: .perRequestEnablePLD) { perRequestEnablePLD = v }
        if let v = try c.decodeIfPresent(TriState.self, forKey: .perRequestEnableDrafter) { perRequestEnableDrafter = v }
        if let v = try c.decodeIfPresent(TelegramConfig.self, forKey: .telegram) { telegram = v }
        if let v = try c.decodeIfPresent(SandboxConfig.self, forKey: .sandbox) { sandbox = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .voiceClonePath) { voiceClonePath = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .voiceCloneEnabled) { voiceCloneEnabled = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .voiceCloneLabel) { voiceCloneLabel = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .wakePhrase) { wakePhrase = v }
    }
}

extension ServerOptions.TelegramConfig {
    /// Tolerant decode (same rationale as `ServerOptions.init(from:)`): a stored
    /// telegram blob written before a field existed must decode with that field
    /// defaulted, not throw — a throw here propagates up through ServerOptions
    /// and `load()` silently resets the user's entire config (token included).
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let v = try c.decodeIfPresent(Bool.self, forKey: .enabled) { enabled = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .botToken) { botToken = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .agentMode) { agentMode = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .useMCP) { useMCP = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .enableThinking) { enableThinking = v }
        if let v = try c.decodeIfPresent([Int64].self, forKey: .allowedChatIds) { allowedChatIds = v }
    }
}

extension ServerOptions.SandboxConfig {
    /// Tolerant decode (same rationale as `ServerOptions.init(from:)`): a stored
    /// blob written before a field existed must decode with that field defaulted,
    /// not throw — a throw here propagates up and silently resets the whole config.
    init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let v = try c.decodeIfPresent(Bool.self, forKey: .enabled) { enabled = v }
        if let v = try c.decodeIfPresent(String.self, forKey: .baseImage) { baseImage = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .network) { network = v }
    }
}

// MARK: - Reset to defaults

extension ServerOptions {
    /// The value the Settings screen's "Reset to Defaults" button installs.
    ///
    /// Everything returns to `ServerOptions()` except the Telegram bot token,
    /// which is a credential the user obtained from @BotFather and cannot
    /// re-derive from anything on this machine — wiping it turns a stray reset
    /// into a trip back to Telegram. Every other field (including the rest of
    /// the Telegram block: the enable toggle, agent mode, and the adopted chat
    /// ids, which re-adopt on the next message) is re-derivable and is
    /// deliberately restored. Pure so `SettingsResetTests` can pin the scope of
    /// the carve-out against future fields.
    static func resetToDefaults(preserving current: ServerOptions) -> ServerOptions {
        var fresh = ServerOptions()
        fresh.telegram.botToken = current.telegram.botToken
        return fresh
    }
}

// MARK: - UI introspection metadata

/// Describes a single tunable for the Settings UI: a label, an explainer,
/// and (where relevant) a flag indicating whether changes need a server restart.
struct ServerOptionField {
    let title: String
    let explainer: String
    let needsRestart: Bool
}

extension ServerOptions {
    /// Human-readable metadata for the server-launch fields, in the order the
    /// Settings UI should render them.
    static let serverFlagFields: [String: ServerOptionField] = [
        "host": .init(
            title: "Host",
            explainer: "Bind address. 0.0.0.0 lets other devices on your network reach the server; 127.0.0.1 is local-only. The app itself connects via 127.0.0.1, so pick an address that includes localhost.",
            needsRestart: true),
        "port": .init(
            title: "Port",
            explainer: "HTTP port the server listens on (default 11234). Change if it conflicts with another local service.",
            needsRestart: true),
        "ctxSize": .init(
            title: "Context size",
            explainer: ContextSizeDisplay.helpText,
            needsRestart: true),
        "noVision": .init(
            title: "Disable vision",
            explainer: "Skip loading the SigLIP image encoder. Saves ~3 GB of memory on text-only workloads.",
            needsRestart: true),
        "logLevel": .init(
            title: "Log level",
            explainer: "Server log verbosity. Use 'debug' to capture Jinja errors, KV cache hits and per-request token counts.",
            needsRestart: true),
        "requestTimeout": .init(
            title: "Request timeout (s)",
            explainer: "Max seconds a single HTTP request is allowed to take. 0 = unlimited. Long agent loops may need 600+.",
            needsRestart: true),
        "enableMetrics": .init(
            title: "Metrics panel",
            explainer: "Expose Prometheus metrics at /metrics and a live throughput/latency/GPU/memory panel on the server index page. Zero cost when off.",
            needsRestart: true),
        "apiKey": .init(
            title: "API key",
            explainer: "Require this key for requests from OTHER machines (the OpenAI/Anthropic/Ollama APIs + index page + metrics). Sent as Authorization: Bearer, x-api-key, HTTP Basic, or ?api_key=. Localhost is trusted, so this app never needs it — it protects the server when exposed on the network. Blank = no auth.",
            needsRestart: true),
        "toolAutocorrect": .init(
            title: "Tool-call auto-correct",
            explainer: "Coerce tool-call arguments to the types the tool schema declares (e.g. a model sending Python's \"False\" or a list-as-string) so strict clients like Claude Code stop rejecting them with \"expected boolean, provided string\". Recommended on. Turn off to pass the model's arguments through exactly as written.",
            needsRestart: true),
        "enableMTP": .init(
            title: "Multi-Token Prediction (recommended)",
            explainer: "Qwen 3.5 / 3.6 models that ship a trained MTP head guess several of the next tokens at once and check them all in a single pass — typically a big speed-up on replies, with output identical to normal decoding. It switches on by itself for models that have the head, and does nothing for models that don't, so there's rarely a reason to turn it off.",
            needsRestart: true),
        "mtpDepth": .init(
            title: "Tokens guessed ahead",
            explainer: "How many tokens the MTP head guesses per step. Automatic (recommended) tunes this live — it guesses deeper while the model keeps accepting the guesses and backs off when it doesn't. Pick a fixed number only if you're measuring performance.",
            needsRestart: true),
        "enablePLD": .init(
            title: "Enable PLD (recommended)",
            explainer: "Prompt Lookup Decoding. Big wins on echo-heavy workloads (code editing, RAG, agent loops). The adaptive prompt-time gate auto-disables it on novel content. On models with a native MTP head, MTP takes priority and PLD stays dormant — except MoE models (e.g. 35B-A3B), where PLD is the default speedup.",
            needsRestart: true),
        "pldDraftLen": .init(
            title: "PLD draft length",
            explainer: "Maximum draft tokens proposed per PLD step (default 5). Higher = bigger speedup when matches hit, more wasted work when they miss.",
            needsRestart: true),
        "pldKeyLen": .init(
            title: "PLD key length",
            explainer: "N-gram match key length for PLD lookup (default 3). Shorter keys = more matches, lower precision.",
            needsRestart: true),
        "drafterPath": .init(
            title: "Drafter checkpoint",
            explainer: "Path to a Gemma 4 assistant drafter directory (gemma-4-*-it-assistant-bf16). Must pair with a Gemma 4 target. Empty = no drafter.",
            needsRestart: true),
        "draftBlockSize": .init(
            title: "Drafter block size",
            explainer: "Tokens per drafter round (default 4 = 3 drafter steps + 1 verify token).",
            needsRestart: true),
        "maxConcurrent": .init(
            title: "Concurrent requests",
            explainer: "Continuous batching: how many chat requests share one forward pass. 1 = serial. 2 is a good default for dense models (~1.5× throughput, ~33% per-request latency cost). MoE and hybrid SSM models stay serial regardless.",
            needsRestart: true),
        "kvQuant": .init(
            title: "KV cache quantization",
            explainer: "A memory-for-speed trade, not a free upgrade: shrinks KV-cache RAM (8-bit ≈ 2× smaller, 4-bit ≈ 4×) but makes decode ~10% slower at typical contexts — and slower still on long ones, since every generated token pays a dequantize step. Turn on when memory is the constraint (long contexts or big models on a 16 GB Mac); leave OFF for maximum tokens/sec if you have plenty of RAM. TurboQuant variants add a per-layer Hadamard rotation for heavy-tailed activations.",
            needsRestart: true),
        "prefixCacheEntries": .init(
            title: "Prefix cache entries",
            explainer: "Hot prefix cache size: how many separate KV snapshots to keep across requests. Lets multi-turn chats skip re-prefilling shared system prompts. 0 disables. Auto-capped on RAM-limited Macs (a 16 GB Mac caps to 1) — on hybrid SSM models each entry pins large KV + state snapshots that can otherwise fill memory.",
            needsRestart: true),
        "prefixCacheMem": .init(
            title: "Prefix cache memory cap",
            explainer: "Maximum RAM for the prefix cache. Accepts '2GB', '512MB', '0' (disable byte cap). Default 2GB.",
            needsRestart: true),
        "enablePrefixCacheDisk": .init(
            title: "SSD prefix cache",
            explainer: "Persist seen KV prefixes to disk (~/.mlx-serve/kv-cache) so they survive restarts + RAM evictions — turns a cold 30-50s long-context first-token wait into a fast SSD read. OFF by default because it can use many gigabytes of disk.",
            needsRestart: true),
        "prefixCacheDisk": .init(
            title: "SSD cache size",
            explainer: "Disk budget for the SSD prefix cache when enabled. Accepts '10GB', '2GB', etc. LRU-evicted to this cap. Only used when 'SSD prefix cache' is on.",
            needsRestart: true),
        "skipMemPreflight": .init(
            title: "Skip memory pre-flight check",
            explainer: "Bypass the safety check that refuses to load an MLX model when free RAM looks too low for its weights plus warmup headroom. The check is conservative — macOS reclaims file cache as the model loads — so turn this on if a load you know fits is being refused. A genuine over-commit can hard-crash the server. Passes --skip-mem-preflight.",
            needsRestart: true),
        "llamaKvQuant": .init(
            title: "KV cache quantization",
            explainer: "llama.cpp's KV-quant scheme (ggml Q8_0 / Q4_0). Same trade as the MLX version: less KV RAM, slower decode — leave off for maximum tokens/sec if you have plenty of RAM. Distinct kernels from the MLX KV-quant; auto-enables flash-attn on the server side when non-default.",
            needsRestart: true),
        "llamaCacheEntries": .init(
            title: "Session cache entries",
            explainer: "How many independent llama.cpp KV contexts to keep resident. 1 = legacy single-session (every flip between long prompts evicts the other). >1 keeps the N most-recently-used prompts hot — alternating multi-doc workloads stop cold-prefilling on every flip.",
            needsRestart: true),
        "ssdStreaming": .init(
            title: "SSD weight streaming",
            explainer: "Stream DeepSeek-V4-Flash expert weights from SSD instead of holding the whole model in RAM (skips full residency + warmup). Turn on when the model is larger than available memory — it loads instead of OOMing, trading some decode speed for the disk reads. ds4 / DeepSeek-V4-Flash only; ignored by the MLX and llama.cpp engines. Passes --ssd-streaming.",
            needsRestart: true),
        "tokenizeCacheEntries": .init(
            title: "Tokenize cache entries",
            explainer: "Per-LoadedModel LRU of chat-template render+tokenize results. Skips re-rendering identical message lists on warm reuse — drops warm tokenize_ms from ~240 ms to ~0 on long-prompt repeats. Applies to all engines. 0 disables.",
            needsRestart: true),
    ]

    /// Human-readable metadata for the per-request defaults.
    static let requestDefaultFields: [String: ServerOptionField] = [
        "defaultMaxTokens": .init(
            title: "Max tokens",
            explainer: "Max tokens to generate per chat turn. \"Auto\" pegs it to the remaining context window — the safe choice on a small-RAM / small-context machine. Per-message overrides win when set.",
            needsRestart: false),
        "defaultTemperature": .init(
            title: "Temperature",
            explainer: "0 = deterministic greedy. 0.6–1.0 typical chat. Above 1.0 gets erratic. Applies to the app's chats immediately; also becomes the server default for external clients that omit temperature (Claude Code) after a restart.",
            needsRestart: true),
        "defaultTopP": .init(
            title: "Top-p",
            explainer: "Nucleus sampling threshold. 0.95 keeps all but the long tail. 1.0 disables top-p filtering. Also the server default for external clients that omit top_p (restart needed for that part).",
            needsRestart: true),
        "defaultTopK": .init(
            title: "Top-k",
            explainer: "Cap on candidate tokens per step. 0 = follow the model's own recommendation from generation_config.json (Qwen 3.6: 20, Gemma 4: 64); explicit values override it server-wide after a restart.",
            needsRestart: true),
        "defaultRepeatPenalty": .init(
            title: "Repetition penalty",
            explainer: "Penalty multiplier for tokens already in the context. 1.0 = none. 1.1 is a typical anti-repeat setting.",
            needsRestart: false),
        "defaultPresencePenalty": .init(
            title: "Presence penalty",
            explainer: "Additive penalty per token already present in the context. 0 = none.",
            needsRestart: false),
        "defaultReasoningBudget": .init(
            title: "Reasoning budget",
            explainer: "Max thinking tokens per request. -1 = unlimited. Only applies when thinking is enabled.",
            needsRestart: false),
        "defaultEnableThinking": .init(
            title: "Enable thinking",
            explainer: "Default the chat client to send `enable_thinking: true`. Only models with reasoning support honor this.",
            needsRestart: false),
        "perRequestEnablePLD": .init(
            title: "Per-request PLD",
            explainer: "Auto = follow the server's --pld setting (and the adaptive gate). On/Off forces it.",
            needsRestart: false),
        "perRequestEnableDrafter": .init(
            title: "Per-request drafter",
            explainer: "Auto = follow the server. On/Off forces it. Only meaningful when --drafter is loaded.",
            needsRestart: false),
    ]
}
