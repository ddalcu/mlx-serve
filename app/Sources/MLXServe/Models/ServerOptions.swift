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

    // Speculative decoding (server-launch flags)
    var enablePLD: Bool = true          // --pld is default-on now (CLI flips with --no-pld)
    var pldDraftLen: Int = 5
    var pldKeyLen: Int = 3
    var drafterPath: String = ""        // empty = no drafter
    var draftBlockSize: Int = 4

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
    /// shared system prompts. 0 disables.
    var prefixCacheEntries: Int = 1
    /// Hot prefix cache memory budget. `2GB`, `512MB`, etc. `0` or `off`
    /// disables the byte cap (count cap still applies). Empty = server default.
    var prefixCacheMem: String = "2GB"

    // MARK: GGUF-only (llama.cpp engine)
    /// KV-cache quantization for the embedded llama.cpp engine. MLX's
    /// `--kv-quant` does NOT apply to the llama path (different kernels);
    /// this is the GGUF-equivalent knob. `off` keeps F16 (libllama
    /// default); `q8` ≈ 2× compression near-lossless; `q4` ≈ 4× with
    /// some quality cost. Auto-enables flash-attn server-side when
    /// non-default.
    var llamaKvQuant: LlamaKVQuant = .off
    /// Multi-session LRU size for the embedded llama.cpp engine. 1
    /// keeps the legacy single-session behavior (every flip between
    /// long-doc prompts evicts the other). N > 1 keeps the N most-
    /// recently-used prompts hot in independent KV contexts.
    var llamaCacheEntries: Int = 1

    // MARK: All engines
    /// Per-LoadedModel LRU cache of chat-template render + tokenize
    /// results. Applies to MLX, llama.cpp, and ds4 — same handler
    /// boundary on every path. 0 disables.
    var tokenizeCacheEntries: Int = 4

    // MARK: Per-request defaults (apply immediately, no restart)
    var defaultMaxTokens: Int = 4096
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

    enum LogLevel: String, Codable, CaseIterable, Identifiable {
        case error, warn, info, debug
        var id: String { rawValue }
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
        enablePLD == other.enablePLD &&
        pldDraftLen == other.pldDraftLen &&
        pldKeyLen == other.pldKeyLen &&
        drafterPath == other.drafterPath &&
        draftBlockSize == other.draftBlockSize &&
        maxConcurrent == other.maxConcurrent &&
        kvQuant == other.kvQuant &&
        prefixCacheEntries == other.prefixCacheEntries &&
        prefixCacheMem == other.prefixCacheMem &&
        llamaKvQuant == other.llamaKvQuant &&
        llamaCacheEntries == other.llamaCacheEntries &&
        tokenizeCacheEntries == other.tokenizeCacheEntries
    }

    // MARK: CLI args builder

    /// Translate to the `mlx-serve` CLI flags. The leading `--model <path>` is
    /// passed by ServerManager (since the model path comes from AppState).
    ///
    /// Plan 05 Phase G — `modelDirOverride` lets the caller inject
    /// `--model-dir <path>` so the registry sees siblings (enables
    /// hot-switch). ServerManager passes the parent directory of the
    /// selected model.
    func toCLIArgs(modelDirOverride: String? = nil) -> [String] {
        var args: [String] = [
            "--serve",
            "--port", "\(port)",
            "--host", host,
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
        // Spec-decode: explicit flags either way so the server's CLI defaults
        // can't drift out from under the UI.
        args += [enablePLD ? "--pld" : "--no-pld"]
        args += ["--pld-draft-len", "\(pldDraftLen)"]
        args += ["--pld-key-len", "\(pldKeyLen)"]
        if !drafterPath.isEmpty {
            args += ["--drafter", drafterPath,
                     "--draft-block-size", "\(draftBlockSize)"]
        }
        // Performance: only emit non-default flags so the CLI tail stays
        // readable in log lines and `ps`. Server defaults are 1 / off / 1 / 2GB.
        if maxConcurrent > 1 {
            args += ["--max-concurrent", "\(maxConcurrent)"]
        }
        if kvQuant != .off {
            args += ["--kv-quant", kvQuant.cliValue]
        }
        if prefixCacheEntries != 1 {
            args += ["--prefix-cache-entries", "\(prefixCacheEntries)"]
        }
        let trimmedPrefixMem = prefixCacheMem.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedPrefixMem.isEmpty && trimmedPrefixMem != "2GB" {
            args += ["--prefix-cache-mem", trimmedPrefixMem]
        }
        // GGUF-only performance knobs. Emitted unconditionally when not
        // the default — the server silently ignores them on the MLX path
        // (it never opens an llama session). Keeping them in argv for
        // every launch means the user gets consistent behavior whether
        // they're switching MLX↔GGUF via the menu picker or restarting.
        if llamaKvQuant != .off {
            args += ["--llama-kv-quant", llamaKvQuant.cliValue]
        }
        if llamaCacheEntries != 1 {
            args += ["--llama-cache-entries", "\(llamaCacheEntries)"]
        }
        // All-engines knob — applies to MLX, llama, and ds4 alike.
        if tokenizeCacheEntries != 4 {
            args += ["--tokenize-cache-entries", "\(tokenizeCacheEntries)"]
        }
        return args
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
            explainer: "Bind address. 0.0.0.0 lets other devices on your network reach the server; 127.0.0.1 is local-only.",
            needsRestart: true),
        "port": .init(
            title: "Port",
            explainer: "HTTP port the server listens on (default 11234). Change if it conflicts with another local service.",
            needsRestart: true),
        "ctxSize": .init(
            title: "Context size",
            explainer: "Maximum prompt + completion tokens. 0 means use the model's declared maximum. Higher values use more memory.",
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
        "enablePLD": .init(
            title: "Enable PLD (recommended)",
            explainer: "Prompt Lookup Decoding. Big wins on echo-heavy workloads (code editing, RAG, agent loops). The adaptive prompt-time gate auto-disables it on novel content.",
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
            explainer: "Shrinks the KV cache: 4-bit ≈ 4× smaller, 8-bit ≈ 2×. Lets a 16K context fit on hardware that couldn't hold it dense, or doubles the parallel-request budget. TurboQuant variants add a per-layer Hadamard rotation for heavy-tailed activations.",
            needsRestart: true),
        "prefixCacheEntries": .init(
            title: "Prefix cache entries",
            explainer: "Hot prefix cache size: how many separate KV snapshots to keep across requests. Lets multi-turn chats skip re-prefilling shared system prompts. 0 disables.",
            needsRestart: true),
        "prefixCacheMem": .init(
            title: "Prefix cache memory cap",
            explainer: "Maximum RAM for the prefix cache. Accepts '2GB', '512MB', '0' (disable byte cap). Default 2GB.",
            needsRestart: true),
        "llamaKvQuant": .init(
            title: "KV cache quantization",
            explainer: "llama.cpp's KV-quant scheme (ggml Q8_0 / Q4_0). Distinct from the MLX KV-quant — uses different kernels. Auto-enables flash-attn on the server side when non-default.",
            needsRestart: true),
        "llamaCacheEntries": .init(
            title: "Session cache entries",
            explainer: "How many independent llama.cpp KV contexts to keep resident. 1 = legacy single-session (every flip between long prompts evicts the other). >1 keeps the N most-recently-used prompts hot — alternating multi-doc workloads stop cold-prefilling on every flip.",
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
            explainer: "Default max_tokens to request per chat turn. Per-message overrides win when set.",
            needsRestart: false),
        "defaultTemperature": .init(
            title: "Temperature",
            explainer: "0 = deterministic greedy. 0.6–1.0 typical chat. Above 1.0 gets erratic.",
            needsRestart: false),
        "defaultTopP": .init(
            title: "Top-p",
            explainer: "Nucleus sampling threshold. 0.95 keeps all but the long tail. 1.0 disables top-p filtering.",
            needsRestart: false),
        "defaultTopK": .init(
            title: "Top-k",
            explainer: "Cap on candidate tokens per step. 0 = disabled (use top-p only).",
            needsRestart: false),
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
