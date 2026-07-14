import Foundation

struct ChatSession: Identifiable, Codable {
    let id: UUID
    var title: String
    var messages: [ChatMessage]
    var createdAt: Date
    var updatedAt: Date
    var mode: ChatMode
    var workingDirectory: String?
    /// The attached document folder (mini RAG), persisted so the index can be
    /// rebuilt after a relaunch. The matching security-scoped bookmark
    /// (`SecurityScopedBookmark.attachedFolderName(id)`) is what keeps the path
    /// reachable under the App Sandbox once the NSOpenPanel grant is gone.
    var attachedFolderPath: String?
    /// Non-nil marks this session as the transient vehicle for an unattended task
    /// run (see TaskScheduler). Such sessions are filtered out of the chat sidebar
    /// and never persisted to chat-history.json — their transcript lives under
    /// ~/.mlx-serve/tasks/<taskId>/<runId>/transcript.json instead.
    var taskRunId: UUID?
    /// True marks this as a transient vehicle for an external messaging bridge
    /// (e.g. the Telegram bot). Like task-run sessions these are kept out of the
    /// chat sidebar and never persisted to chat-history.json — the conversation
    /// lives on the messaging platform, not in the app's chat list.
    var isExternalBridge: Bool
    /// Per-session toolbar toggles. Persisted here (not as view `@State` or the
    /// app-global `mcpMode`) so each chat tab remembers its own Think/MCP choice
    /// across tab switches and relaunches — `mode` already does the same for the
    /// Agent toggle. See PerSessionUIStateTests.
    var enableThinking: Bool
    var useMCP: Bool

    init(title: String = "New Chat") {
        self.id = UUID()
        self.title = title
        self.messages = []
        self.createdAt = Date()
        self.updatedAt = Date()
        self.mode = .chat
        self.workingDirectory = ChatSession.defaultWorkingDirectory
        self.attachedFolderPath = nil
        self.taskRunId = nil
        self.isExternalBridge = false
        self.enableThinking = false
        self.useMCP = false
    }

    enum CodingKeys: String, CodingKey {
        case id, title, messages, createdAt, updatedAt, mode, workingDirectory, attachedFolderPath, taskRunId, isExternalBridge, enableThinking, useMCP
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        title = try c.decode(String.self, forKey: .title)
        messages = try c.decode([ChatMessage].self, forKey: .messages)
        createdAt = try c.decode(Date.self, forKey: .createdAt)
        updatedAt = try c.decode(Date.self, forKey: .updatedAt)
        mode = try c.decodeIfPresent(ChatMode.self, forKey: .mode) ?? .chat
        taskRunId = try c.decodeIfPresent(UUID.self, forKey: .taskRunId)
        isExternalBridge = try c.decodeIfPresent(Bool.self, forKey: .isExternalBridge) ?? false
        // Backfill: sessions saved before the per-session Think/MCP toggles
        // existed come back with the keys absent → default both off.
        enableThinking = try c.decodeIfPresent(Bool.self, forKey: .enableThinking) ?? false
        useMCP = try c.decodeIfPresent(Bool.self, forKey: .useMCP) ?? false
        // Backfill: sessions saved before workingDirectory had a default come back as nil. Anchor them
        // at ~/.mlx-serve/workspace so the agent's tools and MCP servers both have a sane default.
        let decoded = try c.decodeIfPresent(String.self, forKey: .workingDirectory)
        workingDirectory = decoded ?? ChatSession.defaultWorkingDirectory
        // Sessions saved before attach-a-folder persisted simply have none.
        attachedFolderPath = try c.decodeIfPresent(String.self, forKey: .attachedFolderPath)
    }

    /// Shared default cwd for all chat sessions. Same path used by CLILauncher, AgentEngine,
    /// and MCPManager.resolveWorkingDirectory.
    static let defaultWorkingDirectory: String = {
        let path = NSString(string: "~/.mlx-serve/workspace").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        return path
    }()
}

/// A tool call made by the assistant, stored on the assistant message for history replay.
struct SerializedToolCall: Codable, Equatable {
    let id: String
    let name: String
    let arguments: String // JSON string
}

struct ChatImage: Identifiable, Codable, Equatable {
    let id: UUID
    let data: Data  // JPEG bytes

    init(data: Data) {
        self.id = UUID()
        self.data = data
    }

    var base64URL: String {
        "data:image/jpeg;base64,\(data.base64EncodedString())"
    }
}

/// An audio clip attached to a message. `pcm` holds raw little-endian float32
/// mono samples at 16 kHz — the format the Gemma 4 12B unified audio embedder
/// frames into 640-sample tokens. Decoded client-side by `AudioPreprocessor`.
struct ChatAudio: Identifiable, Codable, Equatable {
    let id: UUID
    let name: String   // original filename, for the attachment chip
    let pcm: Data       // float32-LE 16 kHz mono samples

    init(name: String, pcm: Data) {
        self.id = UUID()
        self.name = name
        self.pcm = pcm
    }

    /// Number of decoded samples (4 bytes each) and the clip's duration.
    var sampleCount: Int { pcm.count / 4 }
    var durationSeconds: Double { Double(sampleCount) / 16_000.0 }
}

struct ChatMessage: Identifiable, Codable {
    let id: UUID
    var role: Role
    var content: String
    var reasoningContent: String?
    var isStreaming: Bool
    let timestamp: Date
    var agentPlan: AgentPlan?
    var toolResults: [StepResult]?
    var isAgentSummary: Bool
    var promptTokens: Int?
    var completionTokens: Int?
    var tokensPerSecond: Double?
    var toolCallId: String?   // For tool response messages
    var toolName: String?     // For tool response messages
    var toolCalls: [SerializedToolCall]? // Tool calls made BY this assistant message
    var images: [ChatImage]?  // Images attached to this message
    var audio: [ChatAudio]?   // Audio clips attached to this message
    // When true, the message is kept visible in the UI (e.g. preserved reasoning
    // from a cut-off or pad-only retry) but excluded from API history so it
    // can't confuse subsequent iterations of the agent loop.
    var failedRetry: Bool = false
    // Background-process handles (bg1, bg2, …) started by this tool-call round.
    // Drives the inline kill X on the tool-call card. Persisted, but resolves to
    // no live process after a restart (the registry isn't persisted).
    var processHandles: [String]? = nil

    enum Role: String, Codable {
        case system, user, assistant
    }

    init(role: Role, content: String, reasoningContent: String? = nil) {
        self.id = UUID()
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
        self.isStreaming = false
        self.timestamp = Date()
        self.agentPlan = nil
        self.toolResults = nil
        self.isAgentSummary = false
    }

    enum CodingKeys: String, CodingKey {
        case id, role, content, reasoningContent, isStreaming, timestamp
        case agentPlan, toolResults, isAgentSummary
        case promptTokens, completionTokens, tokensPerSecond
        case toolCallId, toolName, toolCalls, images, audio, failedRetry, processHandles
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        role = try c.decode(Role.self, forKey: .role)
        content = try c.decode(String.self, forKey: .content)
        reasoningContent = try c.decodeIfPresent(String.self, forKey: .reasoningContent)
        isStreaming = try c.decode(Bool.self, forKey: .isStreaming)
        timestamp = try c.decode(Date.self, forKey: .timestamp)
        agentPlan = try c.decodeIfPresent(AgentPlan.self, forKey: .agentPlan)
        toolResults = try c.decodeIfPresent([StepResult].self, forKey: .toolResults)
        isAgentSummary = try c.decodeIfPresent(Bool.self, forKey: .isAgentSummary) ?? false
        promptTokens = try c.decodeIfPresent(Int.self, forKey: .promptTokens)
        completionTokens = try c.decodeIfPresent(Int.self, forKey: .completionTokens)
        tokensPerSecond = try c.decodeIfPresent(Double.self, forKey: .tokensPerSecond)
        toolCallId = try c.decodeIfPresent(String.self, forKey: .toolCallId)
        toolName = try c.decodeIfPresent(String.self, forKey: .toolName)
        toolCalls = try c.decodeIfPresent([SerializedToolCall].self, forKey: .toolCalls)
        images = try c.decodeIfPresent([ChatImage].self, forKey: .images)
        audio = try c.decodeIfPresent([ChatAudio].self, forKey: .audio)
        failedRetry = try c.decodeIfPresent(Bool.self, forKey: .failedRetry) ?? false
        processHandles = try c.decodeIfPresent([String].self, forKey: .processHandles)
    }
}

struct ModelInfo {
    var name: String
    var quantBits: Int
    var layers: Int
    var hiddenSize: Int
    var vocabSize: Int
    /// Effective context length the running server is using right now —
    /// `max_context_size` if --ctx-size was passed, else the memory-bounded
    /// safe ceiling. Shifts when the user changes Settings → restarts.
    var contextLength: Int
    /// The model's own `max_position_embeddings` from config.json, capped
    /// only by what the architecture supports. Stable across server restarts;
    /// use this for UI like the "Model max" pill in Settings.
    var modelMaxTokens: Int
    /// `model_type` from config.json — "gemma4", "qwen3_5_moe", "llama", etc.
    /// Empty string when talking to a pre-drafter-UX server build.
    var architecture: String = ""
    /// True when the model has any MoE (sparse expert) layers. Drives the
    /// soft-warning pill in Settings → Drafter, since drafter regresses on
    /// MoE targets at single-stream batch=1.
    var isMoE: Bool = false
    /// True when the model advertises the `audio` capability (Gemma 4 12B
    /// unified). Gates the mic button + audio-file attachment in chat — other
    /// models silently ignore audio, so we only surface it where it works.
    var supportsAudio: Bool = false
    /// True when the model advertises the `vision` capability (a SigLIP-style
    /// encoder is loaded) — `false` for text-only models AND when the server
    /// was launched with `--no-vision`. The Telegram bridge reads this to
    /// decide whether to forward an incoming photo or refuse it.
    var supportsVision: Bool = false
    /// True when the model advertises the `embeddings` capability (encoder-
    /// only BERT entries, loaded or stub). DocumentIndex uses this to pick a
    /// GPU embedder for folder indexing.
    var supportsEmbeddings: Bool = false
    /// Raw `capabilities` array as the server sent it ("chat", "vision",
    /// "image", "video", "audio", "embeddings", …). `slotKind` classifies a
    /// registry entry into a tray slot from these.
    var capabilities: [String] = []
    /// True when the running server was launched with `--drafter <dir>` and
    /// the drafter+target pair validated. Drives the green status pill.
    var drafterLoaded: Bool = false
    /// Absolute path passed to `--drafter` at startup. nil when the server
    /// has no drafter loaded.
    var drafterPath: String? = nil
    /// True when the model dir shipped an `mtp/weights.safetensors` sidecar and
    /// the server loaded the native multi-token-prediction head. Drives the
    /// "+MTP" speedup badge under the model name in the tray.
    var mtpLoaded: Bool = false
    /// Plan 05 Phase G — multi-model fields. All optional so older
    /// servers (single-model) still decode without these.
    /// Whether this entry currently holds resident weights.
    var loaded: Bool = true
    /// Entry state from the registry: "ready" | "unloaded" | "loading" |
    /// "error" | "evicting". nil on pre-Phase-D servers.
    var state: String? = nil
    /// Approximate bytes resident in GPU memory; 0 for unloaded entries.
    var bytesResident: UInt64 = 0
    /// Sum of *.safetensors sizes on disk; nil when scan failed or
    /// pre-Phase-E server.
    var bytesOnDisk: UInt64? = nil

    /// Model-author sampling recommendations from the model's
    /// `generation_config.json` (Qwen 3.6: top_k 20 / top_p 0.95; Gemma 4:
    /// top_k 64 / top_p 0.95). nil when the model ships no recommendation or
    /// the server pre-dates this field. Settings shows these as guidance pills
    /// next to the per-request sampling sliders — the model behaves best near
    /// these values. Note the running server already falls back to them when a
    /// request omits the param AND no launch flag overrides (top_k=0 case).
    var recTemperature: Double? = nil
    var recTopP: Double? = nil
    var recTopK: Int? = nil

    /// Which backend serves this model — derived from `architecture`
    /// (`model_type` in config.json / the GGUF stub). Drives the engine-
    /// aware Settings UI so toggles that don't apply (e.g. MLX `--kv-quant`
    /// on a GGUF target) are hidden instead of silently no-op'ing.
    var engine: ServerEngine {
        switch architecture {
        case "gguf": return .llama
        case "deepseek_v4": return .dsv4
        default: return .mlx
        }
    }

    /// Short "speedup active" badge for the tray under the model name, or nil
    /// when no speculative-decoding head is loaded. MTP takes priority over the
    /// drafter (mirrors server dispatch: MTP > drafter > PLD), so at most one
    /// shows. PLD is intentionally NOT badged — it's content-adaptive (gated off
    /// on novel prompts) rather than a loaded asset.
    var specDecodeBadge: String? {
        if mtpLoaded { return "+MTP" }
        if drafterLoaded { return "+Drafter" }
        return nil
    }

    /// Classify a registry entry into a tray slot from its capabilities.
    /// "chat" wins first: a multimodal chat model also advertises "vision"/
    /// "audio" (input modalities), while gen engines advertise ONLY their
    /// output modality ("image" / "video" / "audio").
    var slotKind: ModelSlotKind {
        if capabilities.contains("chat") { return .chat }
        if capabilities.contains("video") { return .videoGen }
        if capabilities.contains("image") { return .imageGen }
        if capabilities.contains("audio") { return .audioGen }
        if supportsEmbeddings || capabilities.contains("embeddings") { return .embedding }
        return .chat
    }
}

/// What a resident registry entry is FOR — drives the icon + label on the
/// tray's model slots (chat model vs image/video/audio generator).
enum ModelSlotKind {
    case chat, imageGen, videoGen, audioGen, embedding

    var icon: String {
        switch self {
        case .chat:      return "bubble.left.and.bubble.right"
        case .imageGen:  return "photo"
        case .videoGen:  return "film"
        case .audioGen:  return "waveform"
        case .embedding: return "doc.text.magnifyingglass"
        }
    }

    var label: String {
        switch self {
        case .chat:      return "Chat"
        case .imageGen:  return "Image generation"
        case .videoGen:  return "Video generation"
        case .audioGen:  return "Audio generation"
        case .embedding: return "Embeddings"
        }
    }
}

/// Which embedded engine is serving the active model. The mlx-serve binary
/// picks this at load time based on the model file/dir layout (`.gguf` →
/// llama.cpp or ds4; `.safetensors` dir → MLX), so the Swift app reads
/// `ModelInfo.engine` to decide which knobs are relevant. The Settings UI
/// uses this to show MLX-only sections (PLD/drafter/KV-quant) only when
/// they actually do something, and to surface a GGUF-specific section
/// (--llama-kv-quant / --llama-cache-entries) when llama.cpp is active.
enum ServerEngine: String, CaseIterable {
    /// MLX safetensors path (default).
    case mlx
    /// Embedded llama.cpp engine (any `.gguf` except DSV4-Flash).
    case llama
    /// Embedded ds4 engine (DeepSeek-V4-Flash GGUF).
    case dsv4

    /// Short human label for the running-model badge / section headings.
    var label: String {
        switch self {
        case .mlx:   return "MLX"
        case .llama: return "llama.cpp (GGUF)"
        case .dsv4:  return "ds4 (DSV4-Flash)"
        }
    }
}

struct MemoryInfo {
    var activeBytes: Int64
    var peakBytes: Int64
    /// RAM available for a new allocation, computed server-side with the SAME
    /// formula as the model-load pre-flight (`status.getAvailableMemBytes` =
    /// total − wired − compressor). This is reclaimable-available, not unused:
    /// it counts file cache and pageable memory that macOS evicts under
    /// allocation pressure, so it's typically much larger than "free". 0 when the
    /// server build predates the field — the tray hides the line then. Distinct
    /// axis from `activeBytes` (the MLX GPU-allocator footprint).
    var availableBytes: Int64
    var maxSafeContext: Int

    var activeFormatted: String { Self.format(activeBytes) }
    var peakFormatted: String { Self.format(peakBytes) }
    var availableFormatted: String { Self.format(availableBytes) }

    /// Fraction (0...1) of `totalBytes` physical RAM occupied by the model's GPU
    /// (MLX) footprint. The bar's denominator is total RAM — NOT `peak×2`, which
    /// pinned the old bar at exactly 0.5 whenever `active == peak` (the steady
    /// state after load). Returns 0 for a non-positive total.
    func gpuFraction(ofTotal totalBytes: Int64) -> Double {
        Self.fraction(activeBytes, of: totalBytes)
    }

    /// Fraction (0...1) of `totalBytes` physical RAM currently available
    /// (reclaimable). Same denominator as `gpuFraction` so the two bars are
    /// directly comparable.
    func availableFraction(ofTotal totalBytes: Int64) -> Double {
        Self.fraction(availableBytes, of: totalBytes)
    }

    private static func fraction(_ part: Int64, of total: Int64) -> Double {
        guard total > 0 else { return 0 }
        return min(1.0, max(0.0, Double(part) / Double(total)))
    }

    /// Decode the `memory` object from the server's `/props` response. Pure and
    /// testable; `APIClient.fetchProps` calls this after pulling `json["memory"]`.
    static func parse(_ mem: [String: Any]) -> MemoryInfo {
        MemoryInfo(
            activeBytes: mem["active_bytes"] as? Int64 ?? 0,
            peakBytes: mem["peak_bytes"] as? Int64 ?? 0,
            availableBytes: mem["available_bytes"] as? Int64 ?? 0,
            maxSafeContext: mem["max_safe_context"] as? Int ?? 0
        )
    }

    static func format(_ bytes: Int64) -> String {
        let gb = Double(bytes) / (1024 * 1024 * 1024)
        if gb >= 1 { return String(format: "%.1f GB", gb) }
        let mb = Double(bytes) / (1024 * 1024)
        return String(format: "%.0f MB", mb)
    }

    /// Format a `[min, max]` byte range as a single string with a shared unit
    /// (GB if `max ≥ 1 GB`, else MB). Used by `HFModel.ramEstimate` to surface
    /// a GGUF repo's smallest-to-largest quant size in one column slot
    /// (e.g. "1.7–8.5 GB") without blowing past the column's ~80px budget.
    /// The "–" is U+2013 (en dash) — same glyph as `MemoryInfo.format`'s
    /// existing strings stay narrow with.
    static func formatRange(_ minBytes: Int64, _ maxBytes: Int64) -> String {
        let lo = min(minBytes, maxBytes)
        let hi = max(minBytes, maxBytes)
        if lo == hi { return format(lo) }
        let gbDivisor = Double(1024 * 1024 * 1024)
        let mbDivisor = Double(1024 * 1024)
        if Double(hi) / gbDivisor >= 1.0 {
            return String(format: "%.1f\u{2013}%.1f GB", Double(lo) / gbDivisor, Double(hi) / gbDivisor)
        }
        return String(format: "%.0f\u{2013}%.0f MB", Double(lo) / mbDivisor, Double(hi) / mbDivisor)
    }
}

enum ServerStatus: Equatable {
    case stopped
    case starting
    case running
    case error(String)

    var label: String {
        switch self {
        case .stopped: "Stopped"
        case .starting: "Starting..."
        case .running: "Running"
        case .error(let msg): "Error: \(msg)"
        }
    }

    var color: String {
        switch self {
        case .stopped: "red"
        case .starting: "orange"
        case .running: "green"
        case .error: "red"
        }
    }
}

enum LocalModelSource: String, Codable, Hashable {
    case mlxServe
    case lmStudio
    case custom
}

/// Distinguishes a base model from a paired drafter checkpoint. Drafters are
/// `gemma-4-*-it-assistant-bf16` directories whose `config.json` declares
/// `model_type: "gemma4_assistant"` — they aren't loadable as a target on
/// their own, so the Model Browser groups them separately and Settings hides
/// them from the model picker.
enum ModelKind: String, Codable, Hashable {
    case base
    case drafter
}

/// Which embedded engine serves a model. Mirrors the server's auto-routing:
/// safetensors dirs run on the native MLX path; `.gguf` files route to the
/// embedded llama.cpp engine, except DeepSeek-V4-Flash which routes to ds4
/// (`model_discovery.isDs4GgufBasename`, surfaced client-side as
/// `DownloadManager.ggufModelType` → `deepseek_v4`).
enum ModelEngine: String, Hashable {
    case mlx
    case llamaCpp
    case ds4

    /// Compact tag for picker-row disambiguation.
    var shortLabel: String {
        switch self {
        case .mlx: "MLX-Serve"
        case .llamaCpp: "GGUF"
        case .ds4: "DS4"
        }
    }

    /// Weight-format tag for metadata captions — the useful MLX-vs-GGUF
    /// distinction, without the "MLX-Serve" app-name noise (`shortLabel` keeps
    /// that for picker disambiguation).
    var formatLabel: String {
        switch self {
        case .mlx: "MLX"
        case .llamaCpp: "GGUF"
        case .ds4: "GGUF·DS4"
        }
    }

    /// Human-readable engine name for the status menu.
    var displayName: String {
        switch self {
        case .mlx: "MLX-Serve"
        case .llamaCpp: "GGUF · llama.cpp"
        case .ds4: "GGUF · DS4"
        }
    }
}

struct LocalModel: Identifiable, Hashable {
    let id: String
    let name: String
    let path: String
    let sizeFormatted: String
    let modelType: String
    let source: LocalModelSource
    let kind: ModelKind
    // The fields below are read from `config.json` at discovery (the
    // authoritative source); they default to nil/false for GGUF and any model
    // whose config we didn't parse, in which case the accessors fall back to
    // name-derived hints where possible.
    /// `vision_config` present on a non-`_text` architecture.
    var hasVision: Bool = false
    /// Quantization bit-width from `config.json`'s `quantization.bits`.
    var quantBits: Int? = nil
    /// `max_position_embeddings`.
    var contextLength: Int? = nil
    /// Total MoE experts (`num_experts` / `num_local_experts` / `n_routed_experts`).
    var numExperts: Int? = nil
    /// Active MoE experts per token (`num_experts_per_tok`).
    var activeExperts: Int? = nil
    /// The `.gguf` basename this model IS, when it's one quant of a GGUF repo.
    /// A repo folder holds many quants and each is separately loadable, so
    /// discovery emits one `LocalModel` per file and `path` points at the file.
    /// nil for MLX checkpoints, whose `path` is the directory.
    var quantFile: String? = nil

    var isSupportedArchitecture: Bool {
        supportedModelTypes.contains(modelType) || isMediaModelType(modelType)
    }

    /// Offerable by chat-model pickers (tray menu, task sheet, auto-select):
    /// a base checkpoint whose architecture serves chat completions. Excludes
    /// drafters, media models (LTX "AudioVideo", FLUX/Krea, Qwen3-TTS,
    /// Hunyuan3D, AceStep), the Falconsai NSFW classifier ("vit"), and
    /// embeddings-only "bert" encoders — those live under ~/.mlx-serve/models
    /// as gen-pane / doc-RAG dependencies and load by path, never as the
    /// tray's primary model. The Model Browser's Downloaded tab still lists
    /// them (size + delete) and, since they ARE supported architectures,
    /// no longer flags them "Unsupported".
    var isChatPickable: Bool {
        kind == .base && isSupportedArchitecture && modelType != "bert" && !isMediaModelType(modelType)
    }

    /// Likely tool/function-calling support (name heuristic, shared with the
    /// search rows via `HFModel.likelyToolCalling`).
    var hasToolCalling: Bool {
        HFModel.likelyToolCalling(forName: name)
    }

    /// Quantization label. Prefers `config.json`'s `bits` (authoritative);
    /// falls back to the name parser for GGUF / configs without a quant block.
    var quantization: String? {
        if let b = quantBits { return "\(b)-bit" }
        return HFModel.quantizationLabel(forId: name)
    }

    /// Headline parameter count parsed from the name (e.g. "32B", "30B"). This
    /// is the one figure NOT in config.json — it's marketing-rounded in the name
    /// and only exactly recoverable by summing tensor shapes. nil when absent.
    var paramSize: String? {
        HFModel.paramSizeLabel(forName: name)
    }

    /// "8/128 experts" (active/total) when this is an MoE config, else nil.
    var expertSummary: String? {
        guard let total = numExperts, let active = activeExperts else { return nil }
        return "\(active)/\(total) experts"
    }

    /// Context window as a compact label, e.g. 262144 → "256K ctx", 1048576 →
    /// "1M ctx". nil when unknown.
    var contextSummary: String? {
        guard let n = contextLength else { return nil }
        return Self.formatContext(n)
    }

    static func formatContext(_ n: Int) -> String {
        if n >= 1024 * 1024, n % (1024 * 1024) == 0 { return "\(n / (1024 * 1024))M ctx" }
        if n >= 1024 { return "\(n / 1024)K ctx" }
        return "\(n) ctx"
    }

    /// One-line metadata caption for the Downloaded tab, e.g.
    /// "30B · 8-bit · 8/128 experts · 256K ctx · qwen3_moe · MLX". Everything
    /// except the headline param count is config-sourced; tokens the model
    /// doesn't have are omitted. Capabilities (vision / tools) render as icons
    /// in the row, not here.
    var metadataSummary: String {
        var tokens: [String] = []
        if let p = paramSize { tokens.append(p) }
        if let q = quantization { tokens.append(q) }
        if let e = expertSummary { tokens.append(e) }
        if let c = contextSummary { tokens.append(c) }
        tokens.append(modelType)
        tokens.append(engine.formatLabel)
        return tokens.joined(separator: " · ")
    }

    /// Defaults to `.mlx` (MLX-Serve) whenever the engine can't be
    /// positively determined — only a `.gguf` path routes elsewhere.
    var engine: ModelEngine {
        guard path.lowercased().hasSuffix(".gguf") else { return .mlx }
        return modelType == "deepseek_v4" ? .ds4 : .llamaCpp
    }

    /// What every picker and list row shows. For a GGUF quant that's the repo
    /// name plus the quant it is (`unsloth/Qwen3.5-4B-GGUF · Q4_K_M`) — sibling
    /// quants share a `name`, so the name alone can't tell them apart, and
    /// `name` has to stay the repo name because filters and grouping key off it.
    var displayLabel: String {
        guard let quantFile else { return name }
        return "\(name) · \(DownloadManager.quantLabel(forFilename: quantFile))"
    }

    /// Display labels shared by more than one model. macOS `.menu` Pickers key
    /// the checkmark state by item TITLE, so two same-titled rows (one GGUF,
    /// one MLX) both render selected — rows whose label is in this set need an
    /// engine suffix to keep titles unique. Computed on `displayLabel`, not
    /// `name`: two quants of one repo share a name but are already distinct
    /// here, and tagging them both "· GGUF" would leave them identical.
    static func duplicateNames(in models: [LocalModel]) -> Set<String> {
        var seen = Set<String>()
        var dups = Set<String>()
        for m in models {
            if !seen.insert(m.displayLabel).inserted {
                dups.insert(m.displayLabel)
            }
        }
        return dups
    }
}

/// Gemma 4 size designators that have a published assistant drafter today.
/// Naming intentionally matches the segment in `gemma-4-{E2B,...}-it-...`.
enum GemmaVariant: String, CaseIterable, Hashable {
    case E2B
    case E4B
    case gemma12B = "12B"
    case gemma31B = "31B"
    case moe26B = "26B-A4B"

    /// Full HF repo path of the assistant drafter. All variants use the
    /// `mlx-community/...-it-assistant-bf16` uniform path. bf16 is the only
    /// quant mlx-community publishes for the older variants (E2B/E4B/26B-A4B/
    /// 31B as of 2026-06 — HF 401s on the 8bit suffix for those), so we keep
    /// the new 12B unified drafter on the same suffix even though an 8bit
    /// build exists. Adding a new variant? Verify with
    /// `curl -sI https://huggingface.co/api/models/<repo>` first.
    var drafterRepoId: String {
        "mlx-community/gemma-4-\(rawValue)-it-assistant-bf16"
    }

    /// Last path component of the drafter repo — also the on-disk dir name
    /// the discoverer matches against.
    var drafterDirName: String {
        (drafterRepoId as NSString).lastPathComponent
    }

    /// Human-readable target label for the pairing banner ("for E4B").
    var label: String { rawValue }
}

struct LocalDrafter: Hashable {
    let url: URL
    let variant: GemmaVariant
}

struct GemmaModelOption: Identifiable {
    let id: String
    let displayName: String
    let repoId: String
    let sizeEstimate: String
    /// Optional explicit GGUF filename within `repoId`. When non-nil the
    /// downloader resolves a single `.gguf` artifact and the server loads it
    /// through the embedded ds4 engine instead of the MLX/safetensors path.
    let ggufFilename: String?
    /// Minimum host RAM (bytes) before this entry is surfaced in the UI. 0 = no gate.
    let minHostRamBytes: UInt64

    init(id: String, displayName: String, repoId: String, sizeEstimate: String, ggufFilename: String? = nil, minHostRamBytes: UInt64 = 0) {
        self.id = id
        self.displayName = displayName
        self.repoId = repoId
        self.sizeEstimate = sizeEstimate
        self.ggufFilename = ggufFilename
        self.minHostRamBytes = minHostRamBytes
    }
}

let gemmaModelOptions: [GemmaModelOption] = [
    // E2B: 5.1B params, 2.3B active — fits 8 GB+ Macs
    GemmaModelOption(id: "e2b-4bit", displayName: "Gemma 4 E2B (4-bit)", repoId: "mlx-community/gemma-4-e2b-it-4bit", sizeEstimate: "~3.4 GB"),
    GemmaModelOption(id: "e2b-8bit", displayName: "Gemma 4 E2B (8-bit)", repoId: "mlx-community/gemma-4-e2b-it-8bit", sizeEstimate: "~5.5 GB"),
    // E4B: 8B params, 4.5B active — fits 16 GB+ Macs
    GemmaModelOption(id: "e4b-4bit", displayName: "Gemma 4 E4B (4-bit)", repoId: "mlx-community/gemma-4-e4b-it-4bit", sizeEstimate: "~5.2 GB"),
    GemmaModelOption(id: "e4b-8bit", displayName: "Gemma 4 E4B (8-bit)", repoId: "mlx-community/gemma-4-e4b-it-8bit", sizeEstimate: "~8.5 GB"),
    // 12B: dense — fits 16 GB+ Macs (4-bit) or 24 GB+ (8-bit).
    GemmaModelOption(id: "12b-4bit", displayName: "Gemma 4 12B (4-bit)", repoId: "mlx-community/gemma-4-12b-it-4bit", sizeEstimate: "~7.1 GB, needs 16 GB+ RAM"),
    GemmaModelOption(id: "12b-8bit", displayName: "Gemma 4 12B (8-bit)", repoId: "mlx-community/gemma-4-12b-it-8bit", sizeEstimate: "~12.8 GB, needs 24 GB+ RAM"),
    // Gemma 3 12B (4-bit) — a capable chat model in its own right, AND the text
    // encoder LTX-Video uses. Downloading the LTX bundle pulls this; it also
    // shows here so it can be picked as a standalone chat model.
    GemmaModelOption(id: "gemma3-12b-4bit", displayName: "Gemma 3 12B (4-bit)", repoId: "mlx-community/gemma-3-12b-it-4bit", sizeEstimate: "~7.1 GB, needs 16 GB+ RAM — also the LTX-Video text encoder"),
    // 26B-A4B: 25.2B MoE, only 3.8B active per token — fits 24 GB+ Macs (4-bit) or 36 GB+ (8-bit)
    GemmaModelOption(id: "26b-a4b-4bit", displayName: "Gemma 4 26B-A4B (4-bit)", repoId: "mlx-community/gemma-4-26b-a4b-it-4bit", sizeEstimate: "~15.6 GB, needs 24 GB+ RAM"),
    GemmaModelOption(id: "26b-a4b-8bit", displayName: "Gemma 4 26B-A4B (8-bit)", repoId: "mlx-community/gemma-4-26b-a4b-it-8bit", sizeEstimate: "~28 GB, needs 36 GB+ RAM"),
    // 31B: 31B dense — fits 36 GB+ Macs (4-bit) or 48 GB+ (8-bit)
    GemmaModelOption(id: "31b-4bit", displayName: "Gemma 4 31B (4-bit)", repoId: "mlx-community/gemma-4-31b-it-4bit", sizeEstimate: "~18.4 GB, needs 36 GB+ RAM"),
    GemmaModelOption(id: "31b-8bit", displayName: "Gemma 4 31B (8-bit)", repoId: "mlx-community/gemma-4-31b-it-8bit", sizeEstimate: "~33.8 GB, needs 48 GB+ RAM"),
    // Qwen 3.6 27B dense (4-bit) with a native MTP head — fits 24 GB+ Macs. Ships
    // an mtp/weights.safetensors sidecar; the server auto-loads it for multi-token
    // speculative decode (~1.1–1.4× decode on agent/code workloads).
    GemmaModelOption(id: "qwen36-27b-4bit-mtp", displayName: "Qwen 3.6 27B (4-bit, MTP)", repoId: "ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve", sizeEstimate: "~16.6 GB, needs 24 GB+ RAM"),
    // DeepSeek-V4-Flash via ds4 GGUF — 96 GB+ Macs only. Served by the embedded
    // ds4 engine (antirez/ds4) rather than the MLX/safetensors path.
    GemmaModelOption(
        id: "dsv4-flash-gguf",
        displayName: "DeepSeek-V4-Flash (ds4)",
        repoId: "antirez/deepseek-v4-gguf",
        sizeEstimate: "~85 GB, needs 96 GB+ RAM",
        ggufFilename: "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf",
        minHostRamBytes: 96 * (UInt64(1) << 30)
    ),
]

/// Subset of `gemmaModelOptions` surfaced in the menu-bar Download Models
/// popover. 4-bit quants are the default tray choice — they fit the widest
/// range of Macs (the 4-bit 31B is ~18 GB vs 33 GB at 8-bit) so most users
/// can install something useful without bouncing into the full Model Browser.
/// DSV4 has only the one GGUF, so it rides along unconditionally.
/// Excludes Gemma 3/4 12B and E2B since they're available in the model browser.
let gemmaModelOptionsTrayMenu = gemmaModelOptions.filter {
    !$0.id.contains("12b") && !$0.id.contains("e2b") && ($0.id.contains("4bit") || $0.id.contains("dsv4"))
}

/// Subset of `gemmaModelOptions` visible on the current host. Hides entries
/// whose `minHostRamBytes` exceeds the system RAM.
var availableGemmaModelOptions: [GemmaModelOption] {
    let ram = ProcessInfo.processInfo.physicalMemory
    return gemmaModelOptions.filter { $0.minHostRamBytes <= ram }
}
