import Foundation

struct ChatSession: Identifiable, Codable {
    let id: UUID
    var title: String
    var messages: [ChatMessage]
    var createdAt: Date
    var updatedAt: Date
    var mode: ChatMode
    var workingDirectory: String?

    init(title: String = "New Chat") {
        self.id = UUID()
        self.title = title
        self.messages = []
        self.createdAt = Date()
        self.updatedAt = Date()
        self.mode = .chat
        self.workingDirectory = ChatSession.defaultWorkingDirectory
    }

    enum CodingKeys: String, CodingKey {
        case id, title, messages, createdAt, updatedAt, mode, workingDirectory
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        title = try c.decode(String.self, forKey: .title)
        messages = try c.decode([ChatMessage].self, forKey: .messages)
        createdAt = try c.decode(Date.self, forKey: .createdAt)
        updatedAt = try c.decode(Date.self, forKey: .updatedAt)
        mode = try c.decodeIfPresent(ChatMode.self, forKey: .mode) ?? .chat
        // Backfill: sessions saved before workingDirectory had a default come back as nil. Anchor them
        // at ~/.mlx-serve/workspace so the agent's tools and MCP servers both have a sane default.
        let decoded = try c.decodeIfPresent(String.self, forKey: .workingDirectory)
        workingDirectory = decoded ?? ChatSession.defaultWorkingDirectory
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
    // When true, the message is kept visible in the UI (e.g. preserved reasoning
    // from a cut-off or pad-only retry) but excluded from API history so it
    // can't confuse subsequent iterations of the agent loop.
    var failedRetry: Bool = false

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
        case toolCallId, toolName, toolCalls, images, failedRetry
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
        failedRetry = try c.decodeIfPresent(Bool.self, forKey: .failedRetry) ?? false
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
    /// True when the running server was launched with `--drafter <dir>` and
    /// the drafter+target pair validated. Drives the green status pill.
    var drafterLoaded: Bool = false
    /// Absolute path passed to `--drafter` at startup. nil when the server
    /// has no drafter loaded.
    var drafterPath: String? = nil
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
    var maxSafeContext: Int

    var activeFormatted: String { Self.format(activeBytes) }
    var peakFormatted: String { Self.format(peakBytes) }

    static func format(_ bytes: Int64) -> String {
        let gb = Double(bytes) / (1024 * 1024 * 1024)
        if gb >= 1 { return String(format: "%.1f GB", gb) }
        let mb = Double(bytes) / (1024 * 1024)
        return String(format: "%.0f MB", mb)
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

struct LocalModel: Identifiable, Hashable {
    let id: String
    let name: String
    let path: String
    let sizeFormatted: String
    let modelType: String
    let source: LocalModelSource
    let kind: ModelKind

    var isSupportedArchitecture: Bool {
        supportedModelTypes.contains(modelType)
    }
}

/// Gemma 4 size designators that have a published assistant drafter today.
/// Naming intentionally matches the segment in `gemma-4-{E2B,...}-it-...`.
enum GemmaVariant: String, CaseIterable, Hashable {
    case E2B
    case E4B
    case gemma31B = "31B"
    case moe26B = "26B-A4B"

    /// Repo basename (no author prefix) of the assistant drafter.
    var drafterDirName: String {
        "gemma-4-\(rawValue)-it-assistant-bf16"
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
    // 26B-A4B: 25.2B MoE, only 3.8B active per token — fits 24 GB+ Macs (4-bit) or 36 GB+ (8-bit)
    GemmaModelOption(id: "26b-a4b-4bit", displayName: "Gemma 4 26B-A4B (4-bit)", repoId: "mlx-community/gemma-4-26b-a4b-it-4bit", sizeEstimate: "~15.6 GB, needs 24 GB+ RAM"),
    GemmaModelOption(id: "26b-a4b-8bit", displayName: "Gemma 4 26B-A4B (8-bit)", repoId: "mlx-community/gemma-4-26b-a4b-it-8bit", sizeEstimate: "~28 GB, needs 36 GB+ RAM"),
    // 31B: 31B dense — fits 36 GB+ Macs (4-bit) or 48 GB+ (8-bit)
    GemmaModelOption(id: "31b-4bit", displayName: "Gemma 4 31B (4-bit)", repoId: "mlx-community/gemma-4-31b-it-4bit", sizeEstimate: "~18.4 GB, needs 36 GB+ RAM"),
    GemmaModelOption(id: "31b-8bit", displayName: "Gemma 4 31B (8-bit)", repoId: "mlx-community/gemma-4-31b-it-8bit", sizeEstimate: "~33.8 GB, needs 48 GB+ RAM"),
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

let gemmaModelOptions8BitOnly = gemmaModelOptions.filter { $0.id.contains("8bit") || $0.id.contains("dsv4") }

/// Subset of `gemmaModelOptions` visible on the current host. Hides entries
/// whose `minHostRamBytes` exceeds the system RAM.
var availableGemmaModelOptions: [GemmaModelOption] {
    let ram = ProcessInfo.processInfo.physicalMemory
    return gemmaModelOptions.filter { $0.minHostRamBytes <= ram }
}
