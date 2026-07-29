import Foundation

/// A named persona: a saved system prompt plus the settings a conversation with
/// it should run under. You describe the assistant you want, a model writes the
/// prompt (`AgentWriter`), and every surface that can start a turn — chat tabs,
/// the voice tray, scheduled tasks, Telegram, the Quick Launcher — can run as
/// that agent.
///
/// The field names in the first block are byte-identical to the iPhone app's
/// `Agent` (mlx-iphone `Models/Agent.swift`) so the two files stay importable;
/// everything below it is macOS-only and every one of those is an OVERRIDE —
/// `nil` means "use the app default", resolved once in `AgentResolution`. The one
/// thing an agent deliberately does NOT own is the Agent Sandbox: that stays a
/// single global flag.
struct Agent: Identifiable, Codable, Equatable {
    var id = UUID()
    /// Short label for the list, the tray picker and the chat title.
    var name: String
    /// What the user typed. Kept so the agent can be re-written later, and so
    /// the list can show the intent rather than the generated prose.
    var brief: String
    /// The generated (and always hand-editable) system prompt.
    var systemPrompt: String
    /// SF Symbol for the avatar disc.
    var symbol: String = "sparkles"
    /// The iPhone-shaped Kokoro voice name. Superseded by `voice` on macOS (which
    /// can also pick the system synthesizer or the cloned voice) but kept — and
    /// kept written — so an agents file moves between the two apps.
    var voiceID: String?
    var createdAt = Date()

    // MARK: macOS-only overrides — nil means "app default"

    /// Starter agents shipped as code constants: shown in the list, never
    /// editable, "Duplicate" to make one yours. Being constants (not seeded into
    /// the JSON) is what lets a later release improve them for existing users.
    var isBuiltIn: Bool = false
    /// Spoken name for hands-free voice. Stored as typed; consumers normalize
    /// through `WakeWord.normalizePhrase`, and blank falls back to the app phrase.
    var wakePhrase: String?
    /// Which engine + voice this agent speaks with. Wins over `voiceID`.
    var voice: AgentVoice?
    /// Pinned model. nil = "Current" (whatever is selected right now). A local
    /// path, or a LAN `id@peer` id — those pass through untouched.
    var workingDirectory: String?
    var modelPath: String?
    var capabilities = AgentCapabilities()
    var autoApproveTools: Bool?
    var enableThinking: Bool?
    var temperature: Double?
    var maxTokens: Int?

    init(id: UUID = UUID(),
         name: String,
         brief: String = "",
         systemPrompt: String,
         symbol: String = "sparkles",
         voiceID: String? = nil,
         createdAt: Date = Date(),
         isBuiltIn: Bool = false,
         wakePhrase: String? = nil,
         voice: AgentVoice? = nil,
         workingDirectory: String? = nil,
         modelPath: String? = nil,
         capabilities: AgentCapabilities = AgentCapabilities(),
         autoApproveTools: Bool? = nil,
         enableThinking: Bool? = nil,
         temperature: Double? = nil,
         maxTokens: Int? = nil) {
        self.id = id
        self.name = name
        self.brief = brief
        self.systemPrompt = systemPrompt
        self.symbol = symbol
        self.voiceID = voiceID
        self.createdAt = createdAt
        self.isBuiltIn = isBuiltIn
        self.wakePhrase = wakePhrase
        self.voice = voice
        self.workingDirectory = workingDirectory
        self.modelPath = modelPath
        self.capabilities = capabilities
        self.autoApproveTools = autoApproveTools
        self.enableThinking = enableThinking
        self.temperature = temperature
        self.maxTokens = maxTokens
    }

    /// The agent a finished `AgentWriter.Draft` becomes. The symbol is guessed
    /// from the user's own words so a list of agents isn't a column of identical
    /// discs (the user can change it in the editor).
    init(draft: AgentWriter.Draft, brief: String) {
        self.init(name: draft.name,
                  brief: brief.trimmingCharacters(in: .whitespacesAndNewlines),
                  systemPrompt: draft.systemPrompt,
                  symbol: AgentSymbol.pick(for: "\(brief) \(draft.name)"))
    }

    /// The voice this agent speaks with, `voice` first and the iPhone-shaped
    /// Kokoro name as the fallback. nil = follow the app's voice settings.
    var resolvedVoice: AgentVoice? {
        if let voice { return voice }
        if let voiceID, !voiceID.isEmpty { return .kokoro(voiceID) }
        return nil
    }

    // MARK: - Tolerant coding

    // Absent keys default and unknown keys are ignored, so a build that adds a
    // field still reads an older file (and an older build a newer one). The
    // agents file outlives any single build — it survives upgrades and, one day,
    // a trip from the phone.
    enum CodingKeys: String, CodingKey {
        case id, name, brief, systemPrompt, symbol, voiceID, createdAt
        case isBuiltIn, wakePhrase, voice, workingDirectory, modelPath, capabilities
        case autoApproveTools, enableThinking, temperature, maxTokens
        /// iPhone-only spelling of the Web capability.
        case canSearchWeb
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? "Agent"
        brief = try c.decodeIfPresent(String.self, forKey: .brief) ?? ""
        systemPrompt = try c.decodeIfPresent(String.self, forKey: .systemPrompt) ?? ""
        symbol = try c.decodeIfPresent(String.self, forKey: .symbol) ?? "sparkles"
        voiceID = try c.decodeIfPresent(String.self, forKey: .voiceID)
        createdAt = try c.decodeIfPresent(Date.self, forKey: .createdAt) ?? Date()
        isBuiltIn = try c.decodeIfPresent(Bool.self, forKey: .isBuiltIn) ?? false
        wakePhrase = try c.decodeIfPresent(String.self, forKey: .wakePhrase)
        // An engine this build doesn't know must cost the agent its voice, not
        // the whole agent — `try?` so the row still loads.
        voice = (try? c.decodeIfPresent(AgentVoice.self, forKey: .voice)) ?? nil
        workingDirectory = try c.decodeIfPresent(String.self, forKey: .workingDirectory)
        modelPath = try c.decodeIfPresent(String.self, forKey: .modelPath)
        if let caps = try c.decodeIfPresent(AgentCapabilities.self, forKey: .capabilities) {
            capabilities = caps
        } else {
            // A file from the phone carries only `canSearchWeb`.
            var caps = AgentCapabilities()
            if let web = try c.decodeIfPresent(Bool.self, forKey: .canSearchWeb) { caps.web = web }
            capabilities = caps
        }
        autoApproveTools = try c.decodeIfPresent(Bool.self, forKey: .autoApproveTools)
        enableThinking = try c.decodeIfPresent(Bool.self, forKey: .enableThinking)
        temperature = try c.decodeIfPresent(Double.self, forKey: .temperature)
        maxTokens = try c.decodeIfPresent(Int.self, forKey: .maxTokens)
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(name, forKey: .name)
        try c.encode(brief, forKey: .brief)
        try c.encode(systemPrompt, forKey: .systemPrompt)
        try c.encode(symbol, forKey: .symbol)
        try c.encodeIfPresent(voiceID, forKey: .voiceID)
        try c.encode(createdAt, forKey: .createdAt)
        try c.encode(isBuiltIn, forKey: .isBuiltIn)
        try c.encodeIfPresent(wakePhrase, forKey: .wakePhrase)
        try c.encodeIfPresent(voice, forKey: .voice)
        try c.encodeIfPresent(workingDirectory, forKey: .workingDirectory)
        try c.encodeIfPresent(modelPath, forKey: .modelPath)
        try c.encode(capabilities, forKey: .capabilities)
        try c.encodeIfPresent(autoApproveTools, forKey: .autoApproveTools)
        try c.encodeIfPresent(enableThinking, forKey: .enableThinking)
        try c.encodeIfPresent(temperature, forKey: .temperature)
        try c.encodeIfPresent(maxTokens, forKey: .maxTokens)
        // Written for the phone's benefit, read back into `capabilities.web`.
        try c.encode(capabilities.web, forKey: .canSearchWeb)
    }
}

/// Which voice an agent speaks with. Mirrors the three engines the app's voice
/// picker already offers (`VoiceEngine`), so an agent voice is exactly one of
/// the choices a user can make globally — no fourth code path.
enum AgentVoice: Codable, Equatable, Hashable, Sendable {
    /// macOS `AVSpeechSynthesizer`; the identifier ("" = the system default).
    case system(String)
    /// Kokoro voice name, or a comma-separated blend ("af_bella,af_sky").
    case kokoro(String)
    /// Qwen3-TTS voice clone; the reference clip's path.
    case clone(String)

    var engine: VoiceEngine {
        switch self {
        case .system: return .system
        case .kokoro: return .kokoro
        case .clone: return .clone
        }
    }

    var value: String {
        switch self {
        case .system(let v), .kokoro(let v), .clone(let v): return v
        }
    }

    init(engine: VoiceEngine, value: String) {
        switch engine {
        case .system: self = .system(value)
        case .kokoro: self = .kokoro(value)
        case .clone: self = .clone(value)
        }
    }

    private enum CodingKeys: String, CodingKey { case engine, value }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let engine = try c.decode(VoiceEngine.self, forKey: .engine)
        self.init(engine: engine, value: try c.decodeIfPresent(String.self, forKey: .value) ?? "")
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(engine, forKey: .engine)
        try c.encode(value, forKey: .value)
    }
}

/// What an agent may do. Coarse toggles by default; `advancedTools` is the
/// per-tool escape hatch. `resolvedTools()` is the ONE function that turns this
/// into a concrete set — used both to filter the advertised tool list and to
/// refuse a tool at dispatch, so the two can't disagree.
struct AgentCapabilities: Codable, Equatable, Sendable {
    /// The tool loop (what the UI used to call "Agent mode").
    var tools: Bool = false
    var mcp: Bool = false
    /// Web reach: browse + webSearch.
    var web: Bool = true
    /// nil = derive from the coarse flags. Non-nil = the user went Advanced and
    /// their set IS the answer.
    var advancedTools: Set<AgentToolKind>?

    init(tools: Bool = false, mcp: Bool = false, web: Bool = true,
         advancedTools: Set<AgentToolKind>? = nil) {
        self.tools = tools
        self.mcp = mcp
        self.web = web
        self.advancedTools = advancedTools
    }

    /// The tool loop's own tools — everything but the web pair and the
    /// folder-gated `searchDocuments`.
    static let loopTools: Set<AgentToolKind> = [
        .shell, .cwd, .readFile, .writeFile, .editFile, .searchFiles, .listFiles,
        .saveMemory, .createTask, .listProcesses, .readProcessOutput, .killProcess,
        .generateImage, .generateSpeech, .generateMusic, .generateVideo,
    ]

    static let webTools: Set<AgentToolKind> = [.browse, .webSearch]

    /// Coarse flags → concrete tools, or the Advanced set verbatim.
    /// `searchDocuments` is deliberately absent: its gate is whether a document
    /// folder is attached, which is stronger than any capability toggle.
    func resolvedTools() -> Set<AgentToolKind> {
        if let advancedTools { return advancedTools }
        var out: Set<AgentToolKind> = []
        if tools { out.formUnion(Self.loopTools) }
        if web { out.formUnion(Self.webTools) }
        return out
    }

    /// Seed the Advanced set from the coarse resolution the first time the user
    /// opens it, so the two views can never disagree at the moment of switching.
    /// Idempotent — re-opening never clobbers a hand-edited set.
    mutating func openAdvanced() {
        if advancedTools == nil { advancedTools = resolvedTools() }
    }

    /// Back to coarse control.
    mutating func closeAdvanced() { advancedTools = nil }

    /// Tool names that were RENAMED, mapped to what they are now.
    ///
    /// The decoder drops names it doesn't recognise (a tool retired in a later
    /// build must not fail a whole agent's decode), which means a plain rename
    /// silently STRIPS the capability from every agent that had it. `generate_audio`
    /// split into `generate_speech` + `generate_music`; the speech half keeps the
    /// old name's meaning, so upgraders keep what they granted.
    static let renamedTools: [String: AgentToolKind] = ["generate_audio": .generateSpeech]

    /// Resolve one stored tool name: this build's spelling, then a rename, else
    /// nil (drop it).
    static func toolKind(forStoredName name: String) -> AgentToolKind? {
        AgentToolKind(rawValue: name) ?? renamedTools[name]
    }

    enum CodingKeys: String, CodingKey { case tools, mcp, web, advancedTools }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        tools = try c.decodeIfPresent(Bool.self, forKey: .tools) ?? false
        mcp = try c.decodeIfPresent(Bool.self, forKey: .mcp) ?? false
        web = try c.decodeIfPresent(Bool.self, forKey: .web) ?? true
        // A tool name this build doesn't know is dropped, not fatal.
        if let raw = try c.decodeIfPresent([String].self, forKey: .advancedTools) {
            advancedTools = Set(raw.compactMap(Self.toolKind(forStoredName:)))
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(tools, forKey: .tools)
        try c.encode(mcp, forKey: .mcp)
        try c.encode(web, forKey: .web)
        try c.encodeIfPresent(advancedTools.map { $0.map(\.rawValue).sorted() },
                              forKey: .advancedTools)
    }
}

// MARK: - Starters

extension Agent {
    /// Read-only agents that ship with the app so the feature is discoverable
    /// with nothing configured. Code constants, NOT rows seeded into
    /// `agents/index.json` — improving a starter later has to reach users who
    /// already ran the app. Duplicate one to edit it.
    static let starters: [Agent] = [
        Agent(id: UUID(uuidString: "A6E1F0C0-0001-4000-8000-000000000001")!,
              name: "Assistant",
              brief: "a plain, direct general assistant",
              systemPrompt: "You are a direct, capable assistant. You answer the question that was asked, in as few words as it takes, and you say plainly when you don't know something instead of guessing. You never pad an answer with restatements of the question or offers to help further.",
              symbol: "sparkles",
              isBuiltIn: true,
              // Plain chat with a persona — no tools, so a small model isn't
              // handed a tool block it doesn't need.
              capabilities: AgentCapabilities(tools: false, mcp: false, web: false)),
        Agent(id: UUID(uuidString: "A6E1F0C0-0001-4000-8000-000000000002")!,
              name: "Coder",
              brief: "a hands-on programmer that reads and edits real files",
              systemPrompt: "You are a careful programmer working in the user's own project. You answer concisely — the change, where it is, and anything that surprised you. You read the surrounding code before you change anything, and you match the conventions already there rather than importing your own. You make the smallest change that does the job, you run the tests when there are tests to run, and you report what you actually verified — never what you assume works. When something is broken beyond the task at hand, you say so instead of quietly widening the change.",
              symbol: "chevron.left.forwardslash.chevron.right",
              isBuiltIn: true,
              capabilities: AgentCapabilities(tools: true, mcp: false, web: true)),
        Agent(id: UUID(uuidString: "A6E1F0C0-0001-4000-8000-000000000003")!,
              name: "Researcher",
              brief: "looks things up on the web and summarises with sources",
              systemPrompt: "You are a research assistant. You answer briefly and lead with what the sources actually say. You look things up before answering rather than relying on memory, you read more than one source when the answer matters, and you always say where a claim came from. You separate what the sources actually say from your own inference, and you flag when the sources disagree or when the information looks out of date.",
              symbol: "globe",
              isBuiltIn: true,
              // Web only: the loop runs, but browse + webSearch are the whole
              // toolbox — no shell, no file writes.
              capabilities: AgentCapabilities(tools: false, mcp: false, web: true)),
        Agent(id: UUID(uuidString: "A6E1F0C0-0001-4000-8000-000000000004")!,
              name: "Chef",
              brief: "cooks from what's already in the kitchen",
              systemPrompt: "You are a practical home cook. You keep answers short — the dish, the steps, the timings. You suggest dishes from what the user already has, you keep steps short and in order, and you give real quantities and timings. You mention the one thing most likely to go wrong with a recipe, and you offer a substitution when an ingredient is missing rather than sending the user shopping.",
              symbol: "fork.knife",
              isBuiltIn: true,
              capabilities: AgentCapabilities(tools: false, mcp: false, web: false)),
    ]
}
