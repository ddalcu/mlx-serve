import Foundation

enum ChatMode: String, Codable {
    case chat
    case agent
}

/// Every tool the agent loop can dispatch. The raw value IS the wire name, and
/// the set of raw values must equal the set of names in
/// `AgentPrompt.toolDefinitions` — a tool in the JSON with no case here can't be
/// gated by an agent's capabilities, which is a silent hole in the whole
/// feature. `AgentCapabilityGateTests` pins that both ways.
enum AgentToolKind: String, Codable, CaseIterable, Sendable {
    case shell
    case readFile
    case writeFile
    case editFile
    case searchFiles
    case listFiles
    case browse
    case webSearch
    case saveMemory
    case cwd
    case searchDocuments
    case killProcess
    case readProcessOutput
    case listProcesses
    case createTask
    case generateImage = "generate_image"
    // Speech and music are separate tools on purpose. `generate_audio` next to
    // `generate_music` is ambiguous — music IS audio — and their arguments have
    // nothing in common (a line to speak vs. a style prompt, lyrics and a
    // duration). One overloaded tool makes a small local model guess.
    case generateSpeech = "generate_speech"
    case generateMusic = "generate_music"
    case generateVideo = "generate_video"
    /// Drives the sandbox desktop (XFCE in the guest VM) by accessibility
    /// tree + xdotool; screenshots only when the model has vision.
    case computer

    var icon: String {
        switch self {
        case .shell: "terminal"
        case .readFile: "doc.text"
        case .writeFile: "doc.text.fill"
        case .editFile: "pencil"
        case .searchFiles: "magnifyingglass"
        case .listFiles: "folder"
        case .browse: "globe"
        case .webSearch: "magnifyingglass"
        case .saveMemory: "brain"
        case .cwd: "folder.badge.gearshape"
        case .searchDocuments: "doc.text.magnifyingglass"
        case .killProcess: "xmark.octagon"
        case .readProcessOutput: "text.viewfinder"
        case .listProcesses: "list.bullet.rectangle"
        case .createTask: "calendar.badge.clock"
        case .generateImage: "photo"
        case .generateSpeech: "waveform"
        case .generateMusic: "music.note"
        case .generateVideo: "film"
        case .computer: "desktopcomputer"
        }
    }

    var displayName: String {
        switch self {
        case .shell: "Shell"
        case .readFile: "Read File"
        case .writeFile: "Write File"
        case .editFile: "Edit File"
        case .searchFiles: "Search Files"
        case .listFiles: "List Files"
        case .browse: "Browse"
        case .webSearch: "Web Search"
        case .saveMemory: "Save Memory"
        case .cwd: "Change Directory"
        case .searchDocuments: "Search Documents"
        case .killProcess: "Kill Process"
        case .readProcessOutput: "Read Process Output"
        case .listProcesses: "List Processes"
        case .createTask: "Create Task"
        case .generateImage: "Generate Image"
        case .generateSpeech: "Generate Speech"
        case .generateMusic: "Generate Music"
        case .generateVideo: "Generate Video"
        case .computer: "Computer"
        }
    }

    /// Tools the chat's Tools menu can switch off.
    ///
    /// Everything except `searchDocuments`, whose real gate is whether a
    /// document folder is attached — stronger than any toggle, and the reason a
    /// docs-only chat works with Tools off. Offering a switch that the resolver
    /// then ignores would be a lying control, so it is absent from both.
    /// `SessionToolDisableTests` pins the two lists against each other.
    static var chatToggleable: [AgentToolKind] {
        allCases.filter { $0 != .searchDocuments }
    }
}

/// How the Tools menu groups its rows.
///
/// Eighteen flat checkmarks is a list nobody reads; grouped, the menu answers
/// "can this chat touch my files / run commands / reach the web" at a glance.
/// Every toggleable tool belongs to exactly one group — a tool in none is
/// unreachable from the UI, one in two renders twice with two checkmarks that
/// look independent. Pinned by `SessionToolDisableTests`.
enum AgentToolGroup: String, CaseIterable, Sendable {
    case files
    case shell
    case web
    case media
    case knowledge
    case computer

    var title: String {
        switch self {
        case .files: "Files"
        case .shell: "Shell & Processes"
        case .web: "Web"
        case .media: "Media"
        case .knowledge: "Memory & Tasks"
        case .computer: "Computer"
        }
    }

    var tools: [AgentToolKind] {
        switch self {
        case .files: [.readFile, .writeFile, .editFile, .searchFiles, .listFiles, .cwd]
        case .shell: [.shell, .listProcesses, .readProcessOutput, .killProcess]
        case .web: [.browse, .webSearch]
        case .media: [.generateImage, .generateSpeech, .generateMusic, .generateVideo]
        case .knowledge: [.saveMemory, .createTask]
        case .computer: [.computer]
        }
    }

    /// The Tools menu's "Computer only" preset: every toggleable tool outside
    /// the Computer and Shell groups, as the session's disabled set. A 3B
    /// model given the full toolbox drifts to host tools (`webSearch`,
    /// `readFile`) mid-task; with these two groups it finishes (live
    /// 2026-09-05, LFM2.5-VL-3B). Pure, so the menu and the resolver agree.
    static var computerOnlyDisabledSet: Set<AgentToolKind> {
        let kept = Set(AgentToolGroup.computer.tools + AgentToolGroup.shell.tools)
        return Set(AgentToolKind.chatToggleable).subtracting(kept)
    }
}

/// What opening the Desktop view does to the chat beside the screen: Tools
/// ON, MCP OFF, the Computer-only preset. Live 2026-09-06 a 9B with Tools
/// off and an MCP shell curl'd the web from the Mac instead of driving the
/// sandbox, because `computer` was never advertised. Applied ONCE on entry
/// (`AppState.showDesktop`), so the Tools menu can still change it after;
/// a session already on the preset is untouched. Pure.
enum DesktopChatPreset {
    struct Change: Equatable {
        var mode: ChatMode
        var useMCP: Bool
        var disabledTools: [String]
    }

    static func apply(mode: ChatMode, useMCP: Bool, disabledTools: [String]) -> Change? {
        let wanted = Change(mode: .agent, useMCP: false,
                            disabledTools: AgentToolGroup.computerOnlyDisabledSet.map(\.rawValue).sorted())
        let current = Change(mode: mode, useMCP: useMCP, disabledTools: disabledTools.sorted())
        return current == wanted ? nil : wanted
    }
}

struct PlanStep: Identifiable, Codable, Equatable {
    let id: UUID
    var tool: AgentToolKind
    var description: String
    var parameters: [String: String]

    init(tool: AgentToolKind, description: String, parameters: [String: String]) {
        self.id = UUID()
        self.tool = tool
        self.description = description
        self.parameters = parameters
    }
}

enum PlanStatus: String, Codable {
    case pending
    case approved
    case rejected
    case executing
    case completed
    case failed
}

struct AgentPlan: Identifiable, Codable, Equatable {
    let id: UUID
    var steps: [PlanStep]
    var status: PlanStatus

    init(steps: [PlanStep]) {
        self.id = UUID()
        self.steps = steps
        self.status = .pending
    }
}

enum StepStatus: String, Codable {
    case pending
    case running
    case success
    case failed
}

struct StepResult: Identifiable, Codable, Equatable {
    let id: UUID
    let stepId: UUID
    var status: StepStatus
    var output: String
    var error: String?
    var durationMs: Int64

    init(stepId: UUID, status: StepStatus, output: String, error: String? = nil, durationMs: Int64 = 0) {
        self.id = UUID()
        self.stepId = stepId
        self.status = status
        self.output = output
        self.error = error
        self.durationMs = durationMs
    }
}
