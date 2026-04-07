import Foundation

enum ChatMode: String, Codable {
    case chat
    case agent
}

enum AgentToolKind: String, Codable, CaseIterable {
    case shell
    case readFile
    case writeFile
    case editFile
    case searchFiles
    case browse
    case webSearch
    case saveMemory

    var icon: String {
        switch self {
        case .shell: "terminal"
        case .readFile: "doc.text"
        case .writeFile: "doc.text.fill"
        case .editFile: "pencil"
        case .searchFiles: "magnifyingglass"
        case .browse: "globe"
        case .webSearch: "magnifyingglass"
        case .saveMemory: "brain"
        }
    }

    var displayName: String {
        switch self {
        case .shell: "Shell"
        case .readFile: "Read File"
        case .writeFile: "Write File"
        case .editFile: "Edit File"
        case .searchFiles: "Search Files"
        case .browse: "Browse"
        case .webSearch: "Web Search"
        case .saveMemory: "Save Memory"
        }
    }
}

struct PlanStep: Identifiable, Codable {
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

struct AgentPlan: Identifiable, Codable {
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

struct StepResult: Identifiable, Codable {
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
