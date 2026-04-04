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
        self.workingDirectory = nil
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
        workingDirectory = try c.decodeIfPresent(String.self, forKey: .workingDirectory)
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
    }
}

struct ModelInfo {
    var name: String
    var quantBits: Int
    var layers: Int
    var hiddenSize: Int
    var vocabSize: Int
    var contextLength: Int
}

struct MemoryInfo {
    var activeBytes: Int64
    var peakBytes: Int64

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

struct LocalModel: Identifiable, Hashable {
    let id: String
    let name: String
    let path: String
    let sizeFormatted: String
}

struct GemmaModelOption: Identifiable {
    let id: String
    let displayName: String
    let repoId: String
    let sizeEstimate: String
}

let gemmaModelOptions: [GemmaModelOption] = [
    GemmaModelOption(id: "e2b-4bit", displayName: "Gemma 4 E2B (4-bit)", repoId: "mlx-community/gemma-4-e2b-it-4bit", sizeEstimate: "~3.4 GB"),
    GemmaModelOption(id: "e2b-8bit", displayName: "Gemma 4 E2B (8-bit)", repoId: "mlx-community/gemma-4-e2b-it-8bit", sizeEstimate: "~5.5 GB"),
    GemmaModelOption(id: "e4b-4bit", displayName: "Gemma 4 E4B (4-bit)", repoId: "mlx-community/gemma-4-e4b-it-4bit", sizeEstimate: "~5.2 GB"),
    GemmaModelOption(id: "e4b-8bit", displayName: "Gemma 4 E4B (8-bit)", repoId: "mlx-community/gemma-4-e4b-it-8bit", sizeEstimate: "~8.5 GB"),
]
