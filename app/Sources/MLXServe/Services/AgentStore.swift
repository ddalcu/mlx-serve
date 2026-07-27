import Foundation

/// Saved agents, persisted the way the iPhone app does it: one
/// `~/.mlx-serve/agents/index.json`, held whole in memory, written atomically.
/// Best-effort persistence with memory authoritative — a failed write must never
/// lose the agent the user just made from the list they're looking at.
///
/// `ObservableObject` rather than `@Observable` to match every other service in
/// this app (AppState observes it with `@Published`).
@MainActor
final class AgentStore: ObservableObject {

    /// The user's own agents (starters are constants — see `allAgents`).
    @Published private(set) var agents: [Agent] = []

    private let rootDir: URL
    private var indexURL: URL { rootDir.appendingPathComponent("index.json") }

    /// Default store under `~/.mlx-serve/agents`; tests inject a temp dir.
    init(rootDir: URL? = nil) {
        self.rootDir = rootDir ?? URL(fileURLWithPath: NSString(string: "~/.mlx-serve/agents").expandingTildeInPath,
                                      isDirectory: true)
        load()
    }

    /// Newest first — the one you just made is the one you want.
    var sortedAgents: [Agent] { agents.sorted { $0.createdAt > $1.createdAt } }

    /// Everything a picker shows: the user's agents, then the read-only starters.
    var allAgents: [Agent] { sortedAgents + Agent.starters }

    /// Turn sites only carry an id, so this has to find a starter too.
    func agent(id: UUID?) -> Agent? {
        guard let id else { return nil }
        return agents.first { $0.id == id } ?? Agent.starters.first { $0.id == id }
    }

    func add(_ agent: Agent) {
        agents.append(agent)
        persist()
    }

    /// No-op for a starter — they're code constants; "Duplicate" is the way in.
    func update(_ agent: Agent) {
        guard !agent.isBuiltIn, let idx = agents.firstIndex(where: { $0.id == agent.id }) else { return }
        agents[idx] = agent
        persist()
    }

    func delete(id: UUID) {
        agents.removeAll { $0.id == id }
        persist()
    }

    /// An editable copy — the only way to change a starter.
    @discardableResult
    func duplicate(_ agent: Agent) -> Agent {
        var copy = agent
        copy.id = UUID()
        copy.isBuiltIn = false
        copy.createdAt = Date()
        copy.name = "\(agent.name) Copy"
        // A wake phrase is a global gate; a copy sharing one would make both
        // unreachable by voice (see `WakeWord.collides`).
        copy.wakePhrase = nil
        add(copy)
        return copy
    }

    /// Wake phrases already taken, excluding one agent (the one being edited).
    func takenWakePhrases(excluding id: UUID?) -> [String] {
        allAgents
            .filter { $0.id != id }
            .compactMap { $0.wakePhrase }
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
    }

    /// Every agent's spoken gate, for the voice controller's multi-phrase match.
    var wakePhrases: [(id: UUID, phrase: String)] {
        allAgents.compactMap { a in
            guard let raw = a.wakePhrase, let norm = WakeWord.normalizePhrase(raw) else { return nil }
            return (a.id, norm)
        }
    }

    // MARK: - Persistence

    private func load() {
        guard let data = try? Data(contentsOf: indexURL),
              let decoded = try? Self.decoder.decode([Agent].self, from: data) else {
            agents = []
            return
        }
        // A starter's id can never be shadowed by a stored row.
        let starterIds = Set(Agent.starters.map(\.id))
        agents = decoded.filter { !$0.isBuiltIn && !starterIds.contains($0.id) }
    }

    private func persist() {
        do {
            try FileManager.default.createDirectory(at: rootDir, withIntermediateDirectories: true)
            try Self.encoder.encode(agents).write(to: indexURL, options: .atomic)
        } catch {
            // Best-effort, like ChatStore: memory stays authoritative.
        }
    }

    private static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        return e
    }()

    private static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()
}
