import Foundation

struct AgentMemoryData: Codable {
    var recentDirectories: [String] = []
    var commonCommands: [String] = []
    var projectDescriptions: [String: String] = [:]
    var preferences: [String: String] = [:]
}

@MainActor
class AgentMemory: ObservableObject {
    @Published var isEnabled: Bool {
        didSet {
            UserDefaults.standard.set(isEnabled, forKey: "agentMemoryEnabled")
            if isEnabled { load() }
        }
    }

    private(set) var data = AgentMemoryData()

    private let filePath: String = {
        let dir = NSString(string: "~/.mlx-serve").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return (dir as NSString).appendingPathComponent("agent-memory.json")
    }()

    init() {
        let saved = UserDefaults.standard.bool(forKey: "agentMemoryEnabled")
        _isEnabled = Published(wrappedValue: saved)
        if saved { load() }
    }

    func recordDirectory(_ dir: String) {
        guard isEnabled else { return }
        data.recentDirectories.removeAll { $0 == dir }
        data.recentDirectories.insert(dir, at: 0)
        data.recentDirectories = Array(data.recentDirectories.prefix(10))
        save()
    }

    func recordCommand(_ cmd: String) {
        guard isEnabled else { return }
        data.commonCommands.removeAll { $0 == cmd }
        data.commonCommands.insert(cmd, at: 0)
        data.commonCommands = Array(data.commonCommands.prefix(20))
        save()
    }

    func contextSnippet() -> String {
        guard isEnabled else { return "" }
        var parts: [String] = []
        if !data.recentDirectories.isEmpty {
            parts.append("Recent dirs: \(data.recentDirectories.prefix(3).joined(separator: ", "))")
        }
        if !data.commonCommands.isEmpty {
            parts.append("Common cmds: \(data.commonCommands.prefix(5).joined(separator: ", "))")
        }
        return parts.isEmpty ? "" : "\n[Memory]\n" + parts.joined(separator: "\n")
    }

    private func save() {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        guard let encoded = try? encoder.encode(data) else { return }
        try? encoded.write(to: URL(fileURLWithPath: filePath))
    }

    func load() {
        guard FileManager.default.fileExists(atPath: filePath),
              let raw = try? Data(contentsOf: URL(fileURLWithPath: filePath)) else { return }
        data = (try? JSONDecoder().decode(AgentMemoryData.self, from: raw)) ?? AgentMemoryData()
    }
}
