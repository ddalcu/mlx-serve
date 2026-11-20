import Foundation

/// One template video, with what made it, so it can be opened again as a template.
struct TemplateHistoryEntry: Codable, Identifiable, Equatable {
    var id = UUID()
    var template: String
    var createdAt = Date()
    var clip: String
    var captioned: String?
    var gif: String?
    /// Slot → cast member id; a member deleted since is left out on restore.
    var cast: [String: UUID]
    var answers: [String: String]
    var lines: [String: String]
    var shape: String
    var seed: Int

    var videoTemplate: MediaTemplate? { MediaTemplate(rawValue: template) }
    var shareShape: ShareShape { ShareShape(rawValue: shape) ?? .story }
    /// The file to play: the captioned one when there is one.
    var playURL: URL { URL(fileURLWithPath: captioned ?? clip) }
}

/// Template videos made on this Mac, newest first:
/// `~/.mlx-serve/templates/history.json`. The videos themselves stay where the
/// generator wrote them; an entry whose clip is gone is not listed.
@MainActor
final class TemplateHistory: ObservableObject {
    @Published private(set) var entries: [TemplateHistoryEntry] = []

    private let rootDir: URL
    private var indexURL: URL { rootDir.appendingPathComponent("history.json") }

    init(rootDir: URL? = nil) {
        self.rootDir = rootDir ?? URL(fileURLWithPath: NSString(string: "~/.mlx-serve/templates").expandingTildeInPath,
                                      isDirectory: true)
        guard let data = try? Data(contentsOf: indexURL),
              let decoded = try? Self.decoder.decode([TemplateHistoryEntry].self, from: data) else { return }
        let fm = FileManager.default
        entries = decoded.filter { fm.fileExists(atPath: $0.clip) }.map { e in
            var e = e
            if let c = e.captioned, !fm.fileExists(atPath: c) { e.captioned = nil }
            if let g = e.gif, !fm.fileExists(atPath: g) { e.gif = nil }
            return e
        }
    }

    func add(_ entry: TemplateHistoryEntry) {
        entries.insert(entry, at: 0)
        persist()
    }

    func setCaptioned(_ id: UUID, _ path: String) {
        guard let i = entries.firstIndex(where: { $0.id == id }) else { return }
        entries[i].captioned = path
        persist()
    }

    /// Removes the entry and every file it owns: the clip, its captioned copy,
    /// its GIF and the settings sidecar the generator wrote beside it.
    func delete(_ id: UUID) {
        guard let e = entries.first(where: { $0.id == id }) else { return }
        for path in [e.clip, VideoGenService.sidecarPath(forVideo: e.clip)] + [e.captioned, e.gif].compactMap({ $0 }) {
            try? FileManager.default.removeItem(atPath: path)
        }
        entries.removeAll { $0.id == id }
        persist()
    }

    private func persist() {
        try? FileManager.default.createDirectory(at: rootDir, withIntermediateDirectories: true)
        if let data = try? Self.encoder.encode(entries) { try? data.write(to: indexURL, options: .atomic) }
    }

    private static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        e.dateEncodingStrategy = .iso8601
        return e
    }()

    private static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()
}
