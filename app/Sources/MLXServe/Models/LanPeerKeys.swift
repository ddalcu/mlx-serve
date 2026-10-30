import Foundation

/// `~/.mlx-serve/lan-keys.json`: the API key each LAN peer running with
/// `--api-key` requires, by peer name. The SERVER reads it on every discovery
/// fetch and proxied request (`src/host/lan/lan.zig`), so a save applies with no restart.
enum LanPeerKeys {
    static let defaultPath = NSString(string: "~/.mlx-serve/lan-keys.json").expandingTildeInPath

    static func load(path: String = defaultPath) -> [String: String] {
        guard let data = FileManager.default.contents(atPath: path) else { return [:] }
        return (try? JSONDecoder().decode([String: String].self, from: data)) ?? [:]
    }

    /// A blank key is a removal.
    static func save(_ keys: [String: String], path: String = defaultPath) throws {
        let kept = keys.mapValues { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.value.isEmpty }
        let dir = (path as NSString).deletingLastPathComponent
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(kept).write(to: URL(fileURLWithPath: path), options: .atomic)
    }

    /// Settings rows: every peer asking for a key, plus every peer we hold one for.
    static func rows(needingKey: [String], saved: [String: String]) -> [String] {
        Set(needingKey).union(saved.keys).sorted()
    }
}
