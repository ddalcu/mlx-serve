import Foundation

/// The community plugin catalog (`plugins.json` in github.com/CommunityServe/plugins) and the pure rules around
/// it: what the entries look like, which are installed or outdated, which this server can run, and which names
/// are safe to hand to a build script. No network and no process here; `PluginManager` does those.
enum PluginCatalog {
    static let repoURL = "https://github.com/CommunityServe/plugins.git"
    static let ref = "main"
    static let catalogURL = URL(string: "https://raw.githubusercontent.com/CommunityServe/plugins/\(ref)/plugins.json")!

    struct Entry: Decodable, Equatable, Identifiable {
        var name: String
        var kind: String
        var version: String
        var summary: String
        var models: [String]
        var requiresMlxServe: String

        var id: String { name }

        private enum CodingKeys: String, CodingKey {
            case name, kind, version, summary, models
            case requiresMlxServe = "requires_mlx_serve"
        }

        init(name: String, kind: String, version: String, summary: String = "", models: [String] = [], requiresMlxServe: String = "") {
            self.name = name; self.kind = kind; self.version = version
            self.summary = summary; self.models = models; self.requiresMlxServe = requiresMlxServe
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.init(name: try c.decode(String.self, forKey: .name),
                      kind: try c.decode(String.self, forKey: .kind),
                      version: try c.decodeIfPresent(String.self, forKey: .version) ?? "",
                      summary: try c.decodeIfPresent(String.self, forKey: .summary) ?? "",
                      models: try c.decodeIfPresent([String].self, forKey: .models) ?? [],
                      requiresMlxServe: try c.decodeIfPresent(String.self, forKey: .requiresMlxServe) ?? "")
        }
    }

    private struct Document: Decodable { var plugins: [Entry] }

    /// The entries of a catalog file. An entry whose name is not a plain identifier is dropped: the name becomes a
    /// folder in the checkout and an argument to a shell script, so the catalog is untrusted input.
    static func decode(_ data: Data) throws -> [Entry] {
        try JSONDecoder().decode(Document.self, from: data).plugins.filter { isSafeName($0.name) }
    }

    /// `[a-z0-9_]{1,64}`, the shape every plugin folder in the repo has.
    static func isSafeName(_ name: String) -> Bool {
        !name.isEmpty && name.utf8.count <= 64
            && name.utf8.allSatisfy { ($0 >= 97 && $0 <= 122) || ($0 >= 48 && $0 <= 57) || $0 == 95 }
    }

    enum Status: Equatable {
        case notInstalled
        case installed
        case updateAvailable(installed: String)
    }

    static func status(of entry: Entry, receipts: [OutsidePlugins.Receipt]) -> Status {
        guard let r = receipts.first(where: { $0.name == entry.name }) else { return .notInstalled }
        return compare(r.version, entry.version) == .orderedAscending ? .updateAvailable(installed: r.version) : .installed
    }

    /// Numeric dot-separated compare (`26.11.1` vs `26.9`); a missing part is 0, a non-number part is 0.
    static func compare(_ a: String, _ b: String) -> ComparisonResult {
        func parts(_ s: String) -> [Int] { s.split(separator: ".").map { Int($0.filter(\.isNumber)) ?? 0 } }
        let (x, y) = (parts(a), parts(b))
        for i in 0..<max(x.count, y.count) {
            let (p, q) = (i < x.count ? x[i] : 0, i < y.count ? y[i] : 0)
            if p != q { return p < q ? .orderedAscending : .orderedDescending }
        }
        return .orderedSame
    }

    /// Whether a running mlx-serve satisfies an entry's `requires_mlx_serve` (`>=26.11.1`; empty or unknown form
    /// says yes, because the server itself refuses a plugin built against another MLX).
    static func runs(on running: String?, requirement: String) -> Bool {
        guard let running, requirement.hasPrefix(">=") else { return true }
        return compare(running, String(requirement.dropFirst(2)).trimmingCharacters(in: .whitespaces)) != .orderedAscending
    }

    /// The mlx-serve version line of `mlx-serve --version`, if any.
    static func runningVersion(fromVersionOutput text: String) -> String? {
        EngineVersions.parse(text).first { $0.name == "mlx-serve" }?.version
    }

    /// Removes an installed plugin's dylib and receipt. Plugins load at startup, so it takes effect on restart.
    static func uninstall(_ name: String, inDir dir: String = OutsidePlugins.defaultDir) throws {
        guard isSafeName(name) else { throw CocoaError(.fileWriteInvalidFileName) }
        for ext in ["dylib", "json"] {
            let path = (dir as NSString).appendingPathComponent("\(name).\(ext)")
            if FileManager.default.fileExists(atPath: path) { try FileManager.default.removeItem(atPath: path) }
        }
    }
}
