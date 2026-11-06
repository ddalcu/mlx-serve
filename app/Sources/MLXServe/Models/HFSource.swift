import Foundation

/// A Hugging Face repo at one revision (a branch, tag or commit). Every
/// Hugging Face URL the app builds comes from here, so the revision reaches
/// each call. Mirrors `cli.zig`'s `resolveShortName` + `localRepoId`.
struct HFSource: Hashable {
    var repo: String
    var revision: String = "main"

    var isMain: Bool { revision == "main" }

    private static let base = "https://huggingface.co"

    /// `org/repo`, `org/repo@<rev>` (Hugging Face's own spelling; a `:quant`
    /// after it is the CLI's and is dropped), the repo page, or a
    /// `…/tree/<rev>` URL; `%XX` in the revision decoded. nil for anything
    /// that names no repo.
    static func parse(_ text: String) -> HFSource? {
        var rest = Substring(text.trimmingCharacters(in: .whitespacesAndNewlines))
        if let cut = rest.firstIndex(where: { $0 == "?" || $0 == "#" }) { rest = rest[..<cut] }
        for prefix in ["https://", "http://"] where rest.lowercased().hasPrefix(prefix) {
            rest = rest.dropFirst(prefix.count)
            guard let host = ["www.huggingface.co/", "huggingface.co/", "hf.co/"].first(where: { rest.lowercased().hasPrefix($0) }) else { return nil }
            rest = rest.dropFirst(host.count)
        }
        if let host = ["huggingface.co/", "hf.co/"].first(where: { rest.lowercased().hasPrefix($0) }) { rest = rest.dropFirst(host.count) }
        // Repo names cannot hold `@` and refnames cannot hold `:`.
        var revision: Substring?
        if let at = rest.firstIndex(of: "@") {
            var rev = rest[rest.index(after: at)...]
            if let colon = rev.lastIndex(of: ":") { rev = rev[..<colon] }
            guard !rev.isEmpty else { return nil }
            revision = rev
            rest = rest[..<at]
        }
        let parts = rest.split(separator: "/").map(String.init)
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
        guard parts.count >= 2, parts[0...1].allSatisfy({ $0.unicodeScalars.allSatisfy(allowed.contains) }) else { return nil }
        if revision == nil, parts.count >= 4, parts[2] == "tree" { revision = Substring(parts[3]) }
        guard let revision else { return HFSource(repo: "\(parts[0])/\(parts[1])") }
        return HFSource(repo: "\(parts[0])/\(parts[1])", revision: revision.removingPercentEncoding ?? String(revision))
    }

    /// Where a download lands, as a repo-shaped id: main keeps `org/name`, a
    /// branch gets `org/name-<branch>` with anything but `[A-Za-z0-9._-]` as `-`
    /// (a `/`, `@` or `:` in a model id breaks the registry's id grammar).
    var localRepoId: String {
        guard !isMain else { return repo }
        let safe = String(revision.unicodeScalars.map { Self.folderSafe.contains($0) ? Character($0) : "-" })
        return "\(repo)-\(safe)"
    }

    var treeURL: URL { URL(string: "\(Self.base)/api/models/\(Self.path(repo))/tree/\(encodedRevision)?recursive=true")! }

    func fileURL(_ path: String) -> URL {
        URL(string: "\(Self.base)/\(Self.path(repo))/resolve/\(encodedRevision)/\(Self.path(path))")!
    }

    /// The repo page, or the branch's page off main.
    var pageURL: URL {
        URL(string: "\(Self.base)/\(Self.path(repo))" + (isMain ? "" : "/tree/\(encodedRevision)"))!
    }

    static func refsURL(repo: String) -> URL { URL(string: "\(base)/api/models/\(path(repo))/refs")! }

    /// Segment-wise encoding, so a local folder name with a space still makes a URL.
    private static func path(_ p: String) -> String {
        p.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? p
    }

    /// One path segment: `feature/x` is `feature%2Fx`.
    private var encodedRevision: String {
        revision.addingPercentEncoding(withAllowedCharacters: Self.unreserved) ?? revision
    }

    private static let unreserved = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
    private static let folderSafe = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._")
}

/// One branch of a repo, priced by what `DownloadManager.download` fetches from it.
struct HFBranch: Identifiable, Equatable {
    let name: String
    let commit: String?
    let bytes: Int64
    var id: String { name }

    /// The branch as the picker offers it; nil when it holds no weights.
    static func priced(name: String, commit: String?, entries: [[String: Any]]) -> HFBranch? {
        let files = DownloadManager.selectNeededFiles(from: entries)
        guard files.contains(where: { $0.0.hasSuffix(".safetensors") }) else { return nil }
        return HFBranch(name: name, commit: commit, bytes: files.reduce(Int64(0)) { $0 + $1.1 })
    }

    /// `GET /api/models/<repo>/refs` as (branch, commit) pairs.
    static func refs(from data: Data) -> [(name: String, commit: String?)] {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let branches = root["branches"] as? [[String: Any]] else { return [] }
        return branches.compactMap { b in (b["name"] as? String).map { ($0, b["targetCommit"] as? String) } }
    }

    /// main first (it holds weights, or it would not be here), then by name.
    static func ordered(_ branches: [HFBranch]) -> [HFBranch] {
        branches.sorted { ($0.name == "main" ? 0 : 1, $0.name) < ($1.name == "main" ? 0 : 1, $1.name) }
    }

    /// The requested branch when offered, else the first.
    static func preselect(_ offered: [HFBranch], requested: String?) -> String? {
        offered.first { $0.name == requested }?.name ?? offered.first?.name
    }

    /// A repo with only main looks as it always did: no picker.
    static func needsPicker(_ offered: [HFBranch]) -> Bool {
        !offered.isEmpty && offered.map(\.name) != ["main"]
    }
}
