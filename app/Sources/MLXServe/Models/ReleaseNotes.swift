import Foundation

/// The running release's notes for the welcome's "What's new" panel, read from
/// its GitHub release: the release body is that version's CHANGELOG entry.
struct ReleaseNotes: Equatable {
    /// Normalized, no "v" prefix.
    let version: String
    /// Inline markdown, one line each (`items(fromMarkdown:)`).
    let items: [String]
    let page: URL?

    static func url(forVersion version: String) -> URL {
        URL(string: "https://api.github.com/repos/\(UpdateChecker.repo)/releases/tags/v\(version)")!
    }

    private struct Release: Decodable {
        let tag_name: String
        let html_url: String?
        let body: String?
    }

    static func parse(releaseJSON data: Data) -> ReleaseNotes? {
        guard let release = try? JSONDecoder().decode(Release.self, from: data) else { return nil }
        let tag = release.tag_name
        return ReleaseNotes(version: tag.hasPrefix("v") ? String(tag.dropFirst()) : tag,
                            items: items(fromMarkdown: release.body ?? ""),
                            page: release.html_url.flatMap(URL.init(string:)))
    }

    /// The body's bullets (marker stripped) and plain lines; headings, HTML
    /// (pasted screenshots), rules and blank lines are dropped.
    static func items(fromMarkdown body: String) -> [String] {
        body.split(whereSeparator: \.isNewline).compactMap { raw in
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") || line.hasPrefix("<") || line.allSatisfy({ $0 == "-" }) { return nil }
            for marker in ["- ", "* "] where line.hasPrefix(marker) {
                return String(line.dropFirst(marker.count)).trimmingCharacters(in: .whitespaces)
            }
            return line
        }
    }

    /// This version's release, else the latest one (a build whose tag GitHub
    /// doesn't have); nil when offline.
    static func fetch(version: String) async -> ReleaseNotes? {
        for url in [url(forVersion: version), UpdateChecker.latestReleaseURL] {
            var request = URLRequest(url: url)
            request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
            guard let (data, response) = try? await URLSession.shared.data(for: request),
                  (response as? HTTPURLResponse)?.statusCode == 200,
                  let notes = parse(releaseJSON: data) else { continue }
            return notes
        }
        return nil
    }
}
