import Foundation

/// The pure half of the model detail sheet: where a repo's page and README
/// live, how a local folder name maps back to a repo id, and the README
/// front-matter strip. Kept out of the view so each is testable.
enum ModelCard {
    static func pageURL(repoId: String, revision: String = "main") -> URL? {
        HFSource(repo: repoId, revision: revision).pageURL
    }

    static func readmeURL(repoId: String, revision: String = "main") -> URL? {
        HFSource(repo: repoId, revision: revision).fileURL("README.md")
    }

    /// `org/repo` as-is; MTPLX's flat `Org--Name` folded back; a bare folder
    /// name is not a Hugging Face repo, so nil.
    static func repoId(localName: String) -> String? {
        if localName.contains("/") { return localName }
        guard let range = localName.range(of: "--") else { return nil }
        let org = localName[..<range.lowerBound]
        let repo = localName[range.upperBound...]
        guard !org.isEmpty, !repo.isEmpty else { return nil }
        return "\(org)/\(repo)"
    }

    /// Drops a leading `---\n…\n---` YAML block. Anything else is returned
    /// untouched, including an unterminated block.
    static func stripFrontMatter(_ readme: String) -> String {
        guard readme.hasPrefix("---\n") else { return readme }
        let afterOpen = readme.index(readme.startIndex, offsetBy: 4)
        guard let close = readme.range(of: "\n---\n", range: afterOpen..<readme.endIndex) else { return readme }
        return String(readme[close.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// What a row asks the sheet to show. `Identifiable` so rows can present it
/// with `.sheet(item:)`.
struct ModelCardRequest: Identifiable, Hashable {
    let repoId: String
    let title: String
    /// The branch whose card opens first (a pasted `/tree/<branch>` URL, a branch download).
    var revision: String? = nil
    /// Offers the repo's branches with their sizes and a download for the one picked.
    var picksBranch = false
    var id: String { repoId }
}
