import Foundation

/// One page a web-search answer drew on.
struct WebSource: Identifiable, Equatable, Hashable {
    let title: String
    let url: String

    /// Host without scheme or `www.`, for the row's right-hand label — a column
    /// of "https://www." prefixes is noise on every row.
    var domain: String {
        guard let host = URLComponents(string: url)?.host?.lowercased() else { return url }
        return host.hasPrefix("www.") ? String(host.dropFirst(4)) : host
    }

    var id: String { url }
}

/// Recovers the pages behind a web-search answer.
///
/// The URLs only exist in the HIDDEN tool-result messages (`role .system` with a
/// `toolCallId`), which the transcript filters out — so without this the user
/// gets a confident answer and no way to see where it came from.
///
/// Everything here is conservative on purpose: a row that isn't provably a
/// result is dropped rather than guessed at, because a fabricated citation is
/// worse than a missing one.
enum WebSourceExtractor {

    /// Parse one `webSearch` payload: a header line, then title / url / snippet
    /// triples separated by blank lines.
    ///
    /// The handler falls back to dumping raw page text when the search returns
    /// nothing, so a block only counts when its second line is an http(s) URL.
    /// That single check is what keeps page prose from being presented as
    /// sources.
    static func sources(fromSearchOutput output: String) -> [WebSource] {
        var out: [WebSource] = []
        for block in output.components(separatedBy: "\n\n") {
            let lines = block.components(separatedBy: "\n")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            guard lines.count >= 2, isWebURL(lines[1]), !lines[0].isEmpty else { continue }
            out.append(WebSource(title: lines[0], url: lines[1]))
        }
        return out
    }

    /// http(s) only — a `javascript:` or `file:` line must never become a
    /// clickable row.
    private static func isWebURL(_ s: String) -> Bool {
        guard let scheme = URLComponents(string: s)?.scheme?.lowercased(),
              URLComponents(string: s)?.host?.isEmpty == false else { return false }
        return scheme == "http" || scheme == "https"
    }

    /// Sources backing the reply with `messageId`: every `webSearch` result in
    /// the SAME turn.
    ///
    /// The walk stops at the previous user message, so an earlier turn's search
    /// can't be cited by a later, unrelated answer. Duplicates are collapsed
    /// keeping first-seen order — two searches routinely surface the same top
    /// hit, and listing it twice makes the count wrong.
    static func sources(forMessageId messageId: UUID, in messages: [ChatMessage]) -> [WebSource] {
        guard let end = messages.firstIndex(where: { $0.id == messageId }) else { return [] }
        var collected: [WebSource] = []
        var seen = Set<String>()
        var i = end - 1
        while i >= 0 {
            let m = messages[i]
            if m.role == .user { break }
            if m.toolName == "webSearch", m.toolCallId != nil {
                // Prepend: the walk is backwards, but the list should read in
                // the order the model found them.
                collected.insert(contentsOf: sources(fromSearchOutput: m.content), at: 0)
            }
            i -= 1
        }
        return collected.filter { seen.insert($0.url).inserted }
    }
}
