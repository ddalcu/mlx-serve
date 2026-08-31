import Foundation

/// Sidebar conversation search: title OR anything said in the transcript,
/// case- and diacritic-insensitive.
///
/// Hidden machinery stays out of the index, and "hidden" is a bigger set than
/// the raw tool-result rows. An agent thread also carries SUMMARY rows
/// (`isAgentSummary`) — `**name**(args)` for the call and `**name** → output`
/// for the result — which the transcript folds into a single tool-call row.
/// Indexing those did two wrong things at once: it put raw tool output in a
/// sidebar snippet, and it produced a hit whose message id is not the id of any
/// row on screen, so clicking it released follow and scrolled nowhere.
enum SidebarSearch {

    /// A row is searchable when the transcript draws it as itself.
    private static func isSearchable(_ message: ChatMessage) -> Bool {
        message.toolCallId == nil && !message.failedRetry && !message.isAgentSummary
    }

    /// - Parameter displayTitle: what the SIDEBAR draws. An agent thread's row
    ///   shows the AGENT's name, not `session.title`, so a row reading
    ///   "Code Reviewer" vanished when you typed "code reviewer". Callers pass
    ///   the resolved name; nil falls back to the stored title.
    static func matches(_ session: ChatSession, query: String,
                        displayTitle: String? = nil) -> Bool {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return true }
        for title in [displayTitle, session.title].compactMap({ $0 })
        where title.range(of: q, options: [.caseInsensitive, .diacriticInsensitive]) != nil {
            return true
        }
        return session.messages.contains { message in
            guard isSearchable(message) else { return false }
            return message.content.range(of: q, options: [.caseInsensitive, .diacriticInsensitive]) != nil
        }
    }

    static func filter(sessions: [ChatSession], query: String,
                       displayTitle: (ChatSession) -> String? = { _ in nil }) -> [ChatSession] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return sessions }
        return sessions.filter { matches($0, query: q, displayTitle: displayTitle($0)) }
    }

    /// True when the field holds a real query (drives the "no matches" row).
    static func isFiltering(_ query: String) -> Bool {
        !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    // MARK: - Result context

    /// WHERE a session matched, so the row can say so and a click can jump.
    struct SearchHit: Equatable {
        /// Index into `session.messages` (full array, not the visible rows).
        let messageIndex: Int
        /// The matching line, whitespace-collapsed and capped — one caption's
        /// worth of context under the title.
        let snippet: String
    }

    static let snippetMaxLength = 96

    /// The first transcript row matching `query`, with a one-line snippet.
    /// nil when the query is empty or only the TITLE matched — a title hit has
    /// no message to jump to.
    static func firstContentMatch(in session: ChatSession, query: String) -> SearchHit? {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return nil }
        for (idx, message) in session.messages.enumerated() {
            guard isSearchable(message) else { continue }
            guard let range = message.content.range(of: q, options: [.caseInsensitive, .diacriticInsensitive]) else { continue }
            return SearchHit(messageIndex: idx,
                             snippet: snippet(around: range, in: message.content))
        }
        return nil
    }

    /// The line holding the match, collapsed to a single line and capped at
    /// `snippetMaxLength` around the hit.
    private static func snippet(around range: Range<String.Index>, in content: String) -> String {
        // One line: from the match's line start to its line end.
        let lineStart = content[...range.lowerBound].lastIndex(where: { $0 == "\n" })
            .map { content.index(after: $0) } ?? content.startIndex
        let lineEnd = content[range.upperBound...].firstIndex(of: "\n") ?? content.endIndex
        var line = String(content[lineStart..<lineEnd])
            .trimmingCharacters(in: .whitespaces)
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
        guard line.count > snippetMaxLength else { return line }
        // Centre the match in the cap. The offset MUST be measured on `line` —
        // the trimmed, whitespace-collapsed string the cut is applied to. Taken
        // on the raw content instead, a line with heavy leading or internal
        // whitespace cuts past the match and the tail cap then removes it, so
        // the caption shown does not contain what you searched for.
        let matchText = String(content[range])
        let matchOffset = line.range(of: matchText, options: [.caseInsensitive, .diacriticInsensitive])
            .map { line.distance(from: line.startIndex, to: $0.lowerBound) } ?? 0
        let keep = snippetMaxLength
        if matchOffset > keep / 2 {
            let cut = line.index(line.startIndex, offsetBy: min(matchOffset - keep / 2, line.count - 1))
            line = "…" + String(line[cut...])
        }
        if line.count > keep {
            line = String(line.prefix(keep - 1)) + "…"
        }
        return line
    }
}
