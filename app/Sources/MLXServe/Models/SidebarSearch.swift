import Foundation

/// Sidebar conversation search: title OR anything said in the transcript,
/// case- and diacritic-insensitive. Hidden machinery stays out of the index —
/// tool-result rows and failed retries are not what anyone is looking for, and
/// surfacing them would make the sidebar leak internals a search shouldn't see.
enum SidebarSearch {

    static func matches(_ session: ChatSession, query: String) -> Bool {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return true }
        if session.title.range(of: q, options: [.caseInsensitive, .diacriticInsensitive]) != nil {
            return true
        }
        return session.messages.contains { message in
            guard message.toolCallId == nil, !message.failedRetry else { return false }
            return message.content.range(of: q, options: [.caseInsensitive, .diacriticInsensitive]) != nil
        }
    }

    static func filter(sessions: [ChatSession], query: String) -> [ChatSession] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return sessions }
        return sessions.filter { matches($0, query: q) }
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
            guard message.toolCallId == nil, !message.failedRetry else { continue }
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
        // Keep the match centred in the cap rather than clipped by it.
        let matchOffset = content.distance(from: lineStart, to: range.lowerBound)
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
