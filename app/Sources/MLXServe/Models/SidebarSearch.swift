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
}
