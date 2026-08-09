import Foundation

/// What a conversation is called before it has said anything, and what it is
/// called afterwards.
enum ChatSessionTitle {

    /// A plain conversation, waiting for its first message.
    static let plain = "New Chat"
    /// A conversation belonging to an agent. Named for what it IS, so the
    /// sidebar's Agents section reads as a list of agents rather than of chats
    /// that happen to have one.
    static let agent = "New agent"

    /// The placeholder a freshly created session gets.
    static func placeholder(hasAgent: Bool) -> String {
        hasAgent ? agent : plain
    }

    /// Whether this session is still waiting for the content that names it.
    ///
    /// Every placeholder, never one of them: this is the auto-titler's gate.
    static func isPlaceholder(_ title: String) -> Bool {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed == plain || trimmed == agent
    }

    /// What the sidebar actually draws.
    static func display(title: String, agentName: String?) -> String {
        if let named = agentName?.trimmedNonEmpty { return named }
        return isPlaceholder(title) ? placeholder(hasAgent: agentName != nil) : title
    }

    /// The caption under an agent thread's name: what that conversation is
    /// ABOUT.
    static func subject(title: String, agentName: String?) -> String? {
        guard agentName?.trimmedNonEmpty != nil else { return nil }
        return isPlaceholder(title) ? nil : title.trimmedNonEmpty
    }

    /// The title a first user message earns the thread.
    static func derived(fromFirstMessage content: String, limit: Int = 40) -> String? {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        // Count in CHARACTERS, and slice the same way — a prefix taken on one
        // measure and tested on another puts an ellipsis on a title that fits.
        guard trimmed.count > limit else { return trimmed }
        return String(trimmed.prefix(limit)) + "..."
    }
}

extension String {
    /// Trimmed, or nil when that leaves nothing. Blank is not a value: a
    /// half-saved agent name can't title a row and an empty brief can't caption
    /// one, so every such field goes through here.
    var trimmedNonEmpty: String? {
        let t = trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? nil : t
    }
}
