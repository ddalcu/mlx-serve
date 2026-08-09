import Foundation

/// What a conversation is called before it has said anything, and what it is
/// called afterwards.
///
/// Both halves live here because they are ONE rule: the auto-titler fires only
/// while a session still carries its placeholder, so a new placeholder that the
/// titler doesn't know about is a thread named "New agent" forever. That is
/// exactly what a second literal would have produced — the gate used to be
/// `title == "New Chat"` spelled out at the call site.
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
    ///
    /// A placeholder is normalized to the one matching the thread's KIND, so a
    /// thread created before agents had their own section — stored as
    /// "New Chat" with an agent attached — reads as "New agent" without
    /// rewriting anything on disk. Stateless, so it cannot corrupt a title the
    /// way a load-time migration could, and it self-corrects in both directions
    /// if a thread's agent is set or cleared. A real title is never touched.
    static func display(title: String, hasAgent: Bool) -> String {
        isPlaceholder(title) ? placeholder(hasAgent: hasAgent) : title
    }

    /// The title a first user message earns the thread.
    ///
    /// Returns nil when the message can't name anything, so the caller keeps the
    /// placeholder rather than showing a blank row.
    static func derived(fromFirstMessage content: String, limit: Int = 40) -> String? {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        // Count in CHARACTERS, and slice the same way — a prefix taken on one
        // measure and tested on another puts an ellipsis on a title that fits.
        guard trimmed.count > limit else { return trimmed }
        return String(trimmed.prefix(limit)) + "..."
    }
}
