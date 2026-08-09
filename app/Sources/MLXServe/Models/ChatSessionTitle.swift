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
    /// An agent thread is named for its AGENT. The Agents section is a list of
    /// who you talk to, so the row has to say who — it used to say so only in a
    /// caption under a title derived from the first thing you happened to type,
    /// which put the answer on the second line of every row and the wrong thing
    /// on the first. Read LIVE from the agent rather than copied into the title
    /// at creation, so renaming an agent renames its threads and nothing on disk
    /// has to be rewritten.
    ///
    /// Everything else is unchanged: a placeholder is normalized to the one
    /// matching the thread's KIND, so a thread created before agents had their
    /// own section — stored as "New Chat" with an agent attached — still reads
    /// as an agent thread. Stateless, so it cannot corrupt a title the way a
    /// load-time migration could, and it self-corrects in both directions if a
    /// thread's agent is set or cleared.
    static func display(title: String, agentName: String?) -> String {
        if let named = trimmedName(agentName) { return named }
        return isPlaceholder(title) ? placeholder(hasAgent: agentName != nil) : title
    }

    /// The caption under an agent thread's name: what that conversation is
    /// ABOUT.
    ///
    /// It is the thread's own title, displaced from the title line by the
    /// agent's name — and it is what tells a second thread with the same agent
    /// apart from the first, which the sidebar could otherwise only show as two
    /// identical rows. Nil for a plain conversation (its title is already on the
    /// title line) and nil for a thread that hasn't said anything yet, since a
    /// caption repeating the placeholder says nothing twice.
    static func subject(title: String, agentName: String?) -> String? {
        guard trimmedName(agentName) != nil, !isPlaceholder(title) else { return nil }
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    /// An agent whose name is blank (half-saved, or mid-rename) can't name a
    /// row — the caller falls back to the placeholder rather than drawing an
    /// empty one.
    private static func trimmedName(_ name: String?) -> String? {
        guard let name else { return nil }
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
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
