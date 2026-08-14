import Foundation

/// Forking a conversation at a message: the conversation up to that point, in
/// a new thread, with the original left exactly as it was.
///
/// It is the third answer to "this reply isn't what I need", beside the two
/// that already exist. Regenerate throws the reply away (and keeps its earlier
/// versions on the one message); Continue extends it. A fork keeps BOTH
/// branches as real conversations you can come back to — which is what you
/// want the moment the disagreement is about the question, not the answer.
enum ChatFork {

    /// The transcript a fork starts with: everything through `messageId`,
    /// trimmed back to a boundary the model can be handed.
    static func prefix(_ messages: [ChatMessage], through messageId: UUID) -> [ChatMessage] {
        guard let cut = messages.firstIndex(where: { $0.id == messageId }) else { return [] }
        var out = Array(messages[...cut])
        while let last = out.last, !isBoundary(last) { out.removeLast() }
        return out
    }

    /// Whether the command is worth offering here. A cut with nothing left
    /// after trimming would create an empty chat, and a menu item that does
    /// nothing when you pick it is the dead-control class.
    static func isForkable(_ messages: [ChatMessage], at messageId: UUID) -> Bool {
        !prefix(messages, through: messageId).isEmpty
    }

    /// The new conversation, carrying the settings that produced the transcript
    /// it inherits.
    ///
    /// Everything that decides how a turn RUNS comes across — agent, tools
    /// mode, thinking, effort, MCP, the per-chat tool switches, the working
    /// directory — or the fork's next turn answers under different settings
    /// than the turns above it in its own transcript.
    ///
    /// Three things deliberately do not. The attached folder's
    /// security-scoped bookmark is keyed by the SOURCE session's id
    /// (`SecurityScopedBookmark.attachedFolderName`), so the path would arrive
    /// without the grant that makes it readable — a folder chip pointing at
    /// something the sandbox refuses to open. And `taskRunId` / isExternalBridge
    /// mark a session as transient: hidden from the sidebar and never persisted,
    /// so a fork that inherited either would vanish on the next launch.
    ///
    /// Message ids are kept rather than regenerated. Every lookup in the app is
    /// `(sessionId, messageId)`, so two sessions holding the same message id
    /// cannot collide — and a fork IS those messages.
    static func session(from source: ChatSession, messages: [ChatMessage]) -> ChatSession {
        // The fork is that conversation up to the cut, so it keeps its name —
        // and a source still on its placeholder hands over a placeholder, which
        // lets the auto-titler name the fork from its own content rather than
        // pinning "New Chat" on it forever (`ChatSessionTitle`).
        var fork = ChatSession(title: source.title)
        fork.messages = messages
        fork.mode = source.mode
        fork.workingDirectory = source.workingDirectory
        fork.enableThinking = source.enableThinking
        fork.reasoningEffort = source.reasoningEffort
        fork.useMCP = source.useMCP
        fork.agentId = source.agentId
        fork.disabledTools = source.disabledTools
        return fork
    }

    /// Whether the transcript can end here.
    ///
    /// A tool call's results arrive AFTER it, so cutting on the caller — or on
    /// one of the results, with more still to come — hands the model a call
    /// with no answer, which is the shape it apologises for or re-issues. Our
    /// own error cards and tool-call summaries are machinery rather than
    /// something the model said, so they are no better a place to resume from.
    private static func isBoundary(_ message: ChatMessage) -> Bool {
        switch message.role {
        case .user:
            return true
        case .system:
            // Hidden tool results are `.system` carrying a `toolCallId`; a
            // system message is not a turn to resume after either way.
            return false
        case .assistant:
            if let calls = message.toolCalls, !calls.isEmpty { return false }
            return !message.isAgentSummary && !message.failedRetry && message.errorNotice == nil
        }
    }
}
