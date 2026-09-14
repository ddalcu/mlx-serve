import Foundation

/// A model's turn as the reader sees it: everything under their message up to
/// the next one. One round of thinking and prose, or a dozen rounds of tool
/// calls with the thinking and the cards between them.
enum ChatTurn {

    /// Whether the transcript can end here.
    ///
    /// A tool call's results arrive AFTER it, so cutting on the caller — or on
    /// one of the results, with more still to come — hands the model a call
    /// with no answer, which is the shape it apologises for or re-issues. Our
    /// own error cards and tool-call summaries are machinery rather than
    /// something the model said, so they are no better a place to resume from.
    static func isBoundary(_ message: ChatMessage) -> Bool {
        switch message.role {
        case .user:
            return true
        case .system:
            // Hidden tool results are `.system` carrying a `toolCallId`; a
            // system message is not a turn to resume after either way.
            return false
        case .assistant:
            if let calls = message.toolCalls, !calls.isEmpty { return false }
            // A round that only thought, or only carried media, is part of a
            // turn, not the end of one.
            return !message.content.isEmpty && !message.isAgentSummary
                && !message.failedRetry && message.errorNotice == nil
        }
    }

    /// Whether a reply row draws a footer of its own.
    static func hasOwnFooter(_ message: ChatMessage) -> Bool {
        message.role == .assistant && !message.isStreaming
            && !message.isAgentSummary && !message.content.isEmpty
    }

    /// Whether a reply's footer carries the trash. Only where the transcript can
    /// be cut: on a boundary, or on the last message, below which nothing is
    /// left to orphan (prose cut before its tools ran).
    static func footerDeletes(_ message: ChatMessage, isLast: Bool) -> Bool {
        isBoundary(message) || isLast
    }

    /// What the trash under the footer of `messageId` removes: that message and
    /// everything above it back to the nearest boundary, which stays. The
    /// reader's own message is a boundary, so a turn never takes the question
    /// with it; a reply is one, so a later segment stops under it.
    static func deletionRange(endingAt messageId: UUID, in messages: [ChatMessage]) -> Range<Int>? {
        guard let end = messages.firstIndex(where: { $0.id == messageId }) else { return nil }
        var start = end
        while start > 0, !isBoundary(messages[start - 1]) { start -= 1 }
        return start..<(end + 1)
    }

    /// Whether the transcript ends on the model's side without a footer: a
    /// turn cut while thinking, after a tool result, on an error card. That
    /// end gets a footer of its own, so it can be deleted or regenerated like
    /// any reply. Never while the turn is still running — its end moves.
    static func needsEndFooter(_ messages: [ChatMessage], turnInFlight: Bool) -> Bool {
        guard !turnInFlight, let last = messages.last, last.role != .user else { return false }
        return !hasOwnFooter(last)
    }
}
