import Foundation

/// Which single row of the sidebar is lit.
///
/// The panel has two halves that decide "selected" from different state — a
/// destination reads `chatWorkspace`, a conversation reads `activeChatId` — and
/// nothing reconciled them, so opening Tasks lit the Tasks destination while
/// the last chat stayed lit below it. Two "you are here" marks for one window,
/// and each half looked correct on its own.
///
/// Pure, because the rule is one sentence and the alternative is discovering it
/// by clicking around: the active chat is only where you ARE while the window
/// is actually showing conversations.
enum SidebarSelection {

    static func isConversationSelected(sessionId: UUID,
                                       activeChatId: UUID?,
                                       workspace: ChatWorkspace) -> Bool {
        workspace.isConversation && sessionId == activeChatId
    }
}
