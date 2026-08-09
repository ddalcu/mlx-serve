import Foundation

/// Which single row of the sidebar is lit.
enum SidebarSelection {

    static func isConversationSelected(sessionId: UUID,
                                       activeChatId: UUID?,
                                       workspace: ChatWorkspace) -> Bool {
        workspace.isConversation && sessionId == activeChatId
    }
}
