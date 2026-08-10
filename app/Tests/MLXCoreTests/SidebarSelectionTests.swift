import XCTest
@testable import MLXCore

/// Exactly one row of the sidebar is lit at a time.
///
/// The panel's two halves decide "selected" from different state — a
/// destination reads `chatWorkspace`, a conversation reads `activeChatId` — and
/// nothing reconciled them, so opening Tasks lit the Tasks destination while
/// the last chat stayed lit below it. Each half was correct on its own, which
/// is why it survived: there was no single place where the contradiction was
/// visible.
final class SidebarSelectionTests: XCTestCase {

    private let chat = UUID()
    private let otherChat = UUID()

    func testTheActiveChatIsLitWhileViewingConversations() {
        XCTAssertTrue(SidebarSelection.isConversationSelected(
            sessionId: chat, activeChatId: chat, workspace: .conversation))
    }

    func testOnlyTheActiveChatIsLit() {
        XCTAssertFalse(SidebarSelection.isConversationSelected(
            sessionId: otherChat, activeChatId: chat, workspace: .conversation))
    }

    /// The live bug: a chat stayed lit beside the destination you had just
    /// moved to. Every non-conversation mode stands the chat rows down.
    func testNoChatIsLitWhileAPaneIsShowing() {
        for workspace: ChatWorkspace in [.tasks, .agents, .settings,
                                         .models(.recommended), .create(.image)] {
            XCTAssertFalse(
                SidebarSelection.isConversationSelected(
                    sessionId: chat, activeChatId: chat, workspace: workspace),
                "a conversation must not stay lit while \(workspace) is showing")
        }
    }

    /// Going back to the transcript lights it again — the rule defers the
    /// highlight, it does not clear the selection.
    func testReturningToTheConversationLightsItAgain() {
        XCTAssertFalse(SidebarSelection.isConversationSelected(
            sessionId: chat, activeChatId: chat, workspace: .tasks))
        XCTAssertTrue(SidebarSelection.isConversationSelected(
            sessionId: chat, activeChatId: chat, workspace: .conversation))
    }

    func testNoActiveChatLightsNothing() {
        XCTAssertFalse(SidebarSelection.isConversationSelected(
            sessionId: chat, activeChatId: nil, workspace: .conversation))
    }
}
