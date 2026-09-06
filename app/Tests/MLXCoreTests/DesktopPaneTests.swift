import XCTest
@testable import MLXCore

/// The sandbox desktop pane: the guest's screen with the ACTIVE chat beside
/// it, so the user watches the agent drive AND can message it mid-turn
/// ("I solved the captcha, go on") without leaving the screen.
@MainActor
final class DesktopPaneTests: XCTestCase {
    private func source(_ relativePath: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// The pane hosts the real chat view (not a copy of the transcript), for
    /// the active chat, in a split the user can collapse. The hint under the
    /// screen is how the user learns focus follows the click.
    func testThePaneHostsTheActiveChatBesideTheScreen() throws {
        let pane = try source("Sources/MLXServe/Views/DesktopPane.swift")
        XCTAssertTrue(pane.contains("ChatDetailView(sessionId: id)"), "the chat column is ChatDetailView")
        XCTAssertTrue(pane.contains("appState.activeChatId"), "…for the ACTIVE chat")
        XCTAssertTrue(pane.contains("appState.newChatSession()"), "no chat yet → open one, like the window does")
        XCTAssertTrue(pane.contains("splitHandle"), "screen left, chat right, a drag handle between")
        XCTAssertTrue(pane.contains(".frame(width: Self.chatColumnWidth(stored: storedChatWidth, available: geo.size.width))"),
                      "the chat column is the remembered width, clamped to what the window leaves")
        XCTAssertTrue(pane.contains("Click the screen to type into it"), "the focus hint")
        XCTAssertTrue(pane.contains("Esc stops the agent"), "…and the Esc intercept it names")
    }

    /// The remembered column width is clamped: a first launch (0), garbage
    /// (NaN, negative) and a width from a huge window all resolve to something
    /// the composer can be typed into.
    func testChatColumnWidthIsClamped() {
        XCTAssertEqual(DesktopPane.chatColumnWidth(stored: 0), DesktopPane.defaultChatWidth)
        XCTAssertEqual(DesktopPane.chatColumnWidth(stored: .nan), DesktopPane.defaultChatWidth)
        XCTAssertEqual(DesktopPane.chatColumnWidth(stored: -5), DesktopPane.defaultChatWidth)
        XCTAssertEqual(DesktopPane.chatColumnWidth(stored: 100), DesktopPane.minChatWidth)
        XCTAssertEqual(DesktopPane.chatColumnWidth(stored: 5000), DesktopPane.maxChatWidth)
        XCTAssertEqual(DesktopPane.chatColumnWidth(stored: 600), 600)
        XCTAssertEqual(DesktopPane.chatColumnWidth(stored: 0, default: 600), 600)
        // The window's width wins over the remembered width: the screen keeps
        // its minimum, the chat shrinks (to a floor), never overflows.
        XCTAssertEqual(DesktopPane.chatColumnWidth(stored: 480, available: 2000), 480)
        XCTAssertEqual(DesktopPane.chatColumnWidth(stored: 480, available: 800),
                       800 - DesktopPane.minScreenWidth - DesktopPane.handleWidth)
        XCTAssertEqual(DesktopPane.chatColumnWidth(stored: 480, available: 500), DesktopPane.compactChatWidth)
        XCTAssertLessThan(DesktopPane.compactChatWidth, ChatMetrics.compactComposerWidth,
                          "a column at the floor renders the compact composer row")
    }

    /// Send while a turn runs: an agent turn takes the text for its next step,
    /// a plain turn refuses (Stop first), an idle chat starts a turn. The
    /// composer's Return key and `proceedSend` both read this one rule.
    func testComposerSendRoutesByTurnKind() {
        XCTAssertEqual(ComposerSend.decision(state: .idle, agentTurn: false), .start)
        XCTAssertEqual(ComposerSend.decision(state: .idle, agentTurn: true), .start)
        XCTAssertEqual(ComposerSend.decision(state: .generatingHere, agentTurn: true), .appendMidTurn)
        XCTAssertEqual(ComposerSend.decision(state: .generatingHere, agentTurn: false), .refuse)
    }

    /// The view side goes through the rule: a `.appendMidTurn` decision queues
    /// on the engine and never starts a turn; the queued text is rendered in
    /// the transcript at once.
    func testTheComposerAndTranscriptUseTheMidTurnQueue() throws {
        let chat = try source("Sources/MLXServe/Views/ChatView.swift")
        XCTAssertTrue(chat.contains("ComposerSend.decision(state: composerState"), "proceedSend routes by the rule")
        XCTAssertTrue(chat.contains("chatEngine.noteMidTurnMessage(sessionId: sessionId"), "…and queues on the engine")
        XCTAssertTrue(chat.contains("chatEngine.midTurnQueue[sessionId]"), "the transcript shows the queued text")
        let engine = try source("Sources/MLXServe/Services/ChatTurnEngine.swift")
        // Delivery happens at the loop's round boundary, never by appending
        // straight into a streaming session (`updateLastMessage` writes into
        // whatever message is last).
        XCTAssertTrue(engine.contains("deliverMidTurnMessages(sessionId: sessionId)\n\n            // Build message history"),
                      "delivered at the top of a round")
        XCTAssertTrue(engine.contains("if deliverMidTurnMessages(sessionId: sessionId) > 0 { continue }"),
                      "a message typed during the final reply gets one more round")
    }
}
