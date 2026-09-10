import XCTest
@testable import MLXCore

/// The Chats section is ONE list of conversations and terminals, newest first.
final class SidebarChatRowsTests: XCTestCase {

    private func chat(_ title: String, at t: TimeInterval) -> ChatSession {
        var s = ChatSession(title: title)
        s.createdAt = Date(timeIntervalSince1970: t)
        return s
    }

    private func terminal(at t: TimeInterval) -> TerminalSessionList.Session {
        TerminalSessionList.Session(id: UUID(), label: "pi", autoName: "pi", agentId: "pi",
                                    workspace: "/w", createdAt: Date(timeIntervalSince1970: t),
                                    phase: .live)
    }

    func testMergeIsNewestFirstAcrossBothKinds() {
        let a = chat("a", at: 10), c = chat("c", at: 30)
        let t = terminal(at: 20)
        let rows = SidebarChatRows.merge(chats: [c, a], terminals: [t])
        XCTAssertEqual(rows.map(\.id), [c.id, t.id, a.id])
    }

    func testNoTerminalsLeavesChatsUnchanged() {
        let a = chat("a", at: 10), b = chat("b", at: 20)
        let rows = SidebarChatRows.merge(chats: [b, a], terminals: [])
        XCTAssertEqual(rows.map(\.id), [b.id, a.id])
    }
}

extension SidebarChatRowsTests {

    func testAManualOrderWinsAndUnknownRowsStayNewestFirstOnTop() {
        let a = chat("a", at: 10), b = chat("b", at: 20), c = chat("c", at: 30)
        let t = terminal(at: 25)
        // The user dragged a to the top and b below the terminal; c is new.
        let rows = SidebarChatRows.merge(chats: [c, b, a], terminals: [t], order: [a.id, t.id, b.id])
        XCTAssertEqual(rows.map(\.id), [c.id, a.id, t.id, b.id])
        // Ids that no longer exist are ignored, not a hole.
        let stale = SidebarChatRows.merge(chats: [b, a], terminals: [], order: [UUID(), a.id, b.id])
        XCTAssertEqual(stale.map(\.id), [a.id, b.id])
    }

    func testMovedDropsTheRowIntoTheTargetsSlot() {
        let a = UUID(), b = UUID(), c = UUID(), d = UUID()
        XCTAssertEqual(SidebarChatRows.moved(a, onto: c, in: [a, b, c, d]), [b, c, a, d])
        XCTAssertEqual(SidebarChatRows.moved(d, onto: a, in: [a, b, c, d]), [d, a, b, c])
        XCTAssertEqual(SidebarChatRows.moved(b, onto: b, in: [a, b, c, d]), [a, b, c, d])
        XCTAssertEqual(SidebarChatRows.moved(b, onto: UUID(), in: [a, b]), [a, b], "unknown target: no move")
    }
}
