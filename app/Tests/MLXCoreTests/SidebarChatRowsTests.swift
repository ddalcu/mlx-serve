import XCTest
@testable import MLXCore

/// The Chats section is ONE list of conversations and terminals: dragged rows in their order, the rest newest first above them.
final class SidebarChatRowsTests: XCTestCase {

    private func chat(_ title: String, at t: TimeInterval, group: UUID? = nil, position: Int? = nil) -> ChatSession {
        var s = ChatSession(title: title)
        s.createdAt = Date(timeIntervalSince1970: t)
        s.groupId = group
        s.sidebarPosition = position
        return s
    }

    private func terminal(at t: TimeInterval, group: UUID? = nil, position: Int? = nil) -> TerminalSessionList.Session {
        var s = TerminalSessionList.Session(id: UUID(), label: "pi", autoName: "pi", agentId: "pi",
                                            workspace: "/w", createdAt: Date(timeIntervalSince1970: t),
                                            phase: .live)
        s.groupId = group
        s.position = position
        return s
    }

    private typealias Placement = SidebarChatRows.Placement

    func testMergeIsNewestFirstAcrossBothKinds() {
        let a = chat("a", at: 10), c = chat("c", at: 30)
        let t = terminal(at: 20)
        let rows = SidebarChatRows.merge(chats: [c, a], terminals: [t])
        XCTAssertEqual(rows.map(\.id), [c.id, t.id, a.id])
    }

    /// Stored times are whole seconds; terminals arrive oldest first.
    func testTerminalsFromTheSameSecondStayNewestFirst() {
        let older = terminal(at: 20), newer = terminal(at: 20)
        XCTAssertEqual(SidebarChatRows.merge(chats: [], terminals: [older, newer]).map(\.id), [newer.id, older.id])
    }

    func testNoTerminalsLeavesChatsUnchanged() {
        let a = chat("a", at: 10), b = chat("b", at: 20)
        let rows = SidebarChatRows.merge(chats: [b, a], terminals: [])
        XCTAssertEqual(rows.map(\.id), [b.id, a.id])
    }

    func testDraggedRowsFollowTheirPositionsBelowTheUnplacedOnes() {
        // The user dragged a to the top and b below the terminal; c is new.
        let a = chat("a", at: 10, position: 0), b = chat("b", at: 20, position: 2), c = chat("c", at: 30)
        let t = terminal(at: 25, position: 1)
        let rows = SidebarChatRows.merge(chats: [c, b, a], terminals: [t])
        XCTAssertEqual(rows.map(\.id), [c.id, a.id, t.id, b.id])
    }

    func testMovedDropsTheRowIntoTheTargetsSlot() {
        let a = UUID(), b = UUID(), c = UUID(), d = UUID()
        XCTAssertEqual(SidebarChatRows.moved(a, onto: c, in: [a, b, c, d]), [b, c, a, d])
        XCTAssertEqual(SidebarChatRows.moved(d, onto: a, in: [a, b, c, d]), [d, a, b, c])
        XCTAssertEqual(SidebarChatRows.moved(b, onto: b, in: [a, b, c, d]), [a, b, c, d])
        XCTAssertEqual(SidebarChatRows.moved(b, onto: UUID(), in: [a, b]), [a, b], "unknown target: no move")
    }

    func testADropNumbersTheTargetsParentOnly() {
        let work = UUID()
        let x = chat("x", at: 50, group: work), y = chat("y", at: 40, group: work, position: 4)
        let a = chat("a", at: 30), b = chat("b", at: 20), t = terminal(at: 10)
        let visible = SidebarChatRows.merge(chats: [x, y, a, b], terminals: [t])
        XCTAssertEqual(visible.map(\.id), [x.id, a.id, b.id, t.id, y.id])

        let placements = SidebarChatRows.dropped(t.id, onto: a.id, in: visible)
        XCTAssertEqual(placements, [t.id: Placement(group: nil, position: 0),
                                    a.id: Placement(group: nil, position: 1),
                                    b.id: Placement(group: nil, position: 2)],
                       "the group's rows keep their places")
    }

    func testARowDroppedIntoAnotherGroupJoinsIt() {
        let work = UUID()
        let x = chat("x", at: 50, group: work), y = chat("y", at: 40, group: work)
        let a = chat("a", at: 30)
        let visible = SidebarChatRows.merge(chats: [x, y, a], terminals: [])

        let placements = SidebarChatRows.dropped(a.id, onto: x.id, in: visible)
        XCTAssertEqual(placements, [a.id: Placement(group: work, position: 0),
                                    x.id: Placement(group: work, position: 1),
                                    y.id: Placement(group: work, position: 2)])
        XCTAssertEqual(SidebarChatRows.dropped(a.id, onto: UUID(), in: visible), [:], "unknown target: no move")
        XCTAssertEqual(SidebarChatRows.dropped(a.id, onto: a.id, in: visible), [:])
    }

    /// Dropped on the first row below a group's edge: the order stays, the group changes.
    func testARowDroppedOnItsNeighbourInAnotherGroupStillJoinsIt() {
        let work = UUID()
        let a = chat("a", at: 50), x = chat("x", at: 40, group: work), y = chat("y", at: 30, group: work)
        let visible = SidebarChatRows.merge(chats: [a, x, y], terminals: [])
        XCTAssertEqual(visible.map(\.id), [a.id, x.id, y.id])

        XCTAssertEqual(SidebarChatRows.dropped(a.id, onto: x.id, in: visible),
                       [a.id: Placement(group: work, position: 0),
                        x.id: Placement(group: work, position: 1),
                        y.id: Placement(group: work, position: 2)])
        XCTAssertEqual(SidebarChatRows.dropped(x.id, onto: y.id, in: visible), [:], "same group, same order: nothing to write")
    }

    /// Dragged rows of the group land on top of the parent's dragged rows; undragged ones stay sorted by date.
    func testADissolvedGroupsRowsMoveToItsParent() {
        let work = UUID()
        let placedIn = chat("placed in", at: 50, group: work, position: 0)
        let looseIn = terminal(at: 45, group: work)
        let placedOut = chat("placed out", at: 40, position: 0)
        let looseOut = chat("loose out", at: 35)
        let rows = SidebarChatRows.merge(chats: [placedIn, placedOut, looseOut], terminals: [looseIn])

        let placements = SidebarChatRows.dissolving(work, into: nil, rows: rows)
        XCTAssertEqual(placements, [placedIn.id: Placement(group: nil, position: 0),
                                    placedOut.id: Placement(group: nil, position: 1),
                                    looseIn.id: Placement(group: nil, position: nil)])
    }
}
