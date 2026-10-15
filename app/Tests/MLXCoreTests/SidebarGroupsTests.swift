import XCTest
@testable import MLXCore

/// User-made sidebar groups: a tree of named folders; each chat, agent thread or terminal names its own group.
final class SidebarGroupsTests: XCTestCase {

    private func chat(in group: UUID? = nil) -> SidebarChatRows.Row {
        var s = ChatSession(title: "c")
        s.groupId = group
        return .chat(s)
    }

    func testNewGroupsGoLastAmongTheirSiblings() throws {
        var g = SidebarGroups()
        let a = try XCTUnwrap(g.create("A"))
        let b = try XCTUnwrap(g.create("B"))
        let inner = try XCTUnwrap(g.create("Inner", in: a))
        let second = try XCTUnwrap(g.create("Second inner", in: a))

        XCTAssertEqual(g.children(of: nil).map(\.id), [a, b])
        XCTAssertEqual(g.children(of: a).map(\.id), [inner, second])
        XCTAssertEqual(g.children(of: a).map(\.position), [0, 1], "positions count within the parent")
        XCTAssertEqual(g.ordered.map(\.id), [a, inner, second, b], "a parent is followed by its subtree")
    }

    func testPartitionKeepsRowOrderAndListsEmptyGroups() throws {
        var g = SidebarGroups()
        let work = try XCTUnwrap(g.create("Work"))
        let empty = try XCTUnwrap(g.create("Empty"))
        let r = [chat(), chat(in: work), chat(in: UUID()), chat(in: work)]

        let p = g.partition(r)
        XCTAssertEqual(p.groups.map(\.group.id), [work, empty])
        XCTAssertEqual(p.groups[0].rows.map(\.id), [r[1].id, r[3].id])
        XCTAssertTrue(p.groups[1].rows.isEmpty)
        XCTAssertEqual(p.ungrouped.map(\.id), [r[0].id, r[2].id], "a group nobody has is no group")
    }

    /// Its subgroups take its place; where its rows go is `SidebarChatRows.dissolving`.
    func testDeletingAGroupMovesItsSubgroupsUpIntoItsPlace() throws {
        var g = SidebarGroups()
        let a = try XCTUnwrap(g.create("A"))
        let b = try XCTUnwrap(g.create("B"))
        let c = try XCTUnwrap(g.create("C"))
        let b1 = try XCTUnwrap(g.create("B1", in: b))
        let b2 = try XCTUnwrap(g.create("B2", in: b))
        let deep = try XCTUnwrap(g.create("Deep", in: b1))
        let deeper = try XCTUnwrap(g.create("Deeper", in: b1))

        g.delete(b1)
        XCTAssertEqual(g.children(of: b).map(\.id), [deep, deeper, b2])
        XCTAssertEqual(g.groups.first { $0.id == deep }?.parentId, b, "stored, not just drawn there")

        g.delete(b)
        XCTAssertEqual(g.children(of: nil).map(\.id), [a, deep, deeper, b2, c])
        XCTAssertEqual(g.children(of: nil).map(\.position), [0, 1, 2, 3, 4])
        XCTAssertEqual(g.groups.count, 5)
    }

    /// Nothing the app writes, but a hand edit or a future build could: every group still shows.
    func testAGroupWithAMissingOrCyclicParentShowsAtTheRoot() {
        let a = UUID(), b = UUID(), c = UUID(), orphan = UUID(), child = UUID()
        let g = SidebarGroups([
            .init(id: a, name: "A", parentId: nil, position: 0),
            .init(id: b, name: "B", parentId: c, position: 0),
            .init(id: c, name: "C", parentId: b, position: 1),
            .init(id: orphan, name: "Orphan", parentId: UUID(), position: 2),
            .init(id: child, name: "Child of orphan", parentId: orphan, position: 0),
        ])
        XCTAssertNil(g.parent(of: b))
        XCTAssertNil(g.parent(of: c))
        XCTAssertNil(g.parent(of: orphan))
        XCTAssertEqual(g.parent(of: child), orphan, "only the broken link is cut")
        XCTAssertEqual(Set(g.ordered.map(\.id)), [a, b, c, orphan, child])
        XCTAssertEqual(g.ordered.count, 5)
    }

    func testNamesAreTrimmedAndBlankIsRefused() throws {
        var g = SidebarGroups()
        XCTAssertNil(g.create("   "))
        let id = try XCTUnwrap(g.create("  Research "))
        g.rename(id, to: " ")
        XCTAssertEqual(g.groups.first?.name, "Research")
        g.rename(id, to: "Papers")
        XCTAssertEqual(g.groups.first?.name, "Papers")
    }

    func testCollapsingTogglesOneGroup() throws {
        var g = SidebarGroups()
        let a = try XCTUnwrap(g.create("A"))
        _ = try XCTUnwrap(g.create("B"))
        g.toggleCollapsed(a)
        XCTAssertEqual(g.groups.map(\.collapsed), [true, false])
        g.toggleCollapsed(a)
        XCTAssertFalse(g.groups.contains { $0.collapsed })
    }
}
