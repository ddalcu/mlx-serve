import XCTest
@testable import MLXCore

/// User-made sidebar groups: named folders over chats, agent threads and terminals.
final class SidebarGroupsTests: XCTestCase {

    private func rows(_ n: Int) -> [SidebarChatRows.Row] {
        (0..<n).map { _ in .chat(ChatSession(title: "c")) }
    }

    func testPartitionKeepsRowOrderAndListsEmptyGroups() throws {
        let r = rows(4)
        var g = SidebarGroups()
        let work = try XCTUnwrap(g.create("Work", with: [r[3].id, r[1].id]))
        let empty = try XCTUnwrap(g.create("Empty", with: []))

        let p = g.partition(r)
        XCTAssertEqual(p.groups.map(\.group.id), [work, empty])
        XCTAssertEqual(p.groups[0].rows.map(\.id), [r[1].id, r[3].id])
        XCTAssertTrue(p.groups[1].rows.isEmpty)
        XCTAssertEqual(p.ungrouped.map(\.id), [r[0].id, r[2].id])
    }

    func testMoveRemoveAndDeleteNeverLoseRows() throws {
        let r = rows(3)
        var g = SidebarGroups()
        let a = try XCTUnwrap(g.create("A", with: [r[0].id, r[1].id]))
        let b = try XCTUnwrap(g.create("B", with: []))

        g.assign([r[1].id], to: b)
        XCTAssertEqual(g.group(of: r[1].id), b)
        g.assign([r[0].id], to: nil)
        XCTAssertNil(g.group(of: r[0].id))

        g.assign([r[2].id], to: a)
        g.delete(a)
        XCTAssertEqual(g.groups.map(\.id), [b])
        XCTAssertNil(g.group(of: r[2].id))
        XCTAssertEqual(g.partition(r).ungrouped.map(\.id), [r[0].id, r[2].id])
    }

    func testADraggedRowJoinsTheGroupOfTheRowItLandsOn() throws {
        let r = rows(3)
        var g = SidebarGroups()
        let a = try XCTUnwrap(g.create("A", with: [r[0].id]))
        let b = try XCTUnwrap(g.create("B", with: [r[1].id]))

        g.join(r[2].id, groupOf: r[0].id)
        XCTAssertEqual(g.group(of: r[2].id), a)
        g.join(r[2].id, groupOf: r[1].id)
        XCTAssertEqual(g.group(of: r[2].id), b)
        g.join(r[1].id, groupOf: UUID())
        XCTAssertNil(g.group(of: r[1].id))
    }

    func testNamesAreTrimmedAndBlankIsRefused() throws {
        var g = SidebarGroups()
        XCTAssertNil(g.create("   ", with: []))
        let id = try XCTUnwrap(g.create("  Research ", with: []))
        g.rename(id, to: " ")
        XCTAssertEqual(g.groups.first?.name, "Research")
        g.rename(id, to: "Papers")
        XCTAssertEqual(g.groups.first?.name, "Papers")
    }

    func testSurvivesARelaunchMinusRowsThatDidNot() throws {
        let kept = UUID(), gone = UUID()
        var g = SidebarGroups()
        let id = try XCTUnwrap(g.create("Work", with: [kept, gone]))
        g.toggleCollapsed(id)

        var back = try JSONDecoder().decode(SidebarGroups.self, from: JSONEncoder().encode(g))
        back.retain(only: [kept])
        XCTAssertEqual(back.groups, g.groups)
        XCTAssertEqual(back.group(of: kept), id)
        XCTAssertNil(back.group(of: gone))
    }
}
