import XCTest
@testable import MLXCore

/// The sidebar dot: pulsing while a chat's turn runs (green generating, blue
/// running tools), grey once a turn ended in a chat nobody was looking at,
/// gone as soon as that chat is selected.
final class SidebarActivityTests: XCTestCase {

    func testLivePhasesShowOnSelectedAndUnselectedRowsAlike() {
        var activity = SidebarActivity()
        let s = UUID()
        activity.setPhase(.generating, for: s)
        XCTAssertEqual(activity.dot(for: s, isSelected: true), .generating)
        XCTAssertEqual(activity.dot(for: s, isSelected: false), .generating)
        activity.setPhase(.tool, for: s)
        XCTAssertEqual(activity.dot(for: s, isSelected: true), .tool)
        XCTAssertEqual(activity.dot(for: s, isSelected: false), .tool)
    }

    func testATurnEndedUnseenEarnsTheFinishedMarkOnlyWhileUnselected() {
        var activity = SidebarActivity()
        let s = UUID()
        activity.setPhase(.generating, for: s)
        activity.end(for: s, seen: false)
        XCTAssertEqual(activity.dot(for: s, isSelected: false), .finished)
        XCTAssertNil(activity.dot(for: s, isSelected: true), "the selected chat never shows the grey dot")
        activity.markSeen(s)
        XCTAssertNil(activity.dot(for: s, isSelected: false))
    }

    func testATurnEndedOnScreenLeavesNoMark() {
        var activity = SidebarActivity()
        let s = UUID()
        activity.setPhase(.tool, for: s)
        activity.end(for: s, seen: true)
        XCTAssertNil(activity.dot(for: s, isSelected: false))
        XCTAssertTrue(activity.unseen.isEmpty)
    }

    func testANewTurnReplacesTheFinishedMark() {
        var activity = SidebarActivity()
        let s = UUID()
        activity.setPhase(.generating, for: s)
        activity.end(for: s, seen: false)
        activity.setPhase(.generating, for: s)
        XCTAssertEqual(activity.dot(for: s, isSelected: false), .generating)
        activity.end(for: s, seen: true)
        XCTAssertNil(activity.dot(for: s, isSelected: false))
    }

    func testATurnEndedOnAnErrorEarnsTheAttentionMark() {
        var activity = SidebarActivity()
        let s = UUID()
        activity.setPhase(.tool, for: s)
        activity.end(for: s, seen: false, outcome: .attention)
        XCTAssertEqual(activity.dot(for: s, isSelected: false), .attention)
        XCTAssertNil(activity.dot(for: s, isSelected: true))
        activity.markSeen(s)
        XCTAssertNil(activity.dot(for: s, isSelected: false))
    }

    func testSessionsAreIndependent() {
        var activity = SidebarActivity()
        let a = UUID(), b = UUID()
        activity.setPhase(.generating, for: a)
        XCTAssertNil(activity.dot(for: b, isSelected: false))
        activity.end(for: a, seen: false)
        XCTAssertEqual(activity.unseen, [a])
    }
}
