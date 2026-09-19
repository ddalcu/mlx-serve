import XCTest
@testable import MLXCore

/// How many of a conversation's rows the transcript lays out when it opens.
final class TranscriptWindowTests: XCTestCase {

    func testAShortConversationIsShownWhole() {
        XCTAssertEqual(TranscriptWindow.firstRow(total: 40, limit: 150), 0)
        XCTAssertEqual(TranscriptWindow.firstRow(total: 150, limit: 150), 0)
    }

    func testALongConversationOpensOnItsLastRows() {
        XCTAssertEqual(TranscriptWindow.firstRow(total: 500, limit: 150), 350)
    }

    /// The cut is an INDEX fixed when the chat opens, not "the last N": rows
    /// appended by a streaming reply must not push the oldest visible row out
    /// and move everything the reader sees.
    func testRowsAppendedLaterDoNotMoveTheCut() {
        XCTAssertEqual(TranscriptWindow.clamp(first: 350, total: 512, limit: 150), 350)
    }

    /// A transcript that got shorter (a delete, an edit-and-resend) may no
    /// longer reach the cut; then it is cut like a chat that just opened.
    func testACutPastTheEndFallsBackToTheLastRows() {
        XCTAssertEqual(TranscriptWindow.clamp(first: 350, total: 300, limit: 150), 150)
        XCTAssertEqual(TranscriptWindow.clamp(first: 350, total: 100, limit: 150), 0)
    }

    func testRevealingEarlierRowsShowsEverything() {
        XCTAssertEqual(TranscriptWindow.clamp(first: 0, total: 512, limit: 150), 0)
    }

    /// Bounded by markdown layouts, not by rows: ~10 ms per rendered reply on
    /// an M-series Mac, so the limit keeps a chat's open under a second.
    func testTheLimitBoundsOpeningToUnderASecond() {
        XCTAssertLessThanOrEqual(TranscriptWindow.rowLimit, 200)
        XCTAssertGreaterThanOrEqual(TranscriptWindow.rowLimit, 100)
    }
}
