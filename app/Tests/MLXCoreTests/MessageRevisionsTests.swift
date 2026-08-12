import XCTest
@testable import MLXCore

/// Regenerating a reply used to DELETE the one it replaced — a better first
/// answer was gone the moment you asked for a second opinion. These pin the
/// rules behind the pager that keeps them.
final class MessageRevisionsTests: XCTestCase {

    private func rev(_ s: String, reasoning: String? = nil) -> MessageRevision {
        MessageRevision(content: s, reasoningContent: reasoning)
    }

    // MARK: - When the pager exists at all

    func testOneVersionIsNotAChoice() {
        // The ordinary reply must look exactly as it always did: no chrome.
        XCTAssertFalse(MessageRevisions.isPagerVisible([]))
        XCTAssertFalse(MessageRevisions.isPagerVisible([rev("only")]))
    }

    func testTwoVersionsGetAPager() {
        XCTAssertTrue(MessageRevisions.isPagerVisible([rev("a"), rev("b")]))
    }

    // MARK: - Reading it

    func testTheCounterIsOneBased() {
        XCTAssertEqual(MessageRevisions.label(index: 0, count: 3), "1/3")
        XCTAssertEqual(MessageRevisions.label(index: 2, count: 3), "3/3")
    }

    func testTheCounterSurvivesAnOutOfRangeIndex() {
        // A stored index outliving the list it pointed into must not render
        // "4/3" or crash the row.
        XCTAssertEqual(MessageRevisions.label(index: 9, count: 3), "3/3")
        XCTAssertEqual(MessageRevisions.label(index: -1, count: 3), "1/3")
        XCTAssertEqual(MessageRevisions.label(index: 0, count: 0), "")
    }

    // MARK: - Stepping

    func testArrowsStopAtTheEndsRatherThanWrapping() {
        // Wrapping makes both arrows do the same thing at the ends, and 3/3 →
        // 1/3 reads as a new reply arriving rather than a step.
        XCTAssertEqual(MessageRevisions.step(index: 0, by: -1, count: 3), 0)
        XCTAssertEqual(MessageRevisions.step(index: 2, by: 1, count: 3), 2)
        XCTAssertEqual(MessageRevisions.step(index: 1, by: 1, count: 3), 2)
        XCTAssertEqual(MessageRevisions.step(index: 1, by: -1, count: 3), 0)
    }

    func testTheArrowsDisableAtTheEnds() {
        XCTAssertFalse(MessageRevisions.canGoBack(index: 0))
        XCTAssertTrue(MessageRevisions.canGoBack(index: 1))
        XCTAssertTrue(MessageRevisions.canGoForward(index: 1, count: 3))
        XCTAssertFalse(MessageRevisions.canGoForward(index: 2, count: 3))
    }

    // MARK: - Capturing the reply being replaced

    func testTheFirstRegenerationCapturesTheReplyItReplaces() {
        // `regenerate` truncates back to the last user message, so the old
        // reply is destroyed unless it is captured here.
        let seeded = MessageRevisions.seeding(prior: rev("first answer"), existing: [])
        XCTAssertEqual(seeded.map(\.content), ["first answer"])
    }

    func testLaterRegenerationsKeepTheListTheyAlreadyHave() {
        let existing = [rev("first"), rev("second")]
        let seeded = MessageRevisions.seeding(prior: rev("second"), existing: existing)
        XCTAssertEqual(seeded, existing, "seeding twice must not duplicate version 1")
    }

    func testAnEmptyPriorReplyIsNotAVersion() {
        // A failed or empty reply is not worth a page — seeding one shows a
        // pager whose first page is blank.
        XCTAssertTrue(MessageRevisions.seeding(prior: rev(""), existing: []).isEmpty)
        XCTAssertTrue(MessageRevisions.seeding(prior: rev("  \n"), existing: []).isEmpty)
    }

    func testReasoningIsCapturedWithItsOwnVersion() {
        // Revision 2's answer under revision 1's thinking is worse than no
        // thinking at all.
        let seeded = MessageRevisions.seeding(prior: rev("answer", reasoning: "because"), existing: [])
        XCTAssertEqual(seeded.first?.reasoningContent, "because")
    }

    // MARK: - Recording the new one

    func testAFinishedRegenerationBecomesTheNewestVersionAndIsSelected() {
        let (revisions, index) = MessageRevisions.committing(rev("second"), into: [rev("first")])
        XCTAssertEqual(revisions.map(\.content), ["first", "second"])
        XCTAssertEqual(index, 1, "you are looking at the reply you just asked for")
    }

    func testAnOrdinaryReplyRecordsNothing() {
        // No regeneration happened, so the message stays a plain one.
        let (revisions, index) = MessageRevisions.committing(rev("hello"), into: [])
        XCTAssertTrue(revisions.isEmpty)
        XCTAssertEqual(index, 0)
    }

    func testAnIdenticalRegenerationDoesNotGrowTheList() {
        // At temperature 0 the model repeats itself, and every press would
        // otherwise add another identical page to step through.
        let (revisions, index) = MessageRevisions.committing(rev("same"), into: [rev("first"), rev("same")])
        XCTAssertEqual(revisions.count, 2)
        XCTAssertEqual(index, 1)
    }
}
