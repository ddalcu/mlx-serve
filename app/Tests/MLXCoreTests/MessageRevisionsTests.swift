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

/// Editing the model's own reply — putting words in its mouth, then letting it
/// carry on from them.
final class EditAssistantReplyTests: XCTestCase {

    private func rev(_ s: String) -> MessageRevision { MessageRevision(content: s) }

    func testAnEditIsWrittenIntoTheVersionBeingRead() {
        // Otherwise: edit version 2, step to 1, step back — and the edit is
        // gone, because stepping reloads content from the stored revision.
        let updated = MessageRevisions.applyingEdit("edited",
                                                    to: [rev("first"), rev("second")],
                                                    at: 1)
        XCTAssertEqual(updated.map(\.content), ["first", "edited"])
    }

    func testTheOtherVersionsAreUntouched() {
        let updated = MessageRevisions.applyingEdit("edited",
                                                    to: [rev("first"), rev("second"), rev("third")],
                                                    at: 0)
        XCTAssertEqual(updated.map(\.content), ["edited", "second", "third"])
    }

    func testAPlainReplyHasNoVersionsToSync() {
        // The ordinary case: no regeneration ever happened, so the edit lives
        // in `content` alone and the list stays empty.
        XCTAssertTrue(MessageRevisions.applyingEdit("edited", to: [], at: 0).isEmpty)
    }

    func testAnOutOfRangeIndexChangesNothing() {
        let existing = [rev("first")]
        XCTAssertEqual(MessageRevisions.applyingEdit("edited", to: existing, at: 4), existing)
        XCTAssertEqual(MessageRevisions.applyingEdit("edited", to: existing, at: -1), existing)
    }

    /// Regression guard for the fix in bf5846b: `onEdit` drives BOTH the
    /// context menu and a `highPriorityGesture` double-click, and that gesture
    /// beats `textSelection`'s own double-click-to-select-a-word. Making the
    /// model's replies editable must not take word selection away from them —
    /// which is most of what anyone selects in a transcript.
    func testDoubleClickToEditStaysOffTheModelsReplies() {
        let source = SourceScan.source("Views/ChatView.swift", from: #filePath)
        // The APPLICATION site, not the type's declaration — and matched on
        // the bare name, since the argument may wrap onto its own line.
        guard let range = source.range(of: ".modifier(DoubleClickToEdit(") else {
            return XCTFail("the double-click modifier is no longer applied")
        }
        let call = String(source[range.lowerBound...].prefix(200))
        // Either spelling of the same gate — what must not happen is the
        // modifier going back to keying on `onEdit` alone, which is now
        // non-nil for the model's replies too.
        XCTAssertTrue(call.contains("message.role"),
                      "double-click may only open an edit on the USER's own message; "
                      + "assistant replies keep double-click-to-select-a-word. Got: \(call.prefix(120))")
    }
}
