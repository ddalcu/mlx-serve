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

    // MARK: - Seed and commit as ONE step

    /// The seed cannot be written onto the reply when the regeneration STARTS,
    /// because the reply does not exist yet — and on the agent path it does not
    /// exist for several more rounds. `runPlainTurn` appends its streaming
    /// placeholder synchronously, but `runAgentLoop` appends one per iteration
    /// from inside a Task, and the reply the pager belongs to is the LAST of
    /// them. So the seed is held until the turn ends and applied to whatever
    /// reply the turn actually produced.
    func testAHeldSeedIsAppliedToTheReplyTheTurnProduced() {
        let (revisions, index) = MessageRevisions.finishing(
            seed: [rev("first")], existing: [], finished: rev("second"))
        XCTAssertEqual(revisions.map(\.content), ["first", "second"])
        XCTAssertEqual(index, 1)
    }

    /// A message that already carries a list is on its third regeneration —
    /// the seed is stale and must not overwrite what is there.
    func testAnExistingListOutranksTheSeed() {
        let (revisions, _) = MessageRevisions.finishing(
            seed: [rev("stale")], existing: [rev("first"), rev("second")], finished: rev("third"))
        XCTAssertEqual(revisions.map(\.content), ["first", "second", "third"])
    }

    /// No regeneration in flight: an ordinary reply is left exactly as it was,
    /// pager and all (which is to say, none).
    func testNoSeedAndNoListLeavesAnOrdinaryReplyAlone() {
        let (revisions, index) = MessageRevisions.finishing(
            seed: nil, existing: [], finished: rev("hello"))
        XCTAssertTrue(revisions.isEmpty)
        XCTAssertEqual(index, 0)
    }

    /// A turn that failed before streaming anything is not a version — the
    /// same rule `seeding` applies to an empty prior. The reply it was going
    /// to replace still has to survive, or the regeneration destroyed it for
    /// nothing.
    func testAFailedRegenerationKeepsTheReplyItWasReplacing() {
        let (revisions, index) = MessageRevisions.finishing(
            seed: [rev("first")], existing: [], finished: rev("   \n "))
        XCTAssertEqual(revisions.map(\.content), ["first"])
        XCTAssertEqual(index, 0)
    }
}

/// The seed has to survive from the moment a regeneration is asked for to the
/// moment its reply is finished — which on the agent path is many messages
/// later. This pins the wiring that carries it.
final class RegenerationSeedWiringTests: XCTestCase {

    /// The bug: `regenerate` wrote the seed straight onto `messages.last`
    /// immediately after calling `runTurn`. That is the streaming placeholder
    /// on the plain path and the USER message on the agent path, where the
    /// placeholder is appended from inside a Task — so with Tools on, the
    /// role guard failed and the pager silently never appeared.
    func testTheSeedIsHeldRatherThanWrittenOntoWhateverIsLast() throws {
        let source = SourceScan.source("AppState.swift", from: #filePath)
        let body = try XCTUnwrap(
            SourceScan.declarationBody(from: "func seedRevisions", in: source),
            "seedRevisions moved — repoint this scan")
        XCTAssertTrue(body.contains("pendingRevisionSeed"), """
            seedRevisions writes to a message instead of holding the seed — on \
            the agent path the reply it belongs to does not exist yet.
            """)
        XCTAssertFalse(body.contains("messages.indices.last"), """
            seedRevisions still targets whatever message happens to be last at \
            the moment a regeneration starts.
            """)
    }

    /// A continuation is not a regeneration, and the turn exit is the only
    /// place that can tell them apart.
    ///
    /// The pager counts ANSWERS to the same question. A continuation is the
    /// answer you are reading, carrying on — so it must sync into the version
    /// being read (`applyingEdit`) rather than reach `committing`, which filed
    /// it as a new version: a reply regenerated once went to 3/3 the moment you
    /// finished it, and 2/3 was then the same reply with its ending removed.
    func testAContinuationExtendsTheVersionBeingReadRatherThanAddingOne() throws {
        let source = SourceScan.source("AppState.swift", from: #filePath)
        let body = try XCTUnwrap(
            SourceScan.declarationBody(from: "func finishRevisions", in: source),
            "finishRevisions moved — repoint this scan")
        XCTAssertTrue(body.contains("continuingSessions.remove("), """
            finishRevisions no longer consumes the continuation mark, so a \
            continuation is filed as a new version of the reply it extended.
            """)
        XCTAssertTrue(body.contains("MessageRevisions.applyingEdit("), """
            the continuation branch must sync the extended text into the active \
            revision — exactly what an in-place edit does, and for the same \
            reason: stepping away and back reloads content from the revision.
            """)
    }

    /// The mark has the same ordering hazard as the seed, and it is invisible:
    /// placed before `stop`, the turn exit inside stop consumes it and the
    /// continuation files itself as a new version anyway.
    func testTheContinuationMarkIsSetAfterTheTurnItSupersedesHasStopped() throws {
        let source = SourceScan.source("Services/ChatTurnEngine.swift", from: #filePath)
        let body = try XCTUnwrap(
            SourceScan.declarationBody(from: "func continueReply", in: source),
            "continueReply moved — repoint this scan")
        let stopAt = try XCTUnwrap(body.range(of: "stop(sessionId: sessionId)")?.upperBound,
                                   "continueReply no longer supersedes the running turn")
        XCTAssertTrue(body[stopAt...].contains("appState.markContinuing("), """
            markContinuing must come AFTER stop(sessionId:) — stop is a turn \
            exit, so a mark placed before it is consumed immediately and the \
            continuation is recorded as a new version.
            """)
    }

    /// Every turn EXIT applies it, and nothing else does. A per-iteration call
    /// inside the agent loop would land the pager on the first tool round's
    /// bubble rather than on the answer.
    func testOnlyTurnExitsFinishRevisions() throws {
        let source = SourceScan.source("Services/ChatTurnEngine.swift", from: #filePath)
        XCTAssertEqual(SourceScan.count("appState.finishRevisions(", in: source), 3, """
            finishRevisions must be called from exactly the three turn exits — \
            stop(sessionId:), endTurn(sessionId:token:) and appendErrorNotice. \
            A fourth call inside runAgentLoop applies the seed to an \
            intermediate tool round.
            """)
        XCTAssertEqual(SourceScan.count("appState.commitRevision(", in: source), 0, """
            commitRevision was folded into finishRevisions — a surviving call \
            commits without ever seeding.
            """)
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
