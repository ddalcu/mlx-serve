import XCTest
@testable import MLXCore

/// A note typed while a turn runs is handed to the agent at its next step
/// boundary as the next user message. One note per chat, fired once.
final class SteeringNotesTests: XCTestCase {

    /// The row lays out a bounded prefix; the rest is counted, not rendered.
    func testRowPreviewIsBoundedAndCountsTheRest() {
        let short = SteeringNoteRow.preview("short", limit: 10)
        XCTAssertEqual(short.text, "short")
        XCTAssertEqual(short.omitted, 0)
        let long = SteeringNoteRow.preview(String(repeating: "x", count: 25), limit: 10)
        XCTAssertEqual(long.text, String(repeating: "x", count: 10))
        XCTAssertEqual(long.omitted, 15)
    }

    func testSetStoresTrimmedTextPerSession() {
        var notes = SteeringNotes()
        let a = UUID(), b = UUID()
        notes.append("  the VPN is off now, rerun the same command  ", for: a)
        XCTAssertEqual(notes.note(for: a), "the VPN is off now, rerun the same command")
        XCTAssertNil(notes.note(for: b), "another chat's note never leaks across sessions")
    }

    /// A second note joins the first, one blank line between: the user is
    /// adding a thought, not taking the first one back.
    func testASecondNoteIsAppendedWithOneBlankLine() {
        var notes = SteeringNotes()
        let s = UUID()
        notes.append("This is a text.", for: s)
        notes.append("And this is another one.", for: s)
        XCTAssertEqual(notes.note(for: s), "This is a text.\n\nAnd this is another one.")
    }

    func testJoinAddsOnlyTheLineBreaksThatAreMissing() {
        XCTAssertEqual(SteeringNotes.joined("A.", "B."), "A.\n\nB.")
        XCTAssertEqual(SteeringNotes.joined("A.\n", "\nB."), "A.\n\nB.")
        XCTAssertEqual(SteeringNotes.joined("A.\n\n", "\n\nB.\n\n"), "A.\n\nB.")
        XCTAssertEqual(SteeringNotes.joined("A.\n\nB.", "\nC.\n"), "A.\n\nB.\n\nC.")
        XCTAssertEqual(SteeringNotes.joined("", "B."), "B.")
        XCTAssertEqual(SteeringNotes.joined("A.", "  \n"), "A.")
    }

    func testBlankTextChangesNothing() {
        var notes = SteeringNotes()
        let s = UUID()
        notes.append("   \n", for: s)
        XCTAssertNil(notes.note(for: s), "nothing to say, nothing stored")
        notes.append("something", for: s)
        notes.append("   \n", for: s)
        XCTAssertEqual(notes.note(for: s), "something")
    }

    func testTakeFiresOnce() {
        var notes = SteeringNotes()
        let s = UUID()
        notes.append("use port 8081", for: s)
        XCTAssertEqual(notes.take(for: s), "use port 8081")
        XCTAssertNil(notes.take(for: s), "a note that fired is gone")
        XCTAssertNil(notes.note(for: s))
    }

    func testClearDropsTheNote() {
        var notes = SteeringNotes()
        let s = UUID()
        notes.append("never mind", for: s)
        notes.clear(for: s)
        XCTAssertNil(notes.note(for: s))
    }

    func testPausedNoteWaitsForCompositionToCommitOrCancelAndRestoresOnce() {
        for draft in ["日本", ""] {
            var notes = SteeringNotes()
            let session = UUID()
            notes.append("英語で答えて", for: session)
            notes.pause(for: session)

            XCTAssertNil(notes.take(for: session), "Pause removes the agent's sending ownership")
            XCTAssertNil(notes.restore(for: session, into: "", hasMarkedText: true))
            XCTAssertNil(notes.restore(for: session, into: "にほん", hasMarkedText: true))
            XCTAssertEqual(notes.restoringNote(for: session), "英語で答えて")
            XCTAssertEqual(notes.composerNote(for: session), "英語で答えて")

            let expected = draft.isEmpty ? "英語で答えて" : "英語で答えて\n\n日本"
            XCTAssertEqual(notes.restore(for: session, into: draft, hasMarkedText: false), expected)
            XCTAssertNil(notes.restoringNote(for: session))
            XCTAssertNil(notes.composerNote(for: session))
            XCTAssertNil(notes.restore(for: session, into: expected, hasMarkedText: false))
        }
    }

    func testRepeatedPausesPreserveOrderAndKeepNewSendingNotesSeparate() {
        var notes = SteeringNotes()
        let session = UUID()
        notes.append("first", for: session)
        notes.pause(for: session)
        notes.pause(for: session)
        notes.append("second", for: session)
        notes.pause(for: session)
        notes.append("third", for: session)

        XCTAssertEqual(notes.restoringNote(for: session), "first\n\nsecond")
        XCTAssertEqual(notes.composerNote(for: session), "first\n\nsecond\n\nthird")
        XCTAssertEqual(notes.take(for: session), "third")
        XCTAssertEqual(notes.restoringNote(for: session), "first\n\nsecond")
        XCTAssertEqual(notes.restore(for: session, into: "draft", hasMarkedText: false),
                       "first\n\nsecond\n\ndraft")
    }

    func testSessionSwitchDoesNotConsumeAnotherSessionsRestoringNote() {
        var notes = SteeringNotes()
        let a = UUID(), b = UUID()
        notes.append("A paused", for: a)
        notes.pause(for: a)
        notes.append("B sending", for: b)

        XCTAssertNil(notes.restore(for: b, into: "B draft", hasMarkedText: false))
        XCTAssertEqual(notes.composerNote(for: b), "B sending")
        XCTAssertEqual(notes.restoringNote(for: a), "A paused")
        XCTAssertEqual(notes.restore(for: a, into: "A draft", hasMarkedText: false),
                       "A paused\n\nA draft")
        XCTAssertEqual(notes.take(for: b), "B sending")
    }

    func testRetainDropsBothNoteKindsWithoutAnActiveTurn() {
        var notes = SteeringNotes()
        let retained = UUID(), deleted = UUID()
        for session in [retained, deleted] {
            notes.append("paused", for: session)
            notes.pause(for: session)
            notes.append("sending", for: session)
        }

        notes.retain(only: [retained])
        XCTAssertNil(notes.note(for: deleted))
        XCTAssertNil(notes.restoringNote(for: deleted))
        XCTAssertNil(notes.composerNote(for: deleted))
        XCTAssertNil(notes.restore(for: deleted, into: "draft", hasMarkedText: false))
        XCTAssertEqual(notes.note(for: retained), "sending")
        XCTAssertEqual(notes.restoringNote(for: retained), "paused")

        notes.retain(only: [])
        XCTAssertNil(notes.take(for: retained))
        XCTAssertNil(notes.restoringNote(for: retained))
    }

    // MARK: - Composer Return while generating

    func testBareReturnWhileGeneratingQueuesANoteWhereTheFieldCanSteer() {
        XCTAssertEqual(ComposerKey.onReturn(shift: false, isIdle: false, canSteer: true), .steer,
                       "while this chat generates, Return hands the text to the agent's next step")
        XCTAssertEqual(ComposerKey.onReturn(shift: false, isIdle: true, canSteer: true), .send,
                       "an idle chat sends as before")
        XCTAssertEqual(ComposerKey.onReturn(shift: true, isIdle: false, canSteer: true), .newline)
    }
}
