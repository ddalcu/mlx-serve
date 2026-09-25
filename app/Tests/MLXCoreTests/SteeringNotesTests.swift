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

    // MARK: - Composer Return while generating

    func testBareReturnWhileGeneratingQueuesANoteWhereTheFieldCanSteer() {
        XCTAssertEqual(ComposerKey.onReturn(shift: false, isIdle: false, canSteer: true), .steer,
                       "while this chat generates, Return hands the text to the agent's next step")
        XCTAssertEqual(ComposerKey.onReturn(shift: false, isIdle: true, canSteer: true), .send,
                       "an idle chat sends as before")
        XCTAssertEqual(ComposerKey.onReturn(shift: true, isIdle: false, canSteer: true), .newline)
    }
}
