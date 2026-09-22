import XCTest
@testable import MLXCore

/// A note typed while a turn runs is handed to the agent at its next step
/// boundary as the next user message. One note per chat, fired once.
final class SteeringNotesTests: XCTestCase {

    func testSetStoresTrimmedTextPerSession() {
        var notes = SteeringNotes()
        let a = UUID(), b = UUID()
        notes.set("  the VPN is off now, rerun the same command  ", for: a)
        XCTAssertEqual(notes.note(for: a), "the VPN is off now, rerun the same command")
        XCTAssertNil(notes.note(for: b), "another chat's note never leaks across sessions")
    }

    func testANewNoteReplacesTheOldOne() {
        var notes = SteeringNotes()
        let s = UUID()
        notes.set("first", for: s)
        notes.set("second", for: s)
        XCTAssertEqual(notes.note(for: s), "second")
    }

    func testBlankTextClearsInsteadOfStoringNothing() {
        var notes = SteeringNotes()
        let s = UUID()
        notes.set("something", for: s)
        notes.set("   \n", for: s)
        XCTAssertNil(notes.note(for: s))
    }

    func testTakeFiresOnce() {
        var notes = SteeringNotes()
        let s = UUID()
        notes.set("use port 8081", for: s)
        XCTAssertEqual(notes.take(for: s), "use port 8081")
        XCTAssertNil(notes.take(for: s), "a note that fired is gone")
        XCTAssertNil(notes.note(for: s))
    }

    func testClearDropsTheNote() {
        var notes = SteeringNotes()
        let s = UUID()
        notes.set("never mind", for: s)
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
