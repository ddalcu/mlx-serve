import XCTest
@testable import MLXCore

/// Picking a tempo from the menu beside the BPM box did nothing visible while
/// the box had focus: the value changed, the box kept the old number, and it
/// only caught up when focus left. The box declined every value while focused,
/// to protect the caret from being moved mid-edit.
final class NumberFieldRepaintTests: XCTestCase {

    private let bpm = 30...300

    func testAnUnfocusedBoxAlwaysShowsTheValue() {
        XCTAssertTrue(NumberFieldRepaint.shouldRepaint(text: "90", value: 120, range: bpm, focused: false))
    }

    /// The menu wrote 120 while the box said 90: nobody typed that, so it has
    /// to appear.
    func testAValueThatIsNotWhatTheBoxSaysIsPaintedEvenWhileTyping() {
        XCTAssertTrue(NumberFieldRepaint.shouldRepaint(text: "90", value: 120, range: bpm, focused: true))
    }

    /// The box's own typing round-trips: repainting here is what moves the
    /// caret to the end of the field on every keystroke.
    func testTheBoxsOwnValueIsLeftAlone() {
        XCTAssertFalse(NumberFieldRepaint.shouldRepaint(text: "120", value: 120, range: bpm, focused: true))
    }

    /// Empty is the Auto row, and nil is what it parses to: a cleared box that
    /// set the value to nil must not be repainted out from under the caret.
    func testClearingTheBoxIsNotAnOutsideChange() {
        XCTAssertFalse(NumberFieldRepaint.shouldRepaint(text: "", value: nil, range: bpm, focused: true))
        XCTAssertTrue(NumberFieldRepaint.shouldRepaint(text: "", value: 120, range: bpm, focused: true),
                      "the menu picking a tempo into an empty box still has to show")
    }

    /// The reader CLAMPS, so a half-typed "1" is already the value 30 and the
    /// box must be left alone: the bar is agreement with what the field itself
    /// reads, never with the digits on screen.
    func testAHalfTypedNumberIsJudgedByWhatTheFieldWouldRead() {
        XCTAssertEqual(SeedText.parse("1", in: bpm), 30, "the reader clamps into the range")
        XCTAssertFalse(NumberFieldRepaint.shouldRepaint(text: "1", value: 30, range: bpm, focused: true))
        XCTAssertTrue(NumberFieldRepaint.shouldRepaint(text: "1", value: 120, range: bpm, focused: true))
    }
}
