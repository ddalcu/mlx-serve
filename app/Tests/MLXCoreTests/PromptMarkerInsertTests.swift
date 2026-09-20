import XCTest
@testable import MLXCore

/// A reference tile drops its marker into the prompt where the caret is, with
/// the spaces a person would have typed. `|` in the names is the caret.
final class PromptMarkerInsertTests: XCTestCase {

    private func insert(_ text: String, caret: Int, marker: String = "<Picture 1>") -> PromptMarkerInsert.Result {
        PromptMarkerInsert.insert(marker, into: text, replacing: caret..<caret)
    }

    // The five shapes from the spec.

    func testAfterASpaceAtTheEnd() {            // "the person in |"
        let r = insert("the person in ", caret: 14)
        XCTAssertEqual(r.text, "the person in <Picture 1>")
        XCTAssertEqual(r.cursor, r.text.count)
    }

    func testAfterASpaceBeforeAComma() {        // "the person in |, or the other"
        let r = insert("the person in , or the other", caret: 14)
        XCTAssertEqual(r.text, "the person in <Picture 1>, or the other")
        XCTAssertEqual(r.cursor, "the person in <Picture 1>".count)
    }

    func testAgainstAWordBeforeAComma() {       // "the person in|, or the other"
        let r = insert("the person in, or the other", caret: 13)
        XCTAssertEqual(r.text, "the person in <Picture 1>, or the other")
        XCTAssertEqual(r.cursor, "the person in <Picture 1>".count)
    }

    func testBetweenTwoWords() {                // "the person in|walks in."
        let r = insert("the person inwalks in.", caret: 13)
        XCTAssertEqual(r.text, "the person in <Picture 1> walks in.")
        XCTAssertEqual(r.cursor, "the person in <Picture 1> ".count, "the caret lands after the space")
    }

    func testRunsOfSpacesCollapseToOne() {      // "the person in   |    walks in."
        let r = insert("the person in       walks in.", caret: 16)
        XCTAssertEqual(r.text, "the person in <Picture 1> walks in.")
        XCTAssertEqual(r.cursor, "the person in <Picture 1> ".count)
    }

    // Edges the spec implies.

    func testAtTheStartOfTheText() {
        let r = insert("walks in.", caret: 0)
        XCTAssertEqual(r.text, "<Picture 1> walks in.")
        XCTAssertEqual(r.cursor, "<Picture 1> ".count)
    }

    func testIntoAnEmptyPrompt() {
        let r = insert("", caret: 0)
        XCTAssertEqual(r.text, "<Picture 1>")
        XCTAssertEqual(r.cursor, r.text.count)
    }

    /// A newline is a boundary, not a gap: nothing is put on either side of it
    /// and it is never eaten as whitespace.
    func testALineBreakIsLeftAlone() {
        let end = insert("first line\n", caret: 11)
        XCTAssertEqual(end.text, "first line\n<Picture 1>")
        let start = insert("first line\nsecond", caret: 11)
        XCTAssertEqual(start.text, "first line\n<Picture 1> second")
        let before = insert("first line\nsecond", caret: 10)
        XCTAssertEqual(before.text, "first line <Picture 1>\nsecond")
        XCTAssertEqual(before.cursor, "first line <Picture 1>".count)
    }

    /// A selection is what the marker REPLACES, the way typing would.
    func testASelectionIsReplaced() {
        let r = PromptMarkerInsert.insert("<Audio 1>", into: "the man speaks", replacing: 4..<7)
        XCTAssertEqual(r.text, "the <Audio 1> speaks")
        XCTAssertEqual(r.cursor, "the <Audio 1> ".count)
    }

    /// Without a caret the marker goes on the end, one space away, and a
    /// prompt that already ends in a space does not get two.
    func testAppendingGoesOnTheEndWithOneSpace() {
        XCTAssertEqual(PromptMarkerInsert.append("<Video 2>", to: "a chase").text, "a chase <Video 2>")
        XCTAssertEqual(PromptMarkerInsert.append("<Video 2>", to: "a chase  ").text, "a chase <Video 2>")
        XCTAssertEqual(PromptMarkerInsert.append("<Video 2>", to: "").text, "<Video 2>")
    }

    /// A caret past the end (a stale selection) is clamped, not a crash.
    func testAnOutOfRangeCaretIsClamped() {
        let r = insert("short", caret: 99)
        XCTAssertEqual(r.text, "short <Picture 1>")
    }
}
