import XCTest
@testable import MLXCore

/// Deciding whether a user turn is long enough to fold away.
final class LongUserTurnTests: XCTestCase {

    /// Roughly the real bubble: ~80 characters across.
    private let perLine = 80

    func testAShortTurnIsLeftAlone() {
        XCTAssertFalse(LongUserTurn.isCollapsible("Why is the sky blue?", charsPerLine: perLine))
    }

    func testHardNewlinesPastTheLimitFold() {
        let text = (1...40).map { "line \($0)" }.joined(separator: "\n")
        XCTAssertTrue(LongUserTurn.isCollapsible(text, charsPerLine: perLine))
    }

    /// The case a newline count cannot see: one pasted paragraph with no line
    /// breaks in it at all.
    func testOneLongParagraphFolds() {
        let text = String(repeating: "a", count: perLine * 40)
        XCTAssertTrue(LongUserTurn.isCollapsible(text, charsPerLine: perLine))
    }

    /// A "show more" that reveals nothing is a dead control, and the line count
    /// is an ESTIMATE — so a turn has to clear the limit by a margin before it
    /// earns one.
    func testATurnJustOverTheLimitIsLeftAlone() {
        let text = (1...(LongUserTurn.collapsedLineLimit + 1)).map { "line \($0)" }.joined(separator: "\n")
        XCTAssertFalse(LongUserTurn.isCollapsible(text, charsPerLine: perLine))
    }

    func testBlankLinesBetweenParagraphsCount() {
        XCTAssertEqual(LongUserTurn.estimatedLines(of: "a\n\n\nb", charsPerLine: perLine), 4)
    }

    func testAWrappedLineCountsForEveryRowItTakes() {
        XCTAssertEqual(LongUserTurn.estimatedLines(of: String(repeating: "x", count: 161),
                                                   charsPerLine: perLine), 3)
    }

    func testAnEmptyTurnIsOneLineNotZero() {
        XCTAssertEqual(LongUserTurn.estimatedLines(of: "", charsPerLine: perLine), 1)
    }

    // MARK: - How many characters fit across the bubble

    func testAWiderBubbleFitsMoreCharacters() {
        let narrow = LongUserTurn.charsPerLine(textWidth: 300, fontSize: 14)
        let wide = LongUserTurn.charsPerLine(textWidth: 600, fontSize: 14)
        XCTAssertGreaterThan(wide, narrow)
    }

    func testABiggerFontFitsFewerCharacters() {
        let small = LongUserTurn.charsPerLine(textWidth: 600, fontSize: 12)
        let large = LongUserTurn.charsPerLine(textWidth: 600, fontSize: 20)
        XCTAssertLessThan(large, small)
    }

    /// A zero width arrives before the first layout pass; it must not divide by
    /// zero or report a line as infinitely long.
    func testADegenerateWidthStillYieldsAUsableLine() {
        XCTAssertGreaterThan(LongUserTurn.charsPerLine(textWidth: 0, fontSize: 14), 0)
        XCTAssertGreaterThan(LongUserTurn.charsPerLine(textWidth: 600, fontSize: 0), 0)
    }
}
