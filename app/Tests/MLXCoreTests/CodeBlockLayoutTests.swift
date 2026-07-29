import XCTest
@testable import MLXCore

/// The code block's gutter geometry. Sized in code rather than by SwiftUI's
/// intrinsic width so every row's numbers share one column; that means the
/// column has to be provably wide enough for the largest number it will show.
final class CodeBlockLayoutTests: XCTestCase {

    func testGutterGrowsWithDigitCount() {
        let two = CodeBlockLayout.gutterWidth(lineCount: 12)
        let three = CodeBlockLayout.gutterWidth(lineCount: 120)
        let four = CodeBlockLayout.gutterWidth(lineCount: 1200)
        XCTAssertLessThan(two, three)
        XCTAssertLessThan(three, four)
    }

    func testShortBlocksStillReserveTwoDigits() {
        // A 3-line block and a 30-line one share a column width, so a block
        // that grows past 9 lines mid-stream doesn't visibly shift its code.
        XCTAssertEqual(CodeBlockLayout.gutterWidth(lineCount: 3),
                       CodeBlockLayout.gutterWidth(lineCount: 30))
    }

    func testGutterFitsItsWidestNumber() {
        // Monospaced digits are ~0.6em; the column must hold every digit plus
        // the trailing gap, or the highest line numbers clip.
        for count in [1, 9, 10, 99, 100, 999, 1000, 12345] {
            let needed = CGFloat(String(count).count) * CodeBlockLayout.fontSize * 0.62
            XCTAssertGreaterThan(CodeBlockLayout.gutterWidth(lineCount: count), needed,
                                 "gutter clips at \(count) lines")
        }
    }

    func testZeroLineCountIsSafe() {
        // An empty block still renders one row; the width math must not take
        // String(0).count down a degenerate path.
        XCTAssertGreaterThan(CodeBlockLayout.gutterWidth(lineCount: 0), 0)
    }
}
