import XCTest
@testable import MLXCore

final class MarkdownTableTests: XCTestCase {

    private func lines(_ s: String) -> [String] { s.components(separatedBy: "\n") }

    func testGfmPipeTable() {
        let t = MarkdownTable.parse(lines: lines("| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"), start: 0)
        XCTAssertEqual(t?.headers, ["a", "b"])
        XCTAssertEqual(t?.rows, [["1", "2"], ["3", "4"]])
        XCTAssertEqual(t?.alignments, [.left, .left])
        XCTAssertEqual(t?.end, 4)
    }

    func testAlignmentColons() {
        let t = MarkdownTable.parse(lines: lines("| a | b | c |\n|:---|---:|:---:|"), start: 0)
        XCTAssertEqual(t?.alignments, [.left, .right, .center])
    }

    func testLooseFormWithoutLeadingPipe() {
        let t = MarkdownTable.parse(lines: lines("a | b\n--- | ---\n1 | 2"), start: 0)
        XCTAssertEqual(t?.headers, ["a", "b"])
        XCTAssertEqual(t?.rows, [["1", "2"]])
    }

    func testNoFalsePositiveOnStrayPipeWithoutSeparator() {
        XCTAssertNil(MarkdownTable.parse(lines: lines("use | to separate things\nnext line"), start: 0))
    }

    func testGfmHeaderOnlyTableHasNoRows() {
        let t = MarkdownTable.parse(lines: lines("| a | b |\n|---|---|"), start: 0)
        XCTAssertEqual(t?.rows, [])
        XCTAssertEqual(t?.end, 2)
    }

    func testTableNotAtLineZeroIsFoundAtItsStartIndex() {
        let ls = lines("intro\n| a | b |\n|---|---|\n| 1 | 2 |")
        XCTAssertNil(MarkdownTable.parse(lines: ls, start: 0))
        let t = MarkdownTable.parse(lines: ls, start: 1)
        XCTAssertEqual(t?.headers, ["a", "b"])
        XCTAssertEqual(t?.end, 4)
    }

    func testAsciiPseudoTable() {
        let ls = lines("Tip        Explanation\n─────────  ──────────────────\n`$@`       preserve args")
        let t = MarkdownTable.parse(lines: ls, start: 0)
        XCTAssertEqual(t?.headers, ["Tip", "Explanation"])
        XCTAssertEqual(t?.rows, [["`$@`", "preserve args"]])
        XCTAssertEqual(t?.alignments, [.left, .left])
    }

    func testAsciiPseudoTableRequiresAtLeastOneRow() {
        XCTAssertNil(MarkdownTable.parse(lines: lines("Tip        Explanation\n─────────  ──────────────────\n"), start: 0))
    }

    func testAsciiPseudoTableWrappedCellJoinsIntoPreviousRow() {
        let ls = lines("Tip        Explanation\n─────────  ──────────────────\n`$@`       preserve args with\n           spaces in them")
        let t = MarkdownTable.parse(lines: ls, start: 0)
        XCTAssertEqual(t?.rows, [["`$@`", "preserve args with spaces in them"]])
    }

    func testColumnWidthsSumToOne() {
        let w = MarkdownTable.columnWidths(headers: ["Tip", "Explanation"],
                                            rows: [["a", "b"], ["cc", "dddd"]])
        XCTAssertEqual(w.reduce(0, +), 1.0, accuracy: 0.0001)
        XCTAssertEqual(w.count, 2)
    }

    func testWiderColumnGetsMoreShare() {
        let w = MarkdownTable.columnWidths(
            headers: ["Tip", "Explanation"],
            rows: [["`$@`", "Use \"$@\" to preserve arguments with spaces"]])
        XCTAssertGreaterThan(w[1], w[0])
    }

    func testSingleColumnTakesTheWholeWidth() {
        XCTAssertEqual(MarkdownTable.columnWidths(headers: ["a"], rows: [["b"]]), [1.0])
    }

    func testEmptyColumnsStillGetAVisibleFloor() {
        let w = MarkdownTable.columnWidths(headers: ["a", ""], rows: [["x", ""], ["y", ""]])
        // Both columns bottom out at the same floor, so they split evenly.
        XCTAssertEqual(w[0], w[1], accuracy: 0.0001)
    }

    func testRaggedRowsDoNotCrashAndReturnOneFractionPerHeader() {
        let w = MarkdownTable.columnWidths(headers: ["a", "b", "c"], rows: [["x"], ["y", "z", "w", "extra"]])
        XCTAssertEqual(w.count, 3)
    }

    func testNoHeadersReturnsEmpty() {
        XCTAssertEqual(MarkdownTable.columnWidths(headers: [], rows: []), [])
    }
}
