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
}
