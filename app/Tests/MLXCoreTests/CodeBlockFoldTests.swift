import XCTest
@testable import MLXCore

/// Folding a long code block to its first lines, behind a "Show more".
final class CodeBlockFoldTests: XCTestCase {

    private func code(lines n: Int) -> String {
        (1...n).map { "line \($0)" }.joined(separator: "\n")
    }

    func testAShortBlockShowsWhole() {
        let block = code(lines: 5)
        XCTAssertFalse(CodeBlockFold.folds(block))
        XCTAssertEqual(CodeBlockFold.visible(block, expanded: false), block)
    }

    /// A "Show more" that would reveal a line or two is a dead control.
    func testABlockJustOverTheLimitIsLeftWhole() {
        let block = code(lines: CodeBlockFold.collapsedLines + CodeBlockFold.foldMargin - 1)
        XCTAssertFalse(CodeBlockFold.folds(block))
    }

    func testALongBlockShowsItsFirstLinesUntilExpanded() {
        let block = code(lines: CodeBlockFold.collapsedLines + CodeBlockFold.foldMargin)
        XCTAssertTrue(CodeBlockFold.folds(block))
        XCTAssertEqual(CodeBlockFold.visible(block, expanded: false), code(lines: CodeBlockFold.collapsedLines))
        XCTAssertEqual(CodeBlockFold.visible(block, expanded: true), block)
    }

    /// A block still streaming folds as soon as it is long enough; its unfinished last line stays hidden.
    func testAStreamingBlockFoldsOnceItIsLong() {
        let block = code(lines: 40) + "\nlet partial = "
        XCTAssertEqual(CodeBlockFold.visible(block, finished: false, expanded: false), code(lines: CodeBlockFold.collapsedLines))
    }

    /// The line being written does not count yet, or the closing fence's first backtick folds a block of 22.
    func testAnOpenBlockCountsOnlyItsFinishedLines() {
        let lines = CodeBlockFold.collapsedLines + CodeBlockFold.foldMargin - 1
        let block = code(lines: lines) + "\n`"
        XCTAssertEqual(CodeBlockFold.lineCount(block, finished: false), lines)
        XCTAssertFalse(CodeBlockFold.folds(block, finished: false))
        XCTAssertTrue(CodeBlockFold.folds(block, finished: true), "a fence the model never closed is just a line")
    }

    func testTheFoldNamesTheBlocksLength() {
        XCTAssertEqual(CodeBlockFold.linesTotal(124), "124 lines total")
    }

    /// An opened block stays open through a transcript rebuild, per reply and per block.
    @MainActor
    func testFoldMemoryKeepsEachBlockOfEachReply() {
        let store = FoldStore()
        let reply = UUID(), other = UUID()
        store.set(reply, block: 2, expanded: true)
        XCTAssertTrue(store.isExpanded(reply, block: 2))
        XCTAssertFalse(store.isExpanded(reply, block: 3))
        XCTAssertFalse(store.isExpanded(other, block: 2))
        XCTAssertFalse(store.isExpanded(reply), "a long turn's own fold is separate")
        store.clear()
        XCTAssertFalse(store.isExpanded(reply, block: 2))
    }

    /// A fresh context per row render must not count as a change, or every block re-lexes per streamed token.
    func testRowContextsAlwaysCompareEqual() {
        let a = CodeBlockRow(willResize: {}, didResize: {}, isExpanded: { _ in true }, setExpanded: { _, _ in })
        let b = CodeBlockRow(willResize: {}, didResize: {}, isExpanded: { _ in false }, setExpanded: { _, _ in })
        XCTAssertEqual(a, b)
    }

    /// The count is exact: code never wraps, and an empty line is a line.
    func testLinesAreCountedExactly() {
        XCTAssertEqual(CodeBlockFold.lineCount(""), 1)
        XCTAssertEqual(CodeBlockFold.lineCount("a"), 1)
        XCTAssertEqual(CodeBlockFold.lineCount("a\n\nb\n"), 4)
    }

    /// The status slot spins only while the block is open AND the reply is still coming.
    func testABlockStreamsOnlyWhileItsFenceIsOpenAndTheReplyRuns() {
        XCTAssertEqual(CodeBlockStatus.of(closed: false, replyStreaming: true), .streaming)
        XCTAssertEqual(CodeBlockStatus.of(closed: true, replyStreaming: true), .copy, "closed while the prose goes on")
        XCTAssertEqual(CodeBlockStatus.of(closed: false, replyStreaming: false), .copy, "a fence the model never closed")
    }

    /// `\r\n` is one Character, so a count by Characters would see one long line.
    func testWindowsLineEndingsFoldLikeAnyOther() {
        let block = (1...30).map { "line \($0)" }.joined(separator: "\r\n")
        XCTAssertEqual(CodeBlockFold.lineCount(block), 30)
        let folded = CodeBlockFold.visible(block, expanded: false)
        XCTAssertEqual(CodeBlockFold.lineCount(folded), CodeBlockFold.collapsedLines)
        XCTAssertFalse(folded.hasSuffix("\r"), "the cut takes the whole line ending")
    }
}
