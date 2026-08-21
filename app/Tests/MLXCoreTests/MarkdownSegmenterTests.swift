import XCTest
@testable import MLXCore

/// Splits an assistant reply into prose runs and fenced code blocks.
///
/// Why a separate pass from `MarkdownText`'s block parser: prose (including
/// tables) is rendered by ONE NSTextView per run so drag-selection crosses
/// paragraphs, lists, and tables, while each code block becomes its own view
/// with a gutter and a copy button. Consecutive prose blocks must therefore
/// stay in a SINGLE segment — splitting per block is what used to break
/// selection at every boundary.
final class MarkdownSegmenterTests: XCTestCase {

    private func segs(_ s: String) -> [MarkdownSegmenter.Segment] {
        MarkdownSegmenter.segments(s)
    }

    func testPlainTextIsOneProseSegment() {
        XCTAssertEqual(segs("hello world"), [.prose("hello world")])
    }

    func testConsecutiveProseBlocksStayInOneSegment() {
        // Headings, paragraphs and lists between two fences are ONE run — the
        // whole point of segmenting at fences rather than at blocks.
        let src = "# Title\n\npara one\n\n- a\n- b"
        XCTAssertEqual(segs(src), [.prose(src)])
    }

    func testProseCodeProseSplitsInOrder() {
        let src = "before\n```swift\nlet a = 1\n```\nafter"
        XCTAssertEqual(segs(src), [
            .prose("before"),
            .code(language: "swift", code: "let a = 1"),
            .prose("after"),
        ])
    }

    func testFenceLanguageIsCaptured() {
        XCTAssertEqual(segs("```tsx\nx\n```"), [.code(language: "tsx", code: "x")])
    }

    func testFenceWithoutLanguageHasEmptyLabel() {
        XCTAssertEqual(segs("```\nx\n```"), [.code(language: "", code: "x")])
    }

    func testLeadingAndTrailingFencesProduceNoEmptyProse() {
        // An empty text view between two blocks shows up as a stray gap.
        XCTAssertEqual(segs("```\na\n```"), [.code(language: "", code: "a")])
        XCTAssertEqual(segs("```\na\n```\n```\nb\n```"), [
            .code(language: "", code: "a"),
            .code(language: "", code: "b"),
        ])
    }

    func testWhitespaceOnlyProseRunsAreDropped() {
        let out = segs("```\na\n```\n   \n\n```\nb\n```")
        XCTAssertEqual(out, [
            .code(language: "", code: "a"),
            .code(language: "", code: "b"),
        ], "blank filler between fences must not become an empty prose view")
    }

    func testUnterminatedFenceStillRendersAsCode() {
        // Every streaming reply passes through this state on its way to a
        // closed fence; the half-written block must render as code, not as
        // prose that reflows into a code block a keystroke later.
        XCTAssertEqual(segs("intro\n```python\ndef f():"), [
            .prose("intro"),
            .code(language: "python", code: "def f():"),
        ])
    }

    func testEmptyCodeBlockIsKept() {
        // A fence pair with nothing inside is what an empty file looks like;
        // dropping it would silently lose the model's answer.
        XCTAssertEqual(segs("```\n```"), [.code(language: "", code: "")])
    }

    func testEmptySourceProducesNoSegments() {
        XCTAssertEqual(segs(""), [])
        XCTAssertEqual(segs("   \n  "), [])
    }

    func testTextIsPreservedAcrossSegments() {
        // Class guard: whatever the fence layout, every non-fence line of the
        // source must survive into some segment. A segmenter that silently
        // drops a line loses model output with nothing to show for it.
        let sources = [
            "a\n```\nb\n```\nc",
            "```\na\n```b\n",
            "no fences at all",
            "```js\nconst a = 1\n```\n\ntail\n\n```\nx\n```",
            "```\nunterminated",
        ]
        for src in sources {
            let joined = segs(src).map { seg -> String in
                switch seg {
                case .prose(let t): return t
                case .code(_, let c): return c
                }
            }.joined(separator: "\n")
            for line in src.components(separatedBy: "\n")
            where !line.hasPrefix("```") && !line.trimmingCharacters(in: .whitespaces).isEmpty {
                XCTAssertTrue(joined.contains(line),
                              "line \(line.debugDescription) vanished from \(src.debugDescription)")
            }
        }
    }

    func testPipeWithoutSeparatorStaysOneProseSegment() {
        XCTAssertEqual(segs("a | b\nplain"), [.prose("a | b\nplain")])
    }

    func testTableStaysInsideItsSurroundingProseSegment() {
        // Unlike a fence, a table is NOT a segment boundary — it renders as
        // an NSTextTable inside the same NSTextView as the surrounding
        // prose, via MarkdownText.parseBlocks, so drag-selection can span
        // prose and table together.
        let s = "before\n| a | b |\n|---|---|\n| 1 | 2 |\nafter"
        XCTAssertEqual(segs(s), [.prose(s)])
    }

    func testTableThenCodeFenceSplitsOnlyAtTheFence() {
        let s = "| a | b |\n|---|---|\n| 1 | 2 |\n```\nx\n```"
        XCTAssertEqual(segs(s), [
            .prose("| a | b |\n|---|---|\n| 1 | 2 |"),
            .code(language: "", code: "x"),
        ])
    }

}
