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
            "intro\n```html\n<b>hi</b>\n```\ntail",
            "```html\n<b>half",
        ]
        for src in sources {
            let joined = segs(src).map { seg -> String in
                switch seg {
                case .prose(let t): return t
                case .code(_, let c): return c
                case .html(_, let c): return c
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


    // MARK: - HTML artifact fences
    //
    // A closed fence whose language renders — html/svg — becomes its own
    // segment so the transcript can mount a live document for it instead of a
    // code block. Everything about WHEN that is safe lives here, because the
    // view must not be the thing deciding whether a half-streamed document is
    // ready to execute.

    func testClosedHtmlFenceIsItsOwnSegment() {
        XCTAssertEqual(segs("```html\n<b>hi</b>\n```"),
                       [.html(language: "html", code: "<b>hi</b>")])
    }

    func testHtmlSegmentSitsBetweenProseRuns() {
        XCTAssertEqual(segs("before\n```html\n<b>hi</b>\n```\nafter"), [
            .prose("before"),
            .html(language: "html", code: "<b>hi</b>"),
            .prose("after"),
        ])
    }

    func testUnterminatedHtmlFenceStaysCode() {
        // THE streaming rule. Every html reply passes through a half-written
        // document on its way to a closed fence; mounting a web view for one
        // executes a script whose function bodies are still arriving, and
        // reloads it on every token after that. Code until the fence closes.
        XCTAssertEqual(segs("```html\n<script>for (;;"),
                       [.code(language: "html", code: "<script>for (;;")])
    }

    func testSvgFenceRendersToo() {
        let svg = "<svg viewBox=\"0 0 2 2\"><rect width=\"2\" height=\"2\"/></svg>"
        XCTAssertEqual(segs("```svg\n" + svg + "\n```"),
                       [.html(language: "svg", code: svg)])
    }

    func testFenceLabelCaseAndSpacingDoNotMatter() {
        // Models write ``` HTML and ```Html as readily as ```html.
        XCTAssertEqual(segs("```HTML\n<i>x</i>\n```"),
                       [.html(language: "HTML", code: "<i>x</i>")])
        XCTAssertEqual(segs("``` Html \n<i>x</i>\n```"),
                       [.html(language: "Html", code: "<i>x</i>")])
    }

    func testHtmlLanguageIsPreservedVerbatim() {
        // The header shows what the model wrote, and the code arm picks its
        // lexer from it — same contract as `.code`.
        guard case .html(let language, _)? = segs("```htm\n<b>x</b>\n```").first else {
            return XCTFail("expected an html segment")
        }
        XCTAssertEqual(language, "htm")
    }

    func testOtherLanguagesAreUnaffected() {
        for fence in ["swift", "js", "python", "json", "xml", "", "markdown"] {
            let out = segs("```\(fence)\n<b>x</b>\n```")
            XCTAssertEqual(out, [.code(language: fence, code: "<b>x</b>")],
                           "fence \(fence.debugDescription) must stay a code block")
        }
    }

    func testHtmlFenceWithoutMarkupStaysCode() {
        // A model explaining HTML in prose inside an html fence has nothing to
        // render; a blank web view where the answer should be is worse than the
        // text.
        XCTAssertEqual(segs("```html\njust a sentence\n```"),
                       [.code(language: "html", code: "just a sentence")])
        XCTAssertEqual(segs("```html\n```"),
                       [.code(language: "html", code: "")])
    }

}
