import XCTest
@testable import MLXCore

/// Turns whole-block syntax spans into per-line runs for the gutter renderer.
///
/// Splitting matters because the tokenizer works on the whole block (a block
/// comment or triple-quoted string legitimately crosses newlines) while the view
/// draws one row per line so the line numbers stay aligned. Every multi-line
/// span therefore has to be cut at each newline and re-emitted per row.
final class CodeLayoutTests: XCTestCase {

    private func plain(_ lines: [CodeLine]) -> [String] {
        lines.map { $0.runs.map(\.text).joined() }
    }

    func testLineCountMatchesSourceLines() {
        let out = CodeLayout.lines(code: "a\nb\nc", language: nil)
        XCTAssertEqual(out.count, 3)
        XCTAssertEqual(out.map(\.number), [1, 2, 3])
        XCTAssertEqual(plain(out), ["a", "b", "c"])
    }

    func testEachLineReassemblesToItsExactText() {
        // Class guard: whatever the highlighting, concatenating a row's runs
        // must give back that line verbatim — a renderer that drops or
        // duplicates a character is showing the user code they didn't write.
        let code = "// note\nfunc f(x: Int) -> String {\n    return \"a\\\"b\"  // t\n}"
        let out = CodeLayout.lines(code: code, language: .swift)
        XCTAssertEqual(plain(out), code.components(separatedBy: "\n"))
    }

    func testMultiLineCommentIsColoredOnEveryLineItCovers() {
        let code = "/* one\n   two */\nx"
        let out = CodeLayout.lines(code: code, language: .cFamily)
        XCTAssertEqual(out[0].runs.map(\.kind), [.comment])
        XCTAssertEqual(out[1].runs.map(\.kind), [.comment],
                       "the continuation line of a block comment must stay colored")
        XCTAssertEqual(out[2].runs.map(\.kind), [nil])
    }

    func testNewlinesAreNotPartOfAnyRun() {
        // A trailing "\n" inside a run would print as a blank row and push every
        // later line out of step with its gutter number.
        let out = CodeLayout.lines(code: "/* a\nb */", language: .cFamily)
        for line in out {
            for run in line.runs {
                XCTAssertFalse(run.text.contains("\n"), "run \(run.text.debugDescription) carries a newline")
            }
        }
    }

    func testUnhighlightedTextBecomesNilKindRuns() {
        let out = CodeLayout.lines(code: "let a = 1", language: .swift)
        let kinds = out[0].runs.map(\.kind)
        XCTAssertEqual(kinds.first, .keyword, "`let` is a keyword")
        XCTAssertTrue(kinds.contains(nil), "the gaps between tokens stay uncolored")
        XCTAssertTrue(kinds.contains(.number))
    }

    func testNoLanguageLeavesEverythingPlain() {
        let out = CodeLayout.lines(code: "func f() {}", language: nil)
        XCTAssertEqual(out[0].runs.map(\.kind), [nil])
        XCTAssertEqual(plain(out), ["func f() {}"])
    }

    func testEmptyLinesSurviveAsEmptyRows() {
        // Blank lines inside a block are meaningful spacing; dropping them
        // renumbers everything below.
        let out = CodeLayout.lines(code: "a\n\nb", language: nil)
        XCTAssertEqual(out.count, 3)
        XCTAssertEqual(plain(out), ["a", "", "b"])
    }

    func testEmptyCodeIsOneEmptyLine() {
        let out = CodeLayout.lines(code: "", language: nil)
        XCTAssertEqual(plain(out), [""])
    }

    func testNonAsciiLinesReassembleExactly() {
        // UTF-16 span offsets against Character-based line slicing is where an
        // emoji silently shifts every colour after it.
        let code = "// 🚀 go\nlet x = \"héllo\""
        let out = CodeLayout.lines(code: code, language: .swift)
        XCTAssertEqual(plain(out), code.components(separatedBy: "\n"))
    }
}
