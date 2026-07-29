import XCTest
@testable import MLXCore

/// The code-block tokenizer behind `CodeBlockView`.
///
/// Offsets are UTF-16 units because every consumer is an `NSRange` into an
/// `NSAttributedString` — a Character-based span would be off by one for every
/// emoji or combining mark earlier in the block, silently mis-coloring the rest
/// of the file.
final class SyntaxHighlighterTests: XCTestCase {

    /// Convenience: the substring a span covers, so assertions read as the text
    /// the user sees rather than a pair of integers.
    private func text(_ source: String, _ span: SyntaxSpan) -> String {
        let ns = source as NSString
        return ns.substring(with: NSRange(location: span.start, length: span.length))
    }

    private func spans(_ source: String, _ lang: SyntaxLanguage) -> [SyntaxSpan] {
        SyntaxHighlighter.spans(source, language: lang)
    }

    private func first(_ source: String, _ lang: SyntaxLanguage, kind: SyntaxKind) -> String? {
        spans(source, lang).first { $0.kind == kind }.map { text(source, $0) }
    }

    // MARK: - Fence → language

    func testFenceAliasesResolveToLanguages() {
        XCTAssertEqual(SyntaxLanguage(fence: "swift"), .swift)
        XCTAssertEqual(SyntaxLanguage(fence: "ts"), .javascript)
        XCTAssertEqual(SyntaxLanguage(fence: "tsx"), .javascript)
        XCTAssertEqual(SyntaxLanguage(fence: "py"), .python)
        XCTAssertEqual(SyntaxLanguage(fence: "sh"), .shell)
        XCTAssertEqual(SyntaxLanguage(fence: "zsh"), .shell)
        XCTAssertEqual(SyntaxLanguage(fence: "c++"), .cFamily)
        XCTAssertEqual(SyntaxLanguage(fence: "objective-c"), .cFamily)
    }

    func testFenceMatchIsCaseAndWhitespaceInsensitive() {
        // Models emit "```Swift", "```JSON", and "```python " freely.
        XCTAssertEqual(SyntaxLanguage(fence: "Swift"), .swift)
        XCTAssertEqual(SyntaxLanguage(fence: "  JSON "), .json)
    }

    func testUnknownAndEmptyFencesHaveNoLanguage() {
        // An unfenced block must not be guessed at: coloring random prose as
        // code is worse than leaving it plain.
        XCTAssertNil(SyntaxLanguage(fence: ""))
        XCTAssertNil(SyntaxLanguage(fence: "brainfuck"))
    }

    // MARK: - Comments

    func testLineCommentStopsAtEndOfLine() {
        let src = "let a = 1 // note\nlet b = 2"
        let comment = spans(src, .swift).first { $0.kind == .comment }
        XCTAssertEqual(comment.map { text(src, $0) }, "// note",
                       "a line comment must not swallow the following line")
    }

    func testBlockCommentSpansLines() {
        let src = "a\n/* one\n   two */\nb"
        XCTAssertEqual(first(src, .cFamily, kind: .comment), "/* one\n   two */")
    }

    func testUnterminatedBlockCommentRunsToEndOfSource() {
        // Streaming shows half-written code constantly; the tokenizer must not
        // drop the tail or crash when the closing delimiter never arrives.
        let src = "code\n/* still typing"
        XCTAssertEqual(first(src, .cFamily, kind: .comment), "/* still typing")
    }

    func testHashCommentInShellAndPython() {
        XCTAssertEqual(first("echo hi # say hi", .shell, kind: .comment), "# say hi")
        XCTAssertEqual(first("x = 1  # count", .python, kind: .comment), "# count")
    }

    // MARK: - Strings

    func testStringWithEscapedQuoteStaysOneSpan() {
        let src = #"let s = "a \" b" + t"#
        XCTAssertEqual(first(src, .swift, kind: .string), #""a \" b""#)
    }

    func testCommentMarkerInsideStringIsNotAComment() {
        let src = #"let url = "https://example.com" // real comment"#
        let all = spans(src, .swift)
        XCTAssertEqual(all.first { $0.kind == .string }.map { text(src, $0) }, #""https://example.com""#,
                       "the // inside the literal must not end the string")
        XCTAssertEqual(all.first { $0.kind == .comment }.map { text(src, $0) }, "// real comment")
    }

    func testQuoteInsideCommentDoesNotOpenAString() {
        // The classic lexer bug: an apostrophe in a comment swallowing the rest
        // of the file as a string literal.
        let src = "// don't do this\nlet a = 1"
        XCTAssertNil(spans(src, .swift).first { $0.kind == .string },
                     "an apostrophe inside a comment is just text")
    }

    func testUnterminatedStringStopsAtEndOfLine() {
        // Single-quote/double-quote literals don't span lines in these
        // languages, so a missing close must not eat the whole block.
        let src = "let a = \"oops\nlet b = 2"
        XCTAssertEqual(first(src, .swift, kind: .string), "\"oops")
    }

    func testPythonTripleQuotedStringSpansLines() {
        let src = "doc = \"\"\"line one\nline two\"\"\"\nx = 1"
        XCTAssertEqual(first(src, .python, kind: .string), "\"\"\"line one\nline two\"\"\"")
    }

    // MARK: - Keywords, types, numbers

    func testKeywordsMatchWholeWordsOnly() {
        let src = "functional func f"
        let keywords = spans(src, .swift).filter { $0.kind == .keyword }.map { text(src, $0) }
        XCTAssertEqual(keywords, ["func"],
                       "`functional` contains `func` but is an identifier")
    }

    func testNumberLiteralForms() {
        let src = "a = 42; b = 0xFF; c = 3.14; d = 1e-9"
        let nums = spans(src, .cFamily).filter { $0.kind == .number }.map { text(src, $0) }
        XCTAssertEqual(nums, ["42", "0xFF", "3.14", "1e-9"])
    }

    func testIdentifierWithDigitsIsNotANumber() {
        let src = "let utf8 = x2"
        XCTAssertTrue(spans(src, .swift).filter { $0.kind == .number }.isEmpty,
                      "digits inside an identifier are part of the name")
    }

    func testCapitalizedIdentifiersReadAsTypes() {
        let src = "let v: MyType = other"
        XCTAssertEqual(first(src, .swift, kind: .type), "MyType")
    }

    func testCallSitesReadAsFunctions() {
        let src = "result = doWork(a)"
        XCTAssertEqual(first(src, .javascript, kind: .function), "doWork")
    }

    // MARK: - JSON

    func testJsonKeysAndValuesAreDistinguished() {
        let src = #"{"name": "mlx", "n": 3, "ok": true}"#
        let all = spans(src, .json)
        XCTAssertEqual(all.first { $0.kind == .property }.map { text(src, $0) }, #""name""#,
                       "the string before a colon is a key, not a value")
        XCTAssertEqual(all.first { $0.kind == .string }.map { text(src, $0) }, #""mlx""#)
        XCTAssertEqual(all.first { $0.kind == .number }.map { text(src, $0) }, "3")
        XCTAssertEqual(all.first { $0.kind == .keyword }.map { text(src, $0) }, "true")
    }

    // MARK: - Markup

    func testMarkupTagsAndAttributes() {
        let src = #"<link rel="sitemap" href="/x.xml"/>"#
        let all = spans(src, .markup)
        XCTAssertEqual(all.first { $0.kind == .keyword }.map { text(src, $0) }, "link")
        XCTAssertEqual(all.first { $0.kind == .property }.map { text(src, $0) }, "rel")
        XCTAssertEqual(all.first { $0.kind == .string }.map { text(src, $0) }, #""sitemap""#)
    }

    func testMarkupCommentIsNotATag() {
        let src = "<!-- hidden <div> -->\n<p>"
        XCTAssertEqual(first(src, .markup, kind: .comment), "<!-- hidden <div> -->")
    }

    // MARK: - Offsets

    func testOffsetsAreUtf16SoNonAsciiDoesNotShiftColors() {
        // An emoji is 2 UTF-16 units; a Character-based offset would put every
        // later span one unit short and paint the wrong text.
        let src = "// 🚀\nfunc go() {}"
        let ns = src as NSString
        let keyword = spans(src, .swift).first { $0.kind == .keyword }
        XCTAssertNotNil(keyword)
        XCTAssertEqual(ns.substring(with: NSRange(location: keyword!.start, length: keyword!.length)), "func")
    }

    // MARK: - Class invariants (hold for every language on every sample)

    /// Whatever the input, spans must be sorted, non-overlapping, and inside the
    /// string — an NSAttributedString `addAttribute` with an out-of-bounds range
    /// is an uncatchable crash, and overlapping ranges silently repaint.
    func testSpansAreSortedNonOverlappingAndInBoundsForEverySample() {
        let samples: [(SyntaxLanguage, String)] = [
            (.swift, "struct A { let x = \"s\" /* c */ }\n// tail"),
            (.javascript, "const f = (a) => `t${a}` // x\n/* b */"),
            (.python, "def f(a):\n    '''doc'''\n    return {'k': 1}  # c"),
            (.json, #"{"a": [1, 2, {"b": null}], "c": "d"}"#),
            (.shell, "for f in *.txt; do\n  echo \"$f\" # c\ndone"),
            (.markup, "<a href='x'>t</a><!-- c -->"),
            (.cFamily, "int main(void) { /* c */ return 0; } // t"),
            (.zig, "const std = @import(\"std\"); // c"),
            (.rust, "fn main() { let s = r\"raw\"; } // c"),
            (.go, "func main() { s := `raw` } // c"),
            // Degenerate inputs that have crashed hand-rolled lexers before.
            (.swift, ""),
            (.swift, "\"unterminated"),
            (.cFamily, "/*"),
            (.markup, "<"),
            (.python, "\"\"\"unclosed"),
            (.json, "{"),
        ]
        for (lang, src) in samples {
            let out = spans(src, lang)
            let limit = (src as NSString).length
            var prevEnd = 0
            for span in out {
                XCTAssertGreaterThanOrEqual(span.start, prevEnd,
                    "spans must be sorted and non-overlapping — \(lang) / \(src.debugDescription)")
                XCTAssertGreaterThan(span.length, 0,
                    "empty span in \(lang) / \(src.debugDescription)")
                XCTAssertLessThanOrEqual(span.start + span.length, limit,
                    "span past end of source in \(lang) / \(src.debugDescription)")
                prevEnd = span.start + span.length
            }
        }
    }
}
