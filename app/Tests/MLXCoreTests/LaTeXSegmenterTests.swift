import XCTest
@testable import MLXCore

/// TeX delimiter recognition for assistant prose. Typesetting belongs to
/// SwaTex; this seam only decides which source bytes are math and must never
/// eat ordinary Markdown, currency, code, or an unfinished streaming token.
final class LaTeXSegmenterTests: XCTestCase {

    private typealias Segment = LaTeXSegmenter.Segment

    func testPlainTextStaysOneSegment() {
        XCTAssertEqual(LaTeXSegmenter.segments("No equations here."), [
            .text("No equations here."),
        ])
    }

    func testRecognizesInlineDollarAndParenthesisDelimiters() {
        let source = #"Euler wrote $e^{i\pi}+1=0$, while \(a^2+b^2=c^2\) is familiar."#
        XCTAssertEqual(LaTeXSegmenter.segments(source), [
            .text("Euler wrote "),
            .inline(latex: #"e^{i\pi}+1=0"#, raw: #"$e^{i\pi}+1=0$"#),
            .text(", while "),
            .inline(latex: #"a^2+b^2=c^2"#, raw: #"\(a^2+b^2=c^2\)"#),
            .text(" is familiar."),
        ])
    }

    func testRecognizesMultilineDisplayDelimitersAndTrimsTheirPadding() {
        let source = #"""
        Before
        $$
          \sum_{i=1}^{n} i
        $$
        middle
        \[
          \int_0^1 x^2\,dx
        \]
        after
        """#
        XCTAssertEqual(LaTeXSegmenter.segments(source), [
            .text("Before\n"),
            .display(latex: #"\sum_{i=1}^{n} i"#, raw: "$$\n  \\sum_{i=1}^{n} i\n$$"),
            .text("\nmiddle\n"),
            .display(latex: #"\int_0^1 x^2\,dx"#, raw: "\\[\n  \\int_0^1 x^2\\,dx\n\\]"),
            .text("\nafter"),
        ])
    }

    func testRecognizesStandaloneEquationEnvironments() {
        let source = #"""
Result:
\begin{align*}
a &= b + c \\
d &= e
\end{align*}
Done.
"""#
        let raw = #"""
\begin{align*}
a &= b + c \\
d &= e
\end{align*}
"""#
        XCTAssertEqual(LaTeXSegmenter.segments(source), [
            .text("Result:\n"),
            .display(latex: raw, raw: raw),
            .text("\nDone."),
        ])
    }

    func testEscapedDollarsCurrencyAndInlineCodeAreNotEquations() {
        let source = #"A ticket costs $5 today and $10 tomorrow; write \$5 or `$HOME`, then render $x_1$."#
        XCTAssertEqual(LaTeXSegmenter.segments(source), [
            .text(#"A ticket costs $5 today and $10 tomorrow; write \$5 or `$HOME`, then render "#),
            .inline(latex: "x_1", raw: "$x_1$"),
            .text("."),
        ])
    }

    func testUnclosedDelimiterStaysLiteralWhileStreaming() {
        for source in [#"Partial $\frac{1}{"#, #"Partial $$\sum_i"#, #"Partial \[x+y"#] {
            XCTAssertEqual(LaTeXSegmenter.segments(source), [.text(source)])
        }
    }

    func testMalformedInlineDelimiterDoesNotStealAFollowingDisplay() {
        XCTAssertEqual(LaTeXSegmenter.segments("$x$$y$$"), [
            .text("$x"),
            .display(latex: "y", raw: "$$y$$"),
        ])
    }

    func testOversizedFormulaStaysLiteralInsteadOfDoingUnboundedRenderWork() {
        let source = "$" + String(repeating: "x+", count: 9_000) + "x$"
        XCTAssertEqual(LaTeXSegmenter.segments(source), [.text(source)])
    }

    func testEveryInputByteCanBeReconstructedFromSegments() {
        let sources = [
            #"Text $x^2$ tail"#,
            "before\n$$\nx+y\n$$\nafter",
            #"escaped \$ and `code $x$` plus \(y\)"#,
            #"\begin{cases}x & x > 0 \\ -x & x < 0\end{cases}"#,
            #"unfinished $x + y"#,
        ]

        for source in sources {
            let reconstructed = LaTeXSegmenter.segments(source).map { segment -> String in
                switch segment {
                case .text(let text): return text
                case .inline(_, let raw), .display(_, let raw): return raw
                }
            }.joined()
            XCTAssertEqual(reconstructed, source, "segmenting must not lose model output")
        }
    }
}
