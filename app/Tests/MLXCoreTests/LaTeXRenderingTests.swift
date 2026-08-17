import AppKit
import XCTest
@testable import MLXCore

final class LaTeXRenderingTests: XCTestCase {
    private var appDirectory: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func values(
        for key: NSAttributedString.Key,
        in attributed: NSAttributedString
    ) -> [Any] {
        var found: [Any] = []
        attributed.enumerateAttribute(
            key,
            in: NSRange(location: 0, length: attributed.length)
        ) { value, _, _ in
            if let value { found.append(value) }
        }
        return found
    }

    func testInlineFormulaRendersAsANativeAttachment() {
        let source = #"Euler: $e^{i\pi}+1=0$."#
        let rendered = MarkdownText.attributedString(for: source)

        XCTAssertEqual(values(for: .attachment, in: rendered).count, 1)
        XCTAssertEqual(
            values(for: .mlxLaTeXSource, in: rendered) as? [String],
            [#"$e^{i\pi}+1=0$"#]
        )
        XCTAssertEqual(LaTeXCopyText.string(from: rendered), source)
    }

    func testInlineFormulaRendersInsideATableCell() {
        // Exercises the MarkdownTableView → renderInline path: a cell string
        // containing an inline LaTeX formula must produce an attachment with
        // the correct source, mirroring the deleted
        // testInlineFormulaRendersInsideAMarkdownTable that used the old
        // attributedString(for:) path which no longer renders tables.
        let cellText = "$\\text{e}^{i\\pi} + 1 = 0$"
        let rendered = MarkdownText.renderInline(cellText, theme: .light)

        XCTAssertEqual(values(for: .attachment, in: rendered).count, 1)
        XCTAssertEqual(
            values(for: .mlxLaTeXSource, in: rendered) as? [String],
            [#"$\text{e}^{i\pi} + 1 = 0$"#]
        )
        XCTAssertEqual(LaTeXCopyText.string(from: rendered), cellText)
    }

    func testParenthesizedDollarFormulaWithTextCommandRendersInAHeading() {
        let source = #"## 1. Euler's Identity ($\text{e}^{i\pi} + 1 = 0$)"#
        let rendered = MarkdownText.attributedString(for: source)

        XCTAssertEqual(values(for: .attachment, in: rendered).count, 1)
        XCTAssertEqual(
            values(for: .mlxLaTeXSource, in: rendered) as? [String],
            [#"$\text{e}^{i\pi} + 1 = 0$"#]
        )
    }

    func testInvalidFormulaFallsBackToItsExactSource() {
        let source = #"Broken $\frac{$ stays readable."#
        let rendered = MarkdownText.attributedString(for: source)

        XCTAssertTrue(values(for: .attachment, in: rendered).isEmpty)
        XCTAssertTrue(values(for: .mlxLaTeXSource, in: rendered).isEmpty)
        XCTAssertEqual(rendered.string, source)
    }

    func testDisplayPreflightAcceptsMathAndRejectsMalformedTeX() {
        XCTAssertTrue(DisplayLaTeXRenderer.canRender(
            #"\int_0^1 x^2\,dx"#,
            theme: .light,
            fontSize: 20
        ))
        XCTAssertFalse(DisplayLaTeXRenderer.canRender(
            #"\frac{"#,
            theme: .light,
            fontSize: 20
        ))
    }

    func testCopyRehydratesOnlyTheSelectedEquation() {
        let attributed = NSMutableAttributedString(string: "before \u{FFFC} after")
        attributed.addAttribute(
            .mlxLaTeXSource,
            value: #"\(x^2\)"#,
            range: NSRange(location: 7, length: 1)
        )

        XCTAssertEqual(
            LaTeXCopyText.string(from: attributed, range: NSRange(location: 7, length: 1)),
            #"\(x^2\)"#
        )
        XCTAssertEqual(
            LaTeXCopyText.string(from: attributed, range: NSRange(location: 0, length: 8)),
            #"before \(x^2\)"#
        )
    }

    func testMessageBubbleScopesRichRenderingToAssistantResponses() throws {
        let chat = try String(
            contentsOf: appDirectory.appendingPathComponent("Sources/MLXServe/Views/ChatView.swift"),
            encoding: .utf8
        )
        let pattern = #"if message\.role == \.assistant \{(?s:.*?)MarkdownText\(message\.content(?s:.*?)\} else \{(?s:.*?)Text\(message\.content\)"#
        XCTAssertNotNil(
            chat.range(of: pattern, options: .regularExpression),
            "assistant answers use MarkdownText/LaTeX while the user's own prompt remains plain Text"
        )
    }
}
