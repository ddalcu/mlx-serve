import AppKit
import XCTest
@testable import MLXCore

/// Quotes and inline code in the transcript.
final class MarkdownBlockStylingTests: XCTestCase {

    private func attributed(_ source: String) -> NSAttributedString {
        MarkdownText.attributedString(for: source)
    }

    private func attribute(_ key: NSAttributedString.Key,
                           at needle: String, in source: String) -> Any? {
        let string = attributed(source)
        guard let range = string.string.range(of: needle) else { return nil }
        let location = string.string.distance(from: string.string.startIndex,
                                              to: range.lowerBound)
        return string.attribute(key, at: location, effectiveRange: nil)
    }

    // MARK: - Quotes

    /// `>` was not handled at all: the marker rendered as literal text in an
    /// ordinary paragraph.
    func testAQuoteDropsItsMarker() {
        let out = attributed("> quoted line").string
        XCTAssertTrue(out.contains("quoted line"), out)
        XCTAssertFalse(out.contains(">"), out)
    }

    /// The bar is a single-cell text table with a left border — the only way to
    /// get a rule beside text in an NSAttributedString.
    func testAQuoteIsDrawnAsABarredBlock() {
        let style = attribute(.paragraphStyle, at: "quoted line",
                              in: "> quoted line") as? NSParagraphStyle
        XCTAssertEqual(style?.textBlocks.count, 1)
        let block = style?.textBlocks.first
        XCTAssertGreaterThan(block?.width(for: .border, edge: .minX) ?? 0, 0,
                             "the quote needs its rule on the leading edge")
    }

    /// Consecutive quoted lines are ONE quote, not a stack of them: a model
    /// wraps a quoted paragraph at its own width, and each wrapped line arrives
    /// with its own marker.
    func testConsecutiveQuotedLinesMergeIntoOneBlock() {
        let string = attributed("> first line\n> second line")
        XCTAssertTrue(string.string.contains("first line"))
        XCTAssertTrue(string.string.contains("second line"))
        let first = attribute(.paragraphStyle, at: "first line",
                              in: "> first line\n> second line") as? NSParagraphStyle
        let second = attribute(.paragraphStyle, at: "second line",
                               in: "> first line\n> second line") as? NSParagraphStyle
        XCTAssertEqual(first?.textBlocks.first, second?.textBlocks.first,
                       "both lines belong to the same cell, or they draw two bars")
    }

    func testProseIsNotAQuote() {
        let style = attribute(.paragraphStyle, at: "plain", in: "plain sentence") as? NSParagraphStyle
        XCTAssertEqual(style?.textBlocks.count ?? 0, 0)
    }

    // MARK: - Inline code

    /// Marked, not coloured: `.backgroundColor` fills the whole line fragment,
    /// which at prose leading reaches the line above. The text view draws this
    /// marker at the height of the code's own letters.
    func testInlineCodeGetsItsOwnGround() {
        XCTAssertNotNil(attribute(.inlineCodeGround, at: "code", in: "some `code` here"),
                        "an inline code span must be marked for its ground")
        XCTAssertNil(attribute(.backgroundColor, at: "code", in: "some `code` here"),
                     "a line-height-tall background is what this replaced")
    }

    /// Asserted by IDENTITY, not by the `.monoSpace` symbolic trait:
    /// `NSFont.monospacedSystemFont` does not advertise that trait, so a trait
    /// check calls SF Mono prose and would have this passing on nothing.
    func testInlineCodeIsSetInTheCodeFace() {
        let font = attribute(.font, at: "code", in: "some `code` here") as? NSFont
        XCTAssertEqual(font?.fontName,
                       NSFont.monospacedSystemFont(ofSize: ChatMetrics.transcriptCodeFontSize,
                                                   weight: .regular).fontName)
        XCTAssertEqual(font?.pointSize, ChatMetrics.transcriptCodeFontSize,
                       "code reads at the code size, not the body size")
    }

    func testProseAroundItIsNotTinted() {
        XCTAssertNil(attribute(.inlineCodeGround, at: "some", in: "some `code` here"))
        XCTAssertNil(attribute(.backgroundColor, at: "some", in: "some `code` here"))
    }

    // MARK: - Thematic break

    /// Whether the run containing `needle` is drawn as a bordered block.
    private func isBorderedBlock(_ needle: String, in source: String) -> Bool {
        guard let style = attribute(.paragraphStyle, at: needle, in: source) as? NSParagraphStyle,
              let block = style.textBlocks.first
        else { return false }
        return block.width(for: .border, edge: .minY) > 0
    }

    /// `---` between sections is a rule. It used to render as three hyphens in
    /// a paragraph of their own.
    func testAThematicBreakIsARuleNotThreeHyphens() {
        let out = attributed("before\n\n---\n\nafter").string
        XCTAssertTrue(out.contains("before") && out.contains("after"), out)
        XCTAssertFalse(out.contains("---"), out)
    }

    func testTheRuleIsDrawnAsABorderedBlock() {
        XCTAssertTrue(isBorderedBlock("\u{00A0}", in: "before\n\n---\n\nafter"),
                      "the rule is a bordered block, like the quote bar")
    }

    func testTheOtherTwoSpellingsAreRulesToo() {
        XCTAssertFalse(attributed("a\n\n***\n\nb").string.contains("***"))
        XCTAssertFalse(attributed("a\n\n___\n\nb").string.contains("___"))
    }

    /// Two hyphens are not a rule, and neither is a line that carries text.
    func testALineThatIsNotOnlyTheRuleStaysText() {
        XCTAssertTrue(attributed("--").string.contains("--"))
        XCTAssertTrue(attributed("--- and more").string.contains("--- and more"))
    }

    // MARK: - Strikethrough

    /// Foundation parses `~~` into an intent; nothing read it, so the span
    /// rendered as ordinary prose.
    func testStrikethroughIsStruckThrough() {
        let value = attribute(.strikethroughStyle, at: "gone", in: "this is ~~gone~~ now") as? Int
        XCTAssertEqual(value, NSUnderlineStyle.single.rawValue)
    }

    func testProseAroundItIsNotStruckThrough() {
        XCTAssertNil(attribute(.strikethroughStyle, at: "this", in: "this is ~~gone~~ now"))
    }
}
