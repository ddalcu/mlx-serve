import XCTest
@testable import MLXCore

/// Ordered lists in the transcript. Tested through the rendered attributed
/// string rather than the parser, which is the seam the view actually uses.
final class MarkdownListTests: XCTestCase {

    private func rendered(_ source: String) -> String {
        MarkdownText.attributedString(for: source).string
    }

    /// Whether the run containing `needle` is a list ITEM rather than prose
    /// that happens to start with a marker. Blocks are joined with a single
    /// `\n`, so the rendered TEXT of a paragraph followed by a list and of a
    /// paragraph that swallowed one are identical — the hanging indent is what
    /// tells them apart.
    private func isListItem(_ needle: String, in source: String) -> Bool {
        let attributed = MarkdownText.attributedString(for: source)
        guard let range = attributed.string.range(of: needle) else { return false }
        let location = attributed.string.distance(from: attributed.string.startIndex,
                                                  to: range.lowerBound)
        let style = attributed.attribute(.paragraphStyle, at: location,
                                         effectiveRange: nil) as? NSParagraphStyle
        return (style?.headIndent ?? 0) > 0
    }

    /// The parser used to strip the number and the renderer put a bullet in
    /// front of whatever was left, so every numbered list arrived as bullets —
    /// and "step 3" in the text below pointed at nothing the reader could see.
    func testANumberedListKeepsItsNumbers() {
        let out = rendered("1. first\n2. second\n3. third")
        XCTAssertTrue(out.contains("1. first"), out)
        XCTAssertTrue(out.contains("2. second"), out)
        XCTAssertTrue(out.contains("3. third"), out)
        XCTAssertFalse(out.contains("•"), "a numbered list must not render as bullets")
    }

    /// The model's own numbering is kept, not recomputed: a list that starts at
    /// 4 is usually continuing one interrupted by a code block.
    func testTheModelsOwnNumbersAreKept() {
        XCTAssertTrue(rendered("4. fourth\n5. fifth").contains("4. fourth"))
    }

    func testAClosingParenthesisIsAlsoAListMarker() {
        XCTAssertTrue(rendered("1) first").contains("1) first"))
    }

    func testBulletsStillRenderAsBullets() {
        let out = rendered("- one\n* two")
        XCTAssertTrue(out.contains("• one"), out)
        XCTAssertTrue(out.contains("• two"), out)
    }

    /// A sentence that merely STARTS with a digit is not a list, and the old
    /// test (first character is a number, and a ". " appears anywhere in the
    /// line) silently ate everything up to that full stop: "1 pes. A kočka"
    /// rendered as "• A kočka".
    func testASentenceStartingWithADigitIsNotAList() {
        let out = rendered("1 pes. A kočka spolu.")
        XCTAssertTrue(out.contains("1 pes."), "the sentence must survive whole: \(out)")
        XCTAssertFalse(out.contains("•"), out)
    }

    func testAVersionNumberInProseIsNotAList() {
        let out = rendered("26.9.1 is the release. Nothing here is a list.")
        XCTAssertFalse(out.contains("•"), out)
        XCTAssertTrue(out.contains("26.9.1 is the release."), out)
    }

    /// A paragraph ended on the first bullet already; it has to end on the
    /// first numbered item too, or the list is swallowed into the sentence
    /// above it — which is what models write when they skip the blank line.
    func testAParagraphEndsWhereANumberedListBegins() {
        let source = "Here are the steps:\n1. first\n2. second"
        XCTAssertTrue(rendered(source).contains("1. first"))
        XCTAssertTrue(isListItem("1. first", in: source),
                      "the list must be its own block, not the tail of the sentence above it")
        XCTAssertFalse(isListItem("Here are the steps:", in: source))
    }

    /// Every item is its own line.
    ///
    /// Items are separate blocks, and the spacer BETWEEN blocks used to be the
    /// only newline in the string — so tightening the rhythm by dropping that
    /// spacer ran the whole list into one paragraph with the markers sitting
    /// inline in the prose.
    func testEachItemStartsItsOwnLine() {
        XCTAssertTrue(rendered("- one\n- two").contains("one\n• two"))
        XCTAssertTrue(rendered("1. first\n2. second").contains("first\n2. second"))
    }

    /// A number with no text after it is still a list item, not a paragraph
    /// that happens to look like one.
    func testAnEmptyItemDoesNotCrashOrVanish() {
        XCTAssertTrue(rendered("1. ").contains("1."))
    }

    // MARK: - Nesting

    /// The hanging indent of the run containing `needle`, which is what tells
    /// an item's depth (and an item from prose) apart.
    private func headIndent(_ needle: String, in source: String) -> CGFloat {
        let attributed = MarkdownText.attributedString(for: source)
        guard let range = attributed.string.range(of: needle) else { return -1 }
        let location = attributed.string.distance(from: attributed.string.startIndex,
                                                  to: range.lowerBound)
        let style = attributed.attribute(.paragraphStyle, at: location,
                                         effectiveRange: nil) as? NSParagraphStyle
        return style?.headIndent ?? 0
    }

    /// An indented item is a level of its own. The marker was read at the
    /// start of the line only, so a nested item fell through to a paragraph
    /// and every outline a model wrote arrived flat.
    func testANestedItemSitsDeeperThanItsParent() {
        let source = "- top\n  - nested\n- second"
        XCTAssertTrue(rendered(source).contains("• nested"), rendered(source))
        XCTAssertGreaterThan(headIndent("nested", in: source), headIndent("top", in: source))
        XCTAssertEqual(headIndent("second", in: source), headIndent("top", in: source),
                       "coming back out returns to the parent's depth")
    }

    /// Depth counts the STEPS the list takes, not spaces: models write two or
    /// four for the same one level, and both must read as one.
    func testFourSpaceNestingIsOneLevelLikeTwoSpaceNesting() {
        let two = "- top\n  - nested"
        let four = "- top\n    - nested"
        XCTAssertEqual(headIndent("nested", in: two), headIndent("nested", in: four))
    }

    func testThirdLevelGoesDeeperStill() {
        let source = "- one\n  - two\n    - three"
        XCTAssertGreaterThan(headIndent("three", in: source), headIndent("two", in: source))
    }

    /// A directory tree written as a list is deeper than the column is wide:
    /// unbounded, twelve levels is 198pt of indent before the text starts, and
    /// the items end up in a sliver at the right edge.
    func testAnOutlineDeeperThanTheColumnStopsIndenting() {
        let source = (0..<9).map { String(repeating: " ", count: $0 * 2) + "- item\($0)" }
            .joined(separator: "\n")
        let capped = headIndent("item5", in: source)
        XCTAssertEqual(headIndent("item8", in: source), capped, "past the cap the indent holds")
        XCTAssertGreaterThan(capped, headIndent("item4", in: source), "levels below it still step")
    }

    /// A line under an item, indented to its text, is part of that item: it
    /// used to end the list and start a paragraph back at the margin.
    func testAnIndentedLineContinuesTheItemAboveIt() {
        let source = "- first\n  continued\n- second"
        let out = rendered(source)
        XCTAssertTrue(out.contains("continued"), out)
        XCTAssertFalse(out.contains("• continued"), "it is the item's own text, not a new item")
        XCTAssertGreaterThan(headIndent("continued", in: source), 0,
                             "it stays inside the item, at the item's indent")
    }

    /// A hard newline would start a paragraph, and a paragraph takes the
    /// first-line indent (the margin) plus a paragraph's air above it — the
    /// item's own second line would look like prose that left the list.
    func testTheContinuationIsASoftBreakNotANewParagraph() {
        let out = rendered("- first\n  continued\n- second")
        XCTAssertTrue(out.contains("continued"), out)
        XCTAssertFalse(out.contains("first\ncontinued"), "that break starts a paragraph")
    }

    /// Ordinary prose after a list, at the margin, still ends the list.
    func testAnUnindentedLineEndsTheList() {
        let source = "- first\nplain prose"
        XCTAssertEqual(headIndent("plain prose", in: source), 0)
    }

    // MARK: - Task lists

    /// `- [ ]` is a list marker with a box, not a bullet followed by two
    /// brackets: models write plans and checklists this way.
    func testATaskListRendersBoxes() {
        let out = rendered("- [ ] open\n- [x] done")
        XCTAssertTrue(out.contains("\u{25A1} open"), out)
        XCTAssertTrue(out.contains("\u{2611} done"), out)
        XCTAssertFalse(out.contains("["), out)
    }

    func testAnUppercaseXIsAlsoTicked() {
        XCTAssertTrue(rendered("- [X] done").contains("\u{2611} done"))
    }

    /// Brackets that are not a checkbox stay text.
    func testBracketsInAnItemAreNotABox() {
        let out = rendered("- [link] to something")
        XCTAssertTrue(out.contains("• [link] to something"), out)
    }
}
