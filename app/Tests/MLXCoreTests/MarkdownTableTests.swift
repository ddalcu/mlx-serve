import XCTest
import AppKit
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

    func testColumnFractionsWeightByLongestCellInEachColumn() {
        let f = MarkdownTable.columnFractions(headers: ["a", "bbbb"], rows: [["x", "y"]])
        XCTAssertEqual(f.count, 2)
        XCTAssertGreaterThan(f[1], f[0])
        XCTAssertEqual(f.reduce(0, +), 1.0, accuracy: 0.0001)
    }

    func testColumnFractionsGiveEmptyColumnAVisibleFloor() {
        let f = MarkdownTable.columnFractions(headers: ["a", ""], rows: [["x", ""], ["y", ""]])
        // Both columns bottom out at the same floor, so they split evenly.
        XCTAssertEqual(f[0], f[1], accuracy: 0.0001)
    }

    func testColumnFractionsRaggedRowsDoNotCrashAndReturnOneFractionPerHeader() {
        let f = MarkdownTable.columnFractions(headers: ["a", "b", "c"], rows: [["x"], ["y", "z", "w", "extra"]])
        XCTAssertEqual(f.count, 3)
    }

    func testColumnFractionsNoHeadersReturnsEmpty() {
        XCTAssertEqual(MarkdownTable.columnFractions(headers: [], rows: []), [])
    }

    // MARK: - The grid

    private func tableBlocks(_ source: String) -> [NSTextTableBlock] {
        let rendered = MarkdownText.attributedString(for: source)
        var blocks: [NSTextTableBlock] = []
        rendered.enumerateAttribute(.paragraphStyle, in: NSRange(location: 0, length: rendered.length)) { value, _, _ in
            guard let style = value as? NSParagraphStyle else { return }
            blocks.append(contentsOf: style.textBlocks.compactMap { $0 as? NSTextTableBlock })
        }
        return blocks
    }

    private static let gridSource = """
    | a | b |
    | --- | --- |
    | 1 | 2 |
    | 3 | 4 |
    """

    /// A neighbour's top edge would land on the same line as this one's bottom
    /// and draw it twice as thick, so only `.maxY` is ever set between rows.
    /// The kind of thing that reads as "the middle rule looks wrong somehow"
    /// and takes an afternoon to find.
    func testARowRuleIsNeverDrawnTwice() {
        let blocks = tableBlocks(Self.gridSource)
        XCTAssertFalse(blocks.isEmpty, "sanity: the source really renders a table")
        for block in blocks where block.startingRow > 0 {
            XCTAssertEqual(block.width(for: .border, edge: .minY), 0,
                           "row \(block.startingRow) redraws the rule above it")
        }
    }

    /// Vertical rules between columns turn a comparison into a spreadsheet, so
    /// only the outermost columns carry a side. Easy to reintroduce by setting
    /// a border on every edge of every cell.
    func testThereAreNoRulesBetweenColumns() {
        let blocks = tableBlocks("| a | b | c |\n| --- | --- | --- |\n| 1 | 2 | 3 |")
        for block in blocks where block.startingColumn == 1 {
            XCTAssertEqual(block.width(for: .border, edge: .minX), 0)
            XCTAssertEqual(block.width(for: .border, edge: .maxX), 0)
        }
    }

    // MARK: - Room for the words themselves

    /// The comparison table that exposed this: one column of long sentences
    /// beside one holding "2-3 years".
    private static let lifespanTable: (headers: [String], rows: [[String]]) = (
        headers: ["Animal", "Habitat", "Food", "Care effort", "Lifespan"],
        rows: [
            ["Hamster (Syrian/golden)",
             "Cage or tank ~100 x 50 cm min, solid bottom, deep bedding to burrow in, exercise wheel; kept alone",
             "Pellets + fresh veg, occasional treats/insects",
             "Moderate - daily food/water, spot-clean daily, deep-clean weekly; mostly active at night",
             "2-3 years"],
            ["Snake (corn snake / ball python)",
             "Secure glass/plastic terrarium with a hide, heat source (lamp/mat) + thermostat, humidity control, safe substrate",
             "Whole pre-killed rodents (mice/rats) on a schedule",
             "Low daily, but setup-sensitive - feeds less often (weekly to every 1-2 wks), but needs accurate temp/humidity and handling frozen food",
             "15-20+ years"],
        ]
    )

    /// Weighting purely by how much text a column holds starves the short one:
    /// a column of sentences wraps onto more lines and reads fine, a column
    /// holding one word has nowhere to go and breaks MID-WORD ("Life spa n").
    /// So the weight is compressed, and a column never asks for less than its
    /// own longest word.
    func testAShortColumnKeepsRoomForItsLongestWord() {
        let t = Self.lifespanTable
        let f = MarkdownTable.columnFractions(headers: t.headers, rows: t.rows)
        let lifespan = f[4]

        // "Lifespan" is 8 characters; the widest column's cells run past 130.
        // Purely proportional, that is a share of about 0.035 - which is where
        // the header came apart. The word is the floor now.
        XCTAssertGreaterThan(lifespan, 0.09, "the header cannot fit in this share")
        XCTAssertEqual(f.reduce(0, +), 1.0, accuracy: 0.0001)
    }

    /// The compression must not flatten the table into equal columns either:
    /// a column of sentences still deserves more room than a column of dates.
    func testALongColumnStillOutweighsAShortOne() {
        let t = Self.lifespanTable
        let f = MarkdownTable.columnFractions(headers: t.headers, rows: t.rows)
        XCTAssertGreaterThan(f[3], f[4], "care effort holds ten times the text")
        XCTAssertGreaterThan(f[1], f[0])
    }

    /// One unbreakable monster - a snake_case identifier, a URL, a hash - must
    /// not be able to claim the table. Past the cap it is on its own and
    /// `NSTextTable` breaks it mid-word, which is the right outcome: one ugly
    /// column beats three unreadable ones.
    func testOneEnormousWordCannotClaimTheTable() {
        let f = MarkdownTable.columnFractions(
            headers: ["key", "value", "note"],
            rows: [["nazev_opravdu_velmi_dlouheho_json_klice_ktery_nekdo_napsal",
                    "a reasonable value",
                    "and a sentence of explanation here"]])
        XCTAssertLessThan(f[0], 0.5, "one word must not take the table")
        XCTAssertGreaterThan(f[1], 0.2, "the other columns stay usable")
        XCTAssertGreaterThan(f[2], 0.2)
        XCTAssertEqual(f.reduce(0, +), 1.0, accuracy: 0.0001)
    }

    /// The cap is what stops it. Two columns with the SAME amount of text, one
    /// of it unbreakable: past the cap the monster gets no more than the
    /// sentence does, and is broken mid-word instead.
    func testPastTheCapAnUnbreakableWordStopsGrowing() {
        let sentence = "one two three four five six seven eight nine ten eleven"
        let monster = String(repeating: "x", count: sentence.count)
        let f = MarkdownTable.columnFractions(headers: ["a", "b"], rows: [[sentence, monster]])
        XCTAssertEqual(f[0], f[1], accuracy: 0.0001,
                       "the cap is off: an unbreakable run is still buying width")
    }

    /// Under the cap it is honoured, which is the whole point - a 10-character
    /// header should not be broken across lines to save 20 points.
    func testAWordUnderTheCapIsHonouredInFull() {
        let withWord = MarkdownTable.columnFractions(headers: ["a", "thermostat"], rows: [["x", "y"]])
        let withoutWord = MarkdownTable.columnFractions(headers: ["a", "ab"], rows: [["x", "y"]])
        XCTAssertGreaterThan(withWord[1], withoutWord[1])
    }

    /// A cell holding several words wraps, so its longest WORD is what it
    /// needs, not its whole length.
    func testTheFloorIsTheLongestWordNotTheLongestCell() {
        let manyWords = MarkdownTable.columnFractions(
            headers: ["a", "b"], rows: [["x", "one two one two one two one two"]])
        let oneWord = MarkdownTable.columnFractions(
            headers: ["a", "b"], rows: [["x", "onetwoonetwoonetwoonetwoonetwo"]])
        XCTAssertGreaterThan(oneWord[1], manyWords[1],
                             "the unbreakable one needs more room than the same text with spaces")
    }

    /// Every fraction must stay a usable share: `NSTextTable` will happily
    /// render a 0.4% column as a sliver of broken letters.
    func testNoColumnIsEverSqueezedToNothing() {
        let f = MarkdownTable.columnFractions(
            headers: ["a", "b", "c", "d"],
            rows: [["x", "y", "z", String(repeating: "long sentence ", count: 40)]])
        for (i, share) in f.enumerated() {
            XCTAssertGreaterThan(share, 0.02, "column \(i) collapsed")
        }
    }

    func testTableRendersAsNSTextTableInsideTheSharedAttributedString() {
        // Regression guard for the drag-selection fix: a table must be part
        // of the SAME NSAttributedString as its surrounding prose (an
        // NSTextTable paragraph attribute), not a separate SwiftUI view, so
        // selection and copy span prose and table together.
        let source = "before\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nafter"
        let attributed = MarkdownText.attributedString(for: source)
        var foundTable = false
        attributed.enumerateAttribute(
            .paragraphStyle, in: NSRange(location: 0, length: attributed.length)
        ) { value, _, _ in
            if let style = value as? NSParagraphStyle, style.textBlocks.contains(where: { $0 is NSTextTableBlock }) {
                foundTable = true
            }
        }
        XCTAssertTrue(foundTable)
        XCTAssertTrue(attributed.string.contains("before"))
        XCTAssertTrue(attributed.string.contains("after"))
    }
}
