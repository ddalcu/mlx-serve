# Chat Table Rendering & Content Column Width Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace MLX Core Chat's monospaced ASCII-art table rendering with a real
SwiftUI table view (bold header, divider, proportional columns, wrapped cells,
inline markdown inside cells), and widen the chat's reading column from a
fixed 740pt to 80% of the panel's visible width.

**Architecture:** Table detection moves out of `MarkdownText.parseBlocks` into
a shared `MarkdownTable` parser used by `MarkdownSegmenter`, so a table becomes
its own segment (like a fenced code block already is) instead of monospaced
text glued into a prose run. `MarkdownText` renders that segment with a new
`MarkdownTableView` — a plain `VStack`/`HStack` grid, not `Grid`, so column
widths can be proportional fractions measured once via `onGeometryChange`.
The content-column width changes from `ChatMetrics.contentMaxWidth` (a fixed
740pt constant) to `ChatMetrics.contentWidthFraction` (0.8) applied to the
detail column's measured width, stored on `ChatDetailView` and read by its
three existing capped sites (transcript, composer, empty-state greeting).

**Tech Stack:** SwiftUI (macOS 26.2 deployment target — `onGeometryChange` and
`AttributedString(NSAttributedString)` are both available), XCTest.

**Spec:** No separate spec file — this is a bounded change; the design below
was proposed and approved in chat on 2026-08-17 (table style: "minimal GFM" —
bold header + one divider, no vertical borders, chosen over full-grid and
row-divider alternatives; content width: proportional 80% of the *detail
column* — not the whole window — because the panel has a session sidebar).

## Global Constraints

- Every step in this plan is TDD: write the failing test, watch it fail, write
  the minimal implementation, watch it pass, commit.
- No project-wide `swift test` / `xcodebuild` runs mid-task — each task runs
  only the tests it just wrote (`swift test --filter <TestClass>`) plus a
  build. The full suite runs once, in the final verification task.
- `app/AGENTS.md` truncation rule applies to every `write` call in this
  session: any file body near ~150 lines is created with the first chunk via
  `write` and the remainder appended with `bash` heredocs.
- macOS-only code (`NSFont`, `NSColor`, `NSAttributedString`) stays under
  `app/Sources/MLXServe`; the shared parser (`MarkdownTable`) is pure
  `Foundation`/`SwiftUI` (`CGFloat`) with no AppKit import, matching
  `MarkdownSegmenter`'s existing style.
- Follow existing file conventions: doc comments explain *why*, not *what*
  (see any existing `///` block in `MarkdownSegmenter.swift` or
  `ChatMetrics.swift` for tone); `Equatable` on parser types where the
  existing types already are.

---

## Task 1: Extract the shared `MarkdownTable` parser

**Files:**
- Create: `app/Sources/MLXServe/Services/MarkdownTable.swift`
- Modify: `app/Sources/MLXServe/Views/ChatView.swift:4183-4183` (delete nested
  `enum TableAlignment`), `:4183-4192` (the `Block.table` case keeps using
  `TableAlignment`, now resolved from the new file), `:4331-4463` (delete
  `ParsedTable`, `tryParseTable`, `tryParseAsciiPseudoTable`,
  `splitOnDoubleSpace`, `isAsciiRule`, `parseTableRow`, `isTableSeparator`,
  `parseTableAlignments` — replaced by a single call into `MarkdownTable`),
  the `.table` arm in `parseBlocks` (search `Self.tryParseTable`)
- Test: `app/Tests/MLXCoreTests/MarkdownTableTests.swift`

**Interfaces:**
- Produces: `enum TableAlignment: Equatable { case left, right, center }`
  (top-level, replaces `MarkdownText.TableAlignment`); `struct
  MarkdownTable.ParsedTable { let headers: [String]; let rows: [[String]]; let
  alignments: [TableAlignment]; let end: Int }`; `static func
  MarkdownTable.parse(lines: [String], start: Int) -> ParsedTable?`. Task 2
  adds `MarkdownTable.columnWidths`. Task 3 (`MarkdownSegmenter`) and the
  existing `MarkdownText.parseBlocks` both call `MarkdownTable.parse`.

This is a pure extraction — behavior must not change. `MarkdownText`'s table
parsing (GFM pipe tables + the whitespace-aligned ASCII pseudo-table fallback
for models that skip GFM syntax) moves verbatim into the new file; call sites
in `MarkdownText` are updated to use it.

- [ ] **Step 1: Write the failing tests**

```swift
import XCTest
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
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd app && swift test --filter MarkdownTableTests`
Expected: build failure — `MarkdownTable` does not exist (`Services/MarkdownTable.swift` hasn't been created yet).

- [ ] **Step 3: Create `MarkdownTable.swift` with the extracted parser**

```swift
import Foundation

/// Column alignment for a rendered table, from GFM's `:---:` markers (the
/// ASCII pseudo-table fallback below has no alignment syntax, so it is
/// always `.left`).
enum TableAlignment: Equatable {
    case left, right, center
}

/// Detects and parses markdown tables: GFM pipe tables, and the
/// whitespace-aligned "pseudo-tables" smaller models emit when asked for
/// tabular data without GFM syntax. Shared by `MarkdownSegmenter` (which
/// splits a table into its own segment) and `MarkdownText` (whose
/// `parseBlocks` still calls this for anything the segmenter didn't already
/// split out) so the two passes can never disagree about what is a table.
enum MarkdownTable {

    struct ParsedTable {
        let headers: [String]
        let rows: [[String]]
        let alignments: [TableAlignment]
        /// Index of the line *after* the table, for the caller's loop.
        let end: Int
    }

    /// Detect a table starting at `lines[start]`. Requires a header row and a
    /// confirming separator row so a stray `|` (or a double space) in prose
    /// never becomes a false positive. Returns nil if the structural check
    /// fails, so the caller falls through to its own handling.
    static func parse(lines: [String], start: Int) -> ParsedTable? {
        guard start + 1 < lines.count else { return nil }
        let headerLine = lines[start].trimmingCharacters(in: .whitespaces)
        let sepLine = lines[start + 1].trimmingCharacters(in: .whitespaces)
        // Strict GFM form: pipes + dashed separator.
        if headerLine.contains("|"), isTableSeparator(sepLine) {
            let headers = parseTableRow(headerLine)
            let alignments = parseTableAlignments(sepLine)
            guard !headers.isEmpty else { return nil }
            var rows: [[String]] = []
            var i = start + 2
            while i < lines.count {
                let r = lines[i].trimmingCharacters(in: .whitespaces)
                guard r.contains("|") else { break }
                if isTableSeparator(r) { break }
                rows.append(parseTableRow(r))
                i += 1
            }
            return ParsedTable(headers: headers, rows: rows, alignments: alignments, end: i)
        }
        return parseAsciiPseudoTable(lines: lines, start: start)
    }

    /// Recognise the whitespace-aligned "table" shape smaller models emit
    /// when asked for tabular data without using GFM pipe syntax:
    ///   Header1   Header2   Header3
    ///   ---------------------------
    ///   value1    value2    value3
    /// Requires a dashed-rule line within the next two lines and at least 2
    /// columns in the header so a paragraph with one double space doesn't
    /// false-positive.
    private static func parseAsciiPseudoTable(lines: [String], start: Int) -> ParsedTable? {
        let header = lines[start]
        let headerCells = splitOnDoubleSpace(header)
        guard headerCells.count >= 2 else { return nil }
        var sepIdx = start + 1
        while sepIdx < min(start + 3, lines.count) {
            let candidate = lines[sepIdx].trimmingCharacters(in: .whitespaces)
            if isAsciiRule(candidate) { break }
            if !candidate.isEmpty { return nil }
            sepIdx += 1
        }
        guard sepIdx < lines.count, isAsciiRule(lines[sepIdx].trimmingCharacters(in: .whitespaces)) else {
            return nil
        }
        var rows: [[String]] = []
        var i = sepIdx + 1
        while i < lines.count {
            let raw = lines[i]
            let t = raw.trimmingCharacters(in: .whitespaces)
            if t.isEmpty { i += 1; break }
            if isAsciiRule(t) { i += 1; break }
            let cells = splitOnDoubleSpace(raw)
            // Tolerate single-cell continuation lines (a long cell the model
            // wrapped) by appending to the previous row's last cell.
            if cells.count == 1, !rows.isEmpty {
                rows[rows.count - 1][rows[rows.count - 1].count - 1] += " " + cells[0]
            } else {
                rows.append(cells)
            }
            i += 1
        }
        guard !rows.isEmpty else { return nil }
        let alignments = [TableAlignment](repeating: .left, count: headerCells.count)
        return ParsedTable(headers: headerCells, rows: rows, alignments: alignments, end: i)
    }

    /// Split on runs of two-or-more whitespace. Trims each cell. Drops the
    /// empty leading element if the line was indented.
    private static func splitOnDoubleSpace(_ line: String) -> [String] {
        line.components(separatedBy: "  ")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    /// True if the (already-trimmed) line consists entirely of dashes / box-
    /// drawing chars / spaces and is at least 3 chars long.
    private static func isAsciiRule(_ line: String) -> Bool {
        guard line.count >= 3 else { return false }
        let allowed: Set<Character> = ["-", "─", "=", " ", "|"]
        let allAllowed = line.allSatisfy { allowed.contains($0) }
        let hasDash = line.contains("-") || line.contains("─") || line.contains("=")
        return allAllowed && hasDash
    }

    private static func parseTableRow(_ line: String) -> [String] {
        var t = line.trimmingCharacters(in: .whitespaces)
        if t.hasPrefix("|") { t.removeFirst() }
        if t.hasSuffix("|") { t.removeLast() }
        return t.split(separator: "|", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
    }

    private static func isTableSeparator(_ line: String) -> Bool {
        let cells = parseTableRow(line)
        guard !cells.isEmpty else { return false }
        return cells.allSatisfy { cell in
            let c = cell.replacingOccurrences(of: " ", with: "")
            return c.range(of: "^:?-{3,}:?$", options: .regularExpression) != nil
        }
    }

    private static func parseTableAlignments(_ line: String) -> [TableAlignment] {
        parseTableRow(line).map { cell in
            let c = cell.replacingOccurrences(of: " ", with: "")
            let leftColon = c.hasPrefix(":")
            let rightColon = c.hasSuffix(":")
            if leftColon && rightColon { return .center }
            if rightColon { return .right }
            return .left
        }
    }
}
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `cd app && swift test --filter MarkdownTableTests`
Expected: PASS (9 tests).

- [ ] **Step 5: Point `MarkdownText` at the shared parser and delete the old code**

In `ChatView.swift`:
1. Delete the nested `enum TableAlignment { case left, right, center }`
   (search `enum TableAlignment` inside `struct MarkdownText`) — the
   top-level one from `MarkdownTable.swift` is used instead; no import
   needed, both files are in the same module.
2. Delete `private struct ParsedTable`, `private static func tryParseTable`,
   `private static func tryParseAsciiPseudoTable`, `private static func
   splitOnDoubleSpace`, `private static func isAsciiRule`, `private static
   func parseTableRow`, `private static func isTableSeparator`, `private
   static func parseTableAlignments` (the whole "MARK: Table parsing"
   section).
3. In `parseBlocks`, replace the block that calls `Self.tryParseTable(lines:
   start:)`:

```swift
            if let table = MarkdownTable.parse(lines: lines, start: i) {
                blocks.append(.table(table.headers, table.rows, table.alignments))
                i = table.end
                continue
            }
```

   (unchanged in shape — only the callee changes from `Self.tryParseTable` to
   `MarkdownTable.parse`).

- [ ] **Step 6: Build and run the full existing markdown test suite**

Run: `cd app && swift test --filter MarkdownTableTests --filter MarkdownSegmenterTests`
Expected: PASS. Also run `bash app/build.sh` to confirm the app still
compiles (the deleted functions must have no other call sites — `grep -n
"tryParseTable\|tryParseAsciiPseudoTable" app/Sources/MLXServe/Views/ChatView.swift`
should return nothing).

- [ ] **Step 7: Commit**

```bash
git add app/Sources/MLXServe/Services/MarkdownTable.swift app/Sources/MLXServe/Views/ChatView.swift app/Tests/MLXCoreTests/MarkdownTableTests.swift
git commit -m "refactor: extract MarkdownTable parser from MarkdownText"
```

---

## Task 2: Proportional column-width helper

**Files:**
- Modify: `app/Sources/MLXServe/Services/MarkdownTable.swift` (add
  `columnWidths`)
- Test: `app/Tests/MLXCoreTests/MarkdownTableTests.swift` (append)

**Interfaces:**
- Consumes: nothing new (pure function over `[String]`/`[[String]]`).
- Produces: `static func MarkdownTable.columnWidths(headers: [String], rows:
  [[String]]) -> [CGFloat]` — fractions in `(0, 1]` summing to `1.0`, one per
  header column, in header order. Task 4's `MarkdownTableView` multiplies
  these by the measured table width to size each column.

A pure helper (no SwiftUI/AppKit dependency — `CGFloat` only, matching
`ChatMetrics`) so column sizing is unit-testable without a view.

- [ ] **Step 1: Write the failing tests**

```swift
    func testColumnWidthsSumToOne() {
        let w = MarkdownTable.columnWidths(headers: ["Tip", "Explanation"],
                                            rows: [["a", "b"], ["cc", "dddd"]])
        XCTAssertEqual(w.reduce(0, +), 1.0, accuracy: 0.0001)
        XCTAssertEqual(w.count, 2)
    }

    func testWiderColumnGetsMoreShare() {
        let w = MarkdownTable.columnWidths(
            headers: ["Tip", "Explanation"],
            rows: [["`$@`", "Use \"$@\" to preserve arguments with spaces"]])
        XCTAssertGreaterThan(w[1], w[0])
    }

    func testSingleColumnTakesTheWholeWidth() {
        XCTAssertEqual(MarkdownTable.columnWidths(headers: ["a"], rows: [["b"]]), [1.0])
    }

    func testEmptyColumnsStillGetAVisibleFloor() {
        let w = MarkdownTable.columnWidths(headers: ["a", ""], rows: [["x", ""], ["y", ""]])
        // Both columns bottom out at the same floor, so they split evenly.
        XCTAssertEqual(w[0], w[1], accuracy: 0.0001)
    }

    func testRaggedRowsDoNotCrashAndReturnOneFractionPerHeader() {
        let w = MarkdownTable.columnWidths(headers: ["a", "b", "c"], rows: [["x"], ["y", "z", "w", "extra"]])
        XCTAssertEqual(w.count, 3)
    }

    func testNoHeadersReturnsEmpty() {
        XCTAssertEqual(MarkdownTable.columnWidths(headers: [], rows: []), [])
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd app && swift test --filter MarkdownTableTests`
Expected: FAIL — `columnWidths` is not a member of `MarkdownTable`.

- [ ] **Step 3: Implement `columnWidths`**

Add to `enum MarkdownTable` (needs `import SwiftUI` for `CGFloat` — actually
`CGFloat` lives in `CoreGraphics`/`Foundation` on Apple platforms, no new
import required since `Foundation` is already imported):

```swift
    /// Relative column widths (fractions summing to `1.0`) computed from
    /// content length: each column's share starts at its longest cell
    /// (header or data), floored so an empty or all-short column still gets
    /// a visible sliver instead of collapsing to zero. Pure, so
    /// `MarkdownTableView`'s layout can be pinned without spinning up a view.
    static func columnWidths(headers: [String], rows: [[String]]) -> [CGFloat] {
        let cols = headers.count
        guard cols > 0 else { return [] }
        var longest = headers.map { CGFloat($0.count) }
        for row in rows {
            for (j, cell) in row.prefix(cols).enumerated() {
                longest[j] = max(longest[j], CGFloat(cell.count))
            }
        }
        let floor: CGFloat = 4
        let weighted = longest.map { max($0, floor) }
        let total = weighted.reduce(0, +)
        return weighted.map { $0 / total }
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd app && swift test --filter MarkdownTableTests`
Expected: PASS (15 tests total).

- [ ] **Step 5: Commit**

```bash
git add app/Sources/MLXServe/Services/MarkdownTable.swift app/Tests/MLXCoreTests/MarkdownTableTests.swift
git commit -m "feat: add proportional column-width helper to MarkdownTable"
```

---

## Task 3: `MarkdownSegmenter` splits at table boundaries

**Files:**
- Modify: `app/Sources/MLXServe/Services/MarkdownSegmenter.swift`
- Modify: `app/Sources/MLXServe/Views/ChatView.swift` (temporary rendering
  wire-up in `MarkdownText.body`, described below)
- Test: `app/Tests/MLXCoreTests/MarkdownSegmenterTests.swift`

**Interfaces:**
- Produces: `MarkdownSegmenter.Segment` gains `case table(headers: [String],
  rows: [[String]], alignments: [TableAlignment])`. Task 4 switches on this
  case to render `MarkdownTableView`; until then this task renders it through
  the existing (still-present) `MarkdownText.renderTable` so the app stays in
  a working, visually-unchanged state after this task.

Tables currently live *inside* a prose run and get rendered as monospaced
text mid-paragraph. To become a real view, a table must be its own segment —
the same reason fenced code already is one (see the doc comment at the top of
`MarkdownSegmenter.swift`).

- [ ] **Step 1: Write the failing tests**

Append to `MarkdownSegmenterTests.swift`:

```swift
    func testGfmTableBecomesItsOwnSegment() {
        let s = "before\n| a | b |\n|---|---|\n| 1 | 2 |\nafter"
        XCTAssertEqual(segs(s), [
            .prose("before"),
            .table(headers: ["a", "b"], rows: [["1", "2"]], alignments: [.left, .left]),
            .prose("after"),
        ])
    }

    func testTableAtTheVeryStartHasNoLeadingEmptyProse() {
        let s = "| a | b |\n|---|---|\n| 1 | 2 |"
        XCTAssertEqual(segs(s), [.table(headers: ["a", "b"], rows: [["1", "2"]], alignments: [.left, .left])])
    }

    func testPipeWithoutSeparatorStaysOneProseSegment() {
        XCTAssertEqual(segs("a | b\nplain"), [.prose("a | b\nplain")])
    }

    func testUnterminatedTableMidStreamStillRendersAsTable() {
        let s = "| a | b |\n|---|---|\n| 1"
        XCTAssertEqual(segs(s), [.table(headers: ["a", "b"], rows: [["1"]], alignments: [.left, .left])])
    }

    func testAsciiPseudoTableBecomesItsOwnSegment() {
        let s = "Tip        Explanation\n─────────  ──────────────────\n`$@`       preserve args"
        XCTAssertEqual(segs(s), [
            .table(headers: ["Tip", "Explanation"], rows: [["`$@`", "preserve args"]], alignments: [.left, .left]),
        ])
    }

    func testTableThenCodeFenceBothSplitOut() {
        let s = "| a | b |\n|---|---|\n| 1 | 2 |\n```\nx\n```"
        XCTAssertEqual(segs(s), [
            .table(headers: ["a", "b"], rows: [["1", "2"]], alignments: [.left, .left]),
            .code(language: "", code: "x"),
        ])
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd app && swift test --filter MarkdownSegmenterTests`
Expected: build failure — `Segment` has no `.table` case yet.

- [ ] **Step 3: Add the `.table` case and the split**

Full replacement of the `Segment` enum and the `segments(_:)` loop in
`MarkdownSegmenter.swift`:

```swift
    enum Segment: Equatable {
        case prose(String)
        case code(language: String, code: String)
        case table(headers: [String], rows: [[String]], alignments: [TableAlignment])
    }
```

```swift
    static func segments(_ source: String) -> [Segment] {
        var out: [Segment] = []
        var prose: [String] = []
        let lines = source.components(separatedBy: "\n")
        var i = 0

        func flushProse() {
            let text = prose.joined(separator: "\n")
            if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                out.append(.prose(text))
            }
            prose.removeAll()
        }

        while i < lines.count {
            let line = lines[i]

            if line.hasPrefix(Self.fence) {
                flushProse()
                let language = String(line.dropFirst(Self.fence.count))
                    .trimmingCharacters(in: .whitespaces)
                var body: [String] = []
                i += 1
                while i < lines.count, !lines[i].hasPrefix(Self.fence) {
                    body.append(lines[i])
                    i += 1
                }
                if i < lines.count { i += 1 }
                out.append(.code(language: language, code: body.joined(separator: "\n")))
                continue
            }

            if let table = MarkdownTable.parse(lines: lines, start: i) {
                flushProse()
                out.append(.table(headers: table.headers, rows: table.rows, alignments: table.alignments))
                i = table.end
                continue
            }

            prose.append(line)
            i += 1
        }
        flushProse()
        return out
    }
```

Also update the file's top doc comment (currently claims prose selection
spans "paragraphs, lists and tables") — tables are now their own view, same
as code blocks:

```swift
/// Splits an assistant reply at fenced code blocks and markdown tables.
///
/// The renderer needs this because prose, code, and tables want different
/// surfaces: a run of prose becomes ONE NSTextView (so drag-selection crosses
/// paragraphs and lists in a single motion), a code block becomes a view with
/// a language header and a copy button, and a table becomes a real grid view
/// with proportional columns.
///
/// So segmentation happens at FENCES and TABLE BOUNDARIES, not at markdown
/// blocks — consecutive prose blocks must stay in one segment or selection
/// breaks at every heading. Block-level parsing still belongs to
/// `MarkdownText`, which each prose run is handed verbatim. Table detection
/// is shared with `MarkdownText.parseBlocks` via `MarkdownTable.parse` so the
/// two passes can never disagree about what is a table.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd app && swift test --filter MarkdownSegmenterTests`
Expected: PASS.

- [ ] **Step 5: Wire the new segment into `MarkdownText.body` (temporary, visually unchanged)**

`MarkdownText.body`'s `ForEach` over `MarkdownSegmenter.segments(source)` is
missing the new case — this is a compile error until handled. Wire it through
the *existing* `renderTable` (still present; Task 5 deletes it) so behavior
is unchanged until Task 4 lands the real view:

```swift
                case .table(let headers, let rows, let alignments):
                    SelectableMarkdownNSText(
                        attributed: Self.renderTable(headers: headers, rows: rows,
                                                      alignments: alignments, theme: latexTheme)
                    )
```

Add this case alongside the existing `.prose`/`.code` cases in the outer
`switch segment` in `MarkdownText.body` (the `ForEach(Array(MarkdownSegmenter.segments(source).enumerated())`
loop).

- [ ] **Step 6: Build and smoke-test**

Run: `bash app/build.sh`
Expected: builds clean. Launch the app (`open "app/MLX Core.app"` or via
`build.sh`'s own launch step if it has one) and confirm a message containing
a markdown table still renders as monospaced text, unchanged from before this
task — this task only moves *where* the table is detected, not how it looks
yet.

- [ ] **Step 7: Commit**

```bash
git add app/Sources/MLXServe/Services/MarkdownSegmenter.swift app/Sources/MLXServe/Views/ChatView.swift app/Tests/MLXCoreTests/MarkdownSegmenterTests.swift
git commit -m "feat: split markdown tables into their own segment"
```

---

## Task 4: `MarkdownTableView` — real grid rendering

**Files:**
- Create: `app/Sources/MLXServe/Views/MarkdownTableView.swift`
- Modify: `app/Sources/MLXServe/Views/ChatView.swift` (`renderInline` becomes
  `internal` + gains a `weight` parameter; `MarkdownText.body`'s `.table` case
  swaps to `MarkdownTableView`)

**Interfaces:**
- Consumes: `MarkdownTable.columnWidths(headers:rows:)` (Task 2),
  `TableAlignment` (Task 1), `MarkdownText.renderInline(_:theme:weight:fontSize:)`
  (widened in this task from `private` to internal, with a new `weight`
  parameter), `LaTeXTheme` (existing type — `.light`/`.dark`, used by
  `MarkdownText`).
- Produces: `struct MarkdownTableView: View { let headers: [String]; let
  rows: [[String]]; let alignments: [TableAlignment] }` — the type Task 3's
  temporary wiring in `MarkdownText.body` is replaced with.

Minimal-GFM style, per the approved design: semibold header row, one
`Divider()` under it, no vertical borders. Columns take proportional widths
from `MarkdownTable.columnWidths`, measured once via `onGeometryChange` so
cells can wrap instead of overflowing. Cell text goes through
`renderInline` so `**bold**`, `` `code` ``, links, and inline math inside a
cell render exactly like they do in a paragraph.

- [ ] **Step 1: Widen `renderInline`'s visibility and add a `weight` parameter**

In `ChatView.swift`, change the `renderInline` signature (search `private
static func renderInline`) from:

```swift
    private static func renderInline(
        _ text: String,
        theme: LaTeXTheme,
        fontSize: CGFloat = ChatMetrics.transcriptFontSize
    ) -> NSAttributedString {
        let bodyFont = NSFont.systemFont(ofSize: fontSize)
```

to:

```swift
    static func renderInline(
        _ text: String,
        theme: LaTeXTheme,
        weight: NSFont.Weight = .regular,
        fontSize: CGFloat = ChatMetrics.transcriptFontSize
    ) -> NSAttributedString {
        let bodyFont = NSFont.systemFont(ofSize: fontSize, weight: weight)
```

No other line in the function changes — `applyInlineTypography` already
applies `bodyFont` to every span that doesn't already carry an explicit font
from the markdown parser (i.e. every span, since Foundation's
markdown-to-`AttributedString` bridge conveys bold/italic via presentation
intent, not an explicit font), so passing `weight: .semibold` bolds the whole
rendered span — exactly what a header row needs.

This is a signature-widening change with one new defaulted parameter — every
existing call site (`renderInline(text, theme:)` inside `MarkdownText`'s own
`.paragraph`/`.heading`/`.listItem` handling) keeps compiling unchanged.

- [ ] **Step 2: Create `MarkdownTableView.swift`**

```swift
import SwiftUI

/// A parsed markdown table as a real grid: semibold header row, one divider,
/// no vertical borders — the "minimal GFM" look chat UIs use, replacing the
/// monospaced space-padded text `MarkdownSegmenter`/`MarkdownText` used to
/// assemble inline. Columns take a proportional share of the available width
/// (`MarkdownTable.columnWidths`) so a short "Tip" column doesn't eat as much
/// room as a long "Explanation" one, and cells wrap instead of overflowing.
struct MarkdownTableView: View {
    let headers: [String]
    let rows: [[String]]
    let alignments: [TableAlignment]

    @Environment(\.colorScheme) private var colorScheme
    /// The table's laid-out width, measured once via `onGeometryChange`. Zero
    /// for the first frame, during which columns fall back to equal flexible
    /// widths rather than collapsing to zero.
    @State private var tableWidth: CGFloat = 0

    private var theme: LaTeXTheme { colorScheme == .dark ? .dark : .light }
    private var fractions: [CGFloat] { MarkdownTable.columnWidths(headers: headers, rows: rows) }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            row(headers, bold: true)
            Divider()
            ForEach(Array(rows.enumerated()), id: \.offset) { _, r in
                row(r, bold: false)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .onGeometryChange(for: CGFloat.self) { proxy in
            proxy.size.width
        } action: { tableWidth = $0 }
        .textSelection(.enabled)
    }

    private func row(_ cells: [String], bold: Bool) -> some View {
        HStack(alignment: .top, spacing: 0) {
            ForEach(headers.indices, id: \.self) { j in
                cell(j < cells.count ? cells[j] : "", column: j, bold: bold)
            }
        }
        .padding(.vertical, 4)
    }

    private func alignment(_ column: Int) -> Alignment {
        guard column < alignments.count else { return .leading }
        switch alignments[column] {
        case .left: return .leading
        case .right: return .trailing
        case .center: return .center
        }
    }

    @ViewBuilder
    private func cell(_ text: String, column: Int, bold: Bool) -> some View {
        let attributed = MarkdownText.renderInline(text, theme: theme, weight: bold ? .semibold : .regular)
        let label = Text(AttributedString(attributed))
            .padding(.horizontal, 8)
        if tableWidth > 0, column < fractions.count {
            label.frame(width: tableWidth * fractions[column], alignment: alignment(column))
        } else {
            label.frame(maxWidth: .infinity, alignment: alignment(column))
        }
    }
}
```

- [ ] **Step 3: Swap `MarkdownText.body`'s `.table` case to the new view**

Replace the temporary wiring added in Task 3, Step 5:

```swift
                case .table(let headers, let rows, let alignments):
                    MarkdownTableView(headers: headers, rows: rows, alignments: alignments)
```

- [ ] **Step 4: Build**

Run: `bash app/build.sh`
Expected: builds clean — no test to run here (view code; verified visually
next), but confirm no compiler errors/warnings about the new file.

- [ ] **Step 5: Visual verification**

Launch the app, open (or start) a chat, and send a prompt that returns a
table, e.g. ask the model: "Give me a markdown table of 3 bash tips with a
Tip and Explanation column." Confirm:
- Header row is bold, sits above a single horizontal divider, no vertical
  lines between columns.
- The narrow "Tip" column is visibly narrower than the wide "Explanation"
  column (proportional widths, not equal split).
- `**bold**`, `` `code` `` spans, and links inside cells render as bold/code/
  link, not literal markup characters.
- Long explanation text wraps inside its column instead of overflowing or
  getting clipped.
- While the reply is still streaming, the table renders progressively (rows
  appear as they arrive) rather than staying blank until the whole message
  completes.
- Dragging to select table text and copying it works (`.textSelection`).
- Toggle System Settings' appearance (or use the app's own dark/light toggle
  if present) and confirm the header/divider/text colors flip correctly.

- [ ] **Step 6: Commit**

```bash
git add app/Sources/MLXServe/Views/MarkdownTableView.swift app/Sources/MLXServe/Views/ChatView.swift
git commit -m "feat: render markdown tables as a real grid instead of monospaced text"
```

---

## Task 5: Remove the now-dead monospaced table code

**Files:**
- Modify: `app/Sources/MLXServe/Views/ChatView.swift`

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing new — pure deletion, verified by the full existing test
  suite staying green and the build staying clean.

After Task 4, `MarkdownSegmenter` always splits tables out before
`MarkdownText.parseBlocks` ever sees the source (Task 3), so the `.table`
case in `parseBlocks`/`Block` and `renderTable` are unreachable. Two
exceptions to check before deleting: (1) `Block.table` itself — remove the
case since nothing constructs it anymore; (2) `replaceInlineMathAttachments`
— confirm its only caller is `renderTable` before deleting it.

- [ ] **Step 1: Confirm nothing else references the code being deleted**

Run: `cd app && grep -n "replaceInlineMathAttachments\|renderTable(\|case table\|\.table(" Sources/MLXServe/Views/ChatView.swift`
Expected output: `replaceInlineMathAttachments` appears exactly twice (its
`func` line and its one call site inside `renderTable`); `renderTable(`
appears exactly twice (its `func` line and its one call site inside
`MarkdownText.body`'s old `.table` case, already replaced in Task 4 — so this
should show only the `func` line, meaning zero remaining callers); `case
table`/`.table(` appear only in `Block`'s enum declaration and its one
constructing site inside `parseBlocks`.

- [ ] **Step 2: Delete the dead code**

In `ChatView.swift`, delete:
1. `fileprivate enum Block`'s `case table([String], [[String]],
   [TableAlignment])` line.
2. The `if let table = MarkdownTable.parse(lines: lines, start: i) { ... }`
   block inside `parseBlocks` (the one from Task 1 Step 5 — it's now dead
   because the segmenter already stripped every table out of the source
   before `parseBlocks` runs on it).
3. The `case .table(let headers, let rows, let alignments): result.append(renderTable(...))`
   arm inside `buildAttributedString`'s `switch block`.
4. `private static func renderTable(...) -> NSAttributedString { ... }` in
   full (the "Render a markdown table as monospaced columns..." doc comment
   through its closing brace).
5. `private static func replaceInlineMathAttachments(...)` in full, and its
   doc comment ("Tables deliberately keep their raw monospaced layout...").

- [ ] **Step 3: Run the full markdown-related test suite**

Run: `cd app && swift test --filter MarkdownTableTests --filter MarkdownSegmenterTests --filter MarkdownLinkTests --filter LaTeXRenderingTests --filter LaTeXSegmenterTests`
Expected: PASS — deleting rendering code that's already unreachable must not
change any parser-level test outcome.

- [ ] **Step 4: Build**

Run: `bash app/build.sh`
Expected: builds clean, zero warnings about unused code in the touched
region.

- [ ] **Step 5: Commit**

```bash
git add app/Sources/MLXServe/Views/ChatView.swift
git commit -m "refactor: remove dead monospaced table rendering"
```

---

## Task 6: Content column — 740pt fixed → 80% of the panel's visible width

**Files:**
- Modify: `app/Sources/MLXServe/Views/ChatMetrics.swift:17` (replace
  `contentMaxWidth` with `contentWidthFraction` + `contentFallbackWidth`)
- Modify: `app/Sources/MLXServe/Views/ChatView.swift` (`ChatDetailView`: add
  `columnWidth` state + `contentWidth` computed property + `onGeometryChange`
  on the body's root `VStack`; swap the three `.frame(maxWidth:
  ChatMetrics.contentMaxWidth)` sites at lines 2051, 2146, 2280 to
  `.frame(maxWidth: contentWidth)`)
- Test: `app/Tests/MLXCoreTests/ChatColumnMetricsTests.swift` (rewrite)

**Interfaces:**
- Produces: `ChatMetrics.contentWidthFraction: CGFloat` (0.8),
  `ChatMetrics.contentFallbackWidth: CGFloat` (740 — used only for the one
  frame before `ChatDetailView`'s geometry has been measured);
  `ChatDetailView.contentWidth: CGFloat` (a private computed property other
  tasks don't consume — internal to this task).

`ChatMetrics.contentMaxWidth` was a fixed 740pt, applied identically at three
sites in `ChatDetailView` (transcript, composer, empty-state greeting) so all
three stay aligned. On a wide panel that's roughly 45–50% of the visible
detail column (the column next to the session sidebar, not the whole
window), leaving large dead margins. The fix keeps the "one shared measure,
applied at all three sites" invariant but makes the measure a fraction of
the *detail column's* measured width (not the window's — the panel has a
180–280pt sidebar, so window-relative sizing would overshoot).

- [ ] **Step 1: Write the failing test for the new constant**

Rewrite `ChatColumnMetricsTests.swift` in full:

```swift
import XCTest
@testable import MLXCore

/// Reading measure for the chat column.
///
/// The measure used to be a fixed 740pt: on a fullscreen window that left
/// roughly 40-50% dead space on either side, because the panel next to the
/// session sidebar is wider than 740/0.8. It is now proportional — 80% of
/// the detail column's measured width, centered — so the column grows with
/// the window instead of stopping at a hardcoded cap. The transcript, the
/// composer, and the empty-state greeting all share the same measure: a
/// composer narrower or wider than the answer above it is the seam you
/// can't unsee.
final class ChatColumnMetricsTests: XCTestCase {

    /// Pinned so "make it a bit wider" is a decision someone makes on
    /// purpose. Below 0.7 it's back to visible dead space; above 0.9 prose
    /// runs close enough to the window edge to lose the "column" feel.
    func testTheMeasureIsProportionalAndInTheReadableBand() {
        XCTAssertGreaterThanOrEqual(ChatMetrics.contentWidthFraction, 0.7)
        XCTAssertLessThanOrEqual(ChatMetrics.contentWidthFraction, 0.9)
    }

    /// The fallback only covers the single frame before `ChatDetailView`
    /// measures its own width — it should still sit in a sane reading-width
    /// range so that frame doesn't flash something absurd.
    func testTheFallbackIsInTheReadableRange() {
        XCTAssertGreaterThanOrEqual(ChatMetrics.contentFallbackWidth, 640)
        XCTAssertLessThanOrEqual(ChatMetrics.contentFallbackWidth, 820)
    }

    // MARK: - Source audit

    /// One measure, applied by all three. A transcript capped without the
    /// composer (or the other way round) is exactly the misalignment this
    /// replaced.
    func testTranscriptComposerAndGreetingAllUseTheSharedMeasure() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Views/ChatView.swift")
        let source = try String(contentsOf: url, encoding: .utf8)
        let uses = source.components(separatedBy: "maxWidth: contentWidth").count - 1
        XCTAssertGreaterThanOrEqual(uses, 3, """
            The reading measure must be applied to the transcript, the composer \
            AND the empty state's greeting block — found \(uses) use(s).
            """)
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd app && swift test --filter ChatColumnMetricsTests`
Expected: FAIL — `ChatMetrics.contentWidthFraction`/`contentFallbackWidth`
don't exist yet, and `maxWidth: contentWidth` isn't in the source yet.

- [ ] **Step 3: Update `ChatMetrics`**

In `ChatMetrics.swift`, replace:

```swift
    /// The reading measure — the width the transcript AND the composer are
    /// capped at, centred in whatever the window gives them.
    static let contentMaxWidth: CGFloat = 740
```

with:

```swift
    /// Fraction of the detail column's measured width the reading measure
    /// takes — the transcript, composer, and empty-state greeting are all
    /// capped at this fraction, centred in whatever the panel gives them.
    /// 0.8, not 1.0: the window can be as wide as the user wants, but prose
    /// still shouldn't run edge to edge. Pinned by `ChatColumnMetricsTests`.
    static let contentWidthFraction: CGFloat = 0.8

    /// Reading width used for the single frame before `ChatDetailView` has
    /// measured its own column (`onGeometryChange` hasn't fired yet).
    static let contentFallbackWidth: CGFloat = 740
```

- [ ] **Step 4: Add width measurement and the computed property to `ChatDetailView`**

Add a new `@State` near the view's other `@State` declarations (next to
`scrollPosition`/`pasteMonitor`, around where `@State private var
composerHeight: CGFloat = 36` is declared):

```swift
    /// The detail column's measured width — the panel next to the session
    /// sidebar, not the whole window. Drives `contentWidth` below. Zero until
    /// `body`'s root view reports its first `onGeometryChange`.
    @State private var columnWidth: CGFloat = 0

    /// The shared reading measure all three capped sites (transcript,
    /// composer, empty-state greeting) apply. See `ChatMetrics.contentWidthFraction`.
    private var contentWidth: CGFloat {
        columnWidth > 0 ? columnWidth * ChatMetrics.contentWidthFraction : ChatMetrics.contentFallbackWidth
    }
```

Then attach the measurement to the root `VStack(spacing: 0)` of `body`
(search for `.onChange(of: sessionId) { _, _ in` near the end of `body` —
this is the last modifier chained onto that root `VStack`; add the new
modifier immediately after that `onChange` block's closing brace, still
before `body`'s own closing brace):

```swift
        .onGeometryChange(for: CGFloat.self) { proxy in
            proxy.size.width
        } action: { columnWidth = $0 }
```

- [ ] **Step 5: Swap the three capped sites**

Replace each of the three occurrences of:

```swift
        .frame(maxWidth: ChatMetrics.contentMaxWidth)
```

(at the empty-state greeting block, the transcript `ScrollView`'s content,
and the composer) with:

```swift
        .frame(maxWidth: contentWidth)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd app && swift test --filter ChatColumnMetricsTests`
Expected: PASS (3 tests).

- [ ] **Step 7: Build and visually verify**

Run: `bash app/build.sh`, launch the app, open a chat with an existing
conversation:
- At a normal window size, the transcript/composer/greeting column is
  visibly wider than before but still centered, with roughly equal margins
  left and right.
- Resize the window to full screen (or a very wide external-display size):
  confirm the dead margins shrink dramatically compared to before this
  change — the column should now read as ~80% of the panel, not ~50%.
- Resize the window narrow: confirm the column still tracks (no longer
  visually detached from the window edges), and doesn't shrink to something
  unreadably narrow.
- Confirm the transcript, the composer, and the empty-state greeting stay
  edge-aligned with each other at every width (the "shared measure"
  invariant the removed test comment called out).

- [ ] **Step 8: Commit**

```bash
git add app/Sources/MLXServe/Views/ChatMetrics.swift app/Sources/MLXServe/Views/ChatView.swift app/Tests/MLXCoreTests/ChatColumnMetricsTests.swift
git commit -m "feat: size the chat content column to 80% of the panel width"
```

---

## Task 7: End-to-end verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `cd app && swift test`
Expected: PASS, zero failures, zero new warnings introduced by this plan's
changes.

- [ ] **Step 2: Full build**

Run: `bash app/build.sh`
Expected: clean build.

- [ ] **Step 3: Combined visual smoke test**

Launch the app and, in one conversation:
1. Ask for a markdown table with at least 3 columns of visibly different
   content lengths (e.g. the bash-tips example from this plan's design
   discussion: `Tip | Explanation`, plus a third narrow column like
   `Command`).
2. While the reply streams, confirm the table appears progressively and
   settles into its final proportional-width layout without a visible
   "jump" once streaming finishes.
3. Confirm the table's width matches the width of the surrounding prose
   paragraphs in the same reply (both bounded by the same `contentWidth`).
4. Resize the window full-screen mid-conversation; confirm both the table
   and the surrounding prose grow together, staying edge-aligned.
5. Drag-select across a table cell and a paragraph in the same reply
   separately (selection is per-segment now, same as it already was for code
   blocks) and copy each; confirm both copy their visible text.
6. Open `QuickLauncherView` (or any other `MarkdownText` call site) with a
   reply containing a table and confirm it renders the same way — this view
   gets the new table rendering "for free" since it shares `MarkdownText`.

- [ ] **Step 4: Review the diff for leftover references**

Run: `cd app && grep -rn "contentMaxWidth" Sources/MLXServe/Views/ChatView.swift Sources/MLXServe/Views/ChatMetrics.swift`
Expected: no matches (all three sites and the constant itself were replaced
in Task 6; `AgentEditorChrome.swift`'s unrelated `AgentEditorMetrics.contentMaxWidth`
is a different, untouched constant and won't appear in this grep since it's a
different file).

No commit for this task — it's verification of the six prior commits.
