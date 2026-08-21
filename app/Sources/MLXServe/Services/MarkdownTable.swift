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
/// leaves table lines in the prose it hands to `MarkdownText`) and
/// `MarkdownText.parseBlocks` (which calls this to detect and render them),
/// so the two passes can never disagree about what is a table.
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

    /// Column weight for `NSTextTable`'s percentage-of-container width: each
    /// column's share is proportional to its longest cell's character
    /// count, floored so an empty or all-short column still gets a visible
    /// sliver instead of collapsing to nothing. `NSTextTable` has no native
    /// "tight to content" sizing mode — a table always fills the available
    /// width — so these are relative weights, not point widths.
    static func columnFractions(headers: [String], rows: [[String]]) -> [CGFloat] {
        let cols = headers.count
        guard cols > 0 else { return [] }
        var longest = headers.map { CGFloat($0.count) }
        for row in rows {
            for (j, cell) in row.prefix(cols).enumerated() {
                longest[j] = max(longest[j], CGFloat(cell.count))
            }
        }
        let floor: CGFloat = 3
        let weighted = longest.map { max($0, floor) }
        let total = weighted.reduce(0, +)
        guard total > 0 else { return Array(repeating: 1.0 / CGFloat(cols), count: cols) }
        return weighted.map { $0 / total }
    }
}
