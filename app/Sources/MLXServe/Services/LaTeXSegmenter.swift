import Foundation

/// Separates TeX from assistant prose without interpreting it. Rendering is
/// intentionally delegated to SwaTex; this type only owns delimiter safety.
enum LaTeXSegmenter {
    enum Segment: Equatable, Sendable {
        case text(String)
        case inline(latex: String, raw: String)
        case display(latex: String, raw: String)
    }

    /// Avoid sending an accidentally unbounded streaming fragment through the
    /// typesetter. Oversized and incomplete expressions remain literal text.
    static let maximumFormulaLength = 16_384

    private static let displayEnvironments: Set<String> = [
        "align", "align*", "aligned", "array", "bmatrix", "cases",
        "equation", "equation*", "gather", "gather*", "matrix",
        "multline", "multline*", "pmatrix", "smallmatrix", "split",
        "Vmatrix", "vmatrix",
    ]

    static func segments(_ source: String) -> [Segment] {
        guard !source.isEmpty else { return [.text("")] }

        var result: [Segment] = []
        var plainStart = source.startIndex
        var cursor = source.startIndex

        while cursor < source.endIndex {
            if source[cursor] == "`", !isEscaped(cursor, in: source) {
                cursor = indexAfterMarkdownCodeSpan(startingAt: cursor, in: source)
                continue
            }

            if let match = environmentMatch(startingAt: cursor, in: source) {
                appendText(source[plainStart..<cursor], to: &result)
                result.append(.display(latex: match.latex, raw: match.raw))
                cursor = match.end
                plainStart = match.end
                continue
            }

            if source[cursor...].hasPrefix("$$"), !isEscaped(cursor, in: source),
               let match = delimitedMatch(
                   startingAt: cursor,
                   opening: "$$",
                   closing: "$$",
                   in: source,
                   style: .display
               ) {
                appendText(source[plainStart..<cursor], to: &result)
                result.append(match.segment)
                cursor = match.end
                plainStart = match.end
                continue
            }

            if source[cursor...].hasPrefix(#"\["#), !isEscaped(cursor, in: source),
               let match = delimitedMatch(
                   startingAt: cursor,
                   opening: #"\["#,
                   closing: #"\]"#,
                   in: source,
                   style: .display
               ) {
                appendText(source[plainStart..<cursor], to: &result)
                result.append(match.segment)
                cursor = match.end
                plainStart = match.end
                continue
            }

            if source[cursor...].hasPrefix(#"\("#), !isEscaped(cursor, in: source),
               let match = delimitedMatch(
                   startingAt: cursor,
                   opening: #"\("#,
                   closing: #"\)"#,
                   in: source,
                   style: .inline
               ) {
                appendText(source[plainStart..<cursor], to: &result)
                result.append(match.segment)
                cursor = match.end
                plainStart = match.end
                continue
            }

            if source[cursor] == "$",
               !source[cursor...].hasPrefix("$$"),
               (cursor == source.startIndex || source[source.index(before: cursor)] != "$"),
               !isEscaped(cursor, in: source),
               isValidDollarOpening(at: cursor, in: source),
               let match = dollarMatch(startingAt: cursor, in: source) {
                appendText(source[plainStart..<cursor], to: &result)
                result.append(match.segment)
                cursor = match.end
                plainStart = match.end
                continue
            }

            cursor = source.index(after: cursor)
        }

        appendText(source[plainStart..<source.endIndex], to: &result)
        return result
    }

    private enum Style {
        case inline
        case display
    }

    private struct Match {
        let segment: Segment
        let end: String.Index
    }

    private struct EnvironmentMatch {
        let latex: String
        let raw: String
        let end: String.Index
    }

    private static func delimitedMatch(
        startingAt start: String.Index,
        opening: String,
        closing: String,
        in source: String,
        style: Style
    ) -> Match? {
        let contentStart = source.index(start, offsetBy: opening.count)
        guard let closingStart = closingIndex(
            after: contentStart,
            delimiter: closing,
            in: source,
            stopAtFirstDollar: false
        ) else { return nil }

        let end = source.index(closingStart, offsetBy: closing.count)
        let raw = String(source[start..<end])
        let latex = String(source[contentStart..<closingStart])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !latex.isEmpty, latex.count <= maximumFormulaLength else { return nil }

        let segment: Segment = switch style {
        case .inline: .inline(latex: latex, raw: raw)
        case .display: .display(latex: latex, raw: raw)
        }
        return Match(segment: segment, end: end)
    }

    private static func dollarMatch(startingAt start: String.Index, in source: String) -> Match? {
        let contentStart = source.index(after: start)
        guard let closingStart = closingIndex(
            after: contentStart,
            delimiter: "$",
            in: source,
            stopAtFirstDollar: true
        ), !source[closingStart...].hasPrefix("$$"),
           isValidDollarClosing(at: closingStart, in: source) else { return nil }

        let end = source.index(after: closingStart)
        let raw = String(source[start..<end])
        let latex = String(source[contentStart..<closingStart])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !latex.isEmpty, latex.count <= maximumFormulaLength else { return nil }
        return Match(segment: .inline(latex: latex, raw: raw), end: end)
    }

    private static func closingIndex(
        after start: String.Index,
        delimiter: String,
        in source: String,
        stopAtFirstDollar: Bool
    ) -> String.Index? {
        var cursor = start
        var scanned = 0

        while cursor < source.endIndex, scanned <= maximumFormulaLength {
            if delimiter == "$", source[cursor] == "`", !isEscaped(cursor, in: source) {
                let spanEnd = indexAfterMarkdownCodeSpan(startingAt: cursor, in: source)
                scanned += source.distance(from: cursor, to: spanEnd)
                cursor = spanEnd
                continue
            }
            if source[cursor...].hasPrefix(delimiter), !isEscaped(cursor, in: source) {
                if delimiter != "$" || !source[cursor...].hasPrefix("$$") {
                    return cursor
                }
            }
            if stopAtFirstDollar, source[cursor] == "$", !isEscaped(cursor, in: source) {
                return cursor
            }
            cursor = source.index(after: cursor)
            scanned += 1
        }
        return nil
    }

    private static func environmentMatch(
        startingAt start: String.Index,
        in source: String
    ) -> EnvironmentMatch? {
        let prefix = #"\begin{"#
        guard source[start...].hasPrefix(prefix), !isEscaped(start, in: source) else { return nil }

        let nameStart = source.index(start, offsetBy: prefix.count)
        guard let nameEnd = source[nameStart...].firstIndex(of: "}"),
              source.distance(from: nameStart, to: nameEnd) <= 24 else { return nil }
        let name = String(source[nameStart..<nameEnd])
        guard displayEnvironments.contains(name) else { return nil }

        let closing = "\\end{\(name)}"
        let contentStart = source.index(after: nameEnd)
        guard let closingRange = source.range(of: closing, range: contentStart..<source.endIndex) else {
            return nil
        }
        let end = closingRange.upperBound
        let raw = String(source[start..<end])
        guard raw.count <= maximumFormulaLength else { return nil }
        return EnvironmentMatch(latex: raw, raw: raw, end: end)
    }

    private static func isValidDollarOpening(at index: String.Index, in source: String) -> Bool {
        let next = source.index(after: index)
        guard next < source.endIndex else { return false }
        return !source[next].isWhitespace
    }

    private static func isValidDollarClosing(at index: String.Index, in source: String) -> Bool {
        guard index > source.startIndex else { return false }
        let previous = source[source.index(before: index)]
        guard !previous.isWhitespace else { return false }

        let next = source.index(after: index)
        return next == source.endIndex || !source[next].isNumber
    }

    private static func isEscaped(_ index: String.Index, in source: String) -> Bool {
        var cursor = index
        var slashCount = 0
        while cursor > source.startIndex {
            let previous = source.index(before: cursor)
            guard source[previous] == "\\" else { break }
            slashCount += 1
            cursor = previous
        }
        return slashCount.isMultiple(of: 2) == false
    }

    private static func indexAfterMarkdownCodeSpan(
        startingAt start: String.Index,
        in source: String
    ) -> String.Index {
        var openerEnd = start
        var tickCount = 0
        while openerEnd < source.endIndex, source[openerEnd] == "`" {
            tickCount += 1
            openerEnd = source.index(after: openerEnd)
        }

        var cursor = openerEnd
        while cursor < source.endIndex {
            guard source[cursor] == "`" else {
                cursor = source.index(after: cursor)
                continue
            }

            var runEnd = cursor
            var runCount = 0
            while runEnd < source.endIndex, source[runEnd] == "`" {
                runCount += 1
                runEnd = source.index(after: runEnd)
            }
            if runCount == tickCount { return runEnd }
            cursor = runEnd
        }

        // An unfinished Markdown code span owns the rest of the streamed text.
        return source.endIndex
    }

    private static func appendText(_ text: Substring, to result: inout [Segment]) {
        guard !text.isEmpty else { return }
        let value = String(text)
        if case .text(let previous)? = result.last {
            result[result.count - 1] = .text(previous + value)
        } else {
            result.append(.text(value))
        }
    }
}
