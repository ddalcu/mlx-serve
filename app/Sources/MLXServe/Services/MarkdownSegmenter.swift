import Foundation

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
enum MarkdownSegmenter {

    enum Segment: Equatable {
        case prose(String)
        case code(language: String, code: String)
        case table(headers: [String], rows: [[String]], alignments: [TableAlignment])
    }

    /// Fences are matched exactly as `MarkdownText.parseBlocks` matches them —
    /// a line STARTING with three backticks — so the two passes can never
    /// disagree about what is code.
    private static let fence = "```"

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
}
