import Foundation

/// Splits an assistant reply at fenced code blocks only.
///
/// The renderer needs this because prose and code want different surfaces: a
/// run of prose becomes ONE NSTextView (so drag-selection crosses paragraphs,
/// lists, and tables in a single motion — see `MarkdownText.parseBlocks`,
/// which detects tables via `MarkdownTable.parse` and renders them as an
/// `NSTextTable` inside that same continuous run), while a code block becomes
/// a view with a language header and a copy button.
///
/// So segmentation happens at FENCES, not at markdown blocks — consecutive
/// prose blocks (including tables) must stay in one segment or selection
/// breaks at every boundary. Block-level parsing belongs to `MarkdownText`,
/// which each prose run is handed verbatim.
enum MarkdownSegmenter {

    enum Segment: Equatable {
        case prose(String)
        case code(language: String, code: String)
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

            prose.append(line)
            i += 1
        }
        flushProse()
        return out
    }
}
