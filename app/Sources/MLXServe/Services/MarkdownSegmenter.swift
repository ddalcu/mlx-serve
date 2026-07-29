import Foundation

/// Splits an assistant reply at fenced code blocks.
///
/// The renderer needs this because prose and code want different surfaces: a
/// run of prose becomes ONE NSTextView (so drag-selection crosses paragraphs,
/// lists and tables in a single motion) while a code block becomes a view with
/// a language header, a line-number gutter and a copy button.
///
/// So segmentation happens at FENCES, not at markdown blocks — consecutive
/// prose blocks must stay in one segment or selection breaks at every heading.
/// Block-level parsing still belongs to `MarkdownText`, which each prose run is
/// handed verbatim.
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

        /// Flush the pending prose run. Whitespace-only runs are dropped: an
        /// empty text view between two blocks renders as a stray gap.
        func flushProse() {
            let text = prose.joined(separator: "\n")
            if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                out.append(.prose(text))
            }
            prose.removeAll()
        }

        while i < lines.count {
            let line = lines[i]
            guard line.hasPrefix(Self.fence) else {
                prose.append(line)
                i += 1
                continue
            }
            flushProse()
            let language = String(line.dropFirst(Self.fence.count))
                .trimmingCharacters(in: .whitespaces)
            var body: [String] = []
            i += 1
            while i < lines.count, !lines[i].hasPrefix(Self.fence) {
                body.append(lines[i])
                i += 1
            }
            // Unterminated fence (mid-stream) — emit what exists rather than
            // letting a half-written block reflow as prose until it closes.
            if i < lines.count { i += 1 }
            out.append(.code(language: language, code: body.joined(separator: "\n")))
        }
        flushProse()
        return out
    }
}
