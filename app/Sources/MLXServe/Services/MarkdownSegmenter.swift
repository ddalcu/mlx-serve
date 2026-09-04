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
///
/// A third surface joins them: an `html`/`svg` fence the model has CLOSED
/// becomes `.html` and is mounted as a live document (`HTMLArtifactView`), so
/// a reply can answer with a chart or a small widget rather than describing
/// one. What qualifies is `HTMLArtifact.rendersLive` — asked, never restated
/// here, because a second copy of that rule is how a web view ends up mounted
/// over source the predicate had already declined.
enum MarkdownSegmenter {

    enum Segment: Equatable {
        case prose(String)
        case code(language: String, code: String)
        /// A CLOSED fence whose language renders as a live document
        /// (`HTMLArtifact.rendersLive`) — mounted as a web view rather than
        /// coloured as source.
        ///
        /// Closed is the load-bearing word. Every html reply passes through a
        /// half-written document on its way to a closed fence, and mounting one
        /// executes a script whose function bodies are still arriving, then
        /// reloads it on the next token. Until the closing fence lands the
        /// block is `.code`, exactly as it was before this existed.
        case html(language: String, code: String)
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
                // Only a fence the model actually closed can render; running
                // out of lines means the reply is still streaming.
                let closed = i < lines.count
                if closed { i += 1 }
                let code = body.joined(separator: "\n")
                if closed, HTMLArtifact.rendersLive(language: language, code: code) {
                    out.append(.html(language: language, code: code))
                } else {
                    out.append(.code(language: language, code: code))
                }
                continue
            }

            prose.append(line)
            i += 1
        }
        flushProse()
        return out
    }
}
