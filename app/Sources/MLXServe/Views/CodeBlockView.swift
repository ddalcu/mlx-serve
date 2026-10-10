import SwiftUI
import AppKit

/// Colors for a rendered code block.
///
/// Every value is a DYNAMIC `NSColor`, resolved per appearance, because a chat
/// transcript is read in both light and dark mode and a block hard-coded for one
/// is unreadable in the other. Hues follow the Xcode/VS Code convention most
/// people already read code in, so the mapping needs no learning.
enum CodeTheme {

    /// Every colour is built ONCE. The dynamic provider is what makes a colour
    /// appearance-aware, but constructing one is not free and the block asks for
    /// a colour per syntax run — building them per call allocated thousands of
    /// `NSColor`s on every render of a large block.
    private static func dynamic(light: NSColor, dark: NSColor) -> NSColor {
        NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light
        }
    }

    /// Block background. Deliberately a small step off the surrounding surface
    /// rather than pure black/white — the block should read as inset, not as a
    /// hole punched in the transcript.
    static let backgroundNS = dynamic(
        light: NSColor(red: 0.96, green: 0.96, blue: 0.97, alpha: 1),
        dark: NSColor(red: 0.11, green: 0.11, blue: 0.13, alpha: 1))
    static let headerNS = dynamic(
        light: NSColor(red: 0.92, green: 0.92, blue: 0.94, alpha: 1),
        dark: NSColor(red: 0.15, green: 0.15, blue: 0.17, alpha: 1))
    static let borderNS = dynamic(
        light: NSColor(white: 0.0, alpha: 0.10),
        dark: NSColor(white: 1.0, alpha: 0.10))
    static let plainTextNS = dynamic(
        light: NSColor(white: 0.15, alpha: 1),
        dark: NSColor(white: 0.90, alpha: 1))
    static let background = Color(nsColor: backgroundNS)
    static let header = Color(nsColor: headerNS)
    static let border = Color(nsColor: borderNS)

    private static let kindColors: [SyntaxKind: NSColor] = [
        .keyword: dynamic(
            light: NSColor(red: 0.61, green: 0.11, blue: 0.55, alpha: 1),
            dark: NSColor(red: 0.78, green: 0.57, blue: 0.92, alpha: 1)),
        .type: dynamic(
            light: NSColor(red: 0.06, green: 0.48, blue: 0.42, alpha: 1),
            dark: NSColor(red: 0.31, green: 0.81, blue: 0.69, alpha: 1)),
        .function: dynamic(
            light: NSColor(red: 0.16, green: 0.36, blue: 0.75, alpha: 1),
            dark: NSColor(red: 0.51, green: 0.67, blue: 1.00, alpha: 1)),
        .property: dynamic(
            light: NSColor(red: 0.63, green: 0.35, blue: 0.00, alpha: 1),
            dark: NSColor(red: 1.00, green: 0.80, blue: 0.42, alpha: 1)),
        .string: dynamic(
            light: NSColor(red: 0.12, green: 0.48, blue: 0.24, alpha: 1),
            dark: NSColor(red: 0.65, green: 0.84, blue: 0.65, alpha: 1)),
        .number: dynamic(
            light: NSColor(red: 0.70, green: 0.28, blue: 0.00, alpha: 1),
            dark: NSColor(red: 0.97, green: 0.55, blue: 0.42, alpha: 1)),
        .comment: dynamic(
            light: NSColor(white: 0.43, alpha: 1),
            dark: NSColor(white: 0.52, alpha: 1)),
    ]

    /// The colour the text system paints with. `nil` ⇒ unclassified, which the
    /// lexer leaves deliberately plain.
    static func nsColor(for kind: SyntaxKind?) -> NSColor {
        kind.flatMap { kindColors[$0] } ?? plainTextNS
    }
}

/// Builds the attributed string a code block draws, coloured by the lexer's
/// spans.
///
/// Pure, so the thing that used to be spread across a view body is testable.
/// Colouring by `NSRange` is why `SyntaxSpan` carries UTF-16 offsets — no
/// per-line span splitting, no re-emitting a block comment on each row it
/// crosses, and no arithmetic that an emoji can shift.
enum CodeBlockText {

    static let font = NSFont.monospacedSystemFont(ofSize: CodeBlockLayout.fontSize, weight: .regular)

    /// The text system's own line height for `font`, resolved once. Every line
    /// of a code block is this tall — one font, no attachments, no wrapping.
    static let lineHeight: CGFloat = NSLayoutManager().defaultLineHeight(for: font)

    private static let paragraph: NSParagraphStyle = {
        let p = NSMutableParagraphStyle()
        p.lineSpacing = CodeBlockLayout.lineSpacing
        return p
    }()

    static func code(_ source: String, language: SyntaxLanguage?) -> NSAttributedString {
        let out = NSMutableAttributedString(string: source, attributes: [
            .font: font,
            .paragraphStyle: paragraph,
            .foregroundColor: CodeTheme.plainTextNS,
        ])
        guard let language, !source.isEmpty else { return out }
        let length = (source as NSString).length
        for span in SyntaxHighlighter.spans(source, language: language) {
            // The lexer's own invariant test pins spans in bounds; clamp anyway
            // rather than let a future lexer bug raise out of a view body.
            let end = min(span.start + span.length, length)
            guard span.start >= 0, end > span.start else { continue }
            out.addAttribute(.foregroundColor, value: CodeTheme.nsColor(for: span.kind),
                             range: NSRange(location: span.start, length: end - span.start))
        }
        return out
    }

    /// The size the text system WOULD lay this block out at, computed instead of
    /// laid out — `nil` when the arithmetic cannot be exact and the caller must
    /// measure for real.
    ///
    /// Asking TextKit costs a full layout of every line, including the ones off
    /// screen (6.4 ms for 300 lines), and a streaming block pays it on every
    /// flush. A monospaced block that never wraps doesn't need one: every line
    /// is `lineHeight` tall and every character is one advance wide, so both
    /// axes are arithmetic. Exact, not an estimate — pinned against TextKit's
    /// own answer in `CodeBlockTextTests`, because a height that drifts clips
    /// the last lines or leaves a gap under them.
    ///
    /// Declines on anything that breaks "one advance per character": a tab snaps
    /// to a tab stop, and a wide or non-Latin glyph is not one advance.
    static func measuredSize(of source: String) -> NSSize? {
        let text = source as NSString
        // Empty storage is the one case the arithmetic gets wrong: TextKit sizes
        // it from its extra line fragment, not from the font's line height. It
        // is also free to lay out, so measure it.
        guard text.length > 0 else { return nil }
        var lines = 1, longest = 0, current = 0
        for i in 0..<text.length {
            let c = text.character(at: i)
            if c == 0x0A { lines += 1; longest = max(longest, current); current = 0; continue }
            guard c >= 0x20, c < 0x7F else { return nil }
            current += 1
        }
        longest = max(longest, current)

        let height = CGFloat(lines) * lineHeight + CGFloat(lines - 1) * CodeBlockLayout.lineSpacing
        return NSSize(width: ceil(CGFloat(longest) * font.maximumAdvancement.width), height: ceil(height))
    }

    /// The offset from which `old` and `new` stop agreeing, comparing CHARACTERS
    /// AND ATTRIBUTES — so replacing everything from there reproduces `new`
    /// exactly. `nil` when there is nothing worth patching (no shared prefix).
    ///
    /// Attributes have to be part of it: the lexer runs an unterminated string
    /// or comment to end-of-source, so the token that finally closes one
    /// re-colours text that is already on screen. A character-only diff would
    /// leave that text painted wrong for the rest of the reply.
    static func changedSuffix(from old: NSAttributedString, to new: NSAttributedString) -> Int? {
        let oldText = old.string as NSString, newText = new.string as NSString
        let limit = min(oldText.length, newText.length)
        guard limit > 0 else { return nil }

        var i = 0
        while i < limit {
            var oldRange = NSRange(), newRange = NSRange()
            let oldColor = old.attribute(.foregroundColor, at: i, effectiveRange: &oldRange)
            let newColor = new.attribute(.foregroundColor, at: i, effectiveRange: &newRange)
            guard (oldColor as? NSColor) == (newColor as? NSColor) else { break }
            // Both runs are uniform to the shorter of the two ends; compare that
            // slab of text in one go rather than a character at a time.
            let end = min(NSMaxRange(oldRange), NSMaxRange(newRange), limit)
            let range = NSRange(location: i, length: end - i)
            guard oldText.substring(with: range) == newText.substring(with: range) else { break }
            i = end
        }
        return i > 0 ? i : nil
    }
}

/// Folding a long code block to its first lines. Code never wraps, so the
/// count is exact and the fold ends on a line boundary.
enum CodeBlockFold {

    /// Lines kept while folded.
    static let collapsedLines = 20

    /// Lines a block must clear the limit by to earn a "Show more", as for a
    /// long user turn: a control that reveals a line or two is a dead one.
    static let foldMargin = 3

    /// Lines end at `\n`, so CRLF counts too; scalars, not Characters, because
    /// `\r\n` is one Character. Until the block is `finished`, the line being
    /// written does not count: it may be the closing fence arriving.
    static func lineCount(_ code: String, finished: Bool = true) -> Int {
        let lines = code.unicodeScalars.reduce(1) { $1 == "\n" ? $0 + 1 : $0 }
        return finished ? lines : lines - 1
    }

    static func folds(_ code: String, finished: Bool = true) -> Bool {
        lineCount(code, finished: finished) >= collapsedLines + foldMargin
    }

    /// What the block draws: its first lines while folded, all of it otherwise.
    static func visible(_ code: String, finished: Bool = true, expanded: Bool) -> String {
        guard !expanded, folds(code, finished: finished) else { return code }
        let scalars = code.unicodeScalars
        var newlines = 0
        for i in scalars.indices where scalars[i] == "\n" {
            newlines += 1
            guard newlines == collapsedLines else { continue }
            let crlf = i > scalars.startIndex && scalars[scalars.index(before: i)] == "\r"
            return String(code[..<(crlf ? scalars.index(before: i) : i)])
        }
        return code
    }

    static func linesTotal(_ count: Int) -> String {
        L10n.format("%lld lines total", Int64(count))
    }
}

/// What a code block's status slot carries: Copy, or a spinner while the
/// block is still being written, so half a block is not copied as if whole.
enum CodeBlockStatus: Equatable {
    case copy, streaming

    /// A fence the model never closed is finished once the reply is.
    static func of(closed: Bool, replyStreaming: Bool) -> CodeBlockStatus {
        !closed && replyStreaming ? .streaming : .copy
    }
}

/// What a transcript row hands its code blocks: the bracket around a change
/// that shortens the row, so the reader keeps their place (`ChatScroll`'s
/// `rowWillResize` / `rowDidResize`), and fold memory by block index that
/// outlives a rebuild of the transcript (`FoldStore`). Always equal: a fresh
/// one per row render must not re-render every block in the row.
struct CodeBlockRow: Equatable {
    let willResize: () -> Void
    let didResize: () -> Void
    let isExpanded: (Int) -> Bool
    let setExpanded: (Int, Bool) -> Void

    static func == (_: Self, _: Self) -> Bool { true }
}

private struct CodeBlockRowKey: EnvironmentKey {
    static let defaultValue: CodeBlockRow? = nil
}

extension EnvironmentValues {
    var codeBlockRow: CodeBlockRow? {
        get { self[CodeBlockRowKey.self] }
        set { self[CodeBlockRowKey.self] = newValue }
    }
}

/// Layout constants for a code block.
enum CodeBlockLayout {
    /// The code block's text, on the same ladder as everything else: the
    /// `callout` step, so a block reads at the same size as a chat bubble's
    /// secondary line instead of at a number of its own.
    static let fontSize = AppType.pointSize(for: .callout)
    static let cornerRadius: CGFloat = 10
    static let lineSpacing: CGFloat = 2.5
}

/// A fenced code block: language header with a copy button, and syntax-colored
/// code that scrolls horizontally. A long block folds to its first lines
/// (`CodeBlockFold`) with a Show more / Show less footer.
///
/// Rendered as its own view rather than as a run inside the message's text view.
/// That costs cross-block drag-selection (each block is now its own selection
/// island) and buys per-token color plus a copy button that yields the code
/// alone — which is what people actually do with a code block. Prose either side
/// still selects in one motion, because `MarkdownSegmenter` keeps consecutive
/// prose in a single text view.
struct CodeBlockView: View {
    /// The fence label verbatim (`swift`, `tsx`, ``). Kept raw so the header can
    /// show what the model wrote when we don't recognize it.
    let language: String
    let code: String
    var status: CodeBlockStatus = .copy
    /// Its place among the reply's segments: the key of its fold memory.
    var index = 0

    @State private var copied = false
    @State private var expanded = false
    @Environment(\.codeBlockRow) private var row

    private var resolved: SyntaxLanguage? { SyntaxLanguage(fence: language) }

    /// Header label: the fence the model wrote, else nothing to claim.
    ///
    /// Deliberately NOT the lexer's name. Several fences share one lexer, so
    /// naming it labelled a `tsx` block "JavaScript", a `java` block "C", and
    /// `svg` / `vue` / `svelte` all "HTML" — describing the highlighter rather
    /// than the code in front of you.
    private var label: String {
        let trimmed = language.trimmingCharacters(in: .whitespaces)
        return trimmed.isEmpty ? "Code" : trimmed
    }

    var body: some View {
        let finished = status == .copy
        let folds = CodeBlockFold.folds(code, finished: finished)
        // Folded, the text view holds only the kept lines, so a long block
        // still streaming past the fold lays nothing out.
        let shown = CodeBlockFold.visible(code, finished: finished, expanded: expanded)
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider().opacity(0.5)
            ScrollView(.horizontal, showsIndicators: false) {
                // ONE text view for the whole block. This was a `ForEach`
                // building a `Text` per line, so a 300-line block carried
                // hundreds of nodes in SwiftUI's attribute graph and a streaming
                // reply rebuilt every one of them per token.
                CodeNSText(attributed: CodeBlockText.code(shown, language: resolved), selectable: true,
                           computed: CodeBlockText.measuredSize(of: shown))
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
            }
            .overlay(alignment: .bottom) {
                if folds && !expanded { foldFade }
            }
            if folds {
                Divider().opacity(0.5)
                foldToggle(lines: CodeBlockFold.lineCount(code, finished: finished))
            }
        }
        .background(CodeTheme.background)
        .onAppear { expanded = row?.isExpanded(index) ?? false }
        .clipShape(RoundedRectangle(cornerRadius: CodeBlockLayout.cornerRadius))
        .overlay(
            RoundedRectangle(cornerRadius: CodeBlockLayout.cornerRadius)
                .stroke(CodeTheme.border, lineWidth: 1)
        )
    }

    private var header: some View {
        HStack(spacing: 6) {
            Text(L10n.text(label))
                .font(.app(.caption2, weight: .medium))
                .foregroundStyle(.secondary)
            Spacer()
            statusSlot(vertical: 3, trailing: 6)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(CodeTheme.header)
    }

    /// Copy, or "Streaming" while the block is still written. The button stays
    /// laid out underneath, so the header keeps its height when the block ends.
    /// The insets are inside the button: they are its hit area.
    private func statusSlot(vertical: CGFloat, trailing: CGFloat) -> some View {
        copyButton(vertical: vertical, trailing: trailing)
            .opacity(status == .copy ? 1 : 0)
            .allowsHitTesting(status == .copy)
            .accessibilityHidden(status != .copy)
            .overlay(alignment: .trailing) {
                if status == .streaming {
                    HStack(spacing: 4) {
                        ProgressView().controlSize(.mini)
                        Text(L10n.text("Streaming"))
                            .font(.app(.caption2, weight: .medium))
                    }
                    .foregroundStyle(.secondary)
                    .padding(.leading, 6)
                    .padding(.trailing, trailing)
                    .fixedSize()
                }
            }
    }

    private func copyButton(vertical: CGFloat, trailing: CGFloat) -> some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(code, forType: .string)
            copied = true
            // The tick is the whole confirmation — a copy with no feedback
            // reads as a dead button and gets clicked again.
            Task {
                try? await Task.sleep(nanoseconds: 1_400_000_000)
                copied = false
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: copied ? "checkmark" : "doc.on.doc")
                    .font(.app(.caption2, weight: .medium))
                Text(L10n.text(copied ? "Copied" : "Copy"))
                    .font(.app(.caption2, weight: .medium))
            }
            .foregroundStyle(copied ? Color.green : Color.secondary)
            .padding(.leading, 6)
            .padding(.trailing, trailing)
            .padding(.vertical, vertical)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Copy this code block")
    }

    /// Fades the last kept lines into the block, so the fold reads as more
    /// below rather than as the end. An overlay in the block's colour, not a
    /// mask, for the same reason as a folded user turn.
    private var foldFade: some View {
        LinearGradient(colors: [CodeTheme.background.opacity(0), CodeTheme.background.opacity(0.85)],
                       startPoint: .top, endPoint: .bottom)
            .frame(height: 2 * (CodeBlockText.lineHeight + CodeBlockLayout.lineSpacing) + 10)
            .allowsHitTesting(false)
    }

    /// The whole footer is the button, in the header's colours. Folded, it
    /// says how long the block is, counting while it streams; unfolded, it
    /// repeats the status slot at the right, so a long block can be copied (or
    /// seen still streaming) from its end: an overlay, hit before the toggle.
    private func foldToggle(lines: Int) -> some View {
        Button(action: toggleFold) {
            HStack(spacing: 4) {
                Text(L10n.text(expanded ? "Show less" : "Show more"))
                    .font(.app(.caption2, weight: .medium))
                if !expanded {
                    Text(verbatim: "·").font(.app(.caption2))
                    Text(CodeBlockFold.linesTotal(lines)).font(.app(.caption2))
                }
            }
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .overlay(alignment: .trailing) {
            if expanded { statusSlot(vertical: 8, trailing: 16) }
        }
        .background(CodeTheme.header)
    }

    private func toggleFold() {
        row?.setExpanded(index, !expanded)
        guard expanded else { expanded = true; return }
        // Folding shortens the row under the reader, so it is bracketed; the
        // end is reported once the shorter layout has landed.
        row?.willResize()
        expanded = false
        DispatchQueue.main.async { DispatchQueue.main.async { row?.didResize() } }
    }
}

/// A code block's text, drawn by the text system rather than by a stack of
/// `Text` views.
///
/// Never wraps — indentation is how code is read, and a soft-wrapped line
/// re-indents to nothing. The container is unbounded in width and the enclosing
/// `ScrollView` scrolls to reach the rest.
private struct CodeNSText: NSViewRepresentable {
    let attributed: NSAttributedString
    let selectable: Bool
    /// The size, worked out arithmetically. `nil` ⇒ let the text system measure.
    let computed: NSSize?

    func makeNSView(context: Context) -> UnwrappedTextView {
        let tv = UnwrappedTextView()
        tv.computed = computed
        tv.isEditable = false
        tv.isSelectable = selectable
        tv.drawsBackground = false
        tv.textContainerInset = .zero
        tv.textContainer?.lineFragmentPadding = 0
        tv.textContainer?.widthTracksTextView = false
        tv.textContainer?.size = NSSize(width: CGFloat.greatestFiniteMagnitude,
                                        height: CGFloat.greatestFiniteMagnitude)
        tv.isHorizontallyResizable = true
        tv.isVerticallyResizable = true
        tv.textStorage?.setAttributedString(attributed)
        return tv
    }

    func updateNSView(_ nsView: UnwrappedTextView, context: Context) {
        nsView.isSelectable = selectable
        nsView.computed = computed
        // Streaming calls this many times a second. An unconditional replace
        // would re-lay out an unchanged block and drop an active selection on
        // every frame.
        guard let storage = nsView.textStorage, storage.isEqual(to: attributed) == false else { return }

        // A streamed block is its own previous value plus a tail, so replacing
        // the whole storage makes the text system re-lay out every line that
        // did not change — 20 times a second, over a block that keeps growing.
        // Rewrite only the part that actually differs. `changedSuffix` compares
        // attributes as well as characters, so the retroactive re-colouring the
        // lexer does when a string or comment finally closes is included in the
        // range and nothing renders stale.
        if let from = CodeBlockText.changedSuffix(from: storage, to: attributed) {
            storage.replaceCharacters(
                in: NSRange(location: from, length: storage.length - from),
                with: attributed.attributedSubstring(from: NSRange(location: from, length: attributed.length - from)))
        } else {
            storage.setAttributedString(attributed)
        }
        nsView.invalidateIntrinsicContentSize()
    }
}

/// Reports its laid-out size in BOTH axes so SwiftUI can size the column and the
/// horizontal `ScrollView` has something wider than itself to scroll.
///
/// Prefers the arithmetic answer, and CACHES the measured one: auto-layout asks
/// several times per pass and measuring costs a full `ensureLayout` of the
/// block, so an uncached getter lays a 300-line block out repeatedly per frame.
private final class UnwrappedTextView: NSTextView {
    /// Set by the representable when the size is exactly computable.
    var computed: NSSize?
    private var cached: NSSize?

    override var intrinsicContentSize: NSSize {
        if let computed { return computed }
        if let cached { return cached }
        guard let lm = layoutManager, let tc = textContainer else { return super.intrinsicContentSize }
        lm.ensureLayout(for: tc)
        let used = lm.usedRect(for: tc)
        let size = NSSize(width: ceil(used.width), height: ceil(used.height))
        cached = size
        return size
    }

    override func invalidateIntrinsicContentSize() {
        cached = nil
        super.invalidateIntrinsicContentSize()
    }

    override func didChangeText() {
        super.didChangeText()
        invalidateIntrinsicContentSize()
    }
}
