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

/// Layout constants for a code block.
enum CodeBlockLayout {
    static let fontSize: CGFloat = 12
    static let cornerRadius: CGFloat = 10
    static let lineSpacing: CGFloat = 2.5
}

/// A fenced code block: language header with a copy button, and syntax-colored
/// code that scrolls horizontally.
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

    @State private var copied = false

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
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider().opacity(0.5)
            ScrollView(.horizontal, showsIndicators: false) {
                // ONE text view for the whole block. This was a `ForEach`
                // building a `Text` per line, so a 300-line block carried
                // hundreds of nodes in SwiftUI's attribute graph and a streaming
                // reply rebuilt every one of them per token.
                CodeNSText(attributed: CodeBlockText.code(code, language: resolved), selectable: true,
                           computed: CodeBlockText.measuredSize(of: code))
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
            }
        }
        .background(CodeTheme.background)
        .clipShape(RoundedRectangle(cornerRadius: CodeBlockLayout.cornerRadius))
        .overlay(
            RoundedRectangle(cornerRadius: CodeBlockLayout.cornerRadius)
                .stroke(CodeTheme.border, lineWidth: 1)
        )
    }

    private var header: some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)
            Spacer()
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
                        .font(.system(size: 10, weight: .medium))
                    Text(copied ? "Copied" : "Copy")
                        .font(.system(size: 10, weight: .medium))
                }
                .foregroundStyle(copied ? Color.green : Color.secondary)
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Copy this code block")
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(CodeTheme.header)
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
