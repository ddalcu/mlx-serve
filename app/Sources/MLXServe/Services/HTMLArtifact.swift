import CoreGraphics
import Foundation

/// The pure half of the transcript's inline HTML renderer.
///
/// A fenced ```html / ```svg block from the model is mounted as a live
/// document, so a reply can answer with a chart, a diagram, or a small
/// interactive widget instead of describing one. `HTMLArtifactView` owns the
/// `WKWebView`; everything a test could check lives here.
///
/// `Package.swift` explains that the LaTeX renderer is a native Swift parser
/// specifically to avoid "a WebView/JavaScript renderer and the security,
/// selection, and streaming seams that would come with one". Those three seams
/// are real, and this type is where two of them are answered:
///
/// - **Streaming**: nothing renders until the model closes the fence.
///   `MarkdownSegmenter` only emits `.html` for a CLOSED fence, so a document
///   whose `<script>` is still arriving a token at a time is a code block, not
///   a page being reloaded twenty times a second.
/// - **Security**: the page is loaded with NO base URL, from a non-persistent
///   data store, with every network scheme blocked by a content rule list —
///   see `HTMLArtifactView`. This type's job is to never widen that: the
///   scaffold it wraps a fragment in references nothing outside the document,
///   pinned by `HTMLArtifactTests.testScaffoldNeverReferencesTheNetwork`.
///
/// The third seam, selection, is the one genuinely given up: text inside a
/// preview selects within the page, not as part of a drag across the whole
/// reply. That is why the block keeps a Code toggle — the source is always one
/// click away, in the same `NSTextView` machinery every other code block uses.
enum HTMLArtifact {

    // MARK: - What runs

    /// Fence labels that mount a document.
    ///
    /// Deliberately narrower than `SyntaxLanguage.markup`, which also covers
    /// `xml`, `vue` and `svelte`: those mean data or a component source the
    /// model wants read, and a browser renders all three as near-blank.
    static let renderableFences: Set<String> = ["html", "htm", "svg"]

    /// Whether this fenced block renders as a live document rather than code.
    ///
    /// The ONE decision — `MarkdownSegmenter` calls it rather than repeating
    /// the rule, pinned by `testSegmenterAndPredicateAgree`.
    static func rendersLive(language: String, code: String) -> Bool {
        let key = language.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return renderableFences.contains(key) && containsMarkup(code)
    }

    /// A `<` that opens a tag, a closing tag, a comment, or a doctype.
    ///
    /// The point is to tell markup from a fence a model labelled `html` while
    /// writing about HTML in prose — `a < b` is not a document, and a web view
    /// showing nothing is worse than the sentence it replaced.
    static func containsMarkup(_ code: String) -> Bool {
        var afterAngle = false
        for scalar in code.unicodeScalars {
            if afterAngle,
               CharacterSet.letters.contains(scalar) || scalar == "/" || scalar == "!" {
                return true
            }
            afterAngle = scalar == "<"
        }
        return false
    }

    // MARK: - The bytes the web view loads

    /// Whether the model wrote a whole page rather than a fragment.
    static func isCompleteDocument(_ code: String) -> Bool {
        let head = code.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return head.hasPrefix("<!doctype") || head.contains("<html") || head.contains("<body")
    }

    /// What the web view is actually handed.
    ///
    /// `networkBlocked` is the load-bearing argument: with no content rule
    /// list installed, a `<script src>` or a tracking pixel in model-written
    /// markup would reach the network from inside the user's transcript. The
    /// answer to a blocker that failed to compile is to render the refusal,
    /// never the document — so the unsafe path cannot be reached by forgetting
    /// a branch at the call site.
    static func payload(for code: String, networkBlocked: Bool) -> String {
        networkBlocked ? document(for: code) : previewUnavailableDocument
    }

    /// A complete page is loaded verbatim; a fragment gets a minimal scaffold.
    ///
    /// Verbatim matters both ways round: wrapping a page puts a second `<html>`
    /// around it, and restyling one fights the CSS the model shipped with it.
    static func document(for code: String) -> String {
        isCompleteDocument(code) ? code : Self.scaffoldOpen + code + Self.scaffoldClose
    }

    /// Wrapper for a fragment: a document element to live in, and nothing more.
    ///
    /// It carries NO styling. Everything a page needs to be legible in the
    /// transcript — the palette, the type, the padding, the transparent
    /// background that lets the card show through — is injected at document
    /// start by `HTMLArtifactRuntime`, which a complete document gets too. A
    /// stylesheet here would sit LATER in the cascade than that one and beat
    /// it, so a fragment and a page would be styled by different rules; worse,
    /// this one used to paint an opaque `Canvas` background, which is a white
    /// slab inside a dark card the moment the app and the page disagree.
    ///
    /// No font file, no reset stylesheet, no `<base>`: the block is loaded with
    /// a nil base URL and every remote load is blocked, so anything the
    /// scaffold pulled in would be the one request that had to be let through.
    private static let scaffoldOpen = """
    <!doctype html><html><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    </head><body>
    """

    private static let scaffoldClose = "</body></html>"

    /// Shown in place of the preview when the network blocker is unavailable.
    ///
    /// Carries none of the model's markup — the whole point is that nothing
    /// from the reply executes on this path. Its one inline rule is a colour it
    /// inherits from the stage, so the refusal reads on whatever the app is
    /// wearing without naming a colour of its own.
    static let previewUnavailableDocument = Self.scaffoldOpen + """
    <p style="opacity:.7;font-size:13px;margin:0">
    Preview unavailable — the sandbox that keeps this page offline could not be
    prepared. Switch to Code to read the source.</p>
    """ + Self.scaffoldClose

    // MARK: - Which half opens

    /// Which half of the block is showing.
    enum ViewMode: Equatable {
        case preview
        case source
    }

    /// The half a block OPENS on, from Settings ▸ Chat.
    ///
    /// A default, not a gate: the header's Preview/Code switch works the same
    /// either way, so turning previews off makes the source the first thing you
    /// see rather than making the rendered page unreachable.
    static func defaultMode(previewsEnabled: Bool) -> ViewMode {
        previewsEnabled ? .preview : .source
    }

    // MARK: - Height

    /// Height before the page has reported its own — a plausible one, because a
    /// block that opens at zero and jumps to 400 shoves the transcript under
    /// whatever the reader was looking at.
    static let placeholderHeight: CGFloat = 260
    /// A one-line fragment measuring 12pt would draw as a sliver in a border.
    static let minHeight: CGFloat = 44
    static let collapsedMaxHeight: CGFloat = 520
    /// Even expanded, one block must not swallow a whole scroll of transcript;
    /// past this the page keeps its own scroller.
    static let expandedMaxHeight: CGFloat = 1_200

    /// The block's frame height, given what the page measured itself at.
    ///
    /// Total on non-finite input: a page with `height: 1e9`, or a measurement
    /// that arrives NaN, is a mistake rather than a layout instruction — and a
    /// NaN frame does not break one block, it breaks the whole chat column.
    static func frameHeight(measured: CGFloat?, expanded: Bool) -> CGFloat {
        guard let measured, !measured.isNaN else { return placeholderHeight }
        return min(max(measured, minHeight), expanded ? expandedMaxHeight : collapsedMaxHeight)
    }

    /// Whether the header offers Expand — only when it would actually reveal
    /// something.
    static func canExpand(measured: CGFloat?) -> Bool {
        guard let measured, measured.isFinite else { return false }
        return measured > collapsedMaxHeight
    }
}
