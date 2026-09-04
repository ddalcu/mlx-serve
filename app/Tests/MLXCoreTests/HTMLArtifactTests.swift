import WebKit
import XCTest
@testable import MLXCore

/// The pure half of the inline HTML renderer: which fenced blocks become a live
/// document, what bytes that document is, and how tall the block stands.
///
/// All of it is testable on purpose. The view around it owns a `WKWebView`,
/// which no unit test can assert against — so every decision that could go
/// wrong quietly (running a half-streamed script, escaping the model's markup
/// into visible source, a scaffold that reaches the network) is made here
/// instead, where a test can see it.
final class HTMLArtifactTests: XCTestCase {

    // MARK: - What renders

    func testHtmlAndSvgFencesRender() {
        XCTAssertTrue(HTMLArtifact.rendersLive(language: "html", code: "<b>x</b>"))
        XCTAssertTrue(HTMLArtifact.rendersLive(language: "htm", code: "<b>x</b>"))
        XCTAssertTrue(HTMLArtifact.rendersLive(language: "svg", code: "<svg></svg>"))
    }

    func testFenceLabelIsMatchedLooselyLikeTheSyntaxHighlighter() {
        XCTAssertTrue(HTMLArtifact.rendersLive(language: "HTML", code: "<b>x</b>"))
        XCTAssertTrue(HTMLArtifact.rendersLive(language: "  Html  ", code: "<b>x</b>"))
    }

    func testOtherLanguagesNeverRender() {
        // `xml` is deliberately out: the highlighter groups it with markup, but
        // a model emitting XML means data, not a page to run.
        for language in ["", "swift", "js", "javascript", "xml", "vue", "svelte", "markdown", "text"] {
            XCTAssertFalse(HTMLArtifact.rendersLive(language: language, code: "<b>x</b>"),
                           "fence \(language.debugDescription) must not run as a document")
        }
    }

    func testSourceWithoutMarkupDoesNotRender() {
        // An empty or prose-only block renders as an empty web view — a hole in
        // the transcript where the model's answer should be.
        XCTAssertFalse(HTMLArtifact.rendersLive(language: "html", code: ""))
        XCTAssertFalse(HTMLArtifact.rendersLive(language: "html", code: "   \n  "))
        XCTAssertFalse(HTMLArtifact.rendersLive(language: "html", code: "use a div for that"))
        XCTAssertFalse(HTMLArtifact.rendersLive(language: "html", code: "a < b and c > d"))
    }

    func testCommentsAndDoctypesCountAsMarkup() {
        XCTAssertTrue(HTMLArtifact.rendersLive(language: "html", code: "<!doctype html><html></html>"))
        XCTAssertTrue(HTMLArtifact.rendersLive(language: "html", code: "<!-- note --><p>x</p>"))
        XCTAssertTrue(HTMLArtifact.rendersLive(language: "html", code: "<div/>"))
    }

    func testSegmenterAndPredicateAgree() {
        // Class guard: ONE decision about what runs. The segmenter is what the
        // transcript actually consults, so a second copy of the rule in it is
        // how the two would drift — and the drift would be a web view mounted
        // over source the predicate had already declined.
        let languages = ["html", "HTML", "svg", "htm", "xml", "js", "swift", ""]
        let bodies = ["<b>x</b>", "", "plain sentence", "<svg><rect/></svg>", "<!-- c -->"]
        for language in languages {
            for body in bodies {
                let out = MarkdownSegmenter.segments("```\(language)\n\(body)\n```")
                let isHTML: Bool
                if case .html? = out.first { isHTML = true } else { isHTML = false }
                XCTAssertEqual(isHTML, HTMLArtifact.rendersLive(language: language, code: body),
                               "fence \(language.debugDescription) body \(body.debugDescription)")
            }
        }
    }

    // MARK: - The bytes handed to the web view

    func testCompleteDocumentIsLoadedVerbatim() {
        // The model wrote a page. Wrapping it would put a second <html> around
        // it, and re-styling it would fight the CSS it shipped.
        let doc = "<!DOCTYPE html>\n<html><head><style>body{background:#111}</style></head>"
            + "<body><canvas id=c></canvas><script>console.log(1)</script></body></html>"
        XCTAssertTrue(HTMLArtifact.isCompleteDocument(doc))
        XCTAssertEqual(HTMLArtifact.document(for: doc), doc)
    }

    func testDocumentDetectionAcceptsEitherMarker() {
        XCTAssertTrue(HTMLArtifact.isCompleteDocument("<html><body>x</body></html>"))
        XCTAssertTrue(HTMLArtifact.isCompleteDocument("  <!doctype html>\n<div>x</div>"))
        XCTAssertTrue(HTMLArtifact.isCompleteDocument("<HTML LANG=\"en\">x</HTML>"))
        XCTAssertFalse(HTMLArtifact.isCompleteDocument("<div>x</div>"))
        XCTAssertFalse(HTMLArtifact.isCompleteDocument("<svg><rect/></svg>"))
        // The word alone is not a tag — a fragment that merely mentions html.
        XCTAssertFalse(HTMLArtifact.isCompleteDocument("<p>html is a markup language</p>"))
    }

    func testFragmentIsScaffoldedAndCarriedVerbatimInsideIt() {
        let fragment = "<div class=\"chart\"><script>draw()</script></div>"
        let out = HTMLArtifact.document(for: fragment)
        XCTAssertTrue(out.contains(fragment),
                      "the model's markup must reach the page unmodified — escaping it renders source")
        XCTAssertTrue(out.lowercased().hasPrefix("<!doctype html>"))
        XCTAssertEqual(out.lowercased().components(separatedBy: "<html").count - 1, 1,
                       "exactly one document element")
    }

    func testTheScaffoldStylesNothingItself() {
        // A transcript is read in both appearances, and following one is the
        // STAGE's job (`HTMLArtifactRuntime`), which a complete document gets
        // too. A stylesheet here would sit later in the cascade than that one
        // and beat it — so a fragment and a page would be styled by different
        // rules, and the opaque `Canvas` background this used to paint is a
        // white slab inside a dark card.
        let out = HTMLArtifact.document(for: "<div>x</div>").lowercased()
        XCTAssertFalse(out.contains("<style"))
        XCTAssertFalse(out.contains("canvastext"))
        XCTAssertTrue(HTMLArtifactRuntime.stageCSS(theme: .init(foreground: "#111", background: "#fff",
                                                               accent: "#07f", dark: false))
                        .contains("color-scheme"))
    }

    func testScaffoldNeverReferencesTheNetwork() {
        // Class guard. The block is loaded with no base URL and every remote
        // load is blocked at the web view — a font or reset stylesheet pulled
        // in by OUR wrapper would be the one request that had to be allowed,
        // and the hole would be ours rather than the model's.
        for source in ["<div>x</div>", "<svg><rect/></svg>", "<!doctype html><html></html>"] {
            let out = HTMLArtifact.document(for: source).lowercased()
            let scaffold = out.replacingOccurrences(of: source.lowercased(), with: "")
            for needle in ["http://", "https://", "//fonts.", "<base", "url("] {
                XCTAssertFalse(scaffold.contains(needle),
                               "scaffold reaches out via \(needle) for \(source.debugDescription)")
            }
        }
    }

    func testSvgFragmentIsScaffoldedLikeAnyOtherFragment() {
        let svg = "<svg viewBox=\"0 0 10 10\"><circle cx=\"5\" cy=\"5\" r=\"4\"/></svg>"
        let out = HTMLArtifact.document(for: svg)
        XCTAssertTrue(out.contains(svg))
        XCTAssertTrue(out.lowercased().contains("<body"))
    }

    func testWithoutTheNetworkBlockerNothingFromTheReplyIsLoaded() {
        // The blocker is what keeps a model-written `<script src>` or tracking
        // pixel from reaching the network out of the user's transcript. If it
        // could not be prepared, the honest answer is a refusal — rendering the
        // page anyway is the one failure mode that would be silent.
        let code = "<img src=\"https://tracker.example/p.gif\"><b>x</b>"
        let refused = HTMLArtifact.payload(for: code, networkBlocked: false)
        XCTAssertFalse(refused.contains("tracker.example"))
        XCTAssertFalse(refused.contains("<b>x</b>"))
        XCTAssertEqual(refused, HTMLArtifact.previewUnavailableDocument)
        XCTAssertEqual(HTMLArtifact.payload(for: code, networkBlocked: true),
                       HTMLArtifact.document(for: code))
    }

    func testTheRefusalPageIsItselfSelfContained() {
        let out = HTMLArtifact.previewUnavailableDocument.lowercased()
        XCTAssertTrue(out.hasPrefix("<!doctype html>"))
        for needle in ["http://", "https://", "<base", "<script", "url("] {
            XCTAssertFalse(out.contains(needle), "refusal page contains \(needle)")
        }
    }

    func testTheNetworkBlockerCompiles() throws {
        // The one guard that has already caught a real defect. `url-filter` is
        // NOT full regex — WebKit's content-extension engine has no
        // disjunction, and the first version of this list
        // (`^(https?|wss?|ftp|file)://`) failed to compile with "Disjunctions
        // are not supported yet".
        //
        // Nothing about that failure is loud: compilation is asynchronous and
        // its error goes to a callback, `withNetworkBlocker` hands back nil,
        // and every artifact in the app quietly renders the refusal page
        // instead of the model's work. The type checker cannot see inside a
        // JSON string literal, so this is the only thing that can.
        //
        // Compiled into a throwaway store, not the app's, so the test writes
        // nothing a later run can inherit.
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mlx-artifact-rules-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try XCTUnwrap(WKContentRuleListStore(url: directory))

        var compiled: WKContentRuleList?
        var failure: Error?
        let finished = expectation(description: "rule list compiled")
        store.compileContentRuleList(forIdentifier: "artifact-offline",
                                     encodedContentRuleList: ArtifactWebEnvironment.blockAllNetwork) {
            compiled = $0
            failure = $1
            finished.fulfill()
        }
        wait(for: [finished], timeout: 30)
        XCTAssertNotNil(compiled,
                        "the artifact network blocker must compile, or every preview refuses: "
                        + String(describing: failure))
    }

    // MARK: - The default-view setting
    //
    // Settings ▸ Chat ▸ "Render HTML blocks as live previews". It chooses which
    // half of the block OPENS; the header's Preview/Code switch is unaffected
    // either way, so nothing is ever unreachable because of it.

    func testAFreshInstallOpensOnThePreview() {
        XCTAssertTrue(ServerOptions().htmlPreviewsByDefault)
    }

    func testTheSettingOnlyChoosesTheOpeningView() {
        XCTAssertEqual(HTMLArtifact.defaultMode(previewsEnabled: true), .preview)
        XCTAssertEqual(HTMLArtifact.defaultMode(previewsEnabled: false), .source)
    }

    func testAConfigWrittenBeforeTheSettingExistedKeepsPreviews() throws {
        // Changing a persisted DEFAULT does nothing for existing users — the
        // tolerant decoder answers for a key that is not there, and the answer
        // has to be the shipped behaviour rather than `false`, or everyone who
        // upgrades silently loses the feature.
        let old = Data(#"{"host":"127.0.0.1","port":8080}"#.utf8)
        let decoded = try JSONDecoder().decode(ServerOptions.self, from: old)
        XCTAssertTrue(decoded.htmlPreviewsByDefault)
    }

    func testTheSettingRoundTripsThroughTheStoredConfig() throws {
        for value in [true, false] {
            var options = ServerOptions()
            options.htmlPreviewsByDefault = value
            let data = try JSONEncoder().encode(options)
            let back = try JSONDecoder().decode(ServerOptions.self, from: data)
            XCTAssertEqual(back.htmlPreviewsByDefault, value)
        }
    }

    func testFlippingTheSettingNeverPromptsAServerRestart() {
        // A transcript rendering preference is app-side: it is not a launch
        // flag, so — like `toolsOnlyWhenAsked` and `voiceClonePath` — it stays
        // out of `serverLaunchEquals` and `toCLIArgs`. Otherwise toggling how a
        // code block LOOKS puts the restart banner up over a running server.
        var flipped = ServerOptions()
        flipped.htmlPreviewsByDefault = false
        XCTAssertTrue(ServerOptions().serverLaunchEquals(flipped))
        XCTAssertEqual(ServerOptions().toCLIArgs(), flipped.toCLIArgs())
    }

    // MARK: - Height

    func testHeightBeforeMeasurementIsThePlaceholder() {
        // The page reports its height once it has laid out. Until then the
        // block still needs a size, and it should be a plausible one — a block
        // that opens at zero and jumps to 400 shoves the transcript.
        XCTAssertEqual(HTMLArtifact.frameHeight(measured: nil, expanded: false),
                       HTMLArtifact.placeholderHeight)
        XCTAssertEqual(HTMLArtifact.frameHeight(measured: nil, expanded: true),
                       HTMLArtifact.placeholderHeight)
    }

    func testShortContentGetsExactlyItsOwnHeight() {
        XCTAssertEqual(HTMLArtifact.frameHeight(measured: 180, expanded: false), 180)
    }

    func testAVeryShortPageStillHasAFloor() {
        // A one-line fragment measuring 12pt would render as a sliver with a
        // border around it.
        XCTAssertEqual(HTMLArtifact.frameHeight(measured: 4, expanded: false),
                       HTMLArtifact.minHeight)
    }

    func testTallContentIsCappedUntilExpanded() {
        let tall: CGFloat = 2_000
        XCTAssertEqual(HTMLArtifact.frameHeight(measured: tall, expanded: false),
                       HTMLArtifact.collapsedMaxHeight)
        XCTAssertEqual(HTMLArtifact.frameHeight(measured: tall, expanded: true),
                       HTMLArtifact.expandedMaxHeight,
                       "even expanded, one block must not take a whole scroll of transcript")
        XCTAssertEqual(HTMLArtifact.frameHeight(measured: 700, expanded: true), 700)
    }

    func testExpandIsOfferedOnlyWhenSomethingIsHiddenByIt() {
        XCTAssertFalse(HTMLArtifact.canExpand(measured: nil))
        XCTAssertFalse(HTMLArtifact.canExpand(measured: 200))
        XCTAssertFalse(HTMLArtifact.canExpand(measured: HTMLArtifact.collapsedMaxHeight))
        XCTAssertTrue(HTMLArtifact.canExpand(measured: HTMLArtifact.collapsedMaxHeight + 1))
    }

    func testAbsurdMeasurementsCannotBlowUpTheTranscript() {
        // A page with `height: 1e9` or a NaN measurement is a model's mistake,
        // not a layout instruction — SwiftUI given a NaN frame logs and breaks
        // the whole column.
        XCTAssertEqual(HTMLArtifact.frameHeight(measured: 1e9, expanded: true),
                       HTMLArtifact.expandedMaxHeight)
        XCTAssertEqual(HTMLArtifact.frameHeight(measured: .nan, expanded: false),
                       HTMLArtifact.placeholderHeight)
        XCTAssertEqual(HTMLArtifact.frameHeight(measured: .infinity, expanded: false),
                       HTMLArtifact.collapsedMaxHeight)
        XCTAssertEqual(HTMLArtifact.frameHeight(measured: -50, expanded: false),
                       HTMLArtifact.minHeight)
    }
}
