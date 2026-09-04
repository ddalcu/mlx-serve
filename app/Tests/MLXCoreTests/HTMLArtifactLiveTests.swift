import WebKit
import XCTest
@testable import MLXCore

/// Stands a REAL `WKWebView` up around the stage and the probe.
///
/// The whole reason this file exists: the defects this feature keeps producing
/// live inside a JSON string literal or a JS string literal, where no type
/// checker and no unit test can see them. The content rule list's missing
/// disjunction was found this way, by hand, in a throwaway script; the
/// viewport clamp and the background hoist are the same shape of risk — CSS
/// that either applies or silently does not.
///
/// Opt-in (`MLX_SERVE_LIVE_ARTIFACT=1`) because it drives a web content process
/// and waits on real layout. Everything it checks that CAN be checked purely is
/// also checked in `HTMLArtifactRuntimeTests`; what it adds is that the strings
/// actually do what they say inside WebKit.
final class HTMLArtifactLiveTests: XCTestCase {

    private var live: Bool { ProcessInfo.processInfo.environment["MLX_SERVE_LIVE_ARTIFACT"] == "1" }

    /// Loads `html` the way `HTMLArtifactView` does — same stage, same probe,
    /// same blocker, same nil base URL — and returns the last report the page
    /// posted within `timeout`.
    @MainActor
    private func render(_ html: String, frameHeight: CGFloat = 260,
                        timeout: TimeInterval = 12) throws -> HTMLArtifactRuntime.Report {
        let blocker = try compiledBlocker()
        let theme = HTMLArtifactRuntime.Theme(foreground: "#eaeaea", background: "#1c1c1e",
                                              accent: "#0a84ff", dark: true)

        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()
        config.defaultWebpagePreferences.allowsContentJavaScript = true
        let content = WKUserContentController()
        content.addUserScript(WKUserScript(source: HTMLArtifactRuntime.stageScript(theme: theme),
                                           injectionTime: .atDocumentStart, forMainFrameOnly: true))
        content.addUserScript(WKUserScript(source: HTMLArtifactRuntime.probeScript,
                                           injectionTime: .atDocumentEnd, forMainFrameOnly: true))
        let sink = ReportSink()
        content.add(sink, name: HTMLArtifactRuntime.messageHandler)
        content.add(blocker)
        config.userContentController = content

        // The frame is what `100vh` resolves to, so the placeholder height is
        // the number a viewport-locked page would report back if the clamp did
        // not work — which is exactly the bug being tested for.
        let web = WKWebView(frame: NSRect(x: 0, y: 0, width: 600, height: frameHeight),
                            configuration: config)
        web.underPageBackgroundColor = .clear
        web.loadHTMLString(HTMLArtifact.payload(for: html, networkBlocked: true), baseURL: nil)

        let deadline = Date().addingTimeInterval(timeout)
        // Reports keep arriving as the page settles; take the last one before
        // the deadline rather than the first.
        while Date() < deadline {
            RunLoop.current.run(until: Date().addingTimeInterval(0.05))
        }
        content.removeScriptMessageHandler(forName: HTMLArtifactRuntime.messageHandler)
        web.loadHTMLString("", baseURL: nil)
        return try XCTUnwrap(sink.last, "the page never reported anything")
    }

    private func compiledBlocker() throws -> WKContentRuleList {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mlx-artifact-live-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let store = try XCTUnwrap(WKContentRuleListStore(url: directory))
        var compiled: WKContentRuleList?
        let done = expectation(description: "blocker")
        store.compileContentRuleList(forIdentifier: "artifact-offline",
                                     encodedContentRuleList: ArtifactWebEnvironment.blockAllNetwork) { list, _ in
            compiled = list
            done.fulfill()
        }
        wait(for: [done], timeout: 30)
        return try XCTUnwrap(compiled)
    }

    private final class ReportSink: NSObject, WKScriptMessageHandler {
        var last: HTMLArtifactRuntime.Report?
        func userContentController(_ controller: WKUserContentController,
                                   didReceive message: WKScriptMessage) {
            if let report = HTMLArtifactRuntime.report(from: message.body) { last = report }
        }
    }

    // MARK: - The three things the strings have to actually do

    @MainActor
    func testAViewportLockedPageCollapsesToItsOwnContent() throws {
        try XCTSkipUnless(live, "set MLX_SERVE_LIVE_ARTIFACT=1")
        // The shape a model writes when it thinks it is writing for a browser
        // window: a full-viewport hero with one card centred in it. Inside a
        // transcript `100vh` resolves to whatever height we guessed, so the
        // page reports exactly our guess back — a two-line widget standing in a
        // 400pt block of its own background colour, and a measurement that can
        // never say anything else no matter how long it settles.
        let report = try render("""
        <!doctype html><html><head><style>
          body { min-height: 100vh; display: flex; align-items: center; justify-content: center;
                 background: #0f172a; color: #e2e8f0; margin: 0; }
          .card { height: 90px; width: 300px; background: #1e293b; }
        </style></head><body><div class="card">short</div></body></html>
        """, frameHeight: 400)
        let height = try XCTUnwrap(report.height)
        XCTAssertLessThan(height, 200, "the page measured its frame (100vh), not its content")
        XCTAssertGreaterThanOrEqual(height, 90)
    }

    @MainActor
    func testAViewportLockedWRAPPERCollapsesTheSameWay() throws {
        try XCTSkipUnless(live, "set MLX_SERVE_LIVE_ARTIFACT=1")
        // `.wrap { min-height: 100vh }` locks a page exactly as hard as `body`
        // does, and a stylesheet clamp on html/body cannot reach it — which is
        // why the probe also walks the descendants.
        let report = try render("""
        <div class="wrap" style="min-height:100vh;background:#101418">
          <div style="height:70px">short</div>
        </div>
        """, frameHeight: 400)
        let height = try XCTUnwrap(report.height)
        XCTAssertLessThan(height, 220, "a viewport-locked wrapper measured the frame, not its content")
    }

    @MainActor
    func testTallContentStillReportsAllOfItself() throws {
        try XCTSkipUnless(live, "set MLX_SERVE_LIVE_ARTIFACT=1")
        // The clamp removes a FLOOR. Nothing may be made shorter than its own
        // content, or Expand reveals a page cut off at the knee.
        let report = try render("""
        <div style="height:900px;background:#0f172a">tall</div>
        """, frameHeight: 260)
        XCTAssertGreaterThan(try XCTUnwrap(report.height), 880)
    }

    @MainActor
    func testAPageThatPaintsItselfReportsTheColourTheCardShouldWear() throws {
        try XCTSkipUnless(live, "set MLX_SERVE_LIVE_ARTIFACT=1")
        let report = try render("""
        <!doctype html><html><head><style>body{background:#0f172a;color:#e2e8f0}</style></head>
        <body><p>hello</p></body></html>
        """)
        let fill = try XCTUnwrap(report.surface.fill, "the page's own background never reached Swift")
        XCTAssertEqual(fill.red, 15.0 / 255, accuracy: 0.01)
        XCTAssertEqual(fill.blue, 42.0 / 255, accuracy: 0.01)
        XCTAssertEqual(report.surface.chrome, .dark)
    }

    @MainActor
    func testAFragmentPaintsNothingSoTheCardShowsThrough() throws {
        try XCTSkipUnless(live, "set MLX_SERVE_LIVE_ARTIFACT=1")
        // The stage's transparent default is what lets the transcript's own card
        // be the artifact's background. If some later edit gives the stage an
        // opaque body, every fragment goes back to sitting on a slab.
        let report = try render("<div style=\"height:120px\">plain</div>")
        XCTAssertNil(report.surface.fill)
        XCTAssertEqual(report.surface.chrome, .app)
    }

    @MainActor
    func testABlockedCDNScriptIsCOUNTEDRatherThanLeavingABlankBox() throws {
        try XCTSkipUnless(live, "set MLX_SERVE_LIVE_ARTIFACT=1")
        let report = try render("""
        <div id="chart">chart</div>
        <script src="https://cdn.example.com/chart.min.js"></script>
        """)
        XCTAssertGreaterThanOrEqual(report.blockedRemoteLoads, 1,
                                    "a blocked remote script must be reported, not rendered as nothing")
        XCTAssertNotNil(HTMLArtifactRuntime.diagnostic(blockedRemoteLoads: report.blockedRemoteLoads,
                                                       scriptError: report.scriptError))
    }

    @MainActor
    func testAThrowingInlineScriptIsNamed() throws {
        try XCTSkipUnless(live, "set MLX_SERVE_LIVE_ARTIFACT=1")
        // The listener is installed at document START for exactly this: a
        // listener added at document end has already missed the throw.
        let report = try render("<p>x</p><script>notDefinedAnywhere();</script>")
        XCTAssertNotNil(report.scriptError)
    }

    @MainActor
    func testAnInteractiveWidgetGrowsWhenItsOwnControlChangesTheLayout() throws {
        try XCTSkipUnless(live, "set MLX_SERVE_LIVE_ARTIFACT=1")
        // The point of the whole feature: a slider that reveals content has to
        // grow the block holding it, which means the probe must still be
        // watching long after the page finished loading.
        let report = try render("""
        <div id="box" style="height:80px">a</div>
        <script>
          setTimeout(function () { document.getElementById('box').style.height = '640px'; }, 300);
        </script>
        """)
        let height = try XCTUnwrap(report.height)
        XCTAssertGreaterThan(height, 600, "a late layout change never reached the frame")
    }

    @MainActor
    func testThePagesOWNDefaultPaintIsTheCardsColourAndIsNotMistakenForTheModelsChoice() throws {
        try XCTSkipUnless(live, "set MLX_SERVE_LIVE_ARTIFACT=1")
        // WebKit paints an opaque backdrop under a transparent page, and the
        // only macOS switch for that is a private one — measured:
        // `underPageBackgroundColor = .clear` does NOT composite, and this app
        // ships to the App Store, so KVC into `drawsBackground` is out. So the
        // stage gives the page the transcript card's own colour as its DEFAULT
        // background: there is no seam to hide because both sides paint the
        // same thing.
        //
        // Which makes recognising our own paint load-bearing. If the probe
        // reported it as the page's, every artifact — including a plain
        // fragment — would come back claiming a surface, and the `.app` chrome
        // branch would be dead code.
        let report = try render("<div style=\"height:120px\">plain</div>")
        XCTAssertNil(report.surface.fill, "the stage's own default paint was read as the model's choice")
        XCTAssertEqual(report.surface.chrome, .app)
    }
}
