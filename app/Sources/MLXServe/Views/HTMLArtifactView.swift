import AppKit
import SwiftUI
import WebKit

/// A closed ```html / ```svg block from the model, rendered as a live page.
///
/// This is what lets a reply answer with a chart, a diagram or a small
/// interactive widget — a slider that redraws a curve, a stepper that fills a
/// table — instead of describing one.
///
/// The shape it grew out of wore `CodeBlockView`'s full chrome: a grey header
/// strip with Preview/Code chips, a hairline, and the model's page in the well
/// underneath. That reads as a code block showing its result, and it looked it:
/// a page that paints itself dark became a dark rectangle inside a light card
/// with a grey strip on top — three surfaces where the reader sees one object.
///
/// So the preview is now edge to edge and the chrome floats: nothing is drawn
/// over the page until the pointer is inside it, and then it is one capsule of
/// icons in the corner. What the page paints, the CARD paints — the surface it
/// reports comes back through `HTMLArtifactRuntime` and becomes the block's own
/// fill, so the artifact reads as one intentional surface in the model's
/// palette rather than as something pasted into the transcript.
///
/// Source is still one click away, and switching to it puts the ordinary
/// `CodeBlockHeader` + `CodeBlockBody` back — because in that half it IS a code
/// block, and it should be indistinguishable from every other one.
///
/// What runs, and what it runs inside: `HTMLArtifact`, `HTMLArtifactRuntime`.
/// When it runs (never before the fence closes): `MarkdownSegmenter`. How it is
/// contained: below.
struct HTMLArtifactView: View {
    /// The fence label verbatim, for the header and the source lexer.
    let language: String
    let code: String

    /// Settings ▸ Chat, reaching the block through the ENVIRONMENT rather than
    /// `@EnvironmentObject var appState`.
    ///
    /// `MarkdownText` renders inside `ModelDetailSheet` as well as the
    /// transcript, and a sheet does NOT inherit the environment of the view it
    /// hangs on — reading an `@EnvironmentObject` here would trap at first
    /// render on a surface that never injected one (the live crash
    /// `SheetEnvironmentAuditTests` was written for). An environment KEY has a
    /// default, so a surface that says nothing gets previews and no surface can
    /// crash for staying quiet.
    @Environment(\.htmlPreviewsEnabled) private var previewsEnabled
    /// What the app is wearing, handed to the page as `--mlx-*` custom
    /// properties so a widget can match the chat it is sitting in.
    @Environment(\.colorScheme) private var colorScheme

    /// The half the reader CHOSE, if they chose one. `nil` means "whatever the
    /// setting says" — so flipping the setting moves every block nobody has
    /// touched, and leaves alone the ones somebody did.
    @State private var chosenMode: HTMLArtifact.ViewMode?
    @State private var expanded = false
    /// Everything the page has told us about itself. Nil until it lays out.
    @State private var report: HTMLArtifactRuntime.Report?
    @State private var hovering = false
    @State private var copied = false

    private var mode: HTMLArtifact.ViewMode {
        chosenMode ?? HTMLArtifact.defaultMode(previewsEnabled: previewsEnabled)
    }

    private var showsSource: Bool { mode == .source }
    private var measured: CGFloat? { report?.height }
    private var surface: HTMLArtifactRuntime.Surface { report?.surface ?? .unpainted }

    /// The colour the card wears. `nil` ⇒ the transcript's own.
    private var fill: Color? {
        guard !showsSource, let rgb = surface.fill else { return nil }
        return Color(red: rgb.red, green: rgb.green, blue: rgb.blue)
    }

    /// Which way the floating controls read. A page paints its own surface, so
    /// the app's appearance is only the answer when the page has no opinion.
    private var chromeScheme: ColorScheme {
        switch surface.chrome {
        case .dark: return .dark
        case .light: return .light
        case .app: return colorScheme
        }
    }

    private var theme: HTMLArtifactRuntime.Theme { .current(colorScheme) }

    private var diagnostic: String? {
        guard let report, !showsSource else { return nil }
        return HTMLArtifactRuntime.diagnostic(blockedRemoteLoads: report.blockedRemoteLoads,
                                              scriptError: report.scriptError)
    }

    private var frameHeight: CGFloat {
        HTMLArtifact.frameHeight(measured: measured, expanded: expanded)
    }

    /// Whether the page is taller than the frame showing it. Also what the page
    /// is told, so it can stop consuming scroll it cannot use.
    private var clipped: Bool {
        !expanded && HTMLArtifact.canExpand(measured: measured)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if showsSource {
                sourceHeader
                Divider().opacity(0.5)
                CodeBlockBody(language: language, code: code)
            } else {
                preview
            }
            if let diagnostic { diagnosticStrip(diagnostic) }
        }
        .modifier(CodeBlockChrome(fill: fill))
        .contextMenu {
            Button(showsSource ? "Show Preview" : "Show Source") {
                chosenMode = showsSource ? .preview : .source
            }
            if !showsSource, HTMLArtifact.canExpand(measured: measured) {
                Button(expanded ? "Collapse" : "Expand") { expanded.toggle() }
            }
            Button("Copy Source", action: copySource)
        }
    }

    // MARK: - Preview

    private var preview: some View {
        HTMLArtifactWebView(source: code,
                            theme: theme,
                            collapsed: clipped,
                            report: $report)
            .frame(height: frameHeight)
            // A widget that reveals a row when a slider moves should grow, not
            // jump: the page reports its new height a frame later, and an
            // un-animated frame change snaps the whole transcript under it.
            .animation(.easeOut(duration: 0.16), value: frameHeight)
            // The page keeps drawing to the card's edges; the corners are the
            // card's, so nothing square pokes out of a rounded rectangle.
            .overlay(alignment: .bottom) { if clipped { moreBelow } }
            .overlay(alignment: .topTrailing) { floatingControls }
            .onHover { hovering = $0 }
    }

    /// The one thing that has to be visible without hovering: that there is
    /// more page than the block is showing. A hard cut at 520pt reads as a
    /// broken render.
    private var moreBelow: some View {
        LinearGradient(colors: [(fill ?? CodeTheme.background).opacity(0), fill ?? CodeTheme.background],
                       startPoint: .top, endPoint: .bottom)
            .frame(height: 44)
            .allowsHitTesting(false)
    }

    /// Chrome that is not there until you reach for it.
    ///
    /// Icons rather than words, on a material capsule: over a page whose colours
    /// we do not choose, a translucent capsule is the one background that reads
    /// on any of them, and `chromeScheme` points it at what the page actually
    /// painted rather than at what the app is wearing.
    private var floatingControls: some View {
        HStack(spacing: 1) {
            if HTMLArtifact.canExpand(measured: measured) {
                controlButton(expanded ? "arrow.down.right.and.arrow.up.left"
                                       : "arrow.up.left.and.arrow.down.right",
                              help: expanded ? "Collapse" : "Expand") { expanded.toggle() }
            }
            controlButton("chevron.left.forwardslash.chevron.right", help: "Show source") {
                chosenMode = .source
            }
            controlButton(copied ? "checkmark" : "doc.on.doc",
                          help: "Copy this block's source", tint: copied ? .green : nil,
                          action: copySource)
        }
        .padding(3)
        .background(.regularMaterial, in: Capsule())
        .overlay(Capsule().strokeBorder(Color.primary.opacity(0.10)))
        .environment(\.colorScheme, chromeScheme)
        .padding(7)
        .opacity(hovering ? 1 : 0)
        .animation(.easeOut(duration: 0.12), value: hovering)
    }

    private func controlButton(_ symbol: String, help: String, tint: Color? = nil,
                               action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(tint ?? Color.secondary)
                .frame(width: 22, height: 18)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(help)
    }

    /// What the page did not get.
    ///
    /// Artifacts run offline by design, and a model reaching for a charting
    /// library on a CDN produces an empty box. Saying so turns a bug report into
    /// a limitation the reader can act on — by asking for a self-contained
    /// version, which local models do write when told.
    private func diagnosticStrip(_ text: String) -> some View {
        HStack(spacing: 5) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 9, weight: .medium))
            Text(text)
                .font(.system(size: 10))
                .lineLimit(2)
            Spacer(minLength: 0)
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(.thinMaterial)
        .environment(\.colorScheme, chromeScheme)
    }

    // MARK: - Source

    private var sourceHeader: some View {
        CodeBlockHeader(label: CodeBlockLabel.text(for: language)) {
            HStack(spacing: 2) {
                chip("Preview", active: false) { chosenMode = .preview }
                chip("Code", active: true) { chosenMode = .source }
            }
            CodeBlockCopyButton(code: code, help: "Copy this block's source")
        }
    }

    /// Header controls are hand-built rather than a segmented `Picker`: the
    /// strip is a 10pt row, and an AppKit control sizes itself to its own
    /// metrics and makes the header of every HTML block taller than the header
    /// of every code block beside it.
    private func chip(_ title: String, active: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(active ? Color.primary : Color.secondary)
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
                .background(
                    RoundedRectangle(cornerRadius: 4)
                        .fill(active ? CodeTheme.background : Color.clear)
                )
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func copySource() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(code, forType: .string)
        copied = true
        Task {
            try? await Task.sleep(nanoseconds: 1_400_000_000)
            copied = false
        }
    }
}

// MARK: - The web view

/// The `WKWebView` an artifact runs in, and every lock on it.
///
/// The threat model is plain: this is markup a local model wrote, from a
/// conversation that may itself contain text the user pasted from somewhere
/// else. It gets to draw, and nothing else.
///
/// - **No network.** Every scheme with an authority — http, https, ws, ftp,
///   file — is blocked by a content rule list (`ArtifactWebEnvironment`). A nil
///   base URL alone is not enough: it stops relative URLs and cross-origin
///   `fetch`, but a `<script src>` or a tracking pixel with an absolute URL is
///   a subresource load that no navigation delegate is ever asked about. If the
///   rule list cannot be compiled the preview refuses rather than loading
///   (`HTMLArtifact.payload`). Measured to still allow `data:` and `blob:`
///   URLs, Web Workers and `srcdoc` frames, which is why the filter is by
///   scheme rather than `.*`.
/// - **No navigation.** Everything except the initial about:blank document is
///   cancelled; a clicked link opens in the user's own browser instead.
/// - **No windows, panels or pickers.** `window.open`, `alert`, `confirm`,
///   `prompt` and `<input type=file>` all complete unshown — a page inside a
///   transcript does not get to put a modal in front of the app.
/// - **No persistence.** A non-persistent data store, shared between artifacts
///   so a long transcript shares content processes instead of spawning one per
///   block; nothing an artifact writes outlives the app.
///
/// The view itself is TRANSPARENT (`underPageBackgroundColor = .clear`, the
/// public spelling — no KVC into `drawsBackground`, which the App Store build
/// shares). Without it WebKit paints an opaque white page behind everything,
/// so a fragment with no background of its own sat on a white slab inside a
/// dark card: the exact complaint this rewrite answers, and not something any
/// amount of CSS on our side could have fixed.
private struct HTMLArtifactWebView: NSViewRepresentable {
    let source: String
    let theme: HTMLArtifactRuntime.Theme
    /// True while the page is taller than the frame showing it.
    let collapsed: Bool
    @Binding var report: HTMLArtifactRuntime.Report?

    func makeCoordinator() -> Coordinator { Coordinator(report: $report) }

    func makeNSView(context: Context) -> WKWebView {
        let web = WKWebView(frame: .zero,
                            configuration: context.coordinator.makeConfiguration(theme: theme))
        web.navigationDelegate = context.coordinator
        web.uiDelegate = context.coordinator
        web.allowsMagnification = false
        web.allowsBackForwardNavigationGestures = false
        web.allowsLinkPreview = false
        web.underPageBackgroundColor = .clear
        context.coordinator.load(source, into: web)
        return web
    }

    func updateNSView(_ web: WKWebView, context: Context) {
        // Streaming calls this many times a second while the rest of the reply
        // arrives. Every apply below compares first and no-ops, so an artifact
        // is not re-run — and a re-run would restart its animations and lose
        // whatever the reader had already interacted with.
        context.coordinator.report = $report
        context.coordinator.load(source, into: web)
        context.coordinator.apply(theme: theme, to: web)
        context.coordinator.apply(collapsed: collapsed, to: web)
    }

    static func dismantleNSView(_ web: WKWebView, coordinator: Coordinator) {
        coordinator.tearDown(web)
    }

    final class Coordinator: NSObject, WKNavigationDelegate, WKUIDelegate, WKScriptMessageHandler {
        var report: Binding<HTMLArtifactRuntime.Report?>
        /// The source currently loaded, so an unchanged update is free.
        private var loaded: String?
        private var appliedTheme: HTMLArtifactRuntime.Theme?
        private var appliedCollapsed: Bool?
        /// Set once the view is dismantled. The blocker compiles asynchronously,
        /// so a block scrolled away (or a reply deleted) while that is in flight
        /// would otherwise have its callback start the page up again in a web
        /// view already torn down.
        private var dismantled = false

        init(report: Binding<HTMLArtifactRuntime.Report?>) {
            self.report = report
        }

        func makeConfiguration(theme: HTMLArtifactRuntime.Theme) -> WKWebViewConfiguration {
            let config = WKWebViewConfiguration()
            config.websiteDataStore = ArtifactWebEnvironment.shared.dataStore
            config.defaultWebpagePreferences.allowsContentJavaScript = true
            config.preferences.javaScriptCanOpenWindowsAutomatically = false
            // A model that writes `<video autoplay>` must not make noise in a
            // transcript somebody is reading.
            config.mediaTypesRequiringUserActionForPlayback = .all

            let content = WKUserContentController()
            // Document START: the stylesheet has to land ahead of the page's own
            // CSS to be a floor rather than an override, and the error listeners
            // have to be installed before the page's inline scripts run to see
            // one of them throw.
            content.addUserScript(WKUserScript(source: HTMLArtifactRuntime.stageScript(theme: theme),
                                               injectionTime: .atDocumentStart,
                                               forMainFrameOnly: true))
            content.addUserScript(WKUserScript(source: HTMLArtifactRuntime.probeScript,
                                               injectionTime: .atDocumentEnd,
                                               forMainFrameOnly: true))
            content.add(WeakScriptMessageHandler(self), name: HTMLArtifactRuntime.messageHandler)
            config.userContentController = content
            appliedTheme = theme
            return config
        }

        func load(_ source: String, into web: WKWebView) {
            guard loaded != source else { return }
            loaded = source
            // New content is a new everything. Setting it here would be a state
            // write inside SwiftUI's own update, so it lands on the next turn.
            DispatchQueue.main.async { [weak self] in self?.report.wrappedValue = nil }
            appliedCollapsed = nil

            ArtifactWebEnvironment.shared.withNetworkBlocker { [weak self, weak web] blocker in
                guard let self, !self.dismantled, let web else { return }
                let content = web.configuration.userContentController
                content.removeAllContentRuleLists()
                if let blocker { content.add(blocker) }
                // The rule list has to be installed BEFORE the load it applies
                // to, which is the whole reason this is a callback.
                web.loadHTMLString(HTMLArtifact.payload(for: source, networkBlocked: blocker != nil),
                                   baseURL: nil)
            }
        }

        /// The app changed appearance under a page that is already running.
        ///
        /// Reloading would restart it — animations from zero, a slider back at
        /// its default — for a colour change, so the stage's stylesheet is
        /// REPLACED in place. Same builder as the injected copy, so the live
        /// page and a fresh one cannot end up with different palettes.
        func apply(theme: HTMLArtifactRuntime.Theme, to web: WKWebView) {
            guard appliedTheme != theme else { return }
            appliedTheme = theme
            web.evaluateJavaScript(HTMLArtifactRuntime.styleInstallScript(theme: theme))
        }

        /// A collapsed block hands its scroll back to the transcript.
        ///
        /// A page taller than its frame keeps its own scroller, and a trackpad
        /// gesture over it moves the artifact instead of the conversation —
        /// which is maddening in a long reply, since the artifact is exactly
        /// what the pointer is over while you read past it. With the document's
        /// overflow hidden the wheel event goes unhandled and continues up the
        /// responder chain; Expand gives the page its scroller back.
        func apply(collapsed: Bool, to web: WKWebView) {
            guard appliedCollapsed != collapsed else { return }
            appliedCollapsed = collapsed
            web.evaluateJavaScript(
                "window.__mlxArtifact && window.__mlxArtifact.setCollapsed(\(collapsed));")
        }

        func tearDown(_ web: WKWebView) {
            dismantled = true
            web.stopLoading()
            web.navigationDelegate = nil
            web.uiDelegate = nil
            let content = web.configuration.userContentController
            content.removeAllUserScripts()
            content.removeAllContentRuleLists()
            content.removeScriptMessageHandler(forName: HTMLArtifactRuntime.messageHandler)
            // An artifact scrolled out of the transcript can still be running a
            // `requestAnimationFrame` loop; replacing the document stops it.
            web.loadHTMLString("", baseURL: nil)
        }

        // MARK: What the page reports

        func userContentController(_ controller: WKUserContentController,
                                   didReceive message: WKScriptMessage) {
            guard message.name == HTMLArtifactRuntime.messageHandler,
                  var incoming = HTMLArtifactRuntime.report(from: message.body) else { return }
            guard let current = report.wrappedValue else { return report.wrappedValue = incoming }

            // Hysteresis on the HEIGHT alone. A page whose layout settles a
            // fraction of a point at a time would otherwise re-lay out the whole
            // transcript per frame — but a colour or a diagnostic arriving with
            // an unchanged height still has to land, so the rest of the report
            // is compared for equality rather than swallowed with it.
            let settled: Bool
            switch (current.height, incoming.height) {
            case let (old?, new?): settled = abs(old - new) < 1
            case (nil, nil): settled = true
            default: settled = false
            }
            if settled { incoming.height = current.height }
            guard incoming != current else { return }
            report.wrappedValue = incoming
        }

        // MARK: Containment

        func webView(_ webView: WKWebView,
                     decidePolicyFor navigationAction: WKNavigationAction,
                     decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            let url = navigationAction.request.url
            // The only load this view makes is its own document, from a string
            // with no base URL — which arrives as `.other` at about:blank.
            if navigationAction.navigationType == .other,
               url == nil || url?.absoluteString == "about:blank" {
                return decisionHandler(.allow)
            }
            // A link in an artifact opens in the user's browser, where they can
            // see where it goes — never in a frame inside the transcript.
            if navigationAction.navigationType == .linkActivated,
               let url, url.scheme == "http" || url.scheme == "https" {
                NSWorkspace.shared.open(url)
            }
            decisionHandler(.cancel)
        }

        func webView(_ webView: WKWebView,
                     createWebViewWith configuration: WKWebViewConfiguration,
                     for navigationAction: WKNavigationAction,
                     windowFeatures: WKWindowFeatures) -> WKWebView? {
            nil
        }

        func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String,
                     initiatedByFrame frame: WKFrameInfo,
                     completionHandler: @escaping () -> Void) {
            completionHandler()
        }

        func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String,
                     initiatedByFrame frame: WKFrameInfo,
                     completionHandler: @escaping (Bool) -> Void) {
            completionHandler(false)
        }

        func webView(_ webView: WKWebView, runJavaScriptTextInputPanelWithPrompt prompt: String,
                     defaultText: String?, initiatedByFrame frame: WKFrameInfo,
                     completionHandler: @escaping (String?) -> Void) {
            completionHandler(nil)
        }

        func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters,
                     initiatedByFrame frame: WKFrameInfo,
                     completionHandler: @escaping ([URL]?) -> Void) {
            completionHandler(nil)
        }
    }
}

/// `WKUserContentController` retains a message handler strongly, and the web
/// view owns the controller — so registering the coordinator directly is a
/// cycle that keeps a content process alive for every artifact the reader ever
/// scrolled past.
private final class WeakScriptMessageHandler: NSObject, WKScriptMessageHandler {
    private weak var target: WKScriptMessageHandler?

    init(_ target: WKScriptMessageHandler) {
        self.target = target
    }

    func userContentController(_ controller: WKUserContentController,
                               didReceive message: WKScriptMessage) {
        target?.userContentController(controller, didReceive: message)
    }
}

/// Process-wide pieces every artifact web view shares. Main thread only.
final class ArtifactWebEnvironment {
    static let shared = ArtifactWebEnvironment()

    /// Non-persistent, and SHARED: web views on one data store share a content
    /// process rather than each spawning their own, and a long transcript can
    /// hold a lot of them. Nothing an artifact writes outlives the app.
    let dataStore = WKWebsiteDataStore.nonPersistent()

    /// Blocks every network scheme, for every artifact.
    ///
    /// Two things about this list are MEASURED, not assumed, and both were
    /// wrong on the first attempt:
    ///
    /// - **`url-filter` is not full regex.** WebKit's content-extension engine
    ///   has no disjunction: `^(https?|wss?|ftp|file)://` fails to compile with
    ///   "Disjunctions are not supported yet". A rule list that fails to
    ///   compile is SILENT — `withNetworkBlocker` hands back nil and every
    ///   artifact in the app renders the refusal page instead of the model's
    ///   work. `HTMLArtifactTests.testTheNetworkBlockerCompiles` is the guard.
    /// - **`.*` over-blocks.** It compiles, and it blocks `blob:` URLs and Web
    ///   Workers along with the network — so a chart that exports a canvas, or
    ///   any worker-backed library, breaks. Filtering by scheme leaves `data:`
    ///   (how a model embeds an image), `blob:`, workers and `srcdoc` frames
    ///   working while still blocking remote subresources and `fetch`.
    ///
    /// The last rule subsumes the four before it. They stay because they name
    /// the schemes that actually matter, and a generic pattern is a single
    /// point of failure for the one property this whole feature rests on.
    static let blockAllNetwork = """
    [{"trigger":{"url-filter":"^https?://"},"action":{"type":"block"}},
     {"trigger":{"url-filter":"^wss?://"},"action":{"type":"block"}},
     {"trigger":{"url-filter":"^ftp://"},"action":{"type":"block"}},
     {"trigger":{"url-filter":"^file://"},"action":{"type":"block"}},
     {"trigger":{"url-filter":"^[a-z][a-z0-9+.-]*://"},"action":{"type":"block"}}]
    """

    private var blocker: WKContentRuleList?
    private var waiting: [(WKContentRuleList?) -> Void]?

    /// Hands back the compiled blocker, compiling it on first use.
    ///
    /// Compilation is asynchronous and a load started before the rule list is
    /// installed is not covered by it, so every artifact goes through here and
    /// loads from the callback. `nil` means the preview must refuse — see
    /// `HTMLArtifact.payload`.
    func withNetworkBlocker(_ body: @escaping (WKContentRuleList?) -> Void) {
        if let blocker { return body(blocker) }
        if waiting != nil { return waiting?.append(body) ?? () }
        waiting = [body]

        guard let store = WKContentRuleListStore.default() else { return finish(nil) }
        store.compileContentRuleList(forIdentifier: "mlx-chat-artifact-offline",
                                     encodedContentRuleList: Self.blockAllNetwork) { [weak self] list, _ in
            DispatchQueue.main.async { self?.finish(list) }
        }
    }

    /// A failed compile leaves `blocker` nil, so the next artifact tries again
    /// rather than inheriting one transient failure for the life of the app.
    private func finish(_ list: WKContentRuleList?) {
        blocker = list
        let pending = waiting ?? []
        waiting = nil
        pending.forEach { $0(list) }
    }
}


/// Whether a `.html` block opens on its preview — `ServerOptions
/// .htmlPreviewsByDefault`, handed to the transcript by `ChatDetailView`.
///
/// An environment key rather than an `@EnvironmentObject` read, and the default
/// is the shipped behaviour: `MarkdownText` also renders in `ModelDetailSheet`,
/// and a sheet presents in its own hosting context, so a view that DEMANDED an
/// object there would trap at first render rather than fall back.
private struct HTMLPreviewsEnabledKey: EnvironmentKey {
    static let defaultValue: Bool = true
}

extension EnvironmentValues {
    var htmlPreviewsEnabled: Bool {
        get { self[HTMLPreviewsEnabledKey.self] }
        set { self[HTMLPreviewsEnabledKey.self] = newValue }
    }
}
