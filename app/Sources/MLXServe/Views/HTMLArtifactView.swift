import AppKit
import SwiftUI
import WebKit

/// A closed ```html / ```svg block from the model, rendered as a live page.
///
/// This is what lets a reply answer with a chart, a diagram or a small
/// interactive widget instead of describing one. It wears the same card, radius
/// and header strip as `CodeBlockView` (`CodeBlockChrome`, `CodeBlockHeader`),
/// because from the reader's side it IS a code block — one showing its result
/// first. Code is always one click away, and the toggle shows the very same
/// `CodeBlockBody` a plain fence would have rendered.
///
/// What runs and what it is wrapped in: `HTMLArtifact`. When it runs (never
/// before the fence closes): `MarkdownSegmenter`. How it is contained: below.
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

    /// The half the reader CHOSE, if they chose one. `nil` means "whatever the
    /// setting says" — so flipping the setting moves every block nobody has
    /// touched, and leaves alone the ones somebody did.
    @State private var chosenMode: HTMLArtifact.ViewMode?
    @State private var expanded = false
    /// What the page reported its own height as; nil until it has laid out.
    @State private var measured: CGFloat?

    private var mode: HTMLArtifact.ViewMode {
        chosenMode ?? HTMLArtifact.defaultMode(previewsEnabled: previewsEnabled)
    }

    private var showsSource: Bool { mode == .source }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider().opacity(0.5)
            if showsSource {
                CodeBlockBody(language: language, code: code)
            } else {
                HTMLArtifactWebView(source: code, measured: $measured)
                    .frame(height: HTMLArtifact.frameHeight(measured: measured, expanded: expanded))
            }
        }
        .modifier(CodeBlockChrome())
        .contextMenu {
            Button(showsSource ? "Show Preview" : "Show Source") {
                chosenMode = showsSource ? .preview : .source
            }
            Button("Copy Source") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(code, forType: .string)
            }
        }
    }

    private var header: some View {
        CodeBlockHeader(label: CodeBlockLabel.text(for: language)) {
            if !showsSource, HTMLArtifact.canExpand(measured: measured) {
                chip(expanded ? "Collapse" : "Expand", active: false) { expanded.toggle() }
            }
            HStack(spacing: 2) {
                chip("Preview", active: !showsSource) { chosenMode = .preview }
                chip("Code", active: showsSource) { chosenMode = .source }
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
/// Height is measured by the page and reported back, because a document's
/// height is not knowable from the outside — see the injected script.
private struct HTMLArtifactWebView: NSViewRepresentable {
    let source: String
    @Binding var measured: CGFloat?

    func makeCoordinator() -> Coordinator { Coordinator(measured: $measured) }

    func makeNSView(context: Context) -> WKWebView {
        let web = WKWebView(frame: .zero, configuration: context.coordinator.makeConfiguration())
        web.navigationDelegate = context.coordinator
        web.uiDelegate = context.coordinator
        web.allowsMagnification = false
        web.allowsBackForwardNavigationGestures = false
        web.allowsLinkPreview = false
        context.coordinator.load(source, into: web)
        return web
    }

    func updateNSView(_ web: WKWebView, context: Context) {
        // Streaming calls this many times a second while the rest of the reply
        // arrives. `load` compares the source and no-ops, so an artifact is not
        // re-run — and a re-run would restart its animations and lose whatever
        // the reader had already interacted with.
        context.coordinator.measured = $measured
        context.coordinator.load(source, into: web)
    }

    static func dismantleNSView(_ web: WKWebView, coordinator: Coordinator) {
        coordinator.tearDown(web)
    }

    final class Coordinator: NSObject, WKNavigationDelegate, WKUIDelegate, WKScriptMessageHandler {
        /// Also the name the injected script posts to — one spelling.
        static let heightHandler = "mlxArtifactHeight"

        var measured: Binding<CGFloat?>
        /// The source currently loaded, so an unchanged update is free.
        private var loaded: String?
        /// Set once the view is dismantled. The blocker compiles asynchronously,
        /// so a block scrolled away (or a reply deleted) while that is in flight
        /// would otherwise have its callback start the page up again in a web
        /// view already torn down.
        private var dismantled = false

        init(measured: Binding<CGFloat?>) {
            self.measured = measured
        }

        func makeConfiguration() -> WKWebViewConfiguration {
            let config = WKWebViewConfiguration()
            config.websiteDataStore = ArtifactWebEnvironment.shared.dataStore
            config.defaultWebpagePreferences.allowsContentJavaScript = true
            config.preferences.javaScriptCanOpenWindowsAutomatically = false
            // A model that writes `<video autoplay>` must not make noise in a
            // transcript somebody is reading.
            config.mediaTypesRequiringUserActionForPlayback = .all

            let content = WKUserContentController()
            content.addUserScript(WKUserScript(source: Self.heightScript,
                                               injectionTime: .atDocumentEnd,
                                               forMainFrameOnly: true))
            content.add(WeakScriptMessageHandler(self), name: Self.heightHandler)
            config.userContentController = content
            return config
        }

        func load(_ source: String, into web: WKWebView) {
            guard loaded != source else { return }
            loaded = source
            // New content is a new height. Setting it here would be a state
            // write inside SwiftUI's own update, so it lands on the next turn.
            DispatchQueue.main.async { [weak self] in self?.measured.wrappedValue = nil }

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

        func tearDown(_ web: WKWebView) {
            dismantled = true
            web.stopLoading()
            web.navigationDelegate = nil
            web.uiDelegate = nil
            let content = web.configuration.userContentController
            content.removeAllUserScripts()
            content.removeAllContentRuleLists()
            content.removeScriptMessageHandler(forName: Self.heightHandler)
            // An artifact scrolled out of the transcript can still be running a
            // `requestAnimationFrame` loop; replacing the document stops it.
            web.loadHTMLString("", baseURL: nil)
        }

        // MARK: Height

        func userContentController(_ controller: WKUserContentController,
                                   didReceive message: WKScriptMessage) {
            guard message.name == Self.heightHandler,
                  let number = message.body as? NSNumber else { return }
            let height = CGFloat(truncating: number)
            guard height.isFinite, height >= 0 else { return }
            // Hysteresis. A page whose layout settles a fraction of a point at a
            // time would otherwise re-lay out the whole transcript per frame.
            if let current = measured.wrappedValue, abs(current - height) < 1 { return }
            measured.wrappedValue = height
        }

        /// Reports the document's own height, now and whenever it changes.
        ///
        /// Measured from `body`, not `documentElement`: the latter never
        /// reports less than the viewport, so a block sized to it could grow
        /// and never shrink. Body margins are added back because a complete
        /// document the model wrote keeps the browser's default 8px, which the
        /// bounding rect excludes.
        ///
        /// The timers are for content that lays out after `load` — a canvas a
        /// script draws, an image decoded late, a font swapping in.
        private static let heightScript = #"""
        (function () {
          var last = -1;
          function report() {
            var body = document.body;
            if (!body) { return; }
            var style = window.getComputedStyle(body);
            var margins = (parseFloat(style.marginTop) || 0) + (parseFloat(style.marginBottom) || 0);
            var height = Math.max(body.scrollHeight + margins,
                                  body.getBoundingClientRect().height + margins);
            if (!isFinite(height)) { return; }
            height = Math.ceil(Math.min(height, 100000));
            if (Math.abs(height - last) < 1) { return; }
            last = height;
            try { window.webkit.messageHandlers.mlxArtifactHeight.postMessage(height); } catch (e) {}
          }
          report();
          window.addEventListener('load', report);
          window.addEventListener('resize', report);
          if (window.ResizeObserver) { new ResizeObserver(report).observe(document.body); }
          [16, 120, 400, 1200, 3000].forEach(function (delay) { window.setTimeout(report, delay); });
        })();
        """#

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
