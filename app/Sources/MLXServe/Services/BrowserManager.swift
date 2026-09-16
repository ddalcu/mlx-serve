import Foundation
import WebKit

@MainActor
class BrowserManager: ObservableObject {
    static let shared = BrowserManager()

    // Mirrored from the webView by KVO, so link clicks, back/forward and
    // same-document pushState reach the URL bar, not only tool navigations.
    @Published var currentURL: String = ""
    @Published var pageTitle: String = ""
    @Published var isLoading: Bool = false
    @Published var canGoBack: Bool = false
    @Published var canGoForward: Bool = false
    /// Bumped by `requestShow`; the app scene opens the Browser window on change.
    @Published var showRequestTick = 0

    /// Always available — created eagerly so tools work without the Browser window.
    let webView: WKWebView

    /// Never shown: the frame `window.outerWidth/outerHeight` is read from
    /// while no Browser pane holds the webView (0 reads as a headless bot).
    let hostWindow: NSWindow
    private let uiDelegate = WindowFrameUIDelegate()
    private let navDelegate = NavigationDelegate()
    private var observations: [NSKeyValueObservation] = []

    private init() {
        let config = WKWebViewConfiguration()
        config.preferences.isElementFullscreenEnabled = true
        let frame = NSRect(x: 0, y: 0, width: 1024, height: 768)
        self.webView = WKWebView(frame: frame, configuration: config)
        webView.allowsBackForwardNavigationGestures = true
        hostWindow = NSWindow(contentRect: frame, styleMask: [.titled, .resizable],
                              backing: .buffered, defer: false)
        hostWindow.isReleasedWhenClosed = false
        hostWindow.isExcludedFromWindowsMenu = true
        hostWindow.contentView = webView
        webView.uiDelegate = uiDelegate
        webView.navigationDelegate = navDelegate
        observations = [
            webView.observe(\.url, options: [.initial, .new]) { [weak self] wv, _ in
                MainActor.assumeIsolated { self?.currentURL = wv.url?.absoluteString ?? "" }
            },
            webView.observe(\.title, options: [.initial, .new]) { [weak self] wv, _ in
                MainActor.assumeIsolated { self?.pageTitle = wv.title ?? "" }
            },
            webView.observe(\.isLoading, options: [.initial, .new]) { [weak self] wv, _ in
                MainActor.assumeIsolated { self?.isLoading = wv.isLoading }
            },
            webView.observe(\.canGoBack, options: [.initial, .new]) { [weak self] wv, _ in
                MainActor.assumeIsolated { self?.canGoBack = wv.canGoBack }
            },
            webView.observe(\.canGoForward, options: [.initial, .new]) { [weak self] wv, _ in
                MainActor.assumeIsolated { self?.canGoForward = wv.canGoForward }
            },
        ]
    }

    func requestShow() { showRequestTick += 1 }

    /// What a model-typed target means: an explicit scheme as is, a path on disk
    /// (absolute, or relative to the working directory) as file://, a loopback
    /// host as http:// (a dev server has no certificate), anything else https://.
    nonisolated static func resolveURL(_ raw: String, workingDirectory: String?) -> URL? {
        let s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !s.isEmpty else { return nil }
        if s.contains("://") { return URL(string: s) }
        if s.hasPrefix("/") || s.hasPrefix("~") {
            return URL(fileURLWithPath: (s as NSString).expandingTildeInPath)
        }
        if let wd = workingDirectory {
            let path = (wd as NSString).appendingPathComponent(s)
            if FileManager.default.fileExists(atPath: path) {
                return URL(fileURLWithPath: (path as NSString).standardizingPath)
            }
        }
        let host = s.split(whereSeparator: { $0 == "/" || $0 == ":" }).first.map(String.init) ?? ""
        let loopback = ["localhost", "127.0.0.1", "0.0.0.0", "[", "::1"].contains { host.hasPrefix($0) }
        return URL(string: (loopback ? "http://" : "https://") + s)
    }

    /// Re-parents the webView back into the hidden host once a pane lets go of it.
    func returnToHost() {
        guard webView.window !== hostWindow else { return }
        webView.removeFromSuperview()
        hostWindow.contentView = webView
    }

    /// Loads `url` and returns once the navigation finished (30 s cap).
    func load(_ url: URL) async throws -> String {
        try await withThrowingTaskGroup(of: String.self) { group in
            group.addTask { @MainActor in
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, Error>) in
                    self.navDelegate.begin(continuation)
                    if url.isFileURL {
                        self.webView.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
                    } else {
                        self.webView.load(URLRequest(url: url))
                    }
                }
            }
            group.addTask {
                try await Task.sleep(nanoseconds: 30_000_000_000)
                throw ToolError.executionFailed("Navigation timed out after 30s: \(url.absoluteString)")
            }
            let result = try await group.next()!
            group.cancelAll()
            return result
        }
    }

    func navigate(to urlString: String, workingDirectory: String? = nil) async throws -> String {
        guard let url = Self.resolveURL(urlString, workingDirectory: workingDirectory) else {
            throw ToolError.executionFailed("Invalid URL: \(urlString)")
        }
        let navResult = try await load(url)

        // Auto-read page content after navigation
        try await Task.sleep(nanoseconds: 500_000_000) // let JS render
        let rawText = try await readText()
        let title = webView.title ?? ""
        let text = Self.cleanExtractedText(rawText)
        return "\(navResult)\nTitle: \(title)\n\nPage content:\n\(text)"
    }

    func readText() async throws -> String {
        let js = """
        (function() {
            var root = document.querySelector('main') || document.querySelector('[role="main"]') || document.querySelector('article') || document.body;
            var clone = root.cloneNode(true);
            var remove = clone.querySelectorAll('script,style,nav,header,footer,form,iframe,noscript,svg,img,video,audio,canvas,button,input,select,textarea,details,menu,datalist,[role="navigation"],[role="banner"],[role="contentinfo"],[role="complementary"],[role="listbox"],[role="combobox"],[aria-hidden="true"],.nav,.navbar,.menu,.sidebar,.footer,.header,.ad,.ads,.advert,.cookie,.popup,.modal,.overlay,.social,.share,.comment,.related');
            for (var i = 0; i < remove.length; i++) remove[i].remove();
            var text = clone.innerText || '';
            // Aggressive cleanup: trim each line, collapse blank lines, remove noise
            text = text.split('\\n')
                .map(function(line) { return line.trim(); })
                .filter(function(line) { return line.length > 0; })
                .join('\\n');
            // Collapse any remaining multi-newlines
            text = text.replace(/\\n{3,}/g, '\\n\\n');
            return text.substring(0, 3000);
        })()
        """
        return try await evaluateJSWithTimeout(js, description: "readText")
    }

    func extractText(selector: String) async throws -> String {
        let escaped = selector
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "'", with: "\\'")
        let js = """
        (function() {
            var els = document.querySelectorAll('\(escaped)');
            if (!els || els.length === 0) return 'No elements match selector: \(escaped)';
            var parts = [];
            var max = Math.min(els.length, 50);
            for (var i = 0; i < max; i++) {
                var t = (els[i].innerText || '').trim();
                if (t.length > 0) parts.push(t);
            }
            if (parts.length === 0) return 'No elements match selector: \(escaped)';
            var joined = parts.join('\\n---\\n');
            return joined.substring(0, 2900);
        })()
        """
        return try await evaluateJSWithTimeout(js, description: "extractText")
    }

    func readHTML() async throws -> String {
        let js = "document.documentElement.outerHTML.substring(0, 8000)"
        return try await evaluateJSWithTimeout(js, description: "readHTML")
    }

    func click(selector: String) async throws -> String {
        let escaped = selector.replacingOccurrences(of: "'", with: "\\'")
        let js = """
        (function() {
            var el = document.querySelector('\(escaped)');
            if (!el) return 'Element not found: \(escaped)';
            el.click();
            return 'Clicked: ' + (el.tagName || '') + ' ' + (el.textContent || '').substring(0, 50);
        })()
        """
        return try await evaluateJSWithTimeout(js, description: "click")
    }

    func evaluateJS(_ script: String) async throws -> String {
        // Strip leading "return" — WKWebView evaluates expressions, not function bodies,
        // so bare "return" causes a SyntaxError. Models often add it by habit.
        var js = script.trimmingCharacters(in: .whitespacesAndNewlines)
        if js.hasPrefix("return ") || js.hasPrefix("return\n") {
            js = String(js.dropFirst(7))
        }
        return try await evaluateJSWithTimeout(js, description: "evaluateJS")
    }

    /// Wrap `evaluateJavaScript` in a 25s timeout so a stuck page can't freeze
    /// the agent loop. WKWebView itself has no built-in JS-eval timeout.
    /// Result is converted to String on the main actor to avoid sending non-Sendable
    /// `Any?` values across task boundaries.
    private func evaluateJSWithTimeout(_ script: String, description: String) async throws -> String {
        try await withThrowingTaskGroup(of: String.self) { group in
            let capturedWebView = webView
            group.addTask { @MainActor in
                let raw = try await capturedWebView.evaluateJavaScript(script)
                return jsResultToString(raw)
            }
            group.addTask {
                try await Task.sleep(nanoseconds: 25_000_000_000)
                throw ToolError.executionFailed("browser \(description) timed out after 25s")
            }
            let result = try await group.next()!
            group.cancelAll()
            return result
        }
    }

    func takeScreenshot() async -> Data? {
        let config = WKSnapshotConfiguration()
        config.snapshotWidth = NSNumber(value: 1024)
        do {
            let image = try await webView.takeSnapshot(configuration: config)
            guard let tiff = image.tiffRepresentation,
                  let bitmap = NSBitmapImageRep(data: tiff),
                  let jpeg = bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.8]) else {
                return nil
            }
            return jpeg
        } catch {
            return nil
        }
    }

    func getInfo() async throws -> String {
        let url = webView.url?.absoluteString ?? "about:blank"
        let title = webView.title ?? ""
        return "Title: \(title)\nURL: \(url)"
    }

    /// Clean extracted page text for optimal LLM consumption.
    /// Removes whitespace noise, short junk lines, and caps length.
    static func cleanExtractedText(_ raw: String) -> String {
        let lines = raw.components(separatedBy: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { line in
                guard !line.isEmpty else { return false }
                // Drop very short lines that are usually UI artifacts (e.g., "Ad", single chars)
                if line.count < 3 { return false }
                // Drop lines that are only punctuation/symbols
                let alphanumeric = line.unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }
                if alphanumeric.count == 0 { return false }
                return true
            }

        var result = lines.joined(separator: "\n")

        // Collapse remaining multi-newlines
        while result.contains("\n\n\n") {
            result = result.replacingOccurrences(of: "\n\n\n", with: "\n\n")
        }

        // Cap at 1500 chars — small models work better with concise context
        if result.count > 1500 {
            result = String(result.prefix(1500))
            // Don't cut mid-line
            if let lastNewline = result.lastIndex(of: "\n") {
                result = String(result[...lastNewline])
            }
        }

        return result
    }
}

// MARK: - Navigation Delegate

/// WebKit answers the page's window-frame query with a ZERO rect unless the UI
/// delegate implements this (private) selector; hosting alone never sets it.
/// Compiled out for the store, whose binary must carry no private selector.
private final class WindowFrameUIDelegate: NSObject, WKUIDelegate {
    #if !MAS_BUILD
    @objc func _webView(_ webView: WKWebView, getWindowFrameWithCompletionHandler handler: @escaping (CGRect) -> Void) {
        handler(webView.window?.frame ?? .zero)
    }
    #endif
}

/// One delegate for the webView's whole life; a tool navigation parks its
/// continuation here and the next finish or failure resolves it.
private final class NavigationDelegate: NSObject, WKNavigationDelegate {
    private var pending: CheckedContinuation<String, Error>?

    func begin(_ continuation: CheckedContinuation<String, Error>) {
        pending?.resume(throwing: ToolError.executionFailed("Navigation superseded"))
        pending = continuation
    }

    private func finish(_ result: Result<String, Error>) {
        guard let c = pending else { return }
        pending = nil
        c.resume(with: result)
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        finish(.success("Navigated to \(webView.url?.absoluteString ?? "")"))
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        finish(.failure(ToolError.executionFailed("Navigation failed: \(error.localizedDescription)")))
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        finish(.failure(ToolError.executionFailed("Navigation failed: \(error.localizedDescription)")))
    }
}

/// Safely convert JavaScript evaluation result (Any?) to String without Optional() wrapper.
private func jsResultToString(_ result: Any?) -> String {
    guard let result else { return "" }
    if let str = result as? String { return str }
    return "\(result)"
}
