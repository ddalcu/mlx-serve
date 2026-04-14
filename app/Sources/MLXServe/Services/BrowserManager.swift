import Foundation
import WebKit

@MainActor
class BrowserManager: ObservableObject {
    static let shared = BrowserManager()

    @Published var currentURL: String = ""
    @Published var pageTitle: String = ""
    @Published var isLoading: Bool = false

    /// Always available — created eagerly so tools work without the Browser window.
    let webView: WKWebView

    private init() {
        let config = WKWebViewConfiguration()
        config.preferences.isElementFullscreenEnabled = true
        self.webView = WKWebView(frame: NSRect(x: 0, y: 0, width: 1024, height: 768), configuration: config)
        webView.allowsBackForwardNavigationGestures = true
    }

    func navigate(to urlString: String) async throws -> String {
        // Auto-fix missing scheme — models often omit https://
        var normalized = urlString
        if !normalized.hasPrefix("http://") && !normalized.hasPrefix("https://") {
            normalized = "https://" + normalized
        }
        guard let url = URL(string: normalized) else {
            throw ToolError.executionFailed("Invalid URL: \(urlString)")
        }

        let navResult = try await withThrowingTaskGroup(of: String.self) { group in
            group.addTask { @MainActor in
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, Error>) in
                    let delegate = NavigationDelegate(continuation: continuation)
                    self.webView.navigationDelegate = delegate
                    objc_setAssociatedObject(self.webView, "navDelegate", delegate, .OBJC_ASSOCIATION_RETAIN)
                    self.webView.load(URLRequest(url: url))
                }
            }
            group.addTask {
                try await Task.sleep(nanoseconds: 30_000_000_000)
                throw ToolError.executionFailed("Navigation timed out after 30s: \(urlString)")
            }
            let result = try await group.next()!
            group.cancelAll()
            return result
        }

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
            var clone = document.body.cloneNode(true);
            var remove = clone.querySelectorAll('script,style,nav,header,footer,form,iframe,noscript,svg,img,video,audio,canvas,button,input,select,textarea,[role="navigation"],[role="banner"],[role="contentinfo"],[role="complementary"],[aria-hidden="true"],.nav,.navbar,.menu,.sidebar,.footer,.header,.ad,.ads,.advert,.cookie,.popup,.modal,.overlay,.social,.share,.comment,.related');
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
        let result = try await webView.evaluateJavaScript(js)
        return jsResultToString(result)
    }

    func readHTML() async throws -> String {
        let js = "document.documentElement.outerHTML.substring(0, 8000)"
        let result = try await webView.evaluateJavaScript(js)
        return jsResultToString(result)
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
        let result = try await webView.evaluateJavaScript(js)
        return jsResultToString(result)
    }

    func evaluateJS(_ script: String) async throws -> String {
        // Strip leading "return" — WKWebView evaluates expressions, not function bodies,
        // so bare "return" causes a SyntaxError. Models often add it by habit.
        var js = script.trimmingCharacters(in: .whitespacesAndNewlines)
        if js.hasPrefix("return ") || js.hasPrefix("return\n") {
            js = String(js.dropFirst(7))
        }
        let result = try await webView.evaluateJavaScript(js)
        return jsResultToString(result)
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

private class NavigationDelegate: NSObject, WKNavigationDelegate {
    let continuation: CheckedContinuation<String, Error>
    var finished = false

    init(continuation: CheckedContinuation<String, Error>) {
        self.continuation = continuation
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        guard !finished else { return }
        finished = true
        let url = webView.url?.absoluteString ?? ""
        Task { @MainActor in
            BrowserManager.shared.currentURL = url
            BrowserManager.shared.pageTitle = webView.title ?? ""
            BrowserManager.shared.isLoading = false
        }
        continuation.resume(returning: "Navigated to \(url)")
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        guard !finished else { return }
        finished = true
        Task { @MainActor in
            BrowserManager.shared.isLoading = false
        }
        continuation.resume(throwing: ToolError.executionFailed("Navigation failed: \(error.localizedDescription)"))
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        guard !finished else { return }
        finished = true
        Task { @MainActor in
            BrowserManager.shared.isLoading = false
        }
        continuation.resume(throwing: ToolError.executionFailed("Navigation failed: \(error.localizedDescription)"))
    }
}

/// Safely convert JavaScript evaluation result (Any?) to String without Optional() wrapper.
private func jsResultToString(_ result: Any?) -> String {
    guard let result else { return "" }
    if let str = result as? String { return str }
    return "\(result)"
}
