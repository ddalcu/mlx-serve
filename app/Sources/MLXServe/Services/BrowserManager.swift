import Foundation
import WebKit

@MainActor
class BrowserManager: ObservableObject {
    static let shared = BrowserManager()

    @Published var currentURL: String = ""
    @Published var pageTitle: String = ""
    @Published var isLoading: Bool = false

    var webView: WKWebView?

    func navigate(to urlString: String) async throws -> String {
        guard let url = URL(string: urlString) else {
            throw ToolError.executionFailed("Invalid URL: \(urlString)")
        }
        guard let webView else {
            throw ToolError.executionFailed("Browser not open. Open the browser window first.")
        }

        let navResult = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, Error>) in
            let delegate = NavigationDelegate(continuation: continuation)
            webView.navigationDelegate = delegate
            objc_setAssociatedObject(webView, "navDelegate", delegate, .OBJC_ASSOCIATION_RETAIN)
            webView.load(URLRequest(url: url))
        }

        // Auto-read page content after navigation
        try await Task.sleep(nanoseconds: 500_000_000) // let JS render
        let text = try await readText()
        let title = webView.title ?? ""
        return "\(navResult)\nTitle: \(title)\n\nPage content:\n\(text)"
    }

    func readText() async throws -> String {
        guard let webView else {
            throw ToolError.executionFailed("Browser not open.")
        }
        let js = """
        (function() {
            var clone = document.body.cloneNode(true);
            var remove = clone.querySelectorAll('script,style,nav,header,footer,form,iframe,noscript,svg,[role="navigation"],[role="banner"],[role="contentinfo"],.nav,.navbar,.menu,.sidebar,.footer,.header,.ad,.ads,.cookie');
            for (var i = 0; i < remove.length; i++) remove[i].remove();
            var text = clone.innerText || '';
            text = text.replace(/\\n{3,}/g, '\\n\\n').replace(/[ \\t]+/g, ' ').trim();
            return text.substring(0, 4000);
        })()
        """
        let result = try await webView.evaluateJavaScript(js)
        return jsResultToString(result)
    }

    func readHTML() async throws -> String {
        guard let webView else {
            throw ToolError.executionFailed("Browser not open.")
        }
        let js = "document.documentElement.outerHTML.substring(0, 8000)"
        let result = try await webView.evaluateJavaScript(js)
        return jsResultToString(result)
    }

    func click(selector: String) async throws -> String {
        guard let webView else {
            throw ToolError.executionFailed("Browser not open.")
        }
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
        guard let webView else {
            throw ToolError.executionFailed("Browser not open.")
        }
        let result = try await webView.evaluateJavaScript(script)
        return jsResultToString(result)
    }

    func getInfo() async throws -> String {
        guard let webView else {
            throw ToolError.executionFailed("Browser not open.")
        }
        let url = webView.url?.absoluteString ?? "about:blank"
        let title = webView.title ?? ""
        return "Title: \(title)\nURL: \(url)"
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
