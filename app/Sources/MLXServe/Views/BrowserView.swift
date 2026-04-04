import SwiftUI
import WebKit

struct BrowserView: View {
    @ObservedObject var browser = BrowserManager.shared
    @State private var urlText: String = "https://www.google.com"

    var body: some View {
        VStack(spacing: 0) {
            // URL bar
            HStack(spacing: 8) {
                Button {
                    browser.webView?.goBack()
                } label: {
                    Image(systemName: "chevron.left")
                }
                .buttonStyle(.plain)
                .disabled(!(browser.webView?.canGoBack ?? false))

                Button {
                    browser.webView?.goForward()
                } label: {
                    Image(systemName: "chevron.right")
                }
                .buttonStyle(.plain)
                .disabled(!(browser.webView?.canGoForward ?? false))

                Button {
                    if browser.isLoading {
                        browser.webView?.stopLoading()
                    } else {
                        browser.webView?.reload()
                    }
                } label: {
                    Image(systemName: browser.isLoading ? "xmark" : "arrow.clockwise")
                }
                .buttonStyle(.plain)

                TextField("URL", text: $urlText)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit {
                        navigateToURL()
                    }
            }
            .padding(8)

            Divider()

            // WebView
            WebViewWrapper(browser: browser)
        }
        .frame(minWidth: 800, minHeight: 600)
        .onAppear {
            if browser.currentURL.isEmpty {
                navigateToURL()
            }
        }
        .onChange(of: browser.currentURL) { _, newURL in
            if !newURL.isEmpty { urlText = newURL }
        }
    }

    private func navigateToURL() {
        var url = urlText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !url.hasPrefix("http://") && !url.hasPrefix("https://") {
            url = "https://" + url
        }
        Task {
            try? await browser.navigate(to: url)
        }
    }
}

struct WebViewWrapper: NSViewRepresentable {
    let browser: BrowserManager

    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.preferences.isElementFullscreenEnabled = true
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.allowsBackForwardNavigationGestures = true
        webView.navigationDelegate = context.coordinator

        Task { @MainActor in
            browser.webView = webView
        }

        return webView
    }

    func updateNSView(_ nsView: WKWebView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(browser: browser)
    }

    class Coordinator: NSObject, WKNavigationDelegate {
        let browser: BrowserManager

        init(browser: BrowserManager) {
            self.browser = browser
        }

        func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
            Task { @MainActor in
                browser.isLoading = true
            }
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            Task { @MainActor in
                browser.currentURL = webView.url?.absoluteString ?? ""
                browser.pageTitle = webView.title ?? ""
                browser.isLoading = false
            }
        }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            Task { @MainActor in
                browser.isLoading = false
            }
        }
    }
}
