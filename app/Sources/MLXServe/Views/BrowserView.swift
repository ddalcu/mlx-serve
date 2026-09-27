import SwiftUI
import WebKit

struct BrowserView: View {
    @ObservedObject var browser = BrowserManager.shared
    @State private var urlText: String = "https://www.google.com"

    var body: some View {
        VStack(spacing: 0) {
            // URL bar
            HStack(spacing: 8) {
                navButton("chevron.left", enabled: browser.canGoBack) { browser.webView.goBack() }
                navButton("chevron.right", enabled: browser.canGoForward) { browser.webView.goForward() }
                navButton(browser.isLoading ? "xmark" : "arrow.clockwise", enabled: true) {
                    if browser.isLoading { browser.webView.stopLoading() } else { browser.webView.reload() }
                }

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

    private func navButton(_ symbol: String, enabled: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.app(.title3))
                .frame(width: 28, height: 28)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
    }

    private func navigateToURL() {
        Task { try? await browser.navigate(to: urlText) }
    }
}

struct WebViewWrapper: NSViewRepresentable {
    let browser: BrowserManager

    func makeNSView(context: Context) -> WKWebView { browser.webView }

    func updateNSView(_ nsView: WKWebView, context: Context) {}

    static func dismantleNSView(_ nsView: WKWebView, coordinator: ()) {
        BrowserManager.shared.returnToHost()
    }
}
