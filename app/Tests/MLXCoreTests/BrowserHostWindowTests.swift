import XCTest
import AppKit
@testable import MLXCore

/// The agent's browse/webSearch tools run with no Browser pane open. Bar: the
/// page still reads a non-zero window.outerWidth/outerHeight (0 = headless bot).
@MainActor
final class BrowserHostWindowTests: XCTestCase {
    #if !MAS_BUILD
    func testPageSeesANonZeroOuterWindowSizeWithNoPaneOpen() async throws {
        let webView = BrowserManager.shared.webView
        webView.loadHTMLString("<html></html>", baseURL: nil)
        for _ in 0..<50 where webView.isLoading { try await Task.sleep(nanoseconds: 100_000_000) }
        let size = try await webView.evaluateJavaScript("[window.outerWidth, window.outerHeight]") as? [Int]
        XCTAssertGreaterThan(size?[0] ?? 0, 0)
        XCTAssertGreaterThan(size?[1] ?? 0, 0)
    }
    #endif

    func testReleasingFromAPaneReturnsItToTheHostWindow() {
        let mgr = BrowserManager.shared
        let other = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 300, height: 200),
                             styleMask: [.titled], backing: .buffered, defer: false)
        other.contentView?.addSubview(mgr.webView)
        XCTAssertTrue(mgr.webView.window === other)
        mgr.returnToHost()
        XCTAssertTrue(mgr.webView.window === mgr.hostWindow)
    }
}
