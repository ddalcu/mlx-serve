import XCTest
import SwiftUI
@testable import MLXCore

/// The banner over the Music settings while AI Radio runs.
@MainActor
final class RadioLockBannerTests: XCTestCase {

    /// The bar is a card inside the column's margin: never under the floating sidebar, never on the window's edge.
    func testTheBannerKeepsTheColumnsMargin() throws {
        let root = Color.clear
            .overlay(alignment: .top) { RadioLockBanner() }
            .safeAreaPadding(.leading, 60)
            .frame(width: 300, height: 80)
        let host = NSHostingView(rootView: root)
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 300, height: 80), styleMask: [.borderless],
                              backing: .buffered, defer: false)
        window.contentView = host
        host.frame = NSRect(x: 0, y: 0, width: 300, height: 80)
        host.layoutSubtreeIfNeeded()
        let rep = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: rep)
        let scale = CGFloat(rep.pixelsWide) / host.bounds.width
        func alpha(_ x: CGFloat, _ y: CGFloat) -> CGFloat {
            rep.colorAt(x: Int(x * scale), y: Int(y * scale))?.alphaComponent ?? 0
        }
        XCTAssertGreaterThan(alpha(180, 20), 0, "the banner draws in the column")
        XCTAssertEqual(alpha(20, 20), 0, "the banner fills the safe area the sidebar covers")
        XCTAssertEqual(alpha(68, 20), 0, "the banner fills the column's margin")
    }
}
