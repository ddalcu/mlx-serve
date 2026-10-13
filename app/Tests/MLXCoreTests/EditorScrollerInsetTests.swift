import XCTest
import AppKit
@testable import MLXCore

final class EditorScrollerInsetTests: XCTestCase {
    /// The width a vertical scroller of `style` takes from a scroll view's content.
    @MainActor
    private func widthTakenByScroller(_ style: NSScroller.Style) -> CGFloat {
        let scroll = NSScrollView(frame: NSRect(x: 0, y: 0, width: 300, height: 100))
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = false
        scroll.scrollerStyle = style
        scroll.documentView = NSView(frame: NSRect(x: 0, y: 0, width: 300, height: 400))
        scroll.tile()
        return scroll.frame.width - scroll.contentView.frame.width
    }

    // The bar: text laid over an editor keeps clear of exactly the room the editor's scroller takes.
    @MainActor
    func testTheInsetIsTheRoomTheScrollerTakes() {
        XCTAssertGreaterThan(widthTakenByScroller(.legacy), 0)
        XCTAssertEqual(EditorScrollerInset.trailing(for: .legacy), widthTakenByScroller(.legacy))
        XCTAssertEqual(EditorScrollerInset.trailing(for: .overlay), widthTakenByScroller(.overlay))
    }
}
