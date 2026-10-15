import XCTest
import SwiftUI
@testable import MLXCore

/// The composer's "/" menu, drawn.
@MainActor
final class SlashSkillMenuTests: XCTestCase {

    private let matches = [SkillSummary(name: "music3", description: "Write a caption plus lyrics"),
                           SkillSummary(name: "review", description: "Review the current changes")]

    /// A 300x220 window whose bottom 80pt is the field, with room above it for the menu. Ordered in far
    /// off-screen: a window that never was takes no clicks.
    private func window(onPick: @escaping (SkillSummary) -> Void = { _ in }) -> NSWindow {
        let root = VStack(spacing: 0) {
            Color.clear.frame(height: 140)
            Color.blue.frame(height: 80)
                .slashSkillMenu(matches: matches, selection: 0, onPick: onPick)
        }
        .frame(width: 300, height: 220)
        let window = NSWindow(contentRect: NSRect(x: -20000, y: -20000, width: 300, height: 220), styleMask: [.borderless],
                              backing: .buffered, defer: false)
        window.contentView = NSHostingView(rootView: root)
        window.orderFrontRegardless()
        addTeardownBlock { window.orderOut(nil) }
        RunLoop.main.run(until: Date().addingTimeInterval(0.2))
        return window
    }

    /// The menu hangs above the field it completes, never over the text being typed.
    func testTheMenuHangsAboveTheField() throws {
        let host = try XCTUnwrap(window().contentView)
        let rep = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: rep)
        let scale = CGFloat(rep.pixelsWide) / host.bounds.width
        func color(_ x: CGFloat, _ y: CGFloat) -> NSColor? {
            rep.colorAt(x: Int(x * scale), y: Int(y * scale))?.usingColorSpace(.sRGB)
        }
        // Against the field's own colour at its bottom edge, so any appearance passes.
        XCTAssertEqual(color(150, 180), try XCTUnwrap(color(150, 215)), "the field shows through")
        XCTAssertGreaterThan(color(150, 122)?.alphaComponent ?? 0, 0.9, "the menu draws above the field")
    }

    /// A row hanging outside the field's bounds still takes the click.
    func testARowAboveTheFieldTakesTheClick() {
        var picked: [String] = []
        let window = window(onPick: { picked.append($0.name) })
        for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
            window.sendEvent(NSEvent.mouseEvent(with: type, location: NSPoint(x: 150, y: 220 - 120), modifierFlags: [],
                                                timestamp: ProcessInfo.processInfo.systemUptime,
                                                windowNumber: window.windowNumber, context: nil, eventNumber: 0,
                                                clickCount: 1, pressure: 1)!)
            RunLoop.main.run(until: Date().addingTimeInterval(0.1))
        }
        XCTAssertEqual(picked, ["review"], "the last row, 12pt above the field")
    }
}
