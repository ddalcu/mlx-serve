import AppKit
import XCTest
@testable import MLXCore

/// The tinted ground behind an inline code span is drawn by the text view, one
/// rounded rect per line the span occupies. A span that wraps used to take the
/// first line all the way to the right margin, because the rects came from
/// `enumerateEnclosingRects`, which is the SELECTION geometry.
final class InlineCodeGroundTests: XCTestCase {

    /// A laid-out paragraph in a container `width` points wide.
    private func layout(_ text: String, width: CGFloat)
        -> (storage: NSTextStorage, manager: NSLayoutManager, container: NSTextContainer) {
        let storage = NSTextStorage(string: text, attributes: [
            .font: NSFont.monospacedSystemFont(ofSize: 12, weight: .regular),
        ])
        let manager = NSLayoutManager()
        let container = NSTextContainer(size: NSSize(width: width, height: .greatestFiniteMagnitude))
        container.lineFragmentPadding = 0
        storage.addLayoutManager(manager)
        manager.addTextContainer(container)
        manager.ensureLayout(for: container)
        return (storage, manager, container)
    }

    /// The storage owns the layout manager and the manager's back reference is
    /// weak, so the whole tuple is held: dropping the storage empties
    /// `layoutManager.textStorage` before the call.
    private func rects(_ text: String, span: NSRange, width: CGFloat) -> [NSRect] {
        let laid = layout(text, width: width)
        let glyphs = laid.manager.glyphRange(forCharacterRange: span, actualCharacterRange: nil)
        return InlineCodeGround.rects(forGlyphRange: glyphs, layoutManager: laid.manager, in: laid.container)
    }

    /// The bar: no ground may reach past the glyphs on its own line. The used
    /// rect of a line fragment is exactly "how far the text got".
    private func assertWithinTheirLines(_ text: String, span: NSRange, width: CGFloat,
                                        file: StaticString = #filePath, line: UInt = #line) {
        let (storage, manager, container) = layout(text, width: width)
        let glyphs = manager.glyphRange(forCharacterRange: span, actualCharacterRange: nil)
        let grounds = InlineCodeGround.rects(forGlyphRange: glyphs, layoutManager: manager, in: container)
        var used: [NSRect] = []
        manager.enumerateLineFragments(
            forGlyphRange: manager.glyphRange(forCharacterRange: NSRange(location: 0, length: storage.length),
                                              actualCharacterRange: nil)
        ) { _, usedRect, _, _, _ in used.append(usedRect) }

        for ground in grounds {
            guard let fragment = used.first(where: { $0.minY <= ground.midY && ground.midY <= $0.maxY }) else {
                XCTFail("a ground outside every line fragment: \(ground)", file: file, line: line)
                continue
            }
            XCTAssertLessThanOrEqual(ground.maxX, fragment.maxX + 0.5,
                                     "a ground ran past its line's own text", file: file, line: line)
        }
    }

    func testASpanOnOneLineGetsOneGround() {
        let text = "a `code` b"
        let grounds = rects(text, span: NSRange(location: 3, length: 4), width: 400)
        XCTAssertEqual(grounds.count, 1)
        assertWithinTheirLines(text, span: NSRange(location: 3, length: 4), width: 400)
    }

    /// The failing shape: a long path broken across lines.
    func testAWrappedSpanGetsOneGroundPerLineAndNoneRunsToTheMargin() {
        let text = "saved at generations/audio/2026-09-14/2026-09-14_20-33-14_this-is-a-fox.wav now"
        let span = NSRange(location: 9, length: 62)
        let grounds = rects(text, span: span, width: 220)
        XCTAssertGreaterThanOrEqual(grounds.count, 2, "a wrapped span is drawn per line")
        assertWithinTheirLines(text, span: span, width: 220)
    }

    /// A trailing space before the break is not part of the ground: it is what
    /// made the first line's rect reach the edge in the first place.
    func testTheGroundStopsAtTheLastGlyphBeforeTheBreak() {
        let text = "lead `one two three four five six seven eight` tail"
        let span = NSRange(location: 6, length: 39)
        assertWithinTheirLines(text, span: span, width: 160)
    }

    func testAnEmptyRangeDrawsNothing() {
        XCTAssertTrue(rects("nothing here", span: NSRange(location: 3, length: 0), width: 200).isEmpty)
    }
}
