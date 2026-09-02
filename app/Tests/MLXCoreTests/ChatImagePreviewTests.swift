import AppKit
import XCTest
@testable import MLXCore

/// Opening an inline chat image in Preview stages its JPEG bytes to a
/// deterministic temp file first (`ChatImage` carries only bytes, no path).
/// The NSWorkspace open is untestable, but the staging is pure filesystem.
final class ChatImagePreviewTests: XCTestCase {

    func testWriteTempFileStagesBytesAtDeterministicPath() throws {
        let data = Data("pretend-jpeg-bytes".utf8)
        let img = ChatImage(data: data)

        let url = try ChatImagePreview.writeTempFile(img)
        // Path is keyed by the image id and is a .jpg.
        XCTAssertEqual(url, ChatImagePreview.tempFileURL(for: img.id))
        XCTAssertTrue(url.lastPathComponent.hasSuffix(".jpg"))
        // Bytes round-trip.
        XCTAssertEqual(try Data(contentsOf: url), data)

        // Repeated staging reuses the same file (no temp litter on re-click).
        let url2 = try ChatImagePreview.writeTempFile(img)
        XCTAssertEqual(url, url2)

        try? FileManager.default.removeItem(at: url)
    }

    func testTempFileURLsDifferPerImage() {
        let a = ChatImage(data: Data([0x1]))
        let b = ChatImage(data: Data([0x2]))
        XCTAssertNotEqual(ChatImagePreview.tempFileURL(for: a.id),
                          ChatImagePreview.tempFileURL(for: b.id))
    }

    // MARK: - How wide an attachment draws

    private func image(_ width: CGFloat, _ height: CGFloat) -> NSImage {
        NSImage(size: NSSize(width: width, height: height))
    }

    /// Every attachment in a message is the same height, so the width is the
    /// only thing left to carry the shape. A landscape shot takes twice the
    /// room of a portrait one, and the row still lines up.
    func testWidthFollowsTheAspectRatioAtAFixedHeight() {
        XCTAssertEqual(ChatImagePreview.displayWidth(for: image(400, 200), height: 100, maxWidth: 900), 200)
        XCTAssertEqual(ChatImagePreview.displayWidth(for: image(200, 400), height: 100, maxWidth: 900), 50)
        XCTAssertEqual(ChatImagePreview.displayWidth(for: image(300, 300), height: 100, maxWidth: 900), 100)
    }

    /// A panorama would be wider than the message it sits in and could never
    /// share a row with anything.
    func testAPanoramaIsClampedToTheMessageWidth() {
        XCTAssertEqual(ChatImagePreview.displayWidth(for: image(6000, 400), height: 200, maxWidth: 900), 900)
    }

    /// And a sliver would be unrecognisable, so it keeps a minimum presence.
    func testASliverKeepsAMinimumWidth() {
        let width = ChatImagePreview.displayWidth(for: image(4, 4000), height: 200, maxWidth: 900)
        XCTAssertGreaterThan(width, 20)
        XCTAssertLessThan(width, 200, "still visibly a tall thing")
    }

    /// A decode that produced an empty representation must not divide by zero.
    func testAnImageWithNoSizeFallsBackToSquare() {
        XCTAssertEqual(ChatImagePreview.displayWidth(for: image(0, 0), height: 200, maxWidth: 900), 200)
    }
}
