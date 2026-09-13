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

    // MARK: - The box a generated picture draws in

    /// A generated picture is capped by its HEIGHT and keeps its own ratio, the
    /// way an attachment does — but its box has to match the picture EXACTLY.
    /// The rounded corners clip the frame, so a frame taller or wider than what
    /// it holds rounds empty space and leaves the picture square.
    func testTheBoxTakesThePicturesOwnRatio() {
        let box = ChatImagePreview.displaySize(for: image(1600, 900), maxHeight: 300, maxWidth: 420)
        XCTAssertEqual(box.height / box.width, 900.0 / 1600.0, accuracy: 0.001)
    }

    func testHeightIsTheCapForAnythingThatFits() {
        let box = ChatImagePreview.displaySize(for: image(1000, 1000), maxHeight: 300, maxWidth: 420)
        XCTAssertEqual(box, CGSize(width: 300, height: 300))
    }

    /// A panorama runs out of WIDTH first, and then the height must come down
    /// with it — clamping width alone is what produces a letterbox.
    func testAPanoramaGivesUpHeightRatherThanStretch() {
        let box = ChatImagePreview.displaySize(for: image(4000, 1000), maxHeight: 300, maxWidth: 420)
        XCTAssertEqual(box.width, 420)
        XCTAssertEqual(box.height, 105, accuracy: 0.001)
    }

    func testNeitherCapIsEverExceeded() {
        for size in [(4000.0, 1000.0), (1000.0, 4000.0), (30.0, 20.0), (1.0, 1.0)] {
            let box = ChatImagePreview.displaySize(for: image(size.0, size.1),
                                                   maxHeight: 300, maxWidth: 420)
            XCTAssertLessThanOrEqual(box.width, 420)
            XCTAssertLessThanOrEqual(box.height, 300)
        }
    }

    func testASizelessImageFallsBackToASquareBox() {
        XCTAssertEqual(ChatImagePreview.displaySize(for: image(0, 0), maxHeight: 300, maxWidth: 420),
                       CGSize(width: 300, height: 300))
    }
}
