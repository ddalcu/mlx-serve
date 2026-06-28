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
}
