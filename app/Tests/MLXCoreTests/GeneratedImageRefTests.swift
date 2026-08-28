import XCTest
@testable import MLXCore

/// A generated image is its FILE, not a second copy of itself in the history.
///
/// The generator writes the original into `~/.mlx-serve/generations`, and the
/// message already carries that path as a `ChatMediaRef` for the caption and
/// the Reveal-in-Finder button. It also used to carry a re-encoded JPEG of the
/// same picture in `ChatMessage.images`, which the transcript drew: measured on
/// a plain 11-message conversation, that second copy was 424 KB of a 1.00 MB
/// `chat-history.json` whose text came to 29 KB.
///
/// The picture now comes from the file. Two consequences are worth pinning:
/// nothing may put those bytes back on the message, and a missing file has to
/// SAY so, because there is no longer anything else on screen when it is gone.
final class GeneratedImageRefTests: XCTestCase {

    private func source(_ relativePath: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// The turn engine builds the visible media message. It may attach the ref
    /// and nothing else: an assignment to that message's `images` is the whole
    /// regression, and it is invisible from the outside because the transcript
    /// would draw the bytes and look correct while the history doubled.
    func testTheGeneratedMediaMessageCarriesNoImageBytes() throws {
        let src = try source("Sources/MLXServe/Services/ChatTurnEngine.swift")
        XCTAssertFalse(src.contains("mediaMsg.images"),
                       "the generated-media message must carry only its ref; bytes belong in the file it points at")
        XCTAssertTrue(src.contains("mediaMsg.media = [ref]"),
                      "the ref is what the transcript draws from now")
    }

    /// `browse` is deliberately untouched: a screenshot has no file behind it,
    /// so its bytes are all there is, and they are vision input rather than a
    /// picture for the user.
    func testBrowseScreenshotsKeepTheirBytes() throws {
        let src = try source("Sources/MLXServe/Services/ChatTurnEngine.swift")
        XCTAssertTrue(src.contains("toolMsg.images = [chatImage]"),
                      "a browse screenshot has no file to draw from")
    }

    /// Before, a missing file drew nothing for an image on the grounds that the
    /// picture was still on screen. It no longer is.
    func testAMissingFileIsReportedForEveryKindIncludingImages() throws {
        let src = try source("Sources/MLXServe/Views/ChatMediaAttachmentView.swift")
        XCTAssertFalse(src.contains("if ref.kind != .image { missingRow }"),
                       "an image with no file on disk now leaves a blank space unless the row says why")
    }

    /// A history written before this change has BOTH the bytes and the ref on
    /// the same message, and each is drawn by a different row. Whichever way
    /// that is resolved, it must not be "draw them both": the same picture
    /// twice reads as a bug, and it is the one thing an existing transcript
    /// would show after upgrading.
    func testAMessageWithAnImageRefDoesNotAlsoDrawItsBytes() throws {
        let src = try source("Sources/MLXServe/Views/ChatView.swift")
        XCTAssertTrue(src.contains("message.media?.contains(where: { $0.kind == .image }) != true"),
                      "the bytes row must stand down when the message carries an image ref")
    }

    /// `exists` is the predicate the transcript branches on, so it decides
    /// picture-or-placeholder for every generated attachment.
    func testRefExistenceFollowsTheFileOnDisk() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mlx-core-genimg-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let file = dir.appendingPathComponent("fox.png")
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: file)

        let ref = ChatMediaRef(kind: .image, path: file.path, prompt: "a red fox")
        XCTAssertTrue(ref.exists)
        XCTAssertEqual(ref.filename, "fox.png")

        try FileManager.default.removeItem(at: file)
        XCTAssertFalse(ref.exists, "a deleted generation must fall back to the placeholder row")
    }
}
