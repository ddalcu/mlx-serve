import XCTest
import AppKit
@testable import MLXCore

/// A user's own attachments live in `~/.mlx-serve/attachments/` and the history
/// carries a PATH, not the picture. Measured on an ordinary 11-message
/// conversation before this change: 1.00 MB of `chat-history.json`, 97.2% of it
/// base64 attachments, 29 KB of it text.
///
/// The parts worth pinning are the ones that are invisible until they bite: a
/// filename that could escape the folder, a delete that could take a fork's
/// pictures with it, and a delete that could reach outside the folder at all.
final class AttachmentStoreTests: XCTestCase {

    private func tempRoot() throws -> String {
        let dir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mlx-core-attachments-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - Naming

    /// A separator is the dangerous character: unfiltered, `a/b.png` names a
    /// file in a directory that does not exist, and the write just fails.
    func testSanitizeKeepsNamesInsideOneComponent() {
        XCTAssertFalse(AttachmentStore.sanitize("a/b").contains("/"))
        XCTAssertFalse(AttachmentStore.sanitize("../../etc/passwd").contains("/"))
        XCTAssertFalse(AttachmentStore.sanitize("a:b").contains(":"))
        XCTAssertEqual(AttachmentStore.sanitize("holiday photo 2026"), "holiday photo 2026")
    }

    /// An empty or all-punctuation name still has to produce a filename.
    func testSanitizeNeverAnswersEmpty() {
        XCTAssertFalse(AttachmentStore.sanitize("").isEmpty)
        XCTAssertFalse(AttachmentStore.sanitize("...").isEmpty)
        XCTAssertFalse(AttachmentStore.sanitize("///").isEmpty)
    }

    /// The uuid is the `ChatImage`'s own id, so a file in the folder names the
    /// record that points at it. It also makes collisions impossible: two
    /// `screenshot.png` from two different folders would otherwise clobber.
    func testFilenameCarriesTheIdAndKeepsTheExtension() {
        let id = UUID()
        let name = AttachmentStore.filename(id: id, original: "screenshot.PNG")
        XCTAssertTrue(name.hasPrefix(id.uuidString + "_"))
        XCTAssertTrue(name.hasSuffix(".png"), name)
        XCTAssertTrue(name.contains("screenshot"), name)

        let other = AttachmentStore.filename(id: UUID(), original: "screenshot.PNG")
        XCTAssertNotEqual(name, other, "two files of the same name must not collide")
    }

    func testFilenameWithoutAnOriginalIsAPastedPng() {
        let name = AttachmentStore.filename(id: UUID(), original: nil)
        XCTAssertTrue(name.hasSuffix("_pasted-image.png"), name)
    }

    /// A browser drag hands over bytes with no name, so the extension comes
    /// from the bytes themselves.
    func testExtensionIsSniffedFromTheBytes() {
        XCTAssertEqual(AttachmentStore.ext(for: Data([0x89, 0x50, 0x4E, 0x47])), "png")
        XCTAssertEqual(AttachmentStore.ext(for: Data([0xFF, 0xD8, 0xFF, 0xE0])), "jpg")
        XCTAssertEqual(AttachmentStore.ext(for: Data([0x47, 0x49, 0x46, 0x38])), "gif")
        XCTAssertEqual(AttachmentStore.ext(for: Data([0x00])), "png", "unknown bytes fall back to png")
    }

    // MARK: - Bytes

    /// The whole point: a file the user picked is stored as it is. Re-encoding
    /// it is what the history used to do, at `compressionFactor 0.85`.
    func testAFilesOwnBytesAreStoredVerbatim() throws {
        let original = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x11, 0x22])
        let pending = PendingImage(image: NSImage(size: .init(width: 1, height: 1)),
                                   original: original,
                                   filename: "cat.png")
        let payload = try XCTUnwrap(AttachmentStore.payload(for: pending, id: UUID()))
        XCTAssertEqual(payload.data, original)
        XCTAssertTrue(payload.name.hasSuffix("_cat.png"), payload.name)
    }

    /// A paste can hand over PNG bytes with no name at all. The bytes are still
    /// kept verbatim, and the name is the one an encoded paste would get: from
    /// the folder the two are the same thing.
    func testAPasteThatCarriedBytesButNoNameIsStillAPastedImage() throws {
        let png = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        let pending = PendingImage(image: NSImage(size: .init(width: 1, height: 1)), original: png)
        let payload = try XCTUnwrap(AttachmentStore.payload(for: pending, id: UUID()))
        XCTAssertEqual(payload.data, png)
        XCTAssertTrue(payload.name.hasSuffix("_pasted-image.png"), payload.name)
    }

    /// A drag that hands over a URL carries the name IN it, and the provider's
    /// `suggestedName` is nil for exactly those providers: asking it first is
    /// how `01b At the gate.png` was stored as `pasted-image.png`.
    func testADroppedFileKeepsItsOwnName() {
        XCTAssertEqual(
            AttachmentStore.droppedName(url: URL(fileURLWithPath: "/tmp/01b At the gate.png"),
                                        suggested: nil),
            "01b At the gate.png")
        XCTAssertEqual(AttachmentStore.droppedName(url: nil, suggested: "from-provider.jpg"),
                       "from-provider.jpg")
        XCTAssertNil(AttachmentStore.droppedName(url: nil, suggested: nil))
    }

    /// A drag out of a browser page can hand over a name with no extension.
    /// Defaulting to `png` there mislabels every JPEG, so the bytes decide.
    func testANameWithNoExtensionTakesOneFromTheBytes() throws {
        let jpeg = Data([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10])
        let pending = PendingImage(image: NSImage(size: .init(width: 1, height: 1)),
                                   original: jpeg,
                                   filename: "photo")
        let payload = try XCTUnwrap(AttachmentStore.payload(for: pending, id: UUID()))
        XCTAssertTrue(payload.name.hasSuffix("_photo.jpg"), payload.name)
    }

    /// With no file behind it there is no original to preserve, so the image is
    /// encoded ONCE — to PNG, because these are mostly screenshots and JPEG
    /// rings around text.
    func testAPasteWithNoFileIsEncodedToPng() throws {
        let image = NSImage(size: .init(width: 4, height: 4))
        image.lockFocus()
        NSColor.red.drawSwatch(in: .init(x: 0, y: 0, width: 4, height: 4))
        image.unlockFocus()

        let payload = try XCTUnwrap(AttachmentStore.payload(for: PendingImage(image: image), id: UUID()))
        XCTAssertEqual(AttachmentStore.ext(for: payload.data), "png")
        XCTAssertTrue(payload.name.hasSuffix("_pasted-image.png"), payload.name)
    }

    // MARK: - Disk

    func testWriteAnswersAPathThatHoldsTheBytes() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        let bytes = Data([1, 2, 3, 4])
        let path = try XCTUnwrap(AttachmentStore.write(bytes, named: "x_y.png", in: root))
        XCTAssertEqual(FileManager.default.contents(atPath: path), bytes)
        XCTAssertTrue(AttachmentStore.isInsideRoot(path, root: root))
    }

    /// The delete path reads `ChatImage.path` out of saved history, so it is
    /// data, not a constant. Nothing outside the folder is ever removed.
    func testOnlyPathsInsideTheRootAreOurs() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        XCTAssertTrue(AttachmentStore.isInsideRoot(root + "/a.png", root: root))
        XCTAssertFalse(AttachmentStore.isInsideRoot("/etc/passwd", root: root))
        XCTAssertFalse(AttachmentStore.isInsideRoot(root, root: root))
        XCTAssertFalse(AttachmentStore.isInsideRoot(root + "/../escape.png", root: root),
                       "a traversal must not resolve back inside")
    }

    // MARK: - Deleting a conversation

    private func session(_ paths: [String]) -> ChatSession {
        var s = ChatSession(title: "t")
        var m = ChatMessage(role: .user, content: "hi")
        m.images = paths.map { ChatImage(data: Data([1]), path: $0) }
        s.messages = [m]
        return s
    }

    func testDeletingAConversationRemovesItsOwnAttachments() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        let doomed = self.session([root + "/a.png"])
        let other = self.session([root + "/b.png"])

        let removable = AttachmentStore.removablePaths(deleting: [doomed.id],
                                                       in: [doomed, other],
                                                       root: root)
        XCTAssertEqual(removable, [root + "/a.png"])
    }

    /// `ChatFork` copies the messages wholesale, so a fork and its source name
    /// the SAME file. Deleting one must not take the picture out of the other:
    /// that is data loss a user would only discover much later.
    func testAForksAttachmentSurvivesDeletingTheOriginal() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        let shared = root + "/shared.png"
        let source = self.session([shared, root + "/only-here.png"])
        let fork = self.session([shared])

        let removable = AttachmentStore.removablePaths(deleting: [source.id],
                                                       in: [source, fork],
                                                       root: root)
        XCTAssertEqual(removable, [root + "/only-here.png"])
        XCTAssertFalse(removable.contains(shared), "the fork still draws this one")
    }

    func testADeleteNeverReachesOutsideTheRoot() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        let doomed = self.session(["/etc/passwd", root + "/a.png"])
        let removable = AttachmentStore.removablePaths(deleting: [doomed.id],
                                                       in: [doomed],
                                                       root: root)
        XCTAssertEqual(removable, [root + "/a.png"])
    }

    // MARK: - The history

    /// The bytes stop being written. A record that still has them (a history
    /// from before this change) must DECODE, or `loadChatHistory`'s `?? []`
    /// turns one unreadable image into an empty history — every conversation,
    /// not one picture.
    func testAHistoryWrittenBeforeAttachmentsStillDecodes() throws {
        let legacy = """
        {"id":"\(UUID().uuidString)","data":"AQID"}
        """.data(using: .utf8)!

        let image = try JSONDecoder().decode(ChatImage.self, from: legacy)
        XCTAssertNil(image.path)
        XCTAssertTrue(image.data.isEmpty, "the picture is gone, and the transcript says so")
    }

    func testEncodingCarriesThePathAndNotTheBytes() throws {
        let image = ChatImage(data: Data([1, 2, 3]), path: "/tmp/x.png")
        let json = try JSONSerialization.jsonObject(with: try JSONEncoder().encode(image)) as? [String: Any]
        XCTAssertEqual(json?["path"] as? String, "/tmp/x.png")
        XCTAssertNil(json?["data"], "bytes belong in the file, not the history")
    }

    /// An attachment we encoded ourselves is PNG, and labelling PNG bytes as
    /// JPEG is the kind of lie that works until something reads the label.
    func testBase64URLNamesTheFormatItActuallyHas() {
        let png = ChatImage(data: Data([0x89, 0x50, 0x4E, 0x47, 0x0D]))
        XCTAssertTrue(png.base64URL.hasPrefix("data:image/png;base64,"))

        let jpeg = ChatImage(data: Data([0xFF, 0xD8, 0xFF, 0xE0]))
        XCTAssertTrue(jpeg.base64URL.hasPrefix("data:image/jpeg;base64,"))
    }

    /// A byte-less image must not become an `image_url` block with an empty
    /// payload: that tells the model there is a picture and hands it nothing.
    func testAnAttachmentWithNoBytesIsNotSent() {
        let blocks = MultimodalContent.build(
            text: "what is this",
            images: [ChatImage(data: Data(), path: "/gone.png"), ChatImage(data: Data([1, 2, 3]))],
            serverPreprocess: true)
        XCTAssertEqual(blocks.filter { $0["type"] as? String == "image_url" }.count, 1)
    }
}
