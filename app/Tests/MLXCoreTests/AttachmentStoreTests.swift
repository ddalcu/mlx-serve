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
    func testFilenameCarriesTheIdAndTheStoredFormat() {
        let id = UUID()
        let name = AttachmentStore.filename(id: id, name: "screenshot.PNG", ext: "png")
        XCTAssertTrue(name.hasPrefix(id.uuidString + "_"))
        XCTAssertEqual(name, "\(id.uuidString)_screenshot.png")

        let other = AttachmentStore.filename(id: UUID(), name: "screenshot.PNG", ext: "png")
        XCTAssertNotEqual(name, other, "two files of the same name must not collide")
    }

    /// The name the file arrived under does not decide its extension: an
    /// attachment converted on the way in is no longer in the format its name
    /// claimed.
    func testTheNamesOwnExtensionIsReplaced() {
        let name = AttachmentStore.filename(id: UUID(), name: "IMG_1234.HEIC", ext: "jpg")
        XCTAssertTrue(name.hasSuffix("_IMG_1234.jpg"), name)
    }

    func testFilenameWithoutAnOriginalIsAPastedImage() {
        let name = AttachmentStore.filename(id: UUID(), name: nil, ext: "png")
        XCTAssertTrue(name.hasSuffix("_pasted-image.png"), name)
    }

    /// A browser drag hands over bytes with no name, so the extension comes
    /// from the bytes themselves.
    ///
    /// The four the sniff knows are exactly the four the server can decode
    /// (stb_image, then libwebp). Anything else must answer nil rather than
    /// guess: a guess of "png" is what let a HEIC be stored under a .png name
    /// and sent to a decoder that cannot read it.
    func testExtensionIsSniffedFromTheBytes() {
        XCTAssertEqual(AttachmentStore.sniffedExt(for: Data([0x89, 0x50, 0x4E, 0x47])), "png")
        XCTAssertEqual(AttachmentStore.sniffedExt(for: Data([0xFF, 0xD8, 0xFF, 0xE0])), "jpg")
        XCTAssertEqual(AttachmentStore.sniffedExt(for: Data([0x47, 0x49, 0x46, 0x38])), "gif")
        XCTAssertEqual(AttachmentStore.sniffedExt(for: Self.webpBytes), "webp")
    }

    func testUnreadableBytesAreNotGuessedAt() {
        XCTAssertNil(AttachmentStore.sniffedExt(for: Self.heicBytes), "HEIC")
        XCTAssertNil(AttachmentStore.sniffedExt(for: Data([0x49, 0x49, 0x2A, 0x00])), "TIFF")
        XCTAssertNil(AttachmentStore.sniffedExt(for: Data([0x42, 0x4D])), "BMP")
        XCTAssertNil(AttachmentStore.sniffedExt(for: Data()))
        XCTAssertNil(AttachmentStore.sniffedExt(for: Data([0x00])))
    }

    /// `RIFF????WEBP`: the four-byte size between the two tags is part of the
    /// container, so a sniff that only reads the first four bytes calls this a
    /// RIFF wave file.
    private static let webpBytes = Data([0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00,
                                         0x57, 0x45, 0x42, 0x50])

    /// An iPhone photo. `ftyp` at offset 4, brand `heic` after it.
    private static let heicBytes = Data([0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70,
                                         0x68, 0x65, 0x69, 0x63, 0x00, 0x00, 0x00, 0x00])

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

    // MARK: - Formats the server cannot read

    /// A photo: three samples per pixel and no alpha channel, the way a camera
    /// writes one. Built from an explicit representation rather than through
    /// `lockFocus`, which always hands back RGBA and would make every image
    /// here look transparent.
    private func opaqueImage() -> NSImage {
        let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 4, pixelsHigh: 4,
                                   bitsPerSample: 8, samplesPerPixel: 3, hasAlpha: false,
                                   isPlanar: false, colorSpaceName: .deviceRGB,
                                   bytesPerRow: 0, bitsPerPixel: 0)!
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        NSColor.red.drawSwatch(in: .init(x: 0, y: 0, width: 4, height: 4))
        NSGraphicsContext.restoreGraphicsState()

        let image = NSImage(size: .init(width: 4, height: 4))
        image.addRepresentation(rep)
        return image
    }

    private func transparentImage() -> NSImage {
        let image = NSImage(size: .init(width: 4, height: 4))
        image.lockFocus()
        NSColor.clear.set()
        NSBezierPath.fill(.init(x: 0, y: 0, width: 4, height: 4))
        image.unlockFocus()
        return image
    }

    /// The regression this whole section exists for.
    ///
    /// The server decodes by CONTENT, not by the data URL's label
    /// (`decodeRgbOwned`: stb_image, then libwebp), and neither reads HEIC. So
    /// an iPhone photo stored and sent as its own bytes decodes to nothing and
    /// drops out of the prompt silently: no error, no 400, just a model
    /// answering about a picture it was never shown. macOS decodes HEIC and the
    /// server never will, so the conversion belongs here.
    ///
    /// The old code re-encoded EVERY attachment to JPEG, which covered this by
    /// accident. Keeping the original bytes is the point of the change, so the
    /// bytes are kept only when they are bytes the server can actually read.
    func testAPhotoTheServerCannotDecodeIsConverted() throws {
        let pending = PendingImage(image: opaqueImage(),
                                   original: Self.heicBytes,
                                   filename: "IMG_1234.HEIC")
        let payload = try XCTUnwrap(AttachmentStore.payload(for: pending, id: UUID()))

        XCTAssertNotEqual(payload.data, Self.heicBytes, "the HEIC must not be stored")
        XCTAssertEqual(AttachmentStore.sniffedExt(for: payload.data), "jpg",
                       "a photo converts to JPEG: HEIC is already lossy, and a lossless copy of a 12 Mpx frame is tens of MB")
        XCTAssertTrue(payload.name.hasSuffix("_IMG_1234.jpg"), payload.name)
    }

    /// JPEG has no alpha channel, so a transparent source would come back with
    /// black where the transparency was.
    func testATransparentImageConvertsToPngInstead() throws {
        let pending = PendingImage(image: transparentImage(),
                                   original: Data([0x49, 0x49, 0x2A, 0x00]),
                                   filename: "logo.tiff")
        let payload = try XCTUnwrap(AttachmentStore.payload(for: pending, id: UUID()))

        XCTAssertEqual(AttachmentStore.sniffedExt(for: payload.data), "png")
        XCTAssertTrue(payload.name.hasSuffix("_logo.png"), payload.name)
    }

    /// The whitelist is what passes through, so a format the server reads is
    /// never re-encoded - that is the saving this change is for.
    func testTheFourFormatsTheServerReadsAreLeftAlone() throws {
        let cases: [(Data, String, String)] = [
            (Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]), "shot.png", "_shot.png"),
            (Data([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10]), "photo.jpg", "_photo.jpg"),
            (Data([0x47, 0x49, 0x46, 0x38, 0x39, 0x61]), "loop.gif", "_loop.gif"),
            (Self.webpBytes, "art.webp", "_art.webp"),
        ]
        for (bytes, filename, suffix) in cases {
            let pending = PendingImage(image: opaqueImage(), original: bytes, filename: filename)
            let payload = try XCTUnwrap(AttachmentStore.payload(for: pending, id: UUID()))
            XCTAssertEqual(payload.data, bytes, filename)
            XCTAssertTrue(payload.name.hasSuffix(suffix), payload.name)
        }
    }

    /// A name that disagrees with the bytes is a file that lies about itself,
    /// and the sniff is the only thing that ever reads it back.
    func testTheExtensionDescribesTheBytesThatWereStored() throws {
        let pending = PendingImage(image: opaqueImage(),
                                   original: Data([0xFF, 0xD8, 0xFF, 0xE0]),
                                   filename: "mislabelled.png")
        let payload = try XCTUnwrap(AttachmentStore.payload(for: pending, id: UUID()))
        XCTAssertTrue(payload.name.hasSuffix("_mislabelled.jpg"), payload.name)
    }

    /// With no file behind it there is no original to preserve, so the image is
    /// encoded ONCE — to PNG, because these are mostly screenshots and JPEG
    /// rings around text.
    func testAPasteWithNoFileIsEncodedToPng() throws {
        let payload = try XCTUnwrap(AttachmentStore.payload(for: PendingImage(image: opaqueImage()),
                                                            id: UUID()))
        XCTAssertEqual(AttachmentStore.sniffedExt(for: payload.data), "png",
                       "an opaque image still goes to PNG here: a paste with no file is a screenshot")
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

    // MARK: - Deleting one message

    /// Deleting a message takes its path out of the session, and the session is
    /// where `removablePaths(deleting:)` looks. So a file whose message is gone
    /// is not orphaned until the chat is deleted - it is orphaned FOREVER,
    /// because by then nothing names it.
    func testDeletingAMessageRemovesItsOwnAttachment() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        var session = ChatSession(title: "t")
        var doomed = ChatMessage(role: .user, content: "look at this")
        doomed.images = [ChatImage(data: Data([1]), path: root + "/gone.png")]
        var kept = ChatMessage(role: .user, content: "and this")
        kept.images = [ChatImage(data: Data([1]), path: root + "/kept.png")]
        session.messages = [kept]  // state AFTER the delete

        let removable = AttachmentStore.removablePaths(orphanedBy: [doomed],
                                                       in: [session],
                                                       root: root)
        XCTAssertEqual(removable, [root + "/gone.png"])
    }

    /// Same rule as deleting a whole chat: a fork copies messages, so the file
    /// may still belong to a conversation that is very much alive.
    func testAMessageDeleteSparesAFilesOtherOwner() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        let shared = root + "/shared.png"
        var doomed = ChatMessage(role: .user, content: "look")
        doomed.images = [ChatImage(data: Data([1]), path: shared)]

        var fork = ChatSession(title: "fork")
        var copied = ChatMessage(role: .user, content: "look")
        copied.images = [ChatImage(data: Data([1]), path: shared)]
        fork.messages = [copied]

        XCTAssertTrue(AttachmentStore.removablePaths(orphanedBy: [doomed], in: [fork], root: root).isEmpty)
    }

    func testAMessageDeleteNeverReachesOutsideTheRoot() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        var doomed = ChatMessage(role: .user, content: "x")
        doomed.images = [ChatImage(data: Data([1]), path: "/etc/passwd")]
        XCTAssertTrue(AttachmentStore.removablePaths(orphanedBy: [doomed], in: [], root: root).isEmpty)
    }

    // MARK: - Where the two removal paths may be called from

    private func source(_ relativePath: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// `truncateMessages` must NOT remove files, and this is the whole reason
    /// message deletion needed its own entry point rather than a hook on every
    /// removal.
    ///
    /// Regenerate and edit-and-resend both truncate the transcript and then
    /// hand the SAME `ChatImage` values, paths included, to the message they
    /// rebuild. A removal there would delete a file that is about to be used
    /// again, and the picture would vanish from a turn the user only meant to
    /// re-run.
    func testTruncationDoesNotRemoveAttachments() throws {
        let src = try source("Sources/MLXServe/AppState.swift")
        let start = try XCTUnwrap(src.range(of: "func truncateMessages("))
        let body = src[start.lowerBound...].prefix(600)
        XCTAssertFalse(body.contains("AttachmentStore"),
                       "truncateMessages must not delete: regenerate re-uses these very paths")
    }

    /// The two places a file may be removed, and nowhere else.
    func testOnlySessionAndMessageDeletionRemoveFiles() throws {
        let src = try source("Sources/MLXServe/AppState.swift")
        let callers = src.components(separatedBy: "AttachmentStore.remove(")
        XCTAssertEqual(callers.count - 1, 2,
                       "expected exactly two removal sites: deleteSessions and deleteMessage")
        XCTAssertTrue(src.contains("removablePaths(orphanedBy:"))
        XCTAssertTrue(src.contains("removablePaths(deleting:"))
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
