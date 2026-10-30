import XCTest
@testable import MLXCore

/// Pins the pure classifier that routes a pasted (or dropped) file URL to the
/// same destination as the attach button: image / PDF / audio attachments, or a
/// document folder for mini-RAG. The classification is a pure static helper so
/// the paste routing is testable without a pasteboard or a rendered view.
///
/// Assertions compare `.rawValue` strings rather than enum cases — XCTAssertEqual's
/// leading-dot generic inference is flaky for some case names in this module's
/// import set, and string comparison sidesteps it entirely.
final class ChatPasteTests: XCTestCase {

    private func kind(_ ext: String, dir: Bool = false, audio: Bool = false, video: Bool = false) -> String {
        PasteFileKind.classify(ext: ext, isDirectory: dir, audioSupported: audio, videoSupported: video).rawValue
    }

    func testDirectoryIsFolderEvenWithAFileExtension() {
        // A directory literally named "notes.pdf" is still a folder.
        XCTAssertEqual(kind("pdf", dir: true), "folder")
        XCTAssertEqual(kind("", dir: true, audio: true), "folder")
    }

    func testPDFIsClassifiedRegardlessOfCase() {
        XCTAssertEqual(kind("pdf"), "pdf")
        XCTAssertEqual(kind("PDF"), "pdf")
    }

    func testCommonImageExtensionsAreImages() {
        for ext in ["png", "jpg", "jpeg", "heic", "gif", "tiff"] {
            XCTAssertEqual(kind(ext), "image", "\(ext) should classify as an image")
        }
    }

    func testAudioIsGatedOnModelSupport() {
        XCTAssertEqual(kind("wav", audio: true), "audio")
        // Model can't hear audio → don't attach it as audio.
        XCTAssertEqual(kind("wav", audio: false), "unhandled")
    }

    func testUnknownExtensionIsUnhandled() {
        XCTAssertEqual(kind("docx", audio: true), "unhandled")
        XCTAssertEqual(kind("", audio: true), "unhandled")
    }

    func testVideoIsGatedOnModelSupport() {
        for ext in ["mov", "mp4", "m4v"] {
            XCTAssertEqual(kind(ext, video: true), "video", "\(ext) should classify as video when supported")
            // Model can't read video → don't attach it as video (and it isn't
            // an image either, so it falls through to unhandled).
            XCTAssertEqual(kind(ext, video: false), "unhandled", "\(ext) should be unhandled when unsupported")
        }
    }

    func testDirectoryBeatsVideoSupportEvenWithAMovieExtension() {
        XCTAssertEqual(kind("mov", dir: true, video: true), "folder")
    }

    // MARK: - Plain text

    func testPlainTextExtensionsAreText() {
        for ext in ["txt", "TXT", "text", "md", "markdown", "log",
                    "json", "yaml", "yml", "toml", "csv", "tsv", "xml"] {
            XCTAssertEqual(kind(ext, audio: true, video: true), "text", "\(ext) should classify as text")
        }
    }

    func testTextRoutingIgnoresModelMediaSupport() {
        // Text rides in the prompt, so it never depends on what the model can hear or see.
        XCTAssertEqual(kind("log", audio: false, video: false), "text")
        XCTAssertEqual(kind("log", audio: true, video: true), "text")
    }

    func testMarkupFormatsAreNotAttachedAsText() {
        // Tagged text, but the raw bytes are markup rather than the prose someone
        // means to attach; PDF keeps its own case because it is extracted, not read.
        for ext in ["html", "htm", "rtf"] {
            XCTAssertEqual(kind(ext), "unhandled", "\(ext) should not attach as plain text")
        }
        XCTAssertEqual(kind("pdf"), "pdf")
    }

    func testPendingDocPreambleTagFollowsKind() {
        XCTAssertEqual(PendingTextDoc(name: "a.pdf", text: "t", kind: .pdf).preambleTag, "PDF")
        XCTAssertEqual(PendingTextDoc(name: "a.log", text: "t", kind: .text).preambleTag, "Text")
    }

    // MARK: - Reading plain text off disk

    /// A fresh temp directory that is removed when the test returns.
    private func withTempDir(_ body: (URL) throws -> Void) throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("chat-paste-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try body(dir)
    }

    @MainActor
    func testPlainTextReaderReturnsFileContentVerbatim() async throws {
        try withTempDir { dir in
            let url = dir.appendingPathComponent("session.log")
            let content = "line one\nline two\n"
            try content.write(to: url, atomically: true, encoding: .utf8)
            XCTAssertEqual(ChatDetailView.extractPlainTextFile(from: url), content)
        }
    }

    @MainActor
    func testPlainTextReaderRejectsBinaryAndEmptyFiles() async throws {
        try withTempDir { dir in
            // A NUL byte means the bytes are not text, however the file is named.
            let binary = dir.appendingPathComponent("blob.txt")
            try Data([0x74, 0x00, 0x78]).write(to: binary)
            XCTAssertNil(ChatDetailView.extractPlainTextFile(from: binary))

            // Whitespace only is nothing to send.
            let blank = dir.appendingPathComponent("blank.txt")
            try "  \n\t\n".write(to: blank, atomically: true, encoding: .utf8)
            XCTAssertNil(ChatDetailView.extractPlainTextFile(from: blank))

            XCTAssertNil(ChatDetailView.extractPlainTextFile(from: dir.appendingPathComponent("missing.txt")))
        }
    }
}
