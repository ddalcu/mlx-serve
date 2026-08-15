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
}
