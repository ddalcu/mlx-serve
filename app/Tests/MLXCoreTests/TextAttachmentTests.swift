import XCTest
@testable import MLXCore

/// A text or source file attached to a chat rides the user message as a fenced block under its name (#841).
final class TextAttachmentTests: XCTestCase {
    private var dir: URL!

    override func setUp() {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("TextAttachmentTests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: dir)
    }

    private func file(_ name: String, _ data: Data) -> URL {
        let url = dir.appendingPathComponent(name)
        try? data.write(to: url)
        return url
    }

    func testTextAndSourceExtensionsAreTextAndMarkupOrBinaryOnesAreNot() {
        for ext in ["txt", "md", "log", "json", "csv", "yaml", "swift", "zig", "py", "ts", "rs", "c", "h", "sh", "html", "TXT"] {
            XCTAssertTrue(TextAttachment.isTextFile(ext: ext), "\(ext) should attach as text")
        }
        for ext in ["pdf", "rtf", "png", "wav", "zip", "docx", ""] {
            XCTAssertFalse(TextAttachment.isTextFile(ext: ext), "\(ext) should not attach as text")
        }
    }

    func testAPasteOrDropOfATextFileIsRoutedAsText() {
        XCTAssertEqual(PasteFileKind.classify(ext: "txt", isDirectory: false, audioSupported: false).rawValue, "text")
        XCTAssertEqual(PasteFileKind.classify(ext: "zig", isDirectory: false, audioSupported: true).rawValue, "text")
        XCTAssertEqual(PasteFileKind.classify(ext: "pdf", isDirectory: false, audioSupported: false).rawValue, "pdf")
        XCTAssertEqual(PasteFileKind.classify(ext: "txt", isDirectory: true, audioSupported: false).rawValue, "folder")
    }

    func testAFileIsReadAsUTF8OrUTF16AndRefusedWhenEmptyBinaryOrTooLarge() {
        XCTAssertEqual(TextAttachment.read(file("a.txt", Data("héllo\nworld\n".utf8))), "héllo\nworld\n")
        XCTAssertEqual(TextAttachment.read(file("b.txt", "wide ✓".data(using: .utf16)!)), "wide ✓")
        XCTAssertNil(TextAttachment.read(file("empty.txt", Data())))
        XCTAssertNil(TextAttachment.read(file("blank.txt", Data("  \n\n".utf8))))
        XCTAssertNil(TextAttachment.read(file("bin.txt", Data([0x50, 0x4B, 0x03, 0x04, 0x00, 0x00, 0xFF, 0xFE, 0x00]))))
        XCTAssertNil(TextAttachment.read(file("big.log", Data(repeating: 0x61, count: TextAttachment.maxBytes + 1))))
        XCTAssertNotNil(TextAttachment.read(file("fits.log", Data(repeating: 0x61, count: TextAttachment.maxBytes))))
        XCTAssertNil(TextAttachment.read(dir.appendingPathComponent("missing.txt")))
    }

    func testTheBlockNamesTheFileAndItsFenceOutlastsAnyFenceInside() {
        XCTAssertEqual(TextAttachment.block(name: "notes.txt", text: "one\ntwo\n"), "[File: notes.txt]\n```\none\ntwo\n```")
        // A file that holds a fence of its own gets a longer one, so the block cannot close early.
        let md = "see:\n```swift\nlet x = 1\n```\nand ````four````"
        let block = TextAttachment.block(name: "a.md", text: md)
        XCTAssertTrue(block.hasPrefix("[File: a.md]\n`````\n"))
        XCTAssertTrue(block.hasSuffix("\n`````"))
        XCTAssertTrue(block.contains(md))
    }

    func testPendingDocumentsJoinIntoOnePreambleByKind() {
        let docs = [PendingTextDoc(name: "spec.pdf", text: "page one", kind: .pdf),
                    PendingTextDoc(name: "main.zig", text: "const x = 1;", kind: .text)]
        XCTAssertEqual(PendingTextDoc.preamble(docs), "[PDF: spec.pdf]\npage one\n\n[File: main.zig]\n```\nconst x = 1;\n```")
        XCTAssertEqual(PendingTextDoc.preamble([]), "")
    }
}
