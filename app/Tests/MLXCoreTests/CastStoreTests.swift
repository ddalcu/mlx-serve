import AppKit
import XCTest
@testable import MLXCore

/// The cast library: members survive a relaunch, photos land as JPEGs H3 can
/// read, and an edit or a removal leaves no orphaned photo behind.
@MainActor
final class CastStoreTests: XCTestCase {
    private var dir: URL!

    override func setUp() {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("cast-\(UUID().uuidString)")
    }

    override func tearDown() { try? FileManager.default.removeItem(at: dir) }

    private func png(_ width: Int, _ height: Int) throws -> URL {
        let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height, bitsPerSample: 8,
                                   samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                                   bytesPerRow: 0, bitsPerPixel: 0)!
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).png")
        try rep.representation(using: .png, properties: [:])!.write(to: url)
        return url
    }

    private func photos() throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { $0.hasSuffix(".jpg") }
    }

    func testAMemberSurvivesARelaunchWithAJPEGNoLargerThanTheReferenceCap() throws {
        let store = CastStore(rootDir: dir)
        let m = try store.add(CastMember(name: "Max", kind: .dog, voice: .goofy, photo: ""), photo: try png(3000, 2000))
        let reopened = CastStore(rootDir: dir)
        XCTAssertEqual(reopened.members.map(\.name), ["Max"])
        let data = try Data(contentsOf: reopened.photoURL(m))
        XCTAssertEqual(Array(data.prefix(2)), [0xFF, 0xD8], "stored as JPEG")
        let rep = try XCTUnwrap(NSBitmapImageRep(data: data))
        XCTAssertEqual(max(rep.pixelsWide, rep.pixelsHigh), 1536)
    }

    func testANewPhotoReplacesTheOldFileAndRemovalDeletesIt() throws {
        let store = CastStore(rootDir: dir)
        var m = try store.add(CastMember(name: "Luna", kind: .cat, voice: .sassy, photo: ""), photo: try png(64, 64))
        m.name = "Luna B"
        try store.update(m, photo: try png(80, 80))
        XCTAssertEqual(try photos().count, 1, "the replaced photo was left behind")
        XCTAssertEqual(store.members.first?.name, "Luna B")
        store.delete(m.id)
        XCTAssertEqual(try photos(), [])
        XCTAssertTrue(CastStore(rootDir: dir).members.isEmpty)
    }

    func testAFileThatIsNotAPhotoIsRefusedByName() throws {
        let store = CastStore(rootDir: dir)
        let notes = FileManager.default.temporaryDirectory.appendingPathComponent("notes-\(UUID().uuidString).txt")
        try Data("hello".utf8).write(to: notes)
        XCTAssertThrowsError(try store.add(CastMember(name: "X", kind: .pet, voice: .calm, photo: ""), photo: notes)) {
            XCTAssertTrue(($0 as? LocalizedError)?.errorDescription?.contains(notes.lastPathComponent) == true)
        }
        XCTAssertTrue(store.members.isEmpty)
    }
}
