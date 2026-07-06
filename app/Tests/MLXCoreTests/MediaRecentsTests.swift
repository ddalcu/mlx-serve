import XCTest
@testable import MLXCore

/// Shared recent-generations list semantics for the Voice/Music history
/// shelves: newest-first, deduped, UNCAPPED (every generation stays listed;
/// disk is the only limit), and the disk scan that rebuilds it across
/// launches.
final class MediaRecentsTests: XCTestCase {

    func testInsertingPutsNewestFirstWithoutCap() {
        var list: [String] = []
        for i in 0..<50 {
            list = MediaRecents.inserting("/out/track-\(i).wav", into: list)
        }
        XCTAssertEqual(list.count, 50, "history must not be capped — every generation stays")
        XCTAssertEqual(list.first, "/out/track-49.wav")
        XCTAssertEqual(list.last, "/out/track-0.wav")
    }

    func testInsertingDedupesToFront() {
        var list = ["/out/b.wav", "/out/a.wav"]
        list = MediaRecents.inserting("/out/a.wav", into: list)
        XCTAssertEqual(list, ["/out/a.wav", "/out/b.wav"])
    }

    func testScanFindsDayBucketedFilesNewestFirstAndFiltersSuffix() throws {
        let fm = FileManager.default
        let root = NSTemporaryDirectory() + "media-recents-test-\(UUID().uuidString)"
        defer { try? fm.removeItem(atPath: root) }
        for (day, name, age) in [("2026-07-04", "old.wav", 100.0),
                                 ("2026-07-05", "new.wav", 10.0),
                                 ("2026-07-05", "ignored.txt", 5.0)] {
            let dir = (root as NSString).appendingPathComponent(day)
            try fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
            let path = (dir as NSString).appendingPathComponent(name)
            fm.createFile(atPath: path, contents: Data([1]))
            try fm.setAttributes([.modificationDate: Date(timeIntervalSinceNow: -age)], ofItemAtPath: path)
        }
        let found = MediaRecents.scan(root: root, suffix: ".wav")
        XCTAssertEqual(found.count, 2, "non-.wav files must be filtered")
        XCTAssertTrue(found[0].hasSuffix("new.wav"), "newest first: \(found)")
        XCTAssertTrue(found[1].hasSuffix("old.wav"))
    }

    func testScanOfMissingRootIsEmpty() {
        XCTAssertEqual(MediaRecents.scan(root: "/nonexistent/nope", suffix: ".wav"), [])
    }
}
