import XCTest
@testable import MLXCore

/// The bounded, cursor-based output ring backing every managed background
/// process. Same tail-trim discipline as ServerManager's log buffer, plus
/// BashOutput-style incremental reads.
final class ProcessOutputBufferTests: XCTestCase {

    func testAppendThenSnapshotReturnsAll() {
        let buf = ProcessOutputBuffer()
        buf.append("hello ")
        buf.append("world")
        XCTAssertEqual(buf.snapshot(), "hello world")
    }

    func testReadNewReturnsOnlyNewSinceLastRead() {
        let buf = ProcessOutputBuffer()
        buf.append("first")
        XCTAssertEqual(buf.readNew(), "first")
        // Nothing appended since → empty.
        XCTAssertEqual(buf.readNew(), "")
        buf.append("second")
        XCTAssertEqual(buf.readNew(), "second")
    }

    func testBoundTrimsTail() {
        let buf = ProcessOutputBuffer(maxBytes: 10)
        buf.append(String(repeating: "a", count: 8))
        buf.append(String(repeating: "b", count: 8))
        let snap = buf.snapshot()
        XCTAssertEqual(snap.count, 10, "ring must cap at maxBytes")
        XCTAssertTrue(snap.hasSuffix("bbbbb"), "keeps the most recent bytes: \(snap)")
        XCTAssertFalse(snap.contains("aaaaaaaa"), "oldest bytes are trimmed: \(snap)")
    }

    func testReadNewReportsDroppedBytesAfterTrim() {
        let buf = ProcessOutputBuffer(maxBytes: 10)
        buf.append(String(repeating: "x", count: 25)) // 15 chars trimmed off the front
        let out = buf.readNew()
        XCTAssertTrue(out.contains("15 bytes dropped"), "drop note missing: \(out)")
        // The retained tail still rides along after the note.
        XCTAssertTrue(out.hasSuffix(String(repeating: "x", count: 10)), out)
    }

    func testReadNewNoDropNoteWhenNothingTrimmed() {
        let buf = ProcessOutputBuffer(maxBytes: 100)
        buf.append("clean")
        let out = buf.readNew()
        XCTAssertEqual(out, "clean")
        XCTAssertFalse(out.contains("dropped"), out)
    }

    func testEmptyAppendIsNoOp() {
        let buf = ProcessOutputBuffer()
        buf.append("")
        buf.append(Data())
        XCTAssertEqual(buf.snapshot(), "")
        XCTAssertEqual(buf.readNew(), "")
    }
}
