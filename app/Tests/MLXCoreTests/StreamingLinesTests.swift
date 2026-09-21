import XCTest
@testable import MLXCore

final class StreamingLinesTests: XCTestCase {
    func testAppendingTextLeavesEveryEarlierLineIdentical() {
        let before = StreamingLines.split("one\n\ntwo\nthr")
        let after = StreamingLines.split("one\n\ntwo\nthree\nfour")
        XCTAssertEqual(before, ["one", " ", "two", "thr"])
        XCTAssertEqual(Array(after.prefix(3)), Array(before.prefix(3)))
        XCTAssertEqual(after.count, 5)
    }
}
