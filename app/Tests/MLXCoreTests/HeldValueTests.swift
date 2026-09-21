import XCTest
@testable import MLXCore

final class HeldValueTests: XCTestCase {
    func testReadsOnceInsideTheHoldAndAgainAfterIt() {
        var clock = Date(timeIntervalSince1970: 0)
        var reads = 0
        let held = HeldValue(hold: 5, now: { clock }) { () -> Int in reads += 1; return reads }

        XCTAssertEqual(held.value, 1)
        clock += 4.9
        XCTAssertEqual(held.value, 1)
        XCTAssertEqual(reads, 1)

        clock += 0.2
        XCTAssertEqual(held.value, 2)
        XCTAssertEqual(reads, 2)
    }
}
