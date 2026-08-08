import XCTest
@testable import MLXCore

/// The Recommended pane compares each model's memory requirement against how
/// much this Mac can actually give a model (the GPU working-set budget), not
/// raw physical RAM. These pin the fit thresholds and the labels.
final class SystemMemoryInfoTests: XCTestCase {
    private let gib: UInt64 = 1_073_741_824

    /// 24 GB machine, ~16 GB usable (typical Metal working set).
    private var mac24: SystemMemoryInfo {
        SystemMemoryInfo(totalBytes: 24 * gib, usableBytes: 16 * gib)
    }

    func testLabelsAreWholeGB() {
        XCTAssertEqual(mac24.totalLabel, "24 GB")
        XCTAssertEqual(mac24.usableLabel, "16 GB")
    }

    func testUsableFraction() {
        XCTAssertEqual(mac24.usableFraction, 16.0 / 24.0, accuracy: 0.0001)
    }

    func testComfortableWellUnderUsable() {
        // Gemma 4 12B needs ~7.6 GB against 16 GB usable → comfortable.
        XCTAssertEqual(mac24.fit(neededGB: 7.6), .comfortable)
    }

    func testTightNearTheUsableCeiling() {
        // 14.5 GB is > 85% of 16 GB but still ≤ 16 → tight.
        XCTAssertEqual(mac24.fit(neededGB: 14.5), .tight)
    }

    func testExceedsAboveUsable() {
        // A 128 GB-class model on a 24 GB Mac → not enough memory.
        XCTAssertEqual(mac24.fit(neededGB: 20.0), .exceeds)
        XCTAssertFalse(MemoryFit.exceeds.fitsAtAll)
        XCTAssertTrue(MemoryFit.comfortable.fitsAtAll)
        XCTAssertTrue(MemoryFit.tight.fitsAtAll)
    }

    func testBoundaryExactlyUsableStillFits() {
        // Exactly the usable budget is not "exceeds" (it's the tight band).
        XCTAssertEqual(mac24.fit(neededGB: 16.0), .tight)
    }

    func testZeroUsableDefaultsToComfortable() {
        // Metal unavailable / not yet measured must never render everything red.
        let unknown = SystemMemoryInfo(totalBytes: 24 * gib, usableBytes: 0)
        XCTAssertEqual(unknown.fit(neededGB: 100), .comfortable)
    }

    func testPreciseGBKeepsOneDecimal() {
        XCTAssertEqual(SystemMemoryInfo.preciseGB(7.56), "7.6 GB")
        XCTAssertEqual(SystemMemoryInfo.wholeGB(23.8), "24 GB")
    }
}
