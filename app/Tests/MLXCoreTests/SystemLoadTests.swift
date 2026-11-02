import XCTest
@testable import MLXCore

/// What the strip's hardware gauges say: Activity Monitor's memory figure, and how a percentage is shaded and spelled.
final class SystemLoadTests: XCTestCase {
    func testMemoryUsedIsAppPlusWiredPlusCompressedMinusPurgeable() {
        // internal 1000 - purgeable 200 + wired 300 + compressed 50, in 16 KiB pages.
        XCTAssertEqual(SystemMetrics.computeMemoryUsed(internalPages: 1000, purgeablePages: 200, wirePages: 300,
                                                       compressorPages: 50, pageSize: 16384),
                       1150 * 16384)
    }

    func testMoreReclaimablePagesThanAppPagesNeverGoesNegative() {
        XCTAssertEqual(SystemMetrics.computeMemoryUsed(internalPages: 10, purgeablePages: 40, wirePages: 5,
                                                       compressorPages: 5, pageSize: 4096),
                       10 * 4096)
    }

    func testTheMemoryLabelIsWholeGibibytesUsedOverTotal() {
        let load = SystemLoad(gpu: 0, gpuTemperature: nil, ane: nil, memoryUsed: 50 << 30 | 123, memoryTotal: 256 << 30)
        XCTAssertEqual(load.memoryLabel, "50/256G")
        XCTAssertEqual(load.memoryPercent, 50.0 / 256 * 100, accuracy: 0.001)
    }

    func testAnUnknownTotalReadsAsEmptyNotAsADivisionByZero() {
        XCTAssertEqual(SystemLoad(gpu: 0, gpuTemperature: nil, ane: nil, memoryUsed: 1, memoryTotal: 0).memoryPercent, 0)
    }

    func testABarTurnsYellowAtSixtyAndRedAtEightyFivePercent() {
        XCTAssertEqual([0, 59.9, 60, 84.9, 85, 100].map(SystemLoad.level),
                       [.normal, .normal, .busy, .busy, .saturated, .saturated])
    }

    func testABarFillsByTheClampedFraction() {
        XCTAssertEqual([-5, 0, 50, 100, 140].map(SystemLoad.fraction), [0, 0, 0.5, 1, 1])
    }
}
