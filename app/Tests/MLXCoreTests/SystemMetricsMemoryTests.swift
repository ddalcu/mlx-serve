import XCTest
@testable import MLXCore

/// The client-side "available for a model" number must match the server's
/// pre-flight arithmetic (`status.zig computeAvailableBytes`) so the memory
/// meter reads the same with or without a running server.
final class SystemMetricsMemoryTests: XCTestCase {
    private let gib: UInt64 = 1_073_741_824

    func testMirrorsServerFormula() {
        let page: UInt64 = 16384
        let ppg = gib / page
        // 16 GB: 3 wired + 1 compressed + 2 anon, no purgeable → 10 available.
        XCTAssertEqual(
            SystemMetrics.computeAvailableForModel(totalBytes: 16 * gib, wirePages: 3 * ppg,
                compressorPages: 1 * ppg, internalPages: 2 * ppg, purgeablePages: 0, pageSize: page),
            10 * gib)
    }

    func testPurgeableIsReclaimed() {
        let page: UInt64 = 16384
        let ppg = gib / page
        // 2 GB of the 9 GB anon set is purgeable → available 5 GB, not 3 GB.
        XCTAssertEqual(
            SystemMetrics.computeAvailableForModel(totalBytes: 16 * gib, wirePages: 3 * ppg,
                compressorPages: 1 * ppg, internalPages: 9 * ppg, purgeablePages: 2 * ppg, pageSize: page),
            5 * gib)
    }

    func testDegenerateQueriesReturnZero() {
        let page: UInt64 = 16384
        let ppg = gib / page
        XCTAssertEqual(SystemMetrics.computeAvailableForModel(totalBytes: 0, wirePages: 1,
            compressorPages: 1, internalPages: 1, purgeablePages: 0, pageSize: page), 0)
        // used ≥ total
        XCTAssertEqual(SystemMetrics.computeAvailableForModel(totalBytes: 8 * gib, wirePages: 4 * ppg,
            compressorPages: 0, internalPages: 5 * ppg, purgeablePages: 0, pageSize: page), 0)
    }

    func testPurgeableNeverUnderflows() {
        let page: UInt64 = 16384
        let ppg = gib / page
        // purgeable > internal must not wrap; treated as full internal reclaimed.
        XCTAssertEqual(
            SystemMetrics.computeAvailableForModel(totalBytes: 16 * gib, wirePages: 3 * ppg,
                compressorPages: 1 * ppg, internalPages: 1 * ppg, purgeablePages: 5 * ppg, pageSize: page),
            12 * gib)
    }
}
