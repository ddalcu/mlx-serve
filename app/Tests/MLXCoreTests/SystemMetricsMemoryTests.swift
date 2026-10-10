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
        // free_count already contains speculative, and it is file-backed inside external.
        XCTAssertEqual(
            SystemMetrics.computeAvailableForModel(totalBytes: 128 * gib, freePages: 8 * ppg,
                speculativePages: 2 * ppg, externalPages: 26 * ppg, wirePages: 83 * ppg,
                compressorPages: 1 * ppg, pageSize: page),
            32 * gib)
    }

    func testWiredHeadroomCapsTheSum() {
        let page: UInt64 = 16384
        let ppg = gib / page
        // wired pages (GPU, shared cache) hide inside the file-backed class: the cap binds.
        XCTAssertEqual(
            SystemMetrics.computeAvailableForModel(totalBytes: 16 * gib, freePages: 6 * ppg,
                speculativePages: 0, externalPages: 6 * ppg, wirePages: 9 * ppg,
                compressorPages: 1 * ppg, pageSize: page),
            6 * gib)
    }

    func testHeavyAnonResidentCollapsesTheSum() {
        let page: UInt64 = 16384
        let ppg = gib / page
        // #45: a resident 7 GB anon model consumed the free/external pages — only 3 GB of the reclaimable classes remain.
        XCTAssertEqual(
            SystemMetrics.computeAvailableForModel(totalBytes: 16 * gib, freePages: 2 * ppg,
                speculativePages: 0, externalPages: 1 * ppg, wirePages: 3 * ppg,
                compressorPages: 1 * ppg, pageSize: page),
            3 * gib)
    }

    func testDegenerateQueriesReturnZero() {
        let page: UInt64 = 16384
        let ppg = gib / page
        // failed query (total 0) → 0; the meter's never-blocks contract.
        XCTAssertEqual(SystemMetrics.computeAvailableForModel(totalBytes: 0, freePages: 1,
            speculativePages: 1, externalPages: 1, wirePages: 1, compressorPages: 1, pageSize: page), 0)
        // no headroom: wired + compressor filling the box → 0.
        XCTAssertEqual(SystemMetrics.computeAvailableForModel(totalBytes: 16 * gib, freePages: 6 * ppg,
            speculativePages: 0, externalPages: 6 * ppg, wirePages: 16 * ppg, compressorPages: 0, pageSize: page), 0)
    }

    func testSpeculativeNeverUnderflows() {
        let page: UInt64 = 16384
        let ppg = gib / page
        // speculative > free must not wrap; the external side carries the sum.
        XCTAssertEqual(
            SystemMetrics.computeAvailableForModel(totalBytes: 16 * gib, freePages: 1 * ppg,
                speculativePages: 5 * ppg, externalPages: 4 * ppg, wirePages: 0,
                compressorPages: 0, pageSize: page),
            4 * gib)
    }
}
