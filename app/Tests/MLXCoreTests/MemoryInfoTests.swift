import XCTest
@testable import MLXCore

/// Pins the decode of the server's `/props` → `memory` block into `MemoryInfo`,
/// including the `available_bytes` (reclaimable-available RAM) field that backs
/// the tray's "Available RAM" line. The decode is its own pure function so it's
/// testable without a live server (`fetchProps` just hands it `json["memory"]`).
final class MemoryInfoTests: XCTestCase {

    func testParseDecodesAvailableRamFromAvailableBytes() {
        let mem: [String: Any] = [
            "active_bytes": Int64(6_900_000_000),
            "peak_bytes": Int64(7_100_000_000),
            "available_bytes": Int64(8_300_000_000),
            "max_safe_context": 16384,
        ]
        let info = MemoryInfo.parse(mem)
        XCTAssertEqual(info.availableBytes, 8_300_000_000)
        XCTAssertEqual(info.activeBytes, 6_900_000_000)
        XCTAssertEqual(info.peakBytes, 7_100_000_000)
        XCTAssertEqual(info.maxSafeContext, 16384)
    }

    /// An older bundled server that predates `available_bytes` must decode to 0
    /// so the tray hides the line rather than rendering a bogus "0 MB".
    func testParseDefaultsAvailableRamToZeroWhenMissing() {
        let mem: [String: Any] = [
            "active_bytes": Int64(100),
            "peak_bytes": Int64(200),
            "max_safe_context": 4096,
        ]
        XCTAssertEqual(MemoryInfo.parse(mem).availableBytes, 0)
    }

    // MARK: - MLX buffer pool (issue #110)

    /// The panel showed 19.6 GB while the process held 81.4 GB. The missing
    /// 61 GB was MLX's reclaimable buffer pool, which nothing the server served
    /// reported — so the tray now reads it too.
    func testParseDecodesTheMlxBufferPoolFromCacheBytes() {
        let mem: [String: Any] = [
            "active_bytes": Int64(19_600_000_000),
            "peak_bytes": Int64(20_000_000_000),
            "available_bytes": Int64(8_300_000_000),
            "max_safe_context": 104_000,
            "cache_bytes": Int64(61_000_000_000),
        ]
        let info = MemoryInfo.parse(mem)
        XCTAssertEqual(info.cacheBytes, 61_000_000_000)
        XCTAssertTrue(info.gpuMemoryLabel.contains("cache"),
                      "the gap the reporter screenshotted must be named on the row")
    }

    /// An older bundled server that predates `cache_bytes` decodes to 0, and the
    /// row falls back to exactly what it rendered before.
    func testParseDefaultsCacheBytesToZeroWhenMissing() {
        let mem: [String: Any] = [
            "active_bytes": Int64(100),
            "peak_bytes": Int64(200),
            "max_safe_context": 4096,
        ]
        let info = MemoryInfo.parse(mem)
        XCTAssertEqual(info.cacheBytes, 0)
        XCTAssertEqual(info.gpuMemoryLabel, info.activeFormatted)
    }

    /// A pool doing its job is not news — the suffix appears only once the pool
    /// is large enough to explain a footprint the user would notice.
    func testGpuMemoryLabelHidesASmallHealthyPool() {
        let small = MemoryInfo(activeBytes: 7_000_000_000, peakBytes: 7_000_000_000,
                               availableBytes: 0, maxSafeContext: 0,
                               cacheBytes: 400 * 1024 * 1024)
        XCTAssertEqual(small.gpuMemoryLabel, small.activeFormatted)
    }

    func testAvailableFormattedUsesSharedFormatter() {
        let info = MemoryInfo(activeBytes: 0, peakBytes: 0,
                              availableBytes: 8_589_934_592, maxSafeContext: 0)  // 8 GiB
        XCTAssertEqual(info.availableFormatted, "8.0 GB")
    }

    // MARK: - Progress-bar fractions (relative to total physical RAM)

    /// Regression: the old GPU bar's `total` was `max(peak,active)*2`, so once
    /// the model settled and `active == peak` the fill was `peak/(peak*2)` =
    /// exactly 0.5 forever, no matter the model. Against total RAM it reflects
    /// real usage.
    func testGpuFractionUsesTotalRamNotPeakTimesTwo() {
        let total: Int64 = 16 * 1024 * 1024 * 1024
        let info = MemoryInfo(activeBytes: 7_000_000_000, peakBytes: 7_000_000_000,
                              availableBytes: 5_000_000_000, maxSafeContext: 0)
        let f = info.gpuFraction(ofTotal: total)
        XCTAssertEqual(f, 7_000_000_000.0 / Double(total), accuracy: 0.001)  // ~0.41
        XCTAssertLessThan(f, 0.45, "must reflect real usage, not be pinned at 0.5")
    }

    func testAvailableFractionReflectsAvailableBytes() {
        let total: Int64 = 16 * 1024 * 1024 * 1024
        let info = MemoryInfo(activeBytes: 0, peakBytes: 0,
                              availableBytes: 8 * 1024 * 1024 * 1024, maxSafeContext: 0)
        XCTAssertEqual(info.availableFraction(ofTotal: total), 0.5, accuracy: 0.001)
    }

    func testFractionsGuardZeroTotalAndClamp() {
        let info = MemoryInfo(activeBytes: 99, peakBytes: 0, availableBytes: 99, maxSafeContext: 0)
        XCTAssertEqual(info.gpuFraction(ofTotal: 0), 0)            // no divide-by-zero
        XCTAssertEqual(info.availableFraction(ofTotal: 0), 0)
        // A part larger than total clamps to a full bar rather than overflowing.
        let big = MemoryInfo(activeBytes: 100, peakBytes: 0, availableBytes: 0, maxSafeContext: 0)
        XCTAssertEqual(big.gpuFraction(ofTotal: 10), 1.0)
    }
}

/// The meter's free RAM splits at the GPU working-set limit (`iogpu.wired_limit_mb`).
final class MemoryMeterSplitTests: XCTestCase {
    private let gb: Int64 = 1 << 30

    func testFreeRamPastTheGpuLimitIsItsOwnSegment() {
        // 128 GB Mac, limit lowered to 64 GB, 20 GB model resident, 80 GB free RAM.
        let s = MemoryMeter.Split(gpu: 20 * gb, available: 80 * gb, total: 128 * gb, gpuLimit: 64 * gb)
        XCTAssertEqual(s.gpuFree, 44 * gb)
        XCTAssertEqual(s.ramOnlyFree, 36 * gb)
        XCTAssertEqual(s.other, 28 * gb)
    }

    func testWhenRamIsTighterThanTheLimitAllFreeIsGpuFree() {
        let s = MemoryMeter.Split(gpu: 20 * gb, available: 30 * gb, total: 128 * gb, gpuLimit: 118 * gb)
        XCTAssertEqual(s.gpuFree, 30 * gb)
        XCTAssertEqual(s.ramOnlyFree, 0)
    }

    func testUnknownOrExceededLimitNeverGoesNegative() {
        XCTAssertEqual(MemoryMeter.Split(gpu: 0, available: 50 * gb, total: 128 * gb, gpuLimit: nil).gpuFree, 50 * gb)
        let over = MemoryMeter.Split(gpu: 70 * gb, available: 40 * gb, total: 128 * gb, gpuLimit: 64 * gb)
        XCTAssertEqual(over.gpuFree, 0)
        XCTAssertEqual(over.ramOnlyFree, 40 * gb)
    }
}

/// The GPU footprint splits into loaded weights, the prefix KV cache and the rest.
final class GpuBreakdownTests: XCTestCase {
    private let gb: Int64 = 1 << 30

    func testActiveSplitsIntoModelCacheAndWorking() throws {
        let m = MemoryInfo.parse(["active_bytes": 30 * gb, "available_bytes": 0,
                                  "weights_bytes": 20 * gb, "kv_cache_bytes": 6 * gb])
        let b = try XCTUnwrap(m.gpuBreakdown)
        XCTAssertEqual([b.model, b.kvCache, b.working], [20 * gb, 6 * gb, 4 * gb])
    }

    func testEstimatesNeverExceedWhatIsActive() throws {
        // The weights figure is the on-disk size; it can read above MLX's own counter.
        let m = MemoryInfo.parse(["active_bytes": 18 * gb, "weights_bytes": 20 * gb, "kv_cache_bytes": 6 * gb])
        let b = try XCTUnwrap(m.gpuBreakdown)
        XCTAssertEqual([b.model, b.kvCache, b.working], [18 * gb, 0, 0])
    }

    func testAnOlderServerHasNoBreakdown() {
        XCTAssertNil(MemoryInfo.parse(["active_bytes": 18 * gb]).gpuBreakdown)
    }
}
