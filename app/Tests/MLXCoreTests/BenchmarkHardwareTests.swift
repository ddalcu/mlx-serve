import XCTest
@testable import MLXCore

/// Hardware identity for benchmark rows.
///
/// The chip string is the primary filter axis on the board ("what will I get
/// on MY Mac?"), so it has to split into family and tier reliably — those are
/// two separate filters, and an M4 Max is a different machine from an M4 in a
/// way that matters far more than the RAM figure next to it.
///
/// GPU core count is here because it's the big WITHIN-tier differentiator: a
/// 32-core M4 Max and a 40-core M4 Max share a chip string and do not share a
/// decode speed. Most public benchmark boards miss this and their M4 Max rows
/// are quietly bimodal.
final class BenchmarkHardwareTests: XCTestCase {

    // MARK: - Chip parsing

    func testBaseChipHasAFamilyAndNoTier() {
        // This dev machine reports exactly "Apple M4".
        XCTAssertEqual(BenchmarkHardware.chipFamily("Apple M4"), "M4")
        XCTAssertEqual(BenchmarkHardware.chipTier("Apple M4"), "")
    }

    func testTieredChipsSplitIntoFamilyAndTier() {
        for (brand, family, tier) in [
            ("Apple M4 Max", "M4", "Max"),
            ("Apple M3 Ultra", "M3", "Ultra"),
            ("Apple M2 Pro", "M2", "Pro"),
            ("Apple M1", "M1", ""),
        ] {
            XCTAssertEqual(BenchmarkHardware.chipFamily(brand), family, "family of \(brand)")
            XCTAssertEqual(BenchmarkHardware.chipTier(brand), tier, "tier of \(brand)")
        }
    }

    func testUnknownChipStringsDegradeToEmptyRatherThanGuessing() {
        // Intel Macs and anything unrecognised must not be filed under a made-up
        // family — an empty filter value is honest, a wrong one is not.
        XCTAssertEqual(BenchmarkHardware.chipFamily("Intel Core i9"), "")
        XCTAssertEqual(BenchmarkHardware.chipFamily(""), "")
        XCTAssertEqual(BenchmarkHardware.chipTier(""), "")
    }

    func testDisplayNameCombinesTheFilterAxesForTheGrid() {
        XCTAssertEqual(
            BenchmarkHardware(chip: "Apple M4 Max", gpuCores: 40, ramGB: 128,
                              osVersion: "27.0", onBattery: false).displayName,
            "Apple M4 Max · 40 GPU · 128 GB")
        // A machine that wouldn't report its core count still reads cleanly.
        XCTAssertEqual(
            BenchmarkHardware(chip: "Apple M4", gpuCores: 0, ramGB: 16,
                              osVersion: "27.0", onBattery: false).displayName,
            "Apple M4 · 16 GB")
    }

    // MARK: - Live read

    func testTheLiveReadFillsInThisMachine() {
        // Not asserting values (this runs on whatever Mac is building), only
        // that the kernel reads land — a silently empty hardware block would
        // make every row from that machine unfilterable.
        let hw = SystemMetrics.benchmarkHardware()
        XCTAssertFalse(hw.chip.isEmpty, "chip brand string did not read")
        XCTAssertGreaterThan(hw.ramGB, 0, "hw.memsize did not read")
        XCTAssertFalse(hw.osVersion.isEmpty, "OS version did not read")
    }
}
