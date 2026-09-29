import XCTest
@testable import MLXCore

/// The welcome screen lists the best model of each type that fits this Mac. It
/// must (a) pick the largest fitting model per family, (b) drop families where
/// nothing fits, and (c) carry a one-line strength.
final class WelcomeModelPicksTests: XCTestCase {
    private let gib: UInt64 = 1_073_741_824
    private func mac(total: UInt64, usable: UInt64) -> SystemMemoryInfo {
        SystemMemoryInfo(totalBytes: total * gib, usableBytes: usable * gib)
    }

    func testTwentyFourGBMacGetsGemma12BAndBonsai() {
        let picks = WelcomeModelPicks.forMemory(mac(total: 24, usable: 16))
        // General → Gemma 4 12B (26B-A4B needs ~17 GB, exceeds 16 usable).
        XCTAssertEqual(picks.first { $0.category == "General" }?.pick.id, "gemma-4-12b")
        // Coding & agents → Bonsai 2 (the 27B packs need ~19-22 GB, exceed).
        XCTAssertEqual(picks.first { $0.category == "Coding & agents" }?.pick.id, "bonsai2-27b")
        XCTAssertEqual(picks.count, 2)
    }

    /// A 32 GB Mac (usable ~27): Gemma 4 31B and Qwen 3.8 27B are the
    /// biggest COMFORTABLE fits. The 8-bit Gemma and the 35B-A3B land tight
    /// there, and the welcome leads with comfort — a tight fit is what fails
    /// under real memory pressure.
    func testThirtyTwoGBMacGetsGemma31BAndQwen27B() {
        let picks = WelcomeModelPicks.forMemory(mac(total: 32, usable: 27))
        XCTAssertEqual(picks.first { $0.category == "General" }?.pick.id, "gemma-4-31b")
        XCTAssertEqual(picks.first { $0.category == "Coding & agents" }?.pick.id, "qwen38-27b")
        XCTAssertEqual(picks.count, 2)
    }

    func testLargeMacGetsTheBiggestOfEachType() {
        let picks = WelcomeModelPicks.forMemory(mac(total: 256, usable: 200))
        XCTAssertEqual(picks.first { $0.category == "General" }?.pick.id, "gemma-4-26b-a4b-8bit")
        XCTAssertEqual(picks.first { $0.category == "Coding & agents" }?.pick.id, "qwen36-35b-a3b")
        XCTAssertNil(picks.first { $0.pick.id == "qwen38-flash-next" }, "Largest is a browser-only tier, not a welcome category")
        XCTAssertEqual(picks.count, 2)
    }

    func testEveryPickHasAOneLineStrength() {
        for p in WelcomeModelPicks.forMemory(mac(total: 256, usable: 200)) {
            XCTAssertFalse(p.strength.isEmpty)
            XCTAssertFalse(p.strength.contains("\n"), "strength must be a single short line")
        }
    }

    func testTinyMacStillGetsAtLeastAGeneralModel() {
        // 8 GB: usable ~6. Only the smallest Gemma fits; coding families drop.
        let picks = WelcomeModelPicks.forMemory(mac(total: 8, usable: 6))
        XCTAssertEqual(picks.first { $0.category == "General" }?.pick.id, "gemma-4-e4b")
    }
}
