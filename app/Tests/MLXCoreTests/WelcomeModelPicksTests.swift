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

    func testTwentyFourGBMacGetsGemma12BAndQwen9BAndDropsLaguna() {
        let picks = WelcomeModelPicks.forMemory(mac(total: 24, usable: 16))
        // General → Gemma 4 12B (26B-A4B needs ~17 GB, exceeds 16 usable).
        XCTAssertEqual(picks.first { $0.category == "General" }?.pick.id, "gemma-4-12b")
        // Coding & agents → Qwen 9B (27B needs ~18 GB, exceeds).
        XCTAssertEqual(picks.first { $0.category == "Coding & agents" }?.pick.id, "qwen35-9b")
        // Coding specialist → Laguna XS needs ~24 GB, exceeds 16 → category dropped.
        XCTAssertNil(picks.first { $0.category == "Coding specialist" })
        XCTAssertEqual(picks.count, 2)
    }

    func testLargeMacGetsTheBiggestOfEachType() {
        let picks = WelcomeModelPicks.forMemory(mac(total: 256, usable: 200))
        XCTAssertEqual(picks.first { $0.category == "General" }?.pick.id, "gemma-4-31b-8bit")
        XCTAssertEqual(picks.first { $0.category == "Coding & agents" }?.pick.id, "qwen36-35b-a3b")
        XCTAssertEqual(picks.first { $0.category == "Coding specialist" }?.pick.id, "laguna-s-2.1-nvfp4")
        XCTAssertEqual(picks.count, 3)
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
        XCTAssertEqual(picks.first { $0.category == "General" }?.pick.id, "gemma-4-e2b")
    }
}
