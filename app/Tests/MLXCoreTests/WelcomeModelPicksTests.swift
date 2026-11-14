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

    /// From 32 GB up the welcome lists ONE model — the starter pick, the same
    /// one the sheet after it offers.
    func testFrom32GBTheWelcomeListsOnlyTheStarterPick() {
        let expected: [(UInt64, String)] = [
            (32, "qwen38-27b"), (36, "qwen38-27b"), (48, "qwen38-27b-6bit"),
            (64, "qwen38-27b-8bit"), (96, "qwen38-27b-8bit"), (256, "qwen38-flash-next"),
        ]
        for (total, id) in expected {
            let memory = mac(total: total, usable: total * 3 / 4)
            let picks = WelcomeModelPicks.forMemory(memory)
            XCTAssertEqual(picks.map(\.pick.id), [id], "\(total) GB")
            XCTAssertEqual(WelcomeModelPicks.recommendedId(in: picks, memory: memory), id)
        }
    }

    /// A starter pick the list doesn't carry (16 GB: Gemma E4B) marks row 0.
    func testRecommendedMarkFallsBackToTheFirstRow() {
        let memory = mac(total: 16, usable: 11)
        let picks = WelcomeModelPicks.forMemory(memory)
        XCTAssertNil(picks.first { $0.pick.id == "gemma-4-e4b" })
        XCTAssertEqual(WelcomeModelPicks.recommendedId(in: picks, memory: memory), picks.first?.id)
    }

    func testEveryPickHasAOneLineStrength() {
        for p in WelcomeModelPicks.forMemory(mac(total: 24, usable: 16)) + WelcomeModelPicks.forMemory(mac(total: 256, usable: 200)) {
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
