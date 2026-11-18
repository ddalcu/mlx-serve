import XCTest
@testable import MLXCore

/// The welcome screen shows the user's RAM and recommends one model for it.
/// This pins the RAM formatting and — crucially — that the recommendation is
/// the SAME `starterPick` the rest of the app uses, tier for tier, so the
/// welcome screen can never drift from the Model Browser / chat gate.
final class WelcomeRecommendationTests: XCTestCase {
    private let gib: UInt64 = 1_073_741_824

    /// A Mac of `gb` GB that gives models `usable` GB, by default 3/4 of RAM.
    private func mem(_ gb: Double, usable: Double? = nil) -> SystemMemoryInfo {
        SystemMemoryInfo(totalBytes: UInt64(gb * Double(gib)), usableBytes: UInt64((usable ?? gb * 3 / 4) * Double(gib)))
    }

    func testRecommendsGemma12BForA24GBMac() {
        let r = WelcomeRecommendation.forMemory(mem(24))
        XCTAssertEqual(r.memoryText, "24 GB")
        XCTAssertEqual(r.pick.id, "gemma-4-12b", "24 GB is the 16–32 GB tier → Gemma 4 12B")
        XCTAssertTrue(r.rationale.contains("24 GB"), "the rationale names the machine's RAM")
    }

    func testMemoryLabelIsWholeGB() {
        XCTAssertEqual(WelcomeRecommendation.formatMemory(8.0), "8 GB")
        XCTAssertEqual(WelcomeRecommendation.formatMemory(16.0), "16 GB")
        XCTAssertEqual(WelcomeRecommendation.formatMemory(24.0), "24 GB")
        XCTAssertEqual(WelcomeRecommendation.formatMemory(128.0), "128 GB")
        // Real reports are exact powers of two, but rounding must still be clean.
        XCTAssertEqual(WelcomeRecommendation.formatMemory(23.8), "24 GB")
    }

    func testTiersMatchTheAppWideStarterPick() {
        // The welcome recommendation must equal `starterPick` at every tier —
        // a second copy of this decision is how surfaces start disagreeing.
        for gb: Double in [6, 8, 12, 16, 24, 32, 64, 128] {
            XCTAssertEqual(
                WelcomeRecommendation.forMemory(mem(gb)).pick,
                RecommendedModelPick.starterPick(memory: mem(gb)),
                "welcome pick must delegate to starterPick for \(gb) GB"
            )
        }
    }

    func testBoundaryTiersFollowStarterPick() {
        XCTAssertEqual(WelcomeRecommendation.forMemory(mem(8)).pick.id, "gemma-4-e4b")
        XCTAssertEqual(WelcomeRecommendation.forMemory(mem(16)).pick.id, "gemma-4-e4b")
        XCTAssertEqual(WelcomeRecommendation.forMemory(mem(96, usable: 96 * 0.84)).pick.id, "qwen38-flash-next")
        XCTAssertEqual(WelcomeRecommendation.forMemory(mem(32)).pick.id, "qwen38-27b")
        XCTAssertEqual(WelcomeRecommendation.forMemory(mem(64)).pick.id, "qwen38-27b-8bit")
    }
}
