import XCTest
@testable import MLXCore

/// When a model load is refused for memory, the crash alert lists the apps
/// using the most memory so the user knows what to quit. These pin the pure
/// ranking/formatting and the memory-failure classifier that triggers it.
final class RunningAppsMemoryTests: XCTestCase {
    private let mb: Int64 = 1024 * 1024
    private func app(_ name: String, _ gb: Double) -> RunningAppsMemory.AppMem {
        RunningAppsMemory.AppMem(name: name, bytes: Int64(gb * 1024) * mb)
    }

    func testRankSortsDescendingAndCaps() {
        let apps = [app("A", 0.5), app("B", 2.0), app("C", 1.2), app("D", 0.9)]
        let top = RunningAppsMemory.rank(apps, limit: 2)
        XCTAssertEqual(top.map(\.name), ["B", "C"])
    }

    func testRankLimitZeroIsEmptyAndNeverCrashes() {
        XCTAssertTrue(RunningAppsMemory.rank([app("A", 1)], limit: 0).isEmpty)
        XCTAssertTrue(RunningAppsMemory.rank([], limit: 5).isEmpty)
    }

    func testSummaryLineNamesEachAppWithItsSize() {
        let line = RunningAppsMemory.summaryLine([app("Figma", 1.5), app("Xcode", 1.0)])
        XCTAssertEqual(line, "Figma 1.5 GB · Xcode 1.0 GB")
    }

    func testTotalBytesSums() {
        let apps = [app("A", 1.5), app("B", 2.5)]
        XCTAssertEqual(RunningAppsMemory.totalBytes(apps), Int64(4.0 * 1024) * mb)
    }

    // MARK: - Memory-failure classifier

    func testPreflightRefusalIsAMemoryFailure() {
        let log = "[preflight] weights ~9.06 GB, available 5.96 GB\nInsufficient memory to load model: weights ~9.1 GB but only 6.0 GB free."
        XCTAssertTrue(ServerManager.isMemoryFailure(log))
    }

    func testMetalOOMIsAMemoryFailure() {
        XCTAssertTrue(ServerManager.isMemoryFailure("[METAL] Command buffer execution failed: Insufficient Memory"))
        XCTAssertTrue(ServerManager.isMemoryFailure("error: kIOGPUCommandBufferCallbackErrorOutOfMemory"))
    }

    func testUnrelatedCrashIsNotAMemoryFailure() {
        XCTAssertFalse(ServerManager.isMemoryFailure("MISSING WEIGHT: lm_head.weight"))
        XCTAssertFalse(ServerManager.isMemoryFailure("exit code 1"))
    }
}
