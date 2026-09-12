import XCTest
@testable import MLXCore

/// #265: when mlx-serve crashes while .running, the app shows a modal alert
/// and stays stopped — every LAN client is down until someone clicks OK.
/// Auto-restart with backoff recovers the server automatically.
@MainActor
final class CrashRecoveryTests: XCTestCase {

    // MARK: - CrashRecoveryMode

    func testDefaultModeIsAsk() {
        XCTAssertEqual(CrashRecoveryMode.defaultMode, .ask)
    }

    // MARK: - shouldAutoRestart (pure decision)

    func testAutoRestartWhenModeIsAutoAndWasRunning() {
        let decision = CrashRecovery.shouldAutoRestart(
            mode: .autoRestart, wasRunning: true, exitCode: 9,
            isMemoryFailure: false, crashCount: 0, maxRetries: 3
        )
        XCTAssertTrue(decision, "auto-restart mode + running → restart")
    }

    func testNoAutoRestartInAskMode() {
        let decision = CrashRecovery.shouldAutoRestart(
            mode: .ask, wasRunning: true, exitCode: 9,
            isMemoryFailure: false, crashCount: 0, maxRetries: 3
        )
        XCTAssertFalse(decision, "ask mode → never auto-restart")
    }

    func testNoAutoRestartOnMemoryFailure() {
        let decision = CrashRecovery.shouldAutoRestart(
            mode: .autoRestart, wasRunning: true, exitCode: 9,
            isMemoryFailure: true, crashCount: 0, maxRetries: 3
        )
        XCTAssertFalse(decision, "OOM will recur immediately — don't retry")
    }

    func testNoAutoRestartAfterMaxRetries() {
        let decision = CrashRecovery.shouldAutoRestart(
            mode: .autoRestart, wasRunning: true, exitCode: 9,
            isMemoryFailure: false, crashCount: 3, maxRetries: 3
        )
        XCTAssertFalse(decision, "exhausted retries → fall back to modal")
    }

    func testNoAutoRestartWhenServerWasNotRunning() {
        let decision = CrashRecovery.shouldAutoRestart(
            mode: .autoRestart, wasRunning: false, exitCode: 1,
            isMemoryFailure: false, crashCount: 0, maxRetries: 3
        )
        XCTAssertFalse(decision, "failed during startup → don't retry")
    }

    func testSilentModeAutoRestarts() {
        let decision = CrashRecovery.shouldAutoRestart(
            mode: .silent, wasRunning: true, exitCode: 11,
            isMemoryFailure: false, crashCount: 0, maxRetries: 3
        )
        XCTAssertTrue(decision, "silent mode + running → restart")
    }

    func testCleanExitDoesNotAutoRestart() {
        let decision = CrashRecovery.shouldAutoRestart(
            mode: .autoRestart, wasRunning: true, exitCode: 0,
            isMemoryFailure: false, crashCount: 0, maxRetries: 3
        )
        XCTAssertFalse(decision, "exit 0 while .running is a clean exit, not a crash")
    }

    // MARK: - backoffDelay (pure)

    func testBackoffDelayExponential() {
        XCTAssertEqual(CrashRecovery.backoffDelay(attempt: 1), 1.0)
        XCTAssertEqual(CrashRecovery.backoffDelay(attempt: 2), 2.0)
        XCTAssertEqual(CrashRecovery.backoffDelay(attempt: 3), 4.0)
    }

    func testBackoffDelayCappedAt30() {
        XCTAssertEqual(CrashRecovery.backoffDelay(attempt: 10), 30.0)
    }

    // MARK: - crashWindowExpired (pure)

    func testCrashCountResetsAfterWindow() {
        let old = Date().addingTimeInterval(-400) // 400s ago, window is 300s
        XCTAssertTrue(CrashRecovery.crashWindowExpired(lastCrash: old, window: 300))
    }

    func testCrashCountPreservedWithinWindow() {
        let recent = Date().addingTimeInterval(-100) // 100s ago
        XCTAssertFalse(CrashRecovery.crashWindowExpired(lastCrash: recent, window: 300))
    }

    func testNilLastCrashTreatedAsExpired() {
        XCTAssertTrue(CrashRecovery.crashWindowExpired(lastCrash: nil, window: 300))
    }
}
