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

    func testAutoRestartAtExactMaxRetries() {
        let decision = CrashRecovery.shouldAutoRestart(
            mode: .autoRestart, wasRunning: true, exitCode: 9,
            isMemoryFailure: false, crashCount: 3, maxRetries: 3
        )
        XCTAssertTrue(decision, "crashCount == maxRetries → still restarts (3rd attempt)")
    }

    func testNoAutoRestartPastMaxRetries() {
        let decision = CrashRecovery.shouldAutoRestart(
            mode: .autoRestart, wasRunning: true, exitCode: 9,
            isMemoryFailure: false, crashCount: 4, maxRetries: 3
        )
        XCTAssertFalse(decision, "crashCount > maxRetries → fall back to modal")
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

    // MARK: - isMemoryFailure (preview-line filtering)

    func testMemoryFailureIgnoresRequestPreviewLines() {
        let log = """
        > "Insufficient memory to load model in the prompt"
        some normal log line
        """
        XCTAssertFalse(ServerManager.isMemoryFailure(log),
                       "A preview line matching OOM needles must not suppress auto-restart")
    }

    func testMemoryFailureDetectsRealOOM() {
        let log = "Insufficient memory to load model"
        XCTAssertTrue(ServerManager.isMemoryFailure(log))
    }

    // MARK: - CounterState (pure, drives the retry budget)

    /// Consecutive crashes within the window accumulate and eventually
    /// exhaust the retry budget.
    func testCounterAccumulatesAcrossRecordCrashCalls() {
        var counter = CrashRecovery.CounterState()
        let now = Date()
        let maxRetries = CrashRecovery.maxRetries

        for i in 1...maxRetries {
            counter.recordCrash(now: now)
            XCTAssertEqual(counter.count, i)
            XCTAssertTrue(CrashRecovery.shouldAutoRestart(
                mode: .autoRestart, wasRunning: true, exitCode: 9,
                isMemoryFailure: false, crashCount: counter.count, maxRetries: maxRetries
            ), "attempt \(i)/\(maxRetries) should restart")
        }
        // One more crash exceeds the budget.
        counter.recordCrash(now: now)
        XCTAssertEqual(counter.count, maxRetries + 1)
        XCTAssertFalse(CrashRecovery.shouldAutoRestart(
            mode: .autoRestart, wasRunning: true, exitCode: 9,
            isMemoryFailure: false, crashCount: counter.count, maxRetries: maxRetries
        ), "budget exhausted → modal")
    }

    /// Resetting the counter mid-sequence restores the full retry budget.
    /// This is the behaviour of a manual start; an auto-restart must NOT
    /// call reset(), or the budget never exhausts.
    func testResetRestoresFullRetryBudget() {
        var counter = CrashRecovery.CounterState()
        let now = Date()

        // Two crashes.
        counter.recordCrash(now: now)
        counter.recordCrash(now: now)
        XCTAssertEqual(counter.count, 2)

        // Manual start resets.
        counter.reset()
        XCTAssertEqual(counter.count, 0)
        XCTAssertNil(counter.lastCrashDate)

        // Next crash is attempt 1 again — budget restored.
        counter.recordCrash(now: now)
        XCTAssertEqual(counter.count, 1,
            "after reset the counter starts from 1, not 3")
    }

    /// The sliding window auto-resets the counter when enough time has
    /// passed since the last crash.
    func testWindowExpiryResetsCounter() {
        var counter = CrashRecovery.CounterState()
        let t0 = Date()
        // Two rapid crashes (within the window).
        counter.recordCrash(now: t0)
        counter.recordCrash(now: t0.addingTimeInterval(1))
        XCTAssertEqual(counter.count, 2)

        // Third crash 400s later — well past the 300s window.
        counter.recordCrash(now: t0.addingTimeInterval(400))
        XCTAssertEqual(counter.count, 1,
            "window expired → counter restarted from 0 before incrementing")
    }

    // MARK: - Settings reset notification

    func testResetPostsNotification() {
        let key = CrashRecoveryMode.defaultsKey
        UserDefaults.standard.set(CrashRecoveryMode.autoRestart.rawValue, forKey: key)

        let expectation = expectation(forNotification: CrashRecoveryMode.didResetNotification, object: nil)
        CrashRecoveryMode.resetIfApplicable(.all)
        wait(for: [expectation], timeout: 1.0)

        XCTAssertNil(UserDefaults.standard.string(forKey: key),
                     "reset should remove the persisted key")
    }

    func testResetDoesNotFireForUnrelatedCategory() {
        let key = CrashRecoveryMode.defaultsKey
        UserDefaults.standard.set(CrashRecoveryMode.autoRestart.rawValue, forKey: key)

        CrashRecoveryMode.resetIfApplicable(.category(.providers))

        XCTAssertEqual(UserDefaults.standard.string(forKey: key),
                       CrashRecoveryMode.autoRestart.rawValue,
                       "non-server reset must leave the key intact")
    }
}
