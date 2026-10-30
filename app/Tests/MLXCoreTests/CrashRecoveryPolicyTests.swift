import XCTest
@testable import MLXCore

/// #265: an unexpected server exit used to be terminal — modal alert, dead
/// server, every agent CLI failing until somebody noticed. Auto-restart is only
/// safe if it CANNOT turn into a relaunch treadmill, so these pin the three
/// loop-breaking rules directly (the whole reaction is a pure function, no
/// process needed):
///
/// 1. `.ask` never restarts — the old behavior survives as the default.
/// 2. the delay grows and the streak is finite, then the alert comes back.
/// 3. a run that stayed healthy resets the streak; a run that never reached
///    `.running` (died during load) counts toward it.
final class CrashRecoveryPolicyTests: XCTestCase {

    func testAskNeverRestarts() {
        var policy = CrashRestartPolicy.standard
        XCTAssertEqual(policy.reaction(mode: .ask, canRelaunch: true, uptime: nil), .alert)
        XCTAssertEqual(policy.reaction(mode: .ask, canRelaunch: true, uptime: 0), .alert)
        XCTAssertEqual(policy.attempts, 0, "an alert must not spend the retry budget")
    }

    func testBothAutomaticModesRestart() {
        for mode in [CrashRecoveryMode.restartQuietly, .restartAndTellMe] {
            var policy = CrashRestartPolicy.standard
            XCTAssertEqual(policy.reaction(mode: mode, canRelaunch: true, uptime: nil),
                           .restart(after: 1))
        }
    }

    func testNoLaunchToReplayFallsBackToAlert() {
        var policy = CrashRestartPolicy.standard
        XCTAssertEqual(policy.reaction(mode: .restartQuietly, canRelaunch: false, uptime: nil), .alert)
        XCTAssertEqual(policy.attempts, 0)
    }

    func testBackoffDelaysGrowAndTheStreakIsFinite() {
        var policy = CrashRestartPolicy.standard
        var delays: [TimeInterval] = []
        for _ in 0..<policy.budget {
            guard case let .restart(delay) = policy.reaction(mode: .restartQuietly,
                                                            canRelaunch: true, uptime: nil) else {
                return XCTFail("streak ended before its budget was spent")
            }
            delays.append(delay)
        }
        XCTAssertEqual(delays, [1, 2, 4, 8, 16, 30])
        XCTAssertEqual(delays, delays.sorted(), "a retry loop must never get faster")

        // Budget gone: the user gets the crash log instead of attempt seven.
        XCTAssertEqual(policy.reaction(mode: .restartQuietly, canRelaunch: true, uptime: nil), .alert)
        XCTAssertEqual(policy.reaction(mode: .restartQuietly, canRelaunch: true, uptime: nil), .alert)
    }

    func testAHealthyRunResetsTheStreak() {
        var policy = CrashRestartPolicy.standard
        for _ in 0..<(policy.budget - 1) {
            _ = policy.reaction(mode: .restartQuietly, canRelaunch: true, uptime: nil)
        }
        XCTAssertEqual(policy.attempts, policy.budget - 1)

        // Ran fine for a while after the last restart: not a crash loop.
        XCTAssertEqual(policy.reaction(mode: .restartQuietly, canRelaunch: true,
                                       uptime: CrashRestartPolicy.stableUptime),
                       .restart(after: 1))
        XCTAssertEqual(policy.attempts, 1, "the streak restarted from the top")
    }

    func testAShortLivedRunContinuesTheStreak() {
        var policy = CrashRestartPolicy.standard
        XCTAssertEqual(policy.reaction(mode: .restartQuietly, canRelaunch: true, uptime: nil),
                       .restart(after: 1))
        // Died 5 s after coming up: still a loop, so it backs off.
        XCTAssertEqual(policy.reaction(mode: .restartQuietly, canRelaunch: true, uptime: 5),
                       .restart(after: 2))
        XCTAssertEqual(policy.attempts, 2)
    }

    func testUptimeAtTheStabilityThresholdCountsAsStable() {
        XCTAssertTrue(CrashRestartPolicy.countsAsStable(uptime: CrashRestartPolicy.stableUptime))
        XCTAssertFalse(CrashRestartPolicy.countsAsStable(
            uptime: CrashRestartPolicy.stableUptime - 0.1))
    }

    func testExplicitResetClearsTheStreak() {
        var policy = CrashRestartPolicy.standard
        _ = policy.nextDelay()
        _ = policy.nextDelay()
        XCTAssertEqual(policy.attempts, 2)
        policy.reset()
        XCTAssertEqual(policy.attempts, 0)
        XCTAssertEqual(policy.nextDelay(), 1, "Stop / manual Start must restore the fast first retry")
    }

    func testModeRoundTripsThroughItsStoredRawValue() {
        // `@AppStorage` writes the raw string; an unknown or missing value must
        // read back as the safe default rather than crash the Settings row.
        for mode in CrashRecoveryMode.allCases {
            XCTAssertEqual(CrashRecoveryMode(rawValue: mode.rawValue), mode)
        }
        XCTAssertEqual(CrashRecoveryMode(rawValue: "nonsense") ?? .ask, .ask)
        XCTAssertEqual(CrashRecoveryMode(rawValue: "") ?? .ask, .ask)
        XCTAssertEqual(CrashRecoveryMode.ask.restartsAutomatically, false)
        XCTAssertEqual(CrashRecoveryMode.restartAndTellMe.postsNotification, true)
        XCTAssertEqual(CrashRecoveryMode.restartQuietly.postsNotification, false)
    }

    /// The defaults are the contract for people who never open Settings: no
    /// behavior change, no surprise notification.
    func testDefaultModeIsAsk() {
        let key = CrashRecoveryMode.prefKey
        let saved = UserDefaults.standard.object(forKey: key)
        UserDefaults.standard.removeObject(forKey: key)
        defer {
            if let saved { UserDefaults.standard.set(saved, forKey: key) }
            else { UserDefaults.standard.removeObject(forKey: key) }
        }
        XCTAssertEqual(CrashRecoveryMode.current, .ask)
    }
}
