import XCTest
@testable import MLXCore

final class VideoPauseTests: XCTestCase {
    func testPauseIsAskedThenConfirmedByTheServerThenResumed() {
        var p = VideoPause()
        p.request()
        XCTAssertEqual(p.state, .pausing)
        p.serverPaused(at: 100)
        XCTAssertEqual(p.state, .paused)
        p.resume()
        XCTAssertEqual(p.state, .running)
        XCTAssertEqual(p.serverResumed(at: 130), 30, accuracy: 0.001)
        XCTAssertEqual(p.state, .running)
    }

    /// Cancel pause before the step ends: a pause the server took anyway still counts as
    /// held time, but the button never flips to Resume.
    func testAWithdrawnPauseStaysWithdrawnEvenIfTheServerHeldBriefly() {
        var p = VideoPause()
        p.request()
        p.withdraw()
        XCTAssertEqual(p.state, .running)
        p.serverPaused(at: 50)
        XCTAssertEqual(p.state, .running)
        XCTAssertEqual(p.serverResumed(at: 50.2), 0.2, accuracy: 0.001)
    }

    func testWithdrawAndResumeOnlyActFromTheirOwnState() {
        var p = VideoPause()
        p.withdraw()
        XCTAssertEqual(p.state, .running)
        XCTAssertEqual(p.serverResumed(at: 10), 0) // never held
        p.request()
        p.request()
        XCTAssertEqual(p.state, .pausing)
    }

    /// The lap that spans a pause prices only the step's own time.
    func testStepClockLeavesPausedTimeOutOfTheLap() {
        var clock = H3StepClock()
        clock.observe(step: 0, at: 0)
        clock.observe(step: 1, at: 100)   // opening lap
        clock.observe(step: 2, at: 110)
        clock.excludePause(500)
        clock.observe(step: 3, at: 620)
        XCTAssertEqual(clock.durations, [100, 10, 10])
    }
}
