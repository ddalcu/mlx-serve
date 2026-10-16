import XCTest
@testable import MLXCore

/// The Video pane's live readout: percent, step line and pace.
final class VideoRunReadoutTests: XCTestCase {

    private func readout(step: Int, total: Int = 30, window: Int? = nil, windows: Int? = nil,
                         shot: Int = 0, shots: Int = 1, share: Double = 0.9,
                         pace: Double? = nil) -> VideoRunReadout {
        VideoRunReadout.make(step: step, total: total, window: window, windows: windows,
                             shot: shot, shots: shots, samplingShare: share,
                             secondsPerStep: pace)
    }

    func testPercentFollowsTheStepsWithinTheDenoisersShare() {
        XCTAssertEqual(readout(step: 0).percent, 0)
        XCTAssertEqual(readout(step: 15).percent, 45)
        XCTAssertEqual(readout(step: 15).stepLine, "Step 15 of 30")
    }

    /// The decode after the last step reports `step == total` for minutes.
    func testTheBarNeverReadsDoneWhileTheVideoIsStillBeingDecoded() {
        XCTAssertEqual(readout(step: 30).percent, 90)
        XCTAssertEqual(readout(step: 30, share: 1.0).percent, 99)
        XCTAssertEqual(readout(step: 31, share: 1.0).fraction, VideoRunReadout.maxRunningFraction)
    }

    func testAStoryboardsShotsShareOneBar() {
        XCTAssertEqual(readout(step: 0, shot: 1, shots: 4).percent, 25)
        XCTAssertEqual(readout(step: 15, shot: 1, shots: 4).fraction, (1 + 0.45) / 4, accuracy: 1e-9)
        XCTAssertLessThan(readout(step: 30, shot: 3, shots: 4).fraction, 1)
    }

    /// The server remaps a chained run's steps onto the whole chain.
    func testAChainedRunNamesItsWindowAndTheStepWithinIt() {
        let r = readout(step: 34, total: 90, window: 2, windows: 3)
        XCTAssertEqual(r.stepLine, "Window 2 of 3 · step 4 of 30")
        XCTAssertEqual(r.fraction, 34.0 / 90 * 0.9, accuracy: 1e-9)
        XCTAssertEqual(readout(step: 34, total: 90, window: 1, windows: 1).stepLine, "Step 34 of 90")
    }

    func testPaceAppearsOnceTheClockHasLaps() {
        XCTAssertNil(readout(step: 2).pace)
        // The decimal mark is the reader's locale's.
        XCTAssertEqual(readout(step: 9, pace: 12.34).pace,
                       String(format: "%.1f s per step", locale: .current, 12.3))
        XCTAssertEqual(readout(step: 9, pace: 150).pace,
                       String(format: "%.1f min per step", locale: .current, 2.5))
    }

    func testH3SharePricesTheStepsAgainstTheStagesAroundThem() {
        XCTAssertEqual(VideoRunReadout.samplingShare(perStep: 10, steps: 30, tail: 100), 0.75, accuracy: 1e-9)
        XCTAssertEqual(VideoRunReadout.samplingShare(perStep: 0, steps: 30, tail: 100),
                       VideoRunReadout.ltxSamplingShare)
    }

    func testTheStepClockReportsTheAmortizedLapPastTheFirst() {
        var clock = H3StepClock()
        clock.observe(step: 0, at: 0)
        clock.observe(step: 1, at: 100)
        clock.observe(step: 2, at: 110)
        XCTAssertNil(clock.secondsPerStep)
        clock.observe(step: 3, at: 130)
        XCTAssertEqual(clock.secondsPerStep, 15)
    }
}
