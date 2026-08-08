import XCTest
@testable import MLXCore

/// Numbers behind the composer's context pill and its popover.
///
/// Split from `ContextMonitor` (which owns the bar's clamped ratio) because the
/// readout must be able to say **100.3%**: clamping is right for a progress bar
/// and wrong for a figure whose whole job is to show you went over.
final class ContextWindowStatsTests: XCTestCase {

    private func stats(prompt: Int, completion: Int = 0, live: Int = 0,
                       context: Int, overflow: ChatErrorNotice? = nil) -> ContextWindowStats {
        ContextWindowStats.make(promptTokens: prompt, completionTokens: completion,
                                liveTokens: live, contextLength: context, overflow: overflow)
    }

    // MARK: - Fractions

    func testFractionIsUnclampedSoOverflowIsVisible() {
        let s = stats(prompt: 4108, context: 4096)
        XCTAssertGreaterThan(s.fraction, 1.0)
        XCTAssertEqual(s.percentText, "100.3%")
    }

    func testBarFractionStaysClampedForDrawing() {
        // The bar is a width multiplier — past 1.0 it draws outside its track.
        let s = stats(prompt: 9000, context: 4096)
        XCTAssertEqual(s.barFraction, 1.0, accuracy: 0.0001)
        XCTAssertGreaterThan(s.fraction, 2.0)
    }

    func testUsedIsPromptPlusReplyPlusLive() {
        let s = stats(prompt: 1000, completion: 200, live: 37, context: 8192)
        XCTAssertEqual(s.usedTokens, 1237)
        XCTAssertEqual(s.remainingTokens, 8192 - 1237)
    }

    func testRemainingNeverGoesNegative() {
        // "Remaining: -12" is nonsense in a readout; zero is the honest floor.
        XCTAssertEqual(stats(prompt: 4108, context: 4096).remainingTokens, 0)
    }

    func testUnknownContextLengthIsInert() {
        // Before the first reply there is no context length; the pill must not
        // divide by zero or claim a percentage it can't know.
        let s = stats(prompt: 100, context: 0)
        XCTAssertEqual(s.fraction, 0)
        XCTAssertEqual(s.barFraction, 0)
        XCTAssertFalse(s.isOverflowed)
        XCTAssertEqual(s.remainingTokens, 0)
    }

    func testOverflowFlagTracksTheRealComparison() {
        XCTAssertFalse(stats(prompt: 4096, context: 4096).isOverflowed, "exactly full still fits")
        XCTAssertTrue(stats(prompt: 4097, context: 4096).isOverflowed)
    }

    // MARK: - The rejected request wins

    func testOverflowNoticeSuppliesTheCountsInsteadOfTheLastGoodTurn() {
        // After a rejected request the last SUCCESSFUL turn's usage is stale and
        // reads comfortably under the limit — exactly when the user is looking
        // to find out why it failed. The rejected request's own counts win.
        let notice = ChatErrorNotice(kind: .contextOverflow, message: "m",
                                     neededTokens: 4108, contextLength: 4096)
        let s = stats(prompt: 1200, completion: 300, context: 4096, overflow: notice)
        XCTAssertEqual(s.promptTokens, 4108)
        XCTAssertEqual(s.usedTokens, 4108)
        XCTAssertEqual(s.contextLength, 4096)
        XCTAssertTrue(s.isOverflowed)
        XCTAssertEqual(s.percentText, "100.3%")
    }

    func testOverflowNoticeWithoutCountsLeavesTheLiveNumbersAlone() {
        // An older server sends no figures — inventing them would be worse than
        // showing the last turn's real usage.
        let notice = ChatErrorNotice(kind: .contextOverflow, message: "m",
                                     neededTokens: nil, contextLength: nil)
        let s = stats(prompt: 1200, completion: 300, context: 4096, overflow: notice)
        XCTAssertEqual(s.usedTokens, 1500)
        XCTAssertEqual(s.contextLength, 4096)
    }

    func testAGenericErrorDoesNotTouchTheCounts() {
        let notice = ChatErrorNotice(kind: .generic, message: "boom",
                                     neededTokens: nil, contextLength: nil)
        let s = stats(prompt: 1200, completion: 300, context: 4096, overflow: notice)
        XCTAssertEqual(s.usedTokens, 1500)
        XCTAssertFalse(s.isOverflowed)
    }

    // MARK: - Compact formatting

    func testCompactTokenCounts() {
        // 1000-based with one decimal, so 4096 and 4108 both read "4.1K" — the
        // pill is a gauge, and the exact figures live in the popover's rows.
        XCTAssertEqual(ContextWindowStats.compact(0), "0")
        XCTAssertEqual(ContextWindowStats.compact(999), "999")
        XCTAssertEqual(ContextWindowStats.compact(4096), "4.1K")
        XCTAssertEqual(ContextWindowStats.compact(4108), "4.1K")
        XCTAssertEqual(ContextWindowStats.compact(78_848), "78.8K")
        XCTAssertEqual(ContextWindowStats.compact(1_048_576), "1.0M")
    }

    func testPercentTextAlwaysCarriesOneDecimal() {
        XCTAssertEqual(stats(prompt: 0, context: 4096).percentText, "0.0%")
        XCTAssertEqual(stats(prompt: 2048, context: 4096).percentText, "50.0%")
    }

    // MARK: - Pressure banding (drives the pill's color)

    func testPressureBandsMatchTheBarColorThresholds() {
        XCTAssertEqual(stats(prompt: 100, context: 4096).pressure, .comfortable)
        XCTAssertEqual(stats(prompt: 2900, context: 4096).pressure, .warm)     // >60%
        XCTAssertEqual(stats(prompt: 3600, context: 4096).pressure, .tight)    // >80%
        XCTAssertEqual(stats(prompt: 4200, context: 4096).pressure, .over)
    }
}

// MARK: - Decode speed

/// The popover's speed row. The FIGURE is the server's own
/// `timings.predicted_per_second` (APIClient prefers it over wall-clock — a
/// client cannot time our own stream: with tools present the server buffers for
/// tool-call detection and every SSE delta lands at once, which read as 937
/// tok/s on a 2B). These pin only the presentation: a rate we don't have must
/// render as nothing rather than as zero.
extension ContextWindowStatsTests {

    func testSpeedTextIsAbsentUntilAReplyHasBeenTimed() {
        XCTAssertNil(ContextWindowStats.speedText(nil))
    }

    /// A zero or negative rate is a MISSING measurement, not a slow model —
    /// "0 tok/s" beside a model that just answered reads as a bug in the model.
    func testANonPositiveRateReadsAsMissingNotAsZero() {
        XCTAssertNil(ContextWindowStats.speedText(0))
        XCTAssertNil(ContextWindowStats.speedText(-1))
    }

    func testSpeedTextRoundsToWholeTokensPerSecond() {
        XCTAssertEqual(ContextWindowStats.speedText(96.1), "96 tok/s")
        XCTAssertEqual(ContextWindowStats.speedText(67.5), "68 tok/s")
    }

    /// Sub-1 tok/s is a real reading on a huge model at long context, and
    /// truncating it to "0 tok/s" would say the same thing as no measurement.
    func testSubOneRateKeepsADecimalRatherThanCollapsingToZero() {
        XCTAssertEqual(ContextWindowStats.speedText(0.4), "0.4 tok/s")
    }
}
