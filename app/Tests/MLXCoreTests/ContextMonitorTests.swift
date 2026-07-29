import XCTest
@testable import MLXCore

/// What the chat counts as occupied context.
///
/// The readout itself is now the composer's `ContextPill`, and the clamped
/// ratio + colour bands it used to own live in `ContextWindowStats` (which needs
/// an UNCLAMPED fraction so it can show 100.3%) — see `ContextWindowStatsTests`.
/// The summing rule stayed here because it is what "used" MEANS, independent of
/// how it is drawn.
final class ContextMonitorTests: XCTestCase {

    func testUsedTokensSumsPromptCompletionAndLive() {
        // Idle after a turn: prompt + the reply that landed, no live tokens.
        XCTAssertEqual(ContextMonitor.usedTokens(promptTokens: 1000, completionTokens: 200, liveTokens: 0), 1200)
        // Mid-stream: the in-flight reply's running count adds on top, so the
        // gauge grows as the answer arrives instead of only snapping at the end.
        XCTAssertEqual(ContextMonitor.usedTokens(promptTokens: 1000, completionTokens: 200, liveTokens: 37), 1237)
    }

    func testStatsAndMonitorAgreeOnUsed() {
        // One definition of "used" — `ContextWindowStats` must not re-derive it,
        // or the pill and anything else reading the monitor would drift apart.
        let stats = ContextWindowStats.make(promptTokens: 900, completionTokens: 80, liveTokens: 12,
                                            contextLength: 8192, overflow: nil)
        XCTAssertEqual(stats.usedTokens,
                       ContextMonitor.usedTokens(promptTokens: 900, completionTokens: 80, liveTokens: 12))
    }
}
