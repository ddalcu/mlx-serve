import XCTest
@testable import MLXCore

/// The chat composer's context readout. It now shows a LIVE view driven by the
/// chat-completions stream: the "gen:" label counts the tokens the current reply
/// is producing (client-side, by counting streamed deltas), and the usage bar
/// grows as those tokens arrive instead of only snapping at end-of-response.
final class ContextMonitorTests: XCTestCase {

    // MARK: - Used-token total (drives the bar + percentage)

    func testUsedTokensSumsPromptCompletionAndLive() {
        // Idle after a turn: prompt + the reply that landed, no live tokens.
        XCTAssertEqual(ContextMonitor.usedTokens(promptTokens: 1000, completionTokens: 200, liveTokens: 0), 1200)
        // Mid-stream: the in-flight reply's running count adds on top, so the bar grows.
        XCTAssertEqual(ContextMonitor.usedTokens(promptTokens: 1000, completionTokens: 200, liveTokens: 37), 1237)
    }

    func testUsageRatioIsClampedToOne() {
        // Over-full context can't push the bar past 100%.
        XCTAssertEqual(ContextMonitor.usageRatio(usedTokens: 5000, contextLength: 4096), 1.0, accuracy: 0.0001)
        XCTAssertEqual(ContextMonitor.usageRatio(usedTokens: 1024, contextLength: 4096), 0.25, accuracy: 0.0001)
        // No divide-by-zero before a context length is known.
        XCTAssertEqual(ContextMonitor.usageRatio(usedTokens: 100, contextLength: 0), 0.0, accuracy: 0.0001)
    }

    // MARK: - "gen:" token count

    func testGenTokensIsLiveCountWhileStreaming() {
        // While this chat is generating, "gen:" is the running live count...
        XCTAssertEqual(ContextMonitor.genTokens(isLive: true, liveTokens: 42, completionTokens: 999), 42)
        // ...and when idle it shows the last completed reply's length.
        XCTAssertEqual(ContextMonitor.genTokens(isLive: false, liveTokens: 42, completionTokens: 200), 200)
    }
}
