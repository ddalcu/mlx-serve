import XCTest
@testable import MLXCore

/// Pins the agent loop's *recoverable-failure* budget (malformed/"ghost" tool
/// calls, truncated tool-call args, empty/pad responses).
///
/// Regression context: these used to be plain counters that accumulated across
/// the whole turn and never reset. So when the model botched one tool call's
/// JSON early (e.g. an `editFile` with unescaped quotes), recovered, did real
/// work, then botched *another* tool call much later, the budget was already
/// spent and the loop `return`ed mid-task — the "agent just stops while it
/// should keep going" bug. The fix makes the budget count *consecutive*
/// failures and reset on progress, so an isolated late failure still gets a retry.
final class AgentRetryBudgetTests: XCTestCase {

    func testGhostRetryIsConsecutiveAndResetsOnProgress() {
        var b = AgentEngine.AgentRetryBudget()
        XCTAssertTrue(b.allowGhostRetry(), "first ghost in a stretch gets a retry")
        XCTAssertFalse(b.allowGhostRetry(), "second *consecutive* ghost gives up")
        // A round that executed real tool calls happened in between.
        b.recordProgress()
        XCTAssertTrue(b.allowGhostRetry(), "an isolated *later* ghost must get its own retry — the fix")
    }

    func testTruncationRetryIsConsecutiveAndResetsOnProgress() {
        var b = AgentEngine.AgentRetryBudget()
        XCTAssertTrue(b.allowTruncationRetry())
        XCTAssertTrue(b.allowTruncationRetry())
        XCTAssertFalse(b.allowTruncationRetry(), "limit is 2 consecutive")
        b.recordProgress()
        XCTAssertTrue(b.allowTruncationRetry(), "reset after progress")
    }

    func testRecordProgressClearsPadCounter() {
        var b = AgentEngine.AgentRetryBudget()
        b.pad = 3
        b.recordProgress()
        XCTAssertEqual(b.pad, 0)
    }
}
