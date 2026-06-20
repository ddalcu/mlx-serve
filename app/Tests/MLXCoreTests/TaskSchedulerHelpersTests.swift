import XCTest
@testable import MLXCore

/// Pure helpers on TaskScheduler. The full headless-run flow needs a live server
/// (covered by manual e2e + the Phase 3 fake-engine integration test); these pin
/// the small, server-free seams.
final class TaskSchedulerHelpersTests: XCTestCase {

    func testDeriveTitleUsesFirstLineAndTrims() {
        XCTAssertEqual(TaskScheduler.deriveTitle(from: "Check HN\nand email me"), "Check HN")
        XCTAssertEqual(TaskScheduler.deriveTitle(from: "   spaced out   "), "spaced out")
        XCTAssertEqual(TaskScheduler.deriveTitle(from: ""), "Untitled task")
    }

    func testDeriveTitleTruncatesLongGoal() {
        let long = String(repeating: "x", count: 80)
        let title = TaskScheduler.deriveTitle(from: long)
        XCTAssertTrue(title.hasSuffix("…"))
        XCTAssertLessThanOrEqual(title.count, 49)
    }

    func testLastAssistantTextPicksLatestNonEmpty() {
        let msgs = [
            ChatMessage(role: .user, content: "do it"),
            ChatMessage(role: .assistant, content: "first"),
            ChatMessage(role: .assistant, content: "   "),     // empty/whitespace skipped
        ]
        XCTAssertEqual(TaskScheduler.lastAssistantText(msgs), "first")
    }

    func testLastAssistantTextNilWhenNoAssistant() {
        XCTAssertNil(TaskScheduler.lastAssistantText([ChatMessage(role: .user, content: "hi")]))
    }

    // MARK: - Full vs. capped result text

    func testFullLastAssistantTextIsNotTruncatedButTimelineRowIs() {
        let long = String(repeating: "y", count: 1000)
        let msgs = [ChatMessage(role: .user, content: "go"),
                    ChatMessage(role: .assistant, content: long)]
        // The Telegram/export path gets the whole answer…
        XCTAssertEqual(TaskScheduler.fullLastAssistantText(msgs)?.count, 1000)
        // …while the in-app timeline row stays capped at 280.
        XCTAssertEqual(TaskScheduler.lastAssistantText(msgs)?.count, 280)
    }

    func testFullLastAssistantTextNilWhenNoAssistant() {
        XCTAssertNil(TaskScheduler.fullLastAssistantText([ChatMessage(role: .user, content: "hi")]))
    }

    // MARK: - Stale-run reconciliation (startup heal)

    func testReconcileStaleRunsFailsOrphanedRunningRuns() {
        let now = Date()
        let tid = UUID()
        let running = TaskRun(taskId: tid, status: .running)
        let needs = TaskRun(taskId: tid, status: .needsApproval)
        let done = TaskRun(taskId: tid, status: .completed, summary: "ok")
        let swept = TaskScheduler.reconcileStaleRuns([running, needs, done], now: now)

        XCTAssertEqual(swept[0].status, .failed, "an interrupted .running run heals to .failed")
        XCTAssertNotNil(swept[0].finishedAt)
        XCTAssertNotNil(swept[0].summary)
        XCTAssertEqual(swept[1].status, .needsApproval, "a paused run is durable, not stale")
        XCTAssertEqual(swept[2].status, .completed, "terminal runs are untouched")
        XCTAssertEqual(swept[2].summary, "ok")
    }

    // MARK: - Clear-finished partition

    func testClearFinishedRemovesTerminalButKeepsActiveAndLive() {
        let tid = UUID()
        let active = TaskRun(taskId: tid, status: .completed)   // currently active → keep
        let done = TaskRun(taskId: tid, status: .completed)
        let failed = TaskRun(taskId: tid, status: .failed)
        let cancelled = TaskRun(taskId: tid, status: .cancelled)
        let needs = TaskRun(taskId: tid, status: .needsApproval)
        let running = TaskRun(taskId: tid, status: .running)
        let (keep, removed) = TaskScheduler.runsAfterClearingFinished(
            [active, done, failed, cancelled, needs, running], activeRunId: active.id)

        XCTAssertEqual(Set(removed.map(\.id)), Set([done.id, failed.id, cancelled.id]))
        XCTAssertEqual(Set(keep.map(\.id)), Set([active.id, needs.id, running.id]))
    }

    func testRunStatusIsTerminal() {
        XCTAssertTrue(RunStatus.completed.isTerminal)
        XCTAssertTrue(RunStatus.failed.isTerminal)
        XCTAssertTrue(RunStatus.cancelled.isTerminal)
        XCTAssertFalse(RunStatus.running.isTerminal)
        XCTAssertFalse(RunStatus.needsApproval.isTerminal)
        XCTAssertFalse(RunStatus.scheduled.isTerminal)
    }
}
