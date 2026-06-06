import XCTest
@testable import MLXCore

/// Pins the policy that decides when the agent loop surfaces the "output
/// truncated" notice.
///
/// Regression context: the notice used to be appended inside the per-iteration
/// stream loop, so a multi-step agent turn that hit the output cap on more than
/// one iteration stacked a notice per iteration ("⚠️ Output truncated…" twice in
/// a row). The fix shows it at most once, only at the turn boundary, and never
/// on an iteration that silently retries a truncated tool call — captured by the
/// pure `TruncationNotice.shouldShow(...)` policy here.
final class TruncationNoticeTests: XCTestCase {

    func testTextMentionsTheCapAndGuidance() {
        let t = TruncationNotice.text(maxTokens: 16384)
        XCTAssertTrue(t.contains("16384"), "notice should name the cap that was hit")
        XCTAssertTrue(t.lowercased().contains("truncated"))
        // Steers the user toward the fix (shorter steps / raise the cap).
        XCTAssertTrue(t.lowercased().contains("smaller") || t.contains("Settings"))
    }

    func testShownOnceAtTurnEnd() {
        XCTAssertTrue(TruncationNotice.shouldShow(maxTokensHit: true, turnEnding: true, willRetry: false))
    }

    func testNotShownMidTurn() {
        // Intermediate iteration (more tool calls coming) — no notice, so a
        // long multi-step turn can't stack one per round.
        XCTAssertFalse(TruncationNotice.shouldShow(maxTokensHit: true, turnEnding: false, willRetry: false))
    }

    func testNotShownWhenSilentlyRecovering() {
        // A truncated tool call triggers a silent retry-with-nudge; the user
        // shouldn't also see a scary truncation banner for it.
        XCTAssertFalse(TruncationNotice.shouldShow(maxTokensHit: true, turnEnding: true, willRetry: true))
    }

    func testNotShownWhenNothingWasTruncated() {
        XCTAssertFalse(TruncationNotice.shouldShow(maxTokensHit: false, turnEnding: true, willRetry: false))
    }
}
