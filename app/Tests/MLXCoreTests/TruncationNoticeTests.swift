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
        let t = TruncationNotice.text(cause: .maxTokens, maxTokens: 16384)
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

    // MARK: - Loop cuts (live 2026-08-05, under pi)

    func testLoopNoticeNamesNoCapBecauseNoCapWasHit() {
        let t = TruncationNotice.text(cause: .repetitionLoop, maxTokens: 16384)
        // The whole incident was a message naming an output limit neither side
        // set — the session had two thirds of its context free and pi sends no
        // max_tokens at all. Naming a cap here sends people to raise a setting
        // that was never the problem.
        XCTAssertFalse(t.contains("16384"))
        XCTAssertFalse(t.lowercased().contains("max tokens"))
        XCTAssertTrue(t.lowercased().contains("repeating"))
    }

    func testARepetitionLoopEndsTheTurnAndNothingElseDoes() {
        // The loop's text is already in the transcript (a streamed delta cannot
        // be retracted, so the server's trim never reaches a streaming client).
        // Continuing would send it back as history and the model resumes it.
        XCTAssertTrue(TruncationNotice.endsTurn(cause: .repetitionLoop))
        // A max_tokens cut keeps every existing recovery path — the reply was
        // fine and simply ran out of room.
        XCTAssertFalse(TruncationNotice.endsTurn(cause: .maxTokens))
        XCTAssertFalse(TruncationNotice.endsTurn(cause: nil))
    }

    func testCauseComesFromTheServersSiblingFieldAndDegradesToMaxTokens() {
        // "length" is the only OpenAI value for both cuts, so `finish_details`
        // is the discriminator. Absent (older server, another backend) must
        // read as .maxTokens — exactly the behaviour this replaced.
        XCTAssertEqual(APIClient.truncationCause(fromChoice: ["finish_reason": "length"]), .maxTokens)
        XCTAssertEqual(
            APIClient.truncationCause(fromChoice: [
                "finish_reason": "length",
                "finish_details": ["type": "repetition_loop"],
            ]),
            .repetitionLoop
        )
        // An unknown cause is still a truncation, not a loop.
        XCTAssertEqual(
            APIClient.truncationCause(fromChoice: [
                "finish_reason": "length",
                "finish_details": ["type": "something_new"],
            ]),
            .maxTokens
        )
        // Any other finish_reason is not a cut at all — including one carrying a
        // stale details object, which the server gates but a proxy might not.
        XCTAssertNil(APIClient.truncationCause(fromChoice: ["finish_reason": "stop"]))
        XCTAssertNil(APIClient.truncationCause(fromChoice: [
            "finish_reason": "stop",
            "finish_details": ["type": "repetition_loop"],
        ]))
        XCTAssertNil(APIClient.truncationCause(fromChoice: nil))
        XCTAssertNil(APIClient.truncationCause(fromChoice: [:]))
    }
}
