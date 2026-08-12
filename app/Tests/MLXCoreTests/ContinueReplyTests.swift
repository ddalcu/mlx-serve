import XCTest
@testable import MLXCore

/// When "Continue" is offered, and when it must not be.
///
/// The affordance sits under a reply the server cut off — `max_tokens` is the
/// case it exists for, and until now the only answer was to type "keep going"
/// and hope the model picked up where it left off. Continuing hands it its own
/// unfinished text instead, so it resumes mid-sentence.
///
/// The refusals matter more than the offer. A continuation appends into the
/// LAST message in the transcript, so anything that makes that message the
/// wrong target has to be caught here rather than discovered as text landing
/// in the wrong bubble.
final class ContinueReplyTests: XCTestCase {

    private func user(_ text: String = "hi") -> ChatMessage {
        ChatMessage(role: .user, content: text)
    }

    private func reply(_ text: String = "Once upon a",
                       streaming: Bool = false,
                       notice: TruncationNotice.Notice? = nil) -> ChatMessage {
        var m = ChatMessage(role: .assistant, content: text)
        m.isStreaming = streaming
        m.truncationNotice = notice
        return m
    }

    // MARK: - Offered

    func testACutReplyCanBeContinued() {
        let msgs = [user(), reply(notice: .init(cause: .maxTokens, maxTokens: 512))]
        XCTAssertTrue(ContinueReply.isEligible(msgs, serverRunning: true, busy: false))
    }

    func testAnOrdinaryFinishedReplyCanAlsoBeContinued() {
        // Not gated on the notice: "say more" is a reasonable thing to want
        // from a reply that ended on its own, and the model is free to stop
        // again immediately.
        XCTAssertTrue(ContinueReply.isEligible([user(), reply()], serverRunning: true, busy: false))
    }

    // MARK: - Refused

    func testNothingToContinueWhenTheLastTurnIsTheUsers() {
        XCTAssertFalse(ContinueReply.isEligible([reply(), user()], serverRunning: true, busy: false))
    }

    func testAnEmptyTranscriptOffersNothing() {
        XCTAssertFalse(ContinueReply.isEligible([], serverRunning: true, busy: false))
    }

    func testAStreamingReplyIsAlreadyBeingWritten() {
        XCTAssertFalse(ContinueReply.isEligible([user(), reply(streaming: true)],
                                                serverRunning: true, busy: false))
    }

    func testABusySessionRefuses() {
        // The turn would be superseded by its own continuation.
        XCTAssertFalse(ContinueReply.isEligible([user(), reply()], serverRunning: true, busy: true))
    }

    func testAStoppedServerRefuses() {
        XCTAssertFalse(ContinueReply.isEligible([user(), reply()], serverRunning: false, busy: false))
    }

    func testAnEmptyReplyHasNothingToResumeFrom() {
        // The server trims a prefill to nothing and renders an ordinary turn,
        // so the button would silently do something else than it says.
        XCTAssertFalse(ContinueReply.isEligible([user(), reply("")], serverRunning: true, busy: false))
        XCTAssertFalse(ContinueReply.isEligible([user(), reply("   \n")], serverRunning: true, busy: false))
    }

    func testAToolCallSummaryIsNotContinuable() {
        // A tool-call card is machinery, not prose — there is no sentence to
        // finish, and the server refuses this shape too.
        var summary = reply("Called search(...)")
        summary.isAgentSummary = true
        XCTAssertFalse(ContinueReply.isEligible([user(), summary], serverRunning: true, busy: false))
    }

    func testAFailedTurnNoticeIsNotContinuable() {
        // An error card is not the model's words; continuing would ask it to
        // finish OUR sentence about a server failure.
        var failed = reply("Error: context overflow")
        failed.failedRetry = true
        XCTAssertFalse(ContinueReply.isEligible([user(), failed], serverRunning: true, busy: false))
    }
}
