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

    // MARK: - The engine has to be able to serve one

    func testTheEmbeddedDs4EngineIsNotOfferedAContinuation() {
        // ds4 renders its chat template INSIDE the engine, so there is nowhere
        // to append the prefill and the server refuses by name. A live button
        // over a guaranteed 400 is the dead-control class — the click buys an
        // error card and nothing else.
        XCTAssertFalse(ContinueReply.isEligible([user(), reply()], serverRunning: true,
                                                busy: false, engine: .dsv4))
    }

    func testEveryOtherEngineRendersThroughOurOwnJinjaAndIsOffered() {
        // MLX and the generic llama.cpp path both render here, where the
        // prefill has somewhere to go — `encodeChatViaLlama` takes the flag.
        for engine in [ServerEngine.mlx, .llama] {
            XCTAssertTrue(ContinueReply.isEligible([user(), reply()], serverRunning: true,
                                                   busy: false, engine: engine),
                          "\(engine) renders through our Jinja and can serve a continuation")
        }
    }

    func testAnUnknownEngineIsNotRefused() {
        // nil is "no model info yet", which the serverRunning gate covers.
        // Refusing on it would hide the button during every model switch.
        XCTAssertTrue(ContinueReply.isEligible([user(), reply()], serverRunning: true,
                                               busy: false, engine: nil))
    }
}

/// A continuation streams into a message that already holds a finished
/// generation, so everything the bubble reports about that message has to keep
/// up with it. The text does by construction (`updateLastMessage` appends); the
/// two things that describe it do not.
final class ContinuedReplyBookkeepingTests: XCTestCase {

    /// `usage` REPLACES the counts, so a 900-token reply finished by a 42-token
    /// continuation reported 42 in its footnote. Same class as the truncation
    /// notice the continuation already clears: the reply changed, and the data
    /// describing the reply has to change with it.
    func testTheFootnoteCountsBothHalvesOfAContinuedReply() throws {
        let source = SourceScan.source("AppState.swift", from: #filePath)
        let body = try XCTUnwrap(
            SourceScan.declarationBody(from: "func updateLastMessage", in: source),
            "updateLastMessage moved — repoint this scan")
        XCTAssertTrue(body.contains("addingCompletionTokens"), """
            updateLastMessage overwrites completionTokens unconditionally, so a \
            continued reply reports only the tokens of the sentence that \
            finished it.
            """)
    }

    /// The flag is only correct on the path that knows it is continuing — the
    /// agent loop never serves one, and a blanket `true` would double-count an
    /// ordinary turn's retry.
    func testTheAccumulationIsDrivenByThePlainTurnsOwnContinuingFlag() throws {
        let source = SourceScan.source("Services/ChatTurnEngine.swift", from: #filePath)
        XCTAssertTrue(source.contains("addingCompletionTokens: continuing"), """
            the usage write must read runPlainTurn's own `continuing` flag — a \
            literal there is either a permanent double-count or a permanent \
            reset.
            """)
        XCTAssertEqual(SourceScan.count("addingCompletionTokens:", in: source), 1, """
            exactly one usage site accumulates: the plain-chat stream. The agent \
            loop appends a fresh placeholder per tool round and never serves a \
            continuation.
            """)
    }

    /// The notice describes a reply that was cut. It is being un-cut.
    func testAContinuationClearsTheTruncationNotice() throws {
        let source = SourceScan.source("Services/ChatTurnEngine.swift", from: #filePath)
        let body = try XCTUnwrap(
            SourceScan.declarationBody(from: "func runPlainTurn", in: source),
            "runPlainTurn moved — repoint this scan")
        XCTAssertTrue(body.contains("clearTruncationNotice"), """
            a continued reply keeps "Stopped — hit the output limit" under a \
            paragraph that carried on.
            """)
    }
}
