import XCTest
@testable import MLXCore

/// A model's turn as the reader sees it: everything under their message up to
/// the next one. Where it can be cut, what its trash removes, and when its
/// end needs a footer of its own.
final class ChatTurnTests: XCTestCase {

    private func user(_ text: String) -> ChatMessage { ChatMessage(role: .user, content: text) }
    private func reply(_ text: String) -> ChatMessage { ChatMessage(role: .assistant, content: text) }

    /// A round that only thought (or was cut while thinking).
    private func thinking(_ text: String) -> ChatMessage {
        ChatMessage(role: .assistant, content: "", reasoningContent: text)
    }

    /// The model's message of a tool round: prose optional, `toolCalls` set.
    private func caller(_ text: String = "") -> ChatMessage {
        var m = ChatMessage(role: .assistant, content: text)
        m.toolCalls = [MLXCore.SerializedToolCall(id: "call_1", name: "shell", arguments: "{}")]
        return m
    }

    private func summary(_ text: String) -> ChatMessage {
        var m = ChatMessage(role: .assistant, content: text)
        m.isAgentSummary = true
        return m
    }

    /// The hidden tool RESULT the model reads.
    private func toolResult(_ text: String) -> ChatMessage {
        var m = ChatMessage(role: .system, content: text)
        m.toolCallId = "call_1"
        m.toolName = "shell"
        return m
    }

    private func failed() -> ChatMessage {
        var m = ChatMessage(role: .assistant, content: "")
        m.failedRetry = true
        return m
    }

    private func errorCard() -> ChatMessage {
        var m = ChatMessage(role: .assistant, content: "")
        m.errorNotice = ChatErrorNotice(kind: .generic, message: "boom")
        return m
    }

    /// A finished tool round, as `ChatTurnEngine` writes it.
    private func round(_ prose: String = "") -> [ChatMessage] {
        [caller(prose), summary("**shell**(cmd)"), summary("**shell** → ok"), toolResult("ok")]
    }

    // MARK: - Where a turn can be cut

    func testAUserMessageAndAPlainReplyAreBoundaries() {
        XCTAssertTrue(ChatTurn.isBoundary(user("q")))
        XCTAssertTrue(ChatTurn.isBoundary(reply("a")))
    }

    /// A tool call's results come AFTER it: cut on the caller and the model is
    /// handed a call with no answer.
    func testAReplyThatCallsToolsIsNotABoundaryEvenWithProse() {
        XCTAssertFalse(ChatTurn.isBoundary(caller("I'll check.")))
    }

    func testMachineryIsNeverABoundary() {
        XCTAssertFalse(ChatTurn.isBoundary(summary("**shell**(cmd)")))
        XCTAssertFalse(ChatTurn.isBoundary(toolResult("ok")))
        XCTAssertFalse(ChatTurn.isBoundary(failed()))
        XCTAssertFalse(ChatTurn.isBoundary(errorCard()))
        XCTAssertFalse(ChatTurn.isBoundary(thinking("hm")))
    }

    /// A generated picture is the model's answer for that round, delivered by
    /// a file: an empty `content` that still ends something.
    func testARowCarryingMediaIsABoundary() {
        var m = ChatMessage(role: .assistant, content: "")
        m.media = [ChatMediaRef(kind: .image, path: "/tmp/a.png", prompt: "a cat")]
        XCTAssertTrue(ChatTurn.isBoundary(m))
    }

    // MARK: - What the trash under a footer removes

    /// The reply's footer deletes the model's whole turn: the thinking, the
    /// card, the hidden result the card stands for, and the reply itself.
    func testDeletingAReplyTakesTheTurnAboveItBackToTheQuestion() {
        let messages = [user("q")] + [thinking("plan")] + round() + [reply("done")]
        let range = ChatTurn.deletionRange(endingAt: messages.last!.id, in: messages)
        XCTAssertEqual(range, 1..<messages.count)
    }

    func testTheReadersOwnMessageIsNeverPartOfTheTurn() {
        let messages = [user("q"), reply("a")]
        XCTAssertEqual(ChatTurn.deletionRange(endingAt: messages[1].id, in: messages), 1..<2)
    }

    /// A reply with a footer of its own is where the previous segment ended;
    /// deleting the next one stops there.
    func testDeletingStopsUnderThePreviousReply() {
        let messages = [user("q"), reply("first"), thinking("more"), reply("second")]
        XCTAssertEqual(ChatTurn.deletionRange(endingAt: messages[3].id, in: messages), 2..<4)
    }

    /// Prose that also called a tool is not a stop: its card sits below it and
    /// must go with it, or the model reads a call with no answer.
    func testAReplyThatCalledToolsGoesWithItsCard() {
        let messages = [user("q")] + round("Let me look.") + [reply("found it")]
        XCTAssertEqual(ChatTurn.deletionRange(endingAt: messages.last!.id, in: messages),
                       1..<messages.count)
    }

    /// The end footer of an interrupted turn deletes whatever the turn left.
    func testAnInterruptedTurnIsDeletedFromItsLastMessage() {
        let cutWhileThinking = [user("q"), thinking("…")]
        XCTAssertEqual(ChatTurn.deletionRange(endingAt: cutWhileThinking[1].id, in: cutWhileThinking), 1..<2)

        let cutAfterATool = [user("q")] + round()
        XCTAssertEqual(ChatTurn.deletionRange(endingAt: cutAfterATool.last!.id, in: cutAfterATool), 1..<5)
    }

    func testAnUnknownMessageDeletesNothing() {
        XCTAssertNil(ChatTurn.deletionRange(endingAt: UUID(), in: [user("q"), reply("a")]))
    }

    // MARK: - Whether the transcript's end needs a footer

    func testAReplyIsItsOwnEnd() {
        XCTAssertFalse(ChatTurn.needsEndFooter([user("q"), reply("a")], turnInFlight: false))
    }

    func testATurnCutShortNeedsOne() {
        XCTAssertTrue(ChatTurn.needsEndFooter([user("q"), thinking("…")], turnInFlight: false))
        XCTAssertTrue(ChatTurn.needsEndFooter([user("q")] + round(), turnInFlight: false),
                      "the last message is the hidden tool result; the reader sees the card")
        XCTAssertTrue(ChatTurn.needsEndFooter([user("q"), errorCard()], turnInFlight: false))
        XCTAssertTrue(ChatTurn.needsEndFooter([user("q"), failed()], turnInFlight: false))
    }

    func testTheReadersMessageNeedsNone() {
        XCTAssertFalse(ChatTurn.needsEndFooter([user("q"), reply("a"), user("more")], turnInFlight: false))
    }

    /// A running turn ends with something every few hundred milliseconds;
    /// a footer under it would be offering to delete a reply mid-sentence.
    func testNothingWhileTheTurnIsRunning() {
        XCTAssertFalse(ChatTurn.needsEndFooter([user("q"), thinking("…")], turnInFlight: true))
    }

    func testAnEmptyTranscriptNeedsNone() {
        XCTAssertFalse(ChatTurn.needsEndFooter([], turnInFlight: false))
    }

    /// Prose that called a tool keeps its footer (time, copy) but carries no
    /// trash, since its card sits below it — unless it is the transcript's
    /// last message, cut before its tools ran, with nothing below it.
    func testTheTrashSitsOnlyWhereTheTranscriptCanBeCut() {
        XCTAssertTrue(ChatTurn.footerDeletes(reply("a"), isLast: false))
        XCTAssertFalse(ChatTurn.footerDeletes(caller("Let me look."), isLast: false))
        XCTAssertTrue(ChatTurn.footerDeletes(caller("Let me look."), isLast: true))
        XCTAssertFalse(ChatTurn.needsEndFooter([user("q"), caller("Let me look.")], turnInFlight: false),
                       "its own footer is the end footer here")
    }

    // MARK: - The footer a reply carries

    /// The view's condition, in one place so the end footer is exactly its
    /// negation.
    func testOnlyAFinishedReplyWithProseCarriesItsOwnFooter() {
        XCTAssertTrue(ChatTurn.hasOwnFooter(reply("a")))
        XCTAssertTrue(ChatTurn.hasOwnFooter(caller("prose too")))
        XCTAssertFalse(ChatTurn.hasOwnFooter(thinking("…")))
        XCTAssertFalse(ChatTurn.hasOwnFooter(summary("**shell**(cmd)")))
        XCTAssertFalse(ChatTurn.hasOwnFooter(user("q")))
        var streaming = reply("half")
        streaming.isStreaming = true
        XCTAssertFalse(ChatTurn.hasOwnFooter(streaming))
    }
}
