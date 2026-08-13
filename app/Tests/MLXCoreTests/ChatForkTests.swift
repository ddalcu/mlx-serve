import XCTest
@testable import MLXCore

/// Forking a conversation at a message: the answer to "this went somewhere I
/// didn't want, but I don't want to lose it either".
///
/// Regenerate throws the reply away and keeps its versions on one message;
/// Continue extends it. A fork is the third answer — the conversation up to
/// that point, in a new thread, with the original left exactly as it was.
final class ChatForkTests: XCTestCase {

    private func user(_ text: String) -> ChatMessage { ChatMessage(role: .user, content: text) }
    private func assistant(_ text: String) -> ChatMessage { ChatMessage(role: .assistant, content: text) }

    /// A hidden tool RESULT: role `.system` carrying a `toolCallId`, which is
    /// what the transcript filters out and the model reads.
    private func toolResult(_ text: String) -> ChatMessage {
        var m = ChatMessage(role: .system, content: text)
        m.toolCallId = "call_1"
        m.toolName = "shell"
        return m
    }

    private func toolCaller(_ text: String) -> ChatMessage {
        var m = ChatMessage(role: .assistant, content: text)
        m.toolCalls = [MLXCore.SerializedToolCall(id: "call_1", name: "shell", arguments: "{\"cmd\":\"ls\"}")]
        return m
    }

    // MARK: - Where the cut lands

    func testTheForkKeepsEverythingThroughTheChosenMessage() {
        let messages = [user("one"), assistant("first answer"), user("two"), assistant("second answer")]
        let fork = ChatFork.prefix(messages, through: messages[1].id)
        XCTAssertEqual(fork.map(\.content), ["one", "first answer"])
    }

    /// Including the message itself, not up to it: forking at a reply is how
    /// you keep that reply and ask the next question differently.
    func testForkingAtAReplyKeepsThatReply() {
        let messages = [user("one"), assistant("keep me")]
        XCTAssertEqual(ChatFork.prefix(messages, through: messages[1].id).last?.content, "keep me")
    }

    /// Forking at your own question drops the answer, so the new thread is
    /// ready to be answered again.
    func testForkingAtAQuestionLeavesItUnanswered() {
        let messages = [user("one"), assistant("an answer"), user("two"), assistant("another")]
        let fork = ChatFork.prefix(messages, through: messages[2].id)
        XCTAssertEqual(fork.map(\.content), ["one", "an answer", "two"])
    }

    func testAnUnknownMessageForksNothing() {
        XCTAssertTrue(ChatFork.prefix([user("one")], through: UUID()).isEmpty)
    }

    // MARK: - The cut has to be somewhere the model can be handed

    /// A tool call's results come AFTER it, so cutting on the caller hands the
    /// model a call with no answer — which is the shape it apologises for, or
    /// re-issues. Trim back to the last place the transcript is whole.
    func testCuttingOnAToolCallTrimsBackToTheQuestion() {
        let messages = [user("list them"), toolCaller(""), toolResult("a\nb"), assistant("Two files.")]
        let fork = ChatFork.prefix(messages, through: messages[1].id)
        XCTAssertEqual(fork.map(\.content), ["list them"])
    }

    /// And cutting on the RESULT is the same cut — a bare tool result with no
    /// call above it is no better.
    func testCuttingOnAToolResultTrimsBackTheSameWay() {
        let messages = [user("list them"), toolCaller(""), toolResult("a\nb"), assistant("Two files.")]
        let fork = ChatFork.prefix(messages, through: messages[2].id)
        XCTAssertEqual(fork.map(\.content), ["list them"])
    }

    /// The answer AFTER the tool round is a clean boundary — the whole round
    /// comes with it.
    func testTheReplyAfterAToolRoundKeepsTheWholeRound() {
        let messages = [user("list them"), toolCaller(""), toolResult("a\nb"), assistant("Two files.")]
        let fork = ChatFork.prefix(messages, through: messages[3].id)
        XCTAssertEqual(fork.count, 4)
    }

    /// Our own failure card is not a place to resume from: it is not something
    /// the model said, and it is excluded from history anyway.
    func testAnErrorCardIsNotABoundary() {
        var failure = ChatMessage(role: .assistant, content: "")
        failure.failedRetry = true
        failure.errorNotice = ChatErrorNotice(kind: .generic, message: "boom")
        let messages = [user("one"), assistant("fine"), failure]
        XCTAssertEqual(ChatFork.prefix(messages, through: failure.id).map(\.content), ["one", "fine"])
    }

    // MARK: - What the menu offers

    func testAForkableMessageOffersTheCommand() {
        let messages = [user("one"), assistant("two")]
        XCTAssertTrue(ChatFork.isForkable(messages, at: messages[1].id))
    }

    /// A cut with nothing left after trimming is not offered — a command that
    /// does nothing when you pick it is the dead-control class.
    func testAForkThatWouldBeEmptyIsNotOffered() {
        let messages = [toolCaller(""), toolResult("out")]
        XCTAssertFalse(ChatFork.isForkable(messages, at: messages[1].id))
    }

    // MARK: - What the new thread inherits

    /// The fork must run under the same settings, or the next turn silently
    /// answers with a different agent, model or tool set than the turns above
    /// it in its own transcript.
    func testTheForkInheritsTheSettingsThatProducedTheTranscript() {
        var source = ChatSession(title: "Roofline math")
        source.agentId = UUID()
        source.mode = .agent
        source.enableThinking = true
        source.reasoningEffort = .high
        source.useMCP = true
        source.disabledTools = ["webSearch"]
        source.workingDirectory = "/tmp/work"

        let fork = ChatFork.session(from: source, messages: [user("one")])

        XCTAssertNotEqual(fork.id, source.id, "a fork is a new conversation")
        XCTAssertEqual(fork.agentId, source.agentId)
        XCTAssertEqual(fork.mode, source.mode)
        XCTAssertEqual(fork.enableThinking, source.enableThinking)
        XCTAssertEqual(fork.reasoningEffort, source.reasoningEffort)
        XCTAssertEqual(fork.useMCP, source.useMCP)
        XCTAssertEqual(fork.disabledTools, source.disabledTools)
        XCTAssertEqual(fork.workingDirectory, source.workingDirectory)
        XCTAssertEqual(fork.messages.map(\.content), ["one"])
    }

    /// The attached folder is NOT inherited: its security-scoped bookmark is
    /// keyed by the source session's id, so the path would come across without
    /// the grant that makes it readable — a folder chip pointing at something
    /// the sandbox refuses to open.
    func testTheAttachedFolderDoesNotComeAcross() {
        var source = ChatSession(title: "x")
        source.attachedFolderPath = "/Users/me/notes"
        XCTAssertNil(ChatFork.session(from: source, messages: [user("one")]).attachedFolderPath)
    }

    /// A task run's or a bridge's transcript can be forked INTO an ordinary
    /// chat, but the fork must not inherit being transient — those sessions
    /// are hidden from the sidebar and never persisted, so the fork would
    /// vanish.
    func testAForkIsAlwaysAnOrdinaryVisibleChat() {
        var source = ChatSession(title: "run")
        source.taskRunId = UUID()
        source.isExternalBridge = true
        let fork = ChatFork.session(from: source, messages: [user("one")])
        XCTAssertNil(fork.taskRunId)
        XCTAssertFalse(fork.isExternalBridge)
    }

    /// The fork IS that conversation up to the cut, so it keeps its name. A
    /// source still on its placeholder hands over a placeholder, which lets
    /// the auto-titler name the fork from its own content instead of pinning
    /// "New Chat" on it forever (`ChatSessionTitle`).
    func testTheForkKeepsTheNameAndAPlaceholderStaysAPlaceholder() {
        var named = ChatSession(title: "Roofline math")
        named.messages = [user("one")]
        XCTAssertEqual(ChatFork.session(from: named, messages: named.messages).title, "Roofline math")

        let fresh = ChatSession(title: ChatSessionTitle.placeholder(hasAgent: false))
        XCTAssertTrue(ChatSessionTitle.isPlaceholder(
            ChatFork.session(from: fresh, messages: [user("one")]).title))
    }
}
