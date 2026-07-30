import XCTest

@testable import MLXCore

/// Assistant-history reasoning round-trip (app side of the server fix):
/// reasoning the app received on an assistant turn goes back to the server as
/// `reasoning_content` on that history message. Templates that persist
/// reasoning across turns (laguna) otherwise see the empty <think></think>
/// nothink signature on every prior turn and stop thinking from turn 2 of a
/// session — while the Think toggle and the server flag all read correct.
@MainActor
final class ReasoningHistoryRoundTripTests: XCTestCase {

    // MARK: - Plain chat (ChatTurnEngine.plainHistoryDict)

    func testPlainChatDictCarriesAssistantReasoning() {
        let msg = ChatMessage(role: .assistant, content: "4", reasoningContent: "2+2 is 4")
        let d = ChatTurnEngine.plainHistoryDict(msg)
        XCTAssertEqual(d["reasoning_content"] as? String, "2+2 is 4")
        XCTAssertEqual(d["content"] as? String, "4")
    }

    func testPlainChatDictOmitsTheKeyWithoutReasoning() {
        // Absent key, never an explicit null — templates gate on `is string`.
        let plain = ChatTurnEngine.plainHistoryDict(ChatMessage(role: .assistant, content: "4"))
        XCTAssertNil(plain["reasoning_content"])
        // User messages never carry it, even if a decode put one there.
        let user = ChatTurnEngine.plainHistoryDict(
            ChatMessage(role: .user, content: "hi", reasoningContent: "stray"))
        XCTAssertNil(user["reasoning_content"])
    }

    func testPlainChatDictKeepsTheEmptyAssistantContentDrop() {
        // Characterization of the pre-existing behavior the helper was
        // extracted from: an empty assistant content drops the key.
        let d = ChatTurnEngine.plainHistoryDict(ChatMessage(role: .assistant, content: ""))
        XCTAssertNil(d["content"])
        XCTAssertEqual(d["role"] as? String, "assistant")
    }

    // MARK: - Agent loop (AgentEngine.buildAgentHistory)

    func testAgentHistoryCarriesReasoningOnPlainAssistantTurns() {
        var assistant = ChatMessage(role: .assistant, content: "4")
        assistant.reasoningContent = "2+2 is 4"
        let history = AgentEngine.buildAgentHistory(
            messages: [ChatMessage(role: .user, content: "What is 2+2?"), assistant],
            contextLength: 32768, maxTokens: 4096)
        let a = history.first { ($0["role"] as? String) == "assistant" }
        XCTAssertEqual(a?["reasoning_content"] as? String, "2+2 is 4")
    }

    func testAgentHistoryCarriesReasoningOnToolCallTurns() {
        // Agent traffic is where laguna starved live (pi, 2026-07-29): the
        // tool-call turns are the history, so they must carry it too.
        var assistant = ChatMessage(role: .assistant, content: "")
        // Qualified: the test target has its own legacy SerializedToolCall.
        assistant.toolCalls = [MLXCore.SerializedToolCall(id: "c1", name: "shell", arguments: "{}")]
        assistant.reasoningContent = "I should list the files first."
        var tool = ChatMessage(role: .system, content: "ok")
        tool.toolCallId = "c1"
        let history = AgentEngine.buildAgentHistory(
            messages: [ChatMessage(role: .user, content: "list files"), assistant, tool],
            contextLength: 32768, maxTokens: 4096)
        let a = history.first { ($0["tool_calls"] as? [[String: Any]]) != nil }
        XCTAssertEqual(a?["reasoning_content"] as? String, "I should list the files first.")
    }

    func testAgentHistoryOmitsTheKeyWithoutReasoning() {
        let history = AgentEngine.buildAgentHistory(
            messages: [
                ChatMessage(role: .user, content: "hi"),
                ChatMessage(role: .assistant, content: "hello"),
            ],
            contextLength: 32768, maxTokens: 4096)
        let a = history.first { ($0["role"] as? String) == "assistant" }
        XCTAssertNotNil(a)
        XCTAssertNil(a?["reasoning_content"])
    }

    func testTokenCostCountsReasoning() {
        // The budget walk must bill what will actually be sent, or a long
        // thinking trace silently blows the context budget it was never
        // counted against.
        let bare = ChatMessage(role: .assistant, content: "4")
        var thinking = bare
        thinking.reasoningContent = String(repeating: "reason ", count: 200)
        XCTAssertGreaterThan(
            AgentEngine.tokenCostForMessage(thinking),
            AgentEngine.tokenCostForMessage(bare))
    }
}
