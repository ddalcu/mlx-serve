import XCTest
@testable import MLXCore

/// After a tool result the request goes out as the transcript stands (#1085): a message added there in the
/// user role is one the user never wrote, and the model answers it as if they had.
@MainActor
final class ToolRoundHistoryTests: XCTestCase {
    private func call(_ id: String) -> ChatMessage {
        var m = ChatMessage(role: .assistant, content: "")
        m.toolCalls = [MLXCore.SerializedToolCall(id: id, name: "shell", arguments: "{\"command\":\"ls\"}")]
        return m
    }

    private func result(_ id: String) -> ChatMessage {
        var m = ChatMessage(role: .system, content: "a.txt")
        m.toolCallId = id
        return m
    }

    func testAHistoryEndingOnAToolResultIsSentAsItIs() {
        let msgs = [ChatMessage(role: .user, content: "list the folder"), call("a"), result("a")]
        let history = AgentEngine.buildAgentHistory(messages: msgs, contextLength: 131072, maxTokens: 4096)
        let sent = AgentEngine.requestMessages(systemPrompt: "sys", history: history)
        XCTAssertEqual(sent.map { $0["role"] as? String }, ["system", "user", "assistant", "tool"])
        XCTAssertEqual(sent.filter { ($0["role"] as? String) == "user" }.map { $0["content"] as? String }, ["list the folder"])
    }

    func testAResumeNeedsATranscriptThatEndsOnAToolResult() {
        let user = ChatMessage(role: .user, content: "list the folder")
        XCTAssertTrue(ChatTurnEngine.endsOnToolResult([user, call("a"), result("a")]))
        XCTAssertFalse(ChatTurnEngine.endsOnToolResult([user, call("a")]))
        XCTAssertFalse(ChatTurnEngine.endsOnToolResult([user]))
        XCTAssertFalse(ChatTurnEngine.endsOnToolResult([]))
    }
}
