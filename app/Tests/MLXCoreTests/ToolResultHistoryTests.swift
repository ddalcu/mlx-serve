import XCTest
@testable import MLXCore

/// The model sees a tool result in full until it has answered it (#605): cutting the
/// file it just read is what sent it re-reading in ever smaller pieces.
@MainActor
final class ToolResultHistoryTests: XCTestCase {
    private func call(_ id: String) -> ChatMessage {
        var m = ChatMessage(role: .assistant, content: "")
        m.toolCalls = [MLXCore.SerializedToolCall(id: id, name: "readFile", arguments: "{\"path\":\"AGENTS.md\"}")]
        return m
    }

    private func result(_ id: String, _ chars: Int) -> ChatMessage {
        var m = ChatMessage(role: .system, content: String(repeating: "x", count: chars))
        m.toolCallId = id
        return m
    }

    private func toolResultLengths(_ messages: [ChatMessage]) -> [Int] {
        AgentEngine.buildAgentHistory(messages: messages, contextLength: 131072, maxTokens: 4096)
            .filter { ($0["role"] as? String) == "tool" }
            .map { ($0["content"] as? String)?.count ?? -1 }
    }

    func testAResultTheModelHasNotAnsweredGoesInFull() {
        let msgs = [ChatMessage(role: .user, content: "read it"), call("a"), result("a", 7000)]
        XCTAssertEqual(toolResultLengths(msgs), [7000])
    }

    func testAnAnsweredResultIsStillCompacted() {
        let msgs = [ChatMessage(role: .user, content: "read it"), call("a"), result("a", 7000), call("b"), result("b", 300)]
        XCTAssertEqual(toolResultLengths(msgs), [2000, 300])
    }
}
