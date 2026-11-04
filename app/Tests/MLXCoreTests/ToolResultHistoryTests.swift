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

    /// A strict API (DeepSeek, #950) 400s on a tool call without its result or a result without its call.
    func testATruncatedHistoryPairsEveryToolCallWithItsResult() {
        var plan = call("p")
        plan.content = "Plan: read the files one by one."
        var msgs = [ChatMessage(role: .user, content: "read them"), plan, result("p", 4000)]
        for i in 0..<12 { msgs += [call("c\(i)"), result("c\(i)", 4000)] }
        let history = AgentEngine.buildAgentHistory(messages: msgs, contextLength: 16384, maxTokens: 4096)
        var open = Set<String>()
        for m in history {
            if let id = m["tool_call_id"] as? String {
                XCTAssertNotNil(open.remove(id), "result \(id) has no call before it")
                continue
            }
            XCTAssertTrue(open.isEmpty, "calls \(open) have no result")
            for tc in (m["tool_calls"] as? [[String: Any]]) ?? [] { open.insert(tc["id"] as! String) }
        }
        XCTAssertTrue(open.isEmpty, "calls \(open) have no result")
    }

    func testAnAnsweredResultIsStillCompacted() {
        let msgs = [ChatMessage(role: .user, content: "read it"), call("a"), result("a", 7000), call("b"), result("b", 300)]
        XCTAssertEqual(toolResultLengths(msgs), [2000, 300])
    }
}
