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

    func testAnAnsweredResultWithinTheHistoryBudgetStaysComplete() {
        let msgs = [ChatMessage(role: .user, content: "read it"), call("a"), result("a", 7000), call("b"), result("b", 300)]
        XCTAssertEqual(toolResultLengths(msgs), [7000, 300])
    }

    func testMCPTaskBodySurvivesStorageAndLaterToolRounds() throws {
        let id = "test-mcp-history-\(UUID().uuidString)"
        let overflow = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".mlx-serve/tool-output/\(id).txt")
        defer { try? FileManager.default.removeItem(at: overflow) }
        let body = "Build the welcome section with the existing components."
        let output = "{\"metadata\":\"" + String(repeating: "x", count: 5000)
            + "\",\"body\":\"\(body)\"}"
        var firstCall = call(id)
        firstCall.toolCalls = [MLXCore.SerializedToolCall(id: id, name: "github__get_issue", arguments: "{\"issue_number\":6}")]
        var firstResult = result(id, 0)
        firstResult.content = AgentEngine.truncateWithOverflow(output, toolCallId: id, toolName: "github__get_issue")
        let messages = [ChatMessage(role: .user, content: "Explain the task."), firstCall, firstResult,
                        call("next"), result("next", 300), call("last"), result("last", 300)]
        let history = AgentEngine.buildAgentHistory(messages: messages, contextLength: 131072, maxTokens: 4096)
        let stored = try XCTUnwrap(history.first { ($0["tool_call_id"] as? String) == id }?["content"] as? String)
        let object = try JSONSerialization.jsonObject(with: Data(stored.utf8)) as? [String: Any]
        XCTAssertEqual(object?["body"] as? String, body)
    }
}
