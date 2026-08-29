import XCTest
@testable import MLXCore

/// Unit tests for `ChatRowBuilder` — folding the agent's separate tool-call and
/// tool-result summary messages into one collapsible transcript row.
@MainActor
final class ChatRowBuilderTests: XCTestCase {

    private func call(_ name: String = "dbhub__execute_sql",
                      _ args: String = "sql: SELECT 1") -> ChatMessage {
        var m = ChatMessage(role: .assistant, content: "**\(name)**(\(args))")
        m.isAgentSummary = true
        return m
    }
    private func result(_ name: String = "dbhub__execute_sql",
                        _ out: String = "{\"ok\":true}") -> ChatMessage {
        var m = ChatMessage(role: .assistant, content: "**\(name)** → \(out)")
        m.isAgentSummary = true
        return m
    }
    private func normal(_ text: String) -> ChatMessage {
        ChatMessage(role: .assistant, content: text)
    }
    private func hiddenToolMsg() -> ChatMessage {
        var m = ChatMessage(role: .system, content: "raw tool output")
        m.toolCallId = "call_1"
        m.toolName = "dbhub__execute_sql"
        return m
    }

    func testCallAndResultFoldIntoOneRow() {
        let rows = ChatRowBuilder.rows(from: [call(), result()])
        XCTAssertEqual(rows.count, 1)
        guard case .toolCall(_, let results) = rows[0] else { return XCTFail("expected toolCall row") }
        XCTAssertEqual(results.count, 1)
    }

    func testRawToolResultMessageStaysHiddenButResultSummaryGroups() {
        // The role:.system message with a toolCallId is the hidden raw result; it
        // must be filtered out, and the call+result-summary still fold together.
        let rows = ChatRowBuilder.rows(from: [call(), hiddenToolMsg(), result()])
        XCTAssertEqual(rows.count, 1, "the hidden raw tool message must not produce a row")
        guard case .toolCall(_, let results) = rows[0] else { return XCTFail("expected toolCall row") }
        XCTAssertEqual(results.count, 1)
    }

    func testMultiCallRoundGroupsAllResultsUnderTheCall() {
        // One call summary (two tools) followed by two result summaries.
        var twoCalls = ChatMessage(role: .assistant,
            content: "**a**(x: 1)\n**b**(y: 2)")
        twoCalls.isAgentSummary = true
        let rows = ChatRowBuilder.rows(from: [twoCalls, result("a"), result("b")])
        XCTAssertEqual(rows.count, 1)
        guard case .toolCall(_, let results) = rows[0] else { return XCTFail("expected toolCall row") }
        XCTAssertEqual(results.count, 2)
    }

    func testNormalMessagesArePassedThroughAroundGroups() {
        let rows = ChatRowBuilder.rows(from: [normal("hi"), call(), result(), normal("done")])
        XCTAssertEqual(rows.count, 3)
        if case .message = rows[0] {} else { XCTFail("row 0 should be a message") }
        if case .toolCall = rows[1] {} else { XCTFail("row 1 should be a toolCall") }
        if case .message = rows[2] {} else { XCTFail("row 2 should be a message") }
    }

    func testLoneResultWithoutCallRendersAsMessage() {
        // Defensive: a result summary with no preceding call (e.g. a resumed task
        // run) must still render, not vanish.
        let rows = ChatRowBuilder.rows(from: [result()])
        XCTAssertEqual(rows.count, 1)
        if case .message = rows[0] {} else { XCTFail("lone result should fall back to a message row") }
    }

    func testCallWithNoResultYetIsAGroupWithEmptyResults() {
        // Mid-execution: the call summary exists, results not appended yet.
        var streaming = call()
        streaming.isStreaming = true
        let rows = ChatRowBuilder.rows(from: [streaming])
        XCTAssertEqual(rows.count, 1)
        guard case .toolCall(_, let results) = rows[0] else { return XCTFail("expected toolCall row") }
        XCTAssertTrue(results.isEmpty)
    }

    func testClassificationDiscriminatesCallVsResult() {
        XCTAssertTrue(ChatRowBuilder.isCallSummary(call()))
        XCTAssertFalse(ChatRowBuilder.isResultSummary(call()))
        XCTAssertTrue(ChatRowBuilder.isResultSummary(result()))
        XCTAssertFalse(ChatRowBuilder.isCallSummary(result()))
        // A denied result is still a result.
        var denied = ChatMessage(role: .assistant, content: "**shell** → denied by user")
        denied.isAgentSummary = true
        XCTAssertTrue(ChatRowBuilder.isResultSummary(denied))
        // Non-agent-summary content is neither.
        XCTAssertFalse(ChatRowBuilder.isCallSummary(normal("**a** → b")))
    }

    // MARK: - Tool-result summary mirrors the model 1:1

    /// The visible tool-result row must carry the ENTIRE model-facing content,
    /// not a short preview — regression guard for the old 500-char display cap
    /// that made large MCP results (e.g. dbhub__search_objects) look truncated
    /// in the UI even though the model received much more.
    func testToolResultSummaryShowsFullModelContent() {
        // A model-facing result well past the old 500-char display cap.
        let modelContent = String(repeating: "row\n", count: 400) + "FINAL_ROW"
        XCTAssertGreaterThan(modelContent.count, 1000)

        let summary = AgentEngine.toolResultSummary(name: "dbhub__search_objects",
                                                    modelContent: modelContent)

        XCTAssertTrue(summary.hasPrefix("**dbhub__search_objects** → "),
                      "must keep the `**name** → ` discriminator so the row folds")
        XCTAssertTrue(summary.contains("FINAL_ROW"),
                      "content past the old 500-char cap must be present (1:1 with the model)")
        XCTAssertTrue(summary.contains(modelContent),
                      "the visible summary must contain the model content verbatim")

        // And it still classifies as a result summary for ChatRowBuilder folding.
        var m = ChatMessage(role: .assistant, content: summary)
        m.isAgentSummary = true
        XCTAssertTrue(ChatRowBuilder.isResultSummary(m))
    }
}

/// Issue #227: rows are rebuilt on every body pass, so SwiftUI needs to be able
/// to see that nothing changed. That needs `ChatRow` to be `Equatable`.
@MainActor
final class ChatRowEquatableTests: XCTestCase {
    func testSameMessagesProduceEqualRows() {
        var call = ChatMessage(role: .assistant, content: "**t**(x)")
        call.isAgentSummary = true
        var result = ChatMessage(role: .assistant, content: "**t** → ok")
        result.isAgentSummary = true
        let msgs = [ChatMessage(role: .user, content: "hi"), call, result]
        XCTAssertEqual(ChatRowBuilder.rows(from: msgs), ChatRowBuilder.rows(from: msgs))
    }

    func testContentChangeMakesRowsUnequal() {
        let a = ChatMessage(role: .assistant, content: "x")
        var b = a
        b.content = "xy"
        XCTAssertNotEqual(ChatRowBuilder.rows(from: [a]), ChatRowBuilder.rows(from: [b]))
    }
}
