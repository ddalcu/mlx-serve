import XCTest
import Foundation

// =============================================================================
// MARK: - Minimal type replicas (matching the fixed MLX Core agent harness)
// =============================================================================

enum TestRole: String, Codable {
    case system, user, assistant
}

struct SerializedToolCall: Codable, Equatable {
    let id: String
    let name: String
    let arguments: String // JSON string
}

struct TestChatMessage: Codable {
    var role: TestRole
    var content: String
    var isStreaming: Bool = false
    var isAgentSummary: Bool = false
    var toolCallId: String? = nil
    var toolName: String? = nil
    var toolCalls: [SerializedToolCall]? = nil
}

struct TestToolCall: Equatable {
    let id: String
    let name: String
    let arguments: [String: String]
}

enum TestSSEEvent: Equatable {
    case content(String)
    case reasoning(String)
    case toolCalls([TestToolCall])
    case done
}

// =============================================================================
// MARK: - Logic replicas (matching FIXED code)
// =============================================================================

/// Matches fixed ChatView.buildAgentHistory()
func buildAgentHistory(messages: [TestChatMessage]) -> [[String: Any]] {
    let firstUserIdx = messages.firstIndex { $0.role == .user && $0.toolCallId == nil }
    let windowStart = max(0, messages.count - 28)
    let window = Array(messages.suffix(28))
    let needsPin = firstUserIdx != nil && firstUserIdx! < windowStart

    let totalToolResults = window.filter { $0.toolCallId != nil }.count
    var toolResultsSeen = 0

    var history: [[String: Any]] = []

    if needsPin, let idx = firstUserIdx {
        history.append(["role": "user", "content": messages[idx].content])
    }

    for msg in window {
        if let callId = msg.toolCallId {
            let isRecent = toolResultsSeen >= totalToolResults - 2
            let limit = isRecent ? 2000 : 500
            toolResultsSeen += 1
            history.append([
                "role": "tool",
                "tool_call_id": callId,
                "content": String(msg.content.prefix(limit))
            ])
            continue
        }
        if msg.role == .system { continue }
        if msg.isAgentSummary { continue }

        // Assistant with tool_calls: include tool_calls in OpenAI format
        if msg.role == .assistant, let tcs = msg.toolCalls, !tcs.isEmpty {
            var dict: [String: Any] = ["role": "assistant"]
            let content = msg.content
                .replacingOccurrences(of: "<pad>", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            dict["content"] = content.isEmpty ? "" : content
            dict["tool_calls"] = tcs.map { tc -> [String: Any] in
                [
                    "id": tc.id,
                    "type": "function",
                    "function": [
                        "name": tc.name,
                        "arguments": tc.arguments
                    ] as [String: Any]
                ]
            }
            history.append(dict)
            continue
        }

        if msg.role == .assistant && msg.content.isEmpty { continue }
        var content = msg.content
            .replacingOccurrences(of: "<pad>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if msg.role == .assistant && content.count > 500 {
            content = String(content.prefix(500)) + "..."
        }
        if content.isEmpty { continue }
        history.append(["role": msg.role.rawValue, "content": content])
    }

    return history
}

/// Matches fixed APIClient.performStream() SSE parsing
func parseSSEStream(lines: [String]) -> [TestSSEEvent] {
    var events: [TestSSEEvent] = []
    var pendingToolCalls: [String: (name: String, args: String, id: String)] = [:]
    var hasToolCalls = false
    var emittedToolCalls = false

    for line in lines {
        guard line.hasPrefix("data: ") else { continue }
        let payload = String(line.dropFirst(6))
        if payload == "[DONE]" {
            events.append(.done)
            break
        }

        guard let jsonData = payload.data(using: .utf8),
              let chunk = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
            continue
        }

        guard let choices = chunk["choices"] as? [[String: Any]],
              let delta = choices.first?["delta"] as? [String: Any] else {
            continue
        }

        if let content = delta["content"] as? String, !content.isEmpty {
            events.append(.content(content))
        }
        if let reasoning = delta["reasoning_content"] as? String, !reasoning.isEmpty {
            events.append(.reasoning(reasoning))
        }

        if let tcArray = delta["tool_calls"] as? [[String: Any]] {
            hasToolCalls = true
            for tc in tcArray {
                let idx = "\(tc["index"] as? Int ?? 0)"
                var existing = pendingToolCalls[idx] ?? (name: "", args: "", id: "")
                if let id = tc["id"] as? String {
                    existing.id = id
                }
                if let function = tc["function"] as? [String: Any] {
                    if let name = function["name"] as? String { existing.name = name }
                    if let args = function["arguments"] as? String { existing.args += args }
                }
                pendingToolCalls[idx] = existing
            }
        }

        if let fr = choices.first?["finish_reason"] as? String, fr == "tool_calls" {
            var calls: [TestToolCall] = []
            for (_, tc) in pendingToolCalls.sorted(by: { $0.key < $1.key }) {
                var args: [String: String] = [:]
                if let data = tc.args.data(using: .utf8),
                   let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    for (k, v) in dict { args[k] = "\(v)" }
                }
                calls.append(TestToolCall(id: tc.id, name: tc.name, arguments: args))
            }
            if !calls.isEmpty {
                events.append(.toolCalls(calls))
                emittedToolCalls = true
            }
        }
    }

    // Fallback: emit if stream ended without finish_reason
    if hasToolCalls && !pendingToolCalls.isEmpty && !emittedToolCalls {
        var calls: [TestToolCall] = []
        for (_, tc) in pendingToolCalls.sorted(by: { $0.key < $1.key }) {
            if tc.name.isEmpty { continue }
            var args: [String: String] = [:]
            if let data = tc.args.data(using: .utf8),
               let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                for (k, v) in dict { args[k] = "\(v)" }
            }
            calls.append(TestToolCall(id: tc.id, name: tc.name, arguments: args))
        }
        if !calls.isEmpty {
            events.append(.toolCalls(calls))
        }
    }

    return events
}

/// Matches fixed runAgentLoop() state management
struct AgentLoopSimulator {
    var sessionMessages: [TestChatMessage]

    mutating func simulateToolCallResponse(
        modelContent: String,
        toolCalls: [TestToolCall],
        toolResults: [(id: String, name: String, output: String)]
    ) {
        // Add streaming assistant message
        var streamMsg = TestChatMessage(role: .assistant, content: "")
        streamMsg.isStreaming = true
        sessionMessages.append(streamMsg)

        // Accumulate model content via streaming
        let lastIdx = sessionMessages.count - 1
        sessionMessages[lastIdx].content = modelContent
        sessionMessages[lastIdx].isStreaming = false

        // Store tool_calls on the assistant message (FIX)
        if !toolCalls.isEmpty {
            sessionMessages[lastIdx].toolCalls = toolCalls.map { tc in
                let argsJson = (try? JSONSerialization.data(withJSONObject: tc.arguments))
                    .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                return SerializedToolCall(id: tc.id, name: tc.name, arguments: argsJson)
            }
        }

        // Tool call summary as display-only message (FIX: NOT appended to assistant content)
        if !toolCalls.isEmpty {
            let callSummary = toolCalls.map { tc in
                let args = tc.arguments.map { "\($0.key): \($0.value.prefix(80))" }.joined(separator: ", ")
                return "**\(tc.name)**(\(args))"
            }.joined(separator: "\n")
            var summaryMsg = TestChatMessage(role: .assistant, content: callSummary)
            summaryMsg.isAgentSummary = true
            sessionMessages.append(summaryMsg)
        }

        // Display-only tool result summaries
        for tr in toolResults {
            var resultMsg = TestChatMessage(role: .assistant, content: "**\(tr.name)** -> \(String(tr.output.prefix(500)))")
            resultMsg.isAgentSummary = true
            sessionMessages.append(resultMsg)
        }

        // Tool result messages
        for tr in toolResults {
            var toolMsg = TestChatMessage(role: .system, content: "")
            toolMsg.toolCallId = tr.id
            toolMsg.toolName = tr.name
            toolMsg.content = String(tr.output.prefix(2000))
            sessionMessages.append(toolMsg)
        }
    }
}

// =============================================================================
// MARK: - Helpers
// =============================================================================

func makeSSELine(_ json: [String: Any]) -> String {
    let data = try! JSONSerialization.data(withJSONObject: json)
    return "data: \(String(data: data, encoding: .utf8)!)"
}

func makeContentDelta(_ text: String, finishReason: String? = nil) -> String {
    var choice: [String: Any] = ["delta": ["content": text] as [String: Any], "index": 0]
    choice["finish_reason"] = finishReason as Any? ?? NSNull()
    return makeSSELine(["choices": [choice]])
}

func makeToolCallDelta(index: Int, id: String? = nil, name: String? = nil, arguments: String? = nil, finishReason: String? = nil) -> String {
    var funcDict: [String: Any] = [:]
    if let name = name { funcDict["name"] = name }
    if let args = arguments { funcDict["arguments"] = args }
    var tcDict: [String: Any] = ["index": index]
    if let id = id { tcDict["id"] = id }
    if !funcDict.isEmpty { tcDict["function"] = funcDict }
    var choice: [String: Any] = ["delta": ["tool_calls": [tcDict]] as [String: Any], "index": 0]
    choice["finish_reason"] = finishReason as Any? ?? NSNull()
    return makeSSELine(["choices": [choice]])
}

func makeFinishDelta(reason: String) -> String {
    let choice: [String: Any] = ["delta": [:] as [String: Any], "finish_reason": reason, "index": 0]
    return makeSSELine(["choices": [choice]])
}

/// Validates conversation structure. Returns list of violations (empty = valid).
func validateConversationStructure(_ history: [[String: Any]]) -> [String] {
    var violations: [String] = []

    for (i, msg) in history.enumerated() {
        let role = msg["role"] as? String ?? ""

        // Rule 1: tool messages must be preceded by assistant with tool_calls
        if role == "tool" {
            let toolCallId = msg["tool_call_id"] as? String ?? "?"
            if i == 0 {
                violations.append("[\(i)] tool (id=\(toolCallId)) is the first message")
                continue
            }
            var foundAssistant = false
            for j in stride(from: i - 1, through: 0, by: -1) {
                let prevRole = history[j]["role"] as? String ?? ""
                if prevRole == "assistant" {
                    if history[j]["tool_calls"] == nil {
                        violations.append("[\(i)] tool (id=\(toolCallId)) preceded by assistant[\(j)] WITHOUT tool_calls")
                    }
                    foundAssistant = true
                    break
                }
                if prevRole != "tool" { break }
            }
            if !foundAssistant {
                violations.append("[\(i)] tool (id=\(toolCallId)) has no preceding assistant")
            }
        }

        // Rule 2: No fabricated markdown in assistant content
        if role == "assistant" {
            let content = msg["content"] as? String ?? ""
            let markdownToolPattern = try! NSRegularExpression(pattern: #"\*\*\w+\*\*\("#)
            let range = NSRange(content.startIndex..., in: content)
            if markdownToolPattern.firstMatch(in: content, range: range) != nil {
                violations.append("[\(i)] assistant content has fabricated markdown tool summary")
            }
        }

        // Rule 3: tool_call_id must match an id in preceding assistant's tool_calls
        if role == "tool", let toolCallId = msg["tool_call_id"] as? String {
            for j in stride(from: i - 1, through: 0, by: -1) {
                if (history[j]["role"] as? String) == "assistant" {
                    if let tcs = history[j]["tool_calls"] as? [[String: Any]] {
                        let ids = tcs.compactMap { $0["id"] as? String }
                        if !ids.contains(toolCallId) {
                            violations.append("[\(i)] tool_call_id '\(toolCallId)' not in assistant[\(j)].tool_calls ids: \(ids)")
                        }
                    }
                    break
                }
                if (history[j]["role"] as? String) != "tool" { break }
            }
        }

        // Rule 4: Assistant with tool_calls must be followed by tool messages
        if role == "assistant", let tcs = msg["tool_calls"] as? [[String: Any]], !tcs.isEmpty {
            let nextIdx = i + 1
            if nextIdx >= history.count {
                violations.append("[\(i)] assistant has tool_calls but no following tool messages")
            } else if (history[nextIdx]["role"] as? String) != "tool" {
                violations.append("[\(i)] assistant has tool_calls but next is '\(history[nextIdx]["role"] as? String ?? "?")'")
            }
        }
    }

    return violations
}

// =============================================================================
// MARK: - Tests: Core bug fixes (regression tests)
// =============================================================================

final class AgentHarnessTests: XCTestCase {

    // Fix 1: tool_calls included in history
    func testBuildAgentHistory_IncludesToolCalls() {
        var sim = AgentLoopSimulator(sessionMessages: [
            TestChatMessage(role: .user, content: "List files")
        ])
        sim.simulateToolCallResponse(
            modelContent: "I'll list the files for you.",
            toolCalls: [TestToolCall(id: "call_1712345_0", name: "shell", arguments: ["command": "ls -la"])],
            toolResults: [(id: "call_1712345_0", name: "shell", output: "file1.txt\nfile2.txt\nREADME.md")]
        )

        let history = buildAgentHistory(messages: sim.sessionMessages)
        let assistantMsgs = history.filter { ($0["role"] as? String) == "assistant" }
        XCTAssertFalse(assistantMsgs.isEmpty)

        let firstAssistant = assistantMsgs[0]
        XCTAssertNotNil(firstAssistant["tool_calls"],
            "Assistant message must include tool_calls for server chat template")
        let tcs = firstAssistant["tool_calls"] as! [[String: Any]]
        XCTAssertEqual(tcs.count, 1)
        XCTAssertEqual((tcs[0]["function"] as? [String: Any])?["name"] as? String, "shell")
        XCTAssertEqual(tcs[0]["id"] as? String, "call_1712345_0")
    }

    // Fix 1: tool responses are no longer orphaned
    func testBuildAgentHistory_NoOrphanedToolResponses() {
        var sim = AgentLoopSimulator(sessionMessages: [
            TestChatMessage(role: .user, content: "What time is it?")
        ])
        sim.simulateToolCallResponse(
            modelContent: "Let me check.",
            toolCalls: [TestToolCall(id: "call_0", name: "shell", arguments: ["command": "date"])],
            toolResults: [(id: "call_0", name: "shell", output: "Sat Apr 4 10:30:00 PDT 2026")]
        )

        let history = buildAgentHistory(messages: sim.sessionMessages)
        let violations = validateConversationStructure(history)
        XCTAssertTrue(violations.isEmpty,
            "Tool responses should not be orphaned:\n" + violations.joined(separator: "\n"))
    }

    // Fix 2: Assistant content is model's actual output, not fabricated summary
    func testAssistantContentPreserved() {
        let modelOutput = "I'll list the files for you."
        var sim = AgentLoopSimulator(sessionMessages: [
            TestChatMessage(role: .user, content: "List files")
        ])
        sim.simulateToolCallResponse(
            modelContent: modelOutput,
            toolCalls: [TestToolCall(id: "call_0", name: "shell", arguments: ["command": "ls -la"])],
            toolResults: [(id: "call_0", name: "shell", output: "file1.txt")]
        )

        let history = buildAgentHistory(messages: sim.sessionMessages)
        let assistantContent = (history.first { ($0["role"] as? String) == "assistant" }?["content"] as? String) ?? ""

        XCTAssertFalse(assistantContent.contains("**shell**"),
            "Content should not contain fabricated markdown. Got: '\(assistantContent)'")
        XCTAssertEqual(assistantContent, modelOutput)
    }

    // Fix 2: Multi-round content stays clean
    func testMultiRoundContentNotCorrupted() {
        var sim = AgentLoopSimulator(sessionMessages: [
            TestChatMessage(role: .user, content: "Set up a new project")
        ])
        sim.simulateToolCallResponse(
            modelContent: "Creating directory.",
            toolCalls: [TestToolCall(id: "call_0", name: "shell", arguments: ["command": "mkdir proj"])],
            toolResults: [(id: "call_0", name: "shell", output: "")]
        )
        sim.simulateToolCallResponse(
            modelContent: "Writing main file.",
            toolCalls: [TestToolCall(id: "call_1", name: "writeFile", arguments: ["path": "main.py", "content": "print('hello')"])],
            toolResults: [(id: "call_1", name: "writeFile", output: "File written")]
        )

        let history = buildAgentHistory(messages: sim.sessionMessages)
        let corruptedCount = history.filter {
            ($0["role"] as? String) == "assistant" &&
            (($0["content"] as? String) ?? "").contains("**")
        }.count
        XCTAssertEqual(corruptedCount, 0, "No assistant messages should have fabricated markdown")
    }

    // Fix 3: SSE tool calls emitted exactly once
    func testSSEToolCallsEmittedOnce() {
        let lines = [
            makeContentDelta("Let me check."),
            makeToolCallDelta(index: 0, id: "call_1712345_0", name: "shell", arguments: ""),
            makeToolCallDelta(index: 0, arguments: "{\"command\""),
            makeToolCallDelta(index: 0, arguments: ": \"ls -la\"}"),
            makeFinishDelta(reason: "tool_calls"),
            "data: [DONE]"
        ]

        let events = parseSSEStream(lines: lines)
        let count = events.filter { if case .toolCalls = $0 { return true }; return false }.count
        XCTAssertEqual(count, 1, "Tool calls must be emitted exactly once, got \(count)")
    }

    // Fix 4: Server tool call IDs preserved
    func testServerToolCallIdsPreserved() {
        let serverId = "call_1712345678_0"
        let lines = [
            makeToolCallDelta(index: 0, id: serverId, name: "shell", arguments: "{\"command\": \"ls\"}"),
            makeFinishDelta(reason: "tool_calls"),
            "data: [DONE]"
        ]

        let events = parseSSEStream(lines: lines)
        let calls = events.compactMap { e -> [TestToolCall]? in
            if case .toolCalls(let c) = e { return c }; return nil
        }.first ?? []
        XCTAssertEqual(calls.first?.id, serverId, "Server ID must be preserved")
    }

    // Fix 5: ChatMessage stores tool_calls
    func testChatMessageStoresToolCalls() {
        var msg = TestChatMessage(role: .assistant, content: "Checking...")
        msg.toolCalls = [SerializedToolCall(id: "call_1", name: "shell", arguments: "{\"command\":\"ls\"}")]

        let history = buildAgentHistory(messages: [msg])
        XCTAssertNotNil(history.first?["tool_calls"],
            "History should include tool_calls from the ChatMessage")
    }

    // Integration: full conversation produces valid history
    func testFullConversationProducesValidHistory() {
        var sim = AgentLoopSimulator(sessionMessages: [
            TestChatMessage(role: .user, content: "Find Python files and count them")
        ])
        sim.simulateToolCallResponse(
            modelContent: "I'll search for Python files.",
            toolCalls: [TestToolCall(id: "call_0", name: "shell", arguments: ["command": "find . -name '*.py'"])],
            toolResults: [(id: "call_0", name: "shell", output: "./main.py\n./test.py\n./utils.py")]
        )
        sim.simulateToolCallResponse(
            modelContent: "Now let me count them.",
            toolCalls: [TestToolCall(id: "call_1", name: "shell", arguments: ["command": "find . -name '*.py' | wc -l"])],
            toolResults: [(id: "call_1", name: "shell", output: "3")]
        )

        let history = buildAgentHistory(messages: sim.sessionMessages)
        let roles = history.map { $0["role"] as? String ?? "?" }
        XCTAssertEqual(roles, ["user", "assistant", "tool", "assistant", "tool"])

        let violations = validateConversationStructure(history)
        XCTAssertTrue(violations.isEmpty,
            "\(violations.count) violations:\n" + violations.map { "  - \($0)" }.joined(separator: "\n"))
    }

    // Pad retry with fixed history should have a chance of succeeding
    func testPadRetryWithFixedHistoryIsStructurallySound() {
        var sim = AgentLoopSimulator(sessionMessages: [
            TestChatMessage(role: .user, content: "Do something complex")
        ])
        sim.simulateToolCallResponse(
            modelContent: "Working on it.",
            toolCalls: [TestToolCall(id: "call_0", name: "shell", arguments: ["command": "echo hello"])],
            toolResults: [(id: "call_0", name: "shell", output: "hello")]
        )

        let history = buildAgentHistory(messages: sim.sessionMessages)
        let violations = validateConversationStructure(history)
        XCTAssertTrue(violations.isEmpty,
            "Fixed history is structurally sound, retrying has a chance of success")
    }
}

// =============================================================================
// MARK: - Tests: Expected correct behavior
// =============================================================================

final class CorrectBehaviorTests: XCTestCase {

    func testSingleToolCallRoundTrip() {
        var sim = AgentLoopSimulator(sessionMessages: [
            TestChatMessage(role: .user, content: "List files")
        ])
        sim.simulateToolCallResponse(
            modelContent: "I'll list the files.",
            toolCalls: [TestToolCall(id: "call_99_0", name: "shell", arguments: ["command": "ls -la"])],
            toolResults: [(id: "call_99_0", name: "shell", output: "file1.txt\nfile2.txt")]
        )

        let history = buildAgentHistory(messages: sim.sessionMessages)
        let violations = validateConversationStructure(history)
        XCTAssertTrue(violations.isEmpty,
            "Violations:\n" + violations.joined(separator: "\n"))

        let roles = history.map { $0["role"] as? String ?? "?" }
        XCTAssertEqual(roles, ["user", "assistant", "tool"])

        let assistant = history[1]
        XCTAssertNotNil(assistant["tool_calls"])
        XCTAssertEqual(assistant["content"] as? String, "I'll list the files.")
        let tcs = assistant["tool_calls"] as! [[String: Any]]
        XCTAssertEqual(tcs[0]["id"] as? String, "call_99_0")
    }

    func testMultiRoundToolCalling() {
        var sim = AgentLoopSimulator(sessionMessages: [
            TestChatMessage(role: .user, content: "Set up a project")
        ])
        sim.simulateToolCallResponse(
            modelContent: "Creating directory.",
            toolCalls: [TestToolCall(id: "call_1_0", name: "shell", arguments: ["command": "mkdir proj"])],
            toolResults: [(id: "call_1_0", name: "shell", output: "")]
        )
        sim.simulateToolCallResponse(
            modelContent: "Writing main file.",
            toolCalls: [TestToolCall(id: "call_2_0", name: "writeFile", arguments: ["path": "proj/main.py", "content": "print('hi')"])],
            toolResults: [(id: "call_2_0", name: "writeFile", output: "OK")]
        )
        sim.simulateToolCallResponse(
            modelContent: "Verifying.",
            toolCalls: [TestToolCall(id: "call_3_0", name: "shell", arguments: ["command": "cat proj/main.py"])],
            toolResults: [(id: "call_3_0", name: "shell", output: "print('hi')")]
        )

        let history = buildAgentHistory(messages: sim.sessionMessages)
        let violations = validateConversationStructure(history)
        XCTAssertTrue(violations.isEmpty,
            "3-round history:\n" + violations.joined(separator: "\n"))

        let roles = history.map { $0["role"] as? String ?? "?" }
        XCTAssertEqual(roles, ["user", "assistant", "tool", "assistant", "tool", "assistant", "tool"])

        for msg in history where (msg["role"] as? String) == "assistant" {
            XCTAssertNotNil(msg["tool_calls"])
        }
    }

    func testToolCallIdConsistency() {
        var sim = AgentLoopSimulator(sessionMessages: [
            TestChatMessage(role: .user, content: "Do task")
        ])
        sim.simulateToolCallResponse(
            modelContent: "",
            toolCalls: [TestToolCall(id: "call_ABC_0", name: "shell", arguments: ["command": "ls"])],
            toolResults: [(id: "call_ABC_0", name: "shell", output: "file.txt")]
        )

        let history = buildAgentHistory(messages: sim.sessionMessages)
        let assistant = history.first { ($0["role"] as? String) == "assistant" }!
        let tcs = assistant["tool_calls"] as! [[String: Any]]
        let tool = history.first { ($0["role"] as? String) == "tool" }!

        XCTAssertEqual(tcs[0]["id"] as? String, tool["tool_call_id"] as? String)
    }

    func testEmptyContentWithToolCalls() {
        var sim = AgentLoopSimulator(sessionMessages: [
            TestChatMessage(role: .user, content: "Run ls")
        ])
        sim.simulateToolCallResponse(
            modelContent: "",
            toolCalls: [TestToolCall(id: "call_0", name: "shell", arguments: ["command": "ls"])],
            toolResults: [(id: "call_0", name: "shell", output: "file.txt")]
        )

        let history = buildAgentHistory(messages: sim.sessionMessages)
        let assistant = history.first { ($0["role"] as? String) == "assistant" }
        XCTAssertNotNil(assistant, "Assistant with empty content but tool_calls should be included")
        XCTAssertNotNil(assistant?["tool_calls"])
        XCTAssertEqual(assistant?["content"] as? String, "")
    }

    func testConsecutiveToolMessages() {
        var sim = AgentLoopSimulator(sessionMessages: [
            TestChatMessage(role: .user, content: "Read two files")
        ])
        sim.simulateToolCallResponse(
            modelContent: "",
            toolCalls: [
                TestToolCall(id: "call_1_0", name: "readFile", arguments: ["path": "a.txt"]),
                TestToolCall(id: "call_1_1", name: "readFile", arguments: ["path": "b.txt"])
            ],
            toolResults: [
                (id: "call_1_0", name: "readFile", output: "content A"),
                (id: "call_1_1", name: "readFile", output: "content B")
            ]
        )

        let history = buildAgentHistory(messages: sim.sessionMessages)
        let roles = history.map { $0["role"] as? String ?? "?" }
        XCTAssertEqual(roles, ["user", "assistant", "tool", "tool"])

        let violations = validateConversationStructure(history)
        XCTAssertTrue(violations.isEmpty,
            "Parallel tools:\n" + violations.joined(separator: "\n"))

        // Both tool_call_ids should match assistant's tool_calls
        let tcs = (history[1]["tool_calls"] as! [[String: Any]])
        let assistantIds = Set(tcs.map { $0["id"] as! String })
        let toolIds = Set(history.filter { ($0["role"] as? String) == "tool" }.map { $0["tool_call_id"] as! String })
        XCTAssertEqual(assistantIds, toolIds)
    }
}

// =============================================================================
// MARK: - Tests: SSE parsing edge cases
// =============================================================================

final class SSEParsingEdgeCaseTests: XCTestCase {

    func testSSE_ArgumentsSplitAcrossManyChunks() {
        let fullArgs = "{\"command\": \"find /Users/david/projects -name '*.swift' -type f | head -20\"}"
        var lines: [String] = [
            makeToolCallDelta(index: 0, id: "call_1_0", name: "shell", arguments: "")
        ]
        var pos = fullArgs.startIndex
        while pos < fullArgs.endIndex {
            let end = fullArgs.index(pos, offsetBy: 5, limitedBy: fullArgs.endIndex) ?? fullArgs.endIndex
            lines.append(makeToolCallDelta(index: 0, arguments: String(fullArgs[pos..<end])))
            pos = end
        }
        lines.append(makeFinishDelta(reason: "tool_calls"))
        lines.append("data: [DONE]")

        let events = parseSSEStream(lines: lines)
        let calls = events.compactMap { e -> [TestToolCall]? in
            if case .toolCalls(let c) = e { return c }; return nil
        }.first ?? []

        XCTAssertEqual(calls.count, 1)
        XCTAssertTrue(calls[0].arguments["command"]?.contains("find /Users/david/projects") ?? false)
    }

    func testSSE_EmptyArguments() {
        let lines = [
            makeToolCallDelta(index: 0, id: "call_1_0", name: "shell", arguments: "{}"),
            makeFinishDelta(reason: "tool_calls"),
            "data: [DONE]"
        ]
        let calls = parseSSEStream(lines: lines).compactMap { e -> [TestToolCall]? in
            if case .toolCalls(let c) = e { return c }; return nil
        }.first ?? []
        XCTAssertEqual(calls.count, 1)
        XCTAssertTrue(calls[0].arguments.isEmpty)
    }

    func testSSE_ArgumentsWithSpecialCharacters() {
        let lines = [
            makeToolCallDelta(index: 0, id: "call_1_0", name: "writeFile", arguments: ""),
            makeToolCallDelta(index: 0, arguments: "{\"path\": \"/tmp/test.py\", \"content\": \"print(\\\"hello\\\\nworld\\\")\"}"),
            makeFinishDelta(reason: "tool_calls"),
            "data: [DONE]"
        ]
        let calls = parseSSEStream(lines: lines).compactMap { e -> [TestToolCall]? in
            if case .toolCalls(let c) = e { return c }; return nil
        }.first ?? []
        XCTAssertEqual(calls[0].arguments["path"], "/tmp/test.py")
        XCTAssertNotNil(calls[0].arguments["content"])
    }

    func testSSE_ArgumentsWithNestedJSON() {
        let lines = [
            makeToolCallDelta(index: 0, id: "call_1_0", name: "writeFile", arguments: ""),
            makeToolCallDelta(index: 0, arguments: "{\"path\": \"config.json\", \"content\": \"{\\\"key\\\": \\\"value\\\"}\"}"),
            makeFinishDelta(reason: "tool_calls"),
            "data: [DONE]"
        ]
        let calls = parseSSEStream(lines: lines).compactMap { e -> [TestToolCall]? in
            if case .toolCalls(let c) = e { return c }; return nil
        }.first ?? []
        XCTAssertEqual(calls[0].arguments["path"], "config.json")
        XCTAssertNotNil(calls[0].arguments["content"])
    }

    func testSSE_ContentBeforeToolCalls() {
        let lines = [
            makeContentDelta("I need to check the filesystem first. "),
            makeContentDelta("Let me look at the files."),
            makeToolCallDelta(index: 0, id: "call_1_0", name: "shell", arguments: "{\"command\": \"ls\"}"),
            makeFinishDelta(reason: "tool_calls"),
            "data: [DONE]"
        ]

        let events = parseSSEStream(lines: lines)
        var contentParts: [String] = []
        var hitToolCalls = false
        for e in events {
            switch e {
            case .content(let t):
                XCTAssertFalse(hitToolCalls, "Content should come before tool calls")
                contentParts.append(t)
            case .toolCalls: hitToolCalls = true
            default: break
            }
        }
        XCTAssertEqual(contentParts.joined(), "I need to check the filesystem first. Let me look at the files.")
        XCTAssertTrue(hitToolCalls)
    }

    func testSSE_ReasoningWithToolCalls() {
        let lines = [
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Thinking about what to do...\"},\"finish_reason\":null}]}",
            makeToolCallDelta(index: 0, id: "call_1_0", name: "shell", arguments: "{\"command\": \"ls\"}"),
            makeFinishDelta(reason: "tool_calls"),
            "data: [DONE]"
        ]
        let events = parseSSEStream(lines: lines)
        XCTAssertTrue(events.contains { if case .reasoning = $0 { return true }; return false })
        XCTAssertTrue(events.contains { if case .toolCalls = $0 { return true }; return false })
    }

    func testSSE_NoFinishReason_StreamDropped() {
        let lines = [
            makeToolCallDelta(index: 0, id: "call_1_0", name: "shell", arguments: "{\"command\": \"ls\"}"),
        ]
        let events = parseSSEStream(lines: lines)
        let count = events.filter { if case .toolCalls = $0 { return true }; return false }.count
        XCTAssertEqual(count, 1, "Fallback should emit tool calls even without finish_reason")
    }

    func testSSE_FinishReasonStop_NoToolCalls() {
        let lines = [
            makeContentDelta("Hello, how can I help?"),
            makeFinishDelta(reason: "stop"),
            "data: [DONE]"
        ]
        let count = parseSSEStream(lines: lines).filter { if case .toolCalls = $0 { return true }; return false }.count
        XCTAssertEqual(count, 0)
    }

    func testSSE_MalformedJSONSkipped() {
        let lines = [
            "data: not valid json",
            "data: {\"invalid\": true}",
            makeContentDelta("Valid content"),
            "data: [DONE]"
        ]
        let count = parseSSEStream(lines: lines).filter { if case .content = $0 { return true }; return false }.count
        XCTAssertEqual(count, 1)
    }

    func testSSE_NonDataLinesIgnored() {
        let lines = [
            ": this is a comment",
            "event: ping",
            "retry: 3000",
            makeContentDelta("Real content"),
            "data: [DONE]"
        ]
        let count = parseSSEStream(lines: lines).filter { if case .content = $0 { return true }; return false }.count
        XCTAssertEqual(count, 1)
    }

    func testSSE_ThreeParallelToolCalls() {
        let lines = [
            makeToolCallDelta(index: 0, id: "call_1_0", name: "readFile", arguments: ""),
            makeToolCallDelta(index: 1, id: "call_1_1", name: "shell", arguments: ""),
            makeToolCallDelta(index: 2, id: "call_1_2", name: "searchFiles", arguments: ""),
            makeToolCallDelta(index: 0, arguments: "{\"path\": \"main.py\"}"),
            makeToolCallDelta(index: 1, arguments: "{\"command\": \"git log -3\"}"),
            makeToolCallDelta(index: 2, arguments: "{\"pattern\": \"TODO\", \"path\": \".\"}"),
            makeFinishDelta(reason: "tool_calls"),
            "data: [DONE]"
        ]
        let calls = parseSSEStream(lines: lines).compactMap { e -> [TestToolCall]? in
            if case .toolCalls(let c) = e { return c }; return nil
        }.first ?? []
        XCTAssertEqual(calls.count, 3)
        XCTAssertEqual(Set(calls.map(\.name)), ["readFile", "shell", "searchFiles"])
    }

    func testSSE_NumericArgValueStringified() {
        let lines = [
            makeToolCallDelta(index: 0, id: "call_1_0", name: "readFile", arguments: "{\"path\": \"file.txt\", \"startLine\": 10, \"endLine\": 20}"),
            makeFinishDelta(reason: "tool_calls"),
            "data: [DONE]"
        ]
        let calls = parseSSEStream(lines: lines).compactMap { e -> [TestToolCall]? in
            if case .toolCalls(let c) = e { return c }; return nil
        }.first ?? []
        XCTAssertEqual(calls[0].arguments["startLine"], "10")
        XCTAssertEqual(calls[0].arguments["endLine"], "20")
    }

    func testSSE_MultipleServerIdFormats() {
        for serverId in ["call_1712345678_0", "call_1712345678_1", "chatcmpl-abc123-tc0"] {
            let lines = [
                makeToolCallDelta(index: 0, id: serverId, name: "shell", arguments: "{\"command\": \"ls\"}"),
                makeFinishDelta(reason: "tool_calls"),
                "data: [DONE]"
            ]
            let calls = parseSSEStream(lines: lines).compactMap { e -> [TestToolCall]? in
                if case .toolCalls(let c) = e { return c }; return nil
            }.first ?? []
            XCTAssertEqual(calls.first?.id, serverId)
        }
    }
}

// =============================================================================
// MARK: - Tests: History building edge cases
// =============================================================================

final class HistoryEdgeCaseTests: XCTestCase {

    func testHistory_LongToolResultTruncatedTo2000() {
        var msg = TestChatMessage(role: .system, content: "")
        msg.toolCallId = "call_0"
        msg.content = String(repeating: "x", count: 5000)
        let history = buildAgentHistory(messages: [msg])
        XCTAssertEqual((history[0]["content"] as! String).count, 2000)
    }

    func testHistory_LongAssistantContentTruncatedTo500() {
        let msg = TestChatMessage(role: .assistant, content: String(repeating: "y", count: 1000))
        let history = buildAgentHistory(messages: [msg])
        let content = history[0]["content"] as! String
        XCTAssertTrue(content.count <= 503)
        XCTAssertTrue(content.hasSuffix("..."))
    }

    func testHistory_AgentSummaryExcluded() {
        var msg = TestChatMessage(role: .assistant, content: "**shell** -> output")
        msg.isAgentSummary = true
        XCTAssertEqual(buildAgentHistory(messages: [msg]).count, 0)
    }

    func testHistory_EmptyAssistantExcluded() {
        XCTAssertEqual(buildAgentHistory(messages: [TestChatMessage(role: .assistant, content: "")]).count, 0)
    }

    func testHistory_PadOnlyAssistantExcluded() {
        XCTAssertEqual(buildAgentHistory(messages: [TestChatMessage(role: .assistant, content: "<pad><pad>")]).count, 0)
    }

    func testHistory_PadStrippedFromContent() {
        let msg = TestChatMessage(role: .assistant, content: "Hello<pad> world<pad>")
        XCTAssertEqual(buildAgentHistory(messages: [msg])[0]["content"] as? String, "Hello world")
    }

    func testHistory_SystemExcluded() {
        XCTAssertEqual(buildAgentHistory(messages: [TestChatMessage(role: .system, content: "System prompt")]).count, 0)
    }

    func testHistory_EmptyToolResultIncluded() {
        var msg = TestChatMessage(role: .system, content: "")
        msg.toolCallId = "call_0"
        let history = buildAgentHistory(messages: [msg])
        XCTAssertEqual(history.count, 1)
        XCTAssertEqual(history[0]["role"] as? String, "tool")
    }

    func testHistory_TruncationAt28PlusPinnedFirst() {
        var messages: [TestChatMessage] = []
        for i in 0..<40 {
            messages.append(TestChatMessage(role: .user, content: "Message \(i)"))
        }
        let history = buildAgentHistory(messages: messages)
        // 28 from window + 1 pinned first user message = 29
        XCTAssertEqual(history.count, 29)
        // First entry is the pinned original user message
        XCTAssertEqual(history[0]["content"] as? String, "Message 0")
        // Second entry is the start of the suffix(28) window
        XCTAssertEqual(history[1]["content"] as? String, "Message 12")
    }

    func testHistory_ToolCallIdMatchesAcrossMessages() {
        var sim = AgentLoopSimulator(sessionMessages: [
            TestChatMessage(role: .user, content: "Do things")
        ])
        sim.simulateToolCallResponse(
            modelContent: "Running two tools.",
            toolCalls: [
                TestToolCall(id: "call_X_0", name: "shell", arguments: ["command": "ls"]),
                TestToolCall(id: "call_X_1", name: "readFile", arguments: ["path": "f.txt"])
            ],
            toolResults: [
                (id: "call_X_0", name: "shell", output: "file.txt"),
                (id: "call_X_1", name: "readFile", output: "contents")
            ]
        )

        let history = buildAgentHistory(messages: sim.sessionMessages)
        let violations = validateConversationStructure(history)
        XCTAssertTrue(violations.isEmpty, "ID mismatch:\n" + violations.joined(separator: "\n"))

        let tcs = (history.first { ($0["role"] as? String) == "assistant" }!["tool_calls"] as! [[String: Any]])
        let assistantIds = Set(tcs.map { $0["id"] as! String })
        let toolIds = Set(history.filter { ($0["role"] as? String) == "tool" }.map { $0["tool_call_id"] as! String })
        XCTAssertEqual(assistantIds, toolIds)
    }

    func testHistory_TruncationDoesNotSplitToolPairs() {
        var messages: [TestChatMessage] = []
        for i in 0..<14 {
            messages.append(TestChatMessage(role: .user, content: "Q\(i)"))
            messages.append(TestChatMessage(role: .assistant, content: "A\(i)"))
        }

        messages.append(TestChatMessage(role: .user, content: "Use tool"))
        var assistantMsg = TestChatMessage(role: .assistant, content: "Sure")
        assistantMsg.toolCalls = [SerializedToolCall(id: "call_0", name: "shell", arguments: "{\"command\":\"ls\"}")]
        messages.append(assistantMsg)

        var toolMsg = TestChatMessage(role: .system, content: "output")
        toolMsg.toolCallId = "call_0"
        messages.append(toolMsg)

        messages.append(TestChatMessage(role: .user, content: "What happened?"))

        let history = buildAgentHistory(messages: messages)
        let violations = validateConversationStructure(history)
        let orphans = violations.filter { $0.contains("WITHOUT tool_calls") || $0.contains("no preceding assistant") }
        XCTAssertTrue(orphans.isEmpty,
            "Truncation split tool pairs:\n" + orphans.joined(separator: "\n"))
    }

    func testHistory_ProgressiveTruncation() {
        // 4 tool results: older ones should get 500 chars, last 2 get 2000
        let longContent = String(repeating: "x", count: 3000)
        var messages: [TestChatMessage] = [
            TestChatMessage(role: .user, content: "Do things")
        ]
        for i in 0..<4 {
            var toolMsg = TestChatMessage(role: .system, content: longContent)
            toolMsg.toolCallId = "call_\(i)"
            messages.append(toolMsg)
        }
        let history = buildAgentHistory(messages: messages)
        let toolEntries = history.filter { ($0["role"] as? String) == "tool" }
        XCTAssertEqual(toolEntries.count, 4)
        // First 2 tool results (older): truncated to 500
        XCTAssertEqual((toolEntries[0]["content"] as! String).count, 500)
        XCTAssertEqual((toolEntries[1]["content"] as! String).count, 500)
        // Last 2 tool results (recent): truncated to 2000
        XCTAssertEqual((toolEntries[2]["content"] as! String).count, 2000)
        XCTAssertEqual((toolEntries[3]["content"] as! String).count, 2000)
    }

    func testHistory_FirstUserMessagePinned() {
        // Create enough messages that the first user message falls outside the window
        var messages: [TestChatMessage] = [
            TestChatMessage(role: .user, content: "Original task: build a website")
        ]
        for i in 0..<30 {
            messages.append(TestChatMessage(role: .assistant, content: "Step \(i)"))
            messages.append(TestChatMessage(role: .user, content: "Continue \(i)"))
        }
        let history = buildAgentHistory(messages: messages)
        // First entry should be the pinned original user message
        XCTAssertEqual(history[0]["content"] as? String, "Original task: build a website")
    }

    func testHistory_FirstUserNotDuplicatedWhenInWindow() {
        // Small conversation — first user message is already in the window
        let messages: [TestChatMessage] = [
            TestChatMessage(role: .user, content: "Hello"),
            TestChatMessage(role: .assistant, content: "Hi there"),
        ]
        let history = buildAgentHistory(messages: messages)
        XCTAssertEqual(history.count, 2)
        // Only appears once
        let userEntries = history.filter { ($0["role"] as? String) == "user" }
        XCTAssertEqual(userEntries.count, 1)
    }
}

// =============================================================================
// MARK: - Tests: Codable round-trip
// =============================================================================

final class SerializationTests: XCTestCase {

    func testSerializedToolCall_Codable() {
        let tc = SerializedToolCall(id: "call_123", name: "shell", arguments: "{\"command\": \"ls -la\"}")
        let decoded = try! JSONDecoder().decode(SerializedToolCall.self, from: JSONEncoder().encode(tc))
        XCTAssertEqual(decoded, tc)
    }

    func testChatMessage_CodableWithToolCalls() {
        var msg = TestChatMessage(role: .assistant, content: "Using tools.")
        msg.toolCalls = [
            SerializedToolCall(id: "call_1", name: "shell", arguments: "{\"command\":\"ls\"}"),
            SerializedToolCall(id: "call_2", name: "readFile", arguments: "{\"path\":\"f.txt\"}")
        ]
        let decoded = try! JSONDecoder().decode(TestChatMessage.self, from: JSONEncoder().encode(msg))
        XCTAssertEqual(decoded.toolCalls?.count, 2)
        XCTAssertEqual(decoded.toolCalls?[0].id, "call_1")
        XCTAssertEqual(decoded.toolCalls?[1].name, "readFile")
    }

    func testChatMessage_CodableWithoutToolCalls() {
        let msg = TestChatMessage(role: .user, content: "Hello")
        let decoded = try! JSONDecoder().decode(TestChatMessage.self, from: JSONEncoder().encode(msg))
        XCTAssertNil(decoded.toolCalls)
    }

    func testChatMessage_BackwardsCompatible() {
        let oldJson = "{\"role\":\"assistant\",\"content\":\"Hello\",\"isStreaming\":false,\"isAgentSummary\":false}"
        let decoded = try! JSONDecoder().decode(TestChatMessage.self, from: oldJson.data(using: .utf8)!)
        XCTAssertEqual(decoded.content, "Hello")
        XCTAssertNil(decoded.toolCalls, "Missing field must decode as nil for backwards compat")
    }

    func testSerializedToolCall_ArgumentsPreservedExactly() {
        let args = "{\"path\":\"/tmp/config.json\",\"content\":\"{\\\"key\\\": \\\"value\\\"}\"}"
        let tc = SerializedToolCall(id: "call_1", name: "writeFile", arguments: args)
        let decoded = try! JSONDecoder().decode(SerializedToolCall.self, from: JSONEncoder().encode(tc))
        XCTAssertEqual(decoded.arguments, args)
    }
}

// =============================================================================
// MARK: - Tests: Pad detection
// =============================================================================

final class PadDetectionTests: XCTestCase {

    func testPadDetectionLogic() {
        for c in ["<pad><pad><pad>", "<pad>", "  <pad>  ", "", "   ", "\n\n<pad>\n"] {
            let cleaned = c.replacingOccurrences(of: "<pad>", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
            XCTAssertTrue(cleaned.isEmpty, "'\(c)' should be detected as empty")
        }
        for c in ["Hello", "<pad>Content<pad>", "Result."] {
            let cleaned = c.replacingOccurrences(of: "<pad>", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
            XCTAssertFalse(cleaned.isEmpty, "'\(c)' should NOT be empty")
        }
    }
}

// =============================================================================
// MARK: - Tests: Role handling
// =============================================================================

final class RoleHandlingTests: XCTestCase {

    func testToolResultsStoredAsSystemRole() {
        var msg = TestChatMessage(role: .system, content: "hello world")
        msg.toolCallId = "call_0"
        let history = buildAgentHistory(messages: [msg])
        XCTAssertEqual(history[0]["role"] as? String, "tool")
    }

    func testSystemWithoutToolCallIdFiltered() {
        XCTAssertEqual(buildAgentHistory(messages: [TestChatMessage(role: .system, content: "Sys")]).count, 0)
    }

    func testToolResponseIncludesToolCallId() {
        var msg = TestChatMessage(role: .system, content: "result")
        msg.toolCallId = "call_ABC_0"
        XCTAssertEqual(buildAgentHistory(messages: [msg])[0]["tool_call_id"] as? String, "call_ABC_0")
    }

    func testUserMessagesPassThrough() {
        let history = buildAgentHistory(messages: [TestChatMessage(role: .user, content: "Hello")])
        XCTAssertEqual(history[0]["role"] as? String, "user")
        XCTAssertEqual(history[0]["content"] as? String, "Hello")
    }

    func testWhitespaceOnlyAssistantExcluded() {
        XCTAssertEqual(buildAgentHistory(messages: [TestChatMessage(role: .assistant, content: "   \n\n   ")]).count, 0)
    }
}
