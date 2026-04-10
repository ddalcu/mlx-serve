import XCTest
import Foundation

/// Integration tests that connect to a REAL running mlx-serve instance.
/// Tests the exact Swift JSONSerialization → HTTP → Server pipeline.
/// Run with: swift test --filter AgentIntegrationTests
final class AgentIntegrationTests: XCTestCase {

    private let session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 120
        return URLSession(configuration: config)
    }()

    /// The EXACT tool definitions from AgentPrompt.toolDefinitions
    private let allTools: [[String: Any]] = [
        ["type": "function", "function": ["name": "shell", "description": "Run a command", "parameters": ["type": "object", "properties": ["command": ["type": "string"]], "required": ["command"]]] as [String: Any]],
        ["type": "function", "function": ["name": "writeFile", "description": "Write a file", "parameters": ["type": "object", "properties": ["path": ["type": "string"], "content": ["type": "string"]], "required": ["path", "content"]]] as [String: Any]],
        ["type": "function", "function": ["name": "readFile", "description": "Read a file", "parameters": ["type": "object", "properties": ["path": ["type": "string"]], "required": ["path"]]] as [String: Any]],
        ["type": "function", "function": ["name": "editFile", "description": "Find and replace in file", "parameters": ["type": "object", "properties": ["path": ["type": "string"], "find": ["type": "string"], "replace": ["type": "string"]], "required": ["path", "find", "replace"]]] as [String: Any]],
        ["type": "function", "function": ["name": "searchFiles", "description": "Grep for pattern", "parameters": ["type": "object", "properties": ["pattern": ["type": "string"]], "required": ["pattern"]]] as [String: Any]],
        ["type": "function", "function": ["name": "browse", "description": "Browse a URL", "parameters": ["type": "object", "properties": ["action": ["type": "string"], "url": ["type": "string"]], "required": ["action"]]] as [String: Any]],
        ["type": "function", "function": ["name": "webSearch", "description": "Search the web", "parameters": ["type": "object", "properties": ["query": ["type": "string"]], "required": ["query"]]] as [String: Any]],
    ]

    private let systemPrompt = "You are a helpful macOS assistant. Use tools for tasks. Answer directly when no tools needed. For web search use webSearch tool."

    // MARK: - Server discovery

    private func findServerPort() async -> UInt16? {
        for port: UInt16 in [8080, 5000, 8081] {
            let url = URL(string: "http://127.0.0.1:\(port)/health")!
            if let (data, response) = try? await session.data(from: url),
               let http = response as? HTTPURLResponse, http.statusCode == 200,
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               (json["status"] as? String) == "ok" {
                return port
            }
        }
        return nil
    }

    // MARK: - Send request matching APIClient.performStream exactly

    /// Send a non-streaming request using the EXACT JSON body format the Swift app produces.
    private func sendRequest(port: UInt16, messages: [[String: Any]], tools: [[String: Any]]? = nil, maxTokens: Int = 256) async throws -> (content: String, completionTokens: Int, finishReason: String, promptTokens: Int, toolCalls: [[String: Any]]) {
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("close", forHTTPHeaderField: "Connection")
        request.timeoutInterval = 60

        var body: [String: Any] = [
            "model": "mlx-serve",
            "messages": messages,
            "max_tokens": maxTokens,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": false,
        ]
        if let tools { body["tools"] = tools }

        let jsonData = try JSONSerialization.data(withJSONObject: body)

        // Print the EXACT JSON for debugging
        if let pretty = try? JSONSerialization.jsonObject(with: jsonData),
           let prettyData = try? JSONSerialization.data(withJSONObject: pretty, options: .prettyPrinted),
           let prettyStr = String(data: prettyData, encoding: .utf8) {
            print("\n=== REQUEST JSON (\(jsonData.count) bytes) ===")
            // Print just the messages, not the full tools
            if let dict = pretty as? [String: Any], let msgs = dict["messages"] as? [[String: Any]] {
                for (i, msg) in msgs.enumerated() {
                    let role = msg["role"] as? String ?? "?"
                    let content = msg["content"]
                    let hasToolCalls = msg["tool_calls"] != nil
                    let hasToolCallId = msg["tool_call_id"] != nil
                    let contentDesc: String
                    if content is NSNull || content == nil {
                        contentDesc = "null"
                    } else if let s = content as? String {
                        contentDesc = "'\(s.prefix(80))'"
                    } else {
                        contentDesc = "\(type(of: content))"
                    }
                    print("  [\(i)] role=\(role) content=\(contentDesc) tool_calls=\(hasToolCalls) tool_call_id=\(hasToolCallId)")
                    if hasToolCalls, let tcs = msg["tool_calls"] as? [[String: Any]] {
                        for tc in tcs {
                            let fn = tc["function"] as? [String: Any]
                            let name = fn?["name"] as? String ?? "?"
                            let args = fn?["arguments"]
                            let argsType = args.map { "\(type(of: $0))" } ?? "nil"
                            print("       -> \(name) args_type=\(argsType) args=\(String(describing: args).prefix(100))")
                        }
                    }
                }
            }
            print("=== END REQUEST ===\n")
        }

        request.httpBody = jsonData
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let errBody = String(data: data, encoding: .utf8) ?? ""
            throw NSError(domain: "", code: (response as? HTTPURLResponse)?.statusCode ?? 0,
                         userInfo: [NSLocalizedDescriptionKey: "HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0): \(errBody.prefix(500))"])
        }

        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let usage = json["usage"] as? [String: Any] ?? [:]
        let choices = json["choices"] as? [[String: Any]] ?? []
        let choice = choices.first ?? [:]
        let message = choice["message"] as? [String: Any] ?? [:]

        let result = (
            content: message["content"] as? String ?? "",
            completionTokens: usage["completion_tokens"] as? Int ?? 0,
            finishReason: choice["finish_reason"] as? String ?? "",
            promptTokens: usage["prompt_tokens"] as? Int ?? 0,
            toolCalls: message["tool_calls"] as? [[String: Any]] ?? []
        )

        print("=== RESPONSE: prompt=\(result.promptTokens) completion=\(result.completionTokens) finish=\(result.finishReason) ===")
        print("  content: '\(result.content.prefix(300))'")
        print("  tool_calls: \(result.toolCalls.count)")
        return result
    }

    // MARK: - Build messages using the EXACT same logic as buildAgentHistory()

    struct SimToolCall {
        let id: String
        let name: String
        let arguments: [String: String]
    }

    /// Build messages exactly as buildAgentHistory() does in ChatView.swift
    private func buildMessages(
        userMessage: String,
        assistantContent: String,
        toolCalls: [SimToolCall],
        toolResult: String
    ) -> [[String: Any]] {
        var messages: [[String: Any]] = [
            ["role": "system", "content": systemPrompt],
            ["role": "user", "content": userMessage],
        ]

        // Build assistant message with tool_calls — EXACT same code as buildAgentHistory()
        var assistantDict: [String: Any] = ["role": "assistant"]
        let content = assistantContent
            .replacingOccurrences(of: "<pad>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        assistantDict["content"] = content.isEmpty ? "" : content

        assistantDict["tool_calls"] = toolCalls.map { tc -> [String: Any] in
            // This is how SerializedToolCall.arguments is created
            let argsJson = (try? JSONSerialization.data(withJSONObject: tc.arguments))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
            return [
                "id": tc.id,
                "type": "function",
                "function": [
                    "name": tc.name,
                    "arguments": argsJson  // This is a String
                ] as [String: Any]
            ] as [String: Any]
        }
        messages.append(assistantDict)

        // Build tool response — EXACT same code as buildAgentHistory()
        messages.append([
            "role": "tool",
            "tool_call_id": toolCalls[0].id,
            "content": String(toolResult.prefix(2000))
        ])

        return messages
    }

    // MARK: - Tests

    /// THE BUG REPRODUCER: build messages exactly like buildAgentHistory(),
    /// send through JSONSerialization exactly like APIClient, check response.
    func testExactAgentLoopFormat_ShortToolResult() async throws {
        guard let port = await findServerPort() else {
            print("SKIP: No server running")
            return
        }

        let messages = buildMessages(
            userMessage: "What day is today?",
            assistantContent: "",
            toolCalls: [SimToolCall(id: "call_99_0", name: "shell", arguments: ["command": "date"])],
            toolResult: "Fri Apr 4 16:30:00 PDT 2026"
        )

        let result = try await sendRequest(port: port, messages: messages, tools: allTools)

        let cleaned = result.content.replacingOccurrences(of: "<pad>", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertTrue(result.completionTokens > 2 || !cleaned.isEmpty,
            "Model generated \(result.completionTokens) tokens, content='\(cleaned)'. " +
            "If <=2 tokens, the Jinja template rendered an invalid prompt.")
    }

    /// THE BUG REPRODUCER with LONG tool result (DuckDuckGo page)
    func testExactAgentLoopFormat_LongWebSearchResult() async throws {
        guard let port = await findServerPort() else {
            print("SKIP: No server running")
            return
        }

        let longResult = """
        Navigated to https://html.duckduckgo.com/html/?q=finance%20news%20headlines
        Title: finance news headlines at DuckDuckGo

        Page content:
        Stock Market News - Stock Forecast

         Ad

        Viewing ads is privacy protected by DuckDuckGo. Ad clicks are managed by Microsoft's ad network (more info).

         edwardjones.com

        Focus On What Matters Most. Let's Build A Strategy To Help Secure Your Financial Future! We're Here Through Ups And Downs To Help You Live A Life You Love.

        World News: Top & Breaking World News Today | AP News
         apnews.com/world-news

        Stay informed with top world news today. The Associated Press aims to keep you up-to-date with breaking world news stories around the globe.

         World News | Latest Top Stories | Reuters
         www.reuters.com/world
        """

        let messages = buildMessages(
            userMessage: "search the web for finance news and give me the headlines",
            assistantContent: "",
            toolCalls: [SimToolCall(id: "call_99_0", name: "webSearch", arguments: ["query": "finance news headlines"])],
            toolResult: longResult
        )

        let result = try await sendRequest(port: port, messages: messages, tools: allTools)

        let cleaned = result.content.replacingOccurrences(of: "<pad>", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertTrue(result.completionTokens > 2 || !cleaned.isEmpty || !result.toolCalls.isEmpty,
            "REPRODUCES BUG: Model generated \(result.completionTokens) tokens after webSearch result. " +
            "content='\(cleaned.prefix(100))'. This is the 'model couldn't generate' error.")
    }

    /// Verify the JSON serialization produces the right types for arguments
    func testArgumentsSerializedAsString() throws {
        let tc = SimToolCall(id: "call_1", name: "shell", arguments: ["command": "echo hello"])
        let argsJson = (try? JSONSerialization.data(withJSONObject: tc.arguments))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"

        // Build the message dict the same way buildAgentHistory does
        let msg: [String: Any] = [
            "role": "assistant",
            "content": "",
            "tool_calls": [[
                "id": tc.id,
                "type": "function",
                "function": [
                    "name": tc.name,
                    "arguments": argsJson
                ] as [String: Any]
            ] as [String: Any]]
        ]

        // Serialize to JSON
        let body: [String: Any] = ["messages": [msg]]
        let data = try JSONSerialization.data(withJSONObject: body)
        let jsonStr = String(data: data, encoding: .utf8)!

        print("Serialized JSON: \(jsonStr)")

        // Parse back and check arguments type
        let parsed = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let msgs = parsed["messages"] as! [[String: Any]]
        let tcs = msgs[0]["tool_calls"] as! [[String: Any]]
        let fn = tcs[0]["function"] as! [String: Any]
        let args = fn["arguments"]!

        print("arguments type: \(type(of: args))")
        print("arguments value: \(args)")

        XCTAssertTrue(args is String,
            "arguments must be String, got \(type(of: args)). " +
            "The server expects 'arguments' as a JSON string, not a nested dict.")
    }

    /// Compare: what does curl-style JSON look like vs Swift JSONSerialization?
    func testCompareSwiftVsCurlFormat() throws {
        // Build the same message two ways

        // Way 1: Swift (buildAgentHistory style)
        let swiftArgs: [String: String] = ["query": "finance news headlines"]
        let swiftArgsJson = String(data: try JSONSerialization.data(withJSONObject: swiftArgs), encoding: .utf8)!

        let swiftMsg: [String: Any] = [
            "role": "assistant",
            "content": "",
            "tool_calls": [[
                "id": "call_99_0",
                "type": "function",
                "function": [
                    "name": "webSearch",
                    "arguments": swiftArgsJson
                ] as [String: Any]
            ] as [String: Any]]
        ]

        let swiftBody: [String: Any] = ["messages": [
            ["role": "system", "content": "test"] as [String: Any],
            swiftMsg,
            ["role": "tool", "tool_call_id": "call_99_0", "content": "result"] as [String: Any]
        ]]

        let swiftData = try JSONSerialization.data(withJSONObject: swiftBody, options: .sortedKeys)
        let swiftJson = String(data: swiftData, encoding: .utf8)!

        print("\n=== SWIFT JSON ===")
        print(swiftJson)

        // Check critical details
        // 1. content for assistant: should be "" (empty string) or null
        XCTAssertTrue(swiftJson.contains("\"content\":\"\"") || swiftJson.contains("\"content\":null"),
            "Assistant content should be empty string or null")

        // 2. arguments should be a STRING (escaped JSON), not a nested object
        // Good: "arguments":"{\"query\":\"finance news headlines\"}"
        // Bad:  "arguments":{"query":"finance news headlines"}
        XCTAssertTrue(swiftJson.contains("\"arguments\":\"{"),
            "arguments should be a JSON string (starts with \"{), not a nested object")

        // 3. tool message should have role:"tool"
        XCTAssertTrue(swiftJson.contains("\"role\":\"tool\""))

        print("\n=== FORMAT CHECKS PASSED ===")
    }
}
