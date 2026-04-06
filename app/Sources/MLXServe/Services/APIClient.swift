import Foundation

struct TokenUsage {
    let promptTokens: Int
    let completionTokens: Int
    let totalTokens: Int
    let tokensPerSecond: Double
}

enum SSEEvent {
    case content(String)
    case reasoning(String)
    case usage(TokenUsage)
    case toolCalls([APIClient.ToolCall])
    case done
}

class APIClient {
    private let session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 600
        config.timeoutIntervalForResource = 3600
        config.waitsForConnectivity = true
        return URLSession(configuration: config)
    }()
    private let decoder = JSONDecoder()

    func checkHealth(port: UInt16) async throws -> Bool {
        let url = URL(string: "http://127.0.0.1:\(port)/health")!
        let (data, response) = try await session.data(from: url)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            return false
        }
        if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let status = json["status"] as? String {
            return status == "ok"
        }
        return false
    }

    func fetchModels(port: UInt16) async throws -> ModelInfo? {
        let url = URL(string: "http://127.0.0.1:\(port)/v1/models")!
        let (data, _) = try await session.data(from: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let dataArr = json["data"] as? [[String: Any]],
              let first = dataArr.first else { return nil }

        let name = first["id"] as? String ?? "unknown"
        let meta = first["meta"] as? [String: Any] ?? [:]
        return ModelInfo(
            name: name,
            quantBits: meta["quantization_bits"] as? Int ?? 0,
            layers: meta["num_layers"] as? Int ?? 0,
            hiddenSize: meta["hidden_size"] as? Int ?? 0,
            vocabSize: meta["vocab_size"] as? Int ?? 0,
            contextLength: meta["context_length"] as? Int ?? 0
        )
    }

    func fetchProps(port: UInt16) async throws -> MemoryInfo? {
        let url = URL(string: "http://127.0.0.1:\(port)/props")!
        let (data, _) = try await session.data(from: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let mem = json["memory"] as? [String: Any] else { return nil }
        return MemoryInfo(
            activeBytes: mem["active_bytes"] as? Int64 ?? 0,
            peakBytes: mem["peak_bytes"] as? Int64 ?? 0
        )
    }

    // MARK: - Agent Tool Calling

    struct ToolCallResult {
        let content: String?
        let toolCalls: [ToolCall]?
        let usage: TokenUsage?
    }

    struct ToolCall {
        let id: String
        let name: String
        let arguments: [String: String]
    }

    /// Non-streaming chat completion with tool definitions.
    /// Returns either content (no tools needed) or tool_calls (model wants to use tools).
    func chatWithTools(
        port: UInt16,
        messages: [[String: Any]],
        tools: [[String: Any]],
        maxTokens: Int = 8192,
        temperature: Double = 0.7
    ) async throws -> ToolCallResult {
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("close", forHTTPHeaderField: "Connection")
        request.timeoutInterval = 300

        let body: [String: Any] = [
            "model": "mlx-serve",
            "messages": messages,
            "tools": tools,
            "max_tokens": maxTokens,
            "temperature": temperature,
            "stream": false,
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = json["choices"] as? [[String: Any]],
              let choice = choices.first,
              let message = choice["message"] as? [String: Any] else {
            throw URLError(.cannotParseResponse)
        }

        // Parse usage
        var usage: TokenUsage?
        if let u = json["usage"] as? [String: Any] {
            usage = TokenUsage(
                promptTokens: u["prompt_tokens"] as? Int ?? 0,
                completionTokens: u["completion_tokens"] as? Int ?? 0,
                totalTokens: u["total_tokens"] as? Int ?? 0,
                tokensPerSecond: 0
            )
        }

        // Check for tool_calls
        if let tcArray = message["tool_calls"] as? [[String: Any]] {
            var calls: [ToolCall] = []
            for tc in tcArray {
                let id = tc["id"] as? String ?? UUID().uuidString
                guard let function = tc["function"] as? [String: Any],
                      let name = function["name"] as? String else { continue }
                let argsStr = function["arguments"] as? String ?? "{}"
                var args: [String: String] = [:]
                if let argsData = argsStr.data(using: .utf8),
                   let argsDict = try? JSONSerialization.jsonObject(with: argsData) as? [String: Any] {
                    for (k, v) in argsDict {
                        args[k] = "\(v)"
                    }
                }
                calls.append(ToolCall(id: id, name: name, arguments: args))
            }
            let content = message["content"] as? String
            return ToolCallResult(content: content, toolCalls: calls.isEmpty ? nil : calls, usage: usage)
        }

        let content = message["content"] as? String
        return ToolCallResult(content: content, toolCalls: nil, usage: usage)
    }

    func streamChat(
        port: UInt16,
        messages: [[String: Any]],
        maxTokens: Int = 2048,
        temperature: Double = 0.8,
        enableThinking: Bool = false,
        tools: [[String: Any]]? = nil
    ) -> AsyncThrowingStream<SSEEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await self.performStream(
                        port: port, messages: messages,
                        maxTokens: maxTokens, temperature: temperature,
                        enableThinking: enableThinking, tools: tools,
                        continuation: continuation
                    )
                } catch let error as URLError where error.code == .networkConnectionLost {
                    do {
                        try await self.performStream(
                            port: port, messages: messages,
                            maxTokens: maxTokens, temperature: temperature,
                            enableThinking: enableThinking, tools: tools,
                            continuation: continuation
                        )
                    } catch {
                        continuation.finish(throwing: error)
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func performStream(
        port: UInt16,
        messages: [[String: Any]],
        maxTokens: Int,
        temperature: Double,
        enableThinking: Bool,
        tools: [[String: Any]]? = nil,
        continuation: AsyncThrowingStream<SSEEvent, Error>.Continuation
    ) async throws {
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("close", forHTTPHeaderField: "Connection")
        request.timeoutInterval = 300

        var body: [String: Any] = [
            "model": "mlx-serve",
            "messages": messages,
            "max_tokens": maxTokens,
            "temperature": temperature,
            "top_p": 0.95,
            "stream": true,
            "stream_options": ["include_usage": true],
        ]
        if enableThinking {
            body["enable_thinking"] = true
        }
        if let tools {
            body["tools"] = tools
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let streamStart = Date()
        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            print("[APIClient] Bad response: HTTP \(statusCode)")
            try? "Bad response: HTTP \(statusCode)\n".write(toFile: NSString(string: "~/.mlx-serve/debug.log").expandingTildeInPath, atomically: true, encoding: .utf8)
            if let http = response as? HTTPURLResponse, let data = try? await URLSession.shared.data(for: request).0 {
                print("[APIClient] Body: \(String(data: data, encoding: .utf8) ?? "n/a")")
            }
            continuation.finish(throwing: URLError(.badServerResponse))
            return
        }

        var firstTokenTime: Date?
        // Accumulate tool call deltas across chunks, keyed by index
        var pendingToolCalls: [String: (id: String, name: String, args: String)] = [:]
        var hasToolCalls = false
        var emittedToolCalls = false

        for try await line in bytes.lines {
            guard line.hasPrefix("data: ") else { continue }
            let payload = String(line.dropFirst(6))
            if payload == "[DONE]" {
                continuation.yield(.done)
                break
            }

            guard let jsonData = payload.data(using: .utf8),
                  let chunk = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
                continue
            }

            // Check for server-side error events
            if let error = chunk["error"] as? [String: Any],
               let message = error["message"] as? String {
                continuation.yield(.content("\n\n[Server Error: \(message)]"))
                continuation.yield(.done)
                continuation.finish()
                return
            }

            // Parse usage from final chunk
            if let usage = chunk["usage"] as? [String: Any], usage["prompt_tokens"] != nil {
                let prompt = usage["prompt_tokens"] as? Int ?? 0
                let completion = usage["completion_tokens"] as? Int ?? 0
                let total = usage["total_tokens"] as? Int ?? 0
                let elapsed = Date().timeIntervalSince(firstTokenTime ?? streamStart)
                let tps = elapsed > 0 ? Double(completion) / elapsed : 0
                continuation.yield(.usage(TokenUsage(
                    promptTokens: prompt, completionTokens: completion,
                    totalTokens: total, tokensPerSecond: tps
                )))
            }

            guard let choices = chunk["choices"] as? [[String: Any]],
                  let delta = choices.first?["delta"] as? [String: Any] else {
                continue
            }
            if firstTokenTime == nil {
                firstTokenTime = Date()
            }

            if let content = delta["content"] as? String, !content.isEmpty {
                continuation.yield(.content(content))
            }
            if let reasoning = delta["reasoning_content"] as? String, !reasoning.isEmpty {
                continuation.yield(.reasoning(reasoning))
            }

            // Accumulate tool call deltas
            if let tcArray = delta["tool_calls"] as? [[String: Any]] {
                hasToolCalls = true
                for tc in tcArray {
                    let idx = "\(tc["index"] as? Int ?? 0)"
                    var existing = pendingToolCalls[idx] ?? (id: "", name: "", args: "")
                    if let id = tc["id"] as? String {
                        existing.id = id
                    }
                    if let function = tc["function"] as? [String: Any] {
                        if let name = function["name"] as? String {
                            existing.name = name
                        }
                        if let args = function["arguments"] as? String {
                            existing.args += args
                        }
                    }
                    pendingToolCalls[idx] = existing
                }
            }

            // Check finish_reason for tool_calls
            if let fr = choices.first?["finish_reason"] as? String, fr == "tool_calls" {
                var calls: [ToolCall] = []
                for (_, tc) in pendingToolCalls.sorted(by: { $0.key < $1.key }) {
                    var args: [String: String] = [:]
                    if let data = tc.args.data(using: .utf8),
                       let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        for (k, v) in dict { args[k] = "\(v)" }
                    }
                    calls.append(ToolCall(id: tc.id, name: tc.name, arguments: args))
                }
                if !calls.isEmpty {
                    continuation.yield(.toolCalls(calls))
                    emittedToolCalls = true
                }
            }
        }

        // Fallback: emit tool calls if stream ended without finish_reason
        if hasToolCalls && !pendingToolCalls.isEmpty && !emittedToolCalls {
            var calls: [ToolCall] = []
            for (_, tc) in pendingToolCalls.sorted(by: { $0.key < $1.key }) {
                if tc.name.isEmpty { continue }
                var args: [String: String] = [:]
                if let data = tc.args.data(using: .utf8),
                   let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    for (k, v) in dict { args[k] = "\(v)" }
                }
                calls.append(ToolCall(id: tc.id, name: tc.name, arguments: args))
            }
            if !calls.isEmpty {
                continuation.yield(.toolCalls(calls))
            }
        }

        continuation.finish()
    }
}
