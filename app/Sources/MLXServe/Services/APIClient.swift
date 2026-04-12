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
    case maxTokensReached
    case done
}

struct RetryPolicy {
    let maxRetries: Int
    let baseDelayMs: Int
    let maxDelayMs: Int

    static let `default` = RetryPolicy(maxRetries: 5, baseDelayMs: 500, maxDelayMs: 16_000)
    static let aggressive = RetryPolicy(maxRetries: 3, baseDelayMs: 200, maxDelayMs: 4_000)

    func delay(for attempt: Int) -> UInt64 {
        let base = min(baseDelayMs * Int(pow(2.0, Double(attempt - 1))), maxDelayMs)
        let jitter = Int.random(in: 0..<max(1, base / 4))
        return UInt64(base + jitter) * 1_000_000  // nanoseconds
    }
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
            peakBytes: mem["peak_bytes"] as? Int64 ?? 0,
            maxSafeContext: mem["max_safe_context"] as? Int ?? 0
        )
    }

    // MARK: - Agent Tool Calling

    struct ToolCall {
        let id: String
        let name: String
        let arguments: [String: String]
        let rawArguments: String
    }

    func streamChat(
        port: UInt16,
        messages: [[String: Any]],
        maxTokens: Int = 2048,
        temperature: Double = 0.8,
        enableThinking: Bool = false,
        tools: [[String: Any]]? = nil,
        toolsJSON: String? = nil,
        retryPolicy: RetryPolicy = .default
    ) -> AsyncThrowingStream<SSEEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                var lastError: Error?
                for attempt in 0...retryPolicy.maxRetries {
                    do {
                        try await self.performStream(
                            port: port, messages: messages,
                            maxTokens: maxTokens, temperature: temperature,
                            enableThinking: enableThinking, tools: tools,
                            toolsJSON: toolsJSON,
                            continuation: continuation
                        )
                        return  // success
                    } catch let error as URLError where Self.isRetryable(error) {
                        lastError = error
                        if attempt < retryPolicy.maxRetries {
                            let delay = retryPolicy.delay(for: attempt + 1)
                            print("[APIClient] Retry \(attempt + 1)/\(retryPolicy.maxRetries) after \(delay / 1_000_000)ms: \(error.code.rawValue)")
                            try? await Task.sleep(nanoseconds: delay)
                        }
                    } catch {
                        continuation.finish(throwing: error)
                        return
                    }
                }
                continuation.finish(throwing: lastError ?? URLError(.unknown))
            }
        }
    }

    private static func isRetryable(_ error: URLError) -> Bool {
        switch error.code {
        case .networkConnectionLost, .notConnectedToInternet,
             .timedOut, .cannotConnectToHost, .cannotFindHost:
            return true
        default:
            return false
        }
    }

    private func performStream(
        port: UInt16,
        messages: [[String: Any]],
        maxTokens: Int,
        temperature: Double,
        enableThinking: Bool,
        tools: [[String: Any]]? = nil,
        toolsJSON: String? = nil,
        continuation: AsyncThrowingStream<SSEEvent, Error>.Continuation
    ) async throws {
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("close", forHTTPHeaderField: "Connection")
        request.timeoutInterval = 300

        if let toolsJSON {
            // Splice pre-serialized tools JSON to preserve property key order
            let messagesData = try JSONSerialization.data(withJSONObject: messages)
            guard let messagesStr = String(data: messagesData, encoding: .utf8) else {
                continuation.finish(throwing: URLError(.cannotParseResponse))
                return
            }
            var parts = [
                "\"model\":\"mlx-serve\"",
                "\"messages\":\(messagesStr)",
                "\"max_tokens\":\(maxTokens)",
                "\"temperature\":\(temperature)",
                "\"top_p\":0.95",
                "\"stream\":true",
                "\"stream_options\":{\"include_usage\":true}",
            ]
            if enableThinking { parts.append("\"enable_thinking\":true") }
            parts.append("\"tools\":\(toolsJSON)")
            request.httpBody = "{\(parts.joined(separator: ","))}".data(using: .utf8)
        } else {
            var body: [String: Any] = [
                "model": "mlx-serve",
                "messages": messages,
                "max_tokens": maxTokens,
                "temperature": temperature,
                "top_p": 0.95,
                "stream": true,
                "stream_options": ["include_usage": true],
            ]
            if enableThinking { body["enable_thinking"] = true }
            if let tools { body["tools"] = tools }
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        }

        let streamStart = Date()
        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            print("[APIClient] Bad response: HTTP \(statusCode)")
            try? "Bad response: HTTP \(statusCode)\n".write(toFile: NSString(string: "~/.mlx-serve/debug.log").expandingTildeInPath, atomically: true, encoding: .utf8)
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
                        } else if let argsDict = function["arguments"] as? [String: Any] {
                            // Server sent arguments as object instead of string — serialize it
                            if let data = try? JSONSerialization.data(withJSONObject: argsDict),
                               let str = String(data: data, encoding: .utf8) {
                                existing.args += str
                            }
                            Self.appendToolLog("ARGS_AS_OBJECT: name=\(existing.name) args=\(argsDict)")
                        } else if function["arguments"] != nil {
                            // Arguments present but not String or Dict — log the actual type
                            Self.appendToolLog("ARGS_UNKNOWN_TYPE: name=\(existing.name) type=\(type(of: function["arguments"]!)) raw=\(function["arguments"]!)")
                        }
                    }
                    pendingToolCalls[idx] = existing
                }
            }

            // Check finish_reason
            if let fr = choices.first?["finish_reason"] as? String, fr == "length" {
                continuation.yield(.maxTokensReached)
            }
            if let fr = choices.first?["finish_reason"] as? String, fr == "tool_calls" {
                var calls: [ToolCall] = []
                for (_, tc) in pendingToolCalls.sorted(by: { $0.key < $1.key }) {
                    Self.appendToolLog("EMIT: name=\(tc.name) rawArgs=\(tc.args.prefix(500))")
                    let args = Self.parseToolCallArgs(tc.args, toolName: tc.name)
                    Self.appendToolLog("PARSED: name=\(tc.name) keys=\(args.keys.sorted().joined(separator: ","))")
                    calls.append(ToolCall(id: tc.id, name: tc.name, arguments: args, rawArguments: tc.args))
                }
                if !calls.isEmpty {
                    continuation.yield(.toolCalls(calls))
                    emittedToolCalls = true
                }
            }
        }

        // Fallback: emit tool calls if stream ended without finish_reason
        if hasToolCalls && !pendingToolCalls.isEmpty && !emittedToolCalls {
            Self.appendToolLog("FALLBACK_EMIT: no finish_reason, pending=\(pendingToolCalls.count)")
            var calls: [ToolCall] = []
            for (_, tc) in pendingToolCalls.sorted(by: { $0.key < $1.key }) {
                if tc.name.isEmpty { continue }
                Self.appendToolLog("FALLBACK: name=\(tc.name) rawArgs=\(tc.args.prefix(500))")
                let args = Self.parseToolCallArgs(tc.args, toolName: tc.name)
                calls.append(ToolCall(id: tc.id, name: tc.name, arguments: args, rawArguments: tc.args))
            }
            if !calls.isEmpty {
                continuation.yield(.toolCalls(calls))
            }
        }

        continuation.finish()
    }

    /// Parse tool call arguments JSON string into a dictionary.
    /// Logs a warning when parsing fails so we can diagnose intermittent issues.
    private static func parseToolCallArgs(_ argsString: String, toolName: String) -> [String: String] {
        guard !argsString.isEmpty else {
            print("[APIClient] WARNING: empty arguments for tool '\(toolName)'")
            return [:]
        }
        // Try parsing as-is first; if it fails, attempt repair of truncated JSON
        var dict: [String: Any]?
        if let data = argsString.data(using: .utf8) {
            dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        }
        if dict == nil {
            // Attempt to repair truncated JSON by closing open strings and braces/brackets
            var repaired = argsString
            // Close unclosed string literal
            let unescapedQuotes = repaired.replacingOccurrences(of: "\\\"", with: "").filter { $0 == "\"" }.count
            if unescapedQuotes % 2 != 0 { repaired += "\"" }
            // Track unmatched braces/brackets outside quoted regions and close them
            var closers: [Character] = []
            var inStr = false
            var escaped = false
            for ch in repaired {
                if escaped { escaped = false; continue }
                if ch == "\\" && inStr { escaped = true; continue }
                if ch == "\"" { inStr.toggle(); continue }
                if inStr { continue }
                switch ch {
                case "{": closers.append("}")
                case "[": closers.append("]")
                case "}": if closers.last == "}" { closers.removeLast() }
                case "]": if closers.last == "]" { closers.removeLast() }
                default: break
                }
            }
            for closer in closers.reversed() {
                repaired.append(closer)
            }
            if let data = repaired.data(using: .utf8) {
                dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                if dict != nil {
                    print("[APIClient] Repaired truncated JSON for tool '\(toolName)'")
                }
            }
        }
        guard let dict else {
            // JSON repair failed — try to salvage path for file tools
            if toolName == "writeFile" || toolName == "editFile" {
                if let path = extractPathFromTruncatedJSON(argsString) {
                    appendToolLog("SALVAGED_PATH: name=\(toolName) path=\(path)")
                    return ["path": path]
                }
            }
            print("[APIClient] WARNING: failed to parse arguments for tool '\(toolName)': \(argsString.prefix(500))")
            return [:]
        }
        var result: [String: String] = [:]
        for (k, v) in dict {
            if let s = v as? String {
                result[k] = s
            } else if v is NSNull {
                // skip null values
            } else if v is [Any] || v is [String: Any] {
                // Nested arrays/objects — serialize to JSON string
                if let jsonData = try? JSONSerialization.data(withJSONObject: v),
                   let jsonStr = String(data: jsonData, encoding: .utf8) {
                    result[k] = jsonStr
                } else {
                    result[k] = "\(v)"
                }
            } else {
                // Numbers, bools, other scalars
                result[k] = "\(v)"
            }
        }
        // Clean path: models sometimes over-escape, producing paths with literal quotes
        let fileTools: Set<String> = ["writeFile", "readFile", "editFile"]
        if fileTools.contains(toolName), let rawPath = result["path"] {
            let cleaned = cleanPath(rawPath)
            if cleaned != rawPath {
                result["path"] = cleaned
                appendToolLog("CLEANED_PATH: name=\(toolName) raw=\(rawPath) cleaned=\(cleaned)")
            }
        }
        // Layer 2: extract path from truncated JSON when parsing succeeded but path is missing
        if (toolName == "writeFile" || toolName == "editFile") && result["path"] == nil {
            if let path = extractPathFromTruncatedJSON(argsString) {
                result["path"] = path
                appendToolLog("RECOVERED_PATH: name=\(toolName) path=\(path)")
            }
        }
        if result.isEmpty && !dict.isEmpty {
            print("[APIClient] WARNING: all values lost during conversion for tool '\(toolName)': keys=\(dict.keys.joined(separator: ","))")
        }
        return result
    }

    /// Extract `"path"` value from truncated/malformed JSON via string search.
    /// Works even when JSONSerialization fails to parse the full args string.
    private static func extractPathFromTruncatedJSON(_ json: String) -> String? {
        guard let keyRange = json.range(of: "\"path\"") else { return nil }
        let afterKey = json[keyRange.upperBound...]
        guard let colonIdx = afterKey.firstIndex(of: ":") else { return nil }
        let afterColon = afterKey[afterKey.index(after: colonIdx)...]
            .drop(while: { $0 == " " || $0 == "\t" || $0 == "\n" || $0 == "\r" })
        guard afterColon.first == "\"" else { return nil }
        let valueStart = afterColon.index(after: afterColon.startIndex)
        // Walk to closing quote, respecting backslash escapes
        var i = valueStart
        while i < afterColon.endIndex {
            if afterColon[i] == "\\" {
                i = afterColon.index(after: i)
                if i < afterColon.endIndex { i = afterColon.index(after: i) }
                continue
            }
            if afterColon[i] == "\"" {
                var value = String(afterColon[valueStart..<i])
                value = value
                    .replacingOccurrences(of: "\\\"", with: "\"")
                    .replacingOccurrences(of: "\\\\", with: "\\")
                    .replacingOccurrences(of: "\\/", with: "/")
                return cleanPath(value)
            }
            i = afterColon.index(after: i)
        }
        // String truncated before closing quote — use what we have
        let partial = String(afterColon[valueStart...])
        return partial.isEmpty ? nil : cleanPath(partial)
    }

    /// Strip spurious surrounding quotes and whitespace from a path.
    /// Models sometimes over-escape paths, producing `"\"app.py\""` which after
    /// JSON unescaping becomes `"app.py"` with literal quote characters.
    private static func cleanPath(_ path: String) -> String {
        var p = path.trimmingCharacters(in: .whitespacesAndNewlines)
        // Strip one layer of surrounding double quotes
        while p.count >= 2 && p.hasPrefix("\"") && p.hasSuffix("\"") {
            p = String(p.dropFirst().dropLast())
        }
        return p
    }

    /// Append a line to ~/.mlx-serve/tool-calls.log for debugging tool call parsing.
    private static func appendToolLog(_ message: String) {
        let path = NSString(string: "~/.mlx-serve/tool-calls.log").expandingTildeInPath
        let line = "[\(ISO8601DateFormatter().string(from: Date()))] \(message)\n"
        if let handle = FileHandle(forWritingAtPath: path) {
            handle.seekToEndOfFile()
            handle.write(line.data(using: .utf8) ?? Data())
            handle.closeFile()
        } else {
            try? line.write(toFile: path, atomically: true, encoding: .utf8)
        }
    }
}
