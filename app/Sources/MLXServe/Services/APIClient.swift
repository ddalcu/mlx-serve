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

/// Surfacing real HTTP errors instead of the cryptic NSURLErrorDomain -1011. The body snippet is
/// truncated server output (typically a JSON error from mlx-serve, e.g. "Prompt too long").
enum APIError: LocalizedError {
    case badStatus(code: Int, detail: String)

    var errorDescription: String? {
        switch self {
        case .badStatus(let code, let detail):
            // Friendly message for the most common failure: combined system prompt + tools blew past the
            // model's context window. Tool-heavy MCP servers can do this on their own.
            if Self.looksLikeContextOverflow(detail) {
                return """
                Your prompt + tool definitions are larger than the model's context window. Try one of:
                  • Increase context size: menu bar → Settings → bump Context (needs more RAM)
                  • Use a model with a larger context window
                  • Disable some MCP servers (gear icon on the MCP pill) or turn off Agent mode
                """
            }
            if detail.isEmpty {
                return "HTTP \(code) from mlx-serve. Check the server log (menu bar → log icon)."
            }
            return "HTTP \(code) from mlx-serve\(detail)"
        }
    }

    private static func looksLikeContextOverflow(_ detail: String) -> Bool {
        let lower = detail.lowercased()
        return lower.contains("exceeds maximum context length")
            || lower.contains("context length exceeded")
            || lower.contains("prompt too long")
            || lower.contains("maximum context")
    }
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
        // Backwards-compat single-model accessor. Returns the first entry
        // (the server sorts the default model to index 0).
        let all = try await fetchAllModels(port: port)
        return all.first
    }

    /// Plan 05 Phase G — fetch every entry the registry knows about. New
    /// callers prefer this over `fetchModels(port:)` so the picker UI can
    /// show loaded/unloaded badges per model.
    func fetchAllModels(port: UInt16) async throws -> [ModelInfo] {
        let url = URL(string: "http://127.0.0.1:\(port)/v1/models")!
        let (data, _) = try await session.data(from: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let dataArr = json["data"] as? [[String: Any]] else { return [] }
        return dataArr.map { Self.parseModelInfo($0) }
    }

    // Internal (not private) so unit tests can exercise the capability/meta
    // parsing directly via `@testable import`.
    static func parseModelInfo(_ first: [String: Any]) -> ModelInfo {
        let name = first["id"] as? String ?? "unknown"
        let meta = first["meta"] as? [String: Any] ?? [:]
        // Pre-Phase-G servers omit `state`/`loaded`/`bytes_resident` at the
        // top level. Default to `loaded=true` so existing single-model
        // behavior continues unchanged when a new app talks to an old server.
        let topLoaded = first["loaded"] as? Bool ?? true
        let topState = first["state"] as? String
        let topBytesResident = first["bytes_resident"] as? UInt64 ?? 0
        let topBytesOnDisk: UInt64? = {
            if let v = first["bytes_on_disk"] as? UInt64 { return v }
            if let v = meta["bytes_on_disk"] as? UInt64 { return v }
            return nil
        }()
        // Audio support is advertised at the top level of the model object
        // (capabilities / input_modalities), not under `meta`.
        let caps = first["capabilities"] as? [String] ?? []
        let mods = first["input_modalities"] as? [String] ?? []
        let supportsAudio = caps.contains("audio") || mods.contains("audio")
        // Vision is advertised the same way; it drops out automatically when the
        // server runs with `--no-vision` (the encoder isn't loaded), so this is
        // the live "can this model see images right now?" signal.
        let supportsVision = caps.contains("vision") || mods.contains("image")
        // Encoder-only entries advertise "embeddings" even as unloaded stubs
        // (the server peeks model_type at discovery); architecture is the
        // belt-and-suspenders signal for loaded entries.
        let supportsEmbeddings = caps.contains("embeddings")
            || (meta["architecture"] as? String) == "bert"
        return ModelInfo(
            name: name,
            quantBits: meta["quantization_bits"] as? Int ?? 0,
            layers: meta["num_layers"] as? Int ?? 0,
            hiddenSize: meta["hidden_size"] as? Int ?? 0,
            vocabSize: meta["vocab_size"] as? Int ?? 0,
            contextLength: meta["context_length"] as? Int ?? 0,
            modelMaxTokens: meta["model_max_tokens"] as? Int ?? 0,
            architecture: meta["architecture"] as? String ?? "",
            isMoE: meta["is_moe"] as? Bool ?? false,
            supportsAudio: supportsAudio,
            supportsVision: supportsVision,
            supportsEmbeddings: supportsEmbeddings,
            capabilities: caps,
            drafterLoaded: meta["drafter_loaded"] as? Bool ?? false,
            drafterPath: meta["drafter_path"] as? String,
            mtpLoaded: meta["mtp_loaded"] as? Bool ?? false,
            loaded: topLoaded,
            state: topState,
            bytesResident: topBytesResident,
            bytesOnDisk: topBytesOnDisk,
            // Model-author sampling recommendations from generation_config.json.
            // Emitted as JSON `null` (→ NSNull → nil) when absent, or the keys
            // are missing entirely on a pre-this-feature server.
            recTemperature: meta["gen_temperature"] as? Double,
            recTopP: meta["gen_top_p"] as? Double,
            recTopK: meta["gen_top_k"] as? Int
        )
    }

    /// Plan 05 Phase G — POST /v1/load-model. Returns the resulting
    /// `ModelInfo` after the load completes (blocks for seconds on a cold
    /// load). Throws if the id is unknown (404) or load fails (500).
    func loadModel(port: UInt16, id: String, drafterPath: String? = nil) async throws -> ModelInfo {
        let url = URL(string: "http://127.0.0.1:\(port)/v1/load-model")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        // Load can take 10–60 s on a fresh model; raise above the default.
        request.timeoutInterval = 180
        var body: [String: Any] = ["model": id]
        if let drafterPath { body["drafter_path"] = drafterPath }
        // withoutEscapingSlashes: `id` may be an absolute path (the
        // auto-downloaded encoder registers by path) — keep it readable in
        // logs. The server unescapes either form.
        request.httpBody = try JSONSerialization.data(withJSONObject: body, options: [.withoutEscapingSlashes])
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            let snippet = String(data: data, encoding: .utf8)?.prefix(300) ?? ""
            throw APIError.badStatus(code: code, detail: String(snippet))
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let modelObj = json["model"] as? [String: Any] else {
            throw URLError(.cannotParseResponse)
        }
        return Self.parseModelInfo(modelObj)
    }

    /// POST /v1/unload-model — free a model's resident GPU state (the registry
    /// keeps the stub so it can reload). Used by the media-gen
    /// load→generate→unload flow. Idempotent server-side; a non-resident model
    /// returns 200.
    func unloadModel(port: UInt16, id: String) async throws {
        let url = URL(string: "http://127.0.0.1:\(port)/v1/unload-model")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 120
        request.httpBody = try JSONSerialization.data(withJSONObject: ["model": id], options: [.withoutEscapingSlashes])
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            let snippet = String(data: data, encoding: .utf8)?.prefix(300) ?? ""
            throw APIError.badStatus(code: code, detail: String(snippet))
        }
    }

    /// POST a media-generation endpoint with `stream:true` and yield each parsed
    /// SSE `data:` event as a dictionary (`{type:"progress"|"complete"|"error", …}`)
    /// against the MAIN server port. Replaces `NativeGenServer.postStream`. The
    /// connection is torn down when the consuming task is cancelled.
    nonisolated func streamGeneration(port: UInt16, path: String, json: [String: Any]) -> AsyncThrowingStream<[String: Any], Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var body = json
                    body["stream"] = true
                    var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
                    req.httpMethod = "POST"
                    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                    req.httpBody = try JSONSerialization.data(withJSONObject: body, options: [.withoutEscapingSlashes])
                    req.timeoutInterval = 900
                    let (bytes, resp) = try await URLSession.shared.bytes(for: req)
                    let code = (resp as? HTTPURLResponse)?.statusCode ?? -1
                    guard code == 200 else { throw APIError.badStatus(code: code, detail: "stream start failed") }
                    for try await line in bytes.lines {
                        guard line.hasPrefix("data: ") else { continue }
                        let payload = String(line.dropFirst(6))
                        if let d = payload.data(using: .utf8),
                           let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
                            continuation.yield(obj)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// POST /v1/embeddings — batch-embed `input` with an encoder-only model
    /// (the server runs one padded masked GPU forward per 64-text chunk).
    /// Returns one vector per input, in input order.
    func embeddings(port: UInt16, model: String, input: [String]) async throws -> [[Double]] {
        let url = URL(string: "http://127.0.0.1:\(port)/v1/embeddings")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        // First call may cold-load the encoder model (seconds, not minutes).
        request.timeoutInterval = 120
        request.httpBody = try JSONSerialization.data(withJSONObject: ["model": model, "input": input])
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            let snippet = String(data: data, encoding: .utf8)?.prefix(300) ?? ""
            throw APIError.badStatus(code: code, detail: String(snippet))
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rows = json["data"] as? [[String: Any]] else {
            throw URLError(.cannotParseResponse)
        }
        let ordered = rows.sorted { ($0["index"] as? Int ?? 0) < ($1["index"] as? Int ?? 0) }
        return ordered.compactMap { row in
            (row["embedding"] as? [Any])?.compactMap { ($0 as? NSNumber)?.doubleValue }
        }
    }

    func fetchProps(port: UInt16) async throws -> MemoryInfo? {
        let url = URL(string: "http://127.0.0.1:\(port)/props")!
        let (data, _) = try await session.data(from: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let mem = json["memory"] as? [String: Any] else { return nil }
        return MemoryInfo.parse(mem)
    }

    // MARK: - Agent Tool Calling

    struct ToolCall {
        let id: String
        let name: String
        let arguments: [String: String]
        let rawArguments: String
    }

    /// Per-request overrides that come from the user's saved ServerOptions.
    /// Each field is optional — `nil` = leave it out of the request body and
    /// let the server's default win. Pre-built once at the call site (usually
    /// from `ServerOptions`) and passed through unchanged.
    struct RequestDefaults {
        var topP: Double? = nil
        var topK: Int? = nil
        var repeatPenalty: Double? = nil
        var presencePenalty: Double? = nil
        var reasoningBudget: Int? = nil
        var enablePLD: Bool? = nil
        var enableDrafter: Bool? = nil

        static let none = RequestDefaults()

        /// Build from the user's saved settings: per-request TriState overrides
        /// translate to optional booleans; numeric defaults are forwarded only
        /// when they differ from the canonical "off" value (e.g. topK=0 stays
        /// nil — the server already treats 0 as disabled and we want to avoid
        /// gratuitously bloating every request body).
        static func from(_ opts: ServerOptions) -> RequestDefaults {
            var r = RequestDefaults()
            r.topP = opts.defaultTopP
            r.topK = opts.defaultTopK > 0 ? opts.defaultTopK : nil
            r.repeatPenalty = opts.defaultRepeatPenalty != 1.0 ? opts.defaultRepeatPenalty : nil
            r.presencePenalty = opts.defaultPresencePenalty != 0.0 ? opts.defaultPresencePenalty : nil
            r.reasoningBudget = opts.defaultReasoningBudget >= 0 ? opts.defaultReasoningBudget : nil
            r.enablePLD = opts.perRequestEnablePLD.asOptionalBool
            r.enableDrafter = opts.perRequestEnableDrafter.asOptionalBool
            return r
        }
    }

    func streamChat(
        port: UInt16,
        messages: [[String: Any]],
        maxTokens: Int = 2048,
        temperature: Double = 0.8,
        enableThinking: Bool = false,
        tools: [[String: Any]]? = nil,
        toolsJSON: String? = nil,
        defaults: RequestDefaults = .none,
        retryPolicy: RetryPolicy = .default,
        modelId: String? = nil
    ) -> AsyncThrowingStream<SSEEvent, Error> {
        AsyncThrowingStream { continuation in
            // Cancellation plumbing: AsyncThrowingStream does NOT propagate
            // consumer-side Task cancellation into the producer Task. Without
            // this wiring, clicking Stop in the UI cancels the outer agent
            // Task but the inner producer Task keeps pulling bytes from
            // URLSession — the server sees a healthy reader, keeps generating
            // tokens, GPU pegged for the rest of max_tokens.
            //
            // The fix: capture the producer Task and cancel it via
            // `continuation.onTermination`. Cancelling propagates into the
            // `for try await line in bytes.lines` loop inside performStream,
            // which closes the URLSessionDataTask, which FINs the TCP
            // connection, which surfaces to the server as `peerClosed → true`
            // on its next ts.next() iteration.
            let producerTask = Task {
                var lastError: Error?
                for attempt in 0...retryPolicy.maxRetries {
                    do {
                        try await self.performStream(
                            port: port, messages: messages,
                            maxTokens: maxTokens, temperature: temperature,
                            enableThinking: enableThinking, tools: tools,
                            toolsJSON: toolsJSON,
                            defaults: defaults,
                            modelId: modelId,
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
            continuation.onTermination = { _ in
                producerTask.cancel()
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
        defaults: RequestDefaults = .none,
        modelId: String? = nil,
        continuation: AsyncThrowingStream<SSEEvent, Error>.Continuation
    ) async throws {
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("close", forHTTPHeaderField: "Connection")
        request.timeoutInterval = 300

        // Effective top_p: use the user's saved default when set, else 0.95.
        // (`top_p` is special — it has a sane non-disabled default we want to
        // keep on every request.)
        let effectiveTopP = defaults.topP ?? 0.95

        // Plan 05 Phase G — pin the request to the user's selected model
        // when known. Defaults to "mlx-serve" which the server resolves
        // to its default loaded model (matches pre-Phase-G behavior).
        let effectiveModelId = modelId ?? "mlx-serve"
        if let toolsJSON {
            // Splice pre-serialized tools JSON to preserve property key order
            let messagesData = try JSONSerialization.data(withJSONObject: messages)
            guard let messagesStr = String(data: messagesData, encoding: .utf8) else {
                continuation.finish(throwing: URLError(.cannotParseResponse))
                return
            }
            let escapedModelId = effectiveModelId
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
            var parts = [
                "\"model\":\"\(escapedModelId)\"",
                "\"messages\":\(messagesStr)",
                "\"temperature\":\(temperature)",
                "\"top_p\":\(effectiveTopP)",
                "\"stream\":true",
                "\"stream_options\":{\"include_usage\":true}",
            ]
            // maxTokens <= 0 means "Auto": omit the field so the server pegs
            // generation to the remaining context window.
            if maxTokens > 0 { parts.append("\"max_tokens\":\(maxTokens)") }
            if enableThinking { parts.append("\"enable_thinking\":true") }
            if let v = defaults.topK { parts.append("\"top_k\":\(v)") }
            if let v = defaults.repeatPenalty { parts.append("\"repeat_penalty\":\(v)") }
            if let v = defaults.presencePenalty { parts.append("\"presence_penalty\":\(v)") }
            if let v = defaults.reasoningBudget { parts.append("\"reasoning_budget\":\(v)") }
            if let v = defaults.enablePLD { parts.append("\"enable_pld\":\(v)") }
            if let v = defaults.enableDrafter { parts.append("\"enable_drafter\":\(v)") }
            parts.append("\"tools\":\(toolsJSON)")
            request.httpBody = "{\(parts.joined(separator: ","))}".data(using: .utf8)
        } else {
            var body: [String: Any] = [
                "model": effectiveModelId,
                "messages": messages,
                "temperature": temperature,
                "top_p": effectiveTopP,
                "stream": true,
                "stream_options": ["include_usage": true],
            ]
            // maxTokens <= 0 means "Auto": omit so the server pegs to context.
            if maxTokens > 0 { body["max_tokens"] = maxTokens }
            if enableThinking { body["enable_thinking"] = true }
            if let v = defaults.topK { body["top_k"] = v }
            if let v = defaults.repeatPenalty { body["repeat_penalty"] = v }
            if let v = defaults.presencePenalty { body["presence_penalty"] = v }
            if let v = defaults.reasoningBudget { body["reasoning_budget"] = v }
            if let v = defaults.enablePLD { body["enable_pld"] = v }
            if let v = defaults.enableDrafter { body["enable_drafter"] = v }
            if let tools { body["tools"] = tools }
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        }

        let streamStart = Date()
        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            // Drain a small slice of the body for context — the server typically returns a JSON error.
            var body = ""
            do {
                for try await line in bytes.lines.prefix(10) {
                    body += line + "\n"
                    if body.count > 800 { break }
                }
            } catch {}
            let snippet = body.trimmingCharacters(in: .whitespacesAndNewlines)
            print("[APIClient] Bad response: HTTP \(statusCode) body=\(snippet.prefix(400))")
            try? "Bad response: HTTP \(statusCode)\nBody: \(snippet)\n".write(toFile: NSString(string: "~/.mlx-serve/debug.log").expandingTildeInPath, atomically: true, encoding: .utf8)
            let detail = snippet.isEmpty ? "" : " — \(snippet.prefix(300))"
            continuation.finish(throwing: APIError.badStatus(code: statusCode, detail: String(detail)))
            return
        }

        var firstTokenTime: Date?
        // Accumulate tool call deltas across chunks, keyed by index
        var pendingToolCalls: [String: (id: String, name: String, args: String)] = [:]
        var hasToolCalls = false
        var emittedToolCalls = false
        // Accumulated assistant content — used as last-resort source for tool-call
        // recovery when the server streams <tool_call> blocks as plain content
        // (e.g. some Qwen MoE outputs an older binary failed to parse).
        var contentAccumulator = ""

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

            // Parse usage from final chunk. Prefer the server's authoritative
            // decode throughput (`timings.predicted_per_second`, llama.cpp shape)
            // when present — it measures actual GPU decode time, not
            // wall-clock-plus-SSE-buffering. Falls back to client wall-clock for
            // older mlx-serve builds and OpenAI-compat endpoints that don't
            // emit the field.
            if let usage = chunk["usage"] as? [String: Any], usage["prompt_tokens"] != nil {
                let prompt = usage["prompt_tokens"] as? Int ?? 0
                let completion = usage["completion_tokens"] as? Int ?? 0
                let total = usage["total_tokens"] as? Int ?? 0
                let tps: Double = {
                    if let timings = chunk["timings"] as? [String: Any],
                       let serverTps = (timings["predicted_per_second"] as? Double)
                        ?? (timings["predicted_per_second"] as? Int).map(Double.init),
                       serverTps > 0 {
                        return serverTps
                    }
                    let elapsed = Date().timeIntervalSince(firstTokenTime ?? streamStart)
                    return elapsed > 0 ? Double(completion) / elapsed : 0
                }()
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
                contentAccumulator += content
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
                emittedToolCalls = true
            }
            if !calls.isEmpty {
                continuation.yield(.toolCalls(calls))
            }
        }

        // Last-resort: server emitted no tool_calls deltas at all but the
        // assistant content contains <tool_call>...</tool_call> blocks (older
        // server binary / unrecognized format). Recover them from content.
        if !emittedToolCalls && contentAccumulator.contains("<tool_call>") {
            let recovered = Self.extractToolCallsFromContent(contentAccumulator)
            if !recovered.isEmpty {
                Self.appendToolLog("CONTENT_SCAN_RECOVER: count=\(recovered.count)")
                continuation.yield(.toolCalls(recovered))
            }
        }

        continuation.finish()
    }

    /// Extract `<tool_call>{json}</tool_call>` blocks from assistant content.
    /// Handles both standard `{"name":"x","arguments":{...}}` and flat
    /// `{"name":"x","key":"value"}` shapes.
    static func extractToolCallsFromContent(_ content: String) -> [ToolCall] {
        var calls: [ToolCall] = []
        var cursor = content.startIndex
        var index = 0
        while let openRange = content.range(of: "<tool_call>", range: cursor..<content.endIndex) {
            guard let closeRange = content.range(of: "</tool_call>", range: openRange.upperBound..<content.endIndex) else {
                break
            }
            let body = content[openRange.upperBound..<closeRange.lowerBound]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            cursor = closeRange.upperBound

            guard let data = body.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let name = obj["name"] as? String, !name.isEmpty else {
                continue
            }

            // Locate args: prefer "arguments" / "parameters", else synthesize
            // from non-metadata top-level keys (Qwen MoE flat shape).
            var argsDict: [String: Any] = [:]
            if let nested = obj["arguments"] as? [String: Any] {
                argsDict = nested
            } else if let nestedStr = obj["arguments"] as? String,
                      let d = nestedStr.data(using: .utf8),
                      let parsed = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
                argsDict = parsed
            } else if let nested = obj["parameters"] as? [String: Any] {
                argsDict = nested
            } else {
                for (k, v) in obj {
                    if k == "name" || k == "id" || k == "type" { continue }
                    argsDict[k] = v
                }
            }
            if argsDict.isEmpty { continue }

            let argsJSON = (try? JSONSerialization.data(withJSONObject: argsDict))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
            let argsStrings: [String: String] = argsDict.reduce(into: [:]) { acc, kv in
                if let s = kv.value as? String {
                    acc[kv.key] = s
                } else if JSONSerialization.isValidJSONObject([kv.value]),
                          let data = try? JSONSerialization.data(withJSONObject: [kv.value]),
                          let arrStr = String(data: data, encoding: .utf8),
                          arrStr.count >= 2 {
                    // strip the wrapping `[...]`
                    acc[kv.key] = String(arrStr.dropFirst().dropLast())
                } else {
                    acc[kv.key] = "\(kv.value)"
                }
            }
            calls.append(ToolCall(
                id: "call_recovered_\(index)",
                name: name,
                arguments: argsStrings,
                rawArguments: argsJSON
            ))
            index += 1
        }
        return calls
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
