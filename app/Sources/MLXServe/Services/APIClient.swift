import Foundation

enum SSEEvent {
    case content(String)
    case reasoning(String)
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
        let url = URL(string: "http://localhost:\(port)/health")!
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
        let url = URL(string: "http://localhost:\(port)/v1/models")!
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
        let url = URL(string: "http://localhost:\(port)/props")!
        let (data, _) = try await session.data(from: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let mem = json["memory"] as? [String: Any] else { return nil }
        return MemoryInfo(
            activeBytes: mem["active_bytes"] as? Int64 ?? 0,
            peakBytes: mem["peak_bytes"] as? Int64 ?? 0
        )
    }

    func streamAgentChat(
        port: UInt16,
        messages: [[String: Any]],
        systemPrompt: String
    ) -> AsyncThrowingStream<SSEEvent, Error> {
        var allMessages: [[String: Any]] = [["role": "system", "content": systemPrompt]]
        allMessages.append(contentsOf: messages)
        return streamChat(
            port: port,
            messages: allMessages,
            maxTokens: 4096,
            temperature: 0.7
        )
    }

    func streamChat(
        port: UInt16,
        messages: [[String: Any]],
        maxTokens: Int = 2048,
        temperature: Double = 0.8,
        enableThinking: Bool = false
    ) -> AsyncThrowingStream<SSEEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let url = URL(string: "http://localhost:\(port)/v1/chat/completions")!
                    var request = URLRequest(url: url)
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    request.timeoutInterval = 300

                    var body: [String: Any] = [
                        "model": "mlx-serve",
                        "messages": messages,
                        "max_tokens": maxTokens,
                        "temperature": temperature,
                        "top_p": 0.95,
                        "stream": true,
                    ]
                    if enableThinking {
                        body["enable_thinking"] = true
                    }
                    request.httpBody = try JSONSerialization.data(withJSONObject: body)

                    let (bytes, response) = try await self.session.bytes(for: request)
                    guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                        continuation.finish(throwing: URLError(.badServerResponse))
                        return
                    }

                    for try await line in bytes.lines {
                        guard line.hasPrefix("data: ") else { continue }
                        let payload = String(line.dropFirst(6))
                        if payload == "[DONE]" {
                            continuation.yield(.done)
                            break
                        }

                        guard let jsonData = payload.data(using: .utf8),
                              let chunk = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
                              let choices = chunk["choices"] as? [[String: Any]],
                              let delta = choices.first?["delta"] as? [String: Any] else {
                            continue
                        }

                        if let content = delta["content"] as? String, !content.isEmpty {
                            continuation.yield(.content(content))
                        }
                        if let reasoning = delta["reasoning_content"] as? String, !reasoning.isEmpty {
                            continuation.yield(.reasoning(reasoning))
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
