import Foundation

/// Lightweight HTTP server for test automation.
/// Exposes the app's agent loop via REST API so tests can exercise the full code path
/// (streaming, tool calling, tool execution, history management) without UI automation.
///
/// Endpoints:
///   POST /test/chat          — Send a message in chat mode (no tools)
///   POST /test/agent         — Start agent job (non-blocking, returns job_id)
///   GET  /test/agent/status  — Poll agent job progress/result
///   POST /test/reset         — Clear chat history and start fresh
///   GET  /test/history       — Get current chat session messages
///   GET  /test/status        — Get server/model status
@MainActor
class TestServer {
    private var listener: Task<Void, Never>?
    private let port: UInt16 = 8090
    weak var appState: AppState?

    /// Background agent job state — allows non-blocking /test/agent requests
    private var agentJobId: String?
    private var agentJobResult: [String: Any]?
    private var agentJobRunning = false
    private var agentJobTask: Task<Void, Never>?

    func start(appState: AppState) {
        self.appState = appState
        listener = Task.detached { [weak self] in
            await self?.listen()
        }
        print("[TestServer] Listening on http://127.0.0.1:\(port)")
    }

    func stop() {
        listener?.cancel()
        listener = nil
    }

    /// Runs the accept loop on a background thread to avoid blocking the main actor.
    private func listen() async {
        // Move blocking socket work to a background thread
        let listenPort = port
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            DispatchQueue.global(qos: .utility).async { [weak self] in
                let serverSocket = Darwin.socket(AF_INET, SOCK_STREAM, 0)
                guard serverSocket >= 0 else {
                    print("[TestServer] Failed to create socket")
                    cont.resume()
                    return
                }

                var opt: Int32 = 1
                setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, socklen_t(MemoryLayout<Int32>.size))

                var addr = sockaddr_in()
                addr.sin_family = sa_family_t(AF_INET)
                addr.sin_port = listenPort.bigEndian
                addr.sin_addr.s_addr = UInt32(INADDR_LOOPBACK).bigEndian

                let bindResult = withUnsafePointer(to: &addr) { ptr in
                    ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.bind(serverSocket, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
                }
                guard bindResult == 0 else {
                    print("[TestServer] Bind failed")
                    Darwin.close(serverSocket)
                    cont.resume()
                    return
                }

                guard Darwin.listen(serverSocket, 5) == 0 else {
                    print("[TestServer] Listen failed")
                    Darwin.close(serverSocket)
                    cont.resume()
                    return
                }

                // Accept loop — each client handled on its own GCD thread
                while true {
                    let client = Darwin.accept(serverSocket, nil, nil)
                    guard client >= 0 else { continue }

                    // Handle each client concurrently so accept loop isn't blocked
                    DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                        // Read full HTTP request (headers + body) — may arrive in multiple packets
                        var data = Data()
                        var buf = [UInt8](repeating: 0, count: 65536)
                        // First read: get headers
                        let n = Darwin.read(client, &buf, buf.count)
                        guard n > 0 else { Darwin.close(client); return }
                        data.append(contentsOf: buf[..<n])

                        // Check if we need to read more (Content-Length)
                        if let headerEnd = data.range(of: Data("\r\n\r\n".utf8)) {
                            let headers = String(data: data[..<headerEnd.lowerBound], encoding: .utf8) ?? ""
                            let bodyStart = headerEnd.upperBound
                            let bodyReceived = data.count - bodyStart
                            // Parse Content-Length
                            var contentLength = 0
                            for line in headers.components(separatedBy: "\r\n") {
                                if line.lowercased().hasPrefix("content-length:") {
                                    contentLength = Int(line.dropFirst(15).trimmingCharacters(in: .whitespaces)) ?? 0
                                }
                            }
                            // Read remaining body if needed
                            while bodyReceived + (data.count - bodyStart - bodyReceived) < contentLength && data.count < 1_000_000 {
                                let remaining = contentLength - (data.count - bodyStart)
                                if remaining <= 0 { break }
                                let readSize = min(remaining, buf.count)
                                let m = Darwin.read(client, &buf, readSize)
                                if m <= 0 { break }
                                data.append(contentsOf: buf[..<m])
                            }
                        }

                        let request = String(data: data, encoding: .utf8) ?? ""
                        let firstLine = request.prefix(while: { $0 != "\r" && $0 != "\n" })
                        let parts = firstLine.split(separator: " ")
                        let method = parts.count > 0 ? String(parts[0]) : ""
                        let path = parts.count > 1 ? String(parts[1]) : ""

                        let body: String
                        if let range = request.range(of: "\r\n\r\n") {
                            body = String(request[range.upperBound...])
                        } else {
                            body = ""
                        }

                        // Hop to MainActor for routing
                        let sem = DispatchSemaphore(value: 0)
                        var responseBody = "{\"error\":\"internal\"}"
                        Task { @MainActor [weak self] in
                            guard let self else {
                                sem.signal()
                                return
                            }
                            switch (method, path) {
                            case ("POST", "/test/start"):
                                responseBody = await self.handleStart(body: body)
                            case ("POST", "/test/stop"):
                                responseBody = await self.handleStop()
                            case ("POST", "/test/reset"):
                                responseBody = await self.handleReset()
                            case ("GET", "/test/history"):
                                responseBody = await self.handleHistory()
                            case ("GET", "/test/status"):
                                responseBody = await self.handleStatus()
                            case ("POST", "/test/chat"):
                                responseBody = await self.handleChat(body: body)
                            case ("POST", "/test/agent"):
                                responseBody = self.startAgentJob(body: body)
                            case ("GET", "/test/agent/status"):
                                responseBody = self.getAgentJobStatus()
                            case ("GET", "/test/context"):
                                responseBody = await self.handleContext()
                            default:
                                responseBody = self.jsonResponse(["error": "Not found: \(method) \(path)"])
                            }
                            sem.signal()
                        }
                        sem.wait()

                        let httpResponse = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: \(responseBody.utf8.count)\r\nConnection: close\r\n\r\n\(responseBody)"
                        _ = httpResponse.withCString { Darwin.write(client, $0, strlen($0)) }
                        Darwin.close(client)
                    }
                }
            }
        }
    }

    // MARK: - Handlers

    private func handleStart(body: String) async -> String {
        guard let appState else { return jsonError("No app state") }
        if case .running = appState.server.status {
            return jsonResponse(["status": "already_running", "model": appState.server.currentModelPath])
        }

        // Optional model path in body, otherwise use selected
        var modelPath = appState.selectedModelPath
        if let data = body.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let mp = json["model"] as? String, !mp.isEmpty {
            modelPath = mp
            appState.selectedModelPath = mp
        }

        guard !modelPath.isEmpty else {
            return jsonError("No model selected and none provided")
        }

        appState.server.start(modelPath: modelPath, options: appState.serverOptions)

        // Wait for server to become healthy (up to 120s for large models)
        for _ in 0..<240 {
            try? await Task.sleep(nanoseconds: 500_000_000)
            if case .running = appState.server.status { break }
            if case .error(let e) = appState.server.status {
                return jsonError("Server failed to start: \(e)")
            }
        }

        let status: String
        switch appState.server.status {
        case .running: status = "running"
        case .starting: status = "starting"
        case .stopped: status = "stopped"
        case .error(let e): status = "error: \(e)"
        }
        return jsonResponse(["status": status, "model": modelPath])
    }

    private func handleStop() async -> String {
        guard let appState else { return jsonError("No app state") }
        appState.server.stop()
        return jsonResponse(["status": "stopped"])
    }

    private func handleReset() async -> String {
        guard let appState else { return jsonError("No app state") }
        // Cancel any running agent job
        agentJobTask?.cancel()
        agentJobTask = nil
        agentJobRunning = false
        agentJobResult = nil
        agentJobId = nil

        appState.chatSessions = []
        appState.activeChatId = nil
        let sessionId = appState.newChatSession()
        appState.saveChatHistory()
        return jsonResponse(["status": "ok", "session_id": sessionId.uuidString])
    }

    private func handleHistory() async -> String {
        guard let appState else { return jsonError("No app state") }
        guard let session = appState.chatSessions.first(where: { $0.id == appState.activeChatId }) else {
            return jsonResponse(["messages": [String]()])
        }
        let messages = session.messages.map { msg -> [String: Any] in
            var d: [String: Any] = [
                "role": msg.role.rawValue,
                "content": msg.content,
                "id": msg.id.uuidString,
                "isStreaming": msg.isStreaming,
                "isAgentSummary": msg.isAgentSummary,
            ]
            if let pt = msg.promptTokens { d["promptTokens"] = pt }
            if let ct = msg.completionTokens { d["completionTokens"] = ct }
            if let tps = msg.tokensPerSecond { d["tokensPerSecond"] = tps }
            if let tcId = msg.toolCallId { d["toolCallId"] = tcId }
            if let tcs = msg.toolCalls {
                d["toolCalls"] = tcs.map { ["id": $0.id, "name": $0.name, "arguments": $0.arguments] }
            }
            return d
        }
        if let data = try? JSONSerialization.data(withJSONObject: ["messages": messages], options: .prettyPrinted),
           let str = String(data: data, encoding: .utf8) {
            return str
        }
        return jsonError("Serialization failed")
    }

    private func handleContext() async -> String {
        guard let appState else { return jsonError("No app state") }
        guard let session = appState.chatSessions.first(where: { $0.id == appState.activeChatId }) else {
            return jsonResponse(["error": "No active session"])
        }

        // Find last message with usage data
        let lastWithUsage = session.messages.last(where: { $0.promptTokens != nil && $0.promptTokens! > 0 })
        let promptTokens = lastWithUsage?.promptTokens ?? 0
        let completionTokens = lastWithUsage?.completionTokens ?? 0

        // Effective context length (same logic as ChatView)
        let contextLength: Int
        if appState.contextSize > 0 {
            contextLength = appState.contextSize
        } else if let modelCtx = appState.server.modelInfo?.contextLength, modelCtx > 0 {
            contextLength = modelCtx
        } else {
            contextLength = 32768
        }

        let usageRatio = contextLength > 0 ? Double(promptTokens) / Double(contextLength) : 0
        let remaining = max(0, contextLength - promptTokens)
        let genBudget = min(remaining, appState.maxTokens)

        return jsonResponse([
            "context_length": contextLength,
            "prompt_tokens": promptTokens,
            "completion_tokens": completionTokens,
            "max_tokens": appState.maxTokens,
            "usage_ratio": Int(usageRatio * 100),
            "generation_budget": genBudget,
            "status": usageRatio > 0.80 ? "critical" : usageRatio > 0.60 ? "warning" : "ok",
        ])
    }

    private func handleStatus() async -> String {
        guard let appState else { return jsonError("No app state") }
        var serverStatus: String
        switch appState.server.status {
        case .running: serverStatus = "running"
        case .starting: serverStatus = "starting"
        case .stopped: serverStatus = "stopped"
        case .error(let e): serverStatus = "error: \(e)"
        }
        // If ServerManager says "starting" but health endpoint is ok, force transition
        if serverStatus == "starting" {
            let api = APIClient()
            if let healthy = try? await api.checkHealth(port: appState.server.port), healthy {
                appState.server.forceRunning()
                serverStatus = "running"
            }
        }
        return jsonResponse([
            "server": serverStatus,
            "port": appState.server.port,
            "model": appState.server.currentModelPath,
            "sessions": appState.chatSessions.count,
        ])
    }

    private func handleChat(body: String) async -> String {
        guard let appState else { return jsonError("No app state") }
        if case .running = appState.server.status {
            // ok
        } else if await isServerHealthy(appState: appState) {
            appState.server.forceRunning()
        } else {
            return jsonError("Server not running")
        }

        guard let data = body.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let message = json["message"] as? String else {
            return jsonError("Missing 'message' field")
        }

        // Ensure we have an active session
        let sessionId: UUID
        if let active = appState.activeChatId {
            sessionId = active
        } else {
            sessionId = appState.newChatSession()
        }

        // Add user message
        let userMsg = ChatMessage(role: .user, content: message)
        appState.appendMessage(to: sessionId, message: userMsg)

        let api = APIClient()
        // Build the request body BEFORE appending the streaming placeholder
        // so the placeholder never lands in the prompt. Session is the source
        // of truth — no re-adding the user message (that double-add caused
        // DSV4-Flash to degenerate into a repeating-token loop at temp > 0).
        let messagesArray = appState.chatSessions
            .first(where: { $0.id == sessionId })?.messages
            .filter { !$0.isAgentSummary }
            .map { ["role": $0.role.rawValue, "content": $0.content] as [String: Any] }
            ?? []

        // Streaming placeholder for the UI / history endpoint, appended after
        // the request body so it's not part of the prompt.
        var assistantMsg = ChatMessage(role: .assistant, content: "")
        assistantMsg.isStreaming = true
        appState.appendMessage(to: sessionId, message: assistantMsg)

        do {
            let stream = api.streamChat(
                port: appState.server.port,
                messages: messagesArray,
                maxTokens: appState.maxTokens
            )
            for try await event in stream {
                switch event {
                case .content(let text):
                    appState.updateLastMessage(in: sessionId, content: text)
                case .reasoning(let text):
                    appState.updateLastMessage(in: sessionId, reasoning: text)
                case .usage(let usage):
                    appState.updateLastMessage(in: sessionId, usage: usage)
                case .toolCalls, .maxTokensReached, .done:
                    break
                }
            }
        } catch {
            appState.updateLastMessage(in: sessionId, content: "\n[Error: \(error.localizedDescription)]")
        }
        appState.updateLastMessage(in: sessionId, streaming: false)
        appState.saveChatHistory()

        // Return the final message
        let finalContent = appState.chatSessions
            .first(where: { $0.id == sessionId })?.messages.last?.content ?? ""
        return jsonResponse(["status": "ok", "content": finalContent])
    }

    /// Start agent job in a background Task so it doesn't block other endpoints.
    private func startAgentJob(body: String) -> String {
        if agentJobRunning {
            return jsonResponse(["status": "busy", "job_id": agentJobId ?? ""])
        }
        let jobId = UUID().uuidString
        agentJobId = jobId
        agentJobResult = nil
        agentJobRunning = true

        agentJobTask = Task { [weak self] in
            guard let self else { return }
            let result = await self.handleAgent(body: body)
            self.agentJobResult = (try? JSONSerialization.jsonObject(with: result.data(using: .utf8) ?? Data())) as? [String: Any]
            self.agentJobRunning = false
        }

        return jsonResponse(["status": "started", "job_id": jobId])
    }

    /// Poll for agent job result.
    private func getAgentJobStatus() -> String {
        if agentJobRunning {
            // Include current round count from session history for progress monitoring
            let msgCount = appState?.chatSessions
                .first(where: { $0.id == appState?.activeChatId })?.messages.count ?? 0
            return jsonResponse(["status": "running", "job_id": agentJobId ?? "", "message_count": msgCount])
        }
        if let result = agentJobResult {
            return jsonResponse(result)
        }
        return jsonResponse(["status": "idle"])
    }

    private func handleAgent(body: String) async -> String {
        guard let appState else { return jsonError("No app state") }
        if case .running = appState.server.status {
            // ok
        } else if await isServerHealthy(appState: appState) {
            appState.server.forceRunning()
        } else {
            return jsonError("Server not running")
        }

        guard let data = body.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let message = json["message"] as? String else {
            return jsonError("Missing 'message' field")
        }

        let enableThinking = json["thinking"] as? Bool ?? false
        var workDir: String? = json["working_directory"] as? String
            ?? NSString(string: "~/.mlx-serve/workspace").expandingTildeInPath

        // Ensure we have an active session in agent mode
        let sessionId: UUID
        if let active = appState.activeChatId {
            sessionId = active
        } else {
            sessionId = appState.newChatSession()
        }
        if let idx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }) {
            appState.chatSessions[idx].mode = .agent
        }

        let userMsg = ChatMessage(role: .user, content: message)
        appState.appendMessage(to: sessionId, message: userMsg)

        let api = APIClient()
        let maxIterations = json["max_rounds"] as? Int ?? 10
        var padRetries = 0
        let maxPadRetries = 2
        var roundResults: [[String: Any]] = []
        let repetition = AgentEngine.RepetitionTracker()
        var truncationRetries = 0

        for iteration in 0..<maxIterations {
            // Build history using shared engine
            let contextLength = AgentEngine.effectiveContextLength(
                appContextSize: appState.contextSize,
                modelContextLength: appState.server.modelInfo?.contextLength
            )
            let session = appState.chatSessions.first(where: { $0.id == sessionId })
            var history = AgentEngine.buildAgentHistory(
                messages: session?.messages ?? [],
                contextLength: contextLength,
                maxTokens: appState.maxTokens
            )
            let userContent = history.last { ($0["role"] as? String) == "user" }?["content"] as? String ?? ""
            let skills = AgentPrompt.skillManager.matchingSkills(for: userContent)
            var systemPrompt = AgentPrompt.systemPrompt + skills + AgentPrompt.memory + appState.agentMemory.contextSnippet()
            if let wd = workDir {
                systemPrompt += AgentEngine.workingDirectoryContext(wd)
            }
            var messages: [[String: Any]] = [["role": "system", "content": systemPrompt]]
            if let lastRole = history.last?["role"] as? String, lastRole == "tool" {
                history.append(["role": "user", "content": "Continue. If the task is done, summarize the result. If not, take the next step."])
            }
            messages.append(contentsOf: history)

            // Add streaming assistant message
            var streamMsg = ChatMessage(role: .assistant, content: "")
            streamMsg.isStreaming = true
            appState.appendMessage(to: sessionId, message: streamMsg)

            // Stream model response with tools
            var receivedToolCalls: [APIClient.ToolCall] = []
            var maxTokensHit = false
            let stream = api.streamChat(
                port: appState.server.port,
                messages: messages,
                maxTokens: appState.maxTokens,
                temperature: 0.7,
                enableThinking: enableThinking && iteration == 0,
                toolsJSON: AgentPrompt.toolDefinitionsJSON
            )

            do {
                for try await event in stream {
                    switch event {
                    case .content(let text):
                        appState.updateLastMessage(in: sessionId, content: text)
                    case .reasoning(let text):
                        appState.updateLastMessage(in: sessionId, reasoning: text)
                    case .usage(let usage):
                        appState.updateLastMessage(in: sessionId, usage: usage)
                    case .toolCalls(let calls):
                        receivedToolCalls = calls
                    case .maxTokensReached:
                        maxTokensHit = true
                        appState.updateLastMessage(in: sessionId, content: "\n\n⚠️ *Output truncated — max tokens reached.*")
                    case .done:
                        break
                    }
                }
            } catch {
                appState.updateLastMessage(in: sessionId, content: "\n[Error: \(error.localizedDescription)]")
                break
            }
            appState.updateLastMessage(in: sessionId, streaming: false)

            // Truncation recovery
            if maxTokensHit && !receivedToolCalls.isEmpty && truncationRetries < 2 {
                truncationRetries += 1
                if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
                   !appState.chatSessions[sIdx].messages.isEmpty {
                    appState.chatSessions[sIdx].messages.removeLast()
                }
                let nudge = ChatMessage(role: .user, content: "[System: Your last response was cut off because the output was too long, so the tool call was NOT executed. Write shorter responses: for a large file, write it in chunks — writeFile a first part, then editFile to append the rest. (A shell heredoc has the same length limit, so don't switch to that.)]")
                appState.appendMessage(to: sessionId, message: nudge)
                continue
            }

            // Pad detection
            if receivedToolCalls.isEmpty {
                let lastContent = appState.chatSessions
                    .first(where: { $0.id == sessionId })?.messages.last?.content ?? ""
                let cleaned = lastContent.replacingOccurrences(of: "<pad>", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
                if cleaned.isEmpty {
                    if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
                       !appState.chatSessions[sIdx].messages.isEmpty {
                        appState.chatSessions[sIdx].messages.removeLast()
                    }
                    padRetries += 1
                    if padRetries <= maxPadRetries { continue }
                    roundResults.append(["round": iteration + 1, "type": "pad_failure"])
                    break
                }
                roundResults.append(["round": iteration + 1, "type": "content", "content": String(cleaned.prefix(500))])
                break
            }

            // Store tool calls on message
            if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
               !appState.chatSessions[sIdx].messages.isEmpty {
                let mIdx = appState.chatSessions[sIdx].messages.count - 1
                appState.chatSessions[sIdx].messages[mIdx].toolCalls = receivedToolCalls.map { tc in
                    let argsJson = (try? JSONSerialization.data(withJSONObject: tc.arguments))
                        .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                    return SerializedToolCall(id: tc.id, name: tc.name, arguments: argsJson)
                }
            }

            // Track repetition
            repetition.track(toolCalls: receivedToolCalls)

            // Execute tools using shared engine
            for tc in receivedToolCalls {
                let result = await AgentEngine.executeToolCall(
                    tc, workingDirectory: &workDir,
                    repetition: repetition, iteration: iteration,
                    agentMemory: appState.agentMemory,
                    processRegistry: appState.processRegistry,
                    sessionId: sessionId
                )

                var resultMsg = ChatMessage(role: .assistant, content: "**\(result.name)** → \(String(result.output.prefix(500)))")
                resultMsg.isAgentSummary = true
                appState.appendMessage(to: sessionId, message: resultMsg)

                // Store tool result
                var toolMsg = ChatMessage(role: .system, content: "")
                toolMsg.toolCallId = result.id
                toolMsg.toolName = result.name
                toolMsg.content = AgentEngine.truncateWithOverflow(result.output, toolCallId: result.id, toolName: result.name)
                appState.appendMessage(to: sessionId, message: toolMsg)
            }

            let callSummary = receivedToolCalls.map { "\($0.name)(\($0.arguments.keys.joined(separator: ",")))" }.joined(separator: ", ")
            roundResults.append(["round": iteration + 1, "type": "tool_calls", "tools": callSummary])
        }

        appState.saveChatHistory()

        let finalContent = appState.chatSessions
            .first(where: { $0.id == sessionId })?.messages
            .last(where: { $0.role == .assistant && !$0.isAgentSummary })?.content ?? ""
        let cleaned = finalContent.replacingOccurrences(of: "<pad>", with: "").trimmingCharacters(in: .whitespacesAndNewlines)

        return jsonResponse([
            "status": "ok",
            "rounds": roundResults,
            "final_content": String(cleaned.prefix(1000)),
            "message_count": appState.chatSessions.first(where: { $0.id == sessionId })?.messages.count ?? 0,
        ])
    }

    private func isServerHealthy(appState: AppState) async -> Bool {
        let api = APIClient()
        return (try? await api.checkHealth(port: appState.server.port)) ?? false
    }

    private func jsonResponse(_ dict: [String: Any]) -> String {
        if let data = try? JSONSerialization.data(withJSONObject: dict, options: .prettyPrinted),
           let str = String(data: data, encoding: .utf8) {
            return str
        }
        return "{\"error\": \"serialization failed\"}"
    }

    private func jsonError(_ msg: String) -> String {
        jsonResponse(["error": msg])
    }
}
