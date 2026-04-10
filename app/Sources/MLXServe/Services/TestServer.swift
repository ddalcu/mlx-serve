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
                addr.sin_addr.s_addr = INADDR_ANY

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

        appState.server.start(modelPath: modelPath)

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

        // Stream response (non-agent, no tools)
        var assistantMsg = ChatMessage(role: .assistant, content: "")
        assistantMsg.isStreaming = true
        appState.appendMessage(to: sessionId, message: assistantMsg)

        let api = APIClient()
        let messages = appState.chatSessions
            .first(where: { $0.id == sessionId })?.messages
            .filter { !$0.isAgentSummary }
            .dropLast()
            .map { ["role": $0.role.rawValue, "content": $0.content] as [String: Any] }
            ?? []
        let messagesArray = Array(messages) + [["role": "user", "content": message] as [String: Any]]

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
                case .toolCalls, .done:
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
        let workDir = json["working_directory"] as? String ?? NSString(string: "~/.mlx-serve/workspace").expandingTildeInPath

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

        // Add user message
        let userMsg = ChatMessage(role: .user, content: message)
        appState.appendMessage(to: sessionId, message: userMsg)

        // Run the agent loop — same code path as ChatView.runAgentLoop
        let api = APIClient()
        let maxIterations = json["max_rounds"] as? Int ?? 10
        var padRetries = 0
        let maxPadRetries = 2
        var roundResults: [[String: Any]] = []

        let toolHandlers: [AgentToolKind: any ToolHandler] = [
            .shell: ShellHandler(),
            .readFile: ReadFileHandler(),
            .writeFile: WriteFileHandler(),
            .editFile: EditFileHandler(),
            .searchFiles: SearchFilesHandler(),
            .listFiles: ListFilesHandler(),
            .browse: BrowseHandler(),
            .webSearch: WebSearchHandler(),
            .saveMemory: SaveMemoryHandler(),
        ]

        for iteration in 0..<maxIterations {
            // Build history — same as ChatView.buildAgentHistory()
            var history = buildAgentHistory(appState: appState, sessionId: sessionId)
            let userContent = history.last { ($0["role"] as? String) == "user" }?["content"] as? String ?? ""
            let skills = AgentPrompt.skillManager.matchingSkills(for: userContent)
            let systemPrompt = AgentPrompt.systemPrompt + skills + AgentPrompt.memory + appState.agentMemory.contextSnippet()
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
            let stream = api.streamChat(
                port: appState.server.port,
                messages: messages,
                maxTokens: appState.maxTokens,
                temperature: 0.7,
                enableThinking: enableThinking && iteration == 0,
                tools: AgentPrompt.toolDefinitions
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
                    case .done:
                        break
                    }
                }
            } catch {
                appState.updateLastMessage(in: sessionId, content: "\n[Error: \(error.localizedDescription)]")
                break
            }
            appState.updateLastMessage(in: sessionId, streaming: false)

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
                // No tool calls, model gave a text response — done
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

            // Execute tools — same as ChatView
            var toolResults: [(id: String, name: String, output: String)] = []
            for tc in receivedToolCalls {
                let tool = AgentToolKind(rawValue: tc.name)
                let effectiveTool: AgentToolKind?
                if tool == .editFile && tc.arguments["content"] != nil && tc.arguments["find"] == nil {
                    effectiveTool = .writeFile
                } else {
                    effectiveTool = tool
                }

                // Pre-validate required params
                let missing = Self.missingRequiredParams(for: tc.name, arguments: tc.arguments)
                let output: String
                if !missing.isEmpty {
                    let example = Self.toolExample(for: tc.name)
                    output = "Error: \(tc.name) missing required params: \(missing.joined(separator: ", ")). Example: \(example)"
                } else if let effectiveTool, let handler = toolHandlers[effectiveTool] {
                    do {
                        output = try await handler.execute(parameters: tc.arguments, workingDirectory: workDir)
                        if effectiveTool == .shell, let cmd = tc.arguments["command"] {
                            appState.agentMemory.recordCommand(cmd)
                        }
                    } catch {
                        let argsDesc = tc.arguments.isEmpty ? "none" : tc.arguments.map { "\($0.key)=\($0.value.prefix(30))" }.joined(separator: ", ")
                        output = "Error: \(error.localizedDescription). You sent args: [\(argsDesc)]. Example: \(Self.toolExample(for: tc.name))"
                    }
                } else {
                    output = "Error: Unknown tool '\(tc.name)'"
                }
                toolResults.append((id: tc.id, name: tc.name, output: output))

                var resultMsg = ChatMessage(role: .assistant, content: "**\(tc.name)** → \(String(output.prefix(500)))")
                resultMsg.isAgentSummary = true
                appState.appendMessage(to: sessionId, message: resultMsg)
            }

            // Add tool results to history with overflow-to-disk truncation
            for tr in toolResults {
                var toolMsg = ChatMessage(role: .system, content: "")
                toolMsg.toolCallId = tr.id
                toolMsg.toolName = tr.name
                toolMsg.content = ChatDetailView.truncateWithOverflow(tr.output, toolCallId: tr.id, toolName: tr.name)
                appState.appendMessage(to: sessionId, message: toolMsg)
            }

            let callSummary = receivedToolCalls.map { "\($0.name)(\($0.arguments.keys.joined(separator: ",")))" }.joined(separator: ", ")
            roundResults.append(["round": iteration + 1, "type": "tool_calls", "tools": callSummary])
        }

        appState.saveChatHistory()

        // Return summary
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

    // MARK: - Helpers

    /// Replicates ChatView.buildAgentHistory() exactly
    private func buildAgentHistory(appState: AppState, sessionId: UUID) -> [[String: Any]] {
        guard let session = appState.chatSessions.first(where: { $0.id == sessionId }) else { return [] }
        let allMessages = session.messages

        let firstUserIdx = allMessages.firstIndex { $0.role == .user && $0.toolCallId == nil }
        let windowStart = max(0, allMessages.count - 28)
        let window = Array(allMessages.suffix(28))
        let needsPin = firstUserIdx != nil && firstUserIdx! < windowStart

        let totalToolResults = window.filter { $0.toolCallId != nil }.count
        var toolResultsSeen = 0

        var history: [[String: Any]] = []

        if needsPin, let idx = firstUserIdx {
            history.append(["role": "user", "content": allMessages[idx].content])
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
            if msg.role == .assistant && msg.content.contains("couldn't generate a response") { continue }

            if msg.role == .assistant, let tcs = msg.toolCalls, !tcs.isEmpty {
                var dict: [String: Any] = ["role": "assistant"]
                let content = msg.content.replacingOccurrences(of: "<pad>", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
                dict["content"] = content.isEmpty ? "" : content
                dict["tool_calls"] = tcs.map { tc -> [String: Any] in
                    ["id": tc.id, "type": "function", "function": ["name": tc.name, "arguments": tc.arguments] as [String: Any]]
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

    private static func missingRequiredParams(for toolName: String, arguments: [String: String]) -> [String] {
        for def in AgentPrompt.toolDefinitions {
            guard let fn = def["function"] as? [String: Any],
                  fn["name"] as? String == toolName,
                  let params = fn["parameters"] as? [String: Any],
                  let required = params["required"] as? [String] else { continue }
            return required.filter { key in
                guard let val = arguments[key] else { return true }
                return val.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }
        }
        return []
    }

    private static func toolExample(for toolName: String) -> String {
        for def in AgentPrompt.toolDefinitions {
            guard let fn = def["function"] as? [String: Any],
                  fn["name"] as? String == toolName,
                  let desc = fn["description"] as? String,
                  let range = desc.range(of: "Example: ") else { continue }
            return String(desc[range.upperBound...])
        }
        return "{}"
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
