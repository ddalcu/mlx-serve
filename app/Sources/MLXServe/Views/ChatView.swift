import SwiftUI

private struct ContentBottomKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}

private struct ScrollViewHeightKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}

struct ChatView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @State private var columnVisibility = NavigationSplitViewVisibility.automatic

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            ChatSidebar()
                .navigationSplitViewColumnWidth(min: 180, ideal: 220, max: 280)
        } detail: {
            if let sessionId = appState.activeChatId,
               appState.chatSessions.contains(where: { $0.id == sessionId }) {
                ChatDetailView(sessionId: sessionId)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "bubble.left.and.bubble.right")
                        .font(.system(size: 36))
                        .foregroundStyle(.quaternary)
                    Text("Start a conversation")
                        .foregroundStyle(.secondary)
                    Button("New Chat") {
                        _ = appState.newChatSession()
                    }
                    .buttonStyle(.bordered)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .navigationTitle("")
        .onAppear {
            // Menu bar apps need explicit activation for keyboard focus
            NSApp.setActivationPolicy(.regular)
            DispatchQueue.main.async {
                NSApp.activate(ignoringOtherApps: true)
            }
        }
    }
}

// MARK: - Sidebar

struct ChatSidebar: View {
    @EnvironmentObject var appState: AppState
    @State private var hoveredSessionId: UUID?

    var body: some View {
        List(selection: $appState.activeChatId) {
            ForEach(appState.chatSessions) { session in
                let isSelected = session.id == appState.activeChatId
                HStack(spacing: 0) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(session.title)
                            .font(.subheadline.weight(isSelected ? .semibold : .regular))
                            .lineLimit(1)
                            .foregroundStyle(isSelected ? .white : .primary)
                        Text(relativeTime(session.updatedAt))
                            .font(.caption2)
                            .foregroundStyle(isSelected ? Color.white.opacity(0.7) : Color.secondary.opacity(0.5))
                    }
                    Spacer(minLength: 4)
                    if hoveredSessionId == session.id {
                        Button {
                            appState.deleteSession(session.id)
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 14))
                                .symbolRenderingMode(.hierarchical)
                                .foregroundStyle(isSelected ? .white.opacity(0.8) : .secondary)
                        }
                        .buttonStyle(.plain)
                        .help("Delete chat")
                    }
                }
                .tag(session.id)
                .onHover { isHovered in
                    hoveredSessionId = isHovered ? session.id : nil
                }
                .listRowBackground(
                    Group {
                        if isSelected {
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.accentColor)
                                .padding(.horizontal, 6)
                        } else {
                            Color.clear
                        }
                    }
                )
                .listRowSeparator(.visible)
                .contextMenu {
                    Button("Delete", role: .destructive) {
                        appState.deleteSession(session.id)
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .safeAreaInset(edge: .bottom) {
            Button {
                _ = appState.newChatSession()
            } label: {
                Label("New Chat", systemImage: "plus")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .padding(10)
        }
    }

    private func relativeTime(_ date: Date) -> String {
        let seconds = Int(-date.timeIntervalSinceNow)
        if seconds < 60 { return "just now" }
        let minutes = seconds / 60
        if minutes < 60 { return "\(minutes)m ago" }
        let hours = minutes / 60
        if hours < 24 { return "\(hours)h ago" }
        let days = hours / 24
        return "\(days)d ago"
    }
}

// MARK: - Chat Detail

struct ChatDetailView: View {
    let sessionId: UUID
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var toolExecutor: ToolExecutor
    @State private var inputText = ""
    @State private var isGenerating = false
    @State private var enableThinking = false
    @State private var isAgentMode = false
    @State private var executingPlanMessageId: UUID?
    @State private var generationTask: Task<Void, Never>?
    @State private var isNearBottom = true
    @State private var scrollViewHeight: CGFloat = 0
    @State private var scrollMonitor: Any?
    @FocusState private var inputFocused: Bool


    private var session: ChatSession? {
        appState.chatSessions.first { $0.id == sessionId }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(session?.messages ?? []) { message in
                            // Hide tool response messages (role: system with toolCallId)
                            if message.toolCallId == nil {
                                MessageBubble(message: message)
                                    .id(message.id)
                            }
                        }
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .padding(16)
                    .background(
                        GeometryReader { content in
                            Color.clear.preference(
                                key: ContentBottomKey.self,
                                value: content.frame(in: .named("chatScroll")).maxY
                            )
                        }
                    )
                }
                .coordinateSpace(name: "chatScroll")
                .background(
                    GeometryReader { scrollFrame in
                        Color.clear.preference(
                            key: ScrollViewHeightKey.self,
                            value: scrollFrame.size.height
                        )
                    }
                )
                .onPreferenceChange(ScrollViewHeightKey.self) { scrollViewHeight = $0 }
                .onPreferenceChange(ContentBottomKey.self) { contentBottom in
                    guard scrollViewHeight > 0 else { return }
                    let overshoot = contentBottom - scrollViewHeight
                    // Re-engage auto-scroll only when user scrolls back to bottom
                    if overshoot < 60 { isNearBottom = true }
                }
                .onChange(of: session?.messages.count) { _, _ in
                    if isNearBottom { scrollToBottom(proxy) }
                }
                .onChange(of: session?.messages.last?.content) { _, _ in
                    if isNearBottom { scrollToBottom(proxy) }
                }
            }

            Divider()

            // Input area — iMessage style
            VStack(spacing: 4) {
                if isAgentMode {
                    HStack(spacing: 4) {
                        Circle().fill(.orange).frame(width: 6, height: 6)
                        Text("Agent Mode")
                            .font(.caption2.weight(.medium))
                            .foregroundStyle(.orange)
                    }
                }

                HStack(alignment: .bottom, spacing: 8) {
                    Button { enableThinking.toggle() } label: {
                        Image(systemName: "brain")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(enableThinking ? .white : .secondary)
                            .frame(width: 28, height: 28)
                            .background(enableThinking ? .blue : Color.secondary.opacity(0.15))
                            .clipShape(Circle())
                    }
                    .buttonStyle(.plain)
                    .help("Thinking Mode — model reasons step-by-step before answering (\(enableThinking ? "ON" : "OFF"))")

                    Button { isAgentMode.toggle() } label: {
                        Image(systemName: "wrench")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(isAgentMode ? .white : .secondary)
                            .frame(width: 28, height: 28)
                            .background(isAgentMode ? .orange : Color.secondary.opacity(0.15))
                            .clipShape(Circle())
                    }
                    .buttonStyle(.plain)
                    .help("Agent Mode — plan & execute shell, files, web, AppleScript (\(isAgentMode ? "ON" : "OFF"))")

                    // Dark pill input
                    TextField("Message", text: $inputText, axis: .vertical)
                        .font(.body)
                        .textFieldStyle(.plain)
                        .lineLimit(1...15)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .focused($inputFocused)
                        .disabled(server.status != .running || isGenerating)
                        .onKeyPress(.return, phases: .down) { press in
                            if press.modifiers.contains(.shift) {
                                inputText += "\n"
                                return .handled
                            }
                            sendMessage()
                            return .handled
                        }
                        .onAppear {
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                                inputFocused = true
                            }
                        }
                        .background(Color(nsColor: NSColor(white: 0.10, alpha: 1)))
                        .clipShape(RoundedRectangle(cornerRadius: 20))
                        .overlay(
                            RoundedRectangle(cornerRadius: 20)
                                .stroke(Color.secondary.opacity(0.25), lineWidth: 0.5)
                        )

                    Button {
                        if isGenerating {
                            stopGenerating()
                        } else {
                            sendMessage()
                        }
                    } label: {
                        Image(systemName: isGenerating ? "stop.circle.fill" : "arrow.up.circle.fill")
                            .font(.system(size: 28))
                            .foregroundStyle(isGenerating ? .red : .accentColor)
                    }
                    .buttonStyle(.plain)
                    .disabled(server.status != .running || (inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isGenerating))
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
        .toolbar {
            ToolbarItem(placement: .automatic) {
                if isAgentMode {
                    WorkingDirectoryIndicator(path: workingDirectoryBinding)
                }
            }
            ToolbarItem(placement: .automatic) {
                if isAgentMode {
                    Button {
                        let path = NSString(string: "~/.mlx-serve").expandingTildeInPath
                        NSWorkspace.shared.open(URL(fileURLWithPath: path))
                    } label: {
                        Image(systemName: "folder.badge.gearshape")
                            .font(.system(size: 12))
                    }
                    .help("Agent Skills Folder")
                }
            }
            ToolbarItem(placement: .automatic) {
                if isAgentMode && server.status == .running {
                    Button {
                        launchClaudeCodeWithPicker()
                    } label: {
                        ClaudeIcon(size: 12)
                    }
                    .help("Launch Claude Code")
                }
            }
            ToolbarItem(placement: .automatic) {
                Circle()
                    .fill(server.status == .running ? .green : .red)
                    .frame(width: 8, height: 8)
                    .padding(.horizontal, 6)
                    .help(server.status == .running ? "Server running" : "Server stopped")
            }
        }
        .onAppear {
            inputFocused = true
            isAgentMode = session?.mode == .agent
            scrollMonitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { event in
                // Only disengage on upward scroll (viewing earlier messages).
                // Scrolling down toward the bottom lets the preference handler re-engage.
                if event.scrollingDeltaY > 0 {
                    isNearBottom = false
                }
                return event
            }
        }
        .onDisappear {
            generationTask?.cancel()
            generationTask = nil
            if let monitor = scrollMonitor {
                NSEvent.removeMonitor(monitor)
                scrollMonitor = nil
            }
        }
        .onChange(of: isAgentMode) { _, newValue in
            if let idx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }) {
                appState.chatSessions[idx].mode = newValue ? .agent : .chat
            }
        }
        .onChange(of: isGenerating) { _, generating in
            if !generating { inputFocused = true }
        }
    }

    // MARK: - Helpers

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        withAnimation(.easeOut(duration: 0.15)) {
            proxy.scrollTo("bottom", anchor: .bottom)
        }
    }


    private var workingDirectoryBinding: Binding<String?> {
        Binding(
            get: { appState.chatSessions.first { $0.id == sessionId }?.workingDirectory },
            set: { newValue in
                if let idx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }) {
                    appState.chatSessions[idx].workingDirectory = newValue
                    if let dir = newValue {
                        appState.agentMemory.recordDirectory(dir)
                    }
                }
            }
        )
    }

    // MARK: - Stop Generation

    private func stopGenerating() {
        generationTask?.cancel()
        generationTask = nil
        appState.updateLastMessage(in: sessionId, streaming: false)
        appState.saveChatHistory()
        isGenerating = false
    }

    // MARK: - Send Message

    private func sendMessage() {
        isNearBottom = true // snap to bottom on send
        if isAgentMode {
            sendAgentMessage()
            return
        }

        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isGenerating, server.status == .running else { return }
        inputText = ""

        let userMsg = ChatMessage(role: .user, content: text)
        appState.appendMessage(to: sessionId, message: userMsg)

        var assistantMsg = ChatMessage(role: .assistant, content: "")
        assistantMsg.isStreaming = true
        appState.appendMessage(to: sessionId, message: assistantMsg)

        isGenerating = true
        let api = APIClient()

        let messages = (session?.messages ?? []).map { msg -> [String: Any] in
            var dict: [String: Any] = ["role": msg.role.rawValue, "content": msg.content]
            if msg.role == .assistant && msg.content.isEmpty { dict.removeValue(forKey: "content") }
            return dict
        }.dropLast() // Drop the empty assistant message we just added
        let messagesArray = Array(messages) + [["role": "user", "content": text] as [String: Any]]

        generationTask = Task {
            do {
                let stream = api.streamChat(
                    port: server.port,
                    messages: messagesArray,
                    maxTokens: appState.maxTokens,
                    enableThinking: enableThinking
                )
                for try await event in stream {
                    try Task.checkCancellation()
                    switch event {
                    case .content(let text):
                        appState.updateLastMessage(in: sessionId, content: text)
                    case .reasoning(let text):
                        appState.updateLastMessage(in: sessionId, reasoning: text)
                    case .usage(let usage):
                        appState.updateLastMessage(in: sessionId, usage: usage)
                    case .toolCalls:
                        break
                    case .done:
                        break
                    }
                }
            } catch is CancellationError {
                // Stopped by user
            } catch {
                print("[ChatView] Chat error: \(error)")
                try? "Chat error: \(error)\n".write(toFile: NSString(string: "~/.mlx-serve/debug.log").expandingTildeInPath, atomically: true, encoding: .utf8)
                appState.updateLastMessage(in: sessionId, content: "\n\n[Error: \(error.localizedDescription)]")
            }
            appState.updateLastMessage(in: sessionId, streaming: false)
            appState.saveChatHistory()
            isGenerating = false
            generationTask = nil
        }
    }

    // MARK: - Agent Mode (Native Tool Calling)

    private func sendAgentMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isGenerating, server.status == .running else { return }
        inputText = ""

        let userMsg = ChatMessage(role: .user, content: text)
        appState.appendMessage(to: sessionId, message: userMsg)

        isGenerating = true
        let api = APIClient()
        let workDir = session?.workingDirectory

        generationTask = Task {
            do {
                try await runAgentLoop(api: api, workingDirectory: workDir)
            } catch is CancellationError {
                // Stopped by user
            } catch {
                print("[ChatView] Agent error: \(error)")
                try? "Agent error: \(error)\n".write(toFile: NSString(string: "~/.mlx-serve/debug.log").expandingTildeInPath, atomically: true, encoding: .utf8)
                var errorMsg = ChatMessage(role: .assistant, content: "[Error: \(error.localizedDescription)]")
                errorMsg.isStreaming = false
                appState.appendMessage(to: sessionId, message: errorMsg)
            }
            appState.saveChatHistory()
            isGenerating = false
            generationTask = nil
        }
    }

    /// Show folder picker and launch Claude Code in the selected directory.
    private func launchClaudeCodeWithPicker() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Open"
        panel.message = "Select working directory for Claude Code"
        // Default to session's working directory if set
        if let wd = session?.workingDirectory {
            panel.directoryURL = URL(fileURLWithPath: wd)
        }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        launchClaudeCode(baseURL: server.baseURL, workingDirectory: url.path)
    }

    /// Agent loop: call model with tools (streaming), execute tool calls, feed results back, repeat.
    /// Stops when the model responds with content (no tool calls) or after 150 iterations.
    private func runAgentLoop(api: APIClient, workingDirectory: String?) async throws {
        let maxIterations = 150
        var padRetries = 0
        let padRetryPolicy = RetryPolicy.aggressive

        for _ in 0..<maxIterations {
            try Task.checkCancellation()

            // Build message history for API
            var history = buildAgentHistory()
            let userMsg = history.last { ($0["role"] as? String) == "user" }?["content"] as? String ?? ""
            let skills = AgentPrompt.skillManager.matchingSkills(for: userMsg)
            var systemPrompt = AgentPrompt.systemPrompt + skills + AgentPrompt.memory + appState.agentMemory.contextSnippet()
            if let wd = workingDirectory {
                systemPrompt += "\n\n# Working Directory\nYour working directory is `\(wd)`. All relative paths resolve against it. Use relative paths for files inside this directory. Shell commands run here by default."
            }
            var messages: [[String: Any]] = [["role": "system", "content": systemPrompt]]
            // Some models (e.g. Gemma 4 E4B) can't generate after tool results without
            // a user message. Add a nudge so the model knows to synthesize a response.
            if let lastRole = history.last?["role"] as? String, lastRole == "tool" {
                history.append(["role": "user", "content": "Continue. If the task is done, summarize the result. If not, take the next step."])
            }
            messages.append(contentsOf: history)

            // Debug: dump the exact request body to file for analysis
            do {
                let debugBody: [String: Any] = [
                    "messages": messages,
                    "tools": AgentPrompt.toolDefinitions,
                    "max_tokens": appState.maxTokens,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "stream": true,
                    "model": "mlx-serve"
                ]
                let debugData = try JSONSerialization.data(withJSONObject: debugBody, options: .prettyPrinted)
                let debugPath = NSString(string: "~/.mlx-serve/last-agent-request.json").expandingTildeInPath
                try debugData.write(to: URL(fileURLWithPath: debugPath))
            } catch { /* ignore debug errors */ }

            // Add streaming assistant message
            var streamMsg = ChatMessage(role: .assistant, content: "")
            streamMsg.isStreaming = true
            appState.appendMessage(to: sessionId, message: streamMsg)

            // Stream model response with tools
            var receivedToolCalls: [APIClient.ToolCall] = []
            let stream = api.streamChat(
                port: server.port,
                messages: messages,
                maxTokens: appState.maxTokens,
                temperature: 0.7,
                enableThinking: enableThinking,
                tools: AgentPrompt.toolDefinitions
            )
            for try await event in stream {
                try Task.checkCancellation()
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
            appState.updateLastMessage(in: sessionId, streaming: false)

            // Check for pad-only or empty responses — retry limited times
            if receivedToolCalls.isEmpty {
                let lastContent = appState.chatSessions
                    .first(where: { $0.id == sessionId })?.messages.last?.content ?? ""
                let cleaned = lastContent
                    .replacingOccurrences(of: "<pad>", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if cleaned.isEmpty {
                    // Remove the empty/pad message
                    if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
                       !appState.chatSessions[sIdx].messages.isEmpty {
                        appState.chatSessions[sIdx].messages.removeLast()
                    }
                    padRetries += 1
                    if padRetries <= padRetryPolicy.maxRetries {
                        let delay = padRetryPolicy.delay(for: padRetries)
                        try? await Task.sleep(nanoseconds: delay)
                        continue // retry with backoff
                    }
                    // Give up — show error
                    let errorMsg = ChatMessage(role: .assistant, content: "The model couldn't generate a response. Try rephrasing or starting a new chat.")
                    appState.appendMessage(to: sessionId, message: errorMsg)
                    return
                }
            }

            // If no tool calls, we're done
            guard !receivedToolCalls.isEmpty else { return }

            // Store tool calls on the assistant message for history replay
            if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
               !appState.chatSessions[sIdx].messages.isEmpty {
                let mIdx = appState.chatSessions[sIdx].messages.count - 1
                appState.chatSessions[sIdx].messages[mIdx].toolCalls = receivedToolCalls.map { tc in
                    let argsJson = (try? JSONSerialization.data(withJSONObject: tc.arguments))
                        .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                    return SerializedToolCall(id: tc.id, name: tc.name, arguments: argsJson)
                }
            }

            // Show tool call summary as display-only message (not appended to assistant content)
            let callSummary = receivedToolCalls.map { tc in
                let args = tc.arguments.map { "\($0.key): \($0.value.prefix(80))" }.joined(separator: ", ")
                let display = args.isEmpty ? tc.rawArguments.prefix(200) : args[...]
                return "**\(tc.name)**(\(display))"
            }.joined(separator: "\n")
            var summaryMsg = ChatMessage(role: .assistant, content: callSummary)
            summaryMsg.isAgentSummary = true
            appState.appendMessage(to: sessionId, message: summaryMsg)

            // Execute each tool call
            var toolResults: [(id: String, name: String, output: String)] = []
            for tc in receivedToolCalls {
                try Task.checkCancellation()
                let tool = AgentToolKind(rawValue: tc.name)

                // Smart fallback: editFile with content but no find → writeFile
                let effectiveTool: AgentToolKind?
                if tool == .editFile && tc.arguments["content"] != nil && tc.arguments["find"] == nil {
                    effectiveTool = .writeFile
                } else {
                    effectiveTool = tool
                }

                // Pre-validate required params — give the model the expected format on failure
                let missing = Self.missingRequiredParams(for: tc.name, arguments: tc.arguments)
                let output: String
                if !missing.isEmpty {
                    let example = Self.toolExample(for: tc.name)
                    output = "Error: \(tc.name) missing required params: \(missing.joined(separator: ", ")). Example: \(example)"
                } else if let effectiveTool, let handler = toolHandlers[effectiveTool] {
                    do {
                        output = try await handler.execute(parameters: tc.arguments, workingDirectory: workingDirectory)
                        // Record shell commands
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

                // Show result in chat (display-only, marked so it's excluded from API history)
                var resultMsg = ChatMessage(role: .assistant, content: "**\(tc.name)** → \(String(output.prefix(500)))")
                resultMsg.isAgentSummary = true // reuse flag to mark as display-only
                appState.appendMessage(to: sessionId, message: resultMsg)
            }

            // Add tool results as tool role messages to the session
            for tr in toolResults {
                var toolMsg = ChatMessage(role: .system, content: "")
                toolMsg.toolCallId = tr.id
                toolMsg.toolName = tr.name
                toolMsg.content = Self.truncateWithOverflow(tr.output, toolCallId: tr.id, toolName: tr.name)
                appState.appendMessage(to: sessionId, message: toolMsg)
            }
        }

        // Max iterations reached
        let msg = ChatMessage(role: .assistant, content: "(Agent stopped after \(maxIterations) tool call rounds)")
        appState.appendMessage(to: sessionId, message: msg)
    }

    // MARK: - Tool Result Overflow

    /// Per-tool context caps (chars). Oversized results are saved to disk.
    static let toolResultCaps: [String: Int] = [
        "shell": 6000,
        "readFile": 8000,
        "searchFiles": 4000,
        "listFiles": 4000,
        "webSearch": 2000,
        "browse": 3000,
        "editFile": 2000,
        "writeFile": 2000,
        "saveMemory": 500,
    ]

    /// Truncate tool output, saving full result to disk if oversized.
    /// Returns the (possibly truncated) output for context inclusion.
    static func truncateWithOverflow(_ output: String, toolCallId: String, toolName: String) -> String {
        let maxChars = toolResultCaps[toolName] ?? 4000
        guard output.count > maxChars else { return output }

        let dir = NSString(string: "~/.mlx-serve/tool-output").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true, attributes: nil)
        let path = (dir as NSString).appendingPathComponent("\(toolCallId).txt")
        try? output.write(toFile: path, atomically: true, encoding: .utf8)

        let preview = String(output.prefix(maxChars - 200))
        return "\(preview)\n\n[... truncated at \(maxChars) chars. Full output (\(output.count) chars) saved to: \(path) — use readFile to see the rest]"
    }

    /// Clean up overflow files older than 24 hours. Called on session start.
    static func cleanupOverflowFiles() {
        let dir = NSString(string: "~/.mlx-serve/tool-output").expandingTildeInPath
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(atPath: dir) else { return }
        let cutoff = Date().addingTimeInterval(-86400)
        for file in files {
            let path = (dir as NSString).appendingPathComponent(file)
            if let attrs = try? fm.attributesOfItem(atPath: path),
               let modified = attrs[.modificationDate] as? Date,
               modified < cutoff {
                try? fm.removeItem(atPath: path)
            }
        }
    }

    // MARK: - Token-Aware Context Management

    /// Rough token estimation: ~4 bytes per token (no tokenizer needed).
    private func roughTokenCount(_ s: String) -> Int {
        max(1, s.utf8.count / 4)
    }

    /// Estimate token cost for a message including role/format overhead.
    private func tokenCostForMessage(_ msg: ChatMessage) -> Int {
        var cost = 4  // role + formatting envelope
        cost += roughTokenCount(msg.content)
        if let tcs = msg.toolCalls {
            for tc in tcs {
                cost += roughTokenCount(tc.name) + roughTokenCount(tc.arguments) + 8
            }
        }
        return cost
    }

    /// Determine effective context length from user config or model metadata.
    private var effectiveContextLength: Int {
        if appState.contextSize > 0 { return appState.contextSize }
        if let modelCtx = server.modelInfo?.contextLength, modelCtx > 0 { return modelCtx }
        return 32768  // safe default
    }

    private func buildAgentHistory() -> [[String: Any]] {
        let allMessages = session?.messages ?? []

        // Compute token budget for history
        let contextLength = effectiveContextLength
        let safetyBuffer = 1024
        // System prompt is assembled in runAgentLoop; estimate its cost here
        let systemPromptCost = roughTokenCount(AgentPrompt.systemPrompt + AgentPrompt.memory)
        let budget = max(1024, contextLength - appState.maxTokens - safetyBuffer - systemPromptCost)

        // Always pin the first user message (the original task)
        let firstUserIdx = allMessages.firstIndex { $0.role == .user && $0.toolCallId == nil }
        var pinnedCost = 0
        if let idx = firstUserIdx {
            pinnedCost = roughTokenCount(allMessages[idx].content) + 4
        }

        // Walk backward from newest message, accumulating token costs
        var remainingBudget = budget - pinnedCost
        var includeStartIdx = allMessages.count  // exclusive start (will walk backward)

        for i in stride(from: allMessages.count - 1, through: 0, by: -1) {
            let msg = allMessages[i]
            // Skip messages that won't be included in history
            if msg.role == .system && msg.toolCallId == nil { continue }
            if msg.isAgentSummary { continue }
            if msg.role == .assistant && msg.content.contains("couldn't generate a response") { continue }

            let cost = tokenCostForMessage(msg)
            if cost > remainingBudget { break }
            remainingBudget -= cost
            includeStartIdx = i
        }

        // Determine if first user message needs pinning (fell outside included range)
        let needsPin = firstUserIdx != nil && firstUserIdx! < includeStartIdx

        // Count tool results in included range for progressive truncation
        let window = Array(allMessages[includeStartIdx..<allMessages.count])
        let totalToolResults = window.filter { $0.toolCallId != nil }.count
        var toolResultsSeen = 0

        var history: [[String: Any]] = []

        // Pin the first user message if it fell outside the window
        if needsPin, let idx = firstUserIdx {
            history.append(["role": "user", "content": allMessages[idx].content])
        }

        for msg in window {
            // Tool response messages — truncate older results more aggressively
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

            // Assistant messages with tool_calls: include tool_calls in OpenAI format
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

    /// Check which required params are missing for a tool call.
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

    /// Get the example JSON format from a tool's description.
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

    private var toolHandlers: [AgentToolKind: any ToolHandler] {
        [
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
    }
}

// MARK: - Message Bubble

struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            if message.role == .user { Spacer(minLength: 60) }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                // Reasoning (collapsible)
                if let reasoning = message.reasoningContent, !reasoning.isEmpty {
                    DisclosureGroup {
                        Text(reasoning)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    } label: {
                        Label("Thinking", systemImage: "brain")
                            .font(.caption2.weight(.medium))
                            .foregroundStyle(.secondary)
                    }
                    .padding(8)
                    .background(.quaternary.opacity(0.5))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }

                // Content
                if !message.content.isEmpty || message.isStreaming {
                    VStack(alignment: .leading, spacing: 4) {
                        if message.isAgentSummary {
                            Label("Tool Call", systemImage: "wrench.and.screwdriver")
                                .font(.caption2.weight(.medium))
                                .foregroundStyle(.secondary)
                        }
                        if message.role == .assistant {
                            MarkdownText(message.content.isEmpty && message.isStreaming ? " " : message.content)
                                .textSelection(.enabled)
                        } else {
                            Text(message.content)
                                .textSelection(.enabled)
                        }
                        if message.isStreaming {
                            ProgressView()
                                .controlSize(.mini)
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(message.role == .user ? Color.accentColor : Color(.controlBackgroundColor))
                    .foregroundStyle(message.role == .user ? .white : .primary)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                }

                // Token usage stats
                if message.role == .assistant, !message.isStreaming,
                   let prompt = message.promptTokens, let completion = message.completionTokens {
                    HStack(spacing: 8) {
                        Text("\(prompt)+\(completion) tokens")
                        if let tps = message.tokensPerSecond, tps > 0 {
                            Text("~\(Int(tps)) tok/s")
                        }
                    }
                    .font(.caption2.monospaced())
                    .foregroundStyle(.tertiary)
                }
            }

            if message.role == .assistant { Spacer(minLength: 60) }
        }
    }
}

// MARK: - Markdown Rendering

struct MarkdownText: View {
    let source: String

    init(_ source: String) {
        self.source = source
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(parseBlocks().enumerated()), id: \.offset) { _, block in
                blockView(block)
            }
        }
    }

    private enum Block {
        case paragraph(String)
        case heading(Int, String)      // level, text
        case code(String, String)      // language, content
        case listItem(String)
        case xmlBlock(String)          // raw XML/tag content
    }

    private func parseBlocks() -> [Block] {
        var blocks: [Block] = []
        let lines = source.components(separatedBy: "\n")
        var i = 0

        while i < lines.count {
            let line = lines[i]

            // XML-like tag block (<plan>...</plan>, <pad>, etc.)
            if let match = line.range(of: "^<([a-zA-Z_]+)>", options: .regularExpression) {
                let tag = String(line[match]).dropFirst().dropLast() // extract tag name
                let closeTag = "</\(tag)>"
                if line.contains(closeTag) {
                    // Single-line tag block
                    blocks.append(.xmlBlock(line))
                    i += 1
                    continue
                }
                // Multi-line: collect until closing tag
                var xmlLines: [String] = [line]
                i += 1
                while i < lines.count {
                    xmlLines.append(lines[i])
                    if lines[i].contains(closeTag) {
                        i += 1
                        break
                    }
                    i += 1
                }
                blocks.append(.xmlBlock(xmlLines.joined(separator: "\n")))
                continue
            }

            // Standalone tags like <pad><pad><pad>
            if line.hasPrefix("<") && line.contains(">") && !line.hasPrefix("<http") {
                let stripped = line.trimmingCharacters(in: .whitespaces)
                if stripped.range(of: "^(<[a-zA-Z_/]+>\\s*)+$", options: .regularExpression) != nil {
                    blocks.append(.xmlBlock(stripped))
                    i += 1
                    continue
                }
            }

            // Fenced code block
            if line.hasPrefix("```") {
                let lang = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                var code: [String] = []
                i += 1
                while i < lines.count && !lines[i].hasPrefix("```") {
                    code.append(lines[i])
                    i += 1
                }
                i += 1 // skip closing ```
                blocks.append(.code(lang, code.joined(separator: "\n")))
                continue
            }

            // Heading
            if line.hasPrefix("#") {
                let level = line.prefix(while: { $0 == "#" }).count
                if level <= 6 {
                    let text = String(line.dropFirst(level)).trimmingCharacters(in: .whitespaces)
                    if !text.isEmpty {
                        blocks.append(.heading(level, text))
                        i += 1
                        continue
                    }
                }
            }

            // List item
            if line.starts(with: "- ") || line.starts(with: "* ") ||
               (line.count >= 3 && line.first?.isNumber == true && line.contains(". ")) {
                let text: String
                if line.starts(with: "- ") || line.starts(with: "* ") {
                    text = String(line.dropFirst(2))
                } else if let dotIdx = line.firstIndex(of: "."), line[line.index(after: dotIdx)] == " " {
                    text = String(line[line.index(dotIdx, offsetBy: 2)...])
                } else {
                    text = line
                }
                blocks.append(.listItem(text))
                i += 1
                continue
            }

            // Empty line — skip
            if line.trimmingCharacters(in: .whitespaces).isEmpty {
                i += 1
                continue
            }

            // Paragraph — collect consecutive non-empty lines
            var para: [String] = [line]
            i += 1
            while i < lines.count {
                let next = lines[i]
                if next.trimmingCharacters(in: .whitespaces).isEmpty ||
                   next.hasPrefix("#") || next.hasPrefix("```") ||
                   next.starts(with: "- ") || next.starts(with: "* ") ||
                   next.hasPrefix("<") {
                    break
                }
                para.append(next)
                i += 1
            }
            blocks.append(.paragraph(para.joined(separator: "\n")))
        }

        return blocks
    }

    @ViewBuilder
    private func blockView(_ block: Block) -> some View {
        switch block {
        case .paragraph(let text):
            inlineMarkdown(text)

        case .heading(let level, let text):
            Text(text)
                .font(level == 1 ? .title2.bold() : level == 2 ? .title3.bold() : .headline)
                .padding(.top, level == 1 ? 4 : 2)

        case .code(_, let content):
            ScrollView(.horizontal, showsIndicators: false) {
                Text(content)
                    .font(.system(size: 12, design: .monospaced))
                    .textSelection(.enabled)
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(nsColor: NSColor(white: 0.08, alpha: 1)))
            .clipShape(RoundedRectangle(cornerRadius: 8))

        case .listItem(let text):
            HStack(alignment: .top, spacing: 6) {
                Text("\u{2022}")
                    .foregroundStyle(.secondary)
                inlineMarkdown(text)
            }

        case .xmlBlock(let content):
            ScrollView(.horizontal, showsIndicators: false) {
                Text(content)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.purple.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.purple.opacity(0.3), lineWidth: 0.5)
            )
        }
    }

    private func inlineMarkdown(_ text: String) -> Text {
        if let attributed = try? AttributedString(markdown: text, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)) {
            return Text(attributed)
        }
        return Text(text)
    }
}
