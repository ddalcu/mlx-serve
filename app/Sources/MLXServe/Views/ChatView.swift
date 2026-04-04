import SwiftUI

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
                        Text(session.updatedAt, style: .relative)
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
                            RoundedRectangle(cornerRadius: 10)
                                .fill(Color.accentColor)
                        } else {
                            Color.clear
                        }
                    }
                )
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
    @FocusState private var inputFocused: Bool

    private var inputHeight: CGFloat {
        let lineCount = max(1, inputText.components(separatedBy: "\n").count)
        return min(300, CGFloat(lineCount) * 20 + 16)
    }

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
                }
                .onChange(of: session?.messages.count) { _, _ in
                    scrollToBottom(proxy)
                }
                .onChange(of: session?.messages.last?.content) { _, _ in
                    scrollToBottom(proxy)
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

                    // Dark pill — computed height from line count
                    ZStack(alignment: .topLeading) {
                        if inputText.isEmpty {
                            Text("Message")
                                .foregroundStyle(Color(.placeholderTextColor))
                                .padding(.leading, 16)
                                .padding(.top, 10)
                                .allowsHitTesting(false)
                        }
                        TextEditor(text: $inputText)
                            .font(.body)
                            .scrollContentBackground(.hidden)
                            .scrollIndicators(inputHeight >= 300 ? .automatic : .hidden)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .focused($inputFocused)
                            .disabled(server.status != .running || isGenerating)
                            .onKeyPress(.return, phases: .down) { press in
                                if press.modifiers.contains(.shift) {
                                    return .ignored // Shift+Return → newline
                                }
                                sendMessage()
                                return .handled // Return → send
                            }
                    }
                    .frame(height: inputHeight)
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
        }
        .onDisappear {
            generationTask?.cancel()
            generationTask = nil
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
                var errorMsg = ChatMessage(role: .assistant, content: "[Error: \(error.localizedDescription)]")
                errorMsg.isStreaming = false
                appState.appendMessage(to: sessionId, message: errorMsg)
            }
            appState.saveChatHistory()
            isGenerating = false
            generationTask = nil
        }
    }

    /// Agent loop: call model with tools (streaming), execute tool calls, feed results back, repeat.
    /// Stops when the model responds with content (no tool calls) or after 10 iterations.
    private func runAgentLoop(api: APIClient, workingDirectory: String?) async throws {
        let maxIterations = 10
        var padRetries = 0
        let maxPadRetries = 2

        for _ in 0..<maxIterations {
            try Task.checkCancellation()

            // Build message history for API
            let history = buildAgentHistory()
            let systemPrompt = AgentPrompt.systemPrompt + appState.agentMemory.contextSnippet()
            var messages: [[String: Any]] = [["role": "system", "content": systemPrompt]]
            messages.append(contentsOf: history)

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
                    if padRetries <= maxPadRetries {
                        continue // retry
                    }
                    // Give up — show error
                    let errorMsg = ChatMessage(role: .assistant, content: "The model couldn't generate a response. Try rephrasing or starting a new chat.")
                    appState.appendMessage(to: sessionId, message: errorMsg)
                    return
                }
            }

            // If no tool calls, we're done
            guard !receivedToolCalls.isEmpty else { return }

            // Show tool call summary
            let callSummary = receivedToolCalls.map { tc in
                let args = tc.arguments.map { "\($0.key): \($0.value.prefix(80))" }.joined(separator: ", ")
                return "**\(tc.name)**(\(args))"
            }.joined(separator: "\n")
            appState.updateLastMessage(in: sessionId, content: "\n\n" + callSummary)

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

                let output: String
                if let effectiveTool, let handler = toolHandlers[effectiveTool] {
                    do {
                        output = try await handler.execute(parameters: tc.arguments, workingDirectory: workingDirectory)
                        // Record shell commands
                        if effectiveTool == .shell, let cmd = tc.arguments["command"] {
                            appState.agentMemory.recordCommand(cmd)
                        }
                    } catch {
                        output = "Error: \(error.localizedDescription)"
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
                toolMsg.content = String(tr.output.prefix(2000))
                appState.appendMessage(to: sessionId, message: toolMsg)
            }
        }

        // Max iterations reached
        let msg = ChatMessage(role: .assistant, content: "(Agent stopped after \(maxIterations) tool call rounds)")
        appState.appendMessage(to: sessionId, message: msg)
    }

    private func buildAgentHistory() -> [[String: Any]] {
        (session?.messages ?? [])
            .suffix(30)
            .compactMap { msg -> [String: Any]? in
                // Tool response messages
                if let callId = msg.toolCallId {
                    return [
                        "role": "tool",
                        "tool_call_id": callId,
                        "content": String(msg.content.prefix(2000))
                    ]
                }
                if msg.role == .system { return nil }
                if msg.isAgentSummary { return nil } // display-only tool result messages
                if msg.role == .assistant && msg.content.isEmpty { return nil }
                var content = msg.content
                    .replacingOccurrences(of: "<pad>", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if msg.role == .assistant && content.count > 500 {
                    content = String(content.prefix(500)) + "..."
                }
                if content.isEmpty { return nil }
                return ["role": msg.role.rawValue, "content": content]
            }
    }

    private var toolHandlers: [AgentToolKind: any ToolHandler] {
        [
            .shell: ShellHandler(),
            .readFile: ReadFileHandler(),
            .writeFile: WriteFileHandler(),
            .editFile: EditFileHandler(),
            .searchFiles: SearchFilesHandler(),
            .browse: BrowseHandler(),
            .webSearch: WebSearchHandler(),
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
                            Label("Summary", systemImage: "text.document")
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
