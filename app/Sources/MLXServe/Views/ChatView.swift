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
                            VStack(spacing: 8) {
                                MessageBubble(message: message)

                                if let plan = message.agentPlan {
                                    let results = liveResults(for: message)

                                    PlanCardView(
                                        plan: plan,
                                        results: results,
                                        currentStepIndex: executingPlanMessageId == message.id ? toolExecutor.currentStepIndex : nil,
                                        onApprove: { approvePlan(messageId: message.id) },
                                        onReject: { rejectPlan(messageId: message.id) }
                                    )

                                    ForEach(Array(plan.steps.enumerated()), id: \.element.id) { index, step in
                                        if index < results.count {
                                            ToolResultBlockView(step: step, result: results[index])
                                        }
                                    }
                                }
                            }
                            .id(message.id)
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
                        sendMessage()
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
                    .help(server.status == .running ? "Server running" : "Server stopped")
            }
        }
        .onAppear {
            inputFocused = true
            isAgentMode = session?.mode == .agent
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

    private func liveResults(for message: ChatMessage) -> [StepResult] {
        if executingPlanMessageId == message.id {
            return toolExecutor.results
        }
        return message.toolResults ?? []
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

        Task {
            do {
                let stream = api.streamChat(
                    port: server.port,
                    messages: messagesArray,
                    enableThinking: enableThinking
                )
                for try await event in stream {
                    switch event {
                    case .content(let text):
                        appState.updateLastMessage(in: sessionId, content: text)
                    case .reasoning(let text):
                        appState.updateLastMessage(in: sessionId, reasoning: text)
                    case .done:
                        break
                    }
                }
            } catch {
                appState.updateLastMessage(in: sessionId, content: "\n\n[Error: \(error.localizedDescription)]")
            }
            appState.updateLastMessage(in: sessionId, streaming: false)
            appState.saveChatHistory()
            isGenerating = false
        }
    }

    // MARK: - Agent Mode

    private func sendAgentMessage() {
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

        // Build conversation history (exclude the empty assistant we just added)
        let history: [[String: Any]] = (session?.messages ?? [])
            .dropLast()
            .compactMap { msg -> [String: Any]? in
                if msg.role == .assistant && msg.content.isEmpty { return nil }
                return ["role": msg.role.rawValue, "content": msg.content]
            }

        let systemPrompt = AgentPrompt.systemPrompt + appState.agentMemory.contextSnippet()

        Task {
            var fullContent = ""
            do {
                let stream = api.streamAgentChat(
                    port: server.port,
                    messages: history,
                    systemPrompt: systemPrompt
                )
                for try await event in stream {
                    switch event {
                    case .content(let chunk):
                        fullContent += chunk
                        appState.updateLastMessage(in: sessionId, content: chunk)
                    case .reasoning(let chunk):
                        appState.updateLastMessage(in: sessionId, reasoning: chunk)
                    case .done:
                        break
                    }
                }
            } catch {
                appState.updateLastMessage(in: sessionId, content: "\n\n[Error: \(error.localizedDescription)]")
            }

            // Check for plan in response
            if let plan = AgentPrompt.parsePlan(from: fullContent) {
                let cleanContent = AgentPrompt.stripPlanTag(from: fullContent)
                if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }) {
                    let mIdx = appState.chatSessions[sIdx].messages.count - 1
                    appState.chatSessions[sIdx].messages[mIdx].content = cleanContent
                    appState.chatSessions[sIdx].messages[mIdx].agentPlan = plan
                }
            }

            appState.updateLastMessage(in: sessionId, streaming: false)
            appState.saveChatHistory()
            isGenerating = false
        }
    }

    private func approvePlan(messageId: UUID) {
        guard !isGenerating else { return }
        guard let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
              let mIdx = appState.chatSessions[sIdx].messages.firstIndex(where: { $0.id == messageId }),
              let plan = appState.chatSessions[sIdx].messages[mIdx].agentPlan,
              plan.status == .pending else { return }

        appState.chatSessions[sIdx].messages[mIdx].agentPlan?.status = .executing
        executingPlanMessageId = messageId
        isGenerating = true

        Task {
            let results = await toolExecutor.executePlan(plan, workingDirectory: session?.workingDirectory)

            // Save results to the message
            if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
               let mIdx = appState.chatSessions[sIdx].messages.firstIndex(where: { $0.id == messageId }) {
                appState.chatSessions[sIdx].messages[mIdx].toolResults = results
                let allOk = results.allSatisfy { $0.status == .success }
                appState.chatSessions[sIdx].messages[mIdx].agentPlan?.status = allOk ? .completed : .failed
            }

            executingPlanMessageId = nil

            // Record successful shell commands in memory
            for (step, result) in zip(plan.steps, results) where step.tool == .shell && result.status == .success {
                if let cmd = step.parameters["command"] {
                    appState.agentMemory.recordCommand(cmd)
                }
            }

            // Send results to model for summary
            await sendResultsSummary(results: results, plan: plan)
        }
    }

    private func rejectPlan(messageId: UUID) {
        guard let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
              let mIdx = appState.chatSessions[sIdx].messages.firstIndex(where: { $0.id == messageId }) else { return }
        appState.chatSessions[sIdx].messages[mIdx].agentPlan?.status = .rejected
        appState.saveChatHistory()
    }

    private func sendResultsSummary(results: [StepResult], plan: AgentPlan) async {
        let formatted = AgentPrompt.formatResults(results, plan: plan)

        var summaryMsg = ChatMessage(role: .assistant, content: "")
        summaryMsg.isStreaming = true
        summaryMsg.isAgentSummary = true
        appState.appendMessage(to: sessionId, message: summaryMsg)

        let api = APIClient()
        let history: [[String: Any]] = (session?.messages ?? [])
            .dropLast() // Exclude the empty summary message
            .compactMap { msg -> [String: Any]? in
                if msg.role == .assistant && msg.content.isEmpty { return nil }
                return ["role": msg.role.rawValue, "content": msg.content]
            }
        let allMessages = history + [
            ["role": "user", "content": "Tool execution results:\n\n\(formatted)\n\nBriefly summarize what was accomplished."] as [String: Any]
        ]

        let systemPrompt = AgentPrompt.systemPrompt + appState.agentMemory.contextSnippet()

        do {
            let stream = api.streamAgentChat(port: server.port, messages: allMessages, systemPrompt: systemPrompt)
            for try await event in stream {
                switch event {
                case .content(let text):
                    appState.updateLastMessage(in: sessionId, content: text)
                case .reasoning(let text):
                    appState.updateLastMessage(in: sessionId, reasoning: text)
                case .done:
                    break
                }
            }
        } catch {
            appState.updateLastMessage(in: sessionId, content: "\n\n[Error: \(error.localizedDescription)]")
        }

        appState.updateLastMessage(in: sessionId, streaming: false)
        appState.saveChatHistory()
        isGenerating = false
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
        if let attributed = try? AttributedString(markdown: source, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)) {
            Text(attributed)
                .font(.body)
        } else {
            Text(source)
                .font(.body)
        }
    }
}
