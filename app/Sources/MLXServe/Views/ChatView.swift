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
    @State private var pendingImages: [NSImage] = []
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

            // Context usage monitor
            if let usage = contextUsage, usage.promptTokens > 0 {
                ContextMonitor(promptTokens: usage.promptTokens, contextLength: usage.contextLength, maxTokens: appState.maxTokens)
            }

            // Input area — iMessage style
            VStack(spacing: 4) {
                // Pending image thumbnails
                if !pendingImages.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 6) {
                            ForEach(Array(pendingImages.enumerated()), id: \.offset) { idx, img in
                                ZStack(alignment: .topTrailing) {
                                    Image(nsImage: img)
                                        .resizable()
                                        .aspectRatio(contentMode: .fill)
                                        .frame(width: 56, height: 56)
                                        .clipShape(RoundedRectangle(cornerRadius: 8))
                                    Button {
                                        pendingImages.remove(at: idx)
                                    } label: {
                                        Image(systemName: "xmark.circle.fill")
                                            .font(.system(size: 14))
                                            .foregroundStyle(.white)
                                            .background(Circle().fill(.black.opacity(0.5)))
                                    }
                                    .buttonStyle(.plain)
                                    .offset(x: 4, y: -4)
                                }
                            }
                        }
                        .padding(.horizontal, 4)
                    }
                    .frame(height: 64)
                }

                HStack(alignment: .bottom, spacing: 8) {
                    // Image attachment button
                    Button { pickImage() } label: {
                        Image(systemName: "paperclip")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(.secondary)
                            .frame(width: 28, height: 28)
                            .background(Color.secondary.opacity(0.15))
                            .clipShape(Circle())
                    }
                    .buttonStyle(.plain)
                    .help("Attach image")

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
                    .disabled(server.status != .running || (inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && pendingImages.isEmpty && !isGenerating))
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
        .onDrop(of: [.image], isTargeted: nil) { providers in
            for provider in providers {
                provider.loadObject(ofClass: NSImage.self) { image, _ in
                    if let image = image as? NSImage {
                        DispatchQueue.main.async { pendingImages.append(image) }
                    }
                }
            }
            return true
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
                Button { enableThinking.toggle() } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "brain")
                            .font(.system(size: 11, weight: .medium))
                        Text("Think")
                            .font(.caption.weight(.medium))
                    }
                    .foregroundStyle(enableThinking ? .white : .secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(enableThinking ? .blue : Color.secondary.opacity(0.12))
                    .clipShape(Capsule())
                }
                .buttonStyle(.plain)
                .help("Thinking Mode (\(enableThinking ? "ON" : "OFF"))")
                .padding(.leading, 8)
                .padding(.trailing, 4)
            }
            ToolbarItem(placement: .automatic) {
                Button { isAgentMode.toggle() } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "wrench")
                            .font(.system(size: 11, weight: .medium))
                        Text("Agent")
                            .font(.caption.weight(.medium))
                    }
                    .foregroundStyle(isAgentMode ? .white : .secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(isAgentMode ? .orange : Color.secondary.opacity(0.12))
                    .clipShape(Capsule())
                }
                .buttonStyle(.plain)
                .help("Agent Mode (\(isAgentMode ? "ON" : "OFF"))")
            }
            ToolbarItem(placement: .automatic) {
                Circle()
                    .fill(server.status == .running ? .green : .red)
                    .frame(width: 8, height: 8)
                    .padding(.horizontal, 8)
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

    // MARK: - Image Helpers

    private func pickImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.begin { response in
            guard response == .OK else { return }
            for url in panel.urls {
                if let image = NSImage(contentsOf: url) {
                    pendingImages.append(image)
                }
            }
        }
    }

    /// Convert NSImage to JPEG data suitable for API transport.
    private static func nsImageToJPEG(_ image: NSImage) -> Data? {
        guard let tiff = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiff),
              let jpeg = bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.85]) else {
            return nil
        }
        return jpeg
    }

    /// Convert pending NSImages to ChatImage array, clearing the pending list.
    private func consumePendingImages() -> [ChatImage]? {
        guard !pendingImages.isEmpty else { return nil }
        let chatImages = pendingImages.compactMap { img -> ChatImage? in
            guard let data = Self.nsImageToJPEG(img) else { return nil }
            return ChatImage(data: data)
        }
        pendingImages = []
        return chatImages.isEmpty ? nil : chatImages
    }

    /// Build OpenAI-style content blocks for a message with images.
    /// Images are preprocessed to raw float32 pixel data for the vision encoder.
    private static func buildMultimodalContent(text: String, images: [ChatImage]) -> Any {
        var blocks: [[String: Any]] = images.compactMap { img in
            // Preprocess image for vision encoder (768x768 float32 CHW)
            if let pixelData = ImagePreprocessor.preprocess(img.data) {
                return [
                    "type": "image_url",
                    "image_url": [
                        "url": "data:image/x-mlx-pixels;base64,\(pixelData.base64EncodedString())"
                    ] as [String: Any]
                ]
            }
            // Fallback: send JPEG if preprocessing fails
            return [
                "type": "image_url",
                "image_url": ["url": img.base64URL] as [String: Any]
            ]
        }
        if !text.isEmpty {
            blocks.append(["type": "text", "text": text])
        }
        return blocks
    }

    // MARK: - Helpers

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        withAnimation(.easeOut(duration: 0.15)) {
            proxy.scrollTo("bottom", anchor: .bottom)
        }
    }


    /// Latest context usage from the most recent assistant message with token data.
    private var contextUsage: (promptTokens: Int, contextLength: Int)? {
        guard let messages = session?.messages else { return nil }
        // Find last message with prompt token data
        if let last = messages.last(where: { $0.promptTokens != nil && $0.promptTokens! > 0 }) {
            let ctxLen: Int
            if appState.contextSize > 0 {
                ctxLen = appState.contextSize
            } else if let modelCtx = server.modelInfo?.contextLength, modelCtx > 0 {
                ctxLen = modelCtx
            } else {
                ctxLen = 32768
            }
            return (promptTokens: last.promptTokens!, contextLength: ctxLen)
        }
        return nil
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
        let attachedImages = consumePendingImages()
        guard !text.isEmpty || attachedImages != nil, !isGenerating, server.status == .running else { return }
        inputText = ""

        var userMsg = ChatMessage(role: .user, content: text)
        userMsg.images = attachedImages
        appState.appendMessage(to: sessionId, message: userMsg)

        var assistantMsg = ChatMessage(role: .assistant, content: "")
        assistantMsg.isStreaming = true
        appState.appendMessage(to: sessionId, message: assistantMsg)

        isGenerating = true
        let api = APIClient()

        // Strip images from old messages — server only processes the last user message's images.
        // Re-sending old images wastes bandwidth and memory.
        let messages = (session?.messages ?? []).map { msg -> [String: Any] in
            var dict: [String: Any] = ["role": msg.role.rawValue, "content": msg.content]
            if msg.role == .assistant && msg.content.isEmpty { dict.removeValue(forKey: "content") }
            return dict
        }.dropLast() // Drop the empty assistant message we just added
        // Build last user message with potential images
        var lastUserDict: [String: Any] = ["role": "user"]
        if let imgs = attachedImages, !imgs.isEmpty {
            lastUserDict["content"] = Self.buildMultimodalContent(text: text, images: imgs)
        } else {
            lastUserDict["content"] = text
        }
        let messagesArray = Array(messages) + [lastUserDict]

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
                    case .maxTokensReached:
                        appState.updateLastMessage(in: sessionId, content: "\n\n⚠️ *Output truncated — max tokens (\(appState.maxTokens)) reached.*")
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
        let attachedImages = consumePendingImages()
        guard !text.isEmpty || attachedImages != nil, !isGenerating, server.status == .running else { return }
        inputText = ""

        var userMsg = ChatMessage(role: .user, content: text)
        userMsg.images = attachedImages
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


    /// Agent loop: call model with tools (streaming), execute tool calls, feed results back, repeat.
    /// Stops when the model responds with content (no tool calls) or after 150 iterations.
    private func runAgentLoop(api: APIClient, workingDirectory initialWorkDir: String?) async throws {
        var workingDirectory = initialWorkDir
        let maxIterations = 150
        var padRetries = 0
        let padRetryPolicy = RetryPolicy.aggressive
        var recentToolNames: [String] = [] // sliding window for repetition detection
        var permanentlyBlocked: Set<String> = [] // tools blocked for the rest of this loop
        var truncationRetries = 0

        for _ in 0..<maxIterations {
            try Task.checkCancellation()

            // Build message history for API
            var history = buildAgentHistory()
            let userMsg = history.last { ($0["role"] as? String) == "user" }?["content"] as? String ?? ""
            let skills = AgentPrompt.skillManager.matchingSkills(for: userMsg)
            var systemPrompt = AgentPrompt.systemPrompt + skills + AgentPrompt.memory + appState.agentMemory.contextSnippet()
            if let wd = workingDirectory {
                systemPrompt += "\n\n# Working Directory\nYour working directory is `\(wd)`. All file tool operations are confined to this directory — paths that resolve outside it will be rejected. Use relative paths. Shell commands run here by default."
            }
            var messages: [[String: Any]] = [["role": "system", "content": systemPrompt]]
            // Some models (e.g. Gemma 4 E4B) can't generate after tool results without
            // a user message. Add a nudge so the model knows to synthesize a response.
            if let lastRole = history.last?["role"] as? String, lastRole == "tool" {
                history.append(["role": "user", "content": "Continue. If the task is done, summarize the result. If not, take the next step."])
            }
            messages.append(contentsOf: history)

            // Debug: dump the exact request body to file for analysis
            // Uses pre-serialized tools JSON to preserve property key order (path before content)
            do {
                let debugPath = NSString(string: "~/.mlx-serve/last-agent-request.json").expandingTildeInPath
                let messagesData = try JSONSerialization.data(withJSONObject: messages, options: .prettyPrinted)
                let messagesStr = String(data: messagesData, encoding: .utf8) ?? "[]"
                let debugJSON = "{\n  \"model\": \"mlx-serve\",\n  \"max_tokens\": \(appState.maxTokens),\n  \"temperature\": 0.7,\n  \"stream\": true,\n  \"messages\": \(messagesStr),\n  \"tools\": \(AgentPrompt.toolDefinitionsJSON)\n}"
                try debugJSON.data(using: .utf8)?.write(to: URL(fileURLWithPath: debugPath))
            } catch { /* ignore debug errors */ }

            // Add streaming assistant message
            var streamMsg = ChatMessage(role: .assistant, content: "")
            streamMsg.isStreaming = true
            appState.appendMessage(to: sessionId, message: streamMsg)

            // Stream model response with tools
            var receivedToolCalls: [APIClient.ToolCall] = []
            var maxTokensHit = false
            let stream = api.streamChat(
                port: server.port,
                messages: messages,
                maxTokens: appState.maxTokens,
                temperature: 0.7,
                enableThinking: enableThinking,
                toolsJSON: AgentPrompt.toolDefinitionsJSON
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
                case .maxTokensReached:
                    maxTokensHit = true
                    appState.updateLastMessage(in: sessionId, content: "\n\n⚠️ *Output truncated — max tokens (\(appState.maxTokens)) reached. Try breaking the task into smaller steps.*")
                case .done:
                    break
                }
            }
            appState.updateLastMessage(in: sessionId, streaming: false)

            // Truncation recovery: if max_tokens was hit AND tool calls were received,
            // the tool call args are likely truncated (incomplete JSON). Don't execute them —
            // drop the broken message, tell the model what happened, and retry.
            if maxTokensHit && !receivedToolCalls.isEmpty && truncationRetries < 2 {
                truncationRetries += 1
                // Remove the broken assistant message
                if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
                   !appState.chatSessions[sIdx].messages.isEmpty {
                    appState.chatSessions[sIdx].messages.removeLast()
                }
                // Add a user nudge so the model adapts its approach
                let nudge = ChatMessage(role: .user, content: "[System: Your last response was cut off because the output was too long. The tool call was NOT executed. To avoid this, write shorter responses: use shell with heredoc (cat << 'EOF' > file) for file content instead of writeFile, or break large files into smaller pieces.]")
                appState.appendMessage(to: sessionId, message: nudge)
                continue
            }

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

            // Repetition detection: track tool names per round (deduplicated).
            // If same read-only tool appears too often in the sliding window, permanently block it.
            // Write tools (writeFile, editFile, shell) are never blocked — they make progress.
            // Browse/webSearch get a higher threshold (8 in 12) since they're inherently multi-step.
            // Other read-only tools use a tight threshold (4 in 6).
            let writeTools: Set<String> = ["writeFile", "editFile", "shell"]
            let browseTools: Set<String> = ["browse", "webSearch"]
            let uniqueNames = Set(receivedToolCalls.map { $0.name })
            recentToolNames.append(contentsOf: uniqueNames)
            if recentToolNames.count > 12 { recentToolNames = Array(recentToolNames.suffix(12)) }
            for name in uniqueNames {
                if writeTools.contains(name) { continue }
                let count = recentToolNames.filter({ $0 == name }).count
                let threshold = browseTools.contains(name) ? 8 : 4
                if count >= threshold {
                    permanentlyBlocked.insert(name)
                }
            }
            let blockedTools = permanentlyBlocked

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

                // Handle cwd tool: change working directory for subsequent calls
                let output: String
                if effectiveTool == .cwd {
                    if let path = tc.arguments["path"] {
                        let resolved: String
                        if path.hasPrefix("/") || path.hasPrefix("~") {
                            resolved = NSString(string: path).expandingTildeInPath
                        } else if let wd = workingDirectory {
                            resolved = (wd as NSString).appendingPathComponent(path)
                        } else {
                            resolved = path
                        }
                        let normalized = (resolved as NSString).standardizingPath
                        var isDir: ObjCBool = false
                        if FileManager.default.fileExists(atPath: normalized, isDirectory: &isDir), isDir.boolValue {
                            workingDirectory = normalized
                            output = "Changed working directory to \(normalized)"
                        } else {
                            output = "Error: '\(normalized)' is not a directory"
                        }
                    } else {
                        output = "Error: cwd requires a path parameter. Example: {\"path\": \"myproject\"}"
                    }
                } else if blockedTools.contains(tc.name) {
                    output = "BLOCKED: \(tc.name) has been called too many times in a row. This tool is now disabled for this task. Use writeFile to create files, readFile to read them, editFile to modify them, and shell for commands."
                } else {
                    // Pre-validate required params
                    let missing = Self.missingRequiredParams(for: tc.name, arguments: tc.arguments)
                    if !missing.isEmpty {
                        if (tc.name == "writeFile" || tc.name == "editFile") && missing.contains("content") && tc.arguments["path"] != nil {
                            output = "Error: \(tc.name) content was truncated — your output was too long and got cut off before the content was complete. The file was NOT written. To fix this, use shell with a heredoc instead: {\"command\": \"cat << 'FILEEOF' > \(tc.arguments["path"] ?? "path")\\nfile content here\\nFILEEOF\"}"
                        } else {
                            let example = Self.toolExample(for: tc.name)
                            output = "Error: \(tc.name) missing required params: \(missing.joined(separator: ", ")). Example: \(example)"
                        }
                    } else if let effectiveTool, let handler = toolHandlers[effectiveTool] {
                        do {
                            output = try await handler.execute(parameters: tc.arguments, workingDirectory: workingDirectory)
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

                // Extract screenshot image data and attach as vision input
                if tr.name == "browse" && tr.output.contains("data:image/jpeg;base64,") {
                    if let range = tr.output.range(of: "data:image/jpeg;base64,") {
                        // Take only the base64 portion — stop at newline or end of string
                        let remainder = tr.output[range.upperBound...]
                        let b64End = remainder.firstIndex(of: "\n") ?? remainder.endIndex
                        let b64 = String(remainder[..<b64End])
                        if let jpegData = Data(base64Encoded: b64),
                           let chatImage = ChatImage(data: jpegData) as ChatImage? {
                            toolMsg.images = [chatImage]
                            toolMsg.content = "[screenshot captured]"
                        } else {
                            toolMsg.content = Self.truncateWithOverflow(tr.output, toolCallId: tr.id, toolName: tr.name)
                        }
                    } else {
                        toolMsg.content = Self.truncateWithOverflow(tr.output, toolCallId: tr.id, toolName: tr.name)
                    }
                } else {
                    toolMsg.content = Self.truncateWithOverflow(tr.output, toolCallId: tr.id, toolName: tr.name)
                }
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

        // Save full output to disk for debugging, but don't tell the model to read it
        // (the file is outside the workspace so readFile would be blocked by confinement)
        let dir = NSString(string: "~/.mlx-serve/tool-output").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true, attributes: nil)
        let path = (dir as NSString).appendingPathComponent("\(toolCallId).txt")
        try? output.write(toFile: path, atomically: true, encoding: .utf8)

        let preview = String(output.prefix(maxChars))
        return "\(preview)\n\n[... truncated at \(maxChars) of \(output.count) chars]"
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

        // Always pin the first user message (the original task) AND the first
        // assistant response (the plan). Without pinning the plan, context pressure
        // drops the assistant's intent and the model re-interprets the original
        // message literally (e.g. "say hi" → just replies "hi" mid-task).
        let firstUserIdx = allMessages.firstIndex { $0.role == .user && $0.toolCallId == nil }
        let firstAssistantIdx: Int? = {
            guard let uIdx = firstUserIdx else { return nil }
            let afterUser = allMessages.index(after: uIdx)
            guard afterUser < allMessages.count else { return nil }
            return allMessages[afterUser...].firstIndex {
                $0.role == .assistant && !$0.isAgentSummary && !$0.content.isEmpty
            }
        }()
        var pinnedCost = 0
        if let idx = firstUserIdx {
            pinnedCost += roughTokenCount(allMessages[idx].content) + 4
        }
        if let idx = firstAssistantIdx {
            let content = allMessages[idx].content
            // Truncate the plan to 500 chars to avoid blowing the budget on a long first response
            pinnedCost += roughTokenCount(String(content.prefix(500))) + 4
            if let tcs = allMessages[idx].toolCalls {
                for tc in tcs {
                    pinnedCost += roughTokenCount(tc.name) + roughTokenCount(tc.arguments) + 8
                }
            }
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

        // Determine if first user/assistant messages need pinning (fell outside included range)
        let needsPin = firstUserIdx != nil && firstUserIdx! < includeStartIdx
        let needsPinAssistant = firstAssistantIdx != nil && firstAssistantIdx! < includeStartIdx

        // Auto-compact: when context is squeezed, truncate tool results harder.
        // Normal: recent=2000, older=500. Squeezed (<25% free): recent=500, older=100.
        let freeRatio = Double(remainingBudget + pinnedCost) / Double(budget + pinnedCost)
        let squeezed = freeRatio < 0.25
        let recentLimit = squeezed ? 500 : 2000
        let olderLimit = squeezed ? 100 : 500

        // Count tool results in included range for progressive truncation
        let window = Array(allMessages[includeStartIdx..<allMessages.count])
        let totalToolResults = window.filter { $0.toolCallId != nil }.count
        var toolResultsSeen = 0

        var history: [[String: Any]] = []

        // Pin the first user message + assistant plan if they fell outside the window.
        // Strip images from pinned messages — server only processes last user message's images.
        if needsPin, let idx = firstUserIdx {
            history.append(["role": "user", "content": allMessages[idx].content])
        }
        if needsPinAssistant, let idx = firstAssistantIdx {
            let msg = allMessages[idx]
            var content = msg.content
                .replacingOccurrences(of: "<pad>", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if content.count > 500 {
                content = String(content.prefix(500)) + "..."
            }
            var dict: [String: Any] = ["role": "assistant"]
            if !content.isEmpty { dict["content"] = content }
            // Include tool_calls if present so the model remembers its initial actions
            if let tcs = msg.toolCalls, !tcs.isEmpty {
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
            }
            history.append(dict)
        }

        // Find the last user message in the window — only IT gets image content blocks.
        // Server only processes images from the last user message; re-sending old images
        // wastes bandwidth and confuses the vision encoder with stale features.
        let lastUserMsgId = window.last(where: { $0.role == .user && $0.toolCallId == nil })?.id

        for msg in window {
            // Tool response messages — truncate older results more aggressively
            if let callId = msg.toolCallId {
                let isRecent = toolResultsSeen >= totalToolResults - 2
                let limit = isRecent ? recentLimit : olderLimit
                toolResultsSeen += 1
                history.append([
                    "role": "tool",
                    "tool_call_id": callId,
                    "content": String(msg.content.prefix(limit)),
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
            var dict: [String: Any] = ["role": msg.role.rawValue]
            // Only include images for the last user message
            if msg.id == lastUserMsgId, let imgs = msg.images, !imgs.isEmpty {
                dict["content"] = Self.buildMultimodalContent(text: content, images: imgs)
            } else {
                dict["content"] = content
            }
            history.append(dict)
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

// MARK: - Context Monitor

struct ContextMonitor: View {
    let promptTokens: Int
    let contextLength: Int
    let maxTokens: Int

    private var usageRatio: Double {
        guard contextLength > 0 else { return 0 }
        return Double(promptTokens) / Double(contextLength)
    }

    private var generationBudget: Int {
        let remaining = max(0, contextLength - promptTokens)
        return min(remaining, maxTokens)
    }

    private var barColor: Color {
        if usageRatio > 0.80 { return .red }
        if usageRatio > 0.60 { return .orange }
        return .green
    }

    var body: some View {
        VStack(spacing: 2) {
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color.secondary.opacity(0.15))
                    RoundedRectangle(cornerRadius: 2)
                        .fill(barColor.opacity(0.7))
                        .frame(width: geo.size.width * min(1.0, usageRatio))
                }
            }
            .frame(height: 4)

            HStack {
                Text("\(promptTokens)/\(contextLength) tokens (\(Int(usageRatio * 100))%)")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.secondary)
                Spacer()
                Text("gen: \(generationBudget)")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(generationBudget < 2048 ? .red : generationBudget < 4096 ? .orange : .secondary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.top, 4)
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

                // Attached images
                if let images = message.images, !images.isEmpty {
                    HStack(spacing: 4) {
                        ForEach(images) { img in
                            if let nsImage = NSImage(data: img.data) {
                                Image(nsImage: nsImage)
                                    .resizable()
                                    .aspectRatio(contentMode: .fit)
                                    .frame(maxWidth: 200, maxHeight: 150)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                        }
                    }
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
        .contextMenu {
            Button("Copy Message") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(message.content, forType: .string)
            }
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
        .contextMenu {
            Button("Copy All") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(source, forType: .string)
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
