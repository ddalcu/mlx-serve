import SwiftUI
import PDFKit
import UniformTypeIdentifiers
import AppKit

private struct ContentBottomKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}

private struct ScrollViewHeightKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}

/// A tool-call awaiting user approval. The agent loop suspends on
/// `continuation` while the SwiftUI sheet shows the request; the sheet's
/// buttons resume it with the user's choice.
struct ToolApprovalRequest: Identifiable {
    let id = UUID()
    let toolName: String
    let arguments: [String: String]
    /// Raw JSON arguments — used when `arguments` is the post-parse dict but
    /// we want to display the verbatim JSON (handy for nested objects /
    /// arrays the dict-flattening loses).
    let rawArguments: String
    let continuation: CheckedContinuation<ToolApprovalChoice, Never>
}

enum ToolApprovalChoice {
    case allow
    case deny
}

/// Sheet body. Renders the tool name, a pretty-printed argument block, and
/// three buttons. Allow / Deny resume the continuation with that choice;
/// Always Allow flips a per-session flag (in the parent view) and resumes
/// with `.allow`.
private struct ToolApprovalSheet: View {
    let request: ToolApprovalRequest
    let onAllow: () -> Void
    let onDeny: () -> Void
    let onAllowAll: () -> Void

    /// Short, human-readable summary for the most common tools. Falls back to
    /// "Run <tool>" so unknown tools (e.g. MCP server tools) still render.
    private var headline: String {
        switch request.toolName {
        case "shell":      return "Run a shell command"
        case "cwd":        return "Change working directory"
        case "writeFile":  return "Write a file"
        case "editFile":   return "Edit a file"
        case "readFile":   return "Read a file"
        case "searchFiles":return "Search the workspace"
        case "listFiles":  return "List files"
        case "browse":     return "Browse the web"
        case "webSearch":  return "Search the web"
        case "saveMemory": return "Save a memory"
        default:           return "Run \(request.toolName)"
        }
    }

    /// Sorted arg pairs. Prefer the parsed dict; if it's empty (raw is the
    /// only source of truth for arrays/objects), show the raw JSON inline.
    private var argPairs: [(String, String)] {
        request.arguments.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 10) {
                Image(systemName: "shield.lefthalf.filled")
                    .font(.title2)
                    .foregroundStyle(.orange)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Allow this tool call?")
                        .font(.headline)
                    Text(headline)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Tool: \(request.toolName)")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                if argPairs.isEmpty && !request.rawArguments.isEmpty {
                    ScrollView {
                        Text(request.rawArguments)
                            .font(.system(size: 11, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(8)
                    }
                    .frame(maxHeight: 200)
                    .background(Color(NSColor.textBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                } else if argPairs.isEmpty {
                    Text("(no arguments)")
                        .font(.caption.italic())
                        .foregroundStyle(.tertiary)
                } else {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 4) {
                            ForEach(argPairs, id: \.0) { (k, v) in
                                HStack(alignment: .firstTextBaseline, spacing: 6) {
                                    Text(k)
                                        .font(.system(size: 11, design: .monospaced).weight(.semibold))
                                        .foregroundStyle(.secondary)
                                    Text(v)
                                        .font(.system(size: 11, design: .monospaced))
                                        .textSelection(.enabled)
                                        .lineLimit(8)
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(8)
                    }
                    .frame(maxHeight: 240)
                    .background(Color(NSColor.textBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                }
            }

            HStack(spacing: 8) {
                Button(role: .destructive) {
                    onDeny()
                } label: {
                    Text("Deny").frame(minWidth: 70)
                }
                .keyboardShortcut(.cancelAction)

                Spacer()

                Button {
                    onAllowAll()
                } label: {
                    Text("Allow all tools this session").frame(minWidth: 180)
                }

                Button {
                    onAllow()
                } label: {
                    Text("Allow").frame(minWidth: 70)
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(20)
        .frame(width: 520)
    }
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
    @EnvironmentObject var mcpManager: MCPManager
    @Environment(\.openWindow) private var openWindow
    @State private var inputText = ""
    @State private var isGenerating = false
    @State private var enableThinking = false
    @State private var isAgentMode = false
    @State private var showMCPMarketplace = false
    @State private var showThinkingInAgentConfirm = false
    @State private var executingPlanMessageId: UUID?
    @State private var generationTask: Task<Void, Never>?
    @State private var isNearBottom = true
    @State private var scrollViewHeight: CGFloat = 0
    @State private var contentBottom: CGFloat = 0
    @State private var scrollMonitor: Any?
    @State private var pendingImages: [NSImage] = []
    @State private var pendingPDFs: [(name: String, text: String)] = []
    // Tool-approval gate state. `pendingApproval` is set right before each
    // tool call when Agent mode is on; the sheet at the bottom of `body`
    // observes it and resumes `approvalContinuation` with the user's choice.
    // `sessionAllowAll` is a soft "Allow all tools this session" — scoped to
    // *this chat session*: cleared when the user toggles Agent off, and also
    // cleared by the `.onChange(of: sessionId)` below so switching to (or
    // opening) a different chat re-arms the approval prompt.
    @State private var pendingApproval: ToolApprovalRequest?
    @State private var sessionAllowAll = false
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
                        // Bottom anchor — its position relative to the scroll viewport
                        // tells us whether the user has scrolled to the bottom.
                        Color.clear.frame(height: 1).id("bottom")
                            .background(
                                GeometryReader { geo in
                                    Color.clear.preference(
                                        key: ContentBottomKey.self,
                                        value: geo.frame(in: .named("chatScroll")).maxY
                                    )
                                }
                            )
                    }
                    .padding(16)
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
                .onPreferenceChange(ContentBottomKey.self) { bottom in
                    contentBottom = bottom
                    guard scrollViewHeight > 0 else { return }
                    // Re-engage auto-scroll when content bottom is near viewport bottom
                    if bottom - scrollViewHeight < 60 { isNearBottom = true }
                }
                .onChange(of: session?.messages.count) { _, _ in
                    if isNearBottom { scrollToBottom(proxy) }
                }
                .onChange(of: session?.messages.last?.content) { _, _ in
                    if isNearBottom { scrollToBottom(proxy) }
                }
                .overlay(alignment: .trailing) {
                    // Right-edge strip: accent tint when auto-scroll is on, fades out when off.
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color.accentColor.opacity(isNearBottom ? 0.4 : 0))
                        .frame(width: 4)
                        .padding(.vertical, 4)
                        .animation(.easeInOut(duration: 0.3), value: isNearBottom)
                        .allowsHitTesting(false)
                }
            }

            Divider()

            // Context usage monitor
            if let usage = contextUsage, usage.promptTokens > 0 {
                ContextMonitor(promptTokens: usage.promptTokens, contextLength: usage.contextLength, maxTokens: appState.maxTokens)
            }

            // Input area — iMessage style
            VStack(spacing: 4) {
                // Pending attachment thumbnails (images + PDFs)
                if !pendingImages.isEmpty || !pendingPDFs.isEmpty {
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
                            ForEach(Array(pendingPDFs.enumerated()), id: \.offset) { idx, pdf in
                                ZStack(alignment: .topTrailing) {
                                    HStack(spacing: 6) {
                                        Image(systemName: "doc.text.fill")
                                            .font(.system(size: 18))
                                            .foregroundStyle(.white)
                                            .frame(width: 32, height: 32)
                                            .background(Color.red.opacity(0.85))
                                            .clipShape(RoundedRectangle(cornerRadius: 6))
                                        VStack(alignment: .leading, spacing: 1) {
                                            Text(pdf.name)
                                                .font(.caption.weight(.medium))
                                                .lineLimit(1)
                                                .truncationMode(.middle)
                                            Text("PDF · \(pdf.text.count) chars")
                                                .font(.caption2)
                                                .foregroundStyle(.secondary)
                                        }
                                    }
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .frame(maxWidth: 200, minHeight: 56, maxHeight: 56)
                                    .background(Color.secondary.opacity(0.15))
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                                    Button {
                                        pendingPDFs.remove(at: idx)
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
                    // Attachment button (images + PDFs)
                    Button { pickAttachment() } label: {
                        Image(systemName: "paperclip")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(.secondary)
                            .frame(width: 28, height: 28)
                            .background(Color.secondary.opacity(0.15))
                            .clipShape(Circle())
                    }
                    .buttonStyle(.plain)
                    .help("Attach image or PDF")

                    // Dark pill input
                    TextField("Message", text: $inputText, axis: .vertical)
                        .font(.body)
                        .textFieldStyle(.plain)
                        .lineLimit(1...15)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .focused($inputFocused)
                        .disabled(server.status != .running)
                        .onKeyPress(keys: [.return, .init("\u{03}")], phases: .down) { press in
                            if press.modifiers.contains(.shift) {
                                inputText += "\n"
                                return .handled
                            }
                            if !isGenerating {
                                sendMessage()
                            }
                            return .handled
                        }
                        .onAppear {
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                                inputFocused = true
                            }
                        }
                        .background(Color(nsColor: .textBackgroundColor))
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
                    .disabled(server.status != .running || (inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && pendingImages.isEmpty && pendingPDFs.isEmpty && !isGenerating))
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
        .onDrop(of: [.image, .pdf], isTargeted: nil) { providers in
            for provider in providers {
                if provider.hasItemConformingToTypeIdentifier(UTType.pdf.identifier) {
                    provider.loadFileRepresentation(forTypeIdentifier: UTType.pdf.identifier) { url, _ in
                        guard let url = url else { return }
                        let name = url.lastPathComponent
                        if let text = Self.extractPDFText(from: url) {
                            DispatchQueue.main.async {
                                pendingPDFs.append((name: name, text: text))
                            }
                        } else {
                            DispatchQueue.main.async { showPDFError(name) }
                        }
                    }
                } else {
                    provider.loadObject(ofClass: NSImage.self) { image, _ in
                        if let image = image as? NSImage {
                            DispatchQueue.main.async { pendingImages.append(image) }
                        }
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
                    .help("Open ~/.mlx-serve in Finder — your skills/, system-prompt.md, memory.md, chat-history.json, and downloaded models live here.")
                }
            }
            ToolbarItem(placement: .automatic) {
                Button {
                    openWindow(id: "settings")
                } label: {
                    Image(systemName: "gear")
                        .font(.system(size: 12))
                }
                .help("Open Settings (⌘,) — server flags, speculative decoding, performance (continuous batching, KV-quant, prefix cache), and per-request defaults.")
            }
            ToolbarItem(placement: .automatic) {
                Button {
                    if !enableThinking && isAgentMode {
                        showThinkingInAgentConfirm = true
                    } else {
                        enableThinking.toggle()
                    }
                } label: {
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
                .help("Thinking Mode (\(enableThinking ? "ON" : "OFF")) — when the model supports it, it'll emit a private reasoning trace before the visible answer. Slower but better on reasoning-heavy prompts.")
                .padding(.leading, 8)
                .padding(.trailing, 4)
            }
            ToolbarItem(placement: .automatic) {
                Button {
                    isAgentMode.toggle()
                    // Re-arm the approval gate every time the user re-enters
                    // Agent mode. "Always allow this session" decays here.
                    if !isAgentMode { sessionAllowAll = false }
                    // Thinking + tool-calling loops degrade quality on most
                    // local models — auto-off when entering Agent mode.
                    if isAgentMode { enableThinking = false }
                } label: {
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
                .help("""
                Agent Mode (\(isAgentMode ? "ON" : "OFF")) — the model runs a tool-calling loop with these 10 built-in tools:
                  • shell — run a shell command in the workspace
                  • cwd — change the workspace working directory
                  • readFile — read a file (with line range)
                  • writeFile — create or overwrite a small file
                  • editFile — line- or text-based edit of an existing file
                  • searchFiles — ripgrep-style content search across the workspace
                  • listFiles — list paths with glob + recursive options
                  • browse — navigate + extract text/HTML via WKWebView
                  • webSearch — DuckDuckGo search
                  • saveMemory — persist a fact to ~/.mlx-serve/memory.md
                Off: a regular chat with no tools.
                """)
            }
            ToolbarItem(placement: .automatic) {
                HStack(spacing: 0) {
                    Button { appState.mcpMode.toggle() } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "puzzlepiece.extension")
                                .font(.system(size: 11, weight: .medium))
                            Text("MCP")
                                .font(.caption.weight(.medium))
                        }
                        .foregroundStyle(appState.mcpMode ? .white : .secondary)
                        .padding(.leading, 8)
                        .padding(.vertical, 4)
                    }
                    .buttonStyle(.plain)
                    Button { showMCPMarketplace = true } label: {
                        Image(systemName: "gearshape.fill")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(appState.mcpMode ? .white.opacity(0.85) : .secondary)
                            .padding(.trailing, 8)
                            .padding(.leading, 4)
                            .padding(.vertical, 4)
                    }
                    .buttonStyle(.plain)
                    .help("Open MCP Marketplace — browse and enable Model Context Protocol servers (GitHub, Filesystem, Slack, Notion, Playwright, Docker, etc.). Each enabled server's tools become callable in Agent Mode.")
                }
                .background(appState.mcpMode ? .purple : Color.secondary.opacity(0.12))
                .clipShape(Capsule())
                .help("MCP Mode (\(appState.mcpMode ? "ON" : "OFF")) — when on, tools from every enabled Model Context Protocol server are added to the Agent's toolset (alongside the 10 built-ins). Tap the gear to open the Marketplace and toggle servers on/off.")
            }
            ToolbarItem(placement: .automatic) {
                Circle()
                    .fill(server.status == .running ? .green : .red)
                    .frame(width: 8, height: 8)
                    .padding(.horizontal, 8)
                    .help(server.status == .running ? "Server running" : "Server stopped")
            }
        }
        .sheet(isPresented: $showMCPMarketplace) {
            MCPMarketplaceView()
                .environmentObject(mcpManager)
        }
        .alert("Enable thinking in Agent mode?", isPresented: $showThinkingInAgentConfirm) {
            Button("Cancel", role: .cancel) { }
            Button("Enable anyway") { enableThinking = true }
        } message: {
            Text("Thinking is not recommended with Agent mode — most local models tool-call more reliably without it. Do you still want to enable it?")
        }
        .sheet(item: $pendingApproval) { req in
            ToolApprovalSheet(request: req,
                              onAllow: { req.continuation.resume(returning: .allow) ; pendingApproval = nil },
                              onDeny: { req.continuation.resume(returning: .deny) ; pendingApproval = nil },
                              onAllowAll: { sessionAllowAll = true
                                            req.continuation.resume(returning: .allow) ; pendingApproval = nil })
        }
        .onAppear {
            inputFocused = true
            isAgentMode = session?.mode == .agent
            scrollMonitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { event in
                if event.scrollingDeltaY > 0 {
                    // Scrolling up — disengage auto-scroll
                    isNearBottom = false
                } else if event.scrollingDeltaY < -1 {
                    // Scrolling down — re-engage if content bottom is near viewport bottom.
                    // The preference handler catches this when content changes, but during
                    // generation pauses the user needs scroll events to re-engage.
                    if scrollViewHeight > 0 && contentBottom - scrollViewHeight < 80 {
                        isNearBottom = true
                    }
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
        .onChange(of: sessionId) { _, _ in
            // "Allow all tools this session" is scoped to a chat session —
            // switching to a different chat (or opening a new one) must
            // re-prompt on the next tool call. SwiftUI reuses this view
            // across sessionId changes, so reset the flag explicitly.
            sessionAllowAll = false
        }
    }

    // MARK: - Image Helpers

    private func pickAttachment() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image, .pdf]
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.begin { response in
            guard response == .OK else { return }
            for url in panel.urls {
                if url.pathExtension.lowercased() == "pdf" {
                    if let text = Self.extractPDFText(from: url) {
                        pendingPDFs.append((name: url.lastPathComponent, text: text))
                    } else {
                        showPDFError(url.lastPathComponent)
                    }
                } else if let image = NSImage(contentsOf: url) {
                    pendingImages.append(image)
                }
            }
        }
    }

    /// Returns nil if the PDF is unreadable, encrypted, or contains no extractable text
    /// (e.g. scanned-image-only PDFs without an OCR layer).
    static func extractPDFText(from url: URL) -> String? {
        guard let pdf = PDFDocument(url: url),
              let text = pdf.string,
              !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }
        return text
    }

    private func showPDFError(_ name: String) {
        let alert = NSAlert()
        alert.messageText = "Couldn't read PDF"
        alert.informativeText = "\(name) is empty, encrypted, or contains only scanned images (no extractable text)."
        alert.alertStyle = .warning
        alert.runModal()
    }

    /// Build a preamble string that joins all pending PDFs and clears the list.
    /// Returns "" when nothing is pending.
    private func consumePendingPDFsAsText() -> String {
        guard !pendingPDFs.isEmpty else { return "" }
        let combined = pendingPDFs.map { "[PDF: \($0.name)]\n\($0.text)" }.joined(separator: "\n\n")
        pendingPDFs = []
        return combined
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
        if let last = messages.last(where: { $0.promptTokens != nil && $0.promptTokens! > 0 }) {
            let ctxLen = AgentEngine.effectiveContextLength(
                appContextSize: appState.contextSize,
                modelContextLength: server.modelInfo?.contextLength
            )
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
        if isAgentMode || appState.mcpMode {
            sendAgentMessage()
            return
        }

        var text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        let attachedImages = consumePendingImages()
        let pdfText = consumePendingPDFsAsText()
        guard !text.isEmpty || attachedImages != nil || !pdfText.isEmpty, !isGenerating, server.status == .running else { return }
        inputText = ""
        if !pdfText.isEmpty {
            text = text.isEmpty ? pdfText : pdfText + "\n\n" + text
        }

        var userMsg = ChatMessage(role: .user, content: text)
        userMsg.images = attachedImages
        appState.appendMessage(to: sessionId, message: userMsg)

        isGenerating = true
        let api = APIClient()

        // Build the request from the session (its source of truth). We append
        // the streaming placeholder AFTER this so it never lands in the
        // request — same pattern the agent loop uses. Image handling: only
        // the latest user message's images are sent (older turns' images are
        // stripped for bandwidth).
        let sessionMsgs = session?.messages ?? []
        let lastUserIdx = sessionMsgs.lastIndex { $0.role == .user }
        let history: [[String: Any]] = sessionMsgs.enumerated().map { i, msg in
            if i == lastUserIdx, msg.role == .user, let imgs = msg.images, !imgs.isEmpty {
                return ["role": "user", "content": Self.buildMultimodalContent(text: msg.content, images: imgs)]
            }
            var d: [String: Any] = ["role": msg.role.rawValue, "content": msg.content]
            if msg.role == .assistant && msg.content.isEmpty { d.removeValue(forKey: "content") }
            return d
        }
        // Plain chat: no synthesized system message. Earlier versions
        // prepended a "formatNudge" system message asking the model to use
        // Markdown tables / fenced code / bold. That nudge had no persona
        // and was routinely interpreted by the model AS the user's input —
        // first-turn replies came back as meta-commentary like "It seems
        // like you're providing a formatting reminder, but it looks like
        // you may have intended to ask a question". Removing the nudge
        // entirely; smaller models will occasionally emit whitespace-
        // aligned tables instead of Markdown grids, which is a far better
        // failure mode than the model addressing the system instruction
        // instead of the user. Agent mode still has its own system prompt
        // (AgentPrompt.swift) which DOES carry a full persona.
        let messagesArray = history

        // Streaming placeholder for the UI — appended AFTER the request body
        // is built so it doesn't show up in the prompt. If we added it
        // before building, the session would have a trailing empty assistant
        // turn that we'd have to drop back out (and the user message we just
        // appended would be on the line above it — easy to double-add by
        // mistake, which on small / 2-bit-quant models like DSV4-Flash
        // produces a repeating-token collapse at temp > 0).
        var assistantMsg = ChatMessage(role: .assistant, content: "")
        assistantMsg.isStreaming = true
        appState.appendMessage(to: sessionId, message: assistantMsg)

        generationTask = Task {
            do {
                // Plan 05 Phase G — pin the request to the active model
                // (server-resolved default if nil) so hot-switch can finish
                // in-flight requests on the old model before the new one
                // takes over.
                let stream = api.streamChat(
                    port: server.port,
                    messages: messagesArray,
                    maxTokens: appState.maxTokens,
                    temperature: appState.serverOptions.defaultTemperature,
                    enableThinking: enableThinking || appState.serverOptions.defaultEnableThinking,
                    defaults: APIClient.RequestDefaults.from(appState.serverOptions),
                    modelId: appState.server.modelInfo?.name
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
        var text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        let attachedImages = consumePendingImages()
        let pdfText = consumePendingPDFsAsText()
        guard !text.isEmpty || attachedImages != nil || !pdfText.isEmpty, !isGenerating, server.status == .running else { return }
        inputText = ""
        if !pdfText.isEmpty {
            text = text.isEmpty ? pdfText : pdfText + "\n\n" + text
        }

        var userMsg = ChatMessage(role: .user, content: text)
        userMsg.images = attachedImages
        appState.appendMessage(to: sessionId, message: userMsg)

        isGenerating = true
        let api = APIClient()
        let workDir = session?.workingDirectory

        generationTask = Task {
            // Lazy-spawn MCP servers if MCP mode is on. Idempotent — already-connected servers are skipped.
            if appState.mcpMode {
                // Inherit the chat's working directory so filesystem/shell MCP servers anchor at the
                // same dir the agent's built-in tools use. Per-entry `cwd` in mcp.json still wins.
                mcpManager.defaultCwd = session?.workingDirectory
                await mcpManager.startEnabled()
                // Surface startup failures inline in chat — otherwise they're hidden behind the
                // marketplace gear icon and the user just sees "MCP doesn't seem to do anything".
                if !mcpManager.startErrors.isEmpty {
                    let lines = mcpManager.startErrors
                        .sorted(by: { $0.key < $1.key })
                        .map { "• **\($0.key)**: \($0.value)" }
                        .joined(separator: "\n")
                    let hint = mcpManager.sessions.isEmpty
                        ? "No MCP servers are connected — the model has no MCP tools available for this turn. Open the gear icon on the MCP pill to fix or disable broken servers."
                        : "Some MCP servers couldn't start. The model will only see tools from the ones that did connect."
                    let warning = ChatMessage(
                        role: .assistant,
                        content: "⚠️ MCP startup issues:\n\n\(lines)\n\n\(hint)"
                    )
                    appState.appendMessage(to: sessionId, message: warning)
                }
            }
            do {
                try await runAgentLoop(api: api, workingDirectory: workDir)
            } catch is CancellationError {
                // Stopped by user — stopGenerating() already cleared the streaming flag.
            } catch {
                print("[ChatView] Agent error: \(error)")
                try? "Agent error: \(error)\n".write(toFile: NSString(string: "~/.mlx-serve/debug.log").expandingTildeInPath, atomically: true, encoding: .utf8)
                // Clear the spinner on the in-flight assistant message before appending the error;
                // otherwise GeneratingIndicator stays visible on the orphaned streaming bubble.
                appState.updateLastMessage(in: sessionId, streaming: false)
                var errorMsg = ChatMessage(role: .assistant, content: "[Error: \(error.localizedDescription)]")
                errorMsg.isStreaming = false
                appState.appendMessage(to: sessionId, message: errorMsg)
            }
            appState.saveChatHistory()
            isGenerating = false
            generationTask = nil
        }
    }


    /// Ask the user to approve a single tool call. Returns true on Allow /
    /// Always Allow, false on Deny. Bypassed entirely when `sessionAllowAll`
    /// is on. Bounces to the main actor (state mutations + sheet presentation)
    /// and suspends on a checked continuation until the sheet resumes it.
    @MainActor
    private func requestToolApproval(_ tc: APIClient.ToolCall) async -> Bool {
        if sessionAllowAll { return true }
        let choice: ToolApprovalChoice = await withCheckedContinuation { (cont: CheckedContinuation<ToolApprovalChoice, Never>) in
            pendingApproval = ToolApprovalRequest(
                toolName: tc.name,
                arguments: tc.arguments,
                rawArguments: tc.rawArguments,
                continuation: cont
            )
        }
        return choice == .allow
    }

    /// Agent loop: call model with tools (streaming), execute tool calls, feed results back, repeat.
    /// Stops when the model responds with content (no tool calls) or after 150 iterations.
    private func runAgentLoop(api: APIClient, workingDirectory initialWorkDir: String?) async throws {
        var workingDirectory = initialWorkDir
        let maxIterations = 150
        var padRetries = 0
        let padRetryPolicy = RetryPolicy.aggressive
        let repetition = AgentEngine.RepetitionTracker()
        var truncationRetries = 0
        // One retry when the model exits with a malformed/ghost tool-call tag
        // in its content instead of a proper finish — re-prompt for a clean
        // plain-text summary so the user isn't left staring at `<|tool_call>…`.
        var completionRetries = 0

        for iteration in 0..<maxIterations {
            try Task.checkCancellation()

            // Build message history for API
            let contextLength = AgentEngine.effectiveContextLength(
                appContextSize: appState.contextSize,
                modelContextLength: server.modelInfo?.contextLength
            )
            var history = AgentEngine.buildAgentHistory(
                messages: session?.messages ?? [],
                contextLength: contextLength,
                maxTokens: appState.maxTokens,
                buildMultimodalContent: Self.buildMultimodalContent
            )
            let userMsg = history.last { ($0["role"] as? String) == "user" }?["content"] as? String ?? ""
            let mcpToolsJSON = appState.mcpMode ? mcpManager.toolDefinitionsJSON() : nil
            let mcpListing = appState.mcpMode ? mcpManager.toolListingForPrompt() : ""
            var systemPrompt: String
            if isAgentMode {
                let skills = AgentPrompt.skillManager.matchingSkills(for: userMsg)
                systemPrompt = AgentPrompt.systemPrompt + skills + AgentPrompt.memory + appState.agentMemory.contextSnippet()
                if let wd = workingDirectory {
                    systemPrompt += AgentEngine.workingDirectoryContext(wd)
                }
                if !mcpListing.isEmpty {
                    systemPrompt += "\n\n# MCP Tools\nIn addition to the built-in tools above, the user has connected these MCP servers. Their tools are namespaced as `<server>__<tool>`:\n\n\(mcpListing)"
                }
            } else {
                // MCP-only mode: minimal system prompt focused on MCP tool use, no shell/file rules.
                systemPrompt = AgentPrompt.mcpOnlySystemPrompt(toolListing: mcpListing)
            }
            var messages: [[String: Any]] = [["role": "system", "content": systemPrompt]]
            // Some models (e.g. Gemma 4 E4B) can't generate after tool results without
            // a user message. Add a nudge so the model knows to synthesize a response —
            // asks explicitly for a short plain-text summary when finished so the user
            // never sees a conversation that ends on a bare tool-call echo.
            if let lastRole = history.last?["role"] as? String, lastRole == "tool" {
                history.append(["role": "user", "content": "Continue. If the task is complete, reply with a short plain-text summary for the user (what got done, where it lives, any caveats) — no tool calls, no JSON. If more work is needed, make the next tool call."])
            }
            messages.append(contentsOf: history)

            AgentEngine.dumpDebugRequest(messages: messages, maxTokens: appState.maxTokens)

            // Add streaming assistant message
            var streamMsg = ChatMessage(role: .assistant, content: "")
            streamMsg.isStreaming = true
            appState.appendMessage(to: sessionId, message: streamMsg)

            // Stream model response with tools
            var receivedToolCalls: [APIClient.ToolCall] = []
            var maxTokensHit = false
            let combinedToolsJSON = Self.combinedToolsJSON(
                agentMode: isAgentMode,
                mcpToolsJSON: mcpToolsJSON
            )
            let stream = api.streamChat(
                port: server.port,
                messages: messages,
                maxTokens: appState.maxTokens,
                temperature: 0.7,
                enableThinking: enableThinking,
                toolsJSON: combinedToolsJSON,
                defaults: APIClient.RequestDefaults.from(appState.serverOptions),
                modelId: appState.server.modelInfo?.name
            )

            // No client-side stream watchdog: long generations (large
            // contexts, big batches, slow sampling on big MoE) can legitimately
            // sit silent for minutes between events. The user keeps the Stop
            // button as the manual cancel; URLSession's own resource timeout
            // (set in APIClient) handles a truly broken socket.
            let streamTask = Task<(tcs: [APIClient.ToolCall], maxHit: Bool), Error> {
                var tcs: [APIClient.ToolCall] = []
                var maxHit = false
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
                        tcs = calls
                    case .maxTokensReached:
                        maxHit = true
                        appState.updateLastMessage(in: sessionId, content: "\n\n⚠️ *Output truncated — max tokens (\(appState.maxTokens)) reached. Try breaking the task into smaller steps.*")
                    case .done:
                        break
                    }
                }
                return (tcs, maxHit)
            }
            // Wire the user's Stop button through to the inner stream task.
            do {
                let result = try await withTaskCancellationHandler {
                    try await streamTask.value
                } onCancel: {
                    streamTask.cancel()
                }
                receivedToolCalls = result.tcs
                maxTokensHit = result.maxHit
            } catch is CancellationError {
                throw CancellationError()
            }
            appState.updateLastMessage(in: sessionId, streaming: false)

            // Truncation recovery: if max_tokens was hit AND tool calls were received,
            // the tool call args are likely truncated (incomplete JSON). Don't execute them —
            // mark the broken message as non-replayable (preserves reasoning in the UI)
            // and nudge the model to try again more concisely.
            if maxTokensHit && !receivedToolCalls.isEmpty && truncationRetries < 2 {
                truncationRetries += 1
                if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
                   !appState.chatSessions[sIdx].messages.isEmpty {
                    let mIdx = appState.chatSessions[sIdx].messages.count - 1
                    appState.chatSessions[sIdx].messages[mIdx].failedRetry = true
                    appState.chatSessions[sIdx].messages[mIdx].toolCalls = nil
                }
                let nudge = ChatMessage(role: .user, content: "[System: Your last response was cut off because the output was too long. The tool call was NOT executed. To avoid this, write shorter responses: use shell with heredoc (cat << 'EOF' > file) for file content instead of writeFile, or break large files into smaller pieces.]")
                appState.appendMessage(to: sessionId, message: nudge)
                continue
            }

            // Check for pad-only or empty responses — retry limited times.
            // Mark the empty message as failedRetry so it's hidden from API history
            // but its reasoning (if any) stays visible in the UI.
            if receivedToolCalls.isEmpty {
                let lastContent = appState.chatSessions
                    .first(where: { $0.id == sessionId })?.messages.last?.content ?? ""
                let cleaned = lastContent
                    .replacingOccurrences(of: "<pad>", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if cleaned.isEmpty {
                    if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
                       !appState.chatSessions[sIdx].messages.isEmpty {
                        let mIdx = appState.chatSessions[sIdx].messages.count - 1
                        appState.chatSessions[sIdx].messages[mIdx].failedRetry = true
                    }
                    padRetries += 1
                    if padRetries <= padRetryPolicy.maxRetries {
                        let delay = padRetryPolicy.delay(for: padRetries)
                        try? await Task.sleep(nanoseconds: delay)
                        continue
                    }
                    let errorMsg = ChatMessage(role: .assistant, content: "The model couldn't generate a response. Try rephrasing or starting a new chat.")
                    appState.appendMessage(to: sessionId, message: errorMsg)
                    return
                }
            }

            // If no tool calls, we're done — but make sure the user sees a
            // clean completion text. The model sometimes exits with a ghost
            // tool call (malformed <|tool_call>...<tool_call|> or <tool_call>
            // with bad args that didn't parse) as its final content; that's
            // ugly and uninformative. When we detect one, mark the garbled
            // turn as failedRetry (hidden from API history) and ask the model
            // for a plain-text summary before returning control to the user.
            if receivedToolCalls.isEmpty {
                let lastContent = appState.chatSessions
                    .first(where: { $0.id == sessionId })?.messages.last?.content ?? ""
                // Match the full `<tool…` family — `<tool_call>`, `<tool_call name=…>`,
                // `<tool_calls>` wrapper, `<tool name=… arguments=…/>` self-closing,
                // Gemma 4 `<|tool_call>`/`<tool_call|>`, and `<function=` legacy. The
                // server-side `parseToolCalls` already handles all of these; this
                // check is the defense-in-depth that fires the retry nudge when
                // a new model variant slips through the parser before we recognize
                // it (the symptom: hundreds of completion_tokens but the assistant
                // turn ends with markup-as-content and no parsed tool_calls).
                let looksLikeGhostToolCall = lastContent.contains("<|tool_call>") ||
                    lastContent.contains("<tool_call>") ||
                    lastContent.contains("<tool_call ") ||
                    lastContent.contains("<tool_calls>") ||
                    lastContent.contains("<tool_calls ") ||
                    lastContent.contains("<tool_call|>") ||
                    lastContent.contains("<tool name=") ||
                    lastContent.contains("<function=")
                if looksLikeGhostToolCall && completionRetries < 1 {
                    completionRetries += 1
                    if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
                       !appState.chatSessions[sIdx].messages.isEmpty {
                        let mIdx = appState.chatSessions[sIdx].messages.count - 1
                        appState.chatSessions[sIdx].messages[mIdx].failedRetry = true
                    }
                    let nudge = ChatMessage(role: .user, content: "[System: your last response contained a malformed tool-call tag. If you meant to call a tool, call it with proper JSON. If the task is complete, respond with a short plain-text summary of what you did — no tool tags, no JSON — just a sentence or two for the user.]")
                    appState.appendMessage(to: sessionId, message: nudge)
                    continue
                }
                return
            }

            // Track repetition for this round
            repetition.track(toolCalls: receivedToolCalls)

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

            // Show tool call summary as display-only message. Mark streaming so the GeneratingIndicator
            // keeps rendering underneath while tools execute — otherwise a slow / hung MCP tool looks
            // like the chat just froze with no feedback.
            let callSummary = receivedToolCalls.map { tc in
                let args = tc.arguments.map { "\($0.key): \($0.value.prefix(80))" }.joined(separator: ", ")
                let display = args.isEmpty ? tc.rawArguments.prefix(200) : args[...]
                return "**\(tc.name)**(\(display))"
            }.joined(separator: "\n")
            var summaryMsg = ChatMessage(role: .assistant, content: callSummary)
            summaryMsg.isAgentSummary = true
            summaryMsg.isStreaming = true
            let summaryId = summaryMsg.id
            appState.appendMessage(to: sessionId, message: summaryMsg)
            // Stop the spinner on the summary regardless of how we leave the loop (success, throw, cancel).
            defer {
                if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
                   let mIdx = appState.chatSessions[sIdx].messages.firstIndex(where: { $0.id == summaryId }) {
                    appState.chatSessions[sIdx].messages[mIdx].isStreaming = false
                }
            }

            // Execute each tool call. MCP-namespaced names (`<server>__<tool>`) route to MCPManager;
            // everything else flows through the existing AgentEngine dispatch.
            // Tool-approval gate: before every dispatch, ask the user unless
            // they've flipped "Always allow this session". Deny short-circuits
            // to a fabricated error result so the agent loop can react and
            // the user's intent is visible in the transcript.
            for tc in receivedToolCalls {
                try Task.checkCancellation()

                let approved = await requestToolApproval(tc)
                guard approved else {
                    let denied = AgentEngine.ToolResult(
                        id: tc.id,
                        name: tc.name,
                        output: "Error: user denied this tool call. Do not retry this command; ask the user how to proceed or try a different approach."
                    )
                    var deniedMsg = ChatMessage(role: .assistant, content: "**\(tc.name)** → denied by user")
                    deniedMsg.isAgentSummary = true
                    appState.appendMessage(to: sessionId, message: deniedMsg)
                    var toolMsg = ChatMessage(role: .system, content: denied.output)
                    toolMsg.toolCallId = denied.id
                    toolMsg.toolName = denied.name
                    appState.appendMessage(to: sessionId, message: toolMsg)
                    continue
                }

                let result: AgentEngine.ToolResult
                if mcpManager.owns(toolName: tc.name) {
                    let output = await mcpManager.executeToolCall(
                        name: tc.name, arguments: tc.arguments, rawArguments: tc.rawArguments
                    )
                    result = AgentEngine.ToolResult(id: tc.id, name: tc.name, output: output)
                } else {
                    result = await AgentEngine.executeToolCall(
                        tc, workingDirectory: &workingDirectory,
                        repetition: repetition, iteration: iteration,
                        agentMemory: appState.agentMemory
                    )
                }

                // Show result in chat (display-only)
                var resultMsg = ChatMessage(role: .assistant, content: "**\(result.name)** → \(String(result.output.prefix(500)))")
                resultMsg.isAgentSummary = true
                appState.appendMessage(to: sessionId, message: resultMsg)

                // Store tool result as tool role message
                var toolMsg = ChatMessage(role: .system, content: "")
                toolMsg.toolCallId = result.id
                toolMsg.toolName = result.name

                // Extract screenshot image data and attach as vision input
                if result.name == "browse" && result.output.contains("data:image/jpeg;base64,") {
                    if let range = result.output.range(of: "data:image/jpeg;base64,") {
                        let remainder = result.output[range.upperBound...]
                        let b64End = remainder.firstIndex(of: "\n") ?? remainder.endIndex
                        let b64 = String(remainder[..<b64End])
                        if let jpegData = Data(base64Encoded: b64),
                           let chatImage = ChatImage(data: jpegData) as ChatImage? {
                            toolMsg.images = [chatImage]
                            toolMsg.content = "[screenshot captured]"
                        } else {
                            toolMsg.content = AgentEngine.truncateWithOverflow(result.output, toolCallId: result.id, toolName: result.name)
                        }
                    } else {
                        toolMsg.content = AgentEngine.truncateWithOverflow(result.output, toolCallId: result.id, toolName: result.name)
                    }
                } else {
                    toolMsg.content = AgentEngine.truncateWithOverflow(result.output, toolCallId: result.id, toolName: result.name)
                }
                appState.appendMessage(to: sessionId, message: toolMsg)
            }
        }

        // Max iterations reached
        let msg = ChatMessage(role: .assistant, content: "(Agent stopped after \(maxIterations) tool call rounds)")
        appState.appendMessage(to: sessionId, message: msg)
    }

    /// Build the JSON tools array sent to the model. Concatenates agent tools (when agent mode is on) and
    /// MCP tools (when MCP mode is on). Returns nil when no tools should be advertised.
    static func combinedToolsJSON(agentMode: Bool, mcpToolsJSON: String?) -> String? {
        let agent = agentMode ? AgentPrompt.toolDefinitionsJSON : nil
        switch (agent, mcpToolsJSON) {
        case (nil, nil): return nil
        case (let a?, nil): return a
        case (nil, let m?): return m
        case (let a?, let m?):
            // Strip outer brackets and re-wrap. Both inputs are guaranteed to be JSON arrays.
            let aInner = a.trimmingCharacters(in: .whitespacesAndNewlines)
                .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
            let mInner = m.trimmingCharacters(in: .whitespacesAndNewlines)
                .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
            let aTrimmed = aInner.trimmingCharacters(in: .whitespacesAndNewlines)
            let mTrimmed = mInner.trimmingCharacters(in: .whitespacesAndNewlines)
            if aTrimmed.isEmpty { return "[\(mTrimmed)]" }
            if mTrimmed.isEmpty { return "[\(aTrimmed)]" }
            return "[\(aTrimmed),\(mTrimmed)]"
        }
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

// MARK: - Generating Indicator

/// Animated indicator shown while the model is generating, with live GPU and memory stats.
struct GeneratingIndicator: View {
    @State private var gpuPercent: Int = 0
    @State private var memPercent: Int = 0
    @State private var whimsy: String = Self.randomWhimsy()
    @State private var timer: Timer?
    @State private var startDate = Date()

    var body: some View {
        TimelineView(.animation) { context in
            let elapsed = context.date.timeIntervalSince(startDate)
            let outerAngle = elapsed * 120  // degrees per second
            let innerAngle = -elapsed * 168 // counter-rotate, slightly faster

            HStack(spacing: 8) {
                // Spinning arcs — continuous, no reset
                ZStack {
                    // Outer arc — GPU usage mapped to arc length
                    Circle()
                        .trim(from: 0, to: max(0.1, Double(gpuPercent) / 100.0))
                        .stroke(gpuColor, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                        .frame(width: 18, height: 18)
                        .rotationEffect(.degrees(outerAngle))

                    // Inner arc — memory
                    Circle()
                        .trim(from: 0, to: max(0.1, Double(memPercent) / 100.0))
                        .stroke(memColor, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                        .frame(width: 10, height: 10)
                        .rotationEffect(.degrees(innerAngle))

                    // Center dot pulses with GPU activity
                    Circle()
                        .fill(gpuColor)
                        .frame(width: 3, height: 3)
                        .scaleEffect(1.0 + 0.3 * sin(elapsed * 4))
                }
                .frame(width: 20, height: 20)

                // Stats + whimsy
                Text("GPU \(gpuPercent)%")
                    .foregroundStyle(gpuColor)
                Text("·")
                    .foregroundStyle(.tertiary)
                Text("Mem \(memPercent)%")
                    .foregroundStyle(memColor)
                Text("·")
                    .foregroundStyle(.tertiary)
                Text(whimsy)
                    .foregroundStyle(.secondary)
                    .transition(.opacity)
                Text("·")
                    .foregroundStyle(.tertiary)
                Text(Self.formatElapsed(elapsed))
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            .font(.system(size: 10, weight: .medium, design: .monospaced))
        }
        .onAppear {
            startDate = Date()
            pollMetrics()
            timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
                pollMetrics()
            }
        }
        .onDisappear {
            timer?.invalidate()
            timer = nil
        }
    }

    private var gpuColor: Color {
        if gpuPercent > 80 { return .orange }
        if gpuPercent > 50 { return .green }
        return .blue
    }

    private var memColor: Color {
        if memPercent > 85 { return .red }
        if memPercent > 70 { return .orange }
        return .secondary
    }

    private func pollMetrics() {
        gpuPercent = Int(SystemMetrics.gpuUtilization())
        memPercent = Int(SystemMetrics.memoryPressure())
        // Rotate whimsy every ~3 seconds
        if Int(Date().timeIntervalSince(startDate)) % 3 == 0 {
            withAnimation(.easeInOut(duration: 0.3)) {
                whimsy = Self.randomWhimsy()
            }
        }
    }

    private static let whimsies = [
        "marinating", "boondoggling", "razzle-dazzling", "percolating",
        "simmering", "noodling", "cogitating", "ruminating",
        "brainstorming", "daydreaming", "scheming", "concocting",
        "fermenting", "hatching", "brewing", "stewing",
        "tinkering", "finagling", "wrangling", "bamboozling",
        "gallivanting", "meandering", "pondering", "mulling",
        "churning", "synthesizing", "vibing", "manifesting",
        "jazz-handing", "shimmer-thinking", "pixel-wrangling",
        "quantum-leaping", "brain-tickling", "thought-juggling",
    ]

    private static func randomWhimsy() -> String {
        whimsies.randomElement() ?? "thinking"
    }

    /// Compact elapsed-time format: "0s", "9s", "59s", "1m04s", "12m08s",
    /// "1h02m". Designed to read at 10pt monospaced without ever changing
    /// width by more than one glyph as the timer ticks.
    private static func formatElapsed(_ seconds: Double) -> String {
        let total = max(0, Int(seconds))
        if total < 60 { return "\(total)s" }
        if total < 3600 {
            let m = total / 60, s = total % 60
            return String(format: "%dm%02ds", m, s)
        }
        let h = total / 3600, m = (total % 3600) / 60
        return String(format: "%dh%02dm", h, m)
    }
}

/// Reads macOS GPU utilization and memory pressure via IOKit/Mach (same APIs as status.zig).
enum SystemMetrics {

    /// GPU utilization percentage (0–100) via IOKit AGXAccelerator.
    static func gpuUtilization() -> UInt32 {
        var iter: io_iterator_t = 0
        guard let matching = IOServiceMatching("AGXAccelerator") else { return 0 }
        guard IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iter) == KERN_SUCCESS else { return 0 }
        defer { IOObjectRelease(iter) }

        let entry = IOIteratorNext(iter)
        guard entry != 0 else { return 0 }
        defer { IOObjectRelease(entry) }

        var propsRef: Unmanaged<CFMutableDictionary>?
        guard IORegistryEntryCreateCFProperties(entry, &propsRef, kCFAllocatorDefault, 0) == KERN_SUCCESS,
              let props = propsRef?.takeRetainedValue() as? [String: Any],
              let perf = props["PerformanceStatistics"] as? [String: Any],
              let util = perf["Device Utilization %"] as? Int else { return 0 }
        return UInt32(min(max(util, 0), 100))
    }

    /// System memory pressure as percentage (0–100) via Mach host_statistics64.
    static func memoryPressure() -> UInt32 {
        var totalMem: UInt64 = 0
        var len = MemoryLayout<UInt64>.size
        guard sysctlbyname("hw.memsize", &totalMem, &len, nil, 0) == 0, totalMem > 0 else { return 0 }

        var pageSize: vm_size_t = 0
        guard host_page_size(mach_host_self(), &pageSize) == KERN_SUCCESS, pageSize > 0 else { return 0 }

        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<Int32>.size)
        let result = withUnsafeMutablePointer(to: &stats) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                host_statistics64(mach_host_self(), HOST_VM_INFO64, intPtr, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }

        let used = (UInt64(stats.active_count) + UInt64(stats.wire_count) + UInt64(stats.compressor_page_count)) * UInt64(pageSize)
        return UInt32(used * 100 / totalMem)
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
                            GeneratingIndicator()
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
//
// Rendering goes through a single NSTextView (via SelectableMarkdownNSText) so the
// user can drag-select across the *entire* assistant message — paragraphs, lists,
// code blocks, tables, the lot. Stacking individual SwiftUI Text views inside a
// VStack used to break selection at every block boundary because each Text is its
// own NSTextStorage island; a single NSTextView is the only reliable way to get
// macOS-native cross-block selection. Block parsing still happens here in Swift —
// each Block becomes a styled fragment of the assembled NSAttributedString.
struct MarkdownText: View {
    let source: String

    /// Tags emitted by models (thinking, planning, etc.) — rendered as XML blocks.
    /// Standard HTML tags (head, div, meta, etc.) are NOT included — they render as text.
    private static let modelTags: Set<String> = [
        "pad", "plan", "thinking", "thought", "reflection", "output",
        "step", "result", "answer", "reasoning", "tool_call", "tool_response",
    ]

    /// Tags whose content should be hidden from the chat entirely (consumed but
    /// not rendered). Real tool calls show in the dedicated tool-call UI; raw
    /// `<tool_call>` text in the assistant bubble is either a parser fallback or
    /// a malformed/truncated leak — neither is useful to display.
    private static let hiddenTags: Set<String> = [
        "tool_call", "tool_response",
    ]

    init(_ source: String) {
        self.source = source
    }

    var body: some View {
        SelectableMarkdownNSText(attributed: Self.attributedString(for: source))
            .contextMenu {
                Button("Copy All") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(source, forType: .string)
                }
            }
    }

    enum TableAlignment { case left, right, center }

    fileprivate enum Block {
        case paragraph(String)
        case heading(Int, String)              // level, text
        case code(String, String)              // language, content
        case listItem(String)
        case xmlBlock(String)                  // raw XML/tag content
        case table([String], [[String]], [TableAlignment])  // headers, rows, alignments
    }

    fileprivate static func parseBlocks(source: String) -> [Block] {
        var blocks: [Block] = []
        let lines = source.components(separatedBy: "\n")
        var i = 0

        while i < lines.count {
            let line = lines[i]

            // XML-like tag block for model-specific tags (<plan>, <pad>, <thinking>, etc.)
            // Only match known model tags — NOT standard HTML tags like <head>, <div>, <meta>.
            if let match = line.range(of: "^<([a-zA-Z_]+)>", options: .regularExpression) {
                let tag = String(line[match]).dropFirst().dropLast() // extract tag name
                guard Self.modelTags.contains(String(tag)) else {
                    // Not a model tag — fall through to normal paragraph handling
                    i += 1
                    let text = line.trimmingCharacters(in: .whitespaces)
                    if !text.isEmpty { blocks.append(.paragraph(text)) }
                    continue
                }
                let closeTag = "</\(tag)>"
                let isHidden = Self.hiddenTags.contains(String(tag))
                if line.contains(closeTag) {
                    // Single-line tag block
                    if !isHidden { blocks.append(.xmlBlock(line)) }
                    i += 1
                    continue
                }
                // Multi-line: collect until closing tag (or EOF for unclosed)
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
                if !isHidden {
                    blocks.append(.xmlBlock(xmlLines.joined(separator: "\n")))
                }
                continue
            }

            // Standalone model tags like <pad><pad><pad>
            if line.hasPrefix("<") && line.contains(">") && !line.hasPrefix("<http") {
                let stripped = line.trimmingCharacters(in: .whitespaces)
                if stripped.range(of: "^(<[a-zA-Z_/]+>\\s*)+$", options: .regularExpression) != nil {
                    // Only treat as XML block if ALL tags are model tags
                    let tagNames = stripped.components(separatedBy: ">")
                        .compactMap { $0.components(separatedBy: "<").last?.replacingOccurrences(of: "/", with: "") }
                        .filter { !$0.isEmpty }
                    if tagNames.allSatisfy({ Self.modelTags.contains($0) }) {
                        blocks.append(.xmlBlock(stripped))
                        i += 1
                        continue
                    }
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

            // Markdown table: a `|`-leading row immediately followed by a separator
            // row (`|---|---|`, optionally with `:` for alignment). We accept the
            // looser "must have at least one `|`" form too — many models drop the
            // leading pipe — but require the separator line to confirm intent so
            // we don't misinterpret a stray pipe as a table header.
            if let table = Self.tryParseTable(lines: lines, start: i) {
                blocks.append(.table(table.headers, table.rows, table.alignments))
                i = table.end
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
                   next.hasPrefix("<") ||
                   next.trimmingCharacters(in: .whitespaces).hasPrefix("|") {
                    break
                }
                para.append(next)
                i += 1
            }
            blocks.append(.paragraph(para.joined(separator: "\n")))
        }

        return blocks
    }

    // MARK: Table parsing

    private struct ParsedTable {
        let headers: [String]
        let rows: [[String]]
        let alignments: [TableAlignment]
        let end: Int  // index of the line *after* the table
    }

    /// Detect a GFM-style markdown table starting at `lines[start]`. Requires a
    /// header row, a separator row of dashes (with optional colons for alignment),
    /// and zero-or-more data rows. Returns nil if any structural check fails so
    /// the caller falls through to paragraph handling.
    private static func tryParseTable(lines: [String], start: Int) -> ParsedTable? {
        guard start + 1 < lines.count else { return nil }
        let headerLine = lines[start].trimmingCharacters(in: .whitespaces)
        let sepLine = lines[start + 1].trimmingCharacters(in: .whitespaces)
        // First try the strict GFM form (pipes + dashed separator).
        if headerLine.contains("|"), isTableSeparator(sepLine) {
            let headers = parseTableRow(headerLine)
            let alignments = parseTableAlignments(sepLine)
            guard !headers.isEmpty else { return nil }
            var rows: [[String]] = []
            var i = start + 2
            while i < lines.count {
                let r = lines[i].trimmingCharacters(in: .whitespaces)
                guard r.contains("|") else { break }
                if isTableSeparator(r) { break }
                rows.append(parseTableRow(r))
                i += 1
            }
            return ParsedTable(headers: headers, rows: rows, alignments: alignments, end: i)
        }
        // Fallback: ASCII pseudo-table — many smaller models emit
        //   Header1   Header2   Header3
        //   ---------------------------
        //   value1    value2    value3
        // i.e. multi-space column separators + a single row of dashes. Detect
        // it by looking for a header line with at least two 2+-space gaps,
        // followed by a row that's mostly dashes, followed by data rows that
        // also have multi-space gaps. We split each row on `\s{2,}` to recover
        // cells.
        return tryParseAsciiPseudoTable(lines: lines, start: start)
    }

    /// Recognise the whitespace-aligned "table" shape smaller models emit when
    /// asked for tabular data without using GFM pipe syntax. We require a
    /// dashed-rule line within the next two lines and at least 3 columns in the
    /// header so we don't false-positive a paragraph that happens to contain a
    /// double space.
    private static func tryParseAsciiPseudoTable(lines: [String], start: Int) -> ParsedTable? {
        let header = lines[start]
        let headerCells = splitOnDoubleSpace(header)
        guard headerCells.count >= 2 else { return nil }
        // Find the separator line — typically immediately next, sometimes after
        // a blank line. Don't search far so paragraphs don't accidentally match.
        var sepIdx = start + 1
        while sepIdx < min(start + 3, lines.count) {
            let candidate = lines[sepIdx].trimmingCharacters(in: .whitespaces)
            if isAsciiRule(candidate) { break }
            if !candidate.isEmpty { return nil }
            sepIdx += 1
        }
        guard sepIdx < lines.count else { return nil }
        guard isAsciiRule(lines[sepIdx].trimmingCharacters(in: .whitespaces)) else { return nil }
        // Collect data rows: non-blank, with at least one 2+-space gap, and not
        // another rule line.
        var rows: [[String]] = []
        var i = sepIdx + 1
        while i < lines.count {
            let raw = lines[i]
            let t = raw.trimmingCharacters(in: .whitespaces)
            if t.isEmpty { i += 1; break }
            if isAsciiRule(t) { i += 1; break }
            let cells = splitOnDoubleSpace(raw)
            // Tolerate single-cell continuation lines (model wrapping a long
            // cell to the next line) by appending to the previous row's last
            // cell rather than starting a new row.
            if cells.count == 1, !rows.isEmpty {
                rows[rows.count - 1][rows[rows.count - 1].count - 1] += " " + cells[0]
            } else {
                rows.append(cells)
            }
            i += 1
        }
        guard !rows.isEmpty else { return nil }
        // All-left alignment (we have no `:---:` markers in this format).
        let alignments = [TableAlignment](repeating: .left, count: headerCells.count)
        return ParsedTable(headers: headerCells, rows: rows, alignments: alignments, end: i)
    }

    /// Split on runs of two-or-more whitespace. Trims each cell. Drops the
    /// empty leading element if the line was indented.
    private static func splitOnDoubleSpace(_ line: String) -> [String] {
        let parts = line.components(separatedBy: "  ")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        return parts
    }

    /// True if the (already-trimmed) line consists entirely of dashes / box-
    /// drawing chars / spaces and is at least 3 chars long. Catches the
    /// "----------" rule under header rows in pseudo-tables.
    private static func isAsciiRule(_ line: String) -> Bool {
        guard line.count >= 3 else { return false }
        let allowed: Set<Character> = ["-", "─", "=", " ", "|"]
        let allAllowed = line.allSatisfy { allowed.contains($0) }
        let hasDash = line.contains("-") || line.contains("─") || line.contains("=")
        return allAllowed && hasDash
    }

    private static func parseTableRow(_ line: String) -> [String] {
        var t = line.trimmingCharacters(in: .whitespaces)
        if t.hasPrefix("|") { t.removeFirst() }
        if t.hasSuffix("|") { t.removeLast() }
        return t.split(separator: "|", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
    }

    private static func isTableSeparator(_ line: String) -> Bool {
        let cells = parseTableRow(line)
        guard !cells.isEmpty else { return false }
        return cells.allSatisfy { cell in
            let c = cell.replacingOccurrences(of: " ", with: "")
            return c.range(of: "^:?-{3,}:?$", options: .regularExpression) != nil
        }
    }

    private static func parseTableAlignments(_ line: String) -> [TableAlignment] {
        return parseTableRow(line).map { cell in
            let c = cell.replacingOccurrences(of: " ", with: "")
            let leftColon = c.hasPrefix(":")
            let rightColon = c.hasSuffix(":")
            if leftColon && rightColon { return .center }
            if rightColon { return .right }
            return .left
        }
    }

    // MARK: NSAttributedString assembly

    /// Build the NSAttributedString fed to NSTextView. Public-static so the
    /// rendering path can be exercised by tests later if needed.
    static func attributedString(for source: String) -> NSAttributedString {
        let result = NSMutableAttributedString()
        let blocks = parseBlocks(source: source)
        for (idx, block) in blocks.enumerated() {
            if idx > 0 { result.append(blockSpacer()) }
            switch block {
            case .paragraph(let text):
                result.append(renderInline(text))

            case .heading(let level, let text):
                let size: CGFloat = level == 1 ? 18 : level == 2 ? 16 : 14
                let p = NSMutableParagraphStyle()
                p.paragraphSpacingBefore = 4
                p.paragraphSpacing = 2
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: NSFont.systemFont(ofSize: size, weight: .bold),
                    .foregroundColor: NSColor.labelColor,
                    .paragraphStyle: p,
                ]
                result.append(NSAttributedString(string: text, attributes: attrs))

            case .code(_, let content):
                let p = NSMutableParagraphStyle()
                p.paragraphSpacingBefore = 4
                p.paragraphSpacing = 4
                p.firstLineHeadIndent = 8
                p.headIndent = 8
                p.tailIndent = -8
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: NSFont.monospacedSystemFont(ofSize: 12, weight: .regular),
                    .backgroundColor: NSColor.textBackgroundColor.blended(withFraction: 0.85, of: .black) ?? NSColor.darkGray,
                    .foregroundColor: NSColor(white: 0.92, alpha: 1.0),
                    .paragraphStyle: p,
                ]
                result.append(NSAttributedString(string: content, attributes: attrs))

            case .listItem(let text):
                let bullet = NSAttributedString(string: "• ", attributes: [
                    .font: NSFont.systemFont(ofSize: 13),
                    .foregroundColor: NSColor.secondaryLabelColor,
                ])
                let p = NSMutableParagraphStyle()
                p.headIndent = 14
                let inline = renderInline(text)
                let combined = NSMutableAttributedString()
                combined.append(bullet)
                combined.append(inline)
                combined.addAttribute(.paragraphStyle, value: p, range: NSRange(location: 0, length: combined.length))
                result.append(combined)

            case .xmlBlock(let content):
                let p = NSMutableParagraphStyle()
                p.firstLineHeadIndent = 8
                p.headIndent = 8
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .regular),
                    .foregroundColor: NSColor.systemPurple,
                    .backgroundColor: NSColor.systemPurple.withAlphaComponent(0.10),
                    .paragraphStyle: p,
                ]
                result.append(NSAttributedString(string: content, attributes: attrs))

            case .table(let headers, let rows, let alignments):
                result.append(renderTable(headers: headers, rows: rows, alignments: alignments))
            }
        }
        return result
    }

    /// One-and-a-half blank lines between blocks. Encoded as a `\n` with extra
    /// paragraph spacing so tall blocks don't collapse.
    private static func blockSpacer() -> NSAttributedString {
        let p = NSMutableParagraphStyle()
        p.paragraphSpacing = 6
        return NSAttributedString(string: "\n", attributes: [
            .font: NSFont.systemFont(ofSize: 6),
            .paragraphStyle: p,
        ])
    }

    /// Render an inline span by delegating to AttributedString's markdown parser
    /// (handles `**bold**`, `_italic_`, `` `code` ``, `[link](url)`). Falls back
    /// to a plain-text NSAttributedString if the parse fails. Returned string
    /// carries the body font and a dynamic foreground color so the rendering
    /// flips correctly between light and dark modes — Foundation's converter
    /// can leave `**bold**` and link spans with a baked-in `NSColor` that
    /// doesn't adapt, so we overwrite missing-or-static colors with
    /// `.labelColor` (links keep their dynamic `linkColor`).
    private static func renderInline(_ text: String) -> NSAttributedString {
        let bodyFont = NSFont.systemFont(ofSize: 13)
        let result: NSMutableAttributedString
        if let attr = try? AttributedString(
            markdown: text,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        ) {
            result = NSMutableAttributedString(attr)
        } else {
            result = NSMutableAttributedString(string: text)
        }
        let full = NSRange(location: 0, length: result.length)
        // Default font for any character that didn't pick up an explicit font
        // from the markdown parser.
        result.enumerateAttribute(.font, in: full, options: []) { value, range, _ in
            if value == nil {
                result.addAttribute(.font, value: bodyFont, range: range)
            }
        }
        // Force a dynamic foreground for every non-link span. AttributedString's
        // markdown→NSAttributedString bridge sometimes inserts `NSColor.black`
        // for bold/italic — that reads fine in light mode but is invisible on
        // a dark bubble background. Walk the whole string and replace any
        // foreground that's NOT explicitly the dynamic linkColor with
        // labelColor (which adapts).
        result.enumerateAttribute(.foregroundColor, in: full, options: []) { value, range, _ in
            // Spans inside a link keep linkColor; everything else gets labelColor.
            let isLink = result.attribute(.link, at: range.location, effectiveRange: nil) != nil
            if isLink {
                result.addAttribute(.foregroundColor, value: NSColor.linkColor, range: range)
                return
            }
            // If the existing color is already dynamic-equal-to-labelColor we
            // can leave it; checking via `==` handles both the missing case
            // (value nil) and the static-black case Foundation often picks.
            if let existing = value as? NSColor,
               existing.isEqual(NSColor.labelColor) {
                return
            }
            result.addAttribute(.foregroundColor, value: NSColor.labelColor, range: range)
        }
        return result
    }

    /// Render a markdown table as monospaced columns padded to the widest cell
    /// per column. Header row gets a bold font; a horizontal rule separates the
    /// header from the data rows. Looks great in a chat bubble and stays
    /// selectable as part of the surrounding text.
    private static func renderTable(
        headers: [String],
        rows: [[String]],
        alignments: [TableAlignment]
    ) -> NSAttributedString {
        let cols = headers.count
        var widths = [Int](repeating: 0, count: cols)
        let allRows = [headers] + rows
        for row in allRows {
            for (j, cell) in row.prefix(cols).enumerated() {
                widths[j] = max(widths[j], cell.count)
            }
        }
        // Pad cells with at least 1 space so columns don't visually merge.
        for j in 0..<cols { widths[j] = max(widths[j], 1) }

        func pad(_ cell: String, width: Int, align: TableAlignment) -> String {
            let gap = width - cell.count
            if gap <= 0 { return cell }
            switch align {
            case .left:   return cell + String(repeating: " ", count: gap)
            case .right:  return String(repeating: " ", count: gap) + cell
            case .center:
                let l = gap / 2
                return String(repeating: " ", count: l) + cell + String(repeating: " ", count: gap - l)
            }
        }

        func formatRow(_ cells: [String]) -> String {
            var padded = cells
            while padded.count < cols { padded.append("") }
            return padded.prefix(cols).enumerated().map { idx, cell in
                let a = idx < alignments.count ? alignments[idx] : .left
                return pad(cell, width: widths[idx], align: a)
            }.joined(separator: "  ")
        }

        let mono = NSFont.monospacedSystemFont(ofSize: 12, weight: .regular)
        let monoBold = NSFont.monospacedSystemFont(ofSize: 12, weight: .semibold)
        let result = NSMutableAttributedString()

        // Header row (bold) + horizontal rule using box-drawing chars. Explicit
        // `.foregroundColor: .labelColor` so the table flips light/dark with
        // the system mode — without it some macOS versions render the cells
        // in the captured static color from the AttributedString bridge.
        let headerLine = formatRow(headers) + "\n"
        result.append(NSAttributedString(string: headerLine, attributes: [
            .font: monoBold,
            .foregroundColor: NSColor.labelColor,
        ]))
        let rule = widths.map { String(repeating: "─", count: $0) }.joined(separator: "  ") + "\n"
        result.append(NSAttributedString(string: rule, attributes: [
            .font: mono,
            .foregroundColor: NSColor.secondaryLabelColor,
        ]))
        // Data rows.
        for (idx, row) in rows.enumerated() {
            let line = formatRow(row) + (idx == rows.count - 1 ? "" : "\n")
            result.append(NSAttributedString(string: line, attributes: [
                .font: mono,
                .foregroundColor: NSColor.labelColor,
            ]))
        }
        return result
    }
}

// MARK: - SelectableMarkdownNSText (NSTextView wrapper)

/// NSViewRepresentable around an NSTextView. NSTextView is the only AppKit text
/// surface that natively supports drag-selection across an arbitrarily styled
/// attributed string, which is what we need so users can highlight an entire
/// assistant message — paragraphs, list items, code blocks, tables — in one
/// motion and copy the lot. The view reports its intrinsic content size to
/// SwiftUI so layout in a VStack works without forcing a fixed height.
fileprivate struct SelectableMarkdownNSText: NSViewRepresentable {
    let attributed: NSAttributedString

    func makeNSView(context: Context) -> IntrinsicTextView {
        let tv = IntrinsicTextView()
        tv.isEditable = false
        tv.isSelectable = true
        tv.drawsBackground = false
        tv.textContainerInset = .zero
        tv.textContainer?.lineFragmentPadding = 0
        tv.textContainer?.widthTracksTextView = true
        tv.isVerticallyResizable = true
        tv.isHorizontallyResizable = false
        tv.autoresizingMask = [.width]
        // Match the surrounding bubble's text color when no explicit foreground
        // is set on a span (e.g. plain paragraphs).
        tv.textColor = .labelColor
        tv.linkTextAttributes = [
            .foregroundColor: NSColor.linkColor,
            .underlineStyle: NSUnderlineStyle.single.rawValue,
            .cursor: NSCursor.pointingHand,
        ]
        tv.textStorage?.setAttributedString(attributed)
        return tv
    }

    func updateNSView(_ nsView: IntrinsicTextView, context: Context) {
        // Only mutate the storage if the assistant's content actually changed.
        // Streaming chunks call updateNSView many times per second; an unconditional
        // replace would interrupt an active selection on every frame.
        if nsView.textStorage?.isEqual(to: attributed) == false {
            nsView.textStorage?.setAttributedString(attributed)
            nsView.invalidateIntrinsicContentSize()
        }
    }
}

/// NSTextView that reports its laid-out height as its intrinsic content size,
/// so embedding it in SwiftUI's layout system "just works" — no manual height
/// binding required.
fileprivate final class IntrinsicTextView: NSTextView {
    override var intrinsicContentSize: NSSize {
        guard let lm = layoutManager, let tc = textContainer else {
            return super.intrinsicContentSize
        }
        lm.ensureLayout(for: tc)
        let used = lm.usedRect(for: tc)
        return NSSize(width: NSView.noIntrinsicMetric, height: ceil(used.height))
    }

    override func didChangeText() {
        super.didChangeText()
        invalidateIntrinsicContentSize()
    }

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        // Width changes (parent re-flow) require a layout-driven height re-check.
        invalidateIntrinsicContentSize()
    }
}
