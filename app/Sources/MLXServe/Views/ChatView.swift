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
struct ToolApprovalSheet: View {
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
        case "generate_image": return "Generate an image"
        case "generate_audio": return "Generate audio"
        case "generate_video": return "Generate a video"
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

/// Horizontal strip of pending attachment chips (images, PDFs, audio) shown
/// above the message input. Extracted from `ChatDetailView` so its body stays
/// within the Swift type-checker's complexity budget.
private struct AttachmentPreviewRow: View {
    @Binding var images: [NSImage]
    @Binding var pdfs: [(name: String, text: String)]
    @Binding var audio: [ChatAudio]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(Array(images.enumerated()), id: \.offset) { idx, img in
                    imageChip(idx: idx, img: img)
                }
                ForEach(Array(pdfs.enumerated()), id: \.offset) { idx, pdf in
                    fileChip(idx: idx, name: pdf.name, detail: "PDF · \(pdf.text.count) chars",
                             icon: "doc.text.fill", tint: .red) { pdfs.remove(at: idx) }
                }
                ForEach(Array(audio.enumerated()), id: \.offset) { idx, clip in
                    fileChip(idx: idx, name: clip.name, detail: String(format: "Audio · %.1fs", clip.durationSeconds),
                             icon: "waveform", tint: .purple) { audio.remove(at: idx) }
                }
            }
        }
        .frame(height: 64)
    }

    @ViewBuilder
    private func removeButton(_ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: "xmark.circle.fill")
                .font(.system(size: 14))
                .foregroundStyle(.white)
                .background(Circle().fill(.black.opacity(0.5)))
        }
        .buttonStyle(.plain)
        .offset(x: 4, y: -4)
    }

    @ViewBuilder
    private func imageChip(idx: Int, img: NSImage) -> some View {
        ZStack(alignment: .topTrailing) {
            Image(nsImage: img)
                .resizable()
                .aspectRatio(contentMode: .fill)
                .frame(width: 56, height: 56)
                .clipShape(RoundedRectangle(cornerRadius: 8))
            removeButton { images.remove(at: idx) }
        }
    }

    @ViewBuilder
    private func fileChip(idx: Int, name: String, detail: String, icon: String, tint: Color, remove: @escaping () -> Void) -> some View {
        ZStack(alignment: .topTrailing) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 18))
                    .foregroundStyle(.white)
                    .frame(width: 32, height: 32)
                    .background(tint.opacity(0.85))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                VStack(alignment: .leading, spacing: 1) {
                    Text(name)
                        .font(.caption.weight(.medium))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(detail)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .frame(maxWidth: 200, minHeight: 56, maxHeight: 56)
            .background(Color.secondary.opacity(0.15))
            .clipShape(RoundedRectangle(cornerRadius: 10))
            removeButton(remove)
        }
    }
}

/// Chip for the attached document folder (mini RAG): shows live indexing
/// progress, then the indexed file/chunk totals, with an ✕ to detach. Styled
/// to match the `AttachmentPreviewRow` file chips.
private struct DocumentFolderChip: View {
    @ObservedObject var index: DocumentIndex
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: iconName)
                .font(.system(size: 18))
                .foregroundStyle(.white)
                .frame(width: 32, height: 32)
                .background(tint.opacity(0.85))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            VStack(alignment: .leading, spacing: 1) {
                Text(index.folderName)
                    .font(.caption.weight(.medium))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(statusText)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            if case .indexing(let done, let total) = index.state {
                ProgressView(value: total > 0 ? Double(done) / Double(total) : 0)
                    .progressViewStyle(.linear)
                    .frame(width: 70)
            }
            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 14))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Detach folder")
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .frame(maxWidth: 320, minHeight: 44, alignment: .leading)
        .background(Color.secondary.opacity(0.15))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var iconName: String {
        if case .failed = index.state { return "exclamationmark.triangle.fill" }
        return "folder.fill"
    }

    private var tint: Color {
        if case .failed = index.state { return .orange }
        return .blue
    }

    private var statusText: String {
        switch index.state {
        case .preparing:
            return "Preparing embeddings…"
        case .indexing(let done, let total):
            return total > 0 ? "Indexing \(done)/\(total) files…" : "Scanning folder…"
        case .ready(let files, let chunks):
            return "\(files) files · \(chunks) excerpts — ask away"
        case .failed(let msg):
            return msg
        }
    }
}

/// Record-audio button shown next to the paperclip on audio-capable models.
/// Tap to start (mic icon), tap again to stop (red pill with elapsed time).
private struct MicButton: View {
    @ObservedObject var recorder: AudioRecorder
    let toggle: () -> Void

    var body: some View {
        Button(action: toggle) {
            HStack(spacing: 4) {
                Image(systemName: recorder.isRecording ? "stop.fill" : "mic.fill")
                    .font(.system(size: 12, weight: .medium))
                if recorder.isRecording {
                    Text(timeString(recorder.duration))
                        .font(.caption2.monospacedDigit().weight(.medium))
                }
            }
            .foregroundStyle(recorder.isRecording ? Color.white : Color.secondary)
            .frame(minWidth: ChatMetrics.composerIconSize,
                   minHeight: ChatMetrics.composerIconSize, maxHeight: ChatMetrics.composerIconSize)
            .padding(.horizontal, recorder.isRecording ? 8 : 0)
            .background(recorder.isRecording ? Color.red : Color.secondary.opacity(0.15))
            .clipShape(Capsule())
            .overlay(alignment: .leading) {
                if recorder.isRecording {
                    Circle().fill(Color.white.opacity(0.9))
                        .frame(width: 5, height: 5)
                        .scaleEffect(0.6 + 0.4 * CGFloat(recorder.level))
                        .padding(.leading, 3)
                        .allowsHitTesting(false)
                }
            }
            // Same full-height frame as the attach/send controls so the
            // bottom-aligned composer row centers everything against the pill.
            .frame(minWidth: ChatMetrics.composerControlSize, minHeight: ChatMetrics.composerControlSize)
        }
        .buttonStyle(.plain)
        .help(recorder.isRecording ? "Stop recording and attach" : "Record audio for the model to hear")
    }

    private func timeString(_ t: TimeInterval) -> String {
        let s = Int(t)
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}

/// What a pasted/dropped file URL should become, by extension + directory flag.
/// A top-level (non-`@MainActor`) type so the routing is unit-testable without
/// the rendered view — it mirrors the attach button's dispatch (see ChatPasteTests).
enum PasteFileKind: String, Equatable {
    case folder, pdf, audio, image, unhandled

    static func classify(ext: String, isDirectory: Bool, audioSupported: Bool) -> PasteFileKind {
        if isDirectory { return .folder }
        let e = ext.lowercased()
        if e == "pdf" { return .pdf }
        if let ut = UTType(filenameExtension: e) {
            if ut.conforms(to: .audio) { return audioSupported ? .audio : .unhandled }
            if ut.conforms(to: .image) { return .image }
        }
        return .unhandled
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
            ForEach(appState.visibleChatSessions) { session in
                let isSelected = session.id == appState.activeChatId
                HStack(spacing: 0) {
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 4) {
                            if session.isExternalBridge {
                                Image(systemName: "paperplane.fill")
                                    .font(.system(size: 9))
                                    .foregroundStyle(isSelected ? Color.white.opacity(0.85) : Color.accentColor)
                                    .help("Telegram conversation (view only)")
                            }
                            Text(session.title)
                                .font(.subheadline.weight(isSelected ? .semibold : .regular))
                                .lineLimit(1)
                                .foregroundStyle(isSelected ? .white : .primary)
                        }
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
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
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
    @EnvironmentObject var chatEngine: ChatTurnEngine
    @Environment(\.openWindow) private var openWindow
    @State private var inputText = ""
    // The three toolbar toggles mirror the visible session's persisted state
    // (`ChatSession.enableThinking` / `.mode` / `.useMCP`). They're loaded from
    // the session on appear AND on every `sessionId` change, and written back on
    // toggle — so each chat tab remembers its own Think/Agent/MCP choice instead
    // of leaking the active tab's value into the reused ChatDetailView.
    @State private var enableThinking = false
    @State private var isAgentMode = false
    @State private var mcpMode = false
    @State private var showMCPMarketplace = false
    @State private var showVoiceMode = false
    @State private var showThinkingInAgentConfirm = false
    @State private var executingPlanMessageId: UUID?
    @State private var isNearBottom = true
    @State private var scrollViewHeight: CGFloat = 0
    @State private var contentBottom: CGFloat = 0
    @State private var scrollMonitor: Any?
    @State private var pasteMonitor: Any?
    @State private var pendingImages: [NSImage] = []
    @State private var pendingPDFs: [(name: String, text: String)] = []
    @State private var pendingAudio: [ChatAudio] = []
    @StateObject private var recorder = AudioRecorder()
    // Tool-approval gate state. `pendingApproval` is set right before each
    // tool call when Agent mode is on; the sheet at the bottom of `body`
    // observes it and resumes `approvalContinuation` with the user's choice.
    // `toolAllowList` is the soft "Allow all tools this session" decision, keyed
    // by session id so it is remembered PER TAB: SwiftUI reuses this view across
    // `sessionId` changes, so a per-session set survives switching tabs (a plain
    // Bool was shared across tabs and got wiped on every switch). A session
    // re-arms only when the user toggles Agent off in that tab.
    @State private var pendingApproval: ToolApprovalRequest?
    @State private var toolAllowList = SessionToolAllowList()
    // Plain Bool (not @FocusState): the composer is an NSTextView wrapper, so
    // AppKit first-responder is the source of truth and GrowingTextEditor mirrors
    // it back into this flag. The Cmd+V attach monitor reads it; on-appear and
    // post-generation code set it true to (re)focus the field.
    @State private var inputFocused = false
    @State private var composerHeight: CGFloat = 36
    /// Measured width of this chat pane — drives the toolbar pill density
    /// (full captions vs icon-only) via `ChatMetrics.useCompactToggles`.
    @State private var detailWidth: CGFloat = 0
    // Pre-send intent nudge: when a message looks agentic / MCP-bound but the
    // matching mode is off, confirm before sending. `intentSuppress` remembers a
    // per-session "Send anyway" so we stop nagging that chat (keyed by session
    // id — the view is reused across tabs).
    @State private var pendingIntentPrompt: IntentPrompt?
    @State private var intentSuppress = SessionIntentSuppression()


    private var session: ChatSession? {
        appState.chatSessions.first { $0.id == sessionId }
    }

    /// Generation state for THIS chat. The engine runs one turn at a time, so a
    /// chat that doesn't own the active turn must show Send (idle), not the Stop
    /// button — and its Send is disabled while another chat is mid-turn.
    private var composerState: ChatTurnEngine.ComposerState {
        chatEngine.composerState(for: sessionId)
    }

    /// Pull the toolbar toggles from the visible session into local @State.
    /// Called on appear and on every `sessionId` change — the view is reused
    /// across tabs, so without this the toggles would show the previous tab's
    /// values. Telegram sessions read the shared config instead (see
    /// `toolbarToggles`), so they don't sync here.
    private func syncTogglesFromSession() {
        guard !isExternalBridgeSession else { return }
        isAgentMode = session?.mode == .agent
        enableThinking = session?.enableThinking ?? false
        mcpMode = session?.useMCP ?? false
    }

    /// True when the visible session mirrors a Telegram conversation. The
    /// think/agent/MCP toolbar toggles then read & write the shared
    /// `serverOptions.telegram` config (kept in sync with Settings, read live by
    /// the bridge) instead of the in-app per-session / app-level state.
    private var isExternalBridgeSession: Bool { session?.isExternalBridge == true }

    /// Resolved on/off state for the three mode toggles in the toolbar — sourced
    /// from `serverOptions.telegram` for a Telegram session, else the in-app state.
    private var toolbarToggles: ChatModeToggles {
        let tg = appState.serverOptions.telegram
        return ChatModeToggles.resolve(
            isExternalBridge: isExternalBridgeSession,
            telegramThinking: tg.enableThinking, telegramAgent: tg.agentMode, telegramMCP: tg.useMCP,
            inAppThinking: enableThinking, inAppAgent: isAgentMode, inAppMCP: mcpMode)
    }

    // MARK: Toolbar mode pills (Think / Agent / MCP)
    //
    // All three share ChatMetrics' pill geometry — an EXPLICIT height and icon
    // slot, because SF symbols at the same point size have different intrinsic
    // sizes (brain vs wrench vs puzzle) and padding-derived capsules came out
    // subtly unequal. Rendered inside the chatHeaderBar strip.

    /// The chat pane's header control strip: sandbox badge + working-dir chip
    /// (agent mode), voice, settings, the mode pills, and the server dot.
    /// Rendered as OUR OWN row (safeAreaInset), not NSToolbar items — see the
    /// call site for why.
    @ViewBuilder
    private var chatHeaderBar: some View {
        HStack(spacing: 10) {
            Spacer(minLength: 0)
            // Sandbox badge: visible when tools are live (Agent or MCP mode)
            // AND the Agent Sandbox is on — the signal that shell commands
            // run isolated from this Mac. No badge when tools run on the
            // host (absence = not sandboxed; never imply safety we don't have).
            if (isAgentMode || mcpMode) && appState.serverOptions.sandbox.enabled {
                Button {
                    openWindow(id: "sandboxTerminal")
                } label: {
                    Image(systemName: "lock.shield.fill")
                        .font(.system(size: 12))
                        .foregroundStyle(.green)
                }
                .buttonStyle(.borderless)
                .help("Secured by the Agent Sandbox — the agent's shell commands run inside an isolated Linux VM that can only touch the working folder. Click to open the Sandbox Terminal.")
            }
            if isAgentMode {
                WorkingDirectoryIndicator(path: workingDirectoryBinding)
            }
            Button {
                startVoiceMode()
            } label: {
                Image(systemName: "waveform")
                    .font(.system(size: 12))
                    .foregroundStyle(server.status == .running ? .primary : .tertiary)
            }
            .buttonStyle(.borderless)
            .disabled(server.status != .running)
            .help("Voice mode — talk to the model hands-free. Speech-to-text and text-to-speech run locally on your Mac; the model only handles text (and tools/thinking if enabled).")
            Button {
                openWindow(id: "settings")
            } label: {
                Image(systemName: "gear")
                    .font(.system(size: 12))
                    .foregroundStyle(.primary)
            }
            .buttonStyle(.borderless)
            .help("Open Settings (⌘,) — server flags, speculative decoding, performance (continuous batching, KV-quant, prefix cache), and per-request defaults.")
            // Adaptive pills: full when the chat pane is roomy, icon-only when
            // narrow (MCP drops its gear) — decided from the measured pane width.
            modePillCluster(compact: ChatMetrics.useCompactToggles(forDetailWidth: detailWidth))
            Circle()
                .fill(server.status == .running ? .green : .red)
                .frame(width: 8, height: 8)
                .help(server.status == .running ? "Server running" : "Server stopped")
        }
        .padding(.horizontal, ChatMetrics.gutter)
        .padding(.vertical, 8)
        .background(.bar)
        .overlay(alignment: .bottom) { Divider() }
    }

    /// The whole cluster at one density — chosen by the measured pane width
    /// (see `chatHeaderBar`).
    @ViewBuilder
    private func modePillCluster(compact: Bool) -> some View {
        HStack(spacing: ChatMetrics.togglePillSpacing) {
            thinkToggle(compact: compact)
            agentToggle(compact: compact)
            mcpToggle(compact: compact)
        }
        .fixedSize()
    }

    /// Shared pill label: fixed icon slot (+ caption when not compact) inside
    /// the pinned height.
    @ViewBuilder
    private func pillLabel(icon: String, title: String, isOn: Bool, onColor: Color, compact: Bool) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .medium))
                .frame(width: ChatMetrics.togglePillIconSize, height: ChatMetrics.togglePillIconSize)
            if !compact {
                Text(title)
                    .font(.caption.weight(.medium))
            }
        }
        .foregroundStyle(isOn ? .white : .secondary)
        .padding(.horizontal, ChatMetrics.togglePillPaddingH)
        .frame(height: ChatMetrics.togglePillHeight)
        .background(isOn ? onColor : Color.secondary.opacity(0.12))
        .clipShape(Capsule())
        // Whole capsule is clickable — a .plain Button without an explicit
        // content shape only hit-tests the drawn pixels.
        .contentShape(Capsule())
    }

    private func thinkToggle(compact: Bool) -> some View {
        Button {
            if isExternalBridgeSession {
                // Telegram session: write the shared config so the toggle
                // stays in sync with Settings and the bridge reads it live.
                appState.serverOptions.telegram.enableThinking.toggle()
            } else if !enableThinking && isAgentMode {
                showThinkingInAgentConfirm = true
            } else {
                enableThinking.toggle()
            }
        } label: {
            pillLabel(icon: "brain", title: "Think", isOn: toolbarToggles.thinking, onColor: .blue, compact: compact)
        }
        .buttonStyle(.plain)
        .help("Thinking Mode (\(toolbarToggles.thinking ? "ON" : "OFF")) — when the model supports it, it'll emit a private reasoning trace before the visible answer. Slower but better on reasoning-heavy prompts.")
    }

    private func agentToggle(compact: Bool) -> some View {
        Button {
            if isExternalBridgeSession {
                // Telegram session: flip the shared config (in sync with
                // Settings); no per-session approval state applies here.
                appState.serverOptions.telegram.agentMode.toggle()
            } else {
                isAgentMode.toggle()
                // Re-arm the approval gate every time the user re-enters
                // Agent mode. "Always allow this session" decays here —
                // for THIS tab only; other tabs keep their decision.
                if !isAgentMode { toolAllowList.rearm(sessionId) }
                // Thinking + tool-calling loops degrade quality on most
                // local models — auto-off when entering Agent mode.
                if isAgentMode { enableThinking = false }
            }
        } label: {
            pillLabel(icon: "wrench", title: "Agent", isOn: toolbarToggles.agent, onColor: .orange, compact: compact)
        }
        .buttonStyle(.plain)
        .help("""
        Agent Mode (\(toolbarToggles.agent ? "ON" : "OFF")) — the model runs a tool-calling loop with these 10 built-in tools:
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

    private func mcpToggle(compact: Bool) -> some View {
        HStack(spacing: 0) {
            Button {
                if isExternalBridgeSession {
                    // Telegram session: flip the shared config (synced with Settings).
                    appState.serverOptions.telegram.useMCP.toggle()
                } else {
                    mcpMode.toggle()
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "puzzlepiece.extension")
                        .font(.system(size: 11, weight: .medium))
                        .frame(width: ChatMetrics.togglePillIconSize, height: ChatMetrics.togglePillIconSize)
                    if !compact {
                        Text("MCP")
                            .font(.caption.weight(.medium))
                    }
                }
                .foregroundStyle(toolbarToggles.mcp ? .white : .secondary)
                .padding(.leading, ChatMetrics.togglePillPaddingH)
                // Compact drops the gear half, so the toggle carries the
                // trailing padding itself to stay a symmetric capsule.
                .padding(.trailing, compact ? ChatMetrics.togglePillPaddingH : 0)
                .frame(height: ChatMetrics.togglePillHeight)
                // The capsule background lives on the OUTER HStack, so this
                // button's label is transparent — without an explicit content
                // shape only the glyph/text pixels hit-test, which is why the
                // MCP toggle "didn't always click".
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if !compact {
                Button { showMCPMarketplace = true } label: {
                    Image(systemName: "gearshape.fill")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(toolbarToggles.mcp ? .white.opacity(0.85) : .secondary)
                        .padding(.trailing, ChatMetrics.togglePillPaddingH)
                        .padding(.leading, 4)
                        .frame(height: ChatMetrics.togglePillHeight)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Open MCP Marketplace — browse and enable Model Context Protocol servers (GitHub, Filesystem, Slack, Notion, Playwright, Docker, etc.). Each enabled server's tools become callable in Agent Mode.")
            }
        }
        .frame(height: ChatMetrics.togglePillHeight)
        .background(toolbarToggles.mcp ? .purple : Color.secondary.opacity(0.12))
        .clipShape(Capsule())
        .help("MCP Mode (\(toolbarToggles.mcp ? "ON" : "OFF")) — when on, tools from every enabled Model Context Protocol server are added to the Agent's toolset (alongside the 10 built-ins). Tap the gear to open the Marketplace and toggle servers on/off.")
    }

    var body: some View {
        VStack(spacing: 0) {
            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(ChatRowBuilder.rows(from: session?.messages ?? [])) { row in
                            switch row {
                            case .message(let m):
                                MessageBubble(message: m).id(m.id)
                            case .toolCall(let call, let results):
                                ToolCallRow(call: call, results: results).id(call.id)
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
                    .padding(ChatMetrics.gutter)
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

            // Context usage monitor — shows once a turn has reported usage, or
            // live the moment this chat starts generating its first reply.
            if contextUsage != nil || composerState == .generatingHere {
                ContextMonitor(promptTokens: contextUsage?.promptTokens ?? 0,
                               completionTokens: contextUsage?.completionTokens ?? 0,
                               liveTokens: composerState == .generatingHere ? chatEngine.liveCompletionTokens : 0,
                               isLive: composerState == .generatingHere,
                               contextLength: contextUsage?.contextLength
                                   ?? AgentEngine.effectiveContextLength(appContextSize: appState.contextSize,
                                                                         modelContextLength: server.modelInfo?.contextLength))
            }

            // Input area — iMessage style
            VStack(spacing: 4) {
              if session?.isExternalBridge == true {
                // Telegram bridge sessions mirror a phone conversation and are
                // read-only on the Mac: a Telegram bot can only post as itself,
                // so there's no coherent way to inject a Mac-typed user turn.
                // Reply from the phone; the mirror updates live here.
                HStack(spacing: 8) {
                    Image(systemName: "paperplane.fill")
                        .foregroundStyle(.secondary)
                    Text("Telegram conversation — view only. Reply from your phone.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
              } else {
                // Pending attachment thumbnails (images + PDFs + audio)
                if !pendingImages.isEmpty || !pendingPDFs.isEmpty || !pendingAudio.isEmpty {
                    AttachmentPreviewRow(images: $pendingImages, pdfs: $pendingPDFs, audio: $pendingAudio)
                }

                // Attached document folder (mini RAG) — indexing progress / ready chip
                if let docIndex = appState.documentIndexes[sessionId] {
                    DocumentFolderChip(index: docIndex) {
                        docIndex.cancel()
                        appState.documentIndexes.removeValue(forKey: sessionId)
                    }
                }

                HStack(alignment: .bottom, spacing: 8) {
                    // Attachment menu (images/PDFs/audio + document folder)
                    Menu {
                        Button {
                            pickAttachment()
                        } label: {
                            Label(audioSupported ? "Attach Image, PDF, or Audio…" : "Attach Image or PDF…",
                                  systemImage: "photo.on.rectangle")
                        }
                        Button {
                            pickDocumentFolder()
                        } label: {
                            Label("Attach Folder for Q&A…", systemImage: "folder.badge.questionmark")
                        }
                    } label: {
                        Image(systemName: "paperclip")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(.secondary)
                            .frame(width: ChatMetrics.composerIconSize, height: ChatMetrics.composerIconSize)
                            .background(Color.secondary.opacity(0.15))
                            .clipShape(Circle())
                    }
                    // .plain button style (not .borderlessButton menu style) —
                    // the latter substitutes its own chrome on macOS, dropping
                    // the circle background and mis-baselining the glyph next
                    // to the input pill.
                    .menuStyle(.button)
                    .buttonStyle(.plain)
                    .menuIndicator(.hidden)
                    // Full control frame == the pill's resting height, so the
                    // bottom-aligned row centers the circle against the pill
                    // with no nudge padding (ChatMetrics contract).
                    .frame(width: ChatMetrics.composerControlSize, height: ChatMetrics.composerControlSize)
                    .help("Attach files, or a document folder to ask questions about")

                    // Mic — only on models that actually understand audio
                    // (Gemma 4 12B). Tap to record, tap again to attach.
                    if audioSupported {
                        MicButton(recorder: recorder) { toggleRecording() }
                            .disabled(server.status != .running || chatEngine.isGenerating)
                    }

                    // Dark pill input — NSTextView-backed so a big paste stays
                    // smooth and the mouse wheel scrolls once it grows past the cap.
                    GrowingTextEditor(text: $inputText,
                                      isFocused: $inputFocused,
                                      measuredHeight: $composerHeight,
                                      isIdle: composerState == .idle,
                                      onSend: { sendMessage() })
                        .frame(height: max(ChatMetrics.composerMinHeight, composerHeight))
                        .padding(.horizontal, 5)
                        .disabled(server.status != .running)
                        .background(Color(nsColor: .textBackgroundColor))
                        .clipShape(RoundedRectangle(cornerRadius: 20))
                        .overlay(alignment: .topLeading) {
                            if inputText.isEmpty {
                                Text("Message")
                                    .font(.body)
                                    .foregroundStyle(.secondary)
                                    .padding(.leading, 14)
                                    .padding(.top, 8)
                                    .allowsHitTesting(false)
                            }
                        }
                        .overlay(
                            RoundedRectangle(cornerRadius: 20)
                                .stroke(Color.secondary.opacity(0.25), lineWidth: 0.5)
                        )

                    Button {
                        if composerState == .generatingHere {
                            chatEngine.stop()
                        } else {
                            sendMessage()
                        }
                    } label: {
                        Image(systemName: composerState == .generatingHere ? "stop.circle.fill" : "arrow.up.circle.fill")
                            .font(.system(size: ChatMetrics.composerIconSize))
                            .foregroundStyle(composerState == .generatingHere ? .red : .accentColor)
                            .frame(width: ChatMetrics.composerControlSize, height: ChatMetrics.composerControlSize)
                    }
                    .buttonStyle(.plain)
                    // Stop is always tappable for the owning chat. Otherwise: Send,
                    // disabled when the server is down, when this chat has nothing
                    // to send, or while ANOTHER chat is mid-turn (the single-turn
                    // engine can't start a concurrent run — so don't dead-click).
                    .disabled(server.status != .running
                              || composerState == .busyElsewhere
                              || (composerState == .idle
                                  && inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                  && pendingImages.isEmpty && pendingPDFs.isEmpty && pendingAudio.isEmpty))
                }
              }   // end else (non-Telegram composer)
            }
            .padding(.horizontal, ChatMetrics.gutter)
            .padding(.vertical, 8)
        }
        // Track the pane's width for the toolbar pill density decision.
        .background(
            GeometryReader { geo in
                Color.clear
                    .onAppear { detailWidth = geo.size.width }
                    .onChange(of: geo.size.width) { _, w in detailWidth = w }
            }
        )
        .onDrop(of: [.image, .pdf, .audio], isTargeted: nil) { providers in
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
                } else if audioSupported, provider.hasItemConformingToTypeIdentifier(UTType.audio.identifier) {
                    // Decode inside the closure — the temp URL is only valid here.
                    provider.loadFileRepresentation(forTypeIdentifier: UTType.audio.identifier) { url, _ in
                        guard let url = url else { return }
                        let name = url.lastPathComponent
                        let pcm = AudioPreprocessor.preprocess(url: url)
                        DispatchQueue.main.async {
                            if let pcm, pcm.count >= 4 {
                                pendingAudio.append(ChatAudio(name: name, pcm: pcm))
                            } else {
                                showAudioError(name)
                            }
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
        // Header control strip — deliberately NOT NSToolbar items. The AppKit
        // bridge does not re-measure a toolbar item whose SwiftUI content
        // changes size (enabling Agent mode grows the strip with the shield +
        // workspace chip), so NSToolbar clipped the controls into a mangled »
        // overflow menu until a window resize forced a fresh layout — and it
        // did so REGARDLESS of grouping (separate items, ViewThatFits, one
        // merged item all failed). A plain view row pinned to the pane's top
        // has no overflow mechanism to misfire.
        .safeAreaInset(edge: .top, spacing: 0) {
            chatHeaderBar
        }
        .sheet(isPresented: $showMCPMarketplace) {
            MCPMarketplaceView()
                .environmentObject(mcpManager)
        }
        // The in-window orb is just another view of the app-level voice
        // controller. Closing the sheet (Escape) leaves voice running so it
        // persists in the tray; the orb's red ✕ (onClose) explicitly ends it.
        .sheet(isPresented: $showVoiceMode) {
            VoiceModeView(controller: appState.voice,
                          onClose: { showVoiceMode = false; appState.voice.end() })
                .environmentObject(appState)
        }
        .alert("Enable thinking in Agent mode?", isPresented: $showThinkingInAgentConfirm) {
            Button("Cancel", role: .cancel) { }
            Button("Enable anyway") { enableThinking = true }
        } message: {
            Text("Thinking is not recommended with Agent mode — most local models tool-call more reliably without it. Do you still want to enable it?")
        }
        // Suppressed while Voice mode is open — two sheets can't co-present on
        // macOS, so the approval would never appear and the agent loop would hang.
        // Voice mode renders the approval as an overlay inside its own sheet instead.
        .sheet(item: Binding(get: { showVoiceMode ? nil : pendingApproval },
                             set: { pendingApproval = $0 })) { req in
            ToolApprovalSheet(request: req,
                              onAllow: { resolveApproval(.allow) },
                              onDeny: { resolveApproval(.deny) },
                              onAllowAll: { resolveApproval(.allow, allowAll: true) })
        }
        .onAppear {
            inputFocused = true
            syncTogglesFromSession()
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
            // Cmd+V into the focused chat input: if the clipboard holds an image,
            // PDF, or folder, attach it (same as the attach button / drag-drop)
            // and swallow the paste; plain text still pastes into the field.
            pasteMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
                guard inputFocused,
                      event.modifierFlags.intersection(.deviceIndependentFlagsMask) == .command,
                      event.charactersIgnoringModifiers == "v"
                else { return event }
                return pasteAttachmentsFromClipboard() ? nil : event
            }
        }
        .onDisappear {
            // Generation lives on the app-level engine now — closing the chat
            // window must NOT cancel an in-flight turn (the voice assistant may
            // be driving it with no window open). Only tear down this view's
            // scroll monitor.
            if let monitor = scrollMonitor {
                NSEvent.removeMonitor(monitor)
                scrollMonitor = nil
            }
            if let monitor = pasteMonitor {
                NSEvent.removeMonitor(monitor)
                pasteMonitor = nil
            }
        }
        // Pre-send nudge to enable Agent / MCP mode when the message looks like
        // it needs it. "Send Anyway" suppresses the suggestion for this chat.
        .confirmationDialog(
            pendingIntentPrompt == .mcp ? "Enable MCP first?" : "Enable Agent Mode first?",
            isPresented: Binding(get: { pendingIntentPrompt != nil },
                                 set: { if !$0 { pendingIntentPrompt = nil } }),
            titleVisibility: .visible,
            presenting: pendingIntentPrompt
        ) { prompt in
            Button(prompt == .mcp ? "Enable MCP & Send" : "Enable Agent Mode & Send") {
                enableForPrompt(prompt)
                pendingIntentPrompt = nil
                proceedSend()
            }
            Button("Send Anyway") {
                intentSuppress.suppress(prompt, for: sessionId)
                pendingIntentPrompt = nil
                proceedSend()
            }
            Button("Cancel", role: .cancel) { pendingIntentPrompt = nil }
        } message: { prompt in
            Text(prompt == .mcp
                 ? "This looks like it needs one of your MCP servers, but MCP mode is off. Enable it so those tools are available?"
                 : "This looks like a task for the agent (creating files, running commands, browsing the web…), but Agent mode is off. Enable it so the model can use its tools?")
        }
        // Persist the toolbar toggles back onto the visible session so each tab
        // remembers its own Think/Agent/MCP choice. Telegram sessions write the
        // shared config in their button handlers instead, so skip them here.
        .onChange(of: isAgentMode) { _, newValue in
            guard !isExternalBridgeSession,
                  let idx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }) else { return }
            appState.chatSessions[idx].mode = newValue ? .agent : .chat
        }
        .onChange(of: enableThinking) { _, newValue in
            guard !isExternalBridgeSession,
                  let idx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }) else { return }
            appState.chatSessions[idx].enableThinking = newValue
        }
        .onChange(of: mcpMode) { _, newValue in
            guard !isExternalBridgeSession,
                  let idx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }) else { return }
            appState.chatSessions[idx].useMCP = newValue
        }
        .onChange(of: chatEngine.isGenerating) { _, generating in
            if !generating { inputFocused = true }
        }
        .onChange(of: sessionId) { _, _ in
            // The view is reused across tabs, so reload the toolbar toggles from
            // the newly-visible session. The allow-list is NOT reset here — it's
            // keyed by session id, so each tab keeps its own decision across
            // switches (a session re-arms only when its Agent toggle goes off).
            syncTogglesFromSession()
        }
    }

    // MARK: - Document Folder (mini RAG)

    /// Pick a folder of mixed documents to index in-memory for this session.
    private func pickDocumentFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.message = "Choose a folder of documents to ask questions about (txt, md, pdf, json, yaml, csv …)"
        panel.prompt = "Attach"
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            DispatchQueue.main.async { attachDocumentFolder(url) }
        }
    }

    /// Index a folder for mini-RAG. Shared by the folder picker and paste/drop so
    /// every entry point behaves identically. Embeds on the local server's GPU;
    /// auto-downloads the default encoder model (35 MB, one-time) when none is
    /// available. Server down → lexical-only retrieval. Must run on the main actor.
    private func attachDocumentFolder(_ url: URL) {
        appState.documentIndexes[sessionId]?.cancel()
        let index = DocumentIndex(folderURL: url,
                                  embedderProvider: ServerEmbedding.autoProvider(port: server.port))
        appState.documentIndexes[sessionId] = index
        index.startIndexing()
    }

    // MARK: - Paste-to-attach

    /// Route the current clipboard to the same pending-attachment lists as the
    /// attach button (image / PDF / audio) and the folder picker (mini-RAG).
    /// Returns true when something was attached, so the caller can swallow the
    /// Cmd+V instead of letting it fall through to the text field.
    private func pasteAttachmentsFromClipboard() -> Bool {
        let pb = NSPasteboard.general
        var handled = false
        // Finder copies (folder / PDF / image file / audio file) arrive as real
        // file URLs — read them directly (NOT loadFileRepresentation) so a pasted
        // folder is indexed in place rather than as a sandboxed temp copy.
        if let urls = pb.readObjects(forClasses: [NSURL.self],
                                     options: [.urlReadingFileURLsOnly: true]) as? [URL] {
            for url in urls where attachFileURL(url) { handled = true }
        }
        if handled { return true }
        // Raw image data (screenshots, copy-image-from-a-browser) — no file URL.
        if let image = NSImage(pasteboard: pb) {
            pendingImages.append(image)
            return true
        }
        return false
    }

    /// Dispatch one file URL to the matching attachment path. Returns false for
    /// unsupported types so the caller leaves the paste alone.
    private func attachFileURL(_ url: URL) -> Bool {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else { return false }
        switch PasteFileKind.classify(ext: url.pathExtension, isDirectory: isDir.boolValue, audioSupported: audioSupported) {
        case .folder:
            attachDocumentFolder(url)
        case .pdf:
            if let text = Self.extractPDFText(from: url) {
                pendingPDFs.append((name: url.lastPathComponent, text: text))
            } else {
                showPDFError(url.lastPathComponent)
            }
        case .audio:
            addAudioAttachment(url)
        case .image:
            guard let image = NSImage(contentsOf: url) else { return false }
            pendingImages.append(image)
        case .unhandled:
            return false
        }
        return true
    }

    // MARK: - Image Helpers

    private func pickAttachment() {
        let panel = NSOpenPanel()
        // Only offer audio on models that can use it.
        panel.allowedContentTypes = audioSupported ? [.image, .pdf, .audio] : [.image, .pdf]
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
                } else if let utType = UTType(filenameExtension: url.pathExtension), utType.conforms(to: .audio) {
                    addAudioAttachment(url)
                } else if let image = NSImage(contentsOf: url) {
                    pendingImages.append(image)
                }
            }
        }
    }

    /// Decode an audio file to 16 kHz mono float32 PCM (off the main thread —
    /// AVFoundation decode can be slow) and add it as a pending attachment.
    private func addAudioAttachment(_ url: URL) {
        let name = url.lastPathComponent
        DispatchQueue.global(qos: .userInitiated).async {
            let pcm = AudioPreprocessor.preprocess(url: url)
            DispatchQueue.main.async {
                if let pcm, pcm.count >= 4 {
                    pendingAudio.append(ChatAudio(name: name, pcm: pcm))
                } else {
                    showAudioError(name)
                }
            }
        }
    }

    private func showAudioError(_ name: String) {
        let alert = NSAlert()
        alert.messageText = "Couldn't read audio"
        alert.informativeText = "\(name) couldn't be decoded. Supported: wav, mp3, m4a, aiff, caf, flac."
        alert.alertStyle = .warning
        alert.runModal()
    }

    /// Convert pending audio clips to a ChatAudio array, clearing the list.
    private func consumePendingAudio() -> [ChatAudio]? {
        guard !pendingAudio.isEmpty else { return nil }
        let clips = pendingAudio
        pendingAudio = []
        return clips
    }

    /// Whether the active model understands audio (Gemma 4 12B unified). Gates
    /// the mic button and audio-file attachment so they only appear where audio
    /// actually does something.
    private var audioSupported: Bool { server.modelInfo?.supportsAudio ?? false }

    /// Mic tap handler: start recording (after a permission check), or stop and
    /// turn the captured PCM into a pending audio attachment.
    private func toggleRecording() {
        if recorder.isRecording {
            if let pcm = recorder.stop(), pcm.count >= 4 {
                let secs = Double(pcm.count / 4) / AudioRecorder.targetSampleRate
                pendingAudio.append(ChatAudio(name: String(format: "Recording · %.0fs", secs.rounded()), pcm: pcm))
            }
            return
        }
        Task {
            let granted = await AudioRecorder.requestPermission()
            guard granted else { showMicPermissionError(); return }
            do {
                try recorder.start()
            } catch {
                showAudioError("the microphone")
            }
        }
    }

    private func showMicPermissionError() {
        let alert = NSAlert()
        alert.messageText = "Microphone access needed"
        alert.informativeText = "Enable microphone access for MLX Core in System Settings → Privacy & Security → Microphone, then try again."
        alert.alertStyle = .warning
        alert.runModal()
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

    // MARK: - Helpers

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        withAnimation(.easeOut(duration: 0.15)) {
            proxy.scrollTo("bottom", anchor: .bottom)
        }
    }


    /// Latest context usage from the most recent assistant message with token
    /// data — the prompt size + the reply's length of the last completed turn.
    private var contextUsage: (promptTokens: Int, completionTokens: Int, contextLength: Int)? {
        guard let messages = session?.messages else { return nil }
        if let last = messages.last(where: { $0.promptTokens != nil && $0.promptTokens! > 0 }) {
            let ctxLen = AgentEngine.effectiveContextLength(
                appContextSize: appState.contextSize,
                modelContextLength: server.modelInfo?.contextLength
            )
            return (promptTokens: last.promptTokens!, completionTokens: last.completionTokens ?? 0, contextLength: ctxLen)
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

    // MARK: - Voice Mode

    /// Resume the pending tool-approval continuation with the user's choice.
    /// Drives the text-chat approval sheet (the in-window orb and tray panel
    /// resolve their own approvals through the controller).
    private func resolveApproval(_ choice: ToolApprovalChoice, allowAll: Bool = false) {
        guard let req = pendingApproval else { return }
        if allowAll { toolAllowList.allowAll(sessionId) }
        req.continuation.resume(returning: choice)
        pendingApproval = nil
    }

    /// Present the in-window voice orb. The controller is app-level and may
    /// already be running (started from the tray) — only begin a session if it
    /// isn't active yet, ensuring a chat session exists first.
    private func startVoiceMode() {
        guard !showVoiceMode else { return }
        // Sync the voice toggles to the chat session being opened — talking should
        // start in the same Think/Agent/MCP mode as the chat you launched it from.
        if let s = session {
            appState.voice.enableThinking = s.enableThinking
            appState.voice.agentMode = s.mode == .agent
            appState.voice.mcpMode = s.useMCP
        }
        showVoiceMode = true
        guard !appState.voice.isActive else { return }
        if appState.activeChatId == nil { _ = appState.newChatSession() }
        Task { _ = await appState.voice.begin() }   // on permission failure the orb shows the error
    }

    // MARK: - Send Message

    /// Thin wrapper: build the turn config from the toolbar @State, consume the
    /// input draft + attachments (View-owned UI state), and hand the turn to the
    /// shared `ChatTurnEngine`. The engine routes to plain chat or the agent loop
    /// based on `config.agentMode || config.mcpMode` — there is no separate
    /// agent send path here anymore. Voice turns go straight through the
    /// controller and never touch this method.
    private func sendMessage() {
        // Telegram bridge sessions are read-only mirrors on the Mac — never inject
        // a Mac-typed turn (the composer is already replaced by a view-only bar;
        // this is belt-and-suspenders for any other trigger path).
        if session?.isExternalBridge == true { return }

        // Pre-send nudge: if the message looks like it needs a mode that's off,
        // confirm first (unless this chat already declined that suggestion). The
        // dialog's buttons call proceedSend(); nothing is consumed until then.
        let trimmed = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !chatEngine.isGenerating, server.status == .running, !trimmed.isEmpty,
           let prompt = detectIntentPrompt(for: trimmed) {
            pendingIntentPrompt = prompt
            return
        }
        proceedSend()
    }

    /// Names of MCP servers the user currently has enabled (disabled != true).
    private func enabledMCPServerNames() -> [String] {
        var out: [String] = []
        for (id, entry) in mcpManager.config.mcpServers where entry.disabled != true {
            out.append(id)
        }
        return out
    }

    /// Decide whether to nudge before sending. MCP takes priority over Agent
    /// because a named server is the more specific signal; both are gated on the
    /// matching mode being off and the suggestion not already declined this chat.
    private func detectIntentPrompt(for text: String) -> IntentPrompt? {
        let servers = enabledMCPServerNames()
        if !mcpMode, !servers.isEmpty,
           !intentSuppress.isSuppressed(.mcp, for: sessionId),
           ComposerIntent.wantsMCP(text, serverNames: servers) {
            return .mcp
        }
        if !isAgentMode,
           !intentSuppress.isSuppressed(.agent, for: sessionId),
           ComposerIntent.wantsAgent(text) {
            return .agent
        }
        return nil
    }

    /// Enable the mode the nudge suggested, mirroring the toolbar toggles'
    /// side effects (turning Agent on clears Thinking).
    private func enableForPrompt(_ prompt: IntentPrompt) {
        switch prompt {
        case .agent:
            isAgentMode = true
            enableThinking = false
        case .mcp:
            mcpMode = true
        }
    }

    private func proceedSend() {
        isNearBottom = true // snap to bottom on send

        var text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        let attachedImages = consumePendingImages()
        let attachedAudio = consumePendingAudio()
        let pdfText = consumePendingPDFsAsText()
        guard !text.isEmpty || attachedImages != nil || attachedAudio != nil || !pdfText.isEmpty,
              !chatEngine.isGenerating, server.status == .running else { return }
        inputText = ""
        if !pdfText.isEmpty {
            text = text.isEmpty ? pdfText : pdfText + "\n\n" + text
        }

        let config = ChatTurnEngine.TurnConfig(
            agentMode: isAgentMode,
            mcpMode: mcpMode,
            enableThinking: enableThinking,
            voiceStyle: false,
            workingDirectory: session?.workingDirectory,
            documentIndex: appState.documentIndexes[sessionId]
        )
        chatEngine.runTurn(sessionId: sessionId, userText: text,
                           images: attachedImages, audio: attachedAudio,
                           config: config,
                           approval: { await requestToolApproval($0) })
    }

    /// Ask the user to approve a single tool call. Returns true on Allow /
    /// Always Allow, false on Deny. Bypassed entirely when this session is on
    /// the allow-list. Bounces to the main actor (state mutations + sheet
    /// presentation) and suspends on a checked continuation until the sheet
    /// resumes it.
    @MainActor
    private func requestToolApproval(_ tc: APIClient.ToolCall) async -> Bool {
        // Read-only search over a folder the user explicitly attached — never
        // worth an approval interruption (docs-only mode has no other tools).
        if tc.name == "searchDocuments" { return true }
        if toolAllowList.allowsAll(sessionId) { return true }
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

}

// MARK: - Context Monitor

struct ContextMonitor: View {
    /// Prompt + reply tokens of the last COMPLETED turn (context occupied so far).
    let promptTokens: Int
    let completionTokens: Int
    /// Tokens the in-flight reply has produced so far (0 unless THIS chat is the
    /// one generating). Counted client-side from the streaming loop, advanced at
    /// the coalescer's ~20 Hz cadence — so the bar + "gen:" move live.
    let liveTokens: Int
    let isLive: Bool
    let contextLength: Int

    /// Total context occupied right now: last turn (prompt + its reply) plus the
    /// in-flight reply's running count. Pure → ContextMonitorTests.
    static func usedTokens(promptTokens: Int, completionTokens: Int, liveTokens: Int) -> Int {
        promptTokens + completionTokens + liveTokens
    }

    /// Bar fill / percentage, clamped to [0, 1] (a context overflow can't exceed
    /// 100%, and there's no context length before the first response). Pure.
    static func usageRatio(usedTokens: Int, contextLength: Int) -> Double {
        guard contextLength > 0 else { return 0 }
        return min(1.0, Double(usedTokens) / Double(contextLength))
    }

    /// The "gen:" figure: the live running count while this chat streams, else
    /// the last completed reply's length. Pure → ContextMonitorTests.
    static func genTokens(isLive: Bool, liveTokens: Int, completionTokens: Int) -> Int {
        isLive ? liveTokens : completionTokens
    }

    private var usedTokens: Int {
        Self.usedTokens(promptTokens: promptTokens, completionTokens: completionTokens, liveTokens: liveTokens)
    }
    private var usageRatio: Double { Self.usageRatio(usedTokens: usedTokens, contextLength: contextLength) }
    private var genTokens: Int { Self.genTokens(isLive: isLive, liveTokens: liveTokens, completionTokens: completionTokens) }

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
                        .frame(width: geo.size.width * usageRatio)
                        .animation(.linear(duration: 0.1), value: usageRatio)
                }
            }
            .frame(height: 4)

            HStack {
                Text("\(usedTokens)/\(contextLength) tokens (\(Int(usageRatio * 100))%)")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.secondary)
                Spacer()
                // Live count of the current reply's tokens (or the last reply's
                // length when idle). The bar color, not this, signals context pressure.
                Text("gen: \(genTokens)\(isLive ? "…" : "")")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(isLive ? Color.accentColor : .secondary)
            }
        }
        .padding(.horizontal, ChatMetrics.gutter)
        .padding(.top, 6)
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

                // Attached images. Double-click opens the full image in Preview.
                if let images = message.images, !images.isEmpty {
                    HStack(spacing: 4) {
                        ForEach(images) { img in
                            if let nsImage = NSImage(data: img.data) {
                                Image(nsImage: nsImage)
                                    .resizable()
                                    .aspectRatio(contentMode: .fit)
                                    .frame(maxWidth: 400, maxHeight: 300)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                                    .onTapGesture(count: 2) { ChatImagePreview.openInPreview(img) }
                                    .help("Double-click to open in Preview")
                            }
                        }
                    }
                }

                // Attached audio clips
                if let clips = message.audio, !clips.isEmpty {
                    ForEach(clips) { clip in
                        Label(String(format: "%@ · %.1fs", clip.name, clip.durationSeconds), systemImage: "waveform")
                            .font(.caption.weight(.medium))
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(Color.purple.opacity(0.18))
                            .clipShape(Capsule())
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
                    .padding(.horizontal, ChatMetrics.bubblePaddingH)
                    .padding(.vertical, ChatMetrics.bubblePaddingV)
                    .background(message.role == .user ? Color.accentColor : Color(.controlBackgroundColor))
                    .foregroundStyle(message.role == .user ? .white : .primary)
                    .clipShape(RoundedRectangle(cornerRadius: ChatMetrics.bubbleCornerRadius))
                }

                // Token usage stats — indented to the bubble's text column so
                // the caption reads as part of the reply, not the bubble edge.
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
                    .padding(.leading, ChatMetrics.statsIndent)
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

// MARK: - Tool-call grouping (collapse call + result into one collapsible row)

/// A renderable transcript row: a normal message, or a tool call paired with its
/// result(s) so they show as a single collapsible row instead of two bubbles.
enum ChatRow: Identifiable {
    case message(ChatMessage)
    case toolCall(call: ChatMessage, results: [ChatMessage])
    var id: UUID {
        switch self {
        case .message(let m): return m.id
        case .toolCall(let c, _): return c.id
        }
    }
}

/// Folds the agent's separate "tool call" and "tool result" summary messages
/// Which on/off state the chat toolbar's Think / Agent / MCP toggles show.
/// Telegram bridge sessions mirror the shared `serverOptions.telegram` config —
/// so the toolbar stays in lockstep with Settings (one source of truth, read
/// live by the bridge); normal sessions use the per-session / app-level state.
/// Pure → unit-tested in ChatModeTogglesTests.
struct ChatModeToggles: Equatable {
    var thinking: Bool
    var agent: Bool
    var mcp: Bool

    static func resolve(isExternalBridge: Bool,
                        telegramThinking: Bool, telegramAgent: Bool, telegramMCP: Bool,
                        inAppThinking: Bool, inAppAgent: Bool, inAppMCP: Bool) -> ChatModeToggles {
        isExternalBridge
            ? ChatModeToggles(thinking: telegramThinking, agent: telegramAgent, mcp: telegramMCP)
            : ChatModeToggles(thinking: inAppThinking, agent: inAppAgent, mcp: inAppMCP)
    }
}

/// (both `isAgentSummary`) into one row: a `name(args)` header with the result(s)
/// behind an expander. Pure → unit-tested in ChatRowBuilderTests.
enum ChatRowBuilder {
    /// A tool-RESULT summary is `**name** → output`; a tool-CALL summary is
    /// `**name**(args)`. The `** → ` right after the bolded name discriminates
    /// them (and also matches the "→ denied by user" result form).
    static func isResultSummary(_ m: ChatMessage) -> Bool {
        m.isAgentSummary && m.content.contains("** → ")
    }
    static func isCallSummary(_ m: ChatMessage) -> Bool {
        m.isAgentSummary && !m.content.contains("** → ")
    }

    static func rows(from messages: [ChatMessage]) -> [ChatRow] {
        // Same visibility rule as before: the raw tool-result messages
        // (role .system carrying a toolCallId) stay hidden from the transcript.
        let visible = messages.filter { $0.toolCallId == nil }
        var rows: [ChatRow] = []
        var i = 0
        while i < visible.count {
            let m = visible[i]
            if isCallSummary(m) {
                var results: [ChatMessage] = []
                var j = i + 1
                while j < visible.count, isResultSummary(visible[j]) {
                    results.append(visible[j]); j += 1
                }
                rows.append(.toolCall(call: m, results: results))
                i = j
            } else {
                rows.append(.message(m))
                i += 1
            }
        }
        return rows
    }
}

/// One collapsible tool-call row: a `name(args)` header (tap to expand) with the
/// tool result(s) revealed below. Replaces the old two-bubble call+result layout.
private struct ToolCallRow: View {
    let call: ChatMessage
    let results: [ChatMessage]
    @State private var expanded = false
    @State private var hovering = false
    @EnvironmentObject var processRegistry: ProcessRegistry

    /// Live background-process handles this card started — drives the kill X.
    /// Independent of `call.isStreaming` so the X stays after the tool returns,
    /// and it vanishes once the registry flips the process dead.
    private var killableHandles: [String] {
        ProcessCardControls.killable(handles: call.processHandles, isAlive: processRegistry.isAlive)
    }

    /// At least one background process from this card is still alive — drives the
    /// green "running" border. Goes false the moment the registry flips the last
    /// one dead (e.g. you click its X), so border + kill X disappear together.
    private var isRunningBackground: Bool { !killableHandles.isEmpty }

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            VStack(alignment: .leading, spacing: 6) {
                headerRow
                if expanded { expandedResults }
            }
            .padding(.horizontal, ChatMetrics.bubblePaddingH)
            .padding(.vertical, ChatMetrics.bubblePaddingV)
            .background(Color(.controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: ChatMetrics.bubbleCornerRadius))
            .overlay(
                RoundedRectangle(cornerRadius: ChatMetrics.bubbleCornerRadius)
                    .strokeBorder(Color.green.opacity(isRunningBackground ? 0.7 : 0), lineWidth: 1.5)
            )
            .animation(.easeInOut(duration: 0.2), value: isRunningBackground)
            // Recede a settled tool call so the assistant's prose carries the
            // conversation; full opacity while it's running, hovered, or expanded.
            .opacity(call.isStreaming || hovering || expanded ? 1.0 : 0.35)
            .animation(.easeInOut(duration: 0.15), value: hovering)
            .onHover { hovering = $0 }

            Spacer(minLength: 60)
        }
    }

    // Broken out into separately type-checked pieces — a single deeply nested
    // body (expander button + per-handle kill buttons + results) pushed the
    // SwiftUI type-checker into pathological (effectively non-terminating)
    // compile times.
    @ViewBuilder private var headerRow: some View {
        HStack(alignment: .top, spacing: 6) {
            Button {
                withAnimation(.easeInOut(duration: 0.15)) { expanded.toggle() }
            } label: {
                headerLabel
            }
            .buttonStyle(.plain)
            .disabled(results.isEmpty)

            ProcessKillButtons(handles: killableHandles) { processRegistry.kill(handle: $0) }
        }
    }

    @ViewBuilder private var headerLabel: some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: "wrench.and.screwdriver")
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(Self.stripBold(call.content))
                .font(.caption.monospaced())
                .multilineTextAlignment(.leading)
                .foregroundStyle(.primary)
            Spacer(minLength: 6)
            if call.isStreaming {
                GeneratingIndicator()
            } else if !results.isEmpty {
                Image(systemName: expanded ? "chevron.down" : "chevron.right")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .contentShape(Rectangle())
    }

    @ViewBuilder private var expandedResults: some View {
        ForEach(results) { r in
            Text(Self.stripBold(r.content))
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    /// The summary strings use `**name**` markdown bold; the compact mono header
    /// and body render as plain text, so strip the `**` markers.
    static func stripBold(_ s: String) -> String {
        s.replacingOccurrences(of: "**", with: "")
    }
}

/// Per-handle red kill X for a tool-call card's live background processes. Its
/// own type (not an inline ForEach in ToolCallRow.body) so the SwiftUI
/// type-checker handles it as an isolated, trivial unit.
private struct ProcessKillButtons: View {
    let handles: [String]
    let onKill: (String) -> Void

    var body: some View {
        ForEach(handles, id: \.self) { handle in
            Button {
                onKill(handle)
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.title2)
                    .foregroundStyle(.red)
                    .symbolRenderingMode(.hierarchical)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Stop background process \(handle)")
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
                let code = NSMutableAttributedString(string: content, attributes: attrs)
                linkifyBareUrls(code)
                result.append(code)

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
        linkifyBareUrls(result)
        return result
    }

    /// Shared detector — creating an NSDataDetector is not free and renderInline
    /// runs many times per second while streaming.
    private static let urlDetector = try? NSDataDetector(
        types: NSTextCheckingResult.CheckingType.link.rawValue
    )

    /// Add `.link` attributes for http(s) URLs the markdown parser left
    /// unlinked. CommonMark autolinks a bare `http://…` but NOT one inside a
    /// code span — and models love `` `http://localhost:3000` `` — so a URL
    /// would flicker clickable mid-stream (before the closing backtick
    /// arrives) then go dead once the span completes. NSDataDetector handles
    /// boundaries and trailing punctuation; only http/https matches are
    /// linkified (no bare-domain or mailto surprises), and spans that already
    /// carry a link (e.g. from `[text](url)`) are left untouched. Display
    /// styling comes from the text view's `linkTextAttributes`.
    private static func linkifyBareUrls(_ result: NSMutableAttributedString) {
        guard let detector = urlDetector else { return }
        let full = NSRange(location: 0, length: result.length)
        for match in detector.matches(in: result.string, range: full) {
            guard let url = match.url,
                  let scheme = url.scheme?.lowercased(),
                  scheme == "http" || scheme == "https" else { continue }
            var alreadyLinked = false
            result.enumerateAttribute(.link, in: match.range, options: []) { value, _, stop in
                if value != nil {
                    alreadyLinked = true
                    stop.pointee = true
                }
            }
            if !alreadyLinked {
                result.addAttribute(.link, value: url, range: match.range)
            }
        }
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

// MARK: - GrowingTextEditor (editable, auto-growing, scrollable composer)

/// Pure layout math for the auto-growing composer. Factored out of the
/// NSViewRepresentable so it is unit-testable (the view itself is not).
enum ComposerLayout {
    /// Clamp the editor height between `minLines` and `maxLines` worth of text.
    /// Returns the height SwiftUI frames the editor at, plus whether the content
    /// overflows the cap (so the inner scroll view scrolls — the behavior the
    /// old `TextField(axis: .vertical)` never had).
    static func resolve(contentHeight: CGFloat,
                        lineHeight: CGFloat,
                        minLines: Int,
                        maxLines: Int,
                        verticalInset: CGFloat) -> (height: CGFloat, scrolls: Bool) {
        let lo = max(1, minLines)
        let hi = max(lo, maxLines)
        let minH = lineHeight * CGFloat(lo) + verticalInset
        let maxH = lineHeight * CGFloat(hi) + verticalInset
        let natural = contentHeight + verticalInset
        let clamped = Swift.max(minH, Swift.min(natural, maxH))
        return (clamped, natural > maxH + 0.5)
    }
}

/// What a Return keypress does in the composer. Mirrors the prior `.onKeyPress`
/// contract: Shift+Return is always a newline; a bare Return sends only when
/// idle, and is otherwise swallowed (never a stray newline mid-generation).
enum ComposerReturnAction: Equatable { case send, newline, ignore }

enum ComposerKey {
    static func onReturn(shift: Bool, isIdle: Bool) -> ComposerReturnAction {
        if shift { return .newline }
        return isIdle ? .send : .ignore
    }
}

/// Editable, auto-growing, scrollable text input backed by NSTextView in an
/// NSScrollView. SwiftUI's `TextField(axis: .vertical)` re-lays out the whole
/// string on every edit (janky on a big paste) and exposes no scroller (the
/// mouse wheel does nothing past the line limit). TextKit handles large text
/// natively and the scroll view gives real mouse-wheel scrolling.
fileprivate struct GrowingTextEditor: NSViewRepresentable {
    @Binding var text: String
    @Binding var isFocused: Bool
    @Binding var measuredHeight: CGFloat
    var font: NSFont = .preferredFont(forTextStyle: .body)
    var minLines: Int = 1
    var maxLines: Int = 15
    var isIdle: Bool
    var onSend: () -> Void

    let inset = NSSize(width: 7, height: 8)

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSScrollView()
        scroll.drawsBackground = false
        scroll.hasVerticalScroller = true
        scroll.hasHorizontalScroller = false
        scroll.autohidesScrollers = true
        scroll.borderType = .noBorder
        scroll.verticalScrollElasticity = .allowed

        let tv = ComposerTextView()
        tv.delegate = context.coordinator
        tv.isEditable = context.environment.isEnabled
        tv.isSelectable = true
        tv.isRichText = false
        tv.allowsUndo = true
        tv.drawsBackground = false
        tv.font = font
        tv.textColor = .labelColor
        tv.insertionPointColor = .labelColor
        tv.textContainerInset = inset
        tv.isVerticallyResizable = true
        tv.isHorizontallyResizable = false
        tv.textContainer?.widthTracksTextView = true
        tv.textContainer?.lineFragmentPadding = 2
        tv.autoresizingMask = [.width]
        tv.string = text
        tv.onBecomeFocus = { [weak c = context.coordinator] in c?.setFocus(true) }
        tv.onResignFocus = { [weak c = context.coordinator] in c?.setFocus(false) }

        scroll.documentView = tv
        context.coordinator.textView = tv
        DispatchQueue.main.async { context.coordinator.recomputeHeight() }
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        guard let tv = scroll.documentView as? ComposerTextView else { return }
        context.coordinator.parent = self
        // External text changes (e.g. cleared on send) without clobbering an
        // in-progress edit at the same value.
        if tv.string != text {
            tv.string = text
            context.coordinator.recomputeHeight()
        }
        let enabled = context.environment.isEnabled
        if tv.isEditable != enabled { tv.isEditable = enabled }
        // Drive AppKit first-responder from the SwiftUI focus mirror.
        if isFocused, tv.window != nil, tv.window?.firstResponder !== tv {
            DispatchQueue.main.async { tv.window?.makeFirstResponder(tv) }
        }
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: GrowingTextEditor
        weak var textView: ComposerTextView?
        init(_ parent: GrowingTextEditor) { self.parent = parent }

        func textDidChange(_ notification: Notification) {
            guard let tv = textView else { return }
            if parent.text != tv.string { parent.text = tv.string }
            recomputeHeight()
        }

        func textView(_ textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
            guard commandSelector == #selector(NSResponder.insertNewline(_:)) else { return false }
            let shift = NSApp.currentEvent?.modifierFlags.contains(.shift) ?? false
            switch ComposerKey.onReturn(shift: shift, isIdle: parent.isIdle) {
            case .newline:
                textView.insertNewlineIgnoringFieldEditor(self)
                return true
            case .send:
                parent.onSend()
                return true
            case .ignore:
                return true
            }
        }

        func setFocus(_ value: Bool) {
            guard parent.isFocused != value else { return }
            DispatchQueue.main.async { self.parent.isFocused = value }
        }

        func recomputeHeight() {
            guard let tv = textView, let lm = tv.layoutManager, let tc = tv.textContainer else { return }
            lm.ensureLayout(for: tc)
            let content = lm.usedRect(for: tc).height
            let line = tv.font.map { $0.ascender - $0.descender + $0.leading } ?? 16
            let r = ComposerLayout.resolve(contentHeight: content,
                                           lineHeight: max(line, 12),
                                           minLines: parent.minLines,
                                           maxLines: parent.maxLines,
                                           verticalInset: parent.inset.height * 2)
            // Past the cap the scroll view owns overflow; below it the frame grows.
            if let scroll = tv.enclosingScrollView {
                scroll.hasVerticalScroller = r.scrolls
            }
            if abs(parent.measuredHeight - r.height) > 0.5 {
                DispatchQueue.main.async { self.parent.measuredHeight = r.height }
            }
        }
    }
}

/// NSTextView that reports focus transitions so SwiftUI's `inputFocused` mirror
/// stays accurate — the Cmd+V "attach from clipboard" monitor reads it.
fileprivate final class ComposerTextView: NSTextView {
    var onBecomeFocus: (() -> Void)?
    var onResignFocus: (() -> Void)?
    override func becomeFirstResponder() -> Bool {
        let ok = super.becomeFirstResponder()
        if ok { onBecomeFocus?() }
        return ok
    }
    override func resignFirstResponder() -> Bool {
        let ok = super.resignFirstResponder()
        if ok { onResignFocus?() }
        return ok
    }
}
