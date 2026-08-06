import SwiftUI
import PDFKit
import UniformTypeIdentifiers
import AppKit

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
        case "generate_speech": return "Generate spoken audio"
        case "generate_music": return "Generate a music track"
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
    @Environment(\.dismissWindow) private var dismissWindow
    @State private var columnVisibility = NavigationSplitViewVisibility.automatic
    /// Flipped by the gate sheet's Cancel, and by nothing else.
    ///
    /// The sheet was first presented on a `.constant(true)` binding, which made
    /// it properly blocking and made Cancel UNIMPLEMENTABLE: AppKit refuses to
    /// close a window that has an attached sheet, so the click did nothing and
    /// the user was stuck (measured through the accessibility API — the sheet
    /// was still on screen afterwards). The binding's setter still swallows
    /// SwiftUI's own dismissals, so Esc and click-away can't drop the user onto
    /// the dead composer underneath; Cancel is the one door, and it ends the
    /// sheet before closing the window. Window scenes rebuild their content on
    /// reopen, so this resets itself.
    @State private var gateCancelled = false

    /// The starter recommendation this Mac gets — same function the welcome
    /// window and the Model Browser read.
    private var starterPick: RecommendedModelPick {
        RecommendedModelPick.starterPick(physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory)
    }

    /// Blocking whenever nothing on this Mac can serve a chat. The progress
    /// argument is nil here on purpose: this only decides WHETHER to block, and
    /// `appState.localModels` is what flips it back. The sheet itself observes
    /// the download manager for the live figure.
    private var gateIsBlocking: Bool {
        ChatGateState.resolve(localModels: appState.localModels,
                              activeDownload: nil,
                              lanChatModelCount: server.lanModels(capability: "chat").count).isBlocking
    }

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            ChatSidebar()
                .navigationSplitViewColumnWidth(min: 180, ideal: 220, max: 280)
                // Both columns carry the window's toolbar material, so the bar
                // reads as one surface across the split rather than appearing
                // only over the detail side. It is also what the list's
                // scroll-edge effect attaches to.
                .toolbarBackground(.visible, for: .windowToolbar)
        } detail: {
            if let sessionId = appState.activeChatId,
               appState.chatSessions.contains(where: { $0.id == sessionId }) {
                ChatDetailView(sessionId: sessionId)
            } else {
                // No conversation yet — open one immediately rather than showing
                // a "Start a conversation" wall with a button. The first thing a
                // chat app should present is somewhere to type: `ChatDetailView`
                // already renders the greeting above a centered composer while a
                // session has no messages, so a fresh launch and a fresh chat
                // look the same. Creating it in `onAppear` (not during the view
                // update) keeps SwiftUI from seeing state mutate mid-layout; the
                // branch can only fire once, because it sets `activeChatId`.
                Color.clear
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .onAppear { _ = appState.newChatSession() }
            }
        }
        .navigationTitle("")
        // Blocking: the setter drops SwiftUI's own dismissals, so nothing but
        // Cancel takes this sheet down. The getter is recomputed every update,
        // so it also clears ITSELF the moment a chat model lands.
        .sheet(isPresented: Binding(get: { gateIsBlocking && !gateCancelled },
                                    set: { _ in })) {
            ChatModelGateSheet(pick: starterPick, onCancel: cancelGate)
                .environmentObject(appState)
                .environmentObject(appState.downloads)
                .environmentObject(server)
        }
        .onAppear {
            // Menu bar apps need explicit activation for keyboard focus — and
            // the `.regular` flip must come FIRST. (The old comment here claimed
            // ActivationPolicyManager would handle it "when this window becomes
            // key", which is the bug: an inactive accessory app has no key
            // window, so that notification never arrives.)
            DispatchQueue.main.async {
                AppActivation.focus()
            }
            // The gate reads `localModels`; a chat window opened right after a
            // download landed elsewhere must not show a stale one.
            appState.refreshModels()
        }
    }

    /// Cancel on the gate: end the sheet, THEN close the window. Both halves
    /// are required and the order is load-bearing — a window with an attached
    /// sheet can't be closed, and dismissing to the composer underneath is the
    /// dead end this gate exists to replace.
    private func cancelGate() {
        gateCancelled = true
        DispatchQueue.main.async { dismissWindow(id: "chat") }
    }
}

// MARK: - Sidebar

struct ChatSidebar: View {
    @EnvironmentObject var appState: AppState
    @Environment(\.openWindow) private var openWindow
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
                        // Who this chat is talking to, when it isn't the app
                        // defaults. Sits on the timestamp line rather than the
                        // title's so a long agent name can never squeeze the
                        // title, which is what the row is FOR — and it answers
                        // "why does this thread behave differently" without
                        // opening it.
                        HStack(spacing: 4) {
                            if let agent = appState.agents.agent(id: session.agentId) {
                                Image(systemName: agent.symbol)
                                    .font(.system(size: 9))
                                    .foregroundStyle(isSelected ? Color.white.opacity(0.85) : Color.accentColor)
                                Text(agent.name)
                                    .font(.caption2)
                                    .lineLimit(1)
                                    .truncationMode(.tail)
                                    .foregroundStyle(isSelected ? Color.white.opacity(0.85) : Color.accentColor)
                                Text("·")
                                    .font(.caption2)
                                    .foregroundStyle(isSelected ? Color.white.opacity(0.5) : Color.secondary.opacity(0.4))
                            }
                            Text(relativeTime(session.updatedAt))
                                .font(.caption2)
                                .foregroundStyle(isSelected ? Color.white.opacity(0.7) : Color.secondary.opacity(0.5))
                        }
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
        // The platform's own scroll-edge effect at BOTH ends: rows pass under
        // the window's top edge and under the New Chat row (a `safeAreaInset`,
        // so content scrolls beneath it), and a soft edge is how macOS frosts
        // that overlap. Not a hand-drawn band — a custom strip pulled into this
        // area once looked native and swallowed every click in it.
        .scrollEdgeEffectStyle(.soft, for: [.top, .bottom])
        // No Agents / Models entries here: both windows are reachable from the
        // chat toolbar now — "Manage Agents…" in the composer's agent chip and
        // "Manage Models…" in the model picker — and each sat next to the
        // control it configures. The sidebar is the conversation list, and a
        // second route to the same two windows only competed with it.
        .safeAreaInset(edge: .bottom) {
            HStack(spacing: 8) {
                Button {
                    _ = appState.newChatSession()
                } label: {
                    Label("New Chat", systemImage: "plus")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .contentShape(Rectangle())
                }
                // `.plain` + our own frame and background, NOT `.bordered` —
                // see `ChatMetrics.sidebarButtonHeight`. It is the only way the
                // two controls are the same height by construction rather than
                // by whatever a style style decides their labels are worth.
                .buttonStyle(.plain)
                .sidebarActionButton()
                newAgentChatMenu
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
        }
    }

    /// Start a chat AS an agent.
    ///
    /// Next to New Chat because that is WHEN the choice is made: a session's
    /// agent is fixed once the session exists (there is no `setAgent` any more),
    /// so this is the only place it can be picked. It used to be a chip in the
    /// composer, where it configured the whole conversation from the row that
    /// configures one message — and a mid-thread switch silently re-pointed the
    /// prompt, tools, model and voice of a conversation already underway.
    private var newAgentChatMenu: some View {
        Menu {
            if appState.agents.allAgents.isEmpty {
                Text("No agents yet")
            }
            ForEach(appState.agents.allAgents) { agent in
                let decision = appState.agentModelDecision(for: agent)
                Button {
                    appState.startChat(withAgent: agent.id)
                } label: {
                    // Same rule as the old chip: an agent whose pinned model
                    // isn't downloaded says so rather than failing at send.
                    Label(AgentModelSwitch.isSelectable(decision)
                          ? agent.name : "\(agent.name) — model not downloaded",
                          systemImage: agent.symbol)
                }
                .disabled(!AgentModelSwitch.isSelectable(decision))
            }
            Divider()
            Button("Manage Agents…") {
                AppActivation.openWindow(id: "agents", using: openWindow)
            }
        } label: {
            Image(systemName: "person.badge.plus")
                .frame(width: 34, height: ChatMetrics.sidebarButtonHeight)
                .contentShape(Rectangle())
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .sidebarActionButton()
        .help("New chat with an agent — its prompt, tools, model and voice")
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
    @State private var showThinkingInAgentConfirm = false
    @State private var executingPlanMessageId: UUID?
    // Follow-the-newest-line. The decision core is pure (`ChatScrollState`,
    // pinned by ChatScrollTests); the model holds it in a class so per-frame
    // scroll geometry doesn't re-evaluate the whole chat body — only the
    // pinned flag is published. `scrollPosition` is the one handle that moves
    // the transcript, so no view needs a `ScrollViewProxy` passed around.
    @StateObject private var scrollModel = ChatScrollModel()
    @State private var scrollPosition = ScrollPosition(idType: Never.self, edge: .bottom)
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
    /// from `serverOptions.telegram` for a Telegram session, else the in-app
    /// state, and overridden by the tab's agent for whatever it decides.
    private var toolbarToggles: ChatModeToggles {
        let tg = appState.serverOptions.telegram
        return ChatModeToggles.resolve(
            isExternalBridge: isExternalBridgeSession,
            telegramThinking: tg.enableThinking, telegramAgent: tg.agentMode, telegramMCP: tg.useMCP,
            inAppThinking: enableThinking, inAppAgent: isAgentMode, inAppMCP: mcpMode,
            agentLock: agentModeLock)
    }

    /// What this tab's agent decided about Think / Tools / MCP, nil with no agent.
    ///
    /// Built from the SAME `resolvedAgentSettings` the turn runs under, not from
    /// the agent's `capabilities` directly: a second copy of that rule is exactly
    /// how the discs ended up disagreeing with what ran (every agent defaults
    /// `web: true`, so resolution forced the tool loop on while the wrench still
    /// rendered OFF). Cheap — the resolution is a pure fold.
    private var agentModeLock: AgentModeLock? {
        guard let agent = activeAgent else { return nil }
        let resolved = appState.resolvedAgentSettings(
            agentId: agent.id,
            toolsEnabled: isAgentMode,
            mcpEnabled: mcpMode,
            thinkingEnabled: enableThinking,
            workingDirectory: session?.workingDirectory,
            disabledTools: ChatSession.disabledToolKinds(session?.disabledTools ?? []))
        return AgentModeLock(name: agent.name,
                             // The only one an agent may leave to the chat.
                             thinking: agent.enableThinking,
                             tools: resolved.toolsEnabled,
                             mcp: resolved.mcpEnabled)
    }

    // MARK: Mode controls (Think / Tools / MCP)
    //
    // Icon-only, rendered in the COMPOSER row next to the paperclip — see
    // `modeIcon` for why state has to read from colour alone once the captions
    // are gone, and `composerControls` for the row itself.

    /// The chat pane's ONE toolbar resident: model picker, voice, settings, as a
    /// single floating capsule at the leading edge.
    ///
    /// It rides a real `ToolbarItem` because that band's own layer swallows
    /// clicks from anything else placed in it (a custom strip pulled in via
    /// ignoresSafeArea looked native and was completely dead). What changed is
    /// the BACKGROUND: the band no longer paints a 100%-width strip across the
    /// window, so this cluster carries its own material and reads as floating
    /// over the transcript. That material is load-bearing — without it, content
    /// scrolling under the cluster bleeds through the controls, which is exactly
    /// why the band used to be painted at 0.8 opacity.
    ///
    /// The three mode controls that used to sit here are in the COMPOSER row
    /// now: their captions were most of this cluster's width budget, and they
    /// configure the message being written, not the window.
    private var floatingToolbar: some View {
        // Voice moved OUT to the composer row (`voiceToggle`): it configures
        // the conversation you're having, not the window — and it freed this
        // cluster's width budget.
        HStack(spacing: 4) {
            ChatModelPill(showsBackground: false)
            // The server being down is discovered HERE — you type and nothing
            // answers — so the fix is offered here too, next to the picker,
            // instead of only in the tray. Transient by construction: it is
            // gone the moment the server is up, so it costs the cluster's width
            // budget only while it has something to say.
            serverStartControl
            Divider().frame(height: 14)
            Button {
                AppActivation.openWindow(id: "settings", using: openWindow)
            } label: {
                Image(systemName: "gear")
                    .font(.system(size: 12))
                    .foregroundStyle(.primary)
                    .frame(width: 22, height: 22)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Open Settings (⌘,) — server flags, speculative decoding, performance (continuous batching, KV-quant, prefix cache), and per-request defaults.")
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.regularMaterial, in: Capsule())
        .overlay(Capsule().stroke(Color.secondary.opacity(0.20), lineWidth: 0.5))
    }

    /// Start the server from the chat window. Drawn rather than styled
    /// (`.plain` + explicit padding/fill) — a `.bordered` control keeps its
    /// intrinsic size and would sit at a different height to the pill beside
    /// it, which is the sidebar New Chat lesson.
    @ViewBuilder private var serverStartControl: some View {
        let control = ChatServerStartControl.resolve(
            status: server.status,
            hasStartableModel: !appState.selectedModelPath.isEmpty || server.lanChatModelId != nil
        )
        if control != .hidden {
            Button {
                // ONE start path, shared with the LAN toggle: it loads the
                // selected checkpoint, or boots headless when the model
                // answering is on another Mac. A second `server.start` call
                // site here is how the two would drift.
                appState.ensureServerForLan()
            } label: {
                HStack(spacing: 4) {
                    if control == .starting {
                        ProgressView().controlSize(.small).scaleEffect(0.6).frame(width: 10, height: 10)
                    } else {
                        Image(systemName: "play.fill").font(.system(size: 9, weight: .bold))
                    }
                    Text(control.title)
                        .font(.caption.weight(.semibold))
                }
                .foregroundStyle(control.isRed ? Color.white : Color.secondary)
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(control.isRed ? Color.red : Color.secondary.opacity(0.15), in: Capsule())
                .contentShape(Capsule())
            }
            .buttonStyle(.plain)
            .disabled(!control.isEnabled)
            .help(control == .starting
                  ? "Loading the model — this can take a while for a large one."
                  : "The server isn't running, so nothing can answer. Click to start it.")
        }
    }

    // The per-tab agent PICKER used to sit here, between the paperclip and the
    // mode discs. It's next to New Chat now (`ChatSidebar.newAgentChatMenu`):
    // it configures the whole conversation rather than the message being
    // written, and a session's agent is fixed once the session exists — a
    // mid-thread switch left half a conversation running under someone else's
    // prompt, tools, model and voice, and flipped the three discs beside it
    // while it did. The chat still SHOWS who it's talking to (the sidebar row,
    // and the locked discs name the agent on hover).

    /// The agent this tab is talking to (nil = none).
    private var activeAgent: Agent? { appState.agents.agent(id: session?.agentId) }

    /// Images/PDFs/audio for this message, or a folder to ask questions about.
    /// Its own property (rather than inline in `composerControls`) so it carries
    /// a hover card like every other glyph in the row.
    private var attachmentMenu: some View {
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
        // .plain button style (not .borderlessButton menu style) — the latter
        // substitutes its own chrome on macOS, dropping the circle background
        // and mis-baselining the glyph.
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .frame(width: ChatMetrics.composerControlSize, height: ChatMetrics.composerControlSize)
        .composerTip(.attachments(audioSupported: audioSupported))
    }

    /// Shared look for the composer's icon-only mode controls.
    ///
    /// Captioned pills in the toolbar became bare glyphs here, so the state has
    /// to read from COLOR alone: tinted glyph on a tinted disc when on, secondary
    /// on the same neutral disc as the paperclip when off. Same circle geometry
    /// as every other composer control, so the row stays on one baseline.
    /// `lockedBy` draws an inset ring inside the disc — the control still reads
    /// its state from colour, and the ring says the state isn't yours to change.
    /// A dimmed-out glyph was the alternative and it loses the ON/OFF reading,
    /// which is the one thing the disc has to keep saying.
    private func modeIcon(_ icon: String, isOn: Bool, onColor: Color,
                          lockedBy: String? = nil) -> some View {
        Image(systemName: icon)
            .font(.system(size: 13, weight: .medium))
            .foregroundStyle(isOn ? onColor : Color.secondary)
            .frame(width: ChatMetrics.composerIconSize, height: ChatMetrics.composerIconSize)
            .background(isOn ? onColor.opacity(0.20) : Color.secondary.opacity(0.15))
            .clipShape(Circle())
            .overlay {
                if lockedBy != nil {
                    Circle()
                        .inset(by: 1)
                        .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [2, 2]))
                        .foregroundStyle((isOn ? onColor : Color.secondary).opacity(0.65))
                        .frame(width: ChatMetrics.composerIconSize,
                               height: ChatMetrics.composerIconSize)
                }
            }
            .frame(width: ChatMetrics.composerControlSize, height: ChatMetrics.composerControlSize)
            // A .plain Button without an explicit content shape only hit-tests
            // the drawn glyph pixels, not the disc.
            .contentShape(Circle())
    }

    /// What a locked disc offers instead of its own controls: who decided it, and
    /// the way to the place where that decision lives. A locked control that does
    /// nothing at all on click is the dead-control class — this is the same shape
    /// as the tool menu's "not in <agent>'s capabilities" rows.
    @ViewBuilder
    private func lockedModeMenu(_ agentName: String) -> some View {
        Text("Set by \(agentName)")
        Button("Edit Agent…") {
            // ON that agent — the window otherwise opens on whoever sorts
            // first, which is the wrong one every time you got here from a card
            // that just named a different name.
            guard let id = activeAgent?.id else { return }
            appState.openAgentSettings(id, using: openWindow)
        }
    }

    private var thinkToggle: some View {
        Group {
            if let owner = toolbarToggles.thinkingLockedBy {
                Menu {
                    lockedModeMenu(owner)
                } label: {
                    modeIcon("brain", isOn: toolbarToggles.thinking, onColor: .blue, lockedBy: owner)
                }
                .menuStyle(.button)
                .buttonStyle(.plain)
                .menuIndicator(.hidden)
            } else {
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
                    modeIcon("brain", isOn: toolbarToggles.thinking, onColor: .blue)
                }
                .buttonStyle(.plain)
            }
        }
        .composerTip(.thinking(isOn: toolbarToggles.thinking,
                               lockedBy: toolbarToggles.thinkingLockedBy))
    }

    /// One wrench: CLICK flips the tool loop, secondary-click opens the per-tool
    /// switches and the workspace.
    ///
    /// It used to open the menu on click, with on/off as the first row. This is
    /// the composer's most-flipped control, so the frequent action was paying a
    /// click plus a scan down a long list every time. A bare glyph meaning two
    /// things by WHERE you click is still out (that was the split-pill problem);
    /// meaning two things by WHICH button is the standard macOS split, and
    /// `primaryAction:` also gives press-and-hold for free — the context menu is
    /// there because press-and-hold is not something anyone discovers.
    ///
    /// While the tab has an agent this is a LOCKED indicator: `AgentResolution`
    /// takes the loop straight from the agent's capabilities and ignores whatever
    /// the chat says, so a live toggle here would be a control that changes
    /// nothing. It shows what the agent decided and points at the editor.
    private var agentToggle: some View {
        Group {
            if let owner = toolbarToggles.toolsLockedBy {
                Menu {
                    lockedModeMenu(owner)
                } label: {
                    modeIcon("wrench", isOn: toolbarToggles.agent, onColor: .orange, lockedBy: owner)
                }
            } else {
                Menu {
                    toolMenuContent
                } label: {
                    modeIcon("wrench", isOn: toolbarToggles.agent, onColor: .orange)
                } primaryAction: {
                    setToolsEnabled(!toolbarToggles.agent)
                }
                .contextMenu { toolMenuContent }
            }
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .composerTip(.tools(isOn: toolbarToggles.agent,
                            workspace: session?.workingDirectory,
                            lockedBy: toolbarToggles.toolsLockedBy))
    }

    /// Flip the tool loop for this chat. Shared by the wrench click and the
    /// pre-send intent nudge, so the approval re-arm and the thinking auto-off
    /// can't apply on one path and not the other.
    private func setToolsEnabled(_ on: Bool) {
        // The agent decides this one — the disc offers no primary action while
        // locked, but the pre-send nudge calls in here too.
        guard toolbarToggles.toolsLockedBy == nil else { return }
        if isExternalBridgeSession {
            // Telegram session: flip the shared config (in sync with Settings);
            // no per-session approval state applies here.
            appState.serverOptions.telegram.agentMode = on
            return
        }
        isAgentMode = on
        // Re-arm the approval gate every time the user re-enters Agent mode.
        // "Always allow this session" decays here — for THIS tab only; other
        // tabs keep their decision.
        if !on { toolAllowList.rearm(sessionId) }
        // Thinking + tool-calling loops degrade quality on most local models —
        // auto-off when entering Agent mode.
        if on { enableThinking = false }
    }

    // MARK: Per-chat tool switches
    //
    // Subtractive by construction: the menu writes to the SESSION's disabled
    // set, which `AgentResolution` removes from whatever the agent (or the app
    // defaults) already allowed. A chat tab can therefore never hand an agent a
    // capability its own settings forbid — those rows render disabled instead of
    // offering a switch that the resolver would ignore.

    /// What the tab's agent permits at all; everything when there's no agent.
    private var agentAllowedTools: Set<AgentToolKind> {
        activeAgent.map { $0.capabilities.resolvedTools() } ?? Set(AgentToolKind.allCases)
    }

    private var disabledToolSet: Set<AgentToolKind> {
        ChatSession.disabledToolKinds(session?.disabledTools ?? [])
    }

    private func isToolEnabled(_ tool: AgentToolKind) -> Bool {
        agentAllowedTools.contains(tool) && !disabledToolSet.contains(tool)
    }

    private func setTool(_ tool: AgentToolKind, enabled: Bool) {
        guard let idx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }) else { return }
        var disabled = disabledToolSet
        if enabled { disabled.remove(tool) } else { disabled.insert(tool) }
        appState.chatSessions[idx].disabledTools = disabled.map(\.rawValue).sorted()
        appState.saveChatHistory()
    }

    /// Per-tool switches + the workspace they apply to. No on/off row: that is
    /// what a click on the wrench does, and one boolean with two controls is how
    /// the two end up disagreeing. No bulk "all" rows either — they duplicated
    /// the switches sitting directly beneath them, and read as a second way to
    /// turn the loop off.
    @ViewBuilder
    private var toolMenuContent: some View {
        ForEach(AgentToolGroup.allCases, id: \.self) { group in
            Section(group.title) {
                ForEach(group.tools, id: \.self) { tool in
                    let allowed = agentAllowedTools.contains(tool)
                    Button {
                        setTool(tool, enabled: !isToolEnabled(tool))
                    } label: {
                        if isToolEnabled(tool) {
                            Label(tool.displayName, systemImage: "checkmark")
                        } else if allowed {
                            Text(tool.displayName)
                        } else {
                            // The agent forbids it — say so rather than showing
                            // an off switch the user can't turn on.
                            Text("\(tool.displayName) — not in \(activeAgent?.name ?? "agent")'s capabilities")
                        }
                    }
                    .disabled(!allowed || isExternalBridgeSession)
                }
            }
        }

        Divider()
        Button("Workspace…") {
            if let picked = WorkspacePicker.pickDirectory() {
                workingDirectoryBinding.wrappedValue = picked
            }
        }
        .disabled(isExternalBridgeSession)
        Text(session?.workingDirectory ?? "No workspace set")
    }

    /// Flip MCP for this chat — the Telegram bridge writes the shared config it
    /// reads live, everyone else the app-level state.
    private func setMCPEnabled(_ on: Bool) {
        guard toolbarToggles.mcpLockedBy == nil else { return }
        if isExternalBridgeSession {
            appState.serverOptions.telegram.useMCP = on
        } else {
            mcpMode = on
        }
    }

    /// Same shape as `agentToggle`: click toggles, secondary-click opens the
    /// Marketplace the gear half used to hold.
    private var mcpToggle: some View {
        Group {
            if let owner = toolbarToggles.mcpLockedBy {
                Menu {
                    lockedModeMenu(owner)
                } label: {
                    modeIcon("puzzlepiece.extension", isOn: toolbarToggles.mcp,
                             onColor: .purple, lockedBy: owner)
                }
            } else {
                Menu {
                    mcpMenuContent
                } label: {
                    modeIcon("puzzlepiece.extension", isOn: toolbarToggles.mcp, onColor: .purple)
                } primaryAction: {
                    setMCPEnabled(!toolbarToggles.mcp)
                }
                .contextMenu { mcpMenuContent }
            }
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .composerTip(.mcp(isOn: toolbarToggles.mcp, lockedBy: toolbarToggles.mcpLockedBy))
    }

    @ViewBuilder
    private var mcpMenuContent: some View {
        Button("MCP Marketplace…") { showMCPMarketplace = true }
    }

    /// A conversation with nothing in it yet. Rendered instead of an empty
    /// scroll view so the composer sits under a greeting in the middle of the
    /// window rather than pinned to the bottom of a blank page.
    private var isEmptyConversation: Bool {
        (session?.messages.isEmpty ?? true) && composerState != .generatingHere
    }

    /// Greeting + discovery chips, one fixed-height block. The vertical slack
    /// lives OUTSIDE this view (two sibling Spacers in the body) — a Spacer
    /// nested in here shares space unevenly with the body's own trailing one,
    /// which is what pinned the whole group to the bottom of the window.
    private var emptyState: some View {
        VStack(spacing: 8) {
            Text("How can I help you today?")
                .font(.system(size: 30, weight: .semibold, design: .rounded))
                // Subtle top-to-bottom fade for depth; primary-based so it
                // reads in both appearances without picking a color.
                .foregroundStyle(LinearGradient(
                    colors: [.primary, .primary.opacity(0.55)],
                    startPoint: .top, endPoint: .bottom))
            if server.status != .running {
                Text("Start the server to begin.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else if let agent = activeAgent {
                Text("Talking to \(agent.name)")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            // Discovery chips: the features that otherwise live only in the
            // menu-bar tray (media generation, Model Browser, Tasks, the CLI
            // launcher). Under the greeting, gone once the conversation starts.
            if session?.isExternalBridge != true {
                EmptyStateChipRow()
                    .padding(.top, 18)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, ChatMetrics.gutter)
        .padding(.bottom, 14)
    }

    var body: some View {
        VStack(spacing: 0) {
            if isEmptyConversation {
                // Two SIBLING spacers (this one + the trailing one below the
                // composer) split the slack evenly, so greeting + chips +
                // composer sit as one group in the middle of the window.
                Spacer(minLength: 0)
                emptyState
            } else {
            // Messages
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(ChatRowBuilder.rows(from: session?.messages ?? [])) { row in
                            switch row {
                            case .message(let m):
                                MessageBubble(
                                    message: m,
                                    sources: sourcesFor(m),
                                    onIncreaseContext: {
                                        AppActivation.openWindow(id: "settings", using: openWindow)
                                    },
                                    onDelete: {
                                        appState.deleteMessage(in: sessionId, messageId: m.id)
                                    })
                                .id(m.id)
                            case .toolCall(let call, let results):
                                ToolCallRow(call: call, results: results).id(call.id)
                            }
                        }
                        // Live media generation, under the tool-call row that
                        // started it. These block chat decode on the one GPU for
                        // anything from seconds to minutes, so the alternative is
                        // a window that looks frozen. Only in the chat whose turn
                        // ASKED for it — with concurrent turns, ownership rides
                        // `mediaProgressSessionId`, not just "is generating".
                        if composerState == .generatingHere,
                           chatEngine.mediaProgressSessionId == sessionId,
                           let progress = chatEngine.mediaProgress {
                            MediaProgressCard(progress: progress)
                                .id("mediaProgress")
                        }
                    }
                    .padding(ChatMetrics.gutter)
                }
                // Transcript text used to run straight into the floating model
                // picker. The toolbar band's own full-width background stays
                // hidden (the cluster carries its own material — that's what
                // keeps content from bleeding THROUGH the controls); this is
                // the other half, frosting the content as it passes UNDER them,
                // drawn by the scroll view itself so nothing new can intercept
                // a click.
                .scrollEdgeEffectStyle(.soft, for: .top)
                // The transcript is moved from exactly one place — `applyScroll`
                // — and only ever by a decision `ChatScrollState` made.
                .scrollPosition($scrollPosition)
                // While following, the scroll view keeps its own bottom edge
                // glued as the content grows, so a streamed token costs NOTHING:
                // no scroll command, no second layout pass, no state change.
                // Measured over 40 growths (~/claude-tmp/chatscroll-probe):
                // this anchor emitted ONE geometry event, while re-issuing
                // `scrollTo(edge:)` per growth emitted two per token (the drift,
                // then the correction). Flipping the value to nil starts the
                // drift again, which is how the anchor is known to react to a
                // CHANGING value rather than being read once at creation — so
                // the moment the user takes over, growth stops dragging them.
                .defaultScrollAnchor(scrollModel.isPinnedToBottom ? .bottom : nil,
                                     for: .sizeChanges)
                // The scroll view's OWN geometry says how far the end sits below
                // the fold. This replaces a preference key published by a 1pt
                // anchor view, which raced the layout it was measuring and could
                // not see the viewport at all without a second preference key.
                .onScrollGeometryChange(for: CGFloat.self) {
                    ChatScrollState.distanceFromBottom($0)
                } action: { _, distance in
                    applyScroll(.geometryChanged(distanceFromBottom: distance))
                }
                // Who is moving it. The predecessor was an app-global NSEvent
                // scroll-wheel monitor: it fired for every other window in the
                // app, disengaged on a single upward notch (including the
                // rubber-band settle after flinging TO the bottom, which is why
                // auto-follow so often refused to come back), and was blind to
                // scroller drags, keyboard scrolling and window resizes.
                .onScrollPhaseChange { _, phase in
                    applyScroll(.driverChanged(ChatScrollDriver(phase)))
                }
                .overlay(alignment: .bottom) {
                    // The old affordance was a 4pt accent strip with hit-testing
                    // off: it reported the state and offered no way out of it.
                    Group {
                        if !scrollModel.isPinnedToBottom {
                            jumpToLatestButton
                                .transition(.opacity.combined(with: .scale(scale: 0.85)))
                        }
                    }
                    .animation(.easeInOut(duration: 0.18), value: scrollModel.isPinnedToBottom)
                }
            // The divider belongs to the transcript — against the empty
            // state's greeting it would draw a line across mid-window.
            Divider()
            }   // end else (non-empty conversation)

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
                        // Detach is a user decision — drop the persisted pick
                        // too, or the folder would re-attach on next launch.
                        if let idx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }) {
                            appState.chatSessions[idx].attachedFolderPath = nil
                        }
                        SecurityScopedBookmark.clear(
                            name: SecurityScopedBookmark.attachedFolderName(sessionId))
                    }
                }

                // Voice mode lives INLINE: a talking orb just above the input,
                // not a sheet over the transcript (the sheet hid the
                // conversation and duplicated the composer's own toggles).
                // Renders nothing while voice is off.
                VoiceOrbView(controller: appState.voice, sessionId: sessionId)

                // One rounded container, two rows: the input on top with the
                // full width of the column, its controls beneath — inside the
                // same border, so they read as belonging to it.
                //
                // They used to sit BESIDE the input, which cost the field ~200pt
                // of width (most of a line) on every message, and that cost grew
                // with each new control. Stacked, the field is as wide as the
                // transcript it answers.
                VStack(alignment: .leading, spacing: 6) {
                    composerField
                    composerControls
                }
                .padding(8)
                .background(Color(nsColor: .textBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 18))
                .overlay(
                    RoundedRectangle(cornerRadius: 18)
                        .stroke(Color.secondary.opacity(0.25), lineWidth: 0.5)
                )
                // Hover cards for the row's bare glyphs. Drawn HERE, past the
                // clip: an overlay on the control itself is cut off at the
                // container's rounded edge and lands over the text field.
                .composerTipOverlay()
              }   // end else (non-Telegram composer)
            }
            .padding(.horizontal, ChatMetrics.gutter)
            .padding(.vertical, 8)
            // The top spacer's sibling — see the empty-state branch above.
            if isEmptyConversation { Spacer(minLength: 0) }
        }
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
        // ONE toolbar item — see `floatingToolbar` for why a real ToolbarItem is
        // the only clickable resident of that band, and why its width stays
        // bounded (the » eviction gotcha).
        .toolbar {
            if #available(macOS 26.0, *) {
                // Liquid Glass gives every toolbar item its own capsule
                // background + border; hide it so the controls float bare on
                // the content, matching the sidebar's seamless look.
                //
                // The model picker rides its OWN item at the LEADING edge, not
                // the trailing cluster: that cluster is at its width budget and
                // a model name is runtime-variable, which is exactly what
                // re-triggers » eviction. Its label is width-capped for the
                // same reason (`ChatModelPill.maxNameWidth`).
                ToolbarItem(placement: .navigation) {
                    floatingToolbar
                }
                .sharedBackgroundVisibility(.hidden)
            } else {
                ToolbarItem(placement: .navigation) {
                    floatingToolbar
                }
            }
        }
        // The window's own toolbar material, full width. This was hidden for a
        // while — the floating cluster carries its own material, so the band
        // read as a divider above a mostly empty bar. But hiding it left the
        // transcript running to the window's top edge with nothing between the
        // two: text clipped mid-line under the model picker (live 2026-07-30),
        // and `scrollEdgeEffectStyle` had no bar to attach to, so it drew
        // nothing. The system material IS the 100%-width surface here — the
        // hand-drawn strip that predated it is what must not come back.
        .toolbarBackground(.visible, for: .windowToolbar)
        .sheet(isPresented: $showMCPMarketplace) {
            MCPMarketplaceView()
                .environmentObject(mcpManager)
        }
        .alert("Enable thinking with Tools on?", isPresented: $showThinkingInAgentConfirm) {
            Button("Cancel", role: .cancel) { }
            Button("Enable anyway") { enableThinking = true }
        } message: {
            Text("Thinking is not recommended with Tools on — most local models tool-call more reliably without it. Do you still want to enable it?")
        }
        // Typed-turn approvals only. Voice turns approve through the
        // controller's own `pendingApproval`, rendered inline next to the orb
        // (and in the tray) — never through this sheet.
        .sheet(item: $pendingApproval) { req in
            ToolApprovalSheet(request: req,
                              onAllow: { resolveApproval(.allow) },
                              onDeny: { resolveApproval(.deny) },
                              onAllowAll: { resolveApproval(.allow, allowAll: true) })
        }
        .onAppear {
            inputFocused = true
            syncTogglesFromSession()
            restoreAttachedFolderIfNeeded()
            applyScroll(.transcriptShown)
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
            // paste monitor.
            if let monitor = pasteMonitor {
                NSEvent.removeMonitor(monitor)
                pasteMonitor = nil
            }
        }
        // Pre-send nudge to enable Agent / MCP mode when the message looks like
        // it needs it. "Send Anyway" suppresses the suggestion for this chat.
        .confirmationDialog(
            pendingIntentPrompt == .mcp ? "Enable MCP first?" : "Turn Tools on first?",
            isPresented: Binding(get: { pendingIntentPrompt != nil },
                                 set: { if !$0 { pendingIntentPrompt = nil } }),
            titleVisibility: .visible,
            presenting: pendingIntentPrompt
        ) { prompt in
            Button(prompt == .mcp ? "Enable MCP & Send" : "Turn Tools On & Send") {
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
                 : "This looks like a task for the agent (creating files, running commands, browsing the web…), but Tools is off. Turn it on so the model can use them?")
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
        .onChange(of: composerState) { _, state in
            if state == .idle { inputFocused = true }
        }
        .onChange(of: sessionId) { _, _ in
            // The view is reused across tabs, so reload the toolbar toggles from
            // the newly-visible session. The allow-list is NOT reset here — it's
            // keyed by session id, so each tab keeps its own decision across
            // switches (a session re-arms only when its Agent toggle goes off).
            syncTogglesFromSession()
            restoreAttachedFolderIfNeeded()
            // Scroll state is per-view, and the view is reused across tabs — so
            // without this, leaving one chat scrolled up opened the next one
            // unpinned at whatever offset the previous conversation's content
            // happened to leave behind.
            applyScroll(.transcriptShown)
        }
    }

    /// The input field. No background or border of its own — the composer
    /// container draws those around both rows. NSTextView-backed so a big paste
    /// stays smooth and the mouse wheel scrolls once it grows past the cap.
    private var composerField: some View {
        GrowingTextEditor(text: $inputText,
                          isFocused: $inputFocused,
                          measuredHeight: $composerHeight,
                          isIdle: composerState == .idle,
                          onSend: { sendMessage() })
            .frame(height: max(ChatMetrics.composerMinHeight, composerHeight))
            .padding(.horizontal, 5)
            .disabled(server.status != .running)
            .overlay(alignment: .topLeading) {
                if inputText.isEmpty {
                    Text("Ask me anything…")
                        .font(.body)
                        .foregroundStyle(.secondary)
                        .padding(.leading, 9)
                        .padding(.top, 8)
                        .allowsHitTesting(false)
                }
            }
    }

    /// What this turn will run with on the left; what it will cost, and Send,
    /// on the right. Its own property because the composer container otherwise
    /// blows the type-checker's budget.
    @ViewBuilder
    private var composerControls: some View {
        HStack(spacing: 6) {
        attachmentMenu

        // Mic — only on models that actually understand audio
        // (Gemma 4 12B). Tap to record, tap again to attach.
        if audioSupported {
            MicButton(recorder: recorder) { toggleRecording() }
                .disabled(server.status != .running || composerState == .generatingHere)
        }

        // Think / Tools / MCP. Icon-only, and here rather than in the window
        // toolbar: they configure the MESSAGE being written, not the window,
        // and their captions were most of the toolbar cluster's width budget.
        thinkToggle
        agentToggle
        mcpToggle

        Spacer(minLength: 8)

        // Context gauge, immediately left of Send — the control the
        // reading is about. Bounded width (a percentage plus a ring)
        // so it can't push the row around as it changes.
        if showsContextPill {
            ContextPill(stats: contextStats,
                        modelName: server.chatModelInfo?.name,
                        decodeSpeed: lastDecodeSpeed,
                        isLive: composerState == .generatingHere)
                .frame(height: ChatMetrics.composerControlSize)
        }

        // Voice mode, between the context gauge and Send. A toggle: on starts
        // hands-free with this chat's toggles/agent, off ends it. Its own
        // observing view (see `VoiceComposerToggle`) so the tint follows the
        // app-level controller when voice starts from the tray.
        voiceToggle

        Button {
            if composerState == .generatingHere {
                chatEngine.stop(sessionId: sessionId)
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
        // disabled when the server is down or when this chat has nothing to
        // send. Another chat's turn blocks nothing — the engine is multi-turn.
        .disabled(server.status != .running
                  || (composerState == .idle
                      && inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                      && pendingImages.isEmpty && pendingPDFs.isEmpty && pendingAudio.isEmpty))
        }
    }

    /// Composer-row voice toggle — see `VoiceComposerToggle` for why it's an
    /// observing child view rather than a Button reading `appState.voice` here.
    private var voiceToggle: some View {
        VoiceComposerToggle(controller: appState.voice, sessionId: sessionId,
                            disabled: server.status != .running) { startVoiceMode() }
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
        AppActivation.beginPanel(panel) { response in
            guard response == .OK, let url = panel.url else { return }
            DispatchQueue.main.async { attachDocumentFolder(url) }
        }
    }

    /// Index a folder for mini-RAG. Shared by the folder picker and paste/drop so
    /// every entry point behaves identically. Embeds on the local server's GPU;
    /// auto-downloads the default encoder model (35 MB, one-time) when none is
    /// available. Server down → lexical-only retrieval. Must run on the main actor.
    ///
    /// The pick is persisted (path on the session, security-scoped bookmark in
    /// defaults) so the index can be rebuilt after a relaunch — under the App
    /// Sandbox the panel's grant dies with the process; the bookmark is what
    /// makes the folder reachable again. See `restoreAttachedFolderIfNeeded`.
    private func attachDocumentFolder(_ url: URL) {
        SecurityScopedBookmark.store(url, name: SecurityScopedBookmark.attachedFolderName(sessionId))
        if let idx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }) {
            appState.chatSessions[idx].attachedFolderPath = url.path
        }
        appState.documentIndexes[sessionId]?.cancel()
        let index = DocumentIndex(folderURL: url,
                                  embedderProvider: ServerEmbedding.autoProvider(port: server.port))
        appState.documentIndexes[sessionId] = index
        index.startIndexing()
    }

    /// Rebuild a persisted attached folder's index after a relaunch. The view is
    /// reused across tabs, so this runs on appear AND on every session change;
    /// a session with a live index (or none attached) is a no-op.
    private func restoreAttachedFolderIfNeeded() {
        guard !isExternalBridgeSession,
              appState.documentIndexes[sessionId] == nil,
              let path = session?.attachedFolderPath else { return }
        // Resolve the bookmark first (starts the sandbox grant). A missing or
        // dead bookmark (DMG build, folder relocated) falls back to the raw
        // path — outside the sandbox it is directly accessible anyway.
        let url = SecurityScopedBookmark.startAccessOnce(
            name: SecurityScopedBookmark.attachedFolderName(sessionId))
            ?? URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else { return }
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
        AppActivation.beginPanel(panel) { response in
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
    private var audioSupported: Bool { server.chatModelInfo?.supportsAudio ?? false }

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

    /// Route one event through the decision core and carry out whatever it asks
    /// for. The ONLY place in this view that moves the transcript.
    private func applyScroll(_ event: ChatScrollEvent) {
        switch scrollModel.apply(event) {
        case .none:
            break
        case .toBottom(let animated):
            if animated {
                // A discrete jump the user asked for (their own message, the
                // button) — the movement is the feedback that it landed.
                withAnimation(.easeOut(duration: 0.2)) {
                    scrollPosition.scrollTo(edge: .bottom)
                }
            } else {
                // Following the stream is a direct offset set, explicitly
                // unanimated: this used to run a 0.15s `withAnimation` per
                // STREAMED TOKEN, so dozens of animations a second each started
                // over the top of the one still running. That is the stutter,
                // and it got worse the longer the transcript grew.
                var instant = Transaction()
                instant.disablesAnimations = true
                withTransaction(instant) {
                    scrollPosition.scrollTo(edge: .bottom)
                }
            }
        }
    }

    /// Shown only while auto-follow is off — its absence is how the transcript
    /// says it is already following.
    private var jumpToLatestButton: some View {
        Button { applyScroll(.jumpTapped) } label: {
            Image(systemName: "arrow.down")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.primary)
                .frame(width: 28, height: 28)
                .background(.regularMaterial, in: Circle())
                .overlay(Circle().strokeBorder(.separator, lineWidth: 0.5))
        }
        .buttonStyle(.plain)
        .help("Jump to the latest message")
        .padding(.bottom, 12)
    }


    /// Latest context usage from the most recent assistant message with token
    /// data — the prompt size + the reply's length of the last completed turn.
    private var contextUsage: (promptTokens: Int, completionTokens: Int, contextLength: Int)? {
        guard let messages = session?.messages else { return nil }
        if let last = messages.last(where: { $0.promptTokens != nil && $0.promptTokens! > 0 }) {
            let ctxLen = AgentEngine.effectiveContextLength(
                appContextSize: appState.contextSize,
                modelContextLength: server.chatModelInfo?.contextLength
            )
            return (promptTokens: last.promptTokens!, completionTokens: last.completionTokens ?? 0, contextLength: ctxLen)
        }
        return nil
    }

    /// Pages a web search backed this reply with. Only computed for a finished
    /// assistant reply — tool-call summaries and user turns have no provenance
    /// to show, and the walk is bounded by the previous user message so this
    /// stays cheap per row.
    private func sourcesFor(_ message: ChatMessage) -> [WebSource] {
        guard message.role == .assistant, !message.isAgentSummary, !message.isStreaming,
              let messages = session?.messages else { return [] }
        return WebSourceExtractor.sources(forMessageId: message.id, in: messages)
    }

    /// The context-overflow notice from the turn that just ended, if that's how
    /// it ended. Scoped to the LAST message on purpose: an overflow the user has
    /// since worked around (shorter prompt, tools off) must stop colouring the
    /// pill red, or the gauge keeps reporting a failure that no longer applies.
    private var lastOverflowNotice: ChatErrorNotice? {
        guard let notice = session?.messages.last?.errorNotice,
              notice.kind == .contextOverflow else { return nil }
        return notice
    }

    /// Reading behind the composer's context pill.
    private var contextStats: ContextWindowStats {
        let usage = contextUsage
        return ContextWindowStats.make(
            promptTokens: usage?.promptTokens ?? 0,
            completionTokens: usage?.completionTokens ?? 0,
            liveTokens: composerState == .generatingHere ? chatEngine.liveCompletionTokens(for: sessionId) : 0,
            contextLength: usage?.contextLength
                ?? AgentEngine.effectiveContextLength(appContextSize: appState.contextSize,
                                                      modelContextLength: server.chatModelInfo?.contextLength),
            overflow: lastOverflowNotice)
    }

    /// Decode speed of the most recent reply that was actually timed.
    ///
    /// Mirrors `contextUsage`'s shape above: the LAST message carrying a real
    /// figure, not the last message — a turn that errored or is still streaming
    /// has none, and falling back to "no reading" there would blank a number the
    /// previous reply legitimately produced.
    private var lastDecodeSpeed: Double? {
        session?.messages.last { ($0.tokensPerSecond ?? 0) > 0 }?.tokensPerSecond
    }

    /// Hidden until there's something true to report — a pill reading 0.0%
    /// before the first reply is noise, not information.
    private var showsContextPill: Bool {
        contextUsage != nil || composerState == .generatingHere || lastOverflowNotice != nil
    }

    private var workingDirectoryBinding: Binding<String?> {
        Binding(
            get: { appState.chatSessions.first { $0.id == sessionId }?.workingDirectory },
            set: { newValue in
                if let idx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }) {
                    appState.chatSessions[idx].workingDirectory = newValue
                    // Persist the panel's grant: without a security-scoped
                    // bookmark, a folder outside the container is unreachable
                    // after relaunch under the App Sandbox. Resolved at the
                    // agent-turn seam (ChatTurnEngine.runAgentTurn).
                    let slot = SecurityScopedBookmark.workingFolderName(sessionId)
                    if let dir = newValue {
                        appState.agentMemory.recordDirectory(dir)
                        SecurityScopedBookmark.store(URL(fileURLWithPath: dir), name: slot)
                    } else {
                        SecurityScopedBookmark.clear(name: slot)
                    }
                    // No eager remount: /workspace stays the Settings default
                    // (pi/hermes live there). This chat's folder is hot-mounted
                    // at /projects/<slug> the first time a tool runs — no VM
                    // reboot, so live CLI sessions are never torn down. Only a
                    // Settings default change remounts /workspace.
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

    /// Start hands-free voice from the composer toggle. The controller is
    /// app-level; the toggle already ended any voice running elsewhere (the
    /// "move it here" click), so by the time this runs the mic is free. The
    /// orb renders inline in the BOUND session's tab — nothing to "present".
    private func startVoiceMode() {
        guard !appState.voice.isActive else { return }
        // Sync the voice toggles to the chat session being opened — talking should
        // start in the same Think/Tools/MCP mode as the chat you launched it from.
        if let s = session {
            appState.voice.enableThinking = s.enableThinking
            appState.voice.agentMode = s.mode == .agent
            appState.voice.mcpMode = s.useMCP
            // …and to the same AGENT. Voice routes each turn into its agent's own
            // thread, so without this a tray default of "Chef" would pull the
            // conversation out of the tab you launched voice from.
            appState.defaultAgentId = s.agentId
        }
        appState.sessionForAgent(session?.agentId)
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
        if composerState != .generatingHere, server.status == .running, !trimmed.isEmpty,
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
    ///
    /// A mode the tab's agent decides is never nudged about: accepting would
    /// change nothing and the message would send exactly as it was.
    private func detectIntentPrompt(for text: String) -> IntentPrompt? {
        let toggles = toolbarToggles
        let servers = enabledMCPServerNames()
        if !mcpMode, toggles.mcpLockedBy == nil, !servers.isEmpty,
           !intentSuppress.isSuppressed(.mcp, for: sessionId),
           ComposerIntent.wantsMCP(text, serverNames: servers) {
            return .mcp
        }
        if !isAgentMode, toggles.toolsLockedBy == nil,
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
        var text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        let attachedImages = consumePendingImages()
        let attachedAudio = consumePendingAudio()
        let pdfText = consumePendingPDFsAsText()
        guard !text.isEmpty || attachedImages != nil || attachedAudio != nil || !pdfText.isEmpty,
              composerState != .generatingHere, server.status == .running else { return }
        inputText = ""
        if !pdfText.isEmpty {
            text = text.isEmpty ? pdfText : pdfText + "\n\n" + text
        }

        // The toolbar toggles are this surface's DEFAULTS; the tab's agent (if
        // any) overrides what it declared. One builder, so no field is read from
        // a global here — see ChatTurnEngine.TurnConfig.
        let resolved = appState.resolvedAgentSettings(
            agentId: session?.agentId,
            toolsEnabled: isAgentMode,
            mcpEnabled: mcpMode,
            thinkingEnabled: enableThinking,
            autoApprove: false,
            workingDirectory: session?.workingDirectory,
            disabledTools: ChatSession.disabledToolKinds(session?.disabledTools ?? []))
        let config = ChatTurnEngine.TurnConfig.from(
            resolved, documentIndex: appState.documentIndexes[sessionId])
        chatEngine.runTurn(sessionId: sessionId, userText: text,
                           images: attachedImages, audio: attachedAudio,
                           config: config,
                           approval: { await requestToolApproval($0) })
        // Your own message always wins: sending from halfway up the history used
        // to leave you exactly there, because auto-follow was off and nothing
        // else scrolled.
        applyScroll(.userSentMessage)
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

/// What the chat considers "occupied context".
///
/// This was a full-width bar above the composer; it is now the composer's
/// `ContextPill`, which shows the same reading in one control's width and puts
/// the breakdown in a popover. Only the summing rule stayed here — the bar's
/// clamped ratio and colour bands moved to `ContextWindowStats`, which needs an
/// UNCLAMPED figure so it can display 100.3%.
enum ContextMonitor {
    /// Total context occupied right now: the last completed turn (prompt + its
    /// reply) plus the in-flight reply's running count. Pure → ContextMonitorTests.
    static func usedTokens(promptTokens: Int, completionTokens: Int, liveTokens: Int) -> Int {
        promptTokens + completionTokens + liveTokens
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

// `SystemMetrics` (GPU utilization, memory pressure, and the libproc/Mach
// replacements for lsof/ps/vm_stat) lives in Services/SystemMetrics.swift.

// MARK: - Message Bubble

struct MessageBubble: View {
    let message: ChatMessage
    /// Pages a web search backed this reply with. Empty for every reply that
    /// didn't search, which is the normal case — the chip only exists when
    /// there is provenance to show.
    var sources: [WebSource] = []
    /// Opens Settings at the context control, for the overflow card's button.
    /// Defaults to a no-op so surfaces that only DISPLAY a transcript (the task
    /// run viewer) don't have to route an action they have no window for.
    var onIncreaseContext: () -> Void = {}
    /// Removes this message from the conversation. nil on read-only surfaces —
    /// a task run's transcript is a record, so it has no delete affordance
    /// rather than one that silently does nothing.
    var onDelete: (() -> Void)?
    /// Explicit so the accordion HEADER can drive it, not just the chevron.
    @State private var thinkingExpanded = false

    var body: some View {
        // A failure notice is not model output: it renders as its own card
        // across the column, never inside an assistant bubble.
        if let notice = message.errorNotice {
            ChatErrorCard(notice: notice, onIncreaseContext: onIncreaseContext)
        } else {
            messageBody
        }
    }

    /// Reasoning accordion. The WHOLE header toggles, not just the chevron:
    /// macOS only hit-tests the disclosure triangle on a DisclosureGroup's
    /// label, so the "Thinking" text was a dead click target — same fix as the
    /// Agents editor's Advanced row (the label holds no buttons of its own, so
    /// a tap gesture here can't swallow child clicks).
    @ViewBuilder
    private var thinkingBlock: some View {
        if let reasoning = message.reasoningContent, !reasoning.isEmpty {
            DisclosureGroup(isExpanded: $thinkingExpanded) {
                Text(reasoning)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } label: {
                Label("Thinking", systemImage: "brain")
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.secondary)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        withAnimation(.easeInOut(duration: 0.15)) { thinkingExpanded.toggle() }
                    }
            }
            .padding(8)
            .background(.quaternary.opacity(0.5))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    private var messageBody: some View {
        HStack(alignment: .top, spacing: 10) {
            if message.role == .user { Spacer(minLength: 60) }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                // Reasoning (collapsible)
                thinkingBlock

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

                // Generated tracks / clips, attached by path (see ChatMediaRef).
                if let media = message.media, !media.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(media) { ref in
                            ChatMediaAttachmentView(ref: ref)
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

                // Content.
                //
                // Only the USER gets a bubble. An assistant reply is the page's
                // main content — long, formatted, full of code blocks and
                // tables — and boxing it wastes the column's width and fights
                // every block that wants the full measure. A tool-call summary
                // keeps its card so it still reads as machinery, not prose.
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
                    .padding(.horizontal, isBare ? 0 : ChatMetrics.bubblePaddingH)
                    .padding(.vertical, isBare ? 0 : ChatMetrics.bubblePaddingV)
                    .background(bubbleBackground)
                    .foregroundStyle(message.role == .user ? .white : .primary)
                    .clipShape(RoundedRectangle(cornerRadius: isBare ? 0 : ChatMetrics.bubbleCornerRadius))
                    .frame(maxWidth: .infinity, alignment: isBare ? .leading : .trailing)
                }

                // Where the answer came from, above the footer — the provenance
                // belongs with the reply, not with its timings.
                if message.role == .assistant, !message.isStreaming, !sources.isEmpty {
                    WebSourcesChip(sources: sources)
                        .padding(.leading, isBare ? 0 : ChatMetrics.statsIndent)
                }

                if showsFooter { footer }
            }

            if message.role == .assistant { Spacer(minLength: 60) }
        }
        .contextMenu {
            Button("Copy Message") { copyMessage() }
            if onDelete != nil {
                Button("Delete Message", role: .destructive) { onDelete?() }
            }
        }
    }

    // MARK: - Bubble vs bare

    /// Assistant prose renders bare; user turns and tool-call summaries keep a
    /// bubble.
    private var isBare: Bool { message.role == .assistant && !message.isAgentSummary }

    private var bubbleBackground: Color {
        if isBare { return .clear }
        return message.role == .user ? Color.accentColor : Color(.controlBackgroundColor)
    }

    // MARK: - Footer (timestamp · actions · stats)

    private var showsFooter: Bool {
        message.role == .assistant && !message.isStreaming
            && !message.isAgentSummary && !message.content.isEmpty
    }

    /// Timestamp and token stats on the left, actions pinned to the right.
    ///
    /// The actions are ALWAYS visible. Fading them in on hover meant they moved
    /// under the pointer as the row re-rendered mid-stream and were awkward to
    /// hit; two small tertiary glyphs cost nothing to leave on. They sit at the
    /// trailing edge — past the stats, against the reply's right edge — so the
    /// eye finds them in the same place on every message instead of at a spot
    /// that shifts with the length of the stats text.
    private var footer: some View {
        HStack(spacing: 8) {
            Text(message.timestamp.formatted(date: .abbreviated, time: .shortened))
                .font(.caption2)
                .foregroundStyle(.tertiary)

            if let tps = message.tokensPerSecond, tps > 0 {
                Label("\(Int(tps)) tokens/sec", systemImage: "gauge.with.needle")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            if let completion = message.completionTokens {
                Text("(\(completion) tokens)")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            Spacer(minLength: 8)

            HStack(spacing: 2) {
                footerButton("doc.on.doc", help: "Copy this reply") { copyMessage() }
                if let onDelete {
                    footerButton("trash", help: "Delete this message from the conversation",
                                 action: onDelete)
                }
            }
        }
        .padding(.leading, isBare ? 0 : ChatMetrics.statsIndent)
        .padding(.top, 2)
    }

    private func footerButton(_ icon: String, help: String,
                              action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .frame(width: 20, height: 18)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(help)
    }

    private func copyMessage() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(message.content, forType: .string)
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
    /// Who decided each control, when it isn't the chat itself — the agent's
    /// name, for the lock ring and the hover card. nil = the chat's own toggle.
    var thinkingLockedBy: String? = nil
    var toolsLockedBy: String? = nil
    var mcpLockedBy: String? = nil

    var isLocked: Bool { thinkingLockedBy != nil || toolsLockedBy != nil || mcpLockedBy != nil }

    static func resolve(isExternalBridge: Bool,
                        telegramThinking: Bool, telegramAgent: Bool, telegramMCP: Bool,
                        inAppThinking: Bool, inAppAgent: Bool, inAppMCP: Bool,
                        agentLock: AgentModeLock? = nil) -> ChatModeToggles {
        let base = isExternalBridge
            ? ChatModeToggles(thinking: telegramThinking, agent: telegramAgent, mcp: telegramMCP)
            : ChatModeToggles(thinking: inAppThinking, agent: inAppAgent, mcp: inAppMCP)
        guard let lock = agentLock else { return base }
        return ChatModeToggles(
            // Thinking is the one an agent may leave unset, and `AgentResolution`
            // falls back to the surface's own value there — so locking it anyway
            // would take away a control nobody is deciding for you.
            thinking: lock.thinking ?? base.thinking,
            agent: lock.tools,
            mcp: lock.mcp,
            thinkingLockedBy: lock.thinking == nil ? nil : lock.name,
            toolsLockedBy: lock.name,
            mcpLockedBy: lock.name)
    }
}

/// What the chat's agent decided about Think / Tools / MCP.
///
/// `AgentResolution` takes Tools and MCP straight from the agent's capabilities
/// and ignores the chat's own toggles — so without this the discs showed one
/// thing and the turn ran another (every agent defaults `web: true`, which forces
/// the tool loop on while the wrench renders OFF). Built from the SAME resolution
/// the turn uses (`ChatDetailView.agentModeLock`), so the two can't drift.
struct AgentModeLock: Equatable {
    var name: String
    /// nil = the agent left thinking unset; the chat's own toggle stands.
    var thinking: Bool?
    var tools: Bool
    var mcp: Bool
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
        // Fenced code renders as its own view (gutter, colors, copy button);
        // everything between fences stays in ONE text view per run so
        // drag-selection still crosses paragraphs, lists and tables. See
        // `MarkdownSegmenter` for why the split is at fences, not at blocks.
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(MarkdownSegmenter.segments(source).enumerated()), id: \.offset) { _, segment in
                switch segment {
                case .prose(let text):
                    SelectableMarkdownNSText(attributed: Self.attributedString(for: text))
                case .code(let language, let code):
                    CodeBlockView(language: language, code: code)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
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

