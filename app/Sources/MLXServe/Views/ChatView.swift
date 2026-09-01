import SwiftUI
import PDFKit
import UniformTypeIdentifiers
import AppKit
import SwaTexRender

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
    @Binding var images: [PendingImage]
    @Binding var pdfs: [(name: String, text: String)]
    @Binding var videos: [ChatVideo]
    @Binding var audio: [ChatAudio]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(Array(images.enumerated()), id: \.offset) { idx, pending in
                    imageChip(idx: idx, img: pending.image)
                }
                ForEach(Array(pdfs.enumerated()), id: \.offset) { idx, pdf in
                    fileChip(idx: idx, name: pdf.name, detail: "PDF · \(pdf.text.count) chars",
                             icon: "doc.text.fill", tint: .red) { pdfs.remove(at: idx) }
                }
                ForEach(Array(videos.enumerated()), id: \.offset) { idx, vid in
                    fileChip(idx: idx, name: vid.name, detail: "Video · \(vid.frameCount) frames",
                             icon: "video.fill", tint: .orange) { videos.remove(at: idx) }
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
    case folder, pdf, audio, video, image, unhandled

    static func classify(ext: String, isDirectory: Bool, audioSupported: Bool, videoSupported: Bool = false) -> PasteFileKind {
        if isDirectory { return .folder }
        let e = ext.lowercased()
        if e == "pdf" { return .pdf }
        if let ut = UTType(filenameExtension: e) {
            if ut.conforms(to: .audio) { return audioSupported ? .audio : .unhandled }
            if ut.conforms(to: .movie) { return videoSupported ? .video : .unhandled }
            if ut.conforms(to: .image) { return .image }
        }
        return .unhandled
    }
}

struct ChatView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @Environment(\.dismissWindow) private var dismissWindow
    /// The two-column (chat) split's visibility.
    @State private var columnVisibility = NavigationSplitViewVisibility.automatic
    /// The three-column (Tasks / Agents) split's visibility. `.all` is the only
    /// value that means "show all three" — the state above cannot supply it.
    @State private var tasksColumnVisibility = NavigationSplitViewVisibility.all
    /// The Agents pane's editing state, owned here so it survives while the
    /// user moves between agents. The standalone Agents window owns its own —
    /// two surfaces editing one draft would fight over it.
    @StateObject private var agentsModel = AgentsWorkspaceModel()
    /// Flipped by the gate sheet's Cancel, and by nothing else.
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
        // Tasks gets a THIRD column: its list belongs beside the app's sidebar,
        // not inside the content area — a list of tasks is navigation, and
        // nesting it in the detail column made the window look like it had two
        // unrelated sidebars stacked horizontally.
        Group {
            if appState.chatWorkspace.isThreeColumn {
                threeColumnSplitView
            } else {
                standardSplitView
            }
        }
        // ⌘L, on the WINDOW rather than on one split: the picker has to open
        // over Tasks and Agents too, which are the other `NavigationSplitView`.
        // Environment injected AT the sheet — a sheet inherits none.
        .sheet(isPresented: $appState.modelPalettePresented) {
            ModelPaletteSheet()
                .environmentObject(appState)
                .environmentObject(server)
        }
    }

    /// The three-column modes (Tasks, Agents), in ONE split view.
    @ViewBuilder
    private var threeColumnSplitView: some View {
        NavigationSplitView(columnVisibility: $tasksColumnVisibility) {
            ChatSidebar()
                .navigationSplitViewColumnWidth(min: 180, ideal: 220, max: 280)
        } content: {
            // A pane TYPE, never `SomeView().someProperty`: an environment
            // reader has to be the column itself, or its @EnvironmentObject is
            // read out of a value SwiftUI never installed (see `TaskListPane`).
            Group {
                if appState.chatWorkspace.isAgents {
                    AgentListPane(model: agentsModel)
                } else {
                    TaskListPane()
                }
            }
            .navigationSplitViewColumnWidth(min: 240, ideal: 280, max: 340)
        } detail: {
            Group {
                if appState.chatWorkspace.isAgents {
                    AgentDetailPane(model: agentsModel)
                } else {
                    TaskDetailPane()
                }
            }
        }
        .navigationTitle("")
        // Nothing lives in the toolbar here — each pane draws its own title
        // row — so the band carries no material. Its BAR still has to exist:
        .toolbarBackground(.hidden, for: .windowToolbar)
        .onAppear { AppActivation.focus() }
    }

    private var standardSplitView: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            ChatSidebar()
                .navigationSplitViewColumnWidth(min: 180, ideal: 220, max: 280)
        } detail: {
            // Two modes in one column: the transcript, or the model browser
            // (`ChatWorkspace`). The browser used to be its own Window, so
            // every route to it was a route OUT of this one — and a window the
            // user then had to find their own way back from.
            if case .models(let section) = appState.chatWorkspace {
                // Content only — the sections and the way back are the SIDEBAR
                // while this mode is up (`ChatSidebar.modelsRow`).
                ModelBrowserPane(section: Binding(
                    get: { section },
                    set: { appState.selectModelSection($0) }))
            } else if appState.chatWorkspace.isSettings {
                SettingsView()
            } else if case .create(let experiment) = appState.chatWorkspace {
                // The four generators were four Window scenes; they are pages
                // of this window now. Each keeps its own view untouched — only
                // the hosting moved.
                createPane(experiment)
            } else if let sessionId = appState.activeChatId,
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
        // No toolbar material in ANY of this split's modes — the transcript,
        // Models, Settings and the Create pages alike (the three-column split
        // carries the same modifier). On the SPLIT, not on ChatDetailView:
        // there it covered only conversation mode, and the chrome flipped as
        // you switched panes. The BAR itself stays — the traffic lights and
        // the sidebar-collapse button live in it (live 2026-08-09); see the
        // long note above `threeColumnSplitView` and ChatDetailView's body.
        .toolbarBackground(.hidden, for: .windowToolbar)
        // Blocking: the setter drops SwiftUI's own dismissals, so nothing but
        // Cancel takes this sheet down. The getter is recomputed every update,
        // so it also clears ITSELF the moment a chat model lands.
        .sheet(isPresented: Binding(
            get: {
                ChatWorkspace.gateShouldPresent(gateIsBlocking: gateIsBlocking,
                                                cancelled: gateCancelled,
                                                workspace: appState.chatWorkspace,
                                                welcomePresented: appState.showWelcome,
                                                palettePresented: appState.modelPalettePresented)
            },
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

    /// One generator page. `GenExperiment` is the shared catalogue (tray tiles,
    /// discovery chips, Tools menu), so this switch is the only place that maps
    /// a case to its view and cannot fall out of sync with what is offered.
    @ViewBuilder
    private func createPane(_ experiment: GenExperiment) -> some View {
        switch experiment {
        case .image:   ImageGenView().environmentObject(appState.imageGen)
        case .video:   VideoGenView().environmentObject(appState.videoGen)
        case .audio:   AudioGenView()
                           .environmentObject(appState.audioGen)
                           .environmentObject(appState.musicGen)
        case .model3d: Model3DGenView().environmentObject(appState.model3dGen)
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

/// A sidebar destination's chrome: nothing drawn until you hover it, and the
/// SAME gray when it is the selected one.
struct DestinationRowButton<Label: View>: View {
    let selected: Bool
    let action: () -> Void
    @ViewBuilder var label: Label

    @State private var hovering = false

    var body: some View {
        Button(action: action) { label }
            .buttonStyle(.plain)
            .frame(height: ChatMetrics.sidebarButtonHeight)
            .background(
                RoundedRectangle(cornerRadius: ChatMetrics.sidebarButtonCornerRadius)
                    .fill(SidebarRowStyle.fill(selected: selected, hovering: hovering))
            )
            .onHover { hovering = $0 }
    }
}

/// The one place the panel's row fill is decided — destinations and
/// conversations both read it, so "selected" cannot look like two things.
enum SidebarRowStyle {
    static func fill(selected: Bool, hovering: Bool) -> Color {
        if selected { return Color.primary.opacity(0.10) }
        return hovering ? Color.primary.opacity(0.05) : Color.clear
    }
}

/// The conversation list's modifier-aware selection maths. Pure, and at file
/// scope rather than inside `ChatSidebar`, so it can be driven from
/// `SidebarMultiSelectTests` without a rendered view — same reason
/// `ChatRowBuilder` and `ChatModeToggles` live out here.
///
/// The list stopped being a `List` (see `conversationsSidebar`), and with it
/// went the cmd/shift behaviour a `selection:` binding gives you for free.
/// This is that behaviour, written out.
enum SidebarMultiSelect {
    struct Outcome: Equatable {
        var selection: Set<UUID>
        /// What a subsequent shift-click ranges FROM.
        var anchor: UUID?
        /// The chat the detail column should show, or nil to leave it where it
        /// is — a cmd-click that deselects some OTHER row changes what is
        /// selected without changing where you are.
        var activate: UUID?
    }

    /// - Parameters:
    ///   - ordered: every visible row id in panel order, both sections
    ///     flattened — a shift-range crosses the Agents/Chats boundary.
    ///   - active: the chat currently on screen.
    static func click(_ id: UUID,
                      ordered: [UUID],
                      selection: Set<UUID>,
                      anchor: UUID?,
                      active: UUID?,
                      command: Bool,
                      shift: Bool) -> Outcome {
        // Shift wins when both are held, as it does in every macOS list.
        if shift, let anchor, anchor != id,
           let from = ordered.firstIndex(of: anchor),
           let to = ordered.firstIndex(of: id) {
            let span = from <= to ? from...to : to...from
            // The range REPLACES the selection and the anchor stays put, so
            // shift-clicking around re-ranges from one origin instead of
            // accumulating every range you passed through.
            return Outcome(selection: Set(ordered[span]), anchor: anchor, activate: id)
        }
        guard command else {
            return Outcome(selection: [id], anchor: id, activate: id)
        }
        guard selection.contains(id) else {
            return Outcome(selection: selection.union([id]), anchor: id, activate: id)
        }
        // Cmd-clicking the ONLY selected row is a no-op: this selection is also
        // the panel's "you are here", and emptying it would leave a transcript
        // on screen with nothing in the list pointing at it.
        guard selection.count > 1 else {
            return Outcome(selection: selection, anchor: id, activate: nil)
        }
        var next = selection
        next.remove(id)
        // Deselecting the row you were READING moves to the nearest survivor —
        // otherwise the transcript belongs to a row that is no longer lit.
        let activate = active == id ? nearest(to: id, in: ordered, within: next) : nil
        return Outcome(selection: next, anchor: id, activate: activate)
    }

    /// What the selection becomes when the composer takes the keyboard, or nil
    /// when it already says that.
    ///
    /// Typing is a statement that you are working in ONE conversation. Without
    /// this the multi-selection stayed lit behind the field, and since a
    /// multi-selection outranks focus for ⌘⌫ (`ChatDeleteShortcut.route`) the
    /// chord kept raising a delete dialog mid-message — two rules each reading
    /// a true fact and disagreeing about which one meant "the user is deleting
    /// chats". The chat you are typing IN is the survivor: its transcript is
    /// the one on screen above the field.
    ///
    /// Nil rather than an equal set, so a focus event that changes nothing does
    /// not publish.
    static func focusingComposer(in sessionId: UUID, selection: Set<UUID>) -> Set<UUID>? {
        selection == [sessionId] ? nil : [sessionId]
    }

    private static func nearest(to id: UUID, in ordered: [UUID], within set: Set<UUID>) -> UUID? {
        guard let origin = ordered.firstIndex(of: id) else { return set.first }
        return ordered.enumerated()
            .filter { set.contains($0.element) }
            .min { abs($0.offset - origin) < abs($1.offset - origin) }?
            .element
    }
}

/// Which deletions stop and ask first. Pure, and beside `SidebarMultiSelect`
/// for the same reason — the rule is worth pinning, and there is nothing to
/// see on screen until you have already lost the conversations.
enum SidebarDeleteConfirm {
    /// ⌘⌫ ALWAYS asks: the key names no row, so its target is implicit — every
    /// selected row, or, with nothing selected, whichever chat you happen to be
    /// reading. And any BULK delete asks whichever control started it, because
    /// N conversations go on one action and nothing undoes it. A single
    /// deliberate delete on a named row (the row's ✕, its context menu) still
    /// goes straight through, as it always has.
    static func required(count: Int, keyboard: Bool) -> Bool {
        count > 0 && (keyboard || count > 1)
    }

    /// What ⌘⌫ acts on: the sidebar's selection, or — with nothing selected —
    /// the chat being read. Nil when there is nothing to delete, which is also
    /// what disables the menu item, since a command that does nothing when you
    /// pick it is the dead-control class.
    static func target(selection: Set<UUID>, activeChatId: UUID?) -> Set<UUID>? {
        let ids = selection.isEmpty ? (activeChatId.map { Set([$0]) } ?? []) : selection
        return ids.isEmpty ? nil : ids
    }

    /// The count is the thing to check before agreeing, so it is in the title.
    static func title(count: Int) -> String {
        count == 1 ? "Delete this chat?" : "Delete \(count) chats?"
    }
}

/// Where ⌘⌫ goes.
///
/// A menu item's key equivalent is offered the keystroke by
/// `performKeyEquivalent` BEFORE the first responder ever sees it, so a Delete
/// Chat command in the menu bar takes ⌘⌫ away from every text field in the app
/// — including the composer, where it has meant "delete to the start of the
/// line" since long before this app existed. Typing it mid-message raised a
/// delete-chat dialog, and `.disabled(chatDeletionTarget == nil)` cannot help:
/// a chat is open in exactly the state where you are typing into one.
///
/// So the command ROUTES rather than claims. The keyboard's own owner wins
/// while it has focus, and the menu item keeps its slot — which is the half
/// that made the shortcut discoverable in the first place.
enum ChatDeleteShortcut {
    enum Route: Equatable {
        /// Hand it back to the text being edited (see `deleteToBeginningOfLine`
        /// — the menu has already swallowed the event, so returning early
        /// would make ⌘⌫ do nothing at all in the composer).
        case deleteToLineStart
        case deleteChats
    }

    /// - Parameter selectedChats: how many rows the sidebar has picked. Several
    ///   is unambiguous — you were working in the list — and it OUTRANKS focus,
    ///   because focus alone is not a signal this window can be trusted to get
    ///   right: nothing in it takes the keyboard off the composer except
    ///   `KeyboardFocus.resignTextEditor`, and one stuck reading would
    ///   otherwise make ⌘⌫ a line delete forever with a dozen chats selected
    ///   behind it. Belt and braces on a key that deletes conversations.
    static func route(editingText: Bool, selectedChats: Int) -> Route {
        if selectedChats > 1 { return .deleteChats }
        return editingText ? .deleteToLineStart : .deleteChats
    }
}

/// Who holds the keyboard.
///
/// The chat window has no real focus model: its conversation rows are
/// `.buttonStyle(.plain)` Buttons, which take no first responder under macOS's
/// default keyboard navigation — the same fact that made `.onDeleteCommand` on
/// that column never fire. So clicking a chat moved the selection while the
/// COMPOSER kept the keyboard, and every keystroke-owning decision downstream
/// read "the user is typing" forever (live 2026-08-12: after one click in the
/// composer, ⌘⌫ never deleted a chat again).
enum KeyboardFocus {

    /// Whether the responder holding the keyboard is text being typed into.
    ///
    /// `NSTextView` covers both cases and is the only type that needs naming: a
    /// focused `NSTextField` never becomes first responder itself — the
    /// window's FIELD EDITOR does, and that is an `NSTextView`.
    static func isTextEditor(_ responder: NSResponder?) -> Bool {
        responder is NSTextView
    }

    /// Move the keyboard out of a text field, because nothing else in this
    /// window will. Only when a text editor actually has it — an unconditional
    /// `makeFirstResponder(nil)` would yank focus off whatever else legitimately
    /// holds it.
    static func resignTextEditor(in window: NSWindow?) {
        guard let window, isTextEditor(window.firstResponder) else { return }
        window.makeFirstResponder(nil)
    }
}

// MARK: - Sidebar

/// Every measured row, so the quick-switch can number only the ones in the
/// clear. Published per row and merged on the way up.
struct SidebarRowSpansKey: PreferenceKey {
    static let defaultValue: [UUID: SidebarRowSpan] = [:]
    static func reduce(value: inout [UUID: SidebarRowSpan],
                       nextValue: () -> [UUID: SidebarRowSpan]) {
        value.merge(nextValue()) { _, new in new }
    }
}

/// The bottom edge of the frosted destination block, measured in the column's
/// own space — rows above this line are behind glass.
struct SidebarClearBandTopKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

/// The bottom edge of the panel — rows below this line are off the fold.
struct SidebarClearBandBottomKey: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

struct ChatSidebar: View {
    /// The one coordinate space the three band measurements share.
    ///
    /// Load-bearing that it is the COLUMN's own space and not `.global`: a
    /// global frame is not re-published when a view merely MOVES, and entering
    /// fullscreen translates the whole column without resizing the pinned
    /// destination block — so the block went on reporting the maxY it had in
    /// the smaller window while the rows slid out from under it, and rows in
    /// plain sight lost their badges. Measured here, the top edge is the
    /// block's own height and the bottom edge is the column's, both of which
    /// change only when there is genuinely a new layout to report.
    static let bandSpace = "chatSidebarBand"

    @EnvironmentObject var appState: AppState
    /// Observed directly — AppState forwards objectWillChange only for the
    /// server and the agent store, so a badge reading `appState.downloads`
    /// never repainted while a transfer started, progressed or finished.
    @EnvironmentObject var downloads: DownloadManager
    @Environment(\.openWindow) private var openWindow
    @State private var hoveredSessionId: UUID?
    /// Where a shift-click ranges FROM. Moved by every plain / cmd click, left
    /// alone by shift itself so dragging a range up and down keeps re-ranging
    /// from the same origin instead of walking away from it.
    @State private var selectionAnchor: UUID?
    /// Scans for installed agent CLIs — the Code Launcher row renders the tray's
    /// shared menu body, which needs it.
    @StateObject private var cliDetector = CLILauncher()
    /// Holding ⌘ numbers the conversation rows. A modifier held down is STATE,
    /// which SwiftUI does not report outside a hovered view — see the monitor.
    @StateObject private var modifiers = ModifierKeyMonitor()
    /// Where the frosted destination block ENDS, and where the panel ends, in
    /// the column's own space — the band a conversation row has to sit inside
    /// to be readable. Measured rather than derived from the inset's height:
    /// the block's contents vary (the Create list, the download badge), so the
    /// one number that cannot go stale is the one the block reports itself.
    @State private var clearBandTop: CGFloat = 0
    @State private var clearBandBottom: CGFloat = 0
    /// Each conversation row's vertical extent, reported only while ⌘ is down.
    @State private var rowSpans: [UUID: SidebarRowSpan] = [:]

    /// The rows a number may land on: those fully clear of the frosted block
    /// and inside the panel. `nil` means "not measured" — before the first
    /// layout after ⌘ goes down — and numbers everything, which is the old
    /// behaviour rather than a sidebar of blank rows for one frame.
    private var numberedRows: Set<UUID>? {
        guard modifiers.commandHeld else { return nil }
        return ChatQuickSwitch.numbering(rowSpans: rowSpans,
                                         clearBandTop: clearBandTop,
                                         clearBandBottom: clearBandBottom)
    }

    var body: some View {
        conversationsSidebar
    }

    /// ⌘1…⌘9 jumps to the row wearing that number; ⌃⌘1…⌃⌘9 ranges to it from
    /// where you are, selecting everything in between.
    ///
    /// Zero-size hidden buttons, the same idiom as ⌘R regenerate: they need
    /// THIS view's session list, and a key equivalent works wherever focus is,
    /// including while the composer's text view has it.
    ///
    /// **Ranging is ⌃⌘, not the ⇧⌘ every macOS list uses, because macOS took
    /// those digits first**: ⇧⌘3/4/5 are screen capture and ⇧⌘6 grabs the
    /// Touch Bar, all system-level, so they never reach an app — four of the
    /// nine ranged silently and the other five worked, which is worse than a
    /// shortcut that is simply somewhere else. Control is the replacement
    /// rather than Option because it is the one modifier that does not rewrite
    /// the character its key produces (⌥4 is ¢), so key-equivalent matching
    /// stays boring. ⌃ alone would collide with Mission Control's
    /// switch-to-desktop-N; with ⌘ it does not.
    private var quickSwitchShortcuts: some View {
        ForEach(1...ChatQuickSwitch.maxSlots, id: \.self) { slot in
            let key = KeyEquivalent(Character("\(slot)"))
            Group {
                Button { quickSwitch(to: slot, extend: false) } label: { EmptyView() }
                    .keyboardShortcut(key, modifiers: .command)
                Button { quickSwitch(to: slot, extend: true) } label: { EmptyView() }
                    .keyboardShortcut(key, modifiers: [.command, .control])
            }
            .frame(width: 0, height: 0)
            .opacity(0)
        }
    }

    /// Go to a numbered chat, or range to it. Not `selectRow`: that one reads
    /// the CURRENT event's modifier flags, and the event behind these always
    /// has ⌘ down — every jump would toggle the row into a multi-selection
    /// instead of going to it. The flags are stated explicitly instead, and the
    /// decision itself is `ChatQuickSwitch.outcome`.
    private func quickSwitch(to slot: Int, extend: Bool) {
        guard let outcome = ChatQuickSwitch.outcome(
            slot: slot,
            sessions: appState.visibleChatSessions,
            numbering: numberedRows,
            selection: appState.sidebarSelection,
            anchor: selectionAnchor,
            active: appState.activeChatId,
            extend: extend)
        else { return }
        apply(outcome)
    }

    /// Write a selection outcome back. Selection BEFORE `activeChatId`, for the
    /// same reason `selectRow` does it in that order.
    private func apply(_ outcome: SidebarMultiSelect.Outcome) {
        appState.showConversation()
        selectionAnchor = outcome.anchor
        if appState.sidebarSelection != outcome.selection {
            appState.sidebarSelection = outcome.selection
        }
        if let target = outcome.activate, appState.activeChatId != target {
            appState.activeChatId = target
        }
    }

    private var conversationsSidebar: some View {
        // No `selection:` binding: a List draws its own selection tint UNDER
        // `listRowBackground`, which is the double highlight — two grays, the
        // inner one a different value from the destinations above, and an accent
        // agent label sitting on whichever won. Selection is ours now, drawn by
        // the one `SidebarRowStyle` both halves of this panel read — and the
        // cmd/shift behaviour the binding used to supply is `SidebarMultiSelect`.
        // A ScrollView, not a List. These rows draw everything themselves —
        // background, hover, selection, separators — so the only thing
        // `.listStyle(.sidebar)` still contributed was its own horizontal
        // margin around the content, which held every row ~18pt in from the
        // panel edge while the destinations above sat at the 8pt gutter. That
        // margin is NOT what `listRowInsets` controls (zeroing those changed
        // nothing), and there is no API to remove it. A plain stack takes the
        // same `.padding(.horizontal, sidebarGutter)` the destination column
        // takes, so the two halves line up because they are laid out the same
        // way — not because two numbers were talked into agreeing.
        ScrollView {
            // Two sections, one row builder. Agent threads sit above the plain
            // chats — the section is HIDDEN when there are none, because an
            // empty heading is a promise of content that isn't there.
            let groups = SidebarSessionGroups.split(appState.visibleChatSessions)
            // The panel's visual order, both sections flattened — a shift-click
            // ranges across the Agents/Chats boundary, because the split is a
            // heading, not a wall.
            let ordered = groups.agents.map(\.id) + groups.chats.map(\.id)
            LazyVStack(alignment: .leading, spacing: 2) {
                if !groups.agents.isEmpty {
                    sectionHeader("Agents")
                    ForEach(groups.agents) { session in
                        sessionRow(session, ordered: ordered)
                    }
                }
                sectionHeader("Chats")
                ForEach(groups.chats) { session in
                    sessionRow(session, ordered: ordered)
                }
            }
            .padding(.horizontal, ChatMetrics.sidebarGutter)
            .padding(.bottom, 8)
        }
        .onAppear {
            // The rows READ the selection to decide their highlight, so it has
            // to be primed: `activeChatId` is usually set long before this panel
            // first appears, and an .onChange can't fire for a value that was
            // already there.
            if appState.sidebarSelection.isEmpty, let id = appState.activeChatId {
                appState.sidebarSelection = [id]
                selectionAnchor = id
            }
        }
        // Deletions from anywhere else (the tray, a task run, another window)
        // must not leave ids in the selection that no longer name a chat — a
        // stale one would be counted by the context menu's "Delete N Chats" and
        // handed straight back to `deleteSessions` by ⌫.
        .onChange(of: appState.visibleChatSessions.count) { _, _ in
            let live = Set(appState.visibleChatSessions.map(\.id))
            if !appState.sidebarSelection.isSubset(of: live) {
                appState.sidebarSelection.formIntersection(live)
            }
            if let anchor = selectionAnchor, !live.contains(anchor) { selectionAnchor = nil }
        }
        .onChange(of: appState.sidebarSelection) { _, newSelection in
            // When the sidebar's selection becomes a single id, make that the
            // active chat so the detail column follows the user's intent.
            if newSelection.count == 1, let id = newSelection.first {
                if appState.activeChatId != id { appState.activeChatId = id }
            }
        }
        .onChange(of: appState.activeChatId) { _, newActive in
            // Keep the sidebar selection in sync when other parts of the app
            // change the active chat (open-from-tray, quick launcher, etc.).
            // Only COLLAPSE it when the new active chat isn't already IN it:
            // this ran unconditionally, so a cmd-click that added a second row
            // was overwritten by `[id]` on its way out and the panel could never
            // hold more than one — the multi-select bug. A chat the selection
            // already contains is a move WITHIN the selection, and leaves it be.
            guard let id = newActive else {
                appState.sidebarSelection.removeAll()
                selectionAnchor = nil
                return
            }
            if !appState.sidebarSelection.contains(id) {
                appState.sidebarSelection = [id]
                selectionAnchor = id
            }
        }
        // NO `.onDeleteCommand`: it only fires while its view is in the
        // responder chain, which is why it is List API — a `List(selection:)`
        // becomes first responder when you click a row. This column stopped
        // being a List (see above) and its rows are `.buttonStyle(.plain)`
        // Buttons, which take no focus under macOS's default keyboard
        // navigation, so the modifier sat here reading like a working ⌫ and
        // never once ran. The keyboard route is the File menu's Delete Chat
        // (⌘⌫, Finder's own shortcut), which is a menu command and therefore
        // needs no focus at all — and being IN a menu, it is also visible.
        //
        // The dialog it raises is the one every delete control shares, so its
        // state lives on AppState where the menu command can reach it.
        .confirmationDialog(
            SidebarDeleteConfirm.title(count: appState.pendingChatDeletion?.count ?? 1),
            isPresented: Binding(get: { appState.pendingChatDeletion != nil },
                                 set: { if !$0 { appState.pendingChatDeletion = nil } }),
            presenting: appState.pendingChatDeletion
        ) { ids in
            Button("Delete", role: .destructive) {
                deleteChats(ids)
                appState.pendingChatDeletion = nil
            }
            // Return deletes. The dialog is the second time you have said so
            // (a menu command or a row's Delete raised it), and reaching for
            // the trackpad to confirm a decision already made is the whole
            // reason this asked to be a keyboard app. Escape still cancels —
            // AppKit gives the `.cancel` role that for free.
            .keyboardShortcut(.defaultAction)
            Button("Cancel", role: .cancel) { appState.pendingChatDeletion = nil }
        } message: { _ in
            Text("This can't be undone.")
        }
        // ⌘1…⌘9. In the sidebar rather than the window's `.commands` because
        // they address THIS view's conversation list; hidden in a background so
        // they cost no layout.
        .background(quickSwitchShortcuts)
        // The platform's own scroll-edge effect at BOTH ends: rows pass under
        // the window's top edge and under the New Chat row (a `safeAreaInset`,
        // so content scrolls beneath it), and a soft edge is how macOS frosts
        // that overlap. Not a hand-drawn band — a custom strip pulled into this
        // area once looked native and swallowed every click in it.
        .scrollEdgeEffectStyle(.soft, for: [.top, .bottom])
        // No blanket `.onChange(of: activeChatId) { showConversation() }`:
        // every deliberate route into a conversation (the row button above,
        // New Chat, the quick launcher) calls showConversation() itself, and
        // the id ALSO moves on deleteSession's fallback — which yanked the
        // user out of the Models/Create/Settings pane they were browsing when
        // they deleted the active chat from the sidebar.
        // No Agents entry here: "Manage Agents…" lives in the composer's agent
        // chip, next to the control it configures, and a second route to the
        // same window only competed with the conversation list. Models is a
        // different animal — it is not a window any more but a MODE of this
        // one, so this row is the mode switch, not a duplicate route.
        // Destinations above the conversation list, in one column: what the
        // window can BE, then what you've said. Selecting any of them changes
        // only the content area — the sidebar never rearranges itself, so the
        // list of places stays where the eye learned it.
        .safeAreaInset(edge: .top) {
            VStack(spacing: 2) {
                // New Chat, and beside it the choice of WHO the chat is with.
                destinationRow("New Chat", icon: "square.and.pencil",
                               selected: false) {
                    appState.showConversation()
                    _ = appState.newChatSession()
                }
                destinationRow("Models", icon: "square.stack.3d.up",
                               selected: appState.chatWorkspace.isModels,
                               badge: activeDownloadCount) {
                    appState.chatWorkspace.isModels ? appState.showConversation() : appState.showModels()
                }
                destinationRow("Settings", icon: "gearshape",
                               selected: appState.chatWorkspace.isSettings) {
                    appState.chatWorkspace.isSettings ? appState.showConversation() : appState.showSettings()
                }

                // The Create pages, from the SAME catalogue the discovery chips
                // and the Tools menu iterate (`sidebarCreateItems` — a filter on
                // `mediaItems`, so the three surfaces cannot drift). Each row is
                // the mode switch for its generator page, exactly like Models.
                sectionHeader("Create")
                ForEach(ChatEmptyState.sidebarCreateItems) { item in
                    if case .create(let experiment) = item.action {
                        destinationRow(item.title, icon: item.systemImage,
                                       selected: appState.chatWorkspace.experiment == experiment) {
                            if appState.chatWorkspace.experiment == experiment {
                                appState.showConversation()
                            } else {
                                appState.showCreate(experiment)
                            }
                        }
                    }
                }

                // The agent/automation cluster, below Create. A GAP on its
                // first row instead of a heading: nothing here is a Create
                // item, and proximity is what would say otherwise.
                agentsRow
                    .padding(.top, 10)
                destinationRow("Tasks", icon: "clock.badge.checkmark",
                               selected: appState.chatWorkspace.isTasks) {
                    appState.chatWorkspace.isTasks ? appState.showConversation() : appState.showTasks()
                }
                // A launcher is a CHOICE of CLI, so the row is the menu it has
                // always been (the tray's own list, shared) rather than an
                // invented pane with one list in it.
                codeLauncherRow

                // No "Chats" heading here: the list carries its own section
                // headers ("Agents", "Chats"), and one of them appears only
                // when it has rows. A heading pinned in this inset could not
                // do that — it would sit above an empty list announcing a
                // section that isn't there.
            }
            // One gutter for the whole panel — the conversation rows below
            // apply the same constant, so the two halves are the same width by
            // construction rather than by two numbers that happen to agree.
            .padding(.horizontal, ChatMetrics.sidebarGutter)
            .padding(.top, 10)
            .padding(.bottom, 8)
            // Where the frost ENDS. This block is what conversation rows scroll
            // under, so its own bottom edge is the line between "readable" and
            // "behind glass" — no constant to keep in step with its contents.
            .background {
                GeometryReader { proxy in
                    Color.clear.preference(
                        key: SidebarClearBandTopKey.self,
                        value: proxy.frame(in: .named(ChatSidebar.bandSpace)).maxY)
                }
            }
            // No backdrop: the toolbar's BAR is back, so `scrollEdgeEffectStyle`
            // has something to attach to again and the platform frosts what
            // scrolls beneath this block.
        }
        // The panel's own bottom edge closes the band — in this space that is
        // the column's HEIGHT, which is the whole "how much window is there?"
        // question the badge count answers. Applied OUTSIDE the safeAreaInset
        // so it measures the whole column, and so the three readers below are
        // ancestors of the block that publishes the top edge — a preference
        // only travels up.
        //
        // There was a second publisher for the TOP edge here, `frame.minY +
        // safeAreaInsets.top`, described as the same line derived a second way.
        // It never was: measured, it reported 104 against a frost line at 370,
        // because this reader sits below the toolbar (so it has already lost
        // that inset) and the block's height is not part of what it reads back.
        // It was harmless only because the key reduces by `max` and the real
        // measurement was always larger — a fallback that could only ever be
        // wrong is worse than none.
        .background {
            GeometryReader { proxy in
                Color.clear.preference(
                    key: SidebarClearBandBottomKey.self,
                    value: proxy.frame(in: .named(ChatSidebar.bandSpace)).maxY)
            }
        }
        // Declared here, OUTSIDE both readers and the inset, so the block, the
        // column and every row resolve against the same origin.
        .coordinateSpace(.named(ChatSidebar.bandSpace))
        .onPreferenceChange(SidebarClearBandTopKey.self) { value in
            applyMeasurement(value, to: $clearBandTop)
        }
        .onPreferenceChange(SidebarClearBandBottomKey.self) { value in
            applyMeasurement(value, to: $clearBandBottom)
        }
        .onPreferenceChange(SidebarRowSpansKey.self) { value in
            guard rowSpans != value else { return }
            // Async for the same reason the transcript's scroll correction is
            // (issue #136): a geometry-driven write that lands inside the
            // layout flush re-enters layout, and under a scroll it does so
            // every frame. One turn later it coalesces instead.
            DispatchQueue.main.async { rowSpans = value }
        }
    }

    /// Store a measured edge, ignoring sub-pixel noise. The equality gate is
    /// load-bearing, not an optimisation: a write per layout would re-enter
    /// layout and never settle.
    private func applyMeasurement(_ value: CGFloat, to binding: Binding<CGFloat>) {
        guard abs(binding.wrappedValue - value) > 0.5 else { return }
        DispatchQueue.main.async { binding.wrappedValue = value }
    }

    // MARK: Selection

    /// Whether a conversation row can be lit at ALL in the current workspace
    /// mode. Asked of the active chat against itself, so the answer is the mode
    /// question alone — the per-row half is now selection membership.
    private var conversationsAreLit: Bool {
        guard let active = appState.activeChatId else { return false }
        return SidebarSelection.isConversationSelected(
            sessionId: active, activeChatId: active, workspace: appState.chatWorkspace)
    }

    /// One click on a conversation row. The modifier maths is pure and lives in
    /// `SidebarMultiSelect`; this is only the wiring — read the flags off the
    /// event AppKit is currently dispatching (a SwiftUI Button action has no
    /// other way to see them), then apply the outcome.
    ///
    /// Selection is written BEFORE `activeChatId` on purpose: the sync above
    /// collapses the selection for an active chat it doesn't already contain,
    /// so the other order would undo a cmd-click on its way out.
    private func selectRow(_ id: UUID, ordered: [UUID]) {
        // Clicking a row means you are working in the LIST, so the keyboard
        // comes with you. Nothing else moves it: these rows take no first
        // responder, so the composer keeps the keyboard while you click around
        // the sidebar — which is what made ⌘⌫ a line delete forever after one
        // click in the field. Deliberately NOT done by `quickSwitch`: ⌘\<digit\>
        // is for jumping to a chat and typing in it, so the composer stays
        // armed there (a multi-selection is what tells ⌘⌫ otherwise).
        KeyboardFocus.resignTextEditor(in: NSApp.keyWindow)
        let flags = (NSApp.currentEvent?.modifierFlags ?? NSEvent.modifierFlags)
            .intersection(.deviceIndependentFlagsMask)
        let outcome = SidebarMultiSelect.click(
            id, ordered: ordered,
            selection: appState.sidebarSelection,
            anchor: selectionAnchor,
            active: appState.activeChatId,
            command: flags.contains(.command),
            shift: flags.contains(.shift))
        apply(outcome)
    }

    /// Every control that deletes conversations comes through here, so which
    /// ones ask first is `SidebarDeleteConfirm`'s decision and not a per-button
    /// habit. `keyboard` is what separates ⌘⌫ — which names no row — from a
    /// click on one.
    private func requestDeleteChats(_ ids: Set<UUID>, keyboard: Bool) {
        guard !ids.isEmpty else { return }
        if SidebarDeleteConfirm.required(count: ids.count, keyboard: keyboard) {
            appState.pendingChatDeletion = ids
        } else {
            deleteChats(ids)
        }
    }

    /// Delete a set of conversations and leave the panel's own state consistent
    /// — the ids go out of the selection and the anchor FIRST, so nothing that
    /// reads them (the ⌫ handler, the context menu's count) can name a chat that
    /// is already gone.
    private func deleteChats(_ ids: Set<UUID>) {
        guard !ids.isEmpty else { return }
        appState.sidebarSelection.subtract(ids)
        if let anchor = selectionAnchor, ids.contains(anchor) { selectionAnchor = nil }
        appState.deleteSessions(ids)
    }

    /// A section heading, sitting on the same left edge as the rows under it.
    private func sectionHeader(_ title: String) -> some View {
        Text(title)
            .font(.caption.weight(.semibold))
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            // The stack owns the gutter; the heading owes only the row's inner
            // inset, so it sits on the same left line as the labels under it.
            .padding(.horizontal, ChatMetrics.sidebarRowInset)
            .padding(.top, 10)
            .padding(.bottom, 2)
    }

    /// One conversation row, shared by both sections. `ordered` is the panel's
    /// flattened visual order, which shift-click ranges over.
    @ViewBuilder
    private func sessionRow(_ session: ChatSession, ordered: [UUID]) -> some View {
        // Lit rows are the SELECTION, not just the active chat — otherwise a
        // cmd-clicked second row is selected (⌫ deletes it) while looking
        // exactly like an unselected one. A conversation is still only lit while
        // the window is showing conversations: otherwise opening Tasks left the
        // last chat lit alongside the Tasks destination, two "you are here"
        // marks for one window.
        let isSelected = conversationsAreLit && appState.sidebarSelection.contains(session.id)
        // The button IS the row: it carries the padding, the height floor and
        // the contentShape, so every pixel of the fill is clickable. As a
        // sibling sized by an outer frame, the label was CENTRED in the row's
        // height and only its own text band answered a click — the dead strip
        // along the top and bottom of the highlight.
        Button {
            selectRow(session.id, ordered: ordered)
        } label: {
            // An agent thread is named for its AGENT, with the agent's own
            // symbol beside it — the Agents section is a list of who you talk
            // to, so that answer belongs on the first line rather than in a
            // caption under a title derived from whatever you typed first.
            let agent = appState.agents.agent(id: session.agentId)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 4) {
                    if session.isExternalBridge {
                        Image(systemName: "paperplane.fill")
                            .font(.system(size: 9))
                            .foregroundStyle(isSelected ? Color.white.opacity(0.85) : Color.accentColor)
                            .help("Telegram conversation (view only)")
                    }
                    if let agent {
                        Image(systemName: agent.symbol)
                            .font(.system(size: 10))
                            .foregroundStyle(Color.accentColor)
                    }
                    Text(ChatSessionTitle.display(title: session.title,
                                                  agentName: agent?.name))
                        .font(.subheadline.weight(isSelected ? .semibold : .regular))
                        .lineLimit(1)
                        .foregroundStyle(.primary)
                }
                // What this particular conversation is about, displaced from
                // the title line by the agent's name. It is also the only
                // thing telling a second thread with the same agent apart
                // from the first — without it the sidebar draws two identical
                // rows. Absent until the thread has said something, so a new
                // one is a single line exactly like a destination row.
                if let subject = ChatSessionTitle.subject(title: session.title,
                                                          agentName: agent?.name) {
                    Text(subject)
                        .font(.caption2)
                        .lineLimit(1)
                        .truncationMode(.tail)
                        .foregroundStyle(.secondary)
                }
            }
            // Room kept for the delete button at ALL times, not only while
            // hovering: reserving it on hover reflows the title under the
            // pointer, right as you are aiming at it.
            .padding(.leading, ChatMetrics.sidebarRowInset)
            .padding(.trailing, ChatMetrics.sidebarRowInset + 18)
            .padding(.vertical, 5)
            // A row is as tall as what is IN it: one line matches a
            // destination row exactly, and only the rows carrying an agent
            // subtitle grow. The floor lives on the LABEL so the button — the
            // thing that answers clicks — is the full height of the fill.
            // `minHeight` is a floor, never a fixed height.
            .frame(maxWidth: .infinity, minHeight: ChatMetrics.sidebarButtonHeight,
                   alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        // Where this row sits, so the numbering can skip whatever is under the
        // frosted block. Attached ONLY while ⌘ is down: a probe on every row is
        // a preference write per row per scroll frame, and outside this
        // transient mode nothing reads the answer.
        .background {
            if modifiers.commandHeld {
                GeometryReader { proxy in
                    let frame = proxy.frame(in: .named(ChatSidebar.bandSpace))
                    Color.clear.preference(
                        key: SidebarRowSpansKey.self,
                        value: [session.id: SidebarRowSpan(top: frame.minY, bottom: frame.maxY)])
                }
            }
        }
        // One meaning for gray in this panel, and one SHAPE: the fill rides the
        // row's own content inside the stack's gutter, exactly as a
        // destination's `.background` does. (It was a `listRowBackground` once,
        // which fills the whole row rect and ignores the insets beside it — a
        // selected chat ran edge to edge under a column of inset destinations.)
        .background(
            RoundedRectangle(cornerRadius: ChatMetrics.sidebarButtonCornerRadius)
                .fill(SidebarRowStyle.fill(selected: isSelected,
                                           hovering: hoveredSessionId == session.id))
        )
        // A real Button laid OVER the row, never a tap gesture around one: an
        // overlay is hit-tested first, so its clicks reach it rather than the
        // row underneath, and the row keeps its whole area clickable.
        //
        // The badge and the delete button share this one reserved slot, and the
        // badge WINS while ⌘ is down: they would otherwise draw on top of each
        // other on the hovered row, which is exactly the row the pointer is on
        // whenever anyone reads a number. ⌘-click is also how a second row
        // joins the selection, so the pointer is regularly here with ⌘ held —
        // and the ✕ is one pixel away from a click meaning "delete this".
        .overlay(alignment: .trailing) {
            if modifiers.commandHeld, let slot = ChatQuickSwitch.slot(for: session.id,
                                                                      in: appState.visibleChatSessions,
                                                                      numbering: numberedRows) {
                Text("\(slot)")
                    .font(.caption2.weight(.semibold).monospacedDigit())
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color.primary.opacity(0.08))
                    )
                    .padding(.trailing, ChatMetrics.sidebarRowInset)
                    // Decoration: it must never eat the click that selects the
                    // row it is drawn on.
                    .allowsHitTesting(false)
                    .transition(AnyTransition.asymmetric(
                        insertion: .opacity.animation(.easeIn(duration: 0.25).delay(0.2)),
                        removal: .opacity.animation(.easeOut(duration: 0.15))
                     ))
                    .animation(.easeInOut(duration: 0.25), value: modifiers.commandHeld)
            } else if hoveredSessionId == session.id {
                Button {
                    requestDeleteChats([session.id], keyboard: false)
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 14))
                        .symbolRenderingMode(.hierarchical)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .padding(.trailing, ChatMetrics.sidebarRowInset)
                .help("Delete chat")
            }
        }
        .onHover { isHovered in
            hoveredSessionId = isHovered ? session.id : nil
        }
        .contextMenu {
            // Right-clicking INSIDE a multi-selection acts on all of it, and
            // says how many; right-clicking outside one is a single delete.
            if appState.sidebarSelection.count > 1,
               appState.sidebarSelection.contains(session.id) {
                Button("Delete \(appState.sidebarSelection.count) Chats", role: .destructive) {
                    requestDeleteChats(appState.sidebarSelection, keyboard: false)
                }
            } else {
                Button("Delete", role: .destructive) {
                    requestDeleteChats([session.id], keyboard: false)
                }
            }
        }
    }

    /// One destination row. All of them are the same shape by construction —
    /// the mockup's point is that this column reads as ONE list of places, not
    /// as a pile of controls that happen to be stacked.
    private func destinationRow(_ title: String, icon: String, selected: Bool,
                                badge: Int = 0,
                                action: @escaping () -> Void) -> some View {
        DestinationRowButton(selected: selected, action: action) {
            destinationLabel(title, icon: icon, selected: selected, badge: badge)
        }
    }

    @ViewBuilder
    private func destinationLabel(_ title: String, icon: String, selected: Bool,
                                  badge: Int = 0) -> some View {
        HStack(spacing: 7) {
            Image(systemName: icon)
                .font(.system(size: 12, weight: .medium))
                .frame(width: 16)
            Text(title).font(.subheadline.weight(.medium))
            Spacer(minLength: 4)
            if badge > 0 {
                Text("\(badge)")
                    .font(.caption2.monospacedDigit())
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(.quaternary, in: Capsule())
            }
        }
        .foregroundStyle(Color.primary)
        // The SAME inner inset a conversation row uses, so a destination's icon
        // and a chat's title start on one line down the column.
        .padding(.horizontal, ChatMetrics.sidebarRowInset)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .contentShape(Rectangle())
    }

    /// The Agents row. A MENU, because "Agents" is two things: the agents you
    /// can start a conversation as, and the editor for them.
    private var agentsRow: some View {
        destinationRow("Agents", icon: "person.2",
                       selected: appState.chatWorkspace.isAgents) {
            appState.chatWorkspace.isAgents ? appState.showConversation() : appState.showAgents()
        }
    }

    /// The Code Launcher row: the tray's own CLI list, so the two can't drift.
    /// DMG-only — the App Store build can't detect or launch other apps' CLIs,
    /// and a row that can only fail is the dead-control class.
    @ViewBuilder
    private var codeLauncherRow: some View {
        if BuildFeatures.current.cliLauncher {
            Menu {
                CLILauncherMenuItems(
                    detector: cliDetector,
                    baseURL: appState.server.baseURL,
                    servedModelId: appState.server.chatModelId ?? "mlx-serve",
                    serverContextLength: appState.server.chatModelInfo?.contextLength,
                    models: appState.server.allModels,
                    openSandboxAgent: { agentId in
                        appState.pendingSandboxAgentLaunch = .init(agentId: agentId)
                        AppActivation.openWindow(id: "sandboxTerminal", using: openWindow)
                    })
            } label: {
                // "Code", matching the tray's own Code button over the same menu.
                destinationLabel("Code", icon: "terminal", selected: false)
            }
            .menuStyle(.button)
            .buttonStyle(.plain)
            .menuIndicator(.hidden)
            .frame(height: ChatMetrics.sidebarButtonHeight)
        }
    }

    private var activeDownloadCount: Int {
        downloads.downloads.values.filter { $0.status == .downloading }.count
    }

}

// MARK: - Chat Detail

struct ChatDetailView: View {
    let sessionId: UUID
    @EnvironmentObject var appState: AppState
    /// Observed directly (AppState does not forward download publishes) — the
    /// create banner's "not downloaded" pill and the held-prompt readiness
    /// checks must repaint when the bytes land.
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var toolExecutor: ToolExecutor
    @EnvironmentObject var mcpManager: MCPManager
    @EnvironmentObject var chatEngine: ChatTurnEngine
    @Environment(\.openWindow) private var openWindow
    // Settings ▸ Interface, observed HERE because `ChatMetrics` reads these
    // keys straight off UserDefaults with no SwiftUI dependency: without a
    // real observer a size change reaches only rows that happen to re-render,
    // leaving the transcript in two fonts at once. The values feed the `.id`
    // on the row stack, which is what forces the rebuild.
    @AppStorage(InterfacePrefKey.textSize) private var interfaceTextSize = ChatTextSize.medium.rawValue
    @AppStorage(InterfacePrefKey.compactMode) private var interfaceCompact = false
    @State private var inputText = ""
    /// Where ↑/↓ have walked back to in this chat's own history. Per-tab state
    /// like everything else here — `ChatDetailView` is REUSED across tabs, so a
    /// walk left running would resume in someone else's conversation. Stale
    /// indexes are harmless by construction (`ComposerHistory` treats one that
    /// no longer names an entry as no walk at all), but resetting on the switch
    /// is what makes the first ↑ in a new tab mean what it says.
    @State private var composerWalk = ComposerHistory.Walk.idle
    // The three toolbar toggles mirror the visible session's persisted state
    // (`ChatSession.enableThinking` / `.mode` / `.useMCP`). They're loaded from
    // the session on appear AND on every `sessionId` change, and written back on
    // toggle — so each chat tab remembers its own Think/Agent/MCP choice instead
    // of leaking the active tab's value into the reused ChatDetailView.
    @State private var enableThinking = false
    @State private var reasoningEffort = ReasoningEffort.low
    @State private var isAgentMode = false
    @State private var mcpMode = false
    @State private var showMCPMarketplace = false
    @State private var executingPlanMessageId: UUID?
    // Follow-the-newest-line. The decision core is pure (`ChatScrollState`,
    // pinned by ChatScrollTests); the model holds it in a class so per-frame
    // scroll geometry doesn't re-evaluate the whole chat body — only the
    // pinned flag is published. `scrollPosition` is the one handle that moves
    // the transcript, so no view needs a `ScrollViewProxy` passed around.
    @StateObject private var scrollModel = ChatScrollModel()
    @State private var scrollPosition = ScrollPosition(idType: Never.self, edge: .bottom)
    @State private var pasteMonitor: Any?
    @State private var pendingImages: [PendingImage] = []
    @State private var pendingPDFs: [(name: String, text: String)] = []
    @State private var pendingVideos: [ChatVideo] = []
    @State private var pendingAudio: [ChatAudio] = []
    @StateObject private var recorder = AudioRecorder()
    // Tool-approval gate state. `pendingApproval` is set right before each
    // tool call when Agent mode is on; the sheet at the bottom of `body`
    // observes it and resumes `approvalContinuation` with the user's choice.
    @State private var pendingApproval: ToolApprovalRequest?
    @State private var toolAllowList = SessionToolAllowList()
    // Plain Bool (not @FocusState): the composer is an NSTextView wrapper, so
    // AppKit first-responder is the source of truth and GrowingTextEditor mirrors
    // it back into this flag. The Cmd+V attach monitor reads it; on-appear and
    // post-generation code set it true to (re)focus the field.
    @State private var inputFocused = false
    /// The detail column's measured width — the panel next to the session
    /// sidebar, not the whole window. Drives `contentWidth` below. Zero until
    /// `body`'s root view reports its first `onGeometryChange`.
    @State private var columnWidth: CGFloat = 0

    /// The shared reading measure all three capped sites (transcript,
    /// composer, empty-state greeting) apply. See `ChatMetrics.contentWidthFraction`.
    private var contentWidth: CGFloat {
        columnWidth > 0 ? columnWidth * ChatMetrics.contentWidthFraction : ChatMetrics.contentFallbackWidth
    }
    @State private var composerHeight: CGFloat = 36
    // The composer's "create mode" (the chip rewired the composer into a
    // generator) is GONE: a media chip navigates to the Create pane, exactly
    // like the Tools menu. In-chat media generation is the agent tools' job
    // (`generate_image` & co.) — one way to drive a generator from a chat,
    // not two.
    // Pre-send intent nudge: when a message looks agentic / MCP-bound but the
    // matching mode is off, confirm before sending. `intentSuppress` remembers a
    // per-session "Send anyway" so we stop nagging that chat (keyed by session
    // id — the view is reused across tabs).
    @State private var pendingIntentPrompt: IntentPrompt?
    @State private var intentSuppress = SessionIntentSuppression()
    // Issue #227: built once per messages change, not per body pass. Rebuilding
    // inside the ForEach handed SwiftUI a fresh array on every layout pass and
    // the LazyVStack could spin forever.
    @State private var rows: [ChatRow] = []


    private var session: ChatSession? {
        appState.chatSessions.first { $0.id == sessionId }
    }

    /// Generation state for THIS chat. The engine runs one turn at a time, so a
    /// chat that doesn't own the active turn must show Send (idle), not the Stop
    /// button — and its Send is disabled while another chat is mid-turn.
    // MARK: - "/" skill menu

    /// Highlighted row; Escape hides the menu without clearing what was typed.
    @State private var slashSelection: Int = 0
    @State private var slashDismissed: Bool = false

    /// Skills answering the half-typed command, or none when the menu is
    /// closed. Guarded by `SlashCommands.query` FIRST: the skills folder is
    /// stat-ed on read, and this property is evaluated on every body pass
    /// (including ~20 Hz while a reply streams).
    private var slashMatches: [SkillSummary] {
        guard !slashDismissed, let q = SlashCommands.query(in: inputText) else { return [] }
        return SlashCommands.matches(query: q, in: AgentPrompt.skillManager.summaries)
    }

    /// Up/Down/Return/Tab/Escape while the menu is open. Returns false when
    /// it is closed, so the composer keeps every key it owns today.
    private func handleSlashKey(_ command: ComposerKeyCommand) -> Bool {
        let matches = slashMatches
        guard !matches.isEmpty else { return false }
        switch command {
        case .up:
            slashSelection = (slashSelection - 1 + matches.count) % matches.count
        case .down:
            slashSelection = (slashSelection + 1) % matches.count
        case .accept:
            pickSlashSkill(matches[min(slashSelection, matches.count - 1)])
        case .cancel:
            slashDismissed = true
        }
        return true
    }

    private func pickSlashSkill(_ skill: SkillSummary) {
        inputText = SlashCommands.accept(skill.name, in: inputText)
        slashSelection = 0
        inputFocused = true
    }

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
        reasoningEffort = session?.reasoningEffort ?? .low
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
    // mode discs. Starting a chat as an agent lives in the sidebar's Agents
    // destination now:

    /// The agent this tab is talking to (nil = none).
    private var activeAgent: Agent? { appState.agents.agent(id: session?.agentId) }

    /// Images/videos/PDFs/audio for this message, or a folder to ask questions
    /// about. Its own property (rather than inline in `composerControls`) so
    /// it carries a hover card like every other glyph in the row.
    private var attachmentMenu: some View {
        Menu {
            Button {
                pickAttachment()
            } label: {
                Label(attachmentMenuLabel, systemImage: "photo.on.rectangle")
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
        .composerTip(.attachments(audioSupported: audioSupported, videoSupported: videoSupported))
    }

    private var attachmentMenuLabel: String {
        switch (videoSupported, audioSupported) {
        case (true, true): return "Attach Image, Video, PDF, or Audio…"
        case (true, false): return "Attach Image, Video, or PDF…"
        case (false, true): return "Attach Image, PDF, or Audio…"
        case (false, false): return "Attach Image or PDF…"
        }
    }

    /// Shared look for the composer's icon-only mode controls.
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

    /// One brain: CLICK flips thinking, secondary-click picks the reasoning
    /// effort — same idiom as the wrench.
    private var thinkToggle: some View {
        Group {
            if let owner = toolbarToggles.thinkingLockedBy {
                Menu {
                    lockedModeMenu(owner)
                } label: {
                    modeIcon("brain", isOn: toolbarToggles.thinking, onColor: .blue, lockedBy: owner)
                }
            } else if isExternalBridgeSession {
                // Telegram session: write the shared config so the toggle stays
                // in sync with Settings and the bridge reads it live. No effort
                // menu — the bridge sends the plain boolean.
                Button {
                    appState.serverOptions.telegram.enableThinking.toggle()
                } label: {
                    modeIcon("brain", isOn: toolbarToggles.thinking, onColor: .blue)
                }
            } else {
                Menu {
                    reasoningEffortMenu
                } label: {
                    modeIcon("brain", isOn: toolbarToggles.thinking, onColor: .blue)
                } primaryAction: {
                    enableThinking.toggle()
                }
                .contextMenu { reasoningEffortMenu }
            }
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .composerTip(.thinking(isOn: toolbarToggles.thinking,
                               lockedBy: toolbarToggles.thinkingLockedBy))
    }

    /// The brain disc's secondary-click menu: how hard the model thinks while
    /// the toggle is on (`reasoning_effort`).
    @ViewBuilder private var reasoningEffortMenu: some View {
        Picker("Reasoning", selection: $reasoningEffort) {
            ForEach(ReasoningEffort.allCases) { effort in
                Text(effort.label).tag(effort)
            }
        }
        .pickerStyle(.inline)
    }

    /// One wrench: CLICK flips the tool loop, secondary-click opens the per-tool
    /// switches and the workspace.
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
    /// pre-send intent nudge, so the approval re-arm can't apply on one path
    /// and not the other.
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

    /// Which reply Cmd+R / the footer's Regenerate button targets — the
    /// last assistant message, so the button only ever shows on that one.
    private var lastAssistantMessageId: UUID? {
        session?.messages.last { $0.role == .assistant }?.id
    }

    /// Greeting + discovery chips, one fixed-height block. The vertical slack
    /// lives OUTSIDE this view (two sibling Spacers in the body) — a Spacer
    /// nested in here shares space unevenly with the body's own trailing one,
    /// which is what pinned the whole group to the bottom of the window.
    private var emptyState: some View {
        VStack(spacing: 8) {
            // Plain SF Pro, one solid colour. It was `design: .rounded` under a
            // top-to-bottom LinearGradient — a different typeface from the rest
            // of the app, wearing a fade that reads as a rendering artefact at
            // this size rather than as depth.
            // The agent's NAME, not the word "Agent" with the name as its
            // caption — the name is the thing, the category isn't (same
            // inversion the sidebar rows had). Under it, what the agent is
            // FOR, which is what tells you what to ask it.
            Text(ChatGreeting.heading(agentName: activeAgent?.name))
                .font(.system(size: 30, weight: .semibold))
                .foregroundStyle(.primary)
            if let subtitle = ChatGreeting.subtitle(agentBrief: activeAgent?.brief,
                                                    serverRunning: server.status == .running) {
                Text(subtitle)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            // Discovery chips: the features that otherwise live only in the
            // menu-bar tray (media generation, Model Browser, Tasks, the CLI
            // launcher). Under the greeting, gone once the conversation starts
            // — and absent entirely on an agent thread, where they advertise
            // the app to somebody who has already picked who to talk to.
            if ChatGreeting.showsDiscoveryChips(hasAgent: activeAgent != nil,
                                                isExternalBridge: session?.isExternalBridge == true) {
                // A media chip navigates to the Create pane, exactly like the
                // Tools menu — the composer never becomes a generator.
                EmptyStateChipRow()
                    .padding(.top, 18)
            }
        }
        // Same column as the transcript and the composer below it, so the
        // greeting sits over the field rather than spanning the window.
        .frame(maxWidth: contentWidth)
        .frame(maxWidth: .infinity)
        .padding(.horizontal, ChatMetrics.gutter)
        .padding(.bottom, 22)
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
                    LazyVStack(spacing: ChatMetrics.transcriptSpacing) {
                        ForEach(rows) { row in
                            switch row {
                            case .message(let m):
                                MessageBubble(
                                    message: m,
                                    sources: sourcesFor(m),
                                    onIncreaseContext: {
                                        appState.showSettings()
                                    },
                                    onDelete: {
                                        appState.deleteMessage(in: sessionId, messageId: m.id)
                                    },
                                    // Both roles are editable, and they mean
                                    // different things. Editing YOUR message
                                    // is a re-ask: the turns after it answered
                                    // something you no longer said, so they go
                                    // and it resubmits. Editing the MODEL's is
                                    // putting words in its mouth — that turn
                                    // already happened, and the point is to
                                    // steer what comes next (Continue, or the
                                    // following turn), so nothing is dropped
                                    // and nothing re-runs.
                                    onEdit: session?.isExternalBridge == true ? nil : { newText in
                                        if m.role == .user {
                                            editAndResend(messageId: m.id, newText: newText)
                                        } else {
                                            appState.editAssistantMessage(in: sessionId,
                                                                          messageId: m.id,
                                                                          newText: newText)
                                        }
                                    },
                                    onRegenerate: (m.role == .assistant && m.id == lastAssistantMessageId)
                                        ? { regenerateLastResponse() }
                                        : nil,
                                    // Only the last message: a continuation
                                    // streams into the END of the transcript,
                                    // so offering it on an earlier reply would
                                    // append the text to a different bubble.
                                    onContinue: (m.id == session?.messages.last?.id && canContinue)
                                        ? { continueReply() }
                                        : nil,
                                    onSelectRevision: { index in
                                        appState.selectRevision(in: sessionId,
                                                                messageId: m.id,
                                                                index: index)
                                    },
                                    // Branch here. Offered on both roles and at
                                    // any depth — unlike Continue and
                                    // Regenerate, which act on the END of the
                                    // transcript, a fork is a statement about
                                    // this message. A task run or a bridge
                                    // mirror can be branched INTO an ordinary
                                    // chat, so no read-only gate: nothing about
                                    // the source is changed.
                                    onFork: ChatFork.isForkable(session?.messages ?? [], at: m.id)
                                        ? { appState.forkSession(sessionId, from: m.id) }
                                        : nil)
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
                    // New identity when the text size or density changes, so
                    // every row rebuilds with the new metrics at once (see the
                    // @AppStorage pair above). Only fires on a Settings edit —
                    // the transcript isn't even visible then (Settings is a
                    // mode of this window), so the scroll reset is unseen.
                    .id("transcript-\(interfaceTextSize)-\(interfaceCompact)")
                    // The reading measure. The window is free to be as wide as
                    // the user wants; the prose is not (`ChatMetrics`).
                    .frame(maxWidth: contentWidth)
                    .frame(maxWidth: .infinity)
                    .padding(.horizontal, ChatMetrics.gutter)
                    .padding(.vertical, 20)
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
                // Pending attachment thumbnails (images + PDFs + videos + audio)
                if !pendingImages.isEmpty || !pendingPDFs.isEmpty || !pendingVideos.isEmpty || !pendingAudio.isEmpty {
                    AttachmentPreviewRow(images: $pendingImages, pdfs: $pendingPDFs, videos: $pendingVideos, audio: $pendingAudio)
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
                // "/" skill menu — FLOATS over the transcript rather than
                // sitting in the bar: as a sibling it grew the composer and
                // shoved the conversation up on every keystroke. An overlay
                // past the clip, its own bottom guide pinned to the
                // container's top, so it needs no height to position itself.
                // Not an NSPopover — that takes first responder, and the menu
                // has to stay open while you keep typing.
                .overlay(alignment: .topLeading) {
                    if !slashMatches.isEmpty {
                        SlashSkillMenu(matches: slashMatches,
                                       selection: min(slashSelection, max(0, slashMatches.count - 1)),
                                       onPick: { pickSlashSkill($0) })
                            .alignmentGuide(.top) { $0[.bottom] + 8 }
                            .shadow(color: .black.opacity(0.28), radius: 14, y: 4)
                    }
                }
                // Hover cards for the row's bare glyphs. Drawn HERE, past the
                // clip: an overlay on the control itself is cut off at the
                // container's rounded edge and lands over the text field.
                .composerTipOverlay()
              }   // end else (non-Telegram composer)
            }
            .frame(maxWidth: contentWidth)
            .frame(maxWidth: .infinity)
            .padding(.horizontal, ChatMetrics.gutter)
            .padding(.vertical, 10)
            // Once the transcript exists, the composer is the window's bottom
            // BAR and reads as one: a material band under a full-width divider
            // (the divider belongs to the transcript, above). In the empty
            // state it is a floating field under the greeting, so no band —
            // a bar across the middle of a blank window is a seam.
            .background(isEmptyConversation ? AnyShapeStyle(.clear) : AnyShapeStyle(.bar))
            // The top spacer's sibling — see the empty-state branch above.
            if isEmptyConversation { Spacer(minLength: 0) }
        }
        // Cmd+R — regenerate the last reply. A zero-size hidden button rather
        // than a window-level `.commands` entry: it needs THIS tab's session
        // and toolbar state, and disabling it here (rather than graying out a
        // menu item elsewhere) is what keeps it from firing mid-stream or on
        // an empty chat.
        .background(
            Button(action: regenerateLastResponse) { EmptyView() }
                .keyboardShortcut("r", modifiers: .command)
                .disabled(!canRegenerate)
                .frame(width: 0, height: 0)
                .opacity(0)
        )
        .onDrop(of: [.image, .pdf, .audio, .movie], isTargeted: nil) { providers in
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
                } else if videoSupported, provider.hasItemConformingToTypeIdentifier(UTType.movie.identifier) {
                    // Decode inside the closure — the temp URL is only valid here.
                    provider.loadFileRepresentation(forTypeIdentifier: UTType.movie.identifier) { url, _ in
                        guard let url = url else { return }
                        let name = url.lastPathComponent
                        let frames = VideoPreprocessor.extractFrames(url: url)
                        DispatchQueue.main.async {
                            if let frames, !frames.isEmpty {
                                pendingVideos.append(ChatVideo(name: name, frames: frames))
                            } else {
                                showVideoError(name)
                            }
                        }
                    }
                } else if let imageType = provider.registeredTypeIdentifiers.first(where: {
                    UTType($0)?.conforms(to: .image) == true
                }) {
                    // The image's OWN bytes, asked for by the type the provider
                    // actually REGISTERED — a Finder drag of a PNG registers
                    // `public.png`, not `public.image`, so a request for the
                    // umbrella type fails. `loadItem` hands back the file URL
                    // (Finder) or the data, and either way the encoding survives:
                    // `loadObject(ofClass: NSImage.self)` decodes to an NSImage
                    // and throws the bytes away, so a dropped PNG used to be
                    // stored as a re-encoded JPEG.
                    //
                    // The name comes from the URL when there is one and from
                    // `suggestedName` otherwise (`droppedName`); an extension
                    // neither supplies comes from the bytes themselves
                    // (`AttachmentStore.payload`).
                    //
                    // The fallback is for a provider that only offers the decoded
                    // object. It is not what makes a browser drag work: a drag
                    // out of a page registers `public.jpeg` with NO loader block
                    // behind it (measured, Brave), so nothing loads by any route
                    // and the drop does nothing — as it did before this change.
                    let suggested = provider.suggestedName
                    provider.loadItem(forTypeIdentifier: imageType) { item, _ in
                        let url = item as? URL
                        let bytes: Data? = (item as? Data) ?? url.flatMap { try? Data(contentsOf: $0) }
                        if let bytes, let image = NSImage(data: bytes) {
                            let name = AttachmentStore.droppedName(url: url, suggested: suggested)
                            DispatchQueue.main.async {
                                pendingImages.append(PendingImage(image: image, original: bytes, filename: name))
                            }
                            return
                        }
                        provider.loadObject(ofClass: NSImage.self) { image, _ in
                            guard let image = image as? NSImage else { return }
                            DispatchQueue.main.async {
                                pendingImages.append(PendingImage(image: image))
                            }
                        }
                    }
                }
            }
            return true
        }
        // No toolbar: everything that lived in it has a better home. The model
        // picker, the mode discs and the server control are in the COMPOSER row
        // (they configure the message, or report the thing you discover by
        // typing); Settings is a sidebar destination and still ⌘, from the menu
        // bar. What was left was an empty band across the top of the window,
        // so its MATERIAL is hidden — by `standardSplitView`, which hosts every
        // mode of this window, not here (on this view it covered only
        // conversation mode and the chrome flipped as you switched panes).
        //
        // What the material was FOR: it frosted content scrolling under the
        // floating toolbar cluster, and `scrollEdgeEffectStyle` needs a bar to
        // attach to (text clipped mid-line under the model picker, live
        // 2026-07-30). Both were about the CLUSTER, and the cluster is gone —
        // nothing floats over the transcript any more. What still passes under
        // something is the sidebar's pinned destinations, so that block carries
        // its own backdrop rather than relying on an effect with nothing to
        // attach to.
        //
        // Hiding the bar ITSELF (`.toolbar(.hidden)`) is the thing that must
        // not come back: the traffic lights and the sidebar-collapse button are
        // its residents, and it took them with it (live 2026-08-09).
        .sheet(isPresented: $showMCPMarketplace) {
            MCPMarketplaceView()
                .environmentObject(mcpManager)
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
            // The nudge exists to recommend this one, so it is what Return
            // takes. "Send Anyway" stays a deliberate click — it also
            // suppresses the suggestion for the rest of the chat, which is
            // not something to hand to a reflex.
            .keyboardShortcut(.defaultAction)
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
        .onChange(of: reasoningEffort) { _, newValue in
            guard !isExternalBridgeSession,
                  let idx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }) else { return }
            appState.chatSessions[idx].reasoningEffort = newValue
        }
        .onChange(of: mcpMode) { _, newValue in
            guard !isExternalBridgeSession,
                  let idx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }) else { return }
            appState.chatSessions[idx].useMCP = newValue
        }
        .onChange(of: composerState) { _, state in
            if state == .idle { inputFocused = true }
        }
        // The keyboard arriving in the composer collapses the sidebar selection
        // to this chat. Keyed on the focus MIRROR rather than on the click, so
        // it covers every way the field ends up holding the keyboard — and the
        // rule is what keeps ⌘⌫ from raising a delete dialog mid-message with
        // several chats still lit behind the field.
        .onChange(of: inputFocused) { _, focused in
            guard focused,
                  let collapsed = SidebarMultiSelect.focusingComposer(
                      in: sessionId, selection: appState.sidebarSelection)
            else { return }
            appState.sidebarSelection = collapsed
        }
        .onChange(of: session?.messages, initial: true) { _, msgs in
            rows = ChatRowBuilder.rows(from: msgs ?? [])
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
            // A history walk belongs to ONE conversation. Stale indexes are
            // harmless (ComposerHistory reads a mismatched draft as no walk),
            // but the first ↑ in the newly-visible tab has to mean "the last
            // thing I said HERE".
            composerWalk = .idle
        }
        .onGeometryChange(for: CGFloat.self) { proxy in
            proxy.size.width
        } action: { columnWidth = $0 }
    }

    /// The input field. No background or border of its own — the composer
    /// container draws those around both rows. NSTextView-backed so a big paste
    /// stays smooth and the mouse wheel scrolls once it grows past the cap.
    private var composerPlaceholder: String { "Ask me anything…" }

    private var composerField: some View {
        GrowingTextEditor(text: $inputText,
                          isFocused: $inputFocused,
                          measuredHeight: $composerHeight,
                          isIdle: composerState == .idle,
                          onSend: { sendMessage() },
                          // Escape stops the reply being written. Handled here
                          // rather than as a hidden `.cancelAction` button so
                          // the edit bubble and the approval sheet keep the key
                          // while they are up (ComposerKey.onEscape).
                          onCancel: {
                              switch ComposerKey.onEscape(isGenerating: composerState == .generatingHere) {
                              case .stop: stopGeneration(); return true
                              case .pass: return false
                              }
                          },
                          onArrow: { direction, caretAtStart, caretAtEnd in
                              recallHistory(direction, caretAtStart: caretAtStart, caretAtEnd: caretAtEnd)
                          },
                          onKeyCommand: { handleSlashKey($0) })
            .frame(height: max(ChatMetrics.composerMinHeight, composerHeight))
            .padding(.horizontal, ComposerTextMetrics.fieldHorizontalPadding)
            .disabled(server.status != .running)
            // The placeholder stands in for the first character you type, so it
            // has to sit exactly where that character lands — which is three
            // insets in, not one (`ComposerTextMetrics`). It was a literal 9
            // against a real 14, so the caret overlapped its own placeholder.
            // Escape closes the menu for the text as it stands; the next
            // keystroke is a new query, so it opens again.
            .onChange(of: inputText) { _, _ in
                if slashDismissed { slashDismissed = false }
                slashSelection = 0
            }
            .overlay(alignment: .topLeading) {
                if inputText.isEmpty {
                    Text(composerPlaceholder)
                        .font(.body)
                        .foregroundStyle(.secondary)
                        .padding(.leading, ComposerTextMetrics.placeholderLeading)
                        .padding(.top, ComposerTextMetrics.placeholderTop)
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

        // The model answering, right of the discs and left of the gauge. It
        // belongs to the MESSAGE — which model writes the reply — the same
        // reason Think/Tools/MCP moved down here, and it has room for the
        // download affordances the toolbar never did.
        ChatModelPill(compact: true)
        // The recovery goes where the problem is DISCOVERED: the pill's dot
        // going grey is the only thing that says the server is down, so the fix
        // sits next to it. Transient by construction (`ChatServerStartControl`
        // resolves to `.hidden` the moment it is up), which is what earns it a
        // slot in a row that is already at its width budget.
        serverStartControl

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
                stopGeneration()
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
                      && pendingImages.isEmpty && pendingPDFs.isEmpty && pendingVideos.isEmpty && pendingAudio.isEmpty))
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
        //
        // Take the pasteboard's PNG when it has one, so a copied PNG is stored
        // byte for byte. What it usually has is TIFF, which is UNCOMPRESSED: a
        // 3000x2000 screenshot is ~24 MB of it, so those bytes are deliberately
        // NOT kept and `AttachmentStore` encodes the image to PNG instead.
        if let image = NSImage(pasteboard: pb) {
            let png = pb.data(forType: .png)
            pendingImages.append(PendingImage(image: image, original: png))
            return true
        }
        return false
    }

    /// Dispatch one file URL to the matching attachment path. Returns false for
    /// unsupported types so the caller leaves the paste alone.
    private func attachFileURL(_ url: URL) -> Bool {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else { return false }
        switch PasteFileKind.classify(ext: url.pathExtension, isDirectory: isDir.boolValue, audioSupported: audioSupported, videoSupported: videoSupported) {
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
        case .video:
            addVideoAttachment(url)
        case .image:
            guard let bytes = try? Data(contentsOf: url),
                  let image = NSImage(data: bytes) else { return false }
            pendingImages.append(PendingImage(image: image, original: bytes,
                                              filename: url.lastPathComponent))
        case .unhandled:
            return false
        }
        return true
    }

    // MARK: - Image Helpers

    private func pickAttachment() {
        let panel = NSOpenPanel()
        // Only offer audio/video on models that can use them.
        var types: [UTType] = [.image, .pdf]
        if videoSupported { types.append(.movie) }
        if audioSupported { types.append(.audio) }
        panel.allowedContentTypes = types
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
                } else if videoSupported, let utType = UTType(filenameExtension: url.pathExtension), utType.conforms(to: .movie) {
                    addVideoAttachment(url)
                } else if let bytes = try? Data(contentsOf: url),
                          let image = NSImage(data: bytes) {
                    pendingImages.append(PendingImage(image: image, original: bytes,
                                                      filename: url.lastPathComponent))
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

    /// Extract frames from a video file (off the main thread — AVFoundation
    /// decode can be slow) and add it as a pending attachment.
    private func addVideoAttachment(_ url: URL) {
        let name = url.lastPathComponent
        DispatchQueue.global(qos: .userInitiated).async {
            let frames = VideoPreprocessor.extractFrames(url: url)
            DispatchQueue.main.async {
                if let frames, !frames.isEmpty {
                    pendingVideos.append(ChatVideo(name: name, frames: frames))
                } else {
                    showVideoError(name)
                }
            }
        }
    }

    private func showVideoError(_ name: String) {
        let alert = NSAlert()
        alert.messageText = "Couldn't read video"
        alert.informativeText = "\(name) couldn't be decoded."
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

    /// Whether the active model understands video (Qwen3-VL-family checkpoints
    /// declaring `video_token_id`). Gates the video-attach option so it only
    /// appears where the server can actually read it.
    private var videoSupported: Bool { server.chatModelInfo?.supportsVideo ?? false }

    /// Convert pending videos to a ChatVideo array, clearing the list.
    private func consumePendingVideos() -> [ChatVideo]? {
        guard !pendingVideos.isEmpty else { return nil }
        let videos = pendingVideos
        pendingVideos = []
        return videos
    }

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

    /// Write the pending attachments out and hand back the messages' images,
    /// clearing the pending list.
    ///
    /// Written on SEND, not on attach: the preview row lets an attachment be
    /// removed again, and a file per abandoned pick is how a folder fills up
    /// with things no conversation names.
    ///
    /// A failed write is not fatal to the turn. The `ChatImage` keeps its bytes,
    /// so the model still sees the picture and the transcript still draws it —
    /// only the next launch finds no file, and says so.
    private func consumePendingImages() -> [ChatImage]? {
        guard !pendingImages.isEmpty else { return nil }
        let chatImages = pendingImages.compactMap { pending -> ChatImage? in
            var image = ChatImage(data: Data())
            guard let payload = AttachmentStore.payload(for: pending, id: image.id) else { return nil }
            image.data = payload.data
            image.path = AttachmentStore.write(payload.data, named: payload.name)
            return image
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
            if case .geometryChanged = event {
                // `onScrollGeometryChange` delivers its action INSIDE the
                // window's layout flush, and `scrollPosition` is @State —
                // writing it there re-enters layout while AppKit is mid-flush.
                // Under a streaming re-layout storm (code block re-highlights,
                // then markdown re-measures) that loop is the #136 beachball,
                // and the write that lands at the wrong point in the flush is
                // the uncaught NSException crash (live crash log 2026-08-09:
                // StoredLocationBase.beginUpdate → setNeedsUpdateConstraints →
                // _crashOnException). One runloop turn later is outside the
                // flush, and coalesces the storm to one correction per turn.
                DispatchQueue.main.async { performScroll(animated: animated) }
            } else {
                // Button taps and sends run from event handling, not layout —
                // they stay synchronous so the jump lands with the click.
                performScroll(animated: animated)
            }
        }
    }

    private func performScroll(animated: Bool) {
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
        // Sending ends any history walk: the composer is about to empty for a
        // different reason, and a live walk would make the next ↑ resume from
        // wherever the last one left off rather than from what was just sent.
        composerWalk = .idle
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

    /// Decide whether to nudge before sending. The RULE is pure
    /// (`ComposerIntent.nudge` — MCP over Agent, never a mode that is on, locked
    /// by the tab's agent, or already declined here, and nothing at all when the
    /// user has set tools to opt-in); this only gathers what it reads.
    private func detectIntentPrompt(for text: String) -> IntentPrompt? {
        let toggles = toolbarToggles
        return ComposerIntent.nudge(
            for: text,
            onlyToolsWhenAsked: appState.serverOptions.toolsOnlyWhenAsked,
            toolsOn: isAgentMode, toolsLocked: toggles.toolsLockedBy != nil,
            mcpOn: mcpMode, mcpLocked: toggles.mcpLockedBy != nil,
            enabledServers: enabledMCPServerNames(),
            suppressed: intentSuppress, sessionId: sessionId)
    }

    /// Enable the mode the nudge suggested.
    private func enableForPrompt(_ prompt: IntentPrompt) {
        switch prompt {
        case .agent:
            isAgentMode = true
        case .mcp:
            mcpMode = true
        }
    }

    private func proceedSend() {
        var text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        let attachedImages = consumePendingImages()
        let attachedVideos = consumePendingVideos()
        let attachedAudio = consumePendingAudio()
        let pdfText = consumePendingPDFsAsText()
        guard !text.isEmpty || attachedImages != nil || attachedVideos != nil || attachedAudio != nil || !pdfText.isEmpty,
              composerState != .generatingHere, server.status == .running else { return }
        inputText = ""
        if !pdfText.isEmpty {
            text = text.isEmpty ? pdfText : pdfText + "\n\n" + text
        }

        chatEngine.runTurn(sessionId: sessionId, userText: text,
                           images: attachedImages, audio: attachedAudio,
                           config: buildTurnConfig(),
                           approval: { await requestToolApproval($0) })
        // Your own message always wins: sending from halfway up the history used
        // to leave you exactly there, because auto-follow was off and nothing
        // else scrolled.
        applyScroll(.userSentMessage)
    }

    /// The toolbar toggles are this surface's DEFAULTS; the tab's agent (if
    /// any) overrides what it declared. One builder, so no field is read from
    /// a global here — see ChatTurnEngine.TurnConfig. Shared by send, Cmd+R
    /// regenerate, and edit-and-resend so all three run under identical
    /// settings.
    private func buildTurnConfig() -> ChatTurnEngine.TurnConfig {
        let resolved = appState.resolvedAgentSettings(
            agentId: session?.agentId,
            toolsEnabled: isAgentMode,
            mcpEnabled: mcpMode,
            thinkingEnabled: enableThinking,
            autoApprove: false,
            workingDirectory: session?.workingDirectory,
            disabledTools: ChatSession.disabledToolKinds(session?.disabledTools ?? []),
            reasoningEffort: reasoningEffort)
        return ChatTurnEngine.TurnConfig.from(
            resolved, documentIndex: appState.documentIndexes[sessionId])
    }

    /// Cmd+R — regenerate the last reply. Mirrors the footer's Regenerate
    /// button; both funnel through `ChatTurnEngine.regenerate`, which drops
    /// the last user turn and resubmits it fresh.
    private var canRegenerate: Bool {
        server.status == .running && composerState != .generatingHere
            && session?.isExternalBridge != true
            && (session?.messages.contains { $0.role == .user } ?? false)
    }

    /// Whether the reply at the end of this transcript can be finished off.
    private var canContinue: Bool {
        ContinueReply.isEligible(session?.messages ?? [],
                                 serverRunning: server.status == .running,
                                 busy: composerState != .idle,
                                 engine: server.chatModelInfo?.engine)
    }

    /// Hand the last reply back to the model to finish. Same turn config as a
    /// send, so the continuation runs under the settings that produced it.
    private func continueReply() {
        guard canContinue else { return }
        chatEngine.continueReply(sessionId: sessionId, config: buildTurnConfig())
        // The text lands at the END of a reply already on screen, so follow it
        // down the same way a fresh send does.
        applyScroll(.userSentMessage)
    }

    /// ↑ / ↓ in the composer: bring back an earlier message of your own.
    ///
    /// The rule is `ComposerHistory`, which decides from the draft, the caret
    /// position and where the walk currently sits — everything here does is
    /// feed it this chat's history and write the answer back. Returning false
    /// hands the key to AppKit, which is what makes the arrows still move the
    /// caret inside a draft.
    private func recallHistory(_ direction: ComposerHistory.Direction,
                               caretAtStart: Bool, caretAtEnd: Bool) -> Bool {
        let entries = ComposerHistory.entries(session?.messages ?? [])
        let action = direction == .up
            ? ComposerHistory.up(draft: inputText, caretAtStart: caretAtStart,
                                 walk: composerWalk, entries: entries)
            : ComposerHistory.down(draft: inputText, caretAtEnd: caretAtEnd,
                                   walk: composerWalk, entries: entries)
        switch action {
        case .pass:
            return false
        case .recall(let text, let walk):
            inputText = text
            composerWalk = walk
            return true
        }
    }

    /// Stop this chat's turn. The ONE call both the composer's stop disc and
    /// Escape make — per-session, because other tabs may be generating and
    /// neither control may reach across.
    private func stopGeneration() {
        chatEngine.stop(sessionId: sessionId)
    }

    private func regenerateLastResponse() {
        guard canRegenerate else { return }
        chatEngine.regenerate(sessionId: sessionId, config: buildTurnConfig(),
                               approval: { await requestToolApproval($0) })
        // Same rule as a fresh send: the reply you just asked for is the thing
        // to be looking at, wherever in the history the request came from.
        applyScroll(.userSentMessage)
    }

    /// Edit a past user message in place, then resend it: everything from
    /// that message onward (its old reply, any tool chain) is dropped and the
    /// edited text runs as a brand-new turn — the standard "edit and resend"
    /// behaviour rather than silently rewriting history the model already
    /// answered.
    private func editAndResend(messageId: UUID, newText: String) {
        let trimmed = newText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let msgs = session?.messages,
              let idx = msgs.firstIndex(where: { $0.id == messageId })
        else { return }
        let images = msgs[idx].images
        let videos = msgs[idx].videos
        let audio = msgs[idx].audio
        appState.truncateMessages(in: sessionId, keepingFirst: idx)
        chatEngine.runTurn(sessionId: sessionId, userText: trimmed,
                           images: images, videos: videos, audio: audio,
                           config: buildTurnConfig(),
                           approval: { await requestToolApproval($0) })
        // An edit is started from wherever that message sits, which is by
        // definition not the bottom — follow the resend down to it.
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

/// Attaches double-click-to-edit, or attaches NOTHING at all.
///
/// A `nil` action must leave the view untouched rather than install a gesture
/// that does nothing: this one is `highPriorityGesture`, so merely EXISTING is
/// enough to beat `textSelection`'s own double-click-to-select-word. A
/// no-op-inside-the-closure version therefore reads as "double-click stopped
/// selecting words" on exactly the messages nobody can edit.
private struct DoubleClickToEdit: ViewModifier {
    let action: (() -> Void)?

    @ViewBuilder
    func body(content: Content) -> some View {
        if let action {
            content.highPriorityGesture(TapGesture(count: 2).onEnded { _ in action() })
        } else {
            content
        }
    }
}

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
    /// Edit this (user) message's text and resend it, dropping everything
    /// that followed. nil for assistant messages and read-only surfaces —
    /// same reasoning as `onDelete`.
    var onEdit: ((String) -> Void)?
    /// Regenerate this reply. Only ever set on the LAST assistant message —
    /// the caller decides that, so the bubble itself doesn't need to know its
    /// position in the transcript.
    var onRegenerate: (() -> Void)?
    /// Hand this reply back to the model to finish. Present only on the last
    /// message, and only when it is continuable (`ContinueReply.isEligible`).
    var onContinue: (() -> Void)?
    /// Show a different generated version of this reply.
    var onSelectRevision: ((Int) -> Void)?
    /// Branch the conversation here: everything up to this message becomes a
    /// new chat and this one is left alone. nil when there would be nothing to
    /// fork (`ChatFork.isForkable`) or on a read-only surface.
    var onFork: (() -> Void)?
    /// Explicit so the accordion HEADER can drive it, not just the chevron.
    @State private var thinkingExpanded = false
    @State private var isEditing = false
    @State private var editDraft = ""
    /// The edit field is the composer's field (`GrowingTextEditor`), so it
    /// needs the composer's two bindings: where the caret is, and how tall the
    /// text has grown.
    @State private var editFocused = false
    @State private var editHeight: CGFloat = 0

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
                //
                // A message carrying an image REF draws from its file instead
                // (ChatMediaAttachmentView). Generated images stopped shipping
                // bytes, but a history written before that still has both, and
                // drawing both would show the same picture twice. The ref wins:
                // it is the original the generator wrote, not a re-encode.
                if let images = message.images, !images.isEmpty,
                   message.media?.contains(where: { $0.kind == .image }) != true {
                    HStack(spacing: 4) {
                        ForEach(images) { img in
                            // No bytes: the file under `attachments/` is gone,
                            // or this message predates attachments on disk and
                            // its base64 is no longer read. Say so rather than
                            // leave a hole where a picture was.
                            if img.data.isEmpty {
                                Label("attachment no longer on disk", systemImage: "questionmark.folder")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 8)
                                    .background(.quaternary.opacity(0.4))
                                    .clipShape(RoundedRectangle(cornerRadius: 8))
                            } else if let nsImage = NSImage(data: img.data) {
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
                if isEditing, onEdit != nil {
                    editingContent
                } else if !message.content.isEmpty || message.isStreaming {
                    // Only the USER gets a bubble (`isBare` below). An assistant
                    // reply is the page's main content — long, formatted, full
                    // of code blocks and tables — and boxing it wastes the
                    // column's width and fights every block that wants the full
                    // measure. A tool-call summary keeps its card so it still
                    // reads as machinery, not prose.
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
                            // The user's own turn is plain text (no markdown
                            // render), so it needs the transcript size stated —
                            // otherwise your message and the reply to it are
                            // two different sizes in the same column.
                            Text(message.content)
                                .font(.system(size: ChatMetrics.transcriptFontSize))
                                // Same leading as the reply's renderer, or the
                                // two roles read at two densities.
                                .lineSpacing(ChatMetrics.userLineSpacing)
                                .textSelection(.enabled)
                        }
                        if message.isStreaming {
                            GeneratingIndicator()
                        }
                    }
                    // `.highPriorityGesture`, not `.onTapGesture` — the `Text`
                    // above has `.textSelection(.enabled)`, which installs its
                    // OWN double-click-to-select-word handling. A plain
                    // `.onTapGesture(count: 2)` sits behind that in the
                    // gesture hierarchy and never sees the second click, so
                    // double-click silently did nothing. `highPriorityGesture`
                    // makes this the one that wins.
                    //
                    // Which is exactly why the `onEdit` test is on the MODIFIER
                    // and not inside the closure: a message with no edit action
                    // (every assistant reply, every read-only surface) would
                    // otherwise still install a winning gesture that beats
                    // text selection and then does nothing — killing
                    // double-click-to-select-a-word on the replies, which is
                    // most of what anyone selects.
                    // …and why the model's replies are excluded even though they
                    // ARE editable now: the gesture wins over text selection,
                    // and an assistant reply is prose people select words in.
                    // Its edit is reached from the context menu instead.
                    .modifier(DoubleClickToEdit(
                        action: (onEdit == nil || message.role != .user) ? nil : { startEditing() }))
                    .padding(.horizontal, isBare ? 0 : ChatMetrics.bubblePaddingH)
                    .padding(.vertical, isBare ? 0 : ChatMetrics.bubblePaddingV)
                    .background(bubbleBackground)
                    .foregroundStyle(message.role == .user ? .white : .primary)
                    .clipShape(RoundedRectangle(cornerRadius: isBare ? 0 : ChatMetrics.bubbleCornerRadius))
                    .frame(maxWidth: .infinity, alignment: isBare ? .leading : .trailing)
                }

                // A cut reply's notice: DATA on the message, drawn as a footnote
                // under the bubble — never appended into content, which rides
                // back to the model as history.
                if let notice = message.truncationNotice, !message.isStreaming {
                    Text(notice.text)
                        .font(.callout)
                        .italic()
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .padding(.leading, isBare ? 0 : ChatMetrics.statsIndent)
                        .padding(.top, 2)
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
            if onEdit != nil {
                // Named for what it DOES: editing your own message re-asks the
                // question, editing the model's rewrites what it said.
                Button(message.role == .user ? "Edit & Resend" : "Edit Reply") { startEditing() }
            }
            if onRegenerate != nil {
                Button("Regenerate") { onRegenerate?() }
            }
            if let onFork {
                // Between the two destructive answers and Delete: a fork keeps
                // BOTH branches, so it belongs next to the ones that don't.
                Divider()
                Button("Branch Chat From Here", action: onFork)
            }
            if onDelete != nil {
                Button("Delete Message", role: .destructive) { onDelete?() }
            }
        }
    }

    /// Replaces the static bubble while editing: a growing multi-line field
    /// pre-filled with the message's current text, Cancel / Save beneath it.
    /// Save hands the new text to `onEdit`, which drops this message and
    /// everything after it and resubmits — same shape as a fresh send.
    ///
    /// It is the COMPOSER's field, not a `TextEditor`. Editing a message is a
    /// send, so it answers the keyboard like one: Return submits, Shift+Return
    /// breaks the line — `ComposerKey.onReturn` decides for both, and
    /// `editCanSubmit` is the same gate that dims Save, so the key and the
    /// button can never disagree about whether this draft can go.
    private var editingContent: some View {
        VStack(alignment: .trailing, spacing: 8) {
            GrowingTextEditor(text: $editDraft,
                              isFocused: $editFocused,
                              measuredHeight: $editHeight,
                              maxLines: 12,
                              isIdle: ComposerKey.editCanSubmit(editDraft),
                              onSend: { commitEdit() },
                              // Belt and braces: the Cancel button's key
                              // equivalent claims Escape first while this is up.
                              onCancel: { cancelEdit(); return true })
                .frame(height: max(ChatMetrics.composerMinHeight, editHeight))
                .padding(8)
                .background(Color(nsColor: .controlBackgroundColor)) // Distinct surface fill
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.accentColor.opacity(0.6), lineWidth: 1.5) // Glowing outline
                )
                .shadow(color: Color.black.opacity(0.25), radius: 6, x: 0, y: 3) // Depth

            HStack(spacing: 8) {
                Button("Cancel") { cancelEdit() }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .keyboardShortcut(.cancelAction)

                Button("Save") { commitEdit() }
                    .buttonStyle(.borderedProminent)
                    .disabled(!ComposerKey.editCanSubmit(editDraft))
            }
            .font(.caption)
        }
        .padding(4)
        .frame(maxWidth: .infinity, alignment: .trailing)
    }

    private func startEditing() {
        editDraft = message.content
        isEditing = true
        // Put the caret in the field the edit just opened — otherwise Return
        // is typed at whatever still holds focus (the composer below), which
        // sends a NEW message instead of the edit.
        editFocused = true
    }

    private func cancelEdit() {
        isEditing = false
        editFocused = false
        editDraft = ""
    }

    private func commitEdit() {
        guard ComposerKey.editCanSubmit(editDraft) else { return }
        let text = editDraft
        isEditing = false
        editFocused = false
        onEdit?(text)
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

            // Which version of this reply you are reading. Left of the actions
            // because it is a statement about the text above it, not another
            // thing to do to it — and it only exists once there is a choice.
            if MessageRevisions.isPagerVisible(message.revisions) {
                HStack(spacing: 1) {
                    footerButton("chevron.left", help: "Previous version of this reply") {
                        onSelectRevision?(MessageRevisions.step(index: message.activeRevision,
                                                                by: -1,
                                                                count: message.revisions.count))
                    }
                    .disabled(!MessageRevisions.canGoBack(index: message.activeRevision))
                    Text(MessageRevisions.label(index: message.activeRevision,
                                                count: message.revisions.count))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.tertiary)
                    footerButton("chevron.right", help: "Next version of this reply") {
                        onSelectRevision?(MessageRevisions.step(index: message.activeRevision,
                                                                by: 1,
                                                                count: message.revisions.count))
                    }
                    .disabled(!MessageRevisions.canGoForward(index: message.activeRevision,
                                                             count: message.revisions.count))
                }
            }

            HStack(spacing: 2) {
                footerButton("doc.on.doc", help: "Copy this reply") { copyMessage() }
                // The model's replies are editable but have no double-click
                // route into it (that gesture belongs to selecting a word), so
                // without this the only way in is a context menu nobody thinks
                // to open on a paragraph.
                if onEdit != nil, message.role == .assistant {
                    footerButton("pencil", help: "Edit this reply — then Continue to carry on from it") {
                        startEditing()
                    }
                }
                // Continue sits BEFORE regenerate: they are the two answers to
                // "this reply isn't what I need", and the non-destructive one
                // goes first — regenerate throws the reply away.
                if let onContinue {
                    footerButton("text.append",
                                 help: message.truncationNotice != nil
                                     ? "Finish this reply — it was cut short"
                                     : "Keep writing from where this left off",
                                 action: onContinue)
                }
                if let onRegenerate {
                    footerButton("arrow.clockwise", help: "Regenerate this reply (⌘R)",
                                 action: onRegenerate)
                }
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
enum ChatRow: Identifiable, Equatable {
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
    @Environment(\.colorScheme) private var colorScheme

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

    private var latexTheme: LaTeXTheme {
        colorScheme == .dark ? .dark : .light
    }

    var body: some View {
        // Fenced code renders as its own view (colors, copy button);
        // everything between fences — including tables, rendered as an
        // `NSTextTable` inside the shared attributed string — stays in ONE
        // text view per run so drag-selection still crosses paragraphs,
        // lists, and tables. See `MarkdownSegmenter` for why the split is at
        // fences, not at blocks.
        VStack(alignment: .leading, spacing: 4) {
            ForEach(Array(MarkdownSegmenter.segments(source).enumerated()), id: \.offset) { _, segment in
                switch segment {
                case .prose(let text):
                    ForEach(Array(Self.latexBlocks(in: text).enumerated()), id: \.offset) { _, block in
                        switch block {
                        case .prose(let prose):
                            SelectableMarkdownNSText(
                                attributed: Self.attributedString(for: prose, theme: latexTheme)
                            )
                        case .display(let latex, let raw):
                            DisplayLaTeXView(latex: latex, raw: raw, theme: latexTheme)
                        }
                    }
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

    private enum LaTeXBlock {
        case prose(String)
        case display(latex: String, raw: String)
    }

    /// Display formulas own a horizontally scrollable view. Text and inline
    /// formulas remain one prose run so NSTextView can preserve native
    /// selection across paragraphs, lists, and inline equations.
    private static func latexBlocks(in source: String) -> [LaTeXBlock] {
        var blocks: [LaTeXBlock] = []
        var prose = ""

        func flushProse() {
            guard !prose.isEmpty else { return }
            blocks.append(.prose(prose))
            prose = ""
        }

        for segment in LaTeXSegmenter.segments(source) {
            switch segment {
            case .text(let text):
                prose += text
            case .inline(_, let raw):
                prose += raw
            case .display(let latex, let raw):
                flushProse()
                blocks.append(.display(latex: latex, raw: raw))
            }
        }
        flushProse()
        if blocks.isEmpty { blocks.append(.prose(source)) }
        return blocks
    }

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

            // Table (GFM pipe table, or the whitespace-aligned pseudo-table
            // smaller models emit without GFM syntax) — checked before
            // list-item/paragraph so a `- ` inside a cell or numbered header
            // doesn't get misread as a list marker.
            if let table = MarkdownTable.parse(lines: lines, start: i) {
                blocks.append(.table(table.headers, table.rows, table.alignments))
                i = table.end
                continue
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

    // MARK: NSAttributedString assembly

    /// Rendered prose runs, keyed by source + appearance because inline math
    /// attachments are rasterized in the resolved label color.
    private static let renderCache: NSCache<NSString, NSAttributedString> = {
        let c = NSCache<NSString, NSAttributedString>()
        c.countLimit = 256
        return c
    }()

    /// Build the NSAttributedString fed to NSTextView. Public-static so the
    /// rendering path can be exercised by tests later if needed.
    static func attributedString(for source: String) -> NSAttributedString {
        attributedString(for: source, theme: .light)
    }

    static func attributedString(for source: String, theme: LaTeXTheme) -> NSAttributedString {
        // The text size rides the key: fonts are baked into the cached string,
        // so a Settings ▸ Interface change with the old key would hand every
        // re-rendered row back at the size it was built at.
        let key = "\(theme.rawValue)\u{0}\(ChatMetrics.transcriptFontSize)\u{0}\(source)" as NSString
        if let hit = renderCache.object(forKey: key) { return hit }
        let built = buildAttributedString(for: source, theme: theme)
        renderCache.setObject(built, forKey: key)
        return built
    }

    private static func buildAttributedString(
        for source: String,
        theme: LaTeXTheme
    ) -> NSAttributedString {
        let result = NSMutableAttributedString()
        let blocks = parseBlocks(source: source)
        for (idx, block) in blocks.enumerated() {
            if idx > 0 { result.append(blockSpacer()) }
            switch block {
            case .paragraph(let text):
                // Leading + a real gap after each paragraph (single-newline
                // "**Label.** text" runs the models love are paragraphs too),
                // and the reading measure as a POSITIVE tailIndent — an
                // absolute wrap point, so prose stops at ~45em while tables,
                // code and XML keep the full column.
                let p = NSMutableParagraphStyle()
                p.lineHeightMultiple = ChatMetrics.proseLineHeightMultiple
                p.paragraphSpacing = 8
                p.tailIndent = ChatMetrics.proseMeasure
                let para = NSMutableAttributedString(attributedString: renderInline(text, theme: theme))
                para.addAttribute(.paragraphStyle, value: p,
                                  range: NSRange(location: 0, length: para.length))
                result.append(para)

            case .heading(let level, let text):
                // Scaled from the body size, so raising the reading size
                // raises the headings with it instead of flattening them.
                let base = ChatMetrics.transcriptFontSize
                let size: CGFloat = level == 1 ? base + 5 : level == 2 ? base + 3 : base + 1
                let p = NSMutableParagraphStyle()
                p.paragraphSpacingBefore = 10
                p.paragraphSpacing = 2
                p.lineHeightMultiple = ChatMetrics.proseLineHeightMultiple
                p.tailIndent = ChatMetrics.proseMeasure
                let heading = NSMutableAttributedString(
                    attributedString: renderInline(text, theme: theme, fontSize: size)
                )
                let full = NSRange(location: 0, length: heading.length)
                heading.addAttributes([
                    .font: NSFont.systemFont(ofSize: size, weight: .bold),
                    .paragraphStyle: p,
                ], range: full)
                result.append(heading)

            case .code(_, let content):
                let p = NSMutableParagraphStyle()
                p.paragraphSpacingBefore = 4
                p.paragraphSpacing = 4
                p.firstLineHeadIndent = 8
                p.headIndent = 8
                p.tailIndent = -8
                // Less air than prose — a listing wants rows. And never the
                // prose measure: code keeps the full column.
                p.lineHeightMultiple = ChatMetrics.codeLineHeightMultiple
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: NSFont.monospacedSystemFont(ofSize: ChatMetrics.transcriptCodeFontSize, weight: .regular),
                    .backgroundColor: NSColor.textBackgroundColor.blended(withFraction: 0.85, of: .black) ?? NSColor.darkGray,
                    .foregroundColor: NSColor(white: 0.92, alpha: 1.0),
                    .paragraphStyle: p,
                ]
                let code = NSMutableAttributedString(string: content, attributes: attrs)
                linkifyBareUrls(code)
                result.append(code)

            case .listItem(let text):
                let bullet = NSAttributedString(string: "• ", attributes: [
                    .font: NSFont.systemFont(ofSize: ChatMetrics.transcriptFontSize),
                    .foregroundColor: NSColor.secondaryLabelColor,
                ])
                let p = NSMutableParagraphStyle()
                // Hanging indent measured off the bullet itself, so wrapped
                // lines align under the text at every text size.
                p.headIndent = bullet.size().width.rounded(.up)
                p.lineHeightMultiple = ChatMetrics.proseLineHeightMultiple
                p.paragraphSpacing = 4
                p.tailIndent = ChatMetrics.proseMeasure
                let inline = renderInline(text, theme: theme)
                let combined = NSMutableAttributedString()
                combined.append(bullet)
                combined.append(inline)
                combined.addAttribute(.paragraphStyle, value: p, range: NSRange(location: 0, length: combined.length))
                result.append(combined)

            case .table(let headers, let rows, let alignments):
                result.append(renderTable(headers: headers, rows: rows, alignments: alignments, theme: theme))

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

            }
        }
        return result
    }

    /// Render a table as an `NSTextTable` run appended directly into the
    /// message's continuous attributed string, so drag-selection and copy
    /// span prose and table cells together like every other block instead of
    /// stopping at the table's edge. `NSTextTable` has no "tight to content"
    /// sizing mode — every column is a percentage of the available width —
    /// so columns are weighted by content length
    /// reading column.
    ///
    /// Cell content is rendered through `renderCell`, memoized by cell text
    /// - a table's earlier rows/cells don't change once streamed, only the
    /// actively-growing last cell does, so this amortizes the per-cell
    /// markdown/LaTeX-segmentation cost to near zero after the first full
    /// render instead of repeating it for every unchanged cell on every
    /// token.
    private static let cellRenderCache: NSCache<NSString, NSAttributedString> = {
        let c = NSCache<NSString, NSAttributedString>()
        c.countLimit = 2048
        return c
    }()

    private static func renderCell(_ text: String, theme: LaTeXTheme, bold: Bool) -> NSAttributedString {
        let key = "\(theme.rawValue)\u{0}\(bold)\u{0}\(text)" as NSString
        if let hit = cellRenderCache.object(forKey: key) { return hit }
        let rendered = renderInline(text, theme: theme, weight: bold ? .semibold : .regular)
        cellRenderCache.setObject(rendered, forKey: key)
        return rendered
    }

    private static func renderTable(
        headers: [String],
        rows: [[String]],
        alignments: [TableAlignment],
        theme: LaTeXTheme
    ) -> NSAttributedString {
        let result = NSMutableAttributedString()
        let cols = headers.count
        guard cols > 0 else { return result }

        let table = NSTextTable()
        table.numberOfColumns = cols
        let fractions = MarkdownTable.columnFractions(headers: headers, rows: rows)
        let dividerColor = NSColor.separatorColor

        func textAlignment(_ column: Int) -> NSTextAlignment {
            guard column < alignments.count else { return .left }
            switch alignments[column] {
            case .left: return .left
            case .right: return .right
            case .center: return .center
            }
        }

        func appendRow(_ cells: [String], rowIndex: Int, bold: Bool) {
            for column in 0..<cols {
                let text = column < cells.count ? cells[column] : ""
                let block = NSTextTableBlock(
                    table: table, startingRow: rowIndex, rowSpan: 1,
                    startingColumn: column, columnSpan: 1
                )
                block.setContentWidth(Double(fractions[column]) * 100, type: .percentageValueType)
                block.setWidth(6, type: .absoluteValueType, for: .padding)
                // Divider under the header row only — no vertical borders,
                // no rules between data rows, matching the "minimal GFM"
                // look chat UIs use.
                if rowIndex == 0 {
                    block.setBorderColor(dividerColor, for: .maxY)
                    block.setWidth(1, type: .absoluteValueType, for: .border, edge: .maxY)
                }
                let pStyle = NSMutableParagraphStyle()
                pStyle.textBlocks = [block]
                pStyle.alignment = textAlignment(column)
                let cell = NSMutableAttributedString(attributedString: renderCell(text, theme: theme, bold: bold))
                cell.append(NSAttributedString(string: "\n"))
                cell.addAttribute(.paragraphStyle, value: pStyle, range: NSRange(location: 0, length: cell.length))
                result.append(cell)
            }
        }

        appendRow(headers, rowIndex: 0, bold: true)
        for (index, row) in rows.enumerated() {
            appendRow(row, rowIndex: index + 1, bold: false)
        }
        return result
    }

    /// One-and-a-half blank lines between blocks. Encoded as a `\n` with extra
    /// paragraph spacing so tall blocks don't collapse.
    private static func blockSpacer() -> NSAttributedString {
        let p = NSMutableParagraphStyle()
        p.paragraphSpacing = 8
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
    private struct InlineMathReplacement {
        let marker: String
        let latex: String
        let raw: String
    }
    static func renderInline(
        _ text: String,
        theme: LaTeXTheme,
        weight: NSFont.Weight = .regular,
        fontSize: CGFloat = ChatMetrics.transcriptFontSize
    ) -> NSAttributedString {
        let bodyFont = NSFont.systemFont(ofSize: fontSize, weight: weight)
        let prepared = inlineMathPlaceholders(in: text)
        var result = markdownAttributedString(prepared.source)

        let located = prepared.replacements.compactMap { replacement -> (InlineMathReplacement, NSRange)? in
            let range = (result.string as NSString).range(of: replacement.marker)
            return range.location == NSNotFound ? nil : (replacement, range)
        }
        // Foundation should preserve private-use marker scalars. If a future
        // parser version does not, keep every source byte visible instead of
        // silently dropping an equation.
        if located.count != prepared.replacements.count {
            result = markdownAttributedString(text)
            applyInlineTypography(to: result, bodyFont: bodyFont)
            linkifyBareUrls(result)
            return result
        }

        applyInlineTypography(to: result, bodyFont: bodyFont)
        for (replacement, range) in located.reversed() {
            let inherited = result.attributes(at: range.location, effectiveRange: nil)
            let rendered = InlineLaTeXRenderer.attributedAttachment(
                latex: replacement.latex,
                raw: replacement.raw,
                theme: theme,
                fontSize: fontSize
            )
            let inserted: NSMutableAttributedString
            if let rendered {
                inserted = NSMutableAttributedString(attributedString: rendered)
            } else {
                inserted = NSMutableAttributedString(string: replacement.raw)
            }
            inserted.addAttributes(
                inherited,
                range: NSRange(location: 0, length: inserted.length)
            )
            result.replaceCharacters(in: range, with: inserted)
        }
        linkifyBareUrls(result)
        return result
    }

    private static func inlineMathPlaceholders(
        in text: String
    ) -> (source: String, replacements: [InlineMathReplacement]) {
        var source = ""
        var replacements: [InlineMathReplacement] = []

        for segment in LaTeXSegmenter.segments(text) {
            switch segment {
            case .text(let value):
                source += value
            case .display(_, let raw):
                // Display equations are split into their own SwiftUI view by
                // latexBlocks(in:). Direct renderer callers keep them literal.
                source += raw
            case .inline(let latex, let raw):
                var marker = "\u{E000}mlxlatex\(replacements.count)\u{E001}"
                while text.contains(marker) || source.contains(marker) {
                    marker += "x"
                }
                source += marker
                replacements.append(InlineMathReplacement(marker: marker, latex: latex, raw: raw))
            }
        }
        return (source, replacements)
    }

    private static func markdownAttributedString(_ text: String) -> NSMutableAttributedString {
        if let attr = try? AttributedString(
            markdown: text,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        ) {
            return NSMutableAttributedString(attr)
        }
        return NSMutableAttributedString(string: text)
    }

    private static func applyInlineTypography(
        to result: NSMutableAttributedString,
        bodyFont: NSFont
    ) {
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

}

/// A complete display equation is its own horizontally scrollable surface.
/// SwaTex's MathView performs all parsing/layout/drawing; malformed model
/// output falls back to the exact delimiters and TeX the model streamed.
fileprivate struct DisplayLaTeXView: View {
    let latex: String
    let raw: String
    let theme: LaTeXTheme

    private var fontSize: CGFloat { ChatMetrics.transcriptFontSize + 4 }

    var body: some View {
        if DisplayLaTeXRenderer.canRender(latex, theme: theme, fontSize: fontSize) {
            ScrollView(.horizontal) {
                MathView(latex)
                    .font(size: fontSize)
                    .mathColor(SwiftUI.Color.primary)
                    .padding(.vertical, 2)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .contextMenu {
                Button("Copy Equation") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(raw, forType: .string)
                }
            }
        } else {
            SelectableMarkdownNSText(attributed: NSAttributedString(
                string: raw,
                attributes: [
                    .font: NSFont.systemFont(ofSize: ChatMetrics.transcriptFontSize),
                    .foregroundColor: NSColor.labelColor,
                ]
            ))
        }
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
    /// Answering costs a full `ensureLayout` of the run, and auto-layout asks
    /// several times per pass — so the answer is cached until something that can
    /// actually change it happens.
    private var cachedHeight: CGFloat?

    override var intrinsicContentSize: NSSize {
        if let cachedHeight { return NSSize(width: NSView.noIntrinsicMetric, height: cachedHeight) }
        guard let lm = layoutManager, let tc = textContainer else {
            return super.intrinsicContentSize
        }
        lm.ensureLayout(for: tc)
        let height = ceil(lm.usedRect(for: tc).height)
        cachedHeight = height
        return NSSize(width: NSView.noIntrinsicMetric, height: height)
    }

    override func invalidateIntrinsicContentSize() {
        cachedHeight = nil
        super.invalidateIntrinsicContentSize()
    }

    override func didChangeText() {
        super.didChangeText()
        invalidateIntrinsicContentSize()
    }

    override func copy(_ sender: Any?) {
        let range = selectedRange()
        guard range.length > 0, let textStorage else {
            super.copy(sender)
            return
        }

        var containsLaTeX = false
        textStorage.enumerateAttribute(.mlxLaTeXSource, in: range) { value, _, stop in
            if value != nil {
                containsLaTeX = true
                stop.pointee = true
            }
        }
        guard containsLaTeX else {
            super.copy(sender)
            return
        }

        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(
            LaTeXCopyText.string(from: textStorage, range: range),
            forType: .string
        )
    }

    override func setFrameSize(_ newSize: NSSize) {
        // This view WRAPS, so only a width change can alter its height.
        // Invalidating on any frame change fed the height we ourselves just
        // reported straight back in as a fresh invalidation — layout, invalidate,
        // layout again, several times per frame.
        let widthChanged = abs(newSize.width - frame.width) > 0.5
        super.setFrameSize(newSize)
        if widthChanged { invalidateIntrinsicContentSize() }
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

/// What an Escape keypress does in the composer. `.pass` hands the key back to
/// AppKit rather than swallowing it — with no turn to stop, Escape still has
/// its platform meaning here.
enum ComposerEscapeAction: Equatable { case stop, pass }

/// A key the composer offers to the "/" menu before handling it itself.
enum ComposerKeyCommand { case up, down, accept, cancel }

enum ComposerKey {
    static func onReturn(shift: Bool, isIdle: Bool) -> ComposerReturnAction {
        if shift { return .newline }
        return isIdle ? .send : .ignore
    }

    /// Escape stops the reply being written, and does nothing otherwise.
    ///
    /// This runs from `cancelOperation(_:)` — the RESPONDER CHAIN — not from a
    /// `.keyboardShortcut(.cancelAction)`. Key equivalents are offered the
    /// keystroke first, so the edit bubble's Cancel and the tool-approval
    /// sheet's Deny keep Escape whenever they are on screen, and the composer
    /// only sees it when neither is. That ordering is the whole reason this
    /// is not a hidden button like ⌘R regenerate.
    static func onEscape(isGenerating: Bool) -> ComposerEscapeAction {
        isGenerating ? .stop : .pass
    }

    /// Whether an in-place message edit is submittable — the ONE gate the Save
    /// button and the Return key both read. A resubmit drops the old reply and
    /// everything after it, so a blanked draft must not be able to spend that:
    /// the edit field feeds this to `onReturn`'s `isIdle`, which turns a bare
    /// Return on nothing into `.ignore` rather than a send.
    static func editCanSubmit(_ draft: String) -> Bool {
        !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
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
    /// Escape, from the responder chain. Defaults to nothing so a field that
    /// has no use for the key leaves it to AppKit.
    var onCancel: () -> Bool = { false }
    /// ↑ / ↓, with WHERE THE CARET IS — the composer recalls earlier messages,
    /// but only from the edge of the text, so the keys keep moving the caret
    /// inside a multi-line draft (`ComposerHistory`). Returning false hands the
    /// key back to AppKit. Defaults to nothing: the in-place message editor has
    /// no history to walk.
    var onArrow: (ComposerHistory.Direction, _ caretAtStart: Bool, _ caretAtEnd: Bool) -> Bool = { _, _, _ in false }
    /// Keys the "/" menu wants first. Returns true when it consumed one — the
    /// editor's own Return handling, Escape and arrow recall only run when
    /// nothing above claimed the key.
    var onKeyCommand: (ComposerKeyCommand) -> Bool = { _ in false }

    /// Read from the SAME constants the placeholder overlay reads — the two
    /// are related only by arithmetic nobody's type system checks.
    let inset = NSSize(width: ComposerTextMetrics.containerInsetWidth,
                       height: ComposerTextMetrics.containerInsetHeight)

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
        tv.textContainer?.lineFragmentPadding = ComposerTextMetrics.lineFragmentPadding
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
            // Caret to the END of whatever was just put in the field. Setting
            // `string` collapses the selection to the front, which after a
            // history recall means typing lands BEFORE the recalled message and
            // ↓ (which arms at the end) can never walk back out of it.
            tv.setSelectedRange(NSRange(location: (text as NSString).length, length: 0))
            context.coordinator.recomputeHeight()
        }
        let enabled = context.environment.isEnabled
        if tv.isEditable != enabled { tv.isEditable = enabled }
        // Drive AppKit first-responder from the SwiftUI focus mirror — on the
        // EDGE of a request, never on its level.
        //
        // This ran on every update while `isFocused` was true, which made the
        // field impossible to leave: `onResignFocus` publishes the cleared flag
        // ASYNCHRONOUSLY, so any update in between saw a still-true flag and a
        // text view that no longer had the keyboard, and handed it straight
        // back. `isFocused` is set true on appear and again whenever a turn goes
        // idle, and nothing ever cleared it, so the composer held the keyboard
        // for the life of the window — and ⌘⌫ read "the user is typing" forever
        // (live 2026-08-12).
        if isFocused != context.coordinator.appliedFocus {
            context.coordinator.appliedFocus = isFocused
            if isFocused, tv.window != nil, tv.window?.firstResponder !== tv {
                DispatchQueue.main.async { tv.window?.makeFirstResponder(tv) }
            }
        }
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: GrowingTextEditor
        weak var textView: ComposerTextView?
        /// The last focus REQUEST this view acted on, so `updateNSView` can
        /// tell "focus me" from "you are still notionally focused" — see the
        /// note there.
        var appliedFocus = false
        init(_ parent: GrowingTextEditor) { self.parent = parent }

        func textDidChange(_ notification: Notification) {
            guard let tv = textView else { return }
            if parent.text != tv.string { parent.text = tv.string }
            recomputeHeight()
        }

        func textView(_ textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
            // The "/" skill menu gets FIRST REFUSAL on every key it navigates
            // with, and only a menu that is actually open claims one. When it
            // declines, the composer's own bindings below run unchanged — so
            // Escape still stops a reply and ↑/↓ still walk history whenever
            // the menu is closed. Order matters both ways: an open menu that
            // let Escape through would stop the generation instead of closing
            // itself, and arrows reaching history would scroll the transcript
            // out from under the highlighted row.
            switch commandSelector {
            case #selector(NSResponder.moveUp(_:)):
                if parent.onKeyCommand(.up) { return true }
            case #selector(NSResponder.moveDown(_:)):
                if parent.onKeyCommand(.down) { return true }
            case #selector(NSResponder.insertTab(_:)):
                if parent.onKeyCommand(.accept) { return true }
            case #selector(NSResponder.cancelOperation(_:)):
                if parent.onKeyCommand(.cancel) { return true }
            default:
                break
            }
            // Escape. Reached only when no `.cancelAction` key equivalent is on
            // screen to claim it first — see ComposerKey.onEscape.
            if commandSelector == #selector(NSResponder.cancelOperation(_:)) {
                return parent.onCancel()
            }
            // ↑ / ↓ recall earlier messages, but the caret position is what
            // decides — a draft is regularly several lines tall, and swallowing
            // the arrows inside one would make it uneditable. The rule needs
            // both edges, so both are measured here rather than inferred: the
            // caret is at the start and the end simultaneously in an empty
            // field, which is the state the walk arms from.
            if commandSelector == #selector(NSResponder.moveUp(_:))
                || commandSelector == #selector(NSResponder.moveDown(_:)) {
                let selection = textView.selectedRange()
                let length = (textView.string as NSString).length
                let collapsed = selection.length == 0
                return parent.onArrow(
                    commandSelector == #selector(NSResponder.moveUp(_:)) ? .up : .down,
                    collapsed && selection.location == 0,
                    collapsed && selection.location == length)
            }
            guard commandSelector == #selector(NSResponder.insertNewline(_:)) else { return false }
            // Return picks the highlighted command when the menu is open —
            // it must not send a half-typed "/mus".
            if parent.onKeyCommand(.accept) { return true }
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
