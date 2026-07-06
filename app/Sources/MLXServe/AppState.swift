import Combine
import Foundation
import SwiftUI

@MainActor
class AppState: ObservableObject {
    @Published var server = ServerManager()
    private var cancellables = Set<AnyCancellable>()
    @Published var downloads = DownloadManager()
    @Published var localModels: [LocalModel] = []
    @Published var selectedModelPath: String = "" {
        didSet {
            UserDefaults.standard.set(selectedModelPath, forKey: "selectedModelPath")
            guard oldValue != selectedModelPath, !selectedModelPath.isEmpty else { return }
            // Drafter auto-sync: a drafter is paired to a specific Gemma 4
            // size (E2B / E4B / 31B / 26B-A4B). Switching the base model
            // without swapping the drafter to match crashes the server with
            // `DrafterTargetMismatch`. If the user already had a drafter on,
            // try to find the matching one for the new model — fall back to
            // clearing it (auto-disable) when no match is on disk or the new
            // model isn't Gemma 4.
            if !serverOptions.drafterPath.isEmpty {
                if let match = downloads.recommendedDrafterFromPath(selectedModelPath) {
                    serverOptions.drafterPath = match.url.path
                } else {
                    serverOptions.drafterPath = ""
                }
            }
            // Plan 05 Phase G — when hot-switch is enabled AND the server is
            // already running, ask the server to load the new model in-place
            // instead of restarting. Falls back to restart on failure (404
            // because the new path isn't in --model-dir, 503 if out of
            // memory, etc.). Restart path remains the default for clients
            // that don't opt in.
            if (server.status == .running || server.status == .starting) {
                if hotSwitchEnabled, server.status == .running {
                    let id = (selectedModelPath as NSString).lastPathComponent
                    let drafterPath: String? = downloads.recommendedDrafterFromPath(selectedModelPath)?.url.path
                    let mgr = server
                    Task { @MainActor in
                        do {
                            _ = try await mgr.loadModel(id: id, drafterPath: drafterPath)
                        } catch {
                            // Hot-switch failed (likely 404 if the model isn't
                            // under --model-dir on the running server). Fall
                            // back to a full restart so the user's choice still
                            // takes effect.
                            print("[AppState] hot-switch failed (\(error)) — falling back to restart")
                            mgr.stop()
                            mgr.start(modelPath: self.selectedModelPath, options: self.serverOptions)
                        }
                    }
                } else {
                    server.stop()
                    server.start(modelPath: selectedModelPath, options: serverOptions)
                }
            }
        }
    }
    /// Plan 05 Phase G — when true, model picker changes call /v1/load-model
    /// on the running server instead of restarting. Falls back to restart on
    /// failure. Defaults off so existing behavior is unchanged for users who
    /// haven't opted in.
    @Published var hotSwitchEnabled: Bool {
        didSet { UserDefaults.standard.set(hotSwitchEnabled, forKey: "hotSwitchEnabled") }
    }
    @Published var chatSessions: [ChatSession] = []
    @Published var activeChatId: UUID?
    /// Set when a task notification is tapped — the Tasks window observes this to
    /// focus the relevant task, then clears it.
    @Published var pendingTaskDeepLink: UUID?
    /// Set by the menu bar's Voice action; the chat detail view consumes it to
    /// auto-start Voice mode (whether the window was already open or just opened).
    @Published var pendingVoiceLaunch = false
    @Published var agentMemory = AgentMemory()
    @Published var toolExecutor = ToolExecutor()
    /// Owns every agent-spawned background process (started via shell
    /// run_in_background, or adopted by the foreground timeout backstop).
    /// In-memory only — all processes die with the app (and are reaped on quit
    /// by the registry's own willTerminate observer).
    @Published var processRegistry = ProcessRegistry()
    /// Per-session attached document folders (mini RAG). In-memory only — an
    /// index dies with the app and is rebuilt by re-attaching the folder.
    @Published var documentIndexes: [UUID: DocumentIndex] = [:]
    let testServer = TestServer()
    lazy var imageGen = ImageGenService()
    lazy var videoGen = VideoGenService()
    lazy var audioGen = AudioGenService()
    lazy var musicGen = MusicGenService()
    lazy var model3dGen = Model3DGenService()
    @Published var autoStartServer: Bool {
        didSet { UserDefaults.standard.set(autoStartServer, forKey: "autoStartServer") }
    }
    /// All server-launch flags + per-request defaults, mirrored to UserDefaults.
    /// Auto-saves on every mutation. Prefer this over the legacy single-key
    /// `maxTokens`/`contextSize` defaults — those forward into here.
    @Published var serverOptions: ServerOptions {
        didSet {
            serverOptions.save()
            // Reconcile the Telegram bridge whenever options change (cheap no-op
            // unless the bot token / enabled flag actually moved).
            telegramBridge.reconcile()
            // Push the agent-sandbox setting to the shared manager so the next
            // shell command routes to the guest (or the host) accordingly.
            AgentSandbox.shared.configure(enabled: serverOptions.sandbox.enabled,
                                          baseImage: serverOptions.sandbox.baseImage,
                                          network: serverOptions.sandbox.network)
        }
    }
    /// Legacy bridge: `maxTokens` is now stored in `serverOptions.defaultMaxTokens`.
    /// Existing call sites (StatusMenuView max-tokens slider, TestServer agent
    /// loops) keep the old name — both reads and writes route through the new
    /// canonical field so changes show up in Settings instantly.
    var maxTokens: Int {
        get { serverOptions.defaultMaxTokens }
        set { serverOptions.defaultMaxTokens = newValue }
    }
    /// Legacy bridge: `contextSize` is now `serverOptions.ctxSize`.
    var contextSize: Int {
        get { serverOptions.ctxSize }
        set { serverOptions.ctxSize = newValue }
    }
    @Published var mcpMode: Bool {
        didSet { UserDefaults.standard.set(mcpMode, forKey: "mcpMode") }
    }
    let mcpManager = MCPManager()

    /// In-app updater against the GitHub releases page. App-level (not a view)
    /// so the daily background check runs with every window closed; views
    /// observe it directly (`UpdateTrayRow(updates:)`), same pattern as
    /// `telegramBridge`.
    let updates = UpdateChecker()

    /// ⌃Space Spotlight-style prompt panel (tray toggle under Voice).
    /// Registration follows the toggle live; also applied once at launch
    /// (didSet doesn't fire for the init assignment).
    @Published var quickLauncherEnabled: Bool {
        didSet {
            UserDefaults.standard.set(quickLauncherEnabled, forKey: "quickLauncherEnabled")
            quickLauncher.setEnabled(quickLauncherEnabled)
        }
    }
    /// Bumped by the quick launcher's "Open in chat" action. The menu-bar
    /// label observes it (the label is always installed, so this works with no
    /// window open — same bridge as the task-notification deep-link) and opens
    /// the chat window. An Int tick so every bump fires onChange, no reset dance.
    @Published var quickLauncherChatOpenTick = 0

    /// Owns the global hotkey + floating panel. App-level like the voice
    /// controller so it works with every window closed.
    lazy var quickLauncher = QuickLauncherController(appState: self)

    /// The single generation engine shared by the text chat window and the voice
    /// assistant — one code path, no behavioural drift. App-level so generation
    /// is independent of any window.
    lazy var chatEngine = ChatTurnEngine(appState: self)

    /// Telegram bot bridge — message the local model from your phone. Lazily
    /// created; runs only while `serverOptions.telegram` is enabled with a token.
    lazy var telegramBridge = TelegramBridge(appState: self)

    /// Runs unattended scheduled/on-demand agent tasks (the "claw" spine). Lazily
    /// created so it only spins up the first time the Tasks window is opened.
    lazy var taskScheduler = TaskScheduler(appState: self)

    /// The persistent, window-independent voice assistant. Owned here (not in a
    /// view) so it survives chat-window open/close and runs from the menu-bar
    /// tray. `bind` wires it to `chatEngine` and the active session once.
    lazy var voice: VoiceModeController = {
        let controller = VoiceModeController(server: server)
        controller.bind(appState: self)
        return controller
    }()

    private let historyPath: String = {
        let dir = NSString(string: "~/.mlx-serve").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return (dir as NSString).appendingPathComponent("chat-history.json")
    }()

    init() {
        self.autoStartServer = UserDefaults.standard.bool(forKey: "autoStartServer")
        self.hotSwitchEnabled = UserDefaults.standard.bool(forKey: "hotSwitchEnabled")
        self.selectedModelPath = UserDefaults.standard.string(forKey: "selectedModelPath") ?? ""
        // Load ServerOptions, then migrate legacy single-key defaults
        // (`maxTokens`, `contextSize`) into it on first run if the dedicated
        // ServerOptions blob hasn't been written yet. After that the bridges
        // above (var maxTokens / var contextSize) keep them in sync.
        var opts = ServerOptions.load()
        if UserDefaults.standard.object(forKey: "serverOptions") == nil {
            let storedMax = UserDefaults.standard.integer(forKey: "maxTokens")
            if storedMax > 0 { opts.defaultMaxTokens = storedMax }
            let storedCtx = UserDefaults.standard.integer(forKey: "contextSize")
            if storedCtx > 0 { opts.ctxSize = storedCtx }
            opts.save()
        }
        self.serverOptions = opts
        self.mcpMode = UserDefaults.standard.bool(forKey: "mcpMode")
        self.quickLauncherEnabled = UserDefaults.standard.bool(forKey: "quickLauncherEnabled")
        server.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &cancellables)

        refreshModels()
        loadChatHistory()
        // Start background task scheduling (catch-up + timer arming). Notifications
        // route back here to resume paused runs / deep-link into the Tasks window.
        TaskNotifier.shared.appState = self
        taskScheduler.start()
        if ProcessInfo.processInfo.environment["TESTING_MODE"] != nil {
            testServer.start(appState: self)
        }
        AgentEngine.cleanupOverflowFiles()

        // Start the Telegram bridge if the user left it enabled (didSet doesn't
        // fire for the initial serverOptions assignment in init).
        telegramBridge.reconcile()

        // Same for the agent sandbox: apply the persisted setting once at launch.
        AgentSandbox.shared.configure(enabled: serverOptions.sandbox.enabled,
                                      baseImage: serverOptions.sandbox.baseImage,
                                      network: serverOptions.sandbox.network)

        // And the quick launcher's global ⌃Space hotkey.
        if quickLauncherEnabled { quickLauncher.setEnabled(true) }

        // Auto-update: stop the server child before the installer relaunches
        // the app (the old process's willTerminate doesn't stop it), then
        // start the once-a-day releases/latest check.
        updates.willRelaunch = { [weak self] in self?.server.stop() }
        updates.startAutoCheck()

        // Keep the activation policy in sync with open windows: any real
        // window (Chat, media panes, the intro window) makes the app
        // ⌘Tab-selectable; menu-bar-only → back to accessory.
        ActivationPolicyManager.shared.start()

        // Show the welcome window on every launch — it's the app's intro /
        // quick-start screen and hosts the CLI install button.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            Self.showWelcomeWindow { Self._welcomeWindow?.close() }
        }

        // Auto-start server if enabled and a model is available
        if autoStartServer, !selectedModelPath.isEmpty {
            server.start(modelPath: selectedModelPath, options: serverOptions)
        }

        // Fallback health detection — runs detached to avoid blocking MainActor
        if autoStartServer {
            let checkPort = server.port
            let mgr = server
            Task.detached {
                let api = APIClient()
                for _ in 0..<120 {
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                    if let ok = try? await api.checkHealth(port: checkPort), ok {
                        await mgr.forceRunning()
                        return
                    }
                }
            }
        }
    }

    func refreshModels() {
        localModels = downloads.discoverLocalModels()
        // Auto-select a base model if none selected or the current selection is
        // invalid. Drafters and media / non-chat models never get auto-picked —
        // they aren't loadable as the primary chat model (must match the tray
        // picker's filter, or the selection points at a hidden row).
        let baseModels = localModels.filter { $0.isChatPickable }
        if baseModels.first(where: { $0.path == selectedModelPath }) == nil,
           let first = baseModels.first {
            selectedModelPath = first.path
        }
    }

    // MARK: - Chat Session Management

    /// Sessions to show in the chat sidebar. Excludes only the transient
    /// task-run vehicles; Telegram bridge sessions ARE shown — as read-only
    /// mirrors, flagged with a badge in the sidebar. Pure helper so the filter
    /// is unit-testable without standing up an AppState.
    nonisolated static func sidebarSessions(from all: [ChatSession]) -> [ChatSession] {
        all.filter { $0.taskRunId == nil }
    }
    var visibleChatSessions: [ChatSession] { Self.sidebarSessions(from: chatSessions) }

    func newChatSession() -> UUID {
        var session = ChatSession()
        // Seed the new tab's MCP toggle from the global default so a user who
        // generally runs with MCP on keeps it; Think/Agent start off. Each tab
        // then remembers its own choice (ChatSession.useMCP/enableThinking).
        session.useMCP = mcpMode
        chatSessions.insert(session, at: 0)
        activeChatId = session.id
        saveChatHistory()
        return session.id
    }

    func deleteSession(_ id: UUID) {
        // Kill any background processes this session started before dropping it —
        // otherwise they'd survive untracked for the rest of the app's life.
        processRegistry.killSession(id)
        documentIndexes[id]?.cancel()
        documentIndexes.removeValue(forKey: id)
        chatSessions.removeAll { $0.id == id }
        // Stop the in-flight turn if it belonged to this session — otherwise
        // it ghost-runs invisibly, holds the shared engine (every other chat
        // reports "answering another chat" with no Stop control), and no
        // server restart can clear it. See ChatTurnEngine.turnOrphaned.
        chatEngine.stopIfOrphaned()
        if activeChatId == id {
            activeChatId = chatSessions.first?.id
        }
        saveChatHistory()
    }

    var activeSession: ChatSession? {
        get { chatSessions.first { $0.id == activeChatId } }
        set {
            if let newValue, let idx = chatSessions.firstIndex(where: { $0.id == newValue.id }) {
                chatSessions[idx] = newValue
            }
        }
    }

    func appendMessage(to sessionId: UUID, message: ChatMessage) {
        guard let idx = chatSessions.firstIndex(where: { $0.id == sessionId }) else { return }
        chatSessions[idx].messages.append(message)
        chatSessions[idx].updatedAt = Date()
        // Auto-title from first user message
        if chatSessions[idx].title == "New Chat",
           message.role == .user,
           !message.content.isEmpty {
            let title = String(message.content.prefix(40))
            chatSessions[idx].title = title + (message.content.count > 40 ? "..." : "")
        }
    }

    func updateLastMessage(in sessionId: UUID, content: String? = nil, reasoning: String? = nil, streaming: Bool? = nil, usage: TokenUsage? = nil) {
        guard let sIdx = chatSessions.firstIndex(where: { $0.id == sessionId }),
              !chatSessions[sIdx].messages.isEmpty else { return }
        let mIdx = chatSessions[sIdx].messages.count - 1
        if let content { chatSessions[sIdx].messages[mIdx].content += content }
        if let usage {
            chatSessions[sIdx].messages[mIdx].promptTokens = usage.promptTokens
            chatSessions[sIdx].messages[mIdx].completionTokens = usage.completionTokens
            chatSessions[sIdx].messages[mIdx].tokensPerSecond = usage.tokensPerSecond
        }
        if let reasoning { chatSessions[sIdx].messages[mIdx].reasoningContent = (chatSessions[sIdx].messages[mIdx].reasoningContent ?? "") + reasoning }
        if let streaming { chatSessions[sIdx].messages[mIdx].isStreaming = streaming }
    }

    // MARK: - Agent Helpers

    func updatePlanStatus(in sessionId: UUID, planId: UUID, status: PlanStatus) {
        guard let sIdx = chatSessions.firstIndex(where: { $0.id == sessionId }) else { return }
        for mIdx in chatSessions[sIdx].messages.indices {
            if chatSessions[sIdx].messages[mIdx].agentPlan?.id == planId {
                chatSessions[sIdx].messages[mIdx].agentPlan?.status = status
                break
            }
        }
    }

    func appendToolResults(to sessionId: UUID, results: [StepResult]) {
        guard let sIdx = chatSessions.firstIndex(where: { $0.id == sessionId }) else { return }
        for mIdx in chatSessions[sIdx].messages.indices.reversed() {
            if chatSessions[sIdx].messages[mIdx].role == .assistant {
                chatSessions[sIdx].messages[mIdx].toolResults = results
                break
            }
        }
    }

    // MARK: - Persistence

    func saveChatHistory() {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = .prettyPrinted
        // Transient task-run sessions live in `chatSessions` only while their run is
        // in flight (the agent loop reads/appends through AppState). They are never
        // persisted here — their transcript is saved out of line by TaskScheduler.
        let persisted = chatSessions.filter { $0.taskRunId == nil && !$0.isExternalBridge }
        guard let data = try? encoder.encode(persisted) else { return }
        try? data.write(to: URL(fileURLWithPath: historyPath))
    }

    private func loadChatHistory() {
        guard FileManager.default.fileExists(atPath: historyPath),
              let data = try? Data(contentsOf: URL(fileURLWithPath: historyPath)) else { return }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        chatSessions = (try? decoder.decode([ChatSession].self, from: data)) ?? []
        activeChatId = chatSessions.first?.id
    }

    // MARK: - Welcome Window

    private static func showWelcomeWindow(onDismiss: @escaping () -> Void) {
        let view = WelcomeView(onDismiss: onDismiss)
        let hostingView = NSHostingView(rootView: view)

        // Let SwiftUI compute the intrinsic size
        let fittingSize = hostingView.fittingSize
        hostingView.frame = NSRect(origin: .zero, size: fittingSize)

        let window = NSWindow(
            contentRect: NSRect(origin: .zero, size: fittingSize),
            styleMask: [.titled, .closable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.contentView = hostingView
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.isMovableByWindowBackground = true
        window.center()
        window.isReleasedWhenClosed = false
        window.level = .floating
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        _welcomeWindow = window
    }

    private static var _welcomeWindow: NSWindow?
}
