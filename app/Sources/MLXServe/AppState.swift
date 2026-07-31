import Combine
import Foundation
import SwiftUI

/// Selection repair for `refreshModels`: keep a still-pickable selection,
/// swap to the first pickable model otherwise, and CLEAR a dangling one when
/// nothing pickable remains — a deleted models directory otherwise leaves the
/// dead path persisted, and every start site (autostart, the LAN share boot,
/// the tray Start button) launches `--model <gone>` into an instant
/// FileNotFound.
func reconciledModelSelection(current: String, pickablePaths: [String]) -> String {
    if pickablePaths.contains(current) { return current }
    return pickablePaths.first ?? ""
}

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
            // Drafter pairing: a drafter is paired to a specific Gemma 4 size,
            // and carrying the wrong one over crashes the server with
            // `DrafterTargetMismatch` — so every model change re-decides from
            // scratch (`DrafterPairing.decide`). It pairs a dense Gemma 4 with
            // the drafter that came down with it whether or not one was on
            // before: the checkpoint is a dependency of the model now, not
            // something the user went shopping for. `drafterOptOut` is what
            // makes an explicit off stick.
            syncDrafterPairing()
            // Plan 05 Phase G — when hot-switch is enabled AND the server is
            // already running, ask the server to load the new model in-place
            // instead of restarting. Falls back to restart on failure (404
            // because the new path isn't in --model-dir, 503 if out of
            // memory, etc.). Restart path remains the default for clients
            // that don't opt in.
            if (server.status == .running || server.status == .starting) {
                if hotSwitchEnabled, server.status == .running {
                    let id = (selectedModelPath as NSString).lastPathComponent
                    // The decision `syncDrafterPairing()` just made, not a
                    // second read of the disk: a hot-switch that ignores the
                    // user's off switch loads a drafter the restart path
                    // wouldn't, and only one of the two would be reproducible.
                    let drafterPath: String? = serverOptions.drafterPath.isEmpty ? nil : serverOptions.drafterPath
                    let mgr = server
                    // Tracked so `useModelAndAwaitReady` can await this exact
                    // switch — hot-switch never moves `server.status` off
                    // `.running` (the process itself never restarts), so
                    // polling status alone can't tell "old model still
                    // resident" from "new model resident".
                    pendingModelLoadTask = Task { @MainActor in
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
                    pendingModelLoadTask = nil
                    server.stop()
                    server.start(modelPath: selectedModelPath, options: serverOptions)
                }
            } else {
                pendingModelLoadTask = nil
            }
        }
    }
    /// Set only while a hot-switch triggered by `selectedModelPath`'s `didSet`
    /// is in flight — see `useModelAndAwaitReady`.
    private var pendingModelLoadTask: Task<Void, Never>?
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
    /// Set by the tray's "pi/hermes in Sandbox" shortcut; the Sandbox window
    /// consumes it (focus a running session of that agent, else start one)
    /// and clears it. Fresh `id` per click so repeat clicks re-fire onChange.
    struct SandboxAgentLaunch: Equatable {
        let id = UUID()
        let agentId: String
    }
    @Published var pendingSandboxAgentLaunch: SandboxAgentLaunch?
    @Published var agentMemory = AgentMemory()
    /// Saved personas (`~/.mlx-serve/agents/index.json`) plus the read-only
    /// starters. Views observe it directly (`.environmentObject(appState.agents)`),
    /// the same way they observe `server` — see `AppStateAgents` for what picking
    /// one does.
    let agents = AgentStore()
    /// The agent used where there's no per-conversation pick: the voice tray and
    /// the Quick Launcher. nil = none (app defaults), which is the default.
    @Published var defaultAgentId: UUID? {
        didSet {
            UserDefaults.standard.set(defaultAgentId?.uuidString, forKey: "defaultAgentId")
            // The tray/launcher speak with this agent's voice from the next
            // sentence; a chat tab's own pick overrides it when a turn runs there.
            Task { await applyAgentSelection(defaultAgentId, previousWorkingDirectory: nil) }
        }
    }
    /// The agent the Agents window should open ON, set by whoever deep-links
    /// into it (`openAgentSettings`) and consumed by the window. Not persisted —
    /// it's a one-shot request, not a setting.
    @Published var pendingAgentSelection: UUID?
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
                                          network: serverOptions.sandbox.network)
            // Turning LAN sharing/discovery ON means "the server runs" — boot
            // it (headless if no model is selected) on the transition only, so
            // unrelated settings edits never start anything.
            let lanOn = serverOptions.lanShareEnabled || serverOptions.lanDiscoverEnabled
            let lanWasOn = oldValue.lanShareEnabled || oldValue.lanDiscoverEnabled
            if lanOn && !lanWasOn { ensureServerForLan() }
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
    /// "Open the chat window" for callers that can't reach SwiftUI's
    /// `openWindow`: the quick launcher's "Open in chat" (a non-activating
    /// NSPanel) and the welcome window (a bare `NSHostingView` outside the
    /// Scene graph). The menu-bar label observes it — the label is always
    /// installed, so this works with no window open, same bridge as the
    /// task-notification deep-link. An Int tick so every bump fires onChange,
    /// no reset dance. ONE bridge for both callers rather than a second
    /// near-identical tick: they want the same window.
    @Published var pendingChatOpenTick = 0

    /// Bumped by the welcome window's "Browse Models" nudge (shown when no
    /// chat model is downloaded yet) — the welcome window is a bare
    /// `NSHostingView` outside the SwiftUI Scene graph, so it can't reach
    /// `openWindow` itself. Same always-present-menu-bar-label bridge as
    /// `pendingChatOpenTick`.
    @Published var pendingModelBrowserOpenTick = 0

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
        // Defaults to ON when the key is absent — `UserDefaults.bool` would
        // read a never-set key as false, which is why a fresh install used to
        // download a model and then sit there with the server stopped. The
        // launch gate below is `autoStartServer && !selectedModelPath.isEmpty`,
        // so this stays a no-op until a model exists; the first download's
        // completion hook is what actually starts it. No migration: existing
        // users who never touched the toggle get it turned on, which is the
        // intent.
        self.autoStartServer = UserDefaults.standard.object(forKey: "autoStartServer") as? Bool ?? true
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
        self.defaultAgentId = UserDefaults.standard.string(forKey: "defaultAgentId")
            .flatMap(UUID.init(uuidString:))
        self.quickLauncherEnabled = UserDefaults.standard.bool(forKey: "quickLauncherEnabled")
        server.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &cancellables)
        // Same forwarding for the agent store: the chat chip and the tray picker
        // observe AppState, not the store, so a newly created or renamed agent
        // has to reach them without waiting for an unrelated publish.
        agents.objectWillChange
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
                                      network: serverOptions.sandbox.network)

        // And the quick launcher's global ⌃Space hotkey.
        if quickLauncherEnabled { quickLauncher.setEnabled(true) }

        // The app-level agent's voice, applied once at launch (didSet doesn't
        // fire for the init assignment above). Everything else it owns is
        // resolved per turn.
        ActiveAgentVoice.set(agents.agent(id: defaultAgentId)?.resolvedVoice)

        // Auto-update: stop the server child before the installer relaunches
        // the app (the old process's willTerminate doesn't stop it), then
        // start the once-a-day releases/latest check.
        updates.willRelaunch = { [weak self] in self?.server.stop() }
        updates.startAutoCheck()

        // Keep the activation policy in sync with open windows: any real
        // window (Chat, media panes, the intro window) makes the app
        // ⌘Tab-selectable; menu-bar-only → back to accessory.
        ActivationPolicyManager.shared.start()

        // The welcome window is the app's intro / quick-start screen and hosts
        // the CLI install button, so it shows on every launch — unless the user
        // ticked "Don't show this again", in which case the launch goes
        // straight to Chat (`LaunchDecision`). Either way the user ends up in
        // front of a composer rather than looking at an empty desktop.
        let suppressed = UserDefaults.standard.bool(forKey: LaunchDecision.suppressDefaultsKey)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            guard let self else { return }
            let hasChat = self.localModels.contains(where: \.isChatPickable)
            switch LaunchDecision.resolve(welcomeSuppressed: suppressed, hasChatModels: hasChat) {
            case .openChat:
                self.pendingChatOpenTick += 1
            case .showWelcome:
                Self.showWelcomeWindow(
                    appState: self,
                    hasChatModels: hasChat,
                    onDismiss: { Self._welcomeWindow?.close() },
                    onOpenModelBrowser: { self.pendingModelBrowserOpenTick += 1 },
                    onOpenChat: { self.pendingChatOpenTick += 1 }
                )
            }
        }

        // Auto-start server if enabled and a model is available
        if autoStartServer, !selectedModelPath.isEmpty {
            server.start(modelPath: selectedModelPath, options: serverOptions)
        }
        // LAN sharing/discovery lives in the server process — with either
        // enabled the server should be up (headless when nothing was
        // auto-started) so this Mac shares and sees network models.
        if serverOptions.lanShareEnabled || serverOptions.lanDiscoverEnabled {
            ensureServerForLan()
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

    /// Chat on a LAN model: record the remote id and make sure the (proxying)
    /// local server is up — headless when no local model is selected. The
    /// remote id rides every chat request via `server.chatModelId`.
    func selectLanModel(_ id: String) {
        server.lanChatModelId = id
        ensureServerForLan()
    }

    /// Start the server for LAN duty if it isn't running: with the selected
    /// local model when there is one (it keeps serving chat AND the LAN),
    /// else headless over the models root.
    func ensureServerForLan() {
        guard server.status != .running, server.status != .starting else { return }
        if !selectedModelPath.isEmpty {
            server.start(modelPath: selectedModelPath, options: serverOptions)
        } else {
            let root = NSString(string: "~/.mlx-serve/models").expandingTildeInPath
            server.startHeadless(modelsDir: root, options: serverOptions)
        }
    }

    /// Freshen the network-model list for a picker that is about to show it.
    /// No-op when discovery is off; boots the server (headless) when needed.
    func refreshLanModels() async {
        guard serverOptions.lanDiscoverEnabled else { return }
        ensureServerForLan()
        try? await server.waitUntilRunning(timeout: 60)
        await server.refreshModels()
    }

    func refreshModels() {
        localModels = downloads.discoverLocalModels()
        // Auto-select a base model if none selected or the current selection is
        // invalid. Drafters and media / non-chat models never get auto-picked —
        // they aren't loadable as the primary chat model (must match the tray
        // picker's filter, or the selection points at a hidden row).
        let baseModels = localModels.filter { $0.isChatPickable }
        let repaired = reconciledModelSelection(current: selectedModelPath,
                                                pickablePaths: baseModels.map(\.path))
        if repaired != selectedModelPath { selectedModelPath = repaired }
        adoptNewlyAvailableDrafter()
    }

    // MARK: - Drafter pairing

    /// Re-decide the drafter for the selected model. Called on every model
    /// change — it both pairs and UNPAIRS, because a drafter carried onto the
    /// wrong Gemma 4 size is `DrafterTargetMismatch` at server start.
    private func syncDrafterPairing() {
        let paired = DrafterPairing.decide(
            modelPath: selectedModelPath,
            optedOut: serverOptions.drafterOptOut,
            onDiskPath: downloads.recommendedDrafterFromPath(selectedModelPath)?.url.path)
        if serverOptions.drafterPath != paired { serverOptions.drafterPath = paired }
    }

    /// The model list changed (a download landed): fill in a pairing that
    /// wasn't possible a moment ago — downloading a Gemma 4 fetches its drafter
    /// too, and it finishes after the model is already selected.
    ///
    /// Only ever ADDS. Clearing belongs to the model switch: this runs on every
    /// refresh (1 Hz while a download is in flight), and a user who deliberately
    /// switched a drafter on where we don't recommend one — the MoE caution in
    /// Settings — must not have a background rescan take it away.
    private func adoptNewlyAvailableDrafter() {
        guard serverOptions.drafterPath.isEmpty, !serverOptions.drafterOptOut else { return }
        let paired = DrafterPairing.decide(
            modelPath: selectedModelPath,
            optedOut: false,
            onDiskPath: downloads.recommendedDrafterFromPath(selectedModelPath)?.url.path)
        if !paired.isEmpty { serverOptions.drafterPath = paired }
    }

    /// What `useModelAndAwaitReady` must do once `selectedModelPath`'s
    /// `didSet` has run, given the server's status BEFORE that assignment.
    /// Pure so the branch is unit-tested without a real `ServerManager`.
    enum UseModelStartAction: Equatable {
        /// `didSet` only reacts to `.running`/`.starting` — nothing was
        /// kicked off, so the caller must start the server itself.
        case startExplicitly
        /// `didSet` already kicked off a hot-switch or restart as a
        /// fire-and-forget task — the caller just waits for it.
        case awaitPendingSwitch
    }

    nonisolated static func useModelStartAction(forStatusBefore status: ServerStatus) -> UseModelStartAction {
        switch status {
        case .stopped, .error: return .startExplicitly
        case .running, .starting: return .awaitPendingSwitch
        }
    }

    /// Backs the Model Browser's "Use" button: select `path`, make the server
    /// actually serve it (starting it if stopped, hot-switching/restarting if
    /// already running — same logic `selectedModelPath`'s `didSet` always
    /// ran, just now awaitable), and return once it's ready. The caller opens
    /// the Chat window on `true` — a click should end in a ready-to-chat
    /// server, not just a selection the user then has to start by hand.
    /// Returns `false` on failure/timeout (mirrors the existing tray/gen-pane
    /// "start and wait" error handling — the caller just skips opening chat;
    /// `server.status` already surfaces the failure elsewhere).
    @MainActor
    @discardableResult
    func useModelAndAwaitReady(atPath path: String) async -> Bool {
        let statusBefore = server.status
        selectedModelPath = path
        switch Self.useModelStartAction(forStatusBefore: statusBefore) {
        case .startExplicitly:
            server.start(modelPath: path, options: serverOptions)
        case .awaitPendingSwitch:
            // Wait before checking `waitUntilRunning` below (a no-op if the
            // hot-switch left status at `.running` the whole time).
            await pendingModelLoadTask?.value
        }
        do {
            try await server.waitUntilRunning(timeout: 240)
            return true
        } catch {
            return false
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

    func newChatSession(agentId: UUID? = nil) -> UUID {
        var session = ChatSession()
        // Seed the new tab's MCP toggle from the global default so a user who
        // generally runs with MCP on keeps it; Think/Tools start off. Each tab
        // then remembers its own choice (ChatSession.useMCP/enableThinking).
        session.useMCP = mcpMode
        session.agentId = agentId
        chatSessions.insert(session, at: 0)
        activeChatId = session.id
        saveChatHistory()
        return session.id
    }

    /// Which existing thread a turn for `agentId` belongs in, or nil to start a
    /// fresh one. Every agent keeps its OWN conversation, so speaking to Chef
    /// continues Chef's thread instead of talking into whatever tab was open (and
    /// instead of quietly rebranding that tab as Chef).
    ///
    /// An ACTIVE thread of the same agent wins over a more recently touched one —
    /// the user opened that one deliberately. Task-run and Telegram-bridge
    /// sessions are never adopted: they're hidden/transient vehicles (and task
    /// runs now carry an `agentId` too), so a turn landing in one would corrupt a
    /// run or write into a read-only mirror. Pure → `AgentSessionThreadTests`.
    nonisolated static func sessionForAgent(_ agentId: UUID?,
                                           sessions: [ChatSession],
                                           activeId: UUID?) -> UUID? {
        func isConversation(_ s: ChatSession) -> Bool {
            s.taskRunId == nil && !s.isExternalBridge
        }
        let active = sessions.first { $0.id == activeId }.flatMap { isConversation($0) ? $0 : nil }
        guard let agentId else {
            // No agent: today's behavior — keep talking into the active tab.
            return active?.id
        }
        if active?.agentId == agentId { return active?.id }
        return sessions
            .filter { $0.agentId == agentId && isConversation($0) }
            .max { $0.updatedAt < $1.updatedAt }?
            .id
    }

    /// The thread a turn for `agentId` runs in, creating one when the agent
    /// doesn't have a conversation yet. Also makes it the ACTIVE chat: the voice
    /// controller speaks the active session's trailing assistant message, so a
    /// turn running anywhere else would never be read aloud.
    @discardableResult
    func sessionForAgent(_ agentId: UUID?) -> UUID {
        if let existing = Self.sessionForAgent(agentId, sessions: chatSessions, activeId: activeChatId) {
            if activeChatId != existing { activeChatId = existing }
            return existing
        }
        return newChatSession(agentId: agentId)
    }

    func deleteSession(_ id: UUID) {
        // Kill any background processes this session started before dropping it —
        // otherwise they'd survive untracked for the rest of the app's life.
        processRegistry.killSession(id)
        documentIndexes[id]?.cancel()
        documentIndexes.removeValue(forKey: id)
        // Drop the session's security-scoped bookmarks with it — a deleted chat
        // must not keep durable access to the folders it was granted.
        SecurityScopedBookmark.clear(name: SecurityScopedBookmark.workingFolderName(id))
        SecurityScopedBookmark.clear(name: SecurityScopedBookmark.attachedFolderName(id))
        chatSessions.removeAll { $0.id == id }
        // Stop the in-flight turn if it belonged to this session — otherwise
        // it ghost-runs invisibly with no Stop control anywhere, and no server
        // restart can clear it. The sweep is per turn: only the deleted chat's
        // turn stops. See ChatTurnEngine.stopIfOrphaned / TurnLedger.orphaned.
        chatEngine.stopIfOrphaned()
        if activeChatId == id {
            activeChatId = chatSessions.first?.id
        }
        saveChatHistory()
    }

    /// Apply a new DEFAULT agent workspace picked in Settings: persist the
    /// setting, keep a security-scoped bookmark so the App Sandbox build can
    /// reach the folder after relaunch, retarget sessions still on the old
    /// default (the chat toolbar folder stays in sync with Settings), and
    /// remount the sandbox. An EXPLICIT pick remounts even under live CLI
    /// sessions (`restartPinnedSessions` — the Sandbox window restarts them
    /// in the new share); without it, a live terminal quietly kept the old
    /// folder mounted until an app restart.
    func setDefaultAgentWorkspace(_ path: String) {
        let old = ChatSession.defaultWorkingDirectory
        ChatSession.setDefaultWorkingDirectory(path)
        SecurityScopedBookmark.store(URL(fileURLWithPath: path),
                                     name: SecurityScopedBookmark.defaultWorkspaceName)
        SecurityScopedBookmark.startAccessOnce(name: SecurityScopedBookmark.defaultWorkspaceName)
        agentMemory.recordDirectory(path)
        chatSessions = ChatSession.retargeted(chatSessions, from: old, to: path)
        saveChatHistory()
        AgentSandbox.shared.noteWorkspaceChanged(path, restartPinnedSessions: true)
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

    /// Drop one message from a conversation.
    ///
    /// It leaves the history the model sees, not just the view — pruning a bad
    /// turn so the next request isn't built on it is the whole point. Any
    /// HIDDEN tool-result messages belonging to the same assistant turn go with
    /// it: a tool result whose call is gone is an orphan the model can only be
    /// confused by, and it is invisible, so nothing would ever clean it up.
    func deleteMessage(in sessionId: UUID, messageId: UUID) {
        guard let sIdx = chatSessions.firstIndex(where: { $0.id == sessionId }),
              let mIdx = chatSessions[sIdx].messages.firstIndex(where: { $0.id == messageId })
        else { return }
        var removed = IndexSet(integer: mIdx)
        var i = mIdx + 1
        while i < chatSessions[sIdx].messages.count,
              chatSessions[sIdx].messages[i].toolCallId != nil {
            removed.insert(i)
            i += 1
        }
        chatSessions[sIdx].messages.remove(atOffsets: removed)
        chatSessions[sIdx].updatedAt = Date()
        saveChatHistory()
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

    private static func showWelcomeWindow(
        appState: AppState,
        hasChatModels: Bool,
        onDismiss: @escaping () -> Void,
        onOpenModelBrowser: @escaping () -> Void,
        onOpenChat: @escaping () -> Void
    ) {
        // This is an NSHostingView, so it inherits NO environment — the starter
        // card's `@EnvironmentObject`s have to be handed to it explicitly or
        // SwiftUI traps the first time the card renders.
        let view = WelcomeView(onDismiss: onDismiss,
                               hasChatModels: hasChatModels,
                               onOpenModelBrowser: onOpenModelBrowser,
                               onOpenChat: onOpenChat)
            .environmentObject(appState)
            .environmentObject(appState.downloads)
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
