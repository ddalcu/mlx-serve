import AppKit
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
            switch Self.modelSwitchAction(forStatus: server.status, path: selectedModelPath) {
            case .hotSwitch(let id):
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
                // The pill names this model and spins while the load runs: a
                // hot switch never moves `server.status` off `.running` and
                // `chatModelInfo` keeps reporting the OLD model until the new
                // one is resident, so without this a big checkpoint loads for a
                // minute under the previous model's name and a green dot.
                // Generation-tagged: a replaced task keeps running (nothing
                // cancels it), and an unconditional clear in ITS defer would
                // wipe the flag while a NEWER switch is still loading — the
                // spinner vanishes and the pill goes back to the old model,
                // the exact symptom this flag exists to fix. A path compare
                // isn't enough (switch A→B→A: the stale A task's path matches
                // the new flag), so only the LATEST switch's task clears it.
                modelSwitchGeneration += 1
                let generation = modelSwitchGeneration
                loadingModelPath = selectedModelPath
                pendingModelLoadTask = Task { @MainActor in
                    defer { if self.modelSwitchGeneration == generation { self.loadingModelPath = nil } }
                    do {
                        _ = try await mgr.loadModel(id: id, drafterPath: drafterPath, setDefault: true)
                    } catch {
                        // Register-by-path failed (unsupported arch, partial
                        // download) or the load 503'd (memory) — a full
                        // restart with only the new model is the net.
                        print("[AppState] hot-switch failed (\(error)) — falling back to restart")
                        mgr.stop()
                        mgr.start(modelPath: self.selectedModelPath, options: self.serverOptions)
                    }
                }
            case .restart:
                // A restart moves `server.status`, so `ChatServerStartControl`
                // already reports it — this flag is only for the switch that
                // leaves the status alone.
                pendingModelLoadTask = nil
                loadingModelPath = nil
                server.stop()
                server.start(modelPath: selectedModelPath, options: serverOptions)
            case .leaveStopped:
                pendingModelLoadTask = nil
                loadingModelPath = nil
            }
        }
    }

    /// How `selectedModelPath`'s `didSet` makes the server serve the new pick.
    /// A RUNNING server is hot-switched in place via /v1/load-model — no
    /// restart. The id is the model's ABSOLUTE PATH, never the dir basename:
    /// registry ids are two-level `org/name`, so a basename 404s, while
    /// register-by-path resolves either shape (and models outside every
    /// `--model-dir`). `setDefault` rides along so the server re-points the
    /// "mlx-serve" alias and /v1/models' default-first sort — without it a
    /// hot-load leaves every aliased request on the OLD model.
    /// (The old `hotSwitchEnabled` gate shipped default-off with no UI, so
    /// every picker change restarted the server, for everyone, forever.)
    enum ModelSwitchAction: Equatable {
        case hotSwitch(id: String)
        /// Mid-boot: the process is still loading the OLD pick — restart
        /// with the new one.
        case restart
        /// Stopped/error: nothing to switch; explicit starts
        /// (`useModelAndAwaitReady`, the launch gate) own that.
        case leaveStopped
    }

    nonisolated static func modelSwitchAction(forStatus status: ServerStatus, path: String) -> ModelSwitchAction {
        switch status {
        case .running: return .hotSwitch(id: path)
        case .starting: return .restart
        case .stopped, .error: return .leaveStopped
        }
    }
    /// Set only while a hot-switch triggered by `selectedModelPath`'s `didSet`
    /// is in flight — see `useModelAndAwaitReady`.
    private var pendingModelLoadTask: Task<Void, Never>?
    /// The model path a hot-switch is loading right now, or nil. Published so
    /// the composer's picker can name the model it is switching TO and spin
    /// while it arrives (`ChatModelSelection.pillState`): the server answers on
    /// the previous model for the whole load, so nothing else says a switch is
    /// in flight. A RESTART is not tracked here — that one moves the server
    /// status, which the Start control already reports.
    @Published var loadingModelPath: String?
    /// Bumped per hot-switch; each switch task captures its value so only the
    /// LATEST switch's completion clears `loadingModelPath` (see the didSet).
    private var modelSwitchGeneration = 0
    @Published var chatSessions: [ChatSession] = []
    @Published var activeChatId: UUID?
    /// Sidebar selection (multi-select). Bind the sidebar List to this set so
    /// macOS-style multi-selection (Cmd/Shift click, drag) works naturally.
    @Published var sidebarSelection: Set<UUID> = []
    /// Chats waiting on the delete confirmation. It lives out here rather than
    /// in the sidebar because the File menu's ⌘⌫ has to reach it too, and two
    /// confirmations for one decision is how they start disagreeing; the
    /// sidebar owns the single dialog that presents it.
    @Published var pendingChatDeletion: Set<UUID>?
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
    /// NSPanel) and the launch path. The menu-bar label observes it — the label
    /// is always installed, so this works with no window open, same bridge as
    /// the task-notification deep-link. An Int tick so every bump fires
    /// onChange, no reset dance. ONE bridge for both callers rather than a
    /// second near-identical tick: they want the same window.
    @Published var pendingChatOpenTick = 0

    /// Is the welcome screen up? It is a SHEET on the chat window (see
    /// `LaunchDecision`), so this is a flag the scene binds rather than a
    /// window somebody has to remember to close — which is what it was, with
    /// its own `NSHostingView` inheriting no environment and a `.floating`
    /// level that could leave it over an empty desktop.
    @Published var showWelcome = false
    /// What the welcome screen should say about this Mac's library, sampled at
    /// launch. Held beside the flag because the sheet's content is built by the
    /// scene, which has no business re-deriving it.
    @Published var welcomeHasChatModels = false

    /// What the chat window's detail column is showing: the transcript, or the
    /// model browser (`ChatWorkspace`). App-level rather than view-local
    /// because the surfaces that ask for the browser live OUTSIDE that window —
    /// the tray popover, the welcome screen, the Tools menu.
    @Published var chatWorkspace: ChatWorkspace = .conversation
    /// The task showing in the Tasks pane's detail column. App-level because the
    /// list and the detail are SEPARATE columns of the chat window's own
    /// `NavigationSplitView` — neither can own the other's state.
    @Published var selectedTaskId: UUID?

    /// Is the ⌘L model switcher up? A SHEET on the chat window, like the
    /// welcome screen — so the flag lives here rather than in the view, because
    /// the thing that opens it is a menu command with no view to talk to.
    @Published var modelPalettePresented = false

    /// Open the model switcher — the ONE way in, same shape as `showModels()`:
    /// it must both raise the picker AND bring the chat window forward, or ⌘L
    /// from the tray opens a sheet on a window nobody is looking at.
    ///
    /// Refused while the welcome sheet is up: that screen is the first-run
    /// model picker, and a second sheet over it is the one-sheet rule.
    func showModelPalette() {
        guard !showWelcome else { return }
        modelPalettePresented = true
        pendingChatOpenTick += 1
    }

    /// Apply a `ChatModelSelection` tag — what picking a model MEANS, in one
    /// place. The tray, the composer's pill and the ⌘L palette all call this:
    /// each had its own copy of "clear the LAN id, then set the path", which is
    /// the per-surface-copy class the tag semantics were centralised to avoid.
    func applyChatModelPick(_ tag: String) {
        switch ChatModelSelection.action(for: tag) {
        case .selectLan(let id):
            selectLanModel(id)
        case .selectLocal(let path):
            // Picking a local model always clears the LAN choice, or the chat
            // keeps being answered by the other Mac.
            server.lanChatModelId = nil
            selectedModelPath = path
        }
    }

    /// Show the model browser — the ONE way in.
    func showModels(_ section: ModelBrowserSection = .recommended) {
        chatWorkspace = .models(section)
        pendingChatOpenTick += 1
    }

    /// Show the Tasks pane — the one way in, same shape as `showModels()`.
    func showTasks() {
        chatWorkspace = .tasks
        pendingChatOpenTick += 1
    }

    /// Show the Agents pane. The standalone Agents window still exists for the
    /// composer's "Edit Agent…" deep link; this is the sidebar's route.
    func showAgents() {
        chatWorkspace = .agents
        pendingChatOpenTick += 1
    }

    /// Show Settings in the content area — the ONE way in: the sidebar's row
    /// and the menu bar's ⌘, both land here (the Settings Window scene is
    /// gone; a second surface showing the same form is how the two drift).
    func showSettings() {
        chatWorkspace = .settings
        pendingChatOpenTick += 1
    }

    /// Switch the browser's section. Deliberately a NO-OP outside the models
    /// pane: this is the section bar's setter, and a section bar that could
    /// also ENTER the pane would be a second door in — one that doesn't bring
    /// the window forward. Entering is `showModels()`, and only `showModels()`.
    func selectModelSection(_ section: ModelBrowserSection) {
        guard chatWorkspace.isModels else { return }
        chatWorkspace = .models(section)
    }

    /// Show a media generator — the ONE way in, same shape as `showModels()`.
    /// The four used to be four `Window` scenes; they are pages of this window
    /// now, so a request has to both pick the page and bring the window up.
    func showCreate(_ experiment: GenExperiment = .image) {
        chatWorkspace = .create(experiment)
        pendingChatOpenTick += 1
    }

    /// Switch generator page. A no-op outside create mode, for the same reason
    /// `selectModelSection` is: the page list must not be a second door in.
    func selectCreatePage(_ experiment: GenExperiment) {
        guard chatWorkspace.isCreate else { return }
        chatWorkspace = .create(experiment)
    }

    /// "Send to Chat" on a Create-pane result: open a NEW conversation holding
    /// it, and switch to Chats so the user SEES where it went.
    @discardableResult
    func sendGeneratedMediaToNewChat(path: String, prompt: String,
                                     kind: ChatMediaRef.Kind) -> UUID {
        let sessionId = newChatSession(agentId: defaultAgentId)
        if let idx = chatSessions.firstIndex(where: { $0.id == sessionId }) {
            chatSessions[idx].messages.append(
                GeneratedMediaHandoff.message(path: path, prompt: prompt, kind: kind))
            saveChatHistory()
        }
        showConversation()
        return sessionId
    }

    /// Back to the transcript. Selecting a conversation does this too — the
    /// sidebar's mode switcher is the visible version of it.
    func showConversation() {
        chatWorkspace = .conversation
    }

    /// Show the CONVERSATION and bring the window up — the tray's Chat button
    /// and any other outside-the-window caller. `showConversation()` alone is
    /// for switches made INSIDE the window; from the tray it left the window
    /// parked on whatever pane it was showing, so "Chat" just focused Models
    /// (live report 2026-08-09). Same door shape as `showModels()`.
    func showChat() {
        showConversation()
        pendingChatOpenTick += 1
    }

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
        // A Finder-launched bundle has no shell environment, so HF_HOME /
        // HF_HUB_CACHE / XDG_CACHE_HOME are invisible until we ask the login
        // shell. Off-main (it spawns one), and rescan only if the cache moved.
        Task { [weak self] in
            guard let self else { return }
            if await self.downloads.refreshHuggingFaceRootFromLoginShell() {
                self.refreshModels()
            }
        }
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

        // The welcome screen is the app's intro / quick-start screen and hosts
        // the CLI install button, so it shows on every launch — unless the user
        // ticked "Don't show this again", in which case the launch goes
        // straight to Chat (`LaunchDecision`).
        //
        // The chat window opens in BOTH branches: the welcome is a SHEET on it,
        // and a sheet with no host window is a screen nobody can see. That is
        // also why the user can no longer end up in front of nothing — whatever
        // dismisses the sheet, a composer is what was already behind it.
        let suppressed = UserDefaults.standard.bool(forKey: LaunchDecision.suppressDefaultsKey)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            guard let self else { return }
            let hasChat = self.localModels.contains(where: \.isChatPickable)
            let decision = LaunchDecision.resolve(welcomeSuppressed: suppressed,
                                                  hasChatModels: hasChat)
            if decision.opensChatWindow { self.pendingChatOpenTick += 1 }
            if decision.presentsWelcome {
                self.welcomeHasChatModels = hasChat
                self.showWelcome = true
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
            server.startHeadless(modelsDir: ServerManager.modelsRoot, options: serverOptions)
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
        // An agent's thread is called "New agent" until it has something to be
        // named after — the sidebar lists it under Agents, where "New Chat"
        // would describe the wrong thing.
        var session = ChatSession(title: ChatSessionTitle.placeholder(hasAgent: agentId != nil))
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

    /// What the File menu's Delete Chat (⌘⌫) would act on, and nil when that
    /// is nothing — the menu item reads this to disable itself rather than
    /// offer a command that does nothing when you pick it.
    var chatDeletionTarget: Set<UUID>? {
        SidebarDeleteConfirm.target(selection: sidebarSelection, activeChatId: activeChatId)
    }

    /// The File menu's Delete Chat (⌘⌫). A menu command, NOT a key handler:
    /// the sidebar's conversation column is a ScrollView of plain Buttons, so
    /// nothing there ever becomes first responder and a bare `.onDeleteCommand`
    /// never fired. Routed through the same rule every delete control reads, so
    /// the keyboard cannot become the one path that skips the confirmation.
    ///
    /// It also has to give the keystroke BACK when something is being typed
    /// into: a menu key equivalent is offered the event ahead of the first
    /// responder, so this command otherwise stole ⌘⌫ from the composer, where
    /// it deletes to the start of the line (`ChatDeleteShortcut`). Performing
    /// that deletion here rather than returning early is the load-bearing part
    /// — the menu has already swallowed the event, and nothing else will run.
    func requestChatDeletionFromMenu() {
        let responder = NSApp.keyWindow?.firstResponder
        switch ChatDeleteShortcut.route(editingText: KeyboardFocus.isTextEditor(responder),
                                        selectedChats: sidebarSelection.count) {
        case .deleteToLineStart:
            (responder as? NSTextView)?.deleteToBeginningOfLine(nil)
            return
        case .deleteChats:
            break
        }
        guard let ids = chatDeletionTarget else { return }
        if SidebarDeleteConfirm.required(count: ids.count, keyboard: true) {
            pendingChatDeletion = ids
        } else {
            deleteSessions(ids)
        }
    }

    /// Delete one session. Deliberately the bulk path with a set of one rather
    /// than its own copy of the body: the two DID drift — one fell back to
    /// `chatSessions.first`, the other to `visibleChatSessions.first` — so a
    /// single delete could land you on a hidden session the sidebar has no row
    /// for.
    func deleteSession(_ id: UUID) {
        deleteSessions([id])
    }

    /// Delete a set of sessions — the ONE deletion path, whether the sidebar's
    /// multi-select asked or a single row did. Kill any background processes
    /// those sessions started (otherwise they survive untracked for the rest of
    /// the app's life), drop attached indexes and the security-scoped bookmarks
    /// (a deleted chat must not keep durable access to folders it was granted),
    /// then drop the session objects. If the active chat was among them, adopt
    /// the first VISIBLE session — a hidden one is a transcript the sidebar
    /// cannot point at.
    func deleteSessions(_ ids: Set<UUID>) {
        guard !ids.isEmpty else { return }
        for id in ids {
            processRegistry.killSession(id)
            documentIndexes[id]?.cancel()
            documentIndexes.removeValue(forKey: id)
            SecurityScopedBookmark.clear(name: SecurityScopedBookmark.workingFolderName(id))
            SecurityScopedBookmark.clear(name: SecurityScopedBookmark.attachedFolderName(id))
        }
        chatSessions.removeAll { ids.contains($0.id) }
        // Stop the in-flight turns that belonged to these sessions — otherwise
        // they ghost-run invisibly with no Stop control anywhere, and no server
        // restart can clear them. The sweep is per turn: only the deleted
        // chats' turns stop. See ChatTurnEngine.stopIfOrphaned / TurnLedger.orphaned.
        chatEngine.stopIfOrphaned()
        if let active = activeChatId, ids.contains(active) {
            activeChatId = visibleChatSessions.first?.id
        }
        // Remove deleted ids from the sidebar selection so UI state stays sane.
        sidebarSelection.subtract(ids)
        // A pending confirmation naming chats that are already gone would ask
        // about nothing (or, worse, re-present after the deletion it asked for).
        pendingChatDeletion = nil
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

    /// Branch a conversation at a message: the transcript up to that point
    /// becomes a new chat, and the original is left untouched.
    ///
    /// The fork opens immediately — you asked to go somewhere else, so being
    /// left in the old thread wondering whether it worked is the wrong answer
    /// (the same reason regenerate scrolls to the reply it asked for). Both the
    /// active chat and the sidebar selection move, or the panel lights a row
    /// that is not the transcript on screen.
    ///
    /// Returns nil when there is nothing to fork, which is also what stops the
    /// menu item offering it (`ChatFork.isForkable`).
    @discardableResult
    func forkSession(_ sessionId: UUID, from messageId: UUID) -> UUID? {
        guard let source = chatSessions.first(where: { $0.id == sessionId }) else { return nil }
        let messages = ChatFork.prefix(source.messages, through: messageId)
        guard !messages.isEmpty else { return nil }
        let fork = ChatFork.session(from: source, messages: messages)
        chatSessions.insert(fork, at: 0)
        activeChatId = fork.id
        sidebarSelection = [fork.id]
        showConversation()
        saveChatHistory()
        return fork.id
    }

    /// The version list a regeneration in flight will put on whatever reply it
    /// produces, per session.
    ///
    /// Held rather than written straight onto a message because the message
    /// does not exist yet. `runPlainTurn` appends its streaming placeholder
    /// synchronously, so the gap there is one statement wide — but
    /// `runAgentLoop` appends one per tool round from inside a Task, and the
    /// reply the pager belongs to is the LAST of them. Writing at the start
    /// landed the seed on the user's own message, where the role guard dropped
    /// it, and the pager silently never appeared with Tools on.
    private var pendingRevisionSeed: [UUID: [MessageRevision]] = [:]

    /// Start a regeneration's revision list, carrying the reply being replaced.
    ///
    /// `regenerate` truncates the transcript back to the last user message, so
    /// the old reply is destroyed before the new one streams. Capturing it here
    /// is what makes the pager possible at all; `finishRevisions` applies it
    /// when the turn ends.
    func seedRevisions(in sessionId: UUID, from prior: ChatMessage) {
        let priorRevision = MessageRevision(content: prior.content,
                                            reasoningContent: prior.reasoningContent)
        let seeded = MessageRevisions.seeding(prior: priorRevision, existing: prior.revisions)
        // An empty seed is the "there was nothing worth stepping back to" case,
        // and leaving a stale one behind would attach it to a later turn.
        if seeded.isEmpty { pendingRevisionSeed.removeValue(forKey: sessionId) }
        else { pendingRevisionSeed[sessionId] = seeded }
    }

    /// Sessions whose in-flight turn EXTENDS the reply already at the end of
    /// the transcript instead of producing a new one.
    ///
    /// The pager counts REGENERATIONS — answers to the same question — and a
    /// continuation is not one of those: it is the reply you are reading,
    /// carrying on. Without this marker `finishRevisions` reached
    /// `MessageRevisions.committing` and filed the extended text as a new
    /// version, so a reply that had been regenerated once went to 3/3 the
    /// moment you finished it, and stepping back to 2/3 showed the same reply
    /// with the ending removed. Same shape as `pendingRevisionSeed`: the turn
    /// exit is the only place that knows the turn is over, so the fact has to
    /// be held from the moment the continuation is asked for.
    private var continuingSessions: Set<UUID> = []

    /// Declare this session's next turn a continuation. Must be called AFTER
    /// `stop(sessionId:)` — stop is a turn exit, and a mark placed before it
    /// would be consumed immediately (the `seedRevisions` ordering hazard).
    func markContinuing(_ sessionId: UUID) {
        continuingSessions.insert(sessionId)
    }

    /// Apply any held seed to the reply this turn produced and record that
    /// reply as the newest version. Called from turn EXITS only — a
    /// per-iteration call inside the agent loop would land the pager on the
    /// first tool round's bubble instead of on the answer.
    ///
    /// Targets the last ASSISTANT message rather than the last message: a turn
    /// stopped mid-tool-execution ends on a `tool` row, and the reply above it
    /// is still the one the pager belongs to.
    func finishRevisions(in sessionId: UUID) {
        let continuing = continuingSessions.remove(sessionId) != nil
        guard let sIdx = chatSessions.firstIndex(where: { $0.id == sessionId }),
              let mIdx = chatSessions[sIdx].messages.lastIndex(where: { $0.role == .assistant })
        else {
            // Nothing to attach it to, and holding it would leak the seed onto
            // an unrelated later turn.
            pendingRevisionSeed.removeValue(forKey: sessionId)
            return
        }
        let msg = chatSessions[sIdx].messages[mIdx]
        let seed = pendingRevisionSeed.removeValue(forKey: sessionId)
        // A continuation is an in-place extension of the version being read,
        // exactly like an edit — and for the same reason `applyingEdit` exists:
        // stepping away and back reloads `content` from the stored revision, so
        // an unsynced list silently discards the text the model just added.
        if continuing {
            guard !msg.revisions.isEmpty else { return }
            chatSessions[sIdx].messages[mIdx].revisions =
                MessageRevisions.applyingEdit(msg.content, to: msg.revisions, at: msg.activeRevision)
            return
        }
        guard seed != nil || !msg.revisions.isEmpty else { return }
        let finished = MessageRevision(content: msg.content, reasoningContent: msg.reasoningContent)
        let result = MessageRevisions.finishing(seed: seed, existing: msg.revisions, finished: finished)
        chatSessions[sIdx].messages[mIdx].revisions = result.revisions
        chatSessions[sIdx].messages[mIdx].activeRevision = result.index
    }

    /// Rewrite the model's own reply in place.
    ///
    /// Unlike editing YOUR message — which drops everything after it and
    /// resubmits, because the conversation past that point answered something
    /// you no longer said — editing a reply changes only that reply. It is
    /// putting words in the model's mouth: the turn already happened, and the
    /// point is to steer what comes NEXT (Continue, or the following turn),
    /// not to re-run it. Deleting the rest of the thread would take the choice
    /// away; the messages after it are still there to delete by hand.
    func editAssistantMessage(in sessionId: UUID, messageId: UUID, newText: String) {
        guard let sIdx = chatSessions.firstIndex(where: { $0.id == sessionId }),
              let mIdx = chatSessions[sIdx].messages.firstIndex(where: { $0.id == messageId })
        else { return }
        let msg = chatSessions[sIdx].messages[mIdx]
        chatSessions[sIdx].messages[mIdx].content = newText
        chatSessions[sIdx].messages[mIdx].revisions =
            MessageRevisions.applyingEdit(newText, to: msg.revisions, at: msg.activeRevision)
        // The notice described the text that was there. It is not that text
        // any more, and an edited reply is not a truncated one.
        chatSessions[sIdx].messages[mIdx].truncationNotice = nil
        chatSessions[sIdx].updatedAt = Date()
    }

    /// Show a different version of a reply. `content` is what every reader
    /// uses, so switching writes the selected version into it.
    func selectRevision(in sessionId: UUID, messageId: UUID, index: Int) {
        guard let sIdx = chatSessions.firstIndex(where: { $0.id == sessionId }),
              let mIdx = chatSessions[sIdx].messages.firstIndex(where: { $0.id == messageId })
        else { return }
        let msg = chatSessions[sIdx].messages[mIdx]
        guard index >= 0, index < msg.revisions.count else { return }
        chatSessions[sIdx].messages[mIdx].activeRevision = index
        chatSessions[sIdx].messages[mIdx].content = msg.revisions[index].content
        chatSessions[sIdx].messages[mIdx].reasoningContent = msg.revisions[index].reasoningContent
    }

    /// Drop the "this reply was cut short" footnote from the last message.
    ///
    /// Called when a continuation starts: the notice is a statement about the
    /// reply, and the reply is about to stop being cut. Leaving it would put
    /// "Stopped — hit the output limit" under a paragraph that carried on.
    func clearTruncationNotice(in sessionId: UUID) {
        guard let sIdx = chatSessions.firstIndex(where: { $0.id == sessionId }),
              let mIdx = chatSessions[sIdx].messages.indices.last else { return }
        chatSessions[sIdx].messages[mIdx].truncationNotice = nil
    }

    func appendMessage(to sessionId: UUID, message: ChatMessage) {
        guard let idx = chatSessions.firstIndex(where: { $0.id == sessionId }) else { return }
        chatSessions[idx].messages.append(message)
        chatSessions[idx].updatedAt = Date()
        // Auto-title from the first user message. The gate is "is this still a
        // placeholder", NOT one spelled-out literal: an agent thread starts as
        // "New agent", and a literal compare would leave it that way forever.
        if ChatSessionTitle.isPlaceholder(chatSessions[idx].title),
           message.role == .user,
           let title = ChatSessionTitle.derived(fromFirstMessage: message.content) {
            chatSessions[idx].title = title
        }
    }

    /// Drop one message from a conversation.
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

    /// Drops every message from `count` onward, keeping only the first `count`.
    /// Used by regenerate (drop the old user turn + reply so `runTurn` can
    /// re-append a fresh copy) and by edit-and-resend (drop everything from
    /// the edited message onward before resubmitting its new text).
    func truncateMessages(in sessionId: UUID, keepingFirst count: Int) {
        guard let sIdx = chatSessions.firstIndex(where: { $0.id == sessionId }),
              count < chatSessions[sIdx].messages.count else { return }
        chatSessions[sIdx].messages.removeSubrange(count...)
        chatSessions[sIdx].updatedAt = Date()
        saveChatHistory()
    }

    /// - Parameter addingCompletionTokens: the usage describes a SECOND
    ///   generation into a message that already holds one (a continuation), so
    ///   the completion count is added rather than replaced. `content` is
    ///   appended here by construction, and a footnote reading "42 tokens"
    ///   under a reply of 900 describes only the sentence that finished it.
    ///   The prompt count and the rate stay the latest generation's: the prompt
    ///   for a continuation already includes everything before it, and a rate
    ///   is not a quantity to sum.
    func updateLastMessage(in sessionId: UUID, content: String? = nil, reasoning: String? = nil, streaming: Bool? = nil, usage: TokenUsage? = nil, addingCompletionTokens: Bool = false, truncation: TruncationNotice.Notice? = nil) {
        guard let sIdx = chatSessions.firstIndex(where: { $0.id == sessionId }),
              !chatSessions[sIdx].messages.isEmpty else { return }
        let mIdx = chatSessions[sIdx].messages.count - 1
        if let content { chatSessions[sIdx].messages[mIdx].content += content }
        if let truncation { chatSessions[sIdx].messages[mIdx].truncationNotice = truncation }
        if let usage {
            chatSessions[sIdx].messages[mIdx].promptTokens = usage.promptTokens
            chatSessions[sIdx].messages[mIdx].completionTokens = addingCompletionTokens
                ? (chatSessions[sIdx].messages[mIdx].completionTokens ?? 0) + usage.completionTokens
                : usage.completionTokens
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

    // The welcome screen is a SHEET on the chat window (`showWelcome`,
    // presented by the chat scene) — it was a hand-built `NSWindow` around an
    // `NSHostingView` here, which is why it inherited no environment (every
    // object the starter card reads had to be passed in by hand or SwiftUI
    // trapped at first render) and why it could float over an empty desktop.
}
