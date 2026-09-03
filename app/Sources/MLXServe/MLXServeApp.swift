import SwiftUI
import AppKit

/// Process entry point. Normally hands off to the SwiftUI app, but first honors
/// an opt-in diagnostic: `SANDBOX_SMOKE=1` boots the agent-sandbox Linux guest
/// (Virtualization.framework) and runs a few commands through it, then exits —
/// a way to prove the sandbox path end-to-end from a properly-entitled binary
/// (VZ needs the virtualization entitlement on the *process*, which the signed
/// MLXCore binary has but the `xctest` host does not). No effect on normal
/// launches. `CONTAIN_SMOKE=1` is honored as a legacy alias.
@main
struct MLXCoreEntryPoint {
    static func main() {
        let env = ProcessInfo.processInfo.environment
        if env["SANDBOX_SMOKE"] == "1" || env["CONTAIN_SMOKE"] == "1" {
            SandboxSmoke.run()
        }
        MLXCoreApp.main()
    }
}

struct MLXCoreApp: App {
    private static let menuBarIcon: NSImage = {
        guard let img = BundledAsset.image("tray.png") else {
            return NSImage(systemSymbolName: "brain.head.profile", accessibilityDescription: "MLX Core")!
        }
        img.size = NSSize(width: 18, height: 18)
        img.isTemplate = true
        return img
    }()

    @NSApplicationDelegateAdaptor(MLXCoreAppDelegate.self) private var appDelegate
    @StateObject private var appState = AppState()
    @StateObject private var hfSearch = HFSearchService()
    @Environment(\.openWindow) private var openWindow

    private func menuBarIcon(for status: ServerStatus) -> NSImage {
        let color: NSColor?
        switch status {
        case .running: color = nil
        case .starting: color = .systemOrange
        case .stopped, .error: color = .systemRed
        }
        guard let color else { return Self.menuBarIcon }
        let base = Self.menuBarIcon
        let tinted = NSImage(size: base.size, flipped: false) { rect in
            base.draw(in: rect)
            color.set()
            rect.fill(using: .sourceAtop)
            return true
        }
        tinted.isTemplate = false
        return tinted
    }

    /// Accent-tinted variant of the tray icon, shown while the voice assistant
    /// is running so the menu bar reflects the active session at a glance.
    private static let activeMenuBarIcon: NSImage = {
        let base = menuBarIcon
        let tinted = NSImage(size: base.size, flipped: false) { rect in
            base.draw(in: rect)
            NSColor.controlAccentColor.set()
            rect.fill(using: .sourceAtop)
            return true
        }
        tinted.isTemplate = false
        return tinted
    }()

    /// Opening a window used to be `openWindow(id:)` → `activate()` while the
    /// app was still `.accessory` — the inverted order that left the window
    /// semi-focused until the user clicked or typed. `AppActivation` flips to
    /// `.regular` first; see the ordering rule in that file.
    private func openAndFocus(_ id: String) {
        AppActivation.openWindow(id: id, using: openWindow)
    }

    var body: some Scene {
        MenuBarExtra {
            StatusMenuView(
                openChat: { appState.showChat() },
                openModelBrowser: { appState.showModels() },
                openImageGen: { appState.showCreate(.image) },
                openVideoGen: { appState.showCreate(.video) },
                openAudioGen: { appState.showCreate(.audio) },
                openModel3DGen: { appState.showCreate(.model3d) },
                openSettings: { appState.showSettings() },
                openServerLog: { openAndFocus("serverLog") },
                openTasks: { appState.showTasks() },
                openAgents: { openAndFocus("agents") },
            )
                .environmentObject(appState)
                .environmentObject(appState.server)
                .environmentObject(appState.downloads)
                .environmentObject(appState.voice)
        } label: {
            // Observe the voice controller so the tray icon picks up the accent
            // tint the instant a hands-free session starts or stops.
            MenuBarLabel(idleIcon: menuBarIcon(for: appState.server.status),
                         activeIcon: Self.activeMenuBarIcon,
                         voice: appState.voice)
                // A tapped task notification deep-links here; open the Tasks window
                // (the label is always present, so this fires even with no window open).
                .onChange(of: appState.pendingTaskDeepLink) { _, taskId in
                    // A tapped task notification: the Tasks pane is part of the
                    // chat window now, so this brings that window up on it and
                    // TaskListPane consumes the id in .onAppear/.onChange.
                    if taskId != nil { appState.showTasks() }
                }
                // Quick launcher "Open in chat" (⌘↩): same always-present bridge —
                // the launcher panel can't reach SwiftUI's openWindow itself.
                .onChange(of: appState.pendingChatOpenTick) { _, _ in
                    openAndFocus("chat")
                }

        }
        .menuBarExtraStyle(.window)

        Window("MLX Core", id: "chat") {
            ChatView()
                .environmentObject(appState)
                // The Model Browser is a MODE of this window now
                // (`ChatWorkspace`), so everything its panes read has to be
                // injected HERE — there is no second window to inject it into,
                // and SwiftUI reports a missing one as a render-time trap, not
                // a compile error (live crash 2026-08-08 on `downloads`).
                // Pinned by `testTheChatWindowInjectsEveryObjectTheBrowserPaneReads`.
                .environmentObject(hfSearch)
                .environmentObject(appState.downloads)
                // The four media generators are PAGES of this window now
                // (`ChatWorkspace.create`), not windows of their own.
                .environmentObject(appState.imageGen)
                .environmentObject(appState.videoGen)
                .environmentObject(appState.audioGen)
                .environmentObject(appState.musicGen)
                .environmentObject(appState.model3dGen)
                // Settings, Tasks and Agents render here as modes too, so their
                // objects ride this scene (`ChatWorkspace`).
                .environmentObject(appState.taskScheduler)
                .environmentObject(appState.terminals)
                .environmentObject(appState.agents)
                .environmentObject(appState.server)
                .environmentObject(appState.toolExecutor)
                .environmentObject(appState.agentMemory)
                .environmentObject(appState.mcpManager)
                .environmentObject(appState.chatEngine)
                .environmentObject(appState.voice)
                .environmentObject(appState.processRegistry)
                // 1070: this window hosts Models/Settings/Create panes and the
                // composer row carries the model pill now — smaller floors
                // clipped them.
                .frame(minWidth: 1070, minHeight: 500)
                // The intro screen, as a DIALOG over the chat window rather
                // than a floating window of its own. The injections below are
                // NOT redundant: a sheet does not inherit the environment of
                // the view it hangs on (`SheetEnvironmentAuditTests`).
                .sheet(isPresented: $appState.showWelcome) {
                    WelcomeView(onDismiss: { appState.showWelcome = false },
                                hasChatModels: appState.welcomeHasChatModels,
                                onOpenModelBrowser: { appState.showModels() },
                                onOpenChat: { appState.pendingChatOpenTick += 1 })
                        .environmentObject(appState)
                        .environmentObject(appState.downloads)
                        .environmentObject(appState.server)
                }
                .onDisappear {
                    Task { await appState.mcpManager.stopAll() }
                }
                .appAppearance()
        }
        // Roomier than the old 900x650: this window is three things now
        // (transcript, model browser, media generators) and the two it gained
        // were 960pt-wide windows in their own right.
        .defaultSize(width: 1160, height: 780)

        Window("Browser", id: "browser") {
            BrowserView()
                .appAppearance()
        }
        .defaultSize(width: 1024, height: 768)

        // Dedicated terminal-style window for the server's live stderr.
        // The inline log on the tray popover is still there for a glance;
        // this is the one you keep open for long sessions where copy/paste
        // and a roomy scroll-back matter.
        Window("Server Log", id: "serverLog") {
            ServerLogWindowView()
                .environmentObject(appState.server)
                .appAppearance()
        }
        .defaultSize(width: 900, height: 560)

        // A sandbox terminal moved out of the chat window ("Move Tab to New
        // Window", 2026-09-02). One window per session id; the session itself
        // stays in `appState.terminals` — the window only hosts its view, so
        // closing the window puts the terminal back in the sidebar's detail
        // column and ends nothing.
        WindowGroup("Terminal", id: "terminalWindow", for: UUID.self) { $sessionId in
            if let sessionId {
                TerminalWindowView(sessionId: sessionId)
                    .environmentObject(appState)
                    .environmentObject(appState.server)
                    .environmentObject(appState.terminals)
                    .frame(minWidth: 560, minHeight: 360)
                    .appAppearance()
            }
        }
        .defaultSize(width: 900, height: 600)

        // Agents (personas): who you're talking to, and the settings that
        // conversation runs under. Configuration only — chatting with an agent
        // happens in the Chat window.
        Window("Agents", id: "agents") {
            AgentsWindow()
                .environmentObject(appState)
                .environmentObject(appState.agents)
                .environmentObject(appState.server)
                .frame(minWidth: 760, minHeight: 520)
                .appAppearance()
        }
        .defaultSize(width: 900, height: 640)

        .commands {
            CommandGroup(replacing: .newItem) {
                    Button("New Chat") {
                        openAndFocus("chat")
                        _ = appState.newChatSession()
                    }
                    .keyboardShortcut("n", modifiers: [.command])

                    // ⌘⌫, Finder's own "move to trash". A MENU command rather
                    // than a key handler on the sidebar: `.onDeleteCommand`
                    // there only fires while that view is first responder, and
                    // a ScrollView of plain Buttons never is — see the note in
                    // `ChatSidebar.conversationsSidebar`. It also makes the
                    // shortcut discoverable, which a bare key never was.
                    Button("Delete Chat") { appState.requestChatDeletionFromMenu() }
                        .keyboardShortcut(.delete, modifiers: [.command])
                        .disabled(appState.chatDeletionTarget == nil)
                }
            CommandMenu("Agent") {
                Button("Agents…") { openAndFocus("agents") }
                    .keyboardShortcut("a", modifiers: [.command, .shift])

                Button("Browser") { openAndFocus("browser") }
                    .keyboardShortcut("b", modifiers: [.command, .shift])

                Button("Settings…") { appState.showSettings() }
                    .keyboardShortcut(",", modifiers: [.command])

                Button("Edit System Prompt") {
                    AgentPrompt.openSystemPromptInEditor()
                }
                .keyboardShortcut("p", modifiers: [.command, .shift])

                // Pull in the latest built-in default when ours has moved ahead of
                // the on-disk copy. Backs up the user's current prompt first.
                Button("Update System Prompt to Latest…") {
                    AgentPrompt.runSystemPromptUpdateFlow()
                }
                .disabled(!AgentPrompt.isSystemPromptOutdated())

                Button("Open Memory File") {
                    let path = NSString(string: "~/.mlx-serve/memory.md").expandingTildeInPath
                    if !FileManager.default.fileExists(atPath: path) {
                        try? "".write(toFile: path, atomically: true, encoding: .utf8)
                    }
                    NSWorkspace.shared.open(URL(fileURLWithPath: path))
                }

                Button("Open Skills Folder") {
                    // Accessing the shared manager seeds the example skill on
                    // first run; the create is a no-op if it already exists.
                    let path = AgentPrompt.skillManager.skillsDirectory
                    try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
                    NSWorkspace.shared.open(URL(fileURLWithPath: path))
                }

                Button("Open MLX Serve Folder") {
                    let path = NSString(string: "~/.mlx-serve").expandingTildeInPath
                    NSWorkspace.shared.open(URL(fileURLWithPath: path))
                }
            }

            // Menu-bar twin of the chat's empty-state discovery chips
            // (ChatEmptyState): every feature that otherwise lives only in
            // the tray popover, reachable from the menu bar and Help-menu
            // search. The media section iterates the SAME catalog as the
            // chips so the two lists cannot drift.
            CommandMenu("Tools") {
                // ⌘L: the model switcher, over the same rows the composer's
                // pill offers. A menu key equivalent so it works from every
                // window, and it goes through AppState's door — which raises
                // the picker AND brings the chat window forward.
                Button("Switch Model…") { appState.showModelPalette() }
                    .keyboardShortcut("l", modifiers: [.command])

                Button("Browse Models…") { appState.showModels() }
                    .keyboardShortcut("m", modifiers: [.command, .shift])

                Button("Scheduled Tasks…") { appState.showTasks() }
                    .keyboardShortcut("t", modifiers: [.command, .shift])

                Divider()

                // The four generators are PAGES of the chat window now
                // (`.create(...)` actions, windowId nil) — dispatch through the
                // one door, `AppState.showCreate`.
                ForEach(ChatEmptyState.mediaItems) { item in
                    if case .create(let experiment) = item.action {
                        Button("\(item.title)…") { appState.showCreate(experiment) }
                    }
                }

                Divider()

                // DMG builds only — the MAS build can't detect or launch
                // other apps' CLIs (same gate as the tray's Code button).
                if BuildFeatures.current.cliLauncher {
                    Button("Launch Claude Code…") {
                        launchClaudeCodeWithPicker(
                            baseURL: appState.server.baseURL,
                            serverContextLength: appState.server.chatModelInfo?.contextLength)
                    }
                }

                // No .keyboardShortcut here: ⌃Space is registered as a GLOBAL
                // Carbon hotkey (QuickLauncherController); a menu key
                // equivalent on the same combo would race it while the app is
                // frontmost, so the combo rides the title instead.
                Button("Quick Launcher (\(QuickLauncherHotKey.display))") {
                    if !appState.quickLauncherEnabled { appState.quickLauncherEnabled = true }
                    appState.quickLauncher.show()
                }
            }
        }
    }
}

/// Menu-bar label that swaps to an accent-tinted icon while the voice assistant
/// is running. A tiny view so it can `@ObservedObject` the controller — the App
/// scene's `label:` closure can't otherwise react to voice state changes.
private struct MenuBarLabel: View {
    let idleIcon: NSImage
    let activeIcon: NSImage
    @ObservedObject var voice: VoiceModeController

    var body: some View {
        Image(nsImage: voice.isActive ? activeIcon : idleIcon)
    }
}
