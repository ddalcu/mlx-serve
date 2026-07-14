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
        // Try Bundle.main (works in .app bundles) then SPM bundle (works in dev builds)
        let candidates: [URL?] = [
            Bundle.main.resourceURL?.appendingPathComponent("tray.png"),
            Bundle.main.bundleURL.appendingPathComponent("MLXCore_MLXCore.bundle/Resources/tray.png"),
        ]
        for case let url? in candidates {
            if let img = NSImage(contentsOf: url) {
                img.size = NSSize(width: 18, height: 18)
                img.isTemplate = true
                return img
            }
        }
        return NSImage(systemSymbolName: "brain.head.profile", accessibilityDescription: "MLX Core")!
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
                openChat: { openAndFocus("chat") },
                openModelBrowser: { openAndFocus("modelBrowser") },
                openImageGen: { openAndFocus("imageGen") },
                openVideoGen: { openAndFocus("videoGen") },
                openAudioGen: { openAndFocus("audioGen") },
                openModel3DGen: { openAndFocus("model3dGen") },
                openSettings: { openAndFocus("settings") },
                openServerLog: { openAndFocus("serverLog") },
                openTasks: { openAndFocus("tasks") },
                openSandboxTerminal: { openAndFocus("sandboxTerminal") }
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
                    if taskId != nil { openAndFocus("tasks") }
                }
                // Quick launcher "Open in chat" (⌘↩): same always-present bridge —
                // the launcher panel can't reach SwiftUI's openWindow itself.
                .onChange(of: appState.quickLauncherChatOpenTick) { _, _ in
                    openAndFocus("chat")
                }
                // Welcome window's "Browse Models" nudge: same bridge — it's a
                // bare NSHostingView outside the Scene graph.
                .onChange(of: appState.pendingModelBrowserOpenTick) { _, _ in
                    openAndFocus("modelBrowser")
                }
        }
        .menuBarExtraStyle(.window)

        Window("MLX Core", id: "chat") {
            ChatView()
                .environmentObject(appState)
                .environmentObject(appState.server)
                .environmentObject(appState.toolExecutor)
                .environmentObject(appState.agentMemory)
                .environmentObject(appState.mcpManager)
                .environmentObject(appState.chatEngine)
                .environmentObject(appState.voice)
                .environmentObject(appState.processRegistry)
                .frame(minWidth: 700, minHeight: 500)
                .onDisappear {
                    Task { await appState.mcpManager.stopAll() }
                }
        }
        .defaultSize(width: 900, height: 650)

        Window("Browser", id: "browser") {
            BrowserView()
        }
        .defaultSize(width: 1024, height: 768)

        Window("Model Browser", id: "modelBrowser") {
            ModelBrowserView()
                .environmentObject(hfSearch)
                .environmentObject(appState)
                .environmentObject(appState.downloads)
                // Lets model rows tell "selected" from "actually loaded" — the
                // In-use badge reads `server.status`.
                .environmentObject(appState.server)
                // Floor derived from sidebar + detail minimums — a smaller
                // literal here caps the window's reported minimum and lets
                // the Discover table clip off the right edge.
                .frame(minWidth: ModelBrowserMetrics.minWindowWidth, minHeight: 400)
        }
        .defaultSize(width: ModelBrowserMetrics.defaultWindowWidth,
                     height: ModelBrowserMetrics.defaultWindowHeight)

        Window("Image Generation", id: "imageGen") {
            ImageGenView()
                .environmentObject(appState.imageGen)
                .environmentObject(appState.server)
                .environmentObject(appState.downloads)
                .environmentObject(appState)
        }
        .defaultSize(width: 960, height: 700)

        Window("Video Generation", id: "videoGen") {
            VideoGenView()
                .environmentObject(appState.videoGen)
                .environmentObject(appState.server)
                .environmentObject(appState.downloads)
                .environmentObject(appState)
        }
        .defaultSize(width: 960, height: 700)

        Window("Audio Generation", id: "audioGen") {
            AudioGenView()
                .environmentObject(appState.audioGen)
                .environmentObject(appState.musicGen)
                .environmentObject(appState.server)
                .environmentObject(appState.downloads)
                .environmentObject(appState)
        }
        .defaultSize(width: 900, height: 660)

        Window("3D Generation", id: "model3dGen") {
            Model3DGenView()
                .environmentObject(appState.model3dGen)
                .environmentObject(appState.server)
                .environmentObject(appState.downloads)
                .environmentObject(appState)
        }
        .defaultSize(width: 960, height: 700)

        Window("Settings", id: "settings") {
            SettingsView()
                .environmentObject(appState)
                .environmentObject(appState.server)
                .environmentObject(appState.downloads)
                // Wider since the category sidebar landed: it takes ~200pt, and
                // the form's rows (label + explainer + control) were already
                // tight at the old 720 minimum.
                .frame(minWidth: 900, minHeight: 560)
        }
        .defaultSize(width: 1020, height: 700)

        // Dedicated terminal-style window for the server's live stderr.
        // The inline log on the tray popover is still there for a glance;
        // this is the one you keep open for long sessions where copy/paste
        // and a roomy scroll-back matter.
        Window("Server Log", id: "serverLog") {
            ServerLogWindowView()
                .environmentObject(appState.server)
        }
        .defaultSize(width: 900, height: 560)

        // Live terminal into the agent sandbox guest — see the agent's commands
        // and run your own in the same isolated Linux VM.
        Window("Sandbox Terminal", id: "sandboxTerminal") {
            SandboxTerminalView()
        }
        .defaultSize(width: 720, height: 520)

        // Scheduled / on-demand agent tasks — the unattended "claw" surface.
        Window("Tasks", id: "tasks") {
            TasksView()
                .environmentObject(appState)
                .environmentObject(appState.server)
                .environmentObject(appState.taskScheduler)
                .frame(minWidth: 720, minHeight: 480)
        }
        .defaultSize(width: 900, height: 620)
        .commands {
            CommandMenu("Agent") {
                Button("Browser") { openAndFocus("browser") }
                    .keyboardShortcut("b", modifiers: [.command, .shift])

                Button("Settings…") { openAndFocus("settings") }
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
