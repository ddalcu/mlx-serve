import SwiftUI
import AppKit

@main
struct MLXClawApp: App {
    private static let menuBarIcon: NSImage = {
        // Try Bundle.main (works in .app bundles) then SPM bundle (works in dev builds)
        let candidates: [URL?] = [
            Bundle.main.resourceURL?.appendingPathComponent("tray.png"),
            Bundle.main.bundleURL.appendingPathComponent("MLXClaw_MLXClaw.bundle/Resources/tray.png"),
        ]
        for case let url? in candidates {
            if let img = NSImage(contentsOf: url) {
                img.size = NSSize(width: 18, height: 18)
                img.isTemplate = true
                return img
            }
        }
        return NSImage(systemSymbolName: "brain.head.profile", accessibilityDescription: "MLX Claw")!
    }()

    @StateObject private var appState = AppState()
    @Environment(\.openWindow) private var openWindow

    private func openAndFocus(_ id: String) {
        openWindow(id: id)
        NSApplication.shared.activate(ignoringOtherApps: true)
        // Bring the specific window to front
        DispatchQueue.main.async {
            let title = id == "chat" ? "MLX Claw" : "Browser"
            NSApplication.shared.windows
                .first { $0.title == title }?
                .makeKeyAndOrderFront(nil)
        }
    }

    var body: some Scene {
        MenuBarExtra {
            StatusMenuView(openChat: { openAndFocus("chat") }, openBrowser: { openAndFocus("browser") })
                .environmentObject(appState)
                .environmentObject(appState.server)
                .environmentObject(appState.downloads)
        } label: {
            Image(nsImage: Self.menuBarIcon)
        }
        .menuBarExtraStyle(.window)

        Window("MLX Claw", id: "chat") {
            ChatView()
                .environmentObject(appState)
                .environmentObject(appState.server)
                .environmentObject(appState.toolExecutor)
                .environmentObject(appState.agentMemory)
                .frame(minWidth: 700, minHeight: 500)
        }
        .defaultSize(width: 900, height: 650)

        Window("Browser", id: "browser") {
            BrowserView()
        }
        .defaultSize(width: 1024, height: 768)
        .commands {
            CommandMenu("Agent") {
                Button("Edit System Prompt") {
                    AgentPrompt.openSystemPromptInEditor()
                }
                .keyboardShortcut("p", modifiers: [.command, .shift])

                Button("Open Memory File") {
                    let path = NSString(string: "~/.mlx-serve/memory.md").expandingTildeInPath
                    if !FileManager.default.fileExists(atPath: path) {
                        try? "".write(toFile: path, atomically: true, encoding: .utf8)
                    }
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
