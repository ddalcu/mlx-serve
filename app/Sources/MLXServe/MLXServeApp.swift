import SwiftUI
import AppKit

@main
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

    private func openAndFocus(_ id: String) {
        openWindow(id: id)
        NSApplication.shared.activate(ignoringOtherApps: true)
        // Bring the specific window to front
        DispatchQueue.main.async {
            let title: String
            switch id {
            case "chat": title = "MLX Core"
            case "modelBrowser": title = "Model Browser"
            case "imageGen": title = "Image Generation"
            case "videoGen": title = "Video Generation"
            default: title = "Browser"
            }
            NSApplication.shared.windows
                .first { $0.title == title }?
                .makeKeyAndOrderFront(nil)
        }
    }

    var body: some Scene {
        MenuBarExtra {
            StatusMenuView(
                openChat: { openAndFocus("chat") },
                openBrowser: { openAndFocus("browser") },
                openModelBrowser: { openAndFocus("modelBrowser") },
                openImageGen: { openAndFocus("imageGen") },
                openVideoGen: { openAndFocus("videoGen") }
            )
                .environmentObject(appState)
                .environmentObject(appState.server)
                .environmentObject(appState.downloads)
                .environmentObject(appState.python)
        } label: {
            Image(nsImage: menuBarIcon(for: appState.server.status))
        }
        .menuBarExtraStyle(.window)

        Window("MLX Core", id: "chat") {
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

        Window("Model Browser", id: "modelBrowser") {
            ModelBrowserView()
                .environmentObject(hfSearch)
                .environmentObject(appState)
                .environmentObject(appState.downloads)
                .frame(minWidth: 700, minHeight: 400)
        }
        .defaultSize(width: 900, height: 600)

        Window("Image Generation", id: "imageGen") {
            ImageGenView()
                .environmentObject(appState.python)
                .environmentObject(appState.imageGen)
                .environmentObject(appState.server)
        }
        .defaultSize(width: 960, height: 700)

        Window("Video Generation", id: "videoGen") {
            VideoGenView()
                .environmentObject(appState.python)
                .environmentObject(appState.videoGen)
                .environmentObject(appState.server)
        }
        .defaultSize(width: 960, height: 700)
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
