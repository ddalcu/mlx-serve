import SwiftUI

// MARK: - Catalog (pure)

/// The empty conversation's discovery layer: what this app can do beyond the
/// chat itself, offered where a new user is actually looking (right under the
/// greeting of a chat with nothing in it) and gone the moment the
/// conversation starts. Media generation, the Model Browser, Tasks and the
/// coding-agent launcher all lived only in the menu-bar tray, and users
/// reported not finding any of them.
///
/// Pure data so the surfaces are testable: every window id is pinned against
/// the scene graph in ChatEmptyStateTests, the Tools command menu (the
/// always-available twin in the menu bar) iterates the SAME `mediaItems`
/// catalog, and the Code Launcher chip renders the tray Code button's own
/// `CLILauncherMenuItems` — one list per surface pair, so none can drift.
enum ChatEmptyState {
    enum Action: Equatable {
        case window(String)   // a Window scene id, opened via AppActivation
        case mediaMenu        // chip renders as a Menu over `mediaItems`
        case codeLauncher     // chip renders the tray's CLI launcher dropdown
    }

    struct Item: Equatable, Identifiable {
        let id: String
        let title: String
        let systemImage: String
        let tint: Color
        let action: Action
        /// Tooltip; chips carry text labels so this is a courtesy, not the
        /// icon-only hover-card obligation.
        var help: String = ""

        var windowId: String? {
            if case .window(let id) = action { return id }
            return nil
        }
    }

    /// The four generation windows behind the "Create Media" chip — and the
    /// Tools menu's media section (same array, deliberately).
    static let mediaItems: [Item] = [
        Item(id: "media-image", title: "Image Generation", systemImage: "photo",
             tint: .purple, action: .window("imageGen")),
        Item(id: "media-video", title: "Video Generation", systemImage: "film",
             tint: .indigo, action: .window("videoGen")),
        Item(id: "media-audio", title: "Audio & Music", systemImage: "waveform",
             tint: .pink, action: .window("audioGen")),
        Item(id: "media-3d", title: "3D Model", systemImage: "cube",
             tint: .brown, action: .window("model3dGen")),
    ]

    /// The chip row. The Code Launcher is DMG-only — the MAS build can't
    /// detect or launch other apps' CLIs (same gate as the tray's Code
    /// button), and a chip that can only fail is the dead-control class.
    static func chips(cliLauncherAvailable: Bool = BuildFeatures.current.cliLauncher) -> [Item] {
        var items: [Item] = [
            Item(id: "create", title: "Create Media", systemImage: "wand.and.stars",
                 tint: .purple, action: .mediaMenu,
                 help: "Generate images, video, audio, or 3D models"),
            Item(id: "models", title: "Browse Models", systemImage: "magnifyingglass",
                 tint: .blue, action: .window("modelBrowser"),
                 help: "Search and download models"),
            Item(id: "tasks", title: "Tasks", systemImage: "clock.badge.checkmark",
                 tint: .orange, action: .window("tasks"),
                 help: "Scheduled and background agent tasks"),
        ]
        if cliLauncherAvailable {
            items.append(Item(id: "code", title: "Code Launcher", systemImage: "terminal",
                              tint: .green, action: .codeLauncher,
                              help: "Launch a coding agent against this server — on this Mac or inside the sandbox"))
        }
        return items
    }

    /// Windows open fine with the server stopped (they load models on
    /// demand); only a CLI launch bakes the live server's URL into the
    /// session it starts — the tray disables its Code button for the same
    /// reason.
    static func isEnabled(_ item: Item, serverRunning: Bool) -> Bool {
        item.action == .codeLauncher ? serverRunning : true
    }
}

// MARK: - Chip row

/// Discovery chips rendered under the empty conversation's greeting.
struct EmptyStateChipRow: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @Environment(\.openWindow) private var openWindow
    /// Owned here (not by the menu content) — Menu bodies are rebuilt lazily,
    /// which would re-init a @StateObject inside them per open.
    @StateObject private var cliDetector = CLILauncher()

    var body: some View {
        HStack(spacing: 8) {
            ForEach(ChatEmptyState.chips()) { item in
                switch item.action {
                case .mediaMenu:
                    chipMenu(item) {
                        ForEach(ChatEmptyState.mediaItems) { media in
                            Button { open(media) } label: {
                                Label(media.title, systemImage: media.systemImage)
                            }
                        }
                    }
                case .codeLauncher:
                    chipMenu(item) {
                        CLILauncherMenuItems(
                            detector: cliDetector,
                            baseURL: server.baseURL,
                            servedModelId: server.chatModelId ?? "mlx-serve",
                            serverContextLength: server.chatModelInfo?.contextLength,
                            models: server.allModels,
                            openSandboxAgent: { agentId in
                                // Post the request FIRST — the Sandbox window
                                // reads it in .onAppear when this click is
                                // what opens the window (the tray's pattern).
                                appState.pendingSandboxAgentLaunch = .init(agentId: agentId)
                                AppActivation.openWindow(id: "sandboxTerminal", using: openWindow)
                            })
                    }
                case .window:
                    Button { open(item) } label: {
                        EmptyStateChipLabel(item: item)
                    }
                    .buttonStyle(.plain)
                    .help(item.help)
                }
            }
        }
        .frame(maxWidth: .infinity)
        .task { await cliDetector.refresh() }
    }

    /// Shared chrome for the two dropdown chips (Create Media, Code Launcher).
    private func chipMenu<Content: View>(_ item: ChatEmptyState.Item,
                                         @ViewBuilder content: () -> Content) -> some View {
        let enabled = ChatEmptyState.isEnabled(item, serverRunning: server.status == .running)
        return Menu {
            content()
        } label: {
            EmptyStateChipLabel(item: item, showsChevron: true)
        }
        // Same trio as the composer's paperclip: .borderlessButton substitutes
        // its own chrome on macOS, dropping the capsule.
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .disabled(!enabled)
        .opacity(enabled ? 1 : 0.4)
        .help(enabled ? item.help : "Start the server to launch a coding agent")
    }

    private func open(_ item: ChatEmptyState.Item) {
        if case .window(let id) = item.action {
            AppActivation.openWindow(id: id, using: openWindow)
        }
    }
}

/// Shared chip chrome: tinted glyph, title, optional menu chevron. Hover state
/// lives here so the plain Button and the Menu label brighten identically.
private struct EmptyStateChipLabel: View {
    let item: ChatEmptyState.Item
    var showsChevron = false
    @State private var hovering = false

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: item.systemImage)
                .font(.system(size: 11.5, weight: .semibold))
                .foregroundStyle(item.tint)
            Text(item.title)
                .font(.system(size: 12.5, weight: .medium))
                .foregroundStyle(.primary)
            if showsChevron {
                Image(systemName: "chevron.down")
                    .font(.system(size: 8, weight: .semibold))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 11)
        .padding(.vertical, 7)
        .background(Capsule().fill(Color.secondary.opacity(hovering ? 0.20 : 0.10)))
        .overlay(Capsule().stroke(Color.secondary.opacity(0.18), lineWidth: 0.5))
        .contentShape(Capsule())
        .onHover { hovering = $0 }
        .animation(.easeOut(duration: 0.12), value: hovering)
    }
}
