import SwiftUI
import AppKit
import Virtualization

/// The detail column for the sandbox desktop (computer use): the guest's
/// framebuffer, live, with the keyboard and mouse forwarded when the pane has
/// focus — so the user watches the agent work and can take over at any point.
/// A `ChatWorkspace` mode like the terminal panes (`showDesktop()` is the
/// door). Off, installing, failed and ready each get their own state above the
/// screen; the install progress is the last apt line.
struct DesktopPane: View {
    @EnvironmentObject private var appState: AppState
    @ObservedObject private var sandbox = AgentSandbox.shared
    /// The chat column beside the screen: the ACTIVE chat, so what the agent
    /// is doing on the screen and what it says stay in one view, and a
    /// message typed here mid-turn reaches it on its next step.
    @AppStorage("desktopPaneShowsChat") private var showsChat = true
    @AppStorage("desktopPaneChatWidth") private var storedChatWidth: Double = DesktopPane.defaultChatWidth

    /// The composer row (attach, Think, Tools, MCP, model pill, voice, Send)
    /// needs ~440 pt before it clips (live 2026-09-06 at 310 pt: the mode
    /// buttons and the placeholder were cut off, the footer wrapped one
    /// character per line).
    static let defaultChatWidth: Double = 480
    static let minChatWidth: Double = 440
    static let maxChatWidth: Double = 900
    static let minScreenWidth: Double = 360
    static let handleWidth: Double = 7

    /// The remembered chat column width, clamped: a stale or garbage default
    /// never yields a column too narrow to type in or wider than a window.
    /// With `available` (the pane's width) the column also yields to the
    /// screen's minimum: at a small window it shrinks instead of pushing the
    /// composer off the right edge (live 2026-09-06), down to a floor that
    /// still fits the compact composer row.
    static func chatColumnWidth(stored: Double, default fallback: Double = defaultChatWidth,
                                available: Double? = nil) -> Double {
        let wanted = (stored.isFinite && stored > 0) ? stored : fallback
        var w = min(max(wanted, minChatWidth), maxChatWidth)
        if let available, available > 0 {
            let room = available - minScreenWidth - handleWidth
            w = max(min(w, room), Self.compactChatWidth)
        }
        return w
    }

    /// The narrowest column the compact composer row still fits.
    static let compactChatWidth: Double = 320

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if showsChat {
                // Not an HSplitView: it hands the chat either its minimum or
                // its maximum depending on layout priority (live 2026-09-06,
                // both wrong). The chat column is exactly the remembered
                // width, the screen (aspect-fit, any width) takes the rest,
                // and the divider is a drag handle that writes the width.
                GeometryReader { geo in
                    HStack(spacing: 0) {
                        screenColumn
                            .frame(minWidth: Self.minScreenWidth, maxWidth: .infinity, maxHeight: .infinity)
                        splitHandle
                        chatColumn
                            .frame(width: Self.chatColumnWidth(stored: storedChatWidth, available: geo.size.width))
                    }
                }
            } else {
                screenColumn
            }
        }
    }

    @State private var dragStartWidth: Double?

    private var splitHandle: some View {
        Rectangle()
            .fill(Color(nsColor: .separatorColor))
            .frame(width: 1)
            .padding(.horizontal, 3)
            .contentShape(Rectangle())
            .onHover { inside in
                if inside { NSCursor.resizeLeftRight.push() } else { NSCursor.pop() }
            }
            .gesture(DragGesture(minimumDistance: 1)
                .onChanged { value in
                    let start = dragStartWidth ?? Self.chatColumnWidth(stored: storedChatWidth)
                    dragStartWidth = start
                    // Dragging LEFT widens the chat (it sits on the right).
                    storedChatWidth = Self.chatColumnWidth(stored: start - value.translation.width)
                }
                .onEnded { _ in dragStartWidth = nil })
    }

    private var screenColumn: some View {
        VStack(spacing: 0) {
            content
            if appState.serverOptions.sandbox.desktop, sandbox.guestRunning {
                Text("Click the screen to type into it. Esc stops the agent while it works.")
                    .font(.caption2).foregroundStyle(.secondary)
                    .padding(.vertical, 4)
            }
        }
    }

    /// `ChatDetailView` reads everything through the environment objects the
    /// chat window injects (pinned by the injection audit test, which lists
    /// this file). No active chat yet → open one, like the window's own
    /// conversation branch.
    @ViewBuilder
    private var chatColumn: some View {
        if let id = appState.activeChatId, appState.chatSessions.contains(where: { $0.id == id }) {
            ChatDetailView(sessionId: id)
                .frame(maxHeight: .infinity)
        } else {
            Color.clear
                .frame(maxHeight: .infinity)
                .onAppear {
                    _ = appState.newChatSession()
                    appState.applyDesktopChatPreset()
                }
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Image(systemName: "desktopcomputer").foregroundStyle(.secondary)
            Text("Sandbox Desktop").font(.headline)
            Text(Self.stateText(enabled: appState.serverOptions.sandbox.desktop,
                                guestRunning: sandbox.guestRunning,
                                state: sandbox.desktopSetupState))
                .font(.caption).foregroundStyle(.secondary)
                .lineLimit(1).truncationMode(.middle)
            Spacer()
            StopAgentButton(engine: appState.chatEngine)
            Toggle(isOn: $showsChat) {
                Label("Chat", systemImage: "sidebar.trailing")
            }
            .toggleStyle(.button).controlSize(.small)
            .help("Show the active chat beside the screen")
        }
        .padding(.horizontal, 14).padding(.vertical, 8)
    }

    /// Its own view so `isGenerating` re-renders just the button (the engine
    /// is a lazy member of AppState, not an environment object).
    private struct StopAgentButton: View {
        @ObservedObject var engine: ChatTurnEngine
        var body: some View {
            if engine.isGenerating {
                Button(role: .destructive) { engine.stop() } label: {
                    Label("Stop agent", systemImage: "stop.fill")
                }
                .controlSize(.small)
                .help("Stops every in-flight turn (Esc in the desktop does the same while the agent is working)")
            }
        }
    }

    /// One line for the header, from the three facts that decide it. Pure.
    static func stateText(enabled: Bool, guestRunning: Bool, state: AgentSandbox.DesktopSetupState) -> String {
        guard enabled else { return "off" }
        switch state {
        case .ready: return guestRunning ? "\(SandboxDesktop.display.width)x\(SandboxDesktop.display.height), live" : "guest stopped"
        case .installing(let line): return "setting up desktop… " + (line.isEmpty ? "" : line)
        case .failed(let why): return "failed: " + why
        case .idle: return guestRunning ? "starting…" : "not started"
        }
    }

    @ViewBuilder
    private var content: some View {
        if !appState.serverOptions.sandbox.desktop {
            notice {
                Image(systemName: "desktopcomputer").font(.largeTitle).foregroundStyle(.secondary)
                Text("The sandbox desktop is off").font(.headline)
                Text("Turn it on to boot the sandbox with a screen and install a small Linux desktop (XFCE, about 350 MB) the agent can operate with the computer tool. You watch and can take over here.")
                    .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    .frame(maxWidth: 420)
                Button("Install desktop") {
                    appState.serverOptions.sandbox = SandboxDesktop.withDesktop(true, appState.serverOptions.sandbox)
                }
                .keyboardShortcut(.defaultAction)
            }
        } else if case .failed(let why) = sandbox.desktopSetupState {
            notice {
                Image(systemName: "exclamationmark.triangle").font(.largeTitle).foregroundStyle(.orange)
                Text("The desktop could not be set up").font(.headline)
                Text(why).font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    .frame(maxWidth: 480).textSelection(.enabled)
                Button("Try again") {
                    Task { try? await AgentSandbox.shared.ensureDesktopProvisioned() }
                }
            }
        } else if let vm = sandbox.desktopVirtualMachine, sandbox.guestRunning {
            ZStack {
                Color.black
                VirtualMachineScreen(virtualMachine: vm, onEscape: { [engine = appState.chatEngine] in
                    guard engine.isGenerating else { return false }
                    engine.stop()
                    return true
                })
                    .aspectRatio(CGFloat(SandboxDesktop.display.width) / CGFloat(SandboxDesktop.display.height),
                                 contentMode: .fit)
                if case .installing = sandbox.desktopSetupState {
                    installingOverlay
                }
            }
        } else {
            notice {
                ProgressView()
                Text(sandbox.guestRunning ? "Starting the desktop…" : "Starting the sandbox…").font(.headline)
                Text("The screen appears once the guest has booted with a display.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .task {
                // Opening the pane is the one interactive way to kick a
                // provisioning that has not started (fresh launch).
                try? await AgentSandbox.shared.ensureDesktopProvisioned()
            }
        }
    }

    private var installingOverlay: some View {
        VStack(spacing: 6) {
            ProgressView().controlSize(.small)
            Text("Setting up the desktop (apt-get)…").font(.caption).foregroundStyle(.white)
            if case .installing(let line) = sandbox.desktopSetupState, !line.isEmpty {
                Text(line).font(.caption2.monospaced()).foregroundStyle(.white.opacity(0.8))
                    .lineLimit(1).truncationMode(.middle).frame(maxWidth: 520)
            }
        }
        .padding(12)
        .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 8))
    }

    private func notice<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        VStack(spacing: 10) { content() }
            .padding(24)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - VZVirtualMachineView host

/// `VZVirtualMachineView` in SwiftUI. Created and attached on the main thread
/// (the framework's requirement); the machine keeps running on its own queue.
/// Esc is intercepted ONLY while a turn is driving (`onEscape` returns true
/// after stopping it) so the key still reaches the guest otherwise.
struct VirtualMachineScreen: NSViewRepresentable {
    let virtualMachine: VZVirtualMachine
    var onEscape: () -> Bool = { false }

    func makeNSView(context: Context) -> DesktopVMView {
        let v = DesktopVMView()
        v.capturesSystemKeys = false
        v.automaticallyReconfiguresDisplay = true
        v.virtualMachine = virtualMachine
        v.onEscape = onEscape
        return v
    }

    func updateNSView(_ v: DesktopVMView, context: Context) {
        // A rebooted guest is a new machine object; re-point rather than
        // rebuild the view.
        if v.virtualMachine !== virtualMachine { v.virtualMachine = virtualMachine }
        v.onEscape = onEscape
    }
}

final class DesktopVMView: VZVirtualMachineView {
    var onEscape: () -> Bool = { false }

    override func keyDown(with event: NSEvent) {
        if event.keyCode == 53, onEscape() { return }  // 53 = Esc
        super.keyDown(with: event)
    }

    override var acceptsFirstResponder: Bool { true }
}
