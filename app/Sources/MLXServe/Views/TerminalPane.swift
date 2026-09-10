import SwiftUI
import AppKit

/// The detail column while a sandbox terminal row is selected: the live TUI,
/// or the row's state (starting / ended / failed with its fix).
struct TerminalPane: View {
    let sessionId: UUID
    /// What to do when the session is gone (closed, or moved elsewhere):
    /// the chat window goes back to the transcript, a pop-out window closes.
    var onGone: (() -> Void)? = nil
    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var terminals: TerminalSessionStore
    @EnvironmentObject private var server: ServerManager
    @ObservedObject private var sandbox = AgentSandbox.shared
    @State private var copiedSsh = false

    var body: some View {
        if let session = terminals.sessions.session(sessionId) {
            VStack(spacing: 0) {
                content(session)
                Divider()
                HStack(spacing: 8) {
                    Label(session.workspace, systemImage: "folder")
                        .font(.caption2).foregroundStyle(.secondary)
                        .lineLimit(1).truncationMode(.middle)
                    Spacer()
                    if session.kind == .sandbox { sshConnectRow }
                }
                .padding(.horizontal, 14).padding(.vertical, 8)
            }
        } else {
            Color.clear.onAppear { (onGone ?? appState.showConversation)() }
        }
    }

    @ViewBuilder
    private func content(_ session: TerminalSessionList.Session) -> some View {
        switch session.phase {
        case .preparing:
            notice {
                ProgressView()
                Text("Starting \(session.displayName) session…").font(.headline)
                Text("Boots the guest and prepares configs. First-run installs stream into the terminal once it opens.")
                    .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
            }
        case .live:
            if let handle = terminals.handle(for: session.id) {
                EmbeddedTerminalView(handle: handle)
            }
        case .exited:
            notice {
                Text(terminals.sessions.exitNotice(session.id) ?? "session ended")
                    .font(.callout).foregroundStyle(.secondary)
                Button("Close") { appState.closeTerminal(session.id) }
                    .controlSize(.small)
            }
        case .failed(let message):
            notice {
                Image(systemName: "exclamationmark.triangle").font(.largeTitle).foregroundStyle(.orange)
                Text("The \(session.displayName) session could not start").font(.headline)
                Text(message)
                    .font(.callout).foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 480)
                    .textSelection(.enabled)
                HStack {
                    if let fix = TerminalFailureFix.for(message: message) {
                        Button(fix.title) { apply(fix, to: session.id) }
                            .keyboardShortcut(.defaultAction)
                    }
                    Button("Retry") { terminals.retry(session.id) }
                    Button("Close") { appState.closeTerminal(session.id) }
                }
                .controlSize(.small)
            }
        }
    }

    private func notice<C: View>(@ViewBuilder _ content: () -> C) -> some View {
        VStack(spacing: 12, content: content)
            .padding(40)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// Fixes that resolve right here retry the row in place; the two that
    /// send the user elsewhere (re-pull, Settings) leave it for Retry.
    private func apply(_ fix: TerminalFailureFix, to id: UUID) {
        switch fix {
        case .startServer:
            // The app's ONE start path (loads the selection, or boots headless
            // when the chat model is on another Mac); the row shows "starting"
            // until the server is up, then the launch runs by itself.
            let server = self.server
            terminals.retry(id) {
                await MainActor.run { appState.ensureServerForLan() }
                try? await server.waitUntilRunning(timeout: 240)
            }
        case .enableNetworking:
            appState.serverOptions.sandbox.network = true
            terminals.retry(id)
        case .repullImage: Task.detached { AgentSandbox.shared.repullBaseImage() }
        case .openSettings: appState.showSettings()
        }
    }

    /// "Connect from your terminal" — the copyable ssh one-liner (same
    /// option set as the embedded sessions; pinned by SandboxSSHTests).
    /// Sandbox rows only: a host CLI runs on this Mac, the guest's ssh
    /// port has nothing to do with it.
    @ViewBuilder
    private var sshConnectRow: some View {
        if let cmd = sandbox.sshDisplayCommand {
            HStack(spacing: 6) {
                Text("Connect from your terminal:")
                    .font(.caption2).foregroundStyle(.secondary)
                Text(cmd)
                    .font(.caption2.monospaced())
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                    .frame(maxWidth: 360, alignment: .leading)
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(cmd, forType: .string)
                    copiedSsh = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { copiedSsh = false }
                } label: {
                    Label(copiedSsh ? "Copied" : "Copy", systemImage: copiedSsh ? "checkmark" : "doc.on.doc")
                        .labelStyle(.iconOnly)
                }
                .buttonStyle(.borderless)
                .help("Copy the ssh command — opens another session into the same guest")
            }
        }
    }
}

/// The one-click fix a failed row offers, sniffed off our own preflight /
/// boot messages (the same match the old window's alerts made).
enum TerminalFailureFix {
    case startServer, enableNetworking, repullImage, openSettings

    var title: String {
        switch self {
        case .startServer: return "Start Server"
        case .enableNetworking: return "Turn On Networking"
        case .repullImage: return "Re-pull Image"
        case .openSettings: return "Open Settings"
        }
    }

    static func `for`(message: String) -> TerminalFailureFix? {
        if message.contains("predates ssh support") { return .repullImage }
        if message.contains("Agent Sandbox is off") { return .openSettings }
        if message.contains("networking is off") { return .enableNetworking }
        if message.contains("server isn't running") { return .startServer }
        return nil
    }
}

/// A terminal in its own window. Marks the session as popped out while the
/// window lives, so the sidebar row raises this window instead of trying to
/// show the same terminal view in two places.
struct TerminalWindowView: View {
    let sessionId: UUID
    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var terminals: TerminalSessionStore
    @Environment(\.dismissWindow) private var dismissWindow

    var body: some View {
        TerminalPane(sessionId: sessionId, onGone: { dismissWindow(id: "terminalWindow", value: sessionId) })
            .navigationTitle(terminals.sessions.displayName(sessionId))
            .onAppear { terminals.setInOwnWindow(sessionId, true) }
            .onDisappear { terminals.setInOwnWindow(sessionId, false) }
    }
}
