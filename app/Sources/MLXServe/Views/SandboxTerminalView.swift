import SwiftUI
import AppKit

/// The Sandbox window (issue #89 follow-up): real terminals into the guest
/// plus the activity transcript.
///
///  * **Terminal** (default): run pi / hermes / plain-shell sessions INSIDE
///    the sandbox VM over ssh — several at once, one tab per session (each is
///    just another connection into the same guest sshd). A copyable ssh
///    command opens the same guest from the user's own terminal.
///  * **Activity**: the unified transcript of agent/user/system commands —
///    exactly the previous Sandbox Terminal UI.
///
/// LIFETIME RULE (the bug that shaped this layout): every live terminal stays
/// MOUNTED in a ZStack and is only hidden via opacity — switching session
/// tabs or flipping Terminal↔Activity must never unmount an
/// `EmbeddedTerminalView`, because dismantling one terminates its ssh (live
/// 2026-07-19: flipping to Activity killed the pi session).
///
/// One shared guest: sessions here and the in-app agent's tools use the same
/// VM. Live sessions PIN it — an IMPLICIT workspace switch (a chat tab's
/// folder) is declined with a clear message, but an EXPLICIT Settings
/// workspace change remounts anyway and this window restarts its sessions in
/// the new share (`workspaceRemounted`). Quitting the app kills the VM
/// (accepted v1).
struct SandboxTerminalView: View {
    @ObservedObject private var sandbox = AgentSandbox.shared
    /// Per-command appends publish through the dedicated store (NOT through
    /// AgentSandbox, which the tray menu observes — churn class).
    @ObservedObject private var transcriptStore = AgentSandbox.shared.transcriptStore
    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var server: ServerManager

    private enum Pane: String, CaseIterable {
        case terminal = "Terminal"
        case activity = "Activity"
    }

    private struct SandboxAlert: Identifiable {
        enum Action { case none, enableNetworking, repullImage, confirmClose(UUID) }
        let id = UUID()
        let message: String
        let action: Action
    }

    /// Per-session live state the pure tabs model deliberately doesn't hold:
    /// the ssh handle + the terminate bridge. A class so identity is stable
    /// across view updates.
    private final class SessionRuntime {
        let cli: AgentSandbox.CliSession
        let controller = EmbeddedTerminalView.EmbeddedTerminalController()
        init(cli: AgentSandbox.CliSession) { self.cli = cli }
    }

    @State private var pane: Pane = .terminal
    @State private var sessions = SandboxSessionTabs()
    @State private var runtimes: [UUID: SessionRuntime] = [:]
    @State private var alert: SandboxAlert?
    @State private var copiedSsh = false

    // Activity-tab input state (unchanged from the pre-tab window).
    @State private var input = ""
    @State private var running = false
    @FocusState private var inputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if sandbox.isEnabled {
                if pane == .terminal, !sessions.tabs.isEmpty {
                    sessionStrip
                    Divider()
                }
                // Both panes stay mounted; opacity switches visibility. See
                // the LIFETIME RULE above.
                ZStack {
                    activityPane
                        .opacity(pane == .activity ? 1 : 0)
                        .allowsHitTesting(pane == .activity)
                    terminalPane
                        .opacity(pane == .terminal ? 1 : 0)
                        .allowsHitTesting(pane == .terminal)
                }
            } else {
                disabledHint
            }
        }
        .frame(minWidth: 640, minHeight: 440)
        .navigationTitle(sessions.windowTitle)
        .onChange(of: pane) { _, now in
            // Focus moves with the pane — an .onAppear would fire while the
            // hidden pane mounts and steal keys from the live terminal.
            inputFocused = (now == .activity)
        }
        // Tray "pi/hermes in Sandbox" shortcut: onAppear covers the click
        // that OPENED this window; onChange covers an already-open window.
        .onAppear { handleAgentLaunchRequest() }
        .onChange(of: appState.pendingSandboxAgentLaunch) { _, _ in
            handleAgentLaunchRequest()
        }
        // Settings workspace pick under live sessions: the guest was already
        // torn down — restart every living tab in the NEW /workspace share.
        .onReceive(NotificationCenter.default.publisher(for: AgentSandbox.workspaceRemounted)) { _ in
            respawnSessionsAfterRemount()
        }
        .alert(item: $alert) { a in
            switch a.action {
            case .none:
                return Alert(title: Text("Sandbox session"), message: Text(a.message))
            case .enableNetworking:
                return Alert(title: Text("Sandbox session"), message: Text(a.message),
                             primaryButton: .default(Text("Turn On Networking")) {
                                 appState.serverOptions.sandbox.network = true
                             },
                             secondaryButton: .cancel())
            case .repullImage:
                return Alert(title: Text("Sandbox image out of date"), message: Text(a.message),
                             primaryButton: .default(Text("Re-pull Image")) {
                                 Task.detached { AgentSandbox.shared.repullBaseImage() }
                             },
                             secondaryButton: .cancel())
            case .confirmClose(let id):
                return Alert(title: Text("End the \(sessions.displayName(id)) session?"),
                             message: Text(a.message),
                             primaryButton: .destructive(Text("End Session")) { closeTab(id) },
                             secondaryButton: .cancel())
            }
        }
    }

    // MARK: Header

    private var header: some View {
        HStack(spacing: 10) {
            Image(systemName: "shippingbox.fill")
                .foregroundStyle(sandbox.guestRunning ? Color.green : .secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text("MLX Sandbox").font(.headline)
                Text(sandbox.guestRunning ? "Guest running — sessions and commands run in an isolated Linux VM"
                                          : "Idle — the guest boots on the first session or command")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            Spacer()
            Picker("", selection: $pane) {
                ForEach(Pane.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
            if sandbox.guestRunning {
                Button {
                    sandbox.teardown()
                } label: {
                    Label("Stop guest", systemImage: "stop.circle")
                }
                .controlSize(.small)
                .help("Shut the guest down. Live sessions end; it re-boots on the next command.")
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
    }

    // MARK: Session strip (Terminal pane)

    private var sessionStrip: some View {
        HStack(spacing: 6) {
            ForEach(sessions.tabs) { tab in
                sessionChip(tab)
            }
            newSessionMenu {
                Image(systemName: "plus")
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .help("New session (pi, hermes, or a plain shell)")
            Spacer()
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
    }

    /// Tray shortcut consumer: focus the newest RUNNING session of the
    /// requested agent, else start one. Clearing the request here makes the
    /// hand-off one-shot (repeat clicks carry a fresh id, so onChange re-fires).
    private func handleAgentLaunchRequest() {
        guard let req = appState.pendingSandboxAgentLaunch else { return }
        appState.pendingSandboxAgentLaunch = nil
        guard let spec = SandboxAgentRegistry.all.first(where: { $0.id == req.agentId }) else { return }
        pane = .terminal
        if let running = sessions.mostRecentActive(label: spec.displayName) {
            sessions.select(running)
        } else {
            startSession(agent: spec)
        }
    }

    /// The tab ✕: instant for exited tabs, confirmed while a session is alive
    /// (the pill's close target is small — a misclick must not kill a TUI).
    /// The confirm rides the single window-level $alert — a second legacy
    /// .alert(item:) modifier is shadowed by an ancestor's, not just by a
    /// same-node one, and silently never presents (live 2026-07-19).
    private func requestCloseTab(_ id: UUID) {
        if sessions.closeNeedsConfirmation(id) {
            alert = SandboxAlert(message: String(localized: "The session running inside the sandbox will be terminated. Files it wrote are kept."),
                                 action: .confirmClose(id))
        } else {
            closeTab(id)
        }
    }

    private func sessionChip(_ tab: SandboxSessionTabs.Tab) -> some View {
        let isSelected = tab.id == sessions.selectedID
        // Two REAL buttons, no gestures: a parent `.onTapGesture` +
        // `.contentShape` swallows plain-style child buttons on macOS — the
        // chip's ✕ silently did nothing (live 2026-07-19). Roomy on purpose:
        // the ✕ sits beside the select target, so a cramped pill turns every
        // tab switch into a session-close near-miss.
        return HStack(spacing: 0) {
            Button {
                sessions.select(tab.id)
            } label: {
                HStack(spacing: 7) {
                    Circle()
                        .fill(chipColor(tab.phase))
                        .frame(width: 6, height: 6)
                    Text(tab.displayName)
                        .font(.callout)
                }
                .padding(.leading, 14).padding(.trailing, 6).padding(.vertical, 6)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Show this session")
            Button {
                requestCloseTab(tab.id)
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 9, weight: .bold))
                    .frame(width: 20, height: 20)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .padding(.trailing, 6)
            .help("Close this session")
        }
        .background(isSelected ? Color.accentColor.opacity(0.22) : Color.primary.opacity(0.06),
                    in: Capsule())
    }

    private func chipColor(_ phase: SandboxSessionTabs.Tab.Phase) -> Color {
        switch phase {
        case .preparing: return .orange
        case .live: return .green
        case .exited: return .secondary.opacity(0.5)
        }
    }

    @ViewBuilder
    private func newSessionMenu<L: View>(@ViewBuilder label: () -> L) -> some View {
        Menu {
            ForEach(SandboxAgentRegistry.all) { spec in
                Button(spec.displayName) { startSession(agent: spec) }
            }
            Divider()
            Button("Plain shell") { startSession(agent: nil) }
        } label: {
            label()
        }
    }

    // MARK: Terminal pane

    @ViewBuilder
    private var terminalPane: some View {
        if sessions.tabs.isEmpty {
            sessionLauncher
        } else {
            VStack(spacing: 0) {
                ZStack {
                    ForEach(sessions.tabs) { tab in
                        sessionContent(tab)
                            .opacity(tab.id == sessions.selectedID ? 1 : 0)
                            .allowsHitTesting(tab.id == sessions.selectedID)
                    }
                }
                Divider()
                // Just the connect row — ending a session is the tab's ✕
                // (confirmed while live), one affordance instead of two.
                HStack(spacing: 8) {
                    sshConnectRow
                    Spacer()
                }
                .padding(.horizontal, 14).padding(.vertical, 8)
            }
        }
    }

    @ViewBuilder
    private func sessionContent(_ tab: SandboxSessionTabs.Tab) -> some View {
        switch tab.phase {
        case .preparing:
            VStack(spacing: 10) {
                ProgressView()
                Text("Starting \(tab.displayName) session…").font(.headline)
                Text("Boots the guest and prepares configs. First-run installs stream into the terminal once it opens.")
                    .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
            }
            .padding(40)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .live:
            if let rt = runtimes[tab.id] {
                EmbeddedTerminalView(executable: SandboxSSH.sshExecutablePath,
                                     args: rt.cli.sshArgs,
                                     controller: rt.controller,
                                     onExit: { code in sessionExited(tab.id, runtime: rt, code: code) })
                    .id(tab.id)
            }
        case .exited:
            VStack(spacing: 12) {
                Text(sessions.exitNotice(tab.id) ?? "session ended")
                    .font(.callout).foregroundStyle(.secondary)
                Button("Close Tab") { closeTab(tab.id) }
                    .controlSize(.small)
            }
            .padding(40)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    /// No sessions yet: the entry point (+ the ssh row while a guest is up,
    /// so a plain `ssh` from Terminal.app needs no session here).
    private var sessionLauncher: some View {
        VStack(spacing: 12) {
            Image(systemName: "terminal").font(.largeTitle).foregroundStyle(.secondary)
            Text("Run an agent inside the sandbox").font(.headline)
            Text("Each session is its own terminal in the isolated Linux VM, talking to this Mac's local model. Run several at once — nothing runs on the host.")
                .font(.caption).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 400)
            newSessionMenu {
                Label("New Session", systemImage: "plus.circle")
            }
            .menuStyle(.button)
            .buttonStyle(.bordered)
            .fixedSize()
            if sandbox.sshPort != nil {
                sshConnectRow.padding(.top, 8)
            }
        }
        .padding(30)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// "Connect from your terminal" — the copyable ssh one-liner (same
    /// option set as the embedded sessions; pinned by SandboxSSHTests).
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

    // MARK: Session lifecycle

    private func startSession(agent: SandboxAgentSpec?) {
        let opts = appState.serverOptions
        // A plain shell needs no server — only agent sessions gate on it.
        let needsServer = agent != nil
        let issues = SandboxCliPreflight.issues(
            sandboxEnabled: sandbox.isEnabled,
            networkOn: opts.sandbox.network,
            serverRunning: needsServer ? server.status == .running : true,
            serverHost: needsServer ? opts.host : "0.0.0.0")
        guard issues.isEmpty else {
            alert = SandboxAlert(message: issues.joined(separator: "\n\n"),
                                 action: opts.sandbox.network ? .none : .enableNetworking)
            return
        }

        let label = agent?.displayName ?? "shell"
        let id = sessions.addPreparing(label: label)
        launchSession(into: id, agent: agent)
    }

    /// Boot + connect a CLI session into an existing (preparing) tab. Shared
    /// by `startSession` and the workspace-remount respawn.
    private func launchSession(into id: UUID, agent: SandboxAgentSpec?) {
        // Chat chokepoint rule: the model the sandboxed agent targets is
        // `server.chatModelId` (LAN picks win), budgets derive from the
        // advertised context — never hardcoded.
        let model = server.chatModelId
        let port = server.port
        let budget = AgentBudget.forServerContext(server.chatModelInfo?.contextLength)
        let key = appState.serverOptions.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        // The in-agent /model switch list: the chat-capable registry snapshot
        // (hermes bakes it; pi's extension re-fetches live in-guest).
        let entries = AgentModelEntry.chatEntries(from: server.allModels)
        Task {
            do {
                let cli = try await sandbox.startCliSession(
                    agent: agent, model: model, serverPort: port,
                    budget: budget, apiKey: key.isEmpty ? nil : key,
                    entries: entries)
                guard sessions.tabs.contains(where: { $0.id == id }) else {
                    // Tab closed while the guest was booting — balance the pin.
                    sandbox.endCliSession(cli)
                    return
                }
                runtimes[id] = SessionRuntime(cli: cli)
                sessions.markLive(id)
            } catch {
                sessions.close(id)
                let message = (error as? AgentSandbox.SandboxError)?.message ?? "\(error)"
                alert = SandboxAlert(message: message,
                                     action: message.contains("predates ssh support") ? .repullImage : .none)
            }
        }
    }

    /// The default workspace changed under live sessions: the guest was torn
    /// down (`noteWorkspaceChanged(restartPinnedSessions: true)`) — restart
    /// every living tab's CLI so it lands in the NEW /workspace share.
    /// In place: tab identity and display name survive; conversation state
    /// inside the CLIs does not (the VM it lived in is gone).
    private func respawnSessionsAfterRemount() {
        for tab in sessions.tabs where tab.phase == .live || tab.phase == .preparing {
            if let rt = runtimes.removeValue(forKey: tab.id) {
                rt.controller.terminate()
                sandbox.endCliSession(rt.cli)
            }
            sessions.restart(tab.id)
            launchSession(into: tab.id,
                          agent: SandboxAgentRegistry.all.first { $0.displayName == tab.label })
        }
    }

    private func sessionExited(_ id: UUID, runtime: SessionRuntime, code: Int32?) {
        // A stale exit from a session replaced by the remount respawn must not
        // kill the replacement — only the CURRENT runtime's exit counts. (A
        // closed tab's late exit is also caught here: closeTab already removed
        // the runtime.)
        guard runtimes[id] === runtime else { return }
        runtimes.removeValue(forKey: id)
        sandbox.endCliSession(runtime.cli)
        sessions.markExited(id, exitCode: code)
    }

    private func closeTab(_ id: UUID) {
        if let rt = runtimes.removeValue(forKey: id) {
            rt.controller.terminate()
            sandbox.endCliSession(rt.cli)
        }
        sessions.close(id)
    }

    // MARK: Activity pane (the previous Sandbox Terminal UI, verbatim)

    private var activityPane: some View {
        VStack(spacing: 0) {
            transcript
            Divider()
            inputBar
        }
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(transcriptStore.entries) { entry in
                        entryView(entry).id(entry.id)
                    }
                    Color.clear.frame(height: 1).id(bottomID)
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color(NSColor.textBackgroundColor))
            .onChange(of: transcriptStore.entries.count) { _, _ in
                withAnimation(.easeOut(duration: 0.15)) { proxy.scrollTo(bottomID, anchor: .bottom) }
            }
        }
    }

    private let bottomID = "sandbox-terminal-bottom"

    @ViewBuilder
    private func entryView(_ e: AgentSandbox.Entry) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            if e.source == .system {
                Text(e.output)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            } else {
                HStack(spacing: 6) {
                    Text(promptLabel(e.source))
                        .font(.caption.monospaced().weight(.semibold))
                        .foregroundStyle(e.source == .user ? Color.green : Color.blue)
                    Text(e.command)
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                }
                let body = e.output.trimmingCharacters(in: .newlines)
                if !body.isEmpty {
                    Text(body)
                        .font(.caption.monospaced())
                        .foregroundStyle(.primary.opacity(0.9))
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if e.timedOut {
                    Text("[timed out]").font(.caption2.monospaced()).foregroundStyle(.orange)
                } else if let code = e.exitCode, code != 0 {
                    Text("[exit \(code)]").font(.caption2.monospaced()).foregroundStyle(.red)
                }
            }
        }
    }

    private func promptLabel(_ s: AgentSandbox.Entry.Source) -> String {
        switch s {
        case .agent: return "agent $"
        case .user:  return "you $"
        case .system: return "•"
        }
    }

    private var inputBar: some View {
        HStack(spacing: 8) {
            Text("$").font(.body.monospaced()).foregroundStyle(.secondary)
            TextField("Run a command in the sandbox…", text: $input)
                .textFieldStyle(.plain)
                .font(.body.monospaced())
                .focused($inputFocused)
                .onSubmit(run)
                .disabled(running)
            if running {
                ProgressView().controlSize(.small)
            } else {
                Button("Run", action: run)
                    .disabled(input.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
    }

    private func run() {
        let cmd = input
        guard !cmd.trimmingCharacters(in: .whitespaces).isEmpty, !running else { return }
        input = ""
        running = true
        Task {
            await sandbox.runUserCommand(cmd)
            running = false
            inputFocused = true
        }
    }

    // MARK: Disabled

    private var disabledHint: some View {
        VStack(spacing: 10) {
            Image(systemName: "shippingbox").font(.largeTitle).foregroundStyle(.secondary)
            Text("The Agent Sandbox is off.").font(.headline)
            Text("Turn it on in Settings → Agent Sandbox to run agents and commands in an isolated Linux VM.")
                .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
