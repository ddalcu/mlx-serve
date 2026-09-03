import Foundation
import Combine

/// App-level owner of the sandbox terminal sessions (pi / hermes / shell over
/// ssh into the guest VM). Lives on `AppState`, so a session survives the
/// chat window closing — only app quit (which kills the VM) ends one.
///
/// The list model (`TerminalSessionList`) is pure; this class holds what it
/// deliberately doesn't: the ssh handle + the process-owning terminal view.
@MainActor
final class TerminalSessionStore: ObservableObject {

    /// Per-session live state. A class so identity is stable — a stale exit
    /// from a session replaced by a remount respawn must not kill the
    /// replacement (`sessionExited` compares identity).
    final class Runtime {
        /// nil for a host CLI (no guest pin to balance).
        let cli: AgentSandbox.CliSession?
        let handle: EmbeddedTerminalView.Handle
        init(cli: AgentSandbox.CliSession?, handle: EmbeddedTerminalView.Handle) {
            self.cli = cli; self.handle = handle
        }
    }

    @Published private(set) var sessions = TerminalSessionList()
    private var runtimes: [UUID: Runtime] = [:]
    /// The host CLI behind a `.host` row, for retries.
    private var hostSpecs: [UUID: LauncherCLI] = [:]
    private let server: ServerManager
    private let options: () -> ServerOptions
    private let sandbox = AgentSandbox.shared
    private var observers: [AnyCancellable] = []

    init(server: ServerManager, options: @escaping () -> ServerOptions) {
        self.server = server
        self.options = options
        // Settings workspace pick under live sessions: the guest was already
        // torn down — restart every living session in the new guest.
        observers.append(NotificationCenter.default
            .publisher(for: AgentSandbox.workspaceRemounted)
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in self?.respawnAfterRemount() })
        // Settings ▸ Interface changed the default theme/background: every
        // session on the default repaints live.
        observers.append(NotificationCenter.default
            .publisher(for: UserDefaults.didChangeNotification)
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in self?.reapplyThemes() })
    }

    func rename(_ id: UUID, to name: String) { sessions.rename(id, to: name) }

    /// "Move Tab to New Window" / the window closing again.
    func setInOwnWindow(_ id: UUID, _ flag: Bool) {
        sessions.setInOwnWindow(id, flag)
    }

    /// A session's own theme pick (nil = back to the Settings default).
    func setTheme(_ id: UUID, themeId: String?) {
        sessions.setTheme(id, themeId: themeId)
        applyTheme(to: id)
    }

    private func applyTheme(to id: UUID) {
        guard let handle = runtimes[id]?.handle else { return }
        let r = TerminalTheme.resolve(sessionThemeId: sessions.session(id)?.themeId)
        handle.apply(theme: r.theme, background: r.background)
    }

    private func reapplyThemes() {
        for id in runtimes.keys where sessions.session(id)?.themeId == nil { applyTheme(to: id) }
    }

    func handle(for id: UUID) -> EmbeddedTerminalView.Handle? { runtimes[id]?.handle }

    /// Add a row and start the session into it. A preflight failure is a
    /// `.failed` row with the message (and the fix, rendered by the pane).
    @discardableResult
    func start(agent: SandboxAgentSpec?, workspace: String) -> UUID {
        let label = agent?.displayName ?? "shell"
        let id = sessions.addPreparing(label: label, agentId: agent?.id, workspace: workspace)
        preflightAndLaunch(into: id, agent: agent, workspace: workspace)
        return id
    }

    /// A host CLI (Claude Code, opencode, …) in a terminal row: the launcher's
    /// script under a login+interactive zsh, on this Mac.
    @discardableResult
    func startHost(cli: LauncherCLI, workspace: String) -> UUID {
        let id = sessions.addPreparing(label: cli.displayName, agentId: nil, workspace: workspace,
                                       kind: .host)
        hostSpecs[id] = cli
        preflightAndLaunch(into: id, agent: nil, workspace: workspace)
        return id
    }

    /// Try a failed row again in place, after `prepare` (start the server,
    /// flip networking on) has run. The row reads "starting" meanwhile.
    func retry(_ id: UUID, prepare: @escaping () async -> Void = {}) {
        guard let s = sessions.session(id), case .failed = s.phase else { return }
        sessions.retry(id)
        Task {
            await prepare()
            guard sessions.session(id)?.phase == .preparing else { return }
            preflightAndLaunch(into: id, agent: SandboxAgentRegistry.all.first { $0.id == s.agentId },
                               workspace: s.workspace)
        }
    }

    private func preflightAndLaunch(into id: UUID, agent: SandboxAgentSpec?, workspace: String) {
        if let cli = hostSpecs[id] {
            // Same wording as the sandbox preflight, so the row offers the
            // same Start Server fix.
            guard server.status == .running else {
                sessions.markFailed(id, message: "the server isn't running — load a model first; \(cli.displayName) talks to it")
                return
            }
            let cmd = CLILauncher.launchCommand(
                cli, baseURL: server.baseURL, servedModelId: server.chatModelId ?? "mlx-serve",
                budget: AgentBudget.forServerContext(server.chatModelInfo?.contextLength),
                entries: AgentModelEntry.chatEntries(from: server.allModels),
                workingDirectory: workspace)
            install(handle: makeHandle(id: id, executable: cmd.executable, args: cmd.args), cli: nil, for: id)
            return
        }
        let opts = options()
        // A plain shell needs no server — only agent sessions gate on it.
        let needsServer = agent != nil
        let issues = SandboxCliPreflight.issues(
            sandboxEnabled: sandbox.isEnabled,
            networkOn: opts.sandbox.network,
            serverRunning: needsServer ? server.status == .running : true,
            serverHost: needsServer ? opts.host : "0.0.0.0")
        if issues.isEmpty {
            launch(into: id, agent: agent, workspace: workspace)
        } else {
            sessions.markFailed(id, message: issues.joined(separator: "\n\n"))
        }
    }

    /// Boot + connect a CLI session into an existing (preparing) row. Shared
    /// by `start` and the workspace-remount respawn.
    private func launch(into id: UUID, agent: SandboxAgentSpec?, workspace: String) {
        // Chat chokepoint rule: the model the sandboxed agent targets is
        // `server.chatModelId` (LAN picks win), budgets derive from the
        // advertised context — never hardcoded.
        // The "mlx-serve" alias resolves to the default model server-side —
        // the same fallback the host launcher and the tray use. Requiring a
        // resolved id here refused sessions ("no model is loaded") on a
        // running server whose resident model had no chat id yet.
        let model = server.chatModelId ?? "mlx-serve"
        let port = server.port
        let budget = AgentBudget.forServerContext(server.chatModelInfo?.contextLength)
        let key = options().apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        let entries = AgentModelEntry.chatEntries(from: server.allModels)
        Task {
            do {
                let cli = try await sandbox.startCliSession(
                    agent: agent, model: model, serverPort: port,
                    budget: budget, apiKey: key.isEmpty ? nil : key,
                    entries: entries, workingDirectory: workspace)
                guard sessions.session(id)?.phase == .preparing else {
                    // Closed (or replaced) while the guest was booting — balance the pin.
                    sandbox.endCliSession(cli)
                    return
                }
                install(handle: makeHandle(id: id, executable: SandboxSSH.sshExecutablePath,
                                           args: cli.sshArgs),
                        cli: cli, for: id)
            } catch {
                let message = (error as? AgentSandbox.SandboxError)?.message ?? "\(error)"
                sessions.markFailed(id, message: message)
            }
        }
    }

    /// The terminal + process for a row. The exit callback resolves the
    /// runtime lazily (it is registered right after), so a stale exit from a
    /// replaced runtime can be told apart by identity in `sessionExited`.
    private func makeHandle(id: UUID, executable: String, args: [String]) -> EmbeddedTerminalView.Handle {
        var made: EmbeddedTerminalView.Handle?
        let handle = EmbeddedTerminalView.Handle(executable: executable, args: args) { [weak self] code in
            guard let self, let made, let runtime = self.runtimes[id], runtime.handle === made else { return }
            self.sessionExited(id, runtime: runtime, code: code)
        }
        made = handle
        return handle
    }

    private func install(handle: EmbeddedTerminalView.Handle, cli: AgentSandbox.CliSession?, for id: UUID) {
        runtimes[id] = Runtime(cli: cli, handle: handle)
        applyTheme(to: id)
        sessions.markLive(id)
    }

    private func respawnAfterRemount() {
        for s in sessions.sessions where s.isActive && s.kind == .sandbox {
            endRuntime(s.id)
            sessions.restart(s.id)
            launch(into: s.id,
                   agent: SandboxAgentRegistry.all.first { $0.id == s.agentId },
                   workspace: s.workspace)
        }
    }

    private func sessionExited(_ id: UUID, runtime: Runtime, code: Int32?) {
        // Only the CURRENT runtime's exit counts (a closed row's late exit,
        // or a replaced one's, is caught here).
        guard runtimes[id] === runtime else { return }
        runtimes.removeValue(forKey: id)
        if let cli = runtime.cli { sandbox.endCliSession(cli) }
        sessions.markExited(id, exitCode: code)
    }

    private func endRuntime(_ id: UUID) {
        if let rt = runtimes.removeValue(forKey: id) {
            rt.handle.terminate()
            if let cli = rt.cli { sandbox.endCliSession(cli) }
        }
    }

    /// Remove the row; terminates a live session. Confirmation is the
    /// caller's job (`sessions.closeNeedsConfirmation`).
    func close(_ id: UUID) {
        endRuntime(id)
        hostSpecs.removeValue(forKey: id)
        sessions.close(id)
    }
}
