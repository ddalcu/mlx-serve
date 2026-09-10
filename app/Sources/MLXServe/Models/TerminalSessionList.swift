import Foundation

/// The sandbox terminal rows in the chat sidebar. Pure — pinned by
/// TerminalSessionListTests.
///
/// Several agent sessions run concurrently: each is one more ssh connection
/// into the SAME guest sshd (dropbear multiplexes; one mirror port serves
/// them all). The model owns ordering, phases and stable display names. Which
/// row is SHOWING is `ChatWorkspace.terminal(id)`; the processes live in
/// `TerminalSessionStore`.
struct TerminalSessionList: Equatable {

    struct Session: Identifiable, Equatable {
        /// Where the process runs: ssh into the guest VM, or a host CLI
        /// (Claude Code, opencode, …) spawned on this Mac.
        enum Kind: Equatable { case sandbox, host }

        enum Phase: Equatable {
            case preparing
            case live
            case exited(Int32?)
            /// Preflight or boot failed; the row shows the message + its fix.
            case failed(String)
        }
        let id: UUID
        let label: String        // registry label: "pi" / "hermes" / "shell"
        let autoName: String     // "pi", "pi 2" — assigned at creation, never renumbered
        /// A user rename ("Rename…" on the row); nil = `autoName`.
        var customName: String? = nil
        var displayName: String { customName ?? autoName }
        let agentId: String?     // nil = plain shell
        let workspace: String    // host folder, hot-mounted in the guest
        let createdAt: Date
        var kind: Kind = .sandbox
        var phase: Phase
        /// A per-session `TerminalTheme` pick; nil = the Settings default.
        var themeId: String? = nil
        /// Shown in its own window ("Move Tab to New Window") rather than the
        /// chat window's detail column. The terminal view has ONE parent, so
        /// the row raises that window instead of showing the pane.
        var isInOwnWindow = false

        var isActive: Bool {
            switch phase {
            case .preparing, .live: return true
            case .exited, .failed: return false
            }
        }
    }

    private(set) var sessions: [Session] = []
    /// Per-label session ordinals. Monotonic — a closed "pi" never frees its
    /// number, so "pi 2" can't silently become "pi" mid-session.
    private var ordinals: [String: Int] = [:]

    func session(_ id: UUID) -> Session? { sessions.first { $0.id == id } }

    @discardableResult
    mutating func addPreparing(label: String, agentId: String?, workspace: String,
                               kind: Session.Kind = .sandbox) -> UUID {
        let n = (ordinals[label] ?? 0) + 1
        ordinals[label] = n
        let s = Session(id: UUID(), label: label,
                        autoName: n == 1 ? label : "\(label) \(n)",
                        agentId: agentId, workspace: workspace, createdAt: Date(),
                        kind: kind, phase: .preparing)
        sessions.append(s)
        return s.id
    }

    mutating func setTheme(_ id: UUID, themeId: String?) {
        guard let i = sessions.firstIndex(where: { $0.id == id }) else { return }
        sessions[i].themeId = themeId
    }

    /// Blank restores the auto name.
    mutating func rename(_ id: UUID, to name: String) {
        guard let i = sessions.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        sessions[i].customName = trimmed.isEmpty ? nil : trimmed
    }

    mutating func setInOwnWindow(_ id: UUID, _ flag: Bool) {
        guard let i = sessions.firstIndex(where: { $0.id == id }) else { return }
        sessions[i].isInOwnWindow = flag
    }

    mutating func markLive(_ id: UUID) { setPhase(id, .live) }

    mutating func markExited(_ id: UUID, exitCode: Int32?) { setPhase(id, .exited(exitCode)) }

    mutating func markFailed(_ id: UUID, message: String) { setPhase(id, .failed(message)) }

    private mutating func setPhase(_ id: UUID, _ phase: Session.Phase) {
        guard let i = sessions.firstIndex(where: { $0.id == id }) else { return }
        sessions[i].phase = phase
    }

    /// A workspace remount restarts a living session IN PLACE: same row, same
    /// display name. Exited/failed rows stay as they are.
    mutating func restart(_ id: UUID) {
        guard let i = sessions.firstIndex(where: { $0.id == id }), sessions[i].isActive else { return }
        sessions[i].phase = .preparing
    }

    /// A failed row tries again in place (the fix button / Retry).
    mutating func retry(_ id: UUID) {
        guard let i = sessions.firstIndex(where: { $0.id == id }),
              case .failed = sessions[i].phase else { return }
        sessions[i].phase = .preparing
    }

    mutating func close(_ id: UUID) {
        sessions.removeAll { $0.id == id }
    }

    func displayName(_ id: UUID) -> String { session(id)?.displayName ?? "" }

    /// The newest preparing/live session for an agent label.
    func mostRecentActive(label: String) -> UUID? {
        sessions.last { $0.label == label && $0.isActive }?.id
    }

    /// True when closing would kill something — a preparing or live session.
    func closeNeedsConfirmation(_ id: UUID) -> Bool {
        session(id)?.isActive ?? false
    }

    /// Honest per-session exit notice; nil unless the session actually exited.
    func exitNotice(_ id: UUID) -> String? {
        guard let s = session(id), case .exited(let code) = s.phase else { return nil }
        if let code, code != 0 { return "\(s.displayName) session ended (exit \(code))" }
        return "\(s.displayName) session ended"
    }
}
