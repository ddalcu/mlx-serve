import Foundation

/// What switching agents does to the sandbox share.
///
/// Agents each carry their own working directory, so a switch can move the
/// folder out from under a live sandbox CLI session. That makes it an EXPLICIT
/// re-anchor — the Settings-pick path, which remounts and restarts the pinned
/// sessions in place — not the implicit chat-tab pick the pin declines.
///
/// What it must NOT do is write the user's global default workspace: that
/// retargets every session still sitting on the old default, which an agent pick
/// has no business doing and nothing in the UI would show. Pinned by a source
/// audit in `AgentWorkspaceSwitchTests`.
enum AgentWorkspaceSwitch {

    enum Action: Equatable {
        case nothing
        case remount(path: String, restartPinnedSessions: Bool)
    }

    static func decide(from current: String?, to next: String?) -> Action {
        guard let next, !next.isEmpty, next != current else { return .nothing }
        return .remount(path: next, restartPinnedSessions: true)
    }
}
