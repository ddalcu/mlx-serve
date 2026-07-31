import Foundation

/// What a chat model picker shows, and what picking a row means.
///
/// There are two pickers now — the menu-bar tray and the chat window's toolbar —
/// so this is deliberately ONE definition rather than a copy in each. A
/// per-surface version is how a surface ends up ignoring a LAN selection, the
/// same class as the rule that a chat surface routes through
/// `server.chatModelId` instead of reading the local model name for itself.
///
/// Rows are tagged by string because SwiftUI pickers need a single tag type for
/// two different kinds of row: a local checkpoint (its path) and a model shared
/// by another Mac (its `id@peer`, prefixed so it can never collide with a path).
enum ChatModelSelection {

    enum Action: Equatable {
        case selectLan(String)
        case selectLocal(String)
    }

    private static let lanPrefix = "lan:"

    /// The tag the picker should show as selected. A LAN model wins: the local
    /// `selectedModelPath` is still set underneath while chatting over the
    /// network, and ticking it would point at a model that isn't answering.
    static func tag(localPath: String, lanChatModelId: String?) -> String {
        if let lanChatModelId { return lanPrefix + lanChatModelId }
        return localPath
    }

    /// Decode a picked tag. Only the PREFIX marks a network row, so a local
    /// folder whose path happens to contain "lan:" still loads locally.
    static func action(for tag: String) -> Action {
        guard tag.hasPrefix(lanPrefix) else { return .selectLocal(tag) }
        return .selectLan(String(tag.dropFirst(lanPrefix.count)))
    }
}

/// The "Start" button that appears beside the chat model picker while the
/// server is down.
///
/// The chat window is where you FIND OUT the server is down — you type, and
/// nothing answers — but until now the only thing it said about it was the
/// pill's status dot going grey, and the fix lived in the menu-bar tray. So the
/// recovery is offered where the problem shows up.
///
/// Red, and only red in the one state you can act on: `.starting` is the same
/// control still reporting, not a second thing to press. Hidden when there is
/// nothing to start, because a permanently disabled red button that never
/// explains itself is the dead-control class — the pill already says "Select a
/// model" in that case.
enum ChatServerStartControl: Equatable {
    case hidden
    case start
    case starting

    /// - Parameter hasStartableModel: a local checkpoint is selected, or a LAN
    ///   model is (which boots the server headless so the proxy can run).
    static func resolve(status: ServerStatus, hasStartableModel: Bool) -> ChatServerStartControl {
        switch status {
        case .running:  return .hidden
        // Always shown: this state is only reachable because something already
        // started the server, and hiding it mid-load would blink the toolbar
        // for the tens of seconds a model takes to load.
        case .starting: return .starting
        case .stopped, .error:
            return hasStartableModel ? .start : .hidden
        }
    }

    var title: String {
        switch self {
        case .hidden:   return ""
        case .start:    return "Start"
        case .starting: return "Starting…"
        }
    }

    /// Red is the ATTENTION state, so it belongs to the one case that is both
    /// actionable and a problem. A red spinner would be shouting about work
    /// that is already going fine.
    var isRed: Bool { self == .start }

    var isEnabled: Bool { self == .start }
}
