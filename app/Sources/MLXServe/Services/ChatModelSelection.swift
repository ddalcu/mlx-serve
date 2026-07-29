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
