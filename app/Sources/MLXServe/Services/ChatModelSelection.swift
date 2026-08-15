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

    /// Shown when this Mac has nothing picked and nothing is answering.
    static let noModelPlaceholder = "Select a model"

    /// What the picker's label says, and whether a switch to it is still
    /// loading.
    ///
    /// Two things the naive answer (`lan ?? resident ?? picked`) gets wrong:
    ///
    /// **A server id is not a name.** A model in the Hugging Face cache lives at
    /// `models--<org>--<repo>/snapshots/<commit>/`, and the registry keys a
    /// register-by-path entry by its DIRECTORY BASENAME — so the server calls it
    /// `9f0ea3c1d2` and the pill rendered a hex string while the tray, which
    /// reads our own scan, showed the repo id. Where the resident model IS the
    /// one we picked, our label for it wins; where it isn't, the server's id
    /// stays, because borrowing the picked model's label would name the wrong
    /// model.
    ///
    /// **A hot switch keeps answering on the OLD model** until the new one is
    /// resident (the process never restarts, so `status` stays `.running` and
    /// `chatModelInfo` reports the previous model for the whole load). On a big
    /// checkpoint that is a minute of the pill naming a model the user did not
    /// pick, beside a green dot — while the menu's checkmark has already moved.
    /// An in-flight load therefore names its OWN model and reports loading.
    static func pillState(lanChatModelId: String?,
                          residentName: String?,
                          loadingPath: String?,
                          selectedPath: String,
                          models: [LocalModel]) -> ChatModelPillState {
        // A LAN model is answering from another Mac and loads nothing here, so
        // it is never in flight — see `tag`, same reason it wins there.
        if let lanChatModelId, !lanChatModelId.isEmpty {
            return ChatModelPillState(name: lanChatModelId, isLoading: false)
        }
        func label(forPath path: String) -> String? {
            models.first { $0.path == path }?.displayLabel
        }
        if let loadingPath, !loadingPath.isEmpty {
            // No label for the path (turned non-pickable mid-session, custom
            // dir): the dir basename still names the model in flight, where
            // the resident name would put the WRONG model beside the spinner.
            return ChatModelPillState(name: label(forPath: loadingPath) ?? (loadingPath as NSString).lastPathComponent,
                                      isLoading: true)
        }
        if let name = resident(residentName) {
            let picked = models.first { $0.path == selectedPath }
            return ChatModelPillState(name: isTheSameModel(picked, asResident: name) ? picked!.displayLabel : name,
                                      isLoading: false)
        }
        return ChatModelPillState(name: label(forPath: selectedPath) ?? noModelPlaceholder, isLoading: false)
    }

    private static func resident(_ name: String?) -> String? {
        guard let name, !name.isEmpty else { return nil }
        return name
    }

    /// Whether the server's id for the resident model names the model we
    /// picked. Registry ids are either the discovered `org/name` or, for a
    /// register-by-path load, the directory basename — which for a Hugging Face
    /// snapshot is the commit hash. Anything else is a different model.
    private static func isTheSameModel(_ picked: LocalModel?, asResident name: String) -> Bool {
        guard let picked else { return false }
        return picked.name == name || (picked.path as NSString).lastPathComponent == name
    }
}

/// What the composer's model pill draws: the readable name, and whether a
/// switch to it is still loading.
struct ChatModelPillState: Equatable {
    let name: String
    let isLoading: Bool
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
