import Foundation

/// What the chat window's detail column is showing.
///
/// The Model Browser used to be its own `Window`, which meant every route to it
/// was a route OUT of the chat window — and a window the user then had to find
/// their way back from. That is how "Browse all models" on the welcome screen
/// could end with an empty desktop and an app that lives in the menu bar
/// (`WelcomeExit`). It is a MODE of the chat window now: the sidebar keeps the
/// conversation list and gains one row that swaps the detail column, so the way
/// back is always on screen and there is nothing to close.
///
/// Pure so the two rules that make the mode safe — how you get in, and what the
/// gate sheet is allowed to cover — are pinned without rendering a window.
enum ChatWorkspace: Equatable {
    /// The transcript for `AppState.activeChatId`.
    case conversation
    /// The model browser, on one of its sections.
    case models(ModelBrowserSection)
    /// A media generator (image / video / audio / 3D). `GenExperiment` is the
    /// catalogue — the SAME one the tray's tiles and the chat's discovery chips
    /// iterate, so the four can never disagree about what exists or in what
    /// order.
    case create(GenExperiment)
    /// Scheduled/on-demand agent tasks. A window until 2026-08-08; the same
    /// argument as the browser applies — a task is something you set up and then
    /// go back to chatting, not a place you live.
    case tasks
    /// App settings. A window until 2026-08-08 (and still ⌘, from the menu bar,
    /// which macOS expects) — but the sidebar lists it, so it renders here too.
    case settings

    var isModels: Bool {
        if case .models = self { return true }
        return false
    }

    var isTasks: Bool { self == .tasks }

    var isSettings: Bool { self == .settings }

    var isCreate: Bool {
        if case .create = self { return true }
        return false
    }

    /// The transcript — the only mode the chat gate may cover.
    var isConversation: Bool { self == .conversation }

    /// The generator being used, or nil outside create mode.
    var experiment: GenExperiment? {
        if case .create(let exp) = self { return exp }
        return nil
    }

    /// The section being browsed, or nil in conversation mode.
    var section: ModelBrowserSection? {
        if case .models(let section) = self { return section }
        return nil
    }

    /// Where a "browse models" request lands. One section — the friendly front
    /// door — because every entry point that used to open the window opened it
    /// on `.recommended` too, and a request that arrived on a different tab
    /// each time would read as the app forgetting.
    static let defaultEntry: ChatWorkspace = .models(.recommended)

    /// Whether the "you need a model to chat" gate may cover the window.
    ///
    /// The gate is a blocking sheet with exactly one door (Cancel, which closes
    /// the window), and the model browser now lives BEHIND it in the same
    /// window. Presenting it over the models pane would be a locked door
    /// standing in front of its own key: the user is already doing the one
    /// thing the sheet is asking for. It re-presents the moment they go back to
    /// a conversation still having no model — nothing is dismissed, only
    /// deferred.
    /// Keyed on `isConversation`, not on `!isModels`: the gate asks for a CHAT
    /// model, and a media generator needs none of one — blocking the image pane
    /// behind "download a chat model first" would be a demand the pane can't
    /// even use. Every non-transcript mode stands it down.
    static func gateShouldPresent(gateIsBlocking: Bool,
                                  cancelled: Bool,
                                  workspace: ChatWorkspace) -> Bool {
        gateIsBlocking && !cancelled && workspace.isConversation
    }
}
