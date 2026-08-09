import Foundation

/// What the chat window's detail column is showing.
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
    /// The agent list + editor. The sidebar row used to open a MENU of agents,
    /// which made "Agents" mean "start a chat as somebody" — a different verb
    /// from every other row in that column, none of which start anything. It is
    /// a pane like Tasks now: the list is column two, the editor is the detail.
    case agents
    /// App settings. A window until 2026-08-08; the sidebar's row and the
    /// menu bar's ⌘, both switch to this mode now (the Window scene is gone).
    case settings

    var isModels: Bool {
        if case .models = self { return true }
        return false
    }

    var isTasks: Bool { self == .tasks }

    var isAgents: Bool { self == .agents }

    /// Whether this mode is laid out as a three-column split (sidebar, list,
    /// detail) rather than sidebar + content. One answer, read by the view: two
    /// separate checks are how one of them ends up on the wrong split.
    var isThreeColumn: Bool { isTasks || isAgents }

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
    static func gateShouldPresent(gateIsBlocking: Bool,
                                  cancelled: Bool,
                                  workspace: ChatWorkspace,
                                  welcomePresented: Bool) -> Bool {
        gateIsBlocking && !cancelled && workspace.isConversation && !welcomePresented
    }
}
