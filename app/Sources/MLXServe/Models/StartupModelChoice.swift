import Foundation

/// What launch does with the server, and where "the last model used" comes
/// from.
///
/// Startup is TWO decisions, not one: whether the server comes up, and whether
/// a checkpoint goes resident before anybody has asked for one. They used to be
/// the same checkbox — "Auto-start on launch" passed `--model`, which the server
/// treats as an EAGER, BLOCKING load — so ticking a box labelled *start* read
/// tens of gigabytes off disk at login, with nothing in the UI saying so
/// (issue #214). The server has always been able to start without a model
/// (`runHeadlessServe` → `no_initial_load = true`); only the app never asked it
/// to.
///
/// Pure and static on purpose: the gate is one branch in `AppState.init` that
/// nobody can watch run, and its previous shape shipped a multi-gigabyte load
/// behind a checkbox that promised a server.
enum StartupModelChoice {

    /// The startup dropdown's first entry: load whatever loaded LAST, resolved
    /// at start time rather than at the moment the setting was saved. The model
    /// you want next is usually the one you just used, and pinning a path when
    /// the setting was written would freeze that answer forever.
    ///
    /// Empty, so it is also the default for a key that was never set — a fresh
    /// install has no last model, and "load the last one" then correctly
    /// resolves to loading nothing.
    static let lastUsedTag = ""

    // MARK: - Last model used

    private static let lastUsedKey = "lastLoadedModelPath"

    /// Record a chat model the server FINISHED loading.
    ///
    /// Called only from confirmed loads — a load that was *requested* and then
    /// failed is not a model that was used, and writing it here would replay
    /// the same failure on the next launch. Absolute paths only: a registry id
    /// is a directory basename (for a Hugging Face snapshot, a commit hash) and
    /// a LAN id names another Mac's model, so neither is something we can hand
    /// back to `--model`.
    static func recordLoaded(path: String, defaults: UserDefaults = .standard) {
        guard path.hasPrefix("/") else { return }
        defaults.set(path, forKey: lastUsedKey)
    }

    /// The last confirmed load, or nil when there has never been one.
    static func lastUsed(defaults: UserDefaults = .standard) -> String? {
        let stored = defaults.string(forKey: lastUsedKey) ?? ""
        return stored.isEmpty ? nil : stored
    }

    // MARK: - The launch gate

    /// What `AppState.init` should do with the server.
    enum Launch: Equatable {
        /// Auto-start is off — the user starts the server themselves.
        case doNothing
        /// Server up, no model resident. Models load on demand (`/v1/load-model`,
        /// or the first chat turn via `ServerManager.ensureDefaultChatModel`).
        case headless
        /// Server up with `--model <path>` — the eager load, now only ever
        /// reached because the user explicitly asked for it.
        case load(path: String)
    }

    /// `choice` is the saved dropdown pick: `lastUsedTag` for "Last model used",
    /// otherwise a model's absolute path. `installedPaths` is the chat-pickable
    /// library (`LocalModel.isChatPickable`).
    ///
    /// A pick that is no longer on disk — uninstalled between launches, or a
    /// last-used model that has since been deleted — starts the server HEADLESS.
    /// It must not send `--model <gone>` into an instant FileNotFound, and it
    /// must not quietly promote some other model in its place: a startup that
    /// loads a model the user never chose is worse than one that loads none.
    static func launch(autoStart: Bool,
                       loadModelAtStart: Bool,
                       choice: String,
                       lastUsed: String?,
                       installedPaths: [String]) -> Launch {
        guard autoStart else { return .doNothing }
        guard loadModelAtStart else { return .headless }
        let wanted = choice == lastUsedTag ? lastUsed : choice
        guard let wanted, installedPaths.contains(wanted) else { return .headless }
        return .load(path: wanted)
    }
}
