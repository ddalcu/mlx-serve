import Foundation

/// What launch does with the server, and which model — if any — it loads.
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

    /// WHICH model start loads. A mode, stored separately from any path — the
    /// rule is not one of the models, so it must not be spelled as a magic path
    /// value in a field that otherwise holds real ones. A sentinel in a path
    /// field is the same defect in different clothes: every reader has to know
    /// the secret, and one that doesn't treats it as a filename.
    enum Mode: String, CaseIterable, Identifiable, Hashable {
        /// Follow whatever was loaded last, resolved at START time rather than
        /// pinned when the setting was saved. The model you want next is
        /// usually the one you just used.
        case lastUsed
        /// Always this one, whatever has been used since.
        case pinned

        var id: String { rawValue }

        var label: String {
            switch self {
            case .lastUsed: return "Last model used"
            case .pinned:   return "Always this model"
            }
        }

        static let `default`: Mode = .lastUsed
    }

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

    // MARK: - Resolving the choice

    /// The model a start would load right now, or nil for "none — headless".
    ///
    /// The ONE resolution: the launch gate and the Settings readout both ask
    /// this question, so the secondary text under the control cannot promise a
    /// model the gate would decline to load.
    ///
    /// `installedPaths` is the chat-pickable library (`LocalModel.isChatPickable`).
    /// A model that is no longer on disk — uninstalled between launches, or a
    /// last-used one since deleted — resolves to nil. It must not become
    /// `--model <gone>`, an instant FileNotFound, and it must not quietly
    /// promote some other model in its place: a startup that loads a model the
    /// user never chose is worse than one that loads none.
    static func resolved(mode: Mode,
                         pinnedPath: String?,
                         lastUsed: String?,
                         installedPaths: [String]) -> String? {
        let wanted: String?
        switch mode {
        case .lastUsed: wanted = lastUsed
        case .pinned:   wanted = pinnedPath
        }
        guard let wanted, !wanted.isEmpty, installedPaths.contains(wanted) else { return nil }
        return wanted
    }

    /// The pin a FIRST switch to "Always this model" opens on.
    ///
    /// Switching to `.pinned` having never pinned anything would leave the
    /// dropdown matching no row and rendering blank — the dead-control class —
    /// so it is seeded with the answer the other mode was already giving, and
    /// only with the library's first model when that mode had no answer either.
    /// Empty means the Mac has no chat model to pin at all.
    static func seedPin(lastUsed: String?, installedPaths: [String]) -> String {
        resolved(mode: .lastUsed,
                 pinnedPath: nil,
                 lastUsed: lastUsed,
                 installedPaths: installedPaths)
            ?? installedPaths.first
            ?? ""
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

        /// The model this plan puts resident, or nil for none.
        var modelPath: String? {
            guard case .load(let path) = self else { return nil }
            return path
        }
    }

    /// Does the tray's Start button put a model resident?
    ///
    /// "Start Server" is the same sentence the auto-start checkbox makes, so it
    /// answers to the same setting. It used to pass `selectedModelPath` to a
    /// `--model` launch, which is why a tray-started server always had a
    /// checkpoint resident — and why ejecting it brought it back. Either way
    /// the start itself is now headless (`AppState.startServer`); this only
    /// decides whether the selection is hot-loaded straight after.
    static func trayStartLoadsModel(loadModelAtStart: Bool, selectedModelPath: String) -> Bool {
        loadModelAtStart && !selectedModelPath.isEmpty
    }

    /// The model a server started for LAN duty AT LAUNCH should load.
    ///
    /// LAN sharing and discovery live in the server process, so with either on
    /// the app brings a server up even when auto-start is off — and that start
    /// used to pass `selectedModelPath`, which is the eager load this whole
    /// split exists to stop (issue #214). It would have been the back door:
    /// auto-start off, "Load a model at start" off, and login still pays for a
    /// checkpoint because the Mac happens to share models. Empty = headless,
    /// which is all LAN duty needs; only a plan that ASKED for a load names one.
    ///
    /// Only for the launch path. `ensureServerForLan()` called later — from the
    /// chat window's Start button, a LAN model pick, a peer-list refresh — still
    /// loads the selection, because there the user just asked for a server they
    /// intend to chat with.
    static func lanStartPath(plan: Launch) -> String {
        plan.modelPath ?? ""
    }

    static func launch(autoStart: Bool,
                       loadModelAtStart: Bool,
                       mode: Mode,
                       pinnedPath: String?,
                       lastUsed: String?,
                       installedPaths: [String]) -> Launch {
        guard autoStart else { return .doNothing }
        guard loadModelAtStart else { return .headless }
        guard let path = resolved(mode: mode,
                                  pinnedPath: pinnedPath,
                                  lastUsed: lastUsed,
                                  installedPaths: installedPaths) else { return .headless }
        return .load(path: path)
    }
}
