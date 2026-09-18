import Foundation

/// What launch does with the server, and which model, if any, it loads. Two
/// decisions: `--model` is an eager, blocking load, so auto-start alone comes
/// up headless and only "Preload the model when the server starts" adds one.
enum StartupModelChoice {

    /// Which model start loads, stored apart from any path so a rule is never
    /// spelled as a sentinel inside a path field.
    enum Mode: String, CaseIterable, Identifiable, Hashable {
        /// Whatever was loaded last, resolved at start time.
        case lastUsed
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

    /// Record a chat model the server FINISHED loading; a failed load recorded here
    /// would fail again at every launch. Absolute paths only: registry and LAN ids
    /// cannot go back through `--model`.
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

    /// The model a start would load now, or nil for headless; read by both the
    /// launch gate and the Settings readout. A model no longer in `installedPaths`
    /// resolves to nil: never `--model <gone>`, never a substitute nobody picked.
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

    /// The pin a first switch to `.pinned` opens on: what `.lastUsed` resolves to,
    /// else the library's first model, else empty.
    static func seedPin(lastUsed: String?, installedPaths: [String]) -> String {
        resolved(mode: .lastUsed,
                 pinnedPath: nil,
                 lastUsed: lastUsed,
                 installedPaths: installedPaths)
            ?? installedPaths.first
            ?? ""
    }

    // MARK: - The launch gate

    enum Launch: Equatable {
        case doNothing
        /// Server up, nothing resident; models load on demand.
        case headless
        /// Server up with `--model <path>`.
        case load(path: String)

        var modelPath: String? {
            guard case .load(let path) = self else { return nil }
            return path
        }
    }

    /// Whether the tray's Start hot-loads the selection after its headless start.
    static func trayStartLoadsModel(loadModelAtStart: Bool, selectedModelPath: String) -> Bool {
        loadModelAtStart && !selectedModelPath.isEmpty
    }

    /// What a server started for LAN duty at launch loads: the plan's model, or
    /// empty (headless), so sharing models cannot bring back the login load.
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
