import Foundation

/// One section of the Settings screen — and one row of its sidebar.
///
/// The category is the SINGLE source of truth for a section's identity: the
/// sidebar row and the section header both read `title`/`sidebarLabel` from
/// here, so they can't drift, and a section declared without a category would
/// have no sidebar entry at all (pinned by `testEverySettingsSectionDeclaresACategory`).
///
/// Case order IS render order — the sidebar lists them top to bottom exactly as
/// the form lays them out.
enum SettingsCategory: String, CaseIterable, Identifiable, Hashable {
    case modelFolders
    case server
    case specDecode
    /// The universal knobs AND the MLX-only ones, in one section: two adjacent
    /// cards both called "Performance" was a distinction only the maintainers
    /// cared about. The MLX-only rows inside it still hide when a GGUF/DSV4
    /// model is serving (they'd silently no-op), so the section always has
    /// something in it while never offering a knob that does nothing.
    case performance
    case ggufPerformance
    case ds4
    case requestDefaults
    case voice
    case sandbox
    case messaging
    case updates

    var id: String { rawValue }

    /// The section header, verbatim. Long and explicit on purpose — it names the
    /// engine a knob applies to.
    var title: String {
        switch self {
        case .modelFolders:      return "Model Folders"
        case .server:            return "Server"
        case .performance:       return "Performance"
        case .specDecode:        return "Speculative Decoding (MLX only)"
        case .ggufPerformance:   return "GGUF Performance (llama.cpp)"
        case .ds4:               return "DeepSeek-V4 (ds4 engine)"
        case .requestDefaults:   return "Per-Request Defaults"
        case .voice:             return "Voice"
        case .sandbox:           return "Agent Sandbox"
        case .messaging:         return "Messaging — Telegram bot"
        case .updates:           return "Updates"
        }
    }

    /// The sidebar row. Short enough for the column — the header keeps the long
    /// form (the sidebar is already grouped by engine context).
    var sidebarLabel: String {
        switch self {
        case .specDecode:        return "Speculative Decoding"
        case .ggufPerformance:   return "GGUF Performance"
        case .ds4:               return "DeepSeek-V4"
        case .messaging:         return "Messaging"
        default:                 return title
        }
    }

    /// SF Symbol for the sidebar row (free with the OS — no assets).
    var icon: String {
        switch self {
        case .modelFolders:      return "folder"
        case .server:            return "server.rack"
        case .performance:       return "speedometer"
        case .specDecode:        return "hare"
        case .ggufPerformance:   return "shippingbox"
        case .ds4:               return "cube"
        case .requestDefaults:   return "slider.horizontal.3"
        case .voice:             return "waveform"
        case .sandbox:           return "shield.lefthalf.filled"
        case .messaging:         return "paperplane"
        case .updates:           return "arrow.down.circle"
        }
    }

    /// Categories the form actually renders right now, in render order.
    ///
    /// Engine-specific sections hide themselves when they don't apply (flipping
    /// `--kv-quant` on a GGUF model silently no-ops, so we'd rather not show the
    /// picker than mislead) — the sidebar mirrors that, or it would offer rows
    /// that lead nowhere. `engine == nil` (no model loaded) shows everything so
    /// the user can pre-tune, exactly like `EngineAwareSections`.
    static func visible(engine: ServerEngine?, selfUpdate: Bool) -> [SettingsCategory] {
        allCases.filter { category in
            switch category {
            case .specDecode:      return engine == nil || engine == .mlx
            case .ggufPerformance: return engine == nil || engine == .llama
            case .ds4:             return engine == nil || engine == .dsv4
            case .updates:         return selfUpdate
            // `.performance` is always listed: its universal rows apply to every
            // engine. Only the MLX-only rows INSIDE it come and go.
            default:               return true
            }
        }
    }
}

/// What the Settings sidebar has selected: everything, or one category.
///
/// INVARIANT — a selection and search text never coexist. Typing in the filter
/// forces this back to `.all` (`afterQueryEdit`), and picking a category clears
/// the field. Without that, you could select "Server" while a stale query still
/// narrowed its rows and be shown an empty pane with nothing explaining why.
enum SettingsSelection: Hashable {
    case all
    case category(SettingsCategory)

    /// Should the section for `category` render?
    func shows(_ category: SettingsCategory) -> Bool {
        switch self {
        case .all:               return true
        case .category(let c):   return c == category
        }
    }

    /// The selection after the search field changed. A non-blank query means the
    /// user is searching across everything, so snap back to All. A blank query
    /// changes nothing — clearing the field must not undo the category the user
    /// just picked (picking one is what cleared the field).
    static func afterQueryEdit(query: String, current: SettingsSelection) -> SettingsSelection {
        SettingsSearch.tokens(query).isEmpty ? current : .all
    }

    /// Drop a selection whose category is no longer on offer — e.g. "MLX
    /// Performance" is selected and the user loads a GGUF model. Leaving it
    /// selected would show a section that no longer renders: a blank pane.
    static func reconciled(_ selection: SettingsSelection, visible: [SettingsCategory]) -> SettingsSelection {
        switch selection {
        case .all:
            return .all
        case .category(let c):
            return visible.contains(c) ? selection : .all
        }
    }
}
