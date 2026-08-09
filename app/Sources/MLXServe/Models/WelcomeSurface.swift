import Foundation

/// The three places you can drive this app from, as the welcome screen's
/// second panel lists them.
///
/// Two of them ARE the app — the window and the menu-bar icon ship inside the
/// same bundle and cannot be absent — so their trailing state is a fact, not a
/// control. Only the Terminal command is something you install, which is why it
/// is the row with a button and the other two are read as context for it: the
/// panel answers "where can I use this?" and then offers the one piece that
/// isn't already answered.
enum WelcomeSurface: String, CaseIterable, Identifiable, Equatable {
    case app
    case menuBar
    case terminal

    var id: String { rawValue }

    /// Top-to-bottom order: the two you already have, then the one to add.
    static let ordered: [WelcomeSurface] = [.app, .menuBar, .terminal]

    var icon: String {
        switch self {
        case .app:      return "macwindow"
        case .menuBar:  return "menubar.rectangle"
        case .terminal: return "terminal"
        }
    }

    var title: String {
        switch self {
        case .app:      return "App"
        case .menuBar:  return "Menu bar"
        case .terminal: return "Terminal"
        }
    }

    /// What you actually do there. The Terminal row's caption comes from
    /// `CLIInstaller` instead — it depends on where the link would go and
    /// whether that needs a password — so this is the fallback text for it.
    var caption: String {
        switch self {
        case .app:
            return "This window: chat, models, tasks and media generation."
        case .menuBar:
            return "The icon top-right: start the server, switch models, talk."
        case .terminal:
            return "Run mlx-serve from Terminal."
        }
    }

    /// Whether this surface is part of the app itself.
    ///
    /// The window and the menu-bar icon ship in the bundle — there is nothing
    /// to install and nothing that could fail, so the row states it rather than
    /// offering a control that would do nothing. The Terminal command is the
    /// only one whose state has to be probed (`CLIInstaller.status`).
    var shipsWithTheApp: Bool { self != .terminal }
}
