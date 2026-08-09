import Foundation

/// The three places you can drive this app from, as the welcome screen's second
/// panel lists them.
enum WelcomeSurface: String, CaseIterable, Identifiable {
    case app
    case menuBar
    case terminal

    var id: String { rawValue }

    /// The two you already have, then the one to add.
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

    /// Nil for Terminal: its line is live (where the link would go, whether a
    /// password is needed, any failure) and belongs to `CLIInstaller`, so a
    /// constant here would be a second copy that silently stops matching.
    var caption: String? {
        switch self {
        case .app:      return "This window: chat, models, tasks and media generation."
        case .menuBar:  return "The icon top-right: start the server, switch models, talk."
        case .terminal: return nil
        }
    }

    /// Part of the app itself — nothing to install, nothing that could fail.
    var shipsWithTheApp: Bool { self != .terminal }
}
