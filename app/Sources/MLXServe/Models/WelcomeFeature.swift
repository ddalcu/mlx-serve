import Foundation

/// The three features listed as clickable bullets in the welcome screen's left
/// column. Selecting one drives the right column (`rightPanel`). Pure data so
/// the order, copy and panel mapping are testable without standing up a view.
enum WelcomeFeature: String, CaseIterable, Identifiable, Equatable {
    case runModels
    case menuBar
    case agentTools

    var id: String { rawValue }

    /// Top-to-bottom display order. `runModels` first (see the type comment).
    static let ordered: [WelcomeFeature] = [.runModels, .menuBar, .agentTools]

    /// The bullet the screen opens on.
    static let `default`: WelcomeFeature = .runModels

    var icon: String {
        switch self {
        case .runModels:  return "bolt.fill"
        case .menuBar:    return "menubar.rectangle"
        case .agentTools: return "wrench.and.screwdriver.fill"
        }
    }

    var title: String {
        switch self {
        case .runModels:  return "Run models locally"
        case .menuBar:    return "App, Menu Bar, or Terminal"
        case .agentTools: return "Agent with tools"
        }
    }

    var description: String {
        switch self {
        case .runModels:
            return "No cloud, no API keys. All processing stays on your device."
        case .menuBar:
            return "Use the window, the menu-bar icon, or the mlx-serve command — whichever suits the moment."
        case .agentTools:
            return "Let the model read files, run commands, search the web, and write code."
        }
    }

    /// Which right-column panel this feature shows.
    var rightPanel: WelcomeRightPanel {
        switch self {
        case .runModels:  return .modelDownload   // Gemma 4 recommended-download card
        case .menuBar:    return .surfaces         // App / Menu bar / Terminal, the last installable
        case .agentTools: return .toolsDemo        // looping screen recording of the tool loop
        }
    }
}

/// The kinds of content the welcome screen's right column can show. The view
/// renders each; the enum just decides WHICH, so the feature→panel wiring can
/// be pinned in a test.
enum WelcomeRightPanel: Equatable {
    /// The recommended-model download card (`RecommendedStarterCard`).
    case modelDownload
    /// The three places you can drive the app from (`WelcomeSurface`): the
    /// window, the menu bar, and the `mlx-serve` command — the last of which is
    /// the only one with anything to install.
    case surfaces
    /// A looping, silent screen recording of the agent using its tools. A demo
    /// answers "what does this DO" in a way a paragraph cannot.
    case toolsDemo
}
