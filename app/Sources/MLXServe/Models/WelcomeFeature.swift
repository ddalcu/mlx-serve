import Foundation

/// The three features listed as clickable bullets in the welcome screen's left
/// column. Selecting one drives the right column (`rightPanel`). Pure data so
/// the order, copy and panel mapping are testable without standing up a view.
///
/// ORDER IS LOAD-BEARING: "Run models locally" leads and is the default
/// selection — it's the first-run fact that matters most (you can't chat
/// without a model), and its panel is the recommended-download card.
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
        case .menuBar:    return "Lives in your menu bar"
        case .agentTools: return "Agent with tools"
        }
    }

    var description: String {
        switch self {
        case .runModels:
            return "No cloud, no API keys. All processing stays on your device."
        case .menuBar:
            return "Click the icon in the top-right of your screen to start a server, download models, and chat."
        case .agentTools:
            return "Let the model read files, run commands, search the web, and write code."
        }
    }

    /// Which right-column panel this feature shows.
    var rightPanel: WelcomeRightPanel {
        switch self {
        case .runModels:  return .modelDownload   // Gemma 4 recommended-download card
        case .agentTools: return .cliInstall       // Terminal-command install row
        case .menuBar:    return .placeholder       // gray square (for now)
        }
    }
}

/// The kinds of content the welcome screen's right column can show. The view
/// renders each; the enum just decides WHICH, so the feature→panel wiring can
/// be pinned in a test.
enum WelcomeRightPanel: Equatable {
    /// The recommended-model download card (`RecommendedStarterCard`).
    case modelDownload
    /// The `mlx-serve` Terminal-command install row.
    case cliInstall
    /// A neutral gray square stand-in until real art lands.
    case placeholder
}
