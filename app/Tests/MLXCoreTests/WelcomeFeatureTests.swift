import XCTest
@testable import MLXCore

/// The welcome screen's left column is a tab-style selector over three features;
/// each drives a different right-column panel. This pins the spec that survives
/// a redesign: the ORDER ("Run models locally" leads and is the default
/// selection), that every feature carries copy + an icon, and the exact
/// feature→panel mapping the user asked for.
final class WelcomeFeatureTests: XCTestCase {
    func testRunModelsLeadsAndIsTheDefaultSelection() {
        XCTAssertEqual(WelcomeFeature.ordered.first, .runModels,
                       "'Run models locally' must be the top bullet")
        XCTAssertEqual(WelcomeFeature.default, .runModels,
                       "the right column opens on the Run-models panel")
        XCTAssertEqual(WelcomeFeature.ordered, [.runModels, .menuBar, .agentTools])
    }

    func testOrderedCoversEveryCaseExactlyOnce() {
        XCTAssertEqual(Set(WelcomeFeature.ordered), Set(WelcomeFeature.allCases),
                       "a feature missing from `ordered` would be unreachable in the UI")
        XCTAssertEqual(WelcomeFeature.ordered.count, WelcomeFeature.allCases.count,
                       "no feature may appear twice")
    }

    func testEveryFeatureHasCopyAndAnIcon() {
        for feature in WelcomeFeature.ordered {
            XCTAssertFalse(feature.title.isEmpty, "\(feature) needs a title")
            XCTAssertFalse(feature.description.isEmpty, "\(feature) needs a description")
            XCTAssertFalse(feature.icon.isEmpty, "\(feature) needs an SF Symbol")
        }
    }

    func testPanelMappingMatchesTheSpec() {
        // Run models locally        → the Gemma 4 recommended-download card
        XCTAssertEqual(WelcomeFeature.runModels.rightPanel, .modelDownload)
        // App, Menu Bar, or Terminal → the three surfaces, Terminal installable
        XCTAssertEqual(WelcomeFeature.menuBar.rightPanel, .surfaces)
        // Agent with tools           → the looping, silent tools demo
        XCTAssertEqual(WelcomeFeature.agentTools.rightPanel, .toolsDemo)
    }
}
