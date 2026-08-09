import XCTest
@testable import MLXCore

/// The welcome screen's second panel: the three places you can drive this app
/// from. Two of them ARE the app — the window and the menu-bar icon ship in the
/// same bundle — so their trailing state is a fact rather than a control, and
/// only the Terminal command is something to install.
final class WelcomeSurfaceTests: XCTestCase {

    func testTheThreeSurfacesInOrder() {
        XCTAssertEqual(WelcomeSurface.ordered, [.app, .menuBar, .terminal],
                       "the two you already have, then the one to add")
        XCTAssertEqual(Set(WelcomeSurface.ordered), Set(WelcomeSurface.allCases),
                       "a surface missing from `ordered` would never render")
    }

    /// The window and the menu bar cannot be absent, so a control offering to
    /// install them could only ever be dead. Only Terminal has state to probe.
    func testOnlyTheTerminalCommandIsSomethingYouInstall() {
        XCTAssertTrue(WelcomeSurface.app.shipsWithTheApp)
        XCTAssertTrue(WelcomeSurface.menuBar.shipsWithTheApp)
        XCTAssertFalse(WelcomeSurface.terminal.shipsWithTheApp)
    }

    func testEverySurfaceHasCopyAndAnIcon() {
        for surface in WelcomeSurface.ordered {
            XCTAssertFalse(surface.title.isEmpty, "\(surface) needs a title")
            XCTAssertFalse(surface.icon.isEmpty, "\(surface) needs an SF Symbol")
        }
        // Only Terminal has no constant caption: its line is live, and a copy
        // here would be a second one that silently stops matching.
        XCTAssertNil(WelcomeSurface.terminal.caption)
        for surface in WelcomeSurface.ordered where surface != .terminal {
            XCTAssertNotNil(surface.caption, "\(surface) needs a caption")
        }
    }

    // MARK: The card that leads to it

    /// The card names all three, so the panel behind it can't be a surprise.
    func testTheCardNamesWhatThePanelShows() {
        let title = WelcomeFeature.menuBar.title
        XCTAssertEqual(title, "App, Menu Bar, or Terminal")
        for word in ["App", "Menu Bar", "Terminal"] {
            XCTAssertTrue(title.contains(word), "the card must name \(word)")
        }
    }

    /// The description was rewritten with the title; it has to stay in the same
    /// length band or the card's height changes and the connector line that
    /// points at it lands somewhere else.
    func testTheDescriptionStaysInTheSameLengthBand() {
        let text = WelcomeFeature.menuBar.description
        XCTAssertGreaterThan(text.count, 70, "too short — the card collapses")
        XCTAssertLessThan(text.count, 120, "too long — the card grows and the layout shifts")
    }

    func testPanelMappingMatchesTheSpec() {
        XCTAssertEqual(WelcomeFeature.runModels.rightPanel, .modelDownload)
        XCTAssertEqual(WelcomeFeature.menuBar.rightPanel, .surfaces)
        XCTAssertEqual(WelcomeFeature.agentTools.rightPanel, .toolsDemo)
    }

    // MARK: The demo video

    /// The asset has to BE there. A demo panel that silently falls back to a
    /// gray rectangle is indistinguishable from one nobody wired up.
    func testTheToolsDemoAssetShipsInTheResources() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Resources/tools.mov")
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path),
                      "the tools demo video must ship in Sources/MLXServe/Resources")
        let size = try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int ?? 0
        XCTAssertGreaterThan(size, 1000, "the asset is present but empty")
    }

    /// Silent, looping, no controls — it is illustration, not media the user
    /// came here to operate. A welcome screen that makes noise on launch is the
    /// thing everyone remembers about an app.
    func testTheDemoPlaysSilentlyAndLoopsWithNoControls() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Views/LoopingVideoView.swift")
        let s = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(s.contains("isMuted = true"), "the demo must be muted")
        XCTAssertTrue(s.contains("AVPlayerLooper"), "…and loop seamlessly")
        XCTAssertTrue(s.contains(".play()"), "…and start on its own")
        XCTAssertFalse(s.contains("AVPlayerView("),
                       "AVPlayerView brings transport controls — this is a bare layer")
    }
}
