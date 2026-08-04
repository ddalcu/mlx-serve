import XCTest
import SwiftUI
@testable import MLXCore

/// The empty conversation's discovery chips and their menu-bar twin.
///
/// Media generation, the Model Browser, Tasks, the Claude Code launcher and
/// the ⌃Space Quick Launcher all lived only in the tray popover, which is the
/// one place a confused user never looks. The chips surface them under the
/// empty chat's composer and vanish once the conversation starts; the Tools
/// command menu is the always-available twin. Both render from the SAME
/// catalog (`ChatEmptyState`) so the two surfaces cannot drift — these tests
/// pin the catalog, and source audits pin the wiring (a chip row that ChatView
/// quietly stops rendering still compiles and looks fine in every other test).
final class ChatEmptyStateTests: XCTestCase {

    private func source(_ relativePath: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    // MARK: Catalog

    func testChipCatalogCoversTheLostFeatures() {
        let chips = ChatEmptyState.chips(cliLauncherAvailable: true)
        let actions = chips.map(\.action)
        XCTAssertTrue(actions.contains(.mediaMenu), "media generation must be discoverable")
        XCTAssertTrue(actions.contains(.window("modelBrowser")), "Browse Models must be discoverable")
        XCTAssertTrue(actions.contains(.window("tasks")), "Tasks must be discoverable")
        XCTAssertTrue(actions.contains(.codeLauncher), "the code launcher must be discoverable")
        // The ⌃Space chip was removed on request (2026-08-04) — the Tools
        // command menu remains the launcher's discoverability home.
        XCTAssertFalse(chips.contains { $0.id == "launcher" })

        XCTAssertEqual(Set(chips.map(\.id)).count, chips.count, "chip ids must be unique")
        for chip in chips {
            XCTAssertFalse(chip.title.isEmpty, "\(chip.id) needs a title")
            XCTAssertFalse(chip.systemImage.isEmpty, "\(chip.id) needs a glyph")
            XCTAssertFalse(chip.help.isEmpty, "\(chip.id) needs a tooltip")
        }
    }

    func testCodeLauncherChipFollowsTheBuildFeature() {
        // The MAS build can't detect or launch other apps' CLIs — same gate as
        // the tray's Code button. A chip that can only fail is the dead-control
        // class, so it's absent there, and everything else is unchanged.
        let mas = ChatEmptyState.chips(cliLauncherAvailable: false)
        XCTAssertFalse(mas.map(\.action).contains(.codeLauncher))
        let dmg = ChatEmptyState.chips(cliLauncherAvailable: true)
        XCTAssertEqual(dmg.filter { $0.action != .codeLauncher }, mas)
    }

    /// The chip's dropdown is the tray Code button's own menu body — one
    /// shared `CLILauncherMenuItems`, or the two lists drift (the tray gains
    /// a CLI the chip never offers).
    func testCodeLauncherChipSharesTheTrayMenuItems() throws {
        let chipSource = try source("Sources/MLXServe/Views/ChatEmptyState.swift")
        XCTAssertTrue(chipSource.contains("CLILauncherMenuItems("),
                      "the Code Launcher chip must render the shared menu body")
        let traySource = try source("Sources/MLXServe/Services/CLILauncher.swift")
        XCTAssertTrue(traySource.contains("CLILauncherMenuItems("),
                      "the tray Code button must render the shared menu body")
    }

    func testMediaMenuOpensAllFourGenWindows() {
        XCTAssertEqual(ChatEmptyState.mediaItems.compactMap(\.windowId),
                       ["imageGen", "videoGen", "audioGen", "model3dGen"])
        for item in ChatEmptyState.mediaItems {
            XCTAssertFalse(item.title.isEmpty)
            XCTAssertFalse(item.systemImage.isEmpty)
        }
    }

    /// Window ids are string literals on both sides; a renamed scene id would
    /// otherwise leave a chip that silently opens nothing.
    func testEveryWindowIdTheCatalogUsesExistsInTheSceneGraph() throws {
        let app = try source("Sources/MLXServe/MLXServeApp.swift")
        let referenced = (ChatEmptyState.chips(cliLauncherAvailable: true) + ChatEmptyState.mediaItems)
            .compactMap(\.windowId)
        XCTAssertFalse(referenced.isEmpty)
        for id in referenced {
            XCTAssertTrue(app.contains("id: \"\(id)\")"),
                          "catalog opens window \"\(id)\" but no Window scene declares it")
        }
    }

    func testOnlyTheCodeLauncherRequiresTheRunningServer() {
        // Windows open fine with the server stopped (they load on demand);
        // only a CLI launch bakes the live server's URL into the session it
        // starts — the tray disables its Code button for the same reason.
        for chip in ChatEmptyState.chips(cliLauncherAvailable: true) {
            let stopped = ChatEmptyState.isEnabled(chip, serverRunning: false)
            let running = ChatEmptyState.isEnabled(chip, serverRunning: true)
            XCTAssertTrue(running, "\(chip.id) must be enabled with the server up")
            XCTAssertEqual(stopped, chip.action != .codeLauncher,
                           "\(chip.id) has the wrong server gating")
        }
    }

    // MARK: Wiring (source audits)

    func testChatViewRendersTheChipRowOnEmptyConversations() throws {
        let chat = try source("Sources/MLXServe/Views/ChatView.swift")
        XCTAssertTrue(chat.contains("EmptyStateChipRow("),
                      "ChatView must render the discovery chips for an empty conversation")
    }

    func testToolsCommandMenuExistsAndSharesTheMediaCatalog() throws {
        let app = try source("Sources/MLXServe/MLXServeApp.swift")
        XCTAssertTrue(app.contains("CommandMenu(\"Tools\")"),
                      "the menu-bar twin of the chips must exist")
        XCTAssertTrue(app.contains("ChatEmptyState.mediaItems"),
                      "the Tools menu must iterate the shared media catalog, not a second list")
        for id in ["modelBrowser", "tasks"] {
            XCTAssertTrue(app.contains("openAndFocus(\"\(id)\")"),
                          "the Tools menu must open \"\(id)\"")
        }
        XCTAssertTrue(app.contains("BuildFeatures.current.cliLauncher"),
                      "the Claude Code menu item must carry the MAS gate")
    }
}
