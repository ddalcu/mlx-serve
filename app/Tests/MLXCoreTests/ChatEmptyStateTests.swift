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
        // Not `.window(...)`: the browser is the chat window's other mode.
        XCTAssertTrue(actions.contains(.models), "Browse Models must be discoverable")
        // A mode, not a window — Tasks renders in the chat window's own detail
        // column now (`ChatWorkspace.tasks`).
        XCTAssertTrue(actions.contains(.tasks), "Tasks must be discoverable")
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

    /// The four generators are PAGES of the chat window now, not windows —
    /// so the chips carry `.create(...)` actions and no window id at all. The
    /// catalogue must still cover every generator exactly once, in the shared
    /// `GenExperiment` order.
    func testMediaMenuReachesAllFourGeneratorsAsPages() {
        XCTAssertEqual(ChatEmptyState.mediaItems.compactMap(\.windowId), [],
                       "a media chip that still opens a window points at a retired scene")
        let reached = ChatEmptyState.mediaItems.compactMap { item -> GenExperiment? in
            if case .create(let exp) = item.action { return exp }
            return nil
        }
        XCTAssertEqual(reached, GenExperiment.allCases,
                       "every generator, once, in the catalogue's own order")
        for item in ChatEmptyState.mediaItems {
            XCTAssertFalse(item.title.isEmpty)
            XCTAssertFalse(item.systemImage.isEmpty)
        }
    }

    /// The sidebar's Create rows are the SAME catalogue minus 3D — a filter on
    /// `mediaItems`, never a second hand-written list (which is where a title
    /// or an icon quietly drifts from the chips and the Tools menu). 3D's
    /// absence is deliberate: it stays reachable from the Create Media chip,
    /// the Tools menu and the tray.
    func testTheSidebarCreateRowsAreTheCatalogMinus3D() {
        let expected = ChatEmptyState.mediaItems.filter { $0.action != .create(.model3d) }
        XCTAssertEqual(ChatEmptyState.sidebarCreateItems.map(\.id), expected.map(\.id),
                       "same items, same order as the shared catalogue")
        XCTAssertEqual(ChatEmptyState.sidebarCreateItems.count, 3)
        for item in ChatEmptyState.sidebarCreateItems {
            guard case .create = item.action else {
                return XCTFail("\(item.id) is not a Create page — the sidebar rows only link to generators")
            }
        }
    }

    /// Window ids are string literals on both sides; a renamed scene id would
    /// otherwise leave a chip that silently opens nothing.
    func testEveryWindowIdTheCatalogUsesExistsInTheSceneGraph() throws {
        let app = try source("Sources/MLXServe/MLXServeApp.swift")
        // Every chip that still opens a WINDOW must name a live scene. Most are
        // modes of the chat window now, so an empty list is a legitimate state —
        // what must never happen is a chip pointing at a retired scene id.
        let referenced = (ChatEmptyState.chips(cliLauncherAvailable: true) + ChatEmptyState.mediaItems)
            .compactMap(\.windowId)
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
        // Both of the Tools menu's own destinations are modes of the chat
        // window now, reached through their AppState chokepoints.
        XCTAssertTrue(app.contains("Button(\"Scheduled Tasks…\") { appState.showTasks() }"),
                      "the Tools menu must reach the Tasks pane via AppState.showTasks()")
        // Browse Models is no longer a window — the menu item switches the chat
        // window's mode through the one chokepoint.
        XCTAssertTrue(app.contains("Button(\"Browse Models…\") { appState.showModels() }"),
                      "the Tools menu must reach the models pane via AppState.showModels()")
        XCTAssertTrue(app.contains("BuildFeatures.current.cliLauncher"),
                      "the Claude Code menu item must carry the MAS gate")
    }

    /// Iterating the catalog is not the same as RENDERING it: the media items
    /// became `.create(...)` actions (windowId is nil on all four), so a menu
    /// body that filters on `item.windowId` draws an empty section while every
    /// "shares the catalog" grep above stays green — which is exactly how the
    /// Tools menu shipped with no Image/Video/Audio/3D entries.
    func testToolsMenuRendersTheCreateActionsTheCatalogActuallyCarries() throws {
        let app = try source("Sources/MLXServe/MLXServeApp.swift")
        XCTAssertFalse(app.contains("if let windowId = item.windowId"),
                       "the media section filters on a windowId no item carries — it renders nothing")
        XCTAssertTrue(app.contains("case .create(let experiment) = item.action"),
                      "the media section must dispatch the .create actions through appState.showCreate")
    }
}
