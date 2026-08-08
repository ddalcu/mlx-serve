import XCTest
@testable import MLXCore

/// Pins the tray panel's shared visual system.
///
/// The menu-bar popover had grown one ad-hoc treatment per section — seven
/// full-width dividers, four different section-header styles, and horizontal
/// padding written as a literal `16` in eleven places while the dividers used
/// `12`. The redesign puts every section on `TrayMetrics` + `TrayCard`, and
/// these are the guards that keep it there.
final class TrayChromeTests: XCTestCase {

    // MARK: - Status chip

    /// The header chip is a GLANCE. `ServerStatus.label` interpolates the whole
    /// error message ("Error: MISSING WEIGHT: model.embed_tokens.weight …"),
    /// which in a 340pt panel either truncates to nothing useful or blows the
    /// header's height — the full text belongs in the tooltip and the error row
    /// under the server controls, both of which already show it.
    func testStatusChipNeverRendersTheRawErrorMessage() {
        let raw = "MISSING WEIGHT: model.embed_tokens.weight — checkpoint is incomplete"
        let chip = TrayStatusChipModel(status: .error(raw))

        XCTAssertEqual(chip.label, "Error")
        XCTAssertEqual(chip.tone, .error)
        XCTAssertFalse(chip.label.contains(raw))
    }

    func testStatusChipLabelsAndTones() {
        XCTAssertEqual(TrayStatusChipModel(status: .running).label, "Running")
        XCTAssertEqual(TrayStatusChipModel(status: .running).tone, .running)
        XCTAssertEqual(TrayStatusChipModel(status: .starting).label, "Loading")
        XCTAssertEqual(TrayStatusChipModel(status: .starting).tone, .starting)
        XCTAssertEqual(TrayStatusChipModel(status: .stopped).label, "Stopped")
        XCTAssertEqual(TrayStatusChipModel(status: .stopped).tone, .stopped)
    }

    // MARK: - Server control emphasis

    /// Only ONE filled control per panel, and it goes to the action the user is
    /// most likely to want. With the server up that action is "open Chat", not
    /// "Stop Server" — a full-width red fill was the loudest thing on screen for
    /// the state the app spends all its time in. Start/Loading stay prominent:
    /// there, starting the server IS the thing to do next.
    func testOnlyTheStartAndLoadingStatesAreProminent() {
        XCTAssertTrue(ServerControlButtonPresentation(status: .stopped).isProminent)
        XCTAssertTrue(ServerControlButtonPresentation(status: .starting).isProminent)
        XCTAssertTrue(ServerControlButtonPresentation(status: .error("x")).isProminent)
        XCTAssertFalse(ServerControlButtonPresentation(status: .running).isProminent)
    }

    /// The titles/tints the older tests pin must survive the emphasis change —
    /// stop is still red, it just isn't filled.
    func testRunningKeepsItsRedStopPresentation() {
        let running = ServerControlButtonPresentation(status: .running)
        XCTAssertEqual(running.title, "Stop Server")
        XCTAssertEqual(running.tint, .red)
    }

    // MARK: - Source audit: one spacing system

    /// The `StatusMenuView` declaration only — the file also holds the server
    /// log window, which is a full window with its own (correct) 12pt toolbar
    /// inset and must not be dragged into the panel's rhythm.
    private func traySource() throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent("Sources/MLXServe/Views/StatusMenuView.swift")
        let whole = try String(contentsOf: url, encoding: .utf8)
        let start = try XCTUnwrap(whole.range(of: "struct StatusMenuView"),
                                  "StatusMenuView is gone from StatusMenuView.swift")
        let end = try XCTUnwrap(whole.range(of: "struct UpdateTrayRow"),
                                "This audit slices at UpdateTrayRow — rename it and the slice is wrong")
        return String(whole[start.lowerBound..<end.lowerBound])
    }

    /// Every gutter in the panel comes from `TrayMetrics`, so widening the panel
    /// or retuning the rhythm is one edit. The literals this replaced were the
    /// reason the dividers (12) sat inset from the content (16).
    func testTrayPanelDoesNotHardcodeHorizontalGutters() throws {
        let source = try traySource()
        for literal in ["padding(.horizontal, 16)", "padding(.horizontal, 12)"] {
            XCTAssertFalse(source.contains(literal), """
                StatusMenuView must take its horizontal gutter from \
                TrayMetrics.gutter, not the literal `\(literal)` — mixed \
                literals are how the section rhythm drifted apart.
                """)
        }
        XCTAssertTrue(source.contains("TrayMetrics.gutter"))
    }

    /// The panel's width is a layout constant, not a magic number typed at the
    /// bottom of a 600-line view.
    func testTrayPanelWidthComesFromTrayMetrics() throws {
        let source = try traySource()
        XCTAssertTrue(source.contains("frame(width: TrayMetrics.width)"))
        XCTAssertFalse(source.contains("frame(width: 320)"))
    }

    /// Grouping is done by cards, not by hairlines. A few `Divider()`s survive
    /// (inside a card, above the footer bar); a wall of them is what the
    /// redesign removed, and it crept in one section at a time.
    func testTrayPanelGroupsWithCardsNotAWallOfDividers() throws {
        let source = try traySource()
        let dividers = source.components(separatedBy: "Divider()").count - 1
        XCTAssertLessThanOrEqual(dividers, 3, """
            \(dividers) Divider()s in the tray panel — group sections with \
            TrayCard + section headers instead. Hairlines between every \
            section is the look this redesign replaced.
            """)
        XCTAssertTrue(source.contains("TrayCard"))
    }
}
