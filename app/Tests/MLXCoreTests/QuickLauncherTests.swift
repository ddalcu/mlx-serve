import XCTest
@testable import MLXCore

/// Pure decision/geometry logic for the ⌃Space quick launcher (Spotlight-style
/// prompt panel). The AppKit panel + Carbon hotkey are untestable surfaces, so
/// everything decidable lives in `QuickLauncherLogic` — same extraction pattern
/// as `ComposerLayout` / `AppState.sidebarSessions`.
final class QuickLauncherTests: XCTestCase {

    // MARK: Hotkey combo (Ctrl+Space)

    func testHotKeyComboIsCtrlSpace() {
        XCTAssertEqual(QuickLauncherHotKey.keyCode, 49, "kVK_Space")
        XCTAssertEqual(QuickLauncherHotKey.carbonModifiers, 4096, "Carbon controlKey (1 << 12)")
    }

    // MARK: Turn config — quick launcher turns are plain chat

    func testTurnConfigIsPlainChat() {
        let config = QuickLauncherLogic.turnConfig()
        XCTAssertFalse(config.agentMode)
        XCTAssertFalse(config.mcpMode)
        XCTAssertFalse(config.enableThinking)
        XCTAssertFalse(config.voiceStyle)
        XCTAssertNil(config.workingDirectory)
        XCTAssertNil(config.documentIndex)
        XCTAssertNil(config.telegramChatId)
    }

    // MARK: Submit decision

    func testEmptyTextIsIgnored() {
        XCTAssertEqual(QuickLauncherLogic.submitDecision(text: "   \n", serverRunning: true, composer: .idle), .ignore)
        // Ignore wins even when other guards would also fire.
        XCTAssertEqual(QuickLauncherLogic.submitDecision(text: "", serverRunning: false, composer: .busyElsewhere), .ignore)
    }

    func testServerDownBlocksWithExplanation() {
        let decision = QuickLauncherLogic.submitDecision(text: "hi", serverRunning: false, composer: .idle)
        guard case .blocked(let message) = decision else {
            return XCTFail("expected .blocked, got \(decision)")
        }
        XCTAssertTrue(message.localizedCaseInsensitiveContains("server"))
    }

    func testBusyElsewhereBlocks() {
        // Another chat's turn is in flight — the launcher must never clobber it
        // (ChatTurnEngine.runTurn cancels the in-flight turn on submission).
        let decision = QuickLauncherLogic.submitDecision(text: "hi", serverRunning: true, composer: .busyElsewhere)
        guard case .blocked = decision else {
            return XCTFail("expected .blocked, got \(decision)")
        }
    }

    func testGeneratingHereStopsThenSubmits() {
        // A new question in the launcher's own conversation supersedes the
        // in-flight answer — engine.runTurn no-ops while isGenerating, so the
        // caller must stop() first.
        XCTAssertEqual(QuickLauncherLogic.submitDecision(text: "hi", serverRunning: true, composer: .generatingHere), .stopThenSubmit)
    }

    func testIdleSubmits() {
        XCTAssertEqual(QuickLauncherLogic.submitDecision(text: "hi", serverRunning: true, composer: .idle), .submit)
    }

    // MARK: Session reuse

    func testNeedsNewSessionWhenNoneOrDeleted() {
        let a = UUID(), b = UUID()
        XCTAssertTrue(QuickLauncherLogic.needsNewSession(current: nil, existing: [a, b]))
        XCTAssertTrue(QuickLauncherLogic.needsNewSession(current: UUID(), existing: [a, b]),
                      "session deleted from the sidebar → start fresh")
        XCTAssertFalse(QuickLauncherLogic.needsNewSession(current: a, existing: [a, b]))
    }

    // MARK: Panel geometry

    func testPanelHeightExpandsWithConversation() {
        let compact = QuickLauncherLogic.panelHeight(hasConversation: false)
        let expanded = QuickLauncherLogic.panelHeight(hasConversation: true)
        XCTAssertGreaterThan(expanded, compact)
    }

    func testPanelOriginCentersHorizontallyWithTopAtAQuarterDown() {
        let screen = CGRect(x: 0, y: 0, width: 1000, height: 800)
        let size = CGSize(width: QuickLauncherLogic.panelWidth, height: 120)
        let origin = QuickLauncherLogic.panelOrigin(panelSize: size, screenFrame: screen)
        XCTAssertEqual(origin.x, (1000 - QuickLauncherLogic.panelWidth) / 2, accuracy: 0.5)
        // Cocoa coords: top edge = origin.y + height, pinned 25% down the screen.
        XCTAssertEqual(origin.y + 120, 800 - 800 * 0.25, accuracy: 0.5)
    }

    func testPanelOriginClampsToNarrowScreen() {
        let screen = CGRect(x: 0, y: 0, width: 600, height: 400)
        let origin = QuickLauncherLogic.panelOrigin(panelSize: CGSize(width: QuickLauncherLogic.panelWidth, height: 120),
                                                    screenFrame: screen)
        XCTAssertGreaterThanOrEqual(origin.x, screen.minX, "panel must not start off-screen left")
    }

    func testPanelOriginRespectsScreenOriginOffset() {
        // Secondary display arranged to the right: frame origin is not (0,0).
        let screen = CGRect(x: 1512, y: 200, width: 1000, height: 800)
        let size = CGSize(width: QuickLauncherLogic.panelWidth, height: 120)
        let origin = QuickLauncherLogic.panelOrigin(panelSize: size, screenFrame: screen)
        XCTAssertEqual(origin.x, 1512 + (1000 - QuickLauncherLogic.panelWidth) / 2, accuracy: 0.5)
        XCTAssertEqual(origin.y + 120, 200 + 800 - 800 * 0.25, accuracy: 0.5)
    }
}
