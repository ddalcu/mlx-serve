import XCTest
@testable import MLXCore

/// The sidebar's terminal rows — pure, no SwiftTerm, no VM. Several
/// pi/hermes/shell sessions run concurrently (each its own ssh into the shared
/// guest); this model owns ordering, stable display names and phases. Which
/// one is SHOWING is `ChatWorkspace.terminal(id)`, not this list.
final class TerminalSessionListTests: XCTestCase {

    private func list() -> TerminalSessionList { TerminalSessionList() }

    func testAddPreparingRecordsAgentAndWorkspace() {
        var m = list()
        let id = m.addPreparing(label: "pi", agentId: "pi", workspace: "/Users/x/proj")
        XCTAssertEqual(m.sessions.count, 1)
        XCTAssertEqual(m.sessions.first?.phase, .preparing)
        XCTAssertEqual(m.sessions.first?.agentId, "pi")
        XCTAssertEqual(m.sessions.first?.workspace, "/Users/x/proj")
        m.markLive(id)
        XCTAssertEqual(m.sessions.first?.phase, .live)
    }

    func testConcurrentSameAgentSessionsGetStableNumberedNames() {
        var m = list()
        let a = m.addPreparing(label: "pi", agentId: "pi", workspace: "/a")
        let b = m.addPreparing(label: "pi", agentId: "pi", workspace: "/b")
        let c = m.addPreparing(label: "hermes", agentId: "hermes", workspace: "/c")
        XCTAssertEqual(m.displayName(a), "pi")
        XCTAssertEqual(m.displayName(b), "pi 2")
        XCTAssertEqual(m.displayName(c), "hermes")
        // Names are assigned at creation and NEVER renumber — "pi 2" turning
        // into "pi" mid-session would gaslight the user about which is which.
        m.close(a)
        XCTAssertEqual(m.displayName(b), "pi 2")
        let d = m.addPreparing(label: "pi", agentId: "pi", workspace: "/d")
        XCTAssertEqual(m.displayName(d), "pi 3")
    }

    func testExitKeepsTheSessionWithAnHonestNotice() {
        var m = list()
        let id = m.addPreparing(label: "hermes", agentId: "hermes", workspace: "/a")
        m.markLive(id)
        m.markExited(id, exitCode: 7)
        XCTAssertEqual(m.sessions.count, 1, "an exited session stays until closed — its notice explains what happened")
        XCTAssertEqual(m.exitNotice(id), "hermes session ended (exit 7)")
        let b = m.addPreparing(label: "pi", agentId: "pi", workspace: "/a")
        m.markExited(b, exitCode: 0)
        XCTAssertEqual(m.exitNotice(b), "pi session ended")
        let c = m.addPreparing(label: "pi", agentId: "pi", workspace: "/a")
        m.markExited(c, exitCode: nil)
        XCTAssertEqual(m.exitNotice(c), "pi 2 session ended")
    }

    func testCloseConfirmationOnlyWhenASessionWouldActuallyDie() {
        var m = list()
        let a = m.addPreparing(label: "pi", agentId: "pi", workspace: "/a")
        XCTAssertTrue(m.closeNeedsConfirmation(a), "preparing: a session is being created — closing abandons it")
        m.markLive(a)
        XCTAssertTrue(m.closeNeedsConfirmation(a), "live: closing terminates the session")
        m.markExited(a, exitCode: 0)
        XCTAssertFalse(m.closeNeedsConfirmation(a), "exited: nothing to lose, no nag")
        XCTAssertFalse(m.closeNeedsConfirmation(UUID()))
    }

    func testFailedSessionsCloseWithoutConfirmAndNeverCountAsActive() {
        // A preflight/boot failure is a ROW now (the message + fix sit inline
        // where the window's alert used to be). Nothing is running behind it.
        var m = list()
        let a = m.addPreparing(label: "pi", agentId: "pi", workspace: "/a")
        m.markFailed(a, message: "guest networking is off")
        XCTAssertEqual(m.sessions.first?.phase, .failed("guest networking is off"))
        XCTAssertFalse(m.closeNeedsConfirmation(a))
        XCTAssertNil(m.mostRecentActive(label: "pi"))
        XCTAssertNil(m.exitNotice(a))
        m.restart(a)
        XCTAssertEqual(m.sessions.first?.phase, .failed("guest networking is off"),
                       "a remount respawns living sessions only")
    }

    func testRetryTakesAFailedRowBackToPreparingAndNothingElse() {
        // "Start Server" / "Retry" on a failed row re-runs the launch IN PLACE:
        // same row, same name — never a second numbered row for the user's one
        // attempt. Living and exited rows are not retry targets.
        var m = list()
        let a = m.addPreparing(label: "pi", agentId: "pi", workspace: "/a")
        m.markFailed(a, message: "the server isn't running")
        m.retry(a)
        XCTAssertEqual(m.sessions.first?.phase, .preparing)
        XCTAssertEqual(m.displayName(a), "pi")
        m.markExited(a, exitCode: 0)
        m.retry(a)
        XCTAssertEqual(m.sessions.first?.phase, .exited(0))
    }

    func testMostRecentActiveFindsTheNewestLivingSessionOfAnAgent() {
        var m = list()
        XCTAssertNil(m.mostRecentActive(label: "pi"))
        let a = m.addPreparing(label: "pi", agentId: "pi", workspace: "/a")
        m.markLive(a)
        let b = m.addPreparing(label: "pi", agentId: "pi", workspace: "/b")
        m.markLive(b)
        _ = m.addPreparing(label: "hermes", agentId: "hermes", workspace: "/c")
        XCTAssertEqual(m.mostRecentActive(label: "pi"), b, "newest living session wins")
        m.markExited(b, exitCode: 0)
        XCTAssertEqual(m.mostRecentActive(label: "pi"), a, "an exited session is not a focus target")
        m.markExited(a, exitCode: 0)
        XCTAssertNil(m.mostRecentActive(label: "pi"))
        let c = m.addPreparing(label: "pi", agentId: "pi", workspace: "/a")
        XCTAssertEqual(m.mostRecentActive(label: "pi"), c)
    }

    func testRestartKeepsIdentityAndReturnsToPreparing() {
        // A workspace remount (Settings pick under live sessions) restarts
        // each session IN PLACE: same row, same display name.
        var m = list()
        let pi = m.addPreparing(label: "pi", agentId: "pi", workspace: "/a")
        m.markLive(pi)
        m.restart(pi)
        XCTAssertEqual(m.sessions.first?.phase, .preparing)
        XCTAssertEqual(m.displayName(pi), "pi", "restart must not renumber")
    }

    func testRestartLeavesExitedSessionsAlone() {
        var m = list()
        let pi = m.addPreparing(label: "pi", agentId: "pi", workspace: "/a")
        m.markLive(pi)
        m.markExited(pi, exitCode: 0)
        m.restart(pi)
        XCTAssertEqual(m.sessions.first?.phase, .exited(0))
    }

    func testCloseOfUnknownIdIsANoop() {
        var m = list()
        let a = m.addPreparing(label: "pi", agentId: "pi", workspace: "/a")
        m.close(UUID())
        XCTAssertEqual(m.sessions.map(\.id), [a])
    }

    // MARK: - Source audit

    /// The whole reason terminals moved out of their window: dismantling the
    /// hosting view used to `terminate()` the ssh, so closing the chat window
    /// (or any re-layout that unmounted the view) killed a live TUI. The
    /// process is owned by `EmbeddedTerminalView.Handle`, which outlives the
    /// view; dismantle only un-parents the terminal.
    func testDismantlingTheTerminalHostNeverTerminatesTheProcess() {
        let src = SourceScan.source("Views/EmbeddedTerminalView.swift", from: #filePath)
        guard let body = SourceScan.declarationBody(from: "static func dismantleNSView", in: src) else {
            return XCTFail("EmbeddedTerminalView.dismantleNSView moved — re-anchor this scan")
        }
        XCTAssertFalse(body.contains("terminate("), """
            dismantleNSView must never terminate the terminal's process — the \
            view is re-parented across windows; the Handle owns the ssh.
            """)
    }
}

extension TerminalSessionListTests {
    func testAPerSessionThemeIsRememberedOnTheRow() {
        var m = TerminalSessionList()
        let a = m.addPreparing(label: "pi", agentId: "pi", workspace: "/a")
        XCTAssertNil(m.session(a)?.themeId, "new rows follow the Settings default")
        m.setTheme(a, themeId: "dracula")
        XCTAssertEqual(m.session(a)?.themeId, "dracula")
        m.setTheme(a, themeId: nil)
        XCTAssertNil(m.session(a)?.themeId)
    }
}

extension TerminalSessionListTests {
    func testMovingATerminalToItsOwnWindowIsRememberedOnTheRow() {
        // "Move Tab to New Window": the row stays in the sidebar (it is still a
        // session) but points at the window; closing the window moves it back.
        var m = TerminalSessionList()
        let a = m.addPreparing(label: "pi", agentId: "pi", workspace: "/a")
        XCTAssertFalse(m.session(a)?.isInOwnWindow ?? true)
        m.setInOwnWindow(a, true)
        XCTAssertTrue(m.session(a)?.isInOwnWindow ?? false)
        m.setInOwnWindow(a, false)
        XCTAssertFalse(m.session(a)?.isInOwnWindow ?? true)
    }
}

extension TerminalSessionListTests {
    func testRenameOverridesTheAutoNameAndBlankRestoresIt() {
        var m = TerminalSessionList()
        let a = m.addPreparing(label: "pi", agentId: "pi", workspace: "/a")
        _ = m.addPreparing(label: "pi", agentId: "pi", workspace: "/b")
        m.rename(a, to: "  backend  ")
        XCTAssertEqual(m.displayName(a), "backend")
        XCTAssertEqual(m.exitNotice(a), nil)
        m.markExited(a, exitCode: 0)
        XCTAssertEqual(m.exitNotice(a), "backend session ended", "notices use the name on the row")
        m.rename(a, to: "   ")
        XCTAssertEqual(m.displayName(a), "pi", "blank = back to the numbered auto name")
    }
}
