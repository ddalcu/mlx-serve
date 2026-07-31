import XCTest
@testable import MLXCore

/// Pins the ghost-turn guard (live capture 2026-07-03): deleting a chat while
/// its agent turn was in flight left the turn running invisibly — every
/// append/update no-ops against the gone session, the empty-response check
/// reads "" and pad-retries with FULL multi-minute generations, and the slot
/// stays busy forever with no Stop control anywhere. Server stop/restart can't
/// clear it (the turn is app-side). The rule: a turn whose session no longer
/// exists is ORPHANED and must stop — `AppState.deleteSession` stops it
/// immediately, and the agent loop re-checks per iteration as defense in depth
/// for any other session-removal path.
///
/// Multi-turn corollary: the sweep is PER TURN. Deleting one chat stops only
/// that chat's turn; every other session's stream keeps running.
final class OrphanedTurnTests: XCTestCase {

    func testOrphanedWhenTurnSessionIsGone() {
        var ledger = TurnLedger()
        let turn = UUID()
        let other = UUID()
        _ = ledger.begin(session: turn)
        XCTAssertEqual(ledger.orphaned(existingSessions: [other]), [turn])
        XCTAssertEqual(ledger.orphaned(existingSessions: []), [turn])
    }

    func testNotOrphanedWhileItsSessionStillExists() {
        var ledger = TurnLedger()
        let turn = UUID()
        let other = UUID()
        _ = ledger.begin(session: turn)
        XCTAssertEqual(ledger.orphaned(existingSessions: [turn, other]), [])
    }

    func testSweepNamesOnlyTheDeletedSessionsTurn() {
        var ledger = TurnLedger()
        let kept = UUID(), deleted = UUID()
        _ = ledger.begin(session: kept)
        _ = ledger.begin(session: deleted)
        XCTAssertEqual(ledger.orphaned(existingSessions: [kept]), [deleted],
                       "the surviving chat's turn must NOT be stopped")
    }

    /// An idle engine is never orphaned — no entries, no sweep.
    func testIdleEngineIsNeverOrphaned() {
        let ledger = TurnLedger()
        XCTAssertEqual(ledger.orphaned(existingSessions: []), [])
        XCTAssertEqual(ledger.orphaned(existingSessions: [UUID()]), [])
    }
}
