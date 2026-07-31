import XCTest
@testable import MLXCore

/// The multi-turn engine: `ChatTurnEngine` runs one turn PER SESSION, not one
/// app-wide. The bookkeeping lives in the pure `TurnLedger` (token-identified
/// turns so a superseded task can never clear its successor's slot), and the
/// composer decision loses `.busyElsewhere` — another chat's turn no longer
/// blocks Send anywhere.
final class MultiTurnEngineTests: XCTestCase {

    // MARK: - Composer state (per session, no busyElsewhere)

    func testTwoSessionsBothShowGeneratingHere() {
        let a = UUID(), b = UUID()
        let active: Set<UUID> = [a, b]
        XCTAssertEqual(ChatTurnEngine.composerState(activeTurnSessionIds: active, for: a), .generatingHere)
        XCTAssertEqual(ChatTurnEngine.composerState(activeTurnSessionIds: active, for: b), .generatingHere)
    }

    func testUninvolvedSessionIsIdleWhileOthersGenerate() {
        let a = UUID(), c = UUID()
        XCTAssertEqual(ChatTurnEngine.composerState(activeTurnSessionIds: [a], for: c), .idle,
                       "another chat's turn must not block this chat's Send")
    }

    func testNoTurnsMeansIdleEverywhere() {
        XCTAssertEqual(ChatTurnEngine.composerState(activeTurnSessionIds: [], for: UUID()), .idle)
    }

    // MARK: - TurnLedger

    func testBeginTracksSessionAndEndClears() {
        var ledger = TurnLedger()
        let s = UUID()
        let token = ledger.begin(session: s)
        XCTAssertTrue(ledger.isBusy)
        XCTAssertEqual(ledger.activeSessionIds, [s])
        XCTAssertTrue(ledger.end(session: s, token: token))
        XCTAssertFalse(ledger.isBusy)
        XCTAssertTrue(ledger.activeSessionIds.isEmpty)
    }

    /// The supersede race: session's old task is cancelled, a NEW turn begins
    /// in the same session, then the old task's async unwind runs its cleanup.
    /// With token-identified turns the stale cleanup is a no-op; without it,
    /// the old task would mark the session idle while the new turn streams.
    func testSupersededTurnsCleanupCannotClearItsSuccessor() {
        var ledger = TurnLedger()
        let s = UUID()
        let oldToken = ledger.begin(session: s)
        let newToken = ledger.begin(session: s)          // supersedes
        XCTAssertFalse(ledger.end(session: s, token: oldToken),
                       "stale cleanup must not clear the successor's slot")
        XCTAssertEqual(ledger.activeSessionIds, [s])
        XCTAssertTrue(ledger.end(session: s, token: newToken))
        XCTAssertTrue(ledger.activeSessionIds.isEmpty)
    }

    func testConcurrentSessionsAreIndependent() {
        var ledger = TurnLedger()
        let a = UUID(), b = UUID()
        let ta = ledger.begin(session: a)
        _ = ledger.begin(session: b)
        XCTAssertEqual(ledger.activeSessionIds, [a, b])
        XCTAssertTrue(ledger.end(session: a, token: ta))
        XCTAssertEqual(ledger.activeSessionIds, [b],
                       "ending A's turn must leave B's running")
    }

    func testLiveTokensAreScopedPerSession() {
        var ledger = TurnLedger()
        let a = UUID(), b = UUID()
        _ = ledger.begin(session: a)
        _ = ledger.begin(session: b)
        ledger.setLiveTokens(42, session: a)
        ledger.setLiveTokens(7, session: b)
        XCTAssertEqual(ledger.liveTokens(session: a), 42)
        XCTAssertEqual(ledger.liveTokens(session: b), 7)
        XCTAssertEqual(ledger.liveTokens(session: UUID()), 0)
    }

    // MARK: - Orphan sweep (per turn, not all-or-nothing)

    func testOrphanSweepStopsOnlyTheGoneSessionsTurn() {
        var ledger = TurnLedger()
        let kept = UUID(), deleted = UUID()
        _ = ledger.begin(session: kept)
        _ = ledger.begin(session: deleted)
        XCTAssertEqual(ledger.orphaned(existingSessions: [kept]), [deleted],
                       "deleting one chat must stop ONLY that chat's turn")
        XCTAssertEqual(ledger.orphaned(existingSessions: [kept, deleted]), [])
    }

    func testIdleLedgerHasNoOrphans() {
        let ledger = TurnLedger()
        XCTAssertEqual(ledger.orphaned(existingSessions: []), [])
    }

    // MARK: - MediaTurnBudget under concurrent turns

    /// Two turns generating CONCURRENTLY interleave their claims. The old
    /// single-token budget reset `spent` on every token change, so an
    /// interleaved pair got unlimited generations. Each turn owns its budget.
    func testInterleavedConcurrentTurnsEachKeepTheirOwnBudget() {
        var budget = MediaTurnBudget()
        let turnA = UUID(), turnB = UUID()
        XCTAssertNil(budget.claim(.image, turn: turnA), "A's first claim passes")
        XCTAssertNil(budget.claim(.image, turn: turnB), "B's first claim passes")
        XCTAssertNotNil(budget.claim(.image, turn: turnA),
                        "A's SECOND claim must be refused even though B claimed in between")
        XCTAssertNotNil(budget.claim(.speech, turn: turnB),
                        "B's second claim (any modality) must be refused too")
    }

    func testBudgetStillRefusesWithinOneTurnAndResetsOnANewTurn() {
        var budget = MediaTurnBudget()
        let t1 = UUID(), t2 = UUID()
        XCTAssertNil(budget.claim(.music, turn: t1))
        XCTAssertNotNil(budget.claim(.music, turn: t1))
        XCTAssertNil(budget.claim(.music, turn: t2), "a new user turn gets a fresh budget")
    }
}
