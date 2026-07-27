import XCTest
@testable import MLXCore

/// Each agent keeps its OWN conversation. Starting voice with Chef picked must
/// continue Chef's thread — not talk into whatever tab happened to be open, and
/// not rebrand that tab as Chef. "New" still starts a fresh thread and leaves the
/// old one in the sidebar, exactly as it always has.
///
/// The routing is a pure decision so the voice controller and the tray agree.
final class AgentSessionThreadTests: XCTestCase {

    private func session(agent: UUID? = nil, updated: TimeInterval,
                         taskRun: UUID? = nil, bridge: Bool = false) -> ChatSession {
        var s = ChatSession(title: "t")
        s.agentId = agent
        s.updatedAt = Date(timeIntervalSince1970: updated)
        s.taskRunId = taskRun
        s.isExternalBridge = bridge
        return s
    }

    private func route(_ agentId: UUID?, _ sessions: [ChatSession], active: UUID?) -> UUID? {
        AppState.sessionForAgent(agentId, sessions: sessions, activeId: active)
    }

    // MARK: - No agent: today's behavior, untouched

    func testNoAgentKeepsTalkingIntoTheActiveTab() {
        let a = session(updated: 100)
        XCTAssertEqual(route(nil, [a], active: a.id), a.id)
    }

    func testNoAgentAndNoActiveTabStartsAFreshThread() {
        XCTAssertNil(route(nil, [], active: nil), "nil = caller creates one")
        let a = session(updated: 100)
        XCTAssertNil(route(nil, [a], active: nil))
    }

    func testAStaleActiveIdIsNotAdopted() {
        // The active chat was deleted underneath us — a dangling id would run the
        // turn into a session that no longer exists (the ghost-turn shape).
        let a = session(updated: 100)
        XCTAssertNil(route(nil, [a], active: UUID()))
    }

    // MARK: - An agent gets its own thread

    func testAnAgentWithNoThreadYetStartsOne() {
        let chef = UUID()
        let other = session(updated: 100)
        XCTAssertNil(route(chef, [other], active: other.id),
                     "never adopt another conversation as Chef's")
    }

    func testAnAgentContinuesItsOwnThreadEvenWhenAnotherTabIsActive() {
        let chef = UUID()
        let chefThread = session(agent: chef, updated: 100)
        let plainActive = session(updated: 200)
        XCTAssertEqual(route(chef, [plainActive, chefThread], active: plainActive.id), chefThread.id)
    }

    func testTheAgentsMostRecentThreadWins() {
        let chef = UUID()
        let old = session(agent: chef, updated: 100)
        let recent = session(agent: chef, updated: 300)
        XCTAssertEqual(route(chef, [old, recent], active: nil), recent.id)
    }

    func testAnActiveThreadOfTheSameAgentIsPreferredOverANewerOne() {
        // The user opened an older Chef thread and then spoke — answer THERE,
        // rather than yanking them to the most recently touched one.
        let chef = UUID()
        let older = session(agent: chef, updated: 100)
        let newer = session(agent: chef, updated: 300)
        XCTAssertEqual(route(chef, [older, newer], active: older.id), older.id)
    }

    // MARK: - Vehicles that are never conversations

    func testATaskRunSessionIsNeverAdopted() {
        // Scheduled runs now carry `agentId` too, and their sessions are hidden,
        // transient and harvested into a transcript — speaking into one would
        // corrupt a run and vanish from the sidebar.
        let chef = UUID()
        let taskRun = session(agent: chef, updated: 300, taskRun: UUID())
        XCTAssertNil(route(chef, [taskRun], active: nil))
    }

    func testATelegramBridgeSessionIsNeverAdopted() {
        let chef = UUID()
        let bridge = session(agent: chef, updated: 300, bridge: true)
        XCTAssertNil(route(chef, [bridge], active: nil))
        // …not even when it's the active tab (it's a read-only mirror).
        XCTAssertNil(route(chef, [bridge], active: bridge.id))
    }

    func testAnActiveTaskOrBridgeSessionIsNotAdoptedForTheNoAgentCaseEither() {
        let taskRun = session(updated: 300, taskRun: UUID())
        XCTAssertNil(route(nil, [taskRun], active: taskRun.id))
    }
}
