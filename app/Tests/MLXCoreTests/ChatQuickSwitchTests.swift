import XCTest
@testable import MLXCore

/// Holding ⌘ numbers the conversation rows; ⌘1…⌘9 jumps to one.
///
/// The numbers have to mean what the eye sees, so the ordering is DERIVED from
/// `SidebarSessionGroups.split` rather than re-implemented here — the sidebar
/// draws agent threads above plain chats, and a quick-switch numbered in
/// `visibleChatSessions` order would put 1 on whichever row happens to be
/// newest. Two orderings that agree today and drift on the next section is the
/// whole failure mode; there is one ordering.
final class ChatQuickSwitchTests: XCTestCase {

    private func session(agent: UUID? = nil, title: String = ChatSessionTitle.plain) -> ChatSession {
        var s = ChatSession(title: title)
        s.agentId = agent
        return s
    }

    // MARK: - Ordering

    func testNumbersFollowTheSidebarsOwnOrderNotTheRawList() {
        // The plain chat is FIRST in the raw list (newest), but the sidebar
        // draws the agent thread above it — so ⌘1 is the agent thread.
        let chat = session(title: "dolphins")
        let agentThread = session(agent: UUID(), title: "Dinner plans")
        let sessions = [chat, agentThread]

        XCTAssertEqual(ChatQuickSwitch.slot(for: agentThread.id, in: sessions), 1)
        XCTAssertEqual(ChatQuickSwitch.slot(for: chat.id, in: sessions), 2)
    }

    func testTheOrderIsTheSplitsOrder() {
        // Pinned as a DERIVATION, so adding a section to the sidebar cannot
        // leave the numbering behind.
        let sessions = [session(), session(agent: UUID()), session(), session(agent: UUID())]
        let groups = SidebarSessionGroups.split(sessions)
        XCTAssertEqual(ChatQuickSwitch.ordered(sessions).map(\.id),
                       (groups.agents + groups.chats).map(\.id))
    }

    // MARK: - Slots

    func testSlotsAreOneBased() {
        let sessions = [session(), session()]
        XCTAssertEqual(ChatQuickSwitch.slot(for: sessions[0].id, in: sessions), 1)
        XCTAssertEqual(ChatQuickSwitch.slot(for: sessions[1].id, in: sessions), 2)
    }

    func testOnlyTheFirstNineGetANumber() {
        // There is no ⌘10, so the tenth row draws no badge rather than one
        // that promises a shortcut nothing can press.
        let sessions = (0..<12).map { _ in session() }
        XCTAssertEqual(ChatQuickSwitch.slot(for: sessions[8].id, in: sessions), 9)
        XCTAssertNil(ChatQuickSwitch.slot(for: sessions[9].id, in: sessions))
        XCTAssertNil(ChatQuickSwitch.slot(for: sessions[11].id, in: sessions))
    }

    func testAnUnknownSessionHasNoSlot() {
        XCTAssertNil(ChatQuickSwitch.slot(for: UUID(), in: [session()]))
        XCTAssertNil(ChatQuickSwitch.slot(for: UUID(), in: []))
    }

    // MARK: - The digit → chat direction

    func testADigitFindsTheRowItIsDrawnOn() {
        let sessions = [session(title: "a"), session(agent: UUID(), title: "b"), session(title: "c")]
        // Round-trip: whatever slot a row shows is the digit that reaches it.
        for s in sessions {
            guard let slot = ChatQuickSwitch.slot(for: s.id, in: sessions) else {
                return XCTFail("every row in a 3-chat sidebar has a slot")
            }
            XCTAssertEqual(ChatQuickSwitch.id(forSlot: slot, in: sessions), s.id)
        }
    }

    func testDigitsPastTheEndDoNothing() {
        let sessions = [session(), session()]
        XCTAssertNil(ChatQuickSwitch.id(forSlot: 3, in: sessions))
        XCTAssertNil(ChatQuickSwitch.id(forSlot: 9, in: sessions))
    }

    func testSlotZeroAndNegativesAreRefused() {
        // ⌘0 is not a slot. Without this the 1-based number would index -1.
        let sessions = [session(), session()]
        XCTAssertNil(ChatQuickSwitch.id(forSlot: 0, in: sessions))
        XCTAssertNil(ChatQuickSwitch.id(forSlot: -1, in: sessions))
    }

    func testAnEmptySidebarHasNothingToJumpTo() {
        XCTAssertNil(ChatQuickSwitch.id(forSlot: 1, in: []))
        XCTAssertTrue(ChatQuickSwitch.ordered([]).isEmpty)
    }

    // MARK: - Wiring

    func testTheSidebarDrawsBadgesAndBindsEveryDigit() {
        let source = SourceScan.source("Views/ChatView.swift", from: #filePath)
        XCTAssertFalse(source.isEmpty, "could not read ChatView.swift")

        XCTAssertTrue(source.contains("ChatQuickSwitch.slot(for:"),
                      "the row must ask for its own number rather than counting rows itself")

        // A drawn number that nothing can press is the failure. Rather than
        // look for nine literal shortcuts — which is what drifts, one deleted
        // digit at a time — pin that the badges and the key bindings are
        // generated from the SAME constant, so they cannot disagree at all.
        XCTAssertEqual(ChatQuickSwitch.maxSlots, 9, "⌘0 is not a slot and there is no ⌘10")
        XCTAssertTrue(source.contains("ForEach(1...ChatQuickSwitch.maxSlots"),
                      "the shortcuts must be generated over maxSlots, the same cap slot(for:) applies")
        XCTAssertTrue(source.contains("keyboardShortcut(KeyEquivalent"),
                      "each generated row needs a ⌘-digit key equivalent")
        XCTAssertTrue(source.contains("ChatQuickSwitch.id(forSlot:"),
                      "a digit must resolve through the same ordering the badge was drawn from")
    }

    func testTheBadgeIsGatedOnTheCommandKey() {
        let source = SourceScan.source("Views/ChatView.swift", from: #filePath)
        XCTAssertTrue(source.contains("commandHeld"),
                      "badges appear only while ⌘ is down — otherwise they are permanent chrome")
    }
}
