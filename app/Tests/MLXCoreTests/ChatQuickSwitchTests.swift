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

    // MARK: - Only the rows in the clear get a number

    func testNumbersSkipRowsHiddenBehindTheFrostedBlock() {
        // The list scrolls UNDER the pinned destinations, so a scrolled sidebar
        // had ⌘1…⌘5 behind glass and the first row you could read was 6. The
        // rows in the clear get 1, 2, 3…
        let rows = (0..<8).map { _ in session() }
        let visible = Set(rows[3...].map(\.id))          // first three are under the blur

        XCTAssertNil(ChatQuickSwitch.slot(for: rows[0].id, in: rows, numbering: visible))
        XCTAssertNil(ChatQuickSwitch.slot(for: rows[2].id, in: rows, numbering: visible))
        XCTAssertEqual(ChatQuickSwitch.slot(for: rows[3].id, in: rows, numbering: visible), 1)
        XCTAssertEqual(ChatQuickSwitch.slot(for: rows[4].id, in: rows, numbering: visible), 2)
    }

    func testTheFilterKeepsTheSidebarsOrderNotTheSetsOrder() {
        // A Set has no order — the numbering must still run agents-then-chats,
        // top to bottom, or the badges count in an order nothing on screen has.
        let chat = session(title: "dolphins")
        let agentThread = session(agent: UUID(), title: "Dinner plans")
        let sessions = [chat, agentThread]
        let visible: Set<UUID> = [chat.id, agentThread.id]

        XCTAssertEqual(ChatQuickSwitch.slot(for: agentThread.id, in: sessions, numbering: visible), 1)
        XCTAssertEqual(ChatQuickSwitch.slot(for: chat.id, in: sessions, numbering: visible), 2)
    }

    func testADigitLandsOnTheRowShowingIt() {
        // The invariant that matters: badge and shortcut read ONE list, so a
        // digit can never reach a row other than the one wearing that number.
        let rows = (0..<9).map { _ in session() }
        let visible = Set(rows[2...6].map(\.id))
        for row in rows {
            guard let slot = ChatQuickSwitch.slot(for: row.id, in: rows, numbering: visible) else { continue }
            XCTAssertEqual(ChatQuickSwitch.id(forSlot: slot, in: rows, numbering: visible), row.id)
        }
    }

    func testNineVisibleRowsDeepInTheListAllGetNumbers() {
        // Scrolled far down: the numbers follow the window, not the top of the
        // list, so all nine are usable wherever you are.
        let rows = (0..<40).map { _ in session() }
        let visible = Set(rows[20..<29].map(\.id))
        XCTAssertEqual(ChatQuickSwitch.slot(for: rows[20].id, in: rows, numbering: visible), 1)
        XCTAssertEqual(ChatQuickSwitch.slot(for: rows[28].id, in: rows, numbering: visible), 9)
        XCTAssertEqual(ChatQuickSwitch.id(forSlot: 9, in: rows, numbering: visible), rows[28].id)
    }

    func testMoreThanNineVisibleRowsStillStopAtNine() {
        let rows = (0..<20).map { _ in session() }
        let visible = Set(rows.map(\.id))
        XCTAssertEqual(ChatQuickSwitch.slot(for: rows[8].id, in: rows, numbering: visible), 9)
        XCTAssertNil(ChatQuickSwitch.slot(for: rows[9].id, in: rows, numbering: visible))
    }

    func testNothingVisibleMeansNoNumbers() {
        // An EMPTY set says what it means. Falling back to numbering
        // everything here would put badges back under the glass.
        let rows = [session(), session()]
        XCTAssertNil(ChatQuickSwitch.slot(for: rows[0].id, in: rows, numbering: []))
        XCTAssertNil(ChatQuickSwitch.id(forSlot: 1, in: rows, numbering: []))
    }

    func testNoVisibilityInformationNumbersEverything() {
        // nil is not "nothing visible" — it is "nobody has measured yet",
        // which is the state before the first layout reports. Numbering from
        // the top is the honest answer there, and it is exactly the behaviour
        // of a sidebar with no overlay at all.
        let rows = [session(), session()]
        XCTAssertEqual(ChatQuickSwitch.slot(for: rows[0].id, in: rows, numbering: nil), 1)
        XCTAssertEqual(ChatQuickSwitch.id(forSlot: 2, in: rows, numbering: nil), rows[1].id)
    }

    // MARK: - ⇧⌘<digit>: range from where you are to the numbered row

    /// The composition both shortcut families run through. Two different lists
    /// are in play and mixing them up is the whole hazard: the DIGIT resolves
    /// against the numbered (visible) rows, while the RANGE runs over the full
    /// list — everything between the two ends is selected, including rows
    /// scrolled under the frost that never wore a badge.
    private func outcome(slot: Int, sessions: [ChatSession], numbering: Set<UUID>? = nil,
                         selection: Set<UUID> = [], anchor: UUID? = nil, active: UUID? = nil,
                         extend: Bool) -> SidebarMultiSelect.Outcome? {
        ChatQuickSwitch.outcome(slot: slot, sessions: sessions, numbering: numbering,
                                selection: selection, anchor: anchor, active: active,
                                extend: extend)
    }

    func testShiftRangesFromTheCurrentConversationToTheNumberedRow() {
        let rows = (0..<6).map { _ in session() }
        let result = outcome(slot: 4, sessions: rows, active: rows[0].id, extend: true)

        XCTAssertEqual(result?.selection, Set(rows[0...3].map(\.id)))
        XCTAssertEqual(result?.activate, rows[3].id, "you also move to the row you ranged to")
    }

    func testTheRangeCrossesRowsThatHaveNoBadge() {
        // Rows 0…2 are behind the frost, so only 3… are numbered. Ranging from
        // the active chat at row 0 to ⌘2 (= row 4) must still select 0…4:
        // a selection is a contiguous run of the LIST, not of the badges.
        let rows = (0..<8).map { _ in session() }
        let visible = Set(rows[3...].map(\.id))

        XCTAssertEqual(ChatQuickSwitch.id(forSlot: 2, in: rows, numbering: visible), rows[4].id)
        let result = outcome(slot: 2, sessions: rows, numbering: visible,
                             active: rows[0].id, extend: true)
        XCTAssertEqual(result?.selection, Set(rows[0...4].map(\.id)))
    }

    func testRangingBackwardsWorksTheSame() {
        let rows = (0..<6).map { _ in session() }
        let result = outcome(slot: 2, sessions: rows, anchor: rows[4].id, active: rows[4].id,
                             extend: true)
        XCTAssertEqual(result?.selection, Set(rows[1...4].map(\.id)))
    }

    func testTheAnchorOutranksTheActiveChat() {
        // Same rule as shift-clicking in the panel: repeated ranges re-range
        // from ONE origin instead of accumulating. After a previous range the
        // anchor is that origin, and the active chat is where it ended.
        let rows = (0..<8).map { _ in session() }
        let result = outcome(slot: 6, sessions: rows, anchor: rows[1].id, active: rows[4].id,
                             extend: true)
        XCTAssertEqual(result?.selection, Set(rows[1...5].map(\.id)))
        XCTAssertEqual(result?.anchor, rows[1].id, "the origin stays put")
    }

    func testWithNoAnchorAndNoActiveChatShiftJustGoesThere() {
        // Nothing to range FROM. Selecting the target alone beats doing
        // nothing, which would read as the shortcut being broken.
        let rows = (0..<4).map { _ in session() }
        let result = outcome(slot: 3, sessions: rows, extend: true)
        XCTAssertEqual(result?.selection, [rows[2].id])
        XCTAssertEqual(result?.activate, rows[2].id)
    }

    func testWithoutShiftItIsAPlainJump() {
        // ⌘<digit> replaces the selection even when several rows are selected.
        let rows = (0..<5).map { _ in session() }
        let result = outcome(slot: 2, sessions: rows,
                             selection: Set(rows.map(\.id)), anchor: rows[0].id,
                             active: rows[0].id, extend: false)
        XCTAssertEqual(result?.selection, [rows[1].id])
        XCTAssertEqual(result?.anchor, rows[1].id, "a plain jump re-anchors, so the NEXT shift ranges from here")
    }

    func testADigitWithNoRowDoesNothingEitherWay() {
        let rows = [session(), session()]
        XCTAssertNil(outcome(slot: 7, sessions: rows, active: rows[0].id, extend: true))
        XCTAssertNil(outcome(slot: 7, sessions: rows, active: rows[0].id, extend: false))
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
        XCTAssertTrue(source.contains("keyboardShortcut(key, modifiers: .command)"),
                      "each slot needs its ⌘-digit key equivalent")
        XCTAssertTrue(source.contains("keyboardShortcut(key, modifiers: [.command, .control])"),
                      "each slot needs its ⌃⌘-digit key equivalent — same digit, ranging instead of jumping")

        XCTAssertTrue(source.contains("ChatQuickSwitch.outcome("),
                      "both shortcut families must go through the one decision, which resolves the "
                      + "digit against the same numbering the badge was drawn from")

        // Badge and shortcut must read the SAME visibility set. If only one of
        // them filters, ⌘3 goes to a different row than the one wearing 3 —
        // silently, and only when the sidebar is scrolled.
        XCTAssertEqual(SourceScan.count("numbering: numberedRows", in: source), 2,
                       "both the badge and the ⌘-digit lookup pass the visible set")
    }


    func testTheRowProbeIsGatedOnTheCommandKey() {
        // A GeometryReader on every row is a preference write per row per
        // scroll frame. Outside ⌘-held nothing reads the answer, so nothing
        // should be measuring.
        let source = SourceScan.source("Views/ChatView.swift", from: #filePath)
        // `key:` pins this to the PUBLISHER — the plain type name also appears
        // on the reader and on the declaration.
        guard let probe = source.range(of: "key: SidebarRowSpansKey.self") else {
            return XCTFail("the row probe is gone")
        }
        let before = String(source[..<probe.lowerBound].suffix(400))
        XCTAssertTrue(before.contains("modifiers.commandHeld"),
                      "the row geometry probe must be attached only while ⌘ is down")
    }

    func testTheBadgeIsGatedOnTheCommandKey() {
        let source = SourceScan.source("Views/ChatView.swift", from: #filePath)
        XCTAssertTrue(source.contains("commandHeld"),
                      "badges appear only while ⌘ is down — otherwise they are permanent chrome")
    }
}
