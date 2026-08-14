import XCTest
@testable import MLXCore

/// The conversation list's modifier-aware selection maths.
///
/// The list stopped being a `List`, and with it went the cmd/shift behaviour a
/// `selection:` binding gives you for free — `SidebarMultiSelect` is that
/// behaviour written out by hand. It is pure precisely so it can be driven from
/// here: every rule below is invisible until someone shift-clicks across the
/// Agents/Chats heading or cmd-clicks the row they were reading, and by then
/// the panel has already done the wrong thing.
final class SidebarMultiSelectTests: XCTestCase {

    // Four rows in panel order: two agent threads, then two plain chats. The
    // heading between b and c is a heading, not a wall — `ordered` is the two
    // sections flattened, which is what makes a range able to cross it.
    private let a = UUID(), b = UUID(), c = UUID(), d = UUID()
    private var ordered: [UUID] { [a, b, c, d] }

    private func click(_ id: UUID,
                       selection: Set<UUID>,
                       anchor: UUID? = nil,
                       active: UUID? = nil,
                       command: Bool = false,
                       shift: Bool = false) -> SidebarMultiSelect.Outcome {
        SidebarMultiSelect.click(id, ordered: ordered, selection: selection,
                                 anchor: anchor, active: active,
                                 command: command, shift: shift)
    }

    // MARK: - Plain click

    func testAPlainClickReplacesTheSelectionAndMovesTheAnchor() {
        let out = click(c, selection: [a, b], anchor: a, active: a)
        XCTAssertEqual(out.selection, [c])
        XCTAssertEqual(out.anchor, c)
        XCTAssertEqual(out.activate, c)
    }

    // MARK: - Shift range

    /// The reason `ordered` is both sections flattened: a range that stopped at
    /// the Agents/Chats heading would make half the panel unreachable by shift.
    func testShiftRangesAcrossTheAgentsChatsBoundary() {
        let out = click(d, selection: [b], anchor: b, active: b, shift: true)
        XCTAssertEqual(out.selection, [b, c, d])
        XCTAssertEqual(out.activate, d)
    }

    func testShiftRangesUpwardsToo() {
        let out = click(a, selection: [c], anchor: c, active: c, shift: true)
        XCTAssertEqual(out.selection, [a, b, c])
    }

    /// The range REPLACES the selection and leaves the anchor put, so dragging
    /// a shift-click up and down re-ranges from one origin instead of
    /// accumulating every range it passed through.
    func testShiftReRangesFromTheSameAnchorInsteadOfAccumulating() {
        let first = click(d, selection: [b], anchor: b, active: b, shift: true)
        XCTAssertEqual(first.anchor, b)
        let second = click(c, selection: first.selection, anchor: first.anchor,
                           active: d, shift: true)
        XCTAssertEqual(second.selection, [b, c], "d should have been dropped, not kept")
        XCTAssertEqual(second.anchor, b)
    }

    func testShiftWinsWhenBothModifiersAreHeld() {
        let out = click(c, selection: [a], anchor: a, active: a,
                        command: true, shift: true)
        XCTAssertEqual(out.selection, [a, b, c])
    }

    /// With nothing to range FROM there is no range — it degrades to the plain
    /// replace rather than selecting some arbitrary span.
    func testShiftWithNoAnchorIsAPlainClick() {
        let out = click(c, selection: [a], anchor: nil, active: a, shift: true)
        XCTAssertEqual(out.selection, [c])
        XCTAssertEqual(out.anchor, c)
    }

    /// A row that is no longer in the list (deleted from the tray, another
    /// window) can't anchor a range — the panel must not go blank over it.
    func testShiftFromAStaleAnchorIsAPlainClick() {
        let out = click(c, selection: [a], anchor: UUID(), active: a, shift: true)
        XCTAssertEqual(out.selection, [c])
    }

    // MARK: - Cmd click

    func testCmdClickAddsARowWithoutDisturbingTheRest() {
        let out = click(c, selection: [a], anchor: a, active: a, command: true)
        XCTAssertEqual(out.selection, [a, c])
        XCTAssertEqual(out.anchor, c)
        XCTAssertEqual(out.activate, c)
    }

    func testCmdClickDeselectsARowThatWasAlreadySelected() {
        let out = click(c, selection: [a, c], anchor: a, active: a, command: true)
        XCTAssertEqual(out.selection, [a])
    }

    /// Deselecting a row you were NOT reading changes what is selected without
    /// changing where you are — `activate: nil` is how the wiring is told to
    /// leave the detail column alone.
    func testDeselectingSomeOtherRowLeavesTheTranscriptWhereItIs() {
        let out = click(c, selection: [a, c], anchor: a, active: a, command: true)
        XCTAssertNil(out.activate)
    }

    /// Cmd-clicking the LAST selected row is a no-op: this selection is also
    /// the panel's "you are here", and emptying it would leave a transcript on
    /// screen with nothing in the list pointing at it.
    func testCmdClickingTheOnlySelectedRowKeepsIt() {
        let out = click(a, selection: [a], anchor: a, active: a, command: true)
        XCTAssertEqual(out.selection, [a])
        XCTAssertNil(out.activate)
    }

    // MARK: - The nearest survivor

    /// Deselecting the row you were READING moves to the nearest survivor —
    /// otherwise the transcript on screen belongs to a row that is no longer lit.
    func testDeselectingTheRowYouAreReadingMovesToTheNearestSurvivor() {
        // Reading c; b and d both survive, and b is the nearer of the two by
        // panel distance (|1-2| == 1 vs |3-2| == 1 — the first minimum wins,
        // which is the row ABOVE, the same direction a list would move).
        let out = click(c, selection: [b, c, d], anchor: b, active: c, command: true)
        XCTAssertEqual(out.selection, [b, d])
        XCTAssertNotNil(out.activate)
        XCTAssertEqual(out.activate, b)
    }

    func testTheNearestSurvivorIsMeasuredInPanelOrderNotSetOrder() {
        // Reading a; d is adjacent in the set's iteration order about as often
        // as not, but c is three rows away and b is one.
        let out = click(a, selection: [a, b, d], anchor: a, active: a, command: true)
        XCTAssertEqual(out.activate, b)
    }

    // MARK: - Typing collapses the selection

    /// Clicking into the composer is a statement that you are working in ONE
    /// conversation. Without that, a multi-selection stayed lit behind the
    /// field you were typing in — and since a multi-selection outranks focus
    /// for ⌘⌫ (`ChatDeleteShortcut.route`), the chord kept raising a delete
    /// dialog mid-message: the two rules each read a true fact and disagreed
    /// about which one meant "the user is deleting chats".
    func testTakingTheKeyboardIntoTheComposerCollapsesTheSelection() {
        let typing = UUID(), other1 = UUID(), other2 = UUID()
        XCTAssertEqual(
            SidebarMultiSelect.focusingComposer(in: typing, selection: [typing, other1, other2]),
            [typing])
    }

    /// The collapse keeps the chat you are IN, never the anchor or the first
    /// of the set — you are typing into that one, and its transcript is on
    /// screen above the field.
    func testTheChatYouAreTypingInIsTheOneThatSurvives() {
        let typing = UUID(), other = UUID()
        XCTAssertEqual(SidebarMultiSelect.focusingComposer(in: typing, selection: [other]),
                       [typing])
    }

    /// Nothing to do — and it must say so rather than write an equal set back,
    /// which publishes a change on every focus event for no reason.
    func testASelectionThatAlreadyNamesOnlyThatChatIsLeftAlone() {
        let typing = UUID()
        XCTAssertNil(SidebarMultiSelect.focusingComposer(in: typing, selection: [typing]))
    }

    func testAnEmptySelectionBecomesTheChatYouAreTypingIn() {
        let typing = UUID()
        XCTAssertEqual(SidebarMultiSelect.focusingComposer(in: typing, selection: []), [typing])
    }

    /// The rule needs a caller, and the caller has to be the composer's own
    /// focus mirror — that is the one signal that means "the keyboard is in
    /// the field now", whether it got there by a click or by a finished turn
    /// handing it back.
    func testTheComposerCollapsesTheSelectionWhenItTakesTheKeyboard() throws {
        let source = SourceScan.source("Views/ChatView.swift", from: #filePath)
        XCTAssertTrue(source.contains("SidebarMultiSelect.focusingComposer"), """
            nothing collapses the sidebar selection when the composer takes \
            the keyboard, so ⌘⌫ still raises a delete dialog while typing with \
            several chats selected.
            """)
    }
}

/// Which deletions stop and ask first.
///
/// Nothing is visible on screen until the conversations are already gone, so
/// the rule is pinned here rather than by trying it.
final class SidebarDeleteConfirmTests: XCTestCase {

    /// ⌘⌫ names no row: its target is either every selected row or, with
    /// nothing selected, whichever chat you happen to be reading. Both are
    /// implicit.
    func testTheDeleteKeyAlwaysAsksEvenForOneChat() {
        XCTAssertTrue(SidebarDeleteConfirm.required(count: 1, keyboard: true))
        XCTAssertTrue(SidebarDeleteConfirm.required(count: 9, keyboard: true))
    }

    // MARK: - What ⌘⌫ acts on

    func testTheKeyboardDeleteTargetsTheWholeSelection() {
        let a = UUID(), b = UUID()
        XCTAssertEqual(SidebarDeleteConfirm.target(selection: [a, b], activeChatId: a),
                       [a, b])
    }

    /// With nothing selected the key falls back to the chat on screen — the
    /// only thing it could sensibly mean.
    func testWithNothingSelectedItTargetsTheChatYouAreReading() {
        let a = UUID()
        XCTAssertEqual(SidebarDeleteConfirm.target(selection: [], activeChatId: a), [a])
    }

    /// Nothing selected AND nothing on screen is nothing to delete. Nil is what
    /// disables the menu item — a command that does nothing when you pick it is
    /// worse than one that isn't offered.
    func testNothingSelectedAndNoActiveChatIsNoTarget() {
        XCTAssertNil(SidebarDeleteConfirm.target(selection: [], activeChatId: nil))
    }

    /// The selection WINS over the active chat: cmd-clicking three rows and
    /// hitting ⌘⌫ must not quietly delete only the one you were reading.
    func testTheSelectionOutranksTheActiveChat() {
        let a = UUID(), b = UUID(), reading = UUID()
        XCTAssertEqual(SidebarDeleteConfirm.target(selection: [a, b], activeChatId: reading),
                       [a, b])
    }

    /// A click on a named row is deliberate and stays immediate — a confirmation
    /// on every single delete is the dialog people learn to dismiss unread.
    func testASingleClickedRowGoesStraightThrough() {
        XCTAssertFalse(SidebarDeleteConfirm.required(count: 1, keyboard: false))
    }

    /// Bulk asks whichever control started it: N conversations go on one action
    /// and nothing undoes it.
    func testBulkAsksEvenFromAMenu() {
        XCTAssertTrue(SidebarDeleteConfirm.required(count: 2, keyboard: false))
    }

    func testNothingToDeleteAsksNothing() {
        XCTAssertFalse(SidebarDeleteConfirm.required(count: 0, keyboard: true))
        XCTAssertFalse(SidebarDeleteConfirm.required(count: 0, keyboard: false))
    }

    /// The count is what the user has to check before agreeing, so it is in the
    /// title rather than in a body nobody reads.
    func testTheTitleNamesTheCount() {
        XCTAssertEqual(SidebarDeleteConfirm.title(count: 1), "Delete this chat?")
        XCTAssertEqual(SidebarDeleteConfirm.title(count: 4), "Delete 4 chats?")
    }
}
