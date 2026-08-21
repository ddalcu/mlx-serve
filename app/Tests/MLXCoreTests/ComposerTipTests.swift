import XCTest
@testable import MLXCore

/// The composer's five controls are bare glyphs — the only thing that says what
/// they do is the hover card, so its TEXT is the feature. Pure content tests:
/// the card view has no seam worth testing, the sentences do.
final class ComposerTipTests: XCTestCase {

    private var all: [ComposerTip] {
        [.attachments(audioSupported: true), .attachments(audioSupported: false),
         .attachments(audioSupported: true, videoSupported: true), .attachments(audioSupported: false, videoSupported: true),
         .thinking(isOn: true), .thinking(isOn: false),
         .tools(isOn: true, workspace: "/tmp/w"), .tools(isOn: false, workspace: nil),
         .mcp(isOn: true), .mcp(isOn: false)]
            + lockedTips
    }

    private var lockedTips: [ComposerTip] {
        [.thinking(isOn: true, lockedBy: "Chef"),
         .tools(isOn: true, workspace: "/tmp/w", lockedBy: "Chef"),
         .mcp(isOn: false, lockedBy: "Chef")]
    }

    func testEveryTipHasATitleAndABody() {
        for tip in all {
            XCTAssertFalse(tip.title.trimmingCharacters(in: .whitespaces).isEmpty)
            XCTAssertFalse(tip.body.trimmingCharacters(in: .whitespaces).isEmpty)
            // The title is a label, not a sentence — it sits on one line in a
            // 280pt card and wrapping it looks like a bug.
            XCTAssertLessThanOrEqual(tip.title.count, 32, "title too long to sit on one line: \(tip.title)")
            // A hover card is a glance, not documentation — the first version
            // read as three sentences of prose and nobody hovers to read that.
            // Detail that needs a paragraph belongs in Settings or the menu.
            XCTAssertLessThanOrEqual(tip.body.count, 120, "card body is too long to glance at: \(tip.body)")
        }
    }

    /// A toggle's card must say what it IS and what a click would DO — those are
    /// opposites, and a card reading "ON … click to turn it on" is the classic
    /// way to make the user click twice and end up where they started.
    func testToggleTipsNameTheCurrentStateAndTheOppositeAction() {
        let cases: [(on: ComposerTip, off: ComposerTip)] = [
            (.thinking(isOn: true), .thinking(isOn: false)),
            (.tools(isOn: true, workspace: nil), .tools(isOn: false, workspace: nil)),
            (.mcp(isOn: true), .mcp(isOn: false)),
        ]
        for (on, off) in cases {
            XCTAssertTrue(on.title.contains("ON"), "\(on.title) must read as on")
            XCTAssertTrue(off.title.contains("OFF"), "\(off.title) must read as off")
            XCTAssertTrue(on.body.lowercased().contains("turn it off"), "on-state card must offer OFF: \(on.body)")
            XCTAssertTrue(off.body.lowercased().contains("turn it on"), "off-state card must offer ON: \(off.body)")
        }
    }

    /// Same reason the old `.help` strings named it: secondary-click is the only
    /// way into the per-tool switches and the Marketplace since click became the
    /// toggle, and nothing on screen announces it.
    func testToolsAndMcpTipsNameTheRightClickMenu() {
        for tip in [ComposerTip.tools(isOn: true, workspace: nil), .mcp(isOn: true)] {
            XCTAssertTrue(tip.body.lowercased().contains("right-click"),
                          "the secondary-click menu is undiscoverable unless the card names it: \(tip.body)")
        }
    }

    /// Same rule for the brain disc since its right-click grew the
    /// reasoning-effort picker.
    func testThinkingTipNamesTheRightClickMenu() {
        for tip in [ComposerTip.thinking(isOn: true), .thinking(isOn: false)] {
            XCTAssertTrue(tip.body.lowercased().contains("right-click"),
                          "the effort picker is undiscoverable unless the card names it: \(tip.body)")
        }
    }

    /// The workspace is what every file and shell call resolves against, and it
    /// is otherwise two clicks away inside the menu.
    func testToolsTipNamesTheWorkspaceOrSaysItIsUnset() {
        XCTAssertEqual(ComposerTip.tools(isOn: true, workspace: "/Users/x/proj").detail,
                       "Workspace: /Users/x/proj")
        let unset = try! XCTUnwrap(ComposerTip.tools(isOn: true, workspace: nil).detail)
        XCTAssertTrue(unset.lowercased().contains("not set"), unset)
    }

    /// Offering audio on a model that can't hear it is the dead-control class the
    /// media presets exist to avoid — the menu row is gated, so the card is too.
    func testAttachmentTipMentionsAudioOnlyWhenTheModelCanHearIt() {
        XCTAssertTrue(ComposerTip.attachments(audioSupported: true).body.lowercased().contains("audio"))
        XCTAssertFalse(ComposerTip.attachments(audioSupported: false).body.lowercased().contains("audio"))
    }

    /// Same dead-control class as audio, for video (Qwen3-VL-family only).
    func testAttachmentTipMentionsVideoOnlyWhenTheModelCanSeeIt() {
        XCTAssertTrue(ComposerTip.attachments(audioSupported: false, videoSupported: true).body.lowercased().contains("video"))
        XCTAssertFalse(ComposerTip.attachments(audioSupported: false, videoSupported: false).body.lowercased().contains("video"))
        // Both together — neither offer crowds out the other.
        let both = ComposerTip.attachments(audioSupported: true, videoSupported: true).body.lowercased()
        XCTAssertTrue(both.contains("audio") && both.contains("video"))
    }

    /// A card that appears the instant the pointer crosses a disc flashes five
    /// times on the way to Send.
    func testHoverDelayIsLongEnoughToSurviveAPassingPointer() {
        XCTAssertGreaterThanOrEqual(ComposerTip.hoverDelay, 0.3)
        XCTAssertLessThanOrEqual(ComposerTip.hoverDelay, 0.8)
    }

    // MARK: - Locked by an agent

    /// A locked control still reads its STATE from the title (that's what the
    /// disc's colour is saying), but the body must name who decided it and where
    /// to change it — offering "click to turn it on" on a control that can't be
    /// clicked is the dead-offer class.
    func testLockedTipsNameTheAgentAndWhereToChangeIt() {
        for tip in lockedTips {
            XCTAssertTrue(tip.body.contains("Chef"), "a locked card must name the agent: \(tip.body)")
            XCTAssertTrue(tip.body.lowercased().contains("agent"), tip.body)
            XCTAssertFalse(tip.body.lowercased().contains("click to turn it"),
                           "a locked control can't be toggled — don't offer it: \(tip.body)")
        }
    }

    func testLockedTipsStillSayWhetherTheControlIsOnOrOff() {
        XCTAssertTrue(ComposerTip.thinking(isOn: true, lockedBy: "Chef").title.contains("ON"))
        XCTAssertTrue(ComposerTip.mcp(isOn: false, lockedBy: "Chef").title.contains("OFF"))
    }

    /// The right-click menu is gone while locked (there is nothing to configure
    /// from here), so the card must stop advertising it.
    func testLockedToolsAndMcpTipsDoNotAdvertiseTheRightClickMenu() {
        for tip in [ComposerTip.tools(isOn: true, workspace: nil, lockedBy: "Chef"),
                    .mcp(isOn: true, lockedBy: "Chef")] {
            XCTAssertFalse(tip.body.lowercased().contains("right-click"), tip.body)
        }
    }

    /// The workspace comes from the agent while one is selected, but it's still
    /// what every file and shell call resolves against — keep naming it.
    func testLockedToolsTipStillNamesTheWorkspace() {
        XCTAssertEqual(ComposerTip.tools(isOn: true, workspace: "/w", lockedBy: "Chef").detail,
                       "Workspace: /w")
    }

    // MARK: - Dismissal
    //
    // The card is an overlay with no hit-testing, so nothing dismisses it except
    // the pointer leaving — and opening a menu over the composer does NOT deliver
    // a hover-exit. The card then sits under the open menu until you hover the
    // control again.

    func testAPendingRevealIsCancelledByADismiss() {
        var state = ComposerTipHoverState()
        let token = state.hoverBegan()
        state.dismiss()
        // The delayed reveal fires AFTER the menu opened — it must not put the
        // card back up on top of it.
        XCTAssertFalse(state.reveal(token: token))
        XCTAssertFalse(state.shown)
    }

    func testAPendingRevealIsCancelledByThePointerLeaving() {
        var state = ComposerTipHoverState()
        let token = state.hoverBegan()
        state.hoverEnded()
        XCTAssertFalse(state.reveal(token: token))
        XCTAssertFalse(state.shown)
    }

    func testARevealThatIsStillCurrentShowsTheCard() {
        var state = ComposerTipHoverState()
        let token = state.hoverBegan()
        XCTAssertTrue(state.reveal(token: token))
        XCTAssertTrue(state.shown)
    }

    func testDismissHidesACardThatIsAlreadyUp() {
        var state = ComposerTipHoverState()
        _ = state.reveal(token: state.hoverBegan())
        XCTAssertTrue(state.shown)
        state.dismiss()
        XCTAssertFalse(state.shown)
    }

    /// Re-entering after a dismiss must work — a stale token can't be allowed to
    /// permanently wedge the control's card off.
    func testHoveringAgainAfterADismissStillShowsTheCard() {
        var state = ComposerTipHoverState()
        _ = state.hoverBegan()
        state.dismiss()
        XCTAssertTrue(state.reveal(token: state.hoverBegan()))
    }
}
