import XCTest
@testable import MLXCore

/// The composer's five controls are bare glyphs — the only thing that says what
/// they do is the hover card, so its TEXT is the feature. Pure content tests:
/// the card view has no seam worth testing, the sentences do.
final class ComposerTipTests: XCTestCase {

    private var all: [ComposerTip] {
        [.agent(name: "Chef"), .agent(name: nil),
         .attachments(audioSupported: true), .attachments(audioSupported: false),
         .thinking(isOn: true), .thinking(isOn: false),
         .tools(isOn: true, workspace: "/tmp/w"), .tools(isOn: false, workspace: nil),
         .mcp(isOn: true), .mcp(isOn: false)]
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

    /// The workspace is what every file and shell call resolves against, and it
    /// is otherwise two clicks away inside the menu.
    func testToolsTipNamesTheWorkspaceOrSaysItIsUnset() {
        XCTAssertEqual(ComposerTip.tools(isOn: true, workspace: "/Users/x/proj").detail,
                       "Workspace: /Users/x/proj")
        let unset = try! XCTUnwrap(ComposerTip.tools(isOn: true, workspace: nil).detail)
        XCTAssertTrue(unset.lowercased().contains("not set"), unset)
    }

    func testAgentTipNamesTheAgentOrTheAppDefaults() {
        let chef = ComposerTip.agent(name: "Chef")
        XCTAssertTrue(chef.title.contains("Chef") || chef.body.contains("Chef"))
        let none = ComposerTip.agent(name: nil)
        XCTAssertFalse(none.body.contains("Chef"))
        XCTAssertTrue(none.body.lowercased().contains("app"), none.body)
    }

    /// Offering audio on a model that can't hear it is the dead-control class the
    /// media presets exist to avoid — the menu row is gated, so the card is too.
    func testAttachmentTipMentionsAudioOnlyWhenTheModelCanHearIt() {
        XCTAssertTrue(ComposerTip.attachments(audioSupported: true).body.lowercased().contains("audio"))
        XCTAssertFalse(ComposerTip.attachments(audioSupported: false).body.lowercased().contains("audio"))
    }

    /// A card that appears the instant the pointer crosses a disc flashes five
    /// times on the way to Send.
    func testHoverDelayIsLongEnoughToSurviveAPassingPointer() {
        XCTAssertGreaterThanOrEqual(ComposerTip.hoverDelay, 0.3)
        XCTAssertLessThanOrEqual(ComposerTip.hoverDelay, 0.8)
    }
}
