import XCTest
@testable import MLXCore

/// The Agents panel was rebuilt to a mockup. Everything it could do before it
/// still has to do — a redesign is where affordances disappear quietly, because
/// nothing fails when a control is simply no longer drawn.
final class AgentsPanelTests: XCTestCase {

    private func source(_ relativePath: String = "Sources/MLXServe/Views/AgentsWindow.swift") throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// Every affordance the old editor had. Named individually so a failure
    /// says WHICH one went missing rather than "the panel changed".
    func testEveryEditorAffordanceSurvivedTheRedesign() throws {
        let s = try source()
        let required: [String: String] = [
            "onStartChat": "start a chat as this agent",
            "onDuplicate": "duplicate",
            "onDelete": "delete",
            "onWrite": "write the prompt from the description",
            "$agent.name": "rename",
            "$agent.brief": "the description",
            "$agent.systemPrompt": "the prompt itself",
            "agent.symbol": "the symbol picker",
            "wakePhrase": "the wake phrase",
            "voiceRows": "the voice picker",
            "capabilitiesSection": "capabilities",
            "modelSection": "the pinned model",
            "workspaceSection": "the workspace",
            "samplingSection": "sampling",
            "moreOptionsSection": "the progressive-disclosure row",
        ]
        for (needle, what) in required {
            XCTAssertTrue(s.contains(needle), "the redesign dropped \(what) (`\(needle)`)")
        }
    }

    /// Delete is HIDDEN on a built-in, not shown disabled: it can only fail
    /// there, and a dead control is worse than an absent one. Duplicate stays,
    /// because copying a starter is how you make one yours.
    func testABuiltInOffersDuplicateButNotDelete() throws {
        let s = try source()
        guard let start = s.range(of: "private var agentActions: some ToolbarContent"),
              let end = s.range(of: "private var startChatButton",
                                range: start.upperBound..<s.endIndex) else {
            return XCTFail("the agent toolbar actions moved — update this audit")
        }
        let actions = String(s[start.upperBound..<end.lowerBound])
        XCTAssertTrue(actions.contains("if !readOnly"),
                      "Delete must be gated on the agent being editable")
        // Duplicate sits OUTSIDE that gate.
        guard let dup = actions.range(of: "onDuplicate"),
              let gate = actions.range(of: "if !readOnly") else {
            return XCTFail("expected both a Duplicate action and a readOnly gate")
        }
        XCTAssertLessThan(dup.lowerBound, gate.lowerBound,
                          "Duplicate must be offered on built-ins too")
    }

    /// The create control offers the TYPES, and there is exactly one of them —
    /// the list's own row. A second in the toolbar would be two create routes
    /// side by side.
    func testOneCreateRouteAndItOffersTheTypes() throws {
        let s = try source()
        XCTAssertTrue(s.contains("Create New Agent"), "the list needs its create row")
        XCTAssertTrue(s.contains("newAgentMenuItems"), "the create row offers the agent types")
        XCTAssertTrue(s.contains("Blank agent"), "…and a blank one at the bottom")
        XCTAssertFalse(s.contains("private var newAgentMenu:"),
                       "the toolbar's + is gone; the row replaced it")
    }

    /// Picking a type COPIES the starter into a new editable identity. Landing
    /// on the starter itself would be a form that silently discards what you
    /// type into it, since `commit` returns early on a built-in.
    func testCreatingFromATypeCopiesRatherThanSelectsTheStarter() throws {
        let s = try source()
        guard let start = s.range(of: "private func newAgent(basedOn"),
              let end = s.range(of: "\n    }", range: start.upperBound..<s.endIndex) else {
            return XCTFail("the create helper moved — update this audit")
        }
        let body = String(s[start.upperBound..<end.lowerBound])
        XCTAssertTrue(body.contains("store.add("), "a new agent is added to the store")
        XCTAssertTrue(body.contains("starter?.systemPrompt"), "the type's prompt is copied")
        XCTAssertTrue(body.contains("starter.capabilities"), "…and its capabilities")
    }

    /// The row's own start-chat control is a real Button beside the row's
    /// button, not a gesture on the row: on macOS a parent tap gesture
    /// swallows a child button's clicks silently.
    func testTheRowStartChatIsARealButton() throws {
        let s = try source()
        // Bounded to the row's own struct: the editor further down has a
        // legitimate tap gesture on its "More options" LABEL (macOS hit-tests
        // only a DisclosureGroup's chevron), and an unbounded scan forbids it.
        guard let start = s.range(of: "private struct AgentListRow") else {
            return XCTFail("the agent row moved — update this audit")
        }
        let after = s[start.upperBound...]
        let end = after.range(of: "\nprivate struct ") ?? after.range(of: "\nstruct ")
        let row = String(after[..<(end?.lowerBound ?? after.endIndex)])
        XCTAssertTrue(row.contains("Button(action: startChat)"),
                      "the per-row start-chat must be a Button")
        XCTAssertFalse(row.contains(".onTapGesture"),
                       "a tap gesture around the row would eat the child button's clicks")
    }
}
