import XCTest
@testable import MLXCore

/// Per-chat tool switches, applied at the ONE resolution chokepoint.
///
/// The menu can only take tools AWAY. Letting it grant one would mean a chat tab
/// could hand an agent a capability its own settings forbid, which is the whole
/// point of the capability list — so the disable set subtracts from whatever the
/// agent (or the app defaults) already allowed, and never adds.
final class SessionToolDisableTests: XCTestCase {

    private func resolve(agent: Agent? = nil, disabled: Set<AgentToolKind>,
                         toolsEnabled: Bool = true) -> ResolvedAgentSettings {
        var defaults = AppDefaultsSnapshot()
        defaults.toolsEnabled = toolsEnabled
        defaults.disabledTools = disabled
        return AgentResolution.resolve(agent: agent, defaults: defaults)
    }

    // MARK: - Subtraction

    func testDisabledToolsAreRemovedWithNoAgent() {
        let r = resolve(disabled: [.shell, .browse])
        XCTAssertFalse(r.tools.contains(.shell))
        XCTAssertFalse(r.tools.contains(.browse))
        XCTAssertTrue(r.tools.contains(.readFile), "untouched tools stay available")
    }

    func testEmptyDisableSetReproducesTodaysBehaviourExactly() {
        // The upgrade guarantee: a user who never opens the menu must get a
        // byte-identical resolution.
        XCTAssertEqual(AgentResolution.resolve(agent: nil, defaults: AppDefaultsSnapshot()),
                       resolve(disabled: [], toolsEnabled: false))
    }

    /// Web-only agent: the loop's own tools are off, the web pair is on.
    private func webOnlyAgent() -> Agent {
        Agent(name: "Researcher", systemPrompt: "",
              capabilities: AgentCapabilities(tools: false, mcp: false, web: true))
    }

    func testDisablingCannotGrantAToolTheAgentForbids() {
        // A web-only agent stays web-only no matter what the chat's menu says —
        // subtraction can't add.
        let agent = webOnlyAgent()
        XCTAssertFalse(resolve(agent: agent, disabled: []).tools.contains(.shell))
        XCTAssertFalse(resolve(agent: agent, disabled: [.readFile]).tools.contains(.shell))
    }

    func testDisableAppliesOnTopOfAnAgentsCapabilities() {
        let r = resolve(agent: webOnlyAgent(), disabled: [.webSearch])
        XCTAssertFalse(r.tools.contains(.webSearch), "the chat's own switch still applies to an agent")
        XCTAssertTrue(r.tools.contains(.browse))
    }

    // MARK: - searchDocuments is not a capability

    func testSearchDocumentsSurvivesTheDisableSet() {
        // Its real gate is whether a folder is attached, which is stronger than
        // any toggle — docs-only chats have Tools off and still need it. It is
        // deliberately absent from the menu, so nothing can switch it off here.
        let r = resolve(disabled: Set(AgentToolKind.allCases))
        XCTAssertTrue(r.tools.contains(.searchDocuments))
    }

    func testMenuNeverOffersSearchDocuments() {
        XCTAssertFalse(AgentToolKind.chatToggleable.contains(.searchDocuments))
    }

    func testEveryOtherToolIsReachableFromTheMenu() {
        // Class guard: a tool with no menu row can never be switched off, which
        // is a silent hole in the feature — the same shape as a tool in the JSON
        // with no `AgentToolKind` case.
        let offered = Set(AgentToolKind.chatToggleable)
        let expected = Set(AgentToolKind.allCases).subtracting([.searchDocuments])
        XCTAssertEqual(offered, expected)
    }

    func testEveryToggleableToolBelongsToExactlyOneMenuGroup() {
        // The menu renders groups; a tool in none of them is invisible, and one
        // in two renders twice with two independent-looking checkmarks.
        for tool in AgentToolKind.chatToggleable {
            let groups = AgentToolGroup.allCases.filter { $0.tools.contains(tool) }
            XCTAssertEqual(groups.count, 1, "\(tool.rawValue) is in \(groups.count) groups")
        }
    }

    // MARK: - Turning everything off

    func testDisablingEveryToolStopsTheLoopRunning() {
        // Advertising nothing while still running the tool loop is a dead
        // offer — the turn should just be plain chat.
        let r = resolve(disabled: Set(AgentToolKind.chatToggleable), toolsEnabled: true)
        XCTAssertFalse(r.toolsEnabled)
    }

    func testDisablingSomeToolsLeavesTheLoopOn() {
        XCTAssertTrue(resolve(disabled: [.shell], toolsEnabled: true).toolsEnabled)
    }

    func testDisableSetCannotSwitchTheLoopOn() {
        // Tools pill off means off, whatever the per-tool menu holds.
        XCTAssertFalse(resolve(disabled: [], toolsEnabled: false).toolsEnabled)
    }

    // MARK: - Persistence

    func testDisabledToolsRoundTripOnTheSession() throws {
        var session = ChatSession(title: "t")
        session.disabledTools = ["shell", "browse"]
        let back = try JSONDecoder().decode(ChatSession.self, from: JSONEncoder().encode(session))
        XCTAssertEqual(Set(back.disabledTools), ["shell", "browse"])
    }

    func testSessionsSavedBeforeTheMenuExistedHaveNothingDisabled() throws {
        let json = #"""
        {"id":"\#(UUID().uuidString)","title":"t","messages":[],"createdAt":0,"updatedAt":0,
         "mode":"chat","isExternalBridge":false,"enableThinking":false,"useMCP":false}
        """#
        let back = try JSONDecoder().decode(ChatSession.self, from: Data(json.utf8))
        XCTAssertTrue(back.disabledTools.isEmpty)
    }

    func testUnknownToolNamesOnDiskAreIgnored() throws {
        // A tool removed in a later build leaves its raw value in saved sessions;
        // it must not crash or disable something else by accident.
        var session = ChatSession(title: "t")
        session.disabledTools = ["shell", "toolThatNoLongerExists"]
        let resolved = ChatSession.disabledToolKinds(session.disabledTools)
        XCTAssertEqual(resolved, [.shell])
    }
}
