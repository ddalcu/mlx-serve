import XCTest
@testable import MLXCore

/// The one-line summary beside the Agents editor's collapsed "More options".
///
/// Collapsing Capabilities / Model / Workspace / Sampling hides settings that
/// may be SET, and a hidden non-default is a setting the user can't discover
/// they turned on. The summary is the mitigation: it names every collapsed area
/// that differs from the defaults, so the row itself says whether opening it
/// would show anything.
final class AgentAdvancedSummaryTests: XCTestCase {

    private func summary(_ agent: Agent) -> String? {
        AgentAdvancedSummary.text(for: agent)
    }

    private func plainAgent() -> Agent {
        Agent(name: "A", systemPrompt: "")
    }

    func testAnUntouchedAgentSummarizesAsNothing() {
        // Nothing to advertise — the row shouldn't imply there is.
        XCTAssertNil(summary(plainAgent()))
    }

    func testToolsAndMcpAreNamed() {
        var agent = plainAgent()
        agent.capabilities = AgentCapabilities(tools: true, mcp: true, web: true)
        XCTAssertEqual(summary(agent), "Tools, MCP")
    }

    func testWebOffIsWorthNamingBecauseItIsOnByDefault() {
        // Web defaults ON, so switching it off is a change the user made and
        // would otherwise be invisible behind the collapsed row.
        var agent = plainAgent()
        agent.capabilities = AgentCapabilities(tools: false, mcp: false, web: false)
        XCTAssertEqual(summary(agent), "No web")
    }

    func testAdvancedToolSelectionIsNamed() {
        var agent = plainAgent()
        agent.capabilities = AgentCapabilities(tools: true, mcp: false, web: true,
                                               advancedTools: [.shell, .readFile])
        XCTAssertEqual(summary(agent), "Tools, custom tools")
    }

    func testPinnedModelAndWorkspaceAreNamed() {
        var agent = plainAgent()
        agent.modelPath = "/models/qwen"
        agent.workingDirectory = "/Users/me/project"
        XCTAssertEqual(summary(agent), "Model, workspace")
    }

    func testSamplingOverridesAreNamedOnce() {
        // Temperature and max tokens are one concept in the UI; naming both
        // makes the line long for no extra information.
        var agent = plainAgent()
        agent.temperature = 0.2
        XCTAssertEqual(summary(agent), "Sampling")
        agent.maxTokens = 512
        XCTAssertEqual(summary(agent), "Sampling")
    }

    func testEverySamplingKnobCountsTowardTheSummary() {
        // Each knob alone must light the row — a set-but-unnamed override is
        // exactly the hidden-setting hazard the summary exists to prevent.
        for mutate in [
            { (a: inout Agent) in a.topP = 0.8 },
            { (a: inout Agent) in a.topK = 40 },
            { (a: inout Agent) in a.repeatPenalty = 1.1 },
            { (a: inout Agent) in a.presencePenalty = 0.5 },
            { (a: inout Agent) in a.reasoningBudget = 1024 },
        ] {
            var agent = plainAgent()
            mutate(&agent)
            XCTAssertEqual(summary(agent), "Sampling")
        }
    }

    func testAreasAreListedInTheOrderTheSectionsAppear() {
        var agent = plainAgent()
        agent.capabilities = AgentCapabilities(tools: true, mcp: false, web: true)
        agent.modelPath = "/models/qwen"
        agent.workingDirectory = "/tmp/w"
        agent.temperature = 0.3
        XCTAssertEqual(summary(agent), "Tools, model, workspace, sampling")
    }

    func testEmptyStringsAreNotOverrides() {
        // A cleared field decodes as "" from an older build; treating it as a
        // pin would claim a customization that isn't there.
        var agent = plainAgent()
        agent.modelPath = ""
        agent.workingDirectory = "   "
        XCTAssertNil(summary(agent))
    }

    func testThinkingAndApprovalOverridesCount() {
        // Both live in the collapsed Capabilities section and are tri-state
        // (nil = app default), so a set value is a hidden change.
        var agent = plainAgent()
        agent.enableThinking = true
        XCTAssertEqual(summary(agent), "Thinking")
        agent.autoApproveTools = false
        XCTAssertEqual(summary(agent), "Thinking, tool approval")
    }
}
