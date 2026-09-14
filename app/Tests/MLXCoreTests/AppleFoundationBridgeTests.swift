import XCTest
@testable import MLXCore

/// The pure half of the Apple Foundation Models route: our OpenAI-shaped
/// message dicts and tool JSON in, a transcript plan and a schema tree out.
/// The framework half (Transcript, DynamicGenerationSchema, the session) is a
/// mechanical translation of these values.
final class AppleFoundationBridgeTests: XCTestCase {

    func testSystemMessagesBecomeInstructionsAndTheLastUserIsThePrompt() {
        let plan = AppleFoundationBridge.plan(messages: [
            ["role": "system", "content": "You are terse."],
            ["role": "user", "content": "hi"],
            ["role": "assistant", "content": "hello"],
            ["role": "user", "content": "again"],
        ])
        XCTAssertEqual(plan.prompt, "again")
        XCTAssertEqual(plan.entries, [
            .instructions("You are terse."),
            .prompt("hi"),
            .response("hello"),
        ])
    }

    /// The agent loop re-requests after running a tool, so the tail is a tool
    /// RESULT, not a user turn — and the framework needs a prompt to answer.
    /// The results are that prompt; the call that produced them stays in the
    /// transcript.
    func testTrailingToolResultsBecomeThePrompt() {
        let plan = AppleFoundationBridge.plan(messages: [
            ["role": "user", "content": "what is in the file"],
            ["role": "assistant", "content": "", "tool_calls": [
                ["id": "c1", "type": "function",
                 "function": ["name": "readFile", "arguments": "{\"path\":\"a.txt\"}"]],
            ]],
            ["role": "tool", "tool_call_id": "c1", "content": "hello world"],
        ])
        XCTAssertEqual(plan.entries, [
            .prompt("what is in the file"),
            .toolCalls([AppleToolCallRef(id: "c1", name: "readFile",
                                         argumentsJSON: "{\"path\":\"a.txt\"}")]),
        ])
        XCTAssertTrue(plan.prompt.contains("readFile"), plan.prompt)
        XCTAssertTrue(plan.prompt.contains("hello world"), plan.prompt)
    }

    func testAssistantTextAndToolCallsBothSurvive() {
        let plan = AppleFoundationBridge.plan(messages: [
            ["role": "user", "content": "go"],
            ["role": "assistant", "content": "on it", "tool_calls": [
                ["id": "c1", "type": "function",
                 "function": ["name": "shell", "arguments": "{}"]],
            ]],
            ["role": "tool", "tool_call_id": "c1", "content": "ok"],
        ])
        XCTAssertEqual(plan.entries[1], .response("on it"))
        XCTAssertEqual(plan.entries[2], .toolCalls([
            AppleToolCallRef(id: "c1", name: "shell", argumentsJSON: "{}"),
        ]))
    }

    /// A multimodal user turn is an array of blocks; the on-device model is
    /// text-only, so the text parts join in order and the images drop.
    func testMultimodalContentJoinsItsTextParts() {
        let plan = AppleFoundationBridge.plan(messages: [
            ["role": "user", "content": [
                ["type": "text", "text": "look"],
                ["type": "image_url", "image_url": ["url": "data:…"]],
                ["type": "text", "text": "at this"],
            ]],
        ])
        XCTAssertEqual(plan.prompt, "look at this")
    }

    // MARK: - Tool schemas

    func testToolSchemaCarriesTypesRequirednessAndEnums() throws {
        let json = """
        [{"type":"function","function":{"name":"shell","description":"Run a command",
          "parameters":{"type":"object","properties":{
            "command":{"type":"string","description":"the command"},
            "mode":{"type":"string","enum":["fast","slow"]},
            "timeout":{"type":"integer"},
            "paths":{"type":"array","items":{"type":"string"}}},
          "required":["command"]}}}]
        """
        let tools = AppleFoundationBridge.tools(fromJSON: json)
        XCTAssertEqual(tools.count, 1)
        let t = try XCTUnwrap(tools.first)
        XCTAssertEqual(t.name, "shell")
        XCTAssertEqual(t.description, "Run a command")
        // Required first, then alphabetical: a JSON object has no order, and
        // the model must see the same tool the same way twice in a row.
        XCTAssertEqual(t.properties.map(\.name), ["command", "mode", "paths", "timeout"])
        XCTAssertFalse(t.properties[0].isOptional)
        XCTAssertTrue(t.properties[1].isOptional)
        XCTAssertEqual(t.properties[1].node, .string(choices: ["fast", "slow"]))
        XCTAssertEqual(t.properties[2].node, .array(.string(choices: [])))
        XCTAssertEqual(t.properties[3].node, .integer)
    }

    /// An unknown or missing type is a string, never a dropped parameter: a
    /// tool the model cannot fill in is worse than one typed loosely.
    func testUnknownParameterTypeFallsBackToString() {
        let tools = AppleFoundationBridge.tools(fromJSON: """
        [{"type":"function","function":{"name":"t","parameters":{"type":"object",
          "properties":{"x":{"description":"no type"}}}}}]
        """)
        XCTAssertEqual(tools.first?.properties.first?.node, .string(choices: []))
    }

    func testMalformedToolJSONYieldsNoTools() {
        XCTAssertTrue(AppleFoundationBridge.tools(fromJSON: "not json").isEmpty)
    }

    // MARK: - Streaming

    /// The framework yields CUMULATIVE snapshots; our stream events are deltas.
    func testCumulativeSnapshotsBecomeDeltas() {
        XCTAssertEqual(AppleFoundationBridge.delta(previous: "he", cumulative: "hello"), "llo")
        XCTAssertEqual(AppleFoundationBridge.delta(previous: "", cumulative: "hi"), "hi")
        XCTAssertEqual(AppleFoundationBridge.delta(previous: "hello", cumulative: "hello"), "")
        // A snapshot that is not an extension of the last one is emitted whole
        // rather than silently diffed into nonsense.
        XCTAssertEqual(AppleFoundationBridge.delta(previous: "abc", cumulative: "xyz"), "xyz")
    }
}

/// Where the on-device route changes what the rest of the app may do.
@MainActor
final class AppleFoundationGateTests: XCTestCase {

    /// It answers with no mlx-serve process, so the "server must be running"
    /// gate on a turn does not apply to it.
    func testATurnRunsWithTheServerDownOnTheOnDeviceModel() {
        XCTAssertTrue(ChatTurnEngine.canRunTurn(serverRunning: false, apple: true))
        XCTAssertFalse(ChatTurnEngine.canRunTurn(serverRunning: false, apple: false))
        XCTAssertTrue(ChatTurnEngine.canRunTurn(serverRunning: true, apple: false))
    }

    /// Nothing to start, so the red Start button has nothing to offer.
    func testStartControlHidesForTheOnDeviceModel() {
        XCTAssertEqual(ChatServerStartControl.resolve(status: .stopped, hasStartableModel: false),
                       .hidden)
    }

    /// The framework answers a prompt; it cannot extend a reply already
    /// written, so Continue is not offered.
    func testContinuationIsNotOfferedOnTheOnDeviceModel() {
        var msg = ChatMessage(role: .assistant, content: "half a sentence")
        msg.isStreaming = false
        XCTAssertFalse(ContinueReply.isEligible([msg], serverRunning: true, busy: false,
                                                apple: true))
        XCTAssertTrue(ContinueReply.isEligible([msg], serverRunning: true, busy: false,
                                               apple: false))
    }

    /// The on-device model's window is a fixed 4k shared between prompt and
    /// reply, and no API raises it — so the history budget must be ITS size,
    /// not the app's context setting, or every tool-bearing turn overflows.
    func testTheOnDeviceModelBudgetsAgainstItsOwnFixedWindow() {
        XCTAssertEqual(AgentEngine.effectiveContextLength(appContextSize: 131072,
                                                          modelContextLength: nil,
                                                          apple: true),
                       AppleFoundationChat.contextTokens)
        XCTAssertEqual(AgentEngine.effectiveContextLength(appContextSize: 131072,
                                                          modelContextLength: 40960,
                                                          apple: false),
                       40960)
    }

    /// The reply is drawn from the same 4k, so it takes a reservation rather
    /// than the app's max-tokens setting.
    func testTheReplyIsCappedInsideThatWindow() {
        XCTAssertLessThan(AppleFoundationChat.maxResponseTokens, AppleFoundationChat.contextTokens)
        XCTAssertEqual(AppleFoundationChat.responseTokens(requested: 65536),
                       AppleFoundationChat.maxResponseTokens)
        XCTAssertEqual(AppleFoundationChat.responseTokens(requested: 256), 256)
    }

    /// The on-device model has no thinking mode, and a 4k window the full tool
    /// blob cannot fit — so the resolver, the ONE place a turn's capabilities
    /// are decided, clamps it to browse + search whatever the surface asked
    /// for. Nothing downstream can hand it more.
    func testTheOnDeviceModelIsClampedToBrowseAndSearch() {
        var defaults = AppDefaultsSnapshot()
        defaults.toolsEnabled = true
        defaults.mcpEnabled = true
        defaults.thinkingEnabled = true
        defaults.appleModel = true
        let r = AgentResolution.resolve(agent: nil, defaults: defaults)
        XCTAssertEqual(r.tools, AppleFoundationChat.allowedTools)
        XCTAssertEqual(AppleFoundationChat.allowedTools, [.browse, .webSearch])
        XCTAssertFalse(r.thinkingEnabled)
        XCTAssertFalse(r.mcpEnabled)
        XCTAssertTrue(r.toolsEnabled)
    }

    /// Switching a tool off still works inside the clamp, and switching them
    /// all off stops the loop rather than advertising nothing to a loop that
    /// still runs.
    func testTheClampStillHonoursTheSurfacesOwnSwitches() {
        var defaults = AppDefaultsSnapshot()
        defaults.toolsEnabled = true
        defaults.appleModel = true
        defaults.disabledTools = [.browse]
        XCTAssertEqual(AgentResolution.resolve(agent: nil, defaults: defaults).tools, [.webSearch])

        defaults.disabledTools = [.browse, .webSearch]
        let off = AgentResolution.resolve(agent: nil, defaults: defaults)
        XCTAssertTrue(off.tools.isEmpty)
        XCTAssertFalse(off.toolsEnabled)
    }

    /// Nothing changes for every other model.
    func testAServedModelIsNotClamped() {
        var defaults = AppDefaultsSnapshot()
        defaults.toolsEnabled = true
        defaults.thinkingEnabled = true
        let r = AgentResolution.resolve(agent: nil, defaults: defaults)
        XCTAssertTrue(r.tools.contains(.shell))
        XCTAssertTrue(r.thinkingEnabled)
    }
}
