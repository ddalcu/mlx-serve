import XCTest
import AppKit
@testable import MLXCore

final class ChatGenerationSettingsTests: XCTestCase {
    func testSessionEditorCoversSamplingWhileThinkingKeepsOneComposerControl() {
        XCTAssertEqual(Set(GenerationField.clientFields), Set(GenerationField.allCases).subtracting([.thinking, .effort]))
        XCTAssertFalse(GenerationField.temperature.clientHelp.contains("Ignore client override"))
        XCTAssertFalse(GenerationField.budget.clientHelp.contains("locked"))
        XCTAssertEqual(ComposerTip.generationSettings.title, "Chat generation settings")
        XCTAssertLessThanOrEqual(ComposerTip.generationSettings.body.count, 120)
    }

    func testInheritanceUmbrellaTracksOffOnMixedAndPreservesExistingValues() {
        var profile = GenerationDefaults()
        XCTAssertEqual(profile.inheritanceState(), true)
        profile.setInherited(false, field: .temperature)
        XCTAssertNil(profile.inheritanceState())
        profile.rules["temperature"] = .init(value: .number(0.25), ignoreClient: true)
        profile.setAllInherited(false)
        XCTAssertEqual(profile.inheritanceState(), false)
        XCTAssertEqual(profile.rules["temperature"]?.value, .number(0.25))
        XCTAssertTrue(profile.rules["temperature"]!.ignoreClient)
        profile.setAllInherited(true)
        XCTAssertEqual(profile.inheritanceState(), true)
        XCTAssertTrue(profile.rules.isEmpty)
    }

    @MainActor
    func testNativeUmbrellasRenderMixedStateAndClickSelectsAll() {
        var clicked: Bool?
        let button = MixedCheckboxButton(title: "Default", value: nil) { clicked = $0 }
        XCTAssertEqual(button.state, .mixed)
        XCTAssertTrue(button.allowsMixedState)
        button.performClick(nil)
        XCTAssertEqual(clicked, true)
        button.setValue(true)
        XCTAssertEqual(button.state, .on)
        button.performClick(nil)
        XCTAssertEqual(clicked, false)
        button.setValue(false)
        XCTAssertEqual(button.state, .off)
    }

    func testLockUmbrellaTouchesOnlyConfiguredRowsAndTracksMixedState() {
        var profile = GenerationDefaults()
        XCTAssertEqual(profile.clientLockState(), false)
        profile.rules["temperature"] = .init(value: .number(0.5), ignoreClient: true)
        profile.rules["top_k"] = .init(value: .number(0))
        XCTAssertNil(profile.clientLockState())
        profile.setAllClientLocks(true)
        XCTAssertEqual(profile.clientLockState(), true)
        XCTAssertEqual(profile.rules.count, 2)
        XCTAssertEqual(profile.rules["top_k"]?.value, .number(0))
        profile.setAllClientLocks(false)
        XCTAssertEqual(profile.clientLockState(), false)
        XCTAssertNil(profile.rules["reasoning_budget"])
    }

    func testSessionGenerationSettingsRoundTripBackfillAndStayIsolated() throws {
        var first = ChatSession()
        first.generationParams.rules["temperature"] = .init(value: .number(0.25))
        first.generationParams.rules["reasoning_budget"] = .init(value: .number(1024))
        let encoded = try JSONEncoder().encode(first)
        let decoded = try JSONDecoder().decode(ChatSession.self, from: encoded)
        XCTAssertEqual(decoded.generationParams, first.generationParams)
        XCTAssertTrue(ChatSession().generationParams.rules.isEmpty)
        var old = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        old.removeValue(forKey: "generationParams")
        let backfilled = try JSONDecoder().decode(ChatSession.self, from: JSONSerialization.data(withJSONObject: old))
        XCTAssertTrue(backfilled.generationParams.rules.isEmpty)
    }

    func testForkCopiesSessionGenerationSettingsWithoutSharingMutableState() {
        var source = ChatSession()
        source.generationParams.rules["top_k"] = .init(value: .number(0))
        var fork = ChatFork.session(from: source, messages: [])
        XCTAssertEqual(fork.generationParams, source.generationParams)
        fork.generationParams.rules["top_k"] = .init(value: .number(20))
        XCTAssertEqual(source.generationParams.rules["top_k"]?.value, .number(0))
    }

    func testSessionOverridesAgentValuesAndOmissionsKeepAgentValues() {
        var turn = ChatTurnEngine.TurnConfig(agentMode: true, mcpMode: false, enableThinking: true, voiceStyle: false)
        turn.temperature = 0.7
        turn.topP = 0.9
        turn.repeatPenalty = 1.2
        let profile = GenerationDefaults(rules: [
            "temperature": .init(value: .number(0)),
            "top_k": .init(value: .number(0)),
            "max_tokens": .init(value: .number(0)),
            "reasoning_budget": .init(value: .number(-1)),
        ])
        let applied = turn.applyingGeneration(profile)
        XCTAssertEqual(applied.temperature, 0)
        XCTAssertEqual(applied.topP, 0.9)
        XCTAssertEqual(applied.topK, 0)
        XCTAssertEqual(applied.maxTokens, 0)
        XCTAssertEqual(applied.reasoningBudget, -1)
        XCTAssertEqual(applied.repeatPenalty, 1.2)
        XCTAssertTrue(applied.enableThinking)
    }

    func testSessionFrequencyOverrideReplacesInheritedRepetition() {
        var turn = ChatTurnEngine.TurnConfig(agentMode: true, mcpMode: false, enableThinking: false, voiceStyle: false)
        turn.repeatPenalty = 1.2
        let applied = turn.applyingGeneration(.init(rules: ["frequency_penalty": .init(value: .number(0.5))]))
        XCTAssertNil(applied.repeatPenalty)
        let defaults = applied.requestDefaults(from: ServerOptions(), inheritGeneration: true)
        var body: [String: Any] = [:]
        defaults.applyGeneration(to: &body, maxTokens: 64, temperature: 0.7, enableThinking: false, effort: nil)
        XCTAssertEqual(body["frequency_penalty"] as? Double, 0.5)
        XCTAssertNil(body["repeat_penalty"])
    }

    func testClientBodiesKeepSessionNeutralOverridesWithAndWithoutTools() throws {
        let profile = GenerationDefaults(rules: [
            "temperature": .init(value: .number(0)),
            "top_k": .init(value: .number(0)),
            "repeat_penalty": .init(value: .number(1)),
            "presence_penalty": .init(value: .number(0)),
            "frequency_penalty": .init(value: .number(0)),
            "max_tokens": .init(value: .number(0)),
            "reasoning_budget": .init(value: .number(-1)),
        ])
        let turn = ChatTurnEngine.TurnConfig(agentMode: false, mcpMode: false, enableThinking: false, voiceStyle: false)
            .applyingGeneration(profile)
        let toolsJSON = "[{\"type\":\"function\",\"function\":{\"name\":\"write\",\"parameters\":{\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}}}]"
        for inherited in [false, true] {
            for tools in [nil, toolsJSON] {
                let data = try APIClient.chatRequestBody(messages: [["role": "user", "content": "hi"]],
                    maxTokens: 64, temperature: 0.8, enableThinking: false,
                    toolsJSON: tools, defaults: turn.requestDefaults(from: ServerOptions(), inheritGeneration: inherited))
                let body = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
                XCTAssertEqual(body["temperature"] as? Double, 0)
                XCTAssertEqual(body["max_tokens"] as? Int, 0)
                XCTAssertEqual(body["top_k"] as? Int, 0)
                XCTAssertEqual(body["repeat_penalty"] as? Double, 1)
                XCTAssertEqual(body["presence_penalty"] as? Double, 0)
                XCTAssertEqual(body["frequency_penalty"] as? Double, 0)
                XCTAssertEqual(body["reasoning_budget_tokens"] as? Int, -1)
                if let tools { XCTAssertTrue(String(decoding: data, as: UTF8.self).contains(tools)) }
            }
        }
    }

    func testInheritedClientBodyDoesNotPinSamplingAndRemoteBodyKeepsExistingDefaults() throws {
        var local = APIClient.RequestDefaults()
        local.inheritGeneration = true
        let data = try APIClient.chatRequestBody(messages: [], maxTokens: 64, temperature: 0.8,
                                                 enableThinking: false, defaults: local)
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertNil(body["temperature"])
        XCTAssertNil(body["top_p"])
        XCTAssertNil(body["max_tokens"])
        XCTAssertEqual(body["enable_thinking"] as? Bool, false)
        let remote = try APIClient.chatRequestBody(messages: [], maxTokens: 64, temperature: 0.8, enableThinking: false)
        let remoteBody = try XCTUnwrap(JSONSerialization.jsonObject(with: remote) as? [String: Any])
        XCTAssertEqual(remoteBody["temperature"] as? Double, 0.8)
        XCTAssertEqual(remoteBody["top_p"] as? Double, 0.95)
        XCTAssertEqual(remoteBody["max_tokens"] as? Int, 64)
        XCTAssertNil(remoteBody["enable_thinking"])
    }
}
