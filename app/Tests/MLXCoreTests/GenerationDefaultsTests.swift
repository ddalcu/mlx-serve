import XCTest
@testable import MLXCore

final class GenerationDefaultsTests: XCTestCase {
    func testCanonicalParametersLiveInLabelsNotDescriptions() {
        for field in GenerationField.allCases {
            XCTAssertEqual(field.parameterName, field.rawValue)
            XCTAssertFalse(field.help.hasPrefix(field.rawValue + " —"), field.rawValue)
        }
        XCTAssertTrue(GenerationField.budget.help.contains("reasoning_budget_tokens"))
    }

    func testTopKHelpExplainsCandidateCountInsteadOfInheritance() {
        let help = GenerationField.topK.help
        XCTAssertTrue(help.contains("most likely next tokens"))
        XCTAssertTrue(help.contains("0 disables this filter"))
        XCTAssertFalse(help.contains("Inherit"))
    }

    func testRetiredSessionParametersAreIgnoredAndNotPersisted() throws {
        let existing = try JSONEncoder().encode(ChatSession())
        var stored = try XCTUnwrap(JSONSerialization.jsonObject(with: existing) as? [String: Any])
        stored["generationParams"] = ["temperature": ["value": 0.1], "top_k": ["value": 0]]
        let session = try JSONDecoder().decode(ChatSession.self, from: JSONSerialization.data(withJSONObject: stored))
        let encoded = try JSONEncoder().encode(session)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        XCTAssertNil(object["generationParams"])
    }

    func testMinPProfileRoundTripsAndRejectsOutOfRangeValues() throws {
        let data = Data("{\"min_p\":{\"value\":0.05,\"ignore_client\":true}}".utf8)
        let profile = try JSONDecoder().decode(GenerationDefaults.self, from: data)
        XCTAssertEqual(profile.rules["min_p"]?.value, .number(0.05))
        XCTAssertTrue(profile.rules["min_p"]!.ignoreClient)
        XCTAssertEqual(try JSONDecoder().decode(GenerationDefaults.self, from: JSONEncoder().encode(profile)), profile)
        XCTAssertThrowsError(try JSONDecoder().decode(GenerationDefaults.self,
            from: Data("{\"min_p\":{\"value\":1.1}}".utf8)))
    }

    func testAgentSamplingOverridesReachPlainAndToolRequests() throws {
        var agent = Agent(name: "Sampling", systemPrompt: "Answer briefly.")
        agent.temperature = 0
        agent.maxTokens = 0
        agent.topP = 0.8
        agent.topK = 0
        agent.repeatPenalty = 1
        agent.presencePenalty = 0
        agent.reasoningBudget = -1
        let resolved = AgentResolution.resolve(agent: agent, defaults: .init())
        let turn = ChatTurnEngine.TurnConfig.from(resolved)
        let tools = "[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"parameters\":{\"type\":\"object\"}}}]"
        for toolsJSON in [nil, tools] {
            let data = try APIClient.chatRequestBody(messages: [], maxTokens: 64, temperature: 0.8,
                enableThinking: false, toolsJSON: toolsJSON,
                defaults: turn.requestDefaults(from: ServerOptions(), inheritGeneration: true))
            let body = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            XCTAssertEqual(body["temperature"] as? Double, 0)
            XCTAssertEqual(body["max_tokens"] as? Int, 0)
            XCTAssertEqual(body["top_p"] as? Double, 0.8)
            XCTAssertEqual(body["top_k"] as? Int, 0)
            XCTAssertEqual(body["repeat_penalty"] as? Double, 1)
            XCTAssertEqual(body["presence_penalty"] as? Double, 0)
            XCTAssertEqual(body["reasoning_budget_tokens"] as? Int, -1)
            XCTAssertEqual(body["tools"] != nil, toolsJSON != nil)
        }
    }

    func testLocalExplicitThinkingOffDoesNotResendLegacyGlobalThinking() {
        var options = ServerOptions()
        options.defaultEnableThinking = true
        let turn = ChatTurnEngine.TurnConfig(agentMode: false, mcpMode: false, enableThinking: false, voiceStyle: false)
        XCTAssertFalse(turn.thinkingForRequest(options, inheritGeneration: true))
        XCTAssertTrue(turn.thinkingForRequest(options, inheritGeneration: false))
    }

    func testSamplingFieldsKeepSlidersAndIntegerBudgetsUseExactValues() {
        XCTAssertTrue(GenerationField.temperature.showsSlider)
        XCTAssertTrue(GenerationField.topP.showsSlider)
        XCTAssertTrue(GenerationField.repeatPenalty.showsSlider)
        XCTAssertTrue(GenerationField.presencePenalty.showsSlider)
        XCTAssertTrue(GenerationField.maxTokens.showsSlider)
        XCTAssertTrue(GenerationField.budget.showsSlider)
        XCTAssertFalse(GenerationField.thinking.showsSlider)
        XCTAssertFalse(GenerationField.effort.showsSlider)
    }

    func testResetExplainsNextRequestGenerationScope() {
        let message = SettingsReset.confirmMessage(.category(.requestDefaults))
        XCTAssertTrue(message.contains("next request"))
        XCTAssertFalse(message.contains("Restart Now"))
        XCTAssertTrue(SettingsReset.confirmMessage(.all).contains("Generation Defaults"))
    }

    func testRulesRoundTripPreservesNeutralValuesAndIndependentLocks() throws {
        let json = Data("""
        {"top_k":{"value":0,"ignore_client":true},"enable_thinking":{"value":false},"reasoning_budget":{"value":1024,"ignore_client":true}}
        """.utf8)
        let profile = try JSONDecoder().decode(GenerationDefaults.self, from: json)
        XCTAssertEqual(profile.rules["top_k"]?.value, .number(0))
        XCTAssertEqual(profile.rules["enable_thinking"]?.value, .boolean(false))
        XCTAssertFalse(profile.rules["enable_thinking"]!.ignoreClient)
        XCTAssertTrue(profile.rules["reasoning_budget"]!.ignoreClient)
        XCTAssertEqual(try JSONDecoder().decode(GenerationDefaults.self, from: JSONEncoder().encode(profile)), profile)
    }

    func testLegacyMigrationKeepsOnlyServerWideValuesAndDoesNotOverwriteExistingProfile() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("generation-settings.json").path
        var options = ServerOptions()
        options.defaultReasoningBudget = 1024
        options.defaultRepeatPenalty = 1.1
        options.defaultTemperature = 0.8
        try GenerationDefaultsFile.migrate(options, path: path)
        let profile = try GenerationDefaultsFile.load(path: path)
        XCTAssertEqual(profile.rules["temperature"]?.value, .number(0.8))
        XCTAssertNil(profile.rules["top_k"])
        // App-chat-only defaults stay off the server: they would cap and penalize every client.
        XCTAssertNil(profile.rules["reasoning_budget"])
        XCTAssertNil(profile.rules["repeat_penalty"])
        XCTAssertNil(profile.rules["max_tokens"])
        XCTAssertTrue(profile.rules.values.allSatisfy { !$0.ignoreClient })
        try GenerationDefaultsFile.save(.init(), path: path)
        try GenerationDefaultsFile.migrate(options, path: path)
        XCTAssertTrue(try GenerationDefaultsFile.load(path: path).rules.isEmpty)
    }

    func testInheritedRequestControlsAreOmittedButExplicitNeutralControlsRemain() {
        var inherited = APIClient.RequestDefaults()
        inherited.inheritGeneration = true
        var body: [String: Any] = ["temperature": 0.8, "top_p": 0.95, "max_tokens": 16384, "enable_thinking": false]
        inherited.applyGeneration(to: &body, maxTokens: 16384, temperature: 0.8, enableThinking: false, effort: nil)
        XCTAssertNil(body["temperature"])
        XCTAssertNil(body["top_p"])
        XCTAssertNil(body["max_tokens"])
        XCTAssertEqual(body["enable_thinking"] as? Bool, false)
        inherited.temperatureOverride = 0
        inherited.maxTokensOverride = 0
        inherited.topK = 0
        inherited.repeatPenalty = 1
        inherited.reasoningBudget = -1
        inherited.applyGeneration(to: &body, maxTokens: 16384, temperature: 0.8, enableThinking: true, effort: "low")
        XCTAssertEqual(body["temperature"] as? Double, 0)
        XCTAssertEqual(body["max_tokens"] as? Int, 0)
        XCTAssertEqual(body["top_k"] as? Int, 0)
        XCTAssertEqual(body["repeat_penalty"] as? Double, 1)
        XCTAssertEqual(body["reasoning_budget_tokens"] as? Int, -1)
    }

    func testInvalidPolicyIsRejectedAndUnknownModelPolicySurvivesEditing() throws {
        for json in [
            "{\"top_k\":{\"value\":-1,\"ignore_client\":true}}",
            "{\"reasoning_budget\":{\"value\":-2}}",
            "{\"enable_thinking\":{\"value\":1}}",
            "{\"reasoning_effort\":{\"value\":\"banana\"}}",
            "{\"top_k\":{\"value\":0,\"ignore_client\":null}}",
            "{\"top_k\":{\"value\":0,\"ignore_clent\":true}}",
        ] {
            XCTAssertThrowsError(try JSONDecoder().decode(GenerationDefaults.self, from: Data(json.utf8)))
        }
        let raw: [String: Any] = ["ctx_size": 8192, "generation_defaults": ["future_setting": ["value": 1, "ignore_client": true]]]
        var model = ModelOverride(json: raw)
        model.alias = "qwen"
        XCTAssertNotNil(model.json["generation_defaults"])
    }

    func testSaveRejectsInvalidPolicyWithoutReplacingExistingFile() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("generation-settings.json").path
        let good = GenerationDefaults(rules: ["reasoning_budget": .init(value: .number(1024), ignoreClient: true)])
        try GenerationDefaultsFile.save(good, path: path)
        let invalid = GenerationDefaults(rules: ["reasoning_budget": .init(value: .number(-2), ignoreClient: true)])
        XCTAssertThrowsError(try GenerationDefaultsFile.save(invalid, path: path))
        XCTAssertEqual(try GenerationDefaultsFile.load(path: path), good)
    }

    func testGenerationEditsDoNotRequireModelReload() {
        var model = ModelOverride(ctxSize: 8192)
        let old = model
        model.generationDefaults.rules["reasoning_budget"] = .init(value: .number(1024), ignoreClient: true)
        XCTAssertFalse(model.changesLoad(from: old))
        XCTAssertTrue(model.hasSettings)
        XCTAssertEqual(ModelOverride(json: model.json), model)
        model.kvQuant = .bits8
        XCTAssertTrue(model.changesLoad(from: old))
    }
}
