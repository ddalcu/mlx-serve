import XCTest
import SwiftUI
@testable import MLXCore

final class GenerationDefaultsRowsTests: XCTestCase {
    @MainActor
    func testSharedRowsRenderAtModelSheetWidth() throws {
        var profile = GenerationDefaults()
        profile.setMode(.enabled, field: .temperature)
        profile.setMode(.forced, field: .topP)
        let view = GenerationDefaultsRows(profile: .constant(profile), inheritance: "Model default")
            .padding(20).frame(width: 620).background(Color(nsColor: .windowBackgroundColor))
            .environment(\.colorScheme, .dark)
        let hosting = NSHostingView(rootView: view)
        hosting.appearance = NSAppearance(named: .darkAqua)
        let size = hosting.fittingSize
        XCTAssertGreaterThan(size.height, 500)
        XCTAssertEqual(size.width, 620)
        hosting.frame = NSRect(origin: .zero, size: size)
        hosting.layoutSubtreeIfNeeded()
        if let path = ProcessInfo.processInfo.environment["GENERATION_UI_SNAPSHOT"] {
            let bitmap = try XCTUnwrap(hosting.bitmapImageRepForCachingDisplay(in: hosting.bounds))
            hosting.cacheDisplay(in: hosting.bounds, to: bitmap)
            let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
            try png.write(to: URL(fileURLWithPath: path))
        }
    }

    @MainActor
    func testModeSwitchStaysCompactAndKeepsLabelSpaceForMixedState() {
        let states: [GenerationDefaults.Mode?] = [nil, .inherited, .enabled, .forced]
        for allowsForce in [false, true] {
            var previous: CGSize?
            for state in states where allowsForce || state != .forced {
                let view = GenerationModeSwitch(value: state, allowsForce: allowsForce,
                    inheritance: "Model default", title: "Temperature", onChange: { _ in })
                let size = NSHostingView(rootView: view).fittingSize
                XCTAssertLessThanOrEqual(size.width, allowsForce ? 68 : 48)
                XCTAssertGreaterThanOrEqual(size.height, 44)
                if let previous { XCTAssertEqual(size, previous) }
                previous = size
            }
        }
    }

    @MainActor
    func testBulkMixedStateDoesNotMoveRows() {
        var mixed = GenerationDefaults()
        mixed.setMode(.enabled, field: .thinking)
        var enabled = mixed
        enabled.setMode(.enabled, field: .effort)
        let inherited = GenerationDefaults(rules: [
            "enable_thinking": .init(value: .boolean(true)),
            "reasoning_effort": .init(value: .text("low")),
        ])
        let sizes = [GenerationDefaults(), mixed, enabled].map { profile in
            NSHostingView(rootView: GenerationDefaultsRows(profile: .constant(profile),
                fields: [.thinking, .effort], inherited: inherited).frame(width: 620)).fittingSize
        }
        XCTAssertEqual(sizes[0], sizes[1])
        XCTAssertEqual(sizes[1], sizes[2])
    }

    func testSelectorStatesCreateRulesPreserveValuesAndClearInheritance() {
        var profile = GenerationDefaults()
        XCTAssertEqual(profile.mode(.temperature), .inherited)
        profile.setMode(.forced, field: .temperature, inherited: .number(0.6))
        XCTAssertEqual(profile.rules["temperature"]?.value, .number(0.6))
        XCTAssertEqual(profile.mode(.temperature), .forced)
        profile.rules["temperature"]?.value = .number(0.25)
        profile.setMode(.enabled, field: .temperature)
        XCTAssertEqual(profile.rules["temperature"]?.value, .number(0.25))
        XCTAssertEqual(profile.mode(.temperature), .enabled)
        profile.setMode(.inherited, field: .temperature)
        XCTAssertNil(profile.rules["temperature"])
    }

    func testBulkSelectorReportsMixedStatesAndOnlyEditsVisibleFields() {
        var profile = GenerationDefaults()
        profile.setMode(.enabled, field: .temperature)
        profile.setMode(.forced, field: .topP)
        XCTAssertNil(profile.mode(fields: [.temperature, .topP]))
        profile.setMode(.forced, fields: [.temperature, .topK])
        XCTAssertEqual(profile.mode(fields: [.temperature, .topP, .topK]), .forced)
        profile.setMode(.inherited, fields: [.temperature, .topK])
        XCTAssertEqual(profile.mode(.topP), .forced)
        XCTAssertEqual(profile.mode(fields: [.temperature, .topK]), .inherited)
    }

    func testAgentAdapterPreservesNeutralOverridesAndUnrelatedSettings() {
        var agent = Agent(name: "Sampling", systemPrompt: "Keep this prompt.")
        agent.enableThinking = true
        agent.topK = 0
        agent.reasoningBudget = -1
        var profile = GenerationDefaults(agent: agent)
        XCTAssertEqual(profile.mode(.topK), .enabled)
        XCTAssertEqual(profile.number(.topK), 0)
        profile.setMode(.enabled, field: .temperature, inherited: .number(0.6))
        profile.rules["repeat_penalty"] = .init(value: .number(1))
        profile.apply(to: &agent)
        XCTAssertEqual(agent.temperature, 0.6)
        XCTAssertEqual(agent.topK, 0)
        XCTAssertEqual(agent.repeatPenalty, 1)
        XCTAssertEqual(agent.reasoningBudget, -1)
        XCTAssertEqual(agent.enableThinking, true)
        XCTAssertEqual(agent.systemPrompt, "Keep this prompt.")
        profile.setMode(.inherited, fields: GenerationField.agentFields)
        profile.apply(to: &agent)
        XCTAssertNil(agent.temperature)
        XCTAssertNil(agent.topK)
        XCTAssertNil(agent.reasoningBudget)
        XCTAssertEqual(agent.enableThinking, true)
    }

    func testInheritedProfileMergesModelValuesAndSeedsWithoutCopyingLocks() {
        let inherited = GenerationDefaults(rules: [
            "temperature": .init(value: .number(0.6), ignoreClient: true),
            "top_k": .init(value: .number(20)),
        ])
        var profile = GenerationDefaults()
        profile.setMode(.enabled, fields: [.temperature, .topK], inherited: inherited)
        XCTAssertEqual(profile.number(.temperature), 0.6)
        XCTAssertEqual(profile.number(.topK), 20)
        XCTAssertEqual(profile.mode(.temperature), .enabled)
        XCTAssertFalse(profile.rules["temperature"]!.ignoreClient)
    }

    func testSharedGuidanceAndPresetsPreserveExactTypedValues() {
        XCTAssertEqual(GenerationField.temperature.guidance?.low, "Focused")
        XCTAssertEqual(GenerationField.temperature.guidance?.high, "Creative")
        XCTAssertEqual(GenerationField.repeatPenalty.guidance?.low, "Off")
        XCTAssertEqual(GenerationField.topK.guidance?.low, "Off")
        XCTAssertEqual(GenerationField.maxTokens.guidance?.low, "Auto")
        for field in GenerationField.allCases where field.showsSlider {
            XCTAssertNotNil(field.guidance, field.rawValue)
        }
        XCTAssertEqual(GenerationField.maxTokens.numberValue(12345), .number(12345))
        XCTAssertEqual(GenerationField.topK.numberValue(20.6), .number(21))
        XCTAssertNil(GenerationField.temperature.numberValue(.nan))
        XCTAssertNil(GenerationField.topP.numberValue(1.1))
        XCTAssertEqual(GenerationField.budget.presets?.first, -1)
        XCTAssertTrue(GenerationField.maxTokens.presets!.contains(16384))
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
