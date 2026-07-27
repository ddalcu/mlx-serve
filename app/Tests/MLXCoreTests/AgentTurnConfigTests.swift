import XCTest
@testable import MLXCore

/// Five surfaces start turns (chat tab, voice tray, scheduled task, Telegram,
/// Quick Launcher). They all build their `TurnConfig` through ONE builder from a
/// `ResolvedAgentSettings`, so a surface can't quietly ignore the agent — the
/// same class as the LAN rule about reading `modelInfo?.name` at a new surface.
final class AgentTurnConfigTests: XCTestCase {

    private func resolved(_ mutate: (inout Agent) -> Void) -> ResolvedAgentSettings {
        var a = Agent(name: "Coder", brief: "b", systemPrompt: "You are a coder.")
        mutate(&a)
        return AgentResolution.resolve(agent: a, defaults: AppDefaultsSnapshot(
            toolsEnabled: false, mcpEnabled: false, thinkingEnabled: false,
            autoApprove: false, workingDirectory: "/tmp/default",
            temperature: 0.8, maxTokens: 4096))
    }

    func testBuilderCarriesEveryResolvedFieldIntoTheTurn() {
        let r = resolved {
            $0.capabilities = AgentCapabilities(tools: true, mcp: true, web: false)
            $0.enableThinking = true
            $0.autoApproveTools = true
            $0.workingDirectory = "/tmp/agent"
            $0.temperature = 0.25
            $0.maxTokens = 999
        }
        let config = ChatTurnEngine.TurnConfig.from(r)

        XCTAssertEqual(config.agentId, r.agentId)
        XCTAssertTrue(config.agentMode)
        XCTAssertTrue(config.mcpMode)
        XCTAssertTrue(config.enableThinking)
        XCTAssertTrue(config.autoApprove)
        XCTAssertEqual(config.workingDirectory, "/tmp/agent")
        XCTAssertEqual(config.temperature, 0.25)
        XCTAssertEqual(config.maxTokens, 999)
        XCTAssertEqual(config.tools, r.tools)
        XCTAssertTrue(config.systemPromptPrefix.hasPrefix("You are a coder."))
        XCTAssertFalse(config.voiceStyle)
    }

    func testSamplingOverridesAreNilWhenTheAgentDidNotSetThem() {
        // nil means "this path's own default" — the agent loop's temperature and
        // plain chat's are DIFFERENT numbers, so a decided value here would
        // silently change one of them for every existing user.
        let config = ChatTurnEngine.TurnConfig.from(resolved { _ in })
        XCTAssertNil(config.temperature)
        XCTAssertNil(config.maxTokens)
    }

    func testTheAgentsVoiceRidesTheTurnSoItCanNeverGoStale() {
        // The voice used to be published only when an agent was SELECTED, so
        // editing the active agent's voice — or answering as a tab's agent that
        // isn't the tray's — spoke in the app's voice instead (an uploaded clone
        // clip never reached Qwen3-TTS at all). It's per-TURN now.
        let r = resolved { $0.voice = .clone("/clips/morgan.wav") }
        XCTAssertEqual(ChatTurnEngine.TurnConfig.from(r).voice, .clone("/clips/morgan.wav"))
    }

    func testNoAgentVoiceLeavesTheLiveSettingsReadAlone() {
        // nil, NOT the global fallback: publishing the resolved value would pin
        // the turn's voice and stop a mid-answer Settings change from applying.
        let r = AgentResolution.resolve(agent: nil, defaults: AppDefaultsSnapshot(
            voice: .kokoro("af_sky")))
        XCTAssertNil(ChatTurnEngine.TurnConfig.from(r).voice)
        XCTAssertEqual(r.voice, .kokoro("af_sky"), "the decided value is still there for display")

        let noVoiceAgent = resolved { _ in }
        XCTAssertNil(ChatTurnEngine.TurnConfig.from(noVoiceAgent).voice)
    }

    func testNoAgentProducesTodaysConfig() {
        let r = AgentResolution.resolve(agent: nil, defaults: AppDefaultsSnapshot(
            toolsEnabled: true, mcpEnabled: false, thinkingEnabled: true,
            autoApprove: false, workingDirectory: "/tmp/default"))
        let config = ChatTurnEngine.TurnConfig.from(r, documentIndex: nil, telegramChatId: 42)

        XCTAssertNil(config.agentId)
        XCTAssertEqual(config.systemPromptPrefix, "")
        XCTAssertTrue(config.agentMode)
        XCTAssertFalse(config.mcpMode)
        XCTAssertTrue(config.enableThinking)
        XCTAssertEqual(config.workingDirectory, "/tmp/default")
        XCTAssertEqual(config.tools, Set(AgentToolKind.allCases), "full access, exactly as before")
        XCTAssertEqual(config.telegramChatId, 42)
        XCTAssertNil(config.temperature)
        XCTAssertNil(config.maxTokens)
    }

    func testVoiceAndDocumentContextRideAlongsideTheAgent() {
        let config = ChatTurnEngine.TurnConfig.from(resolved { $0.capabilities.tools = true },
                                                   voiceStyle: true)
        XCTAssertTrue(config.voiceStyle)
        XCTAssertTrue(config.agentMode)
    }

    // MARK: - Where the persona lands

    func testPersonaPrecedesTheStableBlockAndAllVolatileContent() {
        // The persona must be in the STABLE prefix: the volatile tail is
        // re-prefilled every turn, so a persona there would cost a re-prefill
        // per message instead of one per agent switch.
        let out = ChatTurnEngine.composeSystemPrompt(
            persona: "You are a coder.\n\n",
            stable: "STABLE-BLOCK",
            volatileTail: "VOLATILE-TAIL",
            grounding: "Today is Monday.")
        let personaAt = out.range(of: "You are a coder.")!.lowerBound
        let stableAt = out.range(of: "STABLE-BLOCK")!.lowerBound
        let tailAt = out.range(of: "VOLATILE-TAIL")!.lowerBound
        let groundingAt = out.range(of: "Today is Monday.")!.lowerBound
        XCTAssertTrue(personaAt < stableAt)
        XCTAssertTrue(stableAt < tailAt)
        XCTAssertTrue(tailAt < groundingAt)
    }

    func testNoPersonaLeavesThePromptByteIdentical() {
        let withoutArg = ChatTurnEngine.composeSystemPrompt(
            stable: "S", volatileTail: "V", grounding: "G")
        let withEmpty = ChatTurnEngine.composeSystemPrompt(
            persona: "", stable: "S", volatileTail: "V", grounding: "G")
        XCTAssertEqual(withEmpty, withoutArg,
                       "an install with no agents must produce the exact prompt it does today")
    }
}
