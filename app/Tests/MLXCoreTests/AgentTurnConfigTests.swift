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
        XCTAssertNil(config.topP)
        XCTAssertNil(config.topK)
        XCTAssertNil(config.repeatPenalty)
        XCTAssertNil(config.presencePenalty)
        XCTAssertNil(config.reasoningBudget)
    }

    // MARK: - Reasoning effort (the brain disc's right-click pick)

    func testReasoningEffortRidesResolutionIntoTheTurnAndDefaultsLow() {
        XCTAssertEqual(ChatTurnEngine.TurnConfig.from(resolved { _ in }).reasoningEffort, .low)
        var d = AppDefaultsSnapshot()
        d.reasoningEffort = .high
        let r = AgentResolution.resolve(agent: nil, defaults: d)
        XCTAssertEqual(ChatTurnEngine.TurnConfig.from(r).reasoningEffort, .high)
    }

    func testReasoningEffortIsOnlySentWhileThinkingIsOn() {
        // The server reads `reasoning_effort` as a thinking OPT-IN, so sending
        // it with the toggle off would silently turn thinking on.
        var config = ChatTurnEngine.TurnConfig.from(resolved { _ in })
        config.reasoningEffort = .medium
        XCTAssertEqual(config.reasoningEffortParam(thinking: true), "medium")
        XCTAssertNil(config.reasoningEffortParam(thinking: false))
    }

    // MARK: - Request defaults (the agent's sampling laid over the user's)

    func testRequestDefaultsWithoutOverridesMatchTheGlobalBuild() {
        // No agent overrides ⇒ byte-identical to what the call sites sent before
        // (RequestDefaults.from) — the upgrade guarantee at the request layer.
        var opts = ServerOptions()
        opts.defaultTopP = 0.9
        opts.defaultTopK = 20
        opts.defaultRepeatPenalty = 1.15
        opts.defaultPresencePenalty = 0.6
        opts.defaultReasoningBudget = 1024
        let config = ChatTurnEngine.TurnConfig.from(resolved { _ in })
        XCTAssertEqual(config.requestDefaults(from: opts), APIClient.RequestDefaults.from(opts))
    }

    func testRequestDefaultsApplyTheAgentsSamplingOverrides() {
        var opts = ServerOptions()
        opts.defaultTopP = 0.9
        opts.defaultTopK = 20
        opts.defaultRepeatPenalty = 1.15
        opts.defaultPresencePenalty = 0.6
        opts.defaultReasoningBudget = 1024

        // Each override REPLACES the saved default — including replacing it
        // with the canonical "off" value, which must clear the global rather
        // than leave it standing (an agent that wants top_k off against a
        // global top_k 20 would otherwise be un-expressible). The off values
        // map exactly as RequestDefaults.from maps them: omitted from the body.
        let config = ChatTurnEngine.TurnConfig.from(resolved {
            $0.topP = 0.7
            $0.topK = 0          // 0 = model default ⇒ field omitted
            $0.repeatPenalty = 1.0   // 1.0 = off ⇒ omitted
            $0.presencePenalty = 1.3
            $0.reasoningBudget = -1  // -1 = unlimited ⇒ omitted
        })
        let d = config.requestDefaults(from: opts)
        XCTAssertEqual(d.topP, 0.7)
        XCTAssertNil(d.topK)
        XCTAssertNil(d.repeatPenalty)
        XCTAssertEqual(d.presencePenalty, 1.3)
        XCTAssertNil(d.reasoningBudget)

        // And the non-off direction: a value where the global had none.
        var quiet = ServerOptions()
        quiet.defaultTopK = 0
        quiet.defaultRepeatPenalty = 1.0
        let loud = ChatTurnEngine.TurnConfig.from(resolved {
            $0.topK = 40
            $0.repeatPenalty = 1.1
            $0.reasoningBudget = 2048
        })
        let d2 = loud.requestDefaults(from: quiet)
        XCTAssertEqual(d2.topK, 40)
        XCTAssertEqual(d2.repeatPenalty, 1.1)
        XCTAssertEqual(d2.reasoningBudget, 2048)
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

    func testAgentPersonaIsTheEntireSystemPrompt() {
        // An agent's prompt REPLACES the normal system prompt — base
        // instructions, listings, volatile tail and grounding are the app's
        // voice, and composing them after a persona is how the persona got
        // overridden (live 2026-07-29: Laguna answered "who are you?" with
        // "I'm poolside Malibu" under an Elon Musk persona — the agent-prompt
        // body opens with its own "You are an autonomous agent" claim). Tools
        // still ride the request's tools JSON, so dispatch is unaffected; the
        // agent's prompt has to carry anything else it needs. Matches the
        // plain-chat path, where a persona is already the whole system message.
        let out = ChatTurnEngine.composeSystemPrompt(
            persona: "You are Elon Musk.\n\n",
            stable: "You are an autonomous agent. STABLE-BLOCK",
            volatileTail: "VOLATILE-TAIL",
            grounding: "Today is Monday.")
        XCTAssertEqual(out, "You are Elon Musk.",
                       "the persona alone, trimmed — nothing composed around it")
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
