import XCTest
@testable import MLXCore

/// `AgentResolution` is the ONE chokepoint every turn site reads instead of the
/// globals it used to reach for individually. Two properties matter most:
///
///  1. **No agent reproduces today's behavior, field for field.** That's the
///     upgrade guarantee — an existing install with no agents must be
///     byte-identical to the previous build.
///  2. **Every agent field is an OVERRIDE.** nil means "app default", so an
///     agent that only sets a voice must not quietly re-anchor the working
///     directory or reset sampling.
final class AgentResolutionTests: XCTestCase {

    private func defaults() -> AppDefaultsSnapshot {
        AppDefaultsSnapshot(
            toolsEnabled: true,
            mcpEnabled: true,
            thinkingEnabled: true,
            autoApprove: true,
            tools: Set(AgentToolKind.allCases),
            workingDirectory: "/tmp/app-default",
            modelPath: "/models/current",
            temperature: 0.8,
            maxTokens: 4096,
            voice: .system("com.apple.voice.compact.en-US.Samantha"),
            wakePhrase: "hey loki"
        )
    }

    private func agent(_ mutate: (inout Agent) -> Void = { _ in }) -> Agent {
        var a = Agent(name: "Coder", brief: "b", systemPrompt: "You are a coder.")
        mutate(&a)
        return a
    }

    // MARK: - The upgrade guarantee

    func testNilAgentReproducesAppDefaultsFieldForField() {
        let d = defaults()
        let r = AgentResolution.resolve(agent: nil, defaults: d)

        XCTAssertNil(r.agentId)
        XCTAssertEqual(r.systemPromptPrefix, "", "no agent ⇒ no persona in front of the prompt")
        XCTAssertEqual(r.tools, d.tools)
        XCTAssertEqual(r.toolsEnabled, d.toolsEnabled)
        XCTAssertEqual(r.mcpEnabled, d.mcpEnabled)
        XCTAssertEqual(r.thinkingEnabled, d.thinkingEnabled)
        XCTAssertEqual(r.autoApprove, d.autoApprove)
        XCTAssertEqual(r.workingDirectory, d.workingDirectory)
        XCTAssertEqual(r.modelPath, d.modelPath)
        XCTAssertEqual(r.temperature, d.temperature)
        XCTAssertEqual(r.maxTokens, d.maxTokens)
        XCTAssertEqual(r.voice, d.voice)
        XCTAssertEqual(r.wakePhrase, d.wakePhrase)
    }

    func testAgentWithNoOverridesKeepsEveryAppDefaultExceptItsPersona() {
        let d = defaults()
        // Capabilities default to tools off / web on, so those two ARE decided by
        // the agent; everything else is nil and must fall through.
        let r = AgentResolution.resolve(agent: agent(), defaults: d)

        XCTAssertEqual(r.workingDirectory, d.workingDirectory)
        XCTAssertEqual(r.modelPath, d.modelPath)
        XCTAssertEqual(r.temperature, d.temperature)
        XCTAssertEqual(r.maxTokens, d.maxTokens)
        XCTAssertEqual(r.autoApprove, d.autoApprove)
        XCTAssertEqual(r.thinkingEnabled, d.thinkingEnabled)
        XCTAssertEqual(r.voice, d.voice)
        XCTAssertEqual(r.wakePhrase, d.wakePhrase)
        XCTAssertTrue(r.systemPromptPrefix.hasPrefix("You are a coder."),
                      "the persona leads the prefix: \(r.systemPromptPrefix)")
        XCTAssertTrue(r.systemPromptPrefix.hasSuffix("\n\n"),
                      "the prefix carries its own separator so callers can concatenate")
    }

    // MARK: - Each override wins

    func testEachOverrideWins() {
        let d = defaults()
        let a = agent {
            $0.workingDirectory = "/tmp/agent-wd"
            $0.modelPath = "/models/pinned"
            $0.temperature = 0.2
            $0.maxTokens = 512
            $0.autoApproveTools = false
            $0.enableThinking = false
            $0.wakePhrase = "hey coder"
            $0.voice = .kokoro("af_bella")
            $0.capabilities.tools = true
            $0.capabilities.mcp = false
        }
        let r = AgentResolution.resolve(agent: a, defaults: d)

        XCTAssertEqual(r.agentId, a.id)
        XCTAssertEqual(r.workingDirectory, "/tmp/agent-wd")
        XCTAssertEqual(r.modelPath, "/models/pinned")
        XCTAssertEqual(r.temperature, 0.2)
        XCTAssertEqual(r.maxTokens, 512)
        XCTAssertFalse(r.autoApprove)
        XCTAssertFalse(r.thinkingEnabled)
        XCTAssertEqual(r.wakePhrase, "hey coder")
        XCTAssertEqual(r.voice, .kokoro("af_bella"))
        XCTAssertTrue(r.toolsEnabled)
        XCTAssertFalse(r.mcpEnabled, "an agent's MCP flag overrides the surface toggle")
    }

    func testEmptySystemPromptYieldsNoPrefix() {
        let a = agent { $0.systemPrompt = "   \n " }
        XCTAssertEqual(AgentResolution.resolve(agent: a, defaults: defaults()).systemPromptPrefix, "")
    }

    func testWakePhraseIsNormalized() {
        let a = agent { $0.wakePhrase = "  Hey,  CODER! " }
        XCTAssertEqual(AgentResolution.resolve(agent: a, defaults: defaults()).wakePhrase, "hey coder")
    }

    func testBlankWakePhraseFallsBackToTheAppPhrase() {
        let a = agent { $0.wakePhrase = "  " }
        XCTAssertEqual(AgentResolution.resolve(agent: a, defaults: defaults()).wakePhrase, "hey loki")
    }

    func testIPhoneVoiceIDIsHonoredWhenNoMacVoiceIsSet() {
        // voiceID is the iPhone-shaped Kokoro name kept for import/export; the
        // macOS `voice` field wins when both are present.
        var a = agent { $0.voiceID = "af_heart" }
        XCTAssertEqual(AgentResolution.resolve(agent: a, defaults: defaults()).voice, .kokoro("af_heart"))
        a.voice = .clone("/tmp/clip.wav")
        XCTAssertEqual(AgentResolution.resolve(agent: a, defaults: defaults()).voice, .clone("/tmp/clip.wav"))
    }

    // MARK: - Coarse flags → concrete tool set

    func testCoarseFlagsMapToToolSets() {
        var caps = AgentCapabilities(tools: false, mcp: false, web: false)
        XCTAssertEqual(caps.resolvedTools(), [], "everything off ⇒ no tools")

        caps.web = true
        XCTAssertEqual(caps.resolvedTools(), [.browse, .webSearch],
                       "Web maps to exactly browse + webSearch")

        caps.tools = true
        let all = caps.resolvedTools()
        XCTAssertTrue(all.isSuperset(of: [.shell, .readFile, .writeFile, .editFile, .cwd, .createTask]),
                      "Tools brings the loop's own tools")
        XCTAssertTrue(all.contains(.browse), "Web is additive to Tools")
        XCTAssertFalse(all.contains(.searchDocuments),
                       "searchDocuments is gated by the attached folder, never by a capability toggle")

        caps.web = false
        XCTAssertFalse(caps.resolvedTools().contains(.browse))
        XCTAssertFalse(caps.resolvedTools().contains(.webSearch))
    }

    func testAdvancedSetIsReturnedVerbatim() {
        var caps = AgentCapabilities(tools: true, mcp: true, web: true)
        caps.advancedTools = [.readFile]
        XCTAssertEqual(caps.resolvedTools(), [.readFile],
                       "once the user goes Advanced, their set is the answer — coarse flags don't add back")
    }

    func testOpeningAdvancedSeedsFromTheCoarseResolution() {
        var caps = AgentCapabilities(tools: false, mcp: false, web: true)
        caps.openAdvanced()
        XCTAssertEqual(caps.advancedTools, [.browse, .webSearch],
                       "the two views can never disagree at the moment Advanced opens")
        // Idempotent: re-opening must not clobber a hand-edited set.
        caps.advancedTools = [.shell]
        caps.openAdvanced()
        XCTAssertEqual(caps.advancedTools, [.shell])
    }

    func testResolvedToolsAlwaysCarrySearchDocuments() {
        // The dispatch gate reads `ResolvedAgentSettings.tools`; docs-only mode
        // has Tools off yet must still be able to run searchDocuments.
        let a = agent { $0.capabilities = AgentCapabilities(tools: false, mcp: false, web: false) }
        let r = AgentResolution.resolve(agent: a, defaults: defaults())
        XCTAssertTrue(r.tools.contains(.searchDocuments))
        XCTAssertFalse(r.toolsEnabled)
    }
}
