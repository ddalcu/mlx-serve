import Foundation

/// The ONE place an agent's overrides meet the app's defaults.
///
/// The rule this file exists to enforce: **a turn site never reads a global
/// setting directly — it reads `ResolvedAgentSettings`.** Same class as the LAN
/// rule (chat routes through `server.chatModelId`, never `modelInfo?.name` at a
/// new surface): a per-surface read is exactly how one surface ends up silently
/// ignoring the agent.
///
/// Pure and `nonisolated` so it's testable with no app instance.
struct AppDefaultsSnapshot: Sendable, Equatable {
    /// The surface's own toggles (chat tab's Tools/MCP/Think, the tray's, a
    /// task's) — what the turn would have used before agents existed.
    var toolsEnabled: Bool = false
    var mcpEnabled: Bool = false
    var thinkingEnabled: Bool = false
    var autoApprove: Bool = false
    /// Full access is the historical default: the whole tool literal is
    /// advertised whenever the loop runs.
    var tools: Set<AgentToolKind> = Set(AgentToolKind.allCases)
    var workingDirectory: String?
    /// nil = whatever model is selected right now.
    var modelPath: String?
    var temperature: Double = 0.8
    var maxTokens: Int = 4096
    /// The app's global voice (engine + value).
    var voice: AgentVoice?
    var wakePhrase: String = WakeWord.defaultPhrase
}

/// Every field decided — no optionals left except the ones that are genuinely
/// "unset" downstream too (a nil working directory, a nil model = current).
struct ResolvedAgentSettings: Sendable, Equatable {
    var agentId: UUID?
    /// The persona, ready to be prepended to the STABLE system-prompt prefix
    /// (never the volatile tail — that re-prefills every turn). Carries its own
    /// trailing blank line, and is "" when there's no agent.
    var systemPromptPrefix: String = ""
    var tools: Set<AgentToolKind> = []
    var toolsEnabled: Bool = false
    var mcpEnabled: Bool = false
    var thinkingEnabled: Bool = false
    var autoApprove: Bool = false
    var workingDirectory: String?
    var modelPath: String?
    var temperature: Double = 0.8
    var maxTokens: Int = 4096
    /// The agent's RAW sampling overrides, nil when it didn't set them.
    ///
    /// Kept alongside the decided values above because the two chat paths have
    /// DIFFERENT defaults (the tool loop's temperature is not plain chat's), so a
    /// decided number handed to `TurnConfig` would silently change one of them for
    /// every user who never makes an agent. nil = "this path's own default".
    var temperatureOverride: Double?
    var maxTokensOverride: Int?
    var voice: AgentVoice?
    /// The AGENT's own voice, nil when it didn't pick one.
    ///
    /// Same reason the sampling overrides are kept raw: the synthesizer re-reads
    /// the user's voice settings once per utterance, so publishing the
    /// global-fallback value would PIN the turn's voice and stop a mid-answer
    /// Settings change from applying. nil = follow Settings, live.
    var voiceOverride: AgentVoice?
    var wakePhrase: String = WakeWord.defaultPhrase
}

enum AgentResolution {

    /// Fold an agent's overrides into the app defaults. `agent == nil` returns
    /// the defaults verbatim — that's the upgrade guarantee, and it has its own
    /// test.
    nonisolated static func resolve(agent: Agent?, defaults: AppDefaultsSnapshot) -> ResolvedAgentSettings {
        guard let agent else {
            return ResolvedAgentSettings(
                agentId: nil,
                systemPromptPrefix: "",
                tools: defaults.tools,
                toolsEnabled: defaults.toolsEnabled,
                mcpEnabled: defaults.mcpEnabled,
                thinkingEnabled: defaults.thinkingEnabled,
                autoApprove: defaults.autoApprove,
                workingDirectory: defaults.workingDirectory,
                modelPath: defaults.modelPath,
                temperature: defaults.temperature,
                maxTokens: defaults.maxTokens,
                voice: defaults.voice,
                wakePhrase: defaults.wakePhrase
            )
        }

        // The persona is the prompt TEXT, verbatim — nothing is bolted on here.
        // Brevity is written into the prompt when a model writes it
        // (`AgentWriter.brevityLine`), so it's visible and editable rather than a
        // hidden setting.
        let persona = agent.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let capabilityTools = agent.capabilities.resolvedTools()

        return ResolvedAgentSettings(
            agentId: agent.id,
            systemPromptPrefix: persona.isEmpty ? "" : persona + "\n\n",
            // searchDocuments rides along unconditionally: its real gate is
            // whether a document folder is attached to the turn, which is
            // stronger than any capability toggle (docs-only mode has Tools off
            // and still needs it).
            tools: capabilityTools.union([.searchDocuments]),
            // The loop runs whenever the agent has any tool at all — an
            // advertised tool the loop never executes would be a dead offer.
            toolsEnabled: !capabilityTools.isEmpty,
            mcpEnabled: agent.capabilities.mcp,
            thinkingEnabled: agent.enableThinking ?? defaults.thinkingEnabled,
            autoApprove: agent.autoApproveTools ?? defaults.autoApprove,
            workingDirectory: agent.workingDirectory ?? defaults.workingDirectory,
            modelPath: agent.modelPath ?? defaults.modelPath,
            temperature: agent.temperature ?? defaults.temperature,
            maxTokens: agent.maxTokens ?? defaults.maxTokens,
            temperatureOverride: agent.temperature,
            maxTokensOverride: agent.maxTokens,
            voice: agent.resolvedVoice ?? defaults.voice,
            voiceOverride: agent.resolvedVoice,
            wakePhrase: agent.wakePhrase.flatMap(WakeWord.normalizePhrase) ?? defaults.wakePhrase
        )
    }
}
