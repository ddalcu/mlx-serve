import Foundation

/// One line naming what an agent has changed behind the editor's collapsed
/// "More options".
///
/// Capabilities, Model, Workspace and Sampling are collapsed because most
/// agents are a prompt and a voice, and the rest is noise on first use. The
/// hazard that creates is a setting that IS set but invisible — nobody expands
/// a row that looks empty. So the row carries this: the collapsed areas that
/// differ from the defaults, in the order their sections appear, or nil when
/// there is genuinely nothing to see.
enum AgentAdvancedSummary {

    static func text(for agent: Agent) -> String? {
        var parts: [String] = []

        // Capabilities. `web` defaults ON, so its absence is the change worth
        // naming; the other two default off.
        let caps = agent.capabilities
        if caps.tools { parts.append("Tools") }
        if caps.mcp { parts.append("MCP") }
        if !caps.web { parts.append("No web") }
        if caps.advancedTools != nil { parts.append("custom tools") }
        // Tri-state (nil = follow the app), so any value is a deliberate choice.
        if agent.enableThinking != nil { parts.append("Thinking") }
        if agent.autoApproveTools != nil { parts.append("tool approval") }

        if isSet(agent.modelPath) { parts.append("Model") }
        if isSet(agent.workingDirectory) { parts.append("Workspace") }
        // One concept in the UI — naming both knobs lengthens the line without
        // adding information.
        if agent.temperature != nil || agent.maxTokens != nil { parts.append("Sampling") }

        guard !parts.isEmpty else { return nil }
        // Sentence-style: first item capitalized as written, the rest lowered,
        // so the line reads as a list rather than a row of headings.
        let head = parts[0]
        let tail = parts.dropFirst().map { $0.lowercasedFirst() }
        return ([head] + tail).joined(separator: ", ")
    }

    /// A cleared field decodes as "" (or whitespace) from an older build;
    /// treating that as a pin would claim a customization that isn't there.
    private static func isSet(_ value: String?) -> Bool {
        guard let value else { return false }
        return !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

private extension String {
    /// Lowercase only the FIRST character — "No web" → "no web", while
    /// "MCP" must not become "mcp".
    func lowercasedFirst() -> String {
        guard let first, first.isUppercase, !allSatisfy({ $0.isUppercase || !$0.isLetter })
        else { return self }
        return first.lowercased() + dropFirst()
    }
}
