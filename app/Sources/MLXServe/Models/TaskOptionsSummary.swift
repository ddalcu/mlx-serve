import Foundation

/// The one-line summary under the New Task sheet's collapsed "Options" row.
///
/// Progressive disclosure creates exactly one hazard: a setting that IS set but
/// invisible, because nobody expands a row that looks empty. So the row names
/// every collapsed option that DIFFERS from its default — and returns nil when
/// none of them do, because "Options" with a blank line under it reads as a
/// section that failed to load rather than one with nothing to say.
///
/// Same rule, same shape as `AgentAdvancedSummary`; the two are deliberately
/// siblings so the app has ONE idea of what a collapsed section owes the user.
enum TaskOptionsSummary {

    /// - Parameters:
    ///   - agentName: the chosen agent's name, or nil for "app defaults".
    ///   - modelName: the pinned model's name, or nil for "use current model".
    ///   - useMCP: whether MCP tools were switched on (defaults off).
    static func text(agentName: String?, modelName: String?, useMCP: Bool) -> String? {
        var parts: [String] = []
        // The VALUE, not the field name: "Chef" says more than "Agent", and the
        // point of the line is to answer "what did I set?" without expanding.
        if let agentName = nonEmpty(agentName) { parts.append(agentName) }
        if let modelName = nonEmpty(modelName) { parts.append(modelName) }
        if useMCP { parts.append("MCP") }

        guard !parts.isEmpty else { return nil }
        return parts.joined(separator: " · ")
    }

    /// A field cleared to "" (or whitespace) is not a choice — treating it as
    /// one would claim a customization that isn't there.
    private static func nonEmpty(_ value: String?) -> String? {
        guard let value else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
