import Foundation

/// What an empty conversation says above its composer, and whether it offers
/// the discovery chips underneath.
///
/// Pure, because the interesting part is a precedence and a gate rather than a
/// layout: a thread belonging to an AGENT is not a blank app. You have already
/// chosen who you are talking to, so the greeting names THEM (the name is the
/// thing — "Agent" was the 30pt word with the name as its caption, the same
/// inversion the sidebar rows had) and says what they are for.
///
/// The chips advertise what the APP can do — media generation, the Model
/// Browser, Tasks, the CLI launcher — because those lived only in the menu-bar
/// tray where nobody found them. That is a job for a blank chat, done once. On
/// an agent thread three of the four navigate OUT of the conversation and the
/// fourth rewires the composer into a generator the agent has no part in, all
/// directly above the field where you were about to type to someone specific —
/// and an agent whose capabilities exclude image generation cannot do what
/// "Create Media" sits there offering, which is the locked-composer-disc rule
/// (never offer what the resolver will refuse). Nothing is lost by hiding
/// them: `CommandMenu("Tools")` is their always-available twin and iterates the
/// same catalog, pinned by `ChatGreetingTests`.
enum ChatGreeting {

    /// The app's own greeting, for a conversation with nobody in particular.
    static let plainHeading = "How can I help you today?"

    /// The one thing on this screen the user has to act on before anything
    /// else works, so it outranks whatever else the caption would have said.
    static let serverStopped = "Start the server to begin."

    /// The heading: the agent's name, or the app's greeting.
    ///
    /// A half-saved agent (blank name, mid-rename) can't head a screen with an
    /// empty string, so it falls through — the same rule as
    /// `ChatSessionTitle.display`.
    static func heading(agentName: String?) -> String {
        trimmed(agentName) ?? plainHeading
    }

    /// The line under the heading: what this agent is FOR, which is what tells
    /// you what to ask it. Nil rather than an empty line holding space when the
    /// agent has no description.
    static func subtitle(agentBrief: String?, serverRunning: Bool) -> String? {
        if !serverRunning { return serverStopped }
        return trimmed(agentBrief)
    }

    /// The discovery chips belong to a blank app, not to a conversation with
    /// somebody. An external bridge (Telegram) is read-only and never offered
    /// them.
    static func showsDiscoveryChips(hasAgent: Bool, isExternalBridge: Bool) -> Bool {
        !hasAgent && !isExternalBridge
    }

    private static func trimmed(_ value: String?) -> String? {
        guard let value else { return nil }
        let t = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? nil : t
    }
}
