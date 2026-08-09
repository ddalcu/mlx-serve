import Foundation

/// What an empty conversation says above its composer, and whether it offers
/// the discovery chips underneath.
///
/// An agent thread is not a blank app: you have already chosen who you are
/// talking to, so it is named for THEM and says what they are for. The chips
/// advertise what the app can do — on an agent thread three of the four
/// navigate out of the conversation and the fourth rewires the composer into a
/// generator the agent has no part in. Nothing is lost by hiding them: the
/// Tools menu is the always-available twin (pinned by `ChatGreetingTests`).
enum ChatGreeting {
    static let plainHeading = "How can I help you today?"
    /// Outranks the brief: it is the one thing that must be acted on first.
    static let serverStopped = "Start the server to begin."

    static func heading(agentName: String?) -> String {
        agentName?.trimmedNonEmpty ?? plainHeading
    }

    /// What this agent is FOR — the line that tells you what to ask it.
    static func subtitle(agentBrief: String?, serverRunning: Bool) -> String? {
        serverRunning ? agentBrief?.trimmedNonEmpty : serverStopped
    }

    /// An external bridge (Telegram) is read-only and never offered them.
    static func showsDiscoveryChips(hasAgent: Bool, isExternalBridge: Bool) -> Bool {
        !hasAgent && !isExternalBridge
    }
}
