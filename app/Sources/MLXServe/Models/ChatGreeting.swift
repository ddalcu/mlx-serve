import Foundation

/// What an empty conversation says above its composer, and whether it offers
/// the discovery chips underneath.
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
