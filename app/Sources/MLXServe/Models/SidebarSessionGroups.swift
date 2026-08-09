import Foundation

/// How the sidebar's conversation list is divided: agent threads above, plain
/// chats below.
enum SidebarSessionGroups {

    /// - Returns: the agent-owned threads and the plain ones, each in the order
    ///   they arrived (the list is already newest-first).
    static func split(_ sessions: [ChatSession]) -> (agents: [ChatSession], chats: [ChatSession]) {
        var agents: [ChatSession] = []
        var chats: [ChatSession] = []
        for session in sessions {
            // Keyed on the session's OWN agentId, never on whether that agent
            // still exists: a thread created as an agent's stays one after the
            // agent is deleted. Resolving the name is the row's problem, and it
            // already drops the subtitle when there is nobody to name.
            if session.agentId != nil {
                agents.append(session)
            } else {
                chats.append(session)
            }
        }
        return (agents, chats)
    }
}
