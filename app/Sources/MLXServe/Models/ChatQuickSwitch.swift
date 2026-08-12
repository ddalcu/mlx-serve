import Foundation

/// ⌘1…⌘9 over the sidebar's conversation list: which row wears which number,
/// and which chat a digit reaches.
///
/// The ordering is DERIVED from `SidebarSessionGroups.split`, never rebuilt
/// here. The sidebar draws agent threads above plain chats, so numbering the
/// raw `visibleChatSessions` would put ⌘1 on whichever thread is newest — a
/// badge that names a shortcut landing somewhere else. One ordering, and a
/// section added to the sidebar carries the numbers with it.
enum ChatQuickSwitch {

    /// The keyboard has nine of these. A tenth row draws no badge rather than
    /// one promising a shortcut nothing can press.
    static let maxSlots = 9

    /// The conversation rows top to bottom, exactly as the sidebar draws them.
    static func ordered(_ sessions: [ChatSession]) -> [ChatSession] {
        let groups = SidebarSessionGroups.split(sessions)
        return groups.agents + groups.chats
    }

    /// The 1-based number drawn on a row, or nil past the ninth.
    static func slot(for id: UUID, in sessions: [ChatSession]) -> Int? {
        guard let index = ordered(sessions).prefix(maxSlots).firstIndex(where: { $0.id == id })
        else { return nil }
        return index + 1
    }

    /// The chat a digit reaches, or nil when that row isn't there.
    static func id(forSlot slot: Int, in sessions: [ChatSession]) -> UUID? {
        guard slot >= 1, slot <= maxSlots else { return nil }
        let rows = ordered(sessions)
        guard slot <= rows.count else { return nil }
        return rows[slot - 1].id
    }
}
