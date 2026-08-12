import Foundation

/// ⌘1…⌘9 over the sidebar's conversation list: which row wears which number,
/// and which chat a digit reaches.
///
/// The ordering is DERIVED from `SidebarSessionGroups.split`, never rebuilt
/// here. The sidebar draws agent threads above plain chats, so numbering the
/// raw `visibleChatSessions` would put ⌘1 on whichever thread is newest — a
/// badge that names a shortcut landing somewhere else. One ordering, and a
/// section added to the sidebar carries the numbers with it.
///
/// The nine numbers go to rows you can SEE. The conversation list scrolls under
/// the pinned destination block, so a scrolled sidebar had ⌘1…⌘5 sitting behind
/// frosted glass — five of nine shortcuts spent on rows the user cannot read,
/// while the rows in front of them started at 6. `numbering` is that filter:
/// the caller passes the rows currently in the clear and the numbers land on
/// them in the same top-to-bottom order.
enum ChatQuickSwitch {

    /// The keyboard has nine of these. A tenth row draws no badge rather than
    /// one promising a shortcut nothing can press.
    static let maxSlots = 9

    /// The conversation rows top to bottom, exactly as the sidebar draws them.
    static func ordered(_ sessions: [ChatSession]) -> [ChatSession] {
        let groups = SidebarSessionGroups.split(sessions)
        return groups.agents + groups.chats
    }

    /// The rows that get a number, top to bottom.
    ///
    /// - Parameter numbering: the rows eligible for a number — those clear of
    ///   the sidebar's frosted overlay. `nil` means "no visibility information",
    ///   which numbers everything: that is the honest answer before the first
    ///   layout has reported, and it is what a caller with no overlay wants.
    ///   An EMPTY set is different and means what it says — nothing is in the
    ///   clear, so nothing is numbered.
    static func numbered(_ sessions: [ChatSession], numbering: Set<UUID>? = nil) -> [ChatSession] {
        let rows = ordered(sessions)
        guard let numbering else { return Array(rows.prefix(maxSlots)) }
        return Array(rows.filter { numbering.contains($0.id) }.prefix(maxSlots))
    }

    /// The 1-based number drawn on a row, or nil when it gets none.
    static func slot(for id: UUID, in sessions: [ChatSession],
                     numbering: Set<UUID>? = nil) -> Int? {
        guard let index = numbered(sessions, numbering: numbering)
            .firstIndex(where: { $0.id == id })
        else { return nil }
        return index + 1
    }

    /// The chat a digit reaches, or nil when no row wears that number.
    ///
    /// Reads the same `numbered` list the badge was drawn from, so the digit
    /// cannot land on a different row than the one showing it.
    static func id(forSlot slot: Int, in sessions: [ChatSession],
                   numbering: Set<UUID>? = nil) -> UUID? {
        guard slot >= 1, slot <= maxSlots else { return nil }
        let rows = numbered(sessions, numbering: numbering)
        guard slot <= rows.count else { return nil }
        return rows[slot - 1].id
    }
}
