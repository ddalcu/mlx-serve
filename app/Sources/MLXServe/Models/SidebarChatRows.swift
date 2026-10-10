import Foundation

/// The Sessions section of the sidebar: conversations and sandbox terminals in
/// ONE list. Within a group (or the root) the rows the user dragged follow
/// their positions, and the rest sit above them, newest first.
enum SidebarChatRows {

    /// Where a row sits: its group (nil = the root) and its position among the
    /// group's rows (nil = never dragged).
    struct Placement: Equatable {
        var group: UUID?
        var position: Int?
    }

    enum Row: Identifiable {
        case chat(ChatSession)
        case terminal(TerminalSessionList.Session)

        var id: UUID {
            switch self {
            case .chat(let s): return s.id
            case .terminal(let t): return t.id
            }
        }

        var createdAt: Date {
            switch self {
            case .chat(let s): return s.createdAt
            case .terminal(let t): return t.createdAt
            }
        }

        var groupId: UUID? {
            switch self {
            case .chat(let s): return s.groupId
            case .terminal(let t): return t.groupId
            }
        }

        var position: Int? {
            switch self {
            case .chat(let s): return s.sidebarPosition
            case .terminal(let t): return t.position
            }
        }
    }

    /// Positions count within a group, and splitting the list by group keeps
    /// this order, so one sort serves every group. Both lists newest first, so
    /// rows from the same second keep that (terminals arrive oldest first).
    static func merge(chats: [ChatSession], terminals: [TerminalSessionList.Session]) -> [Row] {
        (chats.map(Row.chat) + terminals.reversed().map(Row.terminal)).sorted { a, b in
            switch (a.position, b.position) {
            case (nil, nil): return a.createdAt > b.createdAt
            case (nil, _?): return true
            case (_?, nil): return false
            case let (x?, y?): return x < y
            }
        }
    }

    /// Drop `moved` into `target`'s slot; the rest keep their relative order.
    static func moved(_ moved: UUID, onto target: UUID, in ids: [UUID]) -> [UUID] {
        guard moved != target, let from = ids.firstIndex(of: moved),
              ids.contains(target) else { return ids }
        var out = ids
        out.remove(at: from)
        guard let to = out.firstIndex(of: target) else { return ids }
        out.insert(moved, at: from < to ? to + 1 : to)
        return out
    }

    /// A row dropped onto another joins that row's group in its slot, and the
    /// group's rows are numbered in the order they now show. Dropped on its
    /// neighbour across a group's edge, only the group changes. `visible` is
    /// the panel in its current order.
    static func dropped(_ id: UUID, onto target: UUID, in visible: [Row]) -> [UUID: Placement] {
        guard id != target, let row = visible.first(where: { $0.id == id }),
              let group = visible.first(where: { $0.id == target }).map(\.groupId) else { return [:] }
        let order = moved(id, onto: target, in: visible.map(\.id))
        guard order != visible.map(\.id) || row.groupId != group else { return [:] }
        let inGroup = Set(visible.filter { $0.groupId == group }.map(\.id)).union([id])
        var placements: [UUID: Placement] = [:]
        for rowId in order where inGroup.contains(rowId) {
            placements[rowId] = Placement(group: group, position: placements.count)
        }
        return placements
    }

    /// A deleted group's rows move to its parent: the dragged ones first among
    /// the parent's dragged rows, the others among its undragged ones by date.
    /// `rows` is in sidebar order.
    static func dissolving(_ group: UUID, into parent: UUID?, rows: [Row]) -> [UUID: Placement] {
        let leaving = rows.filter { $0.groupId == group }
        let placed = leaving.filter { $0.position != nil } + rows.filter { $0.groupId == parent && $0.position != nil }
        var placements: [UUID: Placement] = [:]
        for (position, row) in placed.enumerated() {
            placements[row.id] = Placement(group: parent, position: position)
        }
        for row in leaving where row.position == nil {
            placements[row.id] = Placement(group: parent, position: nil)
        }
        return placements
    }
}
