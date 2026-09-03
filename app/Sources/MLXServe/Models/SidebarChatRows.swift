import Foundation

/// The Sessions section of the sidebar: conversations and sandbox terminals in
/// ONE list — newest first by default, or in the order the user dragged them
/// into (`AppState.sidebarOrder`).
enum SidebarChatRows {

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
    }

    /// Rows the order doesn't know yet (new ones) sit on top, newest first;
    /// the rest follow the order. Stale ids are skipped.
    static func merge(chats: [ChatSession], terminals: [TerminalSessionList.Session],
                      order: [UUID] = []) -> [Row] {
        let all = (chats.map(Row.chat) + terminals.map(Row.terminal))
            .sorted { $0.createdAt > $1.createdAt }
        return apply(order: order, to: all)
    }

    static func apply(order: [UUID], to rows: [Row]) -> [Row] {
        guard !order.isEmpty else { return rows }
        let byId = Dictionary(uniqueKeysWithValues: rows.map { ($0.id, $0) })
        let known = Set(order)
        return rows.filter { !known.contains($0.id) } + order.compactMap { byId[$0] }
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
}
