import Foundation

/// User-made sidebar groups: named, collapsible folders over chats, agent
/// threads and terminals. A row sits in at most one group; deleting a group
/// only ungroups its rows. Persisted by `AppState.sidebarGroups`.
struct SidebarGroups: Codable, Equatable {

    struct Group: Identifiable, Codable, Equatable {
        let id: UUID
        var name: String
        var collapsed = false
    }

    private(set) var groups: [Group] = []
    /// Row id -> group id.
    private var membership: [UUID: UUID] = [:]

    func group(of row: UUID) -> UUID? { membership[row] }

    /// nil when the name is blank.
    @discardableResult
    mutating func create(_ name: String, with rows: some Sequence<UUID>) -> UUID? {
        let name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }
        let group = Group(id: UUID(), name: name)
        groups.append(group)
        assign(rows, to: group.id)
        return group.id
    }

    mutating func rename(_ id: UUID, to name: String) {
        let name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, let i = groups.firstIndex(where: { $0.id == id }) else { return }
        groups[i].name = name
    }

    mutating func delete(_ id: UUID) {
        groups.removeAll { $0.id == id }
        membership = membership.filter { $0.value != id }
    }

    /// nil takes the rows out of whatever group they are in.
    mutating func assign(_ rows: some Sequence<UUID>, to group: UUID?) {
        for row in rows { membership[row] = group }
    }

    /// A row dropped onto another takes that row's group (none included).
    mutating func join(_ row: UUID, groupOf target: UUID) {
        membership[row] = membership[target]
    }

    mutating func toggleCollapsed(_ id: UUID) {
        guard let i = groups.firstIndex(where: { $0.id == id }) else { return }
        groups[i].collapsed.toggle()
    }

    /// Drop memberships of rows that no longer exist (terminals end at quit).
    mutating func retain(only live: Set<UUID>) {
        membership = membership.filter { live.contains($0.key) }
    }

    /// Groups in creation order (empty ones too), each with its rows in the
    /// order given; the rest stay where they were.
    func partition(_ rows: [SidebarChatRows.Row])
        -> (groups: [(group: Group, rows: [SidebarChatRows.Row])], ungrouped: [SidebarChatRows.Row]) {
        var byGroup: [UUID: [SidebarChatRows.Row]] = [:]
        var ungrouped: [SidebarChatRows.Row] = []
        for row in rows {
            if let g = membership[row.id], groups.contains(where: { $0.id == g }) {
                byGroup[g, default: []].append(row)
            } else {
                ungrouped.append(row)
            }
        }
        return (groups.map { ($0, byGroup[$0.id] ?? []) }, ungrouped)
    }
}
