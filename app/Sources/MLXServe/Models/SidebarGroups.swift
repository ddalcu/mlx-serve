import Foundation

/// User-made sidebar groups: named, collapsible folders over chats, agent
/// threads and terminals. A group sits in a parent group (nil = the root) at a
/// position among its siblings; each row names its own group. Stored in
/// `sidebar_groups` (`SidebarStore`).
struct SidebarGroups: Equatable {

    struct Group: Identifiable, Equatable {
        let id: UUID
        var name: String
        var parentId: UUID? = nil
        var position: Int
        var collapsed = false
    }

    private(set) var groups: [Group]

    init(_ groups: [Group] = []) {
        self.groups = groups
    }

    /// The parent the group shows under. A link that names a missing group or
    /// closes a cycle is cut, so every group shows somewhere.
    func parent(of id: UUID) -> UUID? {
        guard let group = groups.first(where: { $0.id == id }), let parentId = group.parentId else { return nil }
        var seen = Set<UUID>()
        var current: UUID? = parentId
        while let step = current, seen.insert(step).inserted {
            guard let node = groups.first(where: { $0.id == step }) else { return step == parentId ? nil : parentId }
            if step == id { return nil }
            current = node.parentId
        }
        return parentId
    }

    /// In sidebar order: by position, a tie keeping the stored order.
    func children(of parent: UUID?) -> [Group] {
        groups.filter { self.parent(of: $0.id) == parent }.sorted { $0.position < $1.position }
    }

    /// Every group, each followed by its subtree.
    var ordered: [Group] {
        func subtree(_ parent: UUID?) -> [Group] {
            children(of: parent).flatMap { [$0] + subtree($0.id) }
        }
        return subtree(nil)
    }

    /// Last among its siblings; nil when the name is blank.
    @discardableResult
    mutating func create(_ name: String, in parent: UUID? = nil) -> UUID? {
        let name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }
        let position = (children(of: parent).map(\.position).max() ?? -1) + 1
        let group = Group(id: UUID(), name: name, parentId: parent, position: position)
        groups.append(group)
        return group.id
    }

    mutating func rename(_ id: UUID, to name: String) {
        let name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, let i = groups.firstIndex(where: { $0.id == id }) else { return }
        groups[i].name = name
    }

    /// Its subgroups take its place among its siblings. Its rows are the
    /// caller's (`SidebarChatRows.dissolving`).
    mutating func delete(_ id: UUID) {
        guard groups.contains(where: { $0.id == id }) else { return }
        let parent = parent(of: id)
        let order = children(of: parent).flatMap { $0.id == id ? children(of: id) : [$0] }.map(\.id)
        groups.removeAll { $0.id == id }
        for (position, child) in order.enumerated() {
            guard let i = groups.firstIndex(where: { $0.id == child }) else { continue }
            groups[i].parentId = parent
            groups[i].position = position
        }
    }

    mutating func toggleCollapsed(_ id: UUID) {
        guard let i = groups.firstIndex(where: { $0.id == id }) else { return }
        groups[i].collapsed.toggle()
    }

    /// Groups in sidebar order (empty ones too), each with its rows in the
    /// order given; rows naming no known group stay where they were.
    func partition(_ rows: [SidebarChatRows.Row])
        -> (groups: [(group: Group, rows: [SidebarChatRows.Row])], ungrouped: [SidebarChatRows.Row]) {
        let known = Set(groups.map(\.id))
        var byGroup: [UUID: [SidebarChatRows.Row]] = [:]
        var ungrouped: [SidebarChatRows.Row] = []
        for row in rows {
            if let g = row.groupId, known.contains(g) {
                byGroup[g, default: []].append(row)
            } else {
                ungrouped.append(row)
            }
        }
        return (ordered.map { ($0, byGroup[$0.id] ?? []) }, ungrouped)
    }
}
