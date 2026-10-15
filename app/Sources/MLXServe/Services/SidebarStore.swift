import Foundation

/// The sidebar's groups and terminal rows in `mlx-serve.db`, on the chats'
/// connection (`ChatStore` owns it and the schema). A chat's place is a
/// column of its own row. Like the chats, a save writes only what changed and
/// never deletes a row it could not read.
final class SidebarStore {

    /// `TerminalSessionList.Session`'s stored coding keys and their columns.
    static let terminalColumns: [ChatStore.Column] = [
        .init(key: "id", name: "id", kind: .text),
        .init(key: "label", name: "label", kind: .text),
        .init(key: "autoName", name: "auto_name", kind: .text),
        .init(key: "customName", name: "custom_name", kind: .text),
        .init(key: "agentId", name: "agent_id", kind: .text),
        .init(key: "workspace", name: "workspace", kind: .text),
        .init(key: "kind", name: "kind", kind: .text),
        .init(key: "themeId", name: "theme_id", kind: .text),
        .init(key: "createdAt", name: "created_at", kind: .text),
        .init(key: "groupId", name: "group_id", kind: .text),
        .init(key: "position", name: "position", kind: .integer),
    ]

    private static let groupColumns = ["id", "name", "parent_id", "position", "collapsed"]

    let database: SQLiteDatabase

    /// Each row as it was when it last reached the disk.
    private var savedGroups: [UUID: [SQLiteDatabase.Value]] = [:]
    private var savedTerminals: [UUID: [SQLiteDatabase.Value]] = [:]
    /// Groups a save could not write yet. A row naming one fails its foreign
    /// key, so every row save retries them first.
    private var unsavedGroups: SidebarGroups?

    init(database: SQLiteDatabase) {
        self.database = database
    }

    // MARK: - Groups

    func loadGroups() -> SidebarGroups {
        var groups: [SidebarGroups.Group] = []
        do {
            try database.query("SELECT \(Self.groupColumns.joined(separator: ", ")) FROM sidebar_groups ORDER BY rowid") { row in
                guard let id = row.text(0).flatMap(UUID.init(uuidString:)), let name = row.text(1) else {
                    Self.log("skipped group \(row.text(0) ?? "?"): unreadable")
                    return
                }
                let group = SidebarGroups.Group(id: id, name: name, parentId: row.text(2).flatMap(UUID.init(uuidString:)),
                                                position: row.integer(3), collapsed: row.integer(4) != 0)
                savedGroups[id] = Self.values(of: group)
                groups.append(group)
            }
        } catch {
            Self.log("could not read the sidebar groups: \(error)")
        }
        return SidebarGroups(groups)
    }

    /// A parent before its children, so a new subtree satisfies its foreign keys.
    func saveGroups(_ groups: SidebarGroups) {
        let rows = groups.ordered.map { (id: $0.id, values: Self.values(of: $0)) }
        unsavedGroups = sync("sidebar_groups", columns: Self.groupColumns, rows: rows, saved: &savedGroups) ? nil : groups
    }

    /// Before a chat or terminal save (`unsavedGroups`).
    func retryUnsavedGroups() {
        if let unsavedGroups { saveGroups(unsavedGroups) }
    }

    private static func values(of group: SidebarGroups.Group) -> [SQLiteDatabase.Value] {
        [.text(group.id.uuidString), .text(group.name), group.parentId.map { .text($0.uuidString) } ?? .null,
         .integer(group.position), .integer(group.collapsed ? 1 : 0)]
    }

    // MARK: - Terminals

    /// In the order they were opened.
    func loadTerminals() -> [TerminalSessionList.Session] {
        var sessions: [TerminalSessionList.Session] = []
        do {
            let names = Self.terminalColumns.map(\.name).joined(separator: ", ")
            try database.query("SELECT \(names) FROM terminal_sessions ORDER BY created_at, rowid") { row in
                do {
                    let session = try ChatStore.decode(TerminalSessionList.Session.self, from: row, columns: Self.terminalColumns)
                    savedTerminals[session.id] = try ChatStore.columnValues(of: session, columns: Self.terminalColumns)
                    sessions.append(session)
                } catch {
                    Self.log("skipped terminal \(row.text(0) ?? "?"): \(error)")
                }
            }
        } catch {
            Self.log("could not read the terminal rows: \(error)")
        }
        return sessions
    }

    func saveTerminals(_ sessions: [TerminalSessionList.Session]) {
        retryUnsavedGroups()
        do {
            let rows = try sessions.map { (id: $0.id, values: try ChatStore.columnValues(of: $0, columns: Self.terminalColumns)) }
            sync("terminal_sessions", columns: Self.terminalColumns.map(\.name), rows: rows, saved: &savedTerminals)
        } catch {
            Self.log("could not save the terminal rows: \(error)")
        }
    }

    // MARK: - Writing

    /// One transaction per table: a failure leaves `saved` as it was, so the
    /// next save retries all of it. False when it failed.
    @discardableResult
    private func sync(_ table: String, columns: [String], rows: [(id: UUID, values: [SQLiteDatabase.Value])],
                      saved: inout [UUID: [SQLiteDatabase.Value]]) -> Bool {
        let live = Set(rows.map(\.id))
        let changed = rows.filter { saved[$0.id] != $0.values }
        guard !changed.isEmpty || saved.keys.contains(where: { !live.contains($0) }) else { return true }
        do {
            try database.transaction {
                for id in saved.keys where !live.contains(id) {
                    try database.run("DELETE FROM \(table) WHERE id = ?", [.text(id.uuidString)])
                }
                for row in changed {
                    if saved[row.id] == nil {
                        try insert(into: table, columns: columns, row.values)
                    } else {
                        let assignments = columns.dropFirst().map { "\($0) = ?" }.joined(separator: ", ")
                        try database.run("UPDATE \(table) SET \(assignments) WHERE id = ?",
                                         Array(row.values.dropFirst()) + [row.values[0]])
                    }
                }
            }
            saved = Dictionary(uniqueKeysWithValues: rows.map { ($0.id, $0.values) })
            return true
        } catch {
            Self.log("could not save \(table): \(error)")
            return false
        }
    }

    private func insert(into table: String, columns: [String], _ values: [SQLiteDatabase.Value]) throws {
        let marks = Array(repeating: "?", count: columns.count).joined(separator: ", ")
        try database.run("INSERT INTO \(table) (\(columns.joined(separator: ", "))) VALUES (\(marks))", values)
    }

    // MARK: - Moving off UserDefaults

    /// Runs inside the version 2 migration. The old global order becomes a
    /// position within each group (or the root), which draws the same sidebar;
    /// a value that does not read is logged and skipped, the rest still comes.
    func importLegacy(_ legacy: LegacySidebar) throws {
        // A repeated id keeps its first occurrence: a failed insert would roll
        // back the chats' import with it.
        var seenGroups = Set<UUID>(), seenTerminals = Set<UUID>()
        let groups = legacy.decodedGroups()
        for (position, group) in groups.groups.filter({ seenGroups.insert($0.id).inserted }).enumerated() {
            try insert(into: "sidebar_groups", columns: Self.groupColumns,
                       Self.values(of: .init(id: group.id, name: group.name, position: position,
                                             collapsed: group.collapsed ?? false)))
        }
        let known = Set(groups.groups.map(\.id))
        let membership = groups.membership.filter { known.contains($0.value) }
        let terminals = legacy.decodedTerminals().filter { seenTerminals.insert($0.id).inserted }

        var chats: [UUID] = []
        try database.query("SELECT id FROM chat_sessions") { row in
            if let id = row.text(0).flatMap(UUID.init(uuidString:)) { chats.append(id) }
        }
        let rows = Set(terminals.map(\.id) + chats)
        var positions: [UUID: Int] = [:]
        var next: [UUID?: Int] = [:]
        for id in legacy.order where rows.contains(id) && positions[id] == nil {
            positions[id] = next[membership[id], default: 0]
            next[membership[id], default: 0] += 1
        }

        for var session in terminals {
            session.groupId = membership[session.id]
            session.position = positions[session.id]
            try insert(into: "terminal_sessions", columns: Self.terminalColumns.map(\.name),
                       ChatStore.columnValues(of: session, columns: Self.terminalColumns))
        }
        for id in chats where membership[id] != nil || positions[id] != nil {
            try database.run("UPDATE chat_sessions SET group_id = ?, position = ? WHERE id = ?",
                             [membership[id].map { .text($0.uuidString) } ?? .null,
                              positions[id].map { .integer($0) } ?? .null, .text(id.uuidString)])
        }
    }

    private static func log(_ message: String) {
        NSLog("[sidebar] %@", message)
    }
}

/// The sidebar state older builds kept in UserDefaults: read for the version 2
/// import, then retired once the database holds it.
struct LegacySidebar {

    static let orderKey = "sidebarRowOrder"
    static let groupsKey = "sidebarGroups"
    static let terminalsKey = "terminalSessions"
    static let keys = [orderKey, groupsKey, terminalsKey]

    static var backupPath: String { ChatStore.inAppFolder("sidebar-defaults.migrated.json") }

    /// The dragged order over chats and terminals, the whole panel in one list.
    var order: [UUID] = []
    /// JSON `{groups: [{id, name, collapsed}], membership: [row, group, …]}`.
    var groups: Data?
    /// JSON `TerminalSessionList`.
    var terminals: Data?

    init(order: [UUID] = [], groups: Data? = nil, terminals: Data? = nil) {
        self.order = order
        self.groups = groups
        self.terminals = terminals
    }

    init(defaults: UserDefaults) {
        order = (defaults.stringArray(forKey: Self.orderKey) ?? []).compactMap(UUID.init(uuidString:))
        groups = defaults.data(forKey: Self.groupsKey)
        terminals = defaults.data(forKey: Self.terminalsKey)
    }

    struct Groups: Decodable {
        struct Group: Decodable {
            let id: UUID
            let name: String
            let collapsed: Bool?
        }
        var groups: [Group] = []
        var membership: [UUID: UUID] = [:]
    }

    func decodedGroups() -> Groups {
        guard let groups else { return Groups() }
        do {
            return try JSONDecoder().decode(Groups.self, from: groups)
        } catch {
            NSLog("[sidebar] %@", "the old sidebar groups do not read, not imported: \(error)")
            return Groups()
        }
    }

    func decodedTerminals() -> [TerminalSessionList.Session] {
        guard let terminals else { return [] }
        do {
            return try JSONDecoder().decode(TerminalSessionList.self, from: terminals).sessions
        } catch {
            NSLog("[sidebar] %@", "the old terminal rows do not read, not imported: \(error)")
            return []
        }
    }

    /// After the import has committed: the old values go to a backup file (an
    /// earlier backup is kept, this one takes a dated name), and only once it
    /// is on disk are the keys removed.
    static func retire(from defaults: UserDefaults, backupPath: String) {
        var backup: [String: Any] = [:]
        if let order = defaults.stringArray(forKey: orderKey) { backup[orderKey] = order }
        for key in [groupsKey, terminalsKey] {
            guard let data = defaults.data(forKey: key) else { continue }
            backup[key] = (try? JSONSerialization.jsonObject(with: data)) ?? data.base64EncodedString()
        }
        guard !backup.isEmpty else { return }
        var target = backupPath
        if FileManager.default.fileExists(atPath: target) {
            let stamp = DateFormatter()
            stamp.locale = Locale(identifier: "en_US_POSIX")
            stamp.dateFormat = "yyyyMMdd-HHmmss"
            target = (backupPath as NSString).deletingPathExtension + "-\(stamp.string(from: Date())).json"
        }
        do {
            let data = try JSONSerialization.data(withJSONObject: backup, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: target), options: .withoutOverwriting)
        } catch {
            NSLog("[sidebar] %@", "could not back up the old sidebar to \(target), keys kept: \(error)")
            return
        }
        for key in keys { defaults.removeObject(forKey: key) }
    }
}
