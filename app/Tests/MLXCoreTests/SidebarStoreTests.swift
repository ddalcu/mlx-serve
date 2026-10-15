import XCTest
@testable import MLXCore

/// Groups, row places and terminal rows in `mlx-serve.db`: the same sidebar after the move off UserDefaults, nothing lost on the way.
final class SidebarStoreTests: XCTestCase {

    private var dir: String!
    private var suite: String!
    private var defaults: UserDefaults!

    override func setUpWithError() throws {
        dir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mlx-core-sidebar-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        suite = "mlx-core-sidebar-tests-\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suite)
    }

    override func tearDownWithError() throws {
        defaults.removePersistentDomain(forName: suite)
        try? FileManager.default.removeItem(atPath: dir)
    }

    private func path(_ name: String) -> String { (dir as NSString).appendingPathComponent(name) }
    private var dbPath: String { path("mlx-serve.db") }
    private var backupPath: String { path("sidebar-defaults.migrated.json") }

    private func open(_ legacy: LegacySidebar? = nil) throws -> (chats: ChatStore, sidebar: SidebarStore) {
        let chats = try ChatStore(path: dbPath, legacyHistoryPath: path("chat-history.json"), legacySidebar: legacy)
        return (chats, SidebarStore(database: chats.database))
    }

    private func at(_ seconds: TimeInterval) -> Date {
        Date(timeIntervalSince1970: 1_790_000_000 + seconds)
    }

    private func chat(_ title: String, created: TimeInterval) -> ChatSession {
        var s = ChatSession(title: title)
        s.createdAt = at(created)
        s.updatedAt = at(created)
        return s
    }

    private func terminal(_ autoName: String, label: String = "pi", created: TimeInterval,
                          kind: TerminalSessionList.Session.Kind = .sandbox) -> TerminalSessionList.Session {
        TerminalSessionList.Session(id: UUID(), label: label, autoName: autoName, agentId: label,
                                    workspace: "/w/\(autoName)", createdAt: at(created), kind: kind, phase: .live)
    }

    /// The stored fields only, in one comparable shape.
    private func json(_ sessions: [TerminalSessionList.Session]) throws -> String {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        return String(decoding: try encoder.encode(sessions), as: UTF8.self)
    }

    /// A database as the build before this one left it.
    private func makeVersion1Database(_ chats: [ChatSession]) throws {
        let database = try SQLiteDatabase(path: dbPath)
        try database.execute(ChatStore.schemaV1)
        let columns = ChatStore.sessionColumns.filter { $0.key != "groupId" && $0.key != "sidebarPosition" }
        for chat in chats {
            let names = columns.map(\.name).joined(separator: ", ")
            let marks = Array(repeating: "?", count: columns.count).joined(separator: ", ")
            try database.run("INSERT INTO chat_sessions (\(names)) VALUES (\(marks))",
                             try ChatStore.columnValues(of: chat, columns: columns))
        }
        try database.setUserVersion(1)
    }

    // MARK: - The old UserDefaults values, in the shapes the old build wrote

    private struct OldGroups: Encodable {
        struct Group: Encodable {
            let id: UUID
            let name: String
            let collapsed: Bool
        }
        let groups: [Group]
        let membership: [UUID: UUID]
    }

    private func oldGroups(_ groups: [OldGroups.Group], membership: [UUID: UUID]) throws -> Data {
        try JSONEncoder().encode(OldGroups(groups: groups, membership: membership))
    }

    private struct OldTerminals: Encodable {
        let sessions: [TerminalSessionList.Session]
        let ordinals: [String: Int]
    }

    private func oldTerminals(_ sessions: [TerminalSessionList.Session]) throws -> Data {
        try JSONEncoder().encode(OldTerminals(sessions: sessions, ordinals: ["pi": sessions.count]))
    }

    // MARK: - Upgrading a version 1 database

    func testAVersion1DatabaseUpgradesWithItsChatsUnplaced() throws {
        try makeVersion1Database([chat("New", created: 10), chat("Old", created: 0)])
        let store = try open()

        XCTAssertEqual(try store.chats.database.userVersion(), 2)
        let chats = store.chats.load()
        XCTAssertEqual(chats.map(\.title), ["New", "Old"])
        XCTAssertTrue(chats.allSatisfy { $0.groupId == nil && $0.sidebarPosition == nil })
        XCTAssertTrue(store.sidebar.loadGroups().groups.isEmpty)
        XCTAssertTrue(store.sidebar.loadTerminals().isEmpty)
    }

    // MARK: - Round trips

    func testGroupsSurviveTheRoundTrip() throws {
        var groups = SidebarGroups()
        let work = try XCTUnwrap(groups.create("Work"))
        _ = try XCTUnwrap(groups.create("Inside", in: work))
        _ = try XCTUnwrap(groups.create("Home"))
        groups.toggleCollapsed(work)
        try open().sidebar.saveGroups(groups)

        XCTAssertEqual(try open().sidebar.loadGroups().ordered, groups.ordered)
    }

    func testEveryStoredTerminalFieldSurvivesTheRoundTrip() throws {
        let store = try open()
        var groups = SidebarGroups()
        let work = try XCTUnwrap(groups.create("Work"))
        store.sidebar.saveGroups(groups)

        var shell = terminal("shell", label: "shell", created: 0)
        shell.themeId = "dracula"
        var claude = terminal("Claude Code", label: "Claude Code", created: 10, kind: .host)
        claude.customName = "backend"
        claude.groupId = work
        claude.position = 4
        store.sidebar.saveTerminals([shell, claude])

        let loaded = try open().sidebar.loadTerminals()
        XCTAssertEqual(try json(loaded), try json([shell, claude]))
        XCTAssertTrue(loaded.allSatisfy { $0.phase == .suspended && $0.resumes }, "no process outlives a quit")
    }

    /// A field added to a terminal row without a column would be dropped by every save.
    func testEveryTerminalCodingKeyHasAColumn() {
        XCTAssertEqual(Set(TerminalSessionList.Session.CodingKeys.allCases.map(\.rawValue)),
                       Set(SidebarStore.terminalColumns.map(\.key)))
    }

    func testASaveWritesOnlyWhatChanged() throws {
        let store = try open()
        var groups = SidebarGroups()
        let work = try XCTUnwrap(groups.create("Work"))
        var list = TerminalSessionList(restoring: [terminal("pi", created: 0), terminal("pi 2", created: 1)])
        let first = list.sessions[0].id
        store.sidebar.saveGroups(groups)
        store.sidebar.saveTerminals(list.sessions)

        var before = store.chats.database.totalChanges
        store.sidebar.saveGroups(groups)
        list.markExited(first, exitCode: 0)
        store.sidebar.saveTerminals(list.sessions)
        XCTAssertEqual(store.chats.database.totalChanges - before, 0, "a process ending is not stored")

        before = store.chats.database.totalChanges
        groups.toggleCollapsed(work)
        store.sidebar.saveGroups(groups)
        list.rename(first, to: "backend")
        store.sidebar.saveTerminals(list.sessions)
        XCTAssertEqual(store.chats.database.totalChanges - before, 2, "one group, one terminal")

        before = store.chats.database.totalChanges
        list.close(first)
        store.sidebar.saveTerminals(list.sessions)
        XCTAssertEqual(store.chats.database.totalChanges - before, 1)
        XCTAssertEqual(try open().sidebar.loadTerminals().map(\.autoName), ["pi 2"])
    }

    /// A row naming a group fails its foreign key until the group is on disk.
    func testAGroupThatFailedToSaveIsWrittenBeforeTheRowsThatNameIt() throws {
        let store = try open()
        try store.chats.database.execute("CREATE TRIGGER refuse BEFORE INSERT ON sidebar_groups BEGIN SELECT RAISE(ABORT, 'refused'); END")
        var groups = SidebarGroups()
        let work = try XCTUnwrap(groups.create("Work"))
        store.sidebar.saveGroups(groups)
        try store.chats.database.execute("DROP TRIGGER refuse")

        var t = terminal("pi", created: 0)
        t.groupId = work
        store.sidebar.saveTerminals([t])
        var c = chat("In work", created: 0)
        c.groupId = work
        store.sidebar.retryUnsavedGroups()
        store.chats.save([c])

        let reopened = try open()
        XCTAssertEqual(reopened.sidebar.loadGroups().groups.map(\.id), [work])
        XCTAssertEqual(reopened.sidebar.loadTerminals().first?.groupId, work)
        XCTAssertEqual(reopened.chats.load().first?.groupId, work)
    }

    /// The foreign keys only ever let go: what was in a deleted group stays, at the root.
    func testDeletingAGroupRowNeverDeletesWhatWasInIt() throws {
        let store = try open()
        var groups = SidebarGroups()
        let work = try XCTUnwrap(groups.create("Work"))
        let inner = try XCTUnwrap(groups.create("Inner", in: work))
        store.sidebar.saveGroups(groups)
        var c = chat("In work", created: 0)
        c.groupId = work
        store.chats.save([c])
        var t = terminal("pi", created: 0)
        t.groupId = work
        store.sidebar.saveTerminals([t])

        store.sidebar.saveGroups(SidebarGroups(groups.groups.filter { $0.id != work }))

        let reopened = try open()
        XCTAssertEqual(reopened.chats.load().map(\.title), ["In work"])
        XCTAssertNil(reopened.chats.load().first?.groupId)
        XCTAssertNil(reopened.sidebar.loadTerminals().first?.groupId)
        XCTAssertEqual(reopened.sidebar.loadGroups().groups.map(\.id), [inner])
        XCTAssertNil(reopened.sidebar.loadGroups().groups.first?.parentId)
    }

    // MARK: - Moving off UserDefaults

    func testTheOldSidebarIsImported() throws {
        let a = chat("a", created: 30), b = chat("b", created: 20), c = chat("c", created: 10)
        try makeVersion1Database([a, b, c])
        let t1 = terminal("pi", created: 25), t2 = terminal("pi 2", created: 15)
        let work = UUID(), empty = UUID(), ghost = UUID()
        let legacy = LegacySidebar(
            order: [t2.id, b.id, a.id, UUID(), c.id, t2.id],
            groups: try oldGroups([.init(id: work, name: "Work", collapsed: true),
                                   .init(id: empty, name: "Empty", collapsed: false)],
                                  membership: [a.id: work, t1.id: work, c.id: ghost]),
            terminals: try oldTerminals([t1, t2]))

        let store = try open(legacy)

        let groups = store.sidebar.loadGroups()
        XCTAssertEqual(groups.ordered.map(\.id), [work, empty])
        XCTAssertEqual(groups.ordered.map(\.collapsed), [true, false])
        let chats = Dictionary(uniqueKeysWithValues: store.chats.load().map { ($0.id, $0) })
        XCTAssertEqual(chats[a.id].map { SidebarChatRows.Placement(group: $0.groupId, position: $0.sidebarPosition) },
                       .init(group: work, position: 0))
        XCTAssertEqual(chats[b.id]?.sidebarPosition, 1, "positions count within the group")
        XCTAssertNil(chats[c.id]?.groupId, "a membership in a group that no longer exists is dropped")
        XCTAssertEqual(chats[c.id]?.sidebarPosition, 2)
        let terminals = Dictionary(uniqueKeysWithValues: store.sidebar.loadTerminals().map { ($0.id, $0) })
        XCTAssertEqual(terminals[t1.id]?.groupId, work)
        XCTAssertNil(terminals[t1.id]?.position, "never dragged")
        XCTAssertEqual(terminals[t2.id]?.position, 0, "the first of a repeated id counts")
        XCTAssertEqual(terminals[t2.id]?.autoName, "pi 2")
    }

    /// The sidebar the old build drew from these values, without its code.
    private func oldPicture(chats: [ChatSession], terminals: [TerminalSessionList.Session],
                            order: [UUID], groups: [UUID], membership: [UUID: UUID]) -> [[UUID]] {
        let rows = (chats.map { ($0.id, $0.createdAt) } + terminals.map { ($0.id, $0.createdAt) })
            .sorted { $0.1 > $1.1 }.map(\.0)
        let known = Set(order)
        let ordered = order.isEmpty ? rows : rows.filter { !known.contains($0) } + order.filter(rows.contains)
        let grouped = groups.map { g in ordered.filter { membership[$0] == g } }
        return grouped + [ordered.filter { membership[$0].map(groups.contains) != true }]
    }

    private func newPicture(_ store: (chats: ChatStore, sidebar: SidebarStore)) -> [[UUID]] {
        let rows = SidebarChatRows.merge(chats: store.chats.load(), terminals: store.sidebar.loadTerminals())
        let parts = store.sidebar.loadGroups().partition(rows)
        return parts.groups.map { $0.rows.map(\.id) } + [parts.ungrouped.map(\.id)]
    }

    func testTheImportDrawsTheSameSidebar() throws {
        let chats = (0..<8).map { chat("c\($0)", created: TimeInterval($0 * 10)) }
        let terminals = [terminal("pi", created: 15), terminal("pi 2", created: 45), terminal("pi 3", created: 75)]
        try makeVersion1Database(chats)
        let work = UUID(), home = UUID()
        let membership = [chats[1].id: work, chats[4].id: work, terminals[1].id: work, chats[6].id: home, chats[2].id: UUID()]
        let undragged = Set([chats[7].id, terminals[0].id, chats[4].id])
        let ids = chats.map(\.id) + terminals.map(\.id)
        // Every 5th of 11 ids: a fixed scramble that reaches each one once.
        let order = (0..<ids.count).map { ids[$0 * 5 % ids.count] }.filter { !undragged.contains($0) }

        let store = try open(LegacySidebar(
            order: order,
            groups: try oldGroups([.init(id: work, name: "Work", collapsed: false),
                                   .init(id: home, name: "Home", collapsed: false)], membership: membership),
            terminals: try oldTerminals(terminals)))

        XCTAssertEqual(newPicture(store), oldPicture(chats: chats, terminals: terminals, order: order,
                                                     groups: [work, home], membership: membership))
    }

    /// A never-dragged sidebar is newest first everywhere, before and after.
    func testAnUndraggedSidebarStaysNewestFirst() throws {
        let chats = (0..<4).map { chat("c\($0)", created: TimeInterval($0 * 10)) }
        let terminals = [terminal("pi", created: 25)]
        try makeVersion1Database(chats)
        let work = UUID()
        let membership = [chats[0].id: work, chats[3].id: work]

        let store = try open(LegacySidebar(
            groups: try oldGroups([.init(id: work, name: "Work", collapsed: false)], membership: membership),
            terminals: try oldTerminals(terminals)))

        XCTAssertEqual(newPicture(store), oldPicture(chats: chats, terminals: terminals, order: [],
                                                     groups: [work], membership: membership))
    }

    func testAnOldValueThatCannotBeReadIsSkippedAndTheRestImported() throws {
        let a = chat("a", created: 0)
        try makeVersion1Database([a])
        let t = terminal("pi", created: 5)

        let store = try open(LegacySidebar(order: [t.id, a.id], groups: Data("garbage".utf8),
                                           terminals: try oldTerminals([t])))

        XCTAssertTrue(store.sidebar.loadGroups().groups.isEmpty)
        XCTAssertEqual(store.sidebar.loadTerminals().map(\.position), [0])
        XCTAssertEqual(store.chats.load().map(\.sidebarPosition), [1])
    }

    /// A repeated id must not fail the migration, which would take the chats' import down with it.
    func testARepeatedIdInTheOldValuesKeepsItsFirstOccurrence() throws {
        try makeVersion1Database([chat("a", created: 0)])
        let work = UUID()
        let t = terminal("pi", created: 0)

        let store = try open(LegacySidebar(
            groups: try oldGroups([.init(id: work, name: "First", collapsed: false),
                                   .init(id: work, name: "Second", collapsed: true)], membership: [:]),
            terminals: try oldTerminals([t, t])))

        XCTAssertEqual(store.chats.load().map(\.title), ["a"])
        XCTAssertEqual(store.sidebar.loadGroups().groups.map(\.name), ["First"])
        XCTAssertEqual(store.sidebar.loadTerminals().map(\.id), [t.id])
    }

    /// Values an older build writes again later are not imported over what the user has done since.
    func testTheOldSidebarIsImportedOnlyOnce() throws {
        try makeVersion1Database([chat("a", created: 0)])
        let work = UUID()
        let legacy = LegacySidebar(groups: try oldGroups([.init(id: work, name: "Work", collapsed: false)], membership: [:]),
                                   terminals: try oldTerminals([terminal("pi", created: 0)]))
        let first = try open(legacy)
        var groups = first.sidebar.loadGroups()
        groups.rename(work, to: "Renamed")
        first.sidebar.saveGroups(groups)

        let again = try open(legacy)
        XCTAssertEqual(again.sidebar.loadGroups().groups.map(\.name), ["Renamed"])
        XCTAssertEqual(again.sidebar.loadTerminals().count, 1)
    }

    /// Straight from a build before the database: the chats and their places arrive together.
    func testChatsFromTheOldHistoryFileArriveInTheirGroups() throws {
        let a = chat("a", created: 10), b = chat("b", created: 0)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode([a, b]).write(to: URL(fileURLWithPath: path("chat-history.json")))
        let work = UUID()

        let store = try open(LegacySidebar(order: [b.id, a.id],
                                           groups: try oldGroups([.init(id: work, name: "Work", collapsed: false)],
                                                                 membership: [a.id: work])))

        let chats = store.chats.load()
        XCTAssertEqual(chats.map(\.title), ["a", "b"])
        XCTAssertEqual(chats.map(\.groupId), [work, nil])
        XCTAssertEqual(chats.map(\.sidebarPosition), [0, 0])
    }

    // MARK: - Retiring the old keys

    private func setOldKeys() throws {
        defaults.set([UUID().uuidString], forKey: LegacySidebar.orderKey)
        defaults.set(try oldGroups([.init(id: UUID(), name: "Work", collapsed: false)], membership: [:]),
                     forKey: LegacySidebar.groupsKey)
        defaults.set(try oldTerminals([terminal("pi", created: 0)]), forKey: LegacySidebar.terminalsKey)
    }

    private var oldKeysLeft: [String] {
        LegacySidebar.keys.filter { defaults.object(forKey: $0) != nil }
    }

    func testTheOldKeysAreRemovedOnlyOnceTheyAreBackedUp() throws {
        try setOldKeys()
        let read = LegacySidebar(defaults: defaults)
        XCTAssertEqual(read.order.count, 1)
        XCTAssertNotNil(read.groups)
        XCTAssertNotNil(read.terminals)

        LegacySidebar.retire(from: defaults, backupPath: backupPath)

        XCTAssertEqual(oldKeysLeft, [])
        let backup = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: backupPath))) as? [String: Any])
        XCTAssertEqual(Set(backup.keys), Set(LegacySidebar.keys))
        XCTAssertEqual(backup[LegacySidebar.orderKey] as? [String], read.order.map(\.uuidString))
        XCTAssertNotNil(backup[LegacySidebar.groupsKey] as? [String: Any], "kept readable, not as bytes")
    }

    func testABackupThatCannotBeWrittenKeepsTheKeys() throws {
        try setOldKeys()
        LegacySidebar.retire(from: defaults, backupPath: path("missing/sidebar-defaults.migrated.json"))
        XCTAssertEqual(oldKeysLeft, LegacySidebar.keys)
    }

    /// Keys an older build wrote again get a backup of their own.
    func testAnEarlierBackupIsNeverOverwritten() throws {
        try Data("earlier".utf8).write(to: URL(fileURLWithPath: backupPath))
        try setOldKeys()

        LegacySidebar.retire(from: defaults, backupPath: backupPath)

        XCTAssertEqual(oldKeysLeft, [])
        XCTAssertEqual(try String(contentsOfFile: backupPath, encoding: .utf8), "earlier")
        let backups = try FileManager.default.contentsOfDirectory(atPath: dir).filter { $0.hasPrefix("sidebar-defaults.migrated") }
        XCTAssertEqual(backups.count, 2)
    }

    func testNoOldKeysWritesNoBackup() {
        LegacySidebar.retire(from: defaults, backupPath: backupPath)
        XCTAssertFalse(FileManager.default.fileExists(atPath: backupPath))
    }
}
