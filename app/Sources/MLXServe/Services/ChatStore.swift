import Foundation

/// The chat history in `~/.mlx-serve/mlx-serve.db`: a row per chat, a row per
/// message holding the message's JSON. A save writes only what changed, each
/// chat in its own transaction, and never touches a row it could not read.
final class ChatStore {

    struct Column {
        enum Kind { case text, bool, json }
        let key: String
        let name: String
        let kind: Kind
    }

    /// `ChatSession`'s coding keys and the column each lives in. A key with no
    /// entry here and none in `keysWithoutColumn` would be dropped by every save.
    static let sessionColumns: [Column] = [
        Column(key: "id", name: "id", kind: .text),
        Column(key: "title", name: "title", kind: .text),
        Column(key: "createdAt", name: "created_at", kind: .text),
        Column(key: "updatedAt", name: "updated_at", kind: .text),
        Column(key: "mode", name: "mode", kind: .text),
        Column(key: "agentId", name: "agent_id", kind: .text),
        Column(key: "workingDirectory", name: "working_directory", kind: .text),
        Column(key: "attachedFolderPath", name: "attached_folder_path", kind: .text),
        Column(key: "enableThinking", name: "enable_thinking", kind: .bool),
        Column(key: "reasoningEffort", name: "reasoning_effort", kind: .text),
        Column(key: "useMCP", name: "use_mcp", kind: .bool),
        Column(key: "disabledTools", name: "disabled_tools", kind: .json),
    ]

    /// Messages have their own table; a session carrying either of the others
    /// is never written at all.
    static let keysWithoutColumn: Set<String> = ["messages", "taskRunId", "isExternalBridge"]

    static var defaultPath: String { inAppFolder("mlx-serve.db") }
    static var legacyHistoryPath: String { inAppFolder("chat-history.json") }

    let database: SQLiteDatabase

    /// Each chat as it was when it last reached the disk.
    private var saved: [UUID: (columns: [SQLiteDatabase.Value], messages: [ChatMessage])] = [:]

    init(path: String = ChatStore.defaultPath, legacyHistoryPath: String? = ChatStore.legacyHistoryPath) throws {
        database = try Self.open(path)
        try migrate(legacyHistoryPath: legacyHistoryPath)
    }

    /// A file that is not a database is moved aside and a new one started: its
    /// chats stay on disk for recovery, and new ones are saved instead of lost.
    private static func open(_ path: String) throws -> SQLiteDatabase {
        do {
            return try SQLiteDatabase(path: path)
        } catch let failure as SQLiteDatabase.Failure where failure.fileIsNotADatabase {
            let stamp = asideStamp.string(from: Date())
            for file in [path, path + "-wal", path + "-shm"] where FileManager.default.fileExists(atPath: file) {
                try FileManager.default.moveItem(atPath: file, toPath: "\(file).unreadable-\(stamp)")
            }
            Self.log("\(path) is not a database (\(failure)); moved aside, starting a new one")
            return try SQLiteDatabase(path: path)
        }
    }

    private static let asideStamp: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter
    }()

    // MARK: - Load

    /// Newest chat first, each one's messages in order. A chat or a message
    /// that does not decode is logged and left out, and stays on disk as it is.
    func load() -> [ChatSession] {
        var messages: [String: [ChatMessage]] = [:]
        var sessions: [ChatSession] = []
        do {
            try database.query("SELECT session_id, id, body FROM chat_messages ORDER BY seq") { row in
                let sessionId = row.text(0) ?? ""
                do {
                    let body = Data((row.text(2) ?? "").utf8)
                    messages[sessionId, default: []].append(try Self.decoder.decode(ChatMessage.self, from: body))
                } catch {
                    Self.log("skipped message \(row.text(1) ?? "?") of chat \(sessionId): \(error)")
                }
            }
            let columns = Self.sessionColumns.map { $0.name }.joined(separator: ", ")
            try database.query("SELECT \(columns) FROM chat_sessions ORDER BY created_at DESC, rowid DESC") { row in
                do {
                    var session = try Self.session(from: row)
                    session.messages = messages[session.id.uuidString] ?? []
                    saved[session.id] = (try Self.columnValues(of: session), session.messages)
                    sessions.append(session)
                } catch {
                    Self.log("skipped chat \(row.text(0) ?? "?"): \(error)")
                }
            }
        } catch {
            Self.log("could not read the chat history: \(error)")
        }
        return sessions
    }

    // MARK: - Save

    /// Writes what changed since the last save. A chat that fails to write is
    /// logged and retried whole by the next save; the others are unaffected.
    func save(_ sessions: [ChatSession]) {
        let persisted = sessions.filter { $0.taskRunId == nil && !$0.isExternalBridge }
        let live = Set(persisted.map(\.id))
        for id in saved.keys where !live.contains(id) {
            do {
                try database.run("DELETE FROM chat_sessions WHERE id = ?", [.text(id.uuidString)])
                saved[id] = nil
            } catch {
                Self.log("could not delete chat \(id): \(error)")
            }
        }
        // Oldest first, so chats created within one second keep their order (`rowid` breaks the tie).
        for session in persisted.reversed() {
            do {
                try write(session)
            } catch {
                Self.log("could not save chat \(session.id): \(error)")
            }
        }
    }

    private func write(_ session: ChatSession) throws {
        let columns = try Self.columnValues(of: session)
        let messages = Self.firstOfEachId(session.messages)
        let previous = saved[session.id]
        if let previous, previous.columns == columns, previous.messages == messages { return }
        if messages.count < session.messages.count {
            Self.log("chat \(session.id): \(session.messages.count - messages.count) message(s) repeat an earlier id and are not saved")
        }
        try database.transaction {
            if let previous {
                if previous.columns != columns { try updateSession(columns) }
                try writeMessages(messages, of: session.id, over: previous.messages)
            } else {
                try insertSession(columns)
                try insertMessages(messages[...], of: session.id)
            }
        }
        saved[session.id] = (columns, messages)
    }

    /// A message is keyed by (chat, id), so a later message reusing an id stays
    /// unsaved rather than overwriting or reordering the first.
    private static func firstOfEachId(_ messages: [ChatMessage]) -> [ChatMessage] {
        var seen = Set<UUID>()
        return messages.filter { seen.insert($0.id).inserted }
    }

    /// Messages are only ever appended or removed, so a kept message keeps its
    /// row and its `seq` (its place), and a new one goes in after everything
    /// stored. A message that lands BETWEEN stored ones rewrites the tail from it.
    private func writeMessages(_ messages: [ChatMessage], of sessionId: UUID, over previous: [ChatMessage]) throws {
        let previousIndex = Dictionary(previous.enumerated().map { ($1.id, $0) }, uniquingKeysWith: { first, _ in first })
        let current = Set(messages.map(\.id))
        for message in previous where !current.contains(message.id) {
            try deleteMessage(message.id, of: sessionId)
        }
        var kept = 0
        var last = -1
        for message in messages {
            guard let index = previousIndex[message.id], index > last else { break }
            if message != previous[index] { try updateMessage(message, of: sessionId) }
            last = index
            kept += 1
        }
        let tail = messages[kept...]
        for message in tail where previousIndex[message.id] != nil {
            try deleteMessage(message.id, of: sessionId)
        }
        try insertMessages(tail, of: sessionId)
    }

    private func insertSession(_ columns: [SQLiteDatabase.Value]) throws {
        let names = Self.sessionColumns.map { $0.name }
        let placeholders = Array(repeating: "?", count: names.count).joined(separator: ", ")
        try database.run("INSERT INTO chat_sessions (\(names.joined(separator: ", "))) VALUES (\(placeholders))", columns)
    }

    private func updateSession(_ columns: [SQLiteDatabase.Value]) throws {
        let assignments = Self.sessionColumns.dropFirst().map { "\($0.name) = ?" }.joined(separator: ", ")
        try database.run("UPDATE chat_sessions SET \(assignments) WHERE id = ?", Array(columns.dropFirst()) + [columns[0]])
    }

    private func insertMessages(_ messages: ArraySlice<ChatMessage>, of sessionId: UUID) throws {
        for message in messages {
            try database.run("INSERT INTO chat_messages (session_id, id, role, created_at, body) VALUES (?, ?, ?, ?, ?)",
                             try [.text(sessionId.uuidString)] + Self.messageValues(message))
        }
    }

    /// An UPDATE, never an INSERT OR REPLACE: a replaced row gets a new `seq`
    /// and the message would come back at the end of its chat.
    private func updateMessage(_ message: ChatMessage, of sessionId: UUID) throws {
        let values = try Self.messageValues(message)
        try database.run("UPDATE chat_messages SET role = ?, created_at = ?, body = ? WHERE session_id = ? AND id = ?",
                         Array(values.dropFirst()) + [.text(sessionId.uuidString), values[0]])
    }

    private func deleteMessage(_ id: UUID, of sessionId: UUID) throws {
        try database.run("DELETE FROM chat_messages WHERE session_id = ? AND id = ?",
                         [.text(sessionId.uuidString), .text(id.uuidString)])
    }

    // MARK: - Rows

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }()

    private static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    /// The session's own encoding, read column by column, so every default and
    /// backfill its decoder applies keeps applying.
    private static func columnValues(of session: ChatSession) throws -> [SQLiteDatabase.Value] {
        var withoutMessages = session
        withoutMessages.messages = []
        let object = try JSONSerialization.jsonObject(with: encoder.encode(withoutMessages)) as? [String: Any] ?? [:]
        return try sessionColumns.map { column in
            switch (column.kind, object[column.key]) {
            case (_, nil), (_, is NSNull): return .null
            case (.bool, let value as Bool): return .integer(value ? 1 : 0)
            case (.json, let value?):
                let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .withoutEscapingSlashes])
                return .text(String(decoding: data, as: UTF8.self))
            case (_, let value as String): return .text(value)
            case (_, let value?): throw SQLiteDatabase.Failure(description: "\(column.key) has no column form: \(value)")
            }
        }
    }

    private static func session(from row: SQLiteDatabase.Row) throws -> ChatSession {
        var object: [String: Any] = ["messages": []]
        for (index, column) in sessionColumns.enumerated() {
            guard let text = row.text(Int32(index)) else { continue }
            switch column.kind {
            case .text: object[column.key] = text
            case .bool: object[column.key] = row.integer(Int32(index)) != 0
            case .json: object[column.key] = try JSONSerialization.jsonObject(with: Data(text.utf8))
            }
        }
        return try decoder.decode(ChatSession.self, from: JSONSerialization.data(withJSONObject: object))
    }

    /// id, role, created_at, body. Throws rather than write a body that will not load.
    private static func messageValues(_ message: ChatMessage) throws -> [SQLiteDatabase.Value] {
        let body = String(decoding: try encoder.encode(message), as: UTF8.self)
        return [.text(message.id.uuidString), .text(message.role.rawValue),
                .text(timestamp.string(from: message.timestamp)), .text(body)]
    }

    /// The format `.iso8601` writes into the JSON, so a column and the body agree.
    private static let timestamp = ISO8601DateFormatter()

    // MARK: - Schema

    private static let schema = """
        CREATE TABLE chat_sessions (
            id                   TEXT PRIMARY KEY,
            title                TEXT NOT NULL,
            created_at           TEXT NOT NULL,
            updated_at           TEXT NOT NULL,
            mode                 TEXT NOT NULL,
            agent_id             TEXT,
            working_directory    TEXT,
            attached_folder_path TEXT,
            enable_thinking      INTEGER NOT NULL DEFAULT 0,
            reasoning_effort     TEXT NOT NULL DEFAULT 'low',
            use_mcp              INTEGER NOT NULL DEFAULT 0,
            disabled_tools       TEXT NOT NULL DEFAULT '[]'
        );
        CREATE INDEX chat_sessions_by_created ON chat_sessions (created_at);
        CREATE TABLE chat_messages (
            seq        INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES chat_sessions (id) ON DELETE CASCADE,
            id         TEXT NOT NULL,
            role       TEXT NOT NULL,
            created_at TEXT NOT NULL,
            body       TEXT NOT NULL,
            UNIQUE (session_id, id)
        );
        CREATE INDEX chat_messages_by_session ON chat_messages (session_id, seq);
        """

    /// Version 1 creates the tables and imports `chat-history.json` in ONE
    /// transaction, so a crash midway leaves nothing and the next launch redoes
    /// it. The old file is then renamed to `chat-history.migrated.json`.
    private func migrate(legacyHistoryPath: String?) throws {
        guard try database.userVersion() < 1 else {
            if let legacyHistoryPath, FileManager.default.fileExists(atPath: legacyHistoryPath) {
                Self.log("\(legacyHistoryPath) found beside an already migrated database; left as is, not imported")
            }
            return
        }
        var imported = false
        try database.transaction {
            // Read again under the write lock: another process may have migrated since.
            guard try database.userVersion() < 1 else { return }
            try database.execute(Self.schema)
            if let legacyHistoryPath { imported = try importLegacy(from: legacyHistoryPath) }
            try database.setUserVersion(1)
        }
        guard imported, let legacyHistoryPath else { return }
        let backup = (legacyHistoryPath as NSString).deletingPathExtension + ".migrated.json"
        do {
            try FileManager.default.moveItem(atPath: legacyHistoryPath, toPath: backup)
        } catch {
            Self.log("imported, but could not rename \(legacyHistoryPath): \(error)")
        }
    }

    /// Chat by chat, so one that no longer decodes is logged and skipped instead
    /// of costing all the others. False when there is no file or it is not a
    /// JSON array at all; the file is then left where it is.
    private func importLegacy(from path: String) throws -> Bool {
        guard let data = FileManager.default.contents(atPath: path) else { return false }
        guard let array = (try? JSONSerialization.jsonObject(with: data)) as? [Any] else {
            Self.log("\(path) is not a chat history, left in place")
            return false
        }
        var seen = Set<UUID>()
        // Oldest first: the file lists newest first and `rowid` breaks createdAt ties.
        for element in array.reversed() {
            let session: ChatSession
            do {
                session = try Self.decoder.decode(ChatSession.self, from: JSONSerialization.data(withJSONObject: element))
            } catch {
                Self.log("skipped a chat that no longer decodes: \(error)")
                continue
            }
            guard session.taskRunId == nil, !session.isExternalBridge, seen.insert(session.id).inserted else { continue }
            try insertSession(Self.columnValues(of: session))
            try insertMessages(Self.firstOfEachId(session.messages)[...], of: session.id)
        }
        return true
    }

    /// Through a format string: a `%` in an error's text must not be read as one.
    private static func log(_ message: String) {
        NSLog("[chats] %@", message)
    }

    private static func inAppFolder(_ name: String) -> String {
        let dir = NSString(string: "~/.mlx-serve").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return (dir as NSString).appendingPathComponent(name)
    }
}
