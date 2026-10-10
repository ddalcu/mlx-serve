import Foundation
import SQLite3

/// One connection to a SQLite file through the system library: statements,
/// transactions and the schema version. What the tables mean belongs to the
/// stores that use it (`ChatStore`).
final class SQLiteDatabase {

    struct Failure: Error, CustomStringConvertible {
        let description: String
        var code: Int32 = SQLITE_ERROR

        /// The file exists but holds something else (or a damaged header).
        var fileIsNotADatabase: Bool { code & 0xff == SQLITE_NOTADB }
    }

    enum Value: Equatable {
        case text(String)
        case integer(Int)
        case null
    }

    /// One result row; columns are read by index, in the order the query names them.
    struct Row {
        fileprivate let statement: OpaquePointer?

        /// Read by byte length, so a NUL inside the text does not end it.
        func text(_ column: Int32) -> String? {
            guard let bytes = sqlite3_column_text(statement, column) else { return nil }
            let count = Int(sqlite3_column_bytes(statement, column))
            return String(decoding: UnsafeBufferPointer(start: bytes, count: count), as: UTF8.self)
        }

        func integer(_ column: Int32) -> Int {
            Int(sqlite3_column_int64(statement, column))
        }
    }

    private var handle: OpaquePointer?

    init(path: String) throws {
        let status = sqlite3_open_v2(path, &handle, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nil)
        guard status == SQLITE_OK else {
            let message = handle.map { String(cString: sqlite3_errmsg($0)) } ?? "status \(status)"
            sqlite3_close(handle)
            handle = nil   // deinit runs after a throwing init and would close it again
            throw Failure(description: "open \(path): \(message)", code: status)
        }
        // The timeout first, so a lock held by another process waits instead of
        // failing the rest. WAL: a reader never blocks the writer. Foreign keys
        // are off unless each connection asks.
        try execute("PRAGMA busy_timeout = 5000; PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON")
    }

    deinit {
        sqlite3_close(handle)
    }

    /// Statements that take no values and return no rows (schema, pragmas).
    func execute(_ sql: String) throws {
        var error: UnsafeMutablePointer<CChar>?
        let status = sqlite3_exec(handle, sql, nil, nil, &error)
        guard status == SQLITE_OK else {
            let message = error.map { String(cString: $0) } ?? lastError
            sqlite3_free(error)
            throw Failure(description: message, code: status)
        }
    }

    func run(_ sql: String, _ values: [Value] = []) throws {
        try withStatement(sql, values) { statement in
            guard sqlite3_step(statement) == SQLITE_DONE else { throw lastFailure }
        }
    }

    func query(_ sql: String, _ values: [Value] = [], row: (Row) throws -> Void) throws {
        try withStatement(sql, values) { statement in
            while true {
                let status = sqlite3_step(statement)
                if status == SQLITE_DONE { return }
                guard status == SQLITE_ROW else { throw lastFailure }
                try row(Row(statement: statement))
            }
        }
    }

    /// All or nothing: a throw anywhere in `body` rolls every statement back.
    func transaction(_ body: () throws -> Void) throws {
        try execute("BEGIN IMMEDIATE")
        do {
            try body()
            try execute("COMMIT")
        } catch {
            try? execute("ROLLBACK")
            throw error
        }
    }

    func userVersion() throws -> Int {
        var version = 0
        try query("PRAGMA user_version") { version = $0.integer(0) }
        return version
    }

    func setUserVersion(_ version: Int) throws {
        try execute("PRAGMA user_version = \(version)")
    }

    /// Rows inserted, updated or deleted since the connection opened.
    var totalChanges: Int {
        Int(sqlite3_total_changes(handle))
    }

    private var lastError: String {
        String(cString: sqlite3_errmsg(handle))
    }

    private var lastFailure: Failure {
        Failure(description: lastError, code: sqlite3_errcode(handle))
    }

    private func withStatement(_ sql: String, _ values: [Value],
                               _ body: (OpaquePointer?) throws -> Void) throws {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(handle, sql, -1, &statement, nil) == SQLITE_OK else {
            throw Failure(description: "\(lastError) in: \(sql)", code: sqlite3_errcode(handle))
        }
        defer { sqlite3_finalize(statement) }
        for (offset, value) in values.enumerated() {
            let index = Int32(offset + 1)
            let status: Int32
            switch value {
            case .text(let text):
                // Bound by byte length, so a NUL inside the text does not end it.
                status = sqlite3_bind_text(statement, index, text, Int32(text.utf8.count), transient)
            case .integer(let number):
                status = sqlite3_bind_int64(statement, index, sqlite3_int64(number))
            case .null:
                status = sqlite3_bind_null(statement, index)
            }
            guard status == SQLITE_OK else { throw lastFailure }
        }
        try body(statement)
    }
}

/// SQLite copies a value bound with this before the call returns.
private let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
