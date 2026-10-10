import XCTest
@testable import MLXCore

/// Chats in `mlx-serve.db`: nothing lost, only changes written, order kept, unreadable rows never overwritten.
final class ChatStoreTests: XCTestCase {

    private var dir: String!

    override func setUpWithError() throws {
        dir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mlx-core-chats-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(atPath: dir)
    }

    private var dbPath: String { (dir as NSString).appendingPathComponent("mlx-serve.db") }
    private var legacyPath: String { (dir as NSString).appendingPathComponent("chat-history.json") }
    private var backupPath: String { (dir as NSString).appendingPathComponent("chat-history.migrated.json") }

    private func open() throws -> ChatStore {
        try ChatStore(path: dbPath, legacyHistoryPath: legacyPath)
    }

    /// Reopens the file, so a check reads the disk and not the store's memory.
    private func reloaded() throws -> [ChatSession] {
        try open().load()
    }

    /// Whole seconds, the stored precision.
    private func at(_ seconds: TimeInterval) -> Date {
        Date(timeIntervalSince1970: 1_790_000_000 + seconds)
    }

    /// The history's own JSON shape: what "the same chats" means across a round trip.
    private func json(_ sessions: [ChatSession]) throws -> String {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        return String(decoding: try encoder.encode(sessions), as: UTF8.self)
    }

    private func ids(_ session: ChatSession?) -> [UUID] {
        session?.messages.map(\.id) ?? []
    }

    private func chat(_ title: String, created: TimeInterval = 0, messages: Int = 0) -> ChatSession {
        var s = ChatSession(title: title)
        s.createdAt = at(created)
        s.updatedAt = at(created)
        s.messages = (0..<messages).map { ChatMessage(role: $0 % 2 == 0 ? .user : .assistant, content: "\(title) \($0)") }
        return s
    }

    /// Every field of a chat and of its messages set away from its default.
    private func everything() -> ChatSession {
        var s = ChatSession(title: "Every\u{0}thing")
        s.createdAt = at(0)
        s.updatedAt = at(60)
        s.mode = .agent
        s.workingDirectory = "/tmp/work"
        s.attachedFolderPath = "/tmp/docs"
        s.enableThinking = true
        s.reasoningEffort = .high
        s.useMCP = true
        s.agentId = UUID()
        s.disabledTools = ["shell", "browse"]

        var user = ChatMessage(role: .user, content: "A \"quoted\" line, a NUL \u{0} and an emoji 🎉")
        user.images = [ChatImage(data: Data(), path: "/tmp/a.png")]
        user.audio = [ChatAudio(name: "clip.wav", pcm: Data(), path: "/tmp/clip.wav")]
        user.videos = [ChatVideo(name: "clip.mov", frames: [], path: "/tmp/frames")]

        var reply = ChatMessage(role: .assistant, content: "Answer", reasoningContent: "Thinking")
        reply.promptTokens = 10
        reply.completionTokens = 20
        reply.tokensPerSecond = 33.5
        reply.thinkingSeconds = 2.25
        reply.toolCalls = [MLXCore.SerializedToolCall(id: "c1", name: "shell", arguments: "{\"command\":\"ls\"}")]
        reply.media = [ChatMediaRef(kind: .image, path: "/tmp/generated.png", prompt: "a cat")]
        reply.processHandles = ["bg1"]
        reply.truncationNotice = TruncationNotice.Notice(cause: .repetitionLoop, maxTokens: 512)
        reply.revisions = [MessageRevision(content: "First"), MessageRevision(content: "Answer", reasoningContent: "Thinking")]
        reply.activeRevision = 1
        reply.isAgentSummary = true
        reply.agentPlan = AgentPlan(steps: [])
        reply.toolResults = [StepResult(stepId: UUID(), status: .success, output: "ok", durationMs: 5)]

        var tool = ChatMessage(role: .system, content: "file.txt")
        tool.toolCallId = "c1"
        tool.toolName = "shell"

        var failed = ChatMessage(role: .assistant, content: "")
        failed.failedRetry = true
        failed.errorNotice = ChatErrorNotice(kind: .contextOverflow, message: "too long",
                                             neededTokens: 9000, contextLength: 8192)

        s.messages = [user, reply, tool, failed]
        return s
    }

    private func messageRows(_ store: ChatStore, session: UUID) throws -> Int {
        var count = 0
        try store.database.query("SELECT count(*) FROM chat_messages WHERE session_id = ?",
                                 [.text(session.uuidString)]) { count = $0.integer(0) }
        return count
    }

    // MARK: - Round trip

    func testEveryFieldOfAChatAndItsMessagesSurvivesTheRoundTrip() throws {
        let original = everything()
        try open().save([original])
        XCTAssertEqual(try json(reloaded()), try json([original]))
    }

    /// A field added to `ChatSession` without a column would be dropped by every save.
    func testEveryChatCodingKeyHasAColumn() {
        XCTAssertEqual(Set(ChatSession.CodingKeys.allCases.map(\.rawValue)),
                       Set(ChatStore.sessionColumns.map { $0.key }).union(ChatStore.keysWithoutColumn))
    }

    /// A fork keeps its source's message ids, so they are unique only per chat.
    func testAForkSharingMessageIdsKeepsBothChats() throws {
        let source = chat("Source", created: 0, messages: 3)
        var fork = ChatFork.session(from: source, messages: source.messages)
        fork.createdAt = at(10)
        let store = try open()
        store.save([fork, source])

        fork.messages[0].content = "edited in the fork"
        store.save([fork, source])

        let loaded = try reloaded()
        XCTAssertEqual(loaded.count, 2)
        XCTAssertEqual(ids(loaded.first { $0.id == source.id }), ids(source))
        XCTAssertEqual(loaded.first { $0.id == source.id }?.messages[0].content, "Source 0")
        XCTAssertEqual(loaded.first { $0.id == fork.id }?.messages[0].content, "edited in the fork")
    }

    func testChatsLoadNewestFirst() throws {
        try open().save([chat("New", created: 20), chat("Mid", created: 10), chat("Old", created: 0)])
        XCTAssertEqual(try reloaded().map(\.title), ["New", "Mid", "Old"])
    }

    /// Two chats made within one second keep the order the app held them in.
    func testChatsCreatedInTheSameSecondKeepTheirOrder() throws {
        try open().save([chat("Second", created: 5), chat("First", created: 5)])
        XCTAssertEqual(try reloaded().map(\.title), ["Second", "First"])
    }

    /// Their conversation lives elsewhere (a task's transcript, the messaging platform).
    func testTaskRunAndBridgeSessionsAreNeverWritten() throws {
        var run = chat("Task run", messages: 1)
        run.taskRunId = UUID()
        var bridge = chat("Telegram", messages: 1)
        bridge.isExternalBridge = true
        try open().save([run, bridge, chat("Kept")])
        XCTAssertEqual(try reloaded().map(\.title), ["Kept"])
    }

    // MARK: - What a save writes

    func testASaveWritesOnlyWhatChanged() throws {
        var a = chat("A", messages: 4)
        let b = chat("B", messages: 4)
        let store = try open()
        store.save([a, b])

        var before = store.database.totalChanges
        store.save([a, b])
        XCTAssertEqual(store.database.totalChanges - before, 0, "nothing changed, nothing written")

        before = store.database.totalChanges
        a.messages.append(ChatMessage(role: .user, content: "one more"))
        store.save([a, b])
        XCTAssertEqual(store.database.totalChanges - before, 1, "one new message, one row")

        before = store.database.totalChanges
        a.messages[a.messages.count - 1].content = "edited"
        store.save([a, b])
        XCTAssertEqual(store.database.totalChanges - before, 1, "one edited message, one row")
    }

    /// A row written anew sorts last, so an edit that replaced it would move the message to the end.
    func testAnEditedMessageKeepsItsPlace() throws {
        var a = chat("A", messages: 4)
        let order = ids(a)
        let store = try open()
        store.save([a])
        a.messages[0].content = "edited"
        store.save([a])

        let loaded = try reloaded().first
        XCTAssertEqual(ids(loaded), order)
        XCTAssertEqual(loaded?.messages[0].content, "edited")
    }

    func testDeletingFromTheMiddleKeepsTheRestInOrder() throws {
        var a = chat("A", messages: 6)
        let store = try open()
        store.save([a])
        a.messages.removeSubrange(1...2)
        store.save([a])

        XCTAssertEqual(ids(try reloaded().first), ids(a))
        XCTAssertEqual(try messageRows(store, session: a.id), 4)
    }

    /// Regenerate and Edit & Resend: the tail is cut, then new messages follow.
    func testNewMessagesAfterATruncationFollowTheKeptOnes() throws {
        var a = chat("A", messages: 4)
        let store = try open()
        store.save([a])
        a.messages.removeSubrange(2...)
        a.messages.append(ChatMessage(role: .user, content: "again"))
        a.messages.append(ChatMessage(role: .assistant, content: "new answer"))
        store.save([a])

        XCTAssertEqual(ids(try reloaded().first), ids(a))
    }

    /// The app only appends, but a message put between stored ones still comes back in place.
    func testAMessageInsertedBetweenStoredOnesComesBackInPlace() throws {
        var a = chat("A", messages: 4)
        let store = try open()
        store.save([a])
        a.messages.insert(ChatMessage(role: .user, content: "inserted"), at: 1)
        store.save([a])

        XCTAssertEqual(ids(try reloaded().first), ids(a))
        XCTAssertEqual(try messageRows(store, session: a.id), 5)
    }

    func testADeletedChatTakesItsMessagesWithIt() throws {
        let a = chat("A", messages: 3)
        let b = chat("B", messages: 3)
        let store = try open()
        store.save([a, b])
        store.save([b])

        XCTAssertEqual(try reloaded().map(\.title), ["B"])
        XCTAssertEqual(try messageRows(store, session: a.id), 0)
    }

    func testChatSettingsChangesAreSaved() throws {
        var a = chat("A", messages: 2)
        let store = try open()
        store.save([a])
        a.title = "Renamed"
        a.enableThinking = true
        a.disabledTools = ["shell"]
        store.save([a])

        let loaded = try reloaded().first
        XCTAssertEqual(loaded?.title, "Renamed")
        XCTAssertEqual(loaded?.enableThinking, true)
        XCTAssertEqual(loaded?.disabledTools, ["shell"])
    }

    /// A failed write leaves the last-saved state alone, so the next save retries it.
    func testAFailedSaveIsRetriedByTheNextOne() throws {
        var a = chat("A", messages: 1)
        let store = try open()
        store.save([a])
        try store.database.execute("CREATE TRIGGER refuse BEFORE INSERT ON chat_messages BEGIN SELECT RAISE(ABORT, 'refused'); END")
        a.messages.append(ChatMessage(role: .assistant, content: "reply"))
        store.save([a])
        XCTAssertEqual(try messageRows(store, session: a.id), 1)

        try store.database.execute("DROP TRIGGER refuse")
        store.save([a])
        XCTAssertEqual(try messageRows(store, session: a.id), 2)
    }

    /// A message JSON cannot encode is not written as an empty placeholder that would be lost on reload.
    func testAMessageThatCannotBeEncodedIsNotWrittenAsAPlaceholder() throws {
        var a = chat("A", messages: 1)
        let store = try open()
        store.save([a])
        var reply = ChatMessage(role: .assistant, content: "reply")
        reply.tokensPerSecond = .nan
        a.messages.append(reply)
        store.save([a])
        XCTAssertEqual(try messageRows(store, session: a.id), 1)

        a.messages[1].tokensPerSecond = 40
        store.save([a])
        XCTAssertEqual(try reloaded().first?.messages.last?.content, "reply")
    }

    /// A second message with an id already in the chat never costs the first one its row.
    func testADuplicateMessageIdNeverDeletesTheOriginal() throws {
        var a = chat("A", messages: 2)
        let store = try open()
        store.save([a])
        var copy = a.messages[0]
        copy.content = "same id, later"
        a.messages.append(copy)
        store.save([a])

        let loaded = try reloaded().first
        XCTAssertEqual(loaded?.messages.map(\.content), ["A 0", "A 1"])
    }

    // MARK: - Opening the file

    func testAPathThatCannotBeOpenedThrows() {
        let path = (dir as NSString).appendingPathComponent("missing/mlx-serve.db")
        XCTAssertThrowsError(try SQLiteDatabase(path: path))
    }

    /// A file that is not a database is set aside, never overwritten, and saving starts afresh.
    func testAnUnreadableDatabaseIsSetAsideAndSavingContinues() throws {
        let garbage = Data(repeating: 0x42, count: 8192)
        try garbage.write(to: URL(fileURLWithPath: dbPath))

        let store = try open()
        XCTAssertEqual(store.load().count, 0)
        store.save([chat("After", messages: 1)])
        XCTAssertEqual(try reloaded().map(\.title), ["After"])

        let aside = try FileManager.default.contentsOfDirectory(atPath: dir)
            .filter { $0.hasPrefix("mlx-serve.db.unreadable-") }
        XCTAssertEqual(aside.count, 1)
        XCTAssertEqual(FileManager.default.contents(atPath: (dir as NSString).appendingPathComponent(aside[0])), garbage)
    }

    // MARK: - Rows that cannot be read

    func testAnUnreadableMessageIsSkippedAndKeptOnDisk() throws {
        var a = chat("A", messages: 3)
        let store = try open()
        store.save([a])
        try store.database.run(
            "INSERT INTO chat_messages (session_id, id, role, created_at, body) VALUES (?, ?, 'user', '2026-01-01T00:00:00Z', 'not json')",
            [.text(a.id.uuidString), .text(UUID().uuidString)])

        let reopened = try open()
        a = try XCTUnwrap(reopened.load().first)
        XCTAssertEqual(a.messages.count, 3)

        a.messages.remove(at: 0)
        a.messages.append(ChatMessage(role: .user, content: "more"))
        reopened.save([a])
        XCTAssertEqual(try messageRows(reopened, session: a.id), 4, "the unreadable row is still there")
    }

    func testAnUnreadableChatIsSkippedAndKeptOnDisk() throws {
        let a = chat("A", created: 10, messages: 2)
        var b = chat("B", created: 0, messages: 2)
        let store = try open()
        store.save([a, b])
        try store.database.run("UPDATE chat_sessions SET mode = 'from-the-future' WHERE id = ?", [.text(a.id.uuidString)])

        let reopened = try open()
        XCTAssertEqual(reopened.load().map(\.title), ["B"])
        b.title = "B renamed"
        reopened.save([b])

        var rows = 0
        try reopened.database.query("SELECT count(*) FROM chat_sessions") { rows = $0.integer(0) }
        XCTAssertEqual(rows, 2)
        XCTAssertEqual(try messageRows(reopened, session: a.id), 2)
    }

    // MARK: - Moving off chat-history.json

    /// The old file, written exactly as `saveChatHistory` wrote it: newest first.
    private func writeLegacy(_ sessions: [ChatSession]) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = .prettyPrinted
        try encoder.encode(sessions).write(to: URL(fileURLWithPath: legacyPath))
    }

    func testTheOldHistoryIsImportedAndKeptAsABackup() throws {
        let legacy = [everything(), chat("Older", created: -100, messages: 5), chat("Oldest", created: -200, messages: 1)]
        try writeLegacy(legacy)

        XCTAssertEqual(try json(open().load()), try json(legacy))
        XCTAssertFalse(FileManager.default.fileExists(atPath: legacyPath))
        XCTAssertTrue(FileManager.default.fileExists(atPath: backupPath))
    }

    /// A file an older build writes again later is neither imported nor touched.
    func testTheOldHistoryIsImportedOnlyOnce() throws {
        try writeLegacy([chat("Imported", messages: 2)])
        _ = try open()
        try writeLegacy([chat("Written by an older build", messages: 2)])

        XCTAssertEqual(try reloaded().map(\.title), ["Imported"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: legacyPath), "left alone")
    }

    /// The old file lists newest first; two chats from one second keep that order.
    func testImportedChatsFromTheSameSecondKeepTheirOrder() throws {
        try writeLegacy([chat("Second", created: 5), chat("First", created: 5)])
        XCTAssertEqual(try open().load().map(\.title), ["Second", "First"])
    }

    func testAChatThatNoLongerDecodesIsSkippedOnImport() throws {
        try writeLegacy([chat("Good", created: 10, messages: 2), chat("Broken", created: 0, messages: 2)])
        var array = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: legacyPath))) as? [[String: Any]])
        array[1]["title"] = nil
        try JSONSerialization.data(withJSONObject: array).write(to: URL(fileURLWithPath: legacyPath))

        XCTAssertEqual(try open().load().map(\.title), ["Good"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: backupPath), "the backup still holds it")
    }

    func testAnUnreadableHistoryFileIsLeftInPlace() throws {
        try Data("not json".utf8).write(to: URL(fileURLWithPath: legacyPath))
        XCTAssertEqual(try open().load().count, 0)
        XCTAssertTrue(FileManager.default.fileExists(atPath: legacyPath))
        XCTAssertFalse(FileManager.default.fileExists(atPath: backupPath))
    }

    func testNoHistoryFileStartsEmpty() throws {
        let store = try open()
        XCTAssertEqual(store.load().count, 0)
        store.save([chat("First", messages: 1)])
        XCTAssertEqual(try reloaded().map(\.title), ["First"])
    }

    /// Opt-in, on a COPY of a real history: `MLX_SERVE_LIVE_CHAT_HISTORY=<path to chat-history.json>`.
    func testARealHistoryImportsLosslessly() throws {
        guard let source = ProcessInfo.processInfo.environment["MLX_SERVE_LIVE_CHAT_HISTORY"] else {
            throw XCTSkip("set MLX_SERVE_LIVE_CHAT_HISTORY to a chat-history.json to check")
        }
        try FileManager.default.copyItem(atPath: source, toPath: legacyPath)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let legacy = try decoder.decode([ChatSession].self, from: Data(contentsOf: URL(fileURLWithPath: legacyPath)))

        XCTAssertEqual(try json(open().load()), try json(legacy))
        XCTAssertEqual(try json(reloaded()), try json(legacy))
    }
}
