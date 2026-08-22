import XCTest
@testable import MLXCore

/// The chat window's missing basics: regenerate, edit & resend, branch (fork),
/// export, rename, search, per-tab drafts. Pure decision cores are unit-tested
/// here; every control is pinned by a source scan so a redesign cannot quietly
/// stop drawing one (the quiet-affordance-loss rule).
final class ChatWindowBasicsTests: XCTestCase {

    // MARK: - Fixtures

    private func msg(_ role: ChatMessage.Role, _ content: String,
                     failedRetry: Bool = false) -> ChatMessage {
        var m = ChatMessage(role: role, content: content)
        m.failedRetry = failedRetry
        return m
    }

    private func source(_ relativePath: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    // MARK: - Regenerate planning

    func testRegeneratePlanFindsTheLastUserTurn() {
        let messages = [
            msg(.user, "hello"),
            msg(.assistant, "hi there"),
        ]
        let plan = ChatRewind.regeneratePlan(in: messages)
        XCTAssertEqual(plan?.userIdx, 0)
        XCTAssertEqual(plan?.removeFrom, 1, "the reply at index 1 is what gets dropped")
    }

    func testRegeneratePlanRemovesErrorNoticeRowsToo() {
        var failure = ChatMessage(role: .assistant, content: "")
        failure.failedRetry = true
        failure.errorNotice = ChatErrorNotice(kind: .generic, message: "boom")
        let messages = [
            msg(.user, "do a thing"),
            msg(.assistant, "partial"),
            failure,
        ]
        let plan = ChatRewind.regeneratePlan(in: messages)
        XCTAssertEqual(plan?.userIdx, 0)
        XCTAssertEqual(plan?.removeFrom, 1, "everything after the prompt goes, notices included")
    }

    func testRegeneratePlanSkipsSyntheticSystemNudges() {
        // A truncated tool-call round leaves a [System: …] USER row behind;
        // regenerating must rewind to the human's prompt, never the nudge.
        let nudge = ChatTurnEngine.truncatedToolCallNudge
        XCTAssertTrue(nudge.hasPrefix("[System:"), "the synthetic-nudge marker moved")
        let messages = [
            msg(.user, "real prompt"),
            msg(.assistant, "working"),
            msg(.user, nudge),
        ]
        let plan = ChatRewind.regeneratePlan(in: messages)
        XCTAssertEqual(plan?.userIdx, 0)
        XCTAssertEqual(plan?.removeFrom, 1, "the nudge and everything after it is rewound")
    }

    func testRegeneratePlanWithNoReplyStillReruns() {
        let messages = [msg(.user, "only a prompt")]
        let plan = ChatRewind.regeneratePlan(in: messages)
        XCTAssertEqual(plan?.userIdx, 0)
        XCTAssertEqual(plan?.removeFrom, 1, "nothing to remove; the turn just runs again")
    }

    func testRegeneratePlanNeedsARealUserPrompt() {
        XCTAssertNil(ChatRewind.regeneratePlan(in: []))
        XCTAssertNil(ChatRewind.regeneratePlan(in: [msg(.assistant, "no prompt here")]))
    }

    // MARK: - Edit & resubmit

    func testEditingAUserMessageReplacesItsTextAndTruncatesTheTail() {
        let messages = [
            msg(.user, "first"),
            msg(.assistant, "answer one"),
            msg(.user, "second"),
            msg(.assistant, "answer two"),
        ]
        let edited = AppState.editedUserMessages(messages, messageId: messages[2].id,
                                                 newText: "second, corrected")
        XCTAssertNotNil(edited)
        XCTAssertEqual(edited?.count, 3, "the reply being re-asked is gone")
        XCTAssertEqual(edited?[2].content, "second, corrected")
        XCTAssertEqual(edited?[2].role, .user)
        XCTAssertEqual(edited?[1].content, "answer one", "earlier turns are untouched")
    }

    func testEditingRefusesNonUserAndUnknownTargets() {
        let messages = [msg(.user, "q"), msg(.assistant, "a")]
        XCTAssertNil(AppState.editedUserMessages(messages, messageId: messages[1].id,
                                                 newText: "x"), "only your own turns are editable")
        XCTAssertNil(AppState.editedUserMessages(messages, messageId: UUID(),
                                                 newText: "x"))
        XCTAssertNil(AppState.editedUserMessages(messages, messageId: messages[0].id,
                                                 newText: "   "), "an empty edit is a delete, not an edit — refuse it")
    }

    // MARK: - Fork / branch

    func testForkCopiesHistoryUpToAndIncludingTheBranchPoint() throws {
        var source = ChatSession(title: "Deep dive")
        source.messages = [
            msg(.user, "one"),
            msg(.assistant, "two"),
            msg(.user, "three"),
            msg(.assistant, "four"),
        ]
        let forkId = UUID()
        let now = Date(timeIntervalSince1970: 100)
        let fork = try XCTUnwrap(AppState.forkedSession(
            from: source, upTo: source.messages[1].id, newId: forkId, now: now))
        XCTAssertEqual(fork.id, forkId)
        XCTAssertEqual(fork.title, "Deep dive", "the branch keeps the conversation's name")
        XCTAssertEqual(fork.messages.map(\.content), ["one", "two"])
        XCTAssertEqual(fork.createdAt, now)
        XCTAssertEqual(fork.updatedAt, now)
    }

    func testForkMintsFreshIdsSoNoRowIsSharedBetweenThreads() throws {
        var source = ChatSession(title: "s")
        source.messages = [msg(.user, "a"), msg(.assistant, "b")]
        let fork = try XCTUnwrap(AppState.forkedSession(
            from: source, upTo: source.messages[1].id, newId: UUID(), now: Date()))
        XCTAssertEqual(Set(fork.messages.map(\.id)).intersection(source.messages.map(\.id)), [],
                       "message ids must be remapped")
        XCTAssertNotEqual(fork.id, source.id)
    }

    func testForkBecomesAnOrdinaryConversation() throws {
        var source = ChatSession(title: "s")
        source.taskRunId = UUID()
        source.isExternalBridge = true
        source.agentId = UUID()
        source.workingDirectory = "/tmp/wd"
        source.useMCP = true
        source.enableThinking = true
        source.disabledTools = ["shell"]
        source.messages = [msg(.user, "a")]
        let fork = try XCTUnwrap(AppState.forkedSession(
            from: source, upTo: source.messages[0].id, newId: UUID(), now: Date()))
        XCTAssertNil(fork.taskRunId, "a fork is never a task vehicle")
        XCTAssertFalse(fork.isExternalBridge, "a fork is never a bridge mirror")
        XCTAssertEqual(fork.agentId, source.agentId, "the persona comes along")
        XCTAssertEqual(fork.workingDirectory, "/tmp/wd")
        XCTAssertTrue(fork.useMCP)
        XCTAssertTrue(fork.enableThinking)
        XCTAssertEqual(fork.disabledTools, ["shell"])
    }

    func testForkOfAnUnknownOrTrailingEmptyPrefixIsRefused() {
        var source = ChatSession(title: "s")
        source.messages = [msg(.user, "a")]
        XCTAssertNil(AppState.forkedSession(from: source, upTo: UUID(),
                                            newId: UUID(), now: Date()))
        // Forking AT the last message would duplicate the whole thread with
        // nothing new above it — still allowed (that IS "duplicate chat"), so
        // assert the inclusive prefix instead of a refusal here.
        let wholeThread = AppState.forkedSession(from: source, upTo: source.messages[0].id,
                                                 newId: UUID(), now: Date())
        XCTAssertEqual(wholeThread?.messages.count, 1)
    }

    // MARK: - Sidebar search

    func testSearchMatchesTitlesCaseInsensitively() {
        var s = ChatSession(title: "Rust Borrow Checker")
        s.messages = [msg(.user, "explain lifetimes")]
        XCTAssertTrue(SidebarSearch.matches(s, query: "borrow"))
        XCTAssertTrue(SidebarSearch.matches(s, query: "rust"))
        XCTAssertFalse(SidebarSearch.matches(s, query: "kubernetes"))
    }

    func testSearchLooksInsideTranscripts() {
        var s = ChatSession(title: "New Chat")
        s.messages = [msg(.user, "explain lifetimes")]
        XCTAssertTrue(SidebarSearch.matches(s, query: "lifetimes"))
        XCTAssertFalse(SidebarSearch.matches(s, query: "lifetimess"))
    }

    func testSearchIgnoresHiddenToolRowsAndDiacriticsFold() {
        var s = ChatSession(title: "café plans")
        s.messages = [
            ChatMessage(role: .assistant, content: "calling shell"),
            { var r = ChatMessage(role: .assistant, content: "secret tool output")
              r.toolCallId = "t1"; return r }(),
        ]
        XCTAssertTrue(SidebarSearch.matches(s, query: "cafe"), "diacritics fold")
        XCTAssertFalse(SidebarSearch.matches(s, query: "secret tool output"),
                       "hidden tool results are not searchable noise")
    }

    func testEmptyQueryKeepsEverythingInOrder() {
        let sessions = [ChatSession(title: "b"), ChatSession(title: "a")]
        XCTAssertEqual(SidebarSearch.filter(sessions: sessions, query: "").map(\.title),
                       ["b", "a"])
        XCTAssertEqual(SidebarSearch.filter(sessions: sessions, query: "  ").count, 2,
                       "whitespace-only is empty")
    }

    // MARK: - Composer drafts

    func testDraftsStashRestoreAndClearPerSession() {
        var drafts = ComposerDrafts()
        let a = UUID(), b = UUID()
        drafts.stash("half-typed thought", for: a)
        XCTAssertEqual(drafts.restore(for: a), "half-typed thought")
        XCTAssertEqual(drafts.restore(for: b), "", "another tab starts empty")
        drafts.stash("second idea", for: a)
        XCTAssertEqual(drafts.restore(for: a), "second idea", "stashing overwrites")
        drafts.clear(for: a)
        XCTAssertEqual(drafts.restore(for: a), "")
    }

    func testDraftsNeverStoreWhitespaceOnlyTextAsContent() {
        var drafts = ComposerDrafts()
        let a = UUID()
        drafts.stash("   ", for: a)
        XCTAssertEqual(drafts.restore(for: a), "",
                       "a cleared field must not resurrect as whitespace")
    }

    // MARK: - Export

    func testExportMarkdownLabelsRolesAndSkipsMachineRows() {
        var err = ChatMessage(role: .assistant, content: "")
        err.failedRetry = true
        err.errorNotice = ChatErrorNotice(kind: .generic, message: "boom")
        var toolResult = ChatMessage(role: .assistant, content: "hidden")
        toolResult.toolCallId = "t1"
        var withImage = ChatMessage(role: .user, content: "what is this?")
        withImage.images = [ChatImage(data: Data([0xFF]))]

        let md = ChatExport.markdown(
            title: "Trip plan",
            messages: [
                withImage,
                msg(.assistant, "**Day 1:** arrive\n\n```swift\nlet x = 1\n```"),
                err,
                toolResult,
                msg(.user, "thanks"),
            ],
            dateText: "2026-08-22")

        XCTAssertTrue(md.contains("# Trip plan"))
        XCTAssertTrue(md.contains("2026-08-22"))
        XCTAssertTrue(md.contains("**You**"), md)
        XCTAssertTrue(md.contains("**Assistant**"))
        XCTAssertTrue(md.contains("```swift"), "code fences survive verbatim")
        XCTAssertTrue(md.contains("_Attached image_"), "media becomes a note line")
        XCTAssertFalse(md.contains("hidden"), "tool-result rows stay out")
        XCTAssertFalse(md.contains("boom"), "failure cards stay out")
        XCTAssertFalse(md.contains("reasoningContent") , "nothing meta leaks")
    }

    func testExportSanitizesTitleSlashesForAFilename() {
        XCTAssertEqual(ChatExport.suggestedFilename(title: "a/b: c?"), "a-b-c.md")
    }

    // MARK: - Rename

    func testRenamedTitleIsTrimmedAndEmptyRenameIsRefused() {
        var session = ChatSession(title: "Old")
        let renamed = AppState.renamedTitle("  New name  ") ?? ""
        session.title = renamed
        XCTAssertEqual(session.title, "New name")
        XCTAssertNil(AppState.renamedTitle("   "), "empty rename must not blank a title")
    }

    // MARK: - Source scans (affordance pins)

    func testMessageBubbleOffersTheNewActions() throws {
        let s = try source("Sources/MLXServe/Views/ChatView.swift")
        for (needle, what) in [
            ("var onRegenerate: (() -> Void)?", "regenerate callback"),
            ("var onEditResend: ((String) -> Void)?", "edit & resend callback"),
            ("var onForkFromHere: (() -> Void)?", "branch-from-here callback"),
            ("arrow.clockwise", "the footer regenerate button"),
            ("Regenerate Reply", "the context-menu regenerate item"),
            ("Edit & Resend", "the context-menu edit item"),
            ("Branch From Here", "the context-menu fork item"),
        ] {
            XCTAssertTrue(s.contains(needle), "MessageBubble lost \(what) (`\(needle)`)")
        }
    }

    func testRegenerateIsOfferedOnlyOnTheLastReplyWhileIdle() throws {
        let s = try source("Sources/MLXServe/Views/ChatView.swift")
        // The caller decides reachability; the bubble alone can't know its row.
        XCTAssertTrue(s.contains("canRegenerate"),
                      "ChatDetailView must compute whether THIS row may regenerate")
    }

    func testSidebarCarriesRenameSearchAndExportAffordances() throws {
        let s = try source("Sources/MLXServe/Views/ChatView.swift")
        for (needle, what) in [
            ("Rename…", "the rename menu item"),
            ("Export as Markdown…", "the export menu item"),
            ("NSSavePanel", "the save panel"),
            ("appState.renameSession(", "the rename action"),
            ("appState.forkSession(", "the fork action"),
            ("SidebarSearch.filter", "the search wiring"),
            ("Search chats", "the search field placeholder"),
        ] {
            XCTAssertTrue(s.contains(needle), "the sidebar lost \(what) (`\(needle)`)")
        }
    }

    func testDraftsAreWiredIntoTabSwitching() throws {
        let s = try source("Sources/MLXServe/Views/ChatView.swift")
        XCTAssertTrue(s.contains("@State private var drafts = ComposerDrafts()"),
                      "the detail view holds the draft store")
        XCTAssertTrue(s.contains("drafts.stash("), "switching tabs stashes the draft")
        XCTAssertTrue(s.contains("drafts.restore(for:"), "returning to a tab restores it")
    }

    func testEngineShipsARerunSeamThatDoesNotAppendASecondPrompt() throws {
        let engineSrc = try source("Sources/MLXServe/Services/ChatTurnEngine.swift")
        XCTAssertTrue(engineSrc.contains("func rerunLastTurn(sessionId:"),
                      "the shared regenerate/edit-resend entry point")
        XCTAssertTrue(engineSrc.contains("appendUserMessage"),
                      "both turn runners must be able to stream into existing history")
        XCTAssertTrue(engineSrc.contains("truncateMessagesAfter("),
                      "rerun rewinds the transcript before streaming")
        let appStateSrc = try source("Sources/MLXServe/AppState.swift")
        XCTAssertTrue(appStateSrc.contains("func truncateMessagesAfter("),
                      "the truncation lives beside the other transcript mutations")
        XCTAssertTrue(appStateSrc.contains("nonisolated static func forkedSession("))
        XCTAssertTrue(appStateSrc.contains("nonisolated static func editedUserMessages("))
    }
}
