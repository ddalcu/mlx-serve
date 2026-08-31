import XCTest
@testable import MLXCore

/// The four chat-window basics main does NOT already have: export, rename,
/// search, per-tab drafts.
///
/// Regenerate, Edit & Resend and Branch are deliberately absent — main shipped
/// all three while PR #261 was being written, which is why that branch could
/// not merge. Re-landing a second implementation of a feature that exists is
/// worse than landing nothing.
///
/// Pure decision cores are unit-tested; every control is pinned by a source
/// scan so a redesign cannot quietly stop drawing one.
final class ChatBasicsSlimTests: XCTestCase {

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
        let appStateSrc = try source("Sources/MLXServe/AppState.swift")
        // Drafts live ON AppState now, so they can survive relaunch beside the
        // chat history they belong to.
        XCTAssertTrue(s.contains("appState.stashDraft("), "switching tabs stashes the draft")
        XCTAssertTrue(s.contains("appState.draft(for:"), "returning to a tab restores it")
        XCTAssertTrue(appStateSrc.contains("composer-drafts.json"),
                      "drafts persist to disk beside chat history")
        XCTAssertTrue(appStateSrc.contains("func stashDraft("))
    }

    // MARK: - Handoff: search result context

    func testFirstContentMatchFindsTheMessageAndASnippet() {
        var s = ChatSession(title: "Rust")
        s.messages = [msg(.user, "hello"),
                      msg(.assistant, "\n\nuse borrow checker carefully\n")]
        let hit = SidebarSearch.firstContentMatch(in: s, query: "borrow")
        XCTAssertEqual(hit?.messageIndex, 1)
        XCTAssertTrue(hit?.snippet.contains("borrow") ?? false, hit?.snippet ?? "no snippet")
        XCTAssertFalse(hit?.snippet.contains("\n\n") ?? true, "snippet is one line")
    }

    func testFirstContentMatchNilWhenOnlyTitleMatched() {
        var s = ChatSession(title: "Borrow checker")
        s.messages = [msg(.user, "hi")]
        XCTAssertNil(SidebarSearch.firstContentMatch(in: s, query: "borrow"),
                     "a title-only hit has no transcript row to jump to")
    }

    func testFirstContentMatchSkipsHiddenRows() {
        var hidden = msg(.system, "needle in tool output")
        hidden.toolCallId = "c1"
        var s = ChatSession(title: "t")
        s.messages = [hidden, msg(.user, "visible needle")]
        XCTAssertEqual(SidebarSearch.firstContentMatch(in: s, query: "needle")?.messageIndex, 1)
        XCTAssertNil(SidebarSearch.firstContentMatch(in: s, query: ""),
                     "an empty query names nothing")
    }

    // MARK: - Handoff: drafts survive relaunch

    func testDraftsCodableRoundtrip() throws {
        var drafts = ComposerDrafts()
        let a = UUID(), b = UUID()
        drafts.stash("survives", for: a)
        drafts.stash("  ", for: b)
        let data = try JSONEncoder().encode(drafts)
        let back = try JSONDecoder().decode(ComposerDrafts.self, from: data)
        XCTAssertEqual(back.restore(for: a), "survives")
        XCTAssertEqual(back.restore(for: b), "")
    }

    // MARK: - Handoff: export formats

    func testExportJSONCarriesRolesContentTimestampsAndSkipsMachineRows() throws {
        var toolResult = msg(.system, "hidden")
        toolResult.toolCallId = "t1"
        var withReasoning = msg(.assistant, "answer")
        withReasoning.reasoningContent = "scratchpad"
        let data = try XCTUnwrap(ChatExport.jsonData(
            title: "T",
            messages: [msg(.user, "q"), withReasoning, toolResult],
            dateText: "2026-08-22"))
        let obj = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(obj["title"] as? String, "T")
        XCTAssertEqual(obj["exportedAt"] as? String, "2026-08-22")
        let messages = try XCTUnwrap(obj["messages"] as? [[String: Any]])
        XCTAssertEqual(messages.count, 2, "hidden tool rows stay out of the export")
        XCTAssertEqual(messages[0]["role"] as? String, "user")
        XCTAssertEqual(messages[0]["content"] as? String, "q")
        XCTAssertNotNil(messages[0]["timestamp"], "re-import wants the when as well as the what")
        XCTAssertEqual(messages[1]["reasoning"] as? String, "scratchpad")
    }

    func testExportMarkdownOfSeveralChatsJoinsThemUnderOneHeader() {
        var one = ChatSession(title: "One")
        one.messages = [msg(.user, "first chat")]
        var two = ChatSession(title: "Two")
        two.messages = [msg(.assistant, "second chat")]
        let md = ChatExport.markdown(
            sessions: [(title: "One", messages: one.messages),
                       (title: "Two", messages: two.messages)],
            dateText: "2026-08-22")
        XCTAssertTrue(md.contains("# One"))
        XCTAssertTrue(md.contains("# Two"))
        XCTAssertTrue(md.components(separatedBy: "2026-08-22").count == 2,
                      "exactly one exported-at line for the whole file")
    }

    // MARK: - Handoff affordance scans

    func testUserBubbleCarriesAHoverActionTray() throws {
        let s = try source("Sources/MLXServe/Views/ChatView.swift")
        for (needle, what) in [
            ("MessageActionTray", "the hover tray"),
            ("arrow.triangle.branch", "the branch icon"),
            ("pencil", "the edit icon"),
            ("copied ? \"checkmark\"", "the Copied flip"),
        ] {
            XCTAssertTrue(s.contains(needle), "the tray lost \(what) (`\(needle)`)")
        }
    }

    func testSearchJumpsToTheMatchingMessage() throws {
        let chat = try source("Sources/MLXServe/Views/ChatView.swift")
        let appStateSrc = try source("Sources/MLXServe/AppState.swift")
        let scrollSrc = try source("Sources/MLXServe/Services/ChatScroll.swift")
        XCTAssertTrue(appStateSrc.contains("pendingSearchJump"),
                      "AppState carries the jump request between sidebar and detail view")
        XCTAssertTrue(chat.contains("firstContentMatch"),
                      "rows surface the matching snippet")
        XCTAssertTrue(chat.contains("scrollTo(id:"),
                      "the detail view scrolls to the matched message id")
        XCTAssertTrue(scrollSrc.contains("case messageTargeted"),
                      "the scroll core knows a targeted jump releases follow")
    }

    func testSidebarOffersJSONAndMultiChatExport() throws {
        let s = try source("Sources/MLXServe/Views/ChatView.swift")
        XCTAssertTrue(s.contains("Export as JSON…"), "the JSON sibling")
        XCTAssertTrue(s.contains("Chats…"), "the N-chats export over the multi-selection")
        XCTAssertTrue(s.contains("jsonData("), "the JSON serializer is reachable")
    }

    /// Both of these are WIRING bugs: the pure cores were right and the pure
    /// tests passed, while the feature did not work. They are pinned by source
    /// scan because that seam is the only place they exist.

    /// A search hit in a chat you are NOT looking at flips `sessionId`, and
    /// that handler re-pins the transcript to the bottom. The async centered
    /// scroll lands first and is then corrected straight back down, so the
    /// jump silently did nothing in its main case — you search precisely
    /// because the chat is not in front of you.
    func testAConsumedSearchJumpIsNotUndoneByTheTabSwitchRepin() throws {
        let s = try Self.chatViewSource()
        XCTAssertTrue(s.contains("if !attemptSearchJump() {"),
                      "the tab-switch re-pin must yield to a jump that already aimed this view")
        XCTAssertFalse(s.contains("attemptSearchJump()\n            applyScroll(.transcriptShown)"),
                       "an unconditional transcriptShown after a jump re-pins to the bottom")
    }

    /// Drafts were stashed only on a tab SWITCH, so quitting or closing the
    /// window with a half-typed message threw away the very thing persistence
    /// was added for.
    func testTheVisibleTabsDraftIsStashedOnTeardownNotOnlyOnSwitch() throws {
        let s = try Self.chatViewSource()
        let onDisappear = try XCTUnwrap(s.range(of: ".onDisappear {"))
        let window = String(s[onDisappear.lowerBound...].prefix(600))
        XCTAssertTrue(window.contains("stashDraft(inputText, for: sessionId)"),
                      "closing the window must flush the visible tab's draft")
    }

    private static func chatViewSource() throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Views/ChatView.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }

    // MARK: - Review findings

    /// An agent thread's SUMMARY rows (`**name**(args)` / `**name** → output`)
    /// are folded into one tool-call row whose id is the CALL's. Indexing them
    /// leaked raw tool output into a sidebar caption and produced a hit whose
    /// id no row carries — the click released follow and scrolled nowhere.
    func testSearchIgnoresAgentSummaryRows() {
        var call = msg(.assistant, "**read_file**(path: secrets.env)")
        call.isAgentSummary = true
        var result = msg(.assistant, "**read_file** → DATABASE_PASSWORD=hunter2")
        result.isAgentSummary = true
        var session = ChatSession()
        session.title = "Some thread"
        session.messages = [msg(.user, "look at it"), call, result]

        XCTAssertFalse(SidebarSearch.matches(session, query: "hunter2"),
                       "raw tool output must not make a conversation a search hit")
        XCTAssertNil(SidebarSearch.firstContentMatch(in: session, query: "read_file"),
                     "a hit on a folded row names an id no row on screen carries")
        XCTAssertNotNil(SidebarSearch.firstContentMatch(in: session, query: "look at it"),
                        "ordinary rows still match")
    }

    /// An agent row DISPLAYS the agent's name, not `session.title`. Searching
    /// the stored title made a row literally reading "Code Reviewer" vanish
    /// when you typed "code reviewer".
    func testSearchMatchesTheTitleTheSidebarActuallyDraws() {
        var session = ChatSession()
        session.title = "New agent"
        session.messages = [msg(.user, "unrelated")]

        XCTAssertFalse(SidebarSearch.matches(session, query: "code reviewer"))
        XCTAssertTrue(SidebarSearch.matches(session, query: "code reviewer",
                                            displayTitle: "Code Reviewer"),
                      "search must read the string the row shows")
        XCTAssertEqual(SidebarSearch.filter(sessions: [session], query: "review",
                                            displayTitle: { _ in "Code Reviewer" }).count, 1)
    }

    /// The snippet is cut from the trimmed, whitespace-collapsed line, so the
    /// offset has to be measured on that same string. Measured on the raw
    /// content, a heavily indented line cut past the match and the tail cap
    /// then removed it — a caption that does not contain what you searched for.
    func testTheSnippetAlwaysContainsTheMatch() {
        let padding = String(repeating: " ", count: 60)
        let filler = String(repeating: "alpha ", count: 20)
        var session = ChatSession()
        session.messages = [msg(.user, padding + filler + "NEEDLE" + filler)]

        let hit = SidebarSearch.firstContentMatch(in: session, query: "NEEDLE")
        XCTAssertNotNil(hit)
        XCTAssertTrue(hit?.snippet.localizedCaseInsensitiveContains("NEEDLE") == true,
                      "snippet lost the match: \(hit?.snippet ?? "nil")")
        XCTAssertLessThanOrEqual(hit?.snippet.count ?? 0, SidebarSearch.snippetMaxLength)
    }

    /// A draft belongs to a conversation. When the conversation goes, so does
    /// it — otherwise composer-drafts.json only ever grows.
    func testDraftsAreForgottenWithTheirConversation() {
        var drafts = ComposerDrafts()
        let gone = UUID(), kept = UUID()
        drafts.stash("half a message", for: gone)
        drafts.stash("keep me", for: kept)

        drafts.clear(for: gone)
        XCTAssertEqual(drafts.restore(for: gone), "")
        XCTAssertEqual(drafts.restore(for: kept), "keep me")
    }

    /// Deleting the ACTIVE chat flips `activeChatId`, and the view then stashes
    /// the outgoing field under the id just removed — resurrecting a dead
    /// session's draft. Both the delete sweep and the stash guard are pinned.
    func testDeletingAConversationDropsItsDraftAndCannotResurrectIt() throws {
        let s = try Self.appStateSource()
        XCTAssertTrue(s.contains("forgetDrafts(ids)"),
                      "deleteSessions must drop the drafts of what it removed")
        let stash = try XCTUnwrap(s.range(of: "func stashDraft("))
        let window = String(s[stash.lowerBound...].prefix(700))
        XCTAssertTrue(window.contains("chatSessions.contains(where:"),
                      "a stash for a session that no longer exists must be dropped")
    }

    /// The save panel closes whether the write worked or not, so silence reads
    /// as success. A full disk must not look like a saved file.
    func testAFailedExportIsReported() throws {
        let s = try Self.chatViewSource()
        XCTAssertTrue(s.contains("func reportExportFailure("),
                      "there is no way to tell the user an export failed")
        XCTAssertFalse(s.contains("try? data.write("),
                       "a swallowed write error is an export that silently did nothing")
    }

    private static func appStateSource() throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/AppState.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }
}
