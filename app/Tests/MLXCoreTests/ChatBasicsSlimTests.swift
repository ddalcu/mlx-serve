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
}
