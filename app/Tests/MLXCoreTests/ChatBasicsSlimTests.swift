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
        XCTAssertTrue(s.contains("@State private var drafts = ComposerDrafts()"),
                      "the detail view holds the draft store")
        XCTAssertTrue(s.contains("drafts.stash("), "switching tabs stashes the draft")
        XCTAssertTrue(s.contains("drafts.restore(for:"), "returning to a tab restores it")
    }
}
