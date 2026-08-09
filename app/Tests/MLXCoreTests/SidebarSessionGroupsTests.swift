import XCTest
@testable import MLXCore

/// The sidebar splits its conversation list into Agents and Chats, and names a
/// thread after its content once it has some. These pin both.
final class SidebarSessionGroupsTests: XCTestCase {

    private func session(agent: UUID? = nil, title: String = ChatSessionTitle.plain) -> ChatSession {
        var s = ChatSession(title: title)
        s.agentId = agent
        return s
    }

    // MARK: - The split

    /// Nothing in the Agents half means the section is hidden — an empty
    /// heading promises content that isn't there.
    func testNoAgentThreadsLeavesTheAgentsHalfEmpty() {
        let groups = SidebarSessionGroups.split([session(), session()])
        XCTAssertTrue(groups.agents.isEmpty)
        XCTAssertEqual(groups.chats.count, 2)
    }

    func testAgentThreadsGoAboveAndPlainChatsBelow() {
        let chef = UUID()
        let groups = SidebarSessionGroups.split([
            session(agent: chef, title: "Dinner plans"),
            session(title: "Tell me about dolphins"),
        ])
        XCTAssertEqual(groups.agents.map(\.title), ["Dinner plans"])
        XCTAssertEqual(groups.chats.map(\.title), ["Tell me about dolphins"])
    }

    /// Exhaustive and disjoint: every session lands in exactly one half. A
    /// session in neither is invisible with nothing to point at, and one in
    /// both renders twice.
    func testEverySessionLandsInExactlyOneHalf() {
        let all = [session(agent: UUID()), session(), session(agent: UUID()), session()]
        let groups = SidebarSessionGroups.split(all)
        XCTAssertEqual(groups.agents.count + groups.chats.count, all.count)
        let ids = Set(groups.agents.map(\.id)).intersection(groups.chats.map(\.id))
        XCTAssertTrue(ids.isEmpty, "a session appears in both sections")
        XCTAssertEqual(Set(all.map(\.id)),
                       Set(groups.agents.map(\.id)).union(groups.chats.map(\.id)))
    }

    /// The list is already newest-first; the split must not reorder it.
    func testOrderIsPreservedWithinEachHalf() {
        let a = UUID()
        let groups = SidebarSessionGroups.split([
            session(agent: a, title: "first"),
            session(title: "second"),
            session(agent: a, title: "third"),
            session(title: "fourth"),
        ])
        XCTAssertEqual(groups.agents.map(\.title), ["first", "third"])
        XCTAssertEqual(groups.chats.map(\.title), ["second", "fourth"])
    }

    /// Keyed on the session's own agentId, never on whether that agent still
    /// exists — a thread created as an agent's stays one after the agent is
    /// deleted, and the row already drops the subtitle when there is nobody to
    /// name.
    func testAThreadWhoseAgentWasDeletedStaysAnAgentThread() {
        let groups = SidebarSessionGroups.split([session(agent: UUID(), title: "orphaned")])
        XCTAssertEqual(groups.agents.map(\.title), ["orphaned"])
        XCTAssertTrue(groups.chats.isEmpty)
    }

    // MARK: - Titles

    func testAnAgentThreadStartsAsNewAgentAndAPlainOneAsNewChat() {
        XCTAssertEqual(ChatSessionTitle.placeholder(hasAgent: true), "New agent")
        XCTAssertEqual(ChatSessionTitle.placeholder(hasAgent: false), "New Chat")
    }

    /// The auto-titler's gate. Both placeholders, or an agent thread keeps the
    /// name "New agent" for the rest of its life.
    func testEveryPlaceholderIsRecognized() {
        XCTAssertTrue(ChatSessionTitle.isPlaceholder("New Chat"))
        XCTAssertTrue(ChatSessionTitle.isPlaceholder("New agent"))
        XCTAssertFalse(ChatSessionTitle.isPlaceholder("Tell me about dolphins"),
                       "a titled thread must never be renamed out from under the user")
        XCTAssertFalse(ChatSessionTitle.isPlaceholder(""))
    }

    /// Threads created before agents had their own section are stored as
    /// "New Chat" with an agent attached. Normalizing at DISPLAY fixes them
    /// without rewriting anything on disk, and self-corrects both ways. An
    /// agent thread is named for its agent — the rest of that rule lives in
    /// `ChatSessionTitleTests`.
    func testAPlaceholderIsDrawnAsTheKindOfThreadItIs() {
        XCTAssertEqual(ChatSessionTitle.display(title: "New Chat", agentName: "Chef"), "Chef")
        XCTAssertEqual(ChatSessionTitle.display(title: "New agent", agentName: nil), "New Chat")
        XCTAssertEqual(ChatSessionTitle.display(title: "New Chat", agentName: nil), "New Chat")
    }

    /// A title the thread earned is never rewritten — that would rename a
    /// conversation out from under the user. On an agent thread the agent's
    /// name takes the title line, so the thread's own title becomes its
    /// SUBJECT rather than being lost.
    func testARealTitleIsNeverNormalized() {
        XCTAssertEqual(ChatSessionTitle.display(title: "Dinner plans", agentName: nil),
                       "Dinner plans")
        XCTAssertEqual(ChatSessionTitle.subject(title: "Dinner plans", agentName: "Chef"),
                       "Dinner plans")
    }

    func testTheFirstMessageNamesTheThread() {
        XCTAssertEqual(ChatSessionTitle.derived(fromFirstMessage: "Tell me about dolphins"),
                       "Tell me about dolphins")
    }

    /// A message that can't name anything leaves the placeholder in place —
    /// better "New agent" than a blank row.
    func testAnEmptyOrBlankMessageNamesNothing() {
        XCTAssertNil(ChatSessionTitle.derived(fromFirstMessage: ""))
        XCTAssertNil(ChatSessionTitle.derived(fromFirstMessage: "   \n  "))
    }

    /// Long content is cut and marked; content that fits is left alone — the
    /// ellipsis must not appear on a title that fits.
    func testLongContentIsTruncatedAndShortContentIsNot() {
        let exact = String(repeating: "a", count: 40)
        XCTAssertEqual(ChatSessionTitle.derived(fromFirstMessage: exact), exact,
                       "exactly at the limit still fits")
        let long = String(repeating: "a", count: 41)
        XCTAssertEqual(ChatSessionTitle.derived(fromFirstMessage: long),
                       String(repeating: "a", count: 40) + "...")
    }

    /// Counting in characters and slicing the same way: an emoji earlier in the
    /// message must not shift where the cut lands.
    func testTruncationCountsCharactersNotBytes() {
        let content = "👨‍👩‍👧‍👦 " + String(repeating: "b", count: 50)
        let title = ChatSessionTitle.derived(fromFirstMessage: content)
        XCTAssertEqual(title?.count, 43, "40 characters plus the ellipsis")
    }
}
