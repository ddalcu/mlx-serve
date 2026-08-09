import XCTest
@testable import MLXCore

/// What the sidebar calls a conversation.
///
/// An agent thread is named for its AGENT: the Agents section is a list of who
/// you talk to, so the row has to say who, and it said so only in a caption
/// under a title derived from the first thing you happened to type. The
/// conversation's own subject moves to that caption instead, which is also what
/// keeps a second thread with the same agent distinguishable from the first.
final class ChatSessionTitleTests: XCTestCase {

    func testAnAgentThreadIsNamedForItsAgent() {
        XCTAssertEqual(ChatSessionTitle.display(title: "whats that picture above?",
                                                agentName: "UX Designer"),
                       "UX Designer")
    }

    /// Renaming the agent renames every one of its rows, because the row reads
    /// the agent's name rather than a copy taken when the thread was made.
    func testTheNameIsTheAgentsCurrentName() {
        let before = ChatSessionTitle.display(title: "New agent", agentName: "Chef")
        let after = ChatSessionTitle.display(title: "New agent", agentName: "Sous Chef")
        XCTAssertEqual(before, "Chef")
        XCTAssertEqual(after, "Sous Chef")
    }

    /// A plain conversation is untouched — it has no agent to be named for.
    func testAPlainConversationKeepsItsOwnTitle() {
        XCTAssertEqual(ChatSessionTitle.display(title: "Tell me about dolphins", agentName: nil),
                       "Tell me about dolphins")
        XCTAssertEqual(ChatSessionTitle.display(title: "New Chat", agentName: nil),
                       ChatSessionTitle.plain)
    }

    /// A half-saved agent with a blank name can't name a row: falling through
    /// to the placeholder beats drawing an empty one.
    func testABlankAgentNameFallsBackToThePlaceholder() {
        XCTAssertEqual(ChatSessionTitle.display(title: "New Chat", agentName: "   "),
                       ChatSessionTitle.agent)
        XCTAssertEqual(ChatSessionTitle.display(title: "New Chat", agentName: ""),
                       ChatSessionTitle.agent)
    }

    /// A thread stored before agents had a section — "New Chat" with an agent
    /// attached — still reads as an agent thread, with no migration on disk.
    func testAnOldPlaceholderNormalizesToTheThreadsKind() {
        XCTAssertEqual(ChatSessionTitle.display(title: "New Chat", agentName: nil),
                       ChatSessionTitle.plain)
        XCTAssertEqual(ChatSessionTitle.display(title: "New Chat", agentName: "Coder"),
                       "Coder")
    }

    // MARK: The subject moves to the caption

    /// With the agent's name on the title line, the conversation's own subject
    /// is what tells two threads of the same agent apart.
    func testTheConversationsOwnSubjectBecomesTheCaption() {
        XCTAssertEqual(ChatSessionTitle.subject(title: "Fix the login bug", agentName: "Coder"),
                       "Fix the login bug")
    }

    /// A thread that has said nothing has no subject — a caption repeating the
    /// placeholder says nothing twice.
    func testAnUnstartedThreadHasNoCaption() {
        XCTAssertNil(ChatSessionTitle.subject(title: "New agent", agentName: "Coder"))
        XCTAssertNil(ChatSessionTitle.subject(title: "New Chat", agentName: "Coder"))
    }

    /// A plain conversation's title is already on its title line, so it never
    /// gets a caption repeating it.
    func testAPlainConversationHasNoCaption() {
        XCTAssertNil(ChatSessionTitle.subject(title: "Tell me about dolphins", agentName: nil))
    }

    /// The auto-titler's gate is unchanged: it fires while a thread still
    /// carries a placeholder, whichever one.
    func testThePlaceholderGateStillCoversBothKinds() {
        XCTAssertTrue(ChatSessionTitle.isPlaceholder("New Chat"))
        XCTAssertTrue(ChatSessionTitle.isPlaceholder("New agent"))
        XCTAssertFalse(ChatSessionTitle.isPlaceholder("Coder"))
    }
}
