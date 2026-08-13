import XCTest
@testable import MLXCore

/// ↑ in an empty composer brings back what you last said — the recall every
/// shell and every chat app has, and the thing you reach for the moment you
/// send a message with a typo in it.
///
/// The whole design problem is that ↑ already means something: move the caret.
/// So the walk arms only from the EDGE of the text (↑ at the very start, ↓ at
/// the very end), and it ends the instant you edit what came back — after that
/// the arrows are caret keys again, inside a draft that is now yours.
final class ComposerHistoryTests: XCTestCase {

    private func user(_ text: String) -> ChatMessage {
        ChatMessage(role: .user, content: text)
    }

    private func assistant(_ text: String) -> ChatMessage {
        ChatMessage(role: .assistant, content: text)
    }

    // MARK: - What is in the history

    func testOnlyYourOwnMessagesAreRecalled() {
        let messages = [user("first"), assistant("a reply"), user("second")]
        XCTAssertEqual(ComposerHistory.entries(messages), ["first", "second"])
    }

    func testTheOldestIsFirstSoUpWalksBackwards() {
        let messages = [user("oldest"), user("newest")]
        XCTAssertEqual(ComposerHistory.entries(messages).last, "newest")
    }

    /// Sending the same thing twice in a row (a retry) should cost one ↑, not
    /// two — the second press would otherwise look like it did nothing.
    func testConsecutiveRepeatsCollapse() {
        let messages = [user("go"), user("go"), user("stop"), user("go")]
        XCTAssertEqual(ComposerHistory.entries(messages), ["go", "stop", "go"])
    }

    /// An empty or whitespace-only turn is not something anyone wants back.
    func testBlankMessagesAreNotHistory() {
        XCTAssertEqual(ComposerHistory.entries([user("  \n "), user("real")]), ["real"])
    }

    /// An image-only turn carries no text to recall. (The pictures are not
    /// re-attached either — recall is about the words.)
    func testAnImageOnlyTurnHasNothingToRecall() {
        var msg = ChatMessage(role: .user, content: "")
        msg.images = [ChatImage(data: Data())]
        XCTAssertTrue(ComposerHistory.entries([msg]).isEmpty)
    }

    // MARK: - Walking back

    func testUpInAnEmptyComposerBringsBackTheLastThingYouSaid() {
        let action = ComposerHistory.up(draft: "", caretAtStart: true,
                                        walk: .idle, entries: ["first", "second"])
        XCTAssertEqual(action, .recall(text: "second", walk: .init(index: 1)))
    }

    func testUpAgainStepsFurtherBack() {
        let action = ComposerHistory.up(draft: "second", caretAtStart: true,
                                        walk: .init(index: 1), entries: ["first", "second"])
        XCTAssertEqual(action, .recall(text: "first", walk: .init(index: 0)))
    }

    /// The end of the history is the end. Wrapping around to the newest would
    /// read as a different message arriving, the same reason the revision
    /// pager refuses to wrap.
    func testUpStopsAtTheOldest() {
        XCTAssertEqual(ComposerHistory.up(draft: "first", caretAtStart: true,
                                          walk: .init(index: 0), entries: ["first", "second"]),
                       .pass)
    }

    func testNothingToRecallPassesTheKeyThrough() {
        XCTAssertEqual(ComposerHistory.up(draft: "", caretAtStart: true,
                                          walk: .idle, entries: []),
                       .pass)
    }

    // MARK: - The two things that must NOT be swallowed

    /// A draft you typed is not a walk. ↑ with the caret at the start of a
    /// paragraph you are writing must move the caret — recalling there would
    /// replace work with an old message, and there is no undo for that.
    func testUpNeverReplacesADraftYouTyped() {
        XCTAssertEqual(ComposerHistory.up(draft: "half a thought", caretAtStart: true,
                                          walk: .idle, entries: ["earlier"]),
                       .pass)
    }

    /// Inside a multi-line recalled message the arrows go back to being arrows
    /// until the caret reaches the edge — which is exactly how a terminal
    /// behaves, and the only way to edit line two of what came back.
    func testUpMovesTheCaretUntilItReachesTheStart() {
        XCTAssertEqual(ComposerHistory.up(draft: "line one\nline two", caretAtStart: false,
                                          walk: .init(index: 0), entries: ["line one\nline two"]),
                       .pass)
    }

    /// Editing what came back ends the walk: the field holds your text now, so
    /// ↑ is a caret key again rather than something that discards it.
    func testEditingWhatCameBackEndsTheWalk() {
        XCTAssertEqual(ComposerHistory.up(draft: "second, but longer", caretAtStart: true,
                                          walk: .init(index: 1), entries: ["first", "second"]),
                       .pass)
    }

    // MARK: - Walking forward

    func testDownStepsBackTowardsTheNewest() {
        XCTAssertEqual(ComposerHistory.down(draft: "first", caretAtEnd: true,
                                            walk: .init(index: 0), entries: ["first", "second"]),
                       .recall(text: "second", walk: .init(index: 1)))
    }

    /// Past the newest is the empty composer you started from — the walk ends
    /// there rather than sticking on the last message.
    func testDownPastTheNewestLeavesYouWithAnEmptyComposer() {
        XCTAssertEqual(ComposerHistory.down(draft: "second", caretAtEnd: true,
                                            walk: .init(index: 1), entries: ["first", "second"]),
                       .recall(text: "", walk: .idle))
    }

    func testDownDoesNothingWhenNoWalkIsInProgress() {
        XCTAssertEqual(ComposerHistory.down(draft: "", caretAtEnd: true,
                                            walk: .idle, entries: ["first"]),
                       .pass)
    }

    func testDownMovesTheCaretUntilItReachesTheEnd() {
        XCTAssertEqual(ComposerHistory.down(draft: "first", caretAtEnd: false,
                                            walk: .init(index: 0), entries: ["first", "second"]),
                       .pass)
    }

    /// A walk started in one conversation must not survive into another, and
    /// the transcript can also grow underneath one (a reply lands mid-walk).
    /// A stale index is treated as no walk at all rather than read out of
    /// bounds.
    func testAStaleIndexIsNotAWalk() {
        XCTAssertEqual(ComposerHistory.down(draft: "gone", caretAtEnd: true,
                                            walk: .init(index: 7), entries: ["first"]),
                       .pass)
        XCTAssertEqual(ComposerHistory.up(draft: "gone", caretAtStart: true,
                                          walk: .init(index: 7), entries: ["first"]),
                       .pass)
    }

    // MARK: - Wiring

    /// The composer has to hand the keys over, and it has to hand over the
    /// caret position with them — without that the rule cannot tell "move the
    /// caret" from "recall", and it would swallow both.
    func testTheComposerFieldRoutesTheArrowKeysWithTheCaretPosition() throws {
        let source = SourceScan.source("Views/ChatView.swift", from: #filePath)
        XCTAssertTrue(source.contains("NSResponder.moveUp"),
                      "the composer never sees ↑ — it is a doCommandBy selector")
        XCTAssertTrue(source.contains("NSResponder.moveDown"),
                      "the composer never sees ↓")
        XCTAssertTrue(source.contains("ComposerHistory.up("),
                      "↑ is intercepted but the recall rule is never asked")
        XCTAssertTrue(source.contains("ComposerHistory.down("),
                      "↓ is intercepted but the recall rule is never asked")
    }

    /// Sending ends the walk. Otherwise the next ↑ resumes from wherever the
    /// last one left off, in a composer that is empty for a different reason.
    func testSendingEndsTheWalk() throws {
        let source = SourceScan.source("Views/ChatView.swift", from: #filePath)
        let body = try XCTUnwrap(
            SourceScan.declarationBody(from: "private func sendMessage", in: source),
            "sendMessage moved — repoint this scan")
        XCTAssertTrue(body.contains("composerWalk = .idle"),
                      "a sent message must end the recall walk")
    }
}
