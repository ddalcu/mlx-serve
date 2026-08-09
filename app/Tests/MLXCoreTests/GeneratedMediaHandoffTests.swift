import XCTest
@testable import MLXCore

/// "Send to Chat" on a Create-pane result. The two ways to make an asset stay
/// distinct — Create is the workshop, chat is the conversation — and this is the
/// one bridge between them.
final class GeneratedMediaHandoffTests: XCTestCase {

    func testTheHandoffIsAUserMessage() {
        let msg = GeneratedMediaHandoff.message(
            path: "/tmp/out/fox.png", prompt: "a red fox in snow", kind: .image)

        // NOT an assistant message. The model in the new chat did not make this
        // and has never seen it; filing it as model output is the same class of
        // lie as rendering an error as something the model said.
        XCTAssertEqual(msg.role, .user)
        XCTAssertEqual(msg.content, "a red fox in snow")
        XCTAssertEqual(msg.media?.count, 1)
        XCTAssertEqual(msg.media?.first?.path, "/tmp/out/fox.png")
        XCTAssertEqual(msg.media?.first?.kind, .image)
        // The prompt rides the ref too — it is the caption under the player, and
        // what makes a bare filename mean something in a months-old transcript.
        XCTAssertEqual(msg.media?.first?.prompt, "a red fox in snow")
    }

    /// A generation with no prompt still has to say what the message IS, or the
    /// transcript opens with an empty bubble and a file.
    func testAnEmptyPromptFallsBackToNamingTheAsset() {
        for (kind, expected) in [(ChatMediaRef.Kind.image, "Generated image"),
                                 (.video, "Generated video"),
                                 (.audio, "Generated audio")] {
            let msg = GeneratedMediaHandoff.message(path: "/tmp/x", prompt: "   ", kind: kind)
            XCTAssertEqual(msg.content, expected)
            // The ref's own caption stays empty rather than echoing our
            // stand-in: nothing was asked for, and inventing a prompt would
            // put words in the user's mouth.
            XCTAssertEqual(msg.media?.first?.prompt, "")
        }
    }

    /// The panes must not each build their own hand-off. Four copies of "make a
    /// session, make a message, switch modes" is four chances for one of them to
    /// land in the wrong chat.
    func testEveryPaneUsesTheSharedHandoff() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        for pane in ["ImageGenView", "VideoGenView", "AudioGenView"] {
            let text = try String(
                contentsOf: root.appendingPathComponent("Sources/MLXServe/Views/\(pane).swift"),
                encoding: .utf8)
            XCTAssertTrue(text.contains("sendGeneratedMediaToNewChat("), """
                \(pane) must hand off through AppState.sendGeneratedMediaToNewChat \
                — the one place that decides WHICH chat a result lands in.
                """)
        }
    }
}
