import XCTest
@testable import MLXCore

/// A finished row must compare equal across transcript rebuilds, or every
/// streamed batch re-renders the whole conversation.
final class MessageBubbleEquatableTests: XCTestCase {
    private let message = ChatMessage(role: .assistant, content: "hello")

    func testFreshClosuresOverTheSameMessageCompareEqual() {
        let a = MessageBubble(message: message, onDelete: {}, onRegenerate: {})
        let b = MessageBubble(message: message, onDelete: {}, onRegenerate: {})
        XCTAssertEqual(a, b)
    }

    func testChangedContentIsNotEqual() {
        var grown = message
        grown.content += " world"
        XCTAssertNotEqual(MessageBubble(message: message), MessageBubble(message: grown))
    }

    func testGainingOrLosingAnActionIsNotEqual() {
        XCTAssertNotEqual(MessageBubble(message: message, onRegenerate: {}),
                          MessageBubble(message: message))
        XCTAssertNotEqual(MessageBubble(message: message, onContinue: {}),
                          MessageBubble(message: message))
    }
}
