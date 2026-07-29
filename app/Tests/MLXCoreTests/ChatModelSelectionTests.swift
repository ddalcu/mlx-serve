import XCTest
@testable import MLXCore

/// The chat model a picker shows and what picking a row means.
///
/// Extracted because there are now TWO pickers (the menu-bar tray and the chat
/// window's toolbar) and they must agree. A per-surface copy of this logic is
/// exactly how one picker ends up ignoring a LAN selection — the same class as
/// the rule that a chat surface routes through `server.chatModelId` rather than
/// reading `modelInfo?.name` for itself.
final class ChatModelSelectionTests: XCTestCase {

    func testLanSelectionWinsOverTheLocalPath() {
        // A LAN chat is served by another Mac; the local `selectedModelPath` is
        // still set underneath and must not be what the picker ticks.
        XCTAssertEqual(
            ChatModelSelection.tag(localPath: "/models/local", lanChatModelId: "qwen@studio"),
            "lan:qwen@studio")
    }

    func testLocalPathIsTheTagWhenNoLanModelIsSelected() {
        XCTAssertEqual(ChatModelSelection.tag(localPath: "/models/local", lanChatModelId: nil),
                       "/models/local")
    }

    func testPickingALanRowSelectsTheLanModel() {
        XCTAssertEqual(ChatModelSelection.action(for: "lan:qwen@studio"), .selectLan("qwen@studio"))
    }

    func testPickingALocalRowClearsTheLanSelection() {
        // Without the clear, a local pick would leave the LAN id set and every
        // turn would keep going out to the network.
        XCTAssertEqual(ChatModelSelection.action(for: "/models/local"), .selectLocal("/models/local"))
    }

    func testTagsRoundTrip() {
        // Class guard: whatever the picker shows must decode back to the same
        // choice, or the checkmark lands on a row that isn't what loads.
        for (path, lan) in [("/a", nil), ("/b", "m@peer"), ("", "x@y")] as [(String, String?)] {
            let tag = ChatModelSelection.tag(localPath: path, lanChatModelId: lan)
            switch ChatModelSelection.action(for: tag) {
            case .selectLan(let id): XCTAssertEqual(id, lan)
            case .selectLocal(let p): XCTAssertEqual(p, path)
            }
        }
    }

    func testAPathContainingTheLanWordIsNotTreatedAsALanId() {
        // Only the "lan:" PREFIX marks a network row — a local folder called
        // "lan" or a path with "lan:" inside it must still load locally.
        XCTAssertEqual(ChatModelSelection.action(for: "/Users/me/lan/models"),
                       .selectLocal("/Users/me/lan/models"))
    }
}
