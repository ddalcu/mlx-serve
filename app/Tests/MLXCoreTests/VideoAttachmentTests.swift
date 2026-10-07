import XCTest
@testable import MLXCore

/// An attached video's frames are files under `attachments/`, never base64 in the history (#429).
final class VideoAttachmentTests: XCTestCase {

    private func tempRoot() throws -> String {
        let dir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mlx-core-video-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return dir
    }

    private let frames = [Data("frame one bytes".utf8), Data("frame two bytes".utf8)]

    func testAStoredVideoRoundTripsThroughTheHistoryByPath() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }
        let stored = AttachmentStore.storedVideo(ChatVideo(name: "clip.mp4", frames: frames), root: root)
        XCTAssertNotNil(stored.path)

        var msg = ChatMessage(role: .user, content: "describe this")
        msg.videos = [stored]
        let json = try JSONEncoder().encode(msg)
        XCTAssertFalse(String(decoding: json, as: UTF8.self).contains(frames[0].base64EncodedString()))
        let back = try JSONDecoder().decode(ChatMessage.self, from: json)
        XCTAssertEqual(back.videos?.first?.frames, frames)
    }

    func testDeletingTheChatRemovesTheVideosFrames() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }
        let stored = AttachmentStore.storedVideo(ChatVideo(name: "clip.mp4", frames: frames), root: root)
        var session = ChatSession(title: "t")
        var msg = ChatMessage(role: .user, content: "hi")
        msg.videos = [stored]
        session.messages = [msg]
        XCTAssertEqual(AttachmentStore.removablePaths(deleting: [session.id], in: [session], root: root),
                       [stored.path!])
    }

    @MainActor
    func testTheToolLoopSendsAVideoOrAudioTurnThatHasNoText() {
        var video = ChatMessage(role: .user, content: "")
        video.videos = [ChatVideo(name: "clip.mp4", frames: frames)]
        var audio = ChatMessage(role: .user, content: "")
        audio.audio = [ChatAudio(name: "Recording", pcm: Data(count: 64))]
        let history = AgentEngine.buildAgentHistory(
            messages: [video, ChatMessage(role: .assistant, content: "seen"), audio],
            contextLength: 32768, maxTokens: 4096,
            buildMultimodalContent: { (_: String, m: ChatMessage) -> Any in
                [["videos": m.videos?.count ?? 0, "clips": m.audio?.count ?? 0]]
            },
            historyImages: true)
        let users = history.filter { ($0["role"] as? String) == "user" }.compactMap { $0["content"] as? [[String: Int]] }
        XCTAssertEqual(users, [[["videos": 1, "clips": 0]], [["videos": 0, "clips": 1]]])
    }

    func testAVideoWhoseFramesAreGoneSendsNoBlock() {
        let blocks = MultimodalContent.build(text: "x", images: [], videos: [ChatVideo(name: "gone.mp4", frames: [])])
        XCTAssertFalse(blocks.contains { ($0["type"] as? String) == "video_url" })
    }
}
