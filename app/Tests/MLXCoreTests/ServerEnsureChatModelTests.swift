import XCTest
@testable import MLXCore

/// The gen-first→chat-later hole (live 2026-07-05): a server launched
/// HEADLESS for media generation has no default model, so chat surfaces that
/// address the "mlx-serve" alias got `503 no_model` even with a model
/// selected in the app. Chat surfaces now hot-load the selected model first,
/// whenever no chat model is resident (an unload or eviction included).
final class ServerEnsureChatModelTests: XCTestCase {

    func testEnsureFiresOnlyForRunningHeadlessServerWithASelection() {
        // The bug case: running + headless + selection → must load.
        XCTAssertTrue(ServerManager.shouldEnsureChatDefault(
            running: true, launchedModelPath: "", chatResident: false,
            selectedModelPath: "/m/mlx-community/gemma-4-12b-it-4bit"))
        // Launched WITH --model → the server has a default; never interfere.
        XCTAssertFalse(ServerManager.shouldEnsureChatDefault(
            running: true, launchedModelPath: "/m/gemma", chatResident: false,
            selectedModelPath: "/m/gemma"))
        // Not running yet → the normal start path owns model selection.
        XCTAssertFalse(ServerManager.shouldEnsureChatDefault(
            running: false, launchedModelPath: "", chatResident: false,
            selectedModelPath: "/m/gemma"))
        // A chat model is resident → no per-turn load.
        XCTAssertFalse(ServerManager.shouldEnsureChatDefault(
            running: true, launchedModelPath: "", chatResident: true,
            selectedModelPath: "/m/gemma"))
        // Nothing selected → nothing to offer; the request 503s honestly.
        XCTAssertFalse(ServerManager.shouldEnsureChatDefault(
            running: true, launchedModelPath: "", chatResident: false,
            selectedModelPath: ""))
    }

    /// #925: with a provider or LAN model picked, the remembered local model is not a load target.
    @MainActor
    func testRemoteChatNeverLoadsTheRememberedLocalModel() {
        let saved = UserDefaults.standard.string(forKey: "lanChatModelId")
        defer { UserDefaults.standard.set(saved, forKey: "lanChatModelId") }
        let server = ServerManager()
        server.status = .running
        let local = "/models/org/unrelated-local-model"
        for remote in ["qwen@studio", "/models/qwen@Splash"] {
            server.lanChatModelId = remote
            XCTAssertNil(server.localChatLoadPath(selectedPath: local))
            XCTAssertFalse(server.chatLoadNeeded(selectedModelPath: local))
        }
        server.lanChatModelId = nil
        XCTAssertEqual(server.localChatLoadPath(selectedPath: local), local)
        XCTAssertNil(server.localChatLoadPath(selectedPath: ""))
        XCTAssertTrue(server.chatLoadNeeded(selectedModelPath: local))
    }

    func testComposerLoadProgressFollowsTheChatTargetWhileTheTrayShowsEveryLoad() {
        for remote in [nil, "qwen@studio"] as [String?] {
            for apple in [false, true] {
                XCTAssertEqual(ChatModelSelection.showsLocalLoadProgress(lanChatModelId: remote, apple: apple, chatTargetOnly: true),
                               remote == nil && !apple)
                XCTAssertTrue(ChatModelSelection.showsLocalLoadProgress(lanChatModelId: remote, apple: apple, chatTargetOnly: false))
            }
        }
    }
}
