import XCTest
@testable import MLXCore

/// The gen-first→chat-later hole (live 2026-07-05): a server launched
/// HEADLESS for media generation has no default model, so chat surfaces that
/// address the "mlx-serve" alias got `503 no_model` even with a model
/// selected in the app. Chat surfaces now hot-load the selected model first —
/// exactly once per server process, and only when it's actually needed.
final class ServerEnsureChatModelTests: XCTestCase {

    func testEnsureFiresOnlyForRunningHeadlessServerWithASelection() {
        // The bug case: running + headless + selection → must load.
        XCTAssertTrue(ServerManager.shouldEnsureChatDefault(
            running: true, launchedModelPath: "", alreadyEnsured: false,
            selectedModelPath: "/m/mlx-community/gemma-4-12b-it-4bit"))
        // Launched WITH --model → the server has a default; never interfere.
        XCTAssertFalse(ServerManager.shouldEnsureChatDefault(
            running: true, launchedModelPath: "/m/gemma", alreadyEnsured: false,
            selectedModelPath: "/m/gemma"))
        // Not running yet → the normal start path owns model selection.
        XCTAssertFalse(ServerManager.shouldEnsureChatDefault(
            running: false, launchedModelPath: "", alreadyEnsured: false,
            selectedModelPath: "/m/gemma"))
        // Already ensured once this process → no per-turn HTTP round trip.
        XCTAssertFalse(ServerManager.shouldEnsureChatDefault(
            running: true, launchedModelPath: "", alreadyEnsured: true,
            selectedModelPath: "/m/gemma"))
        // Nothing selected → nothing to offer; the request 503s honestly.
        XCTAssertFalse(ServerManager.shouldEnsureChatDefault(
            running: true, launchedModelPath: "", alreadyEnsured: false,
            selectedModelPath: ""))
    }
}
