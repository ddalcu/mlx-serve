import XCTest
@testable import MLXCore

/// `/v1/models` was scoped to the selected model's ONE org folder because the
/// `--model` launch passed the model's parent dir as `--model-dir`. So a user
/// with models under several orgs (mlx-community, ddalcu, antirez, …) saw only
/// a subset in `/v1/models` even though the app's dropdown scanned the whole
/// tree. `discoveryModelDir` now points `--model-dir` at the whole models root
/// (the server discovers `<root>/<org>/<model>` two levels deep) so the two
/// match.
final class ServerDiscoveryModelDirTests: XCTestCase {

    private let root = "/Users/x/.mlx-serve/models"

    /// A model under the models root → scan the WHOLE root (all orgs), not just
    /// its org folder. This is the fix.
    func testModelUnderRootScansTheWholeRoot() {
        XCTAssertEqual(
            ServerManager.discoveryModelDir(
                selectedModel: "\(root)/mlx-community/gemma-4-26B-A4B-it-qat-4bit",
                modelsRoot: root),
            root)
        // A deeper/other-org model still resolves to the same root.
        XCTAssertEqual(
            ServerManager.discoveryModelDir(
                selectedModel: "\(root)/ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve",
                modelsRoot: root),
            root)
    }

    /// A model OUTSIDE the models root (LM Studio / a custom folder) → fall back
    /// to its parent so at least its siblings surface (the server takes a single
    /// --model-dir, so we can't scan both trees).
    func testModelOutsideRootFallsBackToItsParent() {
        XCTAssertEqual(
            ServerManager.discoveryModelDir(
                selectedModel: "/Users/x/.lmstudio/models/lmstudio-community/Some-Model-GGUF/model.gguf",
                modelsRoot: root),
            "/Users/x/.lmstudio/models/lmstudio-community/Some-Model-GGUF")
    }

    /// A trailing slash on the root must not defeat the under-root check.
    func testTrailingSlashOnRootIsNormalized() {
        XCTAssertEqual(
            ServerManager.discoveryModelDir(
                selectedModel: "\(root)/mlx-community/gemma-4-e4b-it-4bit",
                modelsRoot: root + "/"),
            root + "/")
    }
}
