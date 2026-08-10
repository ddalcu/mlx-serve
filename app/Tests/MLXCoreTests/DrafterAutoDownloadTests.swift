import XCTest
@testable import MLXCore

/// The Gemma 4 assistant drafter is a DEPENDENCY of the model it pairs with,
/// not a thing to shop for.
///
/// It used to have its own Model Browser destination — a catalog of five
/// `*-it-assistant-bf16` repos with no obvious relationship to the models
/// people had actually downloaded. Now the pairing checkpoint comes down with
/// its target, the same way a ds4 GGUF quant pulls its MTP head
/// (`resolveGgufDownloadFiles`).
///
/// The MoE target is deliberately excluded: the drafter REGRESSES decode there
/// (verify pays expert routing — the server defaults it off on MoE targets), so
/// fetching it would cost bandwidth for something we then refuse to use.
@MainActor
final class DrafterAutoDownloadTests: XCTestCase {
    private var tempRoot: String!
    private var savedSession: URLSession!

    private let denseRepo = "mlx-community/gemma-4-e4b-it-4bit"
    private var drafterRepo: String { GemmaVariant.E4B.drafterRepoId }
    private let moeRepo = "mlx-community/gemma-4-26b-a4b-it-4bit"

    override func setUpWithError() throws {
        tempRoot = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mlx-serve-drafter-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: tempRoot, withIntermediateDirectories: true)
        savedSession = DownloadSession.shared
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [HuggingFaceStubProtocol.self]
        DownloadSession.shared = URLSession(configuration: config)
        HuggingFaceStubProtocol.reset()
    }

    override func tearDownWithError() throws {
        DownloadSession.shared = savedSession
        HuggingFaceStubProtocol.reset()
        try? FileManager.default.removeItem(atPath: tempRoot)
    }

    // MARK: - The pairing rule

    func testEveryDenseGemma4PairsWithItsOwnDrafter() {
        let cases: [(String, GemmaVariant)] = [
            ("mlx-community/gemma-4-e2b-it-4bit", .E2B),
            ("mlx-community/gemma-4-e4b-it-8bit", .E4B),
            ("mlx-community/gemma-4-12b-it-4bit", .gemma12B),
            ("mlx-community/gemma-4-31b-it-8bit", .gemma31B),
            // Case is a repo author's choice, not a signal.
            ("google/gemma-4-E4B-it-qat", .E4B),
        ]
        for (repo, variant) in cases {
            XCTAssertEqual(DownloadManager.companionDrafterRepo(forRepoId: repo), variant.drafterRepoId,
                           "\(repo) must pull the \(variant.label) drafter")
        }
    }

    func testMuseGlimmerPairsWithItsDFlashAssistant() {
        for repo in ["ddalcu/Muse-Glimmer-30B-MLX-Serve-8bit", "meta-models/Muse-Glimmer-30B"] {
            XCTAssertEqual(DownloadManager.companionDrafterRepo(forRepoId: repo),
                           "meta-models/Muse-Glimmer-30B-assistant",
                           "\(repo) must pull the DFlash assistant")
        }
        // The assistant must not pull itself.
        XCTAssertNil(DownloadManager.companionDrafterRepo(forRepoId: "meta-models/Muse-Glimmer-30B-assistant"))
    }

    func testTheMoeGemmaHasNoCompanionDrafter() {
        for repo in ["mlx-community/gemma-4-26b-a4b-it-4bit", "mlx-community/gemma-4-26b-a4b-it-8bit"] {
            XCTAssertNil(DownloadManager.companionDrafterRepo(forRepoId: repo),
                         "the drafter regresses decode on the MoE target — never fetch it there")
        }
    }

    func testNothingElsePullsADrafter() {
        // A drafter repo must not pull itself, or a download is an infinite regress.
        XCTAssertNil(DownloadManager.companionDrafterRepo(forRepoId: GemmaVariant.E4B.drafterRepoId))
        // Another family that happens to carry the same size token.
        XCTAssertNil(DownloadManager.companionDrafterRepo(forRepoId: "Qwen/Qwen3.5-12B-Instruct"))
        XCTAssertNil(DownloadManager.companionDrafterRepo(forRepoId: "mlx-community/gemma-3-12b-it-4bit"))
        // GGUF Gemma runs on llama.cpp, which has no drafter path at all.
        XCTAssertNil(DownloadManager.companionDrafterRepo(forRepoId: "unsloth/gemma-4-12b-it-GGUF"))
        // A size we publish no drafter for.
        XCTAssertNil(DownloadManager.companionDrafterRepo(forRepoId: "mlx-community/gemma-4-270m-it-4bit"))
    }

    // MARK: - The download actually brings it

    func testDownloadingADenseGemmaAlsoLandsItsDrafter() async throws {
        HuggingFaceStubProtocol.serve(repos: [denseRepo: Self.repoFiles(), drafterRepo: Self.repoFiles()])
        let manager = DownloadManager(modelsRoot: tempRoot)

        await start(manager, repoId: denseRepo)

        XCTAssertTrue(manager.isReady(denseRepo), "the model itself must land")
        XCTAssertTrue(manager.isReady(drafterRepo),
                      "the drafter is the model's dependency — it comes down with it")
    }

    func testDownloadingTheMoeGemmaAsksForNoDrafter() async throws {
        HuggingFaceStubProtocol.serve(repos: [moeRepo: Self.repoFiles(),
                                              GemmaVariant.moe26B.drafterRepoId: Self.repoFiles()])
        let manager = DownloadManager(modelsRoot: tempRoot)

        await start(manager, repoId: moeRepo)

        XCTAssertTrue(manager.isReady(moeRepo))
        XCTAssertFalse(HuggingFaceStubProtocol.requests.contains { $0.path.contains("assistant") },
                       "the MoE target must not even ASK for a drafter")
    }

    func testADrafterAlreadyOnDiskIsNotFetchedAgain() async throws {
        try Self.plant(repo: drafterRepo, under: tempRoot)
        HuggingFaceStubProtocol.serve(repos: [denseRepo: Self.repoFiles(), drafterRepo: Self.repoFiles()])
        let manager = DownloadManager(modelsRoot: tempRoot)
        XCTAssertTrue(manager.isReady(drafterRepo), "fixture should already read as ready")

        await start(manager, repoId: denseRepo)

        XCTAssertFalse(HuggingFaceStubProtocol.requests.contains { $0.path.contains("assistant") },
                       "a drafter already on disk must not be re-fetched")
    }

    // MARK: - Helpers

    private func start(_ manager: DownloadManager, repoId: String) async {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            manager.start(repoId: repoId) { cont.resume() }
        }
    }

    /// The minimum shape `isReady` accepts: config + tokenizer + a shard over
    /// its 1 MB "not a stub" floor.
    private static func repoFiles() -> [(String, Data)] {
        [
            ("config.json", Data(#"{"model_type":"gemma4"}"#.utf8)),
            ("tokenizer.json", Data(#"{"version":"1.0"}"#.utf8)),
            ("model.safetensors", Data(count: 2 << 20)),
        ]
    }

    private static func plant(repo: String, under root: String) throws {
        let dir = DownloadManager.newLayoutDir(rootDir: root, repoId: repo)
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        for (name, body) in repoFiles() {
            try body.write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent(name)))
        }
    }
}
