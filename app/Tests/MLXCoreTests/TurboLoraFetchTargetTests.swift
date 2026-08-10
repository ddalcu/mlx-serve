import XCTest
@testable import MLXCore

/// The Turbo-adapter fetch targets the PACK, and its cancel is surgical.
///
/// Live 2026-08-08: the fetch wrote to the destination root while the pack
/// lived in another owned root, creating a fragment dir (config + tokenizer +
/// turbo_lora only) that read as a model to every resolver — it shadowed the
/// real pack and the server died loading it. The adapter's whole point is to
/// sit beside weights that already exist, so the download goes where the pack
/// IS. And because that directory is a live 40-69 GB pack, cancelling the
/// fetch must never take the whole-dir wipe the generic cancel path uses.
@MainActor
final class TurboLoraFetchTargetTests: XCTestCase {
    private var tempRoot: String!
    private var savedSession: URLSession!

    private let repoId = "ddalcu/fixture-h3-pack"
    private let configBody = Data("{\"model_type\":\"minimax_h3\"}".utf8)

    override func setUpWithError() throws {
        tempRoot = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mlx-serve-turbo-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: tempRoot, withIntermediateDirectories: true)
        savedSession = DownloadSession.shared
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [HuggingFaceStubProtocol.self]
        config.httpMaximumConnectionsPerHost = 20
        DownloadSession.shared = URLSession(configuration: config)
        HuggingFaceStubProtocol.reset()
    }

    override func tearDownWithError() throws {
        DownloadSession.shared = savedSession
        HuggingFaceStubProtocol.reset()
        try? FileManager.default.removeItem(atPath: tempRoot)
    }

    /// A pack at any resolvable dir that is NOT `newLayoutDir(for: repoId)` —
    /// the legacy flat path here, a different owned root in the live failure.
    private func makePack(at dir: String) throws {
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        try configBody.write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("config.json")))
        try Data("REAL-WEIGHTS".utf8).write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("transformer.safetensors")))
        try Data("tok".utf8).write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("tokenizer.json")))
    }

    private func serveAdapterRepo() {
        // Same-byte config/tokenizer so the size-matched skip leaves the
        // pack's own copies alone; the big transformer must never transfer
        // (the selection filters it), so the stub serves a DIFFERENT body —
        // if it lands anywhere, an assertion catches the overwrite.
        HuggingFaceStubProtocol.serve(repo: repoId, files: [
            ("config.json", configBody),
            ("tokenizer.json", Data("tok".utf8)),
            ("transformer.safetensors", Data("STUB-MUST-NOT-TRANSFER".utf8)),
            ("turbo_lora.safetensors", Data("LORA".utf8)),
        ])
    }

    func testAdapterLandsBesideThePackNotInAFreshDestinationDir() async throws {
        let packDir = (tempRoot as NSString).appendingPathComponent("fixture-h3-pack")
        try makePack(at: packDir)
        serveAdapterRepo()
        let manager = DownloadManager(modelsRoot: tempRoot)

        await withCheckedContinuation { cont in
            manager.startTurboLora(repoId: repoId) { cont.resume() }
        }

        let fm = FileManager.default
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: (packDir as NSString).appendingPathComponent("turbo_lora.safetensors"))),
                       Data("LORA".utf8), "adapter did not land beside the pack")
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: (packDir as NSString).appendingPathComponent("transformer.safetensors"))),
                       Data("REAL-WEIGHTS".utf8), "the pack's weights must be untouched")
        // The live bug: a fragment dir at the destination layout path.
        XCTAssertFalse(fm.fileExists(atPath: DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: repoId)),
                       "fetch created a fragment dir at the destination layout path")
    }

    func testCancelWithoutAFetchIsANoOpOnTheLivePack() throws {
        // The pack at the DESTINATION layout path — the normal case. The
        // generic `cancel(_:)` with no active task wipes the repo's whole
        // download dir, which here IS the pack; the turbo cancel must not.
        let packDir = DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: repoId)
        try makePack(at: packDir)
        let manager = DownloadManager(modelsRoot: tempRoot)

        manager.cancelTurboLora(repoId: repoId)

        XCTAssertTrue(FileManager.default.fileExists(atPath: (packDir as NSString).appendingPathComponent("transformer.safetensors")),
                      "an idle turbo cancel deleted the pack")
    }

    func testCancelMidFetchLeavesThePackAndNoPartials() async throws {
        let packDir = (tempRoot as NSString).appendingPathComponent("fixture-h3-pack")
        try makePack(at: packDir)
        serveAdapterRepo()
        let manager = DownloadManager(modelsRoot: tempRoot)

        let done = expectation(description: "fetch settled")
        manager.startTurboLora(repoId: repoId) { done.fulfill() }
        manager.cancelTurboLora(repoId: repoId)
        await fulfillment(of: [done], timeout: 10)

        let fm = FileManager.default
        // Whether the cancel won the race or the tiny fetch finished first,
        // the pack survives intact and nothing half-written stays behind.
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: (packDir as NSString).appendingPathComponent("transformer.safetensors"))),
                       Data("REAL-WEIGHTS".utf8))
        XCTAssertTrue(fm.fileExists(atPath: (packDir as NSString).appendingPathComponent("config.json")))
        let strays = (try? fm.contentsOfDirectory(atPath: packDir).filter { $0.contains(".partial") }) ?? []
        XCTAssertTrue(strays.isEmpty, "left behind: \(strays)")
        XCTAssertFalse(fm.fileExists(atPath: DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: repoId)))
    }
}
