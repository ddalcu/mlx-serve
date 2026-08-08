import XCTest
@testable import MLXCore

/// After the download destination moves, `~/.mlx-serve/models` must stay a
/// SOURCE: the server kept scanning it (`ModelRoots.scanRoots`), but the app's
/// own reads — picker, ready-checks, media-gen resolution — read only the
/// destination, so the whole pre-move library vanished from the picker while
/// `/v1/models` was still serving it.
@MainActor
final class OwnedRootDiscoveryTests: XCTestCase {

    private var made: [String] = []

    override func tearDown() {
        for p in made { try? FileManager.default.removeItem(atPath: p) }
        made = []
        super.tearDown()
    }

    private func tempRoot(_ name: String) -> String {
        let p = NSTemporaryDirectory() + "OwnedRootTests-\(name)-\(UUID().uuidString)"
        try? FileManager.default.createDirectory(atPath: p, withIntermediateDirectories: true)
        made.append(p)
        return p
    }

    /// Minimal loadable-looking checkpoint: config.json + one safetensors file
    /// (what `makeLocalModels` and `holdsWeightLayout` gate on).
    private func makeModel(root: String, repo: String) throws {
        let dir = (root as NSString).appendingPathComponent(repo)
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        try #"{"model_type": "llama"}"#.write(
            toFile: (dir as NSString).appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8)
        try Data("x".utf8).write(
            to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("model.safetensors")))
    }

    private func resolved(_ p: String) -> String { (p as NSString).resolvingSymlinksInPath }

    // MARK: - Picker discovery

    /// Models in every owned root are listed, the FIRST root's copy winning a
    /// repeated id — the same first-wins rule the server applies to repeated
    /// `--model-dir` flags, with the destination first in both lists.
    func testModelsInEveryOwnedRootAreListedFirstRootWinningARepeatedId() throws {
        let dest = tempRoot("dest")
        let builtIn = tempRoot("builtin")
        try makeModel(root: dest, repo: "org/shared")
        try makeModel(root: builtIn, repo: "org/shared")
        try makeModel(root: builtIn, repo: "org/old-only")

        let models = DownloadManager.mlxServeModels(inRoots: [dest, builtIn])
        XCTAssertEqual(models.count, 2, "the duplicate id must collapse, the pre-move model must appear")
        XCTAssertTrue(models.allSatisfy { $0.source == .mlxServe })

        let shared = models.first { $0.name == "org/shared" }
        XCTAssertEqual(shared?.path, resolved((dest as NSString).appendingPathComponent("org/shared")),
                       "the destination's copy wins a repeated id")
        XCTAssertTrue(models.contains { $0.name == "org/old-only" },
                      "models downloaded before the destination moved must stay in the picker")
    }

    // MARK: - "Is this repo on disk?"

    /// The repo resolver checks every owned root in order — this is what keeps
    /// ready-checks and browser badges honest about a library downloaded
    /// before the destination moved.
    func testExistingModelDirFallsBackToTheBuiltInRootCopy() throws {
        let dest = tempRoot("dest")
        let builtIn = tempRoot("builtin")
        try makeModel(root: builtIn, repo: "org/old-only")

        XCTAssertEqual(DownloadManager.existingModelDir(roots: [dest, builtIn], repoId: "org/old-only"),
                       (builtIn as NSString).appendingPathComponent("org/old-only"))

        // Present in both → the destination's copy, not the built-in one.
        try makeModel(root: dest, repo: "org/old-only")
        XCTAssertEqual(DownloadManager.existingModelDir(roots: [dest, builtIn], repoId: "org/old-only"),
                       (dest as NSString).appendingPathComponent("org/old-only"))

        XCTAssertNil(DownloadManager.existingModelDir(roots: [dest, builtIn], repoId: "org/absent"))
    }

    /// Media-gen resolution ("is the pack on disk?") checks every owned root —
    /// resolving against the destination alone would offer a full re-download
    /// of a pack sitting in `~/.mlx-serve/models`.
    func testMediaGenResolutionChecksEveryOwnedRoot() throws {
        let dest = tempRoot("dest")
        let builtIn = tempRoot("builtin")
        try makeModel(root: builtIn, repo: "org/video-pack")

        XCTAssertEqual(ServerManager.resolveModelDir(repo: "org/video-pack", roots: [dest, builtIn]),
                       (builtIn as NSString).appendingPathComponent("org/video-pack"))
        XCTAssertNil(ServerManager.resolveModelDir(repo: "org/absent", roots: [dest, builtIn]))
    }

    // MARK: - Test hermeticity

    /// A test-pinned DownloadManager never grows the real built-in root: a
    /// temp-dir test must not resolve into — or delete from — the developer's
    /// own library.
    func testAPinnedDownloadManagerStaysHermetic() {
        let pinned = tempRoot("pinned")
        let dm = DownloadManager(modelsRoot: pinned)
        XCTAssertEqual(dm.ownedRoots, [pinned])
    }
}
