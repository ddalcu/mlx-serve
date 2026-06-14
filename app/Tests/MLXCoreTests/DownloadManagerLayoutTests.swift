import XCTest
@testable import MLXCore

/// Tests for the LM-Studio-style `<author>/<repo>` on-disk layout in
/// DownloadManager. New downloads land in the 2-level layout; existing flat
/// dirs continue to resolve via the dual-scan fallback. No auto-migration —
/// users redownload or move dirs manually.
final class DownloadManagerLayoutTests: XCTestCase {
    private var tempRoot: String!

    override func setUpWithError() throws {
        tempRoot = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mlx-serve-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: tempRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(atPath: tempRoot)
    }

    // MARK: - Path resolution

    func testNewLayoutDirSplitsAuthorAndName() {
        let p = DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: "mlx-community/Qwen3.6-27B-mtp")
        XCTAssertEqual(p, (tempRoot as NSString)
            .appendingPathComponent("mlx-community")
            .appending("/Qwen3.6-27B-mtp"))
    }

    func testNewLayoutDirBareNameFallsBackToTopLevel() {
        // No author component — caller passed a bare name. Land at top level so
        // we don't fabricate an author dir.
        let p = DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: "Qwen3.6-27B-mtp")
        XCTAssertEqual(p, (tempRoot as NSString).appendingPathComponent("Qwen3.6-27B-mtp"))
    }

    func testExistingModelDirPrefersNewLayout() throws {
        // Set up both: legacy flat AND new <author>/<name>.
        let name = "demo"
        let legacy = (tempRoot as NSString).appendingPathComponent(name)
        let nested = ((tempRoot as NSString).appendingPathComponent("acme") as NSString)
            .appendingPathComponent(name)
        try makeFakeModel(at: legacy)
        try makeFakeModel(at: nested)

        let resolved = DownloadManager.existingModelDir(rootDir: tempRoot, repoId: "acme/\(name)")
        XCTAssertEqual(resolved, nested, "new layout should win over legacy when both exist")
    }

    func testExistingModelDirFallsBackToLegacy() throws {
        // Only legacy exists. With a 2-level repoId we still want it found.
        let legacy = (tempRoot as NSString).appendingPathComponent("legacy-only")
        try makeFakeModel(at: legacy)

        let resolved = DownloadManager.existingModelDir(rootDir: tempRoot, repoId: "mlx-community/legacy-only")
        XCTAssertEqual(resolved, legacy, "legacy flat layout must remain discoverable until migrated")
    }

    func testExistingModelDirReturnsNilWhenAbsent() {
        XCTAssertNil(DownloadManager.existingModelDir(rootDir: tempRoot, repoId: "nobody/missing"))
    }

    // MARK: - Drafter discovery

    func testDiscoverDraftersFindsAllPublishedVariants() throws {
        // Drafters live under different authors today (mlx-community for the
        // older bf16 quants, google for the 12B official upload). The discoverer
        // must surface every variant regardless of its author prefix.
        for variant in GemmaVariant.allCases {
            let parts = variant.drafterRepoId.split(separator: "/")
            let dir = ((tempRoot as NSString).appendingPathComponent(String(parts[0])) as NSString)
                .appendingPathComponent(String(parts[1]))
            try makeDrafterDir(at: dir)
        }
        let found = DownloadManager.discoverDrafters(in: [tempRoot])
        XCTAssertEqual(Set(found.map { $0.variant }), Set(GemmaVariant.allCases))
    }

    func testDiscoverDraftersSkipsDirsWithWrongModelType() throws {
        // Wrong dirname: looks Gemma-shaped but isn't on the list.
        let bogus = ((tempRoot as NSString).appendingPathComponent("mlx-community") as NSString)
            .appendingPathComponent("gemma-4-other-it-assistant-bf16")
        try makeDrafterDir(at: bogus)
        // Right dirname but wrong model_type — NOT a drafter.
        let lookalike = ((tempRoot as NSString).appendingPathComponent("mlx-community") as NSString)
            .appendingPathComponent(GemmaVariant.E2B.drafterDirName)
        try FileManager.default.createDirectory(atPath: lookalike, withIntermediateDirectories: true)
        let cfg = (lookalike as NSString).appendingPathComponent("config.json")
        try "{\"model_type\":\"gemma4\"}".write(toFile: cfg, atomically: true, encoding: .utf8)

        XCTAssertTrue(DownloadManager.discoverDrafters(in: [tempRoot]).isEmpty)
    }

    func testDiscoverDraftersFirstRootWins() throws {
        // Same variant in two roots — earlier root takes precedence so a
        // user copy in ~/.mlx-serve/ wins over a leftover LM Studio copy.
        let alt = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mlx-serve-tests-alt-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(atPath: alt) }
        let primary = ((tempRoot as NSString).appendingPathComponent("mlx-community") as NSString)
            .appendingPathComponent(GemmaVariant.E4B.drafterDirName)
        let secondary = ((alt as NSString).appendingPathComponent("mlx-community") as NSString)
            .appendingPathComponent(GemmaVariant.E4B.drafterDirName)
        try makeDrafterDir(at: primary)
        try makeDrafterDir(at: secondary)

        let found = DownloadManager.discoverDrafters(in: [tempRoot, alt])
        XCTAssertEqual(found.count, 1)
        XCTAssertEqual(found.first?.url.path, primary)
    }

    func testGemmaVariantParsing() {
        XCTAssertEqual(DownloadManager.gemmaVariantFor(modelPath: "/m/gemma-4-e4b-it-4bit", isMoE: false), .E4B)
        XCTAssertEqual(DownloadManager.gemmaVariantFor(modelPath: "/m/gemma-4-e2b-it-8bit", isMoE: false), .E2B)
        XCTAssertEqual(DownloadManager.gemmaVariantFor(modelPath: "/m/gemma-4-12b-it-4bit", isMoE: false), .gemma12B)
        XCTAssertEqual(DownloadManager.gemmaVariantFor(modelPath: "/m/gemma-4-31b-it-4bit", isMoE: false), .gemma31B)
        XCTAssertEqual(DownloadManager.gemmaVariantFor(modelPath: "/m/gemma-4-26b-a4b-it-4bit", isMoE: true), .moe26B)
        // isMoE alone should also pick MoE so we route correctly even if the
        // path doesn't include the size designator.
        XCTAssertEqual(DownloadManager.gemmaVariantFor(modelPath: "/m/something-weird", isMoE: true), .moe26B)
        XCTAssertNil(DownloadManager.gemmaVariantFor(modelPath: "/m/qwen3-7b-4bit", isMoE: false))
    }

    // MARK: - Per-variant drafter repo paths

    /// All Gemma 4 drafters use the uniform mlx-community bf16 path —
    /// pinned because mlx-community only publishes 8bit for the new 12B
    /// drafter, and an earlier wholesale switch to 8bit was reverted after
    /// HF 401'd on the four older variants. Keep one suffix for consistency.
    func testDrafterRepoIdMatchesPublishedConvention() {
        XCTAssertEqual(GemmaVariant.E2B.drafterRepoId,      "mlx-community/gemma-4-E2B-it-assistant-bf16")
        XCTAssertEqual(GemmaVariant.E4B.drafterRepoId,      "mlx-community/gemma-4-E4B-it-assistant-bf16")
        XCTAssertEqual(GemmaVariant.gemma12B.drafterRepoId, "mlx-community/gemma-4-12B-it-assistant-bf16")
        XCTAssertEqual(GemmaVariant.moe26B.drafterRepoId,   "mlx-community/gemma-4-26B-A4B-it-assistant-bf16")
        XCTAssertEqual(GemmaVariant.gemma31B.drafterRepoId, "mlx-community/gemma-4-31B-it-assistant-bf16")
    }

    /// The 12B drafter declares `model_type: "gemma4_unified_assistant"` —
    /// a newer "unified" architecture spanning dense + MoE targets, distinct
    /// from the original `gemma4_assistant`. Both must classify as drafters
    /// so the dir doesn't surface as a base model in the tray-menu picker
    /// and doesn't trip the red "Unsupported architecture" label.
    func testDiscoverDraftersAcceptsUnifiedAssistantModelType() throws {
        let dir = ((tempRoot as NSString).appendingPathComponent("mlx-community") as NSString)
            .appendingPathComponent(GemmaVariant.gemma12B.drafterDirName)
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let cfg = (dir as NSString).appendingPathComponent("config.json")
        try "{\"model_type\":\"gemma4_unified_assistant\"}".write(toFile: cfg, atomically: true, encoding: .utf8)

        let found = DownloadManager.discoverDrafters(in: [tempRoot])
        XCTAssertEqual(found.first?.variant, .gemma12B)
    }

    // MARK: - GGUF classification & discovery

    func testGgufModelTypeRoutesDsv4ToDs4AndOthersToLlama() {
        // DeepSeek-V4-Flash → ds4 engine (case-insensitive).
        XCTAssertEqual(DownloadManager.ggufModelType(forBasename: "DeepSeek-V4-Flash-Q4_K_M.gguf"), "deepseek_v4")
        XCTAssertEqual(DownloadManager.ggufModelType(forBasename: "deepseek-v4-flash-bf16.gguf"), "deepseek_v4")
        // Any other GGUF → llama.cpp engine ("gguf").
        XCTAssertEqual(DownloadManager.ggufModelType(forBasename: "qwen2.5-0.5b-instruct-q4_k_m.gguf"), "gguf")
        XCTAssertEqual(DownloadManager.ggufModelType(forBasename: "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"), "gguf")
        // Not a GGUF → nil (won't be surfaced via the GGUF fast-path).
        XCTAssertNil(DownloadManager.ggufModelType(forBasename: "model.safetensors"))
        XCTAssertNil(DownloadManager.ggufModelType(forBasename: "config.json"))
    }

    func testGgufModelTypesAreSupportedArchitectures() {
        // Both engines' GGUF modelTypes must pass the architecture gate so the
        // model browser doesn't flag them "Unsupported architecture".
        for mt in ["gguf", "deepseek_v4"] {
            let m = LocalModel(
                id: "test:\(mt)", name: mt, path: "/tmp/x.gguf",
                sizeFormatted: "1 GB", modelType: mt, source: .custom, kind: .base
            )
            XCTAssertTrue(m.isSupportedArchitecture, "\"\(mt)\" must be in supportedModelTypes")
        }
    }

    func testQwen3MoeIsSupportedArchitecture() {
        // Qwen3-30B-A3B / Qwen3-Coder-30B-A3B ship model_type "qwen3_moe".
        // A locally-discovered checkpoint must NOT be flagged "Unsupported
        // architecture" in the model manager (issue #19).
        for mt in ["qwen3_moe", "qwen3_moe_text"] {
            let m = LocalModel(
                id: "test:\(mt)", name: mt, path: "/tmp/Qwen3-Coder-30B-A3B-8bit",
                sizeFormatted: "32 GB", modelType: mt, source: .custom, kind: .base
            )
            XCTAssertTrue(m.isSupportedArchitecture, "\"\(mt)\" must be in supportedModelTypes")
        }
    }

    // MARK: - mmproj sidecar filtering

    /// `mmproj-*.gguf` files are CLIP / audio encoders, not language models —
    /// llama.cpp refuses them with "unsupported model architecture: 'clip'".
    /// The model-picker must skip them when scanning a vision-enabled folder
    /// (Gemma 4 VL, Qwen 3.6 VL, etc. ship both files side-by-side).
    func testIsMmprojGgufMatchesRealSidecars() {
        // Real mmproj basenames seen across the model zoo.
        XCTAssertTrue(DownloadManager.isMmprojGguf("mmproj-gemma-4-E4B-it-BF16.gguf"))
        XCTAssertTrue(DownloadManager.isMmprojGguf("mmproj-gemma-4-E2B-it-BF16.gguf"))
        XCTAssertTrue(DownloadManager.isMmprojGguf("mmproj-Qwen3.6-27B-VL-BF16.gguf"))
        XCTAssertTrue(DownloadManager.isMmprojGguf("MMPROJ-foo.gguf"))   // case-insensitive
        XCTAssertTrue(DownloadManager.isMmprojGguf("mmproj.gguf"))       // bare prefix
        // Real LLM .gguf files MUST NOT match.
        XCTAssertFalse(DownloadManager.isMmprojGguf("gemma-4-E4B-it-Q4_K_M.gguf"))
        XCTAssertFalse(DownloadManager.isMmprojGguf("Qwen3.5-4B-IQ4_NL.gguf"))
        XCTAssertFalse(DownloadManager.isMmprojGguf("DeepSeek-V4-Flash-Q4_K_M.gguf"))
        // Suffix-only — "model-mmproj.gguf" is NOT the wild-type convention.
        XCTAssertFalse(DownloadManager.isMmprojGguf("model-mmproj.gguf"))
        // Non-.gguf — not a sidecar.
        XCTAssertFalse(DownloadManager.isMmprojGguf("mmproj-readme.md"))
        XCTAssertFalse(DownloadManager.isMmprojGguf("mmproj"))
    }

    func testIsSupportedGgufExcludesMmprojSidecars() {
        // Real LLM .gguf is supported.
        XCTAssertTrue(DownloadManager.isSupportedGguf("gemma-4-E4B-it-Q4_K_M.gguf"))
        XCTAssertTrue(DownloadManager.isSupportedGguf("DeepSeek-V4-Flash-Q4_K_M.gguf"))
        // mmproj sidecars are NOT — this is the regression that made the model
        // picker hand the wrong .gguf to the server.
        XCTAssertFalse(DownloadManager.isSupportedGguf("mmproj-gemma-4-E4B-it-BF16.gguf"))
        XCTAssertFalse(DownloadManager.isSupportedGguf("mmproj-Qwen3.6-27B-VL-BF16.gguf"))
        // Non-.gguf: not a GGUF at all.
        XCTAssertFalse(DownloadManager.isSupportedGguf("config.json"))
    }

    // MARK: - Cancellation cleanup (full wipe)
    //
    // User-cancel removes the ENTIRE download dir — completed shards, config,
    // and `.partial` files alike — so a cancel leaves zero footprint: no remnant
    // that masquerades as a complete model, no undeletable config-only orphan.
    // (Network-error resume is a separate path that keeps `.partial`s; it does
    // NOT go through this wipe.)

    func testCancelWipeRemovesCompletedShardsNotJustPartials() throws {
        // The late-cancel case: some shards finished before the user hit cancel.
        let repoId = "acme/demo"
        let author = (tempRoot as NSString).appendingPathComponent("acme")
        let dir = DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: repoId)
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let finalCfg = (dir as NSString).appendingPathComponent("config.json")
        let doneShard = (dir as NSString).appendingPathComponent("model-00001.safetensors")
        let inFlight = (dir as NSString).appendingPathComponent("model-00002.safetensors.partial")
        try "{}".write(toFile: finalCfg, atomically: true, encoding: .utf8)
        FileManager.default.createFile(atPath: doneShard, contents: Data("w".utf8))
        FileManager.default.createFile(atPath: inFlight, contents: Data())

        DownloadManager.removeModelFiles(at: dir, roots: [tempRoot])

        let fm = FileManager.default
        XCTAssertFalse(fm.fileExists(atPath: dir), "whole download dir must be gone, completed shards included")
        XCTAssertFalse(fm.fileExists(atPath: author), "now-empty author dir must be pruned")
        XCTAssertTrue(fm.fileExists(atPath: tempRoot), "scan root must survive")
    }

    func testCancelWipeRemovesConfigOnlyOrphan() throws {
        // The previously-invisible case: config downloaded, no shard yet. Must
        // not be left behind (it wouldn't surface as a deletable LocalModel).
        let repoId = "acme/justconfig"
        let dir = DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: repoId)
        try makeFakeModel(at: dir)   // config.json only

        DownloadManager.removeModelFiles(at: dir, roots: [tempRoot])

        XCTAssertFalse(FileManager.default.fileExists(atPath: dir))
    }

    func testCancelWipeNoOpWhenDirMissing() {
        // Cancelling a fresh download that bailed before mkdir must not crash.
        let dir = DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: "ghost/never-created")
        XCTAssertFalse(DownloadManager.removeModelFiles(at: dir, roots: [tempRoot]))
    }

    // MARK: - Delete by on-disk path
    //
    // `LocalModelRow` used to delete via `deleteModel(repoId: model.id)`, but a
    // LocalModel's `id` is source-prefixed (`"mlxServe:author/name"`), so the
    // repoId-based path resolver looked under `<root>/mlxServe:author/name` and
    // deleted nothing — the trash button silently no-op'd and the user had to
    // `rm -rf` from a terminal. Deletion now keys off the model's real resolved
    // `path` instead. These pin that behavior.

    func testRemoveModelFilesDeletesNestedDirAndPrunesAuthor() throws {
        let author = (tempRoot as NSString).appendingPathComponent("acme")
        let modelDir = (author as NSString).appendingPathComponent("demo")
        try makeFakeModel(at: modelDir)
        let partial = (modelDir as NSString).appendingPathComponent("model.safetensors.partial")
        FileManager.default.createFile(atPath: partial, contents: Data())

        let removed = DownloadManager.removeModelFiles(at: modelDir, roots: [tempRoot])

        let fm = FileManager.default
        XCTAssertTrue(removed)
        XCTAssertFalse(fm.fileExists(atPath: modelDir), "model dir (incl. .partial) must be gone")
        XCTAssertFalse(fm.fileExists(atPath: author), "now-empty author dir must be pruned")
        XCTAssertTrue(fm.fileExists(atPath: tempRoot), "the scan root itself must never be removed")
    }

    func testRemoveModelFilesFromGgufFilePath() throws {
        // A GGUF model's `path` is the .gguf file, not its containing dir.
        let modelDir = ((tempRoot as NSString).appendingPathComponent("team") as NSString)
            .appendingPathComponent("mygguf")
        try FileManager.default.createDirectory(atPath: modelDir, withIntermediateDirectories: true)
        let gguf = (modelDir as NSString).appendingPathComponent("model-Q4_K_M.gguf")
        FileManager.default.createFile(atPath: gguf, contents: Data("x".utf8))

        let removed = DownloadManager.removeModelFiles(at: gguf, roots: [tempRoot])

        XCTAssertTrue(removed)
        XCTAssertFalse(FileManager.default.fileExists(atPath: modelDir),
                       "deleting by a file path must remove its containing model dir")
    }

    func testRemoveModelFilesKeepsAuthorWithSurvivingSiblings() throws {
        let author = (tempRoot as NSString).appendingPathComponent("acme")
        let a = (author as NSString).appendingPathComponent("model-a")
        let b = (author as NSString).appendingPathComponent("model-b")
        try makeFakeModel(at: a)
        try makeFakeModel(at: b)

        DownloadManager.removeModelFiles(at: a, roots: [tempRoot])

        let fm = FileManager.default
        XCTAssertFalse(fm.fileExists(atPath: a))
        XCTAssertTrue(fm.fileExists(atPath: author), "author dir must survive while a sibling model remains")
        XCTAssertTrue(fm.fileExists(atPath: b))
    }

    func testRemoveModelFilesLegacyFlatDoesNotPruneRoot() throws {
        // Legacy flat layout: the model dir sits directly under a root, so its
        // "author" IS the root — pruning must stop there.
        let modelDir = (tempRoot as NSString).appendingPathComponent("flatmodel")
        try makeFakeModel(at: modelDir)

        DownloadManager.removeModelFiles(at: modelDir, roots: [tempRoot])

        let fm = FileManager.default
        XCTAssertFalse(fm.fileExists(atPath: modelDir))
        XCTAssertTrue(fm.fileExists(atPath: tempRoot), "a root must never be pruned even when emptied")
    }

    func testRemoveModelFilesRefusesToDeleteARoot() throws {
        // Defensive: never remove a root directory itself.
        let removed = DownloadManager.removeModelFiles(at: tempRoot, roots: [tempRoot])
        XCTAssertFalse(removed)
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempRoot))
    }

    func testRemoveModelFilesMissingPathIsNoOp() {
        let ghost = (tempRoot as NSString).appendingPathComponent("nope/missing")
        XCTAssertFalse(DownloadManager.removeModelFiles(at: ghost, roots: [tempRoot]))
    }

    // MARK: - Cancellation detection
    //
    // Cancelling an in-flight download surfaces as URLSession's
    // NSURLErrorCancelled, NOT Swift's CancellationError — so the retry loop
    // must recognize both, or a cancelled transfer flashes "retrying…" before
    // it finally unwinds.

    func testIsCancellationMatchesCancellationErrorAndURLCancel() {
        XCTAssertTrue(DownloadManager.isCancellation(CancellationError()))
        XCTAssertTrue(DownloadManager.isCancellation(URLError(.cancelled)))
        XCTAssertTrue(DownloadManager.isCancellation(
            NSError(domain: NSURLErrorDomain, code: NSURLErrorCancelled)))
    }

    func testIsCancellationRejectsTransientFailures() {
        // These are genuine transient errors the retry loop must keep retrying.
        XCTAssertFalse(DownloadManager.isCancellation(URLError(.timedOut)))
        XCTAssertFalse(DownloadManager.isCancellation(URLError(.networkConnectionLost)))
        XCTAssertFalse(DownloadManager.isCancellation(URLError(.notConnectedToInternet)))
    }

    // MARK: - Downloaded tab live-refresh trigger
    //
    // The Downloaded tab's "Size on Disk" comes from a disk re-scan
    // (`refreshModels`), which previously only fired on tab entry — so sizes
    // froze mid-download until the user toggled the button. The tab now
    // live-polls, but only while it's showing AND a download is in flight.

    func testShouldLivePollOnlyWhenTabOpenAndDownloading() {
        XCTAssertTrue(ModelBrowserView.shouldLivePoll(downloadedTab: true, hasActiveDownloads: true))
        // No active downloads → nothing to refresh, don't spin.
        XCTAssertFalse(ModelBrowserView.shouldLivePoll(downloadedTab: true, hasActiveDownloads: false))
        // Not in the Downloaded tab → the sizes aren't even visible.
        XCTAssertFalse(ModelBrowserView.shouldLivePoll(downloadedTab: false, hasActiveDownloads: true))
        XCTAssertFalse(ModelBrowserView.shouldLivePoll(downloadedTab: false, hasActiveDownloads: false))
    }

    // MARK: - Helpers

    /// Minimal model dir layout: just `config.json`. The path-resolution and
    /// migration logic only checks for that file's presence.
    private func makeFakeModel(at path: String) throws {
        try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        let cfg = (path as NSString).appendingPathComponent("config.json")
        try "{}".write(toFile: cfg, atomically: true, encoding: .utf8)
    }

    /// Drafter dir: config.json with `model_type: "gemma4_assistant"`.
    private func makeDrafterDir(at path: String) throws {
        try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        let cfg = (path as NSString).appendingPathComponent("config.json")
        try "{\"model_type\":\"gemma4_assistant\"}".write(toFile: cfg, atomically: true, encoding: .utf8)
    }
}
