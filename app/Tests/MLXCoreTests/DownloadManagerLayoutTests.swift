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

    // MARK: - Cancellation cleanup

    /// User-cancel must leave `.partial` files gone (no Resume path) and final
    /// files untouched. Pinned because the cancel button promises this exact
    /// behavior; a regression would silently leave half-files on disk and the
    /// "Resume" button would reappear on the next launch.
    func testCleanupPartialsRemovesPartialsKeepsFinalFiles() throws {
        let repoId = "acme/demo"
        let dir = DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: repoId)
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let finalCfg = (dir as NSString).appendingPathComponent("config.json")
        let topPartial = (dir as NSString).appendingPathComponent("model.safetensors.partial")
        let subdir = (dir as NSString).appendingPathComponent("nested")
        try FileManager.default.createDirectory(atPath: subdir, withIntermediateDirectories: true)
        let nestedPartial = (subdir as NSString).appendingPathComponent("shard-1.safetensors.partial")
        try "{}".write(toFile: finalCfg, atomically: true, encoding: .utf8)
        FileManager.default.createFile(atPath: topPartial, contents: Data())
        FileManager.default.createFile(atPath: nestedPartial, contents: Data())

        DownloadManager.cleanupPartials(rootDir: tempRoot, repoId: repoId)

        let fm = FileManager.default
        XCTAssertTrue(fm.fileExists(atPath: finalCfg), "completed files must stay on disk")
        XCTAssertFalse(fm.fileExists(atPath: topPartial), "top-level .partial must be removed")
        XCTAssertFalse(fm.fileExists(atPath: nestedPartial), "nested .partial must be removed")
        XCTAssertTrue(fm.fileExists(atPath: dir), "dest dir must remain when other files survive")
    }

    func testCleanupPartialsRemovesEmptyDestDir() throws {
        let repoId = "acme/empty"
        let dir = DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: repoId)
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let onlyPartial = (dir as NSString).appendingPathComponent("only.partial")
        FileManager.default.createFile(atPath: onlyPartial, contents: Data())

        DownloadManager.cleanupPartials(rootDir: tempRoot, repoId: repoId)

        XCTAssertFalse(FileManager.default.fileExists(atPath: dir),
                       "dest dir should be removed when it's empty after partials are deleted")
    }

    func testCleanupPartialsNoOpWhenDirMissing() {
        // Cancelling a fresh download that bailed before mkdir must not crash.
        DownloadManager.cleanupPartials(rootDir: tempRoot, repoId: "ghost/never-created")
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
