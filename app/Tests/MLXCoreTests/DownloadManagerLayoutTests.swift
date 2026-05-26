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

    func testDiscoverDraftersFindsAllFourVariants() throws {
        for variant in GemmaVariant.allCases {
            let dir = ((tempRoot as NSString).appendingPathComponent("mlx-community") as NSString)
                .appendingPathComponent(variant.drafterDirName)
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
        XCTAssertEqual(DownloadManager.gemmaVariantFor(modelPath: "/m/gemma-4-31b-it-4bit", isMoE: false), .gemma31B)
        XCTAssertEqual(DownloadManager.gemmaVariantFor(modelPath: "/m/gemma-4-26b-a4b-it-4bit", isMoE: true), .moe26B)
        // isMoE alone should also pick MoE so we route correctly even if the
        // path doesn't include the size designator.
        XCTAssertEqual(DownloadManager.gemmaVariantFor(modelPath: "/m/something-weird", isMoE: true), .moe26B)
        XCTAssertNil(DownloadManager.gemmaVariantFor(modelPath: "/m/qwen3-7b-4bit", isMoE: false))
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
