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

    // MARK: - File selection (recursive tree, incl. mtp/ sidecar)

    /// Regression for the silent MTP-sidecar drop: a model download must pull
    /// the nested `mtp/weights.safetensors` head (else the model loses its
    /// speculative-decoding speedup), while NOT pulling unrelated nested
    /// subdirectories (e.g. `original/` alternate-precision shadow weights).
    func testSelectNeededFilesIncludesMtpSidecarSkipsOtherNestedDirs() {
        let entries: [[String: Any]] = [
            ["path": "config.json", "type": "file", "size": 100],
            ["path": "model-00001-of-00003.safetensors", "type": "file", "size": 5_000_000_000],
            ["path": "tokenizer.json", "type": "file", "size": 19_000_000],
            ["path": "chat_template.jinja", "type": "file", "size": 7_000],
            ["path": "README.md", "type": "file", "size": 5_000],
            ["path": ".DS_Store", "type": "file", "size": 6_000],
            ["path": "mtp", "type": "directory", "size": 0],
            ["path": "mtp/weights.safetensors", "type": "file", "size": 524_000_000],
            ["path": "original/model.safetensors", "type": "file", "size": 50_000_000_000],
        ]
        let paths = Set(DownloadManager.selectNeededFiles(from: entries).map { $0.0 })

        XCTAssertTrue(paths.contains("mtp/weights.safetensors"), "MTP sidecar must be downloaded")
        XCTAssertTrue(paths.contains("config.json"))
        XCTAssertTrue(paths.contains("model-00001-of-00003.safetensors"))
        XCTAssertTrue(paths.contains("tokenizer.json"))
        XCTAssertTrue(paths.contains("chat_template.jinja"))

        XCTAssertFalse(paths.contains("mtp"), "a directory entry is not a downloadable file")
        XCTAssertFalse(paths.contains("README.md"), "non-weight markdown skipped")
        XCTAssertFalse(paths.contains(".DS_Store"), "files without a needed extension skipped")
        XCTAssertFalse(paths.contains("original/model.safetensors"),
                       "nested non-mtp shadow weights must not be pulled")

        // Sidecar size is threaded through for the progress/space pre-check.
        let sidecar = DownloadManager.selectNeededFiles(from: entries).first { $0.0 == "mtp/weights.safetensors" }
        XCTAssertEqual(sidecar?.1, 524_000_000)
    }

    /// oMLX OptiQ repos ship the MTP head as `optiq/mtp.safetensors` (a sibling
    /// of mlx-serve's `mtp/` layout) alongside `optiq/optiq_vision.safetensors`.
    /// The head must be pulled (server auto-loads it, delta-norms folded at load)
    /// but the relocated vision tower must NOT (the server can't use it and it's
    /// GBs). Mirrors mtp.sidecar_rel_paths.
    func testSelectNeededFilesIncludesOptiQMtpHeadSkipsOptiQVision() {
        let entries: [[String: Any]] = [
            ["path": "config.json", "type": "file", "size": 108_000],
            ["path": "model-00001-of-00004.safetensors", "type": "file", "size": 5_300_000_000],
            ["path": "tokenizer.json", "type": "file", "size": 19_000_000],
            ["path": "optiq", "type": "directory", "size": 0],
            ["path": "optiq/mtp.safetensors", "type": "file", "size": 314_000_000],
            ["path": "optiq/optiq_vision.safetensors", "type": "file", "size": 1_200_000_000],
        ]
        let picked = DownloadManager.selectNeededFiles(from: entries)
        let paths = Set(picked.map { $0.0 })

        XCTAssertTrue(paths.contains("optiq/mtp.safetensors"), "OptiQ MTP head must be downloaded")
        XCTAssertTrue(paths.contains("config.json"))
        XCTAssertTrue(paths.contains("model-00001-of-00004.safetensors"))
        XCTAssertFalse(paths.contains("optiq/optiq_vision.safetensors"),
                       "relocated vision tower is unusable by the server — must not be pulled")
        XCTAssertEqual(picked.first { $0.0 == "optiq/mtp.safetensors" }?.1, 314_000_000)
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

    // MARK: - Internal helper models (hidden from the Model Browser)

    /// Class guard: the NSFW classifier is an app-internal dependency, not a
    /// model the user chose — it must be in the exclusion set `discoverLocalModels`
    /// filters against, or it reappears in My Models (as a confusing red
    /// "Unsupported" row, since its `vit` architecture isn't a chat model).
    func testInternalHelperReposExcludesTheNsfwClassifier() {
        XCTAssertTrue(DownloadManager.internalHelperRepos.contains(DownloadManager.nsfwClassifierRepo))
    }

    /// Regression guard for the exact filter `discoverLocalModels` applies to
    /// its scan results: the classifier must never survive it, while a real
    /// downloaded model passes through untouched.
    func testInternalHelperFilterHidesOnlyTheClassifier() {
        let classifier = LocalModel(
            id: "mlxServe:\(DownloadManager.nsfwClassifierRepo)", name: DownloadManager.nsfwClassifierRepo,
            path: "/tmp/classifier", sizeFormatted: "1 GB", modelType: "vit", source: .mlxServe, kind: .base
        )
        let realModel = LocalModel(
            id: "mlxServe:mlx-community/gemma-4-e4b-it-4bit", name: "mlx-community/gemma-4-e4b-it-4bit",
            path: "/tmp/gemma", sizeFormatted: "5 GB", modelType: "gemma4", source: .mlxServe, kind: .base
        )
        let filtered = [classifier, realModel].filter { !DownloadManager.internalHelperRepos.contains($0.name) }
        XCTAssertEqual(filtered.map(\.name), [realModel.name])
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

    func testMediaModelTypesAreSupportedArchitectures() {
        // Native media-gen checkpoints (3D shape+paint, image, video, TTS,
        // music) are real, loadable models — just not chat models. They must
        // NOT be flagged "Unsupported architecture" in the Downloaded tab.
        // Mirrors `model_discovery.isMediaModelType` (Zig).
        for mt in ["hunyuan3d_2_1", "hunyuan3d_2_1_paint", "krea2_turbo", "AudioVideo", "qwen3_tts", "flux2-klein-4b", "acestep"] {
            let m = LocalModel(
                id: "test:\(mt)", name: mt, path: "/tmp/\(mt)",
                sizeFormatted: "1 GB", modelType: mt, source: .custom, kind: .base
            )
            XCTAssertTrue(m.isSupportedArchitecture, "\"\(mt)\" must be a recognized media model type")
        }
    }

    func testGemma3TextIsSupportedArchitecture() {
        // Text-only Gemma 3 (Gemma3ForCausalLM) ships model_type "gemma3_text"
        // — e.g. mlx-community/gemma-3-12b-it-qat-abliterated-lm-4bit. A locally
        // discovered checkpoint must NOT be flagged "Unsupported architecture"
        // in the model manager (mirrors the gemma4_text / qwen3_moe_text tags).
        for mt in ["gemma3", "gemma3_text"] {
            let m = LocalModel(
                id: "test:\(mt)", name: mt, path: "/tmp/gemma-3-12b-it-qat-abliterated-lm-4bit",
                sizeFormatted: "7 GB", modelType: mt, source: .custom, kind: .base
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

    // MARK: - Live-refresh trigger
    //
    // "Size on Disk" comes from a disk re-scan (`refreshModels`), which
    // originally only fired on tab entry — so sizes froze mid-download until the
    // user toggled the button. The panes that show on-disk state now live-poll,
    // but only while one of them is showing AND a download is in flight.
    // Full coverage of the section matrix lives in `ModelBrowserSectionTests`.

    func testShouldLivePollOnlyOnDiskPanesAndOnlyWhileDownloading() {
        XCTAssertTrue(ModelBrowserSection.shouldLivePoll(section: .myModels, hasActiveDownloads: true))
        // No active downloads → nothing to refresh, don't spin.
        XCTAssertFalse(ModelBrowserSection.shouldLivePoll(section: .myModels, hasActiveDownloads: false))
        // Discover reads published DownloadManager state, not the disk.
        XCTAssertFalse(ModelBrowserSection.shouldLivePoll(section: .discover, hasActiveDownloads: true))
        XCTAssertFalse(ModelBrowserSection.shouldLivePoll(section: .discover, hasActiveDownloads: false))
    }

    // MARK: - LocalModel metadata caption
    //
    // The Downloaded tab used to render only a name + delete button. These pin
    // the parsed metadata (params / quant / architecture / engine) that now
    // makes each row actually informative.

    private func localModel(
        name: String, type: String, path: String,
        vision: Bool = false, quantBits: Int? = nil, ctx: Int? = nil,
        numExperts: Int? = nil, activeExperts: Int? = nil
    ) -> LocalModel {
        LocalModel(id: "mlxServe:\(name)", name: name, path: path,
                   sizeFormatted: "10 GB", modelType: type, source: .mlxServe, kind: .base,
                   hasVision: vision, quantBits: quantBits, contextLength: ctx,
                   numExperts: numExperts, activeExperts: activeExperts)
    }

    // Captions below use the real config values dumped from the user's models:
    // Qwen2.5-Coder-32B (qwen2, bits 8, ctx 32768, dense),
    // Qwen3-Coder-30B-A3B (qwen3_moe, bits 8, ctx 262144, 128/8 experts),
    // Qwen3-Coder-Next (qwen3_next, bits 4, ctx 262144, 512/10 experts).

    func testMetadataSummaryDenseFromConfig() {
        let m = localModel(name: "Qwen2.5-Coder-32B-Instruct-8bit", type: "qwen2",
                           path: "/m/Qwen2.5-Coder-32B-Instruct-8bit",
                           quantBits: 8, ctx: 32768)
        XCTAssertEqual(m.paramSize, "32B")
        XCTAssertNil(m.expertSummary, "dense model → no expert token")
        // Format reads "MLX" (the weight format), not the "MLX-Serve" app name.
        XCTAssertEqual(m.metadataSummary, "32B · 8-bit · 32K ctx · qwen2 · MLX")
        XCTAssertTrue(m.hasToolCalling)
        XCTAssertFalse(m.hasVision)
    }

    func testMetadataSummaryMoEShowsConfigExperts() {
        let m = localModel(name: "Qwen3-Coder-30B-A3B-Instruct-8bit", type: "qwen3_moe",
                           path: "/m/Qwen3-Coder-30B-A3B-Instruct-8bit",
                           quantBits: 8, ctx: 262144, numExperts: 128, activeExperts: 8)
        XCTAssertEqual(m.expertSummary, "8/128 experts")
        XCTAssertEqual(m.metadataSummary, "30B · 8-bit · 8/128 experts · 256K ctx · qwen3_moe · MLX")
    }

    func testMetadataSummaryConfigQuantOverridesNameAndNoParamToken() {
        // Name says nothing about params; config says bits 4. Caption must use
        // the config bits and skip the missing param token.
        let n = localModel(name: "Qwen3-Coder-Next-4bit", type: "qwen3_next",
                           path: "/m/Qwen3-Coder-Next-4bit",
                           quantBits: 4, ctx: 262144, numExperts: 512, activeExperts: 10)
        XCTAssertNil(n.paramSize)
        XCTAssertEqual(n.quantization, "4-bit")
        XCTAssertEqual(n.metadataSummary, "4-bit · 10/512 experts · 256K ctx · qwen3_next · MLX")
    }

    func testMetadataSummaryGgufFallsBackToNameQuant() {
        // GGUF has no config.json here → quant comes from the name, format = GGUF.
        let g = LocalModel(id: "mlxServe:team/m", name: "Meta-Llama-3.1-8B-Instruct-Q4_K_M",
                           path: "/m/team/m/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                           sizeFormatted: "5 GB", modelType: "gguf", source: .mlxServe, kind: .base)
        XCTAssertEqual(g.paramSize, "8B")
        XCTAssertEqual(g.quantization, "4-bit")   // Q4_K_M → 4-bit, via name fallback
        XCTAssertEqual(g.metadataSummary, "8B · 4-bit · gguf · GGUF")
        XCTAssertTrue(g.hasToolCalling)
    }

    func testContextFormatting() {
        XCTAssertEqual(LocalModel.formatContext(262144), "256K ctx")
        XCTAssertEqual(LocalModel.formatContext(32768), "32K ctx")
        XCTAssertEqual(LocalModel.formatContext(1048576), "1M ctx")
        XCTAssertEqual(LocalModel.formatContext(512), "512 ctx")
    }

    // MARK: - config.json parsing (the authoritative metadata source)

    func testParseConfigMetadataReadsQuantCtxExpertsVision() throws {
        let dir = (tempRoot as NSString).appendingPathComponent("cfg-moe")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let cfg = (dir as NSString).appendingPathComponent("config.json")
        try """
        {"model_type":"qwen3_moe","quantization":{"group_size":64,"bits":8},
         "max_position_embeddings":262144,"num_experts":128,"num_experts_per_tok":8}
        """.write(toFile: cfg, atomically: true, encoding: .utf8)

        let meta = DownloadManager.parseConfigMetadata(atPath: cfg)
        XCTAssertEqual(meta.modelType, "qwen3_moe")
        XCTAssertEqual(meta.quantBits, 8)
        XCTAssertEqual(meta.contextLength, 262144)
        XCTAssertEqual(meta.numExperts, 128)
        XCTAssertEqual(meta.activeExperts, 8)
        XCTAssertFalse(meta.hasVision)
    }

    func testParseConfigMetadataQuantizationConfigAndVision() throws {
        // Some checkpoints use `quantization_config`; vision via `vision_config`.
        let dir = (tempRoot as NSString).appendingPathComponent("cfg-vision")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let cfg = (dir as NSString).appendingPathComponent("config.json")
        try """
        {"model_type":"gemma4","quantization_config":{"bits":4},"vision_config":{"hidden_size":1152}}
        """.write(toFile: cfg, atomically: true, encoding: .utf8)

        let meta = DownloadManager.parseConfigMetadata(atPath: cfg)
        XCTAssertEqual(meta.quantBits, 4)
        XCTAssertTrue(meta.hasVision)
    }

    func testParseConfigMetadataTextOnlyArchSuppressesVision() throws {
        // A `_text` arch with a vestigial vision_config must NOT report vision.
        let dir = (tempRoot as NSString).appendingPathComponent("cfg-text")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let cfg = (dir as NSString).appendingPathComponent("config.json")
        try #"{"model_type":"qwen3_5_moe_text","vision_config":{"x":1}}"#
            .write(toFile: cfg, atomically: true, encoding: .utf8)

        XCTAssertFalse(DownloadManager.parseConfigMetadata(atPath: cfg).hasVision)
    }

    func testParseConfigMetadataMissingFileDefaults() {
        let meta = DownloadManager.parseConfigMetadata(atPath: "/nope/config.json")
        XCTAssertEqual(meta, DownloadManager.ConfigMetadata())
    }

    // MARK: - Hugging Face hub cache discovery
    //
    // The huggingface_hub / mlx_lm default cache lays each repo out as
    // `models--<org>--<repo>/snapshots/<commit>/`, whose files are SYMLINKS into
    // a sibling `blobs/` dir; `refs/main` names the active commit. This is a
    // different shape from LM Studio's plain `<org>/<repo>/`, so it gets its own
    // scan helpers — these pin them.

    func testHuggingFaceRepoIdFromCacheDirRecoversOrgAndName() {
        // HF replaces `/` with `--`; the repo NAME itself may carry single dashes.
        XCTAssertEqual(
            DownloadManager.huggingFaceRepoId(fromCacheDir: "models--Jundot--Qwen3.6-35B-A3B-oQ4e-mtp"),
            "Jundot/Qwen3.6-35B-A3B-oQ4e-mtp")
        XCTAssertEqual(
            DownloadManager.huggingFaceRepoId(fromCacheDir: "models--gpt2"),
            "gpt2", "a bare repo with no org stays a single component")
    }

    func testHuggingFaceActiveSnapshotResolvesRefMain() throws {
        let commit = "2523e7a5702d38a1a319c50d193cbac20f9ecb78"
        let snap = try makeFakeHFRepo(root: tempRoot, org: "acme", repo: "demo",
                                      commit: commit, files: ["config.json": "{}"])
        let repoDir = (tempRoot as NSString).appendingPathComponent("models--acme--demo")
        XCTAssertEqual(DownloadManager.huggingFaceActiveSnapshotDir(repoDir: repoDir), snap,
                       "refs/main must select the matching snapshot dir")
    }

    func testHuggingFaceActiveSnapshotFallsBackToSoleSnapshot() throws {
        // No refs/main (writeRef: false) but exactly one snapshot → use it.
        let snap = try makeFakeHFRepo(root: tempRoot, org: "acme", repo: "solo",
                                      commit: "abc123", files: ["config.json": "{}"],
                                      writeRef: false)
        let repoDir = (tempRoot as NSString).appendingPathComponent("models--acme--solo")
        XCTAssertEqual(DownloadManager.huggingFaceActiveSnapshotDir(repoDir: repoDir), snap)
    }

    func testHuggingFaceActiveSnapshotNilWhenAmbiguousAndNoRef() throws {
        // Two snapshots and no ref to disambiguate → refuse to guess.
        _ = try makeFakeHFRepo(root: tempRoot, org: "acme", repo: "multi",
                               commit: "aaa", files: ["config.json": "{}"], writeRef: false)
        _ = try makeFakeHFRepo(root: tempRoot, org: "acme", repo: "multi",
                               commit: "bbb", files: ["config.json": "{}"], writeRef: false)
        let repoDir = (tempRoot as NSString).appendingPathComponent("models--acme--multi")
        XCTAssertNil(DownloadManager.huggingFaceActiveSnapshotDir(repoDir: repoDir))
    }

    func testDiscoverHuggingFaceModelsFindsCompleteSnapshotAsHFSource() throws {
        let commit = "cafebabe"
        // 2 MiB so it formats to a non-zero "MB" (MemoryInfo.format truncates to
        // whole MB below 1 GB) — the point is that the symlinked blob's real size
        // is counted, not the ~20 B link size that would read as "0 MB".
        let bigWeights = String(repeating: "x", count: 2 * 1024 * 1024)
        let snap = try makeFakeHFRepo(
            root: tempRoot, org: "mlx-community", repo: "Qwen3-Demo-4bit", commit: commit,
            files: ["config.json": "{\"model_type\":\"qwen3\"}",
                    "model.safetensors": bigWeights,
                    "tokenizer.json": "{}"])

        let models = DownloadManager.discoverHuggingFaceModels(in: tempRoot)
        XCTAssertEqual(models.count, 1)
        let m = try XCTUnwrap(models.first)
        XCTAssertEqual(m.source, .huggingFace)
        XCTAssertEqual(m.name, "mlx-community/Qwen3-Demo-4bit", "display name is the recovered repo id")
        XCTAssertEqual(m.modelType, "qwen3")
        // Path is the SNAPSHOT dir the server loads (standardized, so /private-normalized).
        XCTAssertEqual((m.path as NSString).standardizingPath, (snap as NSString).standardizingPath)
        // Size must resolve through the symlinks — the weight blob is 4 KB, so a
        // link-size read (~20 B → "Zero KB") is the regression this guards.
        XCTAssertNotEqual(m.sizeFormatted, MemoryInfo.format(0))
    }

    func testDiscoverHuggingFaceModelsFindsSymlinkedGgufQuant() throws {
        // A complete GGUF in the HF cache is a SYMLINK into blobs/. The ≥1 MB
        // "servable quant" filter stats the file, so it must follow the link —
        // otherwise the ~76 B link size fails the filter and EVERY HF-cached
        // GGUF (bartowski, Hy3-GGUF, …) is silently invisible.
        let bigGguf = String(repeating: "g", count: 2 * 1024 * 1024)
        _ = try makeFakeHFRepo(root: tempRoot, org: "bartowski", repo: "Demo-GGUF",
                               commit: "feedface", files: ["Demo-Q4_K_M.gguf": bigGguf])
        let models = DownloadManager.discoverHuggingFaceModels(in: tempRoot)
        let m = try XCTUnwrap(models.first { $0.name == "bartowski/Demo-GGUF" })
        XCTAssertEqual(m.source, .huggingFace)
        XCTAssertNotNil(m.quantFile, "a GGUF row carries its quant filename")
        XCTAssertNotEqual(m.sizeFormatted, MemoryInfo.format(0), "blob size, not link size")
    }

    func testDiscoverHuggingFaceModelsDropsSubOneMegabyteGgufStub() throws {
        // The incomplete/pointer case (a 76 B LFS stub, real in the live cache):
        // following the link reveals it's tiny, so it's correctly not offered.
        _ = try makeFakeHFRepo(root: tempRoot, org: "acme", repo: "Stub-GGUF",
                               commit: "0bad", files: ["Stub-Q4_K_M.gguf": "tiny"])
        XCTAssertTrue(DownloadManager.discoverHuggingFaceModels(in: tempRoot).isEmpty)
    }

    func testDiscoverHuggingFaceModelsSkipsIncompleteSnapshot() throws {
        // A snapshot with only a README (no config.json / no safetensors / no
        // gguf) is a partial or metadata-only pull — not loadable, must be dropped.
        _ = try makeFakeHFRepo(root: tempRoot, org: "acme", repo: "readme-only",
                               commit: "deadbeef", files: ["README.md": "hi"])
        XCTAssertTrue(DownloadManager.discoverHuggingFaceModels(in: tempRoot).isEmpty)
    }

    func testDiscoverHuggingFaceModelsIgnoresNonModelPrefixDirs() throws {
        // datasets--/spaces-- cache dirs share the hub root but aren't models.
        let datasets = ((tempRoot as NSString).appendingPathComponent("datasets--acme--corpus") as NSString)
            .appendingPathComponent("snapshots")
        try FileManager.default.createDirectory(atPath: datasets, withIntermediateDirectories: true)
        XCTAssertTrue(DownloadManager.discoverHuggingFaceModels(in: tempRoot).isEmpty)
    }

    // Size accounting must follow symlinks so an HF snapshot (all symlinked
    // blobs) doesn't report ~0 B. Real files are unaffected (no-op resolve).
    func testDirectorySizeResolvesSymlinkedBlobs() throws {
        let blobs = (tempRoot as NSString).appendingPathComponent("blobs")
        let snap = (tempRoot as NSString).appendingPathComponent("snap")
        try FileManager.default.createDirectory(atPath: blobs, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(atPath: snap, withIntermediateDirectories: true)
        let blob = (blobs as NSString).appendingPathComponent("weights")
        try String(repeating: "y", count: 2000).write(toFile: blob, atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(
            atPath: (snap as NSString).appendingPathComponent("model.safetensors"),
            withDestinationPath: "../blobs/weights")

        XCTAssertGreaterThanOrEqual(DownloadManager.directorySizeForTesting(snap), 2000,
            "symlinked blob size must be counted, not the ~20 B link size")
    }

    // MARK: - Helpers

    /// Build a Hugging Face hub-cache repo: `models--<org>--<repo>/` with a
    /// `blobs/` dir, a `snapshots/<commit>/` whose files are RELATIVE symlinks
    /// into blobs (exactly how huggingface_hub materializes them), and — unless
    /// `writeRef` is false — `refs/main` → commit. Returns the snapshot dir path.
    @discardableResult
    private func makeFakeHFRepo(root: String, org: String, repo: String, commit: String,
                               files: [String: String], writeRef: Bool = true) throws -> String {
        let fm = FileManager.default
        let repoDir = (root as NSString).appendingPathComponent("models--\(org)--\(repo)")
        let blobs = (repoDir as NSString).appendingPathComponent("blobs")
        let snap = ((repoDir as NSString).appendingPathComponent("snapshots") as NSString)
            .appendingPathComponent(commit)
        try fm.createDirectory(atPath: blobs, withIntermediateDirectories: true)
        try fm.createDirectory(atPath: snap, withIntermediateDirectories: true)
        for (name, contents) in files {
            let blobName = "blob-\(name)"
            try contents.write(toFile: (blobs as NSString).appendingPathComponent(blobName),
                               atomically: true, encoding: .utf8)
            try fm.createSymbolicLink(atPath: (snap as NSString).appendingPathComponent(name),
                                      withDestinationPath: "../../blobs/\(blobName)")
        }
        if writeRef {
            let refsDir = (repoDir as NSString).appendingPathComponent("refs")
            try fm.createDirectory(atPath: refsDir, withIntermediateDirectories: true)
            try commit.write(toFile: (refsDir as NSString).appendingPathComponent("main"),
                             atomically: true, encoding: .utf8)
        }
        return snap
    }

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
