import XCTest
@testable import MLXCore

/// A multi-variant MLX repo ships one COMPLETE model per subfolder — the same
/// "a repo is a shelf, not a model" shape a GGUF repo has, expressed as
/// directories instead of files (`LiquidAI/LFM2.5-2.6B-MLX`: 4bit, 5bit, 6bit,
/// 8bit, bf16, mxfp4, mxfp8, nvfp4).
///
/// Before this the browser had no idea: the row offered a plain "Download",
/// `selectNeededFiles`' top-level-only filter matched nothing under a
/// subfolder, and the click created an EMPTY `~/.mlx-serve/models/LiquidAI/
/// LFM2.5-2.6B-MLX/` and reported success.
final class MlxVariantScanTests: XCTestCase {

    private func entry(_ path: String, _ size: Int64) -> HFSearchService.TreeFileEntry {
        HFSearchService.TreeFileEntry(path: path, size: size)
    }

    /// The real repo's shape, trimmed to three quants.
    private var liquidTree: [HFSearchService.TreeFileEntry] {
        var out: [HFSearchService.TreeFileEntry] = [
            entry(".gitattributes", 1970),
            entry("LICENSE", 10574),
            entry("README.md", 2804),
        ]
        for (folder, weights) in [("4bit", Int64(1_583_152_892)), ("mxfp4", 1_564_390_336)] {
            out += [
                entry("\(folder)/config.json", 2202),
                entry("\(folder)/chat_template.jinja", 5443),
                entry("\(folder)/generation_config.json", 307),
                entry("\(folder)/model.safetensors", weights),
                entry("\(folder)/model.safetensors.index.json", 41983),
                entry("\(folder)/tokenizer.json", 17_905_598),
                entry("\(folder)/tokenizer_config.json", 363),
            ]
        }
        // bf16 is sharded — both shards count toward its size.
        out += [
            entry("bf16/config.json", 1747),
            entry("bf16/chat_template.jinja", 5443),
            entry("bf16/model-00001-of-00002.safetensors", 5_350_374_478),
            entry("bf16/model-00002-of-00002.safetensors", 44_052_910),
            entry("bf16/model.safetensors.index.json", 22521),
            entry("bf16/tokenizer.json", 17_905_598),
            entry("bf16/tokenizer_config.json", 363),
        ]
        return out
    }

    func testFindsEveryQuantSubfolderWithItsOwnLabelAndSize() {
        let variants = MlxVariantScan.variants(files: liquidTree)

        // Smallest first — the menu is a "which one fits my Mac" question.
        XCTAssertEqual(variants.map(\.folder), ["mxfp4", "4bit", "bf16"])
        XCTAssertEqual(variants.map(\.label), ["MXFP4", "4-bit", "BF16"])

        // Size is the folder's OWN weights, never the repo total: summing all
        // eight quants would report ~20 GB for a 2.6B model.
        XCTAssertEqual(variants.first { $0.folder == "4bit" }?.sizeBytes, 1_583_152_892)
        XCTAssertEqual(variants.first { $0.folder == "bf16" }?.sizeBytes, 5_350_374_478 + 44_052_910)
    }

    /// The gate that keeps every normal repo out: a root `config.json` means the
    /// repo IS one model, and a nested `mtp/` or `original/` folder must not
    /// turn it into a picker.
    func testARepoWithARootConfigIsOneModelNotAShelf() {
        let files = [
            entry("config.json", 2202),
            entry("tokenizer.json", 17_905_598),
            entry("model.safetensors", 1_583_152_892),
            entry("mtp/weights.safetensors", 524_000_000),
            entry("original/config.json", 2202),
            entry("original/tokenizer.json", 17_905_598),
            entry("original/model.safetensors", 5_000_000_000),
        ]
        XCTAssertEqual(MlxVariantScan.variants(files: files), [])
    }

    /// A diffusers repo (MageFlow) keeps a `config.json` in every component
    /// subdir — `transformer/`, `text_encoder/`, `vae/` are PARTS of one model,
    /// not alternatives. `model_index.json` at the root says so.
    func testADiffusersRepoIsNotAVariantShelf() {
        let files = [
            entry("model_index.json", 800),
            entry("transformer/config.json", 1200),
            entry("transformer/diffusion_pytorch_model.safetensors", 8_000_000_000),
            entry("text_encoder/config.json", 1000),
            entry("text_encoder/model.safetensors", 4_000_000_000),
            entry("tokenizer/tokenizer.json", 17_000_000),
            entry("tokenizer/tokenizer_config.json", 400),
        ]
        XCTAssertEqual(MlxVariantScan.variants(files: files), [])
    }

    /// `mlx-community/flux2-klein-9b-4bit` ships NO root json at all and four
    /// weight subdirs — the exact shape a naive "subfolders with safetensors"
    /// rule would read as four quants of one model. Its component dirs carry no
    /// `config.json` and its tokenizer dir carries no weights.
    func testKleinStyleConfiglessWeightSubdirsAreNotVariants() {
        let files = [
            entry("README.md", 5000),
            entry("text_encoder/0.safetensors", 4_000_000_000),
            entry("text_encoder/model.safetensors.index.json", 30000),
            entry("tokenizer/tokenizer.json", 17_000_000),
            entry("tokenizer/tokenizer_config.json", 400),
            entry("transformer/0.safetensors", 5_000_000_000),
            entry("vae/0.safetensors", 300_000_000),
        ]
        XCTAssertEqual(MlxVariantScan.variants(files: files), [])
    }

    /// A variant has to be loadable ON ITS OWN — config + weights + tokenizer.
    /// A subfolder missing any of the three is a component, not a choice.
    func testASubfolderMissingATokenizerIsNotAStandaloneModel() {
        let files = [
            entry("4bit/config.json", 2202),
            entry("4bit/model.safetensors", 1_583_152_892),
            entry("4bit/tokenizer.json", 17_905_598),
            entry("8bit/config.json", 1952),
            entry("8bit/model.safetensors", 2_866_086_056),
        ]
        XCTAssertEqual(MlxVariantScan.variants(files: files).map(\.folder), ["4bit"])
    }

    /// Only IMMEDIATE subfolders are variants — a model nested two levels deep
    /// would land somewhere the layout can't express.
    func testNestedSubfoldersAreNotVariants() {
        let files = [
            entry("quants/4bit/config.json", 2202),
            entry("quants/4bit/model.safetensors", 1_583_152_892),
            entry("quants/4bit/tokenizer.json", 17_905_598),
        ]
        XCTAssertEqual(MlxVariantScan.variants(files: files), [])
    }

    /// The layout decision, pinned: a variant is downloaded as its OWN 2-level
    /// model dir (`<org>/<repo>-<variant>`), never nested under the repo.
    /// `model_discovery.discoverModelsInDir` walks exactly two levels
    /// (`<root>/<org>/<model>`), so `<org>/<repo>/4bit` would be invisible to
    /// `list`, `/v1/models` and the tray picker — downloaded and unloadable.
    func testLocalRepoIdStaysTwoLevelSoServerDiscoveryFindsIt() {
        let id = MlxVariantScan.localRepoId(repoId: "LiquidAI/LFM2.5-2.6B-MLX", folder: "4bit")
        XCTAssertEqual(id, "LiquidAI/LFM2.5-2.6B-MLX-4bit")
        XCTAssertEqual(id.split(separator: "/").count, 2)

        // Distinct per variant — two quants must never resolve to one folder.
        let ids = ["4bit", "8bit", "bf16"].map {
            MlxVariantScan.localRepoId(repoId: "LiquidAI/LFM2.5-2.6B-MLX", folder: $0)
        }
        XCTAssertEqual(Set(ids).count, 3)
    }
}

final class MlxVariantMenuModelTests: XCTestCase {

    private let remote = [
        MlxVariant(folder: "4bit", label: "4-bit", sizeBytes: 1_583_152_892),
        MlxVariant(folder: "8bit", label: "8-bit", sizeBytes: 2_866_086_056),
        MlxVariant(folder: "bf16", label: "BF16", sizeBytes: 5_394_427_388),
    ]

    func testBuildSplitsWhatYouHaveFromWhatYouDont() {
        let menu = MlxVariantMenuModel.build(remote: remote, onDisk: ["8bit"])
        XCTAssertEqual(menu.onDisk.map(\.folder), ["8bit"])
        XCTAssertEqual(menu.available.map(\.folder), ["4bit", "bf16"])
    }

    /// Same rule as the GGUF menu: owning a quant the repo no longer publishes
    /// still lists it, or it would be stranded with no way to use or delete it.
    func testAnOnDiskVariantTheRepoNoLongerPublishesIsStillListed() {
        let menu = MlxVariantMenuModel.build(remote: remote, onDisk: ["3bit"])
        XCTAssertEqual(menu.onDisk.map(\.folder), ["3bit"])
        XCTAssertEqual(menu.available.count, 3)
    }

    func testButtonLabelReportsWhatYouOwnOverTheLastClick() {
        XCTAssertEqual(MlxVariantMenuModel.buttonLabel(onDisk: [], failed: false, hasPartial: false), "Download")
        XCTAssertEqual(MlxVariantMenuModel.buttonLabel(onDisk: [], failed: false, hasPartial: true), "Resume")
        XCTAssertEqual(MlxVariantMenuModel.buttonLabel(onDisk: [], failed: true, hasPartial: false), "Retry")
        XCTAssertEqual(MlxVariantMenuModel.buttonLabel(onDisk: [remote[0]], failed: false, hasPartial: false), "✓ 4-bit")
        // Owning something outranks a failed transfer of a DIFFERENT variant.
        XCTAssertEqual(MlxVariantMenuModel.buttonLabel(onDisk: Array(remote.prefix(2)), failed: true, hasPartial: false),
                       "✓ 2 on disk")
    }
}

final class MlxVariantRepoRowTests: XCTestCase {

    private func model(variants: [MlxVariant]) -> HFModel {
        var m = HFModel(id: "LiquidAI/LFM2.5-2.6B-MLX", downloads: 0, likes: 18,
                        lastModified: nil, tags: ["mlx", "safetensors"], safetensors: nil,
                        pipelineTag: "text-generation")
        m.mlxVariants = variants
        return m
    }

    func testAVariantRepoBadgesMultiAndSpansItsQuantsInTheRamColumn() {
        let m = model(variants: [
            MlxVariant(folder: "4bit", label: "4-bit", sizeBytes: 1_583_152_892),
            MlxVariant(folder: "bf16", label: "BF16", sizeBytes: 5_394_427_388),
        ])
        XCTAssertTrue(m.isMlxVariantRepo)
        // The id names no single quant — same answer a multi-quant GGUF gives.
        XCTAssertEqual(m.quantization, "Multi")
        XCTAssertTrue(m.ramEstimate.contains("–"), "expected a range, got \(m.ramEstimate)")
        // Sort/fitness need one number: the largest, so a row can't colour green
        // on its smallest quant.
        XCTAssertEqual(m.ramEstimateBytes, Int64(Double(5_394_427_388) * 1.2))
    }

    func testAnOrdinaryRepoIsUnaffected() {
        let m = model(variants: [])
        XCTAssertFalse(m.isMlxVariantRepo)
        XCTAssertNil(m.quantization)
    }
}

final class MlxVariantDownloadSelectionTests: XCTestCase {

    private let tree: [[String: Any]] = [
        ["path": "README.md", "type": "file", "size": 2804],
        ["path": "4bit", "type": "directory", "size": 0],
        ["path": "4bit/config.json", "type": "file", "size": 2202],
        ["path": "4bit/chat_template.jinja", "type": "file", "size": 5443],
        ["path": "4bit/model.safetensors", "type": "file", "size": 1_583_152_892],
        ["path": "4bit/tokenizer.json", "type": "file", "size": 17_905_598],
        ["path": "8bit/config.json", "type": "file", "size": 1952],
        ["path": "8bit/model.safetensors", "type": "file", "size": 2_866_086_056],
        ["path": "bf16/model-00001-of-00002.safetensors", "type": "file", "size": 5_350_374_478],
    ]

    func testAVariantDownloadPullsThatFolderAndNothingElse() {
        let picked = DownloadManager.selectNeededFiles(from: tree, selection: .mlxVariant("4bit"))
        XCTAssertEqual(Set(picked.map { $0.0 }), [
            "4bit/config.json", "4bit/chat_template.jinja",
            "4bit/model.safetensors", "4bit/tokenizer.json",
        ])
        XCTAssertEqual(picked.first { $0.0 == "4bit/model.safetensors" }?.1, 1_583_152_892)
    }

    /// The variant dir must be a COMPLETE model at its own root, so the
    /// subfolder prefix is stripped on the way to disk. Without this the files
    /// land at `<repo>-4bit/4bit/config.json` and nothing can load them.
    func testVariantFilesLoseTheirSubfolderPrefixOnDisk() {
        let sel = FileSelection.mlxVariant("4bit")
        XCTAssertEqual(sel.localPath(forRemote: "4bit/config.json"), "config.json")
        XCTAssertEqual(sel.localPath(forRemote: "4bit/model.safetensors"), "model.safetensors")
        // Every other selection is pass-through — the mtp sidecar keeps its dir.
        XCTAssertEqual(FileSelection.chatDefault.localPath(forRemote: "mtp/weights.safetensors"),
                       "mtp/weights.safetensors")
    }

    /// The live bug: with nothing loadable at the repo root, the whole-repo path
    /// selected zero files, created an empty
    /// `~/.mlx-serve/models/LiquidAI/LFM2.5-2.6B-MLX/` and reported success.
    func testAShelfRepoHasNothingToDownloadAtItsRoot() {
        XCTAssertTrue(DownloadManager.selectNeededFiles(from: tree).isEmpty)
    }

    /// The tree fetch the search already runs for a size must not report the
    /// SUM of every quant (~20 GB for a 2.6B model) — it reports the variants.
    func testFallbackSizeReportsVariantsInsteadOfSummingEveryQuant() {
        let files = tree.compactMap { e -> HFSearchService.TreeFileEntry? in
            guard (e["type"] as? String) == "file", let p = e["path"] as? String,
                  let s = e["size"] as? Int else { return nil }
            return HFSearchService.TreeFileEntry(path: p, size: Int64(s))
        }
        // 4bit is a complete model; 8bit has no tokenizer and bf16 no config,
        // so only the complete one is offered.
        guard case .mlxVariants(let variants)? = HFSearchService.parseFallbackSize(files: files) else {
            return XCTFail("expected .mlxVariants, got \(String(describing: HFSearchService.parseFallbackSize(files: files)))")
        }
        XCTAssertEqual(variants.map(\.folder), ["4bit"])
    }
}

/// Drives the REAL download loop for a variant against the stub HF origin
/// (`HuggingFaceStubProtocol`, shared with `DownloadManagerTransferTests`).
/// The pure tests above prove which files are picked; these prove where they
/// land, that a sibling quant is untouched, and that a shelf repo refuses the
/// whole-repo path instead of leaving an empty folder.
@MainActor
final class MlxVariantDownloadTests: XCTestCase {
    private var tempRoot: String!
    private var savedSession: URLSession!

    private let repoId = "acme/shelf-model"

    override func setUpWithError() throws {
        tempRoot = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mlx-serve-variant-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: tempRoot, withIntermediateDirectories: true)
        savedSession = DownloadSession.shared
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [HuggingFaceStubProtocol.self]
        DownloadSession.shared = URLSession(configuration: config)
        HuggingFaceStubProtocol.serve(repo: repoId, files: Self.shelfFiles())
    }

    override func tearDownWithError() throws {
        DownloadSession.shared = savedSession
        HuggingFaceStubProtocol.reset()
        try? FileManager.default.removeItem(atPath: tempRoot)
    }

    /// Two complete models, one per subfolder — the LiquidAI shape at test size.
    private static func shelfFiles() -> [(String, Data)] {
        var out: [(String, Data)] = [("README.md", Data("shelf".utf8))]
        for (folder, byte) in [("4bit", UInt8(1)), ("8bit", UInt8(2))] {
            out += [
                ("\(folder)/config.json", Data(#"{"model_type":"lfm2"}"#.utf8)),
                ("\(folder)/tokenizer.json", Data(#"{"version":"1.0"}"#.utf8)),
                ("\(folder)/model.safetensors", Data(repeating: byte, count: 2 << 20)),
            ]
        }
        return out
    }

    private func variant(_ folder: String) -> MlxVariant {
        MlxVariant(folder: folder, label: MlxVariantScan.label(forFolder: folder), sizeBytes: 2 << 20)
    }

    func testOneVariantLandsAsACompleteModelInItsOwnTwoLevelDir() async throws {
        let manager = DownloadManager(modelsRoot: tempRoot)
        let dest = MlxVariantScan.localRepoId(repoId: repoId, folder: "4bit")

        await manager.download(repoId: repoId, selection: .mlxVariant("4bit"), destRepoId: dest)

        XCTAssertEqual(manager.downloads[repoId]?.status, .completed,
                       "error: \(manager.downloads[repoId]?.error ?? "-")")
        let dir = DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: dest)
        // Files sit at the variant dir's ROOT — a nested `4bit/config.json`
        // would be a model neither the server nor `isReady` can find.
        for name in ["config.json", "tokenizer.json", "model.safetensors"] {
            XCTAssertTrue(FileManager.default.fileExists(atPath: (dir as NSString).appendingPathComponent(name)), name)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: (dir as NSString).appendingPathComponent("4bit")))
        XCTAssertTrue(manager.isReady(dest))

        // The other quant was never fetched, and the repo's own dir is not a model.
        let fetched = HuggingFaceStubProtocol.requests.map(\.path)
        XCTAssertFalse(fetched.contains { $0.contains("8bit/") }, "pulled the wrong quant: \(fetched)")
        XCTAssertFalse(manager.isReady(repoId))
    }

    func testCancellingOneVariantLeavesItsSiblingOnDisk() async throws {
        let manager = DownloadManager(modelsRoot: tempRoot)
        let kept = MlxVariantScan.localRepoId(repoId: repoId, folder: "4bit")
        await manager.download(repoId: repoId, selection: .mlxVariant("4bit"), destRepoId: kept)
        XCTAssertTrue(manager.isReady(kept))

        HuggingFaceStubProtocol.serve(repo: repoId, files: Self.shelfFiles(), throttle: true)
        let finished = expectation(description: "second variant settled")
        manager.startMlxVariant(repoId: repoId, variant: variant("8bit")) { finished.fulfill() }
        try await Task.sleep(nanoseconds: 150_000_000)
        manager.cancel(repoId)
        await fulfillment(of: [finished], timeout: 30)

        let cancelled = MlxVariantScan.localRepoId(repoId: repoId, folder: "8bit")
        XCTAssertFalse(FileManager.default.fileExists(atPath: DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: cancelled)),
                       "a cancelled variant must leave zero footprint")
        XCTAssertTrue(manager.isReady(kept), "cancelling one quant must not delete another")
        XCTAssertNil(manager.downloads[repoId])
    }

    func testTheWholeRepoPathRefusesInsteadOfLeavingAnEmptyFolder() async throws {
        let manager = DownloadManager(modelsRoot: tempRoot)

        await manager.download(repoId: repoId, alertOnFailure: false)

        XCTAssertEqual(manager.downloads[repoId]?.status, .failed)
        XCTAssertEqual(manager.downloads[repoId]?.error?.contains("subfolders"), true,
                       "the message must say what to do: \(manager.downloads[repoId]?.error ?? "-")")
        XCTAssertFalse(FileManager.default.fileExists(atPath: DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: repoId)),
                       "a refused download must not leave a directory that reads as a model")
    }
}

/// Live legs against the real `LiquidAI/LFM2.5-2.6B-MLX`. Gated like the other
/// network tests — `MLX_SERVE_LIVE_VARIANT=1` runs them, everything else skips.
/// `MLX_SERVE_LIVE_VARIANT_ROOT` names a models root to download into, so the
/// result can be handed to `mlx-serve list --model-dir` and actually served.
@MainActor
final class MlxVariantLiveTests: XCTestCase {
    private let repoId = "LiquidAI/LFM2.5-2.6B-MLX"

    private func requireLive() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLX_SERVE_LIVE_VARIANT"] == "1")
    }

    func testTheRealRepoIsDetectedAsAShelf() async throws {
        try requireLive()
        let url = URL(string: "https://huggingface.co/api/models/\(repoId)/tree/main?recursive=true")!
        let (data, _) = try await DownloadSession.shared.data(for: DownloadManager.hfApiRequest(url))
        let raw = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [[String: Any]])
        let variants = MlxVariantScan.variants(files: HFSearchService.treeEntries(from: raw))

        XCTAssertEqual(variants.map(\.folder).sorted(),
                       ["4bit", "5bit", "6bit", "8bit", "bf16", "mxfp4", "mxfp8", "nvfp4"])
        XCTAssertEqual(variants.first?.folder, "mxfp4", "smallest first")
        // The whole-repo path has nothing to pull — the empty-download bug.
        XCTAssertTrue(DownloadManager.selectNeededFiles(from: raw).isEmpty)
    }

    func testDownloadingOneVariantProducesALoadableModelDir() async throws {
        try requireLive()
        let root = try XCTUnwrap(ProcessInfo.processInfo.environment["MLX_SERVE_LIVE_VARIANT_ROOT"],
                                 "set MLX_SERVE_LIVE_VARIANT_ROOT to a models root")
        let folder = ProcessInfo.processInfo.environment["MLX_SERVE_LIVE_VARIANT_FOLDER"] ?? "mxfp4"
        try FileManager.default.createDirectory(atPath: root, withIntermediateDirectories: true)
        let manager = DownloadManager(modelsRoot: root)
        let dest = MlxVariantScan.localRepoId(repoId: repoId, folder: folder)

        await manager.download(repoId: repoId, selection: .mlxVariant(folder),
                               alertOnFailure: false, destRepoId: dest)

        XCTAssertEqual(manager.downloads[repoId]?.status, .completed,
                       "error: \(manager.downloads[repoId]?.error ?? "-")")
        XCTAssertTrue(manager.isReady(dest))
        print("[live] downloaded to \(DownloadManager.newLayoutDir(rootDir: root, repoId: dest))")
    }
}
