import XCTest
@testable import MLXCore

/// A Kev pack's root config.json says `qwen3_5`; `kev_config.json` is what makes it a
/// decision model, or it is offered as a chat model it can't serve. Server twin:
/// `model_discovery.peekKevPack`.
final class KevModelTests: XCTestCase {

    private func makeKevDir() throws -> (root: String, dir: String) {
        let fm = FileManager.default
        let root = NSTemporaryDirectory() + "kev-\(UUID().uuidString)"
        let dir = (root as NSString).appendingPathComponent("aselea/Kev-4B-MLX-Serve-8bit")
        try fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let files = ["config.json": #"{"model_type": "qwen3_5"}"#, "kev_config.json": #"{"format": "kev"}"#,
                     "tokenizer.json": "{}", "kev_head.safetensors": ""]
        for (name, body) in files {
            fm.createFile(atPath: (dir as NSString).appendingPathComponent(name), contents: Data(body.utf8))
        }
        fm.createFile(atPath: (dir as NSString).appendingPathComponent("model.safetensors"),
                      contents: Data(count: Int(DownloadManager.minimumWeightBytes) + 1))
        return (root, dir)
    }

    func testAKevPackIsListedAsKevNotAsItsQwenTrunk() throws {
        let (root, dir) = try makeKevDir()
        defer { try? FileManager.default.removeItem(atPath: root) }

        let models = DownloadManager.makeLocalModels(
            atDir: dir, displayName: "aselea/Kev-4B-MLX-Serve-8bit",
            idKey: "aselea/Kev-4B-MLX-Serve-8bit", source: .mlxServe)
        let m = try XCTUnwrap(models.first)
        XCTAssertEqual(m.modelType, "kev")
        XCTAssertNil(m.defect)
        XCTAssertTrue(m.isSupportedArchitecture, "must not badge Unsupported")
        XCTAssertFalse(m.isChatPickable, "the trunk has no lm_head to chat with")
        XCTAssertTrue(isDecisionModelType(m.modelType), "gets the Decisions Use button")
    }

    func testAKevDownloadIsReadyOnlyWithEveryShardItsIndexNames() throws {
        let fm = FileManager.default
        let root = NSTemporaryDirectory() + "kev-ready-\(UUID().uuidString)"
        let dir = (root as NSString).appendingPathComponent("aselea/Kev-4B-MLX-Serve")
        try fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(atPath: root) }
        let comp = try XCTUnwrap(CustomMediaModels.bundle(arch: "kev", repoId: "aselea/Kev-4B-MLX-Serve")).components[0]
        let put = { (name: String, body: String) in
            fm.createFile(atPath: (dir as NSString).appendingPathComponent(name), contents: Data(body.utf8))
        }
        let index = #"{"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}"#
        for (name, body) in ["kev_config.json": "{}", "kev_head.safetensors": "x", "config.json": "{}",
                             "tokenizer.json": "{}", "model.safetensors.index.json": index] { put(name, body) }
        XCTAssertFalse(DownloadManager.componentReady(comp, modelsRoot: root), "the head alone is not the pack")
        put("model-00001-of-00002.safetensors", "x")
        XCTAssertFalse(DownloadManager.componentReady(comp, modelsRoot: root), "one shard of two is not the pack")
        put("model-00002-of-00002.safetensors", "x")
        XCTAssertTrue(DownloadManager.componentReady(comp, modelsRoot: root))
    }

    func testASearchRowTaggedKevVerifiesAgainstTheRealRepoTree() throws {
        // What `aselea/Kev-4B-MLX-Serve-8bit` actually ships (tree API, 2026-09).
        let tree = [".gitattributes", "README.md", "chat_template.jinja", "config.json", "kev_config.json",
                    "kev_head.safetensors", "model.safetensors", "model.safetensors.index.json",
                    "tokenizer.json", "tokenizer_config.json"]
            .map { HFSearchService.TreeFileEntry(path: $0, size: 1) }
        let bundle = try XCTUnwrap(CustomMediaModels.bundle(arch: "kev", repoId: "aselea/Kev-4B-MLX-Serve-8bit"))
        let markers = bundle.components[0].readyMarkers
        XCTAssertTrue(HFSearchService.mediaStructureSatisfied(markers: markers, files: tree))
        XCTAssertFalse(HFSearchService.mediaStructureSatisfied(markers: markers, files: tree.filter { $0.path != "kev_head.safetensors" }),
                       "a Qwen trunk without the head is not a Kev pack")

        // Tags as Hugging Face reports them for the published pack.
        let row = HFModel(id: "aselea/Kev-4B-MLX-Serve-8bit", downloads: 1, likes: 0, lastModified: nil,
                          tags: ["mlx-serve", "safetensors", "qwen3_5", "mlx", "kev", "decisions", "text-classification"],
                          safetensors: nil, pipelineTag: "text-classification")
        XCTAssertEqual(row.mediaFamilyModelType, "kev")
        XCTAssertTrue(row.isSupportedArchitecture)
        XCTAssertNil(MediaModality(modelType: "kev"), "no create pane; Use opens Decisions")
    }
}
