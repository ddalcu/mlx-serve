import XCTest
@testable import MLXCore

/// A Laya typed-decision checkpoint ships NO root config.json (its configs
/// are `rl_agent_config.json` + `encoder/config.json`), so every gate keyed on
/// that file dropped it: invisible in the Downloaded tab, never "on disk" for
/// the resolver, and no bundle to verify a search row against. Server twin:
/// `model_discovery.peekLayaCheckpoint`.
final class LayaModelTests: XCTestCase {

    private func makeLayaDir() throws -> (root: String, dir: String) {
        let fm = FileManager.default
        let root = NSTemporaryDirectory() + "laya-\(UUID().uuidString)"
        let dir = (root as NSString).appendingPathComponent("aac6fef/laya-multilingual-mlx")
        for sub in ["encoder", "tokenizer"] {
            try fm.createDirectory(atPath: (dir as NSString).appendingPathComponent(sub), withIntermediateDirectories: true)
        }
        fm.createFile(atPath: (dir as NSString).appendingPathComponent("rl_agent_config.json"), contents: Data("{}".utf8))
        fm.createFile(atPath: (dir as NSString).appendingPathComponent("encoder/config.json"), contents: Data("{}".utf8))
        fm.createFile(atPath: (dir as NSString).appendingPathComponent("tokenizer/tokenizer.json"), contents: Data("{}".utf8))
        fm.createFile(atPath: (dir as NSString).appendingPathComponent("model.safetensors"),
                      contents: Data(count: Int(DownloadManager.minimumWeightBytes) + 1))
        return (root, dir)
    }

    func testAConfiglessLayaDirIsListedAsALayaModelAndNeverChatPickable() throws {
        let (root, dir) = try makeLayaDir()
        defer { try? FileManager.default.removeItem(atPath: root) }

        let models = DownloadManager.makeLocalModels(
            atDir: dir, displayName: "aac6fef/laya-multilingual-mlx",
            idKey: "aac6fef/laya-multilingual-mlx", source: .mlxServe)
        XCTAssertEqual(models.count, 1)
        let m = try XCTUnwrap(models.first)
        XCTAssertEqual(m.modelType, "laya")
        XCTAssertNil(m.defect)
        XCTAssertTrue(m.isSupportedArchitecture, "must not badge Unsupported")
        XCTAssertFalse(m.isChatPickable)
        XCTAssertTrue(DownloadManager.holdsWeightLayout(dir), "the resolver must see the download as present")
        XCTAssertNotNil(DownloadManager.existingModelDir(rootDir: root, repoId: "aac6fef/laya-multilingual-mlx"))
    }

    func testHalfOfTheLayaMarkersIsNotAModel() throws {
        let fm = FileManager.default
        let root = NSTemporaryDirectory() + "laya-half-\(UUID().uuidString)"
        let dir = (root as NSString).appendingPathComponent("x/y")
        try fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(atPath: root) }
        fm.createFile(atPath: (dir as NSString).appendingPathComponent("rl_agent_config.json"), contents: Data("{}".utf8))
        XCTAssertFalse(DownloadManager.holdsWeightLayout(dir))
        XCTAssertTrue(DownloadManager.makeLocalModels(atDir: dir, displayName: "x/y", idKey: "x/y", source: .mlxServe).isEmpty)
    }

    func testASearchRowTaggedLayaVerifiesAgainstTheRealRepoTree() throws {
        // What `aac6fef/laya-multilingual-mlx` actually ships (tree API, 2026-09).
        let tree = ["LICENSE", "NOTICE", "README.md", "encoder/config.json", "manifest.json", "mlx_config.json",
                    "model.safetensors", "rl_agent_config.json", "tokenizer/tokenizer.json",
                    "tokenizer/tokenizer_config.json", "validation.json"]
            .map { HFSearchService.TreeFileEntry(path: $0, size: 1) }
        let bundle = try XCTUnwrap(CustomMediaModels.bundle(arch: "laya", repoId: "aac6fef/laya-multilingual-mlx"))
        XCTAssertTrue(HFSearchService.mediaStructureSatisfied(markers: bundle.components[0].readyMarkers, files: tree))
        XCTAssertFalse(HFSearchService.mediaStructureSatisfied(markers: bundle.components[0].readyMarkers,
                                                              files: [HFSearchService.TreeFileEntry(path: "model.safetensors", size: 1)]),
                       "a bare weights file is not a Laya pack")

        let row = HFModel(id: "aac6fef/laya-multilingual-mlx", downloads: 1, likes: 0, lastModified: nil,
                          tags: ["mlx", "safetensors", "laya", "text-classification"], safetensors: nil,
                          pipelineTag: "text-classification")
        XCTAssertEqual(row.mediaFamilyModelType, "laya")
        XCTAssertTrue(row.isSupportedArchitecture)
        XCTAssertNil(MediaModality(modelType: "laya"), "no create pane, so no Use button")
    }
}
