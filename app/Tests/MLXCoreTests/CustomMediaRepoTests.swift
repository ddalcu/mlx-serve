import XCTest
@testable import MLXCore

/// Community media packs in the Model Browser's search: a repo declaring a
/// served media arch (`config.model_type` from the HF API, or a tag spelling
/// it verbatim) is downloadable ONLY after its file tree proves the family
/// bundle's ready markers — "would it work in our engine" is checked before
/// the button exists, not discovered 40 GB later. Downloads reuse the family
/// bundle (same allowlists + markers as the catalog packs) with the repo
/// swapped in.
final class CustomMediaRepoTests: XCTestCase {

    private func repo(_ id: String, tags: [String], pipeline: String?,
                      modelType: String? = nil) -> HFModel {
        var m = HFModel(id: id, downloads: 10, likes: 1, lastModified: nil,
                        tags: tags, safetensors: nil, pipelineTag: pipeline)
        if let modelType { m.config = HFConfigMeta(modelType: modelType) }
        return m
    }

    /// The antocorr repo's real tree shape (checked live 2026-08-08).
    private let h3PackTree: [HFSearchService.TreeFileEntry] = [
        .init(path: "config.json", size: 724),
        .init(path: "transformer.safetensors", size: 18_698_813_290),
        .init(path: "text_encoder.safetensors", size: 9_595_816_442),
        .init(path: "video_vae.safetensors", size: 5_207_808_496),
        .init(path: "audio_vae.safetensors", size: 605_254_808),
        .init(path: "tokenizer.json", size: 7_032_403),
        .init(path: "tokenizer_config.json", size: 11_003),
    ]

    // MARK: - Detection

    func testMediaRepoDetectedFromConfigModelTypeAndJudgedByStructure() {
        var m = repo("antocorr/MiniMax-H3-FL2VA-MLX-Serve-2bit-text-encoder",
                     tags: ["mlx", "mlx-serve"], pipeline: "text-to-video",
                     modelType: "minimax_h3")
        XCTAssertTrue(m.isServedMediaRepo)
        XCTAssertTrue(m.isSupportedArchitecture)
        // Unverified (tree not fetched yet, or unfetchable): no Download
        // button — a raw upstream checkpoint must never get one by default.
        XCTAssertFalse(m.isCompatible)
        XCTAssertNotNil(m.incompatibleReason)
        m.mediaStructureVerified = true
        XCTAssertTrue(m.isCompatible)
        XCTAssertNil(m.incompatibleReason)
        m.mediaStructureVerified = false
        XCTAssertFalse(m.isCompatible)
    }

    func testMediaRepoDetectedFromVerbatimTagWhenConfigAbsent() {
        let tagged = repo("someone/h3-pack", tags: ["mlx", "minimax_h3"], pipeline: "text-to-video")
        XCTAssertEqual(tagged.mediaFamilyModelType, "minimax_h3")
        // Kokoro is voice-mode-only — never enters through this door.
        let kokoro = repo("someone/kokoro-tuned", tags: ["mlx", "kokoro"], pipeline: "text-to-speech")
        XCTAssertFalse(kokoro.isServedMediaRepo)
        // A random diffusers repo keeps its old incompatibility, unchanged.
        let sd = repo("x/diffusion", tags: ["diffusers"], pipeline: "text-to-image")
        XCTAssertFalse(sd.isServedMediaRepo)
        XCTAssertFalse(sd.isCompatible)
    }

    func testDiffusersRepoDetectedFromItsPipelineClassName() throws {
        // A diffusers repo carries NO `model_type` — HF surfaces the pipeline's
        // own `_class_name` instead. Live shape from
        // "mage-flow-community/Mage-Flow-Turbo" (the renamed microsoft org,
        // 2026-08-09), tags verbatim: none spells a served model_type. Only
        // classes our engines load in that repo's OWN layout map to a family;
        // any other diffusers repo keeps its old incompatibility.
        let json = Data("""
        [{"id":"mage-flow-community/Mage-Flow-Turbo","pipeline_tag":"text-to-image",
          "tags":["diffusers","safetensors","text-to-image","diffusion","mage-flow"],
          "config":{"diffusers":{"_class_name":"MageFlowPipeline"}}},
         {"id":"x/sdxl","pipeline_tag":"text-to-image","tags":["diffusers"],
          "config":{"diffusers":{"_class_name":"StableDiffusionXLPipeline"}}}]
        """.utf8)
        let models = try JSONDecoder().decode([HFModel].self, from: json)
        XCTAssertEqual(models[0].mediaFamilyModelType, "mage_flow")
        XCTAssertTrue(models[0].isServedMediaRepo)
        XCTAssertFalse(models[0].isCompatible) // unverified until the tree check
        XCTAssertFalse(models[1].isServedMediaRepo)
        XCTAssertFalse(models[1].isCompatible)

        // The tree check runs the mage_flow family markers, and the Edit repo
        // adopts the edit family by its name — the server's dirIsEdit rule.
        let bundle = CustomMediaModels.bundle(
            arch: "mage_flow", repoId: "mage-flow-community/Mage-Flow-Edit-Turbo")!
        XCTAssertEqual(bundle.primaryRepo, "mage-flow-community/Mage-Flow-Edit-Turbo")
        let markers = bundle.components[0].readyMarkers
        let tree: [HFSearchService.TreeFileEntry] = [
            .init(path: "model_index.json", size: 120),
            .init(path: "transformer/diffusion_pytorch_model.safetensors", size: 9),
            .init(path: "vae/diffusion_pytorch_model.safetensors", size: 9),
            .init(path: "text_encoder/model.safetensors", size: 9),
            .init(path: "scheduler/scheduler_config.json", size: 9),
        ]
        XCTAssertTrue(HFSearchService.mediaStructureSatisfied(markers: markers, files: tree))
        XCTAssertFalse(HFSearchService.mediaStructureSatisfied(
            markers: markers, files: tree.filter { !$0.path.hasPrefix("transformer/") }))
    }

    // MARK: - Tree verification

    func testStructureCheckIsTheFamilyBundlesOwnMarkers() {
        let markers = CustomMediaModels.bundle(
            arch: "minimax_h3", repoId: "antocorr/x")!.components[0].readyMarkers
        XCTAssertTrue(HFSearchService.mediaStructureSatisfied(markers: markers, files: h3PackTree))
        // A raw upstream repo (no converted transformer.safetensors) fails.
        let raw = h3PackTree.filter { $0.path != "transformer.safetensors" }
        XCTAssertFalse(HFSearchService.mediaStructureSatisfied(markers: markers, files: raw))
    }

    func testDirectoryMarkersMatchByPathPrefix() {
        // FLUX markers name SUBDIRS ("vae", "tokenizer") — the tree only has
        // files, so a marker must match anything living under it.
        XCTAssertTrue(HFSearchService.mediaStructureSatisfied(
            markers: ["config.json", "vae"],
            files: [.init(path: "config.json", size: 1),
                    .init(path: "vae/diffusion_pytorch_model.safetensors", size: 9)]))
        XCTAssertFalse(HFSearchService.mediaStructureSatisfied(
            markers: ["vae"],
            files: [.init(path: "vae.txt", size: 1)]))
    }

    // MARK: - Fetch gate

    func testMediaRowsAreFetchedForVerificationEvenWithSizeMetadata() {
        var m = repo("someone/h3-pack", tags: ["minimax_h3"], pipeline: "text-to-video")
        // Size metadata present — a plain row would skip the tree fetch, but
        // an unverified media row still needs it to earn (or lose) its button.
        XCTAssertTrue(HFSearchService.needsFallbackFetch(m))
        m.mediaStructureVerified = true
        XCTAssertFalse(HFSearchService.needsFallbackFetch(m))
    }

    // MARK: - Family bundle synthesis

    func testFamilyBundleSwapsTheRepoAndKeepsTheFamilyContract() {
        let b = CustomMediaModels.bundle(
            arch: "minimax_h3",
            repoId: "antocorr/MiniMax-H3-FL2VA-MLX-Serve-2bit-text-encoder")!
        XCTAssertEqual(b.primaryRepo, "antocorr/MiniMax-H3-FL2VA-MLX-Serve-2bit-text-encoder")
        XCTAssertEqual(b.components[0].readyMarkers,
                       VideoModelPreset.minimaxH3.bundle.components[0].readyMarkers)
        // LTX customs still pull the Gemma text-encoder dependency.
        let ltx = CustomMediaModels.bundle(arch: "AudioVideo", repoId: "someone/ltx-pack")!
        XCTAssertEqual(ltx.dependencyRepos, [MediaBundle.ltxGemmaRepo])
        // Non-media archs and Kokoro synthesize nothing.
        XCTAssertNil(CustomMediaModels.bundle(arch: "gemma4", repoId: "x/y"))
        XCTAssertNil(CustomMediaModels.bundle(arch: "kokoro", repoId: "x/y"))
    }

    func testSearchURLAsksForTheConfigBlock() {
        let url = HFSearchService.searchURL(query: "minimax", filter: "mlx", skip: 0, limit: 30)!
        XCTAssertTrue(url.absoluteString.contains("expand%5B%5D=config"))
    }

    func testConfigBlockDecodesFromTheApiShape() throws {
        // The list API's real shape (config carries more than model_type, and
        // can be absent entirely on repos with no root config.json).
        let json = Data("""
        [{"id":"a/b","pipeline_tag":"text-to-video","tags":["mlx"],
          "config":{"model_type":"minimax_h3","tokenizer_config":{"bos_token":null}}},
         {"id":"c/d","pipeline_tag":"text-generation","tags":["mlx"]}]
        """.utf8)
        let models = try JSONDecoder().decode([HFModel].self, from: json)
        XCTAssertEqual(models[0].config?.modelType, "minimax_h3")
        XCTAssertEqual(models[0].mediaFamilyModelType, "minimax_h3")
        XCTAssertNil(models[1].config?.modelType)
    }
}
