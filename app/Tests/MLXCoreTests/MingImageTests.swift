import XCTest
@testable import MLXCore

/// Ming-Image: the preset declares what the server's `ming_image` plugin honours, and a pack on disk or on the hub
/// is recognised as that family.
@MainActor
final class MingImageTests: XCTestCase {

    private let preset = ImageModelPreset.mingImage_8bit

    func testPresetOffersOnlyWhatTheBackendHonors() {
        XCTAssertEqual(preset.repo, "ddalcu/Ming-Image-0.2-Design-MLX-Serve-8bit")
        XCTAssertTrue(ImageModelPreset.all.contains(preset))
        XCTAssertFalse(preset.supportsGuidance, "the release has no negative prompt; the server refuses one")
        XCTAssertFalse(preset.supportsImg2Img, "text-to-image only")
        XCTAssertFalse(preset.supportsReferenceEdit)
        XCTAssertFalse(preset.supportsLoRA)
        XCTAssertEqual(preset.condWeightCount, 0)
        XCTAssertEqual(preset.settings(preset.defaultQuality).steps, 12, "the release's default")
        let grid = preset.resolutionGrid
        for r in preset.resolutions {
            XCTAssertEqual(grid.snap(r.width), r.width, r.label)
            XCTAssertEqual(grid.snap(r.height), r.height, r.label)
            XCTAssertLessThanOrEqual(max(r.width, r.height), grid.maxDim, r.label)
        }
    }

    func testPacksOnDiskAndOnTheHubAdoptTheFamily() throws {
        XCTAssertEqual(MediaModality(modelType: "bailingmm_moe_v2_lite"), .image)
        XCTAssertEqual(DownloadManager.requiredMediaMarker(modelType: "bailingmm_moe_v2_lite"),
                       "vae/diffusion_pytorch_model.safetensors")
        XCTAssertEqual(CustomMediaModels.imagePresets(from: [
            ModelInfo(name: "me/ming-tuned", quantBits: 8, layers: 0, hiddenSize: 0, vocabSize: 0,
                      contextLength: 0, modelMaxTokens: 0, architecture: "bailingmm_moe_v2_lite",
                      capabilities: ["image"], lanPeer: nil),
        ]).first?.variant, .mingImage)

        // The published pack's tree; a Ming omni chat repo shares the root model_type but has no DiT.
        let markers = try XCTUnwrap(CustomMediaModels.bundle(arch: "bailingmm_moe_v2_lite", repoId: "x/y"))
            .components[0].readyMarkers
        let tree: [HFSearchService.TreeFileEntry] = [
            .init(path: "config.json", size: 9),
            .init(path: "mllm/model-00001-of-00004.safetensors", size: 9),
            .init(path: "connector/model.safetensors", size: 9),
            .init(path: "mlp/model.safetensors", size: 9),
            .init(path: "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", size: 9),
            .init(path: "vae/diffusion_pytorch_model.safetensors", size: 9),
            .init(path: "scheduler/scheduler_config.json", size: 9),
        ]
        XCTAssertTrue(HFSearchService.mediaStructureSatisfied(markers: markers, files: tree))
        XCTAssertFalse(HFSearchService.mediaStructureSatisfied(
            markers: markers, files: tree.filter { !$0.path.hasPrefix("transformer/") && !$0.path.hasPrefix("connector/") }))
    }
}
