import XCTest
@testable import MLXCore

/// MOSS-Image: the preset declares what `gen.zig`'s `.moss_image` arm honours,
/// the request keeps CFG explicit against a backend whose default is 3, and
/// Enhance writes the model's own caption-or-layout prompt.
@MainActor
final class MossImageTests: XCTestCase {

    private let preset = ImageModelPreset.mossImageSFT_bf16

    // MARK: - Preset

    func testPresetOffersOnlyWhatTheBackendHonors() throws {
        for preset in [ImageModelPreset.mossImageSFT_8bit, .mossImageSFT_bf16] {
            XCTAssertTrue(preset.supportsGuidance, "undistilled, real CFG")
            XCTAssertEqual(preset.defaultGuidance, 3.0, "the authors' default")
            XCTAssertEqual(preset.defaultNegativePrompt, "nsfw, worst quality, bad anatomy, watermark, jpeg artifact")
            XCTAssertTrue(preset.supportsLoRA, "block linears take stacked adapters (the DMD few-step LoRAs)")
            XCTAssertFalse(preset.supportsImg2Img, "text-to-image only")
            XCTAssertFalse(preset.supportsReferenceEdit)
            XCTAssertFalse(preset.stepsAreFixed)
            XCTAssertEqual(preset.condWeightCount, 0)
            XCTAssertEqual(preset.settings(preset.defaultQuality).steps, 50, "the card: best at 50 steps or more")
            XCTAssertEqual(preset.resolutionGrid.alignment, 16)
            for r in preset.resolutions {
                XCTAssertEqual(r.width % 16, 0)
                XCTAssertEqual(r.height % 16, 0)
                XCTAssertLessThanOrEqual(max(r.width, r.height), preset.resolutionGrid.maxDim)
            }
        }
        // The quantized pack is the one offered first; bf16 stays as the quality reference.
        let all = ImageModelPreset.all
        XCTAssertLessThan(try XCTUnwrap(all.firstIndex(of: .mossImageSFT_8bit)), try XCTUnwrap(all.firstIndex(of: .mossImageSFT_bf16)))
        XCTAssertEqual(ImageModelPreset.flux2Klein9BBase_Q4.defaultGuidance, 1.0)
        XCTAssertEqual(ImageModelPreset.qwenImage21_8bit.defaultNegativePrompt, "")
    }

    // MARK: - Wire

    func testGuidanceIsOmittedOnlyWhenUnset() {
        var req = ImageGenRequest(model: preset, prompt: "x", width: 1024, height: 1024, steps: 50)
        XCTAssertNil(ImageGenService.requestJson(for: req, modelName: "m", seed: 1)["guidance_scale"])
        // 1.0 turns MOSS's CFG off; omitted, the server would run its default 3.
        req.guidanceScale = 1.0
        XCTAssertEqual(ImageGenService.requestJson(for: req, modelName: "m", seed: 1)["guidance_scale"] as? Double, 1.0)
    }

    func testChatGenerationsUseTheRecommendedNegativeAndServerCfg() throws {
        let req = try MediaToolArgs.image(["prompt": "a fox"], model: preset, saved: preset.defaultResolution,
                                          seed: 1, keepResident: false, lanId: nil)
        XCTAssertNil(req.guidanceScale)
        XCTAssertEqual(req.negativePrompt, preset.defaultNegativePrompt)
        let flux = try MediaToolArgs.image(["prompt": "a fox"], model: .flux2Klein4B_Q4,
                                           saved: ImageModelPreset.flux2Klein4B_Q4.defaultResolution,
                                           seed: 1, keepResident: false, lanId: nil)
        XCTAssertEqual(flux.negativePrompt, "")
    }

    // MARK: - Discovery + downloads

    func testCustomAndCommunityPacksAdoptTheFamily() throws {
        XCTAssertEqual(CustomMediaModels.imagePresets(from: [
            ModelInfo(name: "me/moss-tuned", quantBits: 16, layers: 0, hiddenSize: 0, vocabSize: 0,
                      contextLength: 0, modelMaxTokens: 0, architecture: "moss_image", capabilities: ["image"], lanPeer: nil),
        ]).first?.variant, .mossImage)

        let json = Data("""
        [{"id":"OpenMOSS-Team/MOSS-Image-SFT","pipeline_tag":"text-to-image","tags":["diffusers","moss-image"],
          "config":{"diffusers":{"_class_name":"MOSSImagePipeline"}}}]
        """.utf8)
        XCTAssertEqual(try JSONDecoder().decode([HFModel].self, from: json)[0].mediaFamilyModelType, "moss_image")

        // The mlx-community pack's tree (checked live 2026-10-18), minus the shards' siblings.
        let markers = try XCTUnwrap(CustomMediaModels.bundle(arch: "moss_image", repoId: "x/y")).components[0].readyMarkers
        let tree: [HFSearchService.TreeFileEntry] = [
            .init(path: "model_index.json", size: 412),
            .init(path: "scheduler/scheduler_config.json", size: 494),
            .init(path: "text_encoder/model-00001-of-00004.safetensors", size: 9),
            .init(path: "tokenizer/tokenizer.json", size: 9),
            .init(path: "transformer/diffusion_pytorch_model-00001-of-00003.safetensors", size: 9),
            .init(path: "vae/diffusion_pytorch_model.safetensors", size: 9),
        ]
        XCTAssertTrue(HFSearchService.mediaStructureSatisfied(markers: markers, files: tree))
        XCTAssertFalse(HFSearchService.mediaStructureSatisfied(
            markers: markers, files: tree.filter { !$0.path.hasPrefix("tokenizer/") }))
    }

    // MARK: - Layout compile (port of the enhancer's compile_prompt.py tests)

    private let canvas = MossLayout.Canvas(width: 1536, height: 1024)

    private func draft(bbox: String, extra: String = "") -> String {
        """
        {"width":1536,"height":1024,"bbox_coord_space":"abs","bbox_format":"xyxy",
         "summary":{"meta":["poster"],"canvas":"single","caption":"A ceramic campaign."},
         "items":[{"label":"text","caption":"Dark-green lettering.","text":"把日常，过成诗。\\n\\"MOSS\\" 🪴","bbox":\(bbox)}\(extra)]}
        """
    }

    func testLandscapeConversionIsOneLineInTheDocumentedOrder() throws {
        let out = try XCTUnwrap(MossLayout.compile(draft(bbox: "[928, 272, 1408, 448]"), canvas: canvas))
        XCTAssertFalse(out.contains("\n"))
        XCTAssertEqual(out, #"{"summary":{"meta":["poster"],"canvas":"single","caption":"A ceramic campaign."},"items":[{"label":"text","caption":"Dark-green lettering.","text":"把日常，过成诗。\n\"MOSS\" 🪴","bbox":[-240,160,-64,640]}],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}"#)
    }

    func testFractionalEdgesAreEnclosedBeforeCentering() {
        XCTAssertEqual(MossLayout.centeredBox([0, 0, 768, 1344], width: 768, height: 1344), [-672, -384, 672, 384])
        XCTAssertEqual(MossLayout.centeredBox([10.9, 20.1, 100.01, 200.99], width: 768, height: 1344), [-652, -374, -471, -283])
        XCTAssertEqual(MossLayout.centeredBox([1.00000000001, 2, 100, 200], width: 768, height: 1344), [-670, -383, -472, -284])
    }

    /// A local chat model has no second try: a box past the edge is clipped, a box with no area dropped.
    func testOutOfCanvasBoxesAreClippedAndEmptyOnesDropped() throws {
        let extra = #",{"label":"object","caption":"Nothing.","bbox":[50,50,50,90]}"#
        let out = try XCTUnwrap(MossLayout.compile(draft(bbox: "[-20, 900, 1600, 1100]", extra: extra), canvas: canvas))
        XCTAssertTrue(out.contains(#""bbox":[388,-768,512,768]"#), out)
        XCTAssertFalse(out.contains("Nothing."))
    }

    /// Boxes drawn on another canvas are scaled to the one the pane generates.
    func testADraftOnAnotherCanvasIsScaledToThePanes() throws {
        let out = try XCTUnwrap(MossLayout.compile(draft(bbox: "[0, 0, 768, 512]"), canvas: .init(width: 768, height: 512)))
        XCTAssertTrue(out.contains(#""bbox":[-256,-384,0,0]"#), out)
    }

    func testCaptionDraftsAndProseComeBackAsText() {
        XCTAssertEqual(MossLayout.compile(#"{"caption":"A fox in snow."}"#, canvas: canvas), "A fox in snow.")
        XCTAssertNil(MossLayout.compile("A fox in snow.", canvas: canvas))
        XCTAssertNil(MossLayout.compile(#"{"summary":{"caption":"x"},"items":[]}"#, canvas: canvas))
        let compiled = ImagePromptExamples.mossImage.examples.last!.body
        XCTAssertEqual(MossLayout.compile(compiled, canvas: canvas), compiled)
    }

    // MARK: - Enhance

    func testEnhanceAsksForACaptionOrAPixelLayoutOnThePanesCanvas() {
        let r = PromptRewriter.image(text: "a poster", editing: false, groups: preset.promptExamples(editing: false),
                                     denseText: false, mossCanvas: .init(width: 1280, height: 768))
        XCTAssertTrue(r.system.contains("1280"))
        XCTAssertTrue(r.system.contains("\"xyxy\""))
        XCTAssertEqual(r.layoutCanvas, .init(width: 1280, height: 768))
        XCTAssertEqual(PromptRewriter.finish(#"{"caption":"A fox."}"#, request: r), "A fox.")
        XCTAssertEqual(PromptRewriter.finish("A fox.", request: r), "A fox.")
        // Without a layout canvas the reply is only cleaned.
        let plain = PromptRewriter.image(text: "x", editing: false, groups: ImagePromptExamples.textToImage, denseText: false)
        XCTAssertNil(plain.layoutCanvas)
        XCTAssertEqual(PromptRewriter.finish(#"{"caption":"A fox."}"#, request: plain), #"{"caption":"A fox."}"#)
    }
}
