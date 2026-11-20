import XCTest
@testable import MLXCore

/// MiniMax-H3-Image: the preset declares what the `minimax_h3_image` plugin honours (six fixed PDD steps, edits of up
/// to 16 pictures named `<Picture i>`, no guidance, LoRAs or variations, any size from 16 to 6272 per side), and the
/// catalog offers the 4-bit pack alone.
@MainActor
final class MiniMaxH3ImageTests: XCTestCase {

    private let preset = ImageModelPreset.minimaxH3Image_4bit

    func testPresetOffersOnlyWhatTheBackendHonors() {
        XCTAssertTrue(preset.supportsReferenceEdit)
        XCTAssertEqual(preset.maxReferenceImages, 15, "16 pictures per edit, the source included")
        XCTAssertFalse(preset.supportsImg2Img, "a variation is a named 400")
        XCTAssertFalse(preset.supportsLoRA, "LoRAs are a named 400")
        XCTAssertFalse(preset.supportsGuidance, "distilled: no classifier-free guidance")
        XCTAssertTrue(preset.stepsAreFixed)
        XCTAssertTrue(preset.stepsSetByCheckpoint, "the pack's own six-step grid")
        XCTAssertEqual(preset.fixedSteps, 6)
        for q in QualityPreset.allCases { XCTAssertEqual(preset.settings(q).steps, 6) }
        XCTAssertEqual(preset.condWeightCount, 0)
        XCTAssertEqual(preset.capabilityLabel, "Best for creating images")
        XCTAssertEqual(preset.defaultResolution.width, 2048)
        XCTAssertEqual(preset.defaultResolution.height, 2048)
    }

    /// The server returns exactly the requested size (it generates on its own 32-pixel canvas and resizes back), so
    /// the pane corrects nothing inside 16-6272 and refuses outside it.
    func testAnySizeTheServerTakesIsTakenAsTyped() {
        let grid = preset.resolutionGrid
        XCTAssertEqual(grid.resolve(width: 1000, height: 768), .ok(width: 1000, height: 768))
        XCTAssertEqual(grid.resolve(width: 6272, height: 2688), .ok(width: 6272, height: 2688))
        if case .ok = grid.resolve(width: 6300, height: 2048) { XCTFail("6300 is past the 6272 side limit") }
        if case .ok = grid.resolve(width: 8, height: 8) { XCTFail("8 is under the 16 side limit") }
    }

    func testOnlyTheFourBitPackIsListed() {
        XCTAssertTrue(ImageModelPreset.all.contains(preset))
        XCTAssertEqual(preset.repo, "ddalcu/MiniMax-H3-Image-MLX-Serve-4bit")
        XCTAssertEqual(ImageModelPreset.all.filter { $0.variant == .minimaxH3Image }, [preset])
    }

    /// Any other step count is a named 400, so the field never goes out; the checkpoint's grid decides.
    func testRequestCarriesNoStepsOrGuidance() {
        let req = ImageGenRequest(model: preset, prompt: "a cow", width: 2048, height: 2048, steps: 4)
        let json = ImageGenService.requestJson(for: req, modelName: "m", seed: 1)
        XCTAssertNil(json["steps"])
        XCTAssertNil(json["guidance_scale"])
        XCTAssertEqual(json["size"] as? String, "2048x2048")
        let flux = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "a cow", width: 1024, height: 1024, steps: 8)
        XCTAssertEqual(ImageGenService.requestJson(for: flux, modelName: "m", seed: 1)["steps"] as? Int, 8)
    }

    /// The release names its pictures `<Picture i>`; the other editors' prompts say "image i".
    func testPicturesAreNamedTheWayTheModelReadsThem() {
        XCTAssertEqual(preset.referenceName(1), "<Picture 1>")
        XCTAssertEqual(preset.referenceName(3), "<Picture 3>")
        XCTAssertEqual(ImageModelPreset.flux2Klein4B_Q4.referenceName(2), "image 2")
    }

    func testBundleFetchesThePackAndKnowsWhenItIsComplete() throws {
        let c = try XCTUnwrap(preset.bundle.components.first)
        XCTAssertEqual(c.repo, preset.repo)
        XCTAssertEqual(c.selection.keepSafetensors,
                       ["transformer.safetensors", "text_encoder.safetensors", "image_vae.safetensors"])
        for marker in ["config.json", "transformer.safetensors", "text_encoder.safetensors",
                       "image_vae.safetensors", "tokenizer.json"] {
            XCTAssertTrue(c.readyMarkers.contains(marker), marker)
        }
    }

    /// The release expects one detailed English caption, the form MiniMax's own expander writes; a short prompt never
    /// reaches it. Edits keep the one-line instruction: written blind, the caption form redrew the source.
    func testEnhanceWritesTheCaptionTheModelExpects() {
        let t2i = PromptRewriter.image(text: "captain picard on the bridge", editing: false,
                                       groups: preset.promptExamples(editing: false), denseText: false, detailedCaption: true)
        XCTAssertTrue(t2i.system.contains("ONE paragraph"))
        XCTAssertTrue(t2i.system.contains("gets no looks the user did not give"))
        XCTAssertTrue(t2i.system.contains(ImagePromptExamples.minimaxH3.examples[0].body))

        let edit = PromptRewriter.image(text: "make it night", editing: true, groups: preset.promptExamples(editing: true),
                                        denseText: false, detailedCaption: true)
        XCTAssertEqual(edit, PromptRewriter.image(text: "make it night", editing: true,
                                                  groups: ImagePromptExamples.genericEdit, denseText: false))
    }

    func testTemplatesAreWrittenInTheModelsFormat() {
        XCTAssertEqual(preset.promptExamples(editing: false), [ImagePromptExamples.minimaxH3])
        XCTAssertEqual(preset.promptExamples(editing: true), ImagePromptExamples.genericEdit)
        for ex in ImagePromptExamples.minimaxH3.examples {
            XCTAssertTrue(ex.body.hasPrefix("The image is "), ex.title)
        }
    }

    /// A pack on disk (the 8-bit one included) rides this family under On This Mac.
    func testAPackOnDiskAdoptsTheFamily() {
        let found = CustomMediaModels.imagePresets(from: [
            ModelInfo(name: "ddalcu/MiniMax-H3-Image-MLX-Serve-8bit", quantBits: 8, layers: 0, hiddenSize: 0,
                      vocabSize: 0, contextLength: 0, modelMaxTokens: 0, architecture: "minimax_h3_image",
                      capabilities: ["image"], lanPeer: nil),
        ])
        XCTAssertEqual(found.first?.variant, .minimaxH3Image)
        XCTAssertEqual(found.first?.repo, "ddalcu/MiniMax-H3-Image-MLX-Serve-8bit")
    }
}
