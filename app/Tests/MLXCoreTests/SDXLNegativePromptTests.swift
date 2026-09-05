import XCTest
@testable import MLXCore

/// The negative prompt's WIRE contract. The distinction these pin is not
/// cosmetic: on SDXL an absent `negative_prompt` zeroes the unconditional
/// branch, while an empty string is ENCODED (BOS + EOS + 75 pads through both
/// text towers) and is a different tensor. Measured end to end against
/// diffusers, collapsing the two is worth cos 0.975 vs 0.997.
@MainActor
final class SDXLNegativePromptTests: XCTestCase {

    private func req(_ negative: String, model: ImageModelPreset = .sdxlBase10) -> [String: Any] {
        let r = ImageGenRequest(
            model: model, prompt: "a garden", width: 1024, height: 1024, steps: 30,
            negativePrompt: negative
        )
        return ImageGenService.requestJson(for: r, modelName: model.id, seed: 1)
    }

    func testBlankNegativePromptOmitsTheKeyEntirely() {
        // A user who never touched the box means ABSENT, not empty.
        XCTAssertNil(req("")["negative_prompt"])
        XCTAssertNil(req("   ")["negative_prompt"], "whitespace-only is still 'untouched'")
        XCTAssertNil(req("\n")["negative_prompt"])
    }

    func testTypedNegativePromptIsSentTrimmed() {
        XCTAssertEqual(req("blurry, watermark")["negative_prompt"] as? String, "blurry, watermark")
        XCTAssertEqual(req("  blurry  ")["negative_prompt"] as? String, "blurry")
    }

    func testOnlyGuidanceCapableModelsAdvertiseTheField() {
        // SDXL runs real classifier-free guidance, so it has an unconditional
        // branch to steer. Every other preset here is distilled and generates
        // guidance-free — the box would be decoration.
        XCTAssertTrue(ImageModelPreset.sdxlBase10.supportsNegativePrompt)
        // Turbo is SDXL too, but adversarially distilled and guidance-free —
        // it has no unconditional branch, so it must NOT advertise the field.
        XCTAssertFalse(ImageModelPreset.sdxlTurbo.supportsNegativePrompt)
        // The community finetunes (Illustrious / Pony / NoobAI) are base-SDXL
        // descendants, NOT distills: they run the same real guidance, and the
        // anime-SDXL ecosystem steers with negative prompts more than base does.
        // So the guidance-capable set is two variants, not one. SD 1.5 runs
        // real guidance too — it is the SDXL backend's config, not a distill.
        // SD 3.5 Large and Medium likewise run real CFG (a batch-2 forward at
        // guidance ~4.5); only `.sd3Turbo`, the 4-step distill, is guidance-free,
        // which is exactly the base-vs-distill split this set encodes.
        let guidanceCapable: Set<FluxVariant> = [.sdxlBase10, .sdxlFinetune, .sd1, .sd3]
        XCTAssertFalse(ImageModelPreset.all.filter { $0.variant == .sdxlFinetune }.isEmpty,
                       "a finetune must be in the catalog or this assertion is vacuous")
        for p in ImageModelPreset.all where p.variant == .sdxlFinetune {
            XCTAssertTrue(p.supportsNegativePrompt, "\(p.id) runs real guidance and must offer the field")
        }
        for p in ImageModelPreset.all where !guidanceCapable.contains(p.variant) {
            XCTAssertFalse(p.supportsNegativePrompt, "\(p.id) does not read a negative prompt")
        }
    }

    func testSdxlPresetsAreRegisteredAndOnTrainingBuckets() {
        for preset in [ImageModelPreset.sdxlBase10, .sdxlTurbo] {
            XCTAssertTrue(ImageModelPreset.all.contains(preset),
                          "\(preset.id) must be in `all` or it never reaches the picker")
            // SDXL is trained on /64 buckets and drifts off-distribution
            // between them; every offered resolution must land on one, and the
            // grid the pane enforces must agree with the server's
            // `gen.clampSdxlDim` or the app accepts sizes the server re-snaps.
            for r in preset.resolutions {
                XCTAssertEqual(r.width % 64, 0, "\(r.width) is not a multiple of 64")
                XCTAssertEqual(r.height % 64, 0, "\(r.height) is not a multiple of 64")
            }
            XCTAssertEqual(preset.resolutionGrid.alignment, 64)
            XCTAssertEqual(preset.resolutionGrid.minDim, 512)
            XCTAssertEqual(preset.resolutionGrid.maxDim, 2048)
            // A default the menu does not contain leaves the picker showing
            // nothing selected (`ResolutionOption` is Hashable over its label).
            XCTAssertTrue(preset.resolutions.contains(preset.defaultResolution),
                          "\(preset.id)'s default resolution is not one of its own options")
        }
        // Base is not a distill: it needs real step counts, not 4-8. Turbo is,
        // and must stay in single digits or it is being run like the base.
        XCTAssertGreaterThanOrEqual(ImageModelPreset.sdxlBase10.settings(.good).steps, 20)
        XCTAssertLessThanOrEqual(ImageModelPreset.sdxlTurbo.settings(.good).steps, 4)
    }
}
