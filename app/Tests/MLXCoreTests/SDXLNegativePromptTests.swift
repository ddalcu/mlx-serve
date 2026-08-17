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
        for p in ImageModelPreset.all where p.variant != .sdxlBase {
            XCTAssertFalse(p.supportsNegativePrompt, "\(p.id) does not read a negative prompt")
        }
    }

    func testSdxlPresetIsRegisteredAndOnTrainingBuckets() {
        XCTAssertTrue(ImageModelPreset.all.contains(ImageModelPreset.sdxlBase10),
                      "the preset must be in `all` or it never reaches the picker")
        // SDXL is trained on /64 buckets and drifts off-distribution between
        // them; every offered resolution must land on one.
        for r in ImageModelPreset.sdxlBase10.resolutions {
            XCTAssertEqual(r.width % 64, 0, "\(r.width) is not a multiple of 64")
            XCTAssertEqual(r.height % 64, 0, "\(r.height) is not a multiple of 64")
        }
        // Not a distill: it needs real step counts, not 4-8.
        XCTAssertGreaterThanOrEqual(ImageModelPreset.sdxlBase10.settings(.good).steps, 20)
    }
}
