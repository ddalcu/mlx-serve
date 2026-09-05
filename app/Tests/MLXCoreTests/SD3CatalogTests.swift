import XCTest
@testable import MLXCore

/// SD 3.5 (Large, Large-Turbo, Medium) reaches the same image backend as every
/// other preset, so the risk is not that it fails to load — it is that a
/// capability flag defaulting to `true` advertises a control the server's own
/// arm cannot honour.
final class SD3CatalogTests: XCTestCase {

    private var sd3Presets: [ImageModelPreset] {
        ImageModelPreset.all.filter { $0.variant == .sd3 || $0.variant == .sd3Turbo }
    }

    func testAllThreeCheckpointsAreInTheCatalog() {
        let ids = Set(sd3Presets.map(\.id))
        XCTAssertEqual(ids, [
            "stabilityai/stable-diffusion-3.5-medium",
            "stabilityai/stable-diffusion-3.5-large",
            "stabilityai/stable-diffusion-3.5-large-turbo",
        ])
        // Every one routes to the `sd3` server config, never to sdxl's.
        for p in sd3Presets { XCTAssertEqual(p.configName, "sd3") }
    }

    /// A capability flag is a claim about the SERVER's arm for this backend.
    ///
    /// `supportsLoRA` and `supportsImg2Img` both fall through `default: true`,
    /// so a new variant inherits `true` for both the moment it is declared —
    /// which is right for one of them here and wrong for the other. `gen.zig`'s
    /// `.sd3` arm returns 0 matched from `attachLora` by construction (SD 3.5
    /// adapters target the MMDiT's joint blocks, not the UNet module tree
    /// `lora.canonicalizeSdxl` speaks), so offering the picker would advertise a
    /// control that always 400s.
    func testLoRAIsNotOfferedBecauseTheServerArmMatchesNothing() {
        for p in sd3Presets {
            XCTAssertFalse(p.supportsLoRA, "\(p.id) has no server-side LoRA arm — the picker must stay hidden")
        }
    }

    /// The other side of that default, and this one IS earned: `sd3_vae.Encoder`
    /// loads, the pipeline holds it as `vae_enc`, and `gen.supportsImg2Img`
    /// reads exactly that.
    func testImg2ImgIsOfferedBecauseTheVaeEncoderIsReal() {
        for p in sd3Presets {
            XCTAssertTrue(p.supportsImg2Img, "\(p.id) loads a VAE encoder and takes a source image")
        }
        // Instruction editing is a TRAINED capability base SD 3.5 does not have,
        // and the server refuses it by name rather than silently ignoring it.
        for p in sd3Presets { XCTAssertFalse(p.supportsReferenceEdit) }
    }

    /// Real CFG on the base checkpoints, guidance-free on the distill. This is
    /// the same base-vs-distill split the SDXL presets encode, and getting it
    /// backwards hides the negative-prompt field on the models that read it.
    func testGuidanceSplitsBaseFromTheTurboDistill() {
        for p in sd3Presets where p.variant == .sd3 {
            XCTAssertTrue(p.supportsNegativePrompt, "\(p.id) runs a batch-2 CFG forward")
            XCTAssertTrue(p.supportsGuidance)
        }
        for p in sd3Presets where p.variant == .sd3Turbo {
            XCTAssertFalse(p.supportsNegativePrompt, "\(p.id) is distilled guidance-free")
        }
    }

    /// Mirrors `gen.zig`'s `.sd3 => clampKreaDim`: VAE x8 and MMDiT patch x2, so
    /// the patch grid is exact only on multiples of 16. Drift here shows up as
    /// the app accepting a size the server then silently snaps somewhere else.
    func testResolutionGridIsSixteenNotSdxlsBucketList() {
        for p in sd3Presets {
            XCTAssertEqual(p.resolutionGrid.alignment, 16, "\(p.id): VAE x8 * patch x2")
            XCTAssertEqual(p.resolutionGrid.maxDim, 2048)
            XCTAssertEqual(p.validResolution(p.defaultResolution, editMode: false), p.defaultResolution,
                           "\(p.id): its own default must survive its own grid")
        }
    }

    /// Turbo is four steps, not a slower profile of Large's twenty-eight.
    func testTurboQualityProfilesAreFewStep() {
        guard let turbo = sd3Presets.first(where: { $0.variant == .sd3Turbo }) else {
            return XCTFail("no turbo preset")
        }
        XCTAssertEqual(turbo.qualityProfiles[.good]?.steps, 4)
        for (_, profile) in turbo.qualityProfiles {
            XCTAssertLessThanOrEqual(profile.steps, 8, "a 4-step distill never needs 28")
        }
    }

    /// The bundle is what makes the preset downloadable at all — a preset with
    /// no matching bundle arm falls through to `.flux`, whose recursive default
    /// has no basename allowlist and would pull the 16 GB merged root
    /// checkpoint plus the ComfyUI `text_encoders/` drop: a second AND third
    /// copy of the same weights.
    func testBundleIsTheSd3ArmAndExcludesTheDuplicateCopies() {
        for p in sd3Presets {
            let bundle = p.bundle
            XCTAssertTrue(bundle.id.hasPrefix("sd3:"), "\(p.id) must reach the sd3 bundle arm, not flux's default")
            guard let component = bundle.components.first else { return XCTFail("no component") }
            // The T5 tower is required to serve at all: leaving it out of the
            // markers would let a half-download read as complete.
            XCTAssertTrue(component.readyMarkers.contains("text_encoder_3"))
            XCTAssertTrue(component.readyMarkers.contains("transformer"))
            XCTAssertFalse(component.readyMarkers.contains("unet"), "SD 3.5 has no UNet")
            // The ComfyUI flat drop is a third copy of the towers.
            XCTAssertTrue(component.selection.excludeSubstrings.contains("text_encoders/"))
            // Sharded T5: the PLAIN shards, because `model.safetensors.index.json`
            // names those and the server reads no other index.
            let keep = component.selection.keepSafetensors ?? []
            XCTAssertTrue(keep.contains("model-00001-of-00002.safetensors"))
            XCTAssertFalse(keep.contains(where: { $0.contains(".fp16-") }),
                           "an fp16 shard set is skipped shard-for-shard by indexShardSet")
        }
    }
}
