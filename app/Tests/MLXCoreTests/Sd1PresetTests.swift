import XCTest
@testable import MLXCore

/// Stable Diffusion 1.5's app-side wiring. The server backend
/// (`sd1_pipeline.zig`) reuses SDXL's UNet/VAE/CLIP-L building blocks at
/// SD 1.x's own config, and the app side has to declare that config's real
/// facts rather than inherit SDXL's — its own resolution grid (trained at
/// 512, not SDXL's 1024) and its own download bundle (one text tower, so
/// `sdxlDiffusers`'s `text_encoder_2` ready marker would never be satisfied).
final class Sd1PresetTests: XCTestCase {

    func testSd15IsInTheCuratedCatalog() {
        XCTAssertTrue(ImageModelPreset.all.contains(.sd15),
                      "sd15 must be in `all` or it never reaches the picker")
        XCTAssertEqual(ImageModelPreset.sd15.variant, .sd1)
        XCTAssertEqual(ImageModelPreset.sd15.configName, "sd1")
    }

    func testSd15RunsRealGuidance() {
        // Same class of backend as SDXL base — real CFG, not a distill.
        XCTAssertTrue(ImageModelPreset.sd15.supportsNegativePrompt)
        XCTAssertTrue(ImageModelPreset.sd15.supportsGuidance)
        XCTAssertFalse(ImageModelPreset.sd15.stepsAreFixed)
    }

    func testSd15ResolutionGridMirrorsTheServersClampSd1Dim() {
        // gen.clampSd1Dim: multiple of 64, [256, 1536].
        let grid = ImageModelPreset.sd15.resolutionGrid
        XCTAssertEqual(grid.alignment, 64)
        XCTAssertEqual(grid.minDim, 256)
        XCTAssertEqual(grid.maxDim, 1536)
        // The grid must not just equal SDXL's — SD 1.x's floor is lower.
        XCTAssertNotEqual(grid.minDim, ImageModelPreset.sdxlBase10.resolutionGrid.minDim)
        for r in ImageModelPreset.sd15.resolutions {
            XCTAssertEqual(r.width % 64, 0, "\(r.width) is not a multiple of 64")
            XCTAssertEqual(r.height % 64, 0, "\(r.height) is not a multiple of 64")
        }
        XCTAssertTrue(ImageModelPreset.sd15.resolutions.contains(ImageModelPreset.sd15.defaultResolution))
    }

    func testSd15DownloadsThroughItsOwnBundleNotSdxlsAllotment() {
        let bundle = ImageModelPreset.sd15.bundle
        XCTAssertTrue(bundle.id.hasPrefix("sd1:"), "expected the sd1Diffusers bundle, got \(bundle.id)")
        let markers = bundle.components[0].readyMarkers
        XCTAssertTrue(markers.contains("text_encoder"))
        // The load-bearing difference from `.sdxlDiffusers`: no second tower to
        // wait for, or a complete SD 1.5 download would read as incomplete
        // forever (the class this test exists to catch).
        XCTAssertFalse(markers.contains("text_encoder_2"),
                       "SD 1.x has no text_encoder_2 — requiring it makes a complete download unreadable")
        XCTAssertFalse(markers.contains("tokenizer_2"))
    }

    // MARK: - SD-Turbo

    /// SD-Turbo is an SD 2.1 distill (OpenCLIP-H tower), NOT an SD 1.5 one —
    /// it needs its OWN `FluxVariant` case rather than riding `.sd1`,
    /// because `.sd1`'s guidance/negative-prompt flags are true and Turbo is
    /// guidance-free (same reasoning as SDXL Turbo beside SDXL base).
    func testSdTurboIsInTheCuratedCatalogAndGuidanceFree() {
        XCTAssertTrue(ImageModelPreset.all.contains(.sdTurbo))
        XCTAssertEqual(ImageModelPreset.sdTurbo.variant, .sdTurbo)
        XCTAssertNotEqual(ImageModelPreset.sdTurbo.variant, .sd1,
                          "Turbo must not ride SD 1.5's variant or it inherits a negative-prompt box it can't use")
        XCTAssertFalse(ImageModelPreset.sdTurbo.supportsNegativePrompt)
        XCTAssertFalse(ImageModelPreset.sdTurbo.supportsGuidance)
        XCTAssertLessThanOrEqual(ImageModelPreset.sdTurbo.settings(.good).steps, 4)
    }

    func testSdTurboSharesSd15sResolutionGridAndBundleShape() {
        XCTAssertEqual(ImageModelPreset.sdTurbo.resolutionGrid.alignment,
                       ImageModelPreset.sd15.resolutionGrid.alignment)
        XCTAssertEqual(ImageModelPreset.sdTurbo.resolutionGrid.minDim,
                       ImageModelPreset.sd15.resolutionGrid.minDim)
        let bundle = ImageModelPreset.sdTurbo.bundle
        XCTAssertTrue(bundle.id.hasPrefix("sd1:"), "expected the sd1Diffusers bundle, got \(bundle.id)")
        XCTAssertFalse(bundle.components[0].readyMarkers.contains("text_encoder_2"),
                       "SD-Turbo has no text_encoder_2 either — one OpenCLIP-H tower")
    }
}
