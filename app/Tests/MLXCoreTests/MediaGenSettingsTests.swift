import XCTest
@testable import MLXCore

/// Sticky generation settings: each of the three media panels persists its
/// last-used controls to UserDefaults (Codable JSON, migration-safe decode) and
/// reconstructs the non-Codable presets / resolutions by stable `id`. Mirrors
/// the `ServerOptions` / `MediaBundle` persistence contract.
final class MediaGenSettingsTests: XCTestCase {

    // MARK: - Round-trip encode/decode equality

    func testImageSettingsRoundTrips() throws {
        var s = ImageGenSettings()
        s.modelId = "mflux/flux2-klein-4b-q4"
        s.quality = .superQuality
        s.resolutionId = "1216x832"
        s.steps = 33
        s.seed = 7
        s.keepResident = true
        let decoded = try JSONDecoder().decode(ImageGenSettings.self, from: try JSONEncoder().encode(s))
        XCTAssertEqual(decoded, s)
    }

    /// Settings saved by an older build still carry `guidance`, `negativePrompt`
    /// and `safeMode` — all retired. The tolerant decoder must ignore the
    /// leftovers rather than throwing, or every existing user's image settings
    /// reset on upgrade.
    func testImageSettingsIgnoresRetiredKeysFromOlderBuilds() throws {
        let legacy = Data("""
        {"modelId":"mflux/flux2-klein-4b-q4","quality":"Quality","resolutionId":"1216x832",
         "steps":33,"guidance":4.5,"seed":7,"negativePrompt":"blurry","safeMode":false,
         "keepResident":true,"strength":0.4,"editMode":false}
        """.utf8)
        let s = try JSONDecoder().decode(ImageGenSettings.self, from: legacy)
        XCTAssertEqual(s.steps, 33)
        XCTAssertEqual(s.seed, 7)
        XCTAssertEqual(s.resolutionId, "1216x832")
        XCTAssertTrue(s.keepResident)
        XCTAssertEqual(s.strength, 0.4)
        XCTAssertFalse(s.editMode)
    }

    func testAudioSettingsRoundTrips() throws {
        var s = AudioGenSettings()
        s.modelId = "mlx-audio/qwen3-tts-1.7b-base"
        s.speed = 1.25
        s.temperature = 0.9
        s.keepResident = true
        let decoded = try JSONDecoder().decode(AudioGenSettings.self, from: try JSONEncoder().encode(s))
        XCTAssertEqual(decoded, s)
    }

    func testVideoSettingsRoundTrips() throws {
        var s = VideoGenSettings()
        s.quality = .quality
        s.resolutionId = "768x512"
        s.numFrames = 49
        s.fps = 24
        s.mode = .twoStage
        s.steps = 30
        s.cfgScale = 3.0
        s.stgScale = 1.0
        s.seed = 99
        s.keepResident = true
        // REGRESSION: the tolerant decoder listed every field EXCEPT
        // bestQuality, so H3's "Max quality" toggle silently reset to off on
        // every app relaunch (saved, then dropped on load).
        s.bestQuality = true
        let decoded = try JSONDecoder().decode(VideoGenSettings.self, from: try JSONEncoder().encode(s))
        XCTAssertEqual(decoded, s)
    }

    /// A persisted LAN pick ("lan:<model>@<peer>") whose base id matches a
    /// local preset resolves to THAT preset, not the LTX fallback — the pane
    /// gates ladders, resolutions and request fields on `resolvedModel`, so
    /// falling back to LTX sent a remote H3 off-canvas sizes and frame counts
    /// below its trained floor (the cross-pollution class).
    func testVideoResolvesLanModelToItsMatchingPreset() {
        var s = VideoGenSettings()
        s.modelId = "lan:" + VideoModelPreset.minimaxH3.id + "@studio"
        XCTAssertEqual(s.resolvedModel.id, VideoModelPreset.minimaxH3.id)
        // An unknown remote model keeps today's LTX fallback.
        s.modelId = "lan:someone/custom-video@studio"
        XCTAssertEqual(s.resolvedModel.id, VideoModelPreset.ltx23Q4.id)
    }

    // MARK: - Reconstruct preset / resolution by id

    func testImageResolvesModelById() {
        var s = ImageGenSettings()
        s.modelId = "krea/krea-2-turbo-mlx-serve"
        XCTAssertEqual(s.resolvedModel.id, "krea/krea-2-turbo-mlx-serve")
    }

    func testImageResolvesResolutionById() {
        var s = ImageGenSettings()
        s.resolutionId = "1216x832"
        let m = ImageModelPreset.flux2Klein4B_Q4
        XCTAssertEqual(s.resolvedResolution(for: m).id, "1216x832")
        XCTAssertEqual(s.resolvedResolution(for: m).width, 1216)
        XCTAssertEqual(s.resolvedResolution(for: m).height, 832)
    }

    func testAudioResolvesModelById() {
        var s = AudioGenSettings()
        s.modelId = "mlx-audio/qwen3-tts-1.7b-base"
        XCTAssertEqual(s.resolvedModel.id, "mlx-audio/qwen3-tts-1.7b-base")
    }

    func testVideoResolvesModelAndResolutionById() {
        var s = VideoGenSettings()
        s.modelId = VideoModelPreset.ltx23Q4.id
        s.resolutionId = "768x512"
        XCTAssertEqual(s.resolvedModel.id, VideoModelPreset.ltx23Q4.id)
        XCTAssertEqual(s.resolvedResolution(for: s.resolvedModel).id, "768x512")
    }

    // MARK: - Unknown id falls back to the preset default

    func testImageUnknownModelFallsBackToDefault() {
        var s = ImageGenSettings()
        s.modelId = "does/not-exist"
        XCTAssertEqual(s.resolvedModel.id, ImageModelPreset.flux2Klein4B_Q4.id)
    }

    func testImageUnknownResolutionFallsBackToModelDefault() {
        var s = ImageGenSettings()
        s.resolutionId = "9999x9999"
        let m = ImageModelPreset.flux2Klein4B_Q4
        XCTAssertEqual(s.resolvedResolution(for: m).id, m.defaultResolution.id)
    }

    func testAudioUnknownModelFallsBackToDefault() {
        var s = AudioGenSettings()
        s.modelId = "nope/nope"
        XCTAssertEqual(s.resolvedModel.id, AudioModelPreset.qwen3TTS06B8bit.id)
    }

    func testVideoUnknownModelAndResolutionFallBack() {
        var s = VideoGenSettings()
        s.modelId = "nope/nope"
        s.resolutionId = "1x1"
        XCTAssertEqual(s.resolvedModel.id, VideoModelPreset.ltx23Q4.id)
        let m = s.resolvedModel
        // With nothing saved the canvas is sized for THIS Mac, not a static
        // default — but it is always a rung the picker offers.
        let r = s.resolvedResolution(for: m)
        XCTAssertEqual(r.id, m.recommendedResolution(totalGB: RAMChecker.totalGB).id)
        XCTAssertTrue(m.resolutions.contains(r))
    }

    // MARK: - Migration-safe decode: a missing key defaults, never throws

    func testImageMigrationSafeDecodeDropsKey() throws {
        var obj = try JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(ImageGenSettings())) as! [String: Any]
        obj.removeValue(forKey: "steps")
        let decoded = try JSONDecoder().decode(
            ImageGenSettings.self, from: try JSONSerialization.data(withJSONObject: obj))
        XCTAssertEqual(decoded.steps, ImageGenSettings().steps)
    }

    func testAudioMigrationSafeDecodeDropsKey() throws {
        var obj = try JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(AudioGenSettings())) as! [String: Any]
        obj.removeValue(forKey: "temperature")
        let decoded = try JSONDecoder().decode(
            AudioGenSettings.self, from: try JSONSerialization.data(withJSONObject: obj))
        XCTAssertEqual(decoded.temperature, AudioGenSettings().temperature)
    }

    func testVideoMigrationSafeDecodeDropsKey() throws {
        var obj = try JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(VideoGenSettings())) as! [String: Any]
        obj.removeValue(forKey: "mode")
        obj.removeValue(forKey: "numFrames")
        let decoded = try JSONDecoder().decode(
            VideoGenSettings.self, from: try JSONSerialization.data(withJSONObject: obj))
        XCTAssertEqual(decoded.mode, VideoGenSettings().mode)
        XCTAssertEqual(decoded.numFrames, VideoGenSettings().numFrames)
    }

    // MARK: - 3D settings (Hunyuan3D pane)

    func testModel3DSettingsRoundTrips() throws {
        var s = Model3DGenSettings()
        s.steps = 45
        s.guidance = 6.5
        s.resolution = 128
        s.keepResident = true
        s.turntable = false
        let decoded = try JSONDecoder().decode(Model3DGenSettings.self, from: try JSONEncoder().encode(s))
        XCTAssertEqual(decoded, s)
    }

    func testModel3DLegacy380ResolutionMigratesTo384() throws {
        // Pre-FlashVDM builds persisted a "380 (fine)" picker option; the pane
        // now offers the reference-default 384 — decode migrates 380 so the
        // segmented picker never comes up with no selected option.
        var s = Model3DGenSettings()
        s.resolution = 380
        let decoded = try JSONDecoder().decode(Model3DGenSettings.self, from: try JSONEncoder().encode(s))
        XCTAssertEqual(decoded.resolution, 384)
    }

    func testModel3DResolvesModelByIdAndFallsBack() {
        var s = Model3DGenSettings()
        s.modelId = Model3DModelPreset.hunyuan3d21_8bit.id
        XCTAssertEqual(s.resolvedModel.id, Model3DModelPreset.hunyuan3d21_8bit.id)
        s.modelId = "nope/nope"
        XCTAssertEqual(s.resolvedModel.id, Model3DModelPreset.hunyuan3d21_8bit.id)
    }

    func testModel3DDefaultsMatchThePreset() {
        // The pane's defaults must line up with the request contract: 30 steps,
        // guidance 5.0, octree 384 (reference default — affordable since the
        // FlashVDM hierarchical volume decode), turntable on, keep-loaded off.
        let s = Model3DGenSettings()
        XCTAssertEqual(s.steps, 30)
        XCTAssertEqual(s.guidance, 5.0)
        XCTAssertEqual(s.resolution, 384)
        XCTAssertTrue(s.turntable)
        XCTAssertFalse(s.keepResident)
    }

    func testModel3DMigrationSafeDecodeDropsKey() throws {
        var obj = try JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(Model3DGenSettings())) as! [String: Any]
        obj.removeValue(forKey: "resolution")
        obj.removeValue(forKey: "turntable")
        let decoded = try JSONDecoder().decode(
            Model3DGenSettings.self, from: try JSONSerialization.data(withJSONObject: obj))
        XCTAssertEqual(decoded.resolution, Model3DGenSettings().resolution)
        XCTAssertEqual(decoded.turntable, Model3DGenSettings().turntable)
    }

    // MARK: - Resizable prompt editor

    /// A dragged height is persisted, so a value from a different window size —
    /// or a garbage one — must never leave the editor unusable or off-screen.
    func testPromptEditorHeightIsClampedBothWays() {
        XCTAssertEqual(PromptEditorHeight.clamp(4000), PromptEditorHeight.maxHeight)
        XCTAssertEqual(PromptEditorHeight.clamp(0), PromptEditorHeight.minHeight)
        XCTAssertEqual(PromptEditorHeight.clamp(-50), PromptEditorHeight.minHeight)
        XCTAssertEqual(PromptEditorHeight.clamp(.nan), PromptEditorHeight.defaultHeight)
        XCTAssertEqual(PromptEditorHeight.clamp(240), 240)
    }

    func testVideoSettingsClampAStalePromptHeightOnDecode() throws {
        var obj = try JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(VideoGenSettings())) as! [String: Any]
        obj["promptHeight"] = 9999.0
        let decoded = try JSONDecoder().decode(
            VideoGenSettings.self, from: try JSONSerialization.data(withJSONObject: obj))
        XCTAssertEqual(decoded.promptHeight, PromptEditorHeight.maxHeight)
        // Absent key = an older build's blob → the default, not zero.
        obj.removeValue(forKey: "promptHeight")
        let old = try JSONDecoder().decode(
            VideoGenSettings.self, from: try JSONSerialization.data(withJSONObject: obj))
        XCTAssertEqual(old.promptHeight, PromptEditorHeight.defaultHeight)
    }
}
