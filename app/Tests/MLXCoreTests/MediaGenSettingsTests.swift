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
        s.modelId = "mflux/dev-q4"
        s.quality = .superQuality
        s.resolutionId = "1216x832"
        s.steps = 33
        s.guidance = 4.5
        s.seed = 7
        s.negativePrompt = "blurry, low quality"
        s.safeMode = false
        s.keepResident = true
        let decoded = try JSONDecoder().decode(ImageGenSettings.self, from: try JSONEncoder().encode(s))
        XCTAssertEqual(decoded, s)
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
        let decoded = try JSONDecoder().decode(VideoGenSettings.self, from: try JSONEncoder().encode(s))
        XCTAssertEqual(decoded, s)
    }

    // MARK: - Reconstruct preset / resolution by id

    func testImageResolvesModelById() {
        var s = ImageGenSettings()
        s.modelId = "mflux/dev-q4"
        XCTAssertEqual(s.resolvedModel.id, "mflux/dev-q4")
    }

    func testImageResolvesResolutionById() {
        var s = ImageGenSettings()
        s.resolutionId = "1216x832"
        let m = ImageModelPreset.devQ4
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
        let m = ImageModelPreset.devQ4
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
        XCTAssertEqual(s.resolvedResolution(for: m).id, m.defaultResolution.id)
    }

    // MARK: - Migration-safe decode: a missing key defaults, never throws

    func testImageMigrationSafeDecodeDropsKey() throws {
        var obj = try JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(ImageGenSettings())) as! [String: Any]
        obj.removeValue(forKey: "steps")
        obj.removeValue(forKey: "safeMode")
        let decoded = try JSONDecoder().decode(
            ImageGenSettings.self, from: try JSONSerialization.data(withJSONObject: obj))
        XCTAssertEqual(decoded.steps, ImageGenSettings().steps)
        XCTAssertEqual(decoded.safeMode, ImageGenSettings().safeMode)
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
}
