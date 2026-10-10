import XCTest
@testable import MLXCore

/// Sound Effects tab (Stable Audio 3): preset catalog, the
/// `/v1/audio/sound-generations` wire contract, the official repo's
/// config-less layout, routing, and sticky settings.
final class SoundGenTests: XCTestCase {

    func testRequestBodyClampsIntoTheModelsRangeAndResolvesTheSeed() {
        let p = SoundModelPreset.stableAudio3SmallSFX
        var req = SoundGenRequest(model: p, prompt: "door creak", durationSeconds: 500, steps: 99, seed: 7)
        var body = SoundGenService.requestBody(req, modelName: "m")
        XCTAssertEqual(body["model"] as? String, "m")
        XCTAssertEqual(body["prompt"] as? String, "door creak")
        XCTAssertEqual(body["duration_seconds"] as? Double, p.durationRange.upperBound)
        XCTAssertEqual(body["steps"] as? Int, p.stepsRange.upperBound)
        XCTAssertEqual(body["seed"] as? Int, 7)
        XCTAssertEqual(body["stream"] as? Bool, true)
        // No steps chosen = the server's default; a random seed is resolved
        // HERE so the sidecar can name it.
        req.steps = nil
        req.seed = -1
        req.durationSeconds = 0
        body = SoundGenService.requestBody(req, modelName: "m")
        XCTAssertNil(body["steps"])
        XCTAssertGreaterThanOrEqual(body["seed"] as? Int ?? -1, 0)
        XCTAssertEqual(body["duration_seconds"] as? Double, p.durationRange.lowerBound)
    }

    func testTheSidecarRecordsWhatTheServerWasSent() {
        let req = SoundGenRequest(model: .stableAudio3SmallSFX, prompt: " rain on a tin roof ", durationSeconds: 4.5, steps: 12, seed: 3)
        let txt = SoundGenService.settingsText(body: SoundGenService.requestBody(req, modelName: "m"))
        XCTAssertTrue(txt.contains("model: m"))
        XCTAssertTrue(txt.contains("duration_seconds: 4.5"))
        XCTAssertTrue(txt.contains("steps: 12"))
        XCTAssertTrue(txt.contains("seed: 3"))
        XCTAssertEqual(AudioSidecar.prompt(forTrack: "/nonexistent.wav"), "")
        XCTAssertTrue(txt.contains("# Style prompt\nrain on a tin roof\n"), "AudioSidecar reads the prompt back from this section")
    }

    func testTheOfficialRepoIsRecognizedFromModelConfigJson() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("sa3-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("t5gemma-b-b-ul2"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let cfg = #"{"model_type":"diffusion_cond_inpaint","model":{"conditioning":{"configs":[{"id":"prompt","type":"t5gemma"}]}}}"#
        try cfg.write(to: dir.appendingPathComponent("model_config.json"), atomically: true, encoding: .utf8)
        XCTAssertEqual(DownloadManager.markerModelType(inDir: dir.path), "stable_audio3")
        XCTAssertTrue(DownloadManager.holdsWeightLayout(dir.path))
        // The T5Gemma subdir is the completion marker, as on the server.
        XCTAssertFalse(DownloadManager.holdsCompleteMediaPack(dir.path))
        try Data().write(to: dir.appendingPathComponent("t5gemma-b-b-ul2/model.safetensors"))
        XCTAssertTrue(DownloadManager.holdsCompleteMediaPack(dir.path))
        // Another stable-audio-tools pipeline (Stable Audio Open's T5) is not ours.
        try #"{"model_type":"diffusion_cond","model":{"conditioning":{"configs":[{"type":"t5"}]}}}"#
            .write(to: dir.appendingPathComponent("model_config.json"), atomically: true, encoding: .utf8)
        XCTAssertNil(DownloadManager.markerModelType(inDir: dir.path))
    }

    func testTheBundleWaitsForTheTextEncoderAndSkipsTheThumbnail() {
        let b = SoundModelPreset.stableAudio3SmallSFX.bundle
        XCTAssertEqual(b.components.count, 1)
        let c = b.components[0]
        XCTAssertEqual(c.repo, "ddalcu/Stable-Audio-3-Small-SFX-MLX-Serve")
        for m in ["model_config.json", "model.safetensors", "t5gemma-b-b-ul2/model.safetensors", "t5gemma-b-b-ul2/tokenizer.json"] {
            XCTAssertTrue(c.readyMarkers.contains(m), m)
        }
        XCTAssertTrue(c.selection.recursive)
        XCTAssertTrue(c.selection.excludeSubstrings.contains(".png"))
    }

    func testStableAudioRoutesToTheSoundEffectsTab() {
        XCTAssertTrue(isMediaModelType("stable_audio3"))
        XCTAssertEqual(MediaModality(modelType: "stable_audio3"), .sound)
        XCTAssertEqual(MediaModality.sound.experiment, .audio)
        XCTAssertEqual(MediaModality.sound.audioTab, .sound)
        XCTAssertEqual(DownloadManager.requiredMediaMarker(modelType: "stable_audio3"), "t5gemma-b-b-ul2/model.safetensors")
    }

    func testSettingsRoundTripEveryFieldOffItsDefault() throws {
        var s = SoundGenSettings()
        s.modelId = "custom/sfx"
        s.prompt = "glass shatter"
        s.durationSeconds = 2.5
        s.steps = 16
        s.seed = 42
        s.keepResident = true
        s.showAdvanced = false
        let back = try JSONDecoder().decode(SoundGenSettings.self, from: JSONEncoder().encode(s))
        XCTAssertEqual(back, s)
        // A blob from before any key existed decodes to the defaults, never throws.
        XCTAssertEqual(try JSONDecoder().decode(SoundGenSettings.self, from: Data("{}".utf8)), SoundGenSettings())
    }
}
