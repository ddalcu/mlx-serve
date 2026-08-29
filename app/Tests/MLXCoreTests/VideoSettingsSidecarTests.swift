import XCTest
@testable import MLXCore

/// The `<clip>.txt` prompt/settings sidecar written beside each generated video.
final class VideoSettingsSidecarTests: XCTestCase {
    func testSidecarDocumentsPromptRequestAndActualOutput() {
        var req = VideoGenRequest(
            model: .ltx25Q8,
            prompt: "  A lighthouse in a winter storm.  \n",
            seed: 123,
            width: 704,
            height: 480,
            numFrames: 97,
            fps: 24,
            mode: .twoStage,
            steps: 30,
            cfgScale: 3.0)
        req.stgScale = 1.0
        req.diffusionDecoder = true
        req.firstFrameImagePath = "/private/reference/start.png"
        req.lastFrameImagePath = "/private/reference/end.png"
        req.audioPath = "/private/reference/dialogue.m4a"
        req.loras = [LoraAdapter(path: "/private/loras/cinematic.safetensors", scale: 0.75)]

        let text = VideoGenService.settingsText(
            req, modelId: "remote/ltx-model",
            outputWidth: 960, outputHeight: 544,
            outputFrames: 105, outputFps: 25)

        XCTAssertTrue(text.contains("model: remote/ltx-model"))
        XCTAssertTrue(text.contains("preset: \(req.model.name)"))
        XCTAssertTrue(text.contains("seed: 123"))
        XCTAssertTrue(text.contains("width: 704"))
        XCTAssertTrue(text.contains("height: 480"))
        XCTAssertTrue(text.contains("frames: 97"))
        XCTAssertTrue(text.contains("fps: 24"))
        XCTAssertTrue(text.contains("steps: 30"))
        XCTAssertTrue(text.contains("output_width: 960"))
        XCTAssertTrue(text.contains("output_height: 544"))
        XCTAssertTrue(text.contains("output_frames: 105"))
        XCTAssertTrue(text.contains("output_fps: 25"))
        XCTAssertTrue(text.contains("pipeline: two_stage"))
        XCTAssertTrue(text.contains("cfg_scale: 3.00"))
        XCTAssertTrue(text.contains("stg_scale: 1.00"))
        XCTAssertTrue(text.contains("decoder: diffusion"))
        XCTAssertTrue(text.contains("first_frame: start.png"))
        XCTAssertTrue(text.contains("last_frame: end.png"))
        XCTAssertTrue(text.contains("input_audio: dialogue.m4a"))
        XCTAssertTrue(text.contains("lora_1_file: cinematic.safetensors"))
        XCTAssertTrue(text.contains("lora_1_scale: 0.75"))
        XCTAssertTrue(text.contains("# Prompt\nA lighthouse in a winter storm.\n"))
        XCTAssertFalse(text.contains("/private/reference"), "store portable basenames, not private paths")
        XCTAssertFalse(text.contains("/private/loras"), "store portable basenames, not private paths")
    }

    func testMatchingOutputDoesNotAddRedundantOutputFields() {
        let req = VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                  numFrames: 97, fps: 24, mode: .oneStage,
                                  steps: 8, cfgScale: 1.0)
        let text = VideoGenService.settingsText(
            req, modelId: "m", outputWidth: 704, outputHeight: 480,
            outputFrames: 97, outputFps: 24)
        XCTAssertFalse(text.contains("output_width:"))
        XCTAssertFalse(text.contains("output_height:"))
        XCTAssertFalse(text.contains("output_frames:"))
        XCTAssertFalse(text.contains("output_fps:"))
    }

    func testAudioUpgradeRecordsEffectivePipelineAndOmitsDroppedGuidance() {
        var req = VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                  numFrames: 97, fps: 24, mode: .oneStage,
                                  steps: 8, cfgScale: 1.0)
        req.stgScale = 0.5
        req.audioPath = "/clips/voice.wav"
        let text = VideoGenService.settingsText(req, modelId: "m")
        XCTAssertTrue(text.contains("pipeline: two_stage"))
        XCTAssertTrue(text.contains("input_audio: voice.wav"))
        XCTAssertFalse(text.contains("cfg_scale:"), "one-stage guidance is not sent after the audio upgrade")
        XCTAssertFalse(text.contains("stg_scale:"), "one-stage guidance is not sent after the audio upgrade")
    }

    func testMultipleReferencesRemainOrderedAndUseBasenames() {
        var req = VideoGenRequest(model: .minimaxH3Ref2VA, prompt: "p", width: 544, height: 960,
                                  numFrames: 90, fps: 24, mode: .oneStage,
                                  steps: 30, cfgScale: 1.0)
        req.refImagePaths = ["/refs/character.png", "/refs/style.jpg"]
        req.refVideoPaths = ["/refs/walk.mp4", "/refs/camera.mov"]
        req.refAudioPaths = ["/refs/voice.wav", "/refs/music.m4a"]
        req.refImageSize = .max
        let text = VideoGenService.settingsText(req, modelId: "h3-ref")

        XCTAssertTrue(text.contains("reference_image_1: character.png"))
        XCTAssertTrue(text.contains("reference_image_2: style.jpg"))
        XCTAssertTrue(text.contains("reference_video_1: walk.mp4"))
        XCTAssertTrue(text.contains("reference_video_2: camera.mov"))
        XCTAssertTrue(text.contains("reference_audio_1: voice.wav"))
        XCTAssertTrue(text.contains("reference_audio_2: music.m4a"))
        XCTAssertTrue(text.contains("reference_image_size: max"))
        XCTAssertLessThan(text.range(of: "character.png")!.lowerBound,
                          text.range(of: "style.jpg")!.lowerBound)
        XCTAssertFalse(text.contains("/refs/"))
    }

    func testUnsupportedStaleSettingsAreNotDocumentedAsUsed() {
        var req = VideoGenRequest(model: .minimaxH3, prompt: "p", width: 960, height: 544,
                                  numFrames: 90, fps: 24, mode: .twoStageHQ,
                                  steps: 30, cfgScale: 7.0)
        req.stgScale = 2.0
        req.audioPath = "/clips/not-sent.wav"
        req.diffusionDecoder = true
        req.refImagePaths = ["/refs/not-sent.png"]
        req.refVideoPaths = ["/refs/not-sent.mp4"]
        req.refAudioPaths = ["/refs/not-sent.m4a"]
        req.refImageSize = .max
        let text = VideoGenService.settingsText(req, modelId: "h3")

        XCTAssertFalse(text.contains("pipeline:"))
        XCTAssertFalse(text.contains("cfg_scale:"))
        XCTAssertFalse(text.contains("stg_scale:"))
        XCTAssertFalse(text.contains("input_audio:"))
        XCTAssertFalse(text.contains("decoder:"))
        XCTAssertFalse(text.contains("reference_image_"))
        XCTAssertFalse(text.contains("reference_video_"))
        XCTAssertFalse(text.contains("reference_audio_"))
        XCTAssertTrue(text.contains("fast_recipe: true"))
        XCTAssertTrue(text.contains("turbo: false"))
        XCTAssertTrue(text.contains("chain_windows: 1"))
    }

    func testTurboRecordsFastRecipeAsDisabled() {
        var req = VideoGenRequest(model: .minimaxH3, prompt: "p", width: 960, height: 544,
                                  numFrames: 90, fps: 24, mode: .oneStage,
                                  steps: 4, cfgScale: 1.0)
        req.turbo = true
        let text = VideoGenService.settingsText(req, modelId: "h3")
        XCTAssertTrue(text.contains("turbo: true"))
        XCTAssertTrue(text.contains("fast_recipe: false"))
    }

    func testSidecarPathAndAtomicWrite() throws {
        XCTAssertEqual(VideoGenService.sidecarPath(forVideo: "/a/my.clip.v2.mp4"),
                       "/a/my.clip.v2.txt")

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("video-sidecar-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let videoPath = directory.appendingPathComponent("clip.mp4").path

        try VideoGenService.writeSettingsSidecar("settings\n", forVideo: videoPath)
        let written = try String(contentsOfFile: directory.appendingPathComponent("clip.txt").path,
                                 encoding: .utf8)
        XCTAssertEqual(written, "settings\n")
    }
}
