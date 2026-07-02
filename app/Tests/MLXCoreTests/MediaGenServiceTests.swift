import XCTest
import AVFoundation
@testable import MLXCore

/// Tests for the unified media-generation path: image/audio/video now run
/// through the ONE main `mlx-serve` server (registry-hosted) instead of a
/// dedicated `NativeGenServer` subprocess. Covers the pure response-decode
/// contracts + the load→generate→unload residency default.
@MainActor
final class MediaGenServiceTests: XCTestCase {

    // MARK: - Image response decode (the /v1/images/generations contract)

    func testDecodePngB64ExtractsImage() throws {
        let pngBytes = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x01, 0x02])
        let b64 = pngBytes.base64EncodedString()
        let body = try JSONSerialization.data(withJSONObject: ["data": [["b64_json": b64]]])
        let decoded = ImageGenService.decodePngB64(body)
        XCTAssertEqual(decoded, pngBytes)
    }

    func testDecodePngB64RejectsMalformed() {
        XCTAssertNil(ImageGenService.decodePngB64(Data("not json".utf8)))
        let noData = try! JSONSerialization.data(withJSONObject: ["error": "boom"])
        XCTAssertNil(ImageGenService.decodePngB64(noData))
        let emptyArr = try! JSONSerialization.data(withJSONObject: ["data": []])
        XCTAssertNil(ImageGenService.decodePngB64(emptyArr))
    }

    // MARK: - Video request body (the /v1/video/generations REQUEST contract)

    func testRequestBodyTwoStageCarriesPipelineAndGuidance() {
        // The confirmed bug: pipeline/cfg_scale/stg_scale were modeled in the UI
        // (VideoPipelineMode, VideoGenRequest) but never put in the HTTP body —
        // the Quality preset (cfg 3.0, twoStage) silently ran as unguided
        // one-stage. This pins the full wire shape for a .twoStage request.
        var req = VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                  numFrames: 97, fps: 24, mode: .twoStage, steps: 30, cfgScale: 3.0)
        req.stgScale = 1.0
        let body = VideoGenService.requestBody(model: "ltx", prompt: "a prompt", request: req, firstFrameB64: nil)
        XCTAssertEqual(body["pipeline"] as? String, "two_stage")
        XCTAssertEqual(body["cfg_scale"] as? Double, 3.0)
        XCTAssertEqual(body["stg_scale"] as? Double, 1.0)
        XCTAssertEqual(body["steps"] as? Int, 30)
        // The pre-existing fields keep their shape.
        XCTAssertEqual(body["model"] as? String, "ltx")
        XCTAssertEqual(body["prompt"] as? String, "a prompt")
        XCTAssertEqual(body["num_frames"] as? Int, 97)
        XCTAssertEqual(body["height"] as? Int, 480)
        XCTAssertEqual(body["width"] as? Int, 704)
        XCTAssertEqual(body["seed"] as? Int, 42)
        // first_frame_image stays conditional — absent when there's no image.
        XCTAssertNil(body["first_frame_image"])
    }

    func testRequestBodyPipelineModeMapping() {
        func pipeline(_ mode: VideoPipelineMode) -> String? {
            let req = VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                      numFrames: 9, fps: 24, mode: mode, steps: 8, cfgScale: 1.0)
            return VideoGenService.requestBody(model: "m", prompt: "p", request: req, firstFrameB64: nil)["pipeline"] as? String
        }
        XCTAssertEqual(pipeline(.oneStage), "one_stage")
        XCTAssertEqual(pipeline(.twoStage), "two_stage")
        XCTAssertEqual(pipeline(.twoStageHQ), "two_stage_hq")
    }

    func testRequestBodyIncludesFirstFrameWhenPresent() {
        let req = VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                  numFrames: 9, fps: 24, mode: .oneStage, steps: 8, cfgScale: 1.0)
        let body = VideoGenService.requestBody(model: "m", prompt: "p", request: req, firstFrameB64: "QUJD")
        XCTAssertEqual(body["first_frame_image"] as? String, "QUJD")
    }

    // MARK: - Video response decode (the /v1/video/generations contract)

    func testDecodeFramesParsesRgb8Body() {
        let frames = 2, w = 2, h = 2
        let rgb = Data(repeating: 7, count: frames * w * h * 3)
        let obj: [String: Any] = [
            "format": "rgb8", "frames": frames, "height": h, "width": w,
            "fps": 24, "data": rgb.base64EncodedString(),
        ]
        let decoded = VideoGenService.decodeFrames(obj)
        XCTAssertEqual(decoded?.frames, frames)
        XCTAssertEqual(decoded?.rgb.count, frames * w * h * 3)
    }

    func testDecodeFramesRejectsSizeMismatch() {
        // rgb byte count must equal frames*h*w*3, else the body is corrupt.
        let obj: [String: Any] = [
            "format": "rgb8", "frames": 2, "height": 2, "width": 2,
            "data": Data(repeating: 1, count: 8).base64EncodedString(),  // wrong size
        ]
        XCTAssertNil(VideoGenService.decodeFrames(obj))
    }

    func testDecodeFramesParsesOptionalAudioTrack() {
        let frames = 2, w = 2, h = 2
        let rgb = Data(repeating: 7, count: frames * w * h * 3)
        let pcm = Data(repeating: 3, count: 320 * 2 * 2)  // 320 stereo frames, s16le
        let obj: [String: Any] = [
            "format": "rgb8", "frames": frames, "height": h, "width": w, "fps": 24,
            "data": rgb.base64EncodedString(),
            "audio_format": "pcm_s16le", "audio_sample_rate": 16000, "audio_channels": 2,
            "audio_data": pcm.base64EncodedString(),
        ]
        let decoded = VideoGenService.decodeFrames(obj)
        XCTAssertEqual(decoded?.audioPCM?.count, pcm.count)
        XCTAssertEqual(decoded?.audioSampleRate, 16000)
        XCTAssertEqual(decoded?.audioChannels, 2)
    }

    func testDecodeFramesAudioAbsentLeavesPcmNil() {
        // A video-only body (no audio fields) must still decode, with no audio.
        let obj: [String: Any] = [
            "format": "rgb8", "frames": 1, "height": 2, "width": 2, "fps": 24,
            "data": Data(repeating: 7, count: 12).base64EncodedString(),
        ]
        let decoded = VideoGenService.decodeFrames(obj)
        XCTAssertNotNil(decoded)
        XCTAssertNil(decoded?.audioPCM)
    }

    func testWriteMP4WithAudioProducesAnAudioTrack() async throws {
        let frames = 3, w = 16, h = 16, fps = 24
        let rgb = Data(repeating: 120, count: frames * w * h * 3)
        // 0.25s of a quiet tone, 16 kHz stereo s16le.
        let sr = 16000, ch = 2, nFrames = sr / 4
        var pcm = Data(count: nFrames * ch * 2)
        pcm.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: Int16.self)
            for i in 0..<nFrames {
                let v = Int16(2000.0 * sin(Double(i) * 0.2))
                p[i * 2] = v; p[i * 2 + 1] = v
            }
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mlxserve-audiomux-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }
        try VideoGenService.writeMP4(rgb: rgb, frames: frames, width: w, height: h, fps: fps, to: url,
                                     audioPCM: pcm, audioSampleRate: sr, audioChannels: ch)

        let asset = AVURLAsset(url: url)
        let videoTracks = try await asset.loadTracks(withMediaType: .video)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        XCTAssertEqual(videoTracks.count, 1, "expected one video track")
        XCTAssertEqual(audioTracks.count, 1, "audio track missing — mux did not add sound")
    }

    func testWriteMP4WithAudioDoesNotDeadlockAtRealisticScale() throws {
        // A multi-input AVAssetWriter deadlocks when every video frame is pushed
        // before any audio: the muxer stops accepting video (isReadyForMoreMediaData
        // stays false) to bound how far video can lead the still-empty audio track,
        // while the audio is only appended AFTER the video loop — which never ends.
        // Toy-scale clips (a few tiny frames) stay under the muxer's backpressure
        // window and falsely pass, so this reproduces at the ~97-frame scale a real
        // LTX clip hits. A deadlock surfaces here as a wait() timeout, not a hang.
        let frames = 97, w = 256, h = 256, fps = 24
        let rgb = Data(repeating: 120, count: frames * w * h * 3)
        let sr = 16000, ch = 2, nAudio = sr * frames / fps
        var pcm = Data(count: nAudio * ch * 2)
        pcm.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: Int16.self)
            for i in 0..<nAudio {
                let v = Int16(2000.0 * sin(Double(i) * 0.2))
                p[i * 2] = v; p[i * 2 + 1] = v
            }
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mlxserve-deadlock-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }

        let done = expectation(description: "writeMP4 completes (no mux deadlock)")
        let muxError = MuxErrorBox()
        Thread.detachNewThread {
            do {
                try VideoGenService.writeMP4(rgb: rgb, frames: frames, width: w, height: h, fps: fps, to: url,
                                             audioPCM: pcm, audioSampleRate: sr, audioChannels: ch)
            } catch { muxError.value = error }
            done.fulfill()
        }
        wait(for: [done], timeout: 30)
        XCTAssertNil(muxError.value, "writeMP4 threw while muxing audio")
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), "no mp4 written")
    }

    /// Tiny boxed error holder so the detached mux thread can hand a failure back
    /// to the (sendable-checked) test closure.
    private final class MuxErrorBox: @unchecked Sendable { var value: Error? }

    func testWriteMP4WithSubFramePCMCompletesAtRealisticScale() throws {
        // A non-empty PCM payload smaller than one audio frame (3 bytes < the
        // 4-byte stereo s16 frame) yields zero appendable frames. appendAudio's
        // `guard numFrames > 0 else { return }` used to bail WITHOUT marking the
        // audio input finished — leaving a starved, never-finished sibling input
        // that wedges the video loop (same multi-input AVAssetWriter
        // backpressure class as the append-order deadlock above). Realistic
        // frame count so the backpressure window is actually exceeded; a
        // deadlock surfaces as a wait() timeout, not a hang.
        let frames = 97, w = 256, h = 256, fps = 24
        let rgb = Data(repeating: 120, count: frames * w * h * 3)
        let pcm = Data([1, 2, 3])  // non-empty, sub-frame
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mlxserve-subframe-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }

        let done = expectation(description: "writeMP4 completes with sub-frame PCM")
        let muxError = MuxErrorBox()
        Thread.detachNewThread {
            do {
                try VideoGenService.writeMP4(rgb: rgb, frames: frames, width: w, height: h, fps: fps, to: url,
                                             audioPCM: pcm, audioSampleRate: 16000, audioChannels: 2)
            } catch { muxError.value = error }
            done.fulfill()
        }
        wait(for: [done], timeout: 30)
        XCTAssertNil(muxError.value, "writeMP4 threw on sub-frame PCM")
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), "no mp4 written")
    }

    func testWriteMP4WithZeroAudioChannelsSkipsAudio() async throws {
        // audio_channels is SERVER-controlled: 0 must not divide-by-zero
        // (bytesPerFrame = 2 * channels) or wedge the mux — the audio input is
        // skipped entirely for invalid channels/sampleRate.
        let frames = 3, w = 16, h = 16
        let rgb = Data(repeating: 90, count: frames * w * h * 3)
        let pcm = Data(repeating: 1, count: 3200)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mlxserve-zerochan-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }
        try VideoGenService.writeMP4(rgb: rgb, frames: frames, width: w, height: h, fps: 24, to: url,
                                     audioPCM: pcm, audioSampleRate: 16000, audioChannels: 0)
        let asset = AVURLAsset(url: url)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        XCTAssertEqual(audioTracks.count, 0, "invalid channel count must skip the audio track")
    }

    func testDecodeFramesDropsAudioWithInvalidChannels() {
        // Same server-controlled field at the decode layer: a body claiming
        // audio_channels 0 parses (video is fine) but the audio is dropped.
        let obj: [String: Any] = [
            "format": "rgb8", "frames": 1, "height": 2, "width": 2, "fps": 24,
            "data": Data(repeating: 7, count: 12).base64EncodedString(),
            "audio_format": "pcm_s16le", "audio_sample_rate": 16000, "audio_channels": 0,
            "audio_data": Data(repeating: 3, count: 64).base64EncodedString(),
        ]
        let decoded = VideoGenService.decodeFrames(obj)
        XCTAssertNotNil(decoded)
        XCTAssertNil(decoded?.audioPCM, "audio with 0 channels must be dropped")
    }

    func testWriteMP4WithoutAudioHasNoAudioTrack() async throws {
        let frames = 2, w = 16, h = 16
        let rgb = Data(repeating: 90, count: frames * w * h * 3)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mlxserve-noaudio-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }
        try VideoGenService.writeMP4(rgb: rgb, frames: frames, width: w, height: h, fps: 24, to: url)
        let asset = AVURLAsset(url: url)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        XCTAssertEqual(audioTracks.count, 0)
    }

    // MARK: - Model resolution (moved from NativeGenServer to ServerManager)

    func testResolveModelDirMissingRepoIsNil() {
        XCTAssertNil(ServerManager.resolveModelDir(repo: "nonexistent-owner/definitely-not-a-real-model-xyz"))
    }

    // MARK: - Residency default

    func testKeepResidentDefaultsOff() {
        // Decision: load→generate→unload by default; "Keep loaded" is opt-in.
        let img = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "x", width: 1024, height: 1024, steps: 4, guidance: 0.5)
        XCTAssertFalse(img.keepResident)
        let vid = VideoGenRequest(model: .ltx23Q4, prompt: "x", width: 384, height: 256, numFrames: 9, fps: 24, mode: .oneStage, steps: 6, cfgScale: 1.0)
        XCTAssertFalse(vid.keepResident)
        let aud = AudioGenRequest(model: .qwen3TTS06B, text: "x")
        XCTAssertFalse(aud.keepResident)
    }
}
