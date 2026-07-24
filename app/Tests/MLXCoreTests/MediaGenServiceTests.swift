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

    func testRequestBodyCarriesLoraWhenSet() {
        var req = VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                  numFrames: 9, fps: 24, mode: .oneStage, steps: 8, cfgScale: 1.0)
        req.loraPath = "/tmp/style.safetensors"
        req.loraScale = 0.8
        let body = VideoGenService.requestBody(model: "m", prompt: "p", request: req, firstFrameB64: nil)
        XCTAssertEqual(body["lora_path"] as? String, "/tmp/style.safetensors")
        XCTAssertEqual(body["lora_scale"] as? Double, 0.8)
        // No LoRA → fields absent (a missing lora_path means detach server-side).
        req.loraPath = nil
        let bare = VideoGenService.requestBody(model: "m", prompt: "p", request: req, firstFrameB64: nil)
        XCTAssertNil(bare["lora_path"])
        XCTAssertNil(bare["lora_scale"])
    }

    func testCancelledErrorsMapToCancellationNotFailure() {
        // A user cancel surfaces from URLSession as URLError.cancelled, NOT
        // CancellationError — treating it as generic failure showed "Failed"
        // after every Cancel click.
        XCTAssertTrue(VideoGenService.isCancellation(CancellationError()))
        XCTAssertTrue(VideoGenService.isCancellation(URLError(.cancelled)))
        XCTAssertFalse(VideoGenService.isCancellation(URLError(.timedOut)))
        XCTAssertFalse(VideoGenService.isCancellation(APIError.badStatus(code: 500, detail: "x")))
    }

    func testResidencyEntryMatching() {
        // Discovered two-level id ("org/model") — the normal pull layout.
        XCTAssertTrue(VideoGenService.entryMatches(
            id: "dgrauet/ltx-2.3-mlx-q4", repo: "dgrauet/ltx-2.3-mlx-q4", dirBasename: "ltx-2.3-mlx-q4"))
        // Path-registered ids: absolute path or bare basename.
        XCTAssertTrue(VideoGenService.entryMatches(
            id: "/x/models/dgrauet/ltx-2.3-mlx-q4", repo: "dgrauet/ltx-2.3-mlx-q4", dirBasename: "ltx-2.3-mlx-q4"))
        XCTAssertTrue(VideoGenService.entryMatches(
            id: "ltx-2.3-mlx-q4", repo: "dgrauet/ltx-2.3-mlx-q4", dirBasename: "ltx-2.3-mlx-q4"))
        // A different model never matches.
        XCTAssertFalse(VideoGenService.entryMatches(
            id: "google/gemma-3-12b-it-4bit", repo: "dgrauet/ltx-2.3-mlx-q4", dirBasename: "ltx-2.3-mlx-q4"))
        XCTAssertFalse(VideoGenService.entryMatches(
            id: "org/other", repo: "dgrauet/ltx-2.3-mlx-q4", dirBasename: nil))
    }

    func testResidencyComputedFromModelsListNotProps() {
        // The live bug: GPU memory came from /props, which 503s on a headless
        // gen-only boot ("No default model configured") → "GPU memory 0 MB"
        // while 30 GB of LTX was resident. Residency must reduce over the
        // /v1/models snapshot (a no-model endpoint) instead.
        let ltx = APIClient.parseModelInfo([
            "id": "dgrauet/ltx-2.3-mlx-q4", "loaded": true,
            "bytes_resident": UInt64(31_801_302_892),
        ])
        let chatLoaded = APIClient.parseModelInfo([
            "id": "google/gemma-4-12b", "loaded": true,
            "bytes_resident": UInt64(8_000_000_000),
        ])
        let chatUnloaded = APIClient.parseModelInfo([
            "id": "google/gemma-3-4b", "loaded": false, "bytes_resident": UInt64(0),
        ])

        let r = VideoGenService.residency(
            from: [ltx, chatLoaded, chatUnloaded],
            repo: "dgrauet/ltx-2.3-mlx-q4", dirBasename: "ltx-2.3-mlx-q4")
        XCTAssertTrue(r.loaded)
        XCTAssertEqual(r.bytesResident, 31_801_302_892)
        // GPU total sums every LOADED entry — unloaded stubs contribute nothing.
        XCTAssertEqual(r.gpuResidentBytes, 39_801_302_892)

        // Pane's model absent from the registry → not loaded, but the total
        // still reports who holds the GPU.
        let miss = VideoGenService.residency(
            from: [chatLoaded], repo: "dgrauet/ltx-2.3-mlx-q4", dirBasename: "ltx-2.3-mlx-q4")
        XCTAssertFalse(miss.loaded)
        XCTAssertEqual(miss.gpuResidentBytes, 8_000_000_000)
    }

    func testRequestBodyIncludesFirstFrameWhenPresent() {
        let req = VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                  numFrames: 9, fps: 24, mode: .oneStage, steps: 8, cfgScale: 1.0)
        let body = VideoGenService.requestBody(model: "m", prompt: "p", request: req, firstFrameB64: "QUJD")
        XCTAssertEqual(body["first_frame_image"] as? String, "QUJD")
    }

    // MARK: - Audio-to-video request contract

    func testRequestBodyCarriesAudioWhenPresent() {
        let req = VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                  numFrames: 97, fps: 24, mode: .twoStage, steps: 30, cfgScale: 3.0)
        let body = VideoGenService.requestBody(model: "m", prompt: "p", request: req,
                                               firstFrameB64: nil, audioB64: "V0FW")
        XCTAssertEqual(body["audio"] as? String, "V0FW")
        // absent when there's no clip — the server treats presence as intent
        let none = VideoGenService.requestBody(model: "m", prompt: "p", request: req,
                                               firstFrameB64: nil, audioB64: nil)
        XCTAssertNil(none["audio"])
    }

    func testRequestBodyAudioForcesTwoStageAndReferenceGuidance() {
        // a2vid is two-stage only (the server 400s one_stage+audio). A Fast
        // (one-stage) preset with a clip attached must upgrade to two_stage AND
        // drop its one-stage guidance values (cfg 1.0 would run stage 1
        // unguided) so the server's reference defaults (cfg 3/7) apply.
        let req = VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                  numFrames: 97, fps: 24, mode: .oneStage, steps: 12, cfgScale: 1.0)
        let body = VideoGenService.requestBody(model: "m", prompt: "p", request: req,
                                               firstFrameB64: nil, audioB64: "V0FW")
        XCTAssertEqual(body["pipeline"] as? String, "two_stage")
        XCTAssertNil(body["cfg_scale"])
        XCTAssertNil(body["stg_scale"])
        // An explicit two-stage request keeps the user's guidance untouched.
        var hq = VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                 numFrames: 97, fps: 24, mode: .twoStageHQ, steps: 15, cfgScale: 4.0)
        hq.stgScale = 0.5
        let hqBody = VideoGenService.requestBody(model: "m", prompt: "p", request: hq,
                                                 firstFrameB64: nil, audioB64: "V0FW")
        XCTAssertEqual(hqBody["pipeline"] as? String, "two_stage_hq")
        XCTAssertEqual(hqBody["cfg_scale"] as? Double, 4.0)
        XCTAssertEqual(hqBody["stg_scale"] as? Double, 0.5)
    }

    func testAudioFileToWavBase64TranscodesToPcm16Wav() throws {
        // Write a float32 WAV via AVAudioFile (any AVFoundation-readable format
        // works — this pins the transcode-to-PCM16-WAV contract the Zig server
        // parses), then round-trip through the helper.
        let dir = FileManager.default.temporaryDirectory
        let src = dir.appendingPathComponent("a2v_test_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: src) }
        let sr = 22050.0
        // Scope the writer: AVAudioFile flushes its header on dealloc (there is
        // no explicit close on this deployment target) — reading before the
        // writer dies sees an empty file.
        try autoreleasepool {
            let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sr, channels: 1, interleaved: false)!
            let file = try AVAudioFile(forWriting: src, settings: fmt.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
            let n: AVAudioFrameCount = 22050 // 1 s
            let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: n)!
            buf.frameLength = n
            for i in 0..<Int(n) {
                buf.floatChannelData![0][i] = 0.4 * sin(2.0 * .pi * 440.0 * Float(i) / Float(sr))
            }
            try file.write(from: buf)
        }

        let b64 = VideoGenService.audioFileToWavBase64(path: src.path)
        let wav = try XCTUnwrap(b64.flatMap { Data(base64Encoded: $0) })
        // RIFF/WAVE with a PCM16 fmt chunk at the source rate.
        XCTAssertEqual(String(data: wav.prefix(4), encoding: .ascii), "RIFF")
        XCTAssertEqual(String(data: wav.subdata(in: 8..<12), encoding: .ascii), "WAVE")
        let audioFormat = wav.subdata(in: 20..<22).withUnsafeBytes { $0.load(as: UInt16.self) }
        let bits = wav.subdata(in: 34..<36).withUnsafeBytes { $0.load(as: UInt16.self) }
        let rate = wav.subdata(in: 24..<28).withUnsafeBytes { $0.load(as: UInt32.self) }
        XCTAssertEqual(audioFormat, 1) // PCM
        XCTAssertEqual(bits, 16)
        XCTAssertEqual(rate, 22050)
        XCTAssertGreaterThan(wav.count, 44 + 20000) // ~1 s of PCM16 mono
        // Unreadable path → nil, never a throw/crash.
        XCTAssertNil(VideoGenService.audioFileToWavBase64(path: "/nonexistent/clip.m4a"))
    }

    func testFramesCoveringAudioDurationSnapsUpOnLadder() {
        // Attaching a clip auto-suggests a frame count that COVERS it: the
        // smallest 8N+1 ladder value ≥ duration*fps, capped at the model max.
        let m = VideoModelPreset.ltx23Q4
        XCTAssertEqual(m.framesCovering(durationSeconds: 2.0), 49)   // 48 frames → 49
        XCTAssertEqual(m.framesCovering(durationSeconds: 0.1), m.frameOptions.first) // tiny clip → floor
        XCTAssertEqual(m.framesCovering(durationSeconds: 3600), m.frameOptions.last) // longer than cap → max
        XCTAssertNil(m.framesCovering(durationSeconds: 0))           // no/empty clip → no suggestion
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

    /// Regression: Mage-Flow ships the diffusers layout — `model_index.json` +
    /// weight subdirs, NO root config.json. `resolveModelDir` gated on
    /// config.json alone, so a fully-downloaded Mage-Flow read as "not
    /// downloaded" and every gen service threw `.modelMissing`. It must accept
    /// the diffusers root marker too, while still rejecting a bare empty folder.
    func testResolveModelDirAcceptsDiffusersModelIndexLayout() throws {
        let fm = FileManager.default
        let root = NSTemporaryDirectory() + "resolvedir-\(UUID().uuidString)"
        defer { try? fm.removeItem(atPath: root) }

        // Diffusers layout (model_index.json, NO config.json) → resolves.
        let mageDir = (root as NSString).appendingPathComponent("microsoft/Mage-Flow-Turbo")
        try fm.createDirectory(atPath: mageDir, withIntermediateDirectories: true)
        fm.createFile(atPath: (mageDir as NSString).appendingPathComponent("model_index.json"), contents: Data("{}".utf8))
        XCTAssertEqual(ServerManager.resolveModelDir(repo: "microsoft/Mage-Flow-Turbo", modelsRoot: root), mageDir)

        // Standard MLX layout (config.json) → still resolves.
        let stdDir = (root as NSString).appendingPathComponent("acme/std")
        try fm.createDirectory(atPath: stdDir, withIntermediateDirectories: true)
        fm.createFile(atPath: (stdDir as NSString).appendingPathComponent("config.json"), contents: Data("{}".utf8))
        XCTAssertEqual(ServerManager.resolveModelDir(repo: "acme/std", modelsRoot: root), stdDir)

        // A bare empty folder (no marker) → nil (not downloaded).
        let emptyDir = (root as NSString).appendingPathComponent("acme/empty")
        try fm.createDirectory(atPath: emptyDir, withIntermediateDirectories: true)
        XCTAssertNil(ServerManager.resolveModelDir(repo: "acme/empty", modelsRoot: root))
    }

    // MARK: - Residency default

    func testKeepResidentDefaultsOff() {
        // Decision: load→generate→unload by default; "Keep loaded" is opt-in.
        let img = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "x", width: 1024, height: 1024, steps: 4)
        XCTAssertFalse(img.keepResident)
        let vid = VideoGenRequest(model: .ltx23Q4, prompt: "x", width: 384, height: 256, numFrames: 9, fps: 24, mode: .oneStage, steps: 6, cfgScale: 1.0)
        XCTAssertFalse(vid.keepResident)
        let aud = AudioGenRequest(model: .qwen3TTS06B, text: "x")
        XCTAssertFalse(aud.keepResident)
    }

    // MARK: - Per-model capabilities (the Advanced panel is gated on these)

    /// Every flag mirrors a server-side fact. A control the backend ignores is
    /// worse than a missing one, and Mage-Flow shipped with five of them: CFG,
    /// negative prompt, conditioning rebalance, LoRA and variation strength all
    /// showed for a model that answers 400 or silently drops them.
    func testMageFlowDeclaresTheCapabilitiesItActuallyHas() {
        for p in [ImageModelPreset.mageFlowTurbo, .mageFlowEditTurbo] {
            XCTAssertFalse(p.supportsImg2Img, "\(p.id): no VAE-encoder variation path (server 400s)")
            XCTAssertFalse(p.supportsLoRA, "\(p.id): no LoRA path (0 matched modules -> 400)")
            XCTAssertEqual(p.condWeightCount, 0, "\(p.id): single final hidden state, no layers to tap")
            XCTAssertTrue(p.stepsAreFixed, "\(p.id): distilled 4-step schedule")
            XCTAssertEqual(p.fixedSteps, 4)
            // Distillation-fixed means every tier is the same; a tier that only
            // buys time is the no-op class in another costume.
            for q in QualityPreset.allCases {
                XCTAssertEqual(p.settings(q).steps, 4, "\(p.id) \(q): tiers must not sell steps")
            }
        }
        // Only the Edit checkpoint edits; only it can do reference edits.
        XCTAssertTrue(ImageModelPreset.mageFlowEditTurbo.supportsReferenceEdit)
        XCTAssertFalse(ImageModelPreset.mageFlowTurbo.supportsReferenceEdit)
        // The backends that DO have these paths keep them.
        for p in [ImageModelPreset.flux2Klein4B_Q4, .krea2Turbo] {
            XCTAssertTrue(p.supportsImg2Img)
            XCTAssertTrue(p.supportsLoRA)
            XCTAssertFalse(p.stepsAreFixed)
            XCTAssertGreaterThan(p.condWeightCount, 0)
        }
    }

    /// The 8-bit mirrors are the SAME architecture at half the download, so they
    /// reuse their bf16 sibling's variant — quantization is a property of the
    /// checkpoint, not of what the model can do. If they ever diverge on a
    /// capability flag, one of the two is lying to the UI.
    func testMageFlow8BitPresetsMatchTheirBf16Siblings() {
        let pairs = [
            (ImageModelPreset.mageFlowTurbo, ImageModelPreset.mageFlowTurbo8bit),
            (ImageModelPreset.mageFlowEditTurbo, ImageModelPreset.mageFlowEditTurbo8bit),
        ]
        for (bf16, q8) in pairs {
            XCTAssertEqual(q8.variant, bf16.variant, "\(q8.id): same architecture, same variant")
            XCTAssertEqual(q8.configName, bf16.configName)
            XCTAssertEqual(q8.supportsReferenceEdit, bf16.supportsReferenceEdit)
            XCTAssertEqual(q8.supportsImg2Img, bf16.supportsImg2Img)
            XCTAssertEqual(q8.supportsLoRA, bf16.supportsLoRA)
            XCTAssertEqual(q8.stepsAreFixed, bf16.stepsAreFixed)
            XCTAssertEqual(q8.fixedSteps, bf16.fixedSteps)
            XCTAssertEqual(q8.resolutions, bf16.resolutions)
            XCTAssertNotEqual(q8.id, bf16.id, "distinct ids or the picker collapses them")
            XCTAssertLessThan(q8.approxDownloadGB, bf16.approxDownloadGB, "\(q8.id): must be the smaller one")
            XCTAssertTrue(q8.repo.hasPrefix("ddalcu/"), "\(q8.id): our mirror, not microsoft's")
        }
        // The engine gates EDIT capability on the directory name containing
        // "mage-flow-edit" (dirIsEdit in src/mage_flow.zig), and the download
        // dir is the repo id — so the edit mirror's name is load-bearing.
        let editDir = ImageModelPreset.mageFlowEditTurbo8bit.repo
            .split(separator: "/").last.map(String.init) ?? ""
        XCTAssertTrue(editDir.lowercased().contains("mage-flow-edit"),
                      "\(editDir) would come up in text-to-image mode")
        XCTAssertFalse(
            (ImageModelPreset.mageFlowTurbo8bit.repo.split(separator: "/").last.map(String.init) ?? "")
                .lowercased().contains("mage-flow-edit"),
            "the txt2img mirror must NOT match dirIsEdit")
        // Both are in the catalog, cheapest-first ordering intact.
        let all = ImageModelPreset.all
        XCTAssertTrue(all.contains { $0.id == ImageModelPreset.mageFlowTurbo8bit.id })
        XCTAssertTrue(all.contains { $0.id == ImageModelPreset.mageFlowEditTurbo8bit.id })
        for (i, p) in all.enumerated() where i > 0 {
            XCTAssertGreaterThanOrEqual(p.approxDownloadGB, all[i - 1].approxDownloadGB,
                                        "catalog must stay ordered cheapest → heaviest")
        }
    }

    func testMageFlowResolutionsCoverTheNativeRange() {
        let r = ImageModelPreset.mageFlowTurbo.resolutions
        // The model card claims native 512-2048 on any aspect, "including
        // extreme 4:1" — and measured cost tracks megapixels, not shape, so
        // there's no reason to hide the panoramas.
        XCTAssertTrue(r.contains { $0.width == 2048 && $0.height == 2048 }, "missing the 2048 ceiling")
        XCTAssertTrue(r.contains { $0.width == 2048 && $0.height == 512 }, "missing the 4:1 panorama")
        XCTAssertTrue(r.contains { $0.width == 512 && $0.height == 2048 }, "missing the 1:4 tall shape")
        for o in r {
            XCTAssertEqual(o.width % 16, 0, "\(o.label): VAE is /16")
            XCTAssertEqual(o.height % 16, 0, "\(o.label): VAE is /16")
            XCTAssertTrue((512...2048).contains(o.width) && (512...2048).contains(o.height),
                          "\(o.label) is outside the model's native 512-2048 range")
        }
    }

    func testMatchSourceIsOfferedOnlyWhenEditingAndSurvivesModeChanges() {
        let edit = ImageModelPreset.mageFlowEditTurbo
        // Editing: "Match source" leads, because an editor that changes the
        // geometry of its input is wrong by default.
        XCTAssertEqual(edit.resolutionOptions(editMode: true).first, .matchSource)
        XCTAssertFalse(edit.resolutionOptions(editMode: false).contains(.matchSource))
        // A txt2img model never offers it (nothing to match).
        XCTAssertFalse(ImageModelPreset.mageFlowTurbo.resolutionOptions(editMode: true).contains(.matchSource))
        // Leaving edit mode with "Match source" selected must re-point the
        // picker instead of leaving it on an off-menu value.
        XCTAssertEqual(edit.validResolution(.matchSource, editMode: false), edit.defaultResolution)
        XCTAssertEqual(edit.validResolution(.matchSource, editMode: true), .matchSource)
        // Switching models drops a resolution the new model doesn't offer.
        let panorama = ResolutionOption(width: 2048, height: 512, label: "x")
        XCTAssertEqual(ImageModelPreset.flux2Klein4B_Q4.validResolution(panorama, editMode: false),
                       ImageModelPreset.flux2Klein4B_Q4.defaultResolution)
    }

    // MARK: - Prompt examples (an editor's repertoire IS its prompt vocabulary)

    func testPromptExamplesMatchTheModeAndTheModel() {
        // No source image → text-to-image starters, whatever the model.
        for p in [ImageModelPreset.mageFlowEditTurbo, .mageFlowTurbo, .flux2Klein4B_Q4, .krea2Turbo] {
            XCTAssertEqual(p.promptExamples(editing: false), ImagePromptExamples.textToImage, "\(p.id)")
        }
        // Editing on Mage-Flow-Edit → the full published repertoire.
        let mage = ImageModelPreset.mageFlowEditTurbo.promptExamples(editing: true)
        XCTAssertEqual(mage, ImagePromptExamples.mageFlowEdit)
        XCTAssertTrue(mage.contains { $0.name == "Control maps" })
        XCTAssertTrue(mage.contains { $0.name == "Restore" })
        // Editing on another in-context editor → generic instructions only. We
        // verified control maps and restoration on Mage-Flow, not on FLUX;
        // offering them there would be advertising, not a feature.
        let flux = ImageModelPreset.flux2Klein4B_Q4.promptExamples(editing: true)
        XCTAssertEqual(flux, ImagePromptExamples.genericEdit)
        XCTAssertFalse(flux.contains { $0.name == "Control maps" })
        // A model that can't edit never shows edit instructions, even if the
        // pane somehow asks for them.
        XCTAssertEqual(ImageModelPreset.mageFlowTurbo.promptExamples(editing: true), ImagePromptExamples.textToImage)
    }

    /// These strings are the model's OWN published phrasings, and the exact ones
    /// verified live against the port (a depth map, canny edges, a
    /// white-background cutout). An editor keys on the wording: shortening
    /// "Generate a grayscale monocular depth map… closer regions brighter and
    /// farther regions darker." to "make a depth map" returns a grey photo, not
    /// a depth map. Rewording them is a quality regression, so pin them.
    func testControlMapPromptsAreTheModelsOwnWordingVerbatim() {
        let maps = ImagePromptExamples.mageFlowEdit.first { $0.name == "Control maps" }!
        func body(_ title: String) -> String { maps.examples.first { $0.title == title }!.body }
        XCTAssertEqual(body("Depth map"),
            "Generate a grayscale monocular depth map of this image. Represent relative distance at pixel level, with closer regions brighter and farther regions darker.")
        XCTAssertEqual(body("Canny edges"),
            "Convert this image into a clean black-and-white Canny edge map showing only the important object and scene contours.")
        XCTAssertEqual(body("Segmentation map"),
            "Generate a semantic segmentation map that assigns clearly different flat colors to the main subject, other foreground objects, and the background regions.")
        let content = ImagePromptExamples.mageFlowEdit.first { $0.name == "Content" }!
        XCTAssertEqual(content.examples.first { $0.title == "Cut out the subject" }!.body,
            "Extract the main foreground subject from the image and isolate it on a clean pure white background. Preserve its shape, identity, texture, and fine boundary details.")
    }

    func testEveryExampleGroupIsUsableInAMenu() {
        for lib in [ImagePromptExamples.textToImage, ImagePromptExamples.mageFlowEdit, ImagePromptExamples.genericEdit] {
            var seenGroups = Set<String>()
            for g in lib {
                XCTAssertFalse(g.examples.isEmpty, "\(g.name): an empty submenu is a dead end")
                XCTAssertTrue(seenGroups.insert(g.name).inserted, "duplicate group \(g.name)")
                var seenTitles = Set<String>()
                for e in g.examples {
                    // SwiftUI keys the menu rows on `title`; a duplicate silently
                    // collapses two entries into one.
                    XCTAssertTrue(seenTitles.insert(e.title).inserted, "\(g.name): duplicate title \(e.title)")
                    XCTAssertFalse(e.title.isEmpty)
                    XCTAssertGreaterThan(e.body.count, 12, "\(e.title): too short to be a useful starting point")
                }
            }
        }
    }

    // MARK: - Image request body (img2img + rebalance + LoRA)

    func testImageRequestJsonOmitsSizeForMatchSource() {
        // width/height 0 = "Match source": the server must not see a `size`, so
        // an edit comes back at the source's own resolution (the reference
        // pipeline's max_size default) instead of being re-gridded to a bucket.
        var req = ImageGenRequest(model: .mageFlowEditTurbo, prompt: "x", width: 0, height: 0, steps: 4)
        req.editMode = true
        let json = ImageGenService.requestJson(for: req, modelName: "m", seed: 3)
        XCTAssertNil(json["size"], "match-source must omit size entirely")
        XCTAssertEqual(json["steps"] as? Int, 4)
        // A real size is still sent verbatim.
        let sized = ImageGenRequest(model: .mageFlowEditTurbo, prompt: "x", width: 2048, height: 512, steps: 4)
        XCTAssertEqual(ImageGenService.requestJson(for: sized, modelName: "m", seed: 3)["size"] as? String, "2048x512")
    }


    func testImageRequestJsonDefaultsOmitImg2ImgRebalanceAndLora() {
        let req = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "x", width: 1024, height: 1024, steps: 4)
        let json = ImageGenService.requestJson(for: req, modelName: "m", seed: 7)
        XCTAssertEqual(json["prompt"] as? String, "x")
        XCTAssertEqual(json["seed"] as? Int, 7)
        // No behavior change for plain text-to-image requests.
        XCTAssertNil(json["image"])
        XCTAssertNil(json["strength"])
        XCTAssertNil(json["cond_gain"])
        XCTAssertNil(json["cond_weights"])
        XCTAssertNil(json["lora_path"])
    }

    func testImageRequestJsonIncludesImg2ImgFields() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("img2img-src-\(UUID().uuidString).png")
        let bytes = Data([0x89, 0x50, 0x4E, 0x47, 1, 2, 3, 4])
        try bytes.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        var req = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "x", width: 1024, height: 1024, steps: 8)
        req.initImagePath = tmp.path
        req.strength = 0.45
        let json = ImageGenService.requestJson(for: req, modelName: "m", seed: 1)
        XCTAssertEqual(json["image"] as? String, bytes.base64EncodedString())
        XCTAssertEqual(json["strength"] as? Double, 0.45)
    }

    func testImageRequestJsonMissingSourceFileDropsImg2Img() {
        var req = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "x", width: 1024, height: 1024, steps: 8)
        req.initImagePath = "/definitely/not/a/real/file.png"
        req.strength = 0.4
        let json = ImageGenService.requestJson(for: req, modelName: "m", seed: 1)
        XCTAssertNil(json["image"])
        XCTAssertNil(json["strength"])
    }

    func testImageRequestJsonIncludesRebalanceAndLora() {
        var req = ImageGenRequest(model: .krea2Turbo, prompt: "x", width: 1024, height: 1024, steps: 8)
        req.condGain = 1.5
        req.condWeightsText = "1, 1 1 1 1 1 0.5 1 1 1 1 2"
        req.loraPath = "/tmp/style.safetensors"
        let json = ImageGenService.requestJson(for: req, modelName: "m", seed: 1)
        XCTAssertEqual(json["cond_gain"] as? Double, 1.5)
        let w = json["cond_weights"] as? [Double]
        XCTAssertEqual(w?.count, 12)
        XCTAssertEqual(w?[6], 0.5)
        XCTAssertEqual(w?[11], 2)
        XCTAssertEqual(json["lora_path"] as? String, "/tmp/style.safetensors")
    }

    func testParseCondWeightsAcceptsCommasAndSpacesRejectsGarbage() {
        XCTAssertEqual(ImageGenRequest.parseCondWeights("1,2,3"), [1, 2, 3])
        XCTAssertEqual(ImageGenRequest.parseCondWeights(" 0.5  1\t-2 "), [0.5, 1, -2])
        XCTAssertEqual(ImageGenRequest.parseCondWeights("1, 2,, 3"), [1, 2, 3])
        XCTAssertNil(ImageGenRequest.parseCondWeights("1,x,3"))
        XCTAssertNil(ImageGenRequest.parseCondWeights(""))
        XCTAssertNil(ImageGenRequest.parseCondWeights("  "))
    }

    func testCondWeightCountFollowsBackend() {
        // FLUX taps encoder layers 9/18/27 → 3 weights; Krea taps 12 layers.
        XCTAssertEqual(ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "x", width: 1024, height: 1024, steps: 4).condWeightCount, 3)
        XCTAssertEqual(ImageGenRequest(model: .krea2Turbo, prompt: "x", width: 1024, height: 1024, steps: 8).condWeightCount, 12)
    }

    // MARK: - Instruction edit mode (FLUX.2 in-context reference conditioning)

    func testImageRequestJsonEditModeSendsModeAndOmitsStrength() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("edit-src-\(UUID().uuidString).png")
        try Data([1, 2, 3]).write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        var req = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "make the hair blue", width: 1024, height: 1024, steps: 8)
        req.initImagePath = tmp.path
        req.editMode = true
        req.strength = 0.4
        let json = ImageGenService.requestJson(for: req, modelName: "m", seed: 1)
        XCTAssertEqual(json["mode"] as? String, "edit")
        XCTAssertNotNil(json["image"])
        // Edit conditions on the clean reference — strength does not apply.
        XCTAssertNil(json["strength"])
    }

    func testImageRequestJsonVariationModeOmitsModeField() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("var-src-\(UUID().uuidString).png")
        try Data([1, 2, 3]).write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        var req = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "x", width: 1024, height: 1024, steps: 8)
        req.initImagePath = tmp.path
        req.editMode = false
        let json = ImageGenService.requestJson(for: req, modelName: "m", seed: 1)
        XCTAssertNil(json["mode"]) // default server behavior = variation
        XCTAssertNotNil(json["strength"])
    }

    func testImageRequestJsonEditModeIncludesRefImages() throws {
        let tmpDir = FileManager.default.temporaryDirectory
        let src = tmpDir.appendingPathComponent("edit-src-\(UUID().uuidString).png")
        try Data([1, 2, 3]).write(to: src)
        defer { try? FileManager.default.removeItem(at: src) }
        let refBytes = Data([9, 8, 7, 6])
        let ref = tmpDir.appendingPathComponent("edit-ref-\(UUID().uuidString).png")
        try refBytes.write(to: ref)
        defer { try? FileManager.default.removeItem(at: ref) }

        var req = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "replace the face in image 1 with the face from image 2", width: 1024, height: 1024, steps: 8)
        req.initImagePath = src.path
        req.editMode = true
        req.refImagePaths = [ref.path, "/definitely/not/a/real/ref.png"] // missing file skipped
        let json = ImageGenService.requestJson(for: req, modelName: "m", seed: 1)
        XCTAssertEqual(json["mode"] as? String, "edit")
        XCTAssertEqual(json["ref_images"] as? [String], [refBytes.base64EncodedString()])
    }

    func testImageRequestJsonRefImagesOmittedInVariationAndTextToImage() throws {
        let tmpDir = FileManager.default.temporaryDirectory
        let ref = tmpDir.appendingPathComponent("var-ref-\(UUID().uuidString).png")
        try Data([1]).write(to: ref)
        defer { try? FileManager.default.removeItem(at: ref) }

        // Variation mode: extra references have no meaning (the server 400s).
        let src = tmpDir.appendingPathComponent("var-src-\(UUID().uuidString).png")
        try Data([2]).write(to: src)
        defer { try? FileManager.default.removeItem(at: src) }
        var vreq = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "x", width: 1024, height: 1024, steps: 8)
        vreq.initImagePath = src.path
        vreq.editMode = false
        vreq.refImagePaths = [ref.path]
        XCTAssertNil(ImageGenService.requestJson(for: vreq, modelName: "m", seed: 1)["ref_images"])

        // Text-to-image (no source at all): refs are meaningless too.
        var treq = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "x", width: 1024, height: 1024, steps: 8)
        treq.editMode = true
        treq.refImagePaths = [ref.path]
        XCTAssertNil(ImageGenService.requestJson(for: treq, modelName: "m", seed: 1)["ref_images"])
    }

    func testSupportsReferenceEditFollowsVariant() {
        // Editing is a trained FLUX.2 capability; Krea doesn't have it.
        XCTAssertTrue(ImageModelPreset.flux2Klein4B_Q4.supportsReferenceEdit)
        XCTAssertFalse(ImageModelPreset.krea2Turbo.supportsReferenceEdit)
    }
}
