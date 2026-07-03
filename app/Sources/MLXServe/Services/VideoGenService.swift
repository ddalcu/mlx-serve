import Foundation
import SwiftUI
import AppKit
import AVFoundation
import CoreVideo

/// Runs LTX-Video 2.3 text-to-video via the native `mlx-serve` engine (no Python).
///
/// Serves the LTX model with a dedicated `mlx-serve` instance, POSTs
/// `/v1/video/generations` (which returns base64 RGB frames), then muxes the
/// frames into an mp4 with AVFoundation under `~/.mlx-serve/generations/video`.
@MainActor
final class VideoGenService: ObservableObject {

    enum Phase: Equatable {
        case idle
        case running(step: Int, total: Int, message: String)
        case completed(path: String)
        case cancelled
        case failed(String)
    }

    /// Live residency of the pane's model: is it loaded server-side, and how
    /// many bytes the server holds resident across ALL loaded models. Polled
    /// by the view from `/v1/models` only — see `refreshResidency`.
    struct Residency: Equatable {
        var loaded: Bool
        var bytesResident: UInt64
        var gpuResidentBytes: Int64
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var recent: [String] = []
    @Published private(set) var log: [String] = []
    @Published private(set) var residency: Residency? = nil

    private var task: Task<Void, Never>?
    private let api = APIClient()
    /// Monotonic generation id. Phase writes from a superseded task are
    /// dropped — cancel-then-regenerate used to race the old task's catch
    /// (setting .failed/.idle) against the new run's .running.
    private var generationSeq = 0

    private func setPhase(_ p: Phase, for gen: Int) {
        guard gen == generationSeq else { return }
        phase = p
    }

    /// A cancelled URLSession request surfaces as URLError.cancelled, not
    /// CancellationError — both mean "the user hit Cancel", never "Failed".
    nonisolated static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        if let u = error as? URLError, u.code == .cancelled { return true }
        return false
    }

    init() {
        loadRecent()
    }

    var isRunning: Bool {
        if case .running = phase { return true }
        return false
    }

    /// Generate through the ONE main server: ensure running (headless if
    /// needed), load the LTX model on demand, stream `/v1/video/generations`,
    /// mux the returned frames to mp4, then unload unless "Keep loaded" is set.
    func generate(_ request: VideoGenRequest, server: ServerManager) {
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            phase = .failed("Prompt is empty.")
            return
        }
        guard let modelDir = ServerManager.resolveModelDir(repo: request.model.repo) else {
            phase = .failed("Model \(request.model.repo) is not downloaded. Download it first.")
            return
        }

        task?.cancel()
        generationSeq += 1
        let gen = generationSeq
        phase = .running(step: 0, total: 3, message: "Loading model…")
        log = []

        let outputPath = Self.makeOutputPath(prompt: request.prompt)
        let prompt = request.prompt
        let fps = request.fps
        let steps = request.steps
        let keep = request.keepResident
        let firstFramePath = request.firstFrameImagePath
        let audioPath = request.audioPath

        task = Task {
            var loadedId: String? = nil
            func releaseIfNeeded() async {
                if !keep, let id = loadedId { try? await server.unloadModel(id: id) }
            }
            do {
                // Image-to-video: read the first-frame image file → base64 OFF the
                // main actor (the file can be multi-MB; reading it synchronously in
                // generate() blocked the UI). The server VAE-encodes it and pins it
                // as the clean first frame. Mirrors AudioGenService's `ref_audio`.
                let firstFrameB64: String? = await Task.detached(priority: .userInitiated) {
                    firstFramePath.flatMap { path in
                        (try? Data(contentsOf: URL(fileURLWithPath: path)))?.base64EncodedString()
                    }
                }.value
                // Audio-to-video: transcode the clip to a PCM16 WAV off-main
                // (AVFoundation decode of an mp3/m4a can take a moment). A
                // failed transcode is a hard error — the user asked for THIS
                // soundtrack, so silently generating audio instead would be a
                // wrong result (mirrors the server's explicit 400s).
                let audioB64: String? = await Task.detached(priority: .userInitiated) {
                    audioPath.flatMap { Self.audioFileToWavBase64(path: $0) }
                }.value
                if audioPath != nil, audioB64 == nil {
                    setPhase(.failed("Couldn't read the audio clip. Pick a WAV, MP3, M4A, or AAC file."), for: gen)
                    return
                }
                let port = try await server.ensureRunning(forGenModelDir: modelDir)
                if Task.isCancelled { setPhase(.cancelled, for: gen); return }
                let info = try await server.loadModel(id: modelDir)
                loadedId = info.name
                // Cancelled right after load: deliberately leave the model
                // resident (an unload from a cancelled task can't run anyway,
                // and the likely next action is a retry — the residency row
                // in the pane shows the state).
                if Task.isCancelled { setPhase(.cancelled, for: gen); return }
                let body = Self.requestBody(model: info.name, prompt: prompt,
                                            request: request, firstFrameB64: firstFrameB64,
                                            audioB64: audioB64)
                // SSE: the server pushes `progress` events per denoise step, then a
                // `complete` event with the frames. Drive a determinate bar from them.
                var decoded: DecodedFrames? = nil
                for try await ev in api.streamGeneration(
                    port: port, path: "/v1/video/generations", json: body) {
                    switch ev["type"] as? String {
                    case "progress":
                        let step = ev["step"] as? Int ?? 0
                        let total = ev["total"] as? Int ?? steps
                        let stage = ev["stage"] as? String ?? "Generating"
                        setPhase(.running(step: step, total: max(total, 1), message: "\(stage)…"), for: gen)
                    case "complete":
                        decoded = Self.decodeFrames(ev)
                    case "error":
                        await releaseIfNeeded()
                        setPhase(.failed(ev["message"] as? String ?? "Generation failed."), for: gen)
                        return
                    default:
                        break
                    }
                }
                await releaseIfNeeded()
                guard let frames = decoded else {
                    setPhase(.failed("Server returned no video frames."), for: gen)
                    return
                }
                if Task.isCancelled { setPhase(.cancelled, for: gen); return }
                setPhase(.running(step: steps, total: steps, message: "Encoding mp4…"), for: gen)
                let outFps = frames.fps > 0 ? frames.fps : fps
                try await Task.detached(priority: .userInitiated) {
                    try VideoGenService.writeMP4(
                        rgb: frames.rgb, frames: frames.frames,
                        width: frames.width, height: frames.height,
                        fps: outFps, to: URL(fileURLWithPath: outputPath),
                        audioPCM: frames.audioPCM, audioSampleRate: frames.audioSampleRate,
                        audioChannels: frames.audioChannels)
                }.value
                setPhase(.completed(path: outputPath), for: gen)
                insertRecent(outputPath)
            } catch {
                if Task.isCancelled || Self.isCancellation(error) {
                    // User cancelled. No unload: the server aborts the denoise
                    // loop itself when the socket closes, and the model stays
                    // resident for an instant retry (visible in the pane's
                    // residency row).
                    setPhase(.cancelled, for: gen)
                    return
                }
                await releaseIfNeeded()
                setPhase(.failed(error.localizedDescription), for: gen)
            }
        }
    }

    func cancel() {
        task?.cancel()
        task = nil
        // Instant feedback; the cancelled task's own catch re-confirms it.
        if isRunning { phase = .cancelled }
    }

    // MARK: - Residency (model loaded? GPU memory?)

    /// Refresh `residency` from `/v1/models` ONLY — a no-model endpoint that
    /// reports `loaded` + `bytes_resident` per registry entry. Deliberately
    /// NOT `/props`: that route resolves the DEFAULT model, so on a headless
    /// gen-only boot it 503s (the live "GPU memory 0 MB" bug), and it can
    /// even cold-load an evicted default chat model from a mere status poll.
    /// Cheap localhost GET; never starts the server — a stopped server reads
    /// as "not loaded".
    func refreshResidency(repo: String, server: ServerManager) async {
        guard server.status == .running else {
            residency = nil
            return
        }
        guard let entries = try? await api.fetchAllModels(port: server.port) else {
            residency = nil
            return
        }
        let dirBase = ServerManager.resolveModelDir(repo: repo)
            .map { URL(fileURLWithPath: $0).lastPathComponent }
        residency = Self.residency(from: entries, repo: repo, dirBasename: dirBase)
    }

    /// Pure reduction of a `/v1/models` snapshot into the pane's residency:
    /// this model's loaded flag, plus the summed resident bytes of every
    /// loaded entry ("what does the GPU hold" — chat models included).
    nonisolated static func residency(from entries: [ModelInfo], repo: String, dirBasename: String?) -> Residency {
        let entry = entries.first { entryMatches(id: $0.name, repo: repo, dirBasename: dirBasename) }
        let total = entries.filter(\.loaded)
            .reduce(Int64(0)) { $0 + Int64(clamping: $1.bytesResident) }
        return Residency(
            loaded: entry?.loaded ?? false,
            bytesResident: entry?.bytesResident ?? 0,
            gpuResidentBytes: total
        )
    }

    /// Registry ids are `org/model` for discovered dirs, or an absolute path /
    /// bare basename for path-registered models — match any of those shapes.
    nonisolated static func entryMatches(id: String, repo: String, dirBasename: String?) -> Bool {
        if id == repo { return true }
        guard let base = dirBasename, !base.isEmpty else { return false }
        return id == base || id.hasSuffix("/" + base)
    }

    // MARK: - Request body (pure so tests can pin the wire contract)

    /// Build the `/v1/video/generations` request body. Pure + static because the
    /// pipeline/CFG/STG fields silently not being sent is exactly the bug that
    /// made the Quality preset (cfg 3.0, twoStage) run as unguided one-stage —
    /// tests pin every field here so the UI model can't drift from the wire.
    nonisolated static func requestBody(model: String, prompt: String,
                                        request: VideoGenRequest, firstFrameB64: String?,
                                        audioB64: String? = nil) -> [String: Any] {
        var pipeline: String
        switch request.mode {
        case .oneStage:   pipeline = "one_stage"
        case .twoStage:   pipeline = "two_stage"
        case .twoStageHQ: pipeline = "two_stage_hq"
        }
        // Audio-to-video is two-stage only (the server 400s one_stage+audio).
        // A one-stage preset with a clip attached upgrades to two_stage and
        // DROPS its guidance values — one-stage's cfg 1.0 would run stage 1
        // unguided; omitting the fields lets the server apply the reference
        // two-stage defaults (cfg 3.0 video / 7.0 audio).
        let hasAudio = (audioB64?.isEmpty == false)
        var dropGuidance = false
        if hasAudio, request.mode == .oneStage {
            pipeline = "two_stage"
            dropGuidance = true
        }
        var body: [String: Any] = [
            "model": model, "prompt": prompt, "num_frames": request.numFrames,
            "height": request.height, "width": request.width, "steps": request.steps,
            "seed": request.seed,
            "pipeline": pipeline,
        ]
        if !dropGuidance {
            body["cfg_scale"] = request.cfgScale
            body["stg_scale"] = request.stgScale
        }
        if let firstFrameB64 { body["first_frame_image"] = firstFrameB64 }
        if hasAudio, let audioB64 { body["audio"] = audioB64 }
        if let lora = request.loraPath, !lora.isEmpty {
            body["lora_path"] = lora
            if request.loraScale != 1.0 { body["lora_scale"] = request.loraScale }
        }
        return body
    }

    /// Longest clip shipped to the server. LTX's frame ladder tops out around
    /// 8 s of video; 30 s keeps the base64 payload bounded when a user picks a
    /// full song (the server trims to the video duration anyway).
    nonisolated static let maxAudioSeconds: Double = 30

    /// Read ANY AVFoundation-readable audio file (wav/mp3/m4a/aac/…) and
    /// transcode to a 16-bit PCM WAV at the source sample rate, ≤2 channels,
    /// base64-encoded for the `audio` request field. Returns nil on any
    /// failure (the caller surfaces it as a user-facing error — a2vid must
    /// never silently fall back to generated audio).
    nonisolated static func audioFileToWavBase64(path: String) -> String? {
        guard let file = try? AVAudioFile(forReading: URL(fileURLWithPath: path)) else { return nil }
        let srcFmt = file.processingFormat
        let sr = srcFmt.sampleRate
        let ch = min(srcFmt.channelCount, 2)
        guard sr > 0, ch > 0,
              let dstFmt = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: sr,
                                         channels: ch, interleaved: true),
              let converter = AVAudioConverter(from: srcFmt, to: dstFmt) else { return nil }

        let totalFrames = AVAudioFrameCount(min(Double(file.length), sr * maxAudioSeconds))
        guard totalFrames > 0,
              let srcBuf = AVAudioPCMBuffer(pcmFormat: srcFmt, frameCapacity: totalFrames),
              let dstBuf = AVAudioPCMBuffer(pcmFormat: dstFmt, frameCapacity: totalFrames) else { return nil }
        do { try file.read(into: srcBuf, frameCount: totalFrames) } catch { return nil }

        var fed = false
        var convErr: NSError?
        let status = converter.convert(to: dstBuf, error: &convErr) { _, outStatus in
            if fed { outStatus.pointee = .endOfStream; return nil }
            fed = true
            outStatus.pointee = .haveData
            return srcBuf
        }
        guard status != .error, convErr == nil, dstBuf.frameLength > 0,
              let data = dstBuf.int16ChannelData else { return nil }

        let frames = Int(dstBuf.frameLength)
        let channels = Int(ch)
        let dataBytes = frames * channels * 2
        var wav = Data(capacity: 44 + dataBytes)
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { wav.append(contentsOf: $0) } }
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { wav.append(contentsOf: $0) } }
        wav.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + dataBytes))
        wav.append(contentsOf: Array("WAVE".utf8))
        wav.append(contentsOf: Array("fmt ".utf8)); u32(16)
        u16(1); u16(UInt16(channels)); u32(UInt32(sr))
        u32(UInt32(sr) * UInt32(channels) * 2); u16(UInt16(channels * 2)); u16(16)
        wav.append(contentsOf: Array("data".utf8)); u32(UInt32(dataBytes))
        // int16ChannelData is interleaved when the format is interleaved:
        // channel 0's pointer covers frames*channels samples.
        data[0].withMemoryRebound(to: UInt8.self, capacity: dataBytes) { p in
            wav.append(UnsafeBufferPointer(start: p, count: dataBytes))
        }
        return wav.base64EncodedString()
    }

    // MARK: - Decode + mux (pure / nonisolated so they're testable + off-main)

    struct DecodedFrames: Equatable {
        var rgb: Data        // [frames * height * width * 3] row-major RGB
        var frames: Int
        var height: Int
        var width: Int
        var fps: Int
        // Optional sound track (present when the server decoded the LTX audio
        // latent): interleaved signed-16-bit little-endian PCM.
        var audioPCM: Data? = nil
        var audioSampleRate: Int = 16000
        var audioChannels: Int = 2
    }

    /// Parse the native server's `{frames,height,width,fps,format,data,…audio}` body.
    nonisolated static func decodeFrames(_ body: Data) -> DecodedFrames? {
        guard let obj = try? JSONSerialization.jsonObject(with: body) as? [String: Any] else { return nil }
        return decodeFrames(obj)
    }

    /// Same, from an already-parsed object (the SSE `complete` event).
    nonisolated static func decodeFrames(_ obj: [String: Any]) -> DecodedFrames? {
        guard let format = obj["format"] as? String, format == "rgb8",
              let frames = obj["frames"] as? Int,
              let height = obj["height"] as? Int,
              let width = obj["width"] as? Int,
              let b64 = obj["data"] as? String,
              let rgb = Data(base64Encoded: b64),
              rgb.count == frames * height * width * 3
        else { return nil }
        let fps = (obj["fps"] as? Int) ?? 24
        var out = DecodedFrames(rgb: rgb, frames: frames, height: height, width: width, fps: fps)
        // Audio is optional + best-effort: a malformed/absent track never blocks
        // the (always-present) video.
        if obj["audio_format"] as? String == "pcm_s16le",
           let ab64 = obj["audio_data"] as? String,
           let pcm = Data(base64Encoded: ab64), !pcm.isEmpty {
            let sr = (obj["audio_sample_rate"] as? Int) ?? 16000
            let ch = (obj["audio_channels"] as? Int) ?? 2
            // Server-controlled fields: an invalid sample rate / channel count
            // drops the audio rather than crash the mux downstream
            // (bytesPerFrame = 2 * channels would divide by zero).
            if sr > 0, ch > 0 {
                out.audioPCM = pcm
                out.audioSampleRate = sr
                out.audioChannels = ch
            }
        }
        return out
    }

    enum MuxError: Error { case writerInit, noPool, finishFailed(String), audioBuffer }

    /// Mux raw RGB frames (+ optional stereo PCM) → h264/aac mp4 via AVAssetWriter.
    nonisolated static func writeMP4(rgb: Data, frames: Int, width: Int, height: Int, fps: Int, to url: URL,
                                     audioPCM: Data? = nil, audioSampleRate: Int = 16000, audioChannels: Int = 2) throws {
        try? FileManager.default.removeItem(at: url)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: attrs)
        guard writer.canAdd(input) else { throw MuxError.writerInit }
        writer.add(input)

        // Optional audio track (AAC, transcoded from the source LPCM).
        // channels/sampleRate are server-controlled: invalid values (≤ 0) skip
        // the audio input ENTIRELY — never divide by zero in appendAudio, never
        // create a starved sibling input the video loop would wedge on.
        var audioInput: AVAssetWriterInput? = nil
        if let pcm = audioPCM, !pcm.isEmpty, audioChannels > 0, audioSampleRate > 0 {
            // No explicit bitrate: at 16 kHz the AAC encoder rejects high rates
            // (e.g. 128 kbps → -12651 "encoding parameters not supported"); let it
            // pick a valid rate for the sample rate/channel count.
            let aset: [String: Any] = [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: audioSampleRate,
                AVNumberOfChannelsKey: audioChannels,
            ]
            let ai = AVAssetWriterInput(mediaType: .audio, outputSettings: aset)
            ai.expectsMediaDataInRealTime = false
            if writer.canAdd(ai) { writer.add(ai); audioInput = ai }
        }

        guard writer.startWriting() else { throw writer.error ?? MuxError.writerInit }
        writer.startSession(atSourceTime: .zero)

        // Append the FULL audio track (one buffer) and mark it finished BEFORE the
        // video loop. A multi-input AVAssetWriter applies backpressure to keep the
        // tracks interleaved: if we pushed every video frame first, the muxer would
        // stop accepting video (isReadyForMoreMediaData → false forever) to wait for
        // audio data near the same timeline — but that audio only gets appended
        // after the loop, so the video busy-wait deadlocks. Finishing audio up front
        // leaves the video input with no active sibling to wait on.
        if let ai = audioInput, let pcm = audioPCM {
            try appendAudio(ai, pcm: pcm, sampleRate: audioSampleRate, channels: audioChannels)
        }

        guard let pool = adaptor.pixelBufferPool else { throw MuxError.noPool }

        let ts: Int32 = 600
        rgb.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let src = raw.bindMemory(to: UInt8.self).baseAddress!
            for f in 0..<frames {
                while !input.isReadyForMoreMediaData { usleep(500) }
                var pbOut: CVPixelBuffer?
                CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pbOut)
                guard let pb = pbOut else { continue }
                CVPixelBufferLockBaseAddress(pb, [])
                if let base = CVPixelBufferGetBaseAddress(pb) {
                    let dst = base.assumingMemoryBound(to: UInt8.self)
                    let bpr = CVPixelBufferGetBytesPerRow(pb)
                    for h in 0..<height {
                        let rowBase = ((f * height + h) * width) * 3
                        for w in 0..<width {
                            let s = rowBase + w * 3
                            let d = h * bpr + w * 4
                            dst[d + 0] = src[s + 2] // B
                            dst[d + 1] = src[s + 1] // G
                            dst[d + 2] = src[s + 0] // R
                            dst[d + 3] = 255        // A
                        }
                    }
                }
                CVPixelBufferUnlockBaseAddress(pb, [])
                let pts = CMTime(value: Int64(f) * Int64(ts) / Int64(max(fps, 1)), timescale: ts)
                adaptor.append(pb, withPresentationTime: pts)
            }
        }
        input.markAsFinished()

        let sem = DispatchSemaphore(value: 0)
        writer.finishWriting { sem.signal() }
        sem.wait()
        if writer.status != .completed {
            throw MuxError.finishFailed(String(describing: writer.error))
        }
    }

    /// Wrap interleaved s16le PCM in a single CMSampleBuffer and append it to the
    /// audio writer input (the writer transcodes LPCM → AAC).
    nonisolated static func appendAudio(_ ai: AVAssetWriterInput, pcm: Data, sampleRate: Int, channels: Int) throws {
        // Every early return MUST finish the input: an added-but-never-finished
        // audio input is a starved sibling the muxer waits on forever, wedging
        // the video loop (the documented multi-input AVAssetWriter class).
        guard channels > 0, sampleRate > 0 else { ai.markAsFinished(); return }
        let bytesPerFrame = 2 * channels
        let numFrames = pcm.count / bytesPerFrame
        guard numFrames > 0 else { ai.markAsFinished(); return }

        var asbd = AudioStreamBasicDescription(
            mSampleRate: Float64(sampleRate),
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
            mBytesPerPacket: UInt32(bytesPerFrame),
            mFramesPerPacket: 1,
            mBytesPerFrame: UInt32(bytesPerFrame),
            mChannelsPerFrame: UInt32(channels),
            mBitsPerChannel: 16,
            mReserved: 0)
        // A channel layout is required for the AAC encoder to accept multi-channel
        // input (otherwise finishWriting fails with "Cannot Encode Media").
        var layout = AudioChannelLayout()
        layout.mChannelLayoutTag = channels == 1 ? kAudioChannelLayoutTag_Mono : kAudioChannelLayoutTag_Stereo
        var format: CMAudioFormatDescription?
        let fmtStatus = withUnsafePointer(to: &layout) { lp in
            CMAudioFormatDescriptionCreate(allocator: kCFAllocatorDefault, asbd: &asbd,
                                           layoutSize: MemoryLayout<AudioChannelLayout>.size, layout: lp,
                                           magicCookieSize: 0, magicCookie: nil, extensions: nil,
                                           formatDescriptionOut: &format)
        }
        guard fmtStatus == noErr, let fmt = format else { throw MuxError.audioBuffer }

        var blockBuffer: CMBlockBuffer?
        let dataLen = numFrames * bytesPerFrame
        guard CMBlockBufferCreateWithMemoryBlock(allocator: kCFAllocatorDefault, memoryBlock: nil,
                                                  blockLength: dataLen, blockAllocator: kCFAllocatorDefault,
                                                  customBlockSource: nil, offsetToData: 0, dataLength: dataLen,
                                                  flags: kCMBlockBufferAssureMemoryNowFlag, blockBufferOut: &blockBuffer) == noErr,
              let bb = blockBuffer else { throw MuxError.audioBuffer }
        let copyStatus = pcm.withUnsafeBytes { raw -> OSStatus in
            CMBlockBufferReplaceDataBytes(with: raw.baseAddress!, blockBuffer: bb,
                                          offsetIntoDestination: 0, dataLength: dataLen)
        }
        guard copyStatus == noErr else { throw MuxError.audioBuffer }

        var sampleBuffer: CMSampleBuffer?
        var timing = CMSampleTimingInfo(duration: CMTime(value: 1, timescale: Int32(sampleRate)),
                                        presentationTimeStamp: CMTime(value: 0, timescale: Int32(sampleRate)),
                                        decodeTimeStamp: .invalid)
        var sampleSize = bytesPerFrame
        guard CMSampleBufferCreate(allocator: kCFAllocatorDefault, dataBuffer: bb, dataReady: true,
                                   makeDataReadyCallback: nil, refcon: nil, formatDescription: fmt,
                                   sampleCount: numFrames, sampleTimingEntryCount: 1, sampleTimingArray: &timing,
                                   sampleSizeEntryCount: 1, sampleSizeArray: &sampleSize,
                                   sampleBufferOut: &sampleBuffer) == noErr, let sb = sampleBuffer
        else { throw MuxError.audioBuffer }

        while !ai.isReadyForMoreMediaData { usleep(500) }
        ai.append(sb)
        ai.markAsFinished()
    }

    // MARK: - Private

    private func insertRecent(_ path: String) {
        recent.removeAll { $0 == path }
        recent.insert(path, at: 0)
        if recent.count > 40 { recent.removeLast(recent.count - 40) }
    }

    private func loadRecent() {
        let root = MediaStorage.videosRoot
        let fm = FileManager.default
        guard let days = try? fm.contentsOfDirectory(atPath: root) else { return }
        var paths: [(String, Date)] = []
        for day in days.sorted(by: >) {
            let dayDir = (root as NSString).appendingPathComponent(day)
            guard let files = try? fm.contentsOfDirectory(atPath: dayDir) else { continue }
            for f in files where f.hasSuffix(".mp4") {
                let full = (dayDir as NSString).appendingPathComponent(f)
                let date = (try? fm.attributesOfItem(atPath: full)[.modificationDate] as? Date) ?? .distantPast
                paths.append((full, date))
            }
        }
        recent = paths.sorted { $0.1 > $1.1 }.prefix(40).map(\.0)
    }

    private static func makeOutputPath(prompt: String) -> String {
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd"
        let day = df.string(from: Date())
        let dayDir = (MediaStorage.videosRoot as NSString).appendingPathComponent(day)
        try? FileManager.default.createDirectory(atPath: dayDir, withIntermediateDirectories: true)
        let tf = DateFormatter()
        tf.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let slug = prompt
            .lowercased()
            .replacingOccurrences(of: #"[^a-z0-9]+"#, with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
            .prefix(40)
        let filename = "\(tf.string(from: Date()))_\(slug).mp4"
        return (dayDir as NSString).appendingPathComponent(filename)
    }
}
