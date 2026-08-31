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
    @Published private(set) var livePreview: NSImage? = nil
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
        switch p {
        case .running: break
        default: livePreview = nil
        }
    }

    private func setLivePreview(_ img: NSImage?, for gen: Int) {
        guard gen == generationSeq else { return }
        livePreview = img
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
        guard request.lanModelId != nil || ServerManager.resolveModelDir(repo: request.model.repo) != nil else {
            phase = .failed("Model \(request.model.repo) is not downloaded. Download it first.")
            return
        }

        task?.cancel()
        generationSeq += 1
        let gen = generationSeq
        livePreview = nil
        phase = .running(step: 0, total: 3, message: "Loading model…")
        log = []

        let outputPath = Self.makeOutputPath(prompt: request.prompt)
        let prompt = request.prompt
        let fps = request.fps
        let steps = request.steps
        let keep = request.keepResident
        let firstFramePath = request.firstFrameImagePath
        let lastFramePath = request.lastFrameImagePath
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
                // The last-frame anchor reads the same way. Both keyframes are
                // ordinary image files; the server owns the per-anchor resize
                // policy (first stretches, last center-covers).
                let lastFrameB64: String? = await Task.detached(priority: .userInitiated) {
                    lastFramePath.flatMap { path in
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
                // ref2va: reading 9 images and pulling frames out of 3 clips is
                // seconds of work, so it goes off-main like the two above. A
                // reference that cannot be read is a hard error — generating
                // while dropping one silently is exactly what the server's
                // named 400s exist to prevent.
                let refPayloads: VideoRefPayloads? = await Task.detached(priority: .userInitiated) {
                    Self.refPayloads(for: request)
                }.value
                guard let refs = refPayloads else {
                    setPhase(.failed("Couldn't read one of the reference files. Images must be PNG or JPEG, clips a QuickTime/MP4 movie of at least 5 frames, and audio a WAV, MP3, M4A or AAC file."), for: gen)
                    return
                }
                let (port, modelId, unloadId) = try await server.prepareGenModel(
                    lanModelId: request.lanModelId, repo: request.model.repo)
                loadedId = unloadId
                // Cancelled right after load: deliberately leave the model
                // resident (an unload from a cancelled task can't run anyway,
                // and the likely next action is a retry — the residency row
                // in the pane shows the state).
                if Task.isCancelled { setPhase(.cancelled, for: gen); return }
                let body = Self.requestBody(model: modelId, prompt: prompt,
                                            request: request, firstFrameB64: firstFrameB64,
                                            lastFrameB64: lastFrameB64,
                                            audioB64: audioB64, refs: refs)
                // SSE: the server pushes `progress` events per denoise step, then a
                // `complete` event with the frames. Drive a determinate bar from them.
                var decoded: DecodedFrames? = nil
                // Live ETA from the run's own cadence — the only estimate that
                // knows what this machine is doing right now. A three-hour job
                // with a bar and no number is indistinguishable from a hang.
                var clock = H3StepClock()
                // Floor + tail for the live number (H3 only; LTX keeps the
                // plain lap mean — its laps are near-uniform and it has no
                // pre-run model to floor against). `fast` is effectiveFast:
                // turbo forces the server recipe off, never !bestQuality alone.
                let pricing: (perStep: Double, tail: Double) =
                    request.model.backend == .minimaxH3
                    ? H3TimeEstimate.livePricing(model: request.model,
                                                 width: request.width, height: request.height,
                                                 frames: request.numFrames, steps: steps,
                                                 fast: !request.bestQuality && !request.turbo)
                    : (0, 0)
                let startedAt = ProcessInfo.processInfo.systemUptime
                for try await ev in api.streamGeneration(
                    port: port, path: "/v1/video/generations", json: body) {
                    switch ev["type"] as? String {
                    case "progress":
                        let step = ev["step"] as? Int ?? 0
                        let total = ev["total"] as? Int ?? steps
                        let stage = ev["stage"] as? String ?? "Generating"
                        clock.observe(step: step)
                        var message = "\(stage)…"
                        if let eta = clock.eta(totalSteps: max(total, 1),
                                               floorPerStep: pricing.perStep, tail: pricing.tail),
                           eta > 0 {
                            message += " \(H3TimeEstimate.duration(eta)) left"
                        }
                        if let data = MediaSSE.previewJPEG(ev), let img = NSImage(data: data) {
                            setLivePreview(img, for: gen)
                        }
                        setPhase(.running(step: step, total: max(total, 1), message: message), for: gen)
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
                // Calibrate this Mac against the anchor model, so the next
                // estimate is measured rather than extrapolated. Recorded from
                // the SAMPLING span only — the encode below is ours, not the
                // model's, and the estimate does not include it.
                H3RunHistory.remember(model: request.model, width: request.width, height: request.height,
                                      frames: request.numFrames, steps: steps, fast: !request.bestQuality,
                                      measuredSeconds: ProcessInfo.processInfo.systemUptime - startedAt)
                setPhase(.running(step: steps, total: steps, message: "Encoding mp4…"), for: gen)
                let outFps = frames.fps > 0 ? frames.fps : fps
                let settings = Self.settingsText(
                    request, modelId: modelId,
                    outputWidth: frames.width, outputHeight: frames.height,
                    outputFrames: frames.frames, outputFps: outFps)
                try await Task.detached(priority: .userInitiated) {
                    try VideoGenService.writeMP4(
                        rgb: frames.rgb, frames: frames.frames,
                        width: frames.width, height: frames.height,
                        fps: outFps, to: URL(fileURLWithPath: outputPath),
                        audioPCM: frames.audioPCM, audioSampleRate: frames.audioSampleRate,
                        audioChannels: frames.audioChannels)
                    // The mp4 is the primary artifact. Match audio/music generation:
                    // a sidecar failure must not discard a successfully encoded clip.
                    try? VideoGenService.writeSettingsSidecar(settings, forVideo: outputPath)
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

    /// Awaitable generation for the agent's `generate_video` tool. Same load →
    /// stream → mux → unload pipeline as `generate`, returning the output mp4
    /// path (or throwing), but WITHOUT touching this service's UI state
    /// (`phase`/`task`/`recent`/`generationSeq`) — so a chat generation never
    /// hijacks the Video window. `onProgress` drives the chat's own meter, which
    /// matters most here: this is the tool that takes minutes.
    func generateForAgent(_ request: VideoGenRequest, server: ServerManager,
                          onProgress: ((MediaGenProgress) -> Void)? = nil) async throws -> String {
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MediaGenError.emptyInput("Prompt")
        }
        guard request.lanModelId != nil || ServerManager.resolveModelDir(repo: request.model.repo) != nil else {
            throw MediaGenError.notDownloaded(request.model.name)
        }

        let outputPath = Self.makeOutputPath(prompt: request.prompt)
        let keep = request.keepResident
        let steps = request.steps
        let startedAt = Date()
        func report(_ step: Int, _ total: Int, _ message: String) {
            onProgress?(MediaGenProgress(kind: .video, step: step, total: total,
                                         message: message, startedAt: startedAt))
        }
        report(0, 0, "Loading model")

        let (port, modelId, unloadId) = try await server.prepareGenModel(
            lanModelId: request.lanModelId, repo: request.model.repo)
        func releaseIfNeeded() async {
            if !keep, let id = unloadId { try? await server.unloadModel(id: id) }
        }
        do {
            var decoded: DecodedFrames? = nil
            let body = Self.requestBody(model: modelId, prompt: request.prompt,
                                        request: request, firstFrameB64: nil, audioB64: nil)
            for try await ev in api.streamGeneration(
                port: port, path: "/v1/video/generations", json: body) {
                switch MediaSSE.classify(ev) {
                case .progress(let step, let total, let stage):
                    report(step, total == 0 ? steps : total, MediaSSE.stageLabel(stage))
                case .complete:
                    decoded = Self.decodeFrames(ev)
                case .failed(let m):
                    throw MediaGenError.server(m)
                case .ignored:
                    break
                }
            }
            guard let frames = decoded else {
                throw MediaGenError.server("Server returned no video frames.")
            }
            report(steps, steps, "Encoding mp4")
            let outFps = frames.fps > 0 ? frames.fps : request.fps
            let settings = Self.settingsText(
                request, modelId: modelId,
                outputWidth: frames.width, outputHeight: frames.height,
                outputFrames: frames.frames, outputFps: outFps)
            try await Task.detached(priority: .userInitiated) {
                try VideoGenService.writeMP4(
                    rgb: frames.rgb, frames: frames.frames,
                    width: frames.width, height: frames.height,
                    fps: outFps, to: URL(fileURLWithPath: outputPath),
                    audioPCM: frames.audioPCM, audioSampleRate: frames.audioSampleRate,
                    audioChannels: frames.audioChannels)
                try? VideoGenService.writeSettingsSidecar(settings, forVideo: outputPath)
            }.value
            await releaseIfNeeded()
            return outputPath
        } catch {
            await releaseIfNeeded()
            throw error
        }
    }

    func cancel() {
        task?.cancel()
        task = nil
        // Instant feedback; the cancelled task's own catch re-confirms it.
        if isRunning {
            phase = .cancelled
            livePreview = nil
        }
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
                                        lastFrameB64: String? = nil,
                                        audioB64: String? = nil,
                                        refs: VideoRefPayloads = .init()) -> [String: Any] {
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
        ]
        // Send only what the BACKEND declares it can honor. Hiding a control is
        // not the same as not sending its field: `pipeline` used to go out
        // unconditionally, which meant every MiniMax-H3 request carried a field
        // that backend has no concept of. Gating the request against the same
        // capabilities the pane gates its controls on keeps the two honest.
        if request.model.supportsPipelineModes {
            body["pipeline"] = pipeline
        }
        if request.model.supportsCFG, !dropGuidance {
            body["cfg_scale"] = request.cfgScale
            body["stg_scale"] = request.stgScale
            // The audio guider only exists on the a2vid path, so the scale
            // rides the clip rather than the preset: without one it would set
            // a knob on a guider that never runs. It drops with the rest of
            // the guidance on an upgraded one-stage request, because the whole
            // point of that drop is to let the server's reference two-stage
            // defaults (3.0 video / 7.0 audio) apply as a SET.
            if hasAudio { body["cfg_audio_scale"] = request.cfgAudioScale }
        }
        // Stage-2 refine steps. Two-stage only — one-stage has no refine pass,
        // so the field would be a no-op the server still parses. 0 is Auto and
        // stays absent, keeping "absent = the server's default" true.
        if request.model.supportsPipelineModes, pipeline != "one_stage", request.stage2Steps > 0 {
            body["stage2_steps"] = request.stage2Steps
        }
        if let firstFrameB64 { body["first_frame_image"] = firstFrameB64 }
        // The other half of fl2va. Capability-gated like every field above: a
        // preset switch leaves the picked file in state, and LTX's handler has
        // no `last_frame_image` to ignore it with.
        if request.model.supportsLastFrame, let lastFrameB64 {
            body["last_frame_image"] = lastFrameB64
        }
        // The fast recipe is the SERVER's default — the app only speaks up to
        // opt OUT, and only on a backend that has the recipe at all.
        if request.model.supportsFastRecipe, request.bestQuality { body["fast"] = false }
        // The conv decoder is the server's default, so the app only speaks up
        // to ask for the DiffVAE — and only on a pack that ships it.
        if request.model.supportsDiffusionDecoder, request.diffusionDecoder {
            body["decoder"] = "diffusion"
        }
        // Turbo + chained windows: capability-gated like every H3 field above,
        // and emitted only when engaged — the server's defaults are the
        // absent-field behavior.
        if request.model.supportsTurbo, request.turbo { body["turbo"] = true }
        if request.model.supportsChainedWindows, request.chainWindows > 1 {
            body["chain_windows"] = request.chainWindows
        }
        if request.model.supportsAudioInput, hasAudio, let audioB64 { body["audio"] = audioB64 }
        // Stacked style LoRAs (adapters sum, so several can attach at once —
        // see ImageGenService.requestJson for the same pattern). Capability-
        // gated like every other field: a preset switch leaves the rows in
        // state, and a backend without LoRA support must not see the arrays.
        let loras = request.loras.filter { !$0.path.isEmpty }
        if request.model.supportsLoRA, !loras.isEmpty {
            body["lora_paths"] = loras.map(\.path)
            body["lora_scales"] = loras.map(\.scale)
        }
        // ref2va. Gated on the pack's own capability for the same reason as
        // `pipeline` above: an FL2VA checkpoint handed references 400s, and a
        // preset switch leaves the picked files in the request state.
        if request.model.supportsReferences {
            if !refs.images.isEmpty { body["ref_images"] = refs.images }
            if !refs.videos.isEmpty {
                body["ref_videos"] = refs.videos.map { v -> [String: Any] in
                    var o: [String: Any] = ["frames": v.frames]
                    // Omitted rather than null: the soundtrack is a field on
                    // the clip it belongs to, and "absent" is the only way to
                    // say a clip is silent.
                    if let a = v.audio, !a.isEmpty { o["audio"] = a }
                    return o
                }
            }
            if !refs.audios.isEmpty { body["ref_audios"] = refs.audios }
            // `match` is the server's default; only an opt-out is stated.
            if request.refImageSize != .match { body["ref_image_size"] = request.refImageSize.rawValue }
        }
        // Per-step latent previews on the SSE stream (issue #208), opt-in from
        // the pane's own toggle. Absent = off, so a client that never asks
        // pays nothing; asking costs an x0 solve plus a host copy of the
        // previewed frames on every step.
        if request.livePreview {
            body["preview"] = true
            body["preview_frames"] = 1
            body["preview_max_side"] = 256
        }

        return body
    }

    // MARK: - Settings sidecar

    /// Human-readable `<clip>.txt` companion for a generated video. This is
    /// deliberately built from paths/settings, never from the wire body: the
    /// latter contains multi-megabyte base64 images, PCM audio, and video frames.
    /// Optional fields are capability-gated exactly like `requestBody`, so a
    /// stale control value left behind by a preset switch is not documented as
    /// something the selected backend actually used.
    nonisolated static func settingsText(_ request: VideoGenRequest, modelId: String,
                                         outputWidth: Int? = nil, outputHeight: Int? = nil,
                                         outputFrames: Int? = nil, outputFps: Int? = nil) -> String {
        var lines: [String] = [
            "model: \(modelId)",
            "preset: \(request.model.name)",
            "seed: \(request.seed)",
            "width: \(request.width)",
            "height: \(request.height)",
            "frames: \(request.numFrames)",
            "fps: \(request.fps)",
            "steps: \(request.steps)",
        ]

        if let outputWidth, let outputHeight,
           outputWidth != request.width || outputHeight != request.height {
            lines.append("output_width: \(outputWidth)")
            lines.append("output_height: \(outputHeight)")
        }
        if let outputFrames, outputFrames != request.numFrames {
            lines.append("output_frames: \(outputFrames)")
        }
        if let outputFps, outputFps != request.fps {
            lines.append("output_fps: \(outputFps)")
        }

        let hasAudio = request.model.supportsAudioInput &&
            request.audioPath?.isEmpty == false
        let upgradedForAudio = hasAudio && request.mode == .oneStage
        if request.model.supportsPipelineModes {
            let pipeline: String
            if upgradedForAudio {
                pipeline = "two_stage"
            } else {
                switch request.mode {
                case .oneStage:   pipeline = "one_stage"
                case .twoStage:   pipeline = "two_stage"
                case .twoStageHQ: pipeline = "two_stage_hq"
                }
            }
            lines.append("pipeline: \(pipeline)")
        }
        // A one-stage audio request intentionally drops its one-stage guidance
        // values and lets the server use two-stage defaults.
        if request.model.supportsCFG, !upgradedForAudio {
            lines.append("cfg_scale: \(String(format: "%.2f", request.cfgScale))")
            lines.append("stg_scale: \(String(format: "%.2f", request.stgScale))")
        }
        if request.model.supportsFastRecipe {
            lines.append("fast_recipe: \(!request.bestQuality && !request.turbo)")
        }
        if request.model.supportsDiffusionDecoder {
            lines.append("decoder: \(request.diffusionDecoder ? "diffusion" : "convolution")")
        }
        if request.model.supportsTurbo {
            lines.append("turbo: \(request.turbo)")
        }
        if request.model.supportsChainedWindows {
            lines.append("chain_windows: \(max(request.chainWindows, 1))")
        }

        func basename(_ path: String) -> String {
            (path as NSString).lastPathComponent
        }
        if let path = request.firstFrameImagePath, !path.isEmpty {
            lines.append("first_frame: \(basename(path))")
        }
        if let path = request.lastFrameImagePath, !path.isEmpty {
            lines.append("last_frame: \(basename(path))")
        }
        if hasAudio, let path = request.audioPath {
            lines.append("input_audio: \(basename(path))")
        }

        if request.model.supportsLoRA {
            for (index, lora) in request.loras.filter({ !$0.path.isEmpty }).enumerated() {
                lines.append("lora_\(index + 1)_file: \(basename(lora.path))")
                lines.append("lora_\(index + 1)_scale: \(String(format: "%.2f", lora.scale))")
            }
        }
        if request.model.supportsReferences {
            for (index, path) in request.refImagePaths.filter({ !$0.isEmpty }).enumerated() {
                lines.append("reference_image_\(index + 1): \(basename(path))")
            }
            for (index, path) in request.refVideoPaths.filter({ !$0.isEmpty }).enumerated() {
                lines.append("reference_video_\(index + 1): \(basename(path))")
            }
            for (index, path) in request.refAudioPaths.filter({ !$0.isEmpty }).enumerated() {
                lines.append("reference_audio_\(index + 1): \(basename(path))")
            }
            lines.append("reference_image_size: \(request.refImageSize.rawValue)")
        }

        var out = lines.joined(separator: "\n")
        out += "\n\n# Prompt\n" + request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        return out + "\n"
    }

    /// `<clip>.mp4` → `<clip>.txt` companion path.
    nonisolated static func sidecarPath(forVideo videoPath: String) -> String {
        (videoPath as NSString).deletingPathExtension + ".txt"
    }

    /// Kept separate from mp4 encoding so tests can pin actual filesystem
    /// behavior without synthesizing video frames through AVFoundation.
    nonisolated static func writeSettingsSidecar(_ text: String, forVideo videoPath: String) throws {
        try text.write(to: URL(fileURLWithPath: sidecarPath(forVideo: videoPath)),
                       atomically: true, encoding: .utf8)
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

    // MARK: - ref2va reference payloads

    /// Longest edge a reference frame is shipped at. The server normalizes a
    /// reference video to a 768 short edge anyway, so sending phone-native
    /// frames is pure payload — a 124-frame clip at 4K is hundreds of MB of
    /// base64 against a 64 MB request cap.
    nonisolated static let refFrameMaxEdge: CGFloat = 1024

    /// How many frames of a reference clip to actually ship. H3's ladder is
    /// 17k+5 and the SERVER snaps DOWN, so anything past the snapped count is
    /// extracted, encoded, uploaded and discarded. 0 means the clip is too
    /// short to condition on (the server's own 5-frame floor).
    nonisolated static func refVideoFrameCount(available: Int, cap: Int) -> Int {
        var n = min(available, cap)
        while n >= 5, n % 17 != 5 { n -= 1 }
        return n < 5 ? 0 : n
    }

    /// Presentation timestamps for `count` frames at H3's fixed 24 fps.
    nonisolated static func refFrameTimes(count: Int, fps: Int) -> [Double] {
        guard count > 0, fps > 0 else { return [] }
        return (0..<count).map { Double($0) / Double(fps) }
    }

    /// Read an image file straight through as base64 — the server decodes
    /// PNG/JPEG and resizes to the canvas it picked, so re-encoding here would
    /// only lose detail the `max` sizing mode exists to keep.
    nonisolated static func imageFileToBase64(path: String) -> String? {
        (try? Data(contentsOf: URL(fileURLWithPath: path)))?.base64EncodedString()
    }

    /// Pull a reference clip apart into base64 JPEG frames at 24 fps plus its
    /// soundtrack. JPEG, not PNG: a reference is conditioning, not a pixel-exact
    /// input, and PNG frames blow through the request cap.
    nonisolated static func videoFileToRefPayload(path: String, maxFrames: Int) -> VideoRefPayloads.Video? {
        let asset = AVURLAsset(url: URL(fileURLWithPath: path))
        let seconds = CMTimeGetSeconds(asset.duration)
        guard seconds.isFinite, seconds > 0 else { return nil }
        let fps = 24
        let available = Int(seconds * Double(fps))
        let count = refVideoFrameCount(available: available, cap: maxFrames)
        guard count > 0 else { return nil }

        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.requestedTimeToleranceBefore = .zero
        gen.requestedTimeToleranceAfter = .zero
        gen.maximumSize = CGSize(width: refFrameMaxEdge, height: refFrameMaxEdge)

        var frames: [String] = []
        frames.reserveCapacity(count)
        for t in refFrameTimes(count: count, fps: fps) {
            let time = CMTime(seconds: t, preferredTimescale: CMTimeScale(fps * 1000))
            guard let cg = try? gen.copyCGImage(at: time, actualTime: nil) else { return nil }
            let rep = NSBitmapImageRep(cgImage: cg)
            guard let jpeg = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.9])
            else { return nil }
            frames.append(jpeg.base64EncodedString())
        }
        // A silent clip is a legitimate reference: its motion and framing still
        // condition the generation, so a missing track is not a failure.
        return .init(frames: frames, audio: audioFileToWavBase64(path: path))
    }

    /// Resolve a request's reference PATHS into wire payloads. Returns nil when
    /// something the user picked could not be read — the caller surfaces that
    /// rather than generating while quietly dropping a reference.
    nonisolated static func refPayloads(for request: VideoGenRequest) -> VideoRefPayloads? {
        guard request.model.supportsReferences else { return VideoRefPayloads() }
        var out = VideoRefPayloads()
        for p in request.refImagePaths {
            guard let b64 = imageFileToBase64(path: p) else { return nil }
            out.images.append(b64)
        }
        for p in request.refVideoPaths {
            guard let v = videoFileToRefPayload(path: p, maxFrames: request.numFrames) else { return nil }
            out.videos.append(v)
        }
        for p in request.refAudioPaths {
            guard let b64 = audioFileToWavBase64(path: p) else { return nil }
            out.audios.append(b64)
        }
        return out
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

    enum MuxError: Error { case writerInit, noPool, frameBuffer(Int), frameAppend(Int, String), finishFailed(String), audioBuffer }

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
        // Every frame is appended or the mux FAILS. A dropped frame (pool
        // exhausted, encoder refused the buffer) used to `continue` and the
        // writer still finished "successfully" — issue #170's 28 KB black mp4
        // after an hours-long H3 render. Pool misses fall back to a standalone
        // buffer; a refused append throws with the writer's own error.
        var muxFailure: MuxError? = nil
        rgb.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let src = raw.bindMemory(to: UInt8.self).baseAddress!
            for f in 0..<frames {
                while !input.isReadyForMoreMediaData { usleep(500) }
                var pbOut: CVPixelBuffer?
                CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pbOut)
                if pbOut == nil {
                    CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pbOut)
                }
                guard let pb = pbOut else { muxFailure = .frameBuffer(f); return }
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
                if !adaptor.append(pb, withPresentationTime: pts) {
                    muxFailure = .frameAppend(f, String(describing: writer.error))
                    return
                }
            }
        }
        input.markAsFinished()
        if let failure = muxFailure {
            audioInput?.markAsFinished()
            writer.cancelWriting()
            try? FileManager.default.removeItem(at: url)
            throw failure
        }

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
