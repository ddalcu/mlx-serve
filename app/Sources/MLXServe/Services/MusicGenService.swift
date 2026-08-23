import Foundation
import SwiftUI
import AppKit

/// Drives prompt-based music generation (ACE-Step) on the native mlx-serve
/// server. Mirrors `AudioGenService`: same `Phase` lifecycle, same JSON-event
/// stream, writes a `.wav` under `~/.mlx-serve/generations/music`.
@MainActor
final class MusicGenService: ObservableObject {

    enum Phase: Equatable {
        case idle
        case running(step: Int, total: Int, message: String)
        case completed(path: String)
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var recent: [String] = []
    @Published private(set) var log: [String] = []

    private var task: Task<Void, Never>?
    private let api = APIClient()

    init() {
        loadRecent()
    }

    var isRunning: Bool {
        if case .running = phase { return true }
        return false
    }

    /// The `/v1/audio/music-generations` request body. Static + pure so unit
    /// tests pin the wire contract (omit-empty fields, seed resolution).
    nonisolated static func requestBody(_ request: MusicGenRequest, modelName: String, refAudioB64: String? = nil, srcAudioB64: String? = nil) -> [String: Any] {
        // Sticky settings outlive a model switch: clamp the duration into THIS
        // model's server-valid range rather than earn a 400.
        let range = request.model.durationRange
        let duration = Int(min(max(Double(request.durationSeconds), range.lowerBound), range.upperBound))
        var body: [String: Any] = [
            "model": modelName,
            "prompt": request.prompt,
            "duration_seconds": duration,
            "stream": true,
        ]
        // `instrumental` and lyrics are a named 400 on BOTH backends, so the
        // flag WINS here rather than letting the pair reach the server. On
        // Music 3 an omitted lyrics field is the only spelling of "no words"
        // that is accepted at all.
        if request.instrumental {
            body["instrumental"] = true
        } else {
            let lyrics = request.lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
            if !lyrics.isEmpty { body["lyrics"] = lyrics }
        }
        // Only the backend that reads `steps` gets it — ACE-Step Turbo ignores
        // the field. Clamping mirrors the duration clamp above: sticky settings
        // outlive a model switch and must not earn a 400.
        if request.model.supportsSteps, let steps = request.steps {
            let r = request.model.stepsRange
            body["steps"] = min(max(steps, r.lowerBound), r.upperBound)
        }
        // The musical-metadata knob set is ACE-Step's; Music 3 names each
        // field a 400 — gate the FIELDS here, not just the pane's controls
        // (values may linger from an ACE session).
        // Tempo and key go to BOTH engines; the server decides whether they
        // are conditioning fields (ACE-Step) or caption text (Music 3).
        if request.model.supportsTempoAndKey {
            if let bpm = request.bpm { body["bpm"] = bpm }
            let ks = request.keyscale.trimmingCharacters(in: .whitespacesAndNewlines)
            if !ks.isEmpty { body["keyscale"] = ks }
        }
        // These two remain ACE-Step-only and are a named 400 elsewhere, so the
        // FIELDS are gated here and not just the pane's controls — values
        // linger in @State across a model switch.
        if request.model.supportsMusicalMeta {
            let lang = request.vocalLanguage.trimmingCharacters(in: .whitespacesAndNewlines)
            if !lang.isEmpty { body["vocal_language"] = lang }
            let ts = request.timesignature.trimmingCharacters(in: .whitespacesAndNewlines)
            if !ts.isEmpty { body["timesignature"] = ts }
        }
        // Reference audio is ACE-Step's timbre slot; Music 3 names the field a
        // 400, so the FIELD is gated like `steps` (a clip lingers in @State
        // across a model switch).
        if request.model.supportsReferenceAudio, let refAudioB64, !refAudioB64.isEmpty {
            body["ref_audio"] = refAudioB64
        }
        // Source-audio tasks are ACE-Step's; the task decides which of its
        // knobs travel (the server names a stray one a 400), and without a
        // source clip the request stays plain text2music.
        if request.model.supportsSourceAudio, request.task.needsSource, let srcAudioB64, !srcAudioB64.isEmpty {
            body["task"] = request.task.rawValue
            body["src_audio"] = srcAudioB64
            switch request.task {
            case .cover:
                body["cover_strength"] = min(max(request.coverStrength, 0), 1)
                body["cover_noise_strength"] = min(max(request.coverNoiseStrength, 0), 1)
            case .complete:
                let classes = request.trackClasses.filter { MusicTask.trackClasses.contains($0) }
                if !classes.isEmpty { body["track_classes"] = classes }
            case .text2music:
                break
            }
        }
        // -1 = fresh random seed, resolved HERE so the log can show it.
        body["seed"] = request.seed >= 0 ? request.seed : Int.random(in: 0..<1_000_000_000)
        return body
    }

    /// The `<track>.txt` settings sidecar written beside each generated WAV so
    /// a track is reproducible/documented. `resolvedSeed` is the concrete seed
    /// actually used (never -1). Omits fields the request left to the model.
    nonisolated static func settingsText(_ request: MusicGenRequest, resolvedSeed: Int, modelName: String) -> String {
        var lines: [String] = [
            "model: \(modelName)",
            "duration_seconds: \(request.durationSeconds)",
            "seed: \(resolvedSeed)",
        ]
        // A setting that changed the output but not the sidecar is a silent
        // setting — the .txt is what makes a track reproducible.
        if request.instrumental { lines.append("instrumental: true") }
        if request.model.supportsSteps, let steps = request.steps {
            let r = request.model.stepsRange
            lines.append("steps: \(min(max(steps, r.lowerBound), r.upperBound))")
        }
        if request.model.supportsTempoAndKey {
            if let bpm = request.bpm { lines.append("bpm: \(bpm)") }
            let ks = request.keyscale.trimmingCharacters(in: .whitespacesAndNewlines)
            if !ks.isEmpty { lines.append("keyscale: \(ks)") }
        }
        if request.model.supportsMusicalMeta {
            let lang = request.vocalLanguage.trimmingCharacters(in: .whitespacesAndNewlines)
            if !lang.isEmpty, lang != "unknown" { lines.append("vocal_language: \(lang)") }
            let ts = request.timesignature.trimmingCharacters(in: .whitespacesAndNewlines)
            if !ts.isEmpty { lines.append("timesignature: \(ts)") }
        }
        if request.model.supportsReferenceAudio, let ref = request.refAudioPath, !ref.isEmpty {
            lines.append("ref_audio: \((ref as NSString).lastPathComponent)")
        }
        if request.model.supportsSourceAudio, request.task.needsSource, let src = request.srcAudioPath, !src.isEmpty {
            lines.append("task: \(request.task.rawValue)")
            lines.append("src_audio: \((src as NSString).lastPathComponent)")
            if request.task == .cover {
                lines.append("cover_strength: \(request.coverStrength)")
                lines.append("cover_noise_strength: \(request.coverNoiseStrength)")
            } else if !request.trackClasses.isEmpty {
                lines.append("track_classes: \(request.trackClasses.joined(separator: ", "))")
            }
        }
        var out = lines.joined(separator: "\n")
        out += "\n\n# Style prompt\n" + request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let lyr = request.lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        out += "\n\n# Lyrics\n" + (request.instrumental || lyr.isEmpty ? "[Instrumental]" : lyr)
        return out + "\n"
    }

    /// The reference clip (already a 48 kHz stereo WAV from
    /// `AudioReference.referenceWav`) as base64 for `ref_audio`; nil when the
    /// model has no timbre slot or no clip is attached.
    nonisolated static func referenceB64(_ request: MusicGenRequest) -> String? {
        guard request.model.supportsReferenceAudio, let path = request.refAudioPath, !path.isEmpty else { return nil }
        return (try? Data(contentsOf: URL(fileURLWithPath: path)))?.base64EncodedString()
    }

    /// The cover / complete source as base64; nil unless the model and task
    /// take one.
    nonisolated static func sourceB64(_ request: MusicGenRequest) -> String? {
        guard request.model.supportsSourceAudio, request.task.needsSource,
              let path = request.srcAudioPath, !path.isEmpty else { return nil }
        return (try? Data(contentsOf: URL(fileURLWithPath: path)))?.base64EncodedString()
    }

    /// Cover mode reads the FSQ tokenizer, shipped as `fsq.safetensors` beside
    /// `model.safetensors`. Packs downloaded before it exist without the file.
    nonisolated static let coverWeightsFile = "fsq.safetensors"
    nonisolated static func coverWeightsMissing(packDir: String) -> Bool {
        !FileManager.default.fileExists(atPath: (packDir as NSString).appendingPathComponent(coverWeightsFile))
    }

    /// `<track>.wav` → `<track>.txt` companion path.
    nonisolated static func sidecarPath(forWav wavPath: String) -> String {
        (wavPath as NSString).deletingPathExtension + ".txt"
    }

    /// Generate through the ONE main server: ensure running (headless if
    /// needed), load the music model on demand, stream
    /// `/v1/audio/music-generations`, then unload unless "Keep loaded" is set.
    func generate(_ request: MusicGenRequest, server: ServerManager, downloads: DownloadManager? = nil) {
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            phase = .failed("Prompt is empty.")
            return
        }
        if request.model.supportsSourceAudio, request.task.needsSource, request.srcAudioPath == nil {
            phase = .failed("\(request.task.label) needs a source audio file.")
            return
        }
        // TEMPORARY migration (2026-08-22): a pack downloaded before cover mode
        // lacks fsq.safetensors; fetch just that file into the pack from the
        // same repo, then generate. Drop once installs have re-downloaded.
        if let downloads, request.lanModelId == nil, request.task == .cover,
           let dir = ServerManager.resolveModelDir(repo: request.model.repo),
           Self.coverWeightsMissing(packDir: dir) {
            task?.cancel()
            phase = .running(step: 0, total: 0, message: "Downloading cover weights (\(Self.coverWeightsFile))…")
            downloads.startPackFile(repoId: request.model.repo, fileName: Self.coverWeightsFile) { [weak self] in
                self?.generate(request, server: server)
            }
            return
        }
        if !MusicGenRequest.lyricsSatisfied(model: request.model, lyrics: request.lyrics,
                                            instrumental: request.instrumental) {
            phase = .failed("\(request.model.name) needs lyrics — put structure tags like [verse] on their own lines, or tick Instrumental.")
            return
        }
        guard request.lanModelId != nil || ServerManager.resolveModelDir(repo: request.model.repo) != nil else {
            phase = .failed("Model \(request.model.repo) is not installed. Convert or download it first.")
            return
        }

        task?.cancel()
        phase = .running(step: 0, total: 3, message: "Loading model…")
        log = []

        let outputPath = Self.makeOutputPath(prompt: request.prompt)
        let keep = request.keepResident
        let refB64 = Self.referenceB64(request)
        let srcB64 = Self.sourceB64(request)

        task = Task {
            var loadedId: String? = nil
            func releaseIfNeeded() async {
                if !keep, let id = loadedId { try? await server.unloadModel(id: id) }
            }
            do {
                let (port, modelId, unloadId) = try await server.prepareGenModel(
                    lanModelId: request.lanModelId, repo: request.model.repo)
                loadedId = unloadId
                if Task.isCancelled { await releaseIfNeeded(); phase = .idle; return }
                // SSE stages: encode (conditioning) → diffuse (8 turbo steps)
                // → decode (VAE chunks); the `complete` event carries the WAV.
                var wav: Data? = nil
                let reqJson = Self.requestBody(request, modelName: modelId, refAudioB64: refB64, srcAudioB64: srcB64)
                let resolvedSeed = reqJson["seed"] as? Int ?? request.seed
                for try await ev in api.streamGeneration(
                    port: port, path: "/v1/audio/music-generations",
                    json: reqJson) {
                    switch ev["type"] as? String {
                    case "progress":
                        let step = ev["step"] as? Int ?? 0
                        let total = ev["total"] as? Int ?? 0
                        let stage = ev["stage"] as? String ?? "Generating"
                        let label: String
                        switch stage {
                        case "encode", "prefill": label = "Encoding prompt…"
                        case "frames": label = "Composing (frame \(step)/\(total))…"
                        case "diffuse": label = "Composing (step \(step)/\(total))…"
                        case "decode": label = "Rendering audio (\(step)/\(total))…"
                        default: label = "\(stage)…"
                        }
                        phase = .running(step: step, total: total, message: label)
                    case "complete":
                        if let b64 = ev["data"] as? String { wav = Data(base64Encoded: b64) }
                    case "error":
                        await releaseIfNeeded()
                        phase = .failed(ev["message"] as? String ?? "Music generation failed.")
                        return
                    default:
                        break
                    }
                }
                await releaseIfNeeded()
                guard let wav, wav.count > 44 else {
                    phase = .failed("Server returned an empty audio response.")
                    return
                }
                try wav.write(to: URL(fileURLWithPath: outputPath))
                // Settings sidecar: <track>.txt with the prompt/lyrics/params,
                // so every generated track is documented + reproducible.
                let settings = Self.settingsText(request, resolvedSeed: resolvedSeed, modelName: modelId)
                try? settings.write(to: URL(fileURLWithPath: Self.sidecarPath(forWav: outputPath)),
                                    atomically: true, encoding: .utf8)
                phase = .completed(path: outputPath)
                insertRecent(outputPath)
            } catch is CancellationError {
                await releaseIfNeeded()
                phase = .idle
            } catch {
                await releaseIfNeeded()
                phase = .failed(error.localizedDescription)
            }
        }
    }

    /// Awaitable generation for the agent's `generate_music` tool. Same load →
    /// stream → write → unload pipeline as `generate`, returning the output WAV
    /// path (or throwing), but WITHOUT touching this service's UI state
    /// (`phase`/`task`/`recent`) — so a chat generation never hijacks the Music
    /// window. `onProgress` drives the chat's own meter.
    func generateForAgent(_ request: MusicGenRequest, server: ServerManager,
                          onProgress: ((MediaGenProgress) -> Void)? = nil) async throws -> String {
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MediaGenError.emptyInput("Prompt")
        }
        if !MusicGenRequest.lyricsSatisfied(model: request.model, lyrics: request.lyrics,
                                            instrumental: request.instrumental) {
            throw MediaGenError.emptyInput("Lyrics (this model is lyric-conditioned; or set instrumental)")
        }
        guard request.lanModelId != nil || ServerManager.resolveModelDir(repo: request.model.repo) != nil else {
            throw MediaGenError.notDownloaded(request.model.name)
        }

        let outputPath = Self.makeOutputPath(prompt: request.prompt)
        let keep = request.keepResident
        let refB64 = Self.referenceB64(request)
        let startedAt = Date()
        func report(_ step: Int, _ total: Int, _ message: String) {
            onProgress?(MediaGenProgress(kind: .music, step: step, total: total,
                                         message: message, startedAt: startedAt))
        }
        report(0, 0, "Loading model")

        let (port, modelId, unloadId) = try await server.prepareGenModel(
            lanModelId: request.lanModelId, repo: request.model.repo)
        func releaseIfNeeded() async {
            if !keep, let id = unloadId { try? await server.unloadModel(id: id) }
        }
        do {
            var wav: Data? = nil
            let reqJson = Self.requestBody(request, modelName: modelId, refAudioB64: refB64)
            let resolvedSeed = reqJson["seed"] as? Int ?? request.seed
            for try await ev in api.streamGeneration(
                port: port, path: "/v1/audio/music-generations", json: reqJson) {
                switch MediaSSE.classify(ev) {
                case .progress(let step, let total, let stage):
                    report(step, total, MediaSSE.stageLabel(stage))
                case .complete:
                    if let b64 = ev["data"] as? String { wav = Data(base64Encoded: b64) }
                case .failed(let m):
                    throw MediaGenError.server(m)
                case .ignored:
                    break
                }
            }
            guard let wav, wav.count > 44 else {
                throw MediaGenError.server("Server returned an empty audio response.")
            }
            try wav.write(to: URL(fileURLWithPath: outputPath))
            try? Self.settingsText(request, resolvedSeed: resolvedSeed, modelName: modelId)
                .write(to: URL(fileURLWithPath: Self.sidecarPath(forWav: outputPath)),
                       atomically: true, encoding: .utf8)
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
    }

    // MARK: - Private

    private func insertRecent(_ path: String) {
        recent = MediaRecents.inserting(path, into: recent)
    }

    private func loadRecent() {
        recent = MediaRecents.scan(root: MediaStorage.musicRoot, suffix: ".wav")
    }

    /// Slug + dated `.wav` path under `musicRoot`, mirroring the audio output
    /// layout. `internal static` so a unit test can pin the slug contract.
    nonisolated static func makeOutputPath(prompt: String) -> String {
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd"
        let day = df.string(from: Date())
        let dayDir = (MediaStorage.musicRoot as NSString).appendingPathComponent(day)
        try? FileManager.default.createDirectory(atPath: dayDir, withIntermediateDirectories: true)
        let tf = DateFormatter()
        tf.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let slug = prompt
            .lowercased()
            .replacingOccurrences(of: #"[^a-z0-9]+"#, with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
            .prefix(40)
        let filename = "\(tf.string(from: Date()))_\(slug).wav"
        return (dayDir as NSString).appendingPathComponent(filename)
    }
}
