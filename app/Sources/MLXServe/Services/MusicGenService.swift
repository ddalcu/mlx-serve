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
    nonisolated static func requestBody(_ request: MusicGenRequest, modelName: String) -> [String: Any] {
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
        let lyrics = request.lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        if !lyrics.isEmpty { body["lyrics"] = lyrics }
        // The musical-metadata knob set is ACE-Step's; Music 3 names each
        // field a 400 — gate the FIELDS here, not just the pane's controls
        // (values may linger from an ACE session).
        if request.model.supportsMusicalMeta {
            let lang = request.vocalLanguage.trimmingCharacters(in: .whitespacesAndNewlines)
            if !lang.isEmpty { body["vocal_language"] = lang }
            if let bpm = request.bpm { body["bpm"] = bpm }
            let ks = request.keyscale.trimmingCharacters(in: .whitespacesAndNewlines)
            if !ks.isEmpty { body["keyscale"] = ks }
            let ts = request.timesignature.trimmingCharacters(in: .whitespacesAndNewlines)
            if !ts.isEmpty { body["timesignature"] = ts }
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
        if request.model.supportsMusicalMeta {
            let lang = request.vocalLanguage.trimmingCharacters(in: .whitespacesAndNewlines)
            if !lang.isEmpty, lang != "unknown" { lines.append("vocal_language: \(lang)") }
            if let bpm = request.bpm { lines.append("bpm: \(bpm)") }
            let ks = request.keyscale.trimmingCharacters(in: .whitespacesAndNewlines)
            if !ks.isEmpty { lines.append("keyscale: \(ks)") }
            let ts = request.timesignature.trimmingCharacters(in: .whitespacesAndNewlines)
            if !ts.isEmpty { lines.append("timesignature: \(ts)") }
        }
        var out = lines.joined(separator: "\n")
        out += "\n\n# Style prompt\n" + request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let lyr = request.lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        out += "\n\n# Lyrics\n" + (lyr.isEmpty ? "[Instrumental]" : lyr)
        return out + "\n"
    }

    /// `<track>.wav` → `<track>.txt` companion path.
    nonisolated static func sidecarPath(forWav wavPath: String) -> String {
        (wavPath as NSString).deletingPathExtension + ".txt"
    }

    /// Generate through the ONE main server: ensure running (headless if
    /// needed), load the music model on demand, stream
    /// `/v1/audio/music-generations`, then unload unless "Keep loaded" is set.
    func generate(_ request: MusicGenRequest, server: ServerManager) {
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            phase = .failed("Prompt is empty.")
            return
        }
        if request.model.requiresLyrics,
           request.lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            phase = .failed("\(request.model.name) needs lyrics — put structure tags like [verse] on their own lines.")
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
                let reqJson = Self.requestBody(request, modelName: modelId)
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
        if request.model.requiresLyrics,
           request.lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw MediaGenError.emptyInput("Lyrics (this model is lyric-conditioned)")
        }
        guard request.lanModelId != nil || ServerManager.resolveModelDir(repo: request.model.repo) != nil else {
            throw MediaGenError.notDownloaded(request.model.name)
        }

        let outputPath = Self.makeOutputPath(prompt: request.prompt)
        let keep = request.keepResident
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
            let reqJson = Self.requestBody(request, modelName: modelId)
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
