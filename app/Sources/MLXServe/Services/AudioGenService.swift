import Foundation
import SwiftUI
import AppKit

/// Drives neural text-to-speech (with zero-shot voice cloning) on the native
/// mlx-serve server (Qwen3-TTS). Mirrors `ImageGenService` / `VideoGenService`:
/// same `Phase` lifecycle, same JSON-event stream, writes a `.wav` under
/// `~/.mlx-serve/generations/audio`.
///
/// Reference clips are normalized to 24 kHz mono WAV in Swift (`AudioReference`)
/// and sent to the server as base64; the engine writes the output WAV itself.
@MainActor
final class AudioGenService: ObservableObject {

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

    /// Synthesize through the ONE main server: ensure running (headless if
    /// needed), load the TTS model on demand, stream `/v1/audio/speech`, then
    /// unload unless "Keep loaded" is set.
    func generate(_ request: AudioGenRequest, server: ServerManager) {
        guard !request.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            phase = .failed("Text is empty.")
            return
        }
        guard request.lanModelId != nil || ServerManager.resolveModelDir(repo: request.model.repo) != nil else {
            phase = .failed("Model \(request.model.repo) is not downloaded. Download it first.")
            return
        }

        task?.cancel()
        phase = .running(step: 0, total: 3, message: "Loading model…")
        log = []

        let outputPath = Self.makeOutputPath(text: request.text)
        let text = request.text
        let keep = request.keepResident
        let sidecar = Self.settingsText(request, modelName: request.model.name)
        // Reference voice for zero-shot cloning: the recorded/picked clip is
        // already normalized to 24 kHz mono WAV by AudioReference. Send it
        // base64 as `ref_audio`; the server runs it through the ECAPA-TDNN
        // speaker encoder and conditions the talker on it.
        let refB64: String? = request.refAudioPath.flatMap { path in
            (try? Data(contentsOf: URL(fileURLWithPath: path)))?.base64EncodedString()
        }

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
                // SSE: audio length is model-determined, so `progress` events carry
                // a growing frame count (total=0 → indeterminate bar); the
                // `complete` event carries the WAV as base64.
                var wav: Data? = nil
                var reqJson: [String: Any] = ["model": modelId, "input": text]
                if let refB64 { reqJson["ref_audio"] = refB64 }
                for try await ev in api.streamGeneration(
                    port: port, path: "/v1/audio/speech",
                    json: reqJson) {
                    switch ev["type"] as? String {
                    case "progress":
                        let step = ev["step"] as? Int ?? 0
                        let total = ev["total"] as? Int ?? 0
                        let stage = ev["stage"] as? String ?? "Generating audio"
                        // ~0.08s of audio per talker frame (1920 samples @ 24 kHz).
                        let secs = Double(step) * 1920.0 / 24000.0
                        let msg = total == 0 && step > 0
                            ? String(format: "%@ — ~%.1fs", stage, secs) : "\(stage)…"
                        phase = .running(step: step, total: total, message: msg)
                    case "complete":
                        if let b64 = ev["data"] as? String { wav = Data(base64Encoded: b64) }
                    case "error":
                        await releaseIfNeeded()
                        phase = .failed(ev["message"] as? String ?? "Synthesis failed.")
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
                // Settings sidecar: <clip>.txt with the text + voice params.
                try? sidecar.write(to: URL(fileURLWithPath: Self.sidecarPath(forWav: outputPath)),
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

    /// Awaitable synthesis for the agent's `generate_speech` tool. Same load →
    /// stream → write → unload pipeline as `generate`, returning the output WAV
    /// path (or throwing), but WITHOUT touching this service's UI state
    /// (`phase`/`task`/`recent`) — so a chat generation never hijacks the Audio
    /// window. `onProgress` drives the chat's own meter.
    func generateForAgent(_ request: AudioGenRequest, server: ServerManager,
                          onProgress: ((MediaGenProgress) -> Void)? = nil) async throws -> String {
        guard !request.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MediaGenError.emptyInput("Text")
        }
        guard request.lanModelId != nil || ServerManager.resolveModelDir(repo: request.model.repo) != nil else {
            throw MediaGenError.notDownloaded(request.model.name)
        }

        let outputPath = Self.makeOutputPath(text: request.text)
        let keep = request.keepResident
        let startedAt = Date()
        func report(_ step: Int, _ total: Int, _ message: String) {
            onProgress?(MediaGenProgress(kind: .speech, step: step, total: total,
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
            var reqJson: [String: Any] = ["model": modelId, "input": request.text,
                                          "speed": request.speed]
            if let ref = request.refAudioPath,
               let data = try? Data(contentsOf: URL(fileURLWithPath: ref)) {
                reqJson["ref_audio"] = data.base64EncodedString()
            }
            for try await ev in api.streamGeneration(
                port: port, path: "/v1/audio/speech", json: reqJson) {
                switch MediaSSE.classify(ev) {
                case .progress(let step, let total, let stage):
                    // Speech length is model-determined: total is 0 and the step
                    // is a talker frame (~0.08s of audio at 1920 samples/24 kHz),
                    // so the seconds produced is the only honest number here.
                    let secs = Double(step) * 1920.0 / 24000.0
                    let msg = total == 0 && step > 0
                        ? String(format: "%@ — ~%.1fs of audio", MediaSSE.stageLabel(stage), secs)
                        : MediaSSE.stageLabel(stage)
                    report(step, total, msg)
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
            try? Self.settingsText(request, modelName: modelId)
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

    private func appendLog(_ line: String) {
        log.append(line)
        if log.count > 400 { log.removeFirst(log.count - 400) }
    }

    private func insertRecent(_ path: String) {
        recent = MediaRecents.inserting(path, into: recent)
    }

    private func loadRecent() {
        recent = MediaRecents.scan(root: MediaStorage.audiosRoot, suffix: ".wav")
    }

    /// `<clip>.txt` settings sidecar written beside each generated clip.
    nonisolated static func settingsText(_ request: AudioGenRequest, modelName: String) -> String {
        var lines: [String] = [
            "model: \(modelName)",
            "speed: \(String(format: "%.2f", request.speed))",
            "temperature: \(String(format: "%.2f", request.temperature))",
        ]
        if let ref = request.refAudioPath, !ref.isEmpty {
            lines.append("reference_voice: \((ref as NSString).lastPathComponent)")
        }
        let refText = request.refText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !refText.isEmpty { lines.append("reference_transcript: \(refText)") }
        var out = lines.joined(separator: "\n")
        out += "\n\n# Text\n" + request.text.trimmingCharacters(in: .whitespacesAndNewlines)
        return out + "\n"
    }

    /// `<clip>.wav` → `<clip>.txt` companion path.
    nonisolated static func sidecarPath(forWav wavPath: String) -> String {
        (wavPath as NSString).deletingPathExtension + ".txt"
    }

    /// Slug + dated `.wav` path under `audiosRoot`, mirroring the image/video
    /// output layout. Exposed `internal static` so a unit test can pin the
    /// slugging + extension contract.
    static func makeOutputPath(text: String) -> String {
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd"
        let day = df.string(from: Date())
        let dayDir = (MediaStorage.audiosRoot as NSString).appendingPathComponent(day)
        try? FileManager.default.createDirectory(atPath: dayDir, withIntermediateDirectories: true)
        let tf = DateFormatter()
        tf.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let slug = text
            .lowercased()
            .replacingOccurrences(of: #"[^a-z0-9]+"#, with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
            .prefix(40)
        let filename = "\(tf.string(from: Date()))_\(slug).wav"
        return (dayDir as NSString).appendingPathComponent(filename)
    }
}
