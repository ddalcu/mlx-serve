import Foundation
import SwiftUI
import AppKit

/// Runs neural text-to-speech (with zero-shot voice cloning) via the shared
/// `PythonManager` venv and the embedded `mlx-audio` library. Mirrors
/// `ImageGenService` / `VideoGenService`: same `Phase` lifecycle, same
/// JSON-event stream, writes a `.wav` under `~/.mlx-serve/generations/audio`.
///
/// Unlike video, audio needs no ffmpeg — reference clips are normalized to
/// 24 kHz mono WAV in Swift (`AudioReference`) before they reach Python, and
/// mlx-audio writes the output wav itself.
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
        guard let modelDir = ServerManager.resolveModelDir(repo: request.model.repo) else {
            phase = .failed("Model \(request.model.repo) is not downloaded. Download it first.")
            return
        }

        task?.cancel()
        phase = .running(step: 0, total: 3, message: "Loading model…")
        log = []

        let outputPath = Self.makeOutputPath(text: request.text)
        let text = request.text
        let keep = request.keepResident
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
                let port = try await server.ensureRunning(forGenModelDir: modelDir)
                if Task.isCancelled { phase = .idle; return }
                let info = try await server.loadModel(id: modelDir)
                loadedId = info.name
                if Task.isCancelled { await releaseIfNeeded(); phase = .idle; return }
                // SSE: audio length is model-determined, so `progress` events carry
                // a growing frame count (total=0 → indeterminate bar); the
                // `complete` event carries the WAV as base64.
                var wav: Data? = nil
                var reqJson: [String: Any] = ["model": info.name, "input": text]
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
        recent.removeAll { $0 == path }
        recent.insert(path, at: 0)
        if recent.count > 40 { recent.removeLast(recent.count - 40) }
    }

    private func loadRecent() {
        let root = MediaStorage.audiosRoot
        let fm = FileManager.default
        guard let days = try? fm.contentsOfDirectory(atPath: root) else { return }
        var paths: [(String, Date)] = []
        for day in days.sorted(by: >) {
            let dayDir = (root as NSString).appendingPathComponent(day)
            guard let files = try? fm.contentsOfDirectory(atPath: dayDir) else { continue }
            for f in files where f.hasSuffix(".wav") {
                let full = (dayDir as NSString).appendingPathComponent(f)
                let date = (try? fm.attributesOfItem(atPath: full)[.modificationDate] as? Date) ?? .distantPast
                paths.append((full, date))
            }
        }
        recent = paths.sorted { $0.1 > $1.1 }.prefix(40).map(\.0)
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
