import Foundation
import SwiftUI

/// Drives text-to-audio (Stable Audio 3) on the native mlx-serve server:
/// load on demand, stream `/v1/audio/sound-generations`, write the WAV and its
/// settings sidecar under `~/.mlx-serve/generations/sound`, unload unless
/// "Keep loaded" is set. The pane and the chat's `generate_sound` tool share
/// ONE pipeline (`render`); only the pane's `phase`/`recent` differ.
@MainActor
final class SoundGenService: ObservableObject {

    enum Phase: Equatable {
        case idle
        case running(step: Int, total: Int, message: String)
        case completed(path: String)
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var recent: [String] = []

    private var task: Task<Void, Never>?
    private let api = APIClient()

    init() {
        recent = MediaRecents.scan(root: MediaStorage.soundRoot, suffix: ".wav")
    }

    var isRunning: Bool {
        if case .running = phase { return true }
        return false
    }

    /// The `/v1/audio/sound-generations` body. Sticky settings outlive a model
    /// switch, so duration and steps clamp into THIS model's range rather than
    /// earn a 400; a random seed (-1) is resolved here so the sidecar names it.
    nonisolated static func requestBody(_ request: SoundGenRequest, modelName: String) -> [String: Any] {
        let r = request.model.durationRange
        var body: [String: Any] = [
            "model": modelName,
            "prompt": request.prompt,
            "duration_seconds": min(max(request.durationSeconds, r.lowerBound), r.upperBound),
            "seed": request.seed >= 0 ? request.seed : Int.random(in: 0..<1_000_000_000),
            "stream": true,
        ]
        if let steps = request.steps {
            body["steps"] = min(max(steps, request.model.stepsRange.lowerBound), request.model.stepsRange.upperBound)
        }
        return body
    }

    /// The `<sound>.txt` sidecar: the body the server was sent, so a sound is
    /// reproducible. `AudioSidecar.prompt` reads the prompt section back.
    nonisolated static func settingsText(body: [String: Any]) -> String {
        var lines = ["model", "duration_seconds", "seed"].map { "\($0): \(body[$0] ?? "")" }
        if let steps = body["steps"] { lines.append("steps: \(steps)") }
        let prompt = (body["prompt"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        return lines.joined(separator: "\n") + "\n\n# Style prompt\n" + prompt + "\n"
    }

    /// The pane's Generate.
    func generate(_ request: SoundGenRequest, server: ServerManager) {
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            phase = .failed("Describe the sound first.")
            return
        }
        guard request.lanModelId != nil || ServerManager.resolveModelDir(repo: request.model.repo) != nil else {
            phase = .failed("\(request.model.name) is not downloaded yet.")
            return
        }
        task?.cancel()
        phase = .running(step: 0, total: 0, message: L10n.text("Loading model…"))
        task = Task {
            do {
                let path = try await render(request, server: server) { [weak self] p in
                    self?.phase = .running(step: p.step, total: p.total, message: L10n.format("%@…", L10n.text(p.message)))
                }
                phase = .completed(path: path)
                recent = MediaRecents.inserting(path, into: recent)
            } catch is CancellationError {
                phase = .idle
            } catch {
                phase = .failed(error.localizedDescription)
            }
        }
    }

    /// The chat's `generate_sound` tool: same pipeline, the pane untouched.
    func generateForAgent(_ request: SoundGenRequest, server: ServerManager,
                          onProgress: ((MediaGenProgress) -> Void)? = nil) async throws -> String {
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MediaGenError.emptyInput("Prompt")
        }
        guard request.lanModelId != nil || ServerManager.resolveModelDir(repo: request.model.repo) != nil else {
            throw MediaGenError.notDownloaded(request.model.name)
        }
        let startedAt = Date()
        return try await render(request, server: server) { p in
            onProgress?(MediaGenProgress(kind: .sound, step: p.step, total: p.total,
                                         message: p.message, startedAt: startedAt))
        }
    }

    func cancel() {
        task?.cancel()
        task = nil
    }

    private struct Step { let step: Int; let total: Int; let message: String }

    /// Load → stream → write → unload. Returns the WAV path.
    private func render(_ request: SoundGenRequest, server: ServerManager,
                        report: (Step) -> Void) async throws -> String {
        report(Step(step: 0, total: 0, message: "Loading model"))
        let (port, modelId, unloadId) = try await server.prepareGenModel(
            lanModelId: request.lanModelId, repo: request.model.repo)
        let keep = request.keepResident
        func release() async {
            if !keep, let id = unloadId { try? await server.unloadModel(id: id) }
        }
        do {
            let body = Self.requestBody(request, modelName: modelId)
            var wav: Data? = nil
            for try await ev in api.streamGeneration(port: port, path: "/v1/audio/sound-generations", json: body) {
                switch MediaSSE.classify(ev) {
                case .progress(let step, let total, let stage):
                    report(Step(step: step, total: total, message: MediaSSE.stageLabel(stage)))
                case .complete:
                    if let b64 = ev["data"] as? String { wav = Data(base64Encoded: b64) }
                case .failed(let m):
                    throw MediaGenError.server(m)
                case .ignored:
                    break
                }
            }
            guard let wav, wav.count > 44 else { throw MediaGenError.server("Server returned an empty audio response.") }
            let path = MediaStorage.datedPath(root: MediaStorage.soundRoot, prompt: request.prompt, ext: "wav")
            try wav.write(to: URL(fileURLWithPath: path))
            try? Self.settingsText(body: body)
                .write(toFile: (path as NSString).deletingPathExtension + ".txt", atomically: true, encoding: .utf8)
            await release()
            return path
        } catch {
            await release()
            throw error
        }
    }
}
