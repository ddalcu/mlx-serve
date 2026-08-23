import Foundation
import SwiftUI
import AppKit

/// Drives image generation (FLUX.2 / Krea-2-Turbo) on the native mlx-serve server.
///
/// Keeps UI-facing state (progress, current output, history) so the view can
/// just bind to it. Cancellation is handled by dropping the running `Task`,
/// which tears the Python subprocess down via `AsyncThrowingStream`'s
/// `onTermination`.
@MainActor
final class ImageGenService: ObservableObject {

    enum Phase: Equatable {
        case idle
        case running(step: Int, total: Int, message: String)
        case completed(path: String)
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var recent: [String] = []  // recent output paths, newest first
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

    /// Generate through the ONE main server: ensure it's running (headless if
    /// needed), load the FLUX model on demand, stream `/v1/images/generations`,
    /// then unload unless the user pinned "Keep loaded". Coexists with a chat
    /// model on the same process.
    func generate(_ request: ImageGenRequest, server: ServerManager) {
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            phase = .failed("Prompt is empty.")
            return
        }
        // Network (LAN) models skip the local-download requirement — the
        // hosting Mac has the weights.
        guard request.lanModelId != nil || ServerManager.resolveModelDir(repo: request.model.repo) != nil else {
            phase = .failed("Model \(request.model.repo) is not downloaded. Download it first.")
            return
        }

        task?.cancel()
        phase = .running(step: 0, total: request.steps, message: "Loading model…")
        log = []

        let outputPath = Self.makeOutputPath(prompt: request.prompt)
        let steps = request.steps
        // Send a concrete seed so "random" (-1) actually varies; the server
        // otherwise defaults to a fixed seed.
        let seedToSend = request.seed >= 0 ? request.seed : Int.random(in: 0...0xFFFF_FFFF)
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
                // SSE: per-step `progress` events drive a determinate bar, then a
                // `complete` event carries the PNG.
                var png: Data? = nil
                let genJson = Self.requestJson(for: request, modelName: modelId, seed: seedToSend)
                for try await ev in api.streamGeneration(
                    port: port, path: "/v1/images/generations",
                    json: genJson) {
                    switch ev["type"] as? String {
                    case "progress":
                        let step = ev["step"] as? Int ?? 0
                        let total = ev["total"] as? Int ?? steps
                        let stage = ev["stage"] as? String ?? "Generating"
                        phase = .running(step: step, total: max(total, 1), message: "\(stage)…")
                    case "complete":
                        png = Self.decodePngB64(ev)
                    case "error":
                        await releaseIfNeeded()
                        phase = .failed(ev["message"] as? String ?? "Generation failed.")
                        return
                    default:
                        break
                    }
                }
                await releaseIfNeeded()
                guard let png else {
                    phase = .failed("Server returned no image data.")
                    return
                }
                try png.write(to: URL(fileURLWithPath: outputPath))
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

    /// Errors surfaced by the awaitable agent path (`generateForAgent`).
    enum GenError: LocalizedError {
        case emptyPrompt
        case notDownloaded(String)
        case server(String)
        var errorDescription: String? {
            switch self {
            case .emptyPrompt:           return "Prompt is empty."
            case .notDownloaded(let n):  return "\(n) is not downloaded."
            case .server(let m):         return m
            }
        }
    }

    /// Awaitable image generation for the agent's `generate_image` tool. Runs the
    /// SAME load → stream → write → unload pipeline as `generate`, returning the
    /// output PNG path (or throwing), but WITHOUT touching the menu-bar UI state
    /// (`phase`/`task`/`recent`) — so an agent generation never hijacks the Image
    /// window. Honors the request's `keepResident` like the interactive path.
    /// `onProgress` drives the chat's own meter.
    func generateForAgent(_ request: ImageGenRequest, server: ServerManager,
                          onProgress: ((MediaGenProgress) -> Void)? = nil) async throws -> String {
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw GenError.emptyPrompt
        }
        guard request.lanModelId != nil || ServerManager.resolveModelDir(repo: request.model.repo) != nil else {
            throw GenError.notDownloaded(request.model.name)
        }

        let outputPath = Self.makeOutputPath(prompt: request.prompt)
        let seedToSend = request.seed >= 0 ? request.seed : Int.random(in: 0...0xFFFF_FFFF)
        let keep = request.keepResident
        let steps = request.steps
        let startedAt = Date()
        func report(_ step: Int, _ total: Int, _ message: String) {
            onProgress?(MediaGenProgress(kind: .image, step: step, total: total,
                                         message: message, startedAt: startedAt))
        }
        report(0, 0, "Loading model")

        let (port, modelId, unloadId) = try await server.prepareGenModel(
            lanModelId: request.lanModelId, repo: request.model.repo)
        func releaseIfNeeded() async {
            if !keep, let id = unloadId { try? await server.unloadModel(id: id) }
        }
        do {
            var png: Data? = nil
            let genJson = Self.requestJson(for: request, modelName: modelId, seed: seedToSend)
            for try await ev in api.streamGeneration(
                port: port, path: "/v1/images/generations", json: genJson) {
                switch MediaSSE.classify(ev) {
                case .progress(let step, let total, let stage):
                    report(step, total == 0 ? steps : total, MediaSSE.stageLabel(stage))
                case .complete:
                    png = Self.decodePngB64(ev)
                case .failed(let m):
                    throw GenError.server(m)
                case .ignored:
                    break
                }
            }
            guard let png else { throw GenError.server("Server returned no image data.") }
            try png.write(to: URL(fileURLWithPath: outputPath))
            await releaseIfNeeded()
            return outputPath
        } catch {
            await releaseIfNeeded()
            throw error
        }
    }

    /// Build the `/v1/images/generations` body for a request. Pure + static so
    /// the contract is unit-testable: plain text-to-image bodies carry ONLY the
    /// classic fields; img2img (`image`+`strength`), edit references
    /// (`mode`+`ref_images`), conditioning rebalance
    /// (`cond_gain`+`cond_weights`), and LoRA (`lora_paths`+`lora_scales`) are
    /// added only when set, so the server sees no behavior change otherwise.
    static func requestJson(for request: ImageGenRequest, modelName: String, seed: Int) -> [String: Any] {
        var json: [String: Any] = [
            "model": modelName,
            "prompt": request.prompt,
            "steps": request.steps,
            "seed": seed,
        ]
        // width/height 0 = "Match source": OMIT `size` so the server keeps the
        // reference image's own resolution instead of re-gridding it to a bucket.
        if request.width > 0 && request.height > 0 {
            json["size"] = "\(request.width)x\(request.height)"
        }
        if let src = request.initImagePath,
           let data = FileManager.default.contents(atPath: src) {
            json["image"] = data.base64EncodedString()
            if request.editMode {
                json["mode"] = "edit" // clean in-context reference; strength n/a
                // Extra references (multi-reference edit) — missing files are
                // skipped like a missing primary source.
                let refs = request.refImagePaths.compactMap {
                    FileManager.default.contents(atPath: $0)?.base64EncodedString()
                }
                if !refs.isEmpty { json["ref_images"] = refs }
            } else {
                json["strength"] = request.strength
            }
        }
        if request.condGain != 1.0 { json["cond_gain"] = request.condGain }
        if !request.condWeightsText.trimmingCharacters(in: .whitespaces).isEmpty,
           let weights = ImageGenRequest.parseCondWeights(request.condWeightsText),
           weights.count == request.condWeightCount {
            json["cond_weights"] = weights
        }
        // Stacked style LoRAs: several `.safetensors` adapters attach at once
        // and their effects sum (mirrors mflux's `lora_paths`/`lora_scales`).
        // Half-filled rows (no path picked yet) are dropped here rather than
        // sent as an empty string the server would reject.
        let loras = request.loras.filter { !$0.path.isEmpty }
        if !loras.isEmpty {
            json["lora_paths"] = loras.map(\.path)
            json["lora_scales"] = loras.map(\.scale)
        }
        return json
    }

    /// Extract the base64 PNG from an OpenAI `{data:[{b64_json}]}` response body.
    /// Pure + static so it's unit-testable without a running server.
    static func decodePngB64(_ body: Data) -> Data? {
        guard let obj = try? JSONSerialization.jsonObject(with: body) as? [String: Any] else { return nil }
        return decodePngB64(obj)
    }

    /// Same, from an already-parsed object (the SSE `complete` event).
    static func decodePngB64(_ obj: [String: Any]) -> Data? {
        guard let arr = obj["data"] as? [[String: Any]],
              let b64 = arr.first?["b64_json"] as? String,
              let png = Data(base64Encoded: b64)
        else { return nil }
        return png
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
        if recent.count > 60 { recent.removeLast(recent.count - 60) }
    }

    /// Scan the generations/images/ tree for existing files so the history
    /// shelf shows something on first launch.
    private func loadRecent() {
        let root = MediaStorage.imagesRoot
        let fm = FileManager.default
        guard let days = try? fm.contentsOfDirectory(atPath: root) else { return }
        var paths: [(String, Date)] = []
        for day in days.sorted(by: >) {
            let dayDir = (root as NSString).appendingPathComponent(day)
            guard let files = try? fm.contentsOfDirectory(atPath: dayDir) else { continue }
            for f in files where f.hasSuffix(".png") || f.hasSuffix(".jpg") {
                let full = (dayDir as NSString).appendingPathComponent(f)
                let date = (try? fm.attributesOfItem(atPath: full)[.modificationDate] as? Date) ?? .distantPast
                paths.append((full, date))
            }
        }
        recent = paths.sorted { $0.1 > $1.1 }.prefix(60).map(\.0)
    }

    // MARK: - Paths / args

    private static func makeOutputPath(prompt: String) -> String {
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd"
        let day = df.string(from: Date())
        let dayDir = (MediaStorage.imagesRoot as NSString).appendingPathComponent(day)
        try? FileManager.default.createDirectory(atPath: dayDir, withIntermediateDirectories: true)
        let tf = DateFormatter()
        tf.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let slug = prompt
            .lowercased()
            .replacingOccurrences(of: #"[^a-z0-9]+"#, with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
            .prefix(40)
        let filename = "\(tf.string(from: Date()))_\(slug).png"
        return (dayDir as NSString).appendingPathComponent(filename)
    }
}
