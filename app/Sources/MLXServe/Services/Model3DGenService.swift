import Foundation
import SwiftUI
import AppKit

/// Drives single-image-to-3D generation (Hunyuan3D 2.1 shape stage) on the
/// native mlx-serve server. Mirrors `ImageGenService` / `AudioGenService`: same
/// `Phase` lifecycle, same JSON-event stream, writes a `.glb` under
/// `~/.mlx-serve/generations/models3d`.
///
/// The source photo's subject is cut out and composited on white (`SubjectCutout`)
/// off the main actor before base64 encoding — matching the reference pipeline's
/// rembg step. The engine returns the GLB bytes in the `complete` event.
@MainActor
final class Model3DGenService: ObservableObject {

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
    /// needed), load the 3D model on demand, stream `/v1/3d/generations`, save
    /// the returned GLB, then unload unless the user pinned "Keep loaded".
    func generate(_ request: Model3DGenRequest, server: ServerManager) {
        guard !request.photoPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            phase = .failed("Choose a photo first.")
            return
        }
        guard let modelDir = ServerManager.resolveModelDir(repo: request.model.repo) else {
            phase = .failed("Model \(request.model.repo) isn't available locally. Convert it first.")
            return
        }

        task?.cancel()
        phase = .running(step: 0, total: request.steps, message: "Loading model…")
        log = []

        let outputPath = Self.makeOutputPath(photoPath: request.photoPath)
        let photoPath = request.photoPath
        let steps = request.steps
        // Send a concrete seed so "random" (-1) actually varies per run.
        let seedToSend = request.seed >= 0 ? request.seed : Int.random(in: 0...0xFFFF_FFFF)
        let keep = request.keepResident

        task = Task {
            var loadedId: String? = nil
            func releaseIfNeeded() async {
                if !keep, let id = loadedId { try? await server.unloadModel(id: id) }
            }
            do {
                // Subject cutout + base64 OFF the main actor: reading a multi-MB
                // photo and running Vision synchronously would block the UI
                // (VideoGenService's first-frame pattern). Falls back to the raw
                // bytes if Vision finds no subject — the server composites
                // alpha-on-white itself.
                let imageB64: String? = await Task.detached(priority: .userInitiated) {
                    guard let raw = try? Data(contentsOf: URL(fileURLWithPath: photoPath)) else { return nil }
                    return SubjectCutout.cutoutOnWhite(imageData: raw).base64EncodedString()
                }.value
                guard let imageB64 else {
                    phase = .failed("Couldn't read the photo. Pick a PNG or JPEG.")
                    return
                }
                let port = try await server.ensureRunning(forGenModelDir: modelDir)
                if Task.isCancelled { phase = .idle; return }
                let info = try await server.loadModel(id: modelDir)  // registry id = dir basename
                loadedId = info.name
                if Task.isCancelled { await releaseIfNeeded(); phase = .idle; return }
                // SSE: per-step `progress` events drive a determinate bar, then a
                // `complete` event carries the GLB as base64.
                var glb: Data? = nil
                let genJson = Self.requestJson(for: request, modelName: info.name, imageB64: imageB64, seed: seedToSend)
                for try await ev in api.streamGeneration(
                    port: port, path: "/v1/3d/generations",
                    json: genJson) {
                    switch ev["type"] as? String {
                    case "progress":
                        let step = ev["step"] as? Int ?? 0
                        let total = ev["total"] as? Int ?? steps
                        let stage = ev["stage"] as? String ?? "Generating"
                        phase = .running(step: step, total: max(total, 1), message: "\(stage)…")
                    case "complete":
                        glb = Self.decodeGlb(ev)
                    case "error":
                        await releaseIfNeeded()
                        phase = .failed(ev["message"] as? String ?? "Generation failed.")
                        return
                    default:
                        break
                    }
                }
                await releaseIfNeeded()
                guard let glb, glb.count > 12 else {
                    phase = .failed("Server returned no 3D model data.")
                    return
                }
                try glb.write(to: URL(fileURLWithPath: outputPath))
                phase = .completed(path: outputPath)
                insertRecent(outputPath)
                // History-shelf thumbnail, rendered off the main actor (SceneKit
                // offscreen). Best-effort — a failed render just leaves the
                // shelf's cube placeholder.
                Task.detached(priority: .utility) {
                    Model3DThumbnailer.ensure(glbPath: outputPath)
                }
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

    // MARK: - Request / response (pure so tests pin the wire contract)

    /// Build the `/v1/3d/generations` body. Pure + static so the contract is
    /// unit-testable; `imageB64` is the already-encoded (cutout-on-white) photo
    /// so the file I/O + Vision stay off the main actor. `stream` is injected by
    /// `APIClient.streamGeneration`, not here (mirrors image/video/audio).
    static func requestJson(for request: Model3DGenRequest, modelName: String, imageB64: String, seed: Int) -> [String: Any] {
        [
            "model": modelName,
            "image": imageB64,
            "steps": request.steps,
            "guidance_scale": request.guidanceScale,
            "octree_resolution": request.octreeResolution,
            "seed": seed,
            "texture": request.texture,
        ]
    }

    /// Extract the base64 GLB from a `{format:"glb", data:…}` response body.
    /// Pure + static so it's unit-testable without a running server.
    static func decodeGlb(_ body: Data) -> Data? {
        guard let obj = try? JSONSerialization.jsonObject(with: body) as? [String: Any] else { return nil }
        return decodeGlb(obj)
    }

    /// Same, from an already-parsed object (the SSE `complete` event).
    static func decodeGlb(_ obj: [String: Any]) -> Data? {
        guard obj["format"] as? String == "glb",
              let b64 = obj["data"] as? String,
              let glb = Data(base64Encoded: b64)
        else { return nil }
        return glb
    }

    // MARK: - History shelf

    /// Thumbnail sibling for a generated GLB: `<file>.glb.thumb.png`. The
    /// non-.glb suffix keeps it invisible to the history scan.
    nonisolated static func thumbnailPath(for glbPath: String) -> String {
        glbPath + ".thumb.png"
    }

    /// Show a previously generated model in the preview pane. A file that has
    /// vanished (user cleaned the folder) is pruned from the shelf instead.
    func showHistoryItem(_ path: String) {
        guard !isRunning else { return }
        if FileManager.default.fileExists(atPath: path) {
            phase = .completed(path: path)
        } else {
            recent.removeAll { $0 == path }
        }
    }

    // MARK: - Private

    private func insertRecent(_ path: String) {
        recent.removeAll { $0 == path }
        recent.insert(path, at: 0)
        if recent.count > 40 { recent.removeLast(recent.count - 40) }
    }

    /// Scan the generations/models3d/ tree for existing files so the history
    /// shelf shows something on first launch.
    private func loadRecent() {
        let root = MediaStorage.models3dRoot
        let fm = FileManager.default
        guard let days = try? fm.contentsOfDirectory(atPath: root) else { return }
        var paths: [(String, Date)] = []
        for day in days.sorted(by: >) {
            let dayDir = (root as NSString).appendingPathComponent(day)
            guard let files = try? fm.contentsOfDirectory(atPath: dayDir) else { continue }
            for f in files where f.hasSuffix(".glb") {
                let full = (dayDir as NSString).appendingPathComponent(f)
                let date = (try? fm.attributesOfItem(atPath: full)[.modificationDate] as? Date) ?? .distantPast
                paths.append((full, date))
            }
        }
        recent = paths.sorted { $0.1 > $1.1 }.prefix(40).map(\.0)
    }

    /// Slug + dated `.glb` path under `models3dRoot`, mirroring the image/video/
    /// audio output layout. The slug comes from the source photo's name.
    /// `internal static` so a unit test can pin the shape.
    static func makeOutputPath(photoPath: String) -> String {
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd"
        let day = df.string(from: Date())
        let dayDir = (MediaStorage.models3dRoot as NSString).appendingPathComponent(day)
        try? FileManager.default.createDirectory(atPath: dayDir, withIntermediateDirectories: true)
        let tf = DateFormatter()
        tf.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let stem = ((photoPath as NSString).lastPathComponent as NSString).deletingPathExtension
            .lowercased()
            .replacingOccurrences(of: #"[^a-z0-9]+"#, with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
            .prefix(40)
        let slug = stem.isEmpty ? "model" : String(stem)
        let filename = "\(tf.string(from: Date()))_\(slug).glb"
        return (dayDir as NSString).appendingPathComponent(filename)
    }
}
