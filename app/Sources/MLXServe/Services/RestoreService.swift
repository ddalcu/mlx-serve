import Foundation
import SwiftUI
import AppKit

/// SeedVR2's pixel-geometry rule (`restoreGeometryRefusal` in `src/gen.zig`):
/// the VAE downsamples /8 and the patchifier another /2, so both dimensions
/// must be multiples of 16 or the server 400s BY NAME rather than silently
/// cropping. Real photos are essentially never already on that grid (905×960,
/// a common phone/scan size, misses by 9px), so the app has to do the
/// snapping — same division of labor as `ResolutionGrid` for generation.
enum RestoreGeometry {
    /// Snap DOWN to the nearest multiple of 16, floored at 16. Only ever
    /// rounds down: rounding up would need to invent border pixels that
    /// aren't the photo's own content, which is a worse distortion than
    /// losing a few edge pixels to a crop.
    static func snap(_ v: Int) -> Int {
        max(16, (v / 16) * 16)
    }

    struct CropRect: Equatable { let x: Int; let y: Int; let width: Int; let height: Int }

    /// The centered crop that puts `width`x`height` on the /16 grid, or nil
    /// when it's already there. Centered so a portrait crop takes evenly off
    /// both sides rather than favoring one edge.
    static func centeredCrop(width: Int, height: Int) -> CropRect? {
        // `snap` floors at 16, so a source below that would snap UP and yield a
        // rect larger than the picture at a negative origin. There is no crop
        // that puts it on the grid — the server's geometry 400 is the answer.
        guard width >= 16, height >= 16 else { return nil }
        let w = snap(width), h = snap(height)
        guard w != width || h != height else { return nil }
        return CropRect(x: (width - w) / 2, y: (height - h) / 2, width: w, height: h)
    }

    /// What one OUTPUT pixel costs in transient GPU memory, on top of the
    /// resident checkpoint. Mirrors `RESTORE_TRANSIENT_BYTES_PER_PIXEL` in
    /// `src/gen.zig`, where it was measured — the server owns the number and
    /// enforces it against FREE memory; this copy exists only so the pane can
    /// catch what is impossible on this Mac at ALL, before spending minutes
    /// loading a model. The two gates deliberately compare different things:
    /// the server knows what is free right now, and only total RAM is stable
    /// enough to warn about before anything has been loaded.
    static let transientBytesPerPixel = 4100

    /// A sentence to show before starting, when the target canvas cannot fit
    /// on this Mac at all, or nil.
    ///
    /// WHY A GATE ON THE TARGET AND NOT THE SOURCE: the failure it replaces
    /// was a small photo at 2x. MLX at the Metal working-set edge returns
    /// degenerate values rather than erroring, so the restore came back as a
    /// full-size PNG of a single flat colour with a 200 behind it — a pure
    /// white 1808x1920 in the user's own output folder. The source was never
    /// the expensive part; the canvas is.
    ///
    /// `totalRAMGB == 0` means the machine could not be measured, and a gate
    /// that cannot see must not warn about everything.
    static func memoryWarning(targetWidth: Int, targetHeight: Int,
                              modelGB: Int, totalRAMGB: Int) -> String? {
        guard totalRAMGB > 0, targetWidth > 0, targetHeight > 0 else { return nil }
        let bytes = Double(targetWidth) * Double(targetHeight) * Double(transientBytesPerPixel)
        let transientGB = bytes / 1_073_741_824
        let needed = Double(modelGB) + transientGB
        guard needed > Double(totalRAMGB) else { return nil }
        return String(
            format: "A %d × %d result needs about %.0f GB of memory (%.0f GB of working space plus the %d GB model), "
                  + "but your Mac has %d GB. Lower the scale or pick a smaller photo — otherwise this can come back blank.",
            targetWidth, targetHeight, needed.rounded(), transientGB.rounded(), modelGB, totalRAMGB)
    }

    /// The canvas an upscale targets: `width`x`height` scaled by `factor`,
    /// rounded to the NEAREST multiple of 16 (never just down) — this canvas
    /// is synthesized by the resize step below, not sampled from the source,
    /// so there's no "losing the photo's own pixels" concern the crop path
    /// has, and rounding to nearest keeps the result closest to the factor
    /// the user actually picked.
    static func upscaledTarget(width: Int, height: Int, factor: Double) -> (width: Int, height: Int) {
        func nearest16(_ v: Double) -> Int { max(16, Int((v / 16).rounded()) * 16) }
        return (nearest16(Double(width) * factor), nearest16(Double(height) * factor))
    }

    /// "2×" for a whole factor, "1.5×" for anything with a fractional part —
    /// the slider's step (0.1) means most picks are whole, and a trailing
    /// ".0" on every label would read as false precision.
    static func formatFactor(_ factor: Double) -> String {
        factor.rounded() == factor
            ? String(format: "%.0f×", factor)
            : String(format: "%.1f×", factor)
    }
}

/// The "before" half of the enlarge preview's before/after slider.
///
/// An upscale records nothing about where it came from, and the source photo
/// can move, be deleted, or be an earlier result the user has since trashed —
/// so the comparison keeps its OWN copy: the exact bytes SeedVR2 was handed.
/// That picture is already the restore's canvas (cropped onto the /16 grid, or
/// bicubic-resized for a scale above 1x), so it lines up with the result pixel
/// for pixel and the slider never stretches one side to meet the other. It is
/// kept exactly as sent, lossless: SeedVR2 removes compression artefacts, and a
/// JPEG "before" would credit it with repairing damage the comparison did.
enum RestoreComparison {

    /// A dot-folder inside the result's own day folder: Finder hides it, and
    /// the `recent` scan reads only the day folder's files, so a stored input
    /// is never listed as a result of its own (`RestoreService.recentPaths`).
    static let folderName = ".before"

    /// Keyed by the result's filename, so a result always finds its input
    /// and two results can never share one.
    static func inputPath(forResult resultPath: String) -> String {
        let dir = (resultPath as NSString).deletingLastPathComponent
        let name = (resultPath as NSString).lastPathComponent
        return ((dir as NSString).appendingPathComponent(folderName) as NSString)
            .appendingPathComponent(name)
    }

    /// Best-effort: a comparison is a nicety, and failing to store one must
    /// never fail the restore that already worked.
    @discardableResult
    static func saveInput(_ data: Data, forResult resultPath: String) -> Bool {
        let path = inputPath(forResult: resultPath)
        do {
            try FileManager.default.createDirectory(
                atPath: (path as NSString).deletingLastPathComponent,
                withIntermediateDirectories: true)
            try data.write(to: URL(fileURLWithPath: path))
            return true
        } catch {
            return false
        }
    }

    /// nil for a result made before comparisons existed; the preview then
    /// shows the plain picture, as it always did.
    static func existingInput(forResult resultPath: String) -> String? {
        let path = inputPath(forResult: resultPath)
        return FileManager.default.fileExists(atPath: path) ? path : nil
    }

    /// Deleted outright rather than trashed: it is a copy the app derived, not
    /// the user's picture, and a second file of the same name in the Trash
    /// reads as a duplicate of the one they just threw away.
    static func removeInput(forResult resultPath: String) {
        try? FileManager.default.removeItem(atPath: inputPath(forResult: resultPath))
    }

    /// The divider's position, as a fraction of the picture's width, for a
    /// pointer at `x`. Clamped, so a drag past either edge parks the divider on
    /// that edge instead of losing it; a frame not laid out yet sits centred.
    static func fraction(x: CGFloat, width: CGFloat) -> CGFloat {
        guard width > 0 else { return 0.5 }
        return min(1, max(0, x / width))
    }
}

/// Drives image restoration/upscaling (SeedVR2) on the native mlx-serve
/// server via `POST /v1/images/upscales`.
///
/// Unlike `ImageGenService` this endpoint is a single request/response, not
/// an SSE stream — a restoration is "minutes of GPU on a big image" with no
/// intermediate step events, so the UI shows an indeterminate spinner rather
/// than a progress bar.
@MainActor
final class RestoreService: ObservableObject {

    enum Phase: Equatable {
        case idle
        case running(message: String)
        case completed(path: String)
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var recent: [String] = []  // recent output paths, newest first

    private var task: Task<Void, Never>?
    private let session: URLSession = {
        let cfg = URLSessionConfiguration.default
        // A restoration can run for many minutes on a large image (no
        // intermediate byte traffic to reset a shorter timeout against, the
        // way `streamGeneration`'s SSE keepalives do).
        cfg.timeoutIntervalForRequest = 3600
        return URLSession(configuration: cfg)
    }()

    init() {
        loadRecent()
    }

    var isRunning: Bool {
        if case .running = phase { return true }
        return false
    }

    /// Restore through the ONE main server: ensure it's running, load the
    /// restore model on demand, POST the image, then unload unless the user
    /// pinned "Keep loaded". Coexists with a chat model on the same process.
    func restore(sourcePath: String, model: RestoreModelPreset, lanModelId: String?,
                 scale: Double = 1, seed: Int, keepResident: Bool, server: ServerManager) {
        guard FileManager.default.fileExists(atPath: sourcePath) else {
            phase = .failed("Could not read \(sourcePath).")
            return
        }
        guard lanModelId != nil || ServerManager.resolveModelDir(repo: model.repo) != nil else {
            phase = .failed("Model \(model.repo) is not downloaded. Download it first.")
            return
        }

        task?.cancel()
        phase = .running(message: "Loading model…")

        let outputPath = Self.makeOutputPath(sourcePath: sourcePath)
        let seedToSend = seed >= 0 ? seed : Int.random(in: 0...0xFFFF_FFFF)

        task = Task {
            var loadedId: String? = nil
            func releaseIfNeeded() async {
                if !keepResident, let id = loadedId { try? await server.unloadModel(id: id) }
            }
            do {
                // Off the main actor: NSImage decode + crop of a big photo is
                // real work, and the caller shouldn't stall on it.
                let prepared = try await Task.detached { try Self.preparedImageData(sourcePath: sourcePath, scale: scale) }.value
                if let note = prepared.note {
                    phase = .running(message: "Loading model… (\(note))")
                }
                let (port, modelId, unloadId) = try await server.prepareGenModel(
                    lanModelId: lanModelId, repo: model.repo)
                loadedId = unloadId
                if Task.isCancelled { await releaseIfNeeded(); phase = .idle; return }
                phase = .running(message: prepared.note.map { "Restoring… (\($0))" } ?? "Restoring…")
                let body: [String: Any] = [
                    "model": modelId,
                    "image": prepared.data.base64EncodedString(),
                    "seed": seedToSend,
                ]
                let png = try await Self.postForPng(session: session, port: port,
                                                    path: "/v1/images/upscales", json: body)
                await releaseIfNeeded()
                try png.write(to: URL(fileURLWithPath: outputPath))
                // After the result, never before: a write that throws above
                // must not leave a "before" with nothing to compare it to.
                RestoreComparison.saveInput(prepared.data, forResult: outputPath)
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

    /// The bytes actually sent to the server, plus a human note describing
    /// what was done to the source before sending — nil when it went
    /// through untouched (1x, already on-grid), so the common restore-only
    /// case says nothing.
    ///
    /// `scale` <= 1 is the restore-only path: crop DOWN to the /16 grid,
    /// losing at most 15px per edge, never inventing pixels. `scale` > 1
    /// resizes UP to `width*scale` x `height*scale` (nearest /16) via a
    /// bicubic draw — SeedVR2 has no upscaling of its own (confirmed against
    /// `docs/seedvr2-arch.md` and the community ComfyUI node's `resolution`
    /// param: both resize to a target canvas FIRST, then let the DiT fill in
    /// detail at that size), so a scale factor is this client-side resize,
    /// not a server capability.
    nonisolated static func preparedImageData(sourcePath: String, scale: Double) throws -> (data: Data, note: String?) {
        guard let src = NSImage(contentsOfFile: sourcePath),
              let cgImage = src.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw APIError.badStatus(code: 0, detail: "Could not decode \((sourcePath as NSString).lastPathComponent) as an image.")
        }
        let w = cgImage.width, h = cgImage.height

        if scale > 1 {
            let target = RestoreGeometry.upscaledTarget(width: w, height: h, factor: scale)
            guard let resized = resize(cgImage, to: target), let data = pngData(from: resized) else {
                throw APIError.badStatus(code: 0, detail: "Could not upscale the source image.")
            }
            let note = "upscaled \(w)×\(h) → \(target.width)×\(target.height) (\(RestoreGeometry.formatFactor(scale))) before restoring detail"
            return (data, note)
        }

        guard let crop = RestoreGeometry.centeredCrop(width: w, height: h) else {
            guard let data = pngData(from: cgImage) else {
                throw APIError.badStatus(code: 0, detail: "Could not encode the source image.")
            }
            return (data, nil)
        }
        let rect = CGRect(x: crop.x, y: crop.y, width: crop.width, height: crop.height)
        guard let cropped = cgImage.cropping(to: rect), let data = pngData(from: cropped) else {
            throw APIError.badStatus(code: 0, detail: "Could not crop the source image.")
        }
        let note = "cropped \(w)×\(h) → \(crop.width)×\(crop.height): SeedVR2 needs both dimensions divisible by 16"
        return (data, note)
    }

    /// Bicubic resize via `NSGraphicsContext` at high interpolation quality —
    /// the same "resize before restoring" step the community SeedVR2 node
    /// does client-side, not something the DiT does for free.
    nonisolated private static func resize(_ cgImage: CGImage, to target: (width: Int, height: Int)) -> CGImage? {
        guard let ctx = CGContext(
            data: nil, width: target.width, height: target.height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: target.width, height: target.height))
        return ctx.makeImage()
    }

    nonisolated private static func pngData(from cgImage: CGImage) -> Data? {
        NSBitmapImageRep(cgImage: cgImage).representation(using: .png, properties: [:])
    }

    func cancel() {
        task?.cancel()
        task = nil
    }

    // MARK: - Transport

    /// Plain (non-SSE) POST — `/v1/images/upscales` answers ONE JSON body,
    /// not a stream, so this bypasses `APIClient.streamGeneration` (which
    /// always sends `stream:true` and parses `data:` lines) and reads the
    /// whole response instead.
    static func postForPng(session: URLSession, port: UInt16, path: String, json: [String: Any]) async throws -> Data {
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: json, options: [.withoutEscapingSlashes])
        let (data, resp) = try await session.data(for: req)
        guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
            let code = (resp as? HTTPURLResponse)?.statusCode ?? -1
            throw APIError.badStatus(code: code, detail: APIError.errorDetail(fromBody: data))
        }
        guard let png = ImageGenService.decodePngB64(data) else {
            throw APIError.badStatus(code: 200, detail: "Server returned no image data.")
        }
        return png
    }

    // MARK: - Private

    /// Drop a path the user has deleted. The list AND a `.completed` phase can
    /// each hold it, and a phase left pointing at a trashed file redraws it in
    /// the preview from `NSImage`'s own cache — so both let go together.
    func forget(path: String) {
        recent.removeAll { $0 == path }
        if case .completed(let p) = phase, p == path { phase = .idle }
    }

    private func insertRecent(_ path: String) {
        recent.removeAll { $0 == path }
        recent.insert(path, at: 0)
        if recent.count > 60 { recent.removeLast(recent.count - 60) }
    }

    private func loadRecent() {
        recent = Self.recentPaths(root: MediaStorage.upscalesRoot)
    }

    /// Every result under `root`, newest first. Reads each day folder's own
    /// files and never descends further — which is what keeps the stored
    /// comparison inputs in `.before/`, filenames identical to their results,
    /// out of the strip (`RestoreComparison`).
    nonisolated static func recentPaths(root: String, limit: Int = 60) -> [String] {
        let fm = FileManager.default
        guard let days = try? fm.contentsOfDirectory(atPath: root) else { return [] }
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
        return paths.sorted { $0.1 > $1.1 }.prefix(limit).map(\.0)
    }

    private static func makeOutputPath(sourcePath: String) -> String {
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd"
        let day = df.string(from: Date())
        let dayDir = (MediaStorage.upscalesRoot as NSString).appendingPathComponent(day)
        try? FileManager.default.createDirectory(atPath: dayDir, withIntermediateDirectories: true)
        let tf = DateFormatter()
        tf.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let base = (sourcePath as NSString).lastPathComponent
        let stem = (base as NSString).deletingPathExtension
        let filename = "\(tf.string(from: Date()))_\(stem)_upscaled.png"
        return (dayDir as NSString).appendingPathComponent(filename)
    }
}
