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
        phase = .running(step: 0, total: 3, message: "Loading model…")
        log = []

        let outputPath = Self.makeOutputPath(prompt: request.prompt)
        let prompt = request.prompt
        let fps = request.fps
        let numFrames = request.numFrames
        let height = request.height
        let width = request.width
        let seed = request.seed
        let steps = request.steps
        let keep = request.keepResident

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
                let body: [String: Any] = [
                    "model": info.name, "prompt": prompt, "num_frames": numFrames,
                    "height": height, "width": width, "steps": steps, "seed": seed,
                ]
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
                        phase = .running(step: step, total: max(total, 1), message: "\(stage)…")
                    case "complete":
                        decoded = Self.decodeFrames(ev)
                    case "error":
                        await releaseIfNeeded()
                        phase = .failed(ev["message"] as? String ?? "Generation failed.")
                        return
                    default:
                        break
                    }
                }
                await releaseIfNeeded()
                guard let frames = decoded else {
                    phase = .failed("Server returned no video frames.")
                    return
                }
                if Task.isCancelled { phase = .idle; return }
                phase = .running(step: steps, total: steps, message: "Encoding mp4…")
                let outFps = frames.fps > 0 ? frames.fps : fps
                try await Task.detached(priority: .userInitiated) {
                    try VideoGenService.writeMP4(
                        rgb: frames.rgb, frames: frames.frames,
                        width: frames.width, height: frames.height,
                        fps: outFps, to: URL(fileURLWithPath: outputPath))
                }.value
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

    // MARK: - Decode + mux (pure / nonisolated so they're testable + off-main)

    struct DecodedFrames: Equatable {
        var rgb: Data        // [frames * height * width * 3] row-major RGB
        var frames: Int
        var height: Int
        var width: Int
        var fps: Int
    }

    /// Parse the native server's `{frames,height,width,fps,format,data}` body.
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
        return DecodedFrames(rgb: rgb, frames: frames, height: height, width: width, fps: fps)
    }

    enum MuxError: Error { case writerInit, noPool, finishFailed(String) }

    /// Mux raw RGB frames → h264 mp4 via AVAssetWriter.
    nonisolated static func writeMP4(rgb: Data, frames: Int, width: Int, height: Int, fps: Int, to url: URL) throws {
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
        guard writer.startWriting() else { throw writer.error ?? MuxError.writerInit }
        writer.startSession(atSourceTime: .zero)
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
            throw MuxError.finishFailed(writer.error?.localizedDescription ?? "unknown")
        }
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
