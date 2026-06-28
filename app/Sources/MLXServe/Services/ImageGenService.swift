import Foundation
import SwiftUI
import AppKit

/// Runs FLUX.2 image generation via the shared `PythonManager` venv.
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

    init() {
        loadRecent()
    }

    var isRunning: Bool {
        if case .running = phase { return true }
        return false
    }

    func generate(_ request: ImageGenRequest) {
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            phase = .failed("Prompt is empty.")
            return
        }
        // Native path: serve the FLUX model with a dedicated mlx-serve instance and
        // POST /v1/images/generations. No Python.
        guard let modelDir = NativeGenServer.resolveModelDir(repo: request.model.repo) else {
            phase = .failed("Model \(request.model.repo) is not downloaded. Download it first.")
            return
        }

        task?.cancel()
        phase = .running(step: 0, total: request.steps, message: "Loading model…")
        log = []

        let outputPath = Self.makeOutputPath(prompt: request.prompt)
        let repo = request.model.repo
        let prompt = request.prompt
        let size = "\(request.width)x\(request.height)"
        let steps = request.steps

        task = Task {
            do {
                _ = try await NativeGenServer.shared.ensure(modelDir: modelDir)
                if Task.isCancelled { phase = .idle; return }
                phase = .running(step: 1, total: steps, message: "Generating…")
                let data = try await NativeGenServer.shared.post(
                    path: "/v1/images/generations",
                    json: ["model": repo, "prompt": prompt, "size": size]
                )
                guard let png = Self.decodePngB64(data) else {
                    phase = .failed("Server returned no image data.")
                    return
                }
                try png.write(to: URL(fileURLWithPath: outputPath))
                phase = .completed(path: outputPath)
                insertRecent(outputPath)
            } catch is CancellationError {
                phase = .idle
            } catch {
                phase = .failed(error.localizedDescription)
            }
        }
    }

    /// Extract the base64 PNG from an OpenAI `{data:[{b64_json}]}` response body.
    /// Pure + static so it's unit-testable without a running server.
    static func decodePngB64(_ body: Data) -> Data? {
        guard let obj = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let arr = obj["data"] as? [[String: Any]],
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
