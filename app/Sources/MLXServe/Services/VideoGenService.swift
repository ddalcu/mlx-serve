import Foundation
import SwiftUI
import AppKit

/// Runs LTX-Video generation via the shared `PythonManager` venv. Mirrors
/// `ImageGenService`, but writes an mp4 and preserves a lot more metadata
/// (frames, fps) per request.
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

    private let python: PythonManager
    private var task: Task<Void, Never>?

    init(python: PythonManager) {
        self.python = python
        loadRecent()
    }

    var isRunning: Bool {
        if case .running = phase { return true }
        return false
    }

    func generate(_ request: VideoGenRequest) {
        guard python.status.isReady else {
            phase = .failed("Python environment is not ready.")
            return
        }
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            phase = .failed("Prompt is empty.")
            return
        }

        task?.cancel()
        // 4-phase progress: Download → Load → Generate → Encode.
        phase = .running(step: 0, total: 4, message: "Starting...")
        log = []

        let outputPath = Self.makeOutputPath(prompt: request.prompt)
        let args = Self.buildArgs(request, outputPath: outputPath)

        task = Task {
            do {
                let stream = python.runScript(source: Self.script, args: args)
                for try await event in stream {
                    switch event {
                    case .progress(let step, let total, let message):
                        phase = .running(step: step, total: total, message: message)
                    case .complete(let path):
                        phase = .completed(path: path)
                        insertRecent(path)
                    case .log(let line):
                        appendLog(line)
                    }
                }
                if FileManager.default.fileExists(atPath: outputPath) {
                    if case .completed = phase { /* already set */ } else {
                        phase = .completed(path: outputPath)
                        insertRecent(outputPath)
                    }
                } else if case .running = phase {
                    phase = .failed("Generation finished without output.")
                }
            } catch GenError.cancelled {
                phase = .idle
            } catch {
                let actionable = log.last(where: { $0.hasPrefix("ERROR: ") })
                    .map { String($0.dropFirst("ERROR: ".count)) }
                phase = .failed(actionable ?? error.localizedDescription)
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
        let root = PythonManager.videosRoot
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
        let dayDir = (PythonManager.videosRoot as NSString).appendingPathComponent(day)
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

    private static func buildArgs(_ r: VideoGenRequest, outputPath: String) -> [String] {
        [
            "--repo", r.model.repo,
            "--mode", r.mode.rawValue,
            "--prompt", r.prompt,
            "--width", String(r.width),
            "--height", String(r.height),
            "--frames", String(r.numFrames),
            "--steps", String(r.steps),
            "--cfg", String(r.cfgScale),
            "--seed", String(r.seed),
            "--output", outputPath,
        ]
    }

    /// Python script for LTX-Video 2.3 generation via `ltx-2-mlx`. Single
    /// pipeline path — no fallbacks — mirrors the shape of the mflux image
    /// script. Progress is coarse 4-phase (Download → Load → Generate →
    /// Encode); the TextToVideoPipeline does not expose a per-step callback,
    /// so finer granularity would need monkey-patching `mlx_arsenal.scheduler`.
    static let script: String = #"""
import sys, json, argparse, traceback, os, shutil

def emit(obj):
    print(json.dumps(obj), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--mode", required=True, choices=["oneStage","twoStage","twoStageHQ"])
    p.add_argument("--prompt", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=704)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--frames", type=int, default=97)
    p.add_argument("--steps", type=int, default=8,
                   help="num_steps for oneStage; stage1_steps for two-stage modes.")
    p.add_argument("--cfg", type=float, default=3.0,
                   help="CFG scale; ignored by oneStage.")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    # Up-front ffmpeg check so the user sees an actionable error instead of
    # the generic RuntimeError from inside ltx_core_mlx after Gemma loads.
    if not shutil.which("ffmpeg"):
        emit({"type":"error","message":"ffmpeg not found on PATH. Install it with: brew install ffmpeg"})
        sys.exit(1)

    try:
        from ltx_pipelines_mlx import TextToVideoPipeline, TwoStagePipeline, TwoStageHQPipeline
        from huggingface_hub import snapshot_download
    except Exception as e:
        emit({"type":"error","message":f"ltx-2-mlx import failed: {e}. Re-run the installer."})
        traceback.print_exc()
        sys.exit(1)

    emit({"type":"progress","step":0,"total":4,
          "message":f"Downloading {args.repo} (first run only, ~41 GB)..."})
    try:
        local = snapshot_download(repo_id=args.repo)
    except Exception as e:
        emit({"type":"error","message":f"Model download failed: {e}"})
        traceback.print_exc()
        sys.exit(1)

    emit({"type":"progress","step":1,"total":4,
          "message":"Loading Gemma text encoder + LTX transformer..."})
    PIPELINES = {
        "oneStage":   TextToVideoPipeline,
        "twoStage":   TwoStagePipeline,
        "twoStageHQ": TwoStageHQPipeline,
    }
    try:
        pipe = PIPELINES[args.mode](
            model_dir=local,
            gemma_model_id="mlx-community/gemma-3-12b-it-4bit",
            low_memory=True,
        )
    except Exception as e:
        emit({"type":"error","message":f"Pipeline load failed: {e}"})
        traceback.print_exc()
        sys.exit(1)

    emit({"type":"progress","step":2,"total":4,
          "message":f"Generating {args.frames} frames @ {args.width}x{args.height} ({args.mode})..."})
    try:
        if args.mode == "oneStage":
            pipe.generate_and_save(
                prompt=args.prompt,
                output_path=args.output,
                height=args.height, width=args.width,
                num_frames=args.frames, seed=args.seed,
                num_steps=args.steps,
            )
        else:
            pipe.generate_and_save(
                prompt=args.prompt,
                output_path=args.output,
                height=args.height, width=args.width,
                num_frames=args.frames, seed=args.seed,
                stage1_steps=args.steps, cfg_scale=args.cfg,
            )
    except Exception as e:
        emit({"type":"error","message":str(e)})
        traceback.print_exc()
        sys.exit(1)

    emit({"type":"progress","step":3,"total":4,"message":"Encoding mp4 with audio (ffmpeg)..."})
    if not os.path.exists(args.output):
        emit({"type":"error","message":"Pipeline finished but output file is missing."})
        sys.exit(1)
    emit({"type":"progress","step":4,"total":4,"message":"Done."})
    emit({"type":"complete","path":args.output})

if __name__ == "__main__":
    main()
"""#
}
