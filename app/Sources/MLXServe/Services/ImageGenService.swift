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

    func generate(_ request: ImageGenRequest) {
        guard python.status.imagesReady else {
            phase = .failed("Python environment is not ready.")
            return
        }
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            phase = .failed("Prompt is empty.")
            return
        }

        task?.cancel()
        phase = .running(step: 0, total: request.steps, message: "Starting...")
        log = []

        let outputPath = Self.makeOutputPath(prompt: request.prompt)
        let args = Self.buildArgs(request, outputPath: outputPath)

        task = Task {
            do {
                let stream = python.runScript(source: Self.mfluxScript, args: args)
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
                // Stream finished without a complete event — treat as success
                // if the file materialized, otherwise as failure.
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
                // Prefer the actual error message the Python script emitted
                // (kept in `log` with an "ERROR: " prefix) over the generic
                // "Python exited with code N". The Python message usually
                // includes actionable context (e.g. HF gated-repo auth steps).
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
        if recent.count > 60 { recent.removeLast(recent.count - 60) }
    }

    /// Scan the generations/images/ tree for existing files so the history
    /// shelf shows something on first launch.
    private func loadRecent() {
        let root = PythonManager.imagesRoot
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
        let dayDir = (PythonManager.imagesRoot as NSString).appendingPathComponent(day)
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

    private static func buildArgs(_ r: ImageGenRequest, outputPath: String) -> [String] {
        var args = [
            "--variant", r.model.variant.rawValue,
            "--config", r.model.configName,
            "--repo", r.model.repo,
            "--prompt", r.prompt,
            "--width", String(r.width),
            "--height", String(r.height),
            "--steps", String(r.steps),
            "--guidance", String(r.guidance),
            "--seed", String(r.seed),
            "--output", outputPath,
        ]
        if !r.negativePrompt.isEmpty {
            args.append(contentsOf: ["--negative", r.negativePrompt])
        }
        return args
    }

    // MARK: - Embedded Python

    /// mflux-based generation script. Uses the actual mflux 0.x API
    /// discovered by inspecting the installed package — the public top-
    /// level exports (`from mflux import Flux1`) are NOT available in
    /// current mflux releases despite older docs. The real API:
    ///
    ///   from mflux.models.flux.variants.txt2img.flux import Flux1
    ///   from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
    ///   from mflux.models.common.config.model_config import ModelConfig
    ///
    ///   flux = Flux1(model_config=ModelConfig.schnell(), quantize=4)
    ///   img  = flux.generate_image(seed=..., prompt=..., num_inference_steps=...,
    ///                              height=..., width=..., guidance=...)
    ///   img.save(...)  # GeneratedImage.save accepts a positional path
    ///
    /// Wire format (JSON-per-line on stdout):
    ///   {"type":"progress","step":N,"total":M,"message":"..."}
    ///   {"type":"complete","path":"/abs/path/to/output.png"}
    ///   {"type":"error","message":"..."}
    ///
    /// Per-step progress comes via tqdm interception — mflux uses tqdm
    /// internally for its denoise loop and doesn't expose a step callback.
    static let mfluxScript: String = #"""
import sys, json, argparse, traceback, random

def emit(obj):
    print(json.dumps(obj), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["flux1","flux2Klein4B","flux2Klein9B"])
    p.add_argument("--config", required=True, help="schnell|dev|flux2_klein_4b|flux2_klein_9b")
    p.add_argument("--repo", required=True,
                   help="Pre-quantized mflux-format HuggingFace repo to download.")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default="")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance", type=float, default=0.0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    # Hook tqdm before importing mflux so the patched class is what mflux
    # picks up for its denoise loop. mflux doesn't expose a step callback.
    #
    # We only emit progress for tqdm bars whose total looks like an inference
    # step counter (<= 200). HuggingFace downloads also use tqdm (file count,
    # byte counter in the billions) and emitting those would spam the UI with
    # nonsensical "Step 0/4619599996" events.
    try:
        import tqdm
        _orig = tqdm.tqdm
        class _ProgressTqdm(_orig):
            def update(self, n=1):
                super().update(n)
                if self.total and 0 < self.total <= 200:
                    emit({"type":"progress","step":int(self.n),
                          "total":int(self.total),
                          "message":f"Step {int(self.n)}/{int(self.total)}"})
        tqdm.tqdm = _ProgressTqdm
        try:
            import tqdm.auto
            tqdm.auto.tqdm = _ProgressTqdm
        except Exception:
            pass
    except Exception:
        pass  # progress will be coarse; generation still works

    try:
        from mflux.models.common.config.model_config import ModelConfig
        if args.variant == "flux1":
            from mflux.models.flux.variants.txt2img.flux import Flux1 as ModelClass
        else:
            from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein as ModelClass
    except Exception as e:
        emit({"type":"error","message":f"mflux import failed: {e}"})
        traceback.print_exc()
        sys.exit(1)

    # Resolve ModelConfig factory by name (e.g. "schnell" -> ModelConfig.schnell()).
    try:
        if not hasattr(ModelConfig, args.config):
            emit({"type":"error","message":f"Unknown ModelConfig factory: {args.config}"})
            sys.exit(1)
        model_config = getattr(ModelConfig, args.config)()
    except Exception as e:
        emit({"type":"error","message":f"ModelConfig failed: {e}"})
        traceback.print_exc()
        sys.exit(1)

    emit({"type":"progress","step":0,"total":args.steps,
          "message":f"Downloading {args.repo} (first run only)..."})
    try:
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(repo_id=args.repo)
        flux = ModelClass(model_config=model_config,
                          model_path=local_path, quantize=None)
    except Exception as e:
        emit({"type":"error","message":f"Load failed: {e}"})
        traceback.print_exc()
        sys.exit(1)

    seed = args.seed if args.seed >= 0 else random.randint(0, 2**32 - 1)
    emit({"type":"progress","step":0,"total":args.steps,"message":"Generating..."})

    # generate_image() in current mflux takes individual kwargs (no Config arg).
    # FLUX.1 supports negative_prompt; FLUX.2 doesn't expose it as a top-level
    # arg, so only forward it for flux1.
    gen_kwargs = dict(
        seed=seed,
        prompt=args.prompt,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        guidance=args.guidance,
    )
    if args.variant == "flux1" and args.negative:
        gen_kwargs["negative_prompt"] = args.negative

    try:
        result = flux.generate_image(**gen_kwargs)
    except Exception as e:
        emit({"type":"error","message":str(e)})
        traceback.print_exc()
        sys.exit(1)

    result.save(args.output)
    emit({"type":"complete","path":args.output})

if __name__ == "__main__":
    main()
"""#
}
