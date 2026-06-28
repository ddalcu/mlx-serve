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

    func generate(_ request: AudioGenRequest) {
        guard !request.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            phase = .failed("Text is empty.")
            return
        }
        // Native path: serve the TTS model with a dedicated mlx-serve instance and
        // POST /v1/audio/speech. No Python.
        guard let modelDir = NativeGenServer.resolveModelDir(repo: request.model.repo) else {
            phase = .failed("Model \(request.model.repo) is not downloaded. Download it first.")
            return
        }

        task?.cancel()
        phase = .running(step: 0, total: 3, message: "Loading model…")
        log = []

        let outputPath = Self.makeOutputPath(text: request.text)
        let repo = request.model.repo
        let text = request.text

        task = Task {
            do {
                _ = try await NativeGenServer.shared.ensure(modelDir: modelDir)
                if Task.isCancelled { phase = .idle; return }
                phase = .running(step: 1, total: 3, message: "Synthesizing…")
                let data = try await NativeGenServer.shared.post(
                    path: "/v1/audio/speech",
                    json: ["model": repo, "input": text]
                )
                guard data.count > 44 else {
                    phase = .failed("Server returned an empty audio response.")
                    return
                }
                try data.write(to: URL(fileURLWithPath: outputPath))
                phase = .completed(path: outputPath)
                insertRecent(outputPath)
            } catch is CancellationError {
                phase = .idle
            } catch {
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
        let root = PythonManager.audiosRoot
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
        let dayDir = (PythonManager.audiosRoot as NSString).appendingPathComponent(day)
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

    static func buildArgs(_ r: AudioGenRequest, outputPath: String) -> [String] {
        var args = [
            "--repo", r.model.repo,
            "--text", r.text,
            "--speed", String(r.speed),
            "--temperature", String(r.temperature),
            "--output", outputPath,
        ]
        if let ref = r.refAudioPath, !ref.isEmpty {
            args.append(contentsOf: ["--ref-audio", ref])
            let t = r.refText.trimmingCharacters(in: .whitespacesAndNewlines)
            if !t.isEmpty {
                args.append(contentsOf: ["--ref-text", t])
            }
        }
        return args
    }

    /// Python script for neural TTS / voice cloning via `mlx-audio`. Passes the
    /// HF repo straight to `generate_audio`, which downloads + loads it. When a
    /// reference clip is given it clones that voice (auto-transcribing with
    /// Whisper if no `--ref-text`). `join_audio=True` yields a single file; we
    /// then move whatever it produced onto the exact `--output` path so the
    /// Swift side can trust the `complete` event.
    static let script: String = #"""
import sys, json, argparse, traceback, os, glob

def emit(obj):
    print(json.dumps(obj), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--ref-audio", dest="ref_audio", default=None)
    p.add_argument("--ref-text", dest="ref_text", default=None)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    try:
        from mlx_audio.tts.generate import generate_audio
    except Exception as e:
        emit({"type":"error","message":f"mlx-audio import failed: {e}. Re-run the installer."})
        traceback.print_exc()
        sys.exit(1)

    out_dir = os.path.dirname(args.output)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.output))[0]
    prefix = base + "__seg"

    emit({"type":"progress","step":0,"total":3,
          "message":f"Loading {args.repo} (first run downloads weights)..."})

    kwargs = dict(
        text=args.text, model=args.repo, speed=args.speed, temperature=args.temperature,
        output_path=out_dir, file_prefix=prefix, audio_format="wav",
        join_audio=True, save=True, verbose=False,
    )
    if args.ref_audio:
        if not os.path.exists(args.ref_audio):
            emit({"type":"error","message":f"Reference audio not found: {args.ref_audio}"})
            sys.exit(1)
        kwargs["ref_audio"] = args.ref_audio
        if args.ref_text:
            kwargs["ref_text"] = args.ref_text
        emit({"type":"progress","step":1,"total":3,"message":"Cloning reference voice and synthesizing..."})
    else:
        emit({"type":"progress","step":1,"total":3,"message":"Synthesizing speech..."})

    try:
        generate_audio(**kwargs)
    except Exception as e:
        emit({"type":"error","message":str(e)})
        traceback.print_exc()
        sys.exit(1)

    produced = sorted(glob.glob(os.path.join(out_dir, prefix + "*.wav")))
    if not produced:
        emit({"type":"error","message":"Generation finished but produced no audio file."})
        sys.exit(1)
    try:
        os.replace(produced[0], args.output)
        for extra in produced[1:]:
            try: os.remove(extra)
            except OSError: pass
    except OSError as e:
        emit({"type":"error","message":f"Could not finalize output: {e}"})
        sys.exit(1)

    emit({"type":"progress","step":2,"total":3,"message":"Done."})
    emit({"type":"complete","path":args.output})

if __name__ == "__main__":
    main()
"""#
}
