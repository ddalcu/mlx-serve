import Foundation
import SwiftUI

/// Manages a shared Python venv at `~/.mlx-serve/venv` used by the optional
/// image/video generation features. Both features spawn Python subprocesses
/// against this venv — the Zig server itself stays Python-free.
///
/// Responsibilities:
///   - discover a suitable system `python3`,
///   - create & populate the venv with heavy ML deps (mflux, ltx-2-mlx, …),
///   - report install progress back to the UI,
///   - run generation scripts with streamed JSON progress events.
@MainActor
final class PythonManager: ObservableObject {

    /// Where the venv lives. Shared by all Python features.
    nonisolated static let venvDir: String = {
        let dir = NSString(string: "~/.mlx-serve/venv").expandingTildeInPath
        return dir
    }()

    /// Output roots for generated media.
    nonisolated static let imagesRoot: String = {
        let dir = NSString(string: "~/.mlx-serve/generations/images").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return dir
    }()

    nonisolated static let videosRoot: String = {
        let dir = NSString(string: "~/.mlx-serve/generations/videos").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return dir
    }()

    /// Python path to use when invoking scripts — the venv's `bin/python`.
    nonisolated static var venvPython: String { (venvDir as NSString).appendingPathComponent("bin/python") }
    nonisolated static var venvPip: String { (venvDir as NSString).appendingPathComponent("bin/pip") }

    /// Packages required for image + video generation.
    ///
    /// - **Image**: `mflux` runs FLUX natively on MLX with built-in 4/8-bit
    ///   quantization — the only path that actually fits FLUX in <32 GB of
    ///   unified memory on Apple Silicon.
    /// - **Video**: `ltx-2-mlx` is a native MLX pipeline for LTX-Video 2.3
    ///   on Apple Silicon (much faster than diffusers on MPS, quantized
    ///   weights fit in 24 GB). It's a uv workspace, so pip can't resolve
    ///   workspace siblings — install each subpackage explicitly (core first).
    ///
    /// Both packages share `safetensors`, `pillow`, `huggingface_hub` etc.
    /// so a single install covers both features.
    nonisolated static let requiredPackages: [String] = [
        "mflux>=0.5.0",
        // ltx-2-mlx workspace subpackages — pip can't resolve uv-workspace siblings,
        // so we install each one explicitly. Order matters: core first.
        "ltx-core-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git#subdirectory=packages/ltx-core-mlx",
        "ltx-pipelines-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git#subdirectory=packages/ltx-pipelines-mlx",
        "huggingface_hub",
        "safetensors",
        "pillow",
    ]

    /// Module names we probe to tell whether the venv is usable. Must match
    /// the import names, not the pip distribution names.
    nonisolated static let requiredImports: [String] = [
        "mflux", "ltx_pipelines_mlx", "ltx_core_mlx",
        "PIL", "safetensors", "huggingface_hub",
    ]

    enum Status: Equatable {
        case unknown              // haven't checked yet
        case missingPython        // no system python3 found at all
        case needsVenv            // python3 found but venv not created
        case needsPackages        // venv exists but missing modules
        case needsFFmpeg          // venv ready, but system ffmpeg missing (video muxing)
        case ready                // venv + packages + ffmpeg present

        /// Fully ready for every feature (image + video).
        var isReady: Bool { self == .ready }

        /// Image generation (mflux) doesn't shell out to ffmpeg, so
        /// `.needsFFmpeg` is still a usable state for it.
        var imagesReady: Bool { self == .ready || self == .needsFFmpeg }
    }

    @Published private(set) var status: Status = .unknown
    @Published private(set) var systemPython: String? = nil
    /// Live install/uninstall log — the install sheet streams this.
    @Published private(set) var installLog: [String] = []
    @Published private(set) var isInstalling: Bool = false
    /// Set when a package check / install fails, so the UI can surface the cause.
    @Published private(set) var lastError: String? = nil

    private var installProcess: Process?

    init() {
        Task { await refresh() }
    }

    // MARK: - Detection

    /// Re-scan the environment: does python3 exist, is the venv there, are
    /// required modules importable? Cheap — single python invocation when the
    /// venv already exists.
    func refresh() async {
        // Try to locate system python3 only if we don't have one yet. Cached
        // between refreshes since it rarely changes.
        if systemPython == nil {
            systemPython = await Self.findSystemPython()
        }
        guard systemPython != nil else {
            status = .missingPython
            return
        }
        let fm = FileManager.default
        guard fm.fileExists(atPath: Self.venvPython) else {
            status = .needsVenv
            return
        }
        let ok = await Self.checkPackages()
        guard ok else { status = .needsPackages; return }
        status = Self.checkFFmpeg() ? .ready : .needsFFmpeg
    }

    /// Probe common Homebrew + system paths for a `ffmpeg` binary. The
    /// ltx-2-mlx pipeline shells out to `ffmpeg` for video muxing (via
    /// `shutil.which`), so a missing binary surfaces as an opaque runtime
    /// error deep inside generation — we gate on it up front instead.
    nonisolated static func checkFFmpeg() -> Bool {
        let candidates = [
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/usr/bin/ffmpeg",
        ]
        for path in candidates where FileManager.default.isExecutableFile(atPath: path) {
            return true
        }
        return false
    }

    /// Look for `python3` on common system paths and via an interactive login
    /// shell (picks up pyenv, asdf, Homebrew, etc.). Same approach as
    /// CLILauncher — `-i -l -c` because users usually configure PATH in
    /// `.zshrc`, not `.zprofile`.
    nonisolated static func findSystemPython() async -> String? {
        let candidates = [
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ]
        for path in candidates where FileManager.default.isExecutableFile(atPath: path) {
            return path
        }
        return await Task.detached { () -> String? in
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/bin/zsh")
            proc.arguments = ["-i", "-l", "-c", "command -v python3 2>/dev/null"]
            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = Pipe()
            do {
                try proc.run()
                proc.waitUntilExit()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                let out = (String(data: data, encoding: .utf8) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                return out.isEmpty ? nil : out
            } catch { return nil }
        }.value
    }

    /// Probe the venv by importing every module in `requiredImports`. Using one
    /// `python -c` call keeps latency low (single interpreter startup).
    nonisolated static func checkPackages() async -> Bool {
        guard FileManager.default.isExecutableFile(atPath: venvPython) else { return false }
        let script = "import " + requiredImports.joined(separator: ", ")
        return await Task.detached { () -> Bool in
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: venvPython)
            proc.arguments = ["-c", script]
            proc.standardOutput = Pipe()
            proc.standardError = Pipe()
            do {
                try proc.run()
                proc.waitUntilExit()
                return proc.terminationStatus == 0
            } catch {
                return false
            }
        }.value
    }

    // MARK: - Install

    /// Install everything: create venv if missing, then `pip install` all
    /// required packages. Streams output into `installLog` so the install
    /// sheet can tail it.
    func install() async {
        guard !isInstalling else { return }
        isInstalling = true
        lastError = nil
        installLog = []
        defer { isInstalling = false }

        guard let python = systemPython else {
            appendLog("ERROR: system python3 not found.")
            lastError = "System python3 not found. Install Python 3 from python.org or via Homebrew."
            return
        }

        let fm = FileManager.default
        if !fm.fileExists(atPath: Self.venvPython) {
            appendLog("Creating venv at \(Self.venvDir)...")
            let ok = await runAndStream(
                executable: python,
                args: ["-m", "venv", Self.venvDir]
            )
            guard ok else {
                lastError = "Failed to create venv. Check Python installation."
                return
            }
        }

        // Upgrade pip first — the stock pip bundled with Python is often too
        // old to resolve torch's dependency markers cleanly.
        appendLog("Upgrading pip...")
        _ = await runAndStream(
            executable: Self.venvPython,
            args: ["-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"]
        )

        appendLog("Installing packages (this can take several minutes)...")
        var args = ["-m", "pip", "install", "--upgrade"]
        args.append(contentsOf: Self.requiredPackages)
        let ok = await runAndStream(executable: Self.venvPython, args: args)
        if !ok {
            lastError = "pip install failed. See log for details."
            await refresh()
            return
        }

        appendLog("Verifying imports...")
        let works = await Self.checkPackages()
        if works {
            appendLog("Done.")
        } else {
            appendLog("ERROR: import check failed after install.")
            lastError = "Packages installed but imports failed. Check log."
        }
        await refresh()
    }

    /// Remove the entire venv. Cheap escape hatch if an install is broken or
    /// the user wants to reclaim disk.
    func uninstall() async {
        if isInstalling { return }
        try? FileManager.default.removeItem(atPath: Self.venvDir)
        installLog = []
        await refresh()
    }

    // MARK: - Script execution (generation-side)

    /// Spawn a Python script and return an async stream of events. The script
    /// is expected to print JSON-per-line events to stdout. The Swift caller
    /// owns cancellation: cancelling the task sends SIGTERM to the process.
    ///
    /// Events have shape `{"type":"progress"|"complete"|"error", ...}`; any
    /// non-JSON stderr line becomes a `.log` event so the UI can surface
    /// traceback details.
    func runScript(source: String, args: [String] = [], env: [String: String] = [:]) -> AsyncThrowingStream<GenEvent, Error> {
        let py = Self.venvPython
        return AsyncThrowingStream { continuation in
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: py)
            proc.arguments = ["-c", source] + args

            var environment = ProcessInfo.processInfo.environment
            // Unbuffered stdout so we see progress live rather than in one
            // flush at the end. `-u` on the interpreter does the same thing
            // but this is more explicit for children Python may spawn.
            environment["PYTHONUNBUFFERED"] = "1"
            environment["PYTHONIOENCODING"] = "utf-8"
            for (k, v) in env { environment[k] = v }
            proc.environment = environment

            let outPipe = Pipe()
            let errPipe = Pipe()
            proc.standardOutput = outPipe
            proc.standardError = errPipe

            // stdout: JSON events line-by-line.
            outPipe.fileHandleForReading.readabilityHandler = { handle in
                let data = handle.availableData
                guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
                for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
                    let trimmed = line.trimmingCharacters(in: .whitespaces)
                    guard !trimmed.isEmpty else { continue }
                    if let event = GenEvent.parse(trimmed) {
                        continuation.yield(event)
                    } else {
                        continuation.yield(.log(trimmed))
                    }
                }
            }

            // stderr: pure log lines — carries tracebacks, HF download
            // progress, torch warnings.
            errPipe.fileHandleForReading.readabilityHandler = { handle in
                let data = handle.availableData
                guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
                for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
                    let trimmed = String(line)
                    if !trimmed.isEmpty { continuation.yield(.log(trimmed)) }
                }
            }

            proc.terminationHandler = { p in
                outPipe.fileHandleForReading.readabilityHandler = nil
                errPipe.fileHandleForReading.readabilityHandler = nil
                if p.terminationStatus == 0 {
                    continuation.finish()
                } else if p.terminationReason == .uncaughtSignal {
                    continuation.finish(throwing: GenError.cancelled)
                } else {
                    continuation.finish(throwing: GenError.exited(Int(p.terminationStatus)))
                }
            }

            continuation.onTermination = { _ in
                // Cancellation path: graceful SIGTERM. If the child ignores
                // it (torch is usually responsive), SIGKILL after a beat.
                if proc.isRunning {
                    proc.terminate()
                    DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
                        if proc.isRunning { kill(proc.processIdentifier, SIGKILL) }
                    }
                }
            }

            do { try proc.run() }
            catch { continuation.finish(throwing: error) }
        }
    }

    // MARK: - Private helpers

    private func appendLog(_ line: String) {
        installLog.append(line)
        // Cap log length to keep memory bounded on long installs.
        if installLog.count > 4000 {
            installLog.removeFirst(installLog.count - 4000)
        }
    }

    /// Run a child process, tee stdout+stderr into `installLog`, return whether
    /// it exited zero. Blocks on the caller until the child exits.
    private func runAndStream(executable: String, args: [String]) async -> Bool {
        await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: executable)
            proc.arguments = args
            var env = ProcessInfo.processInfo.environment
            env["PYTHONUNBUFFERED"] = "1"
            proc.environment = env
            let outPipe = Pipe()
            let errPipe = Pipe()
            proc.standardOutput = outPipe
            proc.standardError = errPipe

            installProcess = proc

            let onData: @Sendable (FileHandle) -> Void = { [weak self] handle in
                let data = handle.availableData
                guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
                Task { @MainActor in
                    for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
                        self?.appendLog(String(line))
                    }
                }
            }
            outPipe.fileHandleForReading.readabilityHandler = onData
            errPipe.fileHandleForReading.readabilityHandler = onData

            proc.terminationHandler = { p in
                outPipe.fileHandleForReading.readabilityHandler = nil
                errPipe.fileHandleForReading.readabilityHandler = nil
                continuation.resume(returning: p.terminationStatus == 0)
            }

            do {
                try proc.run()
            } catch {
                Task { @MainActor in self.appendLog("ERROR: \(error.localizedDescription)") }
                continuation.resume(returning: false)
            }
        }
    }
}

/// One event from a running generation script.
enum GenEvent {
    case progress(step: Int, total: Int, message: String)
    case complete(path: String)
    case log(String)

    /// Parse a JSON line from Python. Returns nil if not a valid event so the
    /// caller can fall back to treating it as log output.
    static func parse(_ line: String) -> GenEvent? {
        guard let data = line.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = obj["type"] as? String else { return nil }
        switch type {
        case "progress":
            let step = (obj["step"] as? Int) ?? 0
            let total = (obj["total"] as? Int) ?? 0
            let msg = (obj["message"] as? String) ?? ""
            return .progress(step: step, total: total, message: msg)
        case "complete":
            guard let path = obj["path"] as? String else { return nil }
            return .complete(path: path)
        case "error":
            let msg = (obj["message"] as? String) ?? "unknown error"
            return .log("ERROR: " + msg)
        default:
            return nil
        }
    }
}

enum GenError: LocalizedError {
    case cancelled
    case exited(Int)
    case notReady
    case insufficientRAM(requiredGB: Int, availableGB: Int)

    var errorDescription: String? {
        switch self {
        case .cancelled: return "Generation cancelled."
        case .exited(let code): return "Python exited with code \(code)."
        case .notReady: return "Python environment is not ready."
        case .insufficientRAM(let r, let a):
            return "This model needs ~\(r) GB free RAM but only ~\(a) GB is available."
        }
    }
}
