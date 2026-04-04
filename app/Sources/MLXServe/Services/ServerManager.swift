import Foundation
import AppKit

@MainActor
class ServerManager: ObservableObject {
    @Published var status: ServerStatus = .stopped
    @Published var modelInfo: ModelInfo?
    @Published var memoryInfo: MemoryInfo?
    @Published var port: UInt16 = 8080
    @Published var currentModelPath: String = ""
    @Published var lastError: String = ""

    private var process: Process?
    private var healthTimer: Timer?
    private let api = APIClient()
    private var stderrBuf = ""

    var baseURL: String { "http://localhost:\(port)" }

    func start(modelPath: String) {
        guard status != .running, status != .starting else { return }

        // Resolve symlinks for the model path
        let resolvedModel = (modelPath as NSString).resolvingSymlinksInPath
        currentModelPath = resolvedModel
        status = .starting
        lastError = ""
        stderrBuf = ""

        let binaryPath = resolveBinaryPath()
        guard FileManager.default.fileExists(atPath: binaryPath) else {
            status = .error("Binary not found at: \(binaryPath)")
            lastError = "mlx-serve binary not found"
            return
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: binaryPath)
        proc.arguments = [
            "--model", resolvedModel,
            "--serve",
            "--port", "\(port)",
            "--log-level", "info"
        ]

        // Inherit environment + add framework paths for dylib loading
        var env = ProcessInfo.processInfo.environment
        if let frameworksPath = Bundle.main.privateFrameworksPath {
            let existing = env["DYLD_LIBRARY_PATH"] ?? ""
            env["DYLD_LIBRARY_PATH"] = existing.isEmpty ? frameworksPath : "\(frameworksPath):\(existing)"
            env["MLX_METAL_PATH"] = frameworksPath
        }
        proc.environment = env

        // Capture stderr to show errors in the UI
        let errPipe = Pipe()
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = errPipe

        errPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let str = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in
                self?.stderrBuf += str
                // Keep only last 2KB
                if let s = self, s.stderrBuf.count > 2048 {
                    s.stderrBuf = String(s.stderrBuf.suffix(2048))
                }
            }
        }

        proc.terminationHandler = { [weak self] proc in
            Task { @MainActor in
                self?.handleTermination(exitCode: proc.terminationStatus)
            }
        }

        do {
            try proc.run()
            process = proc
            startHealthPolling()
        } catch {
            status = .error(error.localizedDescription)
            lastError = error.localizedDescription
        }
    }

    func stop() {
        healthTimer?.invalidate()
        healthTimer = nil

        if let proc = process, proc.isRunning {
            proc.terminate()
            DispatchQueue.global().asyncAfter(deadline: .now() + 3) {
                if proc.isRunning { proc.interrupt() }
            }
        }
        process = nil
        status = .stopped
        modelInfo = nil
        memoryInfo = nil
    }

    func toggle(modelPath: String) {
        if status == .running || status == .starting {
            stop()
        } else {
            start(modelPath: modelPath)
        }
    }

    private func handleTermination(exitCode: Int32) {
        healthTimer?.invalidate()
        healthTimer = nil
        process = nil

        let errSnippet = stderrBuf.trimmingCharacters(in: .whitespacesAndNewlines)
        let shortErr = errSnippet.isEmpty ? "exit code \(exitCode)" : String(errSnippet.suffix(200))

        if case .running = status {
            status = .error("Exited unexpectedly")
            lastError = shortErr
        } else if case .starting = status {
            status = .error("Failed to start")
            lastError = shortErr
        } else {
            status = .stopped
        }
    }

    private func startHealthPolling() {
        healthTimer?.invalidate()
        healthTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in
                await self.pollHealth()
            }
        }
    }

    private func pollHealth() async {
        do {
            let healthy = try await api.checkHealth(port: port)
            if healthy && status != .running {
                status = .running
                if let info = try? await api.fetchModels(port: port) {
                    modelInfo = info
                }
                // Slow down polling once running
                healthTimer?.invalidate()
                healthTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
                    guard let self else { return }
                    Task { @MainActor in
                        await self.refreshStatus()
                    }
                }
            }
        } catch {
            if case .running = status {
                status = .error("Lost connection")
            }
        }
    }

    private func refreshStatus() async {
        if let mem = try? await api.fetchProps(port: port) {
            memoryInfo = mem
        }
    }

    private func resolveBinaryPath() -> String {
        // 1. Bundled: Contents/MacOS/mlx-serve (same dir as the Swift binary)
        if let execURL = Bundle.main.executableURL {
            let bundled = execURL.deletingLastPathComponent().appendingPathComponent("mlx-serve").path
            if FileManager.default.fileExists(atPath: bundled) {
                return bundled
            }
        }
        // 2. Development: zig-out/bin/mlx-serve relative to this source file's repo root
        //    The Swift package lives in app/, so the repo root is one level up.
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Services/
            .deletingLastPathComponent() // MLXServe/
            .deletingLastPathComponent() // Sources/
            .deletingLastPathComponent() // app/
            .path
        let zigOut = (repoRoot as NSString).appendingPathComponent("zig-out/bin/mlx-serve")
        if FileManager.default.fileExists(atPath: zigOut) {
            return zigOut
        }
        return "mlx-serve"
    }

    deinit {
        healthTimer?.invalidate()
        if let proc = process, proc.isRunning {
            proc.terminate()
        }
    }
}
