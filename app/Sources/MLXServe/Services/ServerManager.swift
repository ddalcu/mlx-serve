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
    private var healthTask: Task<Void, Never>?
    private var pollSource: DispatchSourceTimer?
    private let api = APIClient()
    @Published var serverLog = ""

    var baseURL: String { "http://localhost:\(port)" }

    func start(modelPath: String) {
        guard status != .running, status != .starting else { return }

        // Resolve symlinks for the model path
        let resolvedModel = (modelPath as NSString).resolvingSymlinksInPath
        currentModelPath = resolvedModel
        status = .starting
        lastError = ""
        serverLog = ""

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
                self?.serverLog += str
                // Keep only last 2KB
                if let s = self, s.serverLog.count > 65536 {
                    s.serverLog = String(s.serverLog.suffix(65536))
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
        pollSource?.cancel()
        pollSource = nil
        healthTask?.cancel()
        healthTask = nil
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
        pollSource?.cancel()
        pollSource = nil
        healthTask?.cancel()
        healthTask = nil
        healthTimer?.invalidate()
        healthTimer = nil
        process = nil

        let errSnippet = serverLog.trimmingCharacters(in: .whitespacesAndNewlines)
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
        healthTask?.cancel()
        let checkPort = port
        // Use a GCD timer on the main queue — guaranteed to fire even during init.
        // URLSession completion runs on a background queue and dispatches back to main.
        let source = DispatchSource.makeTimerSource(queue: .main)
        source.schedule(deadline: .now() + 1, repeating: 1.0)
        source.setEventHandler { [weak self] in
            guard let self else { source.cancel(); return }
            let url = URL(string: "http://127.0.0.1:\(checkPort)/health")!
            URLSession.shared.dataTask(with: url) { data, response, error in
                guard let http = response as? HTTPURLResponse, http.statusCode == 200,
                      let data, let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      json["status"] as? String == "ok" else { return }
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    if self.status != .running {
                        self.transitionToRunning()
                    }
                    // Switch to slow polling
                    source.cancel()
                    self.startSlowPolling()
                }
            }.resume()
        }
        source.resume()
        self.pollSource = source
    }

    private func startSlowPolling() {
        let source = DispatchSource.makeTimerSource(queue: .main)
        source.schedule(deadline: .now() + 3, repeating: 3.0)
        source.setEventHandler { [weak self] in
            guard let self else { source.cancel(); return }
            Task { await self.refreshStatus() }
        }
        source.resume()
        self.pollSource = source
    }

    private func transitionToRunning() {
        guard status != .running else { return }
        status = .running
        Task {
            if let info = try? await api.fetchModels(port: port) {
                modelInfo = info
            }
        }
    }

    /// Called by TestServer when it detects health is ok but status is still starting
    func forceRunning() {
        transitionToRunning()
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
        pollSource?.cancel()
        healthTask?.cancel()
        healthTimer?.invalidate()
        if let proc = process, proc.isRunning {
            proc.terminate()
        }
    }
}
