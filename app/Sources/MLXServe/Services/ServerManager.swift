import Foundation
import AppKit
import Darwin

@MainActor
class ServerManager: ObservableObject {
    @Published var status: ServerStatus = .stopped
    @Published var modelInfo: ModelInfo?
    /// Plan 05 Phase G — full registry snapshot, refreshed on the slow-poll
    /// loop. UI uses this for the "Loaded Models" section and the model
    /// picker. First entry mirrors `modelInfo` (server sorts default first).
    @Published var allModels: [ModelInfo] = []
    @Published var memoryInfo: MemoryInfo?
    @Published var port: UInt16 = 11234
    @Published var currentModelPath: String = ""
    @Published var lastError: String = ""

    private var process: Process?
    private var healthTimer: Timer?
    private var healthTask: Task<Void, Never>?
    private var pollSource: DispatchSourceTimer?
    private let api = APIClient()
    @Published var serverLog = ""
    /// Snapshot of the ServerOptions used the last time `start()` was called.
    /// Settings UI compares this against the current options to decide whether
    /// the restart banner should appear.
    @Published var lastLaunchedOptions: ServerOptions?

    var baseURL: String { "http://localhost:\(port)" }

    /// True when the user has edited any server-launch field since the running
    /// process was started. Per-request defaults never trigger this — they
    /// apply on the next chat request.
    func needsRestartFor(_ current: ServerOptions) -> Bool {
        guard let last = lastLaunchedOptions else { return false }
        return !last.serverLaunchEquals(current)
    }

    func start(modelPath: String, options: ServerOptions) {
        guard status != .running, status != .starting else { return }

        // Resolve symlinks for the model path
        let resolvedModel = (modelPath as NSString).resolvingSymlinksInPath
        currentModelPath = resolvedModel
        port = options.port
        status = .starting
        lastError = ""
        serverLog = ""

        // Reap orphaned mlx-serve processes still bound to our port (e.g. left
        // behind after a crash). We only target processes whose command name is
        // `mlx-serve` so we never kill an unrelated app that happens to be on
        // the port.
        killOrphanedServers(on: port)

        let binaryPath = resolveBinaryPath()
        guard FileManager.default.fileExists(atPath: binaryPath) else {
            status = .error("Binary not found at: \(binaryPath)")
            lastError = "mlx-serve binary not found"
            return
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: binaryPath)
        var args = ["--model", resolvedModel]
        // Plan 05 Phase G — pass the parent directory as --model-dir so the
        // registry discovers all siblings at startup. Lets the user hot-load
        // any sibling later via the model picker (when hot-switch is on).
        let modelDir = (resolvedModel as NSString).deletingLastPathComponent
        args += options.toCLIArgs(modelDirOverride: modelDir.isEmpty ? nil : modelDir)
        proc.arguments = args
        lastLaunchedOptions = options

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
            // Intentional shutdown: detach handlers so the trailing exit and any
            // final stderr (e.g. "Shutting down gracefully...") can't bleed into
            // the next server's serverLog or trigger handleTermination's error path
            // if start() runs before the old process has fully exited.
            proc.terminationHandler = nil
            if let pipe = proc.standardError as? Pipe {
                pipe.fileHandleForReading.readabilityHandler = nil
            }
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

    func toggle(modelPath: String, options: ServerOptions) {
        if status == .running || status == .starting {
            stop()
        } else {
            start(modelPath: modelPath, options: options)
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
        let fullLog = errSnippet.isEmpty ? "(no stderr captured — exit code \(exitCode))" : errSnippet

        if case .running = status {
            status = .error("Exited unexpectedly")
            lastError = shortErr
            presentCrashAlert(title: "mlx-serve exited unexpectedly", log: fullLog, exitCode: exitCode)
        } else if case .starting = status {
            status = .error("Failed to start")
            lastError = shortErr
            presentCrashAlert(title: "mlx-serve failed to start", log: fullLog, exitCode: exitCode)
        } else {
            status = .stopped
        }
    }

    /// Modal alert for server crashes — surfaces the full stderr log in a
    /// scrollable, selectable text view so the user can copy/paste it into
    /// a bug report instead of digging through the menu-bar log icon.
    private func presentCrashAlert(title: String, log: String, exitCode: Int32) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = "Exit code \(exitCode). Full server log below — select & copy, or use the Copy Log button."
        alert.alertStyle = .warning

        // Scrollable, selectable, monospaced log view as the accessory.
        let scroll = NSScrollView(frame: NSRect(x: 0, y: 0, width: 640, height: 280))
        scroll.hasVerticalScroller = true
        scroll.hasHorizontalScroller = true
        scroll.borderType = .bezelBorder
        scroll.autohidesScrollers = false

        let textView = NSTextView(frame: scroll.bounds)
        textView.isEditable = false
        textView.isSelectable = true
        textView.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        textView.string = log
        textView.textContainer?.widthTracksTextView = false
        textView.textContainer?.containerSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude
        )
        textView.isHorizontallyResizable = true
        textView.autoresizingMask = [.width]
        textView.scrollToEndOfDocument(nil)

        scroll.documentView = textView
        alert.accessoryView = scroll

        alert.addButton(withTitle: "Copy Log")
        alert.addButton(withTitle: "OK")

        // LSUIElement app: surface the alert above any focused app.
        NSApp.activate(ignoringOtherApps: true)
        let response = alert.runModal()
        if response == .alertFirstButtonReturn {
            let pb = NSPasteboard.general
            pb.clearContents()
            pb.setString(log, forType: .string)
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
        Task { await self.refreshModels() }
    }

    /// Called by TestServer when it detects health is ok but status is still starting
    func forceRunning() {
        transitionToRunning()
    }

    private func refreshStatus() async {
        if let mem = try? await api.fetchProps(port: port) {
            memoryInfo = mem
        }
        // Plan 05 Phase G — refresh registry snapshot too. Cheap (a few KB
        // per call) and keeps the UI's loaded/unloaded badges in sync with
        // server-side hot-load / eviction events.
        await refreshModels()
    }

    private func refreshModels() async {
        if let all = try? await api.fetchAllModels(port: port) {
            allModels = all
            if let first = all.first { modelInfo = first }
        }
    }

    /// Plan 05 Phase G — explicit hot-load. Posts /v1/load-model and
    /// refreshes the model list on success. Throws on 404/500/timeout so
    /// callers can fall back to a server restart if hot-switch fails.
    func loadModel(id: String, drafterPath: String? = nil) async throws -> ModelInfo {
        let info = try await api.loadModel(port: port, id: id, drafterPath: drafterPath)
        await refreshModels()
        return info
    }

    private func killOrphanedServers(on port: UInt16) {
        let pids = pidsListening(on: port)
        let myPid = ProcessInfo.processInfo.processIdentifier
        for pid in pids where pid != myPid {
            guard processName(pid: pid).hasPrefix("mlx-serve") else { continue }
            kill(pid, SIGTERM)
            for _ in 0..<20 {
                if kill(pid, 0) != 0 { break } // process gone
                Thread.sleep(forTimeInterval: 0.1)
            }
            if kill(pid, 0) == 0 { kill(pid, SIGKILL) }
        }
    }

    private func pidsListening(on port: UInt16) -> [pid_t] {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
        task.arguments = ["-nP", "-iTCP:\(port)", "-sTCP:LISTEN", "-t"]
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = FileHandle.nullDevice
        do { try task.run() } catch { return [] }
        task.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""
        return output.split(whereSeparator: { $0.isNewline || $0.isWhitespace })
            .compactMap { pid_t($0) }
    }

    private func processName(pid: pid_t) -> String {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/ps")
        task.arguments = ["-p", "\(pid)", "-o", "comm="]
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = FileHandle.nullDevice
        do { try task.run() } catch { return "" }
        task.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let raw = (String(data: data, encoding: .utf8) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        // `ps -o comm=` returns the full path of the executable on macOS;
        // take the last path component for prefix matching.
        return (raw as NSString).lastPathComponent
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
