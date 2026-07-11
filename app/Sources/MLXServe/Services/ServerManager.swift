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
    /// True while the tray popover is on screen. Drives the live /props
    /// ticker — when the popover is closed there's nothing to render, so we
    /// stop polling entirely instead of burning 3 s ticks in the background.
    /// Toggled via `setMenuVisible(_:)` from `StatusMenuView`'s
    /// `onAppear`/`onDisappear`.
    private var menuIsVisible = false

    /// Off-main raw stderr buffer. **There is no `@Published` mirror.**
    ///
    /// Why: SwiftUI's `@EnvironmentObject` re-evaluates a view's `body` on
    /// any `@Published` change of the observed object, regardless of which
    /// properties the body actually reads. `ChatView` observes
    /// `ServerManager` (for status / model info), so a `@Published` log
    /// would force a ChatView body recompute on every flush — competing
    /// with the SSE token loop on the main thread. Even throttled to
    /// ~10 Hz that was enough to make generation visibly choppy when the
    /// log window was open.
    ///
    /// Instead, the log views own a small `LogPoller` (`@StateObject`)
    /// that ticks at its own rate and reads `currentServerLogSnapshot()`.
    /// Only those views re-render on log activity; everything else
    /// (ChatView, Settings, the menu popover header) is fully insulated
    /// from stderr volume.
    let logBuffer = ThrottledLogBuffer(maxBytes: serverLogMaxBytes)
    /// Hard cap on the retained stderr tail shown in the Server Log window.
    /// Was 64 KB (~800 lines) — a single chunked-prefill trace or a model-load
    /// dump scrolled everything else away. The FULL history now lives on disk
    /// at `~/.mlx-serve/logs/mlx-serve-<port>.log` (32 MB, rotating); this is
    /// just the live tail, so it only needs to be big enough to read back a
    /// long agent session. `ThrottledLogBuffer` trims with hysteresis, so the
    /// raised cap costs one amortized copy per 256 KB appended, not one scan
    /// per write.
    nonisolated static let serverLogMaxBytes = 1_048_576           // 1 MB retained tail

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

        var args = ["--model", resolvedModel]
        // Plan 05 Phase G — pass the parent directory as --model-dir so the
        // registry discovers all siblings at startup. Lets the user hot-load
        // any sibling later via the model picker (when hot-switch is on).
        let modelDir = (resolvedModel as NSString).deletingLastPathComponent
        args += options.toCLIArgs(modelDirOverride: modelDir.isEmpty ? nil : modelDir)
        launch(args: args, options: options)
    }

    /// Has a headless server already had the selected chat model hot-loaded
    /// by `ensureDefaultChatModel`? Reset on every launch; only consulted for
    /// headless launches (`currentModelPath` empty).
    private var chatDefaultEnsured = false

    /// Should a chat surface hot-load the selected model before its turn?
    /// True exactly when: the server is running, it was launched HEADLESS
    /// (media-first — no `--model`, so the registry has NO default and the
    /// "mlx-serve" alias 503s with no_model), we haven't already ensured it,
    /// and the app actually has a selected model to offer. Pure + static so
    /// the gen-first→chat-later hole (live 2026-07-05) stays unit-pinned.
    nonisolated static func shouldEnsureChatDefault(running: Bool, launchedModelPath: String,
                                                    alreadyEnsured: Bool, selectedModelPath: String) -> Bool {
        running && launchedModelPath.isEmpty && !alreadyEnsured && !selectedModelPath.isEmpty
    }

    /// Called by chat surfaces (chat window / quick launcher via
    /// ChatTurnEngine, the avatar) before a turn: when the running server was
    /// started headless for media generation, hot-load the user's selected
    /// chat model by ABSOLUTE PATH (works for org/name two-level dirs; the
    /// server dedups by path and promotes the first chat-capable load to its
    /// default, so the alias-addressed request that follows resolves).
    /// Failures are left to the request itself to surface.
    func ensureDefaultChatModel(selectedModelPath: String) async {
        guard Self.shouldEnsureChatDefault(running: status == .running,
                                           launchedModelPath: currentModelPath,
                                           alreadyEnsured: chatDefaultEnsured,
                                           selectedModelPath: selectedModelPath) else { return }
        if (try? await loadModel(id: selectedModelPath)) != nil {
            chatDefaultEnsured = true
        }
    }

    /// Start the server HEADLESS — no `--model`, just `--model-dir` so the
    /// registry discovers chat models while media + chat load on demand via
    /// `/v1/load-model`. Used by `ensureRunning(forGenModelDir:)` when the user
    /// generates media before any chat server is up. One process hosts
    /// chat + image + audio + video under one memory budget.
    func startHeadless(modelsDir: String, options: ServerOptions) {
        guard status != .running, status != .starting else { return }
        currentModelPath = ""
        let args = options.toCLIArgs(modelDirOverride: modelsDir.isEmpty ? nil : modelsDir)
        launch(args: args, options: options)
    }

    /// Shared process launch for `start` / `startHeadless`. `args` already
    /// carries `--model`/`--model-dir`; this wires the env, stderr capture,
    /// termination handler, and health polling.
    private func launch(args: [String], options: ServerOptions) {
        port = options.port
        status = .starting
        lastError = ""
        chatDefaultEnsured = false
        clearServerLog()

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

        // Echo the launch command at the top of the log (the buffer was just
        // cleared, so this lands above the server's own first line). Lets the
        // user see and reproduce the exact invocation.
        logBuffer.append(Self.launchCommandLine(binaryPath: binaryPath, args: args) + "\n\n")

        // Capture stderr to show errors in the UI
        let errPipe = Pipe()
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = errPipe

        errPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let str = String(data: data, encoding: .utf8) else { return }
            // Pure off-main append. No `Task @MainActor`, no `DispatchQueue.main`,
            // no `@Published` publish — none of which the inference path can
            // afford. The log views poll `currentServerLogSnapshot()` on their
            // own clock to pick this up.
            self?.logBuffer.append(str)
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

    /// Distill a crashed server's stderr tail into one human-meaningful line for
    /// the menu-bar error. A blind `suffix(N)` lands mid-native-backtrace (e.g.
    /// "…wqthread + 8") and buries the real cause; this surfaces the actual fatal
    /// line — GPU OOM, our own pre-flight refusal, or a recognized error
    /// signature — falling back to the last non-empty line. Pure + testable.
    nonisolated static func summarizeCrash(_ log: String, exitCode: Int32) -> String {
        let trimmed = log.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return "exit code \(exitCode)" }
        let lines = trimmed.split(whereSeparator: \.isNewline).map(String.init)

        // Our own pre-flight already emits a clean, actionable line — prefer it.
        if let l = lines.last(where: { $0.contains("Insufficient memory to load model") }) {
            return l
        }
        // A Metal GPU out-of-memory crash → friendly, actionable message. Match
        // the Metal/IOGPU marker specifically so we don't catch unrelated lines
        // that merely contain "insufficient memory".
        if lines.contains(where: {
            $0.contains("kIOGPUCommandBufferCallbackErrorOutOfMemory")
                || ($0.contains("[METAL]") && $0.localizedCaseInsensitiveContains("Insufficient Memory"))
        }) {
            return "Out of GPU memory while loading the model — free memory (close other models/apps) and try again."
        }
        // Otherwise surface the most meaningful fatal line, scanning from the end.
        let needles = ["terminating due to", "MLX error", "MISSING WEIGHT", "[fatal]", "panic", "error:"]
        for line in lines.reversed() {
            if needles.contains(where: { line.localizedCaseInsensitiveContains($0) }) {
                // Strip the C++ "...uncaught exception of type T: " preamble.
                if let r = line.range(of: "std::runtime_error: ") {
                    return String(line[r.upperBound...])
                }
                return line
            }
        }
        return lines.last ?? "exit code \(exitCode)"
    }

    private func handleTermination(exitCode: Int32) {
        pollSource?.cancel()
        pollSource = nil
        healthTask?.cancel()
        healthTask = nil
        healthTimer?.invalidate()
        healthTimer = nil
        process = nil

        // Read straight from the locked buffer — the throttled `serverLog`
        // can be up to one flush-interval (~100 ms) behind, which on a fast
        // crash means we'd present the alert before the fatal stderr line
        // made it to @Published.
        let errSnippet = currentServerLogSnapshot()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let shortErr = Self.summarizeCrash(errSnippet, exitCode: exitCode)
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

    // MARK: - Server log (throttled)

    /// Read the current raw buffer. Safe from any thread. Use this when you
    /// need the *latest* stderr (e.g. crash reporting, Copy/Save in the log
    /// window); the @Published `serverLog` view-mirror can be up to ~100 ms
    /// behind.
    nonisolated func currentServerLogSnapshot() -> String {
        logBuffer.snapshot()
    }

    /// Wipe the raw buffer. Log views will see the empty result on their
    /// next poll tick (typically within 500 ms). Called from `start()` and
    /// the log window's Clear button.
    nonisolated func clearServerLog() {
        logBuffer.clear()
    }

    /// Pure helper: clamp `buf` to at most `maxBytes` characters by keeping
    /// the tail. Mirrors `String(buf.suffix(maxBytes))` but avoids the
    /// alloc when already in range, and well-defined at the degenerate
    /// `maxBytes == 0` boundary. Lives on the type (not the instance) so
    /// tests can drive it without standing up a real `ServerManager`, and
    /// is `nonisolated` so the off-main `ThrottledLogBuffer` can call it
    /// inside its lock.
    nonisolated static func trimLogTail(_ buf: inout String, toAtMost maxBytes: Int) {
        if maxBytes <= 0 {
            buf = ""
            return
        }
        if buf.count > maxBytes {
            buf = String(buf.suffix(maxBytes))
        }
    }

    /// A copy-pasteable launch command line, shown at the top of the server log
    /// so the user can see (and reproduce) exactly how mlx-serve was invoked.
    /// Every launch knob is a CLI flag (already in `args`); bundle-internal
    /// dylib env (DYLD_LIBRARY_PATH / MLX_METAL_PATH) is intentionally omitted
    /// as per-install noise that isn't reproducible outside the app. Args
    /// containing whitespace (e.g. a model path with spaces) are quoted.
    nonisolated static func launchCommandLine(binaryPath: String, args: [String]) -> String {
        func q(_ s: String) -> String {
            (s.isEmpty || s.contains(where: { $0 == " " || $0 == "\t" })) ? "\"\(s)\"" : s
        }
        return ([q(binaryPath)] + args.map(q)).joined(separator: " ")
    }

    /// Builds the scrollable, selectable, monospaced log view shown in the
    /// crash alert. Lines WRAP to the visible width instead of scrolling
    /// horizontally — a horizontal scroller cuts off the tail of long
    /// backtrace / exception lines, and the real crash cause (e.g.
    /// "…execution failed: Insufficient Memory") sits at the far right where
    /// it's unreachable. Factored out so the wrapping configuration is
    /// unit-testable without running the modal alert.
    @MainActor
    static func makeCrashLogScrollView(log: String, width: CGFloat, height: CGFloat) -> NSScrollView {
        let scroll = NSScrollView(frame: NSRect(x: 0, y: 0, width: width, height: height))
        scroll.hasVerticalScroller = true
        scroll.hasHorizontalScroller = false
        scroll.borderType = .bezelBorder
        scroll.autohidesScrollers = false

        let textView = NSTextView(frame: NSRect(origin: .zero, size: scroll.contentSize))
        textView.isEditable = false
        textView.isSelectable = true
        textView.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        textView.string = log
        // Wrap lines to the visible width instead of horizontal-scrolling.
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude,
                                  height: CGFloat.greatestFiniteMagnitude)
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.containerSize = NSSize(
            width: scroll.contentSize.width,
            height: CGFloat.greatestFiniteMagnitude
        )
        textView.scrollToEndOfDocument(nil)

        scroll.documentView = textView
        return scroll
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
        alert.accessoryView = Self.makeCrashLogScrollView(log: log, width: 640, height: 280)

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
                    // Health handshake done — cancel the 1 Hz health source.
                    // The /props slow-poll is started by transitionToRunning
                    // (only when the menu is open) or by setMenuVisible(true).
                    source.cancel()
                }
            }.resume()
        }
        source.resume()
        self.pollSource = source
    }

    private func startSlowPolling() {
        // Idempotent — cancel any in-flight ticker before installing a new one,
        // so back-to-back calls (e.g. popover-open + transition-to-running
        // racing) can't leave two timers feeding /props at 2× the rate.
        pollSource?.cancel()
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
        // If the user has the menu open at the exact moment the server comes
        // up, start the live /props ticker now so the GPU-memory bar fills in
        // immediately. Otherwise it'll start the next time they open the menu.
        if menuIsVisible {
            Task { await self.refreshStatus() }
            startSlowPolling()
        }
    }

    /// Called by TestServer when it detects health is ok but status is still starting
    func forceRunning() {
        transitionToRunning()
    }

    /// Wire the popover's open/close into the live-polling state. Called
    /// from `StatusMenuView`'s `onAppear`/`onDisappear`. Drives the /props
    /// ticker so it only runs when there's a UI on screen to consume it.
    ///
    /// - When the menu opens while the server is `.running`: immediate
    ///   `refreshStatus()` + start the 3 s ticker.
    /// - When the menu closes while the server is `.running`: cancel the
    ///   ticker.
    /// - During `.starting` / `.stopped`: no-op on close (we'd cancel the
    ///   health-check poll by accident otherwise). On open during `.starting`,
    ///   the health source is already ticking; we just record the visibility
    ///   for when the server transitions.
    func setMenuVisible(_ visible: Bool) {
        guard menuIsVisible != visible else { return }
        menuIsVisible = visible
        guard status == .running else { return }
        if visible {
            Task { await self.refreshStatus() }
            // One-shot registry snapshot so the model slots are fresh when the
            // popover opens (covers loads/unloads made outside this app). The
            // 3 s ticker still deliberately skips /v1/models — see
            // refreshStatus for the rationale.
            Task { await self.refreshModels() }
            startSlowPolling()
        } else {
            pollSource?.cancel()
            pollSource = nil
        }
    }

    private func refreshStatus() async {
        if let mem = try? await api.fetchProps(port: port) {
            memoryInfo = mem
        }
        // Intentionally do NOT poll /v1/models here. The registry snapshot is
        // already populated once on `transitionToRunning` and again after every
        // explicit `loadModel(id:)` the app makes — those are the only events
        // that can change the loaded/unloaded badges from within this app.
        // Hot-load / eviction triggered by a different client (curl, another
        // app) won't be reflected until the next user-initiated load or a
        // restart; that's an acceptable trade for not hitting /v1/models
        // every 3 s for the lifetime of the session.
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

    /// Free a model's resident GPU state (registry keeps the stub). Used by the
    /// media-gen load→generate→unload flow. Idempotent.
    func unloadModel(id: String) async throws {
        try await api.unloadModel(port: port, id: id)
        await refreshModels()
    }

    /// Errors surfaced to the media-gen services.
    enum GenServerError: LocalizedError {
        case binaryMissing(String)
        case modelMissing(String)
        case startFailed(String)
        var errorDescription: String? {
            switch self {
            case .binaryMissing(let p): return "mlx-serve binary not found at \(p)."
            case .modelMissing(let r): return "Model \(r) is not downloaded. Download it first."
            case .startFailed(let s): return "Server did not start: \(s)"
            }
        }
    }

    /// Ensure the main server is running so a media model can load on demand,
    /// then return its port. Reuses an already-running server (so a chat model
    /// and a gen model coexist on ONE process); otherwise starts the server
    /// HEADLESS (no `--model`) against the models root. The caller then
    /// `loadModel(id:)`s the media model by path. `dir` is accepted for symmetry
    /// / future per-model routing; the headless `--model-dir` is the models root
    /// so chat models stay discoverable.
    func ensureRunning(forGenModelDir dir: String) async throws -> UInt16 {
        _ = dir
        if status == .running { return port }
        if status != .starting {
            let modelsRoot = NSString(string: "~/.mlx-serve/models").expandingTildeInPath
            startHeadless(modelsDir: modelsRoot, options: lastLaunchedOptions ?? ServerOptions())
        }
        try await waitUntilRunning(timeout: 240)
        return port
    }

    /// Poll `status` until the health loop flips it to `.running` (or `.error`).
    private func waitUntilRunning(timeout: TimeInterval) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            switch status {
            case .running: return
            case .error(let m): throw GenServerError.startFailed(m)
            default: break
            }
            try? await Task.sleep(nanoseconds: 300_000_000)
        }
        throw GenServerError.startFailed("server did not become ready in time")
    }

    /// Resolve a HuggingFace repo id to its local model directory under
    /// `~/.mlx-serve/models` — the SINGLE source of truth for downloaded
    /// models. No HF-cache fallback: the app owns downloads (chat + media), so
    /// a model the user hasn't downloaded through us simply isn't available.
    /// nil when the dir (with a `config.json`) doesn't exist. The media-gen
    /// services call this before loading.
    static func resolveModelDir(repo: String) -> String? {
        let fm = FileManager.default
        let modelsRoot = NSString(string: "~/.mlx-serve/models").expandingTildeInPath
        if let dir = DownloadManager.existingModelDir(rootDir: modelsRoot, repoId: repo),
           fm.fileExists(atPath: (dir as NSString).appendingPathComponent("config.json")) {
            return dir
        }
        return nil
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

    /// Was `/usr/sbin/lsof -nP -iTCP:<port> -sTCP:LISTEN -t`; now libproc, which
    /// is what lsof itself reads. `lsof` is not reachable from inside the App
    /// Sandbox container. See `SystemMetrics`.
    private func pidsListening(on port: UInt16) -> [pid_t] {
        SystemMetrics.pidsListening(onTCPPort: port)
    }

    /// Was `/bin/ps -p <pid> -o comm=` plus `lastPathComponent`; now
    /// `proc_pidpath`. Note the two can report different *paths* for the same
    /// process (ps reports the launch path, proc_pidpath resolves symlinks) —
    /// the last component, which is all this compares, agrees.
    private func processName(pid: pid_t) -> String {
        SystemMetrics.processName(pid: pid)
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

/// Lock-guarded string buffer that holds the recent server stderr tail.
/// Lives outside `ServerManager`'s `@MainActor` isolation so the stderr
/// readability handler (running on a background `FileHandle` queue) can
/// append without hopping to main on every chunk — that hop was the
/// per-token bottleneck that starved ChatView's SSE loop when the log
/// window was open.
///
/// `@unchecked Sendable` is honest here: the only shared state (`content`)
/// is fully guarded by `lock`. There are no escaping references.
final class ThrottledLogBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var content = ""
    /// Kept incrementally. `String.count` walks grapheme clusters — O(n) — and
    /// at a 1 MB tail that is far too expensive to pay on every stderr chunk.
    private var length = 0
    private let maxBytes: Int

    /// Only re-trim once the buffer exceeds the cap by 25%, then cut back to
    /// the cap. That amortizes the O(n) `suffix` copy over `maxBytes / 4`
    /// appended characters instead of paying it on every append.
    ///
    /// CONTRACT: the buffer always retains AT LEAST the last `maxBytes`
    /// characters and never exceeds `maxRetained`. Trimming down to a
    /// low-water mark instead would keep a strict `maxBytes` ceiling but throw
    /// away a quarter of the tail every cycle — worse for a log.
    var maxRetained: Int { maxBytes + maxBytes / 4 }
    private var highWater: Int { maxRetained }

    init(maxBytes: Int) {
        self.maxBytes = maxBytes
    }

    func append(_ str: String) {
        lock.lock()
        defer { lock.unlock() }
        content.append(str)
        length += str.count
        guard length > highWater else { return }
        ServerManager.trimLogTail(&content, toAtMost: maxBytes)
        length = content.count
    }

    func snapshot() -> String {
        lock.lock()
        defer { lock.unlock() }
        return content
    }

    /// Retained character count, O(1). The log window renders this as a byte
    /// readout every tick — recomputing it from the string would re-walk 1 MB.
    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return length
    }

    func clear() {
        lock.lock()
        defer { lock.unlock() }
        content = ""
        length = 0
    }
}

/// Pull-based bridge from `ThrottledLogBuffer` (off-main, lock-guarded
/// source of truth) to a SwiftUI view that wants to render the log.
///
/// `ServerManager` deliberately does NOT publish the log — see the
/// comment above `logBuffer` for why. Views that want live log content
/// own a `LogPoller` as `@StateObject`, call `start()` on appear and
/// `stop()` on disappear. The view re-renders on each `text` change at
/// its own rate (default ~2 Hz); the rest of the app — ChatView,
/// Settings, the menu popover — is fully insulated from log volume.
@MainActor
final class LogPoller: ObservableObject {
    /// Latest snapshot fetched from the source. Only assigned when
    /// changed, so no-op ticks don't trigger view re-renders.
    @Published private(set) var text: String = ""

    /// `text.count`, computed once per change. `String.count` walks grapheme
    /// clusters (O(n)); reading it straight from the view body would re-walk
    /// the whole 1 MB tail on every render, twice a second.
    @Published private(set) var characterCount: Int = 0

    private var timer: DispatchSourceTimer?
    private let interval: TimeInterval
    private var snapshot: () -> String

    /// - Parameters:
    ///   - interval: poll interval in seconds. 0.5 (2 Hz) is the default —
    ///     smooth-enough for human reading, cheap for the main thread.
    ///   - snapshot: closure returning the latest content. Injected so
    ///     tests can drive deterministically without spinning up a real
    ///     `ServerManager`. In production this is
    ///     `{ [weak server] in server?.currentServerLogSnapshot() ?? "" }`.
    init(interval: TimeInterval = 0.5, snapshot: @escaping () -> String) {
        self.interval = interval
        self.snapshot = snapshot
    }

    /// Begin periodic polling. Safe to call repeatedly — cancels any
    /// previous timer first. Fires one immediate `refresh()` so the view
    /// shows content without waiting an interval.
    func start() {
        stop()
        refresh()
        let t = DispatchSource.makeTimerSource(queue: .main)
        t.schedule(deadline: .now() + interval, repeating: interval)
        t.setEventHandler { [weak self] in self?.refresh() }
        t.resume()
        timer = t
    }

    /// Cancel polling. Idempotent.
    func stop() {
        timer?.cancel()
        timer = nil
    }

    /// Swap the snapshot source. Useful for `@StateObject` SwiftUI views
    /// that can't pass an environment-injected dependency through the
    /// `init` closure — they construct the poller with a placeholder, then
    /// `bind` to the real source on `onAppear`.
    func bind(_ snapshot: @escaping () -> String) {
        self.snapshot = snapshot
    }

    /// Pull the current snapshot. Only assigns `text` when the value
    /// actually changed, so unchanged ticks cost one closure call and a
    /// string compare — no SwiftUI work.
    func refresh() {
        let new = snapshot()
        guard new != text else { return }
        text = new
        characterCount = new.count
    }

    deinit {
        // Cancel without touching `self.timer` (deinit is nonisolated).
        timer?.cancel()
    }
}
