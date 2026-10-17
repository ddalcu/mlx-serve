import AppKit

/// `mlx-serve worker` while this Mac is a cluster worker: it waits for the host,
/// loads its part of the host's model, and comes back after the host stops it.
@MainActor
final class ClusterWorker: ObservableObject {
    enum State: Equatable {
        case stopped
        case waiting
        case loading(String)
        case ready(String)
        case failed(String)
    }

    @Published private(set) var state: State = .stopped
    private var process: Process?
    private var wanted = false
    private var options = ServerOptions()

    init() {
        // A child process outlives the app unless it is stopped on the way out.
        NotificationCenter.default.addObserver(forName: NSApplication.willTerminateNotification,
                                               object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.process?.terminate() }
        }
    }

    /// The worker's state after one of its log lines.
    nonisolated static func state(after line: String, current: State) -> State {
        if line.contains("[dist] worker ready") {
            if case .loading(let what) = current { return .ready(what) }
            return .ready("")
        }
        if line.contains("[dist] worker: ") && (line.contains("(pipeline)") || line.contains("(tensor)") || line.contains("(video steps)")) {
            let what = line.components(separatedBy: "[dist] worker: ").last?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return .loading(what)
        }
        if line.contains("not listening yet") || line.contains("waiting for the host") { return .waiting }
        if line.contains("worker load failed") || line.contains("worker: no model named")
            || line.contains("different geometry") || line.contains("JACCL is unavailable") {
            let why = line.components(separatedBy: "] ").last?.trimmingCharacters(in: .whitespacesAndNewlines) ?? line
            return .failed(why)
        }
        return current
    }

    /// Runs the worker while `options` make this Mac one; stops it otherwise.
    func reconcile(_ options: ServerOptions) {
        let changed = options.clusterDevice != self.options.clusterDevice
            || options.clusterCoordinator != self.options.clusterCoordinator
        self.options = options
        wanted = options.clusterRole == .worker
        if !wanted || changed { stop() }
        if wanted, process == nil { launch() }
    }

    private func stop() {
        process?.terminationHandler = nil
        process?.terminate()
        process = nil
        state = .stopped
    }

    private func launch() {
        let dirs = Array(ModelRoots().scanRoots(toolRoots: ToolModelRoots.detected()).prefix(ModelRoots.serverRootLimit))
        guard let args = ClusterSetup.workerArgs(options, modelDirs: dirs) else {
            state = .failed(L10n.text("No Thunderbolt RDMA port is up. Connect the cable and turn RDMA on."))
            return
        }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: ServerManager.resolveBinaryPath())
        p.arguments = args
        p.environment = ServerManager.launchEnvironment()
        p.standardOutput = FileHandle.nullDevice
        let err = Pipe()
        p.standardError = err
        err.fileHandleForReading.readabilityHandler = { [weak self] h in
            let text = String(decoding: h.availableData, as: UTF8.self)
            guard !text.isEmpty else { return }
            Task { @MainActor in
                guard let self else { return }
                for line in text.split(separator: "\n") {
                    self.state = Self.state(after: String(line), current: self.state)
                }
            }
        }
        // The worker exits when its host stops it; it waits for the next host.
        p.terminationHandler = { [weak self] _ in
            Task { @MainActor in
                guard let self, self.process === p else { return }
                self.process = nil
                if case .failed = self.state {} else { self.state = .stopped }
                guard self.wanted else { return }
                try? await Task.sleep(for: .seconds(3))
                if self.wanted, self.process == nil { self.launch() }
            }
        }
        do {
            try p.run()
            process = p
            state = .waiting
        } catch {
            state = .failed(error.localizedDescription)
        }
    }
}
