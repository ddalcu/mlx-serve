import Foundation

/// Drives Settings > Server > Plugins > Manage: loads the catalog, and installs a plugin in two steps so the user
/// confirms what will be compiled. (1) `prepare` fetches the repo at `PluginCatalog.ref` into
/// ~/.mlx-serve/plugin-src and reports the exact commit; (2) `confirmInstall` runs the repo's
/// `scripts/build.sh <name>` (the pinned Zig, the plugin's tests) and then this app's own
/// `mlx-serve plugin install <dylib> --yes` (hash + signature receipt): `installSteps`.
/// Nothing prebuilt is ever downloaded or run.
@MainActor
final class PluginManager: ObservableObject {
    enum Phase: Equatable {
        case idle
        case fetching(name: String)
        /// Source is on disk at `commit`; waiting for the user to accept it.
        case awaitingConfirmation(name: String, commit: String, version: String)
        case building(name: String)
        case done(name: String)
        case failed(name: String, message: String)
    }

    @Published private(set) var catalog: [PluginCatalog.Entry] = []
    @Published private(set) var catalogError: String?
    @Published private(set) var loadingCatalog = false
    @Published private(set) var receipts: [OutsidePlugins.Receipt] = OutsidePlugins.receipts()
    @Published private(set) var phase: Phase = .idle
    @Published private(set) var log = ""
    @Published private(set) var runningVersion: String?
    /// What the running server said of each installed plugin at its start (`/props`); empty when it is not running.
    @Published private(set) var serverStatus: [String: PluginCatalog.ServerStatus] = [:]
    /// The plugins the install in progress builds, in order (one, or every stale one for Rebuild all).
    @Published private(set) var pending: [String] = []

    static let sourceDir = NSString(string: "~/.mlx-serve/plugin-src").expandingTildeInPath
    private var process: Process?

    var isBusy: Bool {
        switch phase {
        case .fetching, .building: return true
        default: return false
        }
    }

    func refreshReceipts() { receipts = OutsidePlugins.receipts() }

    // MARK: Catalog

    func loadCatalog(port: UInt16? = nil) async {
        loadingCatalog = true
        defer { loadingCatalog = false }
        async let version = Self.probeRunningVersion()
        if let port { serverStatus = await Self.fetchServerStatus(port: port) }
        do {
            var request = URLRequest(url: PluginCatalog.catalogURL)
            request.cachePolicy = .reloadIgnoringLocalCacheData
            request.timeoutInterval = 20
            let (data, response) = try await URLSession.shared.data(for: request)
            if let code = (response as? HTTPURLResponse)?.statusCode, code != 200 {
                throw URLError(.badServerResponse, userInfo: [NSLocalizedDescriptionKey: "HTTP \(code)"])
            }
            catalog = try PluginCatalog.decode(data)
            catalogError = nil
        } catch {
            catalogError = error.localizedDescription
        }
        runningVersion = await version
        refreshReceipts()
    }

    nonisolated private static func fetchServerStatus(port: UInt16) async -> [String: PluginCatalog.ServerStatus] {
        guard let url = URL(string: "http://127.0.0.1:\(port)/props") else { return [:] }
        var request = URLRequest(url: url)
        request.timeoutInterval = 3
        guard let (data, _) = try? await URLSession.shared.data(for: request) else { return [:] }
        return PluginCatalog.serverStatus(fromProps: data)
    }

    nonisolated private static func probeRunningVersion() async -> String? {
        await Task.detached {
            let out = runCapture(Self.binaryPath(), ["--version"], env: ServerManager.launchEnvironment())
            return PluginCatalog.runningVersion(fromVersionOutput: out)
        }.value
    }

    // MARK: Install

    /// Step 1: fetch the source and name the commit. Nothing is compiled or run from it yet.
    func prepare(_ entry: PluginCatalog.Entry) { prepare([entry]) }

    /// Several at once (Rebuild all): one fetch, one confirmation, then each built and installed in turn.
    func prepare(_ entries: [PluginCatalog.Entry]) {
        guard !isBusy, !entries.isEmpty, entries.allSatisfy({ PluginCatalog.isSafeName($0.name) }) else { return }
        log = ""
        pending = entries.map(\.name)
        let name = pending.joined(separator: ", ")
        phase = .fetching(name: name)
        Task {
            do {
                try Self.requireDeveloperTools()
                let commit = try await fetchSource()
                for n in pending where !FileManager.default.fileExists(atPath: Self.sourceDir + "/plugins/\(n)/plugin.json") {
                    throw Failure("\(n) is not in the repository at \(commit.prefix(10))")
                }
                phase = .awaitingConfirmation(name: name, commit: commit, version: entries.count == 1 ? entries[0].version : "")
            } catch {
                phase = .failed(name: name, message: error.localizedDescription)
            }
        }
    }

    /// The plugin's own `plugin.json`, shown in the confirmation sheet beside the commit.
    func manifestText(for name: String) -> String {
        guard PluginCatalog.isSafeName(name),
              let text = try? String(contentsOfFile: Self.sourceDir + "/plugins/\(name)/plugin.json", encoding: .utf8) else { return "" }
        return text
    }

    /// Step 2: the user accepted `commit`. Build and install exactly the tree `prepare` fetched.
    func confirmInstall() {
        guard case let .awaitingConfirmation(name, commit, _) = phase else { return }
        phase = .building(name: name)
        Task {
            do {
                // The checkout must still be the commit the user accepted.
                let head = try await Self.git(["rev-parse", "HEAD"], in: Self.sourceDir).trimmingCharacters(in: .whitespacesAndNewlines)
                guard head == commit else { throw Failure("the source changed after you reviewed it; start again") }
                var env = ServerManager.launchEnvironment()
                let binDir = (Self.binaryPath() as NSString).deletingLastPathComponent
                env["PATH"] = [binDir, "/usr/bin", "/bin", "/usr/sbin", "/sbin", env["PATH"] ?? ""].filter { !$0.isEmpty }.joined(separator: ":")
                if let kit = try await kitForRunningServer() {
                    env["MLX_SERVE_KIT"] = kit
                    // The repo's fetch-kit keeps a staged kit of its own KIT_VERSION over MLX_SERVE_KIT.
                    try? FileManager.default.removeItem(atPath: Self.sourceDir + "/.kit")
                }
                for n in pending {
                    for step in Self.installSteps(name: n, sourceDir: Self.sourceDir, binary: Self.binaryPath()) {
                        let status = try await stream(step.exe, step.args, cwd: Self.sourceDir, env: env)
                        guard status == 0 else { throw Failure(PluginCatalog.installFailure(status: status, log: log, running: runningVersion)) }
                    }
                    serverStatus[n] = nil
                }
                refreshReceipts()
                phase = .done(name: name)
            } catch {
                phase = .failed(name: name, message: error.localizedDescription)
            }
        }
    }

    func cancel() {
        process?.terminate()
        if case let .awaitingConfirmation(name, _, _) = phase { phase = .failed(name: name, message: "Cancelled") }
    }

    func dismiss() {
        guard !isBusy else { return }
        phase = .idle
        pending = []
    }

    func uninstall(_ name: String) {
        do {
            try PluginCatalog.uninstall(name)
            refreshReceipts()
        } catch {
            phase = .failed(name: name, message: error.localizedDescription)
        }
    }

    // MARK: Plumbing

    private struct Failure: LocalizedError {
        var message: String
        init(_ message: String) { self.message = message }
        var errorDescription: String? { message }
    }

    nonisolated static func binaryPath() -> String { ServerManager.resolveBinaryPath() }

    /// The two processes an install runs, in order: the repo's build (its pinned Zig and kit, the plugin's own tests,
    /// a ReleaseFast dylib), then this app's mlx-serve copying that dylib in, confirmed because the user just accepted
    /// the commit. `name` is a `isSafeName` identifier.
    nonisolated static func installSteps(name: String, sourceDir: String, binary: String) -> [(exe: String, args: [String])] {
        [("/bin/bash", [sourceDir + "/scripts/build.sh", name]),
         (binary, ["plugin", "install", "\(sourceDir)/plugins/\(name)/zig-out/lib/lib\(name).dylib", "--yes"])]
    }

    /// The kit of the running release in ~/.mlx-serve/kit, downloaded from that release's assets when the folder
    /// holds another; nil when no release carries one (a dev build), and the build fetches the repo's own.
    private func kitForRunningServer() async throws -> String? {
        let home = PluginCatalog.homeKitDir
        if let kit = PluginCatalog.kit(at: home, matching: runningVersion) { return kit }
        guard let version = runningVersion, let url = PluginCatalog.kitURL(version: version) else { return nil }
        appendLog("$ fetch \(url.absoluteString)\n")
        guard let (file, response) = try? await URLSession.shared.download(from: url),
              (response as? HTTPURLResponse)?.statusCode == 200 else { appendLog("no kit released for \(version)\n"); return nil }
        let fm = FileManager.default
        let staging = (home as NSString).deletingLastPathComponent + "/.kit-staging"
        try? fm.removeItem(atPath: staging)
        try fm.createDirectory(atPath: staging, withIntermediateDirectories: true)
        defer { try? fm.removeItem(atPath: staging) }
        guard try await stream("/usr/bin/tar", ["-xzf", file.path, "-C", staging], cwd: nil, env: ["PATH": "/usr/bin:/bin"]) == 0,
              PluginCatalog.kit(at: staging + "/mlx-serve-plugin-kit-\(version)", matching: version) != nil
        else { throw Failure("the kit for \(version) did not unpack; see the log") }
        try? fm.removeItem(atPath: home)
        try fm.moveItem(atPath: staging + "/mlx-serve-plugin-kit-\(version)", toPath: home)
        return home
    }

    /// git and the Zig build both need the Command Line Tools; say so by name instead of letting the shim pop a dialog.
    private static func requireDeveloperTools() throws {
        guard !runCapture("/usr/bin/xcode-select", ["-p"], env: nil).isEmpty else {
            throw Failure("Install the Xcode Command Line Tools first: run `xcode-select --install` in Terminal.")
        }
    }

    /// Shallow-fetches `PluginCatalog.ref` into `sourceDir` (a clone the first time) and returns the commit.
    private func fetchSource() async throws -> String {
        let dir = Self.sourceDir
        let fm = FileManager.default
        let env = ["GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin"]
        if !fm.fileExists(atPath: dir + "/.git") {
            try fm.createDirectory(atPath: (dir as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
            try? fm.removeItem(atPath: dir)
            guard try await stream("/usr/bin/git", ["clone", "--depth", "1", "--branch", PluginCatalog.ref, PluginCatalog.repoURL, dir],
                                   cwd: nil, env: env) == 0 else { throw Failure("git clone failed; see the log") }
        } else {
            guard try await stream("/usr/bin/git", ["-C", dir, "fetch", "--depth", "1", "origin", PluginCatalog.ref], cwd: nil, env: env) == 0,
                  try await stream("/usr/bin/git", ["-C", dir, "reset", "--hard", "FETCH_HEAD"], cwd: nil, env: env) == 0
            else { throw Failure("git fetch failed; see the log") }
        }
        let commit = try await Self.git(["rev-parse", "HEAD"], in: dir).trimmingCharacters(in: .whitespacesAndNewlines)
        guard commit.count == 40, commit.allSatisfy(\.isHexDigit) else { throw Failure("could not read the fetched commit") }
        return commit
    }

    nonisolated private static func git(_ args: [String], in dir: String) async throws -> String {
        await Task.detached { runCapture("/usr/bin/git", ["-C", dir] + args, env: ["PATH": "/usr/bin:/bin"]) }.value
    }

    /// Runs a process, appending its output to `log` as it arrives; returns the exit status.
    private func stream(_ exe: String, _ args: [String], cwd: String?, env: [String: String]) async throws -> Int32 {
        log += "$ \((exe as NSString).lastPathComponent) \(args.joined(separator: " "))\n"
        let p = Process()
        p.executableURL = URL(fileURLWithPath: exe)
        p.arguments = args
        p.environment = env
        if let cwd { p.currentDirectoryURL = URL(fileURLWithPath: cwd) }
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in self?.appendLog(text) }
        }
        process = p
        defer { process = nil; pipe.fileHandleForReading.readabilityHandler = nil }
        return try await withCheckedThrowingContinuation { cont in
            p.terminationHandler = { cont.resume(returning: $0.terminationStatus) }
            do { try p.run() } catch { p.terminationHandler = nil; cont.resume(throwing: error) }
        }
    }

    private func appendLog(_ text: String) {
        log += text
        if log.count > 200_000 { log = String(log.suffix(150_000)) }
    }

    nonisolated private static func runCapture(_ exe: String, _ args: [String], env: [String: String]?) -> String {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: exe)
        p.arguments = args
        if let env { p.environment = env }
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = Pipe()
        guard (try? p.run()) != nil else { return "" }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        return String(data: data, encoding: .utf8) ?? ""
    }
}
