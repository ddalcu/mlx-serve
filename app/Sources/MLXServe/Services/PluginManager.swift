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

    func loadCatalog() async {
        loadingCatalog = true
        defer { loadingCatalog = false }
        async let version = Self.probeRunningVersion()
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

    nonisolated private static func probeRunningVersion() async -> String? {
        await Task.detached {
            let out = runCapture(Self.binaryPath(), ["--version"], env: ServerManager.launchEnvironment())
            return PluginCatalog.runningVersion(fromVersionOutput: out)
        }.value
    }

    // MARK: Install

    /// Step 1: fetch the source and name the commit. Nothing is compiled or run from it yet.
    func prepare(_ entry: PluginCatalog.Entry) {
        guard !isBusy, PluginCatalog.isSafeName(entry.name) else { return }
        log = ""
        phase = .fetching(name: entry.name)
        let name = entry.name
        Task {
            do {
                try Self.requireDeveloperTools()
                let commit = try await fetchSource()
                guard FileManager.default.fileExists(atPath: Self.sourceDir + "/plugins/\(name)/plugin.json") else {
                    throw Failure("\(name) is not in the repository at \(commit.prefix(10))")
                }
                phase = .awaitingConfirmation(name: name, commit: commit, version: entry.version)
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
                if let kit = PluginCatalog.kit(at: PluginCatalog.homeKitDir, matching: runningVersion) { env["MLX_SERVE_KIT"] = kit }
                for step in Self.installSteps(name: name, sourceDir: Self.sourceDir, binary: Self.binaryPath()) {
                    let status = try await stream(step.exe, step.args, cwd: Self.sourceDir, env: env)
                    guard status == 0 else { throw Failure(PluginCatalog.installFailure(status: status, log: log, running: runningVersion)) }
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

    func dismiss() { if !isBusy { phase = .idle } }

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
