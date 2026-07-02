import Foundation

/// Shared manager that routes the agent's shell commands into an isolated Linux
/// guest (Apple Virtualization.framework — see `VzGuest`) instead of the host.
/// Off by default; enabled from Settings via `configure`. `ShellHandler`
/// consults `AgentSandbox.shared` on each command.
///
/// Model (v1): ONE warm guest for the app, booted lazily on the first sandboxed
/// command. The command's working directory at boot time becomes the shared root
/// (virtiofs-mounted at `/workspace`); later commands `cd` into their mapped
/// subpath. A cwd outside the shared root isn't visible in the guest (runs at
/// /workspace). Background commands (`&` / run_in_background) are sandboxed
/// too: `ShellHandler` routes them into the guest (backgrounded INSIDE the
/// guest shell, output to a per-invocation /tmp log) — never to the host
/// ProcessRegistry while the sandbox is on.
///
/// Provisioning (all Swift, no external tools beyond bsdtar):
///  - kernel: contain's prebuilt minimal arm64 kernel, fetched once from its
///    GitHub kernels release and gunzip'd into the cache. A cached kernel that
///    predates virtiofs support is detected (`kernelHasVirtiofsSupport`) and
///    re-fetched — the guest root rides on virtiofs, so an old kernel can't boot.
///  - rootfs: the base OCI image, pulled anonymously + unpacked by `OCIClient`
///    into the cache; the image's Env/WorkingDir ride along in a sidecar JSON.
final class AgentSandbox: ObservableObject, @unchecked Sendable {
    static let shared = AgentSandbox()

    /// One executed command in the guest — agent-issued or typed by the user in
    /// the Sandbox Terminal. Both flow through the same shell, so the transcript
    /// is a unified live view of everything happening in the sandbox.
    struct Entry: Identifiable, Equatable {
        enum Source: String { case agent, user, system }
        let id = UUID()
        let source: Source
        let command: String
        var output: String
        var exitCode: Int32?      // nil = still running / timed out
        var timedOut: Bool = false
        let at: Date
    }

    /// Live transcript (capped) — observed by the Sandbox Terminal window.
    /// Kept in a SEPARATE ObservableObject so per-command appends don't publish
    /// through `AgentSandbox` itself: the tray menu observes that, and a
    /// @Published append there re-renders the whole tray on every command
    /// (documented MenuBarExtra churn class). `AgentSandbox`'s own @Published
    /// is limited to the coarse state the tray actually needs (`guestRunning`).
    let transcriptStore = SandboxTranscript()
    /// Convenience read for non-UI callers (e.g. the smoke harness).
    var transcript: [Entry] { transcriptStore.entries }
    /// Whether a guest is currently booted (for the tray/terminal status).
    @Published private(set) var guestRunning = false
    /// Coarse guest RAM readout for the tray ("384 MB / 987 MB RAM"), fed by
    /// the guest's once-a-second /proc/meminfo report. QUANTIZED to 16 MB steps
    /// and published only when the string changes, so the per-second snapshots
    /// don't re-render the tray (the MenuBarExtra churn class). nil while no
    /// guest is running.
    @Published private(set) var guestMemoryText: String?

    private let lock = NSLock()
    private var enabled = false
    /// Seeded from the settings model's default (single source of truth —
    /// pinned by ServerOptionsTests); AppState overwrites it via `configure()`
    /// at launch and on every settings change.
    private var baseImage = ServerOptions.SandboxConfig().baseImage
    /// Guest networking + live port mapping (see SandboxConfig.network).
    private var networkEnabled = ServerOptions.SandboxConfig().network

    /// Append to the transcript on the main thread (@Published must mutate there).
    private func record(_ entry: Entry) {
        DispatchQueue.main.async {
            self.transcriptStore.append(entry)
        }
    }
    private func setGuestRunning(_ v: Bool) {
        DispatchQueue.main.async {
            if self.guestRunning != v { self.guestRunning = v }
            if !v, self.guestMemoryText != nil { self.guestMemoryText = nil }
        }
    }
    private func setGuestMemoryText(_ v: String?) {
        DispatchQueue.main.async { if self.guestMemoryText != v { self.guestMemoryText = v } }
    }

    /// Tray RAM readout: used (total − available) quantized to 16 MB so nearby
    /// readings map to the SAME string (no per-second tray re-render); totals
    /// ≥ 1000 MB read in GB with one decimal.
    static func memoryDisplayText(availableKB: Int, totalKB: Int) -> String {
        let usedMB = Double(max(0, totalKB - availableKB)) / 1024.0
        let quantized = Int((usedMB / 16.0).rounded()) * 16
        let totalMB = Double(totalKB) / 1024.0
        let totalText = totalMB >= 1000
            ? String(format: "%.1f GB", Double(totalKB) / (1024.0 * 1024.0))
            : "\(Int(totalMB.rounded())) MB"
        return "\(quantized) MB / \(totalText) RAM"
    }

    // Live guest + the host dir it shares at /workspace. Guarded by `bootLock`.
    private let bootLock = NSLock()
    private var guest: VzGuest?
    private var sharedRoot: String?
    /// Host side of the live guest→host port map (created per boot when
    /// networking is on; fed by the guest's hvc2 net-report stream).
    private var forwarder: SandboxPortForwarder?

    private init() {}

    // MARK: Configuration (from Settings)

    var isEnabled: Bool { lock.lock(); defer { lock.unlock() }; return enabled }

    /// Apply the Settings values. Turning the sandbox off, or changing the base
    /// image or the network mode, tears down any live guest so the next command
    /// re-provisions with the new configuration.
    func configure(enabled: Bool, baseImage: String, network: Bool = ServerOptions.SandboxConfig().network) {
        let trimmed = baseImage.trimmingCharacters(in: .whitespacesAndNewlines)
        lock.lock()
        let imageChanged = trimmed != self.baseImage && !trimmed.isEmpty
        let networkChanged = network != self.networkEnabled
        let wasEnabled = self.enabled
        self.enabled = enabled
        self.networkEnabled = network
        if !trimmed.isEmpty { self.baseImage = trimmed }
        lock.unlock()
        if (!enabled && wasEnabled) || (enabled && (imageChanged || networkChanged)) {
            teardown()
        }
    }

    /// Free the live guest (if any). Next sandboxed command re-boots.
    ///
    /// Returns immediately: call sites are main-thread (`AppState.serverOptions.
    /// didSet` fires per keystroke while editing the Settings base-image field;
    /// the Sandbox Terminal Stop button), and `VzGuest.shutdown` blocks up to
    /// 10s on the VZ stop semaphore. The guest reference is detached
    /// SYNCHRONOUSLY (so UI state and a fresh `ensureBooted` are correct at
    /// once — the old VzGuest instance is independent and keeps stopping in the
    /// background) and the blocking stop runs on a background queue.
    func teardown() {
        teardown(shutdownBlocking: { $0.shutdown() })
    }

    /// Seam for tests: `shutdownBlocking` is the (potentially slow) stop that
    /// must run off the calling thread. `bootLock` is NOT held across the async
    /// hop, so a concurrent `ensureBooted` can boot a fresh guest safely.
    func teardown(shutdownBlocking: @escaping (VzGuest) -> Void) {
        bootLock.lock()
        let g = guest
        let fwd = forwarder
        guest = nil
        sharedRoot = nil
        forwarder = nil
        bootLock.unlock()
        fwd?.stop()
        setGuestRunning(false)
        guard let g else { return }
        DispatchQueue.global(qos: .utility).async { shutdownBlocking(g) }
    }

    // MARK: Test hooks (no VM — install/inspect the detached guest reference)

    func _testInstallGuest(_ g: VzGuest?) {
        bootLock.lock(); guest = g; bootLock.unlock()
    }
    var _testHasGuest: Bool {
        bootLock.lock(); defer { bootLock.unlock() }; return guest != nil
    }

    // MARK: Pure helpers (unit-tested)

    /// Filesystem-safe cache dir name for an OCI image ref (no `/`, `:`, or `..`).
    static func imageDirName(_ image: String) -> String {
        var out = ""
        for ch in image {
            if ch == "/" || ch == ":" || ch == " " { out.append("_") }
            else { out.append(ch) }
        }
        return out.replacingOccurrences(of: "..", with: "__")
    }

    /// Map a host working directory into the guest's `/workspace` share.
    /// Returns the guest path and whether the host path was actually under the
    /// shared root (false → not visible; command runs at `/workspace`).
    static func guestPath(hostPath: String?, sharedRoot: String) -> (path: String, mapped: Bool) {
        guard let hostPath else { return ("/workspace", false) }
        let host = (hostPath as NSString).standardizingPath
        let root = (sharedRoot as NSString).standardizingPath
        if host == root { return ("/workspace", true) }
        // Segment-boundary check so /a/proj2 isn't treated as under /a/proj.
        let rootWithSep = root.hasSuffix("/") ? root : root + "/"
        if host.hasPrefix(rootWithSep) {
            let rel = String(host.dropFirst(rootWithSep.count))
            return (rel.isEmpty ? "/workspace" : "/workspace/" + rel, true)
        }
        return ("/workspace", false)
    }

    /// Wrap a command so it runs in the mapped guest dir. `cd` failure is ignored
    /// (2>/dev/null) so `$?` reflects the command, not the cd. The cwd is
    /// shell-quoted: an unescaped apostrophe in a folder name would unbalance
    /// the quoting and desync the ShellSentinel framing (every command then
    /// times out).
    static func wrap(command: String, guestCwd: String) -> String {
        "cd \(VzGuest.shellQuote(guestCwd)) 2>/dev/null; \(command)"
    }

    /// The host dir shared into the guest when no working directory was
    /// supplied (e.g. the Sandbox Terminal boots the guest before any agent
    /// command): the app's default session workspace — the same
    /// `~/.mlx-serve/workspace` every chat session starts in. NEVER the user's
    /// home folder or the process cwd (`/` for a Finder-launched app); mounting
    /// those rw into the guest defeats the point of the sandbox.
    static func fallbackSharedRoot() -> String {
        ChatSession.defaultWorkingDirectory
    }

    /// True when a live guest's `/workspace` share no longer covers the
    /// requested working directory — i.e. the user switched the session's
    /// working folder to somewhere outside the shared root. The guest then
    /// reboots with the new share (sub-second on virtiofs) so `/workspace`
    /// stays in sync with the folder shown in the chat toolbar.
    static func needsRemount(sharedRoot: String, requestedCwd: String?) -> Bool {
        guard let requestedCwd else { return false }
        return !guestPath(hostPath: requestedCwd, sharedRoot: sharedRoot).mapped
    }

    /// Guest CPU architecture. The Virtualization.framework guest on Apple
    /// Silicon is ALWAYS arm64, so we must pull the arm64 image variant
    /// explicitly — letting the registry default to amd64 boots to an ENOEXEC
    /// panic (`/bin/sh exists but couldn't execute it`, "No working init").
    static let guestArch = "arm64"

    /// Marker filename recording the arch a cached rootfs was pulled for, so an
    /// old wrong-arch cache (which still has bin/sh) forces a re-pull. (The name
    /// keeps the historical `contain` prefix so caches pulled by the previous
    /// libcontain integration stay recognizable.)
    static func archMarkerName() -> String { ".contain-arch-\(guestArch)" }

    /// A guest kernel must have virtio-fs compiled in (the rootfs is served over
    /// it). The registered filesystem name appears as a literal in the binary;
    /// kernels from before the config landed don't contain it — those caches are
    /// stale and must be re-fetched.
    static func kernelHasVirtiofsSupport(_ kernelData: Data) -> Bool {
        kernelData.range(of: Data("virtiofs".utf8)) != nil
    }

    // MARK: Execution

    struct SandboxError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    // MARK: Sandbox background process control (guest-backed handles)

    /// Guest-side command that stops process `pid`: SIGTERM, a brief grace, then
    /// SIGKILL any survivor. The grace kill is backgrounded inside the guest (same
    /// `(…) &` shape the background-launch wrapper uses) so the exec returns at
    /// once. Pure + testable — the routing decision needs no live VM.
    static func guestKillCommand(pid: Int32) -> String {
        "kill -TERM \(pid) 2>/dev/null; (sleep 3; kill -KILL \(pid) 2>/dev/null) &"
    }

    /// Guest-side command that tails a background log (bounded to `maxBytes`). The
    /// path is shell-quoted for the same reason `wrap` quotes the cwd.
    static func guestReadLogCommand(logPath: String, maxBytes: Int = 65_536) -> String {
        "tail -c \(maxBytes) \(VzGuest.shellQuote(logPath)) 2>/dev/null"
    }

    /// Stop a sandbox background process by its GUEST pid, inside the
    /// ALREADY-BOOTED guest. NEVER boots one — if the guest is gone the process is
    /// already dead, so there's nothing to kill. Called off the main thread by
    /// `ProcessRegistry.kill` (the guest exec blocks).
    func killGuestProcess(pid: Int32) {
        guard pid > 0 else { return }
        let g: VzGuest? = { bootLock.lock(); defer { bootLock.unlock() }; return guest }()
        guard let g, !g.isFinished else { return }
        let cmd = Self.guestKillCommand(pid: pid)
        _ = try? g.exec(cmd, timeout: 10)
        record(Entry(source: .system, command: cmd, output: "", exitCode: 0, at: Date()))
    }

    /// Read a sandbox background process's log from the ALREADY-BOOTED guest (never
    /// boots one). Returns nil when no live guest exists — the caller then reports
    /// the handle's status without log contents. Runs the blocking guest exec off
    /// the main thread.
    func tailGuestLog(logPath: String, timeout: Double = 15) async -> String? {
        await withCheckedContinuation { (cont: CheckedContinuation<String?, Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                // Read the live guest INSIDE the hop (never boots one) so the
                // non-Sendable VzGuest is never captured across the closure.
                let g: VzGuest? = { self.bootLock.lock(); defer { self.bootLock.unlock() }; return self.guest }()
                guard let g, !g.isFinished else { cont.resume(returning: nil); return }
                if let r = try? g.exec(Self.guestReadLogCommand(logPath: logPath), timeout: timeout) {
                    cont.resume(returning: TerminalOutput.sanitize(r.output))
                } else {
                    cont.resume(returning: nil)
                }
            }
        }
    }

    /// Run a foreground command in the sandbox. Boots + provisions lazily on first
    /// use. Returns output formatted like the host shell path (`ShellMessages`).
    /// Throws `SandboxError` when the guest can't be provisioned/booted — we do
    /// NOT silently fall back to the host, since the user enabled isolation on
    /// purpose.
    func runForeground(command: String, workingDirectory: String?, timeout: Double) async throws -> String {
        let image = { lock.lock(); defer { lock.unlock() }; return baseImage }()
        let hostCwd = workingDirectory ?? Self.fallbackSharedRoot()

        return try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let (g, root) = try self.ensureBooted(image: image, workingDirectory: workingDirectory)
                    let gp = Self.guestPath(hostPath: workingDirectory ?? hostCwd, sharedRoot: root)
                    let wrapped = Self.wrap(command: command, guestCwd: gp.path)
                    let r = try g.exec(wrapped, timeout: timeout)
                    // hvc1 is a tty → tools emit ANSI color + progress animations;
                    // strip them so the terminal AND the agent see clean text.
                    let cleaned = TerminalOutput.sanitize(r.output)
                    self.record(Entry(source: .agent, command: command, output: cleaned,
                                      exitCode: r.timedOut ? nil : r.exitCode,
                                      timedOut: r.timedOut, at: Date()))
                    var body = cleaned
                    if !gp.mapped {
                        body = "[sandbox: cwd not under the shared folder; ran in /workspace]\n" + body
                    }
                    if r.timedOut {
                        cont.resume(returning: ShellMessages.timedOutKilled(
                            cwd: hostCwd, seconds: Int(timeout), body: body))
                    } else {
                        cont.resume(returning: ShellMessages.completed(
                            cwd: hostCwd, body: body, exitCode: r.exitCode))
                    }
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Run a command typed by the user in the Sandbox Terminal. Same guest/shell
    /// as the agent (state is shared), recorded in the transcript. Unlike the
    /// agent path it does NOT cd-wrap, so an interactive `cd` sticks. Never
    /// throws — provisioning/boot errors are recorded as a `.system` entry.
    func runUserCommand(_ command: String, timeout: Double = 60) async {
        let trimmed = command.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let image = { lock.lock(); defer { lock.unlock() }; return baseImage }()
        let root: String = {
            bootLock.lock(); defer { bootLock.unlock() }
            return sharedRoot ?? Self.fallbackSharedRoot()
        }()
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let (g, _) = try self.ensureBooted(image: image, workingDirectory: root)
                    let r = try g.exec(trimmed, timeout: timeout)
                    self.record(Entry(source: .user, command: trimmed, output: TerminalOutput.sanitize(r.output),
                                      exitCode: r.timedOut ? nil : r.exitCode,
                                      timedOut: r.timedOut, at: Date()))
                } catch {
                    self.record(Entry(source: .system, command: trimmed,
                                      output: "\(error)", exitCode: -1, at: Date()))
                }
                cont.resume()
            }
        }
    }

    /// Ensure a live guest exists, provisioning the kernel + rootfs on first use.
    /// Serialized by `bootLock`. Returns the guest and its shared-root host path.
    private func ensureBooted(image: String, workingDirectory: String?) throws -> (VzGuest, String) {
        bootLock.lock(); defer { bootLock.unlock() }
        if let g = guest, !g.isFinished, let root = sharedRoot,
           !Self.needsRemount(sharedRoot: root, requestedCwd: workingDirectory) {
            return (g, root)
        }
        // Stale/dead guest, or the working folder moved outside the shared
        // root (remount) — tear down and boot fresh with the right share.
        guest?.shutdown(); guest = nil; sharedRoot = nil
        forwarder?.stop(); forwarder = nil

        #if !arch(arm64)
        throw SandboxError(message: "the Agent Sandbox requires Apple Silicon (the guest kernel is arm64)")
        #else
        let kernel = try provisionKernel()
        let rootfs = try provisionRootfs(image: image)
        let root = workingDirectory ?? Self.fallbackSharedRoot()

        // The image's Env/WorkingDir (written by the pull) feed the init script;
        // absent sidecar (dev rootfs override) → plain defaults.
        let imageConfig = (try? Data(contentsOf: URL(fileURLWithPath: rootfs)
            .appendingPathComponent(OCIClient.configSidecarName)))
            .map(OCIClient.parseImageConfig) ?? OCIClient.ImageConfig()

        let g = VzGuest()
        var cfg = VzGuest.Config(kernelPath: kernel, rootfsDir: rootfs)
        cfg.workspacePath = root
        cfg.guestWorkspacePath = "/workspace"
        cfg.imageEnv = imageConfig.env
        cfg.workdir = "/workspace"
        let net = { lock.lock(); defer { lock.unlock() }; return networkEnabled }()
        cfg.network = net
        // The rootfs is demand-paged over virtiofs (not RAM-resident like the old
        // initramfs boot), so guest RAM is pure workload headroom.
        let ramGB = ProcessInfo.processInfo.environment["SANDBOX_RAM_GB"].flatMap { UInt64($0) } ?? 1
        cfg.ramBytes = ramGB * 1024 * 1024 * 1024

        // Guest monitor consumer: every snapshot carries /proc/meminfo (tray RAM
        // readout — always), and when networked also the guest IP + listening
        // ports, which reconcile the host-side forwarder: a server the agent
        // starts on guest port N appears at localhost:N and disappears with it.
        if net {
            let fwd = SandboxPortForwarder()
            fwd.onMappingsChanged = { [weak self] ports in
                let list = ports.sorted().map { "localhost:\($0)" }.joined(separator: ", ")
                self?.record(Entry(source: .system, command: "",
                                   output: ports.isEmpty ? "port map: (none)" : "port map: \(list) → sandbox",
                                   exitCode: 0, at: Date()))
            }
            forwarder = fwd
        }
        var announcedIP: String?
        g.onNetSnapshot = { [weak self, weak fwd = forwarder] text in
            guard let self else { return }
            let snap = GuestNetParser.parse(text)
            if let total = snap.memTotalKB, let avail = snap.memAvailableKB {
                self.setGuestMemoryText(Self.memoryDisplayText(availableKB: avail, totalKB: total))
            }
            guard let fwd else { return }
            if let ip = snap.ip {
                fwd.setTarget(host: ip)
                if announcedIP != ip {
                    announcedIP = ip
                    self.record(Entry(source: .system, command: "",
                                      output: "network up — guest \(ip); guest ports auto-map to localhost",
                                      exitCode: 0, at: Date()))
                }
            }
            fwd.update(ports: snap.ports)
        }
        do {
            try g.boot(cfg, readyTimeout: 60)
        } catch {
            // Surface the guest console so a boot failure is diagnosable (kernel
            // panic, ENOEXEC, etc.) instead of an opaque "guest exited".
            let tail = String(g.consoleSnapshot().suffix(1500))
            g.shutdown()
            forwarder?.stop(); forwarder = nil
            NSLog("[sandbox] boot failed: \(error)\n--- guest console tail ---\n\(tail)\n--- end ---")
            throw SandboxError(message: "sandbox failed to start: \(error). Turn off the Agent Sandbox in Settings to run on the host, or check the base image. (guest console tail written to the server log)")
        }
        guest = g; sharedRoot = root
        setGuestRunning(true)
        record(Entry(source: .system, command: "", output: "sandbox ready — \(image), sharing \(root) at /workspace", exitCode: 0, at: Date()))
        return (g, root)
        #endif
    }

    // MARK: Provisioning

    /// contain's prebuilt minimal guest kernel (6.6, virtio-pci + virtiofs +
    /// virtio-console, ~37 MB raw / ~12 MB gz). Same asset the contain CLI
    /// fetches; pinned by tag, bumped only when the kernel is rebuilt.
    static let kernelURL = URL(string:
        "https://github.com/ddalcu/contain/releases/download/kernels-v2/kernel-contain-arm64.gz")!

    /// Base directory for cached sandbox artifacts. Migrates the pre-VZ
    /// `~/.mlx-serve/contain` cache in place on first touch (same layout — the
    /// pulled images stay valid; the kernel is re-validated separately).
    private var cacheDir: URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let dir = home.appendingPathComponent(".mlx-serve/sandbox", isDirectory: true)
        let legacy = home.appendingPathComponent(".mlx-serve/contain", isDirectory: true)
        let fm = FileManager.default
        if !fm.fileExists(atPath: dir.path), fm.fileExists(atPath: legacy.path) {
            try? fm.moveItem(at: legacy, to: dir)
        }
        return dir
    }

    /// Fetch (once) the guest kernel, self-healing a stale (pre-virtiofs) cache.
    /// Dev overrides: SANDBOX_KERNEL (or legacy CONTAIN_KERNEL).
    private func provisionKernel() throws -> String {
        let env = ProcessInfo.processInfo.environment
        if let override = env["SANDBOX_KERNEL"] ?? env["CONTAIN_KERNEL"], !override.isEmpty {
            return override
        }
        let fm = FileManager.default
        let dest = cacheDir.appendingPathComponent("kernel")
        try fm.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        if let cached = try? Data(contentsOf: dest, options: .alwaysMapped) {
            if Self.kernelHasVirtiofsSupport(cached) { return dest.path }
            try? fm.removeItem(at: dest) // pre-virtiofs kernel — re-fetch
        }
        let (gz, resp) = try OCIClient.httpGet(Self.kernelURL)
        guard resp.statusCode == 200 else {
            throw SandboxError(message: "could not download the sandbox kernel (HTTP \(resp.statusCode))")
        }
        let kernel = try OCIClient.gunzip(gz)
        guard Self.kernelHasVirtiofsSupport(kernel) else {
            throw SandboxError(message: "downloaded sandbox kernel lacks virtiofs support — release asset mismatch")
        }
        try kernel.write(to: dest, options: .atomic)
        return dest.path
    }

    /// Pull (once) + cache the base image rootfs for the guest arch. Dev
    /// overrides: SANDBOX_ROOTFS (or legacy CONTAIN_ROOTFS).
    private func provisionRootfs(image: String) throws -> String {
        let env = ProcessInfo.processInfo.environment
        if let override = env["SANDBOX_ROOTFS"] ?? env["CONTAIN_ROOTFS"], !override.isEmpty {
            return override
        }
        let fm = FileManager.default
        let dir = cacheDir.appendingPathComponent("images/\(Self.imageDirName(image))", isDirectory: true)
        let marker = dir.appendingPathComponent(Self.archMarkerName())
        let shPath = dir.appendingPathComponent("bin/sh").path
        let sidecar = dir.appendingPathComponent(OCIClient.configSidecarName).path
        // Cache hit only if it was pulled for THIS arch (marker present), has a
        // shell, AND carries the image-config sidecar (caches unpacked by the old
        // libcontain path lack it → re-pull so image env reaches the guest).
        if fm.fileExists(atPath: marker.path), fm.fileExists(atPath: shPath), fm.fileExists(atPath: sidecar) {
            return dir.path
        }
        // Absent / stale / wrong-arch → start clean and pull the correct arch.
        try? fm.removeItem(at: dir)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        do {
            _ = try OCIClient.pull(image: image, into: dir, arch: Self.guestArch) { msg in
                NSLog("[sandbox] \(msg)")
            }
        } catch {
            throw SandboxError(message: "could not pull the sandbox base image \"\(image)\" (\(Self.guestArch)): \(error.localizedDescription)")
        }
        fm.createFile(atPath: marker.path, contents: Data())
        return dir.path
    }
}

/// The sandbox transcript, isolated from `AgentSandbox`'s own published state so
/// that per-command appends re-render ONLY the Sandbox Terminal (which observes
/// this store) and never the tray menu (which observes `AgentSandbox.shared`).
/// Appends happen on the main thread (`AgentSandbox.record` dispatches there);
/// capped at `maxEntries` (oldest dropped).
final class SandboxTranscript: ObservableObject {
    static let maxEntries = 400

    @Published private(set) var entries: [AgentSandbox.Entry] = []

    func append(_ entry: AgentSandbox.Entry) {
        entries.append(entry)
        if entries.count > Self.maxEntries {
            entries.removeFirst(entries.count - Self.maxEntries)
        }
    }
}
