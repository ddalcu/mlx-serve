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
    private var enabled = AgentSandbox.resolveEnabled(requested: false)
    /// Transport the live guest booted with. Read by the MCP spawner and the
    /// background-process path; `.legacyConsole` until a guest exists.
    private(set) var transport: VzGuest.Transport = .legacyConsole
    /// Sandbox Terminal working directory. Under vsock each command is its own
    /// shell, so an interactive `cd` is carried here instead of in the guest.
    private var terminalCwd = "/workspace"
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
    /// Used-only readout ("384 MB RAM") — the "/ total" suffix was tray
    /// noise. `totalKB` still feeds the used computation (meminfo reports
    /// available, not used).
    static func memoryDisplayText(availableKB: Int, totalKB: Int) -> String {
        let usedMB = Double(max(0, totalKB - availableKB)) / 1024.0
        let quantized = Int((usedMB / 16.0).rounded()) * 16
        return "\(quantized) MB RAM"
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

    /// A build with no host shell (the App Store build) cannot have the sandbox
    /// off: `ShellHandler.route` already forces every command into the guest,
    /// so a false `isEnabled` would only LIE downstream — the system prompt's
    /// execution-environment section would describe a macOS/zsh/brew host the
    /// commands never touch, and the Sandbox Terminal + tray section would stay
    /// hidden while a guest is actually running. The settings toggle (and any
    /// stale persisted `enabled:false`) still rules the Developer ID build.
    static func resolveEnabled(requested: Bool,
                               hostShellAllowed: Bool = BuildFeatures.current.hostShell) -> Bool {
        requested || !hostShellAllowed
    }

    /// Apply the Settings values. Turning the sandbox off, or changing the base
    /// image or the network mode, tears down any live guest so the next command
    /// re-provisions with the new configuration.
    func configure(enabled: Bool, baseImage: String, network: Bool = ServerOptions.SandboxConfig().network) {
        let effective = Self.resolveEnabled(requested: enabled)
        let trimmed = baseImage.trimmingCharacters(in: .whitespacesAndNewlines)
        lock.lock()
        let imageChanged = trimmed != self.baseImage && !trimmed.isEmpty
        let networkChanged = network != self.networkEnabled
        let wasEnabled = self.enabled
        self.enabled = effective
        self.networkEnabled = network
        if !trimmed.isEmpty { self.baseImage = trimmed }
        lock.unlock()
        if (!effective && wasEnabled) || (effective && (imageChanged || networkChanged)) {
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

    // MARK: Sandbox Terminal cwd (pure)

    /// Marker the terminal wrapper prints its final `$PWD` on.
    static let cwdMarker = "__VZ_CWD__"

    /// The Sandbox Terminal lets an interactive `cd` stick between commands.
    /// The legacy transport got that for free: one long-lived shell held the
    /// cwd. Under vsock every command is its own `sh -c`, so the shell's final
    /// `$PWD` is echoed on a marker line and carried forward by the host.
    ///
    /// The exit status is preserved explicitly — `printf` would otherwise
    /// clobber `$?`.
    static func wrapUserCommand(_ command: String, guestCwd: String) -> String {
        "cd \(VzGuest.shellQuote(guestCwd)) 2>/dev/null\n"
            + command + "\n"
            + "__vz_rc=$?; printf '\\n\(cwdMarker)%s\\n' \"$PWD\"; exit $__vz_rc"
    }

    /// Strip the marker line and report the cwd it carried. A command that
    /// killed its own shell never prints one — then the cwd simply doesn't move.
    static func splitCwdMarker(_ output: String) -> (body: String, cwd: String?) {
        guard let range = output.range(of: cwdMarker, options: .backwards) else {
            return (output, nil)
        }
        let cwd = output[range.upperBound...]
            .prefix { !$0.isNewline }
            .trimmingCharacters(in: .whitespaces)
        var body = String(output[..<range.lowerBound])
        if body.hasSuffix("\n") { body.removeLast() }
        return (body, cwd.isEmpty ? nil : cwd)
    }

    /// vsock arm of `runUserCommand`: the marker rides STDOUT (the wrapper's
    /// `printf` writes it there), so it must be parsed out of stdout ALONE and
    /// stderr re-attached afterwards. Splitting the MERGED output instead
    /// silently drops everything past the marker — which is exactly where a
    /// failing command's stderr lands.
    static func composeUserOutput(stdout: String, stderr: String) -> (body: String, cwd: String?) {
        let split = splitCwdMarker(stdout)
        guard !stderr.isEmpty else { return split }
        if split.body.isEmpty { return (stderr, split.cwd) }
        let joiner = split.body.hasSuffix("\n") ? "" : "\n"
        return (split.body + joiner + stderr, split.cwd)
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

    /// The vsock transport needs `CONFIG_VSOCKETS` + `CONFIG_VIRTIO_VSOCKETS`
    /// (contain's `kernels-v3`). Without them the guest agent's `bind()` fails
    /// with EAFNOSUPPORT and the guest boots into a VM nobody can talk to, so we
    /// fall back to the legacy console shell rather than hanging on the ready
    /// handshake.
    static func kernelHasVsockSupport(_ kernelData: Data) -> Bool {
        kernelData.range(of: Data("virtio_vsock".utf8)) != nil
    }

    /// Where the `vz-agent` ELF lives, most-preferred first:
    ///  1. `VZ_AGENT_PATH` — dev override, and how the smoke test injects one.
    ///  2. `Contents/Resources/guest/vz-agent` — the shipped app bundle.
    ///  3. `zig-out/guest/vz-agent` beside the executable — `swift run` builds.
    static func agentBinaryPath(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        bundleResourceURL: URL? = Bundle.main.resourceURL,
        executableURL: URL? = Bundle.main.executableURL
    ) -> String? {
        let fm = FileManager.default
        if let override = environment["VZ_AGENT_PATH"], fm.isReadableFile(atPath: override) {
            return override
        }
        if let bundled = bundleResourceURL?.appendingPathComponent("guest/vz-agent").path,
           fm.isReadableFile(atPath: bundled) {
            return bundled
        }
        if let dev = executableURL?
            .deletingLastPathComponent()
            .appendingPathComponent("../../../../zig-out/guest/vz-agent")
            .standardizedFileURL.path,
           fm.isReadableFile(atPath: dev) {
            return dev
        }
        return nil
    }

    /// vsock only when BOTH halves are present. Either missing → the legacy
    /// console shell, which still works. Never a hang, never a silent no-op.
    static func chooseTransport(kernelData: Data?, agentBinary: String?) -> VzGuest.Transport {
        guard agentBinary != nil, let kernelData, kernelHasVsockSupport(kernelData) else {
            return .legacyConsole
        }
        return .vsock
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

    /// Run a command typed by the user in the Sandbox Terminal. Same guest as
    /// the agent (filesystem state is shared), recorded in the transcript. An
    /// interactive `cd` sticks between commands.
    ///
    /// Under `.legacyConsole` that came free — one long-lived shell held the
    /// cwd. Under `.vsock` each command is its own `sh -c`, so the shell echoes
    /// its final `$PWD` on a marker line and we carry it forward. Never throws —
    /// provisioning/boot errors are recorded as a `.system` entry.
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

                    var output: String
                    var result: VzGuest.ExecResult
                    if self.transport == .vsock {
                        let cwd = { self.bootLock.lock(); defer { self.bootLock.unlock() }; return self.terminalCwd }()
                        result = try g.exec(Self.wrapUserCommand(trimmed, guestCwd: cwd), timeout: timeout)
                        let split = Self.composeUserOutput(stdout: result.stdout, stderr: result.stderr)
                        output = split.body
                        if let moved = split.cwd {
                            self.bootLock.lock(); self.terminalCwd = moved; self.bootLock.unlock()
                        }
                    } else {
                        result = try g.exec(trimmed, timeout: timeout)
                        output = result.output
                    }

                    self.record(Entry(source: .user, command: trimmed, output: TerminalOutput.sanitize(output),
                                      exitCode: result.timedOut ? nil : result.exitCode,
                                      timedOut: result.timedOut, at: Date()))
                } catch {
                    self.record(Entry(source: .system, command: trimmed,
                                      output: "\(error)", exitCode: -1, at: Date()))
                }
                cont.resume()
            }
        }
    }

    // MARK: MCP servers in the guest

    /// Preflight for `GuestMCPSpawner`: does this command exist in the guest?
    func guestCommandSucceeds(_ command: String) async -> Bool {
        let image = { lock.lock(); defer { lock.unlock() }; return baseImage }()
        return await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let (g, _) = try self.ensureBooted(image: image, workingDirectory: nil)
                    let r = try g.exec(command, timeout: 20)
                    cont.resume(returning: !r.timedOut && r.exitCode == 0)
                } catch {
                    cont.resume(returning: false)
                }
            }
        }
    }

    /// Start a stdio MCP server inside the guest and bridge its stdio to a
    /// descriptor the MCP SDK can drive.
    ///
    /// This is the fix for the confinement hole: previously the server ran on
    /// the host with the user's full permissions even while the Agent Sandbox
    /// was on. `cwd` was set, but a cwd is not a permission boundary.
    func openMCPBridge(command: String, hostCwd: String, env: [String: String]) async throws -> GuestMCPBridge {
        let image = { lock.lock(); defer { lock.unlock() }; return baseImage }()
        return try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let (g, root) = try self.ensureBooted(image: image, workingDirectory: hostCwd)
                    guard self.transport == .vsock else {
                        throw SandboxError(message:
                            "MCP servers need the vsock guest transport (kernel \(Self.kernelTag) + vz-agent); "
                            + "the guest booted the legacy console shell")
                    }
                    // Only the shared folder exists in the guest; anything else
                    // maps to /workspace rather than silently landing elsewhere.
                    let guestCwd = Self.guestPath(hostPath: hostCwd, sharedRoot: root).path

                    let stream = try g.openStream()
                    let request = GuestProtocol.Request(
                        command: command,
                        cwd: guestCwd,
                        env: env.sorted { $0.key < $1.key }.map { (key: $0.key, value: $0.value) })
                    try stream.write(GuestProtocol.frame(.request, request.encode()))

                    cont.resume(returning: try GuestMCPBridge(stream: stream))
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// The kernel+rootfs source for this build: bundled on the App Store,
    /// downloaded on Developer ID. The download arm delegates to the existing
    /// `provisionKernel`/`provisionRootfs` (which own the caching); the bundled
    /// arm unpacks `Contents/Resources/guest/` into the container.
    private func makeProvisioner() -> SandboxProvisioner {
        switch SandboxProvisionerFactory.kind() {
        case .downloading:
            return DownloadingProvisioner(
                fetchKernel: { try self.provisionKernel() },
                pullRootfs: { try self.provisionRootfs(image: $0) })
        case .bundled:
            return BundledProvisioner(
                resourcesURL: Bundle.main.resourceURL ?? URL(fileURLWithPath: "."),
                containerRoot: cacheDir)
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
        terminalCwd = "/workspace" // a fresh guest starts in its workdir

        #if !arch(arm64)
        throw SandboxError(message: "the Agent Sandbox requires Apple Silicon (the guest kernel is arm64)")
        #else
        // Developer ID fetches the kernel + rootfs; the App Store build unpacks
        // them from the bundle (guideline 2.5.2 forbids downloading them). Same
        // guest either way — just a different source.
        let provisioner = makeProvisioner()
        let kernel = try provisioner.kernelPath()
        let rootfs = try provisioner.rootfsDir(image: image)
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

        // vsock only when the kernel supports it AND we have an agent to inject;
        // otherwise the legacy console shell, which every kernel can run.
        let agentBinary = Self.agentBinaryPath()
        let kernelData = try? Data(contentsOf: URL(fileURLWithPath: kernel), options: .mappedIfSafe)
        cfg.transport = Self.chooseTransport(kernelData: kernelData, agentBinary: agentBinary)
        cfg.agentBinaryPath = agentBinary
        transport = cfg.transport

        // The App Store build's guest is bundled: there is no download path to a
        // different kernel or a missing agent, so a legacy fallback there means
        // something is wrong with the bundle. Fail loudly rather than boot a VM
        // whose MCP servers can never connect.
        if BuildFeatures.current.isMAS && cfg.transport == .legacyConsole {
            throw SandboxError(message: "the bundled guest is missing vsock support or the vz-agent binary — the app bundle is incomplete")
        }
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
    /// virtio-console + virtio-vsock, ~37 MB raw / ~12 MB gz). Same asset the
    /// contain CLI fetches; pinned by tag, bumped only when the kernel is rebuilt.
    ///
    /// `kernels-v3` added `CONFIG_VSOCKETS` + `CONFIG_VIRTIO_VSOCKETS`, which the
    /// guest agent needs. A `kernels-v2` cache still boots — it just falls back
    /// to the legacy console shell (see `chooseTransport`).
    static let kernelTag = "kernels-v3"
    static let kernelURL = URL(string:
        "https://github.com/ddalcu/contain/releases/download/\(kernelTag)/kernel-contain-arm64.gz")!

    /// Cache filename. Tag-versioned rather than sniffed, so bumping the kernel
    /// invalidates every old cache by construction instead of by remembering to
    /// add another `kernelHas…Support` byte check.
    static func kernelCacheName(tag: String = kernelTag) -> String { "kernel-\(tag)" }

    /// Cached kernels from other tags, which `provisionKernel` prunes.
    static func staleKernelNames(_ existing: [String], tag: String = kernelTag) -> [String] {
        existing.filter { ($0 == "kernel" || $0.hasPrefix("kernel-")) && $0 != kernelCacheName(tag: tag) }
    }

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
        let dest = cacheDir.appendingPathComponent(Self.kernelCacheName())
        try fm.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        // Drop caches from older tags — including the untagged `kernel` file the
        // pre-vsock builds wrote.
        let existing = (try? fm.contentsOfDirectory(atPath: cacheDir.path)) ?? []
        for stale in Self.staleKernelNames(existing) {
            try? fm.removeItem(at: cacheDir.appendingPathComponent(stale))
        }

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
