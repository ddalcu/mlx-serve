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
    /// Host loopback port mirroring the guest's sshd. nil until a NETWORKED
    /// guest with ssh support has booted. Changes at boot/teardown cadence
    /// (same as `guestRunning`) — safe for the tray to observe.
    @Published private(set) var sshPort: UInt16?

    /// Copyable one-liner for the "connect from your own terminal" row.
    var sshDisplayCommand: String? { sshPort.map { SandboxSSH.displayCommand(port: $0) } }

    private let lock = NSLock()
    private var enabled = AgentSandbox.resolveEnabled(requested: false)
    /// Transport the live guest booted with. Read by the MCP spawner and the
    /// background-process path; `.legacyConsole` until a guest exists.
    private(set) var transport: VzGuest.Transport = .legacyConsole
    /// Why the live guest is NOT on vsock (nil when it is, or before boot) —
    /// captured at boot so the MCP error can name the missing half.
    private(set) var transportFallback: String?
    /// Sandbox Terminal working directory. Under vsock each command is its own
    /// shell, so an interactive `cd` is carried here instead of in the guest.
    private var terminalCwd = "/workspace"
    /// The pinned sandbox image (single source of truth —
    /// `ServerOptions.SandboxConfig.baseImage`); only tests/smoke override it
    /// via `configure(baseImage:)`.
    private var baseImage = ServerOptions.SandboxConfig.baseImage
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
    private func setSshPort(_ v: UInt16?) {
        DispatchQueue.main.async { if self.sshPort != v { self.sshPort = v } }
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
    /// Serializes a project hot-mount (the `setProjectShares` swap + verify +
    /// `mountedProjects` update). Separate from `bootLock` so it can be held
    /// across the guest exec that verifies the mount without blocking every
    /// other guest command — two commands first-using two DIFFERENT folders
    /// concurrently would otherwise clobber each other's share (last full-map
    /// write wins, dropping the other slug).
    private let mountLock = NSLock()
    private var guest: VzGuest?
    private var sharedRoot: String?
    /// Host side of the live guest→host port map (created per boot when
    /// networking is on; fed by the guest's hvc2 net-report stream).
    private var forwarder: SandboxPortForwarder?
    /// Dedicated ssh mirror: `localhost:<sshPort>` → guest `:22` (dropbear).
    /// Separate from `forwarder` so the guest's own listener churn can never
    /// close the ssh mapping. Zero changes to SandboxPortForwarder itself —
    /// `targetPortOverride` is the seam.
    private var sshForwarder: SandboxPortForwarder?
    /// Live CLI sessions (embedded terminal / agent TUIs) pinning the guest —
    /// while non-empty, a workspace switch that needs a remount is DECLINED
    /// instead of silently rebooting the VM out from under the session.
    private var cliPins: [String] = []
    /// Per-chat project folders currently hot-mounted in the live guest at
    /// `/projects/<slug>` (slug → host path). `/workspace` stays the Settings
    /// default (pi/hermes + chats on the default); a chat whose folder differs
    /// gets its folder hot-mounted here on first tool use — no VM reboot.
    /// Cleared on teardown (a fresh guest re-mounts on demand). Guarded by
    /// `bootLock`.
    private var mountedProjects: [String: String] = [:]
    /// The live guest's ssh mirror port and rootfs dir — synchronous mirrors
    /// of what `ensureBooted` set up (`sshPort` is @Published on main, so it
    /// lags; the session-start path must not race it). Guarded by `bootLock`.
    private var currentSshPort: UInt16?
    private var rootfsPath: String?

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

    /// Posted by `configure` when the EXECUTION PLACEMENT changed (see
    /// `mcpRestartNeeded`). MCPManager listens and respawns its running stdio
    /// servers where the current setting routes them — a server is pinned to
    /// the placement it was spawned with, so without this, servers started
    /// before "enable sandbox" keep running on the host with full permissions.
    static let placementChanged = Notification.Name("AgentSandboxPlacementChanged")

    /// Pure rule: must running MCP servers be respawned after this configure()?
    /// Yes when the effective toggle flipped (either direction — off→on leaves
    /// host processes unconfined; on→off just killed the guest under live
    /// bridges), or when a guest re-provision (image/network change) tears down
    /// a guest that MCP bridges may be running in.
    static func mcpRestartNeeded(wasEnabled: Bool, nowEnabled: Bool, guestConfigChanged: Bool) -> Bool {
        wasEnabled != nowEnabled || (nowEnabled && guestConfigChanged)
    }

    /// Apply the Settings values. Turning the sandbox off, or changing the base
    /// image or the network mode, tears down any live guest so the next command
    /// re-provisions with the new configuration. `baseImage` is only overridden
    /// by tests/smoke — the app always runs the pinned image.
    func configure(enabled: Bool, baseImage: String = ServerOptions.SandboxConfig.baseImage, network: Bool = ServerOptions.SandboxConfig().network) {
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
        if Self.mcpRestartNeeded(wasEnabled: wasEnabled, nowEnabled: effective,
                                 guestConfigChanged: imageChanged || networkChanged) {
            NotificationCenter.default.post(name: Self.placementChanged, object: nil)
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
        guard let g = detachGuest() else { return }
        DispatchQueue.global(qos: .utility).async { shutdownBlocking(g) }
    }

    /// Detach the live guest + forwarders SYNCHRONOUSLY (UI state and a fresh
    /// `ensureBooted` are correct at once) and return the detached guest —
    /// the caller decides how its blocking stop runs (fire-and-forget for
    /// teardown; strictly BEFORE any rootfs delete for re-pull/reset).
    private func detachGuest() -> VzGuest? {
        bootLock.lock()
        let g = guest
        let fwd = forwarder
        let sfwd = sshForwarder
        guest = nil
        sharedRoot = nil
        mountedProjects = [:]
        forwarder = nil
        sshForwarder = nil
        currentSshPort = nil
        rootfsPath = nil
        bootLock.unlock()
        fwd?.stop()
        sfwd?.stop()
        setGuestRunning(false)
        setSshPort(nil)
        return g
    }

    // MARK: Test hooks (no VM — install/inspect the detached guest reference)

    func _testInstallGuest(_ g: VzGuest?, sharedRoot: String? = nil) {
        bootLock.lock()
        guest = g
        self.sharedRoot = (g == nil) ? nil : (sharedRoot ?? self.sharedRoot)
        bootLock.unlock()
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

    /// A filesystem-safe `/projects` subdirectory name for a host folder: the
    /// sanitized basename plus a short DETERMINISTIC suffix derived from the full
    /// path, so two different `.../src` folders never collide on one slug while
    /// the SAME folder always maps to the SAME slug (stable across commands).
    /// Guaranteed to satisfy `VZMultipleDirectoryShare` name rules (no `/`, not
    /// empty, well under NAME_MAX).
    static func projectSlug(hostPath: String) -> String {
        let host = (hostPath as NSString).standardizingPath
        let base = (host as NSString).lastPathComponent
        let safe = String(base.map { ch -> Character in
            (ch.isLetter || ch.isNumber || ch == "." || ch == "_" || ch == "-") ? ch : "-"
        })
        var name = safe
        if name.isEmpty || name == "." || name == ".." { name = "dir" }
        if name.count > 40 { name = String(name.suffix(40)) }
        // FNV-1a over the full path → 6 hex chars. Deterministic (no hashValue,
        // which is per-process randomized).
        var h: UInt64 = 1469598103934665603
        for b in host.utf8 { h = (h ^ UInt64(b)) &* 1099511628211 }
        return "\(name)-\(String(h & 0xffffff, radix: 16))"
    }

    /// Where a chat's host working folder lands in the guest, and whether that
    /// requires a hot-mount. `/workspace` covers the Settings default (and its
    /// descendants); an already-mounted project folder resolves under its
    /// `/projects/<slug>`; anything else is a NEW project that must be mounted
    /// (`mountSlug`/`mountHost` non-nil). Pure — the mount itself is the
    /// caller's side effect.
    struct GuestCwd: Equatable {
        var path: String
        var mountSlug: String?
        var mountHost: String?
        var mapped: Bool
    }

    static func resolveGuestCwd(hostPath: String?, defaultRoot: String,
                                mounted: [String: String]) -> GuestCwd {
        // 1. The default workspace (or a descendant) → /workspace.
        let onDefault = guestPath(hostPath: hostPath, sharedRoot: defaultRoot)
        if onDefault.mapped {
            return GuestCwd(path: onDefault.path, mountSlug: nil, mountHost: nil, mapped: true)
        }
        guard let hostPath else {
            return GuestCwd(path: "/workspace", mountSlug: nil, mountHost: nil, mapped: false)
        }
        let host = (hostPath as NSString).standardizingPath
        // 2. Already hot-mounted project (exact folder or a descendant of one).
        //    Longest-root-first so a nested mount wins over its ancestor.
        for (slug, mhost) in mounted.sorted(by: { $0.value.count > $1.value.count }) {
            let root = (mhost as NSString).standardizingPath
            let base = "\(VzGuest.guestProjectsPath)/\(slug)"
            if host == root {
                return GuestCwd(path: base, mountSlug: nil, mountHost: nil, mapped: true)
            }
            let rootWithSep = root.hasSuffix("/") ? root : root + "/"
            if host.hasPrefix(rootWithSep) {
                let rel = String(host.dropFirst(rootWithSep.count))
                return GuestCwd(path: rel.isEmpty ? base : "\(base)/\(rel)",
                                mountSlug: nil, mountHost: nil, mapped: true)
            }
        }
        // 3. A new project folder — needs a hot-mount at /projects/<slug>.
        let slug = projectSlug(hostPath: host)
        return GuestCwd(path: "\(VzGuest.guestProjectsPath)/\(slug)",
                        mountSlug: slug, mountHost: host, mapped: true)
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

    // MARK: CLI-session pinning (agent CLIs living in the guest)

    /// A workspace switch that would remount (= reboot) the guest is declined
    /// while a CLI session is live — the session's shell/TUI would die
    /// silently mid-keystroke otherwise. nil = proceed as before.
    static func remountBlockMessage(pinnedLabels: [String], needsRemount: Bool) -> String? {
        guard needsRemount, let label = pinnedLabels.last else { return nil }
        return "a sandboxed \(label) session is using the VM — close it before switching the workspace"
    }

    /// The most recent live CLI session's label (nil = unpinned). For the
    /// remount-decline message and the window title.
    var pinnedCliSessionLabel: String? {
        bootLock.lock(); defer { bootLock.unlock() }; return cliPins.last
    }

    /// The live pinned guest's shared root + most recent CLI-session label,
    /// read by the file-tool gate (`FileToolSandboxGate`). (nil, nil) unless a
    /// LIVE guest is pinned — a dead guest's pin doesn't block file tools,
    /// exactly as `ensureBooted` reboots past it for shell.
    var pinnedWorkspace: (root: String?, label: String?) {
        bootLock.lock(); defer { bootLock.unlock() }
        guard let g = guest, !g.isFinished, let root = sharedRoot, let label = cliPins.last else {
            return (nil, nil)
        }
        return (root, label)
    }

    // MARK: Workspace change (Settings default / chat folder pick)

    /// What a user workspace pick means for the live guest.
    enum WorkspaceChangeAction: Equatable {
        case none            // no live guest, or the share already covers the folder
        case teardown        // stale share → free the guest; next command reboots with the new share
        /// Stale share under live CLI sessions, but the pick was EXPLICIT
        /// (Settings) — tear down anyway; the terminal UI respawns its
        /// sessions in the new share (`workspaceRemounted` notification).
        case teardownRestartingSessions
        case blocked(String) // pinned + implicit switch — decline, same as shell
    }

    /// Posted after a workspace change tore down a PINNED guest (explicit
    /// Settings pick). The Sandbox window observes it and restarts every
    /// living session tab in the new share — without this, a live terminal
    /// quietly kept the OLD folder mounted until an app restart (live
    /// 2026-07-20: `ls /workspace` showed the previous workspace).
    static let workspaceRemounted = Notification.Name("AgentSandboxWorkspaceRemounted")

    /// Pure rule (unit-tested): mirrors `ensureBooted`'s remount decision so an
    /// EAGER pick-time remount can never disagree with the lazy command-time
    /// one. `restartPinnedSessions` distinguishes an EXPLICIT user pick (the
    /// Settings workspace row — re-anchoring the sandbox is the point, so
    /// live sessions restart) from an IMPLICIT switch (a chat tab's folder —
    /// still declined under a pin, exactly like shell).
    static func workspaceChangeAction(guestAlive: Bool, sharedRoot: String?,
                                      newWorkspace: String?, pinnedLabels: [String],
                                      restartPinnedSessions: Bool = false) -> WorkspaceChangeAction {
        guard guestAlive, let root = sharedRoot,
              needsRemount(sharedRoot: root, requestedCwd: newWorkspace) else { return .none }
        guard let blocked = remountBlockMessage(pinnedLabels: pinnedLabels, needsRemount: true) else {
            return .teardown
        }
        return restartPinnedSessions ? .teardownRestartingSessions : .blocked(blocked)
    }

    /// Called when the user picks a new agent workspace. Tears down a live
    /// guest whose `/workspace` share doesn't cover the new folder so the
    /// NEXT command boots sharing the right one (sub-second on virtiofs) —
    /// the same teardown `configure` does for an image/network change. Both
    /// teardown arms post `placementChanged` so guest-side MCP bridges (which
    /// die with the VM) respawn into the fresh guest.
    ///
    /// `restartPinnedSessions: true` (the Settings default-workspace row) also
    /// remounts under live CLI sessions and posts `workspaceRemounted` so the
    /// terminal respawns them in the new share. `false` (a chat tab's folder
    /// pick) returns the decline message while pinned — the chat's next tool
    /// call explains with the same words.
    @discardableResult
    func noteWorkspaceChanged(_ newWorkspace: String?, restartPinnedSessions: Bool = false) -> String? {
        bootLock.lock()
        let alive = guest.map { !$0.isFinished } ?? false
        let root = sharedRoot
        let pins = cliPins
        bootLock.unlock()
        switch Self.workspaceChangeAction(guestAlive: alive, sharedRoot: root,
                                          newWorkspace: newWorkspace, pinnedLabels: pins,
                                          restartPinnedSessions: restartPinnedSessions) {
        case .none:
            return nil
        case .teardown:
            recordRemount(newWorkspace, restartingSessions: false)
            teardown()
            NotificationCenter.default.post(name: Self.placementChanged, object: nil)
            return nil
        case .teardownRestartingSessions:
            recordRemount(newWorkspace, restartingSessions: true)
            teardown()
            NotificationCenter.default.post(name: Self.placementChanged, object: nil)
            NotificationCenter.default.post(name: Self.workspaceRemounted, object: nil)
            return nil
        case .blocked(let msg):
            return msg
        }
    }

    private func recordRemount(_ newWorkspace: String?, restartingSessions: Bool) {
        let dest = newWorkspace ?? Self.fallbackSharedRoot()
        record(Entry(source: .system, command: "",
                     output: "workspace changed — guest restarts sharing \(dest) at /workspace"
                         + (restartingSessions ? "; live sessions restart there" : ""),
                     exitCode: 0, at: Date()))
    }

    func pinCliSession(label: String) {
        bootLock.lock(); cliPins.append(label); bootLock.unlock()
    }

    func unpinCliSession(label: String) {
        bootLock.lock()
        if let i = cliPins.lastIndex(of: label) { cliPins.remove(at: i) }
        bootLock.unlock()
    }

    /// Preflight verdict for a cached rootfs that predates the ssh-enabled
    /// image (no dropbear baked in). Surfaced with a re-pull action — never a
    /// silent boot into a guest the terminal can't reach.
    static func staleImageMessage(image: String) -> String {
        "the cached sandbox image \"\(image)\" predates ssh support (no dropbear) — re-pull the image to update it"
    }

    /// Does the LIVE (or lazily-booted) guest ship dropbear? The ssh preflight.
    func guestHasDropbear() async -> Bool {
        await guestCommandSucceeds("command -v dropbear")
    }

    /// The stale-image fix: drop the cached rootfs for the current base image
    /// and tear down any live guest — the next boot re-pulls fresh.
    func repullBaseImage() {
        repullBaseImage(shutdownBlocking: { $0.shutdown() },
                        deleteImageDir: { try? FileManager.default.removeItem(at: $0) })
    }

    /// Seam for tests (same pattern as `teardown(shutdownBlocking:)`), and the
    /// reason repull is NOT `teardown()` + delete: teardown's stop is
    /// fire-and-forget, and deleting the cached image dir — the live guest's
    /// virtiofs root — under a still-stopping VM risks a partial delete that
    /// `provisionRootfs` later mistakes for a valid cache (marker + bin/sh +
    /// sidecar surviving = cache hit on a gutted tree). The stop completes
    /// FIRST; call off-main (the alert site uses Task.detached — the stop
    /// blocks up to ~10 s).
    func repullBaseImage(shutdownBlocking: (VzGuest) -> Void,
                         deleteImageDir: (URL) -> Void) {
        let image = { lock.lock(); defer { lock.unlock() }; return baseImage }()
        if let g = detachGuest() { shutdownBlocking(g) }
        let dir = cacheDir.appendingPathComponent("images/\(Self.imageDirName(image))", isDirectory: true)
        deleteImageDir(dir)
    }

    // MARK: Factory reset (Settings → Reset Sandbox)

    /// The one directory the factory reset deletes (`~/.mlx-serve/sandbox`):
    /// cached kernel, every pulled image — including ALL in-guest state
    /// (installed CLIs, agent configs, files outside /workspace) — and the
    /// app-owned ssh identity. Scope is pinned by AgentSandboxTests: never
    /// `~/.mlx-serve` itself (models/logs/workspace live next door).
    var dataDirectory: URL { cacheDir }

    /// Reset the sandbox to factory state: stop any live guest (live CLI
    /// sessions die with it — the confirmation UI says so in red), delete
    /// `dataDirectory`, and clear the transcript. The next use re-provisions
    /// everything from scratch (kernel, image, fresh ssh keypair).
    ///
    /// The guest shutdown is BLOCKING (bounded ~10 s) and runs off-main
    /// before the delete — removing a rootfs out from under a still-stopping
    /// VM risks a partial delete that a later boot mistakes for a valid
    /// cache. Guest detach is synchronous (same shape as `teardown`).
    func resetAllData(completion: (@Sendable () -> Void)? = nil) {
        let g = detachGuest()
        bootLock.lock()
        terminalCwd = "/workspace"
        bootLock.unlock()
        let dir = dataDirectory
        DispatchQueue.global(qos: .userInitiated).async {
            g?.shutdown() // blocking, bounded — files closed before the delete
            try? FileManager.default.removeItem(at: dir)
            DispatchQueue.main.async {
                self.transcriptStore.reset()
                self.transcriptStore.append(Entry(
                    source: .system, command: "",
                    output: "sandbox reset — guest stopped; cached kernel, images (with all in-guest data) and the ssh identity deleted. Next use re-provisions from scratch.",
                    exitCode: 0, at: Date()))
                completion?()
            }
        }
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

    /// Why `chooseTransport` fell back, for the MCP error message — the two
    /// halves fail independently and send the user to different fixes (issue
    /// #89: every Developer ID build shipped without the vz-agent, and the
    /// collective "kernel + vz-agent" wording hid which half was broken).
    /// nil = vsock is available, no fallback.
    static func transportFallbackReason(kernelData: Data?, agentBinary: String?) -> String? {
        var missing: [String] = []
        if agentBinary == nil {
            missing.append("the vz-agent guest binary is missing from the app bundle (reinstall or rebuild the app)")
        }
        if let kernelData {
            if !kernelHasVsockSupport(kernelData) {
                missing.append("the guest kernel predates \(kernelTag) (no vsock support — delete ~/.mlx-serve/sandbox to re-fetch)")
            }
        } else {
            missing.append("the guest kernel could not be read")
        }
        return missing.isEmpty ? nil : missing.joined(separator: "; ")
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
                    // Map the chat's folder to /workspace (the default) or hot-mount
                    // it at /projects/<slug> on first use — no VM reboot.
                    let gp = self.resolveAndMountProject(guest: g,
                                                         hostPath: workingDirectory ?? hostCwd,
                                                         defaultRoot: root)
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

    /// Resolve a chat command's guest cwd, hot-mounting the folder at
    /// `/projects/<slug>` on FIRST use if it isn't the Settings default (or an
    /// already-mounted project). Runs on the exec thread (never the main
    /// thread); `bootLock` is taken only to read/update `mountedProjects`, never
    /// held across a guest `exec`. Returns `mapped: false` only when the folder
    /// can't be mounted (not a real dir, or the hot-swap failed) — the command
    /// then runs in `/workspace` with the existing warning.
    private func resolveAndMountProject(guest g: VzGuest, hostPath: String?,
                                        defaultRoot: String) -> GuestCwd {
        let snapshot: [String: String] = {
            bootLock.lock(); defer { bootLock.unlock() }; return mountedProjects
        }()
        let r = Self.resolveGuestCwd(hostPath: hostPath, defaultRoot: defaultRoot, mounted: snapshot)
        guard let slug = r.mountSlug, let host = r.mountHost else { return r }

        // Only mount a folder that actually exists as a directory on the host.
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: host, isDirectory: &isDir), isDir.boolValue else {
            return GuestCwd(path: "/workspace", mountSlug: nil, mountHost: nil, mapped: false)
        }

        // Serialize the actual mount so a concurrent first-use of a different
        // folder can't clobber this slug (each rebuilds the FULL share map).
        mountLock.lock(); defer { mountLock.unlock() }
        let current: [String: String] = {
            bootLock.lock(); defer { bootLock.unlock() }; return mountedProjects
        }()
        if current[slug] == host {
            return GuestCwd(path: r.path, mountSlug: nil, mountHost: nil, mapped: true)
        }
        var next = current
        next[slug] = host
        guard g.setProjectShares(next) else {
            // Legacy guest without the projects device — fall back to /workspace.
            return GuestCwd(path: "/workspace", mountSlug: nil, mountHost: nil, mapped: false)
        }
        // The runtime `.share` swap usually surfaces the new subdir live; if the
        // guest hasn't picked it up, an in-guest remount of the projects tag does
        // (still NO VM reboot, so live CLI sessions survive).
        if !Self.guestHasProjectDir(g, slug: slug) {
            _ = try? g.exec("mount -t virtiofs \(VzGuest.projectsTag) \(VzGuest.guestProjectsPath) 2>/dev/null; true",
                            timeout: 10)
            if !Self.guestHasProjectDir(g, slug: slug) {
                return GuestCwd(path: "/workspace", mountSlug: nil, mountHost: nil, mapped: false)
            }
        }
        bootLock.lock(); mountedProjects[slug] = host; bootLock.unlock()
        record(Entry(source: .system, command: "",
                     output: "mounted \(host) at \(VzGuest.guestProjectsPath)/\(slug)", exitCode: 0, at: Date()))
        return GuestCwd(path: r.path, mountSlug: nil, mountHost: nil, mapped: true)
    }

    /// Ensure a chat's folder is hot-mounted at `/projects/<slug>` when a LIVE
    /// guest exists — the host-side file tools (writeFile, readFile, …) call
    /// this so a file-only session still shows up in the VM, not just sessions
    /// that ran `shell`. Fire-and-forget: the file tool operates on the host
    /// bytes immediately and never waits on the mount. Deliberately does NOT
    /// boot a guest — a host-side file write shouldn't spin up a VM; the folder
    /// mounts when a shell first runs (which does boot). No-op when the sandbox
    /// is off, no folder was given, or the folder is the default workspace.
    func ensureProjectMountedAsync(workingDirectory: String?) {
        guard isEnabled, let wd = workingDirectory else { return }
        DispatchQueue.global(qos: .utility).async {
            let live: (g: VzGuest, root: String)? = {
                self.bootLock.lock(); defer { self.bootLock.unlock() }
                guard let g = self.guest, !g.isFinished, let root = self.sharedRoot else { return nil }
                return (g, root)
            }()
            guard let live else { return }        // no guest booted → mounts on first shell
            _ = self.resolveAndMountProject(guest: live.g, hostPath: wd, defaultRoot: live.root)
        }
    }

    /// Does `/projects/<slug>` exist in the live guest? (Confirms a hot-mount
    /// actually surfaced.)
    private static func guestHasProjectDir(_ g: VzGuest, slug: String) -> Bool {
        let path = "\(VzGuest.guestProjectsPath)/\(slug)"
        guard let r = try? g.exec("test -d \(VzGuest.shellQuote(path)) && echo __VZ_OK__", timeout: 10)
        else { return false }
        return r.output.contains("__VZ_OK__")
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
                        let why = self.transportFallback ?? "the guest booted the legacy console shell"
                        throw SandboxError(message:
                            "MCP servers need the vsock guest transport: \(why)")
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

    // MARK: CLI sessions in the guest (embedded terminal / user ssh)

    /// Everything the embedded terminal needs to open one session. `sshArgs`
    /// is the argv for `/usr/bin/ssh`; `displayCommand` is the copyable
    /// "connect from your own terminal" line (same option set, pinned by
    /// SandboxSSHTests).
    struct CliSession {
        let label: String        // "pi" / "hermes" / "shell" — window title + pin
        let agentId: String?     // nil = plain login shell
        let sshPort: UInt16
        let sshArgs: [String]
        let displayCommand: String
    }

    /// Boot (if needed), verify the image ships dropbear, materialize the
    /// agent's config + bootstrap into the rootfs, wait for the ssh mirror,
    /// and pin the guest. The caller MUST balance with `endCliSession(_:)`
    /// when the terminal exits. Throws `SandboxError` with an actionable
    /// message on every distinct failure — never a hung terminal.
    func startCliSession(agent: SandboxAgentSpec?, model: String?, serverPort: UInt16,
                         budget: AgentBudget.Budget, apiKey: String?,
                         entries: [AgentModelEntry] = []) async throws -> CliSession {
        let image = { lock.lock(); defer { lock.unlock() }; return baseImage }()
        return try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let (g, _) = try self.ensureBooted(image: image, workingDirectory: nil)
                    let (port, rootfs): (UInt16?, String?) = {
                        self.bootLock.lock(); defer { self.bootLock.unlock() }
                        return (self.currentSshPort, self.rootfsPath)
                    }()
                    guard let sshPort = port, let rootfsDir = rootfs else {
                        throw SandboxError(message: "the sandbox guest has no ssh channel — turn on guest networking in Settings → Agent Sandbox")
                    }
                    // Stale-cache preflight: the ssh transport needs dropbear
                    // BAKED into the image; an old cached rootfs simply lacks
                    // it (the fix is a re-pull, offered by the UI).
                    let probe = try g.exec("command -v dropbear", timeout: 20)
                    guard !probe.timedOut, probe.exitCode == 0 else {
                        throw SandboxError(message: Self.staleImageMessage(image: image))
                    }

                    var remoteCommand: String?
                    if let agent {
                        guard let model, !model.isEmpty else {
                            throw SandboxError(message: "no model is loaded — start the server before opening a \(agent.displayName) session")
                        }
                        let bootstrap = try SandboxAgentRegistry.materialize(
                            spec: agent, model: model, serverPort: serverPort,
                            budget: budget, apiKey: apiKey, entries: entries,
                            rootfsDir: rootfsDir)
                        remoteCommand = "sh \(bootstrap)"
                    }

                    // The mirror listens once the first net snapshot delivers
                    // the guest IP (~1 s cadence). Bounded wait, distinct error.
                    let deadline = Date().addingTimeInterval(20)
                    while Date() < deadline, !self.sshMirrorActive(port: sshPort) {
                        Thread.sleep(forTimeInterval: 0.1)
                    }
                    guard self.sshMirrorActive(port: sshPort) else {
                        throw SandboxError(message: "the guest network never came up — no ssh mirror on localhost:\(sshPort)")
                    }

                    let label = agent?.displayName ?? "shell"
                    self.pinCliSession(label: label)
                    self.record(Entry(source: .system, command: "",
                                      output: "\(label) session opened — ssh mirror localhost:\(sshPort) → guest :22",
                                      exitCode: 0, at: Date()))
                    cont.resume(returning: CliSession(
                        label: label, agentId: agent?.id, sshPort: sshPort,
                        sshArgs: SandboxSSH.sshArgs(port: sshPort,
                                                    keyPath: SandboxSSH.privateKeyPath,
                                                    knownHostsPath: SandboxSSH.knownHostsPath,
                                                    remoteCommand: remoteCommand),
                        displayCommand: SandboxSSH.displayCommand(port: sshPort)))
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Balance for `startCliSession` — releases the workspace pin. Safe to
    /// call after the guest already died (the pin list is host state).
    func endCliSession(_ session: CliSession) {
        unpinCliSession(label: session.label)
        record(Entry(source: .system, command: "",
                     output: "\(session.label) session closed", exitCode: 0, at: Date()))
    }

    private func sshMirrorActive(port: UInt16) -> Bool {
        let fwd: SandboxPortForwarder? = {
            bootLock.lock(); defer { bootLock.unlock() }; return sshForwarder
        }()
        return fwd?.activePorts.contains(port) ?? false
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
    /// Serialized by `bootLock`. Returns the guest and its `/workspace` shared
    /// root (always the Settings default). A live guest is REUSED regardless of
    /// which chat folder the command targets: a folder outside `/workspace` is
    /// hot-mounted at `/projects/<slug>` by the caller, never a VM reboot. Only
    /// a Settings default change (`noteWorkspaceChanged`, which tears the guest
    /// down) or a dead guest boots a fresh one.
    private func ensureBooted(image: String, workingDirectory: String?) throws -> (VzGuest, String) {
        bootLock.lock(); defer { bootLock.unlock() }
        if let g = guest, !g.isFinished, let root = sharedRoot {
            return (g, root)
        }
        // Stale/dead guest — tear down and boot fresh sharing the default.
        guest?.shutdown(); guest = nil; sharedRoot = nil; mountedProjects = [:]
        forwarder?.stop(); forwarder = nil
        sshForwarder?.stop(); sshForwarder = nil
        currentSshPort = nil; rootfsPath = nil
        setSshPort(nil)
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
        // `/workspace` is ALWAYS the Settings default (pi/hermes + chats on the
        // default live here); a chat's own folder is hot-mounted at /projects on
        // first use. `workingDirectory` no longer selects the boot share.
        _ = workingDirectory
        let root = Self.fallbackSharedRoot()

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
        transportFallback = Self.transportFallbackReason(kernelData: kernelData, agentBinary: agentBinary)

        // The App Store build's guest is bundled: there is no download path to a
        // different kernel or a missing agent, so a legacy fallback there means
        // something is wrong with the bundle. Fail loudly rather than boot a VM
        // whose MCP servers can never connect.
        if BuildFeatures.current.isMAS && cfg.transport == .legacyConsole {
            throw SandboxError(message: "the bundled guest is missing vsock support or the vz-agent binary — the app bundle is incomplete")
        }
        let net = { lock.lock(); defer { lock.unlock() }; return networkEnabled }()
        cfg.network = net

        // SSH: every networked boot gets dropbear (key-only) + a dedicated
        // loopback mirror — the embedded terminal, the copyable ssh command,
        // and the smoke all ride the same transport. Failure here degrades to
        // a guest without ssh (shell/agent tools keep working), never a boot
        // failure.
        var bootSshPort: UInt16?
        if net {
            do {
                try SandboxSSH.ensureKeypair()
                SandboxSSH.resetKnownHosts() // host keys churn with re-pulls; stale entry = MITM banner
                try SandboxSSH.injectAuthorizedKeys(rootfsDir: rootfs)
                try SandboxSSH.injectGuestProfile(rootfsDir: rootfs)
                if let port = SandboxSSH.allocateSshPort(taken: []) {
                    cfg.sshEnabled = true
                    bootSshPort = port
                } else {
                    NSLog("[sandbox] ssh mirror skipped: no free loopback port from 2222")
                }
            } catch {
                NSLog("[sandbox] ssh setup skipped: \(error)")
            }
        }
        // The rootfs is demand-paged over virtiofs (not RAM-resident like the old
        // initramfs boot), so guest RAM is pure workload headroom — a CAP, not a
        // reservation (VZ commits pages lazily). 1 GB OOM-killed hermes's
        // browser-deps `npm install` at ~850 MB RSS (issue #150 follow-up), and
        // an agent that compiles or runs headless Chromium needs real headroom.
        let ramGB = ProcessInfo.processInfo.environment["SANDBOX_RAM_GB"].flatMap { UInt64($0) } ?? 4
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
        if let sp = bootSshPort {
            // Own forwarder for ssh: same class, fixed mapping localhost:sp →
            // guest :22, immune to the guest's own listener churn (dropbear
            // binds 0.0.0.0:22, which the general forwarder would ALSO mirror
            // to localhost:22 — the dedicated instance is the stable address).
            let sfwd = SandboxPortForwarder()
            sfwd.targetPortOverride = { _ in 22 }
            sshForwarder = sfwd
        }
        var announcedIP: String?
        g.onNetSnapshot = { [weak self, weak fwd = forwarder, weak sshFwd = sshForwarder] text in
            guard let self else { return }
            let snap = GuestNetParser.parse(text)
            if let total = snap.memTotalKB, let avail = snap.memAvailableKB {
                self.setGuestMemoryText(Self.memoryDisplayText(availableKB: avail, totalKB: total))
            }
            if let sshFwd, let ip = snap.ip, let sp = bootSshPort {
                sshFwd.setTarget(host: ip)
                sshFwd.update(ports: [sp])
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
            // The guest's own :22 listener rides the DEDICATED mirror only —
            // mirroring it here too would fight the ssh forwarder for
            // localhost:22 (and usually lose to a host sshd anyway).
            fwd.update(ports: snap.ports.filter { $0 != 22 })
        }
        do {
            try g.boot(cfg, readyTimeout: 60)
        } catch {
            // Surface the guest console so a boot failure is diagnosable (kernel
            // panic, ENOEXEC, etc.) instead of an opaque "guest exited".
            let tail = String(g.consoleSnapshot().suffix(1500))
            g.shutdown()
            forwarder?.stop(); forwarder = nil
            sshForwarder?.stop(); sshForwarder = nil
            NSLog("[sandbox] boot failed: \(error)\n--- guest console tail ---\n\(tail)\n--- end ---")
            throw SandboxError(message: "sandbox failed to start: \(error). Turn off the Agent Sandbox in Settings to run on the host, or check the base image. (guest console tail written to the server log)")
        }
        guest = g; sharedRoot = root
        currentSshPort = bootSshPort; rootfsPath = rootfs
        setGuestRunning(true)
        setSshPort(bootSshPort)
        record(Entry(source: .system, command: "", output: "sandbox ready — \(image), sharing \(root) at /workspace", exitCode: 0, at: Date()))
        return (g, root)
        #endif
    }

    // MARK: Provisioning

    /// Our prebuilt minimal guest kernel (6.6, virtio-pci + virtiofs +
    /// virtio-console + virtio-vsock, ~37 MB raw / ~12 MB gz), built by
    /// `containers/guest-kernel/build.sh` and published as a release asset on
    /// THIS repo; pinned by tag, bumped only when the kernel is rebuilt.
    /// (`kernels-v2`/`v3` came from the contain repo and stay published there.)
    ///
    /// `kernels-v3` added `CONFIG_VSOCKETS` + `CONFIG_VIRTIO_VSOCKETS`, which the
    /// guest agent needs. A `kernels-v2` cache still boots — it just falls back
    /// to the legacy console shell (see `chooseTransport`).
    ///
    /// `kernels-v4` (6.6.151) makes the guest a normally writable Linux:
    /// - fuse patch so no virtiofs inode ever lacks owner-read (issue #150):
    ///   Apple's host-side server runs unprivileged, making any owner-read-less
    ///   file a black hole the guest can't create, stat, open, or unlink —
    ///   which broke `apt-get install` (dpkg's mode-000 .dpkg-new staging) and
    ///   hermes's Node extraction (GNU tar's mode-000 dangling-symlink
    ///   placeholders). Creates, mkdir and chmod are clamped; see
    ///   `containers/guest-kernel/patches/`.
    /// - ≥6.6.69 is load-bearing on M4 for the stable SVE-hwcap filter: older
    ///   kernels advertised sve2 the CPU can't execute outside streaming mode
    ///   (M4 has SME, no real SVE), and capability probes SIGILL'd — hermes
    ///   died on launch importing python-cryptography (OpenSSL's armcap
    ///   probe); Go binaries are the same class.
    static let kernelTag = "kernels-v4"
    static let kernelURL = URL(string:
        "https://github.com/ddalcu/mlx-serve/releases/download/\(kernelTag)/kernel-arm64.gz")!

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

    /// Factory reset — the transcript is sandbox data too.
    func reset() {
        entries.removeAll()
    }
}
