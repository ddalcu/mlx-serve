import Foundation
import AppKit
import Darwin

/// Lifecycle status of an agent-spawned background process. `.exited` carries
/// the real termination code (from `Process.terminationHandler`); `.killed`
/// marks a deliberate kill so a trailing termination callback can't overwrite it.
enum ProcessStatus: Equatable {
    case running
    case exited(Int32)
    case killed

    var isAlive: Bool { self == .running }
}

/// One agent-spawned background process. Reference type so the registry can flip
/// its `status` in place and republish `processes` without copying the retained
/// `Process` + pipes. Output is captured off-`@Published` into a lock-guarded
/// `ProcessOutputBuffer` — same rationale as `ServerManager`'s `logBuffer`: a
/// per-chunk publish would re-render `ChatView` during the SSE token loop.
@MainActor
final class ManagedProcess: Identifiable {
    /// A managed process is EITHER a real host `Process` (retained so status comes
    /// from `terminationHandler`, never `kill(pid,0)` — avoids PID-reuse ambiguity
    /// and zombie/fd leaks) OR a background command running INSIDE the Agent
    /// Sandbox guest, which owns no host `Process` — only a guest pid + the log
    /// file the guest appends its output to. The two share the same handle/status
    /// machinery so the chat card renders the running badge + kill X identically.
    enum Kind {
        case host(Process)
        case sandbox(logPath: String)
    }

    let handle: String          // monotonic "bg1", "bg2", …
    fileprivate(set) var pid: Int32   // host pid, or the GUEST pid for a sandbox entry
    let label: String           // first token of the command (for compact UI)
    let command: String
    let sessionId: UUID?
    let startedAt: Date
    fileprivate(set) var status: ProcessStatus
    fileprivate let kind: Kind
    let output: ProcessOutputBuffer

    nonisolated var id: String { handle }

    init(handle: String, pid: Int32, label: String, command: String,
         sessionId: UUID?, startedAt: Date, status: ProcessStatus,
         kind: Kind, output: ProcessOutputBuffer) {
        self.handle = handle
        self.pid = pid
        self.label = label
        self.command = command
        self.sessionId = sessionId
        self.startedAt = startedAt
        self.status = status
        self.kind = kind
        self.output = output
    }

    /// The retained host `Process`, or nil for a sandbox (guest-backed) entry.
    fileprivate var hostProcess: Process? {
        if case .host(let p) = kind { return p }
        return nil
    }

    /// True when this entry is a background command running inside the sandbox
    /// guest (killed via a guest `kill`, output tailed from a guest log).
    var isSandboxed: Bool {
        if case .sandbox = kind { return true }
        return false
    }

    /// The guest pid for a sandbox entry (nil for a host entry).
    var guestPID: Int32? { isSandboxed ? pid : nil }

    /// The guest log path a sandbox entry appends its output to (nil for host).
    var logPath: String? {
        if case .sandbox(let lp) = kind { return lp }
        return nil
    }

    var statusLabel: String {
        switch status {
        case .running: return "running"
        case .exited(let c): return "exited(\(c))"
        case .killed: return "killed"
        }
    }
}

/// Lock-guarded, bounded ring of a process's combined stdout/stderr. Modeled on
/// `ServerManager.ThrottledLogBuffer` — lives outside any actor so the
/// `FileHandle` readability handler (a background queue) can append without
/// hopping to the main actor on every chunk. `readNew()` adds cursor-based
/// incremental reads (BashOutput semantics) on top, with a dropped-bytes note
/// when the tail-trim discarded output the caller never saw.
///
/// `@unchecked Sendable` is honest: all shared state is guarded by `lock`.
final class ProcessOutputBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var content = ""        // retained tail (≤ maxBytes)
    private let maxBytes: Int
    private var droppedPrefix = 0   // logical chars trimmed off the front
    private var totalAppended = 0   // logical chars ever appended
    private var readCursor = 0      // logical position consumed by readNew()

    init(maxBytes: Int = 65_536) {
        self.maxBytes = maxBytes
    }

    func append(_ str: String) {
        guard !str.isEmpty else { return }
        lock.lock()
        defer { lock.unlock() }
        content += str
        totalAppended += str.count
        if content.count > maxBytes {
            let overflow = content.count - maxBytes
            content = String(content.suffix(maxBytes))
            droppedPrefix += overflow
        }
    }

    /// Append raw bytes (lossy-decoded as UTF-8) — the readability-handler path.
    func append(_ data: Data) {
        guard !data.isEmpty else { return }
        append(String(decoding: data, as: UTF8.self))
    }

    /// The full retained tail.
    func snapshot() -> String {
        lock.lock()
        defer { lock.unlock() }
        return content
    }

    /// Output appended since the previous `readNew()` (BashOutput semantics).
    /// When the ring trimmed past the cursor, prepends a note for the gap.
    func readNew() -> String {
        lock.lock()
        defer { lock.unlock() }
        var note = ""
        var effectiveCursor = readCursor
        if effectiveCursor < droppedPrefix {
            note = "[... \(droppedPrefix - effectiveCursor) bytes dropped]\n"
            effectiveCursor = droppedPrefix
        }
        let startInContent = effectiveCursor - droppedPrefix
        let new = startInContent >= content.count ? "" : String(content.dropFirst(startInContent))
        readCursor = totalAppended
        return note + new
    }
}

/// Single-use, lock-guarded out-parameter for a freshly created/adopted handle.
/// `ShellHandler` (a `Sendable` struct whose `execute` returns only a `String`)
/// writes the handle here so `AgentEngine.executeToolCall` can surface it
/// structurally on `ToolResult.backgroundHandle` — no string-parsing.
final class ProcessHandleBox: @unchecked Sendable {
    private let lock = NSLock()
    private var value: String?
    func set(_ h: String) { lock.lock(); value = h; lock.unlock() }
    var handle: String? { lock.lock(); defer { lock.unlock() }; return value }
}

/// Owns every agent-spawned background process. Modeled on `ServerManager`:
/// retained `Process`, streaming capture via `readabilityHandler` into a bounded
/// buffer, graceful `terminate()` → grace → `SIGKILL`, exit via
/// `terminationHandler`. Owned by `AppState`; reached by the agent loop (so
/// Telegram-driven turns get it too) and by the chat tool-call card's kill X.
@MainActor
final class ProcessRegistry: ObservableObject {
    /// Status-only snapshot for the UI. Output is intentionally NOT published —
    /// see `ProcessOutputBuffer` / `ServerManager.logBuffer`.
    @Published private(set) var processes: [ManagedProcess] = []

    private var counter = 0
    private var quitObserver: NSObjectProtocol?

    init() {
        // No AppDelegate / applicationWillTerminate exists in this app, and
        // `deinit` isn't guaranteed at process exit — so reap on the quit
        // notification. Delivered on the main thread (`queue: .main`).
        quitObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.killAll() }
        }
    }

    deinit {
        if let quitObserver { NotificationCenter.default.removeObserver(quitObserver) }
    }

    // MARK: - Shared process construction

    /// The exact spawn shape both `ShellHandler` and the registry use: a
    /// `/bin/zsh -l -c` login shell, stdin from `/dev/null` (an interactive
    /// prompt hits EOF instead of hanging), and a pipe each for stdout/stderr.
    nonisolated static func makeProcess(command: String, workingDirectory: String?) -> Process {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-l", "-c", command]
        if let wd = workingDirectory {
            process.currentDirectoryURL = URL(fileURLWithPath: wd)
        }
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        process.standardInput = FileHandle.nullDevice
        return process
    }

    /// First whitespace-delimited token of a command — the compact label/icon hint.
    nonisolated static func commandLabel(_ command: String) -> String {
        command.split(whereSeparator: { $0 == " " || $0 == "\t" || $0 == "\n" })
            .first.map(String.init) ?? command
    }

    // MARK: - Start / adopt

    /// Create, register, and launch a new background process. Returns immediately
    /// (the caller's tool call continues, the assistant keeps talking).
    func start(command: String, workingDirectory: String?, sessionId: UUID?) -> ManagedProcess {
        counter += 1
        let handle = "bg\(counter)"
        let process = Self.makeProcess(command: command, workingDirectory: workingDirectory)
        let buffer = ProcessOutputBuffer()
        let managed = ManagedProcess(
            handle: handle, pid: 0, label: Self.commandLabel(command), command: command,
            sessionId: sessionId, startedAt: Date(), status: .running,
            kind: .host(process), output: buffer)

        attachCapture(process: process, buffer: buffer)
        process.terminationHandler = { [weak self] proc in
            let code = proc.terminationStatus
            Task { @MainActor in self?.markExited(handle: handle, code: code) }
        }

        do {
            try process.run()
            managed.pid = process.processIdentifier
        } catch {
            buffer.append("failed to launch: \(error.localizedDescription)\n")
            managed.status = .exited(-1)
        }
        processes.append(managed)
        return managed
    }

    /// Adopt an already-running `Process` handed over by the foreground backstop
    /// (`ShellHandler` at its timeout). The caller has detached its own readers
    /// and snapshotted `priorOutput`; we seed the buffer, re-attach capture, and
    /// take over the termination callback. The real `Process` is retained so PID
    /// reuse can never confuse us.
    func adopt(process: Process, command: String, workingDirectory: String?,
               sessionId: UUID?, priorOutput: String) -> ManagedProcess {
        counter += 1
        let handle = "bg\(counter)"
        let buffer = ProcessOutputBuffer()
        buffer.append(priorOutput)
        let managed = ManagedProcess(
            handle: handle, pid: process.processIdentifier, label: Self.commandLabel(command),
            command: command, sessionId: sessionId, startedAt: Date(), status: .running,
            kind: .host(process), output: buffer)

        attachCapture(process: process, buffer: buffer)
        process.terminationHandler = { [weak self] proc in
            let code = proc.terminationStatus
            Task { @MainActor in self?.markExited(handle: handle, code: code) }
        }
        // It may have exited in the gap between the backstop's check and now.
        if !process.isRunning {
            managed.status = .exited(process.terminationStatus)
        }
        processes.append(managed)
        return managed
    }

    /// Register a background command running INSIDE the Agent Sandbox guest.
    /// There's no host `Process` to retain — the guest owns the real process; we
    /// track its guest pid (for a guest `kill`) and the log file it appends to
    /// (for `readProcessOutput` to tail). Returns immediately with a `bgN` handle
    /// so the chat card shows the SAME running badge + kill X as a host process.
    func registerSandboxed(command: String, guestPID: Int32, logPath: String,
                           sessionId: UUID?) -> ManagedProcess {
        counter += 1
        let handle = "bg\(counter)"
        let managed = ManagedProcess(
            handle: handle, pid: guestPID, label: Self.commandLabel(command), command: command,
            sessionId: sessionId, startedAt: Date(), status: .running,
            kind: .sandbox(logPath: logPath), output: ProcessOutputBuffer())
        processes.append(managed)
        return managed
    }

    // MARK: - Query

    func list(sessionId: UUID?) -> [ManagedProcess] {
        guard let sessionId else { return processes }
        return processes.filter { $0.sessionId == sessionId }
    }

    func isAlive(handle: String) -> Bool {
        processes.first { $0.handle == handle }?.status.isAlive ?? false
    }

    /// Incremental output since the last read, or nil for an unknown handle.
    func readOutput(handle: String) -> String? {
        processes.first { $0.handle == handle }?.output.readNew()
    }

    /// The guest log path for a sandbox-backed handle, or nil for a host handle /
    /// unknown handle. `readProcessOutput` uses it to tail the guest log (a host
    /// entry keeps its in-process capture buffer instead).
    func sandboxLogPath(handle: String) -> String? {
        processes.first { $0.handle == handle }?.logPath
    }

    // MARK: - Kill

    /// Graceful stop of the WHOLE process subtree. A tracked server often spawns
    /// children (`npm run dev` → node + esbuild; a `cd x && server` shell that
    /// stays the parent), and killing only the tracked pid orphans them. So we
    /// snapshot the live descendant set FIRST (while the tree is intact — the
    /// stripped-`&` start keeps the leader alive, so nothing has reparented yet),
    /// detach readers, flip status to `.killed`, SIGTERM the leader + every
    /// descendant, then SIGKILL any survivor after a bounded grace.
    func kill(handle: String) {
        guard let p = processes.first(where: { $0.handle == handle }), p.status.isAlive else { return }

        // Sandbox (guest-backed) entry: no host Process — route a `kill` INSIDE
        // the guest (SIGTERM → grace → SIGKILL) off the main thread. Flip status
        // synchronously so the card's badge + X clear at once (the guest kill is
        // fire-and-forget; if the guest is already gone the process is too).
        guard let proc = p.hostProcess else {
            objectWillChange.send()
            p.status = .killed
            let guestPID = p.pid
            DispatchQueue.global(qos: .utility).async {
                AgentSandbox.shared.killGuestProcess(pid: guestPID)
            }
            return
        }

        let leaderPid = proc.processIdentifier
        // Snapshot the subtree before any signal perturbs it.
        let tree = leaderPid > 0 ? [leaderPid] + Self.descendantPids(of: leaderPid) : []

        detachCapture(proc)
        proc.terminationHandler = nil   // status is now ours; don't let the trailing callback flip it
        objectWillChange.send()
        p.status = .killed

        guard !tree.isEmpty else { return }
        for pid in tree { Darwin.kill(pid, SIGTERM) }
        DispatchQueue.global().asyncAfter(deadline: .now() + 3) {
            for pid in tree where Darwin.kill(pid, 0) == 0 { Darwin.kill(pid, SIGKILL) }
        }
    }

    /// All transitive child pids of `pid` from one libproc snapshot. Used to tear
    /// down a server's whole subtree on kill. Best-effort: a process that has
    /// double-forked / re-parented away (a true daemon) escapes this, same as it
    /// would escape a process-group kill.
    ///
    /// Was `/bin/ps -axo pid=,ppid=`; `ps` is not reachable from inside the App
    /// Sandbox container. See `SystemMetrics.processParentMap`.
    nonisolated static func descendantPids(of pid: Int32) -> [Int32] {
        // Build parent → children adjacency, then BFS from `pid`.
        var children: [Int32: [Int32]] = [:]
        for (child, parent) in SystemMetrics.processParentMap() {
            children[parent, default: []].append(child)
        }
        var result: [Int32] = []
        var queue = children[pid] ?? []
        while let next = queue.first {
            queue.removeFirst()
            result.append(next)
            queue.append(contentsOf: children[next] ?? [])
        }
        return result
    }

    func killSession(_ id: UUID) {
        for p in processes where p.sessionId == id { kill(handle: p.handle) }
    }

    func killAll() {
        for p in processes { kill(handle: p.handle) }
    }

    // MARK: - Private

    private func attachCapture(process: Process, buffer: ProcessOutputBuffer) {
        (process.standardOutput as? Pipe)?.fileHandleForReading.readabilityHandler = { h in
            buffer.append(h.availableData)
        }
        (process.standardError as? Pipe)?.fileHandleForReading.readabilityHandler = { h in
            buffer.append(h.availableData)
        }
    }

    private func detachCapture(_ process: Process) {
        (process.standardOutput as? Pipe)?.fileHandleForReading.readabilityHandler = nil
        (process.standardError as? Pipe)?.fileHandleForReading.readabilityHandler = nil
    }

    private func markExited(handle: String, code: Int32) {
        guard let p = processes.first(where: { $0.handle == handle }), p.status.isAlive else { return }
        objectWillChange.send()
        p.status = .exited(code)
    }
}

/// Pure presentation seam for the chat tool-call card's kill controls: from the
/// handles persisted on the message and the registry's live `isAlive`, return
/// only the handles still worth showing an X for. Persisted handles from a prior
/// run resolve to no live process (the registry isn't persisted) → no buttons.
enum ProcessCardControls {
    static func killable(handles: [String]?, isAlive: (String) -> Bool) -> [String] {
        (handles ?? []).filter(isAlive)
    }
}
