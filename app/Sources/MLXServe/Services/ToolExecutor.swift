import Foundation

// MARK: - Error Type

enum ToolError: LocalizedError {
    case missingParameter(String)
    case executionFailed(String)
    case unsupportedTool(String)

    var errorDescription: String? {
        switch self {
        case .missingParameter(let p): "Missing parameter: \(p)"
        case .executionFailed(let msg): msg
        case .unsupportedTool(let t): "Unsupported tool: \(t)"
        }
    }
}

// MARK: - Handler Protocol

protocol ToolHandler: Sendable {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String
}

// MARK: - File-tool sandbox gate

/// With the Agent Sandbox ON, the five file tools stay HOST-side (the guest
/// shares the workspace via virtiofs, so in-workspace bytes are identical
/// either way — routing them through the VM would move the same bytes through
/// a shell pipe to the same place) — but they must honor the same confinement
/// story as `shell`: never in a folder the pinned VM isn't sharing. Live
/// 2026-07-20: writeFile wrote to the host workspace in the same turn where
/// shell was declined with the remount-block message. The companion rule —
/// a working directory is MANDATORY in every mode — lives at
/// `resolveAndConfine` (`workspaceRequiredMessage`), not here: it is not a
/// sandbox rule. browse/webSearch/saveMemory are not file tools and stay
/// ungated. A NEW file tool must take this gate and check it first —
/// FileToolSandboxGateTests pins the wiring for all five.
struct FileToolSandboxGate: Sendable {
    /// Read at execution time — the setting changes between calls. Injectable
    /// (same seam as `ShellHandler.sandboxEnabled`) so gate tests never depend
    /// on the build flavor (MAS forces the sandbox on).
    var sandboxEnabled: () -> Bool = { AgentSandbox.shared.isEnabled }
    /// The live pinned guest's shared root + CLI-session label. (nil, nil)
    /// when no live pinned guest exists — then shell would remount silently,
    /// so file tools must not block either.
    var pinnedWorkspace: () -> (root: String?, label: String?) = { AgentSandbox.shared.pinnedWorkspace }
    /// Test seam: force a rejection so each file tool's gate wiring stays
    /// provable now that production never blocks on the pin (hot-mount shares
    /// any folder). Nil in production.
    var forcedRejection: ((String?) -> String?)? = nil
    /// Every file tool hits this chokepoint, so it's where a host-side file op
    /// ensures its folder is hot-mounted into a LIVE guest — otherwise a session
    /// that only ever writes/reads files (never `shell`) never appears in the
    /// VM at /projects/<slug>. Fire-and-forget; injectable so tests don't touch
    /// the shared guest.
    var ensureMounted: (String?) -> Void = { AgentSandbox.shared.ensureProjectMountedAsync(workingDirectory: $0) }

    func check(workingDirectory: String?) throws {
        if let forced = forcedRejection, let reason = forced(workingDirectory) {
            throw ToolError.executionFailed(reason)
        }
        let pin = pinnedWorkspace()
        if let reason = Self.rejectReason(sandboxEnabled: sandboxEnabled(),
                                          workingDirectory: workingDirectory,
                                          pinnedRoot: pin.root, pinnedLabel: pin.label) {
            throw ToolError.executionFailed(reason)
        }
        ensureMounted(workingDirectory)
    }

    /// Pure verdict (unit-tested): nil = proceed. A chat's working folder is now
    /// hot-mounted into the guest at `/projects/<slug>` on first shell use — no
    /// VM reboot, so a CLI session pinned to `/workspace` no longer makes any
    /// folder unreachable. shell therefore never declines on the pin, so file
    /// tools don't either. Confinement + the mandatory-working-directory rule
    /// stay at `resolveAndConfine`; this always returns nil, retained as a
    /// stable seam for any future sandbox rule.
    static func rejectReason(sandboxEnabled: Bool, workingDirectory: String?,
                             pinnedRoot: String?, pinnedLabel: String?) -> String? {
        return nil
    }
}

// MARK: - Shell

struct ShellHandler: ToolHandler {
    /// Max wall-clock before a FOREGROUND command is dealt with. Long enough for
    /// real installs/builds (`npm install`), bounded so a hang can't stall the
    /// agent. Injectable so tests can use a short bound.
    var timeoutSeconds: Double = 120

    /// When present, `run_in_background:"true"` registers a managed process and
    /// returns instantly, and a foreground command still alive at the timeout is
    /// ADOPTED (never killed) instead of being terminated. Absent (e.g. older
    /// call sites / unit tests that don't inject one) → today's behavior:
    /// foreground only, killed at the timeout.
    var registry: ProcessRegistry? = nil
    /// The chat session the spawned process belongs to (for scoped cleanup).
    var sessionId: UUID? = nil
    /// Out-parameter for the handle of any process started/adopted by this call,
    /// so the caller can surface it on `ToolResult.backgroundHandle`.
    var handleBox: ProcessHandleBox? = nil

    /// Whether a host shell exists at all. False in the App Store build, where
    /// every command routes into the guest. Injectable so host-behavior tests
    /// can force the Developer ID path regardless of the build they run under.
    var hostShellAllowed: Bool = BuildFeatures.current.hostShell
    /// Whether the Agent Sandbox is on. A CLOSURE read at execution time (the
    /// setting changes between calls); injectable beside `hostShellAllowed`
    /// because the MAS build forces the sandbox on (`AgentSandbox.resolveEnabled`)
    /// — without the seam, host-behavior tests in the MAS test binary would
    /// route into a guest that xctest can never boot.
    var sandboxEnabled: () -> Bool = { AgentSandbox.shared.isEnabled }

    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        guard let command = parameters["command"] else {
            throw ToolError.missingParameter("command")
        }
        let cwd = workingDirectory ?? FileManager.default.currentDirectoryPath

        // Background detection is model-agnostic: either the explicit flag OR a
        // trailing `&` in the command. Small models won't set the flag — they
        // just write `python3 -m http.server &` — so we treat that identically.
        // The flag itself is read leniently because models send `true`, `"true"`,
        // `1`, `"1"`, `yes`, … (all stringify into arguments) — accepting only
        // the literal `"true"` silently dropped capable models into the 120s
        // foreground/adopt path. No prompt-following required; handled in code.
        let wantsBackground = Self.isTruthyFlag(parameters["run_in_background"])
        let hasTrailingAmp = Self.hasTrailingBackgroundOperator(command)

        switch Self.route(sandboxEnabled: sandboxEnabled(),
                          wantsBackground: wantsBackground,
                          hasTrailingAmp: hasTrailingAmp,
                          hasRegistry: registry != nil,
                          hostShellAllowed: hostShellAllowed) {
        case .hostBackground:
            return await startInBackground(command: command, cwd: cwd, workingDirectory: workingDirectory)
        case .hostBackgroundUnavailable:
            // Explicit flag but no registry wired → graceful error (preserves
            // the old contract). A bare `&` with no registry just runs
            // foreground as before — nothing to manage it, but no worse than today.
            return ShellMessages.backgroundUnavailable(cwd: cwd, seconds: Int(timeoutSeconds))
        case .sandboxBackground:
            // Sandbox ON + explicit background flag: the command must NOT reach
            // the host ProcessRegistry (that executes on the host, defeating the
            // isolation the user opted into). Background it INSIDE the guest —
            // detached from the shell channel, output appended to a
            // per-invocation log — then register it as a GUEST-backed managed
            // process so the chat card shows the SAME running badge + kill X and
            // readProcessOutput/killProcess work exactly like a host bg process.
            let logPath = Self.sandboxBackgroundLogPath()
            let stripped = Self.stripTrailingBackgroundOperator(command)
            let wrapped = Self.sandboxBackgroundCommand(stripped, logPath: logPath)
            let execOut = try await AgentSandbox.shared.runForeground(
                command: wrapped, workingDirectory: workingDirectory, timeout: timeoutSeconds)
            let pid = Self.parseSandboxBackgroundPID(execOut) ?? 0
            return await registerSandboxBackground(command: stripped, guestPID: pid,
                                                    logPath: logPath, cwd: cwd)
        case .sandboxForeground:
            // Sandbox ON: run inside the isolated Linux guest. A trailing `&`
            // rides through unchanged — the guest shell backgrounds it itself.
            // We do NOT fall back to the host on a sandbox error — the user
            // opted into isolation.
            return try await AgentSandbox.shared.runForeground(
                command: command, workingDirectory: workingDirectory, timeout: timeoutSeconds)
        case .hostForeground:
            return try await runForeground(command: command, cwd: cwd, workingDirectory: workingDirectory)
        }
    }

    /// Where a shell command executes. Pure decision (unit-tested) so the
    /// sandbox-vs-host-vs-background routing can't silently regress — the live
    /// bug was the host-background branch running BEFORE the sandbox check, so
    /// `run_in_background:"true"` escaped the guest onto the host.
    enum ShellRoute: Equatable {
        case hostBackground            // ProcessRegistry-managed host process
        case hostBackgroundUnavailable // explicit flag, no registry → graceful error
        case sandboxBackground         // backgrounded INSIDE the guest, log file
        case sandboxForeground         // normal guest path (guest shell owns any `&`)
        case hostForeground
    }

    static func route(sandboxEnabled: Bool, wantsBackground: Bool,
                      hasTrailingAmp: Bool, hasRegistry: Bool,
                      hostShellAllowed: Bool = BuildFeatures.current.hostShell) -> ShellRoute {
        // The App Store build has no host shell (App Review 2.5.2 + the sandbox
        // container can't reach /bin/zsh). Force every command into the guest,
        // regardless of the user's sandbox toggle.
        if sandboxEnabled || !hostShellAllowed {
            // NEVER the host registry while sandboxed — regardless of registry.
            // A bare trailing `&` is background intent, exactly as on the host
            // path below. It must NOT run foreground: under the vsock exec
            // transport the orphaned process holds the shell's stdout pipe, so
            // a foreground `cmd &` would sit until the timeout kills the call
            // (the legacy console shell happened to return promptly, which is
            // why this ever routed foreground).
            return (wantsBackground || hasTrailingAmp) ? .sandboxBackground : .sandboxForeground
        }
        if hasRegistry, wantsBackground || hasTrailingAmp { return .hostBackground }
        if wantsBackground { return .hostBackgroundUnavailable }
        return .hostForeground
    }

    /// Wrap a command so the GUEST shell backgrounds it: detached from stdin
    /// (the shell channel) with all output appended to `logPath` inside the
    /// guest, so the foreground exec returns immediately and the sentinel
    /// framing stays clean. Echoes the backgrounded job's guest pid on a marker
    /// line (`__CTN_BGPID=$!`) so the tool can track + kill it like a host bg
    /// process (parsed back out by `parseSandboxBackgroundPID`).
    static func sandboxBackgroundCommand(_ command: String, logPath: String) -> String {
        "(\(command)) </dev/null >>\(logPath) 2>&1 & echo __CTN_BGPID=$!"
    }

    /// Marker prefix the background wrapper echoes the guest pid on.
    static let sandboxBGPIDMarker = "__CTN_BGPID="

    /// Pull the guest pid out of the background-launch exec output (the
    /// `__CTN_BGPID=<pid>` marker line). nil when absent or unparsable — the
    /// process is still tracked, just without a pid for a guest `kill`.
    static func parseSandboxBackgroundPID(_ output: String) -> Int32? {
        guard let range = output.range(of: sandboxBGPIDMarker) else { return nil }
        let digits = output[range.upperBound...].prefix { $0.isNumber }
        return Int32(digits)
    }

    /// Per-invocation guest log path (millisecond timestamp) so concurrent /
    /// repeated background commands never interleave into one file.
    static func sandboxBackgroundLogPath(now: Date = Date()) -> String {
        "/tmp/mlx-bg-\(UInt64(now.timeIntervalSince1970 * 1000)).log"
    }

    /// Register a long-lived command with the process registry and return at once
    /// so the assistant keeps talking while it runs.
    private func startInBackground(command: String, cwd: String, workingDirectory: String?) async -> String {
        guard let registry else {
            return ShellMessages.backgroundUnavailable(cwd: cwd, seconds: Int(timeoutSeconds))
        }
        // Strip a trailing `&`: backgrounding INSIDE the login shell makes the
        // shell exit immediately, so the tracked pid becomes a dead shell while
        // the real server is orphaned beyond the registry's reach (killProcess /
        // kill X / session+quit cleanup all silently miss it). Stripping it lets
        // the shell exec-replace into the process so the tracked pid IS the
        // server (and child cleanup walks the live subtree — see ProcessRegistry).
        let cleaned = Self.stripTrailingBackgroundOperator(command)
        let info: (handle: String, pid: Int32) = await MainActor.run {
            let managed = registry.start(command: cleaned, workingDirectory: workingDirectory, sessionId: sessionId)
            return (managed.handle, managed.pid)
        }
        handleBox?.set(info.handle)
        return ShellMessages.started(cwd: cwd, handle: info.handle, pid: info.pid)
    }

    /// Register a guest-backed background process with the registry, surface its
    /// handle on `handleBox` (so the card renders the running badge + kill X), and
    /// shape the start message. Split out of the `.sandboxBackground` branch so it
    /// is testable without a live guest: given the parsed guest pid it does the
    /// registration + messaging. No registry (older call sites / unit tests) →
    /// today's log-only message, no handle.
    func registerSandboxBackground(command: String, guestPID: Int32,
                                   logPath: String, cwd: String) async -> String {
        guard let registry else {
            return ShellMessages.sandboxBackgroundStarted(cwd: cwd, handle: nil,
                                                          logPath: logPath, pid: guestPID)
        }
        let handle = await MainActor.run {
            registry.registerSandboxed(command: command, guestPID: guestPID,
                                       logPath: logPath, sessionId: sessionId).handle
        }
        handleBox?.set(handle)
        return ShellMessages.sandboxBackgroundStarted(cwd: cwd, handle: handle,
                                                      logPath: logPath, pid: guestPID)
    }

    /// Lenient truthy read for a string-typed boolean tool flag. Tool arguments
    /// are stringified, and models emit `true`/`1`/`yes`/`on` (any casing) — so
    /// the flag must accept all of them, not just the literal `"true"`.
    static func isTruthyFlag(_ value: String?) -> Bool {
        guard let v = value?.trimmingCharacters(in: .whitespaces).lowercased() else { return false }
        return v == "true" || v == "1" || v == "yes" || v == "y" || v == "on"
    }

    /// True when the command ends with a single shell background operator (`… &`)
    /// — the model's "run this in the background" signal, with or without the
    /// `run_in_background` flag. `&&` (logical-AND) is not a background operator.
    static func hasTrailingBackgroundOperator(_ command: String) -> Bool {
        let trimmed = command.trimmingCharacters(in: .whitespaces)
        return trimmed.hasSuffix("&") && !trimmed.hasSuffix("&&")
    }

    /// Drop a single trailing shell background operator (`… &`). Leaves `&&` and
    /// inner `&` untouched. Pure + testable.
    static func stripTrailingBackgroundOperator(_ command: String) -> String {
        guard hasTrailingBackgroundOperator(command) else { return command }
        let trimmed = command.trimmingCharacters(in: .whitespaces)
        return String(trimmed.dropLast()).trimmingCharacters(in: .whitespaces)
    }

    /// Foreground execution with the timeout backstop. A command still alive at
    /// the timeout is adopted by the registry (never killed) when one is present;
    /// with no registry it's killed and reported — today's behavior.
    private func runForeground(command: String, cwd: String, workingDirectory: String?) async throws -> String {
        let timeout = timeoutSeconds
        let registry = self.registry
        let sessionId = self.sessionId
        let handleBox = self.handleBox

        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, Error>) in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = ProcessRegistry.makeProcess(command: command, workingDirectory: workingDirectory)
                let outPipe = process.standardOutput as! Pipe
                let errPipe = process.standardError as! Pipe

                // Read both pipes incrementally so a kill never blocks on
                // `readDataToEndOfFile` waiting for orphaned grandchildren
                // (e.g. node spawned by npx) to release the write end.
                let cap = ShellCapture()
                outPipe.fileHandleForReading.readabilityHandler = { h in cap.appendOut(h.availableData) }
                errPipe.fileHandleForReading.readabilityHandler = { h in cap.appendErr(h.availableData) }

                // Signal exit via terminationHandler + a bounded semaphore wait
                // (NOT waitUntilExit) so a never-exiting server releases this
                // worker thread at the timeout instead of leaking it forever.
                let exitSem = DispatchSemaphore(value: 0)
                process.terminationHandler = { _ in exitSem.signal() }

                do {
                    try process.run()
                } catch {
                    continuation.resume(throwing: error)
                    return
                }
                let pid = process.processIdentifier

                // Single-resume guard: the adopt branch resumes from a hopped
                // MainActor task, so a near-simultaneous real exit must not
                // double-resume the continuation.
                let resumed = ManagedAtomicFlag()
                let finish: (String) -> Void = { s in
                    if resumed.testAndSet() { continuation.resume(returning: s) }
                }

                let timedOut = exitSem.wait(timeout: .now() + timeout) == .timedOut

                // Raced exit: the wait timed out but the process actually
                // finished right at the deadline → fall through to clean completion.
                if timedOut && process.isRunning {
                    outPipe.fileHandleForReading.readabilityHandler = nil
                    errPipe.fileHandleForReading.readabilityHandler = nil
                    if let registry {
                        let prior = cap.combined()
                        Task { @MainActor in
                            let managed = registry.adopt(process: process, command: command,
                                                         workingDirectory: workingDirectory,
                                                         sessionId: sessionId, priorOutput: prior)
                            handleBox?.set(managed.handle)
                            finish(ShellMessages.adopted(cwd: cwd, handle: managed.handle,
                                                         pid: managed.pid, seconds: Int(timeout)))
                        }
                    } else {
                        process.terminate()
                        kill(pid, SIGKILL)
                        finish(ShellMessages.timedOutKilled(cwd: cwd, seconds: Int(timeout),
                                                            body: cap.foregroundBody()))
                    }
                    return
                }

                // Clean exit (or raced exit at the deadline).
                process.terminationHandler = nil
                outPipe.fileHandleForReading.readabilityHandler = nil
                errPipe.fileHandleForReading.readabilityHandler = nil
                // On a clean exit the pipe EOFs promptly, so drain any final
                // chunk the handler hadn't delivered yet.
                cap.appendOut(try? outPipe.fileHandleForReading.readToEnd())
                cap.appendErr(try? errPipe.fileHandleForReading.readToEnd())

                finish(ShellMessages.completed(cwd: cwd, body: cap.foregroundBody(),
                                               exitCode: process.terminationStatus))
            }
        }
    }
}

/// Thread-safe stdout/stderr accumulator for the foreground shell path. Keeps the
/// readability-handler closures trivial and the message-shaping logic out of the
/// large `runForeground` body the type-checker has to chew through.
final class ShellCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var out = Data()
    private var err = Data()

    func appendOut(_ d: Data?) { guard let d, !d.isEmpty else { return }; lock.lock(); out.append(d); lock.unlock() }
    func appendErr(_ d: Data?) { guard let d, !d.isEmpty else { return }; lock.lock(); err.append(d); lock.unlock() }

    private func snapshot() -> (String, String) {
        lock.lock(); defer { lock.unlock() }
        return (String(data: out, encoding: .utf8) ?? "", String(data: err, encoding: .utf8) ?? "")
    }

    /// stdout followed by stderr — what the registry seeds an adopted process with.
    func combined() -> String {
        let (o, e) = snapshot()
        return e.isEmpty ? o : o + e
    }

    /// stdout with a trailing `[stderr]: …` section — the foreground result shape.
    func foregroundBody() -> String {
        let (o, e) = snapshot()
        return e.isEmpty ? o : o + "\n[stderr]: \(e)"
    }
}

/// Shell tool result strings. Pulled out of the handler so each long interpolated
/// message is type-checked as an isolated, trivial `String` return instead of
/// inside the nested closures of `runForeground` — the latter caused a
/// pathological (multi-minute) type-check of `ShellHandler.execute`.
enum ShellMessages {
    static func started(cwd: String, handle: String, pid: Int32) -> String {
        "[cwd: \(cwd)]\nStarted in background as \(handle) (pid \(pid)). It keeps running — poll it with readProcessOutput {\"handle\": \"\(handle)\"}, stop it with killProcess {\"handle\": \"\(handle)\"}."
    }

    /// The trailing URL steer is load-bearing: the model composes its "server
    /// is up at <url>" reply straight from this result. Live 2026-07-02 an
    /// agent handed the user the Mac's LAN IP (which the loopback-only port
    /// map can never serve) — the base prompt's `http://<local-ip>:<port>`
    /// directive won over the env section's localhost hint. That prompt
    /// conflict is fixed in AgentPrompt; this steer is the belt-and-braces
    /// layer closest to where the model writes the URL.
    static func sandboxBackgroundStarted(cwd: String, handle: String?, logPath: String, pid: Int32) -> String {
        let urlSteer = "If this is a server, every TCP port it listens on inside the sandbox is auto-mapped to the host — the user reaches it at http://localhost:<port> in their browser. When sharing a URL, ALWAYS use http://localhost:<port>, never a LAN or guest IP address."
        if let handle {
            return "[cwd: \(cwd)]\nStarted in the SANDBOX background (isolated Linux guest) as \(handle) (guest pid \(pid)). It keeps running — poll it with readProcessOutput {\"handle\": \"\(handle)\"}, stop it with killProcess {\"handle\": \"\(handle)\"}. Its output is also appended to \(logPath) inside the guest. \(urlSteer)"
        }
        return "[cwd: \(cwd)]\nStarted in the SANDBOX background (isolated Linux guest). Output is appended to \(logPath) inside the guest — check on it with the shell tool, e.g. {\"command\": \"tail -n 50 \(logPath)\"}. \(urlSteer)"
    }

    static func backgroundUnavailable(cwd: String, seconds: Int) -> String {
        "[cwd: \(cwd)]\nError: background execution isn't available in this context. Run the command in the foreground, or it will be stopped at the \(seconds)s timeout."
    }

    static func adopted(cwd: String, handle: String, pid: Int32, seconds: Int) -> String {
        "[cwd: \(cwd)]\nStill running after \(seconds)s — now managed in the background as \(handle) (pid \(pid)), NOT killed. Poll it with readProcessOutput {\"handle\": \"\(handle)\"}, stop it with killProcess {\"handle\": \"\(handle)\"}."
    }

    static func timedOutKilled(cwd: String, seconds: Int, body: String) -> String {
        let note = "[timed out after \(seconds)s and was killed. If this command waits for input, re-run it with non-interactive flags — it cannot read stdin. If it is a long-running server, start it with run_in_background:\"true\".]"
        return "[cwd: \(cwd)]\n\(body)\n\(note)"
    }

    static func completed(cwd: String, body: String, exitCode: Int32) -> String {
        var result = body
        if exitCode != 0 {
            result += "\n[exit code: \(exitCode)]"
        } else if result.isEmpty {
            result = "OK"
        }
        return "[cwd: \(cwd)]\n\(result)"
    }
}

/// Minimal thread-safe test-and-set flag — guards the shell continuation against
/// a double resume when the adopt branch and a near-simultaneous real exit race.
final class ManagedAtomicFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var set = false
    /// Returns true exactly once (the first caller); false thereafter.
    func testAndSet() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if set { return false }
        set = true
        return true
    }
}

// MARK: - Read File

struct ReadFileHandler: ToolHandler {
    var gate = FileToolSandboxGate()

    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        try gate.check(workingDirectory: workingDirectory)
        guard let path = parameters["path"] else {
            throw ToolError.missingParameter("path")
        }

        let fullPath = try resolveAndConfine(path, workingDirectory: workingDirectory)
        guard let data = FileManager.default.contents(atPath: fullPath),
              let content = String(data: data, encoding: .utf8) else {
            throw ToolError.executionFailed("Cannot read file: \(fullPath)")
        }

        let lines = content.components(separatedBy: "\n")
        let totalLines = lines.count
        let startLine = Int(parameters["startLine"] ?? "1") ?? 1
        let endLine = Int(parameters["endLine"] ?? "\(totalLines)") ?? totalLines
        let actualStart = max(1, startLine)
        let actualEnd = min(totalLines, endLine)

        guard actualStart <= actualEnd else {
            return "Invalid line range: \(startLine)-\(endLine) (file has \(totalLines) lines)"
        }

        let slice = lines[actualStart - 1..<actualEnd]
        // Add line numbers so the model can reference specific lines for editFile
        var numbered = slice.enumerated().map { (i, line) in
            "\(actualStart + i)| \(line)"
        }.joined(separator: "\n")

        // Add metadata header for large files so model knows to use line ranges
        if totalLines > 200 || content.utf8.count > 6000 {
            let header = "[File: \(path) | Lines: \(actualStart)-\(actualEnd) of \(totalLines) | \(content.utf8.count) bytes"
            if actualEnd < totalLines {
                numbered = header + " | Use startLine/endLine to read more]\n" + numbered
            } else {
                numbered = header + "]\n" + numbered
            }
        }

        return numbered
    }
}

// MARK: - Write File

struct WriteFileHandler: ToolHandler {
    /// Tolerant boolean for the `append` flag. Models emit it dirty — `"true"`,
    /// `"true,"` (gemma-4-12b adds a trailing comma), `"True"`, even a leftover
    /// `"true,\n…"` before normalization peels the body off. Treat any value
    /// whose first token is `true` (or `1`/`yes`) as append; an exact `== "true"`
    /// match silently OVERWROTE the file when the model sent `"true,"`.
    static func appendFlagIsTrue(_ raw: String?) -> Bool {
        guard let raw else { return false }
        let v = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if v == "true" || v == "1" || v == "yes" { return true }
        // "true" followed immediately by a non-letter (comma/space/newline) →
        // still append; "truely"/"truthy" (next char a letter) → not append.
        if v.hasPrefix("true"), let after = v.dropFirst(4).first, !after.isLetter { return true }
        return false
    }

    var gate = FileToolSandboxGate()

    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        try gate.check(workingDirectory: workingDirectory)
        guard let path = parameters["path"], let content = parameters["content"] else {
            throw ToolError.missingParameter("path and content")
        }
        // append:true grows a file incrementally — the safe way to write a large
        // file across multiple tool calls without any one call overrunning the
        // token budget and getting truncated mid-write (see the writeFile tool
        // description). Default is overwrite, matching prior behavior. Parsed
        // tolerantly (see appendFlagIsTrue) — models dirty the flag value.
        let append = Self.appendFlagIsTrue(parameters["append"])

        let fullPath = try resolveAndConfine(path, workingDirectory: workingDirectory)
        let dir = (fullPath as NSString).deletingLastPathComponent
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)

        if append, FileManager.default.fileExists(atPath: fullPath) {
            let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: fullPath))
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: Data(content.utf8))
            return "Appended \(content.count) characters to \(path)"
        }
        // Overwrite, or create the file when appending to one that doesn't exist yet.
        try content.write(toFile: fullPath, atomically: true, encoding: .utf8)
        return "Wrote \(content.count) characters to \(path)\(append ? " (new file)" : "")"
    }
}

// MARK: - Edit File

struct EditFileHandler: ToolHandler {
    var gate = FileToolSandboxGate()

    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        try gate.check(workingDirectory: workingDirectory)
        guard let path = parameters["path"] else {
            throw ToolError.missingParameter("path")
        }

        let fullPath = try resolveAndConfine(path, workingDirectory: workingDirectory)
        guard let data = FileManager.default.contents(atPath: fullPath),
              var content = String(data: data, encoding: .utf8) else {
            throw ToolError.executionFailed("Cannot read file: \(fullPath)")
        }

        // Line-number-based editing: startLine/endLine + replace
        if let startStr = parameters["startLine"], let startLine = Self.parseLineNumber(startStr) {
            guard let replace = parameters["replace"] else {
                throw ToolError.executionFailed("editFile with startLine/endLine requires 'replace' parameter. You sent startLine=\(startStr) but no replace content. Example: {\"path\": \"file.js\", \"startLine\": \"5\", \"endLine\": \"8\", \"replace\": \"new code\"}")
            }
            let lines = content.components(separatedBy: "\n")
            let endLine = Self.parseLineNumber(parameters["endLine"]) ?? startLine
            let actualStart = max(1, startLine)
            let actualEnd = min(lines.count, endLine)

            guard actualStart <= actualEnd else {
                throw ToolError.executionFailed("Invalid line range: \(startLine)-\(endLine) (file has \(lines.count) lines)")
            }

            var newLines = Array(lines[0..<(actualStart - 1)])
            newLines.append(contentsOf: replace.components(separatedBy: "\n"))
            if actualEnd < lines.count {
                newLines.append(contentsOf: lines[actualEnd...])
            }
            content = newLines.joined(separator: "\n")
            try content.write(toFile: fullPath, atomically: true, encoding: .utf8)
            return "Edited \(path) (replaced lines \(actualStart)-\(actualEnd))"
        }

        // Text-based editing: find + replace
        guard let find = parameters["find"], !find.isEmpty else {
            // Name the gap relative to what WAS sent: a model that dropped only
            // startLine (live: 12 identical retries) needs "add startLine to the
            // call you just made", not a restatement of the two modes.
            if let endStr = parameters["endLine"], !endStr.isEmpty {
                throw ToolError.executionFailed("editFile line-based mode needs BOTH startLine and endLine. You sent endLine=\(endStr) but no startLine — resend the same call with startLine added (the first line to replace, from readFile). Example: {\"path\": \"\(path)\", \"startLine\": \"45\", \"endLine\": \"\(endStr)\", \"replace\": \"new code\"}")
            }
            throw ToolError.missingParameter("Either 'find' (exact text to replace) or 'startLine'+'endLine' (line numbers from readFile) is required")
        }
        let replace = parameters["replace"] ?? ""

        guard content.contains(find) else {
            // Show nearby content to help the model correct its find pattern
            let lines = content.components(separatedBy: "\n")
            let preview = lines.prefix(10).enumerated()
                .map { "\($0.offset + 1)| \($0.element)" }.joined(separator: "\n")
            throw ToolError.executionFailed(
                "Pattern not found in \(path). Use readFile first to see exact content, or use startLine/endLine for line-based editing. First 10 lines:\n\(preview)"
            )
        }

        content = content.replacingOccurrences(of: find, with: replace)
        try content.write(toFile: fullPath, atomically: true, encoding: .utf8)
        return "Edited \(path)"
    }

    /// Tolerant line-number parse: "45", "45,", " 45 " all read as 45. A weak
    /// model's dirty value must not demote a line-based edit into the find
    /// branch (startLine) or silently collapse the range (endLine) — the same
    /// class as WriteFileHandler.appendFlagIsTrue's tolerant flag parse.
    static func parseLineNumber(_ value: String?) -> Int? {
        guard let value else { return nil }
        let digits = value.drop(while: { !$0.isNumber }).prefix(while: { $0.isNumber })
        return digits.isEmpty ? nil : Int(digits)
    }
}

// MARK: - Search Files

struct SearchFilesHandler: ToolHandler {
    var gate = FileToolSandboxGate()

    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        try gate.check(workingDirectory: workingDirectory)
        guard let pattern = parameters["pattern"] else {
            throw ToolError.missingParameter("pattern")
        }
        let path = parameters["path"] ?? "."
        // Confine search path to workspace
        let confinedPath = try resolveAndConfine(path, workingDirectory: workingDirectory)

        let options = FileSearcher.Options(
            root: confinedPath,
            pattern: pattern,
            include: parameters["include"],
            contextLines: Int(parameters["context"] ?? "0") ?? 0,
            maxResults: Int(parameters["maxResults"] ?? "100") ?? 100
        )
        // The walk is synchronous and can touch thousands of files; `execute`
        // inherits the caller's actor (ToolExecutor is @MainActor), so hop off
        // it or a large workspace freezes the UI mid-search.
        let lines = await Task.detached(priority: .userInitiated) {
            FileSearcher.search(options)
        }.value
        return FileSearcher.render(lines, pattern: pattern)
    }
}


// MARK: - List Files

struct ListFilesHandler: ToolHandler {
    var gate = FileToolSandboxGate()

    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        try gate.check(workingDirectory: workingDirectory)
        let path = parameters["path"] ?? "."
        let fullPath = try resolveAndConfine(path, workingDirectory: workingDirectory)
        let pattern = parameters["pattern"]
        let recursive = parameters["recursive"]?.lowercased() == "true"

        let fm = FileManager.default
        guard fm.fileExists(atPath: fullPath) else {
            throw ToolError.executionFailed("Directory not found: \(fullPath)")
        }

        var entries: [String] = []

        if recursive {
            guard let enumerator = fm.enumerator(atPath: fullPath) else {
                throw ToolError.executionFailed("Cannot enumerate: \(fullPath)")
            }
            // If pattern has no path separators (e.g. "*.swift"), match filename only
            let matchFilenameOnly = pattern != nil && !pattern!.contains("/")
            while let item = enumerator.nextObject() as? String {
                if let pattern {
                    let target = matchFilenameOnly ? (item as NSString).lastPathComponent : item
                    if matchesGlob(target, pattern: pattern) {
                        entries.append(item)
                    }
                } else {
                    entries.append(item)
                }
                if entries.count >= 200 { break }
            }
        } else {
            let items = try fm.contentsOfDirectory(atPath: fullPath)
            for item in items.sorted() {
                if let pattern {
                    if matchesGlob(item, pattern: pattern) {
                        entries.append(item)
                    }
                } else {
                    entries.append(item)
                }
                if entries.count >= 200 { break }
            }
        }

        if entries.isEmpty {
            return "No files found in \(path)" + (pattern != nil ? " matching '\(pattern!)'" : "")
        }
        let result = entries.joined(separator: "\n")
        let suffix = entries.count >= 200 ? "\n[... truncated at 200 entries]" : ""
        return result + suffix
    }

    /// Simple glob matching supporting * and ** wildcards. Shared with
    /// `FileSearcher`'s `include` filter — see `Glob`.
    private func matchesGlob(_ path: String, pattern: String) -> Bool {
        Glob.matches(path, pattern: pattern)
    }
}

// MARK: - Browser tool timeout

/// Hard ceiling on any single browser-tool invocation. The inner WKWebView calls
/// already have their own navigate/evaluateJS timeouts, but a page can still
/// hang in ways that freeze the agent loop — this guarantees the tool returns
/// (with an error) within a bounded window so the loop keeps moving.
private let browserToolTimeoutSeconds: UInt64 = 30

private func withBrowserToolTimeout<T: Sendable>(
    _ description: String,
    _ body: @escaping @Sendable () async throws -> T
) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask { try await body() }
        group.addTask {
            try await Task.sleep(nanoseconds: browserToolTimeoutSeconds * 1_000_000_000)
            throw ToolError.executionFailed("\(description) timed out after \(browserToolTimeoutSeconds)s")
        }
        let result = try await group.next()!
        group.cancelAll()
        return result
    }
}

// MARK: - Web Search (DuckDuckGo)

struct WebSearchHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        guard let query = parameters["query"], !query.isEmpty else {
            throw ToolError.missingParameter("query")
        }
        let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query
        let url = "https://html.duckduckgo.com/html/?q=\(encoded)"

        return try await withBrowserToolTimeout("webSearch") {
            let browser = await BrowserManager.shared
            _ = try await browser.navigate(to: url)
            // Wait for search results to render
            try await Task.sleep(nanoseconds: 500_000_000)

            // Extract structured search results instead of raw page text
            let js = """
            (function() {
                var results = [];
                var links = document.querySelectorAll('.result__a, .result__title a, a.result-link');
                if (links.length === 0) links = document.querySelectorAll('a[href*="//"]');
                var seen = new Set();
                for (var i = 0; i < Math.min(links.length, 8); i++) {
                    var a = links[i];
                    var href = a.href || '';
                    if (href.includes('duckduckgo.com') || seen.has(href)) continue;
                    seen.add(href);
                    var title = (a.textContent || '').trim();
                    var snippet = '';
                    var parent = a.closest('.result') || a.closest('.web-result') || a.parentElement;
                    if (parent) {
                        var snipEl = parent.querySelector('.result__snippet, .result-snippet');
                        if (snipEl) snippet = snipEl.textContent.trim();
                    }
                    if (title && href) results.push(title + '\\n' + href + (snippet ? '\\n' + snippet : ''));
                }
                return results.length > 0 ? results.join('\\n\\n') : document.body.innerText.substring(0, 2000);
            })()
            """
            let result = try await browser.evaluateJS(js)
            return "Search results for '\(query)':\n\n\(result)"
        }
    }
}

// MARK: - Built-in Browser (WKWebView)

struct BrowseHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        let action = parameters["action"] ?? "navigate"

        return try await withBrowserToolTimeout("browse(\(action))") {
            let browser = await BrowserManager.shared

            switch action {
            case "navigate":
                guard let url = parameters["url"] else { throw ToolError.missingParameter("url") }
                return try await browser.navigate(to: url)
            case "readText":
                // Navigate to URL first if provided, then read text
                if let url = parameters["url"] {
                    _ = try await browser.navigate(to: url)
                }
                return try await browser.readText()
            case "readHTML":
                if let url = parameters["url"] {
                    _ = try await browser.navigate(to: url)
                }
                return try await browser.readHTML()
            case "extractText":
                if let url = parameters["url"] { _ = try await browser.navigate(to: url) }
                guard let selector = parameters["selector"] else { throw ToolError.missingParameter("selector") }
                return try await browser.extractText(selector: selector)
            case "click":
                guard let selector = parameters["selector"] else { throw ToolError.missingParameter("selector") }
                return try await browser.click(selector: selector)
            case "screenshot":
                if let url = parameters["url"] {
                    _ = try await browser.navigate(to: url)
                }
                guard let data = await browser.takeScreenshot() else {
                    return "Failed to capture screenshot"
                }
                let base64 = data.base64EncodedString()
                return "[screenshot:\(data.count) bytes]\ndata:image/jpeg;base64,\(base64)"
            case "getInfo":
                return try await browser.getInfo()
            case "executeJS":
                guard let script = parameters["script"] ?? parameters["expression"] else {
                    throw ToolError.missingParameter("script")
                }
                return try await browser.evaluateJS(script)
            default:
                // Fallback: if URL is present, treat as navigate
                if let url = parameters["url"] {
                    return try await browser.navigate(to: url)
                }
                return try await browser.readText()
            }
        }
    }
}


// MARK: - Save Memory

struct SaveMemoryHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        guard let memory = parameters["memory"] else {
            throw ToolError.missingParameter("memory")
        }
        AgentPrompt.saveMemory(memory)
        return "OK"
    }
}

// MARK: - Shared Helpers

private func resolvePath(_ path: String, workingDirectory: String?) -> String {
    if path.hasPrefix("/") || path.hasPrefix("~") {
        return NSString(string: path).expandingTildeInPath
    }
    if let wd = workingDirectory {
        return (wd as NSString).appendingPathComponent(path)
    }
    return path
}

/// A file tool ran without a working directory. MANDATORY in every mode since
/// 2026-07-20 — the old contract returned the path UNCONFINED here (the yolo
/// lever), which meant "sandbox on" never confined file tools for a nil-wd
/// run. Yolo task runs now anchor at the default agent workspace instead
/// (TaskScheduler.workDir); yolo keeps its meaning at the APPROVAL layer only.
let workspaceRequiredMessage = "no working folder is set for this session — the agent's file tools always run confined to one, in every mode. Set a working folder (the folder icon on the Agent pill, or Settings → Agent Sandbox) and retry."

/// Resolve a path and verify it stays within the working directory.
/// Returns the resolved absolute path, or throws if it escapes the workspace
/// — or when no workspace is set at all (never unconfined).
private func resolveAndConfine(_ path: String, workingDirectory: String?) throws -> String {
    guard let wd = workingDirectory else {
        throw ToolError.executionFailed(workspaceRequiredMessage)
    }
    let resolved: String
    if path.hasPrefix("/") || path.hasPrefix("~") {
        resolved = NSString(string: path).expandingTildeInPath
    } else {
        resolved = (wd as NSString).appendingPathComponent(path)
    }

    // Normalize to resolve ".." and symlinks for accurate prefix check
    let normalizedResolved = (resolved as NSString).standardizingPath
    let normalizedWd = (wd as NSString).standardizingPath

    guard normalizedResolved == normalizedWd || normalizedResolved.hasPrefix(normalizedWd + "/") else {
        throw ToolError.executionFailed("Access denied: path '\(path)' resolves to '\(normalizedResolved)' which is outside the workspace '\(normalizedWd)'")
    }

    return normalizedResolved
}

// MARK: - Executor

@MainActor
class ToolExecutor: ObservableObject {
    @Published var currentStepIndex: Int?
    @Published var results: [StepResult] = []
    @Published var isExecuting = false

    private let handlers: [AgentToolKind: any ToolHandler] = [
        .shell: ShellHandler(),
        .readFile: ReadFileHandler(),
        .writeFile: WriteFileHandler(),
        .editFile: EditFileHandler(),
        .searchFiles: SearchFilesHandler(),
        .listFiles: ListFilesHandler(),
        .browse: BrowseHandler(),
        .webSearch: WebSearchHandler(),
        .saveMemory: SaveMemoryHandler(),
    ]

    func executePlan(_ plan: AgentPlan, workingDirectory: String?) async -> [StepResult] {
        isExecuting = true
        results = []

        for (index, step) in plan.steps.enumerated() {
            currentStepIndex = index
            let start = DispatchTime.now()

            do {
                // Smart fallback: if editFile is called with content but no find/replace, use writeFile
                let effectiveTool: AgentToolKind
                if step.tool == .editFile && step.parameters["content"] != nil && step.parameters["find"] == nil {
                    effectiveTool = .writeFile
                } else {
                    effectiveTool = step.tool
                }
                guard let handler = handlers[effectiveTool] else {
                    throw ToolError.unsupportedTool(step.tool.rawValue)
                }
                let output = try await handler.execute(parameters: step.parameters, workingDirectory: workingDirectory)
                let elapsed = Int64((DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000)
                results.append(StepResult(stepId: step.id, status: .success, output: output, durationMs: elapsed))
            } catch {
                let elapsed = Int64((DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000)
                results.append(StepResult(stepId: step.id, status: .failed, output: "", error: error.localizedDescription, durationMs: elapsed))
            }
        }

        currentStepIndex = nil
        isExecuting = false
        return results
    }
}
