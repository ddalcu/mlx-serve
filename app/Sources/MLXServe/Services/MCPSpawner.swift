import Darwin
import Foundation
import System

/// Where a stdio MCP server actually runs.
protocol MCPSpawner: Sendable {
    /// Preflight: is `command` runnable where this spawner runs things?
    func commandExists(_ command: String) async -> Bool
    /// Start the server and hand back file descriptors a `StdioTransport` can use.
    func open(command: String, args: [String], cwd: String, env: [String: String]) async throws -> MCPStdio
}

/// The running server, however it runs. `MCPManager` only ever asks whether it
/// is alive, why it died, and tells it to stop.
protocol MCPChild: AnyObject {
    var isRunning: Bool { get }
    /// nil while running.
    var exitStatus: Int32? { get }
    func terminate()
}

extension Process: MCPChild {
    /// `terminationStatus` traps if the process was never launched; every
    /// `MCPChild` has been.
    var exitStatus: Int32? { isRunning ? nil : terminationStatus }
}

struct MCPStdio {
    /// Read the server's stdout from here.
    let input: FileDescriptor
    /// Write the server's stdin here.
    let output: FileDescriptor
    let child: MCPChild
    let stderr: StderrBox
}

// MARK: - Routing

enum MCPSpawnerRouter {
    /// Sandbox ON → the guest. Sandbox OFF → the host, as before.
    ///
    /// Under `MAS_BUILD` there is no host arm at all: `HostMCPSpawner` is
    /// compiled out, so this can only ever return the guest spawner.
    static func spawner(sandboxEnabled: Bool) -> MCPSpawner {
        #if MAS_BUILD
        return GuestMCPSpawner()
        #else
        return sandboxEnabled ? GuestMCPSpawner() : HostMCPSpawner()
        #endif
    }

    /// Pure decision, so the routing table is testable without spawning anything.
    /// Mirrors `spawner(sandboxEnabled:)` exactly.
    static func destination(sandboxEnabled: Bool, masBuild: Bool = BuildFeatures.current.isMAS) -> Destination {
        if masBuild { return .guest }
        return sandboxEnabled ? .guest : .host
    }

    enum Destination: Equatable { case host, guest }
}

// MARK: - Host

#if !MAS_BUILD
/// The historical path: a login `zsh` on the host, pipes for stdio.
struct HostMCPSpawner: MCPSpawner {

    func commandExists(_ command: String) async -> Bool {
        await withCheckedContinuation { continuation in
            let p = Process()
            p.executableURL = URL(fileURLWithPath: "/bin/zsh")
            // Single-quote the command name so weird inputs can't escape into shell injection.
            let safe = command.replacingOccurrences(of: "'", with: "'\"'\"'")
            p.arguments = ["-l", "-c", "command -v '\(safe)' >/dev/null 2>&1"]
            p.standardOutput = Pipe()
            p.standardError = Pipe()
            p.terminationHandler = { proc in continuation.resume(returning: proc.terminationStatus == 0) }
            do { try p.run() } catch { continuation.resume(returning: false) }
        }
    }

    func open(command: String, args: [String], cwd: String, env: [String: String]) async throws -> MCPStdio {
        let process = Process()
        // Run via login zsh so PATH includes Homebrew (npx, uvx, docker) — same approach as CLILauncher.
        // `-l -c "exec ..."` makes the shell exec the target so the FDs survive without an extra hop.
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-l", "-c", MCPCommandLine.shellEscape(command: command, args: args)]
        process.currentDirectoryURL = URL(fileURLWithPath: cwd)
        // `env` carries ONLY the per-server overrides (the spawner contract —
        // the guest spawner lays the same dict over the guest IMAGE's env, so
        // handing it the whole macOS environment would clobber the guest's
        // PATH/HOME). Here on the host, merge over the app's own environment,
        // which is what this path always did.
        var merged = ProcessInfo.processInfo.environment
        for (key, value) in env { merged[key] = value }
        process.environment = merged

        let stdin = Pipe(), stdout = Pipe(), stderr = Pipe()
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = stderr

        // Capture stderr eagerly so a fast crash (npm install failure, missing native dep, bad args)
        // leaves us with a useful error message instead of an empty buffer.
        //
        // Foundation gotcha: once the child closes stderr (EOF), the readabilityHandler keeps firing
        // forever and `availableData` returns empty data immediately — a hot CPU loop. Detect EOF and
        // detach the handler so we don't starve the rest of the program.
        let box = StderrBox()
        stderr.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty { handle.readabilityHandler = nil; return }
            if let chunk = String(data: data, encoding: .utf8) { box.append(chunk) }
        }

        try process.run()

        return MCPStdio(
            input: FileDescriptor(rawValue: stdout.fileHandleForReading.fileDescriptor),
            output: FileDescriptor(rawValue: stdin.fileHandleForWriting.fileDescriptor),
            child: process,
            stderr: box)
    }
}
#endif

// MARK: - Guest

/// Runs the server inside the Agent Sandbox guest, over one vsock connection.
///
/// Servers in the `MCPGuestPrebaked` table run their DIRECT guest bin, never
/// npx — npx re-downloads the package on every start even when it is already
/// installed, which is a 2.5.2 violation on the store build and a needless
/// network dependency on the DMG build. Under `prebakedOnly` (MAS) a server
/// outside the table is an explicit error; the DMG build falls back to the
/// authored npx invocation (and also when an older cached guest image predates
/// the baked bins — same fail-loudly-on-MAS / fall-back-on-DMG split as
/// `AgentSandbox.chooseTransport`).
struct GuestMCPSpawner: MCPSpawner {

    let prebakedOnly: Bool

    init(prebakedOnly: Bool = BuildFeatures.current.guest.mcpServers == .prebaked) {
        self.prebakedOnly = prebakedOnly
    }

    func commandExists(_ command: String) async -> Bool {
        let safe = command.replacingOccurrences(of: "'", with: "'\"'\"'")
        return await AgentSandbox.shared.guestCommandSucceeds("command -v '\(safe)' >/dev/null 2>&1")
    }

    func open(command: String, args: [String], cwd: String, env: [String: String]) async throws -> MCPStdio {
        var invocation = try MCPGuestPrebaked.invocation(command: command, args: args,
                                                         prebakedOnly: prebakedOnly)
        if !prebakedOnly, invocation.command != command,
           !(await commandExists(invocation.command)) {
            // Stale DMG guest image without the baked bin — keep the old npx path.
            invocation = (command, args)
        }
        let line = MCPCommandLine.shellEscape(command: invocation.command, args: invocation.args)
        let bridge = try await AgentSandbox.shared.openMCPBridge(command: line, hostCwd: cwd, env: env)
        return MCPStdio(input: bridge.transportDescriptor,
                        output: bridge.transportDescriptor,
                        child: bridge,
                        stderr: bridge.stderr)
    }
}

// MARK: - Guest stdio bridge

/// Adapts the guest's framed exec protocol to the raw bidirectional descriptor
/// `StdioTransport` expects.
///
/// A `socketpair` gives two ends: the SDK reads and writes one, we shuttle the
/// other to and from the guest. Two threads, because both directions must move
/// independently — an MCP server can stream a large tool result while the client
/// is still writing the next request.
final class GuestMCPBridge: MCPChild, @unchecked Sendable {

    let stderr = StderrBox()
    /// The descriptor handed to `StdioTransport` (readable and writable).
    let transportDescriptor: FileDescriptor

    private let stream: GuestByteStream
    private let ourEnd: Int32
    private let lock = NSLock()
    private var alive = true
    private var exit: Int32?

    var isRunning: Bool { lock.lock(); defer { lock.unlock() }; return alive }
    var exitStatus: Int32? { lock.lock(); defer { lock.unlock() }; return alive ? nil : (exit ?? -1) }

    /// `stream` must already carry a `.request` frame for the server process.
    init(stream: GuestByteStream) throws {
        self.stream = stream

        var fds: [Int32] = [0, 0]
        guard socketpair(AF_UNIX, SOCK_STREAM, 0, &fds) == 0 else {
            stream.close()
            throw GuestProtocol.ProtocolError(message: "could not create the MCP stdio socketpair")
        }
        // Both ends: whichever side dies first, the survivor's next write must
        // return EPIPE rather than raising SIGPIPE and killing the app.
        FileDescriptorStream.suppressSIGPIPE(fds[0])
        FileDescriptorStream.suppressSIGPIPE(fds[1])

        ourEnd = fds[0]
        transportDescriptor = FileDescriptor(rawValue: fds[1])

        Thread.detachNewThread { [weak self] in self?.pumpGuestToTransport() }
        Thread.detachNewThread { [weak self] in self?.pumpTransportToGuest() }
    }

    /// Hanging up the guest stream IS the kill: the agent's `poll` sees it and
    /// terminates the server process. Same mechanism as an exec timeout.
    ///
    /// `shutdown`, not `close`: the pump threads may be BLOCKED in `read`/`poll`
    /// on these very descriptors — shutdown wakes them with EOF and keeps the
    /// fd numbers reserved; the real closes happen once, in `deinit`, which can
    /// only run after both pumps exited (they hold `self` strongly while running).
    func terminate() {
        lock.lock()
        guard alive else { lock.unlock(); return }
        alive = false
        lock.unlock()

        stream.close()
        _ = Darwin.shutdown(ourEnd, SHUT_RDWR)
    }

    private func markDead(exitCode: Int32?) {
        lock.lock()
        guard alive else { lock.unlock(); return }
        alive = false
        exit = exitCode
        lock.unlock()
        // EOFs the SDK's reader, so `disconnect()` doesn't hang.
        _ = Darwin.shutdown(ourEnd, SHUT_RDWR)
    }

    private func pumpGuestToTransport() {
        var reader = GuestFrameReader()
        var exitCode: Int32?

        outer: while true {
            let outcome: ReadOutcome
            // No deadline: an MCP server legitimately sits idle for minutes between
            // tool calls. Liveness is the SDK's problem, not the transport's.
            do { outcome = try stream.read(maxLength: 64 * 1024, deadline: .distantFuture) }
            catch { break }

            guard case .bytes(let chunk) = outcome else { break } // eof / timedOut
            guard let frames = try? reader.feed(chunk) else { break }

            for frame in frames {
                switch frame.channel {
                case .stdout:
                    if !writeAll(ourEnd, frame.payload) { break outer }
                case .stderr:
                    stderr.append(String(decoding: frame.payload, as: UTF8.self))
                case .exit:
                    exitCode = try? GuestProtocol.decodeInt32(frame.payload)
                    break outer
                case .error:
                    stderr.append(String(decoding: frame.payload, as: UTF8.self))
                    break outer
                case .started:
                    continue // pid is not interesting; `terminate()` hangs up instead
                case .request, .stdin, .stdinEOF:
                    break outer // the guest never sends host-only channels
                }
            }
        }
        markDead(exitCode: exitCode)
    }

    private func pumpTransportToGuest() {
        var buffer = [UInt8](repeating: 0, count: 32 * 1024)
        while true {
            let n = Darwin.read(ourEnd, &buffer, buffer.count)
            if n < 0 && errno == EINTR { continue }
            guard n > 0 else { break } // the SDK closed its end
            let frame = GuestProtocol.frame(.stdin, Data(buffer[0..<n]))
            guard (try? stream.write(frame)) != nil else { break }
        }
        try? stream.write(GuestProtocol.frame(.stdinEOF))
    }

    private func writeAll(_ fd: Int32, _ data: Data) -> Bool {
        data.withUnsafeBytes { raw in
            var sent = 0
            while sent < raw.count {
                let n = Darwin.write(fd, raw.baseAddress!.advanced(by: sent), raw.count - sent)
                if n <= 0 {
                    if errno == EINTR { continue }
                    return false
                }
                sent += n
            }
            return true
        }
    }

    deinit {
        terminate()
        // The only real closes. `StdioTransport.disconnect()` never closes the
        // descriptor it was handed, so without this every guest MCP session
        // leaks its socketpair.
        Darwin.close(ourEnd)
        Darwin.close(transportDescriptor.rawValue)
    }
}

// MARK: - Command line

enum MCPCommandLine {
    /// Single-quote each token so spaces/specials don't reinterpret. Embed literal single-quotes via '"'"'.
    /// `exec` replaces the shell so the child's FDs are the ones we handed the shell.
    static func shellEscape(command: String, args: [String]) -> String {
        func quote(_ token: String) -> String {
            "'" + token.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
        }
        return (["exec", quote(command)] + args.map(quote)).joined(separator: " ")
    }
}
