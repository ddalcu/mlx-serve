import Darwin
import XCTest
@testable import MLXCore

/// Frame reassembly is where a stream transport actually breaks: payloads split
/// across reads, several frames in one read, a header straddling a boundary.
/// Golden bytes can't see any of it.
final class GuestFrameReaderTests: XCTestCase {

    func testTwoFramesArrivingInOneRead() throws {
        var reader = GuestFrameReader()
        let bytes = GuestProtocol.frame(.stdout, Data("ab".utf8))
            + GuestProtocol.frame(.stderr, Data("c".utf8))

        let frames = try reader.feed(bytes)
        XCTAssertEqual(frames.count, 2)
        XCTAssertEqual(frames[0].channel, .stdout)
        XCTAssertEqual(frames[0].payload, Data("ab".utf8))
        XCTAssertEqual(frames[1].channel, .stderr)
        XCTAssertEqual(frames[1].payload, Data("c".utf8))
        XCTAssertEqual(reader.pendingBytes, 0)
    }

    /// The pathological case: deliver one byte at a time. Nothing may be
    /// emitted until a frame is complete, and then exactly once.
    func testFrameSplitAcrossEveryPossibleBoundary() throws {
        var reader = GuestFrameReader()
        let bytes = GuestProtocol.frame(.stdout, Data("hello".utf8))

        var emitted: [(GuestProtocol.Channel, Data)] = []
        for byte in bytes {
            emitted.append(contentsOf: try reader.feed(Data([byte])))
        }
        XCTAssertEqual(emitted.count, 1)
        XCTAssertEqual(emitted[0].1, Data("hello".utf8))
        XCTAssertEqual(reader.pendingBytes, 0)
    }

    func testPartialFrameIsHeldNotEmitted() throws {
        var reader = GuestFrameReader()
        var bytes = GuestProtocol.frame(.stdout, Data("hello".utf8))
        bytes.removeLast() // one byte short

        XCTAssertTrue(try reader.feed(bytes).isEmpty)
        XCTAssertEqual(reader.pendingBytes, bytes.count)

        let frames = try reader.feed(Data("o".utf8))
        XCTAssertEqual(frames.count, 1)
        XCTAssertEqual(frames[0].payload, Data("hello".utf8))
    }

    func testEmptyPayloadFrame() throws {
        var reader = GuestFrameReader()
        let frames = try reader.feed(GuestProtocol.frame(.stdinEOF))
        XCTAssertEqual(frames.count, 1)
        XCTAssertEqual(frames[0].payload.count, 0)
    }

    func testCorruptChannelThrows() {
        var reader = GuestFrameReader()
        XCTAssertThrowsError(try reader.feed(Data([99, 0, 0, 0, 0])))
    }
}

final class FileDescriptorStreamTests: XCTestCase {

    /// `close()` must WAKE a reader blocked on another thread — the MCP
    /// bridge's pump reads with no deadline, so `terminate()` hanging up the
    /// stream is the only thing that can ever stop it. A literal `close(2)`
    /// does NOT wake a blocked `poll` and frees the fd number for reuse under
    /// the reader's feet; `shutdown(2)` wakes it with EOF and leaves the number
    /// reserved until the stream deallocates.
    func testCloseWakesABlockedReader() throws {
        var fds: [Int32] = [0, 0]
        XCTAssertEqual(socketpair(AF_UNIX, SOCK_STREAM, 0, &fds), 0)
        let stream = FileDescriptorStream(fd: fds[0])
        defer { Darwin.close(fds[1]) }

        let woke = expectation(description: "blocked reader returned")
        Thread.detachNewThread {
            let outcome = try? stream.read(maxLength: 1024, deadline: .distantFuture)
            XCTAssertEqual(outcome, .eof)
            woke.fulfill()
        }
        Thread.sleep(forTimeInterval: 0.1) // let the reader park in poll
        stream.close()
        wait(for: [woke], timeout: 2)
    }

    /// The peer must see the hangup too — that is what makes the guest agent
    /// kill the child on a timeout or a cancelled tool call.
    func testCloseDeliversEOFToThePeer() throws {
        var fds: [Int32] = [0, 0]
        XCTAssertEqual(socketpair(AF_UNIX, SOCK_STREAM, 0, &fds), 0)
        let stream = FileDescriptorStream(fd: fds[0])
        defer { Darwin.close(fds[1]) }

        stream.close()
        var byte: UInt8 = 0
        XCTAssertEqual(read(fds[1], &byte, 1), 0, "peer should read EOF after close()")
    }
}

// MARK: - Cross-language interop

/// The Swift driver and the Zig agent are two implementations of one protocol
/// that can never be compiled together. Golden bytes pin the encoding; only
/// running them against each other pins the *behavior* — streaming, EOF, exit
/// codes, hangup-kills-the-child.
///
/// `zig build vz-agent-host` produces a macOS-native build of the very same
/// `src/vz_agent.zig` that ships into the guest, listening on a unix socket
/// instead of vsock.
///
///     zig build vz-agent-host
///     VZ_AGENT_HOST_BIN=zig-out/bin/vz-agent-host swift test --filter GuestExecInterop
final class GuestExecInteropTests: XCTestCase {

    private var agent: Process?
    private var socketPath = ""

    override func setUpWithError() throws {
        guard let binary = ProcessInfo.processInfo.environment["VZ_AGENT_HOST_BIN"],
              FileManager.default.isExecutableFile(atPath: binary) else {
            throw XCTSkip("set VZ_AGENT_HOST_BIN (build it with `zig build vz-agent-host`)")
        }
        socketPath = NSTemporaryDirectory() + "vz-\(UUID().uuidString.prefix(8)).sock"

        let process = Process()
        process.executableURL = URL(fileURLWithPath: binary)
        var env = ProcessInfo.processInfo.environment
        env["VZ_AGENT_UNIX_SOCKET"] = socketPath
        process.environment = env
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        agent = process

        // Wait for the listener rather than sleeping a fixed amount.
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline, !FileManager.default.fileExists(atPath: socketPath) {
            usleep(20_000)
        }
        guard FileManager.default.fileExists(atPath: socketPath) else {
            throw XCTSkip("vz-agent-host never created \(socketPath)")
        }
    }

    override func tearDown() {
        agent?.terminate()
        try? FileManager.default.removeItem(atPath: socketPath)
    }

    private func rawConnect() throws -> Int32 {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(fd, 0)

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        addr.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
        _ = withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            socketPath.withCString { src in
                strncpy(UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: CChar.self), src, 103)
            }
        }
        let ok = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard ok == 0 else {
            Darwin.close(fd)
            throw XCTSkip("could not connect to \(socketPath)")
        }
        return fd
    }

    private func connect() throws -> GuestByteStream {
        FileDescriptorStream(fd: try rawConnect())
    }

    /// One connection hanging up while the agent is BLOCKED WRITING to it must
    /// not take the whole agent down. Without `SIG_IGN` on SIGPIPE, the blocked
    /// `write` raises SIGPIPE (default: terminate) and every other connection —
    /// all MCP servers plus the shell — dies with this one.
    ///
    /// The blocked-write state is forced by never reading: `yes` fills the
    /// socket buffer within milliseconds and the agent parks inside `write`
    /// until the hangup lands on it.
    func testHangupMidStreamDoesNotKillTheAgent() throws {
        // A RAW fd, deliberately: only a real close(2) reliably errors the
        // agent's in-flight write (a shutdown can leave it parked), and the
        // production hangup — the VZVirtioSocketConnection being torn down —
        // is a close.
        let fd = try rawConnect()
        let frame = GuestProtocol.frame(.request, GuestProtocol.Request(command: "yes 0123456789abcdef").encode())
        _ = frame.withUnsafeBytes { Darwin.write(fd, $0.baseAddress, $0.count) }
        Thread.sleep(forTimeInterval: 0.5) // agent is now blocked mid-write
        Darwin.close(fd)
        Thread.sleep(forTimeInterval: 0.5) // let a SIGPIPE (if any) land

        XCTAssertEqual(agent?.isRunning, true,
                       "the agent died on a single connection's hangup — SIGPIPE is not ignored")

        // And it still serves: a fresh connection runs to completion.
        let second = try connect()
        defer { second.close() }
        let result = try GuestExec.run(stream: second, request: .init(command: "echo alive"))
        XCTAssertEqual(result.stdout, "alive\n")
        XCTAssertEqual(result.exitCode, 0)
    }

    func testStdoutStderrAndExitCode() throws {
        let stream = try connect()
        defer { stream.close() }

        let result = try GuestExec.run(
            stream: stream,
            request: .init(command: "echo hello; echo oops 1>&2; exit 5"))

        XCTAssertEqual(result.stdout, "hello\n")
        XCTAssertEqual(result.stderr, "oops\n")
        XCTAssertEqual(result.exitCode, 5)
        XCTAssertFalse(result.timedOut)
        XCTAssertNotNil(result.pid)
    }

    /// A signalled child must not look like a clean exit — `ShellSentinel` could
    /// never tell these apart.
    func testSignalledChildReports128PlusSignal() throws {
        let stream = try connect()
        defer { stream.close() }
        let result = try GuestExec.run(stream: stream, request: .init(command: "kill -TERM $$"))
        XCTAssertEqual(result.exitCode, 143)
    }

    func testCwdAndEnvAreApplied() throws {
        let stream = try connect()
        defer { stream.close() }
        let result = try GuestExec.run(
            stream: stream,
            request: .init(command: "pwd; printf '%s' \"$VZ_MARK\"",
                           cwd: "/tmp",
                           env: [(key: "VZ_MARK", value: "seen")]))
        XCTAssertTrue(result.stdout.contains("tmp"), result.stdout)
        XCTAssertTrue(result.stdout.hasSuffix("seen"), result.stdout)
        XCTAssertEqual(result.exitCode, 0)
    }

    /// 300 KB crosses the agent's 64 KB read buffer and the driver's, so this is
    /// the test that catches a reassembly bug.
    func testLargeOutputSurvivesReassembly() throws {
        let stream = try connect()
        defer { stream.close() }
        let result = try GuestExec.run(
            stream: stream,
            request: .init(command: "yes abcdefghij | head -30000"),
            timeout: 30)
        XCTAssertEqual(result.stdout.utf8.count, 30000 * 11)
        XCTAssertEqual(result.exitCode, 0)
    }

    func testStreamingCallbacksSeeOutputBeforeExit() throws {
        let stream = try connect()
        defer { stream.close() }

        var chunks = 0
        let result = try GuestExec.run(
            stream: stream,
            request: .init(command: "echo a; echo b"),
            streaming: .init(onStdout: { _ in chunks += 1 }))

        XCTAssertGreaterThan(chunks, 0)
        XCTAssertEqual(result.stdout, "a\nb\n")
    }

    /// The timeout path IS the cancellation path: the driver closes the socket
    /// and the agent kills the child. If that ever silently stopped working, a
    /// cancelled tool call would leave a process running in the guest forever.
    func testTimeoutClosesTheStreamAndKillsTheChild() throws {
        let stream = try connect()
        let started = Date()
        let result = try GuestExec.run(
            stream: stream,
            request: .init(command: "echo up; sleep 30"),
            timeout: 1.0)

        XCTAssertTrue(result.timedOut)
        XCTAssertEqual(result.exitCode, 124)
        XCTAssertLessThan(Date().timeIntervalSince(started), 10)

        // The agent reported the pid before we hung up; the child must be gone.
        let pid = try XCTUnwrap(result.pid)
        usleep(300_000)
        XCTAssertNotEqual(kill(pid, 0), 0, "child \(pid) survived the hangup")
    }

    /// The Sandbox Terminal's sticky `cd`, end to end through the real agent.
    /// The string-shape tests can't see that `printf` clobbers `$?` or that the
    /// marker survives a real shell.
    func testSandboxTerminalCwdWrapperRoundTripsThroughTheRealAgent() throws {
        // `cd` moves, and the marker reports where we landed.
        var stream = try connect()
        var result = try GuestExec.run(
            stream: stream,
            request: .init(command: AgentSandbox.wrapUserCommand("cd /; echo moved", guestCwd: "/tmp")))
        stream.close()

        var split = AgentSandbox.splitCwdMarker(result.stdout)
        XCTAssertEqual(split.cwd, "/")
        // The wrapper's own leading newline is what `splitCwdMarker` strips, so
        // the command's output survives byte-for-byte — trailing newline included.
        XCTAssertEqual(split.body, "moved\n")
        XCTAssertEqual(result.exitCode, 0)

        // A failing command keeps its exit status despite the trailing printf.
        stream = try connect()
        result = try GuestExec.run(
            stream: stream,
            request: .init(command: AgentSandbox.wrapUserCommand("false", guestCwd: "/tmp")))
        stream.close()

        XCTAssertEqual(result.exitCode, 1, "printf clobbered $?")
        split = AgentSandbox.splitCwdMarker(result.stdout)
        XCTAssertTrue(split.cwd?.hasSuffix("tmp") == true, "\(split.cwd ?? "nil")")
    }

    /// Each connection is an independent process — that is what lets an MCP
    /// server hold one open for minutes while the shell tool uses others.
    func testConcurrentConnectionsAreIndependent() throws {
        let group = DispatchGroup()
        var results = [Int32?](repeating: nil, count: 4)
        let lock = NSLock()

        for i in 0..<4 {
            group.enter()
            DispatchQueue.global().async {
                defer { group.leave() }
                guard let stream = try? self.connect() else { return }
                defer { stream.close() }
                let result = try? GuestExec.run(stream: stream, request: .init(command: "exit \(i)"))
                lock.lock(); results[i] = result?.exitCode; lock.unlock()
            }
        }
        XCTAssertEqual(group.wait(timeout: .now() + 30), .success)
        XCTAssertEqual(results, [0, 1, 2, 3])
    }
}
