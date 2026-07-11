import Darwin
import Foundation
import Virtualization

/// Host side of the Agent Sandbox exec transport.
///
/// A `GuestByteStream` is one bidirectional connection to the guest agent.
/// In production that is a vsock connection opened with
/// `VZVirtioSocketDevice.connect(toPort:)`, whose `fileDescriptor` is a plain
/// bidirectional fd. In tests it is a unix socket to a natively-built
/// `vz-agent`, so the Swift driver and the real Zig agent are exercised against
/// each other without a VM.
///
/// `GuestExec` drives one request to completion over such a stream.
protocol GuestByteStream: AnyObject {
    /// Blocking; writes everything or throws.
    func write(_ data: Data) throws
    /// Blocks until bytes arrive, the peer hangs up, or `deadline` passes.
    ///
    /// The deadline is NOT optional. A plain blocking `read` makes the caller's
    /// timeout unenforceable — a child that produces no output (`sleep 30`)
    /// parks the reader inside `read(2)` and the deadline is never consulted.
    /// That is the same "liveness belongs to the transport, not the producer"
    /// mistake as the streaming-keepalive bug in CLAUDE.md.
    func read(maxLength: Int, deadline: Date) throws -> ReadOutcome
    func close()
}

enum ReadOutcome: Equatable {
    case bytes(Data)
    case eof
    case timedOut
}

// MARK: - fd-backed stream

/// Wraps a raw bidirectional fd. Used for both vsock and unix sockets.
final class FileDescriptorStream: GuestByteStream {
    private let fd: Int32
    /// vsock connections close their fd when deallocated, so the connection
    /// object must outlive the stream. Held, never touched.
    private let owner: AnyObject?
    private var closed = false
    private let lock = NSLock()

    init(fd: Int32, owner: AnyObject? = nil) {
        self.fd = fd
        self.owner = owner
        FileDescriptorStream.suppressSIGPIPE(fd)
    }

    /// The fd NUMBER is released here and only here, after every user of the
    /// stream is necessarily done with it. `close()` deliberately does not
    /// release it — see there.
    deinit {
        close()
        // A vsock fd belongs to its VZVirtioSocketConnection, which closes it
        // when it deallocates (releasing `owner` right after this).
        if owner == nil { Darwin.close(fd) }
    }

    /// Writing to a socket whose peer has hung up raises SIGPIPE, whose default
    /// disposition kills the process. A guest that dies mid-write would take the
    /// whole app down — so ask the kernel for `EPIPE` instead, which the write
    /// paths already handle. Best-effort: harmless (and failing) on non-sockets.
    static func suppressSIGPIPE(_ fd: Int32) {
        var on: Int32 = 1
        _ = setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, socklen_t(MemoryLayout<Int32>.size))
    }

    func write(_ data: Data) throws {
        try data.withUnsafeBytes { raw in
            var sent = 0
            while sent < raw.count {
                let n = Darwin.write(fd, raw.baseAddress!.advanced(by: sent), raw.count - sent)
                if n <= 0 {
                    if errno == EINTR { continue }
                    throw GuestProtocol.ProtocolError(message: "guest write failed: \(String(cString: strerror(errno)))")
                }
                sent += n
            }
        }
    }

    func read(maxLength: Int, deadline: Date) throws -> ReadOutcome {
        while true {
            let remaining = deadline.timeIntervalSinceNow
            if remaining <= 0 { return .timedOut }

            // `poll` first: a child that never writes must not park us inside
            // read(2) past the caller's deadline.
            var pfd = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
            let ready = poll(&pfd, 1, Int32(min(remaining * 1000, Double(Int32.max))))
            if ready == 0 { return .timedOut }
            if ready < 0 {
                if errno == EINTR { continue }
                throw GuestProtocol.ProtocolError(message: "guest poll failed: \(String(cString: strerror(errno)))")
            }

            var buffer = [UInt8](repeating: 0, count: maxLength)
            let n = Darwin.read(fd, &buffer, maxLength)
            if n > 0 { return .bytes(Data(buffer[0..<n])) }
            if n == 0 { return .eof }
            if errno == EINTR { continue }
            throw GuestProtocol.ProtocolError(message: "guest read failed: \(String(cString: strerror(errno)))")
        }
    }

    /// Hang up: `shutdown(2)`, not `close(2)`.
    ///
    /// Two reasons. A reader can be BLOCKED in `poll`/`read` on another thread
    /// (the MCP bridge's pump) — `close` does not wake it and frees the fd
    /// number for reuse under its feet, while `shutdown` wakes it with EOF and
    /// keeps the number reserved until `deinit`. And a vsock fd is owned by its
    /// `VZVirtioSocketConnection`, which closes it again on dealloc — closing
    /// it here too would double-close a possibly-reused descriptor.
    ///
    /// The peer sees the same thing either way: EOF. That is what makes the
    /// guest agent kill the child on a timeout or a cancelled tool call.
    func close() {
        lock.lock(); defer { lock.unlock() }
        guard !closed else { return }
        closed = true
        _ = Darwin.shutdown(fd, SHUT_RDWR)
    }
}

// MARK: - Frame reader

/// Reassembles frames from a byte stream. A frame's payload can arrive across
/// any number of reads, and several frames can arrive in one — neither is a
/// special case here.
struct GuestFrameReader {
    private var buffer = Data()

    /// Feed raw bytes; returns every complete frame they finished.
    mutating func feed(_ bytes: Data) throws -> [(channel: GuestProtocol.Channel, payload: Data)] {
        buffer.append(bytes)
        var frames: [(GuestProtocol.Channel, Data)] = []

        while buffer.count >= GuestProtocol.headerLength {
            let header = try GuestProtocol.decodeHeader(buffer)
            let total = GuestProtocol.headerLength + header.length
            guard buffer.count >= total else { break } // payload still in flight

            let start = buffer.startIndex + GuestProtocol.headerLength
            let payload = Data(buffer[start..<(start + header.length)])
            frames.append((header.channel, payload))
            buffer.removeFirst(total)
        }
        return frames
    }

    var pendingBytes: Int { buffer.count }
}

// MARK: - Exec driver

enum GuestExec {

    struct Result {
        var stdout: String
        var stderr: String
        var exitCode: Int32
        var timedOut: Bool
        /// The child's pid inside the guest. Needed to kill a detached process.
        var pid: Int32?
    }

    struct Streams {
        var onStdout: ((Data) -> Void)?
        var onStderr: ((Data) -> Void)?
    }

    /// Run one request to completion. Blocking — call off the main thread.
    ///
    /// On timeout the stream is closed, which is how cancellation is expressed:
    /// the agent's `poll` sees the hangup and kills the child. There is no
    /// separate cancel message to get out of sync.
    static func run(
        stream: GuestByteStream,
        request: GuestProtocol.Request,
        timeout: TimeInterval = 120,
        streaming: Streams = Streams()
    ) throws -> Result {
        try stream.write(GuestProtocol.frame(.request, request.encode()))

        var reader = GuestFrameReader()
        var stdout = Data(), stderr = Data()
        var pid: Int32?
        let deadline = Date().addingTimeInterval(timeout)

        while true {
            let chunk: Data
            switch try stream.read(maxLength: 64 * 1024, deadline: deadline) {
            case .timedOut:
                // Closing the socket IS the cancellation: the agent's `poll`
                // sees the hangup and kills the child. No separate cancel
                // message that could get out of sync.
                stream.close()
                return Result(stdout: text(stdout), stderr: text(stderr),
                              exitCode: 124, timedOut: true, pid: pid)
            case .eof:
                // The agent went away without an exit frame: guest died, or the
                // connection dropped. Surface what we have rather than hanging.
                return Result(stdout: text(stdout), stderr: text(stderr),
                              exitCode: -1, timedOut: false, pid: pid)
            case .bytes(let data):
                chunk = data
            }

            for frame in try reader.feed(chunk) {
                switch frame.channel {
                case .stdout:
                    stdout.append(frame.payload)
                    streaming.onStdout?(frame.payload)
                case .stderr:
                    stderr.append(frame.payload)
                    streaming.onStderr?(frame.payload)
                case .started:
                    pid = try GuestProtocol.decodeInt32(frame.payload)
                case .exit:
                    let code = try GuestProtocol.decodeInt32(frame.payload)
                    return Result(stdout: text(stdout), stderr: text(stderr),
                                  exitCode: code, timedOut: false, pid: pid)
                case .error:
                    throw GuestProtocol.ProtocolError(
                        message: String(decoding: frame.payload, as: UTF8.self))
                case .request, .stdin, .stdinEOF:
                    throw GuestProtocol.ProtocolError(
                        message: "guest sent a host-only channel (\(frame.channel))")
                }
            }
        }
    }

    /// Guest output is whatever the command emitted; invalid UTF-8 must not
    /// lose the rest of the line.
    private static func text(_ data: Data) -> String {
        String(decoding: data, as: UTF8.self)
    }
}

// MARK: - vsock

/// Opening a vsock stream lives on `VzGuest`, not as an extension here: every
/// `VZVirtioSocketDevice` call must be made on the queue the VM was created
/// with, and only `VzGuest` owns that queue. An extension would invite a call
/// from the wrong thread, which VZ punishes at runtime rather than compile time.
///
/// `VZVirtioSocketConnection` closes its `fileDescriptor` when it deallocates,
/// so `FileDescriptorStream` retains it as `owner`.
func makeGuestStream(from connection: VZVirtioSocketConnection) -> GuestByteStream {
    FileDescriptorStream(fd: connection.fileDescriptor, owner: connection)
}
