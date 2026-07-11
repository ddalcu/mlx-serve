import Foundation

/// Host half of the Agent Sandbox guest protocol. The guest half is
/// `src/vz_agent.zig`; the two are pinned together by golden bytes asserted in
/// BOTH test suites (`GuestProtocolTests` here, `"frame header is exactly the
/// bytes Swift expects"` there). Changing one side alone silently desyncs the
/// transport.
///
/// Every message is `[UInt8 channel][UInt32 length big-endian][payload]`.
///
/// The request payload is a length-prefixed binary record rather than JSON.
/// Both ends are ours, and hand-rolled JSON escaping is a bug class this
/// codebase has already paid for twice (see the tool-call escaping gotchas in
/// CLAUDE.md). Binary framing has nothing to escape.
enum GuestProtocol {

    enum Channel: UInt8 {
        /// host → guest: the `Request` record. Always the first frame.
        case request = 0
        /// host → guest: bytes for the child's stdin.
        case stdin = 1
        /// host → guest: close the child's stdin (payload empty).
        case stdinEOF = 2
        /// guest → host: bytes the child wrote to stdout.
        case stdout = 3
        /// guest → host: bytes the child wrote to stderr.
        case stderr = 4
        /// guest → host: the child's exit status, big-endian Int32.
        case exit = 5
        /// guest → host: the child's pid, big-endian Int32. Sent once, before
        /// any output, so a detached process can later be killed.
        case started = 6
        /// guest → host: the agent itself failed (UTF-8 message). Terminal.
        case error = 7
    }

    static let headerLength = 5
    /// Mirrors `vz_agent.max_payload`. A corrupt length must not become a
    /// multi-gigabyte allocation.
    static let maxPayload = 8 << 20

    /// vsock port the guest agent listens on. Mirrors `vz_agent.port_exec`.
    /// One connection per process — the shell tool and every stdio MCP server
    /// get their own.
    static let execPort: UInt32 = 1024

    struct ProtocolError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    // MARK: - Frames

    static func encodeHeader(_ channel: Channel, payloadLength: Int) -> Data {
        var data = Data([channel.rawValue])
        data.append(bigEndian: UInt32(payloadLength))
        return data
    }

    static func frame(_ channel: Channel, _ payload: Data = Data()) -> Data {
        var data = encodeHeader(channel, payloadLength: payload.count)
        data.append(payload)
        return data
    }

    static func decodeHeader(_ bytes: Data) throws -> (channel: Channel, length: Int) {
        guard bytes.count >= headerLength else {
            throw ProtocolError(message: "guest frame header truncated")
        }
        let base = bytes.startIndex
        guard let channel = Channel(rawValue: bytes[base]) else {
            throw ProtocolError(message: "guest sent unknown channel \(bytes[base])")
        }
        let length = Int(bytes.readBigEndianUInt32(at: base + 1))
        guard length <= maxPayload else {
            throw ProtocolError(message: "guest frame claims \(length) bytes, over the \(maxPayload) cap")
        }
        return (channel, length)
    }

    /// The 4-byte payload of `.exit` and `.started`.
    static func decodeInt32(_ payload: Data) throws -> Int32 {
        guard payload.count == 4 else {
            throw ProtocolError(message: "expected a 4-byte payload, got \(payload.count)")
        }
        return Int32(bitPattern: payload.readBigEndianUInt32(at: payload.startIndex))
    }

    // MARK: - Request

    /// Runs as `/bin/sh -c <command>` inside the guest.
    ///
    /// `env` is an ordered array, not a dictionary: the encoding is compared
    /// byte-for-byte against Zig's, and `[String: String]` has no stable order.
    struct Request: Equatable {
        var command: String
        var cwd: String = ""
        /// Detached processes only. Empty → `/dev/null`.
        var logPath: String = ""
        var detach: Bool = false
        var env: [(key: String, value: String)] = []

        static func == (a: Request, b: Request) -> Bool {
            a.command == b.command && a.cwd == b.cwd && a.logPath == b.logPath
                && a.detach == b.detach && a.env.map(\.key) == b.env.map(\.key)
                && a.env.map(\.value) == b.env.map(\.value)
        }

        func encode() -> Data {
            var data = Data([detach ? 1 : 0])
            for field in [command, cwd, logPath] {
                data.append(lengthPrefixed: field)
            }
            data.append(bigEndian: UInt32(env.count))
            for pair in env {
                data.append(lengthPrefixed: pair.key)
                data.append(lengthPrefixed: pair.value)
            }
            return data
        }
    }
}

// MARK: - Byte helpers

private extension Data {
    mutating func append(bigEndian value: UInt32) {
        // Spelled out rather than `withUnsafeBytes(of:)`, which resolves to
        // Data's own instance method here and reads the wrong thing.
        append(contentsOf: [
            UInt8(truncatingIfNeeded: value >> 24),
            UInt8(truncatingIfNeeded: value >> 16),
            UInt8(truncatingIfNeeded: value >> 8),
            UInt8(truncatingIfNeeded: value),
        ])
    }

    mutating func append(lengthPrefixed text: String) {
        let bytes = Data(text.utf8)
        append(bigEndian: UInt32(bytes.count))
        append(bytes)
    }
}

extension Data {
    /// `Data` slices don't start at 0, so every read must be index-relative.
    func readBigEndianUInt32(at index: Index) -> UInt32 {
        (UInt32(self[index]) << 24)
            | (UInt32(self[index + 1]) << 16)
            | (UInt32(self[index + 2]) << 8)
            | UInt32(self[index + 3])
    }
}
