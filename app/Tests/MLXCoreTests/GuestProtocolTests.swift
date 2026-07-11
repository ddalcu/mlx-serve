import XCTest
@testable import MLXCore

/// The host↔guest transport is two implementations of one wire format, in two
/// languages, that can never be built together. The only thing keeping them in
/// sync is a set of golden bytes asserted on both sides.
///
/// The Zig test `"frame header is exactly the bytes Swift expects"` in
/// `src/vz_agent.zig` asserts the SAME three headers this file does. If you
/// change the encoding, both must move together, and both tests must be updated
/// in the same commit — otherwise the guest and the host desync at runtime with
/// no compile error anywhere.
final class GuestProtocolTests: XCTestCase {

    // MARK: - Golden bytes (mirrored in src/vz_agent.zig)

    func testHeaderGoldenBytesMatchTheZigAgent() {
        XCTAssertEqual(GuestProtocol.encodeHeader(.stdout, payloadLength: 12345),
                       Data([3, 0x00, 0x00, 0x30, 0x39]))
        XCTAssertEqual(GuestProtocol.encodeHeader(.request, payloadLength: 0),
                       Data([0, 0x00, 0x00, 0x00, 0x00]))
        XCTAssertEqual(GuestProtocol.encodeHeader(.exit, payloadLength: 4),
                       Data([5, 0x00, 0x00, 0x00, 0x04]))
    }

    /// Channel numbering is the wire contract. A reordering here is invisible
    /// to the compiler and catastrophic at runtime.
    func testChannelRawValuesArePinned() {
        XCTAssertEqual(GuestProtocol.Channel.request.rawValue, 0)
        XCTAssertEqual(GuestProtocol.Channel.stdin.rawValue, 1)
        XCTAssertEqual(GuestProtocol.Channel.stdinEOF.rawValue, 2)
        XCTAssertEqual(GuestProtocol.Channel.stdout.rawValue, 3)
        XCTAssertEqual(GuestProtocol.Channel.stderr.rawValue, 4)
        XCTAssertEqual(GuestProtocol.Channel.exit.rawValue, 5)
        XCTAssertEqual(GuestProtocol.Channel.started.rawValue, 6)
        XCTAssertEqual(GuestProtocol.Channel.error.rawValue, 7)
    }

    /// The exact bytes `Request.encode()` must produce, spelled out by hand.
    /// This is the assertion that would catch a field reordering or an
    /// endianness slip in either language.
    func testRequestEncodingGoldenBytes() {
        let request = GuestProtocol.Request(
            command: "ls", cwd: "/w", logPath: "", detach: true,
            env: [(key: "A", value: "B")])

        var expected = Data()
        expected.append(1)                                  // flags: detach
        expected.append(contentsOf: [0, 0, 0, 2]); expected.append(contentsOf: Array("ls".utf8))
        expected.append(contentsOf: [0, 0, 0, 2]); expected.append(contentsOf: Array("/w".utf8))
        expected.append(contentsOf: [0, 0, 0, 0])           // logPath: empty
        expected.append(contentsOf: [0, 0, 0, 1])           // env count
        expected.append(contentsOf: [0, 0, 0, 1]); expected.append(contentsOf: Array("A".utf8))
        expected.append(contentsOf: [0, 0, 0, 1]); expected.append(contentsOf: Array("B".utf8))

        XCTAssertEqual(request.encode(), expected)
    }

    /// Bytes that would need escaping in JSON must ride through untouched —
    /// that is the entire reason this record is binary.
    func testRequestEncodingIsTransparentToHostileBytes() {
        let nasty = "a\"b\\c\nd\u{0}e"
        let request = GuestProtocol.Request(command: nasty)
        let encoded = request.encode()

        // flags(1) + len(4) + command + len(4) + len(4) + count(4)
        let commandBytes = Array(nasty.utf8)
        XCTAssertEqual(Int(encoded.readBigEndianUInt32(at: 1)), commandBytes.count)
        XCTAssertEqual(Array(encoded[5..<(5 + commandBytes.count)]), commandBytes)
    }

    // MARK: - Header decoding

    func testDecodeHeaderRoundTrip() throws {
        let header = GuestProtocol.encodeHeader(.stderr, payloadLength: 7)
        let decoded = try GuestProtocol.decodeHeader(header)
        XCTAssertEqual(decoded.channel, .stderr)
        XCTAssertEqual(decoded.length, 7)
    }

    func testDecodeHeaderRejectsUnknownChannel() {
        XCTAssertThrowsError(try GuestProtocol.decodeHeader(Data([9, 0, 0, 0, 1]))) { error in
            XCTAssertTrue("\(error)".contains("unknown channel"), "\(error)")
        }
    }

    func testDecodeHeaderRejectsAbsurdLength() {
        XCTAssertThrowsError(try GuestProtocol.decodeHeader(Data([3, 0xff, 0xff, 0xff, 0xff]))) { error in
            XCTAssertTrue("\(error)".contains("cap"), "\(error)")
        }
    }

    func testDecodeHeaderRejectsShortBuffer() {
        XCTAssertThrowsError(try GuestProtocol.decodeHeader(Data([3, 0, 0])))
    }

    /// `Data` slices carry a non-zero `startIndex`. Reading with absolute
    /// offsets works on a fresh `Data` and silently corrupts a slice — which is
    /// exactly what a streaming reader hands you.
    func testDecodeHeaderWorksOnASlicedBuffer() throws {
        var stream = Data([0xAA, 0xBB]) // leading bytes already consumed
        stream.append(GuestProtocol.encodeHeader(.stdout, payloadLength: 300))
        let slice = stream[2...]

        let decoded = try GuestProtocol.decodeHeader(slice)
        XCTAssertEqual(decoded.channel, .stdout)
        XCTAssertEqual(decoded.length, 300)
    }

    // MARK: - Payload decoding

    func testDecodeInt32HandlesNegativeAndSliced() throws {
        var payload = Data([0xFF]) // leading junk to force a non-zero startIndex
        payload.append(contentsOf: [0xFF, 0xFF, 0xFF, 0xFE])
        XCTAssertEqual(try GuestProtocol.decodeInt32(payload[1...]), -2)

        XCTAssertEqual(try GuestProtocol.decodeInt32(Data([0, 0, 0, 3])), 3)
        // 128 + SIGTERM, the shell's convention for a signalled child.
        XCTAssertEqual(try GuestProtocol.decodeInt32(Data([0, 0, 0, 143])), 143)
    }

    func testDecodeInt32RejectsWrongWidth() {
        XCTAssertThrowsError(try GuestProtocol.decodeInt32(Data([0, 0, 3])))
    }

    // MARK: - Frame assembly

    func testFrameConcatenatesHeaderAndPayload() {
        let frame = GuestProtocol.frame(.stdin, Data("hi".utf8))
        XCTAssertEqual(frame, Data([1, 0, 0, 0, 2]) + Data("hi".utf8))
    }

    func testEmptyFrameIsHeaderOnly() {
        XCTAssertEqual(GuestProtocol.frame(.stdinEOF), Data([2, 0, 0, 0, 0]))
    }
}
