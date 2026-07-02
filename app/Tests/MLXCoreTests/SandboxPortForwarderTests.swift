import XCTest
import Network
@testable import MLXCore

/// Pure-parser + loopback-forwarding tests for the sandbox's live port map.
/// No VM: the "guest" in the end-to-end test is a local listener on IPv6
/// loopback, so the forwarder (which binds IPv4 127.0.0.1) can share the port.
final class SandboxPortForwarderTests: XCTestCase {

    // MARK: /proc/net/tcp parsing

    /// Real /proc/net/tcp shape: sl, local_address, rem_address, st, …
    /// 0A = LISTEN. Addresses are little-endian hex; 0100007F = 127.0.0.1.
    private let sample = """
      sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
       0: 00000000:1F90 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 1234 1 0000000000000000 100 0 0 10 0
       1: 0100007F:270F 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 1235 1 0000000000000000 100 0 0 10 0
       2: 0F02000A:0016 0100007F:D431 01 00000000:00000000 00:00000000 00000000     0        0 1236 1 0000000000000000 100 0 0 10 0
    """

    private let sample6 = """
      sl  local_address                         remote_address                        st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
       0: 00000000000000000000000000000000:0BB8 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 2000 1 0000000000000000 100 0 0 10 0
       1: 00000000000000000000000001000000:1A0A 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 2001 1 0000000000000000 100 0 0 10 0
    """

    func testParseSnapshotExtractsListeningNonLoopbackPorts() {
        let snap = GuestNetParser.parse("=IP=192.168.64.7\n" + sample + "\n" + sample6)
        XCTAssertEqual(snap.ip, "192.168.64.7")
        // 0x1F90 = 8080 (wildcard LISTEN → forward), 0x0BB8 = 3000 (v6 :: LISTEN → forward)
        XCTAssertTrue(snap.ports.contains(8080))
        XCTAssertTrue(snap.ports.contains(3000))
        // 0x270F = 9999 bound to 127.0.0.1 in the GUEST — unreachable from the
        // host via the guest IP, must not be forwarded.
        XCTAssertFalse(snap.ports.contains(9999))
        // ::1-bound (guest v6 loopback) likewise excluded (0x1A0A = 6666).
        XCTAssertFalse(snap.ports.contains(6666))
        // st 01 (ESTABLISHED, port 22) is not a listener.
        XCTAssertFalse(snap.ports.contains(22))
    }

    func testParseSnapshotWithoutIpYieldsNilIp() {
        let snap = GuestNetParser.parse("=IP=\n" + sample)
        XCTAssertNil(snap.ip)
        XCTAssertEqual(snap.ports, [8080])
    }

    func testParseSnapshotExtractsMeminfo() {
        let text = "=IP=192.168.64.4\nMemTotal:        1010536 kB\nMemAvailable:     612340 kB\n" + sample
        let snap = GuestNetParser.parse(text)
        XCTAssertEqual(snap.memTotalKB, 1_010_536)
        XCTAssertEqual(snap.memAvailableKB, 612_340)
        XCTAssertEqual(snap.ports, [8080], "meminfo lines must not disturb port parsing")
    }

    func testMemoryDisplayTextQuantizesAndFormats() {
        // Quantized to 16 MB steps so a jittering guest doesn't re-render the
        // tray every second (the MenuBarExtra churn class).
        XCTAssertEqual(AgentSandbox.memoryDisplayText(availableKB: 612_340, totalKB: 1_010_536),
                       "384 MB / 987 MB RAM")
        // Nearby readings quantize to the SAME string (no publish).
        XCTAssertEqual(AgentSandbox.memoryDisplayText(availableKB: 610_000, totalKB: 1_010_536),
                       AgentSandbox.memoryDisplayText(availableKB: 612_340, totalKB: 1_010_536))
        // Multi-GB guests read in GB with one decimal.
        XCTAssertEqual(AgentSandbox.memoryDisplayText(availableKB: 2_000_000, totalKB: 4_100_000),
                       "2048 MB / 3.9 GB RAM")
    }

    func testParseSnapshotSurvivesTtyCrlfLineEndings() {
        // hvc2 is a tty: ONLCR turns every \n into \r\n, and Swift treats
        // "\r\n" as ONE grapheme cluster — a split on "\n" never splits.
        // Live-smoke regression: the whole snapshot parsed as a single line,
        // the "IP" swallowed the tcp table, and no ports were ever mapped.
        let crlf = ("=IP=192.168.64.4\n" + sample + "\n" + sample6)
            .replacingOccurrences(of: "\n", with: "\r\n")
        let snap = GuestNetParser.parse(crlf)
        XCTAssertEqual(snap.ip, "192.168.64.4")
        XCTAssertEqual(snap.ports, [8080, 3000])
    }

    func testSnapshotSplitterFramesOnSentinel() {
        let splitter = GuestNetSnapshotSplitter()
        var got: [String] = []
        got += splitter.feed(Data("=IP=1.2.3.4\nline1\n=EO".utf8))
        XCTAssertTrue(got.isEmpty, "incomplete snapshot must wait for the sentinel")
        got += splitter.feed(Data("S=\n=IP=5.6".utf8))
        XCTAssertEqual(got.count, 1)
        XCTAssertTrue(got[0].contains("line1"))
        got += splitter.feed(Data(".7.8\n=EOS=\n".utf8))
        XCTAssertEqual(got.count, 2)
        XCTAssertTrue(got[1].contains("5.6.7.8"))
    }

    // MARK: end-to-end loopback forward

    /// "Guest" echo service on [::1]:Q + forwarder mapping host 127.0.0.1:P →
    /// ::1:Q via the test seam (NWListener internally probes BOTH address
    /// families of its port, so P must differ from Q in-process; the real
    /// same-port mapping is proven live by SandboxSmoke phase 4).
    func testForwarderRelaysBytesToTarget() throws {
        let port = UInt16.random(in: 20000...30000) // host side (P)
        let servicePort = port + 10000 // fake guest side (Q)
        let q = DispatchQueue(label: "test.guest.service")

        // Echo-once service: BSD IPv6 socket, strictly ::1 (IPV6_V6ONLY).
        let fd = socket(AF_INET6, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(fd, 0)
        var on: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, socklen_t(MemoryLayout<Int32>.size))
        setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &on, socklen_t(MemoryLayout<Int32>.size))
        var addr = sockaddr_in6()
        addr.sin6_family = sa_family_t(AF_INET6)
        addr.sin6_port = servicePort.bigEndian
        addr.sin6_addr = in6addr_loopback
        let bound = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in6>.size))
            }
        }
        XCTAssertEqual(bound, 0, "could not bind the fake guest service")
        XCTAssertEqual(listen(fd, 4), 0)
        defer { close(fd) }
        Thread.detachNewThread {
            let client = accept(fd, nil, nil)
            guard client >= 0 else { return }
            var buf = [UInt8](repeating: 0, count: 1024)
            let n = read(client, &buf, buf.count)
            var reply = Array("echo:".utf8)
            if n > 0 { reply += buf[0..<n] }
            _ = reply.withUnsafeBytes { write(client, $0.baseAddress, $0.count) }
            close(client)
        }

        let fwd = SandboxPortForwarder()
        defer { fwd.stop() }
        fwd.targetPortOverride = { _ in servicePort }
        fwd.setTarget(host: "::1")
        fwd.update(ports: [port])

        // Give the forwarder's listener a beat to bind, then connect to the
        // HOST side (IPv4 loopback) and expect the guest service's reply.
        let reply = expectation(description: "relayed reply")
        let client = NWConnection(
            host: "127.0.0.1", port: NWEndpoint.Port(rawValue: port)!, using: .tcp)
        client.stateUpdateHandler = { state in
            if case .ready = state {
                client.send(content: Data("ping".utf8), completion: .contentProcessed { _ in })
                client.receive(minimumIncompleteLength: 1, maximumLength: 1024) { data, _, _, _ in
                    if let data, String(decoding: data, as: UTF8.self) == "echo:ping" {
                        reply.fulfill()
                    }
                }
            }
        }
        client.start(queue: q)
        defer { client.cancel() }
        wait(for: [reply], timeout: 10)
        XCTAssertEqual(fwd.activePorts, [port])
    }

    func testForwarderUpdateClosesRemovedPorts() {
        let fwd = SandboxPortForwarder()
        defer { fwd.stop() }
        fwd.setTarget(host: "::1")
        let a = UInt16.random(in: 41000...50000)
        let b = UInt16.random(in: 50001...60000)
        fwd.update(ports: [a, b])
        XCTAssertEqual(fwd.activePorts, [a, b])
        fwd.update(ports: [b])
        XCTAssertEqual(fwd.activePorts, [b])
        fwd.stop()
        XCTAssertTrue(fwd.activePorts.isEmpty)
    }

    func testForwarderWithoutTargetOpensNothing() {
        let fwd = SandboxPortForwarder()
        defer { fwd.stop() }
        fwd.update(ports: [12345])
        XCTAssertTrue(fwd.activePorts.isEmpty,
                      "no guest IP yet → nothing to forward to, so no host listener")
    }
}
