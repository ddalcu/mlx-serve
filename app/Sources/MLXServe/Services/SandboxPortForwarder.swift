import Foundation
import Network

// MARK: - Guest net-report parsing (pure — unit tested)

/// Splits the guest's hvc2 byte stream into complete snapshots. The guest's
/// monitor loop frames each report with a trailing `=EOS=` line; bytes after
/// the last sentinel are buffered until the next feed. Bounded: a runaway
/// unframed stream is dropped past 1 MB rather than growing forever.
final class GuestNetSnapshotSplitter: @unchecked Sendable {
    private let lock = NSLock()
    private var pending = ""

    func feed(_ data: Data) -> [String] {
        lock.lock(); defer { lock.unlock() }
        pending += String(decoding: data, as: UTF8.self)
        var out: [String] = []
        while let r = pending.range(of: "=EOS=") {
            out.append(String(pending[..<r.lowerBound]))
            pending = String(pending[r.upperBound...])
        }
        if pending.count > 1 << 20 { pending = "" }
        return out
    }
}

/// Parses one snapshot: the `=IP=<addr>` line plus raw `/proc/net/tcp` and
/// `/proc/net/tcp6` contents. A port is forwardable when some socket LISTENs
/// (state 0A) on a non-loopback address — a guest server bound to the guest's
/// own 127.0.0.1/::1 is unreachable from the host via the NAT address, so
/// mirroring it would only produce dead listeners.
enum GuestNetParser {
    struct Snapshot: Equatable {
        var ip: String?
        var ports: Set<UInt16> = []
        /// From the guest's /proc/meminfo (kB) — feeds the tray RAM readout.
        var memTotalKB: Int?
        var memAvailableKB: Int?
    }

    // /proc/net/tcp encodes addresses as little-endian hex words.
    private static let v4Loopback = "0100007F" // 127.0.0.1
    private static let v6Loopback = "00000000000000000000000001000000" // ::1

    static func parse(_ text: String) -> Snapshot {
        var snap = Snapshot()
        // hvc2 is a tty: the guest's \n arrives as \r\n, and in Swift "\r\n" is
        // ONE grapheme cluster — split(separator: "\n") would not split at all
        // (the whole snapshot became a single "line" in the live smoke). Split
        // on any newline scalar and trim the \r remnants.
        for rawLine in text.split(whereSeparator: \.isNewline) {
            let line = Substring(rawLine.trimmingCharacters(in: .whitespacesAndNewlines))
            if line.hasPrefix("=IP=") {
                let addr = String(line.dropFirst(4))
                if !addr.isEmpty { snap.ip = addr }
                continue
            }
            if line.hasPrefix("MemTotal:") {
                snap.memTotalKB = meminfoKB(line)
                continue
            }
            if line.hasPrefix("MemAvailable:") {
                snap.memAvailableKB = meminfoKB(line)
                continue
            }
            if let port = listeningPort(line) { snap.ports.insert(port) }
        }
        return snap
    }

    /// "MemTotal:        1010536 kB" → 1010536.
    private static func meminfoKB(_ line: Substring) -> Int? {
        let fields = line.split(separator: " ", omittingEmptySubsequences: true)
        guard fields.count >= 2 else { return nil }
        return Int(fields[1])
    }

    /// One `/proc/net/tcp[6]` row → its port, iff LISTEN on a non-loopback bind.
    static func listeningPort(_ line: Substring) -> UInt16? {
        let fields = line.split(separator: " ", omittingEmptySubsequences: true)
        guard fields.count >= 4, fields[3] == "0A" else { return nil }
        let local = fields[1].split(separator: ":")
        guard local.count == 2,
              let port = UInt16(local[1], radix: 16), port != 0 else { return nil }
        let addr = String(local[0])
        if addr == v4Loopback || addr == v6Loopback { return nil }
        return port
    }
}

// MARK: - Host-side forwarder

/// Mirrors the sandbox guest's listening TCP ports onto the HOST's localhost:
/// when the agent starts a server on guest port 8080, `127.0.0.1:8080` on the
/// Mac transparently reaches it (same port number, live — listeners open and
/// close as the guest's `/proc/net/tcp` changes).
///
/// Bind scope is deliberately loopback only — BOTH 127.0.0.1 and ::1, since
/// `localhost` resolves to ::1 first in modern clients and an IPv4-only
/// listener leaves them refused — for the user's browser/tools on this Mac,
/// not for exposing the sandbox to the LAN.
/// A host port already in use (e.g. the mlx-serve server itself) fails to bind
/// and is skipped with a log line — never an error surfaced to the agent.
final class SandboxPortForwarder: @unchecked Sendable {
    private let queue = DispatchQueue(label: "mlxserve.sandbox.portfwd")
    private let lock = NSLock()
    /// Per port: one listener per loopback family (v4 + v6). The port maps
    /// only while BOTH bind — a partial bind would let `localhost` reach the
    /// guest on one family and a different host service on the other.
    private var listeners: [UInt16: [NWListener]] = [:]
    private var target: NWEndpoint.Host?

    /// Called (on the forwarder queue) whenever the set of mapped ports changes.
    var onMappingsChanged: ((Set<UInt16>) -> Void)?

    /// TEST SEAM: maps a host port to the target-side port (identity in
    /// production — guest port N appears at localhost:N). Hermetic tests can't
    /// bind the fake guest service and the forwarder to the same port on
    /// loopback (NWListener probes both address families), so they forward
    /// P → Q instead; the same-port identity is covered live by SandboxSmoke.
    var targetPortOverride: ((UInt16) -> UInt16)?

    var activePorts: Set<UInt16> {
        lock.lock(); defer { lock.unlock() }
        return Set(listeners.keys)
    }

    /// The guest address to forward to. Ports reported before the address is
    /// known are ignored (update is called again on the next snapshot).
    func setTarget(host: String) {
        lock.lock()
        let changed = target != NWEndpoint.Host(host)
        target = NWEndpoint.Host(host)
        lock.unlock()
        if changed { closeAll() } // stale listeners point at the old address
    }

    /// Reconcile the mapped ports with the guest's current listener set.
    func update(ports: Set<UInt16>) {
        lock.lock()
        guard let host = target else { lock.unlock(); return }
        let current = Set(listeners.keys)
        let toClose = current.subtracting(ports)
        let toOpen = ports.subtracting(current)
        var closed: [NWListener] = []
        for p in toClose {
            if let ls = listeners.removeValue(forKey: p) { closed.append(contentsOf: ls) }
        }
        lock.unlock()
        closed.forEach { $0.cancel() }
        for p in toOpen { open(port: p, host: host) }
        if !toClose.isEmpty || !toOpen.isEmpty { onMappingsChanged?(activePorts) }
    }

    func stop() {
        closeAll()
    }

    private func closeAll() {
        lock.lock()
        let all = listeners
        listeners = [:]
        lock.unlock()
        all.values.forEach { $0.forEach { $0.cancel() } }
        if !all.isEmpty { onMappingsChanged?([]) }
    }

    private func open(port: UInt16, host: NWEndpoint.Host) {
        guard let nwPort = NWEndpoint.Port(rawValue: port) else { return }
        let targetRaw = targetPortOverride?(port) ?? port
        guard let targetPort = NWEndpoint.Port(rawValue: targetRaw) else { return }
        var opened: [NWListener] = []
        for bindAddr in [NWEndpoint.Host("127.0.0.1"), NWEndpoint.Host("::1")] {
            let params = NWParameters.tcp
            params.allowLocalEndpointReuse = true
            params.requiredLocalEndpoint = NWEndpoint.hostPort(host: bindAddr, port: nwPort)
            guard let listener = try? NWListener(using: params) else {
                NSLog("[sandbox] port map: could not create listener for \(port) on \(bindAddr)")
                opened.forEach { $0.cancel() }
                return
            }
            listener.newConnectionHandler = { [queue] client in
                Self.relay(client: client, to: host, port: targetPort, on: queue)
            }
            listener.stateUpdateHandler = { [weak self, weak listener] state in
                if case .failed(let err) = state {
                    // Typically addressInUse (something on the host already owns
                    // the port, e.g. mlx-serve itself). Drop the WHOLE mapping —
                    // sibling family included — keep the other ports.
                    NSLog("[sandbox] port map \(port) unavailable on host: \(err)")
                    listener?.cancel()
                    guard let self, let listener else { return }
                    self.lock.lock()
                    let mine = self.listeners[port]?.contains(where: { $0 === listener }) ?? false
                    let siblings = mine ? (self.listeners.removeValue(forKey: port) ?? []) : []
                    self.lock.unlock()
                    siblings.forEach { $0.cancel() }
                }
            }
            opened.append(listener)
        }
        lock.lock()
        listeners[port] = opened
        lock.unlock()
        opened.forEach { $0.start(queue: queue) }
    }

    /// Bidirectional byte pump between an accepted host connection and a fresh
    /// connection into the guest. Either side closing/erroring tears down both.
    private static func relay(client: NWConnection, to host: NWEndpoint.Host, port: NWEndpoint.Port, on queue: DispatchQueue) {
        let upstream = NWConnection(host: host, port: port, using: .tcp)
        client.start(queue: queue)
        upstream.start(queue: queue)
        pump(from: client, to: upstream)
        pump(from: upstream, to: client)
    }

    private static func pump(from src: NWConnection, to dst: NWConnection) {
        src.receive(minimumIncompleteLength: 1, maximumLength: 128 << 10) { data, _, isComplete, error in
            if let data, !data.isEmpty {
                dst.send(content: data, completion: .contentProcessed { _ in })
            }
            if isComplete || error != nil {
                src.cancel()
                dst.cancel()
                return
            }
            pump(from: src, to: dst)
        }
    }
}
