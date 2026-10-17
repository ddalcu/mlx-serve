import Foundation

enum ClusterRole: String, Codable, CaseIterable { case off, host, worker }

/// `--dist`: by layers (pipeline) or by experts (tensor).
enum ClusterMode: String, Codable, CaseIterable { case pipeline, tensor }

/// A Thunderbolt RDMA port as `ibv_devinfo` lists it.
struct RdmaPort: Equatable {
    let device: String
    let active: Bool
    /// The network interface behind it: `rdma_en13` → `en13`.
    var interface: String { device.hasPrefix("rdma_") ? String(device.dropFirst(5)) : device }
}

/// How the two Macs of a Thunderbolt cluster find each other. Each port needs an
/// IPv4 address for RDMA to carry traffic; the host is .1 and the worker .2 of a
/// subnet of the cable's own, because the Thunderbolt Bridge already claims
/// 169.254/16 and would route the worker's call to the host away from the cable.
enum ClusterSetup {
    static let port = 32323
    /// Model types a cluster serves: Flash Next by layers or experts, H3 by video steps.
    static let servableTypes: Set<String> = ["qwen4_exp", "minimax_h3"]
    static let linkSubnet = "10.254.77"
    static func linkAddress(_ role: ClusterRole) -> String { "\(linkSubnet).\(role == .worker ? 2 : 1)" }

    /// `ibv_devinfo`'s ports, in its order.
    static func parsePorts(_ text: String) -> [RdmaPort] {
        var ports: [RdmaPort] = []
        var device: String?
        for line in text.split(separator: "\n") {
            let t = line.trimmingCharacters(in: .whitespaces)
            if t.hasPrefix("hca_id:") {
                device = t.dropFirst(7).trimmingCharacters(in: .whitespaces)
            } else if t.hasPrefix("state:"), let d = device {
                ports.append(RdmaPort(device: d, active: t.contains("PORT_ACTIVE")))
                device = nil
            }
        }
        return ports
    }

    /// The ports on this Mac; empty when RDMA is off (`rdma_ctl enable` in Recovery).
    static func ports() -> [RdmaPort] {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/ibv_devinfo")
        let out = Pipe()
        p.standardOutput = out
        p.standardError = FileHandle.nullDevice
        guard (try? p.run()) != nil else { return [] }
        let data = out.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        return parsePorts(String(decoding: data, as: UTF8.self))
    }

    /// The IPv4 address on `interface`, nil when it has none: the cluster's own when the port
    /// carries several (a self-assigned 169.254 one beside it).
    static func ipv4(on interface: String) -> String? {
        let all = ipv4s(on: interface)
        return all.first { $0.hasPrefix(linkSubnet + ".") } ?? all.first
    }

    private static func ipv4s(on interface: String) -> [String] {
        var out: [String] = []
        var head: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&head) == 0 else { return [] }
        defer { freeifaddrs(head) }
        var cur = head
        while let ifa = cur {
            defer { cur = ifa.pointee.ifa_next }
            guard String(cString: ifa.pointee.ifa_name) == interface,
                  let addr = ifa.pointee.ifa_addr, addr.pointee.sa_family == UInt8(AF_INET) else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            if getnameinfo(addr, socklen_t(addr.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0 {
                out.append(String(cString: host))
            }
        }
        return out
    }

    /// The port this Mac uses: the chosen one, else the first that is up.
    static func device(_ options: ServerOptions, ports: () -> [RdmaPort]) -> String? {
        options.clusterDevice.isEmpty ? ports().first(where: \.active)?.device : options.clusterDevice
    }

    /// Where the worker dials the host: the typed address, else the host's end of
    /// the cable, read off this Mac's own port (a worker's x.y.z.2 → x.y.z.1).
    static func coordinator(_ o: ServerOptions, ports: () -> [RdmaPort] = ClusterSetup.ports,
                            address: (String) -> String? = ClusterSetup.ipv4(on:)) -> String {
        let typed = o.clusterCoordinator.trimmingCharacters(in: .whitespaces)
        if !typed.isEmpty { return typed.contains(":") ? typed : "\(typed):\(port)" }
        var host = linkAddress(.host)
        if let dev = device(o, ports: ports), let ip = address(RdmaPort(device: dev, active: true).interface) {
            host = o.clusterRole == .worker ? ip.split(separator: ".").dropLast().joined(separator: ".") + ".1" : ip
        }
        return "\(host):\(port)"
    }

    /// The host's launch flags; none unless this Mac hosts and has a model and a port.
    static func hostArgs(_ o: ServerOptions, ports: () -> [RdmaPort] = ClusterSetup.ports,
                         address: (String) -> String? = ClusterSetup.ipv4(on:)) -> [String] {
        guard o.clusterRole == .host, !o.clusterModelPath.isEmpty, let dev = device(o, ports: ports) else { return [] }
        return ["--dist", o.clusterMode.rawValue, "--dist-coordinator", coordinator(o, ports: { [RdmaPort(device: dev, active: true)] }, address: address),
                "--dist-device", dev, "--dist-model", o.clusterModelPath]
    }

    /// `mlx-serve worker`'s arguments, or nil without a port.
    static func workerArgs(_ o: ServerOptions, modelDirs: [String], ports: () -> [RdmaPort] = ClusterSetup.ports,
                           address: (String) -> String? = ClusterSetup.ipv4(on:)) -> [String]? {
        guard let dev = device(o, ports: ports) else { return nil }
        return ["worker", "--coordinator", coordinator(o, ports: { [RdmaPort(device: dev, active: true)] }, address: address),
                "--device", dev]
            + ServerOptions.modelDirArgs(modelDirs)
    }

    /// The shell line that gives `interface` this Mac's cluster address (until restart),
    /// dropping the other role's first: a Mac that led and now joins would dial itself.
    static func addressCommand(interface: String, role: ClusterRole) -> String {
        let other = linkAddress(role == .worker ? .host : .worker)
        return "/sbin/ifconfig \(interface) inet \(other) -alias 2>/dev/null; "
            + "/sbin/ifconfig \(interface) inet \(linkAddress(role)) netmask 255.255.255.0 alias"
    }

    /// Runs `addressCommand` behind the macOS admin prompt; the error text, or nil on success.
    static func assignAddress(interface: String, role: ClusterRole) -> String? {
        let script = "do shell script \"\(addressCommand(interface: interface, role: role))\" with administrator privileges"
        var error: NSDictionary?
        NSAppleScript(source: script)?.executeAndReturnError(&error)
        return error.map { ($0[NSAppleScript.errorMessage] as? String) ?? "Could not set the address." }
    }
}

/// `/props` "cluster": the group this server hosts.
struct ClusterInfo: Equatable {
    var mode: String
    var model: String
    var joined: Bool
    var planned: Bool
    var workerReady: Bool
    var layers: Int
    /// Layers this Mac holds (`here` = [0, n)).
    var here: Int
    /// The worker's host name, as its LAN sharing names it.
    var workerName: String = ""
    /// Video steps (`mode == "rows"`): the worker's speed against this Mac's (0 until
    /// measured), the link in GB/s, and what the last video did ("split …" or "alone: …").
    var workerSpeed: Double = 0
    var linkGbps: Double = 0
    var lastJob: String = ""
    /// The last video's rows on each Mac; zeros when it ran here alone.
    var splitRows: [Int] = []
    var workerFreeMib: Int = 0

    var lastSplit: Bool { splitRows.count == 2 && splitRows[1] > 0 }

    /// The run-history key for videos whose steps were split with the worker.
    static let thunderboltPeer = "thunderbolt"

    /// Whether this cluster serves the pack in `dir` (the server matches by folder name).
    func serves(_ dir: String) -> Bool { !model.isEmpty && (dir as NSString).lastPathComponent == model }

    static func parse(_ json: [String: Any]) -> ClusterInfo? {
        guard let c = json["cluster"] as? [String: Any] else { return nil }
        return ClusterInfo(mode: c["mode"] as? String ?? "", model: c["model"] as? String ?? "",
                           joined: c["joined"] as? Bool ?? false, planned: c["planned"] as? Bool ?? false,
                           workerReady: c["worker_ready"] as? Bool ?? false, layers: c["layers"] as? Int ?? 0,
                           here: (c["here"] as? [Int])?.last ?? 0,
                           workerName: c["worker_name"] as? String ?? "",
                           workerSpeed: (c["worker_speed"] as? NSNumber)?.doubleValue ?? 0,
                           linkGbps: (c["link_gbps"] as? NSNumber)?.doubleValue ?? 0,
                           lastJob: c["last_job"] as? String ?? "",
                           splitRows: c["split_rows"] as? [Int] ?? [],
                           workerFreeMib: c["worker_free_mib"] as? Int ?? 0)
    }
}

/// What splitting video steps with the worker is worth, from the measured speeds. The server
/// decides each job on its own; this is the guidance the app shows around that decision.
enum ClusterVideoAdvice: Equatable {
    /// Joined, but nothing measured yet: the first video measures both Macs and the link.
    case unmeasured
    /// About as fast as this Mac: large videos share every step.
    case shares(speed: Double)
    /// Slower: only the largest videos gain, the rest run here.
    case largeOnly(speed: Double)
    /// Much slower: splitting would save little, so videos run here.
    case tooSlow(speed: Double)

    static func of(_ c: ClusterInfo) -> ClusterVideoAdvice {
        let r = c.workerSpeed
        if r <= 0 { return .unmeasured }
        if r >= 0.6 { return .shares(speed: r) }
        return r >= 0.3 ? .largeOnly(speed: r) : .tooSlow(speed: r)
    }

    /// One sentence naming the other Mac.
    func text(worker: String) -> String {
        let name = worker.isEmpty ? L10n.text("The other Mac") : ClusterVideoAdvice.shortName(worker)
        switch self {
        case .unmeasured:
            return L10n.text("The first video measures both Macs and the link, then each video uses the other Mac when that is faster.")
        case .shares(let r):
            return L10n.format("%@ runs at %@ this Mac's speed: every step of a large video is shared, up to %@ faster.",
                               name, Self.times(r), Self.times(1 + r * 0.8))
        case .largeOnly(let r):
            return L10n.format("%@ runs at %@ this Mac's speed: only the largest videos gain; the rest run here.", name, Self.times(r))
        case .tooSlow(let r):
            return L10n.format("%@ runs at %@ this Mac's speed, too slow to make one video faster, so videos stay on this Mac. It can still take whole storyboard shots over your network.",
                               name, Self.times(r))
        }
    }

    /// The worker's speed as a speed-up factor the estimate can divide by: the measured share
    /// minus what the swaps cost, never below 1.
    static func stepSpeedup(_ c: ClusterInfo) -> Double {
        switch of(c) {
        case .shares(let r), .largeOnly(let r): return max(1, 1 + r * 0.8)
        default: return 1
        }
    }

    static func shortName(_ host: String) -> String {
        host.hasSuffix(".local") ? String(host.dropLast(6)) : host
    }

    private static func times(_ r: Double) -> String { String(format: "%.1f×", r) }
}

/// Where this Mac stands with a Thunderbolt pair: two Macs joined by a cable that work on
/// the same job. The Cluster page shows one of these with one button.
enum ThunderboltPair: Equatable {
    /// No RDMA ports: Thunderbolt pairing is turned off in Recovery.
    case off
    /// RDMA is on but no cable is in a Thunderbolt 5 port.
    case noCable
    /// A cable is in and this Mac is not paired. `leader` is a Mac on the network that
    /// leads a pair and waits for one to join it.
    case ready(leader: String?)
    /// Paired, but the port lost its address (a restart): one click gives it back.
    case reconnect(ClusterRole)
    /// This Mac leads: `partner` is the other Mac once it has joined.
    case leading(partner: String?)
    /// This Mac helps the Mac that leads.
    case helping

    static func of(ports: [RdmaPort], role: ClusterRole, hasAddress: Bool,
                   cluster: ClusterInfo?, leader: String?) -> ThunderboltPair {
        if ports.isEmpty { return role == .off ? .off : .reconnect(role) }
        guard ports.contains(where: \.active) else { return .noCable }
        switch role {
        case .off: return .ready(leader: leader)
        case _ where !hasAddress: return .reconnect(role)
        case .host:
            guard let c = cluster, c.joined else { return .leading(partner: nil) }
            return .leading(partner: c.workerName.isEmpty ? L10n.text("the other Mac") : ClusterVideoAdvice.shortName(c.workerName))
        case .worker: return .helping
        }
    }

    /// A pair leader's `/health`: the model it leads with while no Mac has joined it.
    static func waitingLeader(health: Data) -> String? {
        guard let json = try? JSONSerialization.jsonObject(with: health) as? [String: Any],
              let tb = json["thunderbolt"] as? [String: Any],
              let model = tb["leads"] as? String, tb["joined"] as? Bool == false else { return nil }
        return model
    }
}

/// The video the Cluster page times every machine on: the Video window's current one
/// when it is an H3 video, else a 5-second 832×480 clip on the H3 pack.
struct ReferenceClip: Equatable {
    let model: VideoModelPreset
    let width: Int
    let height: Int
    let frames: Int
    let steps: Int
    let fast: Bool

    static func current(_ s: VideoGenSettings = .load()) -> ReferenceClip {
        if let m = VideoModelPreset.all.first(where: { $0.id == s.modelId }), m.backend == .minimaxH3 {
            let r = s.concreteResolution(for: m)
            return ReferenceClip(model: m, width: r.width, height: r.height, frames: s.numFrames,
                                 steps: s.steps, fast: !s.bestQuality && !(s.turbo && m.supportsTurbo))
        }
        let m = VideoModelPreset.minimaxH3Q4
        return ReferenceClip(model: m, width: 832, height: 480, frames: 121,
                             steps: m.settings(.good).steps, fast: true)
    }

    /// "a 10-second 800 × 640 video".
    var label: String {
        L10n.format("a %lld-second %lld × %lld video", Int64(max(1, Int((Double(frames) / 24).rounded()))),
                    Int64(width), Int64(height))
    }

    /// Seconds on a machine (`peer` nil = this Mac), and whether that machine's speed was measured.
    func seconds(peer: String?) -> (seconds: Double, measured: Bool) {
        let anchor = H3TimeEstimate.seconds(model: model, width: width, height: height, frames: frames,
                                            steps: steps, fast: fast, hardware: .anchor)
        let f = H3TimeEstimate.speedFactor(peer: peer)
        return (anchor * f.factor, f.measured)
    }
}
