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
    static let servableTypes: Set<String> = ["qwen4_exp", "deepseek_v41", "minimax_h3"]
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
    /// the cable, read off this Mac's own port (a worker's x.y.z.2 → x.y.z.1). A
    /// self-assigned 169.254 address belongs to the Thunderbolt Bridge, not the pair.
    static func coordinator(_ o: ServerOptions, ports: () -> [RdmaPort] = ClusterSetup.ports,
                            address: (String) -> String? = ClusterSetup.ipv4(on:)) -> String {
        let typed = o.clusterCoordinator.trimmingCharacters(in: .whitespaces)
        if !typed.isEmpty { return typed.contains(":") ? typed : "\(typed):\(port)" }
        var host = linkAddress(.host)
        if let dev = device(o, ports: ports), let ip = address(RdmaPort(device: dev, active: true).interface),
           !ip.hasPrefix("169.254.") {
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
}

/// What keeps the cable's address across restarts: a launchd job that runs as root and gives
/// the Thunderbolt port with a cable this Mac's pair address (or takes it away), at boot, when
/// the role file changes and every 30 s. One password prompt installs it; after that pairing
/// only writes the role file, which holds nothing but host, worker or off.
enum ThunderboltAddressJob {
    static let label = "com.mlx-serve.thunderbolt-address"
    static let dir = "/Library/Application Support/MLX-Serve"
    static let roleFile = dir + "/thunderbolt-role"
    static let scriptPath = dir + "/thunderbolt-address.sh"
    static let plistPath = "/Library/LaunchDaemons/\(label).plist"

    static var installed: Bool {
        FileManager.default.fileExists(atPath: plistPath) && FileManager.default.isWritableFile(atPath: roleFile)
    }

    static var script: String {
        let host = ClusterSetup.linkAddress(.host), worker = ClusterSetup.linkAddress(.worker)
        return """
        #!/bin/sh
        # Installed by MLX Serve: gives the Thunderbolt port with a cable this Mac's pair address.
        case "$(/bin/cat '\(roleFile)' 2>/dev/null)" in
          host) mine=\(host) ;;
          worker) mine=\(worker) ;;
          *) mine= ;;
        esac
        port=$(/usr/bin/ibv_devinfo 2>/dev/null | /usr/bin/awk '/hca_id:/ {d=$2} /PORT_ACTIVE/ {print substr(d, 6); exit}')
        for ifc in $(/sbin/ifconfig -l); do
          for a in \(host) \(worker); do
            if [ "$ifc/$a" != "$port/$mine" ] && /sbin/ifconfig "$ifc" | /usr/bin/grep -qF "inet $a "; then
              /sbin/ifconfig "$ifc" inet "$a" -alias
            fi
          done
        done
        if [ -n "$mine" ] && [ -n "$port" ] && ! /sbin/ifconfig "$port" | /usr/bin/grep -qF "inet $mine "; then
          /sbin/ifconfig "$port" inet "$mine" netmask 255.255.255.0 alias
        fi

        """
    }

    static var plist: String {
        """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key><string>\(label)</string>
            <key>ProgramArguments</key><array><string>/bin/sh</string><string>\(scriptPath)</string></array>
            <key>RunAtLoad</key><true/>
            <key>StartInterval</key><integer>30</integer>
            <key>WatchPaths</key><array><string>\(roleFile)</string></array>
        </dict>
        </plist>

        """
    }

    /// The root shell line that installs the job with `role`. The files ride in base64, so
    /// nothing in it needs quoting inside AppleScript.
    static func installCommand(role: ClusterRole) -> String {
        let b64 = { (s: String) in Data(s.utf8).base64EncodedString() }
        return "/bin/mkdir -p '\(dir)'"
            + " && echo \(b64(script)) | /usr/bin/base64 -D > '\(scriptPath)' && /bin/chmod 755 '\(scriptPath)'"
            + " && echo \(b64(plist)) | /usr/bin/base64 -D > '\(plistPath)' && /bin/chmod 644 '\(plistPath)'"
            + " && printf \(role.rawValue) > '\(roleFile)' && /bin/chmod 666 '\(roleFile)'"
            + " && (/bin/launchctl bootout system/\(label) 2>/dev/null; /bin/launchctl bootstrap system '\(plistPath)')"
    }

    /// Puts the role file back in step with the app's role, when the job is in and they differ.
    static func sync(_ role: ClusterRole) {
        guard installed, (try? String(contentsOfFile: roleFile, encoding: .utf8)) != role.rawValue else { return }
        _ = apply(role)
    }

    /// Sets this Mac's role on the cable: the role file once the job is in, else the one
    /// password prompt that installs it. The error text, or nil.
    static func apply(_ role: ClusterRole) -> String? {
        if installed {
            do {
                try role.rawValue.write(toFile: roleFile, atomically: false, encoding: .utf8)
                return nil
            } catch {
                return error.localizedDescription
            }
        }
        guard role != .off else { return nil }
        let script = "do shell script \"\(installCommand(role: role))\" with administrator privileges"
        var error: NSDictionary?
        NSAppleScript(source: script)?.executeAndReturnError(&error)
        return error.map { ($0[NSAppleScript.errorMessage] as? String) ?? L10n.text("Could not set up the cable.") }
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
    /// Whether the worker has the pair's model: "ok", "missing", "different" or "unknown".
    var workerModel: String = "unknown"
    /// The worker left; the server must restart to pair again.
    var workerGone: Bool = false

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
                           workerFreeMib: c["worker_free_mib"] as? Int ?? 0,
                           workerModel: c["worker_model"] as? String ?? "unknown",
                           workerGone: c["worker_gone"] as? Bool ?? false)
    }
}

/// What pairing a model would put on each Mac, split the way the server splits it: layers by
/// free memory (`distributed.chooseSplit`), experts half each, a video model whole on both.
enum PairPlan {
    enum Fit: Equatable { case alone, needsBoth, tooBig }

    static func share(bytes: Int64, video: Bool, mode: ClusterMode, freeHere: Int64, freeThere: Int64) -> (here: Int64, there: Int64) {
        if video { return (bytes, bytes) }
        switch mode {
        case .tensor:
            return (bytes / 2, bytes - bytes / 2)
        case .pipeline:
            let here = Int64(Double(bytes) * Double(freeHere) / Double(max(freeHere + freeThere, 1)))
            return (here, bytes - here)
        }
    }

    /// Whether the weights fit beside a tenth of headroom on this Mac, on both, or on neither
    /// (`freeThere` nil: the other Mac's memory is not known yet).
    static func fit(bytes: Int64, freeHere: Int64, freeThere: Int64?) -> Fit {
        let need = Double(bytes) * 1.1
        if need <= Double(freeHere) { return .alone }
        if let freeThere, need > Double(freeHere + freeThere) { return .tooBig }
        return .needsBoth
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
}
