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

    /// The IPv4 address on `interface`, nil when it has none.
    static func ipv4(on interface: String) -> String? {
        var head: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&head) == 0 else { return nil }
        defer { freeifaddrs(head) }
        var cur = head
        while let ifa = cur {
            defer { cur = ifa.pointee.ifa_next }
            guard String(cString: ifa.pointee.ifa_name) == interface,
                  let addr = ifa.pointee.ifa_addr, addr.pointee.sa_family == UInt8(AF_INET) else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            if getnameinfo(addr, socklen_t(addr.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0 {
                return String(cString: host)
            }
        }
        return nil
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

    /// The shell line that gives `interface` this Mac's cluster address (until restart).
    static func addressCommand(interface: String, role: ClusterRole) -> String {
        "/sbin/ifconfig \(interface) inet \(linkAddress(role)) netmask 255.255.255.0 alias"
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

    static func parse(_ json: [String: Any]) -> ClusterInfo? {
        guard let c = json["cluster"] as? [String: Any] else { return nil }
        return ClusterInfo(mode: c["mode"] as? String ?? "", model: c["model"] as? String ?? "",
                           joined: c["joined"] as? Bool ?? false, planned: c["planned"] as? Bool ?? false,
                           workerReady: c["worker_ready"] as? Bool ?? false, layers: c["layers"] as? Int ?? 0,
                           here: (c["here"] as? [Int])?.last ?? 0)
    }
}
