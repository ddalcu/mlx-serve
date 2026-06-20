import Foundation

/// Facts the model can't know on its own and must be told fresh each turn —
/// today the wall-clock date/time. Injected into agent and voice system prompts
/// so the assistant answers "what time/day is it" from reality instead of
/// hallucinating (and so it reasons about recency correctly). Pure → testable.
enum SystemGrounding {
    /// One sentence stating the current local date and time, with an instruction
    /// to trust it over the model's own guess.
    static func dateTimeLine(now: Date = Date()) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US")
        f.dateFormat = "EEEE, MMMM d, yyyy 'at' h:mm a zzz"
        return "The current date and time is \(f.string(from: now)). " +
            "Treat this as the present moment — do not guess the date or time from memory; " +
            "answer any date or time question from this."
    }

    /// Date-only grounding (no clock time) for the AGENT system prompt.
    ///
    /// Caching matters here: the server reuses a KV prefix across requests only
    /// while the prompt prefix is byte-identical. A per-minute timestamp changes
    /// every request, so prepending `dateTimeLine` defeated the cache entirely —
    /// every agent turn cold-re-prefilled the full multi-thousand-token system +
    /// tools prompt (slow TTFB even on a fresh chat). Date-only changes at most
    /// once a day; placed at the END of the prompt, even the daily rollover only
    /// re-prefills the short tail. Voice (short, cache-insensitive turns) keeps
    /// the full `dateTimeLine`.
    static func dateLine(now: Date = Date()) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US")
        f.dateFormat = "EEEE, MMMM d, yyyy"
        return "Today's date is \(f.string(from: now)). " +
            "Treat this as the present day — answer any date question from this, not from memory."
    }

    /// A grounding sentence giving this Mac's LAN IP so the agent reports
    /// reachable URLs (http://<ip>:<port>) for any server/app it starts. Returns
    /// "" when offline / no IP. Pure on the resolved string → testable.
    nonisolated static func localIPLine(ip: String?) -> String {
        guard let ip, !ip.isEmpty else { return "" }
        return "This Mac's local network IP is \(ip). When you start a server or app, " +
            "bind it to 0.0.0.0 (not just localhost) and give the user a reachable URL like " +
            "http://\(ip):<port> so they can open it from this machine or another device on the " +
            "same network (e.g. their phone)."
    }

    /// Pick the most useful LAN IPv4 from enumerated (interface, ip) pairs:
    /// prefer en0 (Wi-Fi on Macs), then en1 (Ethernet), then any remaining
    /// address. Loopback / link-local (169.254.x) are filtered out. Pure so the
    /// selection policy is unit-tested without touching the live network stack.
    nonisolated static func pickLanIPv4(from candidates: [(interface: String, ip: String)]) -> String? {
        let usable = candidates.filter { $0.ip != "127.0.0.1" && !$0.ip.hasPrefix("169.254.") }
        func first(_ name: String) -> String? { usable.first { $0.interface == name }?.ip }
        return first("en0") ?? first("en1") ?? usable.first?.ip
    }

    /// This Mac's current LAN IPv4 (e.g. "192.168.1.42"), or nil when offline.
    static func localIPAddress() -> String? {
        pickLanIPv4(from: enumerateIPv4())
    }

    /// Enumerate up, non-loopback IPv4 interfaces as (name, dotted-quad) via
    /// `getifaddrs`. The I/O shell around the pure `pickLanIPv4`.
    private static func enumerateIPv4() -> [(interface: String, ip: String)] {
        var results: [(String, String)] = []
        var ifaddr: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddr) == 0 else { return [] }
        defer { freeifaddrs(ifaddr) }
        var ptr = ifaddr
        while let p = ptr {
            defer { ptr = p.pointee.ifa_next }
            let flags = p.pointee.ifa_flags
            guard (flags & UInt32(IFF_UP)) != 0, (flags & UInt32(IFF_LOOPBACK)) == 0,
                  let addr = p.pointee.ifa_addr, addr.pointee.sa_family == UInt8(AF_INET) else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            if getnameinfo(addr, socklen_t(addr.pointee.sa_len), &host, socklen_t(host.count),
                           nil, 0, NI_NUMERICHOST) == 0 {
                results.append((String(cString: p.pointee.ifa_name), String(cString: host)))
            }
        }
        return results
    }
}
