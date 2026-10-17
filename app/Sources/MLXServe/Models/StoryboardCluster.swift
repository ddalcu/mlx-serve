import Foundation

/// A machine that renders storyboard shots: this Mac, or a LAN peer serving
/// the same pack. One shot at a time each: a server runs one H3 job at once.
struct RenderDevice: Equatable, Identifiable {
    /// The model name a request carries: the local registry id, or `<id>@<peer>`.
    let modelId: String
    /// nil = this Mac.
    let peer: String?
    var id: String { peer ?? "" }
}

/// Shots render in parallel across machines when every join has a picture: then
/// no shot waits for another's last frame (`Storyboard.shotRequest`).
enum StoryboardCluster {

    /// LAN rows of the same pack as `repo`. The pack's folder name carries its
    /// quant, so the same name renders the same shot on any machine; a peer may
    /// list it under its org (`org/name`) or bare.
    static func peers(of repo: String, lanModels: [ModelInfo]) -> [RenderDevice] {
        let name = (repo as NSString).lastPathComponent
        return lanModels.compactMap { m in
            guard let peer = m.lanPeer, m.lanAdvertises("video"),
                  (LanPick.base(of: m.name) as NSString).lastPathComponent == name else { return nil }
            return RenderDevice(modelId: m.name, peer: peer)
        }
    }

    /// What the pane calls a machine.
    static func name(_ device: RenderDevice) -> String {
        device.peer ?? L10n.text("This Mac")
    }

    /// "about 1 h 30 min on 3 machines, 40 min sooner than this Mac alone".
    static func describe(alone: Double, cluster: Double, machines: Int) -> String {
        let saved = alone - cluster
        guard saved >= 60 else {
            return L10n.format("%@ on this Mac; your cluster would not finish it sooner", H3TimeEstimate.duration(alone))
        }
        return L10n.format("%@ on %lld machines, %@ sooner than this Mac alone",
                           H3TimeEstimate.duration(cluster), Int64(machines),
                           H3TimeEstimate.duration(saved).replacingOccurrences(of: "about ", with: ""))
    }

    /// Whether a LAN peer name and a host name are one Mac: LAN sharing names a Mac by its
    /// host name unless told otherwise, with or without ".local".
    static func sameMachine(_ peer: String?, _ host: String) -> Bool {
        guard let peer, !host.isEmpty else { return false }
        func key(_ s: String) -> String {
            String(ClusterVideoAdvice.shortName(s).lowercased().filter { $0.isLetter || $0.isNumber })
        }
        return key(peer) == key(host)
    }

    /// Every join but the film's end has a picture.
    static func joinsAnchored(_ endFrames: [String?]) -> Bool {
        endFrames.dropLast().allSatisfy { $0 != nil }
    }

    /// Wall seconds for `costs` (each shot's seconds on the anchor machine) on
    /// devices whose speeds are `factors` (seconds per anchor second).
    static func wallSeconds(costs: [Double], factors: [Double]) -> Double {
        var d = ShotDispatcher(costs: costs, factors: factors)
        var t = 0.0
        while true {
            for i in factors.indices where d.idle(i) { _ = d.next(for: i, now: t) }
            guard let soonest = d.busyUntil.indices.filter({ !d.idle($0) }).map({ d.busyUntil[$0] }).min()
            else { break }
            t = soonest
            for i in factors.indices where !d.idle(i) && d.busyUntil[i] <= t { d.finished(i) }
        }
        return t
    }
}

/// Which shot a free device starts next. Every time a device frees up, the
/// unstarted shots are planned again, longest first, each onto the device that
/// would finish it soonest given what that device already holds; the free device
/// starts its largest planned shot, or waits when the plan gives it none. So a
/// slow machine takes a shot only when that brings the end of the film closer.
struct ShotDispatcher {
    let costs: [Double]
    private(set) var factors: [Double]
    private(set) var pending: [Int]
    /// Predicted end of each device's current shot; nil while idle.
    private var running: [Double?]
    private var alive: [Bool]

    init(costs: [Double], factors: [Double]) {
        self.costs = costs
        self.factors = factors
        pending = costs.indices.sorted { costs[$0] > costs[$1] }
        running = Array(repeating: nil, count: factors.count)
        alive = Array(repeating: true, count: factors.count)
    }

    var busyUntil: [Double] { running.map { $0 ?? 0 } }
    var done: Bool { pending.isEmpty && running.allSatisfy { $0 == nil } }
    func idle(_ device: Int) -> Bool { running[device] == nil }
    func usable(_ device: Int) -> Bool { alive[device] }

    /// The shot `device` should start at `now`, or nil to stay idle for now.
    mutating func next(for device: Int, now: Double) -> Int? {
        guard alive[device], running[device] == nil else { return nil }
        var load = factors.indices.map { alive[$0] ? max(running[$0] ?? now, now) : .infinity }
        for (k, s) in pending.enumerated() {
            let ends = factors.indices.map { load[$0] + costs[s] * factors[$0] }
            let best = ends.indices.min { ends[$0] < ends[$1] || (ends[$0] == ends[$1] && $0 == device) }!
            if best == device {
                pending.remove(at: k)
                running[device] = ends[best]
                return s
            }
            load[best] = ends[best]
        }
        return nil
    }

    var anyAlive: Bool { alive.contains(true) }

    mutating func finished(_ device: Int) { running[device] = nil }

    /// A device whose shot failed leaves the run; its shot goes back in line.
    mutating func failed(_ device: Int, shot: Int) {
        alive[device] = false
        running[device] = nil
        pending.append(shot)
        pending.sort { costs[$0] > costs[$1] }
    }

    /// A measured speed replaces the guess as soon as a device finishes a shot.
    mutating func measured(_ device: Int, factor: Double) {
        guard factor.isFinite, factor > 0 else { return }
        factors[device] = factor
    }
}
