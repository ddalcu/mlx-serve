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

    /// The device a shot's `machine` choice names in `devices` (this Mac first); nil = any,
    /// or a machine that is not there now.
    static func pin(_ machine: String?, in devices: [RenderDevice]) -> Int? {
        guard let machine else { return nil }
        return machine.isEmpty ? 0 : devices.firstIndex { $0.peer == machine }
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
        simulate(costs: costs, factors: factors, pins: []).seconds
    }

    /// The run `ShotDispatcher` plans: when it ends, and the machine each shot starts on
    /// (`pins` = the machine the user chose per shot, nil = any).
    static func simulate(costs: [Double], factors: [Double], pins: [Int?]) -> (seconds: Double, machine: [Int]) {
        var d = ShotDispatcher(costs: costs, factors: factors, pins: pins)
        var machine = [Int](repeating: 0, count: costs.count)
        var claimed = Set<Int>()
        var t = 0.0
        while true {
            for i in factors.indices where d.idle(i) {
                if let s = d.next(for: i, now: t), claimed.insert(s).inserted { machine[s] = i }
            }
            guard let soonest = d.busyUntil.indices.filter({ !d.idle($0) }).map({ d.busyUntil[$0] }).min()
            else { break }
            t = soonest
            for i in factors.indices where !d.idle(i) && d.busyUntil[i] <= t { d.finished(i) }
        }
        return (t, machine)
    }
}

/// Which shot a free device starts next. Every time a device frees up, the
/// unstarted shots are planned again, longest first, each onto the device that
/// would finish it soonest given what that device already holds; the free device
/// starts its largest planned shot, or waits when the plan gives it none. So a
/// slow machine takes a shot only when that brings the end of the film closer.
/// Once nothing is left to start, a free device renders again the shot that would
/// end last where it runs, when it would finish it well before: a machine whose
/// speed was guessed wrong never makes the film end later than the others alone.
struct ShotDispatcher {
    let costs: [Double]
    private(set) var factors: [Double]
    private(set) var pending: [Int]
    /// The shot each device is rendering and when it started it; nil while idle.
    private var shot: [Int?]
    private var started: [Double]
    private var alive: [Bool]
    /// The device the user put each shot on; nil = whichever plans best.
    private var pins: [Int?]

    /// A second copy starts only when the first would take this much longer to finish.
    static let rerunGain = 1.25

    init(costs: [Double], factors: [Double], pins: [Int?] = []) {
        self.costs = costs
        self.factors = factors
        self.pins = costs.indices.map { $0 < pins.count ? pins[$0].flatMap { factors.indices.contains($0) ? $0 : nil } : nil }
        pending = costs.indices.sorted { costs[$0] > costs[$1] }
        shot = Array(repeating: nil, count: factors.count)
        started = Array(repeating: 0, count: factors.count)
        alive = Array(repeating: true, count: factors.count)
    }

    /// Predicted end of each device's current shot (0 while idle).
    var busyUntil: [Double] { factors.indices.map { end($0) ?? 0 } }
    var done: Bool { pending.isEmpty && shot.allSatisfy { $0 == nil } }
    func idle(_ device: Int) -> Bool { shot[device] == nil }
    func usable(_ device: Int) -> Bool { alive[device] }

    private func end(_ d: Int) -> Double? { shot[d].map { started[d] + costs[$0] * factors[d] } }

    /// The shot `device` should start at `now`, or nil to stay idle for now.
    mutating func next(for device: Int, now: Double) -> Int? {
        guard alive[device], shot[device] == nil else { return nil }
        var load = factors.indices.map { alive[$0] ? max(end($0) ?? now, now) : .infinity }
        for (k, s) in pending.enumerated() {
            if let p = pins[s] {
                if p == device {
                    pending.remove(at: k)
                    return start(s, on: device, now: now)
                }
                load[p] += costs[s] * factors[p]
                continue
            }
            let ends = factors.indices.map { load[$0] + costs[s] * factors[$0] }
            let best = ends.indices.min { ends[$0] < ends[$1] || (ends[$0] == ends[$1] && $0 == device) }!
            if best == device {
                pending.remove(at: k)
                return start(s, on: device, now: now)
            }
            load[best] = ends[best]
        }
        guard pending.isEmpty else { return nil }
        // Nothing left to start: take over the shot that ends last, run once so far.
        let shotCosts = costs, mineFactor = factors[device]
        let mine = { (s: Int) in shotCosts[s] * mineFactor }
        let rerun = factors.indices
            .compactMap { j -> (shot: Int, end: Double)? in
                guard let s = shot[j], pins[s] == nil, let e = end(j), shot.filter({ $0 == s }).count == 1 else { return nil }
                return (s, e)
            }
            .filter { $0.end - now > mine($0.shot) * Self.rerunGain }
            .max { $0.end < $1.end }
        return rerun.map { start($0.shot, on: device, now: now) }
    }

    private mutating func start(_ s: Int, on device: Int, now: Double) -> Int {
        shot[device] = s
        started[device] = now
        return s
    }

    var anyAlive: Bool { alive.contains(true) }

    /// `device` finished its shot; returns the devices still rendering another copy
    /// of it, now idle, whose copies the caller cancels.
    @discardableResult
    mutating func finished(_ device: Int) -> [Int] {
        guard let s = shot[device] else { return [] }
        shot[device] = nil
        let others = factors.indices.filter { shot[$0] == s }
        for j in others { shot[j] = nil }
        return others
    }

    /// A device whose shot failed leaves the run; its shot goes back in line
    /// unless another copy of it is still rendering.
    mutating func failed(_ device: Int, shot s: Int) {
        alive[device] = false
        shot[device] = nil
        for i in pins.indices where pins[i] == device { pins[i] = nil }
        guard !shot.contains(s) else { return }
        pending.append(s)
        pending.sort { costs[$0] > costs[$1] }
    }

    /// A measured speed replaces the guess: from a shot's progress, and when it finishes.
    mutating func measured(_ device: Int, factor: Double) {
        guard factor.isFinite, factor > 0 else { return }
        factors[device] = factor
    }
}
