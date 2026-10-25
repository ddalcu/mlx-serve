import Foundation
import CoreFoundation

enum MonitoringRange: Int, CaseIterable, Identifiable {
    case live = 60, tenMinutes = 600, thirtyMinutes = 1800, hour = 3600, day = 86400
    var id: Int { rawValue }
    var seconds: TimeInterval { Double(rawValue) }
    var title: String {
        switch self {
        case .live: L10n.text("Live")
        case .tenMinutes: L10n.text("10 min")
        case .thirtyMinutes: L10n.text("30 min")
        case .hour: L10n.text("1 hour")
        case .day: L10n.text("24 hours")
        }
    }
}

struct MonitoringSnapshot: Equatable {
    static let counterKeys = Set(["requests_success_total", "requests_failed_total", "requests_rejected_total",
        "requests_cancelled_total", "prompt_tokens_total", "prefill_tokens_total", "generation_tokens_total",
        "prefix_cache_queries_total", "prefix_cache_hits_total", "prefix_cache_tokens_total"])
    struct Histogram: Equatable {
        var sum: Double?
        var count: Double?
    }
    var counters: [String: Double]
    var gauges: [String: Double]
    var histograms: [String: Histogram]
    var sessions: [MonitoringSession]?

    static func number(_ value: Any?) -> Double? {
        guard let value = value as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID() else { return nil }
        let n = value.doubleValue
        return n.isFinite && n >= 0 ? n : nil
    }

    static func parse(_ data: Data) throws -> Self {
        guard let raw = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let counters = raw["counters"] as? [String: Any],
              let gauges = raw["gauges"] as? [String: Any] else { throw MonitoringError.unsupported }
        let histograms = (raw["histograms"] as? [String: [String: Any]] ?? [:]).mapValues {
            Histogram(sum: number($0["sum"]), count: number($0["count"]))
        }
        let sessions = (raw["sessions"] as? [[String: Any]])?.compactMap { row -> MonitoringSession? in
            guard let model = row["model"] as? String, let phase = row["phase"] as? String,
                  let id = row["request_id"] as? NSNumber, number(id) != nil else { return nil }
            return MonitoringSession(requestID: id.stringValue, model: model, phase: phase,
                client: row["client"] as? String ?? "—", context: number(row["context_tokens"]),
                capacity: number(row["context_length"]), cached: number(row["cached_tokens"]),
                generated: number(row["generated_tokens"]), stateBytes: number(row["state_bytes"]))
        }
        let numericCounters = counters.compactMapValues(number), numericGauges = gauges.compactMapValues(number)
        guard !numericCounters.isEmpty || !numericGauges.isEmpty else { throw MonitoringError.unsupported }
        return Self(counters: numericCounters, gauges: numericGauges, histograms: histograms, sessions: sessions)
    }

    func averageMilliseconds(_ name: String) -> Double? {
        guard let h = histograms[name], let sum = h.sum, let count = h.count, count > 0 else { return nil }
        return 1000 * sum / count
    }

    static func ratio(_ numerator: Double?, _ denominator: Double?) -> Double? {
        guard let numerator, let denominator, denominator > 0 else { return nil }
        return 100 * numerator / denominator
    }
}

struct MonitoringSession: Equatable {
    var requestID: String
    var model: String
    var phase: String
    var client: String
    var context: Double?
    var capacity: Double?
    var cached: Double?
    var generated: Double?
    var stateBytes: Double?
}

struct MonitoringSample: Codable, Equatable {
    var time: TimeInterval
    var process: Double?
    var model: String?
    var activeModels: Int?
    var counters: [String: Double]
    var breakBefore = false

    func follows(_ before: Self) -> Bool {
        !breakBefore && time > before.time && time - before.time <= 300 && !resets(before)
    }

    func resets(_ before: Self) -> Bool {
        if let process, let previous = before.process, process != previous { return true }
        return counters.contains { key, value in before.counters[key].map { value < $0 } ?? false }
    }
}

struct MonitoringRequest: Codable, Identifiable {
    var id = UUID()
    var requestID: String
    var process: Double?
    var model: String
    var client: String
    var phase: String
    var startTime: TimeInterval
    var lastTime: TimeInterval
    var endTime: TimeInterval?
    var context: Double?
    var cached: Double?
    var generated: Double?
}

struct MonitoringPoint: Identifiable {
    var time: TimeInterval
    var value: Double?
    var segment: Int
    var id: TimeInterval { time }
    var date: Date { Date(timeIntervalSince1970: time) }
}

struct MonitoringHistory: Codable {
    var samples: [MonitoringSample] = []
    var requests: [MonitoringRequest] = []
    private var lastCompaction: TimeInterval = 0
    private enum CodingKeys: String, CodingKey { case samples, requests }

    mutating func accept(_ snapshot: MonitoringSnapshot, at now: TimeInterval, afterFailure: Bool = false) {
        let models = Set((snapshot.sessions ?? []).filter { $0.phase != "cached" }.map(\.model))
        let sample = MonitoringSample(time: floor(now), process: snapshot.gauges["process_start_time_seconds"],
            model: models.count == 1 ? models.first : nil, activeModels: snapshot.sessions == nil ? nil : models.count,
            counters: snapshot.counters.filter { MonitoringSnapshot.counterKeys.contains($0.key) }, breakBefore: afterFailure)
        let reset = samples.last.map { sample.resets($0) } ?? false
        for index in requests.indices where requests[index].endTime == nil {
            if reset || now - requests[index].lastTime > 300 { requests[index].endTime = requests[index].lastTime }
        }
        if let sessions = snapshot.sessions {
            let active = sessions.filter { $0.phase != "cached" }
            let seen = Set(active.map(\.requestID))
            for index in requests.indices where requests[index].endTime == nil && !seen.contains(requests[index].requestID) {
                requests[index].endTime = now
            }
            for session in active {
                let index: Int
                if let existing = requests.firstIndex(where: { $0.endTime == nil && $0.requestID == session.requestID && $0.model == session.model }) {
                    index = existing
                } else {
                    requests.append(MonitoringRequest(requestID: session.requestID, process: sample.process,
                        model: session.model, client: session.client, phase: session.phase, startTime: now, lastTime: now))
                    index = requests.count - 1
                }
                requests[index].phase = session.phase
                requests[index].lastTime = now
                requests[index].context = session.context
                requests[index].cached = session.cached
                requests[index].generated = session.generated
            }
        }
        if samples.last?.time == sample.time { samples[samples.count - 1] = sample }
        else { samples.append(sample) }
        if now - lastCompaction >= 60 { compact(at: now) }
    }

    mutating func compact(at now: TimeInterval) {
        var previousBucket: Int?
        let original = samples.sorted { $0.time < $1.time }
        samples = original.enumerated().compactMap { i, sample in
            guard sample.time >= now - 86400, sample.time <= now + 1 else { return nil }
            let bucket = Int(floor(sample.time / 60))
            let edge = (i > 0 && (!sample.follows(original[i - 1]) || sample.model != original[i - 1].model || sample.activeModels != original[i - 1].activeModels))
                || (i + 1 < original.count && (!original[i + 1].follows(sample) || sample.model != original[i + 1].model || sample.activeModels != original[i + 1].activeModels))
            let keep = sample.time >= now - 3600 || bucket != previousBucket || edge || i == original.count - 1
            previousBucket = bucket
            var kept = sample
            kept.counters = sample.counters.filter { MonitoringSnapshot.counterKeys.contains($0.key) }
            return keep ? kept : nil
        }
        let recent = requests.filter { $0.lastTime >= now - 86400 && $0.startTime <= now }
        let closed = Set(recent.filter { $0.endTime != nil }.suffix(200).map(\.id))
        requests = recent.filter { $0.endTime == nil || closed.contains($0.id) }
        lastCompaction = now
    }

    func visibleRequests(from: TimeInterval, to: TimeInterval) -> [MonitoringRequest] {
        Array(requests.filter { $0.startTime <= to && ($0.endTime ?? $0.lastTime) >= from }.reversed().prefix(50))
    }

    private func intervals(from: TimeInterval, to: TimeInterval) -> [(MonitoringSample, MonitoringSample, Double)] {
        zip(samples, samples.dropFirst()).compactMap { a, b in
            guard b.follows(a), b.time > from, a.time < to else { return nil }
            return (a, b, max(0, min(b.time, to) - max(a.time, from)) / (b.time - a.time))
        }
    }

    func totals(from: TimeInterval, to: TimeInterval) -> [String: Double] {
        aggregate(from: from, to: to, byModel: false)["all"] ?? [:]
    }

    func modelTotals(from: TimeInterval, to: TimeInterval) -> [String: [String: Double]] {
        aggregate(from: from, to: to, byModel: true)
    }

    private func aggregate(from: TimeInterval, to: TimeInterval, byModel: Bool) -> [String: [String: Double]] {
        var result: [String: [String: Double]] = [:]
        var unknown: [String: Set<String>] = [:]
        for (a, b, fraction) in intervals(from: from, to: to) {
            let model: String?
            if a.model == b.model { model = a.model }
            else if a.activeModels == 0 { model = b.model }
            else if b.activeModels == 0 { model = a.model }
            else { model = nil }
            let key = byModel ? (model ?? "unattributed") : "all"
            for counter in MonitoringSnapshot.counterKeys {
                guard let value = b.counters[counter], let previous = a.counters[counter] else {
                    unknown[key, default: []].insert(counter); continue
                }
                result[key, default: [:]][counter, default: 0] += (value - previous) * fraction
            }
        }
        for (key, counters) in unknown {
            for counter in counters { result[key]?.removeValue(forKey: counter) }
        }
        return result
    }

    func series(keys: [String], from: TimeInterval, to: TimeInterval, count: Int = 60) -> [MonitoringPoint] {
        guard count > 0, to > from else { return [] }
        let width = (to - from) / Double(count)
        var sums = Array(repeating: 0.0, count: count), observed = sums
        var cut = Array(repeating: false, count: count)
        for (a, b) in zip(samples, samples.dropFirst()) where b.time > from && b.time <= to {
            let i = min(count - 1, max(0, Int(ceil((b.time - from) / width)) - 1))
            guard b.follows(a), keys.allSatisfy({ a.counters[$0] != nil && b.counters[$0] != nil }) else {
                cut[i] = true; continue
            }
            let duration = b.time - max(a.time, from)
            let difference = keys.reduce(0.0) { $0 + b.counters[$1]! - a.counters[$1]! }
            sums[i] += difference * duration / (b.time - a.time)
            observed[i] += duration
        }
        var segment = 0
        return (0..<count).map { i in
            if observed[i] == 0 || cut[i] { segment += 1 }
            return MonitoringPoint(time: from + Double(i + 1) * width,
                value: observed[i] > 0 && !cut[i] ? sums[i] / observed[i] : nil, segment: segment)
        }
    }
}

struct MonitoringRates {
    var decode: Double?
    var prefill: Double?
    var requests: Double?
}

struct MonitoringLiveHistory {
    private struct Frame {
        var time: TimeInterval
        var live: Double?
        var prefill: Double?
        var requests: Double?
    }
    private var frames: [Frame] = []
    private var previous: MonitoringSample?
    var decodePoints: [MonitoringPoint] = []
    var prefillPoints: [MonitoringPoint] = []

    mutating func accept(_ snapshot: MonitoringSnapshot, at now: TimeInterval) -> MonitoringRates {
        let sample = MonitoringSample(time: now, process: snapshot.gauges["process_start_time_seconds"], counters: snapshot.counters)
        if let previous, !sample.follows(previous) { self = Self() }
        previous = sample
        let live = snapshot.gauges["generation_tokens_live"] ?? snapshot.counters["generation_tokens_total"]
        let prefill = snapshot.gauges["prefill_tokens_live"]
        func rate(_ key: KeyPath<Frame, Double?>, value: Double?, window: Double) -> Double? {
            guard let value, let first = frames.last(where: { now - $0.time >= window }) ?? frames.first,
                  let before = first[keyPath: key], now > first.time, now - first.time <= 300, value >= before else { return nil }
            return (value - before) / (now - first.time)
        }
        let rates = MonitoringRates(decode: snapshot.gauges["requests_running"] == 0 ? 0 : rate(\.live, value: live, window: 4),
            prefill: prefill == 0 ? 0 : rate(\.prefill, value: prefill, window: 30),
            requests: rate(\.requests, value: snapshot.counters["requests_success_total"], window: 60))
        frames.append(Frame(time: now, live: live, prefill: prefill, requests: snapshot.counters["requests_success_total"]))
        frames.removeAll { now - $0.time > 120 }
        func append(_ points: inout [MonitoringPoint], _ value: Double?) {
            let segment = (points.last?.segment ?? 0) + (value == nil ? 1 : 0)
            points.append(MonitoringPoint(time: floor(now), value: value, segment: segment))
            points.removeAll { now - $0.time > 60 }
        }
        append(&decodePoints, rates.decode)
        append(&prefillPoints, rates.prefill)
        return rates
    }
}
