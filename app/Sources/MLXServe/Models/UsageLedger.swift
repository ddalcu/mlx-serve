import Foundation

/// What a model has done, as the server counts it: requests, tokens in and out, and the time spent on each side.
struct UsageCounters: Equatable {
    var requests: Int64 = 0
    var promptTokens: Int64 = 0
    /// The part of `promptTokens` the prefix cache restored instead of computing.
    var cachedTokens: Int64 = 0
    var generatedTokens: Int64 = 0
    var prefillNs: Int64 = 0
    var decodeNs: Int64 = 0

    var totalTokens: Int64 { promptTokens + generatedTokens }
    var isEmpty: Bool { self == UsageCounters() }

    /// Generated tokens per second of decode, nil with nothing to divide.
    var decodeTPS: Double? {
        guard decodeNs > 0, generatedTokens > 0 else { return nil }
        return Double(generatedTokens) / (Double(decodeNs) / 1e9)
    }

    /// Prompt tokens computed per second of prefill. The cached ones cost no prefill, so they are not in it.
    var prefillTPS: Double? {
        let computed = promptTokens - cachedTokens
        guard prefillNs > 0, computed > 0 else { return nil }
        return Double(computed) / (Double(prefillNs) / 1e9)
    }

    static func + (l: Self, r: Self) -> Self {
        Self(requests: l.requests + r.requests, promptTokens: l.promptTokens + r.promptTokens,
             cachedTokens: l.cachedTokens + r.cachedTokens, generatedTokens: l.generatedTokens + r.generatedTokens,
             prefillNs: l.prefillNs + r.prefillNs, decodeNs: l.decodeNs + r.decodeNs)
    }

    static func - (l: Self, r: Self) -> Self {
        Self(requests: l.requests - r.requests, promptTokens: l.promptTokens - r.promptTokens,
             cachedTokens: l.cachedTokens - r.cachedTokens, generatedTokens: l.generatedTokens - r.generatedTokens,
             prefillNs: l.prefillNs - r.prefillNs, decodeNs: l.decodeNs - r.decodeNs)
    }

    /// A counter that fell: the server's counters began again, so `base` no longer describes them.
    func fell(below base: Self) -> Bool {
        requests < base.requests || promptTokens < base.promptTokens || cachedTokens < base.cachedTokens
            || generatedTokens < base.generatedTokens || prefillNs < base.prefillNs || decodeNs < base.decodeNs
    }
}

/// What one model did between two looks at the server, on the day the second look fell.
struct UsageDelta: Equatable {
    var day: String
    var model: String
    var counters: UsageCounters
}

/// The local calendar day a count belongs to, as the `yyyy-MM-dd` the dashboard and the store key it by.
enum UsageDay {
    static func key(_ date: Date, calendar: Calendar = .current) -> String {
        let c = calendar.dateComponents([.year, .month, .day], from: date)
        return String(format: "%04d-%02d-%02d", c.year ?? 0, c.month ?? 0, c.day ?? 0)
    }

    /// The start of the day a key names; nil for anything that is not one.
    static func date(_ key: String, calendar: Calendar = .current) -> Date? {
        let parts = key.split(separator: "-").compactMap { Int($0) }
        guard parts.count == 3 else { return nil }
        return calendar.date(from: DateComponents(year: parts[0], month: parts[1], day: parts[2]))
    }

    /// "Oct 30, 2026" (or "Oct 30"), in the reader's own order and language.
    static func text(_ key: String, year: Bool = true, calendar: Calendar = .current) -> String {
        guard let date = date(key, calendar: calendar) else { return key }
        return year ? date.formatted(.dateTime.month(.abbreviated).day().year()) : date.formatted(.dateTime.month(.abbreviated).day())
    }

    /// The `count` day keys ending at `today`, oldest first.
    static func lastDays(_ count: Int, endingAt today: String, calendar: Calendar = .current) -> [String] {
        guard let end = date(today, calendar: calendar) else { return [] }
        return (0..<count).reversed().compactMap { back in
            calendar.date(byAdding: .day, value: -back, to: end).map { key($0, calendar: calendar) }
        }
    }
}

enum UsageFormat {
    /// 999, 1.0K, 12K, 123K, 1.2M, 2.5B: short enough for a narrow strip. Truncated, so a count is never overstated.
    static func compact(_ n: Int64) -> String {
        let value = Double(n)
        for (scale, suffix) in [(1e9, "B"), (1e6, "M"), (1e3, "K")] where value >= scale {
            let x = (value / scale * 10).rounded(.down) / 10
            return (x < 10 ? String(format: "%.1f", x) : "\(Int(x))") + suffix
        }
        return "\(n)"
    }
}

/// Turns the server's cumulative per-model counters into additions. The server only ever counts up, and only
/// within one process, so what a look adds is its counters minus the last look's, and a new process starts from zero.
struct UsageLedger {
    /// The counters at the last look. nil until the first one, which only sets the baseline: what a server did
    /// before this app looked is nobody's to date.
    private var seen: [String: UsageCounters]?
    private var processStart: Int64 = 0

    mutating func ingest(_ snapshot: ThroughputSnapshot, day: String) -> [UsageDelta] {
        defer {
            seen = snapshot.usage
            processStart = snapshot.processStart
        }
        guard let before = seen else { return [] }
        let restarted = snapshot.processStart != processStart
        return snapshot.usage.compactMap { model, now in
            let base = restarted ? UsageCounters() : (before[model] ?? UsageCounters())
            let added = now.fell(below: base) ? now : now - base
            return added.isEmpty ? nil : UsageDelta(day: day, model: model, counters: added)
        }
        .sorted { $0.model < $1.model }
    }
}
