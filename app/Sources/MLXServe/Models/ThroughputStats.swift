import Foundation

/// Server-wide serving throughput, read from `/metrics.json` (requires the
/// server to run with `--metrics`). Totals come from counters; the "now" rates
/// are derived from the live gauges across two polls, because a counter only
/// moves when a request FINISHES — a long prefill or a still-streaming reply
/// would otherwise read as zero.
struct ThroughputSnapshot: Equatable {
    var generatedTokens: Int64 = 0
    var prefillTokens: Int64 = 0
    var decodeSeconds: Double = 0
    var prefillSeconds: Double = 0
    /// Completed tokens PLUS tokens generated so far by in-flight slots — nil
    /// on a server too old to publish the gauge. `generatedTokens` alone only
    /// moves when a request FINISHES, so a long reply sits frozen on it.
    var generatedLive: Int64?
    var prefillLive: Int64 = 0
    /// What the prefill in flight will forward in all (after the prefix cache), same scale as `prefillLive`.
    var prefillExpected: Int64 = 0
    var prefilling: Bool = false
    /// A request is being served or waiting its turn.
    var busy: Bool = false
    var takenAt: TimeInterval = 0
    /// Unix seconds the server process started: a different one means its counters began again from zero.
    var processStart: Int64 = 0
    /// Every model's share of the success counters since the process started, keyed by model id.
    var usage: [String: UsageCounters] = [:]

    /// How much of the prefill in flight is done, 0...1. nil when none runs or the server did not say how long it is.
    var prefillProgress: Double? {
        guard prefilling, prefillExpected > 0 else { return nil }
        return min(1, Double(prefillLive) / Double(prefillExpected))
    }

    /// What "tokens generated" should read: the live figure when the server
    /// publishes it (at rest it equals the counter).
    var displayedTokens: Int64 { generatedLive ?? generatedTokens }

    var avgDecodeTPS: Double? {
        guard decodeSeconds > 0, generatedTokens > 0 else { return nil }
        return Double(generatedTokens) / decodeSeconds
    }

    var avgPrefillTPS: Double? {
        guard prefillSeconds > 0, prefillTokens > 0 else { return nil }
        return Double(prefillTokens) / prefillSeconds
    }

    /// Tokens/s between two polls, nil when the interval is unusable or nothing
    /// moved. A gauge that RESET (server restart, counter rollover) reads as no
    /// rate rather than a negative one.
    static func rate(_ previous: Int64, _ current: Int64, seconds: TimeInterval) -> Double? {
        guard seconds > 0.2, current > previous else { return nil }
        return Double(current - previous) / seconds
    }

    func decodeTPS(since previous: ThroughputSnapshot) -> Double? {
        Self.rate(previous.displayedTokens, displayedTokens, seconds: takenAt - previous.takenAt)
    }

    func prefillTPS(since previous: ThroughputSnapshot) -> Double? {
        Self.rate(previous.prefillLive, prefillLive, seconds: takenAt - previous.takenAt)
    }

    static func parse(_ json: [String: Any], at now: TimeInterval) -> ThroughputSnapshot {
        let counters = json["counters"] as? [String: Any] ?? [:]
        let gauges = json["gauges"] as? [String: Any] ?? [:]
        let hists = json["histograms"] as? [String: Any] ?? [:]
        func int(_ d: [String: Any], _ k: String) -> Int64 {
            (d[k] as? NSNumber)?.int64Value ?? 0
        }
        func sum(_ name: String) -> Double {
            ((hists[name] as? [String: Any])?["sum"] as? NSNumber)?.doubleValue ?? 0
        }
        return ThroughputSnapshot(
            generatedTokens: int(counters, "generation_tokens_total"),
            prefillTokens: int(counters, "prefill_tokens_total"),
            decodeSeconds: sum("decode_time_seconds"),
            prefillSeconds: sum("prefill_time_seconds"),
            generatedLive: (gauges["generation_tokens_live"] as? NSNumber)?.int64Value,
            prefillLive: int(gauges, "prefill_tokens_live"),
            prefillExpected: int(gauges, "prefill_tokens_expected"),
            prefilling: int(gauges, "requests_prefilling") > 0,
            busy: int(gauges, "requests_running") + int(gauges, "requests_waiting") + int(gauges, "requests_prefilling") > 0,
            takenAt: now,
            processStart: int(gauges, "process_start_time_seconds"),
            usage: usage(json["usage"] as? [[String: Any]] ?? [])
        )
    }

    private static func usage(_ rows: [[String: Any]]) -> [String: UsageCounters] {
        func int(_ r: [String: Any], _ k: String) -> Int64 { (r[k] as? NSNumber)?.int64Value ?? 0 }
        var out: [String: UsageCounters] = [:]
        for r in rows {
            guard let model = r["model"] as? String else { continue }
            out[model] = UsageCounters(requests: int(r, "requests"), promptTokens: int(r, "prompt_tokens"),
                                       cachedTokens: int(r, "cached_tokens"), generatedTokens: int(r, "generation_tokens"),
                                       prefillNs: int(r, "prefill_ns"), decodeNs: int(r, "decode_ns"))
        }
        return out
    }

    static func formatTPS(_ tps: Double?) -> String {
        guard let tps, tps > 0 else { return "—" }
        return tps >= 100 ? String(format: "%.0f", tps) : String(format: "%.1f", tps)
    }

    static func formatTokens(_ n: Int64) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }
}
