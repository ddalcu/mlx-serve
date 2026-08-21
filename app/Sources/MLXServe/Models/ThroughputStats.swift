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
    var prefilling: Bool = false
    var takenAt: TimeInterval = 0

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
            prefilling: int(gauges, "requests_prefilling") > 0,
            takenAt: now
        )
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
