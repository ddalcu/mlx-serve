import Foundation

/// Everything the composer's context pill and its popover display.
///
/// Separate from `ContextMonitor`, which owns the bar's CLAMPED ratio: the
/// readout has to be able to say **100.3%**, because the moment the user goes
/// looking at it is the moment they went over. Clamping is right for a bar's
/// width and wrong for the number beside it.
struct ContextWindowStats: Equatable {

    /// How close to full, in bands. Drives the pill's tint; the thresholds match
    /// `ContextMonitor.barColor` so the pill and the bar never disagree.
    enum Pressure: Equatable {
        case comfortable, warm, tight, over
    }

    let promptTokens: Int
    let usedTokens: Int
    let contextLength: Int
    /// True when the reading came from a REJECTED request rather than a
    /// completed turn — the popover says so, since the numbers otherwise look
    /// like a turn that simply used a lot.
    let fromRejectedRequest: Bool

    /// Build from the chat's live counters, letting a context-overflow notice
    /// override them.
    ///
    /// The override matters: after a rejected request the last SUCCESSFUL turn's
    /// usage is stale and sits comfortably under the limit — exactly when the
    /// user opens the popover to find out why it failed. The rejected request's
    /// own counts are the only ones that explain anything, so they win when the
    /// server reported them.
    static func make(promptTokens: Int, completionTokens: Int, liveTokens: Int,
                     contextLength: Int, overflow: ChatErrorNotice?) -> ContextWindowStats {
        if let overflow, overflow.kind == .contextOverflow,
           let needed = overflow.neededTokens {
            return ContextWindowStats(promptTokens: needed, usedTokens: needed,
                                      contextLength: overflow.contextLength ?? contextLength,
                                      fromRejectedRequest: true)
        }
        return ContextWindowStats(
            promptTokens: promptTokens,
            usedTokens: ContextMonitor.usedTokens(promptTokens: promptTokens,
                                                  completionTokens: completionTokens,
                                                  liveTokens: liveTokens),
            contextLength: contextLength,
            // An overflow with no figures still means the last request was
            // rejected — we just can't put numbers to it.
            fromRejectedRequest: overflow?.kind == .contextOverflow)
    }

    /// Unclamped fill. Zero when no context length is known yet (before the
    /// first reply), so the pill claims nothing it can't know.
    var fraction: Double {
        guard contextLength > 0 else { return 0 }
        return Double(usedTokens) / Double(contextLength)
    }

    /// Clamped for drawing — past 1.0 a bar renders outside its own track.
    var barFraction: Double { min(1.0, fraction) }

    var isOverflowed: Bool { contextLength > 0 && usedTokens > contextLength }

    var remainingTokens: Int { max(0, contextLength - usedTokens) }

    var pressure: Pressure {
        guard contextLength > 0 else { return .comfortable }
        if isOverflowed { return .over }
        if fraction > 0.80 { return .tight }
        if fraction > 0.60 { return .warm }
        return .comfortable
    }

    /// One decimal always, so the figure reads the same at 0.3% and 100.3% and
    /// crossing 100 is visible rather than rounded away.
    var percentText: String { String(format: "%.1f%%", fraction * 100) }

    /// Gauge-scale token count: 1000-based with one decimal. 4096 and 4108 both
    /// read "4.1K" — deliberate, since the pill is an at-a-glance gauge and the
    /// exact figures are one click away in the popover's rows.
    static func compact(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1fM", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.1fK", Double(n) / 1_000) }
        return "\(n)"
    }

    /// Decode speed for the popover, or nil when there is nothing to report.
    ///
    /// The VALUE is the server's own `timings.predicted_per_second`, measured
    /// around its forward passes (`APIClient` prefers it over wall-clock). That
    /// distinction is the whole reason this is trustworthy: a client cannot time
    /// our own stream — with `tools` present the server buffers tokens for
    /// tool-call detection and flushes at the end, so every SSE delta lands at
    /// once and a wall-clock rate reads ~937 tok/s on a 2B.
    ///
    /// A non-positive rate is a MISSING measurement, not a slow model, and
    /// renders as nothing: "0 tok/s" next to a model that just answered reads as
    /// a fault in the model. Sub-1 rates keep a decimal for the same reason —
    /// they are real on a large model at long context.
    static func speedText(_ tokensPerSecond: Double?) -> String? {
        guard let r = tokensPerSecond, r > 0 else { return nil }
        if r < 1 { return String(format: "%.1f tok/s", r) }
        return "\(Int(r.rounded())) tok/s"
    }
}
