import Foundation

/// One model's load as `GET /v1/load-progress` reports it (`src/host/registry/load_progress.zig`): a GPU load
/// reads weights (bytes), a Neural Engine load builds programs (compiled, or restored from its
/// compile cache), and both end warming up.
struct LoadProgress: Equatable {
    enum Phase: String {
        case starting
        case readingWeights = "reading_weights"
        case anePrograms = "ane_programs"
        case warmingUp = "warming_up"
    }

    var phase: Phase
    var done: Int64
    var total: Int64
    /// A Neural Engine program missed the compile cache: a first load on this Mac.
    var compiling: Bool
    /// Seconds left at the rate so far; nil until there is a rate.
    var eta: Int?

    /// 0...1 for a counted phase; nil draws an indeterminate bar.
    var fraction: Double? {
        guard total > 0, phase == .readingWeights || phase == .anePrograms else { return nil }
        return min(1, Double(done) / Double(total))
    }

    var label: String {
        var parts: [String]
        switch phase {
        case .starting:
            parts = [L10n.text("Starting")]
        case .warmingUp:
            parts = [L10n.text("Warming up")]
        case .readingWeights:
            parts = [L10n.format("Reading weights · %@ of %@ GB", Self.gb(done), Self.gb(total))]
        case .anePrograms:
            parts = [compiling
                ? L10n.formatUngrouped("Compiling for the Neural Engine · %d of %d", Int(done), Int(total))
                : L10n.formatUngrouped("Loading from the Neural Engine cache · %d of %d", Int(done), Int(total))]
        }
        if let eta {
            parts.append(eta < 60 ? L10n.formatUngrouped("about %d s left", eta)
                                  : L10n.formatUngrouped("about %d min left", (eta + 59) / 60))
        }
        return parts.joined(separator: " · ")
    }

    private static func gb(_ bytes: Int64) -> String {
        String(format: "%.1f", Double(bytes) / 1e9)
    }

    /// The load of the model at `path`; the only load running when the server spells the path
    /// differently (a symlink resolved, a trailing slash).
    static func lookup(_ all: [String: LoadProgress], path: String) -> LoadProgress? {
        all[path] ?? (all.count == 1 ? all.values.first : nil)
    }

    /// A load long enough that someone looked away, finished while the app is in the background.
    static func notifiesReady(after seconds: TimeInterval, appActive: Bool) -> Bool {
        !appActive && seconds >= 15
    }

    /// Every loading model's progress by its path; empty on anything unreadable.
    static func parse(_ data: Data) -> [String: LoadProgress] {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let loading = root["loading"] as? [[String: Any]] else { return [:] }
        var out: [String: LoadProgress] = [:]
        for row in loading {
            guard let path = row["path"] as? String,
                  let p = row["progress"] as? [String: Any],
                  let phase = Phase(rawValue: p["phase"] as? String ?? "") else { continue }
            out[path] = LoadProgress(
                phase: phase,
                done: (p["done"] as? NSNumber)?.int64Value ?? 0,
                total: (p["total"] as? NSNumber)?.int64Value ?? 0,
                compiling: p["compiling"] as? Bool ?? false,
                eta: (p["eta_s"] as? NSNumber)?.intValue)
        }
        return out
    }
}
