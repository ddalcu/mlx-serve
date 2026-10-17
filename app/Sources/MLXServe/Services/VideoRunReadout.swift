import Foundation

/// What the Video pane says beside a run's bar: how far along it is, which step,
/// and how fast the steps go.
///
/// The server reports `step`/`total` over the DENOISER only (over the whole chain
/// on a chained run), and the VAE decode that follows the last step reports
/// `step == total` for as long as it takes. So the denoiser fills only
/// `samplingShare` of the bar where a time model prices the decode (H3), and
/// the bar stops at `maxRunningFraction` until the clip is written.
struct VideoRunReadout: Equatable {
    /// 0...1 over every shot of the run.
    var fraction: Double
    var stepLine: String
    var pace: String?

    var percent: Int { Int((fraction * 100).rounded(.down)) }

    static let maxRunningFraction = 0.99

    /// H3's share from its own time model: the sampling steps against the stages
    /// around them. Without a model (LTX) the steps are the whole bar.
    static func samplingShare(perStep: Double, steps: Int, tail: Double) -> Double {
        let sampling = perStep * Double(steps)
        guard sampling > 0, tail >= 0 else { return 1 }
        return sampling / (sampling + tail)
    }

    static func make(step: Int, total: Int, window: Int? = nil, windows: Int? = nil,
                     shot: Int = 0, shots: Int = 1, samplingShare: Double,
                     secondsPerStep: Double?) -> VideoRunReadout {
        let done = total > 0 ? Double(min(max(step, 0), total)) / Double(total) : 0
        let shotCount = max(shots, 1)
        let fraction = min((Double(min(max(shot, 0), shotCount - 1)) + done * samplingShare) / Double(shotCount),
                           maxRunningFraction)

        let stepLine: String
        if let windows, windows > 1, let window, total % windows == 0 {
            let perWindow = total / windows
            let local = min(max(step - (window - 1) * perWindow, 0), perWindow)
            stepLine = L10n.formatUngrouped("Window %lld of %lld · step %lld of %lld",
                                            Int64(window), Int64(windows), Int64(local), Int64(perWindow))
        } else {
            stepLine = L10n.formatUngrouped("Step %lld of %lld", Int64(min(max(step, 0), total)), Int64(total))
        }

        return VideoRunReadout(fraction: fraction, stepLine: stepLine, pace: secondsPerStep.map(Self.pace))
    }

    /// Under H3's fast recipe most laps reuse a cached velocity, so this is the
    /// AMORTIZED cost of a step, not the cost of a full forward.
    static func pace(_ seconds: Double) -> String {
        seconds < 60
            ? L10n.format("%.1f s per step", seconds)
            : L10n.format("%.1f min per step", seconds / 60)
    }
}
