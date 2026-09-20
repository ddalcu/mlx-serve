import Foundation

/// Which quality tier a set of live video settings corresponds to, if any:
/// the switcher reports what the values ARE, so "do these still mean Good?"
/// needs an answer.
///
/// The answer is compared against what a tier would leave behind RIGHT HERE,
/// not against what its profile declares: `applyQualityDefaults` ends in a
/// frame clamp, so on a canvas whose ladder caps below a tier's length,
/// picking that tier immediately produces values its own profile no longer
/// equals. Mirroring the clamp is what keeps a preset click from reading as
/// Custom the instant it lands — and it is why a resolution change leaves the
/// highlight alone, which is the behaviour people expect.
///
/// Stage-2 refine steps are deliberately absent: no tier sets them, so they
/// are the user's own value and moving them is not a departure from a tier.
enum VideoQualityMatch {

    /// The values applying a tier leaves behind on a given canvas. Mirrors
    /// `VideoGenView.applyQualityDefaults`, clamp included — the two are one
    /// contract and must not drift.
    struct Resolved: Equatable {
        var mode: VideoPipelineMode
        var steps: Int
        var cfgScale: Double
        var stgScale: Double
        var numFrames: Int
        /// A tier describes a FULL render — its step counts are the
        /// undistilled schedule's — so it always turns Turbo off. Turbo on is
        /// therefore a state no tier claims.
        var turbo: Bool = false
    }

    static func resolve(tier: QualityPreset, model: VideoModelPreset,
                        width: Int, height: Int, chainWindows: Int) -> Resolved {
        let s = model.settings(tier)
        return Resolved(
            mode: s.mode,
            // The tier's own steps clamped into the backend's range, exactly
            // as applying it would: a profile value the model's slider cannot
            // show would otherwise never be reachable and so never match.
            steps: min(model.stepsRange.upperBound, max(model.stepsRange.lowerBound, s.steps)),
            cfgScale: s.cfgScale,
            stgScale: s.stgScale,
            numFrames: clampToLadder(s.numFrames, model: model, width: width,
                                     height: height, chainWindows: chainWindows))
    }

    /// `clampFramesToRAM`'s rule: the ladder's two ends, nothing in between.
    /// The ladder is per-canvas and per-window-count because the whole clip
    /// rides back in one response.
    static func clampToLadder(_ frames: Int, model: VideoModelPreset,
                              width: Int, height: Int, chainWindows: Int) -> Int {
        let opts = model.frameOptions(width: width, height: height, chainWindows: chainWindows)
        guard let lo = opts.first, let hi = opts.last else { return frames }
        return min(hi, max(lo, frames))
    }

    /// The tier these values mean, or nil for Custom.
    ///
    /// `preferring` is the last tier the user explicitly picked, and it settles
    /// a real ambiguity rather than a cosmetic tie: LTX's Fast and Good differ
    /// in nothing but length, so at a canvas whose ladder caps below both they
    /// resolve to the SAME state and no property of the values can tell them
    /// apart. Honouring the user's own last click is the one answer that never
    /// moves the highlight under them. It never overrides a tier the values
    /// genuinely match.
    static func match(_ current: Resolved, model: VideoModelPreset,
                      width: Int, height: Int, chainWindows: Int,
                      preferring: QualityPreset?) -> QualityPreset? {
        let candidates = QualityPreset.allCases.filter {
            resolve(tier: $0, model: model, width: width, height: height,
                    chainWindows: chainWindows) == current
        }
        if let preferred = preferring, candidates.contains(preferred) { return preferred }
        return candidates.first
    }
}
