import Foundation

/// The Image pane's Quality switcher reads the DERIVED tier: a tier owns one
/// step count, so the steps say which tier is set, or that none is (Custom).
enum ImageQualityMatch {
    /// The tier `steps` means, or nil for Custom. `preferring` is the last
    /// tier the user picked and settles a tie between tiers that share a step
    /// count; it never overrides a tier the steps do not match.
    static func match(steps: Int, model: ImageModelPreset, preferring: QualityPreset?) -> QualityPreset? {
        let candidates = QualityPreset.allCases.filter { model.settings($0).steps == steps }
        if let preferred = preferring, candidates.contains(preferred) { return preferred }
        return candidates.first
    }
}
