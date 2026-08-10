import Foundation

/// The welcome screen's "here's what fits your Mac" summary: the machine's
/// physical RAM (as a label) plus the ONE model recommended for it.
struct WelcomeRecommendation: Equatable {
    /// e.g. "24 GB" — the machine's physical memory, rounded to whole GB.
    let memoryText: String
    /// The recommended model for this Mac's RAM tier.
    let pick: RecommendedModelPick
    /// One plain-English line on why this pick fits, naming the RAM.
    let rationale: String

    private static let bytesPerGiB: Double = 1_073_741_824

    static func forPhysicalMemory(bytes: UInt64) -> WelcomeRecommendation {
        let gib = Double(bytes) / bytesPerGiB
        let text = formatMemory(gib)
        return WelcomeRecommendation(
            memoryText: text,
            pick: RecommendedModelPick.starterPick(physicalMemoryBytes: bytes),
            rationale: rationale(memoryText: text)
        )
    }

    /// Whole-GB label. Macs report physical memory in exact powers of two
    /// (24 GB = 24·2³⁰ bytes), so rounding to the nearest GB is exact in
    /// practice and never shows a distracting "23.8 GB".
    static func formatMemory(_ gib: Double) -> String {
        "\(Int(gib.rounded())) GB"
    }

    /// Each RAM tier is deliberately the largest pick that still leaves the
    /// machine headroom (see `starterPick`), so "with memory to spare" is
    /// honest for every tier.
    static func rationale(memoryText: String) -> String {
        "Chosen to run smoothly on your \(memoryText) Mac, with memory to spare for everything else."
    }
}
