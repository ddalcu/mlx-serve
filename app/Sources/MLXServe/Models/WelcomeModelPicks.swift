import Foundation

/// One "best model of this type for your Mac" entry on the welcome screen: a
/// category label, the model, and one (very short) sentence on what it's good
/// at.
struct WelcomeModelPick: Equatable, Identifiable {
    let category: String
    let pick: RecommendedModelPick
    /// One sentence — what this kind of model is really good at.
    let strength: String
    var id: String { pick.id }
}

/// The welcome screen lists the best model of each type that actually fits this
/// Mac's memory. Pure — delegates fit to `SystemMemoryInfo`.
enum WelcomeModelPicks {
    private struct Category {
        let label: String
        let strength: String
        let catalog: [RecommendedModelPick]
    }

    /// Types the welcome screen offers, in order. Strengths are family-level
    /// (every Gemma is a good all-rounder; every Qwen is a coder), kept to one
    /// short sentence deliberately. Only Macs below 32 GB see these; above,
    /// the starter pick is the whole list.
    private static let codingCategory = Category(
        label: "Coding & agents",
        strength: "Strong at coding and multi-step agent work like using tools.",
        catalog: RecommendedModelPick.qwenCatalog)

    private static let categories: [Category] = [
        Category(label: "General",
                 strength: "Best all-rounder for everyday chat, writing, and quick questions.",
                 catalog: RecommendedModelPick.gemmaCatalog),
        codingCategory,
    ]

    /// The best model of each type for this Mac's usable memory, with its
    /// one-line strength. "Best" is the largest pick that fits COMFORTABLY (room
    /// to spare) — a tight fit is exactly what fails to load under real memory
    /// pressure — falling back to the largest that merely fits when nothing is
    /// comfortable. Categories where nothing fits at all are dropped, so the
    /// list only ever offers models that will actually run. From 32 GB up the
    /// list is the starter pick alone (Qwen 3.8 27B or Flash-Next), so the
    /// welcome and the sheet after it name the same model.
    static func forMemory(_ memory: SystemMemoryInfo) -> [WelcomeModelPick] {
        func biggest(_ picks: [RecommendedModelPick]) -> RecommendedModelPick? {
            picks.max { $0.approxRAMNeededGB < $1.approxRAMNeededGB }
        }
        let starter = RecommendedModelPick.starterPick(physicalMemoryBytes: memory.totalBytes)
        switch starter.family {
        case .gemma, .liquid: break
        case .qwen:
            return [WelcomeModelPick(category: codingCategory.label, pick: starter, strength: codingCategory.strength)]
        case .largest:
            return [WelcomeModelPick(category: "Frontier", pick: starter,
                                     strength: "Our smartest model: deep reasoning, coding and agent work.")]
        }
        return categories.compactMap { category in
            let comfortable = category.catalog.filter {
                memory.fit(neededGB: $0.approxRAMNeededGB) == .comfortable
            }
            let fits = category.catalog.filter {
                memory.fit(neededGB: $0.approxRAMNeededGB) != .exceeds
            }
            guard let best = biggest(comfortable) ?? biggest(fits) else { return nil }
            return WelcomeModelPick(category: category.label, pick: best, strength: category.strength)
        }
    }

    /// The row to mark recommended: the app-wide starter pick when it is
    /// listed, else the first row.
    static func recommendedId(in picks: [WelcomeModelPick], memory: SystemMemoryInfo) -> String? {
        let starter = RecommendedModelPick.starterPick(physicalMemoryBytes: memory.totalBytes)
        return picks.first { $0.pick.id == starter.id }?.id ?? picks.first?.id
    }
}
