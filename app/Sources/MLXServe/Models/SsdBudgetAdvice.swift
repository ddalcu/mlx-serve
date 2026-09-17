import Foundation

/// The SSD budget the settings sheet offers for a streamed checkpoint, in GiB of
/// TOTAL resident weights, experts included (the `--ssd-budget-gb` unit). 0.47 x RAM
/// is 60 on a 128 GiB Mac; 16 GiB is the floor below which the trunk and the union
/// workspace leave no expert cache.
enum SsdBudgetAdvice {
    static let minimumGiB = 16

    static func recommendedGiB(physicalMemoryBytes: UInt64) -> Int {
        let gib = Int(physicalMemoryBytes / 1_073_741_824)
        return max(minimumGiB, Int(0.47 * Double(gib)))
    }

    static func presets(physicalMemoryBytes: UInt64) -> [Int] {
        let gib = Int(physicalMemoryBytes / 1_073_741_824)
        let rec = recommendedGiB(physicalMemoryBytes: physicalMemoryBytes)
        let ladder = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
            .filter { $0 >= minimumGiB && $0 <= max(gib, minimumGiB) }
        return Array(Set(ladder + [rec])).sorted()
    }

    static func label(_ giB: Int, recommended: Int) -> String {
        giB == recommended ? "\(giB) GiB (recommended)" : "\(giB) GiB"
    }

    static let liveRecommendedGiB = recommendedGiB(
        physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory)
    static let livePresets = presets(
        physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory)
}
