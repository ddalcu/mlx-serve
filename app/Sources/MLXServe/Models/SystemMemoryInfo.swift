import Foundation
#if canImport(Metal)
import Metal
#endif

/// This Mac's memory, framed the way a model picker needs it: how much RAM the
/// machine HAS, and how much of it can actually be USED for a model.
struct SystemMemoryInfo: Equatable {
    let totalBytes: UInt64
    let usableBytes: UInt64

    private static let bytesPerGiB = 1_073_741_824.0

    var totalGB: Double { Double(totalBytes) / Self.bytesPerGiB }
    var usableGB: Double { Double(usableBytes) / Self.bytesPerGiB }

    /// Whole-GB labels — Macs report memory in exact powers of two, so rounding
    /// is exact in practice and avoids a distracting "23.8 GB".
    var totalLabel: String { Self.wholeGB(totalGB) }
    var usableLabel: String { Self.wholeGB(usableGB) }

    /// Usable as a fraction of total, for the capacity bar (0…1).
    var usableFraction: Double {
        totalBytes > 0 ? min(1, max(0, Double(usableBytes) / Double(totalBytes))) : 0
    }

    static func wholeGB(_ gb: Double) -> String { "\(Int(gb.rounded())) GB" }

    /// One-decimal GB for a model's requirement ("7.6 GB") — the numbers this
    /// compares are close enough that whole-GB rounding would hide differences.
    static func preciseGB(_ gb: Double) -> String { String(format: "%.1f GB", gb) }

    /// How a model needing `neededGB` fits this Mac's USABLE budget. The 0.85
    /// band flags models that technically fit but leave almost no room for
    /// context or anything else running.
    func fit(neededGB: Double) -> MemoryFit {
        guard usableGB > 0 else { return .comfortable }
        if neededGB > usableGB { return .exceeds }
        if neededGB > usableGB * 0.85 { return .tight }
        return .comfortable
    }

    /// Live machine values. Total = physical RAM; usable = Metal's
    /// recommendedMaxWorkingSetSize, falling back to ~75% of RAM when Metal is
    /// unavailable (headless/CI).
    static func current() -> SystemMemoryInfo {
        let total = ProcessInfo.processInfo.physicalMemory
        var usable = total / 4 * 3
        #if canImport(Metal)
        if let device = MTLCreateSystemDefaultDevice() {
            let workingSet = device.recommendedMaxWorkingSetSize
            if workingSet > 0 { usable = workingSet }
        }
        #endif
        return SystemMemoryInfo(totalBytes: total, usableBytes: min(usable, total))
    }
}

/// A model's memory requirement measured against this Mac's usable budget.
/// Pure — the view maps each case to an icon/color/tint.
enum MemoryFit: Equatable {
    /// Fits with room to spare.
    case comfortable
    /// Fits, but leaves almost no headroom.
    case tight
    /// Needs more memory than this Mac can give a model.
    case exceeds

    var label: String {
        switch self {
        case .comfortable: return "Comfortable"
        case .tight:       return "Tight fit"
        case .exceeds:     return "Not enough memory"
        }
    }

    var fitsAtAll: Bool { self != .exceeds }
}
