import Foundation

/// Per-Mac advice for the "Neural Engine prefill boost" toggle.
///
/// The server refuses `--ane-prefill` by name where it can't run (wrong
/// model family, under 96 GB RAM), so the toggle needs no hard gate — but a
/// switch that silently does nothing is the dead-control class, so the row
/// carries a caution naming why THIS Mac won't (or shouldn't) benefit. Pure:
/// the chip brand string and RAM come in as values, the sysctl probe lives
/// in `current()`.
enum AnePrefillAdvice {
    /// Below this the per-model fit check (server-side: resident weights +
    /// the int8 copy + shared I/O planes + the GPU rest slices + headroom
    /// vs total RAM) will decline everything but small models — a 27B bills
    /// ~11 GB extra under the channel split (2026-08-18; the retired row
    /// split billed ~32). The app doesn't know which model will load, so
    /// this is a soft advisory, not a gate; the server names its exact
    /// numbers when it declines.
    static let smallMacBytes: UInt64 = 32 << 30

    /// A caution to show under the toggle, or nil when this Mac is a good
    /// candidate. The RAM note outranks the M5 note: a Mac where most
    /// models won't fit should hear that, not a performance nuance.
    static func caution(chipBrand: String?, physicalMemoryBytes: UInt64) -> String? {
        if physicalMemoryBytes < smallMacBytes {
            return "On this Mac only small models will fit the extra Neural Engine copy — the server checks the exact fit per model at load and declines by name when it doesn't."
        }
        if isM5Family(chipBrand) {
            return "Not recommended on M5-family Macs yet: their GPU's neural accelerator (NAX) cores already speed up prompt processing, so this adds memory cost for little or no gain."
        }
        return nil
    }

    /// "Apple M5", "Apple M5 Pro/Max/Ultra" — a token-exact family check,
    /// never a substring scan (an "M45" must not match). An unreadable brand
    /// string is no information, not a warning.
    static func isM5Family(_ brand: String?) -> Bool {
        guard let brand else { return false }
        return brand.split(separator: " ").contains("M5")
    }

    /// The live advice for this machine, computed once — the chip and the
    /// RAM don't change at runtime, and a SwiftUI body re-evaluates often.
    static let liveCaution: String? = caution(
        chipBrand: chipBrandString(),
        physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory)

    private static func chipBrandString() -> String? {
        var size = 0
        guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0,
              size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0) == 0 else { return nil }
        return String(cString: buffer)
    }
}
