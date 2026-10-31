import SwiftUI
import AppKit

/// The dashboard's colours: one green ramp, stepped separately for light and dark rather than flipped, and checked as an
/// ordinal ramp (lightness monotone, steps at least 0.06 apart, the faintest at least 2:1 against its surface).
/// An idle day is a neutral, never a green.
enum UsagePalette {
    private static func dynamic(light: UInt32, dark: UInt32) -> Color {
        func ns(_ hex: UInt32) -> NSColor {
            NSColor(srgbRed: CGFloat(hex >> 16 & 255) / 255, green: CGFloat(hex >> 8 & 255) / 255,
                    blue: CGFloat(hex & 255) / 255, alpha: 1)
        }
        return Color(nsColor: NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? ns(dark) : ns(light)
        })
    }

    /// Levels 1...4, the more used the stronger.
    static let ramp: [Color] = [
        dynamic(light: 0x60C473, dark: 0x016123),
        dynamic(light: 0x3AA151, dark: 0x128537),
        dynamic(light: 0x017F31, dark: 0x48AE5D),
        dynamic(light: 0x015B21, dark: 0x73D885),
    ]

    static let idle = Color.primary.opacity(0.08)

    /// The single series of a bar chart.
    static let series = ramp[2]

    static func color(level: Int) -> Color { level <= 0 ? idle : ramp[min(level, ramp.count) - 1] }
}
